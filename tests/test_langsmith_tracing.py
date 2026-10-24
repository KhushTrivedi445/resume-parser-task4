"""
Unit tests for LangSmith tracing, serialization, and RAG evaluation metrics.
All tests use mock fixtures where appropriate to avoid external paid API calls.
"""

from unittest.mock import MagicMock
import pytest
from langchain_core.documents import Document
from langsmith.schemas import Example, Run

from app.schemas.matching import EvidenceItem, MatchingResponse
from app.services.rag.retriever import (
    build_vector_store,
    chunk_text,
    retrieve_passages_with_scores,
)
from app.services.rag.matching_service import collect_and_classify_evidence
from evaluation.evaluators import (
    eval_answer_correctness,
    eval_context_precision,
    eval_context_recall,
    eval_faithfulness,
    eval_operational_metrics,
    eval_retrieval_relevance,
)


def test_evidence_item_and_response_serialization():
    """Verify EvidenceItem and MatchingResponse serialize cleanly for LangSmith."""
    item = EvidenceItem(
        requirement="Python",
        supporting_passage="Built APIs with Python and FastAPI.",
        match_classification="explicit_match",
        relevance_score=0.88,
        source="resume",
    )
    dumped = item.model_dump()
    assert dumped["requirement"] == "Python"
    assert dumped["relevance_score"] == 0.88

    response = MatchingResponse(
        match_score=85.0,
        score_explanation="Detailed transparent explanation.",
        matched_skills=["Python", "FastAPI"],
        missing_skills=["Kubernetes"],
        evidence=[item],
    )
    res_dump = response.model_dump()
    assert res_dump["match_score"] == 85.0
    assert len(res_dump["evidence"]) == 1
    assert res_dump["evidence"][0]["requirement"] == "Python"


def test_retriever_metadata_preservation():
    """Verify chunk_text and FAISS retrieval preserve chunk metadata and score types."""
    sample_text = (
        "Candidate with 4 years Python development.\n\n"
        "Expertise in FastAPI, Docker, and PostgreSQL databases."
    )
    store = build_vector_store(sample_text, source="candidate_resume")
    passages = retrieve_passages_with_scores(
        store, query="Experience with Python", k=1, requirement="Python"
    )

    assert len(passages) == 1
    doc, score = passages[0]
    assert isinstance(doc, Document)
    assert isinstance(score, float)
    assert doc.metadata.get("source") == "candidate_resume"
    assert "chunk_id" in doc.metadata


def test_collect_and_classify_evidence_unsupported_requirement():
    """Verify unevidenced requirement (e.g. FAISS) is properly classified as unevidenced."""
    resume_text = (
        "Web Developer experienced in HTML, CSS, JavaScript, and React.\n\n"
        "Built responsive frontend layouts."
    )
    store = build_vector_store(resume_text, source="resume")
    evidence = collect_and_classify_evidence(
        vector_store=store,
        all_jd_requirements=["FAISS"],
        resume_skills_set={"React", "JavaScript"},
        cleaned_jd="Looking for FAISS vector search engineer.",
    )

    assert len(evidence) == 1
    ev = evidence[0]
    assert ev.requirement == "FAISS"
    assert ev.match_classification == "unevidenced"
    assert ev.relevance_score is not None
    assert 0.0 <= ev.relevance_score <= 1.0


def test_eval_retrieval_relevance_metric():
    """Verify retrieval relevance metric rewards valid evidence and identifies unevidenced items."""
    mock_run = MagicMock(spec=Run)
    mock_run.outputs = {
        "evidence": [
            {
                "requirement": "Python",
                "supporting_passage": "5 years Python development.",
                "match_classification": "explicit_match",
                "relevance_score": 0.85,
            },
            {
                "requirement": "FAISS",
                "supporting_passage": "President of badminton club.",
                "match_classification": "unevidenced",
                "relevance_score": 0.40,
            },
        ]
    }
    mock_example = MagicMock(spec=Example)
    mock_example.outputs = {
        "expected_unsupported_requirements": ["FAISS"],
        "unrelated_drift_terms": ["badminton"],
    }

    result = eval_retrieval_relevance(mock_run, mock_example)
    assert result.key == "retrieval_relevance"
    assert result.score == 1.0


def test_eval_context_precision_and_recall():
    """Verify precision and recall metric calculations on mock outputs."""
    mock_run = MagicMock(spec=Run)
    mock_run.outputs = {
        "evidence": [
            {"requirement": "Python", "match_classification": "explicit_match", "relevance_score": 0.9},
            {"requirement": "FastAPI", "match_classification": "explicit_match", "relevance_score": 0.85},
            {"requirement": "FAISS", "match_classification": "unevidenced", "relevance_score": 0.35},
        ]
    }
    mock_example = MagicMock(spec=Example)
    mock_example.outputs = {
        "expected_relevant_evidence": {
            "Python": ["Python"],
            "FastAPI": ["FastAPI"],
        }
    }

    precision_res = eval_context_precision(mock_run, mock_example)
    assert precision_res.key == "context_precision"
    # 2 explicit matches out of 3 = 0.667
    assert precision_res.score == 0.667

    recall_res = eval_context_recall(mock_run, mock_example)
    assert recall_res.key == "context_recall"
    # 2 expected matched out of 2 = 1.0
    assert recall_res.score == 1.0


def test_eval_faithfulness_detects_hallucination():
    """Verify faithfulness evaluator detects and penalizes hallucinated claims."""
    # Run claiming strength in a missing skill
    mock_run_hallucinating = MagicMock(spec=Run)
    mock_run_hallucinating.outputs = {
        "strengths": ["Demonstrated strong competency in FAISS vector databases."],
        "relevant_experience": [],
        "recommendations": [],
    }
    mock_example = MagicMock(spec=Example)
    mock_example.outputs = {
        "reference_missing_skills": ["FAISS", "Kubernetes"],
    }

    result_hallucinating = eval_faithfulness(mock_run_hallucinating, mock_example)
    assert result_hallucinating.score < 1.0
    assert "hallucination detected" in result_hallucinating.comment.lower()

    # Faithful run
    mock_run_faithful = MagicMock(spec=Run)
    mock_run_faithful.outputs = {
        "strengths": ["Strong proficiency in Python and FastAPI backend engineering."],
        "relevant_experience": [],
        "recommendations": ["Gain experience with FAISS vector indexing."],
    }
    result_faithful = eval_faithfulness(mock_run_faithful, mock_example)
    assert result_faithful.score == 1.0


def test_eval_answer_correctness_and_operational():
    """Verify answer correctness and operational health metrics."""
    mock_run = MagicMock(spec=Run)
    mock_run.outputs = {
        "match_score": 80.0,
        "matched_skills": ["Python", "FastAPI"],
        "missing_skills": ["Docker"],
        "evidence": [{"requirement": "Python"}],
    }
    mock_run.error = None
    mock_run.start_time = None
    mock_run.end_time = None

    mock_example = MagicMock(spec=Example)
    mock_example.outputs = {
        "reference_match_score": 80.0,
        "reference_matched_skills": ["Python", "FastAPI"],
        "reference_missing_skills": ["Docker"],
    }

    correctness = eval_answer_correctness(mock_run, mock_example)
    assert correctness.score == 1.0

    operational = eval_operational_metrics(mock_run, mock_example)
    assert operational.score == 1.0
    assert operational.metadata["has_error"] is False
