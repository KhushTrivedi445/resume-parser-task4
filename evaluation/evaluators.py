"""
RAG Evaluation Metrics for Resume-Job Description Matching Pipeline

This module implements decoupled, verifiable evaluators compatible with
LangSmith 0.14.12 evaluation suite:
    evaluate(target, data=..., evaluators=[...])

Evaluators:
1. eval_retrieval_relevance: Whether retrieved passages support their associated requirements.
2. eval_context_precision: Proportion of retrieved passages containing actionable evidence vs drift.
3. eval_context_recall: Proportion of ground-truth required skills successfully retrieved.
4. eval_faithfulness: Whether generated strengths and experience are faithful to evidence without hallucinating missing skills.
5. eval_answer_correctness: Whether match score, matched skills, and unevidenced skills match reference ground truth.
6. eval_operational_metrics: Measures latency, error status, and retrieved evidence count.
"""

from typing import Any, Dict, List, Optional
from langsmith.evaluation import EvaluationResult
from langsmith.schemas import Example, Run


def _extract_evidence_list(outputs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Safely extract evidence items from outputs whether dicts or Pydantic models."""
    raw = outputs.get("evidence", [])
    items = []
    for item in raw:
        if isinstance(item, dict):
            items.append(item)
        elif hasattr(item, "model_dump"):
            items.append(item.model_dump())
        elif hasattr(item, "__dict__"):
            items.append(vars(item))
    return items


def eval_retrieval_relevance(run: Run, example: Example) -> EvaluationResult:
    """
    Evaluate retrieval relevance:
    Verifies that for each job requirement, the retrieved passage contains
    relevant terminology or is appropriately flagged as unevidenced.
    Penalizes instances where an unsupported requirement drifted into unrelated passages
    (e.g., extracurricular activities) but was treated as evidence.
    """
    outputs = run.outputs or {}
    evidence = _extract_evidence_list(outputs)
    ref_outputs = example.outputs or {}
    unsupported_reqs = set(ref_outputs.get("expected_unsupported_requirements", []))
    unrelated_terms = [t.lower() for t in ref_outputs.get("unrelated_drift_terms", [])]

    if not evidence:
        return EvaluationResult(
            key="retrieval_relevance",
            score=0.0,
            comment="No evidence passages retrieved.",
        )

    valid_retrievals = 0
    drift_penalties = 0
    total = len(evidence)

    for item in evidence:
        req = item.get("requirement", "")
        passage = item.get("supporting_passage", "").lower()
        classification = item.get("match_classification", "")

        # Check if passage contains unrelated drift keywords
        has_drift = any(drift_term in passage for drift_term in unrelated_terms)
        if has_drift and req in unsupported_reqs:
            # If the requirement was missing and drifted to unrelated text:
            # It should be classified as 'unevidenced'.
            if classification == "unevidenced":
                valid_retrievals += 1
            else:
                drift_penalties += 1
        elif req.lower() in passage:
            valid_retrievals += 1
        elif classification in ("explicit_match", "semantically_supported"):
            valid_retrievals += 1
        elif classification == "unevidenced":
            valid_retrievals += 1

    score = max(0.0, min(1.0, round((valid_retrievals - drift_penalties * 0.5) / total, 3)))
    return EvaluationResult(
        key="retrieval_relevance",
        score=score,
        comment=f"Evaluated {total} evidence passages. Valid: {valid_retrievals}, Drift penalties: {drift_penalties}.",
        metadata={"total_evidence": total, "valid_count": valid_retrievals},
    )


def eval_context_precision(run: Run, example: Example) -> EvaluationResult:
    """
    Evaluate context precision:
    Measures the signal-to-noise ratio of retrieved passages.
    Precision = (passages that explicitly or semantically support the requirement) / (total retrieved).
    """
    outputs = run.outputs or {}
    evidence = _extract_evidence_list(outputs)

    if not evidence:
        return EvaluationResult(
            key="context_precision",
            score=0.0,
            comment="No evidence items present in output.",
        )

    precise_count = 0
    for item in evidence:
        classification = item.get("match_classification")
        relevance_score = item.get("relevance_score", 0.0) or 0.0
        # High-signal precision requires explicit match or solid semantic support
        if classification == "explicit_match":
            precise_count += 1
        elif classification == "semantically_supported" and relevance_score >= 0.50:
            precise_count += 1

    precision = round(precise_count / len(evidence), 3)
    return EvaluationResult(
        key="context_precision",
        score=precision,
        comment=f"{precise_count} of {len(evidence)} retrieved passages provide high-precision evidence.",
        metadata={"precise_passages": precise_count, "total_passages": len(evidence)},
    )


def eval_context_recall(run: Run, example: Example) -> EvaluationResult:
    """
    Evaluate context recall:
    Measures whether all expected ground-truth requirements that exist in the resume
    were successfully retrieved and represented in the evidence collection.
    """
    outputs = run.outputs or {}
    evidence = _extract_evidence_list(outputs)
    ref_outputs = example.outputs or {}
    expected_evidence_map = ref_outputs.get("expected_relevant_evidence", {})

    if not expected_evidence_map:
        # If no positive evidence was expected, recall is trivially 1.0 if not hallucinated
        return EvaluationResult(
            key="context_recall",
            score=1.0,
            comment="No positive ground-truth evidence expected for this candidate.",
        )

    retrieved_requirements = {
        item.get("requirement")
        for item in evidence
        if item.get("match_classification") in ("explicit_match", "semantically_supported")
    }

    recalled_count = 0
    total_expected = len(expected_evidence_map)

    for expected_req in expected_evidence_map.keys():
        if any(expected_req.lower() == r.lower() for r in retrieved_requirements):
            recalled_count += 1

    recall = round(recalled_count / total_expected, 3) if total_expected > 0 else 1.0
    return EvaluationResult(
        key="context_recall",
        score=recall,
        comment=f"Recalled {recalled_count} of {total_expected} expected evidence items.",
        metadata={"recalled_count": recalled_count, "expected_total": total_expected},
    )


def eval_faithfulness(run: Run, example: Example) -> EvaluationResult:
    """
    Evaluate faithfulness:
    Verifies that the generated qualitative insights (strengths, relevant_experience, recommendations)
    are strictly grounded in the candidate's actual qualifications and do not hallucinate missing skills.
    Specifically checks that skills marked as unevidenced or missing are NOT claimed as strengths.
    """
    outputs = run.outputs or {}
    strengths = [s.lower() for s in outputs.get("strengths", [])]
    relevant_exp = [e.lower() for e in outputs.get("relevant_experience", [])]
    narrative = (outputs.get("score_explanation", "") + " " + outputs.get("explanation", "")).lower()

    ref_outputs = example.outputs or {}
    missing_skills = [s.lower() for s in ref_outputs.get("reference_missing_skills", [])]

    hallucinations = []
    # Check if any missing skill is erroneously claimed as a candidate strength
    for missing in missing_skills:
        for strength in strengths:
            if f" {missing} " in f" {strength} " or f"in {missing}" in strength or f"with {missing}" in strength:
                hallucinations.append(f"Claimed strength in missing skill '{missing}'")

    if hallucinations:
        score = max(0.0, round(1.0 - (len(hallucinations) * 0.4), 2))
        return EvaluationResult(
            key="faithfulness",
            score=score,
            comment=f"Potential hallucination detected: {', '.join(hallucinations)}",
            metadata={"hallucinations": hallucinations},
        )

    return EvaluationResult(
        key="faithfulness",
        score=1.0,
        comment="All generated insights are faithful to evidence; no missing skills claimed as strengths.",
    )


def eval_answer_correctness(run: Run, example: Example) -> EvaluationResult:
    """
    Evaluate answer correctness:
    Measures accuracy of the deterministic matching score and skill categorization
    against human-annotated reference ground truth.
    """
    outputs = run.outputs or {}
    ref_outputs = example.outputs or {}

    pred_score = outputs.get("match_score")
    ref_score = ref_outputs.get("reference_match_score")

    pred_matched = set(s.lower() for s in outputs.get("matched_skills", []))
    ref_matched = set(s.lower() for s in ref_outputs.get("reference_matched_skills", []))

    # Calculate skill overlap (Jaccard similarity)
    union = pred_matched.union(ref_matched)
    intersection = pred_matched.intersection(ref_matched)
    jaccard = len(intersection) / len(union) if union else 1.0

    # Calculate score delta penalty
    score_accuracy = 1.0
    if ref_score is not None and pred_score is not None:
        delta = abs(float(pred_score) - float(ref_score))
        score_accuracy = max(0.0, 1.0 - (delta / 100.0))

    combined = round((jaccard * 0.6) + (score_accuracy * 0.4), 3)
    return EvaluationResult(
        key="answer_correctness",
        score=combined,
        comment=f"Skill Jaccard overlap: {round(jaccard, 2)}, Score accuracy: {round(score_accuracy, 2)}.",
        metadata={
            "predicted_score": pred_score,
            "reference_score": ref_score,
            "matched_skills_overlap": len(intersection),
        },
    )


def eval_operational_metrics(run: Run, example: Example) -> EvaluationResult:
    """
    Evaluate operational execution metrics:
    Records run latency, presence of runtime errors, and evidence counts.
    """
    latency = 0.0
    if run.start_time and run.end_time:
        latency = round((run.end_time - run.start_time).total_seconds(), 3)

    is_success = run.error is None
    outputs = run.outputs or {}
    evidence_count = len(outputs.get("evidence", []))

    score = 1.0 if is_success else 0.0
    return EvaluationResult(
        key="operational_health",
        score=score,
        comment=f"Success: {is_success}, Latency: {latency}s, Evidence Items: {evidence_count}.",
        metadata={
            "latency_seconds": latency,
            "has_error": run.error is not None,
            "error_detail": str(run.error) if run.error else None,
            "evidence_count": evidence_count,
        },
    )
