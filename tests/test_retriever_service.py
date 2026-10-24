import pytest
from app.services.rag.retriever import (
    build_vector_store,
    chunk_text,
    retrieve_passages_with_scores,
    retrieve_relevant_passages,
)


def test_chunk_text_preserves_metadata():
    """Verify chunk_text produces documents with metadata."""
    sample_text = (
        "Experienced software developer building backend systems.\n\n"
        "Proficient in Python, FastAPI, and PostgreSQL databases.\n\n"
        "Deployed applications using Docker and AWS."
    )
    docs = chunk_text(sample_text, source="candidate_resume")
    assert len(docs) >= 1
    for doc in docs:
        assert doc.metadata["source"] == "candidate_resume"
        assert "chunk_id" in doc.metadata
        assert "total_chunks" in doc.metadata
        assert len(doc.page_content) > 0


def test_build_vector_store_rejects_empty_text():
    """Verify empty or whitespace text raises ValueError."""
    with pytest.raises(ValueError, match="Cannot build a vector store"):
        build_vector_store("")

    with pytest.raises(ValueError, match="Cannot build a vector store"):
        build_vector_store("   \n\t  ")


def test_retrieve_rejects_empty_query_or_invalid_k():
    """Verify empty query or invalid k raises ValueError."""
    store = build_vector_store("Python developer building REST APIs.", source="test")

    with pytest.raises(ValueError, match="Query cannot be empty"):
        retrieve_relevant_passages(store, query="")

    with pytest.raises(ValueError, match="k must be at least 1"):
        retrieve_relevant_passages(store, query="Python", k=0)


def test_retrieve_passages_and_scores():
    """Verify retrieval returns relevant passages with similarity scores."""
    resume_text = (
        "Developed machine learning models using scikit-learn and PyTorch.\n\n"
        "Created fast web services with FastAPI and Python.\n\n"
        "Managed database schema migrations with PostgreSQL."
    )
    store = build_vector_store(resume_text, source="resume")
    results = retrieve_passages_with_scores(store, query="machine learning PyTorch", k=2)

    assert len(results) >= 1
    top_doc, distance = results[0]
    assert "machine learning" in top_doc.page_content.lower()
    assert isinstance(distance, float)
