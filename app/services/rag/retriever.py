import re
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings

# Global cached embedding model instance to avoid re-loading weights per request
_embedding_instance: HuggingFaceEmbeddings | None = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Get or initialize the HuggingFaceEmbeddings model singleton.
    Uses the model specified in settings.EMBEDDING_MODEL.
    """
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL
        )
    return _embedding_instance


def chunk_text(
    text: str,
    source: str = "resume",
    max_chunk_size: int = 350,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    Chunk input text into coherent passages preserving metadata.
    Splits by paragraph or sentence blocks, ensuring long passages are subdivided.
    """
    raw_paragraphs = [
        p.strip() for p in re.split(r"\n\s*\n|\r\n\s*\r\n", text) if p.strip()
    ]

    # If document has no clear blank-line paragraph separation, fallback to line groups
    if len(raw_paragraphs) <= 1:
        raw_paragraphs = [
            line.strip() for line in text.splitlines() if line.strip()
        ]

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for para in raw_paragraphs:
        para_len = len(para)
        if para_len > max_chunk_size:
            # If current accumulator has content, flush it
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_len = 0
            # Split long paragraph by sentence or fixed window
            words = para.split()
            sub_chunk: List[str] = []
            sub_len = 0
            for word in words:
                sub_chunk.append(word)
                sub_len += len(word) + 1
                if sub_len >= max_chunk_size:
                    chunks.append(" ".join(sub_chunk))
                    # Retain last few words for overlap
                    overlap_words = sub_chunk[-max(1, chunk_overlap // 8):]
                    sub_chunk = list(overlap_words)
                    sub_len = sum(len(w) + 1 for w in sub_chunk)
            if sub_chunk:
                chunks.append(" ".join(sub_chunk))
        else:
            if current_len + para_len + 1 > max_chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = [para]
                current_len = para_len
            else:
                current_chunk.append(para)
                current_len += para_len + 1

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    if not chunks:
        # Fallback if text was not empty but produced no chunks
        chunks = [text.strip()]

    documents = [
        Document(
            page_content=chunk,
            metadata={
                "source": source,
                "chunk_id": idx,
                "total_chunks": len(chunks),
                "char_length": len(chunk),
            },
        )
        for idx, chunk in enumerate(chunks)
    ]
    return documents


def build_vector_store(
    text: str,
    source: str = "resume",
) -> FAISS:
    """
    Build an in-memory FAISS vector index for a single document.
    Ensures total request isolation: no persistent or cross-request vector leakage.
    """
    if not text or not text.strip():
        raise ValueError("Cannot build a vector store from empty text.")

    documents = chunk_text(text, source=source)
    if not documents:
        raise ValueError("Cannot build a vector store: no readable passages extracted.")

    embeddings = get_embedding_model()
    return FAISS.from_documents(documents, embeddings)


def retrieve_relevant_passages(
    vector_store: FAISS,
    query: str,
    k: int = 4,
) -> List[Document]:
    """
    Retrieve top-k semantically relevant passages for a query.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    return vector_store.similarity_search(query, k=k)


def retrieve_passages_with_scores(
    vector_store: FAISS,
    query: str,
    k: int = 4,
) -> List[Tuple[Document, float]]:
    """
    Retrieve top-k passages with L2 distance / similarity scores.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    raw_results = vector_store.similarity_search_with_score(query, k=k)
    return [(doc, float(score)) for doc, score in raw_results]
