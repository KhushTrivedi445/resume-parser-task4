from app.services.rag.embeddings import embed_text
from app.services.rag.vector_store import VectorStore


def retrieve(
    query: str,
    store: VectorStore,
    query_embedding: list[float] | None = None,
    top_k: int = 3,
) -> list[str]:
    vector = query_embedding if query_embedding is not None else embed_text(query)
    return [text for text, _score in store.search(vector, top_k=top_k)]
