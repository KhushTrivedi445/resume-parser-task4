from dataclasses import dataclass

from app.services.rag.embeddings import cosine_similarity


@dataclass(frozen=True)
class StoredDocument:
    text: str
    embedding: list[float]


class VectorStore:
    def __init__(self) -> None:
        self._documents: list[StoredDocument] = []

    def add_documents(self, documents: list[str], embeddings: list[list[float]]) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("Each document must have one embedding")
        self._documents.extend(
            StoredDocument(text=text, embedding=embedding)
            for text, embedding in zip(documents, embeddings)
        )

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[tuple[str, float]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        ranked = (
            (document.text, cosine_similarity(query_embedding, document.embedding))
            for document in self._documents
        )
        return sorted(ranked, key=lambda item: item[1], reverse=True)[:top_k]
