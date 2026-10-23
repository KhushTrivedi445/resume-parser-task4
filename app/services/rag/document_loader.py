from pathlib import Path

from app.services.pdf_extractor import extract_pdf_text


def load_document(path: str | Path) -> str:
    document = Path(path)
    if document.suffix.casefold() == ".pdf":
        return extract_pdf_text(document.read_bytes())
    if document.suffix.casefold() in {".txt", ".md"}:
        return document.read_text(encoding="utf-8")
    raise ValueError("Supported document types are PDF, TXT, and Markdown")
