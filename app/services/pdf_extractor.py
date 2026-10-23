
from io import BytesIO

from pypdf import PdfReader


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract readable text from a PDF document.

    Args:
        pdf_bytes: The uploaded PDF file contents.

    Returns:
        The extracted text from all pages.

    Raises:
        ValueError: If the PDF is invalid or contains no readable text.
    """
    try:
        reader = PdfReader(BytesIO(pdf_bytes))

        extracted_pages = []

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                extracted_pages.append(page_text)

    except Exception as exc:
        raise ValueError("The uploaded file is not a valid readable PDF.") from exc

    full_text = "\n\n".join(extracted_pages).strip()

    if not full_text:
        raise ValueError(
            "No readable text was found in the PDF. "
            "The document may be scanned or empty."
        )

    return full_text

