
import pytest
from fastapi import HTTPException, UploadFile
from io import BytesIO

from app.utils.file_validation import validate_resume_file


@pytest.mark.asyncio
async def test_reject_non_pdf_file():
    file = UploadFile(
        filename="resume.txt",
        file=BytesIO(b"This is not a PDF"),
        headers={"content-type": "text/plain"},
    )

    with pytest.raises(HTTPException) as exc_info:
        await validate_resume_file(file)

    assert exc_info.value.status_code == 415


@pytest.mark.asyncio
async def test_reject_empty_file():
    file = UploadFile(
        filename="resume.pdf",
        file=BytesIO(b""),
        headers={"content-type": "application/pdf"},
    )

    with pytest.raises(HTTPException) as exc_info:
        await validate_resume_file(file)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_reject_invalid_pdf_signature():
    file = UploadFile(
        filename="resume.pdf",
        file=BytesIO(b"This is not a real PDF"),
        headers={"content-type": "application/pdf"},
    )

    with pytest.raises(HTTPException) as exc_info:
        await validate_resume_file(file)

    assert exc_info.value.status_code == 400
