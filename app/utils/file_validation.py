
from fastapi import HTTPException, UploadFile


MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


async def validate_resume_file(file: UploadFile) -> bytes:
    """
    Validate an uploaded resume PDF and return its contents.

    Checks:
    - Filename and extension
    - Content type
    - File size
    - PDF file signature
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=415,
            detail="Only PDF files are supported.",
        )

    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=415,
            detail="The uploaded file must have PDF content type.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="The file exceeds the maximum size of 5 MB.",
        )

    if not contents.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file does not have a valid PDF signature.",
        )

    return contents
