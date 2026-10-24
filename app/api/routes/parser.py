
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.resume import ResumeResponse
from app.services.pdf_extractor import extract_text_from_pdf
from app.services.resume_parser import parse_resume_text
from app.utils.file_validation import validate_resume_file


router = APIRouter(
    prefix="/api/v1/parser",
    tags=["Resume Parser"],
)


@router.post("/parse", response_model=ResumeResponse)
async def parse_resume(file: UploadFile = File(...)):
    """
    Upload a resume PDF and extract structured information.
    """

    try:
        pdf_bytes = await validate_resume_file(file)

        resume_text = extract_text_from_pdf(pdf_bytes)
        parsed_resume = parse_resume_text(resume_text)

        return parsed_resume

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    finally:
        await file.close()
