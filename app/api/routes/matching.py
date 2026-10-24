import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, status
from starlette.datastructures import UploadFile

from app.schemas.matching import MatchingResponse
from app.services.pdf_extractor import extract_text_from_pdf
from app.services.rag.matching_service import match_resume_to_job
from app.utils.file_validation import validate_resume_file

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/matching",
    tags=["Matching"],
)


@router.post(
    "/match",
    response_model=MatchingResponse,
    status_code=status.HTTP_200_OK,
    summary="Match Resume to Job Description",
    description=(
        "Accepts a resume PDF (via multipart/form-data upload) or plain text resume, "
        "along with a target job description. Performs semantic retrieval with FAISS, "
        "deterministic skill matching, and LLM-powered evidence-based insights."
    ),
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "file": {
                                "type": "string",
                                "format": "binary",
                                "description": "Resume PDF document (up to 5 MB)",
                            },
                            "job_description": {
                                "type": "string",
                                "description": "Target job description and requirements text",
                            },
                            "resume_text": {
                                "type": "string",
                                "description": "Optional raw resume text (if not uploading PDF)",
                            },
                        },
                        "required": ["job_description"],
                    }
                },
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "resume_text": {
                                "type": "string",
                                "description": "Candidate resume text content",
                            },
                            "job_description": {
                                "type": "string",
                                "description": "Target job description text",
                            },
                        },
                        "required": ["resume_text", "job_description"],
                    }
                },
            }
        }
    },
)
async def match_resume_endpoint(request: Request) -> MatchingResponse:
    """
    Process resume and job description to generate an explainable matching report.
    """
    content_type = request.headers.get("content-type", "").lower()
    resume_text: str = ""
    job_description: str = ""

    try:
        if "application/json" in content_type:
            try:
                payload = await request.json()
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid JSON payload.",
                ) from exc

            if not isinstance(payload, dict):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="Request body must be a valid JSON object.",
                )

            resume_text = str(payload.get("resume_text", "")).strip()
            job_description = str(payload.get("job_description", "")).strip()

            if not resume_text:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="resume_text cannot be empty.",
                )

            if not job_description:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="job_description cannot be empty.",
                )

        else:
            # Multipart form data / urlencoded
            try:
                form = await request.form()
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to parse form data.",
                ) from exc

            raw_file = form.get("file")
            raw_resume_text = form.get("resume_text")
            raw_jd = form.get("job_description")

            if raw_jd and isinstance(raw_jd, str):
                job_description = raw_jd.strip()

            if not job_description:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="job_description is required and cannot be empty.",
                )

            # Check if file was provided (UploadFile or has filename)
            if raw_file and hasattr(raw_file, "filename") and raw_file.filename:
                # Validate and extract PDF
                try:
                    pdf_bytes = await validate_resume_file(raw_file)
                    resume_text = extract_text_from_pdf(pdf_bytes)
                finally:
                    await raw_file.close()
            elif raw_resume_text and isinstance(raw_resume_text, str) and raw_resume_text.strip():
                resume_text = raw_resume_text.strip()
            else:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="Either a resume PDF ('file') or 'resume_text' must be provided.",
                )

        # Execute matching service
        try:
            report = match_resume_to_job(
                resume_text=resume_text,
                job_description=job_description,
            )
            return report

        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        logger.error(f"Unexpected matching error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while analyzing the resume and job description.",
        ) from exc
