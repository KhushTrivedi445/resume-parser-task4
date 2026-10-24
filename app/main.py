from fastapi import FastAPI

from app.api.routes.parser import router as parser_router
from app.api.routes.matching import router as matching_router

app = FastAPI(
    title="Resume Parser & Job Matcher",
    description=(
        "API for structured resume parsing "
        "and RAG-based resume-job description matching."
    ),
    version="0.1.0",
)

app.include_router(parser_router)
app.include_router(matching_router)


@app.get("/")
def root():
    return {
        "message": "Welcome to Resume Parser & Job Matcher API"
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}
