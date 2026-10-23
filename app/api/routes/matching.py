from fastapi import APIRouter

from app.schemas.matching import MatchRequest, MatchResponse
from app.services.rag.answer_generator import generate_answer
from app.services.rag.chunker import chunk_text
from app.services.rag.embeddings import embed_text
from app.services.rag.retriever import retrieve
from app.services.rag.vector_store import VectorStore
from app.services.resume_parser import match_resume_to_job

router = APIRouter(prefix="/matching", tags=["matching"])


@router.post("/match", response_model=MatchResponse)
def match_resume(request: MatchRequest) -> MatchResponse:
    result = match_resume_to_job(request.resume_text, request.job_description)
    chunks = chunk_text(request.job_description)
    store = VectorStore()
    store.add_documents(chunks, [embed_text(chunk) for chunk in chunks])
    matches = retrieve(
        request.resume_text,
        store,
        query_embedding=embed_text(request.resume_text),
        top_k=2,
    )
    return MatchResponse(
        score=result.score,
        matched_skills=result.matched_skills,
        missing_skills=result.missing_skills,
        explanation=generate_answer(matches),
    )
