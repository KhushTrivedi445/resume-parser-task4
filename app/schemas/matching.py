from pydantic import BaseModel, Field


class MatchRequest(BaseModel):
    resume_text: str = Field(min_length=1, max_length=50_000)
    job_description: str = Field(min_length=1, max_length=50_000)


class MatchResponse(BaseModel):
    score: float = Field(ge=0, le=100)
    matched_skills: list[str]
    missing_skills: list[str]
    explanation: str
