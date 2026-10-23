
from pydantic import BaseModel, Field


class Education(BaseModel):
    degree: str | None = None
    institution: str | None = None
    graduation_year: int | None = None


class ResumeResponse(BaseModel):
    name: str | None = None
    email: str | None = None
    phone_number: str | None = None
    skills: list[str] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)

