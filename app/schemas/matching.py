from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


class EvidenceItem(BaseModel):
    requirement: str = Field(..., description="Job requirement or skill evaluated.")
    supporting_passage: str = Field(
        ..., description="Extracted resume passage supporting or relating to the requirement."
    )
    match_classification: str = Field(
        ...,
        description="Match classification: 'explicit_match', 'semantically_supported', or 'unevidenced'.",
    )
    relevance_score: Optional[float] = Field(
        default=None,
        description="Semantic similarity or confidence score between 0.0 and 1.0.",
    )
    source: str = Field(default="resume", description="Source document tag.")


class MatchingRequest(BaseModel):
    """Schema for optional direct text-based matching request."""
    resume_text: Optional[str] = Field(
        default=None, description="Plain text extracted from resume."
    )
    job_description: str = Field(
        ..., min_length=1, description="Job description text."
    )


class MatchingResponse(BaseModel):
    match_score: Optional[float] = Field(
        default=None,
        description="Deterministic score between 0 and 100, or null if no identifiable requirements exist.",
    )
    # Convenience alias field for backwards compatibility with tests and callers
    score: Optional[float] = Field(
        default=None,
        description="Alias for match_score.",
    )
    score_explanation: str = Field(
        ...,
        description="Detailed transparent explanation of how the score was calculated.",
    )
    # Convenience alias field for explanation
    explanation: str = Field(
        default="",
        description="Alias for score_explanation.",
    )
    skill_coverage: Optional[float] = Field(
        default=None,
        description="Unweighted baseline ratio of matched requirements over total requirements.",
    )
    total_identified_requirements: int = Field(
        default=0,
        description="Total number of distinct requirements or skills identified in the JD.",
    )
    matched_requirements_count: int = Field(
        default=0,
        description="Number of JD requirements evidenced in the candidate's resume.",
    )
    matched_skills: List[str] = Field(
        default_factory=list,
        description="Required skills explicitly identified in both JD and resume.",
    )
    missing_skills: List[str] = Field(
        default_factory=list,
        description="Required skills from the JD that are not evidenced in the resume.",
    )
    preferred_skills_matched: List[str] = Field(
        default_factory=list,
        description="Preferred/bonus skills identified in both JD and resume.",
    )
    preferred_skills_missing: List[str] = Field(
        default_factory=list,
        description="Preferred/bonus skills from the JD not evidenced in the resume.",
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Key candidate strengths relevant to the job description.",
    )
    relevant_experience: List[str] = Field(
        default_factory=list,
        description="Summary of candidate's relevant experience and projects.",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable recommendations for improving alignment with the role.",
    )
    evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="Retrieved resume passages supporting specific requirements.",
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Methodological limitations of the match analysis.",
    )

    @model_validator(mode="after")
    def sync_aliases(self) -> "MatchingResponse":
        if self.score is None and self.match_score is not None:
            self.score = self.match_score
        elif self.match_score is None and self.score is not None:
            self.match_score = self.score
        if not self.explanation and self.score_explanation:
            self.explanation = self.score_explanation
        elif not self.score_explanation and self.explanation:
            self.score_explanation = self.explanation
        return self
