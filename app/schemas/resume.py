from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


class Education(BaseModel):
    degree: Optional[str] = None
    institution: Optional[str] = None
    graduation_year: Optional[int] = None


class ResumeResponse(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone_number: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    institution: Optional[str] = None
    graduation_year: Optional[int] = None

    @model_validator(mode="after")
    def populate_top_level_education(self) -> "ResumeResponse":
        """
        Populate top-level institution and graduation_year from education entries
        if available, while preserving the full education list.
        """
        if self.education and len(self.education) > 0:
            primary = self.education[0]
            if self.institution is None and primary.institution:
                self.institution = primary.institution
            if self.graduation_year is None and primary.graduation_year:
                self.graduation_year = primary.graduation_year
        return self
