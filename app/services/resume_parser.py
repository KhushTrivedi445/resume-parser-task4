
import re

from app.schemas.resume import Education, ResumeResponse


# Initial skills vocabulary. We can expand this as we test more resumes.
KNOWN_SKILLS = [
    "Python",
    "Java",
    "C++",
    "JavaScript",
    "TypeScript",
    "SQL",
    "Machine Learning",
    "Deep Learning",
    "Natural Language Processing",
    "Computer Vision",
    "Artificial Intelligence",
    "Data Science",
    "Data Analysis",
    "Pandas",
    "NumPy",
    "Scikit-learn",
    "TensorFlow",
    "PyTorch",
    "FastAPI",
    "Flask",
    "Django",
    "React",
    "HTML",
    "CSS",
    "Git",
    "Docker",
    "AWS",
    "Azure",
    "LangChain",
    "LangGraph",
    "FAISS",
    "ChromaDB",
]


def extract_email(text: str) -> str | None:
    """Extract the first email address found in the resume."""
    match = re.search(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        text,
    )
    return match.group(0) if match else None


def extract_phone_number(text: str) -> str | None:
    """Extract a likely phone number from the resume."""
    match = re.search(
        r"(?<!\w)(?:\+?\d{1,3}[-.\s]?)?"
        r"(?:\(?\d{3,5}\)?[-.\s]?)?"
        r"\d{5,10}(?!\w)",
        text,
    )

    if not match:
        return None

    candidate = match.group(0).strip()
    digit_count = sum(character.isdigit() for character in candidate)

    # Reject short numeric sequences that are unlikely to be phone numbers.
    if digit_count < 10 or digit_count > 15:
        return None

    return candidate


def extract_name(text: str) -> str | None:
    """Infer a candidate name from the opening lines of a resume."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines[:8]:
        normalized = line.strip(" |:-\t")

        if len(normalized) > 60:
            continue

        if "@" in normalized or any(char.isdigit() for char in normalized):
            continue

        if re.search(
            r"\b(resume|curriculum vitae|cv|portfolio|"
            r"linkedin|github|education|experience|skills|"
            r"contact|summary|objective)\b",
            normalized,
            re.IGNORECASE,
        ):
            continue

        words = normalized.split()

        if 2 <= len(words) <= 4 and all(
            re.fullmatch(r"[A-Za-z][A-Za-z'.-]*", word)
            for word in words
        ):
            return normalized

    return None


def extract_skills(text: str) -> list[str]:
    """Find known skills mentioned in the resume using boundary-safe normalized matching."""
    from app.services.rag.skill_matcher import extract_skills_from_text
    return extract_skills_from_text(text)



def extract_education(text: str) -> list[Education]:
    """Extract education entries by examining the education section."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # Locate the education section.
    education_start = None

    for index, line in enumerate(lines):
        if re.match(r"^EDUCATION\b", line, re.IGNORECASE):
            education_start = index + 1
            break

    if education_start is None:
        return []

    # Stop at the next major resume section.
    section_endings = {
        "experience",
        "work experience",
        "projects",
        "technical skills",
        "skills",
        "certifications",
        "certifications & activities",
        "activities",
        "summary",
    }

    education_lines = []

    for line in lines[education_start:]:
        normalized = line.strip().rstrip(":").lower()

        if normalized in section_endings:
            break

        education_lines.append(line)

    if not education_lines:
        return []

    # Find degree lines and associate them with nearby institution/year lines.
    entries = []

    degree_pattern = re.compile(
        r"\b(?:B\.?\s?Tech(?:nology)?|M\.?\s?Tech(?:nology)?|"
        r"B\.?\s?E\.?|M\.?\s?E\.?|B\.?\s?Sc\.?|M\.?\s?Sc\.?|"
        r"B\.?\s?CA|M\.?\s?CA|MBA|Ph\.?D\.?|"
        r"Bachelor(?:'s)?|Master(?:'s)?)\b",
        re.IGNORECASE,
    )

    year_pattern = re.compile(r"\b(?:19|20)\d{2}\b")

    for index, line in enumerate(education_lines):
        degree_match = degree_pattern.search(line)

        if not degree_match:
            continue

        degree = degree_match.group(0)
        year_matches = year_pattern.findall(line)

        # Also inspect the preceding line for dates, such as
        # "2023 – July 2027 (Expected)".
        if index > 0:
            year_matches = (
                year_pattern.findall(education_lines[index - 1])
                + year_matches
            )

        graduation_year = (
            int(year_matches[-1]) if year_matches else None
        )

        institution = None

        # The institution may appear on the line immediately before
        # the degree, as it does in the supplied resume.
        if index > 0:
            previous_line = education_lines[index - 1]

            if not degree_pattern.search(previous_line):
                if re.search(
                    r"\b(university|college|institute|school)\b",
                    previous_line,
                    re.IGNORECASE,
                ):
                    institution = re.split(
                        r"\s+\d{4}\b",
                        previous_line,
                        maxsplit=1,
                    )[0].strip(" |:-")

        # Include the full degree line as the degree description.
        degree_description = line.split("|")[0].strip()

        entry = Education(
            degree=degree_description,
            institution=institution,
            graduation_year=graduation_year,
        )

        # Avoid adding the same qualification twice.
        if entry not in entries:
            entries.append(entry)

    return entries


def parse_resume_text(text: str) -> ResumeResponse:
    """Convert extracted resume text into structured fields."""
    if not text.strip():
        raise ValueError("Resume text cannot be empty.")

    return ResumeResponse(
        name=extract_name(text),
        email=extract_email(text),
        phone_number=extract_phone_number(text),
        skills=extract_skills(text),
        education=extract_education(text),
    )

