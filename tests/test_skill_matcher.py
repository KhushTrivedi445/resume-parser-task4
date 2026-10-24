import pytest
from app.services.rag.skill_matcher import (
    calculate_deterministic_score,
    classify_jd_requirements,
    extract_skills_from_text,
)


def test_extract_exact_and_alias_skills():
    """Verify exact names and common aliases resolve to canonical skills."""
    text = "Proficient with sklearn, postgres, k8s, py, and fast api."
    skills = extract_skills_from_text(text)

    assert "Scikit-learn" in skills
    assert "PostgreSQL" in skills
    assert "Kubernetes" in skills
    assert "Python" in skills
    assert "FastAPI" in skills


def test_extract_skills_case_insensitivity():
    """Verify case-insensitive matching."""
    text = "DEVELOPED APIS USING FASTAPI, DOCKER, AND AWS."
    skills = extract_skills_from_text(text)

    assert "FastAPI" in skills
    assert "Docker" in skills
    assert "AWS" in skills


def test_boundary_safe_matching_prevents_false_positives():
    """Verify single-character or short skills like 'C' do not match inside CSS or Machine."""
    # Text with CSS and Machine Learning, but NOT programming language C
    text = "Styled responsive frontends using CSS and trained Machine Learning models in Python."
    skills = extract_skills_from_text(text)

    assert "CSS" in skills
    assert "Machine Learning" in skills
    assert "Python" in skills
    assert "C" not in skills  # Crucial: 'C' must not match inside CSS or Machine

    # Text that genuinely has C
    c_text = "Core embedded firmware programmed in C and C++."
    c_skills = extract_skills_from_text(c_text)
    assert "C" in c_skills
    assert "C++" in c_skills


def test_skills_deduplication():
    """Verify multiple occurrences or aliases of same skill are deduplicated."""
    text = "Python, python3, py programming. Postgres and PostgreSQL databases."
    skills = extract_skills_from_text(text)

    assert skills.count("Python") == 1
    assert skills.count("PostgreSQL") == 1


def test_classify_required_versus_preferred():
    """Verify classification of required vs preferred skills in job descriptions."""
    jd = """
    Required Qualifications:
    - 3+ years experience with Python and FastAPI
    - Strong knowledge of SQL databases

    Nice to have / Plus:
    - Experience with Docker is a plus
    - AWS and Kubernetes knowledge preferred
    """
    required, preferred = classify_jd_requirements(jd)

    assert "Python" in required
    assert "FastAPI" in required
    assert "SQL" in required

    assert "Docker" in preferred
    assert "AWS" in preferred
    assert "Kubernetes" in preferred


def test_deterministic_scoring_required_only():
    """Verify deterministic scoring with required skills only."""
    required = ["Python", "FastAPI", "SQL", "Docker"]
    preferred = []
    resume = ["Python", "FastAPI"]

    res = calculate_deterministic_score(required, preferred, resume)

    assert res["matched_requirements_count"] == 2
    assert res["total_identified_requirements"] == 4
    # 2/4 = 50.0%
    assert res["match_score"] == 50.0
    assert res["matched_skills"] == ["Python", "FastAPI"]
    assert res["missing_skills"] == ["SQL", "Docker"]


def test_deterministic_scoring_weighted():
    """Verify 80/20 weighted formula when required and preferred skills are present."""
    required = ["Python", "FastAPI"]  # 2 required
    preferred = ["Docker", "AWS"]     # 2 preferred
    resume = ["Python", "FastAPI", "Docker"]  # 2/2 required (100%), 1/2 preferred (50%)

    res = calculate_deterministic_score(required, preferred, resume)

    # 0.80 * 100% + 0.20 * 50% = 80.0 + 10.0 = 90.0%
    assert res["match_score"] == 90.0
    assert res["required_skill_coverage"] == 100.0
    assert res["preferred_skill_coverage"] == 50.0
    assert res["skill_coverage"] == 75.0  # 3 of 4 total skills = 75%
    assert "Docker" in res["preferred_skills_matched"]
    assert "AWS" in res["preferred_skills_missing"]


def test_deterministic_scoring_consistency():
    """Verify identical inputs yield exact same score on multiple executions."""
    required = ["Python", "PostgreSQL", "Docker"]
    preferred = ["Kubernetes"]
    resume = ["Python", "PostgreSQL"]

    score1 = calculate_deterministic_score(required, preferred, resume)
    score2 = calculate_deterministic_score(required, preferred, resume)

    assert score1["match_score"] == score2["match_score"]
    assert score1["score_explanation"] == score2["score_explanation"]
    assert score1["matched_skills"] == score2["matched_skills"]


def test_deterministic_scoring_zero_requirements():
    """Verify score is None when no requirements are present in the JD."""
    res = calculate_deterministic_score([], [], ["Python", "SQL"])
    assert res["match_score"] is None
    assert res["total_identified_requirements"] == 0
    assert "No identifiable" in res["score_explanation"]
