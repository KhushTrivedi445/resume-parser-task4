from io import BytesIO
import pytest


def test_matching_endpoint_json_success(client):
    """Verify JSON-based matching returns structured report and deterministic score."""
    response = client.post(
        "/api/v1/matching/match",
        json={
            "resume_text": "Experienced Python developer with FastAPI and SQL skills.",
            "job_description": "We require Python and FastAPI. AWS experience is a plus.",
        },
    )

    assert response.status_code == 200
    body = response.json()

    # Required: Python, FastAPI (2/2 matched = 100%)
    # Preferred: AWS (0/1 matched = 0%)
    # Weighted score: 0.80 * 100 + 0.20 * 0 = 80.0
    assert body["match_score"] == 80.0
    assert body["score"] == 80.0
    assert body["skill_coverage"] == 66.67  # 2 matched / 3 total identified = 66.67%
    assert "Python" in body["matched_skills"]
    assert "FastAPI" in body["matched_skills"]
    assert "AWS" in body["preferred_skills_missing"]
    assert isinstance(body["strengths"], list) and len(body["strengths"]) > 0
    assert isinstance(body["recommendations"], list) and len(body["recommendations"]) > 0
    assert isinstance(body["evidence"], list) and len(body["evidence"]) > 0
    assert "80.0" in body["score_explanation"] or "weighted" in body["score_explanation"].lower()


def test_matching_endpoint_multipart_pdf_success(client, create_test_pdf):
    """Verify PDF upload matching with ReportLab-generated resume PDF."""
    pdf_bytes = create_test_pdf([
        "Jane Doe",
        "jane.doe@example.com",
        "SKILLS",
        "Python, Docker, PostgreSQL, REST APIs",
        "EXPERIENCE",
        "Built backend microservices using FastAPI and PostgreSQL databases.",
    ])

    response = client.post(
        "/api/v1/matching/match",
        data={
            "job_description": "Backend Engineer: Python, Docker, and PostgreSQL required. Kubernetes is a plus."
        },
        files={
            "file": ("resume.pdf", pdf_bytes, "application/pdf")
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert "Python" in body["matched_skills"]
    assert "Docker" in body["matched_skills"]
    assert "PostgreSQL" in body["matched_skills"]
    assert "Kubernetes" in body["preferred_skills_missing"]
    # 3/3 required (100%), 0/1 preferred (0%) -> 80.0%
    assert body["match_score"] == 80.0
    assert body["total_identified_requirements"] == 4
    assert body["matched_requirements_count"] == 3
    assert len(body["evidence"]) > 0


def test_matching_rejects_empty_inputs(client):
    """Verify empty resume text or empty job description returns 422."""
    response = client.post(
        "/api/v1/matching/match",
        json={"resume_text": "", "job_description": "Python"},
    )
    assert response.status_code == 422

    response_no_jd = client.post(
        "/api/v1/matching/match",
        json={"resume_text": "Python developer", "job_description": ""},
    )
    assert response_no_jd.status_code == 422


def test_matching_rejects_non_pdf_file(client):
    """Verify non-PDF file upload returns 415."""
    response = client.post(
        "/api/v1/matching/match",
        data={"job_description": "Python developer"},
        files={"file": ("resume.txt", b"plain text resume", "text/plain")},
    )
    assert response.status_code == 415


def test_matching_rejects_empty_file(client):
    """Verify empty file upload returns 400."""
    response = client.post(
        "/api/v1/matching/match",
        data={"job_description": "Python developer"},
        files={"file": ("resume.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400


def test_matching_rejects_malformed_pdf(client):
    """Verify malformed PDF signature returns 400 or 422."""
    response = client.post(
        "/api/v1/matching/match",
        data={"job_description": "Python developer"},
        files={"file": ("resume.pdf", b"NOT A VALID PDF CONTENT", "application/pdf")},
    )
    assert response.status_code in (400, 422)


def test_matching_no_identifiable_jd_requirements(client):
    """Verify no misleading 0 score when JD has no identifiable requirements."""
    response = client.post(
        "/api/v1/matching/match",
        json={
            "resume_text": "Python FastAPI developer.",
            "job_description": "We are seeking a great team player with high enthusiasm and passion for excellence.",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["match_score"] is None
    assert body["total_identified_requirements"] == 0
    assert "No identifiable" in body["score_explanation"]
