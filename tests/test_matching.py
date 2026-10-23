def test_matching_endpoint_reports_overlap_and_explanation(client):
    response = client.post(
        "/api/v1/matching/match",
        json={
            "resume_text": "Python FastAPI SQL",
            "job_description": "Python and FastAPI required; AWS is a plus.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["score"] == 66.67
    assert body["matched_skills"] == ["python", "fastapi"]
    assert body["missing_skills"] == ["aws"]
    assert "Relevant job-description content" in body["explanation"]


def test_matching_rejects_empty_inputs(client):
    response = client.post(
        "/api/v1/matching/match",
        json={"resume_text": "", "job_description": "Python"},
    )

    assert response.status_code == 422
