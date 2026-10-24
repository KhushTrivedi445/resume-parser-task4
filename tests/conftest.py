import json
from io import BytesIO
import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.database import Base, get_db
from app.main import app


@pytest.fixture
def client():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture
def create_test_pdf():
    """Helper fixture to generate valid PDF documents using ReportLab."""
    def _create_pdf(lines: list[str]) -> bytes:
        buffer = BytesIO()
        pdf_canvas = canvas.Canvas(buffer)
        y = 780
        for line in lines:
            pdf_canvas.drawString(50, y, line)
            y -= 20
        pdf_canvas.save()
        return buffer.getvalue()
    return _create_pdf


@pytest.fixture(autouse=True)
def mock_groq_llm(monkeypatch):
    """
    Automatically mock the Groq LLM across test runs to ensure tests are fast,
    deterministic, offline-capable, and do not consume live API quota.
    """
    class MockMessage:
        def __init__(self, content: str):
            self.content = content

    def fake_invoke(self, prompt, **kwargs):
        payload = {
            "strengths": [
                "Strong demonstrated background in core required technical skills.",
                "Relevant hands-on projects and software engineering experience."
            ],
            "relevant_experience": [
                "Built and maintained scalable services and databases."
            ],
            "recommendations": [
                "Include concrete evidence or certifications for any unevidenced skills."
            ],
            "limitations": [
                "Evaluation is limited to explicit resume text; unevidenced skills do not prove absence of competence."
            ],
            "narrative_summary": "Candidate displays strong technical alignment with core requirements."
        }
        return MockMessage(json.dumps(payload))

    monkeypatch.setattr("langchain_groq.ChatGroq.invoke", fake_invoke, raising=False)
