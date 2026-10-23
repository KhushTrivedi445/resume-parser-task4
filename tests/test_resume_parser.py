
from app.services.resume_parser import parse_resume_text
from pprint import pprint
from fastapi.testclient import TestClient
from app.main import app
from io import BytesIO
from pypdf import PdfWriter
from reportlab.pdfgen import canvas


client = TestClient(app)

def test_parse_resume_text():
    resume_text = """
    Aarav Shah
    aarav.shah@example.com
    +91 9876543210

    SKILLS
    Python,python, Machine Learning, FastAPI, SQL,fastapi

    EDUCATION
    B.Tech in Computer Science - 2027
    """

    result = parse_resume_text(resume_text)
    result = parse_resume_text(resume_text)

    print("\nParsed Resume Output:")
    pprint(result.model_dump())


    assert result.name == "Aarav Shah"
    assert result.email == "aarav.shah@example.com"
    assert result.phone_number == "+91 9876543210"

    assert "Python" in result.skills
    assert "Machine Learning" in result.skills
    assert "FastAPI" in result.skills
    assert "SQL" in result.skills

    assert len(result.education) == 1
    assert result.education[0].graduation_year == 2027


def test_parse_resume_with_missing_fields():
    resume_text = """
    Aarav Shah
    Interested in learning new technologies.
    """

    result = parse_resume_text(resume_text)

    assert result.name == "Aarav Shah"
    assert result.email is None
    assert result.phone_number is None
    assert result.skills == []
    assert result.education == []


def test_parse_empty_resume_text():
    try:
        parse_resume_text("   ")
        assert False, "Expected ValueError for empty resume text"
    except ValueError as exc:
        assert str(exc) == "Resume text cannot be empty."

def test_parse_endpoint_rejects_non_pdf():
    response = client.post(
        "/api/v1/parser/parse",
        files={
            "file": (
                "resume.txt",
                b"This is not a PDF",
                "text/plain",
            )
        },
    )

    assert response.status_code == 415
    assert response.json()["detail"] == "Only PDF files are supported."




def test_parse_endpoint_accepts_valid_pdf():
    pdf_buffer = BytesIO()
    pdf = canvas.Canvas(pdf_buffer)

    pdf.drawString(50, 780, "Aarav Shah")
    pdf.drawString(50, 760, "aarav.shah@example.com")
    pdf.drawString(50, 740, "+91 9876543210")
    pdf.drawString(50, 700, "EDUCATION")
    pdf.drawString(50, 680, "B.Tech in Computer Science - 2027")

    pdf.save()
    pdf_bytes = pdf_buffer.getvalue()

    response = client.post(
        "/api/v1/parser/parse",
        files={
            "file": (
                "resume.pdf",
                pdf_bytes,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200

    result = response.json()
    assert result["name"] == "Aarav Shah"
    assert result["email"] == "aarav.shah@example.com"
    assert result["phone_number"] == "+91 9876543210"
    assert isinstance(result["skills"], list)
    assert isinstance(result["education"], list)


def test_parse_endpoint_rejects_malformed_pdf():
    response = client.post(
        "/api/v1/parser/parse",
        files={
            "file": (
                "resume.pdf",
                b"%PDF-1.7 this is not a valid PDF document",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "The uploaded file is not a valid readable PDF."
    )


def test_parse_endpoint_rejects_oversized_pdf():
    oversized_content = b"%PDF-" + b"A" * (5 * 1024 * 1024)

    response = client.post(
        "/api/v1/parser/parse",
        files={
            "file": (
                "large_resume.pdf",
                oversized_content,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 413
    assert response.json()["detail"] == (
        "The file exceeds the maximum size of 5 MB."
    )


def test_parse_endpoint_rejects_missing_file():
    response = client.post("/api/v1/parser/parse")

    assert response.status_code == 422
    assert "file" in str(response.json()["detail"])

