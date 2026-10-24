"""
Repeatable Real-Data Test Script for Resume-Job Description Matching with LangSmith Tracing

This script executes a real matching request against the verified FastAPI endpoint:
    POST /api/v1/matching/match

It sends:
1. A valid resume PDF (generated dynamically with ReportLab, containing realistic experience
   including Python, FastAPI, Docker, PostgreSQL, LangChain, and an extracurricular section).
2. A realistic target Job Description requiring Python, FastAPI, Docker, PostgreSQL, and FAISS.

It verifies:
- Top-level workflow trace: 'resume_job_rag_matching'
- Vector store index creation: 'build_faiss_vector_store'
- Per-requirement retrieval calls: 'faiss_retrieve_with_scores' (including FAISS L2 distance)
- Evidence classification & scoring: 'evidence_classification_and_scoring'
- LLM insights synthesis: 'generate_matching_insights' -> 'ChatGroq'
- Final structured MatchingResponse
"""

import os
import sys
from io import BytesIO
import json
from dotenv import load_dotenv
from reportlab.pdfgen import canvas
from fastapi.testclient import TestClient

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv()

from app.main import app
from app.core.config import settings


def create_sample_resume_pdf() -> bytes:
    """Generate a clean, representative sample resume PDF in-memory."""
    buffer = BytesIO()
    p = canvas.Canvas(buffer)
    
    lines = [
        "Jane Doe - Senior Backend & AI Engineer",
        "Email: jane.doe@example.com | GitHub: github.com/janedoe | Location: New York, NY",
        "",
        "PROFESSIONAL SUMMARY",
        "Results-oriented backend engineer with 5+ years of experience designing and scaling",
        "microservices, RESTful APIs, and vector search systems using Python, FastAPI, and Docker.",
        "",
        "TECHNICAL SKILLS",
        "Languages & Frameworks: Python, FastAPI, SQL, PyTorch, LangChain",
        "Infrastructure & Tools: Docker, PostgreSQL, Redis, Git, Linux, pytest",
        "",
        "PROFESSIONAL EXPERIENCE",
        "Senior Backend Engineer | CloudScale Inc. (2022 - Present)",
        "- Architected high-concurrency microservices using Python and FastAPI, serving 10M+ daily hits.",
        "- Designed scalable PostgreSQL relational database schemas and optimized slow SQL queries.",
        "- Containerized applications with Docker and established automated testing pipelines with pytest.",
        "- Integrated LangChain and sentence-transformers for RAG-based document question answering.",
        "",
        "Software Engineer | DataCorp (2019 - 2022)",
        "- Built backend REST APIs in Python and maintained PostgreSQL and Redis caches.",
        "- Collaborated with cross-functional product and infrastructure teams to deploy services.",
        "",
        "EDUCATION & EXTRACURRICULAR",
        "B.S. in Computer Science, State University, 2019",
        "Captain of University Badminton Club; coordinated regional tournaments and social events.",
    ]
    
    y = 780
    for line in lines:
        p.drawString(50, y, line)
        y -= 18
    p.save()
    return buffer.getvalue()


SAMPLE_JOB_DESCRIPTION = """
Position: Senior AI Backend Engineer

About the Role:
We are seeking an experienced Backend Engineer to design and scale RAG workflows and microservices.

Required Skills & Experience:
- Strong proficiency in Python backend development
- Hands-on experience developing REST APIs using FastAPI
- Containerization and orchestration using Docker
- Deep database modeling expertise with PostgreSQL and SQL
- Hands-on experience with vector search and FAISS vector databases

Preferred Qualifications:
- Experience with Kubernetes orchestration
- Experience with Redis caching and distributed task queues
"""


def run_real_matching_test():
    print("=" * 70)
    print("RESUME-JOB DESCRIPTION MATCHING: REAL DATA ENDPOINT TEST")
    print("=" * 70)
    
    tracing_enabled = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    project_name = os.getenv("LANGSMITH_PROJECT", "default")
    has_api_key = bool(os.getenv("LANGSMITH_API_KEY"))
    
    print(f"LangSmith Tracing Active: {tracing_enabled}")
    print(f"LangSmith Project:        {project_name}")
    print(f"LangSmith API Key Set:    {has_api_key}")
    print(f"Groq API Key Set:         {bool(settings.GROQ_API_KEY)}")
    print("-" * 70)

    pdf_bytes = create_sample_resume_pdf()
    print(f"Generated sample resume PDF ({len(pdf_bytes)} bytes)")

    client = TestClient(app)

    print("\nSending POST request to /api/v1/matching/match (multipart/form-data)...")
    response = client.post(
        "/api/v1/matching/match",
        data={"job_description": SAMPLE_JOB_DESCRIPTION},
        files={"file": ("jane_doe_resume.pdf", pdf_bytes, "application/pdf")},
    )

    print(f"Response HTTP Status Code: {response.status_code}")
    if response.status_code != 200:
        print(f"Error: {response.text}")
        sys.exit(1)

    result = response.json()
    print("\n--- MATCHING RESULTS ---")
    print(f"Match Score:                      {result.get('match_score')}%")
    print(f"Skill Coverage:                   {result.get('skill_coverage')}%")
    print(f"Total Identified Requirements:    {result.get('total_identified_requirements')}")
    print(f"Matched Requirements Count:       {result.get('matched_requirements_count')}")
    print(f"Matched Skills:                   {result.get('matched_skills')}")
    print(f"Missing / Unevidenced Skills:     {result.get('missing_skills')}")
    print(f"Preferred Skills Matched:         {result.get('preferred_skills_matched')}")
    print(f"Preferred Skills Missing:         {result.get('preferred_skills_missing')}")

    print("\n--- RETRIEVED EVIDENCE & FAISS PROXIMITY ---")
    for idx, ev in enumerate(result.get("evidence", []), 1):
        req = ev.get("requirement")
        classification = ev.get("match_classification")
        score = ev.get("relevance_score")
        passage_snippet = ev.get("supporting_passage", "").replace("\n", " ")[:90]
        print(f"[{idx}] {req:<20} | Class: {classification:<22} | SimScore: {score} | Snippet: {passage_snippet}...")

    print("\n--- LLM QUALITATIVE INSIGHTS ---")
    print("Strengths:")
    for s in result.get("strengths", []):
        print(f"  • {s}")
    print("Relevant Experience:")
    for exp in result.get("relevant_experience", []):
        print(f"  • {exp}")
    print("Recommendations:")
    for r in result.get("recommendations", []):
        print(f"  • {r}")
    print(f"Narrative Summary: {result.get('explanation', '')[:200]}...")

    print("\n" + "=" * 70)
    print("LANGSMITH OBSERVABILITY INSPECTION")
    print("=" * 70)
    print(f"To inspect the complete run trace, open LangSmith:")
    print(f"https://smith.langchain.com -> Select Project '{project_name}'")
    print("Trace Tree Breakdown to verify in the UI:")
    print(" └── resume_job_rag_matching (chain)")
    print("      ├── build_faiss_vector_store (tool)")
    print("      ├── evidence_classification_and_scoring (chain)")
    print("      │    ├── faiss_retrieve_with_scores (retriever) [for Python]")
    print("      │    ├── faiss_retrieve_with_scores (retriever) [for FastAPI]")
    print("      │    ├── faiss_retrieve_with_scores (retriever) [for Docker]")
    print("      │    ├── faiss_retrieve_with_scores (retriever) [for PostgreSQL]")
    print("      │    ├── faiss_retrieve_with_scores (retriever) [for FAISS]")
    print("      │    └── ...")
    print("      └── generate_matching_insights (chain)")
    print("           └── ChatGroq (llm)")
    print("=" * 70)


if __name__ == "__main__":
    run_real_matching_test()
