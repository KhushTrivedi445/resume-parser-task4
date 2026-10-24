# Resume Parser & RAG-Based Job Matching System

An enterprise-grade, privacy-first FastAPI application designed to parse candidate resumes from PDF documents into structured data and perform explainable, evidence-backed matching against job descriptions using Retrieval-Augmented Generation (RAG) and deterministic scoring algorithms.

Built for the **CODTECH AI/ML Internship**.

---

## 1. Project Overview

The **Resume Parser & RAG-Based Job Matching System** solves a critical problem in modern hiring: balancing candidate discovery speed with explainability, fairness, and strict data privacy.

Unlike traditional keyword-counting tools or pure black-box LLM screening systems, this system combines:
1. **Deterministic Information Extraction:** Uses structural layout heuristics and boundary-safe lexical analysis to extract contact details, skills, and educational qualifications without fabricating personal data.
2. **Retrieval-Augmented Generation (RAG):** Uses local transformer sentence embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and isolated in-memory FAISS vector indexing to retrieve verifiable resume passages as evidence for job requirements.
3. **Deterministic & Explainable Scoring:** Decouples the numerical evaluation from LLM hallucination. Scores are computed using Python mathematical formulas with transparent weighting between required and preferred skills.
4. **Evidence-Based LLM Insights:** Uses Groq (`openai/gpt-oss-20b`) to synthesize qualitative candidate strengths, relevant experiences, and tailored recommendations strictly anchored in retrieved passages, accompanied by fallback logic if offline.

---

## 2. Features

- **Robust PDF Resume Parsing:**
  - Extracts Name, Email, Phone Number, Skills, Degree, Institution, and Graduation Year.
  - Handles missing or partially specified fields gracefully (returning `null` or empty lists rather than hallucinating details).
  - Validates uploads: file existence, `.pdf` extension, MIME type, file size limit (5 MB default), and magic PDF byte signatures (`%PDF-`).
  - Detects empty and unreadable/scanned PDFs.
- **RAG-Driven Job Matching:**
  - Ingests a resume PDF (via multipart upload) or text alongside a target job description.
  - Automatically classifies job requirements into **Required Skills** vs. **Preferred/Bonus Skills**.
  - Maps common abbreviations and variants using an extensible alias taxonomy (e.g., `sklearn` $\to$ `Scikit-learn`, `postgres` $\to$ `PostgreSQL`, `k8s` $\to$ `Kubernetes`, `py` $\to$ `Python`).
  - Enforces boundary-safe token matching, eliminating false substring matches (e.g., preventing language `C` from triggering on `CSS` or `Machine`).
- **Mathematical Scoring Engine:**
  - Completely deterministic: identical inputs yield identical scores across runs.
  - Uses an 80/20 weighted formula when both required and preferred skills exist.
  - Returns explicit unweighted baseline coverage metrics alongside weighted results.
  - Emits `null` score with informative notes if the job description lacks identifiable skills, preventing misleading 0% scores.
- **Explainable Evidence Tracing:**
  - Retrieves supporting resume passages for each requirement with normalized relevance scores.
  - Categorizes evidence items into `explicit_match`, `semantically_supported`, or `unevidenced`.
- **Stateless & Privacy-First:**
  - Resumes and vector embeddings exist purely in-memory during request execution and are discarded immediately afterward.
  - No vector store persistence or cross-tenant data leakage.

---

## 3. Architecture & Data Flow

```
                      +-----------------------------+
                      |   Client Request (PDF + JD) |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Upload & Signature Check  |
                      |   (400, 413, 415 Validation)|
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    PDF Text Extraction      |
                      |          (pypdf)            |
                      +--------------+--------------+
                                     |
                     +---------------+---------------+
                     |                               |
                     v                               v
       +----------------------------+  +----------------------------+
       |   Boundary-Safe Skill      |  |  Requirement Classifier    |
       |  Extraction & Normalization|  |  (Required vs. Preferred)  |
       +-------------+--------------+  +-------------+--------------+
                     |                               |
                     +---------------+---------------+
                                     |
                                     v
                      +-----------------------------+
                      | Deterministic Python Scoring|
                      |  (Mathematical Weighting)   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Passage Chunking & FAISS  |
                      |   Vector Similarity Search  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Evidence-Based LLM Call   |
                      |    (Groq gpt-oss-20b)       |
                      |  [Deterministic Fallback]   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  Pydantic MatchingResponse  |
                      +-----------------------------+
```

---

## 4. Technology Stack

- **Framework:** FastAPI (`fastapi>=0.110.0`), Starlette, Uvicorn
- **Language & Runtime:** Python 3.10+ (tested on Python 3.11.9)
- **PDF Extraction:** `pypdf>=4.0.0`
- **Vector Retrieval & RAG:**
  - `sentence-transformers>=2.6.0` (`sentence-transformers/all-MiniLM-L6-v2`)
  - `faiss-cpu>=1.8.0` (In-memory L2 index)
  - `langchain-huggingface>=0.0.3`
  - `langchain>=0.2.0`, `langchain-community>=0.2.0`
- **LLM Engine:**
  - `langchain-groq>=0.1.0`
  - Model: `openai/gpt-oss-20b` (Groq API)
- **Data Validation & Settings:** Pydantic v2, Pydantic-Settings
- **Database & Fixtures:** SQLAlchemy 2.0 (Stateless execution with optional SQLite persistence)
- **Testing:** `pytest>=8.0.0`, `pytest-asyncio`, `httpx`, `reportlab`

---

## 5. Folder Structure

```
resume-parser-task1/
|-- .env                     # Local environment secrets (ignored by git)
|-- .env.example             # Clean environment template with placeholders
|-- .gitignore               # Excludes secrets, caches, and venv
|-- pytest.ini               # Pytest configuration
|-- requirements.txt         # Pinned runtime and testing dependencies
|-- README.md                # Project architecture and technical documentation
|-- app/
|   |-- __init__.py
|   |-- main.py              # FastAPI application factory and route registration
|   |-- api/
|   |   |-- routes/
|   |   |   |-- parser.py    # POST /api/v1/parser/parse
|   |   |   `-- matching.py  # POST /api/v1/matching/match
|   |-- core/
|   |   `-- config.py        # Centralized Settings & environment config
|   |-- db/
|   |   |-- database.py      # SQLAlchemy Base, session generator, and engine
|   |   `-- models.py        # ResumeRecord model schema
|   |-- schemas/
|   |   |-- resume.py        # ResumeResponse and Education Pydantic models
|   |   `-- matching.py      # MatchingResponse, EvidenceItem, and MatchingRequest models
|   |-- services/
|   |   |-- pdf_extractor.py # PDF byte reader with error traps
|   |   |-- resume_parser.py # Contact, skill, and education parser
|   |   `-- rag/
|   |       |-- retriever.py        # FAISS in-memory indexer and passage retriever
|   |       |-- skill_matcher.py    # Skill taxonomy, boundary regex, and deterministic math
|   |       |-- report_generator.py # Groq LLM integration and deterministic fallback
|   |       `-- matching_service.py # End-to-end matching orchestrator
|   `-- utils/
|       `-- file_validation.py      # File size, mime, extension, and signature validation
|-- data/
|   `-- .gitkeep
`-- tests/
    |-- conftest.py                 # Pytest client, in-memory DB, ReportLab PDF, and LLM mocks
    |-- test_file_validation.py     # File validation unit tests
    |-- test_health.py              # Health check endpoint test
    |-- test_resume_parser.py       # Parser service & route integration tests
    |-- test_retriever_service.py   # RAG retriever & FAISS unit tests
    |-- test_skill_matcher.py       # Taxonomy, boundary regex, and math scoring unit tests
    `-- test_matching.py            # Matching route integration tests (PDF & JSON)
```

---

## 6. Prerequisites

- **Operating System:** Windows 10/11, Linux, or macOS.
- **Python:** Python 3.10 or Python 3.11 installed and available in PATH.
- **Groq API Key (Optional for basic tests, required for live LLM insights):** Free API key from [Groq Console](https://console.groq.com).

---

## 7. Virtual Environment Setup (Windows PowerShell)

Open PowerShell in the project directory:

```powershell
# Navigate to the project root
cd C:\Users\Khush\Desktop\CODTECH_TASKS\resume-parser-task1

# Create virtual environment
python -m venv .venv

# Enable script execution if restricted (if required)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

---

## 8. Dependency Installation

With the virtual environment activated:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 9. `.env` Configuration

Copy `.env.example` to create your local `.env`:

```powershell
Copy-Item .env.example .env
```

Populate `.env` with your values:

```env
# Groq API Key for live LLM explanations and recommendations
GROQ_API_KEY=gsk_your_groq_api_key_here

# Hugging Face token (optional for public models)
HUGGINGFACEHUB_API_TOKEN=

# Embedding model identifier
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Maximum PDF upload size in bytes (default: 5 MB)
MAX_FILE_SIZE=5242880

# SQLite database connection URL
DATABASE_URL=sqlite:///./data/resumes.db
```

> **Security Note:** Never commit your `.env` file to version control. The `.gitignore` file is pre-configured to exclude `.env` files.

---

## 10. Starting the FastAPI Server

Launch the development server with live reload:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The server initializes at `http://127.0.0.1:8000`.

### Starting the Streamlit Interactive Web Application

To use the interactive visual interface (with drag-and-drop PDF upload, sample job presets, KPI scorecards, skills breakdown badges, and traceable evidence viewer):

```powershell
streamlit run streamlit_app.py
```

The web dashboard opens automatically in your default browser at `http://localhost:8501`.

---

## 11. Accessing Swagger UI

FastAPI provides interactive OpenAPI documentation:

- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc Documentation:** `http://127.0.0.1:8000/redoc`

The Swagger UI provides an interactive form where you can upload test PDFs and submit job descriptions directly in the browser.

---

## 12. API Endpoint Descriptions & Sample Requests

### A. Health Check
- **Method:** `GET`
- **Path:** `/health`
- **Description:** Verifies service availability.
- **Example Request:**
  ```powershell
  curl -X GET http://127.0.0.1:8000/health
  ```
- **Response:**
  ```json
  {"status": "ok"}
  ```

---

### B. Parse Resume PDF
- **Method:** `POST`
- **Path:** `/api/v1/parser/parse`
- **Consumes:** `multipart/form-data`
- **Parameters:**
  - `file`: Resume PDF file (max 5 MB).
- **Example Request (PowerShell):**
  ```powershell
  curl.exe -X POST "http://127.0.0.1:8000/api/v1/parser/parse" `
    -F "file=@C:\path\to\resume.pdf"
  ```

---

### C. Match Resume PDF to Job Description
- **Method:** `POST`
- **Path:** `/api/v1/matching/match`
- **Consumes:** `multipart/form-data` (or `application/json`)
- **Parameters (Multipart):**
  - `file`: Resume PDF file (max 5 MB).
  - `job_description`: Job description text string.
  - `resume_text` *(optional)*: Alternative raw resume text.
- **Example Request (Multipart):**
  ```powershell
  curl.exe -X POST "http://127.0.0.1:8000/api/v1/matching/match" `
    -F "file=@C:\path\to\resume.pdf" `
    -F "job_description=Seeking Python Backend Engineer with FastAPI and SQL experience. Docker is a plus."
  ```
- **Example Request (JSON Body):**
  ```powershell
  curl.exe -X POST "http://127.0.0.1:8000/api/v1/matching/match" `
    -H "Content-Type: application/json" `
    -d '{
      "resume_text": "Experienced Python backend engineer proficient with FastAPI, PostgreSQL, and REST APIs.",
      "job_description": "Requirements: Python and FastAPI. Preferred: Docker and AWS."
    }'
  ```

---

## 13. Example Successful Response

```json
{
  "match_score": 80.0,
  "score": 80.0,
  "score_explanation": "Deterministic score of 80.00% calculated using weighted formula: 80% weight for required skills (2/2 matched, 100.0%) and 20% weight for preferred skills (0/2 matched, 0.0%). Baseline unweighted skill coverage: 50.00%. Candidate demonstrates strong core engineering competency.",
  "explanation": "Deterministic score of 80.00% calculated using weighted formula: 80% weight for required skills (2/2 matched, 100.0%) and 20% weight for preferred skills (0/2 matched, 0.0%). Baseline unweighted skill coverage: 50.00%. Candidate demonstrates strong core engineering competency.",
  "skill_coverage": 50.0,
  "total_identified_requirements": 4,
  "matched_requirements_count": 2,
  "matched_skills": [
    "FastAPI",
    "Python"
  ],
  "missing_skills": [],
  "preferred_skills_matched": [],
  "preferred_skills_missing": [
    "AWS",
    "Docker"
  ],
  "strengths": [
    "Demonstrated production competency in Python and FastAPI REST services.",
    "Proven background in relational database modeling with PostgreSQL."
  ],
  "relevant_experience": [
    "Built scalable backend microservices with FastAPI and PostgreSQL databases."
  ],
  "recommendations": [
    "Highlight practical containerization experience with Docker.",
    "Add cloud architecture examples using AWS to fulfill preferred qualifications."
  ],
  "evidence": [
    {
      "requirement": "Python",
      "supporting_passage": "Experienced Python backend engineer proficient with FastAPI, PostgreSQL, and REST APIs.",
      "match_classification": "explicit_match",
      "relevance_score": 0.812,
      "source": "resume"
    },
    {
      "requirement": "FastAPI",
      "supporting_passage": "Experienced Python backend engineer proficient with FastAPI, PostgreSQL, and REST APIs.",
      "match_classification": "explicit_match",
      "relevance_score": 0.841,
      "source": "resume"
    },
    {
      "requirement": "Docker",
      "supporting_passage": "Experienced Python backend engineer proficient with FastAPI, PostgreSQL, and REST APIs.",
      "match_classification": "unevidenced",
      "relevance_score": 0.384,
      "source": "resume"
    }
  ],
  "limitations": [
    "Score reflects lexical and semantic keyword matching from supplied resume text.",
    "Unevidenced requirements do not imply candidate incompetence, only absence of written mention in the document.",
    "This evaluation serves as an automated screening aid and not a conclusive hiring decision."
  ]
}
```

---

## 14. Error Responses

The API uses standardized HTTP status codes and RFC 7807 compatible error payloads:

| Scenario | HTTP Status | Response Payload Example |
| :--- | :--- | :--- |
| **Missing Resume** | `422 Unprocessable Content` | `{"detail": "Either a resume PDF ('file') or 'resume_text' must be provided."}` |
| **Empty Job Description** | `422 Unprocessable Content` | `{"detail": "job_description is required and cannot be empty."}` |
| **Non-PDF Upload** | `415 Unsupported Media Type` | `{"detail": "Only PDF files are supported."}` |
| **Empty Uploaded File** | `400 Bad Request` | `{"detail": "The uploaded file is empty."}` |
| **Oversized PDF (>5MB)**| `413 Request Entity Too Large`| `{"detail": "The file exceeds the maximum size of 5 MB."}` |
| **Malformed PDF** | `400 / 422` | `{"detail": "The uploaded file does not have a valid PDF signature."}` |
| **Scanned/Unreadable PDF**| `422 Unprocessable Content`| `{"detail": "No readable text was found in the PDF. The document may be scanned or empty."}` |

Stack traces, internal absolute paths, and secrets are never exposed in error responses.

---

## 15. Running Automated Tests

The repository includes a comprehensive `pytest` test suite covering unit, integration, and security behaviors with mock LLM fixtures (no live API quota consumed):

```powershell
# Run full test suite with verbose output
pytest -v
```

### Test Suite Composition:
- `test_file_validation.py`: Extension check, MIME validation, empty file rejection, and byte signature checks.
- `test_health.py`: Health endpoint status verification.
- `test_resume_parser.py`: PDF text parsing, contact extraction, education normalization, and malformed document handling.
- `test_retriever_service.py`: Text chunking, metadata preservation, empty query rejection, and FAISS scoring.
- `test_skill_matcher.py`: Boundary safety (preventing `C` substring traps), alias resolution, deduplication, required vs. preferred classification, and mathematical scoring consistency.
- `test_matching.py`: Multipart PDF uploads (with ReportLab-generated PDFs), JSON inputs, input validation, and zero-requirement edge cases.

---

## 16. Matching Score Formula

The scoring algorithm is implemented deterministically in Python to guarantee explainability and reproducibility.

### Definitions:
- $N_{req}$: Count of unique identified required skills.
- $M_{req}$: Count of evidenced required skills in resume ($M_{req} \le N_{req}$).
- $N_{pref}$: Count of unique identified preferred/bonus skills.
- $M_{pref}$: Count of evidenced preferred skills in resume ($M_{pref} \le N_{pref}$).
- $N_{total} = N_{req} + N_{pref}$
- $M_{total} = M_{req} + M_{pref}$

### Formulas:

1. **Baseline Unweighted Skill Coverage:**
   $$\text{Coverage} = \left( \frac{M_{total}}{N_{total}} \right) \times 100$$

2. **Weighted Match Score:**
   - **Case 1: Both Required and Preferred Skills Exist ($N_{req} > 0$ and $N_{pref} > 0$):**
     $$\text{ReqCoverage} = \left( \frac{M_{req}}{N_{req}} \right) \times 100$$
     $$\text{PrefCoverage} = \left( \frac{M_{pref}}{N_{pref}} \right) \times 100$$
     $$\text{MatchScore} = \text{round}\left(0.80 \times \text{ReqCoverage} + 0.20 \times \text{PrefCoverage},\ 2\right)$$
   - **Case 2: Required Skills Only ($N_{req} > 0$ and $N_{pref} = 0$):**
     $$\text{MatchScore} = \text{round}\left(\text{ReqCoverage},\ 2\right)$$
   - **Case 3: Preferred Skills Only ($N_{req} = 0$ and $N_{pref} > 0$):**
     $$\text{MatchScore} = \text{round}\left(\text{PrefCoverage},\ 2\right)$$
   - **Case 4: No Identifiable Requirements ($N_{total} = 0$):**
     $$\text{MatchScore} = \text{null}$$
     Returns an explicit explanation: *"No identifiable technical skills or structured requirements were found in the job description to calculate an objective match score."*

---

## 17. RAG & Semantic Retrieval Explanation

The Retrieval-Augmented Generation (RAG) subsystem operates as follows:

1. **Document Chunking:**
   - Raw resume text is split into semantic paragraphs. Long paragraphs exceeding 350 characters are split into sub-chunks with a 50-character overlap using word-boundary preservation.
   - Each chunk retains metadata: `source`, `chunk_id`, `total_chunks`, and `char_length`.
2. **Dense Vector Embeddings:**
   - Embeddings are generated using `sentence-transformers/all-MiniLM-L6-v2` via `HuggingFaceEmbeddings`.
   - The model is loaded once as a cached singleton in memory, avoiding redundant weight downloads and disk reads during requests.
3. **In-Memory FAISS Indexing:**
   - A fresh FAISS L2 vector index is constructed in-memory for each individual request.
   - Resumes are indexed in isolation, guaranteeing zero cross-request contamination.
4. **Targeted Semantic Querying:**
   - For every extracted requirement, FAISS retrieves the closest matching passage.
   - L2 distances are transformed into normalized similarity scores:
     $$\text{Sim} = \frac{1.0}{1.0 + \text{L2\_Distance}}$$
   - Classifications are assigned:
     - `explicit_match`: Skill explicitly present in candidate's parsed skills.
     - `semantically_supported`: Skill not explicitly mentioned, but supporting passage exhibits strong semantic similarity ($\text{Sim} > 0.55$).
     - `unevidenced`: Neither explicit mention nor high-confidence passage found.
5. **LLM Synthesis:**
   - The Groq LLM (`openai/gpt-oss-20b`) receives the deterministic facts, retrieved passages, and untrusted raw text to draft candidate strengths, experience narratives, and actionable suggestions.
   - The LLM **cannot override** the numerical score, matched skills, or missing skills.

---

## 18. Privacy & Security Considerations

- **Stateless Execution:** Uploaded resume PDFs and vector embeddings reside strictly in temporary execution memory and are discarded when the HTTP response completes.
- **No Inadvertent Persistence:** Resumes are never permanently written to disk or third-party vector databases.
- **Prompt Injection Defense:** Resumes and job descriptions are injected into LLM prompts inside designated raw data blocks with strict instructions that treat them as untrusted data, preventing adversarial prompt injections from overriding system instructions.
- **Zero Secret Leakage:**
  - Secrets are managed exclusively via environment variables (`Settings`).
  - `.env` is ignored by `.gitignore`.
  - `.env.example` contains only empty placeholders.
  - API errors and logs never print API keys, tokens, or personal identifiers.
- **Upload Safety:** Comprehensive file checks reject non-PDF MIME types, mismatched extensions, files exceeding 5 MB, and payloads lacking valid `%PDF-` binary magic headers.

---

## 19. Known Limitations

- **Scanned Resumes:** Does not currently include an OCR engine (e.g. Tesseract). Image-only or scanned PDFs without text layers will be rejected with an informative error message.
- **Skill Taxonomy Breadth:** Skill extraction relies on an extensible curated taxonomy of ~50 core programming languages, frameworks, cloud platforms, and tools. Uncatalogued niche technologies may rely primarily on semantic vector retrieval.
- **Single-Page Form Factor:** Very long multi-page CVs (>15 pages) may be truncated for the LLM qualitative summary context window (2,000 characters), although vector indexing indexes the entire text.

---

## 20. Future Improvements

1. **OCR Integration:** Support scanned and image-based PDFs using Tesseract OCR or EasyOCR with fallback detection.
2. **Dynamic Skill Expansion:** Integrate an open-source NER model (e.g., fine-tuned spaCy or GLiNER) to identify emerging technologies outside the built-in taxonomy.
3. **Downloadable PDF Matching Reports:** Add an export endpoint (`GET /api/v1/matching/report/pdf`) generating a styled evaluation PDF for recruiters using ReportLab.
4. **Batch Matching:** Add an asynchronous Celery/Redis queue for ranking a pool of 500+ resumes against a single job description concurrently.
