# Resume Parser and Matcher

A small FastAPI application that extracts contact details, skills, education,
and experience from text-based PDF resumes, stores parsed results in SQLite,
and compares resume text with a job description. Its retrieval components use
a deterministic local feature-hash embedding and extractive snippets; no
external AI service or API key is required.

## Requirements

- Python 3.10 or newer

## Run locally

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`; interactive documentation is
at `/docs`. The health endpoint is `GET /health`.

### Parse a resume

Send a text-based PDF to `POST /api/v1/parser/parse` as multipart form data
with the field name `file`. Uploads are limited to 10 MB. The API returns
extracted resume fields and stores them in the configured database.

### Match a resume

Send JSON to `POST /api/v1/matching/match`:

```json
{
  "resume_text": "Python, FastAPI, and SQL experience",
  "job_description": "Looking for a Python developer with FastAPI and AWS."
}
```

The skill score is the percentage of catalogued job skills found in the
resume. The parser and matcher use a deliberately small built-in skill
catalogue; this is a starter implementation, not a substitute for human
review or a production resume-understanding model. Scanned/image-only PDFs
require OCR and are not supported.

## Run tests

```powershell
pytest
```

## Project layout

- `app/api/routes/`: health, PDF parsing, and matching endpoints
- `app/core/`: application settings
- `app/schemas/`: API request/response models
- `app/services/`: PDF extraction, parsing, matching, and retrieval helpers
- `app/db/`: SQLAlchemy engine, session dependency, and resume model
- `app/utils/`: upload validation
- `tests/`: API and service tests
