"""
Streamlit Web Application for Resume Parser & RAG-Based Job Matching System
CODTECH Internship Project - Advanced AI/ML & Backend Architecture
"""

import json
from io import BytesIO
from typing import Dict, List, Optional
import streamlit as st

from app.core.config import settings
from app.schemas.matching import MatchingResponse
from app.services.pdf_extractor import extract_text_from_pdf
from app.services.rag.matching_service import match_resume_to_job
from app.services.rag.skill_matcher import (
    calculate_deterministic_score,
    classify_jd_requirements,
    extract_skills_from_text,
)
from app.services.resume_parser import parse_resume_text

# Page Configuration
st.set_page_config(
    page_title="Resume Parser & RAG Job Matcher",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Modern, Premium UI
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2563eb, #7c3aed, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.78rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .badge-matched {
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #bbf7d0;
    }
    .badge-missing {
        background-color: #fee2e2;
        color: #b91c1c;
        border: 1px solid #fecaca;
    }
    .badge-pref-matched {
        background-color: #e0f2fe;
        color: #0369a1;
        border: 1px solid #bae6fd;
    }
    .badge-pref-missing {
        background-color: #fef3c7;
        color: #b45309;
        border: 1px solid #fde68a;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .evidence-box {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.8rem;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sample Job Descriptions for instant 1-click testing
SAMPLE_JDS = {
    "Python Backend Developer": (
        "We are looking for a Python Backend Developer to design and build scalable REST APIs.\n\n"
        "Required Qualifications:\n"
        "- Strong experience with Python, FastAPI, and SQL.\n"
        "- Familiarity with PostgreSQL databases and query optimization.\n"
        "- Solid understanding of Git and version control.\n\n"
        "Preferred / Nice to Have:\n"
        "- Docker and containerization experience is a plus.\n"
        "- Knowledge of AWS cloud infrastructure and CI/CD pipelines.\n"
        "- Experience with Redis caching."
    ),
    "Machine Learning & AI Engineer": (
        "Seeking an AI/ML Engineer to deploy real-world predictive models and RAG applications.\n\n"
        "Requirements:\n"
        "- Proficiency in Python, PyTorch, and Scikit-learn.\n"
        "- Solid background in Machine Learning, Natural Language Processing, and Data Science.\n"
        "- Experience with Pandas, NumPy, and data preprocessing.\n\n"
        "Preferred Qualifications:\n"
        "- Hands-on experience with LangChain and FAISS vector databases is preferred.\n"
        "- Knowledge of Hugging Face transformers and Docker is a bonus."
    ),
    "Fullstack Engineer": (
        "Join our product engineering team as a Fullstack Developer.\n\n"
        "Essential Requirements:\n"
        "- Expertise in React, JavaScript, HTML, and CSS.\n"
        "- Backend experience with Python or Node.js.\n"
        "- Experience consuming and designing REST APIs.\n\n"
        "Bonus Points:\n"
        "- TypeScript knowledge is a plus.\n"
        "- Cloud deployment experience with Docker or Kubernetes is advantageous."
    ),
}

# Sidebar Info & Status
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/resume.png",
        width=70,
    )
    st.markdown("### System Architecture")
    st.markdown(
        """
        - **Parser:** Layout Heuristics (`pypdf`)
        - **Embedding Model:**  
          `sentence-transformers/all-MiniLM-L6-v2`
        - **Vector Index:** FAISS (In-Memory)
        - **Scoring Engine:** Deterministic Python Math (80% Required / 20% Preferred)
        - **LLM Engine:** Groq `openai/gpt-oss-20b`
        """
    )
    st.divider()

    st.markdown("### Privacy & Security")
    st.info(
        "🔒 **Stateless Execution:** Resumes and vector embeddings exist purely in temporary memory during request execution and are discarded immediately afterward."
    )

    st.divider()
    groq_status = "🟢 Configured" if settings.GROQ_API_KEY else "🟡 Offline (Fallback Enabled)"
    st.caption(f"Groq API: {groq_status}")
    st.caption(f"Max Upload Limit: {settings.MAX_FILE_SIZE // (1024 * 1024)} MB")


# Main Header
st.markdown('<div class="main-header">🎯 Resume Parser & RAG-Based Job Matcher</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Explainable, evidence-backed candidate matching powered by Local FAISS Retrieval and Deterministic Mathematical Scoring.</div>',
    unsafe_allow_html=True,
)

# Mode Navigation Tabs
tab_match, tab_parse, tab_docs = st.tabs([
    "🎯 Match Resume to Job",
    "📋 Parse Resume PDF",
    "ℹ️ System & Math Guide",
])


# -------------------------------------------------------------------------------------
# TAB 1: RAG JOB MATCHING
# -------------------------------------------------------------------------------------
with tab_match:
    st.markdown("#### Upload Candidate Resume & Target Job Description")

    col_upload, col_jd = st.columns([1, 1], gap="medium")

    with col_upload:
        st.markdown("**1. Resume Document (PDF)**")
        uploaded_resume_file = st.file_uploader(
            "Upload Resume PDF",
            type=["pdf"],
            help="Select a text-based PDF resume up to 5 MB.",
            key="matcher_resume_pdf",
        )

        alternative_text = st.text_area(
            "Or Paste Resume Text (Optional fallback):",
            placeholder="Paste candidate resume text here if you don't have a PDF...",
            height=130,
        )

    with col_jd:
        st.markdown("**2. Target Job Description**")

        # Quick preset selector
        preset_choice = st.selectbox(
            "Pre-fill with Sample Job Description:",
            ["Custom (Enter your own)"] + list(SAMPLE_JDS.keys()),
        )

        initial_jd_val = ""
        if preset_choice in SAMPLE_JDS:
            initial_jd_val = SAMPLE_JDS[preset_choice]

        job_description_text = st.text_area(
            "Job Description Text:",
            value=initial_jd_val,
            placeholder="Paste target job description and requirements here...",
            height=200,
        )

    match_button = st.button("🚀 Analyze Alignment & Compute Match", type="primary", use_container_width=True)

    if match_button:
        # Validate inputs
        resume_raw_text = ""
        if uploaded_resume_file is not None:
            pdf_bytes = uploaded_resume_file.read()
            if len(pdf_bytes) > settings.MAX_FILE_SIZE:
                st.error("Uploaded file exceeds the maximum 5 MB limit.")
                st.stop()
            try:
                resume_raw_text = extract_text_from_pdf(pdf_bytes)
            except ValueError as exc:
                st.error(f"PDF Extraction Error: {exc}")
                st.stop()
        elif alternative_text.strip():
            resume_raw_text = alternative_text.strip()
        else:
            st.warning("Please upload a resume PDF or paste resume text to begin.")
            st.stop()

        if not job_description_text.strip():
            st.warning("Please provide a target job description.")
            st.stop()

        # Run Matching Workflow with Progress Spinner
        with st.spinner("Indexing passages in FAISS, calculating deterministic score, and generating insights..."):
            try:
                result: MatchingResponse = match_resume_to_job(
                    resume_text=resume_raw_text,
                    job_description=job_description_text,
                )
            except Exception as exc:
                st.error(f"Matching analysis failed: {exc}")
                st.stop()

        st.success("✅ Match analysis complete!")

        # Top KPI Score Row
        score_val = result.match_score
        score_display = f"{score_val:.1f}%" if score_val is not None else "N/A"
        coverage_display = f"{result.skill_coverage:.1f}%" if result.skill_coverage is not None else "N/A"

        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        with kpi_col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-lbl">Deterministic Match Score</div>
                    <div class="metric-val" style="color: {'#16a34a' if score_val and score_val >= 70 else '#2563eb' if score_val and score_val >= 50 else '#ea580c'};">{score_display}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-lbl">Baseline Skill Coverage</div>
                    <div class="metric-val">{coverage_display}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-lbl">Requirements Found</div>
                    <div class="metric-val">{result.total_identified_requirements}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-lbl">Requirements Evidenced</div>
                    <div class="metric-val">{result.matched_requirements_count}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")

        # Score Explanation Callout
        st.info(f"📐 **Score Calculation:** {result.score_explanation}")

        # Skills Breakdown Cards
        st.markdown("### 🧩 Skills & Requirements Breakdown")
        sk_col1, sk_col2 = st.columns(2)

        with sk_col1:
            st.markdown("##### 📌 Required Skills Analysis")
            if result.matched_skills:
                st.markdown("**Evidenced in Resume:**")
                matched_badges = "".join([f'<span class="badge badge-matched">✔ {s}</span>' for s in result.matched_skills])
                st.markdown(matched_badges, unsafe_allow_html=True)
            else:
                st.markdown("_No required skills evidenced in resume._")

            st.markdown("")
            if result.missing_skills:
                st.markdown("**Unevidenced / Missing in Resume:**")
                missing_badges = "".join([f'<span class="badge badge-missing">✖ {s}</span>' for s in result.missing_skills])
                st.markdown(missing_badges, unsafe_allow_html=True)
            else:
                st.markdown("🎉 _All required skills are evidenced in the resume!_")

        with sk_col2:
            st.markdown("##### 🌟 Preferred / Bonus Skills Analysis")
            if result.preferred_skills_matched:
                st.markdown("**Evidenced in Resume:**")
                pref_matched_badges = "".join([f'<span class="badge badge-pref-matched">★ {s}</span>' for s in result.preferred_skills_matched])
                st.markdown(pref_matched_badges, unsafe_allow_html=True)
            else:
                st.markdown("_No preferred skills evidenced in resume._")

            st.markdown("")
            if result.preferred_skills_missing:
                st.markdown("**Unevidenced Preferred Skills:**")
                pref_missing_badges = "".join([f'<span class="badge badge-pref-missing">○ {s}</span>' for s in result.preferred_skills_missing])
                st.markdown(pref_missing_badges, unsafe_allow_html=True)
            else:
                st.markdown("_No preferred skills missing or none specified in job description._")

        st.divider()

        # Evidence-Based Qualitative Insights (Groq LLM / Fallback)
        st.markdown("### 💡 Qualitative Insights & Recommendations")
        rec_col1, rec_col2 = st.columns(2)

        with rec_col1:
            st.markdown("##### 🚀 Demonstrated Strengths")
            for strength in result.strengths:
                st.markdown(f"- {strength}")

            st.markdown("##### 💼 Relevant Experience Highlights")
            for exp in result.relevant_experience:
                st.markdown(f"- {exp}")

        with rec_col2:
            st.markdown("##### 🎯 Actionable Recommendations")
            for rec in result.recommendations:
                st.markdown(f"- {rec}")

            st.markdown("##### ⚖️ Limitations & Fairness Note")
            for lim in result.limitations:
                st.caption(f"• {lim}")

        st.divider()

        # Retrieved Evidence Passages (RAG FAISS traceability)
        st.markdown("### 🔍 Retrieved Evidence & Passage Traceability")
        st.markdown("The RAG subsystem retrieves candidate resume passages supporting each requirement via semantic vector similarity search:")

        for item in result.evidence:
            badge_class = "badge-matched" if item.match_classification == "explicit_match" else "badge-pref-matched" if item.match_classification == "semantically_supported" else "badge-missing"
            with st.expander(f"Requirement: {item.requirement} ({item.match_classification.replace('_', ' ').title()})"):
                score_str = f"Relevance Score: {item.relevance_score:.3f}" if item.relevance_score is not None else ""
                st.markdown(f'<span class="badge {badge_class}">{item.match_classification}</span> {score_str}', unsafe_allow_html=True)
                st.markdown(f"""<div class="evidence-box">"{item.supporting_passage}"</div>""", unsafe_allow_html=True)

        st.divider()

        # Download Report Option
        report_json = result.model_dump_json(indent=2)
        st.download_button(
            label="📥 Download Structured Match Report (JSON)",
            data=report_json,
            file_name="matching_report.json",
            mime="application/json",
            use_container_width=True,
        )


# -------------------------------------------------------------------------------------
# TAB 2: STRUCTURED RESUME PARSER
# -------------------------------------------------------------------------------------
with tab_parse:
    st.markdown("#### Structured Information Extraction from Resume PDF")
    st.write("Extract name, email, phone number, catalogued skills, and degree qualifications without hallucinating missing data.")

    parse_file = st.file_uploader(
        "Upload Resume PDF to Parse",
        type=["pdf"],
        help="Select a text-based PDF resume up to 5 MB.",
        key="standalone_parser_pdf",
    )

    if st.button("📋 Parse Resume", type="primary", key="parse_action_btn"):
        if parse_file is None:
            st.warning("Please upload a PDF file to parse.")
        else:
            file_bytes = parse_file.read()
            if len(file_bytes) > settings.MAX_FILE_SIZE:
                st.error("Uploaded file exceeds 5 MB limit.")
            else:
                with st.spinner("Extracting text and identifying structured entities..."):
                    try:
                        extracted_text = extract_text_from_pdf(file_bytes)
                        parsed_res = parse_resume_text(extracted_text)
                    except ValueError as exc:
                        st.error(f"Parser Error: {exc}")
                        st.stop()

                st.success("Resume parsed successfully!")

                # Details Grid
                pcol1, pcol2 = st.columns([1, 1])

                with pcol1:
                    st.markdown("##### 👤 Candidate Profile")
                    st.markdown(f"**Name:** `{parsed_res.name or 'Not identified'}`")
                    st.markdown(f"**Email:** `{parsed_res.email or 'Not identified'}`")
                    st.markdown(f"**Phone:** `{parsed_res.phone_number or 'Not identified'}`")
                    if parsed_res.institution:
                        st.markdown(f"**Institution:** `{parsed_res.institution}`")
                    if parsed_res.graduation_year:
                        st.markdown(f"**Graduation Year:** `{parsed_res.graduation_year}`")

                with pcol2:
                    st.markdown("##### 🎓 Education Entries")
                    if parsed_res.education:
                        for idx, edu in enumerate(parsed_res.education, start=1):
                            st.markdown(f"**Qualification #{idx}:**")
                            st.write(f"- Degree: {edu.degree or 'N/A'}")
                            st.write(f"- Institution: {edu.institution or 'N/A'}")
                            st.write(f"- Graduation Year: {edu.graduation_year or 'N/A'}")
                    else:
                        st.info("No formal education entries identified.")

                st.markdown("##### 🛠 Identified Skills")
                if parsed_res.skills:
                    badges = "".join([f'<span class="badge badge-matched">{s}</span>' for s in parsed_res.skills])
                    st.markdown(badges, unsafe_allow_html=True)
                else:
                    st.info("No skills from the built-in taxonomy were detected.")

                with st.expander("📄 View Full Extracted Raw Text"):
                    st.text(extracted_text)

                st.download_button(
                    label="📥 Download Parsed Profile (JSON)",
                    data=parsed_res.model_dump_json(indent=2),
                    file_name="parsed_resume.json",
                    mime="application/json",
                )


# -------------------------------------------------------------------------------------
# TAB 3: SYSTEM ARCHITECTURE & MATH GUIDE
# -------------------------------------------------------------------------------------
with tab_docs:
    st.markdown("### 📐 Mathematical Scoring & Architectural Design")

    st.markdown(
        """
        #### 1. Why Deterministic Scoring?
        LLMs are prone to inconsistent grading, evaluation drift, and arbitrary percentage estimates when reviewing resumes.
        In this system:
        - The **score is 100% mathematical and deterministic**: the same resume and job description will produce the exact same score every time.
        - The **LLM does not decide the score**. It is only given the deterministic facts to write helpful qualitative explanations and actionable recommendations.

        #### 2. Weighted Scoring Formula
        Job requirements are automatically classified into **Required Skills** ($N_{req}$) and **Preferred Skills** ($N_{pref}$):
        - **Both Required & Preferred skills exist:**
        """
    )
    st.latex(
        r"\text{MatchScore} = 0.80 \times \left(\frac{M_{req}}{N_{req}} \times 100\right) + 0.20 \times \left(\frac{M_{pref}}{N_{pref}} \times 100\right)"
    )
    st.markdown(
        """
        - **Required skills only ($N_{pref} = 0$):**
        """
    )
    st.latex(
        r"\text{MatchScore} = \left(\frac{M_{req}}{N_{req}}\right) \times 100"
    )
    st.markdown(
        """
        - **No identifiable requirements ($N_{total} = 0$):**
          Returns `null` / `N/A` with an explicit notice instead of a misleading 0%.

        #### 3. RAG Retrieval Workflow
        1. **Paragraph Chunking:** Text is partitioned into cohesive chunks (max 350 chars with 50 chars overlap) preserving chunk metadata.
        2. **Local Vector Embeddings:** Generated locally using `sentence-transformers/all-MiniLM-L6-v2`.
        3. **In-Memory FAISS:** Evaluates cosine/L2 semantic similarity to retrieve verifiable passages for each job requirement.
        4. **Zero-Persistence Privacy:** Everything is cleaned up when the request terminates.
        """
    )
