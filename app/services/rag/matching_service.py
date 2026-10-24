import math
from typing import Dict, List, Tuple
from langchain_core.documents import Document

from app.schemas.matching import EvidenceItem, MatchingResponse
from app.services.rag.retriever import (
    build_vector_store,
    retrieve_passages_with_scores,
    retrieve_relevant_passages,
)
from app.services.rag.skill_matcher import (
    calculate_deterministic_score,
    classify_jd_requirements,
    extract_skills_from_text,
)
from app.services.rag.report_generator import generate_llm_insights


def match_resume_to_job(
    resume_text: str,
    job_description: str,
) -> MatchingResponse:
    """
    Complete end-to-end matching workflow:
    1. Validates inputs.
    2. Extracts and normalizes skills from resume and job description.
    3. Classifies JD requirements into required and preferred skills.
    4. Computes deterministic, explainable match scores.
    5. Indexes resume passages with an isolated in-memory FAISS store.
    6. Retrieves relevant evidence for each requirement with similarity scores.
    7. Uses Groq LLM to synthesize evidence-based strengths, experience, and recommendations.
    8. Validates and returns structured MatchingResponse.
    """
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text cannot be empty.")

    if not job_description or not job_description.strip():
        raise ValueError("Job description cannot be empty.")

    cleaned_resume = resume_text.strip()
    cleaned_jd = job_description.strip()

    # 1. Skill extraction and classification
    resume_skills = extract_skills_from_text(cleaned_resume)
    required_skills, preferred_skills = classify_jd_requirements(cleaned_jd)

    # 2. Deterministic scoring
    scoring_result = calculate_deterministic_score(
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        resume_skills=resume_skills,
    )

    # 3. Build FAISS index for resume text
    vector_store = build_vector_store(cleaned_resume, source="resume")

    # 4. Gather evidence for requirements
    evidence_items: List[EvidenceItem] = []
    all_jd_requirements = required_skills + preferred_skills

    resume_skills_set = set(resume_skills)

    for req in all_jd_requirements:
        # Query FAISS for passages relevant to this skill
        passages_with_scores = retrieve_passages_with_scores(
            vector_store, query=f"Experience and projects with {req}", k=1
        )
        if passages_with_scores:
            doc, l2_distance = passages_with_scores[0]
            # Convert FAISS L2 distance to normalized similarity score [0.0, 1.0]
            # typical L2 distance for normalized MiniLM embeddings is 0 to 2
            similarity = max(0.0, min(1.0, round(1.0 / (1.0 + float(l2_distance)), 3)))
            
            is_explicit = req in resume_skills_set
            classification = (
                "explicit_match"
                if is_explicit
                else ("semantically_supported" if similarity > 0.55 else "unevidenced")
            )

            evidence_items.append(
                EvidenceItem(
                    requirement=req,
                    supporting_passage=doc.page_content.strip(),
                    match_classification=classification,
                    relevance_score=similarity,
                    source="resume",
                )
            )

    # If no structured requirements found, retrieve general passages for JD
    if not evidence_items:
        general_passages = retrieve_passages_with_scores(
            vector_store, query=cleaned_jd[:300], k=3
        )
        for doc, dist in general_passages:
            sim = max(0.0, min(1.0, round(1.0 / (1.0 + float(dist)), 3)))
            evidence_items.append(
                EvidenceItem(
                    requirement="General Role Alignment",
                    supporting_passage=doc.page_content.strip(),
                    match_classification="semantically_supported",
                    relevance_score=sim,
                    source="resume",
                )
            )

    # 5. Generate LLM qualitative insights (or deterministic fallback)
    raw_evidence = [item.model_dump() for item in evidence_items]
    insights = generate_llm_insights(
        resume_text=cleaned_resume,
        job_description=cleaned_jd,
        deterministic_facts=scoring_result,
        retrieved_evidence=raw_evidence,
    )

    # 6. Assemble complete response
    final_explanation = scoring_result["score_explanation"]
    if insights.get("narrative_summary"):
        final_explanation = f"{final_explanation} {insights['narrative_summary']}"

    return MatchingResponse(
        match_score=scoring_result["match_score"],
        score=scoring_result["match_score"],
        score_explanation=final_explanation,
        explanation=final_explanation,
        skill_coverage=scoring_result["skill_coverage"],
        total_identified_requirements=scoring_result["total_identified_requirements"],
        matched_requirements_count=scoring_result["matched_requirements_count"],
        matched_skills=scoring_result["matched_skills"],
        missing_skills=scoring_result["missing_skills"],
        preferred_skills_matched=scoring_result["preferred_skills_matched"],
        preferred_skills_missing=scoring_result["preferred_skills_missing"],
        strengths=insights.get("strengths", []),
        relevant_experience=insights.get("relevant_experience", []),
        recommendations=insights.get("recommendations", []),
        evidence=evidence_items,
        limitations=insights.get("limitations", []),
    )
