import re
from typing import Dict, List, Set, Tuple

# Comprehensive taxonomy mapping Canonical Skill Name -> List of aliases/variants
SKILL_TAXONOMY: Dict[str, List[str]] = {
    # Programming Languages
    "Python": ["python", "python3", "py"],
    "Java": ["java"],
    "C++": ["c++", "cpp"],
    "C#": ["c#", "csharp", "c-sharp"],
    "C": ["c"],
    "JavaScript": ["javascript", "js", "es6"],
    "TypeScript": ["typescript", "ts"],
    "Go": ["golang", "go"],
    "Rust": ["rust"],
    "SQL": ["sql"],
    "HTML": ["html", "html5"],
    "CSS": ["css", "css3"],
    "Bash": ["bash", "shell scripting", "shell"],
    # Web & Backend Frameworks
    "FastAPI": ["fastapi", "fast-api", "fast api"],
    "Flask": ["flask"],
    "Django": ["django"],
    "React": ["react", "react.js", "reactjs"],
    "Node.js": ["node.js", "nodejs", "node"],
    "REST APIs": ["rest api", "rest apis", "restful api", "restful apis", "restful", "rest"],
    "GraphQL": ["graphql"],
    # AI / ML / Data Science
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning", "dl"],
    "Natural Language Processing": ["natural language processing", "nlp"],
    "Computer Vision": ["computer vision", "cv"],
    "Artificial Intelligence": ["artificial intelligence", "ai"],
    "Data Science": ["data science"],
    "Data Analysis": ["data analysis", "data analytics"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "TensorFlow": ["tensorflow", "tf"],
    "PyTorch": ["pytorch", "torch"],
    "LangChain": ["langchain"],
    "LangGraph": ["langgraph"],
    "FAISS": ["faiss"],
    "ChromaDB": ["chromadb", "chroma"],
    "Hugging Face": ["hugging face", "huggingface", "transformers"],
    # Databases
    "PostgreSQL": ["postgresql", "postgres", "psql"],
    "MySQL": ["mysql"],
    "MongoDB": ["mongodb", "mongo"],
    "Redis": ["redis"],
    "SQLite": ["sqlite", "sqlite3"],
    # Cloud, DevOps & Tools
    "Docker": ["docker", "containerization", "containers"],
    "Kubernetes": ["kubernetes", "k8s"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure", "microsoft azure"],
    "GCP": ["gcp", "google cloud", "google cloud platform"],
    "Git": ["git", "github", "gitlab"],
    "CI/CD": ["ci/cd", "ci-cd", "cicd", "continuous integration", "continuous deployment"],
    "Linux": ["linux", "unix"],
}

# Regex compiler for boundary-safe search
def _compile_skill_pattern(alias: str) -> re.Pattern:
    """
    Compile a boundary-safe regular expression pattern for a skill or alias.
    Prevents false substring matches (e.g. 'C' matching 'CSS', 'Cloud', 'Machine').
    """
    alias_lower = alias.lower()
    if alias_lower == "c":
        # Strict boundary for single-letter 'C': not preceded or followed by alphanumeric or symbol
        return re.compile(r"(?<![A-Za-z0-9#+])c(?![A-Za-z0-9#+])", re.IGNORECASE)
    if alias_lower in ("c++", "cpp"):
        return re.compile(r"(?<![A-Za-z0-9])(?:c\+\+|cpp)(?![A-Za-z0-9])", re.IGNORECASE)
    if alias_lower in ("c#", "csharp", "c-sharp"):
        return re.compile(r"(?<![A-Za-z0-9])(?:c#|csharp|c-sharp)(?![A-Za-z0-9])", re.IGNORECASE)
    if alias_lower in ("go", "ai", "ml", "dl", "cv", "ts", "js", "py", "tf"):
        # Short 2-letter tokens require strict word boundaries
        return re.compile(r"\b" + re.escape(alias_lower) + r"\b", re.IGNORECASE)

    # General word boundary pattern
    escaped = re.escape(alias_lower)
    return re.compile(r"(?<![A-Za-z0-9])" + escaped + r"(?![A-Za-z0-9])", re.IGNORECASE)


# Precompile all taxonomy patterns: canonical_name -> list of compiled patterns
COMPILED_PATTERNS: Dict[str, List[re.Pattern]] = {
    canonical: [_compile_skill_pattern(alias) for alias in aliases]
    for canonical, aliases in SKILL_TAXONOMY.items()
}


def extract_skills_from_text(text: str) -> List[str]:
    """
    Extract canonical skills from text using boundary-safe regex matching.
    Deduplicates and returns matching canonical skill names in stable order.
    """
    if not text or not text.strip():
        return []

    normalized_text = " " + text + " "
    found_skills: List[str] = []

    for canonical, patterns in COMPILED_PATTERNS.items():
        matched = False
        for pattern in patterns:
            if pattern.search(normalized_text):
                matched = True
                break
        if matched and canonical not in found_skills:
            found_skills.append(canonical)

    return found_skills


PREFERRED_INDICATORS = [
    "preferred", "plus", "nice to have", "bonus", "optional",
    "good to have", "desired", "ideal", "advantage", "beneficial",
    "a plus", "is a plus", "would be a plus"
]

REQUIRED_INDICATORS = [
    "required", "must have", "must", "essential", "minimum",
    "mandatory", "qualifications", "responsibilities", "requirements",
    "core requirements", "basic qualifications"
]


def classify_jd_requirements(jd_text: str) -> Tuple[List[str], List[str]]:
    """
    Parse a Job Description to identify required vs preferred skills.
    Splits text into sentences/bullet points and classifies based on context clues.

    Returns:
        (required_skills, preferred_skills)
    """
    if not jd_text or not jd_text.strip():
        return [], []

    # Split into sentence/line units
    units = [u.strip() for u in re.split(r"[\n\r;.]+", jd_text) if u.strip()]

    preferred_skills: Set[str] = set()
    required_skills: Set[str] = set()
    general_skills: Set[str] = set()

    for unit in units:
        unit_lower = unit.lower()
        skills_in_unit = extract_skills_from_text(unit)
        if not skills_in_unit:
            continue

        is_preferred = any(p in unit_lower for p in PREFERRED_INDICATORS)
        is_required = any(r in unit_lower for r in REQUIRED_INDICATORS)

        if is_preferred and not is_required:
            for s in skills_in_unit:
                preferred_skills.add(s)
        elif is_required and not is_preferred:
            for s in skills_in_unit:
                required_skills.add(s)
        else:
            for s in skills_in_unit:
                general_skills.add(s)

    # Resolve overlaps: if a skill is identified as both required and preferred, required takes precedence
    preferred_skills = preferred_skills - required_skills

    # Any skill in general context defaults to required unless explicitly preferred
    for s in general_skills:
        if s not in preferred_skills:
            required_skills.add(s)

    # If all skills ended up classified, sort them for stable output
    return sorted(list(required_skills)), sorted(list(preferred_skills))


def calculate_deterministic_score(
    required_skills: List[str],
    preferred_skills: List[str],
    resume_skills: List[str],
) -> Dict:
    """
    Calculate an explainable, deterministic matching score in Python.

    Formulas:
    - total_requirements = len(required_skills) + len(preferred_skills)
    - If total_requirements == 0:
        match_score = None (do not return misleading 0)
    - If both required and preferred skills exist:
        required_coverage = (matched_req / total_req) * 100
        preferred_coverage = (matched_pref / total_pref) * 100
        match_score = round(0.80 * required_coverage + 0.20 * preferred_coverage, 2)
    - If only required skills exist:
        match_score = round((matched_req / total_req) * 100, 2)
    - If only preferred skills exist:
        match_score = round((matched_pref / total_pref) * 100, 2)

    Returns a comprehensive dict of deterministic metrics and explanation.
    """
    resume_set = set(resume_skills)

    matched_required = [s for s in required_skills if s in resume_set]
    missing_required = [s for s in required_skills if s not in resume_set]

    matched_preferred = [s for s in preferred_skills if s in resume_set]
    missing_preferred = [s for s in preferred_skills if s not in resume_set]

    total_req_count = len(required_skills)
    total_pref_count = len(preferred_skills)
    total_identified = total_req_count + total_pref_count

    all_matched = matched_required + matched_preferred
    matched_count = len(all_matched)

    if total_identified == 0:
        return {
            "match_score": None,
            "skill_coverage": None,
            "required_skill_coverage": None,
            "preferred_skill_coverage": None,
            "total_identified_requirements": 0,
            "matched_requirements_count": 0,
            "matched_skills": [],
            "missing_skills": [],
            "preferred_skills_matched": [],
            "preferred_skills_missing": [],
            "score_explanation": (
                "No identifiable technical skills or structured requirements "
                "were found in the job description to calculate an objective match score."
            ),
        }

    # Baseline flat coverage across all identified skills
    flat_coverage = round((matched_count / total_identified) * 100, 2)

    required_coverage = None
    if total_req_count > 0:
        required_coverage = round((len(matched_required) / total_req_count) * 100, 2)

    preferred_coverage = None
    if total_pref_count > 0:
        preferred_coverage = round((len(matched_preferred) / total_pref_count) * 100, 2)

    # Determine final match score
    if total_req_count > 0 and total_pref_count > 0:
        # Weighted formula: 80% required, 20% preferred
        match_score = round(
            0.80 * required_coverage + 0.20 * preferred_coverage, 2
        )
        explanation = (
            f"Deterministic score of {match_score:.2f}% calculated using weighted formula: "
            f"80% weight for required skills ({len(matched_required)}/{total_req_count} matched, "
            f"{required_coverage:.1f}%) and 20% weight for preferred skills "
            f"({len(matched_preferred)}/{total_pref_count} matched, {preferred_coverage:.1f}%). "
            f"Baseline unweighted skill coverage: {flat_coverage:.2f}%."
        )
    elif total_req_count > 0:
        match_score = required_coverage
        explanation = (
            f"Deterministic score of {match_score:.2f}% based on required skill coverage: "
            f"{len(matched_required)} of {total_req_count} required skills matched ({required_coverage:.1f}%)."
        )
    else:
        match_score = preferred_coverage
        explanation = (
            f"Deterministic score of {match_score:.2f}% based on preferred skill coverage: "
            f"{len(matched_preferred)} of {total_pref_count} preferred skills matched ({preferred_coverage:.1f}%)."
        )

    return {
        "match_score": match_score,
        "skill_coverage": flat_coverage,
        "required_skill_coverage": required_coverage,
        "preferred_skill_coverage": preferred_coverage,
        "total_identified_requirements": total_identified,
        "matched_requirements_count": matched_count,
        "matched_skills": matched_required,
        "missing_skills": missing_required,
        "preferred_skills_matched": matched_preferred,
        "preferred_skills_missing": missing_preferred,
        "score_explanation": explanation,
    }
