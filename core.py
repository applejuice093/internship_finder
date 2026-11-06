"""Pure logic for the internship recommender (no Streamlit imports).

Keeping this separate from app.py means the parsing, skill extraction
and matching code can be unit tested without starting a Streamlit app.
"""
import json
import re
from io import BytesIO
from typing import Any, Optional
from urllib.parse import urlparse

import docx
import pdfplumber
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Skills we look for in a resume, grouped only for readability.
TARGET_SKILLS = {
    "Core_Languages": ["python", "c++", "c", "java", "javascript", "sql", "typescript", "html", "css"],
    "AI_ML": ["pytorch", "tensorflow", "scikit-learn", "pandas", "llm", "keras"],
    "Frameworks": ["fastapi", "flask", "react", "node", "django", "spring"],
    "Data_Stores": ["mongodb", "redis"],
    "Daily_Tooling": ["git", "github", "docker", "aws", "linux", "excel"],
}

# Built-in sample postings used as the internship "database".
BUILTIN_INTERNSHIPS = [
    {"id": 1, "title": "Junior Backend Intern", "company": "TechGram", "location": "Remote", "education": "Graduate", "skills": ["python", "sql", "api"], "description": "Build robust backend APIs using Python and SQL for rural e-governance platforms.", "apply_url": "https://pminternship.mca.gov.in/"},
    {"id": 2, "title": "Frontend Trainee", "company": "JanMitra", "location": "Maharashtra", "education": "Diploma", "skills": ["javascript", "react", "ui"], "description": "Design accessible, user-friendly interfaces using React and JavaScript for mobile devices.", "apply_url": "https://pminternship.mca.gov.in/"},
    {"id": 3, "title": "Data & Cloud Intern", "company": "AgriData Co.", "location": "Remote", "education": "Graduate", "skills": ["python", "aws", "sql"], "description": "Manage massive agricultural crop databases hosted on AWS cloud.", "apply_url": "https://pminternship.mca.gov.in/"},
    {"id": 4, "title": "Flask Web Intern", "company": "SevaSoft", "location": "Delhi", "education": "Graduate", "skills": ["python", "flask", "mongodb", "linux"], "description": "Ship internal citizen-service portals with Flask, MongoDB and Linux deployment scripts.", "apply_url": "https://pminternship.mca.gov.in/"},
    {"id": 5, "title": "UI Content Intern", "company": "GramUI", "location": "Andhra Pradesh", "education": "12th Pass", "skills": ["html", "css", "javascript", "excel"], "description": "Create accessible HTML/CSS landing pages and maintain outreach trackers in Excel.", "apply_url": "https://pminternship.mca.gov.in/"},
    {"id": 6, "title": "Cache & Ops Trainee", "company": "RapidQueue", "location": "Remote", "education": "Diploma", "skills": ["redis", "linux", "docker", "python"], "description": "Operate Redis caches and Dockerized workers that keep rural service queues responsive.", "apply_url": "https://pminternship.mca.gov.in/"},
]

REQUIRED_INTERNSHIP_FIELDS = ("title", "company", "location", "education", "skills", "description", "apply_url")
INTERNSHIP_DB_USER_AGENT = "internship-finder/1.0 (+https://github.com/applejuice093/internship_finder)"
MAX_INTERNSHIP_DB_BYTES = 2 * 1024 * 1024  # 2 MB cap for remote JSON payloads

def _is_http_url(value):
    """True when value is an absolute http(s) URL with a network location."""
    if not isinstance(value, str):
        return False
    parsed = urlparse(value.strip())
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def load_internships(url: Optional[str] = None, timeout: int = 10) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Return internship postings from a JSON URL, or the built-in list.

    url should point at a JSON array shaped like BUILTIN_INTERNSHIPS (for
    example the "Raw" link of a GitHub Gist). Postings missing a required
    field, or whose apply_url is not an absolute http(s) URL, are skipped.
    Returns (postings, error_message); error_message is None on success and
    explains why the built-in list was used otherwise.
    """
    if not url:
        return BUILTIN_INTERNSHIPS, None
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": INTERNSHIP_DB_USER_AGENT},
            stream=True,
        )
        response.raise_for_status()
        # Cap body size so a huge response cannot exhaust memory.
        chunks = []
        total = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_INTERNSHIP_DB_BYTES:
                return (
                    BUILTIN_INTERNSHIPS,
                    f"Internship data at {url} exceeds {MAX_INTERNSHIP_DB_BYTES} bytes.",
                )
            chunks.append(chunk)
        data = json.loads(b"".join(chunks).decode("utf-8"))
    except (requests.RequestException, ValueError, UnicodeDecodeError) as exc:
        return BUILTIN_INTERNSHIPS, f"Could not load internships from {url} ({exc})."
    if not isinstance(data, list):
        return BUILTIN_INTERNSHIPS, f"Internship data at {url} is not a JSON list."
    postings = [
        item for item in data
        if isinstance(item, dict)
        and all(field in item for field in REQUIRED_INTERNSHIP_FIELDS)
        and isinstance(item["skills"], list)
        and _is_http_url(item["apply_url"])
    ]
    if not postings:
        return BUILTIN_INTERNSHIPS, f"No valid internships found at {url}."
    return postings, None


# Questions used when no Gemini API key is configured (offline mode).
OFFLINE_QUESTIONS = [
    {"q": "What is Object-Oriented Programming (OOP)?", "options": ["A programming paradigm", "A hardware component", "A database structure", "An operating system"], "a": "A programming paradigm"},
    {"q": "Which data structure uses a Last-In-First-Out (LIFO) model?", "options": ["Queue", "Array", "Stack", "Linked List"], "a": "Stack"},
    {"q": "What is the primary purpose of version control systems like Git?", "options": ["Compiling code", "Tracking code changes and collaboration", "Designing user interfaces", "Scanning for viruses"], "a": "Tracking code changes and collaboration"},
    {"q": "What does an HTTP 404 status code mean?", "options": ["Server error", "Resource not found", "Unauthorized request", "Redirect"], "a": "Resource not found"},
    {"q": "In a relational database, what does a PRIMARY KEY uniquely identify?", "options": ["A table column type", "A row in a table", "A database user", "A network port"], "a": "A row in a table"},
    {"q": "Which Linux command lists files in the current directory?", "options": ["cd", "pwd", "ls", "rm"], "a": "ls"},
]


def build_question_prompt(skills, num_questions):
    """Prompt asking the model for multiple-choice questions as JSON."""
    return f"""
        You are a technical interviewer. The candidate has these skills: {', '.join(skills)}.
        Generate {num_questions} UNIQUE and DISTINCT multiple-choice interview questions testing different aspects of these skills.
        Ensure no two questions are about the same exact topic.
        Return ONLY a valid JSON array of objects. Do not use markdown blocks.
        Format: [{{"q": "Question text?", "options": ["A", "B", "C", "D"], "a": "Exact text of correct option"}}]
        """


def _is_valid_question(item):
    """A usable question has text, 2+ unique string options and an answer among them."""
    if not isinstance(item, dict):
        return False
    question, options, answer = item.get("q"), item.get("options"), item.get("a")
    return (
        isinstance(question, str)
        and question.strip() != ""
        and isinstance(options, list)
        and len(options) >= 2
        and all(isinstance(option, str) for option in options)
        and len(set(options)) == len(options)
        and answer in options
    )


def parse_questions(response_text: str) -> list[dict[str, Any]]:
    """Turn the model's text reply into a list of valid question dicts.

    Models sometimes wrap the JSON in markdown fences or add a sentence
    before it, and occasionally return a question whose answer is not one
    of its options (which would make it impossible to answer correctly).
    Those questions are dropped. Raises ValueError if nothing usable is left.
    """
    clean = response_text.replace("```json", "").replace("```", "").strip()
    start, end = clean.find("["), clean.rfind("]")
    if start == -1 or end < start:
        raise ValueError("No JSON array found in the model response")
    try:
        items = json.loads(clean[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model response is not valid JSON: {exc}") from exc
    questions = [item for item in items if _is_valid_question(item)]
    if not questions:
        raise ValueError("Model response contained no valid questions")
    return questions


SUPPORTED_RESUME_TYPES = ["pdf", "docx"]
MAX_RESUME_BYTES = 5 * 1024 * 1024  # 5 MB upload cap before parsing


def assert_resume_size_ok(file_bytes, limit=MAX_RESUME_BYTES):
    """Raise ValueError when an upload exceeds the resume size limit."""
    size = len(file_bytes)
    if size > limit:
        raise ValueError(
            f"Resume is too large ({size} bytes). Maximum allowed is {limit} bytes (5 MB)."
        )


def read_resume_text(filename: str, file_bytes: bytes) -> str:
    """Extract plain text from an uploaded PDF or Word (.docx) resume.

    The extension check is case-insensitive, so "Resume.PDF" works too.
    Raises ValueError for any other file type.
    """
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension == "pdf":
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    if extension == "docx":
        document = docx.Document(BytesIO(file_bytes))
        lines = [paragraph.text for paragraph in document.paragraphs]
        # Many resume templates put skills in tables.
        for table in document.tables:
            for row in table.rows:
                lines.extend(cell.text for cell in row.cells)
        return "\n".join(lines)
    raise ValueError(f"Unsupported resume type: {filename!r}")


def _skill_pattern(skill):
    r"""Regex that matches skill as a whole token.

    Plain \b boundaries fail for symbols: r"\bc\+\+\b" never matches
    "C++ " (there is no word boundary after "+"), and r"\bc\b" matches the
    "c" inside "C++", "C#" and "Objective-C". Lookarounds that also treat
    "+", "#" and "-" as part of a token fix both cases.
    """
    return re.compile(r"(?<![\w+#-])" + re.escape(skill) + r"(?![\w+#])")


def extract_skills_from_text(text: str) -> list[str]:
    """Return the TARGET_SKILLS mentioned in text, in TARGET_SKILLS order."""
    found = []
    text_lower = text.lower()
    for skills in TARGET_SKILLS.values():
        for skill in skills:
            if skill not in found and _skill_pattern(skill).search(text_lower):
                found.append(skill)
    return found


def shared_skills(user_skills, internship_skills):
    """Return candidate skills that also appear on the posting (stable order)."""
    posting = {skill.lower() for skill in internship_skills if isinstance(skill, str)}
    seen = set()
    overlap = []
    for skill in user_skills:
        key = skill.lower() if isinstance(skill, str) else skill
        if key in posting and key not in seen:
            seen.add(key)
            overlap.append(skill)
    return overlap


# Soft boost when location or education equals the candidate profile (capped at 99).
PROFILE_MATCH_BOOST = 5


def apply_profile_boost(base_score, internship, user_profile):
    """Add PROFILE_MATCH_BOOST for matching location/education; cap at 99."""
    score = base_score
    if internship.get("location") == user_profile.get("location"):
        score += PROFILE_MATCH_BOOST
    if internship.get("education") == user_profile.get("education"):
        score += PROFILE_MATCH_BOOST
    return min(score, 99)


def rag_semantic_search(user_profile: dict[str, Any], user_skills: list[str], db_jobs: list[dict[str, Any]], top_n: int = 3) -> list[dict[str, Any]]:
    """Rank internships by TF-IDF cosine similarity to the candidate profile.

    Returns an empty list when db_jobs is empty so TfidfVectorizer is never
    fitted on an empty corpus (which raises ValueError). Matching location
    or education each add PROFILE_MATCH_BOOST points (score still capped at 99).
    """
    if not db_jobs:
        return []

    corpus = [f"{j['title']} {j['company']} {j['location']} {' '.join(j['skills'])} {j['description']}" for j in db_jobs]
    query = f"{user_profile['location']} {user_profile['education']} {' '.join(user_skills)}"

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus + [query])
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    ranked = []
    for idx, score in enumerate(cosine_similarities):
        intern_data = db_jobs[idx].copy()
        base = int(score * 100) + 15
        intern_data["score"] = apply_profile_boost(base, intern_data, user_profile)
        ranked.append(intern_data)

    ranked.sort(key=lambda x: x['score'], reverse=True)
    return ranked[:top_n]
