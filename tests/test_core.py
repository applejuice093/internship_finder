import json

import pytest

from core import (
    BUILTIN_INTERNSHIPS,
    OFFLINE_QUESTIONS,
    extract_skills_from_text,
    parse_questions,
    rag_semantic_search,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Skills: Python, SQL and Docker", ["python", "sql", "docker"]),
        ("Skills: C++ and Python", ["python", "c++"]),
        ("Languages: C, Java", ["c", "java"]),
        ("C/C++ developer", ["c++", "c"]),
        ("Objective-C and C# only", []),
        ("GitHub user", ["github"]),
        ("Built APIs with Node.js and React", ["react", "node"]),
    ],
)
def test_extract_skills(text, expected):
    assert extract_skills_from_text(text) == expected


def test_extract_skills_is_case_insensitive_and_unique():
    assert extract_skills_from_text("PYTHON python Python") == ["python"]


def test_parse_questions_strips_markdown_fences():
    payload = [{"q": "Q?", "options": ["A", "B"], "a": "A"}]
    text = "```json\n" + json.dumps(payload) + "\n```"
    assert parse_questions(text) == payload


def test_offline_questions_have_valid_answers():
    for question in OFFLINE_QUESTIONS:
        assert question["a"] in question["options"]


def test_rag_search_ranks_matching_skills_first():
    profile = {"location": "Remote", "education": "Graduate"}
    ranked = rag_semantic_search(profile, ["javascript", "react"], BUILTIN_INTERNSHIPS, top_n=3)
    assert ranked[0]["title"] == "Frontend Trainee"
    assert len(ranked) == 3
    scores = [job["score"] for job in ranked]
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= s <= 99 for s in scores)


def test_rag_search_does_not_mutate_database():
    profile = {"location": "Remote", "education": "Graduate"}
    rag_semantic_search(profile, ["python"], BUILTIN_INTERNSHIPS)
    assert all("score" not in job for job in BUILTIN_INTERNSHIPS)
