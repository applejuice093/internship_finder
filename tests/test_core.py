import json

import pytest

from core import (
    BUILTIN_INTERNSHIPS,
    OFFLINE_QUESTIONS,
    PROFILE_MATCH_BOOST,
    apply_profile_boost,
    extract_skills_from_text,
    parse_questions,
    rag_semantic_search,
    shared_skills,
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
        ("Flask + MongoDB on Linux", ["flask", "mongodb", "linux"]),
        ("Redis cache and Excel reports", ["redis", "excel"]),
        ("HTML, CSS and Keras", ["html", "css", "keras"]),
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


def test_parse_questions_ignores_text_around_the_array():
    text = 'Here are your questions:\n[{"q": "Q?", "options": ["A", "B"], "a": "B"}]\nGood luck!'
    assert parse_questions(text)[0]["a"] == "B"


def test_parse_questions_drops_malformed_questions():
    payload = [
        {"q": "Good?", "options": ["Yes", "No"], "a": "Yes"},
        {"q": "Answer missing from options?", "options": ["A", "B"], "a": "C"},
        {"q": "No options"},
        {"q": "", "options": ["A", "B"], "a": "A"},
        "not a dict",
    ]
    assert [q["q"] for q in parse_questions(json.dumps(payload))] == ["Good?"]


@pytest.mark.parametrize("text", ["", "no json here", "[not json]", "[]", '[{"q": "x"}]'])
def test_parse_questions_raises_when_nothing_usable(text):
    with pytest.raises(ValueError):
        parse_questions(text)


def test_rag_search_returns_empty_list_for_empty_database():
    profile = {"location": "Remote", "education": "Graduate"}
    assert rag_semantic_search(profile, ["python"], [], top_n=3) == []


def test_profile_match_boosts_score():
    """Location and education matches each add PROFILE_MATCH_BOOST, capped at 99."""
    job = {"location": "Remote", "education": "Graduate"}
    profile = {"location": "Remote", "education": "Graduate"}
    other = {"location": "Delhi", "education": "Diploma"}
    assert apply_profile_boost(40, job, profile) == 40 + 2 * PROFILE_MATCH_BOOST
    assert apply_profile_boost(40, job, other) == 40
    assert apply_profile_boost(40, job, {"location": "Remote", "education": "Diploma"}) == 40 + PROFILE_MATCH_BOOST
    assert apply_profile_boost(95, job, profile) == 99  # still capped


def test_shared_skills_preserves_candidate_order_and_dedupes():
    assert shared_skills(["python", "sql", "python", "aws"], ["aws", "python", "docker"]) == ["python", "aws"]
    assert shared_skills(["react"], ["python"]) == []
