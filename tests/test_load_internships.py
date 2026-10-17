import requests

import core
from core import BUILTIN_INTERNSHIPS, load_internships


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status = payload, status

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"{self.status} error")

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def fake_get(payload, status=200):
    def get(url, timeout):
        assert timeout > 0  # never hang forever on a slow server
        return FakeResponse(payload, status)
    return get


def test_without_url_uses_builtin_list():
    assert load_internships(None) == (BUILTIN_INTERNSHIPS, None)


def test_loads_valid_postings_and_skips_incomplete_ones(monkeypatch):
    good = dict(BUILTIN_INTERNSHIPS[0], title="Remote ML Intern")
    monkeypatch.setattr(core.requests, "get", fake_get([good, {"title": "incomplete"}]))
    postings, error = load_internships("https://example.com/db.json")
    assert error is None
    assert postings == [good]


def test_http_error_falls_back(monkeypatch):
    monkeypatch.setattr(core.requests, "get", fake_get([], status=404))
    postings, error = load_internships("https://example.com/missing.json")
    assert postings is BUILTIN_INTERNSHIPS
    assert "404" in error


def test_invalid_json_falls_back(monkeypatch):
    monkeypatch.setattr(core.requests, "get", fake_get(ValueError("bad json")))
    postings, error = load_internships("https://example.com/db.json")
    assert postings is BUILTIN_INTERNSHIPS
    assert error


def test_non_list_payload_falls_back(monkeypatch):
    monkeypatch.setattr(core.requests, "get", fake_get({"jobs": []}))
    postings, error = load_internships("https://example.com/db.json")
    assert postings is BUILTIN_INTERNSHIPS
    assert "not a JSON list" in error
