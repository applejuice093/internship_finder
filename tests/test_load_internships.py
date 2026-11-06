import requests

import core
from core import BUILTIN_INTERNSHIPS, load_internships


class FakeResponse:
    def __init__(self, payload, status=200, raw_bytes=None):
        self.payload, self.status = payload, status
        if raw_bytes is not None:
            self._raw = raw_bytes
        elif isinstance(payload, Exception):
            self._raw = b""
        else:
            import json as _json
            self._raw = _json.dumps(payload).encode("utf-8")

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"{self.status} error")

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload

    def iter_content(self, chunk_size=64 * 1024):
        if isinstance(self.payload, Exception):
            raise self.payload
        data = self._raw
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]


def fake_get(payload, status=200, raw_bytes=None):
    def get(url, timeout, headers=None, stream=False):
        assert timeout > 0  # never hang forever on a slow server
        assert headers and "User-Agent" in headers
        assert "internship-finder" in headers["User-Agent"]
        assert stream is True
        return FakeResponse(payload, status, raw_bytes=raw_bytes)
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


def test_skips_postings_with_non_http_apply_url(monkeypatch):
    good = dict(BUILTIN_INTERNSHIPS[0], title="Good Apply URL")
    bad_scheme = dict(good, title="Javascript URL", apply_url="javascript:alert(1)")
    missing_host = dict(good, title="Relative URL", apply_url="/apply")
    ftp = dict(good, title="FTP URL", apply_url="ftp://example.com/job")
    monkeypatch.setattr(
        core.requests, "get", fake_get([good, bad_scheme, missing_host, ftp, {"title": "incomplete"}])
    )
    postings, error = load_internships("https://example.com/db.json")
    assert error is None
    assert postings == [good]


def test_all_bad_apply_urls_falls_back(monkeypatch):
    bad = dict(BUILTIN_INTERNSHIPS[0], apply_url="not-a-url")
    monkeypatch.setattr(core.requests, "get", fake_get([bad]))
    postings, error = load_internships("https://example.com/db.json")
    assert postings is BUILTIN_INTERNSHIPS
    assert "No valid internships" in error


def test_rejects_oversized_remote_payload(monkeypatch):
    huge = b"[" + (b"0," * (core.MAX_INTERNSHIP_DB_BYTES // 2)) + b"0]"
    monkeypatch.setattr(core.requests, "get", fake_get([], raw_bytes=huge))
    postings, error = load_internships("https://example.com/huge.json")
    assert postings is BUILTIN_INTERNSHIPS
    assert "exceeds" in error
