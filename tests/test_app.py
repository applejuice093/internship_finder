"""End-to-end smoke tests that drive app.py with Streamlit's AppTest."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tests.test_resume_reading import make_docx

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def app(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)  # force offline mode
    monkeypatch.delenv("INTERNSHIP_DB_URL", raising=False)
    return AppTest.from_file(APP, default_timeout=30).run()


def start_analysis(app, filename="cv.docx", content=None):
    content = content if content is not None else make_docx("Python, SQL and AWS")
    app.file_uploader[0].upload(filename, content).run()
    app.button[0].click().run()
    return app


def test_upload_page_renders(app):
    assert not app.exception
    assert app.file_uploader[0].allowed_type == [".pdf", ".docx"]


def test_offline_notice_survives_rerun(app):
    start_analysis(app)
    assert not app.exception
    assert any("Offline Mode" in w.value for w in app.warning)
    assert app.radio[0].options


def test_full_quiz_flow_reaches_results(app):
    start_analysis(app)
    for _ in range(3):
        radio = app.radio[0]
        radio.set_value(radio.options[0]).run()
        app.button[0].click().run()
    assert not app.exception
    assert app.session_state.app_step == "results"
    assert len(app.metric) == 3


def test_corrupt_resume_shows_error_instead_of_crashing(app):
    start_analysis(app, filename="broken.pdf", content=b"this is not a pdf")
    assert not app.exception
    assert any("Could not read this file" in e.value for e in app.error)
    assert app.session_state.app_step == "upload"


def test_resume_without_text_warns_user(app):
    start_analysis(app, filename="blank.docx", content=make_docx(""))
    assert any("No text could be extracted" in w.value for w in app.warning)


def test_submit_without_answer_does_not_skip_question(app):
    start_analysis(app)
    app.button[0].click().run()  # "Submit Answer" with nothing selected
    assert any("Please select an answer" in w.value for w in app.warning)
    assert app.session_state.quiz_state["current_q"] == 0
