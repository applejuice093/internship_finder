"""Guard the results-page skill badge HTML escaping convention."""
import html


def skill_badges_html(skills):
    """Mirror of the results-page badge builder in app.py."""
    return "".join(
        f"<span class='skill-badge'>{html.escape(skill.upper())}</span>" for skill in skills
    )


def test_skill_badges_escape_html_special_characters():
    html_out = skill_badges_html(["python", "<script>alert(1)</script>", 'a&b"c'])
    assert "<script>" not in html_out
    assert "&lt;SCRIPT&gt;" in html_out
    assert "&amp;" in html_out
    assert "PYTHON" in html_out
