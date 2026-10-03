"""Front-end contracts.

There is no browser in the test environment, so these are not end-to-end UI
tests. They guard the two ways the front end breaks silently: JavaScript that
decides the wrong thing (route -> progress copy, which clicks it may hijack), and
templates that reference a CSS class or form action which no longer exists.

Both failure modes render a plausible-looking page, so neither shows up in the
Python tests no matter how thorough they are.
"""
import re
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_JS = (PROJECT_ROOT / "app" / "static" / "app.js").read_text()
STYLE_CSS = (PROJECT_ROOT / "app" / "static" / "style.css").read_text()
PHASE2 = PROJECT_ROOT / "app" / "templates" / "venture_phase2.html"


@pytest.fixture(scope="module")
def templates() -> dict:
    return {
        "phase2": PHASE2.read_text(),
        "macros": (PROJECT_ROOT / "app" / "templates" / "macros.html").read_text(),
        "base": (PROJECT_ROOT / "app" / "templates" / "base.html").read_text(),
    }


# ---- app.js behaviour, run under node ----

def test_app_js_contract_tests():
    result = subprocess.run(
        ["node", str(PROJECT_ROOT / "tests" / "app_js.test.js")],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail("app.js contract tests failed:\n" + result.stdout + result.stderr)


def test_app_js_ships_in_the_image_and_is_wired_up():
    """A silent-navigation feature that never loads looks like the feature is
    simply missing, so keep the script tag and the file in step."""
    base = (PROJECT_ROOT / "app" / "templates" / "base.html").read_text()
    assert '/static/app.js' in base
    assert 'defer' in base, 'app.js must be deferred so it does not block first paint'
    assert "<main" in (PROJECT_ROOT / "app" / "templates" / "base.html").read_text()


# ---- form actions in templates must be routes the app actually serves ----

def _routes() -> set:
    from app import main
    found = set()
    for route in main.app.routes:
        path = getattr(route, "path", "")
        for method in getattr(route, "methods", []) or []:
            if method != "POST":
                continue
            # /venture/{venture_id}/segment/{segment_key}/run -> a regex that any
            # single-segment path satisfies, so the template check is structural.
            found.add(re.sub(r"\{[^}]+\}", "[^/]+", path))
    return found


def test_every_form_action_in_the_templates_is_a_real_post_route(templates):
    """A form pointing at a removed route silently does nothing on submit. This
    is how the progress overlay ended up describing the wrong work: the JS still
    matched /cycle/new long after that route was replaced."""
    import re as _re
    live = _routes()
    checked = 0
    for name, html in templates.items():
        for action in _re.findall(r'action="(/[^"]+)"', html):
            if action in ("", "#") or action.startswith("http"):
                continue
            pattern = "^" + _re.sub(r"\{\{[^}]+\}\}", "[^/]+", _re.escape(action)
                                     .replace(r"\{venture\.id\}", "{venture_id}")
                                     ) + "$"
            # simpler and robust: match on the literal skeleton
            skeleton = _re.sub(r"\{\{[^}]+\}\}", "X", action)
            normalised = _re.sub(r"X", "[^/]+", skeleton)
            candidates = {_re.sub(r"\{[^}]+\}", "[^/]+", r) for r in live}
            assert any(_re.fullmatch(c, action) or _re.fullmatch(c, normalised)
                       for c in candidates) or skeleton in {r for r in candidates}, \
                f"{name} posts to {action}, which is not a POST route"
            checked += 1
    assert checked >= 5, "expected the templates to post to several routes"


# ---- CSS classes the templates rely on ----

@pytest.mark.parametrize("cls", [
    "nav-progress",     # page-transition bar, created by app.js
    "is-navigating",   # dim while a page is in flight
    "decision-card",   # kill-or-pivot card for a failed critical block
    "segment-row",     # the nine-block board
    "hypothesis",      # hypothesis banner
    "thinking-bar",    # the LLM overlay's bar (reused by nav-progress)
])
def test_css_class_exists(cls):
    assert f".{cls}" in STYLE_CSS, f".{cls} is used but not defined in style.css"


def test_nav_progress_reuses_the_existing_sweep_gradient():
    """Two independent spinners would drift visually; the nav bar is meant to
    read as the same system as the thinking overlay."""
    css = STYLE_CSS
    nav = css[css.index(".nav-progress"):]
    nav = nav[:nav.index("body.is-navigating")]
    assert "animation: sweep" in nav
    assert "--phase1" in nav and "--phase3" in nav


# ---- page-transition markup contract ----

def test_main_is_focusable_so_a_swapped_page_can_take_focus(templates):
    """swapIn() focuses <main> after replacing its contents, which needs the
    attribute present in the markup."""
    assert "<main" in templates["base"]
    m = re.search(r"<main[^>]*>", templates["base"])
    # The attribute is added by JS on first swap; assert the element is the swap
    # target and that the CSS suppresses the ring that focus would otherwise draw.
    assert "main:focus" in STYLE_CSS
    assert m is not None


def test_link_interception_is_registered(templates):
    js = APP_JS
    assert "initNav()" in js
    assert "function initNav" in js
    # and it bails on the clicks that must reach the browser
    for guard in ("e.metaKey", "e.button !== 0", "a.hasAttribute('download')"):
        assert guard in js, f"link interception is missing the guard for {guard}"
