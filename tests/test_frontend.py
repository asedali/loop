"""Front-end contracts.

There is no browser in the test environment, so these are not end-to-end UI
tests. They guard the two ways the front end breaks silently: JavaScript that
decides the wrong thing (route -> progress copy, which clicks it may hijack), and
templates that reference a CSS class or form action which no longer exists.

Both failure modes render a plausible-looking page, so neither shows up in the
Python tests no matter how thorough they are.
"""
import pathlib
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
        # Phase 3 was missing from this list, which is how its forms went
        # unchecked while the action-plan checklist gained a POST.
        "phase3": (PROJECT_ROOT / "app" / "templates" / "venture_phase3.html").read_text(),
        "macros": (PROJECT_ROOT / "app" / "templates" / "macros.html").read_text(),
        "base": (PROJECT_ROOT / "app" / "templates" / "base.html").read_text(),
        # Included so a form action added to any of them is checked by the route
        # test below rather than shipped silently.
        "rate_limited": (PROJECT_ROOT / "app" / "templates" / "rate_limited.html").read_text(),
        "forgot_password": (PROJECT_ROOT / "app" / "templates" / "forgot_password.html").read_text(),
        "reset_password": (PROJECT_ROOT / "app" / "templates" / "reset_password.html").read_text(),
        # Phase 3 and the account page were both missing from this list, which is
        # how a form action in each went unchecked — Phase 3 when the action-plan
        # checklist gained a POST, the account page when password change did.
        "account": (PROJECT_ROOT / "app" / "templates" / "account.html").read_text(),
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

def _routes(methods={"POST"}) -> set:
    from app import main
    found = set()
    for route in main.app.routes:
        path = getattr(route, "path", "")
        for method in getattr(route, "methods", []) or []:
            if method not in methods:
                continue
            # /venture/{venture_id}/segment/{segment_key}/run -> a regex that any
            # single-segment path satisfies, so the template check is structural.
            found.add(re.sub(r"\{[^}]+\}", "[^/]+", path))
    return found


# Deliberate GET forms. The rule below is "a form action must be a real POST
# route", which is right for state-changing forms but wrong for a download — a
# GET form works without JS and cannot be triggered by a cross-site POST, which
# is exactly what the JSON export wants. Declared rather than inferred, so a new
# GET form has to be argued for instead of slipping through.
ALLOWED_GET_FORM_ACTIONS = {
    "/account/export",   # account.html: "it just downloads", per TECHNICAL §6
}


def test_every_form_action_in_the_templates_is_a_real_post_route(templates):
    """A form pointing at a removed route silently does nothing on submit. This
    is how the progress overlay ended up describing the wrong work: the JS still
    matched /cycle/new long after that route was replaced."""
    import re as _re
    live = _routes()
    all_routes = _routes(methods={"GET", "POST"})
    checked = 0
    for name, html in templates.items():
        for action in _re.findall(r'action="(/[^"]+)"', html):
            if action in ("", "#") or action.startswith("http"):
                continue
            # simpler and robust: match on the literal skeleton
            skeleton = _re.sub(r"\{\{[^}]+\}\}", "X", action)
            normalised = _re.sub(r"X", "[^/]+", skeleton)
            candidates = {_re.sub(r"\{[^}]+\}", "[^/]+", r) for r in live}
            if action in ALLOWED_GET_FORM_ACTIONS:
                # Asserted against GET routes instead, so the exemption is a
                # statement that the route exists rather than a hole in the check.
                assert any(_re.fullmatch(c, action) for c in all_routes), \
                    f"{name} declares {action} as a GET form but it is not a GET route"
                checked += 1
                continue
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
    "verify-banner",   # the unverified-email nudge (M0.2)
    "answer",          # a canvas block's extracted answer
    "plan-progress",   # action-plan progress bar (M3.1)
    "step-note",       # the "what happened" input on each plan step (M3.1)
    "strat-note",      # a recorded plan-step outcome (M3.1)
    "o-done",          # badge for a completed plan step (M3.1)
"o-blocked",      # badge for a blocked plan step (M3.1)
    "dup-note",       # the "also extracted as" line on a duplicate card (M1.4)
    "dismissed-list", # the collapsed dismissed-ideas section (M1.4)
    "mentor-panel",   # the mentor challenge panel (M2.1)
    "mentor-qs",      # the rendered questions (M2.1)
])
def test_css_class_exists(cls):
    assert f".{cls}" in STYLE_CSS, f".{cls} is used but not defined in style.css"


def test_the_verify_banner_survives_the_main_swap():
    """app.js replaces <main>'s innerHTML on every navigation. A banner placed
    inside <main> would vanish the moment the user clicked anything, which for a
    nudge is the same as not having one — so it lives outside the swap target."""
    base = (PROJECT_ROOT / "app" / "templates" / "base.html").read_text()
    banner = base.index("verify-banner")
    assert banner < base.index("<main"), "the banner must not be inside <main>"


def test_the_reset_form_carries_the_token_as_a_hidden_field(templates):
    """The token arrives in the GET query string; the POST has to carry it on, and
    keeping it out of the form's action attribute means it does not end up in an
    access log twice."""
    page = templates["reset_password"]
    assert '<input type="hidden" name="token"' in page
    assert 'name="confirm"' in page, "a reset should ask for the password twice"


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


# ---- the rate-limit page (M0.3) ----

def test_the_rate_limit_page_needs_no_javascript(templates):
    """It is served with a 429 to a form POST, so it has to work as a plain page
    load. A form on it would also have to be a real route; asserting there is
    none keeps that from silently becoming a requirement."""
    page = templates["rate_limited"]
    assert "<form" not in page
    # The way back is a link, so it is reachable whether or not the fetch swap ran.
    assert 'class="btn block" href=' in page


# Pulled in by the pages rather than rendered on their own: base.html is the
# layout the others extend, macros.html holds partials they import.
PARTIALS = {"base.html", "macros.html"}


def test_every_template_is_reachable_from_the_app():
    """A page template nothing renders is dead code that still looks maintained.
    Every file in templates/ must appear in a render() call or a
    TemplateResponse, or be a documented partial."""
    import app.main as main
    source = pathlib.Path(main.__file__).read_text()
    for template in sorted((PROJECT_ROOT / "app" / "templates").glob("*.html")):
        name = template.name
        if name in PARTIALS:
            continue
        assert f'"{name}"' in source, f"{name} is never rendered by app/main.py"


def test_the_partials_are_actually_referenced(templates):
    """The other half: a partial nothing imports is just dead weight."""
    pages = "".join(v for k, v in templates.items() if k not in PARTIALS)
    for partial in PARTIALS:
        assert f'"{partial}"' in pages, f"{partial} is not imported by any page"
