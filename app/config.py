"""Every environment-configurable value, in one place.

Two kinds of setting live here, and the split matters:

  * **Boot settings** (DEBUG, SECRET_KEY, SESSION_HTTPS_ONLY) are read once at
    import. They configure the session middleware, so they have to be known
    before the first request, and a missing SECRET_KEY must fail fast at boot
    rather than on first use.

  * **Runtime settings** are read per call through a function. Reading them at
    import would mean a changed value needs a process restart to take effect and
    the test suite could not exercise them.

Anything that is a correctness invariant rather than a preference stays in
constants.py — see the note at the bottom for which is which and why.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

from .constants import SEGMENTS
from .constants import SEGMENT_CAP as _SEGMENT_CAP_DEFAULT

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Loaded at import so `alembic`, `python scripts/...` and uvicorn all see the
# same .env. dotenv never overwrites a variable that is already exported, so a
# platform secret always wins over the file.
load_dotenv(PROJECT_ROOT / ".env")

_TRUE = ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------

def _bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in _TRUE


def _int(name: str, default: int, low: int = 0, high: int = 1_000_000) -> int:
    """Read an int, degrading to the default on unset, blank, junk, or a value
    outside a sane range — a typo in a deploy config must not wedge the app."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if low <= value <= high else default


def _str(name: str, default: str = "") -> str:
    return (os.environ.get(name) or "").strip() or default


# ---------------------------------------------------------------------------
# boot settings — read once, because the session layer needs them immediately
# ---------------------------------------------------------------------------
DEBUG = _bool("LAUNCHLOOP_DEBUG", False)
SECRET_KEY = os.environ.get("SESSION_SECRET_KEY", "")
SESSION_HTTPS_ONLY = _bool("SESSION_HTTPS_ONLY", False)


# ---------------------------------------------------------------------------
# runtime settings — read per call
# ---------------------------------------------------------------------------

# --- auth ---
def min_password_length() -> int:
    return _int("MIN_PASSWORD_LENGTH", 8, low=6, high=128)

def login_window_minutes() -> int:
    return _int("LOGIN_WINDOW_MINUTES", 15, low=1, high=1440)

def max_failed_logins() -> int:
    return _int("MAX_FAILED_LOGINS", 8, low=1, high=1000)


# --- Phase 1 ---
def idea_card_limit() -> int:
    return _int("IDEA_CARD_LIMIT", 12, low=1, high=100)


# --- Phase 2: the per-block run loop ---
def segment_cap() -> int:
    """Runs each canvas block gets before it is parked and asked to be extended."""
    return _int("SEGMENT_CAP", _SEGMENT_CAP_DEFAULT, low=1, high=20)

def venture_run_backstop() -> int:
    """Venture-wide runaway ceiling. Defaults to every block's full cap, so
    raising segment_cap alone widens it too."""
    return _int("VENTURE_RUN_BACKSTOP", len(SEGMENTS) * segment_cap(), low=1, high=10_000)


# --- LLM provider ---
def llm_base_url() -> str:
    return _str("LLM_BASE_URL")

def llm_api_key() -> str:
    return _str("LLM_API_KEY")

def llm_model() -> str:
    return _str("LLM_MODEL")

def llm_provider() -> str:
    return _str("LLM_PROVIDER")

def llm_json_mode() -> bool:
    return _bool("LLM_JSON_MODE", True)

def llm_timeout() -> int:
    return _int("LLM_TIMEOUT", 60, low=5, high=600)

def llm_max_attempts() -> int:
    return _int("LLM_MAX_ATTEMPTS", 3, low=1, high=10)

def llm_max_output_tokens() -> int:
    return _int("LLM_MAX_OUTPUT_TOKENS", 4096, low=256, high=131_072)

def llm_reasoning_effort() -> str:
    """Bound a reasoning model's internal reasoning: "low" | "minimal" | "none".

    Reasoning tokens are drawn from the same budget as output. Left unbounded, a
    reasoning model can spend the entire ceiling thinking and return an empty
    message, which surfaces as an unparseable response. "none" disables the
    parameter entirely, for non-reasoning models that reject it.
    """
    return _str("LLM_REASONING_EFFORT", "low").lower()

def llm_monthly_limit_per_user() -> int:
    """Per-user monthly call cap. 0 disables it."""
    return _int("LLM_MONTHLY_LIMIT_PER_USER", 500, low=0, high=1_000_000)

# Provider endpoints and the fallback model, so pointing the app at a different
# provider needs no code change.
def zen_base_url() -> str:
    return _str("LLM_ZEN_BASE_URL", "https://opencode.ai/zen/v1")

def openrouter_base_url() -> str:
    return _str("LLM_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

def default_model() -> str:
    """Used only when LLM_MODEL is unset. The default is cheap, fast, good at
    structured output, and zero-retention on the provider side. Do NOT point
    this at Big Pickle / mimo-* / ling-* — those are documented as possibly
    training on your prompts, and this app's users paste unpublished research
    material into it."""
    return _str("LLM_DEFAULT_MODEL", "glm-5.3")

def openrouter_api_key() -> str:
    return _str("OPENROUTER_API_KEY")

def openrouter_model() -> str:
    return _str("OPENROUTER_MODEL")


# --- database ---
def database_url() -> str:
    """Read per call: db.py is reachable from scripts, the test suite and a bare
    `python -c "from app import db"`, none of which go through app.main."""
    return os.environ.get("DATABASE_URL", "")


# ---------------------------------------------------------------------------
# Deliberately NOT configurable, and why
#
#   SEGMENTS / SEGMENT_METHODS / RUN_VERDICTS / SEGMENT_OUTCOMES /
#   IDEA_STATUSES / VENTURE_STATUSES / MILESTONE_TYPES / DECISIONS
#       These generate the Postgres CHECK constraints and drive the LLM output
#       allowlists. If a deployment could change them, the running code and the
#       live schema would disagree, and a verdict the model is allowed to emit
#       would be rejected by the database as a 500.
#
#   auth.MAX_PASSWORD_BYTES = 72
#       A limit imposed by bcrypt, not a policy choice. Exposing it as config
#       would let someone silently re-enable silent password truncation.
# ---------------------------------------------------------------------------
