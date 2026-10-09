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


# --- password reset / email verification ---
def password_reset_token_minutes() -> int:
    """How long a reset or verification link stays live. Short on purpose: the
    link is the credential, and it is only useful while the user is present."""
    return _int("PASSWORD_RESET_TOKEN_MINUTES", 30, low=1, high=1440)


# --- transactional email ---
# Backend chosen by MAIL_BACKEND. `console` prints the message (and its links) to
# stdout, which is what dev and the test suite use; `smtp` talks to a real server.
# An unrecognised value degrades to console rather than failing closed — a typo in
# a deploy config must not silently stop password resets working.
def mail_backend() -> str:
    return _str("MAIL_BACKEND", "console").lower()

def mail_from() -> str:
    return _str("MAIL_FROM", "LaunchLoop <no-reply@launchloop.local>")

# Absolute base URL used to build links in emails. Must match how users reach the
# app, or every link in every email is broken.
def app_base_url() -> str:
    return _str("APP_BASE_URL", "http://localhost:8080").rstrip("/")

def smtp_host() -> str:
    return _str("SMTP_HOST")

def smtp_port() -> int:
    return _int("SMTP_PORT", 587, low=1, high=65535)

def smtp_username() -> str:
    return _str("SMTP_USERNAME")

def smtp_password() -> str:
    return _str("SMTP_PASSWORD")

def smtp_starttls() -> bool:
    return _bool("SMTP_STARTTLS", True)

def smtp_timeout() -> int:
    return _int("SMTP_TIMEOUT", 20, low=1, high=300)

# Whether signup sends a confirmation link and the banner nags unverified
# accounts. Off by default because there is no verified sending path yet: the
# only backend configured anywhere is `console`, which prints to stdout, so a
# live flag would mean every account sees a permanent banner asking them to
# click a link they can never receive. Turning it on is one env var once a
# provider and a real From domain exist — see .env.example.
def email_verification_enabled() -> bool:
    return _bool("EMAIL_VERIFICATION_ENABLED", False)


# --- Phase 1 ---
def idea_card_limit() -> int:
    return _int("IDEA_CARD_LIMIT", 12, low=1, high=100)


# --- Phase 1: file upload ---
def max_upload_bytes() -> int:
    """Hard ceiling on an upload. Enforced while reading, not from Content-Length,
    which is attacker-controlled. Uploads are never written to disk."""
    return _int("MAX_UPLOAD_BYTES", 8 * 1024 * 1024, low=1024, high=100 * 1024 * 1024)

def max_pdf_pages() -> int:
    """A thesis is longer than a paper. The user should upload the chapter."""
    return _int("MAX_PDF_PAGES", 200, low=1, high=5000)

def max_extracted_chars() -> int:
    """Caps the text handed to the model, so a long document cannot become a huge
    prompt. The user is told when their text was truncated."""
    return _int("MAX_EXTRACTED_CHARS", 200_000, low=1000, high=2_000_000)

def max_parse_seconds() -> int:
    """Wall-clock budget for extracting one file. Checked *between pages* rather
    than wrapped in a future timeout, because a thread cannot be cancelled —
    see the module docstring in app/upload.py."""
    return _int("MAX_PARSE_SECONDS", 20, low=1, high=300)


# --- Phase 2: the per-block run loop ---
def segment_cap() -> int:
    """Runs each canvas block gets before it is parked and asked to be extended."""
    return _int("SEGMENT_CAP", _SEGMENT_CAP_DEFAULT, low=1, high=20)

def venture_run_backstop() -> int:
    """Venture-wide runaway ceiling. Defaults to every block's full cap, so
    raising segment_cap alone widens it too."""
    return _int("VENTURE_RUN_BACKSTOP", len(SEGMENTS) * segment_cap(), low=1, high=10_000)


# --- Phase 1: identifier import (ORCID / DOI / arXiv) ---
def source_import_enabled() -> bool:
    """Master switch for fetching third-party metadata. Off means the import box is
    not rendered at all and the route refuses — for a deployment that must not make
    outbound requests to third parties, or that wants imports off during a demo."""
    return _bool("SOURCE_IMPORT_ENABLED", True)

def import_timeout_seconds() -> int:
    """Per-request timeout against ORCID/Crossref/arXiv. These are public metadata
    APIs that answer in well under a second; a slow one is a network problem, and
    waiting longer only holds the worker."""
    return _int("IMPORT_TIMEOUT_SECONDS", 10, low=2, high=120)

def import_max_response_bytes() -> int:
    """Hard ceiling on a third-party response body, enforced while streaming rather
    than after. Crossref and ORCID both answer with metadata we truncate anyway, so
    an unexpectedly huge body is not worth buffering."""
    return _int("IMPORT_MAX_RESPONSE_BYTES", 1_000_000, low=10_000, high=10_000_000)

def import_cache_ttl_seconds() -> int:
    """How long a fetched record is served from memory without asking again. Public
    bibliographic metadata is stable, so a long TTL is fine; arXiv records can be
    revised, which is why this is a day rather than a year."""
    return _int("IMPORT_CACHE_TTL_SECONDS", 86_400, low=0, high=604_800)

def import_max_works() -> int:
    """Rows pulled out of a record. An ORCID iD is a *person*, not a document — a
    career's worth of works — so this is what stops a prolific researcher producing
    a 400 KB prompt."""
    return _int("IMPORT_MAX_WORKS", 25, low=1, high=100)

def import_rate_limit_per_min() -> int:
    """Deliberately looser than the LLM routes: an import is one small GET to a
    public metadata API, not a model call, and it costs no monthly quota. It still
    needs a brake, because each request holds a worker thread for the whole round
    trip."""
    return _int("IMPORT_RATE_LIMIT_PER_MIN", 20, low=0, high=1000)

def import_rate_limit_ip_per_min() -> int:
    """Higher than the per-user scope for the same NAT-university reason as
    LLM_RATE_LIMIT_IP_PER_MIN."""
    return _int("IMPORT_RATE_LIMIT_IP_PER_MIN", 60, low=0, high=5000)

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


# --- per-request rate limiting on the LLM routes ---
# The monthly limit above is a *budget*, not a *rate*. These bound how fast one
# account (or one source address) can spend it, so a stuck retry loop or a
# signup bot cannot tie up a worker thread for the duration of a slow provider
# call. 0 disables a scope; see app/ratelimit.py.
def llm_rate_limit_per_min() -> int:
    return _int("LLM_RATE_LIMIT_PER_MIN", 12, low=0, high=10_000)

def llm_rate_limit_ip_per_min() -> int:
    """Per-IP ceiling. Higher than the per-user one on purpose: universities and
    small companies put many researchers behind one NAT address."""
    return _int("LLM_RATE_LIMIT_IP_PER_MIN", 60, low=0, high=100_000)

def llm_rate_limit_window_seconds() -> int:
    return _int("LLM_RATE_LIMIT_WINDOW_SECONDS", 60, low=1, high=3600)

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


def db_app_role() -> str:
    """The NOBYPASSRLS role db.get_conn() switches to with SET LOCAL ROLE.

    Empty (the default) means "do not switch". That is deliberate: RLS is inert
    for a role with BYPASSRLS, and Supabase's service_role — which DATABASE_URL
    points at — has that attribute, so the policies would be decoration. Setting
    this to a NOBYPASSRLS role is what turns them on.

    The switch is membership-tested at runtime, so a deploy that names a role it
    has not been granted keeps working on the user_id filters alone rather than
    500-ing on every request. /healthz reports `rls` either way.
    """
    return _str("DB_APP_ROLE", "launchloop_app")


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
