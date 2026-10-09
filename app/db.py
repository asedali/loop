"""
Postgres database layer for LaunchLoop, via SQLAlchemy Core over psycopg3.

No ORM: the table objects in app.schema are used to build queries, and every
function returns either a RowMapping, a list of them, a scalar, or an int.
Callers index rows with row["col"] and dict(row), which RowMapping supports.

The engine and its pool are created once per process and reused. DDL lives in
Alembic (migrations/), not here — init_db() just runs the migrations.
"""
import hashlib
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import quote, unquote

from sqlalchemy import case, create_engine, func, insert, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

from . import config
from .constants import (
    ACTION_STEP_STATUSES,
    ALLOWED_SEGMENT_FIELDS,
    ALLOWED_VENTURE_FIELDS,
    IDEA_STATUSES,
    LLM_PURPOSES,
    LLM_STATUSES,
    MENTOR_KEYS,
    MENTOR_SUBJECT_KINDS,
    MILESTONE_TYPES,
    RUN_VERDICTS,
    SEGMENTS,
    SEGMENT_KEYS,
    SEGMENT_OUTCOMES,
    SEGMENT_RESOLVED,
    SEGMENT_STATUSES,
    TOKEN_PURPOSES,
    VENTURE_STATUSES,
)
from .schema import (
    action_steps,
    bmc_elements,
    cycles,
    ideas,
    launch_strategy,
    llm_calls,
    login_attempts,
    mentor_challenges,
    password_reset_tokens,
    users,
    ventures,
)

__all__ = [
    "ALLOWED_VENTURE_FIELDS", "IDEA_STATUSES", "LLM_PURPOSES", "LLM_STATUSES",
    "MENTOR_KEYS", "MILESTONE_TYPES", "SEGMENT_KEYS", "SEGMENT_OUTCOMES",
    "SEGMENT_RESOLVED",
    "SEGMENT_STATUSES", "RUN_VERDICTS", "TOKEN_PURPOSES",
    "VENTURE_STATUSES", "get_conn", "get_engine", "init_db", "now", "ping",
    "reset_engine", "set_engine",
    # RLS security context
    "Tenant", "as_tenant", "clear_tenant", "current_tenant", "rls_active",
    "set_app_role", "set_tenant",
    # users
    "create_user", "get_user_by_email", "get_user_by_id", "set_password",
    "mark_email_verified", "user_is_verified",
    # password reset / email verification tokens
    "create_reset_token", "get_reset_token", "consume_reset_token",
    "consume_user_tokens", "redeemable_token",
    # account data control
    "delete_user_account", "export_user_data",
    # ideas
    "create_idea", "list_ideas", "get_idea", "set_idea_status",
    # ventures
    "create_venture", "list_ventures", "get_venture", "get_child_ventures",
    "update_venture",
    # segments (the nine BMC blocks)
    "get_segments", "get_segment", "get_segment_map_for_user", "update_segment",
    "set_segment_outcome", "set_segment_hypothesis", "extend_segment",
    "next_recommended_segment", "all_resolved", "resolved_count", "open_gaps",
    "carry_passed_segments",
    "bump_segment_run",
    # bmc elements (legacy alias kept: same table, same rows)
    "get_bmc_elements", "get_bmc_map_for_user", "update_bmc_element",
    # cycles / runs
    "create_cycle", "get_current_cycle", "get_cycle", "log_cycle_results",
    "save_run_verdict", "list_cycles", "list_segment_runs",
    # launch strategy
    "save_launch_strategy", "get_launch_strategy",
    # action-plan step tracking
    "step_key", "seed_action_steps", "get_action_steps", "get_action_step",
    "set_action_step", "action_plan_progress", "join_action_plan",
    # mentor challenges
    "save_mentor_challenge", "list_mentor_challenges", "latest_mentor_challenge",
    # llm call log / quota
    "log_llm_call", "count_llm_calls_this_month",
    # login throttling
    "record_login_attempt", "recent_failed_logins",
]

_engine: Engine | None = None


def database_url() -> str:
    # Read through app.config, which loads .env itself — db.py is reachable from
    # scripts, the test suite and a bare `python -c "from app import db"`, none
    # of which go through app.main.
    url = config.database_url()
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Put the Supabase connection string in "
            ".env (or the environment) — use the SESSION POOLER (port 5432), "
            "not the transaction pooler on 6543, which breaks server-side "
            "prepared statements."
        )
    # Some hosts (Heroku-style, and some Supabase copy-pastes) prefix the URL
    # with the protocol, which SQLAlchemy then cannot parse.
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]

    # Supabase's TLS cert is issued by Supabase's own CA, which is not in the
    # OS trust store, so sslrootcert has to point at the downloaded root.crt.
    # A relative path there is resolved by libpq against the process CWD, which
    # silently breaks the app whenever it is started from somewhere other than
    # the project root. Anchor it to the project instead.
    m = re.search(r"([?&])sslrootcert=([^&]+)", url)
    if m:
        ca = unquote(m.group(2))
        if not os.path.isabs(ca):
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            resolved = os.path.join(root, ca)
            if not os.path.exists(resolved):
                raise RuntimeError(
                    f"sslrootcert points at {ca!r}, which does not exist. "
                    f"Download the Supabase CA cert (Project Settings -> Database "
                    f"-> Connection string) and save it to {resolved}."
                )
            url = url[:m.start(2)] + quote(resolved) + url[m.end(2):]
    return url


def get_engine() -> Engine:
    """Built once per process and reused; the pool is what makes this cheap."""
    global _engine
    if _engine is None:
        # Pool shape is per-platform, not per-code: 5/5 on Render's single
        # long-lived container, 1/0 on Vercel where every concurrent invocation
        # is its own process. See config.db_pool_size().
        pool_size = config.db_pool_size()
        kwargs = dict(future=True, pool_pre_ping=True, pool_recycle=1800)
        if pool_size == 0:
            # SQLAlchemy requires a real pool class; this is the documented way to
            # open and close a connection per checkout, which is what "no pool"
            # means for a serverless invocation.
            kwargs["poolclass"] = NullPool
        else:
            kwargs["pool_size"] = pool_size
            kwargs["max_overflow"] = config.db_pool_max_overflow()
        _engine = create_engine(database_url(), **kwargs)
    return _engine


def set_engine(engine: Engine | None) -> None:
    """Install an engine directly. Used by the test suite, which builds a
    per-test-schema engine rather than going through DATABASE_URL."""
    global _engine
    _engine = engine


def reset_engine() -> None:
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None


@contextmanager
def get_conn():
    """Yields a Connection in a transaction, committing on clean exit and
    rolling back on any exception. Kept as a context manager so callers and
    tests have the same shape they had against sqlite3.

    Also the single place the per-transaction security context is applied:
    SET LOCAL ROLE for RLS, then the tenant settings the policies read. SET LOCAL
    is transaction-scoped, so the connection returns to the pool clean — which
    matters, because it is the same pool the *next* tenant's request will use.
    """
    conn = get_engine().connect()
    trans = conn.begin()
    try:
        _apply_security_context(conn)
        yield conn
        trans.commit()
    except Exception:
        trans.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Per-transaction security context (RLS)
#
# The RLS policies read two session settings. `app.user_id` is the tenant: unset
# means DENY, because current_setting(..., true) yields NULL and NULL = user_id
# is NULL. `app.lookup_email` exists only for the pre-authentication paths — login
# and signup have to find a row before anyone is signed in.
#
# A ContextVar rather than a user_id parameter through all of this module: a
# parameter can be forgotten at a call site, and a forgotten parameter under RLS
# returns *zero rows*, which is a bug that hides. get_current_user() sets this
# once per request, and every route and every render() already goes through it.
#
# ContextVars across requests: Starlette runs sync handlers in a *reused* AnyIO
# worker thread, so a plain module global would leak one tenant into the next
# request. This was probed rather than assumed — anyio runs each task inside a
# fresh copy_context(), so a value set during one request is invisible in the
# next. TestTenantContextDoesNotLeak keeps that true.
# ---------------------------------------------------------------------------

_tenant_ctx: ContextVar = ContextVar("launchloop_tenant", default=None)

# The NOBYPASSRLS role the app switches to. Empty means "do not switch", which is
# the safe default: a deploy without the membership granted keeps the pre-RLS
# behaviour rather than 500-ing on every request.
_app_role: str = ""


@dataclass(frozen=True)
class Tenant:
    user_id: int | None = None
    lookup_email: str | None = None
    # The hash of a presented reset/verification token. The token is the
    # credential you hold *before* you know whose account it belongs to, so
    # this is what admits the single matching row — then the caller installs the
    # real tenant from the resolved row.
    lookup_token: str | None = None


def current_tenant() -> Tenant:
    return _tenant_ctx.get() or Tenant()


def set_tenant(user_id: int | None = None, lookup_email: str | None = None,
               lookup_token: str | None = None):
    """Install a security context for the current transaction chain.

    Called from `auth.get_current_user()` — the one place every route and every
    `render()` passes through — so a request can never hold a user without also
    holding a tenant.
    """
    _tenant_ctx.set(Tenant(user_id=user_id, lookup_email=lookup_email,
                           lookup_token=lookup_token))


def clear_tenant():
    _tenant_ctx.set(Tenant())


@contextmanager
def as_tenant(user_id: int | None = None, lookup_email: str | None = None,
              lookup_token: str | None = None):
    """Scope a block to one tenant. The test suite and any future background job
    need this; request handlers use `set_tenant` instead."""
    token = _tenant_ctx.set(Tenant(user_id=user_id, lookup_email=lookup_email,
                                   lookup_token=lookup_token))
    try:
        yield
    finally:
        _tenant_ctx.reset(token)


def set_app_role(role: str):
    """Name the NOBYPASSRLS role `get_conn()` should switch to. Empty disables."""
    global _app_role
    _app_role = role or ""


def _apply_security_context(conn):
    role = _app_role
    if role:
        # Membership-tested, not assumed: an ungranted deploy must keep working.
        member = conn.exec_driver_sql(
            "SELECT pg_has_role(current_user, %s, 'member')", (role,)
        ).scalar_one()
        if member:
            conn.exec_driver_sql(f'SET LOCAL ROLE "{role}"')
    tenant = current_tenant()
    if tenant.user_id is not None:
        conn.exec_driver_sql("SELECT set_config('app.user_id', %s, true)",
                             (str(tenant.user_id),))
    if tenant.lookup_email:
        conn.exec_driver_sql("SELECT set_config('app.lookup_email', %s, true)",
                             (tenant.lookup_email,))
    if tenant.lookup_token:
        conn.exec_driver_sql("SELECT set_config('app.lookup_token', %s, true)",
                             (tenant.lookup_token,))


def rls_active(conn=None) -> bool:
    """Whether the RLS policies will actually be enforced for this app's queries.

    Worth asking out loud. A connection as a superuser or as Supabase's
    service_role bypasses every policy regardless of FORCE, so a deploy can have
    these policies installed and still have no enforcement at all.

    The subtlety: this must report on the role **transactions will run as**, not
    the role the pool connects as. `get_conn()` issues SET LOCAL ROLE, so
    `current_user` on a raw pooled connection is the owner — which would answer
    "bypasses" even on a fully-enforced deploy. Ask about the app role when it is
    configured and this connection is a member of it.
    """
    close = False
    if conn is None:
        conn = get_engine().connect()
        close = True
    try:
        role = _app_role
        if role and conn.exec_driver_sql(
                "SELECT pg_has_role(current_user, %s, 'member')", (role,)).scalar_one():
            row = conn.exec_driver_sql(
                "SELECT rolbypassrls FROM pg_roles WHERE rolname = %s",
                (role,)).fetchone()
        else:
            row = conn.exec_driver_sql(
                "SELECT rolbypassrls FROM pg_roles WHERE rolname = current_user"
            ).fetchone()
        return row is not None and not row[0]
    except Exception:
        return False
    finally:
        if close:
            conn.close()


def now() -> datetime:
    return datetime.now(timezone.utc)


def current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def ping() -> bool:
    """Cheap round-trip used by the /healthz probe."""
    with get_conn() as conn:
        return conn.execute(text("SELECT 1")).scalar() == 1


def init_db():
    """Bring the schema up to head. Idempotent and safe to call concurrently:
    a session-level advisory lock serialises the runners."""
    from alembic import command
    from alembic.config import Config

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = Config(os.path.join(root, "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(root, "migrations"))
    # Via attributes, not set_main_option: Alembic's Config wraps configparser,
    # which does %-interpolation, and a percent-encoded sslrootcert path
    # (%2F...) raises "invalid interpolation syntax" before we ever connect.
    cfg.attributes["sqlalchemy_url"] = get_engine().url.render_as_string(
        hide_password=False
    )

    with get_engine().begin() as conn:
        conn.execute(text("SELECT pg_advisory_lock(hashtext('launchloop_alembic'))"))
    try:
        command.upgrade(cfg, "head")
    finally:
        with get_engine().begin() as conn:
            conn.execute(text("SELECT pg_advisory_unlock(hashtext('launchloop_alembic'))"))


# ---------- users ----------

def create_user(email: str, password_hash: str, email_verified_at: datetime = None):
    with get_conn() as conn:
        return conn.execute(
            insert(users).values(
                email=email, password_hash=password_hash,
                email_verified_at=email_verified_at, created_at=now()
            ).returning(users.c.id)
        ).scalar_one()


def get_user_by_email(email: str):
    with get_conn() as conn:
        return conn.execute(
            select(users).where(users.c.email == email)
        ).mappings().fetchone()


def get_user_by_id(user_id: int):
    with get_conn() as conn:
        return conn.execute(
            select(users).where(users.c.id == user_id)
        ).mappings().fetchone()


def set_password(user_id: int, password_hash: str) -> int:
    """Change the password, and invalidate every session that predates it.

    Returns the new `session_epoch` so the caller can re-issue its own cookie —
    otherwise the session performing the change would revoke itself along with
    everyone else's, which is the one outcome this is not for.

    The bump lives HERE rather than being left to callers, and that placement is
    the invariant: it is not possible in this codebase to change a password
    without revoking sessions, because the one function that changes a password
    is the one that advances the counter. A caller that wanted to skip it would
    have to reimplement this UPDATE, and there is nothing to tempt them into
    that.

    Computed as `session_epoch + 1` in SQL rather than read-modify-write in
    Python: two concurrent resets would otherwise both read the current value
    and write the same next one, and the second password change would leave the
    first one's session alive.
    """
    with get_conn() as conn:
        return conn.execute(
            update(users)
            .where(users.c.id == user_id)
            .values(
                password_hash=password_hash,
                session_epoch=users.c.session_epoch + 1,
            )
            .returning(users.c.session_epoch)
        ).scalar_one()


def mark_email_verified(user_id: int):
    """Idempotent. Re-verifying an already-verified address is a no-op, so a
    second click on an old link does not move the timestamp."""
    with get_conn() as conn:
        conn.execute(
            update(users)
            .where(users.c.id == user_id, users.c.email_verified_at.is_(None))
            .values(email_verified_at=now())
        )


def user_is_verified(user_id: int) -> bool:
    """One place for M4.4 (share links) and M5.1 (invites) to ask. Nothing gates
    on it today; it exists so neither has to re-derive the rule.

    True when the flag is off, deliberately. With confirmation disabled no
    account ever gets an `email_verified_at`, so reading the column alone would
    tell M4.4 and M5.1 that every single user is unverified and lock out the
    entire app the moment either one ships.
    """
    if not config.email_verification_enabled():
        return True
    with get_conn() as conn:
        return conn.execute(
            select(users.c.email_verified_at).where(users.c.id == user_id)
        ).scalar_one() is not None


# ---------- password reset / email verification tokens ----------

def create_reset_token(user_id: int, purpose: str, token_hash: str, expires_at: datetime):
    if purpose not in TOKEN_PURPOSES:
        raise ValueError(f"Unknown token purpose: {purpose!r}")
    with get_conn() as conn:
        return conn.execute(
            insert(password_reset_tokens).values(
                user_id=user_id, purpose=purpose, token_hash=token_hash,
                expires_at=expires_at, created_at=now(),
            ).returning(password_reset_tokens.c.id)
        ).scalar_one()


def get_reset_token(token_hash: str, purpose: str):
    """Lookup by hash, scoped to one purpose so a verification link can never be
    redeemed as a password reset, or the reverse."""
    with get_conn() as conn:
        return conn.execute(
            select(password_reset_tokens).where(
                password_reset_tokens.c.token_hash == token_hash,
                password_reset_tokens.c.purpose == purpose,
            )
        ).mappings().fetchone()


def consume_reset_token(token_id: int):
    """Stamp `used_at`. This is the single-use guarantee: the row stays for audit,
    but `redeemable_token()` stops accepting it."""
    with get_conn() as conn:
        conn.execute(
            update(password_reset_tokens)
            .where(password_reset_tokens.c.id == token_id)
            .values(used_at=now())
        )


def consume_user_tokens(user_id: int, purpose: str):
    """Burn every unused token of one purpose for a user.

    Called on password reset: a reset must not leave a second live reset link
    behind, or someone who saw an earlier one keeps access after the owner has
    changed the password.
    """
    with get_conn() as conn:
        conn.execute(
            update(password_reset_tokens)
            .where(
                password_reset_tokens.c.user_id == user_id,
                password_reset_tokens.c.purpose == purpose,
                password_reset_tokens.c.used_at.is_(None),
            )
            .values(used_at=now())
        )


def redeemable_token(token_hash: str, purpose: str):
    """Resolve a token *hash* to its row, or None.

    Takes the hash, not the token: token minting is auth.py's business, and db.py
    must not import it (auth.py imports db, so that would be circular).

    None for every failure — unknown, wrong purpose, already used, expired —
    deliberately indistinguishable to the caller. The user is told "this link is
    invalid or has expired" and never which.
    """
    if not token_hash:
        return None
    row = get_reset_token(token_hash, purpose)
    if not row:
        return None
    if row["used_at"] is not None:
        return None
    if row["expires_at"] <= now():
        return None
    return row


# ---------- ideas ----------

def create_idea(user_id: int, title: str, commercial_framing: str, strength_signal: str, raw_claims: str):
    with get_conn() as conn:
        return conn.execute(
            insert(ideas).values(
                user_id=user_id, title=title, commercial_framing=commercial_framing,
                strength_signal=strength_signal, raw_claims=raw_claims,
                status="candidate", created_at=now(),
            ).returning(ideas.c.id)
        ).scalar_one()


def list_ideas(user_id: int, status: str = None):
    query = select(ideas).where(ideas.c.user_id == user_id)
    if status:
        query = query.where(ideas.c.status == status)
    with get_conn() as conn:
        return conn.execute(query.order_by(ideas.c.created_at.desc())).mappings().fetchall()


def get_idea(idea_id: int, user_id: int):
    with get_conn() as conn:
        return conn.execute(
            select(ideas).where(ideas.c.id == idea_id, ideas.c.user_id == user_id)
        ).mappings().fetchone()


def set_idea_status(idea_id: int, status: str):
    with get_conn() as conn:
        conn.execute(update(ideas).where(ideas.c.id == idea_id).values(status=status))


# ---------- ventures ----------

def create_venture(user_id: int, idea_id: int, parent_venture_id: int = None, pivot_note: str = None):
    """Seeds all nine segments in the recommended testing order, so the board a
    user sees is populated from the first load with nothing to backfill."""
    with get_conn() as conn:
        venture_id = conn.execute(
            insert(ventures).values(
                user_id=user_id, idea_id=idea_id, phase=2, cycle_count=0,
                max_cycles=config.venture_run_backstop(), status="active",
                parent_venture_id=parent_venture_id, pivot_note=pivot_note,
                created_at=now(),
            ).returning(ventures.c.id)
        ).scalar_one()
        conn.execute(
            insert(bmc_elements),
            [
                {
                    "venture_id": venture_id, "element_name": seg["key"],
                    "status": "untested", "notes": "", "updated_at": now(),
                    "position": i, "label": seg["label"], "severity": seg["severity"],
                    "outcome": "pending", "cycle_count": 0, "max_cycles": config.segment_cap(),
                }
                for i, seg in enumerate(SEGMENTS)
            ],
        )
        return venture_id


# The joined columns are added as *labels on a single-entity select* rather
# than a second entity in the FROM. Selecting two entities would emit two `id`
# columns and SQLAlchemy would disambiguate the duplicate, so row["id"] could
# silently resolve to ideas.id instead of ventures.id — which main.py's
# apply_analysis() relies on for phase/cycle_count/max_cycles.
_VENTURE_WITH_IDEA = (
    select(ventures)
    .join(ideas, ventures.c.idea_id == ideas.c.id)
    .add_columns(
        ideas.c.title.label("idea_title"),
        ideas.c.commercial_framing,
    )
)


def list_ventures(user_id: int):
    with get_conn() as conn:
        return conn.execute(
            _VENTURE_WITH_IDEA
            .where(ventures.c.user_id == user_id)
            .order_by(ventures.c.created_at.desc())
        ).mappings().fetchall()


def get_venture(venture_id: int, user_id: int):
    with get_conn() as conn:
        return conn.execute(
            _VENTURE_WITH_IDEA
            .where(ventures.c.id == venture_id, ventures.c.user_id == user_id)
        ).mappings().fetchone()


def get_child_ventures(venture_id: int):
    """Ventures spawned by a pivot decision on this one."""
    with get_conn() as conn:
        return conn.execute(
            _VENTURE_WITH_IDEA
            .where(ventures.c.parent_venture_id == venture_id)
            .order_by(ventures.c.created_at)
        ).mappings().fetchall()


def update_venture(venture_id: int, **fields):
    """Updates venture fields. Field names are checked against an allowlist —
    this builds SQL by column name, so an unchecked key would be injectable."""
    fields = {k: v for k, v in fields.items() if k in ALLOWED_VENTURE_FIELDS}
    if not fields:
        return
    with get_conn() as conn:
        conn.execute(update(ventures).where(ventures.c.id == venture_id).values(**fields))


# ---------- segments (the nine BMC blocks) ----------
#
# The rows live in bmc_elements, which predates the per-segment loop. The
# name is kept so the dashboard grouping and the phase-3 prompt keep working;
# get_bmc_elements stays as an alias below.

def get_segments(venture_id: int):
    """All nine segments, in the recommended testing order."""
    with get_conn() as conn:
        return conn.execute(
            select(bmc_elements)
            .where(bmc_elements.c.venture_id == venture_id)
            .order_by(
                bmc_elements.c.position.asc().nulls_last(),
                bmc_elements.c.element_name,
            )
        ).mappings().fetchall()


def get_segment(venture_id: int, segment_key: str):
    with get_conn() as conn:
        return conn.execute(
            select(bmc_elements).where(
                bmc_elements.c.venture_id == venture_id,
                bmc_elements.c.element_name == segment_key,
            )
        ).mappings().fetchone()


def get_segment_map_for_user(user_id: int) -> dict:
    """Every segment of every venture the user owns, grouped by venture_id, for
    the dashboard's per-venture progress line."""
    with get_conn() as conn:
        rows = conn.execute(
            select(bmc_elements)
            .join(ventures, ventures.c.id == bmc_elements.c.venture_id)
            .where(ventures.c.user_id == user_id)
            .order_by(bmc_elements.c.position.asc().nulls_last())
        ).mappings().fetchall()
    grouped: dict = {}
    for row in rows:
        grouped.setdefault(row["venture_id"], []).append(row)
    return grouped


def _guarded_segment_write(stmt_values: dict) -> None:
    """Applies an update to one segment row, refusing unknown keys and statuses
    rather than letting them reach the CHECK constraints as a 500."""
    values = dict(stmt_values)
    venture_id = values.pop("_venture_id", None)
    segment_key = values.pop("_segment", None)
    if venture_id is None or segment_key not in SEGMENT_KEYS:
        return
    st = {k: v for k, v in values.items() if k in ALLOWED_SEGMENT_FIELDS}
    if not st:
        return
    if st.get("outcome") not in (None, *SEGMENT_OUTCOMES):
        return
    if st.get("status") not in (None, *SEGMENT_STATUSES):
        return
    with get_conn() as conn:
        conn.execute(
            update(bmc_elements)
            .where(
                bmc_elements.c.venture_id == venture_id,
                bmc_elements.c.element_name == segment_key,
            )
            .values(**st)
        )


def update_segment(venture_id: int, segment_key: str, *, status=None, notes=None):
    """Records the evidence read for a segment without touching its outcome."""
    if segment_key not in SEGMENT_KEYS:
        return
    values = {"_venture_id": venture_id, "_segment": segment_key,
              "updated_at": now(), "status": status}
    if notes is not None:
        values["notes"] = notes
    _guarded_segment_write(values)


def set_segment_outcome(venture_id: int, segment_key: str, outcome: str, *,
                        status: str = None, note: str = None,
                        hypothesis: str = None):
    """Moves a segment to a new lifecycle state.

    `status` defaults to whatever the outcome implies — 'passed' requires
    'confirmed' and 'failed' requires 'disconfirmed' — because those pairs are
    enforced by CHECK constraints and would otherwise raise a 500 on the page
    that submits the verdict.
    """
    if segment_key not in SEGMENT_KEYS or outcome not in SEGMENT_OUTCOMES:
        return
    forced = {
        "passed": "confirmed",
        "failed": "disconfirmed",
        "parked": "mixed",
    }.get(outcome)
    values = {
        "_venture_id": venture_id,
        "_segment": segment_key,
        "outcome": outcome,
        "status": status or forced or "mixed",
        "outcome_note": note,
        "updated_at": now(),
        "decided_at": now(),
    }
    if hypothesis is not None:
        values["hypothesis"] = hypothesis
    _guarded_segment_write(values)


def set_segment_hypothesis(venture_id: int, segment_key: str, hypothesis: str):
    """The assumption about to be tested. Set on the first run, revised on pivot."""
    if segment_key not in SEGMENT_KEYS:
        return
    _guarded_segment_write({
        "_venture_id": venture_id, "_segment": segment_key,
        "hypothesis": (hypothesis or "").strip()[:1000], "updated_at": now(),
    })


def bump_segment_run(venture_id: int, segment_key: str) -> None:
    """Counts one run against a segment and marks it active."""
    if segment_key not in SEGMENT_KEYS:
        return
    with get_conn() as conn:
        conn.execute(
            update(bmc_elements)
            .where(
                bmc_elements.c.venture_id == venture_id,
                bmc_elements.c.element_name == segment_key,
            )
            .values(
                cycle_count=bmc_elements.c.cycle_count + 1,
                outcome=case(
                    (bmc_elements.c.outcome.in_(("passed", "parked")), "pending"),
                    else_="active",
                ),
                decided_at=None,
                updated_at=now(),
            )
        )


def extend_segment(venture_id: int, segment_key: str, by: int = None) -> None:
    """Raises one segment's cap. Deliberately scoped to the segment so an
    exhausted block never pauses the whole venture the way the old global cap
    did."""
    if segment_key not in SEGMENT_KEYS:
        return
    by = config.segment_cap() if by is None else by
    with get_conn() as conn:
        conn.execute(
            update(bmc_elements)
            .where(
                bmc_elements.c.venture_id == venture_id,
                bmc_elements.c.element_name == segment_key,
            )
            .values(max_cycles=bmc_elements.c.max_cycles + by, updated_at=now())
        )


def carry_passed_segments(from_venture_id: int, to_venture_id: int) -> int:
    """Copy the evidence from one venture's validated blocks onto another's.

    Used when a pivot spawns a sibling venture: the blocks that were already
    passed do not become unproven just because the hypothesis about a *different*
    block changed. Blocks that were pending, iterating or failed stay fresh in
    the new venture, which is created with them already at 'pending'.
    """
    carried = 0
    with get_conn() as conn:
        for seg in get_segments(from_venture_id):
            if seg["outcome"] != "passed":
                continue
            result = conn.execute(
                update(bmc_elements)
                .where(
                    bmc_elements.c.venture_id == to_venture_id,
                    bmc_elements.c.element_name == seg["element_name"],
                )
                .values(
                    status="confirmed",
                    notes=seg["notes"] or "",
                    hypothesis=seg["hypothesis"],
                    outcome="passed",
                    outcome_note=seg["outcome_note"],
                    cycle_count=seg["cycle_count"],
                    decided_at=seg["decided_at"] or now(),
                    updated_at=now(),
                )
            )
            carried += result.rowcount or 0
    return carried


def next_recommended_segment(venture_id: int):
    """The first unresolved segment in the recommended order, or None.

    Advisory only — the user can open any segment at any time; this just gives
    the board a sensible "start here" suggestion.
    """
    for seg in get_segments(venture_id):
        if seg["outcome"] not in SEGMENT_RESOLVED:
            return seg
    return None


def resolved_count(venture_id: int) -> tuple:
    """(resolved, total) where resolved counts passed + parked.

    A parked segment counts: finishing with a documented gap is a legitimate
    outcome, and the gaps are carried into the launch strategy.
    """
    rows = get_segments(venture_id)
    total = len(rows)
    resolved = sum(1 for r in rows if r["outcome"] in SEGMENT_RESOLVED)
    return resolved, total


def all_resolved(venture_id: int) -> bool:
    """Phase-3 gate: every segment is either passed or parked.

    Replaces the old all_confirmed(), which gated on every block being
    'confirmed' and so could never be reached once a block was parked.
    """
    rows = get_segments(venture_id)
    # A short row set means a segment is missing, and an off-list key means one
    # the board would not render. Either would make this silently unreachable
    # (or, worse, wrongly true), so treat both as unresolved.
    if len(rows) != len(SEGMENT_KEYS):
        return False
    if {r["element_name"] for r in rows} != set(SEGMENT_KEYS):
        return False
    return all(r["outcome"] in SEGMENT_RESOLVED for r in rows)


def open_gaps(venture_id: int) -> list:
    """Parked segments with their notes, for the phase-3 strategy prompt: the
    gaps are real and the launch plan has to account for them."""
    return [
        {"segment": r["element_name"], "label": r["label"], "note": r["outcome_note"]}
        for r in get_segments(venture_id)
        if r["outcome"] == "parked"
    ]


# ---------- legacy aliases ----------
# Same table, same rows. Kept so callers that predate the segment naming do not
# silently keep working and diverge from the new vocabulary.

def get_bmc_elements(venture_id: int):
    return get_segments(venture_id)


def get_bmc_map_for_user(user_id: int) -> dict:
    return get_segment_map_for_user(user_id)


def update_bmc_element(venture_id: int, element_name: str, status: str, notes: str):
    return update_segment(venture_id, element_name, status=status, notes=notes)


# ---------- runs (a run is one pass at one segment) ----------
#
# The table is still called `cycles`; cycle_number is the venture-wide running
# total, and `segment` says which block the run was testing.

def create_cycle(venture_id: int, cycle_number: int, tasks: list, segment: str = None):
    with get_conn() as conn:
        return conn.execute(
            insert(cycles).values(
                venture_id=venture_id, cycle_number=cycle_number, segment=segment,
                todos_json=tasks, created_at=now(),
            ).returning(cycles.c.id)
        ).scalar_one()


def get_current_cycle(venture_id: int):
    """The in-flight run, if any. One at a time per venture, as before."""
    with get_conn() as conn:
        return conn.execute(
            select(cycles)
            .where(cycles.c.venture_id == venture_id)
            .order_by(cycles.c.cycle_number.desc())
            .limit(1)
        ).mappings().fetchone()


def get_cycle(cycle_id: int, venture_id: int):
    with get_conn() as conn:
        return conn.execute(
            select(cycles).where(cycles.c.id == cycle_id, cycles.c.venture_id == venture_id)
        ).mappings().fetchone()


def log_cycle_results(cycle_id: int, results: list):
    with get_conn() as conn:
        conn.execute(update(cycles).where(cycles.c.id == cycle_id).values(results_json=results))


def save_run_verdict(cycle_id: int, analysis: dict, verdict: str):
    if verdict not in RUN_VERDICTS:
        verdict = "iterate"
    with get_conn() as conn:
        conn.execute(
            update(cycles)
            .where(cycles.c.id == cycle_id)
            .values(analysis_json=analysis, decision=verdict)
        )


def list_cycles(venture_id: int):
    with get_conn() as conn:
        return conn.execute(
            select(cycles)
            .where(cycles.c.venture_id == venture_id)
            .order_by(cycles.c.cycle_number)
        ).mappings().fetchall()


def list_segment_runs(venture_id: int, segment_key: str):
    """Every run recorded against one segment, oldest first."""
    with get_conn() as conn:
        return conn.execute(
            select(cycles)
            .where(cycles.c.venture_id == venture_id, cycles.c.segment == segment_key)
            .order_by(cycles.c.cycle_number)
        ).mappings().fetchall()


# ---------- launch strategy ----------

def save_launch_strategy(venture_id: int, funding_matches: list, gtm_channels: list, action_plan: list):
    """Store one strategy, and seed the checklist for its action plan.

    The seed runs inside the same transaction as the insert, so there is no
    window in which a plan exists with no steps to record against it — a
    constraint failure rolls both back together.
    """
    with get_conn() as conn:
        strategy_id = conn.execute(
            insert(launch_strategy).values(
                venture_id=venture_id, funding_matches_json=funding_matches,
                gtm_channels_json=gtm_channels, action_plan_json=action_plan,
                created_at=now(),
            ).returning(launch_strategy.c.id)
        ).scalar_one()
        seed_action_steps(venture_id, action_plan, conn=conn)
    return strategy_id


def get_launch_strategy(venture_id: int):
    with get_conn() as conn:
        return conn.execute(
            select(launch_strategy)
            .where(launch_strategy.c.venture_id == venture_id)
            .order_by(launch_strategy.c.created_at.desc())
            .limit(1)
        ).mappings().fetchone()

# ---------- action-plan step tracking ----------

def step_key(step: str) -> str:
    """The identity of one launch-plan step: the SHA-256 of its text.

    Deliberately NOT the step's index in action_plan_json. Regenerating the
    strategy produces a different list, so index 3 of the new plan is not index 3
    of the old one — keying on position would re-label a completion onto a step
    the researcher never did. Keying on the text means identical wording keeps
    its status across a regeneration and different wording starts clean.

    Only surrounding whitespace is normalised, and only because llm.py already
    strips the step before saving it: normalising case or inner spaces would make
    the key disagree with the text it is derived from, and two genuinely
    different steps could collide.

    migrations/versions/c9e4b2a71d38 carries its own copy of this function,
    because a migration must not import a module that builds a live engine.
    TestActionStepTracking pins the two to the same value.
    """
    return hashlib.sha256(step.strip().encode("utf-8")).hexdigest()

def seed_action_steps(venture_id: int, action_plan: list, conn=None) -> None:
    """Ensure one row exists per step of the newest plan. Idempotent.

    Called from save_launch_strategy with that caller's connection, so the
    checklist is written in the same transaction as the strategy: there is no
    window in which a plan is visible with no steps to record against it, and a
    failure rolls both back together. Passing `conn` is therefore not an
    optimisation, it is what makes that guarantee true.

    ON CONFLICT DO NOTHING rather than an upsert of every column, because a step
    the researcher has already marked `done` must keep its status and its note
    when a regeneration happens to reproduce the same wording. An upsert would
    silently reset it to pending, which is the one outcome this table exists to
    prevent.
    """
    if not isinstance(action_plan, list):
        return
    rows = []
    stamp = now()
    for item in action_plan:
        if not isinstance(item, dict):
            continue
        text_ = item.get("step")
        if not isinstance(text_, str) or not text_.strip():
            continue
        milestone = item.get("milestone_type")
        rows.append({
            "venture_id": venture_id,
            "step_key": step_key(text_),
            "step": text_.strip(),
            "milestone_type": milestone if milestone in MILESTONE_TYPES else "pilot",
            "status": "pending",
            "created_at": stamp,
            "updated_at": stamp,
        })
    if not rows:
        return
    statement = pg_insert(action_steps).values(rows).on_conflict_do_nothing(
        index_elements=[action_steps.c.venture_id, action_steps.c.step_key]
    )
    if conn is not None:
        conn.execute(statement)
        return
    with get_conn() as own:
        own.execute(statement)

def get_action_steps(venture_id: int) -> dict:
    """Every tracked step for a venture, keyed by step_key.

    A dict, because the plan is rendered by looking each step up rather than
    walking two lists in step — the plan's order is the model's, and a completion
    must attach to a step by identity rather than by position.
    """
    with get_conn() as conn:
        rows = conn.execute(
            select(action_steps)
            .where(action_steps.c.venture_id == venture_id)
        ).mappings().fetchall()
    return {r["step_key"]: dict(r) for r in rows}

def get_action_step(venture_id: int, key: str) -> dict:
    """One step, scoped to its venture.

    The venture_id in the WHERE clause is the tenant check at this layer: a
    step_key from another user's plan matches nothing, so the caller cannot read
    or mutate it. RLS enforces the same thing underneath, but the filter is here
    too because a forgotten parameter must not be the only thing standing between
    two tenants.
    """
    with get_conn() as conn:
        row = conn.execute(
            select(action_steps).where(
                action_steps.c.venture_id == venture_id,
                action_steps.c.step_key == key,
            )
        ).mappings().fetchone()
    return dict(row) if row else None

def set_action_step(venture_id: int, key: str, status: str, *, note: str = None) -> bool:
    """Record how far one step got. Returns False if there is no such step.

    `note` is written whenever it is supplied, including on a revert to
    `pending`: un-blocking a step keeps the reason it was blocked, because that
    reason usually still holds and re-typing it would be a small punishment for
    a state the researcher did not create.

    The status allowlist is re-checked here as well as by the CHECK constraint,
    for the same reason update_venture allowlists its field names: an
    unvalidated value would be a 500 from the database rather than a refusal
    from the route.
    """
    if status not in ACTION_STEP_STATUSES:
        return False
    values = {
        "status": status,
        "updated_at": now(),
        # decided_at is the moment a step left `pending`. Reverting clears it:
        # the step is open again, and a stale timestamp would make "decided"
        # mean two different things on the same row.
        "decided_at": now() if status != "pending" else None,
    }
    if note is not None:
        values["outcome_note"] = note
    with get_conn() as conn:
        return conn.execute(
            update(action_steps).where(
                action_steps.c.venture_id == venture_id,
                action_steps.c.step_key == key,
            ).values(values)
        ).rowcount > 0

def join_action_plan(steps: dict, plan: list) -> list:
    """The plan and its tracking records, joined on step identity.

    Each item is `{step, milestone_type, key, index, status, note, tracked}`.

    This is the only place the join happens. The alternative — computing step_key
    inside the template — needs the hash as a Jinja filter, which would put the
    definition of a step's identity in two files that nothing keeps in step, and
    a silent mismatch there shows up as a checklist that silently stops
    reflecting reality rather than as an error.

    `tracked` is False when a plan step has no row at all. That can only happen
    if a launch_strategy row was written without going through
    save_launch_strategy, so the step renders as pending and the controls still
    work — the honest reading of a step nothing has recorded as done.
    """
    items = []
    for index, item in enumerate(plan or []):
        if not isinstance(item, dict):
            continue
        text_ = item.get("step")
        if not isinstance(text_, str) or not text_.strip():
            continue
        key = step_key(text_)
        record = steps.get(key) or {}
        items.append({
            "step": text_.strip(),
            "milestone_type": item.get("milestone_type") or "pilot",
            "key": key,
            "index": index,
            "status": record.get("status") or "pending",
            "note": record.get("outcome_note") or "",
            "tracked": bool(record),
        })
    return items

# ---------- mentor challenges ----------

def save_mentor_challenge(user_id: int, mentor_key: str, subject_kind: str,
                          questions: list, *, idea_id: int = None,
                          venture_id: int = None, segment: str = None,
                          dropped: int = 0) -> int:
    """Store one challenge. Returns its id.

    Deliberately a plain INSERT with no side effects. Nothing in this module, or in
    the path that calls it, touches a segment outcome, a run verdict or a venture
    status — a mentor asks, and the researcher still decides (invariant 4). The
    test that proves it asserts a challenge against a failed block leaves the
    outcome untouched.

    The arguments are validated here as well as by the CHECK constraints, for the
    reason update_venture allowlists its field names: an unvalidated value would
    surface as a 500 from the database rather than a refusal from the route.
    """
    if mentor_key not in MENTOR_KEYS or subject_kind not in MENTOR_SUBJECT_KINDS:
        return None
    if subject_kind == "idea":
        if idea_id is None or venture_id is not None:
            return None
        segment = None
    else:
        if venture_id is None or idea_id is not None:
            return None
        if subject_kind == "segment":
            if segment not in SEGMENT_KEYS:
                return None
        elif segment is not None:
            return None
    with get_conn() as conn:
        return conn.execute(
            insert(mentor_challenges).values(
                user_id=user_id, mentor_key=mentor_key, subject_kind=subject_kind,
                idea_id=idea_id, venture_id=venture_id, segment=segment,
                questions_json=questions, dropped_count=max(0, int(dropped or 0)),
                created_at=now(),
            ).returning(mentor_challenges.c.id)
        ).scalar_one()

def list_mentor_challenges(user_id: int, *, idea_id: int = None,
                           venture_id: int = None, subject_kind: str = None,
                           limit: int = 20) -> list:
    """Challenges newest first, always scoped to one subject.

    Scoped rather than "all for this user" on purpose: a challenge is a reading of
    one specific thing, so showing it anywhere else would be showing a question
    about a block next to a different block.

    `subject_kind` is load-bearing on any venture page, not optional decoration:
    a block challenge and a plan challenge share a venture_id, so filtering by
    venture alone put the plan's questions on top of every block panel. The block
    page is asking about one block; anything else is the wrong question.
    """
    query = select(mentor_challenges).where(mentor_challenges.c.user_id == user_id)
    if idea_id is not None:
        query = query.where(mentor_challenges.c.idea_id == idea_id)
    if venture_id is not None:
        query = query.where(mentor_challenges.c.venture_id == venture_id)
    if subject_kind is not None:
        query = query.where(mentor_challenges.c.subject_kind == subject_kind)
    with get_conn() as conn:
        rows = conn.execute(
            query.order_by(mentor_challenges.c.created_at.desc()).limit(limit)
        ).mappings().fetchall()
    return [dict(r) for r in rows]

def latest_mentor_challenge(user_id: int, *, idea_id: int = None,
                            venture_id: int = None, subject_kind: str = None):
    """The most recent challenge for one subject, or None."""
    found = list_mentor_challenges(user_id, idea_id=idea_id, venture_id=venture_id,
                                   subject_kind=subject_kind, limit=1)
    return found[0] if found else None


def action_plan_progress(steps: dict, plan: list) -> dict:
    """How much of the newest plan is done, for the progress line.

    Counts over `plan`, not over `steps`: a step from a superseded strategy is
    still tracked and still in the export, but it is not part of the plan the
    researcher is looking at, and counting it would let progress read as more
    complete than the visible checklist.
    """
    items = join_action_plan(steps, plan)
    done = sum(1 for i in items if i["status"] == "done")
    blocked = sum(1 for i in items if i["status"] == "blocked")
    return {"total": len(items), "done": done, "blocked": blocked,
            "remaining": len(items) - done}


# ---------- account data control ----------

# Columns deliberately left out of the data export, and why. Anything added here
# needs a reason; a credential or provider-supplied text must never be one.
_EXPORT_EXCLUDED_COLUMNS = {
    "users": {
        "password_hash",
        # Not a credential, so the reasoning that excludes password_hash does not
        # apply — but this is internal state of the session control, it tells the
        # account holder nothing, and exporting it invites the next person to
        # treat it as a secret and rotate it. A round-trip import resets it to 1,
        # which is right: imported rows get fresh sessions (specs/09-M0.7).
        "session_epoch",
    },
    # llm_calls.error is provider-supplied text: untrusted by invariant 9, and it
    # can quote request fragments back. Same reasoning as not storing prompt
    # bodies (docs/TECHNICAL.md §5.6).
    "llm_calls": {"error"},
}
# password_reset_tokens is excluded wholesale, not column-wise: token_hash is a
# live credential for the token's lifetime, so an export containing one is a
# takeover path — and it is the user's own row, so nobody would notice the leak.


def export_user_data(user_id: int) -> dict:
    """Everything this user has, as a versioned flat document.

    Flat rather than nested, with row ids intact, so M4.3's round-trip — export,
    then re-import into a clean database — is possible without inventing a
    nesting scheme now.

    Reads go through the ordinary tenant-scoped functions, so row-level security
    applies here too: this cannot be used to export somebody else's account.
    """
    user = get_user_by_id(user_id)
    ventures = list_ventures(user_id)
    venture_ids = [v["id"] for v in ventures]

    blocks, runs, strategies = [], [], []
    steps = []
    for venture in ventures:
        blocks.extend(dict(r) for r in get_segments(venture["id"]))
        runs.extend(dict(r) for r in list_cycles(venture["id"]))
        strategy = get_launch_strategy(venture["id"])
        if strategy:
            strategies.append(dict(strategy))
        # get_action_steps returns a dict keyed by step_key, which would export
        # as a JSON object and lose the ordering. Re-flatten to a list: this
        # document is meant to be re-readable, and M4.3's round-trip import is
        # going to consume it.
        steps.extend(get_action_steps(venture["id"]).values())

    with get_conn() as conn:
        calls = conn.execute(
            select(llm_calls)
            .where(llm_calls.c.user_id == user_id)
            .order_by(llm_calls.c.created_at)
        ).mappings().fetchall()
        challenges = conn.execute(
            select(mentor_challenges)
            .where(mentor_challenges.c.user_id == user_id)
            .order_by(mentor_challenges.c.created_at)
        ).mappings().fetchall()

    def strip(rows, table):
        skip = _EXPORT_EXCLUDED_COLUMNS.get(table, set())
        return [{k: v for k, v in dict(r).items() if k not in skip} for r in rows]

    return {
        # Versioned so M4.3 can extend the shape without guessing what a given
        # download contained.
        "format": "launchloop.user_export.v1",
        "exported_at": now().isoformat(),
        "account": strip([user], "users")[0] if user else None,
        "ideas": strip(list_ideas(user_id), "ideas"),
        "ventures": strip(ventures, "ventures"),
        "blocks": strip(blocks, "bmc_elements"),
        "runs": strip(runs, "cycles"),
        "launch_strategies": strip(strategies, "launch_strategy"),
        # M3.1. action_steps.outcome_note is deliberately NOT excluded the way
        # llm_calls.error is: it is the researcher's own record of what they did,
        # not provider- or model-supplied text, so it is exactly the kind of thing
        # an export exists to hand back.
        "action_steps": strip(steps, "action_steps"),
        # mentor_challenges.questions_json is generated text, but it is generated
        # FROM the researcher's own record and contains no prompt body and no
        # provider text beyond the questions themselves — the same reasoning that
        # keeps action_steps.outcome_note in the export. Not excluded.
        "mentor_challenges": strip(challenges, "mentor_challenges"),
        "llm_calls": strip(calls, "llm_calls"),
    }


def delete_user_account(user_id: int) -> dict:
    """Hard-delete the account. Returns what was removed, for the confirmation page.

    Relies on ON DELETE CASCADE for the content and ON DELETE SET NULL for
    llm_calls, so the audit metadata outlives the account while the research is
    gone. Deliberately *not* a row-by-row delete: the cascade is what keeps the
    tables consistent, and doing it by hand is how orphans get created.

    Authorised by the same RLS policies as everything else — PostgreSQL runs
    referential-integrity actions with the table owner's rights, so the cascade
    clears policies on children whose parent is mid-delete. Probed before relying
    on it rather than assumed; see specs/05-M0.6.
    """
    counts = {}
    for table in ("ideas", "ventures", "bmc_elements", "cycles",
                  "launch_strategy", "action_steps", "mentor_challenges",
                  "password_reset_tokens"):
        # Counted before the delete, under this user's own tenant.
        with get_conn() as conn:
            if table in ("bmc_elements", "cycles", "launch_strategy", "action_steps"):
                # Reached through the venture.
                n = conn.execute(
                    text(f"SELECT count(*) FROM {table} WHERE venture_id IN "
                         f"(SELECT id FROM ventures WHERE user_id = :u)"),
                    {"u": user_id}).scalar_one()
            elif table == "mentor_challenges":
                # user_id is a real column on this one, so it counts directly —
                # the same shape its RLS policy uses. It has to cover idea
                # challenges too, which have no venture at all.
                n = conn.execute(
                    text(f"SELECT count(*) FROM {table} WHERE user_id = :u"),
                    {"u": user_id}).scalar_one()
            else:
                n = conn.execute(
                    text(f"SELECT count(*) FROM {table} WHERE user_id = :u"),
                    {"u": user_id}).scalar_one()
        counts[table] = n

    with get_conn() as conn:
        conn.execute(text("DELETE FROM users WHERE id = :u"), {"u": user_id})
    return counts


# ---------- llm call log / quota ----------

def log_llm_call(user_id, purpose: str, provider: str, model: str, status: str,
                 attempts: int = 1, input_tokens=None, output_tokens=None,
                 latency_ms=None, error: str = None,
                 prompt_version: str = None, prompt_sha256: str = None):
    """Record one call. Metadata only — never the prompt or the response body.

    `prompt_version` / `prompt_sha256` make a call attributable to the template
    that produced it without keeping the prompt, which would embed the
    researcher's pasted material in the one table that outlives the account.
    """
    with get_conn() as conn:
        conn.execute(
            insert(llm_calls).values(
                user_id=user_id, purpose=purpose, provider=provider, model=model,
                status=status, attempts=attempts, input_tokens=input_tokens,
                output_tokens=output_tokens, latency_ms=latency_ms,
                error=(error or "")[:500] or None,
                prompt_version=prompt_version,
                prompt_sha256=prompt_sha256,
                month=current_month(), created_at=now(),
            )
        )


def count_llm_calls_this_month(user_id: int) -> int:
    with get_conn() as conn:
        return conn.execute(
            select(func.count())
            .select_from(llm_calls)
            .where(llm_calls.c.user_id == user_id, llm_calls.c.month == current_month())
        ).scalar_one()


# ---------- login throttling ----------

def record_login_attempt(email: str, ip: str, succeeded: bool):
    with get_conn() as conn:
        conn.execute(
            insert(login_attempts).values(
                email=email, ip=ip, succeeded=succeeded, created_at=now()
            )
        )


def recent_failed_logins(email: str, since: datetime) -> int:
    """Failures against one account, whatever address they came from."""
    with get_conn() as conn:
        return conn.execute(
            select(func.count())
            .select_from(login_attempts)
            .where(
                login_attempts.c.email == email,
                login_attempts.c.succeeded.is_(False),
                login_attempts.c.created_at >= since,
            )
        ).scalar_one()


def recent_failed_logins_from_ip(ip: str, since: datetime) -> int:
    """Failures from one address, whatever accounts they targeted.

    The email-keyed count cannot see an attacker spraying a single password
    across thousands of addresses — from the app's point of view each account has
    had exactly one failed attempt. This is the other half of that problem.
    """
    if not ip:
        return 0
    with get_conn() as conn:
        return conn.execute(
            select(func.count())
            .select_from(login_attempts)
            .where(
                login_attempts.c.ip == ip,
                login_attempts.c.succeeded.is_(False),
                login_attempts.c.created_at >= since,
            )
        ).scalar_one()


def clear_failed_logins(email: str):
    """Reset an account's failure history.

    Called on a successful login. Without it, six typos in a row lock the account
    out for the rest of the window and the next legitimate login is refused —
    which reads as the app being broken rather than as a security feature.
    """
    with get_conn() as conn:
        conn.execute(
            update(login_attempts)
            .where(login_attempts.c.email == email)
            .values(succeeded=True)
        )
