"""
Postgres database layer for LaunchLoop, via SQLAlchemy Core over psycopg3.

No ORM: the table objects in app.schema are used to build queries, and every
function returns either a RowMapping, a list of them, a scalar, or an int.
Callers index rows with row["col"] and dict(row), which RowMapping supports.

The engine and its pool are created once per process and reused. DDL lives in
Alembic (migrations/), not here — init_db() just runs the migrations.
"""
import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from urllib.parse import quote, unquote

from sqlalchemy import case, create_engine, func, insert, select, text, update
from sqlalchemy.engine import Engine

from . import config
from .constants import (
    ALLOWED_SEGMENT_FIELDS,
    ALLOWED_VENTURE_FIELDS,
    IDEA_STATUSES,
    LLM_PURPOSES,
    LLM_STATUSES,
    RUN_VERDICTS,
    SEGMENTS,
    SEGMENT_KEYS,
    SEGMENT_OUTCOMES,
    SEGMENT_RESOLVED,
    SEGMENT_STATUSES,
    VENTURE_STATUSES,
)
from .schema import (
    bmc_elements,
    cycles,
    ideas,
    launch_strategy,
    llm_calls,
    login_attempts,
    users,
    ventures,
)

__all__ = [
    "ALLOWED_VENTURE_FIELDS", "IDEA_STATUSES", "LLM_PURPOSES", "LLM_STATUSES",
    "SEGMENT_KEYS", "SEGMENT_OUTCOMES", "SEGMENT_RESOLVED",
    "SEGMENT_STATUSES", "RUN_VERDICTS",
    "VENTURE_STATUSES", "get_conn", "get_engine", "init_db", "ping",
    "reset_engine", "set_engine",
    # users
    "create_user", "get_user_by_email", "get_user_by_id",
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
        _engine = create_engine(
            database_url(),
            pool_pre_ping=True,      # drop connections killed by the pooler
            pool_recycle=1800,       # stay under the pooler's idle timeout
            pool_size=5,
            max_overflow=5,
            future=True,
        )
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
    tests have the same shape they had against sqlite3."""
    conn = get_engine().connect()
    trans = conn.begin()
    try:
        yield conn
        trans.commit()
    except Exception:
        trans.rollback()
        raise
    finally:
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

def create_user(email: str, password_hash: str):
    with get_conn() as conn:
        return conn.execute(
            insert(users).values(
                email=email, password_hash=password_hash, created_at=now()
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
    with get_conn() as conn:
        return conn.execute(
            insert(launch_strategy).values(
                venture_id=venture_id, funding_matches_json=funding_matches,
                gtm_channels_json=gtm_channels, action_plan_json=action_plan,
                created_at=now(),
            ).returning(launch_strategy.c.id)
        ).scalar_one()


def get_launch_strategy(venture_id: int):
    with get_conn() as conn:
        return conn.execute(
            select(launch_strategy)
            .where(launch_strategy.c.venture_id == venture_id)
            .order_by(launch_strategy.c.created_at.desc())
            .limit(1)
        ).mappings().fetchone()


# ---------- llm call log / quota ----------

def log_llm_call(user_id, purpose: str, provider: str, model: str, status: str,
                 attempts: int = 1, input_tokens=None, output_tokens=None,
                 latency_ms=None, error: str = None):
    with get_conn() as conn:
        conn.execute(
            insert(llm_calls).values(
                user_id=user_id, purpose=purpose, provider=provider, model=model,
                status=status, attempts=attempts, input_tokens=input_tokens,
                output_tokens=output_tokens, latency_ms=latency_ms,
                error=(error or "")[:500] or None,
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
