"""Table definitions, engine-free.

Alembic's env.py imports `metadata` from here to autogenerate migrations, so
this module must not import app.db (which builds a live engine at import).
"""
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB

from .constants import (
    IDEA_STATUSES,
    LLM_STATUSES,
    RUN_VERDICTS,
    SEGMENT_KEYS,
    SEGMENT_OUTCOMES,
    SEGMENT_STATUSES,
    VENTURE_STATUSES,
    _in_clause,
)

# Deterministic constraint names so Alembic autogenerate diffs stay stable.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)

users = Table(
    "users",
    metadata,
    Column("id", BigInteger, Identity(), primary_key=True),
    Column("email", Text, nullable=False, unique=True),
    Column("password_hash", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

ideas = Table(
    "ideas",
    metadata,
    Column("id", BigInteger, Identity(), primary_key=True),
    Column("user_id", BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("title", Text, nullable=False),
    Column("commercial_framing", Text, nullable=False),
    Column("strength_signal", Text),
    Column("raw_claims", Text),
    Column("status", Text, nullable=False, server_default=text("'candidate'")),
    Column("created_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        f"status IN ({_in_clause(IDEA_STATUSES)})", name="status"
    ),
    Index("ix_ideas_user_id", "user_id"),
)

ventures = Table(
    "ventures",
    metadata,
    Column("id", BigInteger, Identity(), primary_key=True),
    Column("user_id", BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("idea_id", BigInteger, ForeignKey("ideas.id", ondelete="CASCADE"), nullable=False),
    # 2 = validation loop, 3 = launch strategy
    Column("phase", Integer, nullable=False, server_default=text("2")),
    Column("cycle_count", Integer, nullable=False, server_default=text("0")),
    # Venture-wide backstop only. The real constraint is per-segment: each
    # segment parks on its own bmc_elements.max_cycles, so one exhausted block
    # never pauses the whole venture.
    Column("max_cycles", Integer, nullable=False, server_default=text("27")),
    Column("status", Text, nullable=False, server_default=text("'active'")),
    Column(
        "parent_venture_id",
        BigInteger,
        ForeignKey("ventures.id", ondelete="SET NULL"),
    ),
    Column("pivot_note", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        f"status IN ({_in_clause(VENTURE_STATUSES)})", name="status"
    ),
    Index("ix_ventures_user_id", "user_id"),
    Index("ix_ventures_idea_id", "idea_id"),
    Index("ix_ventures_parent_venture_id", "parent_venture_id"),
)

bmc_elements = Table(
    "bmc_elements",
    metadata,
    Column("id", BigInteger, Identity(), primary_key=True),
    Column(
        "venture_id", BigInteger, ForeignKey("ventures.id", ondelete="CASCADE"), nullable=False
    ),
    # Renamed conceptually to "segment"; the column name is kept because it is
    # already referenced by the dashboard grouping and the phase-3 strategy
    # prompt, and the keys are unchanged.
    Column("element_name", Text, nullable=False),
    Column("status", Text, nullable=False, server_default=text("'untested'")),
    Column("notes", Text),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    # --- per-segment loop state ---
    # Position in the recommended testing order (0-based).
    Column("position", Integer),
    # Display name, e.g. "Customer Segments".
    Column("label", Text),
    # critical | important -- decides what a failure means.
    Column("severity", Text, nullable=False, server_default=text("'important'")),
    # The assumption currently under test. Set before the first run, revised
    # on pivot. This is what the run panel leads with.
    Column("hypothesis", Text),
    # pending | active | passed | failed | parked
    Column("outcome", Text, nullable=False, server_default=text("'pending'")),
    Column("outcome_note", Text),
    # Runs used in this segment, and its cap. The venture-level cycle_count /
    # max_cycles remain as a total and a backstop.
    Column("cycle_count", Integer, nullable=False, server_default=text("0")),
    Column("max_cycles", Integer, nullable=False, server_default=text("3")),
    Column("decided_at", DateTime(timezone=True)),
    # Doubles as the lookup index for get_bmc_elements(), so no separate one.
    UniqueConstraint("venture_id", "element_name"),
    CheckConstraint(
        f"status IN ({_in_clause(SEGMENT_STATUSES)})", name="status"
    ),
    CheckConstraint(
        f"element_name IN ({_in_clause(SEGMENT_KEYS)})", name="element_name"
    ),
    CheckConstraint(
        "severity IN ('critical', 'important')", name="severity"
    ),
    CheckConstraint(
        f"outcome IN ({_in_clause(SEGMENT_OUTCOMES)})", name="outcome"
    ),
    # The decision and the evidence read must not contradict each other.
    CheckConstraint(
        "NOT (outcome = 'passed' AND status <> 'confirmed')", name="passed_needs_confirmed"
    ),
    CheckConstraint(
        "NOT (outcome = 'failed' AND status <> 'disconfirmed')", name="failed_needs_disconfirmed"
    ),
)

cycles = Table(
    "cycles",
    metadata,
    Column("id", BigInteger, Identity(), primary_key=True),
    Column(
        "venture_id", BigInteger, ForeignKey("ventures.id", ondelete="CASCADE"), nullable=False
    ),
    Column("cycle_number", Integer, nullable=False),
    # The segment this run targets. Nullable because cycles created before the
    # per-segment loop existed spanned several blocks at once; those rows are
    # backfilled only when every task in the cycle named the same block.
    Column("segment", Text),
    Column("todos_json", JSONB, nullable=False),
    Column("results_json", JSONB),
    Column("analysis_json", JSONB),
    # The run verdict: pass | iterate | fail | pivot.
    Column("decision", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        f"decision IS NULL OR decision IN ({_in_clause(RUN_VERDICTS)})", name="decision"
    ),
    CheckConstraint(
        f"segment IS NULL OR segment IN ({_in_clause(SEGMENT_KEYS)})", name="segment"
    ),
    # get_current_cycle() orders by cycle_number for one venture.
    Index("ix_cycles_venture_id_cycle_number", "venture_id", "cycle_number"),
    # Per-segment history: "show me every run on the revenue block".
    Index("ix_cycles_venture_id_segment", "venture_id", "segment", "cycle_number"),
)

launch_strategy = Table(
    "launch_strategy",
    metadata,
    Column("id", BigInteger, Identity(), primary_key=True),
    Column(
        "venture_id", BigInteger, ForeignKey("ventures.id", ondelete="CASCADE"), nullable=False
    ),
    Column("funding_matches_json", JSONB, nullable=False),
    Column("gtm_channels_json", JSONB, nullable=False),
    Column("action_plan_json", JSONB, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Index("ix_launch_strategy_venture_id", "venture_id"),
)

# One row per LLM call. Doubles as the observability log and the source of
# truth for the per-user monthly quota in app/quota.py.
llm_calls = Table(
    "llm_calls",
    metadata,
    Column("id", BigInteger, Identity(), primary_key=True),
    Column("user_id", BigInteger, ForeignKey("users.id", ondelete="SET NULL")),
    Column("purpose", Text, nullable=False),
    Column("provider", Text, nullable=False),
    Column("model", Text, nullable=False),
    Column("status", Text, nullable=False),
    Column("attempts", Integer, nullable=False, server_default=text("1")),
    Column("input_tokens", Integer),
    Column("output_tokens", Integer),
    Column("latency_ms", Integer),
    Column("error", Text),
    Column("month", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(f"status IN ({_in_clause(LLM_STATUSES)})", name="status"),
    Index("ix_llm_calls_user_month", "user_id", "month"),
)

# Sliding-window login throttling, so a leaked password can't be brute forced.
login_attempts = Table(
    "login_attempts",
    metadata,
    Column("id", BigInteger, Identity(), primary_key=True),
    Column("email", Text, nullable=False),
    Column("ip", Text),
    Column("succeeded", Boolean, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Index("ix_login_attempts_email", "email", "created_at"),
)
