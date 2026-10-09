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
    ACTION_STEP_STATUSES,
    IDEA_STATUSES,
    LLM_STATUSES,
    MENTOR_KEYS,
    MENTOR_SUBJECT_KINDS,
    MILESTONE_TYPES,
    RUN_VERDICTS,
    SEGMENT_KEYS,
    SEGMENT_OUTCOMES,
    SEGMENT_STATUSES,
    TOKEN_PURPOSES,
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
    # Bumped by db.set_password(), so every session cookie minted before the
    # change stops being honoured (M0.7). This is what gives a stateless signed
    # cookie a revocation story without a sessions table.
    #
    # NOT a secret and deliberately not treated as one: the security comes from
    # the cookie still being signed, and the epoch only decides whether an
    # already-valid cookie is still current. Rotating it does nothing an attacker
    # with only this column could not already do.
    #
    # The comparison costs no query, because get_current_user() already loads
    # this row on every request to install the RLS tenant.
    Column("session_epoch", Integer, nullable=False, server_default=text("1")),
    # Null until the address is proven. Not backfilled to created_at for existing
    # users: that would assert a verification that never happened.
    Column("email_verified_at", DateTime(timezone=True)),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

# Password resets AND email verification links. One table because both need the
# same primitive; `purpose` keeps the flows apart. Only the SHA-256 of the token
# is stored, so a database dump cannot be used to take over an account.
password_reset_tokens = Table(
    "password_reset_tokens",
    metadata,
    Column("id", BigInteger, Identity(), primary_key=True),
    Column("user_id", BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("purpose", Text, nullable=False),
    # Redemption is a lookup by hash, so this is UNIQUE and indexed.
    Column("token_hash", Text, nullable=False, unique=True),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    # Set on redemption. This is what makes a token single-use.
    Column("used_at", DateTime(timezone=True)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(f"purpose IN ({_in_clause(TOKEN_PURPOSES)})", name="purpose"),
    Index("ix_password_reset_tokens_user_id", "user_id", "purpose"),
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

# One row per launch-plan step, seeded from the newest launch_strategy's
# action_plan_json (M3.1).
#
# A separate table rather than a "completed" key inside action_plan_json, for two
# reasons. save_launch_strategy INSERTS a new row and get_launch_strategy takes
# the newest, so "Regenerate strategy" already throws the old plan away —
# progress stored inside the strategy row would be destroyed by a button on the
# same page. And a real table is picked up by export_user_data() and
# delete_user_account() without either being taught about a new JSONB key.
#
# step_key is the SHA-256 of the step text, NOT the array index. Regeneration
# produces a *different* list, so index 3 of the new plan is not index 3 of the
# old one; keying on position would silently re-label a completion onto a step
# the researcher never did. Same wording -> same key -> the completion survives.
# Differently-worded step -> new key -> correctly still pending.
#
# The cost of that choice, stated: two steps with identical wording share a row,
# so ticking one ticks both. That is the correct behaviour — they are the same
# instruction — and it is the price of not minting a synthetic id the model
# never sees.
action_steps = Table(
    "action_steps",
    metadata,
    Column("id", BigInteger, Identity(), primary_key=True),
    Column(
        "venture_id", BigInteger, ForeignKey("ventures.id", ondelete="CASCADE"), nullable=False
    ),
    Column("step_key", Text, nullable=False),
    # The step's text, denormalised so a completion stays readable after the
    # plan it came from has been regenerated out of existence.
    Column("step", Text, nullable=False),
    Column(
        "milestone_type",
        Text,
        nullable=False,
        server_default=text("'pilot'"),
    ),
    Column("status", Text, nullable=False, server_default=text("'pending'")),
    # What actually happened. Required by the route for 'done' and 'blocked',
    # which is why nothing in the schema can enforce it — an empty-string note on
    # a 'pending' row is normal and must stay legal.
    Column("outcome_note", Text),
    Column("decided_at", DateTime(timezone=True)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(f"status IN ({_in_clause(ACTION_STEP_STATUSES)})", name="status"),
    CheckConstraint(f"milestone_type IN ({_in_clause(MILESTONE_TYPES)})", name="milestone_type"),
    UniqueConstraint("venture_id", "step_key"),
    Index("ix_action_steps_venture_id", "venture_id"),
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
    # --- prompt provenance, deliberately not the prompt itself ---
    # Which template produced this call (constants.PROMPT_VERSIONS), so editing a
    # prompt does not silently re-label historical results. The bodies are NOT
    # stored: they embed the researcher's pasted material verbatim, and these rows
    # outlive account deletion (user_id is ON DELETE SET NULL).
    Column("prompt_version", Text),
    Column("prompt_sha256", Text),
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
    # Throttling counts failures by source address too, so one attacker spraying a
    # single password across many accounts is visible.
    Index("ix_login_attempts_ip", "ip", "created_at"),
)

# One founder-playbook challenge (M2.1). Questions only, by construction: the
# output has no field an answer could go in, and nothing in the write path calls
# apply_verdict / set_segment_outcome / update_venture.
#
# user_id is a real column rather than reached through a venture, so the RLS
# policy is the simple `user_id = <tenant>` shape (DIRECT_USER_ID) rather than
# the EXISTS-over-ventures form. A Phase 1 challenge points at an idea and has no
# venture at all, so there was nothing to refer to.
mentor_challenges = Table(
    "mentor_challenges",
    metadata,
    Column("id", BigInteger, Identity(), primary_key=True),
    Column("user_id", BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("mentor_key", Text, nullable=False),
    # What the challenge is pointed at. 'idea' carries idea_id, 'segment' and
    # 'plan' carry venture_id (+ segment for 'segment'). Checked below rather than
    # inferred, because a polymorphic subject_type/subject_id pair would be
    # referentially unenforceable: nothing in Postgres can stop a row claiming to
    # be a 'segment' while naming an idea's id.
    Column("subject_kind", Text, nullable=False),
    Column("idea_id", BigInteger, ForeignKey("ideas.id", ondelete="CASCADE")),
    Column("venture_id", BigInteger, ForeignKey("ventures.id", ondelete="CASCADE")),
    Column("segment", Text),
    Column("questions_json", JSONB, nullable=False),
    # How many lines the guard in llm.py dropped for speaking in first person or
    # naming the person. Stored because a challenge that silently discarded most
    # of its output is something the researcher should be able to see, and because
    # a rising count is the signal that the prompt needs work.
    Column("dropped_count", Integer, nullable=False, server_default=text("0")),
    Column("created_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(f"mentor_key IN ({_in_clause(MENTOR_KEYS)})", name="mentor_key"),
    CheckConstraint(
        f"subject_kind IN ({_in_clause(MENTOR_SUBJECT_KINDS)})", name="subject_kind"
    ),
    # Exactly one owner: an idea challenge OR a venture challenge, never both and
    # never neither.
    CheckConstraint(
        "num_nonnulls(idea_id, venture_id) = 1", name="exactly_one_subject"
    ),
    # The subject and its kind have to agree, or `segment_kind`/`idea_id` could
    # disagree and every read would need to guess which was authoritative.
    CheckConstraint(
        "(subject_kind = 'idea' AND idea_id IS NOT NULL AND venture_id IS NULL)"
        " OR (subject_kind <> 'idea' AND venture_id IS NOT NULL AND idea_id IS NULL)",
        name="subject_matches_kind",
    ),
    # Only a segment challenge names a block, and only a valid one.
    CheckConstraint(
        "(subject_kind = 'segment' AND segment IS NOT NULL"
        f"  AND segment IN ({_in_clause(SEGMENT_KEYS)}))"
        " OR (subject_kind <> 'segment' AND segment IS NULL)",
        name="segment_only_for_segments",
    ),
    Index("ix_mentor_challenges_user_id", "user_id", "created_at"),
    Index("ix_mentor_challenges_venture_id", "venture_id"),
    Index("ix_mentor_challenges_idea_id", "idea_id"),
)
