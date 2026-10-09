"""action-plan step tracking, so Phase 3 has somewhere to record execution

The launch strategy's action plan was a read-only JSONB list: the app could
validate nine blocks against evidence and then hand over a document with no
place to say whether any of it happened. This adds `action_steps`, one row per
plan step, seeded from the newest launch_strategy.

WHY A TABLE AND NOT A JSONB KEY
--------------------------------------------------------------------------------
save_launch_strategy INSERTS a new row and get_launch_strategy takes the newest,
so "Regenerate strategy" already discards the previous plan. Progress stored
inside action_plan_json would therefore be destroyed by a button that sits on the
same page as the progress. Progress is the researcher's record of the real
world; it must outlive the model's opinion of what the plan should be.

  * done/blocked require an outcome note, enforced in the route not the schema.
    An empty note on a 'pending' row is normal and has to stay legal, so a CHECK
    could not express "non-empty unless pending" without also forbidding the
    legitimate case of a note typed before the status was chosen.

WHY step_key IS A HASH OF THE TEXT, NOT AN INDEX
--------------------------------------------------------------------------------
Regeneration yields a different list. Index 3 of the new plan is not index 3 of
the old one, so keying on position would re-label a completion onto a step the
researcher never did — silent data corruption dressed up as progress. The
SHA-256 of the step text makes identity survive regeneration when the wording is
unchanged and correctly produce a fresh pending step when it is not.

RLS
--------------------------------------------------------------------------------
action_steps carries venture_id and is reached through ventures, exactly like
cycles and launch_strategy, so it gets the same EXISTS policy. The role and the
ALTER DEFAULT PRIVILEGES grants were installed by e7a2b4c9d016 and already cover
tables created after it — that migration's comment calls out exactly this case,
and it is why no GRANT appears below.

Note on existing rows: they ARE backfilled, in this migration, from each
venture's newest launch_strategy. The alternative — seeding lazily on first read
of the Phase 3 page — was rejected because app.js fetches pages with GET on every
navigation, so a lazy seed is a write on every page load, and a venture with a
strategy but no steps would render its plan with no controls until the user
happened to trigger it. Backfilling here means there is no window, ever, in which
a generated plan is untrackable.

Revision ID: c9e4b2a71d38
Revises: e7a2b4c9d016
"""
import hashlib
import json
from datetime import datetime, timezone

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from app.constants import ACTION_STEP_STATUSES, MILESTONE_TYPES

revision = "c9e4b2a71d38"
down_revision = "e7a2b4c9d016"
branch_labels = None
depends_on = None

APP_ROLE = "launchloop_app"
TENANT = "NULLIF(current_setting('app.user_id', true), '')::bigint"

launch_strategy = sa.table(
    "launch_strategy",
    sa.column("id", sa.BigInteger),
    sa.column("venture_id", sa.BigInteger),
    sa.column("action_plan_json", postgresql.JSONB),
    sa.column("created_at", sa.DateTime(timezone=True)),
)

action_steps = sa.table(
    "action_steps",
    sa.column("venture_id", sa.BigInteger),
    sa.column("step_key", sa.Text),
    sa.column("step", sa.Text),
    sa.column("milestone_type", sa.Text),
    sa.column("status", sa.Text),
    sa.column("outcome_note", sa.Text),
    sa.column("decided_at", sa.DateTime(timezone=True)),
    sa.column("created_at", sa.DateTime(timezone=True)),
    sa.column("updated_at", sa.DateTime(timezone=True)),
)


def step_key(text: str) -> str:
    """The same hash app/db.py:step_key() computes.

    Duplicated rather than imported because db.py reaches a live engine at
    import time and a migration must not. The test suite pins the two to the
    same value for a set of step strings, so a divergence cannot survive.
    """
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _in_clause(values: list) -> str:
    return ", ".join("'" + v + "'" for v in values)


def upgrade() -> None:
    op.create_table(
        "action_steps",
        sa.Column("id", sa.BigInteger, sa.Identity(), nullable=False),
        sa.Column("venture_id", sa.BigInteger, nullable=False),
        sa.Column("step_key", sa.Text(), nullable=False),
        sa.Column("step", sa.Text(), nullable=False),
        sa.Column("milestone_type", sa.Text(), nullable=False, server_default="pilot"),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("outcome_note", sa.Text(), nullable=True),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["venture_id"],
            ["ventures.id"],
            name=op.f("fk_action_steps_venture_id_ventures"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_action_steps")),
        # Written out longhand rather than derived, so the migration and
        # app/schema.py cannot drift if a CHECK is added to one and not the other.
        sa.UniqueConstraint(
            "venture_id", "step_key", name="uq_action_steps_venture_id_step_key"
        ),
    )
    op.create_check_constraint(
        "status", "action_steps", f"status IN ({_in_clause(ACTION_STEP_STATUSES)})"
    )
    op.create_check_constraint(
        "milestone_type",
        "action_steps",
        f"milestone_type IN ({_in_clause(MILESTONE_TYPES)})",
    )
    op.create_index("ix_action_steps_venture_id", "action_steps", ["venture_id"])

    _backfill(op.get_bind())

    conn = op.get_bind()
    schema = conn.exec_driver_sql("SELECT current_schema()").scalar_one()

    # Same shape as cycles/launch_strategy: no user_id of its own, so the
    # policy refers to ventures. That costs no recursion, because ventures'
    # policy does not refer back.
    conn.exec_driver_sql(
        f'ALTER TABLE "action_steps" ENABLE ROW LEVEL SECURITY')
    conn.exec_driver_sql(
        f'ALTER TABLE "action_steps" FORCE ROW LEVEL SECURITY')
    exists = (
        "EXISTS (SELECT 1 FROM ventures v "
        "WHERE v.id = action_steps.venture_id AND v.user_id = "
        f"{TENANT})"
    )
    conn.exec_driver_sql(
        f'CREATE POLICY "action_steps_tenant_isolation" ON "action_steps" FOR ALL '
        f"USING ({exists}) WITH CHECK ({exists})"
    )

    # Redundant with the ALTER DEFAULT PRIVILEGES in e7a2b4c9d016, which already
    # covers tables created after it — this table is exactly that case. Stated
    # rather than assumed, and repeated here so a deployment that somehow lacks
    # the default grant still gets an explicit one.
    #
    # This is NOT wrapped in try/except, and the reason matters more than the
    # statement. Alembic runs each migration in one transaction, and PostgreSQL
    # aborts the whole transaction on the first failing statement — so a
    # "best-effort" GRANT guarded by except would still take the migration down
    # with it at the very next statement. Being best-effort here would require
    # a separate connection and a SAVEPOINT, which is more machinery than this
    # warrants: the statement is written correctly instead.
    #
    # `IN SCHEMA` is only legal after ALL TABLES, never after a single named
    # table, which is why the name is schema-qualified rather than annotated.
    conn.exec_driver_sql(
        f'GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE "{schema}"."action_steps" '
        f'TO "{APP_ROLE}"'
    )


def _backfill(conn) -> None:
    """Seed a row per step of each venture's newest strategy, all `pending`.

    Only the newest strategy per venture, matching get_launch_strategy's
    ORDER BY created_at DESC LIMIT 1: seeding rows for plans the app will never
    show again would put un-deletable orphans in the export.
    """
    newest = sa.select(
        launch_strategy.c.venture_id,
        launch_strategy.c.action_plan_json,
    ).order_by(launch_strategy.c.created_at.desc())

    # A Python timestamp rather than sa.func.now(): psycopg cannot adapt a SQL
    # function as a bound parameter inside an executemany INSERT.
    stamp = datetime.now(timezone.utc)
    rows = []
    for venture_id, plan in conn.execute(newest):
        if not isinstance(plan, list):
            continue
        for item in plan:
            # Defensive against a hand-edited row: the step text is what the key
            # is derived from, so a malformed entry is skipped rather than
            # seeded with a key that cannot be reproduced.
            text = item.get("step") if isinstance(item, dict) else None
            if not isinstance(text, str) or not text.strip():
                continue
            milestone = (item.get("milestone_type")
                         if isinstance(item, dict) else None)
            rows.append({
                "venture_id": venture_id,
                "step_key": step_key(text),
                "step": text.strip(),
                "milestone_type": milestone if milestone in MILESTONE_TYPES else "pilot",
                "status": "pending",
                "created_at": stamp,
                "updated_at": stamp,
            })
    if not rows:
        return
    # ON CONFLICT DO NOTHING rather than a bare insert: two identical step
    # strings in one plan collapse to one row, and a repeated migration run must
    # not fail on the unique constraint.
    conn.execute(
        postgresql.insert(action_steps).values(rows).on_conflict_do_nothing(
            index_elements=["venture_id", "step_key"]
        )
    )


def downgrade() -> None:
    conn = op.get_bind()
    schema = conn.exec_driver_sql("SELECT current_schema()").scalar_one()
    if conn.exec_driver_sql(
        "SELECT 1 FROM pg_roles WHERE rolname = %s", (APP_ROLE,)
    ).first():
        # Also unguarded, for the same reason as the upgrade: a caught error
        # would not save the transaction. REVOKE on a table that still exists
        # cannot fail for any reason a downgrade would survive anyway.
        conn.exec_driver_sql(
            f'REVOKE SELECT, INSERT, UPDATE, DELETE ON TABLE "{schema}"."action_steps" '
            f'FROM "{APP_ROLE}"'
        )
    conn.exec_driver_sql(
        'DROP POLICY IF EXISTS "action_steps_tenant_isolation" ON "action_steps"')
    conn.exec_driver_sql(
        'ALTER TABLE "action_steps" NO FORCE ROW LEVEL SECURITY')
    conn.exec_driver_sql(
        'ALTER TABLE "action_steps" DISABLE ROW LEVEL SECURITY')
    op.drop_index("ix_action_steps_venture_id", table_name="action_steps")
    op.drop_constraint(
        op.f("ck_action_steps_milestone_type"), "action_steps", type_="check"
    )
    op.drop_constraint(op.f("ck_action_steps_status"), "action_steps", type_="check")
    op.drop_table("action_steps")