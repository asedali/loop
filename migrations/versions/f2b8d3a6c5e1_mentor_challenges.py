"""mentor_challenges: the storage for founder-playbook challenges (M2.1)

A challenge is a set of questions, never an answer. Nothing about the schema
enforces that — it is enforced by the shape of the model's output and by the guard
in app/llm.py. This table only has to hold the questions, who asked, and what
they were pointed at.

WHY EXACTLY ONE OF idea_id / venture_id IS SET
--------------------------------------------------------------------------------
Phase 1 challenges point at an idea card and have no venture; Phase 2 and Phase 3
challenges point at a venture and have no idea. A single nullable
subject_type/subject_id pair would have been the shorter schema and it would have
been wrong: Postgres can enforce nothing about a polymorphic reference, so a row
claiming subject_kind='segment' while naming an idea's id would be perfectly
legal, and every reader would have to guess which field was authoritative. Three
CHECKs make the shape self-evident instead.

  * exactly_one_subject     — never both, never neither
  * subject_matches_kind    — an 'idea' row owns idea_id, the other kinds own venture_id
  * segment_only_for_segments — and only a 'segment' row names a block, and only a real one

dropped_count is worth its column. The guard discards any line that speaks in the
first person or names the person; a challenge that quietly dropped most of what the
model produced is something the researcher should be able to see, and a rising
count across users is the signal that the prompt needs work.

Revision ID: f2b8d3a6c5e1
Revises: d4a7f1c93e20
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from app.constants import (
    MENTOR_KEYS,
    MENTOR_SUBJECT_KINDS,
    SEGMENT_KEYS,
)

revision = "f2b8d3a6c5e1"
down_revision = "d4a7f1c93e20"
branch_labels = None
depends_on = None

APP_ROLE = "launchloop_app"
TENANT = "NULLIF(current_setting('app.user_id', true), '')::bigint"


def _in_clause(values: list) -> str:
    return ", ".join("'" + v + "'" for v in values)


def upgrade() -> None:
    op.create_table(
        "mentor_challenges",
        sa.Column("id", sa.BigInteger, sa.Identity(), nullable=False),
        sa.Column("user_id", sa.BigInteger, nullable=False),
        sa.Column("mentor_key", sa.Text(), nullable=False),
        sa.Column("subject_kind", sa.Text(), nullable=False),
        sa.Column("idea_id", sa.BigInteger(), nullable=True),
        sa.Column("venture_id", sa.BigInteger(), nullable=True),
        sa.Column("segment", sa.Text(), nullable=True),
        sa.Column("questions_json", postgresql.JSONB(), nullable=False),
        sa.Column(
            "dropped_count", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"],
            name=op.f("fk_mentor_challenges_user_id_users"), ondelete="CASCADE",
        ),
        # CASCADE, not SET NULL: a challenge is a reading of a venture that no
        # longer exists. Leaving the row behind would put it in the account export
        # as an orphan assertion about work nobody can look at any more.
        sa.ForeignKeyConstraint(
            ["idea_id"], ["ideas.id"],
            name=op.f("fk_mentor_challenges_idea_id_ideas"), ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["venture_id"], ["ventures.id"],
            name=op.f("fk_mentor_challenges_venture_id_ventures"), ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_mentor_challenges")),
    )
    op.create_check_constraint(
        "mentor_key", "mentor_challenges",
        f"mentor_key IN ({_in_clause(MENTOR_KEYS)})",
    )
    op.create_check_constraint(
        "subject_kind", "mentor_challenges",
        f"subject_kind IN ({_in_clause(MENTOR_SUBJECT_KINDS)})",
    )
    op.create_check_constraint(
        "exactly_one_subject", "mentor_challenges",
        "num_nonnulls(idea_id, venture_id) = 1",
    )
    op.create_check_constraint(
        "subject_matches_kind", "mentor_challenges",
        "(subject_kind = 'idea' AND idea_id IS NOT NULL AND venture_id IS NULL)"
        " OR (subject_kind <> 'idea' AND venture_id IS NOT NULL AND idea_id IS NULL)",
    )
    op.create_check_constraint(
        "segment_only_for_segments", "mentor_challenges",
        "(subject_kind = 'segment' AND segment IS NOT NULL"
        f"  AND segment IN ({_in_clause(SEGMENT_KEYS)}))"
        " OR (subject_kind <> 'segment' AND segment IS NULL)",
    )
    op.create_index(
        "ix_mentor_challenges_user_id", "mentor_challenges", ["user_id", "created_at"]
    )
    op.create_index(
        "ix_mentor_challenges_venture_id", "mentor_challenges", ["venture_id"]
    )
    op.create_index(
        "ix_mentor_challenges_idea_id", "mentor_challenges", ["idea_id"]
    )

    conn = op.get_bind()
    schema = conn.exec_driver_sql("SELECT current_schema()").scalar_one()

    # DIRECT_USER_ID policy: user_id is a real column here, so this is the simple
    # shape and needs no reference to ventures — which also means it cannot
    # recurse into anything.
    conn.exec_driver_sql(
        'ALTER TABLE "mentor_challenges" ENABLE ROW LEVEL SECURITY')
    conn.exec_driver_sql(
        'ALTER TABLE "mentor_challenges" FORCE ROW LEVEL SECURITY')
    conn.exec_driver_sql(
        'CREATE POLICY "mentor_challenges_tenant_isolation" '
        'ON "mentor_challenges" FOR ALL '
        f"USING (user_id = {TENANT}) WITH CHECK (user_id = {TENANT})"
    )

    # Unguarded on purpose, for the reason recorded in c9e4b2a71d38: Alembic runs
    # each migration in one transaction and PostgreSQL aborts the whole thing on
    # the first failing statement, so a try/except here would not save the
    # migration. Written correctly instead. `IN SCHEMA` is only legal after
    # ALL TABLES, which is why the table is schema-qualified rather than annotated.
    conn.exec_driver_sql(
        f'GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE "{schema}"."mentor_challenges" '
        f'TO "{APP_ROLE}"'
    )


def downgrade() -> None:
    conn = op.get_bind()
    schema = conn.exec_driver_sql("SELECT current_schema()").scalar_one()
    if conn.exec_driver_sql(
        "SELECT 1 FROM pg_roles WHERE rolname = %s", (APP_ROLE,)
    ).first():
        conn.exec_driver_sql(
            f'REVOKE SELECT, INSERT, UPDATE, DELETE ON TABLE "{schema}"."mentor_challenges" '
            f'FROM "{APP_ROLE}"'
        )
    conn.exec_driver_sql(
        'DROP POLICY IF EXISTS "mentor_challenges_tenant_isolation" '
        'ON "mentor_challenges"')
    conn.exec_driver_sql(
        'ALTER TABLE "mentor_challenges" NO FORCE ROW LEVEL SECURITY')
    conn.exec_driver_sql(
        'ALTER TABLE "mentor_challenges" DISABLE ROW LEVEL SECURITY')
    op.drop_index("ix_mentor_challenges_idea_id", table_name="mentor_challenges")
    op.drop_index("ix_mentor_challenges_venture_id", table_name="mentor_challenges")
    op.drop_index("ix_mentor_challenges_user_id", table_name="mentor_challenges")
    op.drop_table("mentor_challenges")