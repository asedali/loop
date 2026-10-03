"""per-segment validation loop

Adds the per-segment loop state to bmc_elements (hypothesis, outcome, severity,
per-segment cycle cap), scopes cycles to a segment, and backfills:

  * the four blocks the old loop never tracked, plus metadata on all nine
  * existing ventures that already have rows
  * cycles.segment, where every task in a legacy cycle named the same block
  * ventures.max_cycles, raised from 6 to the nine-segment backstop

The CHECK constraints have to be swapped before the backfill, because the old
element_name CHECK only allows the five blocks the previous loop knew about and
would reject the four new rows.

Revision ID: b3d7e91c4a02
Revises: a1c9f0e2b7d1
"""
import sqlalchemy as sa
from alembic import op
from datetime import datetime, timezone

from app.constants import SEGMENTS, VENTURE_RUN_BACKSTOP

revision = "b3d7e91c4a02"
down_revision = "a1c9f0e2b7d1"
branch_labels = None
depends_on = None

bmc_elements = sa.table(
    "bmc_elements",
    sa.column("id", sa.BigInteger),
    sa.column("venture_id", sa.BigInteger),
    sa.column("element_name", sa.Text),
    sa.column("position", sa.Integer),
    sa.column("label", sa.Text),
    sa.column("severity", sa.Text),
    sa.column("max_cycles", sa.Integer),
    sa.column("updated_at", sa.DateTime(timezone=True)),
)

ventures = sa.table(
    "ventures",
    sa.column("id", sa.BigInteger),
    sa.column("max_cycles", sa.Integer),
)


def upgrade() -> None:
    # --- 1. widen the CHECK constraints before touching any data ---
    op.drop_constraint(op.f("ck_bmc_elements_element_name"), "bmc_elements", type_="check")
    op.drop_constraint(op.f("ck_bmc_elements_status"), "bmc_elements", type_="check")
    op.drop_constraint(op.f("ck_cycles_decision"), "cycles", type_="check")

    # Translate the legacy cycle-level decisions into run verdicts. This has to
    # happen while no decision CHECK exists: existing rows hold 'persevere' and
    # 'kill', which the new constraint would reject outright. 'persevere' maps
    # to 'iterate' because both mean "not settled, loop again".
    bind = op.get_bind()
    bind.execute(
        sa.text(
            "UPDATE cycles SET decision = CASE decision"
            "  WHEN 'persevere' THEN 'iterate'"
            "  WHEN 'kill'      THEN 'fail'"
            " ELSE decision END"
            " WHERE decision IN ('persevere', 'kill')"
        )
    )

    keys = ", ".join(f"'{s['key']}'" for s in SEGMENTS)
    statuses = "'untested', 'mixed', 'confirmed', 'disconfirmed'"
    outcomes = "'pending', 'active', 'passed', 'failed', 'parked'"
    verdicts = "'pass', 'iterate', 'fail', 'pivot'"

    op.create_check_constraint("element_name", "bmc_elements", f"element_name IN ({keys})")
    op.create_check_constraint("status", "bmc_elements", f"status IN ({statuses})")
    op.create_check_constraint("decision", "cycles", f"decision IS NULL OR decision IN ({verdicts})")

    # --- 2. per-segment loop state ---
    op.add_column("bmc_elements", sa.Column("position", sa.Integer()))
    op.add_column("bmc_elements", sa.Column("label", sa.Text()))
    op.add_column(
        "bmc_elements",
        sa.Column("severity", sa.Text(), nullable=False, server_default=sa.text("'important'")),
    )
    op.add_column("bmc_elements", sa.Column("hypothesis", sa.Text()))
    op.add_column(
        "bmc_elements",
        sa.Column("outcome", sa.Text(), nullable=False, server_default=sa.text("'pending'")),
    )
    op.add_column("bmc_elements", sa.Column("outcome_note", sa.Text()))
    op.add_column(
        "bmc_elements",
        sa.Column("cycle_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )
    op.add_column(
        "bmc_elements",
        sa.Column("max_cycles", sa.Integer(), nullable=False, server_default=sa.text("3")),
    )
    op.add_column("bmc_elements", sa.Column("decided_at", sa.DateTime(timezone=True)))

    op.add_column("cycles", sa.Column("segment", sa.Text()))

    op.create_check_constraint("severity", "bmc_elements", "severity IN ('critical', 'important')")
    op.create_check_constraint("outcome", "bmc_elements", f"outcome IN ({outcomes})")
    op.create_check_constraint("segment", "cycles", f"segment IS NULL OR segment IN ({keys})")
    # The decision and the evidence read must not contradict each other.
    op.create_check_constraint(
        "passed_needs_confirmed",
        "bmc_elements",
        "NOT (outcome = 'passed' AND status <> 'confirmed')",
    )
    op.create_check_constraint(
        "failed_needs_disconfirmed",
        "bmc_elements",
        "NOT (outcome = 'failed' AND status <> 'disconfirmed')",
    )
    op.create_index(
        "ix_cycles_venture_id_segment", "cycles", ["venture_id", "segment", "cycle_number"]
    )

    # --- 3. backfill ---
    # Infer each block's outcome from the evidence the old loop recorded, so
    # validation work already done does not have to be repeated: a block the old
    # loop confirmed counts as passed, one it disconfirmed becomes a gap.
    # Built as raw SQL from SEGMENTS rather than with sa.case(), whose column
    # references do not resolve inside an UPDATE ... SET clause. Generating it
    # keeps constants.py the single source of truth for the segment list.
    def _sql_list(values):
        return ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)

    label_case = " ".join(
        f"WHEN '{s['key']}' THEN '{s['label'].replace(chr(39), chr(39) * 2)}'"
        for s in SEGMENTS
    )
    position_case = " ".join(
        f"WHEN '{s['key']}' THEN {i}" for i, s in enumerate(SEGMENTS)
    )
    critical = _sql_list([s["key"] for s in SEGMENTS if s["severity"] == "critical"])
    bind.execute(
        sa.text(f"""
            UPDATE bmc_elements SET
                position = CASE element_name {position_case} END,
                label    = CASE element_name {label_case} END,
                severity = CASE WHEN element_name IN ({critical})
                          THEN 'critical' ELSE 'important' END,
                outcome  = CASE
                    WHEN status = 'confirmed'    THEN 'passed'
                    WHEN status = 'disconfirmed' THEN 'parked'
                    ELSE 'pending' END,
                decided_at = CASE
                    WHEN status IN ('confirmed', 'disconfirmed') THEN now()
                    ELSE NULL END
            WHERE outcome = 'pending'
              AND (status IN ('confirmed', 'disconfirmed')
                   OR position IS NULL OR label IS NULL)
        """)
    )

    # Insert the blocks the old loop never tracked. bmc_elements.id is an
    # identity column, so omitting it here avoids having to resync sequences.
    existing_ventures = [r[0] for r in bind.execute(sa.select(ventures.c.id))]
    known = {s["key"] for s in SEGMENTS}
    # A Python timestamp rather than sa.func.now(): psycopg cannot adapt a SQL
    # function as a bound parameter inside an executemany INSERT.
    stamp = datetime.now(timezone.utc)
    rows = []
    for vid in existing_ventures:
        present = {
            r[0]
            for r in bind.execute(
                sa.select(bmc_elements.c.element_name).where(bmc_elements.c.venture_id == vid)
            )
        }
        for idx, seg in enumerate(SEGMENTS):
            if seg["key"] in present:
                continue
            rows.append(
                {
                    "venture_id": vid,
                    "element_name": seg["key"],
                    "status": "untested",
                    "notes": "",
                    "updated_at": stamp,
                    "position": idx,
                    "label": seg["label"],
                    "severity": seg["severity"],
                    "outcome": "pending",
                    "cycle_count": 0,
                    "max_cycles": 3,
                }
            )
    if rows:
        bind.execute(bmc_elements.insert(), rows)

    # A pre-segment cycle only tells us its block when every task named the
    # same one. Anything else stays NULL rather than being guessed at.
    bind.execute(
        sa.text(
            """
            UPDATE cycles c SET segment = sub.target
            FROM (
                SELECT cy.id AS id,
                       MIN(t.task->>'target_element') AS target,
                       COUNT(DISTINCT t.task->>'target_element') AS n
                FROM cycles cy,
                     LATERAL jsonb_array_elements(COALESCE(cy.todos_json, '[]'::jsonb)) AS t(task)
                GROUP BY cy.id
            ) sub
            WHERE c.id = sub.id
              AND sub.n = 1
              AND sub.target = ANY (:keys)
            """
        ).bindparams(sa.bindparam("keys", value=sorted(known), expanding=False)),
    )

    # The venture-level cap is now a backstop, not the real constraint.
    bind.execute(
        ventures.update().where(ventures.c.max_cycles == 6).values(
            max_cycles=VENTURE_RUN_BACKSTOP
        )
    )
    op.alter_column(
        "ventures",
        "max_cycles",
        server_default=sa.text(str(VENTURE_RUN_BACKSTOP)),
    )


def downgrade() -> None:
    # op.drop_constraint re-applies the table's naming convention to whatever
    # name it is given, so every drop has to be marked as already-final.
    op.drop_index("ix_cycles_venture_id_segment", table_name="cycles")
    for name in ("ck_bmc_elements_passed_needs_confirmed",
                 "ck_bmc_elements_failed_needs_disconfirmed",
                 "ck_bmc_elements_outcome",
                 "ck_bmc_elements_severity",
                 "ck_cycles_segment",
                 "ck_cycles_decision",
                 "ck_bmc_elements_element_name",
                 "ck_bmc_elements_status"):
        table = "cycles" if name.startswith("ck_cycles") else "bmc_elements"
        op.drop_constraint(op.f(name), table, type_="check")

    # Undo the decision translation while no CHECK is in place, or restoring
    # the old constraint would reject the rows this migration wrote.
    op.get_bind().execute(
        sa.text(
            "UPDATE cycles SET decision = CASE decision"
            "  WHEN 'pass'    THEN 'persevere'"
            "  WHEN 'iterate' THEN 'persevere'"
            "  WHEN 'fail'    THEN 'kill'"
            " ELSE decision END"
            " WHERE decision IN ('pass', 'iterate', 'fail')"
        )
    )

    op.drop_column("cycles", "segment")
    for col in ("decided_at", "max_cycles", "cycle_count", "outcome_note",
                "outcome", "hypothesis", "label", "position", "severity"):
        op.drop_column("bmc_elements", col)

    # Only the five blocks the old loop tracked can be represented afterwards.
    op.get_bind().execute(
        sa.text(
            "DELETE FROM bmc_elements WHERE element_name NOT IN "
            "('customer', 'problem', 'value_prop', 'revenue', 'channel')"
        )
    )

    op.create_check_constraint(
        "element_name", "bmc_elements",
        "element_name IN ('customer', 'problem', 'value_prop', 'revenue', 'channel')",
    )
    op.create_check_constraint(
        "status", "bmc_elements",
        "status IN ('untested', 'mixed', 'confirmed', 'disconfirmed')",
    )
    op.create_check_constraint(
        "decision", "cycles",
        "decision IS NULL OR decision IN ('persevere', 'pivot', 'kill')",
    )
    # Only the default is reverted, not the column value: a venture that has
    # run more than 6 cycles since the upgrade would end up with
    # cycle_count > max_cycles and be wrongly treated as capped.
    op.alter_column("ventures", "max_cycles", server_default=sa.text("6"))
