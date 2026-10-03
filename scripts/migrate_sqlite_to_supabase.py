#!/usr/bin/env python
"""One-shot ETL: copy a launchloop.db SQLite file into Postgres.

Preserves primary keys, because the ids are referenced across tables and by
anything the user has bookmarked. Re-runnable: every insert is an upsert that
does nothing on a primary-key conflict, so a run interrupted halfway can simply
be repeated.

    # see what would happen
    python scripts/migrate_sqlite_to_supabase.py --dry-run

    # do it (reads DATABASE_URL from the environment, or pass --database-url)
    python scripts/migrate_sqlite_to_supabase.py

Run `alembic upgrade head` first — this script assumes the target schema exists
and does not create it.
"""
import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402
from sqlalchemy import bindparam, create_engine, text, update  # noqa: E402
from sqlalchemy.dialects.postgresql import insert as pg_insert  # noqa: E402

# This script runs outside app.main, so it has to load .env itself or a
# correctly-filled DATABASE_URL there will look unset.
load_dotenv(PROJECT_ROOT / ".env")

from app.schema import metadata, ventures as ventures_table  # noqa: E402

DEFAULT_SQLITE = "launchloop.db"

# Insertion order matters: parents before children. ventures is handled in two
# passes because it self-references through parent_venture_id, and a child can
# precede its parent in the source file's row order.
SIMPLE_TABLES = [
    "users",
    "ideas",
    "ventures",
    "bmc_elements",
    "cycles",
    "launch_strategy",
    "llm_calls",
    "login_attempts",
]

# SQLite stored these as TEXT holding JSON; the Postgres columns are JSONB.
JSON_COLUMNS = {
    "cycles": ["todos_json", "results_json", "analysis_json"],
    "launch_strategy": ["funding_matches_json", "gtm_channels_json", "action_plan_json"],
}

TIMESTAMP_COLUMNS = {
    "users": ["created_at"],
    "ideas": ["created_at"],
    "ventures": ["created_at"],
    "bmc_elements": ["updated_at"],
    "cycles": ["created_at"],
    "launch_strategy": ["created_at"],
    "llm_calls": ["created_at"],
    "login_attempts": ["created_at"],
}

# SQLite INTEGER 0/1; Postgres BOOLEAN.
BOOLEAN_COLUMNS = {"login_attempts": ["succeeded"]}


def _parse_ts(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def _convert(table: str, row: dict) -> dict:
    out = dict(row)
    for col in JSON_COLUMNS.get(table, []):
        raw = out.get(col)
        if isinstance(raw, str):
            out[col] = json.loads(raw) if raw else None
    for col in TIMESTAMP_COLUMNS.get(table, []):
        out[col] = _parse_ts(out.get(col))
    for col in BOOLEAN_COLUMNS.get(table, []):
        if out.get(col) is not None:
            out[col] = bool(out[col])
    return out


def read_source(path: str) -> dict:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        tables = {
            r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        data = {}
        for table in SIMPLE_TABLES:
            if table not in tables:
                print(f"  ! {table} missing from the SQLite file, skipping")
                continue
            rows = [_convert(table, dict(r)) for r in conn.execute(f"SELECT * FROM {table}")]
            data[table] = rows
        return data
    finally:
        conn.close()


def sync_sequences(engine) -> None:
    """Explicit ids were inserted, so each identity sequence still points at 1.
    Without this the first user created in the app would collide with id 1."""
    with engine.begin() as conn:
        for table in SIMPLE_TABLES:
            row = conn.execute(
                text("SELECT COALESCE(MAX(id), 0) FROM " + table)
            ).scalar_one()
            if row == 0:
                continue
            seq = conn.execute(
                text(
                    "SELECT pg_get_serial_sequence(:t, 'id')"
                ),
                {"t": table},
            ).scalar_one()
            if seq:
                conn.execute(text(f"SELECT setval(:seq, :val, true)"), {"seq": seq, "val": row})
                print(f"  = sequence {table}.id_seq -> {row}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sqlite", default=os.environ.get("SQLITE_SOURCE", DEFAULT_SQLITE))
    ap.add_argument("--database-url", default=None,
                    help="Defaults to $DATABASE_URL")
    ap.add_argument("--dry-run", action="store_true",
                    help="Read and report only; write nothing.")
    args = ap.parse_args()

    if not os.path.exists(args.sqlite):
        print(f"error: {args.sqlite} not found", file=sys.stderr)
        return 1

    data = read_source(args.sqlite)
    print(f"Read {args.sqlite}:")
    for table, rows in data.items():
        print(f"  {table:18} {len(rows)} rows")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    url = args.database_url or os.environ.get("DATABASE_URL")
    if not url:
        print("error: no DATABASE_URL. Set it or pass --database-url.", file=sys.stderr)
        return 1

    engine = create_engine(url, pool_pre_ping=True)
    tables = dict(metadata.tables)

    try:
        with engine.begin() as conn:
            present = {
                r[0] for r in conn.execute(text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = current_schema()"
                ))
            }
            missing = [t for t in data if t not in present]
            if missing:
                print(f"error: tables missing in the target: {', '.join(missing)}",
                      file=sys.stderr)
                print("Run: alembic upgrade head", file=sys.stderr)
                return 1

            for table in SIMPLE_TABLES:
                rows = data.get(table)
                if not rows:
                    continue
                stmt = pg_insert(tables[table]).on_conflict_do_nothing()
                result = conn.execute(stmt, rows)
                print(f"  + {table:18} inserted {result.rowcount} of {len(rows)}")

            # Self-referencing parent links, now that every venture exists.
            parents = [
                {"id": r["id"], "parent_venture_id": r["parent_venture_id"]}
                for r in data.get("ventures", [])
                if r.get("parent_venture_id") is not None
            ]
            if parents:
                link = (
                    update(ventures_table)
                    .where(ventures_table.c.id == bindparam("id"))
                    .values(parent_venture_id=bindparam("parent_venture_id"))
                )
                conn.execute(link, parents)
                print(f"  + ventures.parent_venture_id  linked {len(parents)}")

        sync_sequences(engine)
        print("\nDone. Verify with:")
        print("  python -c \"from app import db; print(db.ping())\"")
    finally:
        engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
