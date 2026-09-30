"""
SQLite database layer for LaunchLoop MVP.

Uses plain sqlite3 (no ORM) with a small helper wrapper. The DB file lives
on the server's persistent disk (see fly.toml volume mount) — this only
works with a long-running process, never a serverless/stateless host.
"""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = os.environ.get("LAUNCHLOOP_DB_PATH", "/data/launchloop.db")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ideas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    commercial_framing TEXT NOT NULL,
    strength_signal TEXT,
    raw_claims TEXT,
    status TEXT NOT NULL DEFAULT 'candidate', -- candidate | selected | rejected
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ventures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    idea_id INTEGER NOT NULL REFERENCES ideas(id) ON DELETE CASCADE,
    phase INTEGER NOT NULL DEFAULT 2, -- 2 = validation loop, 3 = launch strategy
    cycle_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active', -- active | validated | killed | paused
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bmc_elements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    venture_id INTEGER NOT NULL REFERENCES ventures(id) ON DELETE CASCADE,
    element_name TEXT NOT NULL, -- customer | problem | value_prop | channel | revenue
    status TEXT NOT NULL DEFAULT 'untested', -- untested | mixed | confirmed | disconfirmed
    notes TEXT,
    updated_at TEXT NOT NULL,
    UNIQUE(venture_id, element_name)
);

CREATE TABLE IF NOT EXISTS cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    venture_id INTEGER NOT NULL REFERENCES ventures(id) ON DELETE CASCADE,
    cycle_number INTEGER NOT NULL,
    todos_json TEXT NOT NULL,
    results_json TEXT,
    analysis_json TEXT,
    decision TEXT, -- persevere | pivot | kill
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS launch_strategy (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    venture_id INTEGER NOT NULL REFERENCES ventures(id) ON DELETE CASCADE,
    funding_matches_json TEXT NOT NULL,
    gtm_channels_json TEXT NOT NULL,
    action_plan_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

BMC_ELEMENTS = ["customer", "problem", "value_prop", "channel", "revenue"]
CYCLE_CAP = 6


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


# ---------- users ----------

def create_user(email: str, password_hash: str):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
            (email, password_hash, now()),
        )
        return cur.lastrowid


def get_user_by_email(email: str):
    with get_conn() as conn:
        return conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()


def get_user_by_id(user_id: int):
    with get_conn() as conn:
        return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


# ---------- ideas ----------

def create_idea(user_id: int, title: str, commercial_framing: str, strength_signal: str, raw_claims: str):
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO ideas (user_id, title, commercial_framing, strength_signal, raw_claims, status, created_at)
               VALUES (?, ?, ?, ?, ?, 'candidate', ?)""",
            (user_id, title, commercial_framing, strength_signal, raw_claims, now()),
        )
        return cur.lastrowid


def list_ideas(user_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM ideas WHERE user_id = ? ORDER BY created_at DESC", (user_id,)
        ).fetchall()


def get_idea(idea_id: int, user_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM ideas WHERE id = ? AND user_id = ?", (idea_id, user_id)
        ).fetchone()


def set_idea_status(idea_id: int, status: str):
    with get_conn() as conn:
        conn.execute("UPDATE ideas SET status = ? WHERE id = ?", (status, idea_id))


# ---------- ventures ----------

def create_venture(user_id: int, idea_id: int):
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO ventures (user_id, idea_id, phase, cycle_count, status, created_at)
               VALUES (?, ?, 2, 0, 'active', ?)""",
            (user_id, idea_id, now()),
        )
        venture_id = cur.lastrowid
        for el in BMC_ELEMENTS:
            conn.execute(
                """INSERT INTO bmc_elements (venture_id, element_name, status, notes, updated_at)
                   VALUES (?, ?, 'untested', '', ?)""",
                (venture_id, el, now()),
            )
        return venture_id


def list_ventures(user_id: int):
    with get_conn() as conn:
        return conn.execute(
            """SELECT ventures.*, ideas.title as idea_title
               FROM ventures JOIN ideas ON ventures.idea_id = ideas.id
               WHERE ventures.user_id = ? ORDER BY ventures.created_at DESC""",
            (user_id,),
        ).fetchall()


def get_venture(venture_id: int, user_id: int):
    with get_conn() as conn:
        return conn.execute(
            """SELECT ventures.*, ideas.title as idea_title, ideas.commercial_framing
               FROM ventures JOIN ideas ON ventures.idea_id = ideas.id
               WHERE ventures.id = ? AND ventures.user_id = ?""",
            (venture_id, user_id),
        ).fetchone()


def update_venture(venture_id: int, **fields):
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    with get_conn() as conn:
        conn.execute(f"UPDATE ventures SET {cols} WHERE id = ?", (*fields.values(), venture_id))


# ---------- bmc elements ----------

def get_bmc_elements(venture_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM bmc_elements WHERE venture_id = ? ORDER BY element_name", (venture_id,)
        ).fetchall()


def update_bmc_element(venture_id: int, element_name: str, status: str, notes: str):
    with get_conn() as conn:
        conn.execute(
            """UPDATE bmc_elements SET status = ?, notes = ?, updated_at = ?
               WHERE venture_id = ? AND element_name = ?""",
            (status, notes, now(), venture_id, element_name),
        )


def all_confirmed(venture_id: int) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as c FROM bmc_elements WHERE venture_id = ? AND status != 'confirmed'",
            (venture_id,),
        ).fetchone()
        return row["c"] == 0


# ---------- cycles ----------

def create_cycle(venture_id: int, cycle_number: int, todos: list):
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO cycles (venture_id, cycle_number, todos_json, created_at)
               VALUES (?, ?, ?, ?)""",
            (venture_id, cycle_number, json.dumps(todos), now()),
        )
        return cur.lastrowid


def get_current_cycle(venture_id: int):
    with get_conn() as conn:
        return conn.execute(
            """SELECT * FROM cycles WHERE venture_id = ?
               ORDER BY cycle_number DESC LIMIT 1""",
            (venture_id,),
        ).fetchone()


def get_cycle(cycle_id: int, venture_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM cycles WHERE id = ? AND venture_id = ?", (cycle_id, venture_id)
        ).fetchone()


def log_cycle_results(cycle_id: int, results: list):
    with get_conn() as conn:
        conn.execute(
            "UPDATE cycles SET results_json = ? WHERE id = ?",
            (json.dumps(results), cycle_id),
        )


def save_cycle_analysis(cycle_id: int, analysis: dict, decision: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE cycles SET analysis_json = ?, decision = ? WHERE id = ?",
            (json.dumps(analysis), decision, cycle_id),
        )


def list_cycles(venture_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM cycles WHERE venture_id = ? ORDER BY cycle_number", (venture_id,)
        ).fetchall()


# ---------- launch strategy ----------

def save_launch_strategy(venture_id: int, funding_matches: list, gtm_channels: list, action_plan: list):
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO launch_strategy (venture_id, funding_matches_json, gtm_channels_json, action_plan_json, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (venture_id, json.dumps(funding_matches), json.dumps(gtm_channels), json.dumps(action_plan), now()),
        )
        return cur.lastrowid


def get_launch_strategy(venture_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM launch_strategy WHERE venture_id = ? ORDER BY created_at DESC LIMIT 1",
            (venture_id,),
        ).fetchone()
