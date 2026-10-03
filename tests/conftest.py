"""Postgres-backed test fixtures.

Every test gets its own Postgres schema, built by running the real Alembic
migrations into it. That keeps the tests honest: they exercise the same DDL
production does, and a migration that only half-applies fails here first.

Transaction rollback is not usable for isolation: app/db.py opens and commits
its own transaction per call (get_conn()), so a fixture-level rollback would
never see the app's writes. A per-test schema is the boundary that holds.

Set TEST_DATABASE_URL to point somewhere. It falls back to DATABASE_URL and
then to a local Postgres, so `python -m pytest` works with no config.
"""
import os
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from app import db

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _url(raw: str) -> str:
    if raw.startswith("postgres://"):
        return "postgresql://" + raw[len("postgres://"):]
    return raw


def _check_placeholder(url: str, name: str) -> None:
    """A copied-from-.env.example URL fails as `role "USER" does not exist`,
    which reads like a Postgres problem rather than an unedited placeholder."""
    if any(tok in url for tok in ("USER", "PASSWORD", "<", "PROJECT")):
        raise pytest.UsageError(
            f"{name} still contains a placeholder from .env.example:\n  {url}\n"
            "Replace USER/PASSWORD with real values, or unset it to fall back "
            "to DATABASE_URL and then to localhost/launchloop_test."
        )


@pytest.fixture(scope="session")
def pg_url() -> str:
    configured = os.environ.get("TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if configured:
        _check_placeholder(configured, "TEST_DATABASE_URL/DATABASE_URL")
        return _url(configured)
    user = os.environ.get("USER") or "postgres"
    return f"postgresql://{user}@localhost:5432/launchloop_test"


def _alembic_config(url: str, version_table_schema: str):
    from alembic.config import Config

    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    # attributes, not set_main_option: configparser %-interpolation rejects a
    # connection URI containing a percent-encoded sslrootcert path.
    cfg.attributes["sqlalchemy_url"] = url
    # env.py reads this to place alembic_version inside the test's own schema.
    cfg.attributes["version_table_schema"] = version_table_schema
    return cfg


def _drop_schema(url: str, schema: str) -> None:
    engine = create_engine(url, isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as conn:
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
    finally:
        engine.dispose()


def _create_schema(url: str, schema: str) -> None:
    engine = create_engine(url, isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as conn:
            conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    finally:
        engine.dispose()


def _migrate_into(url: str, schema: str) -> None:
    from alembic import command

    command.upgrade(_alembic_config(url, schema), "head")


@pytest.fixture
def tmp_db(pg_url):
    """Install a per-test schema as the process-wide engine and yield its name.

    The fixture name is historical — it used to return a temp .db file path. It
    is kept so the existing test signatures don't all churn; the one test that
    used the path was removed.
    """
    schema = f"t_{uuid.uuid4().hex[:12]}"
    _create_schema(pg_url, schema)
    try:
        _migrate_into(pg_url, schema)

        engine = create_engine(
            pg_url,
            pool_pre_ping=True,
            # Every connection this engine opens resolves unqualified names
            # inside the test schema only, so tests cannot see each other's rows.
            connect_args={"options": f"-csearch_path={schema}"},
        )
        db.set_engine(engine)
        try:
            yield schema
        finally:
            db.set_engine(None)
            engine.dispose()
    finally:
        _drop_schema(pg_url, schema)

