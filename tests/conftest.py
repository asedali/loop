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

# app.main refuses to import without a session secret, which is correct in
# production but means a fresh clone cannot even collect the suite. Set a
# throwaway before the app package is imported (conftest runs first), so the
# tests depend on this file rather than on whoever's .env happens to be around.
os.environ.setdefault("SESSION_SECRET_KEY", "test-only-not-a-real-secret")
os.environ.setdefault("LAUNCHLOOP_DEBUG", "1")

from app import db
from app import ratelimit
from app import sources

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


def _migrate_into(url: str, schema: str, revision: str = "head") -> None:
    from alembic import command

    command.upgrade(_alembic_config(url, schema), revision)


@pytest.fixture
def migrate_to(pg_url):
    """Run migrations in a fresh schema up to an arbitrary revision.

    Only useful for testing a migration's *data* handling. `tmp_db` upgrades to
    head on an empty schema, so a backfill that reads rows written by an earlier
    migration never runs anywhere in the suite — which is precisely the code most
    likely to be wrong and least likely to be noticed, because it only executes
    against a database that already has data.
    """
    created = []

    def _run(revision: str):
        from alembic import command

        schema = f"m_{uuid.uuid4().hex[:12]}"
        _create_schema(pg_url, schema)
        created.append(schema)
        command.upgrade(_alembic_config(pg_url, schema), revision)
        return schema

    yield _run
    for schema in created:
        _drop_schema(pg_url, schema)


@pytest.fixture(autouse=True)
def _reset_rate_limiter(monkeypatch):
    """Make the LLM rate limiter non-binding, then clear it, for every test.

    Two reasons, both about isolation:

      * The buckets live in process memory keyed on (user_id, ip). Every test signs
        up as user 1 from the same TestClient address, so without a reset the whole
        suite would share one bucket and start failing each other's throttling.
      * The production defaults (12/min per user) are lower than a single test
        needs — `test_passing_every_block_unlocks_phase_3` drives 18 calls — so
        leaving them in place would mean most tests were silently exercising the
        throttle instead of the state machine.

    So: set the ceiling out of the way by default and let the rate-limit tests
    pick their own, exactly as TestQuota sets its own monthly limit. The default
    values themselves are asserted directly in TestRateLimit.

    The per-user *quota* is database state, already isolated by the per-test
    schema, and is untouched.
    """
    # Inside the readers' accepted range on purpose: _int degrades an
    # out-of-range value to the production default, which would re-impose the
    # throttle this fixture exists to lift.
    monkeypatch.setenv("LLM_RATE_LIMIT_PER_MIN", "10000")
    monkeypatch.setenv("LLM_RATE_LIMIT_IP_PER_MIN", "10000")
    ratelimit.reset()
    yield
    ratelimit.reset()


@pytest.fixture(autouse=True)
def _reset_source_cache():
    """Empty the identifier-import cache around every test.

    The cache lives in process memory keyed on (provider, identifier), and the
    tests deliberately reuse the same handful of identifiers. Without this, a test
    that means to exercise a *fetch* silently gets the previous test's cached record
    instead — a failure that looks like an unrelated assertion rather than a leak.
    """
    sources.cache_clear()
    yield
    sources.cache_clear()


@pytest.fixture(autouse=True)
def _tenant():
    """Give every test an RLS tenant of user 1, and clear it afterwards.

    Most tests sign up one user, whose id is 1. Under RLS an unset tenant denies
    everything, so a test that reaches into db.py directly (rather than through a
    request) would otherwise read zero rows and fail confusingly.

    Deliberately *not* a security control: tests that care about cross-tenant
    behaviour switch tenants explicitly with `db.as_tenant`, so they keep testing
    the query filters instead of passing vacuously because RLS returned nothing.
    """
    with db.as_tenant(1):
        yield


APP_ROLE = "launchloop_app"


def _enable_app_role(engine) -> None:
    """Make get_conn() switch to the NOBYPASSRLS role, so the policies apply.

    Without this the suite runs as the superuser that owns the tables, and a
    superuser bypasses RLS even with FORCE — every isolation test would pass
    vacuously and prove nothing.
    """
    with engine.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_roles WHERE rolname = :r"), {"r": APP_ROLE}).first()
        if exists is None:
            raise pytest.UsageError(
                f"database role {APP_ROLE!r} does not exist, so row-level security "
                "cannot be exercised. It is created by migration e7a2b4c9d016, "
                "which needs CREATE ROLE privilege — the test Postgres user "
                "normally has it. Either grant that, or point TEST_DATABASE_URL "
                "at a superuser."
            )
        if not conn.execute(
                text("SELECT pg_has_role(current_user, :r, 'member')"),
                {"r": APP_ROLE}).scalar_one():
            conn.commit()
            conn.execute(text(f'GRANT "{APP_ROLE}" TO CURRENT_USER'))
            conn.commit()
    db.set_app_role(APP_ROLE)


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
            _enable_app_role(engine)
            yield schema
        finally:
            db.set_app_role("")
            db.set_engine(None)
            engine.dispose()
    finally:
        _drop_schema(pg_url, schema)

