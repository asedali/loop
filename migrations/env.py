"""Alembic environment.

The URL always comes from the environment (DATABASE_URL, or an explicit
-x override from db.init_db()), never from alembic.ini, so no credential can
be committed by accident.

Table metadata comes from app.schema, which is engine-free by design.
"""
import os
import sys
from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The `alembic` CLI does not go through app.main, which is the only place
# load_dotenv() used to run, so without this DATABASE_URL in .env is invisible
# here and `alembic upgrade head` fails even with a correct .env.
load_dotenv(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
))

from app.schema import metadata  # noqa: E402

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = metadata

# The test suite points this at a throwaway schema per test so concurrent
# sessions cannot fight over one alembic_version row.
version_table_schema = config.attributes.get("version_table_schema") or None


def _url() -> str:
    # attributes first: a connection URI can contain a percent-encoded
    # sslrootcert path, and Config.get_main_option() runs it through
    # configparser's %-interpolation, which rejects that outright.
    url = config.attributes.get("sqlalchemy_url")
    if not url:
        try:
            url = config.get_main_option("sqlalchemy.url", "")
        except ValueError:
            # e.g. a %-containing value in alembic.ini trips configparser.
            url = ""
    url = url or os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "No database URL. Set DATABASE_URL, or pass -x sqlalchemy.url=..."
        )
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url


def run_migrations_offline() -> None:
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table_schema=version_table_schema,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = _url()

    # When a schema is pinned (the test suite), the DDL has to land in it too.
    # Without this the tables would be created in `public` while alembic_version
    # went to the pinned schema, and the second test would collide on them.
    connect_args = {}
    if version_table_schema:
        connect_args["options"] = f"-csearch_path={version_table_schema},public"

    connectable = engine_from_config(
        section, prefix="sqlalchemy.", poolclass=pool.NullPool, connect_args=connect_args
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table_schema=version_table_schema,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
