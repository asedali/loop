"""lock tables down from the supabase anon/authenticated roles

The app authenticates in Python and filters every query by user_id, so RLS is
off. That makes this necessary rather than optional: a Supabase project's anon
key is public, and with RLS disabled and no grants revoked it can read every
row in every table — including users.password_hash.

service_role bypasses these grants, which is the role the app connects as, so
revoking anon/authenticated does not affect the app.

Revision ID: a1c9f0e2b7d1
Revises: f1ceb3f3319f
"""
from alembic import op

revision = "a1c9f0e2b7d1"
down_revision = "f1ceb3f3319f"
branch_labels = None
depends_on = None

TABLES = [
    "users", "ideas", "ventures", "bmc_elements",
    "cycles", "launch_strategy", "llm_calls", "login_attempts",
]

# Roles that do not exist outside Supabase, so skip rather than fail when this
# migration is run against a plain Postgres (local dev, CI).
ROLES = ["anon", "authenticated"]


def _roles(conn):
    # Offline mode (`alembic upgrade head --sql`, for pasting the DDL into the
    # Supabase SQL editor) has no connection to query pg_roles, so assume the
    # Supabase roles are there and let the emitted REVALLES land.
    if not hasattr(conn, "exec_driver_sql"):
        return set(ROLES)
    return {
        r for r in ROLES
        if conn.exec_driver_sql(
            "SELECT 1 FROM pg_roles WHERE rolname = %s", (r,)
        ).first() is not None
    }


def upgrade() -> None:
    conn = op.get_bind()
    present = _roles(conn)
    if not present:
        return
    for table in TABLES:
        for role in present:
            conn.exec_driver_sql(f'REVOKE ALL ON TABLE "{table}" FROM "{role}"')


def downgrade() -> None:
    conn = op.get_bind()
    present = _roles(conn)
    if not present:
        return
    for table in TABLES:
        for role in present:
            conn.exec_driver_sql(f'GRANT ALL ON TABLE "{table}" TO "{role}"')
