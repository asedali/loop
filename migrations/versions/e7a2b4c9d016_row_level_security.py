"""row-level security, as defence in depth behind the user_id filters

Every lookup in app/db.py takes a user_id. That is load-bearing, but it is
load-bearing *by convention* — one forgotten WHERE clause is a cross-tenant read
and nothing in the system would notice. These policies move the guarantee out of
the code and into the database: a query with no user_id filter at all still
cannot return another tenant's rows.

THE BYPASS PROBLEM, which is the whole reason this file is not a dozen CREATE
POLICY statements
--------------------------------------------------------------------------------
RLS does nothing for a role with BYPASSRLS. Postgres superusers bypass it
regardless of FORCE ROW LEVEL SECURITY, and so does Supabase's service_role —
which is what DATABASE_URL points at by default. So the policies below are
worthless unless the app connects as a role *without* that attribute.

Hence launchloop_app: NOSUPERUSER NOBYPASSRLS, granted the privileges the app
needs, and reached with SET LOCAL ROLE from app/db.py:get_conn().

Two deliberate safety properties, because a migration runs on boot against
production:

  * CREATE ROLE is BEST-EFFORT and never fatal. A Supabase deploy whose role
    cannot CREATE ROLE still migrates cleanly; it just has policies with no role
    to apply them. Failing the boot would be strictly worse than running with a
    documented gap. db.rls_active() reports which of the two you have.
  * The SET ROLE switch is CONDITIONAL on membership, tested with
    pg_has_role. A deploy that never granted membership keeps working exactly as
    before rather than 500-ing on every request.

Policies
--------------------------------------------------------------------------------
Unset tenant means DENY, not allow: current_setting('app.user_id', true) yields
NULL when never set, NULL = user_id is NULL, and the row is not returned. The
default is closed.

  * ideas, ventures, llm_calls, password_reset_tokens -> user_id directly.
  * bmc_elements, cycles, launch_strategy            -> EXISTS over ventures.
    They carry venture_id, and referring to ventures costs no recursion because
    ventures' policy does not refer back.
  * users -> your own row, OR the single row matching app.lookup_email so the
    pre-authentication login and signup lookups keep working. INSERT is limited
    to the address being registered; UPDATE/DELETE to your own row.

login_attempts is deliberately NOT enabled: see the spec, specs/04-M0.5.

Revision ID: e7a2b4c9d016
Revises: d5f8b2c7a913
"""
from alembic import op
import sqlalchemy as sa

revision = "e7a2b4c9d016"
down_revision = "d5f8b2c7a913"
branch_labels = None
depends_on = None

APP_ROLE = "launchloop_app"

# Set once per transaction by app/db.py. `true` for the second argument means
# "return NULL instead of raising if unset", which is what makes an unset tenant
# deny rather than error.
# NULLIF(..., '') is load-bearing, not decoration. Once a custom setting has been
# assigned anywhere in a session, current_setting(name, true) returns the EMPTY
# STRING after it is reset — not NULL. Casting '' to bigint raises
# `invalid input syntax for type bigint`, which would turn every pooled-connection
# reuse into a 500. NULLIF turns "reset" back into NULL, so an unset tenant is
# NULL and the comparison denies.
TENANT = "NULLIF(current_setting('app.user_id', true), '')::bigint"
LOOKUP = "NULLIF(current_setting('app.lookup_email', true), '')"
LOOKUP_TOKEN = "NULLIF(current_setting('app.lookup_token', true), '')"

DIRECT_USER_ID = ["ideas", "ventures", "llm_calls"]
VIA_VENTURE = ["bmc_elements", "cycles", "launch_strategy"]


def _create_role(conn) -> bool:
    """Best-effort. Returns whether the role exists afterwards."""
    exists = conn.exec_driver_sql(
        "SELECT 1 FROM pg_roles WHERE rolname = %s", (APP_ROLE,)
    ).first() is not None
    if exists:
        return True
    try:
        conn.exec_driver_sql(
            f'CREATE ROLE "{APP_ROLE}" NOSUPERUSER NOBYPASSRLS NOCREATEDB '
            f"NOCREATEROLE NOINHERIT LOGIN"
        )
        return True
    except Exception as exc:  # noqa: BLE001 - see module docstring
        print(
            f"[migrate] could not create role {APP_ROLE}: {exc!r}\n"
            f"[migrate] RLS policies are installed but inert until a "
            f"NOBYPASSRLS role exists and can be reached with SET ROLE."
        )
        return False


def upgrade() -> None:
    conn = op.get_bind()
    role_ready = _create_role(conn)

    if role_ready:
        # Membership, so SET LOCAL ROLE works. Idempotent, and a no-op if the
        # grant is not permitted — same reasoning as the CREATE ROLE above.
        try:
            conn.exec_driver_sql(
                f'GRANT "{APP_ROLE}" TO CURRENT_USER')
        except Exception as exc:  # noqa: BLE001
            print(f"[migrate] could not grant {APP_ROLE} to the current role: {exc!r}")

    # The tables live in whatever schema this migration ran in, which is the
    # per-test throwaway schema under the test suite and `public` in production.
    schema = conn.exec_driver_sql("SELECT current_schema()").scalar_one()

    if role_ready:
        conn.exec_driver_sql(f'GRANT USAGE ON SCHEMA "{schema}" TO "{APP_ROLE}"')
        # Identity columns need sequence privileges, not just table ones — without
        # this every INSERT of a user or a venture fails under RLS.
        conn.exec_driver_sql(
            f'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES '
            f'IN SCHEMA "{schema}" TO "{APP_ROLE}"')
        conn.exec_driver_sql(
            f'GRANT USAGE, SELECT ON ALL SEQUENCES '
            f'IN SCHEMA "{schema}" TO "{APP_ROLE}"')
        # ...and for tables added by *future* migrations. Without this, the next
        # slice that adds a table ships a 500 on every query against it, because
        # the grant above only ever covered what existed at this point.
        conn.exec_driver_sql(
            f'ALTER DEFAULT PRIVILEGES IN SCHEMA "{schema}" '
            f'GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO "{APP_ROLE}"')
        conn.exec_driver_sql(
            f'ALTER DEFAULT PRIVILEGES IN SCHEMA "{schema}" '
            f'GRANT USAGE, SELECT ON SEQUENCES TO "{APP_ROLE}"')

    for table in DIRECT_USER_ID:
        _enable(conn, table)
        _policy(conn, table, f"{table}_tenant_isolation", "ALL",
                f"user_id = {TENANT}", f"user_id = {TENANT}")

    for table in VIA_VENTURE:
        _enable(conn, table)
        exists = f"""EXISTS (SELECT 1 FROM ventures v
                     WHERE v.id = {table}.venture_id AND v.user_id = {TENANT})"""
        _policy(conn, table, f"{table}_tenant_isolation", "ALL", exists, exists)

    # The token IS the credential you present before you know who you are, exactly
    # as the address is for login — so the same shape applies: the tenant, or the
    # one row matching the presented token's hash. Nothing here is guessable: the
    # hash is a 32-byte secret's SHA-256, and the flow is read-only until the row
    # is resolved and the tenant is installed.
    _enable(conn, "password_reset_tokens")
    token_visible = f"user_id = {TENANT} OR token_hash = {LOOKUP_TOKEN}"
    _policy(conn, "password_reset_tokens", "password_reset_tokens_tenant_isolation",
            "ALL", token_visible, f"user_id = {TENANT}")

    _enable(conn, "users")
    _policy(conn, "users", "users_select_own_or_lookup", "SELECT",
            f"id = {TENANT} OR email = {LOOKUP}", None)
    # Registration: you may only create the row for the address you submitted.
    _policy(conn, "users", "users_insert_own", "INSERT", None,
            f"email = {LOOKUP}")
    _policy(conn, "users", "users_update_own", "UPDATE",
            f"id = {TENANT}", f"id = {TENANT}")
    _policy(conn, "users", "users_delete_own", "DELETE", f"id = {TENANT}", None)


def _enable(conn, table: str) -> None:
    # FORCE makes the policies bind the table owner too, not only other roles.
    conn.exec_driver_sql(f'ALTER TABLE "{table}" ENABLE ROW LEVEL SECURITY')
    conn.exec_driver_sql(f'ALTER TABLE "{table}" FORCE ROW LEVEL SECURITY')


def _policy(conn, table: str, name: str, cmd: str, using: str, check: str) -> None:
    parts = [f'CREATE POLICY "{name}" ON "{table}" FOR {cmd}']
    if using:
        parts.append(f"USING ({using})")
    if check:
        parts.append(f"WITH CHECK ({check})")
    conn.exec_driver_sql(" ".join(parts))


def downgrade() -> None:
    conn = op.get_bind()
    schema = conn.exec_driver_sql("SELECT current_schema()").scalar_one()
    # Default privileges are cluster-wide per grantor, so they are only reversed
    # when the role was actually created here; otherwise dropping them could
    # silently strip a grant some other database is relying on.
    if conn.exec_driver_sql(
            "SELECT 1 FROM pg_roles WHERE rolname = %s", (APP_ROLE,)).first():
        for obj_type, privileges in (("TABLES", "SELECT, INSERT, UPDATE, DELETE"),
                                    ("SEQUENCES", "USAGE, SELECT")):
            try:
                conn.exec_driver_sql(
                    f'ALTER DEFAULT PRIVILEGES IN SCHEMA "{schema}" '
                    f'REVOKE {privileges} ON {obj_type} FROM "{APP_ROLE}"')
            except Exception as exc:  # noqa: BLE001
                print(f"[migrate] could not revoke default privileges: {exc!r}")
    for table in DIRECT_USER_ID + VIA_VENTURE:
        _drop(conn, table, f"{table}_tenant_isolation")
    _drop(conn, "password_reset_tokens", "password_reset_tokens_tenant_isolation")
    for name in ("users_select_own_or_lookup", "users_insert_own",
                 "users_update_own", "users_delete_own"):
        _drop(conn, "users", name)
    for table in DIRECT_USER_ID + VIA_VENTURE + ["users", "password_reset_tokens"]:
        # Leave the role in place: another table or deployment may still want it,
        # and dropping a role that still holds grants would fail the downgrade.
        conn.exec_driver_sql(f'ALTER TABLE "{table}" NO FORCE ROW LEVEL SECURITY')
        conn.exec_driver_sql(f'ALTER TABLE "{table}" DISABLE ROW LEVEL SECURITY')


def _drop(conn, table: str, name: str) -> None:
    conn.exec_driver_sql(f'DROP POLICY IF EXISTS "{name}" ON "{table}"')