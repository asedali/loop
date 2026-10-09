"""password reset and email verification

Two additive changes, both nullable, so neither needs a backfill or a table
rewrite on a live database:

  users.email_verified_at
      Null until an address is proven. Deliberately NOT backfilled to created_at
      for existing users — that would assert a verification that never happened.
      They get the unverified banner instead, which is the truth.

  password_reset_tokens
      One table for two flows. Email verification needs exactly the same
      primitive as password reset (unguessable token, stored hashed, expiring,
      single-use), and a second near-identical table would mean two sets of
      token-rotation bugs to maintain. The `purpose` column keeps the two flows
      from ever reading each other's tokens.

Only the SHA-256 of a token is stored, so a database dump cannot be used to take
over an account, and the token_hash column is UNIQUE because a lookup by hash is
the whole redemption path.

Revision ID: d5f8b2c7a913
Revises: c4e7a1b9d205
"""
from alembic import op
import sqlalchemy as sa

from app.constants import TOKEN_PURPOSES, _in_clause

revision = "d5f8b2c7a913"
down_revision = "c4e7a1b9d205"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column(
        "email_verified_at", sa.DateTime(timezone=True), nullable=True))

    op.create_table(
        "password_reset_tokens",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("purpose", sa.Text(), nullable=False),
        sa.Column("token_hash", sa.Text(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], ondelete="CASCADE",
            name="fk_password_reset_tokens_user_id_users"),
        sa.PrimaryKeyConstraint("id", name="pk_password_reset_tokens"),
        # The redemption path is a lookup by hash, so it has to be unique and
        # indexed; without this it is a sequential scan on every email click.
        sa.UniqueConstraint("token_hash", name="uq_password_reset_tokens_token_hash"),
        # Invariant 2: the enum is in constants.py and the CHECK here, same change.
        sa.CheckConstraint(
            f"purpose IN ({_in_clause(TOKEN_PURPOSES)})", name="purpose"),
    )
    op.create_index(
        "ix_password_reset_tokens_user_id",
        "password_reset_tokens", ["user_id", "purpose"])

    # Login throttling is now keyed on the source address as well as the account,
    # so the (email, created_at) index no longer covers every query against it.
    op.create_index(
        "ix_login_attempts_ip", "login_attempts", ["ip", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_login_attempts_ip", table_name="login_attempts")
    op.drop_index("ix_password_reset_tokens_user_id", table_name="password_reset_tokens")
    op.drop_table("password_reset_tokens")
    op.drop_column("users", "email_verified_at")