"""session revocation, so changing a password ends the sessions it should

The session is a stateless signed cookie, so nothing could tell a live session
from one minted before a password change. Changing the password is precisely the
moment a cookie stops being a credential, and here it had no effect on anyone
already holding one.

users.session_epoch
--------------------------------------------------------------------------------
An integer the login flow copies into the cookie, compared against the row
get_current_user() has *already loaded* on every request (it has to: it installs
the RLS tenant from that row). So the check costs no extra query, which is the
whole reason this shape was chosen over a sessions table — a sessions table would
mean a new table with its own RLS policy, a cleanup job for expired rows, and a
second write per request, to answer a question the user row already answers.

The epoch is NOT a secret. All the security comes from the cookie remaining
signed: forging one needs SESSION_SECRET_KEY. The epoch only decides whether an
already-valid cookie is still current — a revocation flag, not a credential. The
deployment consequence is that every session minted before this migration stops
being honoured, which is the correct outcome: those are exactly the cookies the
change exists to kill.

Revision ID: d4a7f1c93e20
Revises: c9e4b2a71d38
"""
from alembic import op
import sqlalchemy as sa

revision = "d4a7f1c93e20"
down_revision = "c9e4b2a71d38"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # NOT NULL with a constant default, which PostgreSQL 11+ applies without a
    # table rewrite — the reason this can be a plain ADD COLUMN on a live users
    # table rather than a nullable column plus a backfill plus a NOT NULL pass.
    op.add_column(
        "users",
        sa.Column(
            "session_epoch",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("1"),
        ),
    )


def downgrade() -> None:
    # Reverting the column revokes every session by making it unreadable, so a
    # downgrade is itself a mass logout. Nothing to preserve: no code outside
    # db/auth/main reads this column.
    op.drop_column("users", "session_epoch")