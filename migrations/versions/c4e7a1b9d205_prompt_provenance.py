"""prompt provenance on llm_calls, without storing prompt bodies

Two nullable columns, added only:

  prompt_version  a constant per template (constants.PROMPT_VERSIONS), so a
                  historical result can be attributed to the prompt that
                  produced it after the template is edited.
  prompt_sha256   sha256 of the exact prompt string sent, so the prompt can be
                  proven identical without keeping it.

The obvious alternative — a `prompt_text` column — is specifically rejected.
Prompts embed the researcher's pasted material verbatim inside the untrusted-data
wrapper, and llm_calls.user_id is ON DELETE SET NULL so its rows deliberately
outlive account deletion. A prompt column would therefore be a copy of
unpublished IP in the one table that survives the account.

Both columns are nullable so this applies to existing rows with no backfill and
no table rewrite.

Revision ID: c4e7a1b9d205
Revises: b3d7e91c4a02
"""
from alembic import op
import sqlalchemy as sa

revision = "c4e7a1b9d205"
down_revision = "b3d7e91c4a02"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("llm_calls", sa.Column(
        "prompt_version", sa.Text(), nullable=True))
    op.add_column("llm_calls", sa.Column(
        "prompt_sha256", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("llm_calls", "prompt_sha256")
    op.drop_column("llm_calls", "prompt_version")