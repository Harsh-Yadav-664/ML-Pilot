"""LLM call log: project, error, privacy level, prompt size

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-06 23:23:30.162578
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0007"
down_revision: str | Sequence[str] | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("llm_calls", schema=None) as batch_op:
        batch_op.add_column(sa.Column("project_id", sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column("error", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("privacy_level", sa.String(length=30), nullable=True))
        batch_op.add_column(
            sa.Column("prompt_chars", sa.Integer(), server_default="0", nullable=False)
        )
        batch_op.create_index(batch_op.f("ix_llm_calls_project_id"), ["project_id"], unique=False)
        batch_op.create_foreign_key(
            batch_op.f("fk_llm_calls_project_id_projects"), "projects", ["project_id"], ["id"]
        )


def downgrade() -> None:
    with op.batch_alter_table("llm_calls", schema=None) as batch_op:
        batch_op.drop_constraint(batch_op.f("fk_llm_calls_project_id_projects"), type_="foreignkey")
        batch_op.drop_index(batch_op.f("ix_llm_calls_project_id"))
        batch_op.drop_column("prompt_chars")
        batch_op.drop_column("privacy_level")
        batch_op.drop_column("error")
        batch_op.drop_column("project_id")
