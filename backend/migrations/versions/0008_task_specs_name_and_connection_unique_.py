"""Task specs: name and connection, unique version per name

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-07 01:35:08.301008
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0008"
down_revision: str | Sequence[str] | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("task_specs", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("name", sa.String(length=64), server_default="", nullable=False)
        )
        batch_op.add_column(sa.Column("connection_id", sa.String(length=36), nullable=True))
        batch_op.create_unique_constraint(
            batch_op.f("uq_task_specs_project_id"), ["project_id", "name", "version"]
        )
        batch_op.create_foreign_key(
            batch_op.f("fk_task_specs_connection_id_connections"),
            "connections",
            ["connection_id"],
            ["id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("task_specs", schema=None) as batch_op:
        batch_op.drop_constraint(
            batch_op.f("fk_task_specs_connection_id_connections"), type_="foreignkey"
        )
        batch_op.drop_constraint(batch_op.f("uq_task_specs_project_id"), type_="unique")
        batch_op.drop_column("connection_id")
        batch_op.drop_column("name")
