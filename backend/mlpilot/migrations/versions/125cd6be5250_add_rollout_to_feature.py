"""add rollout to feature"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "125cd6be5250"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("features", schema=None) as batch_op:
        batch_op.add_column(sa.Column("rollout", sa.Integer(), server_default="0", nullable=False))


def downgrade() -> None:
    with op.batch_alter_table("features", schema=None) as batch_op:
        batch_op.drop_column("rollout")
