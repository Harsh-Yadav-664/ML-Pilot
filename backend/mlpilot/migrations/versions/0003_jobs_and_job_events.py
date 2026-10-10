"""Jobs and job events: durable background work and its ordered event log (#91).

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-06 12:40:04.021890
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=True),
        sa.Column("kind", sa.String(length=40), nullable=False),
        sa.Column("params", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("progress", sa.Float(), nullable=False),
        sa.Column("current_step", sa.String(length=200), nullable=True),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["project_id"], ["projects.id"], name=op.f("fk_jobs_project_id_projects")
        ),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], name=op.f("fk_jobs_run_id_runs")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_jobs")),
    )
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_jobs_status"), ["status"], unique=False)

    op.create_table(
        "job_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.String(length=36), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("type", sa.String(length=20), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], name=op.f("fk_job_events_job_id_jobs")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_job_events")),
        sa.UniqueConstraint("job_id", "seq", name=op.f("uq_job_events_job_id")),
    )


def downgrade() -> None:
    op.drop_table("job_events")
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_jobs_status"))

    op.drop_table("jobs")
