"""domain model

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-06 11:42:43.708830
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "connections",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("dialect", sa.String(length=50), nullable=False),
        sa.Column("host", sa.String(length=255), nullable=True),
        sa.Column("port", sa.Integer(), nullable=True),
        sa.Column("database", sa.String(length=255), nullable=True),
        sa.Column("username", sa.String(length=255), nullable=True),
        sa.Column("secret_ref", sa.String(length=255), nullable=True),
        sa.Column("ssl", sa.JSON(), nullable=True),
        sa.Column("last_tested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("can_write", sa.Boolean(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["project_id"], ["projects.id"], name=op.f("fk_connections_project_id_projects")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_connections")),
    )
    op.create_table(
        "data_versions",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=20), nullable=False),
        sa.Column("source", sa.JSON(), nullable=False),
        sa.Column("n_rows", sa.Integer(), nullable=True),
        sa.Column("n_columns", sa.Integer(), nullable=True),
        sa.Column("max_event_time", sa.JSON(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["project_id"], ["projects.id"], name=op.f("fk_data_versions_project_id_projects")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_data_versions")),
    )
    op.create_table(
        "task_specs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("yaml", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("confirmed_by", sa.String(length=255), nullable=True),
        sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("schema_fingerprint", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["project_id"], ["projects.id"], name=op.f("fk_task_specs_project_id_projects")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_task_specs")),
    )
    op.create_table(
        "runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("task_spec_id", sa.String(length=36), nullable=True),
        sa.Column("data_version_id", sa.String(length=64), nullable=True),
        sa.Column("split_plan", sa.JSON(), nullable=False),
        sa.Column("engine", sa.String(length=100), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("budget", sa.JSON(), nullable=False),
        sa.Column("budget_used", sa.JSON(), nullable=False),
        sa.Column("champion_experiment_id", sa.String(length=36), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["champion_experiment_id"],
            ["experiments.id"],
            name=op.f("fk_runs_champion_experiment_id_experiments"),
        ),
        sa.ForeignKeyConstraint(
            ["data_version_id"],
            ["data_versions.id"],
            name=op.f("fk_runs_data_version_id_data_versions"),
        ),
        sa.ForeignKeyConstraint(
            ["project_id"], ["projects.id"], name=op.f("fk_runs_project_id_projects")
        ),
        sa.ForeignKeyConstraint(
            ["task_spec_id"], ["task_specs.id"], name=op.f("fk_runs_task_spec_id_task_specs")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_runs")),
    )
    op.create_table(
        "features",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("kind", sa.String(length=20), nullable=False),
        sa.Column("sql", sa.Text(), nullable=True),
        sa.Column("formula", sa.Text(), nullable=True),
        sa.Column("ir", sa.JSON(), nullable=True),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("guard_results", sa.JSON(), nullable=True),
        sa.Column("gain", sa.JSON(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], name=op.f("fk_features_run_id_runs")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_features")),
    )
    op.create_table(
        "llm_calls",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=True),
        sa.Column("purpose", sa.String(length=100), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("tokens_in", sa.Integer(), nullable=False),
        sa.Column("tokens_out", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("decision_mode", sa.String(length=20), nullable=False),
        sa.Column("prompt_hash", sa.String(length=64), nullable=False),
        sa.Column("prompt_path", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], name=op.f("fk_llm_calls_run_id_runs")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_llm_calls")),
    )
    with op.batch_alter_table("experiments", schema=None) as batch_op:
        batch_op.add_column(sa.Column("run_id", sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column("data_version_id", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("val_metrics", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("test_metrics", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("decision_detail", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("decision_mode", sa.String(length=20), nullable=True))
        batch_op.create_foreign_key(
            batch_op.f("fk_experiments_run_id_runs"), "runs", ["run_id"], ["id"]
        )
        batch_op.create_foreign_key(
            batch_op.f("fk_experiments_data_version_id_data_versions"),
            "data_versions",
            ["data_version_id"],
            ["id"],
        )

    with op.batch_alter_table("projects", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("settings", sa.JSON(), server_default=sa.text("'{}'"), nullable=False)
        )


def downgrade() -> None:
    with op.batch_alter_table("projects", schema=None) as batch_op:
        batch_op.drop_column("settings")

    with op.batch_alter_table("experiments", schema=None) as batch_op:
        batch_op.drop_constraint(
            batch_op.f("fk_experiments_data_version_id_data_versions"), type_="foreignkey"
        )
        batch_op.drop_constraint(batch_op.f("fk_experiments_run_id_runs"), type_="foreignkey")
        batch_op.drop_column("decision_mode")
        batch_op.drop_column("decision_detail")
        batch_op.drop_column("test_metrics")
        batch_op.drop_column("val_metrics")
        batch_op.drop_column("data_version_id")
        batch_op.drop_column("run_id")

    op.drop_table("llm_calls")
    op.drop_table("features")
    op.drop_table("runs")
    op.drop_table("task_specs")
    op.drop_table("data_versions")
    op.drop_table("connections")
