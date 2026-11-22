"""User reports on job listings + moderation lock on removed postings.

* ``moderation_flags.origin`` separates seeker reports ("user_report") from
  flags raised by admins ("admin") in the same queue.
* One OPEN report per user per job (partial unique index) so a single account
  cannot inflate the "reported by N users" count.
* ``jobs.moderation_removed_at`` / ``moderation_removed_by_user_id`` /
  ``moderation_note`` lock a posting an admin took down: the employer sees why
  and cannot republish it; only an admin can restore it.

Revision ID: 0018_job_reports
Revises: 0017_admin_trust
Create Date: 2026-11-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0018_job_reports"
down_revision: str | None = "0017_admin_trust"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "moderation_flags",
        sa.Column("origin", sa.String(length=16), nullable=False, server_default="admin"),
    )
    op.create_index(
        "uq_moderation_flags_open_report_per_user",
        "moderation_flags",
        ["target_type", "target_id", "created_by_user_id"],
        unique=True,
        postgresql_where=sa.text("status = 'OPEN' AND origin = 'user_report'"),
    )

    op.add_column("jobs", sa.Column("moderation_removed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column(
        "jobs",
        sa.Column(
            "moderation_removed_by_user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.add_column("jobs", sa.Column("moderation_note", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("jobs", "moderation_note")
    op.drop_column("jobs", "moderation_removed_by_user_id")
    op.drop_column("jobs", "moderation_removed_at")
    op.drop_index("uq_moderation_flags_open_report_per_user", table_name="moderation_flags")
    op.drop_column("moderation_flags", "origin")
