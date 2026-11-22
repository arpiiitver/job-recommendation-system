"""career goals: target roles + skills being learned (role/domain switchers)

Revision ID: 0015_career_goals
Revises: 0014_phase2_traceability
Create Date: 2026-11-20 12:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0015_career_goals"
down_revision: Union[str, None] = "0014_phase2_traceability"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Roles the seeker WANTS (may differ from the resume history) and skills
    # they are currently learning; both are additive and nullable.
    op.add_column("profiles", sa.Column("target_roles", postgresql.JSONB(), nullable=True))
    op.add_column("profiles", sa.Column("learning_skills", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("profiles", "learning_skills")
    op.drop_column("profiles", "target_roles")
