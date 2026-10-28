"""Per-facility readiness cut: the probability (in percent) at or above which an admission is
shown as high readiness. 30 is the national starting cut for a 20% false-alarm budget
(ML/README.md, per-hospital cuts).

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "facility",
        sa.Column("readiness_cut_percent", sa.Integer(), server_default="30", nullable=False),
    )
    op.create_check_constraint(
        "readiness_cut_range", "facility", "readiness_cut_percent BETWEEN 5 AND 95"
    )


def downgrade() -> None:
    op.drop_constraint("readiness_cut_range", "facility", type_="check")
    op.drop_column("facility", "readiness_cut_percent")
