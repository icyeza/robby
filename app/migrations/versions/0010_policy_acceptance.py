"""Policy acceptance: policy_acceptance records that a user accepted the privacy notice and the
terms of use, with the versions and the time. Insert-only like the clinical tables: the same
forbid_change() triggers (0001) refuse UPDATE, DELETE and TRUNCATE.

A user has one row per pair of versions accepted; when a version changes the user is asked again
and a new row is added. Downgrade drops the table (acceptances are not clinical data).

Revision ID: 0010
Revises: 0009
Create Date: 2026-11-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE = "policy_acceptance"


def upgrade() -> None:
    op.create_table(
        TABLE,
        sa.Column(
            "id", pg.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("user_id", pg.UUID(as_uuid=True), nullable=False),
        sa.Column("privacy_version", sa.Text(), nullable=False),
        sa.Column("terms_version", sa.Text(), nullable=False),
        sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f(f"pk_{TABLE}")),
        sa.ForeignKeyConstraint(
            ["user_id"], ["app_user.id"], name=op.f(f"fk_{TABLE}_user_id_app_user")
        ),
        sa.UniqueConstraint(
            "user_id",
            "privacy_version",
            "terms_version",
            name=op.f(f"uq_{TABLE}_user_id_privacy_version_terms_version"),
        ),
    )
    op.create_index(op.f(f"ix_{TABLE}_user_id"), TABLE, ["user_id"])
    # Insert-only like every clinical table (forbid_change() is from 0001).
    op.execute(
        f"CREATE TRIGGER {TABLE}_immutable BEFORE UPDATE OR DELETE ON {TABLE} "
        "FOR EACH ROW EXECUTE FUNCTION forbid_change()"
    )
    op.execute(
        f"CREATE TRIGGER {TABLE}_no_truncate BEFORE TRUNCATE ON {TABLE} "
        "FOR EACH STATEMENT EXECUTE FUNCTION forbid_change()"
    )


def downgrade() -> None:
    op.drop_table(TABLE)
