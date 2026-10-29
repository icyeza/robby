"""Lab technician role: records admissions at one facility, like the clinician it is tied to a
facility. The enum value is added outside the transaction (PostgreSQL cannot use a new enum value
in the transaction that adds it). Downgrade restores the constraint; the enum value stays.

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-29
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

CONSTRAINT = "ck_app_user_facility_matches_role"


def _facility_check(roles: tuple[str, ...]) -> None:
    listed = ", ".join(f"'{r}'" for r in roles)
    op.execute(f"ALTER TABLE app_user DROP CONSTRAINT {CONSTRAINT}")
    op.execute(
        f"ALTER TABLE app_user ADD CONSTRAINT {CONSTRAINT} "
        f"CHECK ((role::text IN ({listed})) = (facility_id IS NOT NULL))"
    )


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE user_role ADD VALUE IF NOT EXISTS 'lab_technician'")
    _facility_check(("clinician", "facility_admin", "lab_technician"))


def downgrade() -> None:
    _facility_check(("clinician", "facility_admin"))
