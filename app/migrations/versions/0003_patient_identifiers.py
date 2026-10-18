"""Patient number and name on admissions; 'import' admission source.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ADD VALUE may run inside the migration transaction on PostgreSQL 12+, as long as the new
    # value is not used in the same transaction (the CHECK below does not mention it).
    op.execute("ALTER TYPE admission_source ADD VALUE IF NOT EXISTS 'import'")
    op.add_column("admission", sa.Column("patient_number", sa.Text(), nullable=True))
    op.add_column("admission", sa.Column("patient_name", sa.Text(), nullable=True))
    # NOT VALID: rows saved before identifiers existed are left alone; every new row is checked.
    op.execute(
        "ALTER TABLE admission ADD CONSTRAINT ck_admission_entry_has_patient CHECK "
        "(source = 'fhir' OR (patient_number IS NOT NULL AND patient_name IS NOT NULL)) NOT VALID"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE admission DROP CONSTRAINT ck_admission_entry_has_patient")
    op.drop_column("admission", "patient_name")
    op.drop_column("admission", "patient_number")
    # PostgreSQL cannot drop an enum value; 'import' stays in admission_source.
