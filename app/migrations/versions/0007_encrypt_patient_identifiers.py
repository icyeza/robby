"""Patient number and name encrypted at rest (AES-256-GCM, robby.crypto), with a blind index on
the number for exact search and import duplicate checks. Needs ROBBY_FIELD_KEY.

Existing rows are rewritten in place, so the admission_immutable trigger (0001) is disabled for
the rewrite only, and enabled again in the same transaction. Downgrade decrypts back to text with
the same key.

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-29
"""

from collections.abc import Callable, Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

from robby import crypto

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Associated data, as in robby.models.clinical; written out so this migration never changes.
NUMBER_AD = "admission.patient_number"
NAME_AD = "admission.patient_name"
ENTRY_HAS_PATIENT = (
    "ALTER TABLE admission ADD CONSTRAINT ck_admission_entry_has_patient CHECK "
    "(source = 'fhir' OR (patient_number IS NOT NULL AND patient_name IS NOT NULL)) NOT VALID"
)


def _rewrite(select_sql: str, update_sql: str, convert: Callable[[sa.Row], dict]) -> None:
    """Fill the new columns of every row holding an identifier; the trigger is off meanwhile."""
    bind = op.get_bind()
    rows = [convert(row) for row in bind.execute(sa.text(select_sql))]
    if not rows:
        return
    op.execute("ALTER TABLE admission DISABLE TRIGGER admission_immutable")
    bind.execute(sa.text(update_sql), rows)
    op.execute("ALTER TABLE admission ENABLE TRIGGER admission_immutable")


def upgrade() -> None:
    keyring = crypto.active()  # fails with a clear message when ROBBY_FIELD_KEY is not set
    op.add_column("admission", sa.Column("number_sealed", pg.BYTEA(), nullable=True))
    op.add_column("admission", sa.Column("name_sealed", pg.BYTEA(), nullable=True))
    op.add_column("admission", sa.Column("patient_number_index", sa.CHAR(64), nullable=True))

    def seal(row: sa.Row) -> dict:
        number, name = row.patient_number, row.patient_name
        return {
            "id": row.id,
            "number": None if number is None else keyring.encrypt(number, NUMBER_AD),
            "name": None if name is None else keyring.encrypt(name, NAME_AD),
            "idx": None if number is None else keyring.blind_index(row.facility_id, number),
        }

    _rewrite(
        "SELECT id, facility_id, patient_number, patient_name FROM admission "
        "WHERE patient_number IS NOT NULL OR patient_name IS NOT NULL",
        "UPDATE admission SET number_sealed = :number, name_sealed = :name, "
        "patient_number_index = :idx WHERE id = :id",
        seal,
    )
    op.execute("ALTER TABLE admission DROP CONSTRAINT ck_admission_entry_has_patient")
    op.drop_column("admission", "patient_number")
    op.drop_column("admission", "patient_name")
    op.alter_column("admission", "number_sealed", new_column_name="patient_number")
    op.alter_column("admission", "name_sealed", new_column_name="patient_name")
    op.execute(ENTRY_HAS_PATIENT)
    op.execute(
        "ALTER TABLE admission ADD CONSTRAINT ck_admission_patient_number_indexed CHECK "
        "((patient_number IS NULL) = (patient_number_index IS NULL))"
    )
    op.create_index("ix_admission_patient_number_index", "admission", ["patient_number_index"])


def downgrade() -> None:
    keyring = crypto.active()
    op.drop_index("ix_admission_patient_number_index", table_name="admission")
    op.execute("ALTER TABLE admission DROP CONSTRAINT ck_admission_patient_number_indexed")
    op.add_column("admission", sa.Column("number_plain", sa.Text(), nullable=True))
    op.add_column("admission", sa.Column("name_plain", sa.Text(), nullable=True))

    def open_(row: sa.Row) -> dict:
        number, name = row.patient_number, row.patient_name
        return {
            "id": row.id,
            "number": None if number is None else keyring.decrypt(bytes(number), NUMBER_AD),
            "name": None if name is None else keyring.decrypt(bytes(name), NAME_AD),
        }

    _rewrite(
        "SELECT id, patient_number, patient_name FROM admission "
        "WHERE patient_number IS NOT NULL OR patient_name IS NOT NULL",
        "UPDATE admission SET number_plain = :number, name_plain = :name WHERE id = :id",
        open_,
    )
    op.execute("ALTER TABLE admission DROP CONSTRAINT ck_admission_entry_has_patient")
    op.drop_column("admission", "patient_number")
    op.drop_column("admission", "patient_name")
    op.drop_column("admission", "patient_number_index")
    op.alter_column("admission", "number_plain", new_column_name="patient_number")
    op.alter_column("admission", "name_plain", new_column_name="patient_name")
    op.execute(ENTRY_HAS_PATIENT)
