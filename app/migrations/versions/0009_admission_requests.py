"""Admission requests: admission_request, admission_request_fulfilment and
admission_request_cancellation (insert-only, with the same triggers as the other clinical
tables).

A clinician asks the lab to enter a patient who has no admission yet. The request is closed once,
as a lab request is (0008): a trigger refuses a fulfilment or a cancellation for a request that
already has either. The patient number and name are encrypted by the application (bytea) and the
number has a blind index, as on admission (0007); note and reason are encrypted too.

Downgrade drops the three tables (requests are only a way of asking; the admissions entered for
them stay).

Revision ID: 0009
Revises: 0008
Create Date: 2026-10-30
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ASK_TABLES = (
    "admission_request",
    "admission_request_fulfilment",
    "admission_request_cancellation",
)

CLOSED_ONCE_FUNCTION = """
CREATE FUNCTION admission_request_closed_once() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF EXISTS (SELECT 1 FROM admission_request_fulfilment WHERE request_id = NEW.request_id)
       OR EXISTS (
           SELECT 1 FROM admission_request_cancellation WHERE request_id = NEW.request_id
       ) THEN
        RAISE EXCEPTION USING
            MESSAGE = 'admission request ' || NEW.request_id
                || ' already has an admission or a cancellation',
            ERRCODE = 'unique_violation';
    END IF;
    RETURN NEW;
END;
$$
"""


def _uuid(name: str, nullable: bool = False) -> sa.Column:
    return sa.Column(name, pg.UUID(as_uuid=True), nullable=nullable)


def _id() -> sa.Column:
    return sa.Column(
        "id", pg.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False
    )


def _at(name: str) -> sa.Column:
    return sa.Column(name, sa.DateTime(timezone=True), nullable=False)


def _created_at() -> sa.Column:
    return sa.Column(
        "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
    )


def _fk(table: str, column: str, target: str) -> sa.ForeignKeyConstraint:
    return sa.ForeignKeyConstraint(
        [column], [f"{target}.id"], name=op.f(f"fk_{table}_{column}_{target}")
    )


def upgrade() -> None:
    op.create_table(
        "admission_request",
        _id(),
        _uuid("facility_id"),
        sa.Column("patient_number", pg.BYTEA(), nullable=False),
        sa.Column("patient_name", pg.BYTEA(), nullable=True),
        sa.Column("patient_number_index", sa.CHAR(64), nullable=False),
        sa.Column("urgent", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("note", pg.BYTEA(), nullable=True),
        _uuid("requested_by"),
        _at("requested_at"),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_admission_request")),
        _fk("admission_request", "facility_id", "facility"),
        _fk("admission_request", "requested_by", "app_user"),
    )
    op.create_index(op.f("ix_admission_request_facility_id"), "admission_request", ["facility_id"])
    op.create_index(
        op.f("ix_admission_request_patient_number_index"),
        "admission_request",
        ["patient_number_index"],
    )
    name = "admission_request_fulfilment"
    op.create_table(
        name,
        _id(),
        _uuid("request_id"),
        _uuid("admission_id"),
        _uuid("fulfilled_by"),
        _at("fulfilled_at"),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name=op.f(f"pk_{name}")),
        sa.UniqueConstraint("request_id", name=op.f(f"uq_{name}_request_id")),
        sa.UniqueConstraint("admission_id", name=op.f(f"uq_{name}_admission_id")),
        _fk(name, "request_id", "admission_request"),
        _fk(name, "admission_id", "admission"),
        _fk(name, "fulfilled_by", "app_user"),
    )
    name = "admission_request_cancellation"
    op.create_table(
        name,
        _id(),
        _uuid("request_id"),
        sa.Column("reason", pg.BYTEA(), nullable=False),
        _uuid("cancelled_by"),
        _at("cancelled_at"),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name=op.f(f"pk_{name}")),
        sa.UniqueConstraint("request_id", name=op.f(f"uq_{name}_request_id")),
        _fk(name, "request_id", "admission_request"),
        _fk(name, "cancelled_by", "app_user"),
    )

    op.execute(CLOSED_ONCE_FUNCTION)
    for table in ("admission_request_fulfilment", "admission_request_cancellation"):
        op.execute(
            f"CREATE TRIGGER {table}_closes_once BEFORE INSERT ON {table} "
            "FOR EACH ROW EXECUTE FUNCTION admission_request_closed_once()"
        )
    # Insert-only like every clinical table (forbid_change() is from 0001).
    for table in ASK_TABLES:
        op.execute(
            f"CREATE TRIGGER {table}_immutable BEFORE UPDATE OR DELETE ON {table} "
            "FOR EACH ROW EXECUTE FUNCTION forbid_change()"
        )
        op.execute(
            f"CREATE TRIGGER {table}_no_truncate BEFORE TRUNCATE ON {table} "
            "FOR EACH STATEMENT EXECUTE FUNCTION forbid_change()"
        )


def downgrade() -> None:
    for table in reversed(ASK_TABLES):
        op.drop_table(table)
    op.execute("DROP FUNCTION admission_request_closed_once()")
