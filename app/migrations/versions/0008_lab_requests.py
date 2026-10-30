"""Lab requests: lab_request, lab_result and lab_cancellation (insert-only, with the same
triggers as the other clinical tables), and re-scored predictions.

A request is closed once: a trigger refuses a result or a cancellation for a request that
already has either. Note, value and reason are encrypted by the application (bytea).

A prediction may now be re-scored after an HIV result: the new row supersedes the old one and
names the lab result. The unique (admission_id, model_id) becomes unique for first scores only
(supersedes_id IS NULL), and each prediction can be superseded once.

Downgrade removes the re-scored predictions (and their reconciliation rows) with the lab tables
they came from, disabling the immutability triggers for that delete only.

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-30
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

LAB_TESTS = ("urine_protein", "blood_glucose", "hiv_rapid")
LAB_TABLES = ("lab_request", "lab_result", "lab_cancellation")

CLOSED_ONCE_FUNCTION = """
CREATE FUNCTION lab_request_closed_once() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF EXISTS (SELECT 1 FROM lab_result WHERE request_id = NEW.request_id)
       OR EXISTS (SELECT 1 FROM lab_cancellation WHERE request_id = NEW.request_id) THEN
        RAISE EXCEPTION USING
            MESSAGE = 'lab request ' || NEW.request_id || ' already has a result or a cancellation',
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


def _closing_table(name: str, text_column: str, actor: str, at: str) -> None:
    """lab_result and lab_cancellation: one row per request, an encrypted text, who and when."""
    op.create_table(
        name,
        _id(),
        _uuid("request_id"),
        sa.Column(text_column, pg.BYTEA(), nullable=False),
        _uuid(actor),
        _at(at),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name=op.f(f"pk_{name}")),
        sa.UniqueConstraint("request_id", name=op.f(f"uq_{name}_request_id")),
        _fk(name, "request_id", "lab_request"),
        _fk(name, actor, "app_user"),
    )


def upgrade() -> None:
    pg.ENUM(*LAB_TESTS, name="lab_test").create(op.get_bind())
    op.create_table(
        "lab_request",
        _id(),
        _uuid("admission_id"),
        sa.Column("test", pg.ENUM(name="lab_test", create_type=False), nullable=False),
        sa.Column("urgent", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("note", pg.BYTEA(), nullable=True),
        _uuid("requested_by"),
        _at("requested_at"),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_lab_request")),
        _fk("lab_request", "admission_id", "admission"),
        _fk("lab_request", "requested_by", "app_user"),
    )
    op.create_index(op.f("ix_lab_request_admission_id"), "lab_request", ["admission_id"])
    _closing_table("lab_result", "value", "recorded_by", "resulted_at")
    _closing_table("lab_cancellation", "reason", "cancelled_by", "cancelled_at")

    op.execute(CLOSED_ONCE_FUNCTION)
    for table in ("lab_result", "lab_cancellation"):
        op.execute(
            f"CREATE TRIGGER {table}_closes_once BEFORE INSERT ON {table} "
            "FOR EACH ROW EXECUTE FUNCTION lab_request_closed_once()"
        )
    # Insert-only like every clinical table (forbid_change() is from 0001).
    for table in LAB_TABLES:
        op.execute(
            f"CREATE TRIGGER {table}_immutable BEFORE UPDATE OR DELETE ON {table} "
            "FOR EACH ROW EXECUTE FUNCTION forbid_change()"
        )
        op.execute(
            f"CREATE TRIGGER {table}_no_truncate BEFORE TRUNCATE ON {table} "
            "FOR EACH STATEMENT EXECUTE FUNCTION forbid_change()"
        )

    op.add_column("prediction", _uuid("lab_result_id", nullable=True))
    op.create_foreign_key(
        op.f("fk_prediction_lab_result_id_lab_result"),
        "prediction",
        "lab_result",
        ["lab_result_id"],
        ["id"],
    )
    op.create_unique_constraint(
        op.f("uq_prediction_lab_result_id"), "prediction", ["lab_result_id"]
    )
    op.create_unique_constraint(
        op.f("uq_prediction_supersedes_id"), "prediction", ["supersedes_id"]
    )
    op.drop_constraint("uq_prediction_admission_id_model_id", "prediction", type_="unique")
    op.create_index(
        "uq_prediction_first_score",
        "prediction",
        ["admission_id", "model_id"],
        unique=True,
        postgresql_where=sa.text("supersedes_id IS NULL"),
    )
    op.create_index(op.f("ix_prediction_admission_id"), "prediction", ["admission_id"])


def _remove_rescored_predictions() -> None:
    """Re-scored predictions exist only because of a lab result: they go with the lab tables."""
    rescored = "SELECT id FROM prediction WHERE lab_result_id IS NOT NULL"
    op.execute("ALTER TABLE reconciled_outcome DISABLE TRIGGER reconciled_outcome_immutable")
    op.execute("ALTER TABLE prediction DISABLE TRIGGER prediction_immutable")
    op.execute(f"DELETE FROM reconciled_outcome WHERE prediction_id IN ({rescored})")
    op.execute("DELETE FROM prediction WHERE lab_result_id IS NOT NULL")
    op.execute("ALTER TABLE prediction ENABLE TRIGGER prediction_immutable")
    op.execute("ALTER TABLE reconciled_outcome ENABLE TRIGGER reconciled_outcome_immutable")


def downgrade() -> None:
    _remove_rescored_predictions()
    op.drop_index(op.f("ix_prediction_admission_id"), table_name="prediction")
    op.drop_index("uq_prediction_first_score", table_name="prediction")
    op.create_unique_constraint(
        op.f("uq_prediction_admission_id_model_id"), "prediction", ["admission_id", "model_id"]
    )
    op.drop_constraint(op.f("uq_prediction_supersedes_id"), "prediction", type_="unique")
    op.drop_constraint(op.f("uq_prediction_lab_result_id"), "prediction", type_="unique")
    op.drop_constraint(
        op.f("fk_prediction_lab_result_id_lab_result"), "prediction", type_="foreignkey"
    )
    op.drop_column("prediction", "lab_result_id")
    for table in reversed(LAB_TABLES):
        op.drop_table(table)
    op.execute("DROP FUNCTION lab_request_closed_once()")
    pg.ENUM(name="lab_test").drop(op.get_bind())
