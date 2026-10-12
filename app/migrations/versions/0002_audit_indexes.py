"""Indexes and a uniqueness rule for the audit and national reports (sub-project 5).

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-12
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(op.f("ix_admission_supersedes_id"), "admission", ["supersedes_id"])
    op.create_index(op.f("ix_classification_supersedes_id"), "classification", ["supersedes_id"])
    op.create_index(op.f("ix_delivery_outcome_delivered_at"), "delivery_outcome", ["delivered_at"])
    op.create_index(
        "ix_export_record_facility_id_period_month",
        "export_record",
        ["facility_id", "period_month"],
    )
    op.create_unique_constraint(
        op.f("uq_reference_set_name_version_label"), "reference_set", ["name", "version_label"]
    )


def downgrade() -> None:
    op.drop_constraint(op.f("uq_reference_set_name_version_label"), "reference_set", type_="unique")
    op.drop_index("ix_export_record_facility_id_period_month", table_name="export_record")
    op.drop_index(op.f("ix_delivery_outcome_delivered_at"), table_name="delivery_outcome")
    op.drop_index(op.f("ix_classification_supersedes_id"), table_name="classification")
    op.drop_index(op.f("ix_admission_supersedes_id"), table_name="admission")
