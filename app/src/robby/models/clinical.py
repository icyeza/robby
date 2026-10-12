"""Clinical tables. Rows are never updated or deleted (database triggers enforce it);
corrections are new rows pointing at the row they replace through supersedes_id."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    ARRAY,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from robby.models.base import UUIDPK, Base, CreatedAt
from robby.models.enums import (
    AdmissionSource,
    ClassificationStatus,
    DeliveryMode,
    OmissionField,
    OmissionReason,
    pg_enum,
)


class AdmissionCounter(Base):
    """Per-facility sequence for admission codes; the column holds the next number to issue."""

    __tablename__ = "admission_counter"
    __table_args__ = (CheckConstraint('"next" >= 1', name="next_positive"),)

    facility_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("facility.id"), primary_key=True)
    next_value: Mapped[int] = mapped_column("next", default=1, server_default=text("1"))
    created_at: Mapped[CreatedAt]


class Admission(Base):
    __tablename__ = "admission"
    __table_args__ = (
        CheckConstraint(
            "supersedes_id IS NULL OR correction_reason IS NOT NULL",
            name="correction_reason_required",
        ),
        CheckConstraint("source = 'fhir' OR entered_by IS NOT NULL", name="form_entry_has_author"),
        Index("ix_admission_facility_id_admitted_at", "facility_id", "admitted_at"),
    )

    id: Mapped[UUIDPK]
    facility_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("facility.id"))
    code: Mapped[str] = mapped_column(unique=True)
    admitted_at: Mapped[datetime]
    source: Mapped[AdmissionSource] = mapped_column(pg_enum(AdmissionSource, "admission_source"))
    entered_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("app_user.id"))
    supersedes_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("admission.id"), index=True
    )
    correction_reason: Mapped[str | None]
    parity: Mapped[int | None]
    previous_cs_count: Mapped[int | None]
    plurality: Mapped[int | None]
    fetal_presentation: Mapped[str | None]
    gestational_age_weeks: Mapped[Decimal | None] = mapped_column(Numeric(4, 1))
    onset_of_labour: Mapped[str | None]
    maternal_age: Mapped[int | None]
    systolic_bp: Mapped[int | None]
    diastolic_bp: Mapped[int | None]
    proteinuria: Mapped[str | None]
    glucose_mmol: Mapped[Decimal | None] = mapped_column(Numeric(4, 1))
    details: Mapped[dict[str, Any]] = mapped_column(
        default=dict, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[CreatedAt]


class ScreeningOmission(Base):
    __tablename__ = "screening_omission"

    admission_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("admission.id"), primary_key=True)
    field: Mapped[OmissionField] = mapped_column(
        pg_enum(OmissionField, "omission_field"), primary_key=True
    )
    reason: Mapped[OmissionReason] = mapped_column(pg_enum(OmissionReason, "omission_reason"))
    reason_note: Mapped[str | None]
    created_at: Mapped[CreatedAt]


class Classification(Base):
    __tablename__ = "classification"
    __table_args__ = (
        UniqueConstraint("admission_id", "rule_set_id"),
        CheckConstraint(
            "robson_group IS NULL OR robson_group BETWEEN 1 AND 10", name="robson_group_range"
        ),
        CheckConstraint(
            "status <> 'resolved' OR robson_group IS NOT NULL", name="resolved_has_group"
        ),
    )

    id: Mapped[UUIDPK]
    admission_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("admission.id"))
    rule_set_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("rule_set.id"))
    status: Mapped[ClassificationStatus] = mapped_column(
        pg_enum(ClassificationStatus, "classification_status")
    )
    robson_group: Mapped[int | None]
    subgroup: Mapped[str | None]
    candidates: Mapped[list[int]] = mapped_column(
        ARRAY(Integer), default=list, server_default=text("'{}'")
    )
    resolving_fields: Mapped[list[str]] = mapped_column(
        ARRAY(Text), default=list, server_default=text("'{}'")
    )
    conflict_fields: Mapped[list[str]] = mapped_column(
        ARRAY(Text), default=list, server_default=text("'{}'")
    )
    trace: Mapped[dict[str, Any]]
    supersedes_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("classification.id"), index=True
    )
    created_at: Mapped[CreatedAt]


class DeliveryOutcome(Base):
    __tablename__ = "delivery_outcome"

    id: Mapped[UUIDPK]
    admission_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("admission.id"), unique=True)
    mode: Mapped[DeliveryMode] = mapped_column(pg_enum(DeliveryMode, "delivery_mode"))
    delivered_at: Mapped[datetime] = mapped_column(index=True)
    recorded_by: Mapped[uuid.UUID] = mapped_column(ForeignKey("app_user.id"))
    created_at: Mapped[CreatedAt]


class ReconciledOutcome(Base):
    __tablename__ = "reconciled_outcome"

    id: Mapped[UUIDPK]
    prediction_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("prediction.id"), unique=True)
    delivery_outcome_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("delivery_outcome.id"))
    cesarean: Mapped[bool]
    reconciled_at: Mapped[datetime] = mapped_column(server_default=func.now())
    created_at: Mapped[CreatedAt]
