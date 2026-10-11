"""Model registry: versions, lifecycle transitions and predictions."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    CHAR,
    CheckConstraint,
    ForeignKey,
    Index,
    Numeric,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from robby.models.base import UUIDPK, Base, CreatedAt
from robby.models.enums import ModelState, PredictionMode, pg_enum


class ModelVersion(Base):
    __tablename__ = "model_version"
    __table_args__ = (
        Index(
            "uq_model_version_one_active",
            "state",
            unique=True,
            postgresql_where=text("state = 'active'"),
        ),
        CheckConstraint(
            "training_window_start <= training_window_end", name="training_window_order"
        ),
    )

    id: Mapped[UUIDPK]
    version_label: Mapped[str] = mapped_column(unique=True)
    algorithm: Mapped[str]
    feature_set: Mapped[str]
    feature_list: Mapped[list[str]] = mapped_column(JSONB)
    hyperparameters: Mapped[dict[str, Any]]
    calibration_method: Mapped[str]
    evaluation_metrics: Mapped[dict[str, Any]]
    artefact_hash: Mapped[str] = mapped_column(CHAR(64))
    artefact_path: Mapped[str]
    manifest: Mapped[dict[str, Any]]
    training_window_start: Mapped[date]
    training_window_end: Mapped[date]
    state: Mapped[ModelState] = mapped_column(
        pg_enum(ModelState, "model_state"), default=ModelState.CANDIDATE, server_default="candidate"
    )
    registered_at: Mapped[datetime] = mapped_column(server_default=func.now())
    registered_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("app_user.id"))
    approved_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("app_user.id"))
    approved_at: Mapped[datetime | None]
    created_at: Mapped[CreatedAt]


class ModelTransition(Base):
    __tablename__ = "model_transition"
    __table_args__ = (CheckConstraint("length(btrim(reason)) > 0", name="reason_not_blank"),)

    id: Mapped[UUIDPK]
    model_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("model_version.id"))
    from_state: Mapped[ModelState | None] = mapped_column(pg_enum(ModelState, "model_state"))
    to_state: Mapped[ModelState] = mapped_column(pg_enum(ModelState, "model_state"))
    actor_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("app_user.id"))
    reason: Mapped[str]
    at: Mapped[datetime] = mapped_column(server_default=func.now())
    created_at: Mapped[CreatedAt]


class Prediction(Base):
    __tablename__ = "prediction"
    __table_args__ = (
        UniqueConstraint("admission_id", "model_id"),
        CheckConstraint("probability >= 0 AND probability <= 1", name="probability_range"),
        Index("ix_prediction_model_id_created_at", "model_id", "created_at"),
    )

    id: Mapped[UUIDPK]
    admission_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("admission.id"))
    model_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("model_version.id"))
    mode: Mapped[PredictionMode] = mapped_column(pg_enum(PredictionMode, "prediction_mode"))
    probability: Mapped[Decimal] = mapped_column(Numeric(6, 5))
    feature_vector: Mapped[dict[str, Any]]
    supersedes_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("prediction.id"))
    created_at: Mapped[CreatedAt]
