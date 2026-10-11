"""Reference populations (Vogel, C-Model) and FR10 export bookkeeping."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import CHAR, CheckConstraint, ForeignKey, Numeric, func
from sqlalchemy.orm import Mapped, mapped_column

from robby.models.base import Base, CreatedAt, UUIDPK
from robby.models.enums import ReferenceKind, pg_enum


class ReferenceSet(Base):
    __tablename__ = "reference_set"

    id: Mapped[UUIDPK]
    name: Mapped[str]
    kind: Mapped[ReferenceKind] = mapped_column(pg_enum(ReferenceKind, "reference_kind"))
    citation: Mapped[str]
    population: Mapped[str]
    version_label: Mapped[str]
    loaded_at: Mapped[datetime] = mapped_column(server_default=func.now())
    loaded_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("app_user.id"))
    created_at: Mapped[CreatedAt]


class ReferenceValue(Base):
    __tablename__ = "reference_value"
    __table_args__ = (
        CheckConstraint("robson_group BETWEEN 1 AND 10", name="robson_group_range"),
        CheckConstraint(
            "group_size_pct BETWEEN 0 AND 100 AND cs_rate_pct BETWEEN 0 AND 100",
            name="percentages_range",
        ),
    )

    reference_set_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("reference_set.id"), primary_key=True
    )
    robson_group: Mapped[int] = mapped_column(primary_key=True)
    group_size_pct: Mapped[Decimal] = mapped_column(Numeric(5, 2))
    cs_rate_pct: Mapped[Decimal] = mapped_column(Numeric(5, 2))
    created_at: Mapped[CreatedAt]


class ExportRecord(Base):
    __tablename__ = "export_record"
    __table_args__ = (
        CheckConstraint("EXTRACT(DAY FROM period_month) = 1", name="period_is_month_start"),
    )

    id: Mapped[UUIDPK]
    facility_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("facility.id"))
    period_month: Mapped[date]
    exported_at: Mapped[datetime] = mapped_column(server_default=func.now())
    exported_by: Mapped[uuid.UUID] = mapped_column(ForeignKey("app_user.id"))
    file_sha256: Mapped[str] = mapped_column(CHAR(64))
    created_at: Mapped[CreatedAt]
