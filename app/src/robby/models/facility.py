"""facility: a maternity unit taking part."""

from __future__ import annotations

from sqlalchemy import CheckConstraint, true
from sqlalchemy.orm import Mapped, mapped_column

from robby.models.base import Base, CreatedAt, UUIDPK


class Facility(Base):
    __tablename__ = "facility"
    __table_args__ = (CheckConstraint("code ~ '^[A-Z]{2,6}$'", name="code_format"),)

    id: Mapped[UUIDPK]
    name: Mapped[str] = mapped_column(unique=True)
    code: Mapped[str] = mapped_column(unique=True)
    district: Mapped[str]
    model_facility_key: Mapped[str | None]
    active: Mapped[bool] = mapped_column(default=True, server_default=true())
    created_at: Mapped[CreatedAt]
