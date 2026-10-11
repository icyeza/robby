"""rule_set: Robson rule sets loaded from YAML; exactly one may be active."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import CHAR, Index, false, func, text
from sqlalchemy.orm import Mapped, mapped_column

from robby.models.base import UUIDPK, Base, CreatedAt


class RuleSet(Base):
    __tablename__ = "rule_set"
    __table_args__ = (
        Index(
            "uq_rule_set_one_active", "is_active", unique=True, postgresql_where=text("is_active")
        ),
    )

    id: Mapped[UUIDPK]
    version_label: Mapped[str] = mapped_column(unique=True)
    checksum: Mapped[str] = mapped_column(CHAR(64))
    definition: Mapped[dict[str, Any]]
    loaded_at: Mapped[datetime] = mapped_column(server_default=func.now())
    is_active: Mapped[bool] = mapped_column(default=False, server_default=false())
    created_at: Mapped[CreatedAt]
