"""event_log: append-only, hash-chained record of every state change (NFR5)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import CHAR, BigInteger, ForeignKey, Identity, Index
from sqlalchemy.orm import Mapped, mapped_column

from robby.models.base import Base, CreatedAt


class EventLog(Base):
    __tablename__ = "event_log"
    __table_args__ = (
        Index("ix_event_log_occurred_at", "occurred_at"),
        Index("ix_event_log_entity_type_entity_id", "entity_type", "entity_id"),
    )

    seq: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    id: Mapped[uuid.UUID] = mapped_column(unique=True, default=uuid.uuid4)
    occurred_at: Mapped[datetime]
    actor_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("app_user.id"))
    event_type: Mapped[str]
    entity_type: Mapped[str | None]
    entity_id: Mapped[uuid.UUID | None]
    payload: Mapped[dict[str, Any]]
    prev_hash: Mapped[str] = mapped_column(CHAR(64))
    hash: Mapped[str] = mapped_column(CHAR(64), unique=True)
    created_at: Mapped[CreatedAt]
