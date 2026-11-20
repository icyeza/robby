"""policy_acceptance: a user's acceptance of the privacy notice and the terms of use, by
version. Insert-only (database triggers, migration 0010)."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from robby.models.base import UUIDPK, Base, CreatedAt


class PolicyAcceptance(Base):
    __tablename__ = "policy_acceptance"
    __table_args__ = (UniqueConstraint("user_id", "privacy_version", "terms_version"),)

    id: Mapped[UUIDPK]
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("app_user.id"), index=True)
    privacy_version: Mapped[str]
    terms_version: Mapped[str]
    accepted_at: Mapped[datetime]
    created_at: Mapped[CreatedAt]
