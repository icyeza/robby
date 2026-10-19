"""app_user and user_session."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import CHAR, CheckConstraint, ForeignKey, text, true
from sqlalchemy.orm import Mapped, mapped_column, relationship

from robby.models.base import UUIDPK, Base, CreatedAt
from robby.models.enums import Role, pg_enum
from robby.models.facility import Facility


class AppUser(Base):
    __tablename__ = "app_user"
    __table_args__ = (
        CheckConstraint("username = lower(username)", name="username_lowercase"),
        CheckConstraint(
            "(role IN ('clinician', 'facility_admin')) = (facility_id IS NOT NULL)",
            name="facility_matches_role",
        ),
        CheckConstraint("failed_logins >= 0", name="failed_logins_not_negative"),
    )

    id: Mapped[UUIDPK]
    username: Mapped[str] = mapped_column(unique=True)
    display_name: Mapped[str | None]
    password_hash: Mapped[str]
    role: Mapped[Role] = mapped_column(pg_enum(Role, "user_role"))
    facility_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("facility.id"))
    active: Mapped[bool] = mapped_column(default=True, server_default=true())
    must_change_password: Mapped[bool] = mapped_column(default=True, server_default=true())
    failed_logins: Mapped[int] = mapped_column(default=0, server_default=text("0"))
    locked_until: Mapped[datetime | None]
    last_login_at: Mapped[datetime | None]
    created_at: Mapped[CreatedAt]

    facility: Mapped[Facility | None] = relationship(lazy="joined")


class UserSession(Base):
    __tablename__ = "user_session"

    id: Mapped[UUIDPK]
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("app_user.id"), index=True)
    token_hash: Mapped[str] = mapped_column(CHAR(64), unique=True)
    created_at: Mapped[CreatedAt]
    last_seen_at: Mapped[datetime]
    expires_at: Mapped[datetime]
    revoked_at: Mapped[datetime | None]
    ip: Mapped[str | None]
    user_agent: Mapped[str | None]

    user: Mapped[AppUser] = relationship(lazy="joined")


class ApiToken(Base):
    """A bearer token for automated imports, acting as one clinician account. Only the SHA-256 of
    the token is stored; it is shown once, when it is created."""

    __tablename__ = "api_token"

    id: Mapped[UUIDPK]
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("app_user.id"), index=True)
    label: Mapped[str]
    token_hash: Mapped[str] = mapped_column(CHAR(64), unique=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("app_user.id"))
    created_at: Mapped[CreatedAt]
    last_used_at: Mapped[datetime | None]
    revoked_at: Mapped[datetime | None]

    user: Mapped[AppUser] = relationship(foreign_keys=[user_id], lazy="joined")
