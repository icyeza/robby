"""Bearer tokens for automated imports. A token acts as one lab technician account: its admissions
belong to that account's facility, exactly like the technician's own entries. Only the SHA-256 of a
token is stored; the token itself is shown once, when it is created."""

from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from robby import events
from robby.auth.tokens import hash_token
from robby.models import ApiToken, AppUser, Role

PREFIX = "rbk_"
TOUCH_EVERY = timedelta(minutes=5)  # last_used_at is refreshed at most this often


class TokenError(Exception):
    """The token cannot be created or revoked; the message says why."""


def create(db: Session, actor: AppUser | None, username: str, label: str) -> tuple[str, ApiToken]:
    """A new token for the lab technician account `username`, and the token text (shown once)."""
    user = db.scalar(select(AppUser).where(AppUser.username == username.strip().lower()))
    if user is None or not user.active:
        raise TokenError(f"No active account named “{username}”.")
    if user.role is not Role.LAB_TECHNICIAN:
        raise TokenError("A token can only act as a lab technician account.")
    if not label.strip():
        raise TokenError("Give the token a label, for example “nightly feed from the lab system”.")
    token = PREFIX + secrets.token_urlsafe(32)
    row = ApiToken(
        user_id=user.id,
        label=label.strip(),
        token_hash=hash_token(token),
        created_by=actor.id if actor else None,
    )
    db.add(row)
    db.flush()
    events.append(
        db,
        actor,
        "api_token.created",
        ("api_token", row.id),
        {"username": user.username, "label": row.label},
    )
    return token, row


def revoke(db: Session, actor: AppUser | None, token_id: uuid.UUID, now: datetime) -> ApiToken:
    row = db.get(ApiToken, token_id)
    if row is None:
        raise TokenError("No such token.")
    if row.revoked_at is None:
        row.revoked_at = now
        events.append(db, actor, "api_token.revoked", ("api_token", row.id), {"label": row.label})
    return row


def authenticate(db: Session, authorization: str | None, now: datetime) -> AppUser | None:
    """The lab technician a `Authorization: Bearer <token>` header stands for, or None."""
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization[7:].strip()
    if not token.startswith(PREFIX):
        return None
    row = db.scalar(select(ApiToken).where(ApiToken.token_hash == hash_token(token)))
    if row is None or row.revoked_at is not None:
        return None
    user = row.user
    if not user.active or user.role is not Role.LAB_TECHNICIAN or user.facility_id is None:
        return None
    if row.last_used_at is None or now - row.last_used_at > TOUCH_EVERY:
        row.last_used_at = now
    return user


def all_tokens(db: Session) -> list[ApiToken]:
    return list(db.scalars(select(ApiToken).order_by(ApiToken.created_at)))
