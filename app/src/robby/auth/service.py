"""Sign-in, sessions and password changes (FR1, NFR6). After MAX_FAILED_LOGINS failures in a
row an account is locked for LOCKOUT; a session ends after the configured idle time or absolute
lifetime, whichever comes first."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from functools import lru_cache

from sqlalchemy import select
from sqlalchemy.orm import Session

from robby import events
from robby.auth.passwords import hash_password, needs_rehash, password_problem, verify_password
from robby.auth.tokens import hash_token, new_session_token
from robby.config import Settings
from robby.models import AppUser, UserSession

MAX_FAILED_LOGINS = 5
LOCKOUT = timedelta(minutes=15)
TOUCH_INTERVAL = timedelta(minutes=1)
USER_AGENT_MAX = 400


class LoginFailed(Exception):
    """Sign-in refused. Callers show one generic message whatever the reason."""


class PasswordChangeError(ValueError):
    """The new password was not accepted; the message is shown to the user."""


class SessionState(StrEnum):
    VALID = "valid"
    MISSING = "missing"
    IDLE = "idle"
    ABSOLUTE = "absolute"


@dataclass(frozen=True)
class ResolvedSession:
    state: SessionState
    session: UserSession | None = None


@lru_cache(maxsize=1)
def _dummy_hash() -> str:
    """Verified against for unknown usernames so both failures take the same time."""
    return hash_password("unknown-user-timing-equaliser")


def normalise_username(raw: str) -> str:
    return raw.strip().lower()


def can_sign_in(user: AppUser) -> bool:
    return user.active and (user.facility is None or user.facility.active)


def start_session(
    db: Session,
    settings: Settings,
    user: AppUser,
    *,
    ip: str | None,
    user_agent: str | None,
    now: datetime,
) -> tuple[UserSession, str]:
    """Create a session row and return it with the token for the cookie (never stored)."""
    token, token_hash = new_session_token()
    row = UserSession(
        user_id=user.id,
        token_hash=token_hash,
        last_seen_at=now,
        expires_at=now + timedelta(hours=settings.session_absolute_hours),
        ip=ip,
        user_agent=user_agent[:USER_AGENT_MAX] if user_agent else None,
    )
    db.add(row)
    db.flush()
    return row, token


def _refuse(db: Session, user: AppUser | None, payload: dict[str, object]) -> LoginFailed:
    entity = ("app_user", user.id) if user is not None else None
    events.append(db, None, "session.login_failed", entity, payload)
    return LoginFailed()


def authenticate(
    db: Session,
    settings: Settings,
    username: str,
    password: str,
    *,
    ip: str | None,
    user_agent: str | None,
    now: datetime,
) -> tuple[UserSession, str]:
    """Check the credentials and open a session. Raises LoginFailed (after logging why).

    Five consecutive wrong passwords lock the account for 15 minutes; the counter restarts
    after a lock so the next lock needs five more."""
    user = db.scalar(select(AppUser).where(AppUser.username == normalise_username(username)))
    if user is None:
        verify_password(_dummy_hash(), password)
        raise _refuse(db, None, {"reason": "unknown_user"})
    if not can_sign_in(user):
        raise _refuse(db, user, {"reason": "inactive"})
    if user.locked_until is not None and user.locked_until > now:
        raise _refuse(db, user, {"reason": "locked"})
    if not verify_password(user.password_hash, password):
        user.failed_logins += 1
        locked = user.failed_logins >= MAX_FAILED_LOGINS
        if locked:
            user.locked_until = now + LOCKOUT
            user.failed_logins = 0
        raise _refuse(db, user, {"reason": "bad_password", "locked": locked})
    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(password)
    user.failed_logins = 0
    user.locked_until = None
    user.last_login_at = now
    row, token = start_session(db, settings, user, ip=ip, user_agent=user_agent, now=now)
    events.append(db, user, "session.login", ("user_session", row.id))
    return row, token


def resolve_session(
    db: Session, settings: Settings, token: str | None, *, now: datetime
) -> ResolvedSession:
    """Find the session for a cookie token and say whether it is usable.

    A valid session's last_seen_at slides forward at most once a minute (bookkeeping only: no
    event is logged for it)."""
    if not token:
        return ResolvedSession(SessionState.MISSING)
    row = db.scalar(select(UserSession).where(UserSession.token_hash == hash_token(token)))
    if row is None or row.revoked_at is not None or not can_sign_in(row.user):
        return ResolvedSession(SessionState.MISSING)
    if now >= row.expires_at:
        return ResolvedSession(SessionState.ABSOLUTE)
    if now - row.last_seen_at >= timedelta(minutes=settings.session_idle_minutes):
        return ResolvedSession(SessionState.IDLE)
    if now - row.last_seen_at >= TOUCH_INTERVAL:
        row.last_seen_at = now
        db.flush()
    return ResolvedSession(SessionState.VALID, row)


def logout(db: Session, session_row: UserSession, *, now: datetime) -> None:
    session_row.revoked_at = now
    db.flush()
    events.append(db, session_row.user, "session.logout", ("user_session", session_row.id))


def revoke_sessions(db: Session, user: AppUser, *, now: datetime) -> None:
    """End every open session of a user (part of a reset or deactivation; no event of its own)."""
    open_sessions = db.scalars(
        select(UserSession).where(UserSession.user_id == user.id, UserSession.revoked_at.is_(None))
    )
    for row in open_sessions:
        row.revoked_at = now
    db.flush()


def change_password(db: Session, user: AppUser, current: str, new: str, confirm: str) -> None:
    if not verify_password(user.password_hash, current):
        raise PasswordChangeError("Current password not recognised.")
    if new != confirm:
        raise PasswordChangeError("The new passwords do not match.")
    problem = password_problem(new, username=user.username)
    if problem is not None:
        raise PasswordChangeError(problem)
    if verify_password(user.password_hash, new):
        raise PasswordChangeError("Choose a password different from the current one.")
    user.password_hash = hash_password(new)
    user.must_change_password = False
    db.flush()
    events.append(db, user, "user.password_changed", ("app_user", user.id))
