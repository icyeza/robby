"""Request dependencies: the signed-in user, role checks, and the exceptions they raise.

Every route declares Depends(require(roles...)). Facility-scoped data always takes the facility
from the signed-in user (CurrentUser.user.facility_id), never from the request."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from robby.auth.service import SessionState, resolve_session
from robby.config import Settings
from robby.db import get_db
from robby.models import AppUser, Role, UserSession
from robby.timeutil import utcnow

SESSION_COOKIE = "robby_session"


class NotAuthenticated(Exception):
    """No usable session: go to the sign-in page."""


class SessionEnded(Exception):
    """The session expired: go to the "Session ended" page."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class PasswordChangeRequired(Exception):
    """The account must choose its own password before doing anything else."""


class Forbidden(Exception):
    """Signed in, but the role does not allow this page."""


@dataclass(frozen=True)
class CurrentUser:
    user: AppUser
    session: UserSession


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
DbDep = Annotated[Session, Depends(get_db)]


def current_session(request: Request, db: DbDep, settings: SettingsDep) -> CurrentUser:
    """The signed-in user, or NotAuthenticated / SessionEnded. Does not check roles or the
    forced password change (the change-password page itself uses this directly)."""
    resolved = resolve_session(db, settings, request.cookies.get(SESSION_COOKIE), now=utcnow())
    if resolved.state is SessionState.VALID and resolved.session is not None:
        db.commit()  # keeps the sliding last_seen_at, if it moved
        return CurrentUser(resolved.session.user, resolved.session)
    if resolved.state in (SessionState.IDLE, SessionState.ABSOLUTE):
        raise SessionEnded(resolved.state.value)
    raise NotAuthenticated


def current_session_untouched(request: Request, db: DbDep, settings: SettingsDep) -> CurrentUser:
    """current_session for a request the page sends by itself (polling): the idle timer is not
    moved, so an unattended screen still signs out after the idle limit."""
    resolved = resolve_session(
        db, settings, request.cookies.get(SESSION_COOKIE), now=utcnow(), touch=False
    )
    if resolved.state is SessionState.VALID and resolved.session is not None:
        return CurrentUser(resolved.session.user, resolved.session)
    if resolved.state in (SessionState.IDLE, SessionState.ABSOLUTE):
        raise SessionEnded(resolved.state.value)
    raise NotAuthenticated


def require(*roles: Role, touch: bool = True) -> Callable[[CurrentUser], CurrentUser]:
    """Dependency admitting the listed roles (any role when none are listed), after the forced
    password change. touch=False: see current_session_untouched."""
    allowed = frozenset(roles)
    session_of = current_session if touch else current_session_untouched

    def dependency(current: Annotated[CurrentUser, Depends(session_of)]) -> CurrentUser:
        if current.user.must_change_password:
            raise PasswordChangeRequired
        if allowed and current.user.role not in allowed:
            raise Forbidden
        return current

    return dependency
