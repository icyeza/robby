"""Accounts: creation with temporary passwords, resets, deactivation, and who may do what.

System administrators manage every account. Facility administrators manage clinician and
facility-administrator accounts of their own facility only; the facility always comes from the
acting administrator, never from the request."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from robby import events
from robby.auth.passwords import generate_temporary_password, hash_password, password_problem
from robby.auth.service import revoke_sessions
from robby.errors import FormErrors, NotAllowed, NotFound
from robby.models import FACILITY_ROLES, AppUser, Facility, Role

USERNAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{2,63}")
ADMIN_ROLES = frozenset({Role.SYSTEM_ADMIN, Role.FACILITY_ADMIN})


@dataclass(frozen=True)
class CreatedUser:
    user: AppUser
    temporary_password: str | None


def creatable_roles(actor: AppUser) -> tuple[Role, ...]:
    if actor.role is Role.SYSTEM_ADMIN:
        return tuple(Role)
    if actor.role is Role.FACILITY_ADMIN:
        return (Role.CLINICIAN, Role.FACILITY_ADMIN)
    return ()


def _require_admin(actor: AppUser) -> None:
    if actor.role not in ADMIN_ROLES:
        raise NotAllowed("Only administrators manage accounts.")


def list_users(db: Session, actor: AppUser) -> list[AppUser]:
    _require_admin(actor)
    statement = select(AppUser).order_by(AppUser.username)
    if actor.role is Role.FACILITY_ADMIN:
        statement = statement.where(AppUser.facility_id == actor.facility_id)
    return list(db.scalars(statement))


def get_user_in_scope(db: Session, actor: AppUser, user_id: uuid.UUID) -> AppUser:
    _require_admin(actor)
    user = db.get(AppUser, user_id)
    if user is None or (actor.role is Role.FACILITY_ADMIN and user.facility_id != actor.facility_id):
        raise NotFound("No such account.")
    return user


def create_user(
    db: Session,
    actor: AppUser | None,
    *,
    username: str,
    role: Role | str,
    facility_id: uuid.UUID | None = None,
    display_name: str = "",
    password: str | None = None,
) -> CreatedUser:
    """Create an account. actor None is the CLI (any role). With no password, a temporary one is
    generated, returned once, and must be changed at first sign-in."""
    errors: dict[str, str] = {}
    name = username.strip().lower()
    if not USERNAME_PATTERN.fullmatch(name):
        errors["username"] = (
            "Use 3 to 64 characters: lowercase letters, digits, dots, dashes or underscores, "
            "starting with a letter or digit."
        )
    elif db.scalar(select(AppUser.id).where(AppUser.username == name)) is not None:
        errors["username"] = "This username is already taken."

    chosen: Role | None
    try:
        chosen = Role(role)
    except ValueError:
        chosen = None
        errors["role"] = "Choose a role."
    allowed = creatable_roles(actor) if actor is not None else tuple(Role)
    if chosen is not None and chosen not in allowed:
        errors["role"] = "You cannot create accounts with this role."

    if actor is not None and actor.role is Role.FACILITY_ADMIN:
        facility_id = actor.facility_id
    facility: Facility | None = None
    if chosen is not None and chosen in FACILITY_ROLES:
        facility = db.get(Facility, facility_id) if facility_id is not None else None
        if facility is None or not facility.active:
            errors["facility_id"] = "Choose the active facility this person works at."

    if password is not None:
        problem = password_problem(password, username=name)
        if problem is not None:
            errors["password"] = problem

    if errors or chosen is None:
        raise FormErrors(errors)

    temporary: str | None
    if password is None:
        secret = generate_temporary_password()
        temporary = secret
    else:
        secret = password
        temporary = None
    user = AppUser(
        username=name,
        display_name=display_name.strip() or None,
        password_hash=hash_password(secret),
        role=chosen,
        facility_id=facility.id if facility is not None else None,
        must_change_password=password is None,
    )
    db.add(user)
    db.flush()
    events.append(
        db, actor, "user.created", ("app_user", user.id),
        {
            "username": name,
            "role": chosen.value,
            "facility_id": str(facility.id) if facility is not None else None,
        },
    )
    return CreatedUser(user, temporary)


def reset_password(db: Session, actor: AppUser, target: AppUser, *, now: datetime) -> str:
    """Replace the password with a temporary one, unlock, and end the target's sessions."""
    temporary = generate_temporary_password()
    target.password_hash = hash_password(temporary)
    target.must_change_password = True
    target.failed_logins = 0
    target.locked_until = None
    revoke_sessions(db, target, now=now)
    db.flush()
    events.append(db, actor, "user.password_reset", ("app_user", target.id), {"username": target.username})
    return temporary


def set_active(
    db: Session, actor: AppUser, target: AppUser, *, active: bool, now: datetime
) -> None:
    if not active and target.id == actor.id:
        raise NotAllowed("You cannot deactivate your own account.")
    if target.active == active:
        return
    target.active = active
    if not active:
        revoke_sessions(db, target, now=now)
    db.flush()
    event_type = "user.reactivated" if active else "user.deactivated"
    events.append(db, actor, event_type, ("app_user", target.id), {"username": target.username})
