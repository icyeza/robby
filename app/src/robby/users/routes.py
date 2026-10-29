"""Users screen (mockup users.html): system administrators see every account; facility
administrators see and manage their own facility's accounts only."""

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from sqlalchemy.orm import Session

from robby.auth.dependencies import CurrentUser, DbDep, require
from robby.errors import FormErrors, NotAllowed, NotFound
from robby.facilities.service import active_facilities
from robby.models import ROLE_LABELS, AppUser, Role
from robby.timeutil import utcnow
from robby.users.service import (
    creatable_roles,
    create_user,
    get_user_in_scope,
    list_users,
    reset_password,
    set_active,
)
from robby.web.csrf import CsrfChecked
from robby.web.render import render

router = APIRouter(prefix="/admin/users")
UserAdmin = Annotated[CurrentUser, Depends(require(Role.SYSTEM_ADMIN, Role.FACILITY_ADMIN))]
FormText = Annotated[str, Form()]


def _list_page(
    request: Request,
    db: Session,
    current: CurrentUser,
    *,
    status_code: int = 200,
    open_dialog: bool = False,
    form: dict[str, Any] | None = None,
    errors: dict[str, str] | None = None,
    problem: str | None = None,
) -> Response:
    actor = current.user
    is_system_admin = actor.role is Role.SYSTEM_ADMIN
    facilities: list[tuple[str, str]] = []
    if is_system_admin:
        facilities = [(str(f.id), f.name) for f in active_facilities(db)]
    return render(
        request,
        "users/list.html",
        current=current,
        status_code=status_code,
        users=list_users(db, actor),
        roles=[(role.value, ROLE_LABELS[role]) for role in creatable_roles(actor)],
        facilities=facilities,
        is_system_admin=is_system_admin,
        now=utcnow(),
        open_dialog=open_dialog,
        form=form or {},
        errors=errors or {},
        problem=problem,
    )


def _target(db: Session, current: CurrentUser, user_id: uuid.UUID) -> AppUser:
    try:
        return get_user_in_scope(db, current.user, user_id)
    except NotFound:
        raise HTTPException(status_code=404) from None


def _parse_uuid(raw: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(raw)
    except ValueError:
        return None


def _password_page(
    request: Request, current: CurrentUser, heading: str, intro: str, username: str, password: str
) -> Response:
    return render(
        request,
        "users/password_shown.html",
        current=current,
        heading=heading,
        intro=intro,
        username=username,
        password=password,
        back_url="/admin/users",
        back_label="Back to users",
    )


@router.get("")
def users_page(request: Request, db: DbDep, current: UserAdmin, add: int = 0) -> Response:
    return _list_page(request, db, current, open_dialog=add == 1)


@router.post("")
def add_user(
    request: Request,
    db: DbDep,
    current: UserAdmin,
    _csrf: CsrfChecked,
    display_name: FormText = "",
    username: FormText = "",
    role: FormText = "",
    facility_id: FormText = "",
) -> Response:
    form = {
        "display_name": display_name,
        "username": username,
        "role": role,
        "facility_id": facility_id,
    }
    try:
        created = create_user(
            db,
            current.user,
            username=username,
            display_name=display_name,
            role=role,
            facility_id=_parse_uuid(facility_id),
        )
    except FormErrors as exc:
        db.rollback()
        return _list_page(
            request, db, current, status_code=400, open_dialog=True, form=form, errors=exc.errors
        )
    db.commit()
    assert created.temporary_password is not None
    return _password_page(
        request,
        current,
        "User added",
        f"The account {created.user.username} was created.",
        created.user.username,
        created.temporary_password,
    )


@router.post("/{user_id}/reset-password")
def reset(
    request: Request, user_id: uuid.UUID, db: DbDep, current: UserAdmin, _csrf: CsrfChecked
) -> Response:
    target = _target(db, current, user_id)
    temporary = reset_password(db, current.user, target, now=utcnow())
    db.commit()
    return _password_page(
        request,
        current,
        "Password reset",
        "The old password no longer works and any open sessions of this account were ended.",
        target.username,
        temporary,
    )


def _set_active(
    request: Request, user_id: uuid.UUID, db: Session, current: CurrentUser, active: bool
) -> Response:
    target = _target(db, current, user_id)
    try:
        set_active(db, current.user, target, active=active, now=utcnow())
    except NotAllowed as exc:
        return _list_page(request, db, current, status_code=400, problem=str(exc))
    db.commit()
    return RedirectResponse("/admin/users", status_code=303)


@router.post("/{user_id}/deactivate")
def deactivate(
    request: Request, user_id: uuid.UUID, db: DbDep, current: UserAdmin, _csrf: CsrfChecked
) -> Response:
    return _set_active(request, user_id, db, current, active=False)


@router.post("/{user_id}/reactivate")
def reactivate(
    request: Request, user_id: uuid.UUID, db: DbDep, current: UserAdmin, _csrf: CsrfChecked
) -> Response:
    return _set_active(request, user_id, db, current, active=True)
