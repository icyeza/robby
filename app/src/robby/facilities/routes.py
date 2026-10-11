"""Facilities screen for system administrators (mockup facilities.html)."""

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from sqlalchemy.orm import Session

from robby.auth.dependencies import CurrentUser, DbDep, require
from robby.errors import FormErrors
from robby.facilities.service import create_facility, deactivate_facility, list_facilities
from robby.models import Facility, Role
from robby.users.service import CreatedUser, create_user
from robby.web.csrf import CsrfChecked
from robby.web.render import render

router = APIRouter(prefix="/admin/facilities")
SystemAdmin = Annotated[CurrentUser, Depends(require(Role.SYSTEM_ADMIN))]
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
) -> Response:
    return render(
        request, "facilities/list.html", current=current, status_code=status_code,
        rows=list_facilities(db), open_dialog=open_dialog, form=form or {}, errors=errors or {},
    )


@router.get("")
def facilities_page(request: Request, db: DbDep, current: SystemAdmin, add: int = 0) -> Response:
    return _list_page(request, db, current, open_dialog=add == 1)


@router.post("")
def add_facility(
    request: Request,
    db: DbDep,
    current: SystemAdmin,
    _csrf: CsrfChecked,
    name: FormText = "",
    code: FormText = "",
    district: FormText = "",
    admin_username: FormText = "",
    model_facility_key: FormText = "",
) -> Response:
    form = {
        "name": name, "code": code, "district": district,
        "admin_username": admin_username, "model_facility_key": model_facility_key,
    }
    created: CreatedUser | None = None
    try:
        facility = create_facility(
            db, current.user, name=name, code=code, district=district,
            model_facility_key=model_facility_key,
        )
        if admin_username.strip():
            try:
                created = create_user(
                    db, current.user, username=admin_username, role=Role.FACILITY_ADMIN,
                    facility_id=facility.id,
                )
            except FormErrors as exc:
                message = exc.errors.get("username", "This administrator could not be created.")
                raise FormErrors({"admin_username": message}) from exc
    except FormErrors as exc:
        db.rollback()  # neither the facility nor its administrator is kept
        return _list_page(
            request, db, current, status_code=400, open_dialog=True, form=form, errors=exc.errors
        )
    db.commit()
    if created is None:
        return RedirectResponse("/admin/facilities", status_code=303)
    return render(
        request, "users/password_shown.html", current=current,
        heading="Facility added",
        intro=f"{facility.name} ({facility.code}) was added with its first administrator.",
        username=created.user.username, password=created.temporary_password,
        back_url="/admin/facilities", back_label="Back to facilities",
    )


@router.post("/{facility_id}/deactivate")
def deactivate(
    facility_id: uuid.UUID, db: DbDep, current: SystemAdmin, _csrf: CsrfChecked
) -> Response:
    facility = db.get(Facility, facility_id)
    if facility is None:
        raise HTTPException(status_code=404)
    deactivate_facility(db, current.user, facility)
    db.commit()
    return RedirectResponse("/admin/facilities", status_code=303)
