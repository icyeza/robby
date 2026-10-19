"""Integrations screen for facility administrators: create and revoke the API tokens of their
own facility's automatic feed (for example from OpenClinic), see when each was last used, and see
the latest batches. Another facility's tokens are a 404."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from robby.auth import api_tokens
from robby.auth.dependencies import CurrentUser, DbDep, require
from robby.models import ApiToken, AppUser, EventLog, Role
from robby.timeutil import utcnow
from robby.web.csrf import CsrfChecked
from robby.web.render import render

router = APIRouter(prefix="/admin/integrations")
FacilityAdmin = Annotated[CurrentUser, Depends(require(Role.FACILITY_ADMIN))]
FormText = Annotated[str, Form()]
BATCH_EVENTS = ("admissions.imported", "api_import.rejected", "api_import.nothing_new")
BATCHES_SHOWN = 20


@dataclass(frozen=True)
class Batch:
    when: object
    channel: str  # "Automatic feed" | "Excel upload"
    outcome: str
    detail: str
    by: str


def _facility_id(current: CurrentUser) -> uuid.UUID:
    facility_id = current.user.facility_id
    assert facility_id is not None  # a facility administrator always has one
    return facility_id


def _tokens(db: Session, facility_id: uuid.UUID) -> list[ApiToken]:
    return [t for t in api_tokens.all_tokens(db) if t.user.facility_id == facility_id]


def _batches(db: Session, current: CurrentUser) -> list[Batch]:
    code = current.user.facility.code if current.user.facility else ""
    rows = db.execute(
        select(EventLog, AppUser.username)
        .outerjoin(AppUser, AppUser.id == EventLog.actor_id)
        .where(EventLog.event_type.in_(BATCH_EVENTS), EventLog.payload["facility"].astext == code)
        .order_by(EventLog.seq.desc())
        .limit(BATCHES_SHOWN)
    )
    out: list[Batch] = []
    for event, username in rows:
        p = event.payload
        if event.event_type == "admissions.imported":
            channel = "Automatic feed" if p.get("channel") == "api" else "Excel upload"
            outcome, detail = "Imported", f"{p.get('rows', 0)} admissions"
        elif event.event_type == "api_import.rejected":
            channel, outcome = "Automatic feed", "Rejected"
            detail = f"{p.get('problems', 0)} problems, nothing saved"
        else:
            channel, outcome = "Automatic feed", "Nothing new"
            detail = f"{p.get('skipped', 0)} already recorded"
        out.append(Batch(event.occurred_at, channel, outcome, detail, username or "system"))
    return out


def _page(
    request: Request,
    db: Session,
    current: CurrentUser,
    *,
    problem: str | None = None,
    status: int = 200,
) -> Response:
    facility_id = _facility_id(current)
    clinicians = db.scalars(
        select(AppUser)
        .where(
            AppUser.facility_id == facility_id,
            AppUser.role == Role.CLINICIAN,
            AppUser.active,
        )
        .order_by(AppUser.username)
    )
    return render(
        request,
        "integrations/index.html",
        current=current,
        status_code=status,
        tokens=_tokens(db, facility_id),
        clinicians=[(str(u.id), u.username) for u in clinicians],
        batches=_batches(db, current),
        problem=problem,
    )


@router.get("")
def integrations_page(request: Request, db: DbDep, current: FacilityAdmin) -> Response:
    return _page(request, db, current)


@router.post("/tokens")
def create_token(
    request: Request,
    db: DbDep,
    current: FacilityAdmin,
    _csrf: CsrfChecked,
    label: FormText = "",
    clinician_id: FormText = "",
) -> Response:
    try:
        account = db.get(AppUser, uuid.UUID(clinician_id))
    except ValueError:
        account = None
    if account is None or account.facility_id != _facility_id(current):
        return _page(
            request, db, current, problem="Choose a clinician account of this facility.", status=400
        )
    try:
        token, _ = api_tokens.create(db, current.user, account.username, label)
    except api_tokens.TokenError as exc:
        db.rollback()
        return _page(request, db, current, problem=str(exc), status=400)
    db.commit()
    return render(
        request,
        "integrations/token_shown.html",
        current=current,
        token=token,
        username=account.username,
        label=label.strip(),
        base_url=str(request.base_url).rstrip("/"),
    )


@router.post("/tokens/{token_id}/revoke")
def revoke_token(
    token_id: uuid.UUID, db: DbDep, current: FacilityAdmin, _csrf: CsrfChecked
) -> Response:
    row = db.get(ApiToken, token_id)
    if row is None or row.user.facility_id != _facility_id(current):
        raise HTTPException(status_code=404)
    api_tokens.revoke(db, current.user, token_id, utcnow())
    db.commit()
    return RedirectResponse("/admin/integrations", status_code=303)
