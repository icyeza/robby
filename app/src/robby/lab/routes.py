"""Lab request pages. The clinician asks for tests from the patient page and may cancel an open
request with a reason; the clinician also asks the lab, from Find patient, to enter a patient who
has no admission yet. The lab technician sees both kinds of open request of their facility in one
list (with a count in the navigation), records each result, and enters each admission asked for
(on the admission form, clinical.routes). The facility always comes from the signed-in user;
another facility's admission or request is a 404."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from sqlalchemy.orm import Session

from robby.auth.dependencies import CurrentUser, DbDep, require
from robby.clinical import service as clinical
from robby.clinical.form import PATIENT_NAME_MAX, PATIENT_NUMBER, PATIENT_NUMBER_ERROR
from robby.clinical.routes import Clinician, LabTechnician, runtime_of
from robby.errors import NotAllowed, NotFound
from robby.lab import admission_requests, catalogue, service, views
from robby.models import Admission, LabTest, Role
from robby.web.csrf import CsrfChecked
from robby.web.errors import error_page, redirect
from robby.web.render import render

clinician_router = APIRouter(prefix="/admissions")
ask_router = APIRouter(prefix="/patients/admission-requests")
router = APIRouter(prefix="/lab-requests")
# The badge polls by itself: it must not keep an idle session alive.
LabTechnicianPolling = Annotated[CurrentUser, Depends(require(Role.LAB_TECHNICIAN, touch=False))]


def _admission(db: Session, current: CurrentUser, code: str) -> Admission:
    try:
        return clinical.get_admission(db, current.user, code)
    except NotFound:
        raise HTTPException(status_code=404) from None


def _line(
    db: Session, current: CurrentUser, request_id: uuid.UUID
) -> tuple[service.LabLine, Admission]:
    try:
        return service.get_line(db, current.user, request_id)
    except NotFound:
        raise HTTPException(status_code=404) from None


def _replaced(request: Request, db: Session, admission: Admission) -> Response | None:
    newer = clinical.replacement_of(db, admission)
    if newer is None:
        return None
    return error_page(
        request,
        409,
        title="This admission has been replaced",
        message=f"{admission.code} was replaced by {newer.code}. Use {newer.code} instead.",
    )


# ---------------------------------------------------------------- clinician


def _request_page(
    request: Request,
    db: Session,
    current: CurrentUser,
    admission: Admission,
    values: dict[str, object],
    errors: dict[str, str],
    status_code: int = 200,
) -> Response:
    waiting = service.open_tests(db, admission)
    return render(
        request,
        "lab/request.html",
        current=current,
        status_code=status_code,
        admission=admission,
        tests=[(t.value, catalogue.test_name(t), t in waiting) for t in catalogue.TEST_ORDER],
        values=values,
        errors=errors,
        note_max=catalogue.NOTE_MAX,
    )


@clinician_router.get("/{code}/lab-requests/new")
def request_form(request: Request, code: str, db: DbDep, current: Clinician) -> Response:
    admission = _admission(db, current, code)
    replaced = _replaced(request, db, admission)
    if replaced is not None:
        return replaced
    return _request_page(request, db, current, admission, {"tests": []}, {})


@clinician_router.post("/{code}/lab-requests")
async def request_save(
    request: Request, code: str, db: DbDep, current: Clinician, _csrf: CsrfChecked
) -> Response:
    admission = _admission(db, current, code)
    replaced = _replaced(request, db, admission)
    if replaced is not None:
        return replaced
    form = await request.form()
    chosen = [v for v in form.getlist("test") if isinstance(v, str)]
    note = str(form.get("note") or "").strip()
    urgent = form.get("urgent") == "1"
    values: dict[str, object] = {"tests": chosen, "note": note, "urgent": urgent}
    errors: dict[str, str] = {}
    known = {t.value for t in LabTest}
    if not chosen or any(v not in known for v in chosen):
        errors["test"] = "Choose at least one test."
    if len(note) > catalogue.NOTE_MAX:
        errors["note"] = f"Keep the note to {catalogue.NOTE_MAX} characters or fewer."
    if not errors:
        try:
            service.request_tests(
                db,
                current.user,
                admission,
                [LabTest(v) for v in chosen],
                note=note or None,
                urgent=urgent,
            )
            db.commit()
            return redirect(request, f"/admissions/{admission.code}")
        except NotAllowed as exc:
            db.rollback()
            errors["test"] = str(exc)
    return _request_page(request, db, current, admission, values, errors, status_code=400)


def _cancel_page(
    request: Request,
    current: CurrentUser,
    admission: Admission,
    line: service.LabLine,
    values: dict[str, str],
    errors: dict[str, str],
    status_code: int = 200,
) -> Response:
    return render(
        request,
        "lab/cancel.html",
        current=current,
        status_code=status_code,
        admission=admission,
        line=views.line_view(line),
        values=values,
        errors=errors,
        reason_max=catalogue.REASON_MAX,
    )


def _line_of(
    db: Session, current: CurrentUser, code: str, request_id: uuid.UUID
) -> tuple[service.LabLine, Admission]:
    """The request, which must belong to the admission named in the address."""
    admission = _admission(db, current, code)
    line, owner = _line(db, current, request_id)
    if owner.id != admission.id:
        raise HTTPException(status_code=404)
    return line, admission


@clinician_router.get("/{code}/lab-requests/{request_id}/cancel")
def cancel_form(
    request: Request, code: str, request_id: uuid.UUID, db: DbDep, current: Clinician
) -> Response:
    line, admission = _line_of(db, current, code, request_id)
    if not line.is_open:
        return redirect(request, f"/admissions/{admission.code}")
    return _cancel_page(request, current, admission, line, {}, {})


@clinician_router.post("/{code}/lab-requests/{request_id}/cancel")
async def cancel_save(
    request: Request,
    code: str,
    request_id: uuid.UUID,
    db: DbDep,
    current: Clinician,
    _csrf: CsrfChecked,
) -> Response:
    line, admission = _line_of(db, current, code, request_id)
    form = await request.form()
    reason = str(form.get("reason") or "").strip()
    errors: dict[str, str] = {}
    if not reason:
        errors["reason"] = "Say why the request is cancelled."
    elif len(reason) > catalogue.REASON_MAX:
        errors["reason"] = f"Keep the reason to {catalogue.REASON_MAX} characters or fewer."
    if not errors:
        try:
            service.cancel_request(db, current.user, request_id, reason)
            db.commit()
            return redirect(request, f"/admissions/{admission.code}")
        except NotAllowed as exc:
            db.rollback()
            errors["_form"] = str(exc)
    values = {"reason": reason}
    return _cancel_page(request, current, admission, line, values, errors, status_code=400)


# ---------------------------------------------------------------- clinician: admission requests


def _ask_page(
    request: Request,
    current: CurrentUser,
    values: dict[str, object],
    errors: dict[str, str],
    status_code: int = 200,
) -> Response:
    return render(
        request,
        "lab/admission_request.html",
        current=current,
        status_code=status_code,
        values=values,
        errors=errors,
        note_max=catalogue.NOTE_MAX,
        name_max=PATIENT_NAME_MAX,
    )


@ask_router.get("/new")
def ask_form(request: Request, current: Clinician) -> Response:
    return _ask_page(request, current, {}, {})


@ask_router.post("")
async def ask_save(request: Request, db: DbDep, current: Clinician, _csrf: CsrfChecked) -> Response:
    form = await request.form()
    number = str(form.get("patient_number") or "").strip()
    name = " ".join(str(form.get("patient_name") or "").split())
    note = str(form.get("note") or "").strip()
    urgent = form.get("urgent") == "1"
    values: dict[str, object] = {
        "patient_number": number,
        "patient_name": name,
        "note": note,
        "urgent": urgent,
    }
    errors: dict[str, str] = {}
    if not number:
        errors["patient_number"] = "Enter the patient number."
    elif not PATIENT_NUMBER.fullmatch(number):
        errors["patient_number"] = PATIENT_NUMBER_ERROR
    if len(name) > PATIENT_NAME_MAX:
        errors["patient_name"] = f"Patient name must be at most {PATIENT_NAME_MAX} characters."
    if len(note) > catalogue.NOTE_MAX:
        errors["note"] = f"Keep the note to {catalogue.NOTE_MAX} characters or fewer."
    if not errors:
        try:
            admission_requests.request_admission(
                db,
                current.user,
                patient_number=number,
                patient_name=name or None,
                note=note or None,
                urgent=urgent,
            )
            db.commit()
            return redirect(request, "/patients")
        except NotAllowed as exc:
            db.rollback()
            errors["patient_number"] = str(exc)
    return _ask_page(request, current, values, errors, status_code=400)


def _ask_line(
    db: Session, current: CurrentUser, request_id: uuid.UUID
) -> admission_requests.AskLine:
    try:
        return admission_requests.get_line(db, current.user, request_id)
    except NotFound:
        raise HTTPException(status_code=404) from None


def _ask_cancel_page(
    request: Request,
    current: CurrentUser,
    line: admission_requests.AskLine,
    values: dict[str, str],
    errors: dict[str, str],
    status_code: int = 200,
) -> Response:
    return render(
        request,
        "lab/admission_cancel.html",
        current=current,
        status_code=status_code,
        ask=views.ask_view(line),
        values=values,
        errors=errors,
        reason_max=catalogue.REASON_MAX,
    )


@ask_router.get("/{request_id}/cancel")
def ask_cancel_form(
    request: Request, request_id: uuid.UUID, db: DbDep, current: Clinician
) -> Response:
    line = _ask_line(db, current, request_id)
    if not line.is_open:
        return redirect(request, "/patients")
    return _ask_cancel_page(request, current, line, {}, {})


@ask_router.post("/{request_id}/cancel")
async def ask_cancel_save(
    request: Request,
    request_id: uuid.UUID,
    db: DbDep,
    current: Clinician,
    _csrf: CsrfChecked,
) -> Response:
    line = _ask_line(db, current, request_id)
    form = await request.form()
    reason = str(form.get("reason") or "").strip()
    errors: dict[str, str] = {}
    if not reason:
        errors["reason"] = "Say why the request is cancelled."
    elif len(reason) > catalogue.REASON_MAX:
        errors["reason"] = f"Keep the reason to {catalogue.REASON_MAX} characters or fewer."
    if not errors:
        try:
            admission_requests.cancel_request(db, current.user, request_id, reason)
            db.commit()
            return redirect(request, "/patients")
        except NotAllowed as exc:
            db.rollback()
            errors["_form"] = str(exc)
    values = {"reason": reason}
    return _ask_cancel_page(request, current, line, values, errors, status_code=400)


# ---------------------------------------------------------------- lab technician


@router.get("")
def queue(
    request: Request, db: DbDep, current: LabTechnician, saved: str = "", entered: str = ""
) -> Response:
    just_saved = None
    if saved:
        try:
            line, admission = service.get_line(db, current.user, uuid.UUID(saved))
            just_saved = (catalogue.test_name(line.request.test), admission.code)
        except (NotFound, ValueError):
            just_saved = None
    just_entered = None
    if entered:
        try:
            just_entered = clinical.get_admission(db, current.user, entered)
        except NotFound:
            just_entered = None
    return render(
        request,
        "lab/queue.html",
        current=current,
        rows=views.queue_rows(
            service.queue(db, current.user), admission_requests.queue(db, current.user)
        ),
        saved=just_saved,
        entered=just_entered,
    )


@router.get("/count")
def count(db: DbDep, current: LabTechnicianPolling) -> Response:
    """The navigation badge: the number of open requests of both kinds, or nothing when there
    are none."""
    n = service.open_count(db, current.user) + admission_requests.open_count(db, current.user)
    return HTMLResponse(str(n) if n else "")


def _result_page(
    request: Request,
    current: CurrentUser,
    line: service.LabLine,
    admission: Admission,
    values: dict[str, str],
    errors: dict[str, str],
    status_code: int = 200,
) -> Response:
    return render(
        request,
        "lab/result.html",
        current=current,
        status_code=status_code,
        admission=admission,
        line=views.line_view(line),
        test=line.request.test.value,
        options=catalogue.options_of(line.request.test),
        values=values,
        errors=errors,
    )


@router.get("/{request_id}")
def result_form(
    request: Request, request_id: uuid.UUID, db: DbDep, current: LabTechnician
) -> Response:
    line, admission = _line(db, current, request_id)
    if not line.is_open:
        return redirect(request, "/lab-requests")
    return _result_page(request, current, line, admission, {}, {})


@router.post("/{request_id}")
async def result_save(
    request: Request,
    request_id: uuid.UUID,
    db: DbDep,
    current: LabTechnician,
    _csrf: CsrfChecked,
) -> Response:
    line, admission = _line(db, current, request_id)
    form = await request.form()
    raw = str(form.get("value") or "")
    value, error = catalogue.parse_result(line.request.test, raw)
    errors = {"value": error} if error else {}
    if value is not None:
        try:
            service.record_result(db, runtime_of(request), current.user, request_id, value)
            db.commit()
            return redirect(request, f"/lab-requests?saved={request_id}")
        except NotAllowed as exc:
            db.rollback()
            errors["_form"] = str(exc)
    return _result_page(
        request, current, line, admission, {"value": raw.strip()}, errors, status_code=400
    )
