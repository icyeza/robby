"""FHIR endpoints (FR8, NFR7).

Push: POST /api/v1/fhir/Bundle, authenticated with a bearer API token exactly like the import
API (the token's user gives the facility). Answers are FHIR resources: a transaction-response
Bundle (201 when something was saved, 200 when everything was already recorded), or an
OperationOutcome (422 with every issue and nothing saved; 400, 401, 415, 503).
GET /api/v1/fhir/metadata is the CapabilityStatement.

Lab requests (robby.lab), same token: GET /api/v1/fhir/ServiceRequest?status=active lists the
open requests of the token's facility; POST /api/v1/fhir/Observation with basedOn naming one of
them records its result exactly as the lab technician's form does (201; 422 with the issues;
404 another facility's or unknown request; 409 already closed).

Pull: POST /admissions/fetch, the lab technician's "Fetch from hospital system" button on the
admission form, shown only when ROBBY_FHIR_BASE_URL is set. It pre-fills the form for review and
never saves; when the hospital system fails, the form says so and stays as it was."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from robby.auth import api_tokens
from robby.auth.dependencies import DbDep
from robby.clinical import classify as cls
from robby.clinical.form import PATIENT_NUMBER, TIME_SHOWN, parse_time
from robby.clinical.routes import LabTechnician, _form_page, form_values, runtime_of
from robby.clinical.routes import _specs as specs_of
from robby.config import Settings
from robby.errors import NotAllowed, NotFound
from robby.fhir import adapter, intake, resources
from robby.fhir import lab as fhir_lab
from robby.fhir.client import MANUAL, FetchProblem, FhirClient, HospitalRecords
from robby.lab import service as lab
from robby.models import AppUser
from robby.timeutil import utcnow
from robby.web.csrf import CsrfChecked

logger = logging.getLogger("robby")
api_router = APIRouter(prefix="/api/v1/fhir")
form_router = APIRouter(prefix="/admissions")
ACCEPTED_TYPES = frozenset({resources.FHIR_JSON, "application/json"})


def _fhir(status: int, body: dict[str, Any], headers: dict[str, str] | None = None) -> Response:
    return JSONResponse(body, status_code=status, media_type=resources.FHIR_JSON, headers=headers)


@api_router.get("/metadata")
def metadata() -> Response:
    return _fhir(200, resources.capability_statement(utcnow()))


@api_router.post("/Bundle")
async def post_bundle(request: Request, db: DbDep) -> Response:
    now = utcnow()
    user = api_tokens.authenticate(db, request.headers.get("authorization"), now)
    if user is None:
        return _fhir(
            401,
            resources.problem("Missing, invalid or revoked API token.", "login"),
            {"WWW-Authenticate": "Bearer"},
        )
    db.commit()  # keeps last_used_at
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type not in ACCEPTED_TYPES:
        message = "Send the Bundle as application/fhir+json (or application/json)."
        return _fhir(415, resources.problem(message, "not-supported"))
    data = await request.body()
    try:
        payload = await request.json()
    except ValueError:
        return _fhir(400, resources.problem("The body is not valid JSON.", "structure"))
    wrong = adapter.not_a_bundle(payload)
    if wrong is not None:
        return _fhir(400, resources.problem(wrong, "structure"))

    settings: Settings = request.app.state.settings
    specs = specs_of(request, db)
    read = adapter.read_bundle(
        payload, settings.zone, patient_id_system=settings.fhir_patient_id_system
    )
    checked = intake.check(read, specs, settings.zone, now)
    if checked.issues:
        intake.record_rejection(db, user, len(checked.issues))
        db.commit()
        return _fhir(422, resources.operation_outcome(checked.issues))
    try:
        outcomes = intake.save(
            db, runtime_of(request), user, checked, hashlib.sha256(data).hexdigest()
        )
        db.commit()
    except cls.NoRuleSetError:
        db.rollback()
        message = "No Robson rule set is active, so nothing can be saved."
        return _fhir(503, resources.problem(message, "transient"))
    created = any(o.state == "created" for o in outcomes)
    return _fhir(201 if created else 200, resources.transaction_response(outcomes))


# ---------------------------------------------------------------- lab requests


def _token_user(request: Request, db: DbDep) -> AppUser | Response:
    """The lab technician of the bearer token, or the 401 answer."""
    user = api_tokens.authenticate(db, request.headers.get("authorization"), utcnow())
    if user is None:
        return _fhir(
            401,
            resources.problem("Missing, invalid or revoked API token.", "login"),
            {"WWW-Authenticate": "Bearer"},
        )
    db.commit()  # keeps last_used_at
    return user


@api_router.get("/ServiceRequest")
def service_requests(request: Request, db: DbDep, status: str = "active") -> Response:
    user = _token_user(request, db)
    if isinstance(user, Response):
        return user
    if status != "active":
        message = "Only status=active (requests waiting for a result) can be searched."
        return _fhir(400, resources.problem(message, "not-supported"))
    settings: Settings = request.app.state.settings
    rows = lab.queue(db, user)
    return _fhir(200, fhir_lab.searchset(rows, settings.fhir_patient_id_system))


@api_router.post("/Observation")
async def post_observation(request: Request, db: DbDep) -> Response:
    user = _token_user(request, db)
    if isinstance(user, Response):
        return user
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type not in ACCEPTED_TYPES:
        message = "Send the Observation as application/fhir+json (or application/json)."
        return _fhir(415, resources.problem(message, "not-supported"))
    try:
        payload = await request.json()
    except ValueError:
        return _fhir(400, resources.problem("The body is not valid JSON.", "structure"))
    if not isinstance(payload, dict) or payload.get("resourceType") != "Observation":
        return _fhir(400, resources.problem("The body is not an Observation.", "structure"))
    request_id = fhir_lab.based_on(payload)
    if isinstance(request_id, adapter.Issue):
        return _fhir(422, resources.operation_outcome([request_id]))
    try:
        line, _ = lab.get_line(db, user, request_id)
    except NotFound:
        message = f"No lab request ServiceRequest/{request_id} at this facility."
        return _fhir(404, resources.problem(message, "not-found"))
    read = fhir_lab.read_result(line.request.test, payload)
    if read.value is None:
        return _fhir(422, resources.operation_outcome(read.issues))
    try:
        done = lab.record_result(db, runtime_of(request), user, request_id, read.value)
        db.commit()
    except NotAllowed as exc:
        db.rollback()
        return _fhir(409, resources.problem(str(exc), "conflict"))
    return _fhir(201, fhir_lab.recorded(done.result.id, line.request))


# ---------------------------------------------------------------- pull mode (the form button)


@dataclass(frozen=True)
class FetchNote:
    """What the form says after "Fetch from hospital system"."""

    ok: bool
    message: str
    unused: list[str] = field(default_factory=list)  # values that could not be filled in


def records_of(request: Request, settings: Settings) -> HospitalRecords:
    """The hospital system (tests may put a fake on app.state.fhir_records, or an httpx2
    transport on app.state.fhir_transport)."""
    fake: HospitalRecords | None = getattr(request.app.state, "fhir_records", None)
    if fake is not None:
        return fake
    assert settings.fhir_base_url
    return FhirClient(
        settings.fhir_base_url,
        settings.fhir_timeout_seconds,
        patient_id_system=settings.fhir_patient_id_system,
        transport=getattr(request.app.state, "fhir_transport", None),
    )


SCREENING_KEYS = {"bp": ("bp_sys", "bp_dia"), "protein": ("proteinuria",), "glucose": ("glucose",)}


def prefilled(typed: dict[str, str], fetched: adapter.Mapped, settings: Settings) -> dict[str, str]:
    """The form values with the fetched ones written over what was typed. A screening test
    fetched as a value clears "not measured"; one fetched as not measured clears the value."""
    values = {k: v for k, v in typed.items() if k != "csrf_token"}
    values.update(fetched.values)
    for key, fields in SCREENING_KEYS.items():
        if f"{key}_reason" in fetched.values:
            values[f"{key}_nm"] = "1"
            for name in fields:
                values.pop(name, None)
        elif any(name in fetched.values for name in fields):
            values.pop(f"{key}_nm", None)
            values.pop(f"{key}_reason", None)
    when = parse_time(fetched.values.get("admitted_at", ""), settings.zone)
    if when is not None:
        values["admitted_at"] = when.strftime(TIME_SHOWN)
    return values


@form_router.post("/fetch")
async def fetch_admission(
    request: Request, db: DbDep, current: LabTechnician, _csrf: CsrfChecked
) -> Response:
    settings: Settings = request.app.state.settings
    if not settings.fhir_base_url:
        raise HTTPException(status_code=404)
    values = await form_values(request)
    number = values.get("patient_number", "").strip()
    note: FetchNote
    if not PATIENT_NUMBER.fullmatch(number):
        note = FetchNote(False, "Enter the patient number first, then fetch.")
    else:
        try:
            bundle = await records_of(request, settings).admission_bundle(number)
            read = adapter.read_bundle(
                bundle,
                settings.zone,
                patient_id_system=settings.fhir_patient_id_system,
                require_screening=False,
            )
            if not read.admissions:
                raise FetchProblem("The hospital system sent no admission for this patient.")
        except FetchProblem as exc:
            note = FetchNote(False, str(exc))
        except Exception:  # NFR7: nothing the hospital sends may cost the lab technician the form
            logger.exception("FHIR fetch failed")
            note = FetchNote(False, f"The hospital system's answer could not be used. {MANUAL}")
        else:
            values = prefilled(values, read.admissions[0], settings)
            note = FetchNote(
                True,
                "Filled in from the hospital system. Check every value before saving; "
                "nothing is saved until you press Save admission.",
                [i.message for i in read.issues],
            )
    htmx = request.headers.get("HX-Request") == "true"
    return _form_page(
        request, db, current, values=values, errors={}, fragment=htmx, fetch_note=note
    )
