"""Admission pages. The lab technician records admissions (form with a live Robson panel) and sees
their recent entries; the clinician finds a patient by name or number, reads the admission with
its Robson group and readiness signal, and records the outcome. Nothing entered can be edited
here. Import from Excel is for the facility administrator. The facility always comes from the
signed-in user; another facility's code is a 404."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy.orm import Session
from starlette.datastructures import UploadFile

from robby.auth.dependencies import CurrentUser, DbDep, require
from robby.clinical import classify as cls
from robby.clinical import importer, labels, scoring, service, xlsx_template
from robby.clinical.form import (
    FUTURE_SLACK,
    TIME_SHOWN,
    DetailSpec,
    Parsed,
    detail_specs,
    parse,
    parse_time,
)
from robby.clinical.views import result_view
from robby.errors import NotAllowed, NotFound
from robby.models import Admission, DeliveryMode, Role
from robby.timeutil import utcnow
from robby.web.csrf import CsrfChecked
from robby.web.errors import error_page, redirect
from robby.web.render import render

router = APIRouter(prefix="/admissions")
patients_router = APIRouter(prefix="/patients")
Clinician = Annotated[CurrentUser, Depends(require(Role.CLINICIAN))]
LabTechnician = Annotated[CurrentUser, Depends(require(Role.LAB_TECHNICIAN))]
FacilityAdmin = Annotated[CurrentUser, Depends(require(Role.FACILITY_ADMIN))]


def runtime_of(request: Request) -> scoring.ModelRuntime:
    """One model cache per process, created on first use (tests may put their own on app.state)."""
    existing: scoring.ModelRuntime | None = getattr(request.app.state, "model_runtime", None)
    if existing is None:
        existing = scoring.ModelRuntime(request.app.state.settings.artefact_dir)
        request.app.state.model_runtime = existing
    return existing


async def form_values(request: Request) -> dict[str, str]:
    data = await request.form()
    return {k: v for k, v in data.items() if isinstance(v, str)}


def _get(db: Session, current: CurrentUser, code: str) -> Admission:
    try:
        return service.get_admission(db, current.user, code)
    except NotFound:
        raise HTTPException(status_code=404) from None


def _specs(request: Request, db: Session) -> list[DetailSpec]:
    try:
        rule_row, engine_rules = cls.active_rule_set(db)
    except cls.NoRuleSetError:
        return []
    model = runtime_of(request).active(db, rule_row.id, engine_rules)
    return detail_specs(model.scorer.schema) if model is not None else []


def _panel_for(parsed: Parsed, db: Session) -> cls.PanelView:
    try:
        _, engine_rules = cls.active_rule_set(db)
    except cls.NoRuleSetError:
        view = cls.PanelView(state="empty", heading="No rule set is active")
        view.foot = "Tell your system administrator that no Robson rule set is loaded."
        return view
    result, trace = service.classify_parsed(parsed, engine_rules)
    return cls.view_of_result(result, trace)


def _form_page(
    request: Request,
    db: Session,
    current: CurrentUser,
    *,
    values: Mapping[str, str],
    errors: dict[str, str],
    status_code: int = 200,
    fragment: bool = False,
    saved: Admission | None = None,
    fetch_note: object = None,  # robby.fhir.routes.FetchNote
) -> Response:
    specs = _specs(request, db)
    settings = request.app.state.settings
    parsed, _ = parse(values, specs, settings.zone, utcnow(), screening_required=False)
    details_open = any(k.startswith("d_") and v for k, v in values.items()) or any(
        k.startswith("d_") for k in errors
    )
    return render(
        request,
        "clinical/_form.html" if fragment else "clinical/form.html",
        current=current,
        status_code=status_code,
        values=values,
        errors=errors,
        specs=specs,
        details_open=details_open,
        panel=_panel_for(parsed, db),
        saved=saved,
        saved_group=service.classification_of(db, saved).robson_group if saved else None,
        action="/admissions",
        heading="New admission",
        submit_label="Save admission",
        ready_note="Fill in the screening section to save",
        reasons=labels.REASONS,
        field_names=labels.FIELD_NAMES,
        presentation_options=labels.PRESENTATION_OPTIONS,
        plurality_options=labels.PLURALITY_OPTIONS,
        onset_options=labels.ONSET_OPTIONS,
        proteinuria_options=labels.PROTEINURIA_OPTIONS,
        # FHIR pull mode (robby/fhir): the "Fetch from hospital system" button and its result.
        fhir_pull=bool(settings.fhir_base_url),
        fetch_note=fetch_note,
    )


def _now_text(request: Request) -> str:
    return utcnow().astimezone(request.app.state.settings.zone).strftime(TIME_SHOWN)


@router.get("/new")
def new_admission(request: Request, db: DbDep, current: LabTechnician, saved: str = "") -> Response:
    just_saved = None
    if saved:
        try:
            just_saved = service.get_admission(db, current.user, saved)
        except NotFound:
            just_saved = None
    return _form_page(
        request,
        db,
        current,
        values={"admitted_at": _now_text(request)},
        errors={},
        saved=just_saved,
    )


@router.post("/preview")
async def preview(
    request: Request, db: DbDep, current: LabTechnician, _csrf: CsrfChecked
) -> Response:
    values = await form_values(request)
    parsed, _ = parse(
        values,
        [],
        request.app.state.settings.zone,
        utcnow(),
        screening_required=False,
        patient_required=False,
    )
    return render(
        request,
        "clinical/_panel.html",
        current=current,
        panel=_panel_for(parsed, db),
        field_names=labels.FIELD_NAMES,
    )


@router.post("")
async def create_admission(
    request: Request, db: DbDep, current: LabTechnician, _csrf: CsrfChecked
) -> Response:
    values = await form_values(request)
    settings = request.app.state.settings
    parsed, errors = parse(values, _specs(request, db), settings.zone, utcnow())
    htmx = request.headers.get("HX-Request") == "true"
    if not errors:
        try:
            saved = service.save_admission(db, runtime_of(request), current.user, parsed)
            db.commit()
            return redirect(request, f"/admissions/new?saved={saved.admission.code}")
        except cls.NoRuleSetError:
            db.rollback()
            errors["_form"] = cls.NOTHING_SAVED_ASK_ADMIN
    return _form_page(
        request,
        db,
        current,
        values=values,
        errors=errors,
        status_code=200 if htmx else 400,
        fragment=htmx,
    )


@router.get("")
def recent(request: Request, db: DbDep, current: LabTechnician) -> Response:
    return render(
        request, "clinical/recent.html", current=current, rows=service.recent(db, current.user)
    )


@patients_router.get("")
def find_patient(request: Request, db: DbDep, current: Clinician, q: str = "") -> Response:
    query = q.strip()[:80]
    rows = (
        service.find_patients(db, current.user, query)
        if query
        else service.recent(db, current.user)
    )
    return render(
        request,
        "clinical/patients_results.html"
        if request.headers.get("HX-Request") == "true"
        else "clinical/patients.html",
        current=current,
        query=query,
        rows=rows,
        limit=service.SEARCH_LIMIT,
    )


# ---------------------------------------------------------------- import from Excel
# Facility administrators only, reached from the hospital report. Declared before /{code} so
# "import" is never read as an admission code.

XLSX_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
PROBLEMS_SHOWN = 100


@dataclass
class ImportedLine:
    code: str
    group: int | None
    percent: int | None
    already: bool = False  # recorded before this upload; shown, not imported again


def _imported_lines(
    db: Session,
    fresh: list[tuple[int, Parsed]],
    saved: list[service.Saved],
    recorded: dict[int, Admission],
) -> dict[int, ImportedLine]:
    """A line by Excel row: the admissions just saved, and those recorded before this upload
    (their group and percent read in one batch, not per row)."""
    lines = {
        n: ImportedLine(
            s.admission.code,
            s.classification.robson_group,
            service.percent_of(s.prediction.probability) if s.prediction else None,
        )
        for (n, _), s in zip(fresh, saved, strict=True)
    }
    summaries = service.summaries_of(db, list(recorded.values()))
    for n, admission in recorded.items():
        group, percent = summaries[admission.id]
        lines[n] = ImportedLine(admission.code, group, percent, already=True)
    return lines


def _import_page(
    request: Request,
    current: CurrentUser,
    *,
    problems: list[importer.RowProblem] | None = None,
    lines: list[ImportedLine] | None = None,
    status_code: int = 200,
    other_hospitals: int = 0,
    outcomes: int = 0,
) -> Response:
    problems = problems or []
    return render(
        request,
        "clinical/import.html",
        current=current,
        status_code=status_code,
        problems=problems[:PROBLEMS_SHOWN],
        more=max(0, len(problems) - PROBLEMS_SHOWN),
        lines=lines or [],
        imported=lines is not None,
        already=sum(1 for line in lines or [] if line.already),
        max_rows=importer.MAX_ROWS,
        skipped=len({p.row for p in problems}) if lines else 0,
        other_hospitals=other_hospitals,
        outcomes=outcomes,
    )


@router.get("/import")
def import_page(request: Request, current: FacilityAdmin) -> Response:
    return _import_page(request, current)


@router.get("/import/template.xlsx")
def import_template(request: Request, db: DbDep, current: FacilityAdmin) -> Response:
    return Response(
        xlsx_template.build_template(_specs(request, db)),
        media_type=XLSX_TYPE,
        headers={"Content-Disposition": 'attachment; filename="robby-admissions-template.xlsx"'},
    )


def _import_refused(request: Request, current: CurrentUser, message: str) -> Response:
    """The import page with one problem about the whole upload."""
    problem = importer.RowProblem(0, "", message)
    return _import_page(request, current, problems=[problem], status_code=400)


@router.post("/import")
async def import_upload(
    request: Request, db: DbDep, current: FacilityAdmin, _csrf: CsrfChecked
) -> Response:
    upload = (await request.form()).get("file")
    if not isinstance(upload, UploadFile) or not upload.filename:
        return _import_refused(request, current, "Choose a file to upload.")
    data = await upload.read()
    zone = request.app.state.settings.zone
    facility = service.facility_of(db, current.user)
    read = importer.read_upload(
        data,
        upload.filename,
        _specs(request, db),
        zone,
        utcnow(),
        skip_bad=True,
        facility_keys={facility.name, facility.model_facility_key or ""},
    )
    if not read.rows:
        return _import_page(request, current, problems=read.problems, status_code=400)
    scan = importer.find_duplicates(db, facility.id, read.rows)
    repeated = {p.row for p in scan.in_file}
    rows = [(n, parsed) for n, parsed in read.rows if n not in repeated]
    fresh = [(n, parsed) for n, parsed in rows if n not in scan.recorded]
    digest = hashlib.sha256(data).hexdigest()
    try:
        saved = (
            service.import_admissions(db, runtime_of(request), current.user, fresh, digest)
            if fresh
            else []
        )
        outcomes = service.record_import_outcomes(
            db, current.user, fresh, saved, read.outcomes, zone
        )
        db.commit()
    except cls.NoRuleSetError:
        db.rollback()
        return _import_refused(request, current, cls.NOTHING_SAVED_ASK_ADMIN)
    lines = _imported_lines(db, fresh, saved, scan.recorded)
    return _import_page(
        request,
        current,
        lines=[lines[n] for n, _ in rows],
        problems=read.problems + scan.in_file,
        other_hospitals=read.other_hospitals,
        outcomes=outcomes,
    )


@router.get("/{code}")
def result(request: Request, code: str, db: DbDep, current: Clinician) -> Response:
    admission = _get(db, current, code)
    view = result_view(db, admission, request.app.state.settings.zone)
    return render(request, "clinical/result.html", current=current, admission=admission, view=view)


def _current_or_conflict(
    request: Request, db: Session, current: CurrentUser, code: str
) -> tuple[Admission, Response | None]:
    admission = _get(db, current, code)
    newer = service.replacement_of(db, admission)
    if newer is not None:
        return admission, error_page(
            request,
            409,
            title="This admission has been replaced",
            message=f"{admission.code} was replaced by {newer.code}. Use {newer.code} instead.",
        )
    return admission, None


def _outcome_page(
    request: Request,
    db: Session,
    current: CurrentUser,
    admission: Admission,
    values: dict[str, str],
    errors: dict[str, str],
    status_code: int = 200,
) -> Response:
    return render(
        request,
        "clinical/outcome.html",
        current=current,
        status_code=status_code,
        admission=admission,
        group=service.classification_of(db, admission).robson_group,
        values=values,
        errors=errors,
        delivery_options=labels.DELIVERY_OPTIONS,
    )


@router.get("/{code}/outcome")
def outcome_form(request: Request, code: str, db: DbDep, current: Clinician) -> Response:
    admission, conflict = _current_or_conflict(request, db, current, code)
    if conflict is not None:
        return conflict
    return _outcome_page(request, db, current, admission, {"delivered_at": _now_text(request)}, {})


@router.post("/{code}/outcome")
async def outcome_save(
    request: Request, code: str, db: DbDep, current: Clinician, _csrf: CsrfChecked
) -> Response:
    admission, conflict = _current_or_conflict(request, db, current, code)
    if conflict is not None:
        return conflict
    values = await form_values(request)
    settings = request.app.state.settings
    errors: dict[str, str] = {}
    mode = values.get("mode", "")
    if mode not in dict(labels.DELIVERY_OPTIONS):
        errors["mode"] = "Choose the mode of delivery."
    delivered = parse_time(values.get("delivered_at", "").strip(), settings.zone)
    now = utcnow()
    if delivered is None:
        errors["delivered_at"] = "Delivered at must look like 05 Oct 2026, 19:40."
    elif delivered > now + FUTURE_SLACK:
        errors["delivered_at"] = "Delivered at cannot be in the future."
    elif delivered < admission.admitted_at:
        errors["delivered_at"] = "Delivery cannot be before the admission time."
    if not errors:
        assert delivered is not None
        try:
            service.record_outcome(db, current.user, admission, DeliveryMode(mode), delivered)
            db.commit()
            return redirect(request, f"/admissions/{admission.code}")
        except NotAllowed as exc:
            db.rollback()
            errors["_form"] = str(exc)
    return _outcome_page(request, db, current, admission, values, errors, status_code=400)
