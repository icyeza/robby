"""Clinician pages: new admission, live Robson panel, result, recent admissions, outcome and
correction. The facility always comes from the signed-in user; another facility's code is a 404."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from robby.auth.dependencies import CurrentUser, DbDep, require
from robby.clinical import classify as cls
from robby.clinical import labels, scoring, service
from robby.clinical.form import (
    DetailSpec,
    Parsed,
    detail_specs,
    parse,
    parse_time,
    values_from_admission,
)
from robby.errors import NotAllowed, NotFound
from robby.models import Admission, DeliveryMode, ModelState, ModelVersion, Role
from robby.timeutil import utcnow
from robby.web.csrf import CsrfChecked
from robby.web.errors import error_page, redirect
from robby.web.render import render

router = APIRouter(prefix="/admissions")
Clinician = Annotated[CurrentUser, Depends(require(Role.CLINICIAN))]


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
    mode: str,
    admission: Admission | None = None,
    status_code: int = 200,
    fragment: bool = False,
) -> Response:
    specs = _specs(request, db)
    settings = request.app.state.settings
    parsed, _ = parse(values, specs, settings.zone, utcnow(), screening_required=False)
    details_open = any(k.startswith("d_") and v for k, v in values.items()) or any(
        k.startswith("d_") for k in errors
    )
    if mode == "correct":
        assert admission is not None
        action = f"/admissions/{admission.code}/correct"
        heading = f"Correct admission {admission.code}"
        submit_label = "Save correction"
        ready = f"Saving creates a new code and replaces {admission.code}"
    else:
        action = "/admissions"
        heading = "New admission"
        submit_label = "Save and show result"
        ready = "Fill in the screening section to save"
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
        mode=mode,
        admission=admission,
        action=action,
        heading=heading,
        submit_label=submit_label,
        ready_note=ready,
        reasons=labels.REASONS,
        field_names=labels.FIELD_NAMES,
        presentation_options=labels.PRESENTATION_OPTIONS,
        plurality_options=labels.PLURALITY_OPTIONS,
        onset_options=labels.ONSET_OPTIONS,
        proteinuria_options=labels.PROTEINURIA_OPTIONS,
    )


def _now_text(request: Request) -> str:
    return utcnow().astimezone(request.app.state.settings.zone).strftime("%d %b %Y, %H:%M")


@router.get("/new")
def new_admission(request: Request, db: DbDep, current: Clinician) -> Response:
    return _form_page(
        request, db, current, values={"admitted_at": _now_text(request)}, errors={}, mode="new"
    )


@router.post("/preview")
async def preview(request: Request, db: DbDep, current: Clinician, _csrf: CsrfChecked) -> Response:
    values = await form_values(request)
    parsed, _ = parse(
        values, [], request.app.state.settings.zone, utcnow(), screening_required=False
    )
    return render(
        request,
        "clinical/_panel.html",
        current=current,
        panel=_panel_for(parsed, db),
        field_names=labels.FIELD_NAMES,
    )


def _save_response(
    request: Request,
    db: Session,
    current: CurrentUser,
    values: dict[str, str],
    *,
    mode: str,
    admission: Admission | None,
    correcting: bool,
) -> Response:
    settings = request.app.state.settings
    specs = _specs(request, db)
    parsed, errors = parse(values, specs, settings.zone, utcnow(), reason_required=correcting)
    htmx = request.headers.get("HX-Request") == "true"
    if correcting and not errors:
        assert admission is not None
        original, _ = parse(
            values_from_admission(admission, service.omissions_for(db, admission), settings.zone)
            | {"correction_reason": parsed.correction_reason or ""},
            specs,
            settings.zone,
            admission.admitted_at + timedelta(days=1),
        )
        if original == parsed:
            errors["_form"] = "Nothing was changed. Change at least one entry, or go back."
    if not errors:
        try:
            saved = service.save_admission(
                db, runtime_of(request), current.user, parsed, supersedes=admission
            )
            db.commit()
            return redirect(request, f"/admissions/{saved.admission.code}")
        except cls.NoRuleSetError:
            db.rollback()
            errors["_form"] = (
                "No Robson rule set is active, so nothing can be saved. "
                "Tell your system administrator."
            )
    return _form_page(
        request,
        db,
        current,
        values=values,
        errors=errors,
        mode=mode,
        admission=admission,
        status_code=200 if htmx else 400,
        fragment=htmx,
    )


@router.post("")
async def create_admission(
    request: Request, db: DbDep, current: Clinician, _csrf: CsrfChecked
) -> Response:
    values = await form_values(request)
    return _save_response(
        request, db, current, values, mode="new", admission=None, correcting=False
    )


def _screening_view(admission: Admission, omissions: list[Any]) -> dict[str, dict[str, str | None]]:
    reasons = {o.field.value: labels.REASON_TEXT[o.reason.value] for o in omissions}
    bp = (
        f"{admission.systolic_bp} / {admission.diastolic_bp} mmHg"
        if admission.systolic_bp is not None
        else ""
    )
    return {
        "bp": {"text": bp, "reason": reasons.get("blood_pressure")},
        "protein": {"text": admission.proteinuria or "", "reason": reasons.get("proteinuria")},
        "glucose": {
            "text": f"{admission.glucose_mmol} mmol/L"
            if admission.glucose_mmol is not None
            else "",
            "reason": reasons.get("glucose"),
        },
    }


@router.get("")
def recent(request: Request, db: DbDep, current: Clinician) -> Response:
    return render(
        request, "clinical/recent.html", current=current, rows=service.recent(db, current.user)
    )


@router.get("/{code}")
def result(request: Request, code: str, db: DbDep, current: Clinician) -> Response:
    admission = _get(db, current, code)
    classification = service.classification_of(db, admission)
    panel = cls.panel_view(
        status=classification.status.value,
        group=classification.robson_group,
        subgroup=classification.subgroup,
        candidates=list(classification.candidates),
        resolving=list(classification.resolving_fields),
        conflict=list(classification.conflict_fields),
        trace=classification.trace,
    )
    predicted = service.prediction_of(db, admission)
    view: dict[str, Any] = {
        "panel": panel,
        "percent": None,
        "saved_at": admission.created_at.astimezone(request.app.state.settings.zone).strftime(
            "%H:%M"
        ),
        "replaces": db.get(Admission, admission.supersedes_id) if admission.supersedes_id else None,
        "replaced_by": service.replacement_of(db, admission),
        "outcome": service.chain_outcome(db, admission),
        "screening": _screening_view(admission, service.omissions_for(db, admission)),
        "model_active": False,
        "not_given": [],
    }
    if view["outcome"] is not None:
        view["outcome_text"] = dict(labels.DELIVERY_OPTIONS)[view["outcome"].mode.value]
    if predicted is not None:
        prediction, model = predicted
        view["percent"] = service.percent_of(prediction.probability)
        view["model_label"] = model.version_label
        view["active_since"] = (
            service.activated_at(db, model)
            .astimezone(request.app.state.settings.zone)
            .strftime("%d %b %Y")
        )
        view["not_given"] = [
            labels.DETAIL_LABELS[name]
            for name, _, _ in labels.DETAIL_ORDER
            if name in prediction.feature_vector and prediction.feature_vector[name] is None
        ]
    else:
        view["model_active"] = (
            db.scalar(select(ModelVersion.id).where(ModelVersion.state == ModelState.ACTIVE))
            is not None
        )
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


@router.get("/{code}/correct")
def correct_form(request: Request, code: str, db: DbDep, current: Clinician) -> Response:
    admission, conflict = _current_or_conflict(request, db, current, code)
    if conflict is not None:
        return conflict
    values = values_from_admission(
        admission, service.omissions_for(db, admission), request.app.state.settings.zone
    )
    return _form_page(
        request, db, current, values=values, errors={}, mode="correct", admission=admission
    )


@router.post("/{code}/correct")
async def correct_save(
    request: Request, code: str, db: DbDep, current: Clinician, _csrf: CsrfChecked
) -> Response:
    admission, conflict = _current_or_conflict(request, db, current, code)
    if conflict is not None:
        return conflict
    values = await form_values(request)
    return _save_response(
        request, db, current, values, mode="correct", admission=admission, correcting=True
    )


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
    elif delivered > now + timedelta(minutes=5):
        errors["delivered_at"] = "Delivered at cannot be in the future."
    elif delivered < admission.admitted_at:
        errors["delivered_at"] = "Delivery cannot be before the admission time."
    if not errors:
        assert delivered is not None
        try:
            service.record_outcome(db, current.user, admission, DeliveryMode(mode), delivered)
            db.commit()
            return redirect(request, "/admissions")
        except NotAllowed as exc:
            db.rollback()
            errors["_form"] = str(exc)
    return _outcome_page(request, db, current, admission, values, errors, status_code=400)
