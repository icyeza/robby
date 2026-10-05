"""FastAPI app: Robson classification and the operational readiness signal.

Run from the ML folder::

    uv run uvicorn api.main:app --port 8000

then open http://127.0.0.1:8000/docs. ``ROBSON_MODEL_DIR`` points at another model artefact.
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse

from api.readiness import ReadinessModel, feature_row, load_readiness_model, model_dir
from api.schemas import (
    READINESS_LABEL,
    AdmissionRequest,
    AssessmentResponse,
    Classification,
    ErrorDetail,
    Health,
    ModelCard,
    Readiness,
    RobsonFields,
    Screening,
    TraceItem,
)
from robson_engine import ClassificationResult, RobsonInputs, RuleSet, classify, load_rule_set

DESCRIPTION = """
Deployment MVP of the Robson readiness project.

* **Robson classification** with the WHO Ten-Group rules. Missing inputs give a *partial*
  result listing the candidate groups and the fields that would settle it; contradictory
  inputs give a *conflict* naming the fields. Every result carries its explanation trace and
  the rule-set version.
* **Operational readiness signal**: a calibrated probability that the admission ends in a
  cesarean *under current practice* at facilities like this one. It is meant for planning
  theatre and staff readiness. **It is not a recommendation for or against cesarean.**
* If no model is loaded, admissions are still classified and the readiness signal is `null`
  with a reason.

Inputs are not stored by this service.
"""

TAGS = [
    {"name": "classification", "description": "Robson Ten-Group classification only."},
    {"name": "admissions", "description": "Full admission: screening, Robson, readiness."},
    {"name": "service", "description": "Service and model status."},
]


class ApiError(Exception):
    """An error returned as an :class:`ErrorDetail` body."""

    def __init__(self, status: int, error: str, message: str, fields: list[str] | None = None):
        super().__init__(message)
        self.status = status
        self.detail = ErrorDetail(error=error, message=message, fields=fields or [])


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the rule set and the readiness model once at startup."""
    app.state.rule_set = load_rule_set()
    app.state.model, app.state.model_reason = load_readiness_model(model_dir())
    yield


app = FastAPI(
    title="Robson Readiness API",
    version="0.1.0",
    description=DESCRIPTION,
    openapi_tags=TAGS,
    lifespan=lifespan,
)


@app.exception_handler(ApiError)
async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(status_code=exc.status, content=exc.detail.model_dump())


def _rule_set(request: Request) -> RuleSet:
    rule_set: RuleSet = request.app.state.rule_set
    return rule_set


def _model(request: Request) -> tuple[ReadinessModel | None, str | None]:
    return request.app.state.model, request.app.state.model_reason


def _jsonable(value: object) -> Any:
    return list(value) if isinstance(value, tuple | list | frozenset) else value


def _run_engine(fields: RobsonFields, rule_set: RuleSet) -> Classification:
    ga_range: tuple[float, float] | None = None
    lower, upper = fields.ga_band_lower, fields.ga_band_upper
    if (lower is None) != (upper is None):
        raise ApiError(
            422,
            "invalid_robson_inputs",
            "Give both gestational-age band bounds or neither.",
            ["ga_band_lower", "ga_band_upper"],
        )
    if lower is not None and upper is not None:
        ga_range = (lower, upper)
    try:
        inputs = RobsonInputs(
            parity=fields.parity,
            previous_cs_count=fields.previous_cs_count,
            plurality=fields.plurality,
            fetal_presentation=fields.fetal_presentation,
            gestational_age_weeks=fields.gestational_age_weeks,
            onset_of_labour=fields.onset_of_labour,
            gestational_age_range=ga_range,
        )
    except ValueError as exc:
        raise ApiError(422, "invalid_robson_inputs", str(exc)) from exc
    return _to_classification(classify(inputs, rule_set), inputs, rule_set)


def _to_classification(
    result: ClassificationResult, inputs: RobsonInputs, rule_set: RuleSet
) -> Classification:
    failed = [
        check.name
        for check in rule_set.consistency
        if all(c.evaluate(inputs) == "true" for c in check.when)
    ]
    names = {rule.group: rule.name for rule in rule_set.groups}
    return Classification(
        status=result.status,
        group=result.group,
        subgroup=result.subgroup,
        group_name=names.get(result.group) if result.group is not None else None,
        candidates=sorted(result.candidates),
        resolving_fields=list(result.resolving_fields),
        conflict_fields=list(result.conflict_fields),
        consistency_checks_failed=failed,
        trace=[
            TraceItem(
                group=t.group,
                subgroup=t.subgroup,
                field=t.field,
                op=t.op,
                expected=_jsonable(t.expected),
                outcome=t.outcome,
            )
            for t in result.trace
        ],
        rule_set_version=result.rule_set_version,
    )


def _check_screening(screening: Screening | None) -> Screening:
    """Every screening field needs a value or a reason for not measuring it, not both."""
    screening = screening or Screening()
    blank: list[str] = []
    both: list[str] = []
    bp = screening.blood_pressure
    checks = {
        "blood_pressure": (
            bp is not None and bp.systolic is not None and bp.diastolic is not None,
            bp.not_measured_reason if bp else None,
        ),
        "proteinuria": (
            screening.proteinuria is not None and screening.proteinuria.value is not None,
            screening.proteinuria.not_measured_reason if screening.proteinuria else None,
        ),
        "glucose": (
            screening.glucose is not None and screening.glucose.value_mmol_l is not None,
            screening.glucose.not_measured_reason if screening.glucose else None,
        ),
    }
    for name, (has_value, reason) in checks.items():
        if not has_value and reason is None:
            blank.append(name)
        elif has_value and reason is not None:
            both.append(name)
    if blank:
        labels = ", ".join(name.replace("_", " ") for name in blank)
        raise ApiError(
            422,
            "screening_incomplete",
            f"Screening is incomplete: {labels} is blank. Enter a value, or give "
            "not_measured_reason for each field that was not measured.",
            blank,
        )
    if both:
        raise ApiError(
            422,
            "screening_ambiguous",
            "Give either a value or not_measured_reason for each screening field, not both.",
            both,
        )
    return screening


CLASSIFY_EXAMPLES: dict[str, Any] = {
    "resolved": {
        "summary": "Resolved: group 5a",
        "value": {
            "parity": 2,
            "previous_cs_count": 1,
            "plurality": 1,
            "fetal_presentation": "cephalic",
            "gestational_age_weeks": 39,
            "onset_of_labour": "spontaneous",
        },
    },
    "partial": {
        "summary": "Partial: gestational age missing (group 5 or 10)",
        "value": {
            "parity": 2,
            "previous_cs_count": 1,
            "plurality": 1,
            "fetal_presentation": "cephalic",
        },
    },
    "conflict": {
        "summary": "Conflict: parity 0 with a previous cesarean",
        "value": {
            "parity": 0,
            "previous_cs_count": 1,
            "plurality": 1,
            "fetal_presentation": "cephalic",
            "gestational_age_weeks": 39,
            "onset_of_labour": "spontaneous",
        },
    },
}

_ADMISSION: dict[str, Any] = {
    "facility_id": "Muhima",
    "maternal_age": 29,
    "gestational_age_weeks": 39,
    "parity": 2,
    "previous_cs_count": 1,
    "plurality": 1,
    "fetal_presentation": "cephalic",
    "onset_of_labour": "spontaneous",
    "screening": {
        "blood_pressure": {"systolic": 128, "diastolic": 84},
        "proteinuria": {"not_measured_reason": "equipment_unavailable"},
        "glucose": {"value_mmol_l": 5.2},
    },
    "height_cm": 160,
    "weight_kg": 68,
    "anc_contacts": 4,
    "gravidity": 3,
    "living_children": 2,
    "insurance_type": "Mutuel",
    "residency": "Urban",
}
ASSESS_EXAMPLES: dict[str, Any] = {
    "complete": {"summary": "Complete admission", "value": _ADMISSION},
    "screening_blank": {
        "summary": "Rejected: glucose blank with no reason",
        "value": {**_ADMISSION, "screening": {**_ADMISSION["screening"], "glucose": {}}},
    },
    "minimal": {
        "summary": "Robson fields and screening only",
        "value": {
            key: _ADMISSION[key]
            for key in (
                "gestational_age_weeks",
                "parity",
                "previous_cs_count",
                "plurality",
                "fetal_presentation",
                "onset_of_labour",
                "screening",
            )
        },
    },
}


@app.get("/health", tags=["service"])
def health(request: Request) -> Health:
    """Service status, whether a readiness model is loaded, and the active versions."""
    loaded, reason = _model(request)
    return Health(
        status="ok",
        model_loaded=loaded is not None,
        model_version=loaded.version if loaded else None,
        model_unavailable_reason=reason,
        rule_set_version=_rule_set(request).version_label,
    )


@app.get(
    "/model",
    tags=["service"],
    responses={503: {"model": ErrorDetail, "description": "No model is loaded."}},
)
def model_card(request: Request) -> ModelCard:
    """Aggregate model card of the active readiness model."""
    loaded, reason = _model(request)
    if loaded is None:
        raise ApiError(503, "model_unavailable", reason or "No readiness model is active.")
    s = loaded.summary
    calibration = s.get("calibration") or {}
    return ModelCard(
        version_label=loaded.version,
        algorithm=str(s.get("algorithm", "unknown")),
        family=s.get("family"),
        feature_set=str(s.get("feature_set", "unknown")),
        use_facility=s.get("use_facility"),
        n_features=len(loaded.features),
        features=loaded.features,
        categorical_levels=loaded.levels,
        hyperparameters=s.get("hyperparameters") or {},
        calibration_method=str(calibration.get("method", "unknown")),
        calibration_cv_brier=calibration.get("cv_brier"),
        population=s.get("population"),
        training_window_start=s.get("training_window_start"),
        training_window_end=s.get("training_window_end"),
        n_fit_rows=s.get("n_fit_rows"),
        n_calibration_rows=s.get("n_calibration_rows"),
        label=READINESS_LABEL,
    )


@app.post(
    "/classify",
    tags=["classification"],
    responses={422: {"description": "An input is outside its allowed range."}},
)
def classify_admission(
    request: Request,
    fields: Annotated[RobsonFields, Body(openapi_examples=CLASSIFY_EXAMPLES)],
) -> Classification:
    """Classify one admission into a Robson group from the six Robson inputs.

    Leave out what was not recorded: the result is then `partial`, with the candidate groups
    and the `resolving_fields` that would settle it. Contradictory inputs give `conflict`
    with the `conflict_fields`.
    """
    return _run_engine(fields, _rule_set(request))


@app.post(
    "/admissions/assess",
    tags=["admissions"],
    responses={
        422: {
            "description": "Screening incomplete (blood pressure, proteinuria or glucose blank "
            "without a reason), or an input outside its allowed range.",
        }
    },
)
def assess_admission(
    request: Request,
    admission: Annotated[AdmissionRequest, Body(openapi_examples=ASSESS_EXAMPLES)],
) -> AssessmentResponse:
    """Check screening, classify, and add the operational readiness signal.

    Blood pressure, proteinuria and glucose each need a value or a `not_measured_reason`;
    otherwise the admission is rejected with 422. The readiness signal is the calibrated
    probability of cesarean under current practice. It is not a recommendation. It is `null`
    (with `readiness_unavailable_reason`) when no model is loaded or the Robson inputs
    conflict.
    """
    screening = _check_screening(admission.screening)
    rule_set = _rule_set(request)
    classification = _run_engine(admission, rule_set)
    loaded, reason = _model(request)
    readiness: Readiness | None = None
    if loaded is not None and classification.status == "conflict":
        reason = "Resolve the conflicting Robson inputs before a readiness signal is given."
    elif loaded is not None:
        try:
            frame, not_given, unrecognised = feature_row(admission, loaded, rule_set)
            probability = float(loaded.model.predict_proba(frame)[0, 1])
            if not math.isfinite(probability):
                raise ValueError("non-finite probability")
        except Exception as exc:  # the classification still stands without the model
            reason = f"Readiness signal unavailable: the model failed ({type(exc).__name__})."
        else:
            readiness = Readiness(
                probability=round(probability, 4),
                of_100_women_like_her=round(probability * 100),
                label=READINESS_LABEL,
                model_version=loaded.version,
                inputs_not_given=not_given,
                unrecognised_inputs=unrecognised,
            )
            reason = None
    return AssessmentResponse(
        classification=classification,
        readiness=readiness,
        readiness_unavailable_reason=reason,
        screening=screening,
    )
