"""What the clinician's result page shows for one admission: the Robson panel, the readiness
signal and the recorded values in words. Built here from the service's queries, so the route only
renders it and the template gets typed fields instead of a loose dict."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from robby.clinical import classify as cls
from robby.clinical import labels, service
from robby.models import Admission, DeliveryOutcome, LabResult, ScreeningOmission
from robby.monitoring.readiness import is_high

SAVED_AT_SHOWN = "%H:%M"
ACTIVE_SINCE_SHOWN = "%d %b %Y"


@dataclass(frozen=True)
class ScreeningLine:
    text: str  # the value with its unit, or "" when not measured
    reason: str | None  # why it was not measured, in words


@dataclass(frozen=True)
class ResultView:
    panel: cls.PanelView
    saved_at: str
    replaces: Admission | None
    replaced_by: Admission | None
    outcome: DeliveryOutcome | None
    outcome_text: str | None
    screening: dict[str, ScreeningLine]  # bp | protein | glucose
    details: list[tuple[str, str]]
    # The readiness signal: set only when the active model scored this admission.
    percent: int | None = None
    cut_percent: int | None = None
    high: bool = False
    model_label: str | None = None
    active_since: str | None = None
    not_given: list[str] = field(default_factory=list)  # optional inputs left blank
    model_active: bool = False  # no signal, but a model is active: scoring failed
    rescored_at: datetime | None = None  # the signal was updated after this HIV result


def details_view(admission: Admission) -> list[tuple[str, str]]:
    """The Robson inputs and age as entered, in words; empty text for a value not recorded."""

    def word(options: tuple[tuple[str, str], ...], value: object) -> str:
        return "" if value is None else dict(options).get(str(value), str(value))

    def number(value: object, unit: str = "") -> str:
        return "" if value is None else f"{float(str(value)):g}{unit}"

    # the option "2" reads "two or more"
    babies = None if admission.plurality is None else min(admission.plurality, 2)
    return [
        ("Age", number(admission.maternal_age, " years")),
        ("Previous births (parity)", number(admission.parity)),
        ("Previous cesareans", number(admission.previous_cs_count)),
        ("Babies", word(labels.PLURALITY_OPTIONS, babies)),
        ("Presentation", word(labels.PRESENTATION_OPTIONS, admission.fetal_presentation)),
        ("Gestational age", number(admission.gestational_age_weeks, " weeks")),
        ("Onset of labour", word(labels.ONSET_OPTIONS, admission.onset_of_labour)),
    ]


def screening_view(
    admission: Admission, omissions: list[ScreeningOmission]
) -> dict[str, ScreeningLine]:
    reasons = {o.field.value: labels.REASON_TEXT[o.reason.value] for o in omissions}
    bp = (
        f"{admission.systolic_bp} / {admission.diastolic_bp} mmHg"
        if admission.systolic_bp is not None
        else ""
    )
    glucose = f"{admission.glucose_mmol} mmol/L" if admission.glucose_mmol is not None else ""
    return {
        "bp": ScreeningLine(bp, reasons.get("blood_pressure")),
        "protein": ScreeningLine(admission.proteinuria or "", reasons.get("proteinuria")),
        "glucose": ScreeningLine(glucose, reasons.get("glucose")),
    }


def _panel(db: Session, admission: Admission) -> cls.PanelView:
    classification = service.classification_of(db, admission)
    return cls.panel_view(
        status=classification.status.value,
        group=classification.robson_group,
        subgroup=classification.subgroup,
        candidates=list(classification.candidates),
        resolving=list(classification.resolving_fields),
        conflict=list(classification.conflict_fields),
        trace=classification.trace,
    )


def result_view(db: Session, admission: Admission, zone: ZoneInfo) -> ResultView:
    outcome = service.chain_outcome(db, admission)
    view = ResultView(
        panel=_panel(db, admission),
        saved_at=admission.created_at.astimezone(zone).strftime(SAVED_AT_SHOWN),
        replaces=service.replaced_admission(db, admission),
        replaced_by=service.replacement_of(db, admission),
        outcome=outcome,
        outcome_text=dict(labels.DELIVERY_OPTIONS)[outcome.mode.value] if outcome else None,
        screening=screening_view(admission, service.omissions_for(db, admission)),
        details=details_view(admission),
    )
    predicted = service.prediction_of(db, admission)
    if predicted is None:
        return replace(view, model_active=service.has_active_model(db))
    prediction, model = predicted
    cut = service.readiness_cut_of(db, admission)
    prompted = db.get(LabResult, prediction.lab_result_id) if prediction.lab_result_id else None
    vector = prediction.feature_vector
    return replace(
        view,
        percent=service.percent_of(prediction.probability),
        cut_percent=cut,
        high=is_high(float(prediction.probability), cut),
        model_label=model.version_label,
        active_since=service.activated_at(db, model).astimezone(zone).strftime(ACTIVE_SINCE_SHOWN),
        rescored_at=prompted.resulted_at if prompted else None,
        not_given=[
            labels.DETAIL_LABELS[name]
            for name, _, _ in labels.DETAIL_ORDER
            if name in vector and vector[name] is None
        ],
    )
