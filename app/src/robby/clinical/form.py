"""Reading the admission form (FR2). Nothing is coerced: a value is whole text that fits its
field, or it is an error naming the field. A blank is "not entered", never zero."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from robby.clinical import labels
from robby.models import Admission, OmissionField, OmissionReason, ScreeningOmission

INT = re.compile(r"\d{1,3}")
DECIMAL_1 = re.compile(r"\d{1,3}(\.\d)?")
NUMBER = re.compile(r"\d{1,4}(\.\d{1,2})?")
TIME_FORMATS = ("%d %b %Y, %H:%M", "%d %b %Y %H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M")
TIME_SHOWN = "%d %b %Y, %H:%M"
MAX_PAST = timedelta(days=30)
FUTURE_SLACK = timedelta(minutes=5)

SCREENING_OMISSION = {
    "bp": OmissionField.BLOOD_PRESSURE,
    "protein": OmissionField.PROTEINURIA,
    "glucose": OmissionField.GLUCOSE,
}


@dataclass(frozen=True)
class DetailSpec:
    """One optional model input, from the active model's feature schema."""

    name: str
    label: str
    kind: str  # integer | number | category
    unit: str
    low: float | None
    high: float | None
    levels: tuple[str, ...] | None

    @property
    def is_yes_no(self) -> bool:
        return self.levels is not None and {x.lower() for x in self.levels} == {"yes", "no"}

    @property
    def options(self) -> list[tuple[str, str]]:
        shown = labels.LEVEL_TEXT.get(self.name, {})
        return [(x, shown.get(x, x)) for x in (self.levels or ())]


def detail_specs(schema: Any | None) -> list[DetailSpec]:
    """Optional inputs of a model schema, in the mockup's order (unknown ones after)."""
    if schema is None:
        return []
    by_name = {f.name: f for f in schema.inputs if f.name not in labels.NOT_DETAILS}
    order = [n for n, _, _ in labels.DETAIL_ORDER if n in by_name]
    order += sorted(n for n in by_name if n not in order)
    specs = []
    for name in order:
        f = by_name[name]
        low, high = f.range if f.range is not None else (None, None)
        specs.append(
            DetailSpec(
                name=name,
                label=labels.DETAIL_LABELS.get(name, name.replace("_", " ").capitalize()),
                kind=f.kind,
                unit=labels.DETAIL_UNITS.get(name, ""),
                low=low,
                high=high,
                levels=f.levels,
            )
        )
    return specs


@dataclass(frozen=True)
class Parsed:
    admitted_at: datetime | None = None
    maternal_age: int | None = None
    ga_weeks: int | None = None
    parity: int | None = None
    previous_cs_count: int | None = None
    plurality: int | None = None
    presentation: str | None = None
    onset: str | None = None
    systolic: int | None = None
    diastolic: int | None = None
    bp_reason: str | None = None
    proteinuria: str | None = None
    protein_reason: str | None = None
    glucose: Decimal | None = None
    glucose_reason: str | None = None
    details: dict[str, int | float | str] = field(default_factory=dict)
    correction_reason: str | None = None


def _value(values: Mapping[str, str], name: str) -> str:
    return (values.get(name) or "").strip()


def _whole(
    values: Mapping[str, str], name: str, label: str, low: int, high: int, errors: dict[str, str]
) -> int | None:
    text = _value(values, name)
    if not text:
        return None
    if not INT.fullmatch(text) or not low <= int(text) <= high:
        errors[name] = f"{label} must be a whole number from {low} to {high}."
        return None
    return int(text)


def parse_time(text: str, zone: ZoneInfo) -> datetime | None:
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=zone)
        except ValueError:
            continue
    return None


def _choice(
    values: Mapping[str, str],
    name: str,
    label: str,
    allowed: tuple[tuple[str, str], ...],
    errors: dict[str, str],
) -> str | None:
    text = _value(values, name)
    if not text:
        return None
    if text not in {v for v, _ in allowed}:
        errors[name] = f"{label}: choose one of the listed options."
        return None
    return text


def parse(
    values: Mapping[str, str],
    specs: list[DetailSpec],
    zone: ZoneInfo,
    now: datetime,
    *,
    screening_required: bool = True,
    reason_required: bool = False,
) -> tuple[Parsed, dict[str, str]]:
    """Returns the parsed values (invalid ones as None) and an error per field."""
    errors: dict[str, str] = {}
    admitted_at = None
    text = _value(values, "admitted_at")
    if not text:
        errors["admitted_at"] = "Enter when the woman was admitted."
    else:
        local = parse_time(text, zone)
        if local is None:
            errors["admitted_at"] = "Admitted at must look like 05 Oct 2026, 14:22."
        elif local > now + FUTURE_SLACK:
            errors["admitted_at"] = "Admitted at cannot be in the future."
        elif local < now - MAX_PAST:
            errors["admitted_at"] = "Admitted at cannot be more than 30 days ago."
        else:
            admitted_at = local

    age = _whole(values, "maternal_age", "Maternal age", 10, 60, errors)
    ga = _whole(values, "gestational_age_weeks", "Gestational age", 20, 45, errors)
    parity = _whole(values, "parity", "Parity", 0, 30, errors)
    cs = _whole(values, "previous_cs_count", "Previous cesareans", 0, 20, errors)
    presentation = _choice(
        values, "fetal_presentation", "Fetal presentation", labels.PRESENTATION_OPTIONS, errors
    )
    plurality_text = _choice(
        values, "plurality", "Number of fetuses", labels.PLURALITY_OPTIONS, errors
    )
    onset = _choice(values, "onset_of_labour", "Onset of labour", labels.ONSET_OPTIONS, errors)

    # Screening (FR2): each of the three needs a value, or "not measured" with a reason.
    def reason_of(key: str, title: str) -> str | None:
        r = _choice(values, f"{key}_reason", title, labels.REASONS[key], errors)
        if r is None and f"{key}_reason" not in errors:
            errors[f"{key}_reason"] = f"Choose why {title.lower()} was not measured."
        return r

    systolic = diastolic = None
    bp_reason = protein = protein_reason = glucose = glucose_reason = None
    if values.get("bp_nm"):
        bp_reason = reason_of("bp", "Blood pressure")
    else:
        s_text, d_text = _value(values, "bp_sys"), _value(values, "bp_dia")
        if not s_text and not d_text:
            if screening_required:
                errors["bp"] = (
                    "Blood pressure is blank. Enter a value, or tick “Not measured” and choose "
                    "a reason."
                )
        elif not (INT.fullmatch(s_text) and INT.fullmatch(d_text)):
            errors["bp"] = (
                "Blood pressure needs both numbers, as whole numbers (for example 128 / 84)."
            )
        else:
            s, d = int(s_text), int(d_text)
            if not (40 <= s <= 300 and 20 <= d <= 200):
                errors["bp"] = (
                    "Blood pressure is outside the possible range (40 to 300 over 20 to 200)."
                )
            elif s <= d:
                errors["bp"] = "Systolic pressure must be higher than diastolic."
            else:
                systolic, diastolic = s, d

    if values.get("protein_nm"):
        protein_reason = reason_of("protein", "Proteinuria")
    else:
        protein = _choice(values, "proteinuria", "Proteinuria", labels.PROTEINURIA_OPTIONS, errors)
        if protein is None and "proteinuria" not in errors and screening_required:
            errors["proteinuria"] = (
                "Proteinuria is blank. Choose a result, or tick “Not measured” and choose a reason."
            )

    if values.get("glucose_nm"):
        glucose_reason = reason_of("glucose", "Glucose")
    else:
        g_text = _value(values, "glucose")
        if not g_text:
            if screening_required:
                errors["glucose"] = (
                    "Glucose is blank. Enter a value, or tick “Not measured” and choose a reason."
                )
        elif not DECIMAL_1.fullmatch(g_text) or not Decimal("0.5") <= Decimal(g_text) <= 60:
            errors["glucose"] = (
                "Glucose must be between 0.5 and 60 mmol/L, with at most one decimal."
            )
        else:
            glucose = Decimal(g_text)

    details: dict[str, int | float | str] = {}
    for spec in specs:
        key = f"d_{spec.name}"
        raw = _value(values, key)
        if not raw:
            continue
        if spec.kind == "category":
            if raw not in (spec.levels or ()):
                errors[key] = f"{spec.label}: choose one of the listed options."
            else:
                details[spec.name] = raw
            continue
        pattern = INT if spec.kind == "integer" else NUMBER
        if not pattern.fullmatch(raw):
            errors[key] = f"{spec.label} must be a number."
            continue
        number = float(raw)
        if (spec.low is not None and number < spec.low) or (
            spec.high is not None and number > spec.high
        ):
            low = "" if spec.low is None else f"{spec.low:g}"
            high = "" if spec.high is None else f"{spec.high:g}"
            errors[key] = f"{spec.label} must be between {low or 'any'} and {high or 'any'}."
            continue
        details[spec.name] = int(number) if spec.kind == "integer" else number

    correction = _value(values, "correction_reason") or None
    if reason_required:
        if correction is None:
            errors["correction_reason"] = "Say why the admission is being corrected."
        elif len(correction) > 500:
            errors["correction_reason"] = "Keep the reason to 500 characters or fewer."

    parsed = Parsed(
        admitted_at=admitted_at,
        maternal_age=age,
        ga_weeks=ga,
        parity=parity,
        previous_cs_count=cs,
        plurality=int(plurality_text) if plurality_text else None,
        presentation=presentation,
        onset=onset,
        systolic=systolic,
        diastolic=diastolic,
        bp_reason=bp_reason,
        proteinuria=protein,
        protein_reason=protein_reason,
        glucose=glucose,
        glucose_reason=glucose_reason,
        details=details,
        correction_reason=correction,
    )
    return parsed, errors


def omissions_of(parsed: Parsed) -> list[tuple[OmissionField, OmissionReason]]:
    pairs = [
        ("bp", parsed.bp_reason),
        ("protein", parsed.protein_reason),
        ("glucose", parsed.glucose_reason),
    ]
    return [(SCREENING_OMISSION[k], OmissionReason(r)) for k, r in pairs if r]


def values_from_admission(
    admission: Admission, omissions: list[ScreeningOmission], zone: ZoneInfo
) -> dict[str, str]:
    """The form's field values for an existing admission (correction pre-fill)."""

    def text(v: Any) -> str:
        return "" if v is None else str(v)

    values = {
        "admitted_at": admission.admitted_at.astimezone(zone).strftime(TIME_SHOWN),
        "maternal_age": text(admission.maternal_age),
        "gestational_age_weeks": text(int(admission.gestational_age_weeks))
        if admission.gestational_age_weeks is not None
        else "",
        "parity": text(admission.parity),
        "previous_cs_count": text(admission.previous_cs_count),
        "plurality": text(admission.plurality),
        "fetal_presentation": text(admission.fetal_presentation),
        "onset_of_labour": text(admission.onset_of_labour),
        "bp_sys": text(admission.systolic_bp),
        "bp_dia": text(admission.diastolic_bp),
        "proteinuria": text(admission.proteinuria),
        "glucose": text(admission.glucose_mmol),
    }
    for o in omissions:
        key = {"blood_pressure": "bp", "proteinuria": "protein", "glucose": "glucose"}[
            o.field.value
        ]
        values[f"{key}_nm"] = "1"
        values[f"{key}_reason"] = o.reason.value
    for name, value in (admission.details or {}).items():
        values[f"d_{name}"] = str(value)
    return values
