"""Page-level assembly shared by the facility and national routes (aggregates only)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from robby.audit import formatting
from robby.audit.counts import (
    UnclassifiedBirth,
    awaiting_outcome,
    birth_counts,
)
from robby.audit.measures import GROUP_DEFINITIONS, build_report
from robby.audit.narrative import facility_stands_out
from robby.audit.references import Reference, available_references, choose_reference
from robby.models import Facility

FIELD_LABELS = {
    "parity": "parity",
    "previous_cs_count": "previous cesareans",
    "plurality": "plurality",
    "fetal_presentation": "fetal presentation",
    "gestational_age_weeks": "gestational age",
    "onset_of_labour": "onset of labour",
}
MISSING_LABELS = {
    "parity": "Parity missing",
    "previous_cs_count": "Previous cesareans missing",
    "plurality": "Plurality missing",
    "fetal_presentation": "Presentation missing",
    "gestational_age_weeks": "Gestational age missing",
    "onset_of_labour": "Onset of labour missing",
}
CONFLICT = "Inputs conflict"
NOT_CLASSIFIED = "Not classified"


@dataclass(frozen=True)
class Period:
    start: date
    end: date
    error: str | None = None


def today_local(zone: ZoneInfo) -> date:
    return datetime.now(zone).date()


def parse_period(raw_from: str | None, raw_to: str | None, zone: ZoneInfo) -> Period:
    """Dates are ISO (YYYY-MM-DD). Defaults: 1 January of this year to today. Bad input falls back
    to the defaults and says so."""
    today = today_local(zone)
    start, end, error = today.replace(month=1, day=1), today, None
    try:
        if raw_from:
            start = date.fromisoformat(raw_from)
        if raw_to:
            end = date.fromisoformat(raw_to)
    except ValueError:
        error = "Dates must be written as YYYY-MM-DD; the default period is shown."
        start, end = today.replace(month=1, day=1), today
    if end < start:
        error = "'To' is before 'From'; the default period is shown."
        start, end = today.replace(month=1, day=1), today
    return Period(start, end, error)


def report_context(
    db: Session,
    zone: ZoneInfo,
    facility: Facility,
    period: Period,
    ref_id: str | None,
) -> dict[str, Any]:
    refs = available_references(db)
    reference: Reference | None = choose_reference(refs, ref_id)
    counts = birth_counts(db, period.start, period.end, zone, facility.id).get(facility.id, {})
    waiting = awaiting_outcome(db, period.start, period.end, zone, facility.id).get(facility.id, 0)
    report = build_report(counts, reference, waiting)
    return {
        "facility": facility,
        "period": period,
        "refs": refs,
        "ref_options": [(r.id, r.label) for r in refs],
        "reference": reference,
        "report": report,
        "stands_out": facility_stands_out(report),
        "definitions": GROUP_DEFINITIONS,
        "fmt": formatting,
    }


def reason_labels(row: UnclassifiedBirth) -> list[str]:
    if row.status is None:
        return [NOT_CLASSIFIED]
    if row.status == "conflict":
        return [CONFLICT]
    return [
        MISSING_LABELS.get(f, f"{FIELD_LABELS.get(f, f).capitalize()} missing")
        for f in row.resolving_fields
    ] or ["Missing or imprecise input"]


def month_range(month: date) -> tuple[date, date]:
    nxt = (month.replace(day=28) + timedelta(days=4)).replace(day=1)
    return month, nxt - timedelta(days=1)
