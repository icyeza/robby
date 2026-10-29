"""Page-level assembly shared by the facility and national routes (aggregates only; the
facility-only `unclassified_reasons` reads unclassified births but returns counts), and the
record kept of each monthly export."""

from __future__ import annotations

import hashlib
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from robby import events
from robby.audit import export_format, formatting, plain
from robby.audit.counts import (
    UnclassifiedBirth,
    awaiting_outcome,
    birth_counts,
    unclassified_births,
)
from robby.audit.measures import GROUP_DEFINITIONS, build_report
from robby.audit.narrative import facility_stands_out
from robby.audit.references import Reference, available_references, choose_reference
from robby.models import AppUser, ExportRecord, Facility

EXPORT_MONTHS = 12  # the last complete months a facility can see and export
RECENT_EXPORTS = 5  # exports listed on the report page

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
        "presets": plain.period_presets(today_local(zone), period.start, period.end),
        "plain_groups": plain.PLAIN_GROUPS,
        "verdicts": {r.group: plain.group_verdict(r) for r in report.rows},
        "sources": plain.cesarean_sources(report),
        "in_100": plain.in_100,
        "refs": refs,
        "ref_options": [(r.id, r.label) for r in refs],
        "reference": reference,
        "report": report,
        "stands_out": facility_stands_out(report),
        "definitions": GROUP_DEFINITIONS,
        "fmt": formatting,
    }


def unclassified_reasons(
    db: Session, zone: ZoneInfo, facility_id: uuid.UUID, period: Period, top: int = 3
) -> list[tuple[str, int]]:
    """The commonest reasons births of one facility have no Robson group, with counts."""
    tally: Counter[str] = Counter()
    for row in unclassified_births(db, period.start, period.end, zone, facility_id):
        tally.update(reason_labels(row))
    return tally.most_common(top)


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


def recent_exports(db: Session, facility_id: uuid.UUID) -> Sequence[ExportRecord]:
    """The facility's latest exports, newest first."""
    return db.scalars(
        select(ExportRecord)
        .where(ExportRecord.facility_id == facility_id)
        .order_by(ExportRecord.exported_at.desc())
        .limit(RECENT_EXPORTS)
    ).all()


def record_export(
    db: Session, actor: AppUser, facility: Facility, month: date, body: bytes
) -> None:
    """Keep the hash of an exported monthly file, and log it. Does not commit."""
    digest = hashlib.sha256(body).hexdigest()
    record = ExportRecord(
        facility_id=facility.id,
        period_month=month,
        exported_by=actor.id,
        file_sha256=digest,
    )
    db.add(record)
    db.flush()
    events.append(
        db,
        actor,
        "export.created",
        entity=("export_record", record.id),
        payload={
            "facility_code": facility.code,
            "period_month": month.strftime("%Y-%m"),
            "format_version": export_format.EXPORT_FORMAT_VERSION,
            "sha256": digest,
        },
    )
