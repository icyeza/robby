"""From adapter output to checked, saved admissions (FR8).

The adapter's rows are laid out as the admission template (clinical/xlsx_template.py) and go
through importer.read_table, so a FHIR admission passes the very same checks as the form and the
Excel import. A problem the importer reports by template row and column is turned back into the
FHIR element that supplied the value. Saving follows the import API: all or nothing, a record
already recorded (same patient number and admission time) is skipped, not an error."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from robby import events
from robby.clinical import importer, scoring, service
from robby.clinical.form import DetailSpec, Parsed
from robby.clinical.xlsx_template import columns
from robby.fhir.adapter import BundleRead, Issue, Mapped
from robby.models import Admission, AdmissionSource, AppUser, Facility, patient_number_index

FIRST_ROW = 2  # the importer numbers rows as a sheet: the header is row 1


@dataclass(frozen=True)
class Checked:
    rows: list[tuple[int, Parsed]]  # (template row, parsed), empty when there are issues
    admissions: list[Mapped]  # admissions[row - FIRST_ROW] is the source of a row
    issues: list[Issue]


def template_table(admissions: list[Mapped], specs: list[DetailSpec]) -> list[tuple[object, ...]]:
    """The admissions as a template sheet: header row, then one row per Encounter. Optional
    inputs the active model does not use are left out (the template has no column for them)."""
    cols = columns(specs)
    return [
        tuple(c.label for c in cols),
        *[tuple(a.values.get(c.field) for c in cols) for a in admissions],
    ]


def _issue_of(
    problem: importer.RowProblem, admissions: list[Mapped], fields: dict[str, str]
) -> Issue | None:
    """The importer's problem as an Issue on the FHIR element, or None when the adapter has
    already reported a problem on that field."""
    if problem.row < FIRST_ROW or problem.row - FIRST_ROW >= len(admissions):
        return Issue("Bundle", problem.message)
    admission = admissions[problem.row - FIRST_ROW]
    name = fields.get(problem.column, "")
    if name in admission.flagged or (name == "bp_sys" and "bp_dia" in admission.flagged):
        return None
    return Issue(admission.source_of(name), problem.message)


def check(read: BundleRead, specs: list[DetailSpec], zone: ZoneInfo, now: datetime) -> Checked:
    """Every admission of the Bundle through the template checks; all or nothing."""
    if not read.admissions:
        return Checked([], [], read.issues)
    result = importer.read_table(template_table(read.admissions, specs), specs, zone, now)
    fields = {c.label: c.field for c in columns(specs)}
    issues = list(read.issues)
    for problem in result.problems:
        issue = _issue_of(problem, read.admissions, fields)
        if issue is not None and issue not in issues:
            issues.append(issue)
    return Checked([] if issues else result.rows, read.admissions, issues)


# ---------------------------------------------------------------- saving


@dataclass(frozen=True)
class Outcome:
    """What happened to one Encounter."""

    encounter: str
    admission: Admission
    group: int | None
    percent: int | None
    state: str  # created | already_recorded | repeated (same as an earlier Encounter)
    repeats: str | None = None  # the earlier Encounter, for "repeated"


def _percent(db: Session, admission: Admission) -> int | None:
    predicted = service.prediction_of(db, admission)
    return service.percent_of(predicted[0].probability) if predicted else None


def save(
    db: Session,
    runtime: scoring.ModelRuntime,
    user: AppUser,
    checked: Checked,
    body_sha256: str,
) -> list[Outcome]:
    """Save every new admission of a checked Bundle with source FHIR, each exactly like a form
    entry, plus one event for the Bundle. Raises classify.NoRuleSetError when no rule set is
    active. Does not commit: the caller commits once, so the Bundle is saved whole or not at all.
    Returns one outcome per Encounter, in Bundle order."""
    facility_id = user.facility_id
    assert facility_id is not None
    scan = importer.find_duplicates(db, facility_id, checked.rows)
    # A patient is the same patient by blind index, as find_duplicates compares her; without a
    # number nothing says two Encounters are the same woman, so neither is a repeat.
    keys = {
        n: (patient_number_index(facility_id, p.patient_number), p.admitted_at)
        for n, p in checked.rows
        if p.patient_number
    }
    repeated = {p.row for p in scan.in_file if p.row in keys}
    first_row: dict[tuple[object, ...], int] = {}
    by_row: dict[int, Outcome] = {}
    for n, parsed in checked.rows:
        encounter = checked.admissions[n - FIRST_ROW].encounter
        if n in scan.recorded:
            a = scan.recorded[n]
            group = service.classification_of(db, a).robson_group
            by_row[n] = Outcome(encounter, a, group, _percent(db, a), "already_recorded")
        elif n not in repeated:
            saved = service.save_admission(db, runtime, user, parsed, source=AdmissionSource.FHIR)
            percent = service.percent_of(saved.prediction.probability) if saved.prediction else None
            group = saved.classification.robson_group
            by_row[n] = Outcome(encounter, saved.admission, group, percent, "created")
        if n in keys:
            first_row.setdefault(keys[n], n)
    for n, _ in checked.rows:
        if n in repeated:
            earlier = by_row[first_row[keys[n]]]
            by_row[n] = Outcome(
                checked.admissions[n - FIRST_ROW].encounter,
                earlier.admission,
                earlier.group,
                earlier.percent,
                "repeated",
                repeats=earlier.encounter,
            )
    outcomes = [by_row[n] for n, _ in checked.rows]
    created = sum(1 for o in outcomes if o.state == "created")
    facility = db.get(Facility, user.facility_id)
    assert facility is not None
    if created:
        events.append(
            db,
            user,
            "admissions.imported",
            ("facility", facility.id),
            {"facility": facility.code, "rows": created, "file_sha256": body_sha256,
             "channel": "fhir"},
        )  # fmt: skip
    else:
        events.append(
            db,
            user,
            "api_import.nothing_new",
            ("facility", facility.id),
            {"facility": facility.code, "skipped": len(outcomes), "channel": "fhir"},
        )
    return outcomes


def record_rejection(db: Session, user: AppUser, issues: int) -> None:
    """The event for a Bundle refused because of its issues (nothing saved)."""
    facility_id: uuid.UUID | None = user.facility_id
    assert facility_id is not None
    facility = db.get(Facility, facility_id)
    events.append(
        db,
        user,
        "api_import.rejected",
        ("facility", facility_id),
        {"facility": facility.code if facility else "", "problems": issues, "channel": "fhir"},
    )
