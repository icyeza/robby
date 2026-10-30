"""Lab requests: the clinician asks for tests on an admission and may cancel an open request
with a reason; the lab technician records each result. Every step is an insert (the tables are
insert-only) and one event. The facility always comes from the signed-in user: a request of
another facility is NotFound.

An HIV positive or negative result re-scores the readiness signal: a new active prediction that
supersedes the current one, built from its stored feature vector with hiv_positive filled in and
scored by the active model. A failure there never costs the result (savepoint), as in
clinical.service.save_admission. No other result changes the classification or the signal.

Event payloads carry codes and test names only: never a result, a note, a reason or a patient
identifier."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import ColumnElement, and_, exists, func, select
from sqlalchemy.orm import Session

from robby import events
from robby.clinical import classify as cls
from robby.clinical import scoring
from robby.clinical import service as clinical
from robby.errors import NotAllowed, NotFound
from robby.lab.catalogue import HIV_ANSWER, HIV_INPUT, test_name
from robby.models import (
    Admission,
    AppUser,
    Facility,
    LabCancellation,
    LabRequest,
    LabResult,
    LabTest,
    Prediction,
    PredictionMode,
)
from robby.timeutil import utcnow

logger = logging.getLogger("robby")


@dataclass(frozen=True)
class LabLine:
    """A request with whatever closed it."""

    request: LabRequest
    result: LabResult | None = None
    cancellation: LabCancellation | None = None

    @property
    def status(self) -> str:
        """requested | resulted | cancelled"""
        if self.result is not None:
            return "resulted"
        return "cancelled" if self.cancellation is not None else "requested"

    @property
    def is_open(self) -> bool:
        return self.status == "requested"


@dataclass(frozen=True)
class Recorded:
    result: LabResult
    rescore: str  # ok | not_hiv | indeterminate | no_prediction | no_model | not_an_input | failed
    prediction: Prediction | None = None


def _is_open() -> ColumnElement[bool]:
    return and_(
        ~exists().where(LabResult.request_id == LabRequest.id),
        ~exists().where(LabCancellation.request_id == LabRequest.id),
    )


def _lock(db: Session, admission: Admission) -> None:
    """Serialise lab changes on one admission (two clicks, two tabs), so the open-request and
    closed-once checks below see each other. SELECT ... FOR UPDATE does not fire the triggers."""
    db.execute(select(Admission.id).where(Admission.id == admission.id).with_for_update())


def _facility_code(db: Session, admission: Admission) -> str:
    facility = db.get(Facility, admission.facility_id)
    assert facility is not None
    return facility.code


def lines_for(db: Session, admission: Admission) -> list[LabLine]:
    """Every request of the admission with its result or cancellation, newest first."""
    rows = db.execute(
        select(LabRequest, LabResult, LabCancellation)
        .outerjoin(LabResult, LabResult.request_id == LabRequest.id)
        .outerjoin(LabCancellation, LabCancellation.request_id == LabRequest.id)
        .where(LabRequest.admission_id == admission.id)
        .order_by(LabRequest.requested_at.desc(), LabRequest.test)
    ).all()
    return [LabLine(r[0], r[1], r[2]) for r in rows]


def open_tests(db: Session, admission: Admission) -> set[LabTest]:
    return {line.request.test for line in lines_for(db, admission) if line.is_open}


def get_line(db: Session, user: AppUser, request_id: uuid.UUID) -> tuple[LabLine, Admission]:
    """A request of the user's own facility with its admission, or NotFound."""
    row = db.execute(
        select(LabRequest, Admission, LabResult, LabCancellation)
        .join(Admission, Admission.id == LabRequest.admission_id)
        .outerjoin(LabResult, LabResult.request_id == LabRequest.id)
        .outerjoin(LabCancellation, LabCancellation.request_id == LabRequest.id)
        .where(LabRequest.id == request_id, Admission.facility_id == user.facility_id)
    ).first()
    if row is None:
        raise NotFound(str(request_id))
    return LabLine(row[0], row[2], row[3]), row[1]


def _still_open(db: Session, user: AppUser, request_id: uuid.UUID) -> tuple[LabLine, Admission]:
    line, admission = get_line(db, user, request_id)
    _lock(db, admission)
    line, admission = get_line(db, user, request_id)  # read again under the lock
    if line.result is not None:
        raise NotAllowed(f"A result is already recorded for this {test_name(line.request.test)}.")
    if line.cancellation is not None:
        raise NotAllowed("This request was cancelled by the clinician.")
    return line, admission


# ---------------------------------------------------------------- clinician


def request_tests(
    db: Session,
    user: AppUser,
    admission: Admission,
    tests: list[LabTest],
    *,
    note: str | None,
    urgent: bool,
    now: datetime | None = None,
) -> list[LabRequest]:
    """One request row per test. Refused for another facility's admission (NotFound), a replaced
    admission, or a test already waiting for the lab (NotAllowed). Does not commit."""
    if admission.facility_id != user.facility_id:
        raise NotFound(admission.code)
    if not tests:
        raise NotAllowed("Choose at least one test.")
    if clinical.replacement_of(db, admission) is not None:
        raise NotAllowed("This admission has been replaced by a corrected one; request it there.")
    _lock(db, admission)
    waiting = open_tests(db, admission) & set(tests)
    if waiting:
        names = ", ".join(test_name(t) for t in sorted(waiting))
        raise NotAllowed(f"Already requested and waiting for the lab: {names}.")
    at = now or utcnow()
    rows = [
        LabRequest(
            admission_id=admission.id,
            test=test,
            urgent=urgent,
            note=note or None,
            requested_by=user.id,
            requested_at=at,
        )
        for test in dict.fromkeys(tests)
    ]
    db.add_all(rows)
    db.flush()
    events.append(
        db,
        user,
        "lab.requested",
        ("admission", admission.id),
        {
            "code": admission.code,
            "facility": _facility_code(db, admission),
            "tests": [r.test.value for r in rows],
            "urgent": urgent,
        },
    )
    return rows


def cancel_request(
    db: Session,
    user: AppUser,
    request_id: uuid.UUID,
    reason: str,
    now: datetime | None = None,
) -> LabCancellation:
    """Close an open request without a result. The reason is required. Does not commit."""
    if not reason.strip():
        raise NotAllowed("Say why the request is cancelled.")
    line, admission = _still_open(db, user, request_id)
    cancellation = LabCancellation(
        request_id=line.request.id,
        reason=reason.strip(),
        cancelled_by=user.id,
        cancelled_at=now or utcnow(),
    )
    db.add(cancellation)
    db.flush()
    events.append(
        db,
        user,
        "lab.cancelled",
        ("lab_request", line.request.id),
        {
            "code": admission.code,
            "facility": _facility_code(db, admission),
            "test": line.request.test.value,
        },
    )
    return cancellation


# ---------------------------------------------------------------- lab technician


def queue(db: Session, user: AppUser) -> list[tuple[LabRequest, Admission]]:
    """Open requests of the user's facility: urgent first, then the oldest."""
    rows = db.execute(
        select(LabRequest, Admission)
        .join(Admission, Admission.id == LabRequest.admission_id)
        .where(Admission.facility_id == user.facility_id, _is_open())
        .order_by(
            LabRequest.urgent.desc(), LabRequest.requested_at, Admission.code, LabRequest.test
        )
    ).all()
    return [(r[0], r[1]) for r in rows]


def open_count(db: Session, user: AppUser) -> int:
    """How many requests of the user's facility wait for a result (the navigation badge)."""
    return int(
        db.scalar(
            select(func.count())
            .select_from(LabRequest)
            .join(Admission, Admission.id == LabRequest.admission_id)
            .where(Admission.facility_id == user.facility_id, _is_open())
        )
        or 0
    )


def record_result(
    db: Session,
    runtime: scoring.ModelRuntime,
    user: AppUser,
    request_id: uuid.UUID,
    value: str,
    now: datetime | None = None,
) -> Recorded:
    """Record the result of an open request (`value` already checked by
    catalogue.parse_result). Does not commit."""
    line, admission = _still_open(db, user, request_id)
    result = LabResult(
        request_id=line.request.id, value=value, recorded_by=user.id, resulted_at=now or utcnow()
    )
    db.add(result)
    db.flush()
    recorded = Recorded(result, "not_hiv")
    if line.request.test is LabTest.HIV_RAPID:
        answer = HIV_ANSWER.get(value)
        recorded = (
            Recorded(result, "indeterminate")
            if answer is None
            else _rescore(db, runtime, admission, result, answer)
        )
    events.append(
        db,
        user,
        "lab.resulted",
        ("lab_request", line.request.id),
        {
            "code": admission.code,
            "facility": _facility_code(db, admission),
            "test": line.request.test.value,
        },
    )
    return recorded


def _rescore(
    db: Session,
    runtime: scoring.ModelRuntime,
    admission: Admission,
    result: LabResult,
    hiv_positive: bool,
) -> Recorded:
    """A new active prediction from the current one's feature vector with hiv_positive set,
    by the active model. NFR7: whatever goes wrong, the result is kept (savepoint)."""
    current = clinical.prediction_of(db, admission)
    if current is None:
        return Recorded(result, "no_prediction")  # never scored: no group, or no model then
    try:
        with db.begin_nested():
            rule_row, engine_rules = cls.active_rule_set(db)
            model = runtime.active(db, rule_row.id, engine_rules)
            if model is None:
                return Recorded(result, "no_model")
            level = scoring.yes_no_level(model, HIV_INPUT, hiv_positive)
            if level is None:
                return Recorded(result, "not_an_input")
            vector = scoring.with_input(model, current[0].feature_vector, HIV_INPUT, level)
            scored = scoring.score_row(model, vector)
            prediction = Prediction(
                admission_id=admission.id,
                model_id=scored.model_id,
                mode=PredictionMode.ACTIVE,
                probability=scored.probability,
                feature_vector=scored.feature_vector,
                supersedes_id=current[0].id,
                lab_result_id=result.id,
            )
            db.add(prediction)
            db.flush()
    except Exception as exc:
        logger.warning("Re-scoring failed for %s: %s", admission.code, type(exc).__name__)
        return Recorded(result, "failed")
    return Recorded(result, "ok", prediction)
