"""Admission requests: a clinician asks the lab to enter a patient who has no admission yet, and
may cancel a waiting request with a reason; the lab technician's saved admission fulfils it, in
the same transaction as the admission. Every step is an insert (the tables are insert-only) and
one event. The facility always comes from the signed-in user: a request of another facility is
NotFound.

The patient number and name are encrypted (as on admission); an open request for the same number
is found by its blind index. Nothing matches a request to an admission entered without it: the
request waits until the lab enters it or the clinician cancels it.

Event payloads carry the facility code, the urgency and the admission code only: never a patient
number, a name, a note or a reason."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import ColumnElement, and_, exists, func, or_, select
from sqlalchemy.orm import Session

from robby import events
from robby.errors import NotAllowed, NotFound
from robby.models import (
    Admission,
    AdmissionRequest,
    AdmissionRequestCancellation,
    AdmissionRequestFulfilment,
    AppUser,
    Facility,
    patient_number_index,
)
from robby.timeutil import utcnow

# How long the clinician's list keeps a request after the lab entered it or it was cancelled.
SHOWN_AFTER_CLOSING = timedelta(hours=72)


@dataclass(frozen=True)
class AskLine:
    """A request with whatever closed it (and the admission code once entered)."""

    request: AdmissionRequest
    fulfilment: AdmissionRequestFulfilment | None = None
    cancellation: AdmissionRequestCancellation | None = None
    code: str | None = None

    @property
    def status(self) -> str:
        """waiting | entered | cancelled"""
        if self.fulfilment is not None:
            return "entered"
        return "cancelled" if self.cancellation is not None else "waiting"

    @property
    def is_open(self) -> bool:
        return self.status == "waiting"


def _is_open() -> ColumnElement[bool]:
    return and_(
        ~exists().where(AdmissionRequestFulfilment.request_id == AdmissionRequest.id),
        ~exists().where(AdmissionRequestCancellation.request_id == AdmissionRequest.id),
    )


def _lock(db: Session, facility_id: uuid.UUID) -> None:
    """Serialise admission request changes of one facility (two clicks, two tabs, a save and a
    cancel at once), so the open-request and closed-once checks below see each other. NO KEY
    UPDATE does not block the foreign-key checks of other inserts, and fires no trigger."""
    db.execute(
        select(Facility.id).where(Facility.id == facility_id).with_for_update(key_share=True)
    )


def _facility_code(db: Session, facility_id: uuid.UUID) -> str:
    facility = db.get(Facility, facility_id)
    assert facility is not None
    return facility.code


def _lines(db: Session, *where: ColumnElement[bool]) -> list[AskLine]:
    rows = db.execute(
        select(
            AdmissionRequest,
            AdmissionRequestFulfilment,
            AdmissionRequestCancellation,
            Admission.code,
        )
        .outerjoin(
            AdmissionRequestFulfilment,
            AdmissionRequestFulfilment.request_id == AdmissionRequest.id,
        )
        .outerjoin(
            AdmissionRequestCancellation,
            AdmissionRequestCancellation.request_id == AdmissionRequest.id,
        )
        .outerjoin(Admission, Admission.id == AdmissionRequestFulfilment.admission_id)
        .where(*where)
        .order_by(AdmissionRequest.requested_at.desc(), AdmissionRequest.created_at.desc())
    ).all()
    return [AskLine(r[0], r[1], r[2], r[3]) for r in rows]


def get_line(db: Session, user: AppUser, request_id: uuid.UUID) -> AskLine:
    """A request of the user's own facility, or NotFound."""
    found = _lines(
        db, AdmissionRequest.id == request_id, AdmissionRequest.facility_id == user.facility_id
    )
    if not found:
        raise NotFound(str(request_id))
    return found[0]


def _still_open(db: Session, user: AppUser, request_id: uuid.UUID) -> AskLine:
    line = get_line(db, user, request_id)
    _lock(db, line.request.facility_id)
    line = get_line(db, user, request_id)  # read again under the lock
    if line.fulfilment is not None:
        raise NotAllowed(f"The lab has already entered this patient as {line.code}.")
    if line.cancellation is not None:
        raise NotAllowed("This request was cancelled by the clinician.")
    return line


# ---------------------------------------------------------------- clinician


def request_admission(
    db: Session,
    user: AppUser,
    *,
    patient_number: str,
    patient_name: str | None,
    note: str | None,
    urgent: bool,
    now: datetime | None = None,
) -> AdmissionRequest:
    """Ask the lab to enter a patient (number already checked by the route). Refused while a
    request for the same number waits at the facility (NotAllowed). Does not commit."""
    assert user.facility_id is not None
    if not patient_number.strip():
        raise NotAllowed("Enter the patient number.")
    _lock(db, user.facility_id)
    index = patient_number_index(user.facility_id, patient_number)
    waiting = db.scalar(
        select(func.count())
        .select_from(AdmissionRequest)
        .where(
            AdmissionRequest.facility_id == user.facility_id,
            AdmissionRequest.patient_number_index == index,
            _is_open(),
        )
    )
    if waiting:
        raise NotAllowed("This patient number is already waiting for the lab.")
    row = AdmissionRequest(
        facility_id=user.facility_id,
        patient_number=patient_number,
        patient_name=patient_name or None,
        urgent=urgent,
        note=note or None,
        requested_by=user.id,
        requested_at=now or utcnow(),
    )
    db.add(row)
    db.flush()
    events.append(
        db,
        user,
        "admission_request.created",
        ("admission_request", row.id),
        {"facility": _facility_code(db, user.facility_id), "urgent": urgent},
    )
    return row


def cancel_request(
    db: Session,
    user: AppUser,
    request_id: uuid.UUID,
    reason: str,
    now: datetime | None = None,
) -> AdmissionRequestCancellation:
    """Close a waiting request without an admission. The reason is required. Does not commit."""
    if not reason.strip():
        raise NotAllowed("Say why the request is cancelled.")
    line = _still_open(db, user, request_id)
    cancellation = AdmissionRequestCancellation(
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
        "admission_request.cancelled",
        ("admission_request", line.request.id),
        {"facility": _facility_code(db, line.request.facility_id)},
    )
    return cancellation


def for_clinician(db: Session, user: AppUser, now: datetime | None = None) -> list[AskLine]:
    """The facility's requests still waiting, and those closed in the last 72 hours; newest
    first."""
    since = (now or utcnow()) - SHOWN_AFTER_CLOSING
    return _lines(
        db,
        AdmissionRequest.facility_id == user.facility_id,
        or_(  # both closing tables are outer-joined here
            and_(
                AdmissionRequestFulfilment.id.is_(None), AdmissionRequestCancellation.id.is_(None)
            ),
            AdmissionRequestFulfilment.fulfilled_at >= since,
            AdmissionRequestCancellation.cancelled_at >= since,
        ),
    )


# ---------------------------------------------------------------- lab technician


def queue(db: Session, user: AppUser) -> list[AdmissionRequest]:
    """Waiting requests of the user's facility: urgent first, then the oldest."""
    return list(
        db.scalars(
            select(AdmissionRequest)
            .where(AdmissionRequest.facility_id == user.facility_id, _is_open())
            .order_by(
                AdmissionRequest.urgent.desc(),
                AdmissionRequest.requested_at,
                AdmissionRequest.created_at,
            )
        )
    )


def open_count(db: Session, user: AppUser) -> int:
    """How many requests of the user's facility wait for the lab (part of the navigation badge)."""
    return int(
        db.scalar(
            select(func.count())
            .select_from(AdmissionRequest)
            .where(AdmissionRequest.facility_id == user.facility_id, _is_open())
        )
        or 0
    )


def fulfil(
    db: Session,
    user: AppUser,
    request_id: uuid.UUID,
    admission: Admission,
    now: datetime | None = None,
) -> AdmissionRequestFulfilment:
    """Mark a waiting request entered by `admission`, just saved by the same user in the same
    transaction. NotFound for another facility's request, NotAllowed for a closed one. Does not
    commit."""
    line = _still_open(db, user, request_id)
    if admission.facility_id != line.request.facility_id:
        raise NotFound(str(request_id))
    fulfilment = AdmissionRequestFulfilment(
        request_id=line.request.id,
        admission_id=admission.id,
        fulfilled_by=user.id,
        fulfilled_at=now or utcnow(),
    )
    db.add(fulfilment)
    db.flush()
    events.append(
        db,
        user,
        "admission_request.fulfilled",
        ("admission_request", line.request.id),
        {"code": admission.code, "facility": _facility_code(db, admission.facility_id)},
    )
    return fulfilment
