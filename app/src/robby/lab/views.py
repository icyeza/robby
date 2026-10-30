"""What the pages show of lab requests: the clinician's list on the patient page (with the
screening results completed later), the clinician's admission requests on the Find patient page,
and the lab technician's queue of both kinds. Built here so the templates get words and times,
not rows to interpret."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

from sqlalchemy.orm import Session

from robby.lab import admission_requests, catalogue, service
from robby.models import Admission, AdmissionRequest, LabRequest

STATUS_TEXT = {
    "requested": "Waiting for the lab",
    "resulted": "Result recorded",
    "cancelled": "Cancelled",
}
ASK_STATUS_TEXT = {
    "waiting": "Waiting for the lab",
    "entered": "Entered by the lab",
    "cancelled": "Cancelled",
}
ENTER_ADMISSION = "Enter new admission"


@dataclass(frozen=True)
class LineView:
    request_id: str
    test: str
    status: str  # requested | resulted | cancelled
    status_text: str
    urgent: bool
    note: str | None
    requested_at: datetime
    result: str | None = None  # in words, with the unit
    closed_at: datetime | None = None  # when the result or the cancellation was recorded
    reason: str | None = None  # why it was cancelled


@dataclass(frozen=True)
class LaterResult:
    """A screening test recorded as not measured at admission, completed later by the lab."""

    text: str
    at: datetime


@dataclass(frozen=True)
class LabSection:
    lines: list[LineView]
    can_request: bool  # some test is not waiting already
    later: dict[str, LaterResult]  # screening key (protein | glucose) -> the latest result


@dataclass(frozen=True)
class AskView:
    """An admission request on the clinician's Find patient page."""

    request_id: str
    patient_number: str
    patient_name: str | None
    status: str  # waiting | entered | cancelled
    status_text: str
    urgent: bool
    note: str | None
    requested_at: datetime
    code: str | None = None  # the admission the lab entered
    closed_at: datetime | None = None
    reason: str | None = None  # why it was cancelled


@dataclass(frozen=True)
class QueueRow:
    request_id: str
    test: str
    urgent: bool
    note: str | None
    requested_at: datetime
    code: str | None  # None for a patient the lab has not entered yet
    patient_name: str | None
    patient_number: str | None
    href: str = ""
    action: str = "Record result"


def line_view(line: service.LabLine) -> LineView:
    request = line.request
    base = LineView(
        request_id=str(request.id),
        test=catalogue.test_name(request.test),
        status=line.status,
        status_text=STATUS_TEXT[line.status],
        urgent=request.urgent,
        note=request.note,
        requested_at=request.requested_at,
    )
    if line.result is not None:
        return replace(
            base,
            result=catalogue.result_text(request.test, line.result.value),
            closed_at=line.result.resulted_at,
        )
    if line.cancellation is not None:
        return replace(
            base, closed_at=line.cancellation.cancelled_at, reason=line.cancellation.reason
        )
    return base


def section(db: Session, admission: Admission) -> LabSection:
    lines = service.lines_for(db, admission)
    later: dict[str, LaterResult] = {}
    # oldest first, so the latest result of each test is the one kept
    for line, result in sorted(
        ((x, x.result) for x in lines if x.result is not None), key=lambda pair: pair[1].resulted_at
    ):
        key = catalogue.SCREENING_KEY.get(line.request.test)
        if key:
            text = catalogue.result_text(line.request.test, result.value)
            later[key] = LaterResult(text, result.resulted_at)
    waiting = {line.request.test for line in lines if line.is_open}
    return LabSection(
        lines=[line_view(line) for line in lines],
        can_request=len(waiting) < len(catalogue.TEST_ORDER),
        later=later,
    )


def ask_view(line: admission_requests.AskLine) -> AskView:
    request = line.request
    base = AskView(
        request_id=str(request.id),
        patient_number=request.patient_number,
        patient_name=request.patient_name,
        status=line.status,
        status_text=ASK_STATUS_TEXT[line.status],
        urgent=request.urgent,
        note=request.note,
        requested_at=request.requested_at,
    )
    if line.fulfilment is not None:
        return replace(base, code=line.code, closed_at=line.fulfilment.fulfilled_at)
    if line.cancellation is not None:
        return replace(
            base, closed_at=line.cancellation.cancelled_at, reason=line.cancellation.reason
        )
    return base


def queue_rows(
    rows: list[tuple[LabRequest, Admission]], asks: list[AdmissionRequest] | None = None
) -> list[QueueRow]:
    """Lab requests and admission requests in one list: urgent first, then the oldest (each
    list arrives in that order; the sort is stable)."""
    tests = [
        QueueRow(
            request_id=str(request.id),
            test=catalogue.test_name(request.test),
            urgent=request.urgent,
            note=request.note,
            requested_at=request.requested_at,
            code=admission.code,
            patient_name=admission.patient_name,
            patient_number=admission.patient_number,
            href=f"/lab-requests/{request.id}",
        )
        for request, admission in rows
    ]
    entries = [
        QueueRow(
            request_id=str(ask.id),
            test=ENTER_ADMISSION,
            urgent=ask.urgent,
            note=ask.note,
            requested_at=ask.requested_at,
            code=None,
            patient_name=ask.patient_name,
            patient_number=ask.patient_number,
            href=f"/admissions/new?request={ask.id}",
            action="Enter admission",
        )
        for ask in asks or []
    ]
    return sorted(tests + entries, key=lambda row: (not row.urgent, row.requested_at))
