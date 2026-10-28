"""Theatre readiness board: a unit-level planning figure, never guidance about one woman.

Counts the facility's current admissions of the last 24 hours that have no outcome yet, a resolved
Robson group and an active prediction, and sums their probabilities (expected cesareans, with the
90% interval of the exact Poisson-binomial distribution)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from robson_serving.monitoring import expected_theatre_load
from sqlalchemy import exists, select
from sqlalchemy.orm import Session, aliased

from robby import events
from robby.auth.dependencies import CurrentUser, DbDep, require
from robby.clinical.service import chain_outcome
from robby.models import (
    Admission,
    Classification,
    ClassificationStatus,
    Facility,
    Prediction,
    PredictionMode,
    Role,
)
from robby.timeutil import utcnow
from robby.web.csrf import CsrfChecked
from robby.web.render import render

WINDOW = timedelta(hours=24)


@dataclass
class GroupLine:
    group: int
    n: int
    expected: float


@dataclass
class Board:
    since: datetime
    now: datetime
    awaiting: int = 0  # current admissions without an outcome, included or not
    included: int = 0
    not_included: int = 0
    expected: float = 0.0
    low: int = 0
    high: int = 0
    flagged: int = 0  # included admissions at or above the facility's readiness cut
    cut_percent: int = 30
    groups: list[GroupLine] = field(default_factory=list)


def is_high(probability: float, cut_percent: int) -> bool:
    """High readiness: the probability is at or above the facility's cut."""
    return 100 * float(probability) >= cut_percent


def build(
    db: Session, facility_id: object, now: datetime | None = None, cut_percent: int = 30
) -> Board:
    now = now or utcnow()
    board = Board(since=now - WINDOW, now=now, cut_percent=cut_percent)
    newer = aliased(Admission)
    admissions = db.scalars(
        select(Admission)
        .where(
            Admission.facility_id == facility_id,
            Admission.admitted_at >= board.since,
            Admission.admitted_at <= now,
            ~exists().where(newer.supersedes_id == Admission.id),
        )
        .order_by(Admission.admitted_at)
    ).all()
    waiting = [a for a in admissions if chain_outcome(db, a) is None]
    board.awaiting = len(waiting)
    if not waiting:
        return board
    ids = [a.id for a in waiting]
    groups: dict[object, int] = {}
    for c in db.scalars(
        select(Classification)
        .where(
            Classification.admission_id.in_(ids),
            Classification.status == ClassificationStatus.RESOLVED,
        )
        .order_by(Classification.created_at)
    ):
        if c.robson_group is not None:
            groups[c.admission_id] = c.robson_group  # the latest classification wins
    probabilities: dict[object, float] = {}
    for p in db.scalars(
        select(Prediction)
        .where(Prediction.admission_id.in_(ids), Prediction.mode == PredictionMode.ACTIVE)
        .order_by(Prediction.created_at)
    ):
        probabilities[p.admission_id] = float(p.probability)
    by_group: dict[int, list[float]] = defaultdict(list)
    for a in waiting:
        if a.id in groups and a.id in probabilities:
            by_group[groups[a.id]].append(probabilities[a.id])
    flat = [p for values in by_group.values() for p in values]
    board.included = len(flat)
    board.flagged = sum(is_high(p, cut_percent) for p in flat)
    board.not_included = board.awaiting - board.included
    if flat:
        load = expected_theatre_load(flat)
        board.expected, board.low, board.high = load.expected, load.low, load.high
        board.groups = [GroupLine(g, len(v), sum(v)) for g, v in sorted(by_group.items())]
    return board


router = APIRouter(prefix="/readiness")
Reader = Annotated[CurrentUser, Depends(require(Role.CLINICIAN, Role.FACILITY_ADMIN))]


@router.get("")
def readiness_page(request: Request, db: DbDep, current: Reader) -> Response:
    facility = db.get(Facility, current.user.facility_id) if current.user.facility_id else None
    if facility is None:
        raise HTTPException(status_code=404)
    board = build(db, facility.id, cut_percent=facility.readiness_cut_percent)
    return render(
        request, "monitoring/readiness.html", current=current, board=board, facility=facility
    )


FacilityAdmin = Annotated[CurrentUser, Depends(require(Role.FACILITY_ADMIN))]
CUT_RANGE = (5, 95)


def _own_facility(db: Session, current: CurrentUser) -> Facility:
    facility = db.get(Facility, current.user.facility_id) if current.user.facility_id else None
    if facility is None:
        raise HTTPException(status_code=404)
    return facility


def _level_page(
    request: Request,
    current: CurrentUser,
    facility: Facility,
    problem: str | None = None,
    status: int = 200,
) -> Response:
    return render(
        request,
        "monitoring/readiness_level.html",
        current=current,
        status_code=status,
        facility=facility,
        problem=problem,
    )


@router.get("/level")
def level_page(request: Request, db: DbDep, current: FacilityAdmin) -> Response:
    return _level_page(request, current, _own_facility(db, current))


@router.post("/level")
def set_level(
    request: Request,
    db: DbDep,
    current: FacilityAdmin,
    _csrf: CsrfChecked,
    cut_percent: Annotated[str, Form()] = "",
    reason: Annotated[str, Form()] = "",
) -> Response:
    facility = _own_facility(db, current)
    low, high = CUT_RANGE
    try:
        value = int(cut_percent.strip())
    except ValueError:
        value = -1
    if not low <= value <= high:
        problem = f"Enter a whole number from {low} to {high}."
        return _level_page(request, current, facility, problem, 400)
    if not reason.strip():
        problem = "Give a reason for the change."
        return _level_page(request, current, facility, problem, 400)
    if value != facility.readiness_cut_percent:
        previous = facility.readiness_cut_percent
        facility.readiness_cut_percent = value
        db.flush()
        events.append(
            db,
            current.user,
            "facility.readiness_cut_changed",
            ("facility", facility.id),
            {"code": facility.code, "from": previous, "to": value, "reason": reason.strip()},
        )
        db.commit()
    return RedirectResponse("/readiness/level", status_code=303)
