"""Facilities: create, list with their administrators, deactivate."""

from __future__ import annotations

import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from robby import events
from robby.errors import FormErrors
from robby.models import AppUser, Facility, Role

CODE_PATTERN = re.compile(r"[A-Z]{2,6}")


@dataclass(frozen=True)
class FacilityRow:
    facility: Facility
    administrators: list[str]
    active_users: int


def _squash(value: str) -> str:
    return " ".join(value.split())


def list_facilities(db: Session) -> list[FacilityRow]:
    facilities = list(db.scalars(select(Facility).order_by(Facility.name)))
    people = db.execute(
        select(AppUser.facility_id, AppUser.username, AppUser.role)
        .where(AppUser.active, AppUser.facility_id.is_not(None))
        .order_by(AppUser.username)
    ).all()
    administrators: dict[uuid.UUID, list[str]] = defaultdict(list)
    counts: Counter[uuid.UUID] = Counter()
    for facility_id, username, role in people:
        counts[facility_id] += 1
        if role is Role.FACILITY_ADMIN:
            administrators[facility_id].append(username)
    return [
        FacilityRow(f, list(administrators.get(f.id, [])), counts.get(f.id, 0)) for f in facilities
    ]


def create_facility(
    db: Session,
    actor: AppUser,
    *,
    name: str,
    code: str,
    district: str,
    model_facility_key: str = "",
) -> Facility:
    name, code, district = _squash(name), code.strip().upper(), _squash(district)
    errors: dict[str, str] = {}
    if not name:
        errors["name"] = "Enter the facility name."
    elif db.scalar(select(Facility.id).where(func.lower(Facility.name) == name.lower())):
        errors["name"] = "A facility with this name already exists."
    if not CODE_PATTERN.fullmatch(code):
        errors["code"] = "Use 2 to 6 capital letters, for example FA."
    elif db.scalar(select(Facility.id).where(Facility.code == code)):
        errors["code"] = "This code is already used by another facility."
    if not district:
        errors["district"] = "Enter the district."
    if errors:
        raise FormErrors(errors)
    facility = Facility(
        name=name, code=code, district=district, model_facility_key=model_facility_key.strip() or None
    )
    db.add(facility)
    db.flush()
    events.append(
        db, actor, "facility.created", ("facility", facility.id),
        {"name": name, "code": code, "district": district},
    )
    return facility


def deactivate_facility(db: Session, actor: AppUser, facility: Facility) -> None:
    """Its accounts can no longer sign in (auth.service.can_sign_in); its records are kept."""
    if not facility.active:
        return
    facility.active = False
    db.flush()
    events.append(db, actor, "facility.deactivated", ("facility", facility.id), {"code": facility.code})
