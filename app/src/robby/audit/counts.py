"""Grouped counts for the audit and national reports.

Definitions (shared with the offline ML audit):
- A birth is a *current* admission (no other admission supersedes it) with a recorded delivery
  outcome whose `delivered_at` falls in the period (local calendar dates, both ends included).
- Its Robson group comes from its newest unsuperseded classification; anything not `resolved`
  (partial, conflict, or no classification) is the residual (group key None).
- Admissions in the period with no outcome yet are counted separately as "awaiting outcome".

Every function here returns counts only, except `unclassified_births`, which feeds the facility
administrator's list (codes and patient numbers, never names) and is never used by national
routes."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import text
from sqlalchemy.orm import Session

GroupKey = int | None  # 1..10, or None for the residual


@dataclass(frozen=True)
class GroupCount:
    births: int = 0
    cesareans: int = 0


Counts = dict[GroupKey, GroupCount]

_BORN = """
born AS (
  SELECT a.id, a.facility_id, a.code, a.admitted_at, o.mode
  FROM admission a JOIN delivery_outcome o ON o.admission_id = a.id
  WHERE o.delivered_at >= :lo AND o.delivered_at < :hi
    AND NOT EXISTS (SELECT 1 FROM admission s WHERE s.supersedes_id = a.id)
    {facility}
),
cls AS (
  SELECT DISTINCT ON (c.admission_id) c.admission_id, c.status, c.robson_group,
         c.resolving_fields, c.conflict_fields
  FROM classification c JOIN born b ON b.id = c.admission_id
  WHERE NOT EXISTS (SELECT 1 FROM classification s WHERE s.supersedes_id = c.id)
  ORDER BY c.admission_id, c.created_at DESC
)
"""


def period_bounds(start: date, end: date, zone: ZoneInfo) -> tuple[datetime, datetime]:
    """[lo, hi) instants for the local dates start..end inclusive."""
    return (
        datetime.combine(start, time.min, tzinfo=zone),
        datetime.combine(end + timedelta(days=1), time.min, tzinfo=zone),
    )


def _facility_clause(facility_id: uuid.UUID | None, alias: str = "a") -> str:
    return f"AND {alias}.facility_id = :fid" if facility_id is not None else ""


def birth_counts(
    db: Session,
    start: date,
    end: date,
    zone: ZoneInfo,
    facility_id: uuid.UUID | None = None,
) -> dict[uuid.UUID, Counts]:
    """One grouped statement: births and cesareans per facility and Robson group."""
    lo, hi = period_bounds(start, end, zone)
    sql = text(
        "WITH "
        + _BORN.format(facility=_facility_clause(facility_id))
        + """
        SELECT b.facility_id,
               CASE WHEN cls.status = 'resolved' THEN cls.robson_group END AS grp,
               count(*) AS births,
               count(*) FILTER (WHERE b.mode = 'cesarean') AS cesareans
        FROM born b LEFT JOIN cls ON cls.admission_id = b.id
        GROUP BY b.facility_id, grp
        """
    )
    params: dict[str, object] = {"lo": lo, "hi": hi}
    if facility_id is not None:
        params["fid"] = facility_id
    out: dict[uuid.UUID, Counts] = {}
    for fid, grp, births, cesareans in db.execute(sql, params):
        out.setdefault(fid, {})[grp] = GroupCount(births, cesareans)
    return out


def awaiting_outcome(
    db: Session,
    start: date,
    end: date,
    zone: ZoneInfo,
    facility_id: uuid.UUID | None = None,
) -> dict[uuid.UUID, int]:
    """Current admissions in the period (by admission time) with no outcome yet."""
    lo, hi = period_bounds(start, end, zone)
    sql = text(
        f"""
        SELECT a.facility_id, count(*)
        FROM admission a
        WHERE a.admitted_at >= :lo AND a.admitted_at < :hi
          AND NOT EXISTS (SELECT 1 FROM admission s WHERE s.supersedes_id = a.id)
          AND NOT EXISTS (SELECT 1 FROM delivery_outcome o WHERE o.admission_id = a.id)
          {_facility_clause(facility_id)}
        GROUP BY a.facility_id
        """
    )
    params: dict[str, object] = {"lo": lo, "hi": hi}
    if facility_id is not None:
        params["fid"] = facility_id
    return {fid: n for fid, n in db.execute(sql, params)}


@dataclass(frozen=True)
class UnclassifiedBirth:
    code: str
    admitted_at: datetime
    status: str | None  # 'partial', 'conflict', or None when no classification exists
    resolving_fields: tuple[str, ...]
    conflict_fields: tuple[str, ...]
    patient_number: str | None = None  # never the name: this feeds the administrator's list


def unclassified_births(
    db: Session, start: date, end: date, zone: ZoneInfo, facility_id: uuid.UUID
) -> list[UnclassifiedBirth]:
    """Births of one facility without a resolved group: code, patient number, time, status,
    fields. Newest first. The shared `born` CTE stays free of identifiers; the number is joined
    here only."""
    lo, hi = period_bounds(start, end, zone)
    sql = text(
        "WITH "
        + _BORN.format(facility=_facility_clause(facility_id))
        + """
        SELECT b.code, b.admitted_at, cls.status::text, cls.resolving_fields, cls.conflict_fields,
               a.patient_number
        FROM born b JOIN admission a ON a.id = b.id
        LEFT JOIN cls ON cls.admission_id = b.id
        WHERE cls.status IS DISTINCT FROM 'resolved'
        ORDER BY b.admitted_at DESC, b.code
        """
    )
    params = {"lo": lo, "hi": hi, "fid": facility_id}
    return [
        UnclassifiedBirth(code, at, status, tuple(res or ()), tuple(conf or ()), number)
        for code, at, status, res, conf, number in db.execute(sql, params)
    ]
