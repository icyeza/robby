"""Monthly report status from export_record. Robby cannot see receipt by the national system, so
the word is 'exported', never 'submitted'. A month's export is due on the 10th of the next month."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from robby.models import ExportRecord

DUE_DAY = 10


@dataclass(frozen=True)
class MonthStatus:
    month: date
    state: str  # exported | due | late
    label: str

    @property
    def css(self) -> str:
        return "done" if self.state == "exported" else "awaiting"


def month_start(d: date) -> date:
    return d.replace(day=1)


def previous_month(d: date) -> date:
    first = month_start(d)
    return (first - timedelta(days=1)).replace(day=1)


def complete_months(today: date, n: int) -> list[date]:
    """The last n complete calendar months, newest first."""
    out: list[date] = []
    cursor = month_start(today)
    for _ in range(n):
        cursor = previous_month(cursor)
        out.append(cursor)
    return out


def _due(month: date) -> date:
    nxt = date(month.year + (month.month == 12), month.month % 12 + 1, DUE_DAY)
    return nxt


def monthly_status(
    db: Session, facility_ids: list[uuid.UUID], today: date
) -> dict[uuid.UUID, list[MonthStatus]]:
    months = complete_months(today, 2)
    exported = set(
        db.execute(
            select(ExportRecord.facility_id, ExportRecord.period_month).where(
                ExportRecord.period_month.in_(months)
            )
        ).all()
    )
    out: dict[uuid.UUID, list[MonthStatus]] = {}
    for fid in facility_ids:
        items: list[MonthStatus] = []
        for m in months:
            name = m.strftime("%b")
            if (fid, m) in exported:
                items.append(MonthStatus(m, "exported", f"{name} exported"))
            elif today <= _due(m):
                items.append(MonthStatus(m, "due", f"{name} due {_due(m):%d %b}"))
            else:
                items.append(MonthStatus(m, "late", f"{name} late (due {_due(m):%d %b})"))
        out[fid] = items
    return out
