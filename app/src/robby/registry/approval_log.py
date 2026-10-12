"""Approval log: a read-only view of the event log's approvals, corrections and account changes,
and the integrity check ("Verify the log")."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Annotated, Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from robby import events
from robby.auth.dependencies import CurrentUser, DbDep, SettingsDep, require
from robby.models import AppUser, EventLog, Role, role_label
from robby.web.csrf import CsrfChecked
from robby.web.errors import redirect
from robby.web.render import render

router = APIRouter()
Custodian = Annotated[CurrentUser, Depends(require(Role.MODEL_CUSTODIAN))]

ACTIONS: dict[str, str] = {
    "model.registered": "Registered candidate",
    "model.shadowed": "Moved to shadow",
    "model.promoted": "Promoted to active",
    "model.rolled_back": "Rolled back to",
    "model.retired": "Retired",
    "model.activated": "Activated (development)",
    "admission.corrected": "Correction",
    "export.created": "Exported monthly report",
    "user.created": "Created account",
    "user.deactivated": "Deactivated account",
    "user.reactivated": "Reactivated account",
    "user.password_reset": "Reset password",
    "facility.created": "Added facility",
    "rule_set.loaded": "Loaded rule set",
}
GROUPS: dict[str, tuple[str, tuple[str, ...]]] = {
    "all": ("All", tuple(ACTIONS)),
    "models": ("Models", tuple(k for k in ACTIONS if k.startswith("model."))),
    "corrections": ("Corrections", ("admission.corrected",)),
    "accounts": (
        "Accounts",
        (
            "user.created",
            "user.deactivated",
            "user.reactivated",
            "user.password_reset",
            "facility.created",
        ),
    ),
    "other": ("Other", ("export.created", "rule_set.loaded")),
}
LIMIT = 300


@dataclass
class LogRow:
    when: datetime
    who: str
    role: str
    action: str
    item: str
    reason: str | None


def _item(kind: str, p: dict[str, Any]) -> str:
    if kind.startswith("model."):
        return str(p.get("version_label", "—"))
    if kind == "admission.corrected":
        return f"{p.get('supersedes', '?')} → {p.get('code', '?')}"
    if kind == "export.created":
        return f"{p.get('facility_code', '')} {p.get('period_month', '')}".strip() or "—"
    if kind.startswith("user."):
        return str(p.get("username", "—"))
    if kind == "facility.created":
        return f"{p.get('name', '')} ({p.get('code', '')})"
    if kind == "rule_set.loaded":
        return str(p.get("version_label", "—"))
    return "—"


def _action(kind: str, p: dict[str, Any]) -> str:
    label = ACTIONS.get(kind, kind)
    if kind == "model.rolled_back":
        return "Rolled back"
    return label


def parse_day(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def log_rows(
    db: Session, zone: ZoneInfo, group: str, start: date | None, end: date | None
) -> tuple[list[LogRow], int]:
    types = GROUPS.get(group, GROUPS["all"])[1]
    where: list[Any] = [EventLog.event_type.in_(types)]
    if start:
        where.append(EventLog.occurred_at >= datetime.combine(start, time.min, zone))
    if end:
        where.append(
            EventLog.occurred_at < datetime.combine(end + timedelta(days=1), time.min, zone)
        )
    total = db.scalar(select(func.count()).select_from(EventLog).where(*where)) or 0
    found = db.execute(
        select(
            EventLog.occurred_at,
            EventLog.event_type,
            EventLog.payload,
            AppUser.username,
            AppUser.role,
        )
        .join(AppUser, AppUser.id == EventLog.actor_id, isouter=True)
        .where(*where)
        .order_by(EventLog.seq.desc())
        .limit(LIMIT)
    ).all()
    rows = [
        LogRow(
            when=r.occurred_at,
            who=r.username or "System",
            role=role_label(Role(r.role)) if r.role else "—",
            action=_action(r.event_type, r.payload),
            item=_item(r.event_type, r.payload),
            reason=(str(r.payload["reason"]).strip() or None) if r.payload.get("reason") else None,
        )
        for r in found
    ]
    return rows, int(total)


def last_verification(db: Session) -> tuple[datetime, dict[str, Any]] | None:
    row = db.execute(
        select(EventLog.occurred_at, EventLog.payload)
        .where(EventLog.event_type == "events.verified")
        .order_by(EventLog.seq.desc())
        .limit(1)
    ).first()
    return (row.occurred_at, row.payload) if row else None


def run_verification(db: Session, actor: AppUser) -> dict[str, Any]:
    """Verify the chain and record the outcome (counts only). Never raises."""
    try:
        result = events.verify(db)
        payload: dict[str, Any] = {
            "result": "intact" if result.ok else "altered",
            "checked": result.checked,
        }
        if not result.ok:
            payload["first_bad_seq"] = result.first_bad_seq
            payload["reason"] = result.reason or ""
    except Exception:
        db.rollback()
        payload = {"result": "failed", "checked": 0}
    events.append(db, actor, "events.verified", None, payload)
    return payload


def verification_message(payload: dict[str, Any]) -> tuple[bool, str]:
    kind = payload.get("result")
    if kind == "intact":
        return True, f"Log intact: {payload.get('checked', 0)} entries verified."
    if kind == "altered":
        return (
            False,
            f"Log altered at entry #{payload.get('first_bad_seq')}: {payload.get('reason')}.",
        )
    return False, "The check could not be completed. Try again, or run robby-admin events verify."


@router.get("/approval-log")
def approval_log(
    request: Request,
    db: DbDep,
    current: Custodian,
    settings: SettingsDep,
    type: str = "all",
    date_from: str = "",
    date_to: str = "",
) -> Response:
    group = type if type in GROUPS else "all"
    start, end = parse_day(date_from), parse_day(date_to)
    rows, total = log_rows(db, settings.zone, group, start, end)
    last = last_verification(db)
    verdict = verification_message(last[1]) if last else None
    return render(
        request,
        "models/approval_log.html",
        current=current,
        rows=rows,
        total=total,
        group=group,
        groups=[(key, label) for key, (label, _) in GROUPS.items()],
        date_from=start.isoformat() if start else "",
        date_to=end.isoformat() if end else "",
        verified_at=last[0] if last else None,
        verdict=verdict,
        limit=LIMIT,
    )


@router.post("/approval-log/verify")
def verify_log(request: Request, db: DbDep, current: Custodian, _csrf: CsrfChecked) -> Response:
    run_verification(db, current.user)
    db.commit()
    return redirect(request, "/approval-log")
