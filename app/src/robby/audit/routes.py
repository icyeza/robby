"""Facility administrator pages: hospital report and monthly export (FR10)."""

from __future__ import annotations

import hashlib
import re
from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy import select

from robby import events
from robby.audit import export_format, pdf
from robby.audit.counts import birth_counts
from robby.audit.service import (
    month_range,
    parse_period,
    report_context,
    today_local,
    unclassified_reasons,
)
from robby.audit.status import complete_months
from robby.auth.dependencies import CurrentUser, DbDep, require
from robby.models import ExportRecord, Facility, Role
from robby.web.render import render

router = APIRouter(prefix="/audit")
FacilityAdmin = Annotated[CurrentUser, Depends(require(Role.FACILITY_ADMIN))]
MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def _facility(db: DbDep, current: CurrentUser) -> Facility:
    """The facility always comes from the session, never from the request."""
    facility = db.get(Facility, current.user.facility_id) if current.user.facility_id else None
    if facility is None:
        raise HTTPException(status_code=404)
    return facility


@router.get("")
def audit_page(
    request: Request,
    db: DbDep,
    current: FacilityAdmin,
    ref: str | None = None,
) -> Response:
    zone = request.app.state.settings.zone
    query = request.query_params
    period = parse_period(query.get("from"), query.get("to"), zone)
    facility = _facility(db, current)
    context = report_context(db, zone, facility, period, ref)
    today = today_local(zone)
    recent = db.scalars(
        select(ExportRecord)
        .where(ExportRecord.facility_id == facility.id)
        .order_by(ExportRecord.exported_at.desc())
        .limit(5)
    ).all()
    return render(
        request,
        "audit/report.html",
        current=current,
        readonly=False,
        months=complete_months(today, 12),
        recent_exports=recent,
        reasons=unclassified_reasons(db, zone, facility.id, period),
        zone=zone,
        **context,
    )


@router.get("/report.pdf")
def report_pdf(
    request: Request, db: DbDep, current: FacilityAdmin, ref: str | None = None
) -> Response:
    settings = request.app.state.settings
    period = parse_period(
        request.query_params.get("from"), request.query_params.get("to"), settings.zone
    )
    facility = _facility(db, current)
    context = report_context(db, settings.zone, facility, period, ref)
    context["reasons"] = unclassified_reasons(db, settings.zone, facility.id, period)
    name = f"robby-{facility.code.lower()}-report.pdf"
    return Response(
        pdf.hospital_pdf(context, demo=settings.demo, zone=settings.zone),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@router.get("/export.csv")
def export_csv(request: Request, db: DbDep, current: FacilityAdmin, month: str = "") -> Response:
    zone = request.app.state.settings.zone
    if not MONTH.match(month):
        raise HTTPException(status_code=400, detail="month must be YYYY-MM")
    first = date(int(month[:4]), int(month[5:]), 1)
    if first not in complete_months(today_local(zone), 12):
        raise HTTPException(
            status_code=400, detail="Only the last 12 complete months can be exported."
        )
    facility = _facility(db, current)
    start, end = month_range(first)
    counts = birth_counts(db, start, end, zone, facility.id).get(facility.id, {})
    body = export_format.render_monthly_csv(facility.code, first, counts)
    digest = hashlib.sha256(body).hexdigest()
    record = ExportRecord(
        facility_id=facility.id,
        period_month=first,
        exported_by=current.user.id,
        file_sha256=digest,
    )
    db.add(record)
    db.flush()
    events.append(
        db,
        current.user,
        "export.created",
        entity=("export_record", record.id),
        payload={
            "facility_code": facility.code,
            "period_month": month,
            "format_version": export_format.EXPORT_FORMAT_VERSION,
            "sha256": digest,
        },
    )
    db.commit()
    name = export_format.filename(facility.code, first)
    return Response(
        body,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )
