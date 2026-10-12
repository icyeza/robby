"""National reviewer pages. Aggregates only: these routes call `birth_counts` (grouped counts),
the reference tables, `monthly_status` (export_record) and the facility rows. They never call a
function that returns admission rows, so no admission code or record can reach a template."""

from __future__ import annotations

import csv
import io
import uuid
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy import select

from robby.audit import formatting
from robby.audit.counts import birth_counts
from robby.audit.measures import (
    UNCLASSIFIED_CALLOUT,
    Comparison,
    Report,
    build_report,
    compare_to_expected,
    pct,
)
from robby.audit.narrative import FacilityComparison, national_stands_out
from robby.audit.references import available_references, choose_reference
from robby.audit.service import parse_period, report_context, today_local
from robby.audit.status import MonthStatus, monthly_status
from robby.auth.dependencies import CurrentUser, DbDep, require
from robby.models import Facility, Role
from robby.web.render import render

router = APIRouter(prefix="/national")
Reviewer = Annotated[CurrentUser, Depends(require(Role.NATIONAL_REVIEWER))]
HEAT_GROUPS = (1, 2, 5, 10)


def _tint(diff: Decimal | None) -> str:
    if diff is None:
        return ""
    if diff < 0:
        return "h-neg"
    if diff < 5:
        return "h-lo"
    if diff < 15:
        return "h-mid"
    return "h-hi"


@dataclass(frozen=True)
class HeatCell:
    text: str
    css: str


@dataclass(frozen=True)
class FacilityLine:
    facility: Facility
    report: Report
    comparison: Comparison
    obs_w: int | None
    exp_t: int | None
    statuses: list[MonthStatus]
    heat: list[HeatCell]

    @property
    def unclassified_high(self) -> bool:
        return self.report.unclassified_pct is not None and (
            self.report.unclassified_pct > UNCLASSIFIED_CALLOUT
        )


def _pos(value: Decimal | None, axis: Decimal) -> int | None:
    if value is None:
        return None
    return int(min(Decimal(100), (value * 100 / axis).quantize(Decimal(1), ROUND_HALF_UP)))


def _lines(
    db: DbDep, request: Request, period_from: str | None, period_to: str | None, ref_id: str | None
):  # type: ignore[no-untyped-def]
    zone = request.app.state.settings.zone
    period = parse_period(period_from, period_to, zone)
    refs = available_references(db)
    reference = choose_reference(refs, ref_id)
    facilities = list(db.scalars(select(Facility).where(Facility.active).order_by(Facility.name)))
    counts = birth_counts(db, period.start, period.end, zone)
    status = monthly_status(db, [f.id for f in facilities], today_local(zone))
    comps = {f.id: compare_to_expected(counts.get(f.id, {}), reference) for f in facilities}
    top = max(
        [c.observed or Decimal(0) for c in comps.values()]
        + [c.expected or Decimal(0) for c in comps.values()]
        + [Decimal(0)]
    )
    axis = Decimal(50) if top <= 50 else Decimal(100)
    lines: list[FacilityLine] = []
    for f in facilities:
        report = build_report(counts.get(f.id, {}), reference)
        by_group = {r.group: r for r in report.rows}
        heat = []
        for g in HEAT_GROUPS:
            row = by_group[g]
            usable = row.rate_diff is not None and row.births >= 10
            heat.append(
                HeatCell(
                    formatting.signed_whole(row.rate_diff) if usable else "—",
                    _tint(row.rate_diff) if usable else "",
                )
            )
        comp = comps[f.id]
        lines.append(
            FacilityLine(
                f,
                report,
                comp,
                _pos(comp.observed, axis),
                _pos(comp.expected, axis),
                status[f.id],
                heat,
            )
        )
    return period, refs, reference, lines


@router.get("")
def overview(request: Request, db: DbDep, current: Reviewer, ref: str | None = None) -> Response:
    q = request.query_params
    period, refs, reference, lines = _lines(db, request, q.get("from"), q.get("to"), ref)
    births = sum(line.report.births for line in lines)
    cesareans = sum(line.report.cesareans for line in lines)
    residual = sum(line.report.residual.births for line in lines)
    reporting = sum(1 for line in lines if line.report.births)
    high = sum(1 for line in lines if line.unclassified_high)
    stands = national_stands_out(
        [
            FacilityComparison(
                line.facility.name,
                line.report.births,
                line.comparison.ratio,
                line.report.unclassified_pct,
            )
            for line in lines
            if line.report.births
        ],
        reference is not None,
    )
    return render(
        request,
        "audit/national_overview.html",
        current=current,
        period=period,
        refs=refs,
        reference=reference,
        lines=lines,
        total_facilities=len(lines),
        reporting=reporting,
        births=births,
        cesareans=cesareans,
        cs_rate=pct(cesareans, births),
        residual=residual,
        residual_pct=pct(residual, births),
        high=high,
        stands_out=stands,
        fmt=formatting,
        heat_groups=HEAT_GROUPS,
    )


@router.get("/export.csv")
def overview_csv(
    request: Request, db: DbDep, current: Reviewer, ref: str | None = None
) -> Response:
    q = request.query_params
    period, _refs, reference, lines = _lines(db, request, q.get("from"), q.get("to"), ref)
    out = io.StringIO(newline="")
    w = csv.writer(out, lineterminator="\r\n")
    w.writerow(
        [
            "facility_code",
            "from",
            "to",
            "births",
            "cesareans",
            "unclassified",
            "observed_pct",
            "expected_pct",
            "observed_over_expected",
            "reference",
        ]
    )
    for line in lines:
        c = line.comparison
        w.writerow(
            [
                line.facility.code,
                period.start,
                period.end,
                line.report.births,
                line.report.cesareans,
                line.report.residual.births,
                c.observed or "",
                c.expected or "",
                c.ratio or "",
                reference.label if reference else "",
            ]
        )
    return Response(
        out.getvalue().encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="robby-national-overview.csv"'},
    )


@router.get("/facilities/{facility_id}")
def facility_report(
    facility_id: uuid.UUID, request: Request, db: DbDep, current: Reviewer, ref: str | None = None
) -> Response:
    facility = db.get(Facility, facility_id)
    if facility is None:
        raise HTTPException(status_code=404)
    zone = request.app.state.settings.zone
    period = parse_period(request.query_params.get("from"), request.query_params.get("to"), zone)
    context = report_context(db, zone, facility, period, ref)
    return render(request, "audit/report.html", current=current, readonly=True, **context)
