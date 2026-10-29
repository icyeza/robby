"""Plain-language reading of the reports: group names in everyday words, one verdict per group or
hospital, "about N in 100", the period shortcuts and the ranking of where cesareans come from.
Thresholds are those of `measures`, so a verdict and a flag always agree."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

from robby.audit.measures import (
    MIN_BIRTHS_FOR_RATE_FLAG,
    OE_HIGH,
    OE_LOW,
    RATE_FLAG_POINTS,
    Comparison,
    GroupRow,
    Report,
)

PLAIN_GROUPS = {
    1: "First baby, head first, full term, labour started on its own",
    2: "First baby, head first, full term, labour induced or cesarean before labour",
    3: "Has given birth before, no previous cesarean, head first, full term, labour started on "
    "its own",
    4: "Has given birth before, no previous cesarean, head first, full term, labour induced or "
    "cesarean before labour",
    5: "Has had a cesarean before, head first, full term",
    6: "First baby, bottom first (breech)",
    7: "Has given birth before, bottom first (breech)",
    8: "Twins or more",
    9: "Baby lying sideways",
    10: "Head first, born early (before 37 weeks)",
}
SHORT_GROUPS = {
    1: "first baby, labour started on its own",
    2: "first baby, labour induced or cesarean before labour",
    3: "has given birth before, labour started on its own",
    4: "has given birth before, labour induced or cesarean before labour",
    5: "has had a cesarean before",
    6: "first baby, breech",
    7: "has given birth before, breech",
    8: "twins or more",
    9: "baby lying sideways",
    10: "born early",
}


@dataclass(frozen=True)
class Verdict:
    label: str
    css: str  # high | low | same | muted


def in_100(value: Decimal | None) -> str:
    """'about 28 in 100' for 28.2%; a dash for no value."""
    if value is None:
        return "—"
    return f"about {int(value.quantize(Decimal(1), ROUND_HALF_UP))} in 100"


def group_verdict(row: GroupRow) -> Verdict:
    """How the group's cesarean rate compares with the reference, in words."""
    if row.births == 0:
        return Verdict("No births", "muted")
    if row.rate_diff is None:
        return Verdict("No reference", "muted")
    if row.births < MIN_BIRTHS_FOR_RATE_FLAG:
        return Verdict("Too few births to judge", "muted")
    if row.rate_diff >= RATE_FLAG_POINTS:
        return Verdict("Higher than reference", "high")
    if row.rate_diff <= -RATE_FLAG_POINTS:
        return Verdict("Lower than reference", "low")
    return Verdict("About the same", "same")


def hospital_verdict(comparison: Comparison) -> Verdict:
    """Observed against expected for the hospital's mix of women, in words."""
    if comparison.ratio is None:
        return Verdict("No comparison", "muted")
    if comparison.ratio >= OE_HIGH:
        return Verdict("More than expected", "high")
    if comparison.ratio <= OE_LOW:
        return Verdict("Fewer than expected", "low")
    return Verdict("As expected", "same")


@dataclass(frozen=True)
class Source:
    group: int
    cesareans: int
    share: Decimal  # % of all the hospital's cesareans
    width: int  # bar width 0-100, relative to the largest source


def cesarean_sources(report: Report) -> list[Source]:
    """Groups with at least one cesarean, largest share first."""
    rows = [r for r in report.rows if r.cesareans and r.group and r.share_of_cs_pct is not None]
    rows.sort(key=lambda r: (-r.cesareans, r.group or 0))
    if not rows:
        return []
    top = rows[0].share_of_cs_pct or Decimal(1)
    return [
        Source(
            r.group or 0,
            r.cesareans,
            r.share_of_cs_pct or Decimal(0),
            int(((r.share_of_cs_pct or Decimal(0)) * 100 / top).quantize(Decimal(1))),
        )
        for r in rows
    ]


@dataclass(frozen=True)
class Preset:
    label: str
    start: date
    end: date
    active: bool


def period_presets(today: date, start: date, end: date) -> list[Preset]:
    """This year, the last three months and last month; `active` marks the one shown."""
    first_this_month = today.replace(day=1)
    last_month_end = first_this_month - timedelta(days=1)
    three_back = first_this_month
    for _ in range(3):
        three_back = (three_back - timedelta(days=1)).replace(day=1)
    choices = [
        ("This year", today.replace(month=1, day=1), today),
        ("Last 3 months", three_back, today),
        ("Last month", last_month_end.replace(day=1), last_month_end),
    ]
    return [Preset(label, s, e, (s, e) == (start, end)) for label, s, e in choices]
