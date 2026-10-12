"""The audit arithmetic: pure functions over counts (kept consistent with the offline
`robson_ml.audit_offline` / `casemix` definitions, but this module never imports them).

N = births, C = cesareans, R = residual births. For group g with n_g births, c_g cesareans:
group size = n_g/N; group rate = c_g/n_g; absolute contribution = c_g/N; share of cesareans =
c_g/C. Expected rate for a case mix = sum_g (n_g / N_resolved) x reference rate_g, where
N_resolved = N - R. Observed (for the comparison) = sum c_g / N_resolved."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from robby.audit.counts import Counts, GroupCount
from robby.audit.references import Reference

GROUPS = tuple(range(1, 11))
SIZE_FLAG_POINTS = Decimal("3.0")
RATE_FLAG_POINTS = Decimal("10.0")
MIN_BIRTHS_FOR_RATE_FLAG = 10
UNCLASSIFIED_CALLOUT = Decimal("5.0")
OE_HIGH = Decimal("1.10")
OE_LOW = Decimal("0.90")
TENTH = Decimal("0.1")

GROUP_DEFINITIONS = {
    1: "Nulliparous, single cephalic, ≥37 wk, spontaneous labour",
    2: "Nulliparous, single cephalic, ≥37 wk, induced or pre-labour CS",
    3: "Multiparous, no previous CS, single cephalic, ≥37 wk, spontaneous labour",
    4: "Multiparous, no previous CS, single cephalic, ≥37 wk, induced or pre-labour CS",
    5: "Multiparous with at least one previous CS, single cephalic, ≥37 wk",
    6: "Nulliparous, single breech",
    7: "Multiparous, single breech, including previous CS",
    8: "Multiple pregnancy, including previous CS",
    9: "Single transverse or oblique, including previous CS",
    10: "Single cephalic, <37 wk, including previous CS",
}


def pct(n: int | Decimal, d: int | Decimal) -> Decimal | None:
    """n/d as a percentage with one decimal, half up; None when d is 0."""
    if not d:
        return None
    return (Decimal(n) * 100 / Decimal(d)).quantize(TENTH, rounding=ROUND_HALF_UP)


def _axis_pos(value: Decimal | None, axis_max: Decimal) -> int | None:
    if value is None:
        return None
    return int(min(Decimal(100), (value * 100 / axis_max).quantize(Decimal(1), ROUND_HALF_UP)))


@dataclass(frozen=True)
class GroupRow:
    group: int | None  # None = unclassified residual
    births: int
    cesareans: int
    size_pct: Decimal | None
    rate_pct: Decimal | None
    share_of_cs_pct: Decimal | None
    contribution_pct: Decimal | None
    ref_size: Decimal | None
    ref_rate: Decimal | None
    size_diff: Decimal | None
    rate_diff: Decimal | None
    larger_group: bool
    higher_rate: bool
    size_w: int | None  # bar widths / tick positions, 0-100 (CSS classes w{n} / t{n})
    size_t: int | None
    rate_w: int | None
    rate_t: int | None

    @property
    def close(self) -> bool:
        return self.group is not None and not (self.larger_group or self.higher_rate)


@dataclass(frozen=True)
class Report:
    births: int
    cesareans: int
    cs_rate: Decimal | None
    rows: list[GroupRow]  # groups 1-10
    residual: GroupRow
    unclassified_pct: Decimal | None
    largest_group: int | None
    largest_share: Decimal | None
    reference: Reference | None
    awaiting: int

    @property
    def unclassified_high(self) -> bool:
        return self.unclassified_pct is not None and self.unclassified_pct > UNCLASSIFIED_CALLOUT


def _row(
    group: int | None, gc: GroupCount, n: int, c: int, ref: Reference | None, size_axis: Decimal
) -> GroupRow:
    size = pct(gc.births, n)
    rate = pct(gc.cesareans, gc.births)
    rr = ref.rows.get(group) if (ref is not None and group is not None) else None
    ref_size = rr.size_pct if rr else None
    ref_rate = rr.cs_rate_pct if rr else None
    size_diff = (
        (size - ref_size).quantize(TENTH) if size is not None and ref_size is not None else None
    )
    if size is None and ref_size is not None:
        size_diff = None
    rate_diff = (
        (rate - ref_rate).quantize(TENTH) if rate is not None and ref_rate is not None else None
    )
    return GroupRow(
        group=group,
        births=gc.births,
        cesareans=gc.cesareans,
        size_pct=size,
        rate_pct=rate,
        share_of_cs_pct=pct(gc.cesareans, c),
        contribution_pct=pct(gc.cesareans, n),
        ref_size=ref_size,
        ref_rate=ref_rate,
        size_diff=size_diff,
        rate_diff=rate_diff,
        larger_group=size_diff is not None and size_diff >= SIZE_FLAG_POINTS,
        higher_rate=(
            rate_diff is not None
            and gc.births >= MIN_BIRTHS_FOR_RATE_FLAG
            and rate_diff >= RATE_FLAG_POINTS
        ),
        size_w=_axis_pos(size, size_axis),
        size_t=_axis_pos(ref_size, size_axis),
        rate_w=_axis_pos(rate, Decimal(100)),
        rate_t=_axis_pos(ref_rate, Decimal(100)),
    )


def build_report(counts: Counts, reference: Reference | None, awaiting: int = 0) -> Report:
    n = sum(gc.births for gc in counts.values())
    c = sum(gc.cesareans for gc in counts.values())
    sizes = [pct(counts.get(g, GroupCount()).births, n) or Decimal(0) for g in GROUPS]
    if reference is not None:
        sizes += [r.size_pct for r in reference.rows.values()]
    size_axis = Decimal(45) if max(sizes, default=Decimal(0)) <= 45 else Decimal(100)
    rows = [_row(g, counts.get(g, GroupCount()), n, c, reference, size_axis) for g in GROUPS]
    residual = _row(None, counts.get(None, GroupCount()), n, c, None, size_axis)
    biggest = max(rows, key=lambda r: (r.cesareans, -r.group or 0), default=None)
    has_largest = biggest is not None and biggest.cesareans > 0
    return Report(
        births=n,
        cesareans=c,
        cs_rate=pct(c, n),
        rows=rows,
        residual=residual,
        unclassified_pct=pct(residual.births, n),
        largest_group=biggest.group if has_largest and biggest else None,
        largest_share=biggest.share_of_cs_pct if has_largest and biggest else None,
        reference=reference,
        awaiting=awaiting,
    )


@dataclass(frozen=True)
class Comparison:
    """Observed vs expected cesarean rate for a case mix, on resolved births only."""

    resolved: int
    observed: Decimal | None
    expected: Decimal | None
    ratio: Decimal | None  # observed / expected, 2 decimals

    @property
    def diff(self) -> Decimal | None:
        if self.observed is None or self.expected is None:
            return None
        return (self.observed - self.expected).quantize(TENTH)


def compare_to_expected(counts: Counts, reference: Reference | None) -> Comparison:
    resolved = sum(gc.births for g, gc in counts.items() if g is not None)
    cs = sum(gc.cesareans for g, gc in counts.items() if g is not None)
    observed = (Decimal(cs) * 100 / resolved) if resolved else None
    expected: Decimal | None = None
    if reference is not None and resolved:
        expected = sum(
            (
                Decimal(counts[g].births) * reference.rows[g].cs_rate_pct / resolved
                for g in GROUPS
                if g in counts
            ),
            Decimal(0),
        )
    ratio = (
        (observed / expected).quantize(Decimal("0.01"), ROUND_HALF_UP)
        if observed is not None and expected
        else None
    )
    return Comparison(
        resolved,
        observed.quantize(TENTH, ROUND_HALF_UP) if observed is not None else None,
        expected.quantize(TENTH, ROUND_HALF_UP) if expected is not None else None,
        ratio,
    )
