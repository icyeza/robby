"""Aggregate exploratory-data-analysis helpers with small-cell protection (spec §3, §7).

Every function returns an aggregate table safe to display: counts of 1-4 never appear.
Histograms merge sparse bins instead of hiding them (so nothing is recoverable by
subtraction from the published total), bin edges come from a fixed domain grid or from
released quantiles (never from an observed minimum or maximum, which would be one woman's
value), rate tables go through :func:`robson_ml.privacy.suppress_table` (primary and
secondary suppression), correlations need a minimum number of complete pairs and
calibration curves a minimum number of rows per bin. Nothing here returns or prints a row.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from robson_ml.privacy import (
    SECONDARY,
    SMALL_CELL_THRESHOLD,
    SUPPRESSED,
    DerivedCell,
    SumRelation,
    TableSpec,
    suppress_table,
    suppress_tables,
)
from robson_ml.profile import OVERALL_SCOPE, pct_by_facility, released_quantiles

HIDDEN = (SUPPRESSED, SECONDARY)
MISSING_LEVEL = "(missing)"
RARE_LEVEL = "(rare)"
DEFAULT_HIST_BINS = 20
# Pairwise-complete rows a correlation needs before it is shown (an aggregate over many
# women, never a relation between a handful of values).
MIN_CORRELATION_N = 50
# Rows (and events and non-events) a calibration bin needs before its mean is shown.
MIN_CALIBRATION_BIN = 30
CALIBRATION_BINS = 10
# Fixed domain grids (lower edge, upper edge, width) for the canonical numeric fields; the
# values outside fall into open tail bins. Other fields use released quantiles.
DOMAIN_GRIDS: Mapping[str, tuple[float, float, float]] = {
    "maternal_age": (12.0, 51.0, 3.0),
    "gestational_age_weeks": (24.0, 44.0, 1.0),
    "ga_band_lower": (20.0, 45.0, 1.0),
    "ga_band_upper": (20.0, 45.0, 1.0),
    "parity": (0.0, 10.0, 1.0),
    "previous_cs_count": (0.0, 6.0, 1.0),
    "plurality": (1.0, 4.0, 1.0),
    "anc_contacts": (0.0, 13.0, 1.0),
    "height_cm": (120.0, 200.0, 5.0),
    "weight_kg": (30.0, 150.0, 5.0),
    "bmi": (12.0, 52.0, 2.0),
}


def _is_small(value: float) -> bool:
    return 0 < value < SMALL_CELL_THRESHOLD


def _nice_step(span: float, n_bins: int) -> float:
    """A 1/2/5 x 10^k step giving about ``n_bins`` bins over ``span``."""
    raw = span / max(n_bins, 1)
    if raw <= 0:
        return 1.0
    power = 10 ** math.floor(math.log10(raw))
    for factor in (1, 2, 5, 10):
        if raw <= factor * power:
            return float(factor * power)
    return float(10 * power)


def histogram_edges(
    values: pd.Series, name: str | None = None, n_bins: int = DEFAULT_HIST_BINS
) -> npt.NDArray[np.float64]:
    """Bin edges for a numeric variable that never sit on an observed extreme.

    A field in :data:`DOMAIN_GRIDS` uses its fixed grid. Otherwise the edges span the
    released 5th-95th percentiles (each released only with at least 5 values on either
    side; :func:`robson_ml.profile.released_quantiles`), rounded outward to a 1/2/5 step.
    Values outside the edges fall into open tail bins (:func:`binned_hist_counts`). An
    empty array means no edges can be released (too few values).
    """
    if name is not None and name in DOMAIN_GRIDS:
        low, high, width = DOMAIN_GRIDS[name]
        return np.arange(low, high + width / 2, width, dtype=np.float64)
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    quantiles = released_quantiles(numeric)
    q_low, q_high = quantiles.get("p5"), quantiles.get("p95")
    if q_low is None or q_high is None:
        return np.array([], dtype=np.float64)
    if q_high <= q_low:
        return np.array([q_low - 0.5, q_low + 0.5], dtype=np.float64)
    step = _nice_step(q_high - q_low, n_bins)
    start, stop = math.floor(q_low / step) * step, math.ceil(q_high / step) * step
    return np.arange(start, stop + step / 2, step, dtype=np.float64)


def _fmt_edge(value: float) -> str:
    return f"{value:g}"


def _bin_label(left: float, right: float, integer: bool = False) -> str:
    """``[a, b)``; for whole-number values ``a`` (one value) or ``a-b`` (inclusive)."""
    if left == -math.inf:
        return f"<{_fmt_edge(right)}"
    if right == math.inf:
        return f">={_fmt_edge(left)}"
    if integer:
        last = right - 1
        return _fmt_edge(left) if last == left else f"{_fmt_edge(left)}-{_fmt_edge(last)}"
    return f"[{_fmt_edge(left)}, {_fmt_edge(right)})"


def binned_hist_counts(
    values: pd.Series,
    edges: npt.ArrayLike | None = None,
    by: pd.Series | None = None,
    name: str | None = None,
) -> pd.DataFrame:
    """Counts of ``values`` in bins, split by ``by`` (e.g. the outcome), with no count of 1-4.

    Bins are ``[edge_i, edge_i+1)`` plus open tails below the first and above the last edge
    (default edges: :func:`histogram_edges`). A bin holding 1-4 rows in any ``by`` level is
    merged with its smaller non-empty neighbour until every cell is 0 or at least 5; leading and
    trailing empty bins are dropped. Missing values are not counted (missingness is shown
    separately). Merging, not hiding, keeps every published cell recoverable only as
    itself: the bins still sum to the non-missing count.

    Whole-number variables on a whole-number grid get value labels (``3``, ``5-7``).

    Returns columns ``bin`` (label), ``left``, ``right`` and one count column per ``by``
    level (``n`` when ``by`` is None), plus ``total``. Empty when the counts cannot be
    shown (fewer than 5 non-missing values in some level).
    """
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    grid = np.asarray(histogram_edges(numeric, name) if edges is None else edges, dtype=np.float64)
    if len(grid) < 2:
        return pd.DataFrame(columns=["bin", "left", "right", "total"])
    bounds = np.concatenate([[-math.inf], grid, [math.inf]])
    present = numeric.notna()
    levels = ["n"] if by is None else sorted(by[present].astype(str).unique())
    group = pd.Series("n", index=numeric.index) if by is None else by.astype(str)
    counts = np.zeros((len(bounds) - 1, len(levels)), dtype=np.int64)
    positions = np.searchsorted(bounds, numeric[present].to_numpy(), side="right") - 1
    level_index = {level: i for i, level in enumerate(levels)}
    for position, level in zip(positions, group[present].to_numpy(), strict=True):
        counts[position, level_index[str(level)]] += 1
    lefts, rights = list(bounds[:-1]), list(bounds[1:])
    rows = [list(row) for row in counts]

    def unsafe(row: list[int]) -> bool:
        return any(_is_small(v) for v in row)

    while True:
        bad = [i for i, row in enumerate(rows) if unsafe(row)]
        if not bad:
            break
        if len(rows) == 1:
            return pd.DataFrame(columns=["bin", "left", "right", *levels, "total"])
        i = bad[0]
        # Merge into the smaller non-empty neighbour (an empty one would not help).
        neighbours = [k for k in (i - 1, i + 1) if 0 <= k < len(rows)]
        filled = [k for k in neighbours if sum(rows[k])] or neighbours
        j = min(filled, key=lambda k: (sum(rows[k]), k))
        a, b = min(i, j), max(i, j)
        rows[a] = [x + y for x, y in zip(rows[a], rows[b], strict=True)]
        rights[a] = rights[b]
        del rows[b], lefts[b], rights[b]
    whole = numeric[present].to_numpy()
    integer = bool(np.all(np.mod(whole, 1) == 0) and np.all(np.mod(grid, 1) == 0))
    nonzero = [i for i, row in enumerate(rows) if sum(row)]
    if not nonzero:
        return pd.DataFrame(columns=["bin", "left", "right", *levels, "total"])
    keep = range(nonzero[0], nonzero[-1] + 1)
    table = pd.DataFrame(
        {
            "bin": [_bin_label(lefts[i], rights[i], integer) for i in keep],
            "left": [lefts[i] for i in keep],
            "right": [rights[i] for i in keep],
            **{level: [rows[i][k] for i in keep] for k, level in enumerate(levels)},
        }
    )
    table["total"] = table[levels].sum(axis=1)
    return table


def _labels(series: pd.Series) -> pd.Series:
    return series.astype(object).where(series.notna(), MISSING_LEVEL).astype(str)


def rate_by(frame: pd.DataFrame, by: str | Sequence[str], outcome: str = "cs") -> pd.DataFrame:
    """Rows (``n``), outcome rows (``n_outcome``) and the outcome rate (%) per level of
    ``by`` (one column, or two for a cross-table), suppressed.

    ``outcome`` is a 0/1 (or boolean) column; rows where it is missing are left out. Missing
    levels of ``by`` form the level ``"(missing)"``. Suppression: counts (or the complement
    ``n - n_outcome``) of 1-4 are ``"<5"`` with the rate; secondary suppression (``"*"``)
    protects them from subtraction within the table total (one ``by`` column) or within
    each level of either column (a cross-table: each row and column block sums to a total
    published elsewhere).
    """
    keys = [by] if isinstance(by, str) else list(by)
    if not 1 <= len(keys) <= 2:
        raise ValueError("rate_by takes one or two grouping columns")
    known = frame[outcome].notna()
    sub = pd.DataFrame({k: _labels(frame.loc[known, k]) for k in keys})
    sub["_y"] = frame.loc[known, outcome].astype(float).to_numpy()
    grouped = sub.groupby(keys, sort=True)["_y"].agg(["count", "sum"]).reset_index()
    table = grouped.rename(columns={"count": "n", "sum": "n_outcome"})
    table["n"] = table["n"].astype(int)
    table["n_outcome"] = table["n_outcome"].astype(int)
    table["rate_pct"] = (100.0 * table["n_outcome"] / table["n"]).round(1)
    groups = None
    if len(keys) == 2:
        groups = [
            SumRelation(tuple(block.index))
            for key in keys
            for _, block in table.groupby(key, sort=False)
            if len(block) >= 2
        ]
    return suppress_table(
        table,
        ["n", "n_outcome"],
        linked={"n": ["n_outcome", "rate_pct"], "n_outcome": ["rate_pct"]},
        complements={"n_outcome": "n"},
        groups=groups,
    )


def pool_rare_levels(series: pd.Series) -> pd.Series:
    """``series`` as labels, with levels seen fewer than 5 times pooled into ``"(rare)"``
    and missing values as ``"(missing)"``."""
    labels = _labels(series)
    sizes = labels.value_counts()
    rare = sizes.index[(sizes < SMALL_CELL_THRESHOLD) & (sizes.index != MISSING_LEVEL)]
    return labels.where(~labels.isin(rare), RARE_LEVEL)


def level_rates(series: pd.Series, outcome: pd.Series) -> pd.DataFrame:
    """Level counts and outcome rates of a categorical variable (rare levels pooled),
    suppressed as in :func:`rate_by`. ``series`` and ``outcome`` align by index."""
    frame = pd.DataFrame({"level": pool_rare_levels(series), "outcome": outcome})
    return rate_by(frame, "level", "outcome")


def missingness_matrix(
    frame: pd.DataFrame, columns: Sequence[str], facility: pd.Series, decimals: int = 0
) -> pd.DataFrame:
    """% missing per variable (rows) overall and per facility (columns), suppressed.

    Each row is :func:`robson_ml.profile.pct_by_facility` (primary, secondary and the
    overall rule); percentages are then rounded to ``decimals`` (default whole percent, as
    defence in depth against differencing with other published tables). Hidden cells hold
    the marker (``"<5"`` or ``"*"``).
    """
    labels = facility.astype(str)
    facilities = sorted(labels.unique())
    rows: dict[str, dict[str, Any]] = {}
    for column in columns:
        cells = pct_by_facility(frame[column].isna(), labels, facilities)
        rows[column] = {
            scope: (value if isinstance(value, str) else round(float(value), decimals))
            for scope, value in cells.items()
        }
    return pd.DataFrame.from_dict(rows, orient="index", columns=[OVERALL_SCOPE, *facilities])


def correlation_matrix(
    frame: pd.DataFrame,
    columns: Sequence[str],
    min_n: int = MIN_CORRELATION_N,
    method: str = "spearman",
) -> pd.DataFrame:
    """Pairwise correlations of numeric ``columns``; a pair with fewer than ``min_n``
    complete rows is NaN, and a column with fewer than ``min_n`` values, or constant, is
    left out. An aggregate: no value of any row can be read from it."""
    numeric = frame[list(columns)].apply(pd.to_numeric, errors="coerce").astype(float)
    keep = [c for c in numeric.columns if numeric[c].notna().sum() >= min_n]
    keep = [c for c in keep if numeric[c].nunique() > 1]
    numeric = numeric[keep]
    corr = numeric.corr(method=method, min_periods=min_n)
    present = numeric.notna().astype(int)
    pairs = present.T @ present
    return corr.where(pairs >= min_n)


def calibration_bins(
    y: npt.ArrayLike,
    p: npt.ArrayLike,
    n_bins: int = CALIBRATION_BINS,
    min_count: int = MIN_CALIBRATION_BIN,
) -> pd.DataFrame:
    """A binned calibration curve: mean predicted probability and observed rate per bin.

    Bins are quantiles of ``p``; a bin with fewer than ``min_count`` rows, or 1-4 events
    or non-events, is merged into its smaller neighbour until none is. Returns ``n``,
    ``mean_p`` and ``observed`` per bin (empty if even one bin over all rows fails).
    """
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    order = np.argsort(p_arr, kind="stable")
    y_sorted, p_sorted = y_arr[order], p_arr[order]
    splits = np.array_split(np.arange(len(p_sorted)), n_bins)
    bins = [list(chunk) for chunk in splits if len(chunk)]

    def unsafe(rows: list[int]) -> bool:
        events = float(y_sorted[rows].sum())
        return len(rows) < min_count or _is_small(events) or _is_small(len(rows) - events)

    while bins and any(unsafe(b) for b in bins):
        if len(bins) == 1:
            return pd.DataFrame(columns=["n", "mean_p", "observed"])
        i = next(k for k, b in enumerate(bins) if unsafe(b))
        if i == 0:
            j = 1
        elif i == len(bins) - 1:
            j = i - 1
        else:
            j = i - 1 if len(bins[i - 1]) <= len(bins[i + 1]) else i + 1
        a, b = min(i, j), max(i, j)
        bins[a] = bins[a] + bins[b]
        del bins[b]
    return pd.DataFrame(
        {
            "n": [len(b) for b in bins],
            "mean_p": [float(p_sorted[b].mean()) for b in bins],
            "observed": [float(y_sorted[b].mean()) for b in bins],
        }
    )


def prelabour_onset_share(audit: pd.DataFrame, period: str = "M") -> pd.DataFrame:
    """Among CS in ``P_audit``: the share coded with onset "pre-labour CS", by facility and
    delivery period (spec v1.3 evidence that onset is coded retrospectively), suppressed.

    ``period`` is a pandas period alias for ``delivery_date`` (``"M"``: month). Rows with a
    missing onset are left out of the denominator.
    """
    cs = audit[(audit["cs"] == 1) & audit["onset_of_labour"].notna()].copy()
    cs["period"] = cs["delivery_date"].dt.to_period(period).astype(str)
    cs["prelabour_onset"] = (cs["onset_of_labour"] == "prelabour_cs").astype(int)
    return rate_by(cs, ["facility_id", "period"], "prelabour_onset")


def nested_population_rates(
    populations: Mapping[str, pd.DataFrame], by: str = "facility_id", outcome: str = "cs"
) -> pd.DataFrame:
    """Rows, outcome rows and rate per level of ``by`` for nested populations, protected
    jointly.

    ``populations`` maps a label to a frame, outermost first, each a subset (by index) of
    the previous one (e.g. ``P_audit`` then ``P_pred``). Besides each table's own
    suppression, the difference between consecutive populations (the rows and outcome rows
    excluded per level) is a count a reader can form by subtraction; each such difference of
    1-4 is kept undetermined (:class:`robson_ml.privacy.DerivedCell`).

    Returns a long table: ``population``, ``by`` level, ``n``, ``n_outcome``, ``rate_pct``.
    """
    labels = list(populations)
    tables: dict[Hashable, TableSpec] = {}
    raw_tables: dict[str, pd.DataFrame] = {}
    for label, frame in populations.items():
        counts = (
            pd.DataFrame({"level": _labels(frame[by]), "_y": frame[outcome].astype(float)})
            .groupby("level", sort=True)["_y"]
            .agg(["count", "sum"])
            .rename(columns={"count": "n", "sum": "n_outcome"})
        )
        raw_tables[label] = counts
    levels = sorted(set().union(*(t.index for t in raw_tables.values())))
    for label in labels:
        counts = raw_tables[label].reindex(levels, fill_value=0).reset_index()
        counts["n"] = counts["n"].astype(int)
        counts["n_outcome"] = counts["n_outcome"].astype(int)
        counts["rate_pct"] = np.where(
            counts["n"] > 0, (100.0 * counts["n_outcome"] / counts["n"].clip(lower=1)), np.nan
        ).round(1)
        tables[label] = TableSpec(
            counts,
            ["n", "n_outcome"],
            linked={"n": ["n_outcome", "rate_pct"], "n_outcome": ["rate_pct"]},
            complements={"n_outcome": "n"},
        )
    cross: list[SumRelation] = []
    derived: dict[Any, DerivedCell] = {}
    for outer, inner in itertools.pairwise(labels):
        for row, _ in enumerate(levels):
            for column in ("n", "n_outcome"):
                outer_cell = (outer, row, column)
                inner_cell = (inner, row, column)
                gap = ("~excluded", outer, inner, row, column)
                value = float(tables[outer].df[column].iloc[row]) - float(
                    tables[inner].df[column].iloc[row]
                )
                derived[gap] = DerivedCell(value, inner_cell)
                cross.append(SumRelation((inner_cell, gap), outer_cell))
    safe = suppress_tables(tables, cross, derived)
    out = []
    for label in labels:
        table = safe[label].rename(columns={"level": by})
        table.insert(0, "population", label)
        out.append(table)
    return pd.concat(out, ignore_index=True)


def suppress_partition(table: pd.DataFrame, parts: Sequence[str]) -> pd.DataFrame:
    """Suppress a table whose ``parts`` columns split a known total in every row.

    For example a field's raw values split into mapped / unparsed / out-of-range /
    explicit-missing (their total, the non-missing count, is published elsewhere), or a
    fold's rows split into fit / calibration / excluded / test. A part of 1-4 is ``"<5"``;
    secondary suppression (``"*"``) then keeps it from being recovered as the total minus
    the other parts. Other columns are returned unchanged.
    """
    counts = table[list(parts)].astype(int).T
    counts.columns = pd.RangeIndex(counts.shape[1])
    safe = suppress_table(counts, list(counts.columns)).T
    out = table.copy()
    for part in parts:
        out[part] = safe[part].to_numpy()
    return out
