"""Small-cell suppression and aggregate-only inspection helpers (spec §3)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

SMALL_CELL_THRESHOLD = 5
SUPPRESSED = "<5"
MAX_LEVELS_SHOWN = 30


def _is_small(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return (numeric > 0) & (numeric < SMALL_CELL_THRESHOLD)


def suppress_small_cells(
    df: pd.DataFrame,
    count_columns: Sequence[str],
    linked: Mapping[str, Sequence[str]] | None = None,
    complements: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Replace counts of 1-4 with ``"<5"``, along with the columns derived from them.

    Args:
        df: an aggregate table.
        count_columns: columns holding counts of records.
        linked: for a count column, the derived columns (rates, percentages, other counts)
            that would reveal it and are suppressed on the same rows.
        complements: for a count column, the column holding the total it was drawn from.
            A row is also suppressed for that count column (and its linked columns) when
            the complement (total - count) is 1-4, since revealing the count would then
            reveal the complement.
    Returns:
        A copy; every touched column becomes object dtype. Zero counts are kept (they
        disclose no individual; DECISIONS.md 2026-09-23).
    """
    out = df.copy()
    links = dict(linked or {})
    comps = dict(complements or {})
    for col in count_columns:
        small = _is_small(df[col])
        if col in comps:
            total = pd.to_numeric(df[comps[col]], errors="coerce")
            count = pd.to_numeric(df[col], errors="coerce")
            complement = total - count
            small = small | ((complement > 0) & (complement < SMALL_CELL_THRESHOLD))
        for target in (col, *links.get(col, ())):
            out[target] = out[target].astype(object)
            out.loc[small, target] = SUPPRESSED
    return out


def assert_no_small_cells(df: pd.DataFrame, count_columns: Sequence[str]) -> None:
    """Raise if any count column still holds a count of 1-4."""
    for col in count_columns:
        if _is_small(df[col]).any():
            raise AssertionError(f"column {col} holds a count below {SMALL_CELL_THRESHOLD}")


def fmt_count(n: int) -> str:
    """Format a single count for report text, suppressing 1-4."""
    return SUPPRESSED if 0 < n < SMALL_CELL_THRESHOLD else str(n)


def safe_pct(mask: pd.Series) -> float | str:
    """Percentage of ``True`` in ``mask``; suppressed when either side counts 1-4.

    ``NA`` entries are dropped before computing, so a nullable-boolean mask with
    unresolved rows is judged only on its resolved rows.
    """
    mask = mask.dropna()
    n = len(mask)
    n_true = int(mask.sum())
    if n == 0:
        return float("nan")
    if 0 < n_true < SMALL_CELL_THRESHOLD or 0 < n - n_true < SMALL_CELL_THRESHOLD:
        return SUPPRESSED
    return round(100.0 * n_true / n, 1)


def safe_describe(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column dtype, non-null count, % missing and number of distinct values. No values.

    ``n_nonnull`` is suppressed (spec §3.3) whenever it itself is 1-4, and also when its
    complement ``n_missing`` is 1-4 (revealing a near-complete or near-empty column would
    otherwise disclose the complement's small count). ``pct_missing`` and ``n_unique`` are
    suppressed on the same rows, since either would reveal the same small count.
    """
    n = len(df)
    rows = []
    for col in df.columns:
        s = df[col]
        nonnull = int(s.notna().sum())
        missing = n - nonnull
        suppress = 0 < nonnull < SMALL_CELL_THRESHOLD or 0 < missing < SMALL_CELL_THRESHOLD
        rows.append(
            {
                "column": str(col),
                "dtype": str(s.dtype),
                "n_nonnull": SUPPRESSED if suppress else nonnull,
                "pct_missing": (
                    SUPPRESSED
                    if suppress
                    else (round(100.0 * missing / n, 1) if n else float("nan"))
                ),
                "n_unique": SUPPRESSED if suppress else int(s.dropna().astype(str).nunique()),
            }
        )
    return pd.DataFrame(rows, columns=["column", "dtype", "n_nonnull", "pct_missing", "n_unique"])


def level_counts(series: pd.Series, max_levels: int = MAX_LEVELS_SHOWN) -> pd.DataFrame:
    """Label counts for a low-cardinality column, safe to display.

    Columns with more than ``max_levels`` distinct values are treated as free text: only
    the number of distinct values is returned. Labels seen fewer than 5 times are pooled
    and not shown. A column is also treated as free text when the values belonging to
    rare (<5-occurrence) levels make up more than half of its non-null values -- a sign
    that most values are unique-ish (e.g. names) rather than a small label set, even when
    the raw number of distinct values doesn't exceed ``max_levels``.
    """
    values = series.dropna().astype(str).str.strip()
    counts = values.value_counts()
    rare = counts[counts < SMALL_CELL_THRESHOLD]
    is_free_text = len(counts) > max_levels or (
        len(values) > 0 and int(rare.sum()) > len(values) / 2
    )
    if is_free_text:
        label = f"(free text: {len(counts)} distinct values, not shown)"
        return pd.DataFrame({"level": [label], "n": [fmt_count(len(values))]})
    rows: list[dict[str, object]] = [
        {"level": str(k), "n": int(v)} for k, v in counts.items() if v >= SMALL_CELL_THRESHOLD
    ]
    if len(rare):
        rows.append({"level": f"(other: {len(rare)} rare levels)", "n": fmt_count(int(rare.sum()))})
    rows.append({"level": "(missing)", "n": fmt_count(int(series.isna().sum()))})
    return pd.DataFrame(rows, columns=["level", "n"])
