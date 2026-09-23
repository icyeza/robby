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
) -> pd.DataFrame:
    """Replace counts of 1-4 with ``"<5"``, along with the columns derived from them.

    Args:
        df: an aggregate table.
        count_columns: columns holding counts of records.
        linked: for a count column, the derived columns (rates, percentages, other counts)
            that would reveal it and are suppressed on the same rows.
    Returns:
        A copy; every touched column becomes object dtype. Zero counts are kept (they
        disclose no individual; DECISIONS.md 2026-09-23).
    """
    out = df.copy()
    links = dict(linked or {})
    for col in count_columns:
        small = _is_small(df[col])
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
    """Percentage of ``True`` in ``mask``; suppressed when either side counts 1-4."""
    n = len(mask)
    n_true = int(mask.sum())
    if n == 0:
        return float("nan")
    if 0 < n_true < SMALL_CELL_THRESHOLD or 0 < n - n_true < SMALL_CELL_THRESHOLD:
        return SUPPRESSED
    return round(100.0 * n_true / n, 1)


def safe_describe(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column dtype, non-null count, % missing and number of distinct values. No values."""
    n = len(df)
    rows = []
    for col in df.columns:
        s = df[col]
        nonnull = int(s.notna().sum())
        rows.append(
            {
                "column": str(col),
                "dtype": str(s.dtype),
                "n_nonnull": nonnull,
                "pct_missing": round(100.0 * (1 - nonnull / n), 1) if n else float("nan"),
                "n_unique": int(s.dropna().astype(str).nunique()),
            }
        )
    return pd.DataFrame(rows, columns=["column", "dtype", "n_nonnull", "pct_missing", "n_unique"])


def level_counts(series: pd.Series, max_levels: int = MAX_LEVELS_SHOWN) -> pd.DataFrame:
    """Label counts for a low-cardinality column, safe to display.

    Columns with more than ``max_levels`` distinct values are treated as free text: only
    the number of distinct values is returned. Labels seen fewer than 5 times are pooled
    and not shown.
    """
    values = series.dropna().astype(str).str.strip()
    counts = values.value_counts()
    if len(counts) > max_levels:
        label = f"(free text: {len(counts)} distinct values, not shown)"
        return pd.DataFrame({"level": [label], "n": [fmt_count(len(values))]})
    rows: list[dict[str, object]] = [
        {"level": str(k), "n": int(v)} for k, v in counts.items() if v >= SMALL_CELL_THRESHOLD
    ]
    rare = counts[counts < SMALL_CELL_THRESHOLD]
    if len(rare):
        rows.append({"level": f"(other: {len(rare)} rare levels)", "n": fmt_count(int(rare.sum()))})
    rows.append({"level": "(missing)", "n": fmt_count(int(series.isna().sum()))})
    return pd.DataFrame(rows, columns=["level", "n"])
