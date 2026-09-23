"""Writers for aggregate outputs under reports/ (always small-cell suppressed)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from robson_ml.privacy import assert_no_small_cells, suppress_small_cells


def write_table(
    df: pd.DataFrame,
    path: Path,
    count_columns: Sequence[str],
    linked: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Suppress small cells, verify, and write ``df`` as CSV. Returns what was written."""
    safe = suppress_small_cells(df, count_columns, linked)
    assert_no_small_cells(safe, count_columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe.to_csv(path, index=False)
    return safe


def _fmt(value: object) -> str:
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.3f}"
    return str(value)


def markdown_table(df: pd.DataFrame) -> str:
    """Render a (suppressed) aggregate table as GitHub markdown."""
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *body])
