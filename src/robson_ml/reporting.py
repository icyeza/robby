"""Writers for aggregate outputs under reports/ (always small-cell suppressed)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_scalar

from robson_ml.privacy import assert_no_small_cells, suppress_small_cells


def write_table(
    df: pd.DataFrame,
    path: Path,
    count_columns: Sequence[str],
    linked: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Suppress small cells, verify, and write ``df`` as CSV. Returns what was written.

    Raises:
        ValueError: if ``count_columns`` is empty -- a table with no declared count
            columns cannot be checked for small cells at all. Any ``linked`` column that
            itself holds a count (rather than a derived rate or label) must also be listed
            in ``count_columns`` so it is suppressed and verified in its own right.
    """
    if not count_columns:
        raise ValueError("count_columns must not be empty")
    safe = suppress_small_cells(df, count_columns, linked)
    assert_no_small_cells(safe, count_columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe.to_csv(path, index=False)
    return safe


def _fmt(value: object) -> str:
    # pd.isna only accepts a scalar; a non-scalar (e.g. a list cell) is never itself
    # treated as missing here, so is_scalar guards the call rather than the result.
    if is_scalar(value):
        scalar: Any = value
        if pd.isna(scalar):
            return ""
    if isinstance(value, (float, np.floating)):
        return f"{value:.3f}"
    return str(value)


def markdown_table(df: pd.DataFrame) -> str:
    """Render a (suppressed) aggregate table as GitHub markdown."""
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *body])
