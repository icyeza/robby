"""Read the raw export (every sheet, values untouched) and build an aggregate inventory."""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

MAX_CATEGORY_LEVELS = 30
PARSE_SHARE = 0.9
VERSION_COLUMN = "__version__"


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_workbook(path: Path) -> dict[str, pd.DataFrame]:
    """Read every sheet of the raw export as object columns (no type coercion)."""
    if not path.exists():
        raise FileNotFoundError(f"raw export not found at {path}; expected under data/raw/")
    sheets: dict[str, pd.DataFrame] = pd.read_excel(
        path, sheet_name=None, dtype=object, engine="openpyxl"
    )
    return sheets


def select_sheet(sheets: dict[str, pd.DataFrame], name: str | None) -> pd.DataFrame:
    """Return the named sheet, or the only sheet when ``name`` is None."""
    if name is None:
        if len(sheets) != 1:
            raise ValueError(
                f"workbook has several sheets ({len(sheets)}); name one in the mapping"
            )
        return next(iter(sheets.values()))
    if name not in sheets:
        raise ValueError(f"sheet {name!r} not found in workbook")
    return sheets[name]


def infer_kind(series: pd.Series) -> str:
    """Classify a raw column as empty, datetime, numeric, categorical or text."""
    values = series.dropna()
    if values.empty:
        return "empty"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    is_datetime = values.map(lambda v: isinstance(v, dt.datetime | dt.date))
    if is_datetime.mean() >= PARSE_SHARE:
        return "datetime"
    if pd.to_numeric(values, errors="coerce").notna().mean() >= PARSE_SHARE:
        return "numeric"
    if values.astype(str).nunique() <= MAX_CATEGORY_LEVELS:
        return "categorical"
    return "text"


def _column_entry(series: pd.Series) -> dict[str, Any]:
    return {
        "name": str(series.name),
        "kind": infer_kind(series),
        "n_nonnull": int(series.notna().sum()),
        "n_unique": int(series.dropna().astype(str).nunique()),
    }


def inventory(sheets: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Aggregate description of the workbook: sheets, columns, kinds, counts, form versions.

    Contains column names and counts only, never cell values (other than form-version ids).
    """
    entries = []
    for name, df in sheets.items():
        entry: dict[str, Any] = {
            "sheet": str(name),
            "n_rows": len(df),
            "n_cols": df.shape[1],
            "columns": [_column_entry(df.iloc[:, i]) for i in range(df.shape[1])],
        }
        if VERSION_COLUMN in df.columns:
            version_series = df[VERSION_COLUMN]
            if isinstance(version_series, pd.DataFrame):
                version_series = version_series.iloc[:, 0]
            entry["versions"] = {
                str(version): {
                    "n_rows": len(sub),
                    "nonnull_by_column": {
                        str(df.columns[i]): int(sub.iloc[:, i].notna().sum())
                        for i in range(sub.shape[1])
                    },
                }
                for version, sub in df.groupby(version_series, dropna=False)
            }
        entries.append(entry)
    return {"sheets": entries}
