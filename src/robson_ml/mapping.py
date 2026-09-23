"""Map raw export columns to the canonical schema (spec §5) via configs/mapping_ur_cmhs.yaml.

Nothing is guessed: a raw value that cannot be converted, a category label absent from
``levels`` and an out-of-range number all become missing *and are counted* in the report.
Fields whose meaning is uncertain carry ``status: review`` and are listed for human decision.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from robson_ml.privacy import level_counts
from robson_ml.schema import (
    CANONICAL_BASE_COLUMNS,
    CANONICAL_DTYPES,
    OMISSION_REASON_PATTERN,
    empty_column,
)

KINDS = frozenset(
    {"row_key", "integer", "float", "category", "datetime", "gestational_age", "text"}
)
STATUSES = frozenset({"confirmed", "review"})
GA_FORMATS = frozenset({"decimal_weeks", "weeks_plus_days", "completed_weeks"})
LOCAL_TIMEZONE = "Africa/Kigali"
ROW_KEY_PREFIX = "ADM"
_WEEKS_PLUS_DAYS = re.compile(r"^\s*(\d{1,2})\s*\+\s*([0-6])\s*$")


class MappingError(ValueError):
    """The mapping configuration is invalid."""


@dataclass(frozen=True)
class FieldMapping:
    """How one canonical field is produced from raw column(s)."""

    canonical: str
    kind: str
    raw: tuple[str, ...]
    status: str
    note: str = ""
    levels: dict[str, Any] = field(default_factory=dict)
    dtype: str | None = None
    valid_range: tuple[float, float] | None = None
    ga_format: str | None = None
    datetime_format: str | None = None


@dataclass(frozen=True)
class MappingConfig:
    """A parsed mapping file."""

    source: str
    sheet: str | None
    fields: dict[str, FieldMapping]


@dataclass
class FieldReport:
    """Aggregate account of how one canonical field was mapped."""

    canonical: str
    raw: list[str]
    kind: str
    status: str
    note: str
    n_raw_nonnull: int = 0
    n_mapped: int = 0
    n_unparsed: int = 0
    n_out_of_range: int = 0
    unmapped_levels: dict[str, Any] = field(default_factory=dict)


@dataclass
class MappingReport:
    """Aggregate account of the whole mapping (no row-level values)."""

    fields: list[FieldReport]
    missing_canonical_fields: list[str]
    unreferenced_raw_columns: list[str]
    review_fields: list[str]


def _known_field(name: str) -> bool:
    return name in CANONICAL_DTYPES or bool(OMISSION_REASON_PATTERN.match(name))


def _parse_field(name: str, spec: dict[str, Any]) -> FieldMapping:
    if not _known_field(name):
        raise MappingError(f"unknown canonical field: {name}")
    kind = spec.get("kind")
    if kind not in KINDS:
        raise MappingError(f"{name}: kind must be one of {sorted(KINDS)}")
    status = spec.get("status", "review")
    if status not in STATUSES:
        raise MappingError(f"{name}: status must be confirmed or review")
    raw = spec.get("raw")
    raw_cols = () if raw is None else (str(raw),) if isinstance(raw, str) else tuple(map(str, raw))
    if kind != "row_key" and not raw_cols:
        raise MappingError(f"{name}: a raw column is required")
    levels = spec.get("levels") or {}
    if kind == "category" and not levels:
        raise MappingError(f"{name}: category fields need levels")
    for key, value in levels.items():
        if isinstance(key, bool) or isinstance(value, bool):
            raise MappingError(f"{name}: quote yes/no/true/false labels and values in YAML")
    bounds = spec.get("range")
    ga_format = spec.get("format") if kind == "gestational_age" else None
    if kind == "gestational_age" and ga_format not in GA_FORMATS:
        raise MappingError(f"{name}: gestational_age needs format in {sorted(GA_FORMATS)}")
    return FieldMapping(
        canonical=name,
        kind=str(kind),
        raw=raw_cols,
        status=str(status),
        note=str(spec.get("note", "")),
        levels={str(k).strip(): v for k, v in levels.items()},
        dtype=spec.get("dtype"),
        valid_range=(float(bounds[0]), float(bounds[1])) if bounds else None,
        ga_format=ga_format,
        datetime_format=str(spec.get("format", "ISO8601")) if kind == "datetime" else None,
    )


def load_mapping(path: Path) -> MappingConfig:
    """Parse and validate a mapping YAML file."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_fields = data.get("fields") or {}
    fields = {str(name): _parse_field(str(name), spec) for name, spec in raw_fields.items()}
    sheet = data.get("sheet")
    return MappingConfig(str(data.get("source", "")), None if sheet is None else str(sheet), fields)


def _blank_to_none(value: object) -> object:
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _label(value: object) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _apply_range(values: pd.Series, bounds: tuple[float, float] | None) -> tuple[pd.Series, int]:
    if bounds is None:
        return values, 0
    outside = values.notna() & ((values < bounds[0]) | (values > bounds[1]))
    return values.mask(outside), int(outside.sum())


def _parse_weeks_plus_days(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float | np.integer | np.floating) and not isinstance(value, bool):
        return None if math.isnan(float(value)) else float(value)
    text = str(value)
    match = _WEEKS_PLUS_DAYS.match(text)
    if match:
        return int(match.group(1)) + int(match.group(2)) / 7.0
    try:
        return float(text)
    except ValueError:
        return None


def _map_one(raw: pd.DataFrame, spec: FieldMapping, report: FieldReport) -> pd.Series:
    n = len(raw)
    if spec.kind == "row_key":
        report.n_mapped = n
        return pd.Series([f"{ROW_KEY_PREFIX}{i:06d}" for i in range(n)], dtype=object)
    columns = [raw[c].map(_blank_to_none) for c in spec.raw]
    source = columns[0]
    present = pd.concat(columns, axis=1).notna().any(axis=1)
    report.n_raw_nonnull = int(present.sum())

    result: pd.Series
    if spec.kind == "text":
        result = source.map(_label).astype(object)
    elif spec.kind in ("integer", "float"):
        numeric = pd.to_numeric(source, errors="coerce").astype("float64")
        unparsed = present & numeric.isna()
        if spec.kind == "integer":
            fractional = numeric.notna() & (numeric % 1 != 0)
            unparsed |= fractional
            numeric = numeric.mask(fractional)
        report.n_unparsed = int(unparsed.sum())
        numeric, report.n_out_of_range = _apply_range(numeric, spec.valid_range)
        result = numeric.astype("Int64") if spec.kind == "integer" else numeric
    elif spec.kind == "category":
        labels = source.map(_label)
        known = labels.isin(list(spec.levels))
        unmapped = labels.notna() & ~known
        report.n_unparsed = int(unmapped.sum())
        report.unmapped_levels = dict(level_counts(labels[unmapped]).values.tolist())
        mapped = labels.map(lambda v: spec.levels.get(v) if v is not None else None)
        if spec.dtype:
            try:
                numeric_levels = pd.to_numeric(mapped)
                result = numeric_levels.astype(spec.dtype)  # type: ignore[call-overload]
            except (TypeError, ValueError) as exc:
                raise MappingError(
                    f"{spec.canonical}: non-numeric level value(s) for dtype {spec.dtype}"
                ) from exc
        else:
            result = mapped.astype(object)
    elif spec.kind == "datetime":
        try:
            parsed = pd.to_datetime(source, errors="coerce", format=spec.datetime_format)
        except (ValueError, TypeError):
            parsed = pd.to_datetime(source, errors="coerce", format="mixed")
        if parsed.dtype == object:
            parsed = pd.to_datetime(source, errors="coerce", format="mixed")
        if getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_convert(LOCAL_TIMEZONE).dt.tz_localize(None)
        report.n_unparsed = int((present & parsed.isna()).sum())
        result = parsed.astype("datetime64[ns]")
    else:  # gestational_age
        if spec.ga_format == "weeks_plus_days" and len(columns) == 2:
            weeks = pd.to_numeric(columns[0], errors="coerce")
            days = pd.to_numeric(columns[1], errors="coerce").fillna(0)
            ga = weeks + days.where(days.between(0, 6)) / 7.0
        elif spec.ga_format == "weeks_plus_days":
            ga = source.map(_parse_weeks_plus_days).astype("float64")
        else:
            ga = pd.to_numeric(source, errors="coerce").astype("float64")
        report.n_unparsed = int((present & ga.isna()).sum())
        result, report.n_out_of_range = _apply_range(ga, spec.valid_range)
    report.n_mapped = int(result.notna().sum())
    return result.reset_index(drop=True)


def apply_mapping(raw: pd.DataFrame, config: MappingConfig) -> tuple[pd.DataFrame, MappingReport]:
    """Build the canonical frame from one raw sheet, with an aggregate mapping report."""
    raw = raw.reset_index(drop=True)
    for spec in config.fields.values():
        absent = [c for c in spec.raw if c not in raw.columns]
        if absent:
            raise MappingError(f"{spec.canonical}: raw column(s) not in sheet: {absent}")
    columns: dict[str, pd.Series] = {}
    reports: list[FieldReport] = []
    missing: list[str] = []
    for name in CANONICAL_BASE_COLUMNS:
        if name not in config.fields:
            columns[name] = empty_column(name, len(raw))
            missing.append(name)
            continue
        spec = config.fields[name]
        report = FieldReport(name, list(spec.raw), spec.kind, spec.status, spec.note)
        columns[name] = _map_one(raw, spec, report)
        reports.append(report)
    for name in sorted(n for n in config.fields if OMISSION_REASON_PATTERN.match(n)):
        spec = config.fields[name]
        report = FieldReport(name, list(spec.raw), spec.kind, spec.status, spec.note)
        columns[name] = _map_one(raw, spec, report)
        reports.append(report)
    referenced = {c for spec in config.fields.values() for c in spec.raw}
    canonical = pd.DataFrame(columns)
    return canonical, MappingReport(
        fields=reports,
        missing_canonical_fields=missing,
        unreferenced_raw_columns=[str(c) for c in raw.columns if str(c) not in referenced],
        review_fields=[f.canonical for f in config.fields.values() if f.status == "review"],
    )
