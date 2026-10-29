"""Feature registry loader: configs/features_v1.yaml.

Every raw or canonical variable considered for modelling is listed once, with a status
(include/exclude/review), a human reason, whether it is knowable at admission, and a
feature group. Model code must read only ``status: include`` features
(:meth:`FeatureRegistry.included`). The registry file's hash is exposed as
``FeatureRegistry.sha256`` so it can be logged with every run.

Reports and error messages never contain cell values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from robson_ml.ingest import file_sha256
from robson_ml.mapping import (
    FORMAT_KINDS,
    GA_FORMATS,
    KINDS,
    RANGE_KINDS,
    FieldMapping,
    FieldReport,
    map_field,
)
from robson_ml.mapping import _parse_integer_sum_levels as _integer_sum_levels
from robson_ml.mapping import _parse_levels as _levels
from robson_ml.mapping import _parse_range as _range

# Kinds only meaningful when built from a raw export column (row_key/hash_key are
# canonical-only and never apply to an arbitrary feature).
FEATURE_KINDS = KINDS - {"row_key", "hash_key"}
SOURCES = frozenset({"canonical", "raw", "derived"})
STATUSES = frozenset({"include", "exclude", "review"})
GROUPS = frozenset(
    {
        "G_robson",
        "G_maternal",
        "G_obs",
        "G_missing",
        "G_context",
        "outcome",
        "identifier",
        "admin",
        "none",
    }
)
REQUIRED_KEYS = frozenset(
    {"name", "raw_name", "source", "available_at_admission", "status", "reason", "group"}
)
ALLOWED_KEYS = REQUIRED_KEYS | {"kind", "levels", "range", "format", "date_only", "dtype"}
KIND_REQUIRED_STATUSES = frozenset({"include", "review"})


class FeatureRegistryError(ValueError):
    """The feature registry file is invalid."""


@dataclass(frozen=True)
class FeatureEntry:
    """One row of the feature registry."""

    name: str
    raw_name: tuple[str, ...]
    source: str  # canonical | raw | derived
    status: str  # include | exclude | review
    reason: str
    group: str
    available_at_admission: bool
    mapping: FieldMapping | None  # set only when source == "raw" and a kind was given


@dataclass(frozen=True)
class FeatureRegistry:
    """A parsed and validated feature registry."""

    path: Path
    sha256: str
    entries: tuple[FeatureEntry, ...]

    def included(self) -> tuple[FeatureEntry, ...]:
        """Entries with ``status: include`` -- the only ones model code may read."""
        return tuple(e for e in self.entries if e.status == "include")


def _raw_names(name: str, raw_name: object) -> tuple[str, ...]:
    if raw_name is None:
        return ()
    if isinstance(raw_name, str):
        items = [raw_name]
    elif isinstance(raw_name, list):
        items = list(raw_name)
    else:
        raise FeatureRegistryError(f"{name}: raw_name must be a string, a list, or null")
    if not all(isinstance(item, str) and item.strip() for item in items):
        raise FeatureRegistryError(f"{name}: raw_name entries must be non-empty strings")
    return tuple(str(item) for item in items)


def _check_raw_count(name: str, spec: dict[str, Any], kind: str, raw_cols: tuple[str, ...]) -> None:
    if not raw_cols:
        raise FeatureRegistryError(f"{name}: raw_name is required for kind {kind!r}")
    if kind == "integer_sum":
        if len(raw_cols) < 2:
            raise FeatureRegistryError(f"{name}: integer_sum needs at least two raw columns")
    elif kind == "gestational_age" and spec.get("format") == "weeks_plus_days":
        if len(raw_cols) > 2:
            raise FeatureRegistryError(f"{name}: weeks_plus_days takes one or two raw columns")
    elif len(raw_cols) != 1:
        raise FeatureRegistryError(f"{name}: exactly one raw column is required for kind {kind!r}")


def _mapping_levels(
    name: str, spec: dict[str, Any], kind: str, dtype: str | None
) -> dict[str, Any]:
    if kind == "category":
        return _levels(name, spec.get("levels"), dtype or "object")
    if kind == "integer_sum":
        return _integer_sum_levels(name, spec["levels"]) if "levels" in spec else {}
    if "levels" in spec:
        raise FeatureRegistryError(f"{name}: levels are only allowed on category and integer_sum")
    return {}


def _mapping_range(name: str, spec: dict[str, Any], kind: str) -> tuple[float, float] | None:
    if "range" not in spec:
        return None
    if kind not in RANGE_KINDS:
        raise FeatureRegistryError(
            f"{name}: range is only allowed on integer, integer_sum, float and "
            "gestational_age kinds"
        )
    return _range(name, spec["range"])


def _mapping_formats(name: str, spec: dict[str, Any], kind: str) -> tuple[str | None, str | None]:
    # (gestational-age format, datetime format); a datetime defaults to ISO8601.
    fmt = spec.get("format")
    if kind not in FORMAT_KINDS and fmt is not None:
        raise FeatureRegistryError(f"{name}: format is only allowed on gestational_age/datetime")
    if kind == "gestational_age":
        if not isinstance(fmt, str) or fmt not in GA_FORMATS:
            raise FeatureRegistryError(
                f"{name}: gestational_age needs format in {sorted(GA_FORMATS)}"
            )
        return fmt, None
    if kind != "datetime":
        return None, None
    fmt = fmt or "ISO8601"
    if not isinstance(fmt, str) or not fmt.strip():
        raise FeatureRegistryError(f"{name}: datetime format must be a non-empty string")
    return None, fmt


def _mapping_date_only(name: str, spec: dict[str, Any], kind: str) -> bool:
    date_only = spec.get("date_only", False)
    if "date_only" in spec:
        if kind != "datetime":
            raise FeatureRegistryError(f"{name}: date_only is only allowed on datetime kind")
        if not isinstance(date_only, bool):
            raise FeatureRegistryError(f"{name}: date_only must be true or false")
    return date_only


def _parse_mapping(name: str, spec: dict[str, Any], raw_cols: tuple[str, ...]) -> FieldMapping:
    """Build a :class:`FieldMapping` for a ``source: raw`` entry (mirrors mapping._parse_field,
    without the canonical-schema constraints that don't apply to an arbitrary feature)."""
    kind = spec.get("kind")
    if not isinstance(kind, str) or kind not in FEATURE_KINDS:
        raise FeatureRegistryError(f"{name}: kind must be one of {sorted(FEATURE_KINDS)}")
    _check_raw_count(name, spec, kind, raw_cols)
    dtype = spec.get("dtype")
    if dtype is not None and dtype not in ("Int64", "float64"):
        raise FeatureRegistryError(f"{name}: dtype must be Int64 or float64 when given")
    levels = _mapping_levels(name, spec, kind, dtype)
    valid_range = _mapping_range(name, spec, kind)
    ga_format, datetime_format = _mapping_formats(name, spec, kind)
    return FieldMapping(
        canonical=name,
        kind=kind,
        raw=raw_cols,
        status="confirmed",
        note="",
        levels=levels,
        dtype=dtype,
        valid_range=valid_range,
        ga_format=ga_format,
        datetime_format=datetime_format,
        date_only=_mapping_date_only(name, spec, kind),
    )


@dataclass(frozen=True)
class _EntryHeader:
    """The validated scalar keys of a registry entry."""

    name: str
    source: str
    status: str
    group: str
    available: bool
    reason: str


def _entry_header(spec: object, seen_names: set[str]) -> tuple[dict[str, Any], _EntryHeader]:
    # The entry is a mapping with exactly the allowed keys and valid scalar values.
    if not isinstance(spec, dict):
        raise FeatureRegistryError("each feature entry must be a mapping of keys")
    missing = sorted(REQUIRED_KEYS - spec.keys())
    if missing:
        raise FeatureRegistryError(f"feature entry missing required key(s): {missing}")
    unknown = sorted(k for k in spec if k not in ALLOWED_KEYS)
    if unknown:
        raise FeatureRegistryError(f"feature entry has unknown key(s) {unknown}")
    name = spec["name"]
    if not isinstance(name, str) or not name.strip():
        raise FeatureRegistryError("feature name must be a non-empty string")
    if name in seen_names:
        raise FeatureRegistryError(f"{name}: duplicate feature name")
    seen_names.add(name)
    source, status, group = spec["source"], spec["status"], spec["group"]
    if source not in SOURCES:
        raise FeatureRegistryError(f"{name}: source must be one of {sorted(SOURCES)}")
    if status not in STATUSES:
        raise FeatureRegistryError(f"{name}: status must be one of {sorted(STATUSES)}")
    if group not in GROUPS:
        raise FeatureRegistryError(f"{name}: group must be one of {sorted(GROUPS)}")
    available = spec["available_at_admission"]
    if not isinstance(available, bool):
        raise FeatureRegistryError(f"{name}: available_at_admission must be true or false")
    reason = spec["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise FeatureRegistryError(f"{name}: reason must be a non-empty string")
    if status == "include" and not available:
        raise FeatureRegistryError(f"{name}: status include requires available_at_admission true")
    return spec, _EntryHeader(name, source, status, group, available, reason)


def _entry_raw_columns(
    header: _EntryHeader, raw_name: object, seen_raw: dict[str, str]
) -> tuple[str, ...]:
    # Derived entries read no raw column; every raw column is listed by one entry only.
    name, source = header.name, header.source
    if source == "derived":
        if raw_name is not None:
            raise FeatureRegistryError(f"{name}: derived features must have raw_name: null")
        raw_cols: tuple[str, ...] = ()
    else:
        if raw_name is None:
            raise FeatureRegistryError(f"{name}: raw_name is required for source {source!r}")
        raw_cols = _raw_names(name, raw_name)
    for col in raw_cols:
        if col in seen_raw:
            raise FeatureRegistryError(
                f"raw column {col!r} is listed by both {seen_raw[col]!r} and {name!r}; "
                "every raw column must be listed exactly once"
            )
        seen_raw[col] = name
    return raw_cols


def _entry_mapping(
    header: _EntryHeader, spec: dict[str, Any], raw_cols: tuple[str, ...]
) -> FieldMapping | None:
    # Only a raw entry has a mapping; it needs a kind once its status makes it a model input.
    name, source, status = header.name, header.source, header.status
    has_kind = "kind" in spec
    if source != "raw" and has_kind:
        raise FeatureRegistryError(f"{name}: kind is only allowed on source: raw entries")
    if source != "raw":
        return None
    if has_kind:
        return _parse_mapping(name, spec, raw_cols)
    if status in KIND_REQUIRED_STATUSES:
        raise FeatureRegistryError(
            f"{name}: kind is required for a raw feature with status {status!r}"
        )
    if {"levels", "range", "format", "date_only", "dtype"} & spec.keys():
        raise FeatureRegistryError(f"{name}: mapping keys given without a kind")
    return None


def _parse_entry(spec: object, seen_names: set[str], seen_raw: dict[str, str]) -> FeatureEntry:
    checked, header = _entry_header(spec, seen_names)
    raw_cols = _entry_raw_columns(header, checked["raw_name"], seen_raw)
    return FeatureEntry(
        name=header.name,
        raw_name=raw_cols,
        source=header.source,
        status=header.status,
        reason=header.reason,
        group=header.group,
        available_at_admission=header.available,
        mapping=_entry_mapping(header, checked, raw_cols),
    )


def load_feature_registry(path: Path) -> FeatureRegistry:
    """Parse and strictly validate ``configs/features_v1.yaml``."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise FeatureRegistryError(f"feature registry is not valid YAML: {exc}") from exc
    if not isinstance(data, dict) or "features" not in data:
        raise FeatureRegistryError("feature registry must be a mapping with a 'features' key")
    raw_entries = data["features"]
    if not isinstance(raw_entries, list):
        raise FeatureRegistryError("'features' must be a list of entries")

    seen_names: set[str] = set()
    seen_raw: dict[str, str] = {}
    entries = tuple(_parse_entry(spec, seen_names, seen_raw) for spec in raw_entries)
    return FeatureRegistry(path=path, sha256=file_sha256(path), entries=entries)


def check_coverage(
    registry: FeatureRegistry, raw_columns: list[str]
) -> tuple[list[str], list[str]]:
    """Compare the registry's raw columns against the actual export.

    Returns:
        ``(missing, extra)``: raw columns present in the export but not listed in the
        registry, and raw columns the registry lists that are not in the export.
    """
    registered = {col for entry in registry.entries for col in entry.raw_name}
    present = set(raw_columns)
    missing = sorted(present - registered)
    extra = sorted(registered - present)
    return missing, extra


def build_raw_features(
    raw_df: pd.DataFrame, registry: FeatureRegistry
) -> tuple[pd.DataFrame, list[FieldReport]]:
    """Build feature columns for every ``source: raw`` entry that carries a mapping.

    Reuses :func:`robson_ml.mapping.map_field` (the same per-field mapping machinery and
    counting guarantees as :func:`robson_ml.mapping.apply_mapping`). Columns are named by
    the registry entry's ``name``, not the raw column name. Callers wanting only the
    features approved for modelling should filter to :meth:`FeatureRegistry.included`
    first (or filter the returned frame's columns by name afterwards).
    """
    columns: dict[str, pd.Series] = {}
    reports: list[FieldReport] = []
    for entry in registry.entries:
        if entry.source != "raw" or entry.mapping is None:
            continue
        series, report = map_field(raw_df, entry.mapping)
        columns[entry.name] = series
        reports.append(report)
    index = raw_df.index if not columns else next(iter(columns.values())).index
    return pd.DataFrame(columns, index=index), reports
