"""Map raw export columns to the canonical schema (spec §5) via configs/mapping_ur_cmhs.yaml.

Nothing is guessed. Every present raw value ends up in exactly one bucket, and every bucket
is counted in the report: mapped, explicitly recorded as missing (a level mapped to ``~``),
unparsed (cannot be converted without guessing, including category labels absent from
``levels``) or out of range. Fields whose meaning is uncertain carry ``status: review`` and
are listed for human decision. Reports and error messages never contain cell values.

The raw patient identifier never enters canonical data: ``mother_key`` can only be produced
by the ``hash_key`` kind, a salted one-way hash (spec v1.2 §5).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import math
import re
from collections.abc import Hashable
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
    ONSET_LEVELS,
    PRELABOUR_CS_TYPES,
    PRESENTATION_LEVELS,
    YES_NO,
    empty_column,
)

KINDS = frozenset(
    {
        "row_key",
        "hash_key",
        "integer",
        "integer_sum",
        "float",
        "category",
        "datetime",
        "gestational_age",
        "text",
    }
)
STATUSES = frozenset({"confirmed", "review"})
GA_FORMATS = frozenset({"decimal_weeks", "weeks_plus_days", "completed_weeks", "weeks_days_text"})
ALLOWED_KEYS = frozenset(
    {"raw", "kind", "status", "note", "levels", "dtype", "range", "format", "date_only"}
)
RANGE_KINDS = frozenset({"integer", "integer_sum", "float", "gestational_age"})
FORMAT_KINDS = frozenset({"gestational_age", "datetime"})
# Output dtype each non-category kind produces; it must equal the canonical dtype.
KIND_DTYPES: dict[str, str] = {
    "integer": "Int64",
    "integer_sum": "Int64",
    "float": "float64",
    "gestational_age": "float64",
    "datetime": "datetime64[ns]",
    "text": "object",
    "row_key": "object",
    "hash_key": "object",
}
# Kinds only allowed on one canonical field.
KIND_FIELDS: dict[str, str] = {"admission_id": "row_key", "mother_key": "hash_key"}
# Fields that only one kind may produce: mother_key must never carry the raw identifier.
REQUIRED_KINDS: dict[str, str] = {"mother_key": "hash_key"}
# Canonical datetime fields the schema holds as dates only: they must declare date_only: true.
DATE_ONLY_FIELDS = frozenset({"delivery_date"})
# Canonical fields whose values come from a fixed set (schema.py); level values must be in it.
FIXED_LEVELS: dict[str, frozenset[object]] = {
    "fetal_presentation": frozenset(PRESENTATION_LEVELS),
    "onset_of_labour": frozenset(ONSET_LEVELS),
    "prelabour_cs_type": frozenset(PRELABOUR_CS_TYPES),
    "preeclampsia_recorded": frozenset(YES_NO),
    "gdm_recorded": frozenset(YES_NO),
    "cs": frozenset({0, 1}),
}
LOCAL_TIMEZONE = "Africa/Kigali"
ROW_KEY_PREFIX = "ADM"
HASH_KEY_PREFIX = "MK_"
HASH_KEY_HEX_CHARS = 32  # 128 bits of the SHA-256 digest
HASH_KEY_SEPARATOR = b"\x1f"  # ASCII unit separator between salt and identifier
MIN_SALT_BYTES = 16
SALT_ERROR = f"hash_key needs a salt of at least {MIN_SALT_BYTES} bytes"
MAX_EXACT_INTEGER = 2**53  # beyond this float64 cannot hold every integer exactly
MISSING_ROW_LABEL = "(missing)"
_WEEKS_PLUS_DAYS = re.compile(r"^\s*(\d{1,2})\s*\+\s*([0-6])\s*$")
# Free-text gestational age (format weeks_days_text), matched case-insensitively against the
# whole stripped value. Accepted: "W", "W<unit>", "W<unit><sep>D<dunit>?", "W+D", "W+D<dunit>"
# with W 1-2 ASCII digits and D a single digit 0-6. Anything else (fractions, days >= 7,
# ranges, extra numbers or words) does not match and is counted as unparsed.
_WEEK_UNIT = r"(?:weeks|weekz|week|wks|wk|w)"  # "weekz": a typo seen in the export
_DAY_UNIT = r"(?:days|day|d)"
_TEXT_SEP = r"(?:\s*[,+]\s*|\s+and\s+|\s*)"
_WEEKS_DAYS_TEXT = re.compile(
    rf"(?P<weeks>[0-9]{{1,2}})"
    rf"(?:\s*{_WEEK_UNIT}(?:{_TEXT_SEP}(?P<days>[0-6])(?:\s*{_DAY_UNIT})?)?"
    rf"|\s*\+\s*(?P<plus_days>[0-6])(?:\s*{_DAY_UNIT})?)?",
    re.IGNORECASE,
)
# A clock time followed by "Z" or a numeric UTC offset: the string carries its own timezone.
_TZ_SUFFIX = re.compile(r"\d{2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?\s*(?:Z|[+-]\d{2}(?::?\d{2})?)\s*$")
# Under the default ISO8601 format, only strings with a full YYYY-MM-DD date are trusted;
# reduced-precision ISO strings like "2024" or "2024-03" would otherwise have pandas fabricate
# a day (and month) value.
_ISO_FULL_DATE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}")


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
    date_only: bool = False


@dataclass(frozen=True)
class MappingConfig:
    """A parsed mapping file."""

    source: str
    sheet: str | None
    fields: dict[str, FieldMapping]


@dataclass
class FieldReport:
    """Aggregate account of how one canonical field was mapped.

    ``n_raw_nonnull == n_mapped + n_unparsed + n_out_of_range + n_explicit_missing +
    n_incomplete`` always holds. For a two-column gestational age (or an ``integer_sum``
    field) a row counts as present when any column is; for ``row_key`` every row counts as
    present and mapped. ``n_incomplete`` is only ever nonzero for ``integer_sum`` (a row
    where some but not all columns are present and parsed, so the sum is undefined);
    ``n_explicit_missing`` is always 0 for ``integer_sum`` (an explicit-missing level cell
    counts as incomplete instead).
    """

    canonical: str
    raw: list[str]
    kind: str
    status: str
    note: str
    n_raw_nonnull: int = 0
    n_mapped: int = 0
    n_unparsed: int = 0
    n_out_of_range: int = 0
    n_explicit_missing: int = 0
    n_incomplete: int = 0  # integer_sum: some but not all columns present and parsed
    n_days_blank: int = 0  # two-column GA: weeks present, days blank, taken as weeks + 0
    n_tz_aware: int = 0  # datetime: mapped values that carried a UTC offset (converted)
    n_time_dropped: int = 0  # date_only datetime: mapped values whose time of day was dropped
    unmapped_levels: dict[str, Any] = field(default_factory=dict)


@dataclass
class MappingReport:
    """Aggregate account of the whole mapping (no row-level values)."""

    fields: list[FieldReport]
    missing_canonical_fields: list[str]
    unreferenced_raw_columns: list[str]
    review_fields: list[str]


# --- configuration -------------------------------------------------------------------------


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate keys instead of silently keeping the last one."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Hashable, Any]:
        seen: set[object] = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, Hashable):
                continue  # the base loader reports unhashable keys itself
            if key in seen:
                raise MappingError(
                    f"duplicate key {key!r} in mapping file (line {key_node.start_mark.line + 1})"
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _known_field(name: str) -> bool:
    return name in CANONICAL_DTYPES


def _canonical_dtype(name: str) -> str:
    return CANONICAL_DTYPES[name]


def _is_real(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _finite(value: object) -> float | None:
    """A config number (not bool) as a finite float, else None."""
    if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _raw_columns(name: str, raw: object) -> tuple[str, ...]:
    items: list[object]
    if raw is None:
        items = []
    elif isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, list):
        items = list(raw)
    else:
        raise MappingError(f"{name}: raw must be a column name or a list of column names")
    if not all(isinstance(item, str) and item.strip() for item in items):
        raise MappingError(f"{name}: raw column names must be non-empty strings")
    columns = tuple(str(item) for item in items)
    if len(set(columns)) != len(columns):
        raise MappingError(f"{name}: raw lists the same column twice")
    return columns


def _parse_range(name: str, bounds: object) -> tuple[float, float]:
    if not (isinstance(bounds, list) and len(bounds) == 2 and all(map(_is_real, bounds))):
        raise MappingError(f"{name}: range must be a list of two numbers [low, high]")
    low, high = _finite(bounds[0]), _finite(bounds[1])
    if low is None or high is None or low > high:
        raise MappingError(f"{name}: range must be finite with low <= high")
    return low, high


def _level_label(key: object) -> str | None:
    if isinstance(key, str):
        return key.strip()
    if _is_real(key):
        return _label(key)
    return None


def _level_value(name: str, value: object, dtype: str) -> object:
    if value is None:
        return None
    number = _finite(value)
    if dtype == "Int64":
        if number is None or not number.is_integer():
            raise MappingError(f"{name}: non-integer or non-numeric level value")
        value = int(number)
    elif dtype == "float64":
        if number is None:
            raise MappingError(f"{name}: non-numeric level value")
        value = number
    elif not isinstance(value, str):
        raise MappingError(f"{name}: level values must be quoted strings for a text field")
    allowed = FIXED_LEVELS.get(name)
    if allowed is not None and value not in allowed:
        raise MappingError(
            f"{name}: level value not among the schema's categories {sorted(map(str, allowed))}"
        )
    return value


def _parse_levels(name: str, levels: object, dtype: str) -> dict[str, Any]:
    if not isinstance(levels, dict) or not levels:
        raise MappingError(f"{name}: category fields need a non-empty levels mapping")
    parsed: dict[str, Any] = {}
    for key, value in levels.items():
        if isinstance(key, bool) or isinstance(value, bool):
            raise MappingError(f"{name}: quote yes/no/true/false labels and values in YAML")
        label = _level_label(key)
        if not label:
            raise MappingError(f"{name}: level labels must be non-empty strings or numbers")
        if label in parsed:
            raise MappingError(f"{name}: duplicate level label after trimming whitespace")
        parsed[label] = _level_value(name, value, dtype)
    return parsed


def _integer_sum_level_value(name: str, value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise MappingError(f"{name}: quote yes/no/true/false labels and values in YAML")
    number = _finite(value)
    if number is None or not number.is_integer() or number < 0:
        raise MappingError(f"{name}: integer_sum level values must be non-negative integers")
    return int(number)


def _parse_integer_sum_levels(name: str, levels: object) -> dict[str, int | None]:
    if not isinstance(levels, dict) or not levels:
        raise MappingError(f"{name}: levels mapping must be non-empty")
    parsed: dict[str, int | None] = {}
    for key, value in levels.items():
        if isinstance(key, bool):
            raise MappingError(f"{name}: quote yes/no/true/false labels and values in YAML")
        label = _level_label(key)
        if not label:
            raise MappingError(f"{name}: level labels must be non-empty strings or numbers")
        if label in parsed:
            raise MappingError(f"{name}: duplicate level label after trimming whitespace")
        parsed[label] = _integer_sum_level_value(name, value)
    return parsed


def _parse_field(name: str, spec: object) -> FieldMapping:
    if not _known_field(name):
        raise MappingError(f"unknown canonical field: {name}")
    if not isinstance(spec, dict):
        raise MappingError(f"{name}: field spec must be a mapping of keys")
    unknown = sorted(str(k) for k in spec if k not in ALLOWED_KEYS)
    if unknown:
        raise MappingError(f"{name}: unknown key(s) {unknown}; allowed {sorted(ALLOWED_KEYS)}")
    kind = spec.get("kind")
    if not isinstance(kind, str) or kind not in KINDS:
        raise MappingError(f"{name}: kind must be one of {sorted(KINDS)}")
    status = spec.get("status", "review")
    if not isinstance(status, str) or status not in STATUSES:
        raise MappingError(f"{name}: status must be confirmed or review")

    dtype = _canonical_dtype(name)
    declared = spec.get("dtype")
    if declared is not None and declared != dtype:
        raise MappingError(f"{name}: dtype must match the canonical dtype {dtype} (or be omitted)")
    if kind == "category":
        if dtype == "datetime64[ns]":
            raise MappingError(f"{name}: kind category cannot produce a datetime field")
    elif KIND_DTYPES[kind] != dtype:
        raise MappingError(
            f"{name}: kind {kind} produces {KIND_DTYPES[kind]} but the canonical dtype is {dtype}"
        )

    for field_name, field_kind in KIND_FIELDS.items():
        if kind == field_kind and name != field_name:
            raise MappingError(f"{name}: kind {field_kind} is only allowed on {field_name}")
    required_kind = REQUIRED_KINDS.get(name)
    if required_kind is not None and kind != required_kind:
        raise MappingError(f"{name}: this field can only be mapped with kind {required_kind}")

    fmt = spec.get("format")
    raw_cols = _raw_columns(name, spec.get("raw"))
    if kind == "row_key":
        if raw_cols:
            raise MappingError(f"{name}: kind row_key takes no raw column")
    elif not raw_cols:
        raise MappingError(f"{name}: a raw column is required")
    elif kind == "gestational_age" and fmt == "weeks_plus_days":
        if len(raw_cols) > 2:
            raise MappingError(f"{name}: weeks_plus_days takes one column, or two (weeks, days)")
    elif kind == "integer_sum":
        if len(raw_cols) < 2:
            raise MappingError(f"{name}: integer_sum needs at least two distinct raw columns")
    elif len(raw_cols) != 1:
        raise MappingError(f"{name}: exactly one raw column is required")

    if kind == "category":
        levels = _parse_levels(name, spec.get("levels"), dtype)
    elif kind == "integer_sum":
        levels = _parse_integer_sum_levels(name, spec["levels"]) if "levels" in spec else {}
    elif "levels" in spec:
        raise MappingError(f"{name}: levels are only allowed on category and integer_sum fields")
    else:
        levels = {}

    valid_range = None
    if "range" in spec:
        if kind not in RANGE_KINDS:
            raise MappingError(
                f"{name}: range is only allowed on integer, integer_sum, float and "
                "gestational_age fields"
            )
        valid_range = _parse_range(name, spec["range"])

    if kind not in FORMAT_KINDS and "format" in spec:
        raise MappingError(f"{name}: format is only allowed on gestational_age and datetime fields")
    ga_format = None
    datetime_format = None
    if kind == "gestational_age":
        if not isinstance(fmt, str) or fmt not in GA_FORMATS:
            raise MappingError(f"{name}: gestational_age needs format in {sorted(GA_FORMATS)}")
        ga_format = fmt
    elif kind == "datetime":
        if fmt is None:
            fmt = "ISO8601"
        if not isinstance(fmt, str) or not fmt.strip():
            raise MappingError(f"{name}: datetime format must be a non-empty string")
        if fmt.strip().lower().startswith("mixed") or fmt.strip().lower() == "infer":
            raise MappingError(
                f"{name}: datetime format must not guess (e.g. 'mixed' or 'infer'); "
                "use an explicit strptime format or the ISO8601 default"
            )
        datetime_format = fmt

    date_only = spec.get("date_only", False)
    if "date_only" in spec:
        if kind != "datetime":
            raise MappingError(f"{name}: date_only is only allowed on datetime fields")
        if not isinstance(date_only, bool):
            raise MappingError(f"{name}: date_only must be true or false")
    if name in DATE_ONLY_FIELDS and not date_only:
        raise MappingError(
            f"{name}: the schema holds this field as a date only; set date_only: true"
        )

    note = spec.get("note")
    return FieldMapping(
        canonical=name,
        kind=kind,
        raw=raw_cols,
        status=status,
        note="" if note is None else str(note),
        levels=levels,
        dtype=dtype,
        valid_range=valid_range,
        ga_format=ga_format,
        datetime_format=datetime_format,
        date_only=date_only,
    )


def load_mapping(path: Path) -> MappingConfig:
    """Parse and validate a mapping YAML file."""
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader) or {}
    except yaml.YAMLError as exc:
        raise MappingError(f"mapping file is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise MappingError("mapping file must be a mapping with a 'fields' key")
    raw_fields = data.get("fields") or {}
    if not isinstance(raw_fields, dict):
        raise MappingError("'fields' must be a mapping of canonical field -> spec")
    fields = {str(name): _parse_field(str(name), spec) for name, spec in raw_fields.items()}
    sheet = data.get("sheet")
    return MappingConfig(str(data.get("source", "")), None if sheet is None else str(sheet), fields)


# --- value conversion ----------------------------------------------------------------------


def _blank_to_none(value: object) -> object:
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _present_values(series: pd.Series) -> pd.Series:
    """Object series in which every missing marker (blank, None, NaN, NA, NaT) is None."""
    values = series.map(_blank_to_none).astype(object)
    return values.where(values.notna(), None)


def _label(value: object) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _number(value: object) -> float | None:
    """The value as a finite float within +-2**53, or None if it is not unambiguously one.

    Booleans, non-finite values and anything that is not a number or numeric string are
    rejected rather than coerced.
    """
    if value is None or isinstance(value, bool | np.bool_):
        return None
    number: float
    if isinstance(value, int | np.integer):
        if abs(int(value)) > MAX_EXACT_INTEGER:
            return None
        number = float(value)
    elif isinstance(value, float | np.floating):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if "_" in text:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(number) or abs(number) > MAX_EXACT_INTEGER:
        return None
    return number


def _numbers(values: pd.Series, whole: bool) -> pd.Series:
    """float64 series of parsed numbers; NaN where absent or unparsed (or fractional if whole)."""

    def parse(value: object) -> float:
        number = _number(value)
        if number is None or (whole and not number.is_integer()):
            return math.nan
        return number

    return values.map(parse).astype("float64")


def _weeks_plus_days(value: object) -> float:
    """``N`` -> N weeks, ``N+D`` -> N + D/7. A fractional value is ambiguous (decimal weeks
    or weeks.days) and is rejected."""
    if isinstance(value, str):
        match = _WEEKS_PLUS_DAYS.match(value)
        if match:
            return int(match.group(1)) + int(match.group(2)) / 7.0
    number = _number(value)
    return number if number is not None and number.is_integer() else math.nan


def _weeks_days_text(value: object) -> float:
    """Free-text gestational age (see ``_WEEKS_DAYS_TEXT``) -> W + D/7 decimal weeks.

    A number that is not text is taken as whole weeks when it is an integer from 0 to 99
    (a spreadsheet cell holding just "38"); anything else is NaN (unparsed).
    """
    if isinstance(value, str):
        match = _WEEKS_DAYS_TEXT.fullmatch(value.strip())
        if match is None:
            return math.nan
        days = match.group("days") or match.group("plus_days") or "0"
        return int(match.group("weeks")) + int(days) / 7.0
    number = _number(value)
    if number is None or not number.is_integer() or not 0 <= number <= 99:
        return math.nan
    return number


def _hash_key(value: object, salt: bytes) -> str | None:
    """``MK_`` + the first 32 hex characters of SHA-256(salt || 0x1F || identifier).

    The identifier is the value as trimmed text (case kept; an integral float such as 12345.0
    reads as "12345", like the same number held as text). Blank or missing -> None.
    """
    text = _label(value)
    if not text:
        return None
    digest = hashlib.sha256(salt + HASH_KEY_SEPARATOR + text.encode("utf-8")).hexdigest()
    return HASH_KEY_PREFIX + digest[:HASH_KEY_HEX_CHARS]


def _apply_range(values: pd.Series, bounds: tuple[float, float] | None) -> tuple[pd.Series, int]:
    if bounds is None:
        return values, 0
    outside = values.notna() & ((values < bounds[0]) | (values > bounds[1]))
    return values.mask(outside), int(outside.sum())


def _local_timestamp(value: object) -> tuple[pd.Timestamp | None, bool]:
    """A datetime-like object as naive Africa/Kigali wall time, and whether it was aware."""
    try:
        stamp = pd.Timestamp(value)  # type: ignore[arg-type]
        if stamp is pd.NaT:
            return None, False
        aware = stamp.tzinfo is not None
        if aware:
            stamp = stamp.tz_convert(LOCAL_TIMEZONE).tz_localize(None)
        return stamp.as_unit("ns"), aware
    except (ValueError, TypeError, OverflowError):
        return None, False


def _parse_string(value: str, fmt: str) -> tuple[pd.Timestamp | None, bool]:
    try:
        stamp = pd.to_datetime(value, format=fmt)
    except (ValueError, TypeError, OverflowError):
        return None, False
    return _local_timestamp(stamp)


def _parse_strings(values: pd.Series, fmt: str, aware: bool) -> tuple[pd.Series, pd.Series]:
    """Parse strings with ``fmt``: (naive local datetime64[ns], per-value aware flag)."""
    try:
        parsed = pd.to_datetime(values, errors="coerce", format=fmt, utc=aware)
    except (ValueError, TypeError, OverflowError):
        parsed = None
    if parsed is not None and isinstance(parsed.dtype, pd.DatetimeTZDtype):
        local = parsed.dt.tz_convert(LOCAL_TIMEZONE).dt.tz_localize(None)
        return local.astype("datetime64[ns]"), local.notna()
    if parsed is not None and pd.api.types.is_datetime64_dtype(parsed.dtype):
        return parsed.astype("datetime64[ns]"), pd.Series(False, index=values.index)
    # Mixed or otherwise unusual results: fall back to one value at a time.
    pairs = [_parse_string(value, fmt) for value in values]
    stamps = pd.Series([p[0] for p in pairs], index=values.index, dtype="datetime64[ns]")
    return stamps, pd.Series([p[1] for p in pairs], index=values.index, dtype=bool)


def _datetimes(values: pd.Series, fmt: str) -> tuple[pd.Series, int]:
    """Parse to naive Africa/Kigali wall time. Strings and datetime objects that carry an
    offset are converted; naive ones are taken as local time (never assumed UTC). Numbers
    (e.g. spreadsheet serials) are not guessed at. Returns (datetime64[ns], n aware mapped)."""
    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    aware_flags = pd.Series(False, index=values.index)
    is_string = values.map(lambda v: isinstance(v, str))
    if fmt == "ISO8601":
        # Reduced-precision ISO strings ("2024", "2024-03") would have pandas fabricate a
        # day (and month) value; only strings with a full YYYY-MM-DD date are trusted here.
        is_string = is_string & values.map(
            lambda v: isinstance(v, str) and bool(_ISO_FULL_DATE.match(v))
        )
    has_offset = values.map(lambda v: isinstance(v, str) and bool(_TZ_SUFFIX.search(v)))
    for mask, aware in ((is_string & ~has_offset, False), (is_string & has_offset, True)):
        if mask.any():
            stamps, flags = _parse_strings(values[mask], fmt, aware)
            result.loc[mask] = stamps
            aware_flags.loc[mask] = flags
    is_object = values.map(lambda v: isinstance(v, dt.date | np.datetime64))
    if is_object.any():
        pairs = [_local_timestamp(v) for v in values[is_object]]
        result.loc[is_object] = pd.Series(
            [p[0] for p in pairs], index=values.index[is_object], dtype="datetime64[ns]"
        )
        aware_flags.loc[is_object] = pd.Series(
            [p[1] for p in pairs], index=values.index[is_object], dtype=bool
        )
    return result, int((aware_flags & result.notna()).sum())


# --- mapping -------------------------------------------------------------------------------


def _map_category(labels: pd.Series, spec: FieldMapping, report: FieldReport) -> pd.Series:
    known = labels.isin(list(spec.levels))
    unmapped = labels.notna() & ~known
    mapped = labels.map(lambda v: spec.levels.get(v) if v is not None else None)
    report.n_unparsed = int(unmapped.sum())
    report.n_explicit_missing = int((known & mapped.isna()).sum())
    report.unmapped_levels = {
        str(level): n
        for level, n in level_counts(labels[unmapped]).values.tolist()
        if level != MISSING_ROW_LABEL
    }
    dtype = _canonical_dtype(spec.canonical)
    if dtype in ("Int64", "float64"):
        return pd.to_numeric(mapped.astype("float64")).astype(dtype)  # type: ignore[call-overload]
    return mapped.astype(object)


def _map_integer_sum(
    columns: list[pd.Series], spec: FieldMapping, report: FieldReport, present: pd.Series
) -> pd.Series:
    """Sum >=2 raw columns (spec §derived counts): each cell is parsed like ``integer`` after
    applying ``levels`` (a top-code label -> non-negative int, or ``~`` for explicit missing).

    Per row: unparseable present cell -> unparsed (wins over blank/missing-level cells in the
    same row); else any cell blank or an explicit-missing level (while others parsed) ->
    incomplete (sum undefined); else the cells sum to the row's value.
    """
    idx = columns[0].index
    unparsed_cells: list[pd.Series] = []
    incomplete_cells: list[pd.Series] = []
    values: list[pd.Series] = []
    for col in columns:
        labels = col.map(_label)
        blank = labels.isna()
        known = ~blank & labels.isin(list(spec.levels))
        leveled = labels.map(lambda label: spec.levels.get(label) if label in spec.levels else None)
        missing_level = known & leveled.isna()
        numeric = _numbers(col, whole=True)
        value = pd.Series(np.nan, index=idx, dtype="float64")
        level_ok = known & ~missing_level
        value.loc[level_ok] = leveled.loc[level_ok].astype("float64")
        use_numeric = ~known & ~blank
        value.loc[use_numeric] = numeric.loc[use_numeric]
        unparsed_cells.append(use_numeric & value.isna())
        incomplete_cells.append(blank | missing_level)
        values.append(value)

    unparsed_row = present & pd.concat(unparsed_cells, axis=1).any(axis=1)
    incomplete_row = present & ~unparsed_row & pd.concat(incomplete_cells, axis=1).any(axis=1)
    complete_row = present & ~unparsed_row & ~incomplete_row
    report.n_unparsed = int(unparsed_row.sum())
    report.n_incomplete = int(incomplete_row.sum())
    total = pd.concat(values, axis=1).sum(axis=1, min_count=1)
    return total.where(complete_row)


def _map_gestational_age(
    columns: list[pd.Series], spec: FieldMapping, report: FieldReport
) -> pd.Series:
    if len(columns) == 2:
        weeks = _numbers(columns[0], whole=True)
        weeks = weeks.where(weeks >= 0)
        days = _numbers(columns[1], whole=True)
        days_blank = columns[1].isna()
        usable = weeks.notna() & (days_blank | days.between(0, 6))
        report.n_days_blank = int((weeks.notna() & days_blank).sum())
        return (weeks + days.mask(days_blank, 0.0) / 7.0).where(usable)
    if spec.ga_format == "weeks_plus_days":
        return columns[0].map(_weeks_plus_days).astype("float64")
    if spec.ga_format == "weeks_days_text":
        return columns[0].map(_weeks_days_text).astype("float64")
    return _numbers(columns[0], whole=spec.ga_format == "completed_weeks")


def _map_datetime(source: pd.Series, spec: FieldMapping, report: FieldReport) -> pd.Series:
    result, report.n_tz_aware = _datetimes(source, spec.datetime_format or "ISO8601")
    if spec.date_only:
        dates = result.dt.normalize()
        report.n_time_dropped = int((result.notna() & (result != dates)).sum())
        result = dates
    return result


def _map_one(
    raw: pd.DataFrame, spec: FieldMapping, report: FieldReport, salt: bytes | None
) -> pd.Series:
    n = len(raw)
    result: pd.Series
    if spec.kind == "row_key":
        result = pd.Series([f"{ROW_KEY_PREFIX}{i:06d}" for i in range(n)], dtype=object)
        report.n_raw_nonnull = n
        report.n_mapped = n
        return result
    columns = [_present_values(raw[c]) for c in spec.raw]
    source = columns[0]
    present = pd.concat(columns, axis=1).notna().any(axis=1)
    report.n_raw_nonnull = int(present.sum())

    if spec.kind == "text":
        result = source.map(_label).astype(object)
    elif spec.kind == "hash_key":
        if salt is None:  # apply_mapping checks the salt before any field is mapped
            raise MappingError(SALT_ERROR)
        result = source.map(lambda v: _hash_key(v, salt)).astype(object)
    elif spec.kind == "category":
        result = _map_category(source.map(_label), spec, report)
    elif spec.kind == "datetime":
        result = _map_datetime(source, spec, report)
        report.n_unparsed = int((present & result.isna()).sum())
    else:  # integer, integer_sum, float, gestational_age
        if spec.kind == "gestational_age":
            numeric = _map_gestational_age(columns, spec, report)
        elif spec.kind == "integer_sum":
            numeric = _map_integer_sum(columns, spec, report, present)
        else:
            numeric = _numbers(source, whole=spec.kind == "integer")
        if spec.kind != "integer_sum":
            # for integer_sum, n_unparsed is set by _map_integer_sum: numeric is also NaN for
            # incomplete rows, so the generic present-and-NaN test here would double count them.
            report.n_unparsed = int((present & numeric.isna()).sum())
        numeric, report.n_out_of_range = _apply_range(numeric, spec.valid_range)
        result = numeric.astype("Int64") if spec.kind in ("integer", "integer_sum") else numeric
    report.n_mapped = int(result.notna().sum())
    accounted = (
        report.n_mapped
        + report.n_unparsed
        + report.n_out_of_range
        + report.n_explicit_missing
        + report.n_incomplete
    )
    if accounted != report.n_raw_nonnull:
        raise RuntimeError(
            f"internal error mapping {spec.canonical}: {report.n_raw_nonnull} present raw "
            f"values but {accounted} accounted for"
        )
    return result.reset_index(drop=True)


def apply_mapping(
    raw: pd.DataFrame, config: MappingConfig, *, salt: bytes | None = None
) -> tuple[pd.DataFrame, MappingReport]:
    """Build the canonical frame from one raw sheet, with an aggregate mapping report.

    ``salt`` (at least 16 bytes, kept under data/) is required when the mapping has a
    ``hash_key`` field; it is never written to the frame or the report.
    """
    if any(spec.kind == "hash_key" for spec in config.fields.values()) and not (
        isinstance(salt, bytes) and len(salt) >= MIN_SALT_BYTES
    ):
        raise MappingError(SALT_ERROR)
    raw = raw.reset_index(drop=True)
    repeated = set(raw.columns[raw.columns.duplicated()])
    for spec in config.fields.values():
        absent = [c for c in spec.raw if c not in raw.columns]
        if absent:
            raise MappingError(f"{spec.canonical}: raw column(s) not in sheet: {absent}")
        clashing = [c for c in spec.raw if c in repeated]
        if clashing:
            raise MappingError(
                f"{spec.canonical}: raw column name(s) appear more than once in the sheet: "
                f"{clashing}"
            )
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
        columns[name] = _map_one(raw, spec, report, salt)
        reports.append(report)
    referenced = {c for spec in config.fields.values() for c in spec.raw}
    canonical = pd.DataFrame(columns)
    return canonical, MappingReport(
        fields=reports,
        missing_canonical_fields=missing,
        unreferenced_raw_columns=[str(c) for c in raw.columns if str(c) not in referenced],
        review_fields=[f.canonical for f in config.fields.values() if f.status == "review"],
    )
