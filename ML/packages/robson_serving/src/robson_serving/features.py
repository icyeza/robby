"""Row-level derivations from canonical admission fields to model columns.

The single implementation used when the training feature frame is built
(:mod:`robson_ml.feature_sets` calls it) and when one admission is scored
(:class:`robson_serving.loader.Scorer`). Every function is row-wise and stateless: a row's
output depends on that row only, so a frame holding one admission yields exactly the row the
training frame would hold for it.

- ``numeric_values`` / ``categorical_values``: dtype normalisation (float64 with NaN; strings
  with NaN, whole floats without ``.0``);
- ``bmi``: weight / (height / 100)^2, missing unless both are recorded;
- ``robson_group_no_onset``: the Robson engine with onset forced to missing, groups 1+2
  merged to ``"1_2"`` and 3+4 to ``"3_4"``, every other unresolved result ``"partial"``;
- ``is_missing_<field>``: 1.0 when the field is not recorded, else 0.0.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Mapping
from typing import Any, cast

import numpy as np
import pandas as pd

from robson_engine import (
    INPUT_FIELDS,
    ClassificationResult,
    RobsonInputs,
    RuleSet,
    classify,
    load_rule_set,
)

NUMERIC = "numeric"
CATEGORICAL = "categorical"
ROBSON_NO_ONSET = "robson_group_no_onset"
BMI = "bmi"
MISSING_PREFIX = "is_missing_"
GA_BAND_FIELDS = ("ga_band_lower", "ga_band_upper")
BMI_INPUTS = ("height_cm", "weight_kg")
ROBSON_NO_ONSET_INPUTS = (
    *(field for field in INPUT_FIELDS if field != "onset_of_labour"),
    *GA_BAND_FIELDS,
)
NO_ONSET_MERGES: dict[frozenset[int], str] = {
    frozenset({1, 2}): "1_2",
    frozenset({3, 4}): "3_4",
}
NO_ONSET_UNRESOLVED = "partial"


def is_missing(value: object) -> bool:
    """True for None, pd.NA, pd.NaT and float NaN (the ways "not recorded" reaches a row)."""
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, float | np.floating) and math.isnan(value)


def _clean(value: object) -> object:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _as_int(value: object) -> int | None:
    value = _clean(value)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("boolean where an integer was expected")
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ValueError("non-integer value where an integer was expected")


def _as_float(value: object) -> float | None:
    value = _clean(value)
    if value is None:
        return None
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return float(value)
    raise ValueError("non-numeric value where a number was expected")


def _as_str(value: object) -> str | None:
    value = _clean(value)
    return None if value is None else str(value)


def inputs_from_record(record: Mapping[str, object]) -> RobsonInputs:
    """Build engine inputs from one canonical record; NaN/NA/None all mean not recorded.

    When the exact gestational age is missing and both GA band bounds are recorded, the band
    is passed as ``gestational_age_range`` (coarse input). A presentation of
    ``non_cephalic`` is passed through as the engine's coarse code.
    """
    ga = _as_float(record.get("gestational_age_weeks"))
    ga_range = None
    if ga is None:
        lower = _as_float(record.get("ga_band_lower"))
        upper = _as_float(record.get("ga_band_upper"))
        if lower is not None and upper is not None:
            ga_range = (lower, upper)
    return RobsonInputs(
        parity=_as_int(record.get("parity")),
        previous_cs_count=_as_int(record.get("previous_cs_count")),
        plurality=_as_int(record.get("plurality")),
        fetal_presentation=_as_str(record.get("fetal_presentation")),
        gestational_age_weeks=ga,
        onset_of_labour=_as_str(record.get("onset_of_labour")),
        gestational_age_range=ga_range,
    )


def numeric_values(series: pd.Series) -> pd.Series:
    """float64 values with NaN for anything missing or not a number."""
    values = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        values.to_numpy(dtype=np.float64, na_value=np.nan), index=series.index, dtype=np.float64
    )


def categorical_values(series: pd.Series) -> pd.Series:
    """Object dtype holding strings, with ``np.nan`` for missing (integers keep no ``.0``)."""

    def label(value: object) -> object:
        if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
            return np.nan
        if isinstance(value, float | np.floating) and float(value).is_integer():
            return str(int(value))
        return str(value)

    return pd.Series([label(v) for v in series.astype(object)], index=series.index, dtype=object)


def no_onset_label(result: ClassificationResult) -> str:
    """The ``robson_group_no_onset`` label of one engine result."""
    if result.status == "resolved" and result.group is not None:
        return str(result.group)
    if result.status == "partial":
        return NO_ONSET_MERGES.get(frozenset(result.candidates), NO_ONSET_UNRESOLVED)
    return NO_ONSET_UNRESOLVED  # conflict


def robson_group_no_onset(frame: pd.DataFrame, rule_set: RuleSet | None = None) -> pd.Series:
    """The Robson group computed without onset, as a string category.

    Each row is classified by the engine with ``onset_of_labour`` forced to missing (the GA
    band is used when the exact GA is missing, as in :func:`inputs_from_record`). Resolved
    rows keep their group (only groups 5-10 resolve without onset); a partial row whose
    candidates are exactly {1, 2} is ``"1_2"`` and exactly {3, 4} is ``"3_4"``; every other
    partial or conflict row is ``"partial"``. Never missing.
    """
    rule_set = rule_set if rule_set is not None else load_rule_set()
    columns = [c for c in (*INPUT_FIELDS, *GA_BAND_FIELDS) if c in frame.columns]
    records = cast("list[dict[str, Any]]", frame[columns].to_dict("records"))
    labels = []
    for record in records:
        record["onset_of_labour"] = None
        labels.append(no_onset_label(classify(inputs_from_record(record), rule_set)))
    return pd.Series(labels, index=frame.index, dtype=object, name=ROBSON_NO_ONSET)


def bmi(height: pd.Series, weight: pd.Series) -> pd.Series:
    """Body-mass index; NaN unless both height (cm) and weight (kg) are present."""
    height_cm, weight_kg = numeric_values(height), numeric_values(weight)
    return weight_kg / (height_cm / 100.0) ** 2


def missing_indicator(series: pd.Series) -> pd.Series:
    """1.0 where the value is not recorded, else 0.0."""
    return series.isna().astype(np.float64)


def derived_kind(name: str) -> str | None:
    """The kind of a derived model column, or None when ``name`` is not derived."""
    if name == ROBSON_NO_ONSET:
        return CATEGORICAL
    if name == BMI or name.startswith(MISSING_PREFIX):
        return NUMERIC
    return None


def source_fields(name: str) -> tuple[str, ...]:
    """The canonical fields model column ``name`` is computed from."""
    if name == BMI:
        return BMI_INPUTS
    if name == ROBSON_NO_ONSET:
        return ROBSON_NO_ONSET_INPUTS
    if name.startswith(MISSING_PREFIX):
        return (name.removeprefix(MISSING_PREFIX),)
    return (name,)


def required_inputs(columns: Iterable[str]) -> tuple[str, ...]:
    """Every canonical field the model columns need, in column order, without repeats."""
    return tuple(dict.fromkeys(field for column in columns for field in source_fields(column)))


def model_column(
    name: str, kind: str, frame: pd.DataFrame, rule_set: RuleSet | None = None
) -> pd.Series:
    """Model column ``name`` (of ``kind``) computed row by row from ``frame``.

    Derived columns (``bmi``, ``robson_group_no_onset``, ``is_missing_<field>``) are computed
    from their source fields; a precomputed ``robson_group_no_onset`` column in ``frame`` is
    reused. Every other column is ``frame[name]`` normalised by ``kind``.
    """
    if name == BMI:
        return bmi(frame[BMI_INPUTS[0]], frame[BMI_INPUTS[1]])
    if name == ROBSON_NO_ONSET:
        if ROBSON_NO_ONSET in frame.columns:
            return categorical_values(frame[ROBSON_NO_ONSET])
        return categorical_values(robson_group_no_onset(frame, rule_set))
    if name.startswith(MISSING_PREFIX):
        field = name.removeprefix(MISSING_PREFIX)
        if field not in frame.columns:
            raise ValueError(f"{name}: source field {field!r} is not in the frame")
        return missing_indicator(frame[field])
    if name not in frame.columns:
        raise ValueError(f"column {name!r} is missing")
    if kind == CATEGORICAL:
        return categorical_values(frame[name])
    if kind == NUMERIC:
        return numeric_values(frame[name])
    raise ValueError(f"{name}: unknown kind {kind!r}")


def build_frame(
    frame: pd.DataFrame, columns: Iterable[tuple[str, str]], rule_set: RuleSet | None = None
) -> pd.DataFrame:
    """The model input frame: one :func:`model_column` per ``(name, kind)``, in order."""
    pairs = list(columns)
    if rule_set is None and any(name == ROBSON_NO_ONSET for name, _ in pairs):
        rule_set = load_rule_set()
    data = {name: model_column(name, kind, frame, rule_set) for name, kind in pairs}
    return pd.DataFrame(data, index=frame.index, columns=[name for name, _ in pairs])
