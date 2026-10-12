"""The feature schema of a packaged model, and validation of canonical admissions against it.

``feature_schema.json`` lists the model's columns in order (kind and, for categorical columns,
the training levels) and the canonical input fields they are computed from (kind, allowed
range, allowed levels). :func:`normalise_admission` checks one admission against the inputs.
Errors name the field and never the value (values may be patient data).

Two serving rules live here:

- a precise non-cephalic presentation (breech, transverse, oblique) is coarsened to
  ``non_cephalic`` when the model was trained only on the coarse code, *before* the Robson
  group is derived, so the derived group matches what training saw;
- a field listed in ``unseen_allowed`` (``facility_id``) that is missing is scored as the
  unseen level ``__unseen__``: the one-hot encoder ignores it, so no training facility's
  effect is applied (a facility the model has never seen).
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import numpy as np

from robson_engine import COARSE_PRESENTATIONS
from robson_serving.features import CATEGORICAL, GA_BAND_FIELDS, NUMERIC, is_missing

SCHEMA_FORMAT = 1
UNSEEN_LEVEL = "__unseen__"
INTEGER = "integer"
NUMBER = "number"
CATEGORY = "category"
INPUT_KINDS = frozenset({INTEGER, NUMBER, CATEGORY})
COLUMN_KINDS = frozenset({NUMERIC, CATEGORICAL})
PRESENTATION = "fetal_presentation"
NON_CEPHALIC = "non_cephalic"
PREP_STEP = "prep"
CATEGORICAL_TRANSFORMER = "cat"
ENCODER_STEP = "encode"
NUMERIC_TRANSFORMERS = frozenset({"num", "spline"})
BAND = "/".join(GA_BAND_FIELDS)

Range = tuple[float | None, float | None]


class SchemaError(ValueError):
    """An admission does not fit the model's feature schema. Names the field, never the value."""

    def __init__(self, field: str, problem: str) -> None:
        super().__init__(f"{field}: {problem}")
        self.field = field
        self.problem = problem


@dataclass(frozen=True)
class FeatureColumn:
    """One model column: name, kind (numeric | categorical) and training levels."""

    name: str
    kind: str
    levels: tuple[str, ...] | None = None


@dataclass(frozen=True)
class InputField:
    """One canonical input field: kind (integer | number | category), range and levels."""

    name: str
    kind: str
    range: Range | None = None
    levels: tuple[str, ...] | None = None


@dataclass(frozen=True)
class FeatureSchema:
    """The ordered model columns and the canonical inputs they are computed from."""

    columns: tuple[FeatureColumn, ...]
    inputs: tuple[InputField, ...]
    coarsen_presentation: bool = False
    unseen_allowed: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        """Model column names in model order."""
        return tuple(column.name for column in self.columns)

    def to_dict(self) -> dict[str, Any]:
        """The JSON form written to ``feature_schema.json``."""
        return {
            "format": SCHEMA_FORMAT,
            "columns": [
                {"name": c.name, "kind": c.kind, "levels": _list(c.levels)} for c in self.columns
            ],
            "inputs": [
                {
                    "name": f.name,
                    "kind": f.kind,
                    "range": None if f.range is None else list(f.range),
                    "levels": _list(f.levels),
                }
                for f in self.inputs
            ],
            "coarsen_presentation": self.coarsen_presentation,
            "unseen_level": UNSEEN_LEVEL,
            "unseen_allowed": list(self.unseen_allowed),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FeatureSchema:
        """Parse and validate the JSON form; raises ValueError on anything unexpected."""
        if data.get("format") != SCHEMA_FORMAT:
            raise ValueError(f"feature schema format must be {SCHEMA_FORMAT}")
        if data.get("unseen_level", UNSEEN_LEVEL) != UNSEEN_LEVEL:
            raise ValueError("feature schema names an unexpected unseen level")
        columns = tuple(
            FeatureColumn(str(c["name"]), str(c["kind"]), _levels(c.get("levels")))
            for c in data["columns"]
        )
        inputs = tuple(
            InputField(
                str(f["name"]), str(f["kind"]), _range(f.get("range")), _levels(f.get("levels"))
            )
            for f in data["inputs"]
        )
        for column in columns:
            if column.kind not in COLUMN_KINDS:
                raise ValueError(f"{column.name}: unknown column kind {column.kind!r}")
        for field in inputs:
            if field.kind not in INPUT_KINDS:
                raise ValueError(f"{field.name}: unknown input kind {field.kind!r}")
        return cls(
            columns,
            inputs,
            bool(data.get("coarsen_presentation", False)),
            tuple(str(name) for name in data.get("unseen_allowed", [])),
        )


def _list(levels: tuple[str, ...] | None) -> list[str] | None:
    return None if levels is None else list(levels)


def _levels(value: Any) -> tuple[str, ...] | None:
    return None if value is None else tuple(str(v) for v in value)


def _range(value: Any) -> Range | None:
    if value is None:
        return None
    low, high = value
    return (None if low is None else float(low), None if high is None else float(high))


def _preprocessor(model: Any) -> Any:
    estimator = getattr(model, "estimator", model)
    try:
        return estimator.named_steps[PREP_STEP]
    except (AttributeError, KeyError) as exc:
        raise ValueError("the model has no 'prep' preprocessing step") from exc


def column_kinds(model: Any) -> dict[str, str]:
    """Each input column's kind, read from the fitted ``prep`` ColumnTransformer."""
    kinds: dict[str, str] = {}
    for name, _, columns in _preprocessor(model).transformers_:
        if name == CATEGORICAL_TRANSFORMER:
            kinds.update({str(c): CATEGORICAL for c in columns})
        elif name in NUMERIC_TRANSFORMERS:
            for column in columns:
                kinds.setdefault(str(column), NUMERIC)
    return kinds


def categorical_levels(model: Any) -> dict[str, tuple[str, ...]]:
    """Training levels per categorical column, read from the fitted encoder."""
    for name, transformer, columns in _preprocessor(model).transformers_:
        if name == CATEGORICAL_TRANSFORMER:
            encoder = transformer.named_steps[ENCODER_STEP]
            return {
                str(column): tuple(str(level) for level in levels)
                for column, levels in zip(columns, encoder.categories_, strict=True)
            }
    return {}


def normalise_admission(admission: Mapping[str, Any], schema: FeatureSchema) -> dict[str, Any]:
    """The schema's input fields of ``admission``, checked and normalised.

    Missing keys and NaN/None/NA are "not recorded" (None, or the unseen level for a field in
    ``unseen_allowed``). Numbers become floats; integers must be whole; ranges and levels are
    enforced; a GA band needs both bounds, lower <= upper. Keys the schema does not list are
    ignored. Raises :class:`SchemaError`.
    """
    record = {
        field.name: _value(field, admission.get(field.name), schema) for field in schema.inputs
    }
    if set(GA_BAND_FIELDS) <= record.keys():
        lower: Any
        upper: Any
        lower, upper = (record[name] for name in GA_BAND_FIELDS)
        if (lower is None) != (upper is None):
            raise SchemaError(BAND, "a gestational-age band needs both bounds or neither")
        if lower is not None and upper is not None and lower > upper:
            raise SchemaError(BAND, "the band's lower bound is above its upper bound")
    return record


def _value(field: InputField, value: Any, schema: FeatureSchema) -> float | str | None:
    if is_missing(value):
        return UNSEEN_LEVEL if field.name in schema.unseen_allowed else None
    if field.kind == CATEGORY:
        return _category(field, value, schema)
    return _number(field, value)


def _number(field: InputField, value: Any) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Real | Decimal):
        raise SchemaError(field.name, "expected a number")
    number = float(value)
    if not math.isfinite(number):
        raise SchemaError(field.name, "expected a finite number")
    if field.kind == INTEGER and not number.is_integer():
        raise SchemaError(field.name, "expected a whole number")
    if field.range is not None:
        low, high = field.range
        if (low is not None and number < low) or (high is not None and number > high):
            raise SchemaError(field.name, "outside the allowed range")
    return number


def _category(field: InputField, value: Any, schema: FeatureSchema) -> str:
    if not isinstance(value, str):
        raise SchemaError(field.name, "expected text")
    label = str(value)
    if (
        field.name == PRESENTATION
        and schema.coarsen_presentation
        and label in COARSE_PRESENTATIONS[NON_CEPHALIC]
    ):
        label = NON_CEPHALIC
    if field.levels is not None and label not in field.levels:
        raise SchemaError(field.name, "not a level the model was trained on")
    return label
