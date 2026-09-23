"""Canonical admission schema (spec §5) as a pandera DataFrameSchema."""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

PRESENTATION_LEVELS = ("cephalic", "breech", "transverse", "oblique")
ONSET_LEVELS = ("spontaneous", "induced", "prelabour_cs")
PRELABOUR_CS_TYPES = ("planned", "emergency")
PROTEINURIA_LEVELS = ("neg", "trace", "1+", "2+", "3+")
YES_NO = ("yes", "no")
SUBGROUP_LEVELS = ("2a", "2b", "4a", "4b", "5a", "5b")
STATUS_LEVELS = ("resolved", "partial", "conflict")

GA_RANGE = (20.0, 45.0)
MATERNAL_AGE_RANGE = (12.0, 55.0)
# Plausibility ranges not fixed by the spec; recorded in DECISIONS.md.
HEIGHT_CM_RANGE = (120.0, 200.0)
WEIGHT_KG_RANGE = (30.0, 200.0)
SYSTOLIC_RANGE = (50.0, 260.0)
DIASTOLIC_RANGE = (20.0, 160.0)
GLUCOSE_RANGE = (1.0, 40.0)

OMISSION_REASON_PATTERN = re.compile(r"^omission_reason_[a-z0-9_]+$")

CANONICAL_DTYPES: dict[str, str] = {
    "admission_id": "object",
    "facility_id": "object",
    "admitted_at": "datetime64[ns]",
    "parity": "Int64",
    "previous_cs_count": "Int64",
    "fetal_presentation": "object",
    "plurality": "Int64",
    "gestational_age_weeks": "float64",
    "onset_of_labour": "object",
    "prelabour_cs_type": "object",
    "maternal_age": "float64",
    "height_cm": "float64",
    "weight_kg": "float64",
    "anc_contacts": "Int64",
    "systolic_bp": "float64",
    "diastolic_bp": "float64",
    "proteinuria": "object",
    "glucose_mmol_l": "float64",
    "preeclampsia_recorded": "object",
    "gdm_recorded": "object",
    "mode_of_delivery": "object",
    "cs": "Int64",
    "recorded_indication": "object",
}
CANONICAL_BASE_COLUMNS: tuple[str, ...] = tuple(CANONICAL_DTYPES)


class CanonicalSchemaError(ValueError):
    """The canonical frame failed validation. The message holds counts only, never values."""


def empty_column(name: str, n: int) -> pd.Series:
    """An all-missing column of the canonical dtype for ``name``."""
    return pd.Series(index=pd.RangeIndex(n), dtype=CANONICAL_DTYPES[name])


def _levels(levels: tuple[str, ...]) -> pa.Check:
    return pa.Check.isin(list(levels))


def _range(bounds: tuple[float, float]) -> pa.Check:
    return pa.Check.in_range(bounds[0], bounds[1])


def _col(
    dtype: str | None,
    *checks: pa.Check,
    nullable: bool = True,
    unique: bool = False,
    required: bool = True,
    regex: bool = False,
) -> pa.Column:
    return pa.Column(
        dtype,
        list(checks),
        nullable=nullable,
        unique=unique,
        required=required,
        regex=regex,
    )


CANONICAL_SCHEMA = pa.DataFrameSchema(
    {
        "admission_id": _col(None, nullable=False, unique=True),
        "facility_id": _col(None, nullable=False),
        "admitted_at": _col("datetime64[ns]"),
        "parity": _col("Int64", pa.Check.ge(0)),
        "previous_cs_count": _col("Int64", pa.Check.ge(0)),
        "fetal_presentation": _col(None, _levels(PRESENTATION_LEVELS)),
        "plurality": _col("Int64", pa.Check.ge(1)),
        "gestational_age_weeks": _col("float64", _range(GA_RANGE)),
        "onset_of_labour": _col(None, _levels(ONSET_LEVELS)),
        "prelabour_cs_type": _col(None, _levels(PRELABOUR_CS_TYPES)),
        "maternal_age": _col("float64", _range(MATERNAL_AGE_RANGE)),
        "height_cm": _col("float64", _range(HEIGHT_CM_RANGE)),
        "weight_kg": _col("float64", _range(WEIGHT_KG_RANGE)),
        "anc_contacts": _col("Int64", pa.Check.ge(0)),
        "systolic_bp": _col("float64", _range(SYSTOLIC_RANGE)),
        "diastolic_bp": _col("float64", _range(DIASTOLIC_RANGE)),
        "proteinuria": _col(None, _levels(PROTEINURIA_LEVELS)),
        "glucose_mmol_l": _col("float64", _range(GLUCOSE_RANGE)),
        "preeclampsia_recorded": _col(None, _levels(YES_NO)),
        "gdm_recorded": _col(None, _levels(YES_NO)),
        "mode_of_delivery": _col(None),
        "cs": _col("Int64", pa.Check.isin([0, 1])),
        "recorded_indication": _col(None),
        "robson_group": _col("Int64", pa.Check.in_range(1, 10), required=False),
        "robson_subgroup": _col(None, _levels(SUBGROUP_LEVELS), required=False),
        "robson_status": _col(None, _levels(STATUS_LEVELS), nullable=False, required=False),
        "robson_candidates": _col(None, nullable=False, required=False),
        "robson_resolving_fields": _col(None, required=False),
        "robson_conflict_fields": _col(None, required=False),
        "rule_set_version": _col(None, nullable=False, required=False),
        OMISSION_REASON_PATTERN.pattern: _col(None, regex=True, required=False),
    },
    strict=True,
)


def validate_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a canonical frame (spec §5).

    Raises:
        CanonicalSchemaError: with per-(column, check) failure counts. The original pandera
            error, which contains row values, is deliberately not chained.
    """
    try:
        validated: pd.DataFrame = CANONICAL_SCHEMA.validate(df, lazy=True)
    except SchemaErrors as exc:
        cases = exc.failure_cases
        counts = Counter(zip(cases["column"].astype(str), cases["check"].astype(str), strict=True))
        summary = "; ".join(f"{col} / {check}: {n}" for (col, check), n in sorted(counts.items()))
        raise CanonicalSchemaError(
            f"canonical schema violations (column / check: n): {summary}"
        ) from None
    return validated
