"""Canonical admission schema as a pandera DataFrameSchema."""

from __future__ import annotations

from collections import Counter

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

PRECISE_PRESENTATION_LEVELS = ("cephalic", "breech", "transverse", "oblique")
# Coarse codes: "non_cephalic" = known not cephalic, type unknown. Must
# match the engine's COARSE_PRESENTATIONS keys (checked by tests/test_schema.py).
COARSE_PRESENTATION_LEVELS = ("non_cephalic",)
PRESENTATION_LEVELS = PRECISE_PRESENTATION_LEVELS + COARSE_PRESENTATION_LEVELS
ONSET_LEVELS = ("spontaneous", "induced", "prelabour_cs")
PRELABOUR_CS_TYPES = ("planned", "emergency")
YES_NO = ("yes", "no")
SUBGROUP_LEVELS = ("2a", "2b", "4a", "4b", "5a", "5b")
STATUS_LEVELS = ("resolved", "partial", "conflict")

GA_RANGE = (20.0, 45.0)
MATERNAL_AGE_RANGE = (12.0, 55.0)
# Plausibility ranges chosen for this project (not clinical standards).
HEIGHT_CM_RANGE = (120.0, 200.0)
WEIGHT_KG_RANGE = (30.0, 200.0)
GA_BAND_FIELDS = ("ga_band_lower", "ga_band_upper")

CANONICAL_DTYPES: dict[str, str] = {
    "admission_id": "object",
    "mother_key": "object",
    "facility_id": "object",
    "delivery_date": "datetime64[ns]",
    "parity": "Int64",
    "previous_cs_count": "Int64",
    "fetal_presentation": "object",
    "plurality": "Int64",
    "gestational_age_weeks": "float64",
    "ga_band_lower": "float64",
    "ga_band_upper": "float64",
    "onset_of_labour": "object",
    "prelabour_cs_type": "object",
    "maternal_age": "float64",
    "height_cm": "float64",
    "weight_kg": "float64",
    "anc_contacts": "Int64",
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
) -> pa.Column:
    return pa.Column(dtype, list(checks), nullable=nullable, unique=unique, required=required)


def _date_only(values: pd.Series) -> pd.Series:
    """True where the timestamp has no time of day (00:00)."""
    return values == values.dt.normalize()


def _band_pairs(df: pd.DataFrame) -> tuple[pd.Series, pd.Series] | None:
    if not set(GA_BAND_FIELDS) <= set(df.columns):
        return None  # the missing column is reported by its own column check
    return df["ga_band_lower"], df["ga_band_upper"]


def _ga_band_both_or_neither(df: pd.DataFrame) -> pd.Series | bool:
    pair = _band_pairs(df)
    if pair is None:
        return True
    lower, upper = pair
    return lower.notna() == upper.notna()


def _ga_band_ordered(df: pd.DataFrame) -> pd.Series | bool:
    pair = _band_pairs(df)
    if pair is None:
        return True
    lower, upper = pair
    return lower.isna() | upper.isna() | (lower <= upper)


ROW_CHECKS = frozenset({"ga_band_both_or_neither", "ga_band_lower_le_upper"})

CANONICAL_SCHEMA = pa.DataFrameSchema(
    {
        "admission_id": _col(None, nullable=False, unique=True),
        # Salted one-way hash of the raw patient identifier: groups rows of one woman only.
        "mother_key": _col(None),
        "facility_id": _col(None, nullable=False),
        # Date only: a proxy time axis, never a feature.
        "delivery_date": _col("datetime64[ns]", pa.Check(_date_only, name="date_only")),
        "parity": _col("Int64", pa.Check.ge(0)),
        "previous_cs_count": _col("Int64", pa.Check.ge(0)),
        "fetal_presentation": _col(None, _levels(PRESENTATION_LEVELS)),
        "plurality": _col("Int64", pa.Check.ge(1)),
        "gestational_age_weeks": _col("float64", _range(GA_RANGE)),
        "ga_band_lower": _col("float64", _range(GA_RANGE)),
        "ga_band_upper": _col("float64", _range(GA_RANGE)),
        "onset_of_labour": _col(None, _levels(ONSET_LEVELS)),
        "prelabour_cs_type": _col(None, _levels(PRELABOUR_CS_TYPES)),
        "maternal_age": _col("float64", _range(MATERNAL_AGE_RANGE)),
        "height_cm": _col("float64", _range(HEIGHT_CM_RANGE)),
        "weight_kg": _col("float64", _range(WEIGHT_KG_RANGE)),
        "anc_contacts": _col("Int64", pa.Check.ge(0)),
        "preeclampsia_recorded": _col(None, _levels(YES_NO)),
        "gdm_recorded": _col(None, _levels(YES_NO)),
        "mode_of_delivery": _col(None),  # nullable by design
        "cs": _col("Int64", pa.Check.isin([0, 1])),  # nullable by design
        "recorded_indication": _col(None),
        "robson_group": _col("Int64", pa.Check.in_range(1, 10), required=False),
        "robson_subgroup": _col(None, _levels(SUBGROUP_LEVELS), required=False),
        "robson_status": _col(None, _levels(STATUS_LEVELS), nullable=False, required=False),
        "robson_candidates": _col(None, nullable=False, required=False),
        "robson_resolving_fields": _col(None, required=False),
        "robson_conflict_fields": _col(None, required=False),
        "rule_set_version": _col(None, nullable=False, required=False),
    },
    checks=[
        # Row-wise: a GA band is recorded with both bounds or not at all, lower <= upper.
        pa.Check(_ga_band_both_or_neither, name="ga_band_both_or_neither"),
        pa.Check(_ga_band_ordered, name="ga_band_lower_le_upper"),
    ],
    strict=True,
)


def validate_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a canonical frame.

    Raises:
        CanonicalSchemaError: with per-(column, check) failure counts. The original pandera
            error, which contains row values, is deliberately not chained.
    """
    err: CanonicalSchemaError | None = None
    try:
        validated: pd.DataFrame = CANONICAL_SCHEMA.validate(df, lazy=True)
    except SchemaErrors as exc:
        cases = exc.failure_cases
        # pandera repeats a failing row-wise frame check once per column of the row; those
        # are counted once per row, under the band columns they concern.
        row_check = cases["check"].astype(str).isin(ROW_CHECKS)
        counts = Counter(
            zip(
                cases.loc[~row_check, "column"].astype(str),
                cases.loc[~row_check, "check"].astype(str),
                strict=True,
            )
        )
        rows = cases.loc[row_check, ["check", "index"]].astype(str).drop_duplicates()
        for check in rows["check"]:
            counts["/".join(GA_BAND_FIELDS), check] += 1
        summary = "; ".join(f"{col} / {check}: {n}" for (col, check), n in sorted(counts.items()))
        err = CanonicalSchemaError(f"canonical schema violations (column / check: n): {summary}")
        err.__context__ = None
        err.__cause__ = None

    if err is not None:
        raise err from None

    return validated
