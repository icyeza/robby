import numpy as np
import pandas as pd
import pytest

from robson_engine import COARSE_PRESENTATIONS
from robson_ml.schema import (
    CANONICAL_BASE_COLUMNS,
    COARSE_PRESENTATION_LEVELS,
    CanonicalSchemaError,
    empty_column,
    validate_canonical,
)


def _valid_frame() -> pd.DataFrame:
    n = 2
    df = pd.DataFrame({c: empty_column(c, n) for c in CANONICAL_BASE_COLUMNS})
    df["admission_id"] = ["T1", "T2"]
    df["facility_id"] = ["FAC_A", "FAC_B"]
    df["parity"] = pd.array([0, 2], dtype="Int64")
    df["gestational_age_weeks"] = [39.0, np.nan]
    df["fetal_presentation"] = ["cephalic", None]
    df["mother_key"] = ["MK_0a1b", None]
    df["delivery_date"] = pd.to_datetime(["2023-11-02", None]).astype("datetime64[ns]")
    df["ga_band_lower"] = [38.0, 35.0]
    df["ga_band_upper"] = [40 + 6 / 7, 37 + 6 / 7]
    df["cs"] = pd.array([1, 0], dtype="Int64")
    return df


def test_valid_frame_passes() -> None:
    validate_canonical(_valid_frame())


def test_out_of_range_reports_column_without_value() -> None:
    df = _valid_frame()
    df.loc[0, "gestational_age_weeks"] = 50.5
    with pytest.raises(CanonicalSchemaError) as excinfo:
        validate_canonical(df)
    assert "gestational_age_weeks" in str(excinfo.value)
    assert "50.5" not in str(excinfo.value)


def test_unknown_category_rejected() -> None:
    df = _valid_frame()
    df.loc[0, "fetal_presentation"] = "face"
    with pytest.raises(CanonicalSchemaError, match="fetal_presentation"):
        validate_canonical(df)


def test_unexpected_column_rejected() -> None:
    df = _valid_frame()
    df["birth_weight"] = [3.1, 2.9]
    with pytest.raises(CanonicalSchemaError):
        validate_canonical(df)


def test_duplicate_admission_id_rejected() -> None:
    df = _valid_frame()
    df["admission_id"] = ["T1", "T1"]
    with pytest.raises(CanonicalSchemaError, match="admission_id"):
        validate_canonical(df)


@pytest.mark.parametrize(
    "removed", ["systolic_bp", "proteinuria", "glucose_mmol_l", "omission_reason_systolic_bp"]
)
def test_removed_v1_1_columns_rejected(removed: str) -> None:
    df = _valid_frame()
    df[removed] = [None, None]
    with pytest.raises(CanonicalSchemaError):
        validate_canonical(df)


def test_admitted_at_replaced_by_delivery_date() -> None:
    assert "admitted_at" not in CANONICAL_BASE_COLUMNS
    assert CANONICAL_BASE_COLUMNS[:4] == (
        "admission_id",
        "mother_key",
        "facility_id",
        "delivery_date",
    )


def test_delivery_date_must_be_date_only() -> None:
    df = _valid_frame()
    df.loc[0, "delivery_date"] = pd.Timestamp("2023-11-02 08:30")
    with pytest.raises(CanonicalSchemaError, match="delivery_date") as excinfo:
        validate_canonical(df)
    assert "08:30" not in str(excinfo.value)


def test_non_cephalic_presentation_accepted() -> None:
    df = _valid_frame()
    df.loc[1, "fetal_presentation"] = "non_cephalic"
    validate_canonical(df)


def test_coarse_presentation_levels_match_engine() -> None:
    assert set(COARSE_PRESENTATION_LEVELS) == set(COARSE_PRESENTATIONS)


def test_ga_band_lower_above_upper_rejected() -> None:
    df = _valid_frame()
    df.loc[0, "ga_band_lower"] = 41.0
    df.loc[0, "ga_band_upper"] = 38.0
    with pytest.raises(CanonicalSchemaError, match="ga_band") as excinfo:
        validate_canonical(df)
    assert "41.0" not in str(excinfo.value)


@pytest.mark.parametrize("side", ["ga_band_lower", "ga_band_upper"])
def test_one_sided_ga_band_rejected(side: str) -> None:
    df = _valid_frame()
    df.loc[0, side] = np.nan
    with pytest.raises(CanonicalSchemaError, match="ga_band"):
        validate_canonical(df)


def test_ga_band_absent_on_both_sides_allowed() -> None:
    df = _valid_frame()
    df.loc[0, ["ga_band_lower", "ga_band_upper"]] = np.nan
    validate_canonical(df)


@pytest.mark.parametrize("side", ["ga_band_lower", "ga_band_upper"])
def test_ga_band_out_of_range_rejected(side: str) -> None:
    df = _valid_frame()
    df.loc[1, ["ga_band_lower", "ga_band_upper"]] = [19.0, 46.0]
    with pytest.raises(CanonicalSchemaError, match=side):
        validate_canonical(df)


def test_error_does_not_retain_original_exception() -> None:
    df = _valid_frame()
    df.loc[0, "gestational_age_weeks"] = 50.5
    with pytest.raises(CanonicalSchemaError) as excinfo:
        validate_canonical(df)
    assert excinfo.value.__context__ is None
    assert excinfo.value.__cause__ is None


def test_band_violations_counted_once_per_row() -> None:
    df = _valid_frame()
    df["ga_band_lower"] = [41.0, 40.0]
    df["ga_band_upper"] = [38.0, 36.0]
    with pytest.raises(CanonicalSchemaError) as excinfo:
        validate_canonical(df)
    message = str(excinfo.value)
    assert "ga_band_lower/ga_band_upper / ga_band_lower_le_upper: 2" in message
    assert "admission_id /" not in message
