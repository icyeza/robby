import numpy as np
import pandas as pd
import pytest

from robson_ml.schema import (
    CANONICAL_BASE_COLUMNS,
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
    df["proteinuria"] = ["neg", "2+"]
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


def test_omission_reason_columns_allowed() -> None:
    df = _valid_frame()
    df["omission_reason_systolic_bp"] = ["equipment_unavailable", None]
    validate_canonical(df)
