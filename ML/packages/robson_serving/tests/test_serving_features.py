import numpy as np
import pandas as pd
import pytest

from robson_engine import load_rule_set
from robson_serving.features import (
    CATEGORICAL,
    NUMERIC,
    ROBSON_NO_ONSET,
    bmi,
    build_frame,
    categorical_values,
    derived_kind,
    inputs_from_record,
    is_missing,
    missing_indicator,
    model_column,
    numeric_values,
    required_inputs,
    robson_group_no_onset,
)

RULES = load_rule_set()
BASE: dict[str, object] = {
    "parity": 0,
    "previous_cs_count": 0,
    "plurality": 1,
    "fetal_presentation": "cephalic",
    "gestational_age_weeks": 39.0,
    "ga_band_lower": None,
    "ga_band_upper": None,
}


def _frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parity = rng.integers(0, 4, n).astype(float)
    presentation = pd.Series(rng.choice(["cephalic", "breech", "non_cephalic"], n), dtype=object)
    return pd.DataFrame(
        {
            "parity": parity,
            "previous_cs_count": np.where(parity == 0, 0.0, rng.integers(0, 3, n).astype(float)),
            "plurality": np.where(rng.random(n) < 0.1, 2.0, 1.0),
            "fetal_presentation": presentation.mask(rng.random(n) < 0.1),
            "gestational_age_weeks": np.where(
                rng.random(n) < 0.3, np.nan, np.round(rng.uniform(30, 42, n), 1)
            ),
            "ga_band_lower": np.full(n, np.nan),
            "ga_band_upper": np.full(n, np.nan),
            "height_cm": np.where(rng.random(n) < 0.3, np.nan, rng.uniform(140, 180, n)),
            "weight_kg": rng.uniform(45, 100, n),
            "facility_id": pd.Series(rng.choice(["FAC_A", "FAC_B"], n), dtype=object),
        }
    )


def test_is_missing() -> None:
    assert all(is_missing(v) for v in (None, np.nan, float("nan"), pd.NA, pd.NaT))
    assert not any(is_missing(v) for v in (0, 0.0, "", "x"))


def test_numeric_values_coerce_to_float64() -> None:
    out = numeric_values(pd.Series([1, "2", None, "x", 3.5], dtype=object))
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out.to_numpy(), [1.0, 2.0, np.nan, np.nan, 3.5])


def test_categorical_values_are_strings_with_nan_for_missing() -> None:
    out = categorical_values(pd.Series([1.0, 2.5, "a", None, np.nan, 3], dtype=object))
    assert out.dtype == object
    values = out.tolist()
    assert values[:3] == ["1", "2.5", "a"] and values[5] == "3"
    assert out.isna().tolist() == [False, False, False, True, True, False]


def test_bmi_needs_both_measurements() -> None:
    out = bmi(pd.Series([160.0, None, 150.0]), pd.Series([64.0, 70.0, None]))
    assert out.iloc[0] == pytest.approx(25.0)
    assert out.iloc[1:].isna().all()


def test_missing_indicator() -> None:
    out = missing_indicator(pd.Series([1.0, None, np.nan, 0.0], dtype=object))
    assert out.dtype == np.float64
    assert out.tolist() == [0.0, 1.0, 1.0, 0.0]


@pytest.mark.parametrize(
    ("overrides", "label"),
    [
        ({}, "1_2"),
        ({"parity": 2}, "3_4"),
        ({"parity": 1, "previous_cs_count": 1}, "5"),
        ({"parity": 1, "fetal_presentation": "breech"}, "7"),
        ({"previous_cs_count": 1}, "partial"),  # nulliparous with a previous CS: conflict
        (
            {
                "gestational_age_weeks": None,
                "ga_band_lower": 38.0,
                "ga_band_upper": 40.857142857142854,
            },
            "1_2",
        ),
        (dict.fromkeys(BASE), "partial"),
    ],
)
def test_robson_group_no_onset_ignores_onset(overrides: dict[str, object], label: str) -> None:
    frame = pd.DataFrame([{**BASE, **overrides, "onset_of_labour": "spontaneous"}], dtype=object)
    assert robson_group_no_onset(frame, RULES).tolist() == [label]


def test_inputs_from_record_uses_the_band_only_without_exact_ga() -> None:
    band = {"ga_band_lower": 35.0, "ga_band_upper": 37.857142857142854}
    exact = inputs_from_record({**BASE, **band})
    assert exact.gestational_age_weeks == 39.0 and exact.gestational_age_range is None
    banded = inputs_from_record({**BASE, **band, "gestational_age_weeks": np.nan})
    assert banded.gestational_age_weeks is None
    assert banded.gestational_age_range == (35.0, 37.857142857142854)


def test_required_inputs_follow_column_order() -> None:
    columns = ["bmi", ROBSON_NO_ONSET, "is_missing_education_level", "facility_id", "parity"]
    assert required_inputs(columns) == (
        "height_cm",
        "weight_kg",
        "parity",
        "previous_cs_count",
        "plurality",
        "fetal_presentation",
        "gestational_age_weeks",
        "ga_band_lower",
        "ga_band_upper",
        "education_level",
        "facility_id",
    )


def test_derived_kind() -> None:
    assert derived_kind(ROBSON_NO_ONSET) == CATEGORICAL
    assert derived_kind("bmi") == NUMERIC
    assert derived_kind("is_missing_x") == NUMERIC
    assert derived_kind("parity") is None


def test_model_column_errors() -> None:
    frame = pd.DataFrame({"a": [1.0]})
    with pytest.raises(ValueError, match="'b' is missing"):
        model_column("b", NUMERIC, frame)
    with pytest.raises(ValueError, match="source field 'b'"):
        model_column("is_missing_b", NUMERIC, frame)
    with pytest.raises(ValueError, match="unknown kind"):
        model_column("a", "text", frame)


COLUMNS = [
    ("parity", NUMERIC),
    ("fetal_presentation", CATEGORICAL),
    (ROBSON_NO_ONSET, CATEGORICAL),
    ("bmi", NUMERIC),
    ("is_missing_height_cm", NUMERIC),
    ("facility_id", CATEGORICAL),
]


def test_build_frame_is_row_level() -> None:
    frame = _frame(40, 3)
    whole = build_frame(frame, COLUMNS, RULES)
    pieces = pd.concat([build_frame(frame.iloc[[i]], COLUMNS, RULES) for i in range(len(frame))])
    pd.testing.assert_frame_equal(whole, pieces)
    assert list(whole.columns) == [name for name, _ in COLUMNS]


def test_build_frame_reuses_a_precomputed_robson_column() -> None:
    frame = _frame(20, 4)
    columns = [(ROBSON_NO_ONSET, CATEGORICAL)]
    computed = build_frame(frame, columns, RULES)
    precomputed = frame.assign(**{ROBSON_NO_ONSET: computed[ROBSON_NO_ONSET]})
    given = build_frame(precomputed, columns, RULES)
    pd.testing.assert_frame_equal(computed, given)
