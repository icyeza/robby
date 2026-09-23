import pandas as pd

from robson_engine import load_rule_set
from robson_ml.robson_run import classify_frame
from robson_ml.schema import CANONICAL_BASE_COLUMNS, validate_canonical
from tests.synthetic import FACILITIES, make_admissions


def test_columns_and_schema() -> None:
    df = make_admissions(1500, seed=1)
    assert list(df.columns) == list(CANONICAL_BASE_COLUMNS)
    validate_canonical(df)


def test_deterministic_for_seed() -> None:
    pd.testing.assert_frame_equal(make_admissions(400, seed=3), make_admissions(400, seed=3))


def test_four_facilities_and_plausible_rates() -> None:
    df = classify_frame(make_admissions(4000, seed=2), load_rule_set())
    assert set(df["facility_id"]) == set(FACILITIES)
    assert 0.30 < df["cs"].mean() < 0.65
    rate = df[df["robson_status"] == "resolved"].groupby("robson_group")["cs"].mean()
    assert rate[5] > rate[3]


def test_missingness_and_conflicts_present() -> None:
    df = classify_frame(make_admissions(4000, seed=4), load_rule_set())
    counts = df["robson_status"].value_counts()
    assert counts.get("partial", 0) > 0
    assert counts.get("conflict", 0) > 0
    assert df["systolic_bp"].isna().mean() > 0.2
    assert df["cs"].isna().sum() > 0


def test_prelabour_cs_is_always_cesarean() -> None:
    df = make_admissions(3000, seed=5)
    rows = df[(df["onset_of_labour"] == "prelabour_cs") & df["cs"].notna()]
    assert len(rows) > 0
    assert (rows["cs"] == 1).all()
