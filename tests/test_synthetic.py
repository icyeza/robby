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
    assert df["height_cm"].isna().mean() > 0.2
    assert df["cs"].isna().sum() > 0


def test_prelabour_cs_onset_is_almost_always_cesarean() -> None:
    df = make_admissions(3000, seed=5)
    rows = df[(df["onset_of_labour"] == "prelabour_cs") & df["cs"].notna()]
    assert len(rows) > 0
    vaginal = rows["cs"] == 0
    assert 0 < vaginal.sum() < 0.1 * len(rows)  # a few planned-CS-onset vaginal births
    assert rows.loc[vaginal, "prelabour_cs_type"].isna().all()


def test_cs_type_recorded_for_cs_rows_only() -> None:
    """As in the export: the CS type is filled for (almost) all CS, not only pre-labour CS."""
    df = make_admissions(3000, seed=5)
    cs_type = df["prelabour_cs_type"]
    assert cs_type[df["cs"] == 0].isna().all()
    in_labour_cs = (df["cs"] == 1) & df["onset_of_labour"].isin(["spontaneous", "induced"])
    types = cs_type[in_labour_cs]
    assert (types == "emergency").mean() > 0.8
    assert (types == "planned").sum() > 0
    assert 0 < types.isna().sum() < 0.15 * len(types)


def test_delivery_dates_are_dates_in_the_study_period() -> None:
    dates = make_admissions(2000, seed=6)["delivery_date"]
    assert dates.notna().all()
    assert (dates == dates.dt.normalize()).all()
    assert dates.min() >= pd.Timestamp("2023-11-01")
    assert dates.max() <= pd.Timestamp("2024-03-31")


def test_ga_exact_for_about_two_thirds_and_band_from_true_ga() -> None:
    df = make_admissions(4000, seed=7)
    exact, lower, upper = df["gestational_age_weeks"], df["ga_band_lower"], df["ga_band_upper"]
    assert 0.60 < exact.notna().mean() < 0.72
    assert (lower.notna() == upper.notna()).all()
    assert lower.notna().mean() > 0.95
    both = exact.notna() & lower.notna()
    assert (exact[both] >= lower[both]).all() and (exact[both] <= upper[both]).all()
    # 34+0 to 34+6 weeks falls in no band.
    gap = exact.between(34.0, 34 + 6 / 7)
    assert gap.any()
    assert lower[gap].isna().all()
    assert set(zip(lower.dropna().round(3), upper.dropna().round(3), strict=True)) <= {
        (20.0, 33.857),
        (35.0, 37.857),
        (38.0, 40.857),
        (41.0, 45.0),
    }
    assert (exact.isna() & lower.notna()).sum() > 0


def test_presentation_mostly_coarse_when_not_cephalic() -> None:
    presentation = make_admissions(4000, seed=8)["fetal_presentation"]
    non_cephalic = presentation.notna() & (presentation != "cephalic")
    assert 0.015 < non_cephalic.mean() < 0.045
    coarse = (presentation == "non_cephalic").sum()
    precise = presentation.isin(["breech", "transverse", "oblique"]).sum()
    assert coarse > precise > 0
    assert (presentation == "breech").sum() > 0


def test_about_three_percent_share_a_mother_key() -> None:
    df = make_admissions(4000, seed=9)
    keys = df["mother_key"]
    assert keys.notna().all()
    assert keys.str.fullmatch(r"MK_[0-9a-f]{16}").all()
    shared = keys.duplicated(keep=False)
    assert 0.02 < shared.mean() < 0.04
    groups = df[shared].groupby("mother_key")
    assert (groups.size() == 2).all()
    assert (groups["facility_id"].nunique() == 1).all()
    assert groups["delivery_date"].nunique().eq(1).mean() > 0.7
    assert (df.loc[shared, "plurality"].fillna(1) == 1).all()
