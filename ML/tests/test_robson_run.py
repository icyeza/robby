import numpy as np
import pandas as pd
import pytest

from robson_engine import load_rule_set
from robson_ml.robson_run import (
    BOUNDARY_GA_HIGH,
    HANDCHECK_COLUMNS,
    RobsonValidationError,
    classify_frame,
    handcheck_sample,
    inputs_from_record,
    validate_classification,
)
from tests.synthetic import make_admissions

RULES = load_rule_set()


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "admission_id": ["A", "B", "C", "D"],
            "facility_id": ["F1", "F1", "F2", "F2"],
            "parity": pd.array([0, 1, 0, 2], dtype="Int64"),
            "previous_cs_count": pd.array([0, 1, 1, 2], dtype="Int64"),
            "plurality": pd.array([1, 1, 1, 1], dtype="Int64"),
            "fetal_presentation": ["cephalic", "cephalic", "cephalic", "cephalic"],
            "gestational_age_weeks": [39.0, np.nan, 40.0, 38.0],
            "onset_of_labour": ["spontaneous", "spontaneous", "spontaneous", None],
        }
    )


def test_inputs_from_record_converts_missing_markers() -> None:
    inputs = inputs_from_record(
        {
            "parity": pd.NA,
            "previous_cs_count": np.int64(1),
            "plurality": 1.0,
            "fetal_presentation": np.nan,
            "gestational_age_weeks": np.float64(38.5),
            "onset_of_labour": None,
        }
    )
    assert inputs.parity is None
    assert inputs.previous_cs_count == 1
    assert inputs.plurality == 1
    assert inputs.fetal_presentation is None
    assert inputs.gestational_age_weeks == 38.5


def test_inputs_from_record_uses_band_only_when_exact_ga_missing() -> None:
    band_only = inputs_from_record(
        {"gestational_age_weeks": np.nan, "ga_band_lower": 38.0, "ga_band_upper": 40 + 6 / 7}
    )
    assert band_only.gestational_age_weeks is None
    assert band_only.gestational_age_range == (38.0, 40 + 6 / 7)
    exact = inputs_from_record(
        {"gestational_age_weeks": 39.0, "ga_band_lower": 38.0, "ga_band_upper": 40 + 6 / 7}
    )
    assert exact.gestational_age_weeks == 39.0
    assert exact.gestational_age_range is None
    one_sided = inputs_from_record(
        {"gestational_age_weeks": None, "ga_band_lower": 38.0, "ga_band_upper": np.nan}
    )
    assert one_sided.gestational_age_range is None
    assert inputs_from_record({"parity": 0}).gestational_age_range is None


def _coarse_frame() -> pd.DataFrame:
    """E: exact GA missing, band >= 37 -> resolves. F: non-cephalic, everything else
    recorded -> partial (6 or 9). G: band straddling 37 -> partial (1 or 10)."""
    df = _frame()
    df["ga_band_lower"] = [np.nan] * 4
    df["ga_band_upper"] = [np.nan] * 4
    extra = pd.DataFrame(
        {
            "admission_id": ["E", "F", "G"],
            "facility_id": ["F1", "F2", "F2"],
            "parity": pd.array([0, 0, 0], dtype="Int64"),
            "previous_cs_count": pd.array([0, 0, 0], dtype="Int64"),
            "plurality": pd.array([1, 1, 1], dtype="Int64"),
            "fetal_presentation": ["cephalic", "non_cephalic", "cephalic"],
            "gestational_age_weeks": [np.nan, 39.0, np.nan],
            "ga_band_lower": [38.0, np.nan, 35.0],
            "ga_band_upper": [40 + 6 / 7, np.nan, 37 + 6 / 7],
            "onset_of_labour": ["spontaneous", "spontaneous", "spontaneous"],
        }
    )
    return pd.concat([df, extra], ignore_index=True)


def test_classify_frame_uses_ga_band_when_exact_missing() -> None:
    out = classify_frame(_coarse_frame(), RULES).set_index("admission_id")
    assert out.loc["E", "robson_status"] == "resolved"
    assert out.loc["E", "robson_group"] == 1
    assert out.loc["F", "robson_status"] == "partial"
    assert out.loc["F", "robson_candidates"] == [6, 9]
    assert out.loc["G", "robson_status"] == "partial"
    assert out.loc["G", "robson_candidates"] == [1, 10]


def test_classify_frame_without_band_columns_ignores_bands() -> None:
    out = classify_frame(_frame(), RULES)
    assert out["robson_status"].tolist() == ["resolved", "partial", "conflict", "resolved"]


def test_coarse_records_are_not_complete_inputs() -> None:
    summary = validate_classification(classify_frame(_coarse_frame(), RULES))
    # Only A and C (a conflict) have all six inputs recorded precisely. F has every input
    # but a coarse presentation: it is partial and must not count as complete-yet-unresolved.
    # E and G have GA as a band only; with F, they are the three coarse records.
    assert summary.n_complete_inputs == 2
    assert summary.n_complete_unresolved == 0
    assert summary.n_coarse_inputs == 3
    assert summary.reconciles
    summary.assert_valid()


def test_complete_precise_partial_still_fails_validation() -> None:
    out = classify_frame(_coarse_frame(), RULES)
    out.loc[out["admission_id"] == "A", "robson_status"] = "partial"
    summary = validate_classification(out)
    assert summary.n_complete_unresolved == 1
    with pytest.raises(RobsonValidationError, match="complete"):
        summary.assert_valid()


def test_handcheck_columns_carry_band_not_mother_key() -> None:
    assert {"ga_band_lower", "ga_band_upper"} <= set(HANDCHECK_COLUMNS)
    assert "mother_key" not in HANDCHECK_COLUMNS


def test_inputs_from_record_rejects_non_integer_parity() -> None:
    with pytest.raises(ValueError):
        inputs_from_record({"parity": 1.5})


def test_classify_frame_adds_engine_columns() -> None:
    out = classify_frame(_frame(), RULES)
    assert out["robson_status"].tolist() == ["resolved", "partial", "conflict", "resolved"]
    assert out["robson_group"].tolist()[0] == 1
    assert pd.isna(out["robson_group"].iloc[1])
    assert out["robson_group"].iloc[3] == 5
    assert out["robson_subgroup"].iloc[3] == "5b"
    assert out["robson_candidates"].iloc[1] == [5, 10]
    assert out["robson_resolving_fields"].iloc[1] == "gestational_age_weeks"
    assert out["robson_conflict_fields"].iloc[2] == "parity;previous_cs_count"
    assert set(out["rule_set_version"]) == {"robson-v1.0"}


def test_validation_summary_reconciles() -> None:
    summary = validate_classification(classify_frame(_frame(), RULES))
    assert summary.n_total == 4
    assert summary.status_counts == {"resolved": 2, "partial": 1, "conflict": 1}
    assert summary.group_counts == {1: 1, 5: 1}
    assert summary.n_complete_inputs == 2
    assert summary.n_complete_unresolved == 0
    assert summary.n_multi_group_resolved == 0
    assert summary.reconciles
    summary.assert_valid()


def test_assert_valid_raises_on_multi_group() -> None:
    out = classify_frame(_frame(), RULES)
    out.at[0, "robson_candidates"] = [1, 2]
    with pytest.raises(RobsonValidationError):
        validate_classification(out).assert_valid()


def test_handcheck_strata() -> None:
    df = classify_frame(make_admissions(3000, seed=11), RULES)
    sample = handcheck_sample(df, seed=7)
    assert sample["admission_id"].is_unique
    per_group = sample["strata"].str.extractall(r"group_(\d+)")[0].astype(int).value_counts()
    assert (per_group <= 15).all()
    boundary = sample[sample["strata"].str.contains("ga_boundary")]
    assert len(boundary) <= 50
    assert boundary["gestational_age_weeks"].between(36.0, BOUNDARY_GA_HIGH).all()
    conflicts = set(df.loc[df["robson_status"] == "conflict", "admission_id"])
    assert conflicts <= set(sample["admission_id"])
    assert {"manual_group", "manual_subgroup", "reviewer_note"} <= set(sample.columns)


def test_handcheck_deterministic() -> None:
    df = classify_frame(make_admissions(2000, seed=12), RULES)
    pd.testing.assert_frame_equal(handcheck_sample(df, seed=1), handcheck_sample(df, seed=1))


def test_handcheck_takes_all_records_of_small_groups() -> None:
    df = classify_frame(make_admissions(3000, seed=11), RULES)
    sample = handcheck_sample(df, seed=7)
    resolved = df[df["robson_status"] == "resolved"]
    sizes = resolved["robson_group"].astype(int).value_counts()
    per_group = sample["strata"].str.extractall(r"group_(\d+)")[0].astype(int).value_counts()
    for group, size in sizes.items():
        assert per_group.get(group, 0) == min(size, 15)
