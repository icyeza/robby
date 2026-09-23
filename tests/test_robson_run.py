import numpy as np
import pandas as pd
import pytest

from robson_engine import load_rule_set
from robson_ml.robson_run import (
    RobsonValidationError,
    classify_frame,
    inputs_from_record,
    validate_classification,
)

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
