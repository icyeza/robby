import numpy as np
import pandas as pd
import pytest

from robson_serving.features import CATEGORICAL, NUMERIC
from robson_serving.monitoring import feature_drift, reference_distributions


def test_numeric_reference_uses_decile_edges_and_a_missing_bucket() -> None:
    frame = pd.DataFrame({"x": [*range(100), *([np.nan] * 10)]})
    entry = reference_distributions(frame, {"x": NUMERIC})["features"]["x"]
    assert entry["edges"] == pytest.approx([9.9, 19.8, 29.7, 39.6, 49.5, 59.4, 69.3, 79.2, 89.1])
    assert [b["members"] for b in entry["buckets"]] == [[f"bin:{i}"] for i in range(10)] + [
        ["missing"]
    ]
    assert sum(b["proportion"] for b in entry["buckets"]) == pytest.approx(1.0)
    assert entry["buckets"][-1]["proportion"] == pytest.approx(10 / 110)


def test_small_buckets_are_merged_into_the_largest() -> None:
    values = ["a"] * 50 + ["b"] * 20 + ["c"] * 3 + [None] * 7
    reference = reference_distributions(
        pd.DataFrame({"s": pd.Series(values, dtype=object)}), {"s": CATEGORICAL}
    )
    buckets = reference["features"]["s"]["buckets"]
    assert [b["members"] for b in buckets] == [["level:a", "level:c"], ["level:b"], ["missing"]]
    assert [b["proportion"] for b in buckets] == pytest.approx([53 / 80, 20 / 80, 7 / 80])
    assert reference["min_cell"] == 5


def test_psi_is_zero_on_the_reference_and_matches_a_hand_value() -> None:
    reference_frame = pd.DataFrame({"s": ["a"] * 50 + ["b"] * 50})
    reference = reference_distributions(reference_frame, {"s": CATEGORICAL})
    same = feature_drift(reference_frame, reference)
    assert same.loc[0, "psi"] == pytest.approx(0.0, abs=1e-12)
    assert not bool(same.loc[0, "flagged"])
    shifted = feature_drift(pd.DataFrame({"s": ["a"] * 90 + ["b"] * 10}), reference)
    assert shifted.loc[0, "psi"] == pytest.approx(0.4 * np.log(1.8) + 0.4 * np.log(5.0))
    assert bool(shifted.loc[0, "flagged"]) and shifted.loc[0, "n"] == 100


def test_unseen_levels_drift_and_absent_features_report_nothing() -> None:
    reference = reference_distributions(
        pd.DataFrame({"s": ["a"] * 10, "x": np.arange(10.0)}), {"s": CATEGORICAL, "x": NUMERIC}
    )
    drift = feature_drift(pd.DataFrame({"s": ["z"] * 10}), reference).set_index("feature")
    assert drift.loc["s", "psi"] > 5 and bool(drift.loc["s", "flagged"])
    assert drift.loc["x", "n"] == 0 and np.isnan(drift.loc["x", "psi"])
    assert not bool(drift.loc["x", "flagged"])


def test_a_value_on_an_edge_falls_in_the_lower_bin() -> None:
    reference = {
        "format": 1,
        "min_cell": 5,
        "features": {
            "x": {
                "kind": NUMERIC,
                "edges": [1.0],
                "buckets": [
                    {"members": ["bin:0"], "proportion": 0.5},
                    {"members": ["bin:1"], "proportion": 0.5},
                ],
            }
        },
    }
    drift = feature_drift(pd.DataFrame({"x": [1.0, 1.0, 2.0, 2.0]}), reference)
    assert drift.loc[0, "psi"] == pytest.approx(0.0, abs=1e-12)
