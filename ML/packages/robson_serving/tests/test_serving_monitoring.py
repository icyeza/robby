import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from robson_serving.features import CATEGORICAL, NUMERIC
from robson_serving.monitoring import (
    calibration_drift,
    feature_drift,
    reference_distributions,
    rolling_metrics,
)
from robson_serving.stats import fit_logistic, logit


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


def _scored(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prob = rng.uniform(0.05, 0.95, n)
    return pd.DataFrame(
        {
            "prob": prob,
            "cs": (rng.random(n) < prob).astype(int),
            "facility": pd.Series(rng.choice(["FAC_A", "FAC_B"], n), dtype=object),
            "robson_group": rng.choice([1, 3, 5], n),
            "scored_at": pd.Timestamp("2026-01-01")
            + pd.to_timedelta(rng.integers(0, 60, n), unit="D"),
        }
    )


def test_rolling_metrics_match_a_direct_computation_in_the_window() -> None:
    df = _scored(2000, 1)
    out = rolling_metrics(df, "30D")
    end = df["scored_at"].max()
    recent = df[df["scored_at"] > end - pd.Timedelta("30D")]
    overall = out[out["stratum"] == "all"].iloc[0]
    assert overall["n"] == len(recent) and overall["status"] == "ok"
    y = recent["cs"].to_numpy(float)
    p = recent["prob"].to_numpy(float)
    assert overall["auc"] == pytest.approx(roc_auc_score(y, p))
    assert overall["brier"] == pytest.approx(np.mean((p - y) ** 2))
    assert overall["calibration_slope"] == pytest.approx(fit_logistic(y, logit(p))[1])
    assert overall["calibration_in_the_large"] == pytest.approx(
        fit_logistic(y, offset=logit(p))[0]
    )
    assert set(out.loc[out["stratum"] == "facility", "level"]) == {"FAC_A", "FAC_B"}
    assert set(out.loc[out["stratum"] == "robson_group", "level"]) == {"1", "3", "5"}


def test_small_strata_are_insufficient() -> None:
    df = _scored(2000, 2)
    df.loc[df.index[:20], "facility"] = "FAC_SMALL"
    df.loc[df.index[20:25], "facility"] = None
    out = rolling_metrics(df, "365D").set_index(["stratum", "level"])
    assert out.loc[("facility", "FAC_SMALL"), "status"] == "insufficient"
    assert np.isnan(out.loc[("facility", "FAC_SMALL"), "auc"])
    assert out.loc[("facility", "missing"), "n"] == 5


def test_calibration_drift_flags_a_gap_over_the_threshold() -> None:
    t = pd.Timestamp("2026-03-01")
    df = pd.DataFrame({"prob": [0.3] * 10, "cs": [1] * 4 + [0] * 6, "scored_at": [t] * 10})
    drift = calibration_drift(df, "7D")
    assert drift.n == 10 and drift.difference == pytest.approx(-0.1) and drift.flagged
    assert not calibration_drift(df, "7D", threshold=0.2).flagged
    old = calibration_drift(df, "7D", now=t + pd.Timedelta("30D"))
    assert old.n == 0 and not old.flagged and np.isnan(old.difference)
