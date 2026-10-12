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
    shadow_compare,
)
from robson_serving.stats import fit_logistic, logit


def test_numeric_reference_uses_decile_edges_and_a_separate_missing_share() -> None:
    frame = pd.DataFrame({"x": [*(np.arange(100) + 0.5), *([np.nan] * 10)]})
    entry = reference_distributions(frame, {"x": NUMERIC})["features"]["x"]
    assert entry["edges"] == pytest.approx(
        [10.4, 20.3, 30.2, 40.1, 50.0, 59.9, 69.8, 79.7, 89.6], abs=1e-9
    )
    assert [b["members"] for b in entry["buckets"]] == [[f"bin:{i}"] for i in range(10)]
    assert sum(b["proportion"] for b in entry["buckets"]) == pytest.approx(1.0)
    assert entry["missing"] == pytest.approx(10 / 110)


def test_small_buckets_are_merged_into_the_largest() -> None:
    values = ["a"] * 50 + ["b"] * 20 + ["c"] * 3 + [None] * 7
    reference = reference_distributions(
        pd.DataFrame({"s": pd.Series(values, dtype=object)}), {"s": CATEGORICAL}
    )
    entry = reference["features"]["s"]
    buckets = entry["buckets"]
    assert [b["members"] for b in buckets] == [["level:a", "level:c"], ["level:b"]]
    assert [b["proportion"] for b in buckets] == pytest.approx([53 / 73, 20 / 73])
    assert entry["missing"] == pytest.approx(7 / 80)
    assert reference["min_cell"] == 5 and reference["format"] == 2


def test_whole_number_features_get_integer_aligned_edges() -> None:
    rng = np.random.default_rng(1)
    weeks = rng.integers(30, 43, 2000).astype(float)
    frame = pd.DataFrame({"x": np.append(weeks, [np.nan] * 900)})
    entry = reference_distributions(frame, {"x": NUMERIC})["features"]["x"]
    assert entry["integer"] is True
    assert all(e % 1 == 0.5 for e in entry["edges"])
    keys = {int(w): _bin(w, entry["edges"]) for w in range(30, 43)}
    members = {m for b in entry["buckets"] for m in b["members"]}
    assert set(keys.values()) == members  # every bucket holds at least one whole number
    assert all(b["proportion"] > 0 for b in entry["buckets"])


def _bin(value: float, edges: list[float]) -> str:
    return f"bin:{int(np.searchsorted(np.asarray(edges), value, side='left'))}"


def test_psi_uses_observed_values_and_reports_the_missing_shift_separately() -> None:
    rng = np.random.default_rng(2)
    recorded = rng.integers(30, 43, 700).astype(float)
    reference_frame = pd.DataFrame({"x": np.append(recorded, [np.nan] * 300)})
    reference = reference_distributions(reference_frame, {"x": NUMERIC})
    live = pd.DataFrame({"x": rng.integers(30, 43, 400).astype(float)})
    row = feature_drift(live, reference).iloc[0]
    assert row["psi"] < 0.1 and not bool(row["flagged"]) and row["n"] == 400
    assert row["missing_reference"] == pytest.approx(0.3)
    assert row["missing_live"] == 0.0
    assert row["missing_shift"] == pytest.approx(-0.3)
    assert bool(row["missing_flagged"])
    same = feature_drift(reference_frame, reference).iloc[0]
    assert same["psi"] == pytest.approx(0.0, abs=1e-12) and same["n"] == 700
    assert not bool(same["missing_flagged"])


def test_format_1_references_still_drive_the_drift() -> None:
    reference = {
        "format": 1,
        "min_cell": 5,
        "features": {
            "x": {
                "kind": NUMERIC,
                "edges": [1.0],
                "buckets": [
                    {"members": ["bin:0"], "proportion": 0.35},
                    {"members": ["bin:1"], "proportion": 0.35},
                    {"members": ["missing"], "proportion": 0.3},
                ],
            }
        },
    }
    row = feature_drift(pd.DataFrame({"x": [1.0, 1.0, 2.0, 2.0]}), reference).iloc[0]
    assert row["psi"] == pytest.approx(0.0, abs=1e-12)
    assert row["missing_reference"] == pytest.approx(0.3)


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


def _shadow(n: int, seed: int) -> pd.DataFrame:
    return _scored(n, seed).rename(columns={"prob": "p_active"})


def test_identical_models_differ_by_exactly_zero() -> None:
    df = _shadow(400, 3)
    df["p_candidate"] = df["p_active"]
    out = shadow_compare(df, n_boot=100, seed=1).set_index("metric")
    assert set(out.index) == {
        "auc",
        "brier",
        "log_loss",
        "calibration_in_the_large",
        "calibration_slope",
    }
    assert (out["difference"] == 0).all()
    assert (out["ci_low"] == 0).all() and (out["ci_high"] == 0).all()


def test_shadow_compare_is_seeded_and_oriented_candidate_minus_active() -> None:
    df = _shadow(400, 4)
    df["p_candidate"] = np.clip(df["p_active"] + 0.2, 0, 1)
    first = shadow_compare(df, n_boot=100, seed=7)
    pd.testing.assert_frame_equal(first, shadow_compare(df, n_boot=100, seed=7))
    brier = first.set_index("metric").loc["brier"]
    assert brier["difference"] == pytest.approx(brier["candidate"] - brier["active"])
    assert brier["difference"] > 0
    assert brier["ci_low"] <= brier["difference"] <= brier["ci_high"]


def test_shadow_compare_needs_both_outcomes() -> None:
    df = pd.DataFrame({"p_active": [0.2, 0.3], "p_candidate": [0.2, 0.3], "cs": [0, 0]})
    with pytest.raises(ValueError, match="both outcomes"):
        shadow_compare(df)


def test_mostly_whole_numbers_with_a_few_fractions_are_still_integer_aligned() -> None:
    rng = np.random.default_rng(3)
    weeks = rng.integers(36, 43, 1000).astype(float)
    weeks[:80] += 2 / 7  # weeks plus days
    entry = reference_distributions(pd.DataFrame({"x": weeks}), {"x": NUMERIC})["features"]["x"]
    assert entry["integer"] is True and all(e % 1 == 0.5 for e in entry["edges"])
