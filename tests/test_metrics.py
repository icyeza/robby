import math

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq

from robson_ml.metrics import (
    INSUFFICIENT,
    accuracy_at_05,
    auc_with_ci,
    brier_decomposition,
    calibration_in_the_large,
    calibration_slope,
    evaluate,
    log_loss,
    net_benefit,
    subgroup_metrics,
)


def _grouped(rates: dict[float, int], n: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Rows with prediction p and exactly ``events`` of ``n`` outcomes positive, per p."""
    y, p = [], []
    for prob, events in rates.items():
        y += [1] * events + [0] * (n - events)
        p += [prob] * n
    return np.array(y), np.array(p)


def test_metrics_reference() -> None:
    """Spec §22: metrics match values computed by hand on fixed toy data."""
    # AUC: 3 of the 4 (positive, negative) pairs are ordered correctly.
    auc = auc_with_ci(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]), n_boot=50, seed=1)
    assert auc.auc == pytest.approx(0.75)

    # Perfectly calibrated groups: observed rate equals p in every group, so the MLE is
    # alpha = 0, beta = 1 exactly (both score equations vanish).
    y, p = _grouped({0.2: 2, 0.5: 5, 0.8: 8})
    assert calibration_slope(y, p) == pytest.approx(1.0, abs=1e-6)
    assert calibration_in_the_large(y, p) == pytest.approx(0.0, abs=1e-6)

    # Two groups, p = 0.2 / 0.8 (logit -L / +L, L = log 4), observed 0.5 / 0.8: the
    # saturated fit gives beta = (logit 0.8 - logit 0.5) / 2L = 0.5.
    y, p = _grouped({0.2: 5, 0.8: 8})
    assert calibration_slope(y, p) == pytest.approx(0.5, abs=1e-6)
    big_l = math.log(4)

    def score(a: float) -> float:
        return 1.3 - 1 / (1 + math.exp(-(a - big_l))) - 1 / (1 + math.exp(-(a + big_l)))

    assert calibration_in_the_large(y, p) == pytest.approx(brentq(score, -5, 5), abs=1e-6)
    # One group: CITL is logit(observed) - logit(p) = logit(0.5) - logit(0.2) = log 4.
    y, p = _grouped({0.2: 5})
    assert calibration_in_the_large(y, p) == pytest.approx(math.log(4), abs=1e-6)

    # Brier decomposition with 2 bins: forecasts 0.2 / 0.8, observed 0.5 in each.
    dec = brier_decomposition(np.array([0, 1, 0, 1]), np.array([0.2, 0.2, 0.8, 0.8]), bins=2)
    assert dec["brier"] == pytest.approx((0.04 + 0.64 + 0.64 + 0.04) / 4)
    assert dec["reliability"] == pytest.approx(0.09)
    assert dec["resolution"] == pytest.approx(0.0)
    assert dec["uncertainty"] == pytest.approx(0.25)

    # Net benefit: n = 5, prevalence 0.6.
    y = np.array([1, 1, 0, 0, 1])
    p = np.array([0.9, 0.6, 0.7, 0.2, 0.3])
    nb = net_benefit(y, p, thresholds=np.array([0.25, 0.5])).set_index("threshold")
    # pt = 0.5: TP = 2, FP = 1 -> 2/5 - 1/5 * 1; treat-all 0.6 - 0.4 * 1.
    assert nb.loc[0.5, "model"] == pytest.approx(0.2)
    assert nb.loc[0.5, "treat_all"] == pytest.approx(0.2)
    # pt = 0.25: TP = 3, FP = 1 -> 3/5 - 1/5 * 1/3; treat-all 0.6 - 0.4 / 3.
    assert nb.loc[0.25, "model"] == pytest.approx(0.6 - 0.2 / 3)
    assert nb.loc[0.25, "treat_all"] == pytest.approx(0.6 - 0.4 / 3)
    assert (nb["treat_none"] == 0).all()


def test_brier_decomposition_sums_to_brier() -> None:
    rng = np.random.default_rng(0)
    # Ten distinct forecast values, ten rows each: every quantile bin holds one value,
    # so reliability - resolution + uncertainty equals the Brier score exactly.
    p = np.repeat(np.linspace(0.05, 0.95, 10), 10)
    y = (rng.random(100) < p).astype(int)
    dec = brier_decomposition(y, p)
    assert dec["reliability"] - dec["resolution"] + dec["uncertainty"] == pytest.approx(
        dec["brier"]
    )
    # Continuous forecasts: equal within binning tolerance.
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(int)
    dec = brier_decomposition(y, p)
    assert dec["reliability"] - dec["resolution"] + dec["uncertainty"] == pytest.approx(
        dec["brier"], abs=0.005
    )


def test_default_net_benefit_thresholds() -> None:
    nb = net_benefit(np.array([0, 1, 1, 0]), np.array([0.2, 0.7, 0.6, 0.4]))
    assert list(nb["threshold"]) == pytest.approx(list(np.arange(0.10, 0.901, 0.05)))
    assert len(nb) == 17
    assert list(nb.columns) == ["threshold", "model", "treat_all", "treat_none"]


def test_log_loss_and_accuracy() -> None:
    y = np.array([1, 0, 1, 0])
    p = np.array([0.8, 0.3, 0.4, 0.5])
    expected = -(math.log(0.8) + math.log(0.7) + math.log(0.4) + math.log(0.5)) / 4
    assert log_loss(y, p) == pytest.approx(expected)
    # 0.5 counts as a positive prediction.
    assert accuracy_at_05(y, p) == pytest.approx(0.5)


def test_probabilities_are_clipped_before_logit() -> None:
    y = np.array([0, 1, 1, 1, 0, 0])  # not separable, so the fit is identified
    p = np.array([0.0, 0.3, 1.0, 0.7, 0.4, 0.6])
    assert math.isfinite(calibration_slope(y, p))
    assert math.isfinite(calibration_in_the_large(y, p))
    assert math.isfinite(log_loss(y, p))


def test_auc_ci_is_stratified_and_deterministic() -> None:
    rng = np.random.default_rng(3)
    y = np.array([1] * 30 + [0] * 170)
    p = np.clip(0.3 * y + rng.random(200) * 0.7, 0, 1)
    a = auc_with_ci(y, p, seed=5)
    b = auc_with_ci(y, p, seed=5)
    assert a == b
    assert a.ci_low <= a.auc <= a.ci_high
    assert a.ci_low < a.ci_high


def test_auc_needs_both_classes() -> None:
    with pytest.raises(ValueError):
        auc_with_ci(np.array([1, 1, 1]), np.array([0.2, 0.5, 0.9]), seed=0)


def test_inputs_are_validated() -> None:
    with pytest.raises(ValueError):
        log_loss(np.array([0, 1, 2]), np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError):
        log_loss(np.array([0, 1]), np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError):
        log_loss(np.array([0, 1]), np.array([0.1, np.nan]))


def test_evaluate_bundles_every_metric() -> None:
    rng = np.random.default_rng(4)
    p = rng.random(400)
    y = (rng.random(400) < p).astype(int)
    result = evaluate(y, p, seed=1, n_boot=100)
    for key in (
        "n",
        "n_events",
        "auc",
        "auc_ci_low",
        "auc_ci_high",
        "calibration_slope",
        "calibration_in_the_large",
        "brier",
        "reliability",
        "resolution",
        "uncertainty",
        "log_loss",
        "accuracy_at_05",
        "net_benefit",
    ):
        assert key in result
    assert result["n"] == 400
    assert len(result["net_benefit"]) == 17
    assert result == evaluate(y, p, seed=1, n_boot=100)


def test_subgroup_metrics_reports_insufficient() -> None:
    rng = np.random.default_rng(5)
    n = 400
    group = np.array(["big"] * 300 + ["small"] * 40 + ["rare"] * 60)
    p = rng.random(n)
    y = (rng.random(n) < p).astype(int)
    y[340:400] = 0  # "rare": only 5 events
    y[340:345] = 1
    df = pd.DataFrame({"g": group})
    result = subgroup_metrics(df, y, p, by="g", seed=0, n_boot=50)
    assert result["small"] == INSUFFICIENT
    assert result["rare"] == INSUFFICIENT
    big = result["big"]
    assert isinstance(big, dict)
    assert big["n"] == 300
