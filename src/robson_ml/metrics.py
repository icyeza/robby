"""Evaluation metrics for the readiness model (spec §11.3, §11.4).

Every function takes binary outcomes ``y`` (0/1) and predicted probabilities ``p`` of
``y = 1`` as equal-length array-likes with no missing values. Probabilities are clipped to
[1e-6, 1 - 1e-6] before any logit or log; the Brier score uses the unclipped values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm
from scipy.stats import rankdata

FloatArray = npt.NDArray[np.float64]

PROB_CLIP = 1e-6
AUC_BOOTSTRAP = 1000
AUC_CI_LEVEL = 0.95
BRIER_BINS = 10
DECISION_THRESHOLDS: FloatArray = np.round(np.arange(0.10, 0.90 + 1e-9, 0.05), 2)
ACCURACY_THRESHOLD = 0.5
SUBGROUP_MIN_N = 50
SUBGROUP_MIN_EVENTS = 10
INSUFFICIENT = "insufficient"
MISSING_SUBGROUP = "missing"


@dataclass(frozen=True)
class AucCI:
    """ROC AUC with a percentile bootstrap confidence interval."""

    auc: float
    ci_low: float
    ci_high: float


def _validate(y: npt.ArrayLike, p: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Return (y, p) as float arrays; raise ValueError unless binary y, p in [0, 1]."""
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    p_arr = np.asarray(p, dtype=np.float64).ravel()
    if y_arr.shape != p_arr.shape:
        raise ValueError(f"y and p differ in length: {len(y_arr)} vs {len(p_arr)}")
    if len(y_arr) == 0:
        raise ValueError("no rows to evaluate")
    if np.isnan(y_arr).any() or np.isnan(p_arr).any():
        raise ValueError("y and p must not contain missing values")
    if not np.isin(y_arr, (0.0, 1.0)).all():
        raise ValueError("y must be binary 0/1")
    if (p_arr < 0).any() or (p_arr > 1).any():
        raise ValueError("p must lie in [0, 1]")
    return y_arr, p_arr


def clip_probabilities(p: npt.ArrayLike) -> FloatArray:
    """Clip probabilities to [PROB_CLIP, 1 - PROB_CLIP] (spec §11.3)."""
    return np.clip(np.asarray(p, dtype=np.float64), PROB_CLIP, 1 - PROB_CLIP)


def logit(p: npt.ArrayLike) -> FloatArray:
    """Logit of the clipped probabilities."""
    clipped = clip_probabilities(p)
    return np.log(clipped / (1 - clipped))


def fit_logistic(
    y: npt.ArrayLike, x: npt.ArrayLike | None = None, offset: npt.ArrayLike | None = None
) -> FloatArray:
    """Unpenalised maximum-likelihood logistic regression of ``y`` on an intercept.

    Inputs: binary ``y``; an optional single covariate ``x``; an optional fixed ``offset``
    on the linear predictor. Output: ``[alpha]`` or ``[alpha, beta]``.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    columns = [np.ones_like(y_arr)]
    if x is not None:
        columns.append(np.asarray(x, dtype=np.float64))
    model = sm.GLM(
        y_arr,
        np.column_stack(columns),
        family=sm.families.Binomial(),
        offset=None if offset is None else np.asarray(offset, dtype=np.float64),
    )
    return np.asarray(model.fit().params, dtype=np.float64)


def _auc(y: FloatArray, p: FloatArray) -> float:
    """Mann-Whitney AUC (ties count one half)."""
    positive = y == 1
    n_pos = int(positive.sum())
    n_neg = len(y) - n_pos
    ranks = rankdata(p)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auc_with_ci(
    y: npt.ArrayLike, p: npt.ArrayLike, n_boot: int = AUC_BOOTSTRAP, seed: int = 0
) -> AucCI:
    """ROC AUC with a 95% percentile CI from a stratified bootstrap (spec §11.3).

    Each resample draws positives and negatives separately with replacement, keeping both
    class counts fixed. Raises ValueError unless both classes are present.
    """
    y_arr, p_arr = _validate(y, p)
    pos = np.flatnonzero(y_arr == 1)
    neg = np.flatnonzero(y_arr == 0)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("AUC needs at least one positive and one negative outcome")
    rng = np.random.default_rng(seed)
    y_boot = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    boots = np.empty(n_boot)
    for b in range(n_boot):
        rows = np.concatenate([rng.choice(pos, size=len(pos)), rng.choice(neg, size=len(neg))])
        boots[b] = _auc(y_boot, p_arr[rows])
    tail = (1 - AUC_CI_LEVEL) / 2 * 100
    low, high = np.percentile(boots, [tail, 100 - tail])
    return AucCI(auc=_auc(y_arr, p_arr), ci_low=float(low), ci_high=float(high))


def calibration_slope(y: npt.ArrayLike, p: npt.ArrayLike) -> float:
    """Slope beta in ``logit P(y=1) = alpha + beta * logit(p)`` (unpenalised logistic fit)."""
    y_arr, p_arr = _validate(y, p)
    return float(fit_logistic(y_arr, logit(p_arr))[1])


def calibration_in_the_large(y: npt.ArrayLike, p: npt.ArrayLike) -> float:
    """Intercept alpha in ``logit P(y=1) = alpha + offset(logit(p))`` (0 = calibrated)."""
    y_arr, p_arr = _validate(y, p)
    return float(fit_logistic(y_arr, offset=logit(p_arr))[0])


def brier_decomposition(
    y: npt.ArrayLike, p: npt.ArrayLike, bins: int = BRIER_BINS
) -> dict[str, float]:
    """Brier score and its Murphy decomposition over quantile bins of ``p``.

    Returns ``brier``, ``reliability``, ``resolution`` and ``uncertainty``. Bin edges are
    quantiles of ``p``; equal predictions always share a bin, so there may be fewer than
    ``bins`` bins. ``reliability - resolution + uncertainty`` equals the Brier score exactly
    when predictions are constant within each bin, and approximately otherwise.
    """
    y_arr, p_arr = _validate(y, p)
    edges = np.quantile(p_arr, np.linspace(0, 1, bins + 1))
    bin_id = np.searchsorted(edges[1:-1], p_arr, side="right")
    n = len(y_arr)
    base_rate = y_arr.mean()
    reliability = resolution = 0.0
    for k in np.unique(bin_id):
        in_bin = bin_id == k
        weight = in_bin.sum() / n
        observed = y_arr[in_bin].mean()
        reliability += weight * (p_arr[in_bin].mean() - observed) ** 2
        resolution += weight * (observed - base_rate) ** 2
    return {
        "brier": float(np.mean((p_arr - y_arr) ** 2)),
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(base_rate * (1 - base_rate)),
    }


def net_benefit(
    y: npt.ArrayLike, p: npt.ArrayLike, thresholds: npt.ArrayLike | None = None
) -> pd.DataFrame:
    """Decision-curve net benefit ``TP/n - FP/n * pt/(1 - pt)`` (spec §11.3).

    A row is treated when ``p >= pt``. Returns one row per threshold (default 0.10 to 0.90
    by 0.05) with columns ``threshold``, ``model``, ``treat_all`` and ``treat_none`` (0).
    """
    y_arr, p_arr = _validate(y, p)
    pts = DECISION_THRESHOLDS if thresholds is None else np.asarray(thresholds, dtype=float)
    n = len(y_arr)
    prevalence = y_arr.mean()
    rows = []
    for pt in pts:
        odds = pt / (1 - pt)
        treated = p_arr >= pt
        tp = np.sum(treated & (y_arr == 1))
        fp = np.sum(treated & (y_arr == 0))
        rows.append(
            {
                "threshold": float(pt),
                "model": float(tp / n - fp / n * odds),
                "treat_all": float(prevalence - (1 - prevalence) * odds),
                "treat_none": 0.0,
            }
        )
    return pd.DataFrame(rows, columns=["threshold", "model", "treat_all", "treat_none"])


def log_loss(y: npt.ArrayLike, p: npt.ArrayLike) -> float:
    """Mean negative log-likelihood of the clipped probabilities (the tuning objective)."""
    y_arr, p_arr = _validate(y, p)
    clipped = clip_probabilities(p_arr)
    return float(-np.mean(y_arr * np.log(clipped) + (1 - y_arr) * np.log(1 - clipped)))


def accuracy_at_05(y: npt.ArrayLike, p: npt.ArrayLike) -> float:
    """Accuracy predicting 1 when ``p >= 0.5``; reported only, never a selection criterion."""
    y_arr, p_arr = _validate(y, p)
    return float(np.mean((p_arr >= ACCURACY_THRESHOLD) == (y_arr == 1)))


def evaluate(
    y: npt.ArrayLike, p: npt.ArrayLike, seed: int, n_boot: int = AUC_BOOTSTRAP
) -> dict[str, Any]:
    """Every §11.3 metric in one JSON-serialisable dict.

    Keys: n, n_events, prevalence, auc, auc_ci_low, auc_ci_high, calibration_slope,
    calibration_in_the_large, brier, reliability, resolution, uncertainty, log_loss,
    accuracy_at_05, and net_benefit (a list of per-threshold records).
    """
    y_arr, p_arr = _validate(y, p)
    auc = auc_with_ci(y_arr, p_arr, n_boot=n_boot, seed=seed)
    return {
        "n": len(y_arr),
        "n_events": int(y_arr.sum()),
        "prevalence": float(y_arr.mean()),
        "auc": auc.auc,
        "auc_ci_low": auc.ci_low,
        "auc_ci_high": auc.ci_high,
        "calibration_slope": calibration_slope(y_arr, p_arr),
        "calibration_in_the_large": calibration_in_the_large(y_arr, p_arr),
        **brier_decomposition(y_arr, p_arr),
        "log_loss": log_loss(y_arr, p_arr),
        "accuracy_at_05": accuracy_at_05(y_arr, p_arr),
        "net_benefit": net_benefit(y_arr, p_arr).to_dict(orient="records"),
    }


def subgroup_metrics(
    df: pd.DataFrame,
    y: npt.ArrayLike,
    p: npt.ArrayLike,
    by: str,
    min_n: int = SUBGROUP_MIN_N,
    min_events: int = SUBGROUP_MIN_EVENTS,
    seed: int = 0,
    n_boot: int = AUC_BOOTSTRAP,
) -> dict[str, dict[str, Any] | str]:
    """``evaluate`` per level of ``df[by]`` (spec §11.4); ``y`` and ``p`` align with rows.

    A level with fewer than ``min_n`` rows, or fewer than ``min_events`` rows in either
    outcome class, maps to ``"insufficient"``. Missing levels are grouped as ``"missing"``.
    """
    y_arr, p_arr = _validate(y, p)
    if len(df) != len(y_arr):
        raise ValueError("df, y and p must have the same number of rows")
    levels = df[by].astype(object).where(df[by].notna(), MISSING_SUBGROUP).astype(str)
    level_arr = levels.to_numpy()
    result: dict[str, dict[str, Any] | str] = {}
    for level in sorted(set(level_arr)):
        rows = level_arr == level
        events = int(y_arr[rows].sum())
        if rows.sum() < min_n or min(events, int(rows.sum()) - events) < min_events:
            result[level] = INSUFFICIENT
        else:
            result[level] = evaluate(y_arr[rows], p_arr[rows], seed=seed, n_boot=n_boot)
    return result
