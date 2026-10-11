"""Monitoring library (spec §18), consumed by the application's reconciliation job.

- :func:`reference_distributions` / :func:`feature_drift`: per-feature bucket proportions of
  the training population (written to ``reference_distributions.json``; aggregates only, no
  bucket under ``MIN_CELL`` rows) and the population stability index against them;
- :func:`rolling_metrics`, :func:`calibration_drift`: discrimination and calibration over a
  recent window, overall and per stratum;
- :func:`shadow_compare`: paired metrics of an active and a candidate model with seeded
  stratified-bootstrap CIs of the differences;
- :func:`expected_theatre_load`: expected cesarean count among women in labour with a 90%
  interval from the exact Poisson-binomial distribution.

Nothing here writes files or prints values.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import roc_auc_score

from robson_serving.features import CATEGORICAL, NUMERIC, is_missing
from robson_serving.stats import ConvergenceError, clip_probabilities, fit_logistic, logit

FloatArray = npt.NDArray[np.float64]

REFERENCE_FORMAT = 1
REFERENCE_BINS = 10
MIN_CELL = 5
MISSING_KEY = "missing"
PSI_THRESHOLD = 0.2
PSI_FLOOR = 1e-4
DRIFT_COLUMNS = ("feature", "n", "psi", "flagged")
MIN_N = 50
MIN_EVENTS = 10
OK = "ok"
INSUFFICIENT = "insufficient"
MISSING_LEVEL = "missing"
STRATA = ("facility", "robson_group")
METRIC_COLUMNS = ("auc", "calibration_in_the_large", "calibration_slope", "brier")
ROLLING_COLUMNS = ("stratum", "level", "n", "events", "status", *METRIC_COLUMNS)
CALIBRATION_DRIFT_THRESHOLD = 0.05
BOOTSTRAP = 1000
SHADOW_CI_LEVEL = 0.95
THEATRE_LEVEL = 0.90
CDF_TOLERANCE = 1e-12


def _numeric_keys(values: pd.Series, edges: Sequence[float]) -> list[str]:
    numbers = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    bins = np.searchsorted(np.asarray(edges, dtype=np.float64), numbers, side="left")
    return [
        MISSING_KEY if math.isnan(value) else f"bin:{int(index)}"
        for value, index in zip(numbers, bins, strict=True)
    ]


def _categorical_keys(values: pd.Series) -> list[str]:
    return [MISSING_KEY if is_missing(v) else f"level:{v}" for v in values.astype(object)]


def _key_order(key: str) -> tuple[int, int, str]:
    if key.startswith("bin:"):
        return (0, int(key.removeprefix("bin:")), "")
    if key == MISSING_KEY:
        return (2, 0, "")
    return (1, 0, key)


def _merge_small(counts: Mapping[str, int], min_cell: int) -> list[tuple[list[str], int]]:
    """Buckets as (member keys, count); any bucket under ``min_cell`` joins the largest one."""
    buckets = [([key], count) for key, count in counts.items() if count > 0]
    while len(buckets) > 1:
        small = next((i for i, (_, count) in enumerate(buckets) if count < min_cell), None)
        if small is None:
            break
        members, count = buckets.pop(small)
        target = max(range(len(buckets)), key=lambda i: buckets[i][1])
        buckets[target] = (buckets[target][0] + members, buckets[target][1] + count)
    return buckets


def reference_distributions(
    frame: pd.DataFrame,
    kinds: Mapping[str, str],
    *,
    n_bins: int = REFERENCE_BINS,
    min_cell: int = MIN_CELL,
) -> dict[str, Any]:
    """Bucket proportions per feature of ``frame`` (the training population's model frame).

    Numeric: interior cut points at the deciles of the recorded values (``edges``; a value on
    an edge falls in the lower bin) plus a ``missing`` bucket. Categorical: one bucket per
    level plus ``missing``. Buckets under ``min_cell`` rows are merged into the largest.
    """
    features: dict[str, Any] = {}
    for name, kind in kinds.items():
        values = frame[name]
        entry: dict[str, Any] = {"kind": kind}
        if kind == NUMERIC:
            recorded = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=np.float64)
            cuts = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
            edges = np.unique(np.quantile(recorded, cuts)) if len(recorded) else np.array([])
            entry["edges"] = [float(edge) for edge in edges]
            keys = _numeric_keys(values, entry["edges"])
        elif kind == CATEGORICAL:
            keys = _categorical_keys(values)
        else:
            raise ValueError(f"{name}: unknown kind {kind!r}")
        counts = Counter(keys)
        ordered = {key: counts[key] for key in sorted(counts, key=_key_order)}
        total = len(keys)
        entry["buckets"] = [
            {"members": members, "proportion": count / total}
            for members, count in _merge_small(ordered, min_cell)
        ]
        features[name] = entry
    return {"format": REFERENCE_FORMAT, "min_cell": min_cell, "features": features}


def population_stability_index(expected: Sequence[float], observed: Sequence[float]) -> float:
    """PSI = sum((o - e) * ln(o / e)), proportions floored at ``PSI_FLOOR``."""
    e = np.maximum(np.asarray(expected, dtype=np.float64), PSI_FLOOR)
    o = np.maximum(np.asarray(observed, dtype=np.float64), PSI_FLOOR)
    return float(np.sum((o - e) * np.log(o / e)))


def feature_drift(
    features_df: pd.DataFrame, reference: Mapping[str, Any], threshold: float = PSI_THRESHOLD
) -> pd.DataFrame:
    """PSI of each reference feature in ``features_df``; flagged when PSI > ``threshold``.

    Values are bucketed exactly as the reference was; a value in no reference bucket (an
    unseen level) counts in an extra bucket whose reference proportion is 0. A feature
    absent from ``features_df`` reports n = 0, PSI NaN, not flagged.
    """
    rows = []
    for name, entry in reference["features"].items():
        if name not in features_df.columns or features_df.empty:
            rows.append({"feature": name, "n": 0, "psi": math.nan, "flagged": False})
            continue
        values = features_df[name]
        if entry["kind"] == NUMERIC:
            keys = _numeric_keys(values, entry["edges"])
        else:
            keys = _categorical_keys(values)
        buckets = entry["buckets"]
        bucket_of = {member: i for i, b in enumerate(buckets) for member in b["members"]}
        counts = np.zeros(len(buckets) + 1)
        for key in keys:
            counts[bucket_of.get(key, len(buckets))] += 1
        expected = [float(b["proportion"]) for b in buckets] + [0.0]
        psi = population_stability_index(expected, counts / len(keys))
        rows.append({"feature": name, "n": len(keys), "psi": psi, "flagged": psi > threshold})
    return pd.DataFrame(rows, columns=list(DRIFT_COLUMNS))


def _auc(y: FloatArray, p: FloatArray) -> float:
    return float(roc_auc_score(y, p))


def _brier(y: FloatArray, p: FloatArray) -> float:
    return float(np.mean((p - y) ** 2))


def _log_loss(y: FloatArray, p: FloatArray) -> float:
    q = clip_probabilities(p)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def _citl(y: FloatArray, p: FloatArray) -> float:
    return float(fit_logistic(y, offset=logit(p))[0])


def _slope(y: FloatArray, p: FloatArray) -> float:
    return float(fit_logistic(y, logit(p))[1])


Metric = Callable[[FloatArray, FloatArray], float]


def _safe(metric: Metric, y: FloatArray, p: FloatArray) -> float:
    try:
        return metric(y, p)
    except (ConvergenceError, ValueError):
        return math.nan


def _in_window(df: pd.DataFrame, window: str | pd.Timedelta, now: Any = None) -> pd.DataFrame:
    """Rows with ``end - window < scored_at <= end``; ``end`` is ``now`` or the latest row."""
    if df.empty:
        return df
    times = pd.to_datetime(df["scored_at"])
    end = pd.Timestamp(now) if now is not None else times.max()
    start = end - pd.Timedelta(window)
    return df.loc[(times > start) & (times <= end)]


def _stratum_row(
    stratum: str, level: str, frame: pd.DataFrame, min_n: int, min_events: int
) -> dict[str, Any]:
    y = frame["cs"].to_numpy(dtype=np.float64)
    p = frame["prob"].to_numpy(dtype=np.float64)
    n, events = len(y), int(y.sum())
    row: dict[str, Any] = {"stratum": stratum, "level": level, "n": n, "events": events}
    if n < min_n or min(events, n - events) < min_events:
        return {**row, "status": INSUFFICIENT, **dict.fromkeys(METRIC_COLUMNS, math.nan)}
    return {
        **row,
        "status": OK,
        "auc": _safe(_auc, y, p),
        "calibration_in_the_large": _safe(_citl, y, p),
        "calibration_slope": _safe(_slope, y, p),
        "brier": _brier(y, p),
    }


def rolling_metrics(
    df: pd.DataFrame,
    window: str | pd.Timedelta,
    *,
    now: Any = None,
    strata: Sequence[str] = STRATA,
    min_n: int = MIN_N,
    min_events: int = MIN_EVENTS,
) -> pd.DataFrame:
    """AUC, calibration-in-the-large, slope and Brier over the window, overall and per stratum.

    Input columns: ``prob``, ``cs``, ``scored_at`` and each stratum column (default
    ``facility`` and ``robson_group``; a missing level is ``"missing"``). A stratum with fewer
    than ``min_n`` rows or fewer than ``min_events`` in either outcome class is
    ``"insufficient"`` with NaN metrics (spec §11.4).
    """
    recent = _in_window(df, window, now)
    rows = [_stratum_row("all", "all", recent, min_n, min_events)]
    for column in strata:
        levels = recent[column].astype(object).where(recent[column].notna(), MISSING_LEVEL)
        labels = levels.astype(str)
        for level in sorted(labels.unique()):
            rows.append(
                _stratum_row(column, level, recent.loc[labels == level], min_n, min_events)
            )
    return pd.DataFrame(rows, columns=list(ROLLING_COLUMNS))


@dataclass(frozen=True)
class CalibrationDrift:
    """Mean predicted probability against the observed cesarean rate over a window."""

    n: int
    mean_probability: float
    observed_rate: float
    difference: float
    flagged: bool


def calibration_drift(
    df: pd.DataFrame,
    window: str | pd.Timedelta,
    threshold: float = CALIBRATION_DRIFT_THRESHOLD,
    *,
    now: Any = None,
) -> CalibrationDrift:
    """Flags when |mean(prob) - mean(cs)| exceeds ``threshold`` over the window."""
    recent = _in_window(df, window, now)
    if recent.empty:
        return CalibrationDrift(0, math.nan, math.nan, math.nan, False)
    mean_p = float(recent["prob"].mean())
    rate = float(recent["cs"].mean())
    difference = mean_p - rate
    return CalibrationDrift(len(recent), mean_p, rate, difference, abs(difference) > threshold)


SHADOW_METRICS: dict[str, Metric] = {
    "auc": _auc,
    "brier": _brier,
    "log_loss": _log_loss,
    "calibration_in_the_large": _citl,
    "calibration_slope": _slope,
}


def shadow_compare(
    df: pd.DataFrame,
    *,
    n_boot: int = BOOTSTRAP,
    seed: int = 0,
    level: float = SHADOW_CI_LEVEL,
) -> pd.DataFrame:
    """Paired metrics of the active and candidate models on the same admissions.

    Input columns: ``p_active``, ``p_candidate``, ``cs``. One row per metric with both values,
    ``difference`` = candidate - active and a percentile CI of the difference from a seeded
    stratified bootstrap (cases and controls resampled separately, the same rows for both
    models). Raises ValueError unless both outcomes are present.
    """
    y = df["cs"].to_numpy(dtype=np.float64)
    active = df["p_active"].to_numpy(dtype=np.float64)
    candidate = df["p_candidate"].to_numpy(dtype=np.float64)
    positives, negatives = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(positives) == 0 or len(negatives) == 0:
        raise ValueError("shadow comparison needs both outcomes")
    rng = np.random.default_rng(seed)
    boots = {name: np.full(n_boot, math.nan) for name in SHADOW_METRICS}
    for b in range(n_boot):
        rows = np.concatenate(
            [
                rng.choice(positives, size=len(positives)),
                rng.choice(negatives, size=len(negatives)),
            ]
        )
        for name, metric in SHADOW_METRICS.items():
            boots[name][b] = _safe(metric, y[rows], candidate[rows]) - _safe(
                metric, y[rows], active[rows]
            )
    tail = (1 - level) / 2 * 100
    out = []
    for name, metric in SHADOW_METRICS.items():
        a, c = _safe(metric, y, active), _safe(metric, y, candidate)
        finite = boots[name][np.isfinite(boots[name])]
        low, high = np.percentile(finite, [tail, 100 - tail]) if len(finite) else (math.nan,) * 2
        out.append(
            {
                "metric": name,
                "active": a,
                "candidate": c,
                "difference": c - a,
                "ci_low": float(low),
                "ci_high": float(high),
                "n": len(y),
            }
        )
    return pd.DataFrame(out)


class TheatreLoad(NamedTuple):
    """Expected cesarean count with an equal-tailed interval (counts)."""

    expected: float
    low: int
    high: int


def _probabilities(probs: npt.ArrayLike) -> FloatArray:
    p = np.asarray(probs, dtype=np.float64).ravel()
    if not np.all(np.isfinite(p)) or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be finite and within [0, 1]")
    return p


def poisson_binomial_pmf(probs: npt.ArrayLike) -> FloatArray:
    """Exact distribution of the number of cesareans (dynamic programming, O(n^2))."""
    p = _probabilities(probs)
    pmf = np.zeros(len(p) + 1)
    pmf[0] = 1.0
    for q in p:
        shifted = np.zeros_like(pmf)
        shifted[1:] = pmf[:-1] * q
        pmf = pmf * (1.0 - q) + shifted
    return pmf


def expected_theatre_load(probs: npt.ArrayLike, level: float = THEATRE_LEVEL) -> TheatreLoad:
    """Expected count (sum of p) and the ``level`` interval from the Poisson-binomial.

    ``low`` is the smallest k with P(X <= k) >= (1 - level)/2 and ``high`` the smallest k with
    P(X <= k) >= 1 - (1 - level)/2 (90%: the 5th and 95th percentiles).
    """
    p = _probabilities(probs)
    cdf = np.cumsum(poisson_binomial_pmf(p))
    tail = (1 - level) / 2
    low = int(np.searchsorted(cdf, tail - CDF_TOLERANCE, side="left"))
    high = int(np.searchsorted(cdf, 1 - tail - CDF_TOLERANCE, side="left"))
    return TheatreLoad(float(p.sum()), min(low, len(p)), min(high, len(p)))
