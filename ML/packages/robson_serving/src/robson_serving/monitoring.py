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
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from robson_serving.features import CATEGORICAL, NUMERIC, is_missing

FloatArray = npt.NDArray[np.float64]

REFERENCE_FORMAT = 1
REFERENCE_BINS = 10
MIN_CELL = 5
MISSING_KEY = "missing"
PSI_THRESHOLD = 0.2
PSI_FLOOR = 1e-4
DRIFT_COLUMNS = ("feature", "n", "psi", "flagged")


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
