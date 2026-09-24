"""B1: Robson-group lookup (spec §12.2; the bar the ML must beat). Not tuned."""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from robson_ml.feature_sets import FeatureSpec
from robson_ml.models.base import ModelSpec, no_search, pipeline, register, select_columns

ROBSON_PRIOR_STRENGTH = 10.0
ROBSON_GROUP = "robson_group"
PREDICT_THRESHOLD = 0.5


class RobsonLookup(ClassifierMixin, BaseEstimator):
    """The CS rate per Robson group in the training rows, smoothed toward the overall rate.

    Beta-binomial smoothing: ``rate_g = (events_g + k * overall) / (n_g + k)`` with prior
    strength ``k``. A row whose group is missing (partial or conflict) or unseen in
    training gets the overall training rate.
    """

    def __init__(self, prior_strength: float = ROBSON_PRIOR_STRENGTH) -> None:
        self.prior_strength = prior_strength

    def fit(self, x: pd.DataFrame, y: npt.ArrayLike) -> Self:
        """Estimate the smoothed rate for every group present in ``x["robson_group"]``."""
        groups = pd.Series(pd.DataFrame(x)[ROBSON_GROUP].to_numpy(dtype=object))
        outcome = pd.Series(np.asarray(y, dtype=np.float64))
        self.overall_rate_ = float(outcome.mean())
        known = groups.notna().to_numpy()
        labels = groups[known].astype(str).to_numpy()
        stats = outcome[known].groupby(labels).agg(["sum", "count"])
        k = self.prior_strength
        self.rates_ = {
            str(group): float((row["sum"] + k * self.overall_rate_) / (row["count"] + k))
            for group, row in stats.iterrows()
        }
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, x: pd.DataFrame) -> npt.NDArray[np.float64]:
        """``[1 - rate, rate]`` per row."""
        groups = pd.DataFrame(x)[ROBSON_GROUP].to_numpy(dtype=object)
        q = np.array(
            [
                self.rates_.get(str(g), self.overall_rate_) if pd.notna(g) else self.overall_rate_
                for g in groups
            ],
            dtype=np.float64,
        )
        return np.column_stack([1 - q, q])

    def predict(self, x: pd.DataFrame) -> npt.NDArray[np.int64]:
        """Labels at 0.5 (reporting only)."""
        return (self.predict_proba(x)[:, 1] >= PREDICT_THRESHOLD).astype(np.int64)


def build(params: dict[str, Any], fs: FeatureSpec) -> Pipeline:
    """The lookup, reading only ``robson_group``."""
    if ROBSON_GROUP not in fs.columns:
        raise ValueError("B1 needs robson_group in its feature set")
    return pipeline(select_columns([ROBSON_GROUP]), RobsonLookup())


SPEC = register(
    ModelSpec(
        name="B1",
        family="baseline",
        complexity_rank=0,
        handles_nan=True,
        native_categorical=False,
        needs_scaling=False,
        build=build,
        search_space=no_search,
        grid={},
    )
)
