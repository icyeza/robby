"""Main model: L2 logistic regression (complexity rank 1). Tuned: C (log scale)."""

from __future__ import annotations

from typing import Any

import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from robson_ml.feature_sets import FeatureSpec
from robson_ml.models.base import (
    LOGISTIC_MAX_ITER,
    ModelSpec,
    make_preprocessor,
    pipeline,
    register,
    split_params,
)

C_RANGE = (1e-4, 1e2)


def search_space(trial: optuna.Trial) -> dict[str, Any]:
    """Inverse L2 strength C, log-uniform."""
    return {"C": trial.suggest_float("C", *C_RANGE, log=True)}


def build(params: dict[str, Any], fs: FeatureSpec) -> Pipeline:
    """One-hot categoricals and standardised numerics into an L2 logistic regression."""
    hyper, strategy, seed = split_params(params)
    prep = make_preprocessor(fs, strategy, seed, encoding="onehot", scale=True)
    return pipeline(prep, LogisticRegression(C=float(hyper["C"]), max_iter=LOGISTIC_MAX_ITER))


SPEC = register(
    ModelSpec(
        name="logreg_l2",
        family="linear",
        complexity_rank=1,
        handles_nan=False,
        native_categorical=False,
        needs_scaling=True,
        build=build,
        search_space=search_space,
    )
)
