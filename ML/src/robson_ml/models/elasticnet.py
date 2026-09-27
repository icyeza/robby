"""Elastic-net logistic regression (complexity rank 2). Tuned: C and the L1 ratio.

Same preprocessing as ``logreg_l2`` (one-hot categoricals, standardised numerics); the
``saga`` solver fits the combined L1/L2 penalty, so weak features can be set to exactly 0.
"""

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

C_RANGE = (1e-3, 1e2)


def search_space(trial: optuna.Trial) -> dict[str, Any]:
    """Inverse penalty strength C (log scale) and the L1 share of the penalty."""
    return {
        "C": trial.suggest_float("C", *C_RANGE, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
    }


def build(params: dict[str, Any], fs: FeatureSpec) -> Pipeline:
    """One-hot categoricals and standardised numerics into an elastic-net logistic model."""
    hyper, strategy, seed = split_params(params)
    prep = make_preprocessor(fs, strategy, seed, encoding="onehot", scale=True)
    clf = LogisticRegression(
        C=float(hyper["C"]),
        l1_ratio=float(hyper["l1_ratio"]),
        solver="saga",
        max_iter=LOGISTIC_MAX_ITER,
        random_state=seed,
    )
    return pipeline(prep, clf)


SPEC = register(
    ModelSpec(
        name="elasticnet",
        family="linear",
        complexity_rank=2,
        handles_nan=False,
        native_categorical=False,
        needs_scaling=True,
        build=build,
        search_space=search_space,
    )
)
