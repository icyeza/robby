"""Main model: XGBoost (complexity rank 5).

Native categorical support (pandas ``category`` columns, ``enable_categorical``), raw
unscaled numerics, native NaN handling under M0. Tuned by TPE over learning rate, depth,
subsampling and L2 strength. The number of trees is not searched: during tuning the harness
fits with early stopping on each inner validation fold, and the final refit
uses the mean best number of rounds across the inner folds.
"""

from __future__ import annotations

from typing import Any

import optuna
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from robson_ml.feature_sets import FeatureSpec
from robson_ml.models.base import ModelSpec, make_preprocessor, pipeline, register, split_params

MAX_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 50


def search_space(trial: optuna.Trial) -> dict[str, Any]:
    """Learning rate, depth, row/column subsampling and L2 strength."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def build(params: dict[str, Any], fs: FeatureSpec) -> Pipeline:
    """Native-categorical preprocessing into a seeded ``XGBClassifier`` (hist).

    ``n_estimators`` defaults to ``MAX_ROUNDS`` (the early-stopping ceiling used during
    tuning); the harness passes the early-stopped value for the final refit.
    """
    hyper, strategy, seed = split_params(params)
    prep = make_preprocessor(fs, strategy, seed, encoding="native", scale=False)
    n_estimators = int(hyper.pop("n_estimators", MAX_ROUNDS))
    clf = XGBClassifier(
        n_estimators=n_estimators,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="logloss",
        random_state=seed,
        **hyper,
    )
    return pipeline(prep, clf)


SPEC = register(
    ModelSpec(
        name="xgboost",
        family="tree",
        complexity_rank=5,
        handles_nan=True,
        native_categorical=True,
        needs_scaling=False,
        build=build,
        search_space=search_space,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
)
