"""Random forest (complexity rank 4). Tuned: depth, leaf size and features per split.

One-hot categoricals, unscaled numerics; 300 trees (more changes little at this size).
"""

from __future__ import annotations

from typing import Any

import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from robson_ml.feature_sets import FeatureSpec
from robson_ml.models.base import ModelSpec, make_preprocessor, pipeline, register, split_params

N_TREES = 300
N_JOBS = 1  # little memory: a worker thread ran out of it with 2 (seeded, so same trees)


def search_space(trial: optuna.Trial) -> dict[str, Any]:
    """Maximum depth, minimum leaf size and the share of features tried at each split."""
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 16),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
        "max_features": trial.suggest_float("max_features", 0.1, 0.8),
    }


def build(params: dict[str, Any], fs: FeatureSpec) -> Pipeline:
    """One-hot preprocessing into a seeded ``RandomForestClassifier``."""
    hyper, strategy, seed = split_params(params)
    prep = make_preprocessor(fs, strategy, seed, encoding="onehot", scale=False)
    clf = RandomForestClassifier(
        n_estimators=N_TREES,
        max_depth=int(hyper["max_depth"]),
        min_samples_leaf=int(hyper["min_samples_leaf"]),
        max_features=float(hyper["max_features"]),
        n_jobs=N_JOBS,
        random_state=seed,
    )
    return pipeline(prep, clf)


SPEC = register(
    ModelSpec(
        name="random_forest",
        family="ensemble",
        complexity_rank=4,
        handles_nan=False,
        native_categorical=False,
        needs_scaling=False,
        build=build,
        search_space=search_space,
    )
)
