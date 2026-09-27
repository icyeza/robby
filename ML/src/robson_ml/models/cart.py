"""A single decision tree, CART (complexity rank 3). Tuned: depth, leaf size, pruning.

One-hot categoricals, unscaled numerics. Leaves are kept large enough (``min_samples_leaf``)
for their CS rate to be a usable probability rather than 0 or 1.
"""

from __future__ import annotations

from typing import Any

import optuna
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from robson_ml.feature_sets import FeatureSpec
from robson_ml.models.base import ModelSpec, make_preprocessor, pipeline, register, split_params


def search_space(trial: optuna.Trial) -> dict[str, Any]:
    """Maximum depth, minimum leaf size and cost-complexity pruning strength."""
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200, log=True),
        "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-5, 1e-2, log=True),
    }


def build(params: dict[str, Any], fs: FeatureSpec) -> Pipeline:
    """One-hot preprocessing into a seeded ``DecisionTreeClassifier``."""
    hyper, strategy, seed = split_params(params)
    prep = make_preprocessor(fs, strategy, seed, encoding="onehot", scale=False)
    clf = DecisionTreeClassifier(
        max_depth=int(hyper["max_depth"]),
        min_samples_leaf=int(hyper["min_samples_leaf"]),
        ccp_alpha=float(hyper["ccp_alpha"]),
        random_state=seed,
    )
    return pipeline(prep, clf)


SPEC = register(
    ModelSpec(
        name="cart",
        family="glassbox",
        complexity_rank=3,
        handles_nan=False,
        native_categorical=False,
        needs_scaling=False,
        build=build,
        search_space=search_space,
    )
)
