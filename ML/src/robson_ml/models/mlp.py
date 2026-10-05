"""Main neural model: a multilayer perceptron (complexity rank 7).

scikit-learn's ``MLPClassifier`` is used instead of torch, to avoid a GPU/torch dependency;
one neural model (MLP or FT-Transformer) is enough. One-hot categoricals and
standardised numerics; internal early stopping on a 10% validation split of the fit rows.
"""

from __future__ import annotations

from typing import Any

import optuna
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from robson_ml.feature_sets import FeatureSpec
from robson_ml.models.base import ModelSpec, make_preprocessor, pipeline, register, split_params

HIDDEN_LAYERS = {"32": (32,), "64": (64,), "64-32": (64, 32), "128-64": (128, 64)}
MAX_ITER = 300


def search_space(trial: optuna.Trial) -> dict[str, Any]:
    """Architecture, L2 strength ``alpha`` and the initial learning rate."""
    return {
        "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", list(HIDDEN_LAYERS)),
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
    }


def build(params: dict[str, Any], fs: FeatureSpec) -> Pipeline:
    """One-hot + standardised preprocessing into a seeded, early-stopped MLPClassifier."""
    hyper, strategy, seed = split_params(params)
    prep = make_preprocessor(fs, strategy, seed, encoding="onehot", scale=True)
    clf = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS[str(hyper["hidden_layer_sizes"])],
        alpha=float(hyper["alpha"]),
        learning_rate_init=float(hyper["learning_rate_init"]),
        early_stopping=True,
        max_iter=MAX_ITER,
        random_state=seed,
    )
    return pipeline(prep, clf)


SPEC = register(
    ModelSpec(
        name="mlp",
        family="neural",
        complexity_rank=7,
        handles_nan=False,
        native_categorical=False,
        needs_scaling=True,
        build=build,
        search_space=search_space,
    )
)
