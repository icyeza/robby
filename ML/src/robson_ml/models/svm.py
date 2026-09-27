"""Support vector machine with an RBF kernel (complexity rank 6). Tuned: C and gamma.

``SVC(probability=True)`` would refit five times per fit for its internal Platt scaling,
which makes tuning too slow here. :class:`PlattSVC` fits the SVM once and maps its decision
values to probabilities with a one-feature logistic fit on the same rows. Those training-row
probabilities are somewhat overconfident; the harness recalibrates every model on its own
calibration rows afterwards, so the evaluated probabilities are not affected by this shortcut.
"""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import numpy.typing as npt
import optuna
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from robson_ml.feature_sets import FeatureSpec
from robson_ml.models.base import ModelSpec, make_preprocessor, pipeline, register, split_params


class PlattSVC(ClassifierMixin, BaseEstimator):
    """An RBF ``SVC`` whose decision values are mapped to probabilities by a logistic fit."""

    def __init__(self, C: float = 1.0, gamma: float | str = "scale", seed: int = 0) -> None:  # noqa: N803
        self.C = C
        self.gamma = gamma
        self.seed = seed

    def fit(self, x: Any, y: npt.ArrayLike) -> Self:
        """Fit the SVM, then the logistic map from its decision values to P(CS)."""
        y_arr = np.asarray(y)
        self.svc_ = SVC(C=self.C, gamma=self.gamma, kernel="rbf", random_state=self.seed)
        self.svc_.fit(x, y_arr)
        self.platt_ = LogisticRegression().fit(self._decision(x), y_arr)
        self.classes_ = self.svc_.classes_
        return self

    def _decision(self, x: Any) -> npt.NDArray[np.float64]:
        return np.asarray(self.svc_.decision_function(x), dtype=np.float64).reshape(-1, 1)

    def predict_proba(self, x: Any) -> npt.NDArray[np.float64]:
        """Probabilities of both classes, columns in ``classes_`` order."""
        return np.asarray(self.platt_.predict_proba(self._decision(x)), dtype=np.float64)

    def predict(self, x: Any) -> npt.NDArray[Any]:
        """The more probable class."""
        return np.asarray(self.classes_)[self.predict_proba(x).argmax(axis=1)]


def search_space(trial: optuna.Trial) -> dict[str, Any]:
    """Margin penalty C and RBF width gamma, both log scale."""
    return {
        "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
    }


def build(params: dict[str, Any], fs: FeatureSpec) -> Pipeline:
    """One-hot categoricals and standardised numerics into a :class:`PlattSVC`."""
    hyper, strategy, seed = split_params(params)
    prep = make_preprocessor(fs, strategy, seed, encoding="onehot", scale=True)
    return pipeline(prep, PlattSVC(C=float(hyper["C"]), gamma=float(hyper["gamma"]), seed=seed))


SPEC = register(
    ModelSpec(
        name="svm_rbf",
        family="kernel",
        complexity_rank=6,
        handles_nan=False,
        native_categorical=False,
        needs_scaling=True,
        build=build,
        search_space=search_space,
    )
)
