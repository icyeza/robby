"""Calibration at scoring time: the calibrators and the calibrated model wrapper.

These are the classes a packaged ``model.joblib`` pickles, so they live in the serving
runtime. Choosing and fitting a calibrator on the calibration split is training code and stays
in :mod:`robson_ml.calibration`, which re-exports everything defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Self

import numpy as np
import numpy.typing as npt
from sklearn.isotonic import IsotonicRegression

from robson_serving.stats import fit_logistic, logit

FloatArray = npt.NDArray[np.float64]

PREDICT_THRESHOLD = 0.5


def _platt_fit(y: FloatArray, x: FloatArray) -> FloatArray:
    """``[intercept, slope]`` of the Platt regression.

    Fitting is a training operation. Where statsmodels is installed (the training environment)
    its GLM is used, so refits reproduce the runs already recorded with it bit for bit;
    elsewhere the numpy fit of :mod:`robson_serving.stats` (same maximum-likelihood estimate)
    is used. Scoring (``transform``) never needs either.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        return fit_logistic(y, x)
    design = np.column_stack([np.ones(len(x)), np.asarray(x, dtype=np.float64)])
    model = sm.GLM(np.asarray(y, dtype=np.float64), design, family=sm.families.Binomial())
    return np.asarray(model.fit().params, dtype=np.float64)


class Calibrator(Protocol):
    """Maps raw probabilities of the positive class to calibrated probabilities."""

    def fit(self, p: FloatArray, y: FloatArray) -> Self:
        """Fit on raw probabilities ``p`` and binary outcomes ``y``."""
        ...

    def transform(self, p: FloatArray) -> FloatArray:
        """Calibrated probabilities for raw probabilities ``p``."""
        ...


class IdentityCalibrator:
    """The ``none`` option: returns the raw probabilities unchanged."""

    def fit(self, p: FloatArray, y: FloatArray) -> Self:
        """No-op fit (kept for a uniform interface)."""
        return self

    def transform(self, p: FloatArray) -> FloatArray:
        """The raw probabilities as floats."""
        return np.asarray(p, dtype=np.float64)


class PlattCalibrator:
    """Platt scaling: unpenalised logistic regression of y on logit(p)."""

    def __init__(self) -> None:
        self.intercept_ = 0.0
        self.slope_ = 1.0

    def fit(self, p: FloatArray, y: FloatArray) -> Self:
        """Estimate ``y ~ sigmoid(intercept + slope * logit(p))``."""
        self.intercept_, self.slope_ = (float(v) for v in _platt_fit(y, logit(p)))
        return self

    def transform(self, p: FloatArray) -> FloatArray:
        """``sigmoid(intercept + slope * logit(p))``."""
        z = self.intercept_ + self.slope_ * logit(p)
        return np.asarray(1 / (1 + np.exp(-z)), dtype=np.float64)


class IsotonicCalibrator:
    """Isotonic regression of y on p, clipped to [0, 1] and to the fitted range."""

    def __init__(self) -> None:
        self.model_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(self, p: FloatArray, y: FloatArray) -> Self:
        """Fit a non-decreasing step function of p to y."""
        self.model_.fit(np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64))
        return self

    def transform(self, p: FloatArray) -> FloatArray:
        """The fitted step function at ``p``."""
        return np.asarray(self.model_.predict(np.asarray(p, dtype=np.float64)), dtype=np.float64)


@dataclass(frozen=True)
class CalibrationChoice:
    """The selected method, every option's cross-validated Brier, and the refitted calibrator."""

    method: str
    cv_brier: dict[str, float]
    calibrator: Calibrator


class CalibratedModel:
    """A fitted classifier plus its calibrator, exposing the sklearn classifier interface.

    ``predict_proba`` returns ``[1 - q, q]`` where ``q`` is the calibrated probability of
    ``cs = 1``. Picklable with joblib when the wrapped estimator is.
    """

    def __init__(self, estimator: Any, choice: CalibrationChoice) -> None:
        self.estimator = estimator
        self.choice = choice
        self.classes_ = np.array([0, 1])

    def predict_proba(self, x: Any) -> FloatArray:
        """Calibrated class probabilities, shape (n, 2)."""
        raw = np.asarray(self.estimator.predict_proba(x), dtype=np.float64)[:, 1]
        q = np.clip(self.choice.calibrator.transform(raw), 0.0, 1.0)
        return np.column_stack([1 - q, q])

    def predict(self, x: Any) -> npt.NDArray[np.int64]:
        """Class labels at a 0.5 calibrated probability (reporting only)."""
        return (self.predict_proba(x)[:, 1] >= PREDICT_THRESHOLD).astype(np.int64)
