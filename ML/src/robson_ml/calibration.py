"""Calibration method selection and the calibrated model wrapper (spec §13.1, v1.1 item 3).

The calibrator only ever sees calibration-split rows. Three options are compared by
5-fold stratified cross-validated Brier score within the calibration split (each option is
fitted on four folds and scored on the fifth); ties within 0.001 go to Platt, then ``none``;
the chosen option is then refitted on the whole calibration split.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from robson_ml.metrics import fit_logistic, logit

FloatArray = npt.NDArray[np.float64]

CV_SPLITS = 5
BRIER_TIE = 0.001
# Tie preference (spec §13.1): Platt first, then none, then isotonic.
TIE_ORDER = ("platt", "none", "isotonic")
PREDICT_THRESHOLD = 0.5


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
        self.intercept_, self.slope_ = (float(v) for v in fit_logistic(y, logit(p)))
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


CALIBRATORS: Mapping[str, Callable[[], Calibrator]] = {
    "none": IdentityCalibrator,
    "platt": PlattCalibrator,
    "isotonic": IsotonicCalibrator,
}


@dataclass(frozen=True)
class CalibrationChoice:
    """The selected method, every option's cross-validated Brier, and the refitted calibrator."""

    method: str
    cv_brier: dict[str, float]
    calibrator: Calibrator


def choose_method(cv_brier: Mapping[str, float]) -> str:
    """The lowest-Brier method, with ties within ``BRIER_TIE`` going to Platt, then none."""
    best = min(cv_brier.values())
    return next(m for m in TIE_ORDER if m in cv_brier and cv_brier[m] <= best + BRIER_TIE)


def select_calibrator(
    y_calib: npt.ArrayLike,
    p_calib: npt.ArrayLike,
    seed: int,
    groups: npt.ArrayLike | None = None,
) -> CalibrationChoice:
    """Choose none / Platt / isotonic by out-of-fold Brier on the calibration split.

    Inputs: outcomes and raw probabilities of the calibration-split rows only; optional
    mother-level ``groups`` (then folds are StratifiedGroupKFold so a woman's rows never
    sit on both sides). Each option is fitted on four folds and scored on the fifth; the
    pooled out-of-fold Brier decides (``choose_method``). The chosen option is refitted on
    the whole calibration split.
    """
    y = np.asarray(y_calib, dtype=np.float64)
    p = np.asarray(p_calib, dtype=np.float64)
    if y.shape != p.shape or len(y) == 0:
        raise ValueError("y_calib and p_calib must be non-empty and of equal length")
    splitter: StratifiedKFold | StratifiedGroupKFold
    if groups is None:
        splitter = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=seed)
        folds = splitter.split(p.reshape(-1, 1), y)
    else:
        splitter = StratifiedGroupKFold(n_splits=CV_SPLITS, shuffle=True, random_state=seed)
        folds = splitter.split(p.reshape(-1, 1), y, np.asarray(groups))
    squared_error = {name: np.zeros(len(y)) for name in CALIBRATORS}
    for train, val in folds:
        for name, factory in CALIBRATORS.items():
            fitted = factory().fit(p[train], y[train])
            squared_error[name][val] = (fitted.transform(p[val]) - y[val]) ** 2
    cv_brier = {name: float(errors.mean()) for name, errors in squared_error.items()}
    method = choose_method(cv_brier)
    return CalibrationChoice(method, cv_brier, CALIBRATORS[method]().fit(p, y))


def _rows(data: Any, idx: npt.NDArray[np.int64]) -> Any:
    """Rows ``idx`` (positions) of a DataFrame, Series or array."""
    return data.iloc[idx] if isinstance(data, pd.DataFrame | pd.Series) else np.asarray(data)[idx]


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


def calibrate(
    estimator: Any,
    x: Any,
    y: Any,
    calib_idx: npt.ArrayLike,
    seed: int,
    groups: npt.ArrayLike | None = None,
) -> CalibratedModel:
    """Select and fit a calibrator for a fitted ``estimator`` on the calibration split.

    Inputs: the fitted estimator; the full feature frame ``x`` and outcomes ``y`` of the
    split frame; ``calib_idx`` (row positions of the calibration split); optional
    mother-level ``groups`` aligned with ``x``. Only rows in ``calib_idx`` are predicted
    and seen by the calibrator.
    """
    idx = np.asarray(calib_idx, dtype=np.int64)
    raw = np.asarray(estimator.predict_proba(_rows(x, idx)), dtype=np.float64)[:, 1]
    y_calib = np.asarray(_rows(y, idx), dtype=np.float64)
    group_calib = None if groups is None else np.asarray(groups)[idx]
    return CalibratedModel(estimator, select_calibrator(y_calib, raw, seed, group_calib))
