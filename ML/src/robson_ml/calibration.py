"""Calibration method selection (training side).

The calibrator only ever sees calibration-split rows. Three options are compared by
5-fold stratified cross-validated Brier score within the calibration split (each option is
fitted on four folds and scored on the fifth); ties within 0.001 go to Platt, then ``none``;
the chosen option is then refitted on the whole calibration split.

The runtime classes (calibrators, :class:`CalibrationChoice`, :class:`CalibratedModel`) live
in :mod:`robson_serving.calibration` and are re-exported here, so existing imports keep
working and artefacts pickled under ``robson_ml.calibration`` still load wherever robson_ml
is installed (pickle resolves the names through this module).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from robson_serving.calibration import (
    PREDICT_THRESHOLD,
    CalibratedModel,
    CalibrationChoice,
    Calibrator,
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
)

if TYPE_CHECKING:  # the model registry imports optuna; only its Protocol is needed here
    from robson_ml.models.base import ProbabilisticClassifier

__all__ = [
    "BRIER_TIE",
    "CALIBRATORS",
    "CV_SPLITS",
    "PREDICT_THRESHOLD",
    "TIE_ORDER",
    "CalibratedModel",
    "CalibrationChoice",
    "Calibrator",
    "IdentityCalibrator",
    "IsotonicCalibrator",
    "PlattCalibrator",
    "calibrate",
    "choose_method",
    "select_calibrator",
]

FloatArray = npt.NDArray[np.float64]

CV_SPLITS = 5
BRIER_TIE = 0.001
# Tie preference: Platt first, then none, then isotonic.
TIE_ORDER = ("platt", "none", "isotonic")

CALIBRATORS: Mapping[str, Callable[[], Calibrator]] = {
    "none": IdentityCalibrator,
    "platt": PlattCalibrator,
    "isotonic": IsotonicCalibrator,
}


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


def calibrate(
    estimator: ProbabilisticClassifier,
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
