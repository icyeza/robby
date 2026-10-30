"""Numeric helpers shared by calibration and monitoring (numpy only, no statsmodels)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

PROB_CLIP = 1e-6
NEWTON_MAX_ITER = 100
NEWTON_TOL = 1e-10
# Singular values below this fraction of the largest are treated as zero in the Newton step. NumPy's
# default (about 1e-16 here) can keep rounding noise from a rank-deficient design, and the steps
# then never settle, depending on the platform's linear-algebra library.
PINV_RCOND = 1e-10


class ConvergenceError(ArithmeticError):
    """A maximum-likelihood fit did not converge (for example, perfectly separated outcomes)."""


def clip_probabilities(p: npt.ArrayLike) -> FloatArray:
    """Clip probabilities to [PROB_CLIP, 1 - PROB_CLIP]."""
    return np.clip(np.asarray(p, dtype=np.float64), PROB_CLIP, 1 - PROB_CLIP)


def logit(p: npt.ArrayLike) -> FloatArray:
    """Logit of the clipped probabilities."""
    clipped = clip_probabilities(p)
    return np.log(clipped / (1 - clipped))


def sigmoid(z: npt.ArrayLike) -> FloatArray:
    """The logistic function, computed stably as (1 + tanh(z/2)) / 2."""
    values = np.asarray(z, dtype=np.float64)
    return np.asarray(0.5 * (1.0 + np.tanh(0.5 * values)), dtype=np.float64)


def roc_auc(y: npt.ArrayLike, p: npt.ArrayLike) -> float:
    """ROC AUC as the Mann-Whitney statistic on average ranks (ties count one half).

    Equal to scikit-learn's ``roc_auc_score`` and much cheaper inside bootstrap loops. Raises
    ValueError unless both outcome classes are present.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    p_arr = np.asarray(p, dtype=np.float64)
    positive = y_arr == 1
    n_pos = int(positive.sum())
    n_neg = len(y_arr) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs at least one positive and one negative outcome")
    _, tie_group, tie_counts = np.unique(p_arr, return_inverse=True, return_counts=True)
    average_rank = np.cumsum(tie_counts) - (tie_counts - 1) / 2.0
    ranks = average_rank[tie_group]
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# robson_ml.metrics keeps a statsmodels GLM version for training-side reports: serving must not
# depend on statsmodels, and the two differ on separated or near-separated outcomes.
def fit_logistic(
    y: npt.ArrayLike, x: npt.ArrayLike | None = None, offset: npt.ArrayLike | None = None
) -> FloatArray:
    """Unpenalised maximum-likelihood logistic regression of ``y`` on an intercept.

    Inputs: binary ``y``; an optional single covariate ``x``; an optional fixed ``offset`` on
    the linear predictor. Output: ``[alpha]`` or ``[alpha, beta]``. Newton-Raphson from zero
    with pseudo-inverse steps; raises :class:`ConvergenceError` when the estimates diverge or
    the steps do not shrink below ``NEWTON_TOL`` within ``NEWTON_MAX_ITER``.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    columns = [np.ones_like(y_arr)]
    if x is not None:
        columns.append(np.asarray(x, dtype=np.float64))
    design = np.column_stack(columns)
    fixed = np.zeros_like(y_arr) if offset is None else np.asarray(offset, dtype=np.float64)
    beta = np.zeros(design.shape[1])
    for _ in range(NEWTON_MAX_ITER):
        mu = sigmoid(design @ beta + fixed)
        weight = mu * (1.0 - mu)
        gradient = design.T @ (y_arr - mu)
        hessian = design.T @ (design * weight[:, None])
        # Pseudo-inverse step (as statsmodels' IRLS): a constant covariate, e.g. a baseline that
        # predicts one probability for everyone, gives the minimum-norm solution, not an error.
        step = np.linalg.pinv(hessian, rcond=PINV_RCOND, hermitian=True) @ gradient
        beta = beta + step
        if not np.all(np.isfinite(beta)):
            raise ConvergenceError("logistic fit diverged")
        if float(np.max(np.abs(step))) < NEWTON_TOL:
            return np.asarray(beta, dtype=np.float64)
    raise ConvergenceError("logistic fit did not converge")
