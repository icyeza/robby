"""Model tracking for the training notebook: per-model performance metrics, confusion
matrices and hyperparameter-tuning histories.

Everything returned is an aggregate: a metric over many admissions, a 2x2 count table with
small cells hidden, or one row per tuning *trial*. Nothing holds a row of data.

Accuracy, precision, recall, specificity and F1 need the probability cut at a threshold;
they are reported for comparison only and never used to select a model (the
pre-registered rule reads AUC and calibration).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

from robson_ml.metrics import brier_decomposition, log_loss
from robson_ml.privacy import SECONDARY, SMALL_CELL_THRESHOLD, SUPPRESSED

THRESHOLD = 0.5
OBSERVED = ("vaginal (observed)", "CS (observed)")
PREDICTED = ("vaginal (predicted)", "CS (predicted)")
THRESHOLD_METRICS = ("accuracy", "precision", "recall", "specificity", "F1")
RESERVED_PARAMS = frozenset({"missing_strategy", "seed"})


def _arrays(
    y: npt.ArrayLike, p: npt.ArrayLike
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    y_arr = np.asarray(y, dtype=np.int64)
    p_arr = np.asarray(p, dtype=np.float64)
    if y_arr.shape != p_arr.shape or y_arr.ndim != 1 or len(y_arr) == 0:
        raise ValueError("y and p must be non-empty 1-d arrays of the same length")
    return y_arr, p_arr


def _ratio(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def threshold_metrics(
    y: npt.ArrayLike, p: npt.ArrayLike, threshold: float = THRESHOLD
) -> dict[str, float]:
    """Accuracy, precision, recall (sensitivity), specificity and F1, predicting CS when
    ``p >= threshold``. An undefined ratio (no predicted or no observed CS) is 0."""
    y_arr, p_arr = _arrays(y, p)
    pred = p_arr >= threshold
    tp = int(np.sum(pred & (y_arr == 1)))
    fp = int(np.sum(pred & (y_arr == 0)))
    fn = int(np.sum(~pred & (y_arr == 1)))
    tn = int(np.sum(~pred & (y_arr == 0)))
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return {
        "accuracy": _ratio(tp + tn, len(y_arr)),
        "precision": precision,
        "recall": recall,
        "specificity": _ratio(tn, tn + fp),
        "F1": _ratio(2 * precision * recall, precision + recall) if precision + recall else 0.0,
    }


def performance_metrics(
    y: npt.ArrayLike, p: npt.ArrayLike, threshold: float = THRESHOLD
) -> dict[str, float]:
    """Pooled AUC, log loss and Brier score of the probabilities, plus
    :func:`threshold_metrics` at ``threshold``."""
    y_arr, p_arr = _arrays(y, p)
    auc = float(roc_auc_score(y_arr, p_arr)) if len(np.unique(y_arr)) == 2 else float("nan")
    return {
        "pooled AUC": auc,
        **threshold_metrics(y_arr, p_arr, threshold),
        "log loss": log_loss(y_arr, p_arr),
        "Brier": float(brier_decomposition(y_arr, p_arr)["brier"]),
    }


def confusion_counts(
    y: npt.ArrayLike, p: npt.ArrayLike, threshold: float = THRESHOLD
) -> pd.DataFrame:
    """The 2x2 confusion matrix (rows observed, columns predicted) at ``threshold``.

    A count of 1-4 is shown as ``"<5"``; because each row adds up to a published class
    total, the other cell of that row is then hidden too (``"*"``). Other counts stay
    integers so that they can be coloured.
    """
    y_arr, p_arr = _arrays(y, p)
    pred = (p_arr >= threshold).astype(np.int64)
    counts = [[int(np.sum((y_arr == obs) & (pred == hat))) for hat in (0, 1)] for obs in (0, 1)]
    rows: list[list[Any]] = []
    for row in counts:
        small = [0 < v < SMALL_CELL_THRESHOLD for v in row]
        if any(small):
            rows.append([SUPPRESSED if s else SECONDARY for s in small])
        else:
            rows.append(list(row))
    return pd.DataFrame(rows, index=list(OBSERVED), columns=list(PREDICTED))


def best_per_model(
    runs: pd.DataFrame,
    split: str = "S1",
    population: str = "P_pred",
    exclude: Sequence[str] = ("B0",),
    metric: str = "mean_auc",
) -> pd.DataFrame:
    """The run with the highest ``metric`` for every model of ``split`` and ``population``,
    best first (one row per model)."""
    scope = runs[
        (runs["split"] == split)
        & (runs["population"] == population)
        & ~runs["model"].isin(list(exclude))
    ]
    return (
        scope.sort_values(metric, ascending=False).drop_duplicates("model").reset_index(drop=True)
    )


def performance_table(
    runs: pd.DataFrame,
    load_oof: Callable[[str], pd.DataFrame | None],
    threshold: float = THRESHOLD,
) -> pd.DataFrame:
    """One row per run (indexed by model): its configuration, mean AUC across held-out
    folds (as logged), and :func:`performance_metrics` on its pooled out-of-fold
    predictions. ``load_oof(run_id)`` returns the ``y``/``p`` frame, or None when the file
    is missing (that run is skipped)."""
    rows = []
    for _, run in runs.iterrows():
        oof = load_oof(str(run["run_id"]))
        if oof is None:
            continue
        rows.append(
            {
                "model": run["model"],
                "configuration": f"{run['feature_set']}, {run['missing_strategy']}",
                "mean AUC (per fold)": float(run["mean_auc"]),
                **performance_metrics(oof["y"], oof["p"], threshold),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("model")


def trial_history(study: optuna.Study) -> pd.DataFrame:
    """One row per completed trial: its number, inner-CV mean log loss, the best log loss
    so far, and its hyperparameters (one column each)."""
    rows = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
            continue
        params = trial.user_attrs.get("params", trial.params)
        rows.append(
            {
                "trial": trial.number + 1,
                "log_loss": float(trial.value),
                **{k: v for k, v in params.items() if k not in RESERVED_PARAMS},
            }
        )
    history = pd.DataFrame(rows)
    if not history.empty:
        history.insert(2, "best_log_loss", history["log_loss"].cummin())
    return history


def logged_threshold_metrics(y: npt.ArrayLike, p: npt.ArrayLike) -> dict[str, float]:
    """:func:`threshold_metrics` at 0.5 under the MLflow names every run carries
    (``pooled_<metric>_at_05``); accuracy is already logged by the harness as
    ``pooled_accuracy_at_05``."""
    return {
        f"pooled_{name.lower()}_at_05": value
        for name, value in threshold_metrics(y, p, THRESHOLD).items()
        if name != "accuracy"
    }


def same_params(tuned: dict[str, Any], logged: dict[str, Any]) -> bool:
    """True when every logged hyperparameter is present in ``tuned`` with the same value
    (numbers compared with a relative tolerance of 1e-9, anything else as text)."""
    if not logged:
        return False
    for key, value in logged.items():
        if key not in tuned:
            return False
        a, b = tuned[key], value
        if isinstance(a, int | float) and isinstance(b, int | float):
            if not np.isclose(float(a), float(b), rtol=1e-9, atol=0.0):
                return False
        elif str(a) != str(b):
            return False
    return True


def fold_params_table(params_by_fold: dict[str, dict[str, Any]], model: str) -> pd.DataFrame:
    """Tuned hyperparameters of one run as rows (model, hyperparameter) and one column per
    fold, so the stability of each value across held-out hospitals reads along a row."""
    table = pd.DataFrame(
        {fold.removeprefix("loho_"): params for fold, params in params_by_fold.items() if params}
    )
    if table.empty:
        return table
    table.index = pd.MultiIndex.from_product(
        [[model], table.index], names=["model", "hyperparameter"]
    )
    return table.map(lambda v: round(v, 4) if isinstance(v, float) else v)
