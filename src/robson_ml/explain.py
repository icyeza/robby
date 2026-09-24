"""Interpretation of a selected configuration (spec §14), aggregate outputs only.

The fold models of a logged run are rebuilt without re-tuning (:func:`refit_folds`): the same
folds (same seed), the tuned hyperparameters logged for each fold, the same in-fold
preprocessing, refit and calibration as the harness. Every explanation is then computed on
each fold's *held-out* rows and summarised globally:

- permutation importance (drop in held-out AUC when one input column is shuffled);
- SHAP: ``TreeExplainer`` for tree models, otherwise a model-agnostic permutation explainer
  on a sample of at most 500 held-out rows; reported only as mean |SHAP| per input feature
  (no beeswarm or dependence plots: those draw one point per woman);
- the Robson recovery check (spec §14.3);
- partial dependence (averaged predictions over a grid; no ICE curves, which are per woman);
- per-Robson-group AUC against B1 (from the runs' suppressed subgroup tables, §11.4).

SHAP values explain the model's score (log-odds for trees, calibrated probability for the
agnostic explainer); importances describe what drives predicted CS under current practice,
not what should be done for any woman.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from robson_ml.calibration import CalibratedModel, calibrate
from robson_ml.evaluate import ExperimentConfig, make_folds, metric_key
from robson_ml.feature_sets import ROBSON_NO_ONSET, ModelData, feature_spec
from robson_ml.metrics import INSUFFICIENT
from robson_ml.models import get_model
from robson_ml.models.base import CLF, PREP
from robson_ml.profile import released_quantiles
from robson_ml.splits import Fold, mother_groups

SHAP_MAX_ROWS = 500
SHAP_BACKGROUND = 50
# Antithetic permutations per explained row (each costs 2 x n_features + 1 evaluations).
SHAP_PERMUTATIONS = 3
PERMUTATION_REPEATS = 5
PDP_MIN_ROWS = 50
TOP_K_RECOVERY = 3
ROBSON_RECOVERY_FEATURES = ("previous_cs_count", ROBSON_NO_ONSET)
MISSING_CODE = -1


@dataclass
class FittedFold:
    """One fold's refitted, calibrated model and its held-out rows."""

    fold: Fold
    model: CalibratedModel
    params: dict[str, Any]

    @property
    def test_idx(self) -> npt.NDArray[np.int64]:
        """Row positions the model never saw (the held-out facility or outer fold)."""
        return self.fold.test_idx


def refit_folds(
    data: ModelData, config: ExperimentConfig, fold_params: Mapping[str, Mapping[str, Any]]
) -> list[FittedFold]:
    """Rebuild a run's fold models from its logged per-fold hyperparameters (no tuning).

    Inputs: the population's :class:`ModelData`; the run's configuration; ``fold_params``
    keyed by the sanitised fold name (:func:`robson_ml.select.run_fold_params`). Each fold is
    fitted on ``fit_idx`` and calibrated on ``calib_idx`` exactly as the harness does, so the
    held-out predictions reproduce the run's.
    """
    spec = get_model(config.model)
    fs = feature_spec(data, config.feature_set)
    x = data.x[list(fs.columns)]
    groups = mother_groups(data.meta)
    reserved = {"missing_strategy": config.missing_strategy, "seed": config.seed}
    fitted = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for fold in make_folds(data.meta, config.split, config.seed):
            key = metric_key(fold.name)
            if key not in fold_params:
                raise KeyError(f"no logged hyperparameters for fold {fold.name!r}")
            params = {**dict(fold_params[key]), **reserved}
            pipe = spec.build(params, fs).fit(x.iloc[fold.fit_idx], data.y[fold.fit_idx])
            model = calibrate(pipe, x, data.y, fold.calib_idx, config.seed, groups=groups)
            fitted.append(FittedFold(fold, model, params))
    return fitted


def model_columns(data: ModelData, config: ExperimentConfig) -> list[str]:
    """The input columns the configuration's model reads."""
    return list(feature_spec(data, config.feature_set).columns)


def heldout_auc(fitted: Sequence[FittedFold], x: pd.DataFrame, y: npt.ArrayLike) -> pd.Series:
    """Held-out AUC of each refitted fold (to check the refit reproduces the run)."""
    y_arr = np.asarray(y)
    return pd.Series(
        {
            f.fold.name: float(
                roc_auc_score(y_arr[f.test_idx], f.model.predict_proba(x.iloc[f.test_idx])[:, 1])
            )
            for f in fitted
        }
    )


class _Fitted:
    """A fitted model exposed to sklearn's ``permutation_importance`` (which requires a
    ``fit`` method); ``fit`` is never called."""

    def __init__(self, model: CalibratedModel) -> None:
        self.model = model

    def fit(self, x: Any, y: Any = None) -> _Fitted:
        """Not used: the model is already fitted."""
        raise NotImplementedError("the wrapped model is already fitted")

    def predict_proba(self, x: pd.DataFrame) -> npt.NDArray[np.float64]:
        """The calibrated model's class probabilities."""
        return self.model.predict_proba(x)


def _auc_scorer(estimator: Any, x: pd.DataFrame, y: npt.ArrayLike) -> float:
    return float(roc_auc_score(np.asarray(y), estimator.predict_proba(x)[:, 1]))


def permutation_importances(
    fitted: Sequence[FittedFold],
    x: pd.DataFrame,
    y: npt.ArrayLike,
    n_repeats: int = PERMUTATION_REPEATS,
    seed: int = 0,
) -> pd.DataFrame:
    """Drop in held-out AUC when each input column is shuffled, per fold and on average.

    Computed on each fold's held-out rows with the fold's own model (sklearn
    ``permutation_importance``). Returns one row per input column: ``mean`` (across folds,
    equally weighted), ``sd`` (across folds) and one column per fold, sorted by ``mean``.
    """
    y_arr = np.asarray(y)
    per_fold = {}
    for f in fitted:
        result = permutation_importance(
            _Fitted(f.model),
            x.iloc[f.test_idx],
            y_arr[f.test_idx],
            scoring=_auc_scorer,
            n_repeats=n_repeats,
            random_state=seed,
        )
        per_fold[f.fold.name] = pd.Series(result.importances_mean, index=x.columns)
    table = pd.DataFrame(per_fold)
    table.insert(0, "sd", table.std(axis=1, ddof=0))
    table.insert(0, "mean", table.drop(columns="sd").mean(axis=1))
    return table.sort_values("mean", ascending=False).rename_axis("feature")


def input_feature(name: str, columns: Sequence[str]) -> str:
    """The input column a transformed column name derives from.

    Strips the ColumnTransformer prefix (``num__``, ``cat__``, ...) and a
    ``missingindicator_`` prefix, then takes the longest input column that the rest equals or
    starts with (one-hot levels are ``<column>_<level>``). Unknown names are returned as is.
    """
    rest = name.split("__", 1)[1] if "__" in name else name
    rest = rest.removeprefix("missingindicator_")
    matches = [c for c in columns if rest == c or rest.startswith(f"{c}_")]
    return max(matches, key=len) if matches else rest


class _CodedModel:
    """Scores integer-coded rows: categorical codes are decoded back to labels, so a
    model-agnostic explainer can perturb a numeric array."""

    def __init__(self, model: CalibratedModel, frame: pd.DataFrame) -> None:
        self.model = model
        self.columns = list(frame.columns)
        self.levels: dict[str, list[str]] = {}
        for column in frame.columns:
            if frame[column].dtype == object:
                self.levels[column] = sorted(frame[column].dropna().astype(str).unique())

    def encode(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        out = np.empty((len(frame), len(self.columns)), dtype=np.float64)
        for j, column in enumerate(self.columns):
            if column in self.levels:
                codes = {level: i for i, level in enumerate(self.levels[column])}
                labels = frame[column].astype(object)
                out[:, j] = [
                    codes.get(str(v), MISSING_CODE) if pd.notna(v) else MISSING_CODE for v in labels
                ]
            else:
                out[:, j] = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        return out

    def decode(self, array: npt.NDArray[np.float64]) -> pd.DataFrame:
        data: dict[str, Any] = {}
        for j, column in enumerate(self.columns):
            if column in self.levels:
                # The last entry is the missing value, selected by the code -1.
                labels = np.array([*self.levels[column], np.nan], dtype=object)
                codes = np.nan_to_num(array[:, j], nan=MISSING_CODE).astype(np.int64)
                codes[(codes < 0) | (codes >= len(labels) - 1)] = MISSING_CODE
                data[column] = labels[codes]
            else:
                data[column] = array[:, j]
        return pd.DataFrame(data)

    def __call__(self, array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray(self.model.predict_proba(self.decode(array))[:, 1], dtype=np.float64)


def _sample(rows: npt.NDArray[np.int64], k: int, rng: np.random.Generator) -> np.ndarray:
    return rows if len(rows) <= k else np.sort(rng.choice(rows, size=k, replace=False))


def _tree_shap(f: FittedFold, x: pd.DataFrame) -> tuple[pd.Series, str]:
    """Mean |SHAP| per transformed column of a tree pipeline (log-odds scale)."""
    import shap

    pipe = f.model.estimator
    transformed = pipe.named_steps[PREP].transform(x)
    clf = pipe.named_steps[CLF]
    try:
        values = shap.TreeExplainer(clf).shap_values(transformed)
        method = "shap.TreeExplainer"
    except Exception:  # fall back to the booster's exact TreeSHAP
        import xgboost

        matrix = xgboost.DMatrix(transformed, enable_categorical=True)
        values = clf.get_booster().predict(matrix, pred_contribs=True)[:, :-1]
        method = "xgboost pred_contribs (TreeSHAP)"
    array = np.abs(np.asarray(values, dtype=np.float64))
    if array.ndim == 3:  # (rows, columns, classes)
        array = array[..., -1]
    return pd.Series(array.mean(axis=0), index=list(transformed.columns)), method


def _agnostic_shap(
    f: FittedFold, x: pd.DataFrame, rows: np.ndarray, background_rows: np.ndarray, seed: int
) -> tuple[pd.Series, str]:
    """Mean |SHAP| per input column from a permutation explainer (probability scale)."""
    import shap

    coded = _CodedModel(f.model, x)
    background = coded.encode(x.iloc[background_rows])
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.PermutationExplainer(coded, masker, seed=seed)
    max_evals = SHAP_PERMUTATIONS * (2 * x.shape[1] + 1)
    explanation = explainer(coded.encode(x.iloc[rows]), max_evals=max_evals, silent=True)
    values = np.abs(np.asarray(explanation.values, dtype=np.float64))
    return pd.Series(values.mean(axis=0), index=list(x.columns)), "shap.PermutationExplainer"


def shap_importance(
    fitted: Sequence[FittedFold],
    x: pd.DataFrame,
    family: str,
    max_rows: int = SHAP_MAX_ROWS,
    background: int = SHAP_BACKGROUND,
    seed: int = 0,
) -> tuple[pd.DataFrame, str]:
    """Global mean |SHAP| per input column, averaged over folds (held-out rows only).

    Tree models (``family == "tree"``) use ``shap.TreeExplainer`` on the transformed
    features, which are summed back to their input column (:func:`input_feature`); other
    models use a permutation explainer over the input columns, with a background drawn from
    each fold's training rows. At most ``max_rows`` held-out rows are explained in total
    (split evenly across folds). Returns (table with ``mean_abs_shap`` and one column per
    fold, sorted; the method used).
    """
    rng = np.random.default_rng(seed)
    per_fold_rows = max(1, max_rows // max(len(fitted), 1))
    columns = list(x.columns)
    per_fold: dict[str, pd.Series] = {}
    method = ""
    for f in fitted:
        rows = _sample(f.test_idx, per_fold_rows, rng)
        if family == "tree":
            values, method = _tree_shap(f, x.iloc[rows])
            grouped = values.groupby([input_feature(c, columns) for c in values.index]).sum()
        else:
            train = np.concatenate([f.fold.fit_idx, f.fold.calib_idx])
            values, method = _agnostic_shap(f, x, rows, _sample(train, background, rng), seed)
            grouped = values
        per_fold[f.fold.name] = grouped.reindex(columns, fill_value=0.0)
    table = pd.DataFrame(per_fold)
    table.insert(0, "mean_abs_shap", table.mean(axis=1))
    return table.sort_values("mean_abs_shap", ascending=False).rename_axis("feature"), method


@dataclass(frozen=True)
class RecoveryCheck:
    """Spec §14.3: does a Robson feature rank in the top ``k`` by importance?"""

    passed: bool
    top: tuple[str, ...]
    found: tuple[str, ...]


def robson_recovery(
    importance: pd.Series,
    k: int = TOP_K_RECOVERY,
    features: Sequence[str] = ROBSON_RECOVERY_FEATURES,
) -> RecoveryCheck:
    """Whether ``previous_cs_count`` or ``robson_group_no_onset`` is among the top ``k`` of
    ``importance`` (a Series indexed by input feature). A failure opens a leakage
    investigation (spec §14.3)."""
    top = tuple(importance.sort_values(ascending=False).index[:k])
    found = tuple(f for f in features if f in top)
    return RecoveryCheck(bool(found), top, found)


def pdp_grid(values: pd.Series, step: float | None = None, n_points: int = 15) -> npt.NDArray[Any]:
    """A partial-dependence grid between the released 5th and 95th percentiles of
    ``values`` (so no grid point sits on a sparse extreme): multiples of ``step`` inside
    them when given (e.g. 1 for counts), else ``n_points`` evenly spaced values. Empty
    when the percentiles cannot be released."""
    quantiles = released_quantiles(pd.to_numeric(values, errors="coerce").dropna())
    low, high = quantiles.get("p5"), quantiles.get("p95")
    if low is None or high is None:
        return np.array([])
    if step is not None:
        start, stop = np.ceil(low / step) * step, np.floor(high / step) * step
        return np.arange(start, stop + step / 2, step)
    return np.linspace(low, high, n_points)


def partial_dependence(
    fitted: Sequence[FittedFold],
    x: pd.DataFrame,
    feature: str,
    grid: npt.ArrayLike,
    max_rows: int = SHAP_MAX_ROWS,
    seed: int = 0,
) -> pd.DataFrame:
    """Mean predicted probability with ``feature`` set to each grid value, on held-out rows.

    For each fold, up to ``max_rows / n_folds`` held-out rows are copied with ``feature``
    set to the grid value and scored by the fold's model; the curve is the mean over rows,
    then over folds (an aggregate over at least ``PDP_MIN_ROWS`` rows per fold, else the
    fold is skipped). Returns ``value``, ``mean`` and one column per fold.
    """
    rng = np.random.default_rng(seed)
    per_fold_rows = max(PDP_MIN_ROWS, max_rows // max(len(fitted), 1))
    grid_arr = np.asarray(grid)
    curves: dict[str, list[float]] = {}
    for f in fitted:
        rows = _sample(f.test_idx, per_fold_rows, rng)
        if len(rows) < PDP_MIN_ROWS:
            continue
        base = x.iloc[rows].copy()
        curve = []
        for value in grid_arr:
            base[feature] = value
            curve.append(float(np.mean(f.model.predict_proba(base)[:, 1])))
        curves[f.fold.name] = curve
    table = pd.DataFrame(curves)
    table.insert(0, "mean", table.mean(axis=1) if curves else np.nan)
    table.insert(0, "value", grid_arr)
    return table


def subgroup_comparison(
    selected: pd.DataFrame,
    baseline: pd.DataFrame,
    by: str = ROBSON_NO_ONSET,
    metric: str = "auc",
    labels: tuple[str, str] = ("selected", "B1"),
) -> pd.DataFrame:
    """Per-level ``metric`` of two runs' subgroup tables (``subgroups.csv`` artefacts),
    side by side; levels below the §11.4 minimum counts stay ``"insufficient"``."""

    def column(table: pd.DataFrame) -> pd.Series:
        rows = table[table["subgroup"] == by].set_index("level")[metric]
        return rows.map(lambda v: v if v == INSUFFICIENT else round(float(v), 3))

    out = pd.concat([column(selected), column(baseline)], axis=1, keys=list(labels))
    out.index = out.index.astype(str)
    return out.fillna(INSUFFICIENT).rename_axis(by)
