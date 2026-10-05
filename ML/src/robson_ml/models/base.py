"""Model interface, registry and the shared in-Pipeline preprocessing.

Every model is a :class:`ModelSpec` registered in ``MODEL_REGISTRY`` by its own module under
``robson_ml.models``. ``build(params, feature_spec)`` returns an unfitted Pipeline whose
first step (``prep``) holds every fitted transformation, so imputers, encoders and scalers
are fitted inside each fold on training rows only. ``params`` carries the tuned
hyperparameters plus two reserved keys set by the harness: ``missing_strategy`` (M0, M1,
M2) and ``seed``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Self

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (enables the import)
from sklearn.impute import IterativeImputer, MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, SplineTransformer, StandardScaler

from robson_ml.feature_sets import FeatureSpec

Family = Literal["baseline", "linear", "glassbox", "tree", "kernel", "neural", "ensemble"]
Encoding = Literal["onehot", "ordinal", "native"]
MISSING_STRATEGIES = ("M0", "M1", "M2")
RESERVED_PARAMS = ("missing_strategy", "seed")
SPLINE_KNOTS = 4
ITERATIVE_MAX_ITER = 10
PREP = "prep"
CLF = "clf"


@dataclass(frozen=True)
class ModelSpec:
    """One model of the zoo.

    ``grid`` set to a mapping means the model is tuned over that small grid exhaustively
    instead of by TPE (``{}`` = not tuned at all; the baselines). ``early_stopping_rounds``
    set means the harness fits the final step with early stopping on each inner validation
    fold during tuning and refits with the mean best number of rounds (boosting).
    """

    name: str
    family: Family
    complexity_rank: int
    handles_nan: bool
    native_categorical: bool
    needs_scaling: bool
    build: Callable[[dict[str, Any], FeatureSpec], Pipeline]
    search_space: Callable[[optuna.Trial], dict[str, Any]]
    grid: Mapping[str, Sequence[Any]] | None = None
    early_stopping_rounds: int | None = None

    def strategies(self) -> tuple[str, ...]:
        """The missing-data strategies that apply: M0 only for native-NaN models."""
        return MISSING_STRATEGIES if self.handles_nan else MISSING_STRATEGIES[1:]


MODEL_REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> ModelSpec:
    """Add ``spec`` to ``MODEL_REGISTRY`` (names are unique)."""
    if spec.name in MODEL_REGISTRY:
        raise ValueError(f"model {spec.name!r} is already registered")
    MODEL_REGISTRY[spec.name] = spec
    return spec


def no_search(trial: optuna.Trial) -> dict[str, Any]:
    """Search space of an untuned or grid-tuned model: nothing beyond the grid."""
    return {}


def split_params(params: Mapping[str, Any]) -> tuple[dict[str, Any], str, int]:
    """(model hyperparameters, missing strategy, seed) from a build ``params`` dict."""
    strategy = str(params.get("missing_strategy", "M1"))
    if strategy not in MISSING_STRATEGIES:
        raise ValueError(f"unknown missing strategy {strategy!r}")
    seed = int(params.get("seed", 0))
    return {k: v for k, v in params.items() if k not in RESERVED_PARAMS}, strategy, seed


class CategoryDtypeEncoder(TransformerMixin, BaseEstimator):
    """Casts columns to pandas ``category`` with the levels seen in fit (XGBoost native).

    Unseen levels and missing values become NaN, which native-categorical models handle.
    """

    def fit(self, x: pd.DataFrame, y: Any = None) -> Self:
        """Learn each column's sorted non-missing levels from the training rows."""
        frame = pd.DataFrame(x)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.categories_ = {c: sorted(frame[c].dropna().astype(str).unique()) for c in frame}
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """The columns as ``category`` dtype over the fitted levels."""
        frame = pd.DataFrame(x, columns=self.feature_names_in_)
        return pd.DataFrame(
            {
                c: pd.Categorical(frame[c].astype(object), categories=self.categories_[c])
                for c in frame.columns
            },
            index=frame.index,
        )

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Output column names equal the input names."""
        return np.asarray(self.feature_names_in_, dtype=object)

    def set_output(self, *, transform: str | None = None) -> Self:
        """Always returns pandas; accepted for ColumnTransformer compatibility."""
        return self


def _numeric_imputer(strategy: str, seed: int) -> Any:
    if strategy == "M2":
        return IterativeImputer(max_iter=ITERATIVE_MAX_ITER, random_state=seed)
    return SimpleImputer(strategy="median")


def make_preprocessor(
    fs: FeatureSpec,
    strategy: str,
    seed: int,
    *,
    encoding: Encoding,
    scale: bool,
    spline_columns: Sequence[str] = (),
    columns: Sequence[str] | None = None,
) -> ColumnTransformer:
    """The in-fold preprocessing ColumnTransformer for one model.

    Numerics: M0 raw (NaN passed through); M1 median; M2 ``IterativeImputer`` over the
    numeric columns (single imputation). Categoricals: M0 kept missing; M1/M2 mode. Both
    M1 and M2 add ``MissingIndicator`` columns for every column missing in the training
    rows. Then one-hot (``handle_unknown="ignore"``), ordinal or pandas-category encoding,
    and standardisation of numerics when ``scale``. ``spline_columns`` (numeric) get a
    median-imputed ``SplineTransformer`` with 4 knots in addition to their linear term.
    ``columns`` restricts the model to a subset of the feature set (baselines B1, B2).
    """
    keep = set(fs.columns if columns is None else columns)
    if not keep <= set(fs.columns):
        raise ValueError("a model may only read columns of its feature set")
    numeric = [c for c in (*fs.numeric, *fs.ordinal) if c in keep]
    categorical = [c for c in fs.categorical if c in keep]
    if strategy == "M0" and encoding != "native":
        raise ValueError("M0 (native NaN) needs a model that handles missing values")

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        steps: list[tuple[str, Any]] = []
        if strategy != "M0":
            steps.append(("impute", _numeric_imputer(strategy, seed)))
        if scale:
            steps.append(("scale", StandardScaler()))
        transformers.append(("num", Pipeline(steps) if steps else "passthrough", numeric))
    if categorical:
        cat_steps: list[tuple[str, Any]] = []
        if strategy != "M0":
            cat_steps.append(("impute", SimpleImputer(strategy="most_frequent")))
        if encoding == "onehot":
            encoder: Any = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        elif encoding == "ordinal":
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        else:
            encoder = CategoryDtypeEncoder()
        cat_steps.append(("encode", encoder))
        transformers.append(("cat", Pipeline(cat_steps), categorical))
    splines = [c for c in spline_columns if c in numeric]
    if splines:
        spline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("spline", SplineTransformer(n_knots=SPLINE_KNOTS)),
            ]
        )
        transformers.append(("spline", spline, splines))
    if strategy != "M0":
        indicator = MissingIndicator(features="missing-only", error_on_new=False)
        transformers.append(("missing", indicator, [*numeric, *categorical]))
    prep = ColumnTransformer(transformers, remainder="drop")
    return prep.set_output(transform="pandas")


def pipeline(prep: Any, clf: Any) -> Pipeline:
    """``Pipeline([("prep", prep), ("clf", clf)])``; ``prep`` may be ``"passthrough"``."""
    return Pipeline([(PREP, prep), (CLF, clf)])


# Shared by the logistic models (B2, B3, logreg_l2).
SPLINE_COLUMNS = ("maternal_age", "gestational_age_weeks")
BASELINE_C_GRID = (0.01, 0.1, 1.0, 10.0)
LOGISTIC_MAX_ITER = 5000


def select_columns(columns: Sequence[str]) -> ColumnTransformer:
    """A ColumnTransformer passing ``columns`` through unchanged (pandas output)."""
    keep = ColumnTransformer(
        [("keep", "passthrough", list(columns))], remainder="drop", verbose_feature_names_out=False
    )
    return keep.set_output(transform="pandas")
