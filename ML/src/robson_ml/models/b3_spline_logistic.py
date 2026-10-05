"""B3: L2 logistic regression with splines on FS4 (clinical-prediction baseline).

C from the tiny ``BASELINE_C_GRID``; maternal age and GA get 4-knot spline terms.
"""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from robson_ml.feature_sets import FeatureSpec
from robson_ml.models.base import (
    BASELINE_C_GRID,
    LOGISTIC_MAX_ITER,
    SPLINE_COLUMNS,
    ModelSpec,
    make_preprocessor,
    no_search,
    pipeline,
    register,
    split_params,
)


def build(params: dict[str, Any], fs: FeatureSpec) -> Pipeline:
    """One-hot + standardised features plus age/GA splines into an L2 logistic regression."""
    hyper, strategy, seed = split_params(params)
    prep = make_preprocessor(
        fs, strategy, seed, encoding="onehot", scale=True, spline_columns=SPLINE_COLUMNS
    )
    return pipeline(prep, LogisticRegression(C=float(hyper["C"]), max_iter=LOGISTIC_MAX_ITER))


SPEC = register(
    ModelSpec(
        name="B3",
        family="baseline",
        complexity_rank=0,
        handles_nan=False,
        native_categorical=False,
        needs_scaling=True,
        build=build,
        search_space=no_search,
        grid={"C": BASELINE_C_GRID},
    )
)
