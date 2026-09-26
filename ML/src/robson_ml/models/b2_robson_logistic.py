"""B2: logistic regression on the Robson inputs except onset (spec §12.2, v1.3).

The linear version of B1. Onset is left out: it is outcome-contaminated (coded
retrospectively, spec v1.3), so B2 reads parity, previous CS count, presentation, plurality
and gestational age (exact value plus band bounds). L2 penalty with C from the tiny
``BASELINE_C_GRID``; gestational age also gets 4-knot spline terms (spec §9.3, logistic
baselines).
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

# The Robson inputs (spec §6.1) except onset (v1.3); GA is the exact value plus its band.
ROBSON_INPUTS = (
    "parity",
    "previous_cs_count",
    "fetal_presentation",
    "plurality",
    "gestational_age_weeks",
    "ga_band_lower",
    "ga_band_upper",
)


def build(params: dict[str, Any], fs: FeatureSpec) -> Pipeline:
    """One-hot + standardised Robson inputs, no onset (GA splines), into L2 logistic."""
    hyper, strategy, seed = split_params(params)
    prep = make_preprocessor(
        fs,
        strategy,
        seed,
        encoding="onehot",
        scale=True,
        spline_columns=SPLINE_COLUMNS,
        columns=[c for c in ROBSON_INPUTS if c in fs.columns],
    )
    return pipeline(prep, LogisticRegression(C=float(hyper["C"]), max_iter=LOGISTIC_MAX_ITER))


SPEC = register(
    ModelSpec(
        name="B2",
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
