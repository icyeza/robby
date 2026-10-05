"""B0: predict the training prevalence (the Brier reference). Not tuned."""

from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from robson_ml.feature_sets import FeatureSpec
from robson_ml.models.base import ModelSpec, no_search, pipeline, register


def build(params: dict[str, Any], fs: FeatureSpec) -> Pipeline:
    """A prior-probability DummyClassifier; reads no feature."""
    return pipeline("passthrough", DummyClassifier(strategy="prior"))


SPEC = register(
    ModelSpec(
        name="B0",
        family="baseline",
        complexity_rank=0,
        handles_nan=True,
        native_categorical=False,
        needs_scaling=False,
        build=build,
        search_space=no_search,
        grid={},
    )
)
