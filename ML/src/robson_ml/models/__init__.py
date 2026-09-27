"""The model zoo (spec §12): one module per ModelSpec, registered on import."""

from robson_ml.models import (  # noqa: F401  (imported to register their ModelSpec)
    b0_prevalence,
    b1_robson_lookup,
    b2_robson_logistic,
    b3_spline_logistic,
    cart,
    elasticnet,
    ft_transformer,
    logreg_l2,
    mlp,
    random_forest,
    svm,
    xgb,
)
from robson_ml.models.base import MODEL_REGISTRY, ModelSpec

__all__ = ["MODEL_REGISTRY", "ModelSpec", "get_model"]


def get_model(name: str) -> ModelSpec:
    """The registered ModelSpec called ``name``."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {name!r}; registered: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]
