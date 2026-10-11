"""Serving runtime for the Robson readiness model.

Imports only numpy, pandas, scikit-learn, joblib, pyarrow, pyyaml and robson_engine; never the
training stack (robson_ml, torch, optuna, mlflow, shap, statsmodels).

    scorer = robson_serving.load(path)       # validates manifest, hash and library versions
    row = scorer.build_row(admission)        # the exact feature vector scored
    probabilities = scorer.score(records)    # list[dict] | DataFrame -> list[float]
"""

from robson_serving._version import __version__
from robson_serving.artefact import IncompatibleArtefact
from robson_serving.loader import Scorer, load
from robson_serving.schema import SchemaError

__all__ = ["IncompatibleArtefact", "SchemaError", "Scorer", "__version__", "load"]
