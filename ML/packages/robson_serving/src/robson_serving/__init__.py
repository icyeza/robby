"""Serving runtime for the Robson readiness model.

Imports only numpy, pandas, scikit-learn, joblib, pyarrow, pyyaml and robson_engine; never the
training stack (robson_ml, torch, optuna, mlflow, shap, statsmodels).
"""

from robson_serving._version import __version__

__all__ = ["__version__"]
