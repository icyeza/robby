"""Robson ML workstream: pipeline, evaluation and analytics."""

import os

# No library may send data or usage telemetry to an external service.
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
# Runs are tracked in a local MLflow file store (mlruns/); MLflow 3 refuses it unless opted in.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

__version__ = "0.1.0"
