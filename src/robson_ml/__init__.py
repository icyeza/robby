"""Robson ML workstream: pipeline, evaluation and analytics."""

import os

# No library may send data or usage telemetry to an external service (spec §3, §19).
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
# Spec §19 requires a local MLflow file store (mlruns/); MLflow 3 refuses it unless opted in.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

__version__ = "0.1.0"
