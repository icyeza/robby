"""Robson ML workstream: pipeline, evaluation and analytics."""

import os

# No library may send data or usage telemetry to an external service (spec §3, §19).
os.environ.setdefault("DO_NOT_TRACK", "1")

__version__ = "0.1.0"
