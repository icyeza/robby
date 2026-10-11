"""How robson_ml and robson_serving fit together: the dependency, the shared runtime code and
old artefacts. Synthetic data or artefact files only (never a data row)."""

import importlib.metadata
import re
from pathlib import Path

import joblib
import numpy as np
import pytest

import robson_ml.metrics
from robson_ml import calibration as training_calibration
from robson_serving import calibration as runtime_calibration
from robson_serving.stats import fit_logistic, logit

LEGACY_MODEL = Path("artefacts/readiness-v1.0/model.joblib")
RUNTIME_NAMES = (
    "PREDICT_THRESHOLD",
    "CalibratedModel",
    "CalibrationChoice",
    "Calibrator",
    "IdentityCalibrator",
    "IsotonicCalibrator",
    "PlattCalibrator",
)


def _names(distribution: str) -> set[str]:
    names = set()
    for requirement in importlib.metadata.requires(distribution) or []:
        match = re.match(r"[A-Za-z0-9_.-]+", requirement)
        assert match is not None
        names.add(match.group(0).lower())
    return names


def test_robson_ml_depends_on_robson_serving() -> None:
    assert "robson-serving" in _names("robson-ml")


def test_newton_logistic_matches_statsmodels() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.98, 500)
    truth = 1 / (1 + np.exp(-(0.4 + 0.7 * np.log(p / (1 - p)))))
    y = (rng.random(500) < truth).astype(float)
    np.testing.assert_allclose(
        fit_logistic(y, logit(p)), robson_ml.metrics.fit_logistic(y, logit(p)), atol=1e-8
    )
    np.testing.assert_allclose(
        fit_logistic(y, offset=logit(p)),
        robson_ml.metrics.fit_logistic(y, offset=logit(p)),
        atol=1e-8,
    )


def test_metrics_reuses_the_serving_logit() -> None:
    assert robson_ml.metrics.logit is logit


def test_robson_ml_calibration_reexports_the_runtime() -> None:
    for name in RUNTIME_NAMES:
        assert getattr(training_calibration, name) is getattr(runtime_calibration, name)


def test_platt_fit_in_training_is_bit_identical_to_the_recorded_runs() -> None:
    rng = np.random.default_rng(3)
    p = rng.uniform(0.02, 0.98, 400)
    y = (rng.random(400) < p).astype(float)
    platt = runtime_calibration.PlattCalibrator().fit(p, y)
    expected = robson_ml.metrics.fit_logistic(y, logit(p))
    assert (platt.intercept_, platt.slope_) == (float(expected[0]), float(expected[1]))


@pytest.mark.skipif(not LEGACY_MODEL.is_file(), reason="readiness-v1.0 artefact not present")
def test_legacy_artefact_loads_as_serving_classes_and_redumps_cleanly(tmp_path: Path) -> None:
    assert b"robson_ml.calibration" in LEGACY_MODEL.read_bytes()
    model = joblib.load(LEGACY_MODEL)
    assert isinstance(model, runtime_calibration.CalibratedModel)
    assert isinstance(model.choice, runtime_calibration.CalibrationChoice)
    assert isinstance(model.choice.calibrator, runtime_calibration.PlattCalibrator)
    joblib.dump(model, tmp_path / "model.joblib")
    redumped = (tmp_path / "model.joblib").read_bytes()
    assert b"robson_ml" not in redumped
    assert b"robson_serving.calibration" in redumped
