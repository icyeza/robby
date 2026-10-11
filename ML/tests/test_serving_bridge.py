"""How robson_ml and robson_serving fit together: the dependency, the shared runtime code and
old artefacts. Synthetic data or artefact files only (never a data row)."""

import importlib.metadata
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import robson_ml.metrics
from robson_ml import calibration as training_calibration
from robson_ml import feature_sets, robson_run
from robson_ml.feature_sets import feature_spec
from robson_ml.features import load_feature_registry
from robson_serving import calibration as runtime_calibration
from robson_serving import features as serving_features
from robson_serving.stats import fit_logistic, logit
from tests.test_feature_sets import model_data

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


def test_training_uses_the_serving_derivations() -> None:
    assert feature_sets.robson_group_no_onset is serving_features.robson_group_no_onset
    assert feature_sets.model_column is serving_features.model_column
    assert feature_sets.ROBSON_NO_ONSET == serving_features.ROBSON_NO_ONSET
    assert robson_run.inputs_from_record is serving_features.inputs_from_record


def test_serving_frame_equals_training_frame_on_synthetic_data() -> None:
    data = model_data(load_feature_registry(Path("configs/features_v1.yaml")))
    spec = feature_spec(data, "FS4_deploy")
    columns = [(c, data.kinds[c]) for c in spec.columns]
    inputs = list(serving_features.required_inputs(spec.columns))
    rebuilt = serving_features.build_frame(data.meta[inputs], columns)
    pd.testing.assert_frame_equal(rebuilt, data.x[list(spec.columns)])
