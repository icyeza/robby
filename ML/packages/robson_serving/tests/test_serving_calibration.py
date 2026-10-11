import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from robson_serving.calibration import (
    PREDICT_THRESHOLD,
    CalibratedModel,
    CalibrationChoice,
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
)


def test_identity_returns_the_raw_probabilities() -> None:
    fitted = IdentityCalibrator().fit(np.array([0.2]), np.array([1.0]))
    assert fitted.transform(np.array([0.2, 0.7])).tolist() == [0.2, 0.7]


def test_platt_transform_uses_its_parameters() -> None:
    platt = PlattCalibrator()
    platt.intercept_, platt.slope_ = 0.5, 2.0
    assert platt.transform(np.array([0.5]))[0] == pytest.approx(1 / (1 + np.exp(-0.5)))


def test_platt_fit_recovers_the_generating_parameters() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(0.02, 0.98, 20000)
    truth = 1 / (1 + np.exp(-(0.5 + 0.6 * np.log(p / (1 - p)))))
    y = (rng.random(20000) < truth).astype(float)
    platt = PlattCalibrator().fit(p, y)
    assert platt.intercept_ == pytest.approx(0.5, abs=0.1)
    assert platt.slope_ == pytest.approx(0.6, abs=0.1)


def test_isotonic_is_a_clipped_step_function() -> None:
    iso = IsotonicCalibrator().fit(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.0, 1.0, 0.0, 1.0]))
    assert iso.transform(np.array([0.1, 0.25, 0.4])).tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_calibrated_model_predicts_and_pickles_by_its_serving_path() -> None:
    x = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
    y = np.array([0, 0, 1, 0, 1, 1])
    estimator = LogisticRegression().fit(x, y)
    choice = CalibrationChoice("none", {"none": 0.1}, IdentityCalibrator())
    model = CalibratedModel(estimator, choice)
    proba = model.predict_proba(x)
    np.testing.assert_allclose(proba, estimator.predict_proba(x))
    expected = (proba[:, 1] >= PREDICT_THRESHOLD).astype(int).tolist()
    assert model.predict(x).tolist() == expected
    data = pickle.dumps(model)
    assert b"robson_serving.calibration" in data
    assert b"robson_ml" not in data
    assert isinstance(pickle.loads(data), CalibratedModel)
