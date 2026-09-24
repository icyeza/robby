import itertools
from pathlib import Path
from typing import Any, ClassVar

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from robson_ml import calibration
from robson_ml.calibration import (
    CalibratedModel,
    IsotonicCalibrator,
    PlattCalibrator,
    calibrate,
    choose_method,
    select_calibrator,
)


def _platt_family(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Raw scores whose true probability is a Platt transform of them."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.98, size=n)
    true = 1 / (1 + np.exp(-(0.5 + 0.6 * np.log(p / (1 - p)))))
    return (rng.random(n) < true).astype(int), p


def _brier(y: np.ndarray, q: np.ndarray) -> float:
    return float(np.mean((q - y) ** 2))


class _Spy:
    """Wraps a calibrator class and records every prediction set it is fitted or scored on."""

    log: ClassVar[list[tuple[str, int, frozenset[float]]]] = []
    _ids = itertools.count()

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.uid = next(_Spy._ids)

    def fit(self, p: np.ndarray, y: np.ndarray) -> "_Spy":
        _Spy.log.append(("fit", self.uid, frozenset(np.asarray(p).tolist())))
        self.inner.fit(p, y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        _Spy.log.append(("transform", self.uid, frozenset(np.asarray(p).tolist())))
        return self.inner.transform(p)


def _spy_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    _Spy.log = []
    spied = {
        name: (lambda f=factory: _Spy(f())) for name, factory in calibration.CALIBRATORS.items()
    }
    monkeypatch.setattr(calibration, "CALIBRATORS", spied)


def test_calibration_cv_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spec §22 / §13.1 v1.1: chosen on out-of-fold Brier, never on the rows it was fitted to."""
    y, p = _platt_family(300, seed=2)
    # In-sample, isotonic always wins (it is the best monotone fit to the rows it saw) ...
    in_sample_iso = _brier(y, IsotonicCalibrator().fit(p, y).transform(p))
    in_sample_platt = _brier(y, PlattCalibrator().fit(p, y).transform(p))
    assert in_sample_iso < in_sample_platt
    # ... but out of fold it overfits, and the Platt-shaped truth picks Platt.
    _spy_factories(monkeypatch)
    choice = select_calibrator(y, p, seed=0)
    assert choice.method == "platt"
    assert set(choice.cv_brier) == {"none", "platt", "isotonic"}
    assert choice.cv_brier["isotonic"] > choice.cv_brier["platt"]

    fits = {obj: rows for kind, obj, rows in _Spy.log if kind == "fit"}
    scored = [(obj, rows) for kind, obj, rows in _Spy.log if kind == "transform"]
    assert len(scored) == 3 * 5  # every option scored once per fold
    for obj, rows in scored:
        assert not rows & fits[obj]  # never scored on its own fitting rows
        assert len(rows) + len(fits[obj]) == len(p)
    # The chosen option is refitted on the whole calibration split.
    assert _Spy.log[-1][0] == "fit"
    assert _Spy.log[-1][2] == frozenset(p.tolist())


def test_choose_method_tie_rule() -> None:
    # Platt within 0.001 of the best wins the tie.
    assert choose_method({"none": 0.2000, "platt": 0.2004, "isotonic": 0.1995}) == "platt"
    # Platt out of the tie, none within it.
    assert choose_method({"none": 0.2000, "platt": 0.2015, "isotonic": 0.1995}) == "none"
    assert choose_method({"none": 0.2100, "platt": 0.2015, "isotonic": 0.1990}) == "isotonic"


class _RecordingModel:
    """A fitted stand-in estimator that records which rows it is asked to predict."""

    classes_ = np.array([0, 1])

    def __init__(self) -> None:
        self.seen: list[np.ndarray] = []

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        self.seen.append(x["row"].to_numpy())
        q = x["score"].to_numpy()
        return np.column_stack([1 - q, q])


def test_calibrator_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spec §22: the calibrator is fitted only on calibration-split indices."""
    n = 600
    y, p = _platt_family(n, seed=4)
    x = pd.DataFrame({"row": np.arange(n), "score": p})
    calib_idx = np.sort(np.random.default_rng(1).choice(n, size=150, replace=False))
    model = _RecordingModel()
    _spy_factories(monkeypatch)
    fitted = calibrate(model, x, pd.Series(y, dtype="Int64"), calib_idx, seed=0)
    assert len(model.seen) == 1
    assert np.array_equal(model.seen[0], calib_idx)
    calib_scores = frozenset(p[calib_idx].tolist())
    for kind, _, rows in _Spy.log:
        if kind == "fit":
            assert rows <= calib_scores
    assert _Spy.log[-1][2] == calib_scores
    assert fitted.choice.method in {"none", "platt", "isotonic"}


def test_calibrated_model_predicts_and_pickles(tmp_path: Path) -> None:
    rng = np.random.default_rng(6)
    x = rng.normal(size=(500, 3))
    y = (rng.random(500) < 1 / (1 + np.exp(-x[:, 0]))).astype(int)
    base = LogisticRegression().fit(x[:300], y[:300])
    model = calibrate(base, x, y, np.arange(300, 500), seed=0)
    proba = model.predict_proba(x)
    assert proba.shape == (500, 2)
    assert np.allclose(proba.sum(axis=1), 1)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert list(model.classes_) == [0, 1]
    assert set(model.predict(x)) <= {0, 1}
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    assert np.array_equal(restored.predict_proba(x), proba)
    assert isinstance(restored, CalibratedModel)


def test_selection_is_deterministic() -> None:
    y, p = _platt_family(200, seed=8)
    a = select_calibrator(y, p, seed=3)
    b = select_calibrator(y, p, seed=3)
    assert a.method == b.method
    assert a.cv_brier == b.cv_brier


def test_grouped_selection_keeps_mothers_together(monkeypatch: pytest.MonkeyPatch) -> None:
    y, p = _platt_family(200, seed=9)
    groups = np.repeat(np.arange(100), 2)
    _spy_factories(monkeypatch)
    select_calibrator(y, p, seed=0, groups=groups)
    fits = {obj: rows for kind, obj, rows in _Spy.log if kind == "fit"}
    group_of = dict(zip(p.tolist(), groups.tolist(), strict=True))
    for kind, obj, rows in _Spy.log:
        if kind == "transform":
            assert not {group_of[v] for v in rows} & {group_of[v] for v in fits[obj]}
