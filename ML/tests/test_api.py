"""The HTTP service: classification, enforced screening, readiness and degradation."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main
from api.readiness import DEFAULT_MODEL_DIR, ML_DIR, MODEL_DIR_ENV, ReadinessModel
from api.schemas import READINESS_LABEL

GROUP_5 = {
    "parity": 2,
    "previous_cs_count": 1,
    "plurality": 1,
    "fetal_presentation": "cephalic",
    "gestational_age_weeks": 39,
    "onset_of_labour": "spontaneous",
}
SCREENING = {
    "blood_pressure": {"systolic": 128, "diastolic": 84},
    "proteinuria": {"not_measured_reason": "equipment_unavailable"},
    "glucose": {"value_mmol_l": 5.2},
}
ADMISSION = {**GROUP_5, "facility_id": "Site A", "screening": SCREENING, "height_cm": 160}
FEATURES = ["gestational_age_weeks", "parity", "fetal_presentation", "robson_group_no_onset"]


class FakeModel:
    """Returns a fixed probability and remembers the frame it was given."""

    def __init__(self) -> None:
        self.seen: pd.DataFrame | None = None

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        self.seen = x
        return np.array([[0.3, 0.7]] * len(x))


@pytest.fixture
def no_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv(MODEL_DIR_ENV, str(tmp_path))
    with TestClient(api.main.app) as client:
        yield client


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, FakeModel]]:
    model = FakeModel()
    summary = {"version_label": "fake-v0", "algorithm": "fake", "features": FEATURES}
    levels = {
        "fetal_presentation": ["cephalic", "non_cephalic"],
        "robson_group_no_onset": ["5", "partial"],
    }

    def load(_: Path) -> tuple[ReadinessModel, None]:
        return ReadinessModel(model, summary, levels), None

    monkeypatch.setattr(api.main, "load_readiness_model", load)
    with TestClient(api.main.app) as client:
        yield client, model


def test_classify_resolved(no_model: TestClient) -> None:
    body = no_model.post("/classify", json=GROUP_5).json()
    assert body["status"] == "resolved"
    assert (body["group"], body["subgroup"]) == (5, "5a")
    assert body["rule_set_version"] == "robson-v1.0"
    assert body["trace"] and {t["outcome"] for t in body["trace"]} <= {"true", "false", "unknown"}


def test_classify_partial_lists_candidates_and_resolving_fields(no_model: TestClient) -> None:
    inputs = {k: v for k, v in GROUP_5.items() if k != "gestational_age_weeks"}
    body = no_model.post("/classify", json=inputs).json()
    assert body["status"] == "partial" and body["group"] is None
    assert body["candidates"] == [5, 10]
    assert body["resolving_fields"] == ["gestational_age_weeks"]


def test_classify_conflict_names_fields(no_model: TestClient) -> None:
    body = no_model.post("/classify", json={**GROUP_5, "parity": 0}).json()
    assert body["status"] == "conflict" and body["group"] is None
    assert set(body["conflict_fields"]) == {"parity", "previous_cs_count"}
    assert body["consistency_checks_failed"] == ["nulliparous_with_previous_cs"]


def test_classify_rejects_out_of_range_and_half_band(no_model: TestClient) -> None:
    too_late = no_model.post("/classify", json={**GROUP_5, "gestational_age_weeks": 50})
    assert too_late.status_code == 422
    response = no_model.post("/classify", json={"ga_band_lower": 37})
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_robson_inputs"


@pytest.mark.parametrize("field", ["blood_pressure", "proteinuria", "glucose"])
def test_assess_rejects_blank_screening(no_model: TestClient, field: str) -> None:
    response = no_model.post(
        "/admissions/assess", json={**ADMISSION, "screening": {**SCREENING, field: {}}}
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "screening_incomplete" and body["fields"] == [field]
    assert "not_measured_reason" in body["message"]


def test_assess_rejects_missing_screening_and_value_with_reason(no_model: TestClient) -> None:
    missing = no_model.post("/admissions/assess", json=GROUP_5)
    assert missing.status_code == 422
    assert missing.json()["fields"] == ["blood_pressure", "proteinuria", "glucose"]
    glucose = {"value_mmol_l": 5.0, "not_measured_reason": "other"}
    both = no_model.post(
        "/admissions/assess", json={**ADMISSION, "screening": {**SCREENING, "glucose": glucose}}
    )
    assert both.status_code == 422 and both.json()["error"] == "screening_ambiguous"


def test_no_model_degrades_to_classification_only(no_model: TestClient) -> None:
    health = no_model.get("/health").json()
    assert health["model_loaded"] is False and health["model_version"] is None
    assert health["rule_set_version"] == "robson-v1.0"
    assert no_model.get("/model").status_code == 503
    response = no_model.post("/admissions/assess", json=ADMISSION)
    assert response.status_code == 200
    body = response.json()
    assert body["classification"]["group"] == 5
    assert body["readiness"] is None
    assert "No readiness model" in body["readiness_unavailable_reason"]


def test_assess_with_model_returns_labelled_signal(
    fake_model: tuple[TestClient, FakeModel],
) -> None:
    client, model = fake_model
    assert client.get("/health").json()["model_version"] == "fake-v0"
    admission = {**ADMISSION, "fetal_presentation": "breech", "facility_id": None}
    body = client.post("/admissions/assess", json=admission).json()
    readiness = body["readiness"]
    assert readiness["probability"] == 0.7 and readiness["of_100_women_like_her"] == 70
    assert readiness["label"] == READINESS_LABEL and "Not a recommendation" in READINESS_LABEL
    assert readiness["model_version"] == "fake-v0"
    assert "facility_id" in readiness["inputs_not_given"]
    assert "height_cm" not in readiness["inputs_not_given"]
    assert model.seen is not None and list(model.seen.columns) == FEATURES
    row = model.seen.iloc[0]
    assert row["fetal_presentation"] == "non_cephalic"  # precise value mapped to model level
    assert row["robson_group_no_onset"] == "7"  # multiparous breech resolves without onset
    assert readiness["unrecognised_inputs"] == ["robson_group_no_onset"]  # "7" not in levels


def test_assess_withholds_signal_on_conflict(fake_model: tuple[TestClient, FakeModel]) -> None:
    client, _ = fake_model
    body = client.post("/admissions/assess", json={**ADMISSION, "parity": 0}).json()
    assert body["classification"]["status"] == "conflict"
    assert body["readiness"] is None and "conflicting" in body["readiness_unavailable_reason"]


def test_assess_reports_unrecognised_levels(fake_model: tuple[TestClient, FakeModel]) -> None:
    client, model = fake_model
    body = client.post("/admissions/assess", json=ADMISSION).json()
    assert body["readiness"]["unrecognised_inputs"] == []
    assert model.seen is not None and model.seen.iloc[0]["robson_group_no_onset"] == "5"


REAL_MODEL = ML_DIR / DEFAULT_MODEL_DIR / "model.joblib"


@pytest.mark.skipif(not REAL_MODEL.is_file(), reason="trained readiness artefact not present")
def test_real_model_probability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MODEL_DIR_ENV, raising=False)
    with TestClient(api.main.app) as client:
        assert client.get("/health").json()["model_loaded"] is True
        card = client.get("/model").json()
        assert card["features"] and card["calibration_method"]
        readiness = client.post("/admissions/assess", json=ADMISSION).json()["readiness"]
    assert 0 < readiness["probability"] < 1
