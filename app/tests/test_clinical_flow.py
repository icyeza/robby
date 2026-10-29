"""Smoke tests for the clinical flow: the lab technician's form (validation, classification paths,
saving), the clinician's patient search and read-only patient page, degradation without a model,
outcomes and facility scoping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from robson_serving.schema import FeatureSchema
from sqlalchemy import select
from sqlalchemy.orm import Session

from robby.clinical.scoring import ModelRuntime
from robby.config import Settings, default_rule_set_path
from robby.models import (
    Admission,
    Classification,
    ClassificationStatus,
    ModelState,
    Prediction,
    Role,
    ScreeningOmission,
)
from robby.rulesets import load_rule_set_file
from tests.factories import events_of, make_facility, make_model_version, make_user
from tests.webutil import csrf_for, sign_in_client


class StubScorer:
    """Stands in for a loaded artefact: same surface as robson_serving.Scorer."""

    schema = FeatureSchema.from_dict(
        {
            "format": 1,
            "columns": [],
            "inputs": [
                {"name": "height_cm", "kind": "number", "range": [120, 200], "levels": None},
                {
                    "name": "hiv_positive",
                    "kind": "category",
                    "range": None,
                    "levels": ["No", "Yes"],
                },
            ],
        }
    )

    def build_row(self, admission: dict[str, Any]) -> dict[str, Any]:
        return {"parity": float(admission["parity"]), "height_cm": admission.get("height_cm")}

    def model_frame(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return rows

    def predict(self, frame: list[dict[str, Any]]) -> list[float]:
        return [0.714]


GOOD: dict[str, str] = {
    "patient_number": "ZQ-99",
    "patient_name": "Zzyzx Testname",
    "admitted_at": "",  # filled with "now" per test
    "maternal_age": "29",
    "gestational_age_weeks": "39",
    "parity": "2",
    "previous_cs_count": "1",
    "fetal_presentation": "cephalic",
    "plurality": "1",
    "onset_of_labour": "spontaneous",
    "bp_sys": "128",
    "bp_dia": "84",
    "proteinuria": "negative",
    "glucose": "5.2",
}


@pytest.fixture
def world(app: FastAPI, client: TestClient, db: Session, settings: Settings) -> dict[str, Any]:
    load_rule_set_file(db, default_rule_set_path())
    facility = make_facility(db)
    user = make_user(db, role=Role.LAB_TECHNICIAN, facility=facility)
    row = sign_in_client(client, db, settings, user)
    db.commit()
    app.state.model_runtime = ModelRuntime(
        Path("unused"), loader=lambda path, rule_set=None: StubScorer()
    )
    return {
        "facility": facility,
        "user": user,
        "csrf": csrf_for(settings, row),
        "settings": settings,
    }


def now_text(settings: Settings) -> str:
    from robby.timeutil import utcnow

    return utcnow().astimezone(settings.zone).strftime("%d %b %Y, %H:%M")


def post_form(client: TestClient, world: dict[str, Any], path: str = "/admissions", **changes: str):
    data = {
        **GOOD,
        "admitted_at": now_text(world["settings"]),
        **changes,
        "csrf_token": world["csrf"],
    }
    return client.post(path, data={k: v for k, v in data.items() if v is not None})


def saved_code(response: Any) -> str:
    """The admission code from the redirect after the lab technician saves."""
    location = response.headers["location"]
    assert location.startswith("/admissions/new?saved="), location
    return location.rsplit("=", 1)[1]


def as_clinician(client: TestClient, db: Session, world: dict[str, Any]) -> str:
    """Sign in a clinician of the same facility; returns their CSRF token."""
    client.cookies.clear()
    clinician = make_user(db, role=Role.CLINICIAN, facility=world["facility"])
    return csrf_for(world["settings"], sign_in_client(client, db, world["settings"], clinician))


def activate_model(db: Session) -> None:
    make_model_version(db, state=ModelState.ACTIVE)
    db.commit()


def test_form_page_and_screening_error_names_the_field(
    client: TestClient, world: dict[str, Any], db: Session
) -> None:
    page = client.get("/admissions/new")
    assert page.status_code == 200 and "Robson group" in page.text
    response = post_form(client, world, glucose="")
    assert response.status_code == 400
    assert "Glucose is blank" in response.text
    assert db.scalar(select(Admission)) is None
    # not measured needs its reason
    assert (
        "Choose why glucose was not measured"
        in post_form(client, world, glucose="", glucose_nm="1").text
    )


def test_resolved_admission_is_saved_with_a_prediction(
    client: TestClient, world: dict[str, Any], db: Session
) -> None:
    activate_model(db)
    response = post_form(client, world)
    assert response.status_code == 303
    admission = db.scalar(select(Admission))
    assert admission is not None and saved_code(response) == admission.code
    note = client.get(response.headers["location"]).text
    assert f"Saved {admission.code}" in note and "Robson group 5" in note
    assert "had a cesarean" not in note  # the lab technician never sees the signal
    classification = db.scalar(select(Classification))
    assert (
        classification is not None
        and classification.robson_group == 5
        and classification.subgroup == "5a"
    )
    prediction = db.scalar(select(Prediction))
    assert prediction is not None and float(prediction.probability) == pytest.approx(0.714)
    as_clinician(client, db, world)
    page = client.get(f"/admissions/{admission.code}")
    assert (
        "about 71 had a cesarean" in page.text
        and "Group 5" not in page.text.split("Robson classification")[0]
    )
    assert "subgroup 5a" in page.text


def test_saving_without_a_model_keeps_the_classification(
    client: TestClient, world: dict[str, Any], db: Session
) -> None:
    response = post_form(
        client, world, proteinuria="", protein_nm="1", protein_reason="supplies_out_of_stock"
    )
    assert response.status_code == 303
    assert db.scalar(select(Prediction)) is None
    omission = db.scalar(select(ScreeningOmission))
    assert omission is not None and omission.reason.value == "supplies_out_of_stock"
    as_clinician(client, db, world)
    page = client.get(f"/admissions/{saved_code(response)}")
    assert "Readiness signal unavailable" in page.text and "No model is active" in page.text
    assert "reagent strips out of stock" in page.text


def test_partial_and_conflict_are_saved_without_a_group(
    client: TestClient, world: dict[str, Any], db: Session
) -> None:
    activate_model(db)
    partial = post_form(client, world, gestational_age_weeks="")
    conflict = post_form(client, world, parity="0", previous_cs_count="1")
    assert partial.status_code == conflict.status_code == 303
    statuses = {c.status for c in db.scalars(select(Classification))}
    assert statuses == {ClassificationStatus.PARTIAL, ClassificationStatus.CONFLICT}
    assert db.scalar(select(Prediction)) is None
    # the live panel names the single field that decides it
    panel = client.post(
        "/admissions/preview",
        data={**GOOD, "gestational_age_weeks": "", "csrf_token": world["csrf"]},
    )
    assert "Group 5 or 10" in panel.text and "One field decides it" in panel.text
    bad = client.post(
        "/admissions/preview", data={**GOOD, "parity": "0", "csrf_token": world["csrf"]}
    )
    assert "These can't both be true" in bad.text.replace("&#39;", "'")
    as_clinician(client, db, world)
    page = client.get(f"/admissions/{saved_code(partial)}").text
    assert "No Robson group assigned" in page and "/correct" not in page


def test_nothing_entered_can_be_edited_from_the_web(
    client: TestClient, world: dict[str, Any], db: Session
) -> None:
    code = saved_code(post_form(client, world))
    assert client.get(f"/admissions/{code}/correct").status_code == 404
    assert post_form(client, world, f"/admissions/{code}/correct").status_code in (404, 405)
    as_clinician(client, db, world)
    page = client.get(f"/admissions/{code}").text
    assert "Read only" in page and "/correct" not in page and "Record a correction" not in page
    assert client.get(f"/admissions/{code}/correct").status_code == 404
    assert len(list(db.scalars(select(Admission)))) == 1


def test_outcome_and_facility_scoping(
    client: TestClient, world: dict[str, Any], db: Session, settings: Settings
) -> None:
    code = saved_code(post_form(client, world))
    csrf = as_clinician(client, db, world)
    outcome = client.post(
        f"/admissions/{code}/outcome",
        data={"mode": "cesarean", "delivered_at": now_text(settings), "csrf_token": csrf},
    )
    assert outcome.status_code == 303 and outcome.headers["location"] == f"/admissions/{code}"
    assert "Cesarean" in client.get("/patients?q=Zzyzx").text
    # another facility's clinician cannot see it
    client.cookies.clear()
    other = make_user(db, role=Role.CLINICIAN, facility=make_facility(db))
    sign_in_client(client, db, settings, other)
    assert client.get(f"/admissions/{code}").status_code == 404
    assert client.get(f"/admissions/{code}/outcome").status_code == 404
    assert code not in client.get("/patients").text
    assert "No patient at this facility matches" in client.get("/patients?q=Zzyzx").text


def test_patient_identifiers_are_saved_on_the_admission(
    client: TestClient, world: dict[str, Any], db: Session
) -> None:
    response = post_form(client, world, patient_number=" DA/77-1 ", patient_name="  Uwase   Marie ")
    assert response.status_code == 303
    admission = db.scalar(select(Admission))
    assert admission is not None
    assert (admission.patient_number, admission.patient_name) == ("DA/77-1", "Uwase Marie")


def test_a_missing_patient_number_is_not_saved(
    client: TestClient, world: dict[str, Any], db: Session
) -> None:
    response = post_form(client, world, patient_number="")
    assert response.status_code == 400 and "Enter the patient number." in response.text
    assert db.scalar(select(Admission)) is None


def test_the_clinician_finds_a_patient_by_name_or_number(
    client: TestClient, world: dict[str, Any], db: Session
) -> None:
    post_form(client, world, patient_number="RW-1042", patient_name="Uwase Marie")
    post_form(client, world, patient_number="RW-2000", patient_name="Mukamana Grace")
    post_form(client, world, patient_number="RW-3", patient_name="Ingabire 100% A_lice")
    as_clinician(client, db, world)

    def names(query: str) -> set[str]:
        text = client.get("/patients", params={"q": query}).text.split('id="patient-results"')[1]
        return {n for n in ("Uwase", "Mukamana", "Ingabire") if n in text}

    assert names("uwase") == {"Uwase"}
    assert names("MARIE uwa") == {"Uwase"}  # every word must match, any case, any order
    assert names("rw-2000") == {"Mukamana"}  # the whole number, any case
    assert names("RW-2000 grace") == {"Mukamana"}  # a number and a name word together
    assert names("RW") == set()  # numbers are encrypted: no partial number matches
    assert names("RW-20") == set()
    assert names("_") == {"Ingabire"}  # LIKE wildcards are matched literally
    assert names("%") == {"Ingabire"}
    assert names("Nobody") == set()
    assert names("") == {"Uwase", "Mukamana", "Ingabire"}  # empty: the last 72 hours
    fragment = client.get("/patients", params={"q": "uwase"}, headers={"HX-Request": "true"})
    assert fragment.text.startswith('<section class="tablecard" id="patient-results">')


def test_the_clinician_sees_the_patient_name_and_number(
    client: TestClient, world: dict[str, Any], db: Session
) -> None:
    post_form(client, world)
    admission = db.scalar(select(Admission))
    assert admission is not None
    assert "ZQ-99" in client.get("/admissions").text  # the lab technician's recent entries
    as_clinician(client, db, world)
    for path in (
        "/patients",
        f"/admissions/{admission.code}",
        f"/admissions/{admission.code}/outcome",
    ):
        text = client.get(path).text
        assert "ZQ-99" in text and "Zzyzx Testname" in text, path


def test_identifiers_stay_out_of_events_and_features(
    client: TestClient, world: dict[str, Any], db: Session
) -> None:
    activate_model(db)
    post_form(client, world)
    post_form(client, world, patient_number="ZQ-100")
    payloads = [e.payload for e in events_of(db, "admission.created")]
    assert len(payloads) == 2
    for text in (json.dumps(p) for p in payloads):
        assert "ZQ-99" not in text and "ZQ-100" not in text and "Zzyzx" not in text
    predictions = list(db.scalars(select(Prediction)))
    assert predictions
    for prediction in predictions:
        assert not any("patient" in key for key in prediction.feature_vector)
