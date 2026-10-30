"""Lab requests over FHIR: GET /api/v1/fhir/ServiceRequest?status=active and
POST /api/v1/fhir/Observation (basedOn a ServiceRequest), with the lab technician's API token."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from robby.auth import api_tokens
from robby.clinical import service as clinical
from robby.clinical.form import parse
from robby.clinical.scoring import ModelRuntime
from robby.config import Settings, default_rule_set_path
from robby.fhir import mapping, resources
from robby.lab import service as lab
from robby.models import Admission, LabResult, LabTest, ModelState, Prediction, Role
from robby.rulesets import load_rule_set_file
from robby.timeutil import utcnow
from tests.factories import events_of, make_facility, make_model_version, make_user
from tests.test_clinical_flow import GOOD, now_text
from tests.test_lab_requests import HivStubScorer

SEARCH = "/api/v1/fhir/ServiceRequest?status=active"
POST = "/api/v1/fhir/Observation"


@pytest.fixture
def feed(app: FastAPI, client: TestClient, db: Session, settings: Settings) -> dict[str, Any]:
    load_rule_set_file(db, default_rule_set_path())
    facility = make_facility(db)
    user = make_user(db, role=Role.LAB_TECHNICIAN, facility=facility)
    clinician = make_user(db, role=Role.CLINICIAN, facility=facility)
    make_model_version(db, state=ModelState.ACTIVE)
    token, _ = api_tokens.create(db, None, user.username, "Lab system")
    runtime = ModelRuntime(Path("unused"), loader=lambda path, rule_set=None: HivStubScorer())
    app.state.model_runtime = runtime
    values = {**GOOD, "admitted_at": now_text(settings), "patient_number": "FH-12"}
    parsed, errors = parse(values, [], settings.zone, utcnow())
    assert not errors
    admission = clinical.save_admission(db, runtime, user, parsed).admission
    db.commit()
    return {
        "auth": {"Authorization": f"Bearer {token}", "Content-Type": resources.FHIR_JSON},
        "user": user,
        "clinician": clinician,
        "admission": admission,
    }


def ask(db: Session, feed: dict[str, Any], *tests: LabTest, urgent: bool = False) -> list[Any]:
    rows = lab.request_tests(
        db, feed["clinician"], feed["admission"], list(tests), note="fasting", urgent=urgent
    )
    db.commit()
    return rows


def observation(request_id: Any, test: LabTest, **value: Any) -> dict[str, Any]:
    system, code, _ = mapping.LAB_TESTS[test.value]
    return {
        "resourceType": "Observation",
        "status": "final",
        "basedOn": [{"reference": f"ServiceRequest/{request_id}"}],
        "code": {"coding": [{"system": system, "code": code}]},
        **value,
    }


def post(client: TestClient, feed: dict[str, Any], body: object) -> Any:
    return client.post(POST, content=json.dumps(body), headers=feed["auth"])


def stored(db: Session, request_id: Any) -> str | None:
    result = db.scalar(select(LabResult).where(LabResult.request_id == request_id))
    return None if result is None else result.value


def test_open_requests_are_service_requests_of_the_tokens_facility(
    client: TestClient, db: Session, feed: dict[str, Any]
) -> None:
    hiv, glucose = ask(db, feed, LabTest.HIV_RAPID, LabTest.BLOOD_GLUCOSE, urgent=True)
    lab.record_result(db, ModelRuntime(Path("unused")), feed["user"], glucose.id, "5.5")
    other = make_user(db, role=Role.LAB_TECHNICIAN, facility=make_facility(db))
    other_token, _ = api_tokens.create(db, None, other.username, "Other lab")
    db.commit()

    response = client.get(SEARCH, headers=feed["auth"])
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(resources.FHIR_JSON)
    body = response.json()
    assert (body["resourceType"], body["type"], body["total"]) == ("Bundle", "searchset", 1)
    (entry,) = body["entry"]
    resource = entry["resource"]
    assert resource["id"] == str(hiv.id) and resource["status"] == "active"
    assert (resource["intent"], resource["priority"]) == ("order", "urgent")
    assert resource["code"]["coding"][0]["system"] == mapping.LAB_SYSTEM  # placeholder for HIV
    assert resource["subject"] == {"identifier": {"value": "FH-12"}}
    assert resource["encounter"]["identifier"]["value"] == feed["admission"].code
    assert resource["note"] == [{"text": "fasting"}]
    assert "Zzyzx" not in response.text  # the name is never sent

    elsewhere = client.get(SEARCH, headers={"Authorization": f"Bearer {other_token}"})
    assert elsewhere.json()["total"] == 0
    assert client.get(SEARCH).status_code == 401
    refused = client.get("/api/v1/fhir/ServiceRequest?status=completed", headers=feed["auth"])
    assert refused.status_code == 400


def test_glucose_and_protein_results_are_read_like_the_admission_bundle(
    client: TestClient, db: Session, feed: dict[str, Any]
) -> None:
    glucose, protein = ask(db, feed, LabTest.BLOOD_GLUCOSE, LabTest.URINE_PROTEIN)
    sent = observation(
        glucose.id,
        LabTest.BLOOD_GLUCOSE,
        valueQuantity={"value": 108, "code": "mg/dL", "system": mapping.UCUM},
    )
    response = post(client, feed, sent)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["resourceType"] == "Observation" and body["basedOn"] == sent["basedOn"]
    assert not any(key.startswith("value") for key in body)  # the value is not echoed
    assert stored(db, glucose.id) == "6.0"  # 108 mg/dL
    answered = post(client, feed, observation(protein.id, LabTest.URINE_PROTEIN, valueString="2+"))
    assert answered.status_code == 201 and stored(db, protein.id) == "++"
    assert client.get(SEARCH, headers=feed["auth"]).json()["total"] == 0
    assert [e.payload["test"] for e in events_of(db, "lab.resulted")] == [
        "blood_glucose",
        "urine_protein",
    ]


def test_an_hiv_result_over_fhir_rescores_the_signal(
    client: TestClient, db: Session, feed: dict[str, Any]
) -> None:
    (hiv,) = ask(db, feed, LabTest.HIV_RAPID)
    body = observation(hiv.id, LabTest.HIV_RAPID, valueCodeableConcept={"text": "Positive"})
    assert post(client, feed, body).status_code == 201
    assert stored(db, hiv.id) == "positive"
    admission: Admission = feed["admission"]
    rows = list(db.scalars(select(Prediction).where(Prediction.admission_id == admission.id)))
    assert len(rows) == 2 and any(p.feature_vector["hiv_positive"] == "Yes" for p in rows)
    for event in events_of(db, "lab.resulted"):
        assert "positive" not in json.dumps(event.payload).lower()


@pytest.mark.parametrize(
    ("change", "where"),
    [
        ({"status": "preliminary"}, "Observation.status"),
        ({"code": {"coding": [{"system": mapping.LOINC, "code": "20454-5"}]}}, "Observation.code"),
        ({"valueQuantity": {"value": 99, "code": "mmol/L"}}, "Observation.valueQuantity"),
        ({"valueQuantity": {"value": 5, "code": "g/L"}}, "Observation.valueQuantity.code"),
        ({"basedOn": []}, "Observation.basedOn"),
    ],
)
def test_a_result_the_form_would_refuse_is_an_operation_outcome(
    client: TestClient,
    db: Session,
    feed: dict[str, Any],
    change: dict[str, Any],
    where: str,
) -> None:
    (glucose,) = ask(db, feed, LabTest.BLOOD_GLUCOSE)
    body = {
        **observation(glucose.id, LabTest.BLOOD_GLUCOSE, valueQuantity={"value": 5}),
        **change,
    }
    if "valueQuantity" not in change:
        body["valueQuantity"] = {"value": 5, "code": "mmol/L"}
    response = post(client, feed, body)
    assert response.status_code == 422, response.text
    outcome = response.json()
    assert outcome["resourceType"] == "OperationOutcome"
    assert where in {e for issue in outcome["issue"] for e in issue.get("expression", [])}
    assert stored(db, glucose.id) is None


def test_unknown_closed_and_unreadable_requests(
    client: TestClient, db: Session, feed: dict[str, Any]
) -> None:
    (hiv,) = ask(db, feed, LabTest.HIV_RAPID)
    other = make_user(db, role=Role.LAB_TECHNICIAN, facility=make_facility(db))
    other_token, _ = api_tokens.create(db, None, other.username, "Other lab")
    db.commit()
    body = observation(hiv.id, LabTest.HIV_RAPID, valueString="negative")
    elsewhere = client.post(
        POST,
        content=json.dumps(body),
        headers={**feed["auth"], "Authorization": f"Bearer {other_token}"},
    )
    assert elsewhere.status_code == 404 and stored(db, hiv.id) is None
    assert post(client, feed, body).status_code == 201
    again = post(client, feed, body)
    assert again.status_code == 409 and "already recorded" in again.text
    assert post(client, feed, {"resourceType": "Patient"}).status_code == 400
    plain = client.post(POST, content="x", headers={**feed["auth"], "Content-Type": "text/plain"})
    assert plain.status_code == 415
    assert client.post(POST, content=json.dumps(body)).status_code == 401


def test_the_capability_statement_lists_the_lab_resources(client: TestClient) -> None:
    resources_listed = client.get("/api/v1/fhir/metadata").json()["rest"][0]["resource"]
    kinds = {r["type"]: r for r in resources_listed}
    assert kinds["ServiceRequest"]["interaction"] == [{"code": "search-type"}]
    assert kinds["Observation"]["interaction"] == [{"code": "create"}]
