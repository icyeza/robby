"""POST /api/v1/fhir/Bundle and GET /api/v1/fhir/metadata (FR8)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from robby.auth import api_tokens
from robby.clinical.scoring import ModelRuntime
from robby.config import Settings, default_rule_set_path
from robby.fhir import resources
from robby.models import Admission, AdmissionSource, ModelState, Role, ScreeningOmission
from robby.rulesets import load_rule_set_file
from tests.factories import events_of, make_facility, make_model_version, make_user
from tests.test_clinical_flow import StubScorer
from tests.test_fhir_adapter import EXAMPLES, admission, bundle, condition, obs

URL = "/api/v1/fhir/Bundle"


@pytest.fixture
def feed(app: FastAPI, client: TestClient, db: Session, settings: Settings) -> dict[str, Any]:
    load_rule_set_file(db, default_rule_set_path())
    facility = make_facility(db)
    user = make_user(db, role=Role.LAB_TECHNICIAN, facility=facility)
    make_model_version(db, state=ModelState.ACTIVE)
    token, _ = api_tokens.create(db, None, user.username, "FHIR feed")
    db.commit()
    app.state.model_runtime = ModelRuntime(
        Path("unused"), loader=lambda path, rule_set=None: StubScorer()
    )
    return {
        "auth": {"Authorization": f"Bearer {token}", "Content-Type": resources.FHIR_JSON},
        "user": user,
        "facility": facility,
        "zone": settings.zone,
    }


def hours_ago(feed: dict[str, Any], hours: float) -> str:
    return (datetime.now(feed["zone"]) - timedelta(hours=hours)).isoformat(timespec="seconds")


def example(feed: dict[str, Any], name: str = "two_admissions") -> dict[str, Any]:
    """An example Bundle with its Encounters moved to the last few hours."""
    b = json.loads((EXAMPLES / f"{name}.json").read_text(encoding="utf-8"))
    hours = 3.0
    for entry in b["entry"]:
        if entry["resource"]["resourceType"] == "Encounter":
            entry["resource"].setdefault("period", {})["start"] = hours_ago(feed, hours)
            hours -= 1
    return b


def post(client: TestClient, feed: dict[str, Any], body: object) -> Any:
    return client.post(URL, content=json.dumps(body), headers=feed["auth"])


def count(db: Session) -> int:
    return db.scalar(select(func.count()).select_from(Admission)) or 0


def test_a_bundle_is_saved_as_fhir_admissions_of_the_tokens_facility(
    client: TestClient, feed: dict[str, Any], db: Session
) -> None:
    response = post(client, feed, example(feed))
    assert response.status_code == 201
    assert response.headers["content-type"].startswith(resources.FHIR_JSON)
    body = response.json()
    assert (body["resourceType"], body["type"]) == ("Bundle", "transaction-response")
    saved = list(db.scalars(select(Admission).order_by(Admission.code)))
    assert [(a.patient_number, a.source, a.facility_id) for a in saved] == [
        ("FX-1001", AdmissionSource.FHIR, feed["facility"].id),
        ("FX-1002", AdmissionSource.FHIR, feed["facility"].id),
    ]
    assert all(a.entered_by == feed["user"].id for a in saved)
    first, second = (e["response"] for e in body["entry"])
    assert first["status"] == "201 Created"
    assert first["location"] == f"/admissions/{saved[0].code}"
    extensions = {x["url"]: next(v for k, v in x.items() if k != "url") for x in first["extension"]}
    assert extensions[resources.ADMISSION_CODE] == saved[0].code
    assert extensions[resources.ROBSON_GROUP] == 3  # parity 1, no previous cesarean, 39 weeks
    assert extensions[resources.READINESS] == 71  # StubScorer: 0.714
    assert "Robson group 3; readiness signal 71%" in first["outcome"]["issue"][0]["diagnostics"]
    assert {x["url"]: x for x in second["extension"]}[resources.ROBSON_GROUP]["valueInteger"] == 5
    omitted = db.scalars(
        select(ScreeningOmission).where(ScreeningOmission.admission_id == saved[1].id)
    )
    assert [(o.field.value, o.reason.value) for o in omitted] == [
        ("glucose", "equipment_unavailable")
    ]
    event = events_of(db, "admissions.imported")[0]
    assert (event.payload["channel"], event.payload["rows"]) == ("fhir", 2)
    assert "FX-1001" not in json.dumps(event.payload)


def test_plain_json_is_accepted_too(client: TestClient, feed: dict[str, Any], db: Session) -> None:
    headers = {**feed["auth"], "Content-Type": "application/json"}
    response = client.post(URL, content=json.dumps(example(feed)), headers=headers)
    assert response.status_code == 201 and count(db) == 2


def test_one_bad_admission_saves_nothing_and_names_every_element(
    client: TestClient, feed: dict[str, Any], db: Session
) -> None:
    good = admission("p1", "e1", "FX-1", hours_ago(feed, 2))
    bad = admission("p2", "e2", "FX-2", hours_ago(feed, 1))
    bad[2] = obs("e2-par", "11977-6", enc="e2", valueInteger=31)
    response = post(client, feed, bundle(*good, *bad))
    assert response.status_code == 422
    outcome = response.json()
    assert outcome["resourceType"] == "OperationOutcome"
    assert [(i["severity"], i["expression"]) for i in outcome["issue"]] == [
        ("error", ["Observation/e2-par.valueInteger"])
    ]
    assert count(db) == 0
    rejected = events_of(db, "api_import.rejected")[0]
    assert rejected.payload == {"facility": feed["facility"].code, "problems": 1, "channel": "fhir"}


def test_the_invalid_example_is_refused_whole(
    client: TestClient, feed: dict[str, Any], db: Session
) -> None:
    response = post(client, feed, example(feed, "invalid"))
    assert response.status_code == 422
    expressions = {i["expression"][0] for i in response.json()["issue"]}
    assert "Observation/obs-9-glucose.valueQuantity.code" in expressions
    assert count(db) == 0


def test_resending_skips_what_is_already_recorded(
    client: TestClient, feed: dict[str, Any], db: Session
) -> None:
    b = example(feed)
    first = post(client, feed, b).json()
    again = post(client, feed, b)
    assert again.status_code == 200
    assert [e["response"]["status"] for e in again.json()["entry"]] == ["200 OK", "200 OK"]
    assert [e["response"]["location"] for e in again.json()["entry"]] == [
        e["response"]["location"] for e in first["entry"]
    ]
    assert "Already recorded as" in again.text
    assert count(db) == 2
    assert events_of(db, "api_import.nothing_new")[0].payload["channel"] == "fhir"


def test_an_encounter_repeated_in_the_bundle_is_saved_once(
    client: TestClient, feed: dict[str, Any], db: Session
) -> None:
    when = hours_ago(feed, 2)
    twice = [r for r in admission("p1", "e2", "FX-1", when) if r["resourceType"] != "Patient"]
    response = post(client, feed, bundle(*admission("p1", "e1", "FX-1", when), *twice))
    assert response.status_code == 201
    first, second = (e["response"] for e in response.json()["entry"])
    assert (first["status"], second["status"]) == ("201 Created", "200 OK")
    assert first["location"] == second["location"]
    assert "Same patient and admission time as Encounter/e1" in json.dumps(second)
    assert count(db) == 1


def test_a_repeat_is_found_by_blind_index_whatever_the_case_of_the_number(
    client: TestClient, feed: dict[str, Any], db: Session
) -> None:
    when = hours_ago(feed, 2)
    again = admission("p2", "e2", "fx-1", when)
    response = post(client, feed, bundle(*admission("p1", "e1", "FX-1", when), *again))
    assert response.status_code == 201
    assert [e["response"]["status"] for e in response.json()["entry"]] == ["201 Created", "200 OK"]
    saved = db.scalars(select(Admission)).one()
    assert saved.patient_number == "FX-1" and saved.patient_number_index is not None
    assert post(client, feed, bundle(*again)).json()["entry"][0]["response"]["status"] == "200 OK"


def test_a_condition_fills_the_models_input(
    client: TestClient, feed: dict[str, Any], db: Session
) -> None:
    b = bundle(*admission("p1", "e1", "FX-1", hours_ago(feed, 1)), condition("86406008"))
    assert post(client, feed, b).status_code == 201
    assert db.scalars(select(Admission)).one().details == {"hiv_positive": "Yes"}


@pytest.mark.parametrize(
    ("headers", "body", "status", "code"),
    [
        ({"Authorization": "Bearer rbk_wrong"}, "{}", 401, "login"),
        ({"Authorization": None}, "{}", 401, "login"),
        ({"Content-Type": "text/plain"}, "{}", 415, "not-supported"),
        ({}, "{not json", 400, "structure"),
        ({}, '{"resourceType": "Patient"}', 400, "structure"),
    ],
)
def test_refusals_are_operation_outcomes(
    client: TestClient,
    feed: dict[str, Any],
    db: Session,
    headers: dict[str, str | None],
    body: str,
    status: int,
    code: str,
) -> None:
    merged = {**feed["auth"], **headers}
    sent = {k: v for k, v in merged.items() if v is not None}  # None: header left out
    response = client.post(URL, content=body, headers=sent)
    assert response.status_code == status
    assert response.json()["resourceType"] == "OperationOutcome"
    assert response.json()["issue"][0]["code"] == code
    if status == 401:
        assert response.headers["www-authenticate"] == "Bearer"
    assert count(db) == 0


def test_metadata_is_a_capability_statement(client: TestClient) -> None:
    response = client.get("/api/v1/fhir/metadata")
    assert response.status_code == 200
    body = response.json()
    assert (body["resourceType"], body["fhirVersion"]) == ("CapabilityStatement", "4.0.1")
    kinds = {r["type"] for r in body["rest"][0]["resource"]}
    assert kinds == {"Bundle", "Patient", "Encounter", "Observation", "Condition"}
