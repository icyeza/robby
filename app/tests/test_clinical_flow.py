"""Smoke tests for the clinical flow: form validation, classification paths, saving, the result
page, degradation without a model, corrections and facility scoping."""

from __future__ import annotations

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
from tests.factories import make_facility, make_model_version, make_user
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
    user = make_user(db, role=Role.CLINICIAN, facility=facility)
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
    assert admission is not None and response.headers["location"] == f"/admissions/{admission.code}"
    classification = db.scalar(select(Classification))
    assert (
        classification is not None
        and classification.robson_group == 5
        and classification.subgroup == "5a"
    )
    prediction = db.scalar(select(Prediction))
    assert prediction is not None and float(prediction.probability) == pytest.approx(0.714)
    page = client.get(response.headers["location"])
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
    page = client.get(response.headers["location"])
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
    assert "No Robson group assigned" in client.get(partial.headers["location"]).text
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


def test_correction_adds_a_row_and_leaves_the_original(
    client: TestClient, world: dict[str, Any], db: Session
) -> None:
    activate_model(db)
    first = post_form(client, world)
    original = db.scalar(select(Admission))
    assert original is not None
    snapshot = (original.parity, original.previous_cs_count, original.code)
    page = client.get(f"/admissions/{original.code}/correct")
    assert page.status_code == 200 and "You are correcting" in page.text
    no_reason = post_form(client, world, f"/admissions/{original.code}/correct", parity="3")
    assert no_reason.status_code == 400 and "Say why" in no_reason.text
    done = post_form(
        client,
        world,
        f"/admissions/{original.code}/correct",
        parity="3",
        correction_reason="Parity miscounted",
    )
    assert done.status_code == 303
    db.expire_all()
    rows = list(db.scalars(select(Admission).order_by(Admission.created_at)))
    assert len(rows) == 2 and rows[1].supersedes_id == original.id and rows[1].parity == 3
    assert (rows[0].parity, rows[0].previous_cs_count, rows[0].code) == snapshot
    assert "Replaces" in client.get(done.headers["location"]).text
    assert "Replaced by" in client.get("/admissions").text
    assert client.get(f"/admissions/{original.code}/correct").status_code == 409
    assert first.status_code == 303


def test_outcome_and_facility_scoping(
    client: TestClient, world: dict[str, Any], db: Session, settings: Settings
) -> None:
    code = client.get(post_form(client, world).headers["location"]).url.path.rsplit("/", 1)[1]
    outcome = client.post(
        f"/admissions/{code}/outcome",
        data={"mode": "cesarean", "delivered_at": now_text(settings), "csrf_token": world["csrf"]},
    )
    assert outcome.status_code == 303
    assert "Cesarean" in client.get("/admissions").text
    # another facility's clinician cannot see it
    client.cookies.clear()
    other = make_user(db, role=Role.CLINICIAN, facility=make_facility(db))
    sign_in_client(client, db, settings, other)
    assert client.get(f"/admissions/{code}").status_code == 404
    assert client.get(f"/admissions/{code}/outcome").status_code == 404
    assert code not in client.get("/admissions").text
