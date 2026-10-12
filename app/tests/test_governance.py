"""Smoke tests for model governance: transitions, hot reload, shadow scoring, the approval log
and the integrity check."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from robby import events
from robby.clinical.scoring import ModelRuntime
from robby.config import Settings, default_rule_set_path
from robby.errors import FormErrors, NotAllowed
from robby.models import (
    Admission,
    ModelState,
    ModelTransition,
    ModelVersion,
    Prediction,
    PredictionMode,
    Role,
)
from robby.registry.evidence import MIN_N, build_evidence
from robby.registry.transitions import apply_transition
from robby.rulesets import load_rule_set_file
from tests.factories import events_of, make_facility, make_model_version, make_user
from tests.test_clinical_flow import GOOD, StubScorer, now_text
from tests.webutil import csrf_for, sign_in_client


class ProbScorer(StubScorer):
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict(self, frame: Any) -> list[float]:
        return [self.probability]


def stub_loader(path: Path, rule_set: Any = None) -> ProbScorer:
    if path.name.startswith("broken"):
        raise RuntimeError("cannot load")
    return ProbScorer(float(path.name))


def stub_model(db: Session, probability: str, state: ModelState) -> ModelVersion:
    model = make_model_version(db, state=state)
    model.artefact_path = probability
    db.flush()
    return model


def test_transition_rules(db: Session) -> None:
    custodian = make_user(db, role=Role.MODEL_CUSTODIAN)
    a = stub_model(db, "0.2", ModelState.ACTIVE)
    b = stub_model(db, "0.6", ModelState.CANDIDATE)
    name = custodian.username

    with pytest.raises(NotAllowed):  # a candidate cannot go straight to active
        apply_transition(db, custodian, b.id, "promote", approver_name=name, reason="x")
    with pytest.raises(FormErrors) as blank:
        apply_transition(db, custodian, b.id, "shadow", approver_name=name, reason="   ")
    assert "reason" in blank.value.errors
    with pytest.raises(FormErrors) as other:
        apply_transition(db, custodian, b.id, "shadow", approver_name="someone.else", reason="ok")
    assert "approver_name" in other.value.errors
    with pytest.raises(NotAllowed):  # only custodians approve
        apply_transition(
            db, make_user(db, role=Role.SYSTEM_ADMIN), b.id, "shadow", approver_name="x", reason="x"
        )

    apply_transition(db, custodian, b.id, "shadow", approver_name=name.upper(), reason="check")
    apply_transition(db, custodian, b.id, "promote", approver_name=name, reason="better")
    db.refresh(a)
    db.refresh(b)
    assert (a.state, b.state) == (ModelState.RETIRED, ModelState.ACTIVE)
    assert b.approved_by == custodian.id and b.approved_at is not None
    active = db.scalars(select(ModelVersion).where(ModelVersion.state == ModelState.ACTIVE)).all()
    assert len(active) == 1

    apply_transition(db, custodian, a.id, "rollback", approver_name=name, reason="regression")
    db.refresh(a)
    db.refresh(b)
    assert (a.state, b.state) == (ModelState.ACTIVE, ModelState.RETIRED)
    rows = db.scalars(select(ModelTransition).where(ModelTransition.model_id == b.id)).all()
    assert any(r.reason.startswith("Rolled back to") for r in rows)
    assert [e.payload["reason"] for e in events_of(db, "model.rolled_back")] == ["regression"]
    assert len(events_of(db, "model.promoted")) == 1 and len(events_of(db, "model.shadowed")) == 1
    assert events.verify(db).ok


@pytest.fixture
def world(app: FastAPI, client: TestClient, db: Session, settings: Settings) -> dict[str, Any]:
    load_rule_set_file(db, default_rule_set_path())
    facility = make_facility(db)
    clinician = make_user(db, role=Role.CLINICIAN, facility=facility)
    custodian = make_user(db, role=Role.MODEL_CUSTODIAN)
    db.commit()
    app.state.model_runtime = ModelRuntime(Path("unused"), loader=stub_loader)
    return {"clinician": clinician, "custodian": custodian, "settings": settings, "db": db}


def sign_in(client: TestClient, world: dict[str, Any], who: str) -> str:
    row = sign_in_client(client, world["db"], world["settings"], world[who])
    return csrf_for(world["settings"], row)


def admit(client: TestClient, world: dict[str, Any]) -> None:
    csrf = sign_in(client, world, "clinician")
    data = {**GOOD, "admitted_at": now_text(world["settings"]), "csrf_token": csrf}
    assert client.post("/admissions", data=data).status_code == 303


def active_probabilities(db: Session) -> set[str]:
    rows = db.scalars(select(Prediction).where(Prediction.mode == PredictionMode.ACTIVE))
    return {str(p.probability) for p in rows}


def test_shadow_scoring_and_hot_reload(
    client: TestClient, world: dict[str, Any], db: Session
) -> None:
    active = stub_model(db, "0.2", ModelState.ACTIVE)
    shadow = stub_model(db, "0.6", ModelState.SHADOW)
    stub_model(db, "broken", ModelState.SHADOW)  # a failing shadow must not matter
    db.commit()

    admit(client, world)
    admission = db.scalar(select(Admission))
    assert admission is not None
    preds = {
        p.model_id: p
        for p in db.scalars(select(Prediction).where(Prediction.admission_id == admission.id))
    }
    assert preds[active.id].mode is PredictionMode.ACTIVE
    assert preds[shadow.id].mode is PredictionMode.SHADOW
    assert str(preds[shadow.id].probability) == "0.60000"
    assert len(preds) == 2
    page = client.get(f"/admissions/{admission.code}")
    assert page.status_code == 200 and "60%" not in page.text  # the shadow result is never shown

    # the custodian promotes the shadow model through the page; no restart, next admission uses it
    csrf = sign_in(client, world, "custodian")
    custodian = world["custodian"]
    bad = client.post(
        f"/models/{shadow.version_label}/transition",
        data={"action": "promote", "approver_name": "nobody", "reason": "r", "csrf_token": csrf},
    )
    assert bad.status_code == 400 and 'class="dialog-backdrop open"' in bad.text
    ok = client.post(
        f"/models/{shadow.version_label}/transition",
        data={
            "action": "promote",
            "approver_name": custodian.username,
            "reason": "Better calibration",
            "csrf_token": csrf,
        },
    )
    assert ok.status_code == 303
    detail = client.get(f"/models/{shadow.version_label}?done=promote")
    assert "promoted to active" in detail.text
    admit(client, world)
    assert active_probabilities(db) == {"0.20000", "0.60000"}  # the second admission used v-new


def test_pages_are_for_custodians_only(client: TestClient, world: dict[str, Any]) -> None:
    sign_in(client, world, "clinician")
    for path in ("/models", "/approval-log"):
        assert client.get(path).status_code == 403
    assert client.post("/approval-log/verify").status_code in (403, 400)


def test_evidence_needs_enough_outcomes(db: Session) -> None:
    stub_model(db, "0.2", ModelState.ACTIVE)
    shadow = stub_model(db, "0.6", ModelState.SHADOW)
    result = build_evidence(db, shadow)
    assert result.status == "not_enough"
    assert f"0 of {MIN_N} needed" in result.message


def test_models_pages_render(client: TestClient, world: dict[str, Any], db: Session) -> None:
    stub_model(db, "0.2", ModelState.ACTIVE)
    shadow = stub_model(db, "0.6", ModelState.SHADOW)
    db.commit()
    sign_in(client, world, "custodian")
    listing = client.get("/models")
    assert listing.status_code == 200 and shadow.version_label in listing.text
    detail = client.get(f"/models/{shadow.version_label}")
    assert "Not enough recorded outcomes yet" in detail.text
    assert "style=" not in detail.text and "onclick" not in detail.text
    assert client.get("/models/no-such-version").status_code == 404


def test_approval_log_and_integrity_check(
    client: TestClient, world: dict[str, Any], db: Session
) -> None:
    custodian = world["custodian"]
    model = stub_model(db, "0.2", ModelState.CANDIDATE)
    apply_transition(
        db,
        custodian,
        model.id,
        "shadow",
        approver_name=custodian.username,
        reason="Looks good in validation",
    )
    events.append(
        db,
        custodian,
        "export.created",
        ("export_record", model.id),
        {"facility_code": "FA", "period_month": "2026-09", "sha256": "f" * 64},
    )
    events.append(db, custodian, "session.login", None, {"secret": "do-not-show"})
    db.commit()
    csrf = sign_in(client, world, "custodian")

    page = client.get("/approval-log")
    assert "Looks good in validation" in page.text and "Moved to shadow" in page.text
    assert "Exported monthly report" in page.text
    assert "f" * 64 not in page.text and "do-not-show" not in page.text
    assert "Not checked yet" in page.text
    only_models = client.get("/approval-log?type=corrections")
    assert "Moved to shadow" not in only_models.text

    client.post("/approval-log/verify", data={"csrf_token": csrf})
    assert "Log intact:" in client.get("/approval-log").text

    first = events_of(db, "model.shadowed")[0]
    db.execute(text("ALTER TABLE event_log DISABLE TRIGGER event_log_immutable"))
    db.execute(
        text("UPDATE event_log SET payload = CAST(:p AS jsonb) WHERE seq = :s"),
        {"p": '{"reason": "rewritten"}', "s": first.seq},
    )
    db.execute(text("ALTER TABLE event_log ENABLE TRIGGER event_log_immutable"))
    db.commit()
    done = client.post("/approval-log/verify", data={"csrf_token": csrf})
    assert done.status_code == 303
    assert f"Log altered at entry #{first.seq}" in client.get("/approval-log").text
    recorded = events_of(db, "events.verified")
    assert recorded[-1].payload["result"] == "altered" and "reason" in recorded[-1].payload
