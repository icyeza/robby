"""Smoke tests for reconciliation and monitoring alerts."""

from __future__ import annotations

import uuid
from datetime import timedelta
from decimal import Decimal

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from robby.config import Settings
from robby.models import (
    Admission,
    AdmissionSource,
    Classification,
    ClassificationStatus,
    DeliveryMode,
    DeliveryOutcome,
    Facility,
    ModelState,
    ModelVersion,
    Prediction,
    PredictionMode,
    ReconciledOutcome,
    Role,
    RuleSet,
)
from robby.monitoring import service
from robby.monitoring.reconcile import reconcile
from robby.timeutil import utcnow
from tests.factories import (
    events_of,
    make_facility,
    make_model_version,
    make_rule_set,
    make_user,
)
from tests.webutil import csrf_for, sign_in_client


def add_admission(
    db: Session,
    facility: Facility,
    rule_set: RuleSet,
    model: ModelVersion,
    *,
    probability: str | None = "0.30000",
    group: int | None = 1,
    supersedes: Admission | None = None,
    hours_ago: float = 1,
    outcome: DeliveryMode | None = None,
    recorded_by: uuid.UUID | None = None,
) -> Admission:
    admission = Admission(
        id=uuid.uuid4(),
        facility_id=facility.id,
        code=f"{facility.code}-{uuid.uuid4().hex[:8]}",
        admitted_at=utcnow() - timedelta(hours=hours_ago),
        source=AdmissionSource.FHIR,
        supersedes_id=supersedes.id if supersedes else None,
        correction_reason="fix" if supersedes else None,
        parity=0,
    )
    db.add(admission)
    db.flush()
    db.add(
        Classification(
            admission_id=admission.id,
            rule_set_id=rule_set.id,
            status=ClassificationStatus.RESOLVED if group else ClassificationStatus.PARTIAL,
            robson_group=group,
            candidates=[],
            resolving_fields=[],
            conflict_fields=[],
            trace={},
        )
    )
    if probability is not None:
        db.add(
            Prediction(
                admission_id=admission.id,
                model_id=model.id,
                mode=PredictionMode.ACTIVE,
                probability=Decimal(probability),
                feature_vector={"parity": 0},
            )
        )
    db.flush()
    if outcome is not None:
        assert recorded_by is not None
        db.add(
            DeliveryOutcome(
                admission_id=admission.id,
                mode=outcome,
                delivered_at=utcnow(),
                recorded_by=recorded_by,
            )
        )
        db.flush()
    return admission


def test_reconcile_follows_the_chain_and_is_idempotent(db: Session) -> None:
    facility = make_facility(db)
    user = make_user(db, role=Role.CLINICIAN, facility=facility)
    rules, model = make_rule_set(db), make_model_version(db)
    original = add_admission(db, facility, rules, model)
    corrected = add_admission(
        db, facility, rules, model, supersedes=original, outcome=DeliveryMode.CESAREAN,
        recorded_by=user.id,
    )  # fmt: skip
    lonely = add_admission(db, facility, rules, model)  # no outcome: stays pending

    counts = reconcile(db)
    assert counts["reconciled"] == 2 and counts["awaiting"] >= 1
    rows = db.scalars(select(ReconciledOutcome)).all()
    assert len(rows) == 2 and all(r.cesarean for r in rows)
    reconciled_for = {
        db.scalar(select(Prediction.admission_id).where(Prediction.id == r.prediction_id))
        for r in rows
    }
    assert reconciled_for == {original.id, corrected.id} and lonely.id not in reconciled_for

    again = reconcile(db)
    assert again["reconciled"] == 0
    assert len(db.scalars(select(ReconciledOutcome)).all()) == 2
    runs = events_of(db, "reconciliation.run")
    assert len(runs) == 2 and runs[0].payload["reconciled"] == 2
    assert reconcile(db, log_empty=False)["reconciled"] == 0
    assert len(events_of(db, "reconciliation.run")) == 2


def _calibration_world(db: Session) -> tuple[Facility, ModelVersion]:
    facility = make_facility(db, name="Facility B")
    user = make_user(db, role=Role.CLINICIAN, facility=facility)
    rules = make_rule_set(db)
    model = make_model_version(db, state=ModelState.ACTIVE)
    for i in range(60):  # predicted 20%, observed 50%
        add_admission(
            db, facility, rules, model, probability="0.20000", hours_ago=96,
            outcome=DeliveryMode.CESAREAN if i % 2 else DeliveryMode.VAGINAL_SPONTANEOUS,
            recorded_by=user.id,
        )  # fmt: skip
    reconcile(db)
    return facility, model


def test_calibration_alert_and_acknowledgement(
    client: TestClient, db: Session, settings: Settings
) -> None:
    _calibration_world(db)
    data = service.build(db)
    keys = [a.key for a in data.alerts]
    assert len(keys) == 1 and keys[0].startswith("calibration:")
    assert "30 points above predicted" in data.alerts[0].text
    assert data.open_alerts == 1
    facility_row = data.facilities[0]
    assert facility_row.n == 60 and facility_row.diff_pts == 30 and facility_row.flagged

    custodian = make_user(db, role=Role.MODEL_CUSTODIAN)
    row = sign_in_client(client, db, settings, custodian)
    page = client.get("/monitoring")
    assert page.status_code == 200 and "Facility B" in page.text and "Acknowledge" in page.text
    posted = client.post(
        "/monitoring/alerts/acknowledge",
        data={"key": keys[0], "csrf_token": csrf_for(settings, row)},
    )
    assert posted.status_code == 303
    assert len(events_of(db, "monitoring.alert_acknowledged")) == 1
    assert service.build(db).open_alerts == 0
    assert "Acknowledged by" in client.get("/monitoring").text
    stale = client.post(
        "/monitoring/alerts/acknowledge",
        data={"key": "calibration:nope:2000-W01", "csrf_token": csrf_for(settings, row)},
    )
    assert stale.status_code == 404


def test_monitoring_without_data_says_so(
    client: TestClient, db: Session, settings: Settings
) -> None:
    sign_in_client(client, db, settings, make_user(db, role=Role.MODEL_CUSTODIAN))
    page = client.get("/monitoring")
    assert page.status_code == 200 and "no active model" in page.text.lower()
