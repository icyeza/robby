"""Test data builders. Each builder flushes, so ids and defaults exist when it returns."""

from __future__ import annotations

import itertools
import uuid
from datetime import date
from decimal import Decimal
from functools import lru_cache
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from robby.auth.passwords import hash_password
from robby.models import (
    FACILITY_ROLES,
    Admission,
    AdmissionSource,
    AppUser,
    Classification,
    ClassificationStatus,
    DeliveryMode,
    DeliveryOutcome,
    EventLog,
    Facility,
    ModelState,
    ModelVersion,
    OmissionField,
    OmissionReason,
    Prediction,
    PredictionMode,
    ReconciledOutcome,
    Role,
    RuleSet,
    ScreeningOmission,
)
from robby.timeutil import utcnow

DEFAULT_PASSWORD = "correct-horse-battery"
_sequence = itertools.count(1)


@lru_cache(maxsize=1)
def default_password_hash() -> str:
    return hash_password(DEFAULT_PASSWORD)


def _letters(n: int) -> str:
    out = ""
    while True:
        n, remainder = divmod(n, 26)
        out = chr(ord("A") + remainder) + out
        if n == 0:
            return out


def make_facility(
    db: Session,
    *,
    name: str | None = None,
    code: str | None = None,
    district: str = "District 1",
    active: bool = True,
) -> Facility:
    n = next(_sequence)
    facility = Facility(
        name=name or f"Facility {n}",
        code=code or f"T{_letters(n):A>3}",
        district=district,
        active=active,
    )
    db.add(facility)
    db.flush()
    return facility


def make_user(
    db: Session,
    *,
    role: Role = Role.CLINICIAN,
    facility: Facility | None = None,
    username: str | None = None,
    password: str | None = None,
    must_change: bool = False,
    active: bool = True,
) -> AppUser:
    if role in FACILITY_ROLES and facility is None:
        facility = make_facility(db)
    user = AppUser(
        username=username or f"user{next(_sequence)}",
        password_hash=hash_password(password) if password else default_password_hash(),
        role=role,
        facility_id=facility.id if role in FACILITY_ROLES and facility is not None else None,
        must_change_password=must_change,
        active=active,
    )
    db.add(user)
    db.flush()
    return user


def make_rule_set(db: Session, *, active: bool = False) -> RuleSet:
    rule_set = RuleSet(
        version_label=f"test-{uuid.uuid4().hex[:8]}",
        checksum="0" * 64,
        definition={"groups": []},
        is_active=active,
    )
    db.add(rule_set)
    db.flush()
    return rule_set


def make_model_version(db: Session, *, state: ModelState = ModelState.CANDIDATE) -> ModelVersion:
    model = ModelVersion(
        version_label=f"model-{uuid.uuid4().hex[:8]}",
        algorithm="logistic_regression",
        feature_set="core",
        feature_list=["parity"],
        hyperparameters={},
        calibration_method="none",
        evaluation_metrics={},
        artefact_hash="0" * 64,
        artefact_path="artefacts/test",
        manifest={},
        training_window_start=date(2020, 1, 1),
        training_window_end=date(2021, 12, 31),
        state=state,
    )
    db.add(model)
    db.flush()
    return model


def make_clinical_chain(db: Session) -> dict[str, Any]:
    """One row in every immutable table, linked the way later sub-projects will link them."""
    facility = make_facility(db)
    clinician = make_user(db, role=Role.CLINICIAN, facility=facility)
    rule_set = make_rule_set(db)
    model = make_model_version(db)
    admission = Admission(
        id=uuid.uuid4(),
        facility_id=facility.id,
        code=f"{facility.code}-0001",
        admitted_at=utcnow(),
        source=AdmissionSource.FORM,
        entered_by=clinician.id,
        parity=0,
    )
    omission = ScreeningOmission(
        admission_id=admission.id,
        field=OmissionField.GLUCOSE,
        reason=OmissionReason.EQUIPMENT_UNAVAILABLE,
    )
    classification = Classification(
        admission_id=admission.id,
        rule_set_id=rule_set.id,
        status=ClassificationStatus.PARTIAL,
        candidates=[1, 2],
        resolving_fields=["onset_of_labour"],
        conflict_fields=[],
        trace={},
    )
    prediction = Prediction(
        id=uuid.uuid4(),
        admission_id=admission.id,
        model_id=model.id,
        mode=PredictionMode.ACTIVE,
        probability=Decimal("0.31000"),
        feature_vector={"parity": 0},
    )
    outcome = DeliveryOutcome(
        id=uuid.uuid4(),
        admission_id=admission.id,
        mode=DeliveryMode.CESAREAN,
        delivered_at=utcnow(),
        recorded_by=clinician.id,
    )
    reconciled = ReconciledOutcome(
        prediction_id=prediction.id, delivery_outcome_id=outcome.id, cesarean=True
    )
    event = EventLog(
        id=uuid.uuid4(),
        occurred_at=utcnow(),
        event_type="test.event",
        payload={},
        prev_hash="0" * 64,
        hash=uuid.uuid4().hex * 2,
    )
    # No relationship() links exist, so the unit of work cannot order inserts by foreign key:
    # flush parents before children explicitly.
    db.add(admission)
    db.flush()
    db.add_all([omission, classification, prediction, outcome])
    db.flush()
    db.add_all([reconciled, event])
    db.flush()
    return {
        "facility": facility,
        "clinician": clinician,
        "admission": admission,
        "classification": classification,
        "prediction": prediction,
        "outcome": outcome,
    }


def events_of(db: Session, event_type: str) -> list[EventLog]:
    return list(
        db.scalars(select(EventLog).where(EventLog.event_type == event_type).order_by(EventLog.seq))
    )
