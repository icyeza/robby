"""Saving and reading admissions. One transaction per save: admission, screening omissions,
classification, prediction (only for a resolved group and a working model), event.

Clinical rows are never updated. A correction is a new admission whose supersedes_id names the
original; "replaced" is derived by query."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from robby import events
from robby.clinical import classify as cls
from robby.clinical import scoring
from robby.clinical.form import Parsed, omissions_of
from robby.errors import NotAllowed, NotFound
from robby.facilities.admission_codes import next_admission_code
from robby.models import (
    Admission,
    AdmissionSource,
    AppUser,
    Classification,
    ClassificationStatus,
    DeliveryMode,
    DeliveryOutcome,
    Facility,
    ModelState,
    ModelTransition,
    ModelVersion,
    Prediction,
    PredictionMode,
    ScreeningOmission,
)
from robby.timeutil import utcnow

logger = logging.getLogger("robby")
RECENT_WINDOW = timedelta(hours=72)


@dataclass
class Saved:
    admission: Admission
    classification: Classification
    prediction: Prediction | None
    scoring_state: str  # ok | no_model | failed | not_resolved


def classify_parsed(parsed: Parsed, rule_set: Any) -> tuple[Any, dict[str, Any]]:
    inputs = cls.robson_inputs(
        parsed.parity,
        parsed.previous_cs_count,
        parsed.plurality,
        parsed.presentation,
        parsed.ga_weeks,
        parsed.onset,
    )
    return cls.classify_inputs(inputs, rule_set)


def save_admission(
    db: Session,
    runtime: scoring.ModelRuntime,
    user: AppUser,
    parsed: Parsed,
    *,
    supersedes: Admission | None = None,
) -> Saved:
    """Raises cls.NoRuleSetError when no rule set is active. Does not commit."""
    assert user.facility_id is not None and parsed.admitted_at is not None
    facility = db.get(Facility, user.facility_id)
    assert facility is not None
    rule_row, engine_rules = cls.active_rule_set(db)
    result, trace = classify_parsed(parsed, engine_rules)

    admission = Admission(
        id=uuid.uuid4(),
        facility_id=facility.id,
        code=next_admission_code(db, facility),
        admitted_at=parsed.admitted_at,
        source=AdmissionSource.FORM,
        entered_by=user.id,
        supersedes_id=supersedes.id if supersedes else None,
        correction_reason=parsed.correction_reason if supersedes else None,
        parity=parsed.parity,
        previous_cs_count=parsed.previous_cs_count,
        plurality=parsed.plurality,
        fetal_presentation=parsed.presentation,
        gestational_age_weeks=Decimal(parsed.ga_weeks) if parsed.ga_weeks is not None else None,
        onset_of_labour=parsed.onset,
        maternal_age=parsed.maternal_age,
        systolic_bp=parsed.systolic,
        diastolic_bp=parsed.diastolic,
        proteinuria=parsed.proteinuria,
        glucose_mmol=parsed.glucose,
        details=dict(parsed.details),
    )
    db.add(admission)
    db.flush()
    for field, reason in omissions_of(parsed):
        db.add(ScreeningOmission(admission_id=admission.id, field=field, reason=reason))
    classification = Classification(
        admission_id=admission.id,
        rule_set_id=rule_row.id,
        status=ClassificationStatus(result.status),
        robson_group=result.group,
        subgroup=result.subgroup,
        candidates=sorted(result.candidates),
        resolving_fields=list(result.resolving_fields),
        conflict_fields=list(result.conflict_fields),
        trace=trace,
    )
    db.add(classification)
    db.flush()

    prediction = None
    state = "not_resolved"
    model_label: str | None = None
    if result.status == "resolved":
        # NFR7: whatever goes wrong here, the admission and its classification are kept.
        model = runtime.active(db, rule_row.id, engine_rules)
        if model is None:
            state = "no_model"
        else:
            model_label = model.row.version_label
            try:
                with db.begin_nested():
                    scored = scoring.score(model, parsed, facility)
                    prediction = Prediction(
                        admission_id=admission.id,
                        model_id=scored.model_id,
                        mode=PredictionMode.ACTIVE,
                        probability=scored.probability,
                        feature_vector=scored.feature_vector,
                    )
                    db.add(prediction)
                    db.flush()
                state = "ok"
            except Exception as exc:
                logger.warning("Scoring failed for %s: %s", admission.code, type(exc).__name__)
                prediction = None
                state = "failed"

    payload: dict[str, Any] = {
        "code": admission.code,
        "facility": facility.code,
        "status": result.status,
        "group": result.group,
        "rule_set": rule_row.version_label,
        "scoring": state,
        "model": model_label,
        "omitted": sorted(f.value for f, _ in omissions_of(parsed)),
    }
    if supersedes is not None:
        payload["supersedes"] = supersedes.code
        payload["reason"] = parsed.correction_reason
    events.append(
        db,
        user,
        "admission.corrected" if supersedes else "admission.created",
        ("admission", admission.id),
        payload,
    )
    return Saved(admission, classification, prediction, state)


# ---------------------------------------------------------------- reading


def get_admission(db: Session, user: AppUser, code: str) -> Admission:
    """The facility's own admission, or NotFound (also for another facility's code)."""
    row = db.scalar(
        select(Admission).where(Admission.code == code, Admission.facility_id == user.facility_id)
    )
    if row is None:
        raise NotFound(code)
    return row


def replacement_of(db: Session, admission: Admission) -> Admission | None:
    return db.scalar(select(Admission).where(Admission.supersedes_id == admission.id))


def chain_outcome(db: Session, admission: Admission) -> DeliveryOutcome | None:
    """The outcome of this admission or of any admission it replaced.

    An outcome stays linked to the admission it was recorded on. If that admission is later
    corrected, the outcome is not moved or copied: reconciliation (and this lookup) follow the
    supersedes chain from the current admission back to the original."""
    seen: set[uuid.UUID] = set()
    current: Admission | None = admission
    while current is not None and current.id not in seen:
        seen.add(current.id)
        found = db.scalar(select(DeliveryOutcome).where(DeliveryOutcome.admission_id == current.id))
        if found is not None:
            return found
        current = db.get(Admission, current.supersedes_id) if current.supersedes_id else None
    return None


def classification_of(db: Session, admission: Admission) -> Classification:
    row = db.scalar(
        select(Classification)
        .where(Classification.admission_id == admission.id)
        .order_by(Classification.created_at.desc())
        .limit(1)
    )
    assert row is not None
    return row


def prediction_of(db: Session, admission: Admission) -> tuple[Prediction, ModelVersion] | None:
    row = db.execute(
        select(Prediction, ModelVersion)
        .join(ModelVersion, ModelVersion.id == Prediction.model_id)
        .where(Prediction.admission_id == admission.id, Prediction.mode == PredictionMode.ACTIVE)
        .order_by(Prediction.created_at.desc())
        .limit(1)
    ).first()
    return (row[0], row[1]) if row else None


def activated_at(db: Session, model: ModelVersion) -> datetime:
    at = db.scalar(
        select(ModelTransition.at)
        .where(ModelTransition.model_id == model.id, ModelTransition.to_state == ModelState.ACTIVE)
        .order_by(ModelTransition.at.desc())
        .limit(1)
    )
    return at or model.registered_at


def omissions_for(db: Session, admission: Admission) -> list[ScreeningOmission]:
    return list(
        db.scalars(select(ScreeningOmission).where(ScreeningOmission.admission_id == admission.id))
    )


@dataclass
class RecentRow:
    admission: Admission
    group: int | None
    percent: int | None
    outcome: DeliveryOutcome | None
    replaced_by: str | None


def percent_of(probability: Decimal) -> int:
    return int((probability * 100).to_integral_value(rounding="ROUND_HALF_UP"))


def recent(db: Session, user: AppUser, now: datetime | None = None) -> list[RecentRow]:
    since = (now or utcnow()) - RECENT_WINDOW
    admissions = list(
        db.scalars(
            select(Admission)
            .where(Admission.facility_id == user.facility_id, Admission.admitted_at >= since)
            .order_by(Admission.admitted_at.desc(), Admission.created_at.desc())
        )
    )
    if not admissions:
        return []
    ids = [a.id for a in admissions]
    groups = {
        c.admission_id: c.robson_group
        for c in db.scalars(select(Classification).where(Classification.admission_id.in_(ids)))
    }
    percents = {
        p.admission_id: percent_of(p.probability)
        for p in db.scalars(
            select(Prediction).where(
                Prediction.admission_id.in_(ids), Prediction.mode == PredictionMode.ACTIVE
            )
        )
    }
    replaced = {
        r.supersedes_id: r.code
        for r in db.scalars(select(Admission).where(Admission.supersedes_id.in_(ids)))
    }
    # outcomes of the chain: collect ancestors of the listed admissions, then the outcomes
    parent = {a.id: a.supersedes_id for a in admissions}
    frontier = {p for p in parent.values() if p and p not in parent}
    while frontier:
        found = list(db.scalars(select(Admission).where(Admission.id.in_(frontier))))
        for a in found:
            parent[a.id] = a.supersedes_id
        frontier = {
            a.supersedes_id for a in found if a.supersedes_id and a.supersedes_id not in parent
        }
    outcomes = {
        o.admission_id: o
        for o in db.scalars(
            select(DeliveryOutcome).where(DeliveryOutcome.admission_id.in_(list(parent)))
        )
    }

    def outcome_for(admission_id: uuid.UUID) -> DeliveryOutcome | None:
        current: uuid.UUID | None = admission_id
        while current is not None:
            if current in outcomes:
                return outcomes[current]
            current = parent.get(current)
        return None

    return [
        RecentRow(a, groups.get(a.id), percents.get(a.id), outcome_for(a.id), replaced.get(a.id))
        for a in admissions
    ]


# ---------------------------------------------------------------- outcomes


def record_outcome(
    db: Session, user: AppUser, admission: Admission, mode: DeliveryMode, delivered_at: datetime
) -> DeliveryOutcome:
    if replacement_of(db, admission) is not None:
        raise NotAllowed(
            "This admission has been replaced by a corrected one; record the outcome there."
        )
    if chain_outcome(db, admission) is not None:
        raise NotAllowed("An outcome is already recorded for this admission.")
    outcome = DeliveryOutcome(
        admission_id=admission.id, mode=mode, delivered_at=delivered_at, recorded_by=user.id
    )
    db.add(outcome)
    db.flush()
    events.append(
        db,
        user,
        "outcome.recorded",
        ("admission", admission.id),
        {"code": admission.code, "mode": mode.value},
    )
    return outcome
