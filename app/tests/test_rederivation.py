"""Stored outputs can be re-derived from their stored inputs (requirement NFR4).

Admissions are saved through the real form with a real packaged model (built from synthetic data,
as in test_serving_contract). Every stored classification is then recomputed from the admission's
stored Robson fields under the rule set it names, and every stored prediction from its stored
feature vector under the model it names; both must reproduce the stored result exactly. That
includes a prediction re-scored after an HIV result (robby.lab), whose feature vector must also
be the one the form would have produced had the HIV status been entered at admission.
"""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import robson_serving
from fastapi import FastAPI
from fastapi.testclient import TestClient
from robson_engine.ruleset import parse_rule_set
from robson_serving.artefact import write_artefact
from robson_serving.contract import write_golden
from robson_serving.monitoring import reference_distributions
from sqlalchemy import select
from sqlalchemy.orm import Session

from robby.clinical.classify import classify_inputs, robson_inputs
from robby.clinical.scoring import ModelRuntime
from robby.config import Settings, default_rule_set_path
from robby.lab import service as lab
from robby.models import (
    Admission,
    AppUser,
    Classification,
    LabTest,
    ModelState,
    ModelVersion,
    Prediction,
    Role,
    RuleSet,
)
from robby.rulesets import load_rule_set_file
from tests.factories import make_facility, make_model_version, make_user
from tests.test_clinical_flow import post_form, saved_code
from tests.test_serving_contract import _synthetic_helpers
from tests.webutil import csrf_for, sign_in_client

# Admissions covering resolved groups, a partial and a conflict.
VARIANTS: list[dict[str, str]] = [
    {},
    {"parity": "0", "previous_cs_count": "0"},
    {"parity": "0", "previous_cs_count": "0", "onset_of_labour": "induced"},
    {"fetal_presentation": "breech"},
    {"plurality": "2"},
    {"gestational_age_weeks": "33"},
    {"gestational_age_weeks": ""},  # partial: the field that decides is missing
    {"parity": "0"},  # conflict: nulliparous with a previous cesarean
]
PLACES = Decimal("0.00001")


@pytest.fixture
def packaged(tmp_path: Path) -> Path:
    helpers = _synthetic_helpers()
    model, x = helpers.fit_model(helpers.make_records(600, seed=3))
    schema = helpers.make_schema(model)
    out = tmp_path / "synthetic-v0"
    fields = helpers.manifest_fields(schema)
    write_artefact(out, model, schema, fields, reference_distributions(x, helpers.KINDS))
    write_golden(robson_serving.load(out), out)
    return out


@pytest.fixture
def packaged_with_hiv(tmp_path: Path) -> Path:
    """The synthetic model with hiv_positive among its inputs, as readiness-v1.0 has it."""
    helpers = _synthetic_helpers()  # a fresh module: patching it touches no other test
    helpers.KINDS = {**helpers.KINDS, "hiv_positive": helpers.CATEGORICAL}
    helpers.INPUT_SPECS = {**helpers.INPUT_SPECS, "hiv_positive": (helpers.CATEGORY, None)}
    records = helpers.make_records(600, seed=3)
    pattern = ["Yes", "No", "No", None, "No"]
    records["hiv_positive"] = pd.Series(
        [pattern[i % len(pattern)] for i in range(len(records))], dtype=object
    )
    model, x = helpers.fit_model(records)
    schema = helpers.make_schema(model)
    out = tmp_path / "synthetic-hiv-v0"
    fields = helpers.manifest_fields(schema, "synthetic-hiv-v0")
    write_artefact(out, model, schema, fields, reference_distributions(x, helpers.KINDS))
    write_golden(robson_serving.load(out), out)
    return out


def _admit(client: TestClient, world: dict[str, Any], db: Session, **changes: str) -> Admission:
    response = post_form(client, world, **changes)
    assert response.status_code == 303, response.text
    admission = db.scalar(select(Admission).where(Admission.code == saved_code(response)))
    assert admission is not None
    return admission


def test_a_prediction_rescored_after_an_hiv_result_is_re_derived(
    app: FastAPI,
    client: TestClient,
    db: Session,
    settings: Settings,
    packaged_with_hiv: Path,
) -> None:
    load_rule_set_file(db, default_rule_set_path())
    facility = make_facility(db)
    technician: AppUser = make_user(db, role=Role.LAB_TECHNICIAN, facility=facility)
    clinician = make_user(db, role=Role.CLINICIAN, facility=facility)
    row = sign_in_client(client, db, settings, technician)
    model = make_model_version(db, state=ModelState.ACTIVE)
    model.artefact_path = str(packaged_with_hiv)
    db.commit()
    runtime = ModelRuntime(packaged_with_hiv.parent)
    app.state.model_runtime = runtime
    world: dict[str, Any] = {"csrf": csrf_for(settings, row), "settings": settings}

    told = _admit(client, world, db, patient_number="HIV-1", d_hiv_positive="Yes")
    later = _admit(client, world, db, patient_number="HIV-2")
    (request,) = lab.request_tests(
        db, clinician, later, [LabTest.HIV_RAPID], note=None, urgent=False
    )
    recorded = lab.record_result(db, runtime, technician, request.id, "positive")
    db.commit()
    assert recorded.rescore == "ok" and recorded.prediction is not None

    by_admission = {
        p.admission_id: p
        for p in db.scalars(select(Prediction).where(Prediction.supersedes_id.is_(None)))
    }
    entered, rescored = by_admission[told.id], recorded.prediction
    assert rescored.supersedes_id == by_admission[later.id].id
    assert rescored.feature_vector["hiv_positive"] == "Yes"
    assert rescored.feature_vector == entered.feature_vector  # as the form would have scored it
    assert rescored.probability == entered.probability

    scorer = robson_serving.load(packaged_with_hiv)
    for stored in db.scalars(select(Prediction)):
        probability = scorer.predict(scorer.model_frame([stored.feature_vector]))[0]
        again = Decimal(str(probability)).quantize(PLACES, rounding=ROUND_HALF_EVEN)
        assert again == stored.probability


@pytest.fixture
def saved(
    app: FastAPI, client: TestClient, db: Session, settings: Settings, packaged: Path
) -> list[str]:
    load_rule_set_file(db, default_rule_set_path())
    facility = make_facility(db)
    user = make_user(db, role=Role.LAB_TECHNICIAN, facility=facility)
    row = sign_in_client(client, db, settings, user)
    model = make_model_version(db, state=ModelState.ACTIVE)
    model.artefact_path = str(packaged)
    db.commit()
    app.state.model_runtime = ModelRuntime(packaged.parent)
    world: dict[str, Any] = {"csrf": csrf_for(settings, row), "settings": settings}
    codes = []
    for n, changes in enumerate(VARIANTS):
        response = post_form(client, world, patient_number=f"RD-{n}", **changes)
        assert response.status_code == 303, response.text
        codes.append(saved_code(response))
    return codes


def test_every_classification_is_re_derived_under_its_pinned_rule_set(
    db: Session, saved: list[str]
) -> None:
    statuses = set()
    for code in saved:
        admission = db.scalar(select(Admission).where(Admission.code == code))
        assert admission is not None
        stored = db.scalar(
            select(Classification).where(Classification.admission_id == admission.id)
        )
        assert stored is not None
        rules = db.get(RuleSet, stored.rule_set_id)
        assert rules is not None
        ga = admission.gestational_age_weeks
        inputs = robson_inputs(
            admission.parity,
            admission.previous_cs_count,
            admission.plurality,
            admission.fetal_presentation,
            None if ga is None else int(ga),
            admission.onset_of_labour,
        )
        result, trace = classify_inputs(inputs, parse_rule_set(rules.definition))
        assert str(result.status) == stored.status.value
        assert result.group == stored.robson_group and result.subgroup == stored.subgroup
        assert sorted(result.candidates) == sorted(stored.candidates)
        assert trace == stored.trace
        statuses.add(stored.status.value)
    assert statuses == {"resolved", "partial", "conflict"}


def test_every_prediction_is_re_derived_under_its_pinned_model(
    db: Session, saved: list[str]
) -> None:
    predictions = db.scalars(select(Prediction)).all()
    assert predictions, "the resolved admissions were scored"
    for stored in predictions:
        model = db.get(ModelVersion, stored.model_id)
        assert model is not None
        scorer = robson_serving.load(Path(model.artefact_path))
        probability = scorer.predict(scorer.model_frame([stored.feature_vector]))[0]
        again = Decimal(str(probability)).quantize(PLACES, rounding=ROUND_HALF_EVEN)
        assert again == stored.probability
