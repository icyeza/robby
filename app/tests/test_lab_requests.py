"""Lab requests: the clinician asks for tests, the lab technician records the results.

Covers the insert-only tables, the lifecycle in the service (duplicates, cancellation reasons,
closed once, facility scope), both roles' pages, the HIV re-score of the readiness signal (and
that nothing else re-scores), event payloads without identifiers or results, and the badge."""

from __future__ import annotations

import json
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from robson_serving.schema import FeatureSchema
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session

from robby.clinical import service as clinical
from robby.clinical.scoring import ModelRuntime
from robby.config import Settings, default_rule_set_path
from robby.errors import NotAllowed, NotFound
from robby.lab import service
from robby.models import (
    Admission,
    Classification,
    LabCancellation,
    LabRequest,
    LabResult,
    LabTest,
    ModelState,
    Prediction,
    Role,
    UserSession,
)
from robby.rulesets import load_rule_set_file
from robby.timeutil import utcnow
from tests.factories import events_of, make_facility, make_model_version, make_user
from tests.test_clinical_flow import post_form, saved_code
from tests.test_no_inline_code import assert_no_inline_code
from tests.webutil import csrf_for, sign_in_client

PROBABILITY = {"Yes": 0.8, "No": 0.3, None: 0.5}


class HivStubScorer:
    """A model that reads hiv_positive (as readiness-v1.0 does) and whose answer shows it."""

    schema = FeatureSchema.from_dict(
        {
            "format": 1,
            "columns": [],
            "inputs": [
                {
                    "name": "hiv_positive",
                    "kind": "category",
                    "range": None,
                    "levels": ["No", "Yes"],
                },
            ],
        }
    )
    fail_rescore = False

    def build_row(self, admission: dict[str, Any]) -> dict[str, Any]:
        parity = admission.get("parity")
        return {
            "parity": None if parity is None else float(parity),
            "hiv_positive": admission.get("hiv_positive"),
        }

    def model_frame(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return rows

    def predict(self, frame: list[dict[str, Any]]) -> list[float]:
        if self.fail_rescore and frame[0]["hiv_positive"] is not None:
            raise RuntimeError("model refused the row")
        return [PROBABILITY[frame[0]["hiv_positive"]]]


@pytest.fixture
def scorer() -> HivStubScorer:
    return HivStubScorer()


@pytest.fixture
def world(
    app: FastAPI, client: TestClient, db: Session, settings: Settings, scorer: HivStubScorer
) -> dict[str, Any]:
    load_rule_set_file(db, default_rule_set_path())
    facility = make_facility(db)
    lab = make_user(db, role=Role.LAB_TECHNICIAN, facility=facility)
    clinician = make_user(db, role=Role.CLINICIAN, facility=facility)
    make_model_version(db, state=ModelState.ACTIVE)
    row = sign_in_client(client, db, settings, lab)
    db.commit()
    runtime = ModelRuntime(Path("unused"), loader=lambda path, rule_set=None: scorer)
    app.state.model_runtime = runtime
    return {
        "facility": facility,
        "lab": lab,
        "clinician": clinician,
        "csrf": csrf_for(settings, row),
        "settings": settings,
        "runtime": runtime,
    }


def admit(client: TestClient, db: Session, world: dict[str, Any], **changes: str) -> Admission:
    """An admission saved through the lab technician's form (the client must be signed in as
    the lab technician)."""
    code = saved_code(post_form(client, world, **changes))
    admission = db.scalar(select(Admission).where(Admission.code == code))
    assert admission is not None
    return admission


def sign_in(client: TestClient, db: Session, world: dict[str, Any], user: Any) -> str:
    """Sign `user` in on the client (its session in world["session"]); returns the CSRF token."""
    client.cookies.clear()
    world["session"] = sign_in_client(client, db, world["settings"], user)
    return csrf_for(world["settings"], world["session"])


def ask(db: Session, world: dict[str, Any], admission: Admission, *tests: LabTest, **kw: Any):
    rows = service.request_tests(
        db,
        world["clinician"],
        admission,
        list(tests),
        note=kw.get("note"),
        urgent=kw.get("urgent", False),
        now=kw.get("now"),
    )
    db.commit()
    return rows


def answer(db: Session, world: dict[str, Any], request: LabRequest, value: str):
    recorded = service.record_result(db, world["runtime"], world["lab"], request.id, value)
    db.commit()
    return recorded


def predictions(db: Session, admission: Admission) -> list[Prediction]:
    return list(
        db.scalars(
            select(Prediction)
            .where(Prediction.admission_id == admission.id)
            .order_by(Prediction.created_at, Prediction.supersedes_id.nulls_first())
        )
    )


# ---------------------------------------------------------------- tables


@pytest.mark.parametrize("table", ["lab_request", "lab_result", "lab_cancellation"])
def test_lab_rows_cannot_be_updated_or_deleted(
    client: TestClient, db: Session, world: dict[str, Any], table: str
) -> None:
    admission = admit(client, db, world)
    first, second = ask(db, world, admission, LabTest.URINE_PROTEIN, LabTest.HIV_RAPID)
    answer(db, world, first, "trace")
    service.cancel_request(db, world["clinician"], second.id, "Delivered first")
    db.flush()
    for statement in (f"UPDATE {table} SET created_at = created_at", f"DELETE FROM {table}"):
        with pytest.raises(DBAPIError, match="is not allowed"), db.begin_nested():
            db.execute(text(statement))


def test_a_request_is_closed_once_even_by_raw_inserts(
    client: TestClient, db: Session, world: dict[str, Any]
) -> None:
    admission = admit(client, db, world)
    (request,) = ask(db, world, admission, LabTest.BLOOD_GLUCOSE)
    answer(db, world, request, "6.1")
    with pytest.raises(IntegrityError, match="already has a result"), db.begin_nested():
        db.add(
            LabCancellation(
                request_id=request.id,
                reason="late",
                cancelled_by=world["clinician"].id,
                cancelled_at=utcnow(),
            )
        )
        db.flush()


def test_values_notes_and_reasons_are_stored_encrypted(
    client: TestClient, db: Session, world: dict[str, Any]
) -> None:
    admission = admit(client, db, world)
    hiv, protein = ask(
        db, world, admission, LabTest.HIV_RAPID, LabTest.URINE_PROTEIN, note="Ward 3 bed 2"
    )
    answer(db, world, hiv, "positive")
    service.cancel_request(db, world["clinician"], protein.id, "Strips out of stock")
    db.flush()
    raw = [
        bytes(v)
        for (v,) in db.execute(
            text(
                "SELECT value FROM lab_result UNION ALL SELECT reason FROM lab_cancellation "
                "UNION ALL SELECT note FROM lab_request"
            )
        )
    ]
    assert len(raw) == 4
    for plain in (b"positive", b"Strips", b"Ward 3"):
        assert not any(plain in value for value in raw)


# ---------------------------------------------------------------- lifecycle


def test_request_result_and_the_derived_status(
    client: TestClient, db: Session, world: dict[str, Any]
) -> None:
    admission = admit(client, db, world)
    (request,) = ask(db, world, admission, LabTest.BLOOD_GLUCOSE, note="fasting")
    assert service.open_count(db, world["lab"]) == 1
    assert [r.id for r, _ in service.queue(db, world["lab"])] == [request.id]
    (line,) = service.lines_for(db, admission)
    assert line.status == "requested" and line.request.note == "fasting"
    recorded = answer(db, world, request, "6.1")
    assert recorded.rescore == "not_hiv" and recorded.prediction is None
    (line,) = service.lines_for(db, admission)
    assert line.status == "resulted" and line.result is not None and line.result.value == "6.1"
    assert service.open_count(db, world["lab"]) == 0 and service.queue(db, world["lab"]) == []


def test_an_open_test_cannot_be_requested_twice_but_again_after_closing(
    client: TestClient, db: Session, world: dict[str, Any]
) -> None:
    admission = admit(client, db, world)
    (first,) = ask(db, world, admission, LabTest.HIV_RAPID)
    with pytest.raises(NotAllowed, match="Already requested and waiting for the lab: HIV"):
        ask(db, world, admission, LabTest.URINE_PROTEIN, LabTest.HIV_RAPID)
    db.rollback()
    assert len(service.lines_for(db, admission)) == 1  # nothing of the refused request saved
    service.cancel_request(db, world["clinician"], first.id, "Done at the clinic last week")
    (again,) = ask(db, world, admission, LabTest.HIV_RAPID)
    answer(db, world, again, "negative")
    ask(db, world, admission, LabTest.HIV_RAPID)  # a repeat after a result is allowed too
    statuses = sorted(line.status for line in service.lines_for(db, admission))
    assert statuses == ["cancelled", "requested", "resulted"]


def test_cancelling_needs_a_reason_and_an_open_request(
    client: TestClient, db: Session, world: dict[str, Any]
) -> None:
    admission = admit(client, db, world)
    (request,) = ask(db, world, admission, LabTest.URINE_PROTEIN)
    with pytest.raises(NotAllowed, match="Say why"):
        service.cancel_request(db, world["clinician"], request.id, "   ")
    answer(db, world, request, "+")
    with pytest.raises(NotAllowed, match="already recorded"):
        service.cancel_request(db, world["clinician"], request.id, "Not needed")
    db.rollback()
    with pytest.raises(NotAllowed, match="already recorded"):
        answer(db, world, request, "++")


def test_another_facility_can_neither_request_nor_see_nor_answer(
    client: TestClient, db: Session, world: dict[str, Any]
) -> None:
    admission = admit(client, db, world)
    (request,) = ask(db, world, admission, LabTest.HIV_RAPID)
    other = make_facility(db)
    other_lab = make_user(db, role=Role.LAB_TECHNICIAN, facility=other)
    other_clinician = make_user(db, role=Role.CLINICIAN, facility=other)
    with pytest.raises(NotFound):
        service.request_tests(
            db, other_clinician, admission, [LabTest.URINE_PROTEIN], note=None, urgent=False
        )
    with pytest.raises(NotFound):
        service.get_line(db, other_lab, request.id)
    with pytest.raises(NotFound):
        service.record_result(db, world["runtime"], other_lab, request.id, "negative")
    with pytest.raises(NotFound):
        service.cancel_request(db, other_clinician, request.id, "not mine")
    assert service.open_count(db, other_lab) == 0 and service.queue(db, other_lab) == []

    sign_in(client, db, world, other_lab)
    assert client.get(f"/lab-requests/{request.id}").status_code == 404
    assert client.get("/lab-requests/count").text == ""
    assert admission.code not in client.get("/lab-requests").text
    sign_in(client, db, world, other_clinician)
    assert client.get(f"/admissions/{admission.code}/lab-requests/new").status_code == 404
    assert (
        client.get(f"/admissions/{admission.code}/lab-requests/{request.id}/cancel").status_code
        == 404
    )


# ---------------------------------------------------------------- pages


def test_the_clinician_requests_from_the_patient_page(
    client: TestClient, db: Session, world: dict[str, Any]
) -> None:
    admission = admit(client, db, world, glucose="", glucose_nm="1", glucose_reason="declined")
    csrf = sign_in(client, db, world, world["clinician"])
    page = client.get(f"/admissions/{admission.code}")
    assert "No lab tests requested" in page.text
    assert f'href="/admissions/{admission.code}/lab-requests/new"' in page.text
    form = client.get(f"/admissions/{admission.code}/lab-requests/new")
    assert form.status_code == 200 and "Blood pressure is measured at the bedside" in form.text
    assert_no_inline_code(form.text, "request form")

    empty = client.post(f"/admissions/{admission.code}/lab-requests", data={"csrf_token": csrf})
    assert empty.status_code == 400 and "Choose at least one test." in empty.text
    long = client.post(
        f"/admissions/{admission.code}/lab-requests",
        data={"csrf_token": csrf, "test": "hiv_rapid", "note": "x" * 201},
    )
    assert long.status_code == 400 and "200 characters or fewer" in long.text
    sent = client.post(
        f"/admissions/{admission.code}/lab-requests",
        data={
            "csrf_token": csrf,
            "test": ["blood_glucose", "hiv_rapid"],
            "note": "before theatre",
            "urgent": "1",
        },
    )
    assert sent.status_code == 303 and sent.headers["location"] == f"/admissions/{admission.code}"
    page = client.get(f"/admissions/{admission.code}").text
    assert page.count("Waiting for the lab") == 2 and "Urgent" in page
    assert_no_inline_code(page, "patient page with lab tests")
    form = client.get(f"/admissions/{admission.code}/lab-requests/new").text
    assert form.count("already requested, waiting for the lab") == 2
    twice = client.post(
        f"/admissions/{admission.code}/lab-requests",
        data={"csrf_token": csrf, "test": "hiv_rapid"},
    )
    assert twice.status_code == 400 and "Already requested and waiting for the lab" in twice.text

    glucose = next(r for r in db.scalars(select(LabRequest)) if r.test is LabTest.BLOOD_GLUCOSE)
    cancel_path = f"/admissions/{admission.code}/lab-requests/{glucose.id}/cancel"
    assert cancel_path in page
    assert_no_inline_code(client.get(cancel_path).text, "cancel form")
    blank = client.post(cancel_path, data={"csrf_token": csrf, "reason": " "})
    assert blank.status_code == 400 and "Say why the request is cancelled." in blank.text
    done = client.post(cancel_path, data={"csrf_token": csrf, "reason": "Delivered first"})
    assert done.status_code == 303
    page = client.get(f"/admissions/{admission.code}").text
    assert "Cancelled" in page and "Delivered first" in page


def test_the_lab_technician_sees_the_queue_and_records_results(
    client: TestClient, db: Session, world: dict[str, Any]
) -> None:
    older = admit(client, db, world, patient_number="LQ-1", patient_name="Older Patient")
    newer = admit(client, db, world, patient_number="LQ-2", patient_name="Urgent Patient")
    now = utcnow()
    (protein,) = ask(db, world, older, LabTest.URINE_PROTEIN, now=now - timedelta(hours=2))
    (glucose,) = ask(db, world, older, LabTest.BLOOD_GLUCOSE, now=now - timedelta(hours=1))
    (hiv,) = ask(db, world, newer, LabTest.HIV_RAPID, urgent=True, note="ward 4", now=now)
    csrf = sign_in(client, db, world, world["lab"])
    page = client.get("/lab-requests")
    assert page.status_code == 200
    assert_no_inline_code(page.text, "lab queue")
    body = page.text.split("<tbody>")[1]
    order = [body.index(str(r.id)) for r in (hiv, protein, glucose)]
    assert order == sorted(order)  # urgent first, then the oldest
    for shown in ("Older Patient", "LQ-1", "Urgent Patient", newer.code, "ward 4", "Urgent"):
        assert shown in body

    form = client.get(f"/lab-requests/{glucose.id}")
    assert form.status_code == 200 and "Result (mmol/L)" in form.text
    assert_no_inline_code(form.text, "result form")
    for bad, message in (
        ("", "Enter the glucose result in mmol/L."),
        ("abc", "Glucose must be between 0.5 and 60 mmol/L"),
        ("61", "Glucose must be between 0.5 and 60 mmol/L"),
        ("5.25", "with at most one decimal"),
    ):
        refused = client.post(
            f"/lab-requests/{glucose.id}", data={"csrf_token": csrf, "value": bad}
        )
        assert refused.status_code == 400 and message in refused.text, bad
    missing = client.post(f"/lab-requests/{protein.id}", data={"csrf_token": csrf, "value": ""})
    assert missing.status_code == 400 and "Choose the urine protein result." in missing.text
    wrong = client.post(f"/lab-requests/{hiv.id}", data={"csrf_token": csrf, "value": "maybe"})
    assert wrong.status_code == 400 and "Choose the HIV test result." in wrong.text

    saved = client.post(f"/lab-requests/{glucose.id}", data={"csrf_token": csrf, "value": "7.4"})
    assert (
        saved.status_code == 303
        and saved.headers["location"] == f"/lab-requests?saved={glucose.id}"
    )
    note = client.get(saved.headers["location"]).text
    assert "Result saved" in note and "Blood glucose" in note and older.code in note
    assert str(glucose.id) not in note.split("<tbody>")[1]
    # a closed request is not answered twice: the form goes back to the list
    assert client.get(f"/lab-requests/{glucose.id}").status_code == 303
    again = client.post(f"/lab-requests/{glucose.id}", data={"csrf_token": csrf, "value": "8"})
    assert again.status_code == 400 and "already recorded" in again.text
    assert db.scalar(select(LabResult).where(LabResult.request_id == glucose.id)).value == "7.4"


def test_screening_completed_later_is_shown_next_to_not_measured(
    client: TestClient, db: Session, world: dict[str, Any]
) -> None:
    admission = admit(
        client,
        db,
        world,
        proteinuria="",
        protein_nm="1",
        protein_reason="supplies_out_of_stock",
        glucose="",
        glucose_nm="1",
        glucose_reason="equipment_unavailable",
    )
    group = db.scalar(select(Classification.robson_group))
    before = predictions(db, admission)
    protein, glucose = ask(db, world, admission, LabTest.URINE_PROTEIN, LabTest.BLOOD_GLUCOSE)
    assert answer(db, world, protein, "++").rescore == "not_hiv"
    assert answer(db, world, glucose, "8.2").rescore == "not_hiv"
    assert [p.id for p in predictions(db, admission)] == [p.id for p in before]  # no re-score
    assert [c.robson_group for c in db.scalars(select(Classification))] == [group]
    sign_in(client, db, world, world["clinician"])
    page = client.get(f"/admissions/{admission.code}").text
    assert "Not measured · reagent strips out of stock" in page
    assert "Screening completed later: ++" in page
    assert "Screening completed later: 8.2 mmol/L" in page
    assert "After the HIV test result" not in page


# ---------------------------------------------------------------- re-scoring


@pytest.mark.parametrize(("value", "level"), [("positive", "Yes"), ("negative", "No")])
def test_an_hiv_result_rescores_and_supersedes_the_signal(
    client: TestClient, db: Session, world: dict[str, Any], value: str, level: str
) -> None:
    admission = admit(client, db, world)
    (original,) = predictions(db, admission)
    assert original.feature_vector["hiv_positive"] is None and float(original.probability) == 0.5
    (request,) = ask(db, world, admission, LabTest.HIV_RAPID)
    recorded = answer(db, world, request, value)
    assert recorded.rescore == "ok" and recorded.prediction is not None
    old, new = predictions(db, admission)
    assert old.id == original.id and old.probability == original.probability  # kept as it was
    assert new.supersedes_id == original.id and new.lab_result_id == recorded.result.id
    assert new.model_id == original.model_id
    assert new.feature_vector == {**original.feature_vector, "hiv_positive": level}
    assert float(new.probability) == PROBABILITY[level]
    current = clinical.prediction_of(db, admission)
    assert current is not None and current[0].id == new.id
    assert clinical.summaries_of(db, [admission])[admission.id][1] == round(
        PROBABILITY[level] * 100
    )

    sign_in(client, db, world, world["clinician"])
    page = client.get(f"/admissions/{admission.code}").text
    assert f"about {round(PROBABILITY[level] * 100)} had a cesarean" in page
    assert "After the HIV test result recorded" in page


def test_an_indeterminate_hiv_result_is_recorded_without_a_rescore(
    client: TestClient, db: Session, world: dict[str, Any]
) -> None:
    admission = admit(client, db, world)
    (request,) = ask(db, world, admission, LabTest.HIV_RAPID)
    recorded = answer(db, world, request, "indeterminate")
    assert recorded.rescore == "indeterminate" and len(predictions(db, admission)) == 1
    (line,) = service.lines_for(db, admission)
    assert line.status == "resulted"


def test_a_rescore_that_fails_still_records_the_result(
    client: TestClient, db: Session, world: dict[str, Any], scorer: HivStubScorer
) -> None:
    admission = admit(client, db, world)
    scorer.fail_rescore = True
    (request,) = ask(db, world, admission, LabTest.HIV_RAPID)
    csrf = sign_in(client, db, world, world["lab"])
    saved = client.post(
        f"/lab-requests/{request.id}", data={"csrf_token": csrf, "value": "positive"}
    )
    assert saved.status_code == 303
    assert db.scalar(select(LabResult).where(LabResult.request_id == request.id)) is not None
    assert len(predictions(db, admission)) == 1
    assert [e.payload["test"] for e in events_of(db, "lab.resulted")] == ["hiv_rapid"]


def test_no_rescore_without_a_prediction_to_update(
    client: TestClient, db: Session, world: dict[str, Any]
) -> None:
    partial = admit(client, db, world, gestational_age_weeks="")  # no group, so never scored
    (request,) = ask(db, world, partial, LabTest.HIV_RAPID)
    assert answer(db, world, request, "positive").rescore == "no_prediction"
    assert predictions(db, partial) == []


# ---------------------------------------------------------------- privacy and events


def test_events_carry_codes_and_test_names_only(
    client: TestClient, db: Session, world: dict[str, Any]
) -> None:
    admission = admit(client, db, world, patient_number="EV-7", patient_name="Eventa Private")
    hiv, glucose = ask(
        db, world, admission, LabTest.HIV_RAPID, LabTest.BLOOD_GLUCOSE, note="secret note"
    )
    answer(db, world, hiv, "positive")
    service.cancel_request(db, world["clinician"], glucose.id, "secret reason")
    db.flush()
    (requested,) = events_of(db, "lab.requested")
    assert requested.payload == {
        "code": admission.code,
        "facility": world["facility"].code,
        "tests": ["hiv_rapid", "blood_glucose"],
        "urgent": False,
    }
    (resulted,) = events_of(db, "lab.resulted")
    assert resulted.payload == {
        "code": admission.code,
        "facility": world["facility"].code,
        "test": "hiv_rapid",
    }
    (cancelled,) = events_of(db, "lab.cancelled")
    assert cancelled.payload["test"] == "blood_glucose"
    for event in (requested, resulted, cancelled):
        dumped = json.dumps(event.payload).lower()
        for secret in ("ev-7", "eventa", "positive", "secret", "yes"):
            assert secret not in dumped, (event.event_type, secret)


# ---------------------------------------------------------------- badge


def test_the_badge_counts_open_requests_without_keeping_the_session_alive(
    client: TestClient, db: Session, world: dict[str, Any]
) -> None:
    first = admit(client, db, world)
    second = admit(client, db, world)
    ask(db, world, first, LabTest.HIV_RAPID, LabTest.URINE_PROTEIN)
    (closed,) = ask(db, world, second, LabTest.BLOOD_GLUCOSE)
    answer(db, world, closed, "5.0")
    sign_in(client, db, world, world["lab"])
    page = client.get("/admissions/new").text
    assert 'hx-get="/lab-requests/count" hx-trigger="load, every 30s"' in page
    assert client.get("/lab-requests/count").text == "2"

    session: UserSession = world["session"]
    earlier = utcnow() - timedelta(minutes=10)
    session.last_seen_at = earlier
    db.commit()
    assert client.get("/lab-requests/count").status_code == 200
    db.refresh(session)
    assert session.last_seen_at == earlier  # polling does not move the idle timer
    client.get("/lab-requests")
    db.refresh(session)
    assert session.last_seen_at > earlier  # opening a page does


def test_the_badge_is_empty_with_nothing_open(
    client: TestClient, db: Session, world: dict[str, Any]
) -> None:
    sign_in(client, db, world, world["lab"])
    assert client.get("/lab-requests/count").text == ""
    assert "No lab requests are waiting." in client.get("/lab-requests").text


# ---------------------------------------------------------------- access

SOME_ID = uuid.UUID("00000000-0000-4000-8000-000000000001")
GET_PAGES: list[tuple[str, Role]] = [
    ("/admissions/ZZ-0001/lab-requests/new", Role.CLINICIAN),
    (f"/admissions/ZZ-0001/lab-requests/{SOME_ID}/cancel", Role.CLINICIAN),
    (f"/lab-requests/{SOME_ID}", Role.LAB_TECHNICIAN),
]


@pytest.mark.parametrize("role", list(Role))
@pytest.mark.parametrize(("path", "allowed"), GET_PAGES, ids=[p for p, _ in GET_PAGES])
def test_request_pages_by_role(
    client: TestClient, db: Session, settings: Settings, role: Role, path: str, allowed: Role
) -> None:
    sign_in_client(client, db, settings, make_user(db, role=role))
    # the allowed role reaches the page and finds nothing at this address
    assert client.get(path).status_code == (404 if role is allowed else 403)


def test_a_bad_request_id_is_not_found(
    client: TestClient, db: Session, world: dict[str, Any]
) -> None:
    assert client.get("/lab-requests/not-a-uuid").status_code == 404
