"""Admission requests: the clinician asks the lab to enter a new patient, the lab technician
enters the admission from the request.

Covers the insert-only tables and the encrypted, indexed patient number and name, the lifecycle
in the service (one waiting request per number, cancellation reasons, closed once, facility
scope), both roles' pages (the Find patient list, the one lab queue, the prefilled admission
form), the fulfilment saved in the admission's own transaction, and event payloads without
identifiers."""

from __future__ import annotations

import json
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session

from robby.clinical.scoring import ModelRuntime
from robby.config import Settings, default_rule_set_path
from robby.errors import NotAllowed, NotFound
from robby.lab import admission_requests as asks
from robby.models import (
    Admission,
    AdmissionRequest,
    AdmissionRequestCancellation,
    AdmissionRequestFulfilment,
    LabTest,
    ModelState,
    Role,
    patient_number_index,
)
from robby.rulesets import load_rule_set_file
from robby.timeutil import utcnow
from tests.factories import events_of, make_facility, make_model_version, make_user
from tests.test_clinical_flow import GOOD, now_text, post_form
from tests.test_lab_requests import HivStubScorer, admit, sign_in
from tests.test_lab_requests import ask as ask_tests
from tests.test_no_inline_code import assert_no_inline_code
from tests.webutil import csrf_for, sign_in_client


@pytest.fixture
def world(app: FastAPI, client: TestClient, db: Session, settings: Settings) -> dict[str, Any]:
    load_rule_set_file(db, default_rule_set_path())
    facility = make_facility(db)
    lab = make_user(db, role=Role.LAB_TECHNICIAN, facility=facility)
    clinician = make_user(db, role=Role.CLINICIAN, facility=facility)
    make_model_version(db, state=ModelState.ACTIVE)
    row = sign_in_client(client, db, settings, lab)
    db.commit()
    runtime = ModelRuntime(Path("unused"), loader=lambda path, rule_set=None: HivStubScorer())
    app.state.model_runtime = runtime
    return {
        "facility": facility,
        "lab": lab,
        "clinician": clinician,
        "csrf": csrf_for(settings, row),
        "settings": settings,
        "runtime": runtime,
    }


def ask(db: Session, world: dict[str, Any], number: str, **kw: Any) -> AdmissionRequest:
    row = asks.request_admission(
        db,
        kw.get("user", world["clinician"]),
        patient_number=number,
        patient_name=kw.get("name"),
        note=kw.get("note"),
        urgent=kw.get("urgent", False),
        now=kw.get("now"),
    )
    db.commit()
    return row


def enter(
    client: TestClient, db: Session, world: dict[str, Any], request: AdmissionRequest, **kw: str
) -> Any:
    """The lab technician saves the admission form with the request's hidden field (the client
    must be signed in as the lab technician; world["csrf"] is theirs)."""
    return post_form(
        client,
        world,
        patient_number=request.patient_number,
        patient_name=request.patient_name or "Named At The Lab",
        admission_request=str(request.id),
        **kw,
    )


def admissions(db: Session) -> int:
    return int(db.scalar(select(func.count()).select_from(Admission)) or 0)


# ---------------------------------------------------------------- tables


@pytest.mark.parametrize(
    "table",
    ["admission_request", "admission_request_fulfilment", "admission_request_cancellation"],
)
def test_admission_request_rows_cannot_be_updated_or_deleted(
    client: TestClient, db: Session, world: dict[str, Any], table: str
) -> None:
    entered, withdrawn = ask(db, world, "AR-1"), ask(db, world, "AR-2")
    assert enter(client, db, world, entered).status_code == 303
    asks.cancel_request(db, world["clinician"], withdrawn.id, "Went home")
    db.flush()
    for statement in (f"UPDATE {table} SET created_at = created_at", f"DELETE FROM {table}"):
        with pytest.raises(DBAPIError, match="is not allowed"), db.begin_nested():
            db.execute(text(statement))


def test_a_request_is_closed_once_even_by_raw_inserts(
    client: TestClient, db: Session, world: dict[str, Any]
) -> None:
    request = ask(db, world, "AR-3")
    enter(client, db, world, request)
    with pytest.raises(IntegrityError, match="already has an admission"), db.begin_nested():
        db.add(
            AdmissionRequestCancellation(
                request_id=request.id,
                reason="late",
                cancelled_by=world["clinician"].id,
                cancelled_at=utcnow(),
            )
        )
        db.flush()


def test_number_name_note_and_reason_are_stored_encrypted_with_a_blind_index(
    client: TestClient, db: Session, world: dict[str, Any]
) -> None:
    request = ask(db, world, "SECRET-77", name="Hidden Demo-Name", note="Ward 3 bed 2")
    asks.cancel_request(db, world["clinician"], request.id, "Moved to Kigali")
    db.flush()
    raw = db.execute(
        text(
            "SELECT patient_number, patient_name, note, patient_number_index FROM admission_request"
        )
    ).one()
    (reason,) = db.execute(text("SELECT reason FROM admission_request_cancellation")).one()
    stored = [bytes(v) for v in (*raw[:3], reason)]
    for plain in (b"SECRET", b"Hidden", b"Ward 3", b"Kigali"):
        assert not any(plain in value for value in stored)
    facility = world["facility"]
    assert raw[3] == patient_number_index(facility.id, "secret-77")  # same index as admission
    assert raw[3] != patient_number_index(make_facility(db).id, "SECRET-77")
    db.expire_all()
    again = db.get(AdmissionRequest, request.id)
    assert again is not None and again.patient_name == "Hidden Demo-Name"


# ---------------------------------------------------------------- lifecycle


def test_request_fulfil_and_the_derived_status(
    client: TestClient, db: Session, world: dict[str, Any]
) -> None:
    request = ask(db, world, "AR-10", name="Keza Demo-Abera", note="from the clinic")
    assert asks.open_count(db, world["lab"]) == 1
    assert [r.id for r in asks.queue(db, world["lab"])] == [request.id]
    (line,) = asks.for_clinician(db, world["clinician"])
    assert line.status == "waiting" and line.code is None
    saved = enter(client, db, world, request)
    code = saved.headers["location"].rsplit("=", 1)[1]
    assert saved.headers["location"] == f"/lab-requests?entered={code}"
    (line,) = asks.for_clinician(db, world["clinician"])
    assert line.status == "entered" and line.code == code
    assert line.fulfilment is not None and line.fulfilment.fulfilled_by == world["lab"].id
    assert asks.open_count(db, world["lab"]) == 0 and asks.queue(db, world["lab"]) == []


def test_a_waiting_number_cannot_be_asked_twice_but_again_after_closing(
    db: Session, world: dict[str, Any]
) -> None:
    first = ask(db, world, "AR-20")
    with pytest.raises(NotAllowed, match="already waiting for the lab"):
        ask(db, world, " ar-20 ")  # the same number in another case and spacing
    db.rollback()
    assert asks.open_count(db, world["lab"]) == 1
    asks.cancel_request(db, world["clinician"], first.id, "Wrong number")
    ask(db, world, "AR-20")
    statuses = sorted(line.status for line in asks.for_clinician(db, world["clinician"]))
    assert statuses == ["cancelled", "waiting"]


def test_cancelling_needs_a_reason_and_a_waiting_request(
    client: TestClient, db: Session, world: dict[str, Any]
) -> None:
    request = ask(db, world, "AR-30")
    with pytest.raises(NotAllowed, match="Say why"):
        asks.cancel_request(db, world["clinician"], request.id, "   ")
    enter(client, db, world, request)
    with pytest.raises(NotAllowed, match="already entered this patient"):
        asks.cancel_request(db, world["clinician"], request.id, "Not needed")
    db.rollback()
    withdrawn = ask(db, world, "AR-31")
    asks.cancel_request(db, world["clinician"], withdrawn.id, "Delivered at home")
    other = db.scalars(select(Admission)).first()
    assert other is not None
    with pytest.raises(NotAllowed, match="cancelled by the clinician"):
        asks.fulfil(db, world["lab"], withdrawn.id, other)


def test_the_clinician_list_keeps_closed_requests_for_72_hours(
    db: Session, world: dict[str, Any]
) -> None:
    old = ask(db, world, "AR-40", now=utcnow() - timedelta(days=5))
    asks.cancel_request(db, world["clinician"], old.id, "Old", now=utcnow() - timedelta(hours=73))
    recent = ask(db, world, "AR-41", now=utcnow() - timedelta(days=5))
    asks.cancel_request(db, world["clinician"], recent.id, "Recent")
    waiting = ask(db, world, "AR-42", now=utcnow() - timedelta(days=9))
    shown = [line.request.id for line in asks.for_clinician(db, world["clinician"])]
    assert set(shown) == {recent.id, waiting.id}


def test_another_facility_can_neither_see_nor_cancel_nor_enter(
    client: TestClient, db: Session, world: dict[str, Any]
) -> None:
    request = ask(db, world, "AR-50", name="Teta Demo-Uwera")
    other = make_facility(db)
    other_lab = make_user(db, role=Role.LAB_TECHNICIAN, facility=other)
    other_clinician = make_user(db, role=Role.CLINICIAN, facility=other)
    db.commit()
    with pytest.raises(NotFound):
        asks.get_line(db, other_lab, request.id)
    with pytest.raises(NotFound):
        asks.cancel_request(db, other_clinician, request.id, "not mine")
    admission = admit(client, db, world)
    with pytest.raises(NotFound):
        asks.fulfil(db, other_lab, request.id, admission)
    db.rollback()
    assert asks.open_count(db, other_lab) == 0 and asks.queue(db, other_lab) == []
    assert asks.for_clinician(db, other_clinician) == []
    # the same number at another facility is another patient: it may be asked there
    ask(db, world, "AR-50", user=other_clinician)

    csrf = sign_in(client, db, world, other_lab)
    assert client.get(f"/admissions/new?request={request.id}").status_code == 404
    assert "Teta Demo-Uwera" not in client.get("/lab-requests").text
    assert client.get("/lab-requests/count").text == "1"  # their own request only
    before = admissions(db)
    forged = client.post(
        "/admissions",
        data={
            **GOOD,
            "admitted_at": now_text(world["settings"]),
            "csrf_token": csrf,
            "admission_request": str(request.id),
        },
    )
    assert forged.status_code == 400 and "The request was not found." in forged.text
    assert admissions(db) == before
    sign_in(client, db, world, other_clinician)
    assert "Teta Demo-Uwera" not in client.get("/patients").text
    assert client.get(f"/patients/admission-requests/{request.id}/cancel").status_code == 404


# ---------------------------------------------------------------- pages


def test_the_clinician_asks_from_find_patient_and_cancels(
    client: TestClient, db: Session, world: dict[str, Any]
) -> None:
    csrf = sign_in(client, db, world, world["clinician"])
    page = client.get("/patients")
    assert 'href="/patients/admission-requests/new"' in page.text
    assert "Ask the lab to enter a new patient" in page.text
    assert_no_inline_code(page.text, "find patient with admission requests")
    form = client.get("/patients/admission-requests/new")
    assert form.status_code == 200 and "Patient number" in form.text
    assert_no_inline_code(form.text, "admission request form")

    empty = client.post("/patients/admission-requests", data={"csrf_token": csrf})
    assert empty.status_code == 400 and "Enter the patient number." in empty.text
    bad = client.post(
        "/patients/admission-requests",
        data={"csrf_token": csrf, "patient_number": "AR 60!", "note": "x" * 201},
    )
    assert bad.status_code == 400 and "Patient number must be" in bad.text
    assert "200 characters or fewer" in bad.text
    sent = client.post(
        "/patients/admission-requests",
        data={
            "csrf_token": csrf,
            "patient_number": "AR-60",
            "patient_name": "  Ineza   Demo-Kamana ",
            "note": "in labour, ward 2",
            "urgent": "1",
        },
    )
    assert sent.status_code == 303 and sent.headers["location"] == "/patients"
    page = client.get("/patients").text
    for shown in ("Ineza Demo-Kamana", "AR-60", "Waiting for the lab", "Urgent", "ward 2"):
        assert shown in page
    twice = client.post(
        "/patients/admission-requests", data={"csrf_token": csrf, "patient_number": "ar-60"}
    )
    assert twice.status_code == 400 and "already waiting for the lab" in twice.text

    (request,) = db.scalars(select(AdmissionRequest))
    cancel_path = f"/patients/admission-requests/{request.id}/cancel"
    assert cancel_path in page
    assert_no_inline_code(client.get(cancel_path).text, "admission request cancel form")
    blank = client.post(cancel_path, data={"csrf_token": csrf, "reason": " "})
    assert blank.status_code == 400 and "Say why the request is cancelled." in blank.text
    done = client.post(cancel_path, data={"csrf_token": csrf, "reason": "Went to another hospital"})
    assert done.status_code == 303
    page = client.get("/patients").text
    assert "Cancelled" in page and "Went to another hospital" in page
    assert cancel_path not in page
    assert client.get(cancel_path).status_code == 303  # closed: back to the list


def test_one_queue_for_both_kinds_urgent_first_then_the_oldest(
    client: TestClient, db: Session, world: dict[str, Any]
) -> None:
    admission = admit(client, db, world, patient_number="LQ-9", patient_name="Tested Patient")
    now = utcnow()
    (protein,) = ask_tests(
        db, world, admission, LabTest.URINE_PROTEIN, now=now - timedelta(hours=3)
    )
    older = ask(db, world, "AR-70", name="Older Demo-Asked", now=now - timedelta(hours=4))
    newer = ask(db, world, "AR-71", now=now - timedelta(hours=1))
    urgent = ask(db, world, "AR-72", name="Urgent Demo-Asked", urgent=True, now=now)
    (hiv,) = ask_tests(
        db, world, admission, LabTest.HIV_RAPID, urgent=True, now=now - timedelta(minutes=5)
    )
    sign_in(client, db, world, world["lab"])
    page = client.get("/lab-requests")
    assert page.status_code == 200
    assert_no_inline_code(page.text, "lab queue with admission requests")
    body = page.text.split("<tbody>")[1]
    order = [body.index(str(r.id)) for r in (hiv, urgent, older, protein, newer)]
    assert order == sorted(order)
    assert body.count("New patient") == 3 and body.count("Enter new admission") == 3
    assert f'href="/admissions/new?request={older.id}"' in body
    assert f'href="/lab-requests/{protein.id}"' in body
    for shown in ("Older Demo-Asked", "AR-70", "AR-71", "Urgent Demo-Asked"):
        assert shown in body
    assert client.get("/lab-requests/count").text == "5"


def test_enter_admission_opens_the_form_prefilled_with_the_request(
    client: TestClient, db: Session, world: dict[str, Any]
) -> None:
    request = ask(db, world, "AR-80", name="Shema Demo-Gatete", note="came by ambulance")
    page = client.get(f"/admissions/new?request={request.id}")
    assert page.status_code == 200
    assert_no_inline_code(page.text, "admission form from a request")
    assert 'name="patient_number" type="text" value="AR-80"' in page.text
    assert 'name="patient_name" type="text" value="Shema Demo-Gatete"' in page.text
    assert f'<input type="hidden" name="admission_request" value="{request.id}">' in page.text
    assert "The clinician asked for this admission" in page.text
    assert "came by ambulance" in page.text
    # without a name, the lab technician types it
    unnamed = ask(db, world, "AR-81")
    page = client.get(f"/admissions/new?request={unnamed.id}").text
    assert 'name="patient_name" type="text" value=""' in page
    # a closed request goes back to the list; an unknown one is not found
    asks.cancel_request(db, world["clinician"], unnamed.id, "Found her")
    db.commit()
    gone = client.get(f"/admissions/new?request={unnamed.id}")
    assert gone.status_code == 303 and gone.headers["location"] == "/lab-requests"
    assert client.get(f"/admissions/new?request={uuid.uuid4()}").status_code == 404
    assert client.get("/admissions/new?request=not-a-uuid").status_code == 400
    # the plain form is unchanged
    assert "admission_request" not in client.get("/admissions/new").text


def test_saving_the_admission_fulfils_the_request_in_the_same_transaction(
    client: TestClient, db: Session, world: dict[str, Any]
) -> None:
    request = ask(db, world, "AR-90", name="Bwiza Demo-Nshuti")
    saved = enter(client, db, world, request)
    assert saved.status_code == 303
    code = saved.headers["location"].rsplit("=", 1)[1]
    admission = db.scalar(select(Admission).where(Admission.code == code))
    assert admission is not None and admission.patient_number == "AR-90"
    fulfilment = db.scalar(select(AdmissionRequestFulfilment))
    assert fulfilment is not None
    assert (fulfilment.request_id, fulfilment.admission_id) == (request.id, admission.id)
    queue = client.get(saved.headers["location"]).text
    assert "Admission saved" in queue and code in queue and "Bwiza Demo-Nshuti" in queue
    assert str(request.id) not in queue.split("<tbody>")[1]
    # the clinician sees it entered, with the way to the admission
    sign_in(client, db, world, world["clinician"])
    page = client.get("/patients").text
    assert "Entered by the lab" in page and f'href="/admissions/{code}"' in page


def test_a_request_closed_while_the_form_was_open_saves_nothing(
    client: TestClient, db: Session, world: dict[str, Any]
) -> None:
    request = ask(db, world, "AR-95", name="Ganza Demo-Rugero")
    assert "admission_request" in client.get(f"/admissions/new?request={request.id}").text
    asks.cancel_request(db, world["clinician"], request.id, "Delivered at home")
    db.commit()
    before = admissions(db)
    refused = enter(client, db, world, request)
    assert refused.status_code == 400
    assert "This request was cancelled by the clinician." in refused.text
    assert admissions(db) == before  # the admission went with the failed fulfilment
    assert db.scalar(select(func.count()).select_from(AdmissionRequestFulfilment)) == 0
    assert "AR-95" in refused.text and 'name="admission_request"' not in refused.text
    # saving again enters the admission on its own
    again = post_form(client, world, patient_number="AR-95", patient_name="Ganza Demo-Rugero")
    assert again.status_code == 303 and admissions(db) == before + 1


def test_an_admission_entered_without_the_request_leaves_it_waiting(
    client: TestClient, db: Session, world: dict[str, Any]
) -> None:
    request = ask(db, world, "AR-99")
    admit(client, db, world, patient_number="AR-99")
    (line,) = asks.for_clinician(db, world["clinician"])
    assert line.request.id == request.id and line.status == "waiting"


# ---------------------------------------------------------------- privacy and events


def test_events_carry_no_identifiers(
    client: TestClient, db: Session, world: dict[str, Any]
) -> None:
    entered = ask(db, world, "EV-71", name="Eventa Private", note="secret note", urgent=True)
    withdrawn = ask(db, world, "EV-72", name="Eventb Private")
    saved = enter(client, db, world, entered)
    code = saved.headers["location"].rsplit("=", 1)[1]
    asks.cancel_request(db, world["clinician"], withdrawn.id, "secret reason")
    db.flush()
    created = events_of(db, "admission_request.created")
    assert [e.payload for e in created] == [
        {"facility": world["facility"].code, "urgent": True},
        {"facility": world["facility"].code, "urgent": False},
    ]
    assert [e.entity_id for e in created] == [entered.id, withdrawn.id]
    (fulfilled,) = events_of(db, "admission_request.fulfilled")
    assert fulfilled.payload == {"code": code, "facility": world["facility"].code}
    (cancelled,) = events_of(db, "admission_request.cancelled")
    assert cancelled.payload == {"facility": world["facility"].code}
    for event in (*created, fulfilled, cancelled):
        dumped = json.dumps(event.payload).lower()
        for secret in ("ev-7", "eventa", "eventb", "private", "secret"):
            assert secret not in dumped, (event.event_type, secret)


# ---------------------------------------------------------------- access

SOME_ID = uuid.UUID("00000000-0000-4000-8000-000000000001")
GET_PAGES: list[tuple[str, Role, int]] = [
    (f"/patients/admission-requests/{SOME_ID}/cancel", Role.CLINICIAN, 404),
    (f"/admissions/new?request={SOME_ID}", Role.LAB_TECHNICIAN, 404),
]


@pytest.mark.parametrize("role", list(Role))
@pytest.mark.parametrize(("path", "allowed", "status"), GET_PAGES, ids=[p for p, _, _ in GET_PAGES])
def test_request_pages_by_role(
    client: TestClient,
    db: Session,
    settings: Settings,
    role: Role,
    path: str,
    allowed: Role,
    status: int,
) -> None:
    sign_in_client(client, db, settings, make_user(db, role=role))
    # the allowed role reaches the page and finds nothing at this address
    assert client.get(path).status_code == (status if role is allowed else 403)
