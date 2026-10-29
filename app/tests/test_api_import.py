"""POST /api/v1/admissions/import: bearer-token authentication and the three body formats."""

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
from robby.models import Admission, AdmissionSource, ApiToken, ModelState, Role
from robby.rulesets import load_rule_set_file
from robby.timeutil import utcnow
from tests.factories import events_of, make_facility, make_model_version, make_user
from tests.test_clinical_flow import StubScorer
from tests.test_import import HEADER, XLSX, good_row, make_xlsx
from tests.test_openclinic import HEADER as OC_HEADER
from tests.test_openclinic import oc_row

URL = "/api/v1/admissions/import"


@pytest.fixture
def feed(app: FastAPI, client: TestClient, db: Session, settings: Settings) -> dict[str, Any]:
    load_rule_set_file(db, default_rule_set_path())
    facility = make_facility(db)
    user = make_user(db, role=Role.LAB_TECHNICIAN, facility=facility)
    make_model_version(db, state=ModelState.ACTIVE)
    token, _ = api_tokens.create(db, None, user.username, "OpenClinic feed")
    db.commit()
    app.state.model_runtime = ModelRuntime(
        Path("unused"), loader=lambda path, rule_set=None: StubScorer()
    )
    recent = (datetime.now(settings.zone) - timedelta(hours=2)).strftime("%d/%m/%Y %H:%M")
    return {"auth": {"Authorization": f"Bearer {token}"}, "user": user, "when": recent,
            "facility": facility, "token": token}  # fmt: skip


def count(db: Session) -> int:
    return db.scalar(select(func.count()).select_from(Admission)) or 0


def records(feed: dict[str, Any], *numbers: str) -> list[dict[str, Any]]:
    return [
        dict(
            zip(OC_HEADER, oc_row(**{"Patient ID": n, "Admission date": feed["when"]}), strict=True)
        )
        for n in numbers
    ]


def test_json_rows_are_imported_for_the_tokens_facility(
    client: TestClient, feed: dict[str, Any], db: Session
) -> None:
    response = client.post(URL, json={"rows": records(feed, "A-1", "A-2")}, headers=feed["auth"])
    assert response.status_code == 201
    body = response.json()
    assert body["imported"] == 2 and body["skipped_duplicates"] == []
    assert [a["row"] for a in body["admissions"]] == [1, 2]
    saved = list(db.scalars(select(Admission).order_by(Admission.code)))
    assert [(a.patient_number, a.source, a.facility_id) for a in saved] == [
        ("A-1", AdmissionSource.IMPORT, feed["facility"].id),
        ("A-2", AdmissionSource.IMPORT, feed["facility"].id),
    ]
    assert all(a.entered_by == feed["user"].id for a in saved)
    event = events_of(db, "admissions.imported")[0]
    assert event.payload["channel"] == "api" and event.payload["rows"] == 2
    assert "A-1" not in json.dumps(event.payload)


def test_a_bare_list_and_an_xlsx_body_and_multipart_also_work(
    client: TestClient, feed: dict[str, Any], db: Session
) -> None:
    assert client.post(URL, json=records(feed, "B-1"), headers=feed["auth"]).status_code == 201
    iso = (datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M")
    raw = make_xlsx(HEADER, [good_row("B-2", iso)])
    sent = client.post(URL, content=raw, headers={**feed["auth"], "Content-Type": XLSX})
    assert sent.status_code == 201
    form = client.post(
        URL,
        files={"file": ("x.xlsx", make_xlsx(HEADER, [good_row("B-3", iso)]), XLSX)},
        headers=feed["auth"],
    )
    assert form.status_code == 201
    assert count(db) == 3


def test_resending_a_batch_skips_what_is_already_recorded(
    client: TestClient, feed: dict[str, Any], db: Session
) -> None:
    client.post(URL, json=records(feed, "C-1", "C-2"), headers=feed["auth"])
    again = client.post(URL, json=records(feed, "C-1", "C-2", "C-3"), headers=feed["auth"])
    assert again.status_code == 201
    body = again.json()
    assert body["imported"] == 1 and body["admissions"][0]["row"] == 3
    assert [r["row"] for r in body["already_recorded"]] == [1, 2]
    assert all(r["code"] and r["robson_group"] for r in body["already_recorded"])
    assert count(db) == 3
    nothing = client.post(URL, json=records(feed, "C-1"), headers=feed["auth"])
    assert nothing.status_code == 200 and nothing.json()["imported"] == 0


def test_one_bad_record_saves_nothing_and_names_the_record(
    client: TestClient, feed: dict[str, Any], db: Session
) -> None:
    rows = records(feed, "D-1", "D-2")
    rows[1]["Presentation"] = "sideways"
    response = client.post(URL, json={"rows": rows}, headers=feed["auth"])
    assert response.status_code == 422
    problems = response.json()["problems"]
    assert [(p["row"], p["column"]) for p in problems] == [(2, "Presentation")]
    assert count(db) == 0


@pytest.mark.parametrize("header", [None, "", "Bearer", "Bearer rbk_nope", "Basic abc", "Bearer x"])
def test_without_a_valid_token_nothing_happens(
    client: TestClient, feed: dict[str, Any], db: Session, header: str | None
) -> None:
    headers = {} if header is None else {"Authorization": header}
    response = client.post(URL, json=records(feed, "E-1"), headers=headers)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert count(db) == 0


def test_a_revoked_token_or_deactivated_account_is_refused(
    client: TestClient, feed: dict[str, Any], db: Session
) -> None:
    row = db.scalar(select(ApiToken))
    assert row is not None
    api_tokens.revoke(db, None, row.id, utcnow())
    db.commit()
    assert client.post(URL, json=records(feed, "F-1"), headers=feed["auth"]).status_code == 401
    assert len(events_of(db, "api_token.revoked")) == 1
    token, _ = api_tokens.create(db, None, feed["user"].username, "second")
    feed["user"].active = False
    db.commit()
    headers = {"Authorization": f"Bearer {token}"}
    assert client.post(URL, json=records(feed, "F-1"), headers=headers).status_code == 401


def test_bad_bodies(client: TestClient, feed: dict[str, Any]) -> None:
    assert (
        client.post(
            URL, content=b"{", headers={**feed["auth"], "Content-Type": "application/json"}
        ).status_code
        == 400
    )
    assert client.post(URL, json={"rows": []}, headers=feed["auth"]).status_code == 400
    assert client.post(URL, json={"rows": [1, 2]}, headers=feed["auth"]).status_code == 400
    assert (
        client.post(
            URL, content=b"a,b", headers={**feed["auth"], "Content-Type": "text/csv"}
        ).status_code
        == 415
    )


def test_only_lab_technician_accounts_can_hold_tokens(db: Session) -> None:
    facility = make_facility(db)
    for role in (Role.FACILITY_ADMIN, Role.CLINICIAN):
        other = make_user(db, role=role, facility=facility)
        with pytest.raises(api_tokens.TokenError):
            api_tokens.create(db, None, other.username, "x")
    with pytest.raises(api_tokens.TokenError):
        api_tokens.create(db, None, "nobody", "x")


def test_the_token_is_stored_hashed_and_creation_is_logged(db: Session) -> None:
    facility = make_facility(db)
    user = make_user(db, role=Role.LAB_TECHNICIAN, facility=facility)
    token, row = api_tokens.create(db, None, user.username, "feed")
    db.commit()
    assert token.startswith("rbk_") and token not in row.token_hash and len(row.token_hash) == 64
    event = events_of(db, "api_token.created")[0]
    assert token not in json.dumps(event.payload)
