"""Integrations screen: a facility admin manages their own facility's feed tokens and sees what the
feed did."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from robby.auth import api_tokens
from robby.clinical.scoring import ModelRuntime
from robby.config import Settings, default_rule_set_path
from robby.models import ApiToken, ModelState, Role
from robby.rulesets import load_rule_set_file
from tests.factories import make_facility, make_model_version, make_user
from tests.test_api_import import URL, records
from tests.test_clinical_flow import StubScorer
from tests.webutil import csrf_for, sign_in_client

TOKEN = re.compile(r'<span class="secret">(rbk_[A-Za-z0-9_-]+)</span>')


@pytest.fixture
def admin(app: FastAPI, client: TestClient, db: Session, settings: Settings) -> dict[str, Any]:
    load_rule_set_file(db, default_rule_set_path())
    facility = make_facility(db)
    boss = make_user(db, role=Role.FACILITY_ADMIN, facility=facility)
    feeder = make_user(db, role=Role.CLINICIAN, facility=facility, username="openclinic-feed")
    make_model_version(db, state=ModelState.ACTIVE)
    row = sign_in_client(client, db, settings, boss)
    db.commit()
    app.state.model_runtime = ModelRuntime(
        Path("unused"), loader=lambda path, rule_set=None: StubScorer()
    )
    when = (datetime.now(settings.zone) - timedelta(hours=2)).strftime("%d/%m/%Y %H:%M")
    return {"csrf": csrf_for(settings, row), "facility": facility, "feeder": feeder, "when": when}


def test_admin_creates_a_token_shown_once_and_the_feed_can_use_it(
    client: TestClient, admin: dict[str, Any], db: Session
) -> None:
    page = client.get("/admin/integrations")
    assert "No feed set up yet." in page.text and "openclinic-feed" in page.text
    made = client.post(
        "/admin/integrations/tokens",
        data={"csrf_token": admin["csrf"], "label": "OpenClinic nightly",
              "clinician_id": str(admin["feeder"].id)},
    )  # fmt: skip
    assert made.status_code == 200
    token = TOKEN.search(made.text)
    assert token is not None
    assert token.group(1) not in client.get("/admin/integrations").text
    sent = client.post(
        URL,
        json={"rows": records(admin, "G-1")},
        headers={"Authorization": f"Bearer {token.group(1)}"},
    )
    assert sent.status_code == 201
    page = client.get("/admin/integrations")
    assert "OpenClinic nightly" in page.text and "Never" not in page.text
    assert "Automatic feed" in page.text and "1 admissions" in page.text


def test_revoking_stops_the_feed(client: TestClient, admin: dict[str, Any], db: Session) -> None:
    token, row = api_tokens.create(db, None, admin["feeder"].username, "feed")
    db.commit()
    done = client.post(
        f"/admin/integrations/tokens/{row.id}/revoke",
        data={"csrf_token": admin["csrf"]},
        follow_redirects=False,
    )
    assert done.status_code == 303
    headers = {"Authorization": f"Bearer {token}"}
    assert (
        client.post(URL, json={"rows": records(admin, "H-1")}, headers=headers).status_code == 401
    )
    assert "Revoked" in client.get("/admin/integrations").text


def test_rejected_and_repeated_batches_show_in_the_history(
    client: TestClient, admin: dict[str, Any], db: Session
) -> None:
    token, _ = api_tokens.create(db, None, admin["feeder"].username, "feed")
    db.commit()
    headers = {"Authorization": f"Bearer {token}"}
    bad = records(admin, "I-1")
    bad[0]["Presentation"] = "sideways"
    assert client.post(URL, json={"rows": bad}, headers=headers).status_code == 422
    good = {"rows": records(admin, "I-2")}
    client.post(URL, json=good, headers=headers)
    assert client.post(URL, json=good, headers=headers).status_code == 200
    text = client.get("/admin/integrations").text
    assert "Rejected" in text and "1 problems, nothing saved" in text
    assert "Nothing new" in text and "1 already recorded" in text


def test_another_facilitys_tokens_are_invisible_and_untouchable(
    client: TestClient, admin: dict[str, Any], db: Session
) -> None:
    other = make_facility(db, name="Other Hospital", code="OH")
    stranger = make_user(db, role=Role.CLINICIAN, facility=other)
    _, theirs = api_tokens.create(db, None, stranger.username, "theirs")
    db.commit()
    assert "theirs" not in client.get("/admin/integrations").text
    revoke = client.post(
        f"/admin/integrations/tokens/{theirs.id}/revoke", data={"csrf_token": admin["csrf"]}
    )
    assert revoke.status_code == 404
    forged = client.post(
        "/admin/integrations/tokens",
        data={"csrf_token": admin["csrf"], "label": "x", "clinician_id": str(stranger.id)},
    )
    assert forged.status_code == 400
    assert db.scalars(select(ApiToken).where(ApiToken.revoked_at.is_not(None))).first() is None


def test_creating_needs_a_csrf_token_and_a_label(client: TestClient, admin: dict[str, Any]) -> None:
    form = {"label": "x", "clinician_id": str(admin["feeder"].id)}
    assert client.post("/admin/integrations/tokens", data=form).status_code == 403
    blank = client.post(
        "/admin/integrations/tokens",
        data={"csrf_token": admin["csrf"], "label": " ", "clinician_id": form["clinician_id"]},
    )
    assert blank.status_code == 400 and "label" in blank.text.lower()
