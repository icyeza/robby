"""/privacy and /terms are public; a signed-in user must accept the current versions first; the
acceptance is an insert-only row and a logged event."""

import re

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from robby import events
from robby.config import Settings
from robby.models import PolicyAcceptance, Role
from robby.policy import service
from tests.factories import events_of, make_user
from tests.webutil import csrf_for, sign_in_client


def test_the_pages_are_public_and_show_their_version_and_date(client: TestClient) -> None:
    privacy = client.get("/privacy")
    terms = client.get("/terms")
    assert privacy.status_code == 200 and terms.status_code == 200
    assert "<title>Privacy notice · Robson Readiness</title>" in privacy.text
    assert "<title>Terms of use · Robson Readiness</title>" in terms.text
    assert f"Version {service.PRIVACY_VERSION} · 20 November 2026" in privacy.text
    assert f"Version {service.TERMS_VERSION} · 20 November 2026" in terms.text
    assert "<b>Erasure:</b> not supported" in privacy.text
    assert "30 minutes without use and after 12 hours" in terms.text


def test_the_sign_in_page_and_every_policy_page_link_to_both_documents(
    client: TestClient,
) -> None:
    for path in ("/login", "/privacy", "/terms"):
        page = client.get(path).text
        assert 'href="/privacy"' in page and 'href="/terms"' in page, path
        assert 'class="site-foot"' in page, path


def test_the_pages_have_no_inline_code(client: TestClient) -> None:
    for path in ("/privacy", "/terms"):
        page = client.get(path).text
        assert "<script" not in re.sub(r"<script[^>]* src=[^>]*></script>", "", page)
        assert " style=" not in page and " onclick=" not in page


def test_the_lab_badge_poll_does_not_loop_through_the_acceptance_step(
    client: TestClient, db: Session, settings: Settings
) -> None:
    user = make_user(db, role=Role.LAB_TECHNICIAN, accepted=False)
    sign_in_client(client, db, settings, user)
    poll = client.get("/lab-requests/count", headers={"HX-Request": "true"})
    assert poll.status_code == 200 and poll.headers["HX-Redirect"] == "/accept-policies"
    assert "/lab-requests/count" not in client.get("/accept-policies").text


def test_a_user_who_has_not_accepted_is_sent_to_the_acceptance_step(
    client: TestClient, db: Session, settings: Settings
) -> None:
    user = make_user(db, role=Role.SYSTEM_ADMIN, accepted=False, must_change=True)
    sign_in_client(client, db, settings, user)
    for path in ("/", "/admin/facilities", "/admin/users"):
        response = client.get(path)
        assert response.status_code == 303 and response.headers["location"] == "/accept-policies"
    step = client.get("/accept-policies")
    assert step.status_code == 200 and "I accept" in step.text
    assert "hx-get" not in step.text and "<nav><a" not in step.text  # no polling badge
    assert client.get("/privacy").status_code == 200  # the public pages stay open


def test_accepting_records_an_immutable_row_and_an_event(
    client: TestClient, db: Session, settings: Settings
) -> None:
    user = make_user(db, role=Role.SYSTEM_ADMIN, accepted=False)
    row = sign_in_client(client, db, settings, user)
    assert client.post("/accept-policies", data={}).status_code == 403
    assert db.scalar(select(PolicyAcceptance)) is None
    token = {"csrf_token": csrf_for(settings, row)}
    response = client.post("/accept-policies", data=token)
    assert response.status_code == 303 and response.headers["location"] == "/"
    accepted = db.scalars(select(PolicyAcceptance)).all()
    assert len(accepted) == 1 and accepted[0].user_id == user.id
    assert (accepted[0].privacy_version, accepted[0].terms_version) == (
        service.PRIVACY_VERSION,
        service.TERMS_VERSION,
    )
    logged = events_of(db, "policy.accepted")
    assert len(logged) == 1 and logged[0].actor_id == user.id
    assert logged[0].payload == {
        "privacy_version": service.PRIVACY_VERSION,
        "terms_version": service.TERMS_VERSION,
    }
    assert events.verify(db).ok
    assert client.get("/").headers["location"] == "/admin/facilities"
    assert client.get("/accept-policies").headers["location"] == "/"
    client.post("/accept-policies", data=token)
    assert len(db.scalars(select(PolicyAcceptance)).all()) == 1  # not added twice


def test_the_acceptance_comes_before_the_forced_password_change(
    client: TestClient, db: Session, settings: Settings
) -> None:
    user = make_user(db, role=Role.CLINICIAN, accepted=False, must_change=True)
    row = sign_in_client(client, db, settings, user)
    assert client.get("/patients").headers["location"] == "/accept-policies"
    client.post("/accept-policies", data={"csrf_token": csrf_for(settings, row)})
    assert client.get("/patients").headers["location"] == "/account/password"


def test_a_new_privacy_version_asks_again(
    client: TestClient, db: Session, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    sign_in_client(client, db, settings, make_user(db, role=Role.SYSTEM_ADMIN))
    assert client.get("/admin/facilities").status_code == 200
    monkeypatch.setattr(service, "PRIVACY_VERSION", "9.9")
    assert client.get("/admin/facilities").headers["location"] == "/accept-policies"


def test_a_new_terms_version_asks_again(
    client: TestClient, db: Session, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    sign_in_client(client, db, settings, make_user(db, role=Role.SYSTEM_ADMIN))
    assert client.get("/admin/facilities").status_code == 200
    monkeypatch.setattr(service, "TERMS_VERSION", "9.9")
    assert client.get("/admin/facilities").headers["location"] == "/accept-policies"


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE policy_acceptance SET terms_version = 'x'",
        "DELETE FROM policy_acceptance",
        "TRUNCATE policy_acceptance",
    ],
)
def test_acceptances_cannot_be_changed_or_deleted(db: Session, statement: str) -> None:
    make_user(db, role=Role.SYSTEM_ADMIN)
    with pytest.raises(DBAPIError, match="not allowed"), db.begin_nested():
        db.execute(text(statement))
