"""The CSP forbids inline code, so no rendered page may contain any (spec §5, §6)."""

import re

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from robby.config import Settings
from robby.models import Role
from tests.factories import make_facility, make_user
from tests.webutil import csrf_for, sign_in_client

INLINE = {
    "style attribute": re.compile(r"\sstyle\s*=", re.IGNORECASE),
    "style element": re.compile(r"<style\b", re.IGNORECASE),
    "inline script": re.compile(r"<script\b(?![^>]*\ssrc=)", re.IGNORECASE),
    "event handler attribute": re.compile(r"<[^>]+\son[a-z]+\s*=", re.IGNORECASE),
    "javascript: URL": re.compile(r"javascript:", re.IGNORECASE),
}


def assert_no_inline_code(html: str, page: str) -> None:
    for name, pattern in INLINE.items():
        assert not pattern.search(html), f"{name} on {page}"


@pytest.mark.parametrize("path", ["/login", "/session-ended", "/session-ended?reason=absolute", "/no-such-page"])
def test_public_pages(client: TestClient, path: str) -> None:
    assert_no_inline_code(client.get(path).text, path)


def test_system_administrator_pages(client: TestClient, db: Session, settings: Settings) -> None:
    row = sign_in_client(client, db, settings, make_user(db, role=Role.SYSTEM_ADMIN))
    facility = make_facility(db)
    make_user(db, role=Role.FACILITY_ADMIN, facility=facility)
    db.commit()
    for path in ("/admin/facilities?add=1", "/admin/users?add=1", "/account/password", "/admissions/new"):
        assert_no_inline_code(client.get(path).text, path)
    created = client.post(
        "/admin/users",
        data={"csrf_token": csrf_for(settings, row), "username": "mw.inline", "role": "clinician",
              "facility_id": str(facility.id)},
    )
    assert created.status_code == 200
    assert_no_inline_code(created.text, "password shown")
    invalid = client.post("/admin/facilities", data={"csrf_token": csrf_for(settings, row), "code": "x"})
    assert invalid.status_code == 400
    assert_no_inline_code(invalid.text, "facility form errors")


def test_facility_administrator_and_clinician_pages(
    client: TestClient, db: Session, settings: Settings
) -> None:
    sign_in_client(client, db, settings, make_user(db, role=Role.FACILITY_ADMIN))
    for path in ("/admin/users?add=1", "/audit", "/audit/unclassified"):
        assert_no_inline_code(client.get(path).text, path)
    client.cookies.clear()
    sign_in_client(client, db, settings, make_user(db, role=Role.CLINICIAN, must_change=True))
    assert_no_inline_code(client.get("/account/password").text, "forced password change")
