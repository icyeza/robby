"""Facilities screen: list, add (optionally with a first administrator), deactivate."""

import re
import uuid

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from robby.auth.passwords import verify_password
from robby.auth.service import can_sign_in
from robby.config import Settings
from robby.models import AppUser, Facility, Role
from tests.factories import events_of, make_facility, make_user
from tests.webutil import csrf_for, sign_in_client

SECRET = re.compile(r'<span class="secret">([a-z0-9-]+)</span>')


def as_system_admin(client: TestClient, db: Session, settings: Settings) -> str:
    row = sign_in_client(client, db, settings, make_user(db, role=Role.SYSTEM_ADMIN))
    return csrf_for(settings, row)


def test_list_shows_each_facility_with_its_administrators(
    client: TestClient, db: Session, settings: Settings
) -> None:
    as_system_admin(client, db, settings)
    facility = make_facility(db, name="Facility A", code="FA", district="District 1")
    make_user(db, role=Role.FACILITY_ADMIN, facility=facility, username="fa.a02")
    make_user(db, role=Role.CLINICIAN, facility=facility)
    db.commit()
    response = client.get("/admin/facilities")
    assert response.status_code == 200
    assert "<title>Facilities · Robson Readiness</title>" in response.text
    assert "<b>Facility A</b>" in response.text and "<td>FA</td>" in response.text
    assert "<td>fa.a02</td>" in response.text
    assert '<td class="r num">2</td>' in response.text
    assert '<span class="st done">Active</span>' in response.text
    assert '<div class="dialog-backdrop" id="add-facility">' in response.text


def test_the_add_link_opens_the_dialog_without_javascript(
    client: TestClient, db: Session, settings: Settings
) -> None:
    as_system_admin(client, db, settings)
    response = client.get("/admin/facilities?add=1")
    assert '<div class="dialog-backdrop open" id="add-facility">' in response.text


def test_adding_a_facility(client: TestClient, db: Session, settings: Settings) -> None:
    token = as_system_admin(client, db, settings)
    response = client.post(
        "/admin/facilities",
        data={"csrf_token": token, "name": "Facility K", "code": "fk", "district": "District 9"},
    )
    assert response.status_code == 303 and response.headers["location"] == "/admin/facilities"
    facility = db.scalar(select(Facility).where(Facility.code == "FK"))
    assert facility is not None and facility.name == "Facility K"
    assert len(events_of(db, "facility.created")) == 1


def test_adding_a_facility_with_its_first_administrator(
    client: TestClient, db: Session, settings: Settings
) -> None:
    token = as_system_admin(client, db, settings)
    response = client.post(
        "/admin/facilities",
        data={"csrf_token": token, "name": "Facility K", "code": "FK", "district": "District 9",
              "admin_username": "fa.k01", "model_facility_key": "Site A"},
    )
    assert response.status_code == 200
    assert "Temporary password for fa.k01" in response.text
    assert response.headers["cache-control"] == "no-store"
    match = SECRET.search(response.text)
    assert match is not None
    admin = db.scalar(select(AppUser).where(AppUser.username == "fa.k01"))
    assert admin is not None and admin.role is Role.FACILITY_ADMIN
    assert admin.facility is not None and admin.facility.code == "FK"
    assert admin.facility.model_facility_key == "Site A"
    assert admin.must_change_password is True
    assert verify_password(admin.password_hash, match.group(1))


def test_invalid_input_reopens_the_dialog_with_messages(
    client: TestClient, db: Session, settings: Settings
) -> None:
    token = as_system_admin(client, db, settings)
    response = client.post(
        "/admin/facilities", data={"csrf_token": token, "name": "", "code": "f1", "district": ""}
    )
    assert response.status_code == 400
    assert '<div class="dialog-backdrop open" id="add-facility">' in response.text
    assert "Enter the facility name." in response.text
    assert "Use 2 to 6 capital letters, for example FA." in response.text
    assert "Enter the district." in response.text
    assert 'value="f1"' in response.text


def test_a_taken_administrator_username_adds_nothing(
    client: TestClient, db: Session, settings: Settings
) -> None:
    token = as_system_admin(client, db, settings)
    make_user(db, role=Role.SYSTEM_ADMIN, username="taken.name")
    db.commit()
    response = client.post(
        "/admin/facilities",
        data={"csrf_token": token, "name": "Facility T", "code": "FT", "district": "D",
              "admin_username": "taken.name"},
    )
    assert response.status_code == 400
    assert "This username is already taken." in response.text
    assert db.scalar(select(Facility).where(Facility.code == "FT")) is None


def test_deactivating_a_facility_stops_its_accounts(
    client: TestClient, db: Session, settings: Settings
) -> None:
    token = as_system_admin(client, db, settings)
    facility = make_facility(db)
    clinician = make_user(db, role=Role.CLINICIAN, facility=facility)
    db.commit()
    response = client.post(f"/admin/facilities/{facility.id}/deactivate", data={"csrf_token": token})
    assert response.status_code == 303
    db.refresh(facility)
    db.refresh(clinician)
    assert facility.active is False
    assert can_sign_in(clinician) is False
    page = client.get("/admin/facilities")
    assert '<span class="st retired">Deactivated</span>' in page.text


def test_unknown_facility_is_not_found(client: TestClient, db: Session, settings: Settings) -> None:
    token = as_system_admin(client, db, settings)
    for facility_id in (str(uuid.uuid4()), "not-a-uuid"):
        response = client.post(f"/admin/facilities/{facility_id}/deactivate", data={"csrf_token": token})
        assert response.status_code == 404


def test_actions_need_the_csrf_token(client: TestClient, db: Session, settings: Settings) -> None:
    as_system_admin(client, db, settings)
    response = client.post("/admin/facilities", data={"name": "Facility X", "code": "FX", "district": "D"})
    assert response.status_code == 403
    assert db.scalar(select(Facility).where(Facility.code == "FX")) is None
