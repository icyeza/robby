"""Users screen: scoped listing, adding with a temporary password, reset, deactivate, reactivate."""

import re
from datetime import timedelta

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from robby.auth.passwords import verify_password
from robby.config import Settings
from robby.models import AppUser, Facility, Role
from robby.timeutil import utcnow
from tests.factories import DEFAULT_PASSWORD, make_facility, make_user
from tests.webutil import csrf_for, log_in, sign_in_client

SECRET = re.compile(r'<span class="secret">([a-z0-9-]+)</span>')


def signed_in(
    client: TestClient,
    db: Session,
    settings: Settings,
    role: Role,
    facility: Facility | None = None,
) -> tuple[AppUser, str]:
    user = make_user(db, role=role, facility=facility)
    row = sign_in_client(client, db, settings, user)
    return user, csrf_for(settings, row)


def test_facility_admin_sees_only_their_facility(
    client: TestClient, db: Session, settings: Settings
) -> None:
    admin, _ = signed_in(client, db, settings, Role.FACILITY_ADMIN)
    make_user(db, role=Role.CLINICIAN, facility=admin.facility, username="mw.a014")
    make_user(db, role=Role.CLINICIAN, username="mw.b001")
    db.commit()
    response = client.get("/admin/users")
    assert response.status_code == 200
    assert "<b>mw.a014</b>" in response.text and "mw.b001" not in response.text
    assert admin.facility is not None
    assert (
        f"People who can sign in at {admin.facility.name}, and what each can do." in response.text
    )
    assert "<th>Facility</th>" not in response.text
    assert f'<div class="fixed">{admin.facility.name}</div>' in response.text
    assert '<option value="clinician"' in response.text
    assert '<option value="model_custodian"' not in response.text


def test_system_admin_sees_everyone_and_their_facility(
    client: TestClient, db: Session, settings: Settings
) -> None:
    signed_in(client, db, settings, Role.SYSTEM_ADMIN)
    facility = make_facility(db, name="Facility Z")
    make_user(db, role=Role.CLINICIAN, facility=facility, username="mw.z001")
    db.commit()
    response = client.get("/admin/users")
    assert "<b>mw.z001</b>" in response.text and "<td>Facility Z</td>" in response.text
    assert "<th>Facility</th>" in response.text
    assert '<option value="model_custodian"' in response.text
    assert '<select id="f-facility_id" name="facility_id"' in response.text


def test_facility_admin_adds_a_clinician_to_their_own_facility(
    client: TestClient, db: Session, settings: Settings
) -> None:
    admin, token = signed_in(client, db, settings, Role.FACILITY_ADMIN)
    elsewhere = make_facility(db)
    db.commit()
    response = client.post(
        "/admin/users",
        data={
            "csrf_token": token,
            "display_name": "",
            "username": "MW.New01",
            "role": "clinician",
            "facility_id": str(elsewhere.id),
        },
    )
    assert response.status_code == 200
    assert "Temporary password for mw.new01" in response.text
    assert response.headers["cache-control"] == "no-store"
    match = SECRET.search(response.text)
    assert match is not None
    user = db.scalar(select(AppUser).where(AppUser.username == "mw.new01"))
    assert user is not None and user.facility_id == admin.facility_id
    assert user.must_change_password is True
    assert verify_password(user.password_hash, match.group(1))


def test_a_new_account_chooses_its_password_at_first_sign_in(
    client: TestClient, db: Session, settings: Settings
) -> None:
    _, token = signed_in(client, db, settings, Role.SYSTEM_ADMIN)
    facility = make_facility(db)
    db.commit()
    response = client.post(
        "/admin/users",
        data={
            "csrf_token": token,
            "username": "mw.first",
            "role": "clinician",
            "facility_id": str(facility.id),
        },
    )
    match = SECRET.search(response.text)
    assert match is not None
    client.cookies.clear()
    assert log_in(client, "mw.first", match.group(1)).status_code == 303
    assert client.get("/").headers["location"] == "/account/password"


def test_facility_admin_cannot_create_other_roles(
    client: TestClient, db: Session, settings: Settings
) -> None:
    _, token = signed_in(client, db, settings, Role.FACILITY_ADMIN)
    response = client.post(
        "/admin/users", data={"csrf_token": token, "username": "mc.nope", "role": "model_custodian"}
    )
    assert response.status_code == 400
    assert '<div class="dialog-backdrop open" id="add-user">' in response.text
    assert "You cannot create accounts with this role." in response.text
    assert db.scalar(select(AppUser).where(AppUser.username == "mc.nope")) is None


def test_a_clinician_needs_a_facility(client: TestClient, db: Session, settings: Settings) -> None:
    _, token = signed_in(client, db, settings, Role.SYSTEM_ADMIN)
    response = client.post(
        "/admin/users",
        data={"csrf_token": token, "username": "mw.nofac", "role": "clinician", "facility_id": ""},
    )
    assert response.status_code == 400
    assert "Choose the active facility this person works at." in response.text


def test_reset_password_shows_a_new_temporary_password(
    client: TestClient, db: Session, settings: Settings
) -> None:
    admin, token = signed_in(client, db, settings, Role.FACILITY_ADMIN)
    target = make_user(db, role=Role.CLINICIAN, facility=admin.facility)
    db.commit()
    response = client.post(f"/admin/users/{target.id}/reset-password", data={"csrf_token": token})
    assert response.status_code == 200 and "Password reset" in response.text
    match = SECRET.search(response.text)
    assert match is not None
    db.refresh(target)
    assert verify_password(target.password_hash, match.group(1))
    assert target.must_change_password is True


def test_facility_admin_cannot_touch_accounts_elsewhere(
    client: TestClient, db: Session, settings: Settings
) -> None:
    _, token = signed_in(client, db, settings, Role.FACILITY_ADMIN)
    stranger = make_user(db, role=Role.CLINICIAN)
    db.commit()
    for action in ("reset-password", "deactivate", "reactivate"):
        response = client.post(f"/admin/users/{stranger.id}/{action}", data={"csrf_token": token})
        assert response.status_code == 404, action
    db.refresh(stranger)
    assert stranger.active is True
    assert verify_password(stranger.password_hash, DEFAULT_PASSWORD)


def test_deactivate_then_reactivate(client: TestClient, db: Session, settings: Settings) -> None:
    admin, token = signed_in(client, db, settings, Role.FACILITY_ADMIN)
    target = make_user(db, role=Role.CLINICIAN, facility=admin.facility)
    db.commit()
    response = client.post(f"/admin/users/{target.id}/deactivate", data={"csrf_token": token})
    assert response.status_code == 303 and response.headers["location"] == "/admin/users"
    db.refresh(target)
    assert target.active is False
    page = client.get("/admin/users")
    assert '<span class="st retired">Deactivated</span>' in page.text
    assert f"/admin/users/{target.id}/reactivate" in page.text
    response = client.post(f"/admin/users/{target.id}/reactivate", data={"csrf_token": token})
    assert response.status_code == 303
    db.refresh(target)
    assert target.active is True


def test_nobody_deactivates_themselves(client: TestClient, db: Session, settings: Settings) -> None:
    admin, token = signed_in(client, db, settings, Role.SYSTEM_ADMIN)
    page = client.get("/admin/users")
    assert f"/admin/users/{admin.id}/deactivate" not in page.text
    response = client.post(f"/admin/users/{admin.id}/deactivate", data={"csrf_token": token})
    assert response.status_code == 400
    assert "You cannot deactivate your own account." in response.text
    db.refresh(admin)
    assert admin.active is True


def test_locked_accounts_are_marked(client: TestClient, db: Session, settings: Settings) -> None:
    admin, _ = signed_in(client, db, settings, Role.FACILITY_ADMIN)
    target = make_user(db, role=Role.CLINICIAN, facility=admin.facility)
    target.locked_until = utcnow() + timedelta(minutes=10)
    db.commit()
    assert '<span class="st awaiting">Locked</span>' in client.get("/admin/users").text
