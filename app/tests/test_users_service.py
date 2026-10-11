"""Accounts: who may create which roles where, temporary passwords, resets, deactivation."""

import pytest
from sqlalchemy.orm import Session

from robby.auth.passwords import verify_password
from robby.auth.service import start_session
from robby.config import Settings
from robby.errors import FormErrors, NotAllowed, NotFound
from robby.models import Role
from robby.timeutil import utcnow
from robby.users.service import (
    creatable_roles,
    create_user,
    get_user_in_scope,
    list_users,
    reset_password,
    set_active,
)
from tests.factories import DEFAULT_PASSWORD, events_of, make_facility, make_user


def test_system_admin_creates_a_clinician_with_a_temporary_password(db: Session) -> None:
    admin = make_user(db, role=Role.SYSTEM_ADMIN)
    facility = make_facility(db)
    created = create_user(
        db,
        admin,
        username=" New.User ",
        display_name=" Ama ",
        role="clinician",
        facility_id=facility.id,
    )
    user = created.user
    assert (user.username, user.display_name, user.role) == ("new.user", "Ama", Role.CLINICIAN)
    assert user.facility_id == facility.id
    assert user.must_change_password is True
    assert created.temporary_password is not None
    assert verify_password(user.password_hash, created.temporary_password)
    [event] = events_of(db, "user.created")
    assert event.actor_id == admin.id and event.entity_id == user.id
    assert event.payload == {
        "username": "new.user",
        "role": "clinician",
        "facility_id": str(facility.id),
    }


def test_creation_with_a_chosen_password_needs_no_change(db: Session) -> None:
    created = create_user(
        db, None, username="root.admin", role=Role.SYSTEM_ADMIN, password="a-strong-password"
    )
    assert created.temporary_password is None
    assert created.user.must_change_password is False
    assert created.user.display_name is None
    assert verify_password(created.user.password_hash, "a-strong-password")
    assert events_of(db, "user.created")[0].actor_id is None


def test_short_chosen_password_is_refused(db: Session) -> None:
    with pytest.raises(FormErrors) as caught:
        create_user(db, None, username="root.admin", role=Role.SYSTEM_ADMIN, password="short")
    assert caught.value.errors == {"password": "Use at least 10 characters."}


def test_facility_roles_need_an_active_facility(db: Session) -> None:
    admin = make_user(db, role=Role.SYSTEM_ADMIN)
    closed = make_facility(db, active=False)
    for facility_id in (None, closed.id):
        with pytest.raises(FormErrors) as caught:
            create_user(db, admin, username="mw.x01", role="clinician", facility_id=facility_id)
        assert set(caught.value.errors) == {"facility_id"}


def test_other_roles_are_never_tied_to_a_facility(db: Session) -> None:
    admin = make_user(db, role=Role.SYSTEM_ADMIN)
    facility = make_facility(db)
    created = create_user(
        db, admin, username="mc.01", role="model_custodian", facility_id=facility.id
    )
    assert created.user.facility_id is None


def test_facility_admin_creates_only_facility_roles_in_their_own_facility(db: Session) -> None:
    admin = make_user(db, role=Role.FACILITY_ADMIN)
    elsewhere = make_facility(db)
    created = create_user(db, admin, username="mw.new", role="clinician", facility_id=elsewhere.id)
    assert created.user.facility_id == admin.facility_id
    with pytest.raises(FormErrors) as caught:
        create_user(db, admin, username="mc.new", role="model_custodian")
    assert caught.value.errors == {"role": "You cannot create accounts with this role."}
    assert creatable_roles(admin) == (Role.CLINICIAN, Role.FACILITY_ADMIN)


@pytest.mark.parametrize("username", ["ab", "has space", "-starts.with.dash", "x" * 65])
def test_usernames_follow_the_pattern(db: Session, username: str) -> None:
    with pytest.raises(FormErrors) as caught:
        create_user(
            db, None, username=username, role=Role.SYSTEM_ADMIN, password="a-strong-password"
        )
    assert set(caught.value.errors) == {"username"}


def test_usernames_are_unique(db: Session) -> None:
    make_user(db, role=Role.SYSTEM_ADMIN, username="taken.name")
    with pytest.raises(FormErrors) as caught:
        create_user(
            db, None, username="Taken.Name", role=Role.SYSTEM_ADMIN, password="a-strong-password"
        )
    assert caught.value.errors == {"username": "This username is already taken."}


def test_unknown_role_is_refused(db: Session) -> None:
    with pytest.raises(FormErrors) as caught:
        create_user(db, None, username="who.ever", role="superuser", password="a-strong-password")
    assert caught.value.errors == {"role": "Choose a role."}


def test_listing_is_scoped_to_the_facility_admins_facility(db: Session) -> None:
    admin = make_user(db, role=Role.FACILITY_ADMIN)
    colleague = make_user(db, role=Role.CLINICIAN, facility=admin.facility)
    stranger = make_user(db, role=Role.CLINICIAN)
    system = make_user(db, role=Role.SYSTEM_ADMIN)
    assert {u.id for u in list_users(db, admin)} == {admin.id, colleague.id}
    assert {admin.id, colleague.id, stranger.id, system.id} <= {
        u.id for u in list_users(db, system)
    }
    with pytest.raises(NotAllowed):
        list_users(db, colleague)


def test_users_outside_scope_are_not_found(db: Session) -> None:
    admin = make_user(db, role=Role.FACILITY_ADMIN)
    stranger = make_user(db, role=Role.CLINICIAN)
    with pytest.raises(NotFound):
        get_user_in_scope(db, admin, stranger.id)
    assert get_user_in_scope(db, admin, admin.id) is admin
    with pytest.raises(NotAllowed):
        get_user_in_scope(db, stranger, admin.id)


def test_reset_password_issues_a_new_temporary_password_and_ends_sessions(
    db: Session, settings: Settings
) -> None:
    admin = make_user(db, role=Role.SYSTEM_ADMIN)
    target = make_user(db)
    target.failed_logins = 3
    target.locked_until = utcnow()
    session_row, _ = start_session(db, settings, target, ip=None, user_agent=None, now=utcnow())
    temporary = reset_password(db, admin, target, now=utcnow())
    assert verify_password(target.password_hash, temporary)
    assert not verify_password(target.password_hash, DEFAULT_PASSWORD)
    assert target.must_change_password is True
    assert (target.failed_logins, target.locked_until) == (0, None)
    db.refresh(session_row)
    assert session_row.revoked_at is not None
    [event] = events_of(db, "user.password_reset")
    assert event.entity_id == target.id and event.payload == {"username": target.username}


def test_deactivate_and_reactivate(db: Session, settings: Settings) -> None:
    admin = make_user(db, role=Role.FACILITY_ADMIN)
    target = make_user(db, role=Role.CLINICIAN, facility=admin.facility)
    session_row, _ = start_session(db, settings, target, ip=None, user_agent=None, now=utcnow())
    set_active(db, admin, target, active=False, now=utcnow())
    set_active(db, admin, target, active=False, now=utcnow())
    assert target.active is False
    db.refresh(session_row)
    assert session_row.revoked_at is not None
    set_active(db, admin, target, active=True, now=utcnow())
    assert target.active is True
    assert len(events_of(db, "user.deactivated")) == 1
    assert len(events_of(db, "user.reactivated")) == 1


def test_nobody_deactivates_their_own_account(db: Session) -> None:
    admin = make_user(db, role=Role.SYSTEM_ADMIN)
    with pytest.raises(NotAllowed, match="your own account"):
        set_active(db, admin, admin, active=False, now=utcnow())
