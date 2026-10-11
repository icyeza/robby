"""Creating, listing and deactivating facilities."""

import pytest
from sqlalchemy.orm import Session

from robby.errors import FormErrors
from robby.facilities.service import create_facility, deactivate_facility, list_facilities
from robby.models import Role
from tests.factories import events_of, make_facility, make_user


def test_create_normalises_input_and_logs_an_event(db: Session) -> None:
    admin = make_user(db, role=Role.SYSTEM_ADMIN)
    facility = create_facility(
        db,
        admin,
        name="  Facility   Z ",
        code=" fz ",
        district=" North ",
        model_facility_key=" Site A ",
    )
    assert (facility.name, facility.code, facility.district) == ("Facility Z", "FZ", "North")
    assert facility.model_facility_key == "Site A"
    assert facility.active is True
    [event] = events_of(db, "facility.created")
    assert event.actor_id == admin.id and event.entity_id == facility.id
    assert event.payload == {"name": "Facility Z", "code": "FZ", "district": "North"}


def test_blank_model_key_is_stored_as_null(db: Session) -> None:
    admin = make_user(db, role=Role.SYSTEM_ADMIN)
    facility = create_facility(
        db, admin, name="Facility Y", code="FY", district="D", model_facility_key="  "
    )
    assert facility.model_facility_key is None


def test_every_problem_is_reported_at_once(db: Session) -> None:
    admin = make_user(db, role=Role.SYSTEM_ADMIN)
    with pytest.raises(FormErrors) as caught:
        create_facility(db, admin, name=" ", code="", district="", model_facility_key="")
    assert set(caught.value.errors) == {"name", "code", "district"}


@pytest.mark.parametrize("code", ["F", "FACILTY", "F1", "F-A"])
def test_codes_are_two_to_six_letters(db: Session, code: str) -> None:
    admin = make_user(db, role=Role.SYSTEM_ADMIN)
    with pytest.raises(FormErrors) as caught:
        create_facility(
            db, admin, name=f"Facility {code}", code=code, district="D", model_facility_key=""
        )
    assert caught.value.errors == {"code": "Use 2 to 6 capital letters, for example FA."}


def test_names_and_codes_must_be_new(db: Session) -> None:
    admin = make_user(db, role=Role.SYSTEM_ADMIN)
    make_facility(db, name="Facility Q", code="FQ")
    with pytest.raises(FormErrors) as caught:
        create_facility(
            db, admin, name="facility q", code="fq", district="D", model_facility_key=""
        )
    assert caught.value.errors == {
        "name": "A facility with this name already exists.",
        "code": "This code is already used by another facility.",
    }


def test_list_names_administrators_and_counts_active_users(db: Session) -> None:
    facility = make_facility(db)
    make_user(db, role=Role.FACILITY_ADMIN, facility=facility, username="fa.q02")
    make_user(db, role=Role.FACILITY_ADMIN, facility=facility, username="fa.q01")
    make_user(db, role=Role.CLINICIAN, facility=facility)
    make_user(db, role=Role.CLINICIAN, facility=facility, active=False)
    empty = make_facility(db)
    rows = {row.facility.id: row for row in list_facilities(db)}
    assert rows[facility.id].administrators == ["fa.q01", "fa.q02"]
    assert rows[facility.id].active_users == 3
    assert rows[empty.id].administrators == [] and rows[empty.id].active_users == 0


def test_deactivate_once(db: Session) -> None:
    admin = make_user(db, role=Role.SYSTEM_ADMIN)
    facility = make_facility(db)
    deactivate_facility(db, admin, facility)
    deactivate_facility(db, admin, facility)
    assert facility.active is False
    [event] = events_of(db, "facility.deactivated")
    assert event.entity_id == facility.id and event.payload == {"code": facility.code}
