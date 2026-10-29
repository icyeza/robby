"""Admission codes are <FACILITY CODE>-<zero-padded per-facility sequence>."""

from sqlalchemy.orm import Session

from robby.facilities.admission_codes import next_admission_code
from robby.models import AdmissionCounter
from tests.factories import make_facility


def test_codes_count_up_separately_for_each_facility(db: Session) -> None:
    a = make_facility(db, code="FA")
    b = make_facility(db, code="FB")
    assert next_admission_code(db, a) == "FA-0001"
    assert next_admission_code(db, a) == "FA-0002"
    assert next_admission_code(db, b) == "FB-0001"
    assert next_admission_code(db, a) == "FA-0003"


def test_counter_row_holds_the_next_number(db: Session) -> None:
    facility = make_facility(db, code="FC")
    next_admission_code(db, facility)
    next_admission_code(db, facility)
    counter = db.get(AdmissionCounter, facility.id)
    assert counter is not None
    db.refresh(counter)
    assert counter.next_value == 3


def test_codes_keep_growing_past_four_digits(db: Session) -> None:
    facility = make_facility(db, code="FD")
    db.add(AdmissionCounter(facility_id=facility.id, next_value=10000))
    db.flush()
    assert next_admission_code(db, facility) == "FD-10000"
