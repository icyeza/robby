"""Patient identifiers are encrypted in PostgreSQL: the stored bytes hold neither the name nor the
number, the blind index is set on every insert, and SQL comparisons on ciphertext are refused."""

import uuid
from collections.abc import Callable

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from robby import crypto
from robby.models import Admission, AdmissionSource, patient_number_index
from robby.timeutil import utcnow
from tests.factories import make_facility, make_user

NUMBER, NAME = "ZQ-4242", "Zzyzx Encryptiontest"


def _saved(db: Session, number: str | None = NUMBER, name: str | None = NAME) -> Admission:
    facility = make_facility(db)
    admission = Admission(
        facility_id=facility.id,
        code=f"{facility.code}-ENC-{uuid.uuid4().hex[:6]}",
        admitted_at=utcnow(),
        source=AdmissionSource.FORM if number else AdmissionSource.FHIR,
        entered_by=make_user(db, facility=facility).id,
        patient_number=number,
        patient_name=name,
    )
    db.add(admission)
    db.flush()
    return admission


def test_the_stored_row_contains_neither_name_nor_number(db: Session) -> None:
    admission = _saved(db)
    raw = db.execute(
        text(
            "SELECT patient_number, patient_name, patient_number_index, a::text "
            "FROM admission a WHERE id = :id"
        ),
        {"id": admission.id},
    ).one()
    number, name, index, whole_row = raw
    assert isinstance(number, bytes) and isinstance(name, bytes)
    assert number[:1] == name[:1] == crypto.VERSION
    for plain in (NUMBER, NAME, NUMBER.lower(), "Zzyzx"):
        assert plain.encode() not in number + name
        assert plain not in whole_row and plain not in index
    columns = db.execute(
        text(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = 'admission' AND column_name LIKE 'patient%' ORDER BY column_name"
        )
    ).all()
    assert [tuple(c) for c in columns] == [
        ("patient_name", "bytea"),
        ("patient_number", "bytea"),
        ("patient_number_index", "character"),
    ]


def test_python_reads_the_plain_values_back(db: Session) -> None:
    admission = _saved(db)
    db.expire(admission)
    assert (admission.patient_number, admission.patient_name) == (NUMBER, NAME)


def test_the_blind_index_is_set_on_insert(db: Session) -> None:
    admission = _saved(db)
    assert admission.patient_number_index == patient_number_index(admission.facility_id, NUMBER)
    assert len(admission.patient_number_index or "") == 64


def test_no_number_means_no_index(db: Session) -> None:
    admission = _saved(db, number=None, name=None)
    assert admission.patient_number_index is None


def test_a_caller_cannot_store_a_different_index(db: Session) -> None:
    facility = make_facility(db)
    admission = Admission(
        facility_id=facility.id,
        code=f"{facility.code}-ENC-1",
        admitted_at=utcnow(),
        source=AdmissionSource.FORM,
        entered_by=make_user(db, facility=facility).id,
        patient_number=NUMBER,
        patient_name=NAME,
        patient_number_index="0" * 64,
    )
    db.add(admission)
    db.flush()
    assert admission.patient_number_index == patient_number_index(facility.id, NUMBER)


def test_the_database_refuses_a_number_without_its_index(db: Session) -> None:
    admission = _saved(db)
    with pytest.raises(IntegrityError, match="patient_number_indexed"), db.begin_nested():
        db.execute(
            text(
                "INSERT INTO admission (facility_id, code, admitted_at, source, entered_by, "
                "patient_number, patient_name) SELECT facility_id, 'ENC-RAW', admitted_at, source, "
                "entered_by, patient_number, patient_name FROM admission WHERE id = :id"
            ),
            {"id": admission.id},
        )


def test_a_value_copied_into_the_other_column_does_not_decrypt(db: Session) -> None:
    admission = _saved(db)
    with db.begin_nested():
        db.execute(
            text(
                "INSERT INTO admission (facility_id, code, admitted_at, source, entered_by, "
                "patient_number, patient_name, patient_number_index) SELECT facility_id, "
                "'ENC-SWAP', admitted_at, source, entered_by, patient_name, patient_number, "
                "patient_number_index FROM admission WHERE id = :id"
            ),
            {"id": admission.id},
        )
    with pytest.raises(crypto.DecryptionError):
        db.scalar(select(Admission.patient_name).where(Admission.code == "ENC-SWAP"))


@pytest.mark.parametrize(
    "compare",
    [
        lambda: Admission.patient_name == NAME,
        lambda: Admission.patient_number.in_([NUMBER]),
        lambda: Admission.patient_name.ilike("%zz%"),
    ],
)
def test_sql_comparisons_on_ciphertext_are_refused(compare: Callable[[], object]) -> None:
    with pytest.raises(TypeError, match="encrypted column"):
        compare()


def test_is_null_still_works(db: Session) -> None:
    admission = _saved(db, number=None, name=None)
    found = db.scalars(
        select(Admission.id).where(Admission.id == admission.id, Admission.patient_number.is_(None))
    ).all()
    assert found == [admission.id]
