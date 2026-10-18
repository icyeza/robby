"""Database-level guarantees: UUID defaults, CHECK constraints, partial unique indexes and
the triggers that make clinical rows and the event log immutable."""

import uuid
from datetime import date

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session

from robby.models import (
    Admission,
    AdmissionSource,
    AppUser,
    Classification,
    ClassificationStatus,
    ExportRecord,
    Facility,
    ModelState,
    Role,
)
from robby.timeutil import utcnow
from tests.expected import IMMUTABLE_TABLES
from tests.factories import (
    make_clinical_chain,
    make_facility,
    make_model_version,
    make_rule_set,
    make_user,
)


def assert_rejected(db: Session, *objects: object) -> None:
    with pytest.raises(IntegrityError), db.begin_nested():
        db.add_all(objects)
        db.flush()


def test_database_generates_uuid_ids(db: Session) -> None:
    new_id = db.execute(
        text("INSERT INTO facility (name, code, district) VALUES ('Raw', 'RAW', 'D') RETURNING id")
    ).scalar_one()
    assert isinstance(new_id, uuid.UUID)


@pytest.mark.parametrize("code", ["fa", "F", "FACILTY", "F1"])
def test_facility_code_is_two_to_six_capital_letters(db: Session, code: str) -> None:
    assert_rejected(db, Facility(name=f"Name {code}", code=code, district="D"))


def test_facility_names_and_codes_are_unique(db: Session) -> None:
    make_facility(db, name="Facility Q", code="FQ")
    assert_rejected(db, Facility(name="Facility Q", code="FR", district="D"))
    assert_rejected(db, Facility(name="Facility R", code="FQ", district="D"))


def test_clinician_needs_a_facility(db: Session) -> None:
    assert_rejected(db, AppUser(username="mw.x", password_hash="h", role=Role.CLINICIAN))


def test_system_admin_cannot_belong_to_a_facility(db: Session) -> None:
    facility = make_facility(db)
    assert_rejected(
        db,
        AppUser(
            username="sa.x", password_hash="h", role=Role.SYSTEM_ADMIN, facility_id=facility.id
        ),
    )


def test_usernames_are_stored_lowercase(db: Session) -> None:
    assert_rejected(db, AppUser(username="SA.X", password_hash="h", role=Role.SYSTEM_ADMIN))


def test_only_one_rule_set_is_active(db: Session) -> None:
    make_rule_set(db, active=True)
    with pytest.raises(IntegrityError), db.begin_nested():
        make_rule_set(db, active=True)


def test_only_one_model_version_is_active(db: Session) -> None:
    make_model_version(db, state=ModelState.ACTIVE)
    make_model_version(db, state=ModelState.SHADOW)
    with pytest.raises(IntegrityError), db.begin_nested():
        make_model_version(db, state=ModelState.ACTIVE)


def test_a_correction_needs_a_reason(db: Session) -> None:
    chain = make_clinical_chain(db)
    original = chain["admission"]
    assert_rejected(
        db,
        Admission(
            facility_id=original.facility_id,
            code="CORR-1",
            admitted_at=utcnow(),
            source=AdmissionSource.FORM,
            entered_by=original.entered_by,
            supersedes_id=original.id,
            correction_reason=None,
            patient_number=original.patient_number,
            patient_name=original.patient_name,
        ),
    )


def test_a_form_admission_records_who_entered_it(db: Session) -> None:
    facility = make_facility(db)
    assert_rejected(
        db,
        Admission(
            facility_id=facility.id,
            code="NOAUTH-1",
            admitted_at=utcnow(),
            source=AdmissionSource.FORM,
            entered_by=None,
            patient_number="T-0001",
            patient_name="Test Patient",
        ),
    )


def _bare_admission(
    db: Session,
    source: AdmissionSource,
    *,
    patient_number: str | None = None,
    patient_name: str | None = None,
) -> Admission:
    facility = make_facility(db)
    author = None if source is AdmissionSource.FHIR else make_user(db, facility=facility).id
    return Admission(
        facility_id=facility.id,
        code=f"{facility.code}-PAT-{uuid.uuid4().hex[:6]}",
        admitted_at=utcnow(),
        source=source,
        entered_by=author,
        patient_number=patient_number,
        patient_name=patient_name,
    )


def test_a_form_admission_needs_a_patient_number(db: Session) -> None:
    admission = _bare_admission(db, AdmissionSource.FORM, patient_name="Test Patient")
    with pytest.raises(IntegrityError, match="entry_has_patient"), db.begin_nested():
        db.add(admission)
        db.flush()


def test_an_imported_admission_needs_a_patient_name(db: Session) -> None:
    admission = _bare_admission(db, AdmissionSource.IMPORT, patient_number="T-0001")
    with pytest.raises(IntegrityError, match="entry_has_patient"), db.begin_nested():
        db.add(admission)
        db.flush()


def test_a_fhir_admission_may_have_no_patient_identifiers(db: Session) -> None:
    admission = _bare_admission(db, AdmissionSource.FHIR)
    db.add(admission)
    db.flush()
    assert admission.patient_number is None


def test_patient_identifiers_are_stored_as_given(db: Session) -> None:
    admission = _bare_admission(
        db, AdmissionSource.FORM, patient_number="RW/123-4", patient_name="Uwase Marie"
    )
    db.add(admission)
    db.flush()
    db.expire(admission)
    assert (admission.patient_number, admission.patient_name) == ("RW/123-4", "Uwase Marie")


def test_a_resolved_classification_has_a_group(db: Session) -> None:
    chain = make_clinical_chain(db)
    assert_rejected(
        db,
        Classification(
            admission_id=chain["admission"].id,
            rule_set_id=make_rule_set(db).id,
            status=ClassificationStatus.RESOLVED,
            robson_group=None,
            trace={},
        ),
    )


def test_export_period_is_the_first_of_a_month(db: Session) -> None:
    facility = make_facility(db)
    admin = make_user(db, role=Role.FACILITY_ADMIN, facility=facility)
    assert_rejected(
        db,
        ExportRecord(
            facility_id=facility.id,
            period_month=date(2026, 9, 15),
            exported_by=admin.id,
            file_sha256="0" * 64,
        ),
    )


@pytest.mark.parametrize("table", IMMUTABLE_TABLES)
def test_immutable_rows_cannot_be_updated(db: Session, table: str) -> None:
    make_clinical_chain(db)
    with pytest.raises(DBAPIError, match=rf"UPDATE on {table} is not allowed"), db.begin_nested():
        db.execute(text(f"UPDATE {table} SET created_at = created_at"))


@pytest.mark.parametrize("table", IMMUTABLE_TABLES)
def test_immutable_rows_cannot_be_deleted(db: Session, table: str) -> None:
    make_clinical_chain(db)
    with pytest.raises(DBAPIError, match=rf"DELETE on {table} is not allowed"), db.begin_nested():
        db.execute(text(f"DELETE FROM {table}"))


@pytest.mark.parametrize("table", IMMUTABLE_TABLES)
def test_immutable_tables_cannot_be_truncated(db: Session, table: str) -> None:
    make_clinical_chain(db)
    with pytest.raises(DBAPIError, match=r"TRUNCATE on \w+ is not allowed"), db.begin_nested():
        db.execute(text(f"TRUNCATE {table} CASCADE"))


def test_other_tables_stay_writable(db: Session) -> None:
    facility = make_facility(db)
    db.execute(text("UPDATE facility SET district = 'Moved' WHERE id = :id"), {"id": facility.id})
    db.refresh(facility)
    assert facility.district == "Moved"
