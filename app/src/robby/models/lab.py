"""Lab requests: a clinician asks the lab for a test on an admission, the lab technician records
the result, or the clinician cancels the request with a reason. Insert-only like every clinical
table (database triggers, migration 0008); the status of a request is derived: requested until
a result or a cancellation row exists, and a trigger lets a request be closed only once.

Admission requests (migration 0009) work the same way for a patient the lab has not entered yet:
the clinician asks the lab to enter the admission, the lab technician's saved admission fulfils
the request, or the clinician cancels it with a reason.

Free text and result values are encrypted at rest (EncryptedText): an HIV result is the most
sensitive value the database holds, and a note or reason may name the patient. The patient
number and name of an admission request are encrypted and indexed exactly as on admission."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import CHAR, ForeignKey, event, false
from sqlalchemy.orm import Mapped, Mapper, mapped_column

from robby.models.base import UUIDPK, Base, CreatedAt, EncryptedText
from robby.models.clinical import patient_number_index
from robby.models.enums import LabTest, pg_enum

# Associated data of the encrypted columns (robby.crypto); changing one orphans stored rows.
LAB_NOTE_COLUMN = "lab_request.note"
LAB_VALUE_COLUMN = "lab_result.value"
LAB_REASON_COLUMN = "lab_cancellation.reason"
ASK_NUMBER_COLUMN = "admission_request.patient_number"
ASK_NAME_COLUMN = "admission_request.patient_name"
ASK_NOTE_COLUMN = "admission_request.note"
ASK_REASON_COLUMN = "admission_request_cancellation.reason"


class LabRequest(Base):
    """One test asked for one admission. Several tests asked together are several rows with the
    same note and urgency."""

    __tablename__ = "lab_request"

    id: Mapped[UUIDPK]
    admission_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("admission.id"), index=True)
    test: Mapped[LabTest] = mapped_column(pg_enum(LabTest, "lab_test"))
    urgent: Mapped[bool] = mapped_column(default=False, server_default=false())
    note: Mapped[str | None] = mapped_column(EncryptedText(LAB_NOTE_COLUMN))
    requested_by: Mapped[uuid.UUID] = mapped_column(ForeignKey("app_user.id"))
    requested_at: Mapped[datetime]
    created_at: Mapped[CreatedAt]


class LabResult(Base):
    """The result of a request, as the lab technician entered it (text; robby.lab.tests reads
    it back)."""

    __tablename__ = "lab_result"

    id: Mapped[UUIDPK]
    request_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("lab_request.id"), unique=True)
    value: Mapped[str] = mapped_column(EncryptedText(LAB_VALUE_COLUMN))
    recorded_by: Mapped[uuid.UUID] = mapped_column(ForeignKey("app_user.id"))
    resulted_at: Mapped[datetime]
    created_at: Mapped[CreatedAt]


class LabCancellation(Base):
    """A request the clinician withdrew, with the reason (required, checked in robby.lab)."""

    __tablename__ = "lab_cancellation"

    id: Mapped[UUIDPK]
    request_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("lab_request.id"), unique=True)
    reason: Mapped[str] = mapped_column(EncryptedText(LAB_REASON_COLUMN))
    cancelled_by: Mapped[uuid.UUID] = mapped_column(ForeignKey("app_user.id"))
    cancelled_at: Mapped[datetime]
    created_at: Mapped[CreatedAt]


class AdmissionRequest(Base):
    """A clinician's request that the lab enter the admission of a patient who has none yet."""

    __tablename__ = "admission_request"

    id: Mapped[UUIDPK]
    facility_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("facility.id"), index=True)
    # Encrypted like admission.patient_number / patient_name; the blind index (same key, same
    # facility scope) is set on insert by _index_patient_number below.
    patient_number: Mapped[str] = mapped_column(EncryptedText(ASK_NUMBER_COLUMN))
    patient_name: Mapped[str | None] = mapped_column(EncryptedText(ASK_NAME_COLUMN))
    patient_number_index: Mapped[str] = mapped_column(CHAR(64), index=True)
    urgent: Mapped[bool] = mapped_column(default=False, server_default=false())
    note: Mapped[str | None] = mapped_column(EncryptedText(ASK_NOTE_COLUMN))
    requested_by: Mapped[uuid.UUID] = mapped_column(ForeignKey("app_user.id"))
    requested_at: Mapped[datetime]
    created_at: Mapped[CreatedAt]


@event.listens_for(AdmissionRequest, "before_insert")
def _index_patient_number(
    mapper: Mapper[AdmissionRequest], connection: Any, target: AdmissionRequest
) -> None:
    """As for Admission: every insert derives the index from the number."""
    target.patient_number_index = patient_number_index(target.facility_id, target.patient_number)


class AdmissionRequestFulfilment(Base):
    """The admission the lab technician entered for a request (saved in the same transaction)."""

    __tablename__ = "admission_request_fulfilment"

    id: Mapped[UUIDPK]
    request_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("admission_request.id"), unique=True)
    admission_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("admission.id"), unique=True)
    fulfilled_by: Mapped[uuid.UUID] = mapped_column(ForeignKey("app_user.id"))
    fulfilled_at: Mapped[datetime]
    created_at: Mapped[CreatedAt]


class AdmissionRequestCancellation(Base):
    """An admission request the clinician withdrew, with the reason (required)."""

    __tablename__ = "admission_request_cancellation"

    id: Mapped[UUIDPK]
    request_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("admission_request.id"), unique=True)
    reason: Mapped[str] = mapped_column(EncryptedText(ASK_REASON_COLUMN))
    cancelled_by: Mapped[uuid.UUID] = mapped_column(ForeignKey("app_user.id"))
    cancelled_at: Mapped[datetime]
    created_at: Mapped[CreatedAt]
