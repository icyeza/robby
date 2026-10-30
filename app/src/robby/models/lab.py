"""Lab requests: a clinician asks the lab for a test on an admission, the lab technician records
the result, or the clinician cancels the request with a reason. Insert-only like every clinical
table (database triggers, migration 0008); the status of a request is derived: requested until
a result or a cancellation row exists, and a trigger lets a request be closed only once.

Free text and result values are encrypted at rest (EncryptedText): an HIV result is the most
sensitive value the database holds, and a note or reason may name the patient."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, false
from sqlalchemy.orm import Mapped, mapped_column

from robby.models.base import UUIDPK, Base, CreatedAt, EncryptedText
from robby.models.enums import LabTest, pg_enum

# Associated data of the encrypted columns (robby.crypto); changing one orphans stored rows.
LAB_NOTE_COLUMN = "lab_request.note"
LAB_VALUE_COLUMN = "lab_result.value"
LAB_REASON_COLUMN = "lab_cancellation.reason"


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
