"""Admission codes from admission_counter, issued inside the caller's transaction."""

from __future__ import annotations

from typing import cast

from sqlalchemy import Table
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from robby.models import AdmissionCounter, Facility


def next_admission_code(db: Session, facility: Facility) -> str:
    """Issue the facility's next code. The upsert locks the counter row until the transaction
    ends, so concurrent admissions at one facility never receive the same number."""
    table = cast(Table, AdmissionCounter.__table__)
    statement = (
        pg_insert(table)
        .values({table.c.facility_id: facility.id, table.c.next: 2})
        .on_conflict_do_update(
            index_elements=[table.c.facility_id], set_={table.c.next: table.c.next + 1}
        )
        .returning(table.c.next)
    )
    issued = db.execute(statement).scalar_one() - 1
    return f"{facility.code}-{issued:04d}"
