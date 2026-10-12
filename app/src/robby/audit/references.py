"""Reference populations (Vogel, C-Model) as versioned, cited per-group tables.

Nothing is seeded here: the tables come from a CSV an operator loads with
`robby-admin references load`. Columns: robson_group, group_size_pct, cs_rate_pct."""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from sqlalchemy import select
from sqlalchemy.orm import Session

from robby import events
from robby.models import AppUser, ReferenceKind, ReferenceSet, ReferenceValue

SIZE_SUM_TOLERANCE = Decimal(1)


class ReferenceFileError(ValueError):
    """The reference CSV is not usable; the message says why."""


@dataclass(frozen=True)
class ReferenceRow:
    group: int
    size_pct: Decimal
    cs_rate_pct: Decimal


@dataclass(frozen=True)
class Reference:
    """A loaded reference set with its ten rows, ready for the report arithmetic."""

    id: str
    name: str
    kind: str
    citation: str
    population: str
    version_label: str
    rows: dict[int, ReferenceRow]

    @property
    def label(self) -> str:
        return f"{self.name} ({self.version_label})"


def _number(raw: str, what: str, line: int) -> Decimal:
    try:
        value = Decimal(raw.strip())
    except InvalidOperation as exc:
        raise ReferenceFileError(f"Line {line}: {what} '{raw}' is not a number.") from exc
    if not value.is_finite() or value < 0 or value > 100:
        raise ReferenceFileError(f"Line {line}: {what} {raw} must be between 0 and 100.")
    return value


def parse_reference_csv(text: str) -> list[ReferenceRow]:
    """Validate: header, groups 1-10 once each, percentages 0-100, sizes sum to 100 +/- 1."""
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    needed = {"robson_group", "group_size_pct", "cs_rate_pct"}
    if reader.fieldnames is None or not needed <= {f.strip() for f in reader.fieldnames}:
        raise ReferenceFileError(
            "The CSV needs the columns robson_group, group_size_pct, cs_rate_pct."
        )
    rows: dict[int, ReferenceRow] = {}
    for line, raw_record in enumerate(reader, start=2):
        record = {k.strip(): (v or "") for k, v in raw_record.items() if k}
        try:
            group = int(record["robson_group"].strip())
        except ValueError as exc:
            raise ReferenceFileError(f"Line {line}: robson_group must be 1 to 10.") from exc
        if not 1 <= group <= 10:
            raise ReferenceFileError(f"Line {line}: robson_group {group} must be 1 to 10.")
        if group in rows:
            raise ReferenceFileError(f"Line {line}: group {group} appears twice.")
        rows[group] = ReferenceRow(
            group,
            _number(record["group_size_pct"], "group_size_pct", line),
            _number(record["cs_rate_pct"], "cs_rate_pct", line),
        )
    if sorted(rows) != list(range(1, 11)):
        missing = sorted(set(range(1, 11)) - set(rows))
        raise ReferenceFileError(f"The CSV must hold exactly the 10 groups; missing {missing}.")
    total = sum((r.size_pct for r in rows.values()), Decimal(0))
    if abs(total - 100) > SIZE_SUM_TOLERANCE:
        raise ReferenceFileError(f"Group sizes sum to {total}, not 100 (tolerance 1).")
    return [rows[g] for g in range(1, 11)]


def load_reference_set(
    db: Session,
    actor: AppUser | None,
    *,
    csv_bytes: bytes,
    name: str,
    kind: ReferenceKind,
    citation: str,
    population: str,
    version_label: str,
) -> ReferenceSet:
    rows = parse_reference_csv(csv_bytes.decode("utf-8"))
    if not all(x.strip() for x in (name, citation, population, version_label)):
        raise ReferenceFileError("Name, citation, population and version are all required.")
    exists = db.scalar(
        select(ReferenceSet.id).where(
            ReferenceSet.name == name, ReferenceSet.version_label == version_label
        )
    )
    if exists is not None:
        raise ReferenceFileError(
            f"A reference set '{name}' version '{version_label}' is already loaded; "
            "a correction is a new version."
        )
    ref = ReferenceSet(
        name=name,
        kind=kind,
        citation=citation,
        population=population,
        version_label=version_label,
        loaded_by=actor.id if actor else None,
    )
    db.add(ref)
    db.flush()
    db.add_all(
        ReferenceValue(
            reference_set_id=ref.id,
            robson_group=r.group,
            group_size_pct=r.size_pct,
            cs_rate_pct=r.cs_rate_pct,
        )
        for r in rows
    )
    db.flush()
    events.append(
        db,
        actor,
        "reference_set.loaded",
        entity=("reference_set", ref.id),
        payload={
            "name": name,
            "kind": kind.value,
            "version": version_label,
            "sha256": hashlib.sha256(csv_bytes).hexdigest(),
        },
    )
    return ref


def _to_reference(db: Session, ref: ReferenceSet) -> Reference:
    values = db.scalars(select(ReferenceValue).where(ReferenceValue.reference_set_id == ref.id))
    return Reference(
        id=str(ref.id),
        name=ref.name,
        kind=ref.kind.value,
        citation=ref.citation,
        population=ref.population,
        version_label=ref.version_label,
        rows={
            v.robson_group: ReferenceRow(v.robson_group, v.group_size_pct, v.cs_rate_pct)
            for v in values
        },
    )


def available_references(db: Session) -> list[Reference]:
    """Loaded sets, Vogel first, newest first within a kind."""
    sets = db.scalars(select(ReferenceSet).order_by(ReferenceSet.loaded_at.desc())).all()
    ordered = sorted(sets, key=lambda s: s.kind is not ReferenceKind.VOGEL)
    return [_to_reference(db, s) for s in ordered]


def choose_reference(refs: list[Reference], requested: str | None) -> Reference | None:
    """The requested set, else the newest Vogel set, else the newest of any kind, else none."""
    for ref in refs:
        if requested and ref.id == requested:
            return ref
    return refs[0] if refs else None
