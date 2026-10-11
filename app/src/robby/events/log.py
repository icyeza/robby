"""Append-only, hash-chained event log (spec §4, NFR5).

hash = sha256(prev_hash || canonical_json({id, occurred_at, actor_id, event_type, entity_type,
entity_id, payload})), canonical JSON = sorted keys, no whitespace, UTF-8. The first entry's
prev_hash is 64 zeros. seq is not hashed; the chain itself fixes the order."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from robby.models import AppUser, EventLog

GENESIS_HASH = "0" * 64
ADVISORY_LOCK_KEY = 0x524F4242  # "ROBB": one lock serialises every append, across connections
TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
BROKEN_LINK = "prev_hash does not match the previous entry's hash"
ALTERED_CONTENT = "hash does not match the entry's content"


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    checked: int
    first_bad_seq: int | None
    reason: str | None


def _check_json(value: Any, path: str = "payload") -> None:
    """Payloads hold only values that survive a jsonb round trip unchanged (no floats)."""
    if value is None or isinstance(value, bool | int | str):
        return
    if isinstance(value, float):
        raise TypeError(f"{path} is a float; store numbers as int or str")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _check_json(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} has a non-string key")
            _check_json(item, f"{path}.{key}")
        return
    raise TypeError(f"{path} has unsupported type {type(value).__name__}")


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def compute_hash(
    prev_hash: str,
    *,
    event_id: uuid.UUID,
    occurred_at: datetime,
    actor_id: uuid.UUID | None,
    event_type: str,
    entity_type: str | None,
    entity_id: uuid.UUID | None,
    payload: dict[str, Any],
) -> str:
    body = {
        "id": str(event_id),
        "occurred_at": occurred_at.astimezone(UTC).strftime(TIME_FORMAT),
        "actor_id": str(actor_id) if actor_id is not None else None,
        "event_type": event_type,
        "entity_type": entity_type,
        "entity_id": str(entity_id) if entity_id is not None else None,
        "payload": payload,
    }
    return hashlib.sha256(prev_hash.encode("ascii") + canonical_json(body)).hexdigest()


def append(
    session: Session,
    actor: AppUser | None,
    event_type: str,
    entity: tuple[str, uuid.UUID] | None = None,
    payload: dict[str, Any] | None = None,
) -> EventLog:
    """Append one entry inside the caller's transaction (the only writer of event_log).

    actor None means the system. The advisory lock is held until the transaction ends, so the
    entry read as "latest" is still the latest when this one is inserted."""
    data: dict[str, Any] = dict(payload or {})
    _check_json(data)
    session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": ADVISORY_LOCK_KEY})
    prev_hash = (
        session.scalar(select(EventLog.hash).order_by(EventLog.seq.desc()).limit(1)) or GENESIS_HASH
    )
    entity_type, entity_id = entity if entity is not None else (None, None)
    actor_id = actor.id if actor is not None else None
    event_id = uuid.uuid4()
    occurred_at = datetime.now(UTC)
    row = EventLog(
        id=event_id,
        occurred_at=occurred_at,
        actor_id=actor_id,
        event_type=event_type,
        entity_type=entity_type,
        entity_id=entity_id,
        payload=data,
        prev_hash=prev_hash,
        hash=compute_hash(
            prev_hash,
            event_id=event_id,
            occurred_at=occurred_at,
            actor_id=actor_id,
            event_type=event_type,
            entity_type=entity_type,
            entity_id=entity_id,
            payload=data,
        ),
    )
    session.add(row)
    session.flush()
    return row


def verify(session: Session) -> VerifyResult:
    """Walk the log in seq order, recomputing every hash. Reads columns, not ORM objects, so
    what is checked is what is stored."""
    statement = select(
        EventLog.seq, EventLog.id, EventLog.occurred_at, EventLog.actor_id, EventLog.event_type,
        EventLog.entity_type, EventLog.entity_id, EventLog.payload, EventLog.prev_hash,
        EventLog.hash,
    ).order_by(EventLog.seq)
    expected_prev = GENESIS_HASH
    checked = 0
    for row in session.execute(statement.execution_options(yield_per=1000)):
        if row.prev_hash != expected_prev:
            return VerifyResult(False, checked, row.seq, BROKEN_LINK)
        recomputed = compute_hash(
            row.prev_hash,
            event_id=row.id,
            occurred_at=row.occurred_at,
            actor_id=row.actor_id,
            event_type=row.event_type,
            entity_type=row.entity_type,
            entity_id=row.entity_id,
            payload=row.payload,
        )
        if recomputed != row.hash:
            return VerifyResult(False, checked, row.seq, ALTERED_CONTENT)
        expected_prev = row.hash
        checked += 1
    return VerifyResult(True, checked, None, None)
