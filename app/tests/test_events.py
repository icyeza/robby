"""The hash chain: append serialises and links entries; verify recomputes and finds tampering."""

import hashlib
import json
from datetime import UTC

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from robby import events
from robby.events.log import ADVISORY_LOCK_KEY, GENESIS_HASH, VerifyResult, compute_hash
from tests.factories import make_user


def test_first_entry_chains_from_the_genesis_hash(db: Session) -> None:
    row = events.append(db, None, "test.first", payload={"n": 1})
    assert row.prev_hash == GENESIS_HASH == "0" * 64
    assert len(row.hash) == 64


def test_entries_link_to_the_previous_hash(db: Session) -> None:
    first = events.append(db, None, "test.a")
    second = events.append(db, None, "test.b")
    assert second.prev_hash == first.hash
    assert second.seq > first.seq


def test_hash_is_sha256_of_prev_hash_and_canonical_json(db: Session) -> None:
    actor = make_user(db)
    row = events.append(db, actor, "user.created", ("app_user", actor.id), {"b": 2, "a": "é"})
    body = {
        "actor_id": str(actor.id),
        "entity_id": str(actor.id),
        "entity_type": "app_user",
        "event_type": "user.created",
        "id": str(row.id),
        "occurred_at": row.occurred_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "payload": {"a": "é", "b": 2},
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    expected = hashlib.sha256(GENESIS_HASH.encode("ascii") + canonical.encode("utf-8")).hexdigest()
    assert row.hash == expected
    assert compute_hash(
        GENESIS_HASH, event_id=row.id, occurred_at=row.occurred_at, actor_id=actor.id,
        event_type="user.created", entity_type="app_user", entity_id=actor.id,
        payload={"b": 2, "a": "é"},
    ) == expected


def test_append_holds_the_advisory_lock_until_the_transaction_ends(db: Session) -> None:
    events.append(db, None, "test.lock")
    held = db.execute(
        text(
            "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' "
            "AND CAST(objid AS bigint) = :key AND pid = pg_backend_pid()"
        ),
        {"key": ADVISORY_LOCK_KEY},
    ).scalar_one()
    assert held == 1


def test_payload_must_be_json_without_floats(db: Session) -> None:
    with pytest.raises(TypeError, match="float"):
        events.append(db, None, "test.float", payload={"p": 0.5})


def test_verify_an_empty_log(db: Session) -> None:
    assert events.verify(db) == VerifyResult(ok=True, checked=0, first_bad_seq=None, reason=None)


def test_verify_an_intact_log(db: Session) -> None:
    for n in range(3):
        events.append(db, None, "test.n", payload={"n": n})
    assert events.verify(db) == VerifyResult(ok=True, checked=3, first_bad_seq=None, reason=None)


def _tamper(db: Session, sql: str, params: dict[str, object]) -> None:
    db.execute(text("ALTER TABLE event_log DISABLE TRIGGER event_log_immutable"))
    try:
        db.execute(text(sql), params)
    finally:
        db.execute(text("ALTER TABLE event_log ENABLE TRIGGER event_log_immutable"))


def test_verify_detects_an_altered_payload(db: Session) -> None:
    rows = [events.append(db, None, "test.n", payload={"n": n}) for n in range(3)]
    _tamper(
        db, "UPDATE event_log SET payload = CAST(:p AS jsonb) WHERE seq = :seq",
        {"p": json.dumps({"n": 99}), "seq": rows[1].seq},
    )
    result = events.verify(db)
    assert result.ok is False
    assert result.first_bad_seq == rows[1].seq
    assert result.checked == 1
    assert result.reason == "hash does not match the entry's content"


def test_verify_detects_a_rewritten_entry_by_the_broken_link(db: Session) -> None:
    rows = [events.append(db, None, "test.n", payload={"n": n}) for n in range(3)]
    forged = compute_hash(
        rows[1].prev_hash, event_id=rows[1].id, occurred_at=rows[1].occurred_at, actor_id=None,
        event_type="test.n", entity_type=None, entity_id=None, payload={"n": 99},
    )
    _tamper(
        db, "UPDATE event_log SET payload = CAST(:p AS jsonb), hash = :h WHERE seq = :seq",
        {"p": json.dumps({"n": 99}), "h": forged, "seq": rows[1].seq},
    )
    result = events.verify(db)
    assert result.ok is False
    assert result.first_bad_seq == rows[2].seq
    assert result.checked == 2
    assert result.reason == "prev_hash does not match the previous entry's hash"
