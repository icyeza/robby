"""Reconciliation (FR12): join every prediction to the outcome of its admission's chain.

An outcome stays on the admission version it was recorded on; predictions may sit on any version
of the same chain (original or corrected), so the join goes through the chain root. Rows are
only ever inserted, one per prediction (active and shadow alike); a second run finds nothing
new. The scheduled run takes a Postgres advisory lock so several workers never run it at once."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session, sessionmaker

from robby import events
from robby.models import DeliveryMode, ReconciledOutcome

logger = logging.getLogger("robby")
RECONCILE_LOCK_KEY = 0x52454331  # "REC1"; distinct from the event log's lock

_PENDING = text(
    """
    WITH RECURSIVE chain(id, root) AS (
        SELECT id, id FROM admission WHERE supersedes_id IS NULL
        UNION ALL
        SELECT a.id, c.root FROM admission a JOIN chain c ON a.supersedes_id = c.id
    )
    SELECT DISTINCT ON (p.id) p.id AS prediction_id, o.id AS outcome_id, o.mode AS mode
    FROM prediction p
    JOIN chain cp ON cp.id = p.admission_id
    JOIN chain co ON co.root = cp.root
    JOIN delivery_outcome o ON o.admission_id = co.id
    WHERE NOT EXISTS (SELECT 1 FROM reconciled_outcome r WHERE r.prediction_id = p.id)
    ORDER BY p.id, o.created_at
    """
)


def reconcile(db: Session, *, log_empty: bool = True) -> dict[str, int]:
    """Insert the missing reconciled_outcome rows. Does not commit.

    Returns counts only: reconciled (new rows now), total (rows after this run), awaiting
    (predictions still without an outcome). The event is skipped for an empty run when
    log_empty is False (the scheduler, so a quiet system does not fill the log)."""
    new = 0
    for row in db.execute(_PENDING).all():
        db.add(
            ReconciledOutcome(
                prediction_id=row.prediction_id,
                delivery_outcome_id=row.outcome_id,
                cesarean=row.mode == DeliveryMode.CESAREAN.value,
            )
        )
        new += 1
    db.flush()
    total = db.scalar(select(func.count()).select_from(ReconciledOutcome)) or 0
    predictions = db.scalar(text("SELECT count(*) FROM prediction")) or 0
    counts = {"reconciled": new, "total": int(total), "awaiting": int(predictions) - int(total)}
    if new or log_empty:
        events.append(db, None, "reconciliation.run", None, dict(counts))
    return counts


def run_scheduled(factory: sessionmaker[Session]) -> dict[str, int] | None:
    """One scheduled pass. None when another worker holds the lock."""
    with factory() as db:
        got = db.scalar(text("SELECT pg_try_advisory_xact_lock(:k)"), {"k": RECONCILE_LOCK_KEY})
        if not got:
            return None
        counts = reconcile(db, log_empty=False)
        db.commit()
        return counts


async def reconcile_loop(factory: sessionmaker[Session], interval_minutes: int) -> None:
    """Run forever (until cancelled): wait, reconcile, repeat. Failures are logged only."""
    while True:
        await asyncio.sleep(interval_minutes * 60)
        try:
            counts: dict[str, Any] | None = await asyncio.to_thread(run_scheduled, factory)
            if counts and counts["reconciled"]:
                logger.info("Reconciled %s new predictions", counts["reconciled"])
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Scheduled reconciliation failed; will retry next interval")
