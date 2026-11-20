"""Versions of the two policy documents and users' acceptance of them.

A user must have accepted the current pair of versions before using anything but the public
pages. Changing a version constant (and the page text) asks everyone again at next sign-in.
Acceptances are insert-only rows (migration 0010)."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from robby import events
from robby.models import AppUser, PolicyAcceptance

PRIVACY_VERSION = "1.0"
PRIVACY_DATE = date(2026, 11, 20)
TERMS_VERSION = "1.0"
TERMS_DATE = date(2026, 11, 20)


def has_accepted(db: Session, user: AppUser) -> bool:
    """Whether the user accepted the current version of both documents."""
    return (
        db.scalar(
            select(PolicyAcceptance.id)
            .where(
                PolicyAcceptance.user_id == user.id,
                PolicyAcceptance.privacy_version == PRIVACY_VERSION,
                PolicyAcceptance.terms_version == TERMS_VERSION,
            )
            .limit(1)
        )
        is not None
    )


def accept(db: Session, user: AppUser, now: datetime) -> None:
    """Record the acceptance (once per pair of versions) and log it. Caller commits."""
    if has_accepted(db, user):
        return
    row = PolicyAcceptance(
        user_id=user.id,
        privacy_version=PRIVACY_VERSION,
        terms_version=TERMS_VERSION,
        accepted_at=now,
    )
    db.add(row)
    db.flush()
    events.append(
        db,
        user,
        "policy.accepted",
        entity=("policy_acceptance", row.id),
        payload={"privacy_version": PRIVACY_VERSION, "terms_version": TERMS_VERSION},
    )
