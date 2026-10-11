"""Clock helpers. Application code compares times in Python, never with SQL now()."""

from __future__ import annotations

from datetime import UTC, datetime


def utcnow() -> datetime:
    """The current time as an aware UTC datetime."""
    return datetime.now(UTC)
