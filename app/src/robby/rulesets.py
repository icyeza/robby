"""Load a Robson rule-set YAML into rule_set and make it the active one."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from robson_engine import load_rule_set
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from robby import events
from robby.models import AppUser, RuleSet


class RuleSetConflict(Exception):
    """The version label is already loaded with different content."""


@dataclass(frozen=True)
class LoadOutcome:
    rule_set: RuleSet
    status: Literal["loaded", "activated", "unchanged"]


def load_rule_set_file(db: Session, path: Path, actor: AppUser | None = None) -> LoadOutcome:
    """Validate the file with robson_engine (structure and checksum; RuleSetError otherwise),
    store it if new, and make it the only active rule set. Appends one event unless nothing
    changed."""
    parsed = load_rule_set(path)
    definition: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    existing = db.scalar(select(RuleSet).where(RuleSet.version_label == parsed.version_label))
    if existing is not None and existing.checksum != parsed.checksum:
        raise RuleSetConflict(
            f"Rule set {parsed.version_label} is already loaded with a different checksum. "
            "Give the changed rule set a new version_label."
        )
    if existing is not None and existing.is_active:
        return LoadOutcome(existing, "unchanged")
    db.execute(update(RuleSet).where(RuleSet.is_active).values(is_active=False))
    status: Literal["loaded", "activated"]
    if existing is None:
        row = RuleSet(
            version_label=parsed.version_label,
            checksum=parsed.checksum,
            definition=definition,
            is_active=True,
        )
        db.add(row)
        status = "loaded"
    else:
        existing.is_active = True
        row = existing
        status = "activated"
    db.flush()
    events.append(
        db,
        actor,
        f"rule_set.{status}",
        ("rule_set", row.id),
        {"version_label": row.version_label, "checksum": row.checksum},
    )
    return LoadOutcome(row, status)
