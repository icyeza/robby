"""Loading, validating and checksumming Robson rule sets stored as YAML data (spec §6.2)."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from robson_engine.models import (
    INPUT_FIELDS,
    ONSETS,
    PRESENTATIONS,
    InputValue,
    Outcome,
    RobsonInputs,
)

OPS: frozenset[str] = frozenset({"eq", "ge", "lt", "in"})
DEFAULT_RULE_SET = "robson_v1.0.yaml"
NUMERIC_FIELDS: frozenset[str] = frozenset(
    {"parity", "previous_cs_count", "plurality", "gestational_age_weeks"}
)


class RuleSetError(ValueError):
    """The rule set is malformed or its checksum does not match its content."""


@dataclass(frozen=True)
class Condition:
    """A single comparison of one Robson input against a constant."""

    field: str
    op: str
    value: InputValue | tuple[InputValue, ...]

    def evaluate(self, inputs: RobsonInputs) -> Outcome:
        """Return ``unknown`` for a missing input, otherwise ``true`` or ``false``."""
        actual = inputs.value(self.field)
        if actual is None:
            return "unknown"
        if self.op == "eq":
            ok = actual == self.value
        elif self.op == "in":
            if not isinstance(self.value, tuple):
                raise RuleSetError(f"op 'in' on {self.field} needs a list value")
            ok = actual in self.value
        else:
            if not isinstance(actual, int | float) or not isinstance(self.value, int | float):
                raise RuleSetError(f"op {self.op} on {self.field} needs numeric operands")
            ok = actual >= self.value if self.op == "ge" else actual < self.value
        return "true" if ok else "false"


@dataclass(frozen=True)
class Subgroup:
    """A named refinement of a group (for example 2a/2b), resolved after the group."""

    label: str
    conditions: tuple[Condition, ...]


@dataclass(frozen=True)
class GroupRule:
    """One Robson group: a conjunction of conditions plus optional subgroups."""

    group: int
    name: str
    conditions: tuple[Condition, ...]
    subgroups: tuple[Subgroup, ...]


@dataclass(frozen=True)
class ConsistencyRule:
    """A combination of recorded inputs that is internally contradictory."""

    name: str
    when: tuple[Condition, ...]


@dataclass(frozen=True)
class RuleSet:
    """A validated, checksummed rule set."""

    version_label: str
    checksum: str
    groups: tuple[GroupRule, ...]
    consistency: tuple[ConsistencyRule, ...]


def compute_checksum(data: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON form of the rule set, excluding the ``checksum`` key."""
    payload = {k: v for k, v in data.items() if k != "checksum"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _is_plain_number(value: Any) -> bool:
    """True for an ``int`` or ``float`` that is not a ``bool``."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def _check_domain(field: str, value: Any) -> bool:
    """True if ``value`` is an allowed member of ``field``'s category domain."""
    if field == "fetal_presentation":
        return value in PRESENTATIONS
    if field == "onset_of_labour":
        return value in ONSETS
    return True


def _validate_condition_value(field: str, op: str, value: Any) -> None:
    if op in ("eq", "in") and field in ("fetal_presentation", "onset_of_labour"):
        values = value if op == "in" else (value,)
        if not all(_check_domain(field, v) for v in values):
            raise RuleSetError(f"invalid value for {field} in condition")
    if op in ("ge", "lt"):
        if not _is_plain_number(value):
            raise RuleSetError(f"invalid value for {field} in condition")
    elif op == "eq" and field in NUMERIC_FIELDS and not _is_plain_number(value):
        raise RuleSetError(f"invalid value for {field} in condition")


def _condition(raw: dict[str, Any]) -> Condition:
    field, op, value = raw.get("field"), raw.get("op"), raw.get("value")
    if field not in INPUT_FIELDS:
        raise RuleSetError(f"unknown field in condition: {field}")
    if op not in OPS:
        raise RuleSetError(f"unknown op in condition: {op}")
    if op == "in":
        if not isinstance(value, list):
            raise RuleSetError(f"op 'in' on {field} needs a list value")
        _validate_condition_value(str(field), str(op), value)
        return Condition(str(field), str(op), tuple(value))
    _validate_condition_value(str(field), str(op), value)
    return Condition(str(field), str(op), value)


def parse_rule_set(data: dict[str, Any]) -> RuleSet:
    """Validate a parsed rule set mapping and build a :class:`RuleSet`.

    Raises:
        RuleSetError: on a checksum mismatch, unknown fields or ops, or if groups 1-10 are
            not each defined exactly once.
    """
    actual = compute_checksum(data)
    if data.get("checksum") != actual:
        raise RuleSetError("rule set checksum mismatch: content changed without re-stamping")
    try:
        groups = tuple(
            GroupRule(
                group=int(g["group"]),
                name=str(g["name"]),
                conditions=tuple(_condition(c) for c in g["conditions"]),
                subgroups=tuple(
                    Subgroup(str(s["label"]), tuple(_condition(c) for c in s["conditions"]))
                    for s in g.get("subgroups", [])
                ),
            )
            for g in data["groups"]
        )
        if sorted(g.group for g in groups) != list(range(1, 11)):
            raise RuleSetError("rule set must define groups 1-10 exactly once")
        consistency = tuple(
            ConsistencyRule(str(r["name"]), tuple(_condition(c) for c in r["when"]))
            for r in data.get("consistency", [])
        )
        version_label = str(data["version_label"])
    except RuleSetError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise RuleSetError(f"malformed rule set structure: {exc}") from None
    return RuleSet(version_label, actual, groups, consistency)


def load_rule_set(path: Path | None = None) -> RuleSet:
    """Load a rule set file; with no path, load the packaged default (v1.0)."""
    if path is None:
        text = (
            resources.files("robson_engine.rules")
            .joinpath(DEFAULT_RULE_SET)
            .read_text(encoding="utf-8")
        )
    else:
        text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise RuleSetError("rule set file must contain a mapping")
    return parse_rule_set(data)


def stamp(path: Path) -> str:
    """Recompute a rule set file's checksum and write it into its ``checksum:`` line."""
    text = path.read_text(encoding="utf-8")
    checksum = compute_checksum(yaml.safe_load(text))
    new_text, n = re.subn(r"(?m)^checksum:.*$", f'checksum: "{checksum}"', text)
    if n != 1:
        raise RuleSetError("rule set file needs exactly one top-level 'checksum:' line")
    path.write_text(new_text, encoding="utf-8")
    return checksum


if __name__ == "__main__":
    print(stamp(Path(sys.argv[1])))
