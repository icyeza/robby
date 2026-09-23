"""Value types used by the Robson classification engine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

Status = Literal["resolved", "partial", "conflict"]
Outcome = Literal["true", "false", "unknown"]
InputValue = int | float | str | None

PRESENTATIONS: frozenset[str] = frozenset({"cephalic", "breech", "transverse", "oblique"})
ONSETS: frozenset[str] = frozenset({"spontaneous", "induced", "prelabour_cs"})
INPUT_FIELDS: tuple[str, ...] = (
    "parity",
    "previous_cs_count",
    "plurality",
    "fetal_presentation",
    "gestational_age_weeks",
    "onset_of_labour",
)
GA_MIN_WEEKS = 20.0
GA_MAX_WEEKS = 45.0


@dataclass(frozen=True)
class RobsonInputs:
    """The six Robson inputs for one admission; ``None`` means not recorded.

    Raises:
        ValueError: if a recorded value is outside its allowed domain. Messages name the
            field only, never the value, so they are safe to log.
    """

    parity: int | None = None
    previous_cs_count: int | None = None
    plurality: int | None = None
    fetal_presentation: str | None = None
    gestational_age_weeks: float | None = None
    onset_of_labour: str | None = None

    def __post_init__(self) -> None:
        if self.parity is not None and isinstance(self.parity, bool):
            raise ValueError("parity must be an integer, not a boolean")
        if self.previous_cs_count is not None and isinstance(self.previous_cs_count, bool):
            raise ValueError("previous_cs_count must be an integer, not a boolean")
        if self.plurality is not None and isinstance(self.plurality, bool):
            raise ValueError("plurality must be an integer, not a boolean")
        if self.gestational_age_weeks is not None and isinstance(self.gestational_age_weeks, bool):
            raise ValueError("gestational_age_weeks must be a number, not a boolean")
        if self.parity is not None and self.parity < 0:
            raise ValueError("parity must be >= 0")
        if self.previous_cs_count is not None and self.previous_cs_count < 0:
            raise ValueError("previous_cs_count must be >= 0")
        if self.plurality is not None and self.plurality < 1:
            raise ValueError("plurality must be >= 1")
        if self.fetal_presentation is not None and self.fetal_presentation not in PRESENTATIONS:
            raise ValueError("fetal_presentation is not an allowed category")
        if self.onset_of_labour is not None and self.onset_of_labour not in ONSETS:
            raise ValueError("onset_of_labour is not an allowed category")
        ga = self.gestational_age_weeks
        if ga is not None and (math.isnan(ga) or not GA_MIN_WEEKS <= ga <= GA_MAX_WEEKS):
            raise ValueError("gestational_age_weeks must be within 20.0-45.0")

    def value(self, field: str) -> InputValue:
        """Return the recorded value of one input field (``None`` if not recorded)."""
        if field not in INPUT_FIELDS:
            raise KeyError(f"unknown Robson input field: {field}")
        result: InputValue = getattr(self, field)
        return result

    def missing_fields(self) -> list[str]:
        """Input fields that are not recorded, in canonical order."""
        return [f for f in INPUT_FIELDS if self.value(f) is None]


@dataclass(frozen=True)
class ConditionTrace:
    """One evaluated condition, with the group (and subgroup) it belongs to and its outcome."""

    group: int
    subgroup: str | None
    field: str
    op: str
    expected: object
    outcome: Outcome


@dataclass(frozen=True)
class ClassificationResult:
    """Engine output (spec §6.1). Sequences are tuples so the result is immutable."""

    status: Status
    group: int | None
    subgroup: str | None
    candidates: frozenset[int]
    resolving_fields: tuple[str, ...]
    conflict_fields: tuple[str, ...]
    trace: tuple[ConditionTrace, ...]
    rule_set_version: str
