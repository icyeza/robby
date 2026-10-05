"""Value types used by the Robson classification engine."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Literal

Status = Literal["resolved", "partial", "conflict"]
Outcome = Literal["true", "false", "unknown"]
InputValue = int | float | str | None
CoarseValue = frozenset[str] | tuple[float, float]

PRESENTATIONS: frozenset[str] = frozenset({"cephalic", "breech", "transverse", "oblique"})
# Coarse presentation codes: each stands for a set of precise presentations.
# Rule sets may only reference precise values; inputs may carry either.
COARSE_PRESENTATIONS: Mapping[str, frozenset[str]] = MappingProxyType(
    {"non_cephalic": frozenset({"breech", "transverse", "oblique"})}
)
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


def _validated_ga_range(raw: object) -> tuple[float, float]:
    """Check a GA interval and normalise it to a ``(lower, upper)`` float tuple."""
    if not isinstance(raw, tuple | list) or len(raw) != 2:
        raise ValueError("gestational_age_range must be a (lower, upper) pair")
    bounds: list[float] = []
    for bound in raw:
        if isinstance(bound, bool) or not isinstance(bound, Real):
            raise ValueError("gestational_age_range bounds must be numbers, not booleans or text")
        number = float(bound)
        if not (math.isfinite(number) and GA_MIN_WEEKS <= number <= GA_MAX_WEEKS):
            raise ValueError("gestational_age_range bounds must be finite and within 20.0-45.0")
        bounds.append(number)
    lower, upper = bounds
    if lower > upper:
        raise ValueError("gestational_age_range lower bound must not exceed its upper bound")
    return (lower, upper)


@dataclass(frozen=True)
class RobsonInputs:
    """The six Robson inputs for one admission; ``None`` means not recorded.

    Two inputs may be recorded *coarsely*:

    * ``fetal_presentation`` may be a key of :data:`COARSE_PRESENTATIONS` (``"non_cephalic"``)
      instead of a precise value; it then stands for every presentation in that set.
    * ``gestational_age_range`` is an optional closed interval ``(lower, upper)`` in weeks,
      used only when ``gestational_age_weeks`` is ``None``. When the exact value is recorded
      the range is ignored (exact wins), even if the two disagree.

    ``gestational_age_range`` is a companion of ``gestational_age_weeks``, not a seventh input,
    so it is not in :data:`INPUT_FIELDS`. It is normalised to a tuple of floats.

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
    gestational_age_range: tuple[float, float] | None = None

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
        presentation = self.fetal_presentation
        if (
            presentation is not None
            and presentation not in PRESENTATIONS
            and presentation not in COARSE_PRESENTATIONS
        ):
            raise ValueError("fetal_presentation is not an allowed category")
        if self.onset_of_labour is not None and self.onset_of_labour not in ONSETS:
            raise ValueError("onset_of_labour is not an allowed category")
        ga = self.gestational_age_weeks
        if ga is not None and (math.isnan(ga) or not GA_MIN_WEEKS <= ga <= GA_MAX_WEEKS):
            raise ValueError("gestational_age_weeks must be within 20.0-45.0")
        if self.gestational_age_range is not None:
            normalised = _validated_ga_range(self.gestational_age_range)
            object.__setattr__(self, "gestational_age_range", normalised)

    def value(self, field: str) -> InputValue:
        """Return the recorded value of one input field (``None`` if not recorded).

        A coarse presentation is returned as its code (``"non_cephalic"``); a GA known only as
        a range returns ``None`` here. Use :meth:`coarse_value` for the values they allow.
        """
        if field not in INPUT_FIELDS:
            raise KeyError(f"unknown Robson input field: {field}")
        result: InputValue = getattr(self, field)
        return result

    def coarse_value(self, field: str) -> CoarseValue | None:
        """The values a coarsely recorded field allows, or ``None`` if it is precise or missing.

        Returns the set of precise presentations for a coarse ``fetal_presentation``, and the
        ``(lower, upper)`` interval for ``gestational_age_weeks`` when only a range is recorded.
        """
        if field not in INPUT_FIELDS:
            raise KeyError(f"unknown Robson input field: {field}")
        if field == "fetal_presentation" and self.fetal_presentation is not None:
            return COARSE_PRESENTATIONS.get(self.fetal_presentation)
        if field == "gestational_age_weeks" and self.gestational_age_weeks is None:
            return self.gestational_age_range
        return None

    def coarse_fields(self) -> list[str]:
        """Input fields recorded only coarsely, in canonical order."""
        return [f for f in INPUT_FIELDS if self.coarse_value(f) is not None]

    def missing_fields(self) -> list[str]:
        """Input fields not recorded at all, in canonical order.

        A coarsely recorded field counts as recorded, so it is not listed here (see
        :meth:`coarse_fields`); it can still appear in a result's ``resolving_fields``.
        """
        return [f for f in INPUT_FIELDS if self.value(f) is None and self.coarse_value(f) is None]


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
    """Engine output. Sequences are tuples so the result is immutable."""

    status: Status
    group: int | None
    subgroup: str | None
    candidates: frozenset[int]
    resolving_fields: tuple[str, ...]
    conflict_fields: tuple[str, ...]
    trace: tuple[ConditionTrace, ...]
    rule_set_version: str
