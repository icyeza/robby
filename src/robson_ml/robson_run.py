"""Apply the Robson engine to canonical frames, validate the result, draw the hand-check list."""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import pandas as pd

from robson_engine import INPUT_FIELDS, RobsonInputs, RuleSet, classify


class RobsonValidationError(AssertionError):
    """The dataset-level engine checks of spec §6.5 failed (message holds counts only)."""


def _clean(value: object) -> object:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _as_int(value: object) -> int | None:
    value = _clean(value)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("boolean where an integer was expected")
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ValueError("non-integer value where an integer was expected")


def _as_float(value: object) -> float | None:
    value = _clean(value)
    if value is None:
        return None
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return float(value)
    raise ValueError("non-numeric value where a number was expected")


def _as_str(value: object) -> str | None:
    value = _clean(value)
    return None if value is None else str(value)


def inputs_from_record(record: Mapping[str, object]) -> RobsonInputs:
    """Build engine inputs from one canonical record; NaN/NA/None all mean not recorded."""
    return RobsonInputs(
        parity=_as_int(record.get("parity")),
        previous_cs_count=_as_int(record.get("previous_cs_count")),
        plurality=_as_int(record.get("plurality")),
        fetal_presentation=_as_str(record.get("fetal_presentation")),
        gestational_age_weeks=_as_float(record.get("gestational_age_weeks")),
        onset_of_labour=_as_str(record.get("onset_of_labour")),
    )


def classify_frame(df: pd.DataFrame, rule_set: RuleSet) -> pd.DataFrame:
    """Return a copy of ``df`` with the engine's output columns appended (spec §5)."""
    records = cast("list[dict[str, object]]", df[list(INPUT_FIELDS)].to_dict("records"))
    results = [classify(inputs_from_record(rec), rule_set) for rec in records]
    out = df.copy()
    out["robson_group"] = pd.Series(
        pd.array([r.group for r in results], dtype="Int64"), index=df.index
    )
    out["robson_subgroup"] = pd.Series([r.subgroup for r in results], index=df.index, dtype=object)
    out["robson_status"] = pd.Series([r.status for r in results], index=df.index)
    out["robson_candidates"] = pd.Series(
        [sorted(r.candidates) for r in results], index=df.index, dtype=object
    )
    out["robson_resolving_fields"] = pd.Series(
        [";".join(r.resolving_fields) for r in results], index=df.index
    )
    out["robson_conflict_fields"] = pd.Series(
        [";".join(r.conflict_fields) for r in results], index=df.index
    )
    out["rule_set_version"] = pd.Series([r.rule_set_version for r in results], index=df.index)
    return out


@dataclass(frozen=True)
class RobsonValidation:
    """Dataset-level engine checks (spec §6.5, acceptance criterion 3)."""

    n_total: int
    status_counts: dict[str, int]
    group_counts: dict[int, int]
    n_complete_inputs: int
    n_complete_unresolved: int
    n_multi_group_resolved: int

    @property
    def reconciles(self) -> bool:
        """Group counts plus partial plus conflict equal the record count."""
        residual = self.status_counts.get("partial", 0) + self.status_counts.get("conflict", 0)
        return sum(self.group_counts.values()) + residual == self.n_total

    def assert_valid(self) -> None:
        """Raise if any record has several groups, a complete record fails to resolve, or
        the counts do not reconcile."""
        problems = []
        if self.n_multi_group_resolved:
            problems.append(f"{self.n_multi_group_resolved} resolved records with >1 candidate")
        if self.n_complete_unresolved:
            problems.append(f"{self.n_complete_unresolved} complete, consistent records unresolved")
        if not self.reconciles:
            problems.append("group and residual counts do not reconcile to the total")
        if problems:
            raise RobsonValidationError("; ".join(problems))


def validate_classification(df: pd.DataFrame) -> RobsonValidation:
    """Compute the §6.5 checks on a frame returned by :func:`classify_frame`."""
    status = df["robson_status"]
    resolved = status == "resolved"
    complete = df[list(INPUT_FIELDS)].notna().all(axis=1)
    n_candidates = df["robson_candidates"].map(len)
    groups = df.loc[resolved, "robson_group"].astype(int).value_counts().sort_index()
    return RobsonValidation(
        n_total=len(df),
        status_counts={str(k): int(v) for k, v in status.value_counts().items()},
        group_counts={int(str(k)): int(v) for k, v in groups.items()},
        n_complete_inputs=int(complete.sum()),
        n_complete_unresolved=int((complete & (status == "partial")).sum()),
        n_multi_group_resolved=int((resolved & (n_candidates != 1)).sum()),
    )
