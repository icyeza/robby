"""Apply the Robson engine to canonical frames, validate the result, draw the hand-check list."""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from robson_engine import COARSE_PRESENTATIONS, INPUT_FIELDS, RobsonInputs, RuleSet, classify
from robson_ml.schema import GA_BAND_FIELDS


class RobsonValidationError(AssertionError):
    """The dataset-level engine checks failed (message holds counts only)."""


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
    """Build engine inputs from one canonical record; NaN/NA/None all mean not recorded.

    When the exact gestational age is missing and both GA band bounds are recorded, the band
    is passed as ``gestational_age_range`` (coarse input). A presentation of
    ``non_cephalic`` is passed through as the engine's coarse code.
    """
    ga = _as_float(record.get("gestational_age_weeks"))
    ga_range = None
    if ga is None:
        lower = _as_float(record.get("ga_band_lower"))
        upper = _as_float(record.get("ga_band_upper"))
        if lower is not None and upper is not None:
            ga_range = (lower, upper)
    return RobsonInputs(
        parity=_as_int(record.get("parity")),
        previous_cs_count=_as_int(record.get("previous_cs_count")),
        plurality=_as_int(record.get("plurality")),
        fetal_presentation=_as_str(record.get("fetal_presentation")),
        gestational_age_weeks=ga,
        onset_of_labour=_as_str(record.get("onset_of_labour")),
        gestational_age_range=ga_range,
    )


def classify_frame(df: pd.DataFrame, rule_set: RuleSet) -> pd.DataFrame:
    """Return a copy of ``df`` with the engine's output columns appended.

    The six inputs are required; the GA band columns are used when present.
    """
    columns = [*INPUT_FIELDS, *(c for c in GA_BAND_FIELDS if c in df.columns)]
    records = cast("list[dict[str, object]]", df[columns].to_dict("records"))
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
    """Dataset-level engine checks.

    "Complete inputs" means all six inputs recorded *precisely*: a presentation type (not a
    coarse code such as ``non_cephalic``) and an exact gestational age (not only a band). A
    record with a coarse input may legitimately stay partial, so only
    complete, precise records must resolve.

    Attributes:
        n_complete_inputs: records with all six inputs recorded precisely.
        n_complete_unresolved: of those, records the engine left partial.
        n_coarse_inputs: records with at least one coarse input (a coarse presentation, or
            GA known only as a band).
    """

    n_total: int
    status_counts: dict[str, int]
    group_counts: dict[int, int]
    n_complete_inputs: int
    n_complete_unresolved: int
    n_multi_group_resolved: int
    n_coarse_inputs: int

    @property
    def reconciles(self) -> bool:
        """Group counts plus partial plus conflict equal the record count."""
        residual = self.status_counts.get("partial", 0) + self.status_counts.get("conflict", 0)
        return sum(self.group_counts.values()) + residual == self.n_total

    def assert_valid(self) -> None:
        """Raise if any record has several groups, a complete (precise, consistent) record
        fails to resolve, or the counts do not reconcile."""
        problems = []
        if self.n_multi_group_resolved:
            problems.append(f"{self.n_multi_group_resolved} resolved records with >1 candidate")
        if self.n_complete_unresolved:
            problems.append(f"{self.n_complete_unresolved} complete, consistent records unresolved")
        if not self.reconciles:
            problems.append("group and residual counts do not reconcile to the total")
        if problems:
            raise RobsonValidationError("; ".join(problems))


def _coarse_masks(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(coarse presentation, GA known only as a band) per record."""
    presentation = df["fetal_presentation"].isin(list(COARSE_PRESENTATIONS))
    if set(GA_BAND_FIELDS) <= set(df.columns):
        band = df[list(GA_BAND_FIELDS)].notna().all(axis=1)
        band_only = df["gestational_age_weeks"].isna() & band
    else:
        band_only = pd.Series(False, index=df.index)
    return presentation, band_only


def validate_classification(df: pd.DataFrame) -> RobsonValidation:
    """Compute the dataset-level checks on a frame returned by :func:`classify_frame`."""
    status = df["robson_status"]
    resolved = status == "resolved"
    coarse_presentation, band_only = _coarse_masks(df)
    # All six recorded, and precisely: exact GA (notna) and a presentation type.
    complete = df[list(INPUT_FIELDS)].notna().all(axis=1) & ~coarse_presentation
    n_candidates = df["robson_candidates"].map(len)
    groups = df.loc[resolved, "robson_group"].astype(int).value_counts().sort_index()
    return RobsonValidation(
        n_total=len(df),
        status_counts={str(k): int(v) for k, v in status.value_counts().items()},
        group_counts={int(str(k)): int(v) for k, v in groups.items()},
        n_complete_inputs=int(complete.sum()),
        n_complete_unresolved=int((complete & (status == "partial")).sum()),
        n_multi_group_resolved=int((resolved & (n_candidates != 1)).sum()),
        n_coarse_inputs=int((coarse_presentation | band_only).sum()),
    )


HANDCHECK_PER_GROUP = 15
HANDCHECK_BOUNDARY_N = 50
BOUNDARY_GA_LOW = 36.0
BOUNDARY_GA_HIGH = 37.0 + 6.0 / 7.0
HANDCHECK_COLUMNS = [
    "admission_id",
    "facility_id",
    *INPUT_FIELDS,
    *GA_BAND_FIELDS,
    "robson_status",
    "robson_group",
    "robson_subgroup",
    "robson_candidates",
    "robson_resolving_fields",
    "robson_conflict_fields",
    "rule_set_version",
]


def handcheck_sample(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Draw the list of records to classify by hand. The code never adjudicates it.

    Strata: up to 15 random resolved records per group; up to 50 records with GA between
    36+0 and 37+6 weeks; every conflict record. A record in several strata appears once,
    with all its strata listed. Blank columns are added for the manual review.
    """
    rng = np.random.default_rng(seed)
    positions = np.arange(len(df))
    strata: dict[int, list[str]] = {}

    def add(chosen: np.ndarray, label: str) -> None:
        for pos in sorted(int(p) for p in chosen):
            strata.setdefault(pos, []).append(label)

    def draw(pool: np.ndarray, k: int) -> np.ndarray:
        return pool if len(pool) <= k else rng.choice(pool, size=k, replace=False)

    resolved = (df["robson_status"] == "resolved").to_numpy()
    for group in range(1, 11):
        in_group = (df["robson_group"] == group).fillna(False).to_numpy(dtype=bool)
        add(draw(positions[resolved & in_group], HANDCHECK_PER_GROUP), f"group_{group}")
    ga = df["gestational_age_weeks"]
    boundary = ga.between(BOUNDARY_GA_LOW, BOUNDARY_GA_HIGH).fillna(False).to_numpy(dtype=bool)
    add(draw(positions[boundary], HANDCHECK_BOUNDARY_N), "ga_boundary")
    add(positions[(df["robson_status"] == "conflict").to_numpy()], "conflict")

    order = sorted(strata)
    out = df.iloc[order][HANDCHECK_COLUMNS].copy()
    out["robson_candidates"] = out["robson_candidates"].map(lambda c: ";".join(map(str, c)))
    out.insert(0, "strata", [";".join(strata[p]) for p in order])
    out["manual_group"] = pd.NA
    out["manual_subgroup"] = pd.NA
    out["reviewer_note"] = pd.NA
    return out.reset_index(drop=True)
