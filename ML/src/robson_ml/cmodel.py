"""WHO C-Model expected CS probability.

Coefficients come only from ``data/reference/cmodel_v1.yaml`` (:func:`references.load_cmodel`),
transcribed from Souza et al. (2016). They are never estimated from local data, approximated
or invented, and there is no fallback when the file is absent. If any variable the model
needs has no column in the data, or the column is entirely missing, the C-Model is **not
applicable**: the absent variables are listed and nothing is computed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from robson_ml.references import CModelReference, CModelTerm

NOT_APPLICABLE = "not applicable"


class CModelNotApplicableError(ValueError):
    """A variable the C-Model needs is absent from the data."""


@dataclass(frozen=True)
class CModelApplicability:
    """Whether the C-Model can be applied to a frame, and why not."""

    applicable: bool
    absent_variables: list[str] = field(default_factory=list)

    def statement(self) -> str:
        if self.applicable:
            return "WHO C-Model applicable: every required variable is present in the data."
        return (
            f"WHO C-Model {NOT_APPLICABLE}: required variables absent from the data: "
            f"{', '.join(self.absent_variables)}. It is not approximated."
        )


def cmodel_applicability(frame: pd.DataFrame, reference: CModelReference) -> CModelApplicability:
    """Required C-Model variables with no column, or an entirely missing one, in ``frame``."""
    absent = []
    for variable, column in reference.variables.items():
        if column is None or column not in frame.columns or frame[column].notna().sum() == 0:
            absent.append(variable)
    return CModelApplicability(not absent, sorted(absent))


def _term_value(series: pd.Series, term: CModelTerm) -> pd.Series:
    """The term's contribution per row (NaN where the variable is missing)."""
    missing = series.isna()
    if term.kind == "linear":
        numeric = pd.to_numeric(series, errors="coerce").astype(float)
        return term.coefficient * (numeric - term.centre)
    if term.equals is not None:
        hit = pd.Series(False, index=series.index)
        for value in term.equals:
            if isinstance(value, str):
                hit |= series.astype(object).astype(str).eq(value) & ~missing
            else:
                numeric = pd.to_numeric(series, errors="coerce")
                hit |= numeric.eq(float(value)).fillna(False).astype(bool)  # type: ignore[arg-type]
    else:
        numeric = pd.to_numeric(series, errors="coerce").astype(float)
        low = -np.inf if term.lower is None else term.lower
        high = np.inf if term.upper is None else term.upper
        hit = ((numeric >= low) & (numeric < high)).fillna(False).astype(bool)
    return hit.astype(float).where(~missing) * term.coefficient


def cmodel_probability(frame: pd.DataFrame, reference: CModelReference) -> pd.Series:
    """Per-woman expected CS probability; NaN for a row missing any required value.

    Raises:
        CModelNotApplicableError: when a required variable is absent (never approximated).
    """
    applicability = cmodel_applicability(frame, reference)
    if not applicability.applicable:
        raise CModelNotApplicableError(applicability.statement())
    linear = pd.Series(reference.intercept, index=frame.index, dtype=float)
    for term in reference.terms:
        column = reference.variables[term.variable]
        assert column is not None  # guaranteed by the applicability check
        linear = linear + _term_value(frame[column], term)
    return pd.Series(1.0 / (1.0 + np.exp(-linear.to_numpy())), index=frame.index)


def facility_expected(
    frame: pd.DataFrame,
    reference: CModelReference,
    facility_col: str = "facility_id",
    overall: str = "ALL",
) -> pd.DataFrame:
    """Per facility (and overall): rows, rows scored (all variables recorded), CS among
    them, observed rate and the C-Model expected rate (mean per-woman probability)."""
    probability = cmodel_probability(frame, reference)
    scored = probability.notna()
    rows = []
    scopes = [(overall, frame.index)] + [
        (str(f), sub.index) for f, sub in frame.groupby(facility_col, sort=True)
    ]
    for name, index in scopes:
        mask = scored.loc[index]
        cs = frame.loc[index, "cs"].astype(float)[mask]
        n_scored = int(mask.sum())
        rows.append(
            {
                "facility": name,
                "n": len(index),
                "n_scored": n_scored,
                "n_cs_scored": int(cs.sum()),
                "observed_rate_scored": float(cs.mean()) if n_scored else float("nan"),
                "cmodel_expected_rate": (
                    float(probability.loc[index][mask].mean()) if n_scored else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)
