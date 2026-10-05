"""Analysis populations (population definition v1.3).

``P_pred`` (v1.3) is ``P_audit`` minus the admissions with a CS already planned at admission:
CS typed elective/planned, and "Planned C-section"-onset admissions that delivered vaginally.
The onset field is often coded retrospectively ("Planned C-section" onset ~ "had a CS"), so
it no longer defines the population. The v1.2 onset-based definition is kept as the
sensitivity population ``P_pred_onset_coded`` (:func:`onset_coded_population`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

# Tagged on every harness run; comparisons use only runs of the current version.
POPULATION_VERSION = "v1.3"
P_PRED = "P_pred"
P_PRED_ONSET_CODED = "P_pred_onset_coded"

REASON_MISSING_OUTCOME = "missing mode of delivery (cs undefined)"
REASON_PLANNED_CS = "planned CS (CS typed elective/planned)"
REASON_PLANNED_ONSET_VAGINAL = "planned-CS onset with vaginal birth"


@dataclass(frozen=True)
class ExclusionLog:
    """How many rows a population filter removed, and why."""

    reason: str
    n_excluded: int
    n_remaining: int


def audit_population(df: pd.DataFrame) -> tuple[pd.DataFrame, ExclusionLog]:
    """P_audit: all rows with a non-missing outcome."""
    keep = df["cs"].notna()
    log = ExclusionLog(REASON_MISSING_OUTCOME, int((~keep).sum()), int(keep.sum()))
    return df[keep].copy(), log


def planned_cs_mask(df: pd.DataFrame) -> pd.Series:
    """CS typed elective/planned (the CS-type field is filled for all CS, not only pre-labour)."""
    return ((df["cs"] == 1) & (df["prelabour_cs_type"] == "planned")).fillna(False).astype(bool)


def planned_onset_vaginal_mask(df: pd.DataFrame) -> pd.Series:
    """Onset "Planned C-section" with a vaginal birth (a CS planned at admission, not done)."""
    mask = (df["onset_of_labour"] == "prelabour_cs") & (df["cs"] == 0)
    return mask.fillna(False).astype(bool)


def untyped_cs_mask(df: pd.DataFrame) -> pd.Series:
    """CS with no recorded CS type: kept in ``P_pred`` v1.3 and counted."""
    return ((df["cs"] == 1) & df["prelabour_cs_type"].isna()).fillna(False).astype(bool)


def prediction_population(df: pd.DataFrame) -> tuple[pd.DataFrame, list[ExclusionLog]]:
    """P_pred (v1.3): ``P_audit`` minus admissions with a CS planned at admission.

    Removes rows with a missing outcome, CS typed planned (whatever their onset) and
    "Planned C-section"-onset admissions that delivered vaginally. CS with no recorded type
    are kept; their count is logged. ``df`` may be the canonical frame or ``P_audit`` (the
    outcome filter is then a no-op). Mirrors deployment: a planned CS receives a Robson group
    but no readiness score.

    Returns the population and three logs: missing outcome, planned CS, planned-onset
    vaginal birth (each ``n_remaining`` after that step's population is final).
    """
    audit, audit_log = audit_population(df)
    planned = planned_cs_mask(audit)
    vaginal = planned_onset_vaginal_mask(audit)
    keep = ~(planned | vaginal)
    pred = audit[keep].copy()
    n_untyped = int(untyped_cs_mask(pred).sum())
    logger.info("P_pred %s: %d CS with no recorded type kept", POPULATION_VERSION, n_untyped)
    logs = [
        audit_log,
        ExclusionLog(REASON_PLANNED_CS, int(planned.sum()), len(pred)),
        ExclusionLog(REASON_PLANNED_ONSET_VAGINAL, int(vaginal.sum()), len(pred)),
    ]
    return pred, logs


def onset_coded_population(df: pd.DataFrame) -> tuple[pd.DataFrame, list[ExclusionLog]]:
    """P_pred_onset_coded: the v1.2 ``P_pred``, ``onset_of_labour`` in {spontaneous, induced}.

    Sensitivity population only: onset is often coded retrospectively,
    so this definition silently drops most in-labour CS. ``df`` is assumed to already be
    ``P_audit``. Excluded rows are split out by reason: planned pre-labour CS, emergency
    pre-labour CS, pre-labour CS of unknown type, and a missing onset altogether.
    """
    onset = df["onset_of_labour"]
    prelabour = onset == "prelabour_cs"
    cs_type = df["prelabour_cs_type"]
    planned = prelabour & (cs_type == "planned")
    emergency = prelabour & (cs_type == "emergency")
    unknown_type = prelabour & cs_type.isna()
    missing_onset = onset.isna()
    keep = onset.isin(["spontaneous", "induced"])

    logs = [
        ExclusionLog("onset prelabour_cs planned", int(planned.sum()), int(keep.sum())),
        ExclusionLog("onset prelabour_cs emergency", int(emergency.sum()), int(keep.sum())),
        ExclusionLog("onset prelabour_cs type unknown", int(unknown_type.sum()), int(keep.sum())),
        ExclusionLog("onset missing", int(missing_onset.sum()), int(keep.sum())),
    ]
    return df[keep].copy(), logs


def sensitivity_population(df: pd.DataFrame) -> tuple[pd.DataFrame, list[ExclusionLog]]:
    """P_pred_sens: ``P_pred_onset_coded`` union emergency pre-labour CS rows.

    Superseded by the v1.3 ``P_pred``, which already keeps emergency pre-labour CS; kept for
    existing callers and built on the v1.2 onset-coded population it was defined against.

    Onset is not recorded as "what was known at admission" anywhere in the export, so for
    the added emergency-pre-labour-CS rows ``onset_of_labour`` is set to missing (NA) rather
    than guessed; this is a deliberate simplification of the original definition, since the data
    cannot support recoding to a specific admission-time onset. ``onset_of_labour`` must
    therefore be dropped from any feature set built on ``P_pred_sens`` in the analysis (it
    would otherwise be missing for exactly the added rows, which is itself leakage).
    """
    pred, pred_logs = onset_coded_population(df)
    onset = df["onset_of_labour"]
    prelabour = onset == "prelabour_cs"
    emergency = prelabour & (df["prelabour_cs_type"] == "emergency")
    added = df[emergency].copy()
    added["onset_of_labour"] = added["onset_of_labour"].mask(pd.Series(True, index=added.index))
    combined = pd.concat([pred, added], ignore_index=True)
    logs = [
        *pred_logs,
        ExclusionLog(
            "emergency pre-labour CS added back for sensitivity analysis",
            -int(emergency.sum()),
            len(combined),
        ),
    ]
    return combined, logs
