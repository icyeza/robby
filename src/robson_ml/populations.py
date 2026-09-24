"""Analysis populations (spec §4.4)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ExclusionLog:
    """How many rows a population filter removed, and why."""

    reason: str
    n_excluded: int
    n_remaining: int


def audit_population(df: pd.DataFrame) -> tuple[pd.DataFrame, ExclusionLog]:
    """P_audit: all rows with a non-missing outcome (spec §4.2, §4.4)."""
    keep = df["cs"].notna()
    log = ExclusionLog(
        "missing mode of delivery (cs undefined)", int((~keep).sum()), int(keep.sum())
    )
    return df[keep].copy(), log


def prediction_population(df: pd.DataFrame) -> tuple[pd.DataFrame, list[ExclusionLog]]:
    """P_pred: ``P_audit`` where ``onset_of_labour`` in {spontaneous, induced} (spec §4.4).

    ``df`` is assumed to already be ``P_audit`` (a non-missing outcome). Pre-labour CS rows
    encode the outcome directly in their onset and are excluded, split out by reason so
    reports can state the number of high-risk (CS) admissions this removes (spec §4.4
    reporting obligation): planned pre-labour CS, emergency pre-labour CS, pre-labour CS of
    unknown type, and a missing onset altogether.
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
    """P_pred_sens: ``P_pred`` union emergency pre-labour CS rows (spec §4.4).

    Onset is not recorded as "what was known at admission" anywhere in the export, so for
    the added emergency-pre-labour-CS rows ``onset_of_labour`` is set to missing (NA) rather
    than guessed; this is a deliberate simplification of the spec's wording, since the data
    cannot support recoding to a specific admission-time onset. ``onset_of_labour`` must
    therefore be dropped from any feature set built on ``P_pred_sens`` in the analysis (it
    would otherwise be missing for exactly the added rows, which is itself leakage).
    """
    pred, pred_logs = prediction_population(df)
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
