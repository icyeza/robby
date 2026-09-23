"""Analysis populations (spec §4.4). P_pred and P_pred_sens are added in Phase C."""

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
