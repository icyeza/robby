"""Offline Robson report table (spec §15.1).

Independent implementation: must never import the application's audit service code.
The Vogel reference comparison is added in Phase G.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

OVERALL = "ALL"
RESIDUAL = "residual"
REPORT_COLUMNS = [
    "facility",
    "row",
    "n",
    "pct_of_deliveries",
    "n_cs",
    "cs_rate",
    "abs_contribution",
    "rel_contribution",
]
REPORT_SUPPRESSION: dict[str, Any] = {
    "count_columns": ["n", "n_cs"],
    "linked": {
        "n": ["pct_of_deliveries", "n_cs", "cs_rate", "abs_contribution", "rel_contribution"],
        "n_cs": ["cs_rate", "abs_contribution", "rel_contribution"],
    },
    # n - n_cs (vaginal births in the group) must not be 1-4 either.
    "complements": {"n_cs": "n"},
}


def _ratio(num: float, den: float) -> float:
    return num / den if den else float("nan")


def _row_labels(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [
            str(int(g)) if status == "resolved" else RESIDUAL
            for g, status in zip(df["robson_group"], df["robson_status"], strict=True)
        ],
        index=df.index,
    )


def robson_report_table(df: pd.DataFrame, facility_col: str = "facility_id") -> pd.DataFrame:
    """Robson report table per facility and overall, on P_audit rows.

    For each Robson group (1-10) plus a residual row (partial and conflict records): group
    size (n, % of deliveries), group CS (n, rate), absolute contribution (group CS / all
    deliveries) and relative contribution (group CS / all CS). Unsuppressed; apply
    ``suppress_small_cells(table, **REPORT_SUPPRESSION)`` before any export.
    """
    missing = {facility_col, "robson_status", "robson_group", "cs"} - set(df.columns)
    if missing:
        raise ValueError(f"robson_report_table is missing columns: {sorted(missing)}")
    if df["cs"].isna().any():
        raise ValueError("robson_report_table expects P_audit rows (cs non-missing)")
    labels = _row_labels(df)
    scopes = [(OVERALL, df.index)] + [
        (str(f), sub.index) for f, sub in df.groupby(facility_col, sort=True)
    ]
    rows = []
    for facility, index in scopes:
        sub_cs = df.loc[index, "cs"].astype(int)
        sub_labels = labels.loc[index]
        total, total_cs = len(index), int(sub_cs.sum())
        for label in [*(str(g) for g in range(1, 11)), RESIDUAL]:
            in_row = sub_labels == label
            n, n_cs = int(in_row.sum()), int(sub_cs[in_row].sum())
            rows.append(
                {
                    "facility": facility,
                    "row": label,
                    "n": n,
                    "pct_of_deliveries": _ratio(n, total),
                    "n_cs": n_cs,
                    "cs_rate": _ratio(n_cs, n),
                    "abs_contribution": _ratio(n_cs, total),
                    "rel_contribution": _ratio(n_cs, total_cs),
                }
            )
    return pd.DataFrame(rows, columns=REPORT_COLUMNS)
