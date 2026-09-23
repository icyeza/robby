"""Offline Robson report table (spec §15.1).

Independent implementation: must never import the application's audit service code.
The Vogel reference comparison is added in Phase G.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from robson_ml.privacy import SumRelation, suppress_table

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
# Published to 2 decimals only (see suppress_report_table).
REPORT_ROUNDED = ("pct_of_deliveries", "abs_contribution", "rel_contribution")
REPORT_DECIMALS = 2


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
    :func:`suppress_report_table` before any export.
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


def report_groups(table: pd.DataFrame) -> list[SumRelation]:
    """The sums a reader of the report table can form, as row-label groups.

    Each facility block (and the ALL block) sums to its total n, CS count and vaginal count,
    all derivable (from ``pct_of_deliveries`` and ``rel_contribution``, or the record count);
    and for each row label the facility cells sum to the published ALL cell.
    """
    blocks = table.groupby("facility", sort=False)
    groups = [SumRelation(tuple(block.index)) for _, block in blocks]
    keys = zip(table["facility"], table["row"], strict=True)
    position = {key: i for i, key in zip(table.index, keys, strict=True)}
    for label in dict.fromkeys(table["row"]):
        members = tuple(i for (f, r), i in position.items() if r == label and f != OVERALL)
        total = position.get((OVERALL, label))
        if members and total is not None:
            groups.append(SumRelation(members, total))
    return groups


def _two_decimals(value: object) -> object:
    if isinstance(value, str) or pd.isna(value):  # type: ignore[call-overload]
        return value
    return f"{float(value):.{REPORT_DECIMALS}f}"  # type: ignore[arg-type]


def suppress_report_table(table: pd.DataFrame) -> pd.DataFrame:
    """The report table as it may be exported: primary, then secondary, suppression.

    Primary: ``REPORT_SUPPRESSION`` (n or n_cs of 1-4, or n - n_cs of 1-4, with every value
    derived from them). Secondary: within each facility block, and across the facilities of
    each row label (whose sum is the ALL row), no hidden n, n_cs or n - n_cs is recoverable
    by subtraction (``privacy.suppress_table``). Finally ``pct_of_deliveries``,
    ``abs_contribution`` and ``rel_contribution`` are published to 2 decimals everywhere:
    at 3 decimals, ``rel_contribution`` on the shown rows pins a facility's CS total
    exactly. Secondary suppression already assumes every such total is known, so rounding
    is defence in depth, applied uniformly so the precision itself says nothing about
    which blocks hide cells.
    """
    table = table.reset_index(drop=True)
    safe = suppress_table(table, **REPORT_SUPPRESSION, groups=report_groups(table))
    for column in REPORT_ROUNDED:
        safe[column] = safe[column].astype(object).map(_two_decimals)
    return safe
