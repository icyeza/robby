import math

import pandas as pd
import pytest

from robson_ml.audit_offline import (
    OVERALL,
    REPORT_SUPPRESSION,
    RESIDUAL,
    robson_report_table,
    suppress_report_table,
)
from robson_ml.populations import audit_population
from robson_ml.privacy import SECONDARY, SUPPRESSED, suppress_small_cells

HIDDEN = [SUPPRESSED, SECONDARY]


def _frame() -> pd.DataFrame:
    rows = [
        ("A", "resolved", 1, 0),
        ("A", "resolved", 1, 1),
        ("A", "resolved", 5, 1),
        ("A", "resolved", 5, 1),
        ("A", "resolved", 5, 1),
        ("A", "partial", None, 0),
        ("B", "resolved", 3, 0),
        ("B", "resolved", 3, 0),
        ("B", "conflict", None, 1),
        ("B", "resolved", 5, 1),
    ]
    df = pd.DataFrame(rows, columns=["facility_id", "robson_status", "robson_group", "cs"])
    df["robson_group"] = df["robson_group"].astype("Int64")
    df["cs"] = df["cs"].astype("Int64")
    return df


def _row(table: pd.DataFrame, facility: str, row: str) -> pd.Series:
    return table[(table["facility"] == facility) & (table["row"] == row)].iloc[0]


def test_overall_rows() -> None:
    table = robson_report_table(_frame())
    g1 = _row(table, OVERALL, "1")
    assert (g1["n"], g1["n_cs"]) == (2, 1)
    assert g1["pct_of_deliveries"] == pytest.approx(0.2)
    assert g1["cs_rate"] == pytest.approx(0.5)
    assert g1["abs_contribution"] == pytest.approx(0.1)
    assert g1["rel_contribution"] == pytest.approx(1 / 6)
    g5 = _row(table, OVERALL, "5")
    assert (g5["n"], g5["n_cs"]) == (4, 4)
    assert g5["rel_contribution"] == pytest.approx(4 / 6)
    residual = _row(table, OVERALL, RESIDUAL)
    assert (residual["n"], residual["n_cs"]) == (2, 1)


def test_facility_rows_and_empty_groups() -> None:
    table = robson_report_table(_frame())
    a5 = _row(table, "A", "5")
    assert a5["rel_contribution"] == pytest.approx(3 / 4)
    b9 = _row(table, "B", "9")
    assert b9["n"] == 0
    assert math.isnan(b9["cs_rate"])
    assert len(table) == 3 * 11


def test_requires_p_audit() -> None:
    df = _frame()
    df.loc[0, "cs"] = pd.NA
    with pytest.raises(ValueError, match="P_audit"):
        robson_report_table(df)


def test_audit_population_logs_exclusions() -> None:
    df = _frame()
    df.loc[[0, 1], "cs"] = pd.NA
    kept, log = audit_population(df)
    assert len(kept) == 8
    assert (log.n_excluded, log.n_remaining) == (2, 8)


def test_report_suppression_hides_small_complements() -> None:
    table = pd.DataFrame(
        [
            {
                "facility": "ALL",
                "row": "9",
                "n": 12,
                "pct_of_deliveries": 0.1,
                "n_cs": 10,
                "cs_rate": 10 / 12,
                "abs_contribution": 0.08,
                "rel_contribution": 0.1,
            },
            {
                "facility": "ALL",
                "row": "1",
                "n": 100,
                "pct_of_deliveries": 0.5,
                "n_cs": 50,
                "cs_rate": 0.5,
                "abs_contribution": 0.25,
                "rel_contribution": 0.4,
            },
        ]
    )
    safe = suppress_small_cells(table, **REPORT_SUPPRESSION)
    assert safe.loc[0, "n_cs"] == SUPPRESSED
    assert safe.loc[0, "cs_rate"] == SUPPRESSED
    assert safe.loc[1, "n_cs"] == 50


def test_empty_frame_returns_overall_rows_only() -> None:
    df = pd.DataFrame(columns=["facility_id", "robson_status", "robson_group", "cs"])
    df["robson_group"] = df["robson_group"].astype("Int64")
    df["cs"] = df["cs"].astype("Int64")
    table = robson_report_table(df)
    assert len(table) == 11
    assert (table["facility"] == OVERALL).all()
    assert (table["n"] == 0).all()


def _crafted() -> pd.DataFrame:
    """Two facilities; A's group 3 has 3 deliveries and B's group 4 has 1 vaginal birth."""
    rows: list[tuple[str, str, int | None, int]] = []

    def add(facility: str, group: int, n: int, n_cs: int) -> None:
        rows.extend((facility, "resolved", group, int(i < n_cs)) for i in range(n))

    add("A", 1, 20, 10)
    add("A", 2, 30, 15)
    add("A", 3, 3, 1)
    add("A", 4, 25, 10)
    add("B", 1, 25, 12)
    add("B", 2, 28, 14)
    add("B", 3, 10, 5)
    add("B", 4, 13, 12)
    df = pd.DataFrame(rows, columns=["facility_id", "robson_status", "robson_group", "cs"])
    df["robson_group"] = df["robson_group"].astype("Int64")
    df["cs"] = df["cs"].astype("Int64")
    return df


def _assert_no_lone_suppressed(safe: pd.DataFrame) -> None:
    for column in ("n", "n_cs"):
        for facility, block in safe.groupby("facility"):
            assert block[column].isin(HIDDEN).sum() != 1, (facility, column)
        overall = safe[safe["facility"] == OVERALL].set_index("row")
        for label, cells in safe[safe["facility"] != OVERALL].groupby("row"):
            if overall.loc[label, column] not in HIDDEN:
                assert cells[column].isin(HIDDEN).sum() != 1, (label, column)


def test_suppress_report_table_blocks_subtraction_within_and_across_facilities() -> None:
    table = robson_report_table(_crafted())
    safe = suppress_report_table(table)
    assert _row(safe, "A", "3")["n"] == SUPPRESSED
    assert _row(safe, "B", "4")["n_cs"] == SUPPRESSED
    # Without secondary suppression A/3 = N_A - (other A rows) and B/3 = ALL/3 - A/3.
    _assert_no_lone_suppressed(safe)
    # Wherever n_cs is hidden, every value derived from it is hidden too.
    hidden = safe["n_cs"].isin(HIDDEN)
    for column in ("cs_rate", "abs_contribution", "rel_contribution"):
        assert safe.loc[hidden, column].isin(HIDDEN).all()


def test_suppress_report_table_defeats_contribution_total_recovery() -> None:
    # Reviewer recovery: rel_contribution pins the facility CS total, then the lone hidden
    # n_cs is total - shown. Every block now hides at least two n_cs, so the remainder is a
    # sum of hidden cells, never one cell.
    table = robson_report_table(_crafted())
    safe = suppress_report_table(table)
    for facility, block in safe.groupby("facility"):
        hidden = block[block["n_cs"].isin(HIDDEN)]
        assert len(hidden) == 0 or len(hidden) >= 2, facility


def test_suppress_report_table_rounds_contributions_to_two_decimals() -> None:
    safe = suppress_report_table(robson_report_table(_frame()))
    for column in ("pct_of_deliveries", "abs_contribution", "rel_contribution"):
        for value in safe[column]:
            if value not in HIDDEN and not pd.isna(value):
                assert isinstance(value, str) and len(value.split(".")[1]) == 2, value
