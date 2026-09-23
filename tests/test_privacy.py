from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robson_ml.privacy import (
    SECONDARY,
    SUPPRESSED,
    SumRelation,
    TableSpec,
    fmt_count,
    level_counts,
    protect_cells,
    safe_describe,
    safe_pct,
    suppress_small_cells,
    suppress_table,
    suppress_tables,
)
from robson_ml.reporting import markdown_table, write_table


def test_suppress_counts_one_to_four_and_linked_rates() -> None:
    df = pd.DataFrame({"n": [0, 3, 10], "rate": [np.nan, 0.33, 0.5]})
    out = suppress_small_cells(df, ["n"], {"n": ["rate"]})
    assert out["n"].tolist() == [0, SUPPRESSED, 10]
    assert out["rate"].tolist()[1] == SUPPRESSED
    assert out["rate"].tolist()[2] == 0.5


def test_small_cell_suppression(tmp_path: Path) -> None:
    df = pd.DataFrame({"group": ["a", "b", "c"], "n": [2, 7, 4], "n_cs": [1, 6, 0]})
    path = tmp_path / "t.csv"
    write_table(df, path, ["n", "n_cs"], {"n": ["n_cs"]})
    exported = pd.read_csv(path, dtype=str)
    for col in ["n", "n_cs"]:
        numeric = pd.to_numeric(exported[col], errors="coerce")
        assert not ((numeric > 0) & (numeric < 5)).any()
    assert exported.loc[0, "n_cs"] == SUPPRESSED


def test_safe_pct_suppresses_small_numerator_or_complement() -> None:
    assert safe_pct(pd.Series([True] * 3 + [False] * 97)) == SUPPRESSED
    assert safe_pct(pd.Series([True] * 97 + [False] * 3)) == SUPPRESSED
    assert safe_pct(pd.Series([True] * 50 + [False] * 50)) == 50.0
    assert safe_pct(pd.Series([False] * 100)) == 0.0


def test_fmt_count() -> None:
    assert fmt_count(0) == "0"
    assert fmt_count(4) == SUPPRESSED
    assert fmt_count(12) == "12"


def test_level_counts_hides_rare_labels_and_free_text() -> None:
    s = pd.Series(["x"] * 10 + ["rare_label"] * 2 + [None] * 6)
    out = level_counts(s)
    assert "rare_label" not in out["level"].tolist()
    assert out.loc[out["level"] == "x", "n"].item() == 10
    free = level_counts(pd.Series([f"note {i}" for i in range(40)]))
    assert len(free) == 1
    assert "not shown" in free["level"].item()


def test_safe_describe_has_no_values() -> None:
    df = pd.DataFrame({"a": [1, 2, None], "b": ["zzsecret1", "zzsecret2", "zzsecret3"]})
    out = safe_describe(df)
    assert list(out.columns) == ["column", "dtype", "n_nonnull", "pct_missing", "n_unique"]
    assert "zzsecret" not in out.to_string()


def test_markdown_table() -> None:
    text = markdown_table(pd.DataFrame({"a": [1, 2], "b": [0.12345, np.nan]}))
    assert text.splitlines()[0] == "| a | b |"
    assert "0.123" in text


def test_safe_describe_suppresses_small_nonnull_and_small_missing() -> None:
    n = 5520
    df = pd.DataFrame(
        {
            "few_present": [1.0, 2.0, 3.0] + [np.nan] * (n - 3),
            "few_missing": [1.0] * (n - 2) + [np.nan, np.nan],
            "normal": list(range(n)),
        }
    )
    out = safe_describe(df).set_index("column")

    for col in ["few_present", "few_missing"]:
        assert out.loc[col, "n_nonnull"] == SUPPRESSED
        assert out.loc[col, "pct_missing"] == SUPPRESSED
        assert out.loc[col, "n_unique"] == SUPPRESSED

    assert out.loc["normal", "n_nonnull"] == n
    assert out.loc["normal", "pct_missing"] == 0.0
    assert out.loc["normal", "n_unique"] == n


def test_write_table_requires_count_columns() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError):
        write_table(df, Path("unused.csv"), [])


def test_suppress_small_cells_complement_suppresses_near_total_counts() -> None:
    df = pd.DataFrame({"n_total": [100, 100], "n_cs": [97, 50], "pct": [0.97, 0.50]})
    out = suppress_small_cells(
        df, ["n_cs"], linked={"n_cs": ["pct"]}, complements={"n_cs": "n_total"}
    )
    assert out.loc[0, "n_cs"] == SUPPRESSED
    assert out.loc[0, "pct"] == SUPPRESSED
    assert out.loc[1, "n_cs"] == 50
    assert out.loc[1, "pct"] == 0.50


def test_level_counts_free_text_when_rare_levels_dominate() -> None:
    s = pd.Series(["Jane Doe"] * 5 + [f"unique_{i}" for i in range(20)])
    out = level_counts(s)
    assert len(out) == 1
    assert "not shown" in out["level"].item()
    assert "Jane Doe" not in out["level"].tolist()


def test_level_counts_still_shows_common_label_with_some_rare() -> None:
    s = pd.Series(["x"] * 10 + ["rare_label"] * 2)
    out = level_counts(s)
    assert out.loc[out["level"] == "x", "n"].item() == 10


def test_safe_pct_drops_na_before_computing() -> None:
    mask = pd.Series([True] * 6 + [pd.NA] * 4 + [False] * 6, dtype="boolean")
    assert safe_pct(mask) == 50.0


def test_markdown_table_formats_nat_and_np_float() -> None:
    text = markdown_table(pd.DataFrame({"a": [pd.NaT], "b": [np.float32(0.5)]}))
    row = text.splitlines()[2]
    assert row == "|  | 0.500 |"


def test_suppress_table_lone_suppressed_cell_forces_a_second() -> None:
    # Reviewer recovery: with N published, conflict = N - resolved - partial.
    df = pd.DataFrame(
        {
            "status": ["resolved", "partial", "conflict"],
            "n": [2728, 269, 3],
            "pct": [90.9, 9.0, 0.1],
        }
    )
    out = suppress_table(df, ["n"], {"n": ["pct"]})
    assert out["n"].tolist() == [2728, SECONDARY, SUPPRESSED]
    assert out["pct"].tolist() == [90.9, SECONDARY, SUPPRESSED]


def test_suppress_table_all_ones_forces_a_third() -> None:
    df = pd.DataFrame({"n": [1, 1, 20, 50]})
    out = suppress_table(df, ["n"])
    assert out["n"].tolist() == [SUPPRESSED, SUPPRESSED, SECONDARY, 50]


def test_suppress_table_leaves_undeterminable_pairs_alone() -> None:
    df = pd.DataFrame({"n": [3, 2, 20, 50, 0]})
    out = suppress_table(df, ["n"])
    assert out["n"].tolist() == [SUPPRESSED, SUPPRESSED, 20, 50, 0]


def test_suppress_table_untouched_without_small_cells() -> None:
    df = pd.DataFrame({"n": [30, 20, 0]})
    assert suppress_table(df, ["n"])["n"].tolist() == [30, 20, 0]


def test_suppress_table_group_with_total_row() -> None:
    # Row 0 is the published total of rows 1-3 (e.g. overall vs per facility).
    df = pd.DataFrame({"n_missing": [12, 2, 5, 5], "n_rows": [100, 30, 30, 40]})
    out = suppress_table(
        df,
        ["n_missing"],
        complements={"n_missing": "n_rows"},
        groups=[SumRelation((1, 2, 3), total=0)],
    )
    assert out["n_missing"].tolist() == [12, SUPPRESSED, SECONDARY, 5]


def test_suppress_table_protects_small_complements_in_a_group() -> None:
    # n_cs = 12 of n = 13 is hidden because 1 vaginal birth is small; the block's CS total
    # is derivable, so another n_cs must go too.
    df = pd.DataFrame({"n": [13, 40, 60], "n_cs": [12, 20, 30], "rate": [0.9, 0.5, 0.5]})
    out = suppress_table(
        df,
        ["n", "n_cs"],
        linked={"n": ["n_cs", "rate"], "n_cs": ["rate"]},
        complements={"n_cs": "n"},
    )
    assert out["n"].tolist() == [13, 40, 60]
    assert out["n_cs"].tolist() == [SUPPRESSED, SECONDARY, 30]
    assert out["rate"].tolist() == [SUPPRESSED, SECONDARY, 0.5]


def test_suppress_table_two_small_complements_summing_to_minimum() -> None:
    # Two hidden vaginal counts of 1 each: their sum (2) is derivable, so both are 1.
    df = pd.DataFrame({"n": [13, 21, 60, 70], "n_cs": [12, 20, 30, 35]})
    out = suppress_table(df, ["n", "n_cs"], linked={"n": ["n_cs"]}, complements={"n_cs": "n"})
    hidden = [i for i, v in enumerate(out["n_cs"]) if v in (SUPPRESSED, SECONDARY)]
    assert len(hidden) >= 3


def test_protect_cells_closes_chained_derivations() -> None:
    # a + x and x + y both have derivable totals; with y published, x and then a follow.
    values = {"a": 2, "x": 10, "y": 20, "z": 30}
    relations = [SumRelation(("a", "x")), SumRelation(("x", "y", "z"))]
    hidden = protect_cells(values, {"a", "x"}, relations)
    assert {"a", "x", "y"} <= hidden
    assert "z" not in hidden


def test_protect_cells_ignores_single_member_external_relation() -> None:
    # A one-member group with an external total says the cell is published elsewhere.
    assert protect_cells({"a": 2}, {"a"}, [SumRelation(("a",))]) == {"a"}


def _facility_table(n_true: list[int], n_rows: list[int]) -> pd.DataFrame:
    return suppress_table(
        pd.DataFrame({"n_true": n_true, "n_rows": n_rows}),
        ["n_true"],
        complements={"n_true": "n_rows"},
        groups=[SumRelation((1, 2, 3, 4), total=0)],
    )


def test_secondary_choice_does_not_depend_on_which_side_is_counted() -> None:
    # Reviewer recovery R2: counting missing hid FAC_B, counting recorded hid FAC_D (the
    # smallest recorded count), and together the two files gave FAC_A back.
    rows = [3000, 886, 751, 750, 613]
    missing = [32, 2, 10, 10, 10]
    recorded = [n - m for n, m in zip(rows, missing, strict=True)]
    by_missing = _facility_table(missing, rows)["n_true"].isin([SUPPRESSED, SECONDARY])
    by_recorded = _facility_table(recorded, rows)["n_true"].isin([SUPPRESSED, SECONDARY])
    assert by_missing.tolist() == by_recorded.tolist()
    assert by_missing.tolist() == [False, True, True, False, False]


def test_secondary_cells_get_their_own_marker() -> None:
    df = pd.DataFrame(
        {"status": ["resolved", "partial", "conflict"], "n": [2728, 269, 3], "pct": [1, 2, 3]}
    )
    out = suppress_table(df, ["n"], {"n": ["pct"]})
    assert out["n"].tolist() == [2728, SECONDARY, SUPPRESSED]
    assert out["pct"].tolist() == [1, SECONDARY, SUPPRESSED]


def test_suppress_tables_protects_cells_across_tables() -> None:
    # The second table's rows sum to the first table's 40: hiding the 40 alone would not
    # protect the 3 (40 = 20 + 20), so the joint protection hides the 100 instead.
    first = TableSpec(pd.DataFrame({"n": [100, 40, 3]}), ["n"])
    second = TableSpec(pd.DataFrame({"n": [20, 20]}), ["n"], groups=[])
    cross = [SumRelation((("second", 0, "n"), ("second", 1, "n")), ("first", 1, "n"))]
    out = suppress_tables({"first": first, "second": second}, cross)
    assert out["first"]["n"].tolist() == [SECONDARY, 40, SUPPRESSED]
    assert out["second"]["n"].tolist() == [20, 20]
    alone = suppress_table(first.df, ["n"])
    assert alone["n"].tolist() == [100, SECONDARY, SUPPRESSED]
