from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robson_ml.privacy import (
    SUPPRESSED,
    fmt_count,
    level_counts,
    safe_describe,
    safe_pct,
    suppress_small_cells,
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
