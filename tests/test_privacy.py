from pathlib import Path

import numpy as np
import pandas as pd

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
