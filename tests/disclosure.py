"""Test helper: scan published profile outputs for small cells and subtraction recoveries.

The scanner reads only what would be published (markdown and CSV), the way an outside
reader would, and asserts two things:

(i) no count column holds an unsuppressed count of 1-4;
(ii) in every group of cells whose total is published or derivable, the suppressed cells
     cannot be recovered by subtraction: the group hides no cell or at least two, and when
     the total is known the hidden remainder is neither ``k`` (all ones) nor ``4k`` (all
     fours) for ``k`` hidden cells.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from robson_ml.privacy import SUPPRESSED

COUNT_COLUMNS = frozenset({"n", "n_cs", "n_nonnull"})
MISSING_LABEL = "(missing)"


def markdown_tables(text: str) -> list[pd.DataFrame]:
    """Every GitHub markdown table in ``text``, as string frames, in order."""
    tables: list[pd.DataFrame] = []
    block: list[str] = []
    for line in [*text.splitlines(), ""]:
        if line.startswith("|"):
            block.append(line)
            continue
        if len(block) >= 2:
            tables.append(_parse(block))
        block = []
    return tables


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _parse(lines: Sequence[str]) -> pd.DataFrame:
    header = _cells(lines[0])
    return pd.DataFrame([_cells(line) for line in lines[2:]], columns=header, dtype=str)


def _number(cell: object) -> float | None:
    try:
        return float(str(cell))
    except ValueError:
        return None


def assert_no_small_counts(table: pd.DataFrame, where: str) -> None:
    """(i): no count column in ``table`` shows a count of 1-4."""
    for column in COUNT_COLUMNS & set(table.columns):
        for cell in table[column]:
            value = _number(cell)
            assert value is None or not 1 <= value <= 4, f"{where}: {column}={cell}"


def assert_not_recoverable(cells: Sequence[object], total: float | None, where: str) -> None:
    """(ii): the suppressed cells of a group summing to ``total`` cannot be recovered."""
    hidden = [cell for cell in cells if cell == SUPPRESSED]
    assert len(hidden) != 1, f"{where}: a lone suppressed cell is recoverable by subtraction"
    if total is None or not hidden:
        return
    shown = sum(_number(cell) or 0.0 for cell in cells if cell != SUPPRESSED)
    remainder = total - shown
    k = len(hidden)
    assert remainder not in (k, 4 * k), f"{where}: {k} hidden cells summing to {remainder}"


def _counts_table_checks(table: pd.DataFrame, total: float | None, where: str) -> None:
    assert_no_small_counts(table, where)
    assert_not_recoverable(table["n"].tolist(), total, where)
    recorded = table[table.iloc[:, 0] != MISSING_LABEL]
    if 2 <= len(recorded) < len(table):
        assert_not_recoverable(recorded["n"].tolist(), None, f"{where} (recorded levels)")


def _report_checks(report: pd.DataFrame) -> None:
    assert_no_small_counts(report, "robson report")
    for facility, block in report.groupby("facility", sort=False):
        for column in ("n", "n_cs"):
            assert_not_recoverable(block[column].tolist(), None, f"report {facility} {column}")
    facilities = report[report["facility"] != "ALL"]
    overall = report[report["facility"] == "ALL"].set_index("row")
    for label, cells in facilities.groupby("row", sort=False):
        for column in ("n", "n_cs"):
            if overall.loc[label, column] != SUPPRESSED:
                assert_not_recoverable(
                    cells[column].tolist(), None, f"report row {label} {column} across facilities"
                )


def _wide_checks(table: pd.DataFrame, total_column: str, facility_columns: Sequence[str]) -> None:
    for _, row in table.iterrows():
        if row[total_column] != SUPPRESSED:
            where = f"{row.iloc[0]}: {total_column} vs {list(facility_columns)}"
            assert_not_recoverable([row[c] for c in facility_columns], None, where)


def scan_robson_inputs(text: str) -> None:
    records = int(re.search(r"Records: (\d+)\.", text).group(1))  # type: ignore[union-attr]
    completeness, status, resolving, report = markdown_tables(text)
    _wide_checks(
        completeness, "all", [c for c in completeness.columns if c not in ("input", "all")]
    )
    _counts_table_checks(status, records, "engine status")
    partial = _number(status.set_index("status").loc["partial", "n"])
    _counts_table_checks(resolving, partial, "resolving fields")
    _report_checks(report)


def scan_open_questions(text: str, records: int) -> None:
    sections = re.split(r"\n## Q(\d+)\. ", text)[1:]
    for number, body in zip(sections[::2], sections[1::2], strict=True):
        for index, table in enumerate(markdown_tables(body)):
            assert_no_small_counts(table, f"Q{number}")
            if "n" not in table.columns:
                continue
            # Every counts table covers the whole frame except Q2's second one (pre-labour CS
            # rows only), whose total is the published prelabour_cs count.
            total = None if (number == "2" and index == 1) else records
            _counts_table_checks(table, total, f"Q{number} table {index}")


def scan_variable_profile(profile: pd.DataFrame) -> None:
    assert_no_small_counts(profile, "variable_profile")
    facility_columns = [c for c in profile.columns if c.startswith("pct_missing_")]
    _wide_checks(profile, "pct_missing", facility_columns)


def scan_profile_outputs(out_dir: Path) -> None:
    """Run every check over the three files ``write_profile`` publishes."""
    inputs = (out_dir / "robson_inputs.md").read_text(encoding="utf-8")
    scan_robson_inputs(inputs)
    records = int(re.search(r"Records: (\d+)\.", inputs).group(1))  # type: ignore[union-attr]
    scan_open_questions((out_dir / "open_questions.md").read_text(encoding="utf-8"), records)
    scan_variable_profile(pd.read_csv(out_dir / "variable_profile.csv", dtype=str))
