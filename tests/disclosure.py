"""Test helper: scan published profile outputs for small cells and subtraction recoveries.

The scanner reads only what would be published (markdown and CSV), the way an outside
reader would, and asserts:

(i) no count column holds an unsuppressed count of 1-4;
(ii) in every group of cells whose total is published or derivable, the suppressed cells
     cannot be recovered by subtraction: the group hides no cell or at least two, and when
     the total is known the hidden remainder does not pin a primary (``"<5"``, 1-4) cell.
     A secondary cell (``"*"``) may hold any value from 0 up;
(iii) across tables: the engine status table, the resolving-fields table and the Robson
     report's n cells form one linear system (status sums to Records, resolving fields to
     the partial count, each report block to its derivable total, each report row label
     across facilities to its ALL cell, and partial + conflict to the ALL residual); no
     hidden cell of it is determined;
(iv) for every Robson input published from exactly one raw column, the per-facility cells
     hidden in the variable profile and in the completeness table are the same, so one
     file cannot fill in what the other hides.
"""

from __future__ import annotations

import re
from collections.abc import Hashable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from robson_engine import INPUT_FIELDS
from robson_ml.privacy import SECONDARY, SUPPRESSED

COUNT_COLUMNS = frozenset({"n", "n_cs", "n_nonnull"})
MISSING_LABEL = "(missing)"
HIDDEN = (SUPPRESSED, SECONDARY)


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


def is_hidden(cell: object) -> bool:
    return cell in HIDDEN


def assert_no_small_counts(table: pd.DataFrame, where: str) -> None:
    """(i): no count column in ``table`` shows a count of 1-4."""
    for column in COUNT_COLUMNS & set(table.columns):
        for cell in table[column]:
            value = _number(cell)
            assert value is None or not 1 <= value <= 4, f"{where}: {column}={cell}"


def assert_not_recoverable(cells: Sequence[object], total: float | None, where: str) -> None:
    """(ii): the suppressed cells of a group summing to ``total`` cannot be recovered.

    A reader takes each ``"<5"`` as 1-4 and each ``"*"`` as 0 or more. With a known total
    the hidden cells sum to a known remainder; a primary cell is pinned when that remainder
    leaves it a single value.
    """
    hidden = [cell for cell in cells if is_hidden(cell)]
    assert len(hidden) != 1, f"{where}: a lone suppressed cell is recoverable by subtraction"
    if total is None or not hidden:
        return
    shown = sum(_number(cell) or 0.0 for cell in cells if not is_hidden(cell))
    remainder = total - shown
    primary = sum(cell == SUPPRESSED for cell in hidden)
    if not primary:
        return
    if primary == len(hidden):
        assert remainder not in (primary, 4 * primary), (
            f"{where}: {primary} hidden cells summing to {remainder}"
        )
    else:
        # Secondaries may be 0, so each primary is at most remainder - (primary - 1).
        assert remainder > primary, f"{where}: primaries pinned to 1 by remainder {remainder}"


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
            if not is_hidden(overall.loc[label, column]):
                assert_not_recoverable(
                    cells[column].tolist(), None, f"report row {label} {column} across facilities"
                )


def _wide_checks(table: pd.DataFrame, total_column: str, facility_columns: Sequence[str]) -> None:
    for _, row in table.iterrows():
        if not is_hidden(row[total_column]):
            where = f"{row.iloc[0]}: {total_column} vs {list(facility_columns)}"
            assert_not_recoverable([row[c] for c in facility_columns], None, where)


def determined_cells(
    equations: Sequence[Sequence[Hashable]], signs: Sequence[Sequence[int]], unknown: set
) -> set:
    """Unknown cells fixed by linear equations whose known terms are constants.

    ``equations[i]`` lists cells with coefficients ``signs[i]``. A cell is determined when
    its unit vector lies in the row space of the equations restricted to unknown cells.
    """
    cells = sorted(unknown, key=repr)
    index = {cell: j for j, cell in enumerate(cells)}
    rows = []
    for members, coeffs in zip(equations, signs, strict=True):
        row = np.zeros(len(cells))
        for cell, c in zip(members, coeffs, strict=True):
            if cell in index:
                row[index[cell]] += c
        if row.any():
            rows.append(row)
    if not rows:
        return set()
    matrix = np.array(rows)
    rank = np.linalg.matrix_rank(matrix)
    out = set()
    for cell, j in index.items():
        unit = np.zeros(len(cells))
        unit[j] = 1.0
        if np.linalg.matrix_rank(np.vstack([matrix, unit])) == rank:
            out.add(cell)
    return out


def assert_linked_status_system(
    status: pd.DataFrame, resolving: pd.DataFrame, report: pd.DataFrame
) -> None:
    """(iii): no hidden n in the status / resolving-fields / report system is determined."""
    cells: dict[Hashable, object] = {}
    equations: list[list[Hashable]] = []
    signs: list[list[int]] = []

    def add(members: list[Hashable], coeffs: list[int]) -> None:
        equations.append(members)
        signs.append(coeffs)

    status_cells = [("status", s) for s in status["status"]]
    cells.update(zip(status_cells, status["n"], strict=True))
    add(status_cells, [1] * len(status_cells))  # = Records
    resolving_cells = [("resolving", r) for r in resolving.iloc[:, 0]]
    cells.update(zip(resolving_cells, resolving["n"], strict=True))
    add([*resolving_cells, ("status", "partial")], [1] * len(resolving_cells) + [-1])
    report_cells = {
        (f, r): ("report", f, r) for f, r in zip(report["facility"], report["row"], strict=True)
    }
    cells.update(zip(report_cells.values(), report["n"], strict=True))
    for facility in dict.fromkeys(report["facility"]):
        block = [c for (f, _), c in report_cells.items() if f == facility]
        add(block, [1] * len(block))  # = derivable block total
    for label in dict.fromkeys(report["row"]):
        members = [c for (f, r), c in report_cells.items() if r == label and f != "ALL"]
        if members:
            add([*members, report_cells[("ALL", label)]], [1] * len(members) + [-1])
    # Exact when no record is excluded for missing outcome; otherwise a reader knows it
    # within the excluded count, and treating it as exact is the stronger reader.
    add(
        [("status", "partial"), ("status", "conflict"), report_cells[("ALL", "residual")]],
        [1, 1, -1],
    )
    unknown = {cell for cell, value in cells.items() if is_hidden(value)}
    exposed = determined_cells(equations, signs, unknown)
    assert not exposed, f"status/resolving/report hidden cells determined: {sorted(exposed)}"


def scan_robson_inputs(text: str) -> None:
    records = int(re.search(r"Records: (\d+)\.", text).group(1))  # type: ignore[union-attr]
    completeness, status, resolving, report = markdown_tables(text)
    _wide_checks(
        completeness, "all", [c for c in completeness.columns if c not in ("input", "all")]
    )
    _counts_table_checks(status, records, "engine status")
    partial = _number(status.set_index("status").loc["partial", "n"])
    if partial is not None:
        _counts_table_checks(resolving, partial, "resolving fields")
    else:
        # The partial total is hidden; (iii) checks whether it can be derived.
        assert_no_small_counts(resolving, "resolving fields")
    _report_checks(report)
    assert_linked_status_system(status, resolving, report)


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


def assert_linked_inputs_hide_alike(profile: pd.DataFrame, completeness: pd.DataFrame) -> None:
    """(iv): same hidden facility cells for an input and its single raw column."""
    names = profile["canonical_name"].fillna("")
    by_input = completeness.set_index("input")
    facilities = [c for c in completeness.columns if c not in ("input", "all")]
    for field in INPUT_FIELDS:
        rows = profile[names == field]
        if len(rows) != 1 or field not in by_input.index:
            continue
        raw = rows.iloc[0]
        in_profile = {f for f in facilities if is_hidden(raw[f"pct_missing_{f}"])}
        in_completeness = {f for f in facilities if is_hidden(by_input.loc[field, f])}
        assert in_profile == in_completeness, (
            f"{field}: profile hides {sorted(in_profile)}, completeness {sorted(in_completeness)}"
        )


def scan_profile_outputs(out_dir: Path) -> None:
    """Run every check over the three files ``write_profile`` publishes."""
    inputs = (out_dir / "robson_inputs.md").read_text(encoding="utf-8")
    scan_robson_inputs(inputs)
    records = int(re.search(r"Records: (\d+)\.", inputs).group(1))  # type: ignore[union-attr]
    scan_open_questions((out_dir / "open_questions.md").read_text(encoding="utf-8"), records)
    profile = pd.read_csv(out_dir / "variable_profile.csv", dtype=str)
    scan_variable_profile(profile)
    assert_linked_inputs_hide_alike(profile, markdown_tables(inputs)[0])
