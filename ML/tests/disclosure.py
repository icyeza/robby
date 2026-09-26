"""Test helper: scan published profile outputs for small cells and subtraction recoveries.

The scanner reads only what would be published (markdown and CSV), the way an outside
reader would, and asserts:

(i) no count column holds an unsuppressed count of 1-4;
(ii) in every group of cells whose total is published or derivable, the suppressed cells
     cannot be recovered by subtraction: the group hides no cell or at least two, and when
     the total is known, or itself shown as ``"<5"`` (so 1-4), the bounds do not pin a
     primary (``"<5"``, 1-4) cell or the total. A secondary cell (``"*"``) may hold any
     value from 0 up;
(iii) across tables: the engine status table, the resolving-fields table and the Robson
     report's n cells form one linear system (status sums to Records, resolving fields to
     the partial count, each report block to its derivable total, each report row label
     across facilities to its ALL cell, and partial + conflict to the ALL residual); no
     hidden cell of it is determined;
(iv) for every Robson input published from exactly one raw column, the per-facility cells
     hidden in the variable profile and in the completeness table are the same, so one
     file cannot fill in what the other hides (with the same markers); likewise for the
     ``ga_band (recorded)`` row and the raw column the band is mapped from;
(v) every raw column mapped to ``mother_key`` (the only field a ``hash_key`` may produce)
     publishes ``n_unique`` as ``"*"``: ``n_nonnull - n_unique`` would count the rows that
     repeat a key, which Q9 publishes only suppressed. It publishes no quantile and no
     association either (a quantile of a numeric ID is one patient's ID);
(vi) every published copy of a field's overall missing count agrees on whether it is
     hidden, and with which marker: the ``"(missing)"`` row of its level counts in
     open_questions.md (Q2 onset, Q3 PE and GDM, Q6 months, Q9 plurality), the ``all`` cell
     of its completeness row and of Q1 (a Robson input), and its single raw column's
     ``pct_missing`` and ``n_nonnull`` in the variable profile. So the recorded levels of a
     level table have a known total exactly when the ``"(missing)"`` row is shown.

:class:`CountReader` is a stronger reader for tests: an integer programme over published
percentages and markers, used to check that given never-published counts cannot be pinned.
"""

from __future__ import annotations

import math
import re
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from robson_engine import INPUT_FIELDS
from robson_ml.privacy import SECONDARY, SUPPRESSED
from robson_ml.profile import GA_BAND_RECORDED

COUNT_COLUMNS = frozenset({"n", "n_cs", "n_nonnull"})
MISSING_LABEL = "(missing)"
# open_questions.md (question, table index) -> the canonical field whose levels it counts,
# with a "(missing)" row linked to the field's other published missing counts (vi).
LINKED_LEVEL_TABLES = {
    ("2", 0): "onset_of_labour",
    ("3", 0): "preeclampsia_recorded",
    ("3", 1): "gdm_recorded",
    ("6", 0): "delivery_date",
    ("9", 0): "plurality",
}
QUANTILE_COLUMNS = ("p5", "p25", "p50", "p75", "p95")
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


def _bounds(cell: object) -> tuple[float, float]:
    """What a reader knows of one cell: ``"<5"`` is 1-4, ``"*"`` anything from 0 up."""
    if cell == SUPPRESSED:
        return 1.0, 4.0
    value = _number(cell)
    return (value, value) if value is not None else (0.0, math.inf)


def _pinned(cells: Sequence[object], total: object) -> list[str]:
    """Primary cells (and a ``"<5"`` total) the sum leaves a single value, as messages."""
    bounds = [_bounds(cell) for cell in cells]
    t_lo, t_hi = _bounds(total)
    lo_sum = sum(lo for lo, _ in bounds)
    hi_sum = sum(hi for _, hi in bounds)
    if lo_sum > t_hi or hi_sum < t_lo:
        return []  # this reading of the markers is infeasible
    out = []
    for i, cell in enumerate(cells):
        if cell != SUPPRESSED:
            continue
        others_hi = sum(hi for j, (_, hi) in enumerate(bounds) if j != i)
        others_lo = lo_sum - bounds[i][0]
        if max(1.0, t_lo - others_hi) >= min(4.0, t_hi - others_lo):
            out.append(f"cell {i} pinned")
    if total == SUPPRESSED and max(1.0, lo_sum) >= min(4.0, hi_sum):
        out.append("the '<5' total pinned")
    return out


def assert_not_recoverable(cells: Sequence[object], total: object, where: str) -> None:
    """(ii): the suppressed cells of a group summing to ``total`` cannot be recovered.

    ``total`` is the published total (a number or a marker), or ``None`` when it is
    derivable but not given here. A reader takes each ``"<5"`` as 1-4 and each ``"*"`` as 0
    or more; with a known total, or one shown ``"<5"``, a primary cell (or that total) is
    pinned when the bounds leave it a single value.
    """
    hidden = [cell for cell in cells if is_hidden(cell)]
    if total is None or not is_hidden(total):
        assert len(hidden) != 1, f"{where}: a lone suppressed cell is recoverable by subtraction"
    if total is None or total == SECONDARY or not hidden:
        return
    pinned = _pinned(cells, total)
    assert not pinned, f"{where}: {pinned} (cells {list(cells)}, total {total})"


def _counts_table_checks(
    table: pd.DataFrame, total: object, where: str, linked: bool = False
) -> None:
    """(i) and (ii) for a level-counts table. Its recorded levels sum to the non-missing
    count; for a ``linked`` table that total is published exactly when the ``"(missing)"``
    row is shown (vi), else it is taken as published (a raw column's ``n_nonnull``)."""
    assert_no_small_counts(table, where)
    assert_not_recoverable(table["n"].tolist(), total, where)
    recorded = table[table.iloc[:, 0] != MISSING_LABEL]
    missing = table.loc[table.iloc[:, 0] == MISSING_LABEL, "n"].tolist()
    if linked and missing and is_hidden(missing[0]):
        return
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
    """Percentage rows: the facilities sum to the total column.

    A shown total only forbids a lone hidden facility (counts are not published). A total
    shown ``"<5"`` means its count or its complement is 1-4. On the side that is 1-4 every
    shown facility must be 0 (0.0% or 100.0%) and each ``"<5"`` facility is 1-4, so the
    bounds must not pin them.
    """
    for _, row in table.iterrows():
        total = row[total_column]
        cells = [row[c] for c in facility_columns]
        where = f"{row.iloc[0]}: {total_column} vs {list(facility_columns)}"
        if not is_hidden(total):
            assert_not_recoverable(cells, None, where)
            continue
        if total != SUPPRESSED:
            continue
        shown = [_number(c) for c in cells if not is_hidden(c)]
        for side in (0.0, 100.0):
            if all(value == side for value in shown):
                small_side = [c if is_hidden(c) else 0 for c in cells]
                assert_not_recoverable(small_side, SUPPRESSED, f"{where} ({side:g}% side)")


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
    partial = status.set_index("status").loc["partial", "n"]
    if partial != SECONDARY:
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
            linked = (number, index) in LINKED_LEVEL_TABLES
            _counts_table_checks(table, total, f"Q{number} table {index}", linked)


def scan_variable_profile(profile: pd.DataFrame) -> None:
    assert_no_small_counts(profile, "variable_profile")
    facility_columns = [c for c in profile.columns if c.startswith("pct_missing_")]
    _wide_checks(profile, "pct_missing", facility_columns)


def assert_linked_inputs_hide_alike(profile: pd.DataFrame, completeness: pd.DataFrame) -> None:
    """(iv): same hidden cells, with the same markers, for an input and its single raw
    column, and for the ``ga_band (recorded)`` row and the one raw column both band bounds
    are mapped from."""
    names = profile["canonical_name"].fillna("")
    by_input = completeness.set_index("input")
    facilities = [c for c in completeness.columns if c not in ("input", "all")]
    columns = {"all": "pct_missing", **{f: f"pct_missing_{f}" for f in facilities}}
    pairs = [(field, names == field) for field in INPUT_FIELDS]
    pairs.append((GA_BAND_RECORDED, names == "ga_band_lower;ga_band_upper"))
    for label, matches in pairs:
        rows = profile[matches]
        if len(rows) != 1 or label not in by_input.index:
            continue
        raw = rows.iloc[0]
        in_profile = {s: raw[c] for s, c in columns.items() if is_hidden(raw[c])}
        in_completeness = {
            s: by_input.loc[label, s] for s in columns if is_hidden(by_input.loc[label, s])
        }
        assert in_profile == in_completeness, (
            f"{label}: profile hides {in_profile}, completeness {in_completeness}"
        )


def assert_hash_key_uniques_hidden(profile: pd.DataFrame) -> None:
    """(v): a raw column mapped to mother_key never shows its number of distinct values,
    a quantile or an association."""
    names = profile["canonical_name"].fillna("").str.split(";")
    for _, row in profile[names.map(lambda parts: "mother_key" in parts)].iterrows():
        assert row["n_unique"] == SECONDARY, f"{row['raw_name']}: n_unique {row['n_unique']}"
        for column in (*QUANTILE_COLUMNS, "association"):
            cell = row.get(column)
            assert pd.isna(cell) or is_hidden(cell), f"{row['raw_name']}: {column} {cell}"


def _missing_rows(open_questions: str) -> dict[str, object]:
    """Field -> its ``"(missing)"`` cell in open_questions.md ("0" when there is no row)."""
    sections = re.split(r"\n## Q(\d+)\. ", open_questions)[1:]
    out: dict[str, object] = {}
    for number, body in zip(sections[::2], sections[1::2], strict=True):
        for index, table in enumerate(markdown_tables(body)):
            field = LINKED_LEVEL_TABLES.get((number, index))
            if field is None or "n" not in table.columns:
                continue
            cells = table.loc[table.iloc[:, 0] == MISSING_LABEL, "n"].tolist()
            out[field] = cells[0] if cells else "0"
    return out


def assert_missing_copies_agree(
    profile: pd.DataFrame, completeness: pd.DataFrame, open_questions: str
) -> None:
    """(vi): each copy of a field's overall missing count is hidden alike."""
    by_input = completeness.set_index("input")
    q1 = markdown_tables(open_questions.split("## Q1.", 1)[1])[0].set_index("input")
    names = profile["canonical_name"].fillna("")
    for field, cell in _missing_rows(open_questions).items():
        copies = {"(missing)": cell}
        if field in by_input.index:
            copies["completeness all"] = by_input.loc[field, "all"]
            copies["Q1"] = q1.loc[field, "pct_recorded"]
        rows = profile[names == field]
        if len(rows) == 1:
            copies["pct_missing"] = rows.iloc[0]["pct_missing"]
            copies["n_nonnull"] = rows.iloc[0]["n_nonnull"]
        markers = {where: c if is_hidden(c) else "shown" for where, c in copies.items()}
        if cell == "0":
            continue  # nothing missing: nothing to hide
        assert len(set(markers.values())) == 1, f"{field}: {markers}"


def scan_profile_outputs(out_dir: Path) -> None:
    """Run every check over the three files ``write_profile`` publishes."""
    inputs = (out_dir / "robson_inputs.md").read_text(encoding="utf-8")
    scan_robson_inputs(inputs)
    records = int(re.search(r"Records: (\d+)\.", inputs).group(1))  # type: ignore[union-attr]
    scan_open_questions((out_dir / "open_questions.md").read_text(encoding="utf-8"), records)
    profile = pd.read_csv(out_dir / "variable_profile.csv", dtype=str)
    scan_variable_profile(profile)
    assert_hash_key_uniques_hidden(profile)
    assert_linked_inputs_hide_alike(profile, markdown_tables(inputs)[0])
    open_questions = (out_dir / "open_questions.md").read_text(encoding="utf-8")
    assert_missing_copies_agree(profile, markdown_tables(inputs)[0], open_questions)


class CountReader:
    """A reader's integer programme over published percentages (for tests).

    Each published percentage becomes an integer count within its rounding interval;
    ``"<5"`` means the count or its complement is 1-4, ``"*"`` anything. :meth:`range` is
    the smallest and largest value a linear expression of counts can take.
    """

    def __init__(self) -> None:
        self._index: dict[Hashable, int] = {}
        self._lower: list[float] = []
        self._upper: list[float] = []
        self._rows: list[tuple[dict[Hashable, float], float, float]] = []

    def var(self, key: Hashable, lower: float = 0.0, upper: float = math.inf) -> Hashable:
        if key not in self._index:
            self._index[key] = len(self._index)
            self._lower.append(lower)
            self._upper.append(upper)
        return key

    def constrain(self, coefs: Mapping[Hashable, float], lower: float, upper: float) -> None:
        self._rows.append((dict(coefs), lower, upper))

    def pct(self, key: Hashable, cell: object, n: int, counted: bool = True) -> Hashable:
        """A count out of ``n`` rows published as the percentage ``cell`` (of the count if
        ``counted``, else of its complement)."""
        x = self.var(key, 0, n)
        if cell == SECONDARY:
            return x
        if cell == SUPPRESSED:
            small = self.var((key, "small side"), 0, 1)
            big = n + 10
            # small = 1: 1 <= x <= 4; small = 0: n - 4 <= x <= n - 1.
            self.constrain({x: 1, small: -big}, 1 - big, math.inf)
            self.constrain({x: 1, small: big}, -math.inf, 4 + big)
            self.constrain({x: 1, small: big}, n - 4, math.inf)
            self.constrain({x: 1, small: -big}, -math.inf, n - 1)
            return x
        pct = float(str(cell))
        if not counted:
            pct = 100.0 - pct
        self.constrain({x: 1}, (pct - 0.05) * n / 100 - 1e-9, (pct + 0.05) * n / 100 + 1e-9)
        return x

    def range(self, coefs: Mapping[Hashable, float]) -> tuple[int, int]:
        n = len(self._index)
        matrix = np.zeros((len(self._rows), n))
        lower, upper = np.zeros(len(self._rows)), np.zeros(len(self._rows))
        for i, (row, lo, hi) in enumerate(self._rows):
            for key, c in row.items():
                matrix[i, self._index[key]] += c
            lower[i], upper[i] = lo, hi
        constraints = LinearConstraint(matrix, lower, upper)
        bounds = Bounds(np.array(self._lower), np.array(self._upper))
        out = []
        for sign in (1.0, -1.0):
            c = np.zeros(n)
            for key, w in coefs.items():
                c[self._index[key]] += sign * w
            result = milp(c, constraints=constraints, integrality=np.ones(n), bounds=bounds)
            assert result.success, "the published cells admit no counts"
            out.append(round(sign * result.fun))
        return out[0], out[1]
