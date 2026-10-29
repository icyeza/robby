"""Reading an uploaded Excel file of admissions (one row per admission). The sheet may be in the
app's template, an OpenClinic export or the research export (each a SheetLayout, translated into
the template layout); every row then goes through the same form.parse as the web form. By default
the file is imported only when every row passes; with ``skip_bad`` (the web upload) rows with
problems are left out and listed, and the rest are imported."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from io import BytesIO
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from openpyxl import load_workbook
from sqlalchemy import exists, select
from sqlalchemy.orm import Session, aliased

from robby.clinical import openclinic, research_export
from robby.clinical.form import MAX_PAST, DetailSpec, Parsed, parse
from robby.clinical.xlsx_template import SHEET, Column, columns
from robby.models import Admission, patient_number_index

MAX_ROWS = 2000
SCREENING = (("bp", ("bp_sys", "bp_dia")), ("protein", ("proteinuria",)), ("glucose", ("glucose",)))
# Form errors that belong to no single field, shown against this column.
ERROR_COLUMN = {"bp": "bp_sys"}


@dataclass(frozen=True)
class RowProblem:
    row: int  # Excel row number (header is row 1); 0 for problems with the whole file
    column: str  # column label, or "" for whole-row/file problems
    message: str


@dataclass(frozen=True)
class ReadResult:
    rows: list[tuple[int, Parsed]]  # (Excel row number, parsed admission), only if no problems
    problems: list[RowProblem]
    # Research export only: (DeliveryMode value, delivered at) by Excel row, and the number of
    # rows of other hospitals that were left out.
    outcomes: dict[int, tuple[str, datetime]] = field(default_factory=dict)
    other_hospitals: int = 0


def _file_problem(message: str) -> ReadResult:
    return ReadResult([], [RowProblem(0, "", message)])


def _header_key(value: object) -> str:
    return " ".join(str(value).split()).lower()


def cell_text(value: object) -> str:
    """A cell as the text the form would have received."""
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _choice(text: str, options: tuple[tuple[str, str], ...]) -> str:
    """The stored value for a cell that matches an option's value or label, ignoring case."""
    wanted = text.lower()
    for value, shown in options:
        if wanted in (value.lower(), shown.lower()):
            return value
    return text


def header_index(
    cols: list[Column], header: Sequence[object], ignore_unknown: bool = False
) -> tuple[dict[str, int], list[RowProblem]]:
    """Field name -> cell index for each known header, and the problems with row 1."""
    by_key = {_header_key(c.label): c for c in cols}
    index: dict[str, int] = {}
    problems: list[RowProblem] = []
    for i, raw in enumerate(header):
        if raw is None or not str(raw).strip():
            continue
        col = by_key.get(_header_key(raw))
        shown = " ".join(str(raw).split())
        if col is None and ignore_unknown:
            continue
        if col is None:
            problems.append(
                RowProblem(
                    1, shown, f"Unknown column “{shown}”. Use the headers from the template."
                )
            )
        elif col.field in index:
            problems.append(RowProblem(1, col.label, f"The column “{col.label}” appears twice."))
        else:
            index[col.field] = i
    for col in cols:
        if col.required and col.field not in index:
            problems.append(RowProblem(1, col.label, f"The column “{col.label}” is missing."))
    return index, problems


def row_values(
    cols: list[Column], header_index: dict[str, int], cells: tuple[object, ...]
) -> dict[str, str]:
    """One row as form values: choices as stored values, "not measured" set by a reason."""
    values: dict[str, str] = {}
    for col in cols:
        i = header_index.get(col.field)
        if i is None or i >= len(cells):
            continue
        text = cell_text(cells[i])
        values[col.field] = _choice(text, col.options) if text and col.options else text
    for key, _ in SCREENING:
        if values.get(f"{key}_reason"):
            values[f"{key}_nm"] = "1"
    return values


def _row_problems(
    row: int, values: dict[str, str], errors: dict[str, str], labels: dict[str, str]
) -> list[RowProblem]:
    problems = [
        RowProblem(row, labels.get(ERROR_COLUMN.get(key, key), ""), message)
        for key, message in errors.items()
    ]
    for key, fields in SCREENING:
        if values.get(f"{key}_reason") and any(values.get(f) for f in fields):
            problems.append(
                RowProblem(
                    row,
                    labels[f"{key}_reason"],
                    "Enter a value or a reason it was not measured, not both.",
                )
            )
    return problems


def _sheet_rows(data: bytes) -> list[tuple[object, ...]] | None:
    try:
        wb = load_workbook(BytesIO(data), read_only=True, data_only=True)
        try:
            sheet = wb[SHEET] if SHEET in wb.sheetnames else wb.worksheets[0]
            return [tuple(r) for r in sheet.iter_rows(values_only=True)]
        finally:
            wb.close()
    except Exception:
        return None


def read_upload(
    data: bytes,
    filename: str,
    specs: list[DetailSpec],
    zone: ZoneInfo,
    now: datetime,
    *,
    skip_bad: bool = False,
    facility_keys: set[str] | None = None,
) -> ReadResult:
    """Every row parsed, or every problem found (and no rows; with skip_bad, the good rows)."""
    if not filename.lower().endswith(".xlsx"):
        return _file_problem("Upload an Excel .xlsx file.")
    table = _sheet_rows(data)
    if table is None:
        return _file_problem("The file could not be read as an Excel workbook.")
    return read_table(table, specs, zone, now, skip_bad=skip_bad, facility_keys=facility_keys)


@dataclass(frozen=True)
class Translated:
    """A sheet in the template layout, with what its own layout found on the way."""

    table: list[tuple[object, ...]]
    problems: list[RowProblem]
    outcomes: dict[int, tuple[str, datetime]] = field(default_factory=dict)
    other_hospitals: int = 0
    max_past: timedelta | None = MAX_PAST  # how old an admission time may be


class SheetLayout(Protocol):
    """One spreadsheet layout the import accepts. Each translates its sheet into the template
    layout, so that every row then goes through the same header check and form.parse."""

    def matches(self, header: Sequence[object], cols: list[Column]) -> bool: ...

    def translate(
        self, table: list[tuple[object, ...]], cols: list[Column], skip_bad: bool
    ) -> Translated | ReadResult:
        """The translated sheet, or the ReadResult refusing the whole file."""
        ...


class TemplateLayout:
    """The app's own template (and a JSON batch with its column names): nothing to translate."""

    def matches(self, header: Sequence[object], cols: list[Column]) -> bool:
        return True

    def translate(
        self, table: list[tuple[object, ...]], cols: list[Column], skip_bad: bool
    ) -> Translated:
        return Translated(table, [])


class OpenClinicLayout:
    """An OpenClinic export. Its problems refuse the file unless bad rows are skipped."""

    def matches(self, header: Sequence[object], cols: list[Column]) -> bool:
        return openclinic.looks_like_openclinic(header, cols)

    def translate(
        self, table: list[tuple[object, ...]], cols: list[Column], skip_bad: bool
    ) -> Translated | ReadResult:
        converted, found = openclinic.convert(table, cols)
        problems = [RowProblem(*p) for p in found]
        # No translated sheet at all (required columns missing): refuse the file even when bad
        # rows are skipped, since there is no row to keep.
        if problems and (not skip_bad or not converted):
            return ReadResult([], problems)
        return Translated(converted, problems)


@dataclass(frozen=True)
class ResearchExportLayout:
    """The UR-CMHS research export: past deliveries of several hospitals, with their outcomes."""

    facility_keys: frozenset[str]  # the uploader's hospital: its name and model facility key

    def matches(self, header: Sequence[object], cols: list[Column]) -> bool:
        return research_export.looks_like_research_export(header)

    def translate(
        self, table: list[tuple[object, ...]], cols: list[Column], skip_bad: bool
    ) -> Translated | ReadResult:
        converted, found, outcomes, other = research_export.convert(
            table, cols, set(self.facility_keys)
        )
        problems = [RowProblem(*p) for p in found]
        if not outcomes and not problems:
            return _file_problem(
                "The file has no deliveries of this hospital "
                f"({other} rows of other hospitals were left out)."
            )
        return Translated(converted, problems, outcomes, other, max_past=None)


def layouts(facility_keys: set[str] | None = None) -> tuple[SheetLayout, ...]:
    """The accepted layouts in the order they are tried; the template takes what is left."""
    return (
        ResearchExportLayout(frozenset(facility_keys or ())),
        OpenClinicLayout(),
        TemplateLayout(),
    )


def _body(table: list[tuple[object, ...]]) -> list[tuple[int, tuple[object, ...]]]:
    """The non-blank rows under the header, with their Excel row numbers."""
    return [
        (n, cells) for n, cells in enumerate(table[1:], start=2) if any(cell_text(c) for c in cells)
    ]


def _parse_rows(
    body: list[tuple[int, tuple[object, ...]]],
    cols: list[Column],
    index: dict[str, int],
    specs: list[DetailSpec],
    zone: ZoneInfo,
    now: datetime,
    max_past: timedelta | None,
) -> tuple[list[tuple[int, Parsed]], list[RowProblem]]:
    """Every row through form.parse, and the problems found in each."""
    labels = {c.field: c.label for c in cols}
    rows: list[tuple[int, Parsed]] = []
    problems: list[RowProblem] = []
    for n, cells in body:
        values = row_values(cols, index, cells)
        parsed, errors = parse(values, specs, zone, now, max_past=max_past)
        problems += _row_problems(n, values, errors, labels)
        rows.append((n, parsed))
    return rows, problems


def read_table(
    table: list[tuple[object, ...]],
    specs: list[DetailSpec],
    zone: ZoneInfo,
    now: datetime,
    *,
    skip_bad: bool = False,
    facility_keys: set[str] | None = None,
) -> ReadResult:
    """The header and rows of a sheet (or of a JSON batch), in any of the accepted layouts."""
    if not table:
        return _file_problem("The file has no header row.")
    cols = columns(specs)
    layout = next(lay for lay in layouts(facility_keys) if lay.matches(table[0], cols))
    sheet = layout.translate(table, cols, skip_bad)
    if isinstance(sheet, ReadResult):
        return sheet
    index, header_problems = header_index(cols, sheet.table[0], ignore_unknown=skip_bad)
    if header_problems:
        return ReadResult([], header_problems)
    body = _body(sheet.table)
    if len(body) > MAX_ROWS:
        return _file_problem(f"At most {MAX_ROWS} rows can be imported at once.")
    if not body:
        return _file_problem("The file has no admissions to import.")
    rows, row_problems = _parse_rows(body, cols, index, specs, zone, now, sheet.max_past)
    problems = sheet.problems + row_problems
    outcomes, other = sheet.outcomes, sheet.other_hospitals
    if skip_bad:
        bad_rows = {p.row for p in problems}
        kept = [(n, p) for n, p in rows if n not in bad_rows]
        return ReadResult(kept, problems, {n: outcomes[n] for n, _ in kept if n in outcomes}, other)
    return ReadResult([] if problems else rows, problems, outcomes, other)


@dataclass(frozen=True)
class DuplicateScan:
    in_file: list[RowProblem]  # a row repeating an earlier row of the same file: a file error
    recorded: dict[int, Admission]  # row -> the facility's existing admission it repeats


def find_duplicates(
    db: Session, facility_id: uuid.UUID, rows: list[tuple[int, Parsed]]
) -> DuplicateScan:
    """Rows whose patient number and admission time repeat an earlier row of the file (a problem),
    or an admission of this facility that has not been replaced (already recorded: shown with its
    existing admission, not refused). Numbers are compared by blind index (robby.crypto), so
    case and spacing do not matter."""
    replacement = aliased(Admission)
    indexes = {
        n: patient_number_index(facility_id, p.patient_number) if p.patient_number else None
        for n, p in rows
    }
    stored: dict[tuple[Any, ...], Admission] = {
        (a.patient_number_index, a.admitted_at): a
        for a in db.scalars(
            select(Admission).where(
                Admission.facility_id == facility_id,
                Admission.patient_number_index.in_({i for i in indexes.values() if i}),
                ~exists().where(replacement.supersedes_id == Admission.id),
            )
        )
    }
    seen: dict[tuple[Any, ...], int] = {}
    problems: list[RowProblem] = []
    recorded: dict[int, Admission] = {}
    for n, parsed in rows:
        key = (indexes[n], parsed.admitted_at)
        if key in stored:
            recorded[n] = stored[key]
        elif key in seen:
            message = f"Same patient and admission time as row {seen[key]}."
            problems.append(RowProblem(n, "Patient number", message))
        else:
            seen[key] = n
    return DuplicateScan(problems, recorded)
