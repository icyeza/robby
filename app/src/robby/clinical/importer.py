"""Reading an uploaded Excel file of admissions (one row per admission). Every row goes through
the same form.parse as the web form; the file is imported only when every row passes."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from io import BytesIO
from typing import Any
from zoneinfo import ZoneInfo

from openpyxl import load_workbook
from sqlalchemy import exists, select
from sqlalchemy.orm import Session, aliased

from robby.clinical.form import DetailSpec, Parsed, parse
from robby.clinical.xlsx_template import SHEET, Column, columns
from robby.models import Admission

MAX_BYTES = 2 * 1024 * 1024
MAX_ROWS = 500
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
    cols: list[Column], header: Sequence[object]
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
    data: bytes, filename: str, specs: list[DetailSpec], zone: ZoneInfo, now: datetime
) -> ReadResult:
    """Every row parsed, or every problem found (and no rows)."""
    if not filename.lower().endswith(".xlsx"):
        return _file_problem("Upload an Excel .xlsx file.")
    if len(data) > MAX_BYTES:
        return _file_problem("The file is larger than 2 MB.")
    table = _sheet_rows(data)
    if table is None:
        return _file_problem("The file could not be read as an Excel workbook.")
    if not table:
        return _file_problem("The file has no header row.")
    cols = columns(specs)
    index, problems = header_index(cols, table[0])
    if problems:
        return ReadResult([], problems)
    body = [
        (n, cells) for n, cells in enumerate(table[1:], start=2) if any(cell_text(c) for c in cells)
    ]
    if len(body) > MAX_ROWS:
        return _file_problem(f"At most {MAX_ROWS} rows can be imported at once.")
    if not body:
        return _file_problem("The file has no admissions to import.")
    labels = {c.field: c.label for c in cols}
    rows: list[tuple[int, Parsed]] = []
    for n, cells in body:
        values = row_values(cols, index, cells)
        parsed, errors = parse(values, specs, zone, now)
        problems += _row_problems(n, values, errors, labels)
        rows.append((n, parsed))
    return ReadResult([] if problems else rows, problems)


def find_duplicates(
    db: Session, facility_id: uuid.UUID, rows: list[tuple[int, Parsed]]
) -> list[RowProblem]:
    """Rows whose patient number and admission time repeat an earlier row of the file, or an
    admission of this facility that has not been replaced."""
    replacement = aliased(Admission)
    numbers = {p.patient_number for _, p in rows if p.patient_number}
    stored: dict[tuple[Any, ...], str] = {
        (a.patient_number, a.admitted_at): a.code
        for a in db.scalars(
            select(Admission).where(
                Admission.facility_id == facility_id,
                Admission.patient_number.in_(numbers),
                ~exists().where(replacement.supersedes_id == Admission.id),
            )
        )
    }
    seen: dict[tuple[Any, ...], int] = {}
    problems: list[RowProblem] = []
    for n, parsed in rows:
        key = (parsed.patient_number, parsed.admitted_at)
        if key in seen:
            message = f"Same patient and admission time as row {seen[key]}."
        elif key in stored:
            message = f"Already recorded as {stored[key]}."
        else:
            seen[key] = n
            continue
        problems.append(RowProblem(n, "Patient number", message))
    return problems
