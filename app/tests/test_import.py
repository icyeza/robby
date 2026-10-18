"""Importing admissions from Excel: the template, reading and validating the upload, the routes."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any
from zoneinfo import ZoneInfo

from openpyxl import Workbook, load_workbook
from sqlalchemy.orm import Session

from robby.clinical import importer
from robby.clinical.form import DetailSpec, parse_time
from robby.clinical.importer import RowProblem, find_duplicates, read_upload
from robby.clinical.xlsx_template import build_template, columns
from robby.models import Admission, AdmissionSource
from tests.factories import make_facility, make_user

SPECS = [
    DetailSpec("height_cm", "Height", "number", "cm", 120, 200, None),
    DetailSpec("hiv_positive", "HIV positive", "category", "", None, None, ("No", "Yes")),
]
ZONE = ZoneInfo("Africa/Kigali")
NOW = datetime(2026, 10, 18, 12, 0, tzinfo=ZONE)

HEADER = [
    "Patient number",
    "Patient name",
    "Admitted at",
    "Maternal age",
    "Gestational age (weeks)",
    "Parity",
    "Previous cesareans",
    "Number of fetuses",
    "Fetal presentation",
    "Onset of labour",
    "Systolic BP",
    "Diastolic BP",
    "BP not measured: reason",
    "Proteinuria",
    "Glucose (mmol/L)",
    "HIV positive",
]


def good_row(number: str = "RW-1", admitted: object = "2026-10-18 09:30") -> list[object]:
    return [
        number, "Uwase Marie", admitted, 29, 39, 2, 1, "One", "Cephalic", "Spontaneous",
        128, 84, None, "Negative", 5.2, "yes",
    ]  # fmt: skip


def make_xlsx(header: list[object], rows: list[list[object]], title: str = "Admissions") -> bytes:
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = title
    ws.append(header)
    for row in rows:
        ws.append(row)
    out = BytesIO()
    wb.save(out)
    return out.getvalue()


def read(data: bytes, filename: str = "admissions.xlsx") -> importer.ReadResult:
    return read_upload(data, filename, SPECS, ZONE, NOW)


def problems_of(data: bytes, filename: str = "admissions.xlsx") -> list[RowProblem]:
    result = read(data, filename)
    assert result.rows == []
    return result.problems


# ---------------------------------------------------------------- template


def test_template_has_headers_lists_validation_and_help() -> None:
    wb = load_workbook(BytesIO(build_template(SPECS)))
    assert wb.sheetnames == ["Admissions", "Lists", "How to fill this in"]
    sheet = wb["Admissions"]
    header = [c.value for c in next(sheet.iter_rows(min_row=1, max_row=1))]
    assert header == [c.label for c in columns(SPECS)]
    assert header[:3] == ["Patient number", "Patient name", "Admitted at"]
    assert header[-2:] == ["Height", "HIV positive"]
    lists = wb["Lists"]
    listed = {
        col[0].value: [c.value for c in col[1:] if c.value is not None]
        for col in lists.iter_cols()
        if col[0].value
    }
    assert listed["Fetal presentation"] == ["Cephalic", "Breech", "Transverse", "Oblique"]
    assert listed["HIV positive"] == ["No", "Yes"]
    assert "Height" not in listed
    presentation = header.index("Fetal presentation") + 1
    letter = sheet.cell(row=1, column=presentation).column_letter
    ranges = [str(dv.sqref) for dv in sheet.data_validations.dataValidation]
    assert any(r.startswith(f"{letter}2") for r in ranges)
    assert wb["How to fill this in"]["A1"].value


def test_column_fields_are_the_form_field_names() -> None:
    fields = [c.field for c in columns(SPECS)]
    assert fields[:3] == ["patient_number", "patient_name", "admitted_at"]
    assert {"bp_sys", "bp_dia", "bp_reason", "proteinuria", "protein_reason"} <= set(fields)
    assert {"glucose", "glucose_reason", "d_height_cm", "d_hiv_positive"} <= set(fields)
    assert [c.label for c in columns(SPECS) if c.required] == [
        "Patient number",
        "Patient name",
        "Admitted at",
    ]


def test_the_template_itself_reads_back_with_no_rows() -> None:
    assert problems_of(build_template(SPECS)) == [
        RowProblem(0, "", "The file has no admissions to import.")
    ]


# ---------------------------------------------------------------- reading


def test_a_valid_file_gives_parsed_rows() -> None:
    data = make_xlsx(HEADER, [good_row("RW-1"), good_row("RW-2", datetime(2026, 10, 17, 22, 5))])
    result = read(data)
    assert result.problems == []
    assert [n for n, _ in result.rows] == [2, 3]
    first, second = (p for _, p in result.rows)
    assert first.patient_number == "RW-1" and first.patient_name == "Uwase Marie"
    assert first.admitted_at == parse_time("2026-10-18 09:30", ZONE)
    assert second.admitted_at == datetime(2026, 10, 17, 22, 5, tzinfo=ZONE)
    assert first.presentation == "cephalic" and first.plurality == 1
    assert first.maternal_age == 29 and first.ga_weeks == 39
    assert first.details == {"hiv_positive": "Yes"}  # normalised to the trained spelling


def test_the_sheet_named_admissions_is_preferred() -> None:
    wb = Workbook()
    first = wb.active
    assert first is not None
    first.title = "Notes"
    first.append(["not", "a", "header"])
    sheet = wb.create_sheet("Admissions")
    sheet.append(HEADER)
    sheet.append(good_row())
    out = BytesIO()
    wb.save(out)
    assert len(read(out.getvalue()).rows) == 1


def test_file_level_problems() -> None:
    good = make_xlsx(HEADER, [good_row()])
    assert problems_of(good, "admissions.csv") == [RowProblem(0, "", "Upload an Excel .xlsx file.")]
    assert problems_of(b"not a workbook") == [
        RowProblem(0, "", "The file could not be read as an Excel workbook.")
    ]
    assert problems_of(b"x" * (importer.MAX_BYTES + 1)) == [
        RowProblem(0, "", "The file is larger than 2 MB.")
    ]
    many = make_xlsx(HEADER, [good_row(f"RW-{i}") for i in range(importer.MAX_ROWS + 1)])
    assert problems_of(many) == [RowProblem(0, "", "At most 500 rows can be imported at once.")]


def test_header_problems() -> None:
    unknown = make_xlsx([*HEADER[:5], "Parity!", *HEADER[6:]], [good_row()])
    assert problems_of(unknown) == [
        RowProblem(1, "Parity!", "Unknown column “Parity!”. Use the headers from the template.")
    ]
    no_name = make_xlsx([h for h in HEADER if h != "Patient name"], [])
    assert problems_of(no_name) == [
        RowProblem(1, "Patient name", "The column “Patient name” is missing.")
    ]


def test_headers_match_ignoring_case_and_spacing() -> None:
    header = [h.upper().replace(" ", "  ") for h in HEADER]
    assert len(read(make_xlsx(header, [good_row()])).rows) == 1


def test_blank_rows_are_skipped() -> None:
    data = make_xlsx(HEADER, [good_row("RW-1"), [None] * len(HEADER), good_row("RW-2")])
    assert [n for n, _ in read(data).rows] == [2, 4]


def test_row_problems_name_the_row_and_column() -> None:
    bad = good_row()
    bad[0] = None
    bad[10] = None  # systolic blank, diastolic given: the BP error
    data = make_xlsx(HEADER, [good_row(), bad])
    assert problems_of(data) == [
        RowProblem(3, "Patient number", "Enter the patient number."),
        RowProblem(
            3,
            "Systolic BP",
            "Blood pressure needs both numbers, as whole numbers (for example 128 / 84).",
        ),
    ]


def test_a_reason_stands_in_for_a_missing_value() -> None:
    row = good_row()
    row[10], row[11], row[12] = None, None, "woman declined"
    result = read(make_xlsx(HEADER, [row]))
    assert result.problems == []
    parsed = result.rows[0][1]
    assert parsed.bp_reason == "declined" and parsed.systolic is None


def test_a_value_and_a_reason_together_is_a_problem() -> None:
    row = good_row()
    row[12] = "declined"
    assert problems_of(make_xlsx(HEADER, [row])) == [
        RowProblem(
            2, "BP not measured: reason", "Enter a value or a reason it was not measured, not both."
        )
    ]


def test_cell_text() -> None:
    assert importer.cell_text(None) == ""
    assert importer.cell_text(32.0) == "32"
    assert importer.cell_text(5.2) == "5.2"
    assert importer.cell_text(7) == "7"
    assert importer.cell_text("  RW-1 ") == "RW-1"
    assert importer.cell_text(datetime(2026, 10, 18, 9, 30, 15)) == "2026-10-18 09:30"


# ---------------------------------------------------------------- duplicates


def _existing(db: Session, facility: Any, number: str, at: datetime, **kw: Any) -> Admission:
    user = make_user(db, facility=facility)
    admission = Admission(
        facility_id=facility.id,
        code=f"{facility.code}-{number}-{kw.get('code', '1')}",
        admitted_at=at,
        source=AdmissionSource.FORM,
        entered_by=user.id,
        patient_number=number,
        patient_name="Test Patient",
        supersedes_id=kw.get("supersedes"),
        correction_reason="fix" if kw.get("supersedes") else None,
    )
    db.add(admission)
    db.flush()
    return admission


def test_duplicates_in_the_file_and_in_the_facility(db: Session) -> None:
    facility = make_facility(db)
    other = make_facility(db)
    at = datetime(2026, 10, 18, 9, 30, tzinfo=ZONE)
    stored = _existing(db, facility, "RW-7", at)
    old = _existing(db, facility, "RW-8", at)
    # corrected to another admission time: only the replaced row matches, and it does not count
    _existing(db, facility, "RW-8", at.replace(hour=10), supersedes=old.id, code="2")
    _existing(db, other, "RW-9", at)
    rows = read(
        make_xlsx(
            HEADER,
            [
                good_row("RW-1"),
                good_row("RW-1"),
                good_row("RW-7"),
                good_row("RW-9"),
                good_row("RW-8"),
            ],
        )
    ).rows
    assert len(rows) == 5
    assert find_duplicates(db, facility.id, rows) == [
        RowProblem(3, "Patient number", "Same patient and admission time as row 2."),
        RowProblem(4, "Patient number", f"Already recorded as {stored.code}."),
    ]
