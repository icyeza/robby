"""Importing admissions from Excel: the template, reading and validating the upload, the routes."""

from __future__ import annotations

from io import BytesIO

from openpyxl import load_workbook

from robby.clinical.form import DetailSpec
from robby.clinical.xlsx_template import build_template, columns

SPECS = [
    DetailSpec("height_cm", "Height", "number", "cm", 120, 200, None),
    DetailSpec("hiv_positive", "HIV positive", "category", "", None, None, ("No", "Yes")),
]


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
