"""Importing an OpenClinic export (assumed layout, see robby.clinical.openclinic)."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from robby.clinical.form import omissions_of
from robby.clinical.importer import read_upload
from tests.test_import import make_xlsx

ZONE = ZoneInfo("Africa/Kigali")
NOW = datetime(2026, 10, 18, 12, 0, tzinfo=ZONE)
HEADER = [
    "Patient ID", "First name", "Last name", "Date of birth", "Admission date", "Ward", "Para",
    "Previous C-sections", "Gestational age", "Pregnancy type", "Presentation", "Labour onset",
    "Blood pressure", "Urine protein", "Glucose (mg/dL)",
]  # fmt: skip


def oc_row(**over: object) -> list[object]:
    row: dict[str, object] = {
        "Patient ID": "OC-1001", "First name": "Marie", "Last name": "Uwase",
        "Date of birth": "12/03/1997", "Admission date": "18/10/2026 09:30", "Ward": "Maternity",
        "Para": 2, "Previous C-sections": 1, "Gestational age": "39+2",
        "Pregnancy type": "Singleton", "Presentation": "Vertex", "Labour onset": "Spontaneous",
        "Blood pressure": "128/84", "Urine protein": "neg", "Glucose (mg/dL)": 94,
    }  # fmt: skip
    row.update(over)
    return [row[h] for h in HEADER]


def read(rows: list[list[object]], header: list[object] = HEADER):
    return read_upload(make_xlsx(header, rows), "export.xlsx", [], ZONE, NOW)


def test_an_openclinic_export_is_translated_and_checked() -> None:
    result = read([oc_row()])
    assert result.problems == []
    parsed = result.rows[0][1]
    assert (parsed.patient_number, parsed.patient_name) == ("OC-1001", "Marie Uwase")
    assert (parsed.parity, parsed.previous_cs_count, parsed.plurality) == (2, 1, 1)
    assert parsed.presentation == "cephalic"
    assert parsed.onset == "spontaneous"
    assert (parsed.systolic, parsed.diastolic) == (128, 84)
    assert parsed.maternal_age == 29
    assert parsed.proteinuria == "negative"
    assert round(float(parsed.glucose), 1) == 5.2  # 94 mg/dL
    assert parsed.ga_weeks == 39


def test_unused_columns_are_ignored_and_missing_tests_get_a_reason() -> None:
    header = [*HEADER[:12], "Notes"]
    row = [*oc_row()[:12], "free text"]
    result = read([row], header)
    assert result.problems == []
    assert {f.value for f, _ in omissions_of(result.rows[0][1])} == {
        "blood_pressure",
        "proteinuria",
        "glucose",
    }


def test_bad_values_name_the_openclinic_column() -> None:
    result = read([oc_row(), oc_row(**{"Presentation": "upside down", "Admission date": "soon"})])
    assert [(p.row, p.column) for p in result.problems] == [
        (3, "Admission date"),
        (3, "Presentation"),
    ]
    assert result.rows == []


def test_missing_required_openclinic_columns() -> None:
    header = [h for h in HEADER if h != "Patient ID"]
    result = read([[v for h, v in zip(HEADER, oc_row(), strict=True) if h != "Patient ID"]], header)
    assert [p.column for p in result.problems] == ["Patient ID"]


def test_missing_required_columns_refuse_the_file_when_bad_rows_are_skipped() -> None:
    # The web upload skips bad rows; with no translatable sheet it used to fail with IndexError.
    header = [h for h in HEADER if h != "Patient ID"]
    row = [v for h, v in zip(HEADER, oc_row(), strict=True) if h != "Patient ID"]
    result = read_upload(make_xlsx(header, [row]), "export.xlsx", [], ZONE, NOW, skip_bad=True)
    assert [p.column for p in result.problems] == ["Patient ID"]
    assert result.rows == []


def test_the_apps_own_template_is_not_mistaken_for_openclinic() -> None:
    result = read([["x"] * 3], ["Patient number", "Patient name", "Admitted at", "Para", "Age"])
    assert {p.column for p in result.problems} == {"Para", "Age"}  # template rules, not OpenClinic


def test_iso_dates_with_fractions_are_accepted() -> None:
    for text in ("2026-10-18 09:30:00.250000", "2026-10-18T09:30:00", "2026-10-18"):
        assert read([oc_row(**{"Admission date": text})]).problems == [], text
