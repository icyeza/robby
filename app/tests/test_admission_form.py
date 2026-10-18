"""Reading the admission form: the patient number and name."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from robby.clinical.form import Parsed, parse

ZONE = ZoneInfo("Africa/Kigali")
NOW = datetime(2026, 10, 18, 12, 0, tzinfo=ZONE)
GOOD = {
    "patient_number": "RW-0042",
    "patient_name": "Uwase Marie",
    "admitted_at": "18 Oct 2026, 11:30",
    "parity": "1",
    "bp_sys": "120",
    "bp_dia": "80",
    "proteinuria": "negative",
    "glucose": "5.0",
}


def parsed_with(**changes: str) -> tuple[Parsed, dict[str, str]]:
    return parse({**GOOD, **changes}, [], ZONE, NOW)


def test_valid_identifiers_are_kept() -> None:
    parsed, errors = parsed_with()
    assert errors == {}
    assert (parsed.patient_number, parsed.patient_name) == ("RW-0042", "Uwase Marie")


BAD_NUMBER = "Patient number must be 1–32 letters, digits, / or -."  # noqa: RUF001


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", "Enter the patient number."),
        ("DA 12", BAD_NUMBER),
        ("A" * 33, BAD_NUMBER),
        ("DA#12", BAD_NUMBER),
    ],
)
def test_bad_patient_numbers_are_rejected(value: str, message: str) -> None:
    parsed, errors = parsed_with(patient_number=value)
    assert errors == {"patient_number": message}
    assert parsed.patient_number is None


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", "Enter the patient name."),
        ("   ", "Enter the patient name."),
        ("x" * 121, "Patient name must be at most 120 characters."),
    ],
)
def test_bad_patient_names_are_rejected(value: str, message: str) -> None:
    parsed, errors = parsed_with(patient_name=value)
    assert errors == {"patient_name": message}
    assert parsed.patient_name is None


def test_a_32_character_number_and_120_character_name_are_accepted() -> None:
    parsed, errors = parsed_with(patient_number="A" * 32, patient_name="x" * 120)
    assert errors == {}
    assert parsed.patient_number == "A" * 32 and parsed.patient_name == "x" * 120


def test_identifiers_are_trimmed_and_name_spaces_collapsed() -> None:
    parsed, errors = parsed_with(patient_number=" RW/123-4 ", patient_name="  Uwase   Marie ")
    assert errors == {}
    assert (parsed.patient_number, parsed.patient_name) == ("RW/123-4", "Uwase Marie")


def test_preview_does_not_ask_for_identifiers() -> None:
    _, errors = parse(
        {**GOOD, "patient_number": "", "patient_name": ""},
        [],
        ZONE,
        NOW,
        screening_required=False,
        patient_required=False,
    )
    assert errors == {}


def test_a_changed_name_makes_the_parsed_form_differ() -> None:
    assert parsed_with()[0] != parsed_with(patient_name="Uwase Marie Claire")[0]
