import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robson_ml.mapping import MappingError, apply_mapping, load_mapping
from robson_ml.schema import validate_canonical

MAPPING_YAML = """
source: test
sheet: main
fields:
  admission_id: {kind: row_key, status: confirmed}
  facility_id: {raw: Facility, kind: text, status: confirmed}
  parity: {raw: Parity, kind: integer, status: confirmed, range: [0, 20]}
  previous_cs_count:
    raw: "Previous CS"
    kind: category
    dtype: Int64
    status: review
    note: "count bands"
    levels: {"None": 0, "One": 1, "Two or more": 2, "Unknown": ~}
  gestational_age_weeks:
    raw: GA
    kind: gestational_age
    format: weeks_plus_days
    status: confirmed
    range: [20, 45]
  maternal_age: {raw: Age, kind: float, status: confirmed, range: [12, 55]}
  admitted_at: {raw: When, kind: datetime, status: confirmed}
  mode_of_delivery: {raw: Mode, kind: text, status: confirmed}
  cs:
    raw: Mode
    kind: category
    dtype: Int64
    status: confirmed
    levels: {"Caesarean section": 1, "Normal vaginal": 0, "Vacuum": 0}
"""


def _raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Facility": ["Alpha", "Beta", "Alpha", "Beta", "Alpha", "Alpha"],
            "Parity": [0, "2", 1.0, None, "x", 3],
            "Previous CS": ["None", "One", "Two or more", "One", "Unknown", "Blah"],
            "GA": ["38+2", "36+6", 39, "  ", "abc", "41+0"],
            "Age": [25, 60, 30.5, None, 22, 19],
            "When": ["2023-11-02T10:00:00", None, "2024-01-05T08:30:00", "bad", None, None],
            "Mode": [
                "Caesarean section",
                "Normal vaginal",
                "Vacuum",
                "Caesarean section",
                None,
                "Normal vaginal",
            ],
            "Note": list("abcdef"),
        },
        dtype=object,
    )


@pytest.fixture
def mapping_path(tmp_path: Path) -> Path:
    path = tmp_path / "mapping.yaml"
    path.write_text(MAPPING_YAML, encoding="utf-8")
    return path


def test_apply_mapping_values(mapping_path: Path) -> None:
    canonical, _ = apply_mapping(_raw(), load_mapping(mapping_path))
    assert canonical["admission_id"].tolist()[:2] == ["ADM000000", "ADM000001"]
    assert canonical["parity"].tolist()[:3] == [0, 2, 1]
    assert canonical["parity"].isna().tolist() == [False, False, False, True, True, False]
    assert canonical["previous_cs_count"].tolist()[:4] == [0, 1, 2, 1]
    assert canonical["previous_cs_count"].isna().tolist()[4:] == [True, True]
    ga = canonical["gestational_age_weeks"]
    assert ga.iloc[0] == pytest.approx(38 + 2 / 7)
    assert ga.iloc[1] == pytest.approx(36 + 6 / 7)
    assert ga.iloc[2] == pytest.approx(39.0)
    assert np.isnan(ga.iloc[3]) and np.isnan(ga.iloc[4])
    assert np.isnan(canonical["maternal_age"].iloc[1])
    assert canonical["cs"].tolist()[:4] == [1, 0, 0, 1]
    assert canonical["admitted_at"].notna().sum() == 2
    validate_canonical(canonical)


def test_mapping_report_counts(mapping_path: Path) -> None:
    _, report = apply_mapping(_raw(), load_mapping(mapping_path))
    by = {f.canonical: f for f in report.fields}
    assert (by["parity"].n_raw_nonnull, by["parity"].n_unparsed) == (5, 1)
    assert by["previous_cs_count"].n_unparsed == 1
    assert "Blah" not in str(by["previous_cs_count"].unmapped_levels)
    assert by["gestational_age_weeks"].n_unparsed == 1
    assert by["maternal_age"].n_out_of_range == 1
    assert by["admitted_at"].n_unparsed == 1
    assert "plurality" in report.missing_canonical_fields
    assert report.unreferenced_raw_columns == ["Note"]
    assert report.review_fields == ["previous_cs_count"]


def test_unknown_canonical_field_rejected(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text("fields:\n  birth_weight: {raw: BW, kind: float}\n", encoding="utf-8")
    with pytest.raises(MappingError, match="birth_weight"):
        load_mapping(path)


def test_unquoted_yes_no_levels_rejected(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text(
        "fields:\n  gdm_recorded:\n    raw: GDM\n    kind: category\n"
        "    levels: {Yes: yes, No: no}\n",
        encoding="utf-8",
    )
    with pytest.raises(MappingError, match="quote"):
        load_mapping(path)


def test_bad_kind_rejected(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text("fields:\n  parity: {raw: P, kind: guess}\n", encoding="utf-8")
    with pytest.raises(MappingError, match="kind"):
        load_mapping(path)


def test_empty_mapping_loads(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text("source: x\nsheet: ~\nfields: {}\n", encoding="utf-8")
    assert load_mapping(path).fields == {}


def test_missing_raw_column_raises(mapping_path: Path) -> None:
    raw = _raw().drop(columns=["GA"])
    with pytest.raises(MappingError, match="GA"):
        apply_mapping(raw, load_mapping(mapping_path))


def test_ga_completed_weeks_format(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text(
        "fields:\n"
        "  admission_id: {kind: row_key, status: confirmed}\n"
        "  gestational_age_weeks: {raw: GA, kind: gestational_age, format: completed_weeks, "
        "status: confirmed}\n",
        encoding="utf-8",
    )
    raw = pd.DataFrame({"GA": [36, "38", None]}, dtype=object)
    canonical, _ = apply_mapping(raw, load_mapping(path))
    assert canonical["gestational_age_weeks"].tolist()[:2] == [36.0, 38.0]
    assert np.isnan(canonical["gestational_age_weeks"].iloc[2])


def test_ga_two_column_weeks_plus_days(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text(
        "fields:\n"
        "  admission_id: {kind: row_key, status: confirmed}\n"
        "  gestational_age_weeks:\n"
        "    raw: [W, D]\n"
        "    kind: gestational_age\n"
        "    format: weeks_plus_days\n"
        "    status: confirmed\n",
        encoding="utf-8",
    )
    raw = pd.DataFrame({"W": [37, 40], "D": [3, None]}, dtype=object)
    canonical, _ = apply_mapping(raw, load_mapping(path))
    ga = canonical["gestational_age_weeks"]
    assert ga.iloc[0] == pytest.approx(37 + 3 / 7)
    assert ga.iloc[1] == pytest.approx(40.0)


def test_report_serialises_without_raw_values(mapping_path: Path) -> None:
    _, report = apply_mapping(_raw(), load_mapping(mapping_path))
    payload = json.dumps(dataclasses.asdict(report), default=str)
    # Check for exact raw cell values as JSON string tokens (quoted), not bare substrings:
    # a bare "x" or "text" would also match legitimate structural words like the "text"
    # kind or the literal English word "text" in a free-text placeholder.
    for forbidden in ("Alpha", "x", "abc", "bad", "60"):
        assert f'"{forbidden}"' not in payload
