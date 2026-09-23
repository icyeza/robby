import dataclasses
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from robson_ml.mapping import MappingError, apply_mapping, load_mapping
from robson_ml.schema import CANONICAL_DTYPES, validate_canonical

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


ROW_KEY = "  admission_id: {kind: row_key, status: confirmed}\n"


def _run(tmp_path: Path, fields: str, raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = tmp_path / "m.yaml"
    path.write_text("fields:\n" + ROW_KEY + fields, encoding="utf-8")
    canonical, report = apply_mapping(raw, load_mapping(path))
    return canonical, {f.canonical: f for f in report.fields}


def _load_fields(tmp_path: Path, fields: str) -> None:
    path = tmp_path / "m.yaml"
    path.write_text("fields:\n" + fields, encoding="utf-8")
    load_mapping(path)


GA_WPD = "  gestational_age_weeks: {raw: G, kind: gestational_age, format: weeks_plus_days, "
GA_TWO = "  gestational_age_weeks: {raw: [W, D], kind: gestational_age, format: weeks_plus_days, "

# (canonical field, fields yaml, raw frame,
#  expected (n_raw_nonnull, n_mapped, n_unparsed, n_out_of_range, n_explicit_missing))
KIND_CASES = {
    "row_key": ("admission_id", "", pd.DataFrame({"X": [1, 2, 3]}), (3, 3, 0, 0, 0)),
    "text": (
        "facility_id",
        "  facility_id: {raw: F, kind: text}\n",
        pd.DataFrame({"F": ["A", "  ", None, "B", 5, pd.NA]}, dtype=object),
        (3, 3, 0, 0, 0),
    ),
    "integer": (
        "parity",
        "  parity: {raw: P, kind: integer, range: [0, 20]}\n",
        pd.DataFrame(
            {"P": [0, "2", 1.0, None, "x", 3, 25, 1.5, True, np.inf, 10**20, "inf"]},
            dtype=object,
        ),
        (11, 4, 6, 1, 0),
    ),
    "float": (
        "maternal_age",
        "  maternal_age: {raw: A, kind: float, range: [12, 55]}\n",
        pd.DataFrame(
            {"A": [25, "30.5", 60, None, "abc", -np.inf, False, np.float64(40)]}, dtype=object
        ),
        (7, 3, 3, 1, 0),
    ),
    "category": (
        "previous_cs_count",
        '  previous_cs_count: {raw: X, kind: category, levels: {"None": 0, "One": 1, '
        '"Unknown": ~}}\n',
        pd.DataFrame({"X": ["None", "One", "Unknown", "Unknown", None, "Blah"]}, dtype=object),
        (5, 2, 1, 0, 2),
    ),
    "datetime": (
        "admitted_at",
        "  admitted_at: {raw: W, kind: datetime}\n",
        pd.DataFrame(
            {
                "W": [
                    "2023-11-02T08:30:00",
                    "2023-11-02T06:30:00Z",
                    "bad",
                    None,
                    45000,
                    pd.Timestamp("2024-01-01 10:00"),
                ]
            },
            dtype=object,
        ),
        (5, 3, 2, 0, 0),
    ),
    "ga_decimal_weeks": (
        "gestational_age_weeks",
        "  gestational_age_weeks: {raw: G, kind: gestational_age, format: decimal_weeks, "
        "range: [20, 45]}\n",
        pd.DataFrame({"G": [38.5, "39", "x", 50, None]}, dtype=object),
        (4, 2, 1, 1, 0),
    ),
    "ga_weeks_plus_days": (
        "gestational_age_weeks",
        GA_WPD + "range: [20, 45]}\n",
        pd.DataFrame({"G": ["38+2", 39, "38.2", 38.2, "38+7", 50, None, "40"]}, dtype=object),
        (7, 3, 3, 1, 0),
    ),
    "ga_two_columns": (
        "gestational_age_weeks",
        GA_TWO + "range: [20, 45]}\n",
        pd.DataFrame(
            {
                "W": [37, 40, None, 38, 38, 38.5, 38, None, 50, "  "],
                "D": [3, None, 4, 9, "abc", 2, 2.5, None, 1, "  "],
            },
            dtype=object,
        ),
        (8, 2, 5, 1, 0),
    ),
    "ga_completed_weeks": (
        "gestational_age_weeks",
        "  gestational_age_weeks: {raw: G, kind: gestational_age, format: completed_weeks, "
        "range: [20, 45]}\n",
        pd.DataFrame({"G": [36, "38", 38.5, "38.5", None, 10]}, dtype=object),
        (5, 2, 2, 1, 0),
    ),
}


@pytest.mark.parametrize("case", list(KIND_CASES))
def test_every_present_value_is_counted_once(tmp_path: Path, case: str) -> None:
    name, fields, raw, expected = KIND_CASES[case]
    canonical, by = _run(tmp_path, fields, raw)
    f = by[name]
    counts = (f.n_raw_nonnull, f.n_mapped, f.n_unparsed, f.n_out_of_range, f.n_explicit_missing)
    assert counts == expected
    assert f.n_raw_nonnull == f.n_mapped + f.n_unparsed + f.n_out_of_range + f.n_explicit_missing
    assert int(canonical[name].notna().sum()) == f.n_mapped
    assert str(canonical[name].dtype) == CANONICAL_DTYPES[name]


def test_explicit_missing_level_is_not_a_mapped_value(tmp_path: Path) -> None:
    _, fields, raw, _ = KIND_CASES["category"]
    canonical, by = _run(tmp_path, fields, raw)
    assert canonical["previous_cs_count"].tolist()[:2] == [0, 1]
    assert canonical["previous_cs_count"].isna().tolist() == [False, False] + [True] * 4
    assert by["previous_cs_count"].n_explicit_missing == 2
    assert by["previous_cs_count"].n_unparsed == 1


def test_text_kind_never_turns_missing_markers_into_strings(tmp_path: Path) -> None:
    _, fields, raw, _ = KIND_CASES["text"]
    canonical, _ = _run(tmp_path, fields, raw)
    assert canonical["facility_id"].tolist() == ["A", None, None, "B", "5", None]


def test_ga_two_columns_bad_or_fractional_days_are_unparsed(tmp_path: Path) -> None:
    _, fields, raw, _ = KIND_CASES["ga_two_columns"]
    canonical, by = _run(tmp_path, fields, raw)
    ga = canonical["gestational_age_weeks"]
    assert ga.iloc[0] == pytest.approx(37 + 3 / 7)
    assert ga.iloc[1] == pytest.approx(40.0)
    assert ga.iloc[2:].isna().all()
    assert by["gestational_age_weeks"].n_days_blank == 1


def test_ga_weeks_plus_days_rejects_fractional_values(tmp_path: Path) -> None:
    raw = pd.DataFrame({"G": ["38.2", 38.2, "38+2", 38, "38.0"]}, dtype=object)
    canonical, by = _run(tmp_path, GA_WPD + "status: confirmed}\n", raw)
    ga = canonical["gestational_age_weeks"]
    assert ga.isna().tolist() == [True, True, False, False, False]
    assert ga.iloc[2] == pytest.approx(38 + 2 / 7)
    assert by["gestational_age_weeks"].n_unparsed == 2


def test_ga_completed_weeks_rejects_fractional_values(tmp_path: Path) -> None:
    _, fields, raw, _ = KIND_CASES["ga_completed_weeks"]
    canonical, by = _run(tmp_path, fields, raw)
    assert canonical["gestational_age_weeks"].tolist()[:2] == [36.0, 38.0]
    assert canonical["gestational_age_weeks"].iloc[2:4].isna().all()
    assert by["gestational_age_weeks"].n_unparsed == 2


DT_FIELD = "  admitted_at: {raw: W, kind: datetime}\n"
LOCAL_0830 = pd.Timestamp("2023-11-02 08:30:00")


def test_naive_datetime_is_local_wall_time(tmp_path: Path) -> None:
    raw = pd.DataFrame({"W": ["2023-11-02T08:30:00"]}, dtype=object)
    canonical, by = _run(tmp_path, DT_FIELD, raw)
    assert canonical["admitted_at"].tolist() == [LOCAL_0830]
    assert by["admitted_at"].n_tz_aware == 0


def test_aware_datetime_is_converted_to_kigali(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {"W": ["2023-11-02T06:30:00Z", "2023-11-02T08:30:00+02:00", "2023-11-02T05:30:00-01:00"]},
        dtype=object,
    )
    canonical, by = _run(tmp_path, DT_FIELD, raw)
    assert canonical["admitted_at"].tolist() == [LOCAL_0830] * 3
    assert canonical["admitted_at"].dtype == "datetime64[ns]"
    assert by["admitted_at"].n_tz_aware == 3


def test_mixed_naive_and_aware_datetimes_are_both_parsed(tmp_path: Path) -> None:
    raw = pd.DataFrame({"W": ["2023-11-02T08:30:00", "2023-11-02T06:30:00Z"]}, dtype=object)
    canonical, by = _run(tmp_path, DT_FIELD, raw)
    assert canonical["admitted_at"].tolist() == [LOCAL_0830, LOCAL_0830]
    assert by["admitted_at"].n_tz_aware == 1
    assert by["admitted_at"].n_unparsed == 0


def test_datetime_objects_with_and_without_tzinfo(tmp_path: Path) -> None:
    aware = dt.datetime(2023, 11, 2, 6, 30, tzinfo=dt.UTC)
    naive = dt.datetime(2023, 11, 2, 8, 30)
    raw = pd.DataFrame({"W": [aware, naive, "03/04/2024"]}, dtype=object)
    canonical, by = _run(tmp_path, DT_FIELD, raw)
    assert canonical["admitted_at"].tolist()[:2] == [LOCAL_0830, LOCAL_0830]
    assert pd.isna(canonical["admitted_at"].iloc[2])
    assert (by["admitted_at"].n_tz_aware, by["admitted_at"].n_unparsed) == (1, 1)


def test_datetime_explicit_format_is_not_second_guessed(tmp_path: Path) -> None:
    fields = '  admitted_at: {raw: W, kind: datetime, format: "%d/%m/%Y"}\n'
    raw = pd.DataFrame({"W": ["03/04/2024", "2024-01-05"]}, dtype=object)
    canonical, by = _run(tmp_path, fields, raw)
    assert canonical["admitted_at"].iloc[0] == pd.Timestamp("2024-04-03")
    assert pd.isna(canonical["admitted_at"].iloc[1])
    assert by["admitted_at"].n_unparsed == 1


def test_non_finite_bool_and_huge_numbers_are_unparsed(tmp_path: Path) -> None:
    fields = (
        "  parity: {raw: P, kind: integer}\n"
        "  maternal_age: {raw: A, kind: float}\n"
        "  gestational_age_weeks: {raw: G, kind: gestational_age, format: decimal_weeks}\n"
    )
    bad = [np.inf, "-inf", True, np.bool_(False), 10**20, "1e300"]
    raw = pd.DataFrame({"P": bad, "A": bad, "G": bad}, dtype=object)
    canonical, by = _run(tmp_path, fields, raw)
    for name in ("parity", "maternal_age", "gestational_age_weeks"):
        assert canonical[name].isna().all()
        assert (by[name].n_raw_nonnull, by[name].n_unparsed) == (6, 6)


def test_unmapped_levels_hide_rare_labels_and_the_missing_row(tmp_path: Path) -> None:
    fields = '  gdm_recorded: {raw: X, kind: category, levels: {"Y": "yes", "N": "no"}}\n'
    raw = pd.DataFrame({"X": ["Y"] * 5 + ["Rarelabel"] * 3 + ["Commonlabel"] * 6 + [None]})
    path = tmp_path / "m.yaml"
    path.write_text("fields:\n" + ROW_KEY + fields, encoding="utf-8")
    _, report = apply_mapping(raw, load_mapping(path))
    payload = json.dumps(dataclasses.asdict(report), default=str)
    assert "Rarelabel" not in payload
    assert "Commonlabel" in payload
    assert "(missing)" not in payload


def test_duplicate_raw_column_label_rejected(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text("fields:\n" + ROW_KEY + "  parity: {raw: P, kind: integer}\n", "utf-8")
    raw = pd.DataFrame([[1, 2]], columns=["P", "P"])
    with pytest.raises(MappingError, match="P"):
        apply_mapping(raw, load_mapping(path))


INVALID_CONFIGS = {
    "unknown key": ("  parity: {raw: P, kind: integer, rnage: [0, 20]}\n", "parity"),
    "range three items": ("  parity: {raw: P, kind: integer, range: [0, 1, 2]}\n", "range"),
    "range reversed": ("  parity: {raw: P, kind: integer, range: [20, 0]}\n", "range"),
    "range text": ("  parity: {raw: P, kind: integer, range: [a, 20]}\n", "range"),
    "range infinite": ("  parity: {raw: P, kind: integer, range: [0, .inf]}\n", "range"),
    "range bool": ("  parity: {raw: P, kind: integer, range: [false, 20]}\n", "range"),
    "range scalar": ("  parity: {raw: P, kind: integer, range: 20}\n", "range"),
    "range on text": ("  facility_id: {raw: F, kind: text, range: [0, 1]}\n", "range"),
    "range on category": (
        '  cs: {raw: X, kind: category, range: [0, 1], levels: {"a": 1}}\n',
        "range",
    ),
    "levels on integer": ('  parity: {raw: P, kind: integer, levels: {"a": 1}}\n', "levels"),
    "format on integer": ("  parity: {raw: P, kind: integer, format: x}\n", "format"),
    "format on category": (
        '  cs: {raw: X, kind: category, format: x, levels: {"a": 1}}\n',
        "format",
    ),
    "dtype disagrees": (
        '  previous_cs_count: {raw: X, kind: category, dtype: float64, levels: {"a": 1}}\n',
        "dtype",
    ),
    "dtype bogus": ("  parity: {raw: P, kind: integer, dtype: banana}\n", "dtype"),
    "integer on text field": ("  facility_id: {raw: F, kind: integer}\n", "facility_id"),
    "float on Int64 field": ("  parity: {raw: P, kind: float}\n", "parity"),
    "integer on float field": ("  maternal_age: {raw: A, kind: integer}\n", "maternal_age"),
    "text on datetime field": ("  admitted_at: {raw: W, kind: text}\n", "admitted_at"),
    "datetime on float field": ("  maternal_age: {raw: A, kind: datetime}\n", "maternal_age"),
    "row_key on other field": ("  facility_id: {kind: row_key}\n", "row_key"),
    "row_key with raw": ("  admission_id: {raw: ID, kind: row_key}\n", "admission_id"),
    "spec not a dict": ("  parity: 3\n", "parity"),
    "raw not a string": ("  parity: {raw: 5, kind: integer}\n", "raw"),
    "raw item not a string": (
        "  gestational_age_weeks: {raw: [W, 5], kind: gestational_age, format: weeks_plus_days}\n",
        "raw",
    ),
    "raw empty list": ("  parity: {raw: [], kind: integer}\n", "raw"),
    "two raw on integer": ("  parity: {raw: [A, B], kind: integer}\n", "parity"),
    "two raw on decimal GA": (
        "  gestational_age_weeks: {raw: [W, D], kind: gestational_age, format: decimal_weeks}\n",
        "gestational_age_weeks",
    ),
    "three raw on weeks_plus_days": (
        "  gestational_age_weeks: {raw: [W, D, X], kind: gestational_age, "
        "format: weeks_plus_days}\n",
        "gestational_age_weeks",
    ),
    "presentation level outside schema": (
        '  fetal_presentation: {raw: X, kind: category, levels: {"Bum": "breach"}}\n',
        "fetal_presentation",
    ),
    "onset level outside schema": (
        '  onset_of_labour: {raw: X, kind: category, levels: {"a": "Spontaneous"}}\n',
        "onset_of_labour",
    ),
    "cs level outside 0/1": ('  cs: {raw: X, kind: category, levels: {"a": 2}}\n', "cs"),
    "yes/no level outside schema": (
        '  gdm_recorded: {raw: X, kind: category, levels: {"a": "Yes"}}\n',
        "gdm_recorded",
    ),
    "fractional Int64 level": (
        '  previous_cs_count: {raw: X, kind: category, levels: {"a": 1.5}}\n',
        "non-integer or non-numeric level value",
    ),
    "text Int64 level": (
        '  previous_cs_count: {raw: X, kind: category, levels: {"a": "1"}}\n',
        "non-integer or non-numeric level value",
    ),
    "category on datetime field": (
        '  admitted_at: {raw: X, kind: category, levels: {"a": "b"}}\n',
        "admitted_at",
    ),
    "duplicate field key": (
        "  parity: {raw: P, kind: integer}\n  parity: {raw: Q, kind: integer}\n",
        "duplicate",
    ),
    "duplicate level key": (
        '  cs: {raw: X, kind: category, levels: {"a": 1, "a": 0}}\n',
        "duplicate",
    ),
}


@pytest.mark.parametrize("case", list(INVALID_CONFIGS))
def test_invalid_config_rejected(tmp_path: Path, case: str) -> None:
    fields, match = INVALID_CONFIGS[case]
    with pytest.raises(MappingError, match=match):
        _load_fields(tmp_path, fields)


def test_valid_config_variants_load(tmp_path: Path) -> None:
    _load_fields(
        tmp_path,
        ROW_KEY + "  previous_cs_count: {raw: X, kind: category, dtype: Int64, "
        'levels: {"a": 1, "b": 2.0, "c": ~}}\n'
        '  cs: {raw: Y, kind: category, levels: {"s": 1, "v": 0}}\n'
        '  gdm_recorded: {raw: Z, kind: category, levels: {"Y": "yes", "N": "no", "?": ~}}\n'
        "  gestational_age_weeks: {raw: [W, D], kind: gestational_age, "
        "format: weeks_plus_days, range: [20, 45]}\n"
        "  maternal_age: {raw: A, kind: float, dtype: float64, range: [12.5, 55]}\n",
    )
