import dataclasses
import datetime as dt
import hashlib
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
  delivery_date: {raw: When, kind: datetime, status: confirmed, date_only: true}
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
            "When": ["2023-11-02", None, "2024-01-05", "bad", None, None],
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
    assert canonical["delivery_date"].notna().sum() == 2
    validate_canonical(canonical)


def test_mapping_report_counts(mapping_path: Path) -> None:
    _, report = apply_mapping(_raw(), load_mapping(mapping_path))
    by = {f.canonical: f for f in report.fields}
    assert (by["parity"].n_raw_nonnull, by["parity"].n_unparsed) == (5, 1)
    assert by["previous_cs_count"].n_unparsed == 1
    assert "Blah" not in str(by["previous_cs_count"].unmapped_levels)
    assert by["gestational_age_weeks"].n_unparsed == 1
    assert by["maternal_age"].n_out_of_range == 1
    assert by["delivery_date"].n_unparsed == 1
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
SALT = bytes(range(32))


def _run(tmp_path: Path, fields: str, raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = tmp_path / "m.yaml"
    path.write_text("fields:\n" + ROW_KEY + fields, encoding="utf-8")
    canonical, report = apply_mapping(raw, load_mapping(path), salt=SALT)
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
        "delivery_date",
        "  delivery_date: {raw: W, kind: datetime, date_only: true}\n",
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
    "ga_weeks_days_text": (
        "gestational_age_weeks",
        "  gestational_age_weeks: {raw: G, kind: gestational_age, format: weeks_days_text, "
        "range: [20, 45]}\n",
        pd.DataFrame(
            {"G": ["38", "37 weeks, 4 days", "38.5", "50 weeks", None, "  ", "38+9", 39]},
            dtype=object,
        ),
        (6, 3, 2, 1, 0),
    ),
    "hash_key": (
        "mother_key",
        "  mother_key: {raw: ID, kind: hash_key}\n",
        pd.DataFrame({"ID": ["P-1", " P-1 ", None, "  ", "P-2", 7, pd.NA]}, dtype=object),
        (4, 4, 0, 0, 0),
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


DT_FIELD = "  delivery_date: {raw: W, kind: datetime, date_only: true}\n"
# delivery_date is date-only, so timezone handling is checked where it moves the date:
# 00:30 in Kigali (UTC+2) on 2 November is still 1 November in UTC.
LOCAL_DAY = pd.Timestamp("2023-11-02")


def test_naive_datetime_is_local_wall_time(tmp_path: Path) -> None:
    # Taken as UTC, 23:30 would become 01:30 on 3 November in Kigali.
    raw = pd.DataFrame({"W": ["2023-11-02T23:30:00"]}, dtype=object)
    canonical, by = _run(tmp_path, DT_FIELD, raw)
    assert canonical["delivery_date"].tolist() == [LOCAL_DAY]
    assert by["delivery_date"].n_tz_aware == 0
    assert by["delivery_date"].n_time_dropped == 1


def test_aware_datetime_is_converted_to_kigali_before_truncation(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {"W": ["2023-11-01T22:30:00Z", "2023-11-02T00:30:00+02:00", "2023-11-01T21:30:00-01:00"]},
        dtype=object,
    )
    canonical, by = _run(tmp_path, DT_FIELD, raw)
    assert canonical["delivery_date"].tolist() == [LOCAL_DAY] * 3
    assert canonical["delivery_date"].dtype == "datetime64[ns]"
    assert by["delivery_date"].n_tz_aware == 3
    assert by["delivery_date"].n_time_dropped == 3


def test_mixed_naive_and_aware_datetimes_are_both_parsed(tmp_path: Path) -> None:
    raw = pd.DataFrame({"W": ["2023-11-02T08:30:00", "2023-11-01T22:30:00Z"]}, dtype=object)
    canonical, by = _run(tmp_path, DT_FIELD, raw)
    assert canonical["delivery_date"].tolist() == [LOCAL_DAY, LOCAL_DAY]
    assert by["delivery_date"].n_tz_aware == 1
    assert by["delivery_date"].n_unparsed == 0


def test_datetime_objects_with_and_without_tzinfo(tmp_path: Path) -> None:
    aware = dt.datetime(2023, 11, 1, 22, 30, tzinfo=dt.UTC)
    naive = dt.datetime(2023, 11, 2, 8, 30)
    raw = pd.DataFrame({"W": [aware, naive, "03/04/2024"]}, dtype=object)
    canonical, by = _run(tmp_path, DT_FIELD, raw)
    assert canonical["delivery_date"].tolist()[:2] == [LOCAL_DAY, LOCAL_DAY]
    assert pd.isna(canonical["delivery_date"].iloc[2])
    assert (by["delivery_date"].n_tz_aware, by["delivery_date"].n_unparsed) == (1, 1)


def test_date_only_counts_dropped_times(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {
            "W": [
                "2024-01-05",
                "2024-01-05T10:00:00",
                pd.Timestamp("2024-01-06 00:00"),
                dt.datetime(2024, 1, 7, 23, 59, 59),
                dt.date(2024, 1, 8),
                "bad",
                None,
            ]
        },
        dtype=object,
    )
    canonical, by = _run(tmp_path, DT_FIELD, raw)
    f = by["delivery_date"]
    expected = pd.to_datetime(
        ["2024-01-05", "2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08"]
    )
    assert canonical["delivery_date"].tolist()[:5] == list(expected)
    assert canonical["delivery_date"].isna().tolist()[5:] == [True, True]
    assert (f.n_raw_nonnull, f.n_mapped, f.n_unparsed, f.n_time_dropped) == (6, 5, 1, 2)


def test_time_dropped_is_zero_for_other_kinds(tmp_path: Path) -> None:
    _, by = _run(tmp_path, "  parity: {raw: P, kind: integer}\n", pd.DataFrame({"P": [1, 2]}))
    assert by["parity"].n_time_dropped == 0


def test_datetime_explicit_format_is_not_second_guessed(tmp_path: Path) -> None:
    fields = '  delivery_date: {raw: W, kind: datetime, format: "%d/%m/%Y", date_only: true}\n'
    raw = pd.DataFrame({"W": ["03/04/2024", "2024-01-05"]}, dtype=object)
    canonical, by = _run(tmp_path, fields, raw)
    assert canonical["delivery_date"].iloc[0] == pd.Timestamp("2024-04-03")
    assert pd.isna(canonical["delivery_date"].iloc[1])
    assert by["delivery_date"].n_unparsed == 1


def _load_fields(tmp_path: Path, fields: str) -> None:
    path = tmp_path / "m.yaml"
    path.write_text(f"fields:\n{fields}", encoding="utf-8")
    load_mapping(path)


def test_datetime_format_mixed_is_rejected(tmp_path: Path) -> None:
    fields = "  delivery_date: {raw: W, kind: datetime, format: mixed, date_only: true}\n"
    with pytest.raises(MappingError, match="delivery_date"):
        _load_fields(tmp_path, fields)


def test_datetime_format_infer_is_rejected(tmp_path: Path) -> None:
    fields = "  delivery_date: {raw: W, kind: datetime, format: infer, date_only: true}\n"
    with pytest.raises(MappingError, match="delivery_date"):
        _load_fields(tmp_path, fields)


def test_datetime_format_mixed_variant_is_rejected(tmp_path: Path) -> None:
    fields = "  delivery_date: {raw: W, kind: datetime, format: mixed-format, date_only: true}\n"
    with pytest.raises(MappingError, match="delivery_date"):
        _load_fields(tmp_path, fields)


def test_iso_default_rejects_reduced_precision_dates(tmp_path: Path) -> None:
    raw = pd.DataFrame({"W": ["2024", "2024-03", "2024-03-05"]}, dtype=object)
    canonical, by = _run(tmp_path, DT_FIELD, raw)
    assert by["delivery_date"].n_mapped == 1
    assert by["delivery_date"].n_unparsed == 2
    assert canonical["delivery_date"].tolist()[2] == pd.Timestamp("2024-03-05")
    assert pd.isna(canonical["delivery_date"].iloc[0])
    assert pd.isna(canonical["delivery_date"].iloc[1])


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
    "text on datetime field": ("  delivery_date: {raw: W, kind: text}\n", "delivery_date"),
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
        '  delivery_date: {raw: X, kind: category, levels: {"a": "b"}}\n',
        "delivery_date",
    ),
    "duplicate field key": (
        "  parity: {raw: P, kind: integer}\n  parity: {raw: Q, kind: integer}\n",
        "duplicate",
    ),
    "duplicate level key": (
        '  cs: {raw: X, kind: category, levels: {"a": 1, "a": 0}}\n',
        "duplicate",
    ),
    "hash_key on other field": ("  facility_id: {raw: F, kind: hash_key}\n", "hash_key"),
    "mother_key as text": ("  mother_key: {raw: ID, kind: text}\n", "mother_key"),
    "mother_key as category": (
        '  mother_key: {raw: ID, kind: category, levels: {"a": "b"}}\n',
        "mother_key",
    ),
    "hash_key without raw": ("  mother_key: {kind: hash_key}\n", "mother_key"),
    "hash_key two raw": ("  mother_key: {raw: [A, B], kind: hash_key}\n", "mother_key"),
    "hash_key with range": ("  mother_key: {raw: ID, kind: hash_key, range: [0, 1]}\n", "range"),
    "hash_key with levels": (
        '  mother_key: {raw: ID, kind: hash_key, levels: {"a": "b"}}\n',
        "levels",
    ),
    "hash_key with format": ("  mother_key: {raw: ID, kind: hash_key, format: x}\n", "format"),
    "hash_key with date_only": (
        "  mother_key: {raw: ID, kind: hash_key, date_only: true}\n",
        "date_only",
    ),
    "weeks_days_text two raw": (
        "  gestational_age_weeks: {raw: [W, D], kind: gestational_age, format: weeks_days_text}\n",
        "gestational_age_weeks",
    ),
    "date_only on integer": ("  parity: {raw: P, kind: integer, date_only: true}\n", "date_only"),
    "date_only on GA": (
        "  gestational_age_weeks: {raw: G, kind: gestational_age, format: decimal_weeks, "
        "date_only: true}\n",
        "date_only",
    ),
    "date_only not a bool": (
        '  delivery_date: {raw: W, kind: datetime, date_only: "yes"}\n',
        "date_only",
    ),
    "date_only number": ("  delivery_date: {raw: W, kind: datetime, date_only: 1}\n", "date_only"),
    "delivery_date without date_only": (
        "  delivery_date: {raw: W, kind: datetime}\n",
        "delivery_date.*date_only",
    ),
    "delivery_date date_only false": (
        "  delivery_date: {raw: W, kind: datetime, date_only: false}\n",
        "delivery_date.*date_only",
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
        "  maternal_age: {raw: A, kind: float, dtype: float64, range: [12.5, 55]}\n"
        "  mother_key: {raw: ID, kind: hash_key, status: confirmed, dtype: object}\n"
        "  delivery_date: {raw: DD, kind: datetime, date_only: true}\n",
    )
    _load_fields(
        tmp_path,
        "  gestational_age_weeks: {raw: G, kind: gestational_age, "
        "format: weeks_days_text, range: [20, 45]}\n",
    )


@pytest.mark.parametrize(
    "removed", ["systolic_bp", "proteinuria", "glucose_mmol_l", "omission_reason_systolic_bp"]
)
def test_fields_removed_in_v1_2_are_unknown(tmp_path: Path, removed: str) -> None:
    with pytest.raises(MappingError, match="unknown canonical field"):
        _load_fields(tmp_path, f"  {removed}: {{raw: X, kind: text}}\n")


def test_non_cephalic_presentation_level_allowed(tmp_path: Path) -> None:
    raw = pd.DataFrame({"M": ["Yes", "No", None]}, dtype=object)
    fields = (
        '  fetal_presentation: {raw: M, kind: category, levels: {"Yes": "non_cephalic", '
        '"No": "cephalic"}}\n'
    )
    canonical, _ = _run(tmp_path, fields, raw)
    assert canonical["fetal_presentation"].tolist() == ["non_cephalic", "cephalic", None]


# --- mother_key: salted one-way hash ----------------------------------------------------------

MK_FIELD = "  mother_key: {raw: ID, kind: hash_key, status: confirmed}\n"


def _expected_key(salt: bytes, text: str) -> str:
    return "MK_" + hashlib.sha256(salt + b"\x1f" + text.encode("utf-8")).hexdigest()[:32]


def _keys(tmp_path: Path, ids: list[object], salt: bytes = SALT) -> list[object]:
    path = tmp_path / "m.yaml"
    path.write_text("fields:\n" + ROW_KEY + MK_FIELD, encoding="utf-8")
    raw = pd.DataFrame({"ID": ids}, dtype=object)
    canonical, _ = apply_mapping(raw, load_mapping(path), salt=salt)
    return canonical["mother_key"].tolist()


def test_hash_key_is_salted_sha256_with_prefix(tmp_path: Path) -> None:
    keys = _keys(tmp_path, ["ID-001", "  ID-001\t", "id-001"])
    assert keys[0] == _expected_key(SALT, "ID-001")
    assert keys[1] == keys[0]  # surrounding whitespace is not part of the identifier
    assert keys[2] == _expected_key(SALT, "id-001") != keys[0]  # case is kept
    assert all(isinstance(k, str) and len(k) == 3 + 32 for k in keys)


def test_hash_key_blank_is_missing_not_hashed(tmp_path: Path) -> None:
    raw = pd.DataFrame({"ID": ["A", None, "   ", np.nan, pd.NA, "B"]}, dtype=object)
    canonical, by = _run(tmp_path, MK_FIELD, raw)
    assert canonical["mother_key"].isna().tolist() == [False, True, True, True, True, False]
    f = by["mother_key"]
    assert (f.n_raw_nonnull, f.n_mapped, f.n_unparsed, f.n_explicit_missing) == (2, 2, 0, 0)
    assert f.unmapped_levels == {}


def test_hash_key_integral_numbers_match_their_text(tmp_path: Path) -> None:
    keys = _keys(tmp_path, [12345, "12345", 12345.0, np.int64(12345)])
    assert len(set(keys)) == 1
    assert keys[0] == _expected_key(SALT, "12345")


def test_hash_key_is_deterministic_and_salt_dependent(tmp_path: Path) -> None:
    ids: list[object] = ["P1", "P2", "P1"]
    first = _keys(tmp_path, ids)
    assert first == _keys(tmp_path, ids)
    assert first[0] == first[2] != first[1]
    other = _keys(tmp_path, ids, salt=bytes(range(1, 33)))
    assert not set(first) & set(other)


def test_hash_key_has_no_collisions_on_distinct_ids(tmp_path: Path) -> None:
    ids: list[object] = [f"PT{i:05d}" for i in range(6000)] + [f"pt{i:05d}" for i in range(500)]
    keys = _keys(tmp_path, ids)
    assert len(set(keys)) == len(ids)


def test_hash_key_values_never_reach_frame_or_report(tmp_path: Path) -> None:
    ids = ["SECRET-ID-111", "SECRET-ID-222", "SECRET-ID-111"]
    raw = pd.DataFrame({"ID": ids, "P": [1, 2, 3]}, dtype=object)
    fields = MK_FIELD + "  parity: {raw: P, kind: integer}\n"
    canonical, by = _run(tmp_path, fields, raw)
    frame_text = canonical.astype(str).to_csv()
    report_text = json.dumps({k: dataclasses.asdict(v) for k, v in by.items()}, default=str)
    assert "SECRET" not in frame_text
    assert "SECRET" not in report_text
    for key in canonical["mother_key"]:
        assert key not in report_text


@pytest.mark.parametrize(
    "salt",
    [None, b"", b"short", bytes(15), "a string salt of plenty length"],
    ids=["none", "empty", "short", "fifteen", "str"],
)
def test_hash_key_requires_a_salt_of_16_bytes(tmp_path: Path, salt: Any) -> None:
    path = tmp_path / "m.yaml"
    path.write_text("fields:\n" + ROW_KEY + MK_FIELD, encoding="utf-8")
    raw = pd.DataFrame({"ID": ["SECRET-ID-111"]}, dtype=object)
    with pytest.raises(MappingError, match="hash_key needs a salt of at least 16 bytes") as exc:
        apply_mapping(raw, load_mapping(path), salt=salt)
    assert "SECRET" not in str(exc.value)


def test_sixteen_byte_salt_is_enough(tmp_path: Path) -> None:
    assert _keys(tmp_path, ["A"], salt=bytes(16))[0] == _expected_key(bytes(16), "A")


def test_salt_not_needed_without_hash_key(mapping_path: Path) -> None:
    canonical, _ = apply_mapping(_raw(), load_mapping(mapping_path))
    assert canonical["mother_key"].isna().all()


# --- free-text gestational age: weeks_days_text ----------------------------------------------

GA_TEXT = "  gestational_age_weeks: {raw: G, kind: gestational_age, format: weeks_days_text}\n"

# (raw value, expected decimal weeks); every shape seen in the export, with made-up numbers.
GA_TEXT_ACCEPTED: list[tuple[object, float]] = [
    # "N"
    ("38", 38.0),
    (" 38 ", 38.0),
    ("9", 9.0),
    (38, 38.0),
    (38.0, 38.0),
    (np.int64(36), 36.0),
    # "Nweeks" and unit variants
    ("37weeks", 37.0),
    ("37 weeks", 37.0),
    ("37Weeks", 37.0),
    ("37 WEEKS", 37.0),
    ("37week", 37.0),
    ("37 Week", 37.0),
    ("37wks", 37.0),
    ("37 wk", 37.0),
    ("37w", 37.0),
    ("37 W", 37.0),
    # "Nweekz" (typo seen in the export)
    ("36weekz", 36.0),
    ("36 WeekZ", 36.0),
    # "Nweeks, Ndays" / "Nweeks,Ndays"
    ("37weeks, 4days", 37 + 4 / 7),
    ("37 weeks, 4 days", 37 + 4 / 7),
    ("37weeks,4days", 37 + 4 / 7),
    ("37Weeks,4Days", 37 + 4 / 7),
    ("37 WEEKS , 6 DAYS", 37 + 6 / 7),
    # "Nweeks, Nday" / "Nweeks,Nday"
    ("37weeks, 1day", 37 + 1 / 7),
    ("37weeks,1day", 37 + 1 / 7),
    ("37 weeks, 0 day", 37.0),
    # "N+N", "N+N days", "N+Ndays"
    ("37+4", 37 + 4 / 7),
    ("37 + 4", 37 + 4 / 7),
    ("37+0", 37.0),
    ("37+5 days", 37 + 5 / 7),
    ("37+5days", 37 + 5 / 7),
    ("37+5 Days", 37 + 5 / 7),
    ("37+5d", 37 + 5 / 7),
    ("37+2 day", 37 + 2 / 7),
    # other separators and units
    ("37 weeks and 2 days", 37 + 2 / 7),
    ("37 Weeks AND 2 Days", 37 + 2 / 7),
    ("37wks 2d", 37 + 2 / 7),
    ("37 w 2 d", 37 + 2 / 7),
    ("37w2d", 37 + 2 / 7),
    ("37weeks+3days", 37 + 3 / 7),
    ("37 weeks + 3 days", 37 + 3 / 7),
    ("37 weeks 3 days", 37 + 3 / 7),
    ("37 weeks, 3", 37 + 3 / 7),
    ("37 weeks 3", 37 + 3 / 7),
    ("37wk,6", 37 + 6 / 7),
    ("\t37 weeks,\u00a02 days ", 37 + 2 / 7),
]

GA_TEXT_REJECTED: list[object] = [
    # fractions
    "38.5",
    "38,5",
    "38,5 weeks",
    "38.5 weeks",
    "38.0",
    38.5,
    # days of 7 or more
    "38+7",
    "38+10",
    "38 weeks 7 days",
    "38 weeks, 12 days",
    "38weeks,9days",
    # days only, or days unit on the weeks number
    "38 days",
    "38days",
    "38 d",
    "2 days",
    "38+2 weeks",
    # ranges
    "38-39",
    "38 - 39 weeks",
    "38 to 39 weeks",
    "38/39",
    # extra numbers or text
    "38 2",
    "38 and 2",
    "38 weeks 2 days 3",
    "38 weeks 2 weeks",
    "38 weeks 2 days 1 hour",
    "about 38 weeks",
    "38 weeks.",
    "38 weeks,",
    "38 weeks and",
    "38 weeks,, 2 days",
    "38 wkss",
    "38 wks 2 dys",
    "38weeks2days3",
    "+2",
    "weeks",
    "w",
    "unknown",
    "123",
    "038",
    "38 weeks 02 days",
    "\u0663\u0668",  # non-ASCII digits
    "38+-2",
    # non-text values that are not a whole number of weeks
    True,
    -3,
    100,
    float("inf"),
]


@pytest.mark.parametrize(("value", "expected"), GA_TEXT_ACCEPTED, ids=repr)
def test_weeks_days_text_accepts(tmp_path: Path, value: object, expected: float) -> None:
    canonical, by = _run(tmp_path, GA_TEXT, pd.DataFrame({"G": [value]}, dtype=object))
    assert canonical["gestational_age_weeks"].iloc[0] == pytest.approx(expected)
    assert (by["gestational_age_weeks"].n_mapped, by["gestational_age_weeks"].n_unparsed) == (1, 0)


@pytest.mark.parametrize("value", GA_TEXT_REJECTED, ids=repr)
def test_weeks_days_text_rejects(tmp_path: Path, value: object) -> None:
    canonical, by = _run(tmp_path, GA_TEXT, pd.DataFrame({"G": [value]}, dtype=object))
    assert pd.isna(canonical["gestational_age_weeks"].iloc[0])
    f = by["gestational_age_weeks"]
    assert (f.n_raw_nonnull, f.n_mapped, f.n_unparsed) == (1, 0, 1)


def test_weeks_days_text_applies_range_and_hides_values(tmp_path: Path) -> None:
    fields = GA_TEXT.replace("}\n", ", range: [20, 45]}\n")
    raw = pd.DataFrame({"G": ["19 weeks", "46+1", "45+0", "20", "zz-odd-zz"]}, dtype=object)
    canonical, by = _run(tmp_path, fields, raw)
    f = by["gestational_age_weeks"]
    assert (f.n_mapped, f.n_out_of_range, f.n_unparsed) == (2, 2, 1)
    assert canonical["gestational_age_weeks"].tolist()[2:4] == [45.0, 20.0]
    assert "zz-odd-zz" not in json.dumps(dataclasses.asdict(f), default=str)
