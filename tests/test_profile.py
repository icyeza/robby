from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robson_engine import load_rule_set
from robson_ml.mapping import FieldMapping, MappingConfig
from robson_ml.privacy import SUPPRESSED
from robson_ml.profile import (
    QUESTIONS,
    cramers_v,
    open_questions_markdown,
    robson_inputs_markdown,
    single_feature_auc,
    variable_profile,
    write_profile,
)
from robson_ml.robson_run import classify_frame
from tests.disclosure import (
    assert_not_recoverable,
    markdown_tables,
    scan_profile_outputs,
)
from tests.synthetic import make_admissions


def test_single_feature_auc() -> None:
    cs = pd.Series([0] * 20 + [1] * 20)
    assert single_feature_auc(pd.Series(range(40), dtype=float), cs) == pytest.approx(1.0)
    few = cs.iloc[14:26].reset_index(drop=True)
    assert single_feature_auc(pd.Series(range(12), dtype=float), few) is None


def test_cramers_v() -> None:
    cs = pd.Series([0] * 30 + [1] * 30)
    assert cramers_v(pd.Series(["a"] * 30 + ["b"] * 30), cs) == pytest.approx(1.0)
    assert cramers_v(pd.Series(["a", "b"] * 30), cs) == pytest.approx(0.0)
    assert cramers_v(pd.Series(["a"] * 60), cs) is None


def test_cramers_v_requires_min_class_count() -> None:
    # 7 recorded values perfectly split by cs: V = 1.0 would disclose every outcome.
    values = pd.Series(["pos"] * 3 + ["neg"] * 4 + [None] * 53)
    cs = pd.Series([1] * 3 + [0] * 4 + [0, 1] * 26 + [0])
    assert cramers_v(values, cs) is None


def test_cramers_v_pools_rare_levels() -> None:
    cs = pd.Series([0] * 30 + [1] * 30)
    common = ["a"] * 28 + ["b"] * 28
    # Two rare levels of 2 each pool into one "(rare)" level of 4, still below 5: dropped.
    with_rare = pd.Series([*common, "c", "c", "d", "d"])
    without = pd.Series(common + [None] * 4)
    assert cramers_v(with_rare, cs) == pytest.approx(cramers_v(without, cs))
    # Rare levels pooling to 5 or more are kept as one level.
    pooled = pd.Series(["a"] * 25 + ["b"] * 25 + ["c", "c", "d", "d", "e"] * 2)
    assert cramers_v(pooled, cs) == pytest.approx(
        cramers_v(pd.Series(["a"] * 25 + ["b"] * 25 + ["r"] * 10), cs)
    )
    assert cramers_v(pd.Series([f"l{i // 4}" for i in range(60)]), cs) is None


def _classified() -> pd.DataFrame:
    return classify_frame(make_admissions(3000, seed=21), load_rule_set())


def _raw_and_config() -> tuple[pd.DataFrame, pd.DataFrame, MappingConfig]:
    canonical = _classified()
    raw = pd.DataFrame(
        {
            "Age raw": canonical["maternal_age"].astype(object),
            "Rare flag": ["x"] * 3 + [None] * (len(canonical) - 3),
            "Comment": [f"c{i}" for i in range(len(canonical))],
        }
    )
    mapping = FieldMapping("maternal_age", "float", ("Age raw",), "confirmed")
    config = MappingConfig("t", None, {"maternal_age": mapping})
    return raw, canonical, config


def test_variable_profile_rows_and_suppression() -> None:
    raw, canonical, config = _raw_and_config()
    profile = variable_profile(raw, canonical, config)
    assert profile.columns[0] == "raw_position"
    assert profile["raw_position"].tolist() == [0, 1, 2]
    assert profile["raw_name"].tolist() == ["Age raw", "Rare flag", "Comment"]
    assert profile.loc[0, "canonical_name"] == "maternal_age"
    assert profile.loc[0, "association_metric"] == "auc"
    assert profile.loc[2, "kind"] == "text"
    assert profile.loc[1, "pct_missing"] == SUPPRESSED
    assert (profile["proposed_status"] == "review").all()


def test_robson_inputs_markdown_sections() -> None:
    text = robson_inputs_markdown(_classified())
    for heading in ["## Completeness", "## Engine status", "## Robson report table"]:
        assert heading in text


def test_open_questions_all_answered_or_flagged() -> None:
    canonical = _classified()
    raw = pd.DataFrame(
        {"ANC visits": np.arange(len(canonical)), "Attending doctor": ["d"] * len(canonical)}
    )
    text = open_questions_markdown(canonical, raw, {"q8": "Facility X is private."})
    for number in range(1, len(QUESTIONS) + 1):
        assert f"## Q{number}." in text
    assert "Facility X is private." in text
    assert "Attending doctor" in text
    assert "Not yet answered" in text


def test_write_profile_outputs_are_suppressed(tmp_path) -> None:
    raw, canonical, config = _raw_and_config()
    out_dir = tmp_path / "profile"
    write_profile(raw, canonical, config, {}, out_dir)

    csv_text = (out_dir / "variable_profile.csv").read_text(encoding="utf-8")
    written = pd.read_csv(out_dir / "variable_profile.csv", dtype=str)
    assert "n_nonnull" in written.columns
    counts = pd.to_numeric(written["n_nonnull"], errors="coerce")
    assert not ((counts >= 1) & (counts <= 4)).any()
    assert csv_text  # sanity: file is non-empty

    assert (out_dir / "robson_inputs.md").exists()
    assert (out_dir / "open_questions.md").exists()


def test_variable_profile_rejects_misaligned_frames() -> None:
    raw, canonical, config = _raw_and_config()
    with pytest.raises(ValueError, match="rows"):
        variable_profile(raw.iloc[:-1], canonical, config)


def _facility_positions(canonical: pd.DataFrame, missing: dict[str, int]) -> pd.Series:
    values = pd.Series(np.arange(len(canonical)), dtype=object)
    facility = canonical["facility_id"].astype(str).to_numpy()
    for fac, k in missing.items():
        values.iloc[np.where(facility == fac)[0][:k]] = None
    return values


def test_variable_profile_blocks_facility_differencing() -> None:
    # Reviewer recovery: FAC_A missing 2 was "<5" but overall - (B + C + D) gave it back.
    canonical = _classified()
    raw = pd.DataFrame(
        {"FacDiff": _facility_positions(canonical, {"FAC_A": 2, "FAC_B": 10, "FAC_C": 10})}
    )
    profile = variable_profile(raw, canonical, MappingConfig("t", None, {})).iloc[0]
    facility_cells = [profile[f"pct_missing_{f}"] for f in ["FAC_A", "FAC_B", "FAC_C", "FAC_D"]]
    assert profile["pct_missing_FAC_A"] == SUPPRESSED
    assert profile["pct_missing"] != SUPPRESSED
    assert_not_recoverable(facility_cells, None, "FacDiff")
    # FAC_D has no missing values: a published 0.0 hides nobody and is not a candidate.
    assert profile["pct_missing_FAC_D"] == 0.0


def test_variable_profile_quantiles_need_five_values_each_side() -> None:
    # Reviewer recovery: 2 ones among 40 values gave p95 = 0.05, disclosing the two ones.
    canonical = _classified()
    binary = pd.Series([np.nan] * len(canonical), dtype=object)
    binary.iloc[:40] = [1] * 2 + [0] * 38
    raw = pd.DataFrame({"Bin40": binary, "Age": canonical["maternal_age"].astype(object)})
    profile = variable_profile(raw, canonical, MappingConfig("t", None, {}))
    assert pd.isna(profile.loc[0, "p95"])
    assert profile.loc[0, "p50"] == 0.0
    assert profile.loc[1, ["p5", "p25", "p50", "p75", "p95"]].notna().all()


def _first_table_after(text: str, marker: str) -> pd.DataFrame:
    return markdown_tables(text.split(marker, 1)[1])[0]


def test_counts_table_hides_a_second_level_for_a_lone_small_count() -> None:
    # Reviewer recovery: gdm yes = 1 was "<5" but N - no - (missing) gave it back.
    canonical = _classified()
    assert (canonical["gdm_recorded"] == "yes").sum() == 1
    text = open_questions_markdown(canonical, pd.DataFrame(index=canonical.index), {})
    table = _first_table_after(text, "gdm_recorded:")
    assert table.set_index("value").loc["yes", "n"] == SUPPRESSED
    assert_not_recoverable(table["n"].tolist(), len(canonical), "gdm_recorded")
    recorded = table[table["value"] != "(missing)"]
    assert_not_recoverable(recorded["n"].tolist(), None, "gdm_recorded recorded levels")


def test_status_table_hides_a_second_status_for_three_conflicts() -> None:
    classified = _classified()
    conflicts = classified.index[classified["robson_status"] == "conflict"][3:]
    classified.loc[conflicts, "robson_status"] = "partial"
    text = robson_inputs_markdown(classified)
    status = _first_table_after(text, "## Engine status")
    assert status.set_index("status").loc["conflict", "n"] == SUPPRESSED
    assert_not_recoverable(status["n"].tolist(), len(classified), "engine status")


def test_robson_report_blocks_contribution_total_recovery() -> None:
    # Reviewer recovery: rel_contribution pinned FAC_A's CS total (336), and the lone hidden
    # n_cs (group 6, 12 of 13) followed as total - shown.
    classified = _classified_with_missing_outcomes()
    report = _first_table_after(robson_inputs_markdown(classified), "## Robson report table")
    for facility, block in report.groupby("facility"):
        for column in ("n", "n_cs"):
            assert_not_recoverable(block[column].tolist(), None, f"{facility} {column}")
    for column in ("abs_contribution", "rel_contribution", "pct_of_deliveries"):
        shown = report.loc[report[column] != SUPPRESSED, column]
        assert shown.str.fullmatch(r"\d\.\d\d").all(), column


def _classified_with_missing_outcomes() -> pd.DataFrame:
    admissions = make_admissions(3000, seed=21)
    admissions["cs"] = admissions["cs"].astype("Int64")
    admissions.loc[admissions.index[:3], "cs"] = pd.NA
    return classify_frame(admissions, load_rule_set())


def test_excluded_for_missing_outcome_is_reported_exactly() -> None:
    classified = _classified_with_missing_outcomes()
    excluded = int(classified["cs"].isna().sum())
    text = robson_inputs_markdown(classified)
    assert (
        f"{excluded} rows excluded for missing outcome (reported exactly; data-quality count,"
        " spec §4.2)" in text
    )


def test_q4_q5_blank_strings_count_as_missing() -> None:
    canonical = _classified()
    blanks = pd.Series(["  "] * len(canonical), dtype=object)
    blanks.iloc[:100] = "midwife"
    raw = pd.DataFrame({"Attending nurse": blanks})
    text = open_questions_markdown(canonical, raw, {})
    table = _first_table_after(text, "## Q4.")
    expected = round(100.0 * 100 / len(canonical), 1)
    assert float(table.loc[0, "pct_recorded"]) == pytest.approx(expected)


def _adversarial_profile(out_dir: Path) -> None:
    """The reviewer's crafted frame (b3_adv): every recovery it demonstrated at once."""
    classified = _classified_with_missing_outcomes()
    n = len(classified)
    rng = np.random.default_rng(0)
    binary = pd.Series([np.nan] * n, dtype=object)
    binary.iloc[rng.choice(n, 40, replace=False)] = [1] * 2 + [0] * 38
    near_full = pd.Series(np.arange(n), dtype=object)
    near_full.iloc[:3] = None
    cs = classified["cs"]
    tiny = pd.Series([None] * n, dtype=object)
    tiny.iloc[np.where((cs == 1).fillna(False).to_numpy())[0][:3]] = "pos"
    tiny.iloc[np.where((cs == 0).fillna(False).to_numpy())[0][:4]] = "neg"
    nurse = pd.Series(["n"] * n, dtype=object)
    nurse.iloc[:2] = "  "
    nurse.iloc[2:102] = None
    raw = pd.DataFrame(
        {
            "Age raw": classified["maternal_age"].astype(object),
            "Bin40": binary,
            "NearFull": near_full,
            "FacDiff": _facility_positions(
                classified, {"FAC_A": 2, "FAC_B": 10, "FAC_C": 10, "FAC_D": 10}
            ),
            "Tiny HIV": tiny,
            "Attending nurse": nurse,
        }
    )
    mapping = FieldMapping("maternal_age", "float", ("Age raw",), "confirmed")
    write_profile(raw, classified, MappingConfig("t", None, {"maternal_age": mapping}), {}, out_dir)


def test_write_profile_passes_disclosure_scan(tmp_path: Path) -> None:
    out_dir = tmp_path / "profile"
    _adversarial_profile(out_dir)
    scan_profile_outputs(out_dir)
    profile = pd.read_csv(out_dir / "variable_profile.csv", dtype=str).set_index("raw_name")
    assert pd.isna(profile.loc["Bin40", "p95"])
    assert pd.isna(profile.loc["Tiny HIV", "association"])


def test_write_profile_default_frame_passes_disclosure_scan(tmp_path: Path) -> None:
    raw, canonical, config = _raw_and_config()
    out_dir = tmp_path / "profile"
    write_profile(raw, canonical, config, {}, out_dir)
    scan_profile_outputs(out_dir)
