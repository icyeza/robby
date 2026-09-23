import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robson_engine import load_rule_set
from robson_ml.mapping import FieldMapping, MappingConfig
from robson_ml.privacy import SECONDARY, SUPPRESSED
from robson_ml.profile import (
    GA_BAND_RECORDED,
    QUESTIONS,
    cramers_v,
    input_completeness,
    open_questions_markdown,
    pct_by_facility,
    robson_inputs_markdown,
    single_feature_auc,
    variable_profile,
    write_profile,
)
from robson_ml.robson_run import classify_frame
from robson_ml.schema import GA_BAND_FIELDS
from tests.disclosure import (
    HIDDEN,
    CountReader,
    assert_linked_status_system,
    assert_not_recoverable,
    markdown_tables,
    scan_open_questions,
    scan_profile_outputs,
    scan_robson_inputs,
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
    assert profile["pct_missing"] not in HIDDEN
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
    yes = canonical.index[canonical["gdm_recorded"] == "yes"]
    canonical.loc[yes[1:], "gdm_recorded"] = "no"
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
        shown = report.loc[~report[column].isin(HIDDEN), column]
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


def _reviewer_case_a() -> pd.DataFrame:
    # b3_cross case A: 3 conflicts, partial records resolved by two large field combos.
    classified = _classified()
    conflicts = classified.index[classified["robson_status"] == "conflict"][3:]
    classified.loc[conflicts, "robson_status"] = "partial"
    partial = classified.index[classified["robson_status"] == "partial"]
    classified.loc[partial, "robson_resolving_fields"] = np.where(
        np.arange(len(partial)) % 2, "parity", "gestational_age_weeks"
    )
    return classified


@pytest.mark.parametrize("excluded", [True, False])
def test_status_resolving_fields_and_residual_are_protected_jointly(excluded: bool) -> None:
    # Reviewer recovery R1: partial = sum of the (all shown) resolving-fields rows, then
    # conflict = Records - resolved - partial, although both status cells were hidden.
    classified = _reviewer_case_a()
    if not excluded:
        classified["cs"] = classified["cs"].fillna(0)
    text = robson_inputs_markdown(classified)
    _, status, resolving, report = markdown_tables(text)
    status = status.set_index("status")
    assert status.loc["conflict", "n"] == SUPPRESSED
    partial_known = not resolving["n"].isin(HIDDEN).any()
    resolved_known = status.loc["resolved", "n"] not in HIDDEN
    assert not (partial_known and resolved_known), "conflict = Records - resolved - partial"
    residual = report[(report["facility"] == "ALL") & (report["row"] == "residual")]
    residual_known = residual["n"].iloc[0] not in HIDDEN
    assert not (partial_known and residual_known), "conflict = residual - partial"
    assert_linked_status_system(status.reset_index(), resolving, report)
    scan_robson_inputs(text)


def _reviewer_case_b() -> tuple[pd.DataFrame, pd.DataFrame, MappingConfig]:
    # b3_cross case B': plurality missing in 2 FAC_A records and 10 of each other facility.
    classified = _classified()
    classified["plurality"] = classified["plurality"].fillna(1)
    facility = classified["facility_id"].astype(str).to_numpy()
    for fac, k in {"FAC_A": 2, "FAC_B": 10, "FAC_C": 10, "FAC_D": 10}.items():
        classified.loc[classified.index[np.where(facility == fac)[0][:k]], "plurality"] = np.nan
    raw = pd.DataFrame({"Plurality raw": classified["plurality"].astype(object)})
    mapping = FieldMapping("plurality", "int", ("Plurality raw",), "confirmed")
    return raw, classified, MappingConfig("t", None, {"plurality": mapping})


def test_raw_and_canonical_missingness_hide_the_same_facilities(tmp_path: Path) -> None:
    # Reviewer recovery R2: the profile hid FAC_A and FAC_B, completeness FAC_A and FAC_D;
    # each filled in the other's second cell, and the Q9 (missing) total gave FAC_A.
    raw, classified, config = _reviewer_case_b()
    out_dir = tmp_path / "profile"
    write_profile(raw, classified, config, {}, out_dir)
    profile = pd.read_csv(out_dir / "variable_profile.csv", dtype=str).iloc[0]
    completeness = markdown_tables((out_dir / "robson_inputs.md").read_text(encoding="utf-8"))[0]
    row = completeness.set_index("input").loc["plurality"]
    facilities = ["FAC_A", "FAC_B", "FAC_C", "FAC_D"]
    unknown = [f for f in facilities if profile[f"pct_missing_{f}"] in HIDDEN and row[f] in HIDDEN]
    assert len(unknown) != 1, f"{unknown[0]} = (missing) total - the other facilities"
    assert row["FAC_A"] == SUPPRESSED
    scan_profile_outputs(out_dir)


FACILITIES = ["FAC_A", "FAC_B", "FAC_C", "FAC_D"]


def test_completeness_adds_ga_band_row() -> None:
    classified = _classified()
    table = input_completeness(classified).set_index("input")
    assert table.index[-1] == GA_BAND_RECORDED
    band = classified[list(GA_BAND_FIELDS)].notna().all(axis=1)
    assert float(table.loc[GA_BAND_RECORDED, "all"]) == pytest.approx(round(100 * band.mean(), 1))
    text = robson_inputs_markdown(classified)
    # No row nested in another: their difference would be a count published by subtraction.
    assert "exact or band" not in text and "precise type" not in text
    scan_robson_inputs(text)


def _band_scenario(facility_b: str, band_only_at_a: int) -> pd.DataFrame:
    """Reviewer case (c2_scen): FAC_A has exactly ``band_only_at_a`` records with GA only as
    a band; FAC_B has no band at all (``"noband"``) or no exact GA (``"noexact"``)."""
    admissions = make_admissions(800, seed=5)
    facility = admissions["facility_id"].astype(str)
    band = admissions["ga_band_lower"].notna()
    exact = admissions["gestational_age_weeks"]
    band_only = admissions.index[(facility == "FAC_A") & exact.isna() & band]
    assert len(band_only) > band_only_at_a
    fill = band_only[band_only_at_a:]
    middle = (admissions.loc[fill, "ga_band_lower"] + admissions.loc[fill, "ga_band_upper"]) / 2
    admissions.loc[fill, "gestational_age_weeks"] = middle.round(3)
    if facility_b == "noband":
        admissions.loc[facility == "FAC_B", list(GA_BAND_FIELDS)] = np.nan
    else:
        admissions.loc[facility == "FAC_B", "gestational_age_weeks"] = np.nan
    return classify_frame(admissions, load_rule_set())


def _band_raw_and_config(classified: pd.DataFrame) -> tuple[pd.DataFrame, MappingConfig]:
    exact = classified["gestational_age_weeks"]
    lower, upper = classified["ga_band_lower"], classified["ga_band_upper"]
    band = [
        None if pd.isna(lo) else f"{lo:.2f}-{hi:.2f}" for lo, hi in zip(lower, upper, strict=True)
    ]
    raw = pd.DataFrame(
        {
            "GA text": exact.map(lambda x: None if pd.isna(x) else f"{x:.3f}").astype(object),
            "GA band": pd.Series(band, dtype=object),
        }
    )
    fields = {
        "gestational_age_weeks": FieldMapping(
            "gestational_age_weeks", "gestational_age", ("GA text",), "confirmed"
        ),
        **{f: FieldMapping(f, "category", ("GA band",), "confirmed") for f in GA_BAND_FIELDS},
    }
    return raw, MappingConfig("t", None, fields)


def _ga_reader(out_dir: Path, sizes: dict[str, int]) -> tuple[CountReader, pd.DataFrame]:
    """An integer-programming reader of every published GA cell: the exact and band rows,
    their raw columns' missingness (the same records), facility sums, and per scope the
    unknown count with both (Frechet bounds)."""
    text = (out_dir / "robson_inputs.md").read_text(encoding="utf-8")
    completeness = markdown_tables(text)[0].set_index("input")
    profile = pd.read_csv(out_dir / "variable_profile.csv", dtype=str).set_index("raw_name")
    reader = CountReader()
    for label, raw_name in (("gestational_age_weeks", "GA text"), (GA_BAND_RECORDED, "GA band")):
        for scope, n in sizes.items():
            column = "pct_missing" if scope == "all" else f"pct_missing_{scope}"
            count = reader.pct((label, scope), completeness.loc[label, scope], n)
            raw_count = reader.pct((raw_name, scope), profile.loc[raw_name, column], n, False)
            reader.constrain({count: 1, raw_count: -1}, 0, 0)
        for key in (label, raw_name):
            reader.constrain({(key, "all"): -1, **{(key, f): 1 for f in FACILITIES}}, 0, 0)
    for scope, n in sizes.items():
        e, b = ("gestational_age_weeks", scope), (GA_BAND_RECORDED, scope)
        both = reader.var(("both", scope), 0, n)
        reader.constrain({both: 1, e: -1}, -np.inf, 0)
        reader.constrain({both: 1, b: -1}, -np.inf, 0)
        reader.constrain({both: 1, e: -1, b: -1}, -n, np.inf)
    reader.constrain({("both", "all"): -1, **{("both", f): 1 for f in FACILITIES}}, 0, 0)
    return reader, completeness


@pytest.mark.parametrize(("facility_b", "band_only_at_a"), [("noband", 3), ("noexact", 1)])
def test_band_only_and_neither_counts_are_not_recoverable(
    tmp_path: Path, facility_b: str, band_only_at_a: int
) -> None:
    # Reviewer recovery (c2_attack): the "exact or band" row, the exact row and the raw band
    # column's missingness pinned the band-only and "neither" counts. Now the band row
    # counts only the band, hides alike with its raw column, and an integer-programming
    # reader over every published GA cell can pin neither count nor a hidden cell.
    classified = _band_scenario(facility_b, band_only_at_a)
    raw, config = _band_raw_and_config(classified)
    out_dir = tmp_path / "profile"
    write_profile(raw, classified, config, {}, out_dir)
    scan_profile_outputs(out_dir)
    facility = classified["facility_id"].astype(str)
    sizes = {"all": len(classified), **{f: int((facility == f).sum()) for f in FACILITIES}}
    reader, completeness = _ga_reader(out_dir, sizes)
    exact = classified["gestational_age_weeks"].notna()
    band = classified[list(GA_BAND_FIELDS)].notna().all(axis=1)
    for scope in sizes:
        rows = facility == scope if scope != "all" else pd.Series(True, index=classified.index)
        e, b, both = ("gestational_age_weeks", scope), (GA_BAND_RECORDED, scope), ("both", scope)
        for name, truth, coefs in (
            ("band only", int((rows & band & ~exact).sum()), {b: 1, both: -1}),
            ("neither", int((rows & ~band & ~exact).sum()), {e: -1, b: -1, both: 1}),
        ):
            if 1 <= truth <= 4:
                low, high = reader.range(coefs)
                assert low < high, f"{name} at {scope} pinned to {truth}"
        for key in (e, b):
            if completeness.loc[key[0], scope] in HIDDEN:
                low, high = reader.range({key: 1})
                assert low < high, f"{key} pinned"


def _one_per_facility() -> tuple[pd.Series, pd.Series]:
    facility = pd.Series(np.repeat(FACILITIES, 200))
    mask = pd.Series(False, index=facility.index)
    mask.iloc[[0, 200, 400, 600]] = True
    return mask, facility


@pytest.mark.parametrize("negate", [False, True])
def test_one_missing_per_facility_is_not_pinned(negate: bool) -> None:
    # Reviewer recovery: every facility "<5" (each >= 1) and the overall "<5" (<= 4) gave 1
    # in each facility.
    mask, facility = _one_per_facility()
    out = pct_by_facility(~mask if negate else mask, facility, FACILITIES)
    cells = [out[f] for f in FACILITIES]
    assert out["all"] == SECONDARY
    assert cells == [SUPPRESSED] * 4
    assert_not_recoverable(cells, out["all"], "one missing per facility")
    reader = CountReader()
    reader.pct("all", out["all"], len(mask), not negate)
    for fac in FACILITIES:
        reader.pct(fac, out[fac], 200, not negate)
    reader.constrain({"all": -1, **{f: 1 for f in FACILITIES}}, 0, 0)
    for fac in FACILITIES:
        low, high = reader.range({fac: 1})
        assert low < high, f"{fac} pinned"


def test_one_shared_mother_key_pair_is_not_pinned(tmp_path: Path) -> None:
    # Reviewer recovery (c2_q9): Q9 showed "<5" rows sharing a key, but the profile's
    # n_nonnull - n_unique = 1 for the key column gave exactly one pair (2 rows).
    classified = _classified()
    keys = [f"K{i}" for i in range(len(classified))]
    keys[1] = keys[0]
    classified["mother_key"] = keys
    raw = pd.DataFrame({"Patient ID": pd.Series(keys, dtype=object)})
    mapping = FieldMapping("mother_key", "hash_key", ("Patient ID",), "confirmed")
    out_dir = tmp_path / "profile"
    write_profile(raw, classified, MappingConfig("t", None, {"mother_key": mapping}), {}, out_dir)
    profile = pd.read_csv(out_dir / "variable_profile.csv", dtype=str).set_index("raw_name")
    assert profile.loc["Patient ID", "n_unique"] == SECONDARY
    assert profile.loc["Patient ID", "n_nonnull"] not in HIDDEN
    q9 = (out_dir / "open_questions.md").read_text(encoding="utf-8").split("## Q9.", 1)[1]
    assert f"Rows sharing a mother_key with another row: {SUPPRESSED}." in q9
    scan_profile_outputs(out_dir)


def test_q6_uses_delivery_date_as_proxy() -> None:
    canonical = _classified()
    text = open_questions_markdown(canonical, pd.DataFrame(index=canonical.index), {})
    q6 = text.split("## Q6.", 1)[1].split("## Q7.", 1)[0]
    assert "admitted_at" in q6.splitlines()[0]  # the question keeps the spec wording
    assert "delivery_date recorded" in q6
    assert "no admission timestamp" in q6
    months = markdown_tables(q6)[0]
    assert {"2023-11", "2024-03"} <= set(months["value"])


def _q9(canonical: pd.DataFrame) -> str:
    text = open_questions_markdown(canonical, pd.DataFrame(index=canonical.index), {})
    return text.split("## Q9.", 1)[1]


def test_q9_reports_shared_mother_keys_without_values() -> None:
    canonical = _classified()
    q9 = _q9(canonical)
    shared = canonical["mother_key"].duplicated(keep=False)
    assert f"Rows sharing a mother_key with another row: {int(shared.sum())}." in q9
    assert "Of these, plurality >= 2: 0." in q9
    for key in canonical["mother_key"].dropna():
        assert key not in q9
    scan_open_questions(
        open_questions_markdown(canonical, pd.DataFrame(index=canonical.index), {}),
        len(canonical),
    )


def test_q9_small_multiple_count_is_suppressed() -> None:
    canonical = _classified()
    shared = canonical.index[canonical["mother_key"].duplicated(keep=False)]
    canonical.loc[shared[:2], "plurality"] = 2
    q9 = _q9(canonical)
    assert f"Of these, plurality >= 2: {SUPPRESSED}." in q9


def test_q9_small_non_multiple_remainder_is_protected() -> None:
    # 2 shared rows that are not multiples: shared - multiples would give 2 if both shown.
    canonical = _classified()
    shared = canonical.index[canonical["mother_key"].duplicated(keep=False)]
    canonical.loc[shared[2:], "plurality"] = 2
    q9 = _q9(canonical)
    total = re.search(r"another row: ([^.]+)\.", q9).group(1)  # type: ignore[union-attr]
    multiples = re.search(r"plurality >= 2: ([^.]+)\.", q9).group(1)  # type: ignore[union-attr]
    assert total in HIDDEN or multiples in HIDDEN
