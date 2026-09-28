from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from robson_engine import load_rule_set
from robson_ml.feature_sets import (
    FACILITY,
    FEATURE_SETS,
    LEGACY_ONSET_COLUMNS,
    P_PRED_COMPLETE,
    ModelData,
    allowed_columns,
    build_model_data,
    complete_cases,
    feature_spec,
    robson_group_no_onset,
)
from robson_ml.features import FeatureRegistry, build_raw_features, load_feature_registry
from robson_ml.robson_run import classify_frame
from tests.synthetic import make_admissions, make_raw_sheet

REGISTRY_PATH = Path("configs/features_v1.yaml")
LOHO_SETS = ("FS0", "FS1", "FS2", "FS3", "FS4")


def model_data(
    registry: FeatureRegistry, n: int = 1500, seed: int = 3, population: str = "P_pred"
) -> ModelData:
    canonical = classify_frame(make_admissions(n, seed=seed), load_rule_set())
    raw, _ = build_raw_features(make_raw_sheet(registry, n, seed=seed), registry)
    return build_model_data(canonical, registry, raw, population)


@pytest.fixture(scope="module")
def registry() -> FeatureRegistry:
    return load_feature_registry(REGISTRY_PATH)


@pytest.fixture(scope="module")
def data(registry: FeatureRegistry) -> ModelData:
    return model_data(registry)


def test_population_filter(data: ModelData) -> None:
    """Spec v1.3: P_pred drops planned CS and planned-onset vaginal births, keeps the rest."""
    meta = data.meta
    assert not ((meta["cs"] == 1) & (meta["prelabour_cs_type"] == "planned")).any()
    assert not ((meta["onset_of_labour"] == "prelabour_cs") & (meta["cs"] == 0)).any()
    assert ((meta["onset_of_labour"] == "prelabour_cs") & (meta["cs"] == 1)).any()
    assert ((meta["cs"] == 1) & meta["prelabour_cs_type"].isna()).any()
    assert len(data.x) == len(meta) == len(data.y)
    assert set(np.unique(data.y)) == {0, 1}
    assert not data.legacy_onset
    table = data.exclusion_table.set_index("category")
    assert table.loc["planned CS (elective type)", "n_excluded"] > 0
    assert table.loc["planned-CS onset, vaginal birth", "n_excluded"] > 0
    untyped = table.loc["CS with no recorded type (kept)"]
    assert untyped["n_excluded"] == 0 and untyped["n_kept"] > 0


@pytest.mark.parametrize("fs_name", ["FS0", "FS2"])
def test_complete_cases_keep_only_fully_recorded_rows(data: ModelData, fs_name: str) -> None:
    complete = complete_cases(data, fs_name)
    columns = [c for c in feature_spec(data, fs_name).columns if c != FACILITY]
    expected = data.x[columns].notna().all(axis=1)
    assert complete.population == P_PRED_COMPLETE
    assert len(complete.y) == len(complete.x) == len(complete.meta) == int(expected.sum())
    assert not complete.x[columns].isna().any().any()
    assert list(complete.meta.index) == list(range(len(complete.y)))
    np.testing.assert_array_equal(complete.y, data.y[expected.to_numpy()])
    last = complete.exclusion_table.iloc[-1]
    assert fs_name in last["category"]
    assert last["n_excluded"] == int((~expected).sum())
    assert last["n_cs_excluded"] == int(data.y[~expected.to_numpy()].sum())


def test_complete_cases_only_from_p_pred(registry: FeatureRegistry) -> None:
    onset = model_data(registry, n=400, population="P_pred_onset_coded")
    with pytest.raises(ValueError, match="complete cases"):
        complete_cases(onset, "FS0")
    with pytest.raises(ValueError, match="complete_cases"):
        model_data(registry, n=400, population=P_PRED_COMPLETE)


def test_onset_never_a_feature_in_p_pred(data: ModelData) -> None:
    """Spec v1.3: onset_of_labour and the onset-based robson_group never reach a model."""
    for column in LEGACY_ONSET_COLUMNS:
        assert column not in data.x.columns
        for name in FEATURE_SETS:
            assert column not in feature_spec(data, name).columns
    assert "robson_group_no_onset" in feature_spec(data, "FS0").categorical


def test_registry_including_onset_is_refused(registry: FeatureRegistry, tmp_path: Path) -> None:
    doc = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    for entry in doc["features"]:
        if entry["name"] == "onset_of_labour":
            entry["status"] = "include"
    path = tmp_path / "features.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="onset_of_labour"):
        model_data(load_feature_registry(path), n=200)


def test_onset_coded_population_uses_legacy_onset_features(registry: FeatureRegistry) -> None:
    """P_pred_onset_coded (sensitivity): the v1.2 population with onset and robson_group."""
    legacy = model_data(registry, n=800, population="P_pred_onset_coded")
    assert legacy.legacy_onset
    assert set(legacy.meta["onset_of_labour"]) <= {"spontaneous", "induced"}
    fs0 = feature_spec(legacy, "FS0")
    assert set(LEGACY_ONSET_COLUMNS) <= set(fs0.categorical)
    assert "robson_group_no_onset" not in legacy.x.columns
    assert "onset prelabour_cs planned" in set(legacy.exclusion_table["category"])


def test_legacy_onset_only_for_onset_coded_population(registry: FeatureRegistry) -> None:
    canonical = classify_frame(make_admissions(200, seed=4), load_rule_set())
    raw, _ = build_raw_features(make_raw_sheet(registry, 200, seed=4), registry)
    with pytest.raises(ValueError, match="legacy_onset"):
        build_model_data(canonical, registry, raw, "P_pred", legacy_onset=True)
    with pytest.raises(ValueError, match="population"):
        build_model_data(canonical, registry, raw, "P_pred_sens")


def _robson_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    base: dict[str, object] = {
        "parity": 0,
        "previous_cs_count": 0,
        "plurality": 1,
        "fetal_presentation": "cephalic",
        "gestational_age_weeks": 39.0,
        "ga_band_lower": np.nan,
        "ga_band_upper": np.nan,
        "onset_of_labour": "spontaneous",
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def test_robson_group_no_onset_merges_1_2_and_3_4() -> None:
    rows: list[dict[str, object]] = [
        {},  # group 1 with onset
        {"onset_of_labour": "induced"},  # group 2
        {"onset_of_labour": "prelabour_cs"},  # group 2
        {"onset_of_labour": None},  # partial {1, 2} with onset
        {"parity": 2},  # group 3
        {"parity": 2, "onset_of_labour": "prelabour_cs"},  # group 4
        {"parity": 2, "previous_cs_count": 1},  # 5
        {"fetal_presentation": "breech"},  # 6
        {"parity": 1, "fetal_presentation": "breech"},  # 7
        {"plurality": 2},  # 8
        {"fetal_presentation": "transverse"},  # 9
        {"gestational_age_weeks": 33.0},  # 10
        {"gestational_age_weeks": np.nan, "ga_band_lower": 38.0, "ga_band_upper": 40.857},
        {"gestational_age_weeks": np.nan, "ga_band_lower": 20.0, "ga_band_upper": 33.857},
    ]
    labels = robson_group_no_onset(_robson_frame(rows), load_rule_set()).tolist()
    assert labels == [
        "1_2", "1_2", "1_2", "1_2", "3_4", "3_4", "5", "6", "7", "8", "9", "10", "1_2", "10",
    ]  # fmt: skip


def test_robson_group_no_onset_partial_and_conflict() -> None:
    rows: list[dict[str, object]] = [
        {"parity": None},  # 1/2/3/4/... candidates
        {"gestational_age_weeks": None},  # GA unknown: 1/2 or 10
        {"gestational_age_weeks": np.nan, "ga_band_lower": 35.0, "ga_band_upper": 37.857},
        {"fetal_presentation": "non_cephalic"},  # 6 or 9: partial
        {"parity": 0, "previous_cs_count": 1},  # conflict
    ]
    labels = robson_group_no_onset(_robson_frame(rows), load_rule_set())
    assert labels.tolist() == ["partial"] * len(rows)
    assert labels.dtype == object


def test_robson_group_no_onset_ignores_onset(data: ModelData) -> None:
    meta = data.meta.copy()
    baseline = robson_group_no_onset(meta, load_rule_set())
    meta["onset_of_labour"] = "prelabour_cs"
    pd.testing.assert_series_equal(robson_group_no_onset(meta, load_rule_set()), baseline)
    pd.testing.assert_series_equal(data.x["robson_group_no_onset"], baseline, check_names=False)
    assert set(baseline) <= {"1_2", "3_4", *map(str, range(5, 11)), "partial"}
    assert {"1_2", "3_4", "5", "partial"} <= set(baseline)


def test_no_facility_in_loho_feature_sets(data: ModelData) -> None:
    """Spec §4.5: facility_id is in no FS0-FS4 set, only in FS4_deploy."""
    for name in LOHO_SETS:
        assert FACILITY not in feature_spec(data, name).columns
    assert set(FEATURE_SETS) == {*LOHO_SETS, *(f"{name}_deploy" for name in LOHO_SETS)}
    for name in LOHO_SETS:
        deploy = feature_spec(data, f"{name}_deploy")
        assert FACILITY in deploy.columns and FACILITY in deploy.categorical
        base = feature_spec(data, name).columns
        assert deploy.columns == (*base, FACILITY)


def test_feature_sets_nest_and_partition(data: ModelData) -> None:
    previous: set[str] = set()
    for name in LOHO_SETS:
        spec = feature_spec(data, name)
        assert previous <= set(spec.columns)
        previous = set(spec.columns)
        assert set(spec.categorical) | set(spec.numeric) | set(spec.ordinal) == set(spec.columns)
        assert len(spec.categorical) + len(spec.numeric) + len(spec.ordinal) == len(spec.columns)
    fs0 = feature_spec(data, "FS0")
    assert {"ga_band_lower", "ga_band_upper", "robson_group_no_onset"} <= set(fs0.columns)
    assert "robson_group_no_onset" in fs0.categorical
    assert "gestational_age_band" not in fs0.columns
    assert "bmi" in feature_spec(data, "FS1").columns
    assert "bmi" not in fs0.columns


def test_excluded_features(data: ModelData, registry: FeatureRegistry) -> None:
    """Spec §22: every model column is an include feature (facility only in FS4_deploy)."""
    allowed = allowed_columns(registry)
    for name in LOHO_SETS:
        assert set(feature_spec(data, name).columns) <= allowed
    assert set(data.x.columns) <= allowed | {FACILITY}
    for entry in registry.entries:
        if entry.status != "include" and entry.name != FACILITY:
            assert entry.name not in data.x.columns


def test_excluding_a_feature_removes_it(registry: FeatureRegistry, tmp_path: Path) -> None:
    doc = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    for entry in doc["features"]:
        if entry["name"] in ("maternal_age", "gravidity"):
            entry["status"] = "exclude"
    path = tmp_path / "features.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    changed = load_feature_registry(path)
    data = model_data(changed, n=400)
    assert "maternal_age" not in data.x.columns
    assert "gravidity" not in data.x.columns
    assert "maternal_age" not in feature_spec(data, "FS4").columns


def test_derived_features(data: ModelData) -> None:
    x = data.x
    both = data.meta["height_cm"].notna() & data.meta["weight_kg"].notna()
    assert x.loc[~both, "bmi"].isna().all()
    expected = data.meta["weight_kg"] / (data.meta["height_cm"] / 100) ** 2
    np.testing.assert_allclose(x.loc[both, "bmi"], expected[both])
    missing_ga = data.meta["gestational_age_weeks"].isna().astype(float)
    np.testing.assert_array_equal(x["is_missing_gestational_age_weeks"], missing_ga)
    assert x["is_missing_living_children_band"].isin([0.0, 1.0]).all()
    for column in feature_spec(data, "FS4").numeric:
        assert x[column].dtype == np.float64
    for column in feature_spec(data, "FS4").categorical:
        assert x[column].dtype == object
        assert x[column].dropna().map(type).eq(str).all()


def test_missing_raw_features_raise(registry: FeatureRegistry) -> None:
    canonical = classify_frame(make_admissions(200, seed=4), load_rule_set())
    with pytest.raises(ValueError, match="raw feature"):
        build_model_data(canonical, registry, pd.DataFrame(index=range(len(canonical))))
