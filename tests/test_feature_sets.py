from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from robson_engine import load_rule_set
from robson_ml.feature_sets import (
    FACILITY,
    FEATURE_SETS,
    ModelData,
    allowed_columns,
    build_model_data,
    feature_spec,
)
from robson_ml.features import FeatureRegistry, build_raw_features, load_feature_registry
from robson_ml.robson_run import classify_frame
from tests.synthetic import make_admissions, make_raw_sheet

REGISTRY_PATH = Path("configs/features_v1.yaml")
LOHO_SETS = ("FS0", "FS1", "FS2", "FS3", "FS4")


def model_data(registry: FeatureRegistry, n: int = 1500, seed: int = 3) -> ModelData:
    canonical = classify_frame(make_admissions(n, seed=seed), load_rule_set())
    raw, _ = build_raw_features(make_raw_sheet(registry, n, seed=seed), registry)
    return build_model_data(canonical, registry, raw)


@pytest.fixture(scope="module")
def registry() -> FeatureRegistry:
    return load_feature_registry(REGISTRY_PATH)


@pytest.fixture(scope="module")
def data(registry: FeatureRegistry) -> ModelData:
    return model_data(registry)


def test_population_filter(data: ModelData) -> None:
    """Spec §22: P_pred contains no prelabour_cs onset (and only rows with an outcome)."""
    assert set(data.meta["onset_of_labour"].dropna()) <= {"spontaneous", "induced"}
    assert data.meta["onset_of_labour"].notna().all()
    assert len(data.x) == len(data.meta) == len(data.y)
    assert set(np.unique(data.y)) == {0, 1}
    assert {log.reason for log in data.exclusions} >= {"onset prelabour_cs planned"}


def test_no_facility_in_loho_feature_sets(data: ModelData) -> None:
    """Spec §4.5: facility_id is in no FS0-FS4 set, only in FS4_deploy."""
    for name in LOHO_SETS:
        assert FACILITY not in feature_spec(data, name).columns
    deploy = feature_spec(data, "FS4_deploy")
    assert FACILITY in deploy.columns
    assert set(deploy.columns) - set(feature_spec(data, "FS4").columns) == {FACILITY}


def test_feature_sets_nest_and_partition(data: ModelData) -> None:
    previous: set[str] = set()
    for name in FEATURE_SETS:
        spec = feature_spec(data, name)
        assert previous <= set(spec.columns)
        previous = set(spec.columns)
        assert set(spec.categorical) | set(spec.numeric) | set(spec.ordinal) == set(spec.columns)
        assert len(spec.categorical) + len(spec.numeric) + len(spec.ordinal) == len(spec.columns)
    fs0 = feature_spec(data, "FS0")
    assert {"ga_band_lower", "ga_band_upper", "robson_group", "onset_of_labour"} <= set(fs0.columns)
    assert "robson_group" in fs0.categorical
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
