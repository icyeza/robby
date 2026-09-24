from pathlib import Path

import pandas as pd
import pytest
import yaml

from robson_ml.features import (
    FeatureRegistryError,
    build_raw_features,
    check_coverage,
    load_feature_registry,
)

MINIMAL_FEATURES: list[dict[str, object]] = [
    {
        "name": "facility_id",
        "raw_name": "Hospital Name",
        "source": "canonical",
        "status": "exclude",
        "available_at_admission": True,
        "group": "G_context",
        "reason": "never a feature in LOHO runs",
    },
    {
        "name": "insurance_type",
        "raw_name": "Health Insurance Type",
        "source": "raw",
        "kind": "text",
        "status": "include",
        "available_at_admission": True,
        "group": "G_maternal",
        "reason": "sociodemographic, known at admission",
    },
    {
        "name": "gravidity",
        "raw_name": "Gravidity",
        "source": "raw",
        "kind": "integer",
        "status": "include",
        "available_at_admission": True,
        "group": "G_obs",
        "reason": "obstetric history",
        "range": [1, 30],
    },
    {
        "name": "sector",
        "raw_name": "Sector",
        "source": "raw",
        "status": "exclude",
        "available_at_admission": True,
        "group": "identifier",
        "reason": "quasi-identifier",
    },
    {
        "name": "robson_group",
        "raw_name": None,
        "source": "derived",
        "status": "include",
        "available_at_admission": True,
        "group": "G_robson",
        "reason": "engine output",
    },
]


def _write_registry(tmp_path: Path, features: list[dict[str, object]]) -> Path:
    path = tmp_path / "features_v1.yaml"
    path.write_text(yaml.safe_dump({"version": "test", "features": features}), encoding="utf-8")
    return path


def test_load_feature_registry_parses_minimal_set(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, MINIMAL_FEATURES)
    registry = load_feature_registry(path)
    assert len(registry.entries) == 5
    assert len(registry.sha256) == 64
    included = registry.included()
    assert {e.name for e in included} == {"insurance_type", "gravidity", "robson_group"}


def test_load_feature_registry_rejects_unknown_key(tmp_path: Path) -> None:
    bad = [dict(MINIMAL_FEATURES[1])]
    bad[0]["bogus"] = 1
    path = _write_registry(tmp_path, bad)
    with pytest.raises(FeatureRegistryError):
        load_feature_registry(path)


def test_load_feature_registry_rejects_bad_status(tmp_path: Path) -> None:
    bad = [dict(MINIMAL_FEATURES[1])]
    bad[0]["status"] = "maybe"
    path = _write_registry(tmp_path, bad)
    with pytest.raises(FeatureRegistryError):
        load_feature_registry(path)


def test_load_feature_registry_rejects_bad_group(tmp_path: Path) -> None:
    bad = [dict(MINIMAL_FEATURES[1])]
    bad[0]["group"] = "G_nonsense"
    path = _write_registry(tmp_path, bad)
    with pytest.raises(FeatureRegistryError):
        load_feature_registry(path)


def test_load_feature_registry_include_requires_available_at_admission(tmp_path: Path) -> None:
    bad = [dict(MINIMAL_FEATURES[1])]
    bad[0]["available_at_admission"] = False
    path = _write_registry(tmp_path, bad)
    with pytest.raises(FeatureRegistryError):
        load_feature_registry(path)


def test_load_feature_registry_rejects_missing_kind_for_included_raw(tmp_path: Path) -> None:
    bad = [dict(MINIMAL_FEATURES[1])]
    del bad[0]["kind"]
    path = _write_registry(tmp_path, bad)
    with pytest.raises(FeatureRegistryError):
        load_feature_registry(path)


def test_load_feature_registry_rejects_duplicate_raw_column(tmp_path: Path) -> None:
    dup = [dict(MINIMAL_FEATURES[1]), dict(MINIMAL_FEATURES[1], name="insurance_type_2")]
    path = _write_registry(tmp_path, dup)
    with pytest.raises(FeatureRegistryError):
        load_feature_registry(path)


def test_load_feature_registry_rejects_duplicate_name(tmp_path: Path) -> None:
    dup = [dict(MINIMAL_FEATURES[1]), dict(MINIMAL_FEATURES[1], raw_name="Other Column")]
    path = _write_registry(tmp_path, dup)
    with pytest.raises(FeatureRegistryError):
        load_feature_registry(path)


def test_load_feature_registry_rejects_derived_with_raw_name(tmp_path: Path) -> None:
    bad = [dict(MINIMAL_FEATURES[4])]
    bad[0]["raw_name"] = "something"
    path = _write_registry(tmp_path, bad)
    with pytest.raises(FeatureRegistryError):
        load_feature_registry(path)


def test_check_coverage_reports_missing_and_extra(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, MINIMAL_FEATURES)
    registry = load_feature_registry(path)
    raw_columns = ["Hospital Name", "Health Insurance Type", "Gravidity", "An Extra Column"]
    missing, extra = check_coverage(registry, raw_columns)
    assert missing == ["An Extra Column"]
    assert extra == ["Sector"]


def test_build_raw_features_maps_raw_columns(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, MINIMAL_FEATURES)
    registry = load_feature_registry(path)
    raw = pd.DataFrame(
        {
            "Health Insurance Type": ["RSSB", "Mutuelle", None],
            "Gravidity": ["1", "3", "abc"],
            "Hospital Name": ["FAC_A", "FAC_B", "FAC_C"],
            "Sector": ["x", "y", "z"],
        }
    )
    features, reports = build_raw_features(raw, registry)
    assert list(features.columns) == ["insurance_type", "gravidity"]
    assert features["insurance_type"].tolist() == ["RSSB", "Mutuelle", None]
    assert features["gravidity"].tolist() == [1, 3, pd.NA]  # "abc" unparsed -> missing
    assert {r.canonical for r in reports} == {"insurance_type", "gravidity"}
