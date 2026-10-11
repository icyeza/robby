"""Packaging a deployment artefact for serving (robson_ml.package). Synthetic only."""

import dataclasses
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

from robson_ml.calibration import CalibratedModel, calibrate
from robson_ml.feature_sets import FACILITY, ModelData, feature_spec
from robson_ml.features import FeatureRegistry, load_feature_registry
from robson_ml.models import get_model
from robson_ml.package import (
    PackageResult,
    build_schema,
    evaluation_summaries,
    frame_mismatches,
    input_field,
    manifest_fields,
    package_artefact,
    parity_report,
)
from robson_ml.populations import POPULATION_VERSION
from robson_ml.schema import MATERNAL_AGE_RANGE
from robson_serving import load
from robson_serving.artefact import (
    ARTEFACT_FILES,
    MANIFEST_FIELDS,
    MODEL_FILE,
    REFERENCE_FILE,
    WRITER_FIELDS,
    sha256_file,
)
from robson_serving.contract import check_golden
from robson_serving.features import required_inputs
from robson_serving.schema import CATEGORY, INTEGER, NUMBER
from tests.test_feature_sets import model_data

REGISTRY_PATH = Path("configs/features_v1.yaml")
SPLITS = ("S1", "S2", "S3", "S4")


def _fit_summary(model: CalibratedModel, features: list[str]) -> dict[str, Any]:
    return {
        "version_label": "readiness-test",
        "split_scheme": "S5",
        "config": {
            "model": "logreg_l2",
            "feature_set": "FS2",
            "use_facility": True,
            "missing_strategy": "M2",
            "population": "P_pred",
            "seed": 0,
            "n_trials": 2,
            "version_label": "readiness-test",
        },
        "algorithm": "logreg_l2",
        "family": "linear",
        "feature_set": "FS2_deploy",
        "use_facility": True,
        "facility_decision": None,
        "features": features,
        "hyperparameters": {"C": 1.0},
        "calibration": {"method": model.choice.method, "cv_brier": dict(model.choice.cv_brier)},
        "population": "P_pred",
        "population_version": POPULATION_VERSION,
        "training_window_start": "2023-11",
        "training_window_end": "2024-03",
        "seed": 0,
        "git_commit": "synthetic",
        "selection_rule_commit": "synthetic",
        "data_hash": "synthetic",
        "features_yaml_hash": "synthetic",
        "library_versions": {},
        "mlflow_run_id": "synthetic",
    }


@pytest.fixture(scope="module")
def registry() -> FeatureRegistry:
    return load_feature_registry(REGISTRY_PATH)


@pytest.fixture(scope="module")
def data(registry: FeatureRegistry) -> ModelData:
    return model_data(registry)


@pytest.fixture(scope="module")
def legacy_dir(tmp_path_factory: pytest.TempPathFactory, data: ModelData) -> Path:
    """An artefact laid out like fit-deploy's: model.joblib + fit_summary.json."""
    fs = feature_spec(data, "FS2_deploy")
    x = data.x[list(fs.columns)]
    params = {"C": 1.0, "missing_strategy": "M2", "seed": 0}
    pipe = get_model("logreg_l2").build(params, fs).fit(x, data.y)
    model = calibrate(pipe, x, data.y, np.arange(0, len(x), 2), seed=0)
    out = tmp_path_factory.mktemp("legacy") / "readiness-test"
    out.mkdir()
    joblib.dump(model, out / "model.joblib")
    summary = _fit_summary(model, list(fs.columns))
    (out / "fit_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return out


@pytest.fixture(scope="module")
def packaged(
    tmp_path_factory: pytest.TempPathFactory,
    legacy_dir: Path,
    data: ModelData,
    registry: FeatureRegistry,
) -> PackageResult:
    out = tmp_path_factory.mktemp("packaged") / "readiness-test"
    return package_artefact(legacy_dir, out, data, registry, pd.DataFrame(), n_golden=50)


def test_build_schema_reads_the_fitted_pipeline(
    legacy_dir: Path, registry: FeatureRegistry, data: ModelData
) -> None:
    model = joblib.load(legacy_dir / "model.joblib")
    features = list(feature_spec(data, "FS2_deploy").columns)
    schema = build_schema(model, features, registry)
    assert schema.names == tuple(features)
    assert {c.name: c.kind for c in schema.columns} == {f: data.kinds[f] for f in features}
    facility = next(c for c in schema.columns if c.name == FACILITY)
    assert facility.levels == tuple(sorted(data.x[FACILITY].dropna().unique()))
    assert schema.unseen_allowed == (FACILITY,)
    assert [f.name for f in schema.inputs] == list(required_inputs(features))
    fields = {f.name: f for f in schema.inputs}
    assert fields["maternal_age"].kind == NUMBER
    assert fields["maternal_age"].range == MATERNAL_AGE_RANGE
    assert fields["parity"].kind == INTEGER and fields["parity"].range == (0.0, None)
    gravidity = next(e for e in registry.entries if e.name == "gravidity")
    assert gravidity.mapping is not None
    assert fields["gravidity"].range == gravidity.mapping.valid_range
    assert fields["fetal_presentation"].kind == CATEGORY
    presentation = set(fields["fetal_presentation"].levels or ())
    expected_coarsen = "non_cephalic" in presentation and not (
        presentation & {"breech", "transverse", "oblique"}
    )
    assert schema.coarsen_presentation == expected_coarsen


def test_input_field_rejects_an_unknown_name(registry: FeatureRegistry) -> None:
    with pytest.raises(ValueError, match="not a canonical field"):
        input_field("no_such_field", registry, {})


def test_evaluation_summaries_prefer_the_deploy_set_and_check_the_baseline() -> None:
    base = {
        "model": "logreg_l2",
        "missing_strategy": "M2",
        "population": "P_pred",
        "data_hash": "h",
    }
    runs = pd.DataFrame(
        [
            {
                **base,
                "run_id": "r1",
                "split": "S1",
                "feature_set": "FS2",
                "mean_auc": 0.70,
                "pooled_brier": 0.15,
            },
            {**base, "run_id": "r3", "split": "S3", "feature_set": "FS2", "mean_auc": 0.72},
            {
                **base,
                "run_id": "r3d",
                "split": "S3",
                "feature_set": "FS2_deploy",
                "mean_auc": 0.73,
            },
            {
                **base,
                "run_id": "other",
                "split": "S2",
                "feature_set": "FS2",
                "mean_auc": 0.6,
                "data_hash": "different",
            },
            {
                "model": "B1",
                "missing_strategy": "M0",
                "population": "P_pred",
                "data_hash": "h",
                "run_id": "b1",
                "split": "S1",
                "feature_set": "FS0",
                "mean_auc": 0.65,
            },
        ]
    )
    summary = {
        "config": {
            "model": "logreg_l2",
            "feature_set": "FS2",
            "missing_strategy": "M2",
            "population": "P_pred",
        },
        "feature_set": "FS2_deploy",
        "data_hash": "h",
    }
    metrics, beats = evaluation_summaries(runs, summary)
    assert metrics["S1"] == {
        "run_id": "r1",
        "feature_set": "FS2",
        "mean_auc": 0.70,
        "pooled_brier": 0.15,
    }
    assert metrics["S3"] is not None and metrics["S3"]["run_id"] == "r3d"
    assert metrics["S2"] is None and metrics["S4"] is None
    assert beats is True
    assert evaluation_summaries(pd.DataFrame(), summary) == (dict.fromkeys(SPLITS), None)


def test_manifest_fields_cover_the_manifest(legacy_dir: Path) -> None:
    summary = json.loads((legacy_dir / "fit_summary.json").read_text(encoding="utf-8"))
    fields = manifest_fields(summary, dict.fromkeys(SPLITS), None, "robson-v1.0")
    assert set(MANIFEST_FIELDS) - WRITER_FIELDS <= fields.keys()
    assert fields["runtime_libraries"] == ["scikit-learn"]
    xgb = manifest_fields({**summary, "algorithm": "xgboost"}, {}, None, "robson-v1.0")
    assert xgb["runtime_libraries"] == ["scikit-learn", "xgboost"]


def test_frame_mismatches_count_rows_only() -> None:
    labels = pd.Series(["x", np.nan, "y"], dtype=object)
    expected = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": labels})
    actual = pd.DataFrame({"a": [1.0, np.nan, 4.0], "b": labels.copy()})
    assert frame_mismatches(expected, actual) == {"a": 1}
    assert frame_mismatches(expected, actual.astype({"a": object})) == {"a": 3}
    assert frame_mismatches(expected, actual.drop(columns="b")) == {"a": 1, "b": 3}


def test_package_artefact_round_trip(packaged: PackageResult, data: ModelData) -> None:
    out = packaged.out_dir
    assert packaged.ok, packaged.parity.summary()
    assert packaged.parity.schema_errors == {} and packaged.parity.column_mismatches == {}
    assert packaged.parity.max_probability_difference <= 1e-12
    assert sorted(p.name for p in out.iterdir()) == sorted(ARTEFACT_FILES)
    assert b"robson_ml" not in (out / MODEL_FILE).read_bytes()
    assert packaged.manifest["artefact_hash"] == sha256_file(out / MODEL_FILE)
    assert check_golden(out).ok
    assert load(out).version_label == "readiness-test"
    reference = (out / REFERENCE_FILE).read_text(encoding="utf-8")
    assert "SYN" not in reference
    assert not any(str(v) in reference for v in data.meta["admission_id"].head(50))


def test_package_refuses_another_population(
    legacy_dir: Path, data: ModelData, registry: FeatureRegistry, tmp_path: Path
) -> None:
    other = dataclasses.replace(data, population="P_pred_onset_coded")
    with pytest.raises(ValueError, match="population"):
        package_artefact(legacy_dir, tmp_path / "x", other, registry, pd.DataFrame())
    assert not (tmp_path / "x").exists()


def test_parity_report_detects_a_divergent_training_frame(
    packaged: PackageResult, data: ModelData
) -> None:
    tampered = dataclasses.replace(data, x=data.x.assign(bmi=data.x["bmi"] + 1.0))
    report = parity_report(load(packaged.out_dir), tampered)
    assert set(report.column_mismatches) == {"bmi"}
    assert report.column_mismatches["bmi"] == int(data.x["bmi"].notna().sum())
    assert not report.ok
