import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import robson_serving
from robson_engine import load_rule_set
from robson_serving import IncompatibleArtefact, SchemaError, Scorer, load
from robson_serving.artefact import MANIFEST_FILE, MODEL_FILE, SCHEMA_FILE, sha256_file
from robson_serving.calibration import CalibratedModel
from robson_serving.features import ROBSON_NO_ONSET, build_frame
from robson_serving.schema import UNSEEN_LEVEL, FeatureSchema

GROUP_5 = {
    "parity": 1,
    "previous_cs_count": 1,
    "plurality": 1,
    "fetal_presentation": "cephalic",
    "gestational_age_weeks": 39.0,
    "maternal_age": 30,
    "height_cm": 160.0,
    "weight_kg": 64.0,
    "education_level": "primary",
    "facility_id": None,
}


def test_the_public_api() -> None:
    assert set(robson_serving.__all__) == {
        "IncompatibleArtefact",
        "SchemaError",
        "Scorer",
        "__version__",
        "load",
    }


def test_load_returns_a_scorer(artefact_dir: Path, schema: FeatureSchema) -> None:
    scorer = load(artefact_dir)
    assert scorer.version_label == "synthetic-v0"
    assert scorer.feature_names == schema.names


def test_score_matches_the_model_on_the_training_path(
    artefact_dir: Path,
    fitted: tuple[CalibratedModel, pd.DataFrame],
    records: pd.DataFrame,
    kinds: dict[str, str],
) -> None:
    model, _ = fitted
    scorer = load(artefact_dir)
    sample = records.head(40)
    expected = model.predict_proba(build_frame(sample, kinds.items(), load_rule_set()))[:, 1]
    np.testing.assert_array_equal(np.asarray(scorer.score(sample)), expected)
    assert scorer.score(sample.to_dict("records")) == scorer.score(sample)
    assert scorer.score([]) == []


def test_build_row_returns_the_vector_that_is_scored(artefact_dir: Path) -> None:
    scorer = load(artefact_dir)
    row = scorer.build_row(GROUP_5)
    assert list(row) == list(scorer.feature_names)
    assert row[ROBSON_NO_ONSET] == "5"
    assert row["bmi"] == pytest.approx(25.0)
    assert row["facility_id"] == UNSEEN_LEVEL
    assert row["ga_band_lower"] is None
    assert row["is_missing_gestational_age_weeks"] == 0.0
    assert row["parity"] == 1.0
    assert json.loads(json.dumps(row)) == row
    probability = scorer.predict(scorer.model_frame([row]))[0]
    assert scorer.score([GROUP_5]) == [probability]


def test_an_unknown_derived_level_is_a_schema_error(artefact_dir: Path) -> None:
    scorer = load(artefact_dir)
    columns = tuple(
        replace(c, levels=("partial",)) if c.name == ROBSON_NO_ONSET else c
        for c in scorer.schema.columns
    )
    narrowed = Scorer(
        scorer.model, replace(scorer.schema, columns=columns), scorer.manifest, scorer.rule_set
    )
    with pytest.raises(SchemaError) as info:
        narrowed.build_row(GROUP_5)
    assert info.value.field == ROBSON_NO_ONSET


def test_schema_errors_reach_the_caller(artefact_dir: Path) -> None:
    with pytest.raises(SchemaError, match="maternal_age"):
        load(artefact_dir).score([{**GROUP_5, "maternal_age": 99}])


def test_load_refuses_another_rule_set(artefact_dir: Path) -> None:
    other = replace(load_rule_set(), version_label="robson-v9")
    with pytest.raises(IncompatibleArtefact, match="rule set"):
        load(artefact_dir, rule_set=other)


def test_load_refuses_a_feature_list_that_differs_from_the_schema(artefact_dir: Path) -> None:
    path = artefact_dir / MANIFEST_FILE
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["feature_list"] = list(reversed(manifest["feature_list"]))
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(IncompatibleArtefact, match="different features"):
        load(artefact_dir)


def test_load_refuses_a_model_that_cannot_be_unpickled(artefact_dir: Path) -> None:
    (artefact_dir / MODEL_FILE).write_bytes(b"not a pickle")
    path = artefact_dir / MANIFEST_FILE
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["artefact_hash"] = sha256_file(artefact_dir / MODEL_FILE)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(IncompatibleArtefact, match="could not be loaded"):
        load(artefact_dir)


def test_load_refuses_a_missing_schema(artefact_dir: Path) -> None:
    (artefact_dir / SCHEMA_FILE).unlink()
    with pytest.raises(IncompatibleArtefact, match=SCHEMA_FILE):
        load(artefact_dir)
