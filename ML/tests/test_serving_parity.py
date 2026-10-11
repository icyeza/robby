"""Serving vs training on the real data (data-sharing agreement: aggregates only).

Every assertion compares counts or a maximum difference; nothing prints a row or a value. The
whole module is skipped when the private data are not present (as on any machine but the
project lead's).
"""

from pathlib import Path

import pandas as pd
import pytest

from robson_ml.cli import CANONICAL_ROBSON, PROJECT_CONFIG, load_project_config
from robson_ml.feature_sets import ModelData, build_model_data, feature_spec
from robson_ml.features import build_raw_features, load_feature_registry
from robson_ml.ingest import read_workbook, select_sheet
from robson_ml.mapping import load_mapping
from robson_ml.package import frame_mismatches, parity_report
from robson_ml.populations import P_PRED
from robson_serving import load
from robson_serving.features import build_frame, required_inputs

PACKAGED = Path("artefacts/packaged/readiness-v1.0")


def _real_data_present() -> bool:
    if not PROJECT_CONFIG.is_file():
        return False
    cfg = load_project_config()
    return (cfg.processed_dir / CANONICAL_ROBSON).is_file() and cfg.raw_path.is_file()


pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(not _real_data_present(), reason="real data not present"),
]


@pytest.fixture(scope="module")
def real_data() -> ModelData:
    cfg = load_project_config()
    registry = load_feature_registry(cfg.features_path)
    canonical = pd.read_parquet(cfg.processed_dir / CANONICAL_ROBSON)
    mapping = load_mapping(cfg.mapping_path)
    raw = select_sheet(read_workbook(cfg.raw_path), mapping.sheet).reset_index(drop=True)
    raw_features, _ = build_raw_features(raw, registry)
    return build_model_data(canonical, registry, raw_features, P_PRED)


def test_serving_features_rebuild_every_training_column(real_data: ModelData) -> None:
    spec = feature_spec(real_data, "FS4_deploy")
    columns = [(c, real_data.kinds[c]) for c in spec.columns]
    inputs = list(required_inputs(spec.columns))
    rebuilt = build_frame(real_data.meta[inputs], columns)
    assert frame_mismatches(real_data.x[list(spec.columns)], rebuilt) == {}


@pytest.mark.skipif(not (PACKAGED / "manifest.json").is_file(), reason="not packaged yet")
def test_packaged_scorer_reproduces_training_rows_and_probabilities(real_data: ModelData) -> None:
    report = parity_report(load(PACKAGED), real_data)
    assert report.schema_errors == {}
    assert report.column_mismatches == {}
    assert report.max_probability_difference <= 1e-12
