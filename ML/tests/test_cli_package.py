"""robson-ml package end to end on a synthetic project (data/processed => guarded)."""

import json
from pathlib import Path

import pandas as pd

from robson_serving.artefact import ARTEFACT_FILES, MANIFEST_FILE, MODEL_FILE, REFERENCE_FILE
from robson_serving.contract import check_golden
from tests.test_cli_run import _invoke, project  # noqa: F401  (fixture)
from tests.test_deploy import _write_deployment
from tests.test_preregistration import commit_rule

OUT = "artefacts/packaged/readiness-test"


def _fit(project: Path) -> None:  # noqa: F811
    commit_rule(project)
    config = _write_deployment(project, "readiness-test", True)
    result = _invoke(project, "fit-deploy", config.as_posix())
    assert result.exit_code == 0, result.output  # type: ignore[attr-defined]


def test_package_end_to_end(project: Path) -> None:  # noqa: F811
    _fit(project)
    result = _invoke(project, "package", "artefacts/readiness-test", "--out", OUT)
    assert result.exit_code == 0, result.output  # type: ignore[attr-defined]
    output = result.output  # type: ignore[attr-defined]
    assert "parity: pass" in output
    assert "golden: 200 admissions" in output
    assert "SYN" not in output
    out = project / OUT
    assert sorted(p.name for p in out.iterdir()) == sorted(ARTEFACT_FILES)
    assert b"robson_ml" not in (out / MODEL_FILE).read_bytes()
    manifest_text = (out / MANIFEST_FILE).read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["version_label"] == "readiness-test"
    assert manifest["feature_set"] == "FS2_deploy"
    assert manifest["runtime_libraries"] == ["scikit-learn"]
    assert set(manifest["evaluation_metrics"]) == {"S1", "S2", "S3", "S4"}
    assert manifest["beats_robson_baseline"] is None  # no harness runs in this project
    assert check_golden(out).ok
    canonical = pd.read_parquet(project / "data" / "processed" / "canonical_robson.parquet")
    reference_text = (out / REFERENCE_FILE).read_text(encoding="utf-8")
    for text in (manifest_text, reference_text):
        assert "SYN" not in text
        assert not any(v in text for v in canonical["admission_id"].dropna().astype(str))

    again = _invoke(project, "package", "artefacts/readiness-test", "--out", OUT)
    assert again.exit_code == 1  # type: ignore[attr-defined]


def test_package_refuses_data_that_changed_since_the_fit(project: Path) -> None:  # noqa: F811
    _fit(project)
    path = project / "data" / "processed" / "canonical_robson.parquet"
    pd.read_parquet(path).iloc[:-1].to_parquet(path, index=False)
    result = _invoke(project, "package", "artefacts/readiness-test", "--out", OUT)
    assert result.exit_code == 1  # type: ignore[attr-defined]
    assert "ValueError" in result.output  # type: ignore[attr-defined]
    assert not (project / OUT).exists()
