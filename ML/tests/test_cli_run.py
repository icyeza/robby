import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from robson_engine import load_rule_set
from robson_ml.cli import app
from robson_ml.evaluate import COMPARISON_COLUMNS
from robson_ml.features import load_feature_registry
from robson_ml.robson_run import classify_frame
from tests.synthetic import make_admissions, make_raw_sheet
from tests.test_preregistration import commit_rule, init_repo

REGISTRY = Path("configs/features_v1.yaml").resolve()
N = 1200


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A synthetic project laid out like the real one (data/processed => guarded)."""
    registry = load_feature_registry(REGISTRY)
    (tmp_path / "configs" / "experiments").mkdir(parents=True)
    shutil.copy(REGISTRY, tmp_path / "configs" / "features_v1.yaml")
    (tmp_path / "configs" / "mapping.yaml").write_text(
        yaml.safe_dump({"source": "synthetic", "sheet": "main", "fields": {}}), encoding="utf-8"
    )
    (tmp_path / "configs" / "project.yaml").write_text(
        yaml.safe_dump(
            {
                "raw_path": "data/raw/raw.xlsx",
                "mapping_path": "configs/mapping.yaml",
                "answers_path": "configs/answers.yaml",
                "interim_dir": "data/interim",
                "processed_dir": "data/processed",
                "reports_dir": "reports",
                "features_path": "configs/features_v1.yaml",
                "seed": 3,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "configs" / "experiments" / "b1.yaml").write_text(
        yaml.safe_dump(
            {
                "model": "B1",
                "feature_set": "FS0",
                "missing_strategy": "M0",
                "split": "S1",
                "population": "P_pred",
                "n_trials": 2,
                "seed": 5,
                "n_boot": 20,
            }
        ),
        encoding="utf-8",
    )
    for sub in ("raw", "processed"):
        (tmp_path / "data" / sub).mkdir(parents=True)
    canonical = classify_frame(make_admissions(N, seed=9), load_rule_set())
    canonical.to_parquet(tmp_path / "data" / "processed" / "canonical_robson.parquet", index=False)
    make_raw_sheet(registry, N, seed=9).to_excel(
        tmp_path / "data" / "raw" / "raw.xlsx", index=False, sheet_name="main"
    )
    init_repo(tmp_path)
    return tmp_path


def _invoke(project: Path, *args: str) -> "object":
    cwd = os.getcwd()
    os.chdir(project)
    try:
        return CliRunner().invoke(app, list(args))
    finally:
        os.chdir(cwd)


def test_run_refuses_real_data_without_committed_rule(project: Path) -> None:
    result = _invoke(project, "run", "configs/experiments/b1.yaml")
    assert result.exit_code == 1  # type: ignore[attr-defined]
    assert "PreregistrationError" in result.output  # type: ignore[attr-defined]
    assert not (project / "mlruns").exists()
    assert not (project / "data" / "interim" / "oof").exists()


def test_run_and_compare(project: Path) -> None:
    rule_commit = commit_rule(project)
    result = _invoke(project, "run", "configs/experiments/b1.yaml")
    assert result.exit_code == 0, result.output  # type: ignore[attr-defined]
    output = result.output  # type: ignore[attr-defined]
    assert rule_commit in output
    assert "population version: v1.3" in output
    assert "B1|FS0|M0|S1|P_pred run_id=" in output
    assert "SYN0" not in output
    assert len(list((project / "data" / "interim" / "oof").glob("*.parquet"))) == 1

    result = _invoke(project, "compare")
    assert result.exit_code == 0, result.output  # type: ignore[attr-defined]
    table = pd.read_csv(project / "reports" / "model_comparison.csv")
    assert tuple(table.columns) == COMPARISON_COLUMNS
    assert len(table) == 1
    assert table.loc[0, "model"] == "B1"
    assert table.loc[0, "population_version"] == "v1.3"
    assert not [c for c in table.columns if c == "n" or c.startswith("n_") or "count" in c]


def test_rerun_skips_completed_configuration(project: Path) -> None:
    commit_rule(project)
    first = _invoke(project, "run", "configs/experiments/b1.yaml")
    assert first.exit_code == 0, first.output  # type: ignore[attr-defined]
    second = _invoke(project, "run", "configs/experiments/b1.yaml")
    assert second.exit_code == 0, second.output  # type: ignore[attr-defined]
    assert "already completed" in second.output  # type: ignore[attr-defined]
    assert len(list((project / "data" / "interim" / "oof").glob("*.parquet"))) == 1
