"""``robson-ml casemix`` and ``robson-ml missingness`` on a small synthetic project, with and
without the (FAKE) reference files."""

import os
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from robson_ml.cli import app
from tests.synthetic_project import build_synthetic_project
from tests.synthetic_references import write_fake_references

REPO = Path(__file__).resolve().parents[1]


def _invoke(project: Path, *args: str) -> str:
    cwd = os.getcwd()
    os.chdir(project)
    try:
        result = CliRunner().invoke(app, list(args))
    finally:
        os.chdir(cwd)
    assert result.exit_code == 0, result.output
    return result.output


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = build_synthetic_project(tmp_path_factory.mktemp("phase_g"), REPO, n=500, seed=13)
    path = root / "configs" / "analysis.yaml"
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    settings["casemix"]["n_boot"] = 5
    settings["missingness"]["n_draws"] = 5
    path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    _invoke(root, "ingest")
    _invoke(root, "robson")
    return root


def _no_small_counts(table: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        numeric = pd.to_numeric(table[column], errors="coerce")
        assert not ((numeric > 0) & (numeric < 5)).any(), column


def test_without_references_reports_not_available(project: Path) -> None:
    output = _invoke(project, "casemix")
    assert "reference file absent" in output
    assert "not applicable" in output
    out = project / "reports" / "casemix"
    assert "not available" in (out / "robson_audit_table.md").read_text(encoding="utf-8")
    assert "ref_cs_rate" not in pd.read_csv(out / "robson_audit_table.csv").columns
    output = _invoke(project, "missingness")
    assert "Not run" in output
    assert not list((project / "reports" / "missingness").glob("under_recording_*"))


def test_with_fake_references(project: Path) -> None:
    write_fake_references(project)
    try:
        output = _invoke(project, "casemix")
        assert "FAKE TEST FIXTURE" in output
        out = project / "reports" / "casemix"
        audit = pd.read_csv(out / "robson_audit_table.csv", dtype=str, keep_default_na=False)
        assert "ref_cs_rate" in audit.columns
        _no_small_counts(audit, ["n", "n_cs"])
        oe = pd.read_csv(out / "observed_expected.csv", dtype=str, keep_default_na=False)
        _no_small_counts(oe, ["n", "n_resolved", "n_cs_resolved", "n_scored", "n_cs_scored"])
        _invoke(project, "missingness")
        miss = project / "reports" / "missingness"
        assert (miss / "under_recording_preeclampsia_scenarios.csv").exists()
        table = pd.read_csv(miss / "missingness_by_facility.csv", dtype=str)
        _no_small_counts(table, ["n", "n_missing"])
    finally:
        for path in (project / "data" / "reference").glob("*.yaml"):
            path.unlink()
