import os
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from robson_ml.cli import app
from robson_ml.schema import CANONICAL_DTYPES
from tests.synthetic import make_admissions

KIND_BY_DTYPE = {
    "Int64": "integer",
    "float64": "float",
    "datetime64[ns]": "datetime",
    "object": "text",
}


@pytest.fixture
def project(tmp_path: Path) -> Path:
    df = make_admissions(1200, seed=31)
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    df.drop(columns=["admission_id"]).to_excel(raw_dir / "raw.xlsx", index=False, sheet_name="main")
    fields: dict[str, dict[str, object]] = {
        "admission_id": {"kind": "row_key", "status": "confirmed"}
    }
    for name, dtype in CANONICAL_DTYPES.items():
        if name == "admission_id":
            continue
        kind = "gestational_age" if name == "gestational_age_weeks" else KIND_BY_DTYPE[dtype]
        spec: dict[str, object] = {"raw": name, "kind": kind, "status": "confirmed"}
        if kind == "gestational_age":
            spec["format"] = "decimal_weeks"
        fields[name] = spec
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "mapping_ur_cmhs.yaml").write_text(
        yaml.safe_dump({"source": "synthetic", "sheet": "main", "fields": fields})
    )
    (configs / "project.yaml").write_text(
        yaml.safe_dump(
            {
                "raw_path": "data/raw/raw.xlsx",
                "mapping_path": "configs/mapping_ur_cmhs.yaml",
                "answers_path": "configs/open_questions_answers.yaml",
                "interim_dir": "data/interim",
                "processed_dir": "data/processed",
                "reports_dir": "reports",
                "seed": 5,
            }
        )
    )
    return tmp_path


def _invoke(project: Path, command: str) -> "object":
    cwd = os.getcwd()
    os.chdir(project)
    try:
        return CliRunner().invoke(app, [command])
    finally:
        os.chdir(cwd)


def test_pipeline_end_to_end(project: Path) -> None:
    for command in ["ingest", "robson", "profile"]:
        result = _invoke(project, command)
        assert result.exit_code == 0, result.output
    assert (project / "data/interim/raw_inventory.json").exists()
    assert (project / "data/interim/mapping_report.json").exists()
    assert (project / "data/processed/canonical_robson.parquet").exists()
    assert (project / "data/interim/robson_handcheck.csv").exists()
    profile = pd.read_csv(project / "reports/profile/variable_profile.csv", dtype=str)
    numeric = pd.to_numeric(profile["n_nonnull"], errors="coerce")
    assert not ((numeric > 0) & (numeric < 5)).any()
    assert (project / "reports/profile/robson_inputs.md").exists()
    assert (project / "reports/profile/open_questions.md").exists()


def test_ingest_without_mapping_fields_does_inventory_only(project: Path) -> None:
    (project / "configs/mapping_ur_cmhs.yaml").write_text("source: x\nsheet: ~\nfields: {}\n")
    result = _invoke(project, "ingest")
    assert result.exit_code == 0
    assert "inventory only" in result.output
    assert not (project / "data/processed/canonical.parquet").exists()


def test_console_output_is_aggregate_only(project: Path) -> None:
    outputs = "".join(_invoke(project, c).output for c in ["ingest", "robson", "profile"])
    df = make_admissions(1200, seed=31)
    for column in ["facility_id", "recorded_indication", "mode_of_delivery"]:
        for value in df[column].dropna().astype(str).unique():
            assert value not in outputs, f"row value from {column} printed"
    # no bare small counts: every printed "=<n>" count is 0 or >= 5 or "<5"
    import re

    for match in re.findall(r"=(\d+)\b", outputs):
        assert int(match) == 0 or int(match) >= 5


def test_failing_command_does_not_leak_row_data(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If a command errors out, typer's traceback must not print any row-level sentinel."""
    sentinel = "ROW_LEVEL_SENTINEL_VALUE_XYZ"

    import robson_ml.cli as cli_module

    monkeypatch.setattr(
        cli_module,
        "load_project_config",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError(sentinel)),
    )
    result = _invoke(project, "ingest")
    assert result.exit_code != 0
    assert sentinel not in result.output
