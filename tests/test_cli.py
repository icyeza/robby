import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from robson_ml.cli import app, load_project_config
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
        if name == "mother_key":
            # The synthetic mother_key column stands in for the raw patient identifier.
            kind = "hash_key"
        spec: dict[str, object] = {"raw": name, "kind": kind, "status": "confirmed"}
        if kind == "gestational_age":
            spec["format"] = "decimal_weeks"
        if name == "delivery_date":
            spec["date_only"] = True
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
    canonical = pd.read_parquet(project / "data/processed/canonical_robson.parquet")
    assert str(canonical["delivery_date"].dtype) == "datetime64[ns]"
    assert canonical["delivery_date"].notna().all()
    assert str(canonical["ga_band_lower"].dtype) == "float64"
    assert canonical["ga_band_lower"].notna().any()
    assert canonical["mother_key"].str.startswith("MK_").all()
    assert (canonical["fetal_presentation"] == "non_cephalic").any()
    reports = "".join(
        p.read_text(encoding="utf-8") for p in (project / "reports/profile").iterdir()
    )
    assert "MK_" not in reports


SALT_FILE = "data/interim/mother_key.salt"


def _raw_ids() -> pd.Series:
    return make_admissions(1200, seed=31)["mother_key"]


def _all_text(root: Path) -> str:
    """Every text file under ``root`` (reports, CSV, JSON), concatenated."""
    texts = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".csv", ".json", ".md", ".txt", ".yaml"}:
            texts.append(path.read_text(encoding="utf-8"))
    return "".join(texts)


def test_ingest_hashes_mother_key_with_a_created_salt(project: Path) -> None:
    salt_path = project / SALT_FILE
    assert not salt_path.exists()
    outputs = [_invoke(project, c) for c in ["ingest", "robson", "profile"]]
    for result in outputs:
        assert result.exit_code == 0, result.output
    console = "".join(r.output for r in outputs)
    salt = salt_path.read_bytes()
    assert len(salt) == 32
    assert salt.hex() not in console
    assert "time_dropped=0" in console  # synthetic delivery dates carry no time of day

    raw_ids = _raw_ids()
    canonical = pd.read_parquet(project / "data/processed/canonical.parquet")
    keys = canonical["mother_key"]
    assert keys.str.fullmatch(r"MK_[0-9a-f]{32}").all()
    # Rows share a key exactly when they share a raw identifier.
    pairs = pd.DataFrame({"raw": raw_ids.to_numpy(), "key": keys.to_numpy()})
    assert pairs.groupby("raw")["key"].nunique().eq(1).all()
    assert pairs.groupby("key")["raw"].nunique().eq(1).all()
    assert raw_ids.duplicated().any()

    # The raw identifiers appear nowhere downstream: canonical data, reports, console.
    raw_set = set(raw_ids.astype(str))
    for parquet in ("canonical.parquet", "canonical_robson.parquet"):
        frame = pd.read_parquet(project / "data/processed" / parquet)
        for column in frame.columns:
            assert not raw_set & set(frame[column].dropna().astype(str)), column
    written = (
        _all_text(project / "reports")
        + (project / "data/interim/mapping_report.json").read_text(encoding="utf-8")
        + (project / "data/interim/robson_handcheck.csv").read_text(encoding="utf-8")
    )
    for raw_id in raw_set:
        assert raw_id not in written
        assert raw_id not in console


def test_ingest_reuses_the_salt(project: Path) -> None:
    assert _invoke(project, "ingest").exit_code == 0
    salt = (project / SALT_FILE).read_bytes()
    first = pd.read_parquet(project / "data/processed/canonical.parquet")["mother_key"]
    assert _invoke(project, "ingest").exit_code == 0
    assert (project / SALT_FILE).read_bytes() == salt
    second = pd.read_parquet(project / "data/processed/canonical.parquet")["mother_key"]
    assert first.tolist() == second.tolist()


def test_ingest_uses_configured_salt_path(project: Path) -> None:
    config_path = project / "configs/project.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["salt_path"] = "data/keys/custom.salt"
    config_path.write_text(yaml.safe_dump(config))
    given = bytes(range(100, 132))
    (project / "data/keys").mkdir(parents=True)
    (project / "data/keys/custom.salt").write_bytes(given)
    assert _invoke(project, "ingest").exit_code == 0
    assert not (project / SALT_FILE).exists()
    assert (project / "data/keys/custom.salt").read_bytes() == given
    keys = pd.read_parquet(project / "data/processed/canonical.parquet")["mother_key"]
    first_id = str(_raw_ids().iloc[0])
    expected = "MK_" + hashlib.sha256(given + b"\x1f" + first_id.encode()).hexdigest()[:32]
    assert keys.iloc[0] == expected


def test_ingest_refuses_a_short_salt_file(project: Path) -> None:
    (project / "data/interim").mkdir(parents=True)
    (project / SALT_FILE).write_bytes(b"too short")
    result = _run_cli(project, "ingest")
    assert result.returncode == 1
    assert "details suppressed" in result.stderr
    assert (project / SALT_FILE).read_bytes() == b"too short"  # never silently replaced
    assert not (project / "data/processed/canonical.parquet").exists()


def test_ingest_without_hash_key_creates_no_salt(project: Path) -> None:
    mapping_path = project / "configs/mapping_ur_cmhs.yaml"
    mapping = yaml.safe_load(mapping_path.read_text())
    del mapping["fields"]["mother_key"]
    mapping_path.write_text(yaml.safe_dump(mapping))
    assert _invoke(project, "ingest").exit_code == 0
    assert not (project / SALT_FILE).exists()


def test_project_config_salt_path_defaults_for_older_configs(project: Path) -> None:
    config_path = project / "configs/project.yaml"
    assert "salt_path" not in yaml.safe_load(config_path.read_text())
    assert load_project_config(config_path).salt_path == Path(SALT_FILE)


def test_repository_project_config_keeps_the_salt_under_data() -> None:
    cfg = load_project_config(Path(__file__).parents[1] / "configs/project.yaml")
    assert cfg.salt_path == Path(SALT_FILE)


def test_ingest_without_mapping_fields_does_inventory_only(project: Path) -> None:
    (project / "configs/mapping_ur_cmhs.yaml").write_text("source: x\nsheet: ~\nfields: {}\n")
    result = _invoke(project, "ingest")
    assert result.exit_code == 0
    assert "inventory only" in result.output
    assert not (project / "data/processed/canonical.parquet").exists()


def test_console_output_is_aggregate_only(project: Path) -> None:
    outputs = "".join(_invoke(project, c).output for c in ["ingest", "robson", "profile"])
    df = make_admissions(1200, seed=31)
    for column in ["facility_id", "mother_key", "recorded_indication", "mode_of_delivery"]:
        for value in df[column].dropna().astype(str).unique():
            assert value not in outputs, f"row value from {column} printed"
    # no bare small counts: every printed "=<n>" count is 0 or >= 5 or "<5"
    import re

    for match in re.findall(r"=(\d+)\b", outputs):
        assert int(match) == 0 or int(match) >= 5


def _run_cli(
    project: Path, command: str, *, debug: bool = False
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    if debug:
        env["ROBSON_ML_DEBUG"] = "1"
    else:
        env.pop("ROBSON_ML_DEBUG", None)
    return subprocess.run(
        [sys.executable, "-m", "robson_ml.cli", command],
        cwd=project,
        capture_output=True,
        text=True,
        env=env,
    )


def test_failing_command_does_not_leak_row_data(project: Path) -> None:
    """A mapping that points at an absent raw column fails without leaking details."""
    (project / "configs/mapping_ur_cmhs.yaml").write_text(
        yaml.safe_dump(
            {
                "source": "synthetic",
                "sheet": "main",
                "fields": {
                    "admission_id": {"kind": "row_key", "status": "confirmed"},
                    "facility_id": {
                        "raw": "column_not_in_sheet",
                        "kind": "text",
                        "status": "confirmed",
                    },
                },
            }
        )
    )
    result = _run_cli(project, "ingest")
    combined = result.stdout + result.stderr
    assert result.returncode == 1
    assert "details suppressed" in result.stderr
    assert "column_not_in_sheet" not in combined


def test_failing_command_on_corrupt_workbook_does_not_leak_content(project: Path) -> None:
    """A corrupt xlsx raises an openpyxl/zipfile error whose message could embed file content."""
    sentinel = "zz_sentinel_zz"
    (project / "data/raw/raw.xlsx").write_bytes(sentinel.encode("ascii"))
    result = _run_cli(project, "ingest")
    combined = result.stdout + result.stderr
    assert result.returncode == 1
    assert "details suppressed" in result.stderr
    assert sentinel not in combined


def test_debug_env_var_reraises_original_exception(project: Path) -> None:
    (project / "configs/mapping_ur_cmhs.yaml").write_text(
        yaml.safe_dump(
            {
                "source": "synthetic",
                "sheet": "main",
                "fields": {
                    "admission_id": {"kind": "row_key", "status": "confirmed"},
                    "facility_id": {
                        "raw": "column_not_in_sheet",
                        "kind": "text",
                        "status": "confirmed",
                    },
                },
            }
        )
    )
    result = _run_cli(project, "ingest", debug=True)
    assert result.returncode != 0
    assert "MappingError" in result.stdout + result.stderr
    assert "details suppressed" not in result.stderr
