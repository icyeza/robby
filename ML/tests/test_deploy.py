"""Deploy variants, the S5 deployment fit, the facility decision, local explanations and
the facility contribution. Synthetic only."""

import json
import re
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from robson_ml.calibration import CalibratedModel
from robson_ml.evaluate import (
    EXPERIMENT_NAME,
    ExperimentConfig,
    expand_experiments,
    load_experiments,
    validate_config,
)
from robson_ml.explain import facility_contribution, linear_contributions
from robson_ml.feature_sets import (
    FACILITY,
    FEATURE_SETS,
    build_model_data,
    deploy_variant,
    is_deploy_set,
)
from robson_ml.features import build_raw_features, load_feature_registry
from robson_ml.ingest import file_sha256
from robson_ml.populations import POPULATION_VERSION
from robson_ml.privacy import SUPPRESSED
from tests.test_cli_run import _invoke, project  # noqa: F401  (fixture)
from tests.test_preregistration import commit_rule

EXPERIMENTS = Path("configs/experiments")
REPO_DEPLOYMENT = Path("configs/deployment.yaml")
BASE = ExperimentConfig("logreg_l2", "FS2_deploy", "M2", "S3", "P_pred", 2, 1, 20)


def test_deploy_variants_validated() -> None:
    assert {n for n in FEATURE_SETS if is_deploy_set(n)} == {f"FS{k}_deploy" for k in range(5)}
    assert deploy_variant("FS2") == "FS2_deploy"
    with pytest.raises(ValueError):
        deploy_variant("FS2_deploy")
    for name in (n for n in FEATURE_SETS if is_deploy_set(n)):
        validate_config(replace(BASE, feature_set=name))  # S3: allowed
        for split in ("S1", "S2", "S4"):
            with pytest.raises(ValueError, match="facility"):
                validate_config(replace(BASE, feature_set=name, split=split))
    doc = {
        "model": "logreg_l2",
        "feature_set": ["FS2", "FS2_deploy"],
        "missing_strategy": "M2",
        "split": "S4",
        "population": "P_pred",
        "n_trials": 2,
        "seed": 1,
    }
    with pytest.raises(ValueError, match="facility"):
        expand_experiments(doc)


def test_deployment_check_config() -> None:
    configs = load_experiments(EXPERIMENTS / "deployment_check.yaml")
    assert {c.name for c in configs} == {
        "logreg_l2|FS2|M2|S3|P_pred",
        "logreg_l2|FS2_deploy|M2|S3|P_pred",
        "logreg_l2|FS2|M2|S4|P_pred",
        "B1|FS0|M0|S4|P_pred",
    }
    assert {(c.n_trials, c.seed) for c in configs} == {(50, 20260923)}
    sensitivity = load_experiments(EXPERIMENTS / "sensitivity_onset_coded.yaml")
    assert {c.population for c in sensitivity} == {"P_pred_onset_coded"}


def test_repository_deployment_config() -> None:
    from robson_ml.deploy import load_deployment_config

    config = load_deployment_config(REPO_DEPLOYMENT)
    assert (config.model, config.feature_set, config.missing_strategy) == (
        "logreg_l2",
        "FS2",
        "M2",
    )
    assert config.use_facility == "auto"
    assert config.population == "P_pred"


def _log_s3_run(uri: str, feature_set: str, log_loss: float, data_hash: str = "h") -> str:
    import mlflow

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    tags = {"population_version": POPULATION_VERSION, "data_hash": data_hash}
    name = f"logreg_l2|{feature_set}|M2|S3|P_pred"
    with mlflow.start_run(run_name=name, tags=tags) as run:
        mlflow.log_params(
            {
                "model": "logreg_l2",
                "feature_set": feature_set,
                "missing_strategy": "M2",
                "split": "S3",
                "population": "P_pred",
                "family": "linear",
                "complexity_rank": 1,
                "n_trials": 2,
                "seed": 1,
                "n_boot": 20,
                "n_features": 3,
            }
        )
        mlflow.log_metrics({"mean_log_loss": log_loss, "mean_auc": 0.7})
        return str(run.info.run_id)


def test_deployment_feature_choice(tmp_path: Path) -> None:
    from robson_ml.deploy import deployment_feature_choice

    uri = (tmp_path / "mlruns").resolve().as_uri()
    _log_s3_run(uri, "FS2", 0.50)
    with pytest.raises(LookupError):
        deployment_feature_choice(uri)
    deploy_id = _log_s3_run(uri, "FS2_deploy", 0.48)
    choice = deployment_feature_choice(uri)
    assert choice["use_facility"] is True
    assert choice["chosen_feature_set"] == "FS2_deploy"
    assert choice["run_id_deploy"] == deploy_id
    assert (choice["s3_mean_log_loss"], choice["s3_mean_log_loss_deploy"]) == (0.50, 0.48)
    _log_s3_run(uri, "FS2_deploy", 0.50)  # latest run: a tie is not an improvement
    assert deployment_feature_choice(uri)["use_facility"] is False
    _log_s3_run(uri, "FS2_deploy", 0.52)
    assert deployment_feature_choice(uri)["chosen_feature_set"] == "FS2"
    with pytest.raises(LookupError):
        deployment_feature_choice(uri, data_hash="other")


def _write_deployment(root: Path, label: str, use_facility: object) -> Path:
    path = root / "configs" / f"{label}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": "logreg_l2",
                "feature_set": "FS2",
                "use_facility": use_facility,
                "missing_strategy": "M2",
                "population": "P_pred",
                "seed": 5,
                "n_trials": 2,
                "version_label": label,
            }
        ),
        encoding="utf-8",
    )
    return path.relative_to(root)


def test_fit_deploy_refuses_without_committed_rule(project: Path) -> None:  # noqa: F811
    config = _write_deployment(project, "v-test", False)
    result = _invoke(project, "fit-deploy", config.as_posix())
    assert result.exit_code == 1  # type: ignore[attr-defined]
    assert "PreregistrationError" in result.output  # type: ignore[attr-defined]
    assert not (project / "artefacts").exists()
    assert not (project / "mlruns").exists()


def test_fit_deploy_end_to_end(project: Path) -> None:  # noqa: F811
    rule_commit = commit_rule(project)
    config = _write_deployment(project, "readiness-test", True)
    result = _invoke(project, "fit-deploy", config.as_posix())
    assert result.exit_code == 0, result.output  # type: ignore[attr-defined]
    output = result.output  # type: ignore[attr-defined]
    assert "SYN" not in output
    out = project / "artefacts" / "readiness-test"
    model = joblib.load(out / "model.joblib")
    assert isinstance(model, CalibratedModel)
    summary_text = (out / "fit_summary.json").read_text(encoding="utf-8")
    summary = json.loads(summary_text)

    # Provenance and aggregate facts only.
    assert summary["version_label"] == "readiness-test"
    assert summary["split_scheme"] == "S5"
    assert summary["feature_set"] == "FS2_deploy" and summary["use_facility"] is True
    assert summary["selection_rule_commit"] == rule_commit
    assert summary["population_version"] == POPULATION_VERSION
    assert summary["calibration"]["method"] in {"none", "platt", "isotonic"}
    assert set(summary["calibration"]["cv_brier"]) == {"none", "platt", "isotonic"}
    assert set(summary["hyperparameters"]) == {"C"}
    for key in ("n_fit_rows", "n_calibration_rows"):
        assert isinstance(summary[key], str)
        assert summary[key] == SUPPRESSED or int(summary[key]) > 0
    month = re.compile(r"^\d{4}-\d{2}$")
    assert month.match(summary["training_window_start"])
    assert month.match(summary["training_window_end"])
    assert "scikit-learn" in summary["library_versions"]
    assert summary["data_hash"] == file_sha256(
        project / "data" / "processed" / "canonical_robson.parquet"
    )
    assert FACILITY in summary["features"]
    canonical = pd.read_parquet(project / "data" / "processed" / "canonical_robson.parquet")
    for column in ("admission_id", "mother_key"):
        values = canonical[column].dropna().astype(str)
        assert not any(v in summary_text for v in values)
    assert "SYN" not in summary_text

    # The artefact scores canonical admissions (its own feature columns) in [0, 1].
    registry = load_feature_registry(project / "configs" / "features_v1.yaml")
    raw = pd.read_excel(project / "data" / "raw" / "raw.xlsx", sheet_name="main")
    raw_features, _ = build_raw_features(raw, registry)
    data = build_model_data(canonical, registry, raw_features, "P_pred")
    x = data.x[summary["features"]]
    p = model.predict_proba(x)[:, 1]
    assert p.shape == (len(x),) and np.all((p >= 0) & (p <= 1))

    # Linear contributions reproduce the raw logit; facility offsets are per level.
    contributions, intercept = linear_contributions(model, x.head(50))
    raw_logit = model.estimator.decision_function(x.head(50))
    np.testing.assert_allclose(intercept + contributions.sum(axis=1).to_numpy(), raw_logit)
    table = facility_contribution(model)
    assert set(table["facility_id"]) == set(data.meta[FACILITY].astype(str))
    assert summary["facility_contribution"] == table.to_dict("records")
    report = pd.read_csv(project / "reports" / "interpretation" / "facility_contribution.csv")
    assert list(report.columns) == ["facility_id", "coefficient", "centred"]

    # A version label names one artefact for good.
    again = _invoke(project, "fit-deploy", config.as_posix())
    assert again.exit_code == 1  # type: ignore[attr-defined]


def test_fit_deploy_auto_uses_s3_rule(project: Path) -> None:  # noqa: F811
    commit_rule(project)
    uri = (project / "mlruns").resolve().as_uri()
    data_hash = file_sha256(project / "data" / "processed" / "canonical_robson.parquet")
    _log_s3_run(uri, "FS2", 0.50, data_hash)
    _log_s3_run(uri, "FS2_deploy", 0.55, data_hash)
    config = _write_deployment(project, "readiness-auto", "auto")
    result = _invoke(project, "fit-deploy", config.as_posix())
    assert result.exit_code == 0, result.output  # type: ignore[attr-defined]
    assert "use_facility=False" in result.output  # type: ignore[attr-defined]
    summary = json.loads(
        (project / "artefacts" / "readiness-auto" / "fit_summary.json").read_text(encoding="utf-8")
    )
    assert summary["feature_set"] == "FS2" and summary["use_facility"] is False
    assert summary["facility_decision"]["s3_mean_log_loss_deploy"] == 0.55
    assert FACILITY not in summary["features"]
    assert not (project / "reports" / "interpretation" / "facility_contribution.csv").exists()


def test_explain_local_writes_only_interim(project: Path) -> None:  # noqa: F811
    import mlflow

    commit_rule(project)
    experiment = project / "configs" / "experiments" / "lr.yaml"
    experiment.write_text(
        yaml.safe_dump(
            {
                "model": "logreg_l2",
                "feature_set": "FS1",
                "missing_strategy": "M1",
                "split": "S1",
                "population": "P_pred",
                "n_trials": 2,
                "seed": 5,
                "n_boot": 20,
            }
        ),
        encoding="utf-8",
    )
    ran = _invoke(project, "run", "configs/experiments/lr.yaml")
    assert ran.exit_code == 0, ran.output  # type: ignore[attr-defined]
    mlflow.set_tracking_uri((project / "mlruns").resolve().as_uri())
    run_id = str(mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])["run_id"].iloc[0])

    result = _invoke(project, "explain-local", run_id)
    assert result.exit_code == 0, result.output  # type: ignore[attr-defined]
    output = result.output  # type: ignore[attr-defined]
    path = project / "data" / "interim" / "explanations" / f"{run_id}_local.parquet"
    table = pd.read_parquet(path)
    assert len(table) == 20
    assert table["abs_error"].is_monotonic_decreasing
    np.testing.assert_allclose(table["abs_error"], (table["y"] - table["p"]).abs())
    contribution = [c for c in table.columns if c.startswith("contribution__")]
    np.testing.assert_allclose(
        table["intercept"] + table[contribution].sum(axis=1), table["raw_logit"]
    )
    assert "cases=20" in output
    for value in table["admission_id"].astype(str):
        assert value not in output
    for value in table["p"]:
        assert f"{value:.3f}" not in output and f"{value:.2f}" not in output
    reports = project / "reports"
    assert not reports.exists() or not any(
        "local" in p.name or p.suffix == ".parquet" for p in reports.rglob("*")
    )
