"""Deployment fit (spec §11.1 S5, §13.1, §13.4): the registered model's single fitted artefact.

:func:`fit_deployment` fits one configuration on the whole prediction population, following
S5 (not an evaluation, so no metric of discrimination is produced here):

1. :func:`robson_ml.splits.deployment_split`: 80/20 fit/calibration (stratified cs x
   facility, grouped by ``mother_key``), tuning folds GroupKFold(4) by facility;
2. Optuna TPE tuning (seeded, ``n_trials``) on the inner-CV mean log loss (spec §11.2);
3. refit on the fit rows with the best hyperparameters (preprocessing inside the Pipeline);
4. calibration on the calibration rows only (:func:`robson_ml.calibration.calibrate`).

It writes ``artefacts/<version_label>/model.joblib`` (one fitted
:class:`~robson_ml.calibration.CalibratedModel`, preprocessing included) and
``fit_summary.json`` (provenance and aggregate facts only: no row, no identifier; counts via
``fmt_count``), and logs an MLflow run tagged ``split_scheme = S5`` in its own experiment so
it never enters the evaluation comparison. The full manifest, feature schema and serving
contract belong to Phase H (spec §17).

:func:`deployment_feature_choice` applies spec §13.4: the deploy variant (base set plus
``facility_id``) is used only if its S3 mean log loss is lower than the base set's.

Outputs are probabilities of CS under current practice; nothing here recommends a mode of
delivery.
"""

from __future__ import annotations

import json
import random
import re
import tempfile
import warnings
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning

from robson_ml.calibration import CalibratedModel, calibrate
from robson_ml.evaluate import ExperimentConfig, tune
from robson_ml.feature_sets import (
    BASE_FEATURE_SETS,
    POPULATIONS,
    ModelData,
    deploy_variant,
    feature_spec,
)
from robson_ml.models import get_model
from robson_ml.populations import P_PRED, POPULATION_VERSION
from robson_ml.privacy import fmt_count
from robson_ml.splits import deployment_split, mother_groups

DEPLOY_EXPERIMENT = "robson-readiness-deployment"
SPLIT_SCHEME = "S5"
MODEL_FILE = "model.joblib"
SUMMARY_FILE = "fit_summary.json"
AUTO = "auto"
DEPLOYMENT_KEYS = frozenset(
    {
        "model",
        "feature_set",
        "use_facility",
        "missing_strategy",
        "population",
        "seed",
        "n_trials",
        "version_label",
    }
)
VERSION_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
LIBRARIES = ("scikit-learn", "numpy", "pandas", "scipy", "joblib", "optuna", "xgboost", "mlflow")
MONTH = "%Y-%m"


@dataclass(frozen=True)
class DeploymentConfig:
    """``configs/deployment.yaml``: the specification to fit under S5.

    ``feature_set`` is a base set (FS0-FS4); ``use_facility`` true fits its ``_deploy``
    variant, false the base set, ``"auto"`` decides by spec §13.4
    (:func:`deployment_feature_choice`).
    """

    model: str
    feature_set: str
    use_facility: bool | Literal["auto"]
    missing_strategy: str
    population: str
    seed: int
    n_trials: int
    version_label: str


def load_deployment_config(path: Path) -> DeploymentConfig:
    """Read and validate a deployment YAML; raises ValueError on any unknown or bad value."""
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError("a deployment config must be a mapping")
    unknown, missing = set(doc) - DEPLOYMENT_KEYS, DEPLOYMENT_KEYS - set(doc)
    if unknown or missing:
        raise ValueError(
            f"deployment config keys: unknown {sorted(unknown)}, missing {sorted(missing)}"
        )
    use_facility = doc["use_facility"]
    if use_facility != AUTO and not isinstance(use_facility, bool):
        raise ValueError("use_facility must be true, false or auto")
    config = DeploymentConfig(
        model=str(doc["model"]),
        feature_set=str(doc["feature_set"]),
        use_facility=use_facility,
        missing_strategy=str(doc["missing_strategy"]),
        population=str(doc["population"]),
        seed=int(doc["seed"]),
        n_trials=int(doc["n_trials"]),
        version_label=str(doc["version_label"]),
    )
    validate_deployment_config(config)
    return config


def validate_deployment_config(config: DeploymentConfig) -> None:
    """Reject unknown models, strategies, sets, populations and unsafe version labels."""
    spec = get_model(config.model)
    if config.missing_strategy not in spec.strategies():
        raise ValueError(f"{config.model} does not support {config.missing_strategy}")
    if config.feature_set not in BASE_FEATURE_SETS:
        raise ValueError(
            f"feature_set must be a base set {list(BASE_FEATURE_SETS)}; facility is added by "
            "use_facility, not by naming a _deploy set"
        )
    if config.population not in POPULATIONS:
        raise ValueError(f"unknown population {config.population!r}")
    if config.n_trials < 1:
        raise ValueError("n_trials must be positive")
    if not VERSION_LABEL.match(config.version_label):
        raise ValueError("version_label may hold only letters, digits, '.', '_' and '-'")


def deployment_feature_choice(
    tracking_uri: str,
    model: str = "logreg_l2",
    feature_set: str = "FS2",
    missing_strategy: str = "M2",
    population: str = P_PRED,
    data_hash: str | None = None,
) -> dict[str, Any]:
    """Spec §13.4: use facility only if the deploy variant improves S3 internal log loss.

    Reads the finished S3 runs (current population version, latest run per configuration;
    optionally only runs on ``data_hash``) of ``model``/``missing_strategy``/``population``
    on ``feature_set`` and on its ``_deploy`` variant. Returns both mean log losses, their
    run ids and ``use_facility`` (deploy mean log loss strictly lower). Raises LookupError
    when either run is missing.
    """
    from robson_ml.select import load_runs

    deploy = deploy_variant(feature_set)
    runs = load_runs(tracking_uri)
    scoped = runs[
        (runs["split"] == "S3")
        & (runs["model"] == model)
        & (runs["missing_strategy"] == missing_strategy)
        & (runs["population"] == population)
    ]
    if data_hash is not None:
        scoped = scoped[scoped["data_hash"] == data_hash]
    found: dict[str, pd.Series] = {}
    for name in (feature_set, deploy):
        rows = scoped[scoped["feature_set"] == name]
        if rows.empty or "mean_log_loss" not in rows.columns:
            raise LookupError(
                f"no finished S3 run of {model} {name} {missing_strategy} on {population} "
                f"(population version {POPULATION_VERSION}); run configs/experiments/"
                "phase_f.yaml first"
            )
        found[name] = rows.iloc[0]
    base_loss = float(found[feature_set]["mean_log_loss"])
    deploy_loss = float(found[deploy]["mean_log_loss"])
    use = deploy_loss < base_loss
    return {
        "rule": "spec §13.4: facility used only if the deploy variant's S3 mean log loss is lower",
        "model": model,
        "missing_strategy": missing_strategy,
        "population": population,
        "population_version": POPULATION_VERSION,
        "feature_set": feature_set,
        "deploy_feature_set": deploy,
        "s3_mean_log_loss": base_loss,
        "s3_mean_log_loss_deploy": deploy_loss,
        "run_id": str(found[feature_set]["run_id"]),
        "run_id_deploy": str(found[deploy]["run_id"]),
        "use_facility": bool(use),
        "chosen_feature_set": deploy if use else feature_set,
    }


@dataclass(frozen=True)
class Provenance:
    """Where the deployment run is tracked and the provenance it records (spec §17.3, §19)."""

    tracking_uri: str
    data_hash: str
    features_yaml_hash: str
    git_commit: str
    selection_rule_commit: str


@dataclass
class DeploymentResult:
    """The fitted model, its summary (what ``fit_summary.json`` holds) and where it went."""

    model: CalibratedModel
    summary: dict[str, Any]
    out_dir: Path
    run_id: str


def library_versions(names: tuple[str, ...] = LIBRARIES) -> dict[str, str]:
    """Installed versions of ``names`` (absent packages are left out)."""
    out = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer | np.floating):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _month(value: Any) -> str | None:
    return None if pd.isna(value) else pd.Timestamp(value).strftime(MONTH)


def fit_deployment(
    config: DeploymentConfig,
    data: ModelData,
    provenance: Provenance,
    artefacts_dir: Path,
    use_facility: bool,
    facility_decision: dict[str, Any] | None = None,
) -> DeploymentResult:
    """Fit, calibrate and save the deployment model; see the module docstring.

    ``use_facility`` is the resolved choice (``True`` fits the ``_deploy`` variant);
    ``facility_decision`` (from :func:`deployment_feature_choice`) is recorded when given.
    Refuses to overwrite an existing ``artefacts/<version_label>/``: a version label names
    one artefact for good.
    """
    from robson_ml.explain import facility_contribution

    validate_deployment_config(config)
    if data.population != config.population:
        raise ValueError("data was built for a different population than the config")
    out_dir = artefacts_dir / config.version_label
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} already holds an artefact; choose a new version_label")
    fs_name = deploy_variant(config.feature_set) if use_facility else config.feature_set
    spec = get_model(config.model)
    fs = feature_spec(data, fs_name)
    x = data.x[list(fs.columns)]
    y = data.y
    random.seed(config.seed)
    np.random.seed(config.seed)  # legacy global seed, fixed per run (spec §19)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # tune() reads only the seed, trial budget and missing strategy of the config.
    tuning_config = ExperimentConfig(
        config.model,
        fs_name,
        config.missing_strategy,
        SPLIT_SCHEME,
        config.population,
        config.n_trials,
        config.seed,
    )
    reserved = {"missing_strategy": config.missing_strategy, "seed": config.seed}
    fold = deployment_split(data.meta, config.seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        tuned = tune(spec, fs, tuning_config, x, y, fold.tuning)
        pipe = spec.build({**tuned, **reserved}, fs).fit(x.iloc[fold.fit_idx], y[fold.fit_idx])
    model = calibrate(pipe, x, y, fold.calib_idx, config.seed, groups=mother_groups(data.meta))

    used = np.concatenate([fold.fit_idx, fold.calib_idx])
    dates = pd.to_datetime(data.meta["delivery_date"].iloc[used], errors="coerce")
    summary: dict[str, Any] = {
        "version_label": config.version_label,
        "split_scheme": SPLIT_SCHEME,
        "config": asdict(config),
        "algorithm": config.model,
        "family": spec.family,
        "feature_set": fs_name,
        "use_facility": bool(use_facility),
        "facility_decision": facility_decision,
        "features": list(fs.columns),
        "hyperparameters": tuned,
        "calibration": {
            "method": model.choice.method,
            "cv_brier": dict(model.choice.cv_brier),
        },
        "n_fit_rows": fmt_count(len(fold.fit_idx)),
        "n_calibration_rows": fmt_count(len(fold.calib_idx)),
        "n_tuning_folds": len(fold.tuning),
        "population": config.population,
        "population_version": POPULATION_VERSION,
        "training_window_start": _month(dates.min()),
        "training_window_end": _month(dates.max()),
        "seed": config.seed,
        "git_commit": provenance.git_commit,
        "selection_rule_commit": provenance.selection_rule_commit,
        "data_hash": provenance.data_hash,
        "features_yaml_hash": provenance.features_yaml_hash,
        "library_versions": library_versions(),
    }
    if use_facility:
        table = facility_contribution(model)
        summary["facility_contribution"] = table.to_dict("records")

    run_id = _log_deployment_run(config, fs_name, summary, provenance)
    summary["mlflow_run_id"] = run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / MODEL_FILE)
    (out_dir / SUMMARY_FILE).write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )
    return DeploymentResult(model, summary, out_dir, run_id)


def _log_deployment_run(
    config: DeploymentConfig, fs_name: str, summary: dict[str, Any], provenance: Provenance
) -> str:
    """Log the S5 fit (params, calibration Brier, fit summary) to its own MLflow experiment."""
    import mlflow

    mlflow.set_tracking_uri(provenance.tracking_uri)
    mlflow.set_experiment(DEPLOY_EXPERIMENT)
    tags = {
        "data_hash": provenance.data_hash,
        "features_yaml_hash": provenance.features_yaml_hash,
        "git_commit": provenance.git_commit,
        "selection_rule_commit": provenance.selection_rule_commit,
        "split_scheme": SPLIT_SCHEME,
        "population": config.population,
        "population_version": POPULATION_VERSION,
        "feature_set": fs_name,
        "missing_strategy": config.missing_strategy,
        "seed": str(config.seed),
        "model": config.model,
        "version_label": config.version_label,
        "use_facility": str(summary["use_facility"]).lower(),
    }
    params = {
        "model": config.model,
        "feature_set": fs_name,
        "missing_strategy": config.missing_strategy,
        "population": config.population,
        "n_trials": config.n_trials,
        "seed": config.seed,
        "calibration": summary["calibration"]["method"],
        "hyperparameters": json.dumps(
            summary["hyperparameters"], sort_keys=True, default=_json_default
        ),
    }
    metrics = {
        f"calibration_cv_brier_{option}": float(brier)
        for option, brier in summary["calibration"]["cv_brier"].items()
    }
    with mlflow.start_run(run_name=f"deploy|{config.version_label}", tags=tags) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / SUMMARY_FILE
            path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
            mlflow.log_artifact(str(path))
        return str(run.info.run_id)
