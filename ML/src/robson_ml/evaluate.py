"""The evaluation harness: tune, fit, calibrate and evaluate one configuration.

One configuration = (model, feature set, missing strategy, split scheme, population, seed,
Optuna trials). For every fold of the split scheme:

1. tune by Optuna TPE (seeded; ``n_trials``) on the inner-CV mean log loss over
   ``fold.tuning`` (grid-tuned baselines use their tiny grid; B0/B1 are not tuned; boosting
   early-stops on each inner validation fold);
2. refit on ``fold.fit_idx`` with the best parameters (all preprocessing inside the
   Pipeline, so it is fitted on those rows only);
3. calibrate on ``fold.calib_idx`` only (:func:`robson_ml.calibration.calibrate`);
4. predict ``fold.test_idx``. S2 then orders the held-out facility by delivery date,
   refits the intercept on the first 150 rows (logit(p) as a fixed offset) and evaluates the
   rest; intercept-plus-slope recalibration is reported as a secondary variant.

Metrics per fold, their mean and range, pooled out-of-fold metrics, and pooled
subgroups (facility, onset-free Robson group, nulliparous vs multiparous) are logged
to a local MLflow file store with the plots and provenance tags, including the population
version; :func:`comparison_table` keeps only runs of the current version.
Out-of-fold predictions (row ids, y, p, fold) go to ``<oof_dir>/<run_id>.parquet``, never to
MLflow. Nothing here recommends a mode of delivery: outputs are probabilities of CS under
current practice.
"""

from __future__ import annotations

import itertools
import json
import random
import re
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning

from robson_ml.calibration import CalibratedModel, calibrate
from robson_ml.feature_sets import (
    FACILITY,
    FEATURE_SETS,
    POPULATIONS,
    ROBSON_NO_ONSET,
    FeatureSpec,
    ModelData,
    feature_spec,
    is_deploy_set,
)
from robson_ml.metrics import (
    INSUFFICIENT,
    evaluate,
    fit_logistic,
    log_loss,
    logit,
    net_benefit,
    subgroup_metrics,
)
from robson_ml.models import get_model
from robson_ml.models.base import CLF, PREP, ModelSpec
from robson_ml.plots import calibration_plot, decision_curve_plot, roc_plot
from robson_ml.populations import POPULATION_VERSION
from robson_ml.privacy import SMALL_CELL_THRESHOLD, fmt_count
from robson_ml.splits import (
    Fold,
    internal_nested_folds,
    loho_folds,
    mother_groups,
    recalibration_split,
    temporal_split,
)
from robson_ml.tracking import logged_threshold_metrics

EXPERIMENT_NAME = "robson-readiness"
SPLITS = ("S1", "S2", "S3", "S4")
LOHO_SPLITS = frozenset({"S1", "S2"})
# The only evaluation split a *_deploy (facility) set may run under; S5 is the deployment fit.
DEPLOY_EVAL_SPLITS = frozenset({"S3"})
DEFAULT_N_BOOT = 1000
SUMMARY_METRICS = ("auc", "calibration_slope", "calibration_in_the_large", "brier", "log_loss")
POOLED_EXTRA = ("auc_ci_low", "auc_ci_high", "reliability", "resolution", "accuracy_at_05")
SUBGROUPS = ("facility_id", ROBSON_NO_ONSET, "parity_group")
SLOPE_UPDATE = "slope_update"
CONFIG_KEYS = frozenset(
    {"model", "feature_set", "missing_strategy", "split", "population", "n_trials", "seed"}
)
OPTIONAL_KEYS = frozenset({"n_boot", "name"})


@dataclass(frozen=True)
class ExperimentConfig:
    """One harness run, as declared in ``configs/experiments/*.yaml``."""

    model: str
    feature_set: str
    missing_strategy: str
    split: str
    population: str
    n_trials: int
    seed: int
    n_boot: int = DEFAULT_N_BOOT

    @property
    def name(self) -> str:
        """``model|feature_set|missing_strategy|split|population``."""
        return "|".join(
            (self.model, self.feature_set, self.missing_strategy, self.split, self.population)
        )


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else [value]


def expand_experiments(doc: Mapping[str, Any]) -> list[ExperimentConfig]:
    """Expand one experiment document: any key may hold a list (a grid; cartesian product).

    Combinations whose missing strategy does not apply to the model (M0 without native NaN
    handling) are skipped. Raises ValueError on unknown keys, unknown values, or a
    grid that expands to nothing.
    """
    unknown = set(doc) - CONFIG_KEYS - OPTIONAL_KEYS
    missing = CONFIG_KEYS - set(doc)
    if unknown or missing:
        raise ValueError(
            f"experiment config keys: unknown {sorted(unknown)}, missing {sorted(missing)}"
        )
    keys = sorted(CONFIG_KEYS)
    configs = []
    for values in itertools.product(*(_as_list(doc[k]) for k in keys)):
        item = dict(zip(keys, values, strict=True))
        spec = get_model(str(item["model"]))
        if item["missing_strategy"] not in spec.strategies():
            continue
        config = ExperimentConfig(
            model=str(item["model"]),
            feature_set=str(item["feature_set"]),
            missing_strategy=str(item["missing_strategy"]),
            split=str(item["split"]),
            population=str(item["population"]),
            n_trials=int(item["n_trials"]),
            seed=int(item["seed"]),
            n_boot=int(doc.get("n_boot", DEFAULT_N_BOOT)),
        )
        validate_config(config)
        configs.append(config)
    if not configs:
        raise ValueError("the experiment grid expands to no applicable configuration")
    return configs


def load_experiments(path: Path) -> list[ExperimentConfig]:
    """Read and expand a ``configs/experiments/*.yaml`` file.

    The file may hold several YAML documents (separated by ``---``); each is expanded on its
    own, so a file can pair grids that must not be crossed (e.g. B1 once, main models per strategy).
    """
    docs = [d for d in yaml.safe_load_all(path.read_text(encoding="utf-8")) if d is not None]
    if not docs or not all(isinstance(doc, dict) for doc in docs):
        raise ValueError("an experiment config must be a mapping (or several documents of one)")
    return [config for doc in docs for config in expand_experiments(doc)]


def validate_config(config: ExperimentConfig) -> None:
    """Reject unknown values and ``facility_id`` (a ``*_deploy`` set) under S1/S2/S4."""
    get_model(config.model)
    if config.population not in POPULATIONS:
        raise ValueError(f"unknown population {config.population!r}; expected one of {POPULATIONS}")
    if config.feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {config.feature_set!r}")
    if config.split not in SPLITS:
        raise ValueError(f"unknown split {config.split!r}; expected one of {SPLITS}")
    if is_deploy_set(config.feature_set) and config.split not in DEPLOY_EVAL_SPLITS:
        raise ValueError(
            f"facility_id ({config.feature_set}) is a feature only under S3 (evaluation) or "
            "S5 (deployment fit, robson-ml fit-deploy); never under S1/S2/S4"
        )
    if config.n_trials < 1 or config.n_boot < 1:
        raise ValueError("n_trials and n_boot must be positive")


@dataclass(frozen=True)
class RunContext:
    """Where a run is tracked and the provenance tags it carries."""

    tracking_uri: str
    oof_dir: Path
    data_hash: str
    features_yaml_hash: str
    git_commit: str
    selection_rule_commit: str


@dataclass
class FoldResult:
    """One fold's fitted model, tuned parameters, evaluated rows and metrics."""

    fold: Fold
    model: CalibratedModel
    params: dict[str, Any]
    eval_idx: npt.NDArray[np.int64]
    p: npt.NDArray[np.float64]
    metrics: dict[str, Any]
    p_slope_update: npt.NDArray[np.float64] | None = None
    slope_update_metrics: dict[str, Any] | None = None


@dataclass
class RunResult:
    """Everything one configuration produced; ``summary`` holds the logged scalar metrics."""

    run_id: str
    config: ExperimentConfig
    folds: list[FoldResult]
    summary: dict[str, float]
    report: dict[str, Any] = field(default_factory=dict)


def make_folds(meta: pd.DataFrame, split: str, seed: int) -> list[Fold]:
    """The folds of split scheme ``split`` over the population frame ``meta``."""
    if split in LOHO_SPLITS:
        return loho_folds(meta, seed)
    if split == "S3":
        return internal_nested_folds(meta, seed)
    if split == "S4":
        return [temporal_split(meta, seed)]
    raise ValueError(f"unknown split {split!r}")


def _reserved(config: ExperimentConfig) -> dict[str, Any]:
    return {"missing_strategy": config.missing_strategy, "seed": config.seed}


def _score_candidate(
    spec: ModelSpec,
    params: dict[str, Any],
    fs: FeatureSpec,
    x: pd.DataFrame,
    y: npt.NDArray[np.int64],
    train: npt.NDArray[np.int64],
    val: npt.NDArray[np.int64],
) -> tuple[float, int | None]:
    """Validation log loss of ``params`` fitted on ``train``; early-stopped rounds if any."""
    pipe = spec.build(params, fs)
    if spec.early_stopping_rounds is None:
        pipe.fit(x.iloc[train], y[train])
        p = pipe.predict_proba(x.iloc[val])[:, 1]
        return log_loss(y[val], p), None
    prep, clf = pipe.named_steps[PREP], pipe.named_steps[CLF]
    x_train = prep.fit_transform(x.iloc[train], y[train])
    x_val = prep.transform(x.iloc[val])
    clf.set_params(early_stopping_rounds=spec.early_stopping_rounds)
    clf.fit(x_train, y[train], eval_set=[(x_val, y[val])], verbose=False)
    p = clf.predict_proba(x_val)[:, 1]
    return log_loss(y[val], p), int(clf.best_iteration) + 1


def tune(
    spec: ModelSpec,
    fs: FeatureSpec,
    config: ExperimentConfig,
    x: pd.DataFrame,
    y: npt.NDArray[np.int64],
    tuning: Sequence[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]],
) -> dict[str, Any]:
    """Best hyperparameters by inner-CV mean log loss over ``tuning``.

    TPE with a fixed seed and ``config.n_trials`` trials; a model with a ``grid`` is tuned
    exhaustively over it (``{}``: nothing to tune). For early-stopped models the returned
    ``n_estimators`` is the mean best number of rounds over the inner folds.
    """
    study = tuning_study(spec, fs, config, x, y, tuning)
    return {} if study is None else dict(study.best_trial.user_attrs["params"])


def tuning_study(
    spec: ModelSpec,
    fs: FeatureSpec,
    config: ExperimentConfig,
    x: pd.DataFrame,
    y: npt.NDArray[np.int64],
    tuning: Sequence[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]],
) -> optuna.Study | None:
    """The finished Optuna study behind :func:`tune` (None when there is nothing to tune);
    each trial's value is its inner-CV mean log loss and ``user_attrs["params"]`` its
    hyperparameters."""
    if spec.grid is not None and not spec.grid:
        return None
    reserved = _reserved(config)

    def objective(trial: optuna.Trial) -> float:
        if spec.grid:
            params = {k: trial.suggest_categorical(k, list(v)) for k, v in spec.grid.items()}
        else:
            params = spec.search_space(trial)
        scores, rounds = [], []
        for train, val in tuning:
            score, n_rounds = _score_candidate(spec, {**params, **reserved}, fs, x, y, train, val)
            scores.append(score)
            if n_rounds is not None:
                rounds.append(n_rounds)
        if rounds:
            params = {**params, "n_estimators": round(float(np.mean(rounds)))}
        trial.set_user_attr("params", params)
        return float(np.mean(scores))

    if spec.grid:
        sampler: optuna.samplers.BaseSampler = optuna.samplers.GridSampler(
            {k: list(v) for k, v in spec.grid.items()}, seed=config.seed
        )
        n_trials = int(np.prod([len(v) for v in spec.grid.values()]))
    else:
        sampler = optuna.samplers.TPESampler(seed=config.seed)
        n_trials = config.n_trials
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study


def fold_tuning_study(data: ModelData, config: ExperimentConfig, fold: Fold) -> optuna.Study | None:
    """Re-run the tuning of one fold of ``config`` exactly as :func:`run_experiment` does
    (same inputs, inner folds and seeds) and return its study, to inspect the trials. On a
    run's first fold this reproduces the logged hyperparameters."""
    spec = get_model(config.model)
    fs = feature_spec(data, config.feature_set)
    random.seed(config.seed)
    np.random.seed(config.seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return tuning_study(spec, fs, config, data.x[list(fs.columns)], data.y, fold.tuning)


def _sigmoid(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.asarray(1 / (1 + np.exp(-z)), dtype=np.float64)


def recalibrate(
    p_recal: npt.ArrayLike, y_recal: npt.ArrayLike, p_eval: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """S2 local recalibration: (intercept-only, intercept-plus-slope) updates.

    Intercept only: ``logit q = a + logit(p)`` with ``a`` fitted on the recalibration rows
    (logit(p) as a fixed offset). Secondary: ``logit q = a + b * logit(p)``.
    """
    z_recal, z_eval = logit(p_recal), logit(p_eval)
    (intercept,) = fit_logistic(y_recal, offset=z_recal)
    a, b = fit_logistic(y_recal, z_recal)
    return _sigmoid(z_eval + intercept), _sigmoid(a + b * z_eval)


def _run_fold(
    spec: ModelSpec,
    fs: FeatureSpec,
    config: ExperimentConfig,
    data: ModelData,
    x: pd.DataFrame,
    groups: npt.NDArray[np.str_],
    fold: Fold,
) -> FoldResult:
    y = data.y
    params = {**tune(spec, fs, config, x, y, fold.tuning), **_reserved(config)}
    pipe = spec.build(params, fs).fit(x.iloc[fold.fit_idx], y[fold.fit_idx])
    model = calibrate(pipe, x, y, fold.calib_idx, config.seed, groups=groups)
    if config.split != "S2":
        p = model.predict_proba(x.iloc[fold.test_idx])[:, 1]
        metrics = evaluate(y[fold.test_idx], p, seed=config.seed, n_boot=config.n_boot)
        return FoldResult(fold, model, params, fold.test_idx, p, metrics)
    recal, rest = recalibration_split(fold.test_idx, data.meta)
    p_recal = model.predict_proba(x.iloc[recal])[:, 1]
    p_rest = model.predict_proba(x.iloc[rest])[:, 1]
    p, p_slope = recalibrate(p_recal, y[recal], p_rest)
    metrics = evaluate(y[rest], p, seed=config.seed, n_boot=config.n_boot)
    slope_metrics = evaluate(y[rest], p_slope, seed=config.seed, n_boot=config.n_boot)
    return FoldResult(fold, model, params, rest, p, metrics, p_slope, slope_metrics)


def metric_key(name: str) -> str:
    """``name`` with characters MLflow rejects in metric and param keys replaced by ``_``."""
    return re.sub(r"[^A-Za-z0-9_.\- /]", "_", name)


def _summarise(
    per_fold: Mapping[str, Mapping[str, Any]], pooled: Mapping[str, Any], prefix: str = ""
) -> dict[str, float]:
    """Per-fold, mean, min/max (range) and pooled scalar metrics."""
    out: dict[str, float] = {}
    for metric in SUMMARY_METRICS:
        values = np.array([float(m[metric]) for m in per_fold.values()])
        out[f"{prefix}mean_{metric}"] = float(values.mean())
        out[f"{prefix}min_{metric}"] = float(values.min())
        out[f"{prefix}max_{metric}"] = float(values.max())
        out[f"{prefix}pooled_{metric}"] = float(pooled[metric])
        for name, m in per_fold.items():
            out[metric_key(f"{prefix}{name}_{metric}")] = float(m[metric])
    for metric in POOLED_EXTRA:
        out[f"{prefix}pooled_{metric}"] = float(pooled[metric])
    return out


def _parity_group(parity: pd.Series) -> pd.Series:
    values = pd.to_numeric(parity, errors="coerce").astype(float)
    labels = np.where(values.to_numpy() == 0, "nulliparous", "multiparous")
    return pd.Series(labels, index=parity.index, dtype=object).where(values.notna())


def _subgroups(
    data: ModelData,
    rows: npt.NDArray[np.int64],
    p: npt.NDArray[np.float64],
    config: ExperimentConfig,
) -> dict[str, dict[str, Any]]:
    """Pooled out-of-fold metrics per facility, Robson group and parity.

    The Robson subgroup is the onset-free group, whatever the population.
    """
    frame = pd.DataFrame(
        {
            "facility_id": data.meta[FACILITY].iloc[rows].astype(str).to_numpy(),
            ROBSON_NO_ONSET: data.meta[ROBSON_NO_ONSET].iloc[rows].to_numpy(),
            "parity_group": _parity_group(data.meta["parity"].iloc[rows]).to_numpy(),
        }
    )
    return {
        by: subgroup_metrics(frame, data.y[rows], p, by, seed=config.seed, n_boot=config.n_boot)
        for by in SUBGROUPS
    }


def _subgroup_table(subgroups: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """Subgroup metrics as a table: levels below the minimum counts say ``insufficient``."""
    rows = []
    for by, levels in subgroups.items():
        for level, metrics in levels.items():
            row: dict[str, Any] = {"subgroup": by, "level": level}
            if metrics == INSUFFICIENT:
                row.update({m: INSUFFICIENT for m in SUMMARY_METRICS})
            else:
                row.update({m: metrics[m] for m in SUMMARY_METRICS})
            rows.append(row)
    return pd.DataFrame(rows, columns=["subgroup", "level", *SUMMARY_METRICS])


def _exclusion_report(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Population exclusion (and kept-but-counted) counts, small cells suppressed, share of CS."""
    out = []
    for row in table.to_dict("records"):
        n_cs, n_cs_audit = int(row["n_cs_excluded"]), int(row["n_cs_audit"])
        share: float | str = (
            round(n_cs / n_cs_audit, 4)
            if n_cs_audit and not 0 < n_cs < SMALL_CELL_THRESHOLD
            else fmt_count(n_cs)
        )
        out.append(
            {
                "category": str(row["category"]),
                "n_excluded": fmt_count(int(row["n_excluded"])),
                "n_cs_excluded": fmt_count(n_cs),
                "n_kept": fmt_count(int(row.get("n_kept", 0))),
                "share_of_all_cs": share,
            }
        )
    return out


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer | np.floating):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def run_experiment(config: ExperimentConfig, data: ModelData, ctx: RunContext) -> RunResult:
    """Run one configuration end to end and log it to MLflow; see the module docstring."""
    validate_config(config)
    if data.population != config.population:
        raise ValueError("data was built for a different population than the config")
    spec = get_model(config.model)
    if config.missing_strategy not in spec.strategies():
        raise ValueError(f"{config.model} does not support {config.missing_strategy}")
    fs = feature_spec(data, config.feature_set)
    if config.split not in DEPLOY_EVAL_SPLITS and FACILITY in fs.columns:
        raise AssertionError("facility_id must never be a feature under S1/S2/S4")
    x = data.x[list(fs.columns)]
    random.seed(config.seed)
    np.random.seed(config.seed)  # legacy global seed, fixed per run
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    groups = mother_groups(data.meta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        folds = [
            _run_fold(spec, fs, config, data, x, groups, fold)
            for fold in make_folds(data.meta, config.split, config.seed)
        ]

    rows = np.concatenate([f.eval_idx for f in folds])
    p = np.concatenate([f.p for f in folds])
    pooled = evaluate(data.y[rows], p, seed=config.seed, n_boot=config.n_boot)
    per_fold = {f.fold.name: f.metrics for f in folds}
    summary = _summarise(per_fold, pooled)
    report: dict[str, Any] = {"per_fold": per_fold, "pooled": pooled}
    if config.split == "S2":
        p_slope = np.concatenate([f.p_slope_update for f in folds if f.p_slope_update is not None])
        pooled_slope = evaluate(data.y[rows], p_slope, seed=config.seed, n_boot=config.n_boot)
        slope_folds = {f.fold.name: f.slope_update_metrics or {} for f in folds}
        summary.update(_summarise(slope_folds, pooled_slope, prefix=f"{SLOPE_UPDATE}_"))
        report[SLOPE_UPDATE] = {"per_fold": slope_folds, "pooled": pooled_slope}
    subgroups = _subgroups(data, rows, p, config)
    report["subgroups"] = subgroups
    report["population_exclusions"] = _exclusion_report(data.exclusion_table)
    for f in folds:
        for option, brier in f.model.choice.cv_brier.items():
            summary[metric_key(f"{f.fold.name}_calibration_cv_brier_{option}")] = float(brier)

    oof = pd.DataFrame(
        {
            "admission_id": data.meta["admission_id"].iloc[rows].to_numpy(),
            "fold": np.concatenate([[f.fold.name] * len(f.eval_idx) for f in folds]),
            "y": data.y[rows],
            "p": p,
        }
    )
    if config.split == "S2":
        oof["p_slope_update"] = p_slope
    run_id = _log_run(config, spec, fs, folds, summary, report, subgroups, oof, ctx)
    return RunResult(run_id, config, folds, summary, report)


def _log_run(
    config: ExperimentConfig,
    spec: ModelSpec,
    fs: FeatureSpec,
    folds: list[FoldResult],
    summary: dict[str, float],
    report: dict[str, Any],
    subgroups: dict[str, dict[str, Any]],
    oof: pd.DataFrame,
    ctx: RunContext,
) -> str:
    """Log params, metrics, tags and aggregate artifacts; write OOF outside MLflow."""
    import mlflow

    mlflow.set_tracking_uri(ctx.tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    tags = {
        "data_hash": ctx.data_hash,
        "features_yaml_hash": ctx.features_yaml_hash,
        "git_commit": ctx.git_commit,
        "selection_rule_commit": ctx.selection_rule_commit,
        "split_scheme": config.split,
        "population": config.population,
        "population_version": POPULATION_VERSION,
        "feature_set": config.feature_set,
        "missing_strategy": config.missing_strategy,
        "seed": str(config.seed),
        "model": config.model,
    }
    params: dict[str, Any] = {
        **asdict(config),
        "population_version": POPULATION_VERSION,
        "family": spec.family,
        "complexity_rank": spec.complexity_rank,
        "n_features": len(fs.columns),
    }
    for f in folds:
        params[metric_key(f"{f.fold.name}_calibration")] = f.model.choice.method
        params[metric_key(f"{f.fold.name}_params")] = json.dumps(
            {k: v for k, v in f.params.items() if k not in ("missing_strategy", "seed")},
            sort_keys=True,
            default=_json_default,
        )
    with mlflow.start_run(run_name=config.name, tags=tags) as run:
        run_id = str(run.info.run_id)
        mlflow.log_params(params)
        mlflow.log_metrics(summary)
        mlflow.log_metrics(logged_threshold_metrics(oof["y"].to_numpy(), oof["p"].to_numpy()))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "metrics.json").write_text(
                json.dumps(report, indent=2, default=_json_default), encoding="utf-8"
            )
            (out / "features.json").write_text(json.dumps(list(fs.columns)), encoding="utf-8")
            _subgroup_table(subgroups).to_csv(out / "subgroups.csv", index=False)
            y, p = oof["y"].to_numpy(), oof["p"].to_numpy()
            title = config.name
            calibration_plot(y, p, out / "calibration.png", title)
            roc_plot(y, p, summary["pooled_auc"], out / "roc.png", title)
            decision_curve_plot(net_benefit(y, p), out / "decision_curve.png", title)
            mlflow.log_artifacts(str(out))
    ctx.oof_dir.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(ctx.oof_dir / f"{run_id}.parquet", index=False)
    return run_id


COMPARISON_COLUMNS = (
    "model",
    "feature_set",
    "missing_strategy",
    "split",
    "population",
    "population_version",
    "complexity_rank",
    "mean_auc",
    "min_auc",
    "max_auc",
    "pooled_auc",
    "pooled_auc_ci_low",
    "pooled_auc_ci_high",
    "mean_calibration_slope",
    "pooled_calibration_slope",
    "mean_calibration_in_the_large",
    "pooled_calibration_in_the_large",
    "mean_brier",
    "pooled_brier",
    "mean_log_loss",
    "pooled_log_loss",
    "run_id",
)


def find_completed_run(config: ExperimentConfig, ctx: RunContext) -> str | None:
    """Return the id of a finished run of ``config`` on the same data, registry, population
    version and seed, or None. Runs are logged only after they complete, so a match is a
    finished result; re-running it would only duplicate it."""
    import mlflow

    mlflow.set_tracking_uri(ctx.tracking_uri)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return None
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f"attributes.run_name = '{config.name}' "
            f"and attributes.status = 'FINISHED' "
            f"and tags.data_hash = '{ctx.data_hash}' "
            f"and tags.features_yaml_hash = '{ctx.features_yaml_hash}' "
            f"and tags.population_version = '{POPULATION_VERSION}' "
            f"and tags.seed = '{config.seed}'"
        ),
        output_format="list",
    )
    return str(runs[0].info.run_id) if runs else None


def comparison_table(tracking_uri: str) -> pd.DataFrame:
    """One row per finished harness run of the current population version: metrics only.

    Only runs tagged ``population_version`` equal to :data:`POPULATION_VERSION`
    are compared; runs without the tag predate v1.3 (onset-defined ``P_pred``) and are left
    out, as are runs of any other version. No counts.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        return pd.DataFrame(columns=list(COMPARISON_COLUMNS))
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string="attributes.status = 'FINISHED'",
        output_format="pandas",
    )
    assert isinstance(runs, pd.DataFrame)
    rows = []
    for _, run in runs.iterrows():
        version = run.get("tags.population_version")
        if not isinstance(version, str) or version != POPULATION_VERSION:
            continue
        row: dict[str, Any] = {
            "model": run.get("params.model"),
            "feature_set": run.get("params.feature_set"),
            "missing_strategy": run.get("params.missing_strategy"),
            "split": run.get("params.split"),
            "population": run.get("params.population"),
            "population_version": version,
            "complexity_rank": run.get("params.complexity_rank"),
            "run_id": run["run_id"],
        }
        for column in COMPARISON_COLUMNS:
            if column not in row:
                row[column] = run.get(f"metrics.{column}")
        rows.append(row)
    table = pd.DataFrame(rows, columns=list(COMPARISON_COLUMNS))
    return table.sort_values(["split", "model", "feature_set", "missing_strategy"]).reset_index(
        drop=True
    )
