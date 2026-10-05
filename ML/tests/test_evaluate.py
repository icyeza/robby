from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robson_ml.evaluate import (
    COMPARISON_COLUMNS,
    ExperimentConfig,
    RunContext,
    RunResult,
    comparison_table,
    expand_experiments,
    load_experiments,
    recalibrate,
    run_experiment,
    validate_config,
)
from robson_ml.feature_sets import FACILITY, ModelData, allowed_columns, feature_spec
from robson_ml.features import load_feature_registry
from robson_ml.populations import POPULATION_VERSION
from robson_ml.splits import RECALIBRATION_N_FIRST
from tests.test_feature_sets import model_data

REGISTRY = Path("configs/features_v1.yaml")
EXPERIMENTS = Path("configs/experiments")
BASE = ExperimentConfig(
    model="logreg_l2",
    feature_set="FS4",
    missing_strategy="M1",
    split="S1",
    population="P_pred",
    n_trials=2,
    seed=11,
    n_boot=20,
)


@pytest.fixture(scope="module")
def data() -> ModelData:
    return model_data(load_feature_registry(REGISTRY), n=1500, seed=21)


@pytest.fixture(scope="module")
def ctx(tmp_path_factory: pytest.TempPathFactory) -> RunContext:
    root = tmp_path_factory.mktemp("harness")
    return RunContext(
        tracking_uri=(root / "mlruns").as_uri(),
        oof_dir=root / "oof",
        data_hash="synthetic",
        features_yaml_hash="synthetic",
        git_commit="synthetic",
        selection_rule_commit="synthetic",
    )


@pytest.fixture(scope="module")
def runs(data: ModelData, ctx: RunContext) -> dict[str, RunResult]:
    return {
        "S1": run_experiment(BASE, data, ctx),
        "S2": run_experiment(replace(BASE, split="S2"), data, ctx),
        "S3_M2": run_experiment(replace(BASE, split="S3", missing_strategy="M2"), data, ctx),
    }


def _models(result: RunResult) -> list[object]:
    return [f.model.estimator for f in result.folds]


def test_no_facility_in_loho(runs: dict[str, RunResult], data: ModelData) -> None:
    """Spec §22: facility_id is absent from the features of every S1/S2 run."""
    for split in ("S1", "S2"):
        for pipe in _models(runs[split]):
            assert FACILITY not in list(pipe.feature_names_in_)  # type: ignore[attr-defined]
    for split in ("S1", "S2"):
        with pytest.raises(ValueError, match="facility"):
            validate_config(replace(BASE, split=split, feature_set="FS4_deploy"))
    validate_config(replace(BASE, split="S3", feature_set="FS4_deploy"))


def test_pipeline_fit_in_fold(runs: dict[str, RunResult], data: ModelData) -> None:
    """Spec §22: imputers, scalers and encoders hold statistics of the fit rows only."""
    fs = feature_spec(data, "FS4")
    numeric, categorical = list(fs.numeric), list(fs.categorical)
    for fold in runs["S1"].folds:
        fit = data.x.iloc[fold.fold.fit_idx]
        prep = fold.model.estimator.named_steps["prep"]
        num = prep.named_transformers_["num"]
        medians = fit[numeric].median().to_numpy()
        np.testing.assert_allclose(num.named_steps["impute"].statistics_, medians)
        imputed = fit[numeric].fillna(fit[numeric].median())
        np.testing.assert_allclose(num.named_steps["scale"].mean_, imputed.mean().to_numpy())
        cat = prep.named_transformers_["cat"]
        modes = [fit[c].mode().sort_values().iloc[0] for c in categorical]
        assert list(cat.named_steps["impute"].statistics_) == modes
        filled = fit[categorical].fillna(dict(zip(categorical, modes, strict=True)))
        for levels, column in zip(cat.named_steps["encode"].categories_, categorical, strict=True):
            assert list(levels) == sorted(filled[column].unique())
        indicator = prep.named_transformers_["missing"]
        columns = [*numeric, *categorical]
        with_missing = [i for i, c in enumerate(columns) if fit[c].isna().any()]
        assert list(indicator.features_) == with_missing
    for fold in runs["S3_M2"].folds:
        fit = data.x.iloc[fold.fold.fit_idx]
        imputer = fold.model.estimator.named_steps["prep"].named_transformers_["num"]
        initial = imputer.named_steps["impute"].initial_imputer_.statistics_
        np.testing.assert_allclose(initial, fit[numeric].mean().to_numpy())


def test_excluded_features_never_reach_a_model(runs: dict[str, RunResult], data: ModelData) -> None:
    """Spec §22: only include features (never facility under S1/S2) reach a fitted model."""
    allowed = allowed_columns(load_feature_registry(REGISTRY))
    for result in runs.values():
        for pipe in _models(result):
            assert set(pipe.feature_names_in_) <= allowed  # type: ignore[attr-defined]


def test_calibrator_sees_only_calibration_rows(runs: dict[str, RunResult]) -> None:
    for fold in runs["S1"].folds:
        f = fold.fold
        assert not np.intersect1d(f.calib_idx, f.fit_idx).size
        assert not np.intersect1d(f.calib_idx, f.test_idx).size


def test_s2_recalibrates_on_first_rows(runs: dict[str, RunResult], data: ModelData) -> None:
    result = runs["S2"]
    assert "slope_update_pooled_auc" in result.summary
    for fold in result.folds:
        assert len(fold.eval_idx) <= len(fold.fold.test_idx) - RECALIBRATION_N_FIRST
        assert np.isin(fold.eval_idx, fold.fold.test_idx).all()


def test_recalibrate_intercept_matches_mean_outcome() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.6, size=400)
    y = (rng.random(400) < np.clip(p + 0.2, 0, 1)).astype(float)
    intercept_only, slope = recalibrate(p, y, p)
    assert intercept_only.mean() == pytest.approx(y.mean(), abs=1e-6)
    assert slope.mean() == pytest.approx(y.mean(), abs=1e-6)


def test_run_outputs(runs: dict[str, RunResult], ctx: RunContext) -> None:
    import mlflow

    result = runs["S1"]
    oof = pd.read_parquet(ctx.oof_dir / f"{result.run_id}.parquet")
    assert list(oof.columns) == ["admission_id", "fold", "y", "p"]
    assert oof["fold"].nunique() == 4
    for key in ("mean_auc", "min_auc", "max_auc", "pooled_auc", "pooled_calibration_slope"):
        assert key in result.summary
    assert set(result.report["subgroups"]) == {
        "facility_id",
        "robson_group_no_onset",
        "parity_group",
    }
    mlflow.set_tracking_uri(ctx.tracking_uri)
    run = mlflow.get_run(result.run_id)
    for tag in (
        "data_hash",
        "features_yaml_hash",
        "git_commit",
        "split_scheme",
        "population",
        "feature_set",
        "missing_strategy",
        "seed",
        "selection_rule_commit",
        "population_version",
    ):
        assert tag in run.data.tags
    assert run.data.tags["population_version"] == POPULATION_VERSION == "v1.3"
    assert run.data.params["population_version"] == POPULATION_VERSION
    artifacts = {a.path for a in mlflow.MlflowClient().list_artifacts(result.run_id)}
    assert artifacts == {
        "calibration.png",
        "decision_curve.png",
        "features.json",
        "metrics.json",
        "roc.png",
        "subgroups.csv",
    }
    local = Path(mlflow.artifacts.download_artifacts(run_id=result.run_id))
    for name in ("metrics.json", "subgroups.csv", "features.json"):
        assert "SYN0" not in (local / name).read_text(encoding="utf-8")


def test_determinism(data: ModelData, ctx: RunContext) -> None:
    """Spec §22: the same config and seed produce identical metrics."""
    config = replace(BASE, model="xgboost", feature_set="FS1", missing_strategy="M0")
    first = run_experiment(config, data, ctx)
    second = run_experiment(config, data, ctx)
    assert first.summary == second.summary
    a = pd.read_parquet(ctx.oof_dir / f"{first.run_id}.parquet")
    b = pd.read_parquet(ctx.oof_dir / f"{second.run_id}.parquet")
    pd.testing.assert_frame_equal(a, b)


def test_baselines_run(data: ModelData, ctx: RunContext) -> None:
    for model, fs, strategy in (("B0", "FS0", "M0"), ("B1", "FS0", "M0"), ("B2", "FS0", "M1")):
        config = replace(BASE, model=model, feature_set=fs, missing_strategy=strategy)
        result = run_experiment(config, data, ctx)
        assert np.isfinite(result.summary["pooled_brier"])
    b1 = run_experiment(
        replace(BASE, model="B1", feature_set="FS0", missing_strategy="M0"), data, ctx
    )
    assert b1.summary["pooled_auc"] > 0.6


def test_compare_has_no_counts(runs: dict[str, RunResult], ctx: RunContext, tmp_path: Path) -> None:
    table = comparison_table(ctx.tracking_uri)
    assert tuple(table.columns) == COMPARISON_COLUMNS
    assert {r.run_id for r in runs.values()} <= set(table["run_id"])
    assert not [c for c in table.columns if c == "n" or c.startswith("n_") or "count" in c]


def test_compare_keeps_only_current_population_version(
    runs: dict[str, RunResult], tmp_path: Path
) -> None:
    """Spec v1.3: runs without the tag (pre-v1.3) or with another version are left out."""
    import mlflow

    from robson_ml.evaluate import EXPERIMENT_NAME

    uri = (tmp_path / "mlruns").as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    stale = {}
    for label, tags in (("untagged", {}), ("v1.2", {"population_version": "v1.2"})):
        with mlflow.start_run(run_name=label, tags=tags) as run:
            mlflow.log_params({"model": "B1", "split": "S1", "population": "P_pred"})
            mlflow.log_metrics({"mean_auc": 0.9})
            stale[label] = run.info.run_id
    with mlflow.start_run(tags={"population_version": POPULATION_VERSION}) as run:
        mlflow.log_params({"model": "B1", "split": "S1", "population": "P_pred"})
        mlflow.log_metrics({"mean_auc": 0.7})
        current = run.info.run_id
    table = comparison_table(uri)
    assert list(table["run_id"]) == [current]
    assert list(table["population_version"]) == [POPULATION_VERSION]


def test_onset_coded_sensitivity_run(ctx: RunContext) -> None:
    """P_pred_onset_coded runs end to end with the legacy onset features, tagged v1.3."""
    import mlflow

    legacy = model_data(
        load_feature_registry(REGISTRY), n=1200, seed=21, population="P_pred_onset_coded"
    )
    config = replace(BASE, model="B1", missing_strategy="M0", population="P_pred_onset_coded")
    result = run_experiment(config, legacy, ctx)
    assert np.isfinite(result.summary["pooled_brier"])
    mlflow.set_tracking_uri(ctx.tracking_uri)
    tags = mlflow.get_run(result.run_id).data.tags
    assert tags["population"] == "P_pred_onset_coded"
    assert tags["population_version"] == POPULATION_VERSION
    with pytest.raises(ValueError, match="population"):
        validate_config(replace(BASE, population="P_pred_sens"))


def test_expand_experiments() -> None:
    doc = {
        "model": ["logreg_l2", "xgboost"],
        "feature_set": ["FS0", "FS1"],
        "missing_strategy": ["M0", "M1"],
        "split": "S1",
        "population": "P_pred",
        "n_trials": 50,
        "seed": 1,
    }
    configs = expand_experiments(doc)
    assert len(configs) == 6  # M0 is skipped for logreg_l2
    assert all(not (c.model == "logreg_l2" and c.missing_strategy == "M0") for c in configs)
    with pytest.raises(ValueError, match="facility"):
        expand_experiments({**doc, "feature_set": "FS4_deploy"})
    with pytest.raises(ValueError, match="unknown"):
        expand_experiments({**doc, "extra": 1})


def test_committed_experiment_configs() -> None:
    configs = [c for path in sorted(EXPERIMENTS.glob("*.yaml")) for c in load_experiments(path)]
    assert len({c.name for c in configs}) == len(configs)
    assert all(c.split == "S3" for c in configs if c.feature_set.endswith("_deploy"))
    sensitivity = load_experiments(EXPERIMENTS / "sensitivity_onset_coded.yaml")
    assert {c.population for c in sensitivity} == {"P_pred_onset_coded"}
    assert {(c.feature_set, c.split) for c in sensitivity} == {("FS4", "S1")}
    assert {c.model for c in sensitivity} == {"B1", "logreg_l2", "xgboost", "mlp"}
    assert sum(c.model == "B1" for c in sensitivity) == 1
    # Added after the main comparison: extra model families and the complete-case analysis.
    extra = [c for path in [EXPERIMENTS / "additional_models.yaml"] for c in load_experiments(path)]
    assert {c.model for c in extra} >= {
        "elasticnet",
        "cart",
        "random_forest",
        "svm_rbf",
        "ft_transformer",
    }
    assert all(c.population == "P_pred" and c.split in ("S1", "S4") for c in extra)
    complete = load_experiments(EXPERIMENTS / "complete_case.yaml")
    assert {c.population for c in complete} == {"P_pred_complete"}
    assert {c.feature_set for c in complete} == {"FS0", "FS1", "FS2"}
    assert {c.feature_set for c in complete if c.model == "B1"} == {"FS0", "FS1", "FS2"}
    main = [c for c in configs if c not in sensitivity and c not in extra and c not in complete]
    assert all(c.population == "P_pred" for c in main)
    configs = main
    assert all(c.n_trials == 50 and c.n_boot == 1000 for c in configs)
    baselines = {(c.model, c.split) for c in configs if c.model.startswith("B")}
    expected = {(m, s) for m in ("B0", "B1", "B2", "B3") for s in ("S1", "S2", "S3")}
    assert baselines == expected | {("B1", "S4")}  # B1 is the S4 comparator (deployment_check.yaml)
    p0_s1 = {(c.model, c.feature_set, c.missing_strategy) for c in configs if c.split == "S1"}
    for fs in ("FS0", "FS1", "FS2", "FS3", "FS4"):
        assert ("xgboost", fs, "M0") in p0_s1
        for model in ("logreg_l2", "xgboost", "mlp"):
            assert {(model, fs, "M1"), (model, fs, "M2")} <= p0_s1
