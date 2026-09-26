from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robson_engine import load_rule_set
from robson_ml import explain
from robson_ml.evaluate import ExperimentConfig, RunContext, run_experiment
from robson_ml.feature_sets import ModelData, build_model_data
from robson_ml.features import build_raw_features, load_feature_registry
from robson_ml.metrics import INSUFFICIENT
from robson_ml.robson_run import classify_frame
from robson_ml.select import (
    experiment_config,
    fold_metrics,
    load_runs,
    run_artifact_table,
    run_fold_params,
)
from tests.synthetic import make_admissions, make_raw_sheet

REGISTRY = Path(__file__).resolve().parents[1] / "configs" / "features_v1.yaml"
N = 900
SEED = 3


@dataclass
class Tracked:
    data: ModelData
    uri: str
    runs: pd.DataFrame


@pytest.fixture(scope="module")
def tracked(tmp_path_factory: pytest.TempPathFactory) -> Tracked:
    root = tmp_path_factory.mktemp("explain")
    registry = load_feature_registry(REGISTRY)
    canonical = classify_frame(make_admissions(N, seed=SEED), load_rule_set())
    raw_features, _ = build_raw_features(make_raw_sheet(registry, N, seed=SEED), registry)
    data = build_model_data(canonical, registry, raw_features, "P_pred")
    uri = (root / "mlruns").resolve().as_uri()
    ctx = RunContext(uri, root / "oof", "hash", registry.sha256, "commit", "rule")
    for model, fs, strategy in [
        ("B1", "FS0", "M0"),
        ("logreg_l2", "FS1", "M1"),
        ("xgboost", "FS0", "M0"),
    ]:
        config = ExperimentConfig(model, fs, strategy, "S1", "P_pred", 2, SEED, 20)
        run_experiment(config, data, ctx)
    return Tracked(data, uri, load_runs(uri))


def _fitted(tracked: Tracked, model: str) -> tuple[list[explain.FittedFold], pd.DataFrame, str]:
    row = tracked.runs[tracked.runs["model"] == model].iloc[0]
    config = experiment_config(row)
    fitted = explain.refit_folds(tracked.data, config, run_fold_params(tracked.uri, row["run_id"]))
    x = tracked.data.x[explain.model_columns(tracked.data, config)]
    return fitted, x, str(row["family"])


def test_load_runs_reads_configurations(tracked: Tracked) -> None:
    runs = tracked.runs
    assert sorted(runs["model"]) == ["B1", "logreg_l2", "xgboost"]
    assert set(runs["population_version"]) == {"v1.3"}
    assert runs.set_index("model")["complexity_rank"].to_dict() == {
        "B1": 0,
        "logreg_l2": 1,
        "xgboost": 5,
    }
    assert {"mean_auc", "mean_calibration_slope", "config", "family"} <= set(runs.columns)


@pytest.mark.parametrize("model", ["logreg_l2", "xgboost"])
def test_refit_reproduces_the_logged_fold_auc(tracked: Tracked, model: str) -> None:
    fitted, x, _ = _fitted(tracked, model)
    rebuilt = explain.heldout_auc(fitted, x, tracked.data.y)
    row = tracked.runs[tracked.runs["model"] == model]
    logged = fold_metrics(row, list(rebuilt.index), "auc").iloc[0]
    np.testing.assert_allclose(rebuilt.to_numpy(), logged.to_numpy(dtype=float), atol=1e-9)


def test_permutation_importance_is_global(tracked: Tracked) -> None:
    fitted, x, _ = _fitted(tracked, "logreg_l2")
    table = explain.permutation_importances(fitted, x, tracked.data.y, n_repeats=2, seed=1)
    assert list(table.index.sort_values()) == sorted(x.columns)
    assert list(table.columns[:2]) == ["mean", "sd"]
    assert len(table.columns) == 2 + len(fitted)
    assert table["mean"].is_monotonic_decreasing


def test_tree_shap_sums_back_to_inputs(tracked: Tracked) -> None:
    fitted, x, family = _fitted(tracked, "xgboost")
    assert family == "tree"
    table, method = explain.shap_importance(fitted, x, family, max_rows=80)
    assert "TreeSHAP" in method or "TreeExplainer" in method
    assert set(table.index) == set(x.columns)
    assert (table["mean_abs_shap"] >= 0).all()
    check = explain.robson_recovery(table["mean_abs_shap"])
    assert check.passed  # the synthetic outcome is driven by the Robson group


def test_agnostic_shap_for_non_tree_models(tracked: Tracked) -> None:
    fitted, x, family = _fitted(tracked, "logreg_l2")
    table, method = explain.shap_importance(fitted, x, family, max_rows=8, background=10)
    assert method == "shap.PermutationExplainer"
    assert set(table.index) == set(x.columns)
    assert table["mean_abs_shap"].max() > 0


def test_partial_dependence_is_an_average_curve(tracked: Tracked) -> None:
    fitted, x, _ = _fitted(tracked, "logreg_l2")
    grid = explain.pdp_grid(x["parity"], step=1.0)
    assert len(grid) >= 2 and np.all(np.mod(grid, 1) == 0)
    curve = explain.partial_dependence(fitted, x, "parity", grid, max_rows=200)
    assert curve["value"].tolist() == grid.tolist()
    assert curve["mean"].between(0, 1).all()


def test_pdp_grid_stays_inside_released_percentiles() -> None:
    values = pd.Series(np.r_[np.linspace(36, 41, 300), [20.0, 44.9]])
    grid = explain.pdp_grid(values, step=0.5)
    assert grid.min() >= 36 and grid.max() <= 41
    assert explain.pdp_grid(pd.Series([1.0, 2.0])).size == 0


def test_robson_recovery_check() -> None:
    importance = pd.Series({"a": 0.5, "b": 0.4, "previous_cs_count": 0.3, "c": 0.2})
    assert explain.robson_recovery(importance).found == ("previous_cs_count",)
    importance["previous_cs_count"] = 0.1
    assert not explain.robson_recovery(importance).passed


def test_input_feature_mapping() -> None:
    columns = ["parity", "fetal_presentation", "gestational_age_weeks"]
    assert explain.input_feature("num__parity", columns) == "parity"
    assert explain.input_feature("cat__fetal_presentation_breech", columns) == "fetal_presentation"
    assert (
        explain.input_feature("missing__missingindicator_gestational_age_weeks", columns)
        == "gestational_age_weeks"
    )


def test_subgroup_comparison_keeps_insufficient(tracked: Tracked) -> None:
    runs = tracked.runs.set_index("model")
    table = explain.subgroup_comparison(
        run_artifact_table(tracked.uri, runs.loc["xgboost", "run_id"], "subgroups.csv"),
        run_artifact_table(tracked.uri, runs.loc["B1", "run_id"], "subgroups.csv"),
    )
    assert list(table.columns) == ["selected", "B1"]
    assert (table == INSUFFICIENT).any().any()  # sparse groups 6-9 stay insufficient
