"""Per-model performance tracking: threshold metrics, confusion matrices, tuning histories
and their figures."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from robson_ml import figures
from robson_ml.privacy import SECONDARY, SUPPRESSED
from robson_ml.tracking import (
    best_per_model,
    confusion_counts,
    fold_params_table,
    logged_ranking_metrics,
    logged_threshold_metrics,
    performance_metrics,
    performance_table,
    same_params,
    threshold_metrics,
    threshold_table,
    trial_history,
)

# 4 true positives, 1 false negative, 2 false positives, 13 true negatives at 0.5.
Y = np.array([1] * 5 + [0] * 15)
P = np.array([0.9, 0.8, 0.7, 0.6, 0.2] + [0.7, 0.55] + [0.1] * 13)


def test_threshold_metrics_match_hand_counts() -> None:
    m = threshold_metrics(Y, P)
    assert m["accuracy"] == pytest.approx(17 / 20)
    assert m["precision"] == pytest.approx(4 / 6)
    assert m["recall"] == pytest.approx(4 / 5)
    assert m["specificity"] == pytest.approx(13 / 15)
    assert m["F1"] == pytest.approx(2 * (4 / 6) * (4 / 5) / (4 / 6 + 4 / 5))


def test_threshold_metrics_with_no_predicted_cs_are_zero_not_nan() -> None:
    m = threshold_metrics(Y, np.zeros(len(Y)))
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["F1"] == 0.0
    assert m["specificity"] == 1.0


def test_threshold_metrics_reject_mismatched_inputs() -> None:
    with pytest.raises(ValueError):
        threshold_metrics([1, 0], [0.5])


def test_performance_metrics_include_probability_scores() -> None:
    m = performance_metrics(Y, P)
    assert set(m) == {
        "pooled AUC",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "F1",
        "log loss",
        "Brier",
    }
    assert 0.5 < m["pooled AUC"] <= 1.0
    assert m["log loss"] > 0 and 0 < m["Brier"] < 1


def test_logged_names_follow_the_harness_convention() -> None:
    assert set(logged_threshold_metrics(Y, P)) == {
        "pooled_precision_at_05",
        "pooled_recall_at_05",
        "pooled_specificity_at_05",
        "pooled_f1_at_05",
    }


def test_logged_ranking_metrics_is_the_pooled_average_precision() -> None:
    assert logged_ranking_metrics(Y, P) == {"pooled_pr_auc": average_precision_score(Y, P)}
    perfect = logged_ranking_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert perfect["pooled_pr_auc"] == pytest.approx(1.0)


def test_confusion_counts_keep_large_cells_numeric() -> None:
    y = np.array([1] * 30 + [0] * 30)
    p = np.array([0.9] * 20 + [0.1] * 10 + [0.8] * 10 + [0.2] * 20)
    table = confusion_counts(y, p)
    assert table.to_numpy().tolist() == [[20, 10], [10, 20]]
    assert table.sum().sum() == len(y)


def test_confusion_counts_hide_small_cells_and_their_row_complement() -> None:
    table = confusion_counts(Y, P)  # TP 4, FN 1: the whole CS row is protected
    assert table.loc["CS (observed)"].tolist() == [SUPPRESSED, SUPPRESSED]
    assert table.loc["vaginal (observed)"].tolist() == [SECONDARY, SUPPRESSED]  # FP 2

    y = np.array([1] * 20 + [0] * 20)
    p = np.array([0.9] * 18 + [0.1] * 2 + [0.1] * 20)
    row = confusion_counts(y, p).loc["CS (observed)"].tolist()
    assert row == [SUPPRESSED, SECONDARY]


TABLE_COLUMNS = [
    "threshold",
    "tp",
    "fp",
    "fn",
    "tn",
    "precision",
    "recall",
    "specificity",
    "F1",
    "flagged_pct",
]


def _spread(n: int = 200, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = rng.random(n)
    return (rng.random(n) < p).astype(int), p


def test_threshold_table_has_one_row_per_threshold_matching_threshold_metrics() -> None:
    y, p = _spread()
    table = threshold_table(y, p, thresholds=(0.3, 0.5))
    assert list(table.columns) == TABLE_COLUMNS
    assert table["threshold"].tolist() == [0.3, 0.5]
    for _, row in table.iterrows():
        t = row["threshold"]
        pred = p >= t
        assert row["tp"] == np.sum(pred & (y == 1)) and row["fp"] == np.sum(pred & (y == 0))
        assert row["fn"] == np.sum(~pred & (y == 1)) and row["tn"] == np.sum(~pred & (y == 0))
        expected = threshold_metrics(y, p, t)
        for name in ("precision", "recall", "specificity", "F1"):
            assert row[name] == pytest.approx(expected[name])
        assert row["flagged_pct"] == pytest.approx(100 * pred.mean())


def test_threshold_table_default_thresholds() -> None:
    y, p = _spread()
    assert threshold_table(y, p)["threshold"].tolist() == pytest.approx(
        [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    )


def test_threshold_table_hides_small_counts_and_what_reveals_them() -> None:
    row = threshold_table(Y, P, thresholds=(0.5,)).iloc[0]  # TP 4, FN 1, FP 2, TN 13
    assert [row["tp"], row["fn"]] == [SUPPRESSED, SUPPRESSED]
    assert [row["fp"], row["tn"]] == [SUPPRESSED, SECONDARY]
    for name in ("precision", "recall", "specificity", "F1", "flagged_pct"):
        assert row[name] == SECONDARY

    y = np.array([1] * 20 + [0] * 20)
    p = np.array([0.9] * 18 + [0.1] * 2 + [0.9] * 10 + [0.1] * 10)
    row = threshold_table(y, p, thresholds=(0.5,)).iloc[0]  # TP 18, FN 2, FP 10, TN 10
    assert [row["tp"], row["fn"]] == [SECONDARY, SUPPRESSED]
    assert [row["fp"], row["tn"]] == [10, 10]
    assert row["specificity"] == pytest.approx(0.5)
    assert row["recall"] == SECONDARY and row["precision"] == SECONDARY


def _runs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": ["a", "b", "c", "d", "e"],
            "model": ["logreg_l2", "logreg_l2", "xgboost", "B0", "B1"],
            "feature_set": ["FS1", "FS2", "FS2", "FS0", "FS0"],
            "missing_strategy": ["M1", "M2", "M0", "M0", "M0"],
            "split": ["S1", "S1", "S1", "S1", "S4"],
            "population": ["P_pred"] * 5,
            "mean_auc": [0.70, 0.75, 0.74, 0.5, 0.73],
        }
    )


def test_best_per_model_keeps_the_best_run_of_each_model_in_scope() -> None:
    best = best_per_model(_runs())
    assert best["run_id"].tolist() == ["b", "c"]  # B0 excluded, B1 is S4


def test_performance_table_skips_runs_without_predictions() -> None:
    best = best_per_model(_runs())
    table = performance_table(
        best, lambda run_id: pd.DataFrame({"y": Y, "p": P}) if run_id == "b" else None
    )
    assert table.index.tolist() == ["logreg_l2"]
    assert table.loc["logreg_l2", "configuration"] == "FS2, M2"
    assert table.loc["logreg_l2", "mean AUC (per fold)"] == pytest.approx(0.75)
    assert performance_table(best, lambda _: None).empty


def test_trial_history_tracks_the_best_value_so_far() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))

    def objective(trial: optuna.Trial) -> float:
        c = trial.suggest_float("C", 1e-3, 1e2, log=True)
        trial.set_user_attr("params", {"C": c, "seed": 0, "missing_strategy": "M1"})
        return (np.log10(c) - 0.5) ** 2

    study.optimize(objective, n_trials=12)
    history = trial_history(study)
    assert history["trial"].tolist() == list(range(1, 13))
    assert list(history.columns) == ["trial", "log_loss", "best_log_loss", "C"]
    assert history["best_log_loss"].is_monotonic_decreasing
    assert history["best_log_loss"].iloc[-1] == pytest.approx(study.best_value)


def test_tracking_figures_render() -> None:
    small = confusion_counts(Y, P)
    large = pd.DataFrame([[120, 30], [25, 80]], index=small.index, columns=small.columns)
    fig = figures.confusion_grid(
        {"model a": large, "model b": small, "model c": large}, "t", ncols=2
    )
    assert len([ax for ax in fig.axes if ax.get_visible()]) == 3

    table = pd.DataFrame({"precision": [0.6, 0.7], "recall": [0.5, 0.8]}, index=["a", "b"])
    assert figures.grouped_bars(table, "t", "model", "score", ylim=(0, 1)).axes

    history = pd.DataFrame(
        {"log_loss": [0.6, 0.55, 0.58], "C": [0.01, 1.0, 50.0], "layers": ["64", "32", "64-32"]}
    )
    fig = figures.tuning_panels({"logreg": (history, "C"), "mlp": (history, "layers")}, "t")
    assert fig.axes[0].get_xscale() == "log"


def test_same_params_compares_numbers_with_tolerance_and_text_exactly() -> None:
    assert same_params(
        {"C": 0.1 + 1e-12, "layers": "64-32", "seed": 1}, {"C": 0.1, "layers": "64-32"}
    )
    assert not same_params({"C": 0.2}, {"C": 0.1})
    assert not same_params({"layers": "64"}, {"layers": "64-32"})
    assert not same_params({"C": 0.1}, {"C": 0.1, "alpha": 1.0})
    assert not same_params({"C": 0.1}, {})


def test_fold_params_table_has_one_column_per_fold() -> None:
    table = fold_params_table(
        {"loho_A": {"C": 0.123456, "depth": 3}, "loho_B": {"C": 0.2, "depth": 4}, "loho_C": {}},
        "xgboost",
    )
    assert list(table.columns) == ["A", "B"]
    assert table.loc[("xgboost", "C"), "A"] == 0.1235
    assert fold_params_table({"loho_A": {}}, "B1").empty
