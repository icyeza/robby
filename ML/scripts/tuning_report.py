"""Every hyperparameter-tuning trial of each tuned model, on every held-out hospital.

Usage:
    uv run python scripts/tuning_report.py [--models logreg_l2 xgboost ...]

For the best S1 configuration of each tuned model, re-runs the seeded Optuna search of every
leave-one-hospital-out fold (same inputs, inner folds and seeds as the logged run, so the best
trial is checked against the hyperparameters logged in MLflow) and writes, to
``reports/tuning/``:

* ``trials.csv``: one row per trial (model, fold, trial, inner-CV log loss, hyperparameters);
* ``summary.{csv,md}``: per model and fold, the number of trials, the best / median / worst
  inner-CV log loss, the gain of the best trial over the median one, and whether the best trial
  reproduces the logged hyperparameters;
* ``sensitivity.{csv,md}``: per model and hyperparameter, the rank correlation between its
  value and the inner-CV log loss over all trials (|rho| near 0: the loss hardly depends on it);
* ``<model>.png``: inner-CV log loss of every trial against each hyperparameter, one colour
  per held-out hospital, the best trial of each marked.

Aggregates only: one point per trial, never per woman.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from robson_ml import tracking
from robson_ml.cli import load_project_config
from robson_ml.evaluate import fold_tuning_study, make_folds, metric_key
from robson_ml.feature_sets import build_model_data
from robson_ml.features import build_raw_features, load_feature_registry
from robson_ml.ingest import read_workbook, select_sheet
from robson_ml.mapping import load_mapping
from robson_ml.select import (
    BASELINE_MODELS,
    experiment_config,
    load_runs,
    run_fold_params,
)

REPO = Path(__file__).resolve().parents[1]


def at(cfg_path: Path) -> Path:
    return cfg_path if cfg_path.is_absolute() else REPO / cfg_path


def summarise(trials: pd.DataFrame, logged_match: dict[tuple[str, str], bool]) -> pd.DataFrame:
    rows = []
    for (model, fold), g in trials.groupby(["model", "fold"], sort=False):
        loss = g["log_loss"]
        rows.append(
            {
                "model": model,
                "fold": fold,
                "trials": len(g),
                "best log loss": loss.min(),
                "median log loss": loss.median(),
                "worst log loss": loss.max(),
                "gain vs median": loss.median() - loss.min(),
                "reproduces logged params": logged_match.get((model, fold)),
            }
        )
    return pd.DataFrame(rows)


def sensitivity(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in trials.groupby("model", sort=False):
        for param in [c for c in g.columns if c not in FIXED and g[c].notna().any()]:
            x = g[param]
            if x.nunique() < 2:
                continue
            if not pd.api.types.is_numeric_dtype(x) or pd.api.types.is_bool_dtype(x):
                x = x.astype(str).astype("category").cat.codes
            rho = pd.Series(x.to_numpy()).corr(g["log_loss"].reset_index(drop=True), "spearman")
            rows.append({"model": model, "hyperparameter": param, "spearman rho": rho})
    return pd.DataFrame(rows)


def markdown_table(table: pd.DataFrame) -> str:
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in table.round(4).itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


FIXED = ("model", "fold", "trial", "log_loss", "best_log_loss")


def plot_model(model: str, g: pd.DataFrame, out: Path) -> None:
    params = [c for c in g.columns if c not in FIXED and g[c].notna().any() and g[c].nunique() > 1]
    if not params:
        return
    ncols = min(3, len(params))
    nrows = int(np.ceil(len(params) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    folds = list(dict.fromkeys(g["fold"]))
    for ax, param in zip(axes.flat, params, strict=False):
        x = g[param]
        numeric = pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x)
        levels = sorted(x.astype(str).unique())
        for k, fold in enumerate(folds):
            h = g[g["fold"] == fold]
            xs = (
                h[param].astype(float).to_numpy()
                if numeric
                else np.array([levels.index(v) for v in h[param].astype(str)], dtype=float)
            )
            ys = h["log_loss"].to_numpy()
            ax.scatter(xs, ys, s=14, alpha=0.7, color=f"C{k}", label=fold)
            best = int(np.argmin(ys))
            ax.scatter(xs[best], ys[best], s=80, facecolor="none", edgecolor="black")
        if numeric and x.min() > 0 and x.max() / x.min() > 100:
            ax.set_xscale("log")
        if not numeric:
            ax.set_xticks(range(len(levels)), levels, fontsize=7, rotation=30, ha="right")
        ax.set_xlabel(param, fontsize=8)
        ax.set_ylabel("inner-CV log loss", fontsize=8)
    for ax in list(axes.flat)[len(params) :]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=7, title="held-out hospital", title_fontsize=7)
    fig.suptitle(f"{model}: every tuning trial (circled: best of each hospital)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", help="tuned models to run (default: all)")
    args = parser.parse_args(argv)
    cfg = load_project_config(REPO / "configs" / "project.yaml")
    tracking_uri = (REPO / "mlruns").resolve().as_uri()
    runs = load_runs(tracking_uri)
    best = tracking.best_per_model(runs)
    best = best[~best["model"].isin(BASELINE_MODELS)]
    if args.models:
        best = best[best["model"].isin(args.models)]

    mapping = load_mapping(at(cfg.mapping_path))
    raw = select_sheet(read_workbook(at(cfg.raw_path)), mapping.sheet).reset_index(drop=True)
    classified = pd.read_parquet(at(cfg.processed_dir) / "canonical_robson.parquet")
    registry = load_feature_registry(at(cfg.features_path))
    raw_features, _ = build_raw_features(raw, registry)
    data = build_model_data(classified, registry, raw_features, "P_pred")
    folds = make_folds(data.meta, "S1", cfg.seed)

    frames, logged_match = [], {}
    for _, run in best.iterrows():
        model = str(run["model"])
        logged = run_fold_params(tracking_uri, str(run["run_id"]))
        for fold in folds:
            started = time.time()
            study = fold_tuning_study(data, experiment_config(run), fold)
            if study is None:
                continue
            history = tracking.trial_history(study)
            history.insert(0, "fold", fold.name)
            history.insert(0, "model", model)
            frames.append(history)
            logged_match[(model, fold.name)] = tracking.same_params(
                study.best_trial.user_attrs["params"], logged.get(metric_key(fold.name), {})
            )
            print(f"{model} {fold.name}: {len(history)} trials in {time.time() - started:.0f} s")
    if not frames:
        print("no tuned model to report", file=sys.stderr)
        return 2
    trials = pd.concat(frames, ignore_index=True)
    out = at(cfg.reports_dir) / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    trials.to_csv(out / "trials.csv", index=False)
    summary, sens = summarise(trials, logged_match), sensitivity(trials)
    for name, table in (("summary", summary), ("sensitivity", sens)):
        table.to_csv(out / f"{name}.csv", index=False)
        (out / f"{name}.md").write_text(markdown_table(table), encoding="utf-8")
        print(markdown_table(table))
    for model, g in trials.groupby("model", sort=False):
        plot_model(str(model), g, out / f"{model}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
