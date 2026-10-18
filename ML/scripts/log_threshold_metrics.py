"""Add precision, recall, specificity and F1 at 0.5, and the PR-AUC, to experiment runs
logged before the harness tracked them, so the MLflow UI shows them for every run.

Usage:
    uv run python scripts/log_threshold_metrics.py [--dry-run]

Each finished run of the experiment gets ``pooled_{precision,recall,specificity,f1}_at_05``
and ``pooled_pr_auc`` (average precision), computed from its own out-of-fold predictions
(``data/interim/oof/<run_id>.parquet``). Only the metrics a run lacks are logged; a run
that already carries all of them, or whose out-of-fold file is missing, is left untouched.
Only metrics are added; no param, tag or artefact changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from robson_ml.cli import load_project_config
from robson_ml.evaluate import EXPERIMENT_NAME
from robson_ml.tracking import logged_ranking_metrics, logged_threshold_metrics

REPO = Path(__file__).resolve().parents[1]
THRESHOLD_NAMES = tuple(
    f"pooled_{name}_at_05" for name in ("precision", "recall", "specificity", "f1")
)
RANKING_NAMES = ("pooled_pr_auc",)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_project_config(REPO / "configs" / "project.yaml")
    oof_dir = REPO / cfg.interim_dir / "oof"
    tracking_uri = (REPO / "mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print("no experiment runs found")
        return 0
    client = MlflowClient(tracking_uri)
    runs = client.search_runs([experiment.experiment_id], "attributes.status = 'FINISHED'")
    added = skipped = missing = logged_keys = 0
    names = (*THRESHOLD_NAMES, *RANKING_NAMES)
    for run in runs:
        if all(name in run.data.metrics for name in names):
            skipped += 1
            continue
        path = oof_dir / f"{run.info.run_id}.parquet"
        if not path.exists():
            missing += 1
            continue
        oof = pd.read_parquet(path, columns=["y", "p"])
        y, p = oof["y"].to_numpy(), oof["p"].to_numpy()
        all_metrics = {**logged_threshold_metrics(y, p), **logged_ranking_metrics(y, p)}
        absent = {k: v for k, v in all_metrics.items() if k not in run.data.metrics}
        if not args.dry_run:
            for key, value in absent.items():
                client.log_metric(run.info.run_id, key, value)
        added += 1
        logged_keys += len(absent)
    verb = "would add" if args.dry_run else "added"
    print(
        f"{verb} {logged_keys} missing metrics to {added} runs; {skipped} already had all "
        f"of them; {missing} without an out-of-fold file"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
