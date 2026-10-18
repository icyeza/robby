"""Threshold trade-off of a run's pooled out-of-fold predictions: confusion counts,
precision, recall, specificity, F1 and the share flagged as likely CS at each threshold,
with the PR-AUC, the prevalence and the F1-maximising threshold.

Usage:
    uv run python scripts/threshold_report.py [--run-id RUN_ID]

The default run is the selected model's S1 (leave-one-hospital-out) run, read from the
packaged manifest ``artefacts/packaged/readiness-v1.0/manifest.json``. Reads
``data/interim/oof/<run_id>.parquet`` (columns ``y``, ``p``) and writes
``reports/threshold/<run_id>.csv`` and ``.md`` (aggregates only, small cells hidden).
The 0.5 default threshold and the selection rule are not changed: this is a report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from robson_ml.cli import load_project_config
from robson_ml.tracking import (
    THRESHOLD,
    TRADE_OFF_THRESHOLDS,
    logged_ranking_metrics,
    threshold_metrics,
    threshold_table,
)

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "artefacts" / "packaged" / "readiness-v1.0" / "manifest.json"
SELECTION_SPLIT = "S1"
F1_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def selected_run_id(manifest: Path = MANIFEST) -> str | None:
    """The run id of the packaged model's selection-split evaluation, if recorded."""
    if not manifest.exists():
        return None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    run_id = data.get("evaluation_metrics", {}).get(SELECTION_SPLIT, {}).get("run_id")
    return str(run_id) if run_id else None


def _fmt(value: object, digits: int) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def to_markdown(
    table: pd.DataFrame, run_id: str, pr_auc: float, prevalence: float, best: float
) -> str:
    digits = {"threshold": 2, "flagged_pct": 1}
    header = ["threshold", "TP", "FP", "FN", "TN", "precision", "recall", "specificity"]
    header += ["F1", "flagged %"]
    lines = [
        f"Threshold trade-off, run `{run_id}` (pooled out-of-fold predictions).",
        "",
        f"- PR-AUC (average precision): {pr_auc:.3f}; no-skill baseline = prevalence "
        f"{prevalence:.3f}",
        f"- F1-maximising threshold (0.05-0.95 by 0.01): {best:.2f}",
        f"- Deployed default threshold: {THRESHOLD:.2f} (unchanged)",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---:|" * len(header),
    ]
    for record in table.to_dict("records"):
        cells = [_fmt(record[c], digits.get(c, 3)) for c in table.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="MLflow run id (default: the packaged model's run)")
    args = parser.parse_args(argv)
    run_id = args.run_id or selected_run_id()
    if not run_id:
        print(f"no run id recorded in {MANIFEST}; pass --run-id", file=sys.stderr)
        return 2
    cfg = load_project_config(REPO / "configs" / "project.yaml")
    path = REPO / cfg.interim_dir / "oof" / f"{run_id}.parquet"
    if not path.exists():
        print(f"no out-of-fold predictions at {path}", file=sys.stderr)
        return 2
    oof = pd.read_parquet(path, columns=["y", "p"])
    y, p = oof["y"].to_numpy(), oof["p"].to_numpy()
    pr_auc = logged_ranking_metrics(y, p)["pooled_pr_auc"]
    best = float(max(F1_GRID, key=lambda t: threshold_metrics(y, p, float(t))["F1"]))
    thresholds = sorted({*TRADE_OFF_THRESHOLDS, best})
    table = threshold_table(y, p, thresholds)
    out_dir = REPO / cfg.reports_dir / "threshold"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"{run_id}.csv", index=False)
    markdown = to_markdown(table, run_id, pr_auc, float(np.mean(y)), best)
    (out_dir / f"{run_id}.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
