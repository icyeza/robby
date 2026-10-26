"""Recall of every model at the same operating points, to show that the low recall at the 0.5
cut is a property of the cut and of the 27% base rate, not of the selected model.

Usage:
    uv run python scripts/recall_report.py

For the best S1 configuration of each model (and B1, the Robson-group lookup), reads the pooled
leave-one-hospital-out out-of-fold predictions (``data/interim/oof/<run_id>.parquet``) and reports
recall, precision and specificity at (a) 0.5, the reporting convention, (b) the prevalence, the
cut at which a calibrated model flags about the share of admissions that really have a CS, and
(c) the cut that gives 0.80 and 0.90 specificity, so that models are compared at the same share
of vaginal births wrongly flagged. Writes ``reports/recall/recall_by_model.{csv,md}``
(aggregates only). The deployed 0.5 and the selection rule are not changed: this is a report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from robson_ml.cli import load_project_config
from robson_ml.select import load_runs
from robson_ml.tracking import best_per_model, logged_ranking_metrics, threshold_metrics

REPO = Path(__file__).resolve().parents[1]
SPECIFICITIES = (0.80, 0.90)


def cut_for_specificity(y: np.ndarray, p: np.ndarray, specificity: float) -> float:
    """The probability cut that leaves ``specificity`` of the vaginal births unflagged."""
    return float(np.quantile(p[y == 0], specificity))


def operating_points(y: np.ndarray, p: np.ndarray) -> list[tuple[str, float]]:
    points = [("0.5 (reporting convention)", 0.5), ("prevalence", float(np.mean(y)))]
    points += [(f"specificity {s:.2f}", cut_for_specificity(y, p, s)) for s in SPECIFICITIES]
    return points


def model_rows(model: str, config: str, y: np.ndarray, p: np.ndarray) -> list[dict[str, object]]:
    pr_auc = logged_ranking_metrics(y, p)["pooled_pr_auc"]
    rows = []
    for label, cut in operating_points(y, p):
        m = threshold_metrics(y, p, cut)
        rows.append(
            {
                "model": model,
                "configuration": config,
                "operating point": label,
                "cut": cut,
                "recall": m["recall"],
                "precision": m["precision"],
                "specificity": m["specificity"],
                "flagged %": float(100 * np.mean(p >= cut)),
                "PR-AUC": pr_auc,
            }
        )
    return rows


def to_markdown(table: pd.DataFrame, prevalence: float, n: int) -> str:
    cols = ["model", "configuration", "operating point", "cut", "recall", "precision"]
    cols += ["specificity", "flagged %"]
    lines = [
        f"Recall by model and operating point (pooled out-of-fold, {n:,} admissions, "
        f"CS prevalence {prevalence:.3f}).",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * 3 + "---:|" * (len(cols) - 3),
    ]
    for r in table.to_dict("records"):
        cells = [str(r[c]) if isinstance(r[c], str) else f"{r[c]:.3f}" for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    cfg = load_project_config(REPO / "configs" / "project.yaml")
    runs = load_runs((REPO / "mlruns").resolve().as_uri())
    best = best_per_model(runs)
    rows: list[dict[str, object]] = []
    prevalence, n = float("nan"), 0
    for _, run in best.iterrows():
        path = REPO / cfg.interim_dir / "oof" / f"{run['run_id']}.parquet"
        if not path.exists():
            continue
        oof = pd.read_parquet(path, columns=["y", "p"])
        y, p = oof["y"].to_numpy(), oof["p"].to_numpy()
        prevalence, n = float(np.mean(y)), len(y)
        config = f"{run['feature_set']}, {run['missing_strategy']}"
        rows += model_rows(str(run["model"]), config, y, p)
    if not rows:
        print("no out-of-fold predictions found", file=sys.stderr)
        return 2
    table = pd.DataFrame(rows)
    out_dir = REPO / cfg.reports_dir / "recall"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "recall_by_model.csv", index=False)
    markdown = to_markdown(table, prevalence, n)
    (out_dir / "recall_by_model.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
