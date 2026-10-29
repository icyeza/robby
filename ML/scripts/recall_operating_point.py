"""A recall-oriented operating point for the selected model, chosen without looking at the
hospital it is evaluated on.

Usage:
    uv run python scripts/recall_operating_point.py [--run-id RUN_ID]

The model's ranking (AUC) is fixed; only the probability cut that turns it into "likely CS"
changes. For each target recall, the cut for a held-out hospital is the one that reaches that
recall on the *other* hospitals' out-of-fold predictions, then applied to the held-out
hospital. Reports the recall, specificity, precision and share flagged actually obtained on the
held-out hospitals, pooled and per hospital. Reads
``data/interim/oof/<run_id>.parquet`` (columns ``fold``, ``y``, ``p``) and writes
``reports/recall/operating_point.{csv,md}`` (aggregates only). The deployed 0.5 and the
selection rule are not changed: this is a report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from robson_ml.cli import load_project_config
from robson_ml.tracking import threshold_metrics

REPO = Path(__file__).resolve().parents[1]
TARGETS = (0.50, 0.60, 0.70, 0.75)
COLUMNS = ["target recall", "scope", "mean cut", "recall", "specificity", "precision"]
COLUMNS += ["flagged %"]


def cut_for_recall(y: np.ndarray, p: np.ndarray, target: float) -> float:
    """The largest cut that still flags at least ``target`` of the observed CS."""
    return float(np.quantile(p[y == 1], 1 - target))


def evaluate(oof: pd.DataFrame, target: float) -> list[dict[str, object]]:
    rows, flagged_y, flagged_p, cuts = [], [], [], []
    for fold, held in oof.groupby("fold"):
        rest = oof[oof["fold"] != fold]
        cut = cut_for_recall(rest["y"].to_numpy(), rest["p"].to_numpy(), target)
        y, p = held["y"].to_numpy(), held["p"].to_numpy()
        rows.append(row(target, str(fold).removeprefix("loho_"), [cut], y, p))
        flagged_y.append(y)
        flagged_p.append((p >= cut).astype(float))
        cuts.append(cut)
    pooled_y = np.concatenate(flagged_y)
    pooled_flag = np.concatenate(flagged_p)
    rows.insert(0, row(target, "pooled (all held-out hospitals)", cuts, pooled_y, pooled_flag, 0.5))
    return rows


def row(
    target: float, scope: str, cuts: list[float], y, p, cut: float | None = None
) -> dict[str, object]:
    m = threshold_metrics(y, p, cut if cut is not None else cuts[0])
    return {
        "target recall": target,
        "scope": scope,
        "mean cut": float(np.mean(cuts)),
        "recall": m["recall"],
        "specificity": m["specificity"],
        "precision": m["precision"],
        "flagged %": float(100 * np.mean(p >= (cut if cut is not None else cuts[0]))),
    }


def to_markdown(table: pd.DataFrame, run_id: str) -> str:
    lines = [
        f"Recall-oriented cuts chosen on the other hospitals, run `{run_id}`.",
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "---|" * 2 + "---:|" * (len(COLUMNS) - 2),
    ]
    for r in table.to_dict("records"):
        cells = [str(r[c]) if isinstance(r[c], str) else f"{r[c]:.3f}" for c in COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="MLflow run id (default: the packaged model's run)")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).parent))
    from threshold_report import selected_run_id

    run_id = args.run_id or selected_run_id()
    if not run_id:
        print("no run id; pass --run-id", file=sys.stderr)
        return 2
    cfg = load_project_config(REPO / "configs" / "project.yaml")
    path = REPO / cfg.interim_dir / "oof" / f"{run_id}.parquet"
    if not path.exists():
        print(f"no out-of-fold predictions at {path}", file=sys.stderr)
        return 2
    oof = pd.read_parquet(path, columns=["fold", "y", "p"])
    table = pd.DataFrame([r for t in TARGETS for r in evaluate(oof, t)])
    out_dir = REPO / cfg.reports_dir / "recall"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "operating_point.csv", index=False)
    markdown = to_markdown(table, run_id)
    (out_dir / "operating_point.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
