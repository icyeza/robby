"""Per-hospital cuts set by a false-alarm budget, against one national cut.

Usage:
    uv run python scripts/hospital_cut_report.py [--run-id RUN_ID]

A false-alarm budget is the share of vaginal births a hospital accepts being flagged as likely
CS (a theatre prepared and not used). The cut that meets it is the (1 - budget) quantile of the
probabilities of the vaginal births. Two rules are compared for each budget:

- national: one cut for every hospital, chosen on the *other* hospitals' out-of-fold
  predictions and applied to the held-out one;
- own hospital: each hospital sets its cut on a random half of its own admissions and is
  evaluated on the other half, repeated ``REPEATS`` times, so no admission is graded with a cut
  chosen on it.

Also reports the starting cut for a new hospital (chosen on all four hospitals). Reads
``data/interim/oof/<run_id>.parquet`` (columns ``fold``, ``y``, ``p``) and writes
``reports/recall/hospital_cut.{csv,md}`` (aggregates only). The model, the 0.5 reporting cut and
the selection rule are not changed: this is a report.
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
BUDGETS = (0.20, 0.30)
REPEATS = 200
SEED = 20260923
COLUMNS = ["budget", "rule", "scope", "cut", "cut p10", "cut p90", "recall", "false alarms"]
COLUMNS += ["flagged %", "F1"]


def cut_for_budget(y: np.ndarray, p: np.ndarray, budget: float) -> float:
    """The cut that flags ``budget`` of the observed vaginal births."""
    return float(np.quantile(p[y == 0], 1 - budget))


def scores(y: np.ndarray, flagged: np.ndarray) -> dict[str, float]:
    m = threshold_metrics(y, flagged.astype(float), 0.5)
    return {
        "recall": m["recall"],
        "false alarms": 1 - m["specificity"],
        "flagged %": float(100 * np.mean(flagged)),
        "F1": m["F1"],
    }


def national(oof: pd.DataFrame, budget: float) -> list[dict[str, object]]:
    rows, ys, flags, cuts = [], [], [], []
    for fold, held in oof.groupby("fold"):
        rest = oof[oof["fold"] != fold]
        cut = cut_for_budget(rest["y"].to_numpy(), rest["p"].to_numpy(), budget)
        y, flagged = held["y"].to_numpy(), held["p"].to_numpy() >= cut
        rows.append(row(budget, str(fold).removeprefix("loho_"), [cut], y, flagged))
        ys.append(y)
        flags.append(flagged)
        cuts.append(cut)
    rows.insert(0, row(budget, "pooled", cuts, np.concatenate(ys), np.concatenate(flags)))
    return rows


def own_hospital(oof: pd.DataFrame, budget: float) -> list[dict[str, object]]:
    rng = np.random.default_rng(SEED)
    hospitals = {str(f).removeprefix("loho_"): g for f, g in oof.groupby("fold")}
    per = {h: {"cuts": [], "scores": []} for h in hospitals}
    pooled = []
    for _ in range(REPEATS):
        ys, flags = [], []
        for h, g in hospitals.items():
            y, p = g["y"].to_numpy(), g["p"].to_numpy()
            idx = rng.permutation(len(y))
            choose, test = idx[: len(y) // 2], idx[len(y) // 2 :]
            cut = cut_for_budget(y[choose], p[choose], budget)
            flagged = p[test] >= cut
            per[h]["cuts"].append(cut)
            per[h]["scores"].append(scores(y[test], flagged))
            ys.append(y[test])
            flags.append(flagged)
        pooled.append(scores(np.concatenate(ys), np.concatenate(flags)))
    all_cuts = [c for h in per for c in per[h]["cuts"]]
    rows = [summary(budget, "pooled", all_cuts, pooled)]
    rows += [summary(budget, h, per[h]["cuts"], per[h]["scores"]) for h in per]
    return rows


def row(budget, scope, cuts, y, flagged) -> dict[str, object]:
    return {
        "budget": budget,
        "rule": "national",
        "scope": scope,
        "cut": float(np.mean(cuts)),
        "cut p10": float(np.min(cuts)),
        "cut p90": float(np.max(cuts)),
        **scores(y, flagged),
    }


def summary(budget, scope, cuts, runs) -> dict[str, object]:
    means = pd.DataFrame(runs).mean()
    return {
        "budget": budget,
        "rule": "own hospital",
        "scope": scope,
        "cut": float(np.mean(cuts)),
        "cut p10": float(np.quantile(cuts, 0.1)),
        "cut p90": float(np.quantile(cuts, 0.9)),
        **{k: float(means[k]) for k in ("recall", "false alarms", "flagged %", "F1")},
    }


def to_markdown(table: pd.DataFrame, run_id: str, starts: dict[float, float]) -> str:
    lines = [
        f"Cuts set by a false-alarm budget, run `{run_id}`. National: one cut chosen on the other",
        "hospitals (cut p10/p90 = min/max over hospitals). Own hospital: cut chosen on half of the",
        f"hospital's admissions, evaluated on the other half, {REPEATS} repeats (cut p10/p90 over",
        "repeats).",
        "",
        "Starting cut for a new hospital (all four hospitals): "
        + ", ".join(f"{b:.0%} budget -> {c:.3f}" for b, c in starts.items()),
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "|---:|---|---|" + "---:|" * (len(COLUMNS) - 3),
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
    y, p = oof["y"].to_numpy(), oof["p"].to_numpy()
    starts = {b: cut_for_budget(y, p, b) for b in BUDGETS}
    table = pd.DataFrame([r for b in BUDGETS for r in national(oof, b) + own_hospital(oof, b)])
    out_dir = REPO / cfg.reports_dir / "recall"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "hospital_cut.csv", index=False)
    markdown = to_markdown(table, run_id, starts)
    (out_dir / "hospital_cut.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
