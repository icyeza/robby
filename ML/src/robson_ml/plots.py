"""Per-run evaluation plots: calibration with histogram, ROC, decision curve.

Every plot is drawn from pooled out-of-fold predictions and shows aggregates only (binned
means, curve points); it is written as a PNG to be logged as an MLflow artifact.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import roc_curve

CALIBRATION_BINS = 10
HISTOGRAM_BINS = 40
FIGSIZE = (5.5, 5.5)
DPI = 120


def calibration_plot(y: npt.ArrayLike, p: npt.ArrayLike, path: Path, title: str) -> Path:
    """Observed CS rate against mean predicted probability in quantile bins, over a
    histogram of the predictions."""
    y_arr, p_arr = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.unique(np.quantile(p_arr, np.linspace(0, 1, CALIBRATION_BINS + 1)))
    bins = np.clip(np.searchsorted(edges[1:-1], p_arr, side="right"), 0, None)
    frame = pd.DataFrame({"bin": bins, "p": p_arr, "y": y_arr}).groupby("bin").mean()
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=FIGSIZE, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    top.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="ideal")
    top.plot(frame["p"], frame["y"], marker="o", label="model")
    top.set_ylabel("observed CS rate")
    top.set_title(title, fontsize=9)
    top.legend(loc="upper left")
    bottom.hist(p_arr, bins=HISTOGRAM_BINS, range=(0, 1), color="grey")
    bottom.set_xlabel("predicted probability")
    bottom.set_ylabel("rows")
    return _save(fig, path)


def roc_plot(y: npt.ArrayLike, p: npt.ArrayLike, auc: float, path: Path, title: str) -> Path:
    """ROC curve with the chance diagonal."""
    fpr, tpr, _ = roc_curve(np.asarray(y), np.asarray(p))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.plot(fpr, tpr, label=f"AUC {auc:.3f}")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="lower right")
    return _save(fig, path)


def decision_curve_plot(net_benefit: pd.DataFrame, path: Path, title: str) -> Path:
    """Net benefit of the model against treat-all and treat-none over the thresholds."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(net_benefit["threshold"], net_benefit["model"], marker="o", label="model")
    ax.plot(net_benefit["threshold"], net_benefit["treat_all"], label="treat all")
    ax.plot(net_benefit["threshold"], net_benefit["treat_none"], label="treat none")
    low = min(float(net_benefit["model"].min()), 0.0) - 0.02
    ax.set_ylim(low, max(float(net_benefit[["model", "treat_all"]].max().max()), 0.0) + 0.02)
    ax.set_xlabel("threshold probability")
    ax.set_ylabel("net benefit")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="upper right")
    return _save(fig, path)


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path
