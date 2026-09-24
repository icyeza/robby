"""Aggregate figures for the pipeline notebook: one consistent, colour-blind-safe style.

Every function takes an already aggregated (and suppressed) table and returns a matplotlib
``Figure`` with a title and labelled axes; none takes rows of data, and none draws one mark
per woman. Suppressed cells (``"<5"``, ``"*"``) are never drawn as values: they are shown as
their marker. Nothing here selects a backend, so figures display inline in a notebook.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from cycler import cycler
from matplotlib.figure import Figure

from robson_ml.privacy import SECONDARY, SUPPRESSED

# Categorical order (validated for colour-vision deficiency on adjacent pairs); fixed order,
# never cycled past eight.
PALETTE = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#52514e"
SURFACE = "#fcfcfb"
SEQUENTIAL = "Blues"
DIVERGING = "RdBu_r"
HIDDEN_MARKERS = (SUPPRESSED, SECONDARY)
FIG_WIDTH = 8.0
# Cell labels are written only on heatmaps up to this many cells (larger ones get too dense).
MAX_ANNOTATED_CELLS = 150


def apply_style() -> None:
    """Set the shared matplotlib style (thin marks, recessive grid, readable text)."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": cycler(color=list(PALETTE)),
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "figure.dpi": 100,
            "savefig.dpi": 120,
        }
    )


def _numeric(values: Sequence[object]) -> npt.NDArray[np.float64]:
    return np.array(
        [np.nan if isinstance(v, str) else float(v) for v in values],  # type: ignore[arg-type]
        dtype=np.float64,
    )


def _finish(fig: Figure, title: str, note: str | None = None) -> Figure:
    fig.suptitle(title, fontweight="bold", fontsize=12, x=0.02, ha="left")
    if note:
        fig.text(0.02, 0.005, note, fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03 if note else 0, 1, 1))
    return fig


def bar_chart(
    labels: Sequence[str],
    values: Sequence[object],
    title: str,
    xlabel: str,
    ylabel: str,
    reference: float | None = None,
    reference_label: str | None = None,
    note: str | None = None,
    color: str = PALETTE[0],
) -> Figure:
    """Vertical bars of aggregate ``values``; a suppressed value is written as its marker
    instead of a bar. An optional horizontal reference line (e.g. the overall rate)."""
    heights = _numeric(values)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 4.0))
    positions = np.arange(len(labels))
    ax.bar(positions, np.nan_to_num(heights), color=color, width=0.7)
    top = float(np.nanmax(heights)) if np.isfinite(heights).any() else 1.0
    for x_pos, raw, height in zip(positions, values, heights, strict=True):
        if isinstance(raw, str):
            ax.text(
                float(x_pos), top * 0.02, raw, ha="center", va="bottom", color=MUTED, fontsize=9
            )
        else:
            ax.text(float(x_pos), height, f"{height:.1f}", ha="center", va="bottom", fontsize=8)
    if reference is not None:
        ax.axhline(reference, color=REFERENCE, linestyle="--", linewidth=1.2)
        ax.text(
            len(labels) - 0.5,
            reference,
            f" {reference_label or 'reference'}",
            color=MUTED,
            va="bottom",
            ha="right",
            fontsize=8,
        )
    ax.set_xticks(positions, labels, rotation=45 if len(labels) > 6 else 0, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return _finish(fig, title, note)


def histogram(
    table: pd.DataFrame,
    levels: Sequence[str],
    title: str,
    xlabel: str,
    level_labels: Mapping[str, str] | None = None,
    share: bool = True,
) -> Figure:
    """A binned histogram from :func:`robson_ml.eda.binned_hist_counts` (merged bins, no
    count of 1-4). With ``share`` each level is shown as % of its own total, so levels of
    different sizes (e.g. CS and vaginal births) compare by shape; bars sit side by side."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.8))
    positions = np.arange(len(table))
    width = 0.8 / max(len(levels), 1)
    for k, level in enumerate(levels):
        counts = table[level].astype(float).to_numpy()
        heights = 100.0 * counts / counts.sum() if share and counts.sum() else counts
        label = (level_labels or {}).get(level, level)
        ax.bar(
            positions + (k - (len(levels) - 1) / 2) * width,
            heights,
            width * 0.95,
            label=label,
            color=PALETTE[k % len(PALETTE)],
        )
    ax.set_xticks(positions, table["bin"], rotation=60, ha="right", fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("% of the group" if share else "records")
    if len(levels) > 1:
        ax.legend(loc="upper right")
    return _finish(fig, title, "Bins holding 1-4 records are merged with a neighbour.")


def heatmap(
    matrix: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
    cmap: str = SEQUENTIAL,
    vmin: float | None = None,
    vmax: float | None = None,
    fmt: str = "{:.0f}",
    note: str | None = None,
) -> Figure:
    """A heatmap of an aggregate matrix; hidden cells (markers) are grey and labelled."""
    values = matrix.apply(lambda col: _numeric(list(col))).to_numpy(dtype=np.float64)
    height = max(3.0, 0.32 * len(matrix) + 1.5)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, height))
    shown = np.ma.masked_invalid(values)
    colormap = mpl.colormaps[cmap].with_extremes(bad="#d9d8d4")
    image = ax.imshow(shown, aspect="auto", cmap=colormap, vmin=vmin, vmax=vmax)
    ax.grid(False)
    ax.set_xticks(
        np.arange(matrix.shape[1]), [str(c) for c in matrix.columns], rotation=45, ha="right"
    )
    ax.set_yticks(np.arange(matrix.shape[0]), [str(i) for i in matrix.index])
    if matrix.size <= MAX_ANNOTATED_CELLS:
        low = vmin if vmin is not None else float(np.nanmin(values)) if shown.count() else 0.0
        high = vmax if vmax is not None else float(np.nanmax(values)) if shown.count() else 1.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                raw = matrix.iat[i, j]
                if isinstance(raw, str):
                    text, color = raw, INK
                elif np.isnan(values[i, j]):
                    continue
                else:
                    text = fmt.format(values[i, j] + 0.0).replace("-0.0", "0.0")
                    scaled = (values[i, j] - low) / (high - low) if high > low else 0.0
                    color = (
                        "white" if abs(scaled - (0.5 if cmap == DIVERGING else 0)) > 0.45 else INK
                    )
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(image, ax=ax, label=cbar_label, fraction=0.03)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return _finish(fig, title, note)


def range_plot(
    table: pd.DataFrame,
    label: str,
    mean: str,
    low: str,
    high: str,
    title: str,
    xlabel: str,
    reference: float | None = None,
    reference_label: str | None = None,
    highlight: str | None = None,
) -> Figure:
    """One row per configuration: the mean (dot) and range (line) of a metric across
    folds, with an optional vertical reference (e.g. B1's mean LOHO AUC)."""
    data = table.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, max(3.0, 0.28 * len(data) + 1.4)))
    positions = np.arange(len(data))[::-1]
    for y_pos, (_, row) in zip(positions, data.iterrows(), strict=True):
        color = PALETTE[1] if highlight is not None and row[label] == highlight else PALETTE[0]
        ax.plot([row[low], row[high]], [y_pos, y_pos], color=color, linewidth=2)
        ax.plot(row[mean], y_pos, "o", color=color, markersize=7)
    if reference is not None:
        ax.axvline(
            reference,
            color=REFERENCE,
            linestyle="--",
            linewidth=1.2,
            label=reference_label or "reference",
        )
        ax.legend(loc="lower right")
    ax.set_yticks(positions, data[label], fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("configuration")
    return _finish(fig, title, "Dot: mean across held-out facilities; line: min-max range.")


def scatter_groups(
    table: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    title: str,
    xlabel: str,
    ylabel: str,
    xband: tuple[float, float] | None = None,
    band_label: str | None = None,
) -> Figure:
    """Configuration-level aggregates (one point per configuration, never per woman),
    coloured by ``group`` in the fixed palette order, with an optional shaded x band."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 4.5))
    if xband is not None:
        ax.axvspan(*xband, color=GRID, alpha=0.8, label=band_label)
    for k, (name, rows) in enumerate(table.groupby(group, sort=True)):
        ax.scatter(
            rows[x],
            rows[y],
            s=48,
            color=PALETTE[k % len(PALETTE)],
            label=str(name),
            edgecolor=SURFACE,
            linewidth=1.5,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=8)
    return _finish(fig, title)


def line_curves(
    curves: Mapping[str, pd.DataFrame],
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    diagonal: bool = False,
    references: Mapping[str, pd.DataFrame] | None = None,
    note: str | None = None,
    ylim: tuple[float, float] | None = None,
) -> Figure:
    """Aggregate curves (binned calibration, decision curves, partial dependence), one
    colour per curve in palette order; ``references`` are drawn dashed in grey tones."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 4.8))
    if diagonal:
        ax.plot([0, 1], [0, 1], linestyle=":", color=REFERENCE, linewidth=1, label="ideal")
    for k, (name, frame) in enumerate(curves.items()):
        ax.plot(
            frame[x],
            frame[y],
            marker="o",
            markersize=4,
            color=PALETTE[k % len(PALETTE)],
            label=name,
        )
    styles = ("--", "-.")
    for k, (name, frame) in enumerate((references or {}).items()):
        ax.plot(
            frame[x], frame[y], linestyle=styles[k % 2], color=REFERENCE, linewidth=1.2, label=name
        )
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=8)
    return _finish(fig, title, note)


def stacked_bars(
    table: pd.DataFrame, title: str, xlabel: str, ylabel: str, note: str | None = None
) -> Figure:
    """Horizontal stacked bars: one bar per column of ``table``, one segment per row.
    Hidden (marker) cells are left out of the bar and named in the note."""
    values = table.apply(lambda col: _numeric(list(col))).to_numpy(dtype=np.float64)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, max(2.5, 0.5 * table.shape[1] + 1.2)))
    positions = np.arange(table.shape[1])[::-1]
    left = np.zeros(table.shape[1])
    for k, part in enumerate(table.index):
        widths = np.nan_to_num(values[k])
        ax.barh(
            positions,
            widths,
            left=left,
            color=PALETTE[k % len(PALETTE)],
            label=str(part),
            edgecolor=SURFACE,
            linewidth=2,
            height=0.6,
        )
        left += widths
    ax.set_yticks(positions, [str(c) for c in table.columns])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=min(len(table), 4))
    return _finish(fig, title, note)


def importance_bars(
    table: pd.DataFrame,
    value: str,
    title: str,
    xlabel: str,
    error: str | None = None,
    top: int = 15,
    highlight: Sequence[str] = (),
) -> Figure:
    """Horizontal bars of a global importance (one bar per input feature, top ``top``);
    features in ``highlight`` (e.g. the Robson features) take the second palette colour."""
    data = table.sort_values(value, ascending=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, max(3.0, 0.3 * len(data) + 1.2)))
    colors = [PALETTE[1] if str(i) in highlight else PALETTE[0] for i in data.index]
    ax.barh(
        np.arange(len(data)),
        data[value],
        color=colors,
        height=0.7,
        xerr=data[error] if error else None,
        ecolor=MUTED,
        capsize=2,
    )
    ax.set_yticks(np.arange(len(data)), [str(i) for i in data.index], fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("input feature")
    note = f"Orange: {', '.join(highlight)}." if highlight else None
    return _finish(fig, title, note)
