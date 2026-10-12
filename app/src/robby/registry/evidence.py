# ruff: noqa: RUF001
"""Evidence for a model: reconciled outcomes joined to predictions, shadow versus active.

`reconciled_rows` is the one query helper: predictions whose admission (or an admission it
replaced) has a recorded delivery outcome. It reads delivery_outcome directly, so it works
before and after the reconciliation job fills reconciled_outcome (same facts, same answer)."""

from __future__ import annotations

import math
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
from robson_serving.monitoring import MIN_EVENTS, MIN_N, rolling_metrics, shadow_compare
from sqlalchemy import text
from sqlalchemy.orm import Session

from robby.models import ModelState, ModelVersion
from robby.timeutil import utcnow

WINDOW_DAYS = 90
SEED = 20261011
BIN_MIN = 20
BINS = 10

_SQL = text(
    """
WITH RECURSIVE chain(admission_id, ancestor_id) AS (
    SELECT id, id FROM admission
    UNION ALL
    SELECT chain.admission_id, a.supersedes_id
    FROM chain JOIN admission a ON a.id = chain.ancestor_id
    WHERE a.supersedes_id IS NOT NULL
)
SELECT p.id AS prediction_id, p.admission_id, p.model_id, p.mode::text AS mode,
       p.probability::float8 AS prob, p.created_at AS scored_at,
       (o.mode = 'cesarean') AS cs, f.name AS facility, c.robson_group
FROM prediction p
JOIN admission adm ON adm.id = p.admission_id
JOIN facility f ON f.id = adm.facility_id
JOIN chain ON chain.admission_id = p.admission_id
JOIN delivery_outcome o ON o.admission_id = chain.ancestor_id
LEFT JOIN LATERAL (
    SELECT robson_group FROM classification c
    WHERE c.admission_id = p.admission_id ORDER BY c.created_at DESC LIMIT 1
) c ON true
WHERE p.created_at >= :since
  AND NOT EXISTS (SELECT 1 FROM admission r WHERE r.supersedes_id = p.admission_id)
"""
)

COLUMNS = [
    "prediction_id",
    "admission_id",
    "model_id",
    "mode",
    "prob",
    "scored_at",
    "cs",
    "facility",
    "robson_group",
]


def reconciled_rows(db: Session, since: datetime | None = None) -> pd.DataFrame:
    """One row per prediction that has a known outcome (columns as in COLUMNS)."""
    result = db.execute(_SQL, {"since": since or datetime(1970, 1, 1)})
    frame = pd.DataFrame(result.all(), columns=COLUMNS)
    if not frame.empty:
        frame["cs"] = frame["cs"].astype(float)
    return frame


@dataclass
class MetricRow:
    key: str
    label: str
    candidate: float
    active: float
    difference: float
    low: float
    high: float


@dataclass
class StratumRow:
    level: str
    n: int
    ready: bool
    observed: float | None = None
    candidate_mean: float | None = None
    active_mean: float | None = None
    candidate_auc: float | None = None
    active_auc: float | None = None


Bins = list[tuple[float, float, int]]


@dataclass
class Evidence:
    status: str  # ready | not_enough | no_active | not_applicable
    message: str = ""
    n: int = 0
    needed: int = MIN_N
    active_label: str = ""
    period: str = ""
    metrics: list[MetricRow] = field(default_factory=list)
    facilities: list[StratumRow] = field(default_factory=list)
    groups: list[StratumRow] = field(default_factory=list)
    candidate_bins: Bins = field(default_factory=list)
    active_bins: Bins = field(default_factory=list)


_cache: dict[tuple[Any, ...], Evidence] = {}
LABELS = {
    "auc": "Discrimination (AUC)",
    "calibration_slope": "Calibration slope",
    "calibration_in_the_large": "Calibration-in-the-large",
    "brier": "Brier score",
}


def n_boot() -> int:
    try:
        return max(50, int(os.environ.get("ROBBY_EVIDENCE_BOOTSTRAP", "300")))
    except ValueError:
        return 300


def pair_frame(rows: pd.DataFrame, model_id: uuid.UUID, active_id: uuid.UUID) -> pd.DataFrame:
    """Admissions scored by both models, one row each."""
    mine = rows[rows["model_id"] == model_id].drop_duplicates("admission_id")
    theirs = rows[rows["model_id"] == active_id].drop_duplicates("admission_id")
    joined = mine.merge(theirs[["admission_id", "prob"]], on="admission_id", suffixes=("", "_a"))
    return joined.rename(columns={"prob": "p_candidate", "prob_a": "p_active"})


def _bins(y: pd.Series, p: pd.Series) -> Bins:
    out: Bins = []
    idx = (p * BINS).clip(upper=BINS - 1).astype(int)
    for b in range(BINS):
        mask = idx == b
        if int(mask.sum()) >= BIN_MIN:
            out.append((float(p[mask].mean()), float(y[mask].mean()), int(mask.sum())))
    return out


def _enough(n: int, events: int) -> bool:
    return n >= MIN_N and min(events, n - events) >= MIN_EVENTS


def _strata(pairs: pd.DataFrame, column: str, now: datetime) -> list[StratumRow]:
    frame = pairs.copy()
    window = f"{WINDOW_DAYS}D"
    cand = rolling_metrics(
        frame.assign(prob=frame["p_candidate"]), window, now=now, strata=(column,)
    )
    act = rolling_metrics(frame.assign(prob=frame["p_active"]), window, now=now, strata=(column,))
    labels = frame[column].astype(object).where(frame[column].notna(), "missing").astype(str)
    rows: list[StratumRow] = []
    for _, c in cand[cand["stratum"] == column].iterrows():
        level = str(c["level"])
        a = act[(act["stratum"] == column) & (act["level"] == level)].iloc[0]
        sub = frame[labels == level]
        ready = c["status"] == "ok"
        rows.append(
            StratumRow(
                level=level,
                n=int(c["n"]),
                ready=ready,
                observed=float(sub["cs"].mean()),
                candidate_mean=float(sub["p_candidate"].mean()),
                active_mean=float(sub["p_active"].mean()),
                candidate_auc=float(c["auc"]) if ready and not math.isnan(c["auc"]) else None,
                active_auc=float(a["auc"]) if ready and not math.isnan(a["auc"]) else None,
            )
        )
    if column == "robson_group":
        rows.sort(key=lambda r: (not r.level.isdigit(), int(r.level) if r.level.isdigit() else 0))
    return rows


def build_evidence(
    db: Session, model: ModelVersion, now: datetime | None = None, resamples: int | None = None
) -> Evidence:
    if model.state is ModelState.CANDIDATE:
        return Evidence("not_applicable", "A candidate does not score admissions yet.")
    if model.state is ModelState.ACTIVE:
        return Evidence(
            "not_applicable",
            "This is the active model. Shadow versions are compared against it; its own "
            "performance over time belongs on the Monitoring page.",
        )
    if model.state is ModelState.RETIRED:
        return Evidence("not_applicable", "A retired model is no longer compared.")
    now = now or utcnow()
    active = db.execute(
        text("SELECT id, version_label FROM model_version WHERE state = 'active'")
    ).first()
    if active is None:
        return Evidence("no_active", "There is no active model to compare with.")
    rows = reconciled_rows(db, now - timedelta(days=WINDOW_DAYS))
    if rows.empty:
        pairs = pd.DataFrame(columns=["p_candidate", "p_active", "cs", "scored_at"])
    else:
        pairs = pair_frame(rows, model.id, active.id)
    n = len(pairs)
    events = int(pairs["cs"].sum()) if n else 0
    if not _enough(n, events):
        extra = (
            ""
            if n < MIN_N
            else f", with at least {MIN_EVENTS} cesareans and {MIN_EVENTS} other births"
        )
        return Evidence(
            "not_enough",
            f"Not enough recorded outcomes yet ({n} of {MIN_N} needed{extra}).",
            n=n,
            active_label=active.version_label,
        )
    boots = resamples or n_boot()
    key = (model.id, active.id, n, str(pairs["scored_at"].max()), now.date(), boots)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    table = shadow_compare(pairs, n_boot=boots, seed=SEED)
    metrics = [
        MetricRow(
            str(r["metric"]),
            LABELS[str(r["metric"])],
            r["candidate"],
            r["active"],
            r["difference"],
            r["ci_low"],
            r["ci_high"],
        )
        for _, r in table.iterrows()
        if r["metric"] in LABELS
    ]
    order = list(LABELS)
    metrics.sort(key=lambda m: order.index(m.key))
    result = Evidence(
        "ready",
        n=n,
        active_label=active.version_label,
        period=f"{pairs['scored_at'].min():%d %b} – {pairs['scored_at'].max():%d %b %Y}",
        metrics=metrics,
        facilities=_strata(pairs, "facility", now),
        groups=_strata(pairs, "robson_group", now),
        candidate_bins=_bins(pairs["cs"], pairs["p_candidate"]),
        active_bins=_bins(pairs["cs"], pairs["p_active"]),
    )
    _cache[key] = result
    return result


def calibration_svg(candidate: Bins, active: Bins) -> str:
    """Calibration plot as an SVG using attributes and CSS classes only (CSP: no inline style)."""
    size, pad = 300, 40
    span = size - 2 * pad

    def px(v: float) -> float:
        return pad + v * span

    def py(v: float) -> float:
        return size - pad - v * span

    parts = [
        f'<svg class="calplot" viewBox="0 0 {size} {size}" role="img" '
        'aria-label="Calibration: observed against predicted cesarean rate">',
        f'<rect class="cal-bg" x="{pad}" y="{pad}" width="{span}" height="{span}"/>',
    ]
    for t in (0, 0.25, 0.5, 0.75, 1):
        parts.append(
            f'<line class="cal-grid" x1="{px(t):.1f}" y1="{py(0):.1f}" '
            f'x2="{px(t):.1f}" y2="{py(1):.1f}"/>'
            f'<line class="cal-grid" x1="{px(0):.1f}" y1="{py(t):.1f}" '
            f'x2="{px(1):.1f}" y2="{py(t):.1f}"/>'
            f'<text class="cal-tick" x="{px(t):.1f}" y="{size - pad + 14}" '
            f'text-anchor="middle">{int(t * 100)}%</text>'
            f'<text class="cal-tick" x="{pad - 6}" y="{py(t) + 4:.1f}" '
            f'text-anchor="end">{int(t * 100)}%</text>'
        )
    parts.append(
        f'<line class="cal-ideal" x1="{px(0):.1f}" y1="{py(0):.1f}" '
        f'x2="{px(1):.1f}" y2="{py(1):.1f}"/>'
    )
    for css, series in (("cal-active", active), ("cal-cand", candidate)):
        if len(series) > 1:
            pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y, _ in series)
            parts.append(f'<polyline class="{css}-line" points="{pts}"/>')
        for x, y, k in series:
            parts.append(
                f'<circle class="{css}" cx="{px(x):.1f}" cy="{py(y):.1f}" r="4">'
                f"<title>predicted {x:.0%}, observed {y:.0%} ({k} admissions)</title></circle>"
            )
    parts.append(
        f'<text class="cal-axis" x="{size / 2}" y="{size - 6}" text-anchor="middle">'
        "Predicted cesarean rate</text>"
        f'<text class="cal-axis" x="12" y="{size / 2}" text-anchor="middle" '
        f'transform="rotate(-90 12 {size / 2})">Observed</text></svg>'
    )
    return "".join(parts)
