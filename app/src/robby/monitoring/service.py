"""Monitoring: tiles, alerts and tables computed on the fly from predictions, reconciled outcomes
and the event log (there is no alerts table). The statistics come from robson_serving.monitoring."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from robson_serving import monitoring as stats
from sqlalchemy import exists, select
from sqlalchemy.orm import Session, aliased

from robby import events
from robby.models import (
    Admission,
    AppUser,
    Classification,
    EventLog,
    Facility,
    ModelState,
    ModelVersion,
    Prediction,
    PredictionMode,
    ReconciledOutcome,
)
from robby.timeutil import utcnow

ACK_EVENT = "monitoring.alert_acknowledged"
SCORED_DAYS = 30
CALIBRATION_DAYS = 28
CALIBRATION_ALERT = 0.05
CITL_LIMIT = 0.10
OUTCOME_LIMIT = 0.70
OUTCOME_GRACE = timedelta(hours=72)  # a woman admitted today has no outcome yet, by design
OUTCOME_MIN = 20
DRIFT_MIN = stats.MIN_N
DRIFT_LIMIT = stats.PSI_THRESHOLD
DRIFT_AXIS = 0.4
DRIFT_SHOWN = 10
MINUS = "−"  # noqa: RUF001

FEATURE_LABELS = {
    "anc_contacts": "Antenatal visits",
    "maternal_age": "Maternal age",
    "gestational_age_weeks": "Gestational age",
    "ga_band_lower": "Gestational age band (lower)",
    "ga_band_upper": "Gestational age band (upper)",
    "parity": "Parity",
    "previous_cs_count": "Previous cesareans",
    "bmi": "BMI",
    "facility_id": "Facility",
    "robson_group_no_onset": "Robson group (without onset)",
}


def feature_label(name: str) -> str:
    return FEATURE_LABELS.get(name) or name.replace("_", " ").capitalize()


def week_bucket(day: date) -> tuple[str, date]:
    """ISO week label and its Monday: the date bucket in an alert key."""
    year, week, _ = day.isocalendar()
    return f"{year}-W{week:02d}", day - timedelta(days=day.weekday())


def signed(value: float, digits: int = 2) -> str:
    text = f"{abs(value):.{digits}f}"
    if float(text) == 0:
        return text
    return f"{MINUS}{text}" if value < 0 else f"+{text}"


@dataclass
class Alert:
    key: str
    text: str
    since: date
    acknowledged_by: str | None = None
    acknowledged_on: datetime | None = None

    @property
    def acknowledged(self) -> bool:
        return self.acknowledged_on is not None


@dataclass
class DriftRow:
    name: str
    label: str
    psi: float
    width: int
    flagged: bool


@dataclass
class FacilityRow:
    name: str
    n: int
    observed: float | None
    predicted: float | None
    diff_pts: int | None
    flagged: bool
    enough: bool


@dataclass
class Tile:
    k: str
    v: str
    s: str
    warn: bool = False


@dataclass
class Monitoring:
    model: ModelVersion | None
    tiles: list[Tile] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    drift: list[DriftRow] = field(default_factory=list)
    drift_note: str | None = None
    drift_total: int = 0
    drift_flagged: int = 0
    facilities: list[FacilityRow] = field(default_factory=list)

    @property
    def open_alerts(self) -> int:
        return sum(1 for a in self.alerts if not a.acknowledged)


@lru_cache(maxsize=8)
def _reference(path: str) -> dict[str, Any] | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    except (OSError, ValueError):
        return None


def active_model(db: Session) -> ModelVersion | None:
    return db.scalar(select(ModelVersion).where(ModelVersion.state == ModelState.ACTIVE))


def _frame(db: Session, model: ModelVersion, since: datetime) -> pd.DataFrame:
    """Active-mode predictions of the model on current (not superseded) admissions."""
    newer = aliased(Admission)
    group = (
        select(Classification.robson_group)
        .where(Classification.admission_id == Admission.id)
        .order_by(Classification.created_at.desc())
        .limit(1)
        .scalar_subquery()
    )
    rows = db.execute(
        select(
            Prediction.created_at.label("scored_at"),
            Prediction.probability,
            Prediction.feature_vector,
            Facility.code.label("facility"),
            Facility.name.label("facility_name"),
            group.label("robson_group"),
            ReconciledOutcome.cesarean.label("cs"),
        )
        .join(Admission, Admission.id == Prediction.admission_id)
        .join(Facility, Facility.id == Admission.facility_id)
        .outerjoin(ReconciledOutcome, ReconciledOutcome.prediction_id == Prediction.id)
        .where(
            Prediction.model_id == model.id,
            Prediction.mode == PredictionMode.ACTIVE,
            Prediction.created_at >= since,
            ~exists().where(newer.supersedes_id == Admission.id),
        )
    ).all()
    return pd.DataFrame(
        [
            {
                "scored_at": r.scored_at,
                "prob": float(r.probability),
                "cs": None if r.cs is None else float(r.cs),
                "facility": r.facility,
                "facility_name": r.facility_name,
                "robson_group": r.robson_group,
                "features": r.feature_vector,
            }
            for r in rows
        ],
        columns=[
            "scored_at",
            "prob",
            "cs",
            "facility",
            "facility_name",
            "robson_group",
            "features",
        ],
    )


def _acknowledgements(db: Session, keys: list[str]) -> dict[str, tuple[str | None, datetime]]:
    if not keys:
        return {}
    out: dict[str, tuple[str | None, datetime]] = {}
    rows = db.execute(
        select(EventLog.payload, EventLog.occurred_at, AppUser.username)
        .outerjoin(AppUser, AppUser.id == EventLog.actor_id)
        .where(EventLog.event_type == ACK_EVENT, EventLog.payload["key"].astext.in_(keys))
        .order_by(EventLog.seq)
    )
    for payload, at, username in rows:
        out.setdefault(payload["key"], (username, at))
    return out


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def build(db: Session, now: datetime | None = None, artefact_dir: Path | None = None) -> Monitoring:
    now = now or utcnow()
    model = active_model(db)
    result = Monitoring(model)
    if model is None:
        result.tiles = [
            Tile("Admissions scored (30 days)", "—", "No active model"),
            Tile("Outcomes recorded", "—", "No active model"),
            Tile("Calibration-in-the-large", "—", f"limit ±{CITL_LIMIT:.2f}"),
            Tile("Open alerts", "0", "No active model"),
        ]
        return result

    frame = _frame(db, model, now - timedelta(days=SCORED_DAYS))
    stamp = pd.Timestamp(now)
    alerts: list[Alert] = []
    bucket, since = week_bucket(now.date())

    # ---- outcomes recorded (only predictions old enough to expect an outcome)
    aged = frame[pd.to_datetime(frame["scored_at"]) <= stamp - OUTCOME_GRACE]
    scored = len(frame)
    recorded = int(aged["cs"].notna().sum())
    fraction = recorded / len(aged) if len(aged) else None
    for code, part in aged.groupby("facility"):
        if len(part) < OUTCOME_MIN:
            continue
        share = float(part["cs"].notna().mean())
        if share < OUTCOME_LIMIT:
            alerts.append(
                Alert(
                    f"outcomes:{code}:{bucket}",
                    f"{part['facility_name'].iloc[0]}: outcomes recorded for {_pct(share)} of "
                    f"admissions scored 3 or more days ago (limit {_pct(OUTCOME_LIMIT)}).",
                    since,
                )
            )

    # ---- calibration
    done = frame[frame["cs"].notna()].copy()
    window = f"{CALIBRATION_DAYS}D"
    metrics = stats.rolling_metrics(done, window, now=stamp, strata=("facility",))
    overall = metrics[metrics["stratum"] == "all"].iloc[0]
    citl_tile = Tile("Calibration-in-the-large", "—", "Not enough outcomes yet")
    if overall["status"] == stats.OK and not math.isnan(overall["calibration_in_the_large"]):
        citl = float(overall["calibration_in_the_large"])
        over = abs(citl) > CITL_LIMIT
        citl_tile = Tile(
            "Calibration-in-the-large",
            signed(citl),
            f"above limit ±{CITL_LIMIT:.2f}" if over else f"limit ±{CITL_LIMIT:.2f}",
            over,
        )
    names = dict(zip(frame["facility"], frame["facility_name"], strict=True))
    for _, m in metrics[metrics["stratum"] == "facility"].iterrows():
        code = str(m["level"])
        d = stats.calibration_drift(done[done["facility"] == code], window, now=stamp)
        enough = m["status"] == stats.OK
        flagged = bool(enough and abs(d.difference) > CALIBRATION_ALERT)
        diff = None if not d.n else d.observed_rate - d.mean_probability
        result.facilities.append(
            FacilityRow(
                names.get(code, code),
                d.n,
                d.observed_rate if d.n else None,
                d.mean_probability if d.n else None,
                None if diff is None else round(diff * 100),
                flagged,
                bool(enough),
            )
        )
        if flagged and diff is not None:
            direction = "above" if diff > 0 else "below"
            alerts.append(
                Alert(
                    f"calibration:{code}:{bucket}",
                    f"{names.get(code, code)}: observed cesarean rate has been "
                    f"{abs(diff) * 100:.0f} points {direction} predicted over the last "
                    f"{CALIBRATION_DAYS} days.",
                    since,
                )
            )
    result.facilities.sort(key=lambda r: r.name)

    # ---- input drift
    path = Path(model.artefact_path)
    if not path.is_absolute() and artefact_dir is not None:
        path = artefact_dir / path
    reference = _reference(str(path / "reference_distributions.json"))
    if reference is None:
        result.drift_note = "The reference distributions of the active model could not be read."
    elif len(frame) < DRIFT_MIN:
        result.drift_note = (
            f"Not enough data yet: {len(frame)} admissions scored in the last {SCORED_DAYS} "
            f"days; at least {DRIFT_MIN} are needed to compare inputs with the training data."
        )
    else:
        table = stats.feature_drift(pd.DataFrame(list(frame["features"])), reference)
        table = table[table["n"] > 0].sort_values("psi", ascending=False)
        result.drift_total = len(table)
        result.drift_flagged = int(table["flagged"].sum())
        for r in table.itertuples():
            psi = float(r.psi)
            if r.flagged:
                alerts.append(
                    Alert(
                        f"drift:{r.feature}:{bucket}",
                        f"Input drift: {feature_label(r.feature).lower()} distribution has "
                        f"shifted (PSI {psi:.2f}, limit {DRIFT_LIMIT:.2f}).",
                        since,
                    )
                )
            if len(result.drift) < max(DRIFT_SHOWN, result.drift_flagged):
                result.drift.append(
                    DriftRow(
                        r.feature,
                        feature_label(r.feature),
                        psi,
                        min(100, round(psi / DRIFT_AXIS * 100)),
                        bool(r.flagged),
                    )
                )

    acked = _acknowledgements(db, [a.key for a in alerts])
    for a in alerts:
        if a.key in acked:
            a.acknowledged_by, a.acknowledged_on = acked[a.key]
    alerts.sort(key=lambda a: (a.acknowledged, a.key))
    result.alerts = alerts
    open_count = result.open_alerts
    result.tiles = [
        Tile("Admissions scored (30 days)", f"{scored:,}", "All facilities"),
        Tile(
            "Outcomes recorded",
            f"{recorded:,}",
            "No admissions old enough yet"
            if fraction is None
            else f"{_pct(fraction)} of those scored 3+ days ago",
        ),
        citl_tile,
        Tile(
            "Open alerts", str(open_count), "Waiting to be acknowledged" if open_count else "None"
        ),
    ]
    return result


def acknowledge(db: Session, actor: AppUser, key: str, artefact_dir: Path | None = None) -> bool:
    """Record the acknowledgement of a currently raised alert. Does not commit. False when the
    key is not raised now (a stale page)."""
    alert = next((a for a in build(db, artefact_dir=artefact_dir).alerts if a.key == key), None)
    if alert is None:
        return False
    if not alert.acknowledged:
        events.append(db, actor, ACK_EVENT, None, {"key": key})
    return True
