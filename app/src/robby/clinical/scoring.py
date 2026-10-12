"""The readiness model at serving time (NFR1, NFR7).

The scorer is loaded once per process and cached by (active model id, rule set id). Each request
re-reads only the id of the active model (one indexed query), so a promotion needs no restart.
A model that cannot be loaded is remembered for 30 seconds and never blocks saving."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path
from typing import Any

import robson_serving
from robson_engine.ruleset import RuleSet as EngineRuleSet
from sqlalchemy import select
from sqlalchemy.orm import Session

from robby.clinical.form import Parsed
from robby.models import Facility, ModelState, ModelVersion

logger = logging.getLogger("robby")

FAILURE_MEMORY_SECONDS = 30.0
ANC_TOP_CODE = 4  # the model was trained with antenatal contacts top-coded at "4 or more"
# Gestational-age bands of the data-collection form (mapping_ur_cmhs.yaml); week 34 is in none.
GA_BANDS = (
    (lambda w: w < 34, 20.0, 33.857142857142854),
    (lambda w: 35 <= w <= 37, 35.0, 37.857142857142854),
    (lambda w: 38 <= w <= 40, 38.0, 40.857142857142854),
    (lambda w: w > 40, 40.142857142857146, 45.0),
)


@dataclass(frozen=True)
class ActiveModel:
    row: ModelVersion
    scorer: Any


@dataclass(frozen=True)
class Score:
    model_id: Any
    probability: Decimal
    feature_vector: dict[str, Any]


def ga_band(weeks: int | None) -> tuple[float | None, float | None]:
    if weeks is None:
        return None, None
    for holds, low, high in GA_BANDS:
        if holds(weeks):
            return low, high
    return None, None


def canonical(parsed: Parsed, facility: Facility) -> dict[str, Any]:
    """The admission as the serving package wants it (names from the model's feature schema)."""
    low, high = ga_band(parsed.ga_weeks)
    record: dict[str, Any] = {
        "gestational_age_weeks": parsed.ga_weeks,
        "ga_band_lower": low,
        "ga_band_upper": high,
        "parity": parsed.parity,
        "previous_cs_count": parsed.previous_cs_count,
        "plurality": parsed.plurality,
        "fetal_presentation": parsed.presentation,
        "onset_of_labour": parsed.onset,
        "maternal_age": parsed.maternal_age,
        "facility_id": facility.model_facility_key,
    }
    record.update(parsed.details)
    anc = record.get("anc_contacts")
    if anc is not None:
        record["anc_contacts"] = min(anc, ANC_TOP_CODE)
    return record


class ModelRuntime:
    """Process-wide cache of the loaded scorer."""

    def __init__(self, artefact_dir: Path, loader: Callable[..., Any] | None = None) -> None:
        self.artefact_dir = artefact_dir
        self._loader = loader or robson_serving.load
        self._lock = threading.Lock()
        self._scorers: dict[tuple[Any, Any], Any] = {}
        self._failed: dict[tuple[Any, Any], float] = {}

    def active(
        self, db: Session, rule_set_id: Any, engine_rules: EngineRuleSet
    ) -> ActiveModel | None:
        row = db.scalar(select(ModelVersion).where(ModelVersion.state == ModelState.ACTIVE))
        if row is None:
            return None
        return self._loaded(row, rule_set_id, engine_rules)

    def shadows(
        self, db: Session, rule_set_id: Any, engine_rules: EngineRuleSet
    ) -> list[ActiveModel]:
        """Every shadow model that can be loaded (those that cannot are skipped and logged)."""
        rows = db.scalars(
            select(ModelVersion)
            .where(ModelVersion.state == ModelState.SHADOW)
            .order_by(ModelVersion.registered_at)
        )
        loaded = (self._loaded(row, rule_set_id, engine_rules) for row in rows)
        return [model for model in loaded if model is not None]

    def _loaded(
        self, row: ModelVersion, rule_set_id: Any, engine_rules: EngineRuleSet
    ) -> ActiveModel | None:
        key = (row.id, rule_set_id)
        scorer = self._scorers.get(key)
        if scorer is not None:
            return ActiveModel(row, scorer)
        failed_at = self._failed.get(key)
        if failed_at is not None and time.monotonic() - failed_at < FAILURE_MEMORY_SECONDS:
            return None
        with self._lock:
            try:
                path = Path(row.artefact_path)
                if not path.is_absolute():
                    path = self.artefact_dir / path
                scorer = self._loader(path, rule_set=engine_rules)
            except Exception:
                logger.exception("Model %s could not be loaded", row.version_label)
                self._failed[key] = time.monotonic()
                return None
            self._scorers[key] = scorer
            self._failed.pop(key, None)
        return ActiveModel(row, scorer)


def score(model: ActiveModel, parsed: Parsed, facility: Facility) -> Score:
    """Probability and the exact feature row. Raises on anything the model refuses."""
    scorer = model.scorer
    row = scorer.build_row(canonical(parsed, facility))
    probability = scorer.predict(scorer.model_frame([row]))[0]
    return Score(
        model.row.id,
        Decimal(str(probability)).quantize(Decimal("0.00001"), rounding=ROUND_HALF_EVEN),
        row,
    )
