"""The readiness model at serving time (NFR1, NFR7).

The scorer is loaded once per process and cached by (active model id, rule set id). Each request
re-reads only the id of the active model (one indexed query), so a promotion needs no restart.
A model that cannot be loaded is remembered for FAILURE_MEMORY_SECONDS and never blocks saving."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path
from typing import Any, Protocol

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


class Scorer(Protocol):
    """What the app uses of a loaded model: robson_serving's Scorer, or a stub in tests."""

    @property
    def schema(self) -> Any: ...  # the feature schema; its optional inputs become form fields

    def build_row(self, admission: Mapping[str, Any]) -> dict[str, Any]: ...

    def model_frame(self, rows: Sequence[Mapping[str, Any]]) -> Any: ...

    def predict(self, frame: Any) -> Sequence[float]: ...


class ScorerLoader(Protocol):
    """Loads a packaged model for one rule set (robson_serving.load by default). Injected so the
    tests can serve a stub model without an artefact on disk."""

    def __call__(self, path: Path, *, rule_set: EngineRuleSet) -> Scorer: ...


@dataclass(frozen=True)
class ActiveModel:
    row: ModelVersion
    scorer: Scorer


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

    def __init__(self, artefact_dir: Path, loader: ScorerLoader | None = None) -> None:
        self.artefact_dir = artefact_dir
        self._loader: ScorerLoader = loader or robson_serving.load
        self._lock = threading.Lock()
        self._scorers: dict[tuple[Any, Any], Scorer] = {}
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
    return score_row(model, model.scorer.build_row(canonical(parsed, facility)))


def score_row(model: ActiveModel, row: dict[str, Any]) -> Score:
    """Probability of a feature row that is already built (a stored vector, re-scored)."""
    scorer = model.scorer
    probability = scorer.predict(scorer.model_frame([row]))[0]
    return Score(
        model.row.id,
        Decimal(str(probability)).quantize(Decimal("0.00001"), rounding=ROUND_HALF_EVEN),
        row,
    )


def yes_no_level(model: ActiveModel, name: str, yes: bool) -> str | None:
    """The model's own spelling of yes or no for the yes/no input `name` (e.g. "Yes"), as the
    admission form submits it; None when the model does not read that input."""
    for field in model.scorer.schema.inputs:
        if field.name == name and field.levels is not None:
            wanted = "yes" if yes else "no"
            return next((level for level in field.levels if level.lower() == wanted), None)
    return None


def with_input(
    model: ActiveModel, vector: Mapping[str, Any], name: str, value: Any
) -> dict[str, Any]:
    """`vector` (a stored feature row) with the input `name` set to `value`, spelled by the
    scorer itself: the columns the scorer derives differently once `name` is given (the column
    of that name, and any indicator built from it) are taken from its own row, the rest are kept
    as stored. Raises on a value the model refuses."""
    scorer = model.scorer
    without = scorer.build_row({})
    given = scorer.build_row({name: value})
    row = dict(vector)
    row.update({column: v for column, v in given.items() if without.get(column) != v})
    return row
