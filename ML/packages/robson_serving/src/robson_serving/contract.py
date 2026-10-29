"""Golden inputs and outputs and the contract check.

``golden_inputs.parquet`` holds ``GOLDEN_N`` synthetic canonical admissions drawn from the
feature schema with a fixed seed (never real records): each input is missing with probability
``MISSING_RATE``, categories are drawn from the training levels (and, when the model only knows
``non_cephalic``, sometimes a precise non-cephalic presentation to exercise coarsening),
numbers uniformly within the allowed range (``DEFAULT_SPAN`` above the lower bound when there
is no upper bound). Draws that do not fit the schema are redrawn. ``golden_outputs.json`` holds
their probabilities. :func:`check_golden` rescores the stored inputs and compares with an
absolute tolerance of 1e-9.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robson_engine import COARSE_PRESENTATIONS
from robson_serving.artefact import GOLDEN_INPUTS_FILE, GOLDEN_OUTPUTS_FILE
from robson_serving.features import GA_BAND_FIELDS
from robson_serving.loader import Scorer, load
from robson_serving.schema import (
    CATEGORY,
    INTEGER,
    NON_CEPHALIC,
    PRESENTATION,
    FeatureSchema,
    InputField,
    SchemaError,
)

GOLDEN_N = 200
GOLDEN_SEED = 20261011
TOLERANCE = 1e-9
MISSING_RATE = 0.15
PRECISE_PRESENTATION_RATE = 0.1
DEFAULT_SPAN = 10.0
MAX_DRAWS_PER_ROW = 50
FALLBACK_LEVELS = ("synthetic_a", "synthetic_b")


@dataclass(frozen=True)
class GoldenCheck:
    """Result of rescoring the golden inputs."""

    n: int
    expected_n: int
    max_abs_diff: float
    tolerance: float

    @property
    def ok(self) -> bool:
        """Same number of outputs and every one within the tolerance."""
        return self.n == self.expected_n and self.max_abs_diff <= self.tolerance


def _draw_value(field: InputField, schema: FeatureSchema, rng: np.random.Generator) -> Any:
    if rng.random() < MISSING_RATE:
        return None
    if field.kind == CATEGORY:
        if (
            field.name == PRESENTATION
            and schema.coarsen_presentation
            and rng.random() < PRECISE_PRESENTATION_RATE
        ):
            return str(rng.choice(sorted(COARSE_PRESENTATIONS[NON_CEPHALIC])))
        levels = list(field.levels) if field.levels else list(FALLBACK_LEVELS)
        return str(rng.choice(levels))
    low, high = field.range if field.range is not None else (None, None)
    low = 0.0 if low is None else low
    high = low + DEFAULT_SPAN if high is None else high
    if field.kind == INTEGER:
        return float(rng.integers(math.ceil(low), math.floor(high) + 1))
    return float(np.round(rng.uniform(low, high), 1))


def draw_admission(schema: FeatureSchema, rng: np.random.Generator) -> dict[str, Any]:
    """One synthetic canonical admission over the schema's input fields."""
    record = {field.name: _draw_value(field, schema, rng) for field in schema.inputs}
    if set(GA_BAND_FIELDS) <= record.keys():
        lower, upper = (record[name] for name in GA_BAND_FIELDS)
        if lower is None or upper is None:
            record.update(dict.fromkeys(GA_BAND_FIELDS))
        else:
            record[GA_BAND_FIELDS[0]], record[GA_BAND_FIELDS[1]] = sorted((lower, upper))
    return record


def _typed_frame(records: list[dict[str, Any]], schema: FeatureSchema) -> pd.DataFrame:
    data: dict[str, pd.Series] = {}
    for field in schema.inputs:
        values = [record[field.name] for record in records]
        if field.kind == CATEGORY:
            data[field.name] = pd.Series(values, dtype=object)
        else:
            data[field.name] = pd.Series(
                [np.nan if v is None else v for v in values], dtype=np.float64
            )
    return pd.DataFrame(data, columns=[field.name for field in schema.inputs])


def generate_golden_inputs(
    scorer: Scorer, n: int = GOLDEN_N, seed: int = GOLDEN_SEED
) -> pd.DataFrame:
    """``n`` synthetic admissions that fit ``scorer``'s schema (deterministic in ``seed``)."""
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    draws = 0
    while len(records) < n:
        draws += 1
        if draws > n * MAX_DRAWS_PER_ROW:
            raise RuntimeError("could not draw enough admissions that fit the feature schema")
        record = draw_admission(scorer.schema, rng)
        try:
            scorer.build_row(record)
        except SchemaError:
            continue
        records.append(record)
    return _typed_frame(records, scorer.schema)


def write_golden(
    scorer: Scorer, directory: Path, n: int = GOLDEN_N, seed: int = GOLDEN_SEED
) -> GoldenCheck:
    """Write golden inputs and outputs into ``directory`` and check them once."""
    directory = Path(directory)
    generate_golden_inputs(scorer, n, seed).to_parquet(directory / GOLDEN_INPUTS_FILE, index=False)
    stored = pd.read_parquet(directory / GOLDEN_INPUTS_FILE)
    payload = {
        "version_label": scorer.version_label,
        "n": len(stored),
        "seed": seed,
        "tolerance": TOLERANCE,
        "probabilities": scorer.score(stored),
    }
    (directory / GOLDEN_OUTPUTS_FILE).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return check_golden(directory, scorer)


def check_golden(directory: str | Path, scorer: Scorer | None = None) -> GoldenCheck:
    """Rescore the golden inputs; compare with the golden outputs (absolute tolerance 1e-9)."""
    directory = Path(directory)
    scorer = scorer if scorer is not None else load(directory)
    inputs = pd.read_parquet(directory / GOLDEN_INPUTS_FILE)
    payload = json.loads((directory / GOLDEN_OUTPUTS_FILE).read_text(encoding="utf-8"))
    expected = [float(p) for p in payload["probabilities"]]
    actual = scorer.score(inputs)
    if len(actual) != len(expected):
        return GoldenCheck(len(actual), len(expected), math.inf, TOLERANCE)
    if not actual:
        return GoldenCheck(0, 0, 0.0, TOLERANCE)
    difference = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    return GoldenCheck(len(actual), len(expected), difference, TOLERANCE)
