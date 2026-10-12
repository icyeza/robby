"""Synthetic admission inputs for the demo. Every number here is invented for the demo only
and exists nowhere else in the application."""

from __future__ import annotations

import json
import math
import random
from dataclasses import replace
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from robby.clinical.form import Parsed

# Roughly the facility mix of the training data (the model keys are sorted by name).
FACILITY_WEIGHTS = (0.235, 0.096, 0.326, 0.343)
# Demo-only offsets added to the model probability when sampling outcomes.
FACILITY_OFFSETS = (0.0, 0.08, -0.03, 0.02)
FALLBACK_CS_RATE = 0.28  # outcome probability when no prediction exists (demo only)

DETAILS_GIVEN = 0.985  # share of admissions with the optional details filled in (see README)
SHIFT_DAYS = 35  # the demo story: antenatal visits drop in the most recent weeks
SHIFT_FIELD = "anc_contacts"
# Inputs the admission form asks for elsewhere or derives (not "other details").
_NOT_DETAILS = frozenset(
    {
        "gestational_age_weeks", "ga_band_lower", "ga_band_upper", "parity", "previous_cs_count",
        "plurality", "fetal_presentation", "maternal_age", "onset_of_labour", "facility_id",
    }
)  # fmt: skip


def _pick(rng: random.Random, options: tuple[object, ...], weights: tuple[float, ...]) -> object:
    return rng.choices(options, weights=weights)[0]


class DetailSampler:
    """Draws the optional details from the marginal distributions stored with the model
    (feature_schema.json and reference_distributions.json), so the demo looks like its
    training population. Spellings and levels are the trained ones."""

    def __init__(self, model_dir: Path) -> None:
        schema = json.loads((model_dir / "feature_schema.json").read_text(encoding="utf-8"))
        reference = json.loads(
            (model_dir / "reference_distributions.json").read_text(encoding="utf-8")
        )["features"]
        self.fields: list[tuple[dict[str, Any], dict[str, Any]]] = [
            (item, reference[item["name"]])
            for item in schema.get("inputs", [])
            if item["name"] not in _NOT_DETAILS and item["name"] in reference
        ]

    def sample(self, rng: random.Random, *, low_anc: bool = False) -> dict[str, int | float | str]:
        out: dict[str, int | float | str] = {}
        for item, entry in self.fields:
            name = item["name"]
            if low_anc and name == SHIFT_FIELD:
                out[name] = rng.choices((1, 2, 3, 4), weights=(0.1, 0.15, 0.2, 0.55))[0]
                continue
            if rng.random() < float(entry.get("missing", 0.0)):
                continue  # not recorded, as in the training data (format 2)
            buckets = entry["buckets"]
            bucket = rng.choices(buckets, weights=[b["proportion"] for b in buckets])[0]
            members = [m for m in bucket["members"] if m != "missing"]
            if not members:
                continue  # not recorded, as in the training data
            member = members[0]
            if member.startswith("level:"):
                out[name] = member.removeprefix("level:")
            else:
                out[name] = self._number(rng, item, entry, int(member.removeprefix("bin:")))
        return out

    @staticmethod
    def _number(
        rng: random.Random, item: dict[str, Any], entry: dict[str, Any], index: int
    ) -> int | float:
        """A value that falls in bin `index` (bins are right-closed: edge[i-1] < v <= edge[i])."""
        edges = entry["edges"]
        lo, hi = item.get("range") or (None, None)
        step = (edges[1] - edges[0]) if len(edges) > 1 else 1.0
        low = edges[index - 1] if index > 0 else edges[0] - 2 * step
        high = edges[index] if index < len(edges) else edges[-1] + 2 * step
        if lo is not None:
            low = max(low, lo)
        if hi is not None:
            high = min(high, hi)
        whole = bool(entry.get("integer")) or all(float(e).is_integer() for e in edges)
        if item["kind"] == "integer" or whole:
            first = int(math.ceil(low)) if index == 0 else int(math.floor(low)) + 1
            last = max(int(math.floor(high)), first)
            return rng.randint(first, last)
        value = round(rng.uniform(low, high), 1)
        return min(max(value, round(low + 0.1, 1)), high)


def make_parsed(
    rng: random.Random,
    admitted_at: datetime,
    facility_index: int,
    sampler: DetailSampler | None = None,
    *,
    low_anc: bool = False,
) -> Parsed:
    """One plausible, complete admission. With a sampler, ~90% have the optional details."""
    parity = _pick(rng, (0, 1, 2, 3, 4, 5), (0.38, 0.24, 0.17, 0.11, 0.06, 0.04))
    assert isinstance(parity, int)
    base_age = 23.5 if parity == 0 else 25 + 3.0 * parity
    age = int(min(max(round(rng.gauss(base_age, 6.5)), 15), 45))
    previous_cs = 0
    if parity > 0 and rng.random() < 0.13 + (0.03 if facility_index == 1 else 0):
        previous_cs = 1 if rng.random() < 0.8 else min(parity, 2 + (rng.random() < 0.2))
    presentation = _pick(
        rng, ("cephalic", "breech", "transverse", "oblique"), (0.95, 0.035, 0.012, 0.003)
    )
    plurality = 2 if rng.random() < 0.025 else 1
    if rng.random() < 0.05:
        ga = rng.randint(28, 36)
    else:
        ga_pick = _pick(rng, (37, 38, 39, 40, 41, 42), (0.06, 0.14, 0.28, 0.38, 0.13, 0.01))
        assert isinstance(ga_pick, int)
        ga = ga_pick
    elective = 0.35 if previous_cs else (0.6 if presentation != "cephalic" else 0.1)
    roll = rng.random()
    induced = 0.19 if facility_index == 1 else 0.15
    if roll < elective:
        onset = "prelabour_cs"
    elif roll < elective + induced:
        onset = "induced"
    else:
        onset = "spontaneous"

    details: dict[str, int | float | str] = {}
    if sampler is not None and rng.random() < DETAILS_GIVEN:
        details = sampler.sample(rng, low_anc=low_anc)
        if "gravidity" in details:  # consistent with parity: at least one pregnancy per birth
            details["gravidity"] = parity + 1 + (1 if rng.random() < 0.1 else 0)

    hypertensive = rng.random() < 0.10
    if hypertensive:
        sys_bp, dia_bp = rng.randint(140, 175), rng.randint(90, 112)
    else:
        sys_bp, dia_bp = int(rng.gauss(117, 11)), int(rng.gauss(73, 8))
        sys_bp, dia_bp = min(max(sys_bp, 90), 138), min(max(dia_bp, 55), 88)
    protein_weights = (
        (0.45, 0.2, 0.15, 0.12, 0.08) if hypertensive else (0.86, 0.06, 0.04, 0.03, 0.01)
    )
    protein = _pick(rng, ("negative", "trace", "+", "++", "+++"), protein_weights)
    assert isinstance(protein, str)
    glucose = Decimal(str(round(min(max(rng.gauss(5.0, 0.9), 2.8), 14.0), 1)))

    parsed = Parsed(
        admitted_at=admitted_at,
        maternal_age=age,
        ga_weeks=ga,
        parity=parity,
        previous_cs_count=previous_cs,
        plurality=plurality,
        presentation=str(presentation),
        onset=onset,
        systolic=sys_bp,
        diastolic=dia_bp,
        proteinuria=protein,
        glucose=glucose,
        details=details,
    )
    # Screening not done, with the reason the form asks for.
    if rng.random() < 0.06:
        parsed = replace(
            parsed,
            systolic=None,
            diastolic=None,
            bp_reason=rng.choice(("equipment_unavailable", "declined", "no_time_before_delivery")),
        )
    if rng.random() < 0.08:
        parsed = replace(
            parsed,
            proteinuria=None,
            protein_reason=rng.choice(("supplies_out_of_stock", "no_time_before_delivery")),
        )
    if rng.random() < 0.12:
        parsed = replace(
            parsed,
            glucose=None,
            glucose_reason=rng.choice(("equipment_unavailable", "declined", "other")),
        )
    return parsed


def make_flawed(rng: random.Random, parsed: Parsed) -> Parsed:
    """An incomplete or internally conflicting admission (for the unclassified lists)."""
    kind = rng.choice(("conflict", "no_ga", "no_onset", "no_presentation", "no_parity"))
    if kind == "conflict":
        return replace(parsed, parity=0, previous_cs_count=1)
    if kind == "no_ga":
        return replace(parsed, ga_weeks=None)
    if kind == "no_onset":
        return replace(parsed, onset=None)
    if kind == "no_presentation":
        return replace(parsed, presentation=None)
    return replace(parsed, parity=None)
