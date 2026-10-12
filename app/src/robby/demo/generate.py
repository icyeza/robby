"""Synthetic admission inputs for the demo. Every number here is invented for the demo only
and exists nowhere else in the application."""

from __future__ import annotations

import random
from dataclasses import replace
from datetime import datetime
from decimal import Decimal

from robby.clinical.form import Parsed

FACILITY_WEIGHTS = (0.34, 0.26, 0.22, 0.18)
# Demo-only offsets added to the model probability when sampling outcomes.
FACILITY_OFFSETS = (0.0, 0.08, -0.03, 0.02)
FALLBACK_CS_RATE = 0.28  # outcome probability when no prediction exists (demo only)

_EDUCATION = (
    "Complete Primary school",
    "Completed secondary school",
    "Did not complete primary school",
    "University education or more",
)
_INSURANCE = ("Mutuel", "Mutuel", "Mutuel", "RSSB", "None", "Others")


def _pick(rng: random.Random, options: tuple[object, ...], weights: tuple[float, ...]) -> object:
    return rng.choices(options, weights=weights)[0]


def _details(rng: random.Random, parity: int) -> dict[str, int | float | str]:
    if rng.random() > 0.55:
        return {}
    out: dict[str, int | float | str] = {
        "height_cm": round(rng.gauss(160, 6)),
        "weight_kg": round(rng.gauss(66, 9)),
        "residency": rng.choice(("Rural", "Urban")),
        "anc_contacts": rng.choices((1, 2, 3, 4), weights=(0.05, 0.1, 0.25, 0.6))[0],
    }
    out["height_cm"] = min(max(int(out["height_cm"]), 135), 190)
    out["weight_kg"] = min(max(int(out["weight_kg"]), 42), 130)
    if rng.random() < 0.7:
        out["insurance_type"] = rng.choice(_INSURANCE)
    if rng.random() < 0.5:
        out["education_level"] = rng.choice(_EDUCATION)
    if rng.random() < 0.3:
        out["hiv_positive"] = "Yes" if rng.random() < 0.04 else "No"
    if rng.random() < 0.3:
        out["gravidity"] = parity + 1 + (1 if rng.random() < 0.12 else 0)
    return out


def make_parsed(rng: random.Random, admitted_at: datetime, facility_index: int) -> Parsed:
    """One plausible, complete admission."""
    parity = _pick(rng, (0, 1, 2, 3, 4, 5), (0.38, 0.24, 0.17, 0.11, 0.06, 0.04))
    assert isinstance(parity, int)
    base_age = 23.5 if parity == 0 else 25 + 2.2 * parity
    age = int(min(max(round(rng.gauss(base_age, 4.5)), 15), 45))
    previous_cs = 0
    if parity > 0 and rng.random() < 0.13 + (0.03 if facility_index == 1 else 0):
        previous_cs = 1 if rng.random() < 0.8 else min(parity, 2 + (rng.random() < 0.2))
    presentation = _pick(
        rng, ("cephalic", "breech", "transverse", "oblique"), (0.95, 0.035, 0.012, 0.003)
    )
    plurality = 2 if rng.random() < 0.025 else 1
    if rng.random() < 0.12:
        ga = rng.randint(28, 36)
    else:
        ga_pick = _pick(rng, (37, 38, 39, 40, 41, 42), (0.08, 0.14, 0.28, 0.32, 0.16, 0.02))
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
        details=_details(rng, parity),
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
