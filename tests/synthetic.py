"""Synthetic canonical admissions for tests and golden files (spec §3.5).

Nothing here is derived from real records. Every rate below is invented so that tests see
realistic structure; none is a reference value and none may ever be reported as a finding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from robson_engine import load_rule_set
from robson_ml.robson_run import classify_frame
from robson_ml.schema import CANONICAL_BASE_COLUMNS

FACILITIES = ("FAC_A", "FAC_B", "FAC_C", "FAC_D")
FACILITY_SHARE = (0.30, 0.25, 0.25, 0.20)
FACILITY_LOGIT_SHIFT = {"FAC_A": -0.5, "FAC_B": 0.0, "FAC_C": 0.3, "FAC_D": 0.9}
SYNTHETIC_GROUP_CS_RATE = {
    1: 0.25,
    2: 0.45,
    3: 0.08,
    4: 0.20,
    5: 0.80,
    6: 0.90,
    7: 0.75,
    8: 0.70,
    9: 0.97,
    10: 0.40,
}
UNRESOLVED_CS_RATE = 0.45
HYPERTENSION_RATE = 0.06
HYPERTENSION_LOGIT_SHIFT = 0.6
ROBSON_INPUT_MISSING_RATE = {
    "parity": 0.01,
    "previous_cs_count": 0.02,
    "plurality": 0.005,
    "fetal_presentation": 0.02,
    "onset_of_labour": 0.01,
}
# Exact GA (free text) is recorded for about two thirds of rows; the GA band for every row
# whose true GA falls in a band. Band limits in days: 20+0 to 33+6, 35+0 to 37+6, 38+0 to
# 40+6 and 41+0 to 45+0 weeks. 34+0 to 34+6 falls in no band, like the gap in the real form.
EXACT_GA_RATE = 0.66
GA_BAND_DAYS = ((140, 237), (245, 265), (266, 286), (287, 315))
PRESENTATION_SHARE = (0.972, 0.020, 0.006, 0.002)
# Share of non-cephalic rows recorded with their precise type; the others are recorded as
# the coarse "non_cephalic" (the export records only malpresentation yes/no).
PRECISE_NON_CEPHALIC_RATE = 0.2
# Pairs of rows sharing a mother_key (about 3% of rows): same facility, mostly the same
# delivery date, never a multiple pregnancy.
SHARED_KEY_PAIR_RATE = 0.015
SHARED_KEY_SAME_DATE_RATE = 0.85
CONTRADICTION_RATE = 0.005
OUTCOME_MISSING_RATE = 0.003
PERIOD_START = pd.Timestamp("2023-11-01")
PERIOD_DAYS = 152  # 2023-11-01 to 2024-03-31
ONSETS = np.array(["spontaneous", "induced", "prelabour_cs"])
PRESENTATIONS = np.array(["cephalic", "breech", "transverse", "oblique"])
INDICATIONS = np.array(
    ["synthetic_indication_a", "synthetic_indication_b", "synthetic_indication_c"]
)


def _mask(series: pd.Series, missing: np.ndarray) -> pd.Series:
    return series.mask(missing)


def _ga_bands(days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The (lower, upper) band in weeks containing each GA in days; NaN in the gap."""
    lower = np.full(len(days), np.nan)
    upper = np.full(len(days), np.nan)
    for low, high in GA_BAND_DAYS:
        inside = (days >= low) & (days <= high)
        lower[inside] = low / 7.0
        upper[inside] = high / 7.0
    return lower, upper


def _mother_keys(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distinct ``MK_<hex>`` keys, with about ``2 * SHARED_KEY_PAIR_RATE`` of rows paired.

    Returns (keys, first rows, second rows): each second row takes its first row's key.
    """
    salt = int(rng.integers(0, 2**62))
    # An odd multiplier makes i -> i * m + salt a bijection mod 2**64, so keys are distinct.
    keys = np.array([f"MK_{(i * 0x9E3779B97F4A7C15 + salt) % 2**64:016x}" for i in range(n)])
    n_pairs = round(SHARED_KEY_PAIR_RATE * n)
    rows = rng.permutation(n)[: 2 * n_pairs]
    first, second = rows[:n_pairs], rows[n_pairs:]
    keys[second] = keys[first]
    return keys, first, second


def make_admissions(n: int = 2000, seed: int = 20260923) -> pd.DataFrame:
    """Generate ``n`` synthetic canonical admissions (spec §5 columns, canonical dtypes)."""
    rng = np.random.default_rng(seed)
    facility = rng.choice(FACILITIES, size=n, p=FACILITY_SHARE)
    day = rng.integers(0, PERIOD_DAYS, size=n)
    mother_key, first, second = _mother_keys(n, rng)
    facility[second] = facility[first]
    same_date = rng.random(len(second)) < SHARED_KEY_SAME_DATE_RATE
    day[second[same_date]] = day[first[same_date]]
    parity = np.where(rng.random(n) < 0.4, 0, rng.integers(1, 6, size=n))
    previous_cs = np.where(
        parity == 0, 0, rng.choice([0, 1, 2, 3], size=n, p=[0.72, 0.20, 0.06, 0.02])
    )
    contradiction = rng.random(n) < CONTRADICTION_RATE
    parity = np.where(contradiction, 0, parity)
    previous_cs = np.where(contradiction, 1, previous_cs)
    plurality = np.where(rng.random(n) < 0.02, 2, 1)
    plurality[np.concatenate([first, second])] = 1
    presentation = rng.choice(PRESENTATIONS, size=n, p=PRESENTATION_SHARE)
    recorded_presentation = np.where(
        (presentation != "cephalic") & (rng.random(n) >= PRECISE_NON_CEPHALIC_RATE),
        "non_cephalic",
        presentation,
    )
    ga_days = np.clip(np.round(rng.normal(271.6, 12.0, size=n)), 168, 301)
    band_lower, band_upper = _ga_bands(ga_days)
    exact_ga = pd.Series(ga_days / 7.0).mask(rng.random(n) >= EXACT_GA_RATE)
    onset = np.where(
        previous_cs >= 1,
        rng.choice(ONSETS, size=n, p=[0.40, 0.10, 0.50]),
        rng.choice(ONSETS, size=n, p=[0.72, 0.18, 0.10]),
    )
    planned_share = np.where(previous_cs >= 1, 0.7, 0.5)
    prelabour_type = np.where(
        onset == "prelabour_cs",
        np.where(rng.random(n) < planned_share, "planned", "emergency"),
        None,
    )
    hypertensive = rng.random(n) < HYPERTENSION_RATE
    pe_recorded = np.select(
        [hypertensive & (rng.random(n) < 0.5), rng.random(n) < 0.10], ["yes", "no"], default=""
    )
    gdm_draw = rng.random(n)
    gdm_recorded = np.select([gdm_draw < 0.002, gdm_draw < 0.05], ["yes", "no"], default="")

    df = pd.DataFrame(
        {
            "admission_id": [f"SYN{i:06d}" for i in range(n)],
            "mother_key": pd.Series(mother_key, dtype=object),
            "facility_id": facility,
            "delivery_date": (PERIOD_START + pd.to_timedelta(day, unit="D")).astype(
                "datetime64[ns]"
            ),
            "parity": pd.array(parity, dtype="Int64"),
            "previous_cs_count": pd.array(previous_cs, dtype="Int64"),
            "fetal_presentation": pd.Series(recorded_presentation, dtype=object),
            "plurality": pd.array(plurality, dtype="Int64"),
            "gestational_age_weeks": exact_ga,
            "ga_band_lower": band_lower,
            "ga_band_upper": band_upper,
            "onset_of_labour": pd.Series(onset, dtype=object),
            "prelabour_cs_type": pd.Series(prelabour_type, dtype=object),
            "maternal_age": np.round(np.clip(rng.normal(28.0, 6.0, size=n), 15, 48), 1),
            "height_cm": np.round(np.clip(rng.normal(158.0, 7.0, size=n), 135, 190), 1),
            "weight_kg": np.round(np.clip(rng.normal(68.0, 12.0, size=n), 40, 150), 1),
            "anc_contacts": pd.array(rng.poisson(4.0, size=n), dtype="Int64"),
            "preeclampsia_recorded": pd.Series(pe_recorded, dtype=object).mask(pe_recorded == ""),
            "gdm_recorded": pd.Series(gdm_recorded, dtype=object).mask(gdm_recorded == ""),
        }
    )
    anthropometry_missing = rng.random(n) < 0.5
    df["height_cm"] = _mask(df["height_cm"], anthropometry_missing)
    df["weight_kg"] = _mask(df["weight_kg"], anthropometry_missing)
    for column, rate in ROBSON_INPUT_MISSING_RATE.items():
        df[column] = _mask(df[column], rng.random(n) < rate)

    classified = classify_frame(df, load_rule_set())
    rate = classified["robson_group"].astype(object).map(SYNTHETIC_GROUP_CS_RATE).astype(float)
    rate = rate.fillna(UNRESOLVED_CS_RATE).to_numpy()
    logit = (
        np.log(rate / (1 - rate))
        + pd.Series(facility).map(FACILITY_LOGIT_SHIFT).to_numpy()
        + HYPERTENSION_LOGIT_SHIFT * hypertensive
    )
    cs = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    cs = np.where(onset == "prelabour_cs", 1, cs)
    mode = np.where(
        cs == 1,
        "cesarean",
        np.where(rng.random(n) < 0.93, "spontaneous_vaginal", "instrumental_vaginal"),
    )
    indication = np.where(cs == 1, rng.choice(INDICATIONS, size=n), None)
    outcome_missing = rng.random(n) < OUTCOME_MISSING_RATE
    df["mode_of_delivery"] = _mask(pd.Series(mode, dtype=object), outcome_missing)
    df["cs"] = _mask(pd.Series(pd.array(cs, dtype="Int64")), outcome_missing)
    df["recorded_indication"] = _mask(pd.Series(indication, dtype=object), outcome_missing)
    return df[list(CANONICAL_BASE_COLUMNS)]
