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
HYPERTENSION_LOGIT_SHIFT = 0.6
ROBSON_INPUT_MISSING_RATE = {
    "parity": 0.01,
    "previous_cs_count": 0.02,
    "plurality": 0.005,
    "fetal_presentation": 0.02,
    "gestational_age_weeks": 0.04,
    "onset_of_labour": 0.01,
}
SCREENING_MISSING_RATE = {"FAC_A": 0.30, "FAC_B": 0.50, "FAC_C": 0.60, "FAC_D": 0.80}
CONTRADICTION_RATE = 0.005
OUTCOME_MISSING_RATE = 0.003
PERIOD_START = pd.Timestamp("2023-11-01 00:00")
PERIOD_HOURS = 152 * 24
ONSETS = np.array(["spontaneous", "induced", "prelabour_cs"])
PRESENTATIONS = np.array(["cephalic", "breech", "transverse", "oblique"])
PROTEINURIA = np.array(["neg", "trace", "1+", "2+", "3+"])
INDICATIONS = np.array(
    ["synthetic_indication_a", "synthetic_indication_b", "synthetic_indication_c"]
)


def _mask(series: pd.Series, missing: np.ndarray) -> pd.Series:
    return series.mask(missing)


def make_admissions(n: int = 2000, seed: int = 20260923) -> pd.DataFrame:
    """Generate ``n`` synthetic canonical admissions (spec §5 columns, canonical dtypes)."""
    rng = np.random.default_rng(seed)
    facility = rng.choice(FACILITIES, size=n, p=FACILITY_SHARE)
    parity = np.where(rng.random(n) < 0.4, 0, rng.integers(1, 6, size=n))
    previous_cs = np.where(
        parity == 0, 0, rng.choice([0, 1, 2, 3], size=n, p=[0.72, 0.20, 0.06, 0.02])
    )
    contradiction = rng.random(n) < CONTRADICTION_RATE
    parity = np.where(contradiction, 0, parity)
    previous_cs = np.where(contradiction, 1, previous_cs)
    plurality = np.where(rng.random(n) < 0.02, 2, 1)
    presentation = rng.choice(PRESENTATIONS, size=n, p=[0.94, 0.045, 0.01, 0.005])
    ga = np.clip(np.round(rng.normal(271.6, 12.0, size=n)), 168, 301) / 7.0
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
    systolic = np.round(rng.normal(118.0, 15.0, size=n))
    proteinuria = rng.choice(PROTEINURIA, size=n, p=[0.80, 0.10, 0.05, 0.03, 0.02])
    screening_rate = pd.Series(facility).map(SCREENING_MISSING_RATE).to_numpy()
    bp_missing = rng.random(n) < screening_rate
    hypertensive = (systolic >= 140) & np.isin(proteinuria, ["1+", "2+", "3+"])
    pe_recorded = np.select(
        [hypertensive & (rng.random(n) < 0.5), rng.random(n) < 0.10], ["yes", "no"], default=""
    )
    gdm_draw = rng.random(n)
    gdm_recorded = np.select([gdm_draw < 0.002, gdm_draw < 0.05], ["yes", "no"], default="")

    df = pd.DataFrame(
        {
            "admission_id": [f"SYN{i:06d}" for i in range(n)],
            "facility_id": facility,
            "admitted_at": PERIOD_START
            + pd.to_timedelta(rng.integers(0, PERIOD_HOURS, size=n), unit="h"),
            "parity": pd.array(parity, dtype="Int64"),
            "previous_cs_count": pd.array(previous_cs, dtype="Int64"),
            "fetal_presentation": pd.Series(presentation, dtype=object),
            "plurality": pd.array(plurality, dtype="Int64"),
            "gestational_age_weeks": ga,
            "onset_of_labour": pd.Series(onset, dtype=object),
            "prelabour_cs_type": pd.Series(prelabour_type, dtype=object),
            "maternal_age": np.round(np.clip(rng.normal(28.0, 6.0, size=n), 15, 48), 1),
            "height_cm": np.round(np.clip(rng.normal(158.0, 7.0, size=n), 135, 190), 1),
            "weight_kg": np.round(np.clip(rng.normal(68.0, 12.0, size=n), 40, 150), 1),
            "anc_contacts": pd.array(rng.poisson(4.0, size=n), dtype="Int64"),
            "systolic_bp": systolic,
            "diastolic_bp": np.round(rng.normal(75.0, 10.0, size=n)),
            "proteinuria": pd.Series(proteinuria, dtype=object),
            "glucose_mmol_l": np.round(np.clip(rng.normal(5.2, 1.2, size=n), 2.0, 20.0), 1),
            "preeclampsia_recorded": pd.Series(pe_recorded, dtype=object).mask(pe_recorded == ""),
            "gdm_recorded": pd.Series(gdm_recorded, dtype=object).mask(gdm_recorded == ""),
        }
    )
    anthropometry_missing = rng.random(n) < 0.5
    df["height_cm"] = _mask(df["height_cm"], anthropometry_missing)
    df["weight_kg"] = _mask(df["weight_kg"], anthropometry_missing)
    df["systolic_bp"] = _mask(df["systolic_bp"], bp_missing)
    df["diastolic_bp"] = _mask(df["diastolic_bp"], bp_missing)
    df["proteinuria"] = _mask(
        df["proteinuria"], rng.random(n) < np.minimum(screening_rate + 0.15, 0.98)
    )
    df["glucose_mmol_l"] = _mask(
        df["glucose_mmol_l"], rng.random(n) < np.minimum(screening_rate + 0.10, 0.98)
    )
    for column, rate in ROBSON_INPUT_MISSING_RATE.items():
        df[column] = _mask(df[column], rng.random(n) < rate)

    classified = classify_frame(df, load_rule_set())
    rate = classified["robson_group"].astype(object).map(SYNTHETIC_GROUP_CS_RATE).astype(float)
    rate = rate.fillna(UNRESOLVED_CS_RATE).to_numpy()
    logit = (
        np.log(rate / (1 - rate))
        + pd.Series(facility).map(FACILITY_LOGIT_SHIFT).to_numpy()
        + HYPERTENSION_LOGIT_SHIFT * (systolic >= 140)
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
    df["admitted_at"] = df["admitted_at"].astype("datetime64[ns]")
    return df[list(CANONICAL_BASE_COLUMNS)]
