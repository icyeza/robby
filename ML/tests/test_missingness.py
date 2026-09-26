"""Missingness and under-recording (spec §15.3 v1.2) on toy and synthetic data. Every
prevalence value here is FAKE."""

import numpy as np
import pandas as pd
import pytest

from robson_engine import load_rule_set
from robson_ml.audit_offline import OVERALL
from robson_ml.missingness import (
    MAR_STATEMENT,
    ORSummary,
    add_month,
    comissingness_patterns,
    missingness_by,
    missingness_models,
    missingness_report,
    reclassification,
    suppress_missingness,
    under_recording_sensitivity,
)
from robson_ml.populations import audit_population
from robson_ml.privacy import SECONDARY, SUPPRESSED
from robson_ml.robson_run import classify_frame
from tests.synthetic import make_admissions

HIDDEN = (SUPPRESSED, SECONDARY)


def test_reclassification_arithmetic_non_differential() -> None:
    # 1000 women, 400 CS; recorded yes: 8 in CS, 2 in vaginal (1% recorded).
    rec = reclassification(8, 400, 2, 600, prevalence=0.05, se_ratio=1.0)
    assert rec.feasible
    assert rec.se_cs == pytest.approx(0.01 / 0.05)
    assert rec.se_vaginal == pytest.approx(0.2)
    # True positives: 8 / 0.2 = 40 in CS, 2 / 0.2 = 10 in vaginal (50 = 5% of 1000).
    assert rec.q_cs == pytest.approx((40 - 8) / (400 - 8))
    assert rec.q_vaginal == pytest.approx((10 - 2) / (600 - 2))


def test_reclassification_differential_and_infeasible() -> None:
    rec = reclassification(8, 400, 2, 600, prevalence=0.05, se_ratio=0.5)
    assert rec.se_vaginal == pytest.approx(0.5 * rec.se_cs)
    assert 8 / rec.se_cs + 2 / rec.se_vaginal == pytest.approx(50.0)
    below = reclassification(8, 400, 2, 600, prevalence=0.005)
    assert not below.feasible and "below" in below.reason
    assert not reclassification(0, 400, 0, 600, prevalence=0.05).feasible


def _toy_audit(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    facility = rng.choice(["A", "B"], size=n)
    group = rng.choice([1, 3, 5], size=n)
    condition = rng.random(n) < 0.05
    logit = -0.5 + 1.5 * condition + (group == 5) * 1.5
    cs = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame(
        {
            "facility_id": facility,
            "robson_group": pd.array(group, dtype="Int64"),
            "robson_status": "resolved",
            "cs": pd.array(cs, dtype="Int64"),
            "preeclampsia_recorded": np.where(condition, "yes", "no"),
        }
    )


def test_under_recording_scenarios_and_tipping() -> None:
    audit = _toy_audit()
    result = under_recording_sensitivity(
        audit,
        "preeclampsia_recorded",
        [1.0, 10.0, 30.0],
        condition="pe",
        n_draws=15,
        seed=1,
        conclusions={"OR above 2": lambda s: s.or_median > 2.0},
    )
    scenarios = result.scenarios.set_index("scenario")
    assert scenarios.loc["recorded", "or_median"] > 1
    assert not scenarios.loc["pi=1%, se_ratio=1", "feasible"]  # below the recorded ~5%
    assert scenarios.loc["pi=10%, se_ratio=1", "feasible"]
    assert scenarios.loc["pi=10%, se_ratio=1", "n_draws_fitted"] == 15
    tipping = result.tipping.iloc[0]
    assert tipping["conclusion"] == "OR above 2"
    # Non-differential under-recording pushes the corrected OR away from 1 here, so the
    # tipping point is either absent or a grid value; it is never below the feasible range.
    assert np.isnan(tipping["tipping_prevalence_pct"]) or tipping["tipping_prevalence_pct"] >= 10
    rates = result.group_rates
    recorded = rates[rates["scenario"] == "recorded"]
    assert recorded["n"].sum() == len(audit)


def test_under_recording_is_seeded() -> None:
    audit = _toy_audit()
    a = under_recording_sensitivity(audit, "preeclampsia_recorded", [20.0], n_draws=5, seed=4)
    b = under_recording_sensitivity(audit, "preeclampsia_recorded", [20.0], n_draws=5, seed=4)
    pd.testing.assert_frame_equal(a.scenarios, b.scenarios)


def test_default_conclusions_are_explicit() -> None:
    from robson_ml.missingness import DEFAULT_CONCLUSIONS

    above = ORSummary(2.0, 1.5, 2.5, 1.2, 3.0, 10)
    straddles = ORSummary(1.2, 1.1, 1.3, 0.8, 1.8, 10)
    rules = DEFAULT_CONCLUSIONS
    assert rules["adjusted OR > 1"](above) and rules["adjusted OR > 1"](straddles)
    assert rules["adjusted OR 95% interval excludes 1"](above)
    assert not rules["adjusted OR 95% interval excludes 1"](straddles)


def test_missingness_by_counts_and_suppression() -> None:
    frame = pd.DataFrame(
        {
            "facility_id": ["A"] * 20 + ["B"] * 20,
            "height_cm": [np.nan] * 3 + [150.0] * 17 + [np.nan] * 10 + [160.0] * 10,
        }
    )
    table = missingness_by(frame, ["height_cm"], "facility_id")
    overall = table[table["facility_id"] == OVERALL].iloc[0]
    assert (overall["n"], overall["n_missing"]) == (40, 13)
    safe = suppress_missingness(table, "facility_id").set_index("facility_id")
    assert safe.loc["A", "n_missing"] == SUPPRESSED
    # A's 3 would follow from ALL (13) minus B (10): B is hidden too.
    assert safe.loc["B", "n_missing"] in HIDDEN


def test_patterns_pool_rare_and_suppress() -> None:
    frame = pd.DataFrame({"a": [np.nan] * 12 + [1.0] * 30, "b": [np.nan] * 2 + [1.0] * 40})
    table = comissingness_patterns(frame, ["a", "b"])
    assert "(other patterns)" in set(table["pattern"])
    assert not any(p == "a+b" for p in table["pattern"])  # 2 rows: pooled, never named


def test_missingness_models_and_report_on_synthetic() -> None:
    classified = classify_frame(make_admissions(700, seed=9), load_rule_set())
    audit = audit_population(classified)[0]
    summary, terms = missingness_models(add_month(audit), ["height_cm", "plurality"])
    assert set(terms["field"]) == {"height_cm"}
    summary = summary.set_index("field")
    assert 0.0 <= summary.loc["height_cm", "lr_p_value"] <= 1.0
    assert "MNAR" in summary.loc["height_cm", "interpretation"]
    assert "not modelled" in summary.loc["plurality", "interpretation"]
    report = missingness_report(audit, ["height_cm", "weight_kg"], priors=None)
    text = report.to_markdown()
    assert MAR_STATEMENT in text
    assert "never invented" in text
    assert report.under_recording == []
