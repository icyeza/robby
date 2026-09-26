"""Case-mix adjustment (spec §15.2) on synthetic data; references are FAKE fixtures."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robson_engine import load_rule_set
from robson_ml.audit_offline import OVERALL
from robson_ml.casemix import (
    FULL,
    MERGED,
    NOT_APPLICABLE,
    SECTOR_CONFOUND_STATEMENT,
    casemix_analysis,
    drop_collinear,
    pct_reduction,
    robson_levels,
    separated_levels,
    suppress_observed_expected,
    vogel_expected,
    within_facility_resample,
)
from robson_ml.populations import audit_population
from robson_ml.privacy import SECONDARY, SUPPRESSED
from robson_ml.references import load_vogel
from robson_ml.robson_run import classify_frame
from tests.synthetic import FACILITY_LOGIT_SHIFT, make_admissions
from tests.synthetic_references import fake_vogel

HIDDEN = (SUPPRESSED, SECONDARY)


@pytest.fixture(scope="module")
def audit() -> pd.DataFrame:
    classified = classify_frame(make_admissions(1200, seed=7), load_rule_set())
    return audit_population(classified)[0].reset_index(drop=True)


def test_known_facility_effect_direction_is_recovered(audit: pd.DataFrame) -> None:
    # Synthetic CS logit shift: FAC_D is +1.4 above FAC_A, whatever the Robson group.
    assert FACILITY_LOGIT_SHIFT["FAC_D"] > FACILITY_LOGIT_SHIFT["FAC_A"]
    result = casemix_analysis(audit, n_boot=0, seed=1, reference_facility="FAC_A")
    odds = result.odds_ratios
    for model in ("M_raw", "M_adj1", "M_adj2"):
        row = odds[(odds["classification"] == FULL) & (odds["model"] == model)]
        d = row[row["facility"] == "FAC_D"].iloc[0]
        assert d["or"] > 1.0, model
        assert d["ci_low"] < d["or"] < d["ci_high"]
    assert set(odds["classification"]) == {FULL, MERGED}


def test_sector_statement_is_embedded(audit: pd.DataFrame, tmp_path: Path) -> None:
    result = casemix_analysis(audit, n_boot=0, seed=1)
    assert result.statement == SECTOR_CONFOUND_STATEMENT
    text = result.to_markdown()
    assert SECTOR_CONFOUND_STATEMENT in text
    result.write(tmp_path)
    assert SECTOR_CONFOUND_STATEMENT in (tmp_path / "casemix.md").read_text(encoding="utf-8")
    for name in ("odds_ratios.csv", "reductions.csv", "observed_expected.csv"):
        assert (tmp_path / name).exists()


def test_pct_reduction_arithmetic() -> None:
    assert pct_reduction(1.0, 0.25) == pytest.approx(75.0)
    assert pct_reduction(-0.5, 0.25) == pytest.approx(50.0)
    assert pct_reduction(0.4, 0.8) == pytest.approx(-100.0)
    assert np.isnan(pct_reduction(0.0, 0.3))


def test_vogel_observed_expected_arithmetic(tmp_path: Path) -> None:
    vogel = load_vogel(fake_vogel(tmp_path / "v.yaml"))
    # FAKE reference CS rates: group 1 = 14%, group 5 = 50%.
    rows = [("A", 1, 1)] * 30 + [("A", 1, 0)] * 70 + [("B", 5, 1)] * 60 + [("B", 5, 0)] * 40
    audit = pd.DataFrame(rows, columns=["facility_id", "robson_group", "cs"])
    audit["robson_status"] = "resolved"
    audit["robson_group"] = audit["robson_group"].astype("Int64")
    audit["cs"] = audit["cs"].astype("Int64")
    levels = robson_levels(audit)
    expected = vogel_expected(levels, vogel)
    assert expected.iloc[0] == pytest.approx(0.14)
    result = casemix_analysis(audit, vogel, n_boot=20, seed=3, bootstrap_models=False)
    oe = result.observed_expected.set_index("facility")
    assert oe.loc["A", "vogel_expected_rate"] == pytest.approx(0.14)
    assert oe.loc["A", "vogel_oe"] == pytest.approx(0.30 / 0.14)
    assert oe.loc["B", "vogel_oe"] == pytest.approx(0.60 / 0.50)
    # Overall: sum of group share x reference rate = 0.5 x 0.14 + 0.5 x 0.50.
    assert oe.loc[OVERALL, "vogel_expected_rate"] == pytest.approx(0.32)
    assert oe.loc[OVERALL, "vogel_oe"] == pytest.approx(0.45 / 0.32)
    assert oe.loc["A", "cmodel_oe"] == NOT_APPLICABLE
    low, high = oe.loc["A", "vogel_oe_low"], oe.loc["A", "vogel_oe_high"]
    assert low <= oe.loc["A", "vogel_oe"] <= high


def test_observed_expected_is_suppressed() -> None:
    table = pd.DataFrame(
        {
            "facility": [OVERALL, "A", "B", "C"],
            "n": [300, 100, 100, 100],
            "n_resolved": [253, 100, 100, 53],
            "n_cs_resolved": [103, 50, 50, 3],
            "observed_rate_resolved": [103 / 253, 0.5, 0.5, 3 / 53],
            "vogel_expected_rate": [0.3, 0.3, 0.3, 0.3],
            "vogel_oe": [1.3, 1.6, 1.6, 0.2],
            "vogel_oe_low": [1.1, 1.3, 1.3, 0.1],
            "vogel_oe_high": [1.5, 1.9, 1.9, 0.4],
            "n_scored": [NOT_APPLICABLE] * 4,
            "n_cs_scored": [NOT_APPLICABLE] * 4,
            "observed_rate_scored": [NOT_APPLICABLE] * 4,
            "cmodel_expected_rate": [NOT_APPLICABLE] * 4,
            "cmodel_oe": [NOT_APPLICABLE] * 4,
            "cmodel_oe_low": [NOT_APPLICABLE] * 4,
            "cmodel_oe_high": [NOT_APPLICABLE] * 4,
        }
    )
    safe = suppress_observed_expected(table).set_index("facility")
    assert safe.loc["C", "n_cs_resolved"] == SUPPRESSED
    assert safe.loc["C", "vogel_oe"] == SUPPRESSED
    # Secondary: C's CS count must not follow from ALL minus A minus B.
    hidden = safe["n_cs_resolved"].isin(HIDDEN)
    assert hidden.sum() >= 2


def test_within_facility_resample_keeps_facility_sizes() -> None:
    facility = pd.Series(["A"] * 5 + ["B"] * 3)
    idx = within_facility_resample(facility, np.random.default_rng(0))
    assert len(idx) == 8
    assert (facility.iloc[idx] == "A").sum() == 5


def test_separated_levels_and_collinear_columns() -> None:
    levels = pd.Series(["1", "1", "9", "9"])
    cs = pd.Series([0, 1, 1, 1])
    assert separated_levels(levels, cs) == ["9"]
    x = pd.DataFrame({"const": 1.0, "a": [0, 1, 0, 1.0], "b": [0, 1, 0, 1.0], "c": [1, 1, 0, 0.0]})
    assert list(drop_collinear(x).columns) == ["const", "a", "c"]
