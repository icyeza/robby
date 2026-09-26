import logging

import pandas as pd
import pytest

from robson_ml.populations import (
    POPULATION_VERSION,
    REASON_MISSING_OUTCOME,
    REASON_PLANNED_CS,
    REASON_PLANNED_ONSET_VAGINAL,
    audit_population,
    onset_coded_population,
    prediction_population,
    sensitivity_population,
    untyped_cs_mask,
)
from tests.synthetic import make_admissions


def _toy() -> pd.DataFrame:
    """One row per v1.3 case; ``expect`` says whether P_pred keeps it."""
    rows = [
        # id, onset, cs, type, kept in P_pred v1.3
        ("planned_cs_prelabour_onset", "prelabour_cs", 1, "planned", False),
        ("planned_cs_spontaneous_onset", "spontaneous", 1, "planned", False),
        ("emergency_cs_prelabour_onset", "prelabour_cs", 1, "emergency", True),
        ("emergency_cs_in_labour", "spontaneous", 1, "emergency", True),
        ("untyped_cs", "induced", 1, None, True),
        ("untyped_cs_prelabour_onset", "prelabour_cs", 1, None, True),
        ("planned_onset_vaginal", "prelabour_cs", 0, None, False),
        ("vaginal", "spontaneous", 0, None, True),
        ("vaginal_onset_missing", None, 0, None, True),
        ("missing_outcome", "spontaneous", None, None, False),
    ]
    return pd.DataFrame(
        {
            "admission_id": [r[0] for r in rows],
            "onset_of_labour": pd.Series([r[1] for r in rows], dtype=object),
            "cs": pd.array([r[2] for r in rows], dtype="Int64"),
            "prelabour_cs_type": pd.Series([r[3] for r in rows], dtype=object),
            "expect": [r[4] for r in rows],
        }
    )


def test_population_version() -> None:
    assert POPULATION_VERSION == "v1.3"


def test_audit_population_keeps_non_missing_outcome() -> None:
    df = make_admissions(500, seed=1)
    audit, log = audit_population(df)
    assert audit["cs"].notna().all()
    assert log.n_excluded + log.n_remaining == len(df)
    assert log.n_remaining == len(audit)


def test_prediction_population_v13_on_toy_rows(caplog: pytest.LogCaptureFixture) -> None:
    """Spec v1.3: drop planned CS and planned-onset vaginal births; keep untyped CS."""
    df = _toy()
    with caplog.at_level(logging.INFO, logger="robson_ml.populations"):
        pred, logs = prediction_population(df)
    assert set(pred["admission_id"]) == set(df.loc[df["expect"], "admission_id"])
    by_reason = {log.reason: log for log in logs}
    assert [log.reason for log in logs] == [
        REASON_MISSING_OUTCOME,
        REASON_PLANNED_CS,
        REASON_PLANNED_ONSET_VAGINAL,
    ]
    assert by_reason[REASON_MISSING_OUTCOME].n_excluded == 1
    assert by_reason[REASON_PLANNED_CS].n_excluded == 2
    assert by_reason[REASON_PLANNED_ONSET_VAGINAL].n_excluded == 1
    assert all(log.n_remaining == len(pred) for log in logs[1:])
    # CS with no recorded type are kept and counted.
    assert int(untyped_cs_mask(pred).sum()) == 2
    assert "2 CS with no recorded type kept" in caplog.text


def test_prediction_population_is_idempotent_on_p_audit() -> None:
    df = _toy()
    audit, _ = audit_population(df)
    direct, _ = prediction_population(df)
    via_audit, logs = prediction_population(audit)
    assert list(direct["admission_id"]) == list(via_audit["admission_id"])
    assert logs[0].n_excluded == 0


def test_prediction_population_v13_on_synthetic() -> None:
    df = make_admissions(2000, seed=2)
    pred, logs = prediction_population(df)
    assert pred["cs"].notna().all()
    assert not ((pred["cs"] == 1) & (pred["prelabour_cs_type"] == "planned")).any()
    assert not ((pred["onset_of_labour"] == "prelabour_cs") & (pred["cs"] == 0)).any()
    # emergency pre-labour CS and untyped CS stay in (they were dropped under v1.2)
    assert ((pred["onset_of_labour"] == "prelabour_cs") & (pred["cs"] == 1)).any()
    assert untyped_cs_mask(pred).sum() > 0
    by_reason = {log.reason: log.n_excluded for log in logs}
    assert by_reason[REASON_PLANNED_CS] > 0
    assert by_reason[REASON_PLANNED_ONSET_VAGINAL] > 0
    assert len(pred) == len(df) - sum(by_reason.values())


def test_onset_coded_population_excludes_all_prelabour_cs() -> None:
    """The v1.2 definition, kept as the P_pred_onset_coded sensitivity population."""
    df = make_admissions(2000, seed=2)
    audit, _ = audit_population(df)
    pred, logs = onset_coded_population(audit)

    assert set(pred["onset_of_labour"].dropna().unique()) <= {"spontaneous", "induced"}
    assert not (pred["onset_of_labour"] == "prelabour_cs").any()

    reasons = {log.reason for log in logs}
    assert reasons == {
        "onset prelabour_cs planned",
        "onset prelabour_cs emergency",
        "onset prelabour_cs type unknown",
        "onset missing",
    }
    is_prelabour = audit["onset_of_labour"] == "prelabour_cs"
    n_prelabour = int(is_prelabour.sum())
    n_planned = int((is_prelabour & (audit["prelabour_cs_type"] == "planned")).sum())
    n_emergency = int((is_prelabour & (audit["prelabour_cs_type"] == "emergency")).sum())
    n_unknown = int((is_prelabour & audit["prelabour_cs_type"].isna()).sum())
    n_missing_onset = int(audit["onset_of_labour"].isna().sum())
    assert n_prelabour > 0 and n_planned > 0 and n_emergency > 0  # synthetic data has all cases
    assert len(pred) == len(audit) - n_prelabour - n_missing_onset
    by_reason = {log.reason: log.n_excluded for log in logs}
    assert by_reason["onset prelabour_cs planned"] == n_planned
    assert by_reason["onset prelabour_cs emergency"] == n_emergency
    assert by_reason["onset prelabour_cs type unknown"] == n_unknown
    assert by_reason["onset missing"] == n_missing_onset


def test_sensitivity_population_adds_back_emergency_prelabour_cs() -> None:
    df = make_admissions(2000, seed=3)
    audit, _ = audit_population(df)
    pred, _ = onset_coded_population(audit)
    sens, logs = sensitivity_population(audit)

    is_prelabour = audit["onset_of_labour"] == "prelabour_cs"
    n_emergency = int((is_prelabour & (audit["prelabour_cs_type"] == "emergency")).sum())
    assert len(sens) == len(pred) + n_emergency
    added = sens.iloc[len(pred) :]
    assert added["onset_of_labour"].isna().all()
    assert (added["cs"] == 1).all()  # a CS type is only recorded for CS rows
    sens_reason = "emergency pre-labour CS added back for sensitivity analysis"
    added_log = next(log for log in logs if log.reason == sens_reason)
    assert added_log.n_excluded == -n_emergency


def test_populations_are_deterministic_and_disjoint_from_excluded() -> None:
    df = make_admissions(300, seed=4)
    audit, _ = audit_population(df)
    pred, _ = prediction_population(audit)
    again, _ = prediction_population(audit)
    assert list(pred["admission_id"]) == list(again["admission_id"])
    excluded = audit[~audit["admission_id"].isin(set(pred["admission_id"]))]
    planned_cs = (excluded["cs"] == 1) & (excluded["prelabour_cs_type"] == "planned")
    planned_onset_vaginal = (excluded["onset_of_labour"] == "prelabour_cs") & (excluded["cs"] == 0)
    assert (planned_cs | planned_onset_vaginal).all()
