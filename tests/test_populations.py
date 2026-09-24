from robson_ml.populations import audit_population, prediction_population, sensitivity_population
from tests.synthetic import make_admissions


def test_audit_population_keeps_non_missing_outcome() -> None:
    df = make_admissions(500, seed=1)
    audit, log = audit_population(df)
    assert audit["cs"].notna().all()
    assert log.n_excluded + log.n_remaining == len(df)
    assert log.n_remaining == len(audit)


def test_prediction_population_excludes_all_prelabour_cs() -> None:
    df = make_admissions(2000, seed=2)
    audit, _ = audit_population(df)
    pred, logs = prediction_population(audit)

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
    n_missing_onset = int(audit["onset_of_labour"].isna().sum())
    assert n_prelabour > 0 and n_planned > 0 and n_emergency > 0  # synthetic data has all cases
    assert len(pred) == len(audit) - n_prelabour - n_missing_onset
    by_reason = {log.reason: log.n_excluded for log in logs}
    assert by_reason["onset prelabour_cs planned"] == n_planned
    assert by_reason["onset prelabour_cs emergency"] == n_emergency
    assert by_reason["onset missing"] == n_missing_onset
    # synthetic prelabour_cs_type is always planned or emergency when onset is prelabour_cs
    assert by_reason["onset prelabour_cs type unknown"] == 0


def test_sensitivity_population_adds_back_emergency_prelabour_cs() -> None:
    df = make_admissions(2000, seed=3)
    audit, _ = audit_population(df)
    pred, _ = prediction_population(audit)
    sens, logs = sensitivity_population(audit)

    is_prelabour = audit["onset_of_labour"] == "prelabour_cs"
    n_emergency = int((is_prelabour & (audit["prelabour_cs_type"] == "emergency")).sum())
    assert len(sens) == len(pred) + n_emergency
    added = sens.iloc[len(pred) :]
    assert added["onset_of_labour"].isna().all()
    assert (added["cs"] == 1).all()  # every prelabour CS row is cs = 1 by definition
    sens_reason = "emergency pre-labour CS added back for sensitivity analysis"
    added_log = next(log for log in logs if log.reason == sens_reason)
    assert added_log.n_excluded == -n_emergency


def test_populations_are_deterministic_and_disjoint_from_excluded() -> None:
    df = make_admissions(300, seed=4)
    audit, _ = audit_population(df)
    pred, _ = prediction_population(audit)
    excluded_ids = set(audit["admission_id"]) - set(pred["admission_id"])
    assert excluded_ids.isdisjoint(set(pred["admission_id"]))
    # every excluded row is pre-labour CS or missing onset
    excluded = audit[audit["admission_id"].isin(excluded_ids)]
    onset = excluded["onset_of_labour"]
    assert ((onset == "prelabour_cs") | onset.isna()).all()
