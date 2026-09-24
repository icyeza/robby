from pathlib import Path

import numpy as np
import pandas as pd

from robson_ml.leakage import (
    completeness_pattern_flags,
    loho_single_feature_auc,
    name_pattern_flags,
    run_screens,
)


def _synthetic_frame(n: int = 2000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    facility = rng.choice(["FAC_A", "FAC_B", "FAC_C", "FAC_D"], size=n)
    cs = rng.integers(0, 2, size=n)
    return pd.DataFrame({"cs": cs, "facility_id": facility})


def test_loho_auc_flags_a_leaky_numeric_variable() -> None:
    frame = _synthetic_frame()
    rng = np.random.default_rng(0)
    # Deliberately leaky: (near) equal to cs, plus a little noise.
    leaky = frame["cs"].to_numpy().astype(float) + rng.normal(0, 0.01, size=len(frame))
    result = loho_single_feature_auc(pd.Series(leaky), frame["cs"], frame["facility_id"])
    assert result.flag
    assert max(result.mean_auc, 1 - result.mean_auc) > 0.85


def test_loho_auc_does_not_flag_pure_noise() -> None:
    frame = _synthetic_frame()
    rng = np.random.default_rng(1)
    noise = pd.Series(rng.normal(size=len(frame)))
    result = loho_single_feature_auc(noise, frame["cs"], frame["facility_id"])
    assert not result.flag


def test_loho_auc_categorical_leaky_variable() -> None:
    frame = _synthetic_frame()
    # A categorical variable that almost perfectly encodes cs.
    leaky_cat = np.where(frame["cs"] == 1, "cs_level", "vaginal_level")
    result = loho_single_feature_auc(pd.Series(leaky_cat), frame["cs"], frame["facility_id"])
    assert result.flag


def test_name_pattern_flags_catches_indication_and_variants() -> None:
    flags = name_pattern_flags(
        ["Indication of CS", "birth_weight", "maternal_age", "C-Section type", "residency"]
    )
    assert flags["Indication of CS"]
    assert flags["birth_weight"]
    assert flags["C-Section type"]
    assert not flags["maternal_age"]
    assert not flags["residency"]


def test_completeness_pattern_flags_a_cs_only_column() -> None:
    cs = pd.Series([1] * 400 + [0] * 600)
    # Recorded almost only for cs = 1 rows.
    values = pd.Series([1.0] * 380 + [np.nan] * 20 + [np.nan] * 590 + [1.0] * 10)
    result = completeness_pattern_flags(values, cs)
    assert result.flag
    assert result.share_cs1_of_nonmissing >= 0.9


def test_completeness_pattern_flags_a_balanced_column_is_not_flagged() -> None:
    cs = pd.Series([1] * 400 + [0] * 600)
    values = pd.Series([1.0] * 380 + [np.nan] * 20 + [1.0] * 570 + [np.nan] * 30)
    result = completeness_pattern_flags(values, cs)
    assert not result.flag


def test_run_screens_flags_all_three_kinds_and_writes_table(tmp_path: Path) -> None:
    frame = _synthetic_frame()
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 0.01, size=len(frame))
    frame["leaky_numeric"] = frame["cs"].to_numpy().astype(float) + noise
    frame["Indication of CS"] = rng.normal(size=len(frame))  # flagged by name only
    cs1_only = pd.Series(np.nan, index=frame.index)
    cs1_only.loc[frame["cs"] == 1] = 1.0
    frame["cs_only_field"] = cs1_only
    frame["harmless"] = rng.normal(size=len(frame))

    out_path = tmp_path / "reports" / "leakage" / "leakage_screens.csv"
    candidates = ["leaky_numeric", "Indication of CS", "cs_only_field", "harmless"]
    table = run_screens(frame, candidates, out_path)

    assert out_path.exists()
    by_name = table.set_index("variable")
    assert bool(by_name.loc["leaky_numeric", "auc_flag"])
    assert bool(by_name.loc["Indication of CS", "name_flag"])
    assert bool(by_name.loc["cs_only_field", "completeness_flag"])
    assert not bool(by_name.loc["harmless", "flagged"])
    assert set(table["variable"]) == set(candidates)


def test_run_screens_without_out_path_does_not_write(tmp_path: Path) -> None:
    frame = _synthetic_frame(200)
    frame["x"] = np.random.default_rng(3).normal(size=len(frame))
    table = run_screens(frame, ["x"])
    assert isinstance(table, pd.DataFrame)
    assert not (tmp_path / "reports").exists()
