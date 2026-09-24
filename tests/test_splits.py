import numpy as np
import pandas as pd
import pytest

from robson_ml.splits import (
    Fold,
    deployment_split,
    internal_nested_folds,
    loho_folds,
    recalibration_split,
    temporal_split,
)
from tests.synthetic import make_admissions

SEED = 7


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    frame = make_admissions(3000, seed=11)
    return frame[frame["cs"].notna()].reset_index(drop=True)


def _all_folds(df: pd.DataFrame) -> list[Fold]:
    return [
        *loho_folds(df, SEED),
        *internal_nested_folds(df, SEED),
        temporal_split(df, SEED),
        deployment_split(df, SEED),
    ]


def _keys(df: pd.DataFrame, idx: np.ndarray) -> set[str]:
    return set(df["mother_key"].iloc[idx].dropna())


def test_splits_no_leak(df: pd.DataFrame) -> None:
    """Spec §22: for S1 and S2 no held-out-facility row is in fit, tuning or calibration."""
    folds = loho_folds(df, SEED)
    assert sorted(f.name for f in folds) == [
        f"loho_{h}" for h in ["FAC_A", "FAC_B", "FAC_C", "FAC_D"]
    ]
    facility = df["facility_id"].to_numpy()
    for fold in folds:
        held_out = fold.name.removeprefix("loho_")
        assert set(facility[fold.test_idx]) == {held_out}
        assert np.array_equal(np.sort(fold.test_idx), np.flatnonzero(facility == held_out))
        training_rows = [fold.fit_idx, fold.calib_idx]
        for train_idx, val_idx in fold.tuning:
            training_rows += [train_idx, val_idx]
        for idx in training_rows:
            assert held_out not in set(facility[idx])
        recal_idx, eval_idx = recalibration_split(fold.test_idx, df)
        # S2 recalibrates and evaluates within the held-out facility only.
        assert set(facility[recal_idx]) == {held_out}
        assert set(facility[eval_idx]) == {held_out}
        assert not set(recal_idx) & set(eval_idx)
        for idx in training_rows:
            assert not set(idx) & set(recal_idx)
            assert not set(idx) & set(eval_idx)


def test_loho_structure(df: pd.DataFrame) -> None:
    for fold in loho_folds(df, SEED):
        pool = len(fold.fit_idx) + len(fold.calib_idx) + len(fold.excluded_idx)
        assert pool + len(fold.test_idx) == len(df)
        assert 0.15 < len(fold.calib_idx) / (len(fold.fit_idx) + len(fold.calib_idx)) < 0.25
        assert len(fold.tuning) == 3
        facility = df["facility_id"].to_numpy()
        for train_idx, val_idx in fold.tuning:
            assert len(set(facility[val_idx])) == 1
            assert set(facility[val_idx]).isdisjoint(set(facility[train_idx]))
            assert set(train_idx) | set(val_idx) <= set(fold.fit_idx)


def test_calibration_split_is_stratified(df: pd.DataFrame) -> None:
    fold = loho_folds(df, SEED)[0]
    cs = df["cs"].astype(int).to_numpy()
    assert abs(cs[fold.calib_idx].mean() - cs[fold.fit_idx].mean()) < 0.05
    facility = df["facility_id"].to_numpy()
    for fac in set(facility[fold.fit_idx]):
        share_fit = np.mean(facility[fold.fit_idx] == fac)
        share_calib = np.mean(facility[fold.calib_idx] == fac)
        assert abs(share_fit - share_calib) < 0.05


def test_loho_ignores_test_outcomes(df: pd.DataFrame) -> None:
    changed = df.copy()
    test_rows = changed["facility_id"] == "FAC_A"
    changed.loc[test_rows, "cs"] = 1 - changed.loc[test_rows, "cs"]
    for a, b in zip(loho_folds(df, SEED), loho_folds(changed, SEED), strict=True):
        if a.name == "loho_FAC_A":
            assert np.array_equal(a.fit_idx, b.fit_idx)
            assert np.array_equal(a.calib_idx, b.calib_idx)


def test_mother_key_grouping_holds_in_every_split(df: pd.DataFrame) -> None:
    shared = df["mother_key"].duplicated(keep=False)
    assert shared.sum() > 50  # the synthetic data really has shared keys
    for fold in _all_folds(df):
        parts = [fold.fit_idx, fold.calib_idx, fold.test_idx]
        for i, a in enumerate(parts):
            for b in parts[i + 1 :]:
                assert not set(a) & set(b)
                assert not _keys(df, a) & _keys(df, b)
        for train_idx, val_idx in fold.tuning:
            assert not set(train_idx) & set(val_idx)
            assert not _keys(df, train_idx) & _keys(df, val_idx)


def test_mother_across_facilities_is_excluded_from_training() -> None:
    frame = make_admissions(1500, seed=3)
    frame = frame[frame["cs"].notna()].reset_index(drop=True)
    a_row = int(np.flatnonzero(frame["facility_id"] == "FAC_A")[0])
    b_row = int(np.flatnonzero(frame["facility_id"] == "FAC_B")[0])
    frame.loc[b_row, "mother_key"] = frame.loc[a_row, "mother_key"]
    fold = next(f for f in loho_folds(frame, SEED) if f.name == "loho_FAC_A")
    assert b_row in set(fold.excluded_idx)
    assert b_row not in set(fold.fit_idx) | set(fold.calib_idx)
    fold_b = next(f for f in loho_folds(frame, SEED) if f.name == "loho_FAC_C")
    for train_idx, val_idx in fold_b.tuning:
        assert not _keys(frame, train_idx) & _keys(frame, val_idx)


def test_missing_mother_key_rows_are_their_own_group(df: pd.DataFrame) -> None:
    frame = df.copy()
    frame.loc[:200, "mother_key"] = None
    fold = deployment_split(frame, SEED)
    no_key = set(range(201))
    # Treated as one shared group they would all land on one side.
    assert no_key & set(fold.fit_idx)
    assert no_key & set(fold.calib_idx)
    assert len(fold.fit_idx) + len(fold.calib_idx) == len(frame)


def test_recalibration_split_orders_by_delivery_date(df: pd.DataFrame) -> None:
    fold = loho_folds(df, SEED)[1]
    recal_idx, eval_idx = recalibration_split(fold.test_idx, df)
    assert len(recal_idx) == 150
    dates = df["delivery_date"].to_numpy()
    assert dates[recal_idx].max() <= dates[eval_idx].min()
    expected = sorted(fold.test_idx, key=lambda i: (dates[i], i))[:150]
    assert list(recal_idx) == expected
    assert not _keys(df, recal_idx) & _keys(df, eval_idx)
    assert len(recal_idx) + len(eval_idx) <= len(fold.test_idx)


def test_recalibration_split_ties_by_row_order() -> None:
    frame = pd.DataFrame(
        {
            "mother_key": [f"M{i}" for i in range(6)],
            "delivery_date": pd.to_datetime(
                ["2024-01-02", "2024-01-01", "2024-01-02", "2024-01-01", "2024-01-03", "2024-01-02"]
            ),
        }
    )
    recal_idx, eval_idx = recalibration_split(np.array([5, 4, 3, 2, 1, 0]), frame, n_first=3)
    assert list(recal_idx) == [1, 3, 0]
    assert list(eval_idx) == [2, 5, 4]


def test_recalibration_split_needs_rows_to_evaluate(df: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        recalibration_split(np.arange(100), df, n_first=150)


def test_temporal_split_date_boundary(df: pd.DataFrame) -> None:
    fold = temporal_split(df, SEED, train_end="2024-01-31")
    dates = df["delivery_date"]
    boundary = pd.Timestamp("2024-01-31")
    train = np.concatenate([fold.fit_idx, fold.calib_idx, fold.excluded_idx])
    assert (dates.iloc[train] <= boundary).all()
    assert (dates.iloc[fold.test_idx] > boundary).all()
    assert len(train) + len(fold.test_idx) == len(df)
    assert np.array_equal(np.sort(fold.test_idx), np.flatnonzero(dates > boundary))
    assert len(fold.tuning) == 4


def test_internal_nested_folds_structure(df: pd.DataFrame) -> None:
    folds = internal_nested_folds(df, SEED)
    assert len(folds) == 5
    tests = np.concatenate([f.test_idx for f in folds])
    assert np.array_equal(np.sort(tests), np.arange(len(df)))
    for fold in folds:
        assert len(fold.tuning) == 3
        assert len(fold.fit_idx) + len(fold.calib_idx) + len(fold.test_idx) == len(df)


def test_deployment_split_structure(df: pd.DataFrame) -> None:
    fold = deployment_split(df, SEED)
    assert len(fold.test_idx) == 0
    assert len(fold.fit_idx) + len(fold.calib_idx) == len(df)
    assert 0.15 < len(fold.calib_idx) / len(df) < 0.25
    assert len(fold.tuning) == 4


def test_splits_are_deterministic(df: pd.DataFrame) -> None:
    for a, b in zip(_all_folds(df), _all_folds(df), strict=True):
        assert a.name == b.name
        for field in ("fit_idx", "calib_idx", "test_idx", "excluded_idx"):
            assert np.array_equal(getattr(a, field), getattr(b, field))
        for (ta, va), (tb, vb) in zip(a.tuning, b.tuning, strict=True):
            assert np.array_equal(ta, tb)
            assert np.array_equal(va, vb)
    other = deployment_split(df, SEED + 1)
    assert not np.array_equal(other.calib_idx, deployment_split(df, SEED).calib_idx)


def test_missing_outcome_in_training_rows_raises(df: pd.DataFrame) -> None:
    frame = df.copy()
    frame["cs"] = frame["cs"].astype("Int64")
    frame.loc[frame.index[frame["facility_id"] == "FAC_B"][0], "cs"] = pd.NA
    with pytest.raises(ValueError, match="cs"):
        deployment_split(frame, SEED)
