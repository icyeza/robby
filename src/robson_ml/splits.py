"""Split schemes S1-S5 for the evaluation harness (spec §11.1, v1.1 item 5, v1.2 items 5 and 8).

Every split is index-based: indices are row positions (``iloc``) into the frame passed in.
Splits are deterministic for a given seed. Rows sharing a ``mother_key`` never straddle the
fit, calibration and test sets, nor a tuning train/validation pair: a woman whose rows would
land on both sides of a boundary keeps her rows on the evaluation side, and her rows on the
training side are dropped (recorded in ``Fold.excluded_idx``). A missing ``mother_key`` makes
the row its own group.

The outcome ``cs`` is read only for the rows being stratified (the training pool); the
held-out rows' outcomes are never read by S1, S2, S4 or S5. S3 is ordinary stratified CV and
stratifies its outer folds on every row's outcome, as the spec defines it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

IndexArray = npt.NDArray[np.int64]
TuningFolds = list[tuple[IndexArray, IndexArray]]

# 1 of 5 grouped stratified folds = the 20% calibration split (§11.1).
CALIBRATION_SPLITS = 5
RECALIBRATION_N_FIRST = 150
INTERNAL_OUTER_SPLITS = 5
INTERNAL_INNER_SPLITS = 3
TEMPORAL_TRAIN_END = "2024-01-31"
_MISSING_KEY_PREFIX = "__row_"


class SplitIntegrityError(AssertionError):
    """A split broke disjointness or mother-level grouping (a bug, never a data issue)."""


def _empty() -> IndexArray:
    return np.array([], dtype=np.int64)


@dataclass(frozen=True, eq=False)
class Fold:
    """One evaluation fold; every array holds row positions into the split frame.

    ``fit_idx`` trains the model, ``calib_idx`` fits only the calibrator, ``test_idx`` is
    evaluated (empty for S5), ``tuning`` holds (train, validation) pairs drawn from
    ``fit_idx`` for hyperparameter search, and ``excluded_idx`` holds training-side rows
    dropped because their mother also has a row on the evaluation side.
    """

    name: str
    fit_idx: IndexArray
    calib_idx: IndexArray
    test_idx: IndexArray
    tuning: TuningFolds
    excluded_idx: IndexArray = field(default_factory=_empty)


def _as_index(values: npt.ArrayLike) -> IndexArray:
    return np.asarray(values, dtype=np.int64)


def _groups(df: pd.DataFrame) -> npt.NDArray[np.str_]:
    """Mother-level group labels; a row with no ``mother_key`` is its own group."""
    keys = df["mother_key"].astype(object).to_numpy()
    missing = pd.isna(keys)
    labels = np.where(missing, [f"{_MISSING_KEY_PREFIX}{i}" for i in range(len(df))], keys)
    return labels.astype(str)


def _outcome(df: pd.DataFrame, idx: IndexArray) -> npt.NDArray[np.int64]:
    """``cs`` for the rows in ``idx``; raises if any is missing (splits need P_audit rows)."""
    cs = df["cs"].iloc[idx]
    if cs.isna().any():
        raise ValueError("cs is missing for rows that must be stratified; filter to P_audit")
    return np.asarray(cs.astype(int).to_numpy(), dtype=np.int64)


def _strata(df: pd.DataFrame, idx: IndexArray) -> npt.NDArray[np.str_]:
    """Stratification labels ``cs x facility`` for the rows in ``idx``."""
    facility = df["facility_id"].astype(str).to_numpy(dtype=str)[idx]
    return np.char.add(np.char.add(_outcome(df, idx).astype(str), "|"), facility)


def _drop_shared(
    keep: IndexArray, against: IndexArray, groups: npt.NDArray[np.str_]
) -> tuple[IndexArray, IndexArray]:
    """Split ``keep`` into rows whose mother has no row in ``against`` and the rest."""
    shared = np.isin(groups[keep], np.unique(groups[against]))
    return keep[~shared], keep[shared]


def _grouped_stratified(df: pd.DataFrame, idx: IndexArray, n_splits: int, seed: int) -> TuningFolds:
    """StratifiedGroupKFold over ``idx`` by ``cs x facility``, grouped by mother."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    groups = _groups(df)[idx]
    return [
        (np.sort(idx[train]), np.sort(idx[val]))
        for train, val in splitter.split(np.zeros(len(idx)), _strata(df, idx), groups)
    ]


def _calibration_split(
    df: pd.DataFrame, idx: IndexArray, seed: int
) -> tuple[IndexArray, IndexArray]:
    """Carve 20% of ``idx`` as the calibration split (stratified cs x facility, grouped)."""
    fit, calib = _grouped_stratified(df, idx, CALIBRATION_SPLITS, seed)[0]
    return fit, calib


def _facility_tuning(df: pd.DataFrame, fit_idx: IndexArray) -> TuningFolds:
    """GroupKFold by facility with one fold per facility in ``fit_idx`` (sorted by name).

    A training row whose mother also has a row in the validation facility is dropped from
    that fold's training side.
    """
    facility = df["facility_id"].astype(str).to_numpy()[fit_idx]
    levels = sorted(set(facility))
    if len(levels) < 2:
        raise ValueError("facility-grouped tuning needs at least two facilities in the fit set")
    groups = _groups(df)
    folds: TuningFolds = []
    for level in levels:
        val = fit_idx[facility == level]
        train, _ = _drop_shared(fit_idx[facility != level], val, groups)
        folds.append((train, val))
    return folds


def _check_fold(fold: Fold, groups: npt.NDArray[np.str_]) -> Fold:
    """Raise SplitIntegrityError unless the fold is disjoint and mother-grouped."""
    parts = {
        "fit": fold.fit_idx,
        "calib": fold.calib_idx,
        "test": fold.test_idx,
        "excluded": fold.excluded_idx,
    }
    names = list(parts)
    for i, a in enumerate(names):
        if len(np.unique(parts[a])) != len(parts[a]):
            raise SplitIntegrityError(f"{fold.name}: duplicate rows in {a}")
        for b in names[i + 1 :]:
            if np.intersect1d(parts[a], parts[b]).size:
                raise SplitIntegrityError(f"{fold.name}: {a} and {b} share rows")
            if b != "excluded" and np.intersect1d(groups[parts[a]], groups[parts[b]]).size:
                raise SplitIntegrityError(f"{fold.name}: a mother straddles {a} and {b}")
    for train, val in fold.tuning:
        if not (np.isin(train, fold.fit_idx).all() and np.isin(val, fold.fit_idx).all()):
            raise SplitIntegrityError(f"{fold.name}: tuning rows outside the fit set")
        if np.intersect1d(groups[train], groups[val]).size:
            raise SplitIntegrityError(f"{fold.name}: a mother straddles a tuning fold")
    return fold


def loho_folds(df: pd.DataFrame, seed: int) -> list[Fold]:
    """S1 leave-one-hospital-out folds, one per facility in sorted order (spec §11.1).

    For held-out facility h: the pool is every other facility's rows, minus rows whose
    mother also has a row at h; 20% of the pool (stratified cs x facility, grouped by mother)
    is the calibration split; the rest is the fit set, tuned by GroupKFold over its
    facilities; the test set is all of h. Outcomes of h's rows are never read.
    """
    groups = _groups(df)
    facility = df["facility_id"].astype(str).to_numpy()
    folds = []
    for held_out in sorted(set(facility)):
        test = _as_index(np.flatnonzero(facility == held_out))
        pool, excluded = _drop_shared(_as_index(np.flatnonzero(facility != held_out)), test, groups)
        fit, calib = _calibration_split(df, pool, seed)
        fold = Fold(
            f"loho_{held_out}", fit, calib, test, _facility_tuning(df, fit), np.sort(excluded)
        )
        folds.append(_check_fold(fold, groups))
    return folds


def recalibration_split(
    test_idx: npt.ArrayLike, df: pd.DataFrame, n_first: int = RECALIBRATION_N_FIRST
) -> tuple[IndexArray, IndexArray]:
    """S2: split a held-out facility's rows into (recalibration, evaluation) by time.

    Rows are ordered by ``delivery_date``, ties by row position; the first ``n_first`` rows
    recalibrate and the rest are evaluated, both returned in that order. Evaluation rows
    whose mother has a row in the recalibration set are dropped. Outcomes are not read.
    """
    test = _as_index(test_idx)
    if len(test) <= n_first:
        raise ValueError(f"need more than {n_first} rows to recalibrate and evaluate")
    dates = df["delivery_date"].iloc[test]
    if dates.isna().any():
        raise ValueError("delivery_date is missing for held-out rows")
    ordered = test[np.lexsort((test, dates.to_numpy()))]
    recal, rest = ordered[:n_first], ordered[n_first:]
    groups = _groups(df)
    evaluate, _ = _drop_shared(rest, recal, groups)
    if np.intersect1d(groups[recal], groups[evaluate]).size:
        raise SplitIntegrityError("recalibration: a mother straddles recalibration and evaluation")
    return recal, evaluate


def internal_nested_folds(
    df: pd.DataFrame, seed: int, outer: int = INTERNAL_OUTER_SPLITS
) -> list[Fold]:
    """S3 internal nested CV (spec §11.1): reported only to size the internal-vs-LOHO gap.

    Outer StratifiedGroupKFold by cs x facility grouped by mother; inside each outer
    training set a 20% calibration split (same stratification and grouping) and an inner
    StratifiedGroupKFold(3) over the fit set for tuning.
    """
    groups = _groups(df)
    all_rows = _as_index(np.arange(len(df)))
    folds = []
    for k, (train, test) in enumerate(_grouped_stratified(df, all_rows, outer, seed)):
        fit, calib = _calibration_split(df, train, seed)
        tuning = _grouped_stratified(df, fit, INTERNAL_INNER_SPLITS, seed)
        folds.append(_check_fold(Fold(f"internal_{k}", fit, calib, test, tuning), groups))
    return folds


def temporal_split(df: pd.DataFrame, seed: int, train_end: str = TEMPORAL_TRAIN_END) -> Fold:
    """S4 temporal split (spec §11.1): train on ``delivery_date <= train_end``, test after.

    Within the training period the S1 inner structure applies (20% grouped stratified
    calibration split, facility-grouped tuning). Training rows whose mother also delivers
    in the test period are dropped. Test outcomes are never read.
    """
    dates = df["delivery_date"]
    if dates.isna().any():
        raise ValueError("delivery_date is missing; the temporal split needs every date")
    in_train = (dates.dt.normalize() <= pd.Timestamp(train_end)).to_numpy()
    test = _as_index(np.flatnonzero(~in_train))
    if len(test) == 0 or in_train.sum() == 0:
        raise ValueError(f"train_end {train_end} leaves an empty training or test period")
    groups = _groups(df)
    train, excluded = _drop_shared(_as_index(np.flatnonzero(in_train)), test, groups)
    fit, calib = _calibration_split(df, train, seed)
    fold = Fold("temporal", fit, calib, test, _facility_tuning(df, fit), np.sort(excluded))
    return _check_fold(fold, groups)


def deployment_split(df: pd.DataFrame, seed: int) -> Fold:
    """S5 deployment fit (spec §11.1, not an evaluation): all rows, 80/20 fit/calibration.

    The calibration split is stratified cs x facility and grouped by mother; tuning is
    GroupKFold by facility (one fold per facility). ``test_idx`` is empty.
    """
    fit, calib = _calibration_split(df, _as_index(np.arange(len(df))), seed)
    fold = Fold("deployment", fit, calib, _empty(), _facility_tuning(df, fit))
    return _check_fold(fold, _groups(df))
