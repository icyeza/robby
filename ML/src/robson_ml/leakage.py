"""Leakage screens. These flag variables for human review; they never
auto-exclude anything -- exclusion only ever happens by hand in ``features_v1.yaml``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from robson_ml.reporting import write_table

AUC_FLAG_THRESHOLD = 0.85
COMPLETENESS_DIFF_THRESHOLD = 0.3
COMPLETENESS_SHARE_THRESHOLD = 0.9
TARGET_RATE_PRIOR_STRENGTH = 10.0
MISSING_LEVEL = "__missing__"

# Case-insensitive substrings suggesting post-admission or outcome-side
# information, plus patterns specific to this export. Matched against names normalised to
# lowercase with '_' and '-' turned into spaces, so "birth_weight", "birth-weight" and
# "Birth Weight" are all caught by one entry.
NAME_PATTERNS: tuple[str, ...] = (
    "indication",
    "theatre",
    "operation",
    "anaesth",
    "anesth",
    "incision",
    "apgar",
    "birth weight",
    "blood loss",
    "pph",
    "discharge",
    "length of stay",
    "delivery time",
    "time of delivery",
    "outcome",
    "neonatal",
    "newborn",
    "nicu",
    "c section",
    "surgeon",
    "spinal",
    "epidural",
    "transfusion",
    "death",
    "infant",
    "twin baby",
    "duration of labor",
    "oxytocin",
    "misoprostol",
    "instrumental",
    "episiotomy",
    "perineal",
    "placenta retention",
    "uterus atony",
    "hysterectomy",
    "fistula",
    "resuscitation",
)


def _normalize(name: str) -> str:
    return name.lower().replace("_", " ").replace("-", " ")


@dataclass(frozen=True)
class AucResult:
    """Leave-one-hospital-out single-feature AUC (screen 1)."""

    mean_auc: float
    """Mean AUC across held-out facilities; NaN if no facility fold could be scored."""
    flag: bool
    """``max(mean_auc, 1 - mean_auc) > 0.85``; False (never flagged) when NaN."""


@dataclass(frozen=True)
class CompletenessResult:
    """Completeness-vs-outcome pattern (screen 3)."""

    n_nonmissing_cs1: int
    n_cs1: int
    n_nonmissing_cs0: int
    n_cs0: int
    rate_cs1: float
    rate_cs0: float
    share_cs1_of_nonmissing: float
    flag: bool


def _numeric_fold_scores(
    train_values: pd.Series, train_cs: pd.Series, test_values: pd.Series
) -> np.ndarray | None:
    """Fit a logistic model on ``value`` + a missing indicator, median-imputed in-fold."""
    if train_cs.nunique() < 2:
        return None
    median = train_values.median()
    median = 0.0 if pd.isna(median) else float(median)
    train_imputed = train_values.fillna(median).to_numpy(dtype=float)
    train_missing = train_values.isna().to_numpy(dtype=float)
    test_imputed = test_values.fillna(median).to_numpy(dtype=float)
    test_missing = test_values.isna().to_numpy(dtype=float)
    train_x = np.column_stack([train_imputed, train_missing])
    test_x = np.column_stack([test_imputed, test_missing])
    model = LogisticRegression(max_iter=1000)
    model.fit(train_x, train_cs.to_numpy())
    return np.asarray(model.predict_proba(test_x)[:, 1])


def _categorical_fold_scores(
    train_values: pd.Series, train_cs: pd.Series, test_values: pd.Series
) -> np.ndarray | None:
    """Smoothed target rate per level (prior strength 10), fitted on the training fold."""
    if train_cs.empty:
        return None
    global_rate = float(train_cs.mean())
    train_labels = train_values.astype(object).where(train_values.notna(), MISSING_LEVEL)
    test_labels = test_values.astype(object).where(test_values.notna(), MISSING_LEVEL)
    stats = pd.DataFrame({"label": train_labels, "cs": train_cs.to_numpy()}).groupby("label")["cs"]
    smoothed = (stats.sum() + TARGET_RATE_PRIOR_STRENGTH * global_rate) / (
        stats.count() + TARGET_RATE_PRIOR_STRENGTH
    )
    return test_labels.map(smoothed).fillna(global_rate).to_numpy(dtype=float)


def loho_single_feature_auc(values: pd.Series, cs: pd.Series, facility: pd.Series) -> AucResult:
    """Mean leave-one-hospital-out AUC of one candidate variable alone.

    Numeric ``values`` are scored with a logistic model on the value plus a missing
    indicator (median-imputed inside each training fold). Non-numeric ``values`` are
    scored with a smoothed target rate fitted on each training fold. For each facility
    held out in turn, the model/encoding is fit on every other facility's rows and scored
    (AUC) on the held-out facility; the result is the mean of the per-facility AUCs.
    """
    frame = pd.DataFrame({"value": values, "cs": cs, "facility": facility}).dropna(subset=["cs"])
    numeric = pd.api.types.is_numeric_dtype(values)
    aucs: list[float] = []
    for facility_id in frame["facility"].dropna().unique():
        test_mask = frame["facility"] == facility_id
        train, test = frame[~test_mask], frame[test_mask]
        if test["cs"].nunique() < 2 or train.empty:
            continue
        if numeric:
            scores = _numeric_fold_scores(train["value"], train["cs"], test["value"])
        else:
            scores = _categorical_fold_scores(train["value"], train["cs"], test["value"])
        if scores is None:
            continue
        aucs.append(float(roc_auc_score(test["cs"].to_numpy(), scores)))
    if not aucs:
        return AucResult(float("nan"), False)
    mean_auc = float(np.mean(aucs))
    return AucResult(mean_auc, max(mean_auc, 1 - mean_auc) > AUC_FLAG_THRESHOLD)


def name_pattern_flags(names: Sequence[str]) -> dict[str, bool]:
    """Flag variable names matching a post-admission/outcome-side pattern."""
    flags: dict[str, bool] = {}
    for name in names:
        normalized = _normalize(name)
        flags[name] = any(pattern in normalized for pattern in NAME_PATTERNS)
    return flags


def completeness_pattern_flags(values: pd.Series, cs: pd.Series) -> CompletenessResult:
    """Flag a variable recorded only, or far more often, when ``cs = 1``."""
    is_cs1 = cs == 1
    is_cs0 = cs == 0
    nonmissing = values.notna()
    n_cs1, n_cs0 = int(is_cs1.sum()), int(is_cs0.sum())
    n_nonmissing_cs1 = int((nonmissing & is_cs1).sum())
    n_nonmissing_cs0 = int((nonmissing & is_cs0).sum())
    rate_cs1 = n_nonmissing_cs1 / n_cs1 if n_cs1 else float("nan")
    rate_cs0 = n_nonmissing_cs0 / n_cs0 if n_cs0 else float("nan")
    n_nonmissing = n_nonmissing_cs1 + n_nonmissing_cs0
    share_cs1 = n_nonmissing_cs1 / n_nonmissing if n_nonmissing else float("nan")
    diff = rate_cs1 - rate_cs0 if not (pd.isna(rate_cs1) or pd.isna(rate_cs0)) else float("nan")
    diff_flag = not pd.isna(diff) and diff > COMPLETENESS_DIFF_THRESHOLD
    share_flag = not pd.isna(share_cs1) and share_cs1 >= COMPLETENESS_SHARE_THRESHOLD
    return CompletenessResult(
        n_nonmissing_cs1=n_nonmissing_cs1,
        n_cs1=n_cs1,
        n_nonmissing_cs0=n_nonmissing_cs0,
        n_cs0=n_cs0,
        rate_cs1=rate_cs1,
        rate_cs0=rate_cs0,
        share_cs1_of_nonmissing=share_cs1,
        flag=diff_flag or share_flag,
    )


def run_screens(
    frame: pd.DataFrame, candidates: Sequence[str], out_path: Path | None = None
) -> pd.DataFrame:
    """Run all three leakage screens over ``candidates`` (columns of ``frame``, plus ``cs``
    and ``facility_id``). One row per variable; ``flagged`` is true if any screen fires.

    When ``out_path`` is given, the table is written there via
    :func:`robson_ml.reporting.write_table` (small-cell suppressed; AUCs are never counts
    of records, so they are published as-is).
    """
    cs = frame["cs"]
    facility = frame["facility_id"]
    name_flags = name_pattern_flags(list(candidates))
    rows = []
    for name in candidates:
        values = frame[name]
        auc = loho_single_feature_auc(values, cs, facility)
        completeness = completeness_pattern_flags(values, cs)
        rows.append(
            {
                "variable": name,
                "loho_auc": auc.mean_auc,
                "auc_flag": auc.flag,
                "name_flag": name_flags[name],
                "n_nonmissing_cs1": completeness.n_nonmissing_cs1,
                "n_cs1": completeness.n_cs1,
                "n_nonmissing_cs0": completeness.n_nonmissing_cs0,
                "n_cs0": completeness.n_cs0,
                "rate_cs1": completeness.rate_cs1,
                "rate_cs0": completeness.rate_cs0,
                "share_cs1_of_nonmissing": completeness.share_cs1_of_nonmissing,
                "completeness_flag": completeness.flag,
                "flagged": auc.flag or name_flags[name] or completeness.flag,
            }
        )
    table = pd.DataFrame(
        rows,
        columns=[
            "variable",
            "loho_auc",
            "auc_flag",
            "name_flag",
            "n_nonmissing_cs1",
            "n_cs1",
            "n_nonmissing_cs0",
            "n_cs0",
            "rate_cs1",
            "rate_cs0",
            "share_cs1_of_nonmissing",
            "completeness_flag",
            "flagged",
        ],
    )
    if out_path is None:
        return table
    return write_table(
        table,
        out_path,
        count_columns=["n_nonmissing_cs1", "n_cs1", "n_nonmissing_cs0", "n_cs0"],
        linked={
            "n_nonmissing_cs1": ["rate_cs1", "share_cs1_of_nonmissing"],
            "n_nonmissing_cs0": ["rate_cs0"],
        },
    )
