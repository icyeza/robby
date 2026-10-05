"""Missingness and under-recording (RQ2).

Part A, fields that do have missingness (exact gestational age, anthropometry, parity and
the other inputs, raw-register fields such as living children): the missingness rate per
field by facility and by month, co-missingness patterns, and a missingness model (logistic
regression of ``is_missing`` on facility, month, parity and ``cs``). An association with
observed variables rules out MCAR; it is reported as *consistent with MAR*, never as
evidence for or against MNAR (:data:`MAR_STATEMENT`).

Part B, under-recording of ``preeclampsia_recorded`` and ``gdm_recorded``: both are
recorded yes/no with essentially no blanks, so the question is whether a recorded "no" is a
true "yes". A probabilistic misclassification (bias) analysis assumes perfect specificity
(a recorded "yes" is true) and a true prevalence pi taken from a grid anchored on published
prevalence (:func:`robson_ml.references.load_prevalence`; no default grid is ever
invented). The recording sensitivity is then ``se = observed / pi`` (optionally different
in CS and vaginal births, ``se_ratio = se_vaginal / se_cs``). Within each outcome stratum,
the expected number of true positives is ``recorded yes / se``, and each recorded "no" is
reclassified "yes" with the probability that restores it, over ``n_draws`` seeded draws.
For each pi it re-estimates (a) the adjusted OR of CS for the condition (adjusted for
facility and Robson level) and (b) Robson group CS rates by reclassified status, and it
reports the **tipping point**: the smallest pi at which a conclusion, defined explicitly by
the ``conclusions`` argument, differs from the conclusion on the recorded data.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm

from robson_ml.audit_offline import OVERALL
from robson_ml.casemix import dummy_block, robson_levels, separated_levels
from robson_ml.privacy import (
    SMALL_CELL_THRESHOLD,
    SUPPRESSED,
    SumRelation,
    TableSpec,
    suppress_tables,
)
from robson_ml.reporting import markdown_table

MAR_STATEMENT = (
    "Missingness associated with observed variables rules out MCAR (missing completely at "
    "random). The result is consistent with MAR (missing at random given the covariates "
    "modelled), but it is not evidence for or against MNAR: whether missingness depends on "
    "the unrecorded value itself cannot be tested from the observed data."
)
UNDER_RECORDING_ASSUMPTIONS = (
    "Assumptions: perfect specificity (a recorded 'yes' is a true 'yes'); the true prevalence "
    "is the assumed value pi; recording sensitivity is se = observed / pi, the same in every "
    "facility and Robson group, and in CS and vaginal births unless se_ratio (se_vaginal / "
    "se_cs) says otherwise. Results show how conclusions would move under these assumptions; "
    "they are not estimates of the true prevalence."
)
DEFAULT_MISSINGNESS_FIELDS = (
    "gestational_age_weeks",
    "height_cm",
    "weight_kg",
    "anc_contacts",
    "maternal_age",
    "parity",
    "previous_cs_count",
    "fetal_presentation",
    "plurality",
)
UNDER_RECORDED_FIELDS = ("preeclampsia_recorded", "gdm_recorded")
MONTH = "month"
MISSING_LEVEL = "(missing)"
OTHER_PATTERNS = "(other patterns)"
NO_MISSING_PATTERN = "(none missing)"
MIN_MODEL_EVENTS = 10
MAR_ALPHA = 0.05
DEFAULT_N_DRAWS = 200
YES, NO = "yes", "no"


def add_month(frame: pd.DataFrame, date_col: str = "delivery_date") -> pd.DataFrame:
    """``frame`` with a ``month`` column (YYYY-MM; missing date -> "(missing)")."""
    out = frame.copy()
    dates = pd.to_datetime(out[date_col], errors="coerce")
    out[MONTH] = dates.dt.strftime("%Y-%m").fillna(MISSING_LEVEL)
    return out


# ---------------------------------------------------------------- Part A: describe


def missingness_by(frame: pd.DataFrame, fields: Sequence[str], by: str) -> pd.DataFrame:
    """Per field: rows, missing count and % missing overall (ALL) and per level of ``by``."""
    rows = []
    levels = frame[by].astype(object).where(frame[by].notna(), MISSING_LEVEL).astype(str)
    for name in fields:
        missing = frame[name].isna()
        scopes = [(OVERALL, pd.Series(True, index=frame.index))] + [
            (level, levels == level) for level in sorted(levels.unique())
        ]
        for level, mask in scopes:
            n, n_missing = int(mask.sum()), int((missing & mask).sum())
            rows.append(
                {
                    "field": name,
                    by: level,
                    "n": n,
                    "n_missing": n_missing,
                    "pct_missing": 100.0 * n_missing / n if n else float("nan"),
                }
            )
    return pd.DataFrame(rows, columns=["field", by, "n", "n_missing", "pct_missing"])


def suppress_missingness(table: pd.DataFrame, by: str) -> pd.DataFrame:
    """Primary + secondary suppression: per field, the levels sum to the ALL row, and each
    level's size is the same across fields."""
    table = table.reset_index(drop=True)
    groups = []
    for _, cells in table.groupby("field", sort=False):
        members = tuple(cells.index[cells[by] != OVERALL])
        total = cells.index[cells[by] == OVERALL]
        if members and len(total):
            groups.append(SumRelation(members, total[0]))
    spec = TableSpec(
        table,
        ["n", "n_missing"],
        {"n": ["pct_missing"], "n_missing": ["pct_missing"]},
        {"n_missing": "n"},
        groups,
    )
    out = suppress_tables({"m": spec})["m"]
    out["pct_missing"] = out["pct_missing"].map(
        lambda v: v if isinstance(v, str) else round(float(v), 1)
    )
    return out


def comissingness_patterns(
    frame: pd.DataFrame, fields: Sequence[str], top: int = 10
) -> pd.DataFrame:
    """The most frequent missingness patterns (fields missing together), each with at least
    5 rows; every other pattern is pooled. Suppressed (the counts sum to the row total)."""
    missing = frame[list(fields)].isna()
    patterns = missing.apply(
        lambda row: (
            "+".join(f for f, m in zip(fields, row, strict=True) if m) or NO_MISSING_PATTERN
        ),
        axis=1,
    )
    counts = patterns.value_counts()
    shown = counts[counts >= SMALL_CELL_THRESHOLD].head(top)
    rest = int(counts.sum() - shown.sum())
    table = pd.DataFrame({"pattern": shown.index.astype(str), "n": shown.to_numpy()})
    if rest:
        table = pd.concat(
            [table, pd.DataFrame({"pattern": [OTHER_PATTERNS], "n": [rest]})], ignore_index=True
        )
    total = max(len(frame), 1)
    table["pct"] = 100.0 * table["n"] / total
    spec = TableSpec(table, ["n"], {"n": ["pct"]})
    out = suppress_tables({"p": spec})["p"]
    out["pct"] = out["pct"].map(lambda v: v if isinstance(v, str) else round(float(v), 1))
    return out


def _parity_band(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    band = numeric.clip(upper=3).map(lambda v: MISSING_LEVEL if pd.isna(v) else str(int(v)))
    return band.replace({"3": "3+"})


def missingness_models(
    frame: pd.DataFrame, fields: Sequence[str], min_events: int = MIN_MODEL_EVENTS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Logistic regression of ``is_missing_<field>`` on facility, month, parity and cs.

    Returns ``(summary, coefficients)``: per field the model size, the likelihood-ratio
    p-value against the intercept-only model and the MAR-worded interpretation; per term the
    odds ratio with its 95% CI. A field with fewer than ``min_events`` missing (or recorded)
    values is not modelled. ``frame`` needs ``month`` (:func:`add_month`).
    """
    summary, coefficients = [], []
    for name in fields:
        is_missing = frame[name].isna().astype(float)
        n_missing = int(is_missing.sum())
        base = {"field": name, "n": len(frame)}
        if min(n_missing, len(frame) - n_missing) < min_events:
            summary.append(
                {
                    **base,
                    "lr_p_value": float("nan"),
                    "mcar_rejected": pd.NA,
                    "interpretation": f"not modelled (fewer than {min_events} missing or "
                    "recorded values)",
                }
            )
            continue
        blocks = [
            pd.DataFrame({"const": 1.0}, index=frame.index),
            dummy_block(frame["facility_id"].astype(str), "facility"),
            dummy_block(frame[MONTH].astype(str), "month"),
            pd.DataFrame({"cs": frame["cs"].astype(float)}, index=frame.index),
        ]
        if name != "parity":
            blocks.append(dummy_block(_parity_band(frame["parity"]), "parity"))
        x = pd.concat(blocks, axis=1)
        x = x.loc[:, (x.nunique() > 1) | (x.columns == "const")]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = sm.Logit(is_missing, x).fit(disp=0, maxiter=200)
        except (np.linalg.LinAlgError, ValueError, OverflowError):
            summary.append(
                {
                    **base,
                    "lr_p_value": float("nan"),
                    "mcar_rejected": pd.NA,
                    "interpretation": "model could not be fitted",
                }
            )
            continue
        p_value = float(result.llr_pvalue)
        rejected = p_value < MAR_ALPHA
        summary.append(
            {
                **base,
                "lr_p_value": p_value,
                "mcar_rejected": rejected,
                "interpretation": (
                    "associated with observed variables: not MCAR; consistent with MAR given "
                    "these covariates; says nothing about MNAR"
                    if rejected
                    else "no association detected with these covariates (MCAR not rejected; "
                    "MNAR not assessable)"
                ),
            }
        )
        conf = result.conf_int()
        for term in x.columns:
            if term == "const":
                continue
            coefficients.append(
                {
                    "field": name,
                    "term": term,
                    "or": float(np.exp(result.params[term])),
                    "ci_low": float(np.exp(conf.loc[term, 0])),
                    "ci_high": float(np.exp(conf.loc[term, 1])),
                    "p_value": float(result.pvalues[term]),
                }
            )
    return pd.DataFrame(summary), pd.DataFrame(
        coefficients, columns=["field", "term", "or", "ci_low", "ci_high", "p_value"]
    )


# ---------------------------------------------------------------- Part B: under-recording


@dataclass(frozen=True)
class ORSummary:
    """The adjusted OR of CS for the condition in one scenario.

    ``or_median`` over draws; ``sys_*`` the 2.5/97.5 percentiles from reclassification
    alone; ``total_*`` adding random error (each draw's log OR plus a normal draw with its
    standard error). On the recorded data both intervals are the Wald 95% CI.
    """

    or_median: float
    sys_low: float
    sys_high: float
    total_low: float
    total_high: float
    n_draws_fitted: int


def or_point_above_one(summary: ORSummary) -> bool:
    """Conclusion: the adjusted OR of CS for the condition is above 1."""
    return bool(summary.or_median > 1.0)


def or_interval_excludes_one(summary: ORSummary) -> bool:
    """Conclusion: the 95% (total-uncertainty) interval of the adjusted OR excludes 1."""
    return bool(summary.total_low > 1.0 or summary.total_high < 1.0)


DEFAULT_CONCLUSIONS: Mapping[str, Callable[[ORSummary], bool]] = {
    "adjusted OR > 1": or_point_above_one,
    "adjusted OR 95% interval excludes 1": or_interval_excludes_one,
}


@dataclass(frozen=True)
class Reclassification:
    """Recording sensitivities and reclassification probabilities of one scenario."""

    feasible: bool
    reason: str = ""
    se_cs: float = float("nan")
    se_vaginal: float = float("nan")
    q_cs: float = float("nan")
    q_vaginal: float = float("nan")


def reclassification(
    n_yes_cs: int,
    n_cs: int,
    n_yes_vaginal: int,
    n_vaginal: int,
    prevalence: float,
    se_ratio: float = 1.0,
) -> Reclassification:
    """Sensitivities and the probability that a recorded "no" is a true "yes", per stratum.

    With specificity 1, the true positives in a stratum are ``recorded yes / se``; the
    stratum sensitivities satisfy ``se_vaginal = se_ratio * se_cs`` and make the true
    positives add up to ``prevalence * N``. Infeasible when a sensitivity would exceed 1
    (the assumed prevalence is below what is recorded) or nothing is recorded "yes".
    """
    total = n_cs + n_vaginal
    if n_yes_cs + n_yes_vaginal == 0:
        return Reclassification(False, "no recorded 'yes': sensitivity undefined")
    if not 0.0 < prevalence < 1.0 or se_ratio <= 0 or total == 0:
        return Reclassification(False, "invalid prevalence or se_ratio")
    target = prevalence * total
    se_cs = (n_yes_cs + n_yes_vaginal / se_ratio) / target
    se_vaginal = se_ratio * se_cs
    if se_cs > 1.0 + 1e-12 or se_vaginal > 1.0 + 1e-12:
        return Reclassification(
            False, "assumed prevalence below the recorded prevalence (se > 1)", se_cs, se_vaginal
        )
    true_cs, true_vaginal = n_yes_cs / se_cs, n_yes_vaginal / se_vaginal
    if true_cs > n_cs + 1e-9 or true_vaginal > n_vaginal + 1e-9:
        return Reclassification(False, "more true positives than women in a stratum")

    def q(true: float, yes: int, n: int) -> float:
        return 0.0 if n - yes == 0 else float(min(max((true - yes) / (n - yes), 0.0), 1.0))

    return Reclassification(
        True,
        "",
        se_cs,
        se_vaginal,
        q(true_cs, n_yes_cs, n_cs),
        q(true_vaginal, n_yes_vaginal, n_vaginal),
    )


def _fit_status(
    y: npt.NDArray[Any], base: npt.NDArray[Any], status: npt.NDArray[Any]
) -> tuple[float, float] | None:
    """Log OR and SE of ``status`` in a logistic model with the ``base`` design."""
    if status.min() == status.max():
        return None
    table = [(status == s) & (y == v) for s in (0, 1) for v in (0, 1)]
    if any(not t.any() for t in table):
        return None  # a zero cell: the OR is not estimable
    x = np.column_stack([status, base])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sm.Logit(y, x).fit(disp=0, maxiter=100)
    except (np.linalg.LinAlgError, ValueError, OverflowError):
        return None
    if not result.mle_retvals.get("converged", True):
        return None
    return float(result.params[0]), float(result.bse[0])


def _summarise(fits: list[tuple[float, float]], rng: np.random.Generator) -> ORSummary:
    if not fits:
        nan = float("nan")
        return ORSummary(nan, nan, nan, nan, nan, 0)
    b = np.array([f[0] for f in fits])
    se = np.array([f[1] for f in fits])
    total = b + rng.normal(0.0, 1.0, size=len(b)) * se
    return ORSummary(
        float(np.exp(np.median(b))),
        float(np.exp(np.percentile(b, 2.5))),
        float(np.exp(np.percentile(b, 97.5))),
        float(np.exp(np.percentile(total, 2.5))),
        float(np.exp(np.percentile(total, 97.5))),
        len(fits),
    )


def _group_counts(
    levels: npt.NDArray[Any], status: npt.NDArray[Any], y: npt.NDArray[Any]
) -> dict[tuple[str, str], tuple[float, float]]:
    out = {}
    for label in np.unique(levels):
        in_label = levels == label
        for name, value in ((YES, 1.0), (NO, 0.0)):
            mask = in_label & (status == value)
            out[(str(label), name)] = (float(mask.sum()), float(y[mask].sum()))
    return out


@dataclass
class UnderRecordingResult:
    """Scenario ORs, group CS rates by reclassified status, and tipping points."""

    condition: str
    field: str
    n_used: int
    n_not_recorded: int
    n_recorded_yes: int
    scenarios: pd.DataFrame
    group_rates: pd.DataFrame
    tipping: pd.DataFrame
    notes: list[str] = dataclasses.field(default_factory=list)

    def observed_pct(self) -> float | str:
        """Recorded prevalence (%), ``"<5"`` when fewer than 5 are recorded "yes"."""
        if 0 < self.n_recorded_yes < SMALL_CELL_THRESHOLD:
            return SUPPRESSED
        return 100.0 * self.n_recorded_yes / self.n_used if self.n_used else float("nan")

    def published(self) -> dict[str, pd.DataFrame]:
        scenarios = self.scenarios.copy()
        if 0 < self.n_recorded_yes < SMALL_CELL_THRESHOLD:
            for column in ("se_cs", "se_vaginal"):
                scenarios[column] = SUPPRESSED  # se = observed / pi would reveal the count
        return {
            "scenarios": scenarios,
            "group_rates": suppress_group_rates(self.group_rates),
            "tipping": self.tipping,
        }


def suppress_group_rates(table: pd.DataFrame) -> pd.DataFrame:
    """Suppress the (mean) group counts by status: within each scenario the statuses of a
    group sum to its size, and the groups of a status sum to a known total."""
    table = table.reset_index(drop=True)
    groups = []
    for _, cells in table.groupby(["scenario", "status"], sort=False):
        groups.append(SumRelation(tuple(cells.index)))
    for _, cells in table.groupby(["scenario", "robson_level"], sort=False):
        groups.append(SumRelation(tuple(cells.index)))
    spec = TableSpec(
        table, ["n", "n_cs"], {"n": ["n_cs", "cs_rate"], "n_cs": ["cs_rate"]}, {"n_cs": "n"}, groups
    )
    return suppress_tables({"g": spec})["g"]


def under_recording_sensitivity(
    audit: pd.DataFrame,
    field_name: str,
    grid_pct: Sequence[float],
    *,
    condition: str | None = None,
    n_draws: int = DEFAULT_N_DRAWS,
    seed: int = 0,
    se_ratios: Sequence[float] = (1.0,),
    conclusions: Mapping[str, Callable[[ORSummary], bool]] = DEFAULT_CONCLUSIONS,
) -> UnderRecordingResult:
    """Probabilistic reclassification of recorded "no" over the assumed prevalence grid.

    Args:
        audit: P_audit rows, classified (Robson level and facility adjust the OR).
        field_name: the yes/no field (e.g. ``preeclampsia_recorded``); rows where it is not
            recorded at all are left out and counted.
        grid_pct: assumed true prevalences (%), from the prevalence reference file only.
        n_draws: reclassification draws per scenario (seeded by ``seed``).
        se_ratios: ``se_vaginal / se_cs`` values (1 = non-differential recording).
        conclusions: the substantive conclusions, each a function of an :class:`ORSummary`
            returning True/False; the tipping point of each is the smallest assumed
            prevalence whose conclusion differs from the recorded data's. Default: "the
            adjusted OR > 1" and "its 95% interval excludes 1".
    """
    if audit["cs"].isna().any():
        raise ValueError("under_recording_sensitivity expects P_audit rows (cs non-missing)")
    condition = condition or field_name
    rng = np.random.default_rng(seed)
    recorded = audit[field_name].astype(object)
    used = recorded.notna()
    frame = audit[used].reset_index(drop=True)
    yes = recorded[used].astype(str).str.lower().eq(YES).to_numpy()
    y = frame["cs"].astype(float).to_numpy()
    levels = robson_levels(frame)
    separated = separated_levels(levels, frame["cs"])
    keep = ~levels.isin(separated).to_numpy()
    design = pd.concat(
        [
            pd.DataFrame({"const": 1.0}, index=frame.index),
            dummy_block(frame["facility_id"].astype(str), "facility"),
            dummy_block(levels, "robson"),
        ],
        axis=1,
    )[keep]
    design = design.loc[:, (design.nunique() > 1) | (design.columns == "const")]
    base = design.to_numpy(dtype=float)
    y_keep = y[keep]
    level_array = levels.astype(str).to_numpy()
    notes = []
    if separated:
        notes.append(
            f"Robson levels with no outcome variation left out of the OR model: {separated}"
        )

    counts = {
        "n_yes_cs": int((yes & (y == 1)).sum()),
        "n_cs": int((y == 1).sum()),
        "n_yes_vaginal": int((yes & (y == 0)).sum()),
        "n_vaginal": int((y == 0).sum()),
    }
    scenario_rows: list[dict[str, object]] = []
    rate_rows: list[dict[str, object]] = []
    status0 = yes.astype(float)
    baseline_fit = _fit_status(y_keep, base, status0[keep])
    if baseline_fit is None:
        nan = float("nan")
        baseline = ORSummary(nan, nan, nan, nan, nan, 0)
        notes.append("OR not estimable on the recorded data (a zero cell or no variation)")
    else:
        b, se = baseline_fit
        low, high = (
            float(np.exp(b - 1.959963984540054 * se)),
            float(np.exp(b + 1.959963984540054 * se)),
        )
        baseline = ORSummary(float(np.exp(b)), low, high, low, high, 1)
    baseline_conclusions = {
        name: (rule(baseline) if baseline.n_draws_fitted else None)
        for name, rule in conclusions.items()
    }

    def add_scenario(
        label: str, pi: float, ratio: float, rec: Reclassification, summary: ORSummary
    ) -> None:
        row: dict[str, object] = {
            "condition": condition,
            "scenario": label,
            "assumed_prevalence_pct": pi,
            "se_ratio": ratio,
            "feasible": rec.feasible,
            "reason": rec.reason,
            "se_cs": rec.se_cs,
            "se_vaginal": rec.se_vaginal,
            "or_median": summary.or_median,
            "sys_low": summary.sys_low,
            "sys_high": summary.sys_high,
            "total_low": summary.total_low,
            "total_high": summary.total_high,
            "n_draws_fitted": summary.n_draws_fitted,
        }
        for name, rule in conclusions.items():
            row[name] = rule(summary) if summary.n_draws_fitted else pd.NA
        scenario_rows.append(row)

    def add_rates(label: str, counts_by: dict[tuple[str, str], tuple[float, float]]) -> None:
        for (level, status), (n, n_cs) in sorted(counts_by.items()):
            rate_rows.append(
                {
                    "condition": condition,
                    "scenario": label,
                    "robson_level": level,
                    "status": status,
                    "n": round(n),
                    "n_cs": round(n_cs),
                    "cs_rate": n_cs / n if n else float("nan"),
                }
            )

    add_scenario(
        "recorded", float("nan"), 1.0, Reclassification(True, "", 1.0, 1.0, 0.0, 0.0), baseline
    )
    add_rates("recorded", _group_counts(level_array, status0, y))
    no_cs = (~yes) & (y == 1)
    no_vaginal = (~yes) & (y == 0)
    for ratio in se_ratios:
        for pi in grid_pct:
            label = f"pi={pi:g}%, se_ratio={ratio:g}"
            rec = reclassification(**counts, prevalence=pi / 100.0, se_ratio=ratio)
            if not rec.feasible:
                nan = float("nan")
                add_scenario(label, pi, ratio, rec, ORSummary(nan, nan, nan, nan, nan, 0))
                continue
            fits: list[tuple[float, float]] = []
            sums: dict[tuple[str, str], list[float]] = {}
            for _ in range(n_draws):
                draw = rng.random(len(y))
                flip = (no_cs & (draw < rec.q_cs)) | (no_vaginal & (draw < rec.q_vaginal))
                status = (yes | flip).astype(float)
                fit = _fit_status(y_keep, base, status[keep])
                if fit is not None:
                    fits.append(fit)
                for key, (n, n_cs) in _group_counts(level_array, status, y).items():
                    acc = sums.setdefault(key, [0.0, 0.0])
                    acc[0] += n
                    acc[1] += n_cs
            add_scenario(label, pi, ratio, rec, _summarise(fits, rng))
            add_rates(label, {k: (v[0] / n_draws, v[1] / n_draws) for k, v in sums.items()})

    scenarios = pd.DataFrame(scenario_rows)
    tipping_rows = []
    for ratio in se_ratios:
        subset = scenarios[
            (scenarios["scenario"] != "recorded")
            & (scenarios["se_ratio"] == ratio)
            & scenarios["feasible"].astype(bool)
        ].sort_values("assumed_prevalence_pct")
        for name in conclusions:
            base_value = baseline_conclusions[name]
            tip = float("nan")
            if base_value is not None:
                changed = subset[subset[name].notna() & (subset[name] != base_value)]
                if len(changed):
                    tip = float(changed["assumed_prevalence_pct"].iloc[0])
            if base_value is None:
                statement = "conclusion undefined on the recorded data (OR not estimable)"
            elif np.isnan(tip):
                grid = subset["assumed_prevalence_pct"]
                span = f"{grid.min():g}-{grid.max():g}%" if len(grid) else "no feasible value"
                statement = f"no change within the feasible grid ({span})"
            else:
                statement = f"changes at an assumed prevalence of {tip:g}%"
            tipping_rows.append(
                {
                    "condition": condition,
                    "se_ratio": ratio,
                    "conclusion": name,
                    "on_recorded_data": base_value,
                    "tipping_prevalence_pct": tip,
                    "statement": statement,
                }
            )
    return UnderRecordingResult(
        condition=condition,
        field=field_name,
        n_used=len(frame),
        n_not_recorded=int((~used).sum()),
        n_recorded_yes=int(yes.sum()),
        scenarios=scenarios,
        group_rates=pd.DataFrame(rate_rows),
        tipping=pd.DataFrame(tipping_rows),
        notes=notes,
    )


# ---------------------------------------------------------------- report


@dataclass
class MissingnessReport:
    """Everything ``robson-ml missingness`` publishes (already suppressed)."""

    by_facility: pd.DataFrame
    by_month: pd.DataFrame
    patterns: pd.DataFrame
    model_summary: pd.DataFrame
    model_coefficients: pd.DataFrame
    under_recording: list[UnderRecordingResult]
    under_recording_status: str
    fields: list[str]
    notes: list[str] = dataclasses.field(default_factory=list)

    def tables(self) -> dict[str, pd.DataFrame]:
        out = {
            "missingness_by_facility": self.by_facility,
            "missingness_by_month": self.by_month,
            "comissingness_patterns": self.patterns,
            "missingness_models": self.model_summary,
            "missingness_model_terms": self.model_coefficients,
        }
        for result in self.under_recording:
            for name, table in result.published().items():
                out[f"under_recording_{result.condition}_{name}"] = table
        return out

    def to_markdown(self) -> str:
        lines = [
            "# Missingness and under-recording (RQ2)",
            "",
            "## A. Fields with missing values",
            "",
            f"Fields: {', '.join(self.fields)}.",
            "",
            f"> {MAR_STATEMENT}",
            "",
            "### Missingness models (is_missing ~ facility + month + parity + cs)",
            "",
            markdown_table(self.model_summary),
            "",
            "### Missingness by facility",
            "",
            markdown_table(self.by_facility),
            "",
            "### Co-missingness patterns",
            "",
            markdown_table(self.patterns),
            "",
            "## B. Under-recording of pre-eclampsia and gestational diabetes",
            "",
            self.under_recording_status,
            "",
        ]
        if self.under_recording:
            lines += [f"> {UNDER_RECORDING_ASSUMPTIONS}", ""]
        for result in self.under_recording:
            tables = result.published()
            observed = result.observed_pct()
            shown = observed if isinstance(observed, str) else f"{observed:.2f}%"
            lines += [
                f"### {result.condition} (`{result.field}`)",
                "",
                f"Recorded prevalence: {shown}; rows not recorded (left out): "
                f"{result.n_not_recorded}.",
                *[f"- {note}" for note in result.notes],
                "",
                markdown_table(tables["scenarios"]),
                "",
                "Tipping points:",
                "",
                markdown_table(tables["tipping"]),
                "",
            ]
        lines += [*[f"- {note}" for note in self.notes], ""]
        return "\n".join(lines)

    def write(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, table in self.tables().items():
            table.to_csv(out_dir / f"{name}.csv", index=False)
        (out_dir / "missingness.md").write_text(self.to_markdown(), encoding="utf-8")


def missingness_report(
    audit: pd.DataFrame,
    fields: Sequence[str] = DEFAULT_MISSINGNESS_FIELDS,
    priors: Mapping[str, tuple[str, Sequence[float]]] | None = None,
    *,
    n_draws: int = DEFAULT_N_DRAWS,
    seed: int = 0,
    se_ratios: Sequence[float] = (1.0,),
    priors_absent_reason: str = "prevalence reference file absent",
) -> MissingnessReport:
    """Parts A and B on ``audit`` (P_audit, classified). ``priors`` maps a condition name
    to ``(field, grid_pct)`` from the prevalence reference; ``None`` skips Part B with
    ``priors_absent_reason`` (no grid is ever invented)."""
    frame = add_month(audit)
    present = [f for f in fields if f in frame.columns]
    notes = []
    if absent := [f for f in fields if f not in frame.columns]:
        notes.append(f"fields not in the data (skipped): {absent}")
    summary, terms = missingness_models(frame, present)
    results = []
    if priors is None:
        status = (
            f"Not run: {priors_absent_reason}. The assumed-prevalence grid must be anchored "
            "on published prevalence transcribed into data/reference/prevalence_v1.yaml "
            "(schema in robson_ml.references); it is never invented."
        )
    else:
        status = "Probabilistic reclassification of recorded 'no' over the assumed grid."
        for i, (condition, (field_name, grid)) in enumerate(sorted(priors.items())):
            if field_name not in frame.columns:
                notes.append(f"{condition}: field {field_name} not in the data")
                continue
            results.append(
                under_recording_sensitivity(
                    frame,
                    field_name,
                    grid,
                    condition=condition,
                    n_draws=n_draws,
                    seed=seed + i,
                    se_ratios=se_ratios,
                )
            )
    return MissingnessReport(
        by_facility=suppress_missingness(
            missingness_by(frame, present, "facility_id"), "facility_id"
        ),
        by_month=suppress_missingness(missingness_by(frame, present, MONTH), MONTH),
        patterns=comissingness_patterns(frame, present),
        model_summary=summary,
        model_coefficients=terms,
        under_recording=results,
        under_recording_status=status,
        fields=present,
        notes=notes,
    )
