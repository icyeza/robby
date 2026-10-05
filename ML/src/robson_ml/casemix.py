"""Case-mix adjustment of facility CS rates (RQ1).

Run on ``P_audit`` with statsmodels ``Logit``:

* ``M_raw``: cs ~ C(facility)
* ``M_adj1``: cs ~ C(facility) + C(robson)
* ``M_adj2``: ``M_adj1`` + a maternal-age spline + the C-Model covariates available in the
  data (only when the C-Model reference file lists them; never guessed).

The Robson term uses the full classification (groups 1-10, the 6/7/9 non-cephalic row and the
residual as their own levels). Because onset is often coded retrospectively, the
models are repeated with groups 1+2 and 3+4 merged as a sensitivity analysis.

A Robson level with no outcome variation (every woman had a CS, or none did) is perfectly
separated: its coefficient diverges and, in the limit, its rows carry no information on the
facility contrasts. Such rows are left out of the adjusted models (counted and named), which
is what the maximum-likelihood fit converges to.

Outputs: facility odds ratios with 95% CIs per model; the % reduction in |log OR| from
``M_raw`` to each adjusted model, with bootstrap CIs (resampling within facility); observed /
expected ratios per facility against the Vogel-expected rate (sum of group size x reference
CS rate, on resolved records) and the C-Model expected rate (mean per-woman probability),
each with bootstrap CIs, or "not applicable" when the reference is absent or unusable.

These are descriptive adjustments for case mix, not causal effects. With four facilities,
one of them private, facility and sector are inseparable (:data:`SECTOR_CONFOUND_STATEMENT`,
embedded in every output).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import SplineTransformer

from robson_ml.audit_offline import GROUP_LABELS, OVERALL, row_labels
from robson_ml.cmodel import cmodel_applicability, cmodel_probability
from robson_ml.privacy import SumRelation, TableSpec, suppress_tables
from robson_ml.references import CModelReference, VogelReference
from robson_ml.reporting import markdown_table

SECTOR_CONFOUND_STATEMENT = (
    "Sector confound: with four facilities, one of them private, facility and sector "
    "(public vs private) cannot be separated. Every facility odds ratio and observed/expected "
    "ratio here is a joint facility-and-sector contrast; none can be attributed to sector, or "
    "to facility practice alone. Adjustment is for recorded case mix only (descriptive, not "
    "causal)."
)
NOT_APPLICABLE = "not applicable"
FULL = "full"
MERGED = "merged_1+2_3+4"
MERGE_MAP = {"1": "1+2", "2": "1+2", "3": "3+4", "4": "3+4"}
MODELS = ("M_raw", "M_adj1", "M_adj2")
DEFAULT_N_BOOT = 2000
AGE_COLUMN = "maternal_age"
SPLINE_KNOTS = 4
MISSING_LEVEL = "(missing)"
# Never adjusted for: outcome-side, outcome-contaminated (onset) or the exposure.
EXCLUDED_COVARIATES = frozenset(
    {
        "cs",
        "mode_of_delivery",
        "recorded_indication",
        "onset_of_labour",
        "prelabour_cs_type",
        "facility_id",
        AGE_COLUMN,
    }
)
MAX_CATEGORICAL_LEVELS = 10
MIN_COVARIATE_LEVEL_ROWS = 20
Z_95 = 1.959963984540054


def robson_levels(audit: pd.DataFrame, merged: bool = False) -> pd.Series:
    """The Robson level of each record: "1".."10", the 6/7/9 non-cephalic row or the
    residual; with ``merged``, groups 1+2 and 3+4 become one level each."""
    labels = row_labels(audit, split_non_cephalic=True)
    return labels.replace(MERGE_MAP) if merged else labels


def separated_levels(levels: pd.Series, cs: pd.Series) -> list[str]:
    """Levels whose records all had a CS, or none did (no outcome variation)."""
    rates = cs.astype(float).groupby(levels.to_numpy()).mean()
    return sorted(str(level) for level, rate in rates.items() if rate in (0.0, 1.0))


def dummy_block(values: pd.Series, prefix: str, reference: str | None = None) -> pd.DataFrame:
    values = values.astype(object).where(values.notna(), MISSING_LEVEL).astype(str)
    counts = values.value_counts()
    ref = reference if reference is not None and reference in counts.index else counts.index[0]
    levels = sorted(level for level in counts.index if level != ref)
    return pd.DataFrame(
        {f"{prefix}[{level}]": (values == level).astype(float) for level in levels},
        index=values.index,
    )


def covariate_block(values: pd.Series, prefix: str, outcome: pd.Series) -> pd.DataFrame:
    """Dummies of a categorical covariate whose sparse levels are pooled: a level with fewer
    than ``MIN_COVARIATE_LEVEL_ROWS`` rows, or with no outcome variation (which would not
    converge), joins the reference level. Pooling only coarsens the adjustment."""
    labels = values.astype(object).where(values.notna(), MISSING_LEVEL).astype(str)
    counts = labels.value_counts()
    reference = str(counts.index[0])
    rates = outcome.astype(float).groupby(labels.to_numpy()).mean()
    sparse = [
        level
        for level in counts.index
        if counts[level] < MIN_COVARIATE_LEVEL_ROWS or rates[level] in (0.0, 1.0)
    ]
    return dummy_block(labels.replace(dict.fromkeys(sparse, reference)), prefix, reference)


def drop_collinear(matrix: pd.DataFrame) -> pd.DataFrame:
    """Keep columns in order while each adds rank (the intercept and facility contrasts come
    first, so they are kept; e.g. a covariate level equal to a Robson level is dropped)."""
    kept: list[str] = []
    rank = 0
    for column in matrix.columns:
        trial = np.linalg.matrix_rank(matrix[[*kept, column]].to_numpy())
        if trial > rank:
            kept.append(column)
            rank = trial
    return matrix[kept]


def _numeric_block(values: pd.Series, name: str) -> pd.DataFrame:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    missing = numeric.isna()
    block = pd.DataFrame({name: numeric.fillna(numeric.median())}, index=values.index)
    if missing.any():
        block[f"{name}[missing]"] = missing.astype(float)
    return block


def _spline_block(values: pd.Series) -> pd.DataFrame:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    missing = numeric.isna()
    filled = numeric.fillna(numeric.median()).to_numpy().reshape(-1, 1)
    basis = SplineTransformer(
        n_knots=SPLINE_KNOTS, degree=3, knots="quantile", include_bias=False
    ).fit_transform(filled)
    block = pd.DataFrame(
        basis, index=values.index, columns=[f"age_spline[{i}]" for i in range(basis.shape[1])]
    )
    if missing.any():
        block["age[missing]"] = missing.astype(float)
    return block


def _is_continuous(values: pd.Series) -> bool:
    return bool(
        pd.api.types.is_numeric_dtype(values) and values.dropna().nunique() > MAX_CATEGORICAL_LEVELS
    )


def design_matrix(
    frame: pd.DataFrame,
    model: str,
    levels: pd.Series,
    reference_facility: str,
    covariates: Sequence[str] = (),
) -> pd.DataFrame:
    """The design matrix of ``model`` (intercept, facility contrasts vs
    ``reference_facility``, then the adjustment terms). Missing covariate values are coded
    with a missing indicator (numeric: median-filled; categorical: its own level); sparse
    covariate levels are pooled (:func:`covariate_block`) and collinear columns dropped."""
    blocks = [
        pd.DataFrame({"const": 1.0}, index=frame.index),
        dummy_block(frame["facility_id"].astype(str), "facility", reference_facility),
    ]
    if model in ("M_adj1", "M_adj2"):
        blocks.append(dummy_block(levels, "robson"))
    if model == "M_adj2":
        if AGE_COLUMN in frame.columns:
            blocks.append(_spline_block(frame[AGE_COLUMN]))
        for column in covariates:
            values = frame[column]
            if _is_continuous(values):
                blocks.append(_numeric_block(values, column))
            else:
                blocks.append(covariate_block(values, column, frame["cs"]))
    matrix = pd.concat(blocks, axis=1)
    varying = matrix.columns[(matrix.nunique() > 1) | (matrix.columns == "const")]
    return drop_collinear(matrix[varying])


@dataclass(frozen=True)
class FacilityFit:
    """Facility log odds ratios (vs the reference facility) and 95% CIs of one model."""

    n: int
    log_or: dict[str, float]
    ci_low: dict[str, float]
    ci_high: dict[str, float]


def fit_model(
    frame: pd.DataFrame,
    model: str,
    levels: pd.Series,
    reference_facility: str,
    covariates: Sequence[str] = (),
) -> FacilityFit | None:
    """Fit one model; ``None`` when it cannot be fitted (singular or non-convergent)."""
    y = frame["cs"].astype(float)
    x = design_matrix(frame, model, levels, reference_facility, covariates)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sm.Logit(y, x).fit(disp=0, maxiter=200)
    except (np.linalg.LinAlgError, ValueError, OverflowError):
        return None
    if not result.mle_retvals.get("converged", True):
        return None
    params, conf = result.params, result.conf_int()
    names = [c for c in x.columns if c.startswith("facility[")]
    key = {c: c[len("facility[") : -1] for c in names}
    return FacilityFit(
        n=len(frame),
        log_or={key[c]: float(params[c]) for c in names},
        ci_low={key[c]: float(conf.loc[c, 0]) for c in names},
        ci_high={key[c]: float(conf.loc[c, 1]) for c in names},
    )


@dataclass(frozen=True)
class ModelSet:
    """The three models of one Robson classification."""

    classification: str
    fits: dict[str, FacilityFit | None]
    separated: list[str]
    n_rows: int
    n_adjusted_rows: int


def fit_model_set(
    audit: pd.DataFrame,
    reference_facility: str,
    merged: bool = False,
    covariates: Sequence[str] = (),
) -> ModelSet:
    """``M_raw`` on every row; the adjusted models without the separated Robson levels."""
    levels = robson_levels(audit, merged)
    separated = separated_levels(levels, audit["cs"])
    keep = ~levels.isin(separated)
    fits: dict[str, FacilityFit | None] = {
        "M_raw": fit_model(audit, "M_raw", levels, reference_facility)
    }
    for model in ("M_adj1", "M_adj2"):
        fits[model] = fit_model(audit[keep], model, levels[keep], reference_facility, covariates)
    return ModelSet(MERGED if merged else FULL, fits, separated, len(audit), int(keep.sum()))


def pct_reduction(raw: float, adjusted: float) -> float:
    """Percentage reduction in |log OR| from ``raw`` to ``adjusted`` (NaN if raw is 0)."""
    if raw == 0 or not np.isfinite(raw) or not np.isfinite(adjusted):
        return float("nan")
    return 100.0 * (1.0 - abs(adjusted) / abs(raw))


def _reductions(fits: dict[str, FacilityFit | None]) -> dict[tuple[str, str], float]:
    raw = fits.get("M_raw")
    out: dict[tuple[str, str], float] = {}
    if raw is None:
        return out
    for model in ("M_adj1", "M_adj2"):
        adjusted = fits.get(model)
        if adjusted is None:
            continue
        for facility, value in raw.log_or.items():
            if facility in adjusted.log_or:
                out[(model, facility)] = pct_reduction(value, adjusted.log_or[facility])
    return out


def within_facility_resample(
    facility: pd.Series, rng: np.random.Generator
) -> npt.NDArray[np.int64]:
    """Positions of a bootstrap resample drawn with replacement within each facility."""
    positions = np.arange(len(facility))
    codes = facility.astype(str).to_numpy()
    parts = [
        rng.choice(positions[codes == f], size=int((codes == f).sum()), replace=True)
        for f in sorted(set(codes))
    ]
    return np.concatenate(parts) if parts else positions


def _oe_point(
    facility: npt.NDArray[Any],
    cs: npt.NDArray[Any],
    expected: npt.NDArray[Any],
    scopes: Sequence[str],
) -> dict[str, tuple[int, int, float, float]]:
    """Per scope: (n rows with an expected value, CS among them, observed, expected)."""
    has = ~np.isnan(expected)
    out = {}
    for scope in scopes:
        mask = has if scope == OVERALL else has & (facility == scope)
        n = int(mask.sum())
        out[scope] = (
            n,
            int(cs[mask].sum()),
            float(cs[mask].mean()) if n else float("nan"),
            float(expected[mask].mean()) if n else float("nan"),
        )
    return out


def vogel_expected(levels: pd.Series, vogel: VogelReference) -> pd.Series:
    """Per record, the reference CS rate of its Robson group (NaN when not resolved)."""
    rates = {label: vogel.cs_rate(int(label)) for label in GROUP_LABELS}
    return levels.map(rates).astype(float)


@dataclass
class CaseMixResult:
    """Case-mix outputs (unsuppressed counts; publish via :meth:`published`)."""

    odds_ratios: pd.DataFrame
    reductions: pd.DataFrame
    observed_expected: pd.DataFrame
    reference_facility: str
    covariates: list[str]
    separated: dict[str, list[str]]
    n_boot: int
    n_boot_effective: int
    vogel_status: str
    cmodel_status: str
    notes: list[str] = field(default_factory=list)
    statement: str = SECTOR_CONFOUND_STATEMENT

    def published(self) -> dict[str, pd.DataFrame]:
        """The tables safe to export (small-cell and secondary suppression on the O/E
        counts; the model tables hold estimates and model sizes only)."""
        return {
            "odds_ratios": self.odds_ratios,
            "reductions": self.reductions,
            "observed_expected": suppress_observed_expected(self.observed_expected),
        }

    def to_markdown(self) -> str:
        tables = self.published()
        lines = [
            "# Case-mix adjustment of facility CS rates (RQ1)",
            "",
            f"> {self.statement}",
            "",
            "Models (statsmodels Logit on P_audit): M_raw = cs ~ C(facility); M_adj1 = M_raw + "
            "C(Robson level); M_adj2 = M_adj1 + maternal-age spline (4 knots) + C-Model "
            "covariates available in the data. Facility odds ratios are against the reference "
            f"facility {self.reference_facility}.",
            "",
            f"- Robson classifications: `{FULL}` (primary: groups 1-10, the 6/7/9 non-cephalic "
            f"row and the residual as levels) and `{MERGED}` (sensitivity: groups 1+2 and 3+4 "
            "merged, since onset is often coded retrospectively).",
            "- Levels with no outcome variation (left out of adjusted models): "
            + "; ".join(f"{k}: {v or 'none'}" for k, v in self.separated.items()),
            f"- M_adj2 covariates beyond age: {self.covariates or 'none'}.",
            f"- Bootstrap: {self.n_boot} resamples within facility "
            f"({self.n_boot_effective} with all models fitted); percentile 95% CIs.",
            f"- Vogel-expected rate: {self.vogel_status}",
            f"- C-Model expected rate: {self.cmodel_status}",
            *[f"- {note}" for note in self.notes],
            "",
            "## Facility odds ratios",
            "",
            markdown_table(tables["odds_ratios"]),
            "",
            "## Reduction in |log OR| from M_raw",
            "",
            markdown_table(tables["reductions"]),
            "",
            "## Observed / expected per facility",
            "",
            markdown_table(tables["observed_expected"]),
            "",
            "`<5` = a count of 1-4 (or whose complement is 1-4); `*` = suppressed to protect "
            "another cell.",
            "",
        ]
        return "\n".join(lines)

    def write(self, out_dir: Path) -> None:
        """Write the published tables (CSV) and ``casemix.md`` under ``out_dir``."""
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, table in self.published().items():
            table.to_csv(out_dir / f"{name}.csv", index=False)
        (out_dir / "casemix.md").write_text(self.to_markdown(), encoding="utf-8")


OE_COUNTS = ["n", "n_resolved", "n_cs_resolved", "n_scored", "n_cs_scored"]
OE_LINKED = {
    "n_resolved": ["n_cs_resolved", "observed_rate_resolved", "vogel_oe"],
    "n_cs_resolved": ["observed_rate_resolved", "vogel_oe", "vogel_oe_low", "vogel_oe_high"],
    "n_scored": ["n_cs_scored", "observed_rate_scored", "cmodel_oe"],
    "n_cs_scored": ["observed_rate_scored", "cmodel_oe", "cmodel_oe_low", "cmodel_oe_high"],
}
OE_COMPLEMENTS = {"n_cs_resolved": "n_resolved", "n_cs_scored": "n_scored", "n_resolved": "n"}


def suppress_observed_expected(table: pd.DataFrame) -> pd.DataFrame:
    """Primary and secondary suppression of the O/E counts: facility rows sum to ALL."""
    table = table.reset_index(drop=True)
    counts = [c for c in OE_COUNTS if pd.api.types.is_numeric_dtype(table[c])]
    linked = {k: [c for c in v if c in table.columns] for k, v in OE_LINKED.items() if k in counts}
    complements = {k: v for k, v in OE_COMPLEMENTS.items() if k in counts and v in counts}
    members = tuple(table.index[table["facility"] != OVERALL])
    total = table.index[table["facility"] == OVERALL]
    groups = [SumRelation(members, total[0])] if len(total) else []
    spec = TableSpec(table, counts, linked, complements, groups)
    return suppress_tables({"oe": spec})["oe"]


def cmodel_covariates(frame: pd.DataFrame, cmodel: CModelReference | None) -> list[str]:
    """C-Model variables' columns present in ``frame`` and allowed as covariates."""
    if cmodel is None:
        return []
    return [c for c in cmodel.columns() if c in frame.columns and c not in EXCLUDED_COVARIATES]


def casemix_analysis(
    audit: pd.DataFrame,
    vogel: VogelReference | None = None,
    cmodel: CModelReference | None = None,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    reference_facility: str | None = None,
    bootstrap_models: bool = True,
    vogel_absent_reason: str = "reference file absent",
    cmodel_absent_reason: str = "reference file absent",
) -> CaseMixResult:
    """Run the RQ1 case-mix analysis on ``audit`` (P_audit, classified).

    ``n_boot`` resamples within facility give the O/E CIs and, with ``bootstrap_models``,
    the CIs of the |log OR| reduction (full classification). ``reference_facility``
    defaults to the first facility in sorted order.
    """
    if audit["cs"].isna().any():
        raise ValueError("casemix_analysis expects P_audit rows (cs non-missing)")
    audit = audit.reset_index(drop=True)
    facilities = sorted(audit["facility_id"].astype(str).unique())
    reference = reference_facility or facilities[0]
    if reference not in facilities:
        raise ValueError("reference_facility is not one of the facilities")
    covariates = cmodel_covariates(audit, cmodel)
    notes: list[str] = []

    sets = [
        fit_model_set(audit, reference, merged=False, covariates=covariates),
        fit_model_set(audit, reference, merged=True, covariates=covariates),
    ]
    or_rows = []
    for model_set in sets:
        for model in MODELS:
            fit = model_set.fits[model]
            if fit is None:
                notes.append(f"{model_set.classification} {model}: could not be fitted")
                continue
            for facility in sorted(fit.log_or):
                b = fit.log_or[facility]
                or_rows.append(
                    {
                        "classification": model_set.classification,
                        "model": model,
                        "facility": facility,
                        "n_model": fit.n,
                        "log_or": b,
                        "or": float(np.exp(b)),
                        "ci_low": float(np.exp(fit.ci_low[facility])),
                        "ci_high": float(np.exp(fit.ci_high[facility])),
                    }
                )
    odds_ratios = pd.DataFrame(or_rows)

    # Bootstrap (within facility).
    rng = np.random.default_rng(seed)
    levels = robson_levels(audit)
    facility_codes = audit["facility_id"].astype(str).to_numpy()
    cs = audit["cs"].astype(float).to_numpy()
    scopes = [OVERALL, *facilities]
    expected: dict[str, npt.NDArray[Any] | None] = {"vogel": None, "cmodel": None}
    vogel_status = f"{NOT_APPLICABLE} ({vogel_absent_reason})"
    if vogel is not None:
        expected["vogel"] = vogel_expected(levels, vogel).to_numpy()
        vogel_status = (
            f"sum of group size x reference CS rate, resolved records only ({vogel.citation}, "
            f"{vogel.source_table}, {vogel.population})"
        )
    cmodel_status = f"{NOT_APPLICABLE} ({cmodel_absent_reason})"
    if cmodel is not None:
        applicability = cmodel_applicability(audit, cmodel)
        if applicability.applicable:
            expected["cmodel"] = cmodel_probability(audit, cmodel).to_numpy()
            cmodel_status = f"mean per-woman probability ({cmodel.citation})"
        else:
            cmodel_status = applicability.statement()
    boot_oe: dict[str, dict[str, list[float]]] = {
        k: {s: [] for s in scopes} for k, v in expected.items() if v is not None
    }
    boot_reduction: dict[tuple[str, str], list[float]] = {}
    effective = 0
    for _ in range(n_boot):
        idx = within_facility_resample(audit["facility_id"], rng)
        for kind, values in expected.items():
            if values is None:
                continue
            point = _oe_point(facility_codes[idx], cs[idx], values[idx], scopes)
            for scope, (_, _, obs, exp) in point.items():
                boot_oe[kind][scope].append(obs / exp if exp else float("nan"))
        if bootstrap_models:
            sample = audit.iloc[idx].reset_index(drop=True)
            model_set = fit_model_set(sample, reference, merged=False, covariates=covariates)
            if all(model_set.fits[m] is not None for m in MODELS):
                effective += 1
                for key, value in _reductions(model_set.fits).items():
                    boot_reduction.setdefault(key, []).append(value)
    if not bootstrap_models:
        effective = n_boot

    reduction_rows = []
    for model_set in sets:
        for (model, facility), value in sorted(_reductions(model_set.fits).items()):
            red_row: dict[str, object] = {
                "classification": model_set.classification,
                "model": model,
                "facility": facility,
                "pct_reduction_abs_log_or": value,
                "ci_low": float("nan"),
                "ci_high": float("nan"),
            }
            red_draws = boot_reduction.get((model, facility), [])
            if model_set.classification == FULL and red_draws:
                red_finite = np.asarray(red_draws)[np.isfinite(red_draws)]
                if len(red_finite):
                    red_row["ci_low"], red_row["ci_high"] = (
                        float(v) for v in np.percentile(red_finite, [2.5, 97.5])
                    )
            reduction_rows.append(red_row)
    reductions = pd.DataFrame(reduction_rows)

    oe_rows = []
    points = {
        kind: _oe_point(facility_codes, cs, values, scopes)
        for kind, values in expected.items()
        if values is not None
    }
    resolved = levels.isin(GROUP_LABELS).to_numpy()
    for scope in scopes:
        in_scope = np.ones(len(audit), bool) if scope == OVERALL else facility_codes == scope
        row: dict[str, object] = {
            "facility": scope,
            "n": int(in_scope.sum()),
            "n_resolved": int((in_scope & resolved).sum()),
            "n_cs_resolved": int(cs[in_scope & resolved].sum()),
            "observed_rate_resolved": (
                float(cs[in_scope & resolved].mean()) if (in_scope & resolved).any() else np.nan
            ),
        }
        for kind, prefix in (("vogel", "vogel"), ("cmodel", "cmodel")):
            if kind not in points:
                for suffix in ("expected_rate", "oe", "oe_low", "oe_high"):
                    row[f"{prefix}_{suffix}"] = NOT_APPLICABLE
                if kind == "cmodel":
                    row["n_scored"] = NOT_APPLICABLE
                    row["n_cs_scored"] = NOT_APPLICABLE
                    row["observed_rate_scored"] = NOT_APPLICABLE
                continue
            n_has, n_cs, obs, exp = points[kind][scope]
            if kind == "cmodel":
                row["n_scored"], row["n_cs_scored"], row["observed_rate_scored"] = n_has, n_cs, obs
            row[f"{prefix}_expected_rate"] = exp
            row[f"{prefix}_oe"] = obs / exp if exp else float("nan")
            draws = np.asarray(boot_oe[kind][scope], dtype=float)
            finite = draws[np.isfinite(draws)]
            low, high = np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan)
            row[f"{prefix}_oe_low"], row[f"{prefix}_oe_high"] = float(low), float(high)
        oe_rows.append(row)
    columns = [
        "facility",
        "n",
        "n_resolved",
        "n_cs_resolved",
        "observed_rate_resolved",
        "vogel_expected_rate",
        "vogel_oe",
        "vogel_oe_low",
        "vogel_oe_high",
        "n_scored",
        "n_cs_scored",
        "observed_rate_scored",
        "cmodel_expected_rate",
        "cmodel_oe",
        "cmodel_oe_low",
        "cmodel_oe_high",
    ]
    observed_expected = pd.DataFrame(oe_rows)[columns]
    if cmodel is None or not covariates:
        notes.append(
            "M_adj2 adds no C-Model covariate: "
            + ("the C-Model reference file is absent." if cmodel is None else "none available.")
        )
    return CaseMixResult(
        odds_ratios=odds_ratios,
        reductions=reductions,
        observed_expected=observed_expected,
        reference_facility=reference,
        covariates=covariates,
        separated={s.classification: s.separated for s in sets},
        n_boot=n_boot,
        n_boot_effective=effective,
        vogel_status=vogel_status,
        cmodel_status=cmodel_status,
        notes=notes,
    )
