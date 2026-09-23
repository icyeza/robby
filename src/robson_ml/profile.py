"""Data profile (spec §7). Every output is aggregate and small-cell suppressed."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score

from robson_engine import INPUT_FIELDS
from robson_ml.audit_offline import REPORT_SUPPRESSION, robson_report_table
from robson_ml.ingest import infer_kind
from robson_ml.mapping import MappingConfig
from robson_ml.populations import audit_population
from robson_ml.privacy import fmt_count, safe_pct, suppress_small_cells
from robson_ml.reporting import markdown_table, write_table

MIN_CLASS_COUNT = 10
MIN_N_FOR_QUANTILES = 20
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
CLINICIAN_PATTERN = re.compile(r"clinician|doctor|midwife|nurse|cadre|staff|provider|attend", re.I)
ANTENATAL_PATTERN = re.compile(r"\banc\b|antenatal|ante-natal|prenatal", re.I)
QUESTIONS = (
    "Are all six Robson inputs present, and how complete is each (overall and per facility)?",
    "How is onset of labour coded? Can planned pre-labour CS be distinguished from "
    "emergency pre-labour CS?",
    "For the PE and GDM fields, does a blank mean 'not recorded' or 'no'? Is there any "
    "recorded-negative value?",
    "Is there an attending clinician or cadre identifier?",
    "Are antenatal findings present in the delivery record, or structurally absent?",
    "Is `admitted_at` (or any admission timestamp) available at usable completeness?",
    "Which of the WHO C-Model's required variables exist in the data?",
    "Which facility is private, and which is the intended pilot facility?",
    "Are any records duplicated (for example twins recorded as two rows)? How is "
    "plurality represented?",
)
NOT_ANSWERED = "Not yet answered: needs human review at Checkpoint 1."
PROFILE_LINKED = {
    "n_nonnull": [*(f"p{int(q * 100)}" for q in QUANTILES), "association", "n_unique"],
}
PROFILE_COMPLEMENTS = {"n_nonnull": "n_rows"}


def _blank_to_none(value: object) -> object:
    return None if isinstance(value, str) and not value.strip() else value


def single_feature_auc(values: pd.Series, cs: pd.Series) -> float | None:
    """ROC AUC of one numeric variable against ``cs`` on rows where both are present."""
    both = values.notna() & cs.notna()
    y = cs[both].astype(int)
    if (y == 1).sum() < MIN_CLASS_COUNT or (y == 0).sum() < MIN_CLASS_COUNT:
        return None
    return float(roc_auc_score(y, values[both].astype(float)))


def cramers_v(values: pd.Series, cs: pd.Series) -> float | None:
    """Cramér's V between a categorical variable and ``cs`` (no continuity correction)."""
    both = values.notna() & cs.notna()
    table = pd.crosstab(values[both].astype(str), cs[both].astype(int))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None
    chi2 = chi2_contingency(table, correction=False)[0]
    n = int(table.to_numpy().sum())
    return float(math.sqrt(chi2 / (n * (min(table.shape) - 1))))


def variable_profile(
    raw: pd.DataFrame, canonical: pd.DataFrame, config: MappingConfig
) -> pd.DataFrame:
    """One row per raw variable (spec §7): names, kind, missingness overall and per facility,
    distinct values, quantiles, univariate association with ``cs``, proposed status."""
    raw = raw.reset_index(drop=True)
    canonical = canonical.reset_index(drop=True)
    facility = canonical["facility_id"].astype(str)
    facilities = sorted(facility.unique())
    cs = canonical["cs"]
    to_canonical: dict[str, list[str]] = {}
    for spec in config.fields.values():
        for column in spec.raw:
            to_canonical.setdefault(column, []).append(spec.canonical)
    rows = []
    for position in range(raw.shape[1]):
        column = raw.columns[position]
        values = raw.iloc[:, position].map(_blank_to_none)
        kind = infer_kind(values)
        row: dict[str, object] = {
            "raw_name": str(column),
            "canonical_name": ";".join(to_canonical.get(str(column), [])),
            "kind": kind,
            "n_rows": len(values),
            "n_nonnull": int(values.notna().sum()),
            "pct_missing": safe_pct(values.isna()),
        }
        for fac in facilities:
            row[f"pct_missing_{fac}"] = safe_pct(values[facility == fac].isna())
        row["n_unique"] = int(values.dropna().astype(str).nunique())
        numeric = pd.to_numeric(values, errors="coerce") if kind == "numeric" else None
        enough = numeric is not None and int(numeric.notna().sum()) >= MIN_N_FOR_QUANTILES
        for q in QUANTILES:
            row[f"p{int(q * 100)}"] = (
                float(numeric.quantile(q)) if enough and numeric is not None else None
            )
        if numeric is not None:
            row["association_metric"] = "auc"
            row["association"] = single_feature_auc(numeric, cs)
        elif kind == "categorical":
            row["association_metric"] = "cramers_v"
            row["association"] = cramers_v(values, cs)
        else:
            row["association_metric"] = ""
            row["association"] = None
        row["proposed_status"] = "review"
        rows.append(row)
    return pd.DataFrame(rows)


def robson_inputs_markdown(classified: pd.DataFrame) -> str:
    """Completeness of the six inputs, engine status distribution, Robson report table."""
    facilities = sorted(classified["facility_id"].astype(str).unique())
    completeness = []
    for field_name in INPUT_FIELDS:
        row: dict[str, object] = {
            "input": field_name,
            "all": safe_pct(classified[field_name].notna()),
        }
        for fac in facilities:
            in_fac = classified["facility_id"].astype(str) == fac
            row[fac] = safe_pct(classified.loc[in_fac, field_name].notna())
        completeness.append(row)
    status = (
        classified["robson_status"]
        .value_counts()
        .reindex(["resolved", "partial", "conflict"], fill_value=0)
    )
    status_table = pd.DataFrame(
        {
            "status": status.index,
            "n": status.to_numpy(),
            "pct": (100.0 * status.to_numpy() / max(len(classified), 1)).round(1),
        }
    )
    partial = classified.loc[classified["robson_status"] == "partial", "robson_resolving_fields"]
    resolving = partial.value_counts().rename_axis("resolving_fields").reset_index(name="n")
    audit, log = audit_population(classified)
    report = suppress_small_cells(robson_report_table(audit), **REPORT_SUPPRESSION)
    version = ", ".join(sorted(classified["rule_set_version"].astype(str).unique()))
    return "\n".join(
        [
            "# Robson inputs",
            "",
            f"Rule set: {version}. Records: {len(classified)}.",
            "",
            "## Completeness of the six inputs (% recorded)",
            "",
            markdown_table(pd.DataFrame(completeness)),
            "",
            "## Engine status",
            "",
            markdown_table(suppress_small_cells(status_table, ["n"], {"n": ["pct"]})),
            "",
            "### Partial records: fields that would resolve them",
            "",
            markdown_table(suppress_small_cells(resolving, ["n"])),
            "",
            f"## Robson report table (P_audit; {fmt_count(log.n_excluded)} rows excluded"
            " for missing outcome)",
            "",
            markdown_table(report),
            "",
        ]
    )


def _counts_table(series: pd.Series) -> str:
    labelled = series.astype(object).where(series.notna(), "(missing)").astype(str)
    table = labelled.value_counts().rename_axis("value").reset_index(name="n")
    return markdown_table(suppress_small_cells(table, ["n"]))


def _evidence(number: int, canonical: pd.DataFrame, raw: pd.DataFrame) -> str:
    if number == 1:
        rows = [{"input": f, "pct_recorded": safe_pct(canonical[f].notna())} for f in INPUT_FIELDS]
        return (
            markdown_table(pd.DataFrame(rows)) + "\n\nPer-facility completeness: robson_inputs.md."
        )
    if number == 2:
        prelabour = canonical[canonical["onset_of_labour"] == "prelabour_cs"]
        return (
            "Canonical onset categories:\n\n"
            + _counts_table(canonical["onset_of_labour"])
            + "\n\nPre-labour CS type among pre-labour CS rows:\n\n"
            + _counts_table(prelabour["prelabour_cs_type"])
        )
    if number == 3:
        return (
            "preeclampsia_recorded:\n\n"
            + _counts_table(canonical["preeclampsia_recorded"])
            + "\n\ngdm_recorded:\n\n"
            + _counts_table(canonical["gdm_recorded"])
        )
    if number in (4, 5):
        pattern = CLINICIAN_PATTERN if number == 4 else ANTENATAL_PATTERN
        hits = [i for i, c in enumerate(raw.columns) if pattern.search(str(c))]
        if not hits:
            return "No raw column names match the search pattern."
        rows = [
            {"raw_column": str(raw.columns[i]), "pct_recorded": safe_pct(raw.iloc[:, i].notna())}
            for i in hits
        ]
        return "Raw columns whose names match:\n\n" + markdown_table(pd.DataFrame(rows))
    if number == 6:
        admitted = canonical["admitted_at"]
        months = admitted.dt.strftime("%Y-%m")
        return (
            f"admitted_at recorded: {safe_pct(admitted.notna())}%.\n\nBy month:\n\n"
            + _counts_table(months)
        )
    if number == 7:
        return (
            "Not determinable until the C-Model variable list is transcribed into "
            "data/reference/cmodel_v1.yaml (spec §15.4)."
        )
    if number == 8:
        return "Not determinable from the data."
    plural = canonical[canonical["plurality"].fillna(1) >= 2]
    keys = ["facility_id", "admitted_at", "maternal_age", "parity"]
    sizes = plural.groupby(keys, dropna=False).size()
    return (
        "plurality:\n\n"
        + _counts_table(canonical["plurality"])
        + "\n\nGroups of plurality>=2 rows sharing facility, admission time, maternal age and "
        f"parity (possible one-row-per-baby): {fmt_count(int((sizes >= 2).sum()))}."
    )


def open_questions_markdown(
    canonical: pd.DataFrame, raw: pd.DataFrame, manual: Mapping[str, str]
) -> str:
    """Answers to spec §25: automatic evidence plus the human answer (or a flag if none)."""
    parts = ["# Open questions (spec §25)", ""]
    for number, question in enumerate(QUESTIONS, start=1):
        answer = manual.get(f"q{number}", NOT_ANSWERED)
        parts += [
            f"## Q{number}. {question}",
            "",
            _evidence(number, canonical, raw),
            "",
            f"**Answer:** {answer}",
            "",
        ]
    return "\n".join(parts)


def write_profile(
    raw: pd.DataFrame,
    classified: pd.DataFrame,
    config: MappingConfig,
    manual: Mapping[str, str],
    out_dir: Path,
) -> None:
    """Write the three §7 outputs under ``out_dir`` (reports/profile)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = variable_profile(raw, classified, config)
    # write_table's own suppression only checks n_nonnull's raw value; a small complement
    # (n_rows - n_nonnull) would also reveal a near-complete column, so that guard is
    # applied here first and write_table's pass over the already-suppressed values is a
    # no-op for the rows it already touched.
    profile = suppress_small_cells(
        profile, ["n_nonnull"], linked=PROFILE_LINKED, complements=PROFILE_COMPLEMENTS
    )
    write_table(
        profile,
        out_dir / "variable_profile.csv",
        ["n_nonnull"],
        PROFILE_LINKED,
    )
    (out_dir / "robson_inputs.md").write_text(robson_inputs_markdown(classified), encoding="utf-8")
    (out_dir / "open_questions.md").write_text(
        open_questions_markdown(classified, raw, manual), encoding="utf-8"
    )
