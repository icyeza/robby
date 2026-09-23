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
from robson_ml.audit_offline import robson_report_table, suppress_report_table
from robson_ml.ingest import infer_kind
from robson_ml.mapping import MappingConfig
from robson_ml.populations import audit_population
from robson_ml.privacy import (
    SMALL_CELL_THRESHOLD,
    SUPPRESSED,
    SumRelation,
    fmt_count,
    safe_pct,
    suppress_small_cells,
    suppress_table,
)
from robson_ml.reporting import markdown_table, write_table

MIN_CLASS_COUNT = 10
# A quantile is released only with at least this many values at or below it and at or
# above it (DECISIONS.md 2026-09-23), so it never pins a few extreme values.
QUANTILE_MIN_SIDE = 5
RARE_LEVEL = "(rare)"
MISSING_LABEL = "(missing)"
OVERALL_SCOPE = "all"
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
    """Cramér's V between a categorical variable and ``cs`` (no continuity correction).

    Like the AUC, needs ``MIN_CLASS_COUNT`` rows in each ``cs`` class. Levels with fewer
    than 5 rows are pooled into one ``"(rare)"`` level; if that pool itself has fewer than
    5 rows it is left out. ``None`` if fewer than 2 levels remain.
    """
    both = values.notna() & cs.notna()
    labels = values[both].astype(str)
    sizes = labels.value_counts()
    labels = labels.where(~labels.isin(sizes.index[sizes < SMALL_CELL_THRESHOLD]), RARE_LEVEL)
    if (labels == RARE_LEVEL).sum() < SMALL_CELL_THRESHOLD:
        labels = labels[labels != RARE_LEVEL]
    y = cs[labels.index].astype(int)
    if (y == 1).sum() < MIN_CLASS_COUNT or (y == 0).sum() < MIN_CLASS_COUNT:
        return None
    table = pd.crosstab(labels, y)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None
    chi2 = chi2_contingency(table, correction=False)[0]
    n = int(table.to_numpy().sum())
    return float(math.sqrt(chi2 / (n * (min(table.shape) - 1))))


def pct_by_facility(
    mask: pd.Series, facility: pd.Series, facilities: list[str]
) -> dict[str, float | str]:
    """% of ``True`` in ``mask`` overall (key ``"all"``) and per facility, suppressed.

    The per-facility counts sum to the overall count, which is published, so besides the
    primary rule (count or complement of 1-4) a lone hidden facility cell would be
    recoverable by subtraction; secondary suppression hides another one
    (``privacy.suppress_table``).
    """
    scopes = [mask, *(mask[facility == fac] for fac in facilities)]
    n_trues = [int(s.sum()) for s in scopes]
    n_rows_list = [len(s) for s in scopes]
    counts = pd.DataFrame({"n_true": n_trues, "n_rows": n_rows_list})
    safe = suppress_table(
        counts,
        ["n_true"],
        complements={"n_true": "n_rows"},
        groups=[SumRelation(tuple(range(1, len(scopes))), total=0)],
    )
    out: dict[str, float | str] = {}
    for position, key in enumerate([OVERALL_SCOPE, *facilities]):
        n_true, n_rows = n_trues[position], n_rows_list[position]
        if safe.at[position, "n_true"] == SUPPRESSED:
            out[key] = SUPPRESSED
        else:
            out[key] = round(100.0 * n_true / n_rows, 1) if n_rows else float("nan")
    return out


def _released_quantiles(numeric: pd.Series | None) -> dict[str, float | None]:
    """Quantiles with at least ``QUANTILE_MIN_SIDE`` values on each side; else ``None``."""
    out: dict[str, float | None] = {}
    for q in QUANTILES:
        key = f"p{int(q * 100)}"
        out[key] = None
        if numeric is None or not numeric.notna().any():
            continue
        value = float(numeric.quantile(q))
        below, above = int((numeric <= value).sum()), int((numeric >= value).sum())
        if below >= QUANTILE_MIN_SIDE and above >= QUANTILE_MIN_SIDE:
            out[key] = value
    return out


def variable_profile(
    raw: pd.DataFrame, canonical: pd.DataFrame, config: MappingConfig
) -> pd.DataFrame:
    """One row per raw variable (spec §7): position, names, kind, missingness overall and per
    facility, distinct values, quantiles, univariate association with ``cs``, proposed
    status.

    ``raw`` and ``canonical`` must be row-aligned (same records, same order).
    """
    if len(raw) != len(canonical):
        raise ValueError(
            f"variable_profile needs row-aligned frames: raw has {len(raw)} rows, "
            f"canonical has {len(canonical)}"
        )
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
        missing = pct_by_facility(values.isna(), facility, facilities)
        row: dict[str, object] = {
            "raw_position": position,
            "raw_name": str(column),
            "canonical_name": ";".join(to_canonical.get(str(column), [])),
            "kind": kind,
            "n_rows": len(values),
            "n_nonnull": int(values.notna().sum()),
            "pct_missing": missing[OVERALL_SCOPE],
        }
        for fac in facilities:
            row[f"pct_missing_{fac}"] = missing[fac]
        row["n_unique"] = int(values.dropna().astype(str).nunique())
        numeric = pd.to_numeric(values, errors="coerce") if kind == "numeric" else None
        row.update(_released_quantiles(numeric))
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
        if row["pct_missing"] == SUPPRESSED:
            # n_nonnull reveals the same count as pct_missing: hide it along with what
            # PROFILE_LINKED derives from it (write_profile repeats this for n_nonnull 1-4).
            for name in ("n_nonnull", *PROFILE_LINKED["n_nonnull"]):
                row[name] = SUPPRESSED
        rows.append(row)
    return pd.DataFrame(rows)


def input_completeness(classified: pd.DataFrame) -> pd.DataFrame:
    """% recorded for each of the six Robson inputs, overall (``all``) and per facility,
    with primary and secondary suppression across the facility cells of each row."""
    facility = classified["facility_id"].astype(str)
    facilities = sorted(facility.unique())
    rows = []
    for field_name in INPUT_FIELDS:
        pct = pct_by_facility(classified[field_name].notna(), facility, facilities)
        rows.append({"input": field_name, **pct})
    return pd.DataFrame(rows, columns=["input", OVERALL_SCOPE, *facilities])


def robson_inputs_markdown(classified: pd.DataFrame) -> str:
    """Completeness of the six inputs, engine status distribution, Robson report table."""
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
    report = suppress_report_table(robson_report_table(audit))
    version = ", ".join(sorted(classified["rule_set_version"].astype(str).unique()))
    return "\n".join(
        [
            "# Robson inputs",
            "",
            f"Rule set: {version}. Records: {len(classified)}.",
            "",
            "## Completeness of the six inputs (% recorded)",
            "",
            markdown_table(input_completeness(classified)),
            "",
            "## Engine status",
            "",
            markdown_table(suppress_table(status_table, ["n"], {"n": ["pct"]})),
            "",
            "### Partial records: fields that would resolve them",
            "",
            # Its rows sum to the partial count of the status table.
            markdown_table(suppress_table(resolving, ["n"])),
            "",
            "## Robson report table (P_audit)",
            "",
            f"{log.n_excluded} rows excluded for missing outcome (reported exactly; "
            "data-quality count, spec §4.2).",
            "",
            "pct_of_deliveries, abs_contribution and rel_contribution are given to 2 decimals.",
            "",
            markdown_table(report),
            "",
        ]
    )


def _counts_table(series: pd.Series) -> str:
    """Level counts, suppressed. The levels sum to the published number of rows, and the
    recorded levels to the non-missing count (published for the raw column by the variable
    profile), so both groups are protected from subtraction."""
    labelled = series.astype(object).where(series.notna(), MISSING_LABEL).astype(str)
    table = labelled.value_counts().rename_axis("value").reset_index(name="n")
    groups = [SumRelation(tuple(table.index))]
    recorded = tuple(table.index[table["value"] != MISSING_LABEL])
    if 2 <= len(recorded) < len(table):
        groups.append(SumRelation(recorded))
    return markdown_table(suppress_table(table, ["n"], groups=groups))


def _evidence(number: int, canonical: pd.DataFrame, raw: pd.DataFrame) -> str:
    if number == 1:
        # The "all" column of robson_inputs.md, with the same (secondary) suppression.
        overall = input_completeness(canonical)[["input", OVERALL_SCOPE]]
        return (
            markdown_table(overall.rename(columns={OVERALL_SCOPE: "pct_recorded"}))
            + "\n\nPer-facility completeness: robson_inputs.md."
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
            {
                "raw_column": str(raw.columns[i]),
                "pct_recorded": safe_pct(raw.iloc[:, i].map(_blank_to_none).notna()),
            }
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
