"""Data profile (spec §7). Every output is aggregate and small-cell suppressed."""

from __future__ import annotations

import math
import re
from collections.abc import Hashable, Iterable, Mapping
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score

from robson_engine import INPUT_FIELDS
from robson_ml.audit_offline import (
    OVERALL,
    RESIDUAL,
    report_table_spec,
    robson_report_table,
    round_report_table,
)
from robson_ml.ingest import infer_kind
from robson_ml.mapping import MappingConfig
from robson_ml.populations import audit_population
from robson_ml.privacy import (
    SECONDARY,
    SMALL_CELL_THRESHOLD,
    SUPPRESSED,
    DerivedCell,
    SumRelation,
    TableSpec,
    safe_pct,
    suppress_small_cells,
    suppress_table,
    suppress_tables,
)
from robson_ml.reporting import markdown_table, write_table
from robson_ml.schema import GA_BAND_FIELDS

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
# The completeness row after the six inputs (spec v1.2): GA recorded as a band. It counts
# the band alone, not "exact or band": a row nested in another publishes their difference
# (records with GA only as a band) by subtraction, and the raw columns' missingness in the
# variable profile bounds both rows, so the difference could be pinned.
GA_BAND_RECORDED = "ga_band (recorded)"
# Canonical field -> the completeness row counting it; a raw column mapped to the field
# hides the same scopes as that row (linked_hidden_scopes).
COMPLETENESS_ROWS: dict[str, str] = {
    **{name: name for name in INPUT_FIELDS},
    **{name: GA_BAND_RECORDED for name in GA_BAND_FIELDS},
}
# Canonical fields whose raw column(s) never show n_unique: n_nonnull - n_unique counts the
# rows repeating a key, which Q9 publishes only small-cell suppressed.
KEY_FIELDS = ("mother_key",)
KEY_KINDS = ("hash_key",)
PROFILE_LINKED = {
    "n_nonnull": [*(f"p{int(q * 100)}" for q in QUANTILES), "association", "n_unique"],
}
PROFILE_COMPLEMENTS = {"n_nonnull": "n_rows"}
HIDDEN = (SUPPRESSED, SECONDARY)
SECONDARY_NOTE = "`*` = suppressed to protect another cell (value may be any size)."
MARKERS_NOTE = (
    "Suppression markers (here, in open_questions.md and in variable_profile.csv): `<5` = "
    "a count of 1-4, or one whose complement is 1-4; `*` = suppressed to protect another "
    "cell (value may be any size)."
)


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
    primary rule (count or complement of 1-4, ``"<5"``) a lone hidden facility cell would be
    recoverable by subtraction; secondary suppression hides another one (``"*"``,
    ``privacy.suppress_table``). The facilities are in name order, and the choice of
    secondary cells depends only on ``min(count, rows - count)``, so ``mask`` and ``~mask``
    hide the same facilities.
    """
    spec = _scope_spec(mask, facility, facilities)
    safe = suppress_table(
        spec.df, spec.count_columns, complements=spec.complements, groups=spec.groups
    )
    return _scope_pcts(spec, safe, facilities)


def _scope_spec(mask: pd.Series, facility: pd.Series, facilities: list[str]) -> TableSpec:
    """Counts of ``mask`` overall (row 0) and per facility (rows 1..), for suppression."""
    scopes = [mask, *(mask[facility == fac] for fac in facilities)]
    counts = pd.DataFrame(
        {"n_true": [int(s.sum()) for s in scopes], "n_rows": [len(s) for s in scopes]}
    )
    return TableSpec(
        counts,
        ["n_true"],
        complements={"n_true": "n_rows"},
        groups=[SumRelation(tuple(range(1, len(scopes))), total=0)],
    )


def _scope_counts(spec: TableSpec) -> tuple[list[int], list[int]]:
    """(counts, scope sizes) of a :func:`_scope_spec` table, overall first."""
    return (
        [int(v) for v in spec.df["n_true"].tolist()],
        [int(v) for v in spec.df["n_rows"].tolist()],
    )


def _scope_pcts(
    spec: TableSpec, safe: pd.DataFrame, facilities: list[str]
) -> dict[str, float | str]:
    """Percentages from a suppressed :func:`_scope_spec` table; the marker where hidden."""
    n_trues, n_rows_list = _scope_counts(spec)
    out: dict[str, float | str] = {}
    for position, key in enumerate([OVERALL_SCOPE, *facilities]):
        n_true, n_rows = n_trues[position], n_rows_list[position]
        marker = safe.at[position, "n_true"]
        if isinstance(marker, str) and marker in HIDDEN:
            out[key] = marker
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

    ``raw`` and ``canonical`` must be row-aligned (same records, same order). A raw column
    mapped to ``mother_key`` (or with kind ``hash_key``) shows ``n_unique`` as ``"*"``:
    ``n_nonnull - n_unique`` is the number of rows repeating a key, and would pin the
    suppressed Q9 count of rows sharing one (a single shared pair shows 1).
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
    key_columns: set[str] = set()
    for spec in config.fields.values():
        for column in spec.raw:
            to_canonical.setdefault(column, []).append(spec.canonical)
            if spec.canonical in KEY_FIELDS or spec.kind in KEY_KINDS:
                key_columns.add(column)
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
        if row["pct_missing"] in HIDDEN:
            # n_nonnull reveals the same count as pct_missing: hide it along with what
            # PROFILE_LINKED derives from it (write_profile repeats this for n_nonnull 1-4).
            for name in ("n_nonnull", *PROFILE_LINKED["n_nonnull"]):
                row[name] = row["pct_missing"]
        if str(column) in key_columns:
            row["n_unique"] = SECONDARY
        rows.append(row)
    return pd.DataFrame(rows)


def _hide(cells: pd.DataFrame, row: int, columns: Iterable[str], force: bool = False) -> None:
    """Mark ``columns`` of ``row`` secondary-suppressed, keeping any primary ``"<5"`` unless
    ``force``."""
    for column in columns:
        cells[column] = cells[column].astype(object)
        if force or cells.at[row, column] not in HIDDEN:
            cells.at[row, column] = SECONDARY


def _completeness_masks(classified: pd.DataFrame) -> dict[str, pd.Series]:
    """Row label -> recorded mask: the six inputs (not missing), then the GA band (both
    bounds recorded)."""
    masks = {name: classified[name].notna() for name in INPUT_FIELDS}
    if set(GA_BAND_FIELDS) <= set(classified.columns):
        masks[GA_BAND_RECORDED] = classified[list(GA_BAND_FIELDS)].notna().all(axis=1)
    else:
        masks[GA_BAND_RECORDED] = pd.Series(False, index=classified.index)
    return masks


def input_completeness(
    classified: pd.DataFrame, hide: Mapping[str, Mapping[str, str]] | None = None
) -> pd.DataFrame:
    """% recorded for each of the six Robson inputs, overall (``all``) and per facility,
    then :data:`GA_BAND_RECORDED`.

    Each row gets primary and secondary suppression across its facility cells. No row is
    nested in another (spec v1.2 coarse inputs are not published as "exact or band" or
    "precise type" rows): the difference of nested rows is a count published by
    subtraction, which the raw columns' missingness in the variable profile can bound until
    it is pinned. The GA band row instead counts the band alone and, like the inputs, is
    linked to the raw column(s) it is mapped from. Presentation gets no second row: no raw
    column counts the coarse (type unknown) records, so a "precise type" row could not be
    linked that way, and its effect on classification shows in the engine status tables.

    ``hide``: for a row, further scopes (``all`` or facility) to hide, with the marker to
    show (see :func:`linked_hidden_scopes`). ``"*"`` replaces a ``"<5"`` too.
    """
    facility = classified["facility_id"].astype(str)
    facilities = sorted(facility.unique())
    masks = _completeness_masks(classified)
    specs: dict[Hashable, TableSpec] = {
        label: _scope_spec(mask, facility, facilities) for label, mask in masks.items()
    }
    safe = suppress_tables(specs)
    rows = [
        {"input": label, **_scope_pcts(specs[label], safe[label], facilities)} for label in masks
    ]
    out = pd.DataFrame(rows, columns=["input", OVERALL_SCOPE, *facilities])
    for position, label in enumerate(out["input"]):
        scopes = (hide or {}).get(label, {})
        for marker in (SUPPRESSED, SECONDARY):
            columns = [s for s, m in scopes.items() if m == marker and s in out.columns]
            _hide(out, position, [c for c in columns if c != "input"], marker == SECONDARY)
    return out


def _marker(cells: Iterable[object]) -> str | None:
    """The suppression marker of one published count shown in ``cells`` (a raw column can
    repeat in the frame): ``"*"`` if any shows it, else ``"<5"`` if any does, else None."""
    shown = list(cells)
    if SECONDARY in shown:
        return SECONDARY
    return SUPPRESSED if SUPPRESSED in shown else None


def linked_hidden_scopes(
    profile: pd.DataFrame, completeness: pd.DataFrame, config: MappingConfig
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """Scopes (``all`` or facility) to hide alike for each completeness row and its raw
    column(s), with the marker to show.

    A Robson input's missingness is published per facility twice: for the raw column in
    the variable profile (counting missing) and for the canonical field in the completeness
    table (counting recorded); likewise the GA band (:data:`COMPLETENESS_ROWS`). Where the
    two agree, a cell hidden in one file but shown in the other gives it back. Rows and the
    raw columns they are mapped from are joined into groups, and each group hides the union
    of what any member hides. The marker is ``"<5"`` only where every member shows ``"<5"``,
    else ``"*"`` for all: a ``"<5"`` in one file would restore the "1-4" that a ``"*"`` in
    the other was chosen to withhold (``privacy.protect_cells`` rule (b)).

    Returns:
        (raw column name -> {scope: marker}, row label -> {scope: marker}), for every
        linked column and row.
    """
    by_input = completeness.set_index("input")
    scopes = [c for c in completeness.columns if c != "input"]
    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(node: tuple[str, str]) -> tuple[str, str]:
        parent.setdefault(node, node)
        while parent[node] != node:
            node = parent[node]
        return node

    raw_names = set(profile["raw_name"].astype(str))
    for spec in config.fields.values():
        label = COMPLETENESS_ROWS.get(spec.canonical)
        if label is None or label not in by_input.index:
            continue
        for column in spec.raw:
            if column in raw_names:
                parent[find(("raw", column))] = find(("input", label))
    profile_column = {
        scope: "pct_missing" if scope == OVERALL_SCOPE else f"pct_missing_{scope}"
        for scope in scopes
    }
    hidden: dict[tuple[str, str], set[str]] = {}
    primary: dict[tuple[str, str], set[str]] = {}
    for node in list(parent):
        kind, name = node
        cells: dict[str, str | None] = {}
        for scope in scopes:
            if kind == "input":
                cells[scope] = _marker([by_input.loc[name, scope]])
            elif profile_column[scope] in profile.columns:
                rows = profile.loc[profile["raw_name"].astype(str) == name, profile_column[scope]]
                cells[scope] = _marker(rows.tolist())
        group = find(node)
        found = {scope for scope, cell in cells.items() if cell in HIDDEN}
        hidden.setdefault(group, set()).update(found)
        small = {scope for scope in scopes if cells.get(scope) == SUPPRESSED}
        primary[group] = primary[group] & small if group in primary else small
    raw_hide: dict[str, dict[str, str]] = {}
    input_hide: dict[str, dict[str, str]] = {}
    for node in parent:
        group = find(node)
        target = input_hide if node[0] == "input" else raw_hide
        target[node[1]] = {
            scope: SUPPRESSED if scope in primary[group] else SECONDARY
            for scope in sorted(hidden[group])
        }
    return raw_hide, input_hide


def hide_profile_scopes(
    profile: pd.DataFrame, hide: Mapping[str, Mapping[str, str]]
) -> pd.DataFrame:
    """Mark the given scopes of each raw column's missingness hidden in the profile, with
    the given marker (``"*"`` replaces a ``"<5"`` too)."""
    out = profile.copy()
    for position, name in enumerate(out["raw_name"].astype(str)):
        for marker in (SUPPRESSED, SECONDARY):
            scopes = {s for s, m in hide.get(name, {}).items() if m == marker}
            columns = [f"pct_missing_{s}" for s in scopes if f"pct_missing_{s}" in out.columns]
            if OVERALL_SCOPE in scopes:
                columns += ["pct_missing", "n_nonnull", *PROFILE_LINKED["n_nonnull"]]
            _hide(out, out.index[position], columns, marker == SECONDARY)
    return out


def _table(df: pd.DataFrame) -> str:
    """A suppressed table as markdown, with the ``"*"`` footnote when it has one."""
    text = markdown_table(df)
    if (df.astype(object) == SECONDARY).any().any():
        text += "\n\n" + SECONDARY_NOTE
    return text


def _status_tables(classified: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Engine status, resolving fields and the Robson report table, protected jointly.

    A reader can link them: the status rows sum to Records; the resolving-fields rows sum
    to the status partial count; the report's ALL block sums to Records minus the (exactly
    reported) excluded rows, and its residual row counts the partial and conflict records
    kept in P_audit. That last sum is exact when nothing is excluded and otherwise known to
    within the excluded count; it is treated as exact, which protects against the stronger
    reader. All three tables are suppressed in one pass over these sums.
    """
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
    audit, _ = audit_population(classified)
    report = report_table_spec(robson_report_table(audit))
    keys = zip(report.df["facility"], report.df["row"], strict=True)
    residual = next(
        i for i, key in zip(report.df.index, keys, strict=True) if key == (OVERALL, RESIDUAL)
    )
    cross = [
        SumRelation(tuple(("resolving", i, "n") for i in resolving.index), ("status", 1, "n")),
        SumRelation((("status", 1, "n"), ("status", 2, "n")), ("report", residual, "n")),
    ]
    safe = suppress_tables(
        {
            "status": TableSpec(status_table, ["n"], {"n": ["pct"]}),
            # Its only sum is the partial cell, in the cross relations.
            "resolving": TableSpec(resolving, ["n"], groups=[]),
            "report": report,
        },
        cross,
    )
    return safe["status"], safe["resolving"], round_report_table(safe["report"])


def robson_inputs_markdown(
    classified: pd.DataFrame, hide: Mapping[str, Mapping[str, str]] | None = None
) -> str:
    """Completeness of the six inputs, engine status distribution, Robson report table.

    ``hide`` is passed to :func:`input_completeness`.
    """
    status, resolving, report = _status_tables(classified)
    _, log = audit_population(classified)
    version = ", ".join(sorted(classified["rule_set_version"].astype(str).unique()))
    return "\n".join(
        [
            "# Robson inputs",
            "",
            f"Rule set: {version}. Records: {len(classified)}.",
            "",
            MARKERS_NOTE,
            "",
            "## Completeness of the six inputs (% recorded)",
            "",
            _table(input_completeness(classified, hide)),
            "",
            "## Engine status",
            "",
            _table(status),
            "",
            "### Partial records: fields that would resolve them",
            "",
            _table(resolving),
            "",
            "## Robson report table (P_audit)",
            "",
            f"{log.n_excluded} rows excluded for missing outcome (reported exactly; "
            "data-quality count, spec §4.2).",
            "",
            "pct_of_deliveries, abs_contribution and rel_contribution are given to 2 decimals.",
            "",
            _table(report),
            "",
        ]
    )


def _counts_spec(series: pd.Series) -> TableSpec:
    """Level counts to suppress. The levels sum to the published number of rows, and the
    recorded levels to the non-missing count (published for the raw column by the variable
    profile), so both groups are protected from subtraction."""
    labelled = series.astype(object).where(series.notna(), MISSING_LABEL).astype(str)
    table = labelled.value_counts().rename_axis("value").reset_index(name="n")
    groups = [SumRelation(tuple(table.index))]
    recorded = tuple(table.index[table["value"] != MISSING_LABEL])
    if 2 <= len(recorded) < len(table):
        groups.append(SumRelation(recorded))
    return TableSpec(table, ["n"], groups=groups)


def _counts_table(series: pd.Series) -> str:
    """Level counts as a suppressed markdown table (see :func:`_counts_spec`)."""
    spec = _counts_spec(series)
    return _table(suppress_table(spec.df, spec.count_columns, groups=spec.groups))


def _plurality_evidence(canonical: pd.DataFrame) -> str:
    """Q9: plurality levels, rows sharing a mother_key, and how many of those are multiples.

    Suppressed jointly. Besides each count's own small-cell rule and its complement, a
    reader can subtract: shared - shared multiples (shared rows that are not multiples) and
    the plurality >= 2 levels - shared multiples (multiples that share no key). Both are
    modelled as never-published counts that must stay unknown when they hold 1-4. Only
    counts are published, never a key.
    """
    n = len(canonical)
    plurality = canonical["plurality"]
    counts = _counts_spec(plurality)
    key = canonical["mother_key"]
    shared = key.notna() & key.duplicated(keep=False)
    multiple = (plurality >= 2).fillna(False).astype(bool)
    n_shared, n_shared_multiple = int(shared.sum()), int((shared & multiple).sum())
    sharing = pd.DataFrame({"n": [n_shared, n_shared_multiple], "n_rows": [n, n]})
    shared_cell, multiple_cell = ("sharing", 0, "n"), ("sharing", 1, "n")
    not_multiple = ("~q9", "shared, not multiple")
    derived: dict[Hashable, DerivedCell] = {
        not_multiple: DerivedCell(n_shared - n_shared_multiple, shared_cell)
    }
    cross = [SumRelation((multiple_cell, not_multiple), shared_cell)]
    levels = pd.to_numeric(counts.df["value"], errors="coerce")
    multiple_rows = tuple(("plurality", i, "n") for i in counts.df.index[levels >= 2])
    if multiple_rows:
        all_multiples, not_shared = ("~q9", "multiples"), ("~q9", "multiples, not shared")
        n_multiple = int(multiple.sum())
        derived[all_multiples] = DerivedCell(n_multiple, multiple_rows[0])
        derived[not_shared] = DerivedCell(n_multiple - n_shared_multiple, multiple_cell)
        cross += [
            SumRelation(multiple_rows, all_multiples),
            SumRelation((multiple_cell, not_shared), all_multiples),
        ]
    safe = suppress_tables(
        {
            "plurality": counts,
            "sharing": TableSpec(sharing, ["n"], complements={"n": "n_rows"}, groups=[]),
        },
        cross,
        derived,
    )
    shown_shared, shown_multiple = safe["sharing"]["n"].tolist()
    text = (
        "plurality:\n\n"
        + _table(safe["plurality"])
        + f"\n\nRows sharing a mother_key with another row: {shown_shared}. "
        f"Of these, plurality >= 2: {shown_multiple}. Rows sharing a key are kept, flagged "
        "and grouped by mother_key in every split (spec v1.2); shared rows with plurality "
        ">= 2 may be one row per baby."
    )
    if SECONDARY in (shown_shared, shown_multiple):
        text += "\n\n" + SECONDARY_NOTE
    return text


def _evidence(
    number: int,
    canonical: pd.DataFrame,
    raw: pd.DataFrame,
    hide: Mapping[str, Mapping[str, str]] | None = None,
) -> str:
    if number == 1:
        # The "all" column of robson_inputs.md, with the same (secondary) suppression.
        overall = input_completeness(canonical, hide)[["input", OVERALL_SCOPE]]
        return (
            _table(overall.rename(columns={OVERALL_SCOPE: "pct_recorded"}))
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
        delivered = canonical["delivery_date"]
        months = delivered.dt.strftime("%Y-%m")
        return (
            "The export has no admission timestamp (no admitted_at); delivery_date (date "
            "only) is the proxy time axis (spec §5, v1.2).\n\n"
            f"delivery_date recorded: {safe_pct(delivered.notna())}%.\n\nBy month:\n\n"
            + _counts_table(months)
        )
    if number == 7:
        return (
            "Not determinable until the C-Model variable list is transcribed into "
            "data/reference/cmodel_v1.yaml (spec §15.4)."
        )
    if number == 8:
        return "Not determinable from the data."
    return _plurality_evidence(canonical)


def open_questions_markdown(
    canonical: pd.DataFrame,
    raw: pd.DataFrame,
    manual: Mapping[str, str],
    hide: Mapping[str, Mapping[str, str]] | None = None,
) -> str:
    """Answers to spec §25: automatic evidence plus the human answer (or a flag if none).

    ``hide`` is passed to :func:`input_completeness` (Q1).
    """
    parts = ["# Open questions (spec §25)", ""]
    for number, question in enumerate(QUESTIONS, start=1):
        answer = manual.get(f"q{number}", NOT_ANSWERED)
        parts += [
            f"## Q{number}. {question}",
            "",
            _evidence(number, canonical, raw, hide),
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
    # An input's missingness per facility appears for its raw column here and for the
    # canonical field in robson_inputs.md; both files hide the union of the two patterns.
    raw_hide, input_hide = linked_hidden_scopes(profile, input_completeness(classified), config)
    profile = hide_profile_scopes(profile, raw_hide)
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
    (out_dir / "robson_inputs.md").write_text(
        robson_inputs_markdown(classified, input_hide), encoding="utf-8"
    )
    (out_dir / "open_questions.md").write_text(
        open_questions_markdown(classified, raw, manual, input_hide), encoding="utf-8"
    )
