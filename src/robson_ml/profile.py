"""Data profile (spec §7). Every output is aggregate and small-cell suppressed."""

from __future__ import annotations

import math
import re
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
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
    DisclosureError,
    SumRelation,
    TableSpec,
    check_small_marker,
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
# Identifier fields and kinds: their raw column(s) show no quantile, no association and
# n_unique only as "*" (key_columns, _blank_key_statistics).
KEY_FIELDS = ("mother_key",)
KEY_KINDS = ("hash_key", "row_key")
# Canonical fields whose level counts open_questions.md prints with a "(missing)" row (Q2,
# Q3, Q6 by month, Q9). That row is one more copy of the field's overall missing count,
# published also for its raw column(s) in the variable profile and, for a Robson input, in
# the completeness table; all copies are hidden alike (linked_hidden_scopes).
LEVEL_FIELDS = (
    "onset_of_labour",
    "preeclampsia_recorded",
    "gdm_recorded",
    "delivery_date",
    "plurality",
)
# Rounds of re-protecting the files around each other's hidden cells; each round only adds
# hidden cells, so this is a guard, not a limit reached in practice.
MAX_LINK_ROUNDS = 20
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
    mask: pd.Series,
    facility: pd.Series,
    facilities: list[str],
    forced: Iterable[str] = (),
) -> dict[str, float | str]:
    """% of ``True`` in ``mask`` overall (key ``"all"``) and per facility, suppressed.

    The per-facility counts sum to the overall count, which is published, so besides the
    primary rule (count or complement of 1-4, ``"<5"``) a lone hidden facility cell would be
    recoverable by subtraction; secondary suppression hides another one (``"*"``,
    ``privacy.suppress_table``). Secondary cells are chosen in facility name order among
    the facilities with a non-zero ``min(count, rows - count)``, not by size, so ``mask``
    and ``~mask`` hide the same facilities and the choice says little about the values.
    Whenever two or more facilities are hidden the overall cell is hidden too
    (:func:`_suppress_scopes`).

    ``forced``: scopes (``"all"`` or facility) to hide whatever their value, e.g. because a
    linked copy of the same count is hidden elsewhere; the rest is protected around them.
    """
    spec = _scope_spec(mask, facility, facilities)
    rows = _scope_rows(facilities)
    cells = [("scopes", rows[scope], "n_true") for scope in forced if scope in rows]
    safe = _suppress_scopes({"scopes": spec}, ["scopes"], forced=cells)["scopes"]
    out = _scope_pcts(spec, safe, facilities)
    counts, sizes = _scope_counts(spec)
    for position, scope in enumerate([OVERALL_SCOPE, *facilities]):
        check_small_marker(out[scope], counts[position], sizes[position])
    return out


def _scope_rows(facilities: list[str]) -> dict[str, int]:
    """Scope (``"all"`` or facility) -> its row in a :func:`_scope_spec` table."""
    return {OVERALL_SCOPE: 0, **{fac: i for i, fac in enumerate(facilities, start=1)}}


def _scope_spec(mask: pd.Series, facility: pd.Series, facilities: list[str]) -> TableSpec:
    """Counts of ``mask`` overall (row 0) and per facility (rows 1..), for suppression.

    Secondary cells are chosen by ``rank``: the facilities in name order, the overall cell
    last, never by their values."""
    scopes = [mask, *(mask[facility == fac] for fac in facilities)]
    counts = pd.DataFrame(
        {"n_true": [int(s.sum()) for s in scopes], "n_rows": [len(s) for s in scopes]}
    )
    return TableSpec(
        counts,
        ["n_true"],
        complements={"n_true": "n_rows"},
        groups=[SumRelation(tuple(range(1, len(scopes))), total=0)],
        rank=[*range(1, len(scopes)), 0],
    )


def _suppress_scopes(
    tables: Mapping[Hashable, TableSpec],
    scope_keys: Iterable[Hashable],
    cross: list[SumRelation] | None = None,
    derived: Mapping[Hashable, DerivedCell] | None = None,
    forced: Iterable[Hashable] = (),
) -> dict[Hashable, pd.DataFrame]:
    """:func:`privacy.suppress_tables`, plus a value-independent rule for the
    :func:`_scope_spec` tables among them (``scope_keys``): a row hiding two or more facility
    cells always hides its overall cell too (``"*"``, unless primary suppression marks it
    ``"<5"``), whether or not the sum would pin them. Otherwise an overall cell hidden only
    when the facility cells happen to sum to a pinning total would give that total away.

    The rule is applied before protection too, to the facility cells hidden from the start
    (primary or ``forced``): with the overall cell hidden first, the sum pins nothing and no
    further facility is hidden to unpin it, which would again depend on the values."""
    forced_cells = list(forced)
    keys = list(scope_keys)
    for key in keys:
        spec = tables[key]
        primary = suppress_small_cells(spec.df, spec.count_columns, None, spec.complements)
        hidden = {
            row
            for row in spec.df.index[1:]
            if primary.at[row, "n_true"] == SUPPRESSED or (key, row, "n_true") in forced_cells
        }
        if len(hidden) >= 2:
            forced_cells.append((key, 0, "n_true"))
    while True:
        safe = suppress_tables(tables, cross or [], derived, forced_cells)
        extra = [
            (key, 0, "n_true")
            for key in keys
            if safe[key].at[0, "n_true"] not in HIDDEN
            and int(safe[key]["n_true"].iloc[1:].isin(HIDDEN).sum()) >= 2
        ]
        if not extra:
            return safe
        # Keep what this pass hid: solved again from scratch, a hidden overall cell could
        # stand in for a secondary facility cell and leave one facility hidden.
        for key, spec in tables.items():
            for col in spec.count_columns:
                shown = safe[key][col].astype(object)
                forced_cells += [(key, row, col) for row in shown.index[shown.isin(HIDDEN)]]
        forced_cells = list(dict.fromkeys([*forced_cells, *extra]))


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


def key_columns(config: MappingConfig) -> set[str]:
    """Raw columns holding a record or patient identifier: mapped to ``mother_key``, or to a
    field of kind ``hash_key`` or ``row_key``."""
    out: set[str] = set()
    for spec in config.fields.values():
        if spec.canonical in KEY_FIELDS or spec.kind in KEY_KINDS:
            out.update(spec.raw)
    return out


def _blank_key_statistics(profile: pd.DataFrame, keys: set[str]) -> pd.DataFrame:
    """For identifier columns, publish no quantiles and no association (blank), and
    ``n_unique`` as ``"*"``: a quantile of a numeric ID is one patient's ID, and
    ``n_nonnull - n_unique`` counts the rows repeating a key, which Q9 publishes only
    suppressed (a single shared pair would show 1)."""
    out = profile.copy()
    rows = out["raw_name"].astype(str).isin(keys)
    if not rows.any():
        return out
    for column in [f"p{int(q * 100)}" for q in QUANTILES] + ["association", "n_unique"]:
        if column in out.columns:
            out[column] = out[column].astype(object)
            out.loc[rows, column] = SECONDARY if column == "n_unique" else None
    if "association_metric" in out.columns:
        out.loc[rows, "association_metric"] = ""
    return out


def variable_profile(
    raw: pd.DataFrame,
    canonical: pd.DataFrame,
    config: MappingConfig,
    hide: Mapping[str, Mapping[str, str]] | None = None,
) -> pd.DataFrame:
    """One row per raw variable (spec §7): position, names, kind, missingness overall and per
    facility, distinct values, quantiles, univariate association with ``cs``, proposed
    status.

    ``raw`` and ``canonical`` must be row-aligned (same records, same order). An identifier
    column (:func:`key_columns`) shows only its kind, row counts and missingness, with
    ``n_unique`` as ``"*"`` (:func:`_blank_key_statistics`).

    ``hide``: for a raw column, missingness scopes (``all`` or facility) hidden where a
    linked copy of the same count is published (:func:`linked_hidden_scopes`), with the
    marker to show. They are hidden before the column's own protection, which then covers
    them too.
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
    keys = key_columns(config)
    links = hide or {}
    rows = []
    for position in range(raw.shape[1]):
        column = raw.columns[position]
        name = str(column)
        values = raw.iloc[:, position].map(_blank_to_none)
        kind = infer_kind(values)
        missing = pct_by_facility(values.isna(), facility, facilities, links.get(name, {}))
        row: dict[str, object] = {
            "raw_position": position,
            "raw_name": name,
            "canonical_name": ";".join(to_canonical.get(name, [])),
            "kind": kind,
            "n_rows": len(values),
            "n_nonnull": int(values.notna().sum()),
            "pct_missing": missing[OVERALL_SCOPE],
        }
        for fac in facilities:
            row[f"pct_missing_{fac}"] = missing[fac]
        row["n_unique"] = int(values.dropna().astype(str).nunique())
        is_key = name in keys
        numeric = pd.to_numeric(values, errors="coerce") if kind == "numeric" else None
        row.update(_released_quantiles(None if is_key else numeric))
        row["association_metric"] = ""
        row["association"] = None
        if is_key:
            pass  # an identifier: no statistic of its values (_blank_key_statistics)
        elif numeric is not None:
            row["association_metric"] = "auc"
            row["association"] = single_feature_auc(numeric, cs)
        elif kind == "categorical":
            row["association_metric"] = "cramers_v"
            row["association"] = cramers_v(values, cs)
        row["proposed_status"] = "review"
        if row["pct_missing"] in HIDDEN:
            # n_nonnull reveals the same count as pct_missing: hide it along with what
            # PROFILE_LINKED derives from it (write_profile repeats this for n_nonnull 1-4).
            for linked in ("n_nonnull", *PROFILE_LINKED["n_nonnull"]):
                row[linked] = row["pct_missing"]
        rows.append(row)
    profile = hide_profile_scopes(pd.DataFrame(rows), links)
    return _blank_key_statistics(profile, keys)


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


@dataclass(frozen=True)
class CanonicalCounts:
    """The canonical-frame counts robson_inputs.md and open_questions.md publish, protected
    in one pass (:func:`canonical_counts`)."""

    completeness: pd.DataFrame
    """``input``, ``all`` and one column per facility: % recorded, or a marker."""
    levels: dict[str, pd.DataFrame]
    """:data:`LEVEL_FIELDS` field -> its level counts (``value``, ``n``), suppressed."""
    sharing: tuple[object, object]
    """Q9: rows sharing a mother_key, and of these plurality >= 2 (a count or a marker)."""

    def missing_cell(self, field: str) -> object | None:
        """The ``"(missing)"`` count (or marker) of ``field``'s level counts; None when the
        table has no such row (nothing missing) or the field is not tabled."""
        table = self.levels.get(field)
        if table is None:
            return None
        cells = table.loc[table["value"] == MISSING_LABEL, "n"]
        return None if cells.empty else cells.iloc[0]


def _level_series(canonical: pd.DataFrame, field: str) -> pd.Series:
    """What open_questions.md counts by level for ``field`` (delivery_date by month)."""
    if field == "delivery_date":
        return canonical[field].dt.strftime("%Y-%m")
    return canonical[field]


def _apply_markers(cells: dict[str, object], scopes: Mapping[str, str]) -> None:
    """Hide ``scopes`` of ``cells`` with the given marker; ``"*"`` replaces a ``"<5"`` too,
    and a shown cell is never marked ``"<5"`` here (it gets ``"*"``)."""
    for scope, marker in scopes.items():
        if scope in cells and (marker == SECONDARY or cells[scope] not in HIDDEN):
            cells[scope] = SECONDARY


def canonical_counts(
    canonical: pd.DataFrame, hide: Mapping[str, Mapping[str, str]] | None = None
) -> CanonicalCounts:
    """The completeness rows, the level counts of :data:`LEVEL_FIELDS` and the Q9 shared-key
    counts, suppressed together.

    They are one system. A Robson input's ``"(missing)"`` count and its recorded count
    (completeness ``all``) sum to the number of records, so the pair is protected as one
    relation; and every ``"(missing)"`` count is kept from giving away a recorded count of
    1-4 (its complement, which completeness and the variable profile mark ``"<5"``). The Q9
    counts are linked as in :func:`_q9_specs`. Completeness rows also follow the
    value-independent overall rule of :func:`_suppress_scopes`.

    The recorded levels of a level table are not taken as a known sum: their total is
    ``records - (missing)``, so it is published exactly when the ``"(missing)"`` count is,
    the raw column's ``n_nonnull`` being hidden alike (:func:`linked_hidden_scopes`).

    ``hide``: for a completeness row label or a :data:`LEVEL_FIELDS` field, the scopes
    (``all`` or facility; for a level table only ``all``, its ``"(missing)"`` row) hidden
    because a linked copy of the count is hidden elsewhere, with the marker to show
    (:func:`linked_hidden_scopes`). They are hidden first and the rest is protected around
    them; ``"*"`` replaces a ``"<5"``.
    """
    links = hide or {}
    n = len(canonical)
    facility = canonical["facility_id"].astype(str)
    facilities = sorted(facility.unique())
    scope_rows = _scope_rows(facilities)
    masks = _completeness_masks(canonical)
    tables: dict[Hashable, TableSpec] = {
        ("completeness", label): _scope_spec(mask, facility, facilities)
        for label, mask in masks.items()
    }
    fields = [field for field in LEVEL_FIELDS if field in canonical.columns]
    missing_rows: dict[str, int] = {}
    cross: list[SumRelation] = []
    derived: dict[Hashable, DerivedCell] = {}
    for field in fields:
        spec = _counts_spec(_level_series(canonical, field), recorded_group=False)
        tables[("levels", field)] = spec
        rows = spec.df.index[spec.df["value"] == MISSING_LABEL]
        if not len(rows):
            continue
        row = missing_rows[field] = int(rows[0])
        cell = (("levels", field), row, "n")
        recorded = ("~recorded", field)
        derived[recorded] = DerivedCell(n - float(spec.df["n"][row]), cell)
        cross.append(SumRelation((cell, recorded)))
        if field in masks:
            cross.append(SumRelation((cell, (("completeness", field), 0, "n_true"))))
    if "plurality" in fields and "mother_key" in canonical.columns:
        sharing, q9_cross, q9_derived = _q9_specs(canonical, tables[("levels", "plurality")])
        tables["sharing"] = sharing
        cross += q9_cross
        derived.update(q9_derived)
    forced: list[Hashable] = []
    for key, scopes in links.items():
        for scope in scopes:
            if key in masks and scope in scope_rows:
                forced.append((("completeness", key), scope_rows[scope], "n_true"))
            if key in missing_rows and scope == OVERALL_SCOPE:
                forced.append((("levels", key), missing_rows[key], "n"))
    scope_keys = [("completeness", label) for label in masks]
    safe = _suppress_scopes(tables, scope_keys, cross, derived, forced)
    rows_out = []
    for label in masks:
        spec = tables[("completeness", label)]
        cells: dict[str, object] = dict(
            _scope_pcts(spec, safe[("completeness", label)], facilities)
        )
        _apply_markers(cells, links.get(label, {}))
        counts, sizes = _scope_counts(spec)
        for position, scope in enumerate([OVERALL_SCOPE, *facilities]):
            check_small_marker(cells[scope], counts[position], sizes[position])
        rows_out.append({"input": label, **cells})
    levels: dict[str, pd.DataFrame] = {}
    for field in fields:
        table = safe[("levels", field)].copy()
        if field in missing_rows:
            row = missing_rows[field]
            column = table["n"].astype(object).tolist()
            shown: dict[str, object] = {OVERALL_SCOPE: column[row]}
            _apply_markers(shown, links.get(field, {}))
            column[row] = shown[OVERALL_SCOPE]
            table["n"] = pd.Series(column, index=table.index, dtype=object)
        spec = tables[("levels", field)]
        for row in table.index:
            check_small_marker(table["n"][row], float(spec.df["n"][row]))
        levels[field] = table
    sharing_shown = safe["sharing"]["n"].tolist() if "sharing" in safe else [None, None]
    return CanonicalCounts(
        completeness=pd.DataFrame(rows_out, columns=["input", OVERALL_SCOPE, *facilities]),
        levels=levels,
        sharing=(sharing_shown[0], sharing_shown[1]),
    )


def input_completeness(
    classified: pd.DataFrame, hide: Mapping[str, Mapping[str, str]] | None = None
) -> pd.DataFrame:
    """% recorded for each of the six Robson inputs, overall (``all``) and per facility,
    then :data:`GA_BAND_RECORDED` (from :func:`canonical_counts`).

    Each row gets primary and secondary suppression across its facility cells. No row is
    nested in another (spec v1.2 coarse inputs are not published as "exact or band" or
    "precise type" rows): the difference of nested rows is a count published by
    subtraction, which the raw columns' missingness in the variable profile can bound until
    it is pinned. The GA band row instead counts the band alone and, like the inputs, is
    linked to the raw column(s) it is mapped from. Presentation gets no second row: no raw
    column counts the coarse (type unknown) records, so a "precise type" row could not be
    linked that way, and its effect on classification shows in the engine status tables.

    ``hide``: see :func:`canonical_counts`.
    """
    return canonical_counts(classified, hide).completeness


def _marker(cells: Iterable[object]) -> str | None:
    """The suppression marker of one published count shown in ``cells`` (a raw column can
    repeat in the frame): ``"*"`` if any shows it, else ``"<5"`` if any does, else None."""
    shown = list(cells)
    if SECONDARY in shown:
        return SECONDARY
    return SUPPRESSED if SUPPRESSED in shown else None


def linked_hidden_scopes(
    profile: pd.DataFrame, counts: CanonicalCounts, config: MappingConfig
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """Scopes (``all`` or facility) to hide alike wherever one count is published, with the
    marker to show.

    A field's missing count is published several times: for its raw column(s) in the
    variable profile (overall and per facility), for a Robson input or the GA band in the
    completeness table (counting recorded; :data:`COMPLETENESS_ROWS`), and overall as the
    ``"(missing)"`` row of its level counts in open_questions.md (:data:`LEVEL_FIELDS`).
    Where copies agree, a cell hidden in one place but shown in another gives it back.
    Copies are joined into groups and each group hides the union of what any member hides.
    The marker is ``"<5"`` only where every member publishing that scope shows ``"<5"``,
    else ``"*"`` for all: a ``"<5"`` in one place would restore the "1-4" that a ``"*"``
    elsewhere was chosen to withhold (``privacy.protect_cells`` rule (b)).

    Returns:
        (raw column name -> {scope: marker}, completeness row label or level field ->
        {scope: marker}), for every linked column, row and field.
    """
    completeness = counts.completeness
    by_input = completeness.set_index("input")
    scopes = [c for c in completeness.columns if c != "input"]
    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(node: tuple[str, str]) -> tuple[str, str]:
        parent.setdefault(node, node)
        while parent[node] != node:
            node = parent[node]
        return node

    def join(a: tuple[str, str], b: tuple[str, str]) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    raw_names = set(profile["raw_name"].astype(str))
    for spec in config.fields.values():
        label = COMPLETENESS_ROWS.get(spec.canonical)
        targets = []
        if label is not None and label in by_input.index:
            targets.append(("input", label))
        if spec.canonical in counts.levels:
            targets.append(("level", spec.canonical))
        for column in spec.raw:
            if column in raw_names:
                for target in targets:
                    join(("raw", column), target)
    for field in counts.levels:
        label = COMPLETENESS_ROWS.get(field)
        if label is not None and label in by_input.index:
            join(("level", field), ("input", label))
    profile_column = {
        scope: "pct_missing" if scope == OVERALL_SCOPE else f"pct_missing_{scope}"
        for scope in scopes
    }
    hidden: dict[tuple[str, str], set[str]] = {}
    not_small: dict[tuple[str, str], set[str]] = {}
    for node in list(parent):
        kind, name = node
        cells: dict[str, str | None] = {}
        if kind == "input":
            cells = {scope: _marker([by_input.loc[name, scope]]) for scope in scopes}
        elif kind == "raw":
            rows = profile["raw_name"].astype(str) == name
            for scope in scopes:
                if profile_column[scope] in profile.columns:
                    cells[scope] = _marker(profile.loc[rows, profile_column[scope]].tolist())
        else:
            cell = counts.missing_cell(name)
            if cell is not None:
                cells[OVERALL_SCOPE] = _marker([cell])
        group = find(node)
        hidden.setdefault(group, set()).update(s for s, c in cells.items() if c in HIDDEN)
        not_small.setdefault(group, set()).update(s for s, c in cells.items() if c != SUPPRESSED)
    raw_hide: dict[str, dict[str, str]] = {}
    hide: dict[str, dict[str, str]] = {}
    for node in parent:
        group = find(node)
        out = raw_hide if node[0] == "raw" else hide
        out[node[1]] = {
            scope: SECONDARY if scope in not_small[group] else SUPPRESSED
            for scope in sorted(hidden[group])
        }
    return raw_hide, hide


def _merge_hides(
    old: Mapping[str, Mapping[str, str]], new: Mapping[str, Mapping[str, str]]
) -> dict[str, dict[str, str]]:
    """The union of two :func:`linked_hidden_scopes` results; ``"*"`` wins over ``"<5"``.
    Keys hiding nothing are dropped."""
    out: dict[str, dict[str, str]] = {}
    for source in (old, new):
        for key, scopes in source.items():
            for scope, marker in scopes.items():
                merged = out.setdefault(key, {})
                merged[scope] = SECONDARY if SECONDARY in (marker, merged.get(scope)) else marker
    return {key: dict(sorted(scopes.items())) for key, scopes in sorted(out.items()) if scopes}


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
    classified: pd.DataFrame,
    hide: Mapping[str, Mapping[str, str]] | None = None,
    counts: CanonicalCounts | None = None,
) -> str:
    """Completeness of the six inputs, engine status distribution, Robson report table.

    The completeness table comes from ``counts``, else :func:`canonical_counts` with
    ``hide``.
    """
    counts = counts if counts is not None else canonical_counts(classified, hide)
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
            _table(counts.completeness),
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


def _counts_spec(series: pd.Series, recorded_group: bool = True) -> TableSpec:
    """Level counts to suppress. The levels sum to the published number of rows and, with
    ``recorded_group``, the recorded levels to the non-missing count (published for a raw
    column by the variable profile), so both groups are protected from subtraction. A
    :data:`LEVEL_FIELDS` table leaves the second out: its recorded total is published
    exactly when its ``"(missing)"`` count is (:func:`canonical_counts`)."""
    labelled = series.astype(object).where(series.notna(), MISSING_LABEL).astype(str)
    table = labelled.value_counts().rename_axis("value").reset_index(name="n")
    groups = [SumRelation(tuple(table.index))]
    recorded = tuple(table.index[table["value"] != MISSING_LABEL])
    if recorded_group and 2 <= len(recorded) < len(table):
        groups.append(SumRelation(recorded))
    return TableSpec(table, ["n"], groups=groups)


def _counts_table(series: pd.Series) -> str:
    """Level counts as a suppressed markdown table (see :func:`_counts_spec`)."""
    spec = _counts_spec(series)
    return _table(suppress_table(spec.df, spec.count_columns, groups=spec.groups))


def _q9_specs(
    canonical: pd.DataFrame, plurality: TableSpec
) -> tuple[TableSpec, list[SumRelation], dict[Hashable, DerivedCell]]:
    """Q9's shared-key counts, and how they tie to the plurality levels (table key
    ``("levels", "plurality")``).

    Besides each count's own small-cell rule and its complement, a reader can subtract:
    shared - shared multiples (shared rows that are not multiples) and the plurality >= 2
    levels - shared multiples (multiples that share no key). Both are modelled as
    never-published counts that must stay unknown when they hold 1-4.

    Returns:
        (the ``"sharing"`` table: rows sharing a mother_key, and of these plurality >= 2;
        cross relations; derived cells), for :func:`privacy.suppress_tables`.
    """
    n = len(canonical)
    multiple = (canonical["plurality"] >= 2).fillna(False).astype(bool)
    key = canonical["mother_key"]
    shared = key.notna() & key.duplicated(keep=False)
    n_shared, n_shared_multiple = int(shared.sum()), int((shared & multiple).sum())
    sharing = pd.DataFrame({"n": [n_shared, n_shared_multiple], "n_rows": [n, n]})
    shared_cell, multiple_cell = ("sharing", 0, "n"), ("sharing", 1, "n")
    not_multiple = ("~q9", "shared, not multiple")
    derived: dict[Hashable, DerivedCell] = {
        not_multiple: DerivedCell(n_shared - n_shared_multiple, shared_cell)
    }
    cross = [SumRelation((multiple_cell, not_multiple), shared_cell)]
    levels = pd.to_numeric(plurality.df["value"], errors="coerce")
    table = ("levels", "plurality")
    multiple_rows = tuple((table, i, "n") for i in plurality.df.index[levels >= 2])
    if multiple_rows:
        all_multiples, not_shared = ("~q9", "multiples"), ("~q9", "multiples, not shared")
        n_multiple = int(multiple.sum())
        derived[all_multiples] = DerivedCell(n_multiple, multiple_rows[0])
        derived[not_shared] = DerivedCell(n_multiple - n_shared_multiple, multiple_cell)
        cross += [
            SumRelation(multiple_rows, all_multiples),
            SumRelation((multiple_cell, not_shared), all_multiples),
        ]
    spec = TableSpec(sharing, ["n"], complements={"n": "n_rows"}, groups=[])
    return spec, cross, derived


def _plurality_evidence(canonical: pd.DataFrame, counts: CanonicalCounts | None = None) -> str:
    """Q9: plurality levels, rows sharing a mother_key, and how many of those are multiples
    (suppressed with every other canonical count, :func:`canonical_counts`). Only counts
    are published, never a key."""
    counts = counts if counts is not None else canonical_counts(canonical)
    shown_shared, shown_multiple = counts.sharing
    text = (
        "plurality:\n\n"
        + _table(counts.levels["plurality"])
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
    counts: CanonicalCounts | None = None,
) -> str:
    if number in (1, 2, 3, 6, 9) and counts is None:
        counts = canonical_counts(canonical, hide)
    if number == 1:
        assert counts is not None
        # The "all" column of robson_inputs.md, with the same (secondary) suppression.
        overall = counts.completeness[["input", OVERALL_SCOPE]]
        return (
            _table(overall.rename(columns={OVERALL_SCOPE: "pct_recorded"}))
            + "\n\nPer-facility completeness: robson_inputs.md."
        )
    if number == 2:
        assert counts is not None
        prelabour = canonical[canonical["onset_of_labour"] == "prelabour_cs"]
        return (
            "Canonical onset categories:\n\n"
            + _table(counts.levels["onset_of_labour"])
            + "\n\nPre-labour CS type among pre-labour CS rows:\n\n"
            + _counts_table(prelabour["prelabour_cs_type"])
        )
    if number == 3:
        assert counts is not None
        return (
            "preeclampsia_recorded:\n\n"
            + _table(counts.levels["preeclampsia_recorded"])
            + "\n\ngdm_recorded:\n\n"
            + _table(counts.levels["gdm_recorded"])
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
        assert counts is not None
        delivered = canonical["delivery_date"]
        # One more copy of the "(missing)" count of the month table: shown as it is.
        missing = counts.missing_cell("delivery_date")
        recorded: object = missing
        if missing not in HIDDEN:
            recorded = safe_pct(delivered.notna())
            if recorded == SUPPRESSED:
                raise DisclosureError("delivery_date: a small recorded count left shown")
        return (
            "The export has no admission timestamp (no admitted_at); delivery_date (date "
            "only) is the proxy time axis (spec §5, v1.2).\n\n"
            f"delivery_date recorded: {recorded}%.\n\nBy month:\n\n"
            + _table(counts.levels["delivery_date"])
        )
    if number == 7:
        return (
            "Not determinable until the C-Model variable list is transcribed into "
            "data/reference/cmodel_v1.yaml (spec §15.4)."
        )
    if number == 8:
        return "Not determinable from the data."
    return _plurality_evidence(canonical, counts)


def open_questions_markdown(
    canonical: pd.DataFrame,
    raw: pd.DataFrame,
    manual: Mapping[str, str],
    hide: Mapping[str, Mapping[str, str]] | None = None,
    counts: CanonicalCounts | None = None,
) -> str:
    """Answers to spec §25: automatic evidence plus the human answer (or a flag if none).

    Q1, Q2, Q3, Q6 and Q9 publish counts from ``counts``, else :func:`canonical_counts`
    with ``hide``.
    """
    counts = counts if counts is not None else canonical_counts(canonical, hide)
    parts = ["# Open questions (spec §25)", ""]
    for number, question in enumerate(QUESTIONS, start=1):
        answer = manual.get(f"q{number}", NOT_ANSWERED)
        parts += [
            f"## Q{number}. {question}",
            "",
            _evidence(number, canonical, raw, counts=counts),
            "",
            f"**Answer:** {answer}",
            "",
        ]
    return "\n".join(parts)


def _check_profile_markers(profile: pd.DataFrame, raw: pd.DataFrame, facility: pd.Series) -> None:
    """Raise :class:`privacy.DisclosureError` unless every ``"<5"`` in the profile's
    missingness and ``n_nonnull`` marks a count (or complement) of 1-4."""
    facilities = sorted(facility.unique())
    for position in range(raw.shape[1]):
        missing = raw.iloc[:, position].map(_blank_to_none).isna().reset_index(drop=True)
        row = profile.iloc[position]
        n = len(missing)
        check_small_marker(row["pct_missing"], float(missing.sum()), float(n))
        check_small_marker(row["n_nonnull"], float(n - missing.sum()), float(n))
        for fac in facilities:
            in_fac = missing[(facility == fac).to_numpy()]
            check_small_marker(row[f"pct_missing_{fac}"], float(in_fac.sum()), float(len(in_fac)))


def write_profile(
    raw: pd.DataFrame,
    classified: pd.DataFrame,
    config: MappingConfig,
    manual: Mapping[str, str],
    out_dir: Path,
) -> None:
    """Write the three §7 outputs under ``out_dir`` (reports/profile).

    Each count published more than once (a field's missingness for its raw column, its
    completeness row, its ``"(missing)"`` level count) is hidden alike in every file
    (:func:`linked_hidden_scopes`): the files are suppressed, the union of their hidden
    cells is forced hidden in all of them and each is protected again around it, until no
    file hides anything new.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_hide: dict[str, dict[str, str]] = {}
    hide: dict[str, dict[str, str]] = {}
    for _ in range(MAX_LINK_ROUNDS):
        profile = variable_profile(raw, classified, config, raw_hide)
        counts = canonical_counts(classified, hide)
        new_raw_hide, new_hide = linked_hidden_scopes(profile, counts, config)
        merged_raw, merged = _merge_hides(raw_hide, new_raw_hide), _merge_hides(hide, new_hide)
        if merged_raw == raw_hide and merged == hide:
            break
        raw_hide, hide = merged_raw, merged
    else:
        raise DisclosureError("linked suppression did not settle")
    # write_table's own suppression only checks n_nonnull's raw value; a small complement
    # (n_rows - n_nonnull) would also reveal a near-complete column, so that guard is
    # applied here first and write_table's pass over the already-suppressed values is a
    # no-op for the rows it already touched.
    profile = suppress_small_cells(
        profile, ["n_nonnull"], linked=PROFILE_LINKED, complements=PROFILE_COMPLEMENTS
    )
    profile = _blank_key_statistics(profile, key_columns(config))
    facility = classified["facility_id"].astype(str).reset_index(drop=True)
    _check_profile_markers(profile, raw, facility)
    write_table(
        profile,
        out_dir / "variable_profile.csv",
        ["n_nonnull"],
        PROFILE_LINKED,
    )
    (out_dir / "robson_inputs.md").write_text(
        robson_inputs_markdown(classified, counts=counts), encoding="utf-8"
    )
    (out_dir / "open_questions.md").write_text(
        open_questions_markdown(classified, raw, manual, counts=counts), encoding="utf-8"
    )
