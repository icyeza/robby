"""Offline Robson report table (spec §15.1).

Independent implementation: must never import the application's audit service code.

Phase G adds: a separate "6/7/9 non-cephalic (type unknown)" row (spec v1.2: presentation is
recorded only as malpresentation yes/no, so these records stay partial between groups 6, 7
and 9), the Vogel (2015) comparison columns when ``data/reference/vogel2015_v1.yaml`` exists
(:mod:`robson_ml.references`; never invented), and the onset caveat on groups 2 and 4 (spec
v1.3: the onset field is often coded retrospectively).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import pandas as pd

from robson_ml.privacy import SECONDARY, SUPPRESSED, SumRelation, TableSpec, suppress_tables
from robson_ml.reporting import markdown_table

if TYPE_CHECKING:
    from robson_ml.references import VogelReference

OVERALL = "ALL"
RESIDUAL = "residual"
NON_CEPHALIC = "6/7/9 non-cephalic (type unknown)"
NON_CEPHALIC_CANDIDATES = frozenset({6, 7, 9})
GROUP_LABELS = tuple(str(g) for g in range(1, 11))
HIDDEN_MARKERS = (SUPPRESSED, SECONDARY)
# Spec v1.3: onset is often coded retrospectively ("Planned C-section" ~ "had a CS").
ONSET_AFFECTED_ROWS = ("2", "4")
ONSET_NOTE = "onset-dependent (v1.3)"
ONSET_CAVEAT = (
    "Groups 2 and 4 (and their subgroups 2a/2b and 4a/4b) are defined by the onset of labour, "
    "which the export often codes retrospectively: 'Planned C-section' onset frequently means "
    "'had a CS' (spec v1.3). Their sizes, and by complement those of groups 1 and 3, are "
    "therefore uncertain, and the 2a/2b and 4a/4b split should not be interpreted."
)
NON_CEPHALIC_CAVEAT = (
    "Presentation is recorded only as malpresentation yes/no (spec v1.2): a record known to be "
    "non-cephalic but of unknown type cannot be placed in group 6, 7 or 9. These records form "
    "their own row, separate from the residual (other partial and conflict records)."
)
VOGEL_COLUMNS = ("ref_pct_of_deliveries", "ref_cs_rate", "diff_pct_of_deliveries", "diff_cs_rate")
REPORT_COLUMNS = [
    "facility",
    "row",
    "n",
    "pct_of_deliveries",
    "n_cs",
    "cs_rate",
    "abs_contribution",
    "rel_contribution",
]
REPORT_SUPPRESSION: dict[str, Any] = {
    "count_columns": ["n", "n_cs"],
    "linked": {
        "n": ["pct_of_deliveries", "n_cs", "cs_rate", "abs_contribution", "rel_contribution"],
        "n_cs": ["cs_rate", "abs_contribution", "rel_contribution"],
    },
    # n - n_cs (vaginal births in the group) must not be 1-4 either.
    "complements": {"n_cs": "n"},
}
# Published to 2 decimals only (see suppress_report_table).
REPORT_ROUNDED = ("pct_of_deliveries", "abs_contribution", "rel_contribution")
REPORT_DECIMALS = 2


def _ratio(num: float, den: float) -> float:
    return num / den if den else float("nan")


def _candidates(value: object) -> set[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    try:
        return {int(v) for v in value}  # type: ignore[attr-defined]
    except TypeError:
        return set()


def non_cephalic_mask(df: pd.DataFrame) -> pd.Series:
    """Partial records left between groups 6, 7 and 9 by a presentation recorded only as
    ``non_cephalic`` (spec v1.2): their candidate groups are a subset of {6, 7, 9}."""
    partial = df["robson_status"].astype(object).eq("partial")
    coarse = df.get("fetal_presentation", pd.Series(index=df.index, dtype=object))
    coarse = coarse.astype(object).eq("non_cephalic")
    if "robson_candidates" in df.columns:
        within = df["robson_candidates"].map(
            lambda c: bool(_candidates(c)) and _candidates(c) <= NON_CEPHALIC_CANDIDATES
        )
    else:
        within = pd.Series(True, index=df.index)
    return (partial & coarse & within.astype(bool)).fillna(False).astype(bool)


def row_labels(df: pd.DataFrame, split_non_cephalic: bool = False) -> pd.Series:
    """The report row of each record: its group ("1".."10") when resolved, else the
    residual; with ``split_non_cephalic``, :data:`NON_CEPHALIC` for the 6/7/9 records."""
    labels = pd.Series(
        [
            str(int(g)) if status == "resolved" else RESIDUAL
            for g, status in zip(df["robson_group"], df["robson_status"], strict=True)
        ],
        index=df.index,
        dtype=object,
    )
    if split_non_cephalic and len(df):
        labels[non_cephalic_mask(df)] = NON_CEPHALIC
    return labels


_row_labels = row_labels


def robson_report_table(
    df: pd.DataFrame, facility_col: str = "facility_id", split_non_cephalic: bool = False
) -> pd.DataFrame:
    """Robson report table per facility and overall, on P_audit rows.

    For each Robson group (1-10) plus a residual row (partial and conflict records): group
    size (n, % of deliveries), group CS (n, rate), absolute contribution (group CS / all
    deliveries) and relative contribution (group CS / all CS). With ``split_non_cephalic``
    the partial 6/7/9 non-cephalic records get their own row before the residual (spec
    v1.2). Unsuppressed; apply :func:`suppress_report_table` (or use
    :func:`published_audit_table`) before any export.
    """
    missing = {facility_col, "robson_status", "robson_group", "cs"} - set(df.columns)
    if missing:
        raise ValueError(f"robson_report_table is missing columns: {sorted(missing)}")
    if df["cs"].isna().any():
        raise ValueError("robson_report_table expects P_audit rows (cs non-missing)")
    labels = row_labels(df, split_non_cephalic)
    row_order = [*GROUP_LABELS, *([NON_CEPHALIC] if split_non_cephalic else []), RESIDUAL]
    scopes = [(OVERALL, df.index)] + [
        (str(f), sub.index) for f, sub in df.groupby(facility_col, sort=True)
    ]
    rows = []
    for facility, index in scopes:
        sub_cs = df.loc[index, "cs"].astype(int)
        sub_labels = labels.loc[index]
        total, total_cs = len(index), int(sub_cs.sum())
        for label in row_order:
            in_row = sub_labels == label
            n, n_cs = int(in_row.sum()), int(sub_cs[in_row].sum())
            rows.append(
                {
                    "facility": facility,
                    "row": label,
                    "n": n,
                    "pct_of_deliveries": _ratio(n, total),
                    "n_cs": n_cs,
                    "cs_rate": _ratio(n_cs, n),
                    "abs_contribution": _ratio(n_cs, total),
                    "rel_contribution": _ratio(n_cs, total_cs),
                }
            )
    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


def report_groups(table: pd.DataFrame) -> list[SumRelation]:
    """The sums a reader of the report table can form, as row-label groups.

    Each facility block (and the ALL block) sums to its total n, CS count and vaginal count,
    all derivable (from ``pct_of_deliveries`` and ``rel_contribution``, or the record count);
    and for each row label the facility cells sum to the published ALL cell.
    """
    blocks = table.groupby("facility", sort=False)
    groups = [SumRelation(tuple(block.index)) for _, block in blocks]
    keys = zip(table["facility"], table["row"], strict=True)
    position = {key: i for i, key in zip(table.index, keys, strict=True)}
    for label in dict.fromkeys(table["row"]):
        members = tuple(i for (f, r), i in position.items() if r == label and f != OVERALL)
        total = position.get((OVERALL, label))
        if members and total is not None:
            groups.append(SumRelation(members, total))
    return groups


def _two_decimals(value: object) -> object:
    if isinstance(value, str) or pd.isna(value):  # type: ignore[call-overload]
        return value
    return f"{float(value):.{REPORT_DECIMALS}f}"  # type: ignore[arg-type]


def report_table_spec(table: pd.DataFrame) -> TableSpec:
    """The report table (with a fresh 0..n-1 index) and the sums a reader can form in it,
    for :func:`privacy.suppress_tables` when it is published beside tables it is linked to.
    Round the result with :func:`round_report_table`."""
    table = table.reset_index(drop=True)
    return TableSpec(table, **REPORT_SUPPRESSION, groups=report_groups(table))


def round_report_table(safe: pd.DataFrame) -> pd.DataFrame:
    """Publish ``REPORT_ROUNDED`` columns to 2 decimals (see :func:`suppress_report_table`)."""
    safe = safe.copy()
    for column in REPORT_ROUNDED:
        safe[column] = safe[column].astype(object).map(_two_decimals)
    return safe


def suppress_report_table(table: pd.DataFrame) -> pd.DataFrame:
    """The report table as it may be exported: primary, then secondary, suppression.

    Primary: ``REPORT_SUPPRESSION`` (n or n_cs of 1-4, or n - n_cs of 1-4, with every value
    derived from them), marked ``"<5"``. Secondary (marked ``"*"``): within each facility
    block, and across the facilities of each row label (whose sum is the ALL row), no hidden
    n, n_cs or n - n_cs is recoverable by subtraction (``privacy.suppress_table``). Finally
    ``pct_of_deliveries``, ``abs_contribution`` and ``rel_contribution`` are published to 2
    decimals everywhere: at 3 decimals, ``rel_contribution`` on the shown rows pins a
    facility's CS total exactly. Secondary suppression already assumes every such total is
    known, so rounding is defence in depth, applied uniformly so the precision itself says
    nothing about which blocks hide cells.
    """
    spec = report_table_spec(table)
    return round_report_table(suppress_tables({OVERALL: spec})[OVERALL])


MARKERS_FOOTNOTE = (
    "`<5` = a count of 1-4 (or whose complement is 1-4); `*` = suppressed to protect another "
    "cell (the value may be any size)."
)


def _split_rows_groups(table: pd.DataFrame) -> list[SumRelation]:
    """Sums a reader can form over the non-cephalic and residual rows: per facility their
    sum is the residual published in the profile's report table (taken as known, the
    stronger reader), and across facilities each row sums to its ALL row."""
    groups = [SumRelation(tuple(block.index)) for _, block in table.groupby("facility", sort=False)]
    for _, cells in table.groupby("row", sort=False):
        members = tuple(cells.index[cells["facility"] != OVERALL])
        total = cells.index[cells["facility"] == OVERALL]
        if len(members) and len(total):
            groups.append(SumRelation(members, total[0]))
    return groups


def published_audit_table(
    classified: pd.DataFrame, vogel: VogelReference | None = None
) -> pd.DataFrame:
    """The Phase G Robson audit table, safe to publish beside ``reports/profile``.

    Rows 1-10 are exactly the published (jointly suppressed, rounded) rows of the profile's
    Robson report table (:func:`robson_ml.profile.status_tables`), so the two cannot be
    differenced. Its residual row is replaced by two rows, :data:`NON_CEPHALIC` and the
    remaining residual, suppressed together as a partition of that known residual. A
    ``note`` column flags the onset-dependent groups (:data:`ONSET_CAVEAT`). With ``vogel``,
    the reference group size, reference CS rate and the differences are added
    (:func:`add_vogel_columns`). Input: the classified canonical frame (all rows).
    """
    from robson_ml.populations import audit_population
    from robson_ml.profile import status_tables  # profile imports this module

    _, _, plain = status_tables(classified)
    audit, _ = audit_population(classified)
    split = robson_report_table(audit, split_non_cephalic=True)
    parts = split[split["row"].isin([NON_CEPHALIC, RESIDUAL])].reset_index(drop=True)
    spec = TableSpec(parts, **REPORT_SUPPRESSION, groups=_split_rows_groups(parts))
    safe_parts = round_report_table(suppress_tables({OVERALL: spec})[OVERALL])
    groups = plain[plain["row"].isin(GROUP_LABELS)]
    blocks = []
    for facility in dict.fromkeys(plain["facility"]):
        blocks.append(groups[groups["facility"] == facility])
        blocks.append(safe_parts[safe_parts["facility"] == facility])
    out = pd.concat(blocks, ignore_index=True)
    out["note"] = out["row"].map(lambda r: ONSET_NOTE if r in ONSET_AFFECTED_ROWS else "")
    return add_vogel_columns(out, vogel) if vogel is not None else out


def _hidden(value: object) -> bool:
    return isinstance(value, str) and value in HIDDEN_MARKERS


def _difference(value: object, reference: float, decimals: int | None) -> object:
    if _hidden(value):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    diff = float(value) - reference  # type: ignore[arg-type]
    return diff if decimals is None else f"{diff:.{decimals}f}"


def add_vogel_columns(table: pd.DataFrame, vogel: VogelReference) -> pd.DataFrame:
    """Add the Vogel (2015) reference columns to a published (suppressed) audit table.

    For group rows 1-10: reference share of deliveries and CS rate (fractions, as
    transcribed in the reference file), and the differences observed minus reference. A
    difference is computed from the *published* value (the 2-decimal share), and carries
    the observed cell's marker where that cell is hidden. Non-group rows have no reference.
    """
    out = table.copy()
    columns: dict[str, list[object]] = {c: [] for c in VOGEL_COLUMNS}
    for label, pct, rate in zip(out["row"], out["pct_of_deliveries"], out["cs_rate"], strict=True):
        if str(label) not in GROUP_LABELS:
            for column in VOGEL_COLUMNS:
                columns[column].append(float("nan"))
            continue
        ref_size, ref_rate = vogel.size(int(label)), vogel.cs_rate(int(label))
        columns["ref_pct_of_deliveries"].append(ref_size)
        columns["ref_cs_rate"].append(ref_rate)
        columns["diff_pct_of_deliveries"].append(_difference(pct, ref_size, REPORT_DECIMALS))
        columns["diff_cs_rate"].append(_difference(rate, ref_rate, None))
    for column in VOGEL_COLUMNS:
        out[column] = pd.Series(columns[column], index=out.index, dtype=object)
    return out


def audit_markdown(table: pd.DataFrame, vogel: VogelReference | None) -> str:
    """The published audit table as markdown, with its caveats and reference statement."""
    lines = ["# Robson audit table (P_audit)", "", f"- {ONSET_CAVEAT}", f"- {NON_CEPHALIC_CAVEAT}"]
    if vogel is None:
        lines.append(
            "- Vogel (2015) comparison: not available. The reference file "
            "`data/reference/vogel2015_v1.yaml` is absent; reference values are transcribed "
            "from the paper, never invented (schema in `robson_ml.references`)."
        )
    else:
        lines.append(
            f"- Vogel reference: {vogel.citation}; {vogel.source_table}; population: "
            f"{vogel.population}; file sha256 {vogel.sha256[:12]}. Reference columns are "
            "fractions; differences are observed minus reference."
        )
    lines += [
        "- Shares and contributions are fractions of deliveries or of CS, to 2 decimals.",
        "",
        markdown_table(table),
        "",
        MARKERS_FOOTNOTE,
        "",
    ]
    return "\n".join(lines)
