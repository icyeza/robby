"""Small-cell suppression and aggregate-only inspection helpers (spec §3)."""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import null_space

SMALL_CELL_THRESHOLD = 5
SUPPRESSED = "<5"
MAX_LEVELS_SHOWN = 30
_NULL_TOLERANCE = 1e-9


def _is_small(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return (numeric > 0) & (numeric < SMALL_CELL_THRESHOLD)


def suppress_small_cells(
    df: pd.DataFrame,
    count_columns: Sequence[str],
    linked: Mapping[str, Sequence[str]] | None = None,
    complements: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Replace counts of 1-4 with ``"<5"``, along with the columns derived from them.

    Args:
        df: an aggregate table.
        count_columns: columns holding counts of records.
        linked: for a count column, the derived columns (rates, percentages, other counts)
            that would reveal it and are suppressed on the same rows.
        complements: for a count column, the column holding the total it was drawn from.
            A row is also suppressed for that count column (and its linked columns) when
            the complement (total - count) is 1-4, since revealing the count would then
            reveal the complement.
    Returns:
        A copy; every touched column becomes object dtype. Zero counts are kept (they
        disclose no individual; DECISIONS.md 2026-09-23).
    """
    out = df.copy()
    links = dict(linked or {})
    comps = dict(complements or {})
    for col in count_columns:
        small = _is_small(df[col])
        if col in comps:
            total = pd.to_numeric(df[comps[col]], errors="coerce")
            count = pd.to_numeric(df[col], errors="coerce")
            complement = total - count
            small = small | ((complement > 0) & (complement < SMALL_CELL_THRESHOLD))
        for target in (col, *links.get(col, ())):
            out[target] = out[target].astype(object)
            out.loc[small, target] = SUPPRESSED
    return out


@dataclass(frozen=True)
class SumRelation:
    """``sum(members) == total``: cells whose total a reader knows or can derive.

    ``total`` is another cell, or ``None`` when the total is published or derivable outside
    the cells at hand (a table total, a facility size). In :func:`suppress_table` members and
    total are row labels, applied to each count column; in :func:`protect_cells` they are
    cells.
    """

    members: tuple[Hashable, ...]
    total: Hashable | None = None


def _small_value(value: float) -> bool:
    return 0 < value < SMALL_CELL_THRESHOLD


def _closure(cells: Iterable[Hashable], links: Mapping[Hashable, Iterable[Hashable]]) -> set:
    out: set[Hashable] = set()
    stack = list(cells)
    while stack:
        cell = stack.pop()
        if cell not in out:
            out.add(cell)
            stack.extend(links.get(cell, ()))
    return out


def _determined(unknown: set, relations: Sequence[SumRelation]) -> set:
    """Unknown cells whose value follows exactly from the relations (linear algebra).

    Each relation is an equation in the unknown cells, the known cells folding into its
    constant. A cell is determined when every solution of the homogeneous system leaves it
    unchanged, i.e. its row of the null-space basis is zero.
    """
    index: dict[Hashable, int] = {}
    equations: list[dict[Hashable, int]] = []
    for relation in relations:
        coeffs: dict[Hashable, int] = {}
        for cell in relation.members:
            if cell in unknown:
                coeffs[cell] = coeffs.get(cell, 0) + 1
        if relation.total is not None and relation.total in unknown:
            coeffs[relation.total] = coeffs.get(relation.total, 0) - 1
        coeffs = {cell: c for cell, c in coeffs.items() if c}
        if coeffs:
            equations.append(coeffs)
            for cell in coeffs:
                index.setdefault(cell, len(index))
    if not equations:
        return set()
    matrix = np.zeros((len(equations), len(index)))
    for i, coeffs in enumerate(equations):
        for cell, c in coeffs.items():
            matrix[i, index[cell]] = c
    basis = null_space(matrix)
    return {cell for cell, j in index.items() if np.all(np.abs(basis[j]) < _NULL_TOLERANCE)}


def _bounds_leak(
    relation: SumRelation,
    unknown: set,
    determined: set,
    values: Mapping[Hashable, float],
    never_shown: set,
) -> bool:
    """Whether the hidden members of a relation with a known total are pinned by their sum.

    A reader sees ``k`` suppressed cells and (from the total) their sum. Taking every
    ``"<5"`` as 1-4, a sum of ``k`` means all ones and ``4k`` all fours; those sums are
    refused. A never-published complement (``n - n_cs``) carries no ``"<5"`` of its own,
    and a reader cannot tell a complement of 1-4 from one of 0 or a large one; there the
    sum pins the cells only when their true values really are all ones or all fours.
    """
    total = relation.total
    if total is not None and total in unknown and total not in determined:
        return False
    free = [c for c in dict.fromkeys(relation.members) if c in unknown and c not in determined]
    if len(free) < 2:
        return False
    top = SMALL_CELL_THRESHOLD - 1
    if any(c in never_shown for c in free):
        return all(values[c] == 1 for c in free) or all(values[c] == top for c in free)
    hidden_sum = sum(values[c] for c in free)
    return hidden_sum in (len(free), top * len(free))


def protect_cells(
    values: Mapping[Hashable, float],
    suppressed: Iterable[Hashable],
    relations: Sequence[SumRelation],
    *,
    hidden: Iterable[Hashable] = (),
    linked: Mapping[Hashable, Iterable[Hashable]] | None = None,
    proxies: Mapping[Hashable, Hashable] | None = None,
) -> set[Hashable]:
    """Secondary (complementary) suppression: the cells to hide so none of ``suppressed``
    can be recovered from published totals (DECISIONS.md 2026-09-23).

    Args:
        values: the true value of every cell, published or not.
        suppressed: cells already hidden by primary suppression; each must stay unknown.
        relations: sums a reader knows (see :class:`SumRelation`).
        hidden: cells that are never published (e.g. the complement ``n - n_cs``). Any of
            them holding 1-4 must stay unknown too.
        linked: hiding a cell also hides these cells (e.g. ``n`` hides ``n_cs``).
        proxies: for a hidden cell, the published cell whose suppression hides it.

    Rule, repeated until stable: (a) no suppressed cell (primary or secondary), and no
    never-published cell holding 1-4, may be exactly determined by the relations (checked
    by linear algebra, so chains of subtractions across overlapping groups are caught, not
    only a lone suppressed cell in one group); (b) in a relation with a known total, the
    suppressed members must not be pinned by their sum (all ones, all fours). Each
    violation hides one more cell: for (a), the smallest non-zero published cell whose
    suppression alone frees the exposed cell, else the smallest non-zero published member
    of the offending relation; for (b), the latter. Zeros, then totals, are used only as a
    last resort. When nothing publishable is left to hide, the
    cell is fixed by published totals alone and suppression cannot help; it is left as is.

    Returns:
        The published cells to suppress (``suppressed`` and everything added).
    """
    links = dict(linked or {})
    prox = dict(proxies or {})
    never_shown = set(hidden)
    # A single member with an external total is simply published elsewhere; nothing here
    # can protect it, and its own table must.
    rels = [r for r in relations if len(r.members) >= 2 or r.total is not None]
    by_id = {id(r): r for r in rels}
    by_cell: dict[Hashable, list[int]] = {}
    for relation in rels:
        for cell in {*relation.members, relation.total} - {None}:
            by_cell.setdefault(cell, []).append(id(relation))
    supp = _closure(suppressed, links)
    # Checked in a fixed order so the result is deterministic.
    targets = dict.fromkeys(c for c in suppressed if c in supp)
    targets.update(dict.fromkeys(sorted(supp - set(targets), key=repr)))
    targets.update(dict.fromkeys(c for c in hidden if _small_value(values.get(c, 0.0))))

    def publishable(cell: Hashable) -> Hashable | None:
        shown = prox.get(cell) if cell in never_shown else cell
        return None if shown is None or shown in supp or shown in never_shown else shown

    def members_by_size(relation: SumRelation, zeros: bool) -> list[Hashable]:
        # Sized by the published cell's own value (a complement stands in for its count).
        options = []
        for position, cell in enumerate(relation.members):
            shown = publishable(cell)
            if shown is not None and (values.get(shown, 0.0) == 0) == zeros:
                options.append((values.get(shown, 0.0), position, shown))
        return [shown for *_, shown in sorted(options, key=lambda o: (o[0], o[1]))]

    def reachable(start: list[SumRelation], unknown: set) -> list[SumRelation]:
        # Breadth-first from the offending relations through relations sharing unknowns.
        seen: set[int] = set()
        order: list[SumRelation] = []
        queue = deque(id(r) for r in start)
        while queue:
            key = queue.popleft()
            if key in seen:
                continue
            seen.add(key)
            relation = by_id[key]
            order.append(relation)
            for cell in {*relation.members, relation.total} & unknown:
                queue.extend(k for k in by_cell.get(cell, ()) if k not in seen)
        return order

    def hide(cell: Hashable) -> None:
        added = _closure([cell], links) - supp
        supp.update(added)
        targets.update(dict.fromkeys(sorted(added, key=repr)))

    def remedy(start: list[SumRelation], unknown: set, exposed: Hashable | None) -> bool:
        order = reachable(start, unknown)
        for zeros in (False, True):
            if exposed is not None:
                # Prefer the smallest cell whose suppression alone un-determines the target.
                pool = [c for r in order for c in members_by_size(r, zeros)]
                pool = sorted(dict.fromkeys(pool), key=lambda c: values.get(c, 0.0))
                for cell in pool:
                    trial = supp | _closure([cell], links) | never_shown
                    if exposed not in _determined(trial, rels):
                        hide(cell)
                        return True
            for relation in order:
                sized = members_by_size(relation, zeros)
                if sized:
                    hide(sized[0])
                    return True
        for relation in order:
            total = None if relation.total is None else publishable(relation.total)
            if total is not None:
                hide(total)
                return True
        return False

    def unknown_count(relation: SumRelation, unknown: set) -> int:
        return len({*relation.members, relation.total} & unknown)

    fixed_by_totals: set[Hashable] = set()
    while True:
        unknown = supp | never_shown
        determined = _determined(unknown, rels)
        exposed = next((t for t in targets if t in determined and t not in fixed_by_totals), None)
        if exposed is not None:
            touching = [by_id[k] for k in dict.fromkeys(by_cell.get(exposed, ()))]
            touching.sort(key=lambda r: (unknown_count(r, unknown), len(r.members)))
            if not remedy(touching, unknown, exposed):
                fixed_by_totals.add(exposed)
            continue
        leaky = next(
            (
                r
                for r in rels
                if r not in fixed_by_totals
                and _bounds_leak(r, unknown, determined, values, never_shown)
            ),
            None,
        )
        if leaky is None:
            return supp - never_shown
        if not remedy([leaky], unknown, None):
            fixed_by_totals.add(leaky)


def suppress_table(
    df: pd.DataFrame,
    count_columns: Sequence[str],
    linked: Mapping[str, Sequence[str]] | None = None,
    complements: Mapping[str, str] | None = None,
    groups: Sequence[SumRelation] | None = None,
) -> pd.DataFrame:
    """Primary then secondary suppression of an aggregate table.

    Primary is :func:`suppress_small_cells`. Secondary treats each ``group`` (row labels of
    ``df``; default: the whole table, whose total is taken as known) as a sum a reader can
    form in every count column, and in the complement ``total - count`` of every
    ``complements`` column, then hides further cells (with their linked columns) until no
    suppressed cell or small complement is recoverable (:func:`protect_cells`).
    """
    if not df.index.is_unique:
        raise ValueError("suppress_table needs a unique row index")
    primary = suppress_small_cells(df, count_columns, linked, complements)
    links = {col: list(targets) for col, targets in (linked or {}).items()}
    comps = dict(complements or {})
    counts = set(count_columns)
    rows = list(df.index)
    row_groups = list(groups) if groups is not None else [SumRelation(tuple(rows))]

    def cell(row: Hashable, col: str) -> tuple[Hashable, str]:
        return (row, col)

    def complement(row: Hashable, col: str) -> tuple[str, Hashable, str]:
        return ("~complement", row, col)

    numeric = {col: pd.to_numeric(df[col], errors="coerce").fillna(0.0) for col in counts}
    values: dict[Hashable, float] = {}
    suppressed: list[Hashable] = []
    cell_links: dict[Hashable, list[Hashable]] = {}
    relations: list[SumRelation] = []
    for col in count_columns:
        for row in rows:
            values[cell(row, col)] = float(numeric[col][row])
            if isinstance(primary.at[row, col], str) and primary.at[row, col] == SUPPRESSED:
                suppressed.append(cell(row, col))
            cell_links[cell(row, col)] = [cell(row, c) for c in links.get(col, ()) if c in counts]
        for group in row_groups:
            total = None if group.total is None else cell(group.total, col)
            relations.append(SumRelation(tuple(cell(m, col) for m in group.members), total))
    hidden: list[Hashable] = []
    proxies: dict[Hashable, Hashable] = {}
    for col, total_col in comps.items():
        totals = pd.to_numeric(df[total_col], errors="coerce").fillna(0.0)
        count = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        for row in rows:
            comp = complement(row, col)
            values[comp] = float(totals[row] - count[row])
            hidden.append(comp)
            proxies[comp] = cell(row, col)
            total_cell = cell(row, total_col) if total_col in counts else None
            relations.append(SumRelation((cell(row, col), comp), total_cell))
        for group in row_groups:
            comp_total = None if group.total is None else complement(group.total, col)
            relations.append(
                SumRelation(tuple(complement(m, col) for m in group.members), comp_total)
            )
    final = protect_cells(
        values, suppressed, relations, hidden=hidden, linked=cell_links, proxies=proxies
    )
    out = primary.copy()
    for row, col in sorted(final - set(suppressed), key=repr):  # type: ignore[misc]
        for target in (col, *links.get(col, ())):
            out[target] = out[target].astype(object)
            out.at[row, target] = SUPPRESSED
    return out


def assert_no_small_cells(df: pd.DataFrame, count_columns: Sequence[str]) -> None:
    """Raise if any count column still holds a count of 1-4."""
    for col in count_columns:
        if _is_small(df[col]).any():
            raise AssertionError(f"column {col} holds a count below {SMALL_CELL_THRESHOLD}")


def fmt_count(n: int) -> str:
    """Format a single count for report text, suppressing 1-4."""
    return SUPPRESSED if 0 < n < SMALL_CELL_THRESHOLD else str(n)


def safe_pct(mask: pd.Series) -> float | str:
    """Percentage of ``True`` in ``mask``; suppressed when either side counts 1-4.

    ``NA`` entries are dropped before computing, so a nullable-boolean mask with
    unresolved rows is judged only on its resolved rows.
    """
    mask = mask.dropna()
    n = len(mask)
    n_true = int(mask.sum())
    if n == 0:
        return float("nan")
    if 0 < n_true < SMALL_CELL_THRESHOLD or 0 < n - n_true < SMALL_CELL_THRESHOLD:
        return SUPPRESSED
    return round(100.0 * n_true / n, 1)


def safe_describe(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column dtype, non-null count, % missing and number of distinct values. No values.

    ``n_nonnull`` is suppressed (spec §3.3) whenever it itself is 1-4, and also when its
    complement ``n_missing`` is 1-4 (revealing a near-complete or near-empty column would
    otherwise disclose the complement's small count). ``pct_missing`` and ``n_unique`` are
    suppressed on the same rows, since either would reveal the same small count.
    """
    n = len(df)
    rows = []
    for col in df.columns:
        s = df[col]
        nonnull = int(s.notna().sum())
        missing = n - nonnull
        suppress = 0 < nonnull < SMALL_CELL_THRESHOLD or 0 < missing < SMALL_CELL_THRESHOLD
        rows.append(
            {
                "column": str(col),
                "dtype": str(s.dtype),
                "n_nonnull": SUPPRESSED if suppress else nonnull,
                "pct_missing": (
                    SUPPRESSED
                    if suppress
                    else (round(100.0 * missing / n, 1) if n else float("nan"))
                ),
                "n_unique": SUPPRESSED if suppress else int(s.dropna().astype(str).nunique()),
            }
        )
    return pd.DataFrame(rows, columns=["column", "dtype", "n_nonnull", "pct_missing", "n_unique"])


def level_counts(series: pd.Series, max_levels: int = MAX_LEVELS_SHOWN) -> pd.DataFrame:
    """Label counts for a low-cardinality column, safe to display.

    Columns with more than ``max_levels`` distinct values are treated as free text: only
    the number of distinct values is returned. Labels seen fewer than 5 times are pooled
    and not shown. A column is also treated as free text when the values belonging to
    rare (<5-occurrence) levels make up more than half of its non-null values -- a sign
    that most values are unique-ish (e.g. names) rather than a small label set, even when
    the raw number of distinct values doesn't exceed ``max_levels``.
    """
    values = series.dropna().astype(str).str.strip()
    counts = values.value_counts()
    rare = counts[counts < SMALL_CELL_THRESHOLD]
    is_free_text = len(counts) > max_levels or (
        len(values) > 0 and int(rare.sum()) > len(values) / 2
    )
    if is_free_text:
        label = f"(free text: {len(counts)} distinct values, not shown)"
        return pd.DataFrame({"level": [label], "n": [fmt_count(len(values))]})
    rows: list[dict[str, object]] = [
        {"level": str(k), "n": int(v)} for k, v in counts.items() if v >= SMALL_CELL_THRESHOLD
    ]
    if len(rare):
        rows.append({"level": f"(other: {len(rare)} rare levels)", "n": fmt_count(int(rare.sum()))})
    rows.append({"level": "(missing)", "n": fmt_count(int(series.isna().sum()))})
    return pd.DataFrame(rows, columns=["level", "n"])
