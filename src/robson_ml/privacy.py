"""Small-cell suppression and aggregate-only inspection helpers (spec §3)."""

from __future__ import annotations

import math
from collections import Counter, deque
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.linalg import null_space

SMALL_CELL_THRESHOLD = 5
SUPPRESSED = "<5"
SECONDARY = "*"
MAX_LEVELS_SHOWN = 30
_NULL_TOLERANCE = 1e-9
# Rounds of interval tightening over the relations; a cap, as a chain can shrink slowly.
_MAX_SWEEPS = 100


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


def _reader_bounds(
    relations: Sequence[SumRelation],
    values: Mapping[Hashable, float],
    unknown: set,
    small: set,
) -> dict[Hashable, tuple[float, float]]:
    """The interval a reader can derive for each cell of ``relations``.

    A cell not in ``unknown`` is its value. An unknown cell in ``small`` (shown ``"<5"``,
    or a never-published complement whose count is shown ``"<5"``) starts at 1-4, any other
    unknown cell (``"*"``, a never-published count) at 0 and up. Each relation
    ``sum(members) == total`` then bounds every term by the others' bounds, repeated until
    stable (bounds propagation). A relation without a total cell has a known sum.
    """
    lower: dict[Hashable, float] = {}
    upper: dict[Hashable, float] = {}
    equations: list[tuple[Counter, Hashable | None, float]] = []
    for relation in relations:
        coeffs = Counter(relation.members)
        cells = [*coeffs, *([] if relation.total is None else [relation.total])]
        if not any(c in unknown for c in cells) or relation.total in coeffs:
            continue
        for cell in cells:
            if cell in lower:
                continue
            if cell not in unknown:
                lower[cell] = upper[cell] = float(values.get(cell, 0.0))
            elif cell in small:
                lower[cell], upper[cell] = 1.0, float(SMALL_CELL_THRESHOLD - 1)
            else:
                lower[cell], upper[cell] = 0.0, math.inf
        known = sum(k * float(values.get(c, 0.0)) for c, k in coeffs.items())
        equations.append((coeffs, relation.total, known))

    def tighten(cell: Hashable, low: float, high: float) -> bool:
        low, high = max(lower[cell], low), min(upper[cell], high)
        if low > lower[cell] + _NULL_TOLERANCE or high < upper[cell] - _NULL_TOLERANCE:
            lower[cell], upper[cell] = low, high
            return True
        return False

    for _ in range(_MAX_SWEEPS):
        changed = False
        for coeffs, total, known in equations:
            finite_upper = sum(k * upper[c] for c, k in coeffs.items() if upper[c] < math.inf)
            n_infinite = sum(upper[c] == math.inf for c in coeffs)
            sum_lower = sum(k * lower[c] for c, k in coeffs.items())
            if total is None:
                t_low, t_high = known, known
            else:
                if total in unknown:
                    sum_upper = math.inf if n_infinite else finite_upper
                    changed |= tighten(total, sum_lower, sum_upper)
                t_low, t_high = lower[total], upper[total]
            for cell, k in coeffs.items():
                if cell not in unknown:
                    continue
                others_lower = sum_lower - k * lower[cell]
                if upper[cell] == math.inf:
                    others_upper = math.inf if n_infinite > 1 else finite_upper
                else:
                    others_upper = math.inf if n_infinite else finite_upper - k * upper[cell]
                low = (
                    -math.inf
                    if others_upper == math.inf
                    else math.ceil((t_low - others_upper) / k - _NULL_TOLERANCE)
                )
                high = (
                    math.inf
                    if t_high == math.inf
                    else math.floor((t_high - others_lower) / k + _NULL_TOLERANCE)
                )
                changed |= tighten(cell, low, high)
        if not changed:
            break
    return {cell: (lower[cell], upper[cell]) for cell in lower}


class Protection(NamedTuple):
    """The result of :func:`protect_cells`."""

    hidden: set[Hashable]
    """Published cells to suppress: the primary ones and every secondary one."""
    demoted: set[Hashable]
    """Of ``hidden``, primary cells to mark ``"*"`` rather than ``"<5"`` (rule (b))."""


def protect_cells(
    values: Mapping[Hashable, float],
    suppressed: Iterable[Hashable],
    relations: Sequence[SumRelation],
    *,
    hidden: Iterable[Hashable] = (),
    linked: Mapping[Hashable, Iterable[Hashable]] | None = None,
    proxies: Mapping[Hashable, Hashable] | None = None,
    sizes: Mapping[Hashable, float] | None = None,
    marked: Iterable[Hashable] = (),
) -> Protection:
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
        sizes: how much a published cell holds for choosing secondary cells (default: its
            value). For a count drawn from a total, ``min(count, total - count)``, so a
            table counting one side of a mask hides the same cells as one counting the other.
        marked: never-published cells a reader knows hold 1-4 because their proxy is shown
            ``"<5"`` (a complement of 1-4: ``"<5"`` marks a count or its complement).

    Rule, repeated until stable:

    (a) no suppressed cell (primary or secondary), and no never-published cell holding 1-4,
    may be exactly determined by the relations (checked by linear algebra, so chains of
    subtractions across overlapping groups are caught, not only a lone suppressed cell in
    one group). A violation hides the smallest non-zero published cell whose suppression
    alone frees the exposed cell, else the smallest non-zero published member of an
    offending relation.

    (b) no hidden cell holding 1-4 may be pinned by bounds. A reader takes every ``"<5"``
    (and every ``marked`` cell) as 1-4, every ``"*"`` as 0 or more, and tightens these
    intervals through the relations (:func:`_reader_bounds`). This covers a known total
    whose hidden members sum to their minimum or maximum (all ones, all fours), and a total
    that is itself ``"<5"``: four members each at least 1 under a total of at most 4 are
    all 1, and the total 4. A violation hides the smallest non-zero published cell of the
    connected relations whose suppression alone unpins the cell; failing that, relabels one
    primary cell ``"*"`` (``demoted``; its "at least 1" no longer holds, since ``"*"`` may be
    0), trying the totals of the relations through the pinned cell first, then their
    members. A relabelled cell stays hidden and its complement loses its mark.

    Size is ``sizes`` (ties broken by the order of the relation's members); zeros, then
    totals, are hidden only as a last resort. When nothing publishable is left to hide or
    relabel, the cell is fixed by published totals alone and suppression cannot help; it
    is left as is.

    Returns:
        The published cells to suppress (``suppressed`` and everything added), and those of
        them to mark ``"*"`` instead of ``"<5"``.
    """
    links = dict(linked or {})
    prox = dict(proxies or {})
    never_shown = set(hidden)
    size = dict(values) | dict(sizes or {})
    primary = {c for c in suppressed if _small_value(values.get(c, 0.0))}
    marks = {c for c in marked if c in never_shown}
    demoted: set[Hashable] = set()
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
            if shown is not None and (size.get(shown, 0.0) == 0) == zeros:
                options.append((size.get(shown, 0.0), position, shown))
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
                pool = sorted(dict.fromkeys(pool), key=lambda c: size.get(c, 0.0))
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

    def small_hints(relabelled: set) -> set:
        # Hidden cells a reader knows hold 1-4: "<5" primaries and marked complements.
        return {c for c in primary if c not in relabelled} | {
            c for c in marks if prox.get(c) not in relabelled
        }

    def pinned(unknown: set, determined: set, relabelled: set, relations: Sequence) -> set:
        free = unknown - determined
        bounds = _reader_bounds(relations, values, free, small_hints(relabelled) & free)
        return {
            c
            for c, (low, high) in bounds.items()
            if c in free and low == high and _small_value(values.get(c, 0.0))
        }

    def relabel_options(order: list[SumRelation], touching: list[SumRelation]) -> list:
        # Totals of the relations through the pinned cell first, then their members, then
        # the rest of the connected relations; each as the published cell to relabel.
        cells: list[Hashable] = [r.total for r in touching if r.total is not None]
        for relation in [*touching, *order]:
            cells += [*relation.members, *([] if relation.total is None else [relation.total])]
        hinted = small_hints(demoted)
        options = []
        for cell in dict.fromkeys(cells):
            shown = prox.get(cell) if cell in never_shown else cell
            if shown is None or shown not in supp or shown in demoted or cell not in hinted:
                continue
            options.append(shown)
        return list(dict.fromkeys(options))

    def unpin(cell: Hashable, unknown: set) -> bool:
        touching = [by_id[k] for k in dict.fromkeys(by_cell.get(cell, ()))]
        order = reachable(touching, unknown)

        def still_pinned(trial_supp: set, relabelled: set) -> bool:
            # Cells determined before may not be after hiding more; propagation finds the
            # simple ones again, and the main loop rechecks every choice in full.
            return cell in pinned(trial_supp | never_shown, set(), relabelled, order)

        pool = [c for r in order for c in members_by_size(r, False)]
        for option in sorted(dict.fromkeys(pool), key=lambda c: size.get(c, 0.0)):
            if not still_pinned(supp | _closure([option], links), demoted):
                hide(option)
                return True
        options = relabel_options(order, touching)
        for option in options:
            if not still_pinned(supp, demoted | (_closure([option], links) & supp)):
                demoted.update(_closure([option], links) & supp)
                return True
        if options:
            demoted.update(_closure([options[0]], links) & supp)
            return True
        return remedy(touching, unknown, None)

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
        pins = pinned(unknown, determined, demoted, rels)
        cell = next((t for t in targets if t in pins and t not in fixed_by_totals), None)
        if cell is None:
            return Protection(supp - never_shown, demoted)
        if not unpin(cell, unknown):
            fixed_by_totals.add(cell)


@dataclass(frozen=True)
class TableSpec:
    """An aggregate table to suppress, with the arguments of :func:`suppress_table`."""

    df: pd.DataFrame
    count_columns: Sequence[str]
    linked: Mapping[str, Sequence[str]] | None = None
    complements: Mapping[str, str] | None = None
    groups: Sequence[SumRelation] | None = None


def suppress_table(
    df: pd.DataFrame,
    count_columns: Sequence[str],
    linked: Mapping[str, Sequence[str]] | None = None,
    complements: Mapping[str, str] | None = None,
    groups: Sequence[SumRelation] | None = None,
) -> pd.DataFrame:
    """Primary then secondary suppression of an aggregate table.

    Primary is :func:`suppress_small_cells` (marked ``"<5"``). Secondary treats each
    ``group`` (row labels of ``df``; default: the whole table, whose total is taken as
    known; ``[]``: none) as a sum a reader can form in every count column, and in the
    complement ``total - count`` of every ``complements`` column, then hides further cells
    (with their linked columns; marked ``"*"``, since they may hold any value) until no
    suppressed cell or small complement is recoverable (:func:`protect_cells`). A primary
    cell whose ``"<5"`` would itself pin a cell (rule (b)) is marked ``"*"`` too.
    """
    spec = TableSpec(df, count_columns, linked, complements, groups)
    return suppress_tables({None: spec})[None]


@dataclass(frozen=True)
class DerivedCell:
    """A count never published but formable from published cells through ``cross`` sums.

    For example the difference of two published counts of nested masks. Like a table's
    complement it must stay unknown when it holds 1-4. ``proxy`` is the published cell
    ``(table key, row label, count column)`` whose suppression hides it.
    """

    value: float
    proxy: Hashable


def suppress_tables(
    tables: Mapping[Hashable, TableSpec],
    cross: Sequence[SumRelation] = (),
    derived: Mapping[Hashable, DerivedCell] | None = None,
) -> dict[Hashable, pd.DataFrame]:
    """:func:`suppress_table` over several tables at once, protected jointly.

    ``cross`` are sums a reader can form across tables; their cells are
    ``(table key, row label, count column)`` or keys of ``derived``. A secondary cell hidden
    to protect one table may sit in another, so the tables are safe to publish only
    together, as returned.

    ``derived``: never-published counts that ``cross`` ties to published cells (see
    :class:`DerivedCell`); each one holding 1-4 is kept undetermined. Keys must not clash
    with table cells.
    """
    values: dict[Hashable, float] = {}
    sizes: dict[Hashable, float] = {}
    suppressed: list[Hashable] = []
    cell_links: dict[Hashable, list[Hashable]] = {}
    relations: list[SumRelation] = []
    hidden: list[Hashable] = []
    marked: list[Hashable] = []
    proxies: dict[Hashable, Hashable] = {}
    primaries: dict[Hashable, pd.DataFrame] = {}
    for key, spec in tables.items():
        df = spec.df
        if not df.index.is_unique:
            raise ValueError("suppress_table needs a unique row index")
        primary = suppress_small_cells(df, spec.count_columns, spec.linked, spec.complements)
        primaries[key] = primary
        links = {col: list(targets) for col, targets in (spec.linked or {}).items()}
        counts = set(spec.count_columns)
        rows = list(df.index)
        row_groups = list(spec.groups) if spec.groups is not None else [SumRelation(tuple(rows))]
        numeric = {col: pd.to_numeric(df[col], errors="coerce").fillna(0.0) for col in counts}
        for col in spec.count_columns:
            for row in rows:
                values[(key, row, col)] = float(numeric[col][row])
                if isinstance(primary.at[row, col], str) and primary.at[row, col] == SUPPRESSED:
                    suppressed.append((key, row, col))
                cell_links[(key, row, col)] = [
                    (key, row, c) for c in links.get(col, ()) if c in counts
                ]
            for group in row_groups:
                total = None if group.total is None else (key, group.total, col)
                members = tuple((key, m, col) for m in group.members)
                relations.append(SumRelation(members, total))
        for col, total_col in (spec.complements or {}).items():
            totals = pd.to_numeric(df[total_col], errors="coerce").fillna(0.0)
            count = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            for row in rows:
                comp = ("~complement", key, row, col)
                values[comp] = float(totals[row] - count[row])
                sizes[(key, row, col)] = min(float(count[row]), values[comp])
                hidden.append(comp)
                proxies[comp] = (key, row, col)
                if _small_value(values[comp]):
                    marked.append(comp)  # its count is shown "<5"
                total_cell = (key, row, total_col) if total_col in counts else None
                relations.append(SumRelation(((key, row, col), comp), total_cell))
            for group in row_groups:
                comp_total = None if group.total is None else ("~complement", key, group.total, col)
                comps = tuple(("~complement", key, m, col) for m in group.members)
                relations.append(SumRelation(comps, comp_total))
    for cell_key, cell in (derived or {}).items():
        values[cell_key] = float(cell.value)
        hidden.append(cell_key)
        proxies[cell_key] = cell.proxy
    relations.extend(cross)
    final = protect_cells(
        values,
        suppressed,
        relations,
        hidden=hidden,
        linked=cell_links,
        proxies=proxies,
        sizes=sizes,
        marked=marked,
    )
    secondary: list[tuple[Hashable, Hashable, str]] = sorted(
        final.hidden - set(suppressed),  # type: ignore[arg-type]
        key=repr,
    )
    relabelled: list[tuple[Hashable, Hashable, str]] = sorted(
        final.demoted,  # type: ignore[arg-type]
        key=repr,
    )
    out: dict[Hashable, pd.DataFrame] = {}
    for key, spec in tables.items():
        safe = primaries[key].copy()
        links = {col: list(targets) for col, targets in (spec.linked or {}).items()}
        for table, row, col in secondary:
            if table != key:
                continue
            for target in (col, *links.get(col, ())):
                safe[target] = safe[target].astype(object)
                # A cell already hidden as primary keeps its "<5".
                if safe.at[row, target] != SUPPRESSED:
                    safe.at[row, target] = SECONDARY
        for table, row, col in relabelled:
            if table != key:
                continue
            # Rule (b) of protect_cells: "<5" would pin it or another cell.
            for target in (col, *links.get(col, ())):
                safe[target] = safe[target].astype(object)
                safe.at[row, target] = SECONDARY
        out[key] = safe
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
