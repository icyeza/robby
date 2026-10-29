"""Small-cell suppression and aggregate-only inspection helpers."""

from __future__ import annotations

import math
from collections import Counter, deque
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
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


class DisclosureError(RuntimeError):
    """Internal error: an output about to be published marks ``"<5"`` a cell that neither
    holds 1-4 nor has a complement of 1-4. ``"<5"`` must only ever mean that; every other
    protective hide is ``"*"``."""


def check_small_marker(shown: object, value: float, total: float | None = None) -> None:
    """Raise :class:`DisclosureError` if ``shown`` is ``"<5"`` but neither ``value`` nor
    ``total - value`` (when ``total`` is given) is 1-4."""
    if not (isinstance(shown, str) and shown == SUPPRESSED):
        return
    if _small_value(value) or (total is not None and _small_value(total - value)):
        return
    of = "" if total is None else f" out of {total:g}"
    raise DisclosureError(
        f'"{SUPPRESSED}" marks a count of {value:g}{of}; only a count (or complement) of 1-4 '
        f'may show "{SUPPRESSED}"'
    )


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
        disclose no individual).
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


def _unknown_count(relation: SumRelation, unknown: set) -> int:
    return len({*relation.members, relation.total} & unknown)


class _SecondarySuppression:
    """The working state of :func:`protect_cells`: the cells hidden so far, the relations a
    reader can form, and the two remedies (hide one more published cell, or relabel a
    primary ``"*"``). :meth:`run` applies rules (a) and (b) until nothing is exposed."""

    def __init__(
        self,
        values: Mapping[Hashable, float],
        suppressed: Sequence[Hashable],
        relations: Sequence[SumRelation],
        hidden: Sequence[Hashable],
        linked: Mapping[Hashable, Iterable[Hashable]],
        proxies: Mapping[Hashable, Hashable],
        sizes: Mapping[Hashable, float],
        marked: Iterable[Hashable],
    ) -> None:
        self.values = values
        self.links = dict(linked)
        self.prox = dict(proxies)
        self.never_shown = set(hidden)
        self.size = dict(values) | dict(sizes)
        self.primary = {c for c in suppressed if _small_value(values.get(c, 0.0))}
        self.marks = {c for c in marked if c in self.never_shown}
        self.demoted: set[Hashable] = set()
        # A single member with an external total is simply published elsewhere; nothing here
        # can protect it, and its own table must.
        self.rels = [r for r in relations if len(r.members) >= 2 or r.total is not None]
        self.by_id = {id(r): r for r in self.rels}
        self.by_cell: dict[Hashable, list[int]] = {}
        for relation in self.rels:
            for cell in {*relation.members, relation.total} - {None}:
                self.by_cell.setdefault(cell, []).append(id(relation))
        self.supp = _closure(suppressed, self.links)
        # Checked in a fixed order so the result is deterministic.
        self.targets = dict.fromkeys(c for c in suppressed if c in self.supp)
        self.targets.update(dict.fromkeys(sorted(self.supp - set(self.targets), key=repr)))
        self.targets.update(dict.fromkeys(c for c in hidden if _small_value(values.get(c, 0.0))))

    def run(self) -> Protection:
        """Hide or relabel cells until no target is determined (a) or pinned (b)."""
        fixed_by_totals: set[Hashable] = set()
        while True:
            unknown = self.supp | self.never_shown
            determined = _determined(unknown, self.rels)
            exposed = next(
                (t for t in self.targets if t in determined and t not in fixed_by_totals), None
            )
            if exposed is not None:
                touching = self.touching(exposed)
                touching.sort(key=lambda r: (_unknown_count(r, unknown), len(r.members)))
                if not self.remedy(touching, unknown, exposed):
                    fixed_by_totals.add(exposed)
                continue
            pins = self.pinned(unknown, determined, self.demoted, self.rels)
            cell = next((t for t in self.targets if t in pins and t not in fixed_by_totals), None)
            if cell is None:
                return Protection(self.supp - self.never_shown, self.demoted)
            if not self.unpin(cell, unknown):
                fixed_by_totals.add(cell)

    def publishable(self, cell: Hashable) -> Hashable | None:
        """The published cell that stands for ``cell``, if it is still shown."""
        shown = self.prox.get(cell) if cell in self.never_shown else cell
        if shown is None or shown in self.supp or shown in self.never_shown:
            return None
        return shown

    def members_by_size(self, relation: SumRelation, zeros: bool) -> list[Hashable]:
        """Publishable members of ``relation``, smallest first (zeros only with ``zeros``)."""
        # Sized by the published cell's own value (a complement stands in for its count).
        options = []
        for position, cell in enumerate(relation.members):
            shown = self.publishable(cell)
            if shown is not None and (self.size.get(shown, 0.0) == 0) == zeros:
                options.append((self.size.get(shown, 0.0), position, shown))
        return [shown for *_, shown in sorted(options, key=lambda o: (o[0], o[1]))]

    def touching(self, cell: Hashable) -> list[SumRelation]:
        """The relations through ``cell``, in the order they were given."""
        return [self.by_id[k] for k in dict.fromkeys(self.by_cell.get(cell, ()))]

    def reachable(self, start: list[SumRelation], unknown: set) -> list[SumRelation]:
        """Breadth-first from ``start`` through relations sharing unknown cells."""
        seen: set[int] = set()
        order: list[SumRelation] = []
        queue = deque(id(r) for r in start)
        while queue:
            key = queue.popleft()
            if key in seen:
                continue
            seen.add(key)
            relation = self.by_id[key]
            order.append(relation)
            for cell in {*relation.members, relation.total} & unknown:
                queue.extend(k for k in self.by_cell.get(cell, ()) if k not in seen)
        return order

    def hide(self, cell: Hashable) -> None:
        """Suppress ``cell`` and its linked cells, which become targets themselves."""
        added = _closure([cell], self.links) - self.supp
        self.supp.update(added)
        self.targets.update(dict.fromkeys(sorted(added, key=repr)))

    def remedy(self, start: list[SumRelation], unknown: set, exposed: Hashable | None) -> bool:
        """Rule (a): hide one more published cell of the relations reachable from ``start``;
        False when nothing publishable is left."""
        order = self.reachable(start, unknown)
        for zeros in (False, True):
            if exposed is not None and self._hide_smallest_freeing(order, zeros, exposed):
                return True
            for relation in order:
                sized = self.members_by_size(relation, zeros)
                if sized:
                    self.hide(sized[0])
                    return True
        for relation in order:
            total = None if relation.total is None else self.publishable(relation.total)
            if total is not None:
                self.hide(total)
                return True
        return False

    def _hide_smallest_freeing(
        self, order: list[SumRelation], zeros: bool, exposed: Hashable
    ) -> bool:
        # Prefer the smallest cell whose suppression alone un-determines the target.
        pool = [c for r in order for c in self.members_by_size(r, zeros)]
        for cell in sorted(dict.fromkeys(pool), key=lambda c: self.size.get(c, 0.0)):
            trial = self.supp | _closure([cell], self.links) | self.never_shown
            if exposed not in _determined(trial, self.rels):
                self.hide(cell)
                return True
        return False

    def small_hints(self, relabelled: set) -> set:
        """Hidden cells a reader knows hold 1-4: "<5" primaries and marked complements."""
        return {c for c in self.primary if c not in relabelled} | {
            c for c in self.marks if self.prox.get(c) not in relabelled
        }

    def pinned(
        self, unknown: set, determined: set, relabelled: set, relations: Sequence[SumRelation]
    ) -> set:
        """Rule (b): unknown, undetermined cells holding 1-4 whose reader bounds meet."""
        free = unknown - determined
        bounds = _reader_bounds(relations, self.values, free, self.small_hints(relabelled) & free)
        return {
            c
            for c, (low, high) in bounds.items()
            if c in free and low == high and _small_value(self.values.get(c, 0.0))
        }

    def relabel_options(
        self, order: list[SumRelation], touching: list[SumRelation]
    ) -> list[Hashable]:
        """Primary cells that could show ``"*"`` instead of ``"<5"``: the totals of the
        relations through the pinned cell first, then their members, then the rest of the
        connected relations; each as the published cell to relabel."""
        cells: list[Hashable] = [r.total for r in touching if r.total is not None]
        for relation in [*touching, *order]:
            cells += [*relation.members, *([] if relation.total is None else [relation.total])]
        hinted = self.small_hints(self.demoted)
        options = []
        for cell in dict.fromkeys(cells):
            shown = self.prox.get(cell) if cell in self.never_shown else cell
            if shown is None or shown not in self.supp or shown in self.demoted:
                continue
            if cell in hinted:
                options.append(shown)
        return list(dict.fromkeys(options))

    def unpin(self, cell: Hashable, unknown: set) -> bool:
        """Rule (b) remedies for a pinned ``cell``: hide the smallest cell that unpins it,
        else relabel a primary, else fall back to rule (a)'s remedy."""
        touching = self.touching(cell)
        order = self.reachable(touching, unknown)

        def still_pinned(trial_supp: set, relabelled: set) -> bool:
            # Cells determined before may not be after hiding more; propagation finds the
            # simple ones again, and the main loop rechecks every choice in full.
            return cell in self.pinned(trial_supp | self.never_shown, set(), relabelled, order)

        pool = [c for r in order for c in self.members_by_size(r, False)]
        for option in sorted(dict.fromkeys(pool), key=lambda c: self.size.get(c, 0.0)):
            if not still_pinned(self.supp | _closure([option], self.links), self.demoted):
                self.hide(option)
                return True
        options = self.relabel_options(order, touching)
        for option in options:
            relabelled = _closure([option], self.links) & self.supp
            if not still_pinned(self.supp, self.demoted | relabelled):
                self.demoted.update(relabelled)
                return True
        if options:
            self.demoted.update(_closure([options[0]], self.links) & self.supp)
            return True
        return self.remedy(touching, unknown, None)


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
    can be recovered from published totals.

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
    suppressed = list(suppressed)
    state = _SecondarySuppression(
        values,
        suppressed,
        relations,
        list(hidden),
        linked or {},
        proxies or {},
        sizes or {},
        marked,
    )
    return state.run()


@dataclass(frozen=True)
class TableSpec:
    """An aggregate table to suppress, with the arguments of :func:`suppress_table`.

    ``rank``: row labels in the order secondary cells are chosen among non-zero cells, in
    place of their size, so the choice does not depend on the values (rows not listed come
    after, by size). Zeros are still chosen only as a last resort.
    """

    df: pd.DataFrame
    count_columns: Sequence[str]
    linked: Mapping[str, Sequence[str]] | None = None
    complements: Mapping[str, str] | None = None
    groups: Sequence[SumRelation] | None = None
    rank: Sequence[Hashable] | None = None


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


@dataclass
class _CellProblem:
    """The tables of :func:`suppress_tables` as one :func:`protect_cells` problem; cells are
    ``(table key, row label, count column)``, complements ``("~complement", key, row, col)``."""

    values: dict[Hashable, float] = field(default_factory=dict)
    sizes: dict[Hashable, float] = field(default_factory=dict)
    suppressed: list[Hashable] = field(default_factory=list)
    links: dict[Hashable, list[Hashable]] = field(default_factory=dict)
    relations: list[SumRelation] = field(default_factory=list)
    hidden: list[Hashable] = field(default_factory=list)
    marked: list[Hashable] = field(default_factory=list)
    proxies: dict[Hashable, Hashable] = field(default_factory=dict)

    def add_table(self, key: Hashable, spec: TableSpec, primary: pd.DataFrame) -> None:
        """The count cells of one table, its row-group sums, its complements and its rank."""
        df = spec.df
        links = {col: list(targets) for col, targets in (spec.linked or {}).items()}
        counts = set(spec.count_columns)
        rows = list(df.index)
        row_groups = list(spec.groups) if spec.groups is not None else [SumRelation(tuple(rows))]
        numeric = {col: pd.to_numeric(df[col], errors="coerce").fillna(0.0) for col in counts}
        for col in spec.count_columns:
            for row in rows:
                self.values[(key, row, col)] = float(numeric[col][row])
                if isinstance(primary.at[row, col], str) and primary.at[row, col] == SUPPRESSED:
                    self.suppressed.append((key, row, col))
                self.links[(key, row, col)] = [
                    (key, row, c) for c in links.get(col, ()) if c in counts
                ]
            for group in row_groups:
                total = None if group.total is None else (key, group.total, col)
                members = tuple((key, m, col) for m in group.members)
                self.relations.append(SumRelation(members, total))
        for col, total_col in (spec.complements or {}).items():
            self._add_complements(key, df, col, total_col, counts, row_groups)
        if spec.rank is not None:
            self._rank(key, spec, rows)

    def _add_complements(
        self,
        key: Hashable,
        df: pd.DataFrame,
        col: str,
        total_col: str,
        counts: set[str],
        row_groups: list[SumRelation],
    ) -> None:
        # The never-published complement total - count of each row, and its group sums.
        totals = pd.to_numeric(df[total_col], errors="coerce").fillna(0.0)
        count = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        for row in df.index:
            comp = ("~complement", key, row, col)
            self.values[comp] = float(totals[row] - count[row])
            self.sizes[(key, row, col)] = min(float(count[row]), self.values[comp])
            self.hidden.append(comp)
            self.proxies[comp] = (key, row, col)
            if _small_value(self.values[comp]):
                self.marked.append(comp)  # its count is shown "<5"
            total_cell = (key, row, total_col) if total_col in counts else None
            self.relations.append(SumRelation(((key, row, col), comp), total_cell))
        for group in row_groups:
            comp_total = None if group.total is None else ("~complement", key, group.total, col)
            comps = tuple(("~complement", key, m, col) for m in group.members)
            self.relations.append(SumRelation(comps, comp_total))

    def _rank(self, key: Hashable, spec: TableSpec, rows: list[Hashable]) -> None:
        # Sizes from the fixed rank instead of the values; a zero stays a last resort.
        order = {label: i for i, label in enumerate(spec.rank or ())}
        for col in spec.count_columns:
            for row in rows:
                ranked = (key, row, col)
                size = self.sizes.get(ranked, self.values[ranked])
                self.sizes[ranked] = 0.0 if size == 0 else 1.0 + order.get(row, len(order) + size)

    def add_derived(self, derived: Mapping[Hashable, DerivedCell]) -> None:
        """Never-published counts tied to published cells by cross-table sums."""
        for cell_key, cell in derived.items():
            self.values[cell_key] = float(cell.value)
            self.hidden.append(cell_key)
            self.proxies[cell_key] = cell.proxy

    def protect(self, forced: Iterable[Hashable]) -> Protection:
        """:func:`protect_cells` on every table at once, ``forced`` cells hidden as well."""
        forced_cells = list(dict.fromkeys(forced))
        if missing := [cell for cell in forced_cells if cell not in self.values]:
            raise ValueError(f"forced cells not in any table: {missing}")
        primary = set(self.suppressed)
        return protect_cells(
            self.values,
            [*self.suppressed, *(cell for cell in forced_cells if cell not in primary)],
            self.relations,
            hidden=self.hidden,
            linked=self.links,
            proxies=self.proxies,
            sizes=self.sizes,
            marked=self.marked,
        )


def _mark_hidden(
    primary: pd.DataFrame,
    key: Hashable,
    spec: TableSpec,
    secondary: Sequence[tuple[Hashable, Hashable, str]],
    relabelled: Sequence[tuple[Hashable, Hashable, str]],
) -> pd.DataFrame:
    """Table ``key`` after primary suppression, with its secondary and relabelled cells (and
    their linked columns) shown ``"*"``."""
    safe = primary.copy()
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
    return safe


def suppress_tables(
    tables: Mapping[Hashable, TableSpec],
    cross: Sequence[SumRelation] = (),
    derived: Mapping[Hashable, DerivedCell] | None = None,
    forced: Iterable[Hashable] = (),
) -> dict[Hashable, pd.DataFrame]:
    """:func:`suppress_table` over several tables at once, protected jointly.

    ``cross`` are sums a reader can form across tables; their cells are
    ``(table key, row label, count column)`` or keys of ``derived``. A secondary cell hidden
    to protect one table may sit in another, so the tables are safe to publish only
    together, as returned.

    ``derived``: never-published counts that ``cross`` ties to published cells (see
    :class:`DerivedCell`); each one holding 1-4 is kept undetermined. Keys must not clash
    with table cells.

    ``forced``: cells ``(table key, row label, count column)`` to hide whatever their value
    (e.g. the same count is hidden where another file publishes it), marked ``"*"`` unless
    primary suppression marks them ``"<5"``. Like every hidden cell they are then protected
    from recovery.

    Before returning, every ``"<5"`` is checked to mark a count (or complement) of 1-4, or a
    value derived from one; :class:`DisclosureError` otherwise.
    """
    problem = _CellProblem()
    primaries: dict[Hashable, pd.DataFrame] = {}
    for key, spec in tables.items():
        if not spec.df.index.is_unique:
            raise ValueError("suppress_table needs a unique row index")
        primaries[key] = suppress_small_cells(
            spec.df, spec.count_columns, spec.linked, spec.complements
        )
        problem.add_table(key, spec, primaries[key])
    problem.add_derived(derived or {})
    problem.relations.extend(cross)
    final = problem.protect(forced)
    secondary: list[tuple[Hashable, Hashable, str]] = sorted(
        final.hidden - set(problem.suppressed),  # type: ignore[arg-type]
        key=repr,
    )
    relabelled: list[tuple[Hashable, Hashable, str]] = sorted(
        final.demoted,  # type: ignore[arg-type]
        key=repr,
    )
    out: dict[Hashable, pd.DataFrame] = {}
    for key, spec in tables.items():
        safe = _mark_hidden(primaries[key], key, spec, secondary, relabelled)
        _check_markers(safe, spec)
        out[key] = safe
    return out


def _check_markers(safe: pd.DataFrame, spec: TableSpec) -> None:
    """Raise :class:`DisclosureError` unless every ``"<5"`` in ``safe`` sits on a row where
    a count column it is (or is linked to) holds 1-4 or has a complement of 1-4."""
    df = spec.df
    comps = dict(spec.complements or {})
    owners: dict[str, list[str]] = {}
    for col in spec.count_columns:
        for target in dict.fromkeys((col, *(spec.linked or {}).get(col, ()))):
            owners.setdefault(target, []).append(col)
    for column in safe.columns:
        for row in safe.index[safe[column].astype(object).isin([SUPPRESSED])]:
            reasons: list[str] = []
            for col in owners.get(column, []):
                value = float(pd.to_numeric(df.at[row, col], errors="coerce"))
                total = None
                if col in comps:
                    total = float(pd.to_numeric(df.at[row, comps[col]], errors="coerce"))
                try:
                    check_small_marker(SUPPRESSED, value, total)
                except DisclosureError as error:
                    reasons.append(str(error))
                else:
                    break
            else:
                raise DisclosureError(
                    f"row {row!r}, column {column!r}: " + ("; ".join(reasons) or "not a count")
                )


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

    ``n_nonnull`` is suppressed whenever it itself is 1-4, and also when its
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
