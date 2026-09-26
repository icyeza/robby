"""Published reference values: Vogel 2015 Robson set, WHO C-Model, prevalence priors.

Spec §15.4, §15.5 and §15.3 (v1.2). **No reference value is ever estimated, approximated
or invented in this repository.** Each file below is transcribed by hand from the cited
source, with the exact table, supplement or calculator page of every value. When a file is
absent the loader raises :class:`ReferenceDataMissing`, naming the expected path and schema;
there is no fallback (spec §22, ``test_cmodel_no_fallback``). Tests build their own files in
temporary directories, labelled as fake.

Files live under ``data/reference/`` (the only part of ``data/`` that may be committed).
The schemas are :data:`VOGEL_SCHEMA`, :data:`CMODEL_SCHEMA` and :data:`PREVALENCE_SCHEMA`;
angle-bracket placeholders are to be replaced by the transcribed values.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

VOGEL_PATH = Path("data/reference/vogel2015_v1.yaml")
CMODEL_PATH = Path("data/reference/cmodel_v1.yaml")
PREVALENCE_PATH = Path("data/reference/prevalence_v1.yaml")
ROBSON_GROUPS = tuple(range(1, 11))
# Transcribed group sizes are rounded percentages; they must add up to 100 within this.
SIZE_SUM_TOLERANCE_PCT = 2.0
TERM_KINDS = ("indicator", "linear")

VOGEL_SCHEMA = """\
# data/reference/vogel2015_v1.yaml (spec §15.5): transcribe, never estimate.
citation: <full citation: Vogel JP et al. 2015, Lancet Glob Health 3(5):e260-e270>
version: v1
source_table: <table number in the paper, e.g. "Table N">
population: <which column / country grouping of that table was transcribed>
transcribed_by: <name>            # optional
transcribed_on: <YYYY-MM-DD>      # optional
groups:                           # all ten Robson groups, percentages (0-100)
  1: {group_size_pct: <float>, cs_rate_pct: <float>, source: <table, row and columns>}
  2: {group_size_pct: <float>, cs_rate_pct: <float>, source: <...>}
  # ... through 10
"""

CMODEL_SCHEMA = """\
# data/reference/cmodel_v1.yaml (spec §15.4): transcribe, never estimate or approximate.
citation: <full citation: Souza JP et al. 2016, BJOG 123(3):427-436>
version: v1
link: logit
intercept: {coefficient: <float>, source: <table / supplement / e-calculator page>}
variables:            # EVERY variable the C-Model needs, mapped to an analysis-frame column
  <c-model variable>: {column: <canonical or registry column name, or null if absent>,
                       source: <where the variable is defined>}
terms:                # one entry per coefficient
  - {name: <label>, variable: <c-model variable>, kind: indicator, equals: <value or list>,
     coefficient: <float>, source: <table / supplement, row>}
  - {name: <label>, variable: <c-model variable>, kind: indicator, lower: <float or null>,
     upper: <float or null>, coefficient: <float>, source: <...>}    # lower <= x < upper
  - {name: <label>, variable: <c-model variable>, kind: linear, centre: <float, default 0>,
     coefficient: <float>, source: <...>}
# A variable with column null (not in the data) makes the C-Model NOT APPLICABLE: it is
# reported as such, never approximated.
"""

PREVALENCE_SCHEMA = """\
# data/reference/prevalence_v1.yaml (spec §15.3 v1.2): assumed TRUE prevalence grid for the
# under-recording sensitivity analysis, anchored on published (regional) prevalence.
version: v1
conditions:
  preeclampsia:
    field: preeclampsia_recorded
    grid_pct: [<float>, <float>, ...]   # assumed true prevalence (%), strictly increasing
    anchors:                            # the published values the grid is built around
      - {value_pct: <float>, citation: <full citation>, source: <table / page>,
         population: <country / region / setting>}
  gestational_diabetes:
    field: gdm_recorded
    grid_pct: [...]
    anchors: [...]
"""

SCHEMAS = {
    "vogel": (VOGEL_PATH, VOGEL_SCHEMA),
    "cmodel": (CMODEL_PATH, CMODEL_SCHEMA),
    "prevalence": (PREVALENCE_PATH, PREVALENCE_SCHEMA),
}


class ReferenceDataMissing(FileNotFoundError):  # noqa: N818 (name fixed by the task spec)
    """A reference file is absent. Nothing is estimated in its place (spec §15.4)."""


class ReferenceSchemaError(ValueError):
    """A reference file exists but does not follow its documented schema."""


def _missing(path: Path, schema: str) -> ReferenceDataMissing:
    return ReferenceDataMissing(
        f"reference file not found: {path}. Reference values are transcribed from the cited "
        "source and never estimated, approximated or invented; create the file with this "
        f"schema:\n{schema}"
    )


def _read(path: Path, schema: str) -> tuple[dict[str, Any], str]:
    path = Path(path)
    if not path.is_file():
        raise _missing(path, schema)
    raw = path.read_bytes()
    data = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ReferenceSchemaError(f"{path}: expected a mapping at the top level")
    return data, hashlib.sha256(raw).hexdigest()


def _text(data: Mapping[str, Any], key: str, where: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ReferenceSchemaError(f"{where}: '{key}' must be a non-empty string")
    return value.strip()


def _version(data: Mapping[str, Any], where: str) -> str:
    value = data.get("version")
    if value is None or not str(value).strip():
        raise ReferenceSchemaError(f"{where}: 'version' is required")
    return str(value).strip()


def _number(data: Mapping[str, Any], key: str, where: str, low: float, high: float) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReferenceSchemaError(f"{where}: '{key}' must be a number")
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise ReferenceSchemaError(f"{where}: '{key}' must lie in [{low}, {high}]")
    return number


def _optional_number(data: Mapping[str, Any], key: str, where: str) -> float | None:
    if data.get(key) is None:
        return None
    return _number(data, key, where, -math.inf, math.inf)


@dataclass(frozen=True)
class VogelGroup:
    """One Robson group's reference size and CS rate, in percent, with its source."""

    group: int
    group_size_pct: float
    cs_rate_pct: float
    source: str


@dataclass(frozen=True)
class VogelReference:
    """The Vogel et al. (2015) reference set (spec §15.5)."""

    citation: str
    version: str
    source_table: str
    population: str
    groups: Mapping[int, VogelGroup]
    path: str
    sha256: str

    def size(self, group: int) -> float:
        """Reference share of deliveries in ``group`` (a fraction, 0-1)."""
        return self.groups[group].group_size_pct / 100.0

    def cs_rate(self, group: int) -> float:
        """Reference CS rate of ``group`` (a fraction, 0-1)."""
        return self.groups[group].cs_rate_pct / 100.0


def load_vogel(path: Path = VOGEL_PATH) -> VogelReference:
    """Load and validate the Vogel reference set; raises when absent (no fallback)."""
    data, sha = _read(path, VOGEL_SCHEMA)
    where = str(path)
    groups_raw = data.get("groups")
    if not isinstance(groups_raw, dict):
        raise ReferenceSchemaError(f"{where}: 'groups' must map Robson groups 1-10")
    keys: dict[int, Any] = {}
    for key, value in groups_raw.items():
        try:
            keys[int(key)] = value
        except (TypeError, ValueError):
            raise ReferenceSchemaError(f"{where}: group key {key!r} is not 1-10") from None
    if sorted(keys) != list(ROBSON_GROUPS):
        raise ReferenceSchemaError(f"{where}: 'groups' must hold exactly groups 1-10")
    groups: dict[int, VogelGroup] = {}
    for group, entry in sorted(keys.items()):
        at = f"{where}: group {group}"
        if not isinstance(entry, dict):
            raise ReferenceSchemaError(f"{at}: expected a mapping")
        groups[group] = VogelGroup(
            group,
            _number(entry, "group_size_pct", at, 0.0, 100.0),
            _number(entry, "cs_rate_pct", at, 0.0, 100.0),
            _text(entry, "source", at),
        )
    total = sum(g.group_size_pct for g in groups.values())
    if abs(total - 100.0) > SIZE_SUM_TOLERANCE_PCT:
        raise ReferenceSchemaError(
            f"{where}: group sizes add up to {total:.1f}%, not 100% "
            f"(tolerance {SIZE_SUM_TOLERANCE_PCT}%): check the transcription"
        )
    return VogelReference(
        citation=_text(data, "citation", where),
        version=_version(data, where),
        source_table=_text(data, "source_table", where),
        population=_text(data, "population", where),
        groups=groups,
        path=where,
        sha256=sha,
    )


@dataclass(frozen=True)
class CModelTerm:
    """One C-Model coefficient.

    ``indicator``: adds ``coefficient`` when the variable equals ``equals`` (a value or any
    of a list) or lies in ``[lower, upper)``. ``linear``: adds
    ``coefficient * (value - centre)``.
    """

    name: str
    variable: str
    kind: str
    coefficient: float
    source: str
    equals: tuple[object, ...] | None = None
    lower: float | None = None
    upper: float | None = None
    centre: float = 0.0


@dataclass(frozen=True)
class CModelReference:
    """WHO C-Model coefficients (spec §15.4). ``variables`` maps each C-Model variable to
    the analysis-frame column holding it, or ``None`` when the data has no such column."""

    citation: str
    version: str
    link: str
    intercept: float
    intercept_source: str
    variables: Mapping[str, str | None]
    terms: tuple[CModelTerm, ...]
    path: str
    sha256: str

    def columns(self) -> list[str]:
        """The frame columns the model reads (variables mapped to a column)."""
        return sorted({c for c in self.variables.values() if c is not None})


def _term(entry: Any, index: int, variables: Mapping[str, Any], where: str) -> CModelTerm:
    at = f"{where}: term {index}"
    if not isinstance(entry, dict):
        raise ReferenceSchemaError(f"{at}: expected a mapping")
    variable = _text(entry, "variable", at)
    if variable not in variables:
        raise ReferenceSchemaError(f"{at}: variable '{variable}' is not listed in 'variables'")
    kind = _text(entry, "kind", at)
    if kind not in TERM_KINDS:
        raise ReferenceSchemaError(f"{at}: kind must be one of {TERM_KINDS}")
    coefficient = _number(entry, "coefficient", at, -math.inf, math.inf)
    source = _text(entry, "source", at)
    name = _text(entry, "name", at)
    if kind == "linear":
        centre = _optional_number(entry, "centre", at)
        return CModelTerm(name, variable, kind, coefficient, source, centre=centre or 0.0)
    has_equals = "equals" in entry and entry["equals"] is not None
    lower, upper = _optional_number(entry, "lower", at), _optional_number(entry, "upper", at)
    if has_equals == (lower is not None or upper is not None):
        raise ReferenceSchemaError(f"{at}: an indicator needs either 'equals' or lower/upper")
    if lower is not None and upper is not None and not lower < upper:
        raise ReferenceSchemaError(f"{at}: lower must be below upper")
    equals = None
    if has_equals:
        value = entry["equals"]
        equals = tuple(value) if isinstance(value, list) else (value,)
        if not equals:
            raise ReferenceSchemaError(f"{at}: 'equals' must not be empty")
    return CModelTerm(name, variable, kind, coefficient, source, equals, lower, upper)


def load_cmodel(path: Path = CMODEL_PATH) -> CModelReference:
    """Load and validate the C-Model coefficients; raises when absent (no fallback)."""
    data, sha = _read(path, CMODEL_SCHEMA)
    where = str(path)
    link = _text(data, "link", where)
    if link != "logit":
        raise ReferenceSchemaError(f"{where}: only link 'logit' is supported")
    intercept = data.get("intercept")
    if not isinstance(intercept, dict):
        raise ReferenceSchemaError(f"{where}: 'intercept' must be a mapping")
    variables_raw = data.get("variables")
    if not isinstance(variables_raw, dict) or not variables_raw:
        raise ReferenceSchemaError(f"{where}: 'variables' must list every C-Model variable")
    variables: dict[str, str | None] = {}
    for name, entry in variables_raw.items():
        at = f"{where}: variable {name}"
        if not isinstance(entry, dict) or "column" not in entry:
            raise ReferenceSchemaError(f"{at}: expected {{column: ..., source: ...}}")
        _text(entry, "source", at)
        column = entry["column"]
        if column is not None and (not isinstance(column, str) or not column.strip()):
            raise ReferenceSchemaError(f"{at}: column must be a name or null")
        variables[str(name)] = None if column is None else column.strip()
    terms_raw = data.get("terms")
    if not isinstance(terms_raw, list) or not terms_raw:
        raise ReferenceSchemaError(f"{where}: 'terms' must list the coefficients")
    terms = tuple(_term(entry, i, variables, where) for i, entry in enumerate(terms_raw))
    unused = sorted(set(variables) - {t.variable for t in terms})
    if unused:
        raise ReferenceSchemaError(f"{where}: variables without a term: {unused}")
    return CModelReference(
        citation=_text(data, "citation", where),
        version=_version(data, where),
        link=link,
        intercept=_number(intercept, "coefficient", f"{where}: intercept", -math.inf, math.inf),
        intercept_source=_text(intercept, "source", f"{where}: intercept"),
        variables=variables,
        terms=terms,
        path=where,
        sha256=sha,
    )


@dataclass(frozen=True)
class PrevalenceAnchor:
    """A published prevalence the grid is anchored on."""

    value_pct: float
    citation: str
    source: str
    population: str


@dataclass(frozen=True)
class ConditionPrior:
    """The assumed true-prevalence grid of one under-recorded condition."""

    condition: str
    field: str
    grid_pct: tuple[float, ...]
    anchors: tuple[PrevalenceAnchor, ...]


@dataclass(frozen=True)
class PrevalenceReference:
    """Prevalence grids for the under-recording sensitivity analysis (spec §15.3 v1.2)."""

    version: str
    conditions: Mapping[str, ConditionPrior]
    path: str
    sha256: str


def _grid(value: Any, where: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ReferenceSchemaError(f"{where}: 'grid_pct' must be a non-empty list")
    grid = tuple(_number({"grid_pct": v}, "grid_pct", where, 0.0, 100.0) for v in value)
    if any(v <= 0.0 or v >= 100.0 for v in grid):
        raise ReferenceSchemaError(f"{where}: grid values must lie strictly between 0 and 100")
    if any(b <= a for a, b in pairwise(grid)):
        raise ReferenceSchemaError(f"{where}: 'grid_pct' must be strictly increasing")
    return grid


def load_prevalence(path: Path = PREVALENCE_PATH) -> PrevalenceReference:
    """Load and validate the prevalence grids; raises when absent (no default grid)."""
    data, sha = _read(path, PREVALENCE_SCHEMA)
    where = str(path)
    conditions_raw = data.get("conditions")
    if not isinstance(conditions_raw, dict) or not conditions_raw:
        raise ReferenceSchemaError(f"{where}: 'conditions' must be a non-empty mapping")
    conditions: dict[str, ConditionPrior] = {}
    for name, entry in conditions_raw.items():
        at = f"{where}: condition {name}"
        if not isinstance(entry, dict):
            raise ReferenceSchemaError(f"{at}: expected a mapping")
        grid = _grid(entry.get("grid_pct"), at)
        anchors_raw = entry.get("anchors")
        if not isinstance(anchors_raw, list) or not anchors_raw:
            raise ReferenceSchemaError(f"{at}: at least one published anchor is required")
        anchors: list[PrevalenceAnchor] = []
        for i, anchor in enumerate(anchors_raw):
            a_at = f"{at}: anchor {i}"
            if not isinstance(anchor, dict):
                raise ReferenceSchemaError(f"{a_at}: expected a mapping")
            value = _number(anchor, "value_pct", a_at, 0.0, 100.0)
            if not grid[0] <= value <= grid[-1]:
                raise ReferenceSchemaError(f"{a_at}: the grid must span every anchor")
            anchors.append(
                PrevalenceAnchor(
                    value,
                    _text(anchor, "citation", a_at),
                    _text(anchor, "source", a_at),
                    _text(anchor, "population", a_at),
                )
            )
        conditions[str(name)] = ConditionPrior(
            str(name), _text(entry, "field", at), grid, tuple(anchors)
        )
    return PrevalenceReference(_version(data, where), conditions, where, sha)


def reference_status(paths: Sequence[tuple[str, Path]]) -> list[dict[str, str]]:
    """For display: which reference files exist (``name``, ``path``, ``status``)."""
    return [
        {"reference": name, "path": str(path), "status": "present" if path.is_file() else "absent"}
        for name, path in paths
    ]
