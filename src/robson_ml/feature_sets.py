"""Model input frames and the FS0-FS4 / FS4_deploy feature sets (spec §9.1, §9.2, §4.5).

Only registry ``status: include`` features reach a model (spec §8.1): the input frame holds
the included canonical columns, the included derived columns (``bmi``, ``robson_group``,
``is_missing_<field>``) and the included raw extra features, plus ``facility_id``, which is
used only by FS4_deploy. Cleaning here is row-wise and stateless (dtype normalisation and
derivations); every fitted transformation (imputation, encoding, scaling) happens inside the
model Pipeline, within folds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from robson_ml.features import FeatureEntry, FeatureRegistry
from robson_ml.populations import ExclusionLog, audit_population, prediction_population

FACILITY = "facility_id"
POPULATIONS = ("P_pred",)
# A registry entry naming a canonical concept stored in more than one canonical column.
CANONICAL_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "gestational_age_band": ("ga_band_lower", "ga_band_upper"),
}
CANONICAL_CATEGORICAL = frozenset(
    {"fetal_presentation", "onset_of_labour", "preeclampsia_recorded", "gdm_recorded"}
)
DERIVED_CATEGORICAL = frozenset({"robson_group"})
RAW_CATEGORICAL_KINDS = frozenset({"text", "category"})
RAW_NUMERIC_KINDS = frozenset({"integer", "integer_sum", "float", "gestational_age"})
MISSING_PREFIX = "is_missing_"
GROUP_ORDER = ("G_robson", "G_maternal", "G_obs", "G_missing", "G_context")
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "FS0": GROUP_ORDER[:1],
    "FS1": GROUP_ORDER[:2],
    "FS2": GROUP_ORDER[:3],
    "FS3": GROUP_ORDER[:4],
    "FS4": GROUP_ORDER[:5],
    "FS4_deploy": GROUP_ORDER[:5],
}
DEPLOY_SETS = frozenset({"FS4_deploy"})


@dataclass(frozen=True)
class FeatureSpec:
    """The columns a model may read for one feature set, partitioned by encoding type.

    ``ordinal`` holds numerically coded ordered categories (none in features_v1: the only
    one the spec names, proteinuria, is absent from the export, v1.2).
    """

    name: str
    columns: tuple[str, ...]
    categorical: tuple[str, ...]
    numeric: tuple[str, ...]
    ordinal: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelData:
    """A prediction population ready for the harness; all frames share a 0..n-1 index.

    ``meta`` holds the population's canonical columns (splits, subgroups, row ids; never
    fed to a model), ``x`` the model input columns (included features plus ``facility_id``),
    ``y`` the outcome, ``kinds`` each ``x`` column's type (categorical or numeric) and
    ``groups`` each column's feature group. ``exclusion_table`` holds, per onset category
    excluded from ``P_pred``, the rows and CS rows removed and the CS total of ``P_audit``
    (raw counts: suppress before any export; spec §4.4 reporting obligation).
    """

    population: str
    meta: pd.DataFrame
    x: pd.DataFrame
    y: npt.NDArray[np.int64]
    exclusions: list[ExclusionLog]
    kinds: dict[str, str]
    groups: dict[str, str]
    exclusion_table: pd.DataFrame


def _entry_columns(entry: FeatureEntry) -> tuple[str, ...]:
    return CANONICAL_EXPANSIONS.get(entry.name, (entry.name,))


def allowed_columns(registry: FeatureRegistry) -> set[str]:
    """Every model column the registry's ``include`` entries can produce."""
    return {col for entry in registry.included() for col in _entry_columns(entry)}


def _numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        values.to_numpy(dtype=np.float64, na_value=np.nan), index=series.index, dtype=np.float64
    )


def _categorical(series: pd.Series) -> pd.Series:
    """Object dtype holding strings, with ``np.nan`` for missing (integers keep no ``.0``)."""

    def label(value: object) -> object:
        if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
            return np.nan
        if isinstance(value, float | np.floating) and float(value).is_integer():
            return str(int(value))
        return str(value)

    return pd.Series([label(v) for v in series.astype(object)], index=series.index, dtype=object)


def _population(
    frame: pd.DataFrame, population: str
) -> tuple[pd.DataFrame, list[ExclusionLog], pd.DataFrame]:
    if population not in POPULATIONS:
        raise ValueError(f"unsupported population {population!r}; expected one of {POPULATIONS}")
    audit, audit_log = audit_population(frame)
    pred, pred_logs = prediction_population(audit)
    onset, cs_type = audit["onset_of_labour"], audit["prelabour_cs_type"]
    prelabour = onset == "prelabour_cs"
    categories = {
        "prelabour_cs planned": prelabour & (cs_type == "planned"),
        "prelabour_cs emergency": prelabour & (cs_type == "emergency"),
        "prelabour_cs type unknown": prelabour & cs_type.isna(),
        "onset missing": onset.isna(),
    }
    cs = audit["cs"].astype(int)
    table = pd.DataFrame(
        [
            {
                "category": name,
                "n_excluded": int(mask.sum()),
                "n_cs_excluded": int(cs[mask].sum()),
                "n_cs_audit": int(cs.sum()),
            }
            for name, mask in categories.items()
        ]
    )
    return pred.reset_index(drop=True), [audit_log, *pred_logs], table


def _raw_kind(entry: FeatureEntry) -> str:
    kind = entry.mapping.kind if entry.mapping is not None else None
    if kind in RAW_CATEGORICAL_KINDS:
        return "categorical"
    if kind in RAW_NUMERIC_KINDS:
        return "numeric"
    raise ValueError(f"{entry.name}: raw feature kind {kind!r} cannot be a model input")


def build_model_data(
    canonical: pd.DataFrame,
    registry: FeatureRegistry,
    raw_features: pd.DataFrame,
    population: str = "P_pred",
) -> ModelData:
    """Assemble the model input frame for ``population`` (spec §4.4, §9).

    Inputs: the canonical frame with the Robson engine's columns (``canonical_robson``);
    the registry; ``raw_features`` from :func:`build_raw_features`, aligned with
    ``canonical`` by row position (it must hold every included raw feature). Derived
    features: ``bmi`` only when height and weight are both present; ``robson_group`` as a
    category (missing when partial or conflict); ``is_missing_<field>`` as 0/1.
    """
    if len(raw_features) != len(canonical):
        raise ValueError("raw_features must align with canonical by row position")
    included = registry.included()
    raw_entries = [e for e in included if e.source == "raw"]
    absent = [e.name for e in raw_entries if e.name not in raw_features.columns]
    if absent:
        raise ValueError(f"raw feature(s) missing from raw_features: {absent}")

    frame = canonical.reset_index(drop=True).copy()
    for entry in raw_entries:
        if entry.name in frame.columns:
            raise ValueError(f"raw feature {entry.name!r} clashes with a canonical column")
        frame[entry.name] = raw_features[entry.name].reset_index(drop=True)
    meta, exclusions, exclusion_table = _population(frame, population)

    x: dict[str, pd.Series] = {}
    kinds: dict[str, str] = {}
    groups: dict[str, str] = {}
    for group in GROUP_ORDER:
        for entry in (e for e in included if e.group == group):
            for column, series, kind in _columns_for(entry, meta):
                x[column], kinds[column], groups[column] = series, kind, group
    x[FACILITY] = _categorical(meta[FACILITY])
    kinds[FACILITY], groups[FACILITY] = "categorical", "G_context"
    y = np.asarray(meta["cs"].astype(int).to_numpy(), dtype=np.int64)
    return ModelData(
        population, meta, pd.DataFrame(x), y, exclusions, kinds, groups, exclusion_table
    )


def _columns_for(entry: FeatureEntry, meta: pd.DataFrame) -> list[tuple[str, pd.Series, str]]:
    """(column, cleaned series, kind) for one included registry entry."""
    if entry.source == "raw":
        kind = _raw_kind(entry)
        clean = _categorical if kind == "categorical" else _numeric
        return [(entry.name, clean(meta[entry.name]), kind)]
    if entry.source == "canonical":
        out = []
        for column in _entry_columns(entry):
            if column not in meta.columns:
                raise ValueError(f"canonical column {column!r} is missing")
            if column in CANONICAL_CATEGORICAL:
                out.append((column, _categorical(meta[column]), "categorical"))
            else:
                out.append((column, _numeric(meta[column]), "numeric"))
        return out
    return [(entry.name, _derived(entry.name, meta), _derived_kind(entry.name))]


def _derived_kind(name: str) -> str:
    return "categorical" if name in DERIVED_CATEGORICAL else "numeric"


def _derived(name: str, meta: pd.DataFrame) -> pd.Series:
    if name == "bmi":
        height, weight = _numeric(meta["height_cm"]), _numeric(meta["weight_kg"])
        return weight / (height / 100.0) ** 2  # NaN unless both are present
    if name == "robson_group":
        return _categorical(meta["robson_group"])
    if name.startswith(MISSING_PREFIX):
        field = name.removeprefix(MISSING_PREFIX)
        if field not in meta.columns:
            raise ValueError(f"{name}: source field {field!r} is not in the frame")
        return meta[field].isna().astype(np.float64)
    raise ValueError(f"no derivation defined for derived feature {name!r}")


def feature_spec(data: ModelData, name: str) -> FeatureSpec:
    """The FeatureSpec for feature set ``name`` over ``data`` (spec §9.2).

    FS0-FS4 never contain ``facility_id`` (spec §4.5); FS4_deploy is FS4 plus it. In v1.2
    G_context holds no feature other than facility, so FS4 equals FS3.
    """
    if name not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {name!r}; expected one of {list(FEATURE_SETS)}")
    wanted = set(FEATURE_SETS[name])
    columns = [c for c in data.x.columns if c != FACILITY and data.groups[c] in wanted]
    if name in DEPLOY_SETS:
        columns.append(FACILITY)
    elif FACILITY in columns:
        raise AssertionError("facility_id must never be in FS0-FS4")
    categorical = tuple(c for c in columns if data.kinds[c] == "categorical")
    numeric = tuple(c for c in columns if data.kinds[c] == "numeric")
    return FeatureSpec(name, tuple(columns), categorical, numeric)
