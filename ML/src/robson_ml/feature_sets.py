"""Model input frames, the FS0-FS4 feature sets and their ``_deploy`` variants (spec §9, §4.5).

Only registry ``status: include`` features reach a model (spec §8.1): the input frame holds
the included canonical columns, the included derived columns (``bmi``,
``robson_group_no_onset``, ``is_missing_<field>``) and the included raw extra features, plus
``facility_id``, which is used only by the deploy variants (``FS<k>_deploy`` = ``FS<k>`` plus
``facility_id``; evaluated only under S3 and fitted only under S5). Cleaning here is
row-wise and stateless (dtype normalisation and derivations); every fitted transformation
(imputation, encoding, scaling) happens inside the model Pipeline, within folds.

Spec v1.3: the onset field is outcome-contaminated (coded retrospectively), so
``onset_of_labour`` and the onset-based ``robson_group`` never reach a model built on
``P_pred``. The Robson feature is ``robson_group_no_onset``: the engine run with onset
forced to missing, groups 1+2 merged to ``"1_2"`` and 3+4 to ``"3_4"``. The only exception
is the sensitivity population ``P_pred_onset_coded``, whose model data is built with
``legacy_onset=True``: ``robson_group_no_onset`` is then replaced by the v1.2 pair
``onset_of_labour`` + ``robson_group`` (the onset-coded analysis of spec v1.3 item 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from robson_engine import INPUT_FIELDS, ClassificationResult, RuleSet, classify, load_rule_set
from robson_ml.features import FeatureEntry, FeatureRegistry
from robson_ml.populations import (
    P_PRED,
    P_PRED_ONSET_CODED,
    ExclusionLog,
    audit_population,
    onset_coded_population,
    planned_cs_mask,
    planned_onset_vaginal_mask,
    prediction_population,
    untyped_cs_mask,
)
from robson_ml.robson_run import inputs_from_record
from robson_ml.schema import GA_BAND_FIELDS

FACILITY = "facility_id"
# Complete-case analysis: P_pred restricted to rows with no missing input in the feature set
# (:func:`complete_cases`); never built directly by :func:`build_model_data`.
P_PRED_COMPLETE = "P_pred_complete"
BASE_POPULATIONS = (P_PRED, P_PRED_ONSET_CODED)
POPULATIONS = (*BASE_POPULATIONS, P_PRED_COMPLETE)
ROBSON_NO_ONSET = "robson_group_no_onset"
# Outcome-contaminated (spec v1.3): in model data only for P_pred_onset_coded (legacy_onset).
LEGACY_ONSET_COLUMNS = ("onset_of_labour", "robson_group")
NO_ONSET_MERGES: dict[frozenset[int], str] = {
    frozenset({1, 2}): "1_2",
    frozenset({3, 4}): "3_4",
}
NO_ONSET_UNRESOLVED = "partial"
# A registry entry naming a canonical concept stored in more than one canonical column.
CANONICAL_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "gestational_age_band": ("ga_band_lower", "ga_band_upper"),
}
CANONICAL_CATEGORICAL = frozenset(
    {"fetal_presentation", "onset_of_labour", "preeclampsia_recorded", "gdm_recorded"}
)
DERIVED_CATEGORICAL = frozenset({"robson_group", ROBSON_NO_ONSET})
RAW_CATEGORICAL_KINDS = frozenset({"text", "category"})
RAW_NUMERIC_KINDS = frozenset({"integer", "integer_sum", "float", "gestational_age"})
MISSING_PREFIX = "is_missing_"
GROUP_ORDER = ("G_robson", "G_maternal", "G_obs", "G_missing", "G_context")
BASE_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "FS0": GROUP_ORDER[:1],
    "FS1": GROUP_ORDER[:2],
    "FS2": GROUP_ORDER[:3],
    "FS3": GROUP_ORDER[:4],
    "FS4": GROUP_ORDER[:5],
}
DEPLOY_SUFFIX = "_deploy"
# Every base set has a deploy variant: the same groups plus facility_id (spec §9.2, §13.4).
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    **BASE_FEATURE_SETS,
    **{f"{name}{DEPLOY_SUFFIX}": groups for name, groups in BASE_FEATURE_SETS.items()},
}
DEPLOY_SETS = frozenset(name for name in FEATURE_SETS if name.endswith(DEPLOY_SUFFIX))


def is_deploy_set(name: str) -> bool:
    """True for a deploy variant (``FS<k>_deploy``: the base set plus ``facility_id``)."""
    return name in DEPLOY_SETS


def deploy_variant(name: str) -> str:
    """The deploy variant of base feature set ``name`` (``FS2`` -> ``FS2_deploy``)."""
    if name not in BASE_FEATURE_SETS:
        raise ValueError(
            f"{name!r} is not a base feature set; expected one of {list(BASE_FEATURE_SETS)}"
        )
    return f"{name}{DEPLOY_SUFFIX}"


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

    ``meta`` holds the population's canonical columns plus ``robson_group_no_onset`` (splits,
    subgroups, row ids; never fed to a model), ``x`` the model input columns (included
    features plus ``facility_id``), ``y`` the outcome, ``kinds`` each ``x`` column's type
    (categorical or numeric) and ``groups`` each column's feature group.
    ``exclusion_table`` holds, per category excluded from (or kept but counted in) the
    population, the rows and CS rows removed, the rows kept, and the CS total of ``P_audit``
    (raw counts: suppress before any export; spec §4.4 reporting obligation).
    ``legacy_onset`` marks the onset-coded sensitivity features (spec v1.3).
    """

    population: str
    meta: pd.DataFrame
    x: pd.DataFrame
    y: npt.NDArray[np.int64]
    exclusions: list[ExclusionLog]
    kinds: dict[str, str]
    groups: dict[str, str]
    exclusion_table: pd.DataFrame
    legacy_onset: bool = False


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


def _no_onset_label(result: ClassificationResult) -> str:
    if result.status == "resolved" and result.group is not None:
        return str(result.group)
    if result.status == "partial":
        return NO_ONSET_MERGES.get(frozenset(result.candidates), NO_ONSET_UNRESOLVED)
    return NO_ONSET_UNRESOLVED  # conflict


def robson_group_no_onset(frame: pd.DataFrame, rule_set: RuleSet | None = None) -> pd.Series:
    """The Robson group computed without onset (spec v1.3 item 3), as a string category.

    Each row is classified by the engine with ``onset_of_labour`` forced to missing (the GA
    band is used when the exact GA is missing, as in :func:`inputs_from_record`). Resolved
    rows keep their group (only groups 5-10 resolve without onset); a partial row whose
    candidates are exactly {1, 2} is ``"1_2"`` and exactly {3, 4} is ``"3_4"``; every other
    partial or conflict row is ``"partial"``. Never missing.
    """
    rule_set = rule_set if rule_set is not None else load_rule_set()
    columns = [c for c in (*INPUT_FIELDS, *GA_BAND_FIELDS) if c in frame.columns]
    records = cast("list[dict[str, Any]]", frame[columns].to_dict("records"))
    labels = []
    for record in records:
        record["onset_of_labour"] = None
        labels.append(_no_onset_label(classify(inputs_from_record(record), rule_set)))
    return pd.Series(labels, index=frame.index, dtype=object, name=ROBSON_NO_ONSET)


def _category_row(
    name: str, mask: pd.Series, cs: pd.Series, *, kept: bool = False
) -> dict[str, Any]:
    n, n_cs = int(mask.sum()), int(cs[mask].sum())
    return {
        "category": name,
        "n_excluded": 0 if kept else n,
        "n_cs_excluded": 0 if kept else n_cs,
        "n_kept": n if kept else 0,
        "n_cs_audit": int(cs.sum()),
    }


def _population(
    frame: pd.DataFrame, population: str
) -> tuple[pd.DataFrame, list[ExclusionLog], pd.DataFrame]:
    if population not in BASE_POPULATIONS:
        raise ValueError(
            f"unsupported population {population!r}; expected one of {BASE_POPULATIONS} "
            f"({P_PRED_COMPLETE} comes from complete_cases)"
        )
    audit, audit_log = audit_population(frame)
    cs = audit["cs"].astype(int)
    if population == P_PRED:
        pred, logs = prediction_population(audit)
        logs = [audit_log, *logs[1:]]
        kept_untyped = untyped_cs_mask(audit) & audit.index.isin(pred.index)
        rows = [
            _category_row("planned CS (elective type)", planned_cs_mask(audit), cs),
            _category_row("planned-CS onset, vaginal birth", planned_onset_vaginal_mask(audit), cs),
            _category_row("CS with no recorded type (kept)", kept_untyped, cs, kept=True),
        ]
    else:
        pred, onset_logs = onset_coded_population(audit)
        logs = [audit_log, *onset_logs]
        onset, cs_type = audit["onset_of_labour"], audit["prelabour_cs_type"]
        prelabour = onset == "prelabour_cs"
        rows = [
            _category_row("onset prelabour_cs planned", prelabour & (cs_type == "planned"), cs),
            _category_row("onset prelabour_cs emergency", prelabour & (cs_type == "emergency"), cs),
            _category_row("onset prelabour_cs type unknown", prelabour & cs_type.isna(), cs),
            _category_row("onset missing", onset.isna(), cs),
        ]
    return pred.reset_index(drop=True), logs, pd.DataFrame(rows)


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
    population: str = P_PRED,
    legacy_onset: bool | None = None,
    rule_set: RuleSet | None = None,
) -> ModelData:
    """Assemble the model input frame for ``population`` (spec §4.4, §9, v1.3).

    Inputs: the canonical frame with the Robson engine's columns (``canonical_robson``);
    the registry; ``raw_features`` from :func:`build_raw_features`, aligned with
    ``canonical`` by row position (it must hold every included raw feature). Derived
    features: ``bmi`` only when height and weight are both present;
    ``robson_group_no_onset`` (:func:`robson_group_no_onset`; also added to ``meta`` for
    subgroup reporting); ``is_missing_<field>`` as 0/1.

    ``legacy_onset`` (default: true exactly for ``P_pred_onset_coded``) swaps
    ``robson_group_no_onset`` for the v1.2 ``onset_of_labour`` and ``robson_group``
    (category, missing when partial or conflict). It is refused for any other population,
    and a registry that includes either legacy column is refused outright.
    """
    if len(raw_features) != len(canonical):
        raise ValueError("raw_features must align with canonical by row position")
    if legacy_onset is None:
        legacy_onset = population == P_PRED_ONSET_CODED
    if legacy_onset and population != P_PRED_ONSET_CODED:
        raise ValueError(f"legacy_onset is only allowed for population {P_PRED_ONSET_CODED!r}")
    included = registry.included()
    contaminated = sorted({e.name for e in included} & set(LEGACY_ONSET_COLUMNS))
    if contaminated:
        raise ValueError(
            f"{contaminated} must not be include features: onset is outcome-contaminated "
            "(spec v1.3); the onset-coded sensitivity analysis adds them via legacy_onset"
        )
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
    meta[ROBSON_NO_ONSET] = robson_group_no_onset(meta, rule_set)

    x: dict[str, pd.Series] = {}
    kinds: dict[str, str] = {}
    groups: dict[str, str] = {}
    for group in GROUP_ORDER:
        for entry in (e for e in included if e.group == group):
            if legacy_onset and entry.name == ROBSON_NO_ONSET:
                columns = [
                    (name, _categorical(meta[name]), "categorical") for name in LEGACY_ONSET_COLUMNS
                ]
            else:
                columns = _columns_for(entry, meta)
            for column, series, kind in columns:
                x[column], kinds[column], groups[column] = series, kind, group
    x[FACILITY] = _categorical(meta[FACILITY])
    kinds[FACILITY], groups[FACILITY] = "categorical", "G_context"
    y = np.asarray(meta["cs"].astype(int).to_numpy(), dtype=np.int64)
    return ModelData(
        population,
        meta,
        pd.DataFrame(x),
        y,
        exclusions,
        kinds,
        groups,
        exclusion_table,
        legacy_onset,
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
    if name == ROBSON_NO_ONSET:
        return _categorical(meta[ROBSON_NO_ONSET])
    if name.startswith(MISSING_PREFIX):
        field = name.removeprefix(MISSING_PREFIX)
        if field not in meta.columns:
            raise ValueError(f"{name}: source field {field!r} is not in the frame")
        return meta[field].isna().astype(np.float64)
    raise ValueError(f"no derivation defined for derived feature {name!r}")


def complete_cases(data: ModelData, feature_set: str) -> ModelData:
    """``data`` (P_pred) restricted to the rows with every ``feature_set`` input recorded.

    The complete-case analysis: instead of imputing, drop each admission that has any
    missing input in the feature set (``facility_id`` excepted: it is never missing and never
    a feature under S1). The dropped rows and their CS count are appended to
    ``exclusion_table``. The result's population is ``P_pred_complete``.
    """
    if data.population != P_PRED:
        raise ValueError(f"complete cases are taken from {P_PRED}, not {data.population!r}")
    fs = feature_spec(data, feature_set)
    columns = [c for c in fs.columns if c != FACILITY]
    keep = data.x[columns].notna().all(axis=1).to_numpy()
    dropped = {
        "category": f"any missing {feature_set} input (complete-case analysis)",
        "n_excluded": int((~keep).sum()),
        "n_cs_excluded": int(data.y[~keep].sum()),
        "n_kept": 0,
        "n_cs_audit": int(data.exclusion_table["n_cs_audit"].iloc[0])
        if len(data.exclusion_table)
        else int(data.y.sum()),
    }
    return ModelData(
        P_PRED_COMPLETE,
        data.meta.loc[keep].reset_index(drop=True),
        data.x.loc[keep].reset_index(drop=True),
        data.y[keep],
        data.exclusions,
        data.kinds,
        data.groups,
        pd.concat([data.exclusion_table, pd.DataFrame([dropped])], ignore_index=True),
        data.legacy_onset,
    )


def feature_spec(data: ModelData, name: str) -> FeatureSpec:
    """The FeatureSpec for feature set ``name`` over ``data`` (spec §9.2).

    FS0-FS4 never contain ``facility_id`` (spec §4.5); each ``FS<k>_deploy`` is ``FS<k>``
    plus it. In v1.2 G_context holds no feature other than facility, so FS4 equals FS3.
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
