"""Packaging a deployment artefact for serving: ``robson-ml package``.

Reads ``artefacts/<version_label>/`` (``model.joblib`` and ``fit_summary.json`` from
:func:`robson_ml.deploy.fit_deployment`) and writes the packaged directory through
:mod:`robson_serving`: the model re-dumped so its pickle references robson_serving and sklearn
only (the classes resolve to robson_serving through the re-export in robson_ml.calibration),
``feature_schema.json`` (from the fitted pipeline and the feature registry),
``manifest.json`` (provenance, evaluation and serving fields), ``reference_distributions.json``
(bucket proportions of the training population, no bucket under 5 rows), 200 synthetic golden
admissions and their probabilities. It also checks, on the training population, that the
serving path rebuilds the training feature frame and probabilities exactly
(:func:`parity_report`: counts and a maximum difference only, never a row).
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd

from robson_engine import COARSE_PRESENTATIONS, RuleSet, load_rule_set
from robson_ml.deploy import MODEL_FILE, SUMMARY_FILE
from robson_ml.feature_sets import FACILITY, ModelData
from robson_ml.features import FeatureRegistry
from robson_ml.privacy import fmt_count
from robson_ml.schema import (
    CANONICAL_DTYPES,
    GA_RANGE,
    HEIGHT_CM_RANGE,
    MATERNAL_AGE_RANGE,
    WEIGHT_KG_RANGE,
)
from robson_serving.artefact import write_artefact
from robson_serving.calibration import CalibratedModel
from robson_serving.contract import GOLDEN_N, GOLDEN_SEED, GoldenCheck, write_golden
from robson_serving.features import CATEGORICAL, required_inputs
from robson_serving.loader import Scorer, load
from robson_serving.monitoring import reference_distributions
from robson_serving.schema import (
    CATEGORY,
    INTEGER,
    NON_CEPHALIC,
    NUMBER,
    PRESENTATION,
    FeatureColumn,
    FeatureSchema,
    InputField,
    SchemaError,
    categorical_levels,
    column_kinds,
)

Range = tuple[float | None, float | None]
# Allowed ranges of canonical inputs (the same bounds the canonical schema checks).
CANONICAL_RANGES: dict[str, Range] = {
    "gestational_age_weeks": GA_RANGE,
    "ga_band_lower": GA_RANGE,
    "ga_band_upper": GA_RANGE,
    "maternal_age": MATERNAL_AGE_RANGE,
    "height_cm": HEIGHT_CM_RANGE,
    "weight_kg": WEIGHT_KG_RANGE,
    "parity": (0.0, None),
    "previous_cs_count": (0.0, None),
    "plurality": (1.0, None),
    "anc_contacts": (0.0, None),
}
DTYPE_INPUT_KINDS = {"Int64": INTEGER, "float64": NUMBER, "object": CATEGORY}
RAW_INPUT_KINDS = {
    "integer": INTEGER,
    "integer_sum": INTEGER,
    "float": NUMBER,
    "gestational_age": NUMBER,
    "text": CATEGORY,
    "category": CATEGORY,
}
EXTRA_RUNTIME_LIBRARIES: dict[str, tuple[str, ...]] = {"xgboost": ("xgboost",)}
SPLITS = ("S1", "S2", "S3", "S4")
SUMMARY_METRICS = (
    "mean_auc",
    "min_auc",
    "max_auc",
    "pooled_auc",
    "mean_calibration_slope",
    "mean_calibration_in_the_large",
    "pooled_brier",
    "mean_log_loss",
)
RUN_COLUMNS = frozenset(
    {"run_id", "split", "model", "feature_set", "missing_strategy", "population", "data_hash"}
)
BASELINE_MODEL = "B1"
PARITY_TOLERANCE = 1e-12


def input_field(
    name: str, registry: FeatureRegistry, levels: Mapping[str, tuple[str, ...]]
) -> InputField:
    """The schema entry of canonical input ``name`` (canonical field or mapped raw feature)."""
    valid: Range | None
    if name in CANONICAL_DTYPES:
        kind = DTYPE_INPUT_KINDS[CANONICAL_DTYPES[name]]
        valid = CANONICAL_RANGES.get(name)
    else:
        entry = next((e for e in registry.entries if e.name == name), None)
        if entry is None or entry.mapping is None:
            raise ValueError(f"{name}: not a canonical field or a mapped registry feature")
        kind = RAW_INPUT_KINDS[entry.mapping.kind]
        valid = entry.mapping.valid_range
    if kind == CATEGORY:
        return InputField(name, kind, None, levels.get(name))
    return InputField(name, kind, valid, None)


def build_schema(
    model: CalibratedModel, features: Sequence[str], registry: FeatureRegistry
) -> FeatureSchema:
    """The feature schema of a fitted deployment model.

    Column kinds and categorical levels come from the fitted ``prep`` step; input kinds and
    ranges from the canonical schema and the registry. ``fetal_presentation`` is coarsened at
    scoring time when the model only knows ``non_cephalic``; ``facility_id`` may be unseen.
    """
    kinds = column_kinds(model)
    absent = [f for f in features if f not in kinds]
    if absent:
        raise ValueError(f"the model's preprocessing does not read feature(s) {absent}")
    levels = categorical_levels(model)
    columns = tuple(
        FeatureColumn(f, kinds[f], levels.get(f) if kinds[f] == CATEGORICAL else None)
        for f in features
    )
    inputs = tuple(input_field(name, registry, levels) for name in required_inputs(features))
    presentation = set(levels.get(PRESENTATION, ()))
    coarsen = NON_CEPHALIC in presentation and not (
        presentation & COARSE_PRESENTATIONS[NON_CEPHALIC]
    )
    unseen = (FACILITY,) if FACILITY in features else ()
    return FeatureSchema(columns, inputs, coarsen, unseen)


def _metric_summary(row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {"run_id": str(row["run_id"]), "feature_set": str(row["feature_set"])}
    for key in SUMMARY_METRICS:
        if key in row.index and pd.notna(row[key]):
            out[key] = float(row[key])
    return out


def evaluation_summaries(
    runs: pd.DataFrame, summary: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any] | None], bool | None]:
    """S1-S4 metric summaries of the deployed configuration, and the Robson baseline check.

    Runs are those of :func:`robson_ml.select.load_runs` with the same model, missing
    strategy, population and data hash; per split the deployed feature set is preferred,
    then its base set; a split without a run is None. ``beats_robson_baseline`` compares the
    S1 mean AUC with B1's (None when either run is missing).
    """
    empty: dict[str, dict[str, Any] | None] = dict.fromkeys(SPLITS)
    if runs.empty or not set(runs.columns) >= RUN_COLUMNS:
        return empty, None
    config = summary["config"]
    same_data = (runs["population"] == config["population"]) & (
        runs["data_hash"] == summary["data_hash"]
    )
    scoped = runs[
        same_data
        & (runs["model"] == config["model"])
        & (runs["missing_strategy"] == config["missing_strategy"])
    ]
    metrics = dict(empty)
    for split in SPLITS:
        for name in (str(summary["feature_set"]), str(config["feature_set"])):
            rows = scoped[(scoped["split"] == split) & (scoped["feature_set"] == name)]
            if not rows.empty:
                metrics[split] = _metric_summary(rows.iloc[0])
                break
    baseline = runs[same_data & (runs["split"] == "S1") & (runs["model"] == BASELINE_MODEL)]
    s1 = metrics["S1"]
    beats = None
    if (
        s1 is not None
        and "mean_auc" in s1
        and not baseline.empty
        and "mean_auc" in baseline.columns
        and pd.notna(baseline.iloc[0]["mean_auc"])
    ):
        beats = bool(s1["mean_auc"] > float(baseline.iloc[0]["mean_auc"]))
    return metrics, beats


def manifest_fields(
    summary: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    beats_baseline: bool | None,
    rule_set_version: str,
) -> dict[str, Any]:
    """The manifest fields taken from the fit summary (write_artefact adds hash and versions)."""
    algorithm = str(summary["algorithm"])
    return {
        "version_label": summary["version_label"],
        "algorithm": algorithm,
        "feature_list": list(summary["features"]),
        "hyperparameters": summary["hyperparameters"],
        "calibration_method": summary["calibration"]["method"],
        "calibration_cv_brier": summary["calibration"]["cv_brier"],
        "evaluation_metrics": dict(evaluation),
        "training_window_start": summary["training_window_start"],
        "training_window_end": summary["training_window_end"],
        "data_hash": summary["data_hash"],
        "features_yaml_hash": summary["features_yaml_hash"],
        "selection_rule_commit": summary["selection_rule_commit"],
        "rule_set_version": rule_set_version,
        "git_commit": summary["git_commit"],
        "seed": summary["seed"],
        "population": summary["population"],
        "population_version": summary.get("population_version"),
        "feature_set": summary["feature_set"],
        "use_facility": summary.get("use_facility"),
        "facility_decision": summary.get("facility_decision"),
        "beats_robson_baseline": beats_baseline,
        "runtime_libraries": ["scikit-learn", *EXTRA_RUNTIME_LIBRARIES.get(algorithm, ())],
        "training_library_versions": summary.get("library_versions", {}),
        "split_scheme": summary.get("split_scheme"),
        "mlflow_run_id": summary.get("mlflow_run_id"),
    }


def frame_mismatches(expected: pd.DataFrame, actual: pd.DataFrame) -> dict[str, int]:
    """Per column of ``expected``, how many rows differ in ``actual`` (counts only).

    Both missing counts as equal. A column that is absent, of another dtype, or of another
    length counts every row.
    """
    out: dict[str, int] = {}
    for column in expected.columns:
        if (
            column not in actual.columns
            or expected[column].dtype != actual[column].dtype
            or len(expected) != len(actual)
        ):
            out[str(column)] = len(expected)
            continue
        left = expected[column].reset_index(drop=True)
        right = actual[column].reset_index(drop=True)
        same = (left.isna() & right.isna()) | (left == right)
        differing = int((~same).sum())
        if differing:
            out[str(column)] = differing
    return out


@dataclass(frozen=True)
class ParityReport:
    """Serving vs training on the training population (aggregates only)."""

    n_rows: int
    schema_errors: dict[str, int]
    column_mismatches: dict[str, int]
    max_probability_difference: float

    @property
    def ok(self) -> bool:
        """No schema error, no differing cell, probabilities equal within 1e-12."""
        return (
            not self.schema_errors
            and not self.column_mismatches
            and self.max_probability_difference <= PARITY_TOLERANCE
        )

    def summary(self) -> str:
        """One line with counts (small counts shown as <5) and the maximum difference."""
        errors = {k: fmt_count(v) for k, v in self.schema_errors.items()} or "none"
        cells = {k: fmt_count(v) for k, v in self.column_mismatches.items()} or "none"
        rows = fmt_count(self.n_rows)
        difference = f"{self.max_probability_difference:.1e}"
        return (
            f"rows {rows}; schema errors {errors}; column mismatches {cells}; "
            f"max |probability difference| {difference}"
        )


def parity_report(scorer: Scorer, data: ModelData) -> ParityReport:
    """Rebuild every training row through ``scorer.build_row`` and compare with ``data.x``."""
    features = list(scorer.feature_names)
    inputs = list(required_inputs(features))
    records = cast("list[dict[str, Any]]", data.meta[inputs].to_dict("records"))
    rows, kept, errors = [], [], Counter[str]()
    for position, record in enumerate(records):
        try:
            rows.append(scorer.build_row(record))
        except SchemaError as exc:
            errors[exc.field] += 1
            continue
        kept.append(position)
    expected = data.x[features].iloc[kept].reset_index(drop=True)
    actual = scorer.model_frame(rows)
    mismatches = frame_mismatches(expected, actual)
    difference = 0.0
    if rows:
        p_expected = np.asarray(scorer.predict(expected))
        p_actual = np.asarray(scorer.predict(actual))
        difference = float(np.max(np.abs(p_expected - p_actual)))
    return ParityReport(len(records), dict(errors), mismatches, difference)


@dataclass(frozen=True)
class PackageResult:
    """What ``package_artefact`` wrote and how its checks came out."""

    out_dir: Path
    manifest: dict[str, Any]
    golden: GoldenCheck
    parity: ParityReport

    @property
    def ok(self) -> bool:
        """Golden outputs reproduce and serving matches training."""
        return self.golden.ok and self.parity.ok


def read_fit_summary(artefact_dir: Path) -> dict[str, Any]:
    """``fit_summary.json`` of a fit-deploy artefact directory."""
    path = Path(artefact_dir) / SUMMARY_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{SUMMARY_FILE} not found in {artefact_dir}")
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


def package_artefact(
    artefact_dir: Path,
    out_dir: Path,
    data: ModelData,
    registry: FeatureRegistry,
    runs: pd.DataFrame,
    *,
    n_golden: int = GOLDEN_N,
    seed: int = GOLDEN_SEED,
    rule_set: RuleSet | None = None,
) -> PackageResult:
    """Write the serving directory for ``artefact_dir`` into ``out_dir`` (see module doc).

    ``data`` must be the population the artefact was fitted on; ``runs`` the harness runs
    (may be empty). On any error ``out_dir`` is removed. The caller decides what to do when
    the result is not ``ok`` (the CLI removes the directory).
    """
    summary = read_fit_summary(artefact_dir)
    if data.population != summary["population"]:
        raise ValueError("the data were built for a different population than the artefact")
    model = joblib.load(Path(artefact_dir) / MODEL_FILE)
    if not isinstance(model, CalibratedModel):
        raise TypeError(f"{MODEL_FILE} does not hold a CalibratedModel")
    features = [str(f) for f in summary["features"]]
    rules = rule_set if rule_set is not None else load_rule_set()
    schema = build_schema(model, features, registry)
    evaluation, beats = evaluation_summaries(runs, summary)
    fields = manifest_fields(summary, evaluation, beats, rules.version_label)
    reference = reference_distributions(
        data.x[features], {column.name: column.kind for column in schema.columns}
    )
    manifest = write_artefact(Path(out_dir), model, schema, fields, reference)
    try:
        scorer = load(out_dir, rule_set=rules)
        parity = parity_report(scorer, data)
        golden = write_golden(scorer, Path(out_dir), n_golden, seed)
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    return PackageResult(Path(out_dir), manifest, golden, parity)
