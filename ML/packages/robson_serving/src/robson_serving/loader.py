"""Loading a packaged readiness model and scoring canonical admissions (spec §17.2).

``scorer = load(path)`` validates the manifest (required fields, artefact hash, robson_serving
major version, runtime library major.minor), the rule set version and the feature schema, then
loads ``model.joblib``. ``scorer.build_row(admission)`` turns one canonical admission dict into
the exact feature vector that is scored (JSON-safe: floats, strings, None), and
``scorer.score(records)`` returns calibrated probabilities. Missing optional inputs are scored
as NaN (imputed inside the model pipeline, as in training).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from robson_engine import RuleSet, load_rule_set
from robson_serving.artefact import (
    MODEL_FILE,
    SCHEMA_FILE,
    IncompatibleArtefact,
    check_manifest,
    read_manifest,
)
from robson_serving.features import CATEGORICAL, ROBSON_NO_ONSET, build_frame, is_missing
from robson_serving.schema import UNSEEN_LEVEL, FeatureSchema, SchemaError, normalise_admission

FeatureRow = dict[str, float | str | None]


class Scorer:
    """A loaded artefact: model, feature schema, manifest and the Robson rule set."""

    def __init__(
        self, model: Any, schema: FeatureSchema, manifest: Mapping[str, Any], rule_set: RuleSet
    ) -> None:
        self.model = model
        self.schema = schema
        self.manifest = dict(manifest)
        self.rule_set = rule_set
        self._kinds = [(column.name, column.kind) for column in schema.columns]

    @property
    def version_label(self) -> str:
        """The artefact's version label."""
        return str(self.manifest["version_label"])

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Model columns in model order."""
        return self.schema.names

    def build_row(self, admission: Mapping[str, Any]) -> FeatureRow:
        """The feature vector scored for ``admission`` (raises :class:`SchemaError`)."""
        record = normalise_admission(admission, self.schema)
        try:
            frame = build_frame(pd.DataFrame([record], dtype=object), self._kinds, self.rule_set)
        except SchemaError:
            raise
        except ValueError:
            raise SchemaError(ROBSON_NO_ONSET, "the Robson inputs are not valid") from None
        row: FeatureRow = {}
        for column in self.schema.columns:
            value = frame[column.name].iloc[0]
            if column.kind == CATEGORICAL:
                label = None if is_missing(value) else str(value)
                unseen = label == UNSEEN_LEVEL and column.name in self.schema.unseen_allowed
                if (
                    label is not None
                    and column.levels is not None
                    and label not in column.levels
                    and not unseen
                ):
                    raise SchemaError(column.name, "derived value is not a level the model knows")
                row[column.name] = label
            else:
                row[column.name] = None if is_missing(value) else float(value)
        return row

    def model_frame(self, rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
        """The model input frame for feature rows (None -> NaN; training dtypes)."""
        data: dict[str, pd.Series] = {}
        for column in self.schema.columns:
            values = [np.nan if is_missing(r[column.name]) else r[column.name] for r in rows]
            dtype = object if column.kind == CATEGORICAL else np.float64
            data[column.name] = pd.Series(values, dtype=dtype)
        return pd.DataFrame(data, columns=list(self.schema.names))

    def predict(self, frame: pd.DataFrame) -> list[float]:
        """Calibrated probabilities of cesarean for a model input frame."""
        if frame.empty:
            return []
        probabilities = np.asarray(self.model.predict_proba(frame), dtype=np.float64)[:, 1]
        return [float(p) for p in probabilities]

    def score(self, records: Sequence[Mapping[str, Any]] | pd.DataFrame) -> list[float]:
        """Probabilities for canonical admissions (a list of dicts or a DataFrame)."""
        items: list[Mapping[str, Any]] = (
            [{str(k): v for k, v in r.items()} for r in records.to_dict("records")]
            if isinstance(records, pd.DataFrame)
            else list(records)
        )
        rows = [self.build_row(item) for item in items]
        return self.predict(self.model_frame(rows))


def load(path: str | Path, *, rule_set: RuleSet | None = None) -> Scorer:
    """Load a packaged artefact directory; raises :class:`IncompatibleArtefact` if unusable."""
    directory = Path(path)
    manifest = read_manifest(directory)
    check_manifest(manifest, directory)
    rules = rule_set if rule_set is not None else load_rule_set()
    if rules.version_label != manifest["rule_set_version"]:
        raise IncompatibleArtefact(
            f"rule set {rules.version_label!r} differs from the artefact's "
            f"{manifest['rule_set_version']!r}"
        )
    schema_path = directory / SCHEMA_FILE
    if not schema_path.is_file():
        raise IncompatibleArtefact(f"{SCHEMA_FILE} is missing")
    try:
        schema = FeatureSchema.from_dict(json.loads(schema_path.read_text(encoding="utf-8")))
    except (ValueError, KeyError, TypeError) as exc:
        raise IncompatibleArtefact(f"{SCHEMA_FILE} is invalid") from exc
    if list(schema.names) != list(manifest["feature_list"]):
        raise IncompatibleArtefact(f"{SCHEMA_FILE} and the manifest list different features")
    try:
        model = joblib.load(directory / MODEL_FILE)
    except Exception as exc:  # any unpickling failure means this runtime cannot serve it
        raise IncompatibleArtefact(
            f"{MODEL_FILE} could not be loaded ({type(exc).__name__})"
        ) from exc
    if not hasattr(model, "predict_proba"):
        raise IncompatibleArtefact(f"{MODEL_FILE} does not hold a probabilistic classifier")
    return Scorer(model, schema, manifest, rules)
