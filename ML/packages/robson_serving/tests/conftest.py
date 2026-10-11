"""Synthetic fixtures for the robson_serving suite.

Nothing here comes from real records and nothing imports robson_ml: a small sklearn pipeline is
fitted on synthetic canonical admissions turned into model columns by robson_serving.features,
then wrapped in robson_serving.calibration.CalibratedModel like the deployment model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from robson_engine import load_rule_set
from robson_serving.artefact import write_artefact
from robson_serving.calibration import CalibratedModel, CalibrationChoice, PlattCalibrator
from robson_serving.contract import write_golden
from robson_serving.features import CATEGORICAL, NUMERIC, build_frame, required_inputs
from robson_serving.loader import load
from robson_serving.monitoring import reference_distributions
from robson_serving.schema import (
    CATEGORY,
    INTEGER,
    NUMBER,
    FeatureColumn,
    FeatureSchema,
    InputField,
    categorical_levels,
    column_kinds,
)

FACILITIES = ("FAC_A", "FAC_B", "FAC_C")
EDUCATION = ("none", "primary", "secondary")
GA_BANDS = (
    (20.0, 33.857142857142854),
    (35.0, 37.857142857142854),
    (38.0, 40.857142857142854),
    (40.142857142857146, 45.0),
)
KINDS: dict[str, str] = {
    "gestational_age_weeks": NUMERIC,
    "ga_band_lower": NUMERIC,
    "ga_band_upper": NUMERIC,
    "parity": NUMERIC,
    "previous_cs_count": NUMERIC,
    "plurality": NUMERIC,
    "fetal_presentation": CATEGORICAL,
    "robson_group_no_onset": CATEGORICAL,
    "maternal_age": NUMERIC,
    "bmi": NUMERIC,
    "education_level": CATEGORICAL,
    "is_missing_gestational_age_weeks": NUMERIC,
    "facility_id": CATEGORICAL,
}
GA_RANGE = (20.0, 45.0)
INPUT_SPECS: dict[str, tuple[str, tuple[float | None, float | None] | None]] = {
    "gestational_age_weeks": (NUMBER, GA_RANGE),
    "ga_band_lower": (NUMBER, GA_RANGE),
    "ga_band_upper": (NUMBER, GA_RANGE),
    "parity": (INTEGER, (0.0, None)),
    "previous_cs_count": (INTEGER, (0.0, None)),
    "plurality": (INTEGER, (1.0, None)),
    "fetal_presentation": (CATEGORY, None),
    "maternal_age": (NUMBER, (12.0, 55.0)),
    "height_cm": (NUMBER, (120.0, 200.0)),
    "weight_kg": (NUMBER, (30.0, 200.0)),
    "education_level": (CATEGORY, None),
    "facility_id": (CATEGORY, None),
}


def make_records(n: int, seed: int) -> pd.DataFrame:
    """Synthetic canonical admissions holding every input the synthetic model reads."""
    rng = np.random.default_rng(seed)
    parity = rng.integers(0, 5, n).astype(float)
    band = rng.integers(0, len(GA_BANDS), n)
    has_band = rng.random(n) < 0.8
    ga = np.round(rng.normal(38.5, 2.0, n).clip(24.0, 44.0), 1)
    presentation = np.where(rng.random(n) < 0.06, "non_cephalic", "cephalic")
    education = pd.Series(rng.choice(EDUCATION, n), dtype=object)
    return pd.DataFrame(
        {
            "gestational_age_weeks": np.where(rng.random(n) < 0.3, np.nan, ga),
            "ga_band_lower": np.where(has_band, [GA_BANDS[b][0] for b in band], np.nan),
            "ga_band_upper": np.where(has_band, [GA_BANDS[b][1] for b in band], np.nan),
            "parity": parity,
            "previous_cs_count": np.where(parity == 0, 0.0, rng.integers(0, 3, n).astype(float)),
            "plurality": np.where(rng.random(n) < 0.03, 2.0, 1.0),
            "fetal_presentation": pd.Series(presentation, dtype=object),
            "maternal_age": np.round(rng.uniform(16, 45, n)),
            "height_cm": np.where(
                rng.random(n) < 0.2, np.nan, np.round(rng.normal(160, 7, n).clip(125, 195), 1)
            ),
            "weight_kg": np.where(
                rng.random(n) < 0.2, np.nan, np.round(rng.normal(68, 10, n).clip(35, 150), 1)
            ),
            "education_level": education.mask(rng.random(n) < 0.2),
            "facility_id": pd.Series(rng.choice(FACILITIES, n), dtype=object),
        }
    )


def fit_model(records: pd.DataFrame, seed: int = 0) -> tuple[CalibratedModel, pd.DataFrame]:
    """A Platt-calibrated logistic pipeline on the synthetic model frame, and that frame."""
    x = build_frame(records, KINDS.items(), load_rule_set())
    rng = np.random.default_rng(seed)
    risk = (
        -1.2
        + 1.0 * x["previous_cs_count"].fillna(0).to_numpy()
        + 0.6 * (x["facility_id"] == "FAC_C").to_numpy()
        + 0.8 * (x["robson_group_no_onset"] == "partial").to_numpy()
    )
    y = (rng.random(len(x)) < 1 / (1 + np.exp(-risk))).astype(int)
    numeric = [name for name, kind in KINDS.items() if kind == NUMERIC]
    categorical = [name for name, kind in KINDS.items() if kind == CATEGORICAL]
    numeric_steps = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    prep = ColumnTransformer(
        [("num", numeric_steps, numeric), ("cat", categorical_steps, categorical)],
        remainder="drop",
    ).set_output(transform="pandas")
    pipe = Pipeline([("prep", prep), ("clf", LogisticRegression(C=1.0, max_iter=2000))])
    pipe.fit(x, y)
    raw = pipe.predict_proba(x)[:, 1]
    cv_brier = {"none": 0.2, "platt": 0.19, "isotonic": 0.21}
    choice = CalibrationChoice("platt", cv_brier, PlattCalibrator().fit(raw, y))
    return CalibratedModel(pipe, choice), x


def make_schema(model: CalibratedModel) -> FeatureSchema:
    """The synthetic model's feature schema (as robson-ml package builds it for real models)."""
    kinds = column_kinds(model)
    levels = categorical_levels(model)
    columns = tuple(FeatureColumn(name, kinds[name], levels.get(name)) for name in KINDS)
    inputs = tuple(
        InputField(
            name,
            INPUT_SPECS[name][0],
            INPUT_SPECS[name][1],
            levels.get(name) if INPUT_SPECS[name][0] == CATEGORY else None,
        )
        for name in required_inputs(KINDS)
    )
    return FeatureSchema(
        columns, inputs, coarsen_presentation=True, unseen_allowed=("facility_id",)
    )


def manifest_fields(schema: FeatureSchema, label: str = "synthetic-v0") -> dict[str, Any]:
    """The caller-supplied manifest fields for a synthetic artefact."""
    return {
        "version_label": label,
        "algorithm": "logreg_l2",
        "feature_list": list(schema.names),
        "hyperparameters": {"C": 1.0},
        "calibration_method": "platt",
        "evaluation_metrics": {"S1": None, "S2": None, "S3": None, "S4": None},
        "training_window_start": "2023-11",
        "training_window_end": "2024-03",
        "data_hash": "0" * 64,
        "features_yaml_hash": "0" * 64,
        "selection_rule_commit": "synthetic",
        "rule_set_version": load_rule_set().version_label,
        "git_commit": "synthetic",
        "seed": 0,
        "population": "P_pred",
        "feature_set": "synthetic",
        "beats_robson_baseline": None,
        "runtime_libraries": ["scikit-learn"],
    }


@pytest.fixture(scope="session")
def kinds() -> dict[str, str]:
    return dict(KINDS)


@pytest.fixture(scope="session")
def records() -> pd.DataFrame:
    return make_records(600, seed=1)


@pytest.fixture(scope="session")
def fitted(records: pd.DataFrame) -> tuple[CalibratedModel, pd.DataFrame]:
    return fit_model(records)


@pytest.fixture(scope="session")
def schema(fitted: tuple[CalibratedModel, pd.DataFrame]) -> FeatureSchema:
    return make_schema(fitted[0])


@pytest.fixture
def fields(schema: FeatureSchema) -> dict[str, Any]:
    return manifest_fields(schema)


@pytest.fixture
def artefact_dir(
    tmp_path: Path,
    fitted: tuple[CalibratedModel, pd.DataFrame],
    schema: FeatureSchema,
    fields: dict[str, Any],
) -> Path:
    """A freshly written synthetic artefact without golden files (tests may alter it)."""
    model, x = fitted
    out = tmp_path / "synthetic-v0"
    write_artefact(out, model, schema, fields, reference_distributions(x, KINDS))
    return out


@pytest.fixture(scope="session")
def packaged_dir(
    tmp_path_factory: pytest.TempPathFactory,
    fitted: tuple[CalibratedModel, pd.DataFrame],
    schema: FeatureSchema,
) -> Path:
    """A complete synthetic packaged artefact (all six §17.1 files)."""
    model, x = fitted
    out = tmp_path_factory.mktemp("packaged") / "synthetic-v0"
    write_artefact(out, model, schema, manifest_fields(schema), reference_distributions(x, KINDS))
    write_golden(load(out), out)
    return out
