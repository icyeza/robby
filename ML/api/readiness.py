"""Loading the readiness model artefact and turning an admission into its feature row."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from api.schemas import AdmissionRequest
from robson_engine import COARSE_PRESENTATIONS, RuleSet
from robson_ml.feature_sets import robson_group_no_onset

ML_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = Path("artefacts") / "packaged" / "readiness-v1.0"
MODEL_DIR_ENV = "ROBSON_MODEL_DIR"

# Yes/no inputs sent as booleans; the model's level spelling ("Yes" or "yes") is looked up.
BOOLEAN_INPUTS = (
    "previous_stillbirth",
    "hiv_positive",
    "preeclampsia_recorded",
    "gdm_recorded",
    "ivf",
    "antenatal_admission",
    "malaria_in_pregnancy",
    "gu_infection",
)
TEXT_INPUTS = ("insurance_type", "education_level", "residency", "marital_status", "occupation")
NUMERIC_INPUTS = (
    "maternal_age",
    "height_cm",
    "weight_kg",
    "age_first_pregnancy",
    "gravidity",
    "abortions",
    "died_children",
    "anc_contacts",
    "previous_preterm",
)
# Optional details reported back when blank (the Robson fields are reported by the engine).
OPTIONAL_INPUTS = (
    "facility_id",
    *NUMERIC_INPUTS,
    "living_children",
    *BOOLEAN_INPUTS,
    *TEXT_INPUTS,
)


def model_dir() -> Path:
    """The model directory: ``$ROBSON_MODEL_DIR`` or the default, relative to the ML folder."""
    raw = os.environ.get(MODEL_DIR_ENV)
    path = Path(raw) if raw else DEFAULT_MODEL_DIR
    return path if path.is_absolute() else ML_DIR / path


@dataclass
class ReadinessModel:
    """A loaded calibrated model with its fit summary."""

    model: Any
    summary: dict[str, Any]
    levels: dict[str, list[str]] = field(default_factory=dict)

    @property
    def version(self) -> str:
        """The model's version label from its fit summary (``unknown`` when absent)."""
        return str(self.summary.get("version_label", "unknown"))

    @property
    def features(self) -> list[str]:
        """The feature columns the model was fitted on, in training order."""
        return list(self.summary["features"])


def _categorical_levels(model: Any) -> dict[str, list[str]]:
    """Training levels per categorical column, read from the fitted one-hot encoder."""
    try:
        prep = model.estimator.named_steps["prep"]
        for name, transformer, columns in prep.transformers_:
            if name == "cat":
                encoder = transformer.named_steps["encode"]
                return {
                    str(c): [str(v) for v in cats]
                    for c, cats in zip(columns, encoder.categories_, strict=True)
                }
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    return {}


def summary_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """The fit-summary fields the API reads, taken from a packaged artefact's ``manifest.json``."""
    summary = {
        key: manifest.get(key)
        for key in (
            "version_label",
            "algorithm",
            "feature_set",
            "use_facility",
            "hyperparameters",
            "population",
            "training_window_start",
            "training_window_end",
        )
    }
    summary["features"] = list(manifest["feature_list"])
    summary["calibration"] = {
        "method": manifest.get("calibration_method"),
        "cv_brier": manifest.get("calibration_cv_brier"),
    }
    return summary


def _read_summary(directory: Path, model_path: Path) -> dict[str, Any] | None:
    """``fit_summary.json`` (a fit-deploy output) or ``manifest.json`` (a packaged artefact).

    A packaged artefact's model file must match the hash in its manifest.
    """
    summary_path, manifest_path = directory / "fit_summary.json", directory / "manifest.json"
    if summary_path.is_file():
        return dict(json.loads(summary_path.read_text(encoding="utf-8")))
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if digest != manifest.get("artefact_hash"):
            raise ValueError("model.joblib does not match the manifest's artefact_hash")
        return summary_from_manifest(manifest)
    return None


def load_readiness_model(directory: Path) -> tuple[ReadinessModel | None, str | None]:
    """Load ``model.joblib`` with its fit summary or manifest; on failure return the reason."""
    model_path = directory / "model.joblib"
    if not model_path.is_file():
        return None, "No readiness model is active (model artefact not found)."
    try:
        summary = _read_summary(directory, model_path)
        if summary is None:
            return None, "No readiness model is active (model artefact not found)."
        model = joblib.load(model_path)
        if not hasattr(model, "predict_proba") or "features" not in summary:
            return None, "No readiness model is active (model artefact is not usable)."
    except Exception as exc:  # any unpickling or parsing failure degrades the same way
        return (
            None,
            f"No readiness model is active (model artefact failed to load: {type(exc).__name__}).",
        )
    return ReadinessModel(model, summary, _categorical_levels(model)), None


def _living_children_band(count: int | None) -> str | None:
    if count is None:
        return None
    if count == 0:
        return "0"
    return "Less than 4" if count < 4 else "4 or more"


def _yes_no(value: bool | None, levels: list[str] | None) -> str | None:
    if value is None:
        return None
    wanted = "yes" if value else "no"
    for level in levels or []:
        if level.lower() == wanted:
            return level
    return wanted.capitalize()


def _model_presentation(value: str | None, levels: list[str] | None) -> str | None:
    """Precise non-cephalic presentations become ``non_cephalic`` when the model was fit so."""
    if value is None or levels is None or value in levels:
        return value
    if value in COARSE_PRESENTATIONS["non_cephalic"] and "non_cephalic" in levels:
        return "non_cephalic"
    return value


def feature_row(
    admission: AdmissionRequest, loaded: ReadinessModel, rule_set: RuleSet
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """One-row model frame, plus the blank optional inputs and the unrecognised text inputs."""
    levels = loaded.levels
    values: dict[str, Any] = {
        "gestational_age_weeks": admission.gestational_age_weeks,
        "ga_band_lower": admission.ga_band_lower,
        "ga_band_upper": admission.ga_band_upper,
        "parity": admission.parity,
        "previous_cs_count": admission.previous_cs_count,
        "plurality": admission.plurality,
        "fetal_presentation": _model_presentation(
            admission.fetal_presentation, levels.get("fetal_presentation")
        ),
        "facility_id": admission.facility_id,
        "living_children_band": _living_children_band(admission.living_children),
    }
    for name in NUMERIC_INPUTS:
        values[name] = getattr(admission, name)
    for name in BOOLEAN_INPUTS:
        values[name] = _yes_no(getattr(admission, name), levels.get(name))
    for name in TEXT_INPUTS:
        values[name] = getattr(admission, name)
    height, weight = admission.height_cm, admission.weight_kg
    values["bmi"] = weight / (height / 100) ** 2 if height and weight else None

    robson = pd.DataFrame(
        [
            {
                "parity": admission.parity,
                "previous_cs_count": admission.previous_cs_count,
                "plurality": admission.plurality,
                "fetal_presentation": admission.fetal_presentation,
                "gestational_age_weeks": admission.gestational_age_weeks,
                "onset_of_labour": None,
                "ga_band_lower": admission.ga_band_lower,
                "ga_band_upper": admission.ga_band_upper,
            }
        ]
    )
    values["robson_group_no_onset"] = robson_group_no_onset(robson, rule_set).iloc[0]

    row: dict[str, Any] = {}
    unrecognised: list[str] = []
    text_columns: list[str] = []
    for name in loaded.features:
        value = values.get(name)
        if name in levels or isinstance(value, str):
            text_columns.append(name)
            row[name] = np.nan if value is None else str(value)
            if value is not None and name in levels and str(value) not in levels[name]:
                unrecognised.append(name)
        else:
            row[name] = np.nan if value is None else float(value)
    frame = pd.DataFrame([row], columns=loaded.features)
    for name in text_columns:
        frame[name] = frame[name].astype(object)
    not_given = [name for name in OPTIONAL_INPUTS if getattr(admission, name) is None]
    return frame, not_given, unrecognised
