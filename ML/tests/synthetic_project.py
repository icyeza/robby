"""A complete synthetic project directory, laid out like the real one.

Used by the pipeline notebook in synthetic mode and by its tests. Nothing here is derived
from real records: the raw export is built from :func:`tests.synthetic.make_admissions`
(canonical columns, mapped one-to-one) plus :func:`tests.synthetic.make_raw_sheet` (the
raw columns of the feature registry). The directory holds ``configs/`` (project, mapping,
feature registry and selection rule copies) and ``data/raw/raw.xlsx``; every later file
(``data/interim``, ``data/processed``, ``reports/``, ``mlruns/``) is written by the pipeline.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from robson_ml.features import load_feature_registry
from robson_ml.schema import CANONICAL_DTYPES
from tests.synthetic import make_admissions, make_raw_sheet

SHEET = "main"
RAW_FILE = "data/raw/raw.xlsx"
KIND_BY_DTYPE = {
    "Int64": "integer",
    "float64": "float",
    "datetime64[ns]": "datetime",
    "object": "text",
}


def synthetic_mapping() -> dict[str, object]:
    """A mapping file that maps every canonical field from the raw column of the same name.

    The synthetic ``mother_key`` column stands in for the raw patient identifier and is
    hashed (``hash_key``) exactly as the real one is; ``admission_id`` is a row key.
    """
    fields: dict[str, dict[str, object]] = {
        "admission_id": {"kind": "row_key", "status": "confirmed"}
    }
    for name, dtype in CANONICAL_DTYPES.items():
        if name == "admission_id":
            continue
        kind = "gestational_age" if name == "gestational_age_weeks" else KIND_BY_DTYPE[dtype]
        if name == "mother_key":
            kind = "hash_key"
        spec: dict[str, object] = {"raw": name, "kind": kind, "status": "confirmed"}
        if kind == "gestational_age":
            spec["format"] = "decimal_weeks"
        if name == "delivery_date":
            spec["date_only"] = True
        fields[name] = spec
    return {"source": "synthetic", "sheet": SHEET, "fields": fields}


def build_synthetic_project(root: Path, repo: Path, n: int = 1500, seed: int = 11) -> Path:
    """Write a synthetic project under ``root`` and return ``root``.

    Inputs: ``repo`` is the repository root, whose ``configs/features_v1.yaml``,
    ``configs/selection_rule.yaml`` and ``configs/analysis.yaml`` are copied (so the synthetic
    run uses the approved registry, the pre-registered rule and the analysis settings); ``n``
    synthetic admissions from ``seed``. The raw workbook holds the canonical columns (except
    ``admission_id``) followed by the raw registry columns, row-aligned.
    """
    configs = root / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    for name in ("features_v1.yaml", "selection_rule.yaml", "analysis.yaml"):
        shutil.copy(repo / "configs" / name, configs / name)
    (configs / "mapping_ur_cmhs.yaml").write_text(
        yaml.safe_dump(synthetic_mapping(), sort_keys=False), encoding="utf-8"
    )
    (configs / "project.yaml").write_text(
        yaml.safe_dump(
            {
                "raw_path": RAW_FILE,
                "mapping_path": "configs/mapping_ur_cmhs.yaml",
                "answers_path": "configs/open_questions_answers.yaml",
                "interim_dir": "data/interim",
                "processed_dir": "data/processed",
                "reports_dir": "reports",
                "features_path": "configs/features_v1.yaml",
                "seed": seed,
            }
        ),
        encoding="utf-8",
    )
    registry = load_feature_registry(configs / "features_v1.yaml")
    canonical = make_admissions(n, seed=seed).drop(columns=["admission_id"])
    extra = make_raw_sheet(registry, n, seed=seed)
    clash = sorted(set(canonical.columns) & set(extra.columns))
    if clash:
        raise ValueError(f"synthetic raw columns clash: {clash}")
    raw = canonical.join(extra)
    (root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    raw.to_excel(root / RAW_FILE, index=False, sheet_name=SHEET)
    return root
