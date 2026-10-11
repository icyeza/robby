import json
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from robson_serving._version import __version__
from robson_serving.artefact import (
    MANIFEST_FILE,
    MANIFEST_FORMAT,
    MODEL_FILE,
    REFERENCE_FILE,
    SCHEMA_FILE,
    IncompatibleArtefact,
    check_manifest,
    major_minor,
    read_manifest,
    sha256_file,
    write_artefact,
)
from robson_serving.calibration import CalibratedModel
from robson_serving.monitoring import reference_distributions
from robson_serving.schema import FeatureSchema

Fitted = tuple[CalibratedModel, pd.DataFrame]


def _write(
    tmp_path: Path, fitted: Fitted, schema: FeatureSchema, fields: dict[str, Any], kinds: dict
) -> Path:
    model, x = fitted
    out = tmp_path / "artefact"
    write_artefact(out, model, schema, fields, reference_distributions(x, kinds))
    return out


def test_write_artefact_writes_the_files_and_manifest(
    tmp_path: Path, fitted: Fitted, schema: FeatureSchema, fields: dict[str, Any], kinds: dict
) -> None:
    out = _write(tmp_path, fitted, schema, fields, kinds)
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [MODEL_FILE, SCHEMA_FILE, MANIFEST_FILE, REFERENCE_FILE]
    )
    manifest = read_manifest(out)
    assert manifest["artefact_hash"] == sha256_file(out / MODEL_FILE)
    assert manifest["manifest_format"] == MANIFEST_FORMAT
    assert manifest["robson_serving_version"] == __version__
    assert {"numpy", "pandas", "scikit-learn", "joblib"} <= manifest["library_versions"].keys()
    assert b"robson_serving.calibration" in (out / MODEL_FILE).read_bytes()
    check_manifest(manifest, out)
    stored = json.loads((out / SCHEMA_FILE).read_text(encoding="utf-8"))
    assert FeatureSchema.from_dict(stored) == schema


def test_write_artefact_refuses_a_non_empty_directory(
    tmp_path: Path, fitted: Fitted, schema: FeatureSchema, fields: dict[str, Any], kinds: dict
) -> None:
    out = _write(tmp_path, fitted, schema, fields, kinds)
    with pytest.raises(FileExistsError):
        write_artefact(out, fitted[0], schema, fields, {})


def test_write_artefact_checks_the_fields(
    tmp_path: Path, fitted: Fitted, schema: FeatureSchema, fields: dict[str, Any]
) -> None:
    model = fitted[0]
    with pytest.raises(ValueError, match="set by write_artefact"):
        write_artefact(tmp_path / "a", model, schema, {**fields, "artefact_hash": "x"}, {})
    partial = {k: v for k, v in fields.items() if k != "data_hash"}
    with pytest.raises(ValueError, match="data_hash"):
        write_artefact(tmp_path / "b", model, schema, partial, {})
    with pytest.raises(ValueError, match="feature_list"):
        write_artefact(tmp_path / "c", model, schema, {**fields, "feature_list": ["x"]}, {})


def test_write_artefact_refuses_a_pickle_that_needs_robson_ml(
    tmp_path: Path, schema: FeatureSchema, fields: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy = type("Legacy", (), {"__module__": "robson_ml_stub"})
    module = types.ModuleType("robson_ml_stub")
    setattr(module, "Legacy", legacy)  # noqa: B010
    monkeypatch.setitem(sys.modules, "robson_ml_stub", module)
    out = tmp_path / "legacy"
    with pytest.raises(IncompatibleArtefact, match="robson_ml"):
        write_artefact(out, legacy(), schema, fields, {})
    assert not out.exists()


def test_check_manifest_refusals(
    tmp_path: Path, fitted: Fitted, schema: FeatureSchema, fields: dict[str, Any], kinds: dict
) -> None:
    out = _write(tmp_path, fitted, schema, fields, kinds)
    manifest = read_manifest(out)
    versions = manifest["library_versions"]
    cases = {
        "artefact_hash": {**manifest, "artefact_hash": "0" * 64},
        "scikit-learn": {**manifest, "library_versions": {**versions, "scikit-learn": "0.1.0"}},
        "not installed": {**manifest, "runtime_libraries": ["scikit-learn", "no-such-library"]},
        "robson_serving": {**manifest, "robson_serving_version": "99.0.0"},
        "format": {**manifest, "manifest_format": 2},
    }
    for message, broken in cases.items():
        with pytest.raises(IncompatibleArtefact, match=message):
            check_manifest(broken, out)


def test_read_manifest_names_what_is_missing(
    tmp_path: Path, fitted: Fitted, schema: FeatureSchema, fields: dict[str, Any], kinds: dict
) -> None:
    out = _write(tmp_path, fitted, schema, fields, kinds)
    manifest = read_manifest(out)
    del manifest["data_hash"]
    (out / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(IncompatibleArtefact, match="data_hash"):
        read_manifest(out)
    with pytest.raises(IncompatibleArtefact, match="missing"):
        read_manifest(tmp_path / "nowhere")


def test_major_minor() -> None:
    assert major_minor("1.9.1") == (1, 9)
    assert major_minor("2.4.6.post1") == (2, 4)
    with pytest.raises(ValueError):
        major_minor("abc")
