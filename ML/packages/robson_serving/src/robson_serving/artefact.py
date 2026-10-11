"""The packaged artefact directory (spec §17.1): file names, manifest, integrity checks, writing.

A packaged artefact holds ``model.joblib`` (pickling ``robson_serving`` classes and sklearn
only), ``feature_schema.json``, ``manifest.json`` (spec §17.3 fields plus the serving ones),
``reference_distributions.json``, ``golden_inputs.parquet`` and ``golden_outputs.json``.
:func:`check_manifest` refuses an artefact whose model file does not match its hash, whose
runtime libraries differ from the installed ones in major.minor, or that was packaged for
another major version of robson_serving.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib

from robson_serving._version import __version__
from robson_serving.schema import FeatureSchema

MODEL_FILE = "model.joblib"
SCHEMA_FILE = "feature_schema.json"
MANIFEST_FILE = "manifest.json"
REFERENCE_FILE = "reference_distributions.json"
GOLDEN_INPUTS_FILE = "golden_inputs.parquet"
GOLDEN_OUTPUTS_FILE = "golden_outputs.json"
ARTEFACT_FILES = (
    MODEL_FILE,
    SCHEMA_FILE,
    MANIFEST_FILE,
    REFERENCE_FILE,
    GOLDEN_INPUTS_FILE,
    GOLDEN_OUTPUTS_FILE,
)
MANIFEST_FORMAT = 1
CORE_LIBRARIES = ("numpy", "pandas", "scikit-learn", "joblib")
FORBIDDEN_PICKLE_MODULE = b"robson_ml"
# Spec §17.3 fields plus the serving-side ones; every manifest must hold all of them.
MANIFEST_FIELDS = (
    "manifest_format",
    "version_label",
    "algorithm",
    "feature_list",
    "hyperparameters",
    "calibration_method",
    "evaluation_metrics",
    "artefact_hash",
    "training_window_start",
    "training_window_end",
    "data_hash",
    "features_yaml_hash",
    "selection_rule_commit",
    "rule_set_version",
    "git_commit",
    "seed",
    "library_versions",
    "population",
    "feature_set",
    "beats_robson_baseline",
    "runtime_libraries",
    "robson_serving_version",
)
# Filled in by write_artefact itself, never by the caller.
WRITER_FIELDS = frozenset(
    {"manifest_format", "artefact_hash", "library_versions", "robson_serving_version"}
)
_VERSION = re.compile(r"^(\d+)\.(\d+)")


class IncompatibleArtefact(RuntimeError):  # noqa: N818  (name fixed by the spec, §8.2)
    """The artefact cannot be served here: missing or altered files, or different libraries."""


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def major_minor(version: str) -> tuple[int, int]:
    """``(major, minor)`` of a version string; ValueError if it does not start with them."""
    match = _VERSION.match(version)
    if match is None:
        raise ValueError(f"not a version: {version!r}")
    return int(match.group(1)), int(match.group(2))


def installed_versions(names: Sequence[str]) -> dict[str, str]:
    """Installed versions of ``names`` (absent distributions are left out)."""
    out = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def read_manifest(directory: Path) -> dict[str, Any]:
    """``manifest.json`` of ``directory``, refused unless it holds every manifest field."""
    path = Path(directory) / MANIFEST_FILE
    if not path.is_file():
        raise IncompatibleArtefact(f"{MANIFEST_FILE} is missing")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IncompatibleArtefact(f"{MANIFEST_FILE} is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise IncompatibleArtefact(f"{MANIFEST_FILE} must hold a JSON object")
    missing = [name for name in MANIFEST_FIELDS if name not in manifest]
    if missing:
        raise IncompatibleArtefact(f"{MANIFEST_FILE} lacks required field(s): {missing}")
    return manifest


def check_manifest(manifest: Mapping[str, Any], directory: Path) -> None:
    """Refuse (IncompatibleArtefact) unless the artefact can be served by this installation."""
    if manifest["manifest_format"] != MANIFEST_FORMAT:
        raise IncompatibleArtefact(
            f"manifest format {manifest['manifest_format']!r} is not supported"
        )
    model_path = Path(directory) / MODEL_FILE
    if not model_path.is_file():
        raise IncompatibleArtefact(f"{MODEL_FILE} is missing")
    if sha256_file(model_path) != manifest["artefact_hash"]:
        raise IncompatibleArtefact(f"{MODEL_FILE} does not match the manifest's artefact_hash")
    packaged_for = str(manifest["robson_serving_version"])
    if major_minor(packaged_for)[0] != major_minor(__version__)[0]:
        raise IncompatibleArtefact(
            f"the artefact was packaged for robson_serving {packaged_for}; "
            f"installed is {__version__}"
        )
    recorded = manifest["library_versions"]
    runtime = [str(name) for name in manifest["runtime_libraries"]]
    installed = installed_versions(runtime)
    for name in runtime:
        if name not in installed:
            raise IncompatibleArtefact(f"runtime library {name!r} is not installed")
        if name not in recorded:
            raise IncompatibleArtefact(f"the manifest records no version of {name!r}")
        if major_minor(installed[name]) != major_minor(str(recorded[name])):
            raise IncompatibleArtefact(
                f"{name} {installed[name]} is installed but the artefact needs "
                f"{recorded[name]} (major.minor must match)"
            )


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_artefact(
    out_dir: Path,
    model: Any,
    schema: FeatureSchema,
    fields: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Write model, schema, manifest and reference distributions into a new ``out_dir``.

    ``fields`` holds every manifest field except ``WRITER_FIELDS``, which are filled here:
    the format, the model file's SHA-256, the installed versions of the core and runtime
    libraries, and the robson_serving version. The model is dumped with joblib; a pickle that
    references ``robson_ml`` is refused (and nothing is left behind). Golden files are added
    by :func:`robson_serving.contract.write_golden`. Returns the manifest.
    """
    given = WRITER_FIELDS & fields.keys()
    if given:
        raise ValueError(f"field(s) set by write_artefact, not by the caller: {sorted(given)}")
    missing = [n for n in MANIFEST_FIELDS if n not in WRITER_FIELDS and n not in fields]
    if missing:
        raise ValueError(f"manifest field(s) missing: {missing}")
    if list(fields["feature_list"]) != list(schema.names):
        raise ValueError("feature_list differs from the feature schema's columns")
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} already holds files")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / MODEL_FILE
    joblib.dump(model, model_path)
    if FORBIDDEN_PICKLE_MODULE in model_path.read_bytes():
        shutil.rmtree(out_dir)
        raise IncompatibleArtefact(
            "the model pickles robson_ml classes; the serving runtime cannot load it"
        )
    runtime = [str(name) for name in fields["runtime_libraries"]]
    manifest = {
        "manifest_format": MANIFEST_FORMAT,
        **fields,
        "artefact_hash": sha256_file(model_path),
        "library_versions": installed_versions(list(dict.fromkeys([*CORE_LIBRARIES, *runtime]))),
        "robson_serving_version": __version__,
    }
    _write_json(out_dir / SCHEMA_FILE, schema.to_dict())
    _write_json(out_dir / MANIFEST_FILE, manifest)
    _write_json(out_dir / REFERENCE_FILE, dict(reference))
    return manifest
