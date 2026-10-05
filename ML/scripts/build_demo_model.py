"""Build a readiness model from synthetic data, for running the API without the private data.

Usage, from the ML folder::

    uv run python scripts/build_demo_model.py

It writes a synthetic project to a temporary folder, runs ``ingest``, ``robson`` and
``fit-deploy`` there, and copies the artefact to ``artefacts/demo-synthetic``. The model has
learned nothing about real admissions; it only makes the readiness endpoints work.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from tests.synthetic_project import build_synthetic_project  # noqa: E402

LABEL = "demo-synthetic"
TARGET = ML_DIR / "artefacts" / LABEL
SELECTION_RULE = Path("configs/selection_rule.yaml")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=demo", "-c", "user.email=demo@example.invalid", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _cli(project: Path, *args: str) -> None:
    print(f"robson-ml {' '.join(args)} ...", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "robson_ml.cli", *args],
        cwd=project,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(f"robson-ml {args[0]} failed:\n{result.stdout[-2000:]}{result.stderr[-2000:]}")


def main() -> None:
    """Build the synthetic project, fit the deployment model and copy the artefact."""
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        project = build_synthetic_project(Path(tmp) / "project", ML_DIR, n=1500, seed=11)
        # fit-deploy refuses unless the selection rule is committed (pre-registration check).
        _git(project, "init", "-q")
        _git(project, "add", SELECTION_RULE.as_posix())
        _git(project, "commit", "-q", "-m", "selection rule")
        config = project / "configs" / f"{LABEL}.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model": "logreg_l2",
                    "feature_set": "FS2",
                    "use_facility": True,
                    "missing_strategy": "M2",
                    "population": "P_pred",
                    "seed": 11,
                    "n_trials": 5,
                    "version_label": LABEL,
                }
            ),
            encoding="utf-8",
        )
        _cli(project, "ingest")
        _cli(project, "robson")
        _cli(project, "fit-deploy", config.relative_to(project).as_posix())
        if TARGET.exists():
            shutil.rmtree(TARGET)
        shutil.copytree(project / "artefacts" / LABEL, TARGET)
    print(f"Wrote {TARGET.relative_to(ML_DIR)} in {time.perf_counter() - start:.0f} s.")
    print("Start the API with it (from the ML folder):")
    print(f"  ROBSON_MODEL_DIR=artefacts/{LABEL} uv run uvicorn api.main:app")
    print(f'  PowerShell: $env:ROBSON_MODEL_DIR="artefacts/{LABEL}"; uv run uvicorn api.main:app')


if __name__ == "__main__":
    main()
