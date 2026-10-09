"""Shared set-up for the EDA, training and evaluation notebooks.

Each notebook starts with ``nb = notebook.setup()``, which reads the mode, quietens library
output, applies the figure style and returns a :class:`Project` holding every path. In
**synthetic** mode (the default, and how the notebooks are tested) a synthetic project is
built in a temporary directory and processed by the same CLI commands as the real data
(``ingest``, ``robson``, ``profile``, ``leakage``). In **real** mode those outputs must
already exist.

Environment variables: ``ROBSON_NOTEBOOK_MODE`` (``synthetic`` or ``real``),
``ROBSON_NOTEBOOK_RUN_LIVE`` (``1``: run one demonstration configuration live in real mode)
and ``ROBSON_NOTEBOOK_SYNTHETIC_N`` (synthetic sample size).
"""

from __future__ import annotations

import logging
import os
import platform
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd

from robson_ml.cli import ProjectConfig, load_project_config

MODES = ("synthetic", "real")
SYNTHETIC_N = 1000
SYNTHETIC_SEED = 11
PROCESS_COMMANDS = ("ingest", "robson", "profile", "leakage")
LIBRARIES = ("numpy", "pandas", "scikit-learn", "xgboost", "shap", "mlflow", "optuna")


def find_repo_root(start: Path) -> Path:
    """The first of ``start`` and its parents holding pyproject.toml and src/robson_ml."""
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "robson_ml").exists():
            return candidate
    raise FileNotFoundError("run the notebook from inside the repository")


@dataclass(frozen=True)
class Project:
    """The mode and every path a notebook reads (all absolute)."""

    mode: str
    run_live: bool
    repo: Path
    root: Path
    cfg: ProjectConfig

    def at(self, path: Path) -> Path:
        return path if path.is_absolute() else self.root / path

    @property
    def seed(self) -> int:
        return self.cfg.seed

    @property
    def raw_path(self) -> Path:
        return self.at(self.cfg.raw_path)

    @property
    def processed(self) -> Path:
        return self.at(self.cfg.processed_dir)

    @property
    def interim(self) -> Path:
        return self.at(self.cfg.interim_dir)

    @property
    def reports(self) -> Path:
        return self.at(self.cfg.reports_dir)

    @property
    def classified_path(self) -> Path:
        return self.processed / "canonical_robson.parquet"

    @property
    def oof_dir(self) -> Path:
        return self.interim / "oof"

    @property
    def tracking_uri(self) -> str:
        return (self.root / "mlruns").resolve().as_uri()

    @property
    def selection_rule_path(self) -> Path:
        return self.root / "configs" / "selection_rule.yaml"

    def load_oof(self, run_id: str, columns: tuple[str, ...] = ("y", "p")) -> pd.DataFrame | None:
        """A run's pooled out-of-fold outcome and probability columns (never identifiers
        unless asked for), or None when the file is missing."""
        path = self.oof_dir / f"{run_id}.parquet"
        return pd.read_parquet(path, columns=list(columns)) if path.exists() else None


def run_cli(root: Path, *args: str) -> None:
    """Run one ``robson-ml`` command inside ``root`` (its paths are relative)."""
    from typer.testing import CliRunner

    from robson_ml.cli import app

    cwd = os.getcwd()
    os.chdir(root)
    try:
        result = CliRunner().invoke(app, list(args))
    finally:
        os.chdir(cwd)
    if result.exit_code != 0:
        raise RuntimeError(f"robson-ml {' '.join(args)} failed (details suppressed)")


def _quieten() -> None:
    # Library warnings (convergence, deprecations) and MLflow chatter carry no data.
    # statsmodels registers "always" filters on import, so it is imported before ours.
    import statsmodels.api  # noqa: F401

    warnings.filterwarnings("ignore")
    os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"
    import mlflow  # noqa: F401  (imported first so that its logger exists)

    logging.getLogger("mlflow").setLevel(logging.ERROR)


def setup(start: Path | None = None, commands: tuple[str, ...] = PROCESS_COMMANDS) -> Project:
    """Read the mode, prepare the project (built and processed in synthetic mode) and
    return it. Raises FileNotFoundError in real mode when the CLI outputs are missing."""
    mode = os.environ.get("ROBSON_NOTEBOOK_MODE", "synthetic")
    if mode not in MODES:
        raise ValueError(f"ROBSON_NOTEBOOK_MODE must be 'synthetic' or 'real', not {mode!r}")
    run_live = os.environ.get("ROBSON_NOTEBOOK_RUN_LIVE", "0") == "1"
    repo = find_repo_root((start or Path.cwd()).resolve())
    _quieten()

    from robson_ml import figures

    figures.apply_style()
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 80)

    if mode == "synthetic":
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from tests.synthetic_project import build_synthetic_project

        started = time.time()
        root = Path(tempfile.mkdtemp(prefix="robson_notebook_"))
        n = int(os.environ.get("ROBSON_NOTEBOOK_SYNTHETIC_N", str(SYNTHETIC_N)))
        build_synthetic_project(root, repo, n=n, seed=SYNTHETIC_SEED)
        for command in commands:
            run_cli(root, command)
        print(f"synthetic project built and processed in {time.time() - started:.0f} s")
    else:
        root = repo

    project = Project(
        mode, run_live, repo, root, load_project_config(root / "configs" / "project.yaml")
    )
    required = [
        project.raw_path,
        project.processed / "canonical.parquet",
        project.classified_path,
        project.interim / "mapping_report.json",
        project.reports / "leakage" / "leakage_screens.csv",
    ]
    missing = [str(p.relative_to(root)) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"run the robson-ml CLI first (ingest, robson, profile, leakage); missing: {missing}"
        )
    print(f"data: {mode}")
    return project


def show(fig: Any) -> None:
    """Display a figure inline, then release it."""
    import matplotlib.pyplot as plt
    from IPython.display import display

    display(fig)
    plt.close(fig)


def environment() -> pd.DataFrame:
    """Python and library versions, for the record."""
    import robson_engine
    from robson_engine import load_rule_set

    values: dict[str, Any] = {
        "executed at": time.strftime("%Y-%m-%d %H:%M"),
        "python": platform.python_version(),
        **{lib: version(lib) for lib in LIBRARIES},
        "robson_engine": robson_engine.__version__,
        "rule set": load_rule_set().version_label,
    }
    return pd.DataFrame({"value": values}).rename_axis("item")


def run_context(project: Project, registry: Any, live_dir: Path | None = None) -> Any:
    """The :class:`~robson_ml.evaluate.RunContext` for runs started from a notebook.

    Synthetic mode tracks into the synthetic project's ``mlruns/``. A live real-data run
    (``live_dir``) is tracked in a temporary store, never in ``mlruns/``, and is refused
    unless the selection rule is committed (pre-registration)."""
    from robson_ml.evaluate import RunContext
    from robson_ml.ingest import file_sha256
    from robson_ml.preregistration import check_preregistration, git_commit, selection_rule_commit

    if live_dir is None:
        tracking_uri, oof_dir = project.tracking_uri, project.oof_dir
        rule_commit = selection_rule_commit(project.repo)
    else:
        tracking_uri, oof_dir = (live_dir / "mlruns").as_uri(), live_dir / "oof"
        rule_commit = check_preregistration(project.repo)
    return RunContext(
        tracking_uri=tracking_uri,
        oof_dir=oof_dir,
        data_hash=file_sha256(project.classified_path),
        features_yaml_hash=registry.sha256,
        git_commit=git_commit(project.repo),
        selection_rule_commit=rule_commit,
    )


def train_synthetic(project: Project, data: Any, data_onset: Any, registry: Any) -> list[Any]:
    """Synthetic mode only: run the small synthetic experiment grid (see
    ``tests.synthetic_project.synthetic_grid``) into the synthetic project's ``mlruns/``."""
    if project.mode != "synthetic":
        raise RuntimeError("the synthetic grid never runs on real data")
    from tests.synthetic_project import train_synthetic_grid

    started = time.time()
    results = train_synthetic_grid(data, data_onset, run_context(project, registry), project.seed)
    print(f"{len(results)} synthetic configurations trained in {time.time() - started:.0f} s")
    return results
