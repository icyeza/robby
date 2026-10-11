"""The package installs, reports its version and never pulls in the training stack."""

import importlib.metadata
import re

import robson_serving

RUNTIME = {"joblib", "numpy", "pandas", "pyarrow", "pyyaml", "robson-engine", "scikit-learn"}


def _name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9_.-]+", requirement)
    assert match is not None
    return match.group(0).lower()


def test_version_matches_the_installed_distribution() -> None:
    assert robson_serving.__version__ == importlib.metadata.version("robson-serving")


def test_declares_only_runtime_dependencies() -> None:
    requires = importlib.metadata.requires("robson-serving") or []
    core = {_name(r) for r in requires if "extra ==" not in r}
    assert core == RUNTIME


def test_xgboost_is_an_optional_extra() -> None:
    requires = importlib.metadata.requires("robson-serving") or []
    assert any(
        r.startswith("xgboost") and re.search(r"extra == ['\"]xgboost['\"]", r) for r in requires
    )
