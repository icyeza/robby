"""How robson_ml and robson_serving fit together: the dependency, the shared runtime code and
old artefacts. Synthetic data or artefact files only (never a data row)."""

import importlib.metadata
import re


def _names(distribution: str) -> set[str]:
    names = set()
    for requirement in importlib.metadata.requires(distribution) or []:
        match = re.match(r"[A-Za-z0-9_.-]+", requirement)
        assert match is not None
        names.add(match.group(0).lower())
    return names


def test_robson_ml_depends_on_robson_serving() -> None:
    assert "robson-serving" in _names("robson-ml")
