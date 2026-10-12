"""Contract with robson_serving (spec 8.4): a packaged artefact loads and passes its golden check.

The artefact is built from robson_serving's own synthetic helpers (no real data). Those helpers
live in ML/packages/robson_serving/tests/conftest.py, which is not an installed package, so the
file is loaded by path rather than copied.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from robson_serving import Scorer, load
from robson_serving.artefact import write_artefact
from robson_serving.contract import check_golden, write_golden
from robson_serving.monitoring import reference_distributions

SERVING_CONFTEST = (
    Path(__file__).resolve().parents[2] / "ML/packages/robson_serving/tests/conftest.py"
)


def _synthetic_helpers() -> ModuleType:
    spec = importlib.util.spec_from_file_location("serving_synthetic_helpers", SERVING_CONFTEST)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def packaged(tmp_path_factory: pytest.TempPathFactory) -> Path:
    helpers = _synthetic_helpers()
    model, x = helpers.fit_model(helpers.make_records(600, seed=1))
    schema = helpers.make_schema(model)
    out = tmp_path_factory.mktemp("packaged") / "synthetic-v0"
    write_artefact(
        out,
        model,
        schema,
        helpers.manifest_fields(schema),
        reference_distributions(x, helpers.KINDS),
    )
    write_golden(load(out), out)
    return out


def test_packaged_artefact_loads_and_passes_the_golden_check(packaged: Path) -> None:
    scorer = load(packaged)
    assert isinstance(scorer, Scorer)
    result = check_golden(packaged, scorer)
    assert result.ok


@pytest.mark.parametrize("name", ["torch", "optuna", "mlflow", "xgboost", "robson_ml"])
def test_app_environment_does_not_install_the_training_stack(name: str) -> None:
    assert importlib.util.find_spec(name) is None
