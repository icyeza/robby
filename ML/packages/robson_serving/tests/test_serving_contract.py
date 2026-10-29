"""Contract test: the artefact reproduces its golden outputs to 1e-9, also in a
process where the training stack cannot be imported. The application runs the same check by
calling robson_serving.contract.check_golden on a registered artefact."""

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from robson_serving import load
from robson_serving.artefact import ARTEFACT_FILES, GOLDEN_INPUTS_FILE, GOLDEN_OUTPUTS_FILE
from robson_serving.contract import GOLDEN_N, TOLERANCE, check_golden, generate_golden_inputs

REAL_PACKAGED = Path(__file__).resolve().parents[3] / "artefacts" / "packaged" / "readiness-v1.0"
SERVING_ONLY = textwrap.dedent(
    """
    import importlib.abc
    import sys

    BLOCKED = {"robson_ml", "torch", "optuna", "mlflow", "xgboost", "shap", "statsmodels"}

    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ImportError("blocked in the serving-only check: " + name)
            return None

    sys.meta_path.insert(0, Block())
    from robson_serving.contract import check_golden

    print("ok" if check_golden(sys.argv[1]).ok else "mismatch")
    """
)


def _serving_only(directory: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", SERVING_ONLY, str(directory)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def test_tolerance_is_the_spec_value() -> None:
    assert TOLERANCE == 1e-9 and GOLDEN_N == 200


def test_golden_inputs_are_deterministic_and_valid(packaged_dir: Path) -> None:
    scorer = load(packaged_dir)
    first = generate_golden_inputs(scorer, n=30, seed=5)
    second = generate_golden_inputs(scorer, n=30, seed=5)
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == [field.name for field in scorer.schema.inputs]
    assert len(first) == 30
    for record in first.to_dict("records"):
        scorer.build_row(record)


def test_the_packaged_artefact_reproduces_its_golden_outputs(packaged_dir: Path) -> None:
    assert sorted(p.name for p in packaged_dir.iterdir()) == sorted(ARTEFACT_FILES)
    stored = pd.read_parquet(packaged_dir / GOLDEN_INPUTS_FILE)
    assert len(stored) == GOLDEN_N
    assert stored["facility_id"].isna().any()  # unseen-facility admissions are covered
    result = check_golden(packaged_dir)
    assert result.ok and result.n == GOLDEN_N
    assert result.max_abs_diff <= TOLERANCE


def test_check_golden_detects_a_changed_output(packaged_dir: Path, tmp_path: Path) -> None:
    copy = tmp_path / "copy"
    shutil.copytree(packaged_dir, copy)
    path = copy / GOLDEN_OUTPUTS_FILE
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["probabilities"][0] += 1e-6
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = check_golden(copy)
    assert not result.ok
    assert result.max_abs_diff == pytest.approx(1e-6, rel=1e-3)


def test_the_contract_holds_without_the_training_stack(packaged_dir: Path) -> None:
    done = _serving_only(packaged_dir)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip() == "ok"


@pytest.mark.skipif(
    not (REAL_PACKAGED / "manifest.json").is_file(), reason="packaged readiness-v1.0 not present"
)
def test_packaged_readiness_v1_0_contract() -> None:
    result = check_golden(REAL_PACKAGED)
    assert result.ok and result.n == GOLDEN_N
    done = _serving_only(REAL_PACKAGED)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip() == "ok"
