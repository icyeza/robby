"""Notebook 02: committed outputs are PII-free, it executes in synthetic mode, and its outputs hold
aggregates only (same leak scan as the pipeline notebook)."""

import sys
from pathlib import Path

import nbformat
import pytest

from tests.test_notebook import _texts, scan_outputs

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "02_audit_and_research_questions.ipynb"
sys.path.insert(0, str(REPO / "scripts"))

from run_notebook import execute  # noqa: E402
from strip_notebook import has_outputs  # noqa: E402


def test_committed_notebook_outputs_are_pii_free() -> None:
    """Committed notebooks may carry real-data outputs (the repository is private),
    but only aggregates: no identifiers, no row-level tables, no errors."""
    nb = nbformat.read(NOTEBOOK, as_version=4)
    if has_outputs(nb):
        assert scan_outputs(nb) == []


@pytest.fixture(scope="module")
def executed(tmp_path_factory: pytest.TempPathFactory) -> nbformat.NotebookNode:
    output = tmp_path_factory.mktemp("notebook02") / "executed.ipynb"
    execute("synthetic", output, notebook=NOTEBOOK)
    return nbformat.read(output, as_version=4)


def test_notebook_executes_in_synthetic_mode(executed: nbformat.NotebookNode) -> None:
    code_cells = [c for c in executed.cells if c.cell_type == "code"]
    assert all(c.execution_count is not None for c in code_cells)
    assert not any(o.output_type == "error" for c in code_cells for o in c.outputs)
    text = "\n".join(t for _, _, t in _texts(executed))
    assert "mode: synthetic" in text
    assert "FAKE" in text  # synthetic references are labelled as fake
    assert "Sector confound" in text


def test_executed_outputs_are_aggregate_only(executed: nbformat.NotebookNode) -> None:
    assert scan_outputs(executed) == []


def test_no_output_frames_a_recommendation(executed: nbformat.NotebookNode) -> None:
    text = "\n".join(t for _, _, t in _texts(executed)).lower()
    for phrase in ("recommend a cesarean", "should have a cesarean", "should receive a cs"):
        assert phrase not in text
