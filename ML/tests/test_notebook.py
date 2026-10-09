"""The EDA, training and evaluation notebooks: committed outputs are PII-free, each executes end to
end in synthetic mode, and the executed outputs hold aggregates only (no identifier, no row-level
table)."""

import re
import sys
from pathlib import Path

import nbformat
import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO / "notebooks"
# Each notebook and text its synthetic run must print (proof that its key steps ran).
EXPECTED = {
    "01_eda.ipynb": ("data: synthetic", "canonical schema: valid"),
    "02_model_training.ipynb": ("data: synthetic", "same as logged in MLflow", "SELECTED:"),
    "03_model_evaluation.ipynb": ("data: synthetic", "rebuilt AUC", "deployment-variant decision"),
}
sys.path.insert(0, str(REPO / "scripts"))

from run_notebook import execute  # noqa: E402
from strip_notebook import has_outputs, strip  # noqa: E402

# Synthetic identifiers: generator admission ids, ingest row keys, raw and hashed mother keys.
IDENTIFIER_PATTERNS = (
    re.compile(r"SYN\d{6}"),
    re.compile(r"ADM\d{6}"),
    re.compile(r"MK_[0-9a-f]{6,}"),
)
IDENTIFIER_COLUMNS = frozenset({"admission_id", "mother_key"})
MAX_TABLE_ROWS = 250
TH = re.compile(r"<th[^>]*>(.*?)</th>", re.S)
THEAD = re.compile(r"<thead[^>]*>(.*?)</thead>", re.S)
TR = re.compile(r"<tr[^>]*>", re.S)
TEXT_TYPES = ("text/plain", "text/html", "text/markdown")


def _texts(nb: nbformat.NotebookNode) -> list[tuple[int, str, str]]:
    """(cell index, output type, text) of every textual output."""
    out = []
    for i, cell in enumerate(nb.cells):
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                out.append((i, "stream", output.text))
            elif output.output_type == "error":
                out.append((i, "error", "\n".join(output.traceback)))
            else:
                for kind in TEXT_TYPES:
                    if kind in output.get("data", {}):
                        out.append((i, kind, str(output.data[kind])))
    return out


def scan_outputs(nb: nbformat.NotebookNode) -> list[str]:
    """Privacy violations in an executed notebook's outputs (empty when clean)."""
    problems = []
    for i, kind, text in _texts(nb):
        for pattern in IDENTIFIER_PATTERNS:
            if pattern.search(text):
                problems.append(f"cell {i} ({kind}): identifier pattern {pattern.pattern}")
        if kind == "text/html" and "<table" in text:
            # Column headers only: row labels (e.g. a field named mother_key in the
            # mapping report) are aggregate rows, not an identifier column.
            head = "".join(THEAD.findall(text))
            headers = {re.sub(r"<[^>]+>", "", h).strip() for h in TH.findall(head)}
            if headers & IDENTIFIER_COLUMNS:
                problems.append(f"cell {i}: table with an identifier column")
            if len(TR.findall(text)) > MAX_TABLE_ROWS:
                problems.append(f"cell {i}: table with more than {MAX_TABLE_ROWS} rows")
        if kind == "error":
            problems.append(f"cell {i}: error output")
    return problems


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_committed_notebook_outputs_are_pii_free(name: str) -> None:
    """Committed notebooks may carry real-data outputs (the repository is private),
    but only aggregates: no identifiers, no row-level tables, no errors."""
    nb = nbformat.read(NOTEBOOKS / name, as_version=4)
    if has_outputs(nb):
        assert scan_outputs(nb) == []


def test_strip_clears_outputs() -> None:
    nb = nbformat.v4.new_notebook()
    cell = nbformat.v4.new_code_cell("1 + 1")
    cell.outputs = [nbformat.v4.new_output("execute_result", {"text/plain": "2"})]
    cell.execution_count = 1
    nb.cells.append(cell)
    assert has_outputs(nb)
    assert not has_outputs(strip(nb))


def test_scanner_catches_identifiers_and_row_tables() -> None:
    nb = nbformat.v4.new_notebook()
    cell = nbformat.v4.new_code_cell("")
    cell.outputs = [
        nbformat.v4.new_output("stream", name="stdout", text="ADM000123"),
        nbformat.v4.new_output(
            "display_data",
            {
                "text/html": "<table><thead><tr><th>mother_key</th></tr></thead></table>",
                "text/plain": "",
            },
        ),
    ]
    nb.cells.append(cell)
    problems = scan_outputs(nb)
    assert any("identifier pattern" in p for p in problems)
    assert any("identifier column" in p for p in problems)


@pytest.fixture(scope="module", params=sorted(EXPECTED))
def executed(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[str, nbformat.NotebookNode]:
    output = tmp_path_factory.mktemp("notebook") / request.param
    execute("synthetic", output, notebook=NOTEBOOKS / request.param)
    return request.param, nbformat.read(output, as_version=4)


def test_notebook_executes_in_synthetic_mode(executed: tuple[str, nbformat.NotebookNode]) -> None:
    name, nb = executed
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert all(c.execution_count is not None for c in code_cells)
    assert not any(o.output_type == "error" for c in code_cells for o in c.outputs)
    text = "\n".join(t for _, _, t in _texts(nb))
    for expected in EXPECTED[name]:
        assert expected in text, f"{name}: {expected!r} missing"


def test_executed_outputs_are_aggregate_only(executed: tuple[str, nbformat.NotebookNode]) -> None:
    assert scan_outputs(executed[1]) == []


def test_no_output_frames_a_recommendation(executed: tuple[str, nbformat.NotebookNode]) -> None:
    text = "\n".join(t for _, _, t in _texts(executed[1])).lower()
    for phrase in ("recommend a cesarean", "should have a cesarean", "should receive a cs"):
        assert phrase not in text
