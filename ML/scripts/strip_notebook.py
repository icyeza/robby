"""Clear every output of the pipeline notebook(s) before committing (commit policy).

Usage:
    uv run python scripts/strip_notebook.py [NOTEBOOK ...]          # strip in place
    uv run python scripts/strip_notebook.py --check [NOTEBOOK ...]  # exit 1 if any output

Without arguments, every ``notebooks/*.ipynb`` is processed. Outputs, execution counts and
execution metadata are removed; sources and notebook metadata are kept.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
VOLATILE_CELL_METADATA = ("execution", "collapsed", "scrolled")


def has_outputs(nb: nbformat.NotebookNode) -> bool:
    """True when any code cell carries outputs or an execution count."""
    return any(
        cell.get("outputs") or cell.get("execution_count") is not None
        for cell in nb.cells
        if cell.cell_type == "code"
    )


def strip(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    """``nb`` with every output, execution count and execution metadata removed."""
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
        for key in VOLATILE_CELL_METADATA:
            cell.metadata.pop(key, None)
    nb.metadata.pop("widgets", None)
    return nb


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", type=Path)
    parser.add_argument("--check", action="store_true", help="fail if any output is present")
    args = parser.parse_args(argv)
    paths = args.notebooks or sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    dirty = []
    for path in paths:
        nb = nbformat.read(path, as_version=4)
        if args.check:
            if has_outputs(nb):
                dirty.append(path)
            continue
        nbformat.write(strip(nb), path)
        print(f"stripped {path}")
    for path in dirty:
        print(f"has outputs: {path}", file=sys.stderr)
    return 1 if dirty else 0


if __name__ == "__main__":
    sys.exit(main())
