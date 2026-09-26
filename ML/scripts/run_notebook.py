"""Execute the pipeline notebook and save the executed copy (never commit it).

Usage:
    uv run python scripts/run_notebook.py --mode real \
        --output reports/notebooks/01_cesarean_readiness_pipeline.executed.ipynb

``--mode synthetic`` (the default) runs on a synthetic project in a temporary directory.
``--mode real`` reads configs/, data/processed/, reports/ and mlruns/; add ``--run-live``
to run one demonstration configuration (tracked in a temporary store, not mlruns/). The
executed notebook contains aggregate outputs only; keep it under reports/ (git-ignored).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "01_cesarean_readiness_pipeline.ipynb"
CELL_TIMEOUT_S = 7200


def execute(mode: str, output: Path, notebook: Path = NOTEBOOK, run_live: bool = False) -> float:
    """Execute ``notebook`` in ``mode`` and write it to ``output``; returns seconds taken."""
    os.environ["ROBSON_NOTEBOOK_MODE"] = mode
    os.environ["ROBSON_NOTEBOOK_RUN_LIVE"] = "1" if run_live else "0"
    nb = nbformat.read(notebook, as_version=4)
    started = time.time()
    client = NotebookClient(
        nb,
        timeout=CELL_TIMEOUT_S,
        kernel_name="python3",
        resources={"metadata": {"path": str(notebook.parent)}},
    )
    client.execute()
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, output)
    return time.time() - started


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-live", action="store_true")
    args = parser.parse_args(argv)
    seconds = execute(args.mode, args.output, run_live=args.run_live)
    print(f"executed in {seconds:.0f} s; written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
