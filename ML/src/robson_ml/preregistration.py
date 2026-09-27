"""Pre-registration guard (spec §13.2, v1.1 item 2).

No model, baselines included, may run on real data unless ``configs/selection_rule.yaml``
is committed in git ``HEAD`` with no uncommitted (staged, unstaged or untracked) changes.
The rule file's last commit hash is recorded in every run as ``selection_rule_commit``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

SELECTION_RULE_PATH = Path("configs/selection_rule.yaml")
REAL_DATA_PARTS = ("data", "processed")
NOT_COMMITTED = "not_committed"


class PreregistrationError(RuntimeError):
    """The selection rule is not committed as required before a real-data run."""


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False, timeout=60
    )


def is_real_data(path: Path) -> bool:
    """True when ``path`` lies under a ``data/processed`` directory (real processed data)."""
    parts = [p.lower() for p in Path(path).resolve().parts]
    n = len(REAL_DATA_PARTS)
    return any(tuple(parts[i : i + n]) == REAL_DATA_PARTS for i in range(len(parts) - n + 1))


def selection_rule_commit(repo: Path, rule: Path = SELECTION_RULE_PATH) -> str:
    """The hash of the last commit changing ``rule``'s content, or ``not_committed``.

    Pure renames (moving the project into a subfolder) are followed, not counted.
    """
    result = _git(
        repo, "log", "-1", "--follow", "--diff-filter=AM", "--format=%H", "--", rule.as_posix()
    )
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else NOT_COMMITTED


def check_preregistration(repo: Path, rule: Path = SELECTION_RULE_PATH) -> str:
    """Raise unless ``rule`` is in ``HEAD`` of the git repo at ``repo`` and unmodified.

    Returns the rule's last commit hash. Raises :class:`PreregistrationError` when ``repo``
    is not a git repository, the rule is absent from ``HEAD``, or it has staged, unstaged
    or untracked changes.
    """
    # "HEAD:./path" is relative to ``repo`` (the project may sit below the git root).
    in_head = _git(repo, "cat-file", "-e", f"HEAD:./{rule.as_posix()}")
    if in_head.returncode != 0:
        raise PreregistrationError(
            f"{rule.as_posix()} is not committed in HEAD; commit the selection rule before "
            "running any model on real data (spec §13.2)"
        )
    status = _git(repo, "status", "--porcelain", "--", rule.as_posix())
    if status.returncode != 0 or status.stdout.strip():
        raise PreregistrationError(
            f"{rule.as_posix()} has uncommitted changes; commit or revert them before running "
            "any model on real data (spec §13.2)"
        )
    return selection_rule_commit(repo, rule)


def git_commit(repo: Path) -> str:
    """``HEAD`` commit hash of ``repo`` (``unknown`` outside a git repository)."""
    result = _git(repo, "rev-parse", "HEAD")
    return result.stdout.strip() if result.returncode == 0 else "unknown"
