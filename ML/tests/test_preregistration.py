import subprocess
from pathlib import Path

import pytest

from robson_ml.preregistration import (
    NOT_COMMITTED,
    PreregistrationError,
    check_preregistration,
    is_real_data,
    selection_rule_commit,
)

RULE = Path("configs/selection_rule.yaml")


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def init_repo(repo: Path) -> None:
    git(repo, "init", "-q")
    (repo / "README").write_text("synthetic test repo\n", encoding="utf-8")
    git(repo, "add", "README")
    git(repo, "commit", "-q", "-m", "init")


def commit_rule(repo: Path, text: str = "version: 1\n") -> str:
    (repo / RULE).parent.mkdir(parents=True, exist_ok=True)
    (repo / RULE).write_text(text, encoding="utf-8")
    git(repo, "add", RULE.as_posix())
    git(repo, "commit", "-q", "-m", "rule")
    return git(repo, "rev-parse", "HEAD")


def test_preregistration_guard(tmp_path: Path) -> None:
    """Spec §22: a real-data run refuses when the rule is absent from HEAD or modified."""
    init_repo(tmp_path)
    with pytest.raises(PreregistrationError, match="not committed"):
        check_preregistration(tmp_path)
    (tmp_path / RULE).parent.mkdir(parents=True)
    (tmp_path / RULE).write_text("version: 1\n", encoding="utf-8")
    with pytest.raises(PreregistrationError, match="not committed"):
        check_preregistration(tmp_path)  # untracked
    git(tmp_path, "add", RULE.as_posix())
    with pytest.raises(PreregistrationError, match="not committed"):
        check_preregistration(tmp_path)  # staged, not committed
    git(tmp_path, "commit", "-q", "-m", "rule")
    head = git(tmp_path, "rev-parse", "HEAD")
    assert check_preregistration(tmp_path) == head
    (tmp_path / RULE).write_text("version: 2\n", encoding="utf-8")
    with pytest.raises(PreregistrationError, match="uncommitted"):
        check_preregistration(tmp_path)
    git(tmp_path, "add", RULE.as_posix())
    with pytest.raises(PreregistrationError, match="uncommitted"):
        check_preregistration(tmp_path)


def test_rule_commit_is_last_commit_touching_the_rule(tmp_path: Path) -> None:
    init_repo(tmp_path)
    assert selection_rule_commit(tmp_path) == NOT_COMMITTED
    rule_commit = commit_rule(tmp_path)
    (tmp_path / "other.txt").write_text("x\n", encoding="utf-8")
    git(tmp_path, "add", "other.txt")
    git(tmp_path, "commit", "-q", "-m", "other")
    assert check_preregistration(tmp_path) == rule_commit


def test_outside_a_git_repo_refuses(tmp_path: Path) -> None:
    with pytest.raises(PreregistrationError):
        check_preregistration(tmp_path)


def test_real_data_detection(tmp_path: Path) -> None:
    assert is_real_data(tmp_path / "data" / "processed" / "canonical_robson.parquet")
    assert is_real_data(Path("data/processed/canonical_robson.parquet"))
    assert not is_real_data(tmp_path / "synthetic" / "canonical_robson.parquet")
    assert not is_real_data(tmp_path / "data" / "interim" / "x.parquet")
