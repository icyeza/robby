from pathlib import Path

import pandas as pd
import pytest

from robson_ml.select import (
    NO_ADDED_DISCRIMINATION,
    SelectionRule,
    config_name,
    load_selection_rule,
    select_configuration,
)

REPO_RULE = Path(__file__).resolve().parents[1] / "configs" / "selection_rule.yaml"


def _run(model: str, fs: str, auc: float, slope: float, rank: int, **kw: str) -> dict:
    row = {
        "model": model,
        "feature_set": fs,
        "missing_strategy": kw.get("strategy", "M1"),
        "split": kw.get("split", "S1"),
        "population": kw.get("population", "P_pred"),
        "family": kw.get("family", "baseline" if model.startswith("B") else "linear"),
        "complexity_rank": rank,
        "mean_auc": auc,
        "mean_calibration_slope": slope,
        "run_id": f"run_{model}_{fs}",
    }
    row["config"] = config_name(row)
    return row


@pytest.fixture
def rule() -> SelectionRule:
    return load_selection_rule(REPO_RULE)


def test_repository_rule_parses(rule: SelectionRule) -> None:
    assert rule.split == "S1"
    assert rule.population == "P_pred"
    assert rule.feature_sets == ("FS0", "FS1", "FS2", "FS3", "FS4")
    assert rule.slope_range == (0.8, 1.2)
    assert rule.within_auc == 0.01
    assert rule.baseline == "B1"
    assert not rule.baselines_selectable


def test_tie_band_prefers_lowest_complexity(rule: SelectionRule) -> None:
    runs = pd.DataFrame(
        [
            _run("B1", "FS0", 0.70, 1.0, 0),
            _run("xgboost", "FS4", 0.760, 1.05, 5, family="tree"),
            _run("logreg_l2", "FS2", 0.752, 0.95, 1),  # within 0.01 of the best
            _run("mlp", "FS4", 0.755, 1.10, 7, family="neural"),
        ]
    )
    selection = select_configuration(runs, rule)
    assert selection.selected is not None
    assert selection.selected["model"] == "logreg_l2"
    assert len(selection.tie_band) == 3
    assert selection.beats_baseline is True
    assert not selection.fallback


def test_outside_the_tie_band_the_best_wins(rule: SelectionRule) -> None:
    runs = pd.DataFrame(
        [
            _run("B1", "FS0", 0.70, 1.0, 0),
            _run("xgboost", "FS4", 0.78, 1.05, 5, family="tree"),
            _run("logreg_l2", "FS2", 0.75, 0.95, 1),
        ]
    )
    assert select_configuration(runs, rule).selected["model"] == "xgboost"


def test_ineligible_slope_and_scope_are_excluded(rule: SelectionRule) -> None:
    runs = pd.DataFrame(
        [
            _run("B1", "FS0", 0.99, 1.0, 0),  # baselines are not selectable
            _run("xgboost", "FS4", 0.90, 1.5, 5, family="tree"),  # slope too steep
            _run("mlp", "FS4_deploy", 0.95, 1.0, 7, family="neural"),  # not FS0-FS4
            _run("mlp", "FS4", 0.95, 1.0, 7, family="neural", split="S3"),  # not S1
            _run("mlp", "FS3", 0.95, 1.0, 7, family="neural", population="P_pred_onset_coded"),
            _run("logreg_l2", "FS1", 0.72, 0.9, 1),
        ]
    )
    selection = select_configuration(runs, rule)
    assert selection.selected["config"] == "logreg_l2|FS1|M1|S1|P_pred"
    assert len(selection.eligible) == 1
    assert selection.beats_baseline is False
    assert any(NO_ADDED_DISCRIMINATION in note for note in selection.notes)


def test_fallback_when_nothing_is_eligible(rule: SelectionRule) -> None:
    runs = pd.DataFrame(
        [
            _run("xgboost", "FS4", 0.80, 1.6, 5, family="tree"),
            _run("logreg_l2", "FS1", 0.72, 0.7, 1),
        ]
    )
    selection = select_configuration(runs, rule)
    assert selection.fallback
    assert selection.selected["model"] == "logreg_l2"  # |0.7 - 1| < |1.6 - 1|
    assert selection.baseline is None and selection.beats_baseline is None


def test_nothing_to_select(rule: SelectionRule) -> None:
    runs = pd.DataFrame([_run("B1", "FS0", 0.7, 1.0, 0)])
    selection = select_configuration(runs, rule)
    assert selection.selected is None
    assert selection.baseline is not None


def test_unsupported_rule_is_refused(tmp_path: Path) -> None:
    text = REPO_RULE.read_text(encoding="utf-8").replace(
        "rank_by: mean_loho_auc", "rank_by: pooled_auc"
    )
    path = tmp_path / "rule.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="rank_by"):
        load_selection_rule(path)
