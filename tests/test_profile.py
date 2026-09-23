import numpy as np
import pandas as pd
import pytest

from robson_engine import load_rule_set
from robson_ml.mapping import FieldMapping, MappingConfig
from robson_ml.privacy import SUPPRESSED
from robson_ml.profile import (
    QUESTIONS,
    cramers_v,
    open_questions_markdown,
    robson_inputs_markdown,
    single_feature_auc,
    variable_profile,
    write_profile,
)
from robson_ml.robson_run import classify_frame
from tests.synthetic import make_admissions


def test_single_feature_auc() -> None:
    cs = pd.Series([0] * 20 + [1] * 20)
    assert single_feature_auc(pd.Series(range(40), dtype=float), cs) == pytest.approx(1.0)
    few = cs.iloc[14:26].reset_index(drop=True)
    assert single_feature_auc(pd.Series(range(12), dtype=float), few) is None


def test_cramers_v() -> None:
    cs = pd.Series([0] * 30 + [1] * 30)
    assert cramers_v(pd.Series(["a"] * 30 + ["b"] * 30), cs) == pytest.approx(1.0)
    assert cramers_v(pd.Series(["a", "b"] * 30), cs) == pytest.approx(0.0)
    assert cramers_v(pd.Series(["a"] * 60), cs) is None


def _classified() -> pd.DataFrame:
    return classify_frame(make_admissions(3000, seed=21), load_rule_set())


def _raw_and_config() -> tuple[pd.DataFrame, pd.DataFrame, MappingConfig]:
    canonical = _classified()
    raw = pd.DataFrame(
        {
            "Age raw": canonical["maternal_age"].astype(object),
            "Rare flag": ["x"] * 3 + [None] * (len(canonical) - 3),
            "Comment": [f"c{i}" for i in range(len(canonical))],
        }
    )
    mapping = FieldMapping("maternal_age", "float", ("Age raw",), "confirmed")
    config = MappingConfig("t", None, {"maternal_age": mapping})
    return raw, canonical, config


def test_variable_profile_rows_and_suppression() -> None:
    raw, canonical, config = _raw_and_config()
    profile = variable_profile(raw, canonical, config)
    assert profile["raw_name"].tolist() == ["Age raw", "Rare flag", "Comment"]
    assert profile.loc[0, "canonical_name"] == "maternal_age"
    assert profile.loc[0, "association_metric"] == "auc"
    assert profile.loc[2, "kind"] == "text"
    assert profile.loc[1, "pct_missing"] == SUPPRESSED
    assert (profile["proposed_status"] == "review").all()


def test_robson_inputs_markdown_sections() -> None:
    text = robson_inputs_markdown(_classified())
    for heading in ["## Completeness", "## Engine status", "## Robson report table"]:
        assert heading in text


def test_open_questions_all_answered_or_flagged() -> None:
    canonical = _classified()
    raw = pd.DataFrame(
        {"ANC visits": np.arange(len(canonical)), "Attending doctor": ["d"] * len(canonical)}
    )
    text = open_questions_markdown(canonical, raw, {"q8": "Facility X is private."})
    for number in range(1, len(QUESTIONS) + 1):
        assert f"## Q{number}." in text
    assert "Facility X is private." in text
    assert "Attending doctor" in text
    assert "Not yet answered" in text


def test_write_profile_outputs_are_suppressed(tmp_path) -> None:
    raw, canonical, config = _raw_and_config()
    out_dir = tmp_path / "profile"
    write_profile(raw, canonical, config, {}, out_dir)

    csv_text = (out_dir / "variable_profile.csv").read_text(encoding="utf-8")
    written = pd.read_csv(out_dir / "variable_profile.csv", dtype=str)
    assert "n_nonnull" in written.columns
    counts = pd.to_numeric(written["n_nonnull"], errors="coerce")
    assert not ((counts >= 1) & (counts <= 4)).any()
    assert csv_text  # sanity: file is non-empty

    assert (out_dir / "robson_inputs.md").exists()
    assert (out_dir / "open_questions.md").exists()
