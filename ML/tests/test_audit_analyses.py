"""Audit table additions: the 6/7/9 non-cephalic row, the onset caveat, the
Vogel columns (FAKE reference) and consistency with the published profile rows."""

from pathlib import Path

import pandas as pd
import pytest

from robson_engine import load_rule_set
from robson_ml.audit_offline import (
    GROUP_LABELS,
    NON_CEPHALIC,
    ONSET_CAVEAT,
    ONSET_NOTE,
    OVERALL,
    RESIDUAL,
    add_vogel_columns,
    audit_markdown,
    published_audit_table,
    robson_report_table,
    row_labels,
)
from robson_ml.populations import audit_population
from robson_ml.privacy import SECONDARY, SUPPRESSED
from robson_ml.profile import status_tables
from robson_ml.references import load_vogel
from robson_ml.robson_run import classify_frame
from tests.synthetic import make_admissions
from tests.synthetic_references import fake_vogel

HIDDEN = (SUPPRESSED, SECONDARY)


@pytest.fixture(scope="module")
def classified() -> pd.DataFrame:
    return classify_frame(make_admissions(900, seed=5), load_rule_set())


def test_non_cephalic_rows_are_split_from_the_residual(classified: pd.DataFrame) -> None:
    audit = audit_population(classified)[0]
    plain = row_labels(audit)
    split = row_labels(audit, split_non_cephalic=True)
    moved = split == NON_CEPHALIC
    assert moved.any()
    assert (plain[moved] == RESIDUAL).all()
    assert (audit.loc[moved, "fetal_presentation"] == "non_cephalic").all()
    table = robson_report_table(audit, split_non_cephalic=True)
    overall = table[table["facility"] == OVERALL].set_index("row")
    base = robson_report_table(audit)
    base_residual = base[(base["facility"] == OVERALL) & (base["row"] == RESIDUAL)].iloc[0]
    assert overall.loc[NON_CEPHALIC, "n"] + overall.loc[RESIDUAL, "n"] == base_residual["n"]


def test_published_rows_match_the_profile(classified: pd.DataFrame) -> None:
    table = published_audit_table(classified)
    _, _, plain = status_tables(classified)
    groups = table[table["row"].isin(GROUP_LABELS)].drop(columns="note").reset_index(drop=True)
    expected = plain[plain["row"].isin(GROUP_LABELS)].reset_index(drop=True)
    pd.testing.assert_frame_equal(groups, expected, check_dtype=False)
    assert set(table["row"]) == {*GROUP_LABELS, NON_CEPHALIC, RESIDUAL}
    notes = table.set_index(["facility", "row"])["note"]
    assert notes[(OVERALL, "2")] == ONSET_NOTE and notes[(OVERALL, "4")] == ONSET_NOTE
    assert notes[(OVERALL, "1")] == ""


def test_published_table_has_no_small_counts(classified: pd.DataFrame) -> None:
    table = published_audit_table(classified)
    for column in ("n", "n_cs"):
        numeric = pd.to_numeric(table[column], errors="coerce")
        assert not ((numeric > 0) & (numeric < 5)).any()


def test_vogel_columns_follow_published_values(tmp_path: Path) -> None:
    vogel = load_vogel(fake_vogel(tmp_path / "v.yaml"))
    table = pd.DataFrame(
        {
            "facility": [OVERALL] * 3,
            "row": ["1", "2", RESIDUAL],
            "pct_of_deliveries": ["0.25", SUPPRESSED, "0.10"],
            "cs_rate": [0.30, SUPPRESSED, 0.5],
        }
    )
    out = add_vogel_columns(table, vogel).set_index("row")
    assert out.loc["1", "ref_pct_of_deliveries"] == pytest.approx(0.10)
    assert out.loc["1", "ref_cs_rate"] == pytest.approx(0.14)
    assert out.loc["1", "diff_pct_of_deliveries"] == "0.15"
    assert out.loc["1", "diff_cs_rate"] == pytest.approx(0.16)
    assert out.loc["2", "diff_cs_rate"] == SUPPRESSED
    assert out.loc["2", "diff_pct_of_deliveries"] == SUPPRESSED
    assert pd.isna(out.loc[RESIDUAL, "ref_cs_rate"])


def test_markdown_carries_caveats_and_reference_status(classified: pd.DataFrame) -> None:
    table = published_audit_table(classified)
    text = audit_markdown(table, None)
    assert ONSET_CAVEAT in text
    assert "not available" in text and "never invented" in text
