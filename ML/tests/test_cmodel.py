"""WHO C-Model: no fallback, not applicable when a variable is absent,
exact arithmetic on toy data. Coefficients here are FAKE fixtures."""

import math
from pathlib import Path

import pandas as pd
import pytest
import yaml

from robson_ml.cmodel import (
    CModelNotApplicableError,
    cmodel_applicability,
    cmodel_probability,
    facility_expected,
)
from robson_ml.references import (
    CMODEL_PATH,
    ReferenceDataMissing,
    ReferenceSchemaError,
    load_cmodel,
)
from tests.synthetic_references import fake_cmodel


def test_cmodel_no_fallback(tmp_path: Path) -> None:
    """A missing coefficient file raises; nothing is estimated in its place."""
    path = tmp_path / "data" / "reference" / "cmodel_v1.yaml"
    with pytest.raises(ReferenceDataMissing) as error:
        load_cmodel(path)
    assert str(path) in str(error.value)
    assert "coefficient" in str(error.value)
    assert CMODEL_PATH.as_posix() == "data/reference/cmodel_v1.yaml"


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "facility_id": ["A", "A", "B", "B"],
            "parity": pd.array([0, 2, 1, None], dtype="Int64"),
            "previous_cs_count": pd.array([0, 1, 0, 0], dtype="Int64"),
            "fetal_presentation": ["cephalic", "breech", "non_cephalic", "cephalic"],
            "maternal_age": [28.0, 38.0, 18.0, 30.0],
            "cs": pd.array([0, 1, 1, 0], dtype="Int64"),
        }
    )


def _logistic(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def test_probability_arithmetic(tmp_path: Path) -> None:
    ref = load_cmodel(fake_cmodel(tmp_path / "c.yaml"))
    p = cmodel_probability(_frame(), ref)
    # intercept -1; nulliparous +0.1; previous CS +1; malpresentation +2; 0.01 x (age - 28)
    assert p[0] == pytest.approx(_logistic(-1 + 0.1))
    assert p[1] == pytest.approx(_logistic(-1 + 1.0 + 2.0 + 0.10))
    assert p[2] == pytest.approx(_logistic(-1 + 2.0 - 0.10))
    assert math.isnan(p[3])  # parity missing: not scored


def test_facility_expected_is_mean_probability(tmp_path: Path) -> None:
    ref = load_cmodel(fake_cmodel(tmp_path / "c.yaml"))
    frame = _frame()
    p = cmodel_probability(frame, ref)
    table = facility_expected(frame, ref).set_index("facility")
    assert table.loc["A", "cmodel_expected_rate"] == pytest.approx((p[0] + p[1]) / 2)
    assert table.loc["B", "n_scored"] == 1
    assert table.loc["ALL", "n"] == 4


def test_absent_variable_makes_it_not_applicable(tmp_path: Path) -> None:
    ref = load_cmodel(fake_cmodel(tmp_path / "c.yaml", absent_variable=True))
    applicability = cmodel_applicability(_frame(), ref)
    assert not applicability.applicable
    assert applicability.absent_variables == ["fake_unrecorded"]
    assert "not applicable" in applicability.statement()
    with pytest.raises(CModelNotApplicableError):
        cmodel_probability(_frame(), ref)


def test_mapped_column_missing_from_frame_is_not_applicable(tmp_path: Path) -> None:
    ref = load_cmodel(fake_cmodel(tmp_path / "c.yaml"))
    assert not cmodel_applicability(_frame().drop(columns="maternal_age"), ref).applicable


def test_schema_rejects_term_without_source(tmp_path: Path) -> None:
    path = fake_cmodel(tmp_path / "c.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["terms"][0].pop("source")
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ReferenceSchemaError, match="source"):
        load_cmodel(path)
