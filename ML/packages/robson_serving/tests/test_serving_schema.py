import json
from dataclasses import replace
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from robson_serving.features import CATEGORICAL, NUMERIC
from robson_serving.schema import (
    CATEGORY,
    INTEGER,
    NUMBER,
    UNSEEN_LEVEL,
    FeatureColumn,
    FeatureSchema,
    InputField,
    SchemaError,
    categorical_levels,
    column_kinds,
    normalise_admission,
)

SCHEMA = FeatureSchema(
    columns=(
        FeatureColumn("parity", NUMERIC),
        FeatureColumn("fetal_presentation", CATEGORICAL, ("cephalic", "non_cephalic")),
        FeatureColumn("facility_id", CATEGORICAL, ("FAC_A", "FAC_B")),
    ),
    inputs=(
        InputField("parity", INTEGER, (0.0, None)),
        InputField("maternal_age", NUMBER, (12.0, 55.0)),
        InputField("fetal_presentation", CATEGORY, None, ("cephalic", "non_cephalic")),
        InputField("facility_id", CATEGORY, None, ("FAC_A", "FAC_B")),
        InputField("ga_band_lower", NUMBER, (20.0, 45.0)),
        InputField("ga_band_upper", NUMBER, (20.0, 45.0)),
    ),
    coarsen_presentation=True,
    unseen_allowed=("facility_id",),
)


def test_a_valid_admission_is_normalised() -> None:
    admission = {
        "parity": 2,
        "maternal_age": Decimal("30.5"),
        "fetal_presentation": "cephalic",
        "facility_id": "FAC_B",
        "ga_band_lower": 35.0,
        "ga_band_upper": 37.857142857142854,
        "unrelated": "ignored",
    }
    assert normalise_admission(admission, SCHEMA) == {
        "parity": 2.0,
        "maternal_age": 30.5,
        "fetal_presentation": "cephalic",
        "facility_id": "FAC_B",
        "ga_band_lower": 35.0,
        "ga_band_upper": 37.857142857142854,
    }


def test_missing_values_become_none_and_a_missing_facility_is_unseen() -> None:
    record = normalise_admission({"maternal_age": float("nan"), "facility_id": None}, SCHEMA)
    assert record["parity"] is None and record["maternal_age"] is None
    assert record["facility_id"] == UNSEEN_LEVEL


def test_a_precise_presentation_is_coarsened_when_the_model_only_knows_non_cephalic() -> None:
    coarse = normalise_admission({"fetal_presentation": "breech"}, SCHEMA)
    assert coarse["fetal_presentation"] == "non_cephalic"
    strict = replace(SCHEMA, coarsen_presentation=False)
    with pytest.raises(SchemaError) as info:
        normalise_admission({"fetal_presentation": "breech"}, strict)
    assert info.value.field == "fetal_presentation"


@pytest.mark.parametrize(
    ("field", "value", "problem"),
    [
        ("parity", 1.5, "whole number"),
        ("parity", True, "expected a number"),
        ("parity", "2", "expected a number"),
        ("parity", -1, "allowed range"),
        ("maternal_age", 99, "allowed range"),
        ("maternal_age", float("inf"), "finite"),
        ("fetal_presentation", 3, "expected text"),
        ("fetal_presentation", "sideways", "not a level"),
        ("facility_id", "FAC_Z", "not a level"),
    ],
)
def test_violations_name_the_field_never_the_value(
    field: str, value: object, problem: str
) -> None:
    with pytest.raises(SchemaError) as info:
        normalise_admission({field: value}, SCHEMA)
    assert info.value.field == field
    assert problem in str(info.value)
    assert str(value) not in str(info.value)


def test_a_band_needs_both_bounds_in_order() -> None:
    with pytest.raises(SchemaError, match="both bounds"):
        normalise_admission({"ga_band_lower": 35.0}, SCHEMA)
    with pytest.raises(SchemaError, match="above"):
        normalise_admission({"ga_band_lower": 40.0, "ga_band_upper": 35.0}, SCHEMA)


def test_the_schema_round_trips_through_json() -> None:
    data = json.loads(json.dumps(SCHEMA.to_dict()))
    assert FeatureSchema.from_dict(data) == SCHEMA
    with pytest.raises(ValueError, match="format"):
        FeatureSchema.from_dict({**data, "format": 99})
    bad = {**data, "columns": [{"name": "x", "kind": "text", "levels": None}]}
    with pytest.raises(ValueError, match="unknown column kind"):
        FeatureSchema.from_dict(bad)


def test_kinds_and_levels_come_from_the_fitted_preprocessor() -> None:
    x = pd.DataFrame(
        {
            "age": [20.0, 30.0, np.nan, 40.0],
            "site": pd.Series(["b", "a", "b", np.nan], dtype=object),
        }
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    prep = ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), ["age"]), ("cat", categorical, ["site"])]
    )
    pipe = Pipeline([("prep", prep), ("clf", LogisticRegression())]).fit(x, [0, 1, 0, 1])
    assert column_kinds(pipe) == {"age": NUMERIC, "site": CATEGORICAL}
    assert categorical_levels(pipe) == {"site": ("a", "b")}
    with pytest.raises(ValueError, match="prep"):
        column_kinds(LogisticRegression())
