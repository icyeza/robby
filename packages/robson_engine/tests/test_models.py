import pytest

from robson_engine.models import INPUT_FIELDS, RobsonInputs


def test_all_missing_by_default() -> None:
    inputs = RobsonInputs()
    assert inputs.missing_fields() == list(INPUT_FIELDS)


def test_value_lookup() -> None:
    inputs = RobsonInputs(parity=2, fetal_presentation="breech")
    assert inputs.value("parity") == 2
    assert inputs.value("fetal_presentation") == "breech"
    assert inputs.value("plurality") is None


def test_unknown_field_raises() -> None:
    with pytest.raises(KeyError):
        RobsonInputs().value("birth_weight")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"parity": -1},
        {"previous_cs_count": -1},
        {"plurality": 0},
        {"fetal_presentation": "face"},
        {"onset_of_labour": "caesarean"},
        {"gestational_age_weeks": 19.9},
        {"gestational_age_weeks": 45.1},
        {"gestational_age_weeks": float("nan")},
    ],
)
def test_invalid_values_raise_without_echoing_value(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError) as excinfo:
        RobsonInputs(**kwargs)  # type: ignore[arg-type]
    value = next(iter(kwargs.values()))
    assert str(value) not in str(excinfo.value)
