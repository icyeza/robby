import pytest

from robson_engine.models import COARSE_PRESENTATIONS, INPUT_FIELDS, PRESENTATIONS, RobsonInputs


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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"parity": True},
        {"previous_cs_count": False},
        {"plurality": True},
        {"gestational_age_weeks": True},
    ],
)
def test_boolean_counts_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RobsonInputs(**kwargs)  # type: ignore[arg-type]


# Coarse inputs.


def test_coarse_presentations_are_keys_over_precise_values() -> None:
    assert COARSE_PRESENTATIONS["non_cephalic"] == frozenset({"breech", "transverse", "oblique"})
    assert not set(COARSE_PRESENTATIONS) & PRESENTATIONS
    for allowed in COARSE_PRESENTATIONS.values():
        assert allowed <= PRESENTATIONS
        assert len(allowed) >= 2


def test_coarse_presentation_is_recorded_but_coarse() -> None:
    inputs = RobsonInputs(fetal_presentation="non_cephalic")
    assert inputs.value("fetal_presentation") == "non_cephalic"
    assert inputs.coarse_value("fetal_presentation") == frozenset(
        {"breech", "transverse", "oblique"}
    )
    assert "fetal_presentation" not in inputs.missing_fields()
    assert inputs.coarse_fields() == ["fetal_presentation"]


def test_precise_values_have_no_coarse_value() -> None:
    inputs = RobsonInputs(fetal_presentation="breech", gestational_age_weeks=39.0)
    assert inputs.coarse_value("fetal_presentation") is None
    assert inputs.coarse_value("gestational_age_weeks") is None
    assert inputs.coarse_value("parity") is None
    assert inputs.coarse_fields() == []


def test_coarse_value_unknown_field_raises() -> None:
    with pytest.raises(KeyError):
        RobsonInputs().coarse_value("birth_weight")


def test_ga_range_used_when_exact_ga_missing() -> None:
    inputs = RobsonInputs(gestational_age_range=(35.0, 37.857))
    assert inputs.value("gestational_age_weeks") is None
    assert inputs.coarse_value("gestational_age_weeks") == (35.0, 37.857)
    assert "gestational_age_weeks" not in inputs.missing_fields()
    assert inputs.coarse_fields() == ["gestational_age_weeks"]


def test_exact_ga_wins_over_range() -> None:
    inputs = RobsonInputs(gestational_age_weeks=39.0, gestational_age_range=(20.0, 33.857))
    assert inputs.value("gestational_age_weeks") == 39.0
    assert inputs.coarse_value("gestational_age_weeks") is None
    assert inputs.coarse_fields() == []


def test_ga_range_is_normalised_to_a_float_tuple() -> None:
    inputs = RobsonInputs(gestational_age_range=[35, 38])  # type: ignore[arg-type]
    assert inputs.gestational_age_range == (35.0, 38.0)
    assert isinstance(inputs.gestational_age_range, tuple)
    hash(inputs)


@pytest.mark.parametrize(
    "rng",
    [
        (38.0, 36.0),
        (19.5, 30.0),
        (30.0, 45.5),
        (float("nan"), 38.0),
        (30.0, float("inf")),
        (True, 38.0),
        (30.0, False),
        (35.0,),
        (35.0, 36.0, 37.0),
        ("35", "37"),
        (None, 38.0),
        "35-37",
        35.0,
    ],
)
def test_invalid_ga_range_raises_without_echoing_value(rng: object) -> None:
    with pytest.raises(ValueError) as excinfo:
        RobsonInputs(gestational_age_range=rng)  # type: ignore[arg-type]
    message = str(excinfo.value)
    assert "gestational_age_range" in message
    parts = rng if isinstance(rng, tuple) else (rng,)
    for part in parts:
        assert str(part) not in message


@pytest.mark.parametrize("presentation", ["non_breech", "NON_CEPHALIC", "noncephalic"])
def test_unknown_coarse_presentation_raises_without_echoing_value(presentation: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        RobsonInputs(fetal_presentation=presentation)
    assert presentation not in str(excinfo.value)
