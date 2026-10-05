import copy
from pathlib import Path

import pytest
import yaml

from robson_engine.models import RobsonInputs
from robson_engine.ruleset import (
    Condition,
    RuleSetError,
    compute_checksum,
    load_rule_set,
    parse_rule_set,
    stamp,
)

PACKAGED = Path(__file__).parents[1] / "src" / "robson_engine" / "rules" / "robson_v1.0.yaml"


def _data() -> dict:
    return yaml.safe_load(PACKAGED.read_text(encoding="utf-8"))


def test_packaged_rule_set_loads_and_checksum_matches() -> None:
    rules = load_rule_set()
    assert rules.version_label == "robson-v1.0"
    assert [g.group for g in rules.groups] == list(range(1, 11))
    assert rules.checksum == compute_checksum(_data())


def test_changed_content_without_restamp_is_rejected() -> None:
    data = _data()
    data["groups"][0]["conditions"][3]["value"] = 36.0
    with pytest.raises(RuleSetError, match="checksum"):
        parse_rule_set(data)


def test_missing_group_is_rejected() -> None:
    data = _data()
    data["groups"] = data["groups"][:-1]
    data["checksum"] = compute_checksum(data)
    with pytest.raises(RuleSetError, match="groups 1-10"):
        parse_rule_set(data)


@pytest.mark.parametrize(
    "condition",
    [
        {"field": "birth_weight", "op": "eq", "value": 1},
        {"field": "parity", "op": "gt", "value": 1},
        {"field": "onset_of_labour", "op": "in", "value": "induced"},
    ],
)
def test_malformed_condition_is_rejected(condition: dict) -> None:
    data = copy.deepcopy(_data())
    data["groups"][7]["conditions"] = [condition]
    data["checksum"] = compute_checksum(data)
    with pytest.raises(RuleSetError):
        parse_rule_set(data)


@pytest.mark.parametrize(
    "condition",
    [
        {"field": "fetal_presentation", "op": "eq", "value": "cephalc"},
        {"field": "onset_of_labour", "op": "in", "value": ["induced", "labour"]},
        {"field": "gestational_age_weeks", "op": "ge", "value": "37"},
        {"field": "gestational_age_weeks", "op": "ge", "value": True},
        {"field": "parity", "op": "eq", "value": "0"},
        {"field": "fetal_presentation", "op": "eq", "value": "non_cephalic"},
        {"field": "fetal_presentation", "op": "in", "value": ["non_cephalic", "breech"]},
    ],
)
def test_invalid_domain_value_is_rejected(condition: dict) -> None:
    data = copy.deepcopy(_data())
    data["groups"][7]["conditions"] = [condition]
    data["checksum"] = compute_checksum(data)
    with pytest.raises(RuleSetError):
        parse_rule_set(data)


def test_missing_groups_key_is_rejected() -> None:
    data = copy.deepcopy(_data())
    del data["groups"]
    data["checksum"] = compute_checksum(data)
    with pytest.raises(RuleSetError, match="malformed"):
        parse_rule_set(data)


def test_group_missing_conditions_is_rejected() -> None:
    data = copy.deepcopy(_data())
    del data["groups"][0]["conditions"]
    data["checksum"] = compute_checksum(data)
    with pytest.raises(RuleSetError, match="malformed"):
        parse_rule_set(data)


def test_group_with_empty_conditions_is_rejected() -> None:
    data = copy.deepcopy(_data())
    data["groups"][0]["conditions"] = []
    data["checksum"] = compute_checksum(data)
    with pytest.raises(RuleSetError, match="group 1 has no conditions"):
        parse_rule_set(data)


def test_consistency_rule_with_empty_when_is_rejected() -> None:
    data = copy.deepcopy(_data())
    data["consistency"][0]["when"] = []
    data["checksum"] = compute_checksum(data)
    with pytest.raises(
        RuleSetError, match="consistency rule 'nulliparous_with_previous_cs' has no conditions"
    ):
        parse_rule_set(data)


def test_stamp_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "rules.yaml"
    text = PACKAGED.read_text(encoding="utf-8").replace("value: 37.0", "value: 37.0 ", 1)
    target.write_text(text.replace(_data()["checksum"], "UNSTAMPED"), encoding="utf-8")
    checksum = stamp(target)
    assert load_rule_set(target).checksum == checksum


# Coarse inputs: true if the condition holds for every allowed value, false if
# for none, unknown otherwise.

NON_CEPHALIC = RobsonInputs(fetal_presentation="non_cephalic")


@pytest.mark.parametrize(
    ("condition", "outcome"),
    [
        (Condition("fetal_presentation", "eq", "cephalic"), "false"),
        (Condition("fetal_presentation", "eq", "breech"), "unknown"),
        (Condition("fetal_presentation", "in", ("transverse", "oblique")), "unknown"),
        (Condition("fetal_presentation", "in", ("breech", "transverse", "oblique")), "true"),
        (
            Condition("fetal_presentation", "in", ("cephalic", "breech", "transverse", "oblique")),
            "true",
        ),
        (Condition("fetal_presentation", "in", ("cephalic",)), "false"),
    ],
)
def test_condition_on_coarse_presentation(condition: Condition, outcome: str) -> None:
    assert condition.evaluate(NON_CEPHALIC) == outcome


@pytest.mark.parametrize(
    ("rng", "op", "value", "outcome"),
    [
        ((38.0, 40.857), "ge", 37.0, "true"),
        ((38.0, 40.857), "lt", 37.0, "false"),
        ((35.0, 37.857), "ge", 37.0, "unknown"),
        ((35.0, 37.857), "lt", 37.0, "unknown"),
        ((20.0, 33.857), "ge", 37.0, "false"),
        ((20.0, 33.857), "lt", 37.0, "true"),
        ((37.0, 40.0), "ge", 37.0, "true"),
        ((37.0, 40.0), "lt", 37.0, "false"),
        ((30.0, 37.0), "ge", 37.0, "unknown"),
        ((30.0, 37.0), "lt", 37.0, "unknown"),
        ((36.0, 36.99), "ge", 37.0, "false"),
        ((36.0, 36.99), "lt", 37.0, "true"),
        ((38.0, 38.0), "eq", 38.0, "true"),
        ((38.0, 38.0), "eq", 39.0, "false"),
        ((35.0, 40.0), "eq", 38.0, "unknown"),
        ((20.0, 30.0), "eq", 38.0, "false"),
        ((35.0, 40.0), "in", (38.0, 41.0), "unknown"),
        ((20.0, 30.0), "in", (38.0, 41.0), "false"),
        ((38.0, 38.0), "in", (38.0, 41.0), "true"),
    ],
)
def test_condition_on_ga_range(
    rng: tuple[float, float], op: str, value: object, outcome: str
) -> None:
    inputs = RobsonInputs(gestational_age_range=rng)
    assert Condition("gestational_age_weeks", op, value).evaluate(inputs) == outcome  # type: ignore[arg-type]


def test_exact_ga_is_used_even_when_range_given() -> None:
    inputs = RobsonInputs(gestational_age_weeks=39.0, gestational_age_range=(20.0, 33.857))
    assert Condition("gestational_age_weeks", "ge", 37.0).evaluate(inputs) == "true"
    assert Condition("gestational_age_weeks", "lt", 37.0).evaluate(inputs) == "false"


def test_ordering_op_on_coarse_presentation_is_rejected() -> None:
    with pytest.raises(RuleSetError):
        Condition("fetal_presentation", "ge", 1).evaluate(NON_CEPHALIC)


def test_ordering_op_with_non_numeric_value_on_ga_range_is_rejected() -> None:
    inputs = RobsonInputs(gestational_age_range=(35.0, 38.0))
    with pytest.raises(RuleSetError):
        Condition("gestational_age_weeks", "ge", "37").evaluate(inputs)
