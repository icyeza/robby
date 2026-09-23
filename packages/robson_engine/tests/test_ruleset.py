import copy
from pathlib import Path

import pytest
import yaml

from robson_engine.ruleset import (
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
