"""The package installs and its path dependency on robson_engine resolves."""

import robby
from robson_engine import load_rule_set


def test_version() -> None:
    assert robby.__version__ == "0.1.0"


def test_robson_engine_path_dependency_loads_the_default_rule_set() -> None:
    assert load_rule_set().version_label == "robson-v1.0"
