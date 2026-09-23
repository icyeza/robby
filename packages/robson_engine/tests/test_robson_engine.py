import itertools

import pytest

from robson_engine import INPUT_FIELDS, RobsonInputs, classify, load_rule_set

RULES = load_rule_set()


def full(**overrides: object) -> RobsonInputs:
    base: dict[str, object] = {
        "parity": 0,
        "previous_cs_count": 0,
        "plurality": 1,
        "fetal_presentation": "cephalic",
        "gestational_age_weeks": 39.0,
        "onset_of_labour": "spontaneous",
    }
    base.update(overrides)
    return RobsonInputs(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("inputs", "group", "subgroup"),
    [
        (full(), 1, None),
        (full(onset_of_labour="induced"), 2, "2a"),
        (full(onset_of_labour="prelabour_cs"), 2, "2b"),
        (full(parity=2), 3, None),
        (full(parity=2, onset_of_labour="induced"), 4, "4a"),
        (full(parity=2, onset_of_labour="prelabour_cs"), 4, "4b"),
        (full(parity=1, previous_cs_count=1), 5, "5a"),
        (full(parity=3, previous_cs_count=2, onset_of_labour="prelabour_cs"), 5, "5b"),
        (full(fetal_presentation="breech"), 6, None),
        (full(parity=1, fetal_presentation="breech"), 7, None),
        (full(parity=1, previous_cs_count=1, fetal_presentation="breech"), 7, None),
        (full(plurality=2), 8, None),
        (full(plurality=3, parity=1, previous_cs_count=1, fetal_presentation="breech"), 8, None),
        (full(fetal_presentation="transverse"), 9, None),
        (full(fetal_presentation="oblique", parity=1, previous_cs_count=1), 9, None),
        (full(gestational_age_weeks=34.0), 10, None),
        (full(gestational_age_weeks=34.0, parity=1, previous_cs_count=1), 10, None),
    ],
)
def test_each_group_definition(inputs: RobsonInputs, group: int, subgroup: str | None) -> None:
    result = classify(inputs, RULES)
    assert result.status == "resolved"
    assert result.group == group
    assert result.subgroup == subgroup
    assert result.candidates == frozenset({group})
    assert result.rule_set_version == "robson-v1.0"


@pytest.mark.parametrize(
    ("ga", "group"),
    [(36.0, 10), (36.0 + 6 / 7, 10), (36.99, 10), (37.0, 1), (37.0 + 1 / 7, 1)],
)
def test_ga_boundary_at_37(ga: float, group: int) -> None:
    assert classify(full(gestational_age_weeks=ga), RULES).group == group


def test_conflict_nulliparous_with_previous_cs() -> None:
    result = classify(full(parity=0, previous_cs_count=1), RULES)
    assert result.status == "conflict"
    assert result.group is None
    assert result.conflict_fields == ("parity", "previous_cs_count")


def test_partial_ga_missing_group_5_or_10() -> None:
    result = classify(full(parity=1, previous_cs_count=1, gestational_age_weeks=None), RULES)
    assert result.status == "partial"
    assert result.candidates == frozenset({5, 10})
    assert result.resolving_fields == ("gestational_age_weeks",)


def test_partial_onset_missing_group_1_or_2() -> None:
    result = classify(full(onset_of_labour=None), RULES)
    assert result.candidates == frozenset({1, 2})
    assert result.resolving_fields == ("onset_of_labour",)


def test_all_missing_is_partial_over_all_groups() -> None:
    result = classify(RobsonInputs(), RULES)
    assert result.status == "partial"
    assert result.candidates == frozenset(range(1, 11))
    assert result.resolving_fields == INPUT_FIELDS


def test_trace_records_every_group_condition() -> None:
    result = classify(full(), RULES)
    n_conditions = sum(len(g.conditions) for g in RULES.groups)
    group_traces = [t for t in result.trace if t.subgroup is None]
    assert len(group_traces) == n_conditions
    assert all(t.outcome == "true" for t in group_traces if t.group == 1)


# Exhaustive grid over all input combinations, including missing values.
PARITY = [None, 0, 1, 3]
PREVIOUS_CS = [None, 0, 1, 2]
PLURALITY = [None, 1, 2]
PRESENTATION = [None, "cephalic", "breech", "transverse", "oblique"]
GA = [None, 30.0, 36.0, 36.0 + 6 / 7, 37.0, 41.0]
ONSET = [None, "spontaneous", "induced", "prelabour_cs"]
GRID = [
    RobsonInputs(*values)
    for values in itertools.product(PARITY, PREVIOUS_CS, PLURALITY, PRESENTATION, GA, ONSET)
]


def _contradictory(i: RobsonInputs) -> bool:
    return i.parity == 0 and i.previous_cs_count is not None and i.previous_cs_count >= 1


def test_grid_never_assigns_more_than_one_group() -> None:
    for inputs in GRID:
        result = classify(inputs, RULES)
        if result.status == "resolved":
            assert result.candidates == frozenset({result.group})


def test_grid_complete_consistent_inputs_always_resolve() -> None:
    for inputs in GRID:
        if not inputs.missing_fields() and not _contradictory(inputs):
            assert classify(inputs, RULES).status == "resolved"


def test_grid_conflict_exactly_when_contradictory() -> None:
    for inputs in GRID:
        assert (classify(inputs, RULES).status == "conflict") == _contradictory(inputs)


def test_grid_partial_has_several_candidates_and_missing_resolving_fields() -> None:
    for inputs in GRID:
        result = classify(inputs, RULES)
        if result.status == "partial":
            assert len(result.candidates) >= 2
            assert result.resolving_fields
            assert set(result.resolving_fields) <= set(inputs.missing_fields())


def test_grid_blanking_inputs_keeps_true_group_among_candidates() -> None:
    complete = [g for g in GRID if not g.missing_fields() and not _contradictory(g)]
    for inputs in complete:
        truth = classify(inputs, RULES).group
        for mask in range(1, 2 ** len(INPUT_FIELDS)):
            blanked = {
                f: (None if mask >> k & 1 else inputs.value(f)) for k, f in enumerate(INPUT_FIELDS)
            }
            result = classify(RobsonInputs(**blanked), RULES)  # type: ignore[arg-type]
            assert truth in result.candidates
