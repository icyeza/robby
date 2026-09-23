import dataclasses
import itertools
from collections.abc import Iterator

import pytest

from robson_engine import COARSE_PRESENTATIONS, INPUT_FIELDS, RobsonInputs, classify, load_rule_set
from robson_engine.ruleset import Condition, GroupRule, RuleSet

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
    assert result.candidates == frozenset({1})


def test_zero_admissible_groups_is_conflict() -> None:
    groups = tuple(GroupRule(g, "g", (Condition("plurality", "ge", 5),), ()) for g in range(1, 11))
    rule_set = RuleSet("test", "x", groups=groups, consistency=())
    result = classify(RobsonInputs(plurality=1), rule_set)
    assert result.status == "conflict"
    assert result.group is None
    assert result.candidates == frozenset()
    assert result.conflict_fields == ("plurality",)


def test_partial_ga_missing_group_5_or_10() -> None:
    result = classify(full(parity=1, previous_cs_count=1, gestational_age_weeks=None), RULES)
    assert result.status == "partial"
    assert result.candidates == frozenset({5, 10})
    assert result.resolving_fields == ("gestational_age_weeks",)


def test_partial_onset_missing_group_1_or_2() -> None:
    result = classify(full(onset_of_labour=None), RULES)
    assert result.status == "partial"
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


# Coarse inputs (spec v1.2 §6.2).


@pytest.mark.parametrize(
    ("overrides", "candidates"),
    [
        ({"parity": 0}, {6, 9}),
        ({"parity": 2}, {7, 9}),
        ({"parity": 1, "previous_cs_count": 1}, {7, 9}),
    ],
)
def test_non_cephalic_singleton_is_partial_breech_or_lie(
    overrides: dict[str, object], candidates: set[int]
) -> None:
    result = classify(full(fetal_presentation="non_cephalic", **overrides), RULES)
    assert result.status == "partial"
    assert result.group is None
    assert result.candidates == frozenset(candidates)
    assert result.resolving_fields == ("fetal_presentation",)


def test_non_cephalic_with_parity_missing_lists_both_fields() -> None:
    result = classify(full(fetal_presentation="non_cephalic", parity=None), RULES)
    assert result.status == "partial"
    assert result.candidates == frozenset({6, 7, 9})
    assert result.resolving_fields == ("parity", "fetal_presentation")


def test_non_cephalic_multiple_pregnancy_resolves_to_group_8() -> None:
    result = classify(full(fetal_presentation="non_cephalic", plurality=2), RULES)
    assert result.status == "resolved"
    assert result.group == 8
    assert result.candidates == frozenset({8})


def test_cephalic_is_unaffected_by_coarse_support() -> None:
    result = classify(full(fetal_presentation="cephalic"), RULES)
    assert result.status == "resolved"
    assert result.group == 1


def test_non_cephalic_trace_marks_breech_and_lie_conditions_unknown() -> None:
    result = classify(full(fetal_presentation="non_cephalic"), RULES)
    outcomes = {
        (t.group, str(t.expected)): t.outcome
        for t in result.trace
        if t.field == "fetal_presentation"
    }
    assert outcomes[(6, "breech")] == "unknown"
    assert outcomes[(9, "('transverse', 'oblique')")] == "unknown"
    assert outcomes[(1, "cephalic")] == "false"


def test_ga_range_at_term_resolves_to_group_1() -> None:
    inputs = full(gestational_age_weeks=None, gestational_age_range=(38.0, 40.857))
    result = classify(inputs, RULES)
    assert result.status == "resolved"
    assert result.group == 1


def test_ga_range_straddling_37_is_partial_group_1_or_10() -> None:
    inputs = full(gestational_age_weeks=None, gestational_age_range=(35.0, 37.857))
    result = classify(inputs, RULES)
    assert result.status == "partial"
    assert result.candidates == frozenset({1, 10})
    assert result.resolving_fields == ("gestational_age_weeks",)


def test_ga_range_preterm_resolves_to_group_10() -> None:
    inputs = full(gestational_age_weeks=None, gestational_age_range=(20.0, 33.857))
    result = classify(inputs, RULES)
    assert result.status == "resolved"
    assert result.group == 10


def test_exact_ga_wins_over_contradictory_range() -> None:
    inputs = full(gestational_age_weeks=39.0, gestational_age_range=(20.0, 33.857))
    result = classify(inputs, RULES)
    assert result.status == "resolved"
    assert result.group == 1


def test_ga_range_and_non_cephalic_together() -> None:
    inputs = full(
        fetal_presentation="non_cephalic",
        gestational_age_weeks=None,
        gestational_age_range=(35.0, 37.857),
    )
    result = classify(inputs, RULES)
    assert result.status == "partial"
    assert result.candidates == frozenset({6, 9})
    assert result.resolving_fields == ("fetal_presentation",)


# Exhaustive grid over all input combinations, including missing values.
PARITY = [None, 0, 1, 3]
PREVIOUS_CS = [None, 0, 1, 2]
PLURALITY = [None, 1, 2]
PRESENTATION = [None, "cephalic", "breech", "transverse", "oblique", "non_cephalic"]
GA = [None, 30.0, 36.0, 36.0 + 6 / 7, 37.0, 41.0]
# GA bands, applied only when the exact GA is missing (an exact GA always wins).
GA_RANGES: list[tuple[float, float] | None] = [
    None,
    (38.0, 40.0 + 6 / 7),
    (35.0, 37.0 + 6 / 7),
    (20.0, 33.0 + 6 / 7),
]
ONSET = [None, "spontaneous", "induced", "prelabour_cs"]
GRID = [
    RobsonInputs(*values, gestational_age_range=rng)  # type: ignore[arg-type]
    for values in itertools.product(PARITY, PREVIOUS_CS, PLURALITY, PRESENTATION, GA, ONSET)
    for rng in (GA_RANGES if values[4] is None else [None])
]


def _contradictory(i: RobsonInputs) -> bool:
    return i.parity == 0 and i.previous_cs_count is not None and i.previous_cs_count >= 1


def test_grid_never_assigns_more_than_one_group() -> None:
    for inputs in GRID:
        result = classify(inputs, RULES)
        if result.status == "resolved":
            assert result.candidates == frozenset({result.group})


def _precise_complete(inputs: RobsonInputs) -> bool:
    return not inputs.missing_fields() and not inputs.coarse_fields()


def test_grid_complete_consistent_inputs_always_resolve() -> None:
    for inputs in GRID:
        if _precise_complete(inputs) and not _contradictory(inputs):
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
            imprecise = set(inputs.missing_fields()) | set(inputs.coarse_fields())
            assert set(result.resolving_fields) <= imprecise


def test_grid_blanking_inputs_keeps_true_group_among_candidates() -> None:
    complete = [g for g in GRID if _precise_complete(g) and not _contradictory(g)]
    for inputs in complete:
        truth = classify(inputs, RULES).group
        for mask in range(1, 2 ** len(INPUT_FIELDS)):
            blanked = {
                f: (None if mask >> k & 1 else inputs.value(f)) for k, f in enumerate(INPUT_FIELDS)
            }
            result = classify(RobsonInputs(**blanked), RULES)  # type: ignore[arg-type]
            assert truth in result.candidates


def _refinements(inputs: RobsonInputs) -> Iterator[RobsonInputs]:
    """Precise records a coarse record allows (GA: the band ends, plus 37.0 if inside)."""
    presentation = inputs.fetal_presentation
    if presentation in COARSE_PRESENTATIONS:
        presentations: list[str | None] = sorted(COARSE_PRESENTATIONS[presentation])
    else:
        presentations = [presentation]
    band = inputs.coarse_value("gestational_age_weeks")
    if isinstance(band, tuple):
        lo, hi = band
        gas: list[float | None] = sorted({lo, hi} | ({37.0} if lo <= 37.0 <= hi else set()))
    else:
        gas = [inputs.gestational_age_weeks]
    for p, ga in itertools.product(presentations, gas):
        yield dataclasses.replace(
            inputs, fetal_presentation=p, gestational_age_weeks=ga, gestational_age_range=None
        )


def test_grid_coarse_records_are_sound_and_tight() -> None:
    """Every precise refinement of a coarse record lands inside its candidates (soundness),
    and together the refinements reach every candidate (no spurious group)."""
    coarse = [g for g in GRID if g.coarse_fields()]
    assert len(coarse) > 1000
    for inputs in coarse:
        result = classify(inputs, RULES)
        reached: set[int] = set()
        for refined in _refinements(inputs):
            assert not refined.coarse_fields()
            refined_result = classify(refined, RULES)
            assert refined_result.candidates <= result.candidates
            reached |= refined_result.candidates
            if result.status == "resolved":
                assert refined_result.status == "resolved"
                assert refined_result.group == result.group
                assert refined_result.subgroup == result.subgroup
            assert (refined_result.status == "conflict") == (result.status == "conflict")
        assert reached == set(result.candidates)
