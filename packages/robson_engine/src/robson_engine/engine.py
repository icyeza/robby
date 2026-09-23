"""Constraint-resolution classifier: evaluates every group against the available inputs."""

from __future__ import annotations

from collections.abc import Iterable

from robson_engine.models import (
    INPUT_FIELDS,
    ClassificationResult,
    ConditionTrace,
    Outcome,
    RobsonInputs,
)
from robson_engine.ruleset import RuleSet


def _ordered(fields: Iterable[str]) -> tuple[str, ...]:
    wanted = set(fields)
    return tuple(f for f in INPUT_FIELDS if f in wanted)


def _subgroup(
    rule_set: RuleSet, group: int, inputs: RobsonInputs, trace: list[ConditionTrace]
) -> str | None:
    rule = next(r for r in rule_set.groups if r.group == group)
    for sub in rule.subgroups:
        outcomes = []
        for cond in sub.conditions:
            outcome = cond.evaluate(inputs)
            trace.append(ConditionTrace(group, sub.label, cond.field, cond.op, cond.value, outcome))
            outcomes.append(outcome)
        if all(o == "true" for o in outcomes):
            return sub.label
    return None


def classify(inputs: RobsonInputs, rule_set: RuleSet) -> ClassificationResult:
    """Classify one admission (spec §6.2).

    A condition over a missing input is ``unknown``; a group is admissible when none of its
    conditions is ``false``.

    * ``resolved``: exactly one admissible group, all of its conditions ``true``.
    * ``partial``: otherwise, with at least one admissible group. ``resolving_fields`` lists
      the missing inputs referenced by an ``unknown`` condition of an admissible group.
    * ``conflict``: a consistency rule fires, or no group is admissible.
    """
    trace: list[ConditionTrace] = []
    outcomes: dict[int, list[Outcome]] = {}
    for rule in rule_set.groups:
        results: list[Outcome] = []
        for cond in rule.conditions:
            outcome = cond.evaluate(inputs)
            trace.append(ConditionTrace(rule.group, None, cond.field, cond.op, cond.value, outcome))
            results.append(outcome)
        outcomes[rule.group] = results
    admissible = frozenset(g for g, res in outcomes.items() if "false" not in res)

    conflict: list[str] = []
    for check in rule_set.consistency:
        if all(c.evaluate(inputs) == "true" for c in check.when):
            conflict.extend(c.field for c in check.when)
    if not admissible and not conflict:
        conflict = [f for f in INPUT_FIELDS if inputs.value(f) is not None]
    version = rule_set.version_label

    if conflict:
        return ClassificationResult(
            "conflict", None, None, admissible, (), _ordered(conflict), tuple(trace), version
        )
    if len(admissible) == 1:
        (group,) = admissible
        if all(o == "true" for o in outcomes[group]):
            subgroup = _subgroup(rule_set, group, inputs, trace)
            return ClassificationResult(
                "resolved", group, subgroup, admissible, (), (), tuple(trace), version
            )
    resolving = _ordered(t.field for t in trace if t.outcome == "unknown" and t.group in admissible)
    return ClassificationResult(
        "partial", None, None, admissible, resolving, (), tuple(trace), version
    )
