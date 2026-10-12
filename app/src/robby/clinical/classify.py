"""Robson classification against the active rule set, and the view of a classification
(the live panel and the result page share one builder)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from robson_engine import INPUT_FIELDS, ClassificationResult, RobsonInputs, classify
from robson_engine.ruleset import RuleSet as EngineRuleSet
from robson_engine.ruleset import parse_rule_set
from sqlalchemy import select
from sqlalchemy.orm import Session

from robby.clinical import labels
from robby.models import RuleSet

_PARSED: dict[Any, EngineRuleSet] = {}


class NoRuleSetError(Exception):
    """No rule set is active: nothing can be classified or saved."""


def active_rule_set(db: Session) -> tuple[RuleSet, EngineRuleSet]:
    row = db.scalar(select(RuleSet).where(RuleSet.is_active))
    if row is None:
        raise NoRuleSetError
    parsed = _PARSED.get(row.id)
    if parsed is None:
        parsed = parse_rule_set(row.definition)
        _PARSED[row.id] = parsed
    return row, parsed


def robson_inputs(
    parity: int | None,
    previous_cs_count: int | None,
    plurality: int | None,
    presentation: str | None,
    ga_weeks: int | None,
    onset: str | None,
) -> RobsonInputs:
    return RobsonInputs(
        parity=parity,
        previous_cs_count=previous_cs_count,
        plurality=plurality,
        fetal_presentation=presentation,
        gestational_age_weeks=float(ga_weeks) if ga_weeks is not None else None,
        onset_of_labour=onset,
    )


def stored_trace(result: ClassificationResult, inputs: RobsonInputs) -> dict[str, Any]:
    """The trace as kept in classification.trace (jsonb: no tuples, JSON-safe values)."""
    entries = [
        {
            "group": t.group,
            "subgroup": t.subgroup,
            "field": t.field,
            "op": t.op,
            "expected": list(t.expected) if isinstance(t.expected, tuple) else t.expected,
            "outcome": t.outcome,
        }
        for t in result.trace
    ]
    return {
        "entries": entries,
        "inputs": {name: inputs.value(name) for name in INPUT_FIELDS},
        "rule_set_version": result.rule_set_version,
    }


DEFAULT_FOOT = "Classification only. The readiness signal appears after saving."


@dataclass
class PanelView:
    """What the Robson panel / card shows."""

    state: str  # empty | resolved | partial | conflict
    group: int | None = None
    subgroup: str | None = None
    candidates: list[int] = field(default_factory=list)
    heading: str = ""
    rows: list[dict[str, str]] = field(default_factory=list)
    need_title: str = ""
    need_value: str = ""
    need_why: str = ""
    need_fields: list[str] = field(default_factory=list)
    footnote: str = ""
    foot: str = DEFAULT_FOOT
    rule_set_version: str = ""
    group_name: str = ""
    subgroup_name: str = ""


def _row(kind: str, text: str, value: str) -> dict[str, str]:
    return {"kind": kind, "label": text, "value": value}


def _resolved_rows(
    entries: list[dict[str, Any]], inputs: dict[str, Any], group: int, subgroup: str | None
) -> list[dict[str, str]]:
    rows = []
    for e in entries:
        if e["group"] != group or e["outcome"] != "true":
            continue
        if e["subgroup"] is not None and e["subgroup"] != subgroup:
            continue
        text = labels.condition_label(e["field"], e["op"], e["expected"])
        if e["subgroup"] is not None:
            text = f"Subgroup {e['subgroup']}: {text[0].lower()}{text[1:]}"
        rows.append(_row("ck", text, labels.input_value_text(e["field"], inputs.get(e["field"]))))
    return rows


def _open_rows(
    entries: list[dict[str, Any]], inputs: dict[str, Any], candidates: list[int]
) -> list[dict[str, str]]:
    rows = []
    for name in INPUT_FIELDS:
        mine = [
            e
            for e in entries
            if e["field"] == name and e["subgroup"] is None and e["group"] in candidates
        ]
        if not mine:
            continue
        texts = {labels.condition_label(e["field"], e["op"], e["expected"]) for e in mine}
        if all(e["outcome"] == "true" for e in mine) and len(texts) == 1:
            rows.append(_row("ck", texts.pop(), labels.input_value_text(name, inputs.get(name))))
        else:
            rows.append(_row("q", labels.FIELD_NAMES[name], labels.input_value_text(name, None)))
    return rows


def _why(entries: list[dict[str, Any]], name: str, candidates: list[int]) -> str:
    parts = []
    for group in candidates:
        for e in entries:
            if e["group"] == group and e["field"] == name and e["subgroup"] is None:
                text = labels.condition_label(e["field"], e["op"], e["expected"])
                parts.append(f"{text} → Group {group}.")
                break
    return " ".join(parts)


def panel_view(
    *,
    status: str,
    group: int | None,
    subgroup: str | None,
    candidates: list[int],
    resolving: list[str],
    conflict: list[str],
    trace: dict[str, Any] | None,
    consistency_name: str | None = None,
) -> PanelView:
    entries = list((trace or {}).get("entries", []))
    inputs = dict((trace or {}).get("inputs", {}))
    version = str((trace or {}).get("rule_set_version", ""))
    cands = sorted(candidates)
    view = PanelView(state=status, group=group, subgroup=subgroup, candidates=cands)
    view.rule_set_version = version
    if status == "resolved" and group is not None:
        view.heading = f"Group {group}" + (f" · subgroup {subgroup}" if subgroup else "")
        view.rows = _resolved_rows(entries, inputs, group, subgroup)
        view.group_name = labels.GROUP_NAMES.get(group, "")
        view.subgroup_name = labels.SUBGROUP_NAMES.get(subgroup or "", "")
        view.footnote = f"WHO Robson rule set {version}."
        if not any(e["group"] == group and e["field"] == "onset_of_labour" for e in entries):
            view.footnote += f" Onset of labour does not affect group {group}."
    elif status == "conflict":
        view.heading = "No group assigned"
        pairs = [
            f"{labels.FIELD_NAMES[f].lower()} {_plain(inputs.get(f))}".strip() for f in conflict
        ]
        view.need_title = "These can't both be true"
        view.need_value = (
            " and ".join(pairs) if pairs else "The entered values contradict each other"
        )
        view.need_why = labels.CONSISTENCY_TEXT.get(
            consistency_name or "nulliparous_with_previous_cs",
            "Check which value is wrong.",
        )
        view.need_fields = list(conflict)
        view.footnote = f"Consistency check (rule set {version})."
        view.foot = view.footnote
    else:
        view.candidates = cands
        view.heading = "Group " + " or ".join(str(g) for g in cands) if cands else "No group yet"
        view.rows = _open_rows(entries, inputs, cands) if len(cands) < 10 else []
        if len(cands) >= 10:
            view.state = "empty"
            view.heading = "Waiting for the Robson fields"
        view.need_fields = list(resolving)
        if len(resolving) == 1:
            view.need_title = "One field decides it"
            view.need_value = labels.FIELD_NAMES[resolving[0]]
            view.need_why = _why(entries, resolving[0], cands)
        elif resolving:
            view.need_title = "Fields that decide it"
            view.need_value = ", ".join(labels.FIELD_NAMES[f] for f in resolving)
        view.footnote = ""
        onset_matters = any(
            e["field"] == "onset_of_labour" and e["group"] in cands for e in entries
        )
        if (
            view.state == "partial"
            and inputs.get("onset_of_labour") is None
            and cands
            and not onset_matters
        ):
            word = "group" if len(cands) == 1 else "groups"
            names = " or ".join(str(g) for g in cands)
            view.foot = f"Onset of labour is not needed here: it does not change {word} {names}."
    return view


def _plain(value: Any) -> str:
    return "" if value is None else str(value)


def classify_inputs(
    inputs: RobsonInputs, rule_set: EngineRuleSet
) -> tuple[ClassificationResult, dict[str, Any]]:
    result = classify(inputs, rule_set)
    return result, stored_trace(result, inputs)


def view_of_result(result: ClassificationResult, trace: dict[str, Any]) -> PanelView:
    return panel_view(
        status=result.status,
        group=result.group,
        subgroup=result.subgroup,
        candidates=sorted(result.candidates),
        resolving=list(result.resolving_fields),
        conflict=list(result.conflict_fields),
        trace=trace,
    )
