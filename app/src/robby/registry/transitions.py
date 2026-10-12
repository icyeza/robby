"""The model lifecycle: candidate -> shadow -> active -> retired, with recorded approvals.

Every transition needs the approver's own username and a written reason, and writes the
model_transition rows, model_version.approved_by/approved_at and one event in the caller's
transaction. The partial unique index on model_version(state) 'active' is the backstop for
"exactly one active model"; the demotion is flushed before the promotion."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from robby import events
from robby.errors import FormErrors, NotAllowed, NotFound
from robby.models import AppUser, ModelState, ModelTransition, ModelVersion, Role
from robby.timeutil import utcnow

MAX_REASON = 1000


@dataclass(frozen=True)
class Rule:
    action: str
    sources: tuple[ModelState, ...]
    target: ModelState
    event: str
    button: str
    done: str


RULES: dict[str, Rule] = {
    "shadow": Rule(
        "shadow",
        (ModelState.CANDIDATE,),
        ModelState.SHADOW,
        "model.shadowed",
        "Move to shadow",
        "moved to shadow",
    ),
    "promote": Rule(
        "promote",
        (ModelState.SHADOW,),
        ModelState.ACTIVE,
        "model.promoted",
        "Promote to active",
        "promoted to active",
    ),
    "rollback": Rule(
        "rollback",
        (ModelState.RETIRED,),
        ModelState.ACTIVE,
        "model.rolled_back",
        "Roll back",
        "made active again by rollback",
    ),
    "retire": Rule(
        "retire",
        (ModelState.CANDIDATE, ModelState.SHADOW),
        ModelState.RETIRED,
        "model.retired",
        "Retire",
        "retired",
    ),
}


def actions_for(state: ModelState) -> list[str]:
    return [name for name, rule in RULES.items() if state in rule.sources]


def apply_transition(
    db: Session,
    actor: AppUser,
    model_id: uuid.UUID,
    action: str,
    *,
    approver_name: str,
    reason: str,
) -> ModelVersion:
    """Does not commit. FormErrors for bad input or a lost race, NotAllowed for a refused move."""
    if actor.role is not Role.MODEL_CUSTODIAN:
        raise NotAllowed("Only a model custodian can approve a model change.")
    rule = RULES.get(action)
    if rule is None:
        raise NotAllowed("Unknown action.")
    errors: dict[str, str] = {}
    if approver_name.strip().lower() != actor.username.lower():
        errors["approver_name"] = "Type your own username to confirm that you are approving this."
    clean = reason.strip()
    if not clean:
        errors["reason"] = "Write the reason for this approval."
    elif len(clean) > MAX_REASON:
        errors["reason"] = f"Keep the reason under {MAX_REASON} characters."
    if errors:
        raise FormErrors(errors)

    rows = list(
        db.scalars(
            select(ModelVersion)
            .where((ModelVersion.id == model_id) | (ModelVersion.state == ModelState.ACTIVE))
            .order_by(ModelVersion.id)
            .with_for_update()
        )
    )
    target = next((r for r in rows if r.id == model_id), None)
    if target is None:
        raise NotFound(str(model_id))
    if target.state not in rule.sources:
        raise NotAllowed(
            f"{target.version_label} is {target.state.value}; "
            f"it cannot be {rule.done} from that state."
        )
    current = next((r for r in rows if r.state is ModelState.ACTIVE and r.id != target.id), None)
    from_state = target.state
    now = utcnow()
    replaced: str | None = None
    try:
        with db.begin_nested():
            if rule.target is ModelState.ACTIVE and current is not None:
                replaced = current.version_label
                prefix = "Replaced by" if action == "promote" else "Rolled back to"
                db.execute(
                    update(ModelVersion)
                    .where(ModelVersion.id == current.id)
                    .values(state=ModelState.RETIRED)
                )
                db.add(
                    ModelTransition(
                        model_id=current.id,
                        from_state=ModelState.ACTIVE,
                        to_state=ModelState.RETIRED,
                        actor_id=actor.id,
                        reason=f"{prefix} {target.version_label}: {clean}",
                        at=now,
                    )
                )
                db.flush()
            values: dict[str, object] = {"state": rule.target}
            if rule.target in (ModelState.SHADOW, ModelState.ACTIVE):
                values.update(approved_by=actor.id, approved_at=now)
            db.execute(update(ModelVersion).where(ModelVersion.id == target.id).values(**values))
            db.add(
                ModelTransition(
                    model_id=target.id,
                    from_state=from_state,
                    to_state=rule.target,
                    actor_id=actor.id,
                    reason=clean,
                    at=now,
                )
            )
            db.flush()
    except IntegrityError as exc:
        raise FormErrors(
            {"form": "Another change was made at the same time. Reload the page and try again."}
        ) from exc
    db.refresh(target)
    events.append(
        db,
        actor,
        rule.event,
        ("model_version", target.id),
        {
            "version_label": target.version_label,
            "from_state": from_state.value,
            "to_state": rule.target.value,
            "approved_by": actor.username,
            "reason": clean,
            "replaced": replaced,
        },
    )
    return target
