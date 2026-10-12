# ruff: noqa: RUF001
"""Models list and detail, and the approval actions (custodian only; mockups models.html and
model-detail.html)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from robby.auth.dependencies import CurrentUser, DbDep, require
from robby.errors import FormErrors, NotAllowed, NotFound
from robby.models import AppUser, ModelState, ModelTransition, ModelVersion, Role
from robby.monitoring import service as monitoring
from robby.registry.evidence import build_evidence, calibration_svg
from robby.registry.transitions import RULES, actions_for, apply_transition
from robby.web.csrf import CsrfChecked
from robby.web.errors import redirect
from robby.web.render import render

router = APIRouter(prefix="/models")
Custodian = Annotated[CurrentUser, Depends(require(Role.MODEL_CUSTODIAN))]
FormText = Annotated[str, Form()]

ALGORITHMS = {
    "logreg_l2": "L2 logistic regression",
    "logreg_elasticnet": "Elastic-net logistic regression",
    "xgboost": "Gradient boosting",
}
CALIBRATION = {"platt": "Platt", "isotonic": "Isotonic", "none": "None"}
STATE_LABELS = {
    ModelState.CANDIDATE: "Candidate",
    ModelState.SHADOW: "Shadow",
    ModelState.ACTIVE: "Active",
    ModelState.RETIRED: "Retired",
}


@dataclass
class DialogSpec:
    action: str
    target: ModelVersion
    title: str
    text: str
    button: str
    primary: bool = True


def _beats(model: ModelVersion) -> str:
    value = model.manifest.get("beats_robson_baseline")
    return "Yes" if value is True else "No" if value is False else "—"


def _month(d: Any) -> str:
    return str(d.strftime("%b %Y"))


def _usernames(db: Session, ids: set[uuid.UUID | None]) -> dict[uuid.UUID, str]:
    wanted = [i for i in ids if i is not None]
    if not wanted:
        return {}
    return {u.id: u.username for u in db.scalars(select(AppUser).where(AppUser.id.in_(wanted)))}


def _get(db: Session, label: str) -> ModelVersion:
    model = db.scalar(select(ModelVersion).where(ModelVersion.version_label == label))
    if model is None:
        raise HTTPException(status_code=404)
    return model


def _previous_active(db: Session, current: ModelVersion) -> ModelVersion | None:
    """The model most recently retired from active, if it is still retired."""
    return db.scalar(
        select(ModelVersion)
        .join(ModelTransition, ModelTransition.model_id == ModelVersion.id)
        .where(
            ModelTransition.from_state == ModelState.ACTIVE,
            ModelTransition.to_state == ModelState.RETIRED,
            ModelVersion.state == ModelState.RETIRED,
            ModelVersion.id != current.id,
        )
        .order_by(ModelTransition.at.desc())
        .limit(1)
    )


def _dialogs(db: Session, model: ModelVersion) -> tuple[list[DialogSpec], ModelVersion | None]:
    """The dialogs this page can open, and the model an Active page offers to roll back to."""
    label = model.version_label
    current = db.scalar(select(ModelVersion).where(ModelVersion.state == ModelState.ACTIVE))
    now_active = current.version_label if current else None
    out: list[DialogSpec] = []
    for action in actions_for(model.state):
        if action == "shadow":
            out.append(
                DialogSpec(
                    action,
                    model,
                    f"Move {label} to shadow?",
                    f"{label} will score every new admission in the background. Clinicians will "
                    "not see its results; they are kept to compare it with the active model. "
                    "This approval is recorded with your name and reason.",
                    "Record approval and move to shadow",
                )
            )
        elif action == "promote":
            replaced = f" {now_active} will become retired." if now_active else ""
            out.append(
                DialogSpec(
                    action,
                    model,
                    f"Promote {label} to active?",
                    f"{label} will produce the readiness signal on every new admission.{replaced} "
                    "This approval is recorded with your name and reason.",
                    "Record approval and promote",
                )
            )
        elif action == "rollback":
            replaced = f" and {now_active} will be retired" if now_active else ""
            out.append(
                DialogSpec(
                    action,
                    model,
                    f"Roll back to {label}?",
                    f"{label} will become active again{replaced}. "
                    "This approval is recorded with your name and reason.",
                    "Record approval and roll back",
                )
            )
        elif action == "retire":
            out.append(
                DialogSpec(
                    action,
                    model,
                    f"Retire {label}?",
                    f"{label} will stop scoring admissions and cannot be used again except by a "
                    "rollback. This approval is recorded with your name and reason.",
                    "Record approval and retire",
                    primary=False,
                )
            )
    rollback_to = _previous_active(db, model) if model.state is ModelState.ACTIVE else None
    return out, rollback_to


@router.get("")
def models_page(request: Request, db: DbDep, current: Custodian) -> Response:
    rows = db.scalars(select(ModelVersion).order_by(ModelVersion.registered_at.desc())).all()
    display = [
        {
            "label": m.version_label,
            "state": m.state.value,
            "state_label": STATE_LABELS[m.state],
            "algorithm": ALGORITHMS.get(m.algorithm, m.algorithm),
            "trained": f"{_month(m.training_window_start)} – {_month(m.training_window_end)}",
            "calibration": CALIBRATION.get(m.calibration_method, m.calibration_method),
            "beats": _beats(m),
            "registered": m.registered_at,
        }
        for m in rows
    ]
    try:
        open_alerts = monitoring.build(
            db, artefact_dir=request.app.state.settings.artefact_dir
        ).open_alerts
    except Exception:  # the models page must not depend on monitoring
        open_alerts = 0
    return render(
        request, "models/list.html", current=current, rows=display, open_alerts=open_alerts
    )


def _detail(
    request: Request,
    db: Session,
    current: CurrentUser,
    model: ModelVersion,
    *,
    status_code: int = 200,
    open_dialog: str = "",
    form: dict[str, str] | None = None,
    errors: dict[str, str] | None = None,
    done: str = "",
) -> Response:
    dialogs, rollback_to = _dialogs(db, model)
    history = db.scalars(
        select(ModelTransition)
        .where(ModelTransition.model_id == model.id)
        .order_by(ModelTransition.at.desc(), ModelTransition.created_at.desc())
    ).all()
    names = _usernames(
        db,
        {model.registered_by, model.approved_by, *[t.actor_id for t in history]},
    )
    evidence = build_evidence(db, model)
    plot = (
        calibration_svg(evidence.candidate_bins, evidence.active_bins)
        if evidence.status == "ready"
        else ""
    )
    manifest = model.manifest
    return render(
        request,
        "models/detail.html",
        current=current,
        status_code=status_code,
        model=model,
        state_label=STATE_LABELS[model.state],
        algorithm=ALGORITHMS.get(model.algorithm, model.algorithm),
        calibration=CALIBRATION.get(model.calibration_method, model.calibration_method),
        trained=f"{_month(model.training_window_start)} – {_month(model.training_window_end)}",
        beats=_beats(model),
        rule_set=manifest.get("rule_set_version", "—"),
        features=len(model.feature_list),
        registered_by=names.get(model.registered_by) if model.registered_by else None,
        approved_by=names.get(model.approved_by) if model.approved_by else None,
        history=[(t, names.get(t.actor_id, "—")) for t in history],
        dialogs=dialogs,
        rollback_to=rollback_to,
        open_dialog=open_dialog,
        form=form or {},
        errors=errors or {},
        done=RULES[done].done if done in RULES else "",
        evidence=evidence,
        plot=plot,
        username=current.user.username,
    )


@router.get("/{label}")
def model_page(
    request: Request, label: str, db: DbDep, current: Custodian, dialog: str = "", done: str = ""
) -> Response:
    model = _get(db, label)
    return _detail(request, db, current, model, open_dialog=dialog, done=done)


@router.post("/{label}/transition")
def transition(
    request: Request,
    label: str,
    db: DbDep,
    current: Custodian,
    _csrf: CsrfChecked,
    action: FormText = "",
    approver_name: FormText = "",
    reason: FormText = "",
) -> Response:
    model = _get(db, label)
    form = {"approver_name": approver_name, "reason": reason}
    try:
        apply_transition(
            db, current.user, model.id, action, approver_name=approver_name, reason=reason
        )
        db.commit()
    except FormErrors as exc:
        db.rollback()
        return _detail(
            request,
            db,
            current,
            _get(db, label),
            status_code=400,
            open_dialog=action,
            form=form,
            errors=exc.errors,
        )
    except NotAllowed as exc:
        db.rollback()
        return _detail(
            request,
            db,
            current,
            _get(db, label),
            status_code=400,
            open_dialog=action,
            form=form,
            errors={"form": str(exc)},
        )
    except NotFound as exc:
        raise HTTPException(status_code=404) from exc
    return redirect(request, f"/models/{label}?done={action}")
