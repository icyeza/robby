"""DEVELOPMENT SHORTCUT: `robby-admin model register`.

Registers a packaged artefact so the clinical flow has a model to show. It validates the package
(robson_serving.load and the golden check), copies it under ARTEFACT_DIR/<version_label>/ and
records a model_version row. `--activate` makes it the active model straight away, without the
approval flow, and says so in model_transition and the event log.

Without --activate the model is registered as a candidate and goes through the approval flow on
the model registry pages (shadow, promotion, rollback; see robby.registry.transitions)."""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
from robson_serving import IncompatibleArtefact, load
from robson_serving.contract import check_golden
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from robby import events
from robby.config import get_settings
from robby.db import get_sessionmaker
from robby.models import AppUser, ModelState, ModelTransition, ModelVersion

model_app = typer.Typer(no_args_is_help=True)
DEV_REASON = "development activation"
DEV_REASON_RETIRED = "development activation of another model"


@model_app.callback()
def _model() -> None:
    """Model registry (development shortcut; approvals happen on the model registry pages)."""


def _month_bounds(start: str, end: str) -> tuple[date, date]:
    sy, sm = (int(x) for x in start.split("-")[:2])
    ey, em = (int(x) for x in end.split("-")[:2])
    last = date(ey + (em == 12), em % 12 + 1, 1).toordinal() - 1
    return date(sy, sm, 1), date.fromordinal(last)


def _fail(message: str) -> NoReturn:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)


def _validated_manifest(packaged_dir: Path) -> dict[str, Any]:
    """The manifest of a package that loads and passes its golden check (else exit 1)."""
    try:
        scorer = load(packaged_dir)
        check = check_golden(packaged_dir, scorer)
    except IncompatibleArtefact as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not check.ok:
        _fail(f"golden check failed (max difference {check.max_abs_diff}).")
    return scorer.manifest


def _model_row(manifest: dict[str, Any], label: str, user: AppUser | None) -> ModelVersion:
    """A candidate model_version for the package copied to ARTEFACT_DIR/<label>."""
    start, end = _month_bounds(
        str(manifest["training_window_start"]), str(manifest["training_window_end"])
    )
    return ModelVersion(
        version_label=label,
        algorithm=str(manifest["algorithm"]),
        feature_set=str(manifest["feature_set"]),
        feature_list=list(manifest["feature_list"]),
        hyperparameters=dict(manifest["hyperparameters"]),
        calibration_method=str(manifest["calibration_method"]),
        evaluation_metrics=dict(manifest["evaluation_metrics"]),
        artefact_hash=str(manifest["artefact_hash"]),
        artefact_path=label,
        manifest=manifest,
        training_window_start=start,
        training_window_end=end,
        state=ModelState.CANDIDATE,
        registered_by=user.id if user else None,
    )


def _transition(
    db: Session, model: ModelVersion, from_state: ModelState, to_state: ModelState, user: AppUser
) -> None:
    db.execute(update(ModelVersion).where(ModelVersion.id == model.id).values(state=to_state))
    reason = DEV_REASON if to_state is ModelState.ACTIVE else DEV_REASON_RETIRED
    db.add(
        ModelTransition(
            model_id=model.id,
            from_state=from_state,
            to_state=to_state,
            actor_id=user.id,
            reason=reason,
        )
    )


def _activate(db: Session, row: ModelVersion, user: AppUser) -> None:
    """Make `row` the active model without approval, retiring the current one, and say so."""
    for old in db.scalars(select(ModelVersion).where(ModelVersion.state == ModelState.ACTIVE)):
        _transition(db, old, ModelState.ACTIVE, ModelState.RETIRED, user)
    db.flush()
    _transition(db, row, ModelState.CANDIDATE, ModelState.ACTIVE, user)
    events.append(
        db,
        user,
        "model.activated",
        ("model_version", row.id),
        {"version_label": row.version_label, "reason": DEV_REASON},
    )


@model_app.command("register")
def register(
    packaged_dir: Annotated[
        Path, typer.Argument(help="Packaged artefact directory.", file_okay=False)
    ],
    activate: Annotated[
        bool, typer.Option("--activate", help="Development activation: no approval flow.")
    ] = False,
    actor: Annotated[
        str | None, typer.Option("--actor", help="Username recorded for --activate.")
    ] = None,
) -> None:
    """Validate a packaged model, copy it into ARTEFACT_DIR and register it."""
    settings = get_settings()
    manifest = _validated_manifest(packaged_dir)
    label = str(manifest["version_label"])
    with get_sessionmaker(settings.database_url)() as db:
        if db.scalar(select(ModelVersion).where(ModelVersion.version_label == label)):
            _fail(f"{label} is already registered.")
        user: AppUser | None = None
        if activate:
            user = db.scalar(select(AppUser).where(AppUser.username == (actor or "").lower()))
            if user is None:
                _fail("--activate needs --actor <existing username>.")
        shutil.copytree(packaged_dir, settings.artefact_dir / label)
        row = _model_row(manifest, label, user)
        db.add(row)
        db.flush()
        events.append(
            db, user, "model.registered", ("model_version", row.id), {"version_label": label}
        )
        if activate and user is not None:
            _activate(db, row, user)
        db.commit()
    state = "active (development activation, no approval)" if activate else "candidate"
    typer.echo(f"Registered {label} as {state}.")
