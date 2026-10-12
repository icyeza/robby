"""DEVELOPMENT SHORTCUT: `robby-admin model register`.

Registers a packaged artefact so the clinical flow has a model to show. It validates the package
(robson_serving.load and the golden check), copies it under ARTEFACT_DIR/<version_label>/ and
records a model_version row. `--activate` makes it the active model straight away, without the
approval flow, and says so in model_transition and the event log.

The real governance (candidate review, shadow, approval by a second person, rollback) is
sub-project 4 and replaces this command's activation path."""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path
from typing import Annotated, Any

import typer
from robson_serving import IncompatibleArtefact, load
from robson_serving.contract import check_golden
from sqlalchemy import select, update

from robby import events
from robby.config import get_settings
from robby.db import get_sessionmaker
from robby.models import AppUser, ModelState, ModelTransition, ModelVersion

model_app = typer.Typer(no_args_is_help=True)


@model_app.callback()
def _model() -> None:
    """Model registry (development shortcut; governance arrives in a later sub-project)."""


def _month_bounds(start: str, end: str) -> tuple[date, date]:
    sy, sm = (int(x) for x in start.split("-")[:2])
    ey, em = (int(x) for x in end.split("-")[:2])
    last = date(ey + (em == 12), em % 12 + 1, 1).toordinal() - 1
    return date(sy, sm, 1), date.fromordinal(last)


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
    try:
        scorer = load(packaged_dir)
        check = check_golden(packaged_dir, scorer)
    except IncompatibleArtefact as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not check.ok:
        typer.echo(f"Error: golden check failed (max difference {check.max_abs_diff}).", err=True)
        raise typer.Exit(code=1)
    manifest: dict[str, Any] = scorer.manifest
    label = str(manifest["version_label"])
    with get_sessionmaker(settings.database_url)() as db:
        if db.scalar(select(ModelVersion).where(ModelVersion.version_label == label)):
            typer.echo(f"Error: {label} is already registered.", err=True)
            raise typer.Exit(code=1)
        user: AppUser | None = None
        if activate:
            user = db.scalar(select(AppUser).where(AppUser.username == (actor or "").lower()))
            if user is None:
                typer.echo("Error: --activate needs --actor <existing username>.", err=True)
                raise typer.Exit(code=1)
        target = settings.artefact_dir / label
        shutil.copytree(packaged_dir, target)
        start, end = _month_bounds(
            str(manifest["training_window_start"]), str(manifest["training_window_end"])
        )
        row = ModelVersion(
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
        db.add(row)
        db.flush()
        events.append(
            db, user, "model.registered", ("model_version", row.id), {"version_label": label}
        )
        if activate and user is not None:
            for old in db.scalars(
                select(ModelVersion).where(ModelVersion.state == ModelState.ACTIVE)
            ):
                db.execute(
                    update(ModelVersion)
                    .where(ModelVersion.id == old.id)
                    .values(state=ModelState.RETIRED)
                )
                db.add(
                    ModelTransition(
                        model_id=old.id,
                        from_state=ModelState.ACTIVE,
                        to_state=ModelState.RETIRED,
                        actor_id=user.id,
                        reason="development activation of another model",
                    )
                )
            db.flush()
            db.execute(
                update(ModelVersion)
                .where(ModelVersion.id == row.id)
                .values(state=ModelState.ACTIVE)
            )
            db.add(
                ModelTransition(
                    model_id=row.id,
                    from_state=ModelState.CANDIDATE,
                    to_state=ModelState.ACTIVE,
                    actor_id=user.id,
                    reason="development activation",
                )
            )
            events.append(
                db,
                user,
                "model.activated",
                ("model_version", row.id),
                {"version_label": label, "reason": "development activation"},
            )
        db.commit()
    state = "active (development activation, no approval)" if activate else "candidate"
    typer.echo(f"Registered {label} as {state}.")
