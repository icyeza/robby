"""Reading the model registry for its pages: the models, their transition history and the
usernames those name. Changes of state go through robby.registry.transitions."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from robby.models import AppUser, ModelState, ModelTransition, ModelVersion


def all_models(db: Session) -> Sequence[ModelVersion]:
    """Every registered model, newest first."""
    return db.scalars(select(ModelVersion).order_by(ModelVersion.registered_at.desc())).all()


def model_by_label(db: Session, label: str) -> ModelVersion | None:
    return db.scalar(select(ModelVersion).where(ModelVersion.version_label == label))


def previous_active(db: Session, current: ModelVersion) -> ModelVersion | None:
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


def history_of(db: Session, model: ModelVersion) -> Sequence[ModelTransition]:
    """The model's transitions, newest first."""
    return db.scalars(
        select(ModelTransition)
        .where(ModelTransition.model_id == model.id)
        .order_by(ModelTransition.at.desc(), ModelTransition.created_at.desc())
    ).all()


def usernames(db: Session, ids: Iterable[uuid.UUID | None]) -> dict[uuid.UUID, str]:
    """Username by user id, in one query (None ids are ignored)."""
    wanted = [i for i in ids if i is not None]
    if not wanted:
        return {}
    return {u.id: u.username for u in db.scalars(select(AppUser).where(AppUser.id.in_(wanted)))}
