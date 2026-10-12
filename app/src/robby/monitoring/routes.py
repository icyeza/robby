"""Monitoring page and alert acknowledgement (model custodian)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import Response

from robby.auth.dependencies import CurrentUser, DbDep, require
from robby.models import Role
from robby.monitoring import service
from robby.web.csrf import CsrfChecked
from robby.web.errors import redirect
from robby.web.render import render

router = APIRouter(prefix="/monitoring")
Custodian = Annotated[CurrentUser, Depends(require(Role.MODEL_CUSTODIAN))]


@router.get("")
def monitoring_page(request: Request, db: DbDep, current: Custodian) -> Response:
    settings = request.app.state.settings
    data = service.build(db, artefact_dir=settings.artefact_dir)
    return render(
        request,
        "monitoring/index.html",
        current=current,
        m=data,
    )


@router.post("/alerts/acknowledge")
def acknowledge(
    request: Request,
    db: DbDep,
    current: Custodian,
    _csrf: CsrfChecked,
    key: Annotated[str, Form(max_length=200)],
) -> Response:
    if not service.acknowledge(
        db, current.user, key, artefact_dir=request.app.state.settings.artefact_dir
    ):
        raise HTTPException(status_code=404)
    db.commit()
    return redirect(request, "/monitoring")
