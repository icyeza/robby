"""/privacy and /terms (public, no sign-in) and the acceptance step shown to a signed-in user who
has not accepted the current versions."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse, Response

from robby.auth.dependencies import CurrentUser, DbDep, current_session
from robby.config import Settings
from robby.policy import service
from robby.timeutil import utcnow
from robby.web.csrf import CsrfChecked
from robby.web.render import render

router = APIRouter()
SignedIn = Annotated[CurrentUser, Depends(current_session)]


def _versions() -> dict[str, Any]:
    return {
        "privacy_version": service.PRIVACY_VERSION,
        "privacy_date": service.PRIVACY_DATE,
        "terms_version": service.TERMS_VERSION,
        "terms_date": service.TERMS_DATE,
    }


@router.get("/privacy")
def privacy(request: Request) -> Response:
    return render(request, "policy/privacy.html", **_versions())


@router.get("/terms")
def terms(request: Request) -> Response:
    settings: Settings = request.app.state.settings
    return render(
        request,
        "policy/terms.html",
        idle_minutes=settings.session_idle_minutes,
        absolute_hours=settings.session_absolute_hours,
        **_versions(),
    )


@router.get("/accept-policies")
def accept_form(request: Request, db: DbDep, current: SignedIn) -> Response:
    if service.has_accepted(db, current.user):
        return RedirectResponse("/", status_code=303)
    # No navigation here: its polling badge would be redirected back to this page, in a loop.
    return render(request, "policy/accept.html", current=current, nav=[], **_versions())


@router.post("/accept-policies")
def accept_submit(request: Request, db: DbDep, current: SignedIn, _csrf: CsrfChecked) -> Response:
    service.accept(db, current.user, utcnow())
    db.commit()
    return RedirectResponse("/", status_code=303)
