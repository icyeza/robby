"""Styled error pages (no stack traces) and the handlers that turn exceptions into responses."""

from typing import cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from robby.auth.dependencies import (
    SESSION_COOKIE,
    Forbidden,
    NotAuthenticated,
    PasswordChangeRequired,
    SessionEnded,
)
from robby.config import Settings
from robby.web.csrf import CsrfError
from robby.web.render import render

MESSAGES: dict[int, tuple[str, str]] = {
    400: ("Request not understood", "The form was incomplete or not valid. Go back and try again."),
    403: ("Not allowed", "Your account does not give access to this page."),
    404: (
        "Page not found",
        "There is nothing at this address. Check the link, or go to your start page.",
    ),
    405: ("Not allowed", "This page cannot be used that way."),
    500: (
        "Something went wrong",
        (
            "The page could not be shown. Try again; if it keeps happening, "
            "tell your system administrator."
        ),
    ),
}
CSRF_TITLE = "This form has expired"
CSRF_MESSAGE = (
    "The page was open for too long, or the form did not come from this site. "
    "Go back, reload the page and try again."
)


def error_page(
    request: Request, status_code: int, *, title: str | None = None, message: str | None = None
) -> HTMLResponse:
    fallback = MESSAGES[500] if status_code >= 500 else MESSAGES[400]
    default_title, default_message = MESSAGES.get(status_code, fallback)
    return render(
        request,
        "errors/error.html",
        status_code=status_code,
        status=status_code,
        title=title or default_title,
        message=message or default_message,
    )


def redirect(request: Request, url: str, *, clear_session: bool = False) -> Response:
    """303 for normal requests; HX-Redirect for htmx requests so the whole page moves."""
    response: Response
    if request.headers.get("HX-Request") == "true":
        response = Response(status_code=200, headers={"HX-Redirect": url})
    else:
        response = RedirectResponse(url, status_code=303)
    if clear_session:
        settings: Settings = request.app.state.settings
        response.delete_cookie(
            SESSION_COOKIE,
            path="/",
            secure=settings.secure_cookies,
            httponly=True,
            samesite="strict",
        )
    return response


async def _http_error(request: Request, exc: Exception) -> Response:
    return error_page(request, cast(StarletteHTTPException, exc).status_code)


async def _validation_error(request: Request, exc: Exception) -> Response:
    errors = cast(RequestValidationError, exc).errors()
    in_path = any(tuple(error.get("loc", ()))[:1] == ("path",) for error in errors)
    return error_page(request, 404 if in_path else 400)


async def _csrf_error(request: Request, exc: Exception) -> Response:
    return error_page(request, 403, title=CSRF_TITLE, message=CSRF_MESSAGE)


async def _not_authenticated(request: Request, exc: Exception) -> Response:
    return redirect(request, "/login", clear_session=True)


async def _session_ended(request: Request, exc: Exception) -> Response:
    reason = cast(SessionEnded, exc).reason
    return redirect(request, f"/session-ended?reason={reason}", clear_session=True)


async def _password_change_required(request: Request, exc: Exception) -> Response:
    return redirect(request, "/account/password")


async def _forbidden(request: Request, exc: Exception) -> Response:
    return error_page(request, 403)


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(CsrfError, _csrf_error)
    app.add_exception_handler(NotAuthenticated, _not_authenticated)
    app.add_exception_handler(SessionEnded, _session_ended)
    app.add_exception_handler(PasswordChangeRequired, _password_change_required)
    app.add_exception_handler(Forbidden, _forbidden)
