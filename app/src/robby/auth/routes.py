"""Sign-in, sign-out, "Session ended", password change and the role landing redirect."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from robby.auth.dependencies import (
    SESSION_COOKIE,
    CurrentUser,
    DbDep,
    SettingsDep,
    current_session,
    require,
)
from robby.auth.service import (
    LoginFailed,
    PasswordChangeError,
    SessionState,
    authenticate,
    change_password,
    logout,
    resolve_session,
)
from robby.config import Settings
from robby.timeutil import utcnow
from robby.web.csrf import (
    LOGIN_NONCE_COOKIE,
    LOGIN_TOKEN_MAX_AGE,
    CsrfChecked,
    check_login_token,
    new_login_token,
)
from robby.web.nav import landing_path
from robby.web.render import render

router = APIRouter()
SignedIn = Annotated[CurrentUser, Depends(current_session)]
AnyRole = Annotated[CurrentUser, Depends(require())]
FormText = Annotated[str, Form()]


def _login_page(
    request: Request, settings: Settings, template: str, *, status_code: int = 200, **context: Any
) -> HTMLResponse:
    nonce, token = new_login_token(settings)
    response = render(request, template, status_code=status_code, login_token=token, **context)
    response.set_cookie(
        LOGIN_NONCE_COOKIE, nonce, max_age=LOGIN_TOKEN_MAX_AGE, path="/login",
        httponly=True, secure=settings.secure_cookies, samesite="strict",
    )
    return response


@router.get("/login")
def login_form(request: Request, db: DbDep, settings: SettingsDep) -> Response:
    resolved = resolve_session(db, settings, request.cookies.get(SESSION_COOKIE), now=utcnow())
    if resolved.state is SessionState.VALID:
        return RedirectResponse("/", status_code=303)
    return _login_page(request, settings, "auth/login.html", username="", error=None)


@router.post("/login")
def login(
    request: Request,
    db: DbDep,
    settings: SettingsDep,
    username: FormText = "",
    password: FormText = "",
    csrf_token: FormText = "",
) -> Response:
    if not check_login_token(settings, csrf_token, request.cookies.get(LOGIN_NONCE_COOKIE)):
        return _login_page(
            request, settings, "auth/login.html", status_code=400, username=username, error="expired"
        )
    try:
        _, token = authenticate(
            db, settings, username, password,
            ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            now=utcnow(),
        )
    except LoginFailed:
        db.commit()  # keep the failure count and the event
        return _login_page(
            request, settings, "auth/login.html", status_code=400, username=username,
            error="credentials",
        )
    db.commit()
    response = RedirectResponse("/", status_code=303)
    response.set_cookie(
        SESSION_COOKIE, token, path="/", httponly=True, secure=settings.secure_cookies,
        samesite="strict",
    )
    response.delete_cookie(
        LOGIN_NONCE_COOKIE, path="/login", httponly=True, secure=settings.secure_cookies,
        samesite="strict",
    )
    return response


@router.post("/logout")
def sign_out(db: DbDep, settings: SettingsDep, current: SignedIn, _csrf: CsrfChecked) -> Response:
    logout(db, current.session, now=utcnow())
    db.commit()
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(
        SESSION_COOKIE, path="/", httponly=True, secure=settings.secure_cookies, samesite="strict"
    )
    return response


@router.get("/session-ended")
def session_ended(request: Request, settings: SettingsDep, reason: str = "idle") -> Response:
    return _login_page(
        request, settings, "auth/session_ended.html", username="", error=None,
        reason="absolute" if reason == "absolute" else "idle",
        idle_minutes=settings.session_idle_minutes,
        absolute_hours=settings.session_absolute_hours,
    )


@router.get("/account/password")
def password_form(request: Request, current: SignedIn) -> Response:
    return render(request, "auth/change_password.html", current=current, error=None)


@router.post("/account/password")
def password_submit(
    request: Request,
    db: DbDep,
    current: SignedIn,
    _csrf: CsrfChecked,
    current_password: FormText = "",
    new_password: FormText = "",
    confirm_password: FormText = "",
) -> Response:
    try:
        change_password(db, current.user, current_password, new_password, confirm_password)
    except PasswordChangeError as exc:
        return render(
            request, "auth/change_password.html", current=current, status_code=400, error=str(exc)
        )
    db.commit()
    return RedirectResponse("/", status_code=303)


@router.get("/")
def landing(current: AnyRole) -> Response:
    return RedirectResponse(landing_path(current.user.role), status_code=303)
