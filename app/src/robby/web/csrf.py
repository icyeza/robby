"""CSRF protection.

Signed-in forms carry token = HMAC-SHA256(SECRET_KEY, "csrf:" + session id) in a hidden field;
htmx sends the same token in the X-CSRF-Token header (app.js reads it from a meta tag). The
sign-in form has no session yet: it carries a nonce signed with itsdangerous, and the same nonce
sits in an HttpOnly cookie scoped to /login."""

import hashlib
import hmac
import secrets
import uuid
from typing import Annotated

from fastapi import Depends, Request
from itsdangerous import BadSignature, URLSafeTimedSerializer

from robby.auth.dependencies import CurrentUser, SettingsDep, current_session
from robby.config import Settings

CSRF_FIELD = "csrf_token"
CSRF_HEADER = "X-CSRF-Token"
LOGIN_NONCE_COOKIE = "robby_login"
LOGIN_TOKEN_MAX_AGE = 2 * 60 * 60


class CsrfError(Exception):
    """A state-changing request arrived without a valid CSRF token."""


def session_csrf_token(settings: Settings, session_id: uuid.UUID) -> str:
    message = f"csrf:{session_id}".encode()
    return hmac.new(settings.secret_bytes, message, hashlib.sha256).hexdigest()


def tokens_match(supplied: object, expected: str) -> bool:
    return isinstance(supplied, str) and hmac.compare_digest(
        supplied.encode("utf-8"), expected.encode("utf-8")
    )


async def verify_csrf(
    request: Request,
    current: Annotated[CurrentUser, Depends(current_session)],
    settings: SettingsDep,
) -> None:
    supplied: object = request.headers.get(CSRF_HEADER)
    if supplied is None:
        form = await request.form()
        supplied = form.get(CSRF_FIELD)
    if not tokens_match(supplied, session_csrf_token(settings, current.session.id)):
        raise CsrfError


CsrfChecked = Annotated[None, Depends(verify_csrf)]


def _login_serializer(settings: Settings) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(settings.secret_key.get_secret_value(), salt="robby-login-form")


def new_login_token(settings: Settings) -> tuple[str, str]:
    """A nonce for the cookie and the signed token for the form."""
    nonce = secrets.token_urlsafe(16)
    return nonce, _login_serializer(settings).dumps(nonce)


def check_login_token(
    settings: Settings, token: str, nonce: str | None, max_age: int = LOGIN_TOKEN_MAX_AGE
) -> bool:
    if not token or not nonce:
        return False
    try:
        value = _login_serializer(settings).loads(token, max_age=max_age)
    except BadSignature:
        return False
    return isinstance(value, str) and hmac.compare_digest(value, nonce)
