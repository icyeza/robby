"""Helpers for tests that drive the application through Starlette's TestClient."""

import re
from collections.abc import Iterator

from fastapi import FastAPI
from httpx2 import Response
from sqlalchemy import Connection
from sqlalchemy.orm import Session
from starlette.testclient import TestClient

from robby.auth.dependencies import SESSION_COOKIE
from robby.auth.service import start_session
from robby.config import Settings
from robby.db import get_db
from robby.main import create_app
from robby.models import AppUser, UserSession
from robby.timeutil import utcnow
from robby.web.csrf import session_csrf_token
from tests.dbutil import bound_session
from tests.factories import DEFAULT_PASSWORD


def make_test_app(settings: Settings, connection: Connection) -> FastAPI:
    """The real application, with every request session joined to the test's transaction."""
    application = create_app(settings)

    def joined_session() -> Iterator[Session]:
        session = bound_session(connection)
        try:
            yield session
        finally:
            session.close()

    application.dependency_overrides[get_db] = joined_session
    return application


CSRF_INPUT = re.compile(r'name="csrf_token" value="([^"]+)"')


def sign_in_client(
    client: TestClient, db: Session, settings: Settings, user: AppUser
) -> UserSession:
    """Open a session for `user` directly (no password check) and give the client its cookie."""
    row, token = start_session(db, settings, user, ip="testclient", user_agent="pytest", now=utcnow())
    db.commit()
    client.cookies.set(SESSION_COOKIE, token)
    return row


def csrf_for(settings: Settings, row: UserSession) -> str:
    return session_csrf_token(settings, row.id)


def form_token(html: str) -> str:
    match = CSRF_INPUT.search(html)
    assert match is not None, "no csrf_token field on the page"
    return match.group(1)


def log_in(client: TestClient, username: str, password: str = DEFAULT_PASSWORD) -> Response:
    """Sign in through the real form, as a browser would."""
    page = client.get("/login")
    return client.post(
        "/login",
        data={"username": username, "password": password, "csrf_token": form_token(page.text)},
    )


def set_cookie(response: Response, name: str) -> str:
    """The Set-Cookie header for `name`, lower-cased."""
    for header in response.headers.get_list("set-cookie"):
        if header.startswith(f"{name}="):
            return header.lower()
    raise AssertionError(f"no Set-Cookie header for {name}")
