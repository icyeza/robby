"""Helpers for tests that drive the application through Starlette's TestClient."""

from collections.abc import Iterator

from fastapi import FastAPI
from sqlalchemy import Connection
from sqlalchemy.orm import Session

from robby.config import Settings
from robby.db import get_db
from robby.main import create_app
from tests.dbutil import bound_session


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
