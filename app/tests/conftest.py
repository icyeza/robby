"""Shared fixtures.

One throwaway database per test run: created on the local server, migrated to head once, dropped at
the end. Each test gets one connection inside a transaction that is rolled back afterwards; every
Session used by the test (the `db` fixture, the app's request sessions, CLI sessions) joins that
transaction through savepoints, so tests never see each other's rows."""

from collections.abc import Iterator

import pytest
from sqlalchemy import URL, Connection, Engine, create_engine
from sqlalchemy.orm import Session

from robby.config import Settings
from robby.migrate import upgrade
from tests.dbutil import bound_session, create_database, drop_database, server_url_from_env


@pytest.fixture(scope="session")
def database_url() -> Iterator[URL]:
    url = create_database(server_url_from_env())
    try:
        upgrade(url.render_as_string(hide_password=False))
        yield url
    finally:
        drop_database(url)


@pytest.fixture(scope="session")
def engine(database_url: URL) -> Iterator[Engine]:
    engine = create_engine(database_url)
    yield engine
    engine.dispose()


@pytest.fixture
def connection(engine: Engine) -> Iterator[Connection]:
    with engine.connect() as conn:
        transaction = conn.begin()
        try:
            yield conn
        finally:
            transaction.rollback()


@pytest.fixture
def db(connection: Connection) -> Iterator[Session]:
    session = bound_session(connection)
    try:
        yield session
    finally:
        session.close()


TEST_SECRET = "test-secret-key-that-is-long-enough-0123456789"


@pytest.fixture
def settings(database_url: URL) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        database_url=database_url.render_as_string(hide_password=False),
        secret_key=TEST_SECRET,
        secure_cookies=False,
        session_idle_minutes=30,
        session_absolute_hours=12,
        timezone="Africa/Kigali",
        demo=False,
    )
