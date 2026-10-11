"""Create and drop throwaway PostgreSQL databases on the server named by ROBBY_TEST_DATABASE_URL.

The database name in that URL is only a prefix: each run creates <prefix>_<random> through the
server's maintenance database "postgres", so the role needs CREATEDB (README, "Tests")."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
from dotenv import dotenv_values
from sqlalchemy import URL, Connection, Engine, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

APP_ROOT = Path(__file__).resolve().parents[1]


def server_url_from_env() -> URL:
    raw = os.environ.get("ROBBY_TEST_DATABASE_URL") or dotenv_values(APP_ROOT / ".env").get(
        "ROBBY_TEST_DATABASE_URL"
    )
    if not raw:
        pytest.exit("ROBBY_TEST_DATABASE_URL is not set; see app/README.md, 'Tests'.", returncode=2)
    return make_url(raw)


def _maintenance_engine(url: URL) -> Engine:
    return create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")


def create_database(server: URL) -> URL:
    """Create an empty UTF-8 database and return its URL."""
    name = f"{server.database or 'robby_test'}_{uuid.uuid4().hex[:10]}"
    engine = _maintenance_engine(server)
    try:
        with engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE \"{name}\" TEMPLATE template0 ENCODING 'UTF8'"))
    finally:
        engine.dispose()
    return server.set(database=name)


def drop_database(url: URL) -> None:
    engine = _maintenance_engine(url)
    try:
        with engine.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{url.database}" WITH (FORCE)'))
    finally:
        engine.dispose()


def bound_session(connection: Connection) -> Session:
    """A session joined to the test's transaction: its commit() only releases a savepoint,
    so everything is still rolled back when the test ends."""
    return Session(
        bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False
    )
