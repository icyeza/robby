"""Database engine, sessions and the request-scoped session dependency."""

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from starlette.requests import Request

# Connections per Uvicorn worker. Requests run in a thread pool, so this caps how many reach the
# database at once; a request that finds them all busy waits at most POOL_TIMEOUT_SECONDS and is
# then answered "busy, try again" (web/security.py) instead of hanging. Two workers use at most
# 2 x (POOL_SIZE + MAX_OVERFLOW) = 40 of PostgreSQL's default 100 connections.
POOL_SIZE = 10
MAX_OVERFLOW = 10
POOL_TIMEOUT_SECONDS = 5


@lru_cache(maxsize=4)
def get_engine(database_url: str) -> Engine:
    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_size=POOL_SIZE,
        max_overflow=MAX_OVERFLOW,
        pool_timeout=POOL_TIMEOUT_SECONDS,
    )


def get_sessionmaker(database_url: str) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(database_url), expire_on_commit=False)


def get_db(request: Request) -> Iterator[Session]:
    """One session per request; uncommitted work is rolled back when the session closes."""
    factory: sessionmaker[Session] = request.app.state.sessionmaker
    with factory() as session:
        yield session
