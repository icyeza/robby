"""Database engine, sessions and the request-scoped session dependency."""

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from starlette.requests import Request


@lru_cache(maxsize=4)
def get_engine(database_url: str) -> Engine:
    return create_engine(database_url, pool_pre_ping=True)


def get_sessionmaker(database_url: str) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(database_url), expire_on_commit=False)


def get_db(request: Request) -> Iterator[Session]:
    """One session per request; uncommitted work is rolled back when the session closes."""
    factory: sessionmaker[Session] = request.app.state.sessionmaker
    with factory() as session:
        yield session
