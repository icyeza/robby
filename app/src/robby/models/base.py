"""Declarative base, naming convention and shared column types for every table."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Annotated, Any

from sqlalchemy import Date, DateTime, MetaData, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, mapped_column

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# Generated in Python (so ids exist before flush) and by the database (rows inserted by SQL).
UUIDPK = Annotated[
    uuid.UUID,
    mapped_column(primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()")),
]
CreatedAt = Annotated[datetime, mapped_column(server_default=func.now())]


class Base(DeclarativeBase):
    """Base class of every ORM model."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {  # noqa: RUF012
        uuid.UUID: UUID(as_uuid=True),
        datetime: DateTime(timezone=True),
        date: Date(),
        str: Text(),
        dict[str, Any]: JSONB(),
    }
