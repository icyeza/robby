"""Declarative base, naming convention and shared column types for every table."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Annotated, Any

from sqlalchemy import Date, DateTime, Dialect, LargeBinary, MetaData, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, mapped_column
from sqlalchemy.sql import operators
from sqlalchemy.types import TypeDecorator

from robby import crypto

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


class EncryptedText(TypeDecorator[str]):
    """Text held as AES-256-GCM ciphertext in a bytea column (robby.crypto); Python sees str.
    `column` is the associated data, so a value only decrypts in the column it was written to.
    Each write uses a fresh nonce, so SQL cannot compare values: anything but IS NULL raises."""

    impl = LargeBinary
    cache_ok = True

    def __init__(self, column: str) -> None:
        super().__init__()
        self.column = column

    class comparator_factory(TypeDecorator.Comparator[str]):  # noqa: N801 (SQLAlchemy's name)
        def operate(self, op: Any, *other: Any, **kwargs: Any) -> Any:
            if op in (operators.is_, operators.is_not):
                return super().operate(op, *other, **kwargs)
            raise TypeError("encrypted column: compare decrypted values in Python instead")

        def reverse_operate(self, op: Any, other: Any, **kwargs: Any) -> Any:
            raise TypeError("encrypted column: compare decrypted values in Python instead")

    def process_bind_param(self, value: str | None, dialect: Dialect) -> bytes | None:
        return None if value is None else crypto.active().encrypt(value, self.column)

    def process_result_value(self, value: bytes | None, dialect: Dialect) -> str | None:
        return None if value is None else crypto.active().decrypt(bytes(value), self.column)


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
