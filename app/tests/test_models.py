"""The ORM metadata describes every table the schema must contain (tests/expected.py)."""

from sqlalchemy import Enum

from robby.models import Base
from tests.expected import EXPECTED_ENUMS, EXPECTED_TABLES


def test_every_table_in_the_spec_is_mapped() -> None:
    assert set(Base.metadata.tables) == EXPECTED_TABLES


def test_uuid_primary_keys_default_to_gen_random_uuid() -> None:
    with_uuid_id = [t for t in Base.metadata.sorted_tables if "id" in t.c and t.c.id.primary_key]
    assert len(with_uuid_id) == 17
    for table in with_uuid_id:
        default = table.c.id.server_default
        assert default is not None and "gen_random_uuid()" in str(default.arg), table.name


def test_every_table_has_a_created_at_timestamptz() -> None:
    for table in Base.metadata.sorted_tables:
        column = table.c.created_at
        assert column.type.timezone is True, table.name
        assert column.server_default is not None, table.name


def test_enumerations_are_named_postgres_enums_with_the_spec_values() -> None:
    seen: dict[str, list[str]] = {}
    for table in Base.metadata.sorted_tables:
        for column in table.c:
            if isinstance(column.type, Enum):
                seen[column.type.name] = list(column.type.enums)
    assert set(seen) == set(EXPECTED_ENUMS)
    for name, python_enum in EXPECTED_ENUMS.items():
        assert seen[name] == [member.value for member in python_enum], name


def test_event_log_seq_is_an_identity_primary_key() -> None:
    seq = Base.metadata.tables["event_log"].c.seq
    assert seq.primary_key and seq.identity is not None
