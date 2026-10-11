"""The initial migration builds exactly the schema the models describe, and can be undone."""

from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import Engine, create_engine, inspect, text

from robby.migrate import downgrade, upgrade
from robby.models import Base
from tests.dbutil import create_database, drop_database, server_url_from_env
from tests.expected import EXPECTED_ENUMS, EXPECTED_TABLES

ENUM_LABELS = text(
    "SELECT e.enumlabel FROM pg_type t JOIN pg_enum e ON e.enumtypid = t.oid "
    "WHERE t.typname = :name ORDER BY e.enumsortorder"
)


def test_migrations_create_every_table(engine: Engine) -> None:
    with engine.connect() as conn:
        assert set(inspect(conn).get_table_names()) == EXPECTED_TABLES | {"alembic_version"}


def test_models_match_the_migrated_schema(engine: Engine) -> None:
    with engine.connect() as conn:
        context = MigrationContext.configure(conn, opts={"compare_type": True})
        differences = compare_metadata(context, Base.metadata)
    assert differences == []


def test_postgres_enums_match_the_python_enums(engine: Engine) -> None:
    with engine.connect() as conn:
        for name, python_enum in EXPECTED_ENUMS.items():
            labels = conn.execute(ENUM_LABELS, {"name": name}).scalars().all()
            assert labels == [member.value for member in python_enum], name


def test_downgrade_to_base_then_upgrade_again() -> None:
    url = create_database(server_url_from_env())
    raw = url.render_as_string(hide_password=False)
    try:
        upgrade(raw)
        downgrade(raw, "base")
        engine = create_engine(url)
        with engine.connect() as conn:
            assert set(inspect(conn).get_table_names()) == {"alembic_version"}
            leftover = conn.execute(
                text("SELECT count(*) FROM pg_type WHERE typname = 'user_role'")
            ).scalar_one()
            assert leftover == 0
        engine.dispose()
        upgrade(raw)
        engine = create_engine(url)
        with engine.connect() as conn:
            assert EXPECTED_TABLES <= set(inspect(conn).get_table_names())
        engine.dispose()
    finally:
        drop_database(url)
