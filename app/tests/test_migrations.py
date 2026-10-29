"""The initial migration builds exactly the schema the models describe, and can be undone; 0007
encrypts the identifiers of admissions already stored."""

import uuid
from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import Engine, create_engine, inspect, text

from robby import crypto
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
            assert set(inspect(conn).get_table_names()) >= EXPECTED_TABLES
        engine.dispose()
    finally:
        drop_database(url)


IDENTIFIERS = text("SELECT code, patient_number, patient_name FROM admission ORDER BY code")
TRIGGER_STATE = text("SELECT tgenabled FROM pg_trigger WHERE tgname = 'admission_immutable'")


def _admissions_at_0006(engine: Engine) -> uuid.UUID:
    """A facility with a form admission carrying identifiers and a FHIR one without; its id."""
    with engine.begin() as conn:
        facility = conn.execute(
            text(
                "INSERT INTO facility (name, code, district) "
                "VALUES ('Migration Hospital', 'MIG', 'Test') RETURNING id"
            )
        ).scalar_one()
        user = conn.execute(
            text(
                "INSERT INTO app_user (username, password_hash, role, facility_id) "
                "VALUES ('mig.lab', 'x', 'lab_technician', :f) RETURNING id"
            ),
            {"f": facility},
        ).scalar_one()
        conn.execute(
            text(
                "INSERT INTO admission (facility_id, code, admitted_at, source, entered_by, "
                "patient_number, patient_name) VALUES "
                "(:f, 'MIG-1', now(), 'form', :u, 'RW-1042', 'Uwase Marie'), "
                "(:f, 'MIG-2', now(), 'fhir', NULL, NULL, NULL)"
            ),
            {"f": facility, "u": user},
        )
    return facility


def test_0007_encrypts_stored_identifiers_and_downgrade_restores_them(
    field_key: crypto.Keyring, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    url = create_database(server_url_from_env())
    raw = url.render_as_string(hide_password=False)
    engine = create_engine(url)
    try:
        upgrade(raw, "0006")
        facility = _admissions_at_0006(engine)

        # Without the key the migration stops before changing anything.
        monkeypatch.setattr(crypto, "_active", None)
        monkeypatch.delenv(crypto.KEY_VARIABLE, raising=False)
        monkeypatch.setattr("robby.config.ENV_FILE", tmp_path / "missing.env")
        with pytest.raises(crypto.FieldKeyError, match="ROBBY_FIELD_KEY is not set"):
            upgrade(raw)
        monkeypatch.setattr(crypto, "_active", field_key)
        with engine.connect() as conn:
            assert conn.execute(IDENTIFIERS).all()[0][1:] == ("RW-1042", "Uwase Marie")

        upgrade(raw)
        with engine.connect() as conn:
            (_, number, name), (_, no_number, no_name) = conn.execute(IDENTIFIERS).all()
            assert isinstance(number, bytes) and b"RW-1042" not in number
            assert field_key.decrypt(number, "admission.patient_number") == "RW-1042"
            assert field_key.decrypt(name, "admission.patient_name") == "Uwase Marie"
            assert (no_number, no_name) == (None, None)
            indexes = conn.execute(
                text("SELECT patient_number_index FROM admission ORDER BY code")
            ).scalars()
            assert list(indexes) == [field_key.blind_index(facility, "RW-1042"), None]
            assert conn.execute(TRIGGER_STATE).scalar_one() == "O"  # enabled again

        downgrade(raw, "0006")
        with engine.connect() as conn:
            assert conn.execute(IDENTIFIERS).all() == [
                ("MIG-1", "RW-1042", "Uwase Marie"),
                ("MIG-2", None, None),
            ]
            assert conn.execute(TRIGGER_STATE).scalar_one() == "O"
    finally:
        engine.dispose()
        drop_database(url)
