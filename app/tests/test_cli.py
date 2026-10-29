"""robby-admin: create-system-admin, db upgrade, rules load, events verify."""

import json
from pathlib import Path

import pytest
from robson_engine import load_rule_set
from robson_engine.ruleset import stamp
from sqlalchemy import Connection, select, text
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from robby import cli, events
from robby.auth.passwords import verify_password
from robby.config import Settings
from robby.models import AppUser, Role, RuleSet
from tests.dbutil import bound_session
from tests.factories import events_of, make_user

runner = CliRunner()
DEFAULT_GROUP_1 = 'name: "Nulliparous, single cephalic, >=37 weeks, spontaneous labour"'


@pytest.fixture(autouse=True)
def cli_in_test_transaction(
    monkeypatch: pytest.MonkeyPatch, connection: Connection, settings: Settings
) -> None:
    """CLI sessions join the test transaction; settings point at the test database."""
    monkeypatch.setattr(cli, "open_session", lambda: bound_session(connection))
    monkeypatch.setattr(cli, "get_settings", lambda: settings)


def rule_set_copy(settings: Settings, tmp_path: Path, old: str, new: str) -> Path:
    source = settings.rule_set_path.read_text(encoding="utf-8")
    assert old in source
    path = tmp_path / "rules.yaml"
    path.write_text(source.replace(old, new), encoding="utf-8")
    stamp(path)
    return path


def test_help_lists_every_command() -> None:
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    for name in ("create-system-admin", "db", "rules", "events"):
        assert name in result.output


def test_create_system_admin(db: Session) -> None:
    result = runner.invoke(
        cli.app,
        ["create-system-admin", "--username", "Root.Admin"],
        input="a-strong-password\na-strong-password\n",
    )
    assert result.exit_code == 0, result.output
    assert "Created system administrator root.admin." in result.output
    user = db.scalar(select(AppUser).where(AppUser.username == "root.admin"))
    assert user is not None and user.role is Role.SYSTEM_ADMIN and user.facility_id is None
    assert user.must_change_password is False
    assert verify_password(user.password_hash, "a-strong-password")
    [event] = events_of(db, "user.created")
    assert event.actor_id is None


def test_create_system_admin_needs_matching_passwords(db: Session) -> None:
    result = runner.invoke(
        cli.app,
        ["create-system-admin", "--username", "root.admin"],
        input="a-strong-password\nanother-password\n",
    )
    assert result.exit_code != 0
    assert db.scalar(select(AppUser).where(AppUser.username == "root.admin")) is None


def test_create_system_admin_refuses_a_short_password(db: Session) -> None:
    result = runner.invoke(
        cli.app, ["create-system-admin", "--username", "root.admin"], input="short\nshort\n"
    )
    assert result.exit_code == 1
    assert "Use at least 10 characters." in result.output


def test_create_system_admin_refuses_a_taken_username(db: Session) -> None:
    make_user(db, role=Role.SYSTEM_ADMIN, username="root.admin")
    db.commit()
    result = runner.invoke(
        cli.app,
        ["create-system-admin", "--username", "root.admin"],
        input="a-strong-password\na-strong-password\n",
    )
    assert result.exit_code == 1
    assert "This username is already taken." in result.output


def test_db_upgrade_on_a_current_database_is_a_no_op() -> None:
    result = runner.invoke(cli.app, ["db", "upgrade"])
    assert result.exit_code == 0, result.output
    assert "Database upgraded to head." in result.output


def test_rules_load_the_packaged_rule_set_once(db: Session, settings: Settings) -> None:
    result = runner.invoke(cli.app, ["rules", "load"])
    assert result.exit_code == 0, result.output
    assert "Loaded rule set robson-v1.0 and made it active." in result.output
    row = db.scalar(select(RuleSet).where(RuleSet.version_label == "robson-v1.0"))
    assert row is not None and row.is_active
    assert row.checksum == load_rule_set().checksum
    assert row.definition["version_label"] == "robson-v1.0"
    again = runner.invoke(cli.app, ["rules", "load"])
    assert again.exit_code == 0
    assert "Rule set robson-v1.0 is already active." in again.output
    [event] = events_of(db, "rule_set.loaded")
    assert event.payload == {"version_label": "robson-v1.0", "checksum": row.checksum}


def test_rules_load_a_new_version_replaces_the_active_one(
    db: Session, settings: Settings, tmp_path: Path
) -> None:
    runner.invoke(cli.app, ["rules", "load"])
    newer = rule_set_copy(
        settings, tmp_path, 'version_label: "robson-v1.0"', 'version_label: "robson-v9.9"'
    )
    result = runner.invoke(cli.app, ["rules", "load", "--path", str(newer)])
    assert result.exit_code == 0, result.output
    active = db.scalars(select(RuleSet).where(RuleSet.is_active)).all()
    assert [r.version_label for r in active] == ["robson-v9.9"]
    back = runner.invoke(cli.app, ["rules", "load"])
    assert "Rule set robson-v1.0 was already loaded; it is now active." in back.output
    assert len(events_of(db, "rule_set.activated")) == 1


def test_rules_load_refuses_changed_content_under_an_old_label(
    db: Session, settings: Settings, tmp_path: Path
) -> None:
    runner.invoke(cli.app, ["rules", "load"])
    changed = rule_set_copy(settings, tmp_path, DEFAULT_GROUP_1, 'name: "Group one, renamed"')
    result = runner.invoke(cli.app, ["rules", "load", "--path", str(changed)])
    assert result.exit_code == 1
    assert "already loaded with a different checksum" in result.output


def test_rules_load_refuses_a_file_whose_checksum_is_stale(
    settings: Settings, tmp_path: Path
) -> None:
    source = settings.rule_set_path.read_text(encoding="utf-8")
    path = tmp_path / "stale.yaml"
    path.write_text(
        source.replace(DEFAULT_GROUP_1, 'name: "Edited without re-stamping"'), encoding="utf-8"
    )
    result = runner.invoke(cli.app, ["rules", "load", "--path", str(path)])
    assert result.exit_code == 1
    assert "checksum mismatch" in result.output


def test_events_verify_an_intact_log(db: Session) -> None:
    events.append(db, None, "test.a")
    events.append(db, None, "test.b")
    db.commit()
    result = runner.invoke(cli.app, ["events", "verify"])
    assert result.exit_code == 0
    assert "Event log intact: 2 entries checked." in result.output


def test_events_verify_reports_tampering(db: Session) -> None:
    first = events.append(db, None, "test.a", payload={"n": 1})
    events.append(db, None, "test.b")
    db.execute(text("ALTER TABLE event_log DISABLE TRIGGER event_log_immutable"))
    db.execute(
        text("UPDATE event_log SET payload = CAST(:p AS jsonb) WHERE seq = :seq"),
        {"p": json.dumps({"n": 2}), "seq": first.seq},
    )
    db.execute(text("ALTER TABLE event_log ENABLE TRIGGER event_log_immutable"))
    db.commit()
    result = runner.invoke(cli.app, ["events", "verify"])
    assert result.exit_code == 1
    assert f"Event log broken at seq {first.seq}" in result.output
