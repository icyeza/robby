"""Settings come from ROBBY_* variables or app/.env and are validated."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from robby.config import Settings

URL = "postgresql+psycopg://robby:pw@localhost:5432/robby"
SECRET = "s" * 32
FIELD_KEY = "A" * 43  # base64url of 32 zero bytes
VARIABLES = [
    "ROBBY_DATABASE_URL",
    "ROBBY_SECRET_KEY",
    "ROBBY_FIELD_KEY",
    "ROBBY_ARTEFACT_DIR",
    "ROBBY_RULE_SET_PATH",
    "ROBBY_SESSION_IDLE_MINUTES",
    "ROBBY_SESSION_ABSOLUTE_HOURS",
    "ROBBY_SECURE_COOKIES",
    "ROBBY_TIMEZONE",
    "ROBBY_DEMO",
]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in VARIABLES:
        monkeypatch.delenv(name, raising=False)


def make(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "database_url": URL,
        "secret_key": SECRET,
        "field_key": FIELD_KEY,
        **overrides,
    }
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def test_reads_prefixed_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROBBY_DATABASE_URL", URL)
    monkeypatch.setenv("ROBBY_SECRET_KEY", SECRET)
    monkeypatch.setenv("ROBBY_FIELD_KEY", FIELD_KEY)
    monkeypatch.setenv("ROBBY_DEMO", "true")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.database_url == URL
    assert settings.secret_key.get_secret_value() == SECRET
    assert settings.demo is True


def test_reads_an_env_file(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        f"ROBBY_DATABASE_URL={URL}\nROBBY_SECRET_KEY={SECRET}\nROBBY_FIELD_KEY={FIELD_KEY}\n"
        "ROBBY_SECURE_COOKIES=false\n",
        encoding="utf-8",
    )
    settings = Settings(_env_file=env)  # type: ignore[call-arg]
    assert settings.secure_cookies is False


def test_defaults_match_the_spec() -> None:
    settings = make()
    assert settings.session_idle_minutes == 30
    assert settings.session_absolute_hours == 12
    assert settings.secure_cookies is True
    assert settings.timezone == "Africa/Kigali"
    assert settings.demo is False
    assert settings.artefact_dir == Path("artefacts")
    assert settings.rule_set_path.name == "robson_v1.0.yaml"
    assert settings.rule_set_path.is_file()


def test_secret_key_must_be_at_least_32_bytes() -> None:
    with pytest.raises(ValidationError, match="at least 32 bytes"):
        make(secret_key="too-short")


def test_field_key_is_required() -> None:
    with pytest.raises(ValidationError, match="field_key"):
        Settings(_env_file=None, database_url=URL, secret_key=SECRET)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("value", "message"),
    [("A" * 42, "32 bytes"), ("A" * 44, "32 bytes"), ("not base64!", "base64url")],
)
def test_field_key_must_be_32_base64url_bytes(value: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        make(field_key=value)


def test_database_url_must_use_psycopg() -> None:
    with pytest.raises(ValidationError, match=r"postgresql\+psycopg"):
        make(database_url="postgresql://robby@localhost/robby")


def test_timezone_must_exist() -> None:
    with pytest.raises(ValidationError, match="unknown time zone"):
        make(timezone="Mars/Olympus_Mons")


def test_helpers() -> None:
    settings = make()
    assert settings.secret_bytes == SECRET.encode("utf-8")
    assert settings.zone.key == "Africa/Kigali"
    assert settings.field_key_bytes == bytes(32)
