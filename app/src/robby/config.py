"""Application settings, read from ROBBY_* environment variables or app/.env."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = APP_ROOT / ".env"
DEFAULT_RULE_SET_FILE = "robson_v1.0.yaml"


def default_rule_set_path() -> Path:
    """The rule set YAML shipped inside the robson_engine package."""
    return Path(str(resources.files("robson_engine.rules").joinpath(DEFAULT_RULE_SET_FILE)))


class Settings(BaseSettings):
    """Every setting the application reads (spec §2)."""

    model_config = SettingsConfigDict(
        env_prefix="ROBBY_", env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str
    secret_key: SecretStr
    artefact_dir: Path = Path("artefacts")
    rule_set_path: Path = Field(default_factory=default_rule_set_path)
    session_idle_minutes: int = Field(default=30, ge=1)
    session_absolute_hours: int = Field(default=12, ge=1)
    secure_cookies: bool = True
    timezone: str = "Africa/Kigali"
    demo: bool = False

    @field_validator("database_url")
    @classmethod
    def _psycopg_url(cls, value: str) -> str:
        if not value.startswith("postgresql+psycopg://"):
            raise ValueError("must be a postgresql+psycopg:// URL")
        return value

    @field_validator("secret_key")
    @classmethod
    def _long_secret(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value().encode("utf-8")) < 32:
            raise ValueError("must be at least 32 bytes")
        return value

    @field_validator("timezone")
    @classmethod
    def _known_zone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError(f"unknown time zone: {value}") from None
        return value

    @property
    def zone(self) -> ZoneInfo:
        """The time zone used to show times to people."""
        return ZoneInfo(self.timezone)

    @property
    def secret_bytes(self) -> bytes:
        """The secret key as bytes, for HMAC."""
        return self.secret_key.get_secret_value().encode("utf-8")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Settings for the running process (environment first, then app/.env)."""
    return Settings()  # type: ignore[call-arg]
