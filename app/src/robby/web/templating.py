"""The Jinja2 environment: templates directory, filters and globals."""

from __future__ import annotations

from datetime import datetime
from functools import partial
from pathlib import Path
from zoneinfo import ZoneInfo

from jinja2 import Environment, FileSystemLoader, select_autoescape
from starlette.templating import Jinja2Templates

from robby.config import Settings
from robby.models import role_label

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


def format_local(value: datetime | None, zone: ZoneInfo) -> str:
    """'05 Oct 2026, 14:20' in the configured zone, as the mockups write times."""
    if value is None:
        return "Never"
    return value.astimezone(zone).strftime("%d %b %Y, %H:%M")


def format_local_short(value: datetime | None, zone: ZoneInfo) -> str:
    """'05 Oct, 14:20' for list rows."""
    return "" if value is None else value.astimezone(zone).strftime("%d %b, %H:%M")


def build_templates(settings: Settings) -> Jinja2Templates:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["local_dt"] = partial(format_local, zone=settings.zone)
    env.filters["local_dt_short"] = partial(format_local_short, zone=settings.zone)
    env.globals["role_label"] = role_label
    return Jinja2Templates(env=env)
