"""robby-admin: administration commands (spec §7). Later sub-projects add
`model register`, `references load` and `demo load|purge`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from robson_engine import RuleSetError
from sqlalchemy.orm import Session

from robby import events
from robby.audit.cli_references import references_app
from robby.config import get_settings
from robby.db import get_sessionmaker
from robby.errors import FormErrors
from robby.migrate import upgrade
from robby.registry.dev_register import model_app
from robby.models import Role
from robby.rulesets import RuleSetConflict, load_rule_set_file
from robby.users.service import create_user

app = typer.Typer(
    help="Robson Readiness administration commands.", no_args_is_help=True, add_completion=False
)
db_app = typer.Typer(no_args_is_help=True)
rules_app = typer.Typer(no_args_is_help=True)
events_app = typer.Typer(no_args_is_help=True)
app.add_typer(db_app, name="db")
app.add_typer(rules_app, name="rules")
app.add_typer(events_app, name="events")
app.add_typer(model_app, name="model")
app.add_typer(references_app, name="references")


@db_app.callback()
def _db() -> None:
    """Database schema."""


@rules_app.callback()
def _rules() -> None:
    """Robson rule sets."""


@events_app.callback()
def _events() -> None:
    """The event log."""


def open_session() -> Session:
    return get_sessionmaker(get_settings().database_url)()


@app.command("create-system-admin")
def create_system_admin(
    username: Annotated[str, typer.Option("--username", help="Sign-in name (stored lowercase).")],
    password: Annotated[
        str,
        typer.Option(
            prompt=True, confirmation_prompt=True, hide_input=True, help="Asked for twice."
        ),
    ],
) -> None:
    """Create a system administrator (the first account of a new installation)."""
    with open_session() as db:
        try:
            created = create_user(
                db, None, username=username, role=Role.SYSTEM_ADMIN, password=password
            )
        except FormErrors as exc:
            for message in exc.errors.values():
                typer.echo(f"Error: {message}", err=True)
            raise typer.Exit(code=1) from exc
        db.commit()
        typer.echo(f"Created system administrator {created.user.username}.")


@db_app.command("upgrade")
def db_upgrade(
    revision: Annotated[str, typer.Argument(help="Target revision.")] = "head",
) -> None:
    """Apply database migrations (alembic upgrade)."""
    upgrade(get_settings().database_url, revision)
    typer.echo(f"Database upgraded to {revision}.")


@rules_app.command("load")
def rules_load(
    path: Annotated[
        Path | None,
        typer.Option(
            "--path", help="Rule set YAML (default: ROBBY_RULE_SET_PATH).", dir_okay=False
        ),
    ] = None,
) -> None:
    """Load a rule set YAML and make it the active rule set."""
    source = path or get_settings().rule_set_path
    with open_session() as db:
        try:
            outcome = load_rule_set_file(db, source)
        except (RuleSetError, RuleSetConflict, FileNotFoundError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        db.commit()
    label = outcome.rule_set.version_label
    messages = {
        "loaded": f"Loaded rule set {label} and made it active.",
        "activated": f"Rule set {label} was already loaded; it is now active.",
        "unchanged": f"Rule set {label} is already active.",
    }
    typer.echo(messages[outcome.status])


@events_app.command("verify")
def events_verify() -> None:
    """Recompute the event log's hash chain."""
    with open_session() as db:
        result = events.verify(db)
    if result.ok:
        typer.echo(f"Event log intact: {result.checked} entries checked.")
        return
    typer.echo(
        f"Event log broken at seq {result.first_bad_seq}: {result.reason}. "
        f"The {result.checked} entries before it are intact.",
        err=True,
    )
    raise typer.Exit(code=1)
