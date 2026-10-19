"""robby-admin: administration commands (spec §7). Later sub-projects add
`model register`, `references load` and `demo load|purge`."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated

import typer
from robson_engine import RuleSetError
from sqlalchemy.orm import Session

from robby import events
from robby.audit.cli_references import references_app
from robby.auth import api_tokens
from robby.config import get_settings
from robby.db import get_sessionmaker
from robby.demo.cli import demo_app
from robby.errors import FormErrors
from robby.migrate import upgrade
from robby.models import Role
from robby.registry.dev_register import model_app
from robby.rulesets import RuleSetConflict, load_rule_set_file
from robby.timeutil import utcnow
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
app.add_typer(demo_app, name="demo")
api_token_app = typer.Typer(no_args_is_help=True)
app.add_typer(api_token_app, name="api-token")


@db_app.callback()
def _db() -> None:
    """Database schema."""


@rules_app.callback()
def _rules() -> None:
    """Robson rule sets."""


@api_token_app.callback()
def _api_token() -> None:
    """Tokens for automated imports."""


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


@app.command("reconcile")
def reconcile_command() -> None:
    """Join predictions to recorded outcomes (reconciled_outcome); safe to run repeatedly."""
    from robby.monitoring.reconcile import reconcile

    with open_session() as db:
        counts = reconcile(db)
        db.commit()
    typer.echo(
        f"Reconciled {counts['reconciled']} new predictions; {counts['total']} in total, "
        f"{counts['awaiting']} awaiting an outcome."
    )


@api_token_app.command("create")
def api_token_create(
    username: Annotated[str, typer.Option("--username", help="The clinician account it acts as.")],
    label: Annotated[str, typer.Option("--label", help="What it is for, e.g. OpenClinic feed.")],
) -> None:
    """Create a token for POST /api/v1/admissions/import; it is shown once."""
    with open_session() as db:
        try:
            token, row = api_tokens.create(db, None, username, label)
        except api_tokens.TokenError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        db.commit()
        typer.echo(f"Token {row.id} for {username}. Copy it now; it cannot be shown again:")
        typer.echo(token)


@api_token_app.command("list")
def api_token_list() -> None:
    """List tokens (never the token text)."""
    with open_session() as db:
        for row in api_tokens.all_tokens(db):
            state = "revoked" if row.revoked_at else "active"
            used = row.last_used_at.isoformat(timespec="minutes") if row.last_used_at else "never"
            typer.echo(f"{row.id}  {state:8}  {row.user.username:20}  used {used}  {row.label}")


@api_token_app.command("revoke")
def api_token_revoke(
    token_id: Annotated[str, typer.Argument(help="Id from `api-token list`.")],
) -> None:
    """Stop a token from working."""
    with open_session() as db:
        try:
            api_tokens.revoke(db, None, uuid.UUID(token_id), utcnow())
        except (api_tokens.TokenError, ValueError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        db.commit()
        typer.echo("Token revoked.")
