"""`robby-admin demo load|purge` (mounted in robby.cli)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from sqlalchemy.engine import make_url

from robby.config import get_settings
from robby.demo import loader

demo_app = typer.Typer(
    help="Synthetic demo database, separate from the real one.", no_args_is_help=True
)


def _target(override: str | None) -> str:
    return override or loader.default_demo_url(get_settings().database_url)


@demo_app.command("load")
def demo_load(
    database_url: Annotated[
        str | None, typer.Option("--database-url", help="Demo database (name must end in _demo).")
    ] = None,
    admissions: Annotated[int, typer.Option("--admissions", help="Number of admissions.")] = (
        loader.DEFAULT_ADMISSIONS
    ),
    seed: Annotated[int, typer.Option("--seed", help="Random seed (same seed, same demo).")] = (
        loader.DEFAULT_SEED
    ),
    force: Annotated[
        bool, typer.Option("--force", help="Drop and rebuild a demo database that has data.")
    ] = False,
    model_dir: Annotated[
        Path | None, typer.Option("--model-dir", help="Packaged model to activate in the demo.")
    ] = loader.DEFAULT_MODEL_DIR,
) -> None:
    """Create the demo database and fill it with synthetic admissions."""
    url = _target(database_url)
    try:
        s = loader.load_demo(
            url, admissions=admissions, seed=seed, force=force, model_dir=model_dir, say=typer.echo
        )
    except loader.DemoError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("")
    typer.echo(f"Admissions saved: {s.admissions}")
    typer.echo(
        "Classification: "
        + ", ".join(f"{k} {v}" for k, v in sorted(s.status.items()))
        + f"; with a prediction: {s.scored}"
    )
    typer.echo(f"Outcomes recorded: {s.outcomes}; awaiting outcome: {s.awaiting}")
    typer.echo(f"Lab requests: {s.lab_open} waiting for the lab, {s.lab_resulted} with a result")
    typer.echo(f"New patients asked from the lab: {s.asks_open} waiting")
    for name, (cs, total) in s.cs_by_facility.items():
        typer.echo(f"  {name}: {cs}/{total} cesarean ({100 * cs / total:.1f}%)")
    typer.echo(f"Active model: {s.model or 'none'}")
    verdict = "intact" if s.events_ok else "BROKEN"
    typer.echo(f"Event log {verdict}: {s.events_checked} entries")
    for note in s.notes:
        typer.echo(f"Note: {note}")
    typer.echo("")
    typer.echo(f"Sign-in accounts (password for all: {loader.DEMO_PASSWORD})")
    for username, role, where in s.credentials:
        typer.echo(f"  {username:<18} {role:<18} {where}")
    typer.echo("")
    typer.echo("Start it with: sh scripts/demo.sh")
    if not s.events_ok:
        raise typer.Exit(code=1)


@demo_app.command("purge")
def demo_purge(
    database_url: Annotated[
        str | None, typer.Option("--database-url", help="Demo database (name must end in _demo).")
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", help="Confirm dropping the demo database.")] = False,
) -> None:
    """Drop the demo database (only ever a database whose name ends in _demo)."""
    url = make_url(_target(database_url))
    try:
        name = loader.require_demo_name(url)
        if not yes:
            typer.echo(f"Error: this drops database '{name}'. Add --yes to confirm.", err=True)
            raise typer.Exit(code=1)
        if not loader.database_exists(url):
            typer.echo(f"Database {name} does not exist; nothing to do.")
            return
        loader.drop_database(url)
    except loader.DemoError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Dropped database {name}.")
