"""`robby-admin references load|list` (mounted in robby.cli)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from robby.audit.references import ReferenceFileError, available_references, load_reference_set
from robby.config import get_settings
from robby.db import get_sessionmaker
from robby.models import ReferenceKind

references_app = typer.Typer(
    help="Reference populations (Vogel, C-Model) used by the audit and national reports.",
    no_args_is_help=True,
)


@references_app.command("load")
def references_load(
    path: Annotated[
        Path, typer.Argument(help="CSV: robson_group, group_size_pct, cs_rate_pct.", dir_okay=False)
    ],
    name: Annotated[str, typer.Option("--name", help="Display name, e.g. the paper's short name.")],
    kind: Annotated[ReferenceKind, typer.Option("--kind", help="vogel or c_model.")],
    citation: Annotated[str, typer.Option("--citation", help="Full citation of the source.")],
    population: Annotated[str, typer.Option("--population", help="Population described.")],
    version: Annotated[str, typer.Option("--version", help="Version label (unique per name).")],
) -> None:
    """Load a reference table from a CSV. Values are never built into the application."""
    with get_sessionmaker(get_settings().database_url)() as db:
        try:
            ref = load_reference_set(
                db,
                None,
                csv_bytes=path.read_bytes(),
                name=name,
                kind=kind,
                citation=citation,
                population=population,
                version_label=version,
            )
        except (ReferenceFileError, FileNotFoundError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        db.commit()
        typer.echo(f"Loaded reference set {ref.name} ({ref.version_label}) with 10 groups.")


@references_app.command("list")
def references_list() -> None:
    """List the loaded reference sets."""
    with get_sessionmaker(get_settings().database_url)() as db:
        refs = available_references(db)
    if not refs:
        typer.echo("No reference set is loaded.")
    for ref in refs:
        typer.echo(f"{ref.kind}\t{ref.name}\t{ref.version_label}\t{ref.citation}")
