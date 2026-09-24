"""Command-line entry point. Every command runs from configs/project.yaml; no ad-hoc arguments."""

from __future__ import annotations

import functools
import json
import os
import secrets
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar

import pandas as pd
import typer
import yaml

from robson_engine import load_rule_set
from robson_ml.features import check_coverage, load_feature_registry
from robson_ml.ingest import file_sha256, infer_kind, inventory, read_workbook, select_sheet
from robson_ml.leakage import run_screens
from robson_ml.mapping import apply_mapping, load_mapping
from robson_ml.populations import audit_population
from robson_ml.privacy import fmt_count
from robson_ml.profile import write_profile
from robson_ml.robson_run import classify_frame, handcheck_sample, validate_classification
from robson_ml.schema import validate_canonical

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
    pretty_exceptions_enable=False,
)
PROJECT_CONFIG = Path("configs/project.yaml")
DEFAULT_SALT_PATH = Path("data/interim/mother_key.salt")
DEFAULT_FEATURES_PATH = Path("configs/features_v1.yaml")
SALT_BYTES = 32
# infer_kind results that a leakage screen cannot meaningfully run on.
UNSCREENABLE_KINDS = frozenset({"empty", "text", "datetime"})

F = TypeVar("F", bound=Callable[..., None])


def guarded(func: F) -> F:
    """Catch any exception from a command body and refuse to print its details.

    Patient-derived values can end up inside exception messages (bad paths, bad
    cell contents, etc.). On a real terminal typer/rich would otherwise print the
    exception message and source lines for any uncaught exception. This decorator
    ensures only a value-free, generic message reaches stderr, unless the
    ROBSON_ML_DEBUG=1 escape hatch is set for local debugging in a private session.
    """

    @functools.wraps(func)
    def wrapper(*args: object, **kwargs: object) -> None:
        try:
            func(*args, **kwargs)
        except (typer.Exit, KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            if os.environ.get("ROBSON_ML_DEBUG") == "1":
                raise
            typer.echo(
                f"error: {type(exc).__name__} in '{func.__name__}' "
                "(details suppressed to protect patient data; rerun with "
                "ROBSON_ML_DEBUG=1 in a private session to see them)",
                err=True,
            )
            raise typer.Exit(code=1) from None

    return wrapper  # type: ignore[return-value]


@dataclass(frozen=True)
class ProjectConfig:
    """Paths and seed shared by all commands."""

    raw_path: Path
    mapping_path: Path
    answers_path: Path
    interim_dir: Path
    processed_dir: Path
    reports_dir: Path
    seed: int
    salt_path: Path = DEFAULT_SALT_PATH
    features_path: Path = DEFAULT_FEATURES_PATH


def load_project_config(path: Path = PROJECT_CONFIG) -> ProjectConfig:
    """Read configs/project.yaml (``salt_path``, ``features_path`` are optional)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ProjectConfig(
        raw_path=Path(data["raw_path"]),
        mapping_path=Path(data["mapping_path"]),
        answers_path=Path(data["answers_path"]),
        interim_dir=Path(data["interim_dir"]),
        processed_dir=Path(data["processed_dir"]),
        reports_dir=Path(data["reports_dir"]),
        seed=int(data["seed"]),
        salt_path=Path(data.get("salt_path") or DEFAULT_SALT_PATH),
        features_path=Path(data.get("features_path") or DEFAULT_FEATURES_PATH),
    )


def load_or_create_salt(path: Path) -> bytes:
    """The mother_key salt from ``path``, created there (32 random bytes) if absent.

    An existing file is never replaced, whatever it holds: a new salt would silently change
    every mother_key. apply_mapping rejects a salt that is too short. The salt is never
    printed.
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        fd = os.open(path, flags, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(secrets.token_bytes(SALT_BYTES))
    return path.read_bytes()


@app.command()
@guarded
def ingest() -> None:
    """Inventory the raw export; if the mapping has fields, build the canonical dataset."""
    cfg = load_project_config()
    sheets = read_workbook(cfg.raw_path)
    inv = inventory(sheets)
    inv["data_sha256"] = file_sha256(cfg.raw_path)
    cfg.interim_dir.mkdir(parents=True, exist_ok=True)
    (cfg.interim_dir / "raw_inventory.json").write_text(json.dumps(inv, indent=2), encoding="utf-8")
    typer.echo(f"raw sha256: {inv['data_sha256']}")
    for i, sheet in enumerate(inv["sheets"], start=1):
        kinds = Counter(c["kind"] for c in sheet["columns"])
        versions = len(sheet.get("versions", {}))
        typer.echo(
            f"sheet {i}: rows={fmt_count(sheet['n_rows'])} cols={sheet['n_cols']} "
            f"versions={versions} kinds={dict(kinds)}"
        )

    mapping = load_mapping(cfg.mapping_path)
    if not mapping.fields:
        typer.echo("mapping has no fields yet: inventory only")
        return
    salt = None
    if any(spec.kind == "hash_key" for spec in mapping.fields.values()):
        salt = load_or_create_salt(cfg.salt_path)
    canonical, report = apply_mapping(select_sheet(sheets, mapping.sheet), mapping, salt=salt)
    validate_canonical(canonical)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    canonical.to_parquet(cfg.processed_dir / "canonical.parquet", index=False)
    (cfg.interim_dir / "mapping_report.json").write_text(
        json.dumps(asdict(report), indent=2, default=str), encoding="utf-8"
    )
    typer.echo(f"canonical rows: {fmt_count(len(canonical))}")
    for f in report.fields:
        dropped = f" time_dropped={fmt_count(f.n_time_dropped)}" if f.kind == "datetime" else ""
        typer.echo(
            f"  {f.canonical:<24} {f.status:<9} mapped={fmt_count(f.n_mapped)} "
            f"explicit_missing={fmt_count(f.n_explicit_missing)} "
            f"unparsed={fmt_count(f.n_unparsed)} out_of_range={fmt_count(f.n_out_of_range)}"
            f"{dropped}"
        )
    typer.echo(f"canonical fields with no mapping: {report.missing_canonical_fields}")
    typer.echo(f"raw columns not used by the mapping: {len(report.unreferenced_raw_columns)}")
    typer.echo(f"fields needing review: {report.review_fields}")


@app.command()
@guarded
def robson() -> None:
    """Run the Robson engine over the canonical dataset and export the hand-check list."""
    cfg = load_project_config()
    canonical = pd.read_parquet(cfg.processed_dir / "canonical.parquet")
    classified = classify_frame(canonical, load_rule_set())
    classified.to_parquet(cfg.processed_dir / "canonical_robson.parquet", index=False)
    cfg.interim_dir.mkdir(parents=True, exist_ok=True)
    handcheck = handcheck_sample(classified, cfg.seed)
    handcheck.to_csv(cfg.interim_dir / "robson_handcheck.csv", index=False)
    summary = validate_classification(classified)
    typer.echo(f"records: {fmt_count(summary.n_total)}")
    typer.echo(
        "status: "
        + ", ".join(f"{k}={fmt_count(v)}" for k, v in sorted(summary.status_counts.items()))
    )
    typer.echo(
        "groups: "
        + ", ".join(f"{k}={fmt_count(v)}" for k, v in sorted(summary.group_counts.items()))
    )
    typer.echo(
        f"complete (precise) inputs: {fmt_count(summary.n_complete_inputs)}; "
        f"coarse inputs: {fmt_count(summary.n_coarse_inputs)}; "
        f"reconciles: {summary.reconciles}; hand-check rows: {fmt_count(len(handcheck))}"
    )
    summary.assert_valid()


@app.command()
@guarded
def profile() -> None:
    """Write the spec §7 profile outputs under reports/profile/."""
    cfg = load_project_config()
    mapping = load_mapping(cfg.mapping_path)
    raw = select_sheet(read_workbook(cfg.raw_path), mapping.sheet)
    classified = pd.read_parquet(cfg.processed_dir / "canonical_robson.parquet")
    manual: dict[str, str] = {}
    if cfg.answers_path.exists():
        loaded = yaml.safe_load(cfg.answers_path.read_text(encoding="utf-8")) or {}
        manual = {str(k): str(v) for k, v in loaded.items()}
    out_dir = cfg.reports_dir / "profile"
    write_profile(raw, classified, mapping, manual, out_dir)
    typer.echo(f"profile written to {out_dir}")


@app.command()
@guarded
def leakage() -> None:
    """Run the leakage screens (spec §8.2) over every raw column, within P_audit."""
    cfg = load_project_config()
    mapping = load_mapping(cfg.mapping_path)
    raw = select_sheet(read_workbook(cfg.raw_path), mapping.sheet).reset_index(drop=True)
    canonical = pd.read_parquet(cfg.processed_dir / "canonical_robson.parquet")
    audit, _ = audit_population(canonical)
    raw_audit = raw.loc[audit.index].reset_index(drop=True)

    frame = pd.DataFrame(
        {
            "cs": audit["cs"].reset_index(drop=True),
            "facility_id": audit["facility_id"].reset_index(drop=True),
        }
    )
    candidates: list[str] = []
    n_skipped = 0
    for column in raw.columns:
        kind = infer_kind(raw[column])
        if kind in UNSCREENABLE_KINDS:
            n_skipped += 1
            continue
        series = raw_audit[column]
        if kind == "numeric":
            series = pd.to_numeric(series, errors="coerce")
        frame[str(column)] = series.to_numpy()
        candidates.append(str(column))

    typer.echo(
        f"raw columns: {fmt_count(raw.shape[1])}; screened: {fmt_count(len(candidates))}; "
        f"skipped (empty/text/datetime): {fmt_count(n_skipped)}"
    )
    if not candidates:
        typer.echo("no screenable raw columns; nothing written")
        return

    out_path = cfg.reports_dir / "leakage" / "leakage_screens.csv"
    table = run_screens(frame, candidates, out_path)
    typer.echo(f"leakage screens written to {out_path}")
    typer.echo(
        "flagged: "
        f"auc={fmt_count(int(table['auc_flag'].eq(True).sum()))} "
        f"name={fmt_count(int(table['name_flag'].eq(True).sum()))} "
        f"completeness={fmt_count(int(table['completeness_flag'].eq(True).sum()))} "
        f"any={fmt_count(int(table['flagged'].eq(True).sum()))}"
    )

    if not cfg.features_path.exists():
        typer.echo(f"feature registry not found at {cfg.features_path}; screens run without it")
        return
    registry = load_feature_registry(cfg.features_path)
    missing, extra = check_coverage(registry, [str(c) for c in raw.columns])
    typer.echo(
        f"feature registry sha256: {registry.sha256}; entries: {fmt_count(len(registry.entries))}; "
        f"included: {fmt_count(len(registry.included()))}"
    )
    typer.echo(f"raw columns missing from registry: {fmt_count(len(missing))}")
    typer.echo(f"registry columns not in raw export: {fmt_count(len(extra))}")


if __name__ == "__main__":
    app()
