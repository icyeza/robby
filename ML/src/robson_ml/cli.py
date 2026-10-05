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
from robson_ml.features import (
    FeatureRegistry,
    build_raw_features,
    check_coverage,
    load_feature_registry,
)
from robson_ml.ingest import file_sha256, infer_kind, inventory, read_workbook, select_sheet
from robson_ml.leakage import run_screens
from robson_ml.mapping import apply_mapping, load_mapping
from robson_ml.populations import POPULATION_VERSION, audit_population
from robson_ml.preregistration import (
    check_preregistration,
    git_commit,
    is_real_data,
    selection_rule_commit,
)
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
MLRUNS_DIR = Path("mlruns")
ARTEFACTS_DIR = Path("artefacts")
EXPLANATIONS_SUBDIR = "explanations"
FACILITY_CONTRIBUTION_CSV = "interpretation/facility_contribution.csv"
CANONICAL_ROBSON = "canonical_robson.parquet"
COMPARISON_CSV = "model_comparison.csv"
DEFAULT_SALT_PATH = Path("data/interim/mother_key.salt")
DEFAULT_FEATURES_PATH = Path("configs/features_v1.yaml")
DEFAULT_ANALYSIS_PATH = Path("configs/analysis.yaml")
CASEMIX_SUBDIR = "casemix"
MISSINGNESS_SUBDIR = "missingness"
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
    interim_dir: Path
    processed_dir: Path
    reports_dir: Path
    seed: int
    salt_path: Path = DEFAULT_SALT_PATH
    features_path: Path = DEFAULT_FEATURES_PATH
    analysis_path: Path = DEFAULT_ANALYSIS_PATH
    answers_path: Path | None = None


def load_project_config(path: Path = PROJECT_CONFIG) -> ProjectConfig:
    """Read configs/project.yaml (``salt_path``, ``features_path``, ``analysis_path`` and
    ``answers_path`` are optional)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ProjectConfig(
        raw_path=Path(data["raw_path"]),
        mapping_path=Path(data["mapping_path"]),
        interim_dir=Path(data["interim_dir"]),
        processed_dir=Path(data["processed_dir"]),
        reports_dir=Path(data["reports_dir"]),
        seed=int(data["seed"]),
        salt_path=Path(data.get("salt_path") or DEFAULT_SALT_PATH),
        features_path=Path(data.get("features_path") or DEFAULT_FEATURES_PATH),
        answers_path=Path(data["answers_path"]) if data.get("answers_path") else None,
        analysis_path=Path(data.get("analysis_path") or DEFAULT_ANALYSIS_PATH),
    )


@dataclass(frozen=True)
class AnalysisConfig:
    """Settings of the case-mix and missingness analyses (configs/analysis.yaml); no reference
    values."""

    vogel_path: Path
    cmodel_path: Path
    prevalence_path: Path
    n_boot: int
    reference_facility: str | None
    bootstrap_models: bool
    fields: list[str]
    n_draws: int
    se_ratios: list[float]


def load_analysis_config(path: Path = DEFAULT_ANALYSIS_PATH) -> AnalysisConfig:
    """Read configs/analysis.yaml."""
    from robson_ml.casemix import DEFAULT_N_BOOT
    from robson_ml.missingness import DEFAULT_MISSINGNESS_FIELDS, DEFAULT_N_DRAWS
    from robson_ml.references import CMODEL_PATH, PREVALENCE_PATH, VOGEL_PATH

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    refs = data.get("references") or {}
    casemix_cfg = data.get("casemix") or {}
    missing_cfg = data.get("missingness") or {}
    reference = casemix_cfg.get("reference_facility")
    return AnalysisConfig(
        vogel_path=Path(refs.get("vogel") or VOGEL_PATH),
        cmodel_path=Path(refs.get("cmodel") or CMODEL_PATH),
        prevalence_path=Path(refs.get("prevalence") or PREVALENCE_PATH),
        n_boot=int(casemix_cfg.get("n_boot", DEFAULT_N_BOOT)),
        reference_facility=None if reference is None else str(reference),
        bootstrap_models=bool(casemix_cfg.get("bootstrap_models", True)),
        fields=[str(f) for f in missing_cfg.get("fields") or DEFAULT_MISSINGNESS_FIELDS],
        n_draws=int(missing_cfg.get("n_draws", DEFAULT_N_DRAWS)),
        se_ratios=[float(r) for r in missing_cfg.get("se_ratios") or [1.0]],
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
    """Write the data profile outputs under reports/profile/."""
    cfg = load_project_config()
    mapping = load_mapping(cfg.mapping_path)
    raw = select_sheet(read_workbook(cfg.raw_path), mapping.sheet)
    classified = pd.read_parquet(cfg.processed_dir / "canonical_robson.parquet")
    manual: dict[str, str] = {}
    if cfg.answers_path is not None and cfg.answers_path.exists():
        loaded = yaml.safe_load(cfg.answers_path.read_text(encoding="utf-8")) or {}
        manual = {str(k): str(v) for k, v in loaded.items()}
    out_dir = cfg.reports_dir / "profile"
    write_profile(raw, classified, mapping, manual, out_dir)
    typer.echo(f"profile written to {out_dir}")


@app.command()
@guarded
def leakage() -> None:
    """Run the leakage screens over every raw column, within P_audit."""
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


@dataclass(frozen=True)
class ModelInputs:
    """Everything a model-fitting command reads, loaded after the pre-registration guard."""

    canonical: pd.DataFrame
    registry: FeatureRegistry
    raw_features: pd.DataFrame
    data_hash: str
    git_commit: str
    selection_rule_commit: str


def _model_inputs(cfg: ProjectConfig) -> ModelInputs:
    """Pre-registration guard, then load the canonical data, registry and raw features.

    Real data (anything under data/processed) is refused unless configs/selection_rule.yaml
    is committed and unmodified.
    """
    repo = Path.cwd()
    data_path = cfg.processed_dir / CANONICAL_ROBSON
    if is_real_data(data_path):
        rule_commit = check_preregistration(repo)
    else:
        rule_commit = selection_rule_commit(repo)
    registry = load_feature_registry(cfg.features_path)
    canonical = pd.read_parquet(data_path)
    mapping = load_mapping(cfg.mapping_path)
    raw = select_sheet(read_workbook(cfg.raw_path), mapping.sheet).reset_index(drop=True)
    raw_features, _ = build_raw_features(raw, registry)
    return ModelInputs(
        canonical, registry, raw_features, file_sha256(data_path), git_commit(repo), rule_commit
    )


@app.command()
@guarded
def run(config: Path) -> None:
    """Run every configuration declared in an experiment YAML.

    Refuses to touch real data (anything under data/processed) unless
    configs/selection_rule.yaml is committed and unmodified. Prints aggregate
    metrics only.
    """
    from robson_ml.evaluate import (
        RunContext,
        find_completed_run,
        load_experiments,
        run_experiment,
    )
    from robson_ml.feature_sets import (
        P_PRED_COMPLETE,
        ModelData,
        build_model_data,
        complete_cases,
    )
    from robson_ml.populations import P_PRED

    cfg = load_project_config()
    experiments = load_experiments(config)
    inputs = _model_inputs(cfg)
    rule_commit = inputs.selection_rule_commit
    canonical, registry, raw_features = inputs.canonical, inputs.registry, inputs.raw_features
    ctx = RunContext(
        tracking_uri=MLRUNS_DIR.resolve().as_uri(),
        oof_dir=cfg.interim_dir / "oof",
        data_hash=inputs.data_hash,
        features_yaml_hash=registry.sha256,
        git_commit=inputs.git_commit,
        selection_rule_commit=rule_commit,
    )
    typer.echo(
        f"configurations: {len(experiments)}; selection rule commit: {rule_commit}; "
        f"population version: {POPULATION_VERSION}"
    )
    populations: dict[str | tuple[str, str], ModelData] = {}
    for experiment in experiments:
        done = find_completed_run(experiment, ctx)
        if done is not None:
            typer.echo(f"{experiment.name} already completed (run_id={done}); skipped")
            continue
        key: str | tuple[str, str]
        if experiment.population == P_PRED_COMPLETE:
            # Complete cases depend on the feature set: one population per set.
            key = (experiment.population, experiment.feature_set)
            if key not in populations:
                if P_PRED not in populations:
                    populations[P_PRED] = build_model_data(
                        canonical, registry, raw_features, P_PRED
                    )
                populations[key] = complete_cases(populations[P_PRED], experiment.feature_set)
        else:
            key = experiment.population
            if key not in populations:
                populations[key] = build_model_data(
                    canonical, registry, raw_features, experiment.population
                )
        result = run_experiment(experiment, populations[key], ctx)
        s = result.summary
        typer.echo(
            f"{experiment.name} run_id={result.run_id} "
            f"mean_auc={s['mean_auc']:.3f} (range {s['min_auc']:.3f}-{s['max_auc']:.3f}) "
            f"pooled_auc={s['pooled_auc']:.3f} mean_slope={s['mean_calibration_slope']:.2f} "
            f"mean_citl={s['mean_calibration_in_the_large']:.2f} "
            f"pooled_brier={s['pooled_brier']:.4f}"
        )


@app.command()
@guarded
def compare() -> None:
    """Aggregate the harness runs of the current population version into
    reports/model_comparison.csv (metrics only); older runs are left out."""
    from robson_ml.evaluate import comparison_table

    cfg = load_project_config()
    table = comparison_table(MLRUNS_DIR.resolve().as_uri())
    out = cfg.reports_dir / COMPARISON_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    typer.echo(f"runs (population version {POPULATION_VERSION}): {len(table)}; written to {out}")


@app.command("fit-deploy")
@guarded
def fit_deploy(config: Path) -> None:
    """Fit the deployment model under S5 into artefacts/<version_label>/.

    Reads a deployment YAML (configs/deployment.yaml). ``use_facility: auto`` applies the
    facility rule to the finished S3 runs of the base set and its _deploy variant on the same
    data. The pre-registration guard applies. Prints aggregate facts only.
    """
    from robson_ml.deploy import (
        AUTO,
        Provenance,
        deployment_feature_choice,
        fit_deployment,
        load_deployment_config,
    )
    from robson_ml.feature_sets import build_model_data

    cfg = load_project_config()
    dcfg = load_deployment_config(config)
    inputs = _model_inputs(cfg)
    tracking_uri = MLRUNS_DIR.resolve().as_uri()
    decision = None
    if dcfg.use_facility == AUTO:
        decision = deployment_feature_choice(
            tracking_uri,
            dcfg.model,
            dcfg.feature_set,
            dcfg.missing_strategy,
            dcfg.population,
            data_hash=inputs.data_hash,
        )
        use_facility = bool(decision["use_facility"])
        typer.echo(
            f"facility decision: S3 mean log loss {decision['feature_set']}="
            f"{decision['s3_mean_log_loss']:.4f}, {decision['deploy_feature_set']}="
            f"{decision['s3_mean_log_loss_deploy']:.4f}; use_facility={use_facility}"
        )
    else:
        use_facility = bool(dcfg.use_facility)
        typer.echo(f"use_facility={use_facility} (set in the config, not by the facility rule)")
    data = build_model_data(inputs.canonical, inputs.registry, inputs.raw_features, dcfg.population)
    provenance = Provenance(
        tracking_uri=tracking_uri,
        data_hash=inputs.data_hash,
        features_yaml_hash=inputs.registry.sha256,
        git_commit=inputs.git_commit,
        selection_rule_commit=inputs.selection_rule_commit,
    )
    result = fit_deployment(dcfg, data, provenance, ARTEFACTS_DIR, use_facility, decision)
    s = result.summary
    typer.echo(
        f"{dcfg.version_label}: {s['algorithm']} {s['feature_set']} {dcfg.missing_strategy} "
        f"{dcfg.population}; calibration={s['calibration']['method']}; "
        f"fit rows={s['n_fit_rows']}; calibration rows={s['n_calibration_rows']}; "
        f"selection rule commit: {s['selection_rule_commit']}; run_id={result.run_id}"
    )
    typer.echo(f"artefact written to {result.out_dir}")
    if use_facility:
        out = cfg.reports_dir / FACILITY_CONTRIBUTION_CSV
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(s["facility_contribution"]).to_csv(out, index=False)
        typer.echo(f"facility contribution written to {out}")


@app.command("explain-local")
@guarded
def explain_local(run_id: str) -> None:
    """Local explanations for the 20 highest-error S1 out-of-fold cases of a run.

    Written only to data/interim/explanations/<run_id>_local.parquet (row level; never to
    reports/). The pre-registration guard applies. Prints counts only.
    """
    from robson_ml.explain import CONTRIBUTION_PREFIX, LOCAL_N, local_explanations
    from robson_ml.feature_sets import build_model_data
    from robson_ml.select import experiment_config, load_runs, run_fold_params

    cfg = load_project_config()
    tracking_uri = MLRUNS_DIR.resolve().as_uri()
    runs = load_runs(tracking_uri, latest_only=False)
    match = runs[runs["run_id"] == run_id]
    if match.empty:
        raise LookupError("no finished harness run of the current population version has that id")
    config = experiment_config(match.iloc[0])
    inputs = _model_inputs(cfg)
    data = build_model_data(
        inputs.canonical, inputs.registry, inputs.raw_features, config.population
    )
    table = local_explanations(data, config, run_fold_params(tracking_uri, run_id), n=LOCAL_N)
    out_dir = cfg.interim_dir / EXPLANATIONS_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{run_id}_local.parquet"
    table.to_parquet(out, index=False)
    n_features = sum(str(c).startswith(CONTRIBUTION_PREFIX) for c in table.columns)
    typer.echo(
        f"local explanations: cases={len(table)}; input features={n_features}; "
        f"written to {out} (row level: keep under data/interim, never in reports/)"
    )


R = TypeVar("R")


def _load_reference(loader: Callable[[Path], R], path: Path) -> tuple[R | None, str]:
    """``(reference, "")``, or ``(None, reason)`` when the file is absent. Nothing is ever
    put in its place: the analyses report the comparison as not applicable."""
    from robson_ml.references import ReferenceDataMissing

    try:
        return loader(path), ""
    except ReferenceDataMissing:
        return None, f"reference file absent: {path} (schema in robson_ml.references)"


def _analysis_frames(cfg: ProjectConfig, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The classified canonical frame and P_audit, with any of ``columns`` that are not
    canonical but are feature-registry names built from the raw export (row-aligned)."""
    classified = pd.read_parquet(cfg.processed_dir / CANONICAL_ROBSON)
    wanted = [c for c in dict.fromkeys(columns) if c not in classified.columns]
    if wanted and cfg.features_path.exists():
        registry = load_feature_registry(cfg.features_path)
        names = {entry.name for entry in registry.entries}
        wanted = [c for c in wanted if c in names]
        if wanted:
            mapping = load_mapping(cfg.mapping_path)
            raw = select_sheet(read_workbook(cfg.raw_path), mapping.sheet).reset_index(drop=True)
            raw_features, _ = build_raw_features(raw, registry)
            if len(raw_features) != len(classified):
                raise ValueError("raw export and canonical data are not row-aligned")
            classified = classified.join(raw_features[wanted].reset_index(drop=True))
    audit, _ = audit_population(classified)
    return classified, audit


@app.command()
@guarded
def casemix() -> None:
    """Robson audit table and RQ1 case-mix analysis into
    reports/casemix/. Vogel and C-Model comparisons run only when their reference files
    exist (never invented). Prints aggregate statements only."""
    from robson_ml.audit_offline import audit_markdown, published_audit_table
    from robson_ml.casemix import SECTOR_CONFOUND_STATEMENT, casemix_analysis
    from robson_ml.references import load_cmodel, load_vogel

    cfg = load_project_config()
    acfg = load_analysis_config(cfg.analysis_path)
    vogel, vogel_reason = _load_reference(load_vogel, acfg.vogel_path)
    cmodel, cmodel_reason = _load_reference(load_cmodel, acfg.cmodel_path)
    classified, audit = _analysis_frames(cfg, cmodel.columns() if cmodel else [])
    out = cfg.reports_dir / CASEMIX_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    table = published_audit_table(classified, vogel)
    table.to_csv(out / "robson_audit_table.csv", index=False)
    (out / "robson_audit_table.md").write_text(audit_markdown(table, vogel), encoding="utf-8")
    result = casemix_analysis(
        audit,
        vogel,
        cmodel,
        n_boot=acfg.n_boot,
        seed=cfg.seed,
        reference_facility=acfg.reference_facility,
        bootstrap_models=acfg.bootstrap_models,
        vogel_absent_reason=vogel_reason or "reference file absent",
        cmodel_absent_reason=cmodel_reason or "reference file absent",
    )
    result.write(out)
    typer.echo(f"P_audit rows: {fmt_count(len(audit))}")
    typer.echo(f"Vogel comparison: {result.vogel_status}")
    typer.echo(f"C-Model: {result.cmodel_status}")
    typer.echo(f"bootstrap: {result.n_boot} resamples ({result.n_boot_effective} fully fitted)")
    typer.echo(SECTOR_CONFOUND_STATEMENT)
    typer.echo(f"case-mix outputs written to {out}")


@app.command()
@guarded
def missingness() -> None:
    """RQ2: missingness description and models, and the under-recording
    sensitivity analysis when the prevalence reference exists, into reports/missingness/."""
    from robson_ml.missingness import MAR_STATEMENT, missingness_report
    from robson_ml.references import load_prevalence

    cfg = load_project_config()
    acfg = load_analysis_config(cfg.analysis_path)
    prevalence, reason = _load_reference(load_prevalence, acfg.prevalence_path)
    _, audit = _analysis_frames(cfg, acfg.fields)
    priors = None
    if prevalence is not None:
        priors = {n: (c.field, c.grid_pct) for n, c in prevalence.conditions.items()}
    report = missingness_report(
        audit,
        acfg.fields,
        priors,
        n_draws=acfg.n_draws,
        seed=cfg.seed,
        se_ratios=acfg.se_ratios,
        priors_absent_reason=reason or "prevalence reference file absent",
    )
    out = cfg.reports_dir / MISSINGNESS_SUBDIR
    report.write(out)
    summary = report.model_summary
    rejected = summary.loc[summary["mcar_rejected"].eq(True), "field"].tolist()
    typer.echo(f"fields described: {len(report.fields)}; MCAR rejected for: {rejected}")
    typer.echo(MAR_STATEMENT)
    typer.echo(f"under-recording: {report.under_recording_status}")
    for result in report.under_recording:
        for row in result.tipping.itertuples(index=False):
            typer.echo(f"  {result.condition} [{row.conclusion}]: {row.statement}")
    typer.echo(f"missingness outputs written to {out}")


if __name__ == "__main__":
    app()
