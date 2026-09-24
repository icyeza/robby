"""Pre-registered model selection (spec §13.3) and read-only access to tracked runs.

:func:`load_runs` reads the harness runs of the current population version from the local
MLflow store (params, tags and metrics; never artefacts with rows). :func:`select_configuration`
applies ``configs/selection_rule.yaml`` mechanically:

1. eligible: split S1, population ``P_pred``, feature set FS0-FS4, not a baseline, mean
   LOHO calibration slope in the pre-registered interval;
2. rank eligible configurations by mean LOHO AUC;
3. tie-break: among those within ``within_auc`` of the best, the lowest complexity rank
   (then the higher mean AUC, then the name, for determinism);
4. baseline check: does the selection's mean LOHO AUC exceed B1's (same split and
   population)?
5. no eligible configuration: the one whose mean slope is closest to 1.0, reported as a
   limitation.

The outcome is a mean LOHO AUC of predicted CS under current practice; nothing here ranks
anything as a recommendation for or against a mode of delivery.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from robson_ml.evaluate import EXPERIMENT_NAME, ExperimentConfig, metric_key
from robson_ml.populations import POPULATION_VERSION

BASELINE_MODELS = frozenset({"B0", "B1", "B2", "B3"})
CONFIG_COLUMNS = ("model", "feature_set", "missing_strategy", "split", "population")
PARAM_COLUMNS = (
    *CONFIG_COLUMNS,
    "family",
    "complexity_rank",
    "n_trials",
    "seed",
    "n_boot",
    "n_features",
)
TAG_COLUMNS = ("population_version", "selection_rule_commit", "git_commit", "data_hash")
FOLD_PARAMS_SUFFIX = "_params"
NO_ADDED_DISCRIMINATION = "the model does not add discrimination beyond the Robson classification"


def config_name(row: pd.Series | dict[str, Any]) -> str:
    """``model|feature_set|missing_strategy|split|population`` of a run row."""
    return "|".join(str(row[c]) for c in CONFIG_COLUMNS)


def load_runs(
    tracking_uri: str, population_version: str = POPULATION_VERSION, latest_only: bool = True
) -> pd.DataFrame:
    """Finished harness runs of ``population_version``: one row per run, with the
    configuration params, provenance tags and every logged metric (by metric name).

    With ``latest_only`` a configuration run more than once keeps only its latest run.
    Runs without the population-version tag predate spec v1.3 and are left out.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        return pd.DataFrame(columns=["run_id", "config", *PARAM_COLUMNS, *TAG_COLUMNS])
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string="attributes.status = 'FINISHED'",
        output_format="pandas",
    )
    assert isinstance(runs, pd.DataFrame)
    version = runs.get("tags.population_version")
    if version is None:
        return pd.DataFrame(columns=["run_id", "config", *PARAM_COLUMNS, *TAG_COLUMNS])
    runs = runs[version == population_version]
    out = pd.DataFrame({"run_id": runs["run_id"].to_numpy()})
    for column in PARAM_COLUMNS:
        out[column] = runs.get(f"params.{column}", pd.Series(index=runs.index)).to_numpy()
    for column in TAG_COLUMNS:
        out[column] = runs.get(f"tags.{column}", pd.Series(index=runs.index)).to_numpy()
    out["start_time"] = runs["start_time"].to_numpy()
    metrics = runs[[c for c in runs.columns if c.startswith("metrics.")]]
    metrics.columns = [c.removeprefix("metrics.") for c in metrics.columns]
    out = pd.concat([out, metrics.reset_index(drop=True)], axis=1)
    for column in ("complexity_rank", "n_trials", "seed", "n_boot", "n_features"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out.insert(1, "config", [config_name(r) for _, r in out.iterrows()])
    if latest_only:
        out = out.sort_values("start_time").drop_duplicates("config", keep="last")
    return out.sort_values(["split", "model", "feature_set", "missing_strategy"]).reset_index(
        drop=True
    )


def experiment_config(row: pd.Series) -> ExperimentConfig:
    """The :class:`ExperimentConfig` a run row was produced by."""
    return ExperimentConfig(
        model=str(row["model"]),
        feature_set=str(row["feature_set"]),
        missing_strategy=str(row["missing_strategy"]),
        split=str(row["split"]),
        population=str(row["population"]),
        n_trials=int(row["n_trials"]),
        seed=int(row["seed"]),
        n_boot=int(row["n_boot"]),
    )


def fold_metrics(
    runs: pd.DataFrame, fold_names: Sequence[str], metric: str = "auc"
) -> pd.DataFrame:
    """Per-fold ``metric`` of each run (rows: run config; columns: fold names)."""
    table = pd.DataFrame(index=runs["config"])
    for fold in fold_names:
        column = metric_key(f"{fold}_{metric}")
        table[fold] = runs[column].to_numpy() if column in runs.columns else float("nan")
    return table


def run_fold_params(tracking_uri: str, run_id: str) -> dict[str, dict[str, Any]]:
    """The tuned hyperparameters of each fold of a run, keyed by (sanitised) fold name."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    params = mlflow.get_run(run_id).data.params
    return {
        key.removesuffix(FOLD_PARAMS_SUFFIX): json.loads(value)
        for key, value in params.items()
        if key.endswith(FOLD_PARAMS_SUFFIX)
    }


def run_artifact_table(tracking_uri: str, run_id: str, name: str) -> pd.DataFrame:
    """An aggregate CSV artefact of a run (e.g. ``subgroups.csv``) as a DataFrame."""
    import mlflow

    with tempfile.TemporaryDirectory() as tmp:
        path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=name, dst_path=tmp, tracking_uri=tracking_uri
        )
        return pd.read_csv(path)


@dataclass(frozen=True)
class SelectionRule:
    """``configs/selection_rule.yaml`` (spec §13.3), parsed."""

    version: int
    split: str
    population: str
    feature_sets: tuple[str, ...]
    slope_range: tuple[float, float]
    within_auc: float
    baseline: str
    baselines_selectable: bool
    text: str = ""


def load_selection_rule(path: Path) -> SelectionRule:
    """Parse the pre-registered selection rule; raises ValueError on an unsupported rule."""
    text = path.read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    if doc.get("rank_by") != "mean_loho_auc":
        raise ValueError("only rank_by: mean_loho_auc is implemented")
    if doc["tie_break"].get("prefer") != "lowest_complexity_rank":
        raise ValueError("only tie_break.prefer: lowest_complexity_rank is implemented")
    eligible = doc["eligible"]
    low, high = (float(v) for v in eligible["mean_calibration_slope"])
    return SelectionRule(
        version=int(doc["version"]),
        split=str(eligible["split"]),
        population=str(eligible["population"]),
        feature_sets=tuple(str(f) for f in eligible["feature_sets"]),
        slope_range=(low, high),
        within_auc=float(doc["tie_break"]["within_auc"]),
        baseline=str(doc["baseline_check"]["baseline"]),
        baselines_selectable=bool(doc.get("baselines_selectable", False)),
        text=text,
    )


@dataclass
class Selection:
    """The outcome of applying the rule: the chosen run and the evidence for it."""

    selected: pd.Series | None
    candidates: pd.DataFrame
    eligible: pd.DataFrame
    tie_band: pd.DataFrame
    baseline: pd.Series | None
    beats_baseline: bool | None
    fallback: bool
    notes: list[str] = field(default_factory=list)


def _is_baseline(runs: pd.DataFrame) -> pd.Series:
    return runs["model"].isin(BASELINE_MODELS) | (runs["family"] == "baseline")


def select_configuration(runs: pd.DataFrame, rule: SelectionRule) -> Selection:
    """Apply ``rule`` to ``runs`` (from :func:`load_runs`); see the module docstring."""
    scoped = runs[(runs["split"] == rule.split) & (runs["population"] == rule.population)]
    candidates = scoped[scoped["feature_set"].isin(rule.feature_sets)]
    if not rule.baselines_selectable:
        candidates = candidates[~_is_baseline(candidates)]
    candidates = candidates.dropna(subset=["mean_auc", "mean_calibration_slope"])
    low, high = rule.slope_range
    eligible = candidates[candidates["mean_calibration_slope"].between(low, high)]
    eligible = eligible.sort_values("mean_auc", ascending=False)
    notes: list[str] = []
    fallback = False
    tie_band = eligible.iloc[0:0]
    selected: pd.Series | None = None
    if len(eligible):
        best = float(eligible["mean_auc"].iloc[0])
        tie_band = eligible[eligible["mean_auc"] >= best - rule.within_auc - 1e-12]
        order = tie_band.assign(_neg_auc=-tie_band["mean_auc"]).sort_values(
            ["complexity_rank", "_neg_auc", "config"]
        )
        selected = order.drop(columns="_neg_auc").iloc[0]
        notes.append(
            f"{len(eligible)} of {len(candidates)} candidate configurations are eligible "
            f"(mean LOHO calibration slope in [{low}, {high}]); best mean LOHO AUC "
            f"{best:.3f}; {len(tie_band)} within {rule.within_auc} of it; lowest "
            f"complexity rank among them: {int(selected['complexity_rank'])}."
        )
    elif len(candidates):
        fallback = True
        distance = (candidates["mean_calibration_slope"] - 1.0).abs()
        selected = candidates.iloc[int(np.argmin(distance.to_numpy()))]
        notes.append(
            "No configuration is eligible: selected the one whose mean calibration slope is "
            "closest to 1.0 (a limitation to report)."
        )
    else:
        notes.append("No candidate configuration has been run under the rule's scope.")
    base = scoped[scoped["model"] == rule.baseline]
    baseline = None
    beats: bool | None = None
    if len(base):
        preferred = base[base["feature_set"] == "FS0"]
        baseline = (preferred if len(preferred) else base).iloc[0]
        if selected is not None:
            beats = bool(float(selected["mean_auc"]) > float(baseline["mean_auc"]))
            if not beats:
                notes.append(
                    f"Baseline check failed (mean LOHO AUC {float(selected['mean_auc']):.4f} "
                    f"<= {rule.baseline} {float(baseline['mean_auc']):.4f}): "
                    f"{NO_ADDED_DISCRIMINATION}."
                )
            else:
                notes.append(
                    f"Baseline check passed: mean LOHO AUC {float(selected['mean_auc']):.4f} "
                    f"> {rule.baseline} {float(baseline['mean_auc']):.4f}."
                )
    else:
        notes.append(f"No {rule.baseline} run in scope: the baseline check cannot be made.")
    return Selection(selected, candidates, eligible, tie_band, baseline, beats, fallback, notes)
