from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from robson_engine import load_rule_set
from robson_ml import eda, figures
from robson_ml.mapping import FieldMapping, MappingConfig
from robson_ml.populations import audit_population, prediction_population
from robson_ml.privacy import SECONDARY, SUPPRESSED
from robson_ml.profile import published_profile, write_profile
from robson_ml.robson_run import classify_frame
from tests.synthetic import make_admissions

HIDDEN = (SUPPRESSED, SECONDARY)


@pytest.fixture(scope="module")
def classified() -> pd.DataFrame:
    return classify_frame(make_admissions(1500, seed=4), load_rule_set())


def _counts(table: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = table[columns].astype(object).to_numpy().ravel()
    return np.array([float(v) for v in values if not isinstance(v, str)])


def _no_small(values: np.ndarray) -> bool:
    return not bool(((values > 0) & (values < 5)).any())


def test_histogram_merges_sparse_bins_and_keeps_totals() -> None:
    values = pd.Series([0.0] * 30 + [1.0] * 3 + [2.0] * 20 + [9.0] + [np.nan] * 4)
    table = eda.binned_hist_counts(values, edges=np.arange(0, 11, 1.0))
    assert _no_small(table["n"].to_numpy(dtype=float))
    assert int(table["total"].sum()) == 54  # every non-missing value is counted once
    assert "9" not in set(table["bin"])  # the lone extreme is merged, never shown alone


def test_histogram_by_group_merges_on_any_small_group() -> None:
    rng = np.random.default_rng(0)
    values = pd.Series(rng.normal(30, 5, 400))
    by = pd.Series(np.where(rng.random(400) < 0.5, "CS", "vaginal"))
    table = eda.binned_hist_counts(values, by=by, name="maternal_age")
    assert list(table.columns[3:5]) == ["CS", "vaginal"]
    assert _no_small(_counts(table, ["CS", "vaginal"]))
    assert int(table["CS"].sum()) == int((by == "CS").sum())


def test_histogram_edges_never_sit_on_an_observed_extreme() -> None:
    values = pd.Series(np.r_[np.linspace(10, 20, 200), [97.3]])
    edges = eda.histogram_edges(values)
    assert 97.3 not in edges
    assert edges.max() < 97.3
    assert eda.histogram_edges(pd.Series([1.0, 2.0])).size == 0  # too few to release


def test_histogram_integer_labels() -> None:
    values = pd.Series([0] * 20 + [1] * 20 + [2] * 6 + [3] * 2)
    table = eda.binned_hist_counts(values, name="parity")
    assert table["bin"].tolist() == ["0", "1", "2-3"]


def test_histogram_empty_when_a_group_is_tiny() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 4 + [3.0])
    by = pd.Series(["a"] * 20 + ["b"])
    assert eda.binned_hist_counts(values, edges=[0, 2, 4, 6], by=by).empty


def test_rate_by_suppresses_small_cells_and_complements(classified: pd.DataFrame) -> None:
    audit, _ = audit_population(classified)
    frame = audit.assign(group=classified["robson_group"].astype(object).fillna("none"))
    table = eda.rate_by(frame, "group")
    shown = table[~table["n"].isin(HIDDEN)]
    assert _no_small(shown["n"].astype(float).to_numpy())
    rest = shown[~shown["n_outcome"].isin(HIDDEN)]
    assert _no_small(rest["n_outcome"].astype(float).to_numpy())
    assert _no_small((rest["n"].astype(float) - rest["n_outcome"].astype(float)).to_numpy())
    # a single hidden cell would be recoverable from the total: never exactly one
    assert table["n"].isin(HIDDEN).sum() != 1


def test_rate_by_cross_table(classified: pd.DataFrame) -> None:
    audit, _ = audit_population(classified)
    frame = audit.assign(month=audit["delivery_date"].dt.to_period("M").astype(str))
    table = eda.rate_by(frame, ["facility_id", "month"])
    assert set(table.columns) >= {"facility_id", "month", "n", "n_outcome", "rate_pct"}
    for _, block in table.groupby("facility_id"):
        assert block["n"].isin(HIDDEN).sum() != 1


def test_level_rates_pool_rare_levels() -> None:
    series = pd.Series(["a"] * 30 + ["b"] * 30 + ["c", "d", "e"] + [None] * 6)
    outcome = pd.Series(np.tile([0, 1], len(series) // 2 + 1)[: len(series)])
    table = eda.level_rates(series, outcome)
    assert "c" not in set(table["level"])
    assert eda.RARE_LEVEL in set(table["level"])


def test_missingness_matrix_is_suppressed(classified: pd.DataFrame) -> None:
    matrix = eda.missingness_matrix(
        classified, ["parity", "gestational_age_weeks", "height_cm"], classified["facility_id"]
    )
    assert matrix.columns[0] == "all"
    assert matrix.loc["height_cm", "all"] == pytest.approx(50, abs=5)
    # plurality-like near-complete columns hide their small complements
    for value in matrix.to_numpy().ravel():
        assert isinstance(value, str) or 0 <= value <= 100


def test_correlation_matrix_requires_min_pairs() -> None:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    frame["c"] = np.where(np.arange(200) < 10, frame["a"], np.nan)  # only 10 values
    corr = eda.correlation_matrix(frame, ["a", "b", "c"])
    assert "c" not in corr.columns
    assert corr.loc["a", "a"] == pytest.approx(1.0)


def test_calibration_bins_meet_minimum_counts() -> None:
    rng = np.random.default_rng(2)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    bins = eda.calibration_bins(y, p)
    assert (bins["n"] >= eda.MIN_CALIBRATION_BIN).all()
    events = bins["observed"] * bins["n"]
    assert _no_small(np.round(events.to_numpy())) and _no_small(
        np.round((bins["n"] - events).to_numpy())
    )
    assert int(bins["n"].sum()) == 1000


def test_prelabour_onset_share(classified: pd.DataFrame) -> None:
    audit, _ = audit_population(classified)
    table = eda.prelabour_onset_share(audit)
    assert {"facility_id", "period", "rate_pct"} <= set(table.columns)


def test_nested_population_rates_protect_differences(classified: pd.DataFrame) -> None:
    audit, _ = audit_population(classified)
    pred, _ = prediction_population(audit)
    table = eda.nested_population_rates({"P_audit": audit, "P_pred": pred})
    assert set(table["population"]) == {"P_audit", "P_pred"}
    wide = table.pivot(index="facility_id", columns="population", values="n")
    for _, row in wide.iterrows():
        if not any(isinstance(v, str) for v in row):
            gap = float(row["P_audit"]) - float(row["P_pred"])
            assert not 0 < gap < 5


def test_suppress_partition_protects_a_small_part() -> None:
    table = pd.DataFrame({"a": [100, 50], "b": [3, 40], "c": [60, 20]})
    safe = eda.suppress_partition(table, ["a", "b", "c"])
    assert safe.loc[0, "b"] == SUPPRESSED
    assert (safe.loc[0, ["a", "c"]] == SECONDARY).any()  # else b = total - a - c
    assert safe.loc[1].tolist() == [50, 40, 20]


def test_published_profile_matches_written_files(tmp_path: Path, classified: pd.DataFrame) -> None:
    raw = classified[["facility_id", "parity", "maternal_age"]].reset_index(drop=True)
    config = MappingConfig(
        "t",
        None,
        {"parity": FieldMapping("parity", "integer", ("parity",), "confirmed")},
    )
    write_profile(raw, classified, config, {}, tmp_path)
    profile, counts = published_profile(raw, classified, config)
    written = pd.read_csv(tmp_path / "variable_profile.csv", dtype=str, keep_default_na=False)
    assert profile.astype(str).replace("None", "").replace("nan", "").to_numpy().tolist() == (
        written.to_numpy().tolist()
    )
    markdown = (tmp_path / "robson_inputs.md").read_text(encoding="utf-8")
    assert "Completeness of the six inputs" in markdown
    assert counts.completeness["input"].iloc[0] == "parity"


def test_figures_have_titles_and_labels() -> None:
    figures.apply_style()
    hist = pd.DataFrame({"bin": ["a", "b"], "left": [0, 1], "right": [1, 2], "n": [10, 5]})
    matrix = pd.DataFrame({"all": [10.0, SUPPRESSED], "F": [SECONDARY, 20.0]}, index=["x", "y"])
    curve = pd.DataFrame({"x": [0.1, 0.5], "y": [0.2, 0.4]})
    configs = pd.DataFrame(
        {"c": ["m1", "m2"], "mean": [0.7, 0.6], "lo": [0.6, 0.5], "hi": [0.8, 0.7]}
    )
    importance = pd.DataFrame({"v": [0.3, 0.1]}, index=["previous_cs_count", "age"])
    made = [
        figures.bar_chart(["a", "b"], [12.5, SUPPRESSED], "T", "X", "Y", reference=10.0),
        figures.histogram(hist, ["n"], "T", "X"),
        figures.heatmap(matrix, "T", "X", "Y", "Z"),
        figures.range_plot(configs, "c", "mean", "lo", "hi", "T", "X", reference=0.65),
        figures.scatter_groups(configs, "mean", "hi", "c", "T", "X", "Y", xband=(0.6, 0.7)),
        figures.line_curves({"m": curve}, "x", "y", "T", "X", "Y", diagonal=True, ylim=(0, 1)),
        figures.stacked_bars(pd.DataFrame({"f1": [5, 3]}, index=["fit", "test"]), "T", "X", "Y"),
        figures.importance_bars(importance, "v", "T", "X", highlight=["previous_cs_count"]),
    ]
    for fig in made:
        assert fig._suptitle is not None and fig._suptitle.get_text() == "T"
        ax = fig.axes[0]
        assert ax.get_xlabel() and ax.get_ylabel()
        matplotlib.pyplot.close(fig)
