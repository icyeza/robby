from dataclasses import replace
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest

from robson_ml.feature_sets import FeatureSpec, ModelData, feature_spec
from robson_ml.features import load_feature_registry
from robson_ml.models import MODEL_REGISTRY, b1_robson_lookup, b2_robson_logistic, get_model
from robson_ml.models.b1_robson_lookup import RobsonLookup
from tests.test_feature_sets import model_data

P0 = ("logreg_l2", "xgboost", "mlp")
BASELINES = ("B0", "B1", "B2", "B3")
EXTRA = ("elasticnet", "cart", "random_forest", "svm_rbf", "ft_transformer")


@pytest.fixture(scope="module")
def data() -> ModelData:
    return model_data(load_feature_registry(Path("configs/features_v1.yaml")), n=900, seed=8)


def _params(name: str, strategy: str) -> dict[str, object]:
    spec = get_model(name)
    if spec.grid:
        params: dict[str, object] = {k: v[0] for k, v in spec.grid.items()}
    else:
        trial = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0)).ask()
        params = dict(spec.search_space(trial))
    if name == "xgboost":
        params["n_estimators"] = 20
    return {**params, "missing_strategy": strategy, "seed": 1}


def test_registry_holds_baselines_and_p0() -> None:
    assert set(BASELINES) | set(P0) <= set(MODEL_REGISTRY)
    ranks = {n: MODEL_REGISTRY[n].complexity_rank for n in P0}
    assert ranks == {"logreg_l2": 1, "xgboost": 5, "mlp": 7}
    assert get_model("xgboost").strategies() == ("M0", "M1", "M2")
    assert get_model("mlp").strategies() == ("M1", "M2")


def test_registry_holds_the_extra_models() -> None:
    ranks = {n: MODEL_REGISTRY[n].complexity_rank for n in EXTRA}
    assert ranks == {
        "elasticnet": 2,
        "cart": 3,
        "random_forest": 4,
        "svm_rbf": 6,
        "ft_transformer": 8,
    }
    assert all(get_model(n).strategies() == ("M1", "M2") for n in EXTRA)


def test_b1_lookup_on_toy_data() -> None:
    x = pd.DataFrame(
        {"robson_group_no_onset": ["1_2", "1_2", "1_2", "1_2", "5", "5", np.nan, "3_4"]}
    )
    y = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    model = RobsonLookup().fit(x, y)
    overall = 4 / 8
    assert model.overall_rate_ == pytest.approx(overall)
    assert model.rates_["1_2"] == pytest.approx((1 + 10 * overall) / (4 + 10))
    assert model.rates_["5"] == pytest.approx((2 + 10 * overall) / (2 + 10))
    assert model.rates_["3_4"] == pytest.approx((0 + 10 * overall) / (1 + 10))
    new = pd.DataFrame({"robson_group_no_onset": ["1_2", np.nan, "9"]})
    q = model.predict_proba(new)[:, 1]
    np.testing.assert_allclose(q, [model.rates_["1_2"], overall, overall])


def test_b1_reads_the_onset_free_robson_group(data: ModelData) -> None:
    """Spec v1.3: B1 is the lookup on robson_group_no_onset (1+2 and 3+4 merged)."""
    fs = feature_spec(data, "FS0")
    pipe = get_model("B1").build(_params("B1", "M0"), fs)
    pipe.fit(data.x[list(fs.columns)], data.y)
    lookup = pipe.steps[-1][1]
    assert lookup.column == "robson_group_no_onset"
    assert set(lookup.rates_) <= {"1_2", "3_4", *map(str, range(5, 11)), "partial"}
    assert {"1_2", "3_4"} <= set(lookup.rates_)
    no_group = replace(fs, columns=tuple(c for c in fs.columns if c != "robson_group_no_onset"))
    with pytest.raises(ValueError, match="robson_group"):
        get_model("B1").build(_params("B1", "M0"), no_group)


def test_b1_uses_onset_robson_group_only_for_onset_coded_population() -> None:
    legacy = model_data(
        load_feature_registry(Path("configs/features_v1.yaml")),
        n=600,
        seed=8,
        population="P_pred_onset_coded",
    )
    pipe = get_model("B1").build(_params("B1", "M0"), feature_spec(legacy, "FS4"))
    assert pipe.steps[-1][1].column == b1_robson_lookup.LEGACY_ROBSON_GROUP == "robson_group"


def test_b2_uses_robson_inputs_without_onset() -> None:
    assert b2_robson_logistic.ROBSON_INPUTS == (
        "parity",
        "previous_cs_count",
        "fetal_presentation",
        "plurality",
        "gestational_age_weeks",
        "ga_band_lower",
        "ga_band_upper",
    )


@pytest.mark.parametrize("name", [*BASELINES, *P0, *EXTRA])
def test_models_fit_every_feature_set_and_strategy(data: ModelData, name: str) -> None:
    spec = get_model(name)
    rows = np.arange(600)
    new = np.arange(600, len(data.y))
    feature_sets = ["FS0"] if name in ("B0", "B1", "B2") else ["FS0", "FS2", "FS4"]
    strategies = spec.strategies()
    if name == "ft_transformer":  # slow on CPU: one feature set and strategy suffice
        feature_sets, strategies = ["FS4"], ("M2",)
    for fs_name in feature_sets:
        fs: FeatureSpec = feature_spec(data, fs_name)
        for strategy in strategies:
            model = spec.build(_params(name, strategy), fs)
            x = data.x[list(fs.columns)]
            model.fit(x.iloc[rows], data.y[rows])
            p = model.predict_proba(x.iloc[new])[:, 1]
            assert p.shape == (len(new),)
            assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()


def test_m0_rejected_for_models_without_native_nan(data: ModelData) -> None:
    with pytest.raises(ValueError, match="M0"):
        get_model("logreg_l2").build(_params("logreg_l2", "M0"), feature_spec(data, "FS0"))
