"""How robson_ml and robson_serving fit together: the dependency, the shared runtime code and
old artefacts. Synthetic data or artefact files only (never a data row)."""

import importlib.metadata
import re

import numpy as np

import robson_ml.metrics
from robson_serving.stats import fit_logistic, logit


def _names(distribution: str) -> set[str]:
    names = set()
    for requirement in importlib.metadata.requires(distribution) or []:
        match = re.match(r"[A-Za-z0-9_.-]+", requirement)
        assert match is not None
        names.add(match.group(0).lower())
    return names


def test_robson_ml_depends_on_robson_serving() -> None:
    assert "robson-serving" in _names("robson-ml")


def test_newton_logistic_matches_statsmodels() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.98, 500)
    truth = 1 / (1 + np.exp(-(0.4 + 0.7 * np.log(p / (1 - p)))))
    y = (rng.random(500) < truth).astype(float)
    np.testing.assert_allclose(
        fit_logistic(y, logit(p)), robson_ml.metrics.fit_logistic(y, logit(p)), atol=1e-8
    )
    np.testing.assert_allclose(
        fit_logistic(y, offset=logit(p)),
        robson_ml.metrics.fit_logistic(y, offset=logit(p)),
        atol=1e-8,
    )


def test_metrics_reuses_the_serving_logit() -> None:
    assert robson_ml.metrics.logit is logit
