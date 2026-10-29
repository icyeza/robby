import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from robson_serving.stats import (
    PROB_CLIP,
    clip_probabilities,
    fit_logistic,
    logit,
    roc_auc,
    sigmoid,
)


def test_logit_clips_before_transforming() -> None:
    edge = np.log(PROB_CLIP / (1 - PROB_CLIP))
    np.testing.assert_allclose(logit([0.0, 0.5, 1.0]), [edge, 0.0, -edge])
    assert clip_probabilities([0.0, 1.0]).tolist() == [PROB_CLIP, 1 - PROB_CLIP]


def test_sigmoid_inverts_logit() -> None:
    p = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(sigmoid(logit(p)), p)


def test_intercept_only_fit_is_the_logit_of_the_mean() -> None:
    np.testing.assert_allclose(fit_logistic([1, 1, 1, 0]), [np.log(3.0)])


def test_two_group_fit_matches_the_saturated_solution() -> None:
    # x = -1: 1 of 4 positive; x = +1: 3 of 4 positive  =>  a = 0, b = log(3)
    x = [-1.0] * 4 + [1.0] * 4
    y = [1, 0, 0, 0, 1, 1, 1, 0]
    np.testing.assert_allclose(fit_logistic(y, x), [0.0, np.log(3.0)], atol=1e-10)


def test_offset_fit() -> None:
    offset = np.full(4, np.log(3.0))
    np.testing.assert_allclose(fit_logistic([1, 0, 1, 0], offset=offset), [-np.log(3.0)])


def test_a_constant_covariate_gives_the_minimum_norm_fit() -> None:
    # A baseline predicting one probability for everyone: the design is rank-deficient.
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.3).astype(float)
    alpha, beta = fit_logistic(y, np.full(300, -0.99))
    assert alpha + beta * -0.99 == pytest.approx(np.log(y.mean() / (1 - y.mean())))


def test_roc_auc_matches_sklearn_with_ties() -> None:
    rng = np.random.default_rng(3)
    y = (rng.random(400) < 0.3).astype(float)
    p = np.round(rng.random(400), 1)  # many ties, each counting one half
    assert roc_auc(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-12)
    assert roc_auc([0, 1], [0.5, 0.5]) == 0.5
    with pytest.raises(ValueError):
        roc_auc([1, 1], [0.2, 0.8])
