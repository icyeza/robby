import itertools

import numpy as np
import pytest

from robson_serving.monitoring import expected_theatre_load, poisson_binomial_pmf


def _brute_force_pmf(p: np.ndarray) -> np.ndarray:
    pmf = np.zeros(len(p) + 1)
    for outcome in itertools.product((0, 1), repeat=len(p)):
        chosen = np.array(outcome)
        pmf[chosen.sum()] += np.prod(np.where(chosen == 1, p, 1 - p))
    return pmf


@pytest.mark.parametrize("n", range(1, 9))
def test_pmf_matches_brute_force(n: int) -> None:
    p = np.random.default_rng(n).uniform(0, 1, n)
    np.testing.assert_allclose(poisson_binomial_pmf(p), _brute_force_pmf(p), atol=1e-12, rtol=0)


@pytest.mark.parametrize("n", range(1, 9))
def test_interval_matches_brute_force_quantiles(n: int) -> None:
    p = np.random.default_rng(100 + n).uniform(0, 1, n)
    cdf = np.cumsum(_brute_force_pmf(p))
    load = expected_theatre_load(p)
    assert load.expected == pytest.approx(p.sum())
    assert load.low == int(np.argmax(cdf >= 0.05 - 1e-12))
    assert load.high == int(np.argmax(cdf >= 0.95 - 1e-12))


def test_binomial_case_by_hand() -> None:
    # Binomial(10, 0.5): P(X<=1)=11/1024, P(X<=2)=56/1024, P(X<=7)=968/1024, P(X<=8)=1013/1024
    assert expected_theatre_load([0.5] * 10) == (5.0, 2, 8)


def test_certain_and_empty_inputs() -> None:
    assert expected_theatre_load([1.0, 1.0, 0.0]) == (2.0, 2, 2)
    assert expected_theatre_load([]) == (0.0, 0, 0)
    expected, low, high = expected_theatre_load([0.2, 0.9])
    assert expected == pytest.approx(1.1) and low <= high


@pytest.mark.parametrize("bad", [[-0.1], [1.2], [float("nan")]])
def test_probabilities_are_validated(bad: list[float]) -> None:
    with pytest.raises(ValueError, match="within"):
        expected_theatre_load(bad)
