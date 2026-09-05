import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.special import logsumexp
from scipy.stats import norm

from mapreg.heterogeneity_adaptive_coupling import (
    CouplingEstimationRefusal,
    _log_choose,
    expected_binary_table_from_log_odds,
)
from mapreg.predictive_conditional import normal_mixture_prediction


@pytest.mark.parametrize("mu", [-3.0, 0.0, 2.0])
def test_zero_variance_is_exact_conditional_plugin(mu):
    rows, columns = np.array([20, 108]), np.array([100, 28])
    prediction = normal_mixture_prediction(mu, 0, rows, columns)
    np.testing.assert_allclose(
        prediction.mean_table,
        expected_binary_table_from_log_odds(mu, rows, columns),
        atol=1e-12,
    )
    assert prediction.quadrature_order == 0


@pytest.mark.parametrize("mu,tau2", [(1.0, 1.44), (-1.5, 0.3), (0.0, 4.0)])
def test_mixture_matches_independent_adaptive_quadrature(mu, tau2):
    rows, columns = np.array([20, 108]), np.array([100, 28])
    prediction = normal_mixture_prediction(mu, tau2, rows, columns)
    a = prediction.support
    log_weight = _log_choose(100, a) + _log_choose(28, 20 - a)

    def integrand(z):
        logits = log_weight + (mu + np.sqrt(tau2) * z) * a
        return np.exp(logits - logsumexp(logits)) * norm.pdf(z)

    reference, error = quad_vec(integrand, -12, 12, epsabs=1e-12, epsrel=1e-12)
    assert error < 1e-10
    np.testing.assert_allclose(prediction.probabilities, reference, atol=2e-11)
    np.testing.assert_allclose(prediction.mean_table.sum(axis=1), rows, atol=1e-12)
    np.testing.assert_allclose(prediction.mean_table.sum(axis=0), columns, atol=1e-12)
    assert prediction.probabilities.sum() == pytest.approx(1)
    low, high = prediction.count_interval()
    coverage = prediction.probabilities[(a >= low) & (a <= high)].sum()
    assert coverage >= 0.95


def test_uninformative_margins_have_point_mass_prediction():
    prediction = normal_mixture_prediction(1, 9, [0, 30], [12, 18])
    np.testing.assert_array_equal(prediction.mean_table, [[0, 0], [12, 18]])
    assert prediction.count_interval() == (0, 0)


def test_failed_quadrature_is_not_silently_accepted():
    with pytest.raises(CouplingEstimationRefusal, match="did not converge"):
        normal_mixture_prediction(1, 4, [20, 108], [100, 28], quadrature_orders=(2, 4))


@pytest.mark.parametrize("tau2", [-1, float("nan"), float("inf")])
def test_invalid_variance_is_rejected(tau2):
    with pytest.raises(ValueError):
        normal_mixture_prediction(0, tau2, [5, 5], [5, 5])


def test_predictive_mixture_has_lower_true_expected_deviance_than_plugin():
    mixture = normal_mixture_prediction(1, 1.44, [20, 108], [100, 28])
    plugin = normal_mixture_prediction(1, 0, [20, 108], [100, 28])
    a = mixture.support
    tables = np.array([a, 20 - a, 100 - a, 8 + a]).T

    def risk(prediction):
        terms = tables * np.log(np.maximum(tables, 1e-300) / prediction.ravel())
        return mixture.probabilities @ (2 / 128 * terms.sum(axis=1))

    assert risk(mixture.mean_table) < risk(plugin.mean_table)
