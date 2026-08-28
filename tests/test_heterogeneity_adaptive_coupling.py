from math import comb

import numpy as np
import pytest

from mapreg.classical_residuals import poisson_independence_residuals
from mapreg.coupling_fields import normalized_hypergraph_laplacian
from mapreg.heterogeneity_adaptive_coupling import (
    CouplingEstimationRefusal,
    binary_table_from_helmert_coordinate,
    centered_classical_coordinate,
    centered_classical_residuals,
    centered_haldane_log_odds,
    fit_heterogeneity_adaptive_binary_coupling,
    fit_precision_weighted_coupling,
    helmert_coordinate_to_log_odds,
    log_odds_to_helmert_coordinate,
    paule_mandel_pool,
    pool_binary_coordinate_family,
    product_hypergraph_laplacian,
    signed_deviance_coordinate,
    signed_pearson_coordinate,
)


def test_haldane_centering_enumerates_the_exact_hypergeometric_support():
    table = np.array([[1, 2], [3, 4]])
    estimate = centered_haldane_log_odds(table)
    support = np.arange(4)
    probability = np.array([comb(4, a) * comb(6, 3 - a) / comb(10, 3) for a in support])
    log_odds = np.log(
        (support + 0.5)
        * (7 - 4 + support + 0.5)
        / ((3 - support + 0.5) * (4 - support + 0.5))
    )
    expected = float(probability @ log_odds)
    variance = float(probability @ np.square(log_odds - expected))
    assert estimate.support_lower == 0
    assert estimate.support_upper == 3
    assert estimate.null_mean_log_odds == pytest.approx(expected)
    assert estimate.null_variance == pytest.approx(variance)
    assert estimate.centered_log_odds == pytest.approx(
        estimate.observed_log_odds - expected
    )


def test_haldane_estimate_is_finite_with_zero_cells_and_degenerate_margins():
    sparse = centered_haldane_log_odds(np.array([[8, 0], [0, 5]]))
    degenerate = centered_haldane_log_odds(np.array([[0, 0], [5, 3]]))
    assert np.isfinite(sparse.centered_log_odds)
    assert np.isfinite(sparse.sampling_variance)
    assert sparse.sampling_variance > 0.0
    assert degenerate.centered_log_odds == pytest.approx(0.0)
    assert degenerate.null_variance == pytest.approx(0.0)
    assert np.isfinite(degenerate.sampling_variance)


def test_haldane_centering_changes_sign_under_a_row_swap():
    table = np.array([[12, 3], [4, 11]])
    original = centered_haldane_log_odds(table)
    swapped = centered_haldane_log_odds(table[::-1])
    assert swapped.centered_log_odds == pytest.approx(-original.centered_log_odds)
    assert swapped.null_variance == pytest.approx(original.null_variance)
    assert swapped.sampling_variance == pytest.approx(original.sampling_variance)


def test_fixed_margin_support_can_have_a_strictly_positive_lower_bound():
    estimate = centered_haldane_log_odds(np.array([[4, 3], [2, 1]]))
    assert estimate.support_lower == 3
    assert estimate.support_upper == 6
    assert estimate.supported


def test_full_log_odds_and_helmert_coordinate_round_trip_with_target_margins():
    log_odds = np.log(5.0)
    coordinate = log_odds_to_helmert_coordinate(log_odds)
    assert coordinate == pytest.approx(log_odds / 2.0)
    assert helmert_coordinate_to_log_odds(coordinate) == pytest.approx(log_odds)
    table = binary_table_from_helmert_coordinate(
        float(coordinate), np.array([17.0, 23.0]), np.array([19.0, 21.0])
    )
    reconstructed = np.log(table[0, 0] * table[1, 1] / (table[0, 1] * table[1, 0]))
    assert reconstructed == pytest.approx(log_odds, abs=1e-9)
    np.testing.assert_allclose(table.sum(axis=1), [17.0, 23.0])
    np.testing.assert_allclose(table.sum(axis=0), [19.0, 21.0])


def test_classical_coordinates_are_one_df_and_exactly_centered():
    independent = np.array([[10, 20], [20, 40]])
    associated = np.array([[12, 3], [4, 11]])
    assert signed_pearson_coordinate(independent) == pytest.approx(0.0)
    assert signed_deviance_coordinate(independent) == pytest.approx(0.0)
    assert signed_pearson_coordinate(associated) > 0.0
    assert signed_deviance_coordinate(associated) > 0.0
    for statistic in ("pearson", "deviance"):
        estimate = centered_classical_coordinate(associated, statistic=statistic)
        assert np.isfinite(estimate.centered_coordinate)
        assert estimate.null_variance >= 0.0


@pytest.mark.parametrize("residual", ["pearson", "deviance"])
def test_full_classical_residuals_use_the_same_exact_fixed_margin_null(residual):
    table = np.array([[1, 2], [3, 4]])
    estimate = centered_classical_residuals(table, residual=residual)
    support = np.arange(4)
    probability = np.array([comb(4, a) * comb(6, 3 - a) / comb(10, 3) for a in support])
    null = np.asarray(
        [
            poisson_independence_residuals(
                np.array([[a, 3 - a], [4 - a, 3 + a]]), residual=residual
            )
            for a in support
        ]
    )
    expected = np.tensordot(probability, null, axes=(0, 0))
    variance = np.tensordot(probability, np.square(null - expected), axes=(0, 0))
    np.testing.assert_allclose(estimate.null_mean_residuals, expected, atol=1e-15)
    np.testing.assert_allclose(estimate.null_variance_residuals, variance, atol=1e-15)
    np.testing.assert_allclose(
        estimate.centered_residuals,
        estimate.observed_residuals - expected,
    )


def test_paule_mandel_reduces_to_fixed_effects_without_heterogeneity():
    estimates = np.array([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]])
    variances = np.ones_like(estimates)
    pooled = paule_mandel_pool(estimates, variances)
    assert pooled.tau_squared[0] == pytest.approx(0.0)
    assert pooled.mean[0] == pytest.approx(1.0)
    assert pooled.variance[0] == pytest.approx(1.0 / 3.0)
    assert pooled.tau_squared[1] > 0.0


def test_paule_mandel_has_the_closed_form_equal_variance_solution():
    pooled = paule_mandel_pool(np.array([[-2.0], [2.0]]), np.array([[0.1], [0.1]]))
    assert pooled.mean[0] == pytest.approx(0.0)
    assert pooled.tau_squared[0] == pytest.approx(7.9)
    assert pooled.variance[0] == pytest.approx(4.0)
    assert pooled.q_statistic[0] == pytest.approx(1.0)


def test_paule_mandel_unequal_variance_reference_and_shift_invariance():
    estimates = np.array([[-3.0], [0.0], [5.0]])
    variances = np.array([[0.2], [1.0], [2.0]])
    pooled = paule_mandel_pool(estimates, variances)
    shifted = paule_mandel_pool(estimates + 11.0, variances)
    assert pooled.tau_squared[0] == pytest.approx(15.102707875046628)
    assert pooled.mean[0] == pytest.approx(0.5180070391555032)
    assert pooled.variance[0] == pytest.approx(5.378670767537265)
    assert pooled.q_statistic[0] == pytest.approx(2.0)
    assert shifted.mean[0] == pytest.approx(pooled.mean[0] + 11.0)
    assert shifted.tau_squared[0] == pytest.approx(pooled.tau_squared[0])
    assert shifted.variance[0] == pytest.approx(pooled.variance[0])


def test_paule_mandel_variance_floor_prevents_infinite_precision():
    pooled = paule_mandel_pool(np.zeros((3, 2)), np.zeros((3, 2)), variance_floor=1e-4)
    np.testing.assert_allclose(pooled.precision, 3e4)
    assert np.isfinite(pooled.precision).all()


def test_product_hypergraph_laplacian_matches_kronecker_sum():
    first_incidence = np.array([[1.0], [1.0]])
    second_incidence = np.array([[1.0], [1.0], [1.0]])
    laplacian = product_hypergraph_laplacian(first_incidence, second_incidence)
    assert laplacian.shape == (6, 6)
    assert np.linalg.eigvalsh(laplacian).min() >= -1e-12
    np.testing.assert_allclose(laplacian @ np.ones(6), 0.0, atol=1e-12)


def test_product_hypergraph_order_matches_c_order_matrix_action():
    first_incidence = np.array([[1, 0], [1, 1], [0, 1]], dtype=float)
    second_incidence = np.array([[1], [1]], dtype=float)
    first = normalized_hypergraph_laplacian(first_incidence)
    second = normalized_hypergraph_laplacian(second_incidence)
    product = product_hypergraph_laplacian(first_incidence, second_incidence)
    matrix = np.arange(6, dtype=float).reshape(3, 2)
    expected = first @ matrix + matrix @ second.T
    np.testing.assert_allclose(
        (product @ matrix.ravel(order="C")).reshape(3, 2), expected
    )


def test_precision_weighted_solve_matches_closed_form_graph_smoothing():
    laplacian = np.array([[1.0, -1.0], [-1.0, 1.0]])
    fit = fit_precision_weighted_coupling(
        np.array([1.0, -1.0]),
        np.ones(2),
        graph_laplacian=laplacian,
        graph_penalty=1.0,
    )
    np.testing.assert_allclose(fit.estimate, [1.0 / 3.0, -1.0 / 3.0])
    assert np.isfinite(fit.condition_number)


def test_unequal_precision_gradient_and_family_scale_equivariance():
    mean = np.array([1.0, -2.0, 0.5])
    precision = np.array([0.0, 2.0, 8.0])
    laplacian = np.array([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]])
    fit = fit_precision_weighted_coupling(
        mean,
        precision,
        graph_laplacian=laplacian,
        ridge_penalty=0.3,
        graph_penalty=0.7,
    )
    rescaled = fit_precision_weighted_coupling(
        mean,
        100.0 * precision,
        graph_laplacian=laplacian,
        ridge_penalty=0.3,
        graph_penalty=0.7,
    )
    gradient = (
        fit.precision * (fit.estimate - mean)
        + 0.3 * fit.estimate
        + 0.7 * (laplacian @ fit.estimate)
    )
    np.testing.assert_allclose(gradient, 0.0, atol=1e-12)
    assert fit.precision[0] == 0.0
    assert not fit.support[0]
    assert rescaled.precision_scale == pytest.approx(100.0 * fit.precision_scale)
    np.testing.assert_allclose(rescaled.estimate, fit.estimate)


def test_precision_weighted_fit_refuses_an_unanchored_singular_system():
    with pytest.raises(CouplingEstimationRefusal, match="condition-number"):
        fit_precision_weighted_coupling(np.array([1.0, 2.0]), np.array([1.0, 0.0]))


@pytest.mark.parametrize("family", ["haldane_log_odds", "pearson", "deviance"])
def test_one_df_pool_excludes_degenerate_margin_donors(family):
    tables = np.array(
        [
            [[[0, 0], [5, 3]]],
            [[[8, 2], [2, 8]]],
            [[[7, 3], [3, 7]]],
        ]
    )
    pooled = pool_binary_coordinate_family(tables, family=family)
    assert not pooled.donor_support[0, 0]
    assert pooled.pooled.support_count[0] == 2
    assert pooled.pooled.supported[0]
    assert pooled.variance_convention == "exact_fixed_margin_null_variance"
    reference = paule_mandel_pool(
        pooled.donor_coordinate[1:], pooled.donor_variance[1:]
    )
    assert pooled.pooled.mean[0] == pytest.approx(reference.mean[0])


def test_complete_estimator_is_deterministic_and_shape_preserving():
    donor_tables = np.array(
        [
            [[8, 2], [2, 8]],
            [[7, 3], [3, 7]],
            [[9, 1], [1, 9]],
        ]
    )
    tables = np.empty((3, 2, 2, 2, 2), dtype=int)
    for donor in range(3):
        for first in range(2):
            for second in range(2):
                tables[donor, first, second] = donor_tables[donor]
    incidence = np.array([[1.0], [1.0]])
    one = fit_heterogeneity_adaptive_binary_coupling(
        tables, incidence, incidence, graph_penalty=0.5
    )
    two = fit_heterogeneity_adaptive_binary_coupling(
        tables, incidence, incidence, graph_penalty=0.5
    )
    assert one.structured.estimate.shape == (2, 2)
    np.testing.assert_array_equal(one.structured.estimate, two.structured.estimate)
    assert np.isfinite(one.structured.estimate).all()
    assert np.isfinite(one.structured.condition_number)


@pytest.mark.parametrize(
    "table",
    [
        np.ones((3, 2)),
        np.array([[1.5, 2.0], [3.0, 4.0]]),
        np.array([[1.0, -1.0], [3.0, 4.0]]),
        np.zeros((2, 2)),
    ],
)
def test_fixed_margin_estimator_rejects_invalid_tables(table):
    with pytest.raises(ValueError):
        centered_haldane_log_odds(table)


def test_precision_weighted_fit_rejects_an_indefinite_laplacian():
    with pytest.raises(ValueError, match="positive semidefinite"):
        fit_precision_weighted_coupling(
            np.ones(2),
            np.ones(2),
            graph_laplacian=np.array([[0.0, 1.0], [1.0, 0.0]]),
        )
