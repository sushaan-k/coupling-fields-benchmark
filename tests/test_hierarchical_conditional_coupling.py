import warnings
from math import comb

import numpy as np
import pytest

from mapreg.heterogeneity_adaptive_coupling import (
    CouplingEstimationRefusal,
    expected_binary_table_from_log_odds,
    fit_structured_conditional_log_odds,
    product_hypergraph_laplacian,
)
from mapreg.hierarchical_conditional_coupling import (
    _schur_system,
    evaluate_hierarchical_conditional_log_odds,
    fit_hierarchical_conditional_log_odds,
)


def _small_tables() -> np.ndarray:
    return np.array(
        [
            [[[[12, 8], [8, 12]], [[7, 13], [13, 7]]]],
            [[[[11, 9], [9, 11]], [[8, 12], [12, 8]]]],
            [[[[14, 6], [6, 14]], [[9, 11], [11, 9]]]],
        ]
    )


def _chain_incidence(size: int) -> np.ndarray:
    incidence = np.zeros((size, size - 1), dtype=float)
    for edge in range(size - 1):
        incidence[edge : edge + 2, edge] = 1.0
    return incidence


def test_hierarchical_gradient_hessian_and_unique_convex_fit():
    tables = _small_tables()
    first = np.ones((1, 1))
    second = np.ones((2, 1))
    laplacian = product_hypergraph_laplacian(first, second)
    donor_log_odds = np.array([[[0.2, -0.5]], [[0.1, -0.3]], [[0.6, -0.1]]])
    population_log_odds = np.array([[0.25, -0.2]])
    kwargs = {
        "graph_laplacian": laplacian,
        "heterogeneity_penalty": 0.7,
        "ridge_penalty": 0.2,
        "graph_penalty": 0.4,
        "minimum_informative_donors": 2,
    }
    evaluation = evaluate_hierarchical_conditional_log_odds(
        donor_log_odds,
        population_log_odds,
        tables,
        **kwargs,
    )
    parameters = np.concatenate(
        [donor_log_odds.ravel(order="C"), population_log_odds.ravel(order="C")]
    )

    def value_gradient(parameters_to_test):
        donor_size = donor_log_odds.size
        tested = evaluate_hierarchical_conditional_log_odds(
            parameters_to_test[:donor_size].reshape(donor_log_odds.shape),
            parameters_to_test[donor_size:].reshape(population_log_odds.shape),
            tables,
            **kwargs,
        )
        gradient = np.concatenate(
            [
                tested.donor_gradient.ravel(order="C"),
                tested.population_gradient.ravel(order="C"),
            ]
        )
        return tested.objective, gradient

    step = 1e-5
    finite_gradient = np.empty_like(parameters)
    finite_hessian = np.empty_like(evaluation.hessian)
    for index in range(parameters.size):
        direction = np.zeros_like(parameters)
        direction[index] = step
        plus_value, plus_gradient = value_gradient(parameters + direction)
        minus_value, minus_gradient = value_gradient(parameters - direction)
        finite_gradient[index] = (plus_value - minus_value) / (2.0 * step)
        finite_hessian[:, index] = (plus_gradient - minus_gradient) / (2.0 * step)
    exact_gradient = np.concatenate(
        [
            evaluation.donor_gradient.ravel(order="C"),
            evaluation.population_gradient.ravel(order="C"),
        ]
    )
    np.testing.assert_allclose(exact_gradient, finite_gradient, atol=2e-9)
    np.testing.assert_allclose(evaluation.hessian, finite_hessian, atol=2e-9)
    assert np.linalg.eigvalsh(evaluation.hessian)[0] > 0.0

    full_step = -np.linalg.solve(evaluation.hessian, exact_gradient)
    schur, right_hand_side, theta_curvature, _ = _schur_system(
        donor_log_odds.reshape(3, 2),
        population_log_odds.ravel(order="C"),
        evaluation.donor_gradient.reshape(3, 2),
        evaluation.donor_data_precision.reshape(3, 2) / 3.0,
        laplacian,
        evaluation.effective_heterogeneity_penalty,
        evaluation.effective_ridge_penalty,
        evaluation.effective_graph_penalty,
    )
    population_step = np.linalg.solve(schur, right_hand_side)
    donor_step = (
        -evaluation.donor_gradient.reshape(3, 2)
        + evaluation.effective_heterogeneity_penalty * population_step[None, :]
    ) / theta_curvature
    block_step = np.concatenate([donor_step.ravel(order="C"), population_step])
    np.testing.assert_allclose(block_step, full_step, atol=1e-12)

    null = evaluate_hierarchical_conditional_log_odds(
        np.zeros_like(donor_log_odds),
        np.zeros_like(population_log_odds),
        tables,
        **kwargs,
    )
    expected_heterogeneity_scale = np.median(null.donor_data_precision)
    expected_population_scale = np.median(
        np.sum(null.donor_data_precision.reshape(3, 2), axis=0) / 3.0
    )
    assert null.heterogeneity_penalty_scale == pytest.approx(
        expected_heterogeneity_scale
    )
    assert null.population_penalty_scale == pytest.approx(expected_population_scale)
    assert null.effective_heterogeneity_penalty == pytest.approx(
        0.7 * expected_heterogeneity_scale / 3.0
    )

    fit = fit_hierarchical_conditional_log_odds(
        tables,
        first,
        second,
        heterogeneity_penalty=0.7,
        ridge_penalty=0.2,
        graph_penalty=0.4,
        tolerance=1e-10,
    )
    assert fit.converged
    assert fit.minimum_schur_eigenvalue > 0.0
    assert fit.minimum_theta_curvature > 0.0


def test_common_effect_limit_matches_the_existing_exact_conditional_fit():
    tables = _small_tables()
    first = np.ones((1, 1))
    second = np.ones((2, 1))
    common = fit_structured_conditional_log_odds(
        tables,
        first,
        second,
        ridge_penalty=0.2,
        graph_penalty=0.4,
        tolerance=1e-11,
    )
    hierarchical = fit_hierarchical_conditional_log_odds(
        tables,
        first,
        second,
        heterogeneity_penalty=1e6,
        ridge_penalty=0.2,
        graph_penalty=0.4,
        tolerance=1e-10,
    )
    np.testing.assert_allclose(
        hierarchical.population_log_odds,
        common.log_odds,
        atol=2e-6,
    )
    assert np.max(np.abs(hierarchical.donor_deviation)) < 2e-6


def test_controlled_donor_heterogeneity_is_recovered_without_reweighting_donors():
    target = np.array([-1.5, -0.75, 0.0, 0.75, 1.5])
    tables = np.empty((target.size, 1, 1, 2, 2), dtype=int)
    margins = np.array([60, 60])
    for donor, log_odds in enumerate(target):
        expected = expected_binary_table_from_log_odds(log_odds, margins, margins)
        upper_left = int(np.rint(expected[0, 0]))
        tables[donor, 0, 0] = [
            [upper_left, 60 - upper_left],
            [60 - upper_left, upper_left],
        ]
    fit = fit_hierarchical_conditional_log_odds(
        tables,
        np.ones((1, 1)),
        np.ones((1, 1)),
        heterogeneity_penalty=0.05,
        ridge_penalty=0.01,
        tolerance=1e-10,
    )
    recovered = fit.donor_log_odds[:, 0, 0]
    assert np.all(np.diff(recovered) > 0.0)
    assert np.corrcoef(target, recovered)[0, 1] > 0.99
    assert recovered[-1] - recovered[0] > 1.5
    assert abs(fit.population_log_odds[0, 0]) < 0.1


def test_complete_donor_duplication_preserves_normalized_ridge_and_graph_fit():
    tables = _small_tables()
    first = np.ones((1, 1))
    second = np.ones((2, 1))
    fit = fit_hierarchical_conditional_log_odds(
        tables,
        first,
        second,
        heterogeneity_penalty=0.8,
        ridge_penalty=0.3,
        graph_penalty=0.6,
        tolerance=1e-10,
    )
    duplicated = fit_hierarchical_conditional_log_odds(
        np.concatenate([tables, tables], axis=0),
        first,
        second,
        heterogeneity_penalty=0.8,
        ridge_penalty=0.3,
        graph_penalty=0.6,
        tolerance=1e-10,
    )
    np.testing.assert_allclose(
        duplicated.population_log_odds, fit.population_log_odds, atol=1e-10
    )
    np.testing.assert_allclose(duplicated.donor_log_odds[:3], fit.donor_log_odds)
    np.testing.assert_allclose(duplicated.donor_log_odds[3:], fit.donor_log_odds)
    assert duplicated.objective == pytest.approx(fit.objective)
    assert duplicated.heterogeneity_penalty_scale == pytest.approx(
        fit.heterogeneity_penalty_scale
    )
    assert duplicated.population_penalty_scale == pytest.approx(
        fit.population_penalty_scale
    )
    assert duplicated.effective_ridge_penalty == pytest.approx(
        fit.effective_ridge_penalty
    )
    assert duplicated.effective_graph_penalty == pytest.approx(
        fit.effective_graph_penalty
    )

    loose = fit_hierarchical_conditional_log_odds(
        tables,
        first,
        second,
        heterogeneity_penalty=0.8,
        ridge_penalty=0.3,
        graph_penalty=0.6,
        tolerance=0.3,
    )
    loose_duplicated = fit_hierarchical_conditional_log_odds(
        np.concatenate([tables, tables], axis=0),
        first,
        second,
        heterogeneity_penalty=0.8,
        ridge_penalty=0.3,
        graph_penalty=0.6,
        tolerance=0.3,
    )
    assert loose.iterations == loose_duplicated.iterations
    assert loose.iterations > 0
    np.testing.assert_allclose(
        loose_duplicated.population_log_odds, loose.population_log_odds
    )
    np.testing.assert_allclose(
        loose_duplicated.donor_log_odds[:3], loose.donor_log_odds
    )
    np.testing.assert_allclose(
        loose_duplicated.donor_log_odds[3:], loose.donor_log_odds
    )


def test_hyperedge_multiplicity_does_not_rescale_graph_or_ridge_solution():
    tables = np.tile(_small_tables(), (1, 2, 1, 1, 1))
    first = np.array([[1.0], [1.0]])
    second = np.array([[1.0], [1.0]])
    fit = fit_hierarchical_conditional_log_odds(
        tables,
        first,
        second,
        ridge_penalty=0.4,
        graph_penalty=0.7,
        tolerance=1e-10,
    )
    duplicated_edges = fit_hierarchical_conditional_log_odds(
        tables,
        np.repeat(first, 3, axis=1),
        np.repeat(second, 2, axis=1),
        ridge_penalty=0.4,
        graph_penalty=0.7,
        tolerance=1e-10,
    )
    np.testing.assert_allclose(
        duplicated_edges.population_log_odds,
        fit.population_log_odds,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        duplicated_edges.donor_log_odds,
        fit.donor_log_odds,
        atol=1e-10,
    )


def test_degenerate_margin_donor_has_zero_precision_and_zero_fitted_deviation():
    tables = np.array(
        [
            [[[[0, 0], [5, 3]]]],
            [[[[12, 8], [8, 12]]]],
            [[[[14, 6], [6, 14]]]],
        ]
    )
    incidence = np.ones((1, 1))
    fit = fit_hierarchical_conditional_log_odds(
        tables,
        incidence,
        incidence,
        minimum_informative_donors=2,
        tolerance=1e-9,
    )
    assert not fit.donor_support[0, 0, 0]
    assert fit.donor_data_precision[0, 0, 0] == 0.0
    assert fit.donor_deviation[0, 0, 0] == pytest.approx(0.0, abs=1e-10)

    insufficient = tables.copy()
    insufficient[1, 0, 0] = np.array([[0, 0], [7, 2]])
    with pytest.raises(CouplingEstimationRefusal, match="too few informative"):
        fit_hierarchical_conditional_log_odds(
            insufficient,
            incidence,
            incidence,
            minimum_informative_donors=2,
        )


def test_population_log_odds_predicts_the_exact_target_margin_expectation():
    tables = _small_tables()[:, :, :1]
    incidence = np.ones((1, 1))
    fit = fit_hierarchical_conditional_log_odds(
        tables,
        incidence,
        incidence,
        tolerance=1e-10,
    )
    rows = np.array([31, 49])
    columns = np.array([37, 43])
    prediction = expected_binary_table_from_log_odds(
        fit.population_log_odds[0, 0], rows, columns
    )
    np.testing.assert_allclose(prediction.sum(axis=1), rows, atol=1e-12)
    np.testing.assert_allclose(prediction.sum(axis=0), columns, atol=1e-12)
    assert np.all(prediction >= 0.0)
    support = np.arange(32)
    weights = np.array(
        [
            comb(37, upper_left)
            * comb(43, 31 - upper_left)
            * np.exp(fit.population_log_odds[0, 0] * upper_left)
            for upper_left in support
        ]
    )
    assert prediction[0, 0] == pytest.approx(float(support @ weights / weights.sum()))


def test_condition_limit_and_unregularized_boundary_produce_explicit_refusals():
    tables = _small_tables()
    with pytest.raises(CouplingEstimationRefusal, match="condition-number"):
        fit_hierarchical_conditional_log_odds(
            tables,
            np.ones((1, 1)),
            np.ones((2, 1)),
            heterogeneity_penalty=0.7,
            ridge_penalty=0.2,
            graph_penalty=0.4,
            maximum_condition_number=1.2,
        )

    diagonal = np.array([[1, 0], [0, 1]])
    boundary = np.tile(diagonal, (3, 1, 1, 1, 1))
    with pytest.raises(CouplingEstimationRefusal, match="boundary recession"):
        fit_hierarchical_conditional_log_odds(
            boundary,
            np.ones((1, 1)),
            np.ones((1, 1)),
            ridge_penalty=0.0,
        )


def test_connected_graph_can_anchor_a_boundary_entity_without_ridge():
    boundary = np.array([[10, 0], [0, 10]])
    interior = np.array([[10, 10], [10, 10]])
    tables = np.empty((3, 1, 2, 2, 2), dtype=int)
    tables[:, 0, 0] = boundary
    tables[:, 0, 1] = interior
    fit = fit_hierarchical_conditional_log_odds(
        tables,
        np.ones((1, 1)),
        np.ones((2, 1)),
        ridge_penalty=0.0,
        graph_penalty=1.0,
        tolerance=1e-9,
    )
    assert fit.population_log_odds[0, 0] > 0.0
    assert np.isfinite(fit.population_log_odds).all()
    assert fit.scaled_gradient_norm <= fit.gradient_tolerance

    all_boundary = tables.copy()
    all_boundary[:, 0, 1] = boundary
    with pytest.raises(CouplingEstimationRefusal, match="boundary recession"):
        fit_hierarchical_conditional_log_odds(
            all_boundary,
            np.ones((1, 1)),
            np.ones((2, 1)),
            ridge_penalty=0.0,
            graph_penalty=1.0,
        )


def test_disconnected_boundary_component_refuses_with_graph_curvature_elsewhere():
    boundary = np.array([[10, 0], [0, 10]])
    interior = np.array([[10, 10], [10, 10]])
    tables = np.empty((3, 1, 3, 2, 2), dtype=int)
    tables[:, 0, 0] = interior
    tables[:, 0, 1] = interior
    tables[:, 0, 2] = boundary
    second_incidence = np.array([[1.0], [1.0], [0.0]])
    with pytest.raises(CouplingEstimationRefusal, match="boundary recession"):
        fit_hierarchical_conditional_log_odds(
            tables,
            np.ones((1, 1)),
            second_incidence,
            ridge_penalty=0.0,
            graph_penalty=1.0,
        )


def test_effective_penalty_range_and_extreme_likelihood_fail_explicitly():
    tables = _small_tables()
    donor_log_odds = np.zeros((3, 1, 2))
    population_log_odds = np.zeros((1, 2))
    with pytest.raises(ValueError, match="underflows"):
        evaluate_hierarchical_conditional_log_odds(
            donor_log_odds,
            population_log_odds,
            tables,
            heterogeneity_penalty=np.nextafter(0.0, 1.0),
        )
    with pytest.raises(ValueError, match="overflows"):
        evaluate_hierarchical_conditional_log_odds(
            donor_log_odds,
            population_log_odds,
            tables,
            ridge_penalty=np.finfo(float).max,
        )

    boundary = np.tile(np.array([[20, 0], [0, 20]]), (2, 1, 1, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        finite = evaluate_hierarchical_conditional_log_odds(
            np.full((2, 1), 100.0),
            np.array([100.0]),
            boundary,
        )
        assert np.isfinite(finite.objective)
        assert np.all(finite.donor_data_precision > 0.0)
        with pytest.raises(CouplingEstimationRefusal, match="finite evaluation"):
            evaluate_hierarchical_conditional_log_odds(
                np.full((2, 1), -np.finfo(float).max),
                np.array([0.0]),
                boundary,
            )


def test_warning_free_convergence_on_39_by_100_product_graph():
    donors = 39
    side = 10
    tables = np.empty((donors, side, side, 2, 2), dtype=int)
    for donor in range(donors):
        for first_entity in range(side):
            for second_entity in range(side):
                upper_left = 6 + (donor + 2 * first_entity + 3 * second_entity) % 9
                tables[donor, first_entity, second_entity] = [
                    [upper_left, 20 - upper_left],
                    [20 - upper_left, upper_left],
                ]
    incidence = _chain_incidence(side)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit = fit_hierarchical_conditional_log_odds(
            tables,
            incidence,
            incidence,
            heterogeneity_penalty=0.8,
            ridge_penalty=0.2,
            graph_penalty=0.7,
            maximum_iterations=60,
            tolerance=1e-7,
        )
    assert fit.population_log_odds.shape == (10, 10)
    assert fit.donor_log_odds.shape == (39, 10, 10)
    assert fit.converged
    assert fit.scaled_gradient_norm <= fit.gradient_tolerance
    assert np.isfinite(fit.objective)
    assert np.isfinite(fit.schur_condition_number)
    assert np.isfinite(fit.theta_curvature_condition_number)
    assert fit.minimum_theta_curvature > 0.0
    assert fit.minimum_schur_eigenvalue > 0.0
