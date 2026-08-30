from __future__ import annotations

import numpy as np

from mapreg.longitudinal_conditional_coupling import (
    evaluate_longitudinal_conditional_log_odds,
    fit_longitudinal_conditional_log_odds,
)


def _tables() -> np.ndarray:
    values = np.empty((3, 2, 2, 2, 2, 2), dtype=np.int64)
    pre = np.asarray([[12, 28], [28, 12]])
    post = np.asarray([[28, 12], [12, 28]])
    for donor in range(3):
        for first, second in np.ndindex((2, 2)):
            shift = donor + first + second
            values[donor, 0, first, second] = pre + [[shift, 0], [0, shift]]
            values[donor, 1, first, second] = post + [[shift, 0], [0, shift]]
    return values


def test_analytic_gradient_matches_central_difference() -> None:
    tables = _tables()
    laplacian = np.asarray([[1.0, -1.0], [-1.0, 1.0]])
    generator = np.random.default_rng(17)
    state = generator.normal(scale=0.08, size=tables.shape[:-2])
    arguments = {
        "support_mask": np.ones(tables.shape[:-2], dtype=bool),
        "heterogeneity_penalty": 0.7,
        "population_ridge": 0.2,
        "graph_penalty": 0.3,
    }
    evaluation = evaluate_longitudinal_conditional_log_odds(
        state, tables, laplacian, laplacian, **arguments
    )
    epsilon = 1e-6
    for index in [(0, 0, 0, 0), (1, 1, 0, 1), (2, 0, 1, 1)]:
        left = state.copy()
        right = state.copy()
        left[index] -= epsilon
        right[index] += epsilon
        numerical = (
            evaluate_longitudinal_conditional_log_odds(
                right, tables, laplacian, laplacian, **arguments
            ).objective
            - evaluate_longitudinal_conditional_log_odds(
                left, tables, laplacian, laplacian, **arguments
            ).objective
        ) / (2.0 * epsilon)
        np.testing.assert_allclose(evaluation.gradient[index], numerical, rtol=2e-6)


def test_fit_recovers_paired_visit_direction_and_zero_sum_effects() -> None:
    tables = _tables()
    zero = np.zeros((2, 2), dtype=float)
    fit = fit_longitudinal_conditional_log_odds(
        tables,
        zero,
        zero,
        heterogeneity_penalty=1.0,
        population_ridge=0.1,
        graph_penalty=0.0,
        gradient_tolerance=2e-5,
    )

    assert fit.converged
    assert fit.gradient_norm <= 2e-5
    assert np.all(fit.population_change > 0.0)
    assert np.max(np.abs(fit.donor_baseline_deviation.sum(axis=0))) < 1e-12
    assert np.max(np.abs(fit.donor_change_deviation.sum(axis=0))) < 1e-12
    assert fit.informative_table_count == 24
    assert fit.retained_coordinate_count == 4


def test_support_mask_removes_data_without_removing_field_coordinates() -> None:
    tables = _tables()
    zero = np.zeros((2, 2), dtype=float)
    mask = np.zeros(tables.shape[:-2], dtype=bool)
    mask[:, :, 0, 0] = True
    fit = fit_longitudinal_conditional_log_odds(
        tables,
        zero,
        zero,
        support_mask=mask,
        heterogeneity_penalty=1.0,
        population_ridge=0.1,
        graph_penalty=0.0,
        gradient_tolerance=2e-5,
    )

    assert fit.population_mean.shape == (2, 2)
    assert fit.retained_coordinate_count == 1
    assert fit.informative_table_count == 6
    np.testing.assert_array_equal(fit.population_mean[1], 0.0)
    assert fit.population_mean[0, 1] == 0.0
