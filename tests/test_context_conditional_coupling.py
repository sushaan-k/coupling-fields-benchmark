import math

import numpy as np
import pytest

from mapreg.context_conditional_coupling import (
    fit_context_conditional_log_odds,
    predict_context_log_odds,
)
from mapreg.heterogeneity_adaptive_coupling import CouplingEstimationRefusal


def _trend_tables() -> tuple[np.ndarray, np.ndarray]:
    contexts = np.column_stack((np.ones(6), np.linspace(-1.0, 1.0, 6)))
    first = np.array([7, 8, 9, 11, 12, 13])
    second = first[::-1]
    tables = np.empty((6, 2, 2, 2), dtype=int)
    for donor in range(6):
        tables[donor, 0] = [
            [first[donor], 20 - first[donor]],
            [20 - first[donor], first[donor]],
        ]
        tables[donor, 1] = [
            [second[donor], 20 - second[donor]],
            [20 - second[donor], second[donor]],
        ]
    return tables, contexts


def _conditional_score(table: np.ndarray, log_odds: float) -> tuple[float, float]:
    row_zero = int(table[0].sum())
    column_zero = int(table[:, 0].sum())
    total = int(table.sum())
    lower = max(0, row_zero + column_zero - total)
    upper = min(row_zero, column_zero)
    feasible = np.arange(lower, upper + 1, dtype=float)
    weights = np.asarray(
        [
            math.comb(column_zero, int(value))
            * math.comb(total - column_zero, row_zero - int(value))
            for value in feasible
        ],
        dtype=float,
    )
    centered = feasible - float(table[0, 0])
    tilted = weights * np.exp(centered * log_odds)
    probability = tilted / tilted.sum()
    score = float(probability @ centered)
    precision = float(probability @ np.square(centered - score))
    return score, precision


def test_coordinates_fit_independently_and_recover_opposite_context_slopes() -> None:
    tables, contexts = _trend_tables()
    joint = fit_context_conditional_log_odds(
        tables,
        contexts,
        donor_deviation_penalty=5.0,
        coefficient_ridge_penalty=np.array([0.2, 0.2]),
    )
    separate = [
        fit_context_conditional_log_odds(
            tables[:, entity : entity + 1],
            contexts,
            donor_deviation_penalty=5.0,
            coefficient_ridge_penalty=np.array([0.2, 0.2]),
        )
        for entity in range(2)
    ]

    assert joint.coefficient[1, 0] > 0.0
    assert joint.coefficient[1, 1] < 0.0
    for entity, fitted in enumerate(separate):
        np.testing.assert_allclose(
            joint.coefficient[:, entity], fitted.coefficient[:, 0], atol=1e-12
        )
        np.testing.assert_allclose(
            joint.donor_log_odds[:, entity], fitted.donor_log_odds[:, 0], atol=1e-12
        )
        assert joint.coordinate_objective[entity] == pytest.approx(
            fitted.coordinate_objective[0], abs=1e-12
        )


def test_returned_score_and_factor_certificates_are_external_recomputations() -> None:
    tables, contexts = _trend_tables()
    ridge = np.array([0.3, 0.7])
    deviation_penalty = 2.5
    fit = fit_context_conditional_log_odds(
        tables,
        contexts,
        donor_deviation_penalty=deviation_penalty,
        coefficient_ridge_penalty=ridge,
        tolerance=1e-10,
    )

    for entity in range(2):
        score = np.empty(6)
        precision = np.empty(6)
        for donor in range(6):
            score[donor], precision[donor] = _conditional_score(
                tables[donor, entity], fit.donor_log_odds[donor, entity]
            )
        scale = 1.0 / len(contexts)
        expected_coefficient_gradient = (
            scale * (contexts.T @ score) + ridge * fit.coefficient[:, entity]
        )
        expected_deviation_gradient = scale * (
            score + deviation_penalty * fit.donor_deviation[:, entity]
        )
        np.testing.assert_allclose(
            fit.coefficient_gradient[:, entity],
            expected_coefficient_gradient,
            atol=2e-13,
        )
        np.testing.assert_allclose(
            fit.donor_deviation_gradient[:, entity],
            expected_deviation_gradient,
            atol=2e-13,
        )
        np.testing.assert_allclose(
            fit.donor_data_precision[:, entity], scale * precision, atol=2e-13
        )

        donor_curvature = scale * (precision + deviation_penalty)
        transmitted = (
            scale * precision * deviation_penalty / (precision + deviation_penalty)
        )
        schur = (contexts.T * transmitted) @ contexts + np.diag(ridge)
        eigenvalues = np.linalg.eigvalsh(schur)
        assert fit.minimum_schur_eigenvalue[entity] == pytest.approx(eigenvalues[0])
        assert fit.maximum_schur_eigenvalue[entity] == pytest.approx(eigenvalues[-1])
        assert fit.schur_condition_number[entity] == pytest.approx(
            eigenvalues[-1] / eigenvalues[0]
        )
        assert fit.donor_curvature_condition_number[entity] == pytest.approx(
            donor_curvature.max() / donor_curvature.min()
        )

    assert fit.converged
    assert fit.scaled_gradient_norm <= fit.gradient_tolerance
    assert fit.scaled_gradient_norm <= fit.gradient_norm


def test_complete_donor_panel_duplication_preserves_the_normalized_fit() -> None:
    tables, contexts = _trend_tables()
    arguments = {
        "donor_deviation_penalty": 2.5,
        "coefficient_ridge_penalty": np.array([0.3, 0.7]),
        "tolerance": 1e-10,
    }
    original = fit_context_conditional_log_odds(tables, contexts, **arguments)
    duplicated = fit_context_conditional_log_odds(
        np.concatenate((tables, tables)),
        np.concatenate((contexts, contexts)),
        **arguments,
    )

    np.testing.assert_allclose(duplicated.coefficient, original.coefficient, atol=1e-12)
    np.testing.assert_allclose(
        duplicated.donor_deviation[: len(contexts)],
        original.donor_deviation,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        duplicated.donor_deviation[len(contexts) :],
        original.donor_deviation,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        duplicated.coordinate_objective,
        original.coordinate_objective,
        atol=1e-12,
    )


def test_masked_donor_has_zero_deviation_and_context_only_prediction() -> None:
    tables = np.array(
        [
            [[[0, 0], [0, 0]]],
            [[[4, 1], [1, 4]]],
            [[[1, 4], [4, 1]]],
        ]
    )
    contexts = np.column_stack((np.ones(3), [-1.0, 0.0, 1.0]))
    support_mask = np.array([[False], [True], [True]])
    fit = fit_context_conditional_log_odds(
        tables,
        contexts,
        support_mask=support_mask,
        minimum_informative_donors=2,
    )

    np.testing.assert_array_equal(fit.donor_support[:, 0], [False, True, True])
    assert fit.support_count[0] == 2
    assert fit.donor_deviation[0, 0] == 0.0
    assert fit.donor_log_odds[0, 0] == pytest.approx(fit.context_log_odds[0, 0])
    assert fit.donor_data_precision[0, 0] == 0.0
    assert fit.donor_deviation_gradient[0, 0] == 0.0


def test_positive_penalties_make_boundary_data_finite() -> None:
    tables = np.repeat(np.array([[[[1, 0], [0, 1]]]]), 4, axis=0)
    contexts = np.ones((4, 1))
    fit = fit_context_conditional_log_odds(
        tables,
        contexts,
        donor_deviation_penalty=0.5,
        coefficient_ridge_penalty=0.25,
    )

    assert np.isfinite(fit.coefficient).all()
    assert np.isfinite(fit.donor_log_odds).all()
    assert fit.coefficient.item() > 0.0
    assert fit.minimum_schur_eigenvalue.item() > 0.0
    assert fit.minimum_donor_curvature.item() > 0.0
    assert fit.scaled_gradient_norm <= fit.gradient_tolerance


def test_context_prediction_preserves_new_context_and_entity_axes() -> None:
    tables, contexts = _trend_tables()
    fit = fit_context_conditional_log_odds(tables, contexts)
    new_contexts = np.array([[1.0, -0.25], [1.0, 0.75]])

    predicted = predict_context_log_odds(fit.coefficient, new_contexts)
    expected = np.einsum("np,pe->ne", new_contexts, fit.coefficient)
    np.testing.assert_allclose(predicted, expected)
    np.testing.assert_allclose(
        predict_context_log_odds(fit.coefficient, new_contexts[0]), expected[0]
    )


def test_nonpositive_penalties_and_insufficient_support_are_refused() -> None:
    tables, contexts = _trend_tables()
    with pytest.raises(ValueError, match="positive for every context"):
        fit_context_conditional_log_odds(
            tables, contexts, coefficient_ridge_penalty=np.array([0.1, 0.0])
        )
    with pytest.raises(ValueError, match="finite and positive"):
        fit_context_conditional_log_odds(tables, contexts, donor_deviation_penalty=0.0)

    mask = np.zeros(tables.shape[:-2], dtype=bool)
    mask[0, :] = True
    with pytest.raises(CouplingEstimationRefusal, match="too few informative donors"):
        fit_context_conditional_log_odds(
            tables,
            contexts,
            support_mask=mask,
            minimum_informative_donors=2,
        )
