import numpy as np
import pytest

from mapreg.heterogeneity_adaptive_coupling import (
    CouplingEstimationRefusal,
    product_hypergraph_laplacian,
)
from mapreg.penalty_complete_conditional_coupling import (
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


@pytest.mark.parametrize("graph_penalty", [0.0, 0.7])
def test_positive_ridge_identifies_entities_with_no_informative_donor_tables(
    graph_penalty: float,
) -> None:
    tables = np.empty((3, 1, 2, 2, 2), dtype=int)
    tables[:, 0, 0] = np.asarray(
        [
            [[12, 8], [8, 12]],
            [[11, 9], [9, 11]],
            [[14, 6], [6, 14]],
        ]
    )
    tables[:, 0, 1] = np.asarray([[0, 0], [20, 20]])
    first = np.ones((1, 1))
    second = np.ones((2, 1))
    laplacian = product_hypergraph_laplacian(first, second)
    kwargs = {
        "heterogeneity_penalty": 0.8,
        "ridge_penalty": 0.2,
        "graph_penalty": graph_penalty,
        "minimum_informative_donors": 0,
    }
    evaluation = evaluate_hierarchical_conditional_log_odds(
        np.zeros((3, 1, 2)),
        np.zeros((1, 2)),
        tables,
        graph_laplacian=laplacian,
        **kwargs,
    )
    assert evaluation.support_count.tolist() == [[3, 0]]
    assert np.linalg.eigvalsh(evaluation.hessian)[0] > 0.0
    assert evaluation.heterogeneity_penalty_scale > 0.0
    assert evaluation.population_penalty_scale > 0.0

    fit = fit_hierarchical_conditional_log_odds(
        tables,
        first,
        second,
        tolerance=1e-10,
        **kwargs,
    )
    assert fit.support_count.tolist() == [[3, 0]]
    assert np.isfinite(fit.population_log_odds).all()
    assert np.isfinite(fit.donor_log_odds).all()
    assert fit.minimum_schur_eigenvalue > 0.0
    assert fit.minimum_theta_curvature > 0.0
    np.testing.assert_allclose(
        fit.donor_log_odds[:, 0, 1], fit.population_log_odds[0, 1], atol=1e-10
    )
    if graph_penalty == 0.0:
        assert fit.population_log_odds[0, 1] == pytest.approx(0.0, abs=1e-12)
    else:
        assert abs(fit.population_log_odds[0, 1]) > 1e-5

    duplicated = fit_hierarchical_conditional_log_odds(
        np.concatenate([tables, tables], axis=0),
        first,
        second,
        tolerance=1e-10,
        **kwargs,
    )
    np.testing.assert_allclose(
        duplicated.population_log_odds, fit.population_log_odds, atol=1e-10
    )
    np.testing.assert_allclose(duplicated.donor_log_odds[:3], fit.donor_log_odds)
    np.testing.assert_allclose(duplicated.donor_log_odds[3:], fit.donor_log_odds)


def test_zero_support_mode_requires_integral_floor_and_positive_penalties() -> None:
    tables = np.empty((3, 1, 2, 2, 2), dtype=int)
    tables[:, 0, 0] = np.asarray([[12, 8], [8, 12]])
    tables[:, 0, 1] = np.asarray([[0, 0], [20, 20]])
    first = np.ones((1, 1))
    second = np.ones((2, 1))
    for graph_penalty in (0.0, 1.0):
        with pytest.raises(
            ValueError, match="requires positive heterogeneity and ridge"
        ):
            fit_hierarchical_conditional_log_odds(
                tables,
                first,
                second,
                ridge_penalty=0.0,
                graph_penalty=graph_penalty,
                minimum_informative_donors=0,
            )
    with pytest.raises(CouplingEstimationRefusal, match="too few informative"):
        fit_hierarchical_conditional_log_odds(tables, first, second)
    with pytest.raises(ValueError, match="between zero and donor count"):
        fit_hierarchical_conditional_log_odds(
            tables,
            first,
            second,
            minimum_informative_donors=-1,
        )
    for invalid in (0.9, 1.0, True, "0", np.asarray([0])):
        with pytest.raises(ValueError, match="must be an integer"):
            fit_hierarchical_conditional_log_odds(
                tables,
                first,
                second,
                minimum_informative_donors=invalid,
            )
    with pytest.raises(ValueError, match="between zero and donor count"):
        fit_hierarchical_conditional_log_odds(
            tables,
            first,
            second,
            minimum_informative_donors=10**100,
        )

    entirely_unsupported = np.tile(np.asarray([[0, 0], [20, 20]]), (3, 1, 2, 1, 1))
    with pytest.raises(CouplingEstimationRefusal, match="no donor-entity table"):
        fit_hierarchical_conditional_log_odds(
            entirely_unsupported,
            first,
            second,
            ridge_penalty=0.2,
            minimum_informative_donors=0,
        )


def test_zero_support_evaluation_has_exact_gradient_and_hessian() -> None:
    tables = np.empty((2, 1, 2, 2, 2), dtype=int)
    tables[:, 0, 0] = np.asarray([[[12, 8], [8, 12]], [[10, 10], [10, 10]]])
    tables[:, 0, 1] = np.asarray([[0, 0], [20, 20]])
    theta = np.asarray([[[0.2, -0.1]], [[-0.3, 0.4]]])
    mu = np.asarray([[0.1, -0.2]])
    laplacian = product_hypergraph_laplacian(np.ones((1, 1)), np.ones((2, 1)))
    kwargs = {
        "graph_laplacian": laplacian,
        "heterogeneity_penalty": 0.8,
        "ridge_penalty": 0.2,
        "graph_penalty": 0.7,
        "minimum_informative_donors": 0,
    }
    evaluation = evaluate_hierarchical_conditional_log_odds(theta, mu, tables, **kwargs)
    parameters = np.concatenate([theta.ravel(), mu.ravel()])
    donor_size = theta.size

    def value_gradient(candidate: np.ndarray) -> tuple[float, np.ndarray]:
        observed = evaluate_hierarchical_conditional_log_odds(
            candidate[:donor_size].reshape(theta.shape),
            candidate[donor_size:].reshape(mu.shape),
            tables,
            **kwargs,
        )
        gradient = np.concatenate(
            [observed.donor_gradient.ravel(), observed.population_gradient.ravel()]
        )
        return observed.objective, gradient

    step = 1e-5
    finite_gradient = np.empty_like(parameters)
    finite_hessian = np.empty_like(evaluation.hessian)
    for index in range(len(parameters)):
        delta = np.zeros_like(parameters)
        delta[index] = step
        plus_value, plus_gradient = value_gradient(parameters + delta)
        minus_value, minus_gradient = value_gradient(parameters - delta)
        finite_gradient[index] = (plus_value - minus_value) / (2.0 * step)
        finite_hessian[:, index] = (plus_gradient - minus_gradient) / (2.0 * step)
    exact_gradient = np.concatenate(
        [evaluation.donor_gradient.ravel(), evaluation.population_gradient.ravel()]
    )
    np.testing.assert_allclose(exact_gradient, finite_gradient, atol=2e-9)
    np.testing.assert_allclose(evaluation.hessian, finite_hessian, atol=2e-9)


def test_tolerated_negative_laplacian_mode_is_projected_before_evaluation() -> None:
    tables = _small_tables()
    near_psd = np.diag([1.0, -5e-10])
    evaluation = evaluate_hierarchical_conditional_log_odds(
        np.zeros((3, 1, 2)),
        np.zeros((1, 2)),
        tables,
        graph_laplacian=near_psd,
        heterogeneity_penalty=0.8,
        ridge_penalty=0.2,
        graph_penalty=1e10,
    )
    assert np.linalg.eigvalsh(evaluation.hessian)[0] > 0.0

    with pytest.raises(ValueError, match="positive semidefinite"):
        evaluate_hierarchical_conditional_log_odds(
            np.zeros((3, 1, 2)),
            np.zeros((1, 2)),
            tables,
            graph_laplacian=np.diag([1.0, -2e-9]),
            graph_penalty=1.0,
        )
