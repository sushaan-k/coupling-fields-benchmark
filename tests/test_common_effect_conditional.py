import numpy as np
import pytest

from mapreg.common_effect_conditional import (
    fit_common_effect_conditional_log_odds,
)
from mapreg.heterogeneity_adaptive_coupling import (
    CouplingEstimationRefusal,
    evaluate_conditional_log_odds,
    fit_structured_conditional_log_odds,
)


def _interior_tables() -> np.ndarray:
    return np.array(
        [
            [[[[12, 8], [8, 12]], [[7, 13], [13, 7]]]],
            [[[[11, 9], [9, 11]], [[8, 12], [12, 8]]]],
            [[[[14, 6], [6, 14]], [[9, 11], [11, 9]]]],
        ]
    )


def test_coordinatewise_fit_agrees_with_joint_exact_fit_on_interior_data() -> None:
    tables = _interior_tables()
    coordinatewise = fit_common_effect_conditional_log_odds(tables)
    joint = fit_structured_conditional_log_odds(
        tables,
        np.ones((1, 1)),
        np.ones((2, 1)),
        initial_log_odds=np.zeros((1, 2)),
        ridge_penalty=0.0,
        graph_penalty=0.0,
        minimum_informative_donors=2,
        tolerance=1e-10,
    )

    np.testing.assert_allclose(coordinatewise.log_odds, joint.log_odds, atol=1e-11)
    assert coordinatewise.objective == pytest.approx(joint.objective, abs=1e-12)


def test_fit_carries_an_exact_external_score_certificate() -> None:
    tables = _interior_tables()
    fit = fit_common_effect_conditional_log_odds(tables)
    evaluation = evaluate_conditional_log_odds(
        fit.log_odds,
        tables,
        minimum_informative_donors=2,
    )

    np.testing.assert_allclose(fit.gradient, evaluation.gradient, atol=2e-14)
    np.testing.assert_allclose(fit.data_precision, evaluation.data_precision, atol=2e-14)
    assert fit.gradient_norm == pytest.approx(np.max(np.abs(evaluation.gradient)))
    assert fit.scaled_gradient_norm < 1e-13


def test_bernoulli_strata_have_closed_form_log_two_cmle() -> None:
    tables = np.asarray(
        [
            [[[[1, 0], [0, 1]]]],
            [[[[1, 0], [0, 1]]]],
            [[[[0, 1], [1, 0]]]],
        ]
    )
    fit = fit_common_effect_conditional_log_odds(tables)

    assert fit.log_odds.item() == pytest.approx(np.log(2.0), abs=1e-12)
    assert fit.gradient_norm < 1e-12


def test_near_boundary_finite_fit_is_deterministic() -> None:
    tables = np.array(
        [
            [[[[20, 0], [0, 20]]]],
            [[[[19, 1], [1, 19]]]],
            [[[[18, 2], [2, 18]]]],
        ]
    )
    first = fit_common_effect_conditional_log_odds(tables)
    second = fit_common_effect_conditional_log_odds(tables)

    assert first.converged
    assert np.isfinite(first.log_odds).all()
    assert np.isfinite(first.data_precision).all()
    np.testing.assert_array_equal(first.log_odds, second.log_odds)
    np.testing.assert_array_equal(first.root_iterations, second.root_iterations)


@pytest.mark.parametrize(
    "boundary_table",
    (
        np.array([[20, 0], [0, 20]]),
        np.array([[0, 20], [20, 0]]),
    ),
)
def test_boundary_or_infinite_mle_is_refused(boundary_table: np.ndarray) -> None:
    boundary = np.repeat(boundary_table[None, None, None, :, :], 3, axis=0)
    with pytest.raises(CouplingEstimationRefusal, match="boundary or infinite MLE"):
        fit_common_effect_conditional_log_odds(boundary)


def test_insufficient_informative_support_is_refused() -> None:
    tables = np.array(
        [
            [[[[8, 2], [2, 8]]]],
            [[[[0, 0], [5, 3]]]],
        ]
    )
    with pytest.raises(CouplingEstimationRefusal, match="too few informative donors"):
        fit_common_effect_conditional_log_odds(tables)
