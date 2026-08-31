import math

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import gammaln, xlogy

from experiments import gse217494_heart_core
from mapreg.heterogeneity_adaptive_coupling import (
    expected_binary_table_from_log_odds,
)
from mapreg.poisson_loglinear import (
    PoissonLoglinearRefusal,
    fit_poisson_loglinear_interaction,
    fit_ridge_profiled_poisson_interaction,
    reconstruct_poisson_tables,
)


def _independent_poisson_glm(tables: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit donor nuisances and one interaction without profile formulas."""

    values = np.asarray(tables, dtype=float)
    donors = values.shape[0]
    design = np.zeros((4 * donors, 3 * donors + 1), dtype=float)
    response = values.reshape(-1)
    for donor in range(donors):
        for row in range(2):
            for column in range(2):
                observation = 4 * donor + 2 * row + column
                design[observation, 3 * donor] = 1.0
                design[observation, 3 * donor + 1] = row
                design[observation, 3 * donor + 2] = column
                design[observation, -1] = row * column

    def objective(parameter: np.ndarray) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            linear = design @ parameter
            mean = np.exp(linear)
            return float(np.sum(mean - response * linear + gammaln(response + 1.0)))

    def gradient(parameter: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            return design.T @ (np.exp(design @ parameter) - response)

    def hessian(parameter: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            mean = np.exp(design @ parameter)
            return design.T @ (mean[:, None] * design)

    start = np.zeros(design.shape[1], dtype=float)
    for donor in range(donors):
        rows = values[donor].sum(axis=1)
        columns = values[donor].sum(axis=0)
        total = values[donor].sum()
        start[3 * donor] = math.log(rows[0] * columns[0] / total)
        start[3 * donor + 1] = math.log(rows[1] / rows[0])
        start[3 * donor + 2] = math.log(columns[1] / columns[0])
    result = minimize(
        objective,
        start,
        method="trust-exact",
        jac=gradient,
        hess=hessian,
        options={"gtol": 1e-11, "maxiter": 1_000},
    )
    assert np.max(np.abs(gradient(result.x))) < 2e-8
    return result.x, -objective(result.x)


def _independent_ridge_poisson_glm(
    tables: np.ndarray, ridge_penalty: float
) -> tuple[np.ndarray, float]:
    """Fit the mean-likelihood ridge model without profile formulas."""

    values = np.asarray(tables, dtype=float)
    donors = values.shape[0]
    design = np.zeros((4 * donors, 3 * donors + 1), dtype=float)
    response = values.reshape(-1)
    for donor in range(donors):
        for row in range(2):
            for column in range(2):
                observation = 4 * donor + 2 * row + column
                design[observation, 3 * donor] = 1.0
                design[observation, 3 * donor + 1] = row
                design[observation, 3 * donor + 2] = column
                design[observation, -1] = row * column

    def objective(parameter: np.ndarray) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            linear = design @ parameter
            mean = np.exp(linear)
            likelihood = np.sum(mean - response * linear + gammaln(response + 1.0))
            return float(likelihood / donors + 0.5 * ridge_penalty * parameter[-1] ** 2)

    def gradient(parameter: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            output = design.T @ (np.exp(design @ parameter) - response) / donors
            output[-1] += ridge_penalty * parameter[-1]
            return output

    def hessian(parameter: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            mean = np.exp(design @ parameter)
            output = design.T @ (mean[:, None] * design) / donors
            output[-1, -1] += ridge_penalty
            return output

    start = np.zeros(design.shape[1], dtype=float)
    for donor in range(donors):
        rows = values[donor].sum(axis=1)
        columns = values[donor].sum(axis=0)
        total = values[donor].sum()
        start[3 * donor] = math.log(rows[0] * columns[0] / total)
        start[3 * donor + 1] = math.log(rows[1] / rows[0])
        start[3 * donor + 2] = math.log(columns[1] / columns[0])
    result = minimize(
        objective,
        start,
        method="trust-exact",
        jac=gradient,
        hess=hessian,
        options={"gtol": 1e-11, "maxiter": 1_000},
    )
    assert np.max(np.abs(gradient(result.x))) < 2e-8
    return result.x, objective(result.x)


def _independent_fixed_interaction_profile(
    log_odds: float, rows: np.ndarray, columns: np.ndarray
) -> np.ndarray:
    nuisance = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )
    interaction = np.asarray([0.0, 0.0, 0.0, log_odds])

    def objective(parameter: np.ndarray) -> float:
        return float(np.exp(nuisance @ parameter + interaction).sum())

    constraints = (
        {
            "type": "eq",
            "fun": lambda parameter: (
                np.exp(nuisance @ parameter + interaction).reshape(2, 2).sum(axis=1)
                - rows
            ),
        },
        {
            "type": "eq",
            "fun": lambda parameter: (
                np.exp(nuisance @ parameter + interaction).reshape(2, 2).sum(axis=0)[0]
                - columns[0]
            ),
        },
    )
    result = minimize(
        objective,
        np.asarray([math.log(rows.sum() / 4.0), 0.0, 0.0]),
        method="SLSQP",
        constraints=constraints,
        options={"ftol": 1e-13, "maxiter": 1_000},
    )
    assert result.success, result.message
    return np.exp(nuisance @ result.x + interaction).reshape(2, 2)


def test_profile_mle_matches_independently_parameterized_poisson_glm() -> None:
    tables = np.asarray(
        [
            [[12, 7], [9, 18]],
            [[8, 11], [6, 17]],
            [[15, 5], [13, 19]],
            [[9, 8], [4, 14]],
            [[11, 10], [7, 16]],
        ],
        dtype=np.int64,
    )
    fitted = fit_poisson_loglinear_interaction(tables)
    independent_parameter, independent_log_likelihood = _independent_poisson_glm(tables)

    assert fitted.group_labels == ("__pooled__",)
    assert fitted.log_odds.shape == (1,)
    assert fitted.log_odds.item() == pytest.approx(independent_parameter[-1], abs=2e-9)
    assert fitted.profile_log_likelihood == pytest.approx(
        independent_log_likelihood, abs=2e-9
    )
    assert fitted.maximum_absolute_score < 1e-10
    assert fitted.maximum_scaled_score <= fitted.score_tolerance
    assert fitted.maximum_absolute_log_odds_error < 1e-12
    assert fitted.certificate_tolerance == 1e-8
    assert fitted.pseudocount == 0.0
    assert fitted.converged


def test_mean_profile_ridge_matches_independently_parameterized_poisson_glm() -> None:
    tables = np.asarray(
        [
            [[12, 7], [9, 18]],
            [[8, 11], [6, 17]],
            [[15, 5], [13, 19]],
            [[9, 8], [4, 14]],
            [[11, 10], [7, 16]],
        ],
        dtype=np.int64,
    )
    fitted = fit_ridge_profiled_poisson_interaction(tables, ridge_penalty=0.01)
    independent_parameter, independent_objective = _independent_ridge_poisson_glm(
        tables, 0.01
    )

    assert fitted.log_odds.item() == pytest.approx(independent_parameter[-1], abs=2e-9)
    assert fitted.penalized_objective.item() == pytest.approx(
        independent_objective, abs=2e-9
    )
    assert fitted.mean_score.item() == pytest.approx(
        0.01 * fitted.log_odds.item(), abs=1e-11
    )
    assert fitted.maximum_scaled_penalized_score <= fitted.score_tolerance
    assert fitted.maximum_absolute_log_odds_error < 1e-11
    assert fitted.status.item() == "FINITE"
    assert fitted.ridge_penalty == 0.01
    assert fitted.bracket_bound == 16.0
    assert fitted.converged


def test_donor_stratification_removes_simpson_margin_confounding() -> None:
    tables = np.asarray(
        [
            [[81, 9], [9, 1]],
            [[1, 9], [9, 81]],
        ],
        dtype=np.int64,
    )
    pooled = tables.sum(axis=0)
    pooled_log_odds = math.log(
        pooled[0, 0] * pooled[1, 1] / (pooled[0, 1] * pooled[1, 0])
    )
    fitted = fit_ridge_profiled_poisson_interaction(tables)

    assert pooled_log_odds > 3.0
    assert fitted.log_odds.item() == pytest.approx(0.0, abs=1e-14)
    assert fitted.mean_score.item() == pytest.approx(0.0, abs=1e-14)
    assert fitted.root_iterations.item() == 0


@pytest.mark.parametrize(
    ("table", "sign"),
    [
        (np.asarray([[128, 0], [0, 128]]), 1),
        (np.asarray([[0, 128], [128, 0]]), -1),
    ],
)
def test_mean_profile_ridge_is_finite_at_complete_separation(
    table: np.ndarray, sign: int
) -> None:
    tables = np.repeat(table[None, ...], 3, axis=0)
    with pytest.raises(PoissonLoglinearRefusal, match="boundary or infinite MLE"):
        fit_poisson_loglinear_interaction(tables)

    fitted = fit_ridge_profiled_poisson_interaction(tables)

    assert fitted.status.item() == "FINITE"
    assert 0.0 < sign * fitted.log_odds.item() < 16.0
    assert fitted.bracket_lower.item() == -16.0
    assert fitted.bracket_upper.item() == 16.0
    assert fitted.bracket_lower_score.item() > 0.0
    assert fitted.bracket_upper_score.item() < 0.0
    assert fitted.maximum_scaled_penalized_score <= fitted.score_tolerance
    assert fitted.maximum_absolute_log_odds_error <= fitted.certificate_tolerance


def test_ridge_profile_reports_no_information_without_refusing_other_groups() -> None:
    degenerate = np.asarray([[128, 128], [0, 0]], dtype=np.int64)
    informative = np.asarray([[90, 38], [38, 90]], dtype=np.int64)
    tables = np.asarray([degenerate, degenerate, informative, informative])
    fitted = fit_ridge_profiled_poisson_interaction(
        tables, np.asarray(["empty", "empty", "signal", "signal"])
    )
    by_group = {label: index for index, label in enumerate(fitted.group_labels)}
    empty = by_group["empty"]
    signal = by_group["signal"]

    assert fitted.status[empty] == "NO_INFORMATION"
    assert fitted.log_odds[empty] == 0.0
    assert fitted.informative_table_count[empty] == 0
    assert fitted.degenerate_table_count[empty] == 2
    assert fitted.penalized_information[empty] == 0.01
    assert fitted.penalized_objective[empty] == 0.0
    assert fitted.root_iterations[empty] == 0
    assert fitted.status[signal] == "FINITE"
    assert fitted.informative_table_count[signal] == 2
    assert fitted.minimum_positive_fitted_mean is not None


def test_mean_profile_ridge_is_invariant_to_complete_panel_duplication() -> None:
    informative = np.asarray(
        [
            [[12, 7], [9, 18]],
            [[8, 11], [6, 17]],
            [[15, 5], [13, 19]],
        ],
        dtype=np.int64,
    )
    degenerate = np.asarray([[[0, 0], [4, 6]]], dtype=np.int64)
    tables = np.concatenate((informative, degenerate))
    original = fit_ridge_profiled_poisson_interaction(tables)
    duplicated = fit_ridge_profiled_poisson_interaction(
        np.concatenate((tables, tables))
    )

    np.testing.assert_allclose(duplicated.log_odds, original.log_odds, atol=1e-13)
    np.testing.assert_allclose(
        duplicated.mean_profile_log_likelihood,
        original.mean_profile_log_likelihood,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        duplicated.penalized_objective, original.penalized_objective, atol=1e-13
    )
    np.testing.assert_allclose(
        duplicated.mean_data_information,
        original.mean_data_information,
        atol=1e-13,
    )
    np.testing.assert_array_equal(
        duplicated.informative_table_count, 2 * original.informative_table_count
    )
    np.testing.assert_array_equal(
        duplicated.degenerate_table_count, 2 * original.degenerate_table_count
    )
    np.testing.assert_array_equal(
        duplicated.included_table_count, 2 * original.included_table_count
    )


def test_zero_transport_removes_fitted_ridge_interaction() -> None:
    tables = np.asarray(
        [
            [[90, 38], [38, 90]],
            [[84, 44], [42, 86]],
        ],
        dtype=np.int64,
    )
    fitted = fit_ridge_profiled_poisson_interaction(tables)
    rows = np.asarray([[91.0, 165.0]])
    columns = np.asarray([[128.0, 128.0]])
    transported = reconstruct_poisson_tables(
        fitted.log_odds, rows, columns, transport_scale=0.0
    )
    expected = rows[..., :, None] * columns[..., None, :] / 256.0

    np.testing.assert_allclose(transported.table, expected, atol=1e-13)
    assert transported.transported_log_odds.item() == 0.0


def test_recipient_profile_matches_independent_nuisance_fit_not_fisher_mean() -> None:
    rows = np.asarray([10.0, 14.0])
    columns = np.asarray([12.0, 12.0])
    log_odds = 1.2
    reconstructed = reconstruct_poisson_tables(np.asarray(log_odds), rows, columns)
    independently_profiled = _independent_fixed_interaction_profile(
        log_odds, rows, columns
    )
    fisher_mean = expected_binary_table_from_log_odds(
        log_odds, rows.astype(int), columns.astype(int)
    )

    np.testing.assert_allclose(reconstructed.table, independently_profiled, atol=2e-8)
    assert np.max(np.abs(reconstructed.table - fisher_mean)) > 1e-3
    np.testing.assert_allclose(reconstructed.table.sum(axis=1), rows, atol=1e-12)
    np.testing.assert_allclose(reconstructed.table.sum(axis=0), columns, atol=1e-12)
    assert reconstructed.reconstructed_log_odds.item() == pytest.approx(
        log_odds, abs=1e-12
    )
    assert "IPF-equivalent" in reconstructed.reconstruction


def test_every_degenerate_training_table_is_included_without_changing_score() -> None:
    informative = np.asarray(
        [
            [[7, 3], [2, 8]],
            [[5, 5], [4, 6]],
            [[6, 4], [3, 7]],
        ],
        dtype=np.int64,
    )
    degenerate = np.asarray(
        [
            [[0, 0], [4, 6]],
            [[0, 0], [0, 0]],
        ],
        dtype=np.int64,
    )
    original = fit_poisson_loglinear_interaction(informative)
    augmented = fit_poisson_loglinear_interaction(
        np.concatenate((informative, degenerate))
    )

    np.testing.assert_allclose(augmented.log_odds, original.log_odds, atol=1e-13)
    assert augmented.informative_table_count.item() == 3
    assert augmented.degenerate_table_count.item() == 2
    assert augmented.included_table_count.item() == 5
    degenerate_log_likelihood = float(
        np.sum(xlogy(degenerate, degenerate) - degenerate - gammaln(degenerate + 1.0))
    )
    assert augmented.profile_log_likelihood == pytest.approx(
        original.profile_log_likelihood + degenerate_log_likelihood,
        abs=1e-12,
    )


@pytest.mark.parametrize(
    "tables",
    [
        np.asarray([[[5, 0], [0, 5]], [[7, 0], [0, 3]]]),
        np.asarray([[[0, 5], [5, 0]], [[0, 7], [3, 0]]]),
    ],
)
def test_boundary_interactions_refuse_instead_of_adding_pseudocount(
    tables: np.ndarray,
) -> None:
    with pytest.raises(PoissonLoglinearRefusal, match="boundary or infinite MLE"):
        fit_poisson_loglinear_interaction(tables)


def test_opposite_zero_cell_tables_have_a_finite_unpenalized_common_fit() -> None:
    tables = np.asarray(
        [
            [[5, 0], [0, 5]],
            [[0, 5], [5, 0]],
        ]
    )
    fitted = fit_poisson_loglinear_interaction(tables)

    assert fitted.log_odds.item() == pytest.approx(0.0, abs=1e-14)
    assert fitted.informative_table_count.item() == 2
    assert fitted.lower_boundary_gap.item() == 5
    assert fitted.upper_boundary_gap.item() == 5
    assert fitted.pseudocount == 0.0


def test_machine_zero_profile_score_returns_the_finite_zero_mle() -> None:
    tables = np.asarray(
        [
            [[0, 1], [3, 3]],
            [[3, 0], [3, 1]],
        ],
        dtype=np.int64,
    )

    fitted = fit_poisson_loglinear_interaction(tables)

    assert fitted.lower_boundary_gap.item() == 1.0
    assert fitted.upper_boundary_gap.item() == 1.0
    assert fitted.log_odds.item() == 0.0
    assert fitted.root_iterations.item() == 0
    assert fitted.maximum_scaled_score <= fitted.score_tolerance


def test_unsupported_and_nonfinite_inputs_refuse() -> None:
    degenerate = np.asarray([[[0, 0], [4, 6]], [[3, 7], [0, 0]]])
    with pytest.raises(PoissonLoglinearRefusal, match="no training table"):
        fit_poisson_loglinear_interaction(degenerate)

    with pytest.raises(ValueError, match="integer counts"):
        fit_poisson_loglinear_interaction(np.asarray([[[1.0, np.nan], [2.0, 3.0]]]))
    with pytest.raises(ValueError, match="table total"):
        fit_poisson_loglinear_interaction(
            np.asarray([[[2**53 - 1, 1], [0, 0]]], dtype=np.int64)
        )
    with pytest.raises(ValueError, match="nonempty finite"):
        reconstruct_poisson_tables(
            np.asarray([np.inf]), np.asarray([[2.0, 3.0]]), np.asarray([[2.0, 3.0]])
        )
    with pytest.raises(PoissonLoglinearRefusal, match="finite nonnegative margins"):
        reconstruct_poisson_tables(
            np.asarray([0.5]), np.asarray([[-1.0, 6.0]]), np.asarray([[2.0, 3.0]])
        )


def test_group_labels_preserve_type_identity_and_reject_mixed_nan() -> None:
    table = np.asarray([[[7, 3], [2, 8]]], dtype=np.int64)
    tables = np.repeat(table, 4, axis=0)

    fitted = fit_poisson_loglinear_interaction(tables, ["1", 1, "2", 2])

    assert fitted.group_labels == ("1", 1, "2", 2)
    assert fitted.log_odds.shape == (4,)
    with pytest.raises(ValueError, match="finite non-null scalars"):
        fit_poisson_loglinear_interaction(tables[:2], ["A", np.nan])


def test_group_specific_fit_is_invariant_to_group_names_and_donor_order() -> None:
    tables = np.asarray(
        [
            [[7, 3], [2, 8]],
            [[8, 4], [3, 9]],
            [[4, 7], [8, 5]],
            [[5, 6], [7, 6]],
            [[9, 5], [4, 10]],
            [[10, 6], [5, 11]],
        ],
        dtype=np.int64,
    )
    groups = np.asarray(["donor", "donor", "ami", "ami", "icm", "icm"])
    renamed = np.asarray(["a", "a", "b", "b", "c", "c"])
    original = fit_poisson_loglinear_interaction(tables, groups)
    relabeled = fit_poisson_loglinear_interaction(tables, renamed)
    permutation = np.asarray([4, 0, 2, 5, 1, 3])
    reordered = fit_poisson_loglinear_interaction(
        tables[permutation], groups[permutation]
    )

    np.testing.assert_allclose(original.log_odds, relabeled.log_odds, atol=1e-13)
    original_by_group = dict(zip(original.group_labels, original.log_odds))
    reordered_by_group = dict(zip(reordered.group_labels, reordered.log_odds))
    assert reordered_by_group == pytest.approx(original_by_group, abs=1e-13)
    for label in original.group_labels:
        pooled = fit_poisson_loglinear_interaction(tables[groups == label])
        assert original_by_group[label] == pytest.approx(
            pooled.log_odds.item(), abs=1e-13
        )
    np.testing.assert_array_equal(original.included_table_count, 2)

    two_entity_panel = np.stack((tables, tables[..., ::-1, :]), axis=1)
    two_entity = fit_poisson_loglinear_interaction(two_entity_panel, groups)
    assert two_entity.log_odds.shape == (3, 2)
    np.testing.assert_allclose(
        two_entity.log_odds[:, 1], -two_entity.log_odds[:, 0], atol=1e-12
    )


def test_vectorized_transport_preserves_margins_and_scaled_log_odds() -> None:
    interaction = np.asarray([[0.0, 0.8], [-1.1, 1.7]])
    rows = np.asarray([[[10.0, 14.0], [20.0, 12.0]], [[7.0, 18.0], [16.0, 15.0]]])
    columns = np.asarray([[[12.0, 12.0], [14.0, 18.0]], [[11.0, 14.0], [13.0, 18.0]]])
    fitted = reconstruct_poisson_tables(
        interaction, rows, columns, transport_scale=1.25
    )

    np.testing.assert_allclose(fitted.table.sum(axis=-1), rows, atol=1e-12)
    np.testing.assert_allclose(fitted.table.sum(axis=-2), columns, atol=1e-12)
    np.testing.assert_allclose(
        fitted.reconstructed_log_odds, 1.25 * interaction, atol=1e-12
    )
    assert fitted.maximum_absolute_row_margin_error <= 1e-8
    assert fitted.maximum_absolute_column_margin_error <= 1e-8
    assert fitted.maximum_absolute_log_odds_error <= 1e-8
    assert fitted.minimum_fitted_mean > 0.0

    independence = reconstruct_poisson_tables(
        interaction, rows, columns, transport_scale=0.0
    )
    expected = (
        rows[..., :, None] * columns[..., None, :] / rows.sum(axis=-1)[..., None, None]
    )
    np.testing.assert_allclose(independence.table, expected, atol=1e-13)


def test_degenerate_recipient_margins_return_the_unique_tables() -> None:
    interaction = np.asarray([0.7, -0.4, 1.2, 0.3])
    rows = np.asarray(
        [
            [10.0, 14.0],
            [0.0, 24.0],
            [24.0, 0.0],
            [0.0, 0.0],
        ]
    )
    columns = np.asarray(
        [
            [12.0, 12.0],
            [9.0, 15.0],
            [0.0, 24.0],
            [0.0, 0.0],
        ]
    )

    fitted = reconstruct_poisson_tables(interaction, rows, columns)

    assert fitted.informative_margin_mask.tolist() == [True, False, False, False]
    assert fitted.informative_margin_count == 1
    assert fitted.degenerate_margin_count == 3
    assert np.isnan(fitted.reconstructed_log_odds[1:]).all()
    np.testing.assert_array_equal(fitted.root_iterations[1:], 0)
    np.testing.assert_allclose(fitted.table.sum(axis=-1), rows, atol=1e-12)
    np.testing.assert_allclose(fitted.table.sum(axis=-2), columns, atol=1e-12)
    np.testing.assert_array_equal(fitted.table[1], [[0.0, 0.0], [9.0, 15.0]])
    np.testing.assert_array_equal(fitted.table[2], [[0.0, 24.0], [0.0, 0.0]])
    np.testing.assert_array_equal(fitted.table[3], np.zeros((2, 2)))
    assert fitted.maximum_absolute_log_odds_error <= 1e-8
    assert fitted.minimum_fitted_mean == 0.0

    heart = reconstruct_poisson_tables(
        np.asarray([0.9]),
        np.asarray([[512.0, 0.0]]),
        np.asarray([[256.0, 256.0]]),
    )
    assert (
        gse217494_heart_core.entity_deviance(
            heart.table.astype(np.int64), heart.table
        ).item()
        == 0.0
    )


def test_complete_panel_duplication_scales_evidence_not_interaction() -> None:
    tables = np.asarray(
        [
            [[12, 7], [9, 18]],
            [[8, 11], [6, 17]],
            [[15, 5], [13, 19]],
        ],
        dtype=np.int64,
    )
    original = fit_poisson_loglinear_interaction(tables)
    duplicated = fit_poisson_loglinear_interaction(np.repeat(tables, 2, axis=0))

    np.testing.assert_allclose(duplicated.log_odds, original.log_odds, atol=1e-13)
    assert duplicated.profile_log_likelihood == pytest.approx(
        2.0 * original.profile_log_likelihood, abs=1e-11
    )
    np.testing.assert_allclose(
        duplicated.data_information, 2.0 * original.data_information, atol=1e-12
    )
    np.testing.assert_array_equal(
        duplicated.informative_table_count, 2 * original.informative_table_count
    )
    np.testing.assert_array_equal(
        duplicated.included_table_count, 2 * original.included_table_count
    )
    np.testing.assert_allclose(
        duplicated.lower_boundary_gap, 2.0 * original.lower_boundary_gap
    )
    np.testing.assert_allclose(
        duplicated.upper_boundary_gap, 2.0 * original.upper_boundary_gap
    )
