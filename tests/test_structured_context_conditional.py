import math

import numpy as np
import pytest

from mapreg.context_conditional_coupling import fit_context_conditional_log_odds
from mapreg.heterogeneity_adaptive_coupling import product_hypergraph_laplacian
from mapreg.structured_context_conditional import (
    CouplingEstimationRefusal,
    _validated_entity_laplacian,
    evaluate_structured_context_conditional_log_odds,
    fit_structured_context_conditional_log_odds,
)


def _chain_laplacian(entities: int) -> np.ndarray:
    adjacency = np.diag(np.ones(entities - 1), 1)
    adjacency += adjacency.T
    return np.diag(adjacency.sum(axis=1)) - adjacency


def _tables(entities: int = 4) -> tuple[np.ndarray, np.ndarray]:
    donors = 6
    contexts = np.column_stack((np.ones(donors), np.linspace(-1.0, 1.0, donors)))
    tables = np.empty((donors, entities, 2, 2), dtype=int)
    for donor in range(donors):
        for entity in range(entities):
            diagonal = 5 + ((donor + 2 * entity) % 10)
            tables[donor, entity] = [
                [diagonal, 20 - diagonal],
                [20 - diagonal, diagonal],
            ]
    return tables, contexts


def _arguments(entities: int) -> dict[str, object]:
    return {
        "graph_laplacian": _chain_laplacian(entities),
        "graph_penalty": 0.5,
        "donor_deviation_penalty": 2.5,
        "coefficient_ridge_penalty": np.array([0.3, 0.7]),
        "tolerance": 1e-10,
    }


def test_exact_fixed_margin_objective_and_score_match_enumeration() -> None:
    table = np.array([[[[7, 3], [2, 8]]]])
    contexts = np.array([[1.0, -0.5]])
    coefficient = np.array([[0.4], [-0.2]])
    deviation = np.array([[0.3]])
    ridge = np.array([0.6, 0.9])
    deviation_penalty = 1.7
    evaluation = evaluate_structured_context_conditional_log_odds(
        coefficient,
        deviation,
        table,
        contexts,
        donor_deviation_penalty=deviation_penalty,
        coefficient_ridge_penalty=ridge,
    )

    row_zero = int(table[0, 0, 0].sum())
    column_zero = int(table[0, 0, :, 0].sum())
    total = int(table.sum())
    feasible = np.arange(
        max(0, row_zero + column_zero - total),
        min(row_zero, column_zero) + 1,
        dtype=float,
    )
    weights = np.array(
        [
            math.comb(column_zero, int(cell))
            * math.comb(total - column_zero, row_zero - int(cell))
            for cell in feasible
        ],
        dtype=float,
    )
    weights /= weights.sum()
    theta = float(contexts[0] @ coefficient[:, 0] + deviation[0, 0])
    centered = feasible - table[0, 0, 0, 0]
    mass = weights * np.exp(centered * theta)
    partition = float(mass.sum())
    probability = mass / partition
    score = float(probability @ centered)
    precision = float(probability @ np.square(centered - score))
    expected_objective = math.log(partition)
    expected_objective += 0.5 * deviation_penalty * deviation[0, 0] ** 2
    expected_objective += 0.5 * float(ridge @ np.square(coefficient[:, 0]))

    assert evaluation.objective == pytest.approx(expected_objective)
    np.testing.assert_allclose(
        evaluation.coefficient_gradient[:, 0],
        contexts[0] * score + ridge * coefficient[:, 0],
    )
    assert evaluation.donor_deviation_gradient[0, 0] == pytest.approx(
        score + deviation_penalty * deviation[0, 0]
    )
    assert evaluation.donor_data_precision[0, 0] == pytest.approx(precision)
    assert feasible[0] >= 0
    assert feasible[-1] <= min(row_zero, column_zero)


def test_structured_gradient_matches_finite_differences() -> None:
    tables, contexts = _tables(3)
    coefficient = np.array([[0.2, -0.1, 0.4], [-0.3, 0.25, 0.1]])
    deviation = np.linspace(-0.2, 0.2, 18).reshape(6, 3)
    arguments = {
        "graph_laplacian": _chain_laplacian(3),
        "graph_penalty": 0.8,
        "donor_deviation_penalty": 1.4,
        "coefficient_ridge_penalty": np.array([0.25, 0.65]),
    }
    evaluation = evaluate_structured_context_conditional_log_odds(
        coefficient, deviation, tables, contexts, **arguments
    )
    epsilon = 1e-6

    for index in np.ndindex(coefficient.shape):
        positive = coefficient.copy()
        negative = coefficient.copy()
        positive[index] += epsilon
        negative[index] -= epsilon
        finite_difference = (
            evaluate_structured_context_conditional_log_odds(
                positive, deviation, tables, contexts, **arguments
            ).objective
            - evaluate_structured_context_conditional_log_odds(
                negative, deviation, tables, contexts, **arguments
            ).objective
        ) / (2.0 * epsilon)
        assert finite_difference == pytest.approx(
            evaluation.coefficient_gradient[index], abs=2e-8
        )

    for index in np.ndindex(deviation.shape):
        positive = deviation.copy()
        negative = deviation.copy()
        positive[index] += epsilon
        negative[index] -= epsilon
        finite_difference = (
            evaluate_structured_context_conditional_log_odds(
                coefficient, positive, tables, contexts, **arguments
            ).objective
            - evaluate_structured_context_conditional_log_odds(
                coefficient, negative, tables, contexts, **arguments
            ).objective
        ) / (2.0 * epsilon)
        assert finite_difference == pytest.approx(
            evaluation.donor_deviation_gradient[index], abs=2e-8
        )


def test_zero_graph_reduces_to_independent_context_estimator() -> None:
    tables, contexts = _tables()
    arguments = {
        "donor_deviation_penalty": 2.5,
        "coefficient_ridge_penalty": np.array([0.3, 0.7]),
        "tolerance": 1e-10,
    }
    independent = fit_context_conditional_log_odds(tables, contexts, **arguments)
    structured = fit_structured_context_conditional_log_odds(
        tables,
        contexts,
        graph_laplacian=_chain_laplacian(4),
        graph_penalty=0.0,
        **arguments,
    )

    np.testing.assert_allclose(
        structured.coefficient, independent.coefficient, atol=5e-10
    )
    np.testing.assert_allclose(
        structured.donor_deviation, independent.donor_deviation, atol=5e-10
    )
    np.testing.assert_allclose(
        structured.donor_log_odds, independent.donor_log_odds, atol=5e-10
    )
    assert structured.objective == pytest.approx(independent.objective, abs=1e-12)


def test_complete_panel_duplication_preserves_graph_structured_fit() -> None:
    tables, contexts = _tables()
    arguments = _arguments(4)
    original = fit_structured_context_conditional_log_odds(
        tables, contexts, **arguments
    )
    duplicated = fit_structured_context_conditional_log_odds(
        np.concatenate((tables, tables)),
        np.concatenate((contexts, contexts)),
        **arguments,
    )

    np.testing.assert_allclose(duplicated.coefficient, original.coefficient, atol=2e-12)
    np.testing.assert_allclose(
        duplicated.donor_deviation[: len(contexts)],
        original.donor_deviation,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        duplicated.donor_deviation[len(contexts) :],
        original.donor_deviation,
        atol=2e-12,
    )
    assert duplicated.objective == pytest.approx(original.objective, abs=2e-12)
    assert duplicated.schur_condition_number == pytest.approx(
        original.schur_condition_number, abs=2e-12
    )


def test_entity_permutation_and_binary_label_swap_are_equivariant() -> None:
    tables, contexts = _tables()
    arguments = _arguments(4)
    fitted = fit_structured_context_conditional_log_odds(tables, contexts, **arguments)

    permutation = np.array([2, 0, 3, 1])
    laplacian = _chain_laplacian(4)
    permuted = fit_structured_context_conditional_log_odds(
        tables[:, permutation],
        contexts,
        graph_laplacian=laplacian[np.ix_(permutation, permutation)],
        graph_penalty=0.5,
        donor_deviation_penalty=2.5,
        coefficient_ridge_penalty=np.array([0.3, 0.7]),
        tolerance=1e-10,
    )
    np.testing.assert_allclose(
        permuted.coefficient, fitted.coefficient[:, permutation], atol=2e-12
    )
    np.testing.assert_allclose(
        permuted.donor_deviation,
        fitted.donor_deviation[:, permutation],
        atol=2e-12,
    )

    row_swapped = fit_structured_context_conditional_log_odds(
        tables[..., ::-1, :], contexts, **arguments
    )
    np.testing.assert_allclose(row_swapped.coefficient, -fitted.coefficient, atol=2e-12)
    np.testing.assert_allclose(
        row_swapped.donor_deviation, -fitted.donor_deviation, atol=2e-12
    )
    np.testing.assert_allclose(
        row_swapped.donor_log_odds, -fitted.donor_log_odds, atol=2e-12
    )


def test_one_hot_context_column_permutation_is_equivariant() -> None:
    base_tables, _ = _tables()
    tables = np.concatenate((base_tables, base_tables[:2]))
    contexts = np.eye(4)[np.repeat(np.arange(4), 2)]
    common = {
        "graph_laplacian": _chain_laplacian(4),
        "graph_penalty": 0.5,
        "donor_deviation_penalty": 2.5,
        "coefficient_ridge_penalty": 0.3,
        "tolerance": 1e-10,
    }
    fitted = fit_structured_context_conditional_log_odds(tables, contexts, **common)
    permutation = np.array([2, 0, 3, 1])
    permuted = fit_structured_context_conditional_log_odds(
        tables, contexts[:, permutation], **common
    )

    np.testing.assert_allclose(
        permuted.coefficient, fitted.coefficient[permutation], atol=2e-12
    )
    np.testing.assert_allclose(
        permuted.context_log_odds, fitted.context_log_odds, atol=2e-12
    )
    np.testing.assert_allclose(
        permuted.donor_deviation, fitted.donor_deviation, atol=2e-12
    )


def test_laplacian_validation_and_product_hypergraph_path() -> None:
    laplacian = _chain_laplacian(4)
    validated, residual, minimum, maximum, nullity = _validated_entity_laplacian(
        laplacian, 4
    )
    np.testing.assert_allclose(validated, laplacian)
    np.testing.assert_allclose(validated @ np.ones(4), 0.0, atol=1e-14)
    assert residual == 0.0
    assert minimum >= -1e-14
    assert maximum > 0.0
    assert nullity == 1

    with pytest.raises(ValueError, match="symmetric"):
        _validated_entity_laplacian(laplacian + np.triu(np.ones((4, 4)), 1), 4)
    with pytest.raises(ValueError, match="positive semidefinite"):
        _validated_entity_laplacian(np.diag([0.0, 1.0, 1.0, -1.0]), 4)
    with pytest.raises(ValueError, match="nonempty nullspace"):
        _validated_entity_laplacian(np.eye(4), 4)

    flat_tables, contexts = _tables()
    product_tables = flat_tables.reshape(6, 2, 2, 2, 2)
    first = np.array([[1.0, 0.0], [1.0, 1.0]])
    second = np.array([[1.0], [1.0]])
    fitted = fit_structured_context_conditional_log_odds(
        product_tables,
        contexts,
        first_incidence=first,
        second_incidence=second,
        graph_penalty=0.4,
        donor_deviation_penalty=2.0,
        coefficient_ridge_penalty=np.array([0.2, 0.6]),
    )
    np.testing.assert_allclose(
        fitted.graph_laplacian, product_hypergraph_laplacian(first, second)
    )
    assert fitted.graph_source == "product_hypergraph"
    assert fitted.graph_nullity >= 1
    assert fitted.graph_minimum_eigenvalue >= -1e-12


def test_certificates_recompute_from_returned_fit() -> None:
    tables, contexts = _tables()
    arguments = _arguments(4)
    fit = fit_structured_context_conditional_log_odds(tables, contexts, **arguments)
    evaluation = evaluate_structured_context_conditional_log_odds(
        fit.coefficient,
        fit.donor_deviation,
        tables,
        contexts,
        graph_laplacian=arguments["graph_laplacian"],
        graph_penalty=float(arguments["graph_penalty"]),
        donor_deviation_penalty=float(arguments["donor_deviation_penalty"]),
        coefficient_ridge_penalty=arguments["coefficient_ridge_penalty"],
    )
    np.testing.assert_allclose(
        evaluation.coefficient_gradient, fit.coefficient_gradient
    )
    np.testing.assert_allclose(
        evaluation.donor_deviation_gradient, fit.donor_deviation_gradient
    )
    assert evaluation.objective == pytest.approx(fit.objective)

    entities = tables.shape[1]
    contexts_count = contexts.shape[1]
    ridge = np.asarray(arguments["coefficient_ridge_penalty"])
    schur = 0.5 * np.kron(_chain_laplacian(entities), np.eye(contexts_count))
    diagonal = np.arange(entities * contexts_count)
    schur[diagonal, diagonal] += np.tile(ridge, entities)
    donor_curvature = np.empty_like(fit.donor_data_precision)
    for entity in range(entities):
        precision = fit.donor_data_precision[:, entity]
        curvature = precision + 2.5 / len(contexts)
        donor_curvature[:, entity] = curvature
        transmitted = precision * (2.5 / len(contexts)) / curvature
        block = slice(entity * contexts_count, (entity + 1) * contexts_count)
        schur[block, block] += (contexts.T * transmitted) @ contexts
    eigenvalues = np.linalg.eigvalsh(schur)
    assert fit.minimum_schur_eigenvalue == pytest.approx(eigenvalues[0])
    assert fit.maximum_schur_eigenvalue == pytest.approx(eigenvalues[-1])
    assert fit.schur_condition_number == pytest.approx(eigenvalues[-1] / eigenvalues[0])
    assert fit.minimum_donor_curvature == pytest.approx(donor_curvature.min())
    assert fit.maximum_donor_curvature == pytest.approx(donor_curvature.max())
    assert fit.donor_curvature_condition_number == pytest.approx(
        donor_curvature.max() / donor_curvature.min()
    )
    assert fit.converged
    assert fit.scaled_gradient_norm <= fit.gradient_tolerance


def test_invalid_margins_unsupported_entities_and_nonconvergence_refuse() -> None:
    tables, contexts = _tables()
    with pytest.raises(ValueError, match="nonnegative integer"):
        fit_structured_context_conditional_log_odds(
            tables.astype(float) + 0.25, contexts
        )
    invalid = tables.copy()
    invalid[0, 0, 0, 0] = -1
    with pytest.raises(ValueError, match="nonnegative integer"):
        fit_structured_context_conditional_log_odds(invalid, contexts)
    with pytest.raises(ValueError, match="entity axes must be nonempty"):
        fit_structured_context_conditional_log_odds(
            np.empty((2, 0, 2, 2), dtype=int), np.ones((2, 1))
        )
    with pytest.raises(ValueError, match="entity axes must be nonempty"):
        fit_structured_context_conditional_log_odds(
            np.empty((2, 2, 0, 2, 2), dtype=int), np.ones((2, 1))
        )
    with pytest.raises(ValueError, match="boolean or binary"):
        fit_structured_context_conditional_log_odds(
            tables, contexts, support_mask=np.full(tables.shape[:-2], np.nan)
        )
    with pytest.raises(ValueError, match="boolean or binary"):
        fit_structured_context_conditional_log_odds(
            tables, contexts, support_mask=np.full(tables.shape[:-2], 0.2)
        )
    with pytest.raises(ValueError, match="nonzero graph operator"):
        fit_structured_context_conditional_log_odds(
            tables,
            contexts,
            graph_laplacian=np.zeros((tables.shape[1], tables.shape[1])),
            graph_penalty=0.5,
        )

    unsupported = tables.copy()
    unsupported[:, -1] = np.array([[10, 0], [0, 0]])
    with pytest.raises(CouplingEstimationRefusal, match="too few informative donors"):
        fit_structured_context_conditional_log_odds(
            unsupported,
            contexts,
            graph_laplacian=_chain_laplacian(4),
            graph_penalty=1.0,
        )
    with pytest.raises(CouplingEstimationRefusal, match="gradient certificate"):
        fit_structured_context_conditional_log_odds(
            tables,
            contexts,
            graph_laplacian=_chain_laplacian(4),
            graph_penalty=0.5,
            maximum_iterations=1,
            tolerance=1e-15,
        )
    with pytest.raises(CouplingEstimationRefusal, match="condition-number limit"):
        fit_structured_context_conditional_log_odds(
            tables,
            contexts,
            graph_laplacian=_chain_laplacian(4),
            graph_penalty=0.5,
            maximum_condition_number=1.01,
        )


def test_masked_unequal_support_and_boundary_tables_remain_finite() -> None:
    tables, contexts = _tables()
    tables = tables.copy()
    tables[0, 0] = [[0, 20], [20, 0]]
    tables[1, 0] = [[20, 0], [0, 20]]
    mask = np.ones(tables.shape[:-2], dtype=bool)
    mask[:2, 1] = False
    fitted = fit_structured_context_conditional_log_odds(
        tables,
        contexts,
        support_mask=mask,
        graph_laplacian=_chain_laplacian(tables.shape[1]),
        graph_penalty=0.5,
        donor_deviation_penalty=2.5,
        coefficient_ridge_penalty=np.array([0.3, 0.7]),
        minimum_informative_donors=4,
        tolerance=1e-10,
    )

    assert np.isfinite(fitted.objective)
    assert fitted.support_count[1] == 4
    assert not fitted.donor_support[:2, 1].any()
    assert fitted.scaled_gradient_norm <= fitted.gradient_tolerance


def test_positive_graph_penalty_reduces_coefficient_roughness() -> None:
    tables, contexts = _tables()
    laplacian = _chain_laplacian(4)
    common = {
        "graph_laplacian": laplacian,
        "donor_deviation_penalty": 2.5,
        "coefficient_ridge_penalty": np.array([0.3, 0.7]),
        "tolerance": 1e-10,
    }
    unstructured = fit_structured_context_conditional_log_odds(
        tables, contexts, graph_penalty=0.0, **common
    )
    structured = fit_structured_context_conditional_log_odds(
        tables, contexts, graph_penalty=2.0, **common
    )

    def roughness(coefficient: np.ndarray) -> float:
        return float(sum(row @ laplacian @ row for row in np.asarray(coefficient)))

    assert roughness(structured.coefficient) < roughness(unstructured.coefficient)
