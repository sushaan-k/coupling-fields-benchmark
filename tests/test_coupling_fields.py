import numpy as np
import pytest

from mapreg.coupling_fields import (
    CouplingFieldRefusal,
    association_coordinates,
    association_field,
    conditional_association_coordinates,
    factorial_association_contrast,
    field_from_coordinates,
    fit_structured_coupling_fields,
    helmert_contrast,
    inverse_permutation_variance_weights,
    normalized_hypergraph_laplacian,
)
from mapreg.classical_residuals import (
    conditional_poisson_residual_coordinates,
    conditional_poisson_residuals,
    poisson_independence_residuals,
)


def test_helmert_basis_is_orthonormal_and_centered():
    basis = helmert_contrast(5)
    np.testing.assert_allclose(basis.T @ basis, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(basis.sum(axis=0), 0.0, atol=1e-12)


def test_association_is_exactly_invariant_to_separable_tilts():
    table = np.array([[0.12, 0.08, 0.05], [0.07, 0.18, 0.10], [0.06, 0.09, 0.25]])
    row_tilt = np.array([0.4, 1.7, 3.1])
    column_tilt = np.array([2.2, 0.6, 1.4])
    tilted = row_tilt[:, None] * table * column_tilt[None, :]
    tilted /= tilted.sum()
    np.testing.assert_allclose(
        association_field(table), association_field(tilted), atol=1e-12
    )


def test_cycle_coordinates_round_trip_and_have_zero_marginal_sums():
    rng = np.random.default_rng(4)
    field = association_field(rng.uniform(0.1, 2.0, size=(4, 3)))
    reconstructed = field_from_coordinates(association_coordinates(field))
    np.testing.assert_allclose(reconstructed, field, atol=1e-12)
    np.testing.assert_allclose(reconstructed.sum(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(reconstructed.sum(axis=1), 0.0, atol=1e-12)


def test_factorial_contrast_removes_context_specific_marginal_tilts():
    base = np.array([[0.35, 0.15], [0.10, 0.40]])
    tables = np.empty((2, 2, 2, 2))
    for perturbation in range(2):
        for context in range(2):
            row = np.array([1.0 + perturbation, 0.7 + context])
            column = np.array([0.5 + context, 1.4 + perturbation])
            tables[perturbation, context] = row[:, None] * base * column[None, :]
    np.testing.assert_allclose(
        factorial_association_contrast(tables, 1, 0, 1, 0), 0.0, atol=1e-12
    )


def test_hypergraph_laplacian_is_psd_and_leaves_isolate_unpenalized():
    incidence = np.array([[1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
    laplacian = normalized_hypergraph_laplacian(incidence)
    assert np.linalg.eigvalsh(laplacian).min() >= -1e-12
    np.testing.assert_allclose(laplacian[3], 0.0)
    np.testing.assert_allclose(laplacian[:, 3], 0.0)


def test_zero_penalty_fit_is_identity():
    rng = np.random.default_rng(8)
    values = rng.normal(size=(7, 4))
    fit = fit_structured_coupling_fields(values)
    np.testing.assert_array_equal(fit.coefficient, values)
    assert fit.converged
    assert fit.iterations == 0


def test_inverse_permutation_variance_weights_are_normalized_and_clipped():
    variance = np.array([[1e6, 4.0], [2.0, 1e-6]])
    precision = 1.0 / variance
    expected = np.clip(precision / np.median(precision), 0.05, 20.0)
    np.testing.assert_allclose(
        inverse_permutation_variance_weights(variance), expected
    )


@pytest.mark.parametrize(
    "variance",
    [np.array([]), np.array([0.0, 1.0]), np.array([-1.0]), np.array([np.nan])],
)
def test_inverse_permutation_variance_weights_reject_invalid_values(variance):
    with pytest.raises(ValueError):
        inverse_permutation_variance_weights(variance)


def test_graph_penalty_brings_connected_entities_together():
    values = np.array([[2.0, -1.0], [-2.0, 1.0], [5.0, 5.0]])
    incidence = np.array([[1.0], [1.0], [0.0]])
    laplacian = normalized_hypergraph_laplacian(incidence)
    fit = fit_structured_coupling_fields(
        values,
        graph_laplacian=laplacian,
        graph_penalty=50.0,
        tolerance=1e-10,
    )
    assert np.linalg.norm(fit.coefficient[0] - fit.coefficient[1]) < 0.1
    np.testing.assert_allclose(fit.coefficient[2], values[2], atol=1e-8)


def test_nuclear_penalty_reduces_rank():
    values = np.diag([4.0, 1.0, 0.2])
    fit = fit_structured_coupling_fields(
        values, nuclear_penalty=0.5, tolerance=1e-12
    )
    assert fit.effective_rank == 2
    np.testing.assert_allclose(fit.singular_values[:2], [3.5, 0.5], atol=1e-8)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"observation_weight": np.array([[1.0, 0.0]])},
        {"nuclear_penalty": -1.0},
        {"graph_penalty": 1.0},
        {"graph_laplacian": np.array([[0.0, 1.0], [0.0, 0.0]])},
    ],
)
def test_structured_fit_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        fit_structured_coupling_fields(np.ones((1, 2)), **kwargs)


def test_sparse_table_requires_pseudocount():
    with pytest.raises(CouplingFieldRefusal):
        association_field(np.eye(2))
    assert np.isfinite(association_field(np.eye(2), pseudocount=0.5)).all()


@pytest.mark.parametrize("residual", ["pearson", "deviance"])
def test_poisson_independence_residuals_vanish_for_exact_independence(residual):
    table = np.array([[10.0, 20.0], [20.0, 40.0]])
    np.testing.assert_allclose(
        poisson_independence_residuals(table, residual=residual), 0.0, atol=1e-12
    )


@pytest.mark.parametrize("residual", ["pearson", "deviance"])
def test_poisson_independence_residuals_are_finite_for_sparse_tables(residual):
    values = poisson_independence_residuals(
        np.array([[8.0, 0.0, 2.0], [0.0, 0.0, 0.0], [1.0, 5.0, 0.0]]),
        residual=residual,
    )
    assert values.shape == (3, 3)
    assert np.isfinite(values).all()
    np.testing.assert_array_equal(values[1], 0.0)


def test_poisson_residual_coordinates_use_identical_permutation_protocol():
    first = np.tile(np.arange(3), 20)
    second = np.roll(first, 1)
    draw_count = 8
    seed = 14
    estimate = conditional_poisson_residual_coordinates(
        first,
        second,
        first_levels=3,
        second_levels=3,
        residual="deviance",
        permutations=draw_count,
        seed=seed,
    )

    def coordinates(permuted_second):
        table = np.bincount(
            first * 3 + permuted_second, minlength=9
        ).reshape(3, 3)
        return association_coordinates(
            poisson_independence_residuals(table, residual="deviance")
        )

    generator = np.random.default_rng(seed)
    null = np.asarray(
        [coordinates(second[generator.permutation(len(first))]) for _ in range(9)]
    )
    reference = null[1:].mean(axis=0)
    assert estimate.coordinates.shape == (2, 2)
    np.testing.assert_allclose(estimate.null_mean_coordinates, reference)
    np.testing.assert_allclose(estimate.coordinates, coordinates(second) - reference)
    np.testing.assert_allclose(estimate.destroyed_coordinates, null[0] - reference)
    np.testing.assert_allclose(
        estimate.null_variance_coordinates,
        np.var(null[1:], axis=0, ddof=1) * (1.0 + 1.0 / draw_count),
    )


def test_full_poisson_residual_baseline_preserves_standard_table_shape():
    first = np.tile(np.arange(3), 20)
    second = np.roll(first, 1)
    estimate = conditional_poisson_residuals(
        first,
        second,
        first_levels=3,
        second_levels=3,
        residual="pearson",
        permutations=8,
        seed=19,
    )
    assert estimate.coordinates.shape == (3, 3)
    assert estimate.destroyed_coordinates.shape == (3, 3)
    assert np.isfinite(estimate.null_variance_coordinates).all()


def test_conditional_centering_is_reproducible_and_preserves_shape():
    rng = np.random.default_rng(19)
    first = rng.choice(3, size=600, p=[0.8, 0.15, 0.05])
    second = rng.choice(4, size=600, p=[0.05, 0.15, 0.3, 0.5])
    one = conditional_association_coordinates(
        first,
        second,
        first_levels=3,
        second_levels=4,
        permutations=32,
        seed=9,
    )
    two = conditional_association_coordinates(
        first,
        second,
        first_levels=3,
        second_levels=4,
        permutations=32,
        seed=9,
    )
    np.testing.assert_array_equal(one.coordinates, two.coordinates)
    assert one.coordinates.shape == (2, 3)
    assert one.destroyed_coordinates.shape == (2, 3)
    assert one.null_variance_coordinates.shape == (2, 3)
    assert np.all(one.null_variance_coordinates >= 0.0)
    assert one.permutations == 32


def test_conditional_centering_uses_reference_permutations_only():
    first = np.tile(np.arange(3), 20)
    second = np.roll(first, 1)
    draw_count = 8
    seed = 14
    estimate = conditional_association_coordinates(
        first,
        second,
        first_levels=3,
        second_levels=3,
        permutations=draw_count,
        seed=seed,
    )

    def coordinates(permuted_second):
        table = np.bincount(
            first * 3 + permuted_second, minlength=9
        ).reshape(3, 3)
        return association_coordinates(association_field(table, pseudocount=0.5))

    generator = np.random.default_rng(seed)
    null = np.asarray(
        [coordinates(second[generator.permutation(len(first))]) for _ in range(9)]
    )
    reference = null[1:].mean(axis=0)
    np.testing.assert_allclose(estimate.null_mean_coordinates, reference)
    np.testing.assert_allclose(
        estimate.coordinates, coordinates(second) - reference
    )
    np.testing.assert_allclose(estimate.destroyed_coordinates, null[0] - reference)
    np.testing.assert_allclose(
        estimate.null_variance_coordinates,
        np.var(null[1:], axis=0, ddof=1) * (1.0 + 1.0 / draw_count),
    )


def test_conditional_centering_retains_pairing_signal_but_not_destroyed_links():
    rng = np.random.default_rng(23)
    first = rng.integers(0, 3, size=3_000)
    second = first.copy()
    flip = rng.random(len(first)) < 0.2
    second[flip] = rng.integers(0, 3, size=flip.sum())
    estimate = conditional_association_coordinates(
        first,
        second,
        first_levels=3,
        second_levels=3,
        permutations=64,
        seed=11,
    )
    assert np.linalg.norm(estimate.coordinates) > 10 * np.linalg.norm(
        estimate.destroyed_coordinates
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"first_levels": 1, "second_levels": 2},
        {"first_levels": 2, "second_levels": 2, "permutations": 1},
    ],
)
def test_conditional_centering_rejects_invalid_contract(kwargs):
    with pytest.raises(ValueError):
        conditional_association_coordinates(
            np.array([0, 1]), np.array([1, 0]), **kwargs
        )
