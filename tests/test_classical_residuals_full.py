import numpy as np

from mapreg.classical_residuals import (
    conditional_poisson_residuals,
    poisson_independence_residuals,
)


def test_full_conditional_residuals_retain_all_table_entries():
    first = np.tile(np.arange(3), 20)
    second = np.roll(first, 1)
    estimate = conditional_poisson_residuals(
        first,
        second,
        first_levels=3,
        second_levels=3,
        residual="pearson",
        permutations=8,
        seed=14,
    )
    assert estimate.coordinates.shape == (3, 3)
    assert estimate.destroyed_coordinates.shape == (3, 3)


def test_full_conditional_residuals_use_identical_fixed_margin_permutations():
    first = np.tile(np.arange(3), 20)
    second = np.roll(first, 1)
    generator = np.random.default_rng(14)

    def residual(permuted):
        table = np.bincount(first * 3 + permuted, minlength=9).reshape(3, 3)
        return poisson_independence_residuals(table, residual="deviance")

    null = np.asarray(
        [residual(second[generator.permutation(len(first))]) for _ in range(9)]
    )
    estimate = conditional_poisson_residuals(
        first,
        second,
        first_levels=3,
        second_levels=3,
        residual="deviance",
        permutations=8,
        seed=14,
    )
    reference = null[1:].mean(axis=0)
    np.testing.assert_allclose(estimate.coordinates, residual(second) - reference)
    np.testing.assert_allclose(estimate.destroyed_coordinates, null[0] - reference)
    np.testing.assert_allclose(
        estimate.null_variance_coordinates,
        np.var(null[1:], axis=0, ddof=1) * 1.125,
    )
