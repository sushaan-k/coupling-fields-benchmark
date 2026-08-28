"""Classical log-linear residual comparators for paired finite-state assays."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .coupling_fields import ConditionalAssociationEstimate, association_coordinates


def poisson_independence_residuals(
    table: np.ndarray,
    *,
    residual: Literal["deviance", "pearson"] = "deviance",
) -> np.ndarray:
    """Return residuals from the row-plus-column Poisson independence model."""

    values = np.asarray(table, dtype=float)
    if values.ndim < 2 or min(values.shape[-2:]) < 2:
        raise ValueError("table must end in two state axes of size at least two")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("table must be finite and nonnegative")
    if residual not in {"deviance", "pearson"}:
        raise ValueError("residual must be 'deviance' or 'pearson'")

    total = values.sum(axis=(-2, -1), keepdims=True)
    if np.any(total <= 0.0):
        raise ValueError("every table must have positive total count")
    expected = (
        values.sum(axis=-1, keepdims=True)
        * values.sum(axis=-2, keepdims=True)
        / total
    )
    supported = expected > 0.0
    if residual == "pearson":
        return np.divide(
            values - expected,
            np.sqrt(expected),
            out=np.zeros_like(values),
            where=supported,
        )

    log_ratio = np.zeros_like(values)
    positive = (values > 0.0) & supported
    np.log(
        np.divide(values, expected, out=np.ones_like(values), where=positive),
        out=log_ratio,
        where=positive,
    )
    contribution = np.maximum(values * log_ratio - (values - expected), 0.0)
    return np.sign(values - expected) * np.sqrt(2.0 * contribution)


def _conditional_poisson_residual_estimate(
    first_state: np.ndarray,
    second_state: np.ndarray,
    *,
    first_levels: int,
    second_levels: int,
    residual: Literal["deviance", "pearson"] = "deviance",
    permutations: int = 64,
    seed: int = 0,
    project: bool,
) -> ConditionalAssociationEstimate:
    first = np.asarray(first_state)
    second = np.asarray(second_state)
    row_count = int(first_levels)
    column_count = int(second_levels)
    draw_count = int(permutations)
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
        raise ValueError("first_state and second_state must be equal-length vectors")
    if len(first) < 2:
        raise ValueError("at least two paired observations are required")
    if row_count < 2 or column_count < 2:
        raise ValueError("each assay must have at least two states")
    if draw_count < 2:
        raise ValueError("at least two null permutations are required")
    if not (
        np.issubdtype(first.dtype, np.integer)
        and np.issubdtype(second.dtype, np.integer)
    ):
        raise ValueError("state vectors must contain integer labels")
    if (
        np.any(first < 0)
        or np.any(first >= row_count)
        or np.any(second < 0)
        or np.any(second >= column_count)
    ):
        raise ValueError("state labels are outside the declared level range")

    def statistic(permuted_second: np.ndarray) -> np.ndarray:
        table = np.bincount(
            first * column_count + permuted_second,
            minlength=row_count * column_count,
        ).reshape(row_count, column_count)
        values = poisson_independence_residuals(table, residual=residual)
        return association_coordinates(values) if project else values

    generator = np.random.default_rng(seed)
    null_values = np.asarray(
        [
            statistic(second[generator.permutation(len(first))])
            for _ in range(draw_count + 1)
        ]
    )
    reference = null_values[1:].mean(axis=0)
    observed = statistic(second)
    null_variance = np.var(null_values[1:], axis=0, ddof=1) * (
        1.0 + 1.0 / draw_count
    )
    return ConditionalAssociationEstimate(
        coordinates=observed - reference,
        observed_coordinates=observed,
        null_mean_coordinates=reference,
        null_variance_coordinates=null_variance,
        destroyed_coordinates=null_values[0] - reference,
        permutations=draw_count,
    )


def conditional_poisson_residual_coordinates(
    first_state: np.ndarray,
    second_state: np.ndarray,
    *,
    first_levels: int,
    second_levels: int,
    residual: Literal["deviance", "pearson"] = "deviance",
    permutations: int = 64,
    seed: int = 0,
) -> ConditionalAssociationEstimate:
    """Estimate projected residual coordinates for dimension-matched sensitivity."""

    return _conditional_poisson_residual_estimate(
        first_state,
        second_state,
        first_levels=first_levels,
        second_levels=second_levels,
        residual=residual,
        permutations=permutations,
        seed=seed,
        project=True,
    )


def conditional_poisson_residuals(
    first_state: np.ndarray,
    second_state: np.ndarray,
    *,
    first_levels: int,
    second_levels: int,
    residual: Literal["deviance", "pearson"] = "deviance",
    permutations: int = 64,
    seed: int = 0,
) -> ConditionalAssociationEstimate:
    """Estimate full residual matrices for the primary classical comparator."""

    return _conditional_poisson_residual_estimate(
        first_state,
        second_state,
        first_levels=first_levels,
        second_levels=second_levels,
        residual=residual,
        permutations=permutations,
        seed=seed,
        project=False,
    )

