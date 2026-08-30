"""Paired longitudinal exact-conditional binary coupling fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .heterogeneity_adaptive_coupling import (
    CouplingEstimationRefusal,
    _fixed_margin_support,
    _log_choose,
)


@dataclass(frozen=True)
class LongitudinalConditionalEvaluation:
    """Objective and gradient in donor x visit x field coordinates."""

    objective: float
    gradient: np.ndarray
    data_objective: float
    penalty_objective: float
    population_mean: np.ndarray
    population_change: np.ndarray
    donor_baseline_deviation: np.ndarray
    donor_change_deviation: np.ndarray


@dataclass(frozen=True)
class LongitudinalConditionalFit:
    """Certified fit of the frozen paired longitudinal field."""

    population_mean: np.ndarray
    population_change: np.ndarray
    donor_log_odds: np.ndarray
    donor_baseline_deviation: np.ndarray
    donor_change_deviation: np.ndarray
    objective: float
    gradient_norm: float
    scaled_gradient_norm: float
    iterations: int
    function_evaluations: int
    converged: bool
    optimizer: str
    heterogeneity_penalty: float
    population_ridge: float
    graph_penalty: float
    maximum_baseline_constraint_error: float
    maximum_change_constraint_error: float
    informative_table_count: int
    retained_coordinate_count: int


@dataclass(frozen=True)
class _ConditionalBatch:
    parameter_indices: np.ndarray
    centered_support: np.ndarray
    log_null_probability: np.ndarray
    informative_table_count: int


def _validated_laplacian(values: np.ndarray, size: int, label: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (size, size) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be one finite square matrix")
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0):
        raise ValueError(f"{label} must be symmetric")
    if np.max(np.abs(matrix.sum(axis=1))) > 1e-10:
        raise ValueError(f"{label} rows must sum to zero")
    minimum = float(np.linalg.eigvalsh(matrix)[0])
    if minimum < -1e-10:
        raise ValueError(f"{label} must be positive semidefinite")
    return matrix


def _conditional_batch(tables: np.ndarray, support_mask: np.ndarray) -> _ConditionalBatch:
    values = np.asarray(tables)
    mask = np.asarray(support_mask, dtype=bool)
    if values.ndim != 6 or values.shape[1] != 2 or values.shape[-2:] != (2, 2):
        raise ValueError(
            "tables must have shape donor x two visits x first x second x 2 x 2"
        )
    if mask.shape != values.shape[:-2]:
        raise ValueError("support_mask must match donor, visit, and field axes")
    if values.shape[0] < 2 or values.shape[2] < 1 or values.shape[3] < 1:
        raise ValueError("tables must contain at least two donors and one coordinate")
    numeric = np.asarray(values, dtype=float)
    if (
        not np.isfinite(numeric).all()
        or np.any(numeric < 0.0)
        or not np.array_equal(numeric, np.rint(numeric))
    ):
        raise ValueError("tables must contain finite nonnegative integer counts")

    indices: list[int] = []
    offsets: list[np.ndarray] = []
    log_probabilities: list[np.ndarray] = []
    field_size = values.shape[2] * values.shape[3]
    for donor, visit, first, second in np.argwhere(mask):
        counts, cells, _ = _fixed_margin_support(
            values[donor, visit, first, second]
        )
        support = cells[:, 0].astype(float)
        if len(support) < 2:
            continue
        row_zero = int(counts[0].sum())
        column_zero = int(counts[:, 0].sum())
        total = int(counts.sum())
        log_weights = _log_choose(column_zero, support) + _log_choose(
            total - column_zero, row_zero - support
        )
        log_probability = log_weights - logsumexp(log_weights)
        observed = float(counts[0, 0])
        indices.append((donor * 2 + visit) * field_size + first * values.shape[3] + second)
        offsets.append(support - observed)
        log_probabilities.append(log_probability)
    if not indices:
        raise CouplingEstimationRefusal("no retained table has informative margins")
    width = max(len(value) for value in offsets)
    centered = np.zeros((len(offsets), width), dtype=float)
    log_null = np.full((len(offsets), width), -np.inf, dtype=float)
    for row, (offset, probability) in enumerate(zip(offsets, log_probabilities)):
        centered[row, : len(offset)] = offset
        log_null[row, : len(probability)] = probability
    return _ConditionalBatch(
        parameter_indices=np.asarray(indices, dtype=np.int64),
        centered_support=centered,
        log_null_probability=log_null,
        informative_table_count=len(indices),
    )


def _graph_action(field: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.einsum("ij,...jk->...ik", first, field, optimize=True) + np.einsum(
        "...ij,jk->...ik", field, second, optimize=True
    )


def _evaluate(
    donor_log_odds: np.ndarray,
    batch: _ConditionalBatch,
    first_laplacian: np.ndarray,
    second_laplacian: np.ndarray,
    heterogeneity_penalty: float,
    population_ridge: float,
    graph_penalty: float,
) -> LongitudinalConditionalEvaluation:
    theta = np.asarray(donor_log_odds, dtype=float)
    if theta.ndim != 4 or theta.shape[1] != 2 or not np.isfinite(theta).all():
        raise ValueError("donor_log_odds must be finite donor x two visits x field")
    donors = theta.shape[0]
    flat = theta.ravel(order="C")
    selected = flat[batch.parameter_indices]
    log_mass = (
        batch.log_null_probability
        + batch.centered_support * selected[:, None]
    )
    partitions = logsumexp(log_mass, axis=1)
    probabilities = np.exp(log_mass - partitions[:, None])
    scores = np.sum(probabilities * batch.centered_support, axis=1)
    data_gradient = np.zeros_like(flat)
    data_gradient[batch.parameter_indices] = scores
    data_gradient = data_gradient.reshape(theta.shape)
    data_objective = float(np.sum(partitions))

    midpoint = 0.5 * (theta[:, 0] + theta[:, 1])
    difference = theta[:, 1] - theta[:, 0]
    population_mean = midpoint.mean(axis=0)
    population_change = difference.mean(axis=0)
    baseline = midpoint - population_mean[None, ...]
    change = difference - population_change[None, ...]

    penalty_objective = 0.5 * population_ridge * float(
        np.sum(np.square(population_mean)) + np.sum(np.square(population_change))
    )
    penalty_objective += 0.5 * heterogeneity_penalty * float(
        np.sum(np.square(baseline)) + np.sum(np.square(change))
    )

    mean_action = _graph_action(population_mean, first_laplacian, second_laplacian)
    delta_action = _graph_action(
        population_change, first_laplacian, second_laplacian
    )
    baseline_action = _graph_action(baseline, first_laplacian, second_laplacian)
    change_action = _graph_action(change, first_laplacian, second_laplacian)
    if graph_penalty:
        penalty_objective += 0.5 * graph_penalty * float(
            np.sum(population_mean * mean_action)
            + np.sum(population_change * delta_action)
            + np.sum(baseline * baseline_action)
            + np.sum(change * change_action)
        )

    midpoint_gradient = heterogeneity_penalty * baseline
    midpoint_gradient += (population_ridge / donors) * population_mean[None, ...]
    difference_gradient = heterogeneity_penalty * change
    difference_gradient += (
        population_ridge / donors
    ) * population_change[None, ...]
    if graph_penalty:
        midpoint_gradient += graph_penalty * (
            baseline_action + mean_action[None, ...] / donors
        )
        difference_gradient += graph_penalty * (
            change_action + delta_action[None, ...] / donors
        )
    penalty_gradient = np.empty_like(theta)
    penalty_gradient[:, 0] = 0.5 * midpoint_gradient - difference_gradient
    penalty_gradient[:, 1] = 0.5 * midpoint_gradient + difference_gradient
    gradient = data_gradient + penalty_gradient
    objective = data_objective + penalty_objective
    if not np.isfinite(objective) or not np.isfinite(gradient).all():
        raise CouplingEstimationRefusal("longitudinal objective is not finite")
    return LongitudinalConditionalEvaluation(
        objective=float(objective),
        gradient=gradient,
        data_objective=data_objective,
        penalty_objective=float(penalty_objective),
        population_mean=population_mean,
        population_change=population_change,
        donor_baseline_deviation=baseline,
        donor_change_deviation=change,
    )


def evaluate_longitudinal_conditional_log_odds(
    donor_log_odds: np.ndarray,
    tables: np.ndarray,
    first_laplacian: np.ndarray,
    second_laplacian: np.ndarray,
    *,
    support_mask: np.ndarray | None = None,
    heterogeneity_penalty: float = 1.0,
    population_ridge: float = 0.1,
    graph_penalty: float = 0.0,
) -> LongitudinalConditionalEvaluation:
    """Evaluate the exact frozen objective and analytic gradient."""

    values = np.asarray(tables)
    mask = (
        np.ones(values.shape[:-2], dtype=bool)
        if support_mask is None
        else np.asarray(support_mask, dtype=bool)
    )
    first = _validated_laplacian(first_laplacian, values.shape[2], "first_laplacian")
    second = _validated_laplacian(
        second_laplacian, values.shape[3], "second_laplacian"
    )
    heterogeneity = float(heterogeneity_penalty)
    ridge = float(population_ridge)
    graph = float(graph_penalty)
    if not np.isfinite(heterogeneity) or heterogeneity <= 0.0:
        raise ValueError("heterogeneity_penalty must be finite and positive")
    if not np.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("population_ridge must be finite and positive")
    if not np.isfinite(graph) or graph < 0.0:
        raise ValueError("graph_penalty must be finite and nonnegative")
    return _evaluate(
        donor_log_odds,
        _conditional_batch(values, mask),
        first,
        second,
        heterogeneity,
        ridge,
        graph,
    )


def fit_longitudinal_conditional_log_odds(
    tables: np.ndarray,
    first_laplacian: np.ndarray,
    second_laplacian: np.ndarray,
    *,
    support_mask: np.ndarray | None = None,
    heterogeneity_penalty: float = 1.0,
    population_ridge: float = 0.1,
    graph_penalty: float = 0.0,
    maximum_iterations: int = 400,
    gradient_tolerance: float = 1e-5,
) -> LongitudinalConditionalFit:
    """Fit the paired field with analytic-gradient L-BFGS and strict certificates."""

    values = np.asarray(tables)
    if values.ndim != 6:
        raise ValueError("tables must have six axes")
    mask = (
        np.ones(values.shape[:-2], dtype=bool)
        if support_mask is None
        else np.asarray(support_mask, dtype=bool)
    )
    first = _validated_laplacian(first_laplacian, values.shape[2], "first_laplacian")
    second = _validated_laplacian(
        second_laplacian, values.shape[3], "second_laplacian"
    )
    heterogeneity = float(heterogeneity_penalty)
    ridge = float(population_ridge)
    graph = float(graph_penalty)
    iterations = int(maximum_iterations)
    tolerance = float(gradient_tolerance)
    if not np.isfinite(heterogeneity) or heterogeneity <= 0.0:
        raise ValueError("heterogeneity_penalty must be finite and positive")
    if not np.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("population_ridge must be finite and positive")
    if not np.isfinite(graph) or graph < 0.0:
        raise ValueError("graph_penalty must be finite and nonnegative")
    if iterations < 1 or not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("iteration limit and gradient tolerance must be positive")
    batch = _conditional_batch(values, mask)
    shape = values.shape[:-2]

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        evaluation = _evaluate(
            flat.reshape(shape),
            batch,
            first,
            second,
            heterogeneity,
            ridge,
            graph,
        )
        return evaluation.objective, evaluation.gradient.ravel(order="C")

    result = minimize(
        objective,
        np.zeros(int(np.prod(shape)), dtype=float),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": iterations,
            "maxls": 50,
            "ftol": 1e-13,
            "gtol": min(1e-8, tolerance / 10.0),
            "maxcor": 20,
        },
    )
    final = _evaluate(
        np.asarray(result.x).reshape(shape),
        batch,
        first,
        second,
        heterogeneity,
        ridge,
        graph,
    )
    gradient_norm = float(np.max(np.abs(final.gradient)))
    scale = max(1.0, float(np.max(np.abs(final.population_mean))), float(np.max(np.abs(final.population_change))))
    scaled_gradient = gradient_norm / scale
    baseline_error = float(
        np.max(np.abs(final.donor_baseline_deviation.sum(axis=0)))
    )
    change_error = float(
        np.max(np.abs(final.donor_change_deviation.sum(axis=0)))
    )
    if not result.success or gradient_norm > tolerance:
        raise CouplingEstimationRefusal(
            "longitudinal optimizer missed its convergence certificate"
        )
    if baseline_error > 1e-10 or change_error > 1e-10:
        raise CouplingEstimationRefusal(
            "longitudinal donor effects missed their zero-sum constraints"
        )
    return LongitudinalConditionalFit(
        population_mean=final.population_mean,
        population_change=final.population_change,
        donor_log_odds=np.asarray(result.x).reshape(shape),
        donor_baseline_deviation=final.donor_baseline_deviation,
        donor_change_deviation=final.donor_change_deviation,
        objective=final.objective,
        gradient_norm=gradient_norm,
        scaled_gradient_norm=scaled_gradient,
        iterations=int(result.nit),
        function_evaluations=int(result.nfev),
        converged=True,
        optimizer="analytic_gradient_lbfgsb",
        heterogeneity_penalty=heterogeneity,
        population_ridge=ridge,
        graph_penalty=graph,
        maximum_baseline_constraint_error=baseline_error,
        maximum_change_constraint_error=change_error,
        informative_table_count=batch.informative_table_count,
        retained_coordinate_count=int(np.count_nonzero(np.any(mask, axis=(0, 1)))),
    )


def fit_visit_agnostic_conditional_log_odds(
    tables: np.ndarray,
    first_laplacian: np.ndarray,
    second_laplacian: np.ndarray,
    *,
    support_mask: np.ndarray | None = None,
    heterogeneity_penalty: float = 1.0,
    population_ridge: float = 0.1,
    graph_penalty: float = 0.0,
    maximum_iterations: int = 400,
    gradient_tolerance: float = 1e-5,
) -> LongitudinalConditionalFit:
    """Fit the frozen ablation with one donor field shared by both visits."""

    values = np.asarray(tables)
    if values.ndim != 6:
        raise ValueError("tables must have six axes")
    mask = (
        np.ones(values.shape[:-2], dtype=bool)
        if support_mask is None
        else np.asarray(support_mask, dtype=bool)
    )
    first = _validated_laplacian(first_laplacian, values.shape[2], "first_laplacian")
    second = _validated_laplacian(
        second_laplacian, values.shape[3], "second_laplacian"
    )
    heterogeneity = float(heterogeneity_penalty)
    ridge = float(population_ridge)
    graph = float(graph_penalty)
    iterations = int(maximum_iterations)
    tolerance = float(gradient_tolerance)
    if not np.isfinite(heterogeneity) or heterogeneity <= 0.0:
        raise ValueError("heterogeneity_penalty must be finite and positive")
    if not np.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("population_ridge must be finite and positive")
    if not np.isfinite(graph) or graph < 0.0:
        raise ValueError("graph_penalty must be finite and nonnegative")
    if iterations < 1 or not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("iteration limit and gradient tolerance must be positive")

    batch = _conditional_batch(values, mask)
    donors, _, first_size, second_size = values.shape[:-2]
    field_size = first_size * second_size
    mapped_indices = (
        batch.parameter_indices // (2 * field_size) * field_size
        + batch.parameter_indices % field_size
    )
    shape = (donors, first_size, second_size)

    def evaluate(flat: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        theta = np.asarray(flat, dtype=float).reshape(shape)
        selected = theta.ravel(order="C")[mapped_indices]
        log_mass = (
            batch.log_null_probability
            + batch.centered_support * selected[:, None]
        )
        partitions = logsumexp(log_mass, axis=1)
        probabilities = np.exp(log_mass - partitions[:, None])
        scores = np.sum(probabilities * batch.centered_support, axis=1)
        data_gradient = np.zeros(theta.size, dtype=float)
        np.add.at(data_gradient, mapped_indices, scores)
        data_gradient = data_gradient.reshape(shape)

        population = theta.mean(axis=0)
        deviations = theta - population[None, ...]
        population_action = _graph_action(population, first, second)
        deviation_action = _graph_action(deviations, first, second)
        penalty = 0.5 * ridge * float(np.sum(np.square(population)))
        penalty += 0.5 * heterogeneity * float(np.sum(np.square(deviations)))
        gradient = data_gradient + heterogeneity * deviations
        gradient += (ridge / donors) * population[None, ...]
        if graph:
            penalty += 0.5 * graph * float(
                np.sum(population * population_action)
                + np.sum(deviations * deviation_action)
            )
            gradient += graph * (
                deviation_action + population_action[None, ...] / donors
            )
        objective = float(np.sum(partitions) + penalty)
        if not np.isfinite(objective) or not np.isfinite(gradient).all():
            raise CouplingEstimationRefusal(
                "visit-agnostic objective is not finite"
            )
        return objective, gradient, population, deviations

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient, _, _ = evaluate(flat)
        return value, gradient.ravel(order="C")

    result = minimize(
        objective,
        np.zeros(int(np.prod(shape)), dtype=float),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": iterations,
            "maxls": 50,
            "ftol": 1e-13,
            "gtol": min(1e-8, tolerance / 10.0),
            "maxcor": 20,
        },
    )
    final_theta = np.asarray(result.x).reshape(shape)
    final_objective, final_gradient, population, deviations = evaluate(result.x)
    gradient_norm = float(np.max(np.abs(final_gradient)))
    scale = max(1.0, float(np.max(np.abs(population))))
    constraint_error = float(np.max(np.abs(deviations.sum(axis=0))))
    if not result.success or gradient_norm > tolerance:
        raise CouplingEstimationRefusal(
            "visit-agnostic optimizer missed its convergence certificate"
        )
    if constraint_error > 1e-10:
        raise CouplingEstimationRefusal(
            "visit-agnostic donor effects missed their zero-sum constraint"
        )
    zeros = np.zeros_like(population)
    return LongitudinalConditionalFit(
        population_mean=population,
        population_change=zeros,
        donor_log_odds=np.repeat(final_theta[:, None, ...], 2, axis=1),
        donor_baseline_deviation=deviations,
        donor_change_deviation=np.zeros_like(deviations),
        objective=final_objective,
        gradient_norm=gradient_norm,
        scaled_gradient_norm=gradient_norm / scale,
        iterations=int(result.nit),
        function_evaluations=int(result.nfev),
        converged=True,
        optimizer="analytic_gradient_lbfgsb_visit_agnostic",
        heterogeneity_penalty=heterogeneity,
        population_ridge=ridge,
        graph_penalty=graph,
        maximum_baseline_constraint_error=constraint_error,
        maximum_change_constraint_error=0.0,
        informative_table_count=batch.informative_table_count,
        retained_coordinate_count=int(np.count_nonzero(np.any(mask, axis=(0, 1)))),
    )
