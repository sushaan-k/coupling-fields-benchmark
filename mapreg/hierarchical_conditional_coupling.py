"""Donor-heterogeneity-aware exact conditional binary coupling.

For donor ``d`` and entity ``e``, the model estimates a full log odds ratio
``theta[d, e]`` from the exact fixed-margin noncentral-hypergeometric
likelihood and a population log odds ratio ``mu[e]``. The convex objective is

``mean_d sum_e loss_de(theta_de)``
``+ eta * s_eta/(2D) * sum_de (theta_de - mu_e)^2``
``+ s_mu/2 * (lambda0 ||mu||^2 + lambdaG mu' L mu)``.

Each ``loss`` is the exact conditional negative log likelihood relative to its
value at the independence parameter. ``s_eta`` is the median donor-entity null
Fisher information and ``s_mu`` is the median entity information after
averaging over donors. These fixed scales
make the penalty grid invariant to duplicating the complete donor panel. With
positive ``eta`` and at least one informative table per entity, the Hessian
quadratic form is

``sum_de v_de u_de^2 / D + eta_eff sum_de (u_de-w_e)^2``
``+ ridge_eff ||w||^2 + graph_eff w' L w``.

It vanishes only at ``u=w=0``; therefore every finite fit is unique. This is a
heterogeneity-aware penalized conditional estimator, not an integrated
random-effects likelihood.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .heterogeneity_adaptive_coupling import (
    CouplingEstimationRefusal,
    _fixed_margin_support,
    _log_choose,
    _validated_laplacian,
    log_odds_to_helmert_coordinate,
    product_hypergraph_laplacian,
)


ConditionalRecord = tuple[float, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class HierarchicalConditionalEvaluation:
    """Exact relative objective, derivatives, information, and scales.

    The Hessian orders the C-flattened donor log odds first and the population
    log odds last.
    """

    objective: float
    donor_gradient: np.ndarray
    population_gradient: np.ndarray
    hessian: np.ndarray
    donor_data_precision: np.ndarray
    donor_support: np.ndarray
    support_count: np.ndarray
    heterogeneity_penalty_scale: float
    population_penalty_scale: float
    effective_heterogeneity_penalty: float
    effective_ridge_penalty: float
    effective_graph_penalty: float


@dataclass(frozen=True)
class HierarchicalConditionalLogOddsFit:
    """Unique population and donor fit with strict numerical certificates.

    ``population_data_precision`` is its normalized data contribution after
    donor deviations are eliminated; donor precisions are unnormalized exact
    conditional Fisher information. The two condition numbers guard the exact
    block solve factors separately; neither is presented as the spectral
    condition number of the full dense joint Hessian.
    """

    population_log_odds: np.ndarray
    population_helmert_coordinate: np.ndarray
    donor_log_odds: np.ndarray
    donor_deviation: np.ndarray
    donor_data_precision: np.ndarray
    population_data_precision: np.ndarray
    donor_support: np.ndarray
    support_count: np.ndarray
    objective: float
    gradient_norm: float
    scaled_gradient_norm: float
    donor_gradient_scale: float
    population_gradient_scale: float
    schur_condition_number: float
    theta_curvature_condition_number: float
    minimum_theta_curvature: float
    maximum_theta_curvature: float
    minimum_schur_eigenvalue: float
    maximum_schur_eigenvalue: float
    iterations: int
    converged: bool
    optimizer: str
    heterogeneity_penalty: float
    ridge_penalty: float
    graph_penalty: float
    heterogeneity_penalty_scale: float
    population_penalty_scale: float
    effective_heterogeneity_penalty: float
    effective_ridge_penalty: float
    effective_graph_penalty: float
    maximum_condition_number: float
    gradient_tolerance: float


def _conditional_record_grid(
    tables: np.ndarray,
) -> tuple[
    tuple[int, ...],
    list[list[ConditionalRecord | None]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    values = np.asarray(tables)
    if values.ndim < 3 or values.shape[-2:] != (2, 2):
        raise ValueError("tables must have donor first and end in shape (2, 2)")
    if values.shape[0] < 2:
        raise ValueError("tables must contain at least two donors")
    entity_shape = values.shape[1:-2]
    entity_count = int(np.prod(entity_shape)) if entity_shape else 1
    flat = values.reshape(values.shape[0], entity_count, 2, 2)
    records: list[list[ConditionalRecord | None]] = [
        [None for _ in range(entity_count)] for _ in range(values.shape[0])
    ]
    support_mask = np.zeros((values.shape[0], entity_count), dtype=bool)
    null_precision = np.zeros((values.shape[0], entity_count), dtype=float)
    observed_sum = np.zeros(entity_count, dtype=float)
    lower_sum = np.zeros(entity_count, dtype=float)
    upper_sum = np.zeros(entity_count, dtype=float)
    for donor in range(values.shape[0]):
        for entity in range(entity_count):
            counts, cells, _ = _fixed_margin_support(flat[donor, entity])
            support = cells[:, 0]
            if len(support) < 2:
                continue
            row_one = int(counts[0].sum())
            column_one = int(counts[:, 0].sum())
            total = int(counts.sum())
            log_weights = _log_choose(column_one, support) + _log_choose(
                total - column_one, row_one - support
            )
            log_probability = log_weights - logsumexp(log_weights)
            probability = np.exp(log_probability)
            expected = float(np.sum(probability * support))
            variance = float(np.sum(probability * np.square(support - expected)))
            if not np.isfinite(variance) or variance <= 0.0:
                raise FloatingPointError("informative null precision is not positive")
            observed = float(counts[0, 0])
            records[donor][entity] = (observed, support, log_probability)
            support_mask[donor, entity] = True
            null_precision[donor, entity] = variance
            observed_sum[entity] += observed
            lower_sum[entity] += float(support[0])
            upper_sum[entity] += float(support[-1])
    return (
        entity_shape,
        records,
        support_mask,
        null_precision,
        observed_sum,
        lower_sum,
        upper_sum,
    )


def _normalization_scales(
    null_precision: np.ndarray, support: np.ndarray
) -> tuple[float, float]:
    positive = null_precision[support]
    if positive.size == 0:
        raise CouplingEstimationRefusal("no donor-entity table is informative")
    heterogeneity_scale = float(np.median(positive))
    entity_precision = np.sum(null_precision, axis=0) / null_precision.shape[0]
    positive_entity = entity_precision[entity_precision > 0.0]
    population_scale = float(np.median(positive_entity))
    if (
        not np.isfinite(heterogeneity_scale)
        or heterogeneity_scale <= 0.0
        or not np.isfinite(population_scale)
        or population_scale <= 0.0
    ):
        raise CouplingEstimationRefusal("conditional information scale is not positive")
    return heterogeneity_scale, population_scale


def _validated_penalties(
    heterogeneity_penalty: float,
    ridge_penalty: float,
    graph_penalty: float,
) -> tuple[float, float, float]:
    heterogeneity = float(heterogeneity_penalty)
    ridge = float(ridge_penalty)
    graph = float(graph_penalty)
    if not np.isfinite(heterogeneity) or heterogeneity <= 0.0:
        raise ValueError("heterogeneity_penalty must be finite and positive")
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge_penalty must be finite and nonnegative")
    if not np.isfinite(graph) or graph < 0.0:
        raise ValueError("graph_penalty must be finite and nonnegative")
    return heterogeneity, ridge, graph


def _scaled_penalty(
    value: float,
    scale: float,
    *,
    divisor: int = 1,
    name: str,
) -> float:
    if value == 0.0:
        return 0.0
    log_effective = math.log(value) + math.log(scale) - math.log(divisor)
    if log_effective > math.log(np.finfo(float).max):
        raise ValueError(f"normalized {name} overflows floating-point range")
    if log_effective < math.log(np.nextafter(0.0, 1.0)):
        raise ValueError(f"normalized {name} underflows floating-point range")
    effective = math.exp(log_effective)
    if not np.isfinite(effective) or effective <= 0.0:
        raise ValueError(f"normalized {name} is not finite and positive")
    return effective


def _effective_penalties(
    heterogeneity: float,
    ridge: float,
    graph: float,
    heterogeneity_scale: float,
    population_scale: float,
    donors: int,
) -> tuple[float, float, float]:
    return (
        _scaled_penalty(
            heterogeneity,
            heterogeneity_scale,
            divisor=donors,
            name="heterogeneity penalty",
        ),
        _scaled_penalty(ridge, population_scale, name="ridge penalty"),
        _scaled_penalty(graph, population_scale, name="graph penalty"),
    )


def _likelihood_statistics(
    donor_log_odds: np.ndarray,
    records: list[list[ConditionalRecord | None]],
) -> tuple[float, np.ndarray, np.ndarray]:
    objective = 0.0
    score = np.zeros_like(donor_log_odds)
    precision = np.zeros_like(donor_log_odds)
    for donor, donor_records in enumerate(records):
        for entity, record in enumerate(donor_records):
            if record is None:
                continue
            observed, support, log_probability = record
            with np.errstate(over="ignore", invalid="ignore"):
                log_mass = (
                    log_probability
                    + (support - observed) * donor_log_odds[donor, entity]
                )
            if not np.isfinite(log_mass).all():
                raise CouplingEstimationRefusal(
                    "conditional likelihood exceeds finite evaluation range"
                )
            log_partition = float(logsumexp(log_mass))
            probability = np.exp(log_mass - log_partition)
            expected = float(np.sum(probability * support))
            variance = float(np.sum(probability * np.square(support - expected)))
            if not np.isfinite(variance) or variance <= 0.0:
                raise CouplingEstimationRefusal(
                    "conditional precision underflowed at a finite log odds"
                )
            objective += log_partition
            score[donor, entity] = expected - observed
            precision[donor, entity] = max(variance, 0.0)
    return objective, score, precision


def _state_evaluation(
    donor_log_odds: np.ndarray,
    population_log_odds: np.ndarray,
    records: list[list[ConditionalRecord | None]],
    laplacian: np.ndarray,
    effective_heterogeneity: float,
    effective_ridge: float,
    effective_graph: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    donors = donor_log_odds.shape[0]
    likelihood, raw_score, raw_precision = _likelihood_statistics(
        donor_log_odds, records
    )
    score = raw_score / donors
    precision = raw_precision / donors
    deviation = donor_log_odds - population_log_odds[None, :]
    graph_action = np.einsum("ij,j->i", laplacian, population_log_odds, optimize=False)
    population_penalty_gradient = (
        effective_ridge * population_log_odds + effective_graph * graph_action
    )
    objective = likelihood / donors
    objective += 0.5 * effective_heterogeneity * float(np.sum(np.square(deviation)))
    objective += 0.5 * effective_ridge * float(np.sum(np.square(population_log_odds)))
    objective += (
        0.5
        * effective_graph
        * float(np.einsum("i,i->", population_log_odds, graph_action, optimize=False))
    )
    donor_gradient = score + effective_heterogeneity * deviation
    population_gradient = (
        -effective_heterogeneity * np.sum(deviation, axis=0)
        + population_penalty_gradient
    )
    return (
        float(objective),
        donor_gradient,
        population_gradient,
        raw_precision,
        precision,
    )


def evaluate_hierarchical_conditional_log_odds(
    donor_log_odds: np.ndarray,
    population_log_odds: np.ndarray,
    tables: np.ndarray,
    *,
    graph_laplacian: np.ndarray | None = None,
    heterogeneity_penalty: float = 1.0,
    ridge_penalty: float = 1e-3,
    graph_penalty: float = 0.0,
    minimum_informative_donors: int = 1,
) -> HierarchicalConditionalEvaluation:
    """Evaluate the relative-to-null hierarchical objective and derivatives."""

    (
        entity_shape,
        records,
        support,
        null_precision,
        _,
        _,
        _,
    ) = _conditional_record_grid(tables)
    theta = np.asarray(donor_log_odds, dtype=float)
    mu = np.asarray(population_log_odds, dtype=float)
    expected_mu_shape = entity_shape if entity_shape else ()
    expected_theta_shape = (len(records), *entity_shape)
    if theta.shape != expected_theta_shape or not np.isfinite(theta).all():
        raise ValueError("donor_log_odds shape must match donor and entity axes")
    if mu.shape != expected_mu_shape or not np.isfinite(mu).all():
        raise ValueError("population_log_odds shape must match entity axes")
    support_count = support.sum(axis=0)
    minimum = int(minimum_informative_donors)
    if minimum < 1 or np.any(support_count < minimum):
        raise CouplingEstimationRefusal(
            "too few informative donors for at least one entity"
        )
    heterogeneity, ridge, graph = _validated_penalties(
        heterogeneity_penalty, ridge_penalty, graph_penalty
    )
    entity_count = theta[0].size
    if graph_laplacian is None:
        if graph > 0.0:
            raise ValueError("a positive graph_penalty requires graph_laplacian")
        laplacian = np.zeros((entity_count, entity_count), dtype=float)
    else:
        laplacian = _validated_laplacian(graph_laplacian, entity_count)
    heterogeneity_scale, population_scale = _normalization_scales(
        null_precision, support
    )
    (
        effective_heterogeneity,
        effective_ridge,
        effective_graph,
    ) = _effective_penalties(
        heterogeneity,
        ridge,
        graph,
        heterogeneity_scale,
        population_scale,
        theta.shape[0],
    )
    flat_theta = theta.reshape(theta.shape[0], entity_count)
    flat_mu = mu.reshape(entity_count)
    (
        objective,
        donor_gradient,
        population_gradient,
        raw_precision,
        normalized_precision,
    ) = _state_evaluation(
        flat_theta,
        flat_mu,
        records,
        laplacian,
        effective_heterogeneity,
        effective_ridge,
        effective_graph,
    )
    donor_parameter_count = flat_theta.size
    hessian = np.zeros(
        (donor_parameter_count + entity_count,) * 2,
        dtype=float,
    )
    donor_indices = np.arange(donor_parameter_count)
    hessian[donor_indices, donor_indices] = (
        normalized_precision.ravel(order="C") + effective_heterogeneity
    )
    entity_indices = np.tile(np.arange(entity_count), flat_theta.shape[0])
    population_indices = donor_parameter_count + entity_indices
    hessian[donor_indices, population_indices] = -effective_heterogeneity
    hessian[population_indices, donor_indices] = -effective_heterogeneity
    population_hessian = effective_graph * laplacian
    diagonal = np.arange(entity_count)
    population_hessian[diagonal, diagonal] += (
        flat_theta.shape[0] * effective_heterogeneity + effective_ridge
    )
    hessian[donor_parameter_count:, donor_parameter_count:] = population_hessian
    return HierarchicalConditionalEvaluation(
        objective=objective,
        donor_gradient=donor_gradient.reshape(theta.shape),
        population_gradient=population_gradient.reshape(expected_mu_shape),
        hessian=hessian,
        donor_data_precision=raw_precision.reshape(theta.shape),
        donor_support=support.reshape(theta.shape),
        support_count=support_count.reshape(expected_mu_shape),
        heterogeneity_penalty_scale=heterogeneity_scale,
        population_penalty_scale=population_scale,
        effective_heterogeneity_penalty=effective_heterogeneity,
        effective_ridge_penalty=effective_ridge,
        effective_graph_penalty=effective_graph,
    )


def _schur_system(
    donor_log_odds: np.ndarray,
    population_log_odds: np.ndarray,
    donor_gradient: np.ndarray,
    normalized_precision: np.ndarray,
    laplacian: np.ndarray,
    effective_heterogeneity: float,
    effective_ridge: float,
    effective_graph: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta_curvature = normalized_precision + effective_heterogeneity
    transmitted_precision = (
        effective_heterogeneity * normalized_precision / theta_curvature
    )
    schur = effective_graph * laplacian
    diagonal = np.arange(population_log_odds.size)
    population_data_precision = np.sum(transmitted_precision, axis=0)
    schur[diagonal, diagonal] += population_data_precision + effective_ridge
    deviation = donor_log_odds - population_log_odds[None, :]
    population_penalty_gradient = (
        effective_ridge * population_log_odds
        + effective_graph
        * np.einsum("ij,j->i", laplacian, population_log_odds, optimize=False)
    )
    data_score = donor_gradient - effective_heterogeneity * deviation
    right_hand_side = -population_penalty_gradient + np.sum(
        transmitted_precision * deviation
        - (effective_heterogeneity / theta_curvature) * data_score,
        axis=0,
    )
    return schur, right_hand_side, theta_curvature, population_data_precision


def _schur_diagnostics(
    schur: np.ndarray, maximum_condition_number: float
) -> tuple[float, float, float]:
    scale = float(np.max(np.abs(schur)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise CouplingEstimationRefusal("hierarchical Schur system is singular")
    eigenvalues = np.linalg.eigvalsh(schur / scale) * scale
    minimum = float(eigenvalues[0])
    maximum = float(eigenvalues[-1])
    if not np.isfinite(minimum) or minimum <= 0.0 or not np.isfinite(maximum):
        raise CouplingEstimationRefusal(
            "hierarchical Hessian is not numerically positive definite"
        )
    condition_number = maximum / minimum
    if condition_number > maximum_condition_number:
        raise CouplingEstimationRefusal(
            "hierarchical Schur system exceeds the condition-number limit"
        )
    return condition_number, minimum, maximum


def _theta_curvature_diagnostics(
    theta_curvature: np.ndarray, maximum_condition_number: float
) -> tuple[float, float, float]:
    minimum = float(np.min(theta_curvature))
    maximum = float(np.max(theta_curvature))
    if not np.isfinite(minimum) or minimum <= 0.0 or not np.isfinite(maximum):
        raise CouplingEstimationRefusal(
            "hierarchical donor curvature is not finite and positive"
        )
    condition_number = maximum / minimum
    if condition_number > maximum_condition_number:
        raise CouplingEstimationRefusal(
            "hierarchical donor curvature exceeds the condition-number limit"
        )
    return condition_number, minimum, maximum


def _gradient_certificate(
    donor_gradient: np.ndarray,
    population_gradient: np.ndarray,
    donor_scale: float,
    population_scale: float,
) -> tuple[float, float]:
    donor_norm = float(np.max(np.abs(donor_gradient)))
    population_norm = float(np.max(np.abs(population_gradient)))
    return (
        max(donor_norm, population_norm),
        max(donor_norm / donor_scale, population_norm / population_scale),
    )


def _refuse_boundary_recession(
    observed_sum: np.ndarray,
    lower_sum: np.ndarray,
    upper_sum: np.ndarray,
    penalized_laplacian: np.ndarray,
    ridge_penalty: float,
) -> None:
    if ridge_penalty > 0.0:
        return
    adjacency = np.abs(penalized_laplacian) > 0.0
    np.fill_diagonal(adjacency, False)
    unseen = set(range(observed_sum.size))
    while unseen:
        seed = min(unseen)
        component = {seed}
        stack = [seed]
        unseen.remove(seed)
        while stack:
            current = stack.pop()
            neighbors = set(np.flatnonzero(adjacency[current])) & unseen
            unseen.difference_update(neighbors)
            component.update(neighbors)
            stack.extend(neighbors)
        indices = np.fromiter(sorted(component), dtype=int)
        if np.all(observed_sum[indices] >= upper_sum[indices]) or np.all(
            observed_sum[indices] <= lower_sum[indices]
        ):
            raise CouplingEstimationRefusal(
                "conditional likelihood has a boundary recession direction"
            )


def fit_hierarchical_conditional_log_odds(
    tables: np.ndarray,
    first_incidence: np.ndarray,
    second_incidence: np.ndarray,
    *,
    heterogeneity_penalty: float = 1.0,
    ridge_penalty: float = 1e-3,
    graph_penalty: float = 0.0,
    minimum_informative_donors: int = 2,
    maximum_condition_number: float = 1e12,
    maximum_iterations: int = 100,
    tolerance: float = 1e-8,
) -> HierarchicalConditionalLogOddsFit:
    """Fit the unique donor-heterogeneity-aware exact conditional model.

    Each Newton step eliminates donor parameters analytically and solves the
    exact population Schur complement. A backtracking line search globalizes
    the step. Convergence requires the reported scaled gradient certificate;
    an optimizer status flag cannot override that gate. The condition limit
    applies separately to the donor diagonal block and population Schur
    complement, which are the factors used by the solver.
    """

    values = np.asarray(tables)
    first = np.asarray(first_incidence)
    second = np.asarray(second_incidence)
    if values.ndim != 5 or values.shape[-2:] != (2, 2):
        raise ValueError(
            "tables must have shape donor x first_entity x second_entity x 2 x 2"
        )
    if first.ndim != 2 or first.shape[0] != values.shape[1]:
        raise ValueError("first_incidence rows must match first entities")
    if second.ndim != 2 or second.shape[0] != values.shape[2]:
        raise ValueError("second_incidence rows must match second entities")
    (
        entity_shape,
        records,
        support,
        null_precision,
        observed_sum,
        lower_sum,
        upper_sum,
    ) = _conditional_record_grid(values)
    support_count = support.sum(axis=0)
    minimum = int(minimum_informative_donors)
    if minimum < 1 or minimum > values.shape[0]:
        raise ValueError(
            "minimum_informative_donors must be between one and donor count"
        )
    if np.any(support_count < minimum):
        raise CouplingEstimationRefusal(
            "too few informative donors for at least one entity"
        )
    heterogeneity, ridge, graph = _validated_penalties(
        heterogeneity_penalty, ridge_penalty, graph_penalty
    )
    condition_limit = float(maximum_condition_number)
    iterations_limit = int(maximum_iterations)
    threshold = float(tolerance)
    if not np.isfinite(condition_limit) or condition_limit <= 1.0:
        raise ValueError("maximum_condition_number must be finite and above one")
    if iterations_limit < 1 or not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("maximum_iterations and tolerance must be positive")
    laplacian = product_hypergraph_laplacian(first, second)
    penalized_laplacian = laplacian if graph > 0.0 else np.zeros_like(laplacian)
    _refuse_boundary_recession(
        observed_sum,
        lower_sum,
        upper_sum,
        penalized_laplacian,
        ridge,
    )
    entity_count = int(np.prod(entity_shape))
    heterogeneity_scale, population_scale = _normalization_scales(
        null_precision, support
    )
    (
        effective_heterogeneity,
        effective_ridge,
        effective_graph,
    ) = _effective_penalties(
        heterogeneity,
        ridge,
        graph,
        heterogeneity_scale,
        population_scale,
        values.shape[0],
    )
    donor_log_odds = np.zeros((values.shape[0], entity_count), dtype=float)
    population_log_odds = np.zeros(entity_count, dtype=float)
    donor_gradient_scale = max(1.0, heterogeneity_scale) / values.shape[0]
    population_gradient_scale = max(1.0, population_scale)
    converged = False
    completed_iterations = 0

    for iteration in range(iterations_limit + 1):
        (
            objective,
            donor_gradient,
            population_gradient,
            _,
            normalized_precision,
        ) = _state_evaluation(
            donor_log_odds,
            population_log_odds,
            records,
            laplacian,
            effective_heterogeneity,
            effective_ridge,
            effective_graph,
        )
        gradient_norm, scaled_gradient_norm = _gradient_certificate(
            donor_gradient,
            population_gradient,
            donor_gradient_scale,
            population_gradient_scale,
        )
        if scaled_gradient_norm <= threshold:
            converged = True
            completed_iterations = iteration
            break
        if iteration == iterations_limit:
            break
        schur, right_hand_side, theta_curvature, _ = _schur_system(
            donor_log_odds,
            population_log_odds,
            donor_gradient,
            normalized_precision,
            laplacian,
            effective_heterogeneity,
            effective_ridge,
            effective_graph,
        )
        _schur_diagnostics(schur, condition_limit)
        _theta_curvature_diagnostics(theta_curvature, condition_limit)
        population_step = np.linalg.solve(schur, right_hand_side)
        donor_step = (
            -donor_gradient + effective_heterogeneity * population_step[None, :]
        ) / theta_curvature
        directional_derivative = float(
            np.sum(donor_gradient * donor_step)
            + np.sum(population_gradient * population_step)
        )
        if not np.isfinite(directional_derivative) or directional_derivative >= 0.0:
            raise CouplingEstimationRefusal(
                "hierarchical Newton direction is not a finite descent direction"
            )
        step_size = 1.0
        for _ in range(48):
            candidate_theta = donor_log_odds + step_size * donor_step
            candidate_mu = population_log_odds + step_size * population_step
            try:
                candidate_objective = _state_evaluation(
                    candidate_theta,
                    candidate_mu,
                    records,
                    laplacian,
                    effective_heterogeneity,
                    effective_ridge,
                    effective_graph,
                )[0]
            except CouplingEstimationRefusal:
                candidate_objective = np.inf
            if (
                np.isfinite(candidate_objective)
                and candidate_objective
                <= objective + 1e-4 * step_size * directional_derivative
            ):
                donor_log_odds = candidate_theta
                population_log_odds = candidate_mu
                break
            step_size *= 0.5
        else:
            raise CouplingEstimationRefusal(
                "hierarchical Newton line search did not find a valid step"
            )

    if not converged:
        raise CouplingEstimationRefusal(
            "hierarchical conditional optimizer missed the gradient certificate"
        )
    (
        final_objective,
        final_donor_gradient,
        final_population_gradient,
        final_raw_precision,
        final_normalized_precision,
    ) = _state_evaluation(
        donor_log_odds,
        population_log_odds,
        records,
        laplacian,
        effective_heterogeneity,
        effective_ridge,
        effective_graph,
    )
    final_gradient_norm, final_scaled_gradient = _gradient_certificate(
        final_donor_gradient,
        final_population_gradient,
        donor_gradient_scale,
        population_gradient_scale,
    )
    if final_scaled_gradient > threshold:
        raise CouplingEstimationRefusal(
            "hierarchical conditional optimizer missed the gradient certificate"
        )
    final_schur, _, theta_curvature, population_data_precision = _schur_system(
        donor_log_odds,
        population_log_odds,
        final_donor_gradient,
        final_normalized_precision,
        laplacian,
        effective_heterogeneity,
        effective_ridge,
        effective_graph,
    )
    schur_condition, minimum_eigenvalue, maximum_eigenvalue = _schur_diagnostics(
        final_schur, condition_limit
    )
    (
        theta_condition,
        minimum_theta_curvature,
        maximum_theta_curvature,
    ) = _theta_curvature_diagnostics(theta_curvature, condition_limit)
    population = population_log_odds.reshape(entity_shape)
    donor = donor_log_odds.reshape(values.shape[:3])
    return HierarchicalConditionalLogOddsFit(
        population_log_odds=population,
        population_helmert_coordinate=log_odds_to_helmert_coordinate(population),
        donor_log_odds=donor,
        donor_deviation=donor - population[None, :, :],
        donor_data_precision=final_raw_precision.reshape(values.shape[:3]),
        population_data_precision=population_data_precision.reshape(entity_shape),
        donor_support=support.reshape(values.shape[:3]),
        support_count=support_count.reshape(entity_shape),
        objective=final_objective,
        gradient_norm=final_gradient_norm,
        scaled_gradient_norm=final_scaled_gradient,
        donor_gradient_scale=donor_gradient_scale,
        population_gradient_scale=population_gradient_scale,
        schur_condition_number=schur_condition,
        theta_curvature_condition_number=theta_condition,
        minimum_theta_curvature=minimum_theta_curvature,
        maximum_theta_curvature=maximum_theta_curvature,
        minimum_schur_eigenvalue=minimum_eigenvalue,
        maximum_schur_eigenvalue=maximum_eigenvalue,
        iterations=completed_iterations,
        converged=True,
        optimizer="exact_block_newton_schur_backtracking",
        heterogeneity_penalty=heterogeneity,
        ridge_penalty=ridge,
        graph_penalty=graph,
        heterogeneity_penalty_scale=heterogeneity_scale,
        population_penalty_scale=population_scale,
        effective_heterogeneity_penalty=effective_heterogeneity,
        effective_ridge_penalty=effective_ridge,
        effective_graph_penalty=effective_graph,
        maximum_condition_number=condition_limit,
        gradient_tolerance=threshold,
    )
