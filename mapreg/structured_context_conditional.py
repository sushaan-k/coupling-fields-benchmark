"""Graph-structured context-conditioned exact conditional coupling.

For donor ``d`` and entity ``e``, the model is

``theta[d, e] = contexts[d] @ coefficient[:, e] + deviation[d, e]``.

The data term is the exact fixed-margin conditional negative log likelihood,
relative to its value at zero log odds, for each informative 2 x 2 table. For
entity ``e`` with ``n_e`` informative donors, its likelihood and
donor-deviation ridge are divided by ``n_e``. The coefficient ridge and graph
penalty are

``1/2 sum_e,p ridge[p] coefficient[p,e]^2``
``+ graph/2 sum_p coefficient[p,:] L coefficient[p,:]``.

Thus duplicating a complete donor panel leaves the objective and minimizer
unchanged.  With positive donor and coefficient ridges and a positive
semidefinite ``L``, the Hessian quadratic form is the sum of

``sum_de v_de (x_d b_e + u_de)^2 / n_e``
``+ eta sum_de u_de^2 / n_e + sum_ep ridge_p b_ep^2``
``+ graph sum_p b_p' L b_p``.

It is strictly positive for every nonzero fitted direction, so the objective
is strictly convex and coercive and has one finite minimizer.  The block
Newton solver eliminates donor deviations exactly, solves the coefficient
Schur complement, and returns only after gradient and conditioning
certificates pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve

from .context_conditional_coupling import (
    ConditionalRecord,
    _likelihood,
    _penalties,
    _record_grid,
    _validated_inputs,
)
from .heterogeneity_adaptive_coupling import (
    CouplingEstimationRefusal,
    product_hypergraph_laplacian,
)


__all__ = [
    "CouplingEstimationRefusal",
    "StructuredContextConditionalEvaluation",
    "StructuredContextConditionalFit",
    "evaluate_structured_context_conditional_log_odds",
    "fit_structured_context_conditional_log_odds",
]


@dataclass(frozen=True)
class StructuredContextConditionalEvaluation:
    """Objective, derivatives, and exact conditional information."""

    objective: float
    entity_objective: np.ndarray
    graph_objective: float
    coefficient_gradient: np.ndarray
    donor_deviation_gradient: np.ndarray
    donor_data_precision: np.ndarray
    donor_support: np.ndarray
    support_count: np.ndarray


@dataclass(frozen=True)
class StructuredContextConditionalFit:
    """Unique penalized fit and its numerical certificates.

    ``donor_data_precision`` is exact conditional curvature divided by the
    informative donor count for its entity.  The reported condition numbers
    describe the donor diagonal block and coefficient Schur complement used
    by the solver, rather than an uncomputed dense joint Hessian.
    """

    coefficient: np.ndarray
    context_log_odds: np.ndarray
    donor_log_odds: np.ndarray
    donor_deviation: np.ndarray
    donor_support: np.ndarray
    support_count: np.ndarray
    donor_data_precision: np.ndarray
    coefficient_gradient: np.ndarray
    donor_deviation_gradient: np.ndarray
    entity_objective: np.ndarray
    graph_objective: float
    objective: float
    gradient_norm: float
    scaled_gradient_norm: float
    schur_condition_number: float
    donor_curvature_condition_number: float
    minimum_schur_eigenvalue: float
    maximum_schur_eigenvalue: float
    minimum_donor_curvature: float
    maximum_donor_curvature: float
    graph_laplacian: np.ndarray
    graph_symmetry_residual: float
    graph_minimum_eigenvalue: float
    graph_maximum_eigenvalue: float
    graph_nullity: int
    graph_source: str
    iterations: int
    converged: bool
    optimizer: str
    donor_deviation_penalty: float
    coefficient_ridge_penalty: np.ndarray
    graph_penalty: float
    maximum_condition_number: float
    gradient_tolerance: float


@dataclass(frozen=True)
class _Problem:
    design: np.ndarray
    entity_shape: tuple[int, ...]
    records: tuple[tuple[ConditionalRecord, ...], ...]
    retained: tuple[np.ndarray, ...]
    support: np.ndarray
    support_count: np.ndarray
    support_width: np.ndarray


@dataclass(frozen=True)
class _State:
    objective: float
    entity_objective: np.ndarray
    graph_objective: float
    coefficient_gradient: np.ndarray
    deviation_gradient: np.ndarray
    data_precision: np.ndarray


@dataclass(frozen=True)
class _Graph:
    laplacian: np.ndarray
    symmetry_residual: float
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    nullity: int
    source: str


def _prepare_problem(
    tables: np.ndarray,
    contexts: np.ndarray,
    support_mask: np.ndarray | None,
    minimum_informative_donors: int,
) -> _Problem:
    values, design, _, entity_shape = _validated_inputs(tables, contexts, None)
    if any(size < 1 for size in entity_shape):
        raise ValueError("table entity axes must be nonempty")
    if support_mask is None:
        requested_support = np.ones(values.shape[:-2], dtype=bool)
    else:
        raw_support = np.asarray(support_mask)
        if raw_support.shape != values.shape[:-2]:
            raise ValueError("support_mask must match donor and entity axes")
        if raw_support.dtype.kind == "b":
            requested_support = raw_support.copy()
        elif (
            raw_support.dtype.kind in "iuf"
            and np.isfinite(raw_support).all()
            and np.isin(raw_support, (0, 1)).all()
        ):
            requested_support = raw_support.astype(bool)
        else:
            raise ValueError("support_mask must contain only boolean or binary values")
    minimum = int(minimum_informative_donors)
    if minimum < 1 or minimum > values.shape[0]:
        raise ValueError(
            "minimum_informative_donors must be between one and donor count"
        )
    record_grid, support = _record_grid(values, requested_support)
    support_count = support.sum(axis=0)
    if np.any(support_count < minimum):
        raise CouplingEstimationRefusal(
            "too few informative donors for at least one entity"
        )

    retained: list[np.ndarray] = []
    records: list[tuple[ConditionalRecord, ...]] = []
    widths = np.zeros_like(support, dtype=float)
    for entity, donor_records in enumerate(record_grid):
        indices = np.flatnonzero(support[:, entity])
        current = tuple(donor_records[index] for index in indices)
        if any(record is None for record in current):
            raise AssertionError("retained donor lacks a conditional record")
        typed = tuple(record for record in current if record is not None)
        retained.append(indices)
        records.append(typed)
        widths[indices, entity] = [record[1][-1] - record[1][0] for record in typed]
    return _Problem(
        design=design,
        entity_shape=entity_shape,
        records=tuple(records),
        retained=tuple(retained),
        support=support,
        support_count=support_count,
        support_width=widths,
    )


def _validated_entity_laplacian(
    graph_laplacian: np.ndarray,
    entity_count: int,
) -> tuple[np.ndarray, float, float, float, int]:
    matrix = np.asarray(graph_laplacian, dtype=float)
    if matrix.shape != (entity_count, entity_count) or not np.isfinite(matrix).all():
        raise ValueError("graph_laplacian must be a finite square entity matrix")
    scale = max(1.0, float(np.max(np.abs(matrix))))
    symmetry_residual = float(np.max(np.abs(matrix - matrix.T)))
    tolerance = 1e-10 * scale
    if symmetry_residual > tolerance:
        raise ValueError("graph_laplacian must be symmetric")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if float(eigenvalues[0]) < -1e-9 * scale:
        raise ValueError("graph_laplacian must be positive semidefinite")
    null_tolerance = 1e-9 * scale
    nullity = int(np.count_nonzero(np.abs(eigenvalues) <= null_tolerance))
    if nullity < 1:
        raise ValueError("graph_laplacian must have a nonempty nullspace")
    if float(eigenvalues[0]) < 0.0:
        symmetric = np.einsum(
            "ik,k,jk->ij",
            eigenvectors,
            np.maximum(eigenvalues, 0.0),
            eigenvectors,
            optimize=False,
        )
        symmetric = 0.5 * (symmetric + symmetric.T)
        eigenvalues = np.linalg.eigvalsh(symmetric)
    return (
        symmetric,
        symmetry_residual,
        float(eigenvalues[0]),
        float(eigenvalues[-1]),
        nullity,
    )


def _resolve_graph(
    entity_shape: tuple[int, ...],
    graph_laplacian: np.ndarray | None,
    first_incidence: np.ndarray | None,
    second_incidence: np.ndarray | None,
    graph_penalty: float,
) -> _Graph:
    entity_count = int(np.prod(entity_shape)) if entity_shape else 1
    supplied_incidence = first_incidence is not None or second_incidence is not None
    if supplied_incidence and (first_incidence is None or second_incidence is None):
        raise ValueError(
            "first_incidence and second_incidence must be supplied together"
        )
    if graph_laplacian is not None and supplied_incidence:
        raise ValueError("supply graph_laplacian or product incidences, not both")
    if graph_laplacian is None and not supplied_incidence:
        if graph_penalty > 0.0:
            raise ValueError(
                "a positive graph_penalty requires a Laplacian or product incidences"
            )
        zero = np.zeros((entity_count, entity_count), dtype=float)
        return _Graph(zero, 0.0, 0.0, 0.0, entity_count, "zero")

    source = "entity_laplacian"
    candidate = graph_laplacian
    if supplied_incidence:
        if len(entity_shape) != 2:
            raise ValueError(
                "product incidences require exactly two entity axes in tables"
            )
        first = np.asarray(first_incidence)
        second = np.asarray(second_incidence)
        if first.ndim != 2 or first.shape[0] != entity_shape[0]:
            raise ValueError("first_incidence rows must match the first entity axis")
        if second.ndim != 2 or second.shape[0] != entity_shape[1]:
            raise ValueError("second_incidence rows must match the second entity axis")
        candidate = product_hypergraph_laplacian(first, second)
        source = "product_hypergraph"
    if candidate is None:
        raise AssertionError("resolved graph candidate is missing")
    laplacian, residual, minimum, maximum, nullity = _validated_entity_laplacian(
        candidate, entity_count
    )
    if graph_penalty > 0.0 and nullity == entity_count:
        raise ValueError("a positive graph_penalty requires a nonzero graph operator")
    return _Graph(laplacian, residual, minimum, maximum, nullity, source)


def _validated_graph_penalty(graph_penalty: float) -> float:
    graph = float(graph_penalty)
    if not np.isfinite(graph) or graph < 0.0:
        raise ValueError("graph_penalty must be finite and nonnegative")
    return graph


def _evaluate_state(
    coefficient: np.ndarray,
    deviation: np.ndarray,
    problem: _Problem,
    ridge: np.ndarray,
    deviation_penalty: float,
    laplacian: np.ndarray,
    graph_penalty: float,
) -> _State:
    entities, contexts = coefficient.shape
    donors = problem.design.shape[0]
    if entities != problem.support.shape[1] or contexts != problem.design.shape[1]:
        raise ValueError("coefficient shape does not match contexts and entities")
    if deviation.shape != (donors, entities):
        raise ValueError("deviation shape does not match donors and entities")
    if not np.isfinite(coefficient).all() or not np.isfinite(deviation).all():
        raise ValueError("coefficient and deviation must be finite")
    if np.any(deviation[~problem.support] != 0.0):
        raise ValueError("unsupported donor deviations must be zero")

    coefficient_gradient = np.zeros_like(coefficient)
    deviation_gradient = np.zeros_like(deviation)
    data_precision = np.zeros_like(deviation)
    entity_objective = np.empty(entities, dtype=float)
    for entity in range(entities):
        retained = problem.retained[entity]
        records = problem.records[entity]
        entity_design = problem.design[retained]
        current_deviation = deviation[retained, entity]
        log_odds = entity_design @ coefficient[entity] + current_deviation
        likelihood, score, raw_precision = _likelihood(log_odds, list(records))
        scale = 1.0 / retained.size
        current_objective = scale * likelihood
        current_objective += (
            0.5
            * scale
            * deviation_penalty
            * float(current_deviation @ current_deviation)
        )
        current_objective += 0.5 * float(ridge @ np.square(coefficient[entity]))
        entity_objective[entity] = current_objective
        coefficient_gradient[entity] = (
            scale * (entity_design.T @ score) + ridge * coefficient[entity]
        )
        deviation_gradient[retained, entity] = scale * (
            score + deviation_penalty * current_deviation
        )
        data_precision[retained, entity] = scale * raw_precision

    graph_action = np.einsum("ij,jp->ip", laplacian, coefficient, optimize=False)
    graph_objective = (
        0.5
        * graph_penalty
        * float(np.einsum("ep,ep->", coefficient, graph_action, optimize=False))
    )
    coefficient_gradient += graph_penalty * graph_action
    objective = float(np.sum(entity_objective) + graph_objective)
    if (
        not np.isfinite(objective)
        or not np.isfinite(coefficient_gradient).all()
        or not np.isfinite(deviation_gradient).all()
        or not np.isfinite(data_precision).all()
    ):
        raise CouplingEstimationRefusal(
            "structured context-conditioned objective is not finite"
        )
    return _State(
        objective=objective,
        entity_objective=entity_objective,
        graph_objective=graph_objective,
        coefficient_gradient=coefficient_gradient,
        deviation_gradient=deviation_gradient,
        data_precision=data_precision,
    )


def _coefficient_penalty_hessian(
    entities: int,
    ridge: np.ndarray,
    laplacian: np.ndarray,
    graph_penalty: float,
) -> np.ndarray:
    hessian = graph_penalty * np.kron(laplacian, np.eye(ridge.size))
    diagonal = np.arange(entities * ridge.size)
    hessian[diagonal, diagonal] += np.tile(ridge, entities)
    return hessian


def _newton_system(
    state: _State,
    problem: _Problem,
    ridge: np.ndarray,
    deviation_penalty: float,
    coefficient_penalty_hessian: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    entities = problem.support.shape[1]
    contexts = problem.design.shape[1]
    schur = coefficient_penalty_hessian.copy()
    right_hand_side = -state.coefficient_gradient.ravel(order="C")
    donor_curvature = np.zeros_like(state.data_precision)
    for entity in range(entities):
        retained = problem.retained[entity]
        count = retained.size
        precision = state.data_precision[retained, entity]
        curvature = precision + deviation_penalty / count
        donor_curvature[retained, entity] = curvature
        transmitted = precision * (deviation_penalty / count) / curvature
        entity_design = problem.design[retained]
        block = slice(entity * contexts, (entity + 1) * contexts)
        schur[block, block] += (entity_design.T * transmitted) @ entity_design
        right_hand_side[block] += entity_design.T @ (
            (precision / curvature) * state.deviation_gradient[retained, entity]
        )
    return schur, right_hand_side, donor_curvature


def _gradient_certificate(
    state: _State,
    problem: _Problem,
    ridge: np.ndarray,
    deviation_penalty: float,
    laplacian: np.ndarray,
    graph_penalty: float,
) -> tuple[float, float]:
    coefficient_scale = np.empty_like(state.coefficient_gradient)
    graph_scale = graph_penalty * np.sum(np.abs(laplacian), axis=1)
    for entity, retained in enumerate(problem.retained):
        width = problem.support_width[retained, entity]
        data_scale = np.mean(np.abs(problem.design[retained]) * width[:, None], axis=0)
        coefficient_scale[entity] = np.maximum(
            1.0, data_scale + ridge + graph_scale[entity]
        )
    deviation_scale = np.ones_like(state.deviation_gradient)
    for entity, retained in enumerate(problem.retained):
        deviation_scale[retained, entity] = (
            problem.support_width[retained, entity] + deviation_penalty
        ) / retained.size
    coefficient_norm = float(np.max(np.abs(state.coefficient_gradient)))
    deviation_norm = float(np.max(np.abs(state.deviation_gradient[problem.support])))
    scaled_coefficient = float(
        np.max(np.abs(state.coefficient_gradient) / coefficient_scale)
    )
    scaled_deviation = float(
        np.max(
            np.abs(state.deviation_gradient[problem.support])
            / deviation_scale[problem.support]
        )
    )
    return max(coefficient_norm, deviation_norm), max(
        scaled_coefficient, scaled_deviation
    )


def _factor_diagnostics(
    schur: np.ndarray,
    donor_curvature: np.ndarray,
    support: np.ndarray,
    maximum_condition_number: float,
) -> tuple[float, float, float, float, float, float]:
    scale = float(np.max(np.abs(schur)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise CouplingEstimationRefusal("coefficient Schur complement is singular")
    eigenvalues = np.linalg.eigvalsh(schur / scale) * scale
    minimum_schur = float(eigenvalues[0])
    maximum_schur = float(eigenvalues[-1])
    if (
        not np.isfinite(minimum_schur)
        or minimum_schur <= 0.0
        or not np.isfinite(maximum_schur)
    ):
        raise CouplingEstimationRefusal(
            "coefficient Schur complement is not positive definite"
        )
    schur_condition = maximum_schur / minimum_schur
    supported_curvature = donor_curvature[support]
    minimum_donor = float(np.min(supported_curvature))
    maximum_donor = float(np.max(supported_curvature))
    if (
        not np.isfinite(minimum_donor)
        or minimum_donor <= 0.0
        or not np.isfinite(maximum_donor)
    ):
        raise CouplingEstimationRefusal("donor curvature is not finite and positive")
    donor_condition = maximum_donor / minimum_donor
    if (
        not np.isfinite(schur_condition)
        or schur_condition > maximum_condition_number
        or not np.isfinite(donor_condition)
        or donor_condition > maximum_condition_number
    ):
        raise CouplingEstimationRefusal(
            "structured context-conditioned solve exceeds the condition-number limit"
        )
    return (
        schur_condition,
        donor_condition,
        minimum_schur,
        maximum_schur,
        minimum_donor,
        maximum_donor,
    )


def evaluate_structured_context_conditional_log_odds(
    coefficient: np.ndarray,
    donor_deviation: np.ndarray,
    tables: np.ndarray,
    contexts: np.ndarray,
    *,
    graph_laplacian: np.ndarray | None = None,
    first_incidence: np.ndarray | None = None,
    second_incidence: np.ndarray | None = None,
    support_mask: np.ndarray | None = None,
    donor_deviation_penalty: float = 1.0,
    coefficient_ridge_penalty: float | np.ndarray = 0.1,
    graph_penalty: float = 0.0,
    minimum_informative_donors: int = 1,
) -> StructuredContextConditionalEvaluation:
    """Evaluate the normalized structured objective and exact gradient."""

    problem = _prepare_problem(
        tables, contexts, support_mask, minimum_informative_donors
    )
    ridge, deviation_penalty = _penalties(
        coefficient_ridge_penalty,
        donor_deviation_penalty,
        problem.design.shape[1],
    )
    graph_penalty = _validated_graph_penalty(graph_penalty)
    graph = _resolve_graph(
        problem.entity_shape,
        graph_laplacian,
        first_incidence,
        second_incidence,
        graph_penalty,
    )
    expected_coefficient_shape = (problem.design.shape[1], *problem.entity_shape)
    expected_deviation_shape = (problem.design.shape[0], *problem.entity_shape)
    beta = np.asarray(coefficient, dtype=float)
    deviation = np.asarray(donor_deviation, dtype=float)
    if beta.shape != expected_coefficient_shape:
        raise ValueError("coefficient shape must be context x entity axes")
    if deviation.shape != expected_deviation_shape:
        raise ValueError("donor_deviation shape must be donor x entity axes")
    flat_beta = beta.reshape(problem.design.shape[1], -1).T
    flat_deviation = deviation.reshape(problem.design.shape[0], -1)
    state = _evaluate_state(
        flat_beta,
        flat_deviation,
        problem,
        ridge,
        deviation_penalty,
        graph.laplacian,
        graph_penalty,
    )
    entity_output_shape = problem.entity_shape if problem.entity_shape else ()
    return StructuredContextConditionalEvaluation(
        objective=state.objective,
        entity_objective=state.entity_objective.reshape(entity_output_shape),
        graph_objective=state.graph_objective,
        coefficient_gradient=state.coefficient_gradient.T.reshape(
            expected_coefficient_shape
        ),
        donor_deviation_gradient=state.deviation_gradient.reshape(
            expected_deviation_shape
        ),
        donor_data_precision=state.data_precision.reshape(expected_deviation_shape),
        donor_support=problem.support.reshape(expected_deviation_shape),
        support_count=problem.support_count.reshape(entity_output_shape),
    )


def fit_structured_context_conditional_log_odds(
    tables: np.ndarray,
    contexts: np.ndarray,
    *,
    graph_laplacian: np.ndarray | None = None,
    first_incidence: np.ndarray | None = None,
    second_incidence: np.ndarray | None = None,
    support_mask: np.ndarray | None = None,
    donor_deviation_penalty: float = 1.0,
    coefficient_ridge_penalty: float | np.ndarray = 0.1,
    graph_penalty: float = 0.0,
    minimum_informative_donors: int = 1,
    maximum_condition_number: float = 1e12,
    maximum_iterations: int = 100,
    tolerance: float = 1e-8,
) -> StructuredContextConditionalFit:
    """Fit the unique graph-structured context-conditioned coupling field."""

    problem = _prepare_problem(
        tables, contexts, support_mask, minimum_informative_donors
    )
    ridge, deviation_penalty = _penalties(
        coefficient_ridge_penalty,
        donor_deviation_penalty,
        problem.design.shape[1],
    )
    graph_penalty = _validated_graph_penalty(graph_penalty)
    graph = _resolve_graph(
        problem.entity_shape,
        graph_laplacian,
        first_incidence,
        second_incidence,
        graph_penalty,
    )
    condition_limit = float(maximum_condition_number)
    iteration_limit = int(maximum_iterations)
    threshold = float(tolerance)
    if not np.isfinite(condition_limit) or condition_limit <= 1.0:
        raise ValueError("maximum_condition_number must be finite and above one")
    if iteration_limit < 1 or not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("maximum_iterations and tolerance must be positive")

    donors, entities = problem.support.shape
    contexts_count = problem.design.shape[1]
    coefficient = np.zeros((entities, contexts_count), dtype=float)
    deviation = np.zeros((donors, entities), dtype=float)
    penalty_hessian = _coefficient_penalty_hessian(
        entities, ridge, graph.laplacian, graph_penalty
    )
    converged = False
    completed_iterations = 0

    for iteration in range(iteration_limit + 1):
        state = _evaluate_state(
            coefficient,
            deviation,
            problem,
            ridge,
            deviation_penalty,
            graph.laplacian,
            graph_penalty,
        )
        _, scaled_gradient = _gradient_certificate(
            state,
            problem,
            ridge,
            deviation_penalty,
            graph.laplacian,
            graph_penalty,
        )
        if scaled_gradient <= threshold:
            converged = True
            completed_iterations = iteration
            break
        if iteration == iteration_limit:
            break

        schur, right_hand_side, donor_curvature = _newton_system(
            state,
            problem,
            ridge,
            deviation_penalty,
            penalty_hessian,
        )
        try:
            factor = cho_factor(schur, lower=True, check_finite=False)
            coefficient_step = cho_solve(
                factor, right_hand_side, check_finite=False
            ).reshape(entities, contexts_count)
        except LinAlgError as error:
            raise CouplingEstimationRefusal(
                "coefficient Schur complement is not numerically positive definite"
            ) from error
        deviation_step = np.zeros_like(deviation)
        for entity, retained in enumerate(problem.retained):
            cross_step = problem.design[retained] @ coefficient_step[entity]
            deviation_step[retained, entity] = (
                -state.deviation_gradient[retained, entity]
                - state.data_precision[retained, entity] * cross_step
            ) / donor_curvature[retained, entity]
        directional_derivative = float(
            np.einsum(
                "ep,ep->",
                state.coefficient_gradient,
                coefficient_step,
                optimize=False,
            )
            + np.einsum(
                "de,de->",
                state.deviation_gradient,
                deviation_step,
                optimize=False,
            )
        )
        if not np.isfinite(directional_derivative) or directional_derivative >= 0.0:
            raise CouplingEstimationRefusal(
                "structured context-conditioned Newton direction is not finite descent"
            )

        step_size = 1.0
        for _ in range(48):
            candidate_coefficient = coefficient + step_size * coefficient_step
            candidate_deviation = deviation + step_size * deviation_step
            try:
                candidate_objective = _evaluate_state(
                    candidate_coefficient,
                    candidate_deviation,
                    problem,
                    ridge,
                    deviation_penalty,
                    graph.laplacian,
                    graph_penalty,
                ).objective
            except CouplingEstimationRefusal:
                candidate_objective = np.inf
            if (
                np.isfinite(candidate_objective)
                and candidate_objective
                <= state.objective + 1e-4 * step_size * directional_derivative
            ):
                coefficient = candidate_coefficient
                deviation = candidate_deviation
                break
            step_size *= 0.5
        else:
            raise CouplingEstimationRefusal(
                "structured context-conditioned Newton line search found no valid step"
            )

    if not converged:
        raise CouplingEstimationRefusal(
            "structured context-conditioned optimizer missed the gradient certificate"
        )
    final_state = _evaluate_state(
        coefficient,
        deviation,
        problem,
        ridge,
        deviation_penalty,
        graph.laplacian,
        graph_penalty,
    )
    gradient_norm, scaled_gradient_norm = _gradient_certificate(
        final_state,
        problem,
        ridge,
        deviation_penalty,
        graph.laplacian,
        graph_penalty,
    )
    if scaled_gradient_norm > threshold:
        raise CouplingEstimationRefusal(
            "structured context-conditioned optimizer missed the gradient certificate"
        )
    final_schur, _, final_donor_curvature = _newton_system(
        final_state,
        problem,
        ridge,
        deviation_penalty,
        penalty_hessian,
    )
    diagnostics = _factor_diagnostics(
        final_schur,
        final_donor_curvature,
        problem.support,
        condition_limit,
    )

    coefficient_output_shape = (contexts_count, *problem.entity_shape)
    donor_output_shape = (donors, *problem.entity_shape)
    entity_output_shape = problem.entity_shape if problem.entity_shape else ()
    output_coefficient = coefficient.T.reshape(coefficient_output_shape)
    with np.errstate(over="ignore", invalid="ignore"):
        context_log_odds = np.tensordot(
            problem.design, output_coefficient, axes=([1], [0])
        )
    if not np.isfinite(context_log_odds).all():
        raise CouplingEstimationRefusal(
            "structured context-conditioned prediction is not finite"
        )
    output_deviation = deviation.reshape(donor_output_shape)
    donor_log_odds = context_log_odds + output_deviation
    return StructuredContextConditionalFit(
        coefficient=output_coefficient,
        context_log_odds=context_log_odds,
        donor_log_odds=donor_log_odds,
        donor_deviation=output_deviation,
        donor_support=problem.support.reshape(donor_output_shape),
        support_count=problem.support_count.reshape(entity_output_shape),
        donor_data_precision=final_state.data_precision.reshape(donor_output_shape),
        coefficient_gradient=final_state.coefficient_gradient.T.reshape(
            coefficient_output_shape
        ),
        donor_deviation_gradient=final_state.deviation_gradient.reshape(
            donor_output_shape
        ),
        entity_objective=final_state.entity_objective.reshape(entity_output_shape),
        graph_objective=final_state.graph_objective,
        objective=final_state.objective,
        gradient_norm=gradient_norm,
        scaled_gradient_norm=scaled_gradient_norm,
        schur_condition_number=diagnostics[0],
        donor_curvature_condition_number=diagnostics[1],
        minimum_schur_eigenvalue=diagnostics[2],
        maximum_schur_eigenvalue=diagnostics[3],
        minimum_donor_curvature=diagnostics[4],
        maximum_donor_curvature=diagnostics[5],
        graph_laplacian=graph.laplacian.copy(),
        graph_symmetry_residual=graph.symmetry_residual,
        graph_minimum_eigenvalue=graph.minimum_eigenvalue,
        graph_maximum_eigenvalue=graph.maximum_eigenvalue,
        graph_nullity=graph.nullity,
        graph_source=graph.source,
        iterations=completed_iterations,
        converged=True,
        optimizer="joint_block_newton_schur_backtracking",
        donor_deviation_penalty=deviation_penalty,
        coefficient_ridge_penalty=ridge.copy(),
        graph_penalty=graph_penalty,
        maximum_condition_number=condition_limit,
        gradient_tolerance=threshold,
    )
