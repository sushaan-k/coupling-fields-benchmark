"""Context-conditioned exact conditional coupling for binary assay pairs.

For donor ``d`` and entity ``e``, the fitted log odds ratio is

``theta[d, e] = context[d] @ coefficient[:, e] + deviation[d, e]``.

For an entity observed in ``D`` donors, the objective is the donor-mean exact
fixed-margin conditional negative log likelihood plus
``eta ||deviation||^2 / (2D)`` and a positive coefficient ridge. Its Hessian
quadratic form is

``sum_d [v_d (x_d @ b + u_d)^2 + eta u_d^2] / D``
``+ sum_j lambda_j b_j^2``,

where ``v_d`` is the exact conditional Fisher information. Because every
``lambda_j`` and ``eta`` is positive, this expression is positive for every
nonzero ``(b, u)``. The positive penalties also make the objective coercive,
so the implemented coordinate objective has a unique finite minimizer. The
implementation returns a fit only when its scaled-gradient and factor-condition
certificates pass. Those certificates establish numerical stationarity and
conditioning of the fitted objective, not external validity or uncertainty
calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .heterogeneity_adaptive_coupling import (
    CouplingEstimationRefusal,
    _fixed_margin_support,
    _log_choose,
)


ConditionalRecord = tuple[float, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class ContextConditionalCouplingFit:
    """Unique penalized fit and its numerical certificates.

    ``coefficient`` has context as its first axis. Donor-indexed arrays have
    donor first and then the original entity axes. ``donor_data_precision`` is
    the exact conditional curvature divided by the informative donor count.
    Schur and donor-curvature condition numbers describe the exact factors used
    by the block Newton solve, not the uncomputed dense joint Hessian.
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
    coordinate_objective: np.ndarray
    objective: float
    gradient_norm: float
    scaled_gradient_norm: float
    schur_condition_number: np.ndarray
    donor_curvature_condition_number: np.ndarray
    minimum_schur_eigenvalue: np.ndarray
    maximum_schur_eigenvalue: np.ndarray
    minimum_donor_curvature: np.ndarray
    maximum_donor_curvature: np.ndarray
    iterations: np.ndarray
    converged: bool
    optimizer: str
    donor_deviation_penalty: float
    coefficient_ridge_penalty: np.ndarray
    maximum_condition_number: float
    gradient_tolerance: float


def _validated_inputs(
    tables: np.ndarray,
    contexts: np.ndarray,
    support_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    values = np.asarray(tables)
    design = np.asarray(contexts, dtype=float)
    if values.ndim < 3 or values.shape[-2:] != (2, 2):
        raise ValueError("tables must have donor first and end in shape (2, 2)")
    if values.shape[0] < 1:
        raise ValueError("tables must contain at least one donor")
    if design.ndim != 2 or design.shape[0] != values.shape[0] or design.shape[1] < 1:
        raise ValueError("contexts must be a donor x context matrix")
    if not np.isfinite(design).all():
        raise ValueError("contexts must be finite")
    numeric = np.asarray(values, dtype=float)
    if (
        not np.isfinite(numeric).all()
        or np.any(numeric < 0.0)
        or not np.array_equal(numeric, np.rint(numeric))
    ):
        raise ValueError("tables must contain finite nonnegative integer counts")
    if np.any(numeric > np.iinfo(np.int64).max):
        raise ValueError("table counts exceed the supported integer range")
    mask = (
        np.ones(values.shape[:-2], dtype=bool)
        if support_mask is None
        else np.asarray(support_mask, dtype=bool)
    )
    if mask.shape != values.shape[:-2]:
        raise ValueError("support_mask must match donor and entity axes")
    return values, design, mask, values.shape[1:-2]


def _record_grid(
    tables: np.ndarray,
    support_mask: np.ndarray,
) -> tuple[list[list[ConditionalRecord | None]], np.ndarray]:
    donors = tables.shape[0]
    entity_count = int(np.prod(tables.shape[1:-2])) if tables.ndim > 3 else 1
    flat_tables = tables.reshape(donors, entity_count, 2, 2)
    flat_mask = support_mask.reshape(donors, entity_count)
    records: list[list[ConditionalRecord | None]] = [
        [None] * donors for _ in range(entity_count)
    ]
    support = np.zeros((donors, entity_count), dtype=bool)
    for donor, entity in np.argwhere(flat_mask):
        counts, cells, _ = _fixed_margin_support(flat_tables[donor, entity])
        feasible = cells[:, 0].astype(float)
        if feasible.size < 2:
            continue
        row_zero = int(counts[0].sum())
        column_zero = int(counts[:, 0].sum())
        total = int(counts.sum())
        log_weights = _log_choose(column_zero, feasible) + _log_choose(
            total - column_zero, row_zero - feasible
        )
        log_probability = log_weights - logsumexp(log_weights)
        records[entity][donor] = (
            float(counts[0, 0]),
            feasible,
            log_probability,
        )
        support[donor, entity] = True
    return records, support


def _penalties(
    coefficient_ridge_penalty: float | np.ndarray,
    donor_deviation_penalty: float,
    context_count: int,
) -> tuple[np.ndarray, float]:
    ridge = np.asarray(coefficient_ridge_penalty, dtype=float)
    if ridge.ndim == 0:
        ridge = np.full(context_count, float(ridge), dtype=float)
    if (
        ridge.shape != (context_count,)
        or not np.isfinite(ridge).all()
        or np.any(ridge <= 0.0)
    ):
        raise ValueError(
            "coefficient_ridge_penalty must be finite and positive for every context"
        )
    deviation = float(donor_deviation_penalty)
    if not np.isfinite(deviation) or deviation <= 0.0:
        raise ValueError("donor_deviation_penalty must be finite and positive")
    return ridge, deviation


def _likelihood(
    log_odds: np.ndarray,
    records: list[ConditionalRecord],
) -> tuple[float, np.ndarray, np.ndarray]:
    objective = 0.0
    score = np.empty(len(records), dtype=float)
    precision = np.empty(len(records), dtype=float)
    for index, (theta, record) in enumerate(zip(log_odds, records)):
        observed, feasible, log_probability = record
        centered = feasible - observed
        with np.errstate(over="ignore", invalid="ignore"):
            log_mass = log_probability + centered * theta
        if not np.isfinite(log_mass).all():
            raise CouplingEstimationRefusal(
                "conditional likelihood exceeds finite evaluation range"
            )
        partition = float(logsumexp(log_mass))
        probability = np.exp(log_mass - partition)
        current_score = float(probability @ centered)
        current_precision = float(probability @ np.square(centered - current_score))
        if not np.isfinite(current_precision) or current_precision < 0.0:
            raise CouplingEstimationRefusal(
                "conditional information is not finite and nonnegative"
            )
        objective += partition
        score[index] = current_score
        precision[index] = current_precision
    return float(objective), score, precision


def _evaluate_coordinate(
    coefficient: np.ndarray,
    deviation: np.ndarray,
    design: np.ndarray,
    records: list[ConditionalRecord],
    ridge: np.ndarray,
    deviation_penalty: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    log_odds = design @ coefficient + deviation
    data_objective, score, precision = _likelihood(log_odds, records)
    donor_scale = 1.0 / len(records)
    objective = donor_scale * data_objective
    objective += 0.5 * float(ridge @ np.square(coefficient))
    objective += 0.5 * donor_scale * deviation_penalty * float(deviation @ deviation)
    coefficient_gradient = donor_scale * (design.T @ score) + ridge * coefficient
    deviation_gradient = donor_scale * (score + deviation_penalty * deviation)
    if (
        not np.isfinite(objective)
        or not np.isfinite(coefficient_gradient).all()
        or not np.isfinite(deviation_gradient).all()
    ):
        raise CouplingEstimationRefusal("context-conditioned objective is not finite")
    return (
        float(objective),
        coefficient_gradient,
        deviation_gradient,
        donor_scale * precision,
    )


def _factor_diagnostics(
    values: np.ndarray,
    maximum_condition_number: float,
    label: str,
) -> tuple[float, float, float]:
    scale = float(np.max(np.abs(values)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise CouplingEstimationRefusal(f"{label} is singular")
    if values.ndim == 1:
        minimum = float(np.min(values))
        maximum = float(np.max(values))
    else:
        eigenvalues = np.linalg.eigvalsh(values / scale) * scale
        minimum = float(eigenvalues[0])
        maximum = float(eigenvalues[-1])
    if not np.isfinite(minimum) or minimum <= 0.0 or not np.isfinite(maximum):
        raise CouplingEstimationRefusal(f"{label} is not positive definite")
    condition = maximum / minimum
    if not np.isfinite(condition) or condition > maximum_condition_number:
        raise CouplingEstimationRefusal(f"{label} exceeds the condition-number limit")
    return condition, minimum, maximum


def _newton_factors(
    design: np.ndarray,
    precision: np.ndarray,
    ridge: np.ndarray,
    deviation_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    donor_penalty_curvature = deviation_penalty / design.shape[0]
    donor_curvature = precision + donor_penalty_curvature
    transmitted = precision * donor_penalty_curvature / donor_curvature
    schur = (design.T * transmitted) @ design
    diagonal = np.arange(ridge.size)
    schur[diagonal, diagonal] += ridge
    return schur, donor_curvature


def _gradient_certificate(
    coefficient_gradient: np.ndarray,
    deviation_gradient: np.ndarray,
    design: np.ndarray,
    records: list[ConditionalRecord],
    ridge: np.ndarray,
    deviation_penalty: float,
) -> tuple[float, float]:
    donor_scale = 1.0 / len(records)
    widths = np.asarray(
        [record[1][-1] - record[1][0] for record in records], dtype=float
    )
    coefficient_scale = np.maximum(
        1.0,
        donor_scale * np.sum(np.abs(design) * widths[:, None], axis=0) + ridge,
    )
    deviation_scale = np.maximum(1.0, donor_scale * (widths + deviation_penalty))
    raw = max(
        float(np.max(np.abs(coefficient_gradient))),
        float(np.max(np.abs(deviation_gradient))),
    )
    scaled = max(
        float(np.max(np.abs(coefficient_gradient) / coefficient_scale)),
        float(np.max(np.abs(deviation_gradient) / deviation_scale)),
    )
    return raw, scaled


def fit_context_conditional_log_odds(
    tables: np.ndarray,
    contexts: np.ndarray,
    *,
    support_mask: np.ndarray | None = None,
    donor_deviation_penalty: float = 1.0,
    coefficient_ridge_penalty: float | np.ndarray = 0.1,
    minimum_informative_donors: int = 1,
    maximum_condition_number: float = 1e12,
    maximum_iterations: int = 100,
    tolerance: float = 1e-8,
) -> ContextConditionalCouplingFit:
    """Fit independent context-conditioned coupling coordinates.

    Block Newton steps eliminate donor deviations through a diagonal solve and
    solve the exact coefficient Schur complement. Armijo backtracking accepts
    only finite descent steps. A successful return requires the reported
    scaled gradient norm to meet ``tolerance`` and both final solve factors to
    meet ``maximum_condition_number``. The scaling normalizes each score by
    its feasible-support width.
    """

    values, design, requested_support, entity_shape = _validated_inputs(
        tables, contexts, support_mask
    )
    ridge, deviation_penalty = _penalties(
        coefficient_ridge_penalty,
        donor_deviation_penalty,
        design.shape[1],
    )
    minimum = int(minimum_informative_donors)
    condition_limit = float(maximum_condition_number)
    iteration_limit = int(maximum_iterations)
    threshold = float(tolerance)
    if minimum < 1 or minimum > values.shape[0]:
        raise ValueError(
            "minimum_informative_donors must be between one and donor count"
        )
    if not np.isfinite(condition_limit) or condition_limit <= 1.0:
        raise ValueError("maximum_condition_number must be finite and above one")
    if iteration_limit < 1 or not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("maximum_iterations and tolerance must be positive")

    record_grid, flat_support = _record_grid(values, requested_support)
    support_count = flat_support.sum(axis=0)
    if np.any(support_count < minimum):
        raise CouplingEstimationRefusal(
            "too few informative donors for at least one entity"
        )

    donors = values.shape[0]
    entities = flat_support.shape[1]
    contexts_count = design.shape[1]
    coefficient = np.zeros((contexts_count, entities), dtype=float)
    deviation = np.zeros((donors, entities), dtype=float)
    data_precision = np.zeros((donors, entities), dtype=float)
    coefficient_gradient = np.zeros_like(coefficient)
    deviation_gradient = np.zeros_like(deviation)
    coordinate_objective = np.empty(entities, dtype=float)
    iterations = np.empty(entities, dtype=int)
    schur_condition = np.empty(entities, dtype=float)
    donor_condition = np.empty(entities, dtype=float)
    minimum_schur = np.empty(entities, dtype=float)
    maximum_schur = np.empty(entities, dtype=float)
    minimum_donor = np.empty(entities, dtype=float)
    maximum_donor = np.empty(entities, dtype=float)
    raw_gradient_norm = 0.0
    scaled_gradient_norm = 0.0

    for entity, donor_records in enumerate(record_grid):
        retained = np.flatnonzero(flat_support[:, entity])
        records = [donor_records[index] for index in retained]
        if any(record is None for record in records):
            raise AssertionError("retained donor lacks a conditional record")
        typed_records = [record for record in records if record is not None]
        retained_design = design[retained]
        beta = np.zeros(contexts_count, dtype=float)
        delta = np.zeros(retained.size, dtype=float)
        converged = False

        for iteration in range(iteration_limit + 1):
            objective, beta_gradient, delta_gradient, precision = _evaluate_coordinate(
                beta,
                delta,
                retained_design,
                typed_records,
                ridge,
                deviation_penalty,
            )
            _, scaled_gradient = _gradient_certificate(
                beta_gradient,
                delta_gradient,
                retained_design,
                typed_records,
                ridge,
                deviation_penalty,
            )
            if scaled_gradient <= threshold:
                converged = True
                iterations[entity] = iteration
                break
            if iteration == iteration_limit:
                break

            schur, donor_curvature = _newton_factors(
                retained_design, precision, ridge, deviation_penalty
            )
            _factor_diagnostics(schur, condition_limit, "coefficient Schur factor")
            _factor_diagnostics(
                donor_curvature, condition_limit, "donor-curvature factor"
            )
            right_hand_side = -beta_gradient
            right_hand_side += (retained_design.T * precision) @ (
                delta_gradient / donor_curvature
            )
            beta_step = np.linalg.solve(schur, right_hand_side)
            delta_step = (
                -delta_gradient - precision * (retained_design @ beta_step)
            ) / donor_curvature
            directional_derivative = float(
                beta_gradient @ beta_step + delta_gradient @ delta_step
            )
            if not np.isfinite(directional_derivative) or directional_derivative >= 0.0:
                raise CouplingEstimationRefusal(
                    "context-conditioned Newton direction is not finite descent"
                )

            step_size = 1.0
            for _ in range(48):
                candidate_beta = beta + step_size * beta_step
                candidate_delta = delta + step_size * delta_step
                try:
                    candidate_objective = _evaluate_coordinate(
                        candidate_beta,
                        candidate_delta,
                        retained_design,
                        typed_records,
                        ridge,
                        deviation_penalty,
                    )[0]
                except CouplingEstimationRefusal:
                    candidate_objective = np.inf
                if (
                    np.isfinite(candidate_objective)
                    and candidate_objective
                    <= objective + 1e-4 * step_size * directional_derivative
                ):
                    beta = candidate_beta
                    delta = candidate_delta
                    break
                step_size *= 0.5
            else:
                raise CouplingEstimationRefusal(
                    "context-conditioned Newton line search found no valid step"
                )

        if not converged:
            raise CouplingEstimationRefusal(
                "context-conditioned optimizer missed the gradient certificate"
            )
        final = _evaluate_coordinate(
            beta,
            delta,
            retained_design,
            typed_records,
            ridge,
            deviation_penalty,
        )
        objective, beta_gradient, delta_gradient, precision = final
        gradient_norm, scaled_gradient = _gradient_certificate(
            beta_gradient,
            delta_gradient,
            retained_design,
            typed_records,
            ridge,
            deviation_penalty,
        )
        if scaled_gradient > threshold:
            raise CouplingEstimationRefusal(
                "context-conditioned optimizer missed the gradient certificate"
            )
        schur, donor_curvature = _newton_factors(
            retained_design, precision, ridge, deviation_penalty
        )
        (
            schur_condition[entity],
            minimum_schur[entity],
            maximum_schur[entity],
        ) = _factor_diagnostics(schur, condition_limit, "coefficient Schur factor")
        (
            donor_condition[entity],
            minimum_donor[entity],
            maximum_donor[entity],
        ) = _factor_diagnostics(
            donor_curvature, condition_limit, "donor-curvature factor"
        )
        coefficient[:, entity] = beta
        deviation[retained, entity] = delta
        data_precision[retained, entity] = precision
        coefficient_gradient[:, entity] = beta_gradient
        deviation_gradient[retained, entity] = delta_gradient
        coordinate_objective[entity] = objective
        raw_gradient_norm = max(raw_gradient_norm, gradient_norm)
        scaled_gradient_norm = max(scaled_gradient_norm, scaled_gradient)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        context_log_odds = design @ coefficient
    if not np.isfinite(context_log_odds).all():
        raise CouplingEstimationRefusal("context-conditioned prediction is not finite")
    donor_log_odds = context_log_odds + deviation
    coefficient_shape = (contexts_count, *entity_shape)
    donor_shape = (donors, *entity_shape)
    entity_output_shape = entity_shape if entity_shape else ()
    return ContextConditionalCouplingFit(
        coefficient=coefficient.reshape(coefficient_shape),
        context_log_odds=context_log_odds.reshape(donor_shape),
        donor_log_odds=donor_log_odds.reshape(donor_shape),
        donor_deviation=deviation.reshape(donor_shape),
        donor_support=flat_support.reshape(donor_shape),
        support_count=support_count.reshape(entity_output_shape),
        donor_data_precision=data_precision.reshape(donor_shape),
        coefficient_gradient=coefficient_gradient.reshape(coefficient_shape),
        donor_deviation_gradient=deviation_gradient.reshape(donor_shape),
        coordinate_objective=coordinate_objective.reshape(entity_output_shape),
        objective=float(np.sum(coordinate_objective)),
        gradient_norm=raw_gradient_norm,
        scaled_gradient_norm=scaled_gradient_norm,
        schur_condition_number=schur_condition.reshape(entity_output_shape),
        donor_curvature_condition_number=donor_condition.reshape(entity_output_shape),
        minimum_schur_eigenvalue=minimum_schur.reshape(entity_output_shape),
        maximum_schur_eigenvalue=maximum_schur.reshape(entity_output_shape),
        minimum_donor_curvature=minimum_donor.reshape(entity_output_shape),
        maximum_donor_curvature=maximum_donor.reshape(entity_output_shape),
        iterations=iterations.reshape(entity_output_shape),
        converged=True,
        optimizer="coordinate_block_newton_schur_backtracking",
        donor_deviation_penalty=deviation_penalty,
        coefficient_ridge_penalty=ridge.copy(),
        maximum_condition_number=condition_limit,
        gradient_tolerance=threshold,
    )


def predict_context_log_odds(
    coefficient: np.ndarray,
    contexts: np.ndarray,
) -> np.ndarray:
    """Apply fitted context coefficients to one or more new context rows."""

    beta = np.asarray(coefficient, dtype=float)
    design = np.asarray(contexts, dtype=float)
    if beta.ndim < 1 or beta.shape[0] < 1 or not np.isfinite(beta).all():
        raise ValueError("coefficient must be finite with context first")
    if design.ndim < 1 or design.shape[-1] != beta.shape[0]:
        raise ValueError("contexts must end in the fitted context dimension")
    if not np.isfinite(design).all():
        raise ValueError("contexts must be finite")
    return np.tensordot(design, beta, axes=([-1], [0]))
