"""Robust common-effect estimation for stratified binary tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

from .heterogeneity_adaptive_coupling import (
    CouplingEstimationRefusal,
    _conditional_records,
    log_odds_to_helmert_coordinate,
)


@dataclass(frozen=True)
class CommonEffectConditionalLogOddsFit:
    """Coordinate-wise exact conditional maximum-likelihood estimates."""

    log_odds: np.ndarray
    helmert_coordinate: np.ndarray
    objective: float
    gradient: np.ndarray
    gradient_norm: float
    scaled_gradient_norm: float
    data_precision: np.ndarray
    support_count: np.ndarray
    root_iterations: np.ndarray
    converged: bool


def _coordinate_evaluation(
    log_odds: float,
    records: list[tuple[float, np.ndarray, np.ndarray]],
) -> tuple[float, float, float]:
    objective = 0.0
    gradient = 0.0
    precision = 0.0
    for observed, support, log_probability in records:
        centered_support = support - observed
        log_mass = log_probability + centered_support * log_odds
        log_partition = float(logsumexp(log_mass))
        probability = np.exp(log_mass - log_partition)
        score = float(probability @ centered_support)
        objective += log_partition
        gradient += score
        precision += float(probability @ np.square(centered_support - score))
    return objective, gradient, precision


def _bracket_score(
    records: list[tuple[float, np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    left = -1.0
    right = 1.0
    for _ in range(13):
        left_score = _coordinate_evaluation(left, records)[1]
        right_score = _coordinate_evaluation(right, records)[1]
        if left_score < 0.0 < right_score:
            return left, right
        if left_score >= 0.0:
            left *= 2.0
        if right_score <= 0.0:
            right *= 2.0
    raise CouplingEstimationRefusal(
        "finite conditional maximum-likelihood estimate could not be bracketed"
    )


def fit_common_effect_conditional_log_odds(
    tables: np.ndarray,
    *,
    minimum_informative_donors: int = 2,
    tolerance: float = 1e-10,
) -> CommonEffectConditionalLogOddsFit:
    """Fit one shared log odds ratio per entity across donor strata.

    Each entity is a one-parameter conditional exponential family. Solving its
    monotone score equation independently avoids a multivariate optimizer and
    returns the exact unregularized common-effect CMLE when that finite maximum
    exists.
    """

    (
        entity_shape,
        records,
        support_count,
        observed_sum,
        lower_sum,
        upper_sum,
    ) = _conditional_records(tables)
    minimum = int(minimum_informative_donors)
    threshold = float(tolerance)
    if minimum < 1:
        raise ValueError("minimum_informative_donors must be positive")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if np.any(support_count < minimum):
        raise CouplingEstimationRefusal(
            "too few informative donors for at least one entity"
        )
    if np.any(observed_sum <= lower_sum) or np.any(observed_sum >= upper_sum):
        raise CouplingEstimationRefusal(
            "unregularized conditional likelihood has a boundary or infinite MLE"
        )

    entity_count = len(records)
    fitted = np.empty(entity_count, dtype=float)
    gradient = np.empty(entity_count, dtype=float)
    precision = np.empty(entity_count, dtype=float)
    iterations = np.empty(entity_count, dtype=int)
    objective = 0.0
    support_width = upper_sum - lower_sum
    for entity, entity_records in enumerate(records):
        zero_evaluation = _coordinate_evaluation(0.0, entity_records)
        if zero_evaluation[1] == 0.0:
            root = 0.0
            root_iterations = 0
        else:
            left, right = _bracket_score(entity_records)
            root, result = brentq(
                lambda value: _coordinate_evaluation(value, entity_records)[1],
                left,
                right,
                xtol=np.finfo(float).tiny,
                rtol=8.0 * np.finfo(float).eps,
                maxiter=256,
                full_output=True,
                disp=False,
            )
            if not result.converged:
                raise CouplingEstimationRefusal(
                    "conditional score root solver did not converge"
                )
            root_iterations = int(result.iterations)
        final_objective, final_gradient, final_precision = _coordinate_evaluation(
            float(root), entity_records
        )
        if (
            not np.isfinite(root)
            or not np.isfinite(final_objective)
            or not np.isfinite(final_gradient)
            or not np.isfinite(final_precision)
            or final_precision <= 0.0
        ):
            raise CouplingEstimationRefusal(
                "conditional score solution lacks finite positive information"
            )
        if abs(final_gradient) > threshold * max(1.0, support_width[entity]):
            raise CouplingEstimationRefusal(
                "conditional score solution misses the gradient certificate"
            )
        fitted[entity] = root
        gradient[entity] = final_gradient
        precision[entity] = final_precision
        iterations[entity] = root_iterations
        objective += final_objective

    expected_shape = entity_shape if entity_shape else ()
    fitted = fitted.reshape(expected_shape)
    gradient = gradient.reshape(expected_shape)
    precision = precision.reshape(expected_shape)
    support_count = support_count.reshape(expected_shape)
    iterations = iterations.reshape(expected_shape)
    gradient_norm = float(np.max(np.abs(gradient)))
    scaled_gradient_norm = float(
        np.max(np.abs(gradient.ravel()) / np.maximum(1.0, support_width))
    )
    return CommonEffectConditionalLogOddsFit(
        log_odds=fitted,
        helmert_coordinate=log_odds_to_helmert_coordinate(fitted),
        objective=float(objective),
        gradient=gradient,
        gradient_norm=gradient_norm,
        scaled_gradient_norm=scaled_gradient_norm,
        data_precision=precision,
        support_count=support_count,
        root_iterations=iterations,
        converged=True,
    )
