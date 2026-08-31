"""Profiled 2 x 2 Poisson log-linear interaction comparators.

The model is ``log(mu[d,e,a,b]) = alpha[d,e] + rho[d,e] a + kappa[d,e] b +
theta[g,e] ab``.  A single interaction is shared either across all donors or
within each supplied donor group.  Profiling the donor/entity nuisance effects
is equivalent to fitting a positive table at each donor's observed margins
with the interaction fixed.  The remaining score is one-dimensional and
monotone.

The unpenalized estimator uses no continuity correction or penalty.
Degenerate-margin tables remain in its profile likelihood and contribute zero
interaction score and information.  A separate mean-profile ridge estimator
excludes zero-information tables from its donor mean and remains finite at
separation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln, xlogy


_MAX_EXACT_FLOAT_INTEGER = 2**53 - 1
_ROOT_XTOL = 1e-12
_RIDGE_PROFILE_BRACKET = 16.0


__all__ = [
    "PoissonLoglinearFit",
    "PoissonLoglinearReconstruction",
    "PoissonLoglinearRefusal",
    "RidgeProfiledPoissonFit",
    "fit_poisson_loglinear_interaction",
    "fit_ridge_profiled_poisson_interaction",
    "reconstruct_poisson_tables",
]


class PoissonLoglinearRefusal(ValueError):
    """Raised when a profiled interaction cannot be certified."""


@dataclass(frozen=True)
class PoissonLoglinearFit:
    """Unpenalized profiled interaction estimates and fit certificates.

    Every array begins with the group axis.  A pooled fit has the single label
    ``"__pooled__"`` and therefore a leading dimension of one.
    """

    log_odds: np.ndarray
    group_labels: tuple[object, ...]
    profile_log_likelihood: float
    group_entity_log_likelihood: np.ndarray
    score: np.ndarray
    data_information: np.ndarray
    informative_table_count: np.ndarray
    degenerate_table_count: np.ndarray
    included_table_count: np.ndarray
    lower_boundary_gap: np.ndarray
    upper_boundary_gap: np.ndarray
    root_iterations: np.ndarray
    maximum_absolute_score: float
    maximum_scaled_score: float
    maximum_absolute_row_margin_error: float
    maximum_absolute_column_margin_error: float
    maximum_absolute_log_odds_error: float
    minimum_positive_fitted_mean: float
    score_tolerance: float
    certificate_tolerance: float
    pseudocount: float
    converged: bool
    estimator: str


@dataclass(frozen=True)
class RidgeProfiledPoissonFit:
    """Finite mean-profile ridge interactions and numerical certificates.

    Every array begins with the group axis. Zero-information group--entity
    coordinates have interaction zero and status ``"NO_INFORMATION"``.
    """

    log_odds: np.ndarray
    group_labels: tuple[object, ...]
    status: np.ndarray
    mean_profile_log_likelihood: np.ndarray
    penalized_objective: np.ndarray
    mean_score: np.ndarray
    penalized_score: np.ndarray
    mean_data_information: np.ndarray
    penalized_information: np.ndarray
    mean_margin_width: np.ndarray
    informative_table_count: np.ndarray
    degenerate_table_count: np.ndarray
    included_table_count: np.ndarray
    bracket_lower: np.ndarray
    bracket_upper: np.ndarray
    bracket_lower_score: np.ndarray
    bracket_upper_score: np.ndarray
    root_iterations: np.ndarray
    maximum_absolute_penalized_score: float
    maximum_scaled_penalized_score: float
    maximum_absolute_row_margin_error: float
    maximum_absolute_column_margin_error: float
    maximum_absolute_log_odds_error: float
    minimum_positive_fitted_mean: Optional[float]
    ridge_penalty: float
    score_tolerance: float
    certificate_tolerance: float
    bracket_bound: float
    converged: bool
    estimator: str


@dataclass(frozen=True)
class PoissonLoglinearReconstruction:
    """Same-margin Poisson means and numerical certificates."""

    table: np.ndarray
    transported_log_odds: np.ndarray
    reconstructed_log_odds: np.ndarray
    informative_margin_mask: np.ndarray
    informative_margin_count: int
    degenerate_margin_count: int
    root_iterations: np.ndarray
    maximum_absolute_row_margin_error: float
    maximum_absolute_column_margin_error: float
    maximum_absolute_log_odds_error: float
    minimum_fitted_mean: float
    maximum_fitted_mean: float
    margin_tolerance: float
    log_odds_tolerance: float
    converged: bool
    reconstruction: str


@dataclass(frozen=True)
class _ProfiledTable:
    mean: np.ndarray
    score: float
    information: float
    log_likelihood: float
    row_error: float
    column_error: float
    log_odds_error: Optional[float]
    minimum_positive_mean: float
    iterations: int
    informative: bool


@dataclass(frozen=True)
class _RidgeProfiledEntity:
    log_odds: float
    status: str
    evaluations: tuple[_ProfiledTable, ...]
    mean_profile_log_likelihood: float
    penalized_objective: float
    mean_score: float
    penalized_score: float
    mean_data_information: float
    penalized_information: float
    mean_margin_width: float
    informative_table_count: int
    degenerate_table_count: int
    bracket_lower: float
    bracket_upper: float
    bracket_lower_score: float
    bracket_upper_score: float
    root_iterations: int


def _validated_tables(tables: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    raw = np.asarray(tables)
    if raw.ndim < 3 or raw.shape[0] < 1 or raw.shape[-2:] != (2, 2):
        raise ValueError("tables must have shape (donor, ..., 2, 2)")
    if any(size < 1 for size in raw.shape[1:-2]):
        raise ValueError("table entity axes must be nonempty")
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b" or np.any(raw < 0):
        raise ValueError("tables must contain nonnegative integer counts")
    if raw.size and int(raw.max()) > _MAX_EXACT_FLOAT_INTEGER:
        raise ValueError("table count exceeds exact floating-point integer range")
    if any(
        sum(int(value) for value in table) > _MAX_EXACT_FLOAT_INTEGER
        for table in raw.reshape(-1, 4)
    ):
        raise ValueError("table total exceeds exact floating-point integer range")
    values = raw.astype(float)
    return values, raw.shape[1:-2]


def _group_partition(
    groups: Optional[np.ndarray], donor_count: int
) -> tuple[tuple[object, ...], np.ndarray]:
    if groups is None:
        return ("__pooled__",), np.zeros(donor_count, dtype=int)
    raw = np.asarray(groups, dtype=object)
    if raw.ndim != 1 or len(raw) != donor_count:
        raise ValueError("groups must be a one-dimensional donor-length axis")

    labels: list[object] = []
    lookup: dict[tuple[type, object], int] = {}
    inverse = np.empty(donor_count, dtype=int)
    for index, label in enumerate(raw.tolist()):
        if (
            label is None
            or not np.isscalar(label)
            or (
                isinstance(label, (float, np.floating))
                and not math.isfinite(float(label))
            )
        ):
            raise ValueError("group labels must be finite non-null scalars")
        key = (type(label), label)
        try:
            position = lookup.get(key)
        except TypeError as error:
            raise ValueError("group labels must be hashable scalars") from error
        if position is None:
            position = len(labels)
            labels.append(label)
            lookup[key] = position
        inverse[index] = position
    if not labels:
        raise ValueError("at least one group is required")
    return tuple(labels), inverse


def _margins(table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return table.sum(axis=1), table.sum(axis=0)


def _margin_interval(
    rows: np.ndarray, columns: np.ndarray
) -> tuple[float, float, float]:
    total = float(rows.sum())
    lower = max(0.0, float(rows[0] + columns[0] - total))
    upper = min(float(rows[0]), float(columns[0]))
    return total, lower, upper


def _validated_log_odds_tolerance(value: float, name: str) -> float:
    tolerance = float(value)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return tolerance


def _upper_left_residual(
    upper_left: float,
    rows: np.ndarray,
    columns: np.ndarray,
    total: float,
    log_odds: float,
) -> float:
    cells = (
        upper_left,
        float(rows[0] - upper_left),
        float(columns[0] - upper_left),
        float(total - rows[0] - columns[0] + upper_left),
    )
    if cells[0] <= 0.0 or cells[3] <= 0.0:
        return -math.inf
    if cells[1] <= 0.0 or cells[2] <= 0.0:
        return math.inf
    return (
        math.log(cells[0])
        + math.log(cells[3])
        - math.log(cells[1])
        - math.log(cells[2])
        - log_odds
    )


def _same_margin_mean(
    log_odds: float, rows: np.ndarray, columns: np.ndarray
) -> tuple[np.ndarray, int, bool]:
    total, lower, upper = _margin_interval(rows, columns)
    if lower == upper:
        upper_left = lower
        iterations = 0
        informative = False
    elif log_odds == 0.0:
        upper_left = float(rows[0] * columns[0] / total)
        iterations = 0
        informative = True
    else:
        left = np.nextafter(lower, upper)
        right = np.nextafter(upper, lower)
        try:
            upper_left, result = brentq(
                _upper_left_residual,
                left,
                right,
                args=(rows, columns, total, log_odds),
                xtol=_ROOT_XTOL,
                rtol=4.0 * np.finfo(float).eps,
                maxiter=256,
                full_output=True,
                disp=False,
            )
        except (ValueError, FloatingPointError, OverflowError) as error:
            raise PoissonLoglinearRefusal(
                "same-margin Poisson root is not numerically finite"
            ) from error
        if not result.converged:
            raise PoissonLoglinearRefusal(
                "same-margin Poisson root solver did not converge"
            )
        iterations = int(result.iterations)
        informative = True
    mean = np.asarray(
        [
            [upper_left, rows[0] - upper_left],
            [columns[0] - upper_left, total - rows[0] - columns[0] + upper_left],
        ],
        dtype=float,
    )
    scale = max(1.0, total)
    if not np.isfinite(mean).all() or np.any(
        mean < -32.0 * np.finfo(float).eps * scale
    ):
        raise PoissonLoglinearRefusal("same-margin Poisson mean is invalid")
    mean = np.maximum(mean, 0.0)
    return mean, iterations, informative


def _profiled_table(observed: np.ndarray, log_odds: float) -> _ProfiledTable:
    rows, columns = _margins(observed)
    mean, iterations, informative = _same_margin_mean(log_odds, rows, columns)
    row_error = float(np.max(np.abs(mean.sum(axis=1) - rows)))
    column_error = float(np.max(np.abs(mean.sum(axis=0) - columns)))
    positive = mean > 0.0
    minimum_positive = float(mean[positive].min()) if np.any(positive) else math.inf
    log_likelihood = float(
        np.sum(xlogy(observed, mean) - mean - gammaln(observed + 1.0))
    )
    if not math.isfinite(log_likelihood):
        raise PoissonLoglinearRefusal("profile log likelihood is not finite")

    if informative:
        if not np.all(positive):
            raise PoissonLoglinearRefusal(
                "finite interaction produced a nonpositive fitted mean"
            )
        reconstructed = float(
            math.log(mean[0, 0])
            + math.log(mean[1, 1])
            - math.log(mean[0, 1])
            - math.log(mean[1, 0])
        )
        log_odds_error = abs(reconstructed - log_odds)
        information = 1.0 / float(np.sum(1.0 / mean))
        score = float(observed[0, 0] - mean[0, 0])
    else:
        log_odds_error = None
        information = 0.0
        score = 0.0
    return _ProfiledTable(
        mean=mean,
        score=score,
        information=information,
        log_likelihood=log_likelihood,
        row_error=row_error,
        column_error=column_error,
        log_odds_error=log_odds_error,
        minimum_positive_mean=minimum_positive,
        iterations=iterations,
        informative=informative,
    )


def _profile_score(log_odds: float, tables: np.ndarray) -> float:
    return float(sum(_profiled_table(table, log_odds).score for table in tables))


def _ridge_profile_evaluation(
    log_odds: float, tables: np.ndarray, ridge_penalty: float
) -> tuple[tuple[_ProfiledTable, ...], float, float]:
    evaluations = tuple(_profiled_table(table, log_odds) for table in tables)
    mean_score = float(np.mean([item.score for item in evaluations]))
    return evaluations, mean_score, mean_score - ridge_penalty * log_odds


def _fit_ridge_group_entity(
    tables: np.ndarray, ridge_penalty: float, score_tolerance: float
) -> _RidgeProfiledEntity:
    informative = np.empty(len(tables), dtype=bool)
    widths = np.empty(len(tables), dtype=float)
    for index, table in enumerate(tables):
        rows, columns = _margins(table)
        _, lower, upper = _margin_interval(rows, columns)
        informative[index] = lower < upper
        widths[index] = upper - lower

    selected = tables[informative]
    informative_count = len(selected)
    degenerate_count = len(tables) - informative_count
    if informative_count == 0:
        return _RidgeProfiledEntity(
            log_odds=0.0,
            status="NO_INFORMATION",
            evaluations=(),
            mean_profile_log_likelihood=0.0,
            penalized_objective=0.0,
            mean_score=0.0,
            penalized_score=0.0,
            mean_data_information=0.0,
            penalized_information=ridge_penalty,
            mean_margin_width=0.0,
            informative_table_count=0,
            degenerate_table_count=degenerate_count,
            bracket_lower=0.0,
            bracket_upper=0.0,
            bracket_lower_score=0.0,
            bracket_upper_score=0.0,
            root_iterations=0,
        )

    mean_width = float(np.mean(widths[informative]))
    zero_evaluations, zero_mean_score, zero_penalized_score = _ridge_profile_evaluation(
        0.0, selected, ridge_penalty
    )
    scale = max(1.0, mean_width)
    if abs(zero_penalized_score) <= score_tolerance * scale:
        root = 0.0
        evaluations = zero_evaluations
        mean_score = zero_mean_score
        penalized_score = zero_penalized_score
        bracket_lower = bracket_upper = 0.0
        bracket_lower_score = bracket_upper_score = zero_penalized_score
        root_iterations = 0
    else:
        bracket_lower = -_RIDGE_PROFILE_BRACKET
        bracket_upper = _RIDGE_PROFILE_BRACKET
        _, _, bracket_lower_score = _ridge_profile_evaluation(
            bracket_lower, selected, ridge_penalty
        )
        _, _, bracket_upper_score = _ridge_profile_evaluation(
            bracket_upper, selected, ridge_penalty
        )
        if bracket_lower_score < 0.0 or bracket_upper_score > 0.0:
            raise PoissonLoglinearRefusal(
                "mean-profile ridge Poisson root lies outside the frozen bracket"
            )
        try:
            root, result = brentq(
                lambda value: _ridge_profile_evaluation(
                    float(value), selected, ridge_penalty
                )[2],
                bracket_lower,
                bracket_upper,
                xtol=_ROOT_XTOL,
                rtol=4.0 * np.finfo(float).eps,
                maxiter=256,
                full_output=True,
                disp=False,
            )
        except (ValueError, FloatingPointError, OverflowError) as error:
            raise PoissonLoglinearRefusal(
                "mean-profile ridge Poisson root is not numerically finite"
            ) from error
        if not result.converged:
            raise PoissonLoglinearRefusal(
                "mean-profile ridge Poisson root solver did not converge"
            )
        root_iterations = int(result.iterations)
        evaluations, mean_score, penalized_score = _ridge_profile_evaluation(
            float(root), selected, ridge_penalty
        )

    mean_information = float(np.mean([item.information for item in evaluations]))
    mean_log_likelihood = float(np.mean([item.log_likelihood for item in evaluations]))
    penalized_information = mean_information + ridge_penalty
    penalized_objective = -mean_log_likelihood + 0.5 * ridge_penalty * float(root) ** 2
    if (
        not math.isfinite(float(root))
        or not math.isfinite(mean_score)
        or not math.isfinite(penalized_score)
        or not math.isfinite(mean_information)
        or not math.isfinite(penalized_information)
        or not math.isfinite(penalized_objective)
        or mean_information <= 0.0
        or penalized_information <= ridge_penalty
    ):
        raise PoissonLoglinearRefusal(
            "mean-profile ridge Poisson solution lacks finite positive information"
        )
    if abs(penalized_score) > score_tolerance * scale:
        raise PoissonLoglinearRefusal(
            "mean-profile ridge Poisson solution misses the score certificate"
        )
    return _RidgeProfiledEntity(
        log_odds=float(root),
        status="FINITE",
        evaluations=evaluations,
        mean_profile_log_likelihood=mean_log_likelihood,
        penalized_objective=penalized_objective,
        mean_score=mean_score,
        penalized_score=penalized_score,
        mean_data_information=mean_information,
        penalized_information=penalized_information,
        mean_margin_width=mean_width,
        informative_table_count=informative_count,
        degenerate_table_count=degenerate_count,
        bracket_lower=bracket_lower,
        bracket_upper=bracket_upper,
        bracket_lower_score=bracket_lower_score,
        bracket_upper_score=bracket_upper_score,
        root_iterations=root_iterations,
    )


def _bracket_profile_score(tables: np.ndarray) -> tuple[float, float]:
    left = -1.0
    right = 1.0
    for _ in range(9):
        left_score = _profile_score(left, tables)
        right_score = _profile_score(right, tables)
        if left_score >= 0.0 and right_score <= 0.0:
            return left, right
        if left_score < 0.0:
            left *= 2.0
        if right_score > 0.0:
            right *= 2.0
    raise PoissonLoglinearRefusal(
        "finite profiled Poisson interaction could not be bracketed"
    )


def _fit_group_entity(
    tables: np.ndarray, score_tolerance: float
) -> tuple[float, list[_ProfiledTable], float, float, int, float, float]:
    observed_upper_left = tables[:, 0, 0]
    lower = np.empty(len(tables), dtype=float)
    upper = np.empty(len(tables), dtype=float)
    informative = np.empty(len(tables), dtype=bool)
    for index, table in enumerate(tables):
        rows, columns = _margins(table)
        _, lower[index], upper[index] = _margin_interval(rows, columns)
        informative[index] = lower[index] < upper[index]
    if not np.any(informative):
        raise PoissonLoglinearRefusal(
            "no training table contributes interaction information"
        )

    observed_sum = float(observed_upper_left[informative].sum())
    lower_sum = float(lower[informative].sum())
    upper_sum = float(upper[informative].sum())
    lower_gap = observed_sum - lower_sum
    upper_gap = upper_sum - observed_sum
    if lower_gap <= 0.0 or upper_gap <= 0.0:
        raise PoissonLoglinearRefusal(
            "unpenalized profiled Poisson interaction has a boundary or infinite MLE"
        )

    width = float(np.sum(upper[informative] - lower[informative]))
    zero_score = _profile_score(0.0, tables)
    if abs(zero_score) <= score_tolerance * max(1.0, width):
        root = 0.0
        root_iterations = 0
    else:
        left, right = _bracket_profile_score(tables)
        try:
            root, result = brentq(
                lambda value: _profile_score(float(value), tables),
                left,
                right,
                xtol=_ROOT_XTOL,
                rtol=4.0 * np.finfo(float).eps,
                maxiter=256,
                full_output=True,
                disp=False,
            )
        except (ValueError, FloatingPointError, OverflowError) as error:
            raise PoissonLoglinearRefusal(
                "profiled Poisson score root is not numerically finite"
            ) from error
        if not result.converged:
            raise PoissonLoglinearRefusal(
                "profiled Poisson score root solver did not converge"
            )
        root_iterations = int(result.iterations)

    evaluations = [_profiled_table(table, float(root)) for table in tables]
    final_score = float(sum(item.score for item in evaluations))
    information = float(sum(item.information for item in evaluations))
    if (
        not math.isfinite(float(root))
        or not math.isfinite(final_score)
        or not math.isfinite(information)
        or information <= 0.0
    ):
        raise PoissonLoglinearRefusal(
            "profiled Poisson solution lacks finite positive information"
        )
    if abs(final_score) > score_tolerance * max(1.0, width):
        raise PoissonLoglinearRefusal(
            "profiled Poisson solution misses the score certificate"
        )
    return (
        float(root),
        evaluations,
        final_score,
        information,
        root_iterations,
        lower_gap,
        upper_gap,
    )


def fit_poisson_loglinear_interaction(
    tables: np.ndarray,
    groups: Optional[np.ndarray] = None,
    *,
    score_tolerance: float = 1e-10,
    certificate_tolerance: float = 1e-8,
) -> PoissonLoglinearFit:
    """Fit pooled or group-specific donor-profiled Poisson interactions.

    ``tables`` has shape ``(donor, ..., 2, 2)``.  With ``groups=None``, one
    interaction is fitted per entity across all donors.  Otherwise, one is
    fitted per group and entity.  Every donor table is evaluated, including
    tables whose margins determine a single table and hence no interaction
    information.
    """

    values, entity_shape = _validated_tables(tables)
    threshold = _validated_log_odds_tolerance(score_tolerance, "score_tolerance")
    certificate = _validated_log_odds_tolerance(
        certificate_tolerance, "certificate_tolerance"
    )
    labels, group_index = _group_partition(groups, values.shape[0])
    group_shape = (len(labels),) + entity_shape
    log_odds = np.empty(group_shape, dtype=float)
    group_log_likelihood = np.empty(group_shape, dtype=float)
    score = np.empty(group_shape, dtype=float)
    information = np.empty(group_shape, dtype=float)
    informative_count = np.empty(group_shape, dtype=int)
    degenerate_count = np.empty(group_shape, dtype=int)
    included_count = np.empty(group_shape, dtype=int)
    lower_gap = np.empty(group_shape, dtype=float)
    upper_gap = np.empty(group_shape, dtype=float)
    root_iterations = np.empty(group_shape, dtype=int)
    maximum_row_error = 0.0
    maximum_column_error = 0.0
    maximum_log_odds_error = 0.0
    minimum_positive_mean = math.inf

    flattened = values.reshape((values.shape[0], -1, 2, 2))
    for group in range(len(labels)):
        donors = np.flatnonzero(group_index == group)
        for entity in range(flattened.shape[1]):
            current = flattened[donors, entity]
            (
                fitted,
                evaluations,
                final_score,
                data_information,
                iterations,
                lower_boundary_gap,
                upper_boundary_gap,
            ) = _fit_group_entity(current, threshold)
            output_index = (group,) + np.unravel_index(entity, entity_shape or (1,))
            if not entity_shape:
                output_index = (group,)
            log_odds[output_index] = fitted
            group_log_likelihood[output_index] = sum(
                item.log_likelihood for item in evaluations
            )
            score[output_index] = final_score
            information[output_index] = data_information
            informative_count[output_index] = sum(
                item.informative for item in evaluations
            )
            degenerate_count[output_index] = sum(
                not item.informative for item in evaluations
            )
            included_count[output_index] = len(evaluations)
            lower_gap[output_index] = lower_boundary_gap
            upper_gap[output_index] = upper_boundary_gap
            root_iterations[output_index] = iterations
            maximum_row_error = max(
                maximum_row_error, *(item.row_error for item in evaluations)
            )
            maximum_column_error = max(
                maximum_column_error, *(item.column_error for item in evaluations)
            )
            maximum_log_odds_error = max(
                maximum_log_odds_error,
                *(
                    item.log_odds_error
                    for item in evaluations
                    if item.log_odds_error is not None
                ),
            )
            minimum_positive_mean = min(
                minimum_positive_mean,
                *(item.minimum_positive_mean for item in evaluations),
            )

    scaled_score = np.abs(score) / np.maximum(1.0, lower_gap + upper_gap)
    if (
        maximum_row_error > certificate
        or maximum_column_error > certificate
        or maximum_log_odds_error > certificate
        or not math.isfinite(minimum_positive_mean)
        or minimum_positive_mean <= 0.0
    ):
        raise PoissonLoglinearRefusal(
            "profiled Poisson fit misses its reconstruction certificate"
        )
    return PoissonLoglinearFit(
        log_odds=log_odds,
        group_labels=labels,
        profile_log_likelihood=float(group_log_likelihood.sum()),
        group_entity_log_likelihood=group_log_likelihood,
        score=score,
        data_information=information,
        informative_table_count=informative_count,
        degenerate_table_count=degenerate_count,
        included_table_count=included_count,
        lower_boundary_gap=lower_gap,
        upper_boundary_gap=upper_gap,
        root_iterations=root_iterations,
        maximum_absolute_score=float(np.max(np.abs(score))),
        maximum_scaled_score=float(np.max(scaled_score)),
        maximum_absolute_row_margin_error=maximum_row_error,
        maximum_absolute_column_margin_error=maximum_column_error,
        maximum_absolute_log_odds_error=maximum_log_odds_error,
        minimum_positive_fitted_mean=minimum_positive_mean,
        score_tolerance=threshold,
        certificate_tolerance=certificate,
        pseudocount=0.0,
        converged=True,
        estimator=("unpenalized donor-profiled 2x2 Poisson log-linear interaction"),
    )


def fit_ridge_profiled_poisson_interaction(
    tables: np.ndarray,
    groups: Optional[np.ndarray] = None,
    *,
    ridge_penalty: float = 0.01,
    score_tolerance: float = 1e-10,
    certificate_tolerance: float = 1e-8,
) -> RidgeProfiledPoissonFit:
    """Fit finite group-specific interactions by mean-profile ridge Poisson.

    The likelihood and score are averaged over informative donor tables before
    applying the coefficient ridge. This gives every informative donor equal
    weight and makes complete-panel duplication leave the estimate unchanged.
    Zero-information group--entity coordinates return interaction zero.
    """

    values, entity_shape = _validated_tables(tables)
    penalty = float(ridge_penalty)
    if not math.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("ridge_penalty must be finite and positive")
    threshold = _validated_log_odds_tolerance(score_tolerance, "score_tolerance")
    certificate = _validated_log_odds_tolerance(
        certificate_tolerance, "certificate_tolerance"
    )
    labels, group_index = _group_partition(groups, values.shape[0])
    group_shape = (len(labels),) + entity_shape
    log_odds = np.empty(group_shape, dtype=float)
    status = np.empty(group_shape, dtype="<U14")
    mean_log_likelihood = np.empty(group_shape, dtype=float)
    penalized_objective = np.empty(group_shape, dtype=float)
    mean_score = np.empty(group_shape, dtype=float)
    penalized_score = np.empty(group_shape, dtype=float)
    mean_information = np.empty(group_shape, dtype=float)
    penalized_information = np.empty(group_shape, dtype=float)
    mean_width = np.empty(group_shape, dtype=float)
    informative_count = np.empty(group_shape, dtype=int)
    degenerate_count = np.empty(group_shape, dtype=int)
    included_count = np.empty(group_shape, dtype=int)
    bracket_lower = np.empty(group_shape, dtype=float)
    bracket_upper = np.empty(group_shape, dtype=float)
    bracket_lower_score = np.empty(group_shape, dtype=float)
    bracket_upper_score = np.empty(group_shape, dtype=float)
    root_iterations = np.empty(group_shape, dtype=int)
    maximum_row_error = 0.0
    maximum_column_error = 0.0
    maximum_log_odds_error = 0.0
    minimum_positive_mean = math.inf

    flattened = values.reshape((values.shape[0], -1, 2, 2))
    for group in range(len(labels)):
        donors = np.flatnonzero(group_index == group)
        for entity in range(flattened.shape[1]):
            fitted = _fit_ridge_group_entity(
                flattened[donors, entity], penalty, threshold
            )
            output_index = (group,) + np.unravel_index(entity, entity_shape or (1,))
            if not entity_shape:
                output_index = (group,)
            log_odds[output_index] = fitted.log_odds
            status[output_index] = fitted.status
            mean_log_likelihood[output_index] = fitted.mean_profile_log_likelihood
            penalized_objective[output_index] = fitted.penalized_objective
            mean_score[output_index] = fitted.mean_score
            penalized_score[output_index] = fitted.penalized_score
            mean_information[output_index] = fitted.mean_data_information
            penalized_information[output_index] = fitted.penalized_information
            mean_width[output_index] = fitted.mean_margin_width
            informative_count[output_index] = fitted.informative_table_count
            degenerate_count[output_index] = fitted.degenerate_table_count
            included_count[output_index] = len(donors)
            bracket_lower[output_index] = fitted.bracket_lower
            bracket_upper[output_index] = fitted.bracket_upper
            bracket_lower_score[output_index] = fitted.bracket_lower_score
            bracket_upper_score[output_index] = fitted.bracket_upper_score
            root_iterations[output_index] = fitted.root_iterations
            for evaluation in fitted.evaluations:
                maximum_row_error = max(maximum_row_error, evaluation.row_error)
                maximum_column_error = max(
                    maximum_column_error, evaluation.column_error
                )
                if evaluation.log_odds_error is not None:
                    maximum_log_odds_error = max(
                        maximum_log_odds_error, evaluation.log_odds_error
                    )
                minimum_positive_mean = min(
                    minimum_positive_mean, evaluation.minimum_positive_mean
                )

    scaled_score = np.abs(penalized_score) / np.maximum(1.0, mean_width)
    has_finite = bool(np.any(status == "FINITE"))
    if (
        not np.isfinite(log_odds).all()
        or not np.isfinite(mean_log_likelihood).all()
        or not np.isfinite(penalized_objective).all()
        or not np.isfinite(mean_score).all()
        or not np.isfinite(penalized_score).all()
        or not np.isfinite(mean_information).all()
        or not np.isfinite(penalized_information).all()
        or not np.isfinite(mean_width).all()
        or float(np.max(scaled_score)) > threshold
        or maximum_row_error > certificate
        or maximum_column_error > certificate
        or maximum_log_odds_error > certificate
        or (has_finite and not math.isfinite(minimum_positive_mean))
        or (has_finite and minimum_positive_mean <= 0.0)
    ):
        raise PoissonLoglinearRefusal(
            "mean-profile ridge Poisson fit misses its numerical certificate"
        )
    return RidgeProfiledPoissonFit(
        log_odds=log_odds,
        group_labels=labels,
        status=status,
        mean_profile_log_likelihood=mean_log_likelihood,
        penalized_objective=penalized_objective,
        mean_score=mean_score,
        penalized_score=penalized_score,
        mean_data_information=mean_information,
        penalized_information=penalized_information,
        mean_margin_width=mean_width,
        informative_table_count=informative_count,
        degenerate_table_count=degenerate_count,
        included_table_count=included_count,
        bracket_lower=bracket_lower,
        bracket_upper=bracket_upper,
        bracket_lower_score=bracket_lower_score,
        bracket_upper_score=bracket_upper_score,
        root_iterations=root_iterations,
        maximum_absolute_penalized_score=float(np.max(np.abs(penalized_score))),
        maximum_scaled_penalized_score=float(np.max(scaled_score)),
        maximum_absolute_row_margin_error=maximum_row_error,
        maximum_absolute_column_margin_error=maximum_column_error,
        maximum_absolute_log_odds_error=maximum_log_odds_error,
        minimum_positive_fitted_mean=(minimum_positive_mean if has_finite else None),
        ridge_penalty=penalty,
        score_tolerance=threshold,
        certificate_tolerance=certificate,
        bracket_bound=_RIDGE_PROFILE_BRACKET,
        converged=True,
        estimator=(
            "finite donor-stratified mean-profile ridge 2x2 Poisson "
            "log-linear interaction"
        ),
    )


def reconstruct_poisson_tables(
    log_odds: np.ndarray,
    row_margins: np.ndarray,
    column_margins: np.ndarray,
    *,
    transport_scale: float = 1.0,
    margin_tolerance: float = 1e-8,
    log_odds_tolerance: float = 1e-8,
) -> PoissonLoglinearReconstruction:
    """Refit row/column nuisance terms at recipient margins.

    The 2 x 2 margin root is the unique table obtained by Poisson iterative
    proportional fitting from a seed with the transported log odds ratio.
    Degenerate margins return their unique compatible table. Their odds ratio
    is undefined, so the log-odds certificate covers informative margins only.
    """

    interaction = np.asarray(log_odds, dtype=float)
    rows = np.asarray(row_margins, dtype=float)
    columns = np.asarray(column_margins, dtype=float)
    scale = float(transport_scale)
    row_tolerance = _validated_log_odds_tolerance(margin_tolerance, "margin_tolerance")
    interaction_tolerance = _validated_log_odds_tolerance(
        log_odds_tolerance, "log_odds_tolerance"
    )
    if interaction.size < 1 or not np.isfinite(interaction).all():
        raise ValueError("log_odds must be a nonempty finite array")
    if rows.shape != interaction.shape + (2,) or columns.shape != rows.shape:
        raise ValueError("margins must have shape log_odds.shape + (2,)")
    if (
        not np.isfinite(rows).all()
        or not np.isfinite(columns).all()
        or np.any(rows < 0.0)
        or np.any(columns < 0.0)
    ):
        raise PoissonLoglinearRefusal(
            "recipient Poisson reconstruction requires finite nonnegative margins"
        )
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError("transport_scale must be finite and nonnegative")
    if not np.allclose(rows.sum(axis=-1), columns.sum(axis=-1), rtol=0.0, atol=1e-12):
        raise ValueError("row and column margins must have equal totals")

    transported = scale * interaction
    table = np.empty(interaction.shape + (2, 2), dtype=float)
    reconstructed = np.full(interaction.shape, np.nan, dtype=float)
    informative_mask = np.zeros(interaction.shape, dtype=bool)
    iterations = np.empty(interaction.shape, dtype=int)
    for index in np.ndindex(interaction.shape):
        current, count, informative = _same_margin_mean(
            float(transported[index]), rows[index], columns[index]
        )
        if informative and np.any(current <= 0.0):
            raise PoissonLoglinearRefusal(
                "informative recipient margins produced a nonpositive mean"
            )
        table[index] = current
        iterations[index] = count
        informative_mask[index] = informative
        if informative:
            reconstructed[index] = (
                math.log(current[0, 0])
                + math.log(current[1, 1])
                - math.log(current[0, 1])
                - math.log(current[1, 0])
            )

    maximum_row_error = float(np.max(np.abs(table.sum(axis=-1) - rows)))
    maximum_column_error = float(np.max(np.abs(table.sum(axis=-2) - columns)))
    maximum_interaction_error = (
        float(
            np.max(
                np.abs(reconstructed[informative_mask] - transported[informative_mask])
            )
        )
        if np.any(informative_mask)
        else 0.0
    )
    if (
        maximum_row_error > row_tolerance
        or maximum_column_error > row_tolerance
        or maximum_interaction_error > interaction_tolerance
    ):
        raise PoissonLoglinearRefusal(
            "recipient Poisson reconstruction misses its certificate"
        )
    return PoissonLoglinearReconstruction(
        table=table,
        transported_log_odds=transported,
        reconstructed_log_odds=reconstructed,
        informative_margin_mask=informative_mask,
        informative_margin_count=int(np.count_nonzero(informative_mask)),
        degenerate_margin_count=int(
            informative_mask.size - np.count_nonzero(informative_mask)
        ),
        root_iterations=iterations,
        maximum_absolute_row_margin_error=maximum_row_error,
        maximum_absolute_column_margin_error=maximum_column_error,
        maximum_absolute_log_odds_error=maximum_interaction_error,
        minimum_fitted_mean=float(table.min()),
        maximum_fitted_mean=float(table.max()),
        margin_tolerance=row_tolerance,
        log_odds_tolerance=interaction_tolerance,
        converged=True,
        reconstruction="same-margin profiled Poisson root (IPF-equivalent)",
    )
