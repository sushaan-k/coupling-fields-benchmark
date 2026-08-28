"""Margin-invariant binary coupling estimates for paired assays.

The primary estimator fits full log-odds parameters by exact conditional
noncentral-hypergeometric likelihood with product-hypergraph regularization.
Null-centered Haldane and classical coordinates provide initialization and
matched ablations, not substitutes for the likelihood parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import gammaln, logsumexp

from .classical_residuals import poisson_independence_residuals
from .coupling_fields import normalized_hypergraph_laplacian
from .table_prediction import field_coordinates_to_table


class CouplingEstimationRefusal(ValueError):
    """Raised when support or conditioning cannot identify a stable fit."""


@dataclass(frozen=True)
class DonorBinaryCoupling:
    """A donor-level binary coupling estimate conditional on its margins."""

    observed_log_odds: float
    null_mean_log_odds: float
    centered_log_odds: float
    sampling_variance: float
    null_variance: float
    support_lower: int
    support_upper: int
    supported: bool


@dataclass(frozen=True)
class FixedMarginCoordinate:
    """An exactly centered one-degree-of-freedom classical coordinate."""

    observed_coordinate: float
    null_mean_coordinate: float
    centered_coordinate: float
    null_variance: float
    support_lower: int
    support_upper: int
    supported: bool


@dataclass(frozen=True)
class FixedMarginResiduals:
    """A diagnostic residual matrix centered at its exact null mean.

    The four cells are constrained by shared margins and are not independent
    meta-analysis coordinates. Pool the corresponding one-df signed statistic
    for a fair comparison between estimator families.
    """

    observed_residuals: np.ndarray
    null_mean_residuals: np.ndarray
    centered_residuals: np.ndarray
    null_variance_residuals: np.ndarray
    support_lower: int
    support_upper: int
    supported: bool


@dataclass(frozen=True)
class PauleMandelPool:
    """Entity-wise random-effects estimates with donor heterogeneity."""

    mean: np.ndarray
    variance: np.ndarray
    precision: np.ndarray
    tau_squared: np.ndarray
    q_statistic: np.ndarray
    support_count: np.ndarray
    supported: np.ndarray
    donors: int
    minimum_donors: int
    variance_floor: float


@dataclass(frozen=True)
class BinaryCoordinatePool:
    """A one-df donor-coordinate family pooled under one variance convention."""

    family: str
    variance_convention: str
    donor_coordinate: np.ndarray
    donor_variance: np.ndarray
    donor_support: np.ndarray
    pooled: PauleMandelPool


@dataclass(frozen=True)
class PrecisionWeightedCouplingFit:
    """A precision-weighted coupling field smoothed on a fixed graph."""

    estimate: np.ndarray
    mean: np.ndarray
    raw_precision: np.ndarray
    precision: np.ndarray
    support: np.ndarray
    precision_scale: float
    objective: float
    condition_number: float
    ridge_penalty: float
    graph_penalty: float
    precision_floor: float
    maximum_condition_number: float


@dataclass(frozen=True)
class HeterogeneityAdaptiveCouplingFit:
    """The null-centered Haldane/PM ablation and product-graph field."""

    donor_centered_log_odds: np.ndarray
    donor_sampling_variance: np.ndarray
    donor_pooling_variance: np.ndarray
    donor_null_mean_log_odds: np.ndarray
    donor_support: np.ndarray
    variance_convention: str
    pooled: PauleMandelPool
    structured: PrecisionWeightedCouplingFit


@dataclass(frozen=True)
class ConditionalLogOddsEvaluation:
    """Exact conditional-likelihood value, gradient, and Hessian."""

    objective: float
    gradient: np.ndarray
    hessian: np.ndarray
    data_precision: np.ndarray
    support_count: np.ndarray


@dataclass(frozen=True)
class StructuredConditionalLogOddsFit:
    """A convex structured fit of full binary log-odds parameters."""

    log_odds: np.ndarray
    helmert_coordinate: np.ndarray
    objective: float
    gradient_norm: float
    data_precision: np.ndarray
    support_count: np.ndarray
    condition_number: float
    iterations: int
    converged: bool
    ridge_penalty: float
    graph_penalty: float
    penalty_scale: float


def _integer_binary_table(table: np.ndarray) -> np.ndarray:
    values = np.asarray(table, dtype=float)
    if values.shape != (2, 2):
        raise ValueError("table must have shape (2, 2)")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("table must contain finite nonnegative counts")
    rounded = np.rint(values)
    if not np.array_equal(values, rounded):
        raise ValueError("fixed-margin enumeration requires integer counts")
    if np.any(rounded > np.iinfo(np.int64).max):
        raise ValueError("table counts exceed the supported integer range")
    counts = rounded.astype(np.int64)
    total = sum(int(value) for value in counts.flat)
    if total <= 0:
        raise ValueError("table must have positive total count")
    if total > 2**53 - 1:
        raise ValueError("table total exceeds exact floating-point integer range")
    return counts


def _log_choose(total: int, selected: np.ndarray) -> np.ndarray:
    selected_float = np.asarray(selected, dtype=float)
    return (
        gammaln(total + 1.0)
        - gammaln(selected_float + 1.0)
        - gammaln(total - selected_float + 1.0)
    )


def _fixed_margin_support(
    table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = _integer_binary_table(table)
    row_one = int(counts[0].sum())
    column_one = int(counts[:, 0].sum())
    total = sum(int(value) for value in counts.flat)
    lower = max(0, row_one + column_one - total)
    upper = min(row_one, column_one)
    upper_left = np.arange(lower, upper + 1, dtype=float)
    log_weights = _log_choose(column_one, upper_left) + _log_choose(
        total - column_one, row_one - upper_left
    )
    probabilities = np.exp(log_weights - logsumexp(log_weights))
    probabilities /= probabilities.sum()
    cells = np.column_stack(
        (
            upper_left,
            row_one - upper_left,
            column_one - upper_left,
            total - row_one - column_one + upper_left,
        )
    )
    return counts, cells, probabilities


def _weighted_null_summary(
    coordinates: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    mean = float(probabilities @ coordinates)
    variance = float(probabilities @ np.square(coordinates - mean))
    if not np.isfinite(mean) or not np.isfinite(variance):
        raise FloatingPointError("fixed-margin moments are not finite")
    return mean, max(variance, 0.0)


def centered_haldane_log_odds(table: np.ndarray) -> DonorBinaryCoupling:
    """Return the Haldane log odds ratio centered under exact fixed margins.

    The null distribution enumerates every feasible upper-left count under the
    corresponding hypergeometric law. ``sampling_variance`` is the standard
    Haldane delta-method variance, which remains finite for zero cells.
    """

    counts, cells, probabilities = _fixed_margin_support(table)
    corrected_support = cells + 0.5
    support_log_odds = (
        np.log(corrected_support[:, 0])
        + np.log(corrected_support[:, 3])
        - np.log(corrected_support[:, 1])
        - np.log(corrected_support[:, 2])
    )
    null_mean, null_variance = _weighted_null_summary(support_log_odds, probabilities)
    corrected_observed = counts.astype(float) + 0.5
    observed = float(
        np.log(corrected_observed[0, 0])
        + np.log(corrected_observed[1, 1])
        - np.log(corrected_observed[0, 1])
        - np.log(corrected_observed[1, 0])
    )
    sampling_variance = float(np.reciprocal(corrected_observed).sum())
    return DonorBinaryCoupling(
        observed_log_odds=observed,
        null_mean_log_odds=null_mean,
        centered_log_odds=observed - null_mean,
        sampling_variance=sampling_variance,
        null_variance=null_variance,
        support_lower=int(cells[0, 0]),
        support_upper=int(cells[-1, 0]),
        supported=bool(cells[-1, 0] > cells[0, 0]),
    )


def log_odds_to_helmert_coordinate(log_odds: np.ndarray) -> np.ndarray:
    """Convert a full binary log odds ratio to its Helmert field coordinate."""

    values = np.asarray(log_odds, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("log_odds must be finite")
    return 0.5 * values


def helmert_coordinate_to_log_odds(coordinate: np.ndarray) -> np.ndarray:
    """Convert a binary Helmert field coordinate to the full log odds ratio."""

    values = np.asarray(coordinate, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("coordinate must be finite")
    return 2.0 * values


def binary_table_from_helmert_coordinate(
    coordinate: float,
    row_margin: np.ndarray,
    column_margin: np.ndarray,
) -> np.ndarray:
    """Reconstruct the positive binary table at fixed target margins."""

    value = np.asarray(coordinate, dtype=float)
    if value.ndim != 0 or not np.isfinite(value):
        raise ValueError("coordinate must be one finite scalar")
    return field_coordinates_to_table(value.reshape(1, 1), row_margin, column_margin)


def signed_pearson_coordinate(table: np.ndarray) -> float:
    """Return signed square-root Pearson chi-square for a binary table."""

    counts = _integer_binary_table(table).astype(float)
    total = float(counts.sum())
    rows = counts.sum(axis=1)
    columns = counts.sum(axis=0)
    denominator = float(np.prod(rows) * np.prod(columns))
    if denominator == 0.0:
        return 0.0
    determinant = float(counts[0, 0] * counts[1, 1] - counts[0, 1] * counts[1, 0])
    return determinant * np.sqrt(total / denominator)


def signed_deviance_coordinate(table: np.ndarray) -> float:
    """Return signed square-root Poisson deviance from independence."""

    counts = _integer_binary_table(table).astype(float)
    expected = np.outer(counts.sum(axis=1), counts.sum(axis=0)) / counts.sum()
    positive = counts > 0.0
    deviance = 2.0 * float(
        np.sum(counts[positive] * np.log(counts[positive] / expected[positive]))
    )
    determinant = float(counts[0, 0] * counts[1, 1] - counts[0, 1] * counts[1, 0])
    return float(np.sign(determinant) * np.sqrt(max(deviance, 0.0)))


def centered_classical_coordinate(
    table: np.ndarray,
    *,
    statistic: Literal["pearson", "deviance"],
) -> FixedMarginCoordinate:
    """Exactly center a signed classical independence statistic."""

    counts, cells, probabilities = _fixed_margin_support(table)
    coordinate = (
        signed_pearson_coordinate
        if statistic == "pearson"
        else signed_deviance_coordinate
        if statistic == "deviance"
        else None
    )
    if coordinate is None:
        raise ValueError("statistic must be 'pearson' or 'deviance'")
    support_coordinates = np.asarray(
        [coordinate(candidate.reshape(2, 2)) for candidate in cells], dtype=float
    )
    null_mean, null_variance = _weighted_null_summary(
        support_coordinates, probabilities
    )
    observed = coordinate(counts)
    return FixedMarginCoordinate(
        observed_coordinate=observed,
        null_mean_coordinate=null_mean,
        centered_coordinate=observed - null_mean,
        null_variance=null_variance,
        support_lower=int(cells[0, 0]),
        support_upper=int(cells[-1, 0]),
        supported=bool(cells[-1, 0] > cells[0, 0]),
    )


def centered_classical_residuals(
    table: np.ndarray,
    *,
    residual: Literal["pearson", "deviance"],
) -> FixedMarginResiduals:
    """Exactly center full Poisson-independence residuals at fixed margins.

    This matrix is a reconstruction diagnostic. Its constrained cells must not
    be pooled as independent effects; use :func:`pool_binary_coordinate_family`
    for the one-df classical comparison.
    """

    if residual not in {"pearson", "deviance"}:
        raise ValueError("residual must be 'pearson' or 'deviance'")
    counts, cells, probabilities = _fixed_margin_support(table)
    support_residuals = np.asarray(
        [
            poisson_independence_residuals(candidate.reshape(2, 2), residual=residual)
            for candidate in cells
        ],
        dtype=float,
    )
    null_mean = np.tensordot(probabilities, support_residuals, axes=(0, 0))
    null_variance = np.tensordot(
        probabilities,
        np.square(support_residuals - null_mean),
        axes=(0, 0),
    )
    observed = poisson_independence_residuals(counts, residual=residual)
    return FixedMarginResiduals(
        observed_residuals=observed,
        null_mean_residuals=null_mean,
        centered_residuals=observed - null_mean,
        null_variance_residuals=np.maximum(null_variance, 0.0),
        support_lower=int(cells[0, 0]),
        support_upper=int(cells[-1, 0]),
        supported=bool(cells[-1, 0] > cells[0, 0]),
    )


def _paule_mandel_entity(
    estimates: np.ndarray, variances: np.ndarray, variance_floor: float
) -> tuple[float, float, float, float]:
    effective_variance = np.maximum(variances, variance_floor)
    shifted = estimates - estimates[0]

    def moments(tau_squared: float) -> tuple[float, float, np.ndarray]:
        weights = np.reciprocal(effective_variance + tau_squared)
        mean_shifted = float(weights @ shifted / weights.sum())
        q_statistic = float(weights @ np.square(shifted - mean_shifted))
        return q_statistic, mean_shifted, weights

    degrees_of_freedom = len(estimates) - 1
    q_zero, _, _ = moments(0.0)
    if q_zero <= degrees_of_freedom:
        tau_squared = 0.0
    else:
        spread = float(np.max(np.abs(shifted)))
        upper = max(variance_floor, spread * spread)
        for _ in range(128):
            if moments(upper)[0] <= degrees_of_freedom:
                break
            upper *= 2.0
            if not np.isfinite(upper):
                raise FloatingPointError("could not bracket Paule-Mandel heterogeneity")
        else:
            raise FloatingPointError("could not bracket Paule-Mandel heterogeneity")
        tau_squared = float(
            brentq(
                lambda value: moments(value)[0] - degrees_of_freedom,
                0.0,
                upper,
                xtol=np.finfo(float).tiny,
                rtol=8.0 * np.finfo(float).eps,
                maxiter=256,
            )
        )
    q_statistic, mean_shifted, weights = moments(tau_squared)
    precision = float(weights.sum())
    return (
        float(estimates[0] + mean_shifted),
        1.0 / precision,
        tau_squared,
        q_statistic,
    )


def paule_mandel_pool(
    estimates: np.ndarray,
    variances: np.ndarray,
    *,
    support: np.ndarray | None = None,
    variance_floor: float = 1e-8,
    minimum_donors: int = 2,
) -> PauleMandelPool:
    """Pool donor estimates independently for every trailing entity index.

    Inputs have donor on the first axis. Unsupported donors receive no weight;
    the variance floor applies only to supported observations.
    """

    values = np.asarray(estimates, dtype=float)
    within_variance = np.asarray(variances, dtype=float)
    floor = float(variance_floor)
    minimum = int(minimum_donors)
    if values.ndim < 1 or values.shape[0] < 2 or values.size < 2:
        raise ValueError("estimates must contain at least two donors")
    if within_variance.shape != values.shape:
        raise ValueError("variances must have the same shape as estimates")
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("variance_floor must be finite and positive")
    if minimum < 2 or minimum > values.shape[0]:
        raise ValueError("minimum_donors must be between two and donor count")
    if support is None:
        observed = np.ones(values.shape, dtype=bool)
    else:
        observed = np.asarray(support)
        if observed.shape != values.shape or observed.dtype != np.bool_:
            raise ValueError("support must be a boolean array matching estimates")
    if not np.isfinite(values[observed]).all():
        raise ValueError("supported estimates must be finite")
    if not np.isfinite(within_variance[observed]).all() or np.any(
        within_variance[observed] < 0.0
    ):
        raise ValueError("supported variances must be finite and nonnegative")

    entity_shape = values.shape[1:]
    flat_values = values.reshape(values.shape[0], -1)
    flat_variance = within_variance.reshape(values.shape[0], -1)
    flat_support = observed.reshape(values.shape[0], -1)
    pooled = np.zeros(flat_values.shape[1], dtype=float)
    pooled_variance = np.full_like(pooled, np.inf)
    tau_squared = np.zeros_like(pooled)
    q_statistic = np.zeros_like(pooled)
    support_count = flat_support.sum(axis=0)
    entity_supported = support_count >= minimum
    for entity in range(flat_values.shape[1]):
        if not entity_supported[entity]:
            continue
        selected = flat_support[:, entity]
        (
            pooled[entity],
            pooled_variance[entity],
            tau_squared[entity],
            q_statistic[entity],
        ) = _paule_mandel_entity(
            flat_values[selected, entity], flat_variance[selected, entity], floor
        )
    precision = np.zeros_like(pooled_variance)
    precision[entity_supported] = np.reciprocal(pooled_variance[entity_supported])
    return PauleMandelPool(
        mean=pooled.reshape(entity_shape),
        variance=pooled_variance.reshape(entity_shape),
        precision=precision.reshape(entity_shape),
        tau_squared=tau_squared.reshape(entity_shape),
        q_statistic=q_statistic.reshape(entity_shape),
        support_count=support_count.reshape(entity_shape),
        supported=entity_supported.reshape(entity_shape),
        donors=values.shape[0],
        minimum_donors=minimum,
        variance_floor=floor,
    )


def pool_binary_coordinate_family(
    tables: np.ndarray,
    *,
    family: Literal["haldane_log_odds", "pearson", "deviance"],
    variance_floor: float = 1e-8,
    minimum_donors: int = 2,
) -> BinaryCoordinatePool:
    """Pool one-df binary coordinates under one conditional variance rule.

    Every family uses the exact fixed-margin null variance of its own scalar
    coordinate. Degenerate-margin donors are excluded before pooling.
    """

    values = np.asarray(tables)
    if values.ndim < 3 or values.shape[-2:] != (2, 2):
        raise ValueError("tables must have donor first and end in shape (2, 2)")
    if family not in {"haldane_log_odds", "pearson", "deviance"}:
        raise ValueError("unknown binary coordinate family")
    donor_shape = values.shape[:-2]
    coordinates = np.full(donor_shape, np.nan, dtype=float)
    variances = np.full(donor_shape, np.nan, dtype=float)
    support = np.zeros(donor_shape, dtype=bool)
    for index in np.ndindex(donor_shape):
        if family == "haldane_log_odds":
            estimate = centered_haldane_log_odds(values[index])
            coordinates[index] = estimate.centered_log_odds
        else:
            estimate = centered_classical_coordinate(values[index], statistic=family)
            coordinates[index] = estimate.centered_coordinate
        variances[index] = estimate.null_variance
        support[index] = estimate.supported
    pooled = paule_mandel_pool(
        coordinates,
        variances,
        support=support,
        variance_floor=variance_floor,
        minimum_donors=minimum_donors,
    )
    return BinaryCoordinatePool(
        family=family,
        variance_convention="exact_fixed_margin_null_variance",
        donor_coordinate=coordinates,
        donor_variance=variances,
        donor_support=support,
        pooled=pooled,
    )


def product_hypergraph_laplacian(
    first_incidence: np.ndarray,
    second_incidence: np.ndarray,
    *,
    first_hyperedge_weight: np.ndarray | None = None,
    second_hyperedge_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Return the Kronecker-sum Laplacian for ordered entity pairs.

    The ordering matches C-order flattening of a
    ``first_entity x second_entity`` matrix.
    """

    first = normalized_hypergraph_laplacian(
        first_incidence, hyperedge_weight=first_hyperedge_weight
    )
    second = normalized_hypergraph_laplacian(
        second_incidence, hyperedge_weight=second_hyperedge_weight
    )
    product = np.kron(first, np.eye(second.shape[0])) + np.kron(
        np.eye(first.shape[0]), second
    )
    return 0.5 * (product + product.T)


def _validated_laplacian(laplacian: np.ndarray, size: int) -> np.ndarray:
    matrix = np.asarray(laplacian, dtype=float)
    if matrix.shape != (size, size) or not np.isfinite(matrix).all():
        raise ValueError("graph_laplacian must be a finite square matrix")
    scale = max(1.0, float(np.max(np.abs(matrix))))
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-10 * scale):
        raise ValueError("graph_laplacian must be symmetric")
    symmetric = 0.5 * (matrix + matrix.T)
    if float(np.linalg.eigvalsh(symmetric)[0]) < -1e-9 * scale:
        raise ValueError("graph_laplacian must be positive semidefinite")
    return symmetric


def _conditional_records(
    tables: np.ndarray,
) -> tuple[
    tuple[int, ...],
    list[list[tuple[float, np.ndarray, np.ndarray]]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    values = np.asarray(tables)
    if values.ndim < 3 or values.shape[-2:] != (2, 2):
        raise ValueError("tables must have donor first and end in shape (2, 2)")
    if values.shape[0] < 1:
        raise ValueError("tables must contain at least one donor")
    entity_shape = values.shape[1:-2]
    entity_count = int(np.prod(entity_shape)) if entity_shape else 1
    flat_tables = values.reshape(values.shape[0], entity_count, 2, 2)
    records: list[list[tuple[float, np.ndarray, np.ndarray]]] = [
        [] for _ in range(entity_count)
    ]
    support_count = np.zeros(entity_count, dtype=int)
    observed_sum = np.zeros(entity_count, dtype=float)
    lower_sum = np.zeros(entity_count, dtype=float)
    upper_sum = np.zeros(entity_count, dtype=float)
    for donor in range(flat_tables.shape[0]):
        for entity in range(entity_count):
            counts, cells, _ = _fixed_margin_support(flat_tables[donor, entity])
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
            observed = float(counts[0, 0])
            records[entity].append((observed, support, log_probability))
            support_count[entity] += 1
            observed_sum[entity] += observed
            lower_sum[entity] += float(support[0])
            upper_sum[entity] += float(support[-1])
    return (
        entity_shape,
        records,
        support_count,
        observed_sum,
        lower_sum,
        upper_sum,
    )


def _evaluate_conditional_records(
    log_odds: np.ndarray,
    records: list[list[tuple[float, np.ndarray, np.ndarray]]],
    laplacian: np.ndarray,
    ridge_penalty: float,
    graph_penalty: float,
    support_count: np.ndarray,
) -> ConditionalLogOddsEvaluation:
    theta = np.asarray(log_odds, dtype=float)
    objective = 0.0
    gradient = np.zeros_like(theta)
    data_precision = np.zeros_like(theta)
    for entity, entity_records in enumerate(records):
        for observed, support, log_probability in entity_records:
            log_mass = log_probability + support * theta[entity]
            log_partition = float(logsumexp(log_mass))
            probability = np.exp(log_mass - log_partition)
            expected = float(np.sum(probability * support))
            variance = float(np.sum(probability * np.square(support - expected)))
            objective += log_partition - observed * theta[entity]
            gradient[entity] += expected - observed
            data_precision[entity] += variance
    graph_action = np.einsum("ij,j->i", laplacian, theta, optimize=False)
    objective += 0.5 * ridge_penalty * float(np.sum(np.square(theta)))
    objective += (
        0.5
        * graph_penalty
        * float(np.einsum("i,i->", theta, graph_action, optimize=False))
    )
    gradient += ridge_penalty * theta + graph_penalty * graph_action
    hessian = graph_penalty * laplacian
    diagonal = np.arange(len(theta))
    hessian[diagonal, diagonal] += data_precision + ridge_penalty
    return ConditionalLogOddsEvaluation(
        objective=float(objective),
        gradient=gradient,
        hessian=hessian,
        data_precision=data_precision,
        support_count=support_count.copy(),
    )


def evaluate_conditional_log_odds(
    log_odds: np.ndarray,
    tables: np.ndarray,
    *,
    graph_laplacian: np.ndarray | None = None,
    ridge_penalty: float = 0.0,
    graph_penalty: float = 0.0,
    minimum_informative_donors: int = 1,
) -> ConditionalLogOddsEvaluation:
    """Evaluate the exact convex fixed-margin log-odds likelihood."""

    entity_shape, records, support_count, _, _, _ = _conditional_records(tables)
    theta = np.asarray(log_odds, dtype=float)
    expected_shape = entity_shape if entity_shape else ()
    if theta.shape != expected_shape or not np.isfinite(theta).all():
        raise ValueError("log_odds shape must match the table entity axes")
    minimum = int(minimum_informative_donors)
    if minimum < 1 or np.any(support_count < minimum):
        raise CouplingEstimationRefusal(
            "too few informative donors for at least one entity"
        )
    ridge = float(ridge_penalty)
    graph = float(graph_penalty)
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge_penalty must be finite and nonnegative")
    if not np.isfinite(graph) or graph < 0.0:
        raise ValueError("graph_penalty must be finite and nonnegative")
    size = theta.size
    if graph_laplacian is None:
        if graph > 0.0:
            raise ValueError("a positive graph_penalty requires graph_laplacian")
        laplacian = np.zeros((size, size), dtype=float)
    else:
        laplacian = _validated_laplacian(graph_laplacian, size)
    result = _evaluate_conditional_records(
        theta.ravel(order="C"),
        records,
        laplacian,
        ridge,
        graph,
        support_count,
    )
    return ConditionalLogOddsEvaluation(
        objective=result.objective,
        gradient=result.gradient.reshape(expected_shape),
        hessian=result.hessian,
        data_precision=result.data_precision.reshape(expected_shape),
        support_count=result.support_count.reshape(expected_shape),
    )


def fit_structured_conditional_log_odds(
    tables: np.ndarray,
    first_incidence: np.ndarray,
    second_incidence: np.ndarray,
    *,
    initial_log_odds: np.ndarray | None = None,
    ridge_penalty: float = 1e-3,
    graph_penalty: float = 0.0,
    minimum_informative_donors: int = 2,
    maximum_condition_number: float = 1e12,
    maximum_iterations: int = 200,
    tolerance: float = 1e-9,
) -> StructuredConditionalLogOddsFit:
    """Fit full log-odds parameters by exact conditional likelihood.

    Donor margins enter only through finite noncentral-hypergeometric support.
    The returned ``helmert_coordinate`` is exactly half ``log_odds``.
    Penalties are multiplied by the median null Fisher information and that
    scale is recorded, so duplicating every donor does not retune the grid.
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
        support_count,
        observed_sum,
        lower_sum,
        upper_sum,
    ) = _conditional_records(values)
    minimum = int(minimum_informative_donors)
    if minimum < 1 or np.any(support_count < minimum):
        raise CouplingEstimationRefusal(
            "too few informative donors for at least one entity"
        )
    ridge = float(ridge_penalty)
    graph = float(graph_penalty)
    condition_limit = float(maximum_condition_number)
    iterations = int(maximum_iterations)
    threshold = float(tolerance)
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge_penalty must be finite and nonnegative")
    if not np.isfinite(graph) or graph < 0.0:
        raise ValueError("graph_penalty must be finite and nonnegative")
    if not np.isfinite(condition_limit) or condition_limit <= 1.0:
        raise ValueError("maximum_condition_number must be finite and above one")
    if iterations < 1 or not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("maximum_iterations and tolerance must be positive")
    if ridge == 0.0 and (
        np.any(observed_sum <= lower_sum) or np.any(observed_sum >= upper_sum)
    ):
        raise CouplingEstimationRefusal(
            "unregularized conditional likelihood has a boundary optimum"
        )
    laplacian = product_hypergraph_laplacian(first, second)
    null_evaluation = _evaluate_conditional_records(
        np.zeros(int(np.prod(entity_shape)), dtype=float),
        records,
        laplacian,
        0.0,
        0.0,
        support_count,
    )
    positive_null_precision = null_evaluation.data_precision[
        null_evaluation.data_precision > 0.0
    ]
    penalty_scale = float(np.median(positive_null_precision))
    effective_ridge = ridge * penalty_scale
    effective_graph = graph * penalty_scale
    if initial_log_odds is None:
        initial = pool_binary_coordinate_family(
            values,
            family="haldane_log_odds",
            minimum_donors=minimum,
        ).pooled.mean.ravel(order="C")
    else:
        supplied = np.asarray(initial_log_odds, dtype=float)
        if supplied.shape != entity_shape or not np.isfinite(supplied).all():
            raise ValueError("initial_log_odds shape must match entity axes")
        initial = supplied.ravel(order="C")

    def evaluation(theta: np.ndarray) -> ConditionalLogOddsEvaluation:
        return _evaluate_conditional_records(
            theta,
            records,
            laplacian,
            effective_ridge,
            effective_graph,
            support_count,
        )

    result = minimize(
        lambda theta: evaluation(theta).objective,
        initial,
        method="Newton-CG",
        jac=lambda theta: evaluation(theta).gradient,
        hess=lambda theta: evaluation(theta).hessian,
        options={"xtol": threshold, "maxiter": iterations},
    )
    final = evaluation(result.x)
    gradient_norm = float(np.max(np.abs(final.gradient)))
    converged = bool(result.success or gradient_norm <= threshold)
    if not converged:
        raise CouplingEstimationRefusal(
            "conditional-likelihood optimizer did not converge"
        )
    condition_number = float(np.linalg.cond(final.hessian))
    if not np.isfinite(condition_number) or condition_number > condition_limit:
        raise CouplingEstimationRefusal(
            "conditional-likelihood Hessian exceeds the condition-number limit"
        )
    fitted = result.x.reshape(entity_shape)
    return StructuredConditionalLogOddsFit(
        log_odds=fitted,
        helmert_coordinate=log_odds_to_helmert_coordinate(fitted),
        objective=final.objective,
        gradient_norm=gradient_norm,
        data_precision=final.data_precision.reshape(entity_shape),
        support_count=support_count.reshape(entity_shape),
        condition_number=condition_number,
        iterations=int(result.nit),
        converged=True,
        ridge_penalty=ridge,
        graph_penalty=graph,
        penalty_scale=penalty_scale,
    )


def _integer_binary_margins(
    row_margin: np.ndarray, column_margin: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(row_margin, dtype=float)
    columns = np.asarray(column_margin, dtype=float)
    if rows.shape != (2,) or columns.shape != (2,):
        raise ValueError("binary margins must each have length two")
    if (
        not np.isfinite(rows).all()
        or not np.isfinite(columns).all()
        or np.any(rows < 0.0)
        or np.any(columns < 0.0)
        or not np.array_equal(rows, np.rint(rows))
        or not np.array_equal(columns, np.rint(columns))
        or np.any(rows > np.iinfo(np.int64).max)
        or np.any(columns > np.iinfo(np.int64).max)
    ):
        raise ValueError("conditional prediction requires nonnegative integer margins")
    if rows.sum() <= 0.0 or rows.sum() != columns.sum():
        raise ValueError("row and column margins must have the same positive total")
    if rows.sum() > 2**53 - 1:
        raise ValueError("margin total exceeds exact floating-point integer range")
    return rows.astype(np.int64), columns.astype(np.int64)


def expected_binary_table_from_log_odds(
    log_odds: float,
    row_margin: np.ndarray,
    column_margin: np.ndarray,
) -> np.ndarray:
    """Return the exact conditional expected table at target margins."""

    theta = float(log_odds)
    if not np.isfinite(theta):
        raise ValueError("log_odds must be finite")
    rows, columns = _integer_binary_margins(row_margin, column_margin)
    total = int(rows.sum())
    lower = max(0, int(rows[0] + columns[0] - total))
    seed = np.array(
        [
            [lower, int(rows[0] - lower)],
            [int(columns[0] - lower), int(rows[1] - columns[0] + lower)],
        ]
    )
    _, cells, _ = _fixed_margin_support(seed)
    support = cells[:, 0]
    log_weights = _log_choose(int(columns[0]), support) + _log_choose(
        int(columns[1]), rows[0] - support
    )
    log_mass = log_weights + theta * support
    probability = np.exp(log_mass - logsumexp(log_mass))
    expected_upper_left = float(np.sum(probability * support))
    return np.array(
        [
            [expected_upper_left, rows[0] - expected_upper_left],
            [
                columns[0] - expected_upper_left,
                rows[1] - columns[0] + expected_upper_left,
            ],
        ],
        dtype=float,
    )


def fit_precision_weighted_coupling(
    mean: np.ndarray,
    precision: np.ndarray,
    *,
    graph_laplacian: np.ndarray | None = None,
    ridge_penalty: float = 0.0,
    graph_penalty: float = 0.0,
    precision_floor: float = 1e-8,
    maximum_condition_number: float = 1e12,
) -> PrecisionWeightedCouplingFit:
    """Solve ``(P + lambda0 I + lambdaG L) theta = P m`` exactly.

    Positive precisions are divided by their median before applying penalties,
    making a common penalty grid invariant to family-wide precision scaling.
    Exact zeros remain unsupported observations and never enter ``P``.
    """

    values = np.asarray(mean, dtype=float)
    raw_precision = np.asarray(precision, dtype=float)
    ridge = float(ridge_penalty)
    graph = float(graph_penalty)
    floor = float(precision_floor)
    condition_limit = float(maximum_condition_number)
    if values.size < 1 or not np.isfinite(values).all():
        raise ValueError("mean must be a nonempty finite array")
    if raw_precision.shape != values.shape:
        raise ValueError("precision must have the same shape as mean")
    if not np.isfinite(raw_precision).all() or np.any(raw_precision < 0.0):
        raise ValueError("precision must be finite and nonnegative")
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge_penalty must be finite and nonnegative")
    if not np.isfinite(graph) or graph < 0.0:
        raise ValueError("graph_penalty must be finite and nonnegative")
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("precision_floor must be finite and positive")
    if not np.isfinite(condition_limit) or condition_limit <= 1.0:
        raise ValueError("maximum_condition_number must be finite and above one")

    flat_mean = values.ravel(order="C")
    flat_raw_precision = raw_precision.ravel(order="C")
    support = flat_raw_precision > 0.0
    if not np.any(support):
        raise CouplingEstimationRefusal("no entity has positive data precision")
    precision_scale = float(np.median(flat_raw_precision[support]))
    effective_precision = np.zeros_like(flat_raw_precision)
    effective_precision[support] = np.maximum(
        flat_raw_precision[support] / precision_scale, floor
    )
    if graph_laplacian is None:
        if graph > 0.0:
            raise ValueError("a positive graph_penalty requires graph_laplacian")
        laplacian = np.zeros((flat_mean.size, flat_mean.size), dtype=float)
    else:
        laplacian = _validated_laplacian(graph_laplacian, flat_mean.size)

    system = graph * laplacian
    diagonal = np.arange(flat_mean.size)
    system[diagonal, diagonal] += effective_precision + ridge
    scale = float(np.max(np.abs(system)))
    scaled_system = system / scale
    condition_number = float(np.linalg.cond(scaled_system))
    if not np.isfinite(condition_number) or condition_number > condition_limit:
        raise CouplingEstimationRefusal(
            "precision-weighted system exceeds the condition-number limit"
        )
    scaled_right_hand_side = (effective_precision / scale) * flat_mean
    estimate = np.linalg.solve(scaled_system, scaled_right_hand_side)
    residual = estimate - flat_mean
    objective = 0.5 * float(np.sum(effective_precision * np.square(residual)))
    objective += 0.5 * ridge * float(np.sum(np.square(estimate)))
    graph_action = np.einsum("ij,j->i", laplacian, estimate, optimize=False)
    objective += (
        0.5 * graph * float(np.einsum("i,i->", estimate, graph_action, optimize=False))
    )
    return PrecisionWeightedCouplingFit(
        estimate=estimate.reshape(values.shape),
        mean=values.copy(),
        raw_precision=raw_precision.copy(),
        precision=effective_precision.reshape(values.shape),
        support=support.reshape(values.shape),
        precision_scale=precision_scale,
        objective=objective,
        condition_number=condition_number,
        ridge_penalty=ridge,
        graph_penalty=graph,
        precision_floor=floor,
        maximum_condition_number=condition_limit,
    )


def fit_heterogeneity_adaptive_binary_coupling(
    tables: np.ndarray,
    first_incidence: np.ndarray,
    second_incidence: np.ndarray,
    *,
    variance_floor: float = 1e-8,
    precision_floor: float = 1e-8,
    ridge_penalty: float = 0.0,
    graph_penalty: float = 0.0,
    minimum_donors: int = 2,
    maximum_condition_number: float = 1e12,
) -> HeterogeneityAdaptiveCouplingFit:
    """Fit the null-centered Haldane/PM ablation.

    ``tables`` has shape
    ``donor x first_entity x second_entity x 2 x 2``.
    Use :func:`fit_structured_conditional_log_odds` for the primary parameter
    estimator.
    """

    values = np.asarray(tables)
    first = np.asarray(first_incidence)
    second = np.asarray(second_incidence)
    if values.ndim != 5 or values.shape[-2:] != (2, 2):
        raise ValueError(
            "tables must have shape donor x first_entity x second_entity x 2 x 2"
        )
    if values.shape[0] < 2:
        raise ValueError("at least two donors are required")
    if first.ndim != 2 or first.shape[0] != values.shape[1]:
        raise ValueError("first_incidence rows must match first entities")
    if second.ndim != 2 or second.shape[0] != values.shape[2]:
        raise ValueError("second_incidence rows must match second entities")

    donor_effect = np.empty(values.shape[:3], dtype=float)
    donor_sampling_variance = np.empty_like(donor_effect)
    donor_pooling_variance = np.empty_like(donor_effect)
    donor_null_mean = np.empty_like(donor_effect)
    donor_support = np.empty(values.shape[:3], dtype=bool)
    for index in np.ndindex(values.shape[:3]):
        estimate = centered_haldane_log_odds(values[index])
        donor_effect[index] = estimate.centered_log_odds
        donor_sampling_variance[index] = estimate.sampling_variance
        donor_pooling_variance[index] = estimate.null_variance
        donor_null_mean[index] = estimate.null_mean_log_odds
        donor_support[index] = estimate.supported

    pooled = paule_mandel_pool(
        donor_effect,
        donor_pooling_variance,
        support=donor_support,
        variance_floor=variance_floor,
        minimum_donors=minimum_donors,
    )
    laplacian = product_hypergraph_laplacian(first, second)
    structured = fit_precision_weighted_coupling(
        pooled.mean,
        pooled.precision,
        graph_laplacian=laplacian,
        ridge_penalty=ridge_penalty,
        graph_penalty=graph_penalty,
        precision_floor=precision_floor,
        maximum_condition_number=maximum_condition_number,
    )
    return HeterogeneityAdaptiveCouplingFit(
        donor_centered_log_odds=donor_effect,
        donor_sampling_variance=donor_sampling_variance,
        donor_pooling_variance=donor_pooling_variance,
        donor_null_mean_log_odds=donor_null_mean,
        donor_support=donor_support,
        variance_convention="exact_fixed_margin_null_variance",
        pooled=pooled,
        structured=structured,
    )
