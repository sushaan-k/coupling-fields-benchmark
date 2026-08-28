"""Marginal-invariant association fields for paired finite-state assays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class CouplingFieldRefusal(ValueError):
    """Raised when a coupling field is unsupported or fails to converge."""


@dataclass(frozen=True)
class StructuredCouplingFit:
    """A target-by-field estimate with convex structural shrinkage."""

    coefficient: np.ndarray
    singular_values: np.ndarray
    effective_rank: int
    objective: float
    relative_step: float
    iterations: int
    converged: bool
    nuclear_penalty: float
    graph_penalty: float


@dataclass(frozen=True)
class ConditionalAssociationEstimate:
    """A finite-sample association estimate centered on a pairing-null field."""

    coordinates: np.ndarray
    observed_coordinates: np.ndarray
    null_mean_coordinates: np.ndarray
    null_variance_coordinates: np.ndarray
    destroyed_coordinates: np.ndarray
    permutations: int


def inverse_permutation_variance_weights(variance: np.ndarray) -> np.ndarray:
    """Return clipped, median-normalized inverse permutation variances."""

    values = np.asarray(variance, dtype=float)
    if values.size < 1 or not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("permutation variance must be finite and positive")
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        precision = 1.0 / values
    if not np.isfinite(precision).all():
        raise ValueError("inverse permutation variance must be finite")
    median_precision = float(np.median(precision))
    if not np.isfinite(median_precision) or median_precision <= 0.0:
        raise ValueError("median inverse permutation variance must be positive")
    return np.clip(precision / median_precision, 0.05, 20.0)


def helmert_contrast(levels: int) -> np.ndarray:
    """Return an orthonormal contrast basis perpendicular to the all-ones vector."""

    count = int(levels)
    if count < 2:
        raise ValueError("a contrast basis needs at least two levels")
    basis = np.zeros((count, count - 1), dtype=float)
    for column in range(count - 1):
        scale = np.sqrt((column + 1) * (column + 2))
        basis[: column + 1, column] = 1.0 / scale
        basis[column + 1, column] = -(column + 1) / scale
    return basis


def association_field(table: np.ndarray, *, pseudocount: float = 0.0) -> np.ndarray:
    """Return the row- and column-marginal-free log association of a table."""

    values = np.asarray(table, dtype=float)
    alpha = float(pseudocount)
    if values.ndim < 2 or min(values.shape[-2:]) < 2:
        raise ValueError("table must end in two state axes of size at least two")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("table must be finite and nonnegative")
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("pseudocount must be finite and nonnegative")
    regularized = values + alpha
    if np.any(regularized <= 0.0):
        raise CouplingFieldRefusal(
            "every joint state needs positive support or a positive pseudocount"
        )
    logged = np.log(regularized)
    return (
        logged
        - logged.mean(axis=-2, keepdims=True)
        - logged.mean(axis=-1, keepdims=True)
        + logged.mean(axis=(-2, -1), keepdims=True)
    )


def association_coordinates(field: np.ndarray) -> np.ndarray:
    """Express association fields in an orthonormal cycle-space basis."""

    values = np.asarray(field, dtype=float)
    if values.ndim < 2 or min(values.shape[-2:]) < 2:
        raise ValueError("field must end in two state axes of size at least two")
    if not np.isfinite(values).all():
        raise ValueError("field must be finite")
    row_basis = helmert_contrast(values.shape[-2])
    column_basis = helmert_contrast(values.shape[-1])
    return np.einsum(
        "ui,...uv,vj->...ij", row_basis, values, column_basis, optimize=True
    )


def conditional_association_coordinates(
    first_state: np.ndarray,
    second_state: np.ndarray,
    *,
    first_levels: int,
    second_levels: int,
    pseudocount: float = 0.5,
    permutations: int = 64,
    seed: int = 0,
) -> ConditionalAssociationEstimate:
    """Estimate pairing association after conditional permutation centering.

    The reference permutations preserve both empirical marginal state counts.
    Their mean removes finite-table and pseudocount bias that can otherwise make
    an association field predictable after the links between assays are broken.
    One additional permutation is held out as the destroyed-link control.
    """

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

    def coordinates(permuted_second: np.ndarray) -> np.ndarray:
        table = np.bincount(
            first * column_count + permuted_second,
            minlength=row_count * column_count,
        ).reshape(row_count, column_count)
        return association_coordinates(
            association_field(table, pseudocount=pseudocount)
        )

    generator = np.random.default_rng(seed)
    null_fields = []
    for _ in range(draw_count + 1):
        permutation = generator.permutation(len(first))
        null_fields.append(coordinates(second[permutation]))
    null_values = np.asarray(null_fields)
    reference = null_values[1:].mean(axis=0)
    observed = coordinates(second)
    centered = observed - reference
    destroyed = null_values[0] - reference
    null_variance = np.var(null_values[1:], axis=0, ddof=1) * (
        1.0 + 1.0 / draw_count
    )
    return ConditionalAssociationEstimate(
        coordinates=centered,
        observed_coordinates=observed,
        null_mean_coordinates=reference,
        null_variance_coordinates=null_variance,
        destroyed_coordinates=destroyed,
        permutations=draw_count,
    )


def field_from_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """Lift cycle-space coordinates back to a zero-row/column-sum field."""

    values = np.asarray(coordinates, dtype=float)
    if values.ndim < 2 or min(values.shape[-2:]) < 1:
        raise ValueError("coordinates must end in two nonempty contrast axes")
    if not np.isfinite(values).all():
        raise ValueError("coordinates must be finite")
    row_basis = helmert_contrast(values.shape[-2] + 1)
    column_basis = helmert_contrast(values.shape[-1] + 1)
    return np.einsum(
        "ui,...ij,vj->...uv", row_basis, values, column_basis, optimize=True
    )


def factorial_association_contrast(
    tables: np.ndarray,
    perturbation: int,
    control: int,
    challenge: int,
    vehicle: int,
    *,
    pseudocount: float = 0.0,
) -> np.ndarray:
    """Return a perturbation-by-context contrast with both margins removed."""

    values = np.asarray(tables, dtype=float)
    if values.ndim != 4:
        raise ValueError("tables must have shape perturbation x context x U x V")
    indices = (perturbation, control, challenge, vehicle)
    limits = (values.shape[0], values.shape[0], values.shape[1], values.shape[1])
    if any(index < 0 or index >= limit for index, limit in zip(indices, limits)):
        raise IndexError("factorial contrast index is out of bounds")
    fields = association_field(values, pseudocount=pseudocount)
    return (fields[perturbation, challenge] - fields[perturbation, vehicle]) - (
        fields[control, challenge] - fields[control, vehicle]
    )


def normalized_hypergraph_laplacian(
    incidence: np.ndarray,
    *,
    hyperedge_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Construct the normalized Zhou hypergraph Laplacian.

    Rows are entities and columns are hyperedges. Isolated entities receive a
    zero row and column, so the penalty leaves them unchanged.
    """

    membership = np.asarray(incidence, dtype=float)
    if (
        membership.ndim != 2
        or min(membership.shape) < 1
        or not np.isfinite(membership).all()
        or np.any(membership < 0.0)
    ):
        raise ValueError("incidence must be a finite nonnegative matrix")
    if np.any(membership.sum(axis=0) <= 0.0):
        raise ValueError("every hyperedge must contain at least one entity")
    if hyperedge_weight is None:
        weights = np.ones(membership.shape[1], dtype=float)
    else:
        weights = np.asarray(hyperedge_weight, dtype=float)
        if (
            weights.shape != (membership.shape[1],)
            or not np.isfinite(weights).all()
            or np.any(weights <= 0.0)
        ):
            raise ValueError("hyperedge_weight must be finite and positive")

    edge_degree = membership.sum(axis=0)
    vertex_degree = membership @ weights
    inverse_vertex = np.zeros_like(vertex_degree)
    supported = vertex_degree > 0.0
    inverse_vertex[supported] = 1.0 / np.sqrt(vertex_degree[supported])
    scaled = inverse_vertex[:, None] * membership
    adjacency = (scaled * (weights / edge_degree)[None, :]) @ scaled.T
    laplacian = -adjacency
    diagonal = np.arange(membership.shape[0])
    laplacian[diagonal[supported], diagonal[supported]] += 1.0
    laplacian = 0.5 * (laplacian + laplacian.T)
    return laplacian


def _validate_laplacian(laplacian: np.ndarray, rows: int) -> np.ndarray:
    matrix = np.asarray(laplacian, dtype=float)
    if matrix.shape != (rows, rows) or not np.isfinite(matrix).all():
        raise ValueError("graph_laplacian must be a finite square entity matrix")
    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ValueError("graph_laplacian must be symmetric")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if float(eigenvalues[0]) < -1e-9:
        raise ValueError("graph_laplacian must be positive semidefinite")
    return 0.5 * (matrix + matrix.T)


def _soft_threshold_singular_values(
    matrix: np.ndarray, threshold: float
) -> np.ndarray:
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    retained = np.maximum(singular - threshold, 0.0)
    return (left * retained) @ right


def fit_structured_coupling_fields(
    observed: np.ndarray,
    *,
    observation_weight: np.ndarray | None = None,
    graph_laplacian: np.ndarray | None = None,
    nuclear_penalty: float = 0.0,
    graph_penalty: float = 0.0,
    maximum_iterations: int = 5_000,
    tolerance: float = 1e-8,
) -> StructuredCouplingFit:
    """Denoise target-by-field coordinates with low-rank and graph structure.

    The fitted matrix minimizes

    ``0.5 * ||sqrt(W) * (Y-B)||_F^2 + a||B||_* + b tr(B' L B)``.

    The problem is convex. Positive observation weights make its smooth part
    strongly convex, so the fitted field is unique even when the nuclear norm
    is nondifferentiable.
    """

    values = np.asarray(observed, dtype=float)
    if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError("observed must be a finite entity-by-field matrix")
    if observation_weight is None:
        weights = np.ones_like(values)
    else:
        weights = np.asarray(observation_weight, dtype=float)
        if weights.shape != values.shape:
            try:
                weights = np.broadcast_to(weights, values.shape).copy()
            except ValueError as error:
                raise ValueError("observation_weight must broadcast to observed") from error
        if not np.isfinite(weights).all() or np.any(weights <= 0.0):
            raise ValueError("observation weights must be finite and positive")

    trace_penalty = float(nuclear_penalty)
    smooth_penalty = float(graph_penalty)
    if not np.isfinite(trace_penalty) or trace_penalty < 0.0:
        raise ValueError("nuclear_penalty must be finite and nonnegative")
    if not np.isfinite(smooth_penalty) or smooth_penalty < 0.0:
        raise ValueError("graph_penalty must be finite and nonnegative")
    iterations = int(maximum_iterations)
    threshold = float(tolerance)
    if iterations < 1 or not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("maximum_iterations and tolerance must be positive")

    if graph_laplacian is None:
        laplacian = np.zeros((values.shape[0], values.shape[0]), dtype=float)
    else:
        laplacian = _validate_laplacian(graph_laplacian, values.shape[0])
    if smooth_penalty > 0.0 and graph_laplacian is None:
        raise ValueError("a positive graph_penalty requires graph_laplacian")

    if trace_penalty == 0.0 and smooth_penalty == 0.0:
        singular = np.linalg.svd(values, compute_uv=False)
        rank_tolerance = max(values.shape) * np.finfo(float).eps * max(
            1.0, float(singular[0]) if len(singular) else 1.0
        )
        return StructuredCouplingFit(
            coefficient=values.copy(),
            singular_values=singular,
            effective_rank=int(np.count_nonzero(singular > rank_tolerance)),
            objective=0.0,
            relative_step=0.0,
            iterations=0,
            converged=True,
            nuclear_penalty=trace_penalty,
            graph_penalty=smooth_penalty,
        )

    graph_lipschitz = (
        2.0 * smooth_penalty * float(np.linalg.eigvalsh(laplacian)[-1])
        if smooth_penalty > 0.0
        else 0.0
    )
    lipschitz = float(weights.max()) + graph_lipschitz
    step_size = 1.0 / lipschitz

    coefficient = values.copy()
    extrapolated = coefficient.copy()
    acceleration = 1.0
    converged = False
    relative_step = np.inf
    iteration = 0
    for iteration in range(1, iterations + 1):
        gradient = weights * (extrapolated - values)
        if smooth_penalty > 0.0:
            gradient += 2.0 * smooth_penalty * (laplacian @ extrapolated)
        candidate = _soft_threshold_singular_values(
            extrapolated - step_size * gradient,
            step_size * trace_penalty,
        )
        relative_step = float(
            np.linalg.norm(candidate - coefficient)
            / max(1.0, np.linalg.norm(coefficient))
        )
        previous = coefficient
        coefficient = candidate
        if relative_step <= threshold:
            converged = True
            break
        next_acceleration = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * acceleration**2))
        extrapolated = coefficient + (acceleration - 1.0) / next_acceleration * (
            coefficient - previous
        )
        acceleration = next_acceleration

    if not converged:
        raise CouplingFieldRefusal("structured coupling fit did not converge")
    singular = np.linalg.svd(coefficient, compute_uv=False)
    rank_tolerance = max(coefficient.shape) * np.finfo(float).eps * max(
        1.0, float(singular[0]) if len(singular) else 1.0
    )
    residual = coefficient - values
    objective = 0.5 * float(np.sum(weights * residual**2))
    objective += trace_penalty * float(singular.sum())
    objective += smooth_penalty * float(
        np.sum(coefficient * (laplacian @ coefficient))
    )
    return StructuredCouplingFit(
        coefficient=coefficient,
        singular_values=singular,
        effective_rank=int(np.count_nonzero(singular > rank_tolerance)),
        objective=objective,
        relative_step=relative_step,
        iterations=iteration,
        converged=converged,
        nuclear_penalty=trace_penalty,
        graph_penalty=smooth_penalty,
    )
