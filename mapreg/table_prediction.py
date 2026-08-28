"""Common-table prediction and scoring for finite-state association models."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .coupling_fields import field_from_coordinates


class TablePredictionRefusal(ValueError):
    """Raised when coordinates and margins do not determine a stable table."""


def _validated_margins(
    row_margin: np.ndarray, column_margin: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(row_margin, dtype=float)
    columns = np.asarray(column_margin, dtype=float)
    if rows.ndim != 1 or columns.ndim != 1 or min(len(rows), len(columns)) < 2:
        raise ValueError("margins must be one-dimensional with at least two states")
    if (
        not np.isfinite(rows).all()
        or not np.isfinite(columns).all()
        or np.any(rows <= 0.0)
        or np.any(columns <= 0.0)
    ):
        raise TablePredictionRefusal("every row and column margin must be positive")
    if not np.isclose(rows.sum(), columns.sum(), rtol=1e-10, atol=1e-10):
        raise ValueError("row and column margins must have the same total")
    return rows, columns


def ipf_to_margins(
    seed_table: np.ndarray,
    row_margin: np.ndarray,
    column_margin: np.ndarray,
    *,
    tolerance: float = 1e-8,
    maximum_iterations: int = 10_000,
) -> np.ndarray:
    """Scale a strictly positive table to fixed margins by iterative fitting."""

    rows, columns = _validated_margins(row_margin, column_margin)
    table = np.asarray(seed_table, dtype=float)
    if table.shape != (len(rows), len(columns)):
        raise ValueError("seed_table shape must match the supplied margins")
    if not np.isfinite(table).all() or np.any(table <= 0.0):
        raise TablePredictionRefusal("seed_table must be finite and strictly positive")
    threshold = float(tolerance)
    iterations = int(maximum_iterations)
    if not np.isfinite(threshold) or threshold <= 0.0 or iterations < 1:
        raise ValueError("tolerance and maximum_iterations must be positive")

    fitted = table.copy()
    for _ in range(iterations):
        fitted *= np.divide(rows, fitted.sum(axis=1))[:, None]
        fitted *= np.divide(columns, fitted.sum(axis=0))[None, :]
        row_error = float(np.max(np.abs(fitted.sum(axis=1) - rows)))
        column_error = float(np.max(np.abs(fitted.sum(axis=0) - columns)))
        if max(row_error, column_error) <= threshold * max(1.0, rows.sum()):
            return fitted
    raise TablePredictionRefusal("iterative proportional fitting did not converge")


def field_coordinates_to_table(
    coordinates: np.ndarray,
    row_margin: np.ndarray,
    column_margin: np.ndarray,
) -> np.ndarray:
    """Reconstruct a positive same-margin table from log-association coordinates."""

    rows, columns = _validated_margins(row_margin, column_margin)
    values = np.asarray(coordinates, dtype=float)
    if values.shape != (len(rows) - 1, len(columns) - 1):
        raise ValueError("coordinate shape is incompatible with the supplied margins")
    field = field_from_coordinates(values)
    field -= float(np.max(field))
    seed = np.exp(field)
    return ipf_to_margins(seed, rows, columns)


def _deviance_count(expected: float, residual: float) -> float:
    if residual == 0.0:
        return expected
    target = 0.5 * residual**2

    def objective(value: float) -> float:
        if value == 0.0:
            return expected
        return value * np.log(value / expected) - (value - expected)

    if residual < 0.0:
        lower, upper = 0.0, expected
        if target >= expected:
            return np.finfo(float).tiny
    else:
        lower, upper = expected, max(expected + 1.0, 2.0 * expected)
        while objective(upper) < target:
            upper *= 2.0
            if not np.isfinite(upper):
                raise TablePredictionRefusal("deviance residual inversion overflowed")
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        value = objective(midpoint)
        if (residual < 0.0 and value > target) or (
            residual > 0.0 and value < target
        ):
            lower = midpoint
        else:
            upper = midpoint
    return max(0.5 * (lower + upper), np.finfo(float).tiny)


def residual_coordinates_to_table(
    residual_values: np.ndarray,
    row_margin: np.ndarray,
    column_margin: np.ndarray,
    *,
    residual: Literal["pearson", "deviance"] = "pearson",
) -> np.ndarray:
    """Reconstruct a same-margin table from a full independence-residual matrix."""

    rows, columns = _validated_margins(row_margin, column_margin)
    values = np.asarray(residual_values, dtype=float)
    if values.shape != (len(rows), len(columns)):
        raise ValueError("residual shape is incompatible with the supplied margins")
    if not np.isfinite(values).all():
        raise ValueError("residual values must be finite")
    if residual not in {"pearson", "deviance"}:
        raise ValueError("residual must be 'pearson' or 'deviance'")
    expected = np.outer(rows, columns) / rows.sum()
    # Residual inversion can create a nearly decomposable seed. A fixed mass of
    # 1e-3 of the mean cell count keeps IPF numerically identifiable while
    # changing a 30-cell 3x3 table by less than 0.004 count per entry.
    floor = 1e-3 * rows.sum() / expected.size
    if residual == "pearson":
        seed = expected + values * np.sqrt(expected)
        seed = np.maximum(seed, floor)
    else:
        seed = np.empty_like(expected)
        for index in np.ndindex(expected.shape):
            seed[index] = _deviance_count(expected[index], values[index])
        seed = np.maximum(seed, floor)
    return ipf_to_margins(seed, rows, columns)


def multinomial_deviance_per_observation(
    observed: np.ndarray, predicted: np.ndarray
) -> float:
    """Return multinomial deviance divided by the observed table total."""

    truth = np.asarray(observed, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    if truth.shape != estimate.shape or truth.ndim != 2 or min(truth.shape) < 2:
        raise ValueError("observed and predicted must be equal-size two-dimensional tables")
    if (
        not np.isfinite(truth).all()
        or not np.isfinite(estimate).all()
        or np.any(truth < 0.0)
        or np.any(estimate <= 0.0)
        or truth.sum() <= 0.0
    ):
        raise ValueError("tables must be finite, observed nonnegative, and predicted positive")
    if not np.isclose(truth.sum(), estimate.sum(), rtol=1e-8, atol=1e-8):
        raise ValueError("observed and predicted totals must match")
    positive = truth > 0.0
    deviance = 2.0 * float(
        np.sum(truth[positive] * np.log(truth[positive] / estimate[positive]))
    )
    return deviance / float(truth.sum())
