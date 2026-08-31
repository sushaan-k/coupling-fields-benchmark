"""Pure numerical core for the GSE252762 celiac CITE-seq confirmation.

The module has no filesystem or network access.  It reduces already selected
paired RNA/ADT cells to fixed-margin 2 x 2 tables, fits the calibration-only
models, predicts recipient tables from their margins, and evaluates the frozen
sample-level gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Mapping, Sequence

import numpy as np

from mapreg.heterogeneity_adaptive_coupling import (
    CouplingEstimationRefusal,
    expected_binary_table_from_log_odds,
    product_hypergraph_laplacian,
    signed_deviance_coordinate,
)
from mapreg.poisson_loglinear import (
    PoissonLoglinearRefusal,
    fit_poisson_loglinear_interaction,
    fit_ridge_profiled_poisson_interaction,
    reconstruct_poisson_tables,
)
from mapreg.structured_context_conditional import (
    StructuredContextConditionalFit,
    fit_structured_context_conditional_log_odds,
)


MARKER_PAIRS = (
    ("CD3D", "CD3"),
    ("CD4", "CD4"),
    ("CD8A", "CD8"),
    ("CD27", "CD27"),
    ("CD38", "CD38"),
    ("CD44", "CD44"),
    ("CD69", "CD69"),
    ("ITGAE", "CD103"),
    ("KLRB1", "CD161"),
)
RNA_MARKERS = tuple(pair[0] for pair in MARKER_PAIRS)
ADT_MARKERS = tuple(pair[1] for pair in MARKER_PAIRS)
CONTEXTS = ("CELIAC", "CONTROL")
DEPOSITED_CONDITIONS = ("ACD", "GFD", "CONTROL")
HELD_CONDITIONS = (
    "ACD",
    "ACD",
    "CONTROL",
    "CONTROL",
    "CONTROL",
    "CONTROL",
    "GFD",
    "GFD",
    "GFD",
    "GFD",
    "GFD",
    "GFD",
    "GFD",
)
CELL_BUDGET = 256
ADT_HIGH_COUNT = 128
COEFFICIENT_RIDGE_PENALTY = 0.01
DONOR_PROFILED_RIDGE_PENALTY = 0.01
PRIMARY_TRANSPORT_GRID = (0.0, 0.75, 1.0)
COMPARATOR_TRANSPORT_GRID = (0.0, 0.5, 0.75, 1.0, 1.25)
BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 25_276_201
DIFFERENCE_TOLERANCE = 1e-12
CELL_SALT = "GSE252762-CELIAC-CELL-v1"
ADT_TIE_SALT = "GSE252762-CELIAC-ADT-TIE-v1"
DESTROY_SALT = "GSE252762-CELIAC-DESTROY-v1"

CLASSICAL_METHODS = (
    "donor_stratified_ridge_poisson",
    "bias_reduced_context_poisson",
    "context_signed_deviance",
)
BENCHMARK_TIE_ORDER = (
    "independence",
    "donor_stratified_ridge_poisson",
    "bias_reduced_context_poisson",
    "context_signed_deviance",
)
MANDATORY_METHODS = (
    "primary",
    *CLASSICAL_METHODS,
    "destroyed_links",
    "independence",
)


class NoCompleteConfigurationError(ValueError):
    """No primary grid point completed every calibration fold."""


@dataclass(frozen=True)
class PrimaryConfig:
    """One primary grid point; tuple order is the frozen tie order."""

    donor_deviation_penalty: float
    graph_penalty: float
    transport_multiplier: float
    coefficient_ridge_penalty: float = COEFFICIENT_RIDGE_PENALTY


CONFIGURATIONS = tuple(
    PrimaryConfig(deviation, graph, transport)
    for deviation in (0.1, 1.0, 10.0)
    for graph in (0.0, 0.05, 0.2)
    for transport in PRIMARY_TRANSPORT_GRID
)


def _product_graph() -> np.ndarray:
    incidence = np.ones((len(MARKER_PAIRS), 1), dtype=float)
    graph = product_hypergraph_laplacian(incidence, incidence)
    mean_diagonal = float(np.mean(np.diag(graph)))
    if mean_diagonal <= 0.0 or not np.isfinite(mean_diagonal):
        raise AssertionError("product graph has no finite scale")
    return graph / mean_diagonal


PRODUCT_GRAPH_LAPLACIAN = _product_graph()


def product_graph_laplacian() -> np.ndarray:
    """Return the normalized graph joining pairs that share either marker."""

    return PRODUCT_GRAPH_LAPLACIAN.copy()


def _digest(salt: str, *parts: str) -> bytes:
    payload = salt.encode("utf-8")
    for part in parts:
        payload += b"\0" + part.encode("utf-8")
    return hashlib.sha256(payload).digest()


def _unique_strings(values: Sequence[str], label: str) -> tuple[str, ...]:
    axis = tuple(values)
    if not axis or any(not isinstance(value, str) or not value for value in axis):
        raise ValueError(f"{label} must contain nonempty strings")
    if len(set(axis)) != len(axis):
        raise ValueError(f"{label} must be unique")
    return axis


def _counts(values: np.ndarray, label: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if array.dtype.kind not in "iu" or array.dtype.kind == "b" or np.any(array < 0):
        raise ValueError(f"{label} must contain nonnegative integer counts")
    return array.astype(np.int64, copy=False)


def select_barcodes(
    barcodes: Sequence[str], sample_id: str, *, count: int = CELL_BUDGET
) -> tuple[str, ...]:
    """Select cells by a salted barcode rank independent of deposited order."""

    axis = _unique_strings(barcodes, "barcodes")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be nonempty")
    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError("count must be an integer")
    if count < 1 or count > len(axis):
        raise ValueError("count is outside the barcode axis")
    return tuple(
        sorted(
            axis,
            key=lambda barcode: (
                _digest(CELL_SALT, sample_id, barcode),
                barcode,
            ),
        )[:count]
    )


def rna_detection_states(counts: np.ndarray) -> np.ndarray:
    """Binarize the nine selected RNA markers as detected versus zero."""

    values = _counts(counts, "RNA counts", (CELL_BUDGET, len(RNA_MARKERS)))
    return (values > 0).astype(np.uint8)


def adt_top_states(
    counts: np.ndarray, barcodes: Sequence[str], sample_id: str
) -> np.ndarray:
    """Assign exactly the top 128 cells to the high state for each ADT marker."""

    values = _counts(counts, "ADT counts", (CELL_BUDGET, len(ADT_MARKERS)))
    axis = _unique_strings(barcodes, "selected barcodes")
    if len(axis) != CELL_BUDGET:
        raise ValueError("selected barcodes must contain 256 cells")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be nonempty")
    output = np.zeros(values.shape, dtype=np.uint8)
    for marker_index, marker in enumerate(ADT_MARKERS):
        order = sorted(
            range(CELL_BUDGET),
            key=lambda cell: (
                -int(values[cell, marker_index]),
                _digest(ADT_TIE_SALT, sample_id, marker, axis[cell]),
                axis[cell],
            ),
        )
        output[np.asarray(order[:ADT_HIGH_COUNT], dtype=np.int64), marker_index] = 1
    if not np.all(output.sum(axis=0) == ADT_HIGH_COUNT):
        raise AssertionError("ADT ranking changed its frozen margins")
    return output


def destroy_adt_states(
    states: np.ndarray, barcodes: Sequence[str], sample_id: str
) -> np.ndarray:
    """Shift complete ADT vectors by one deterministic 128-cell half-cycle."""

    values = np.asarray(states)
    axis = _unique_strings(barcodes, "selected barcodes")
    if values.shape != (CELL_BUDGET, len(ADT_MARKERS)) or len(axis) != CELL_BUDGET:
        raise ValueError("ADT states and barcodes must cover 256 selected cells")
    if values.dtype.kind not in "iub" or not np.isin(values, (0, 1)).all():
        raise ValueError("ADT states must be binary")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be nonempty")
    order = np.asarray(
        sorted(
            range(CELL_BUDGET),
            key=lambda cell: (
                _digest(DESTROY_SALT, sample_id, axis[cell]),
                axis[cell],
            ),
        ),
        dtype=np.int64,
    )
    output = np.empty_like(values)
    output[order] = values[np.roll(order, -ADT_HIGH_COUNT)]
    if not np.array_equal(output.sum(axis=0), values.sum(axis=0)):
        raise AssertionError("destroyed links changed an ADT margin")
    return output


def joint_binary_tables(rna_states: np.ndarray, adt_states: np.ndarray) -> np.ndarray:
    """Build all 81 ordered RNA-negative/positive by ADT-low/high tables."""

    rna = np.asarray(rna_states)
    adt = np.asarray(adt_states)
    expected = (CELL_BUDGET, len(MARKER_PAIRS))
    if rna.shape != expected or adt.shape != expected:
        raise ValueError("RNA and ADT states must be 256 by nine")
    if not np.isin(rna, (0, 1)).all() or not np.isin(adt, (0, 1)).all():
        raise ValueError("RNA and ADT states must be binary")
    first = rna.astype(np.int64)
    second = adt.astype(np.int64)
    n11 = first.T @ second
    rna_positive = first.sum(axis=0)[:, None]
    adt_high = second.sum(axis=0)[None, :]
    output = np.empty((len(MARKER_PAIRS), len(MARKER_PAIRS), 2, 2), dtype=np.int64)
    output[..., 1, 1] = n11
    output[..., 1, 0] = rna_positive - n11
    output[..., 0, 1] = adt_high - n11
    output[..., 0, 0] = CELL_BUDGET - rna_positive - adt_high + n11
    return output


def sample_tables(
    rna_counts: np.ndarray,
    adt_counts: np.ndarray,
    barcodes: Sequence[str],
    sample_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return real and destroyed 81-table panels for one selected sample."""

    rna = rna_detection_states(rna_counts)
    adt = adt_top_states(adt_counts, barcodes, sample_id)
    destroyed = destroy_adt_states(adt, barcodes, sample_id)
    return joint_binary_tables(rna, adt), joint_binary_tables(rna, destroyed)


def rna_margin_tables(rna_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return held prediction margins without reading held ADT values."""

    rna = rna_detection_states(rna_counts).astype(np.int64)
    positive = rna.sum(axis=0)
    marker_rows = np.stack((CELL_BUDGET - positive, positive), axis=-1)
    rows = np.broadcast_to(
        marker_rows[:, None, :],
        (len(RNA_MARKERS), len(ADT_MARKERS), 2),
    ).copy()
    columns = np.full(rows.shape, ADT_HIGH_COUNT, dtype=np.int64)
    return rows, columns


def diagnosis_context(diagnosis: str) -> str:
    """Map the frozen deposited diagnosis labels to the two model contexts."""

    if diagnosis in {"ACD", "GFD"}:
        return "CELIAC"
    if diagnosis == "CONTROL":
        return "CONTROL"
    raise ValueError("diagnosis is outside the frozen ACD/GFD/control contrast")


def context_design(contexts: Sequence[str]) -> np.ndarray:
    """Encode CELIAC and CONTROL as symmetric one-hot columns."""

    labels = tuple(contexts)
    if not labels or any(label not in CONTEXTS for label in labels):
        raise ValueError("contexts must contain only CELIAC and CONTROL")
    output = np.zeros((len(labels), len(CONTEXTS)), dtype=float)
    output[np.arange(len(labels)), [CONTEXTS.index(label) for label in labels]] = 1.0
    return output


def _validated_contexts(
    contexts: Sequence[str], count: int, *, require_both: bool = True
) -> tuple[str, ...]:
    labels = tuple(contexts)
    if len(labels) != count:
        raise ValueError("tables and contexts must share the sample axis")
    context_design(labels)
    if require_both and set(labels) != set(CONTEXTS):
        raise ValueError("the source panel must contain both frozen contexts")
    return labels


def _margins(tables: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(tables)
    return values.sum(axis=-1), values.sum(axis=-2)


def _validated_panel(tables: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(tables)
    expected_tail = (len(RNA_MARKERS), len(ADT_MARKERS), 2, 2)
    if values.ndim != 5 or values.shape[1:] != expected_tail or values.shape[0] < 2:
        raise ValueError(f"{label} must have shape (samples, 9, 9, 2, 2)")
    if values.dtype.kind not in "iu" or values.dtype.kind == "b" or np.any(values < 0):
        raise ValueError(f"{label} must contain nonnegative integer counts")
    if np.any(values.sum(axis=(-2, -1)) != CELL_BUDGET):
        raise ValueError(f"every {label} table must contain 256 cells")
    rows, columns = _margins(values)
    if not np.all(columns[..., 0] == ADT_HIGH_COUNT) or not np.all(
        columns[..., 1] == ADT_HIGH_COUNT
    ):
        raise ValueError(f"{label} must retain the frozen 128/128 ADT margins")
    if not np.all(rows == rows[:, :, :1, :]):
        raise ValueError(f"{label} RNA margins must repeat across ADT markers")
    return values.astype(np.int64, copy=False)


def _validated_recipient_margins(
    row_margins: np.ndarray, column_margins: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    shape = (len(RNA_MARKERS), len(ADT_MARKERS), 2)
    rows = _counts(row_margins, "row margins", shape)
    columns = _counts(column_margins, "column margins", shape)
    if np.any(rows.sum(axis=-1) != CELL_BUDGET) or not np.array_equal(
        rows.sum(axis=-1), columns.sum(axis=-1)
    ):
        raise ValueError("every recipient margin must contain 256 cells")
    if not np.all(rows == rows[:, :1, :]):
        raise ValueError("RNA margins must repeat across ADT markers")
    if not np.all(columns == ADT_HIGH_COUNT):
        raise ValueError("ADT margins must be the frozen 128/128 split")
    return rows, columns


def predict_conditional_tables(
    log_odds: np.ndarray,
    row_margins: np.ndarray,
    column_margins: np.ndarray,
) -> np.ndarray:
    """Reconstruct exact conditional means at recipient fixed margins."""

    field = np.asarray(log_odds, dtype=float)
    if field.shape != (len(RNA_MARKERS), len(ADT_MARKERS)):
        raise ValueError("log_odds must be a 9 by 9 field")
    if not np.isfinite(field).all():
        raise ValueError("log_odds must be finite")
    rows, columns = _validated_recipient_margins(row_margins, column_margins)
    output = np.empty((*field.shape, 2, 2), dtype=float)
    for index in np.ndindex(field.shape):
        output[index] = expected_binary_table_from_log_odds(
            float(field[index]), rows[index], columns[index]
        )
    if not np.isfinite(output).all() or np.any(output < 0.0):
        raise FloatingPointError("conditional prediction is not finite")
    if not np.allclose(
        output.sum(axis=-1), rows, atol=1e-8, rtol=0.0
    ) or not np.allclose(output.sum(axis=-2), columns, atol=1e-8, rtol=0.0):
        raise FloatingPointError("conditional prediction changed a recipient margin")
    return output


def entity_deviance(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Return multinomial deviance per cell for every ordered marker pair."""

    truth = np.asarray(observed)
    estimate = np.asarray(predicted, dtype=float)
    if truth.shape != (9, 9, 2, 2) or estimate.shape != truth.shape:
        raise ValueError("observed and predicted must each contain 81 binary tables")
    if truth.dtype.kind not in "iu" or truth.dtype.kind == "b" or np.any(truth < 0):
        raise ValueError("observed tables must contain nonnegative integer counts")
    if np.any(truth.sum(axis=(-2, -1)) != CELL_BUDGET):
        raise ValueError("every observed table must contain 256 cells")
    if not np.isfinite(estimate).all() or np.any(estimate < 0.0):
        raise FloatingPointError("predicted tables must be finite and nonnegative")
    if not np.allclose(truth.sum(axis=-1), estimate.sum(axis=-1), atol=1e-8, rtol=0.0):
        raise FloatingPointError("prediction changed a row margin")
    if not np.allclose(truth.sum(axis=-2), estimate.sum(axis=-2), atol=1e-8, rtol=0.0):
        raise FloatingPointError("prediction changed a column margin")
    positive = truth > 0
    if np.any(estimate[positive] <= 0.0):
        raise FloatingPointError("prediction assigns zero mass to an observed cell")
    terms = np.zeros_like(estimate)
    terms[positive] = truth[positive] * np.log(truth[positive] / estimate[positive])
    return 2.0 * terms.sum(axis=(-2, -1)) / CELL_BUDGET


def sample_loss(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Average the 81 coordinate losses with one equal weight per sample."""

    return float(np.mean(entity_deviance(observed, predicted)))


def _fit_primary(
    tables: np.ndarray, contexts: Sequence[str], config: PrimaryConfig
) -> StructuredContextConditionalFit:
    return fit_structured_context_conditional_log_odds(
        tables,
        context_design(contexts),
        graph_laplacian=PRODUCT_GRAPH_LAPLACIAN,
        donor_deviation_penalty=config.donor_deviation_penalty,
        coefficient_ridge_penalty=config.coefficient_ridge_penalty,
        graph_penalty=config.graph_penalty,
        minimum_informative_donors=2,
        maximum_condition_number=1e14,
        maximum_iterations=200,
        tolerance=1e-8,
    )


def _context_field(coefficient: np.ndarray, context: str, alpha: float) -> np.ndarray:
    if context not in CONTEXTS:
        raise ValueError("unknown context")
    values = np.asarray(coefficient, dtype=float)
    if values.shape != (len(CONTEXTS), len(RNA_MARKERS), len(ADT_MARKERS)):
        raise ValueError("coefficient must follow the frozen context and marker axes")
    return float(alpha) * values[CONTEXTS.index(context)]


def _fit_context_signed_deviance(
    tables: np.ndarray, contexts: Sequence[str]
) -> np.ndarray:
    values = np.asarray(tables)
    labels = tuple(contexts)
    output = np.empty((len(CONTEXTS), len(RNA_MARKERS), len(ADT_MARKERS)))
    for context_index, context in enumerate(CONTEXTS):
        selected = values[np.asarray(labels, dtype=object) == context]
        if len(selected) < 1:
            raise ValueError("each context needs at least one source sample")
        coordinates = np.empty(selected.shape[:3], dtype=float)
        for index in np.ndindex(coordinates.shape):
            coordinates[index] = signed_deviance_coordinate(selected[index])
        output[context_index] = coordinates.mean(axis=0)
    return output


def _fit_bias_reduced_context_poisson(
    tables: np.ndarray, contexts: Sequence[str]
) -> np.ndarray:
    """Fit finite context interactions from pooled cells with a 1/2 correction."""

    values = np.asarray(tables)
    labels = tuple(contexts)
    output = np.empty((len(CONTEXTS), len(RNA_MARKERS), len(ADT_MARKERS)))
    for context_index, context in enumerate(CONTEXTS):
        selected = values[np.asarray(labels, dtype=object) == context]
        if len(selected) < 1:
            raise ValueError("each context needs at least one source sample")
        pooled = selected.sum(axis=0, dtype=np.int64).astype(float) + 0.5
        output[context_index] = np.log(
            pooled[..., 0, 0]
            * pooled[..., 1, 1]
            / (pooled[..., 0, 1] * pooled[..., 1, 0])
        )
    if not np.isfinite(output).all():
        raise FloatingPointError("bias-reduced Poisson interactions are not finite")
    return output


def _ridge_poisson_certificate(fit: object) -> dict[str, object]:
    status = np.asarray(fit.status)
    return {
        "estimator": fit.estimator,
        "converged": bool(fit.converged),
        "ridge_penalty": float(fit.ridge_penalty),
        "score_tolerance": float(fit.score_tolerance),
        "certificate_tolerance": float(fit.certificate_tolerance),
        "bracket_bound": float(fit.bracket_bound),
        "status_grid": status.tolist(),
        "status_counts": {
            label: int(np.count_nonzero(status == label))
            for label in ("FINITE", "NO_INFORMATION")
        },
        "informative_table_count": np.asarray(
            fit.informative_table_count, dtype=int
        ).tolist(),
        "degenerate_table_count": np.asarray(
            fit.degenerate_table_count, dtype=int
        ).tolist(),
        "mean_data_information": np.asarray(
            fit.mean_data_information, dtype=float
        ).tolist(),
        "penalized_information": np.asarray(
            fit.penalized_information, dtype=float
        ).tolist(),
        "mean_profile_log_likelihood": np.asarray(
            fit.mean_profile_log_likelihood, dtype=float
        ).tolist(),
        "penalized_objective": np.asarray(
            fit.penalized_objective, dtype=float
        ).tolist(),
        "mean_score": np.asarray(fit.mean_score, dtype=float).tolist(),
        "penalized_score": np.asarray(fit.penalized_score, dtype=float).tolist(),
        "mean_margin_width": np.asarray(fit.mean_margin_width, dtype=float).tolist(),
        "bracket_lower": np.asarray(fit.bracket_lower, dtype=float).tolist(),
        "bracket_upper": np.asarray(fit.bracket_upper, dtype=float).tolist(),
        "bracket_lower_score": np.asarray(
            fit.bracket_lower_score, dtype=float
        ).tolist(),
        "bracket_upper_score": np.asarray(
            fit.bracket_upper_score, dtype=float
        ).tolist(),
        "root_iterations": np.asarray(fit.root_iterations, dtype=int).tolist(),
        "maximum_absolute_penalized_score": float(fit.maximum_absolute_penalized_score),
        "maximum_scaled_penalized_score": float(fit.maximum_scaled_penalized_score),
        "maximum_absolute_row_margin_error": float(
            fit.maximum_absolute_row_margin_error
        ),
        "maximum_absolute_column_margin_error": float(
            fit.maximum_absolute_column_margin_error
        ),
        "maximum_absolute_log_odds_error": float(fit.maximum_absolute_log_odds_error),
    }


def _fit_donor_stratified_ridge_poisson(
    tables: np.ndarray, contexts: Sequence[str]
) -> tuple[np.ndarray, dict[str, object]]:
    labels = np.asarray(tuple(contexts), dtype=object)
    fit = fit_ridge_profiled_poisson_interaction(
        np.asarray(tables),
        labels,
        ridge_penalty=DONOR_PROFILED_RIDGE_PENALTY,
        score_tolerance=1e-10,
        certificate_tolerance=1e-8,
    )
    label_to_index = {label: index for index, label in enumerate(fit.group_labels)}
    if set(label_to_index) != set(CONTEXTS):
        raise ValueError("ridge Poisson fit lacks a frozen context")
    order = [label_to_index[context] for context in CONTEXTS]
    field = np.asarray(fit.log_odds, dtype=float)[order]
    certificate = _ridge_poisson_certificate(fit)
    certificate["fitted_group_order"] = [str(label) for label in fit.group_labels]
    certificate["reported_context_order"] = list(CONTEXTS)
    return field, certificate


def _fractional_signed_deviance(table: np.ndarray) -> float:
    values = np.asarray(table, dtype=float)
    expected = np.outer(values.sum(axis=1), values.sum(axis=0)) / values.sum()
    positive = values > 0.0
    deviance = 2.0 * float(
        np.sum(values[positive] * np.log(values[positive] / expected[positive]))
    )
    determinant = float(values[0, 0] * values[1, 1] - values[0, 1] * values[1, 0])
    return float(np.sign(determinant) * math.sqrt(max(0.0, deviance)))


def _residual_table(
    coordinate: float, row_margin: np.ndarray, column_margin: np.ndarray
) -> np.ndarray:
    rows = np.asarray(row_margin, dtype=float)
    columns = np.asarray(column_margin, dtype=float)
    total = float(rows.sum())
    lower = max(0.0, float(rows[0] + columns[0] - total))
    upper = min(float(rows[0]), float(columns[0]))

    def table_at(value: float) -> np.ndarray:
        return np.asarray(
            [
                [value, rows[0] - value],
                [columns[0] - value, rows[1] - columns[0] + value],
            ]
        )

    if upper <= lower:
        return table_at(lower)
    epsilon = min(1e-8, 0.25 * (upper - lower))
    left, right = lower + epsilon, upper - epsilon
    target = min(
        max(float(coordinate), _fractional_signed_deviance(table_at(left))),
        _fractional_signed_deviance(table_at(right)),
    )
    for _ in range(96):
        midpoint = 0.5 * (left + right)
        if _fractional_signed_deviance(table_at(midpoint)) < target:
            left = midpoint
        else:
            right = midpoint
    return table_at(0.5 * (left + right))


def _predict_residual_at_margins(
    field: np.ndarray, row_margins: np.ndarray, column_margins: np.ndarray
) -> np.ndarray:
    coordinates = np.asarray(field, dtype=float)
    rows, columns = _validated_recipient_margins(row_margins, column_margins)
    if coordinates.shape != (len(RNA_MARKERS), len(ADT_MARKERS)):
        raise ValueError("residual field must be 9 by 9")
    if np.all(coordinates == 0.0):
        return predict_conditional_tables(np.zeros_like(coordinates), rows, columns)
    output = np.empty((*coordinates.shape, 2, 2), dtype=float)
    for index in np.ndindex(coordinates.shape):
        output[index] = _residual_table(
            float(coordinates[index]), rows[index], columns[index]
        )
    return output


def _predict_poisson_at_margins(
    field: np.ndarray,
    row_margins: np.ndarray,
    column_margins: np.ndarray,
    alpha: float,
) -> np.ndarray:
    rows, columns = _validated_recipient_margins(row_margins, column_margins)
    if float(alpha) == 0.0:
        return predict_conditional_tables(
            np.zeros_like(field, dtype=float), rows, columns
        )
    return reconstruct_poisson_tables(
        np.asarray(field, dtype=float),
        rows,
        columns,
        transport_scale=float(alpha),
    ).table


def _selection_inputs(
    calibration_tables: np.ndarray,
    calibration_contexts: Sequence[str],
    expected_sample_count: int | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    tables = _validated_panel(calibration_tables, "calibration tables")
    if expected_sample_count is not None:
        if (
            isinstance(expected_sample_count, bool)
            or not isinstance(expected_sample_count, int)
            or expected_sample_count < 4
        ):
            raise ValueError(
                "expected_sample_count must be an integer of at least four"
            )
        if len(tables) != expected_sample_count:
            raise ValueError("calibration sample count differs from the frozen count")
    if len(tables) < 4:
        raise ValueError("calibration requires at least four samples")
    labels = _validated_contexts(calibration_contexts, len(tables))
    for validation in range(len(tables)):
        training = labels[:validation] + labels[validation + 1 :]
        if set(training) != set(CONTEXTS):
            raise ValueError(
                "every leave-one-out training fold must retain both contexts"
            )
    return tables, labels


def select_primary_configuration(
    calibration_tables: np.ndarray,
    calibration_contexts: Sequence[str],
    *,
    expected_sample_count: int | None = None,
) -> tuple[PrimaryConfig, dict[PrimaryConfig, np.ndarray]]:
    """Select the primary grid point by calibration-only leave-one-sample-out loss."""

    tables, labels = _selection_inputs(
        calibration_tables, calibration_contexts, expected_sample_count
    )
    count = len(tables)
    losses = {config: np.full(count, np.inf, dtype=float) for config in CONFIGURATIONS}
    indices = np.arange(count)
    for validation in range(count):
        training = indices[indices != validation]
        training_labels = [labels[index] for index in training]
        rows, columns = _margins(tables[validation])
        for deviation in (0.1, 1.0, 10.0):
            for graph in (0.0, 0.05, 0.2):
                representative = PrimaryConfig(deviation, graph, 0.75)
                try:
                    fit = _fit_primary(
                        tables[training], training_labels, representative
                    )
                except (CouplingEstimationRefusal, FloatingPointError):
                    continue
                for alpha in PRIMARY_TRANSPORT_GRID:
                    config = PrimaryConfig(deviation, graph, alpha)
                    prediction = predict_conditional_tables(
                        _context_field(fit.coefficient, labels[validation], alpha),
                        rows,
                        columns,
                    )
                    losses[config][validation] = sample_loss(
                        tables[validation], prediction
                    )
    complete = [
        config for config in CONFIGURATIONS if np.isfinite(losses[config]).all()
    ]
    if not complete:
        raise NoCompleteConfigurationError(
            "no primary configuration completed every calibration fold"
        )
    order = {config: index for index, config in enumerate(CONFIGURATIONS)}
    selected = min(
        complete,
        key=lambda config: (float(losses[config].mean()), order[config]),
    )
    return selected, losses


def select_comparator_alphas(
    calibration_tables: np.ndarray,
    calibration_contexts: Sequence[str],
    *,
    expected_sample_count: int | None = None,
) -> tuple[dict[str, float], dict[str, dict[float, np.ndarray]], np.ndarray]:
    """Select classical transports and score fixed independence by LOSO."""

    tables, labels = _selection_inputs(
        calibration_tables, calibration_contexts, expected_sample_count
    )
    count = len(tables)
    losses = {
        method: {
            alpha: np.empty(count, dtype=float) for alpha in COMPARATOR_TRANSPORT_GRID
        }
        for method in CLASSICAL_METHODS
    }
    independence_losses = np.empty(count, dtype=float)
    indices = np.arange(count)
    for validation in range(count):
        training = indices[indices != validation]
        training_labels = [labels[index] for index in training]
        ridge_poisson, _certificate = _fit_donor_stratified_ridge_poisson(
            tables[training], training_labels
        )
        bias_reduced = _fit_bias_reduced_context_poisson(
            tables[training], training_labels
        )
        residual = _fit_context_signed_deviance(tables[training], training_labels)
        truth = tables[validation]
        rows, columns = _margins(truth)
        context_index = CONTEXTS.index(labels[validation])
        independence = predict_conditional_tables(
            np.zeros((len(RNA_MARKERS), len(ADT_MARKERS))), rows, columns
        )
        independence_losses[validation] = sample_loss(truth, independence)
        for alpha in COMPARATOR_TRANSPORT_GRID:
            ridge_prediction = _predict_poisson_at_margins(
                ridge_poisson[context_index], rows, columns, alpha
            )
            poisson_prediction = _predict_poisson_at_margins(
                bias_reduced[context_index], rows, columns, alpha
            )
            residual_prediction = _predict_residual_at_margins(
                alpha * residual[context_index], rows, columns
            )
            losses["donor_stratified_ridge_poisson"][alpha][validation] = sample_loss(
                truth, ridge_prediction
            )
            losses["bias_reduced_context_poisson"][alpha][validation] = sample_loss(
                truth, poisson_prediction
            )
            losses["context_signed_deviance"][alpha][validation] = sample_loss(
                truth, residual_prediction
            )
    selected = {
        method: min(
            COMPARATOR_TRANSPORT_GRID,
            key=lambda alpha: (
                float(losses[method][alpha].mean()),
                COMPARATOR_TRANSPORT_GRID.index(alpha),
            ),
        )
        for method in CLASSICAL_METHODS
    }
    return selected, losses, independence_losses


def strongest_benchmark_from_calibration(
    selected_alphas: Mapping[str, float],
    calibration_losses: Mapping[str, Mapping[float, np.ndarray]],
    independence_losses: np.ndarray,
) -> str:
    """Freeze the strongest benchmark from calibration LOSO only."""

    if set(selected_alphas) != set(CLASSICAL_METHODS) or set(calibration_losses) != set(
        CLASSICAL_METHODS
    ):
        raise ValueError("all frozen classical comparators are required")
    independence = np.asarray(independence_losses, dtype=float)
    if (
        independence.ndim != 1
        or len(independence) < 1
        or not np.isfinite(independence).all()
    ):
        raise ValueError("independence calibration losses must be finite")
    means = {"independence": float(independence.mean())}
    for method in CLASSICAL_METHODS:
        alpha = float(selected_alphas[method])
        if (
            alpha not in COMPARATOR_TRANSPORT_GRID
            or alpha not in calibration_losses[method]
        ):
            raise ValueError("selected comparator alpha lacks calibration losses")
        values = np.asarray(calibration_losses[method][alpha], dtype=float)
        if values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all():
            raise ValueError("calibration comparator losses must be finite vectors")
        means[method] = float(values.mean())
    return min(
        BENCHMARK_TIE_ORDER,
        key=lambda method: (means[method], BENCHMARK_TIE_ORDER.index(method)),
    )


def _fit_certificate(fit: StructuredContextConditionalFit) -> dict[str, object]:
    return {
        "converged": bool(fit.converged),
        "iterations": int(fit.iterations),
        "objective": float(fit.objective),
        "scaled_gradient_norm": float(fit.scaled_gradient_norm),
        "schur_condition_number": float(fit.schur_condition_number),
        "donor_curvature_condition_number": float(fit.donor_curvature_condition_number),
        "minimum_schur_eigenvalue": float(fit.minimum_schur_eigenvalue),
        "maximum_schur_eigenvalue": float(fit.maximum_schur_eigenvalue),
        "minimum_donor_curvature": float(fit.minimum_donor_curvature),
        "maximum_donor_curvature": float(fit.maximum_donor_curvature),
        "support_count": np.asarray(fit.support_count, dtype=int).tolist(),
        "minimum_support_count": int(np.min(fit.support_count)),
        "minimum_informative_donors": 2,
        "gradient_tolerance": float(fit.gradient_tolerance),
        "maximum_condition_number": float(fit.maximum_condition_number),
        "maximum_iterations": 200,
        "graph_source": fit.graph_source,
        "graph_nullity": int(fit.graph_nullity),
    }


def _profiled_poisson_report(
    tables: np.ndarray, contexts: Sequence[str]
) -> tuple[np.ndarray | None, dict[str, object]]:
    """Fit each context-coordinate profile separately for report-only status."""

    values = np.asarray(tables)
    labels = np.asarray(tuple(contexts), dtype=object)
    field = np.full((len(CONTEXTS), 9, 9), np.nan, dtype=float)
    status_grid: list[list[list[str]]] = []
    records = []
    maximum_scaled_score = 0.0
    counts = {"FINITE": 0, "BOUNDARY": 0, "NO_INFORMATION": 0}
    for context_index, context in enumerate(CONTEXTS):
        selected = values[labels == context]
        context_grid: list[list[str]] = []
        for rna_index, rna_marker in enumerate(RNA_MARKERS):
            row = []
            for adt_index, adt_marker in enumerate(ADT_MARKERS):
                current = selected[:, rna_index, adt_index]
                rows = current.sum(axis=-1)
                columns = current.sum(axis=-2)
                lower = np.maximum(
                    0,
                    rows[:, 0] + columns[:, 0] - current.sum(axis=(-2, -1)),
                )
                upper = np.minimum(rows[:, 0], columns[:, 0])
                reason = None
                if not np.any(upper > lower):
                    status = "NO_INFORMATION"
                else:
                    try:
                        fit = fit_poisson_loglinear_interaction(current)
                        field[context_index, rna_index, adt_index] = float(
                            fit.log_odds[0]
                        )
                        maximum_scaled_score = max(
                            maximum_scaled_score, float(fit.maximum_scaled_score)
                        )
                        status = "FINITE"
                    except (
                        PoissonLoglinearRefusal,
                        FloatingPointError,
                        ValueError,
                    ) as error:
                        status = "BOUNDARY"
                        reason = str(error)
                counts[status] += 1
                row.append(status)
                record = {
                    "context": context,
                    "rna_marker": rna_marker,
                    "adt_marker": adt_marker,
                    "status": status,
                }
                if reason is not None:
                    record["reason"] = reason
                records.append(record)
            context_grid.append(row)
        status_grid.append(context_grid)
    all_finite = counts["FINITE"] == len(CONTEXTS) * 9 * 9
    overall = (
        "VALID" if all_finite else "PARTIAL" if counts["FINITE"] else "UNAVAILABLE"
    )
    return (
        field if all_finite else None,
        {
            "status": overall,
            "coordinate_status_counts": counts,
            "coordinate_status_grid": status_grid,
            "coordinates": records,
            "maximum_scaled_score_over_finite_coordinates": maximum_scaled_score,
            "used_for_selection_or_gate": False,
        },
    )


def fit_models(
    tables: np.ndarray,
    destroyed_tables: np.ndarray,
    contexts: Sequence[str],
    config: PrimaryConfig,
    comparator_alphas: Mapping[str, float],
    *,
    strongest_benchmark: str,
) -> dict[str, object]:
    """Fit the primary field, destroyed control, and fixed classical baselines."""

    values = _validated_panel(tables, "source tables")
    destroyed = _validated_panel(destroyed_tables, "destroyed tables")
    labels = _validated_contexts(contexts, len(values))
    if values.shape != destroyed.shape:
        raise ValueError("real and destroyed panels must have identical shapes")
    if any(
        not np.array_equal(left, right)
        for left, right in zip(_margins(values), _margins(destroyed))
    ):
        raise ValueError("destroyed links must preserve every fixed margin")
    if config not in CONFIGURATIONS:
        raise ValueError("primary configuration is outside the frozen grid")
    if set(comparator_alphas) != set(CLASSICAL_METHODS) or any(
        float(alpha) not in COMPARATOR_TRANSPORT_GRID
        for alpha in comparator_alphas.values()
    ):
        raise ValueError("comparator transports differ from the frozen grid")
    if strongest_benchmark not in BENCHMARK_TIE_ORDER:
        raise ValueError("strongest benchmark was not calibration-frozen")

    primary = _fit_primary(values, labels, config)
    destroyed_fit = _fit_primary(destroyed, labels, config)
    ridge_poisson, ridge_poisson_certificate = _fit_donor_stratified_ridge_poisson(
        values, labels
    )
    profiled_field, profiled_status = _profiled_poisson_report(values, labels)

    return {
        "configuration": config,
        "context_order": CONTEXTS,
        "comparator_alphas": {
            method: float(comparator_alphas[method]) for method in CLASSICAL_METHODS
        },
        "strongest_benchmark": strongest_benchmark,
        "primary_field": np.asarray(primary.coefficient),
        "primary_fit_certificate": _fit_certificate(primary),
        "destroyed_field": np.asarray(destroyed_fit.coefficient),
        "destroyed_fit_certificate": _fit_certificate(destroyed_fit),
        "donor_stratified_ridge_poisson_field": ridge_poisson,
        "donor_stratified_ridge_poisson_certificate": ridge_poisson_certificate,
        "bias_reduced_context_poisson_field": _fit_bias_reduced_context_poisson(
            values, labels
        ),
        "context_signed_deviance_field": _fit_context_signed_deviance(values, labels),
        "profiled_poisson_field": profiled_field,
        "profiled_poisson_status": profiled_status,
    }


def predict_models_at_margins(
    models: Mapping[str, object],
    row_margins: np.ndarray,
    column_margins: np.ndarray,
    context: str,
) -> dict[str, np.ndarray]:
    """Predict one sample before its paired RNA/ADT links are revealed."""

    if context not in CONTEXTS:
        raise ValueError("unknown context")
    config = models.get("configuration")
    if not isinstance(config, PrimaryConfig):
        raise TypeError("models lack a PrimaryConfig")
    if tuple(models.get("context_order", ())) != CONTEXTS:
        raise ValueError("model context order differs from the frozen order")
    alphas = models.get("comparator_alphas")
    if not isinstance(alphas, Mapping) or set(alphas) != set(CLASSICAL_METHODS):
        raise ValueError("model comparator transports are absent")
    if models.get("strongest_benchmark") not in BENCHMARK_TIE_ORDER:
        raise ValueError("model strongest benchmark was not calibration-frozen")
    context_index = CONTEXTS.index(context)
    rows, columns = _validated_recipient_margins(row_margins, column_margins)
    primary = np.asarray(models["primary_field"], dtype=float)
    destroyed = np.asarray(models["destroyed_field"], dtype=float)
    ridge_poisson = np.asarray(
        models["donor_stratified_ridge_poisson_field"], dtype=float
    )
    bias_reduced = np.asarray(models["bias_reduced_context_poisson_field"], dtype=float)
    residual = np.asarray(models["context_signed_deviance_field"], dtype=float)
    expected_shape = (len(CONTEXTS), len(RNA_MARKERS), len(ADT_MARKERS))
    if any(
        field.shape != expected_shape
        for field in (primary, destroyed, ridge_poisson, bias_reduced, residual)
    ):
        raise ValueError("a model field differs from the frozen axes")

    output = {
        "primary": predict_conditional_tables(
            config.transport_multiplier * primary[context_index], rows, columns
        ),
        "donor_stratified_ridge_poisson": _predict_poisson_at_margins(
            ridge_poisson[context_index],
            rows,
            columns,
            float(alphas["donor_stratified_ridge_poisson"]),
        ),
        "bias_reduced_context_poisson": _predict_poisson_at_margins(
            bias_reduced[context_index],
            rows,
            columns,
            float(alphas["bias_reduced_context_poisson"]),
        ),
        "context_signed_deviance": _predict_residual_at_margins(
            float(alphas["context_signed_deviance"]) * residual[context_index],
            rows,
            columns,
        ),
        "destroyed_links": predict_conditional_tables(
            config.transport_multiplier * destroyed[context_index], rows, columns
        ),
        "independence": predict_conditional_tables(
            np.zeros((len(RNA_MARKERS), len(ADT_MARKERS))), rows, columns
        ),
    }
    profiled = models.get("profiled_poisson_field")
    if profiled is not None:
        profile_field = np.asarray(profiled, dtype=float)
        if profile_field.shape != expected_shape:
            raise ValueError("profiled Poisson field differs from the frozen axes")
        output["unpenalized_profiled_poisson"] = _predict_poisson_at_margins(
            profile_field[context_index], rows, columns, 1.0
        )
    return output


def predict_models(
    models: Mapping[str, object], truth: np.ndarray, context: str
) -> dict[str, np.ndarray]:
    """Predict one sample at the observed margins; truth is used only for margins."""

    observed = np.asarray(truth)
    if observed.shape != (9, 9, 2, 2):
        raise ValueError("truth must contain 81 binary tables")
    rows, columns = _margins(observed)
    return predict_models_at_margins(models, rows, columns, context)


def panel_losses(
    models: Mapping[str, object], tables: np.ndarray, contexts: Sequence[str]
) -> dict[str, np.ndarray]:
    """Return one sample-equal deviance for each available method and sample."""

    values = _validated_panel(tables, "evaluation tables")
    labels = _validated_contexts(contexts, len(values), require_both=False)
    output: dict[str, np.ndarray] = {}
    for sample, context in enumerate(labels):
        predictions = predict_models(models, values[sample], context)
        if sample == 0:
            output = {
                method: np.empty(len(values), dtype=float) for method in predictions
            }
        elif set(predictions) != set(output):
            raise AssertionError("prediction availability changed across samples")
        for method, prediction in predictions.items():
            output[method][sample] = sample_loss(values[sample], prediction)
    return output


def exact_one_sided_sign_test(differences: np.ndarray) -> dict[str, object]:
    """Test whether paired primary-minus-comparator differences favor primary."""

    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all():
        raise ValueError("differences must be a nonempty finite vector")
    nonzero = values[np.abs(values) > DIFFERENCE_TOLERANCE]
    favorable = int(np.count_nonzero(nonzero < -DIFFERENCE_TOLERANCE))
    count = len(nonzero)
    probability = (
        float(
            sum(math.comb(count, index) for index in range(favorable, count + 1))
            / (2**count)
        )
        if count
        else 1.0
    )
    return {
        "favorable": favorable,
        "unfavorable": int(count - favorable),
        "ties": int(len(values) - count),
        "nonzero_pairs": int(count),
        "one_sided_probability": probability,
    }


def paired_bootstrap_intervals(
    differences: Mapping[str, np.ndarray],
    conditions: Sequence[str],
    *,
    seed: int = BOOTSTRAP_SEED,
    draws: int = BOOTSTRAPS,
) -> dict[str, tuple[float, float]]:
    """Return paired intervals using one shared stratified resample draw."""

    labels = tuple(conditions)
    values = {key: np.asarray(value, dtype=float) for key, value in differences.items()}
    if not values or any(
        value.ndim != 1 or value.shape != (len(labels),) or not np.isfinite(value).all()
        for value in values.values()
    ):
        raise ValueError("differences must be equal-length finite vectors")
    if labels != HELD_CONDITIONS:
        raise ValueError("bootstrap conditions must follow the frozen held-panel order")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    generator = np.random.default_rng(seed)
    label_array = np.asarray(labels, dtype=object)
    selected_indices = []
    for condition in DEPOSITED_CONDITIONS:
        count = int(np.count_nonzero(label_array == condition))
        selected_indices.append(generator.integers(0, count, size=(draws, count)))
    intervals = {}
    for method, value in values.items():
        resampled = []
        for condition, indices in zip(DEPOSITED_CONDITIONS, selected_indices):
            selected = value[label_array == condition]
            resampled.append(selected[indices])
        means = np.concatenate(resampled, axis=1).mean(axis=1)
        interval = np.quantile(means, (0.025, 0.975), method="linear")
        intervals[method] = (float(interval[0]), float(interval[1]))
    return intervals


def paired_bootstrap_interval(
    differences: np.ndarray,
    conditions: Sequence[str],
    *,
    seed: int = BOOTSTRAP_SEED,
    draws: int = BOOTSTRAPS,
) -> tuple[float, float]:
    """Return one interval through the shared-resample implementation."""

    return paired_bootstrap_intervals(
        {"comparison": differences}, conditions, seed=seed, draws=draws
    )["comparison"]


def _validated_losses(
    losses: Mapping[str, np.ndarray], expected_count: int
) -> dict[str, np.ndarray]:
    if any(method not in losses for method in MANDATORY_METHODS):
        raise ValueError("mandatory method losses are absent")
    output = {}
    for method, raw in losses.items():
        values = np.asarray(raw, dtype=float)
        if (
            values.shape != (expected_count,)
            or not np.isfinite(values).all()
            or np.any(values < 0.0)
        ):
            raise ValueError("method losses must be finite nonnegative sample vectors")
        output[method] = values
    return output


def _relative_reduction(primary: np.ndarray, comparator: np.ndarray) -> float | None:
    denominator = float(np.mean(comparator))
    if denominator <= 0.0:
        return None
    return 1.0 - float(np.mean(primary)) / denominator


def _favorable_count(difference: np.ndarray) -> int:
    return int(np.count_nonzero(difference < -DIFFERENCE_TOLERANCE))


def _reduction_at_least(record: Mapping[str, object], threshold: float) -> bool:
    reduction = record.get("relative_reduction")
    return isinstance(reduction, float) and reduction >= threshold


def _comparison_record(
    primary: np.ndarray, comparator: np.ndarray
) -> dict[str, object]:
    difference = primary - comparator
    return {
        "primary_mean_loss": float(primary.mean()),
        "comparator_mean_loss": float(comparator.mean()),
        "mean_difference": float(difference.mean()),
        "relative_reduction": _relative_reduction(primary, comparator),
        "favorable_samples": _favorable_count(difference),
        "ties": int(np.count_nonzero(np.abs(difference) <= DIFFERENCE_TOLERANCE)),
    }


def pilot_promotion_gate(
    losses: Mapping[str, np.ndarray], strongest_benchmark: str
) -> dict[str, object]:
    """Apply the frozen seven-sample promotion rule."""

    values = _validated_losses(losses, 7)
    if strongest_benchmark not in BENCHMARK_TIE_ORDER:
        raise ValueError("strongest benchmark must be calibration-frozen")
    primary = values["primary"]
    strongest = strongest_benchmark
    destroyed = values["destroyed_links"]
    comparisons = {
        method: _comparison_record(primary, values[method])
        for method in (*CLASSICAL_METHODS, "destroyed_links", "independence")
    }
    checks = {
        "primary_mean_below_each_classical": all(
            primary.mean() < values[method].mean() for method in CLASSICAL_METHODS
        ),
        "primary_mean_below_independence": float(primary.mean())
        < float(values["independence"].mean()),
        "primary_mean_below_destroyed_links": float(primary.mean())
        < float(destroyed.mean()),
        "at_least_five_of_seven_favorable_vs_independence": _favorable_count(
            primary - values["independence"]
        )
        >= 5,
        "at_least_five_of_seven_favorable_vs_donor_stratified_ridge_poisson": (
            _favorable_count(primary - values["donor_stratified_ridge_poisson"]) >= 5
        ),
        "at_least_five_of_seven_favorable_vs_strongest_benchmark": (
            _favorable_count(primary - values[strongest]) >= 5
        ),
        "at_least_five_of_seven_favorable_vs_destroyed_links": _favorable_count(
            primary - destroyed
        )
        >= 5,
        "independence_relative_reduction_at_least_five_percent": (
            _reduction_at_least(comparisons["independence"], 0.05)
        ),
        "donor_stratified_ridge_poisson_relative_reduction_at_least_five_percent": (
            _reduction_at_least(comparisons["donor_stratified_ridge_poisson"], 0.05)
        ),
        "destroyed_link_relative_reduction_at_least_five_percent": (
            _reduction_at_least(comparisons["destroyed_links"], 0.05)
        ),
    }
    return {
        "strongest_benchmark": strongest,
        "comparisons": comparisons,
        "checks": checks,
        "passes": all(checks.values()),
    }


def held_confirmation_gate(
    losses: Mapping[str, np.ndarray],
    conditions: Sequence[str],
    strongest_benchmark: str,
) -> dict[str, object]:
    """Apply the frozen thirteen-sample held-batch confirmation rule."""

    values = _validated_losses(losses, 13)
    labels = tuple(conditions)
    if labels != HELD_CONDITIONS:
        raise ValueError("held conditions must follow the frozen held-panel order")
    if strongest_benchmark not in BENCHMARK_TIE_ORDER:
        raise ValueError("strongest benchmark must be calibration-frozen")
    primary = values["primary"]
    strongest = strongest_benchmark
    destroyed = values["destroyed_links"]
    methods = (*CLASSICAL_METHODS, "destroyed_links", "independence")
    differences = {method: primary - values[method] for method in methods}
    intervals = paired_bootstrap_intervals(differences, labels)
    comparisons = {}
    for method in methods:
        comparisons[method] = _comparison_record(primary, values[method])
        comparisons[method]["paired_sample_bootstrap_95_interval"] = intervals[method]
        comparisons[method]["exact_sign_test"] = exact_one_sided_sign_test(
            differences[method]
        )

    def inferential_checks(method: str, prefix: str) -> dict[str, bool]:
        sign = comparisons[method]["exact_sign_test"]
        if not isinstance(sign, Mapping):
            raise AssertionError("sign-test record is absent")
        return {
            f"{prefix}_bootstrap_upper_below_zero": intervals[method][1] < 0.0,
            f"at_least_ten_of_thirteen_favorable_vs_{prefix}": (
                _favorable_count(differences[method]) >= 10
            ),
            f"{prefix}_one_sided_sign_probability_at_most_0_05": (
                float(sign["one_sided_probability"]) <= 0.05
            ),
        }

    checks = {
        "primary_mean_below_each_classical": all(
            primary.mean() < values[method].mean() for method in CLASSICAL_METHODS
        ),
        "primary_mean_below_independence": float(primary.mean())
        < float(values["independence"].mean()),
        "primary_mean_below_destroyed_links": float(primary.mean())
        < float(destroyed.mean()),
        "independence_relative_reduction_at_least_five_percent": (
            _reduction_at_least(comparisons["independence"], 0.05)
        ),
        "donor_stratified_ridge_poisson_relative_reduction_at_least_five_percent": (
            _reduction_at_least(comparisons["donor_stratified_ridge_poisson"], 0.05)
        ),
        "destroyed_link_relative_reduction_at_least_five_percent": (
            _reduction_at_least(comparisons["destroyed_links"], 0.05)
        ),
        "destroyed_link_bootstrap_upper_below_zero": intervals["destroyed_links"][1]
        < 0.0,
        **inferential_checks("independence", "independence"),
        **inferential_checks(
            "donor_stratified_ridge_poisson", "donor_stratified_ridge_poisson"
        ),
        **inferential_checks(strongest, "strongest_benchmark"),
    }
    return {
        "strongest_benchmark": strongest,
        "comparisons": comparisons,
        "checks": checks,
        "bootstrap": {
            "draws": BOOTSTRAPS,
            "seed": BOOTSTRAP_SEED,
            "stratification": list(DEPOSITED_CONDITIONS),
            "generator": "numpy.random.default_rng/PCG64",
            "same_resample_indices_for_every_comparator": True,
            "within_stratum_donor_order": "input/preflight order",
        },
        "passes": all(checks.values()),
    }


def serialize_models(models: Mapping[str, object]) -> dict[str, object]:
    """Serialize fitted source fields without retaining source tables."""

    config = models.get("configuration")
    if not isinstance(config, PrimaryConfig):
        raise TypeError("models lack a PrimaryConfig")
    output = {
        "configuration": asdict(config),
        "context_order": list(models["context_order"]),
        "comparator_alphas": dict(models["comparator_alphas"]),
        "strongest_benchmark": models["strongest_benchmark"],
        "primary_field": np.asarray(models["primary_field"]).tolist(),
        "primary_fit_certificate": dict(models["primary_fit_certificate"]),
        "destroyed_field": np.asarray(models["destroyed_field"]).tolist(),
        "destroyed_fit_certificate": dict(models["destroyed_fit_certificate"]),
        "donor_stratified_ridge_poisson_field": np.asarray(
            models["donor_stratified_ridge_poisson_field"]
        ).tolist(),
        "donor_stratified_ridge_poisson_certificate": dict(
            models["donor_stratified_ridge_poisson_certificate"]
        ),
        "bias_reduced_context_poisson_field": np.asarray(
            models["bias_reduced_context_poisson_field"]
        ).tolist(),
        "context_signed_deviance_field": np.asarray(
            models["context_signed_deviance_field"]
        ).tolist(),
        "profiled_poisson_status": dict(models["profiled_poisson_status"]),
    }
    profiled = models.get("profiled_poisson_field")
    output["profiled_poisson_field"] = (
        None if profiled is None else np.asarray(profiled).tolist()
    )
    return output


def deserialize_models(payload: Mapping[str, object]) -> dict[str, object]:
    """Restore prediction-only fields from a calibration artifact."""

    configuration = payload.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("serialized configuration is absent")
    config = PrimaryConfig(**configuration)
    if config not in CONFIGURATIONS:
        raise ValueError("serialized configuration is outside the frozen grid")
    context_order = tuple(payload.get("context_order", ()))
    if context_order != CONTEXTS:
        raise ValueError("serialized context order differs")
    alphas = payload.get("comparator_alphas")
    if not isinstance(alphas, Mapping) or set(alphas) != set(CLASSICAL_METHODS):
        raise ValueError("serialized comparator transports are absent")
    if any(
        not isinstance(alphas[method], (int, float))
        or isinstance(alphas[method], bool)
        or not math.isfinite(float(alphas[method]))
        or float(alphas[method]) not in COMPARATOR_TRANSPORT_GRID
        for method in CLASSICAL_METHODS
    ):
        raise ValueError("serialized comparator transports differ from the frozen grid")
    strongest = payload.get("strongest_benchmark")
    if strongest not in BENCHMARK_TIE_ORDER:
        raise ValueError("serialized strongest benchmark is absent")

    shape = (len(CONTEXTS), len(RNA_MARKERS), len(ADT_MARKERS))

    def field(name: str) -> np.ndarray:
        values = np.asarray(payload.get(name), dtype=float)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"serialized {name} differs from the frozen axes")
        return values

    profiled_raw = payload.get("profiled_poisson_field")
    profiled = None
    if profiled_raw is not None:
        profiled = np.asarray(profiled_raw, dtype=float)
        if profiled.shape != shape or not np.isfinite(profiled).all():
            raise ValueError("serialized profiled Poisson field is invalid")
    status = payload.get("profiled_poisson_status")
    if not isinstance(status, Mapping):
        raise ValueError("serialized profiled Poisson status is absent")
    return {
        "configuration": config,
        "context_order": context_order,
        "comparator_alphas": {
            method: float(alphas[method]) for method in CLASSICAL_METHODS
        },
        "strongest_benchmark": strongest,
        "primary_field": field("primary_field"),
        "primary_fit_certificate": dict(payload.get("primary_fit_certificate", {})),
        "destroyed_field": field("destroyed_field"),
        "destroyed_fit_certificate": dict(payload.get("destroyed_fit_certificate", {})),
        "donor_stratified_ridge_poisson_field": field(
            "donor_stratified_ridge_poisson_field"
        ),
        "donor_stratified_ridge_poisson_certificate": dict(
            payload.get("donor_stratified_ridge_poisson_certificate", {})
        ),
        "bias_reduced_context_poisson_field": field(
            "bias_reduced_context_poisson_field"
        ),
        "context_signed_deviance_field": field("context_signed_deviance_field"),
        "profiled_poisson_field": profiled,
        "profiled_poisson_status": dict(status),
    }


def _destroyed_panel(
    tables: np.ndarray, destroyed_tables: np.ndarray, label: str
) -> tuple[np.ndarray, np.ndarray]:
    real = _validated_panel(tables, f"{label} tables")
    destroyed = _validated_panel(destroyed_tables, f"{label} destroyed tables")
    if real.shape != destroyed.shape or any(
        not np.array_equal(left, right)
        for left, right in zip(_margins(real), _margins(destroyed))
    ):
        raise ValueError(f"{label} destroyed links must preserve every fixed margin")
    return real, destroyed


def _configuration_loss_payload(
    losses: Mapping[PrimaryConfig, np.ndarray],
) -> list[dict[str, object]]:
    output = []
    for config in CONFIGURATIONS:
        values = np.asarray(losses[config], dtype=float)
        complete = bool(np.isfinite(values).all())
        output.append(
            {
                "configuration": asdict(config),
                "sample_losses": [
                    float(value) if np.isfinite(value) else None for value in values
                ],
                "mean_loss": float(values.mean()) if complete else None,
                "complete": complete,
            }
        )
    return output


def _comparator_loss_payload(
    losses: Mapping[str, Mapping[float, np.ndarray]],
) -> dict[str, list[dict[str, object]]]:
    return {
        method: [
            {
                "transport_multiplier": alpha,
                "sample_losses": [float(value) for value in losses[method][alpha]],
                "mean_loss": float(np.mean(losses[method][alpha])),
            }
            for alpha in COMPARATOR_TRANSPORT_GRID
        ]
        for method in CLASSICAL_METHODS
    }


def select_source(
    calibration_tables: np.ndarray,
    calibration_destroyed: np.ndarray,
    calibration_contexts: Sequence[str],
    pilot_tables: np.ndarray,
    pilot_destroyed: np.ndarray,
    pilot_contexts: Sequence[str],
) -> dict[str, object]:
    """Select on calibration, evaluate seven pilot samples, and freeze promotion."""

    calibration, calibration_null = _destroyed_panel(
        calibration_tables, calibration_destroyed, "calibration"
    )
    pilot, _ = _destroyed_panel(pilot_tables, pilot_destroyed, "pilot")
    if len(pilot) != 7:
        raise ValueError("pilot must contain exactly seven samples")
    calibration_labels = _validated_contexts(calibration_contexts, len(calibration))
    pilot_labels = _validated_contexts(pilot_contexts, len(pilot), require_both=False)
    selected, primary_losses = select_primary_configuration(
        calibration, calibration_labels, expected_sample_count=9
    )
    alphas, comparator_losses, independence_losses = select_comparator_alphas(
        calibration, calibration_labels, expected_sample_count=9
    )
    strongest = strongest_benchmark_from_calibration(
        alphas, comparator_losses, independence_losses
    )
    models = fit_models(
        calibration,
        calibration_null,
        calibration_labels,
        selected,
        alphas,
        strongest_benchmark=strongest,
    )
    pilot_losses = panel_losses(models, pilot, pilot_labels)
    promotion = pilot_promotion_gate(pilot_losses, strongest)
    return {
        "schema": "gse252762-celiac-source-selection/2.0",
        "status": "PROMOTED" if promotion["passes"] else "NOT_PROMOTED",
        "selected_configuration": asdict(selected),
        "selected_comparator_alphas": alphas,
        "strongest_benchmark": strongest,
        "primary_calibration_loso": _configuration_loss_payload(primary_losses),
        "comparator_calibration_loso": _comparator_loss_payload(comparator_losses),
        "independence_calibration_loso": {
            "sample_losses": independence_losses.tolist(),
            "mean_loss": float(independence_losses.mean()),
        },
        "calibration_models": serialize_models(models),
        "pilot_sample_losses": {
            method: values.tolist() for method, values in pilot_losses.items()
        },
        "pilot_promotion_gate": promotion,
    }


def predict_from_source(
    all_source_tables: np.ndarray,
    all_source_destroyed: np.ndarray,
    all_source_contexts: Sequence[str],
    selection: Mapping[str, object],
    held_row_margins: np.ndarray,
    held_column_margins: np.ndarray,
    held_contexts: Sequence[str],
    *,
    return_fit_report: bool = False,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], dict[str, object]]:
    """Refit the frozen selection on all promoted source samples and predict held."""

    if selection.get("status") != "PROMOTED":
        raise ValueError("held prediction requires a promoted source selection")
    configuration = selection.get("selected_configuration")
    alphas = selection.get("selected_comparator_alphas")
    strongest = selection.get("strongest_benchmark")
    if not isinstance(configuration, Mapping) or not isinstance(alphas, Mapping):
        raise ValueError("source selection is missing its frozen estimators")
    config = PrimaryConfig(**configuration)
    if strongest not in BENCHMARK_TIE_ORDER:
        raise ValueError("source selection lacks its calibration-frozen benchmark")
    source, destroyed = _destroyed_panel(
        all_source_tables, all_source_destroyed, "all-source"
    )
    if len(source) != 16:
        raise ValueError("all-source refit must contain exactly sixteen samples")
    source_labels = _validated_contexts(all_source_contexts, len(source))
    models = fit_models(
        source,
        destroyed,
        source_labels,
        config,
        {method: float(alphas[method]) for method in CLASSICAL_METHODS},
        strongest_benchmark=str(strongest),
    )
    rows = np.asarray(held_row_margins)
    columns = np.asarray(held_column_margins)
    labels = tuple(held_contexts)
    if rows.shape != (13, 9, 9, 2) or columns.shape != rows.shape or len(labels) != 13:
        raise ValueError("held margins and contexts must contain thirteen samples")
    _validated_contexts(labels, 13)
    output: dict[str, np.ndarray] = {}
    for sample, context in enumerate(labels):
        prediction = predict_models_at_margins(
            models, rows[sample], columns[sample], context
        )
        if sample == 0:
            output = {
                method: np.empty((13, 9, 9, 2, 2), dtype=float) for method in prediction
            }
        elif set(prediction) != set(output):
            raise AssertionError("held prediction availability changed across samples")
        for method, tables in prediction.items():
            output[method][sample] = tables
    if not return_fit_report:
        return output
    return output, {
        "configuration": asdict(config),
        "comparator_alphas": dict(models["comparator_alphas"]),
        "strongest_benchmark": models["strongest_benchmark"],
        "primary_fit_certificate": dict(models["primary_fit_certificate"]),
        "destroyed_fit_certificate": dict(models["destroyed_fit_certificate"]),
        "donor_stratified_ridge_poisson_certificate": dict(
            models["donor_stratified_ridge_poisson_certificate"]
        ),
        "profiled_poisson_status": dict(models["profiled_poisson_status"]),
    }


def score_held(
    truth: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    conditions: Sequence[str],
    strongest_benchmark: str,
) -> dict[str, object]:
    """Score sealed held truth and return JSON-serializable confirmatory metrics."""

    observed = _validated_panel(truth, "held truth")
    if len(observed) != 13:
        raise ValueError("held truth must contain thirteen samples")
    losses = {}
    for method, raw in predictions.items():
        estimate = np.asarray(raw, dtype=float)
        if estimate.shape != observed.shape:
            raise ValueError("every held prediction must match held truth")
        losses[method] = np.asarray(
            [
                sample_loss(observed[sample], estimate[sample])
                for sample in range(len(observed))
            ]
        )
    gate = held_confirmation_gate(losses, conditions, strongest_benchmark)
    return {
        "schema": "gse252762-celiac-held-score/2.0",
        "sample_losses": {method: values.tolist() for method, values in losses.items()},
        "mean_losses": {
            method: float(values.mean()) for method, values in losses.items()
        },
        "strongest_benchmark": strongest_benchmark,
        "confirmation_gate": gate,
        "passes": bool(gate["passes"]),
    }
