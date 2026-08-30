"""Pure numerical helpers for the frozen GSE217494 heart confirmation.

This module contains no acquisition or filesystem code.  Every function acts on
arrays already reduced from a caller-supplied training or recipient panel.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from itertools import product
from typing import Iterable, Mapping, Sequence

import numpy as np

from mapreg.heterogeneity_adaptive_coupling import (
    expected_binary_table_from_log_odds,
)


CELL_BUDGET = 512
ADT_HIGH_COUNT = 256
CELL_SALT = "GSE217494-CELL-v1"
ADT_TIE_SALT = "GSE217494-ADT-TIE-v1"
DESTROY_SALT = "GSE217494-DESTROY-v1"
ETIOLOGIES = ("Donor", "AMI", "ICM", "NICM")
KNN_NEIGHBORS = 3
BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 21_749_401
NEIGHBOR_PERMUTATIONS = 10_000
NEIGHBOR_SEED = 21_749_402
MINIMUM_MARKERS = 9
MAXIMUM_MARKERS = 12

MANDATORY_COMPARATORS = (
    "pooled_fixed_interaction_poisson",
    "etiology_specific_fixed_interaction_poisson",
    "strongest_remaining_classical_comparator",
    "destroyed_links",
)
CLASSICAL_COMPARATORS = (
    "standardized_fixed_margin_pearson",
    "exact_common_effect_conditional_field",
    "fixed_margin_independence",
)
FROZEN_MODULES = {
    "endothelial": ("PECAM1", "CDH5", "KDR", "ENG", "TEK"),
    "fibroblast_fibrosis": (
        "FAP",
        "LRRC15",
        "PDGFRA",
        "PDGFRB",
        "THY1",
        "CDH11",
    ),
    "myeloid": (
        "CD14",
        "FCGR1A",
        "FCGR2A",
        "FCGR3A",
        "CSF1R",
        "MRC1",
        "FOLR2",
    ),
}


class MarkerSupportRefusal(ValueError):
    """The frozen marker rule leaves too little informative support."""


class NoCompleteConfigurationError(ValueError):
    """No candidate configuration completed every source fold."""


@dataclass(frozen=True)
class FoldMarkerSelection:
    """Marker axis selected from one training fold only."""

    indices: tuple[int, ...]
    symbols: tuple[str, ...]
    eligible_count: int
    minimum_balance: tuple[int, ...]
    median_balance: tuple[float, ...]


@dataclass(frozen=True)
class MarkerGraph:
    """A symbol-tied three-nearest-neighbor union graph."""

    symbols: tuple[str, ...]
    neighbors: tuple[tuple[str, ...], ...]
    adjacency: np.ndarray
    laplacian: np.ndarray


@dataclass(frozen=True, order=True)
class ConditionalFieldConfig:
    """One frozen structured-field cross-validation configuration."""

    donor_deviation_penalty: float
    coefficient_ridge_penalty: float
    graph_penalty: float
    transport_multiplier: float


@dataclass(frozen=True)
class ClassicalComparatorSelection:
    """Strongest eligible non-Poisson comparator and availability record."""

    selected: str
    eligible: tuple[str, ...]
    ineligible: tuple[str, ...]
    mean_losses: tuple[tuple[str, float], ...]


def _digest(*parts: str) -> bytes:
    return hashlib.sha256("|".join(parts).encode("utf-8")).digest()


def _symbols(values: Sequence[str], label: str) -> tuple[str, ...]:
    output = tuple(values)
    if not output or any(not isinstance(value, str) or not value for value in output):
        raise ValueError(f"{label} must contain nonempty strings")
    if len(set(output)) != len(output):
        raise ValueError(f"{label} must be unique")
    return output


def _barcodes(values: Sequence[str], expected: int | None = None) -> tuple[str, ...]:
    output = _symbols(values, "barcodes")
    if expected is not None and len(output) != expected:
        raise ValueError(f"barcodes must contain exactly {expected} entries")
    return output


def _nonnegative_integer_array(
    values: np.ndarray, label: str, *, ndim: int | None = None
) -> np.ndarray:
    numeric = np.asarray(values)
    floating = np.asarray(numeric, dtype=float)
    if ndim is not None and numeric.ndim != ndim:
        raise ValueError(f"{label} must have {ndim} dimensions")
    if (
        not np.isfinite(floating).all()
        or np.any(floating < 0.0)
        or not np.array_equal(floating, np.rint(floating))
        or np.any(floating > np.iinfo(np.int64).max)
    ):
        raise ValueError(f"{label} must contain finite nonnegative integer counts")
    return floating.astype(np.int64)


def selected_cell_indices(
    barcodes: Sequence[str], sample: str, *, count: int = CELL_BUDGET
) -> np.ndarray:
    """Return the salted Cell Ranger barcode selection in frozen rank order."""

    axis = _barcodes(barcodes)
    if not isinstance(sample, str) or not sample:
        raise ValueError("sample must be a nonempty string")
    requested = int(count)
    if requested < 1 or requested > len(axis):
        raise ValueError("cell selection count must be within the barcode axis")
    order = sorted(
        range(len(axis)),
        key=lambda index: (
            _digest(CELL_SALT, sample, axis[index]),
            axis[index],
        ),
    )
    return np.asarray(order[:requested], dtype=np.int64)


def selected_cell_barcodes(
    barcodes: Sequence[str], sample: str, *, count: int = CELL_BUDGET
) -> tuple[str, ...]:
    """Return selected barcode values rather than their source-axis indices."""

    axis = tuple(barcodes)
    return tuple(
        axis[index] for index in selected_cell_indices(axis, sample, count=count)
    )


def adt_high_states(
    counts: np.ndarray,
    barcodes: Sequence[str],
    sample: str,
    protein_symbols: Sequence[str],
) -> np.ndarray:
    """Assign exactly 256 high cells per protein using the frozen ordering."""

    values = _nonnegative_integer_array(counts, "ADT counts", ndim=2)
    cells = _barcodes(barcodes, CELL_BUDGET)
    proteins = _symbols(protein_symbols, "protein_symbols")
    if values.shape != (CELL_BUDGET, len(proteins)):
        raise ValueError("ADT counts must be cell by protein on the supplied axes")
    if not isinstance(sample, str) or not sample:
        raise ValueError("sample must be a nonempty string")
    output = np.zeros(values.shape, dtype=np.uint8)
    for protein, symbol in enumerate(proteins):
        order = sorted(
            range(CELL_BUDGET),
            key=lambda cell: (
                -int(values[cell, protein]),
                _digest(ADT_TIE_SALT, sample, symbol, cells[cell]),
                cells[cell],
            ),
        )
        output[np.asarray(order[:ADT_HIGH_COUNT], dtype=np.int64), protein] = 1
    if not np.all(output.sum(axis=0) == ADT_HIGH_COUNT):
        raise AssertionError("ADT high-state construction changed its fixed margin")
    return output


def destroy_adt_vectors(
    values: np.ndarray, barcodes: Sequence[str], sample: str
) -> np.ndarray:
    """Shift complete cell-level ADT vectors by 256 salted positions."""

    array = np.asarray(values)
    cells = _barcodes(barcodes, CELL_BUDGET)
    if array.ndim < 1 or array.shape[0] != CELL_BUDGET:
        raise ValueError("ADT vectors must have the 512-cell axis first")
    if not isinstance(sample, str) or not sample:
        raise ValueError("sample must be a nonempty string")
    order = np.asarray(
        sorted(
            range(CELL_BUDGET),
            key=lambda cell: (
                _digest(DESTROY_SALT, sample, cells[cell]),
                cells[cell],
            ),
        ),
        dtype=np.int64,
    )
    output = np.empty_like(array)
    output[order] = array[np.roll(order, -ADT_HIGH_COUNT)]
    return output


def joint_binary_tables(rna_states: np.ndarray, adt_states: np.ndarray) -> np.ndarray:
    """Build RNA-negative/positive by ADT-low/high ordered-pair tables."""

    rna = np.asarray(rna_states)
    adt = np.asarray(adt_states)
    if (
        rna.ndim != 2
        or adt.ndim != 2
        or rna.shape[0] != CELL_BUDGET
        or adt.shape[0] != CELL_BUDGET
    ):
        raise ValueError("state matrices must be 512-cell by marker arrays")
    if not np.all((rna == 0) | (rna == 1)) or not np.all((adt == 0) | (adt == 1)):
        raise ValueError("state matrices must be binary")
    first = rna.astype(np.int64)
    second = adt.astype(np.int64)
    n11 = first.T @ second
    rna_positive = first.sum(axis=0)[:, None]
    adt_high = second.sum(axis=0)[None, :]
    output = np.empty((first.shape[1], second.shape[1], 2, 2), dtype=np.int64)
    output[:, :, 1, 1] = n11
    output[:, :, 1, 0] = rna_positive - n11
    output[:, :, 0, 1] = adt_high - n11
    output[:, :, 0, 0] = CELL_BUDGET - rna_positive - adt_high + n11
    return output


def rna_detection_profile(counts: np.ndarray) -> np.ndarray:
    """Return the positive-UMI fraction for every RNA marker."""

    values = _nonnegative_integer_array(counts, "RNA counts", ndim=2)
    if values.shape[0] != CELL_BUDGET:
        raise ValueError("RNA counts must contain exactly 512 selected cells")
    return np.mean(values > 0, axis=0)


def adt_mean_profile(
    all_adt_counts: np.ndarray, marker_indices: Sequence[int] | None = None
) -> np.ndarray:
    """Return mean library-normalized ADT profiles using all proteins as denominator."""

    values = _nonnegative_integer_array(all_adt_counts, "ADT counts", ndim=2)
    if values.shape[0] != CELL_BUDGET or values.shape[1] < 1:
        raise ValueError("ADT counts must be 512-cell by protein")
    if marker_indices is None:
        indices = np.arange(values.shape[1], dtype=np.int64)
    else:
        indices = np.asarray(tuple(marker_indices), dtype=np.int64)
        if (
            indices.ndim != 1
            or len(indices) < 1
            or np.any(indices < 0)
            or np.any(indices >= values.shape[1])
            or len(set(indices.tolist())) != len(indices)
        ):
            raise ValueError("marker_indices must be unique positions on the ADT axis")
    denominator = np.maximum(1, values.sum(axis=1)).astype(float)
    normalized = 10_000.0 * values[:, indices] / denominator[:, None]
    return np.log1p(normalized).mean(axis=0)


def informative_fixed_margin_support(tables: np.ndarray) -> np.ndarray:
    """Identify tables whose fixed margins permit more than one table."""

    values = _nonnegative_integer_array(tables, "tables")
    if values.ndim < 2 or values.shape[-2:] != (2, 2):
        raise ValueError("tables must end in a 2 by 2 axis")
    total = values.sum(axis=(-2, -1))
    rna_positive = values[..., 1, :].sum(axis=-1)
    adt_high = values[..., :, 1].sum(axis=-1)
    lower = np.maximum(0, rna_positive + adt_high - total)
    upper = np.minimum(rna_positive, adt_high)
    return upper > lower


def select_fold_markers(
    symbols: Sequence[str],
    rna_positive_counts: np.ndarray,
    raw_adt_counts: np.ndarray,
    rna_profiles: np.ndarray,
    adt_profiles: np.ndarray,
    candidate_tables: np.ndarray,
    *,
    minimum_markers: int = MINIMUM_MARKERS,
    maximum_markers: int = MAXIMUM_MARKERS,
) -> FoldMarkerSelection:
    """Apply the complete marker rule to caller-supplied training hearts only."""

    axis = _symbols(symbols, "symbols")
    positives = _nonnegative_integer_array(
        rna_positive_counts, "RNA-positive counts", ndim=2
    )
    adt = _nonnegative_integer_array(raw_adt_counts, "raw ADT counts", ndim=3)
    rna = np.asarray(rna_profiles, dtype=float)
    protein = np.asarray(adt_profiles, dtype=float)
    tables = _nonnegative_integer_array(candidate_tables, "candidate tables")
    donors = positives.shape[0]
    markers = len(axis)
    if donors < 2 or positives.shape[1] != markers:
        raise ValueError("RNA-positive counts must be training-heart by candidate")
    if adt.shape != (donors, CELL_BUDGET, markers):
        raise ValueError("raw ADT counts must be training-heart by cell by candidate")
    if rna.shape != (donors, markers) or protein.shape != rna.shape:
        raise ValueError("profile arrays must be training-heart by candidate")
    if tables.shape != (donors, markers, markers, 2, 2):
        raise ValueError("candidate tables must span training hearts and ordered pairs")
    if np.any(positives > CELL_BUDGET):
        raise ValueError("RNA-positive counts exceed the selected cell budget")
    if not np.array_equal(rna, positives / CELL_BUDGET):
        raise ValueError("RNA profiles differ from the positive-count fractions")
    if np.any(tables.sum(axis=(-2, -1)) != CELL_BUDGET):
        raise ValueError("every candidate table must contain exactly 512 cells")
    table_rna_positive = tables[..., 1, :].sum(axis=-1)
    if not np.array_equal(
        table_rna_positive,
        np.broadcast_to(positives[:, :, None], table_rna_positive.shape),
    ):
        raise ValueError("candidate-table RNA margins differ from positive counts")
    if np.any(tables[..., :, 1].sum(axis=-1) != ADT_HIGH_COUNT):
        raise ValueError(
            "candidate-table ADT margins differ from the 256-cell high state"
        )
    rna_supported = np.all((positives >= 16) & (positives <= 496), axis=0)
    adt_supported = np.ones(markers, dtype=bool)
    for donor, marker in product(range(donors), range(markers)):
        maximum_frequency = int(
            np.unique(adt[donor, :, marker], return_counts=True)[1].max()
        )
        adt_supported[marker] &= CELL_BUDGET - maximum_frequency >= 16
    finite_profiles = np.isfinite(rna).all(axis=0) & np.isfinite(protein).all(axis=0)
    varying_profiles = (np.ptp(rna, axis=0) > 0.0) & (np.ptp(protein, axis=0) > 0.0)
    eligible = rna_supported & adt_supported & finite_profiles & varying_profiles
    balance = np.minimum(positives, CELL_BUDGET - positives)
    minimum_balance = balance.min(axis=0)
    median_balance = np.median(balance, axis=0)
    ranked = sorted(
        np.flatnonzero(eligible).tolist(),
        key=lambda marker: (
            -int(minimum_balance[marker]),
            -float(median_balance[marker]),
            axis[marker],
        ),
    )
    minimum = int(minimum_markers)
    maximum = int(maximum_markers)
    if minimum < 1 or maximum < minimum:
        raise ValueError("marker limits must be positive and ordered")
    if len(ranked) < minimum:
        raise MarkerSupportRefusal(
            f"fewer than {minimum} markers satisfy the training support rule"
        )
    selected = ranked[:maximum]
    selected_tables = tables[:, selected][:, :, selected]
    if not np.all(informative_fixed_margin_support(selected_tables)):
        raise MarkerSupportRefusal(
            "a selected ordered pair has degenerate training-heart support"
        )
    return FoldMarkerSelection(
        indices=tuple(selected),
        symbols=tuple(axis[index] for index in selected),
        eligible_count=len(ranked),
        minimum_balance=tuple(int(minimum_balance[index]) for index in selected),
        median_balance=tuple(float(median_balance[index]) for index in selected),
    )


def marker_knn_graph(
    profiles: np.ndarray,
    symbols: Sequence[str],
    *,
    neighbors: int = KNN_NEIGHBORS,
) -> MarkerGraph:
    """Build the centered-unit-norm, symbol-tied undirected kNN graph."""

    values = np.asarray(profiles, dtype=float)
    axis = _symbols(symbols, "symbols")
    if values.ndim != 2 or values.shape[1] != len(axis) or values.shape[0] < 2:
        raise ValueError("profiles must be training-heart by marker")
    if not np.isfinite(values).all():
        raise ValueError("profiles must be finite")
    k = int(neighbors)
    if k < 1 or k >= len(axis):
        raise ValueError("neighbors must be positive and smaller than marker count")
    centered = values.T - values.T.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1)
    if np.any(~np.isfinite(norm)) or np.any(norm <= 0.0):
        raise ValueError("every marker profile must have finite nonzero norm")
    unit = centered / norm[:, None]
    selected_neighbors: list[tuple[str, ...]] = []
    edges: set[tuple[int, int]] = set()
    for marker in range(len(axis)):
        candidates = [
            candidate for candidate in range(len(axis)) if candidate != marker
        ]
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                float(np.linalg.norm(unit[marker] - unit[candidate])),
                axis[candidate],
            ),
        )
        chosen = ranked[:k]
        selected_neighbors.append(tuple(axis[candidate] for candidate in chosen))
        edges.update(tuple(sorted((marker, candidate))) for candidate in chosen)
    adjacency = np.zeros((len(axis), len(axis)), dtype=float)
    for left, right in edges:
        adjacency[left, right] = 1.0
        adjacency[right, left] = 1.0
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    mean_diagonal = float(np.mean(np.diag(laplacian)))
    if not np.isfinite(mean_diagonal) or mean_diagonal <= 0.0:
        raise ValueError("kNN union graph has no finite edge scale")
    laplacian /= mean_diagonal
    return MarkerGraph(axis, tuple(selected_neighbors), adjacency, laplacian)


def protein_fast_product_laplacian(
    rna_laplacian: np.ndarray, adt_laplacian: np.ndarray
) -> np.ndarray:
    """Return ``(L_RNA kron I + I kron L_ADT)/2`` with protein fastest."""

    rna = np.asarray(rna_laplacian, dtype=float)
    adt = np.asarray(adt_laplacian, dtype=float)
    for values, label in ((rna, "RNA"), (adt, "ADT")):
        if (
            values.ndim != 2
            or values.shape[0] != values.shape[1]
            or values.shape[0] < 1
            or not np.isfinite(values).all()
            or not np.allclose(values, values.T, atol=1e-12)
            or not np.isclose(np.mean(np.diag(values)), 1.0, atol=1e-12)
        ):
            raise ValueError(
                f"{label} Laplacian must be symmetric and mean-diagonal one"
            )
    output = (
        np.kron(rna, np.eye(adt.shape[0])) + np.kron(np.eye(rna.shape[0]), adt)
    ) / 2.0
    if not np.isclose(np.mean(np.diag(output)), 1.0, atol=1e-12):
        raise AssertionError("product Laplacian normalization changed")
    return output


def module_knn_graph(profiles: np.ndarray, symbols: Sequence[str]) -> MarkerGraph:
    """Build a secondary-module graph with ``min(3, marker_count - 1)`` neighbors."""

    axis = _symbols(symbols, "symbols")
    if len(axis) < 3:
        raise ValueError("a secondary module needs at least three selected markers")
    return marker_knn_graph(profiles, axis, neighbors=min(KNN_NEIGHBORS, len(axis) - 1))


def one_hot_context(
    etiologies: Sequence[str], levels: Sequence[str] = ETIOLOGIES
) -> np.ndarray:
    """Encode etiologies without a privileged reference level."""

    context_levels = _symbols(levels, "levels")
    labels = tuple(etiologies)
    lookup = {value: index for index, value in enumerate(context_levels)}
    if any(value not in lookup for value in labels):
        raise ValueError("an etiology is absent from the frozen context levels")
    output = np.zeros((len(labels), len(context_levels)), dtype=float)
    if labels:
        output[np.arange(len(labels)), [lookup[value] for value in labels]] = 1.0
    return output


def context_log_odds(
    coefficient: np.ndarray,
    etiologies: Sequence[str],
    *,
    transport_multiplier: float = 1.0,
    levels: Sequence[str] = ETIOLOGIES,
) -> np.ndarray:
    """Evaluate frozen etiology fields with recipient donor deviation set to zero."""

    values = np.asarray(coefficient, dtype=float)
    design = one_hot_context(etiologies, levels)
    transport = float(transport_multiplier)
    if values.ndim < 1 or values.shape[0] != design.shape[1]:
        raise ValueError("coefficient must begin with the complete context axis")
    if not np.isfinite(values).all() or not np.isfinite(transport) or transport <= 0.0:
        raise ValueError(
            "coefficient and transport multiplier must be finite and valid"
        )
    return transport * np.tensordot(design, values, axes=(1, 0))


def _validated_margins(
    row_margins: np.ndarray, column_margins: np.ndarray, entity_shape: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    rows = _nonnegative_integer_array(row_margins, "row margins")
    columns = _nonnegative_integer_array(column_margins, "column margins")
    expected = (*entity_shape, 2)
    if rows.shape != expected or columns.shape != expected:
        raise ValueError("margin arrays must match the field and end in length two")
    if np.any(rows.sum(axis=-1) != CELL_BUDGET) or not np.array_equal(
        rows.sum(axis=-1), columns.sum(axis=-1)
    ):
        raise ValueError("every recipient margin must contain exactly 512 cells")
    return rows, columns


def predict_conditional_tables(
    log_odds: np.ndarray,
    row_margins: np.ndarray,
    column_margins: np.ndarray,
) -> np.ndarray:
    """Reconstruct exact conditional expected tables at recipient margins."""

    field = np.asarray(log_odds, dtype=float)
    if not np.isfinite(field).all():
        raise ValueError("log_odds must be finite")
    rows, columns = _validated_margins(row_margins, column_margins, field.shape)
    output = np.empty((*field.shape, 2, 2), dtype=float)
    for index in np.ndindex(field.shape):
        output[index] = expected_binary_table_from_log_odds(
            float(field[index]), rows[index], columns[index]
        )
    if not np.isfinite(output).all() or np.any(output < 0.0):
        raise FloatingPointError("conditional prediction is not finite and nonnegative")
    if not np.allclose(
        output.sum(axis=-1), rows, atol=1e-8, rtol=0.0
    ) or not np.allclose(output.sum(axis=-2), columns, atol=1e-8, rtol=0.0):
        raise FloatingPointError("conditional prediction changed a recipient margin")
    return output


def fit_standardized_pearson(
    tables: np.ndarray,
    etiologies: Sequence[str],
    levels: Sequence[str] = ETIOLOGIES,
) -> np.ndarray:
    """Average fixed-margin standardized Pearson coordinates within etiology."""

    values = _nonnegative_integer_array(tables, "tables")
    if values.ndim < 3 or values.shape[-2:] != (2, 2):
        raise ValueError("tables must have donor first and end in 2 by 2")
    labels = tuple(etiologies)
    design = one_hot_context(labels, levels)
    if values.shape[0] != len(labels):
        raise ValueError("tables and etiologies must share the donor axis")
    if np.any(values.sum(axis=(-2, -1)) != CELL_BUDGET):
        raise ValueError("every source table must contain exactly 512 cells")
    rna_positive = values[..., 1, :].sum(axis=-1).astype(float)
    adt_high = values[..., :, 1].sum(axis=-1).astype(float)
    observed = values[..., 1, 1].astype(float)
    expected = rna_positive * adt_high / CELL_BUDGET
    variance = (
        rna_positive
        * adt_high
        * (CELL_BUDGET - rna_positive)
        * (CELL_BUDGET - adt_high)
        / (CELL_BUDGET**2 * (CELL_BUDGET - 1))
    )
    coordinate = np.divide(
        observed - expected,
        np.sqrt(variance),
        out=np.zeros_like(expected),
        where=variance > 0.0,
    )
    output = np.empty((design.shape[1], *values.shape[1:-2]), dtype=float)
    for context in range(design.shape[1]):
        donors = design[:, context] == 1.0
        if not np.any(donors):
            raise ValueError("every frozen etiology must occur in the training panel")
        output[context] = coordinate[donors].mean(axis=0)
    return output


def predict_standardized_pearson(
    coordinate: np.ndarray,
    row_margins: np.ndarray,
    column_margins: np.ndarray,
    *,
    transport_multiplier: float = 1.0,
) -> np.ndarray:
    """Invert standardized Pearson coordinates at fixed recipient margins."""

    field = np.asarray(coordinate, dtype=float)
    transport = float(transport_multiplier)
    if not np.isfinite(field).all() or not np.isfinite(transport) or transport <= 0.0:
        raise ValueError("coordinate and transport multiplier must be finite and valid")
    rows, columns = _validated_margins(row_margins, column_margins, field.shape)
    rna_positive = rows[..., 1].astype(float)
    adt_high = columns[..., 1].astype(float)
    expected = rna_positive * adt_high / CELL_BUDGET
    variance = (
        rna_positive
        * adt_high
        * (CELL_BUDGET - rna_positive)
        * (CELL_BUDGET - adt_high)
        / (CELL_BUDGET**2 * (CELL_BUDGET - 1))
    )
    lower = np.maximum(0.0, rna_positive + adt_high - CELL_BUDGET)
    upper = np.minimum(rna_positive, adt_high)
    n11 = np.clip(expected + transport * field * np.sqrt(variance), lower, upper)
    output = np.empty((*field.shape, 2, 2), dtype=float)
    output[..., 1, 1] = n11
    output[..., 1, 0] = rna_positive - n11
    output[..., 0, 1] = adt_high - n11
    output[..., 0, 0] = CELL_BUDGET - rna_positive - adt_high + n11
    return output


def entity_deviance(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Return multinomial deviance per cell for every ordered pair."""

    truth = _nonnegative_integer_array(observed, "observed tables")
    estimate = np.asarray(predicted, dtype=float)
    if truth.shape != estimate.shape or truth.ndim < 2 or truth.shape[-2:] != (2, 2):
        raise ValueError("observed and predicted tables must have the same 2 by 2 axes")
    if np.any(truth.sum(axis=(-2, -1)) != CELL_BUDGET):
        raise ValueError("every observed table must contain exactly 512 cells")
    if not np.isfinite(estimate).all() or np.any(estimate < 0.0):
        raise FloatingPointError("predicted tables must be finite and nonnegative")
    if not np.allclose(
        truth.sum(axis=-1), estimate.sum(axis=-1), atol=1e-8, rtol=0.0
    ) or not np.allclose(
        truth.sum(axis=-2), estimate.sum(axis=-2), atol=1e-8, rtol=0.0
    ):
        raise FloatingPointError("prediction changed a recipient margin")
    positive = truth > 0
    if np.any(estimate[positive] <= 0.0):
        raise FloatingPointError("prediction assigns zero mass to an observed cell")
    terms = np.zeros_like(estimate)
    terms[positive] = truth[positive] * np.log(truth[positive] / estimate[positive])
    return 2.0 * terms.sum(axis=(-2, -1)) / CELL_BUDGET


def donor_deviance(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Average ordered-pair deviance for one heart."""

    return float(np.mean(entity_deviance(observed, predicted)))


def panel_deviances(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Return one donor-equal loss for every heart on the first axis."""

    values = entity_deviance(observed, predicted)
    if values.ndim < 1:
        raise ValueError("panel tables must contain a donor axis")
    return values.reshape(values.shape[0], -1).mean(axis=1)


def conditional_field_configurations() -> tuple[ConditionalFieldConfig, ...]:
    """Enumerate the frozen primary grid in protocol tie-break order."""

    return tuple(
        ConditionalFieldConfig(deviation, ridge, graph, transport)
        for deviation, ridge, graph, transport in product(
            (0.3, 3.0),
            (0.1, 1.0),
            (0.0, 0.01, 0.1, 1.0),
            (0.75, 1.0, 1.25),
        )
    )


def leave_one_out_training_indices(donor_count: int) -> tuple[np.ndarray, ...]:
    """Return immutable-by-convention training indices for every validation heart."""

    count = int(donor_count)
    if count < 2:
        raise ValueError("leave-one-out cross-validation needs at least two hearts")
    axis = np.arange(count, dtype=np.int64)
    return tuple(axis[axis != validation] for validation in range(count))


def select_cv_configuration(
    losses: Mapping[ConditionalFieldConfig, np.ndarray],
) -> tuple[ConditionalFieldConfig, np.ndarray]:
    """Select complete donor-equal CV loss, then the protocol tuple tie break."""

    complete: list[tuple[ConditionalFieldConfig, np.ndarray]] = []
    donor_count: int | None = None
    for configuration, values in losses.items():
        if not isinstance(configuration, ConditionalFieldConfig):
            raise TypeError("loss keys must be ConditionalFieldConfig values")
        current = np.asarray(values, dtype=float)
        if current.ndim != 1 or len(current) < 1:
            raise ValueError("each configuration needs one loss per validation heart")
        if donor_count is None:
            donor_count = len(current)
        elif len(current) != donor_count:
            raise ValueError("configuration loss vectors must share the donor axis")
        if np.isfinite(current).all():
            complete.append((configuration, current.copy()))
    if not complete:
        raise NoCompleteConfigurationError(
            "no configuration completed every validation fold"
        )
    return min(complete, key=lambda item: (float(item[1].mean()), item[0]))


def select_strongest_classical(
    losses: Mapping[str, np.ndarray | None],
) -> ClassicalComparatorSelection:
    """Freeze the strongest eligible non-Poisson comparator.

    ``None`` records an explicitly unavailable estimator, such as an exact
    common-effect conditional MLE at a boundary.  The standardized residual
    and independence comparators must remain eligible.
    """

    missing = [name for name in CLASSICAL_COMPARATORS if name not in losses]
    if missing:
        raise ValueError(f"missing classical comparator losses: {', '.join(missing)}")
    means: dict[str, float] = {}
    ineligible: list[str] = []
    length: int | None = None
    for name in CLASSICAL_COMPARATORS:
        supplied = losses[name]
        if supplied is None:
            ineligible.append(name)
            continue
        values = np.asarray(supplied, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("classical comparator losses must be finite vectors")
        if length is None:
            length = len(values)
        elif len(values) != length:
            raise ValueError("classical comparator losses must share the donor axis")
        means[name] = float(values.mean())
    required = {
        "standardized_fixed_margin_pearson",
        "fixed_margin_independence",
    }
    if not required <= means.keys():
        raise ValueError("residual and independence comparators must be eligible")
    selected = min(
        means,
        key=lambda name: (means[name], CLASSICAL_COMPARATORS.index(name)),
    )
    return ClassicalComparatorSelection(
        selected=selected,
        eligible=tuple(name for name in CLASSICAL_COMPARATORS if name in means),
        ineligible=tuple(name for name in CLASSICAL_COMPARATORS if name in ineligible),
        mean_losses=tuple(
            (name, means[name]) for name in CLASSICAL_COMPARATORS if name in means
        ),
    )


def stratified_paired_bootstrap(
    differences: np.ndarray,
    etiologies: Sequence[str],
    *,
    draws: int = BOOTSTRAPS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Bootstrap paired donor losses within the four frozen etiologies."""

    values = np.asarray(differences, dtype=float)
    labels = np.asarray(tuple(etiologies), dtype=object)
    if (
        values.ndim != 1
        or labels.shape != values.shape
        or not np.isfinite(values).all()
    ):
        raise ValueError("bootstrap inputs must be paired finite donor vectors")
    count = int(draws)
    if count < 1:
        raise ValueError("bootstrap draws must be positive")
    strata = []
    for etiology in ETIOLOGIES:
        indices = np.flatnonzero(labels == etiology)
        if len(indices) < 1:
            raise ValueError("every frozen etiology must occur in the bootstrap panel")
        strata.append(indices)
    if any(value not in ETIOLOGIES for value in labels):
        raise ValueError("bootstrap labels contain an unknown etiology")
    generator = np.random.default_rng(int(seed))
    sampled = [
        values[indices[generator.integers(0, len(indices), size=(count, len(indices)))]]
        for indices in strata
    ]
    distribution = np.concatenate(sampled, axis=1).mean(axis=1)
    interval = np.quantile(distribution, (0.025, 0.975), method="linear")
    return {
        "mean": float(values.mean()),
        "interval": (float(interval[0]), float(interval[1])),
        "draws": count,
        "seed": int(seed),
    }


def exact_one_sided_sign_probability(differences: np.ndarray) -> float:
    """Return the binomial tail for strictly favorable nonzero donor signs."""

    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("sign probability requires a finite donor vector")
    nonzero = values != 0.0
    n = int(np.count_nonzero(nonzero))
    favorable = int(np.count_nonzero(values[nonzero] < 0.0))
    return float(sum(math.comb(n, count) for count in range(favorable, n + 1)) / (2**n))


def _validated_gate_panel(
    primary_losses: np.ndarray,
    comparator_losses: Mapping[str, np.ndarray],
    etiologies: Sequence[str],
    expected_counts: Mapping[str, int],
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    primary = np.asarray(primary_losses, dtype=float)
    labels = np.asarray(tuple(etiologies), dtype=object)
    expected_total = sum(expected_counts.values())
    if (
        primary.shape != (expected_total,)
        or labels.shape != primary.shape
        or not np.isfinite(primary).all()
        or np.any(primary < 0.0)
    ):
        raise ValueError("gate inputs differ from the frozen donor panel")
    if any(
        np.count_nonzero(labels == name) != count
        for name, count in expected_counts.items()
    ):
        raise ValueError("gate etiology counts differ from the frozen split")
    if any(value not in expected_counts for value in labels):
        raise ValueError("gate labels contain an unknown etiology")
    missing = [name for name in MANDATORY_COMPARATORS if name not in comparator_losses]
    if missing:
        raise ValueError(f"missing mandatory comparator losses: {', '.join(missing)}")
    comparators = {}
    for name in MANDATORY_COMPARATORS:
        values = np.asarray(comparator_losses[name], dtype=float)
        if (
            values.shape != primary.shape
            or not np.isfinite(values).all()
            or np.any(values < 0.0)
        ):
            raise ValueError(
                "mandatory comparator losses must be paired nonnegative finite vectors"
            )
        if values.mean() <= 0.0:
            raise ValueError("mandatory comparator mean loss must be positive")
        comparators[name] = values
    return primary, comparators, labels


def _basic_comparison(
    primary: np.ndarray, comparator: np.ndarray, labels: np.ndarray
) -> dict[str, object]:
    difference = primary - comparator
    improvement = {
        etiology: float(
            np.mean(comparator[labels == etiology] - primary[labels == etiology])
        )
        for etiology in ETIOLOGIES
    }
    return {
        "primary_mean_deviance": float(primary.mean()),
        "comparator_mean_deviance": float(comparator.mean()),
        "relative_reduction": float(1.0 - primary.mean() / comparator.mean()),
        "mean_difference": float(difference.mean()),
        "favorable_hearts": int(np.count_nonzero(difference < 0.0)),
        "etiology_mean_improvement": improvement,
        "exact_one_sided_sign_probability": exact_one_sided_sign_probability(
            difference
        ),
    }


def evaluate_source_gate(
    primary_losses: np.ndarray,
    comparator_losses: Mapping[str, np.ndarray],
    etiologies: Sequence[str],
    marker_counts: Sequence[int],
    *,
    all_reductions_and_fits_complete: bool,
) -> dict[str, object]:
    """Evaluate the frozen 14-heart promotion gate against all mandatory controls."""

    primary, comparators, labels = _validated_gate_panel(
        primary_losses,
        comparator_losses,
        etiologies,
        {"Donor": 4, "AMI": 2, "ICM": 4, "NICM": 4},
    )
    support_array = _nonnegative_integer_array(
        np.asarray(tuple(marker_counts)), "marker_counts", ndim=1
    )
    support = tuple(int(value) for value in support_array)
    if len(support) != 15:
        raise ValueError("marker_counts must cover 14 folds and the final refit")
    if any(value > MAXIMUM_MARKERS for value in support):
        raise ValueError("marker_counts exceed the frozen marker cap")
    comparisons: dict[str, dict[str, object]] = {}
    for name, comparator in comparators.items():
        summary = _basic_comparison(primary, comparator, labels)
        checks = {
            "primary_mean_strictly_lower": summary["mean_difference"] < 0.0,
            "at_least_ten_of_fourteen_hearts_favorable": summary["favorable_hearts"]
            >= 10,
            "every_etiology_mean_improvement_positive": all(
                value > 0.0 for value in summary["etiology_mean_improvement"].values()
            ),
        }
        summary["checks"] = checks
        summary["passes"] = bool(all(checks.values()))
        comparisons[name] = summary
    prerequisites = {
        "all_source_reductions_and_fits_complete": bool(
            all_reductions_and_fits_complete
        ),
        "at_least_nine_markers_in_every_fold_and_final": bool(
            all(value >= MINIMUM_MARKERS for value in support)
        ),
    }
    return {
        "marker_counts": support,
        "prerequisites": prerequisites,
        "comparisons": comparisons,
        "passes": bool(
            all(prerequisites.values())
            and all(value["passes"] for value in comparisons.values())
        ),
    }


def evaluate_held_gate(
    primary_losses: np.ndarray,
    comparator_losses: Mapping[str, np.ndarray],
    etiologies: Sequence[str],
) -> dict[str, object]:
    """Evaluate the frozen eight-heart confirmation gate and uncertainty."""

    primary, comparators, labels = _validated_gate_panel(
        primary_losses,
        comparator_losses,
        etiologies,
        {"Donor": 2, "AMI": 2, "ICM": 2, "NICM": 2},
    )
    comparisons: dict[str, dict[str, object]] = {}
    for name, comparator in comparators.items():
        summary = _basic_comparison(primary, comparator, labels)
        bootstrap = stratified_paired_bootstrap(primary - comparator, labels)
        checks = {
            "relative_reduction_at_least_five_percent": summary["relative_reduction"]
            >= 0.05,
            "stratified_bootstrap_upper_95_below_zero": bootstrap["interval"][1] < 0.0,
            "at_least_seven_of_eight_hearts_favorable": summary["favorable_hearts"]
            >= 7,
            "every_etiology_mean_improvement_positive": all(
                value > 0.0 for value in summary["etiology_mean_improvement"].values()
            ),
        }
        summary["stratified_paired_bootstrap"] = bootstrap
        summary["checks"] = checks
        summary["passes"] = bool(all(checks.values()))
        comparisons[name] = summary
    return {
        "comparisons": comparisons,
        "passes": bool(all(value["passes"] for value in comparisons.values())),
    }


def evaluable_modules(
    selected_symbols: Sequence[str], *, minimum_members: int = 3
) -> dict[str, tuple[str, ...]]:
    """Return frozen modules with enough source-selected marker support."""

    selected = set(_symbols(selected_symbols, "selected_symbols"))
    minimum = int(minimum_members)
    if minimum < 1:
        raise ValueError("minimum_members must be positive")
    return {
        name: tuple(symbol for symbol in members if symbol in selected)
        for name, members in FROZEN_MODULES.items()
        if sum(symbol in selected for symbol in members) >= minimum
    }


def module_pair_mask(
    marker_symbols: Sequence[str], module_members: Iterable[str]
) -> np.ndarray:
    """Select the complete ordered within-module cross-product."""

    axis = _symbols(marker_symbols, "marker_symbols")
    members = set(module_members)
    unknown = members - set(axis)
    if unknown:
        raise ValueError("module members must occur on the selected marker axis")
    selected = np.asarray([symbol in members for symbol in axis], dtype=bool)
    return selected[:, None] & selected[None, :]


def nearest_neighbor_indices(
    fields: np.ndarray,
    symbols: Sequence[str],
    *,
    neighbors: int = KNN_NEIGHBORS,
) -> np.ndarray:
    """Rank RNA-field rows by Euclidean distance with symbol tie breaks."""

    values = np.asarray(fields, dtype=float)
    axis = _symbols(symbols, "symbols")
    if (
        values.ndim != 3
        or values.shape[1] != len(axis)
        or values.shape[2] < 1
        or not np.isfinite(values).all()
    ):
        raise ValueError("fields must be finite donor by RNA marker by feature arrays")
    k = int(neighbors)
    if k < 1 or k >= len(axis):
        raise ValueError("neighbors must be positive and smaller than marker count")
    output = np.empty((values.shape[0], len(axis), k), dtype=np.int64)
    for donor in range(values.shape[0]):
        for marker in range(len(axis)):
            candidates = [index for index in range(len(axis)) if index != marker]
            ranked = sorted(
                candidates,
                key=lambda candidate: (
                    float(
                        np.linalg.norm(values[donor, marker] - values[donor, candidate])
                    ),
                    axis[candidate],
                ),
            )
            output[donor, marker] = ranked[:k]
    return output


def neighbor_overlap_permutation(
    predicted_fields: np.ndarray,
    observed_fields: np.ndarray,
    symbols: Sequence[str],
    *,
    neighbors: int = KNN_NEIGHBORS,
    permutations: int = NEIGHBOR_PERMUTATIONS,
    seed: int = NEIGHBOR_SEED,
) -> dict[str, object]:
    """Compare directed neighbor sets to a joint cross-map label permutation."""

    if np.asarray(predicted_fields).shape != np.asarray(observed_fields).shape:
        raise ValueError("predicted and observed fields must have the same shape")
    predicted = nearest_neighbor_indices(predicted_fields, symbols, neighbors=neighbors)
    observed = nearest_neighbor_indices(observed_fields, symbols, neighbors=neighbors)
    if predicted.shape != observed.shape:
        raise ValueError(
            "predicted and observed fields must share their donor and marker axes"
        )
    count = int(permutations)
    if count < 1:
        raise ValueError("permutations must be positive")
    donors, markers, k = predicted.shape
    predicted_adjacency = np.zeros((donors, markers, markers), dtype=bool)
    observed_adjacency = np.zeros_like(predicted_adjacency)
    donor_axis = np.arange(donors)[:, None, None]
    marker_axis = np.arange(markers)[None, :, None]
    predicted_adjacency[donor_axis, marker_axis, predicted] = True
    observed_adjacency[donor_axis, marker_axis, observed] = True

    def mean_jaccard(candidate: np.ndarray) -> float:
        intersection = np.count_nonzero(candidate & observed_adjacency, axis=-1)
        union = np.count_nonzero(candidate | observed_adjacency, axis=-1)
        return float(np.mean(intersection / union))

    observed_overlap = mean_jaccard(predicted_adjacency)
    generator = np.random.default_rng(int(seed))
    null = np.empty(count, dtype=float)
    for draw in range(count):
        old_for_new = np.argsort(generator.permutation(markers))
        relabeled = predicted_adjacency[:, old_for_new][:, :, old_for_new]
        null[draw] = mean_jaccard(relabeled)
    interval = np.quantile(null, (0.025, 0.975), method="linear")
    return {
        "mean_top_k_jaccard": observed_overlap,
        "neighbors": k,
        "donors": donors,
        "markers": markers,
        "permutations": count,
        "seed": int(seed),
        "null_mean": float(np.mean(null)),
        "null_interval": (float(interval[0]), float(interval[1])),
        "one_sided_monte_carlo_p": float(
            (1 + np.count_nonzero(null >= observed_overlap)) / (count + 1)
        ),
        "joint_permutation": "one RNA-marker relabeling shared across held donors",
    }


def exact_paired_sign_permutation(differences: np.ndarray) -> dict[str, float | int]:
    """Test a negative mean difference by all paired sign assignments."""

    values = np.asarray(differences, dtype=float)
    if (
        values.ndim != 1
        or len(values) < 1
        or len(values) > 20
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "sign permutation needs one to twenty finite paired differences"
        )
    observed = float(values.mean())
    assignments = 1 << len(values)
    favorable = 0
    for mask in range(assignments):
        signs = np.asarray(
            [-1.0 if mask & (1 << index) else 1.0 for index in range(len(values))]
        )
        favorable += float(np.mean(signs * values)) <= observed + 1e-15
    return {
        "observed_mean_difference": observed,
        "assignments": assignments,
        "one_sided_p": favorable / assignments,
    }


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Adjust one finite family of p-values by Benjamini--Hochberg."""

    values = np.asarray(p_values, dtype=float)
    if (
        values.ndim != 1
        or len(values) < 1
        or not np.isfinite(values).all()
        or np.any((values < 0.0) | (values > 1.0))
    ):
        raise ValueError("p_values must be a nonempty finite probability vector")
    order = np.argsort(values, kind="stable")
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(1.0, adjusted_ranked)
    return adjusted


__all__ = [
    "ADT_HIGH_COUNT",
    "ADT_TIE_SALT",
    "BOOTSTRAPS",
    "BOOTSTRAP_SEED",
    "CELL_BUDGET",
    "CELL_SALT",
    "CLASSICAL_COMPARATORS",
    "ClassicalComparatorSelection",
    "ConditionalFieldConfig",
    "DESTROY_SALT",
    "ETIOLOGIES",
    "FROZEN_MODULES",
    "FoldMarkerSelection",
    "KNN_NEIGHBORS",
    "MANDATORY_COMPARATORS",
    "NEIGHBOR_PERMUTATIONS",
    "NEIGHBOR_SEED",
    "MarkerGraph",
    "adt_high_states",
    "adt_mean_profile",
    "benjamini_hochberg",
    "conditional_field_configurations",
    "context_log_odds",
    "destroy_adt_vectors",
    "donor_deviance",
    "entity_deviance",
    "evaluable_modules",
    "evaluate_held_gate",
    "evaluate_source_gate",
    "exact_one_sided_sign_probability",
    "exact_paired_sign_permutation",
    "fit_standardized_pearson",
    "informative_fixed_margin_support",
    "joint_binary_tables",
    "leave_one_out_training_indices",
    "marker_knn_graph",
    "module_knn_graph",
    "module_pair_mask",
    "nearest_neighbor_indices",
    "neighbor_overlap_permutation",
    "one_hot_context",
    "panel_deviances",
    "predict_conditional_tables",
    "predict_standardized_pearson",
    "protein_fast_product_laplacian",
    "rna_detection_profile",
    "select_cv_configuration",
    "select_fold_markers",
    "select_strongest_classical",
    "selected_cell_barcodes",
    "selected_cell_indices",
    "stratified_paired_bootstrap",
]
