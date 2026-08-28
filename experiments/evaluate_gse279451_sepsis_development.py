"""Outcome-sealed development evaluation for GSE279451.

This evaluator reads only the 19-donor reduced JSON. It performs ordinary
leave-one-development-donor-out model selection, records every candidate loss
against its omitted donor, applies the four prespecified development gates,
and fits the all-development source models only after every gate passes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from itertools import product
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
from typing import Any, Callable

import numpy as np
from scipy.special import gammaln, logsumexp

from experiments import reduce_gse279451_sepsis as reducer
from mapreg.heterogeneity_adaptive_coupling import (
    CouplingEstimationRefusal,
    centered_classical_coordinate,
    fit_structured_conditional_log_odds,
    signed_deviance_coordinate,
    signed_pearson_coordinate,
)
from mapreg.hierarchical_conditional_coupling import (
    fit_hierarchical_conditional_log_odds,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/development/gse279451_sepsis/reduced_development_v1.json"
OUTPUT = ROOT / "results/development/gse279451_sepsis_exact_development.json"
EVALUATION_ATTEMPT = (
    ROOT / "data/development/gse279451_sepsis/evaluation_attempt_v1.json"
)
EVALUATION_REFUSAL = (
    ROOT / "results/development/gse279451_sepsis_evaluation_refusal.json"
)
PROTOCOL = ROOT / "docs/GSE279451_SEPSIS_HELD_DONOR_CONFIRMATION_PROTOCOL_2026-08-28.md"
DESIGNATION = ROOT / "data/confirmation/gse279451_sepsis/candidate_designation_v1.json"
FAMILY_POLICY = ROOT / "data/confirmation/gse279451_sepsis/family_policy_v1.json"

MARKERS = reducer.MARKERS
DEVELOPMENT_DONORS = reducer.DEVELOPMENT_DONORS
HELD_DONORS = reducer.HELD_DONORS
CELL_BUDGET = reducer.CELL_BUDGET
MINIMUM_INFORMATIVE_ENTITIES = 64
NEIGHBOR_GRID = (1, 2, 3)
HETEROGENEITY_GRID = (0.1, 1.0, 10.0)
RIDGE_GRID = (0.01, 0.1)
GRAPH_GRID = (0.1, 0.3, 1.0)
ALPHA_GRID = (0.75, 1.0, 1.25)
CV_GRID = {
    "graph_neighborhood": list(NEIGHBOR_GRID),
    "heterogeneity_penalty": list(HETEROGENEITY_GRID),
    "ridge_penalty": list(RIDGE_GRID),
    "graph_penalty": list(GRAPH_GRID),
    "transport_multiplier": list(ALPHA_GRID),
}
METHODS = (
    "primary",
    "best_residual",
    "destroyed_link",
    "hierarchical_ridge_only",
    "common_effect_graph",
    "common_effect_ridge_only",
    "label_permuted_graph",
    "independence",
)
GATE_COMPARATORS = (
    "best_residual",
    "destroyed_link",
    "hierarchical_ridge_only",
    "common_effect_graph",
)
BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 20260828
MAXIMUM_CONDITION_NUMBER = 1e12
HIERARCHICAL_TOLERANCE = 1e-8
COMMON_TOLERANCE = 1e-9
MAXIMUM_WORKERS = min(8, os.cpu_count() or 1, len(DEVELOPMENT_DONORS))


class GraphConstructionRefusal(RuntimeError):
    """A fold-local marginal graph cannot be constructed as declared."""


class DevelopmentEvaluationRefusal(RuntimeError):
    """The one-shot development evaluation could not produce a valid result."""

    def __init__(self, detail: dict[str, Any]) -> None:
        super().__init__("development evaluation could not complete as declared")
        self.detail = detail


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _transitive_bindings() -> dict[str, str]:
    return {name: _sha256(path) for name, path in reducer.TRANSITIVE_ARTIFACTS.items()}


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(serialized)


def _informative(tables: np.ndarray) -> np.ndarray:
    values = np.asarray(tables)
    rows = values.sum(axis=-1)
    columns = values.sum(axis=-2)
    total = values.sum(axis=(-2, -1))
    lower = np.maximum(0, rows[..., 0] + columns[..., 0] - total)
    upper = np.minimum(rows[..., 0], columns[..., 0])
    return upper > lower


def _validated_reduced(path: Path = INPUT) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"{path.name} contains nonfinite JSON token {token}")
        ),
    )
    if not isinstance(payload, dict):
        raise ValueError("reduced development input must be a JSON object")
    audit = payload.get("access_audit", {})
    if not reducer.DEVELOPMENT_ATTEMPT.is_file():
        raise PermissionError("development acquisition attempt marker is absent")
    acquisition_attempt = reducer._read_json(reducer.DEVELOPMENT_ATTEMPT)
    if (
        payload.get("schema") != "gse279451-sepsis-reduced-development/1.0"
        or payload.get("status") != "NONHELD_REDUCTION_COMPLETE"
        or acquisition_attempt.get("status") != "TERMINAL_DEVELOPMENT_ATTEMPT_STARTED"
        or payload.get("development_attempt_sha256")
        != _sha256(reducer.DEVELOPMENT_ATTEMPT)
        or payload.get("development_donors") != list(DEVELOPMENT_DONORS)
        or payload.get("held_donors") != list(HELD_DONORS)
        or payload.get("markers") != list(MARKERS)
        or payload.get("entity_count") != len(MARKERS) ** 2
        or payload.get("primary_cells_per_donor") != CELL_BUDGET
        or payload.get("cell_selection_salt") != reducer.CELL_SELECTION_SALT
        or payload.get("all_cells_sensitivity_included") is not False
        or not isinstance(audit, dict)
        or audit.get("development_matrix_members_decoded") != len(DEVELOPMENT_DONORS)
        or audit.get("held_matrix_members_opened") != 0
        or audit.get("held_matrix_entries_decoded") != 0
        or audit.get("maximum_concurrent_donor_matrices") != 1
    ):
        raise PermissionError("reduced development input violates the frozen seal")
    source_hash = payload.get("source_manifest_sha256")
    if not isinstance(source_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", source_hash
    ):
        raise ValueError("reduced input lacks its source-manifest SHA-256")
    records = payload.get("donors")
    if not isinstance(records, list) or len(records) != len(DEVELOPMENT_DONORS):
        raise ValueError("reduced input must contain exactly 19 development donors")
    if [record.get("accession") for record in records] != list(DEVELOPMENT_DONORS):
        raise PermissionError("reduced donor order differs from the frozen split")
    if any(record.get("accession") in HELD_DONORS for record in records):
        raise PermissionError("a held donor entered the development reduction")
    samples = [record.get("sample") for record in records]
    if any(not isinstance(sample, str) or not sample for sample in samples) or len(
        set(samples)
    ) != len(samples):
        raise ValueError("development sample identities must be nonempty and unique")

    tables = np.asarray([record.get("tables") for record in records])
    destroyed = np.asarray([record.get("destroyed_tables") for record in records])
    expected_shape = (len(DEVELOPMENT_DONORS), len(MARKERS) ** 2, 4)
    for name, values in (("tables", tables), ("destroyed tables", destroyed)):
        if (
            values.shape != expected_shape
            or not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 0)
        ):
            raise ValueError(f"{name} must be nonnegative integer 2x2 tables")
    tables = tables.reshape(len(DEVELOPMENT_DONORS), 9, 9, 2, 2).astype(np.int64)
    destroyed = destroyed.reshape(tables.shape).astype(np.int64)
    if np.any(tables.sum(axis=(-2, -1)) != CELL_BUDGET) or np.any(
        destroyed.sum(axis=(-2, -1)) != CELL_BUDGET
    ):
        raise ValueError("every reduced table must use exactly 1,024 cells")
    row_margins = tables.sum(axis=-1)
    column_margins = tables.sum(axis=-2)
    if not np.array_equal(
        row_margins, np.broadcast_to(row_margins[:, :, :1], row_margins.shape)
    ) or not np.array_equal(
        column_margins,
        np.broadcast_to(column_margins[:, :1, :], column_margins.shape),
    ):
        raise ValueError("ordered entities do not share their endpoint margins")
    if np.any(column_margins != CELL_BUDGET // 2):
        raise ValueError("an ADT endpoint is not split into exactly 512 and 512 cells")
    if not np.array_equal(
        tables.sum(axis=-1), destroyed.sum(axis=-1)
    ) or not np.array_equal(tables.sum(axis=-2), destroyed.sum(axis=-2)):
        raise ValueError("destroyed-link tables changed a fixed margin")

    supplied_support = np.asarray([record.get("informative") for record in records])
    derived_support = _informative(tables).reshape(len(DEVELOPMENT_DONORS), -1)
    if (
        supplied_support.shape != derived_support.shape
        or any(
            not isinstance(value, bool)
            for record in records
            for value in record.get("informative", [])
        )
        or not np.array_equal(supplied_support, derived_support)
    ):
        raise ValueError("reported informative masks differ from exact support")
    support_counts = derived_support.sum(axis=1)
    if np.any(support_counts < MINIMUM_INFORMATIVE_ENTITIES):
        raise ValueError("a development donor misses the fixed support floor")

    rna = np.asarray([record.get("rna_detection_prevalence") for record in records])
    adt = np.asarray([record.get("adt_log_panel_fraction_mean") for record in records])
    if (
        rna.shape != (len(DEVELOPMENT_DONORS), len(MARKERS))
        or adt.shape != rna.shape
        or not np.isfinite(rna).all()
        or not np.isfinite(adt).all()
        or np.any((rna < 0.0) | (rna > 1.0))
        or np.any(adt < 0.0)
    ):
        raise ValueError("fold-local marginal marker profiles are invalid")
    expected_rna = tables[:, :, 0, 1, :].sum(axis=-1) / CELL_BUDGET
    if not np.allclose(rna, expected_rna, rtol=0.0, atol=0.0):
        raise ValueError("RNA graph profiles differ from the reduced table margins")
    for record in records:
        if (
            record.get("role") != "development"
            or record.get("cells") != CELL_BUDGET
            or record.get("cell_selection_salt") != reducer.CELL_SELECTION_SALT
            or record.get("markers") != list(MARKERS)
            or record.get("entity_count") != len(MARKERS) ** 2
            or record.get("predictions_materialized_sha256") is not None
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(record.get("selected_barcode_axis_sha256"))
            )
            or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("matrix_sha256")))
        ):
            raise PermissionError("a donor record differs from the development seal")
    return {
        "source_manifest_sha256": source_hash,
        "donors": list(DEVELOPMENT_DONORS),
        "tables": tables,
        "destroyed_tables": destroyed,
        "rna_profiles": rna.astype(float),
        "adt_profiles": adt.astype(float),
        "support_counts": support_counts.astype(int),
    }


def _knn_edge_incidence(profiles: np.ndarray, neighbors: int) -> np.ndarray:
    values = np.asarray(profiles, dtype=float)
    k = int(neighbors)
    if values.ndim != 2 or values.shape[1] != len(MARKERS):
        raise ValueError("profiles must be training donor by locked marker")
    if not np.isfinite(values).all() or not 1 <= k < len(MARKERS):
        raise ValueError("profiles or graph neighborhood are invalid")
    marker_profiles = values.T
    scale = marker_profiles.std(axis=1, ddof=1)
    zero = np.flatnonzero(scale == 0.0)
    if zero.size:
        names = ", ".join(MARKERS[index] for index in zero)
        raise GraphConstructionRefusal(f"zero-variance marker profile: {names}")
    standardized = (
        marker_profiles - marker_profiles.mean(axis=1, keepdims=True)
    ) / scale[:, None]
    edges: set[tuple[int, int]] = set()
    marker_axis = np.arange(len(MARKERS))
    for marker in marker_axis:
        candidates = marker_axis[marker_axis != marker]
        distance = np.linalg.norm(
            standardized[candidates] - standardized[marker], axis=1
        )
        order = candidates[np.lexsort((candidates, distance))]
        edges.update(tuple(sorted((int(marker), int(other)))) for other in order[:k])
    ordered = sorted(edges)
    incidence = np.zeros((len(MARKERS), len(ordered)), dtype=float)
    for edge, (first, second) in enumerate(ordered):
        incidence[first, edge] = 1.0
        incidence[second, edge] = 1.0
    if not ordered or np.any(incidence.sum(axis=1) == 0.0):
        raise GraphConstructionRefusal("fold-local kNN union has an isolated marker")
    return incidence


def _label_permuted_incidence(
    incidence: np.ndarray, modality: str, training_donors: list[str]
) -> np.ndarray:
    context = "|".join(training_donors)
    order = sorted(
        range(len(MARKERS)),
        key=lambda marker: hashlib.sha256(
            ("label-permuted-v1" + context + modality + MARKERS[marker]).encode()
        ).hexdigest(),
    )
    return np.asarray(incidence, dtype=float)[order]


def _fold_graphs(
    rna_profiles: np.ndarray,
    adt_profiles: np.ndarray,
    training_donors: list[str],
) -> tuple[
    dict[int, tuple[np.ndarray, np.ndarray]],
    dict[int, tuple[np.ndarray, np.ndarray]],
    dict[str, Any],
]:
    graphs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    permuted: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    audit: dict[str, Any] = {"training_donors": training_donors, "neighborhoods": {}}
    for neighbors in NEIGHBOR_GRID:
        rna = _knn_edge_incidence(rna_profiles, neighbors)
        adt = _knn_edge_incidence(adt_profiles, neighbors)
        graphs[neighbors] = (rna, adt)
        permuted[neighbors] = (
            _label_permuted_incidence(rna, "rna", training_donors),
            _label_permuted_incidence(adt, "adt", training_donors),
        )
        audit["neighborhoods"][str(neighbors)] = {
            "rna_edges": int(rna.shape[1]),
            "adt_edges": int(adt.shape[1]),
            "rna_incidence_sha256": _array_sha256(rna),
            "adt_incidence_sha256": _array_sha256(adt),
            "permuted_rna_incidence_sha256": _array_sha256(permuted[neighbors][0]),
            "permuted_adt_incidence_sha256": _array_sha256(permuted[neighbors][1]),
        }
    return graphs, permuted, audit


def _conditional_support(tables: np.ndarray) -> dict[str, np.ndarray]:
    truth = np.asarray(tables, dtype=float).reshape(-1, 2, 2)
    rows = truth.sum(axis=-1).astype(int)
    columns = truth.sum(axis=-2).astype(int)
    total = truth.sum(axis=(-2, -1)).astype(int)
    lower = np.maximum(0, rows[:, 0] + columns[:, 0] - total)
    upper = np.minimum(rows[:, 0], columns[:, 0])
    width = int(np.max(upper - lower + 1))
    support = lower[:, None] + np.arange(width)[None, :]
    valid = support <= upper[:, None]
    support_float = support.astype(float)
    c0 = columns[:, 0, None].astype(float)
    c1 = columns[:, 1, None].astype(float)
    r0 = rows[:, 0, None].astype(float)
    logbase = (
        gammaln(c0 + 1.0)
        - gammaln(support_float + 1.0)
        - gammaln(c0 - support_float + 1.0)
        + gammaln(c1 + 1.0)
        - gammaln(r0 - support_float + 1.0)
        - gammaln(c1 - r0 + support_float + 1.0)
    )
    logbase[~valid] = -np.inf
    return {
        "tables": truth,
        "rows": rows.astype(float),
        "columns": columns.astype(float),
        "informative": upper > lower,
        "support": support_float,
        "valid": valid,
        "logbase": logbase,
    }


def _conditional_expected_tables(
    log_odds: np.ndarray, recipient: dict[str, np.ndarray]
) -> np.ndarray:
    theta = np.asarray(log_odds, dtype=float).reshape(-1)
    if theta.shape != recipient["tables"].shape[:1] or not np.isfinite(theta).all():
        raise ValueError("one finite log odds is required per target entity")
    logmass = recipient["logbase"] + theta[:, None] * recipient["support"]
    probability = np.exp(logmass - logsumexp(logmass, axis=1, keepdims=True))
    probability[~recipient["valid"]] = 0.0
    upper_left = np.sum(probability * recipient["support"], axis=1)
    rows = recipient["rows"]
    columns = recipient["columns"]
    return np.stack(
        (
            upper_left,
            rows[:, 0] - upper_left,
            columns[:, 0] - upper_left,
            rows[:, 1] - columns[:, 0] + upper_left,
        ),
        axis=-1,
    ).reshape(-1, 2, 2)


def _donor_loss(
    truth: np.ndarray,
    predicted: np.ndarray,
    informative: np.ndarray | None = None,
) -> float:
    observed = np.asarray(truth, dtype=float).reshape(-1, 2, 2)
    fitted = np.asarray(predicted, dtype=float).reshape(observed.shape)
    support = _informative(observed) if informative is None else np.asarray(informative)
    if (
        support.shape != observed.shape[:1]
        or np.count_nonzero(support) < MINIMUM_INFORMATIVE_ENTITIES
    ):
        raise ValueError("target donor misses the fixed support floor")
    if not np.isfinite(fitted).all() or np.any(fitted < -1e-9):
        raise FloatingPointError("prediction is not finite and nonnegative")
    fitted = np.maximum(fitted, 0.0)
    if not np.allclose(
        observed.sum(axis=-1), fitted.sum(axis=-1), rtol=0.0, atol=1e-10
    ) or not np.allclose(
        observed.sum(axis=-2), fitted.sum(axis=-2), rtol=0.0, atol=1e-10
    ):
        raise FloatingPointError("prediction changed a target margin")
    positive = observed > 0.0
    if np.any(fitted[positive] <= 0.0):
        raise FloatingPointError("prediction assigns zero mass to an observed cell")
    terms = np.zeros_like(observed)
    terms[positive] = observed[positive] * (
        np.log(observed[positive]) - np.log(fitted[positive])
    )
    entity = 2.0 * terms.sum(axis=(-2, -1)) / observed.sum(axis=(-2, -1))
    return float(entity[support].mean())


def _predict_conditional(
    source_log_odds: np.ndarray,
    alpha: float,
    recipient: dict[str, np.ndarray],
) -> np.ndarray:
    return _conditional_expected_tables(
        float(alpha) * np.asarray(source_log_odds).reshape(-1), recipient
    )


def _independence_prediction(tables: np.ndarray) -> np.ndarray:
    values = np.asarray(tables, dtype=float).reshape(-1, 2, 2)
    rows = values.sum(axis=-1)
    columns = values.sum(axis=-2)
    total = values.sum(axis=(-2, -1))
    return rows[:, :, None] * columns[:, None, :] / total[:, None, None]


def _canonical_table(rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    total = int(rows.sum())
    upper_left = max(0, int(rows[0] + columns[0] - total))
    return np.asarray(
        [
            [upper_left, int(rows[0] - upper_left)],
            [int(columns[0] - upper_left), int(rows[1] - columns[0] + upper_left)],
        ],
        dtype=np.int64,
    )


def _fractional_deviance_coordinate(table: np.ndarray) -> float:
    values = np.asarray(table, dtype=float)
    expected = np.outer(values.sum(axis=1), values.sum(axis=0)) / values.sum()
    positive = values > 0.0
    deviance = 2.0 * float(
        np.sum(
            values[positive] * (np.log(values[positive]) - np.log(expected[positive]))
        )
    )
    determinant = float(values[0, 0] * values[1, 1] - values[0, 1] * values[1, 0])
    return math.copysign(math.sqrt(max(deviance, 0.0)), determinant)


def _fractional_pearson_coordinate(table: np.ndarray) -> float:
    values = np.asarray(table, dtype=float)
    total = float(values.sum())
    rows = values.sum(axis=1)
    columns = values.sum(axis=0)
    determinant = float(values[0, 0] * values[1, 1] - values[0, 1] * values[1, 0])
    return determinant * math.sqrt(total / float(np.prod(rows) * np.prod(columns)))


def _classical_table(
    coordinate: float,
    rows: np.ndarray,
    columns: np.ndarray,
    family: str,
) -> np.ndarray:
    total = float(rows.sum())
    lower = float(max(0.0, rows[0] + columns[0] - total))
    upper = float(min(rows[0], columns[0]))
    if upper <= lower:
        return _canonical_table(rows, columns).astype(float)
    left = float(np.nextafter(lower, upper))
    right = float(np.nextafter(upper, lower))

    def table_at(value: float) -> np.ndarray:
        return np.asarray(
            [
                [value, rows[0] - value],
                [columns[0] - value, rows[1] - columns[0] + value],
            ]
        )

    if family == "pearson":
        statistic = _fractional_pearson_coordinate
    elif family == "deviance":
        statistic = _fractional_deviance_coordinate
    else:
        raise ValueError("classical residual family must be pearson or deviance")
    target = min(
        max(float(coordinate), statistic(table_at(left))), statistic(table_at(right))
    )
    for _ in range(128):
        midpoint = 0.5 * (left + right)
        if statistic(table_at(midpoint)) < target:
            left = midpoint
        else:
            right = midpoint
    upper_left = 0.5 * (left + right)
    return np.asarray(
        [
            [upper_left, rows[0] - upper_left],
            [columns[0] - upper_left, rows[1] - columns[0] + upper_left],
        ]
    )


def _residual_pool(
    tables: np.ndarray, family: str, centered: bool
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(tables).reshape(len(tables), -1, 2, 2)
    support = _informative(values)
    support_count = support.sum(axis=0)
    if np.any(support_count < 2):
        raise CouplingEstimationRefusal(
            "too few informative training donors for a residual entity"
        )
    coordinates = np.full(support.shape, np.nan, dtype=float)
    function: Callable[[np.ndarray], float] = (
        signed_pearson_coordinate if family == "pearson" else signed_deviance_coordinate
    )
    for donor, entity in np.argwhere(support):
        table = values[donor, entity]
        coordinate = (
            centered_classical_coordinate(table, statistic=family).centered_coordinate
            if centered
            else function(table)
        )
        coordinates[donor, entity] = coordinate / math.sqrt(float(table.sum()))
    pooled = np.nanmean(coordinates, axis=0)
    if not np.isfinite(pooled).all():
        raise CouplingEstimationRefusal("residual pooling did not cover every entity")
    return pooled, {
        "support_count_range": [int(support_count.min()), int(support_count.max())],
        "sample_size_normalized": True,
        "donor_equal_pooling": True,
        "pooling_support": "fixed-margin-informative source donors only",
    }


def _target_null_mean(tables: np.ndarray, family: str) -> np.ndarray:
    values = np.asarray(tables).reshape(-1, 2, 2)
    null = np.zeros(len(values), dtype=float)
    for entity, table in enumerate(values):
        rows = table.sum(axis=1)
        columns = table.sum(axis=0)
        if _informative(table[None])[0]:
            canonical = _canonical_table(rows, columns)
            null[entity] = centered_classical_coordinate(
                canonical, statistic=family
            ).null_mean_coordinate
    return null


def _predict_residual(
    pooled: np.ndarray,
    target_tables: np.ndarray,
    *,
    family: str,
    centered: bool,
    alpha: float,
    target_null: np.ndarray,
) -> np.ndarray:
    values = np.asarray(target_tables).reshape(-1, 2, 2)
    coordinate = np.asarray(pooled, dtype=float).reshape(-1)
    if coordinate.shape != values.shape[:1]:
        raise ValueError("residual coordinate and target entity axes differ")
    predicted = np.empty_like(values, dtype=float)
    for entity, table in enumerate(values):
        rows = table.sum(axis=1)
        columns = table.sum(axis=0)
        statistic = float(alpha) * coordinate[entity] * math.sqrt(float(table.sum()))
        if centered:
            statistic += float(target_null[entity])
        predicted[entity] = _classical_table(statistic, rows, columns, family)
    return predicted


def _hierarchical_certificate(fit: Any) -> dict[str, Any]:
    checks = {
        "converged": bool(fit.converged),
        "finite_objective": bool(np.isfinite(fit.objective)),
        "scaled_gradient": bool(fit.scaled_gradient_norm <= fit.gradient_tolerance),
        "schur_condition": bool(
            np.isfinite(fit.schur_condition_number)
            and fit.schur_condition_number <= MAXIMUM_CONDITION_NUMBER
        ),
        "theta_curvature_condition": bool(
            np.isfinite(fit.theta_curvature_condition_number)
            and fit.theta_curvature_condition_number <= MAXIMUM_CONDITION_NUMBER
        ),
        "positive_curvature": bool(
            fit.minimum_schur_eigenvalue > 0.0 and fit.minimum_theta_curvature > 0.0
        ),
        "support": bool(np.all(fit.support_count >= 2)),
    }
    if not all(checks.values()):
        raise CouplingEstimationRefusal(
            "hierarchical fit misses an external numerical certificate"
        )
    return {
        "certificate_family": "hierarchical_exact_block_newton",
        "converged": True,
        "objective": float(fit.objective),
        "gradient_norm": float(fit.gradient_norm),
        "scaled_gradient_norm": float(fit.scaled_gradient_norm),
        "gradient_tolerance": float(fit.gradient_tolerance),
        "schur_condition_number": float(fit.schur_condition_number),
        "theta_curvature_condition_number": float(fit.theta_curvature_condition_number),
        "minimum_schur_eigenvalue": float(fit.minimum_schur_eigenvalue),
        "minimum_theta_curvature": float(fit.minimum_theta_curvature),
        "iterations": int(fit.iterations),
        "support_count_range": [
            int(np.min(fit.support_count)),
            int(np.max(fit.support_count)),
        ],
        "checks": checks,
    }


def _fit_hierarchical(
    tables: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    heterogeneity: float,
    ridge: float,
    graph: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    fit = fit_hierarchical_conditional_log_odds(
        np.asarray(tables),
        first,
        second,
        heterogeneity_penalty=heterogeneity,
        ridge_penalty=ridge,
        graph_penalty=graph,
        minimum_informative_donors=2,
        maximum_condition_number=MAXIMUM_CONDITION_NUMBER,
        tolerance=HIERARCHICAL_TOLERANCE,
    )
    return fit.population_log_odds.reshape(-1), _hierarchical_certificate(fit)


def _common_certificate(fit: Any, tables: np.ndarray) -> dict[str, Any]:
    threshold = 1e-7 + 1e-10 * max(
        1.0, float(np.asarray(tables)[:, :, :, 0, 0].sum(axis=0).max())
    )
    checks = {
        "converged": bool(fit.converged),
        "finite_objective": bool(np.isfinite(fit.objective)),
        "gradient": bool(fit.gradient_norm <= threshold),
        "condition": bool(
            np.isfinite(fit.condition_number)
            and fit.condition_number <= MAXIMUM_CONDITION_NUMBER
        ),
        "support": bool(np.all(fit.support_count >= 2)),
    }
    if not all(checks.values()):
        raise CouplingEstimationRefusal(
            "common-effect fit misses an external numerical certificate"
        )
    return {
        "certificate_family": "common_effect_exact_hessian",
        "converged": True,
        "objective": float(fit.objective),
        "gradient_norm": float(fit.gradient_norm),
        "scaled_gradient_norm": float(fit.gradient_norm),
        "gradient_tolerance": float(threshold),
        "schur_condition_number": float(fit.condition_number),
        "theta_curvature_condition_number": 1.0,
        "iterations": int(fit.iterations),
        "support_count_range": [
            int(np.min(fit.support_count)),
            int(np.max(fit.support_count)),
        ],
        "checks": checks,
    }


def _fit_common(
    tables: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    ridge: float,
    graph: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    fit = fit_structured_conditional_log_odds(
        np.asarray(tables),
        first,
        second,
        initial_log_odds=np.zeros((len(MARKERS), len(MARKERS))),
        ridge_penalty=ridge,
        graph_penalty=graph,
        minimum_informative_donors=2,
        maximum_condition_number=MAXIMUM_CONDITION_NUMBER,
        tolerance=COMMON_TOLERANCE,
    )
    return fit.log_odds.reshape(-1), _common_certificate(fit, tables)


def _configuration(family: str, config: tuple[Any, ...]) -> dict[str, Any]:
    if family in {"primary", "destroyed_link", "label_permuted_graph"}:
        keys = (
            "graph_neighborhood",
            "heterogeneity_penalty",
            "ridge_penalty",
            "graph_penalty",
            "transport_multiplier",
        )
    elif family == "hierarchical_ridge_only":
        keys = (
            "heterogeneity_penalty",
            "ridge_penalty",
            "transport_multiplier",
        )
    elif family == "common_effect_graph":
        keys = (
            "graph_neighborhood",
            "ridge_penalty",
            "graph_penalty",
            "transport_multiplier",
        )
    elif family == "common_effect_ridge_only":
        keys = ("ridge_penalty", "transport_multiplier")
    elif family == "best_residual":
        keys = ("family", "centered", "transport_multiplier")
    else:
        raise KeyError(f"unknown candidate family {family}")
    return dict(zip(keys, config))


class _CandidateBook:
    def __init__(
        self, family: str, configs: list[tuple[Any, ...]], donors: list[str]
    ) -> None:
        self.family = family
        self.configs = configs
        self.donors = donors
        self.losses = {
            config: np.full(len(donors), np.nan, dtype=float) for config in configs
        }
        self.refusals: dict[tuple[Any, ...], dict[str, str]] = {
            config: {} for config in configs
        }
        self.certificates: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {
            config: {} for config in configs
        }

    def record(
        self,
        config: tuple[Any, ...],
        fold: int,
        loss: float,
        certificate: dict[str, Any] | None = None,
    ) -> None:
        value = float(loss)
        if not np.isfinite(value) or value < 0.0:
            raise FloatingPointError("candidate donor deviance is invalid")
        self.losses[config][fold] = value
        if certificate is not None:
            self.certificates[config][self.donors[fold]] = certificate

    def refuse(self, config: tuple[Any, ...], fold: int, reason: str) -> None:
        self.refusals[config][self.donors[fold]] = reason

    def selected(self) -> tuple[Any, ...] | None:
        eligible = [
            config
            for config in self.configs
            if np.isfinite(self.losses[config]).all() and not self.refusals[config]
        ]
        if not eligible:
            return None
        order = {config: index for index, config in enumerate(self.configs)}
        return min(
            eligible,
            key=lambda config: (float(self.losses[config].mean()), order[config]),
        )

    def diagnostics(self) -> dict[str, Any]:
        rows = []
        for config in self.configs:
            losses = self.losses[config]
            eligible = bool(np.isfinite(losses).all() and not self.refusals[config])
            rows.append(
                {
                    "configuration": _configuration(self.family, config),
                    "status": "ELIGIBLE" if eligible else "REFUSED",
                    "mean_donor_equal_deviance": (
                        float(losses.mean()) if eligible else None
                    ),
                    "donor_losses": {
                        donor: (float(value) if np.isfinite(value) else None)
                        for donor, value in zip(self.donors, losses)
                    },
                    "refusals": self.refusals[config],
                    "certified_folds": len(self.certificates[config]),
                }
            )
        selected = self.selected()
        return {
            "expected_candidates": len(self.configs),
            "eligible_candidates": sum(row["status"] == "ELIGIBLE" for row in rows),
            "refused_candidates": sum(row["status"] == "REFUSED" for row in rows),
            "selected_configuration": (
                _configuration(self.family, selected) if selected is not None else None
            ),
            "candidates": rows,
        }


def _candidate_books(donors: list[str]) -> dict[str, _CandidateBook]:
    hierarchical = list(
        product(
            NEIGHBOR_GRID,
            HETEROGENEITY_GRID,
            RIDGE_GRID,
            GRAPH_GRID,
            ALPHA_GRID,
        )
    )
    residual = [
        (family, centered, alpha)
        for family, centered in (
            ("pearson", False),
            ("pearson", True),
            ("deviance", False),
            ("deviance", True),
        )
        for alpha in ALPHA_GRID
    ]
    return {
        "primary": _CandidateBook("primary", hierarchical, donors),
        "destroyed_link": _CandidateBook("destroyed_link", hierarchical.copy(), donors),
        "label_permuted_graph": _CandidateBook(
            "label_permuted_graph", hierarchical.copy(), donors
        ),
        "hierarchical_ridge_only": _CandidateBook(
            "hierarchical_ridge_only",
            list(product(HETEROGENEITY_GRID, RIDGE_GRID, ALPHA_GRID)),
            donors,
        ),
        "common_effect_graph": _CandidateBook(
            "common_effect_graph",
            list(product(NEIGHBOR_GRID, RIDGE_GRID, GRAPH_GRID, ALPHA_GRID)),
            donors,
        ),
        "common_effect_ridge_only": _CandidateBook(
            "common_effect_ridge_only",
            list(product(RIDGE_GRID, ALPHA_GRID)),
            donors,
        ),
        "best_residual": _CandidateBook("best_residual", residual, donors),
    }


def _record_conditional_alphas(
    book: _CandidateBook,
    prefix: tuple[Any, ...],
    fold: int,
    coordinate: np.ndarray,
    certificate: dict[str, Any],
    recipient: dict[str, np.ndarray],
) -> None:
    for alpha in ALPHA_GRID:
        config = (*prefix, alpha)
        try:
            prediction = _predict_conditional(coordinate, alpha, recipient)
            loss = _donor_loss(
                recipient["tables"], prediction, recipient["informative"]
            )
        except (FloatingPointError, ValueError) as error:
            book.refuse(config, fold, f"{type(error).__name__}: {error}")
        else:
            book.record(config, fold, loss, certificate)


def _refuse_alphas(
    book: _CandidateBook,
    prefix: tuple[Any, ...],
    fold: int,
    error: Exception,
) -> None:
    reason = f"{type(error).__name__}: {error}"
    for alpha in ALPHA_GRID:
        book.refuse((*prefix, alpha), fold, reason)


def _cross_validate_serial(
    data: dict[str, Any], folds: tuple[int, ...] | None = None
) -> dict[str, Any]:
    donors = data["donors"]
    books = _candidate_books(donors)
    independence = np.full(len(donors), np.nan, dtype=float)
    graph_audit = []
    identity = np.eye(len(MARKERS), dtype=float)

    selected_folds = tuple(range(len(donors))) if folds is None else folds
    for fold in selected_folds:
        omitted = donors[fold]
        training = np.arange(len(donors)) != fold
        training_donors = [
            donor for index, donor in enumerate(donors) if training[index]
        ]
        target_tables = data["tables"][fold]
        recipient = _conditional_support(target_tables)
        independence[fold] = _donor_loss(
            target_tables,
            _independence_prediction(target_tables),
            recipient["informative"],
        )
        try:
            graphs, permuted, audit = _fold_graphs(
                data["rna_profiles"][training],
                data["adt_profiles"][training],
                training_donors,
            )
        except GraphConstructionRefusal as error:
            graph_audit.append(
                {
                    "omitted_donor": omitted,
                    "training_donors": training_donors,
                    "status": "REFUSED",
                    "reason": str(error),
                }
            )
            for family in (
                "primary",
                "destroyed_link",
                "label_permuted_graph",
                "common_effect_graph",
            ):
                for config in books[family].configs:
                    books[family].refuse(config, fold, str(error))
            graphs = {}
            permuted = {}
        else:
            graph_audit.append({"omitted_donor": omitted, "status": "OK", **audit})

        for neighbors in NEIGHBOR_GRID:
            if neighbors not in graphs:
                continue
            first, second = graphs[neighbors]
            permuted_first, permuted_second = permuted[neighbors]
            for heterogeneity in HETEROGENEITY_GRID:
                for ridge in RIDGE_GRID:
                    for graph in GRAPH_GRID:
                        prefix = (neighbors, heterogeneity, ridge, graph)
                        for family, tables, incidence in (
                            (
                                "primary",
                                data["tables"][training],
                                (first, second),
                            ),
                            (
                                "destroyed_link",
                                data["destroyed_tables"][training],
                                (first, second),
                            ),
                            (
                                "label_permuted_graph",
                                data["tables"][training],
                                (permuted_first, permuted_second),
                            ),
                        ):
                            try:
                                coordinate, certificate = _fit_hierarchical(
                                    tables,
                                    *incidence,
                                    heterogeneity,
                                    ridge,
                                    graph,
                                )
                            except (
                                CouplingEstimationRefusal,
                                FloatingPointError,
                                np.linalg.LinAlgError,
                            ) as error:
                                _refuse_alphas(books[family], prefix, fold, error)
                            else:
                                _record_conditional_alphas(
                                    books[family],
                                    prefix,
                                    fold,
                                    coordinate,
                                    certificate,
                                    recipient,
                                )
            for ridge in RIDGE_GRID:
                for graph in GRAPH_GRID:
                    prefix = (neighbors, ridge, graph)
                    try:
                        coordinate, certificate = _fit_common(
                            data["tables"][training], first, second, ridge, graph
                        )
                    except (
                        CouplingEstimationRefusal,
                        FloatingPointError,
                        np.linalg.LinAlgError,
                    ) as error:
                        _refuse_alphas(
                            books["common_effect_graph"], prefix, fold, error
                        )
                    else:
                        _record_conditional_alphas(
                            books["common_effect_graph"],
                            prefix,
                            fold,
                            coordinate,
                            certificate,
                            recipient,
                        )

        for heterogeneity in HETEROGENEITY_GRID:
            for ridge in RIDGE_GRID:
                prefix = (heterogeneity, ridge)
                try:
                    coordinate, certificate = _fit_hierarchical(
                        data["tables"][training],
                        identity,
                        identity,
                        heterogeneity,
                        ridge,
                        0.0,
                    )
                except (
                    CouplingEstimationRefusal,
                    FloatingPointError,
                    np.linalg.LinAlgError,
                ) as error:
                    _refuse_alphas(
                        books["hierarchical_ridge_only"], prefix, fold, error
                    )
                else:
                    _record_conditional_alphas(
                        books["hierarchical_ridge_only"],
                        prefix,
                        fold,
                        coordinate,
                        certificate,
                        recipient,
                    )
        for ridge in RIDGE_GRID:
            prefix = (ridge,)
            try:
                coordinate, certificate = _fit_common(
                    data["tables"][training], identity, identity, ridge, 0.0
                )
            except (
                CouplingEstimationRefusal,
                FloatingPointError,
                np.linalg.LinAlgError,
            ) as error:
                _refuse_alphas(books["common_effect_ridge_only"], prefix, fold, error)
            else:
                _record_conditional_alphas(
                    books["common_effect_ridge_only"],
                    prefix,
                    fold,
                    coordinate,
                    certificate,
                    recipient,
                )

        target_null = {
            family: _target_null_mean(target_tables, family)
            for family in ("pearson", "deviance")
        }
        for family, centered in (
            ("pearson", False),
            ("pearson", True),
            ("deviance", False),
            ("deviance", True),
        ):
            try:
                pooled, certificate = _residual_pool(
                    data["tables"][training], family, centered
                )
            except (CouplingEstimationRefusal, FloatingPointError) as error:
                _refuse_alphas(books["best_residual"], (family, centered), fold, error)
                continue
            for alpha in ALPHA_GRID:
                config = (family, centered, alpha)
                try:
                    prediction = _predict_residual(
                        pooled,
                        target_tables,
                        family=family,
                        centered=centered,
                        alpha=alpha,
                        target_null=target_null[family],
                    )
                    loss = _donor_loss(
                        target_tables, prediction, recipient["informative"]
                    )
                except (FloatingPointError, ValueError) as error:
                    books["best_residual"].refuse(
                        config, fold, f"{type(error).__name__}: {error}"
                    )
                else:
                    books["best_residual"].record(config, fold, loss, certificate)

    return {
        "books": books,
        "independence": independence,
        "fold_graph_audit": graph_audit,
    }


def _fold_worker(
    payload: tuple[int, dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    fold, data = payload
    return fold, _cross_validate_serial(data, (fold,))


def _cross_validate(
    data: dict[str, Any], workers: int = MAXIMUM_WORKERS
) -> dict[str, Any]:
    count = int(workers)
    if not 1 <= count <= MAXIMUM_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAXIMUM_WORKERS}")
    if count == 1:
        return _cross_validate_serial(data)
    folds = list(range(len(data["donors"])))
    with ProcessPoolExecutor(
        max_workers=count, mp_context=multiprocessing.get_context("fork")
    ) as executor:
        fold_results = list(
            executor.map(_fold_worker, ((fold, data) for fold in folds))
        )
    fold_results.sort(key=lambda item: item[0])
    donors = data["donors"]
    books = _candidate_books(donors)
    independence = np.full(len(donors), np.nan, dtype=float)
    graph_audit: list[dict[str, Any]] = []
    for fold, result in fold_results:
        donor = donors[fold]
        independence[fold] = result["independence"][fold]
        if len(result["fold_graph_audit"]) != 1:
            raise AssertionError("fold worker returned an incomplete graph audit")
        graph_audit.append(result["fold_graph_audit"][0])
        for family, target_book in books.items():
            source_book = result["books"][family]
            for config in target_book.configs:
                value = source_book.losses[config][fold]
                refusal = source_book.refusals[config].get(donor)
                if np.isfinite(value) and refusal is None:
                    target_book.record(
                        config,
                        fold,
                        float(value),
                        source_book.certificates[config].get(donor),
                    )
                elif refusal is not None and not np.isfinite(value):
                    target_book.refuse(config, fold, refusal)
                else:
                    raise AssertionError(
                        "fold worker did not return exactly one candidate outcome"
                    )
    if not np.isfinite(independence).all():
        raise AssertionError("fold workers did not return every independence loss")
    return {
        "books": books,
        "independence": independence,
        "fold_graph_audit": graph_audit,
    }


def _comparison(
    donors: list[str],
    primary: np.ndarray,
    comparator: np.ndarray,
    label: str,
    bootstrap_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    primary_values = np.asarray(primary, dtype=float)
    comparator_values = np.asarray(comparator, dtype=float)
    if (
        primary_values.shape != (len(donors),)
        or comparator_values.shape != primary_values.shape
        or not np.isfinite(primary_values).all()
        or not np.isfinite(comparator_values).all()
        or np.any(primary_values < 0.0)
        or np.any(comparator_values < 0.0)
    ):
        raise ValueError("gate requires one finite paired loss per development donor")
    if comparator_values.mean() <= 0.0:
        raise ValueError("gate comparator mean must be finite and strictly positive")
    difference = primary_values - comparator_values
    if bootstrap_indices is None:
        generator = np.random.default_rng(BOOTSTRAP_SEED)
        indices = generator.integers(
            0, len(donors), size=(BOOTSTRAPS, len(donors)), endpoint=False
        )
    else:
        indices = np.asarray(bootstrap_indices)
        if indices.shape != (BOOTSTRAPS, len(donors)) or np.any(
            (indices < 0) | (indices >= len(donors))
        ):
            raise ValueError("shared bootstrap index matrix is invalid")
    bootstrap = difference[indices].mean(axis=1)
    interval = np.quantile(bootstrap, [0.025, 0.975], method="linear")
    relative = 1.0 - float(primary_values.mean() / comparator_values.mean())
    favorable = int(np.count_nonzero(difference < 0.0))
    passes = {
        "relative_reduction_at_least_five_percent": bool(relative >= 0.05),
        "bootstrap_upper_95_below_zero": bool(interval[1] < 0.0),
        "at_least_fifteen_favorable_donors": bool(favorable >= 15),
    }
    return {
        "primary_mean_loss": float(primary_values.mean()),
        "comparator_mean_loss": float(comparator_values.mean()),
        "relative_reduction": relative,
        "bootstrap_95_ci": interval.tolist(),
        "bootstrap_upper_95": float(interval[1]),
        "bootstrap_draws": BOOTSTRAPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_quantile_method": "linear",
        "bootstrap_indices_shared_across_comparisons": True,
        "bootstrap_unit": "physical donor",
        "favorable_donors": favorable,
        "required_favorable_donors": 15,
        "donor_differences_primary_minus_comparator": {
            donor: float(value) for donor, value in zip(donors, difference)
        },
        "passes": passes,
        "passes_all": all(passes.values()),
    }


def _graph_payload(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    return {
        "representation": "two-node edge incidence of undirected directed-kNN union",
        "rna_incidence": first.tolist(),
        "adt_incidence": second.tolist(),
        "rna_incidence_sha256": _array_sha256(first),
        "adt_incidence_sha256": _array_sha256(second),
    }


def _conditional_method(
    coordinate: np.ndarray,
    alpha: float,
    config: dict[str, Any],
    certificate: dict[str, Any],
    estimator: str,
    graph: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "kind": "conditional_log_odds",
        "estimator": estimator,
        "source_coordinate": (float(alpha) * coordinate).tolist(),
        "unscaled_source_coordinate": coordinate.tolist(),
        "transport_multiplier": float(alpha),
        "selected_configuration": config,
        "numerical_certificate": certificate,
        "recipient_reconstruction": "exact conditional expected table at target margins",
    }
    if graph is not None:
        payload["graph"] = graph
    return payload


def _fit_frozen_models(
    data: dict[str, Any], selections: dict[str, tuple[Any, ...]]
) -> dict[str, Any]:
    graphs, permuted, graph_audit = _fold_graphs(
        data["rna_profiles"], data["adt_profiles"], data["donors"]
    )
    identity = np.eye(len(MARKERS), dtype=float)
    methods: dict[str, Any] = {}

    for family, tables, graph_source in (
        ("primary", data["tables"], graphs),
        ("destroyed_link", data["destroyed_tables"], graphs),
        ("label_permuted_graph", data["tables"], permuted),
    ):
        neighbors, heterogeneity, ridge, graph, alpha = selections[family]
        first, second = graph_source[int(neighbors)]
        coordinate, certificate = _fit_hierarchical(
            tables, first, second, heterogeneity, ridge, graph
        )
        methods[family] = _conditional_method(
            coordinate,
            alpha,
            _configuration(family, selections[family]),
            certificate,
            "donor-heterogeneity-aware exact conditional log odds",
            _graph_payload(first, second),
        )

    heterogeneity, ridge, alpha = selections["hierarchical_ridge_only"]
    coordinate, certificate = _fit_hierarchical(
        data["tables"], identity, identity, heterogeneity, ridge, 0.0
    )
    methods["hierarchical_ridge_only"] = _conditional_method(
        coordinate,
        alpha,
        _configuration(
            "hierarchical_ridge_only", selections["hierarchical_ridge_only"]
        ),
        certificate,
        "donor-heterogeneity-aware ridge-only exact conditional log odds",
        None,
    )

    neighbors, ridge, graph, alpha = selections["common_effect_graph"]
    first, second = graphs[int(neighbors)]
    coordinate, certificate = _fit_common(data["tables"], first, second, ridge, graph)
    methods["common_effect_graph"] = _conditional_method(
        coordinate,
        alpha,
        _configuration("common_effect_graph", selections["common_effect_graph"]),
        certificate,
        "common-effect exact conditional graph fit",
        _graph_payload(first, second),
    )

    ridge, alpha = selections["common_effect_ridge_only"]
    coordinate, certificate = _fit_common(
        data["tables"], identity, identity, ridge, 0.0
    )
    methods["common_effect_ridge_only"] = _conditional_method(
        coordinate,
        alpha,
        _configuration(
            "common_effect_ridge_only", selections["common_effect_ridge_only"]
        ),
        certificate,
        "common-effect ridge-only exact conditional fit",
        None,
    )

    family, centered, alpha = selections["best_residual"]
    coordinate, residual_certificate = _residual_pool(data["tables"], family, centered)
    methods["best_residual"] = {
        "kind": "classical_residual",
        "family": family,
        "centered": bool(centered),
        "source_coordinate": (float(alpha) * coordinate).tolist(),
        "unscaled_source_coordinate": coordinate.tolist(),
        "transport_multiplier": float(alpha),
        "selected_configuration": _configuration(
            "best_residual", selections["best_residual"]
        ),
        "sample_size_normalized": True,
        "normalization": "source/sqrt(n), recipient*sqrt(m)",
        "donor_equal_pooling": True,
        "pooling_support": "fixed-margin-informative source donors only",
        "target_null_restored": bool(centered),
        "target_margin_inversion": True,
        "numerical_certificate": residual_certificate,
    }
    methods["independence"] = {"kind": "independence"}
    if set(methods) != set(METHODS):
        raise AssertionError("frozen method set is incomplete")
    return {
        "entity_order": "row-major RNA marker x ADT marker",
        "entity_count": len(MARKERS) ** 2,
        "methods": methods,
        "all_development_graph_audit": graph_audit,
    }


def _run_development_after_attempt(
    data: dict[str, Any], workers: int
) -> dict[str, Any]:
    evaluated = _cross_validate(data, workers=workers)
    books: dict[str, _CandidateBook] = evaluated["books"]
    selections = {family: book.selected() for family, book in books.items()}
    unavailable = [family for family, config in selections.items() if config is None]
    diagnostics = {family: book.diagnostics() for family, book in books.items()}
    selected_losses = {
        family: books[family].losses[config]
        for family, config in selections.items()
        if config is not None
    }
    selected_losses["independence"] = evaluated["independence"]

    comparisons: dict[str, Any] = {}
    gate_pass = False
    if not unavailable:
        primary = selected_losses["primary"]
        bootstrap_generator = np.random.default_rng(BOOTSTRAP_SEED)
        bootstrap_indices = bootstrap_generator.integers(
            0,
            len(data["donors"]),
            size=(BOOTSTRAPS, len(data["donors"])),
            endpoint=False,
        )
        comparisons = {
            comparator: _comparison(
                data["donors"],
                primary,
                selected_losses[comparator],
                comparator,
                bootstrap_indices,
            )
            for comparator in GATE_COMPARATORS
        }
        gate_pass = all(row["passes_all"] for row in comparisons.values())

    frozen: dict[str, Any] | None = None
    final_refit_error: str | None = None
    if gate_pass:
        try:
            frozen = _fit_frozen_models(
                data,
                {family: config for family, config in selections.items() if config},
            )
        except (
            CouplingEstimationRefusal,
            GraphConstructionRefusal,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as error:
            final_refit_error = f"{type(error).__name__}: {error}"
            gate_pass = False

    status = _completed_development_status(gate_pass, unavailable, final_refit_error)
    payload = {
        "schema": "gse279451-sepsis-exact-development/1.0",
        "status": status,
        "evaluation_attempt_sha256": _sha256(EVALUATION_ATTEMPT),
        "source_manifest_sha256": data["source_manifest_sha256"],
        "reduced_development_sha256": _sha256(INPUT),
        "evaluator_sha256": _sha256(Path(__file__)),
        "protocol_sha256": _sha256(PROTOCOL),
        "candidate_designation_sha256": _sha256(DESIGNATION),
        "family_policy_sha256": _sha256(FAMILY_POLICY),
        "reducer_sha256": _sha256(Path(reducer.__file__)),
        "transitive_bindings": _transitive_bindings(),
        "markers": list(MARKERS),
        "entity_count": len(MARKERS) ** 2,
        "cell_budget_per_donor": CELL_BUDGET,
        "all_cells_sensitivity_used": False,
        "development_donors": data["donors"],
        "selection": {
            "folds": len(DEVELOPMENT_DONORS),
            "held_one_donor_per_fold": True,
            "fold_donors": data["donors"],
            "grid": CV_GRID,
            "selection_loss": (
                "multinomial deviance per cell, averaged over informative "
                "entities within donor and donors equally"
            ),
            "candidate_tie_rule": "lexicographically smallest tuple in declared field order",
            "final_refit_donors": data["donors"] if frozen is not None else [],
        },
        "selected_settings": {
            family: (_configuration(family, config) if config is not None else None)
            for family, config in selections.items()
        },
        "development_losses": {
            family: {
                donor: float(value) for donor, value in zip(data["donors"], losses)
            }
            for family, losses in selected_losses.items()
        },
        "candidate_diagnostics": diagnostics,
        "selected_fold_numerical_certificates": {
            family: books[family].certificates[config]
            for family, config in selections.items()
            if config is not None
        },
        "fold_graph_audit": evaluated["fold_graph_audit"],
        "gate": {
            "required_relative_reduction": 0.05,
            "bootstrap_draws": BOOTSTRAPS,
            "minimum_favorable_donors": 15,
            "comparisons": comparisons,
            "unavailable_candidate_families": unavailable,
            "final_refit_error": final_refit_error,
            "passes_all": gate_pass,
        },
        "frozen_source_model": frozen,
        "access_audit": {
            "reduced_development_donors_read": len(DEVELOPMENT_DONORS),
            "held_matrix_members_opened": 0,
            "held_matrix_entries_decoded": 0,
            "held_margins_computed": 0,
            "held_tables_formed": 0,
            "held_outcomes_used_for_selection": False,
            "raw_count_or_matrix_path_opened": False,
            "all_cells_sensitivity_run": False,
        },
    }
    _write_json_exclusive(OUTPUT, payload)
    return payload


def _completed_development_status(
    gate_pass: bool, unavailable: list[str], final_refit_error: str | None
) -> str:
    if unavailable or final_refit_error is not None:
        raise DevelopmentEvaluationRefusal(
            {
                "unavailable_candidate_families": list(unavailable),
                "final_refit_error": final_refit_error,
            }
        )
    return "DEVELOPMENT_PASS" if gate_pass else "DEVELOPMENT_FAIL"


def run_development(workers: int = MAXIMUM_WORKERS) -> dict[str, Any]:
    if any(path.exists() for path in (OUTPUT, EVALUATION_ATTEMPT, EVALUATION_REFUSAL)):
        raise FileExistsError("a GSE279451 development evaluation artifact exists")
    if not INPUT.is_file() or not reducer.DEVELOPMENT_ATTEMPT.is_file():
        raise FileNotFoundError("reduced input or acquisition attempt is absent")
    reduced_hash = _sha256(INPUT)
    acquisition_attempt_hash = _sha256(reducer.DEVELOPMENT_ATTEMPT)
    _write_json_exclusive(
        EVALUATION_ATTEMPT,
        {
            "schema": "gse279451-sepsis-evaluation-attempt/1.0",
            "status": "TERMINAL_DEVELOPMENT_EVALUATION_STARTED",
            "reduced_development_sha256": reduced_hash,
            "development_attempt_sha256": acquisition_attempt_hash,
            "evaluator_sha256": _sha256(Path(__file__)),
            "protocol_sha256": _sha256(PROTOCOL),
            "candidate_designation_sha256": _sha256(DESIGNATION),
            "family_policy_sha256": _sha256(FAMILY_POLICY),
            "transitive_bindings": _transitive_bindings(),
            "numerical_evaluation_starts_after_this_write": True,
            "held_matrix_members_opened": 0,
        },
    )
    try:
        data = _validated_reduced(INPUT)
        return _run_development_after_attempt(data, workers)
    except Exception as error:
        if not EVALUATION_REFUSAL.exists():
            refusal = {
                "schema": "gse279451-sepsis-evaluation-refusal/1.0",
                "status": "TERMINAL_DEVELOPMENT_EVALUATION_REFUSAL",
                "error_type": type(error).__name__,
                "reason": "development evaluation refused after the terminal attempt",
                "evaluation_attempt_sha256": _sha256(EVALUATION_ATTEMPT),
                "development_attempt_sha256": acquisition_attempt_hash,
                "reduced_development_sha256": reduced_hash,
                "transitive_bindings": _transitive_bindings(),
                "held_matrix_members_opened": 0,
                "rerun_permitted": False,
            }
            if isinstance(error, DevelopmentEvaluationRefusal):
                refusal["evaluation_detail"] = error.detail
            _write_json_exclusive(EVALUATION_REFUSAL, refusal)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=MAXIMUM_WORKERS)
    args = parser.parse_args()
    print(
        json.dumps(
            run_development(workers=args.workers),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
