"""Non-held development for the donor-disjoint BMMC confirmation.

Only fit donors 11466/19593 and bridge donor 15078 are permitted count rows.
The six held donors are represented solely by an explicit forbidden CSR-row
vector constructed from metadata and the H5AD observation axis.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

import h5py
import numpy as np

from experiments import confirm_scmmib_bmmc as lock
from mapreg.heterogeneity_adaptive_coupling import (
    CouplingEstimationRefusal,
    centered_classical_coordinate,
    expected_binary_table_from_log_odds,
    fit_structured_conditional_log_odds,
    signed_deviance_coordinate,
    signed_pearson_coordinate,
)
from mapreg.hierarchical_conditional_coupling import (
    fit_hierarchical_conditional_log_odds,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/development/scmmib_bmmc_exact_development.json"
SOFTWARE_AUDIT = (
    ROOT / "results/development/scmmib_bmmc_hierarchical_software_audit.json"
)
HETEROGENEITY_GRID = (0.1, 1.0, 10.0)
RIDGE_GRID = (0.01, 0.1)
GRAPH_GRID = (0.1, 0.3, 1.0)
ALPHA_GRID = (0.75, 1.0, 1.25)
NEIGHBOR_GRID = (1, 2, 3)
MINIMUM_DEVELOPMENT_ENTITIES = 80
MAXIMUM_CONDITION_NUMBER = 1e12
TOLERANCE = 1e-8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _seed(*parts: object) -> int:
    text = ":".join(map(str, ("BMMC-exact-v1", *parts)))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def _write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _metadata(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"barcode", "DonorID", "Site", "batch", "cell_type.l1"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("BMMC metadata lacks required columns")
    return rows


def _contiguous_runs(rows: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(rows, dtype=int)
    if values.ndim != 1 or not len(values) or np.any(np.diff(values) <= 0):
        raise ValueError("selected H5AD rows must be sorted and unique")
    starts = np.r_[0, np.flatnonzero(np.diff(values) > 1) + 1]
    stops = np.r_[starts[1:], len(values)]
    return [
        (int(values[start]), int(values[stop - 1]) + 1)
        for start, stop in zip(starts, stops)
    ]


def _read_csr_marker_counts(
    path: Path,
    *,
    matrix_path: str,
    selected_rows: np.ndarray,
    forbidden_rows: np.ndarray,
    selected_columns: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Read exact CSR rows and marker columns without spanning a held row."""

    rows = np.asarray(selected_rows, dtype=int)
    forbidden = np.asarray(forbidden_rows, dtype=int)
    columns = np.asarray(selected_columns, dtype=int)
    if np.intersect1d(rows, forbidden).size:
        raise PermissionError("selected count rows include a held donor")
    if len(set(columns.tolist())) != len(columns):
        raise ValueError("selected marker columns are not unique")
    runs = _contiguous_runs(rows)
    output = np.zeros((len(rows), len(columns)), dtype=float)
    column_lookup = {int(value): index for index, value in enumerate(columns)}
    row_offset = 0
    data_slices = 0
    with h5py.File(path, "r") as handle:
        matrix = handle[matrix_path]
        if matrix.attrs.get("encoding-type") != "csr_matrix":
            raise ValueError("bound raw-count matrix is not CSR")
        shape = tuple(int(value) for value in matrix.attrs["shape"])
        if np.any(rows < 0) or np.any(rows >= shape[0]):
            raise IndexError("selected CSR row is out of range")
        if np.any(columns < 0) or np.any(columns >= shape[1]):
            raise IndexError("selected CSR column is out of range")
        indptr_dataset = matrix["indptr"]
        if indptr_dataset.shape != (shape[0] + 1,):
            raise ValueError("CSR indptr shape differs")
        for start, stop in runs:
            exact_rows = np.arange(start, stop, dtype=int)
            if np.intersect1d(exact_rows, forbidden).size:
                raise PermissionError("a CSR read span would cross a held row")
            permitted_indptr = np.asarray(
                indptr_dataset[start : stop + 1], dtype=np.int64
            )
            left = int(permitted_indptr[0])
            right = int(permitted_indptr[-1])
            indices = np.asarray(matrix["indices"][left:right], dtype=np.int64)
            data = np.asarray(matrix["data"][left:right], dtype=float)
            local_indptr = permitted_indptr - left
            for local_row in range(stop - start):
                local_left = int(local_indptr[local_row])
                local_right = int(local_indptr[local_row + 1])
                for feature, value in zip(
                    indices[local_left:local_right], data[local_left:local_right]
                ):
                    destination = column_lookup.get(int(feature))
                    if destination is not None:
                        output[row_offset + local_row, destination] = value
            row_offset += stop - start
            data_slices += 1
    if row_offset != len(rows):
        raise RuntimeError("selective CSR reader lost a selected row")
    if (
        not np.isfinite(output).all()
        or np.any(output < 0.0)
        or not np.array_equal(output, np.rint(output))
    ):
        raise ValueError("selected marker matrix is not nonnegative integer counts")
    return output.astype(np.int64), {
        "matrix_path": matrix_path,
        "rows_read": len(rows),
        "row_index_sha256": _array_sha256(rows.astype(np.int64)),
        "marker_columns": columns.tolist(),
        "contiguous_data_slices": data_slices,
        "full_indptr_read": False,
        "permitted_indptr_slices": data_slices,
        "held_rows_read": 0,
        "x_opened": False,
        "raw_x_opened": False,
    }


def _rank_binary_adt(counts: np.ndarray, cells: list[dict[str, str]]) -> np.ndarray:
    values = np.asarray(counts)
    states = np.empty(values.shape, dtype=np.int8)
    strata: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(cells):
        strata.setdefault((row["DonorID"], row["batch"]), []).append(index)
    for (donor, batch), members in sorted(strata.items()):
        selected = np.asarray(members, dtype=int)
        for marker, name in enumerate(lock.MARKERS):
            ties = np.asarray(
                [
                    _seed("adt-rank", donor, batch, cells[index]["barcode"], name)
                    for index in selected
                ],
                dtype=np.uint64,
            )
            order = np.lexsort((ties, values[selected, marker]))
            states[selected[order[: len(selected) // 2]], marker] = 0
            states[selected[order[len(selected) // 2 :]], marker] = 1
    return states


def _tables(
    rna: np.ndarray,
    adt: np.ndarray,
    cells: list[dict[str, str]],
    group_key: str,
) -> tuple[list[str], np.ndarray]:
    groups = sorted({row[group_key] for row in cells})
    group_axis = np.asarray([row[group_key] for row in cells])
    table = np.empty((len(groups), 10, 10, 2, 2), dtype=np.int64)
    for group_index, group in enumerate(groups):
        mask = group_axis == group
        for first, second in np.ndindex(10, 10):
            table[group_index, first, second] = np.bincount(
                2 * rna[mask, first] + adt[mask, second], minlength=4
            ).reshape(2, 2)
    return groups, table


def _prevalence_incidence(
    states: np.ndarray,
    cells: list[dict[str, str]],
    neighbors: int,
) -> np.ndarray:
    strata = sorted({(row["DonorID"], row["cell_type.l1"]) for row in cells})
    donor = np.asarray([row["DonorID"] for row in cells])
    lineage = np.asarray([row["cell_type.l1"] for row in cells])
    profiles = np.asarray(
        [
            [
                states[
                    (donor == current_donor) & (lineage == current_lineage), marker
                ].mean()
                for current_donor, current_lineage in strata
            ]
            for marker in range(10)
        ]
    )
    incidence = np.zeros((10, 10), dtype=float)
    names = np.asarray(lock.MARKERS)
    for marker in range(10):
        distance = np.linalg.norm(profiles - profiles[marker], axis=1)
        candidates = np.flatnonzero(np.arange(10) != marker)
        order = candidates[np.lexsort((names[candidates], distance[candidates]))]
        incidence[marker, marker] = 1.0
        incidence[order[:neighbors], marker] = 1.0
    return incidence


def _permuted_incidence(incidence: np.ndarray, label: str) -> np.ndarray:
    generator = np.random.default_rng(_seed("graph-label-permutation", label))
    return incidence[generator.permutation(incidence.shape[0])]


def _destroy_links(adt: np.ndarray, cells: list[dict[str, str]]) -> np.ndarray:
    destroyed = np.empty_like(adt)
    strata: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(cells):
        strata.setdefault((row["DonorID"], row["cell_type.l1"]), []).append(index)
    for (donor, lineage), members in sorted(strata.items()):
        selected = np.asarray(members, dtype=int)
        generator = np.random.default_rng(_seed("destroy", donor, lineage))
        destroyed[selected] = adt[generator.permutation(selected)]
    return destroyed


def _informative(tables: np.ndarray) -> np.ndarray:
    rows = tables.sum(axis=-1)
    columns = tables.sum(axis=-2)
    total = tables.sum(axis=(-2, -1))
    lower = np.maximum(0, rows[..., 0] + columns[..., 0] - total)
    upper = np.minimum(rows[..., 0], columns[..., 0])
    return upper > lower


def _deviance(truth: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    if truth.shape != predicted.shape:
        raise ValueError("truth and prediction shapes differ")
    if not np.isfinite(predicted).all() or np.any((truth > 0) & (predicted <= 0.0)):
        raise FloatingPointError("prediction assigns no mass to an observed cell")
    terms = np.zeros_like(predicted, dtype=float)
    positive = truth > 0
    terms[positive] = truth[positive] * np.log(truth[positive] / predicted[positive])
    return 2.0 * terms.sum(axis=(-2, -1)) / truth.sum(axis=(-2, -1))


def _losses(truth: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    support = _informative(truth)
    counts = support.reshape(len(truth), -1).sum(axis=1)
    if np.any(counts < MINIMUM_DEVELOPMENT_ENTITIES):
        raise ValueError("a development batch has fewer than 80 informative entities")
    entity_loss = _deviance(truth, predicted)
    return np.asarray(
        [entity_loss[index][support[index]].mean() for index in range(len(truth))]
    )


def _predict_log_odds(
    log_odds: np.ndarray, target: np.ndarray, alpha: float
) -> np.ndarray:
    predicted = np.empty(target.shape, dtype=float)
    rows = target.sum(axis=-1)
    columns = target.sum(axis=-2)
    for index in np.ndindex(target.shape[:-2]):
        predicted[index] = expected_binary_table_from_log_odds(
            alpha * float(log_odds[index[-2:]]), rows[index], columns[index]
        )
    return predicted


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


def _classical_table(
    coordinate: float,
    rows: np.ndarray,
    columns: np.ndarray,
    family: str,
) -> np.ndarray:
    total = float(rows.sum())
    lower = float(max(0, rows[0] + columns[0] - total))
    upper = float(min(rows[0], columns[0]))
    if upper <= lower:
        return _canonical_table(rows, columns).astype(float)
    epsilon = min(1e-10, 0.25 * (upper - lower))
    if family == "pearson":
        scale = math.sqrt(total / float(rows[0] * rows[1] * columns[0] * columns[1]))
        upper_left = (coordinate / scale + rows[0] * columns[0]) / total
    elif family == "deviance":
        expected = np.outer(rows, columns) / total

        def statistic(value: float) -> float:
            table = np.asarray(
                [
                    [value, rows[0] - value],
                    [columns[0] - value, rows[1] - columns[0] + value],
                ]
            )
            positive = table > 0.0
            terms = np.zeros((2, 2), dtype=float)
            terms[positive] = table[positive] * np.log(
                table[positive] / expected[positive]
            )
            determinant = table[0, 0] * table[1, 1] - table[0, 1] * table[1, 0]
            return math.copysign(
                math.sqrt(max(2.0 * float(terms.sum()), 0.0)), determinant
            )

        left = lower + epsilon
        right = upper - epsilon
        target = min(max(coordinate, statistic(left)), statistic(right))
        for _ in range(80):
            midpoint = 0.5 * (left + right)
            if statistic(midpoint) < target:
                left = midpoint
            else:
                right = midpoint
        upper_left = 0.5 * (left + right)
    else:
        raise ValueError("classical family must be pearson or deviance")
    upper_left = min(max(float(upper_left), lower + epsilon), upper - epsilon)
    return np.asarray(
        [
            [upper_left, rows[0] - upper_left],
            [columns[0] - upper_left, rows[1] - columns[0] + upper_left],
        ]
    )


def _residual_coordinate(tables: np.ndarray, family: str, centered: bool) -> np.ndarray:
    values = np.empty(tables.shape[:3], dtype=float)
    for index in np.ndindex(tables.shape[:3]):
        table = tables[index]
        if centered:
            value = centered_classical_coordinate(table, statistic=family)
            coordinate = value.centered_coordinate
        else:
            function = (
                signed_pearson_coordinate
                if family == "pearson"
                else signed_deviance_coordinate
            )
            coordinate = function(table)
        values[index] = coordinate / math.sqrt(float(table.sum()))
    return values.mean(axis=0)


def _predict_residual(
    coordinate: np.ndarray,
    target: np.ndarray,
    *,
    family: str,
    centered: bool,
    alpha: float,
) -> np.ndarray:
    predicted = np.empty(target.shape, dtype=float)
    rows = target.sum(axis=-1)
    columns = target.sum(axis=-2)
    for index in np.ndindex(target.shape[:-2]):
        statistic = (
            alpha * coordinate[index[-2:]] * math.sqrt(float(target[index].sum()))
        )
        if centered:
            null = centered_classical_coordinate(
                _canonical_table(rows[index], columns[index]), statistic=family
            )
            statistic += null.null_mean_coordinate
        predicted[index] = _classical_table(
            float(statistic), rows[index], columns[index], family
        )
    return predicted


def _predict_independence(target: np.ndarray) -> np.ndarray:
    rows = target.sum(axis=-1)
    columns = target.sum(axis=-2)
    total = target.sum(axis=(-2, -1))
    return rows[..., :, None] * columns[..., None, :] / total[..., None, None]


def _hierarchical_fit(
    tables: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    heterogeneity: float,
    ridge: float,
    graph: float,
):
    return fit_hierarchical_conditional_log_odds(
        tables,
        first,
        second,
        heterogeneity_penalty=heterogeneity,
        ridge_penalty=ridge,
        graph_penalty=graph,
        minimum_informative_donors=2,
        maximum_condition_number=MAXIMUM_CONDITION_NUMBER,
        tolerance=TOLERANCE,
    )


def _common_fit(
    tables: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    ridge: float,
    graph: float,
):
    fit = fit_structured_conditional_log_odds(
        tables,
        first,
        second,
        initial_log_odds=np.zeros(tables.shape[1:3], dtype=float),
        ridge_penalty=ridge,
        graph_penalty=graph,
        minimum_informative_donors=2,
        maximum_condition_number=MAXIMUM_CONDITION_NUMBER,
        tolerance=1e-9,
    )
    threshold = 1e-7 + 1e-10 * float(tables.sum(axis=(-2, -1)).max())
    if fit.gradient_norm > threshold:
        raise CouplingEstimationRefusal(
            "common-effect fit misses the evaluator gradient certificate"
        )
    return fit


def _select_hierarchical(
    fit_tables: np.ndarray,
    development_tables: np.ndarray,
    incidences: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    graph_values: tuple[float, ...],
) -> tuple[dict[str, float | int], np.ndarray, list[dict[str, object]]]:
    candidates: list[dict[str, object]] = []
    for neighbors in NEIGHBOR_GRID:
        first, second = incidences[neighbors]
        for heterogeneity in HETEROGENEITY_GRID:
            for ridge in RIDGE_GRID:
                for graph in graph_values:
                    try:
                        fit = _hierarchical_fit(
                            fit_tables,
                            first,
                            second,
                            heterogeneity=heterogeneity,
                            ridge=ridge,
                            graph=graph,
                        )
                    except (CouplingEstimationRefusal, FloatingPointError) as error:
                        candidates.append(
                            {
                                "neighbors": neighbors,
                                "heterogeneity": heterogeneity,
                                "ridge": ridge,
                                "graph": graph,
                                "status": "REFUSED",
                                "reason": str(error),
                            }
                        )
                        continue
                    for alpha in ALPHA_GRID:
                        losses = _losses(
                            development_tables,
                            _predict_log_odds(
                                fit.population_log_odds, development_tables, alpha
                            ),
                        )
                        candidates.append(
                            {
                                "neighbors": neighbors,
                                "heterogeneity": heterogeneity,
                                "ridge": ridge,
                                "graph": graph,
                                "alpha": alpha,
                                "status": "OK",
                                "mean_loss": float(losses.mean()),
                                "batch_losses": losses.tolist(),
                            }
                        )
    valid = [candidate for candidate in candidates if candidate["status"] == "OK"]
    if not valid:
        raise CouplingEstimationRefusal("all hierarchical candidates refused")
    selected = min(
        valid,
        key=lambda value: (
            value["mean_loss"],
            value["neighbors"],
            value["heterogeneity"],
            value["ridge"],
            value["graph"],
            value["alpha"],
        ),
    )
    settings = {
        key: selected[key]
        for key in ("neighbors", "heterogeneity", "ridge", "graph", "alpha")
    }
    return settings, np.asarray(selected["batch_losses"]), candidates


def _select_common(
    fit_tables: np.ndarray,
    development_tables: np.ndarray,
    incidences: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    graph_values: tuple[float, ...],
) -> tuple[dict[str, float | int], np.ndarray, list[dict[str, object]]]:
    candidates: list[dict[str, object]] = []
    for neighbors in NEIGHBOR_GRID:
        first, second = incidences[neighbors]
        for ridge in RIDGE_GRID:
            for graph in graph_values:
                try:
                    fit = _common_fit(
                        fit_tables, first, second, ridge=ridge, graph=graph
                    )
                except (CouplingEstimationRefusal, FloatingPointError) as error:
                    candidates.append(
                        {
                            "neighbors": neighbors,
                            "ridge": ridge,
                            "graph": graph,
                            "status": "REFUSED",
                            "reason": str(error),
                        }
                    )
                    continue
                for alpha in ALPHA_GRID:
                    losses = _losses(
                        development_tables,
                        _predict_log_odds(fit.log_odds, development_tables, alpha),
                    )
                    candidates.append(
                        {
                            "neighbors": neighbors,
                            "ridge": ridge,
                            "graph": graph,
                            "alpha": alpha,
                            "status": "OK",
                            "mean_loss": float(losses.mean()),
                            "batch_losses": losses.tolist(),
                        }
                    )
    valid = [candidate for candidate in candidates if candidate["status"] == "OK"]
    if not valid:
        raise CouplingEstimationRefusal("all common-effect candidates refused")
    selected = min(
        valid,
        key=lambda value: (
            value["mean_loss"],
            value["neighbors"],
            value["ridge"],
            value["graph"],
            value["alpha"],
        ),
    )
    settings = {key: selected[key] for key in ("neighbors", "ridge", "graph", "alpha")}
    return settings, np.asarray(selected["batch_losses"]), candidates


def _select_residual(
    fit_tables: np.ndarray, development_tables: np.ndarray
) -> tuple[dict[str, object], np.ndarray, list[dict[str, object]]]:
    candidates: list[dict[str, object]] = []
    for family in ("pearson", "deviance"):
        for centered in (False, True):
            coordinate = _residual_coordinate(fit_tables, family, centered)
            for alpha in ALPHA_GRID:
                losses = _losses(
                    development_tables,
                    _predict_residual(
                        coordinate,
                        development_tables,
                        family=family,
                        centered=centered,
                        alpha=alpha,
                    ),
                )
                candidates.append(
                    {
                        "family": family,
                        "centered": centered,
                        "alpha": alpha,
                        "mean_loss": float(losses.mean()),
                        "batch_losses": losses.tolist(),
                    }
                )
    selected = min(
        candidates,
        key=lambda value: (
            value["mean_loss"],
            value["family"],
            value["centered"],
            value["alpha"],
        ),
    )
    settings = {key: selected[key] for key in ("family", "centered", "alpha")}
    return settings, np.asarray(selected["batch_losses"]), candidates


def _hierarchical_certificate(fit) -> dict[str, object]:
    return {
        "certificate_family": "hierarchical_exact_block_newton",
        "converged": bool(fit.converged),
        "objective": float(fit.objective),
        "gradient_norm": float(fit.gradient_norm),
        "scaled_gradient_norm": float(fit.scaled_gradient_norm),
        "gradient_tolerance": float(fit.gradient_tolerance),
        "schur_condition_number": float(fit.schur_condition_number),
        "theta_curvature_condition_number": float(fit.theta_curvature_condition_number),
        "iterations": int(fit.iterations),
    }


def _common_certificate(fit, tables: np.ndarray) -> dict[str, object]:
    threshold = 1e-7 + 1e-10 * float(tables.sum(axis=(-2, -1)).max())
    return {
        "certificate_family": "common_effect_exact_hessian",
        "converged": bool(fit.converged),
        "objective": float(fit.objective),
        "gradient_norm": float(fit.gradient_norm),
        "scaled_gradient_norm": float(fit.gradient_norm),
        "gradient_tolerance": threshold,
        "schur_condition_number": float(fit.condition_number),
        "theta_curvature_condition_number": 1.0,
        "iterations": int(fit.iterations),
    }


def _validated_software_audit() -> dict[str, object]:
    if not SOFTWARE_AUDIT.is_file():
        raise PermissionError("hierarchical estimator software audit is absent")
    audit = json.loads(SOFTWARE_AUDIT.read_text())
    module = ROOT / "mapreg/hierarchical_conditional_coupling.py"
    tests = ROOT / "tests/test_hierarchical_conditional_coupling.py"
    if (
        audit.get("status") != "PASS"
        or audit.get("module_sha256") != _sha256(module)
        or audit.get("test_sha256") != _sha256(tests)
        or audit.get("warnings_as_errors") is not True
    ):
        raise PermissionError("hierarchical estimator software audit differs")
    return audit


def _comparison(
    batch_labels: list[str],
    primary: np.ndarray,
    comparator: np.ndarray,
    *,
    required_favorable: int,
) -> dict[str, object]:
    difference = primary - comparator
    relative = 1.0 - float(primary.mean() / comparator.mean())
    favorable = int(np.count_nonzero(difference < 0.0))
    return {
        "primary_mean_loss": float(primary.mean()),
        "comparator_mean_loss": float(comparator.mean()),
        "relative_reduction": relative,
        "batch_differences_primary_minus_comparator": {
            label: float(value) for label, value in zip(batch_labels, difference)
        },
        "favorable_batches": favorable,
        "required_favorable_batches": required_favorable,
        "passes_five_percent": bool(relative >= 0.05),
        "passes_favorable_count": bool(favorable >= required_favorable),
        "passes": bool(relative >= 0.05 and favorable >= required_favorable),
    }


def run_development() -> dict[str, object]:
    if OUTPUT.exists():
        raise FileExistsError("BMMC development result already exists")
    software_audit = _validated_software_audit()
    source, paths = lock._validated_source()
    metadata = _metadata(paths["metadata"])
    _, roles = lock._metadata_roles(paths["metadata"])
    axis = lock._axis(paths["complete_cite_h5ad"], source["combined_assay"])
    role_rows = lock._row_vectors(axis, roles)
    selected_rows = np.sort(np.r_[role_rows["fit"], role_rows["development"]])
    forbidden_rows = role_rows["held"]
    if np.intersect1d(selected_rows, forbidden_rows).size:
        raise PermissionError("non-held and held CSR rows overlap")
    columns = np.asarray(
        axis["marker_indices"]["rna"] + axis["marker_indices"]["adt"],
        dtype=int,
    )
    counts, read_audit = _read_csr_marker_counts(
        paths["complete_cite_h5ad"],
        matrix_path=source["combined_assay"]["matrix_hdf5_path"],
        selected_rows=selected_rows,
        forbidden_rows=forbidden_rows,
        selected_columns=columns,
    )
    metadata_by_barcode = {row["barcode"]: row for row in metadata}
    selected_cells = [
        metadata_by_barcode[axis["barcodes"][index]] for index in selected_rows
    ]
    if any(row["DonorID"] in lock.HELD_DONORS for row in selected_cells):
        raise PermissionError("a held donor entered non-held development")
    rna_states = (counts[:, :10] > 0).astype(np.int8)
    adt_states = _rank_binary_adt(counts[:, 10:], selected_cells)
    donor_axis = np.asarray([row["DonorID"] for row in selected_cells])
    fit_mask = np.isin(donor_axis, lock.FIT_DONORS)
    development_mask = donor_axis == lock.DEVELOPMENT_DONORS[0]
    if np.count_nonzero(fit_mask) != 1540 or np.count_nonzero(development_mask) != 3067:
        raise ValueError("non-held cell counts differ from the locked preflight")
    fit_cells = [row for row, keep in zip(selected_cells, fit_mask) if keep]
    development_cells = [
        row for row, keep in zip(selected_cells, development_mask) if keep
    ]
    fit_donors, fit_tables = _tables(
        rna_states[fit_mask], adt_states[fit_mask], fit_cells, "DonorID"
    )
    development_batches, development_tables = _tables(
        rna_states[development_mask],
        adt_states[development_mask],
        development_cells,
        "batch",
    )
    if fit_donors != list(lock.FIT_DONORS) or development_batches != [
        "s1d1",
        "s2d1",
        "s3d1",
        "s4d1",
    ]:
        raise ValueError("fit donor or development batch axis differs")
    incidences = {
        neighbors: (
            _prevalence_incidence(rna_states[fit_mask], fit_cells, neighbors),
            _prevalence_incidence(adt_states[fit_mask], fit_cells, neighbors),
        )
        for neighbors in NEIGHBOR_GRID
    }
    primary_settings, primary_losses, primary_candidates = _select_hierarchical(
        fit_tables,
        development_tables,
        incidences,
        graph_values=GRAPH_GRID,
    )
    ridge_settings, ridge_losses, ridge_candidates = _select_hierarchical(
        fit_tables,
        development_tables,
        incidences,
        graph_values=(0.0,),
    )
    common_graph_settings, common_graph_losses, common_graph_candidates = (
        _select_common(
            fit_tables,
            development_tables,
            incidences,
            graph_values=GRAPH_GRID,
        )
    )
    common_ridge_settings, common_ridge_losses, common_ridge_candidates = (
        _select_common(
            fit_tables,
            development_tables,
            incidences,
            graph_values=(0.0,),
        )
    )
    residual_settings, residual_losses, residual_candidates = _select_residual(
        fit_tables, development_tables
    )
    primary_incidence = incidences[int(primary_settings["neighbors"])]
    destroyed_adt = _destroy_links(adt_states[fit_mask], fit_cells)
    _, destroyed_fit_tables = _tables(
        rna_states[fit_mask], destroyed_adt, fit_cells, "DonorID"
    )
    destroyed_fit = _hierarchical_fit(
        destroyed_fit_tables,
        *primary_incidence,
        heterogeneity=float(primary_settings["heterogeneity"]),
        ridge=float(primary_settings["ridge"]),
        graph=float(primary_settings["graph"]),
    )
    destroyed_losses = _losses(
        development_tables,
        _predict_log_odds(
            destroyed_fit.population_log_odds,
            development_tables,
            float(primary_settings["alpha"]),
        ),
    )
    permuted_incidence = (
        _permuted_incidence(primary_incidence[0], "rna"),
        _permuted_incidence(primary_incidence[1], "adt"),
    )
    permuted_fit = _hierarchical_fit(
        fit_tables,
        *permuted_incidence,
        heterogeneity=float(primary_settings["heterogeneity"]),
        ridge=float(primary_settings["ridge"]),
        graph=float(primary_settings["graph"]),
    )
    permuted_losses = _losses(
        development_tables,
        _predict_log_odds(
            permuted_fit.population_log_odds,
            development_tables,
            float(primary_settings["alpha"]),
        ),
    )
    independence_losses = _losses(
        development_tables, _predict_independence(development_tables)
    )
    development_losses = {
        "primary": primary_losses,
        "best_residual": residual_losses,
        "destroyed_link": destroyed_losses,
        "hierarchical_ridge_only": ridge_losses,
        "common_effect_graph": common_graph_losses,
        "common_effect_ridge_only": common_ridge_losses,
        "label_permuted_graph": permuted_losses,
        "independence": independence_losses,
    }
    comparisons = {
        name: _comparison(
            development_batches,
            primary_losses,
            development_losses[name],
            required_favorable=4 if name == "best_residual" else 3,
        )
        for name in lock.GATE_COMPARATORS
    }
    passes_all = all(value["passes"] for value in comparisons.values())

    nonheld_donors, nonheld_tables = _tables(
        rna_states, adt_states, selected_cells, "DonorID"
    )
    if nonheld_donors != ["11466", "15078", "19593"]:
        raise ValueError("non-held refit donor axis differs")
    final_primary = _hierarchical_fit(
        nonheld_tables,
        *primary_incidence,
        heterogeneity=float(primary_settings["heterogeneity"]),
        ridge=float(primary_settings["ridge"]),
        graph=float(primary_settings["graph"]),
    )
    ridge_incidence = incidences[int(ridge_settings["neighbors"])]
    final_ridge = _hierarchical_fit(
        nonheld_tables,
        *ridge_incidence,
        heterogeneity=float(ridge_settings["heterogeneity"]),
        ridge=float(ridge_settings["ridge"]),
        graph=0.0,
    )
    common_graph_incidence = incidences[int(common_graph_settings["neighbors"])]
    final_common_graph = _common_fit(
        nonheld_tables,
        *common_graph_incidence,
        ridge=float(common_graph_settings["ridge"]),
        graph=float(common_graph_settings["graph"]),
    )
    common_ridge_incidence = incidences[int(common_ridge_settings["neighbors"])]
    final_common_ridge = _common_fit(
        nonheld_tables,
        *common_ridge_incidence,
        ridge=float(common_ridge_settings["ridge"]),
        graph=0.0,
    )
    destroyed_nonheld_adt = _destroy_links(adt_states, selected_cells)
    _, destroyed_nonheld_tables = _tables(
        rna_states, destroyed_nonheld_adt, selected_cells, "DonorID"
    )
    final_destroyed = _hierarchical_fit(
        destroyed_nonheld_tables,
        *primary_incidence,
        heterogeneity=float(primary_settings["heterogeneity"]),
        ridge=float(primary_settings["ridge"]),
        graph=float(primary_settings["graph"]),
    )
    final_permuted = _hierarchical_fit(
        nonheld_tables,
        *permuted_incidence,
        heterogeneity=float(primary_settings["heterogeneity"]),
        ridge=float(primary_settings["ridge"]),
        graph=float(primary_settings["graph"]),
    )
    final_residual_coordinate = _residual_coordinate(
        nonheld_tables,
        str(residual_settings["family"]),
        bool(residual_settings["centered"]),
    )

    def conditional_method(fit, settings, tables, coordinate) -> dict[str, object]:
        certificate = (
            _hierarchical_certificate(fit)
            if hasattr(fit, "population_log_odds")
            else _common_certificate(fit, tables)
        )
        return {
            "kind": "conditional_log_odds",
            "alpha": float(settings["alpha"]),
            "settings": settings,
            "source_coordinate": np.asarray(coordinate).ravel(order="C").tolist(),
            "numerical_certificate": certificate,
        }

    methods = {
        "primary": conditional_method(
            final_primary,
            primary_settings,
            nonheld_tables,
            final_primary.population_log_odds,
        ),
        "best_residual": {
            "kind": "classical_residual",
            "family": residual_settings["family"],
            "centered": residual_settings["centered"],
            "sample_size_normalized": True,
            "alpha": float(residual_settings["alpha"]),
            "source_coordinate": final_residual_coordinate.ravel(order="C").tolist(),
        },
        "destroyed_link": conditional_method(
            final_destroyed,
            primary_settings,
            destroyed_nonheld_tables,
            final_destroyed.population_log_odds,
        ),
        "hierarchical_ridge_only": conditional_method(
            final_ridge,
            ridge_settings,
            nonheld_tables,
            final_ridge.population_log_odds,
        ),
        "common_effect_graph": conditional_method(
            final_common_graph,
            common_graph_settings,
            nonheld_tables,
            final_common_graph.log_odds,
        ),
        "common_effect_ridge_only": conditional_method(
            final_common_ridge,
            common_ridge_settings,
            nonheld_tables,
            final_common_ridge.log_odds,
        ),
        "label_permuted_graph": conditional_method(
            final_permuted,
            primary_settings,
            nonheld_tables,
            final_permuted.population_log_odds,
        ),
        "independence": {"kind": "independence"},
    }
    payload = {
        "schema": "scmmib-bmmc-exact-development/1.0",
        "status": "DEVELOPMENT_PASS" if passes_all else "DEVELOPMENT_FAIL",
        "source_manifest_sha256": _sha256(lock.SOURCE_MANIFEST),
        "evaluator_sha256": _sha256(Path(__file__)),
        "protocol_sha256": _sha256(lock.PROTOCOL),
        "split": {
            "fit_donors": list(lock.FIT_DONORS),
            "development_donors": list(lock.DEVELOPMENT_DONORS),
            "held_donors": list(lock.HELD_DONORS),
            "original_is_train_used": False,
            "physical_donor_disjoint": True,
            "site_disjoint": False,
        },
        "markers": list(lock.MARKERS),
        "entity_count": 100,
        "development_batches": development_batches,
        "development_losses": {
            name: {
                label: float(value) for label, value in zip(development_batches, losses)
            }
            for name, losses in development_losses.items()
        },
        "gate": {
            "screening_not_confirmatory_inference": True,
            "required_relative_reduction": 0.05,
            "comparisons": comparisons,
            "passes_all": passes_all,
        },
        "selected_settings": {
            "primary": primary_settings,
            "best_residual": residual_settings,
            "hierarchical_ridge_only": ridge_settings,
            "common_effect_graph": common_graph_settings,
            "common_effect_ridge_only": common_ridge_settings,
        },
        "candidate_diagnostics": {
            "primary": primary_candidates,
            "best_residual": residual_candidates,
            "hierarchical_ridge_only": ridge_candidates,
            "common_effect_graph": common_graph_candidates,
            "common_effect_ridge_only": common_ridge_candidates,
        },
        "frozen_source_model": {
            "entity_order": "row-major RNA marker x ADT marker",
            "methods": methods,
            "graph": {
                "built_from_fit_donors_only": True,
                "development_or_held_outcomes_used": False,
                "selected_neighbors": int(primary_settings["neighbors"]),
                "rna_incidence": primary_incidence[0].tolist(),
                "adt_incidence": primary_incidence[1].tolist(),
                "rna_incidence_sha256": _array_sha256(primary_incidence[0]),
                "adt_incidence_sha256": _array_sha256(primary_incidence[1]),
                "label_permuted_rna_incidence_sha256": _array_sha256(
                    permuted_incidence[0]
                ),
                "label_permuted_adt_incidence_sha256": _array_sha256(
                    permuted_incidence[1]
                ),
            },
        },
        "software_audit": {
            **software_audit,
            "hierarchical_tests_passed": True,
        },
        "access_audit": {
            **read_audit,
            "fit_feature_rows_read": len(role_rows["fit"]),
            "development_feature_rows_read": len(role_rows["development"]),
            "held_feature_rows_forbidden": len(role_rows["held"]),
            "held_feature_rows_read": 0,
            "held_margins_computed": 0,
            "held_tables_formed": 0,
            "held_outcomes_used_for_selection": False,
            "covid_or_sanger_path_accessed": False,
        },
    }
    _write_json_exclusive(OUTPUT, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = run_development()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
