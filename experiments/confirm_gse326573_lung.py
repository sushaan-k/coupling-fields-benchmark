"""Staged held-batch confirmation on the GSE326573 lung CITE-seq study.

``source`` may read only batches 1--3. ``predict`` reads held RNA entries and
freezes fixed-margin predictions. ``score`` is the first stage allowed to read
held ADT entries. Every stage is one shot and binds its predecessor by SHA-256.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import math
import os
import platform
from pathlib import Path
import shutil
import stat
import sys
import tarfile
import tempfile
from typing import Any, Iterable, Optional, Union

import h5py
import numpy as np
import scipy

from mapreg.heterogeneity_adaptive_coupling import (
    CouplingEstimationRefusal,
    expected_binary_table_from_log_odds,
    fit_structured_conditional_log_odds,
    signed_deviance_coordinate,
    signed_pearson_coordinate,
)
from mapreg.hierarchical_conditional_coupling import (
    fit_hierarchical_conditional_log_odds,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/confirmation/gse326573_lung"
DEFAULT_ARCHIVE = Path("/private/tmp/GSE326573_RAW.tar")
DEFAULT_DESIGNATION = DATA_DIR / "candidate_designation_v1.json"
DEFAULT_PREFLIGHT = ROOT / "results/development/gse326573_axis_preflight_v1.json"
DEFAULT_PROTOCOL = DATA_DIR / "protocol_v1.json"
DEFAULT_RUNTIME = DATA_DIR / "runtime_environment_v1.json"
DEFAULT_SOURCE_ATTEMPT = DATA_DIR / "source_attempt_v1.json"
DEFAULT_PREDICTION_ATTEMPT = DATA_DIR / "prediction_attempt_v1.json"
DEFAULT_SCORE_AUTHORIZATION = DATA_DIR / "score_authorization_v1.json"
DEFAULT_SCORE_ATTEMPT = DATA_DIR / "score_attempt_v1.json"
DEFAULT_SOURCE = ROOT / "results/development/gse326573_lung_source_v1.json"
DEFAULT_PREDICTION = ROOT / "results/gse326573_lung_predictions_v1.json"
DEFAULT_SCORE = ROOT / "results/gse326573_lung_confirmation_v1.json"
DEFAULT_PRIVATE_RNA = DATA_DIR / "private_held_rna_states_v1.npz"

CELL_BUDGET = 512
MARKER_COUNT = 11
SOURCE_BATCHES = ("Batch1", "Batch2", "Batch3")
HELD_BATCHES = ("Batch4", "Batch5", "Batch6")
CELL_SALT = "GSE326573-CELL-BUDGET-v1"
ADT_TIE_SALT = "GSE326573-ADT-MIDRANK-v1"
DESTROYED_SALT = "GSE326573-DESTROYED-LINK-v1"
BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 20260829
SENSITIVITY_BOOTSTRAP_SEED = 20260830
MINIMUM_INFORMATIVE_ENTITIES = 64
MAXIMUM_CONDITION_NUMBER = 1e12

NEIGHBOR_GRID = (2,)
HETEROGENEITY_GRID = (0.1, 1.0, 10.0)
RIDGE_GRID = (0.01, 0.1)
GRAPH_GRID = (0.0, 0.03, 0.3)
TRANSPORT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
RESIDUAL_FAMILIES = ("pearson", "root_deviance")


@dataclass(frozen=True, order=True)
class PrimaryConfig:
    graph_neighbors: int
    heterogeneity_penalty: float
    ridge_penalty: float
    graph_penalty: float
    transport_multiplier: float


@dataclass(frozen=True, order=True)
class ResidualConfig:
    family: str
    transport_multiplier: float


@dataclass(frozen=True, order=True)
class OddsConfig:
    method: str
    transport_multiplier: float


def _timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _axis_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _write_json_x(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _claim(path: Path, phase: str, bindings: dict[str, str]) -> None:
    _write_json_x(
        path,
        {
            "schema": "gse326573-lung-attempt/1.0",
            "status": "CLAIMED_ONE_SHOT",
            "phase": phase,
            "created_at_utc": _timestamp(),
            "bindings": bindings,
        },
    )


def _validate_attempt(path: Path, phase: str, bindings: dict[str, str]) -> None:
    attempt = _read_json(path)
    if (
        attempt.get("schema") != "gse326573-lung-attempt/1.0"
        or attempt.get("status") != "CLAIMED_ONE_SHOT"
        or attempt.get("phase") != phase
        or attempt.get("bindings") != bindings
    ):
        raise PermissionError(f"{phase} attempt is absent or bound to different bytes")


def _validate_runtime(path: Path) -> None:
    value = _read_json(path)
    if (
        value.get("schema") != "gse326573-lung-runtime-environment/1.0"
        or value.get("status") != "FROZEN_WITH_PROTOCOL"
    ):
        raise PermissionError("runtime contract is not frozen")
    required = value["required_runtime"]
    python = required["python"]
    packages = required["packages"]
    hdf5 = required["hdf5"]
    system = "macOS" if platform.system() == "Darwin" else platform.system()
    checks = {
        "python implementation": platform.python_implementation()
        == python["implementation"],
        "python version": platform.python_version() == python["version"],
        "python executable": str(Path(sys.executable).resolve())
        == python["resolved_executable"],
        "numpy": np.__version__ == packages["numpy"],
        "scipy": scipy.__version__ == packages["scipy"],
        "h5py": h5py.__version__ == packages["h5py"],
        "HDF5": list(h5py.version.hdf5_version_tuple) == hdf5["runtime_version_tuple"],
        "h5py API": list(h5py.version.api_version_tuple)
        == hdf5["h5py_api_version_tuple"],
        "operating system": system == required["platform"]["operating_system"],
        "architecture": platform.machine() == required["platform"]["architecture"],
    }
    for name, expected in required["thread_environment"].items():
        checks[f"environment {name}"] = os.environ.get(name) == expected
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise PermissionError("runtime differs at " + ", ".join(failures))


def _base_bindings(
    archive_path: Path,
    designation_path: Path,
    preflight_path: Path,
    protocol_path: Path,
    runtime_path: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    designation, _, _ = _designation(designation_path)
    _validate_preflight(preflight_path, designation)
    _validate_runtime(runtime_path)
    protocol = _read_json(protocol_path)
    if (
        protocol.get("schema") != "gse326573-lung-citeseq-held-batch-protocol/1.0"
        or protocol.get("status")
        != "FROZEN_AFTER_AXIS_ONLY_PREFLIGHT_BEFORE_NUMERIC_MATRIX_ACCESS"
    ):
        raise PermissionError("analysis protocol is not frozen pre-outcome")
    candidate_freeze = protocol["candidate_freeze"]
    if (
        candidate_freeze.get("candidate_sha256") != _sha256(designation_path)
        or candidate_freeze.get("axis_preflight_sha256") != _sha256(preflight_path)
        or candidate_freeze.get("numeric_count_values_read_before_freeze") != 0
    ):
        raise PermissionError(
            "protocol does not bind the candidate and preflight bytes"
        )
    archive_sha256 = _verify_archive(archive_path, designation)
    if protocol["official_archive"].get("sha256") != archive_sha256:
        raise PermissionError("protocol does not bind the official archive")
    bound_code = {
        "gitignore_sha256": _sha256(ROOT / ".gitignore"),
        "runner_sha256": _sha256(Path(__file__)),
        "protocol_document_sha256": _sha256(
            ROOT / "docs/GSE326573_LUNG_CITESEQ_HELD_BATCH_PROTOCOL_2026-08-29.md"
        ),
        "coupling_fields_module_sha256": _sha256(ROOT / "mapreg/coupling_fields.py"),
        "mapreg_init_sha256": _sha256(ROOT / "mapreg/__init__.py"),
        "factorial_coupling_module_sha256": _sha256(
            ROOT / "mapreg/factorial_coupling.py"
        ),
        "classical_residuals_module_sha256": _sha256(
            ROOT / "mapreg/classical_residuals.py"
        ),
        "table_prediction_module_sha256": _sha256(ROOT / "mapreg/table_prediction.py"),
        "hierarchical_module_sha256": _sha256(
            ROOT / "mapreg/hierarchical_conditional_coupling.py"
        ),
        "coupling_module_sha256": _sha256(
            ROOT / "mapreg/heterogeneity_adaptive_coupling.py"
        ),
        "test_sha256": _sha256(ROOT / "tests/test_gse326573_lung_confirmation.py"),
    }
    return designation, {
        "candidate_sha256": _sha256(designation_path),
        "preflight_sha256": _sha256(preflight_path),
        "protocol_sha256": _sha256(protocol_path),
        "runtime_sha256": _sha256(runtime_path),
        "archive_sha256": archive_sha256,
        **bound_code,
    }


def _decode(values: np.ndarray) -> list[str]:
    return [
        value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


def _designation(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    value = _read_json(path)
    if (
        value.get("schema") != "gse326573-lung-citeseq-candidate/1.0"
        or value.get("status")
        != "FROZEN_AFTER_AXIS_ONLY_PREFLIGHT_BEFORE_NUMERIC_MATRIX_ACCESS"
    ):
        raise PermissionError("candidate designation is not frozen pre-outcome")
    if value.get("axis_preflight", {}).get("numeric_count_values_read") is not False:
        raise PermissionError("candidate was not frozen before numeric count access")
    markers = value.get("strict_cognates")
    files = value.get("files")
    if not isinstance(markers, list) or len(markers) != MARKER_COUNT:
        raise ValueError("strict cognate panel must contain eleven markers")
    if not isinstance(files, list) or len(files) != 32:
        raise ValueError("candidate must bind all 32 official matrices")
    if any(int(sample.get("cells", 0)) < CELL_BUDGET for sample in files):
        raise ValueError("every frozen sample must contain at least 512 cells")
    if len({sample["file"] for sample in files}) != len(files):
        raise ValueError("candidate contains duplicate archive members")
    source = [sample for sample in files if sample.get("role") == "source_model"]
    held = [sample for sample in files if sample.get("role") == "held_score"]
    excluded = [
        sample for sample in files if sample.get("role") == "excluded_source_overlap"
    ]
    if (
        len(source) != 20
        or len(held) != 10
        or len(excluded) != 2
        or tuple(sorted({sample["batch"] for sample in source})) != SOURCE_BATCHES
        or tuple(sorted({sample["batch"] for sample in held})) != HELD_BATCHES
    ):
        raise ValueError("source, held, or overlap-exclusion split differs")
    if {sample["biological_unit"] for sample in source} & {
        sample["biological_unit"] for sample in held
    }:
        raise ValueError("a held biological unit also appears in source")
    if len({sample["biological_unit"] for sample in held}) != 9:
        raise ValueError("held samples must reduce to nine donor units")
    return value, source, held


def _validate_preflight(path: Path, designation: dict[str, Any]) -> None:
    value = _read_json(path)
    if (
        value.get("schema") != "gse326573-axis-preflight/1.0"
        or value.get("status") != "PASS_BEFORE_NUMERIC_MATRIX_ACCESS"
    ):
        raise PermissionError("axis preflight is not frozen before count access")
    audit = value.get("access_audit", {})
    forbidden = (
        "numeric_count_values_read",
        "cell_identifiers_read",
        "joint_tables_formed",
        "association_statistics_computed",
        "held_losses_computed",
    )
    if any(audit.get(key) != 0 for key in forbidden):
        raise PermissionError("axis preflight accessed a forbidden outcome")
    expected = designation["axis_preflight"]
    observed = value.get("inspected_axes", {})
    comparisons = {
        "files": expected["files"],
        "genes_per_file": expected["gene_features_per_file"],
        "antibodies_per_file": expected["adt_features_per_file"],
        "gene_axis_sha256": expected["gene_axis_sha256"],
    }
    if any(
        observed.get(key) != expected_value
        for key, expected_value in comparisons.items()
    ):
        raise ValueError("axis preflight differs from the candidate designation")


def _verify_archive(path: Path, designation: dict[str, Any]) -> str:
    expected = designation["official_sources"]["raw_archive"]
    if path.stat().st_size != int(expected["bytes"]):
        raise PermissionError("raw archive byte count differs from designation")
    observed = _sha256(path)
    if observed != expected["sha256"]:
        raise PermissionError("raw archive SHA-256 differs from designation")
    return observed


def _selected_cells(barcodes: list[str], sample: str) -> tuple[np.ndarray, list[str]]:
    if len(barcodes) < CELL_BUDGET or len(set(barcodes)) != len(barcodes):
        raise ValueError("filtered barcode axis is too short or nonunique")
    ranked = sorted(
        range(len(barcodes)),
        key=lambda index: (
            hashlib.sha256(
                f"{CELL_SALT}|{sample}|{barcodes[index]}".encode()
            ).hexdigest(),
            barcodes[index],
        ),
    )[:CELL_BUDGET]
    indices = np.asarray(sorted(ranked), dtype=np.int64)
    return indices, [barcodes[index] for index in indices]


def _feature_indices(
    matrix: h5py.Group, markers: list[dict[str, Any]]
) -> tuple[list[int], list[int]]:
    names = _decode(np.asarray(matrix["features/name"][:]))
    types = _decode(np.asarray(matrix["features/feature_type"][:]))
    rna: list[int] = []
    adt: list[int] = []
    for marker in markers:
        rna_hits = [
            index
            for index, (name, kind) in enumerate(zip(names, types))
            if kind == "Gene Expression" and name == marker["rna"]
        ]
        adt_hits = [
            index
            for index, (name, kind) in enumerate(zip(names, types))
            if kind == "Antibody Capture" and name in marker["labels"]
        ]
        if len(rna_hits) != 1 or len(adt_hits) != 1:
            raise ValueError(
                "a frozen RNA or ADT feature does not resolve exactly once"
            )
        rna.append(rna_hits[0])
        adt.append(adt_hits[0])
    return rna, adt


def _read_panel(
    path: Path,
    sample: dict[str, Any],
    markers: list[dict[str, Any]],
    *,
    read_adt: bool,
) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]
        barcodes = _decode(np.asarray(matrix["barcodes"][:]))
        if len(barcodes) != int(sample["cells"]):
            raise ValueError("barcode count differs from the frozen cell axis")
        selected_indices, selected_barcodes = _selected_cells(barcodes, sample["gsm"])
        rna_features, adt_features = _feature_indices(matrix, markers)
        rna_lookup = {
            feature: position for position, feature in enumerate(rna_features)
        }
        adt_lookup = {
            feature: position for position, feature in enumerate(adt_features)
        }
        rna = np.zeros((CELL_BUDGET, MARKER_COUNT), dtype=np.int64)
        adt = np.zeros_like(rna) if read_adt else None
        indptr = matrix["indptr"]
        sparse_indices = matrix["indices"]
        sparse_data = matrix["data"]
        for output_row, column in enumerate(selected_indices):
            start, stop = int(indptr[column]), int(indptr[column + 1])
            local_indices = np.asarray(sparse_indices[start:stop], dtype=np.int64)
            rna_positions = [
                start + offset
                for offset, feature in enumerate(local_indices)
                if int(feature) in rna_lookup
            ]
            if rna_positions:
                raw_values = np.asarray(sparse_data[rna_positions])
                if (
                    not np.isfinite(raw_values).all()
                    or np.any(raw_values < 0)
                    or not np.array_equal(raw_values, np.rint(raw_values))
                ):
                    raise ValueError("RNA panel is not nonnegative integer count data")
                values = raw_values.astype(np.int64)
                for absolute, value in zip(rna_positions, values):
                    feature = int(local_indices[absolute - start])
                    rna[output_row, rna_lookup[feature]] = value
            if read_adt and adt is not None:
                adt_positions = [
                    start + offset
                    for offset, feature in enumerate(local_indices)
                    if int(feature) in adt_lookup
                ]
                if adt_positions:
                    raw_values = np.asarray(sparse_data[adt_positions])
                    if (
                        not np.isfinite(raw_values).all()
                        or np.any(raw_values < 0)
                        or not np.array_equal(raw_values, np.rint(raw_values))
                    ):
                        raise ValueError(
                            "ADT panel is not nonnegative integer count data"
                        )
                    values = raw_values.astype(np.int64)
                    for absolute, value in zip(adt_positions, values):
                        feature = int(local_indices[absolute - start])
                        adt[output_row, adt_lookup[feature]] = value
    if np.any(rna < 0) or (adt is not None and np.any(adt < 0)):
        raise ValueError("count panel contains a negative entry")
    return {
        "rna": rna,
        "adt": adt,
        "barcodes": selected_barcodes,
        "barcode_axis_sha256": _axis_sha256(barcodes),
        "selected_cell_axis_sha256": _axis_sha256(selected_barcodes),
    }


def _member_path(
    archive: tarfile.TarFile,
    sample: dict[str, Any],
    temporary_root: Path,
) -> Path:
    matches = [
        member for member in archive.getmembers() if member.name == sample["file"]
    ]
    if len(matches) != 1 or not matches[0].isfile():
        raise ValueError("designated archive member does not resolve exactly once")
    target = temporary_root / sample["file"]
    source = archive.extractfile(matches[0])
    if source is None:
        raise ValueError("designated archive member cannot be opened")
    with source, target.open("xb") as output:
        shutil.copyfileobj(source, output, length=8 << 20)
    if (
        target.stat().st_size != int(sample["bytes"])
        or _sha256(target) != sample["sha256"]
    ):
        raise PermissionError("designated matrix bytes differ from the frozen file")
    return target


def _read_samples(
    archive_path: Path,
    samples: list[dict[str, Any]],
    markers: list[dict[str, Any]],
    *,
    read_adt: bool,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    with (
        tarfile.open(archive_path, "r") as archive,
        tempfile.TemporaryDirectory(prefix="gse326573-") as temporary,
    ):
        root = Path(temporary)
        for sample in samples:
            path = _member_path(archive, sample, root)
            try:
                output[sample["gsm"]] = _read_panel(
                    path, sample, markers, read_adt=read_adt
                )
            finally:
                path.unlink(missing_ok=True)
    return output


def _adt_states(counts: np.ndarray, barcodes: list[str], sample: str) -> np.ndarray:
    values = np.asarray(counts)
    if values.shape != (CELL_BUDGET, MARKER_COUNT):
        raise ValueError("ADT count panel has the wrong shape")
    states = np.zeros(values.shape, dtype=np.uint8)
    for marker in range(MARKER_COUNT):
        order = sorted(
            range(CELL_BUDGET),
            key=lambda index: (
                int(values[index, marker]),
                hashlib.sha256(
                    f"{ADT_TIE_SALT}|{sample}|{marker}|{barcodes[index]}".encode()
                ).hexdigest(),
                barcodes[index],
            ),
        )
        states[np.asarray(order[CELL_BUDGET // 2 :], dtype=int), marker] = 1
    if not np.all(states.sum(axis=0) == CELL_BUDGET // 2):
        raise AssertionError("deterministic ADT midrank split changed its margin")
    return states


def _destroyed_adt(states: np.ndarray, barcodes: list[str], sample: str) -> np.ndarray:
    order = sorted(
        range(CELL_BUDGET),
        key=lambda index: (
            hashlib.sha256(
                f"{DESTROYED_SALT}|{sample}|{barcodes[index]}".encode()
            ).hexdigest(),
            barcodes[index],
        ),
    )
    order_array = np.asarray(order, dtype=int)
    destroyed = np.empty_like(states)
    destroyed[order_array] = np.asarray(states)[np.roll(order_array, 1)]
    if not np.array_equal(destroyed.sum(axis=0), np.asarray(states).sum(axis=0)):
        raise AssertionError("destroyed-link control changed an ADT margin")
    if sorted(map(tuple, destroyed.tolist())) != sorted(
        map(tuple, np.asarray(states).tolist())
    ):
        raise AssertionError("destroyed-link control changed multivariate ADT states")
    return destroyed


def _tables(rna: np.ndarray, adt: np.ndarray) -> np.ndarray:
    first = np.asarray(rna)
    second = np.asarray(adt)
    if first.shape != (CELL_BUDGET, MARKER_COUNT) or second.shape != first.shape:
        raise ValueError("binary panels have the wrong shape")
    output = np.zeros((MARKER_COUNT, MARKER_COUNT, 2, 2), dtype=np.int64)
    for row, column in np.ndindex((MARKER_COUNT, MARKER_COUNT)):
        code = 2 * first[:, row] + second[:, column]
        output[row, column] = np.bincount(code, minlength=4).reshape(2, 2)
    return output


def _reduce_source(
    panels: dict[str, dict[str, Any]], samples: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for sample in samples:
        gsm = sample["gsm"]
        panel = panels[gsm]
        rna = (panel["rna"] > 0).astype(np.uint8)
        if panel["adt"] is None:
            raise ValueError("source reduction requires ADT counts")
        adt = _adt_states(panel["adt"], panel["barcodes"], gsm)
        observed = _tables(rna, adt)
        destroyed = _tables(rna, _destroyed_adt(adt, panel["barcodes"], gsm))
        records[gsm] = {
            "tables": observed,
            "destroyed_tables": destroyed,
            "rna_profile": rna.mean(axis=0),
            "adt_profile": np.log1p(panel["adt"]).mean(axis=0),
            "table_sha256": _array_sha256(observed),
            "destroyed_table_sha256": _array_sha256(destroyed),
            "barcode_axis_sha256": panel["barcode_axis_sha256"],
            "selected_cell_axis_sha256": panel["selected_cell_axis_sha256"],
        }
    return records


def _informative(tables: np.ndarray) -> np.ndarray:
    values = np.asarray(tables)
    rows = values.sum(axis=-1)
    columns = values.sum(axis=-2)
    total = values.sum(axis=(-2, -1))
    lower = np.maximum(0, rows[..., 0] + columns[..., 0] - total)
    upper = np.minimum(rows[..., 0], columns[..., 0])
    return upper > lower


def _knn_incidence(profiles: np.ndarray, neighbors: int) -> np.ndarray:
    values = np.asarray(profiles, dtype=float).T
    scale = values.std(axis=1, ddof=1)
    if np.any(scale == 0.0) or not np.isfinite(scale).all():
        raise CouplingEstimationRefusal("source marker profile has zero variance")
    values = (values - values.mean(axis=1, keepdims=True)) / scale[:, None]
    edges: set[tuple[int, int]] = set()
    for marker in range(MARKER_COUNT):
        candidates = np.asarray(
            [value for value in range(MARKER_COUNT) if value != marker]
        )
        distances = np.asarray(
            [np.linalg.norm(values[marker] - values[value]) for value in candidates]
        )
        order = candidates[np.lexsort((candidates, distances))]
        edges.update(tuple(sorted((marker, int(value)))) for value in order[:neighbors])
    incidence = np.zeros((MARKER_COUNT, len(edges)), dtype=float)
    for edge, (left, right) in enumerate(sorted(edges)):
        incidence[left, edge] = 1.0
        incidence[right, edge] = 1.0
    return incidence


def _fit_primary(
    tables: np.ndarray,
    rna_profiles: np.ndarray,
    adt_profiles: np.ndarray,
    config: PrimaryConfig,
) -> dict[str, Any]:
    if config.graph_penalty == 0.0:
        first = np.eye(MARKER_COUNT, dtype=float)
        second = np.eye(MARKER_COUNT, dtype=float)
    else:
        first = _knn_incidence(rna_profiles, config.graph_neighbors)
        second = _knn_incidence(adt_profiles, config.graph_neighbors)
    fit = fit_hierarchical_conditional_log_odds(
        np.asarray(tables, dtype=np.int64),
        first,
        second,
        heterogeneity_penalty=config.heterogeneity_penalty,
        ridge_penalty=config.ridge_penalty,
        graph_penalty=config.graph_penalty,
        minimum_informative_donors=2,
        maximum_condition_number=MAXIMUM_CONDITION_NUMBER,
    )
    return {
        "population_log_odds": fit.population_log_odds,
        "fit_certificate": {
            "gradient_norm": fit.gradient_norm,
            "scaled_gradient_norm": fit.scaled_gradient_norm,
            "schur_condition_number": fit.schur_condition_number,
            "theta_curvature_condition_number": fit.theta_curvature_condition_number,
            "iterations": fit.iterations,
            "rna_incidence_sha256": _array_sha256(first),
            "adt_incidence_sha256": _array_sha256(second),
        },
    }


def _fit_common_effect(tables: np.ndarray) -> dict[str, Any]:
    identity = np.eye(MARKER_COUNT, dtype=float)
    fit = fit_structured_conditional_log_odds(
        np.asarray(tables, dtype=np.int64),
        identity,
        identity,
        ridge_penalty=0.0,
        graph_penalty=0.0,
        minimum_informative_donors=2,
        maximum_condition_number=MAXIMUM_CONDITION_NUMBER,
        maximum_iterations=1_000,
        tolerance=1e-9,
    )
    return {
        "population_log_odds": fit.log_odds,
        "fit_certificate": {
            "gradient_norm": fit.gradient_norm,
            "condition_number": fit.condition_number,
            "iterations": fit.iterations,
        },
    }


def _fit_pooled_poisson(tables: np.ndarray) -> dict[str, Any]:
    pooled = np.asarray(tables, dtype=np.int64).sum(axis=0)
    if np.any(pooled <= 0):
        raise CouplingEstimationRefusal(
            "pooled saturated Poisson interaction has a zero cell"
        )
    odds = np.log(pooled[..., 0, 0]) + np.log(pooled[..., 1, 1])
    odds -= np.log(pooled[..., 0, 1]) + np.log(pooled[..., 1, 0])
    if not np.isfinite(odds).all():
        raise CouplingEstimationRefusal("pooled Poisson interaction is nonfinite")
    return {
        "population_log_odds": odds,
        "pooled_tables_sha256": _array_sha256(pooled),
    }


def _residual_pool(tables: np.ndarray, family: str) -> np.ndarray:
    values = np.asarray(tables).reshape(len(tables), -1, 2, 2)
    support = _informative(values)
    if np.any(support.sum(axis=0) < 2):
        raise CouplingEstimationRefusal("classical residual lacks source support")
    coordinates = np.full(support.shape, np.nan)
    statistic = (
        signed_pearson_coordinate
        if family == "pearson"
        else signed_deviance_coordinate
        if family == "root_deviance"
        else None
    )
    if statistic is None:
        raise ValueError("unknown residual family")
    for donor, entity in np.argwhere(support):
        coordinate = statistic(values[donor, entity])
        coordinates[donor, entity] = coordinate / math.sqrt(CELL_BUDGET)
    pooled = np.nanmean(coordinates, axis=0)
    if not np.isfinite(pooled).all():
        raise CouplingEstimationRefusal("classical residual is nonfinite")
    return pooled.reshape(MARKER_COUNT, MARKER_COUNT)


def _fractional_pearson(table: np.ndarray) -> float:
    values = np.asarray(table, dtype=float)
    total = values.sum()
    rows = values.sum(axis=1)
    columns = values.sum(axis=0)
    determinant = values[0, 0] * values[1, 1] - values[0, 1] * values[1, 0]
    return float(determinant * math.sqrt(total / np.prod(rows) / np.prod(columns)))


def _fractional_deviance(table: np.ndarray) -> float:
    values = np.asarray(table, dtype=float)
    expected = np.outer(values.sum(axis=1), values.sum(axis=0)) / values.sum()
    positive = values > 0
    deviance = 2.0 * np.sum(
        values[positive] * np.log(values[positive] / expected[positive])
    )
    determinant = values[0, 0] * values[1, 1] - values[0, 1] * values[1, 0]
    return math.copysign(math.sqrt(max(float(deviance), 0.0)), float(determinant))


def _residual_table(
    coordinate: float,
    rows: np.ndarray,
    columns: np.ndarray,
    family: str,
) -> np.ndarray:
    row = np.asarray(rows, dtype=float)
    column = np.asarray(columns, dtype=float)
    total = row.sum()
    lower = max(0.0, row[0] + column[0] - total)
    upper = min(row[0], column[0])
    if upper <= lower:
        return np.asarray(
            [[lower, row[0] - lower], [column[0] - lower, row[1] - column[0] + lower]]
        )
    epsilon = min(1e-10, 0.25 * (upper - lower))
    left, right = lower + epsilon, upper - epsilon

    def table_at(value: float) -> np.ndarray:
        return np.asarray(
            [[value, row[0] - value], [column[0] - value, row[1] - column[0] + value]]
        )

    statistic = (
        _fractional_pearson
        if family == "pearson"
        else _fractional_deviance
        if family == "root_deviance"
        else None
    )
    if statistic is None:
        raise ValueError("unknown residual family")
    target = float(coordinate)
    target = min(max(target, statistic(table_at(left))), statistic(table_at(right)))
    for _ in range(80):
        midpoint = 0.5 * (left + right)
        if statistic(table_at(midpoint)) < target:
            left = midpoint
        else:
            right = midpoint
    return table_at(0.5 * (left + right))


def _margins(tables: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(tables)
    return values.sum(axis=-1), values.sum(axis=-2)


def _predict_odds(
    log_odds: np.ndarray, rows: np.ndarray, columns: np.ndarray, alpha: float
) -> np.ndarray:
    output = np.empty((MARKER_COUNT, MARKER_COUNT, 2, 2), dtype=float)
    for index in np.ndindex((MARKER_COUNT, MARKER_COUNT)):
        output[index] = expected_binary_table_from_log_odds(
            float(alpha) * float(log_odds[index]), rows[index], columns[index]
        )
    return output


def _predict_residual(
    pooled: np.ndarray, rows: np.ndarray, columns: np.ndarray, config: ResidualConfig
) -> np.ndarray:
    output = np.empty((MARKER_COUNT, MARKER_COUNT, 2, 2), dtype=float)
    for index in np.ndindex((MARKER_COUNT, MARKER_COUNT)):
        coordinate = (
            config.transport_multiplier * pooled[index] * math.sqrt(CELL_BUDGET)
        )
        output[index] = _residual_table(
            coordinate, rows[index], columns[index], config.family
        )
    return output


def _independence(rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    return _predict_odds(np.zeros((MARKER_COUNT, MARKER_COUNT)), rows, columns, 1.0)


def _loss(observed: np.ndarray, predicted: np.ndarray) -> float:
    truth = np.asarray(observed, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    support = _informative(truth)
    if np.count_nonzero(support) < MINIMUM_INFORMATIVE_ENTITIES:
        raise CouplingEstimationRefusal("sample has too few informative marker pairs")
    truth = truth[support]
    estimate = estimate[support]
    if not np.allclose(truth.sum(axis=-1), estimate.sum(axis=-1)) or not np.allclose(
        truth.sum(axis=-2), estimate.sum(axis=-2)
    ):
        raise FloatingPointError("prediction changed a fixed recipient margin")
    positive = truth > 0
    if np.any(estimate[positive] <= 0) or not np.isfinite(estimate).all():
        raise FloatingPointError("prediction assigns invalid mass")
    terms = np.zeros_like(truth)
    terms[positive] = truth[positive] * np.log(truth[positive] / estimate[positive])
    return float((2.0 * terms.sum(axis=(-2, -1)) / CELL_BUDGET).mean())


def _arrays(
    records: dict[str, dict[str, Any]], samples: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    axis = [sample["gsm"] for sample in samples]
    return (
        np.asarray([records[gsm]["tables"] for gsm in axis]),
        np.asarray([records[gsm]["destroyed_tables"] for gsm in axis]),
        np.asarray([records[gsm]["rna_profile"] for gsm in axis]),
        np.asarray([records[gsm]["adt_profile"] for gsm in axis]),
    )


def _primary_configs() -> list[PrimaryConfig]:
    return [
        PrimaryConfig(*values)
        for values in product(
            NEIGHBOR_GRID,
            HETEROGENEITY_GRID,
            RIDGE_GRID,
            GRAPH_GRID,
            TRANSPORT_GRID,
        )
    ]


def _equal_batch_mean(values: np.ndarray, samples: list[dict[str, Any]]) -> float:
    losses = np.asarray(values, dtype=float)
    means = [
        float(
            np.mean(
                [
                    losses[index]
                    for index, sample in enumerate(samples)
                    if sample["batch"] == batch
                ]
            )
        )
        for batch in SOURCE_BATCHES
    ]
    if not np.isfinite(means).all():
        return float("inf")
    return float(np.mean(means))


def _select_source(
    records: dict[str, dict[str, Any]], samples: list[dict[str, Any]]
) -> dict[str, Any]:
    primary_configs = _primary_configs()
    residual_configs = [
        ResidualConfig(*values) for values in product(RESIDUAL_FAMILIES, TRANSPORT_GRID)
    ]
    odds_configs = [
        OddsConfig(method, multiplier)
        for method in ("common_effect_cmle", "pooled_saturated_poisson")
        for multiplier in TRANSPORT_GRID
    ]
    sample_axis = [sample["gsm"] for sample in samples]
    primary_losses = {
        config: np.full(len(samples), np.nan) for config in primary_configs
    }
    residual_losses = {
        config: np.full(len(samples), np.nan) for config in residual_configs
    }
    odds_losses = {config: np.full(len(samples), np.nan) for config in odds_configs}
    independence_losses = np.full(len(samples), np.nan)
    refusals: list[dict[str, Any]] = []
    for batch in SOURCE_BATCHES:
        training = [sample for sample in samples if sample["batch"] != batch]
        validation = [sample for sample in samples if sample["batch"] == batch]
        tables, _, rna_profiles, adt_profiles = _arrays(records, training)
        primary_fits: dict[
            tuple[int, float, float, float], Union[dict[str, Any], Exception]
        ] = {}
        for config in primary_configs:
            key = (
                config.graph_neighbors,
                config.heterogeneity_penalty,
                config.ridge_penalty,
                config.graph_penalty,
            )
            if key not in primary_fits:
                try:
                    primary_fits[key] = _fit_primary(
                        tables, rna_profiles, adt_profiles, config
                    )
                except (
                    ValueError,
                    FloatingPointError,
                    CouplingEstimationRefusal,
                ) as error:
                    primary_fits[key] = error
            fit = primary_fits[key]
            if isinstance(fit, Exception):
                refusals.append(
                    {
                        "batch": batch,
                        "configuration": asdict(config),
                        "reason": str(fit),
                    }
                )
                continue
            for sample in validation:
                truth = records[sample["gsm"]]["tables"]
                rows, columns = _margins(truth)
                index = sample_axis.index(sample["gsm"])
                primary_losses[config][index] = _loss(
                    truth,
                    _predict_odds(
                        fit["population_log_odds"],
                        rows,
                        columns,
                        config.transport_multiplier,
                    ),
                )
        for config in residual_configs:
            try:
                pooled = _residual_pool(tables, config.family)
                for sample in validation:
                    truth = records[sample["gsm"]]["tables"]
                    rows, columns = _margins(truth)
                    index = sample_axis.index(sample["gsm"])
                    residual_losses[config][index] = _loss(
                        truth, _predict_residual(pooled, rows, columns, config)
                    )
            except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
                refusals.append(
                    {
                        "batch": batch,
                        "configuration": asdict(config),
                        "reason": str(error),
                    }
                )
        odds = {
            "common_effect_cmle": _fit_common_effect(tables)["population_log_odds"],
            "pooled_saturated_poisson": _fit_pooled_poisson(tables)[
                "population_log_odds"
            ],
        }
        for sample in validation:
            truth = records[sample["gsm"]]["tables"]
            rows, columns = _margins(truth)
            index = sample_axis.index(sample["gsm"])
            for config in odds_configs:
                odds_losses[config][index] = _loss(
                    truth,
                    _predict_odds(
                        odds[config.method],
                        rows,
                        columns,
                        config.transport_multiplier,
                    ),
                )
            independence_losses[index] = _loss(truth, _independence(rows, columns))
    available_primary = [
        config for config, values in primary_losses.items() if np.isfinite(values).all()
    ]
    available_residual = [
        config
        for config, values in residual_losses.items()
        if np.isfinite(values).all()
    ]
    available_odds = [
        config for config, values in odds_losses.items() if np.isfinite(values).all()
    ]
    if (
        not available_primary
        or not available_residual
        or len(available_odds) != len(odds_configs)
    ):
        raise CouplingEstimationRefusal(
            "leave-one-batch-out CV has no complete candidate"
        )
    primary = min(
        available_primary,
        key=lambda config: (_equal_batch_mean(primary_losses[config], samples), config),
    )
    residual = min(
        available_residual,
        key=lambda config: (
            _equal_batch_mean(residual_losses[config], samples),
            RESIDUAL_FAMILIES.index(config.family),
            config.transport_multiplier,
        ),
    )
    residual_by_family = {
        family: min(
            (config for config in available_residual if config.family == family),
            key=lambda config: (
                _equal_batch_mean(residual_losses[config], samples),
                config,
            ),
        )
        for family in RESIDUAL_FAMILIES
    }
    selected_odds = {
        method: min(
            (config for config in available_odds if config.method == method),
            key=lambda config: (
                _equal_batch_mean(odds_losses[config], samples),
                config,
            ),
        )
        for method in ("common_effect_cmle", "pooled_saturated_poisson")
    }
    classical_losses = {
        "selected_residual": residual_losses[residual],
        **{method: odds_losses[config] for method, config in selected_odds.items()},
        "independence": independence_losses,
    }
    best_classical = min(
        classical_losses,
        key=lambda name: (_equal_batch_mean(classical_losses[name], samples), name),
    )
    primary_values = primary_losses[primary]
    residual_values = residual_losses[residual]
    primary_mean = _equal_batch_mean(primary_values, samples)
    residual_mean = _equal_batch_mean(residual_values, samples)
    common_mean = _equal_batch_mean(
        odds_losses[selected_odds["common_effect_cmle"]], samples
    )
    poisson_mean = _equal_batch_mean(
        odds_losses[selected_odds["pooled_saturated_poisson"]], samples
    )
    residual_batch_differences = {
        batch: float(
            np.mean(
                [
                    primary_values[index] - residual_values[index]
                    for index, sample in enumerate(samples)
                    if sample["batch"] == batch
                ]
            )
        )
        for batch in SOURCE_BATCHES
    }
    independence_mean = _equal_batch_mean(independence_losses, samples)
    independence_batch_differences = {
        batch: float(
            np.mean(
                [
                    primary_values[index] - independence_losses[index]
                    for index, sample in enumerate(samples)
                    if sample["batch"] == batch
                ]
            )
        )
        for batch in SOURCE_BATCHES
    }
    all_candidates_valid = bool(
        not refusals
        and len(available_primary) == len(primary_configs)
        and len(available_residual) == len(residual_configs)
        and len(available_odds) == len(odds_configs)
        and np.isfinite(independence_losses).all()
    )
    gate_checks = {
        "all_three_official_batches_held_out_once": True,
        "all_frozen_candidates_valid": all_candidates_valid,
        "primary_at_least_five_percent_below_calibrated_residual": bool(
            1.0 - primary_mean / residual_mean >= 0.05
        ),
        "primary_favorable_in_at_least_sixteen_source_donors": bool(
            np.count_nonzero(primary_values < residual_values) >= 16
        ),
        "primary_minus_residual_negative_in_every_source_batch": bool(
            all(value < 0.0 for value in residual_batch_differences.values())
        ),
        "primary_at_least_five_percent_below_independence": bool(
            1.0 - primary_mean / independence_mean >= 0.05
        ),
        "primary_favorable_vs_independence_in_at_least_sixteen_source_donors": bool(
            np.count_nonzero(primary_values < independence_losses) >= 16
        ),
        "primary_minus_independence_negative_in_every_source_batch": bool(
            all(value < 0.0 for value in independence_batch_differences.values())
        ),
        "primary_point_loss_below_tuned_common_effect_cmle": bool(
            primary_mean < common_mean
        ),
        "primary_point_loss_below_tuned_pooled_saturated_poisson": bool(
            primary_mean < poisson_mean
        ),
    }
    return {
        "primary": asdict(primary),
        "selected_residual": asdict(residual),
        "residual_by_family": {
            family: asdict(config) for family, config in residual_by_family.items()
        },
        "selected_odds": {
            method: asdict(config) for method, config in selected_odds.items()
        },
        "source_selected_best_classical": best_classical,
        "sample_axis": sample_axis,
        "selected_fold_losses": {
            "primary": primary_values.tolist(),
            "selected_residual": residual_losses[residual].tolist(),
            **{
                f"{family}_residual": residual_losses[config].tolist()
                for family, config in residual_by_family.items()
            },
            **{
                method: odds_losses[config].tolist()
                for method, config in selected_odds.items()
            },
            "independence": independence_losses.tolist(),
        },
        "equal_batch_mean_losses": {
            "primary": primary_mean,
            **{
                method: _equal_batch_mean(values, samples)
                for method, values in classical_losses.items()
            },
        },
        "primary_minus_residual_batch_means": residual_batch_differences,
        "primary_minus_independence_batch_means": independence_batch_differences,
        "source_gate": {"checks": gate_checks, "passes": all(gate_checks.values())},
        "refusals": refusals,
    }


def _fit_models(
    records: dict[str, dict[str, Any]],
    samples: list[dict[str, Any]],
    selection: dict[str, Any],
) -> dict[str, Any]:
    tables, destroyed, rna_profiles, adt_profiles = _arrays(records, samples)
    primary_config = PrimaryConfig(**selection["primary"])
    residual_config = ResidualConfig(**selection["selected_residual"])
    primary = _fit_primary(tables, rna_profiles, adt_profiles, primary_config)
    destroyed_fit = _fit_primary(destroyed, rna_profiles, adt_profiles, primary_config)
    common = _fit_common_effect(tables)
    poisson = _fit_pooled_poisson(tables)
    common_config = OddsConfig(**selection["selected_odds"]["common_effect_cmle"])
    poisson_config = OddsConfig(
        **selection["selected_odds"]["pooled_saturated_poisson"]
    )
    models = {
        "primary": {
            "family": "hierarchical_graph_regularized_exact_conditional",
            "configuration": asdict(primary_config),
            "population_log_odds": primary["population_log_odds"].tolist(),
            "fit_certificate": primary["fit_certificate"],
        },
        "selected_residual": {
            "family": f"poisson_independence_signed_{residual_config.family}",
            "configuration": asdict(residual_config),
            "pooled_coordinate": _residual_pool(
                tables, residual_config.family
            ).tolist(),
        },
        "common_effect_cmle": {
            "family": "unpenalized_common_effect_exact_conditional",
            "configuration": asdict(common_config),
            "population_log_odds": common["population_log_odds"].tolist(),
            "fit_certificate": common["fit_certificate"],
        },
        "pooled_saturated_poisson": {
            "family": "pooled_saturated_poisson_log_linear_interaction",
            "configuration": asdict(poisson_config),
            "population_log_odds": poisson["population_log_odds"].tolist(),
            "pooled_tables_sha256": poisson["pooled_tables_sha256"],
        },
        "independence": {"family": "poisson_row_plus_column_independence"},
        "destroyed_link": {
            "family": "hierarchical_exact_conditional_after_within_sample_link_destruction",
            "configuration": asdict(primary_config),
            "population_log_odds": destroyed_fit["population_log_odds"].tolist(),
            "fit_certificate": destroyed_fit["fit_certificate"],
        },
    }
    for family, raw_config in selection["residual_by_family"].items():
        config = ResidualConfig(**raw_config)
        models[f"{family}_residual"] = {
            "family": f"poisson_independence_signed_{family}",
            "configuration": asdict(config),
            "pooled_coordinate": _residual_pool(tables, family).tolist(),
        }
    return models


def _predict_models(
    models: dict[str, Any], rows: np.ndarray, columns: np.ndarray
) -> dict[str, np.ndarray]:
    primary_config = PrimaryConfig(**models["primary"]["configuration"])
    residual_config = ResidualConfig(**models["selected_residual"]["configuration"])
    common_config = OddsConfig(**models["common_effect_cmle"]["configuration"])
    poisson_config = OddsConfig(**models["pooled_saturated_poisson"]["configuration"])
    output = {
        "primary": _predict_odds(
            np.asarray(models["primary"]["population_log_odds"]),
            rows,
            columns,
            primary_config.transport_multiplier,
        ),
        "selected_residual": _predict_residual(
            np.asarray(models["selected_residual"]["pooled_coordinate"]),
            rows,
            columns,
            residual_config,
        ),
        "common_effect_cmle": _predict_odds(
            np.asarray(models["common_effect_cmle"]["population_log_odds"]),
            rows,
            columns,
            common_config.transport_multiplier,
        ),
        "pooled_saturated_poisson": _predict_odds(
            np.asarray(models["pooled_saturated_poisson"]["population_log_odds"]),
            rows,
            columns,
            poisson_config.transport_multiplier,
        ),
        "independence": _independence(rows, columns),
        "destroyed_link": _predict_odds(
            np.asarray(models["destroyed_link"]["population_log_odds"]),
            rows,
            columns,
            primary_config.transport_multiplier,
        ),
    }
    for family in RESIDUAL_FAMILIES:
        name = f"{family}_residual"
        model = models[name]
        output[name] = _predict_residual(
            np.asarray(model["pooled_coordinate"]),
            rows,
            columns,
            ResidualConfig(**model["configuration"]),
        )
    return output


def _held_margins(rna_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positives = (np.asarray(rna_counts) > 0).sum(axis=0)
    rows = np.repeat(
        np.stack([CELL_BUDGET - positives, positives], axis=1)[:, None, :],
        MARKER_COUNT,
        axis=1,
    )
    columns = np.broadcast_to(
        np.asarray([CELL_BUDGET // 2, CELL_BUDGET // 2]),
        (MARKER_COUNT, MARKER_COUNT, 2),
    ).copy()
    return rows, columns


def _comparison(
    units: list[str],
    unit_batches: list[str],
    primary: np.ndarray,
    comparator: np.ndarray,
    seed: int,
    *,
    formal_transfer: bool,
) -> dict[str, Any]:
    difference = np.asarray(primary) - np.asarray(comparator)
    generator = np.random.default_rng(seed)
    batch_indices = [
        np.flatnonzero(np.asarray(unit_batches) == batch) for batch in HELD_BATCHES
    ]
    if any(len(indices) != 3 for indices in batch_indices):
        raise ValueError(
            "batch-stratified bootstrap requires three units per held batch"
        )
    draws = np.concatenate(
        [
            indices[
                generator.integers(0, len(indices), size=(BOOTSTRAPS, len(indices)))
            ]
            for indices in batch_indices
        ],
        axis=1,
    )
    bootstrap = difference[draws].mean(axis=1)
    interval = np.quantile(bootstrap, [0.025, 0.975], method="linear")
    sensitivity_generator = np.random.default_rng(SENSITIVITY_BOOTSTRAP_SEED)
    sensitivity_indices = sensitivity_generator.integers(
        0, len(units), size=(BOOTSTRAPS, len(units))
    )
    sensitivity_interval = np.quantile(
        difference[sensitivity_indices].mean(axis=1),
        [0.025, 0.975],
        method="linear",
    )
    nonzero = difference[difference != 0]
    favorable = int(np.count_nonzero(nonzero < 0))
    sign_p = (
        float(
            sum(
                math.comb(len(nonzero), value)
                for value in range(favorable, len(nonzero) + 1)
            )
            / (2 ** len(nonzero))
        )
        if len(nonzero)
        else 1.0
    )
    reduction = 1.0 - float(np.mean(primary) / np.mean(comparator))
    batch_differences = {
        batch: float(
            np.mean(
                [
                    difference[index]
                    for index, observed in enumerate(unit_batches)
                    if observed == batch
                ]
            )
        )
        for batch in HELD_BATCHES
    }
    classical_checks = {
        "primary_point_loss_below_comparator": bool(difference.mean() < 0.0),
        "batch_stratified_paired_bootstrap_upper_95_below_zero": bool(
            interval[1] < 0.0
        ),
    }
    transfer_checks = {
        **classical_checks,
        "relative_deviance_reduction_at_least_five_percent": bool(reduction >= 0.05),
        "at_least_eight_of_nine_donors_favorable": bool(favorable >= 8),
        "exact_one_sided_sign_test_p_at_most_0_025": bool(sign_p <= 0.025),
        "primary_minus_comparator_negative_in_every_held_batch": bool(
            all(value < 0.0 for value in batch_differences.values())
        ),
    }
    checks = transfer_checks if formal_transfer else classical_checks
    return {
        "unit_count": len(units),
        "primary_mean_loss": float(np.mean(primary)),
        "comparator_mean_loss": float(np.mean(comparator)),
        "relative_deviance_reduction": reduction,
        "mean_paired_difference": float(difference.mean()),
        "batch_stratified_paired_bootstrap_95_interval": interval.tolist(),
        "unstratified_paired_bootstrap_95_interval_sensitivity": sensitivity_interval.tolist(),
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_seed": seed,
        "sensitivity_bootstrap_seed": SENSITIVITY_BOOTSTRAP_SEED,
        "favorable_non_tied_units": favorable,
        "non_tied_units": int(len(nonzero)),
        "exact_one_sided_sign_test_p": sign_p,
        "held_batch_mean_differences": batch_differences,
        "comparison_gate": "formal_transfer"
        if formal_transfer
        else "classical_increment",
        "checks": checks,
        "passes": all(checks.values()),
        "unit_differences": dict(zip(units, difference.tolist())),
    }


def _aggregate_unit_losses(
    samples: list[dict[str, Any]],
    sample_losses: dict[str, dict[str, float]],
) -> tuple[list[str], dict[str, np.ndarray]]:
    units = sorted({sample["biological_unit"] for sample in samples})
    output = {method: np.empty(len(units)) for method in sample_losses}
    for unit_index, unit in enumerate(units):
        gsms = [
            sample["gsm"] for sample in samples if sample["biological_unit"] == unit
        ]
        for method in output:
            output[method][unit_index] = np.mean(
                [sample_losses[method][gsm] for gsm in gsms]
            )
    return units, output


def _held_unit_batches(samples: list[dict[str, Any]], units: list[str]) -> list[str]:
    output: list[str] = []
    for unit in units:
        batches = {
            sample["batch"] for sample in samples if sample["biological_unit"] == unit
        }
        if len(batches) != 1:
            raise ValueError("a held biological unit spans official batches")
        output.append(next(iter(batches)))
    if set(output) != set(HELD_BATCHES):
        raise ValueError("held biological units do not cover all frozen batches")
    return output


def _confirmation_status(transfer_pass: bool, classical_increment: bool) -> str:
    if transfer_pass and classical_increment:
        return "CONFIRMATION_PASS_WITH_CLASSICAL_INCREMENT"
    if transfer_pass:
        return "TRANSFER_PASS_WITHOUT_CLASSICAL_INCREMENT"
    return "CONFIRMATION_FAIL"


def _one_shot(
    phase: str,
    attempt_path: Path,
    output_path: Path,
    bindings: dict[str, str],
    body: Any,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"{phase} output already exists")
    _validate_attempt(attempt_path, phase, bindings)
    try:
        payload = body()
    except (
        Exception
    ) as error:  # one-shot protocol turns every post-claim error terminal
        payload = {
            "schema": f"gse326573-lung-{phase}/1.0",
            "status": f"TERMINAL_{phase.upper()}_REFUSAL",
            "created_at_utc": _timestamp(),
            "reason_code": type(error).__name__,
            "reason": str(error),
            "bindings": bindings,
        }
    _write_json_x(output_path, payload)
    return payload


def _require_bindings(result: dict[str, Any], bindings: dict[str, str]) -> None:
    if any(result.get(key) != value for key, value in bindings.items()):
        raise PermissionError("predecessor artifact bindings differ")


def _private_rna_mode_is_0600(path: Path) -> bool:
    return stat.S_IMODE(path.stat().st_mode) == 0o600


def _write_private_rna(
    path: Path,
    samples: list[dict[str, Any]],
    panels: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, str]]:
    sample_axis = [sample["gsm"] for sample in samples]
    states = np.asarray(
        [(panels[gsm]["rna"] > 0).astype(np.uint8) for gsm in sample_axis],
        dtype=np.uint8,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        np.savez_compressed(stream, sample_axis=np.asarray(sample_axis), states=states)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o600)
    if not _private_rna_mode_is_0600(path):
        raise PermissionError("private held RNA artifact is not mode 0600")
    return _sha256(path), {
        gsm: _array_sha256(states[index]) for index, gsm in enumerate(sample_axis)
    }


def _read_private_rna(
    path: Path, samples: list[dict[str, Any]]
) -> dict[str, np.ndarray]:
    if not _private_rna_mode_is_0600(path):
        raise PermissionError("private held RNA artifact is not mode 0600")
    with np.load(path, allow_pickle=False) as archive:
        sample_axis = [str(value) for value in archive["sample_axis"]]
        states = np.asarray(archive["states"], dtype=np.uint8)
    expected = [sample["gsm"] for sample in samples]
    if sample_axis != expected or states.shape != (
        len(samples),
        CELL_BUDGET,
        MARKER_COUNT,
    ):
        raise PermissionError("private held RNA sample axis or shape differs")
    if np.any((states != 0) & (states != 1)):
        raise PermissionError("private held RNA artifact is not binary")
    return {gsm: states[index] for index, gsm in enumerate(sample_axis)}


def _stage_bindings(
    phase: str,
    base: dict[str, str],
    source_path: Path,
    prediction_path: Path,
    private_rna_path: Path,
    authorization_path: Path,
) -> dict[str, str]:
    bindings = dict(base)
    if phase in {"prediction", "score"}:
        source = _read_json(source_path)
        if source.get("status") != "SOURCE_GATE_PASS":
            raise PermissionError("source gate did not pass")
        _require_bindings(source, base)
        bindings["source_sha256"] = _sha256(source_path)
    if phase == "score":
        prediction = _read_json(prediction_path)
        if prediction.get("status") != "HELD_MARGIN_ONLY_PREDICTIONS_FROZEN":
            raise PermissionError("held predictions are not frozen")
        _require_bindings(prediction, bindings)
        if not private_rna_path.is_file() or not _private_rna_mode_is_0600(
            private_rna_path
        ):
            raise PermissionError("private held RNA artifact is absent or exposed")
        private_sha256 = _sha256(private_rna_path)
        if prediction.get("private_rna_sha256") != private_sha256:
            raise PermissionError("private held RNA artifact differs from prediction")
        bindings["prediction_sha256"] = _sha256(prediction_path)
        bindings["private_rna_sha256"] = private_sha256
        authorization = _read_json(authorization_path)
        if (
            authorization.get("schema") != "gse326573-lung-score-authorization/1.0"
            or authorization.get("status") != "SCORE_AUTHORIZED"
            or authorization.get("bindings") != bindings
        ):
            raise PermissionError(
                "score authorization is absent or bound to different bytes"
            )
        bindings["score_authorization_sha256"] = _sha256(authorization_path)
    return bindings


def claim_stage(
    phase: str,
    archive_path: Path = DEFAULT_ARCHIVE,
    designation_path: Path = DEFAULT_DESIGNATION,
    preflight_path: Path = DEFAULT_PREFLIGHT,
    protocol_path: Path = DEFAULT_PROTOCOL,
    runtime_path: Path = DEFAULT_RUNTIME,
    source_path: Path = DEFAULT_SOURCE,
    prediction_path: Path = DEFAULT_PREDICTION,
    private_rna_path: Path = DEFAULT_PRIVATE_RNA,
    authorization_path: Path = DEFAULT_SCORE_AUTHORIZATION,
    attempt_path: Optional[Path] = None,
) -> dict[str, Any]:
    if phase not in {"source", "prediction", "score"}:
        raise ValueError("claim phase must be source, prediction, or score")
    _, base = _base_bindings(
        archive_path,
        designation_path,
        preflight_path,
        protocol_path,
        runtime_path,
    )
    bindings = _stage_bindings(
        phase,
        base,
        source_path,
        prediction_path,
        private_rna_path,
        authorization_path,
    )
    default = {
        "source": DEFAULT_SOURCE_ATTEMPT,
        "prediction": DEFAULT_PREDICTION_ATTEMPT,
        "score": DEFAULT_SCORE_ATTEMPT,
    }[phase]
    path = attempt_path or default
    _claim(path, phase, bindings)
    return _read_json(path)


def run_source(
    archive_path: Path = DEFAULT_ARCHIVE,
    designation_path: Path = DEFAULT_DESIGNATION,
    preflight_path: Path = DEFAULT_PREFLIGHT,
    protocol_path: Path = DEFAULT_PROTOCOL,
    runtime_path: Path = DEFAULT_RUNTIME,
    attempt_path: Path = DEFAULT_SOURCE_ATTEMPT,
    output_path: Path = DEFAULT_SOURCE,
) -> dict[str, Any]:
    designation, source, _ = _designation(designation_path)
    _, bindings = _base_bindings(
        archive_path,
        designation_path,
        preflight_path,
        protocol_path,
        runtime_path,
    )

    def body() -> dict[str, Any]:
        markers = designation["strict_cognates"]
        panels = _read_samples(archive_path, source, markers, read_adt=True)
        records = _reduce_source(panels, source)
        selection = _select_source(records, source)
        payload: dict[str, Any] = {
            "schema": "gse326573-lung-source/1.0",
            "status": "SOURCE_GATE_PASS"
            if selection["source_gate"]["passes"]
            else "TERMINAL_SOURCE_GATE_REFUSAL",
            "created_at_utc": _timestamp(),
            **bindings,
            "source_batches": list(SOURCE_BATCHES),
            "source_samples": [sample["gsm"] for sample in source],
            "source_units": [sample["biological_unit"] for sample in source],
            "marker_axis": [marker["adt_canonical"] for marker in markers],
            "selection": selection,
            "source_table_sha256": {
                gsm: record["table_sha256"] for gsm, record in records.items()
            },
            "source_destroyed_table_sha256": {
                gsm: record["destroyed_table_sha256"] for gsm, record in records.items()
            },
            "source_selected_cell_axis_sha256": {
                gsm: record["selected_cell_axis_sha256"]
                for gsm, record in records.items()
            },
            "access": {
                "source_matrices_read": len(source),
                "held_matrices_opened": 0,
                "held_numeric_values_read": 0,
            },
        }
        if selection["source_gate"]["passes"]:
            payload["models"] = _fit_models(records, source, selection)
        return payload

    return _one_shot("source", attempt_path, output_path, bindings, body)


def run_prediction(
    archive_path: Path = DEFAULT_ARCHIVE,
    designation_path: Path = DEFAULT_DESIGNATION,
    preflight_path: Path = DEFAULT_PREFLIGHT,
    protocol_path: Path = DEFAULT_PROTOCOL,
    runtime_path: Path = DEFAULT_RUNTIME,
    source_path: Path = DEFAULT_SOURCE,
    attempt_path: Path = DEFAULT_PREDICTION_ATTEMPT,
    output_path: Path = DEFAULT_PREDICTION,
    private_rna_path: Path = DEFAULT_PRIVATE_RNA,
) -> dict[str, Any]:
    designation, _, held = _designation(designation_path)
    source_result = _read_json(source_path)
    _, base = _base_bindings(
        archive_path,
        designation_path,
        preflight_path,
        protocol_path,
        runtime_path,
    )
    bindings = _stage_bindings(
        "prediction",
        base,
        source_path,
        DEFAULT_PREDICTION,
        private_rna_path,
        DEFAULT_SCORE_AUTHORIZATION,
    )

    def body() -> dict[str, Any]:
        panels = _read_samples(
            archive_path, held, designation["strict_cognates"], read_adt=False
        )
        private_sha256, private_state_hashes = _write_private_rna(
            private_rna_path, held, panels
        )
        samples = []
        for sample in held:
            panel = panels[sample["gsm"]]
            rows, columns = _held_margins(panel["rna"])
            predictions = _predict_models(source_result["models"], rows, columns)
            samples.append(
                {
                    "gsm": sample["gsm"],
                    "biological_unit": sample["biological_unit"],
                    "batch": sample["batch"],
                    "row_margins": rows.tolist(),
                    "column_margins": columns.tolist(),
                    "predicted_tables": {
                        name: values.tolist() for name, values in predictions.items()
                    },
                    "prediction_sha256": {
                        name: _array_sha256(values)
                        for name, values in predictions.items()
                    },
                    "barcode_axis_sha256": panel["barcode_axis_sha256"],
                    "selected_cell_axis_sha256": panel["selected_cell_axis_sha256"],
                    "rna_state_sha256": private_state_hashes[sample["gsm"]],
                }
            )
        return {
            "schema": "gse326573-lung-prediction/1.0",
            "status": "HELD_MARGIN_ONLY_PREDICTIONS_FROZEN",
            "created_at_utc": _timestamp(),
            **bindings,
            "held_batches": list(HELD_BATCHES),
            "held_sample_count": len(held),
            "held_unit_count": len({sample["biological_unit"] for sample in held}),
            "private_rna_sha256": private_sha256,
            "source_selected_best_classical": source_result["selection"][
                "source_selected_best_classical"
            ],
            "samples": samples,
            "access": {
                "held_gex_panel_values_read": len(held) * CELL_BUDGET * MARKER_COUNT,
                "held_sparse_indices_inspected_for_routing": True,
                "held_adt_data_positions_requested": 0,
                "held_adt_numeric_values_read": 0,
            },
        }

    return _one_shot("prediction", attempt_path, output_path, bindings, body)


def authorize_score(
    archive_path: Path = DEFAULT_ARCHIVE,
    designation_path: Path = DEFAULT_DESIGNATION,
    preflight_path: Path = DEFAULT_PREFLIGHT,
    protocol_path: Path = DEFAULT_PROTOCOL,
    runtime_path: Path = DEFAULT_RUNTIME,
    source_path: Path = DEFAULT_SOURCE,
    prediction_path: Path = DEFAULT_PREDICTION,
    private_rna_path: Path = DEFAULT_PRIVATE_RNA,
    output_path: Path = DEFAULT_SCORE_AUTHORIZATION,
) -> dict[str, Any]:
    _, base = _base_bindings(
        archive_path,
        designation_path,
        preflight_path,
        protocol_path,
        runtime_path,
    )
    prediction_bindings = _stage_bindings(
        "prediction",
        base,
        source_path,
        prediction_path,
        private_rna_path,
        output_path,
    )
    prediction = _read_json(prediction_path)
    if prediction.get("status") != "HELD_MARGIN_ONLY_PREDICTIONS_FROZEN":
        raise PermissionError("held predictions are not frozen")
    _require_bindings(prediction, prediction_bindings)
    private_sha256 = _sha256(private_rna_path)
    if (
        not _private_rna_mode_is_0600(private_rna_path)
        or prediction.get("private_rna_sha256") != private_sha256
    ):
        raise PermissionError("private held RNA artifact differs or is exposed")
    bindings = {
        **prediction_bindings,
        "prediction_sha256": _sha256(prediction_path),
        "private_rna_sha256": private_sha256,
    }
    payload = {
        "schema": "gse326573-lung-score-authorization/1.0",
        "status": "SCORE_AUTHORIZED",
        "created_at_utc": _timestamp(),
        "bindings": bindings,
    }
    _write_json_x(output_path, payload)
    return payload


def _verify_frozen_rna(
    panel: dict[str, Any], frozen: dict[str, Any], private_state: np.ndarray
) -> np.ndarray:
    observed = (np.asarray(panel["rna"]) > 0).astype(np.uint8)
    if (
        not np.array_equal(observed, private_state)
        or _array_sha256(observed) != frozen["rna_state_sha256"]
        or frozen["selected_cell_axis_sha256"] != panel["selected_cell_axis_sha256"]
        or frozen["barcode_axis_sha256"] != panel["barcode_axis_sha256"]
    ):
        raise PermissionError("held RNA state or cell axis differs from prediction")
    return observed


def _frozen_prediction_axis(
    prediction: dict[str, Any], held: list[dict[str, Any]]
) -> list[str]:
    expected = [sample["gsm"] for sample in held]
    observed = [sample.get("gsm") for sample in prediction.get("samples", [])]
    if observed != expected or len(set(observed)) != len(observed):
        raise PermissionError("held prediction sample axis differs from designation")
    return expected


def run_score(
    archive_path: Path = DEFAULT_ARCHIVE,
    designation_path: Path = DEFAULT_DESIGNATION,
    preflight_path: Path = DEFAULT_PREFLIGHT,
    protocol_path: Path = DEFAULT_PROTOCOL,
    runtime_path: Path = DEFAULT_RUNTIME,
    source_path: Path = DEFAULT_SOURCE,
    prediction_path: Path = DEFAULT_PREDICTION,
    private_rna_path: Path = DEFAULT_PRIVATE_RNA,
    authorization_path: Path = DEFAULT_SCORE_AUTHORIZATION,
    attempt_path: Path = DEFAULT_SCORE_ATTEMPT,
    output_path: Path = DEFAULT_SCORE,
) -> dict[str, Any]:
    designation, _, held = _designation(designation_path)
    prediction = _read_json(prediction_path)
    _, base = _base_bindings(
        archive_path,
        designation_path,
        preflight_path,
        protocol_path,
        runtime_path,
    )
    bindings = _stage_bindings(
        "score",
        base,
        source_path,
        prediction_path,
        private_rna_path,
        authorization_path,
    )

    def body() -> dict[str, Any]:
        _frozen_prediction_axis(prediction, held)
        private_states = _read_private_rna(private_rna_path, held)
        rna_panels = _read_samples(
            archive_path, held, designation["strict_cognates"], read_adt=False
        )
        frozen = {sample["gsm"]: sample for sample in prediction["samples"]}
        for sample in held:
            gsm = sample["gsm"]
            _verify_frozen_rna(rna_panels[gsm], frozen[gsm], private_states[gsm])
        panels = _read_samples(
            archive_path, held, designation["strict_cognates"], read_adt=True
        )
        methods = (
            "primary",
            "selected_residual",
            *(f"{family}_residual" for family in RESIDUAL_FAMILIES),
            "common_effect_cmle",
            "pooled_saturated_poisson",
            "independence",
            "destroyed_link",
        )
        sample_losses = {method: {} for method in methods}
        truth_hashes = {}
        for sample in held:
            gsm = sample["gsm"]
            panel = panels[gsm]
            frozen_sample = frozen.get(gsm)
            if frozen_sample is None:
                raise PermissionError("a held sample lacks a frozen prediction")
            if (
                frozen_sample["selected_cell_axis_sha256"]
                != panel["selected_cell_axis_sha256"]
                or frozen_sample["barcode_axis_sha256"] != panel["barcode_axis_sha256"]
            ):
                raise PermissionError(
                    "held cell axis differs from the prediction stage"
                )
            rna = _verify_frozen_rna(panel, frozen_sample, private_states[gsm])
            if panel["adt"] is None:
                raise ValueError("held score requires ADT counts")
            adt = _adt_states(panel["adt"], panel["barcodes"], gsm)
            truth = _tables(rna, adt)
            rows, columns = _margins(truth)
            if (
                rows.tolist() != frozen_sample["row_margins"]
                or columns.tolist() != frozen_sample["column_margins"]
            ):
                raise PermissionError("held truth margins differ from frozen margins")
            truth_hashes[gsm] = _array_sha256(truth)
            for method in methods:
                predicted = np.asarray(
                    frozen_sample["predicted_tables"][method], dtype=float
                )
                if (
                    _array_sha256(predicted)
                    != frozen_sample["prediction_sha256"][method]
                ):
                    raise PermissionError("a frozen held prediction changed")
                sample_losses[method][gsm] = _loss(truth, predicted)
        units, unit_losses = _aggregate_unit_losses(held, sample_losses)
        unit_batches = _held_unit_batches(held, units)
        comparisons = {
            f"primary_vs_{method}": _comparison(
                units,
                unit_batches,
                unit_losses["primary"],
                unit_losses[method],
                BOOTSTRAP_SEED,
                formal_transfer=method
                in {"selected_residual", "independence", "destroyed_link"},
            )
            for method in methods[1:]
        }
        transfer_required = (
            "primary_vs_selected_residual",
            "primary_vs_independence",
            "primary_vs_destroyed_link",
        )
        classical_required = (
            "primary_vs_common_effect_cmle",
            "primary_vs_pooled_saturated_poisson",
        )
        transfer_pass = all(comparisons[name]["passes"] for name in transfer_required)
        classical_increment = all(
            comparisons[name]["passes"] for name in classical_required
        )
        status = _confirmation_status(transfer_pass, classical_increment)
        return {
            "schema": "gse326573-lung-score/1.0",
            "status": status,
            "created_at_utc": _timestamp(),
            **bindings,
            "held_batches": list(HELD_BATCHES),
            "held_samples": [sample["gsm"] for sample in held],
            "held_units": units,
            "held_unit_batches": dict(zip(units, unit_batches)),
            "replicate_aggregation": "sample losses averaged within biological unit before inference",
            "source_selected_best_classical": prediction[
                "source_selected_best_classical"
            ],
            "sample_losses": sample_losses,
            "unit_losses": {
                method: dict(zip(units, values.tolist()))
                for method, values in unit_losses.items()
            },
            "comparisons": comparisons,
            "formal_transfer_comparisons": list(transfer_required),
            "classical_increment_comparisons": list(classical_required),
            "transfer_pass": transfer_pass,
            "classical_increment_pass": classical_increment,
            "passes": transfer_pass and classical_increment,
            "truth_table_sha256": truth_hashes,
            "access": {
                "held_sample_matrices_scored": len(held),
                "held_adt_stage": "first numeric ADT access",
            },
        }

    return _one_shot("score", attempt_path, output_path, bindings, body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "claim-source",
            "source",
            "claim-prediction",
            "predict",
            "authorize-score",
            "claim-score",
            "score",
        ),
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--designation", type=Path, default=DEFAULT_DESIGNATION)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--prediction", type=Path, default=DEFAULT_PREDICTION)
    parser.add_argument("--private-rna", type=Path, default=DEFAULT_PRIVATE_RNA)
    parser.add_argument(
        "--authorization", type=Path, default=DEFAULT_SCORE_AUTHORIZATION
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--attempt", type=Path)
    args = parser.parse_args()
    if args.phase.startswith("claim-"):
        result = claim_stage(
            phase=args.phase.removeprefix("claim-"),
            archive_path=args.archive,
            designation_path=args.designation,
            preflight_path=args.preflight,
            protocol_path=args.protocol,
            runtime_path=args.runtime,
            source_path=args.source,
            prediction_path=args.prediction,
            private_rna_path=args.private_rna,
            authorization_path=args.authorization,
            attempt_path=args.attempt,
        )
    elif args.phase == "source":
        result = run_source(
            archive_path=args.archive,
            designation_path=args.designation,
            preflight_path=args.preflight,
            protocol_path=args.protocol,
            runtime_path=args.runtime,
            attempt_path=args.attempt or DEFAULT_SOURCE_ATTEMPT,
            output_path=args.output or DEFAULT_SOURCE,
        )
    elif args.phase == "predict":
        result = run_prediction(
            archive_path=args.archive,
            designation_path=args.designation,
            preflight_path=args.preflight,
            protocol_path=args.protocol,
            runtime_path=args.runtime,
            source_path=args.source,
            attempt_path=args.attempt or DEFAULT_PREDICTION_ATTEMPT,
            output_path=args.output or DEFAULT_PREDICTION,
            private_rna_path=args.private_rna,
        )
    elif args.phase == "authorize-score":
        result = authorize_score(
            archive_path=args.archive,
            designation_path=args.designation,
            preflight_path=args.preflight,
            protocol_path=args.protocol,
            runtime_path=args.runtime,
            source_path=args.source,
            prediction_path=args.prediction,
            private_rna_path=args.private_rna,
            output_path=args.output or args.authorization,
        )
    else:
        result = run_score(
            archive_path=args.archive,
            designation_path=args.designation,
            preflight_path=args.preflight,
            protocol_path=args.protocol,
            runtime_path=args.runtime,
            source_path=args.source,
            prediction_path=args.prediction,
            private_rna_path=args.private_rna,
            authorization_path=args.authorization,
            attempt_path=args.attempt or DEFAULT_SCORE_ATTEMPT,
            output_path=args.output or DEFAULT_SCORE,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
