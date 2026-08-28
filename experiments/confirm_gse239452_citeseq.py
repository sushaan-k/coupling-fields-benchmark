"""Prospective GSE239452 RNA-protein coupling-field confirmation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable
import urllib.request

import h5py
import numpy as np

from experiments import preflight_gse239452_citeseq as metadata
from mapreg.heterogeneity_adaptive_coupling import (
    CouplingEstimationRefusal,
    expected_binary_table_from_log_odds,
    signed_deviance_coordinate,
    signed_pearson_coordinate,
)
from mapreg.hierarchical_conditional_coupling import (
    fit_hierarchical_conditional_log_odds,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/confirmation/gse239452_citeseq"
SOURCE_ROOT = DATA_DIR / "source_cache"
DEFAULT_SOURCE = DATA_DIR / "source_manifest_v1.json"
DEFAULT_PREFLIGHT = (
    ROOT / "results/development/gse239452_citeseq_metadata_preflight.json"
)
DEFAULT_DESIGNATION = DATA_DIR / "candidate_designation_v1.json"
DEFAULT_PROTOCOL = (
    ROOT / "docs/GSE239452_PREGNANCY_CITESEQ_CONFIRMATION_PROTOCOL_2026-08-28.md"
)
DEFAULT_DEVELOPMENT_AUTHORIZATION = DATA_DIR / "development_authorization_v1.json"
DEFAULT_DEVELOPMENT_ATTEMPT = DATA_DIR / "development_attempt_v1.json"
DEFAULT_REDUCED = ROOT / "data/development/gse239452_citeseq/reduced_v1.json"
DEFAULT_PILOT = ROOT / "results/development/gse239452_citeseq_pilot.json"
DEFAULT_HELD_GEX_AUTHORIZATION = DATA_DIR / "held_gex_authorization_v1.json"
DEFAULT_PREDICTION_ATTEMPT = DATA_DIR / "prediction_attempt_v1.json"
DEFAULT_PREDICTION = ROOT / "results/gse239452_citeseq_predictions.json"
DEFAULT_SCORE_AUTHORIZATION = DATA_DIR / "score_authorization_v1.json"
DEFAULT_SCORE_ATTEMPT = DATA_DIR / "score_attempt_v1.json"
DEFAULT_SCORE = ROOT / "results/gse239452_citeseq_confirmation.json"
DEFAULT_TERMINAL_REFUSAL = (
    ROOT / "results/development/gse239452_citeseq_terminal_refusal.json"
)

PUBLIC_OWNER = "sushaan-k"
PUBLIC_REPOSITORY = "coupling-fields-benchmark"
MARKERS = metadata.EXPECTED_MARKERS
CELL_BUDGET = 512
CALIBRATION = ("47", "31", "223", "77", "191", "321", "213")
PILOT = ("94", "103", "350", "182", "1", "325", "382", "50")
HELD = (
    "OB7-CTRL",
    "705385-SEV",
    "803763-ASX",
    "324058-ASX",
    "644394-CTRL",
    "915348-SEV",
    "729106-CTRL",
    "105199-ASX",
    "101607-SEV",
)
CELL_SALT = "GSE239452-COMMON-CELL-v1"
ADT_TIE_SALT = "GSE239452-ADT-MEDIAN-v1"
DESTROYED_SALT = "GSE239452-DESTROYED-LINK-v1"
NEIGHBOR_GRID = (1, 2)
HETEROGENEITY_GRID = (0.1, 1.0, 10.0)
RIDGE_GRID = (0.01, 0.1)
GRAPH_GRID = (0.0, 0.1, 0.3, 1.0)
ALPHA_GRID = (0.5, 0.75, 1.0, 1.25)
RESIDUAL_FAMILIES = ("pearson", "deviance")
BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 20260828
MINIMUM_INFORMATIVE_ENTITIES = 64
MAXIMUM_CONDITION_NUMBER = 1e12

PROTOCOL_BINDINGS = (
    "experiments/confirm_gse239452_citeseq.py",
    "experiments/preflight_gse239452_citeseq.py",
    "tests/test_gse239452_citeseq_confirmation.py",
    "tests/test_gse239452_citeseq_preflight.py",
    "docs/GSE239452_PREGNANCY_CITESEQ_CONFIRMATION_PROTOCOL_2026-08-28.md",
    "data/confirmation/gse239452_citeseq/candidate_designation_v1.json",
    "data/confirmation/gse239452_citeseq/source_manifest_v1.json",
    "results/development/gse239452_citeseq_metadata_preflight.json",
    "mapreg/hierarchical_conditional_coupling.py",
    "mapreg/heterogeneity_adaptive_coupling.py",
    "mapreg/table_prediction.py",
)


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


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _axis_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any], *, exclusive: bool = True) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode) as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(
            "public artifacts must remain inside the repository"
        ) from error


def _immutable_public_bytes(relative: str, commit: str) -> bytes:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise PermissionError("public commit is not an immutable SHA-1")
    url = (
        f"https://raw.githubusercontent.com/{PUBLIC_OWNER}/{PUBLIC_REPOSITORY}/"
        f"{commit}/{relative}"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "coupling-fields/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _require_designated(observed: Path, expected: Path, label: str) -> None:
    if observed.resolve() != expected.resolve():
        raise PermissionError(f"{label} path differs from the frozen designation")


def _require_open() -> None:
    if DEFAULT_TERMINAL_REFUSAL.exists():
        raise PermissionError("terminal refusal permanently closes this candidate")


def _validated_protocol_authorization(
    path: Path, authorization_commit: str, expected_status: str
) -> dict[str, Any]:
    payload = _read_json(path)
    if (
        payload.get("schema") != "gse239452-protocol-authorization/1.0"
        or payload.get("status") != expected_status
    ):
        raise PermissionError("protocol authorization is absent or has the wrong phase")
    protocol_commit = payload.get("public_protocol_commit")
    bindings = payload.get("binding_sha256")
    if not isinstance(bindings, dict) or set(bindings) != set(PROTOCOL_BINDINGS):
        raise PermissionError("protocol authorization has an incomplete binding set")
    for relative in PROTOCOL_BINDINGS:
        local = ROOT / relative
        if not local.is_file() or bindings[relative] != _sha256(local):
            raise PermissionError(f"protocol binding differs for {relative}")
        if (
            _immutable_public_bytes(relative, str(protocol_commit))
            != local.read_bytes()
        ):
            raise PermissionError(f"public protocol bytes differ for {relative}")
    if (
        _immutable_public_bytes(_relative(path), authorization_commit)
        != path.read_bytes()
    ):
        raise PermissionError("public authorization bytes differ from the local file")
    return payload


def _validated_artifact_authorization(
    path: Path,
    authorization_commit: str,
    *,
    status: str,
    artifact_path: Path,
    artifact_field: str,
) -> dict[str, Any]:
    payload = _read_json(path)
    expected = {
        "schema",
        "status",
        artifact_field,
        f"{artifact_field}_sha256",
        f"public_{artifact_field}_commit",
    }
    if (
        set(payload) != expected
        or payload.get("schema") != "gse239452-artifact-authorization/1.0"
        or payload.get("status") != status
    ):
        raise PermissionError("artifact authorization differs from the frozen schema")
    if payload.get(artifact_field) != _relative(artifact_path) or payload.get(
        f"{artifact_field}_sha256"
    ) != _sha256(artifact_path):
        raise PermissionError("authorized artifact path or digest differs")
    public_commit = str(payload.get(f"public_{artifact_field}_commit"))
    if (
        _immutable_public_bytes(_relative(artifact_path), public_commit)
        != artifact_path.read_bytes()
    ):
        raise PermissionError("public artifact bytes differ from the local file")
    if (
        _immutable_public_bytes(_relative(path), authorization_commit)
        != path.read_bytes()
    ):
        raise PermissionError(
            "public artifact authorization differs from the local file"
        )
    return payload


def _preflight_records(preflight_path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(preflight_path)
    if (
        payload.get("schema") != "gse239452-citeseq-metadata-preflight/1.0"
        or payload.get("status") != "PASS"
        or payload.get("source_manifest_sha256") != _sha256(DEFAULT_SOURCE)
        or payload.get("access_audit", {}).get("numeric_matrix_payload_values_read")
        != 0
    ):
        raise PermissionError(
            "metadata preflight differs from the frozen no-payload pass"
        )
    records = payload.get("samples")
    if not isinstance(records, list) or len(records) != 26:
        raise PermissionError("metadata preflight sample universe differs")
    by_donor = {row.get("donor"): row for row in records if isinstance(row, dict)}
    if set(by_donor) != set(CALIBRATION) | set(PILOT) | set(HELD) | {"78", "100"}:
        raise PermissionError("metadata preflight donor set differs")
    return by_donor


def _sample_manifest(source_path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(source_path)
    samples, markers = metadata._validate_manifest(payload)
    expected_donors = (*CALIBRATION, *PILOT, *HELD, "100", "78")
    expected_roles = {
        **dict.fromkeys(CALIBRATION, "calibration"),
        **dict.fromkeys(PILOT, "pilot"),
        **dict.fromkeys(HELD, "held"),
        "100": "excluded_metadata",
        "78": "excluded_metadata",
    }
    if (
        tuple(row["donor"] for row in samples) != expected_donors
        or any(row["role"] != expected_roles[row["donor"]] for row in samples)
        or payload.get("cell_selection_salt") != CELL_SALT
        or payload.get("adt_tie_salt") != ADT_TIE_SALT
        or payload.get("destroyed_link_salt") != DESTROYED_SALT
        or tuple(row["marker"] for row in markers) != MARKERS
    ):
        raise PermissionError(
            "source manifest differs from the frozen execution contract"
        )
    return {row["donor"]: row for row in samples}


def _text_axis(handle: h5py.File, path: str) -> list[str]:
    dataset = handle[path]
    if (
        not isinstance(dataset, h5py.Dataset)
        or h5py.check_string_dtype(dataset.dtype) is None
    ):
        raise ValueError(f"{path} is not a text axis")
    return [str(value) for value in dataset.asstr()[:].tolist()]


def _paired_axes(
    source_root: Path, sample: dict[str, Any], preflight: dict[str, Any]
) -> tuple[list[str], dict[str, int], dict[str, int]]:
    gex_path = source_root / sample["gex"]["h5ad"]
    adt_path = source_root / sample["adt"]["h5ad"]
    if (
        _sha256(gex_path) != preflight["gex"]["h5ad_sha256"]
        or _sha256(adt_path) != preflight["adt"]["h5ad_sha256"]
    ):
        raise PermissionError("local H5AD bytes differ from the metadata preflight")
    with h5py.File(gex_path, "r") as gex, h5py.File(adt_path, "r") as adt:
        gex_index = preflight["gex_obs_index"]
        gex_axis = [
            metadata._canonical_gex_barcode(value)
            for value in _text_axis(gex, f"obs/{gex_index}")
        ]
        adt_axis = _text_axis(adt, "obs/_index")
    gex_rows = {value: index for index, value in enumerate(gex_axis)}
    adt_rows = {value: index for index, value in enumerate(adt_axis)}
    if len(gex_rows) != len(gex_axis) or len(adt_rows) != len(adt_axis):
        raise ValueError("canonical barcode axes are not unique")
    common = sorted(set(gex_rows) & set(adt_rows))
    if (
        len(common) != preflight["common_barcode_count"]
        or _axis_sha256(common) != preflight["common_barcode_axis_sha256"]
    ):
        raise PermissionError("common barcode axis differs from metadata preflight")
    return common, gex_rows, adt_rows


def _selected_barcodes(common: list[str], donor: str) -> list[str]:
    if len(common) < CELL_BUDGET:
        raise ValueError(f"donor {donor} has fewer than {CELL_BUDGET} common barcodes")
    return sorted(
        common,
        key=lambda value: (
            hashlib.sha256(f"{CELL_SALT}|{donor}|{value}".encode()).hexdigest(),
            value,
        ),
    )[:CELL_BUDGET]


def _read_csr_rows_columns(
    path: Path, rows: list[int], columns: list[int]
) -> tuple[np.ndarray, dict[str, int]]:
    with h5py.File(path, "r") as handle:
        matrix = handle["raw/X"]
        encoding = matrix.attrs.get("encoding-type")
        if isinstance(encoding, bytes):
            encoding = encoding.decode()
        if encoding != "csr_matrix":
            raise ValueError("frozen reader requires CSR raw/X")
        column_lookup = {int(column): index for index, column in enumerate(columns)}
        output = np.zeros((len(rows), len(columns)), dtype=float)
        indices_read = 0
        for destination, row in enumerate(rows):
            start = int(matrix["indptr"][row])
            stop = int(matrix["indptr"][row + 1])
            indices = np.asarray(matrix["indices"][start:stop], dtype=np.int64)
            values = np.asarray(matrix["data"][start:stop], dtype=float)
            indices_read += len(indices)
            for source, value in zip(indices.tolist(), values.tolist()):
                target = column_lookup.get(source)
                if target is not None:
                    output[destination, target] = value
    if (
        not np.isfinite(output).all()
        or np.any(output < 0.0)
        or not np.array_equal(output, np.rint(output))
    ):
        raise ValueError(
            "selected raw matrix values are not nonnegative integer counts"
        )
    return output.astype(np.int64), {
        "raw_data_values_decoded": indices_read,
        "raw_indices_values_decoded": indices_read,
        "raw_indptr_values_read": 2 * len(rows),
    }


def _adt_states(counts: np.ndarray, barcodes: list[str], donor: str) -> np.ndarray:
    values = np.asarray(counts)
    if values.shape != (CELL_BUDGET, len(MARKERS)) or len(barcodes) != CELL_BUDGET:
        raise ValueError("ADT median split requires the frozen 512 by 9 panel")
    states = np.zeros(values.shape, dtype=np.uint8)
    for marker in range(len(MARKERS)):
        order = sorted(
            range(CELL_BUDGET),
            key=lambda cell: (
                int(values[cell, marker]),
                hashlib.sha256(
                    f"{ADT_TIE_SALT}|{donor}|{MARKERS[marker]}|{barcodes[cell]}".encode()
                ).hexdigest(),
                barcodes[cell],
            ),
        )
        states[np.asarray(order[CELL_BUDGET // 2 :], dtype=int), marker] = 1
    if not np.all(states.sum(axis=0) == CELL_BUDGET // 2):
        raise AssertionError("ADT median states do not have fixed 256/256 margins")
    return states


def _destroyed_adt(states: np.ndarray, barcodes: list[str], donor: str) -> np.ndarray:
    order = sorted(
        range(CELL_BUDGET),
        key=lambda cell: (
            hashlib.sha256(
                f"{DESTROYED_SALT}|{donor}|{barcodes[cell]}".encode()
            ).hexdigest(),
            barcodes[cell],
        ),
    )
    return np.asarray(states)[np.asarray(order, dtype=int)]


def _binary_tables(rna: np.ndarray, adt: np.ndarray) -> np.ndarray:
    first = np.asarray(rna)
    second = np.asarray(adt)
    if first.shape != (CELL_BUDGET, len(MARKERS)) or second.shape != first.shape:
        raise ValueError("state matrices differ from the frozen 512 by 9 panel")
    tables = np.zeros((len(MARKERS), len(MARKERS), 2, 2), dtype=np.int64)
    for first_marker in range(len(MARKERS)):
        for second_marker in range(len(MARKERS)):
            code = 2 * first[:, first_marker] + second[:, second_marker]
            tables[first_marker, second_marker] = np.bincount(
                code, minlength=4
            ).reshape(2, 2)
    return tables


def _adt_profiles(counts: np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    total = values.sum(axis=1, keepdims=True)
    normalized = np.divide(
        100.0 * values,
        total,
        out=np.zeros_like(values),
        where=total > 0.0,
    )
    return np.log1p(normalized).mean(axis=0)


def _reduce_one(
    donor: str,
    source_root: Path,
    manifest: dict[str, dict[str, Any]],
    preflight: dict[str, dict[str, Any]],
    *,
    read_adt_numeric: bool,
) -> dict[str, Any]:
    sample = manifest[donor]
    metadata_record = preflight[donor]
    common, gex_rows, adt_rows = _paired_axes(source_root, sample, metadata_record)
    selected = _selected_barcodes(common, donor)
    gex_counts, gex_access = _read_csr_rows_columns(
        source_root / sample["gex"]["h5ad"],
        [gex_rows[value] for value in selected],
        list(metadata_record["gex_marker_indices"]),
    )
    rna_states = (gex_counts > 0).astype(np.uint8)
    result: dict[str, Any] = {
        "donor": donor,
        "role": sample["role"],
        "pregnancy": sample["pregnancy"],
        "severity": sample["severity"],
        "cells": CELL_BUDGET,
        "selected_barcode_axis_sha256": _axis_sha256(selected),
        "common_barcode_axis_sha256": metadata_record["common_barcode_axis_sha256"],
        "common_barcode_count": metadata_record["common_barcode_count"],
        "rna_marker_counts": gex_counts.sum(axis=0).astype(int).tolist(),
        "rna_positive_counts": rna_states.sum(axis=0).astype(int).tolist(),
        "gex_access": gex_access,
    }
    if not read_adt_numeric:
        result["adt_numeric_values_read"] = 0
        return result

    adt_counts, adt_access = _read_csr_rows_columns(
        source_root / sample["adt"]["h5ad"],
        [adt_rows[value] for value in selected],
        list(metadata_record["adt_marker_indices"]),
    )
    adt_states = _adt_states(adt_counts, selected, donor)
    tables = _binary_tables(rna_states, adt_states)
    destroyed = _binary_tables(rna_states, _destroyed_adt(adt_states, selected, donor))
    result.update(
        {
            "tables": tables.tolist(),
            "destroyed_tables": destroyed.tolist(),
            "rna_profiles": (rna_states.mean(axis=0)).tolist(),
            "adt_profiles": _adt_profiles(adt_counts).tolist(),
            "adt_marker_counts": adt_counts.sum(axis=0).astype(int).tolist(),
            "adt_high_counts": adt_states.sum(axis=0).astype(int).tolist(),
            "adt_access": adt_access,
            "table_sha256": _array_sha256(tables),
            "destroyed_table_sha256": _array_sha256(destroyed),
        }
    )
    return result


def _informative(tables: np.ndarray) -> np.ndarray:
    values = np.asarray(tables)
    rows = values.sum(axis=-1)
    columns = values.sum(axis=-2)
    total = values.sum(axis=(-2, -1))
    lower = np.maximum(0, rows[..., 0] + columns[..., 0] - total)
    upper = np.minimum(rows[..., 0], columns[..., 0])
    return upper > lower


def _knn_incidence(profiles: np.ndarray, neighbors: int) -> np.ndarray:
    values = np.asarray(profiles, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(MARKERS):
        raise ValueError("graph profiles must be donor by frozen marker")
    marker_profiles = values.T
    scale = marker_profiles.std(axis=1, ddof=1)
    if np.any(scale == 0.0) or not np.isfinite(scale).all():
        raise CouplingEstimationRefusal("a fold-local marker profile has zero variance")
    standardized = (
        marker_profiles - marker_profiles.mean(axis=1, keepdims=True)
    ) / scale[:, None]
    edges: set[tuple[int, int]] = set()
    for marker in range(len(MARKERS)):
        candidates = [value for value in range(len(MARKERS)) if value != marker]
        distances = np.asarray(
            [
                np.linalg.norm(standardized[marker] - standardized[value])
                for value in candidates
            ]
        )
        order = np.asarray(candidates)[np.lexsort((np.asarray(candidates), distances))]
        edges.update(tuple(sorted((marker, int(value)))) for value in order[:neighbors])
    incidence = np.zeros((len(MARKERS), len(edges)), dtype=float)
    for edge, (left, right) in enumerate(sorted(edges)):
        incidence[left, edge] = 1.0
        incidence[right, edge] = 1.0
    return incidence


def _incidences(
    rna_profiles: np.ndarray, adt_profiles: np.ndarray, config: PrimaryConfig
) -> tuple[np.ndarray, np.ndarray]:
    if config.graph_penalty == 0.0:
        identity = np.eye(len(MARKERS), dtype=float)
        return identity, identity
    return (
        _knn_incidence(rna_profiles, config.graph_neighbors),
        _knn_incidence(adt_profiles, config.graph_neighbors),
    )


def _fit_primary(
    tables: np.ndarray,
    rna_profiles: np.ndarray,
    adt_profiles: np.ndarray,
    config: PrimaryConfig,
) -> dict[str, Any]:
    first, second = _incidences(rna_profiles, adt_profiles, config)
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
        "gradient_norm": fit.gradient_norm,
        "scaled_gradient_norm": fit.scaled_gradient_norm,
        "schur_condition_number": fit.schur_condition_number,
        "theta_curvature_condition_number": fit.theta_curvature_condition_number,
        "iterations": fit.iterations,
        "rna_incidence_sha256": _array_sha256(first),
        "adt_incidence_sha256": _array_sha256(second),
    }


def _predict_log_odds(
    log_odds: np.ndarray,
    row_margins: np.ndarray,
    column_margins: np.ndarray,
    alpha: float,
) -> np.ndarray:
    odds = np.asarray(log_odds, dtype=float)
    rows = np.asarray(row_margins, dtype=np.int64)
    columns = np.asarray(column_margins, dtype=np.int64)
    if (
        odds.shape != (len(MARKERS), len(MARKERS))
        or rows.shape
        != (
            len(MARKERS),
            len(MARKERS),
            2,
        )
        or columns.shape != rows.shape
    ):
        raise ValueError("log odds and recipient margin axes differ")
    output = np.empty((*odds.shape, 2, 2), dtype=float)
    for index in np.ndindex(odds.shape):
        output[index] = expected_binary_table_from_log_odds(
            float(alpha) * odds[index], rows[index], columns[index]
        )
    return output


def _margins(tables: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(tables)
    return values.sum(axis=-1), values.sum(axis=-2)


def _donor_loss(observed: np.ndarray, predicted: np.ndarray) -> float:
    truth = np.asarray(observed, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    support = _informative(truth)
    if np.count_nonzero(support) < MINIMUM_INFORMATIVE_ENTITIES:
        raise CouplingEstimationRefusal(
            "recipient has fewer than 64 informative entities"
        )
    truth = truth[support]
    estimate = estimate[support]
    if not np.allclose(truth.sum(axis=-1), estimate.sum(axis=-1)) or not np.allclose(
        truth.sum(axis=-2), estimate.sum(axis=-2)
    ):
        raise FloatingPointError("prediction changed a recipient margin")
    positive = truth > 0.0
    if np.any(estimate[positive] <= 0.0) or not np.isfinite(estimate).all():
        raise FloatingPointError(
            "prediction assigns nonpositive mass to an observed cell"
        )
    terms = np.zeros_like(truth)
    terms[positive] = truth[positive] * np.log(truth[positive] / estimate[positive])
    return float((2.0 * terms.sum(axis=(-2, -1)) / CELL_BUDGET).mean())


def _fractional_pearson(table: np.ndarray) -> float:
    values = np.asarray(table, dtype=float)
    total = float(values.sum())
    rows = values.sum(axis=1)
    columns = values.sum(axis=0)
    determinant = float(values[0, 0] * values[1, 1] - values[0, 1] * values[1, 0])
    return determinant * math.sqrt(total / float(np.prod(rows) * np.prod(columns)))


def _fractional_deviance(table: np.ndarray) -> float:
    values = np.asarray(table, dtype=float)
    expected = np.outer(values.sum(axis=1), values.sum(axis=0)) / values.sum()
    positive = values > 0.0
    deviance = 2.0 * float(
        np.sum(values[positive] * np.log(values[positive] / expected[positive]))
    )
    determinant = float(values[0, 0] * values[1, 1] - values[0, 1] * values[1, 0])
    return math.copysign(math.sqrt(max(deviance, 0.0)), determinant)


def _classical_table(
    coordinate: float, rows: np.ndarray, columns: np.ndarray, family: str
) -> np.ndarray:
    row_margin = np.asarray(rows, dtype=float)
    column_margin = np.asarray(columns, dtype=float)
    total = float(row_margin.sum())
    lower = float(max(0.0, row_margin[0] + column_margin[0] - total))
    upper = float(min(row_margin[0], column_margin[0]))
    if upper <= lower:
        return np.asarray(
            [
                [lower, row_margin[0] - lower],
                [column_margin[0] - lower, row_margin[1] - column_margin[0] + lower],
            ]
        )
    left = float(np.nextafter(lower, upper))
    right = float(np.nextafter(upper, lower))

    def table_at(value: float) -> np.ndarray:
        return np.asarray(
            [
                [value, row_margin[0] - value],
                [
                    column_margin[0] - value,
                    row_margin[1] - column_margin[0] + value,
                ],
            ]
        )

    statistic = _fractional_pearson if family == "pearson" else _fractional_deviance
    target = min(
        max(float(coordinate), statistic(table_at(left))), statistic(table_at(right))
    )
    for _ in range(96):
        midpoint = 0.5 * (left + right)
        if statistic(table_at(midpoint)) < target:
            left = midpoint
        else:
            right = midpoint
    return table_at(0.5 * (left + right))


def _residual_pool(tables: np.ndarray, family: str) -> np.ndarray:
    values = np.asarray(tables).reshape(len(tables), -1, 2, 2)
    support = _informative(values)
    if np.any(support.sum(axis=0) < 2):
        raise CouplingEstimationRefusal(
            "residual comparator has too few informative donors"
        )
    coordinates = np.full(support.shape, np.nan, dtype=float)
    statistic = (
        signed_pearson_coordinate if family == "pearson" else signed_deviance_coordinate
    )
    for donor, entity in np.argwhere(support):
        coordinates[donor, entity] = statistic(values[donor, entity]) / math.sqrt(
            CELL_BUDGET
        )
    pooled = np.nanmean(coordinates, axis=0)
    if not np.isfinite(pooled).all():
        raise CouplingEstimationRefusal(
            "residual comparator does not cover every entity"
        )
    return pooled.reshape(len(MARKERS), len(MARKERS))


def _predict_residual(
    pooled: np.ndarray,
    row_margins: np.ndarray,
    column_margins: np.ndarray,
    config: ResidualConfig,
) -> np.ndarray:
    values = np.asarray(pooled, dtype=float)
    output = np.empty((*values.shape, 2, 2), dtype=float)
    for index in np.ndindex(values.shape):
        coordinate = (
            config.transport_multiplier * values[index] * math.sqrt(CELL_BUDGET)
        )
        output[index] = _classical_table(
            coordinate, row_margins[index], column_margins[index], config.family
        )
    return output


def _primary_configs() -> list[PrimaryConfig]:
    configs: list[PrimaryConfig] = []
    for neighbors, eta, ridge, graph, alpha in product(
        NEIGHBOR_GRID,
        HETEROGENEITY_GRID,
        RIDGE_GRID,
        GRAPH_GRID,
        ALPHA_GRID,
    ):
        if graph == 0.0 and neighbors != NEIGHBOR_GRID[0]:
            continue
        configs.append(PrimaryConfig(neighbors, eta, ridge, graph, alpha))
    return configs


def _config_key(config: PrimaryConfig | ResidualConfig) -> str:
    return json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))


def _records_arrays(
    records: dict[str, dict[str, Any]], donors: tuple[str, ...] | list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ordered = list(donors)
    return (
        np.asarray([records[donor]["tables"] for donor in ordered], dtype=np.int64),
        np.asarray(
            [records[donor]["destroyed_tables"] for donor in ordered], dtype=np.int64
        ),
        np.asarray([records[donor]["rna_profiles"] for donor in ordered], dtype=float),
        np.asarray([records[donor]["adt_profiles"] for donor in ordered], dtype=float),
    )


def _select_calibration(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    configs = _primary_configs()
    primary_losses = {config: np.full(len(CALIBRATION), np.nan) for config in configs}
    residual_configs = [
        ResidualConfig(family, alpha)
        for family, alpha in product(RESIDUAL_FAMILIES, ALPHA_GRID)
    ]
    residual_losses = {
        config: np.full(len(CALIBRATION), np.nan) for config in residual_configs
    }
    refusals: list[dict[str, Any]] = []

    for fold, held_donor in enumerate(CALIBRATION):
        training = tuple(donor for donor in CALIBRATION if donor != held_donor)
        tables, _, rna_profiles, adt_profiles = _records_arrays(records, training)
        truth = np.asarray(records[held_donor]["tables"], dtype=np.int64)
        rows, columns = _margins(truth)
        structural: dict[
            tuple[int, float, float, float], dict[str, Any] | Exception
        ] = {}
        for config in configs:
            key = (
                config.graph_neighbors,
                config.heterogeneity_penalty,
                config.ridge_penalty,
                config.graph_penalty,
            )
            if key not in structural:
                try:
                    structural[key] = _fit_primary(
                        tables, rna_profiles, adt_profiles, config
                    )
                except (
                    ValueError,
                    FloatingPointError,
                    CouplingEstimationRefusal,
                ) as error:
                    structural[key] = error
            fitted = structural[key]
            if isinstance(fitted, Exception):
                refusals.append(
                    {
                        "fold": held_donor,
                        "configuration": asdict(config),
                        "reason": str(fitted),
                    }
                )
                continue
            prediction = _predict_log_odds(
                fitted["population_log_odds"],
                rows,
                columns,
                config.transport_multiplier,
            )
            primary_losses[config][fold] = _donor_loss(truth, prediction)

        for config in residual_configs:
            try:
                pooled = _residual_pool(tables, config.family)
                prediction = _predict_residual(pooled, rows, columns, config)
                residual_losses[config][fold] = _donor_loss(truth, prediction)
            except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
                refusals.append(
                    {
                        "fold": held_donor,
                        "configuration": asdict(config),
                        "reason": str(error),
                    }
                )

    available_primary = [
        config for config, values in primary_losses.items() if np.isfinite(values).all()
    ]
    available_residual = [
        config
        for config, values in residual_losses.items()
        if np.isfinite(values).all()
    ]
    if not available_primary or not available_residual:
        raise CouplingEstimationRefusal("calibration CV has no complete candidate")
    selected_primary = min(
        available_primary,
        key=lambda config: (float(primary_losses[config].mean()), config),
    )
    selected_residual = min(
        available_residual,
        key=lambda config: (float(residual_losses[config].mean()), config),
    )
    graph_zero = min(
        (config for config in available_primary if config.graph_penalty == 0.0),
        key=lambda config: (float(primary_losses[config].mean()), config),
    )
    return {
        "primary": asdict(selected_primary),
        "best_residual": asdict(selected_residual),
        "graph_zero_diagnostic": asdict(graph_zero),
        "primary_candidates": [
            {
                "configuration": asdict(config),
                "fold_losses": {
                    donor: float(value)
                    for donor, value in zip(CALIBRATION, primary_losses[config])
                },
                "mean_loss": float(primary_losses[config].mean()),
            }
            for config in available_primary
        ],
        "residual_candidates": [
            {
                "configuration": asdict(config),
                "fold_losses": {
                    donor: float(value)
                    for donor, value in zip(CALIBRATION, residual_losses[config])
                },
                "mean_loss": float(residual_losses[config].mean()),
            }
            for config in available_residual
        ],
        "refusals": refusals,
    }


def _primary_from_dict(value: dict[str, Any]) -> PrimaryConfig:
    return PrimaryConfig(**value)


def _residual_from_dict(value: dict[str, Any]) -> ResidualConfig:
    return ResidualConfig(**value)


def _fit_panel(
    records: dict[str, dict[str, Any]],
    donors: tuple[str, ...] | list[str],
    selection: dict[str, Any],
) -> dict[str, Any]:
    tables, destroyed, rna_profiles, adt_profiles = _records_arrays(records, donors)
    primary_config = _primary_from_dict(selection["primary"])
    primary = _fit_primary(tables, rna_profiles, adt_profiles, primary_config)
    destroyed_fit = _fit_primary(destroyed, rna_profiles, adt_profiles, primary_config)
    residual_config = _residual_from_dict(selection["best_residual"])
    residual = _residual_pool(tables, residual_config.family)
    graph_zero_config = _primary_from_dict(selection["graph_zero_diagnostic"])
    graph_zero = _fit_primary(tables, rna_profiles, adt_profiles, graph_zero_config)
    return {
        "primary": {
            "configuration": asdict(primary_config),
            "population_log_odds": primary["population_log_odds"].tolist(),
            "fit_certificate": {
                key: value
                for key, value in primary.items()
                if key != "population_log_odds"
            },
        },
        "destroyed_link": {
            "configuration": asdict(primary_config),
            "population_log_odds": destroyed_fit["population_log_odds"].tolist(),
            "fit_certificate": {
                key: value
                for key, value in destroyed_fit.items()
                if key != "population_log_odds"
            },
        },
        "best_residual": {
            "configuration": asdict(residual_config),
            "pooled_coordinate": residual.tolist(),
        },
        "graph_zero_diagnostic": {
            "configuration": asdict(graph_zero_config),
            "population_log_odds": graph_zero["population_log_odds"].tolist(),
            "fit_certificate": {
                key: value
                for key, value in graph_zero.items()
                if key != "population_log_odds"
            },
        },
    }


def _predict_panel(
    models: dict[str, Any], row_margins: np.ndarray, column_margins: np.ndarray
) -> dict[str, np.ndarray]:
    primary = models["primary"]
    primary_config = _primary_from_dict(primary["configuration"])
    destroyed = models["destroyed_link"]
    residual = models["best_residual"]
    graph_zero = models["graph_zero_diagnostic"]
    graph_zero_config = _primary_from_dict(graph_zero["configuration"])
    return {
        "primary": _predict_log_odds(
            np.asarray(primary["population_log_odds"]),
            row_margins,
            column_margins,
            primary_config.transport_multiplier,
        ),
        "destroyed_link": _predict_log_odds(
            np.asarray(destroyed["population_log_odds"]),
            row_margins,
            column_margins,
            primary_config.transport_multiplier,
        ),
        "best_residual": _predict_residual(
            np.asarray(residual["pooled_coordinate"]),
            row_margins,
            column_margins,
            _residual_from_dict(residual["configuration"]),
        ),
        "graph_zero_diagnostic": _predict_log_odds(
            np.asarray(graph_zero["population_log_odds"]),
            row_margins,
            column_margins,
            graph_zero_config.transport_multiplier,
        ),
    }


def _panel_losses(
    records: dict[str, dict[str, Any]], donors: tuple[str, ...], models: dict[str, Any]
) -> dict[str, np.ndarray]:
    losses = {
        method: np.empty(len(donors), dtype=float)
        for method in (
            "primary",
            "best_residual",
            "destroyed_link",
            "graph_zero_diagnostic",
        )
    }
    for index, donor in enumerate(donors):
        truth = np.asarray(records[donor]["tables"], dtype=np.int64)
        rows, columns = _margins(truth)
        predictions = _predict_panel(models, rows, columns)
        for method, prediction in predictions.items():
            losses[method][index] = _donor_loss(truth, prediction)
    return losses


def _comparison(
    donors: tuple[str, ...],
    primary: np.ndarray,
    comparator: np.ndarray,
    required_favorable: int,
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    primary_values = np.asarray(primary, dtype=float)
    comparator_values = np.asarray(comparator, dtype=float)
    if (
        primary_values.shape != (len(donors),)
        or comparator_values.shape != primary_values.shape
    ):
        raise ValueError("comparison requires one paired loss per donor")
    if (
        not np.isfinite(primary_values).all()
        or not np.isfinite(comparator_values).all()
    ):
        raise ValueError("comparison losses must be finite")
    if comparator_values.mean() <= 0.0:
        raise ValueError("comparison mean must be positive")
    difference = primary_values - comparator_values
    if bootstrap_indices.shape != (BOOTSTRAPS, len(donors)):
        raise ValueError("paired bootstrap index shape differs from the freeze")
    bootstrap = difference[bootstrap_indices].mean(axis=1)
    interval = np.quantile(bootstrap, [0.025, 0.975], method="linear")
    relative = 1.0 - float(primary_values.mean() / comparator_values.mean())
    favorable = int(np.count_nonzero(difference < 0.0))
    passes = {
        "relative_deviance_reduction_at_least_five_percent": bool(relative >= 0.05),
        "paired_bootstrap_upper_95_below_zero": bool(interval[1] < 0.0),
        "favorable_donor_count_reached": bool(favorable >= required_favorable),
    }
    return {
        "primary_mean_loss": float(primary_values.mean()),
        "comparator_mean_loss": float(comparator_values.mean()),
        "relative_deviance_reduction": relative,
        "paired_bootstrap_95_ci": interval.tolist(),
        "paired_bootstrap_draws": BOOTSTRAPS,
        "paired_bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_unit": "physical donor",
        "favorable_donors": favorable,
        "required_favorable_donors": required_favorable,
        "donor_differences_primary_minus_comparator": {
            donor: float(value) for donor, value in zip(donors, difference)
        },
        "passes": passes,
        "passes_all": all(passes.values()),
    }


def _gate(
    donors: tuple[str, ...], losses: dict[str, np.ndarray], required_favorable: int
) -> dict[str, Any]:
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(
        0, len(donors), size=(BOOTSTRAPS, len(donors)), endpoint=False
    )
    comparisons = {
        comparator: _comparison(
            donors,
            losses["primary"],
            losses[comparator],
            required_favorable,
            indices,
        )
        for comparator in ("best_residual", "destroyed_link")
    }
    return {
        "comparisons": comparisons,
        "passes": all(value["passes_all"] for value in comparisons.values()),
        "graph_vs_zero_diagnostic": {
            "primary_mean_loss": float(losses["primary"].mean()),
            "graph_zero_mean_loss": float(losses["graph_zero_diagnostic"].mean()),
            "relative_reduction_vs_graph_zero": 1.0
            - float(losses["primary"].mean() / losses["graph_zero_diagnostic"].mean()),
            "gated": False,
        },
    }


def reduce_development(
    source_path: Path,
    preflight_path: Path,
    authorization_path: Path,
    authorization_commit: str,
    source_root: Path,
    attempt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    _require_open()
    for observed, expected, label in (
        (source_path, DEFAULT_SOURCE, "source manifest"),
        (preflight_path, DEFAULT_PREFLIGHT, "metadata preflight"),
        (
            authorization_path,
            DEFAULT_DEVELOPMENT_AUTHORIZATION,
            "development authorization",
        ),
        (attempt_path, DEFAULT_DEVELOPMENT_ATTEMPT, "development attempt"),
        (output_path, DEFAULT_REDUCED, "reduced development"),
    ):
        _require_designated(observed, expected, label)
    permit = _validated_protocol_authorization(
        authorization_path, authorization_commit, "DEVELOPMENT_ACCESS_AUTHORIZED"
    )
    manifest = _sample_manifest(source_path)
    preflight = _preflight_records(preflight_path)
    if attempt_path.exists() or output_path.exists():
        raise FileExistsError("development attempt is one-shot")
    _write_json(
        attempt_path,
        {
            "schema": "gse239452-development-attempt/1.0",
            "status": "TERMINAL_ATTEMPT_STARTED",
            "created_at_utc": _timestamp(),
            "authorization_sha256": _sha256(authorization_path),
            "public_authorization_commit": authorization_commit,
            "numeric_development_access_begins_after_this_record": True,
            "held_numeric_values_read": 0,
        },
    )
    records = [
        _reduce_one(
            donor,
            source_root,
            manifest,
            preflight,
            read_adt_numeric=True,
        )
        for donor in (*CALIBRATION, *PILOT)
    ]
    payload = {
        "schema": "gse239452-reduced-development/1.0",
        "status": "DEVELOPMENT_REDUCTION_COMPLETE",
        "created_at_utc": _timestamp(),
        "source_manifest_sha256": _sha256(source_path),
        "metadata_preflight_sha256": _sha256(preflight_path),
        "development_authorization": {
            "path": _relative(authorization_path),
            "sha256": _sha256(authorization_path),
            "public_commit": authorization_commit,
            "public_protocol_commit": permit["public_protocol_commit"],
        },
        "development_attempt": {
            "path": _relative(attempt_path),
            "sha256": _sha256(attempt_path),
        },
        "runner_sha256": _sha256(Path(__file__)),
        "calibration_donors": list(CALIBRATION),
        "pilot_donors": list(PILOT),
        "samples": records,
        "access_audit": {
            "calibration_samples_read": len(CALIBRATION),
            "pilot_samples_read": len(PILOT),
            "held_gex_numeric_values_read": 0,
            "held_adt_numeric_values_read": 0,
        },
    }
    _write_json(output_path, payload)
    return payload


def _validated_reduced(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    expected_fields = {
        "schema",
        "status",
        "created_at_utc",
        "source_manifest_sha256",
        "metadata_preflight_sha256",
        "development_authorization",
        "development_attempt",
        "runner_sha256",
        "calibration_donors",
        "pilot_donors",
        "samples",
        "access_audit",
    }
    if (
        set(payload) != expected_fields
        or payload.get("schema") != "gse239452-reduced-development/1.0"
        or payload.get("status") != "DEVELOPMENT_REDUCTION_COMPLETE"
        or payload.get("source_manifest_sha256") != _sha256(DEFAULT_SOURCE)
        or payload.get("metadata_preflight_sha256") != _sha256(DEFAULT_PREFLIGHT)
        or payload.get("runner_sha256") != _sha256(Path(__file__))
        or payload.get("calibration_donors") != list(CALIBRATION)
        or payload.get("pilot_donors") != list(PILOT)
        or payload.get("access_audit")
        != {
            "calibration_samples_read": len(CALIBRATION),
            "pilot_samples_read": len(PILOT),
            "held_gex_numeric_values_read": 0,
            "held_adt_numeric_values_read": 0,
        }
    ):
        raise PermissionError("reduced development differs from the frozen runner")
    authorization = payload.get("development_authorization")
    if not isinstance(authorization, dict) or set(authorization) != {
        "path",
        "sha256",
        "public_commit",
        "public_protocol_commit",
    }:
        raise PermissionError("reduced development authorization record differs")
    authorization_commit = str(authorization.get("public_commit"))
    permit = _validated_protocol_authorization(
        DEFAULT_DEVELOPMENT_AUTHORIZATION,
        authorization_commit,
        "DEVELOPMENT_ACCESS_AUTHORIZED",
    )
    if authorization != {
        "path": _relative(DEFAULT_DEVELOPMENT_AUTHORIZATION),
        "sha256": _sha256(DEFAULT_DEVELOPMENT_AUTHORIZATION),
        "public_commit": authorization_commit,
        "public_protocol_commit": permit["public_protocol_commit"],
    }:
        raise PermissionError("reduced development authorization does not replay")
    attempt_record = payload.get("development_attempt")
    if not isinstance(attempt_record, dict) or attempt_record != {
        "path": _relative(DEFAULT_DEVELOPMENT_ATTEMPT),
        "sha256": _sha256(DEFAULT_DEVELOPMENT_ATTEMPT),
    }:
        raise PermissionError("reduced development attempt binding differs")
    attempt = _read_json(DEFAULT_DEVELOPMENT_ATTEMPT)
    if (
        set(attempt)
        != {
            "schema",
            "status",
            "created_at_utc",
            "authorization_sha256",
            "public_authorization_commit",
            "numeric_development_access_begins_after_this_record",
            "held_numeric_values_read",
        }
        or attempt.get("schema") != "gse239452-development-attempt/1.0"
        or attempt.get("status") != "TERMINAL_ATTEMPT_STARTED"
        or attempt.get("authorization_sha256")
        != _sha256(DEFAULT_DEVELOPMENT_AUTHORIZATION)
        or attempt.get("public_authorization_commit") != authorization_commit
        or attempt.get("numeric_development_access_begins_after_this_record")
        is not True
        or attempt.get("held_numeric_values_read") != 0
    ):
        raise PermissionError("development attempt does not replay")
    records = payload.get("samples")
    if not isinstance(records, list) or len(records) != len(CALIBRATION) + len(PILOT):
        raise PermissionError("reduced development sample count differs")
    by_donor = {row.get("donor"): row for row in records if isinstance(row, dict)}
    if set(by_donor) != set(CALIBRATION) | set(PILOT):
        raise PermissionError("reduced development donor set differs")
    manifest = _sample_manifest(DEFAULT_SOURCE)
    for donor, record in by_donor.items():
        tables = np.asarray(record.get("tables"), dtype=np.int64)
        destroyed = np.asarray(record.get("destroyed_tables"), dtype=np.int64)
        if (
            record.get("role") != manifest[donor]["role"]
            or record.get("cells") != CELL_BUDGET
            or tables.shape != (len(MARKERS), len(MARKERS), 2, 2)
            or destroyed.shape != tables.shape
            or not np.all(tables.sum(axis=(-2, -1)) == CELL_BUDGET)
            or not np.array_equal(_margins(tables)[0], _margins(destroyed)[0])
            or not np.array_equal(_margins(tables)[1], _margins(destroyed)[1])
            or record.get("table_sha256") != _array_sha256(tables)
            or record.get("destroyed_table_sha256") != _array_sha256(destroyed)
            or record.get("adt_high_counts") != [CELL_BUDGET // 2] * len(MARKERS)
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(record.get("selected_barcode_axis_sha256"))
            )
        ):
            raise PermissionError(f"reduced record differs for donor {donor}")
    return by_donor


def _pilot_analysis(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    selection = _select_calibration(records)
    calibration_models = _fit_panel(records, CALIBRATION, selection)
    pilot_losses = _panel_losses(records, PILOT, calibration_models)
    gate = _gate(PILOT, pilot_losses, required_favorable=7)
    full_models = (
        _fit_panel(records, (*CALIBRATION, *PILOT), selection)
        if gate["passes"]
        else None
    )
    return {
        "selection": selection,
        "calibration_models": calibration_models,
        "pilot_losses": {
            method: {donor: float(value) for donor, value in zip(PILOT, values)}
            for method, values in pilot_losses.items()
        },
        "pilot_gate": gate,
        "promotion_comparators": ["best_residual", "destroyed_link"],
        "graph_vs_zero_is_diagnostic_only": True,
        "all_development_models": full_models,
        "held_gex_access_authorized": bool(gate["passes"]),
        "held_adt_access_authorized": False,
    }


def fit_pilot(reduced_path: Path, output_path: Path) -> dict[str, Any]:
    _require_open()
    _require_designated(reduced_path, DEFAULT_REDUCED, "reduced development")
    _require_designated(output_path, DEFAULT_PILOT, "pilot result")
    if output_path.exists():
        raise FileExistsError("pilot result is one-shot")
    records = _validated_reduced(reduced_path)
    analysis = _pilot_analysis(records)
    payload = {
        "schema": "gse239452-pilot-result/1.0",
        "status": "PILOT_PASS" if analysis["pilot_gate"]["passes"] else "PILOT_FAIL",
        "created_at_utc": _timestamp(),
        "runner_sha256": _sha256(Path(__file__)),
        "reduced_development_sha256": _sha256(reduced_path),
        "calibration_donors": list(CALIBRATION),
        "pilot_donors": list(PILOT),
        **analysis,
    }
    _write_json(output_path, payload)
    return payload


def _validated_pilot(path: Path, *, require_pass: bool) -> dict[str, Any]:
    payload = _read_json(path)
    expected_fields = {
        "schema",
        "status",
        "created_at_utc",
        "runner_sha256",
        "reduced_development_sha256",
        "calibration_donors",
        "pilot_donors",
        "selection",
        "calibration_models",
        "pilot_losses",
        "pilot_gate",
        "promotion_comparators",
        "graph_vs_zero_is_diagnostic_only",
        "all_development_models",
        "held_gex_access_authorized",
        "held_adt_access_authorized",
    }
    if (
        set(payload) != expected_fields
        or payload.get("schema") != "gse239452-pilot-result/1.0"
        or payload.get("runner_sha256") != _sha256(Path(__file__))
        or payload.get("reduced_development_sha256") != _sha256(DEFAULT_REDUCED)
        or payload.get("calibration_donors") != list(CALIBRATION)
        or payload.get("pilot_donors") != list(PILOT)
        or payload.get("promotion_comparators") != ["best_residual", "destroyed_link"]
        or payload.get("graph_vs_zero_is_diagnostic_only") is not True
    ):
        raise PermissionError("pilot result differs from the frozen runner")
    replay = _pilot_analysis(_validated_reduced(DEFAULT_REDUCED))
    for field, expected in replay.items():
        if _canonical_json_sha256(payload.get(field)) != _canonical_json_sha256(
            expected
        ):
            raise PermissionError(f"pilot {field} does not replay exactly")
    if require_pass and (
        payload.get("status") != "PILOT_PASS"
        or payload.get("pilot_gate", {}).get("passes") is not True
        or payload.get("all_development_models") is None
    ):
        raise PermissionError("pilot gate did not authorize held-GEX access")
    return payload


def _held_margin_arrays(rna_positive: list[int]) -> tuple[np.ndarray, np.ndarray]:
    positive = np.asarray(rna_positive, dtype=np.int64)
    if positive.shape != (len(MARKERS),) or np.any(
        (positive <= 0) | (positive >= CELL_BUDGET)
    ):
        raise CouplingEstimationRefusal("held RNA margins do not support every marker")
    rows = np.empty((len(MARKERS), len(MARKERS), 2), dtype=np.int64)
    rows[..., 0] = (CELL_BUDGET - positive)[:, None]
    rows[..., 1] = positive[:, None]
    columns = np.empty_like(rows)
    columns[..., 0] = CELL_BUDGET // 2
    columns[..., 1] = CELL_BUDGET // 2
    return rows, columns


def predict_held(
    source_path: Path,
    preflight_path: Path,
    pilot_path: Path,
    authorization_path: Path,
    authorization_commit: str,
    source_root: Path,
    attempt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    _require_open()
    for observed, expected, label in (
        (source_path, DEFAULT_SOURCE, "source manifest"),
        (preflight_path, DEFAULT_PREFLIGHT, "metadata preflight"),
        (pilot_path, DEFAULT_PILOT, "pilot result"),
        (authorization_path, DEFAULT_HELD_GEX_AUTHORIZATION, "held GEX authorization"),
        (attempt_path, DEFAULT_PREDICTION_ATTEMPT, "prediction attempt"),
        (output_path, DEFAULT_PREDICTION, "held predictions"),
    ):
        _require_designated(observed, expected, label)
    pilot = _validated_pilot(pilot_path, require_pass=True)
    _validated_artifact_authorization(
        authorization_path,
        authorization_commit,
        status="HELD_GEX_ACCESS_AUTHORIZED",
        artifact_path=pilot_path,
        artifact_field="pilot_result",
    )
    if attempt_path.exists() or output_path.exists():
        raise FileExistsError("held prediction is one-shot")
    _write_json(
        attempt_path,
        {
            "schema": "gse239452-prediction-attempt/1.0",
            "status": "TERMINAL_ATTEMPT_STARTED",
            "created_at_utc": _timestamp(),
            "pilot_result_sha256": _sha256(pilot_path),
            "authorization_sha256": _sha256(authorization_path),
            "public_authorization_commit": authorization_commit,
            "held_gex_numeric_access_begins_after_this_record": True,
            "held_adt_numeric_values_read": 0,
        },
    )
    manifest = _sample_manifest(source_path)
    preflight = _preflight_records(preflight_path)
    models = pilot["all_development_models"]
    samples: list[dict[str, Any]] = []
    for donor in HELD:
        aggregate = _reduce_one(
            donor,
            source_root,
            manifest,
            preflight,
            read_adt_numeric=False,
        )
        rows, columns = _held_margin_arrays(aggregate["rna_positive_counts"])
        predictions = _predict_panel(models, rows, columns)
        samples.append(
            {
                "donor": donor,
                "severity": aggregate["severity"],
                "cells": CELL_BUDGET,
                "selected_barcode_axis_sha256": aggregate[
                    "selected_barcode_axis_sha256"
                ],
                "common_barcode_axis_sha256": aggregate["common_barcode_axis_sha256"],
                "common_barcode_count": aggregate["common_barcode_count"],
                "rna_positive_counts": aggregate["rna_positive_counts"],
                "row_margins": rows.tolist(),
                "column_margins": columns.tolist(),
                "predicted_tables": {
                    method: table.tolist() for method, table in predictions.items()
                },
                "gex_access": aggregate["gex_access"],
                "adt_numeric_values_read": 0,
            }
        )
    payload = {
        "schema": "gse239452-held-predictions/1.0",
        "status": "PREDICTIONS_FROZEN",
        "created_at_utc": _timestamp(),
        "runner_sha256": _sha256(Path(__file__)),
        "source_manifest_sha256": _sha256(source_path),
        "metadata_preflight_sha256": _sha256(preflight_path),
        "pilot_result_sha256": _sha256(pilot_path),
        "held_gex_authorization": {
            "path": _relative(authorization_path),
            "sha256": _sha256(authorization_path),
            "public_commit": authorization_commit,
        },
        "prediction_attempt": {
            "path": _relative(attempt_path),
            "sha256": _sha256(attempt_path),
        },
        "held_donors": list(HELD),
        "models": models,
        "samples": samples,
        "reconstruction": "noncentral-hypergeometric expectation at frozen finite full log odds; no clipping",
        "access_audit": {
            "held_gex_samples_read": len(HELD),
            "held_adt_barcode_axes_read": len(HELD),
            "held_adt_files_opaque_sha256_hashed": len(HELD),
            "held_adt_numeric_values_read": 0,
            "held_pairings_formed": 0,
            "held_truth_tables_formed": 0,
        },
    }
    _write_json(output_path, payload)
    return payload


def _validated_prediction(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    expected_fields = {
        "schema",
        "status",
        "created_at_utc",
        "runner_sha256",
        "source_manifest_sha256",
        "metadata_preflight_sha256",
        "pilot_result_sha256",
        "held_gex_authorization",
        "prediction_attempt",
        "held_donors",
        "models",
        "samples",
        "reconstruction",
        "access_audit",
    }
    expected_access = {
        "held_gex_samples_read": len(HELD),
        "held_adt_barcode_axes_read": len(HELD),
        "held_adt_files_opaque_sha256_hashed": len(HELD),
        "held_adt_numeric_values_read": 0,
        "held_pairings_formed": 0,
        "held_truth_tables_formed": 0,
    }
    if (
        set(payload) != expected_fields
        or payload.get("schema") != "gse239452-held-predictions/1.0"
        or payload.get("status") != "PREDICTIONS_FROZEN"
        or payload.get("runner_sha256") != _sha256(Path(__file__))
        or payload.get("source_manifest_sha256") != _sha256(DEFAULT_SOURCE)
        or payload.get("metadata_preflight_sha256") != _sha256(DEFAULT_PREFLIGHT)
        or payload.get("pilot_result_sha256") != _sha256(DEFAULT_PILOT)
        or payload.get("held_donors") != list(HELD)
        or payload.get("reconstruction")
        != "noncentral-hypergeometric expectation at frozen finite full log odds; no clipping"
        or payload.get("access_audit") != expected_access
    ):
        raise PermissionError("held predictions differ from the frozen runner")
    pilot = _validated_pilot(DEFAULT_PILOT, require_pass=True)
    models = payload.get("models")
    if _canonical_json_sha256(models) != _canonical_json_sha256(
        pilot["all_development_models"]
    ):
        raise PermissionError("held prediction models differ from the replayed pilot")
    authorization = payload.get("held_gex_authorization")
    if not isinstance(authorization, dict) or set(authorization) != {
        "path",
        "sha256",
        "public_commit",
    }:
        raise PermissionError("held-GEX authorization record differs")
    authorization_commit = str(authorization.get("public_commit"))
    _validated_artifact_authorization(
        DEFAULT_HELD_GEX_AUTHORIZATION,
        authorization_commit,
        status="HELD_GEX_ACCESS_AUTHORIZED",
        artifact_path=DEFAULT_PILOT,
        artifact_field="pilot_result",
    )
    if authorization != {
        "path": _relative(DEFAULT_HELD_GEX_AUTHORIZATION),
        "sha256": _sha256(DEFAULT_HELD_GEX_AUTHORIZATION),
        "public_commit": authorization_commit,
    }:
        raise PermissionError("held-GEX authorization does not replay")
    attempt_record = payload.get("prediction_attempt")
    if not isinstance(attempt_record, dict) or attempt_record != {
        "path": _relative(DEFAULT_PREDICTION_ATTEMPT),
        "sha256": _sha256(DEFAULT_PREDICTION_ATTEMPT),
    }:
        raise PermissionError("prediction attempt binding differs")
    attempt = _read_json(DEFAULT_PREDICTION_ATTEMPT)
    if (
        set(attempt)
        != {
            "schema",
            "status",
            "created_at_utc",
            "pilot_result_sha256",
            "authorization_sha256",
            "public_authorization_commit",
            "held_gex_numeric_access_begins_after_this_record",
            "held_adt_numeric_values_read",
        }
        or attempt.get("schema") != "gse239452-prediction-attempt/1.0"
        or attempt.get("status") != "TERMINAL_ATTEMPT_STARTED"
        or attempt.get("pilot_result_sha256") != _sha256(DEFAULT_PILOT)
        or attempt.get("authorization_sha256")
        != _sha256(DEFAULT_HELD_GEX_AUTHORIZATION)
        or attempt.get("public_authorization_commit") != authorization_commit
        or attempt.get("held_gex_numeric_access_begins_after_this_record") is not True
        or attempt.get("held_adt_numeric_values_read") != 0
    ):
        raise PermissionError("prediction attempt does not replay")
    samples = payload.get("samples")
    if not isinstance(samples, list) or [row.get("donor") for row in samples] != list(
        HELD
    ):
        raise PermissionError("held prediction sample order differs")
    expected_methods = {
        "primary",
        "best_residual",
        "destroyed_link",
        "graph_zero_diagnostic",
    }
    manifest = _sample_manifest(DEFAULT_SOURCE)
    preflight = _preflight_records(DEFAULT_PREFLIGHT)
    expected_sample_fields = {
        "donor",
        "severity",
        "cells",
        "selected_barcode_axis_sha256",
        "common_barcode_axis_sha256",
        "common_barcode_count",
        "rna_positive_counts",
        "row_margins",
        "column_margins",
        "predicted_tables",
        "gex_access",
        "adt_numeric_values_read",
    }
    for row in samples:
        donor = row.get("donor")
        rows = np.asarray(row.get("row_margins"), dtype=np.int64)
        columns = np.asarray(row.get("column_margins"), dtype=np.int64)
        expected_rows, expected_columns = _held_margin_arrays(
            row.get("rna_positive_counts")
        )
        gex_access = row.get("gex_access")
        valid_gex_access = (
            isinstance(gex_access, dict)
            and set(gex_access)
            == {
                "raw_data_values_decoded",
                "raw_indices_values_decoded",
                "raw_indptr_values_read",
            }
            and all(type(value) is int and value >= 0 for value in gex_access.values())
            and gex_access["raw_data_values_decoded"]
            == gex_access["raw_indices_values_decoded"]
            and gex_access["raw_indptr_values_read"] == 2 * CELL_BUDGET
        )
        if (
            set(row) != expected_sample_fields
            or row.get("severity") != manifest[donor]["severity"]
            or row.get("cells") != CELL_BUDGET
            or row.get("common_barcode_axis_sha256")
            != preflight[donor]["common_barcode_axis_sha256"]
            or row.get("common_barcode_count")
            != preflight[donor]["common_barcode_count"]
            or rows.shape != (len(MARKERS), len(MARKERS), 2)
            or columns.shape != rows.shape
            or np.any(rows < 0)
            or np.any(columns < 0)
            or not np.all(rows.sum(axis=-1) == CELL_BUDGET)
            or not np.all(columns == CELL_BUDGET // 2)
            or not np.array_equal(rows, expected_rows)
            or not np.array_equal(columns, expected_columns)
            or row.get("adt_numeric_values_read") != 0
            or not valid_gex_access
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(row.get("selected_barcode_axis_sha256"))
            )
        ):
            raise PermissionError("held prediction margins or access record differ")
        frozen_tables = row.get("predicted_tables")
        if (
            not isinstance(frozen_tables, dict)
            or set(frozen_tables) != expected_methods
        ):
            raise PermissionError("held prediction method set differs")
        replay = _predict_panel(models, rows, columns)
        for method, expected in replay.items():
            observed = np.asarray(frozen_tables[method], dtype=float)
            if observed.shape != expected.shape or not np.array_equal(
                observed, expected
            ):
                raise PermissionError(
                    f"held {method} prediction does not recompute exactly"
                )
    return payload


def score_held(
    source_path: Path,
    preflight_path: Path,
    prediction_path: Path,
    authorization_path: Path,
    authorization_commit: str,
    source_root: Path,
    attempt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    _require_open()
    for observed, expected, label in (
        (source_path, DEFAULT_SOURCE, "source manifest"),
        (preflight_path, DEFAULT_PREFLIGHT, "metadata preflight"),
        (prediction_path, DEFAULT_PREDICTION, "held predictions"),
        (authorization_path, DEFAULT_SCORE_AUTHORIZATION, "score authorization"),
        (attempt_path, DEFAULT_SCORE_ATTEMPT, "score attempt"),
        (output_path, DEFAULT_SCORE, "score result"),
    ):
        _require_designated(observed, expected, label)
    prediction = _validated_prediction(prediction_path)
    _validated_artifact_authorization(
        authorization_path,
        authorization_commit,
        status="HELD_ADT_ACCESS_AUTHORIZED",
        artifact_path=prediction_path,
        artifact_field="held_predictions",
    )
    if attempt_path.exists() or output_path.exists():
        raise FileExistsError("held scoring is one-shot")
    _write_json(
        attempt_path,
        {
            "schema": "gse239452-score-attempt/1.0",
            "status": "TERMINAL_ATTEMPT_STARTED",
            "created_at_utc": _timestamp(),
            "held_predictions_sha256": _sha256(prediction_path),
            "authorization_sha256": _sha256(authorization_path),
            "public_authorization_commit": authorization_commit,
            "held_adt_numeric_access_begins_after_this_record": True,
        },
    )
    manifest = _sample_manifest(source_path)
    preflight = _preflight_records(preflight_path)
    predicted_by_donor = {row["donor"]: row for row in prediction["samples"]}
    methods = (
        "primary",
        "best_residual",
        "destroyed_link",
        "graph_zero_diagnostic",
    )
    losses = {method: np.empty(len(HELD), dtype=float) for method in methods}
    sample_results: list[dict[str, Any]] = []
    for index, donor in enumerate(HELD):
        truth_record = _reduce_one(
            donor,
            source_root,
            manifest,
            preflight,
            read_adt_numeric=True,
        )
        frozen = predicted_by_donor[donor]
        if (
            truth_record["selected_barcode_axis_sha256"]
            != frozen["selected_barcode_axis_sha256"]
            or truth_record["rna_positive_counts"] != frozen["rna_positive_counts"]
        ):
            raise PermissionError(
                "held cell selection or RNA margins changed after prediction"
            )
        truth = np.asarray(truth_record["tables"], dtype=np.int64)
        rows, columns = _margins(truth)
        if (
            rows.tolist() != frozen["row_margins"]
            or columns.tolist() != frozen["column_margins"]
        ):
            raise PermissionError("held truth margins differ from frozen prediction")
        donor_losses: dict[str, float] = {}
        for method in methods:
            predicted = np.asarray(frozen["predicted_tables"][method], dtype=float)
            value = _donor_loss(truth, predicted)
            losses[method][index] = value
            donor_losses[method] = value
        sample_results.append(
            {
                "donor": donor,
                "severity": truth_record["severity"],
                "losses": donor_losses,
                "selected_barcode_axis_sha256": truth_record[
                    "selected_barcode_axis_sha256"
                ],
                "truth_table_sha256": truth_record["table_sha256"],
            }
        )
    gate = _gate(HELD, losses, required_favorable=8)
    payload = {
        "schema": "gse239452-held-confirmation/1.0",
        "status": "HELD_PASS" if gate["passes"] else "HELD_FAIL",
        "created_at_utc": _timestamp(),
        "runner_sha256": _sha256(Path(__file__)),
        "held_predictions_sha256": _sha256(prediction_path),
        "score_authorization_sha256": _sha256(authorization_path),
        "held_donors": list(HELD),
        "samples": sample_results,
        "held_losses": {
            method: {donor: float(value) for donor, value in zip(HELD, values)}
            for method, values in losses.items()
        },
        "held_gate": gate,
        "promotion_comparators": ["best_residual", "destroyed_link"],
        "graph_vs_zero_is_diagnostic_only": True,
        "access_audit": {
            "held_gex_samples_read": len(HELD),
            "held_adt_samples_read": len(HELD),
            "held_pairings_formed": len(HELD),
            "held_truth_tables_formed": len(HELD),
            "held_truth_tables_serialized": False,
        },
    }
    _write_json(output_path, payload)
    return payload


def _terminal_wrapper(phase: str, attempt_path: Path, operation: Any) -> dict[str, Any]:
    expected_attempts = {
        "development_reduction": DEFAULT_DEVELOPMENT_ATTEMPT,
        "pilot_evaluation": DEFAULT_DEVELOPMENT_ATTEMPT,
        "held_prediction": DEFAULT_PREDICTION_ATTEMPT,
        "held_score": DEFAULT_SCORE_ATTEMPT,
    }
    if phase not in expected_attempts:
        raise ValueError("unknown terminal phase")
    _require_designated(
        attempt_path, expected_attempts[phase], f"{phase} terminal attempt"
    )
    try:
        return operation()
    except Exception as error:
        if attempt_path.exists() and not DEFAULT_TERMINAL_REFUSAL.exists():
            _write_json(
                DEFAULT_TERMINAL_REFUSAL,
                {
                    "schema": "gse239452-terminal-refusal/1.0",
                    "status": "TERMINAL_REFUSAL",
                    "created_at_utc": _timestamp(),
                    "phase": phase,
                    "error_type": type(error).__name__,
                    "reason": "authorized phase failed after its terminal attempt record",
                    "attempt_path": _relative(attempt_path),
                    "attempt_sha256": _sha256(attempt_path),
                    "runner_sha256": _sha256(Path(__file__)),
                    "rerun_permitted": False,
                },
            )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    reduce_parser = subparsers.add_parser("reduce-development")
    reduce_parser.add_argument("--authorization-commit", required=True)
    reduce_parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)

    subparsers.add_parser("fit-pilot")

    predict_parser = subparsers.add_parser("predict-held")
    predict_parser.add_argument("--authorization-commit", required=True)
    predict_parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)

    score_parser = subparsers.add_parser("score-held")
    score_parser.add_argument("--authorization-commit", required=True)
    score_parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)

    args = parser.parse_args()
    if args.phase == "reduce-development":
        payload = _terminal_wrapper(
            "development_reduction",
            DEFAULT_DEVELOPMENT_ATTEMPT,
            lambda: reduce_development(
                DEFAULT_SOURCE,
                DEFAULT_PREFLIGHT,
                DEFAULT_DEVELOPMENT_AUTHORIZATION,
                args.authorization_commit,
                args.source_root,
                DEFAULT_DEVELOPMENT_ATTEMPT,
                DEFAULT_REDUCED,
            ),
        )
    elif args.phase == "fit-pilot":
        payload = _terminal_wrapper(
            "pilot_evaluation",
            DEFAULT_DEVELOPMENT_ATTEMPT,
            lambda: fit_pilot(DEFAULT_REDUCED, DEFAULT_PILOT),
        )
    elif args.phase == "predict-held":
        payload = _terminal_wrapper(
            "held_prediction",
            DEFAULT_PREDICTION_ATTEMPT,
            lambda: predict_held(
                DEFAULT_SOURCE,
                DEFAULT_PREFLIGHT,
                DEFAULT_PILOT,
                DEFAULT_HELD_GEX_AUTHORIZATION,
                args.authorization_commit,
                args.source_root,
                DEFAULT_PREDICTION_ATTEMPT,
                DEFAULT_PREDICTION,
            ),
        )
    else:
        payload = _terminal_wrapper(
            "held_score",
            DEFAULT_SCORE_ATTEMPT,
            lambda: score_held(
                DEFAULT_SOURCE,
                DEFAULT_PREFLIGHT,
                DEFAULT_PREDICTION,
                DEFAULT_SCORE_AUTHORIZATION,
                args.authorization_commit,
                args.source_root,
                DEFAULT_SCORE_ATTEMPT,
                DEFAULT_SCORE,
            ),
        )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
