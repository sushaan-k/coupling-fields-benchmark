"""Freeze the GSE181897 control-PBMC coupling-field source candidate.

This program consumes only the reduced ``cond == C`` donors from physical
pools 0--7.  Pools 8--11 never appear in an input path and are forbidden on
the loaded donor axis.  Model selection uses nested leave-one-pool-out source
validation; no internal or confirmation value is read here.
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
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments.reduce_gse181897_source import (
    CAMPAIGN_IMPLEMENTATION_FILES,
    CELL_BUDGET,
    CONTROL_CONDITION,
    DEFAULT_MODEL_TERMINAL,
    DEFAULT_REDUCTION_TERMINAL,
    DEFAULT_SOURCE_ATTEMPT,
    DEVELOPMENT_DONORS_BY_BATCH,
    MAXIMUM_MARKER_POSITIVES,
    MINIMUM_MARKER_POSITIVES,
    MINIMUM_SOURCE_COORDINATES,
    PANEL,
    SOURCE_BATCHES,
    _campaign_runtime,
    validate_source_campaign_attempt,
)
from mapreg.common_effect_conditional import (
    fit_common_effect_conditional_log_odds,
)
from mapreg.heterogeneity_adaptive_coupling import (
    CouplingEstimationRefusal,
    binary_table_from_helmert_coordinate,
    expected_binary_table_from_log_odds,
    product_hypergraph_laplacian,
    signed_deviance_coordinate,
    signed_pearson_coordinate,
)
from mapreg.penalty_complete_conditional_coupling import (
    fit_hierarchical_conditional_log_odds,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "data/development/gse181897_source/reduced_batches_0_7_control_v1.npz"
)
DEFAULT_MANIFEST = (
    ROOT / "data/development/gse181897_source/"
    "reduction_batches_0_7_control_manifest_v1.json"
)
DEFAULT_OUTPUT = ROOT / "results/development/gse181897_source_candidate_v1.json"

MARKER_COUNT = 17
COORDINATE_COUNT = MARKER_COUNT**2
EXPECTED_DONOR_COUNT = 39
FORBIDDEN_BATCHES = tuple(range(8, 12))
EXPECTED_DONORS = tuple(
    donor for batch in SOURCE_BATCHES for donor in DEVELOPMENT_DONORS_BY_BATCH[batch]
)
EXPECTED_BATCH_AXIS = tuple(
    batch for batch in SOURCE_BATCHES for _ in DEVELOPMENT_DONORS_BY_BATCH[batch]
)

HETEROGENEITY_GRID = (0.1, 1.0, 10.0)
RIDGE_GRID = (0.01, 0.1)
TRANSPORT_GRID = (0.5, 0.75, 1.0, 1.25, 1.5)
NEIGHBOR_GRID = (2, 3)
GRAPH_GRID = (0.01, 0.03, 0.1, 0.3)
RESIDUAL_FAMILIES = ("pearson", "root_deviance")
MAXIMUM_CONDITION_NUMBER = 1e12
TOPOLOGY_NULL_COUNT = 63
TOPOLOGY_NULL_SALT = "GSE181897-CONTROL-TOPOLOGY-NULL-v1"
DESTROYED_LINK_SALT = "GSE181897-CONTROL-DESTROYED-LINK-v1"
SOURCE_BOOTSTRAPS = 20_000
SOURCE_BOOTSTRAP_SALT = "GSE181897-CONTROL-SOURCE-BOOTSTRAP-v1"
REQUIRED_THREAD_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

IMPLEMENTATION_FILES = (*CAMPAIGN_IMPLEMENTATION_FILES,)

EXPECTED_NPZ_MEMBERS = {
    "donor_axis",
    "free_id_axis",
    "batch_axis",
    "condition_axis",
    "rna_gene_axis",
    "rna_feature_id_axis",
    "adt_protein_axis",
    "adt_feature_axis",
    "protein_denominator_axis",
    "selected_barcodes",
    "selected_row_indices",
    "rna_counts",
    "adt_counts",
    "adt_graph_profile",
    "coordinate_axis",
    "source_comparison_mask",
    "tables",
}


@dataclass(frozen=True, order=True)
class BaseConfig:
    heterogeneity_penalty: float
    ridge_penalty: float
    transport_multiplier: float


@dataclass(frozen=True, order=True)
class StructuredConfig:
    graph_neighbors: int
    heterogeneity_penalty: float
    ridge_penalty: float
    graph_penalty: float
    transport_multiplier: float


@dataclass(frozen=True, order=True)
class ResidualConfig:
    family: str
    transport_multiplier: float


@dataclass
class SourceData:
    donors: list[str]
    free_ids: list[str]
    batches: list[int]
    barcodes: list[list[str]]
    selected_rows: np.ndarray
    rna_counts: np.ndarray
    adt_counts: np.ndarray
    adt_graph_profile: np.ndarray
    tables: np.ndarray
    manifest: dict[str, Any]
    manifest_sha256: str
    attempt_path: str
    attempt_sha256: str
    reduction_terminal_path: str
    reduction_terminal_sha256: str


@dataclass
class SelectionResult:
    donors: list[str]
    batches: list[int]
    selected_base: BaseConfig
    selected_primary: StructuredConfig
    selected_common_transport: float
    selected_poisson_transport: float
    selected_residual: ResidualConfig
    selected_residual_transports: dict[str, float]
    selected_destroyed_transport: float
    base_losses: dict[BaseConfig, np.ndarray]
    structured_losses: dict[StructuredConfig, np.ndarray]
    common_losses: dict[float, np.ndarray]
    poisson_losses: dict[float, np.ndarray]
    residual_losses: dict[ResidualConfig, np.ndarray]
    destroyed_losses: dict[float, np.ndarray]
    independence_losses: np.ndarray
    fold_records: dict[int, dict[str, Any]]
    refusals: list[dict[str, Any]]


class SourceGoRefusal(CouplingEstimationRefusal):
    def __init__(self, message: str, details: dict[str, Any]):
        super().__init__(message)
        self.details = details


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


def _axis_sha256(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON token {token}")
            ),
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path.name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _require_runtime() -> None:
    missing = [
        name
        for name, expected in REQUIRED_THREAD_ENVIRONMENT.items()
        if os.environ.get(name) != expected
    ]
    if missing:
        raise PermissionError(
            "required single-thread environment is absent for " + ", ".join(missing)
        )


def _implementation_snapshot() -> dict[str, Any]:
    return {
        **_campaign_runtime(),
        "files_sha256": {
            relative: _sha256(ROOT / relative) for relative in IMPLEMENTATION_FILES
        },
    }


def _expected_panel() -> list[dict[str, Any]]:
    return [
        {
            "rna_gene": item.rna_gene,
            "rna_feature_id": item.rna_feature_id,
            "protein_label": item.protein_label,
            "adt_feature": item.adt_feature,
        }
        for item in PANEL
    ]


def _validate_manifest(
    manifest_path: Path,
    source_path: Path,
    attempt_path: Path,
    model_output_path: Path,
    model_terminal_path: Path,
    reduction_terminal_path: Path,
) -> tuple[dict[str, Any], str, str, str]:
    _, attempt_sha256 = validate_source_campaign_attempt(
        attempt_path,
        ROOT / _read_json(attempt_path)["binding"]["axis_freeze"]["preflight_path"],
        source_path,
        manifest_path,
        model_output_path,
        model_terminal_path,
        reduction_terminal_path,
    )
    manifest_sha256 = _sha256(manifest_path)
    manifest = _read_json(manifest_path)
    exact = {
        "schema": "gse181897-source-reduction/1.0",
        "status": "SOURCE_REDUCTION_COMPLETE",
        "accession": "GSE181897",
        "stage": "source_development",
        "numeric_batches_processed": list(SOURCE_BATCHES),
        "numeric_condition_processed": CONTROL_CONDITION,
        "held_batches_unopened": list(FORBIDDEN_BATCHES),
        "non_control_rows_unopened": True,
        "condition_0_mispool_rows_unopened": 455,
        "donor_count": EXPECTED_DONOR_COUNT,
        "cell_budget_per_donor": CELL_BUDGET,
        "reducer_sha256": _sha256(ROOT / "experiments/reduce_gse181897_source.py"),
    }
    for key, expected in exact.items():
        if manifest.get(key) != expected:
            raise ValueError(f"source reduction manifest has wrong {key}")
    panel = manifest.get("panel")
    if not isinstance(panel, list) or len(panel) != MARKER_COUNT:
        raise ValueError("source reduction manifest has the wrong panel length")
    reduced_panel = [
        {
            "rna_gene": item.get("rna_gene"),
            "rna_feature_id": item.get("rna_feature_id"),
            "protein_label": item.get("protein_label"),
            "adt_feature": item.get("adt_feature"),
        }
        for item in panel
        if isinstance(item, dict)
    ]
    if reduced_panel != _expected_panel():
        raise ValueError("source reduction manifest has the wrong exact panel")
    output = manifest.get("output")
    if not isinstance(output, dict):
        raise ValueError("source reduction manifest lacks output binding")
    output_exact = {
        "path": _display_path(source_path),
        "bytes": source_path.stat().st_size,
        "sha256": _sha256(source_path),
        "members": sorted(EXPECTED_NPZ_MEMBERS),
    }
    for key, expected in output_exact.items():
        if output.get(key) != expected:
            raise ValueError(f"source reduction output has wrong {key}")
    attempt = manifest.get("source_campaign_attempt")
    if (
        not isinstance(attempt, dict)
        or attempt.get("path") != _display_path(attempt_path)
        or attempt.get("sha256") != attempt_sha256
        or attempt.get("schema") != "gse181897-source-campaign-attempt/1.0"
        or attempt.get("status") != "CLAIMED_ONE_SHOT_SOURCE_CAMPAIGN"
    ):
        raise PermissionError("source reduction manifest has the wrong attempt binding")
    access = manifest.get("numeric_access")
    if not isinstance(access, dict) or any(
        access.get(key) != 0
        for key in (
            "held_batch_rows_decoded",
            "non_control_rows_decoded",
            "unselected_authorized_rows_decoded",
        )
    ):
        raise PermissionError("source reduction crossed the frozen numeric boundary")
    if access.get("numeric_rows_decoded") != EXPECTED_DONOR_COUNT * CELL_BUDGET:
        raise ValueError("source reduction decoded the wrong source row count")
    requested_values = access.get("requested_stored_data_entries_decoded")
    scanned_indices = access.get("csr_index_entries_scanned")
    if (
        access.get("matrix_datasets_indexed") != ["/X/indptr", "/X/indices", "/X/data"]
        or access.get("numeric_feature_columns_decoded") != 113
        or not isinstance(requested_values, int)
        or requested_values < 0
        or access.get("stored_numeric_values_decoded") != requested_values
        or not isinstance(scanned_indices, int)
        or scanned_indices < requested_values
        or access.get("unrequested_stored_data_entries_decoded") != 0
        or access.get("out_of_panel_index_positions_scanned")
        != scanned_indices - requested_values
        or access.get("out_of_panel_indices_used_only_for_membership_filtering")
        is not True
        or access.get("out_of_panel_featurewise_statistics_retained") != 0
        or access.get("out_of_panel_feature_signal_entering_model_outputs") != 0
    ):
        raise PermissionError("source reduction CSR access audit is incomplete")
    preflight = manifest.get("axis_preflight")
    if (
        not isinstance(preflight, dict)
        or preflight.get("schema") != "gse181897-axis-preflight/1.1"
        or preflight.get("status") != "AXES_FROZEN_UNIQUE_X_NUMERIC_UNREAD"
        or not isinstance(preflight.get("path"), str)
        or not isinstance(preflight.get("sha256"), str)
    ):
        raise PermissionError("source reduction lacks the frozen axis preflight")
    preflight_path = ROOT / preflight["path"]
    if not preflight_path.is_file() or _sha256(preflight_path) != preflight["sha256"]:
        raise PermissionError("axis preflight bytes differ from the manifest")
    graph = manifest.get("adt_graph_profile")
    if (
        not isinstance(graph, dict)
        or graph.get("denominator_feature_count") != 96
        or graph.get("denominator_rule") != "all 96 var/genome=BD99AbSeq features"
        or graph.get("cognate_feature_axis") != [item.adt_feature for item in PANEL]
    ):
        raise ValueError("source reduction has the wrong ADT graph-profile contract")
    reduction_terminal = _read_json(reduction_terminal_path)
    if (
        reduction_terminal.get("schema") != "gse181897-source-reduction-terminal/1.0"
        or reduction_terminal.get("status") != "SOURCE_REDUCTION_COMPLETE_MODEL_PENDING"
        or reduction_terminal.get("attempt_path") != _display_path(attempt_path)
        or reduction_terminal.get("attempt_sha256") != attempt_sha256
        or reduction_terminal.get("source_reduction_path") != _display_path(source_path)
        or reduction_terminal.get("source_reduction_sha256") != _sha256(source_path)
        or reduction_terminal.get("source_manifest_path")
        != _display_path(manifest_path)
        or reduction_terminal.get("source_manifest_sha256") != manifest_sha256
        or reduction_terminal.get("model_output_path")
        != _display_path(model_output_path)
        or reduction_terminal.get("model_terminal_path")
        != _display_path(model_terminal_path)
        or reduction_terminal.get("model_numeric_access_authorized") is not False
    ):
        raise PermissionError("source reduction terminal does not authorize modeling")
    return (
        manifest,
        manifest_sha256,
        attempt_sha256,
        _sha256(reduction_terminal_path),
    )


def _load_source(
    path: Path,
    manifest_path: Path,
    attempt_path: Path = DEFAULT_SOURCE_ATTEMPT,
    model_output_path: Path = DEFAULT_OUTPUT,
    model_terminal_path: Path = DEFAULT_MODEL_TERMINAL,
    reduction_terminal_path: Path = DEFAULT_REDUCTION_TERMINAL,
) -> SourceData:
    source_sha256 = _sha256(path)
    manifest, manifest_sha256, attempt_sha256, reduction_terminal_sha256 = (
        _validate_manifest(
            manifest_path,
            path,
            attempt_path,
            model_output_path,
            model_terminal_path,
            reduction_terminal_path,
        )
    )
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != EXPECTED_NPZ_MEMBERS:
            raise ValueError("source reduction has the wrong exact NPZ member set")
        donors = [str(value) for value in data["donor_axis"]]
        free_ids = [str(value) for value in data["free_id_axis"]]
        batches = [int(value) for value in data["batch_axis"]]
        conditions = [str(value) for value in data["condition_axis"]]
        genes = [str(value) for value in data["rna_gene_axis"]]
        gene_ids = [str(value) for value in data["rna_feature_id_axis"]]
        proteins = [str(value) for value in data["adt_protein_axis"]]
        adt_features = [str(value) for value in data["adt_feature_axis"]]
        denominator = [str(value) for value in data["protein_denominator_axis"]]
        coordinate_axis = [str(value) for value in data["coordinate_axis"]]
        source_comparison_mask = np.asarray(data["source_comparison_mask"])
        barcodes = [[str(value) for value in row] for row in data["selected_barcodes"]]
        selected_rows = np.asarray(data["selected_row_indices"])
        rna_counts = np.asarray(data["rna_counts"])
        adt_counts = np.asarray(data["adt_counts"])
        adt_graph_profile = np.asarray(data["adt_graph_profile"])
        flat_tables = np.asarray(data["tables"])

    if tuple(donors) != EXPECTED_DONORS or tuple(batches) != EXPECTED_BATCH_AXIS:
        raise PermissionError("source donor or batch axis differs from pools 0--7")
    if any(
        batch in FORBIDDEN_BATCHES or batch not in SOURCE_BATCHES for batch in batches
    ):
        raise PermissionError("a held physical pool entered the source artifact")
    if (
        len(set(donors)) != EXPECTED_DONOR_COUNT
        or len(set(free_ids)) != EXPECTED_DONOR_COUNT
    ):
        raise ValueError("source donor/free-id axes are not unique")
    if conditions != [CONTROL_CONDITION] * EXPECTED_DONOR_COUNT:
        raise PermissionError("a non-control condition entered the source artifact")
    if genes != [item.rna_gene for item in PANEL]:
        raise ValueError("RNA gene axis differs from the frozen panel")
    if gene_ids != [item.rna_feature_id for item in PANEL]:
        raise ValueError("RNA feature-ID axis differs from the frozen panel")
    if proteins != [item.protein_label for item in PANEL]:
        raise ValueError("ADT display axis differs from the frozen panel")
    if adt_features != [item.adt_feature for item in PANEL]:
        raise ValueError("ADT exact-feature axis differs from the frozen panel")
    if (
        len(denominator) != 96
        or len(set(denominator)) != 96
        or manifest["adt_graph_profile"].get("denominator_feature_axis") != denominator
    ):
        raise ValueError("ADT denominator axis differs from the frozen 96 features")
    expected_coordinates = [
        f"{rna.rna_gene}|{adt.adt_feature}" for rna in PANEL for adt in PANEL
    ]
    if coordinate_axis != expected_coordinates:
        raise ValueError(
            "source coordinate axis differs from the frozen row-major axis"
        )
    if (
        source_comparison_mask.shape != (COORDINATE_COUNT,)
        or source_comparison_mask.dtype != np.dtype(np.uint8)
        or not np.isin(source_comparison_mask, (0, 1)).all()
    ):
        raise ValueError("source comparison mask is not a binary uint8 vector")
    count_shape = (EXPECTED_DONOR_COUNT, CELL_BUDGET, MARKER_COUNT)
    if rna_counts.shape != count_shape or adt_counts.shape != count_shape:
        raise ValueError("source count panels have the wrong shape")
    if adt_graph_profile.shape != (EXPECTED_DONOR_COUNT, MARKER_COUNT):
        raise ValueError("source ADT graph profiles have the wrong shape")
    if flat_tables.shape != (EXPECTED_DONOR_COUNT, COORDINATE_COUNT, 4):
        raise ValueError("source joint-table panel has the wrong shape")
    if selected_rows.shape != (EXPECTED_DONOR_COUNT, CELL_BUDGET):
        raise ValueError("selected source row indices have the wrong shape")
    if any(
        len(axis) != CELL_BUDGET or len(set(axis)) != CELL_BUDGET for axis in barcodes
    ):
        raise ValueError("a selected barcode axis is incomplete or duplicated")
    if len(set(selected_rows.ravel().tolist())) != selected_rows.size:
        raise ValueError("a selected source row appears twice")
    for name, values in (
        ("RNA", rna_counts),
        ("ADT", adt_counts),
        ("table", flat_tables),
        ("selected row", selected_rows),
    ):
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"source {name} values must be integers")
        if np.any(values < 0):
            raise ValueError(f"source {name} values must be nonnegative")
    if (
        adt_graph_profile.dtype != np.dtype(np.float64)
        or not np.isfinite(adt_graph_profile).all()
    ):
        raise ValueError("ADT graph profiles must be finite float64")
    tables = flat_tables.reshape(
        EXPECTED_DONOR_COUNT, MARKER_COUNT, MARKER_COUNT, 2, 2
    ).astype(np.int64)
    for donor in range(EXPECTED_DONOR_COUNT):
        recomputed = _tables(rna_counts[donor] > 0, adt_counts[donor] > 0)
        if not np.array_equal(tables[donor], recomputed):
            raise ValueError("stored source table differs from linked count panels")
    provisional_records = {
        donor: {
            "tables": tables[index],
            "subject_support": (
                ((rna_counts[index] > 0).sum(axis=0) >= MINIMUM_MARKER_POSITIVES)
                & ((rna_counts[index] > 0).sum(axis=0) <= MAXIMUM_MARKER_POSITIVES)
            )[:, None]
            & (
                ((adt_counts[index] > 0).sum(axis=0) >= MINIMUM_MARKER_POSITIVES)
                & ((adt_counts[index] > 0).sum(axis=0) <= MAXIMUM_MARKER_POSITIVES)
            )[None, :]
            & _informative(tables[index]),
        }
        for index, donor in enumerate(donors)
    }
    recomputed_mask, _ = _training_mask(provisional_records, donors)
    stored_mask = source_comparison_mask.reshape(MARKER_COUNT, MARKER_COUNT).astype(
        bool
    )
    if not np.array_equal(stored_mask, recomputed_mask):
        raise ValueError(
            "stored source comparison mask fails independent recomputation"
        )
    availability = manifest.get("availability_diagnostics")
    final_mask_record = (
        availability.get("final_source_mask")
        if isinstance(availability, dict)
        else None
    )
    if (
        not isinstance(final_mask_record, dict)
        or final_mask_record.get("mask_sha256")
        != _array_sha256(stored_mask.astype(np.uint8))
        or final_mask_record.get("mask_coordinates")
        != int(np.count_nonzero(stored_mask))
        or final_mask_record.get("mask_at_least_232_coordinates") is not True
        or final_mask_record.get("every_source_donor_at_least_232_coordinates")
        is not True
    ):
        raise ValueError("manifest final source-mask certificate differs")
    if _sha256(path) != source_sha256 or _sha256(manifest_path) != manifest_sha256:
        raise PermissionError("source or manifest changed during loading")
    return SourceData(
        donors=donors,
        free_ids=free_ids,
        batches=batches,
        barcodes=barcodes,
        selected_rows=selected_rows.astype(np.int64, copy=False),
        rna_counts=rna_counts.astype(np.int32, copy=False),
        adt_counts=adt_counts.astype(np.int32, copy=False),
        adt_graph_profile=adt_graph_profile,
        tables=tables,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        attempt_path=_display_path(attempt_path),
        attempt_sha256=attempt_sha256,
        reduction_terminal_path=_display_path(reduction_terminal_path),
        reduction_terminal_sha256=reduction_terminal_sha256,
    )


def _tables(rna_states: np.ndarray, adt_states: np.ndarray) -> np.ndarray:
    rna = np.asarray(rna_states, dtype=np.uint8)
    adt = np.asarray(adt_states, dtype=np.uint8)
    if rna.shape != (CELL_BUDGET, MARKER_COUNT) or adt.shape != rna.shape:
        raise ValueError("binary state panels have the wrong shape")
    output = np.empty((MARKER_COUNT, MARKER_COUNT, 2, 2), dtype=np.int64)
    for first, second in np.ndindex((MARKER_COUNT, MARKER_COUNT)):
        codes = 2 * rna[:, first].astype(int) + adt[:, second].astype(int)
        output[first, second] = np.bincount(codes, minlength=4).reshape(2, 2)
    return output


def _informative(tables: np.ndarray) -> np.ndarray:
    values = np.asarray(tables)
    rows = values.sum(axis=-1)
    columns = values.sum(axis=-2)
    total = values.sum(axis=(-2, -1))
    lower = np.maximum(0, rows[..., 0] + columns[..., 0] - total)
    upper = np.minimum(rows[..., 0], columns[..., 0])
    return upper > lower


def _destroyed_adt(states: np.ndarray, barcodes: list[str], donor: str) -> np.ndarray:
    values = np.asarray(states, dtype=np.uint8)
    if values.shape != (CELL_BUDGET, MARKER_COUNT) or len(barcodes) != CELL_BUDGET:
        raise ValueError("destroyed-link inputs have the wrong shape")
    order = sorted(
        range(CELL_BUDGET),
        key=lambda index: (
            hashlib.sha256(
                f"{DESTROYED_LINK_SALT}|{donor}|{barcodes[index]}".encode()
            ).hexdigest(),
            barcodes[index],
        ),
    )
    order_array = np.asarray(order, dtype=int)
    destroyed = np.empty_like(values)
    destroyed[order_array] = values[np.roll(order_array, 1)]
    if not np.array_equal(destroyed.sum(axis=0), values.sum(axis=0)):
        raise AssertionError("destroyed-link control changed an ADT margin")
    if sorted(map(tuple, destroyed.tolist())) != sorted(map(tuple, values.tolist())):
        raise AssertionError("destroyed-link control changed ADT multivariate rows")
    return destroyed


def _records(data: SourceData) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for index, donor in enumerate(data.donors):
        rna = (data.rna_counts[index] > 0).astype(np.uint8)
        adt = (data.adt_counts[index] > 0).astype(np.uint8)
        rna_positive = rna.sum(axis=0)
        adt_positive = adt.sum(axis=0)
        rna_support = (rna_positive >= MINIMUM_MARKER_POSITIVES) & (
            rna_positive <= MAXIMUM_MARKER_POSITIVES
        )
        adt_support = (adt_positive >= MINIMUM_MARKER_POSITIVES) & (
            adt_positive <= MAXIMUM_MARKER_POSITIVES
        )
        tables = data.tables[index]
        support = rna_support[:, None] & adt_support[None, :] & _informative(tables)
        destroyed = _tables(rna, _destroyed_adt(adt, data.barcodes[index], donor))
        records[donor] = {
            "batch": data.batches[index],
            "free_id": data.free_ids[index],
            "tables": tables,
            "destroyed_tables": destroyed,
            "subject_support": support,
            "rna_profile": np.log(
                (rna_positive.astype(float) + 0.5)
                / (CELL_BUDGET - rna_positive.astype(float) + 0.5)
            ),
            "adt_profile": data.adt_graph_profile[index],
            "rna_positive": rna_positive,
            "adt_positive": adt_positive,
            "rna_marker_support": rna_support,
            "adt_marker_support": adt_support,
            "table_sha256": _array_sha256(tables),
            "destroyed_table_sha256": _array_sha256(destroyed),
            "selected_cell_axis_sha256": _axis_sha256(data.barcodes[index]),
        }
    return records


def _masked_tables(tables: np.ndarray, support: np.ndarray) -> np.ndarray:
    values = np.asarray(tables, dtype=np.int64)
    mask = np.asarray(support, dtype=bool)
    if mask.shape != values.shape[:-2]:
        raise ValueError("table support has the wrong shape")
    return np.where(mask[..., None, None], values, 0)


def _conditional_missing_tables(tables: np.ndarray, support: np.ndarray) -> np.ndarray:
    values = np.asarray(tables, dtype=np.int64)
    mask = np.asarray(support, dtype=bool)
    if mask.shape != values.shape[:-2]:
        raise ValueError("conditional support has the wrong shape")
    totals = values.sum(axis=(-2, -1))
    if np.any(totals != CELL_BUDGET):
        raise ValueError("every source table must contain exactly 128 cells")
    encoded = np.zeros_like(values)
    encoded[..., 0, 0] = totals
    return np.where(mask[..., None, None], values, encoded)


def _training_mask(
    records: dict[str, dict[str, Any]], donors: list[str]
) -> tuple[np.ndarray, dict[str, Any]]:
    if len(donors) < 2 or len(set(donors)) != len(donors):
        raise ValueError("training donor axis must contain distinct donors")
    tables = np.asarray([records[donor]["tables"] for donor in donors])
    support = np.asarray([records[donor]["subject_support"] for donor in donors])
    masked = _masked_tables(tables, support)
    rows = masked.sum(axis=-1)
    columns = masked.sum(axis=-2)
    total = masked.sum(axis=(-2, -1))
    lower = np.maximum(0, rows[..., 0] + columns[..., 0] - total)
    upper = np.minimum(rows[..., 0], columns[..., 0])
    observed = masked[..., 0, 0]
    pooled = masked.sum(axis=0)
    minimum_support = math.ceil(len(donors) / 2)
    support_count = support.sum(axis=0)
    mask = (
        (support_count >= minimum_support)
        & (observed.sum(axis=0) > lower.sum(axis=0))
        & (observed.sum(axis=0) < upper.sum(axis=0))
        & np.all(pooled > 0, axis=(-2, -1))
    )
    coordinate_count = int(np.count_nonzero(mask))
    per_donor = {
        donor: int(np.count_nonzero(mask & records[donor]["subject_support"]))
        for donor in donors
    }
    checks = {
        "at_least_232_coordinates": coordinate_count >= MINIMUM_SOURCE_COORDINATES,
        "every_training_donor_has_at_least_232_coordinates": all(
            value >= MINIMUM_SOURCE_COORDINATES for value in per_donor.values()
        ),
    }
    details = {
        "training_donors": donors,
        "minimum_training_donor_support": minimum_support,
        "coordinate_count": coordinate_count,
        "mask_sha256": _array_sha256(mask.astype(np.uint8)),
        "support_count_sha256": _array_sha256(support_count),
        "training_donor_supported_coordinate_counts": per_donor,
        "strict_pooled_fixed_margin_interior": True,
        "pooled_four_cell_positivity": True,
        "checks": checks,
    }
    if not all(checks.values()):
        raise SourceGoRefusal("training-only comparison mask failed", details)
    return mask, details


def _within_batch_normalize(
    profiles: np.ndarray, batches: list[int]
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(profiles, dtype=float)
    if values.ndim != 2 or values.shape[1] != MARKER_COUNT:
        raise ValueError("marker profiles have the wrong shape")
    if len(batches) != len(values):
        raise ValueError("profile batch axis has the wrong length")
    batch_axis = np.asarray(batches, dtype=int)
    unique_batches = sorted(set(batches))
    centered = np.empty_like(values)
    centers: dict[str, list[float]] = {}
    for batch in unique_batches:
        indices = np.flatnonzero(batch_axis == batch)
        if len(indices) < 2:
            raise SourceGoRefusal(
                "profile normalization has fewer than two donors in a pool",
                {"batch": batch, "donor_count": len(indices)},
            )
        center = values[indices].mean(axis=0)
        centered[indices] = values[indices] - center
        centers[str(batch)] = center.tolist()
    degrees_of_freedom = len(values) - len(unique_batches)
    scale = np.sqrt(np.square(centered).sum(axis=0) / degrees_of_freedom)
    invalid = ~np.isfinite(scale) | (scale <= 0.0)
    if np.any(invalid):
        raise SourceGoRefusal(
            "a source marker profile has zero or nonfinite within-pool variance",
            {"invalid_marker_indices": np.flatnonzero(invalid).tolist()},
        )
    normalized = centered / scale
    return normalized, {
        "method": "physical-pool centering and pooled within-pool SD",
        "batch_axis": unique_batches,
        "batch_centers": centers,
        "pooled_within_batch_scale": scale.tolist(),
        "degrees_of_freedom": degrees_of_freedom,
        "zero_scale_action": "refusal",
        "normalized_profile_sha256": _array_sha256(normalized),
    }


def _marker_hyperedges(profiles: np.ndarray, neighbors: int) -> np.ndarray:
    if neighbors not in NEIGHBOR_GRID:
        raise ValueError("hypergraph neighbor count is outside the frozen grid")
    values = np.asarray(profiles, dtype=float)
    if values.ndim != 2 or values.shape[1] != MARKER_COUNT:
        raise ValueError("normalized marker profiles have the wrong shape")
    marker_profiles = values.T
    memberships: set[tuple[int, ...]] = set()
    for marker in range(MARKER_COUNT):
        candidates = np.asarray(
            [candidate for candidate in range(MARKER_COUNT) if candidate != marker]
        )
        distances = np.linalg.norm(
            marker_profiles[candidates] - marker_profiles[marker], axis=1
        )
        order = candidates[np.lexsort((candidates, distances))]
        memberships.add(tuple(sorted((marker, *map(int, order[:neighbors])))))
    ordered = sorted(memberships)
    incidence = np.zeros((MARKER_COUNT, len(ordered)), dtype=float)
    for column, members in enumerate(ordered):
        incidence[np.asarray(members), column] = 1.0
    if not np.all(incidence.sum(axis=0) == neighbors + 1):
        raise AssertionError("a marker-centered hyperedge has the wrong size")
    return incidence


def _training_design(
    records: dict[str, dict[str, Any]], donors: list[str], neighbors: int
) -> dict[str, Any]:
    mask, mask_record = _training_mask(records, donors)
    batches = [int(records[donor]["batch"]) for donor in donors]
    rna_raw = np.asarray([records[donor]["rna_profile"] for donor in donors])
    adt_raw = np.asarray([records[donor]["adt_profile"] for donor in donors])
    rna_normalized, rna_normalization = _within_batch_normalize(rna_raw, batches)
    adt_normalized, adt_normalization = _within_batch_normalize(adt_raw, batches)
    rna_incidence = _marker_hyperedges(rna_normalized, neighbors)
    adt_incidence = _marker_hyperedges(adt_normalized, neighbors)
    laplacian = product_hypergraph_laplacian(rna_incidence, adt_incidence)
    return {
        "donors": donors,
        "batches": batches,
        "mask": mask,
        "mask_record": mask_record,
        "rna_raw": rna_raw,
        "adt_raw": adt_raw,
        "rna_normalized": rna_normalized,
        "adt_normalized": adt_normalized,
        "rna_normalization": rna_normalization,
        "adt_normalization": adt_normalization,
        "rna_incidence": rna_incidence,
        "adt_incidence": adt_incidence,
        "rna_incidence_sha256": _array_sha256(rna_incidence),
        "adt_incidence_sha256": _array_sha256(adt_incidence),
        "product_laplacian_sha256": _array_sha256(laplacian),
    }


def _fold_arrays(
    records: dict[str, dict[str, Any]], donors: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([records[donor]["tables"] for donor in donors]),
        np.asarray([records[donor]["destroyed_tables"] for donor in donors]),
        np.asarray([records[donor]["subject_support"] for donor in donors]),
    )


def _fit_structured(
    tables: np.ndarray,
    support: np.ndarray,
    rna_incidence: np.ndarray,
    adt_incidence: np.ndarray,
    config: StructuredConfig,
) -> dict[str, Any]:
    fit = fit_hierarchical_conditional_log_odds(
        _conditional_missing_tables(tables, support),
        rna_incidence,
        adt_incidence,
        heterogeneity_penalty=config.heterogeneity_penalty,
        ridge_penalty=config.ridge_penalty,
        graph_penalty=config.graph_penalty,
        minimum_informative_donors=0,
        maximum_condition_number=MAXIMUM_CONDITION_NUMBER,
    )
    certificate = {
        name: getattr(fit, name)
        for name in (
            "objective",
            "gradient_norm",
            "scaled_gradient_norm",
            "schur_condition_number",
            "theta_curvature_condition_number",
            "minimum_theta_curvature",
            "maximum_theta_curvature",
            "minimum_schur_eigenvalue",
            "maximum_schur_eigenvalue",
            "iterations",
            "converged",
            "optimizer",
            "heterogeneity_penalty_scale",
            "population_penalty_scale",
            "effective_heterogeneity_penalty",
            "effective_ridge_penalty",
            "effective_graph_penalty",
        )
    }
    certificate.update(
        {
            "minimum_support_count": int(fit.support_count.min()),
            "maximum_support_count": int(fit.support_count.max()),
            "support_count_sha256": _array_sha256(fit.support_count),
            "rna_incidence_sha256": _array_sha256(rna_incidence),
            "adt_incidence_sha256": _array_sha256(adt_incidence),
        }
    )
    return {
        "population_log_odds": np.asarray(fit.population_log_odds, dtype=float),
        "fit_certificate": certificate,
    }


def _fit_common(
    tables: np.ndarray, support: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    selected = _conditional_missing_tables(tables, support)[:, mask]
    fit = fit_common_effect_conditional_log_odds(
        selected,
        minimum_informative_donors=math.ceil(len(tables) / 2),
        tolerance=1e-10,
    )
    population = np.zeros((MARKER_COUNT, MARKER_COUNT), dtype=float)
    population[mask] = fit.log_odds
    return {
        "population_log_odds": population,
        "fit_certificate": {
            "objective": fit.objective,
            "gradient_norm": fit.gradient_norm,
            "scaled_gradient_norm": fit.scaled_gradient_norm,
            "minimum_data_precision": float(np.min(fit.data_precision)),
            "maximum_data_precision": float(np.max(fit.data_precision)),
            "minimum_support_count": int(np.min(fit.support_count)),
            "maximum_support_count": int(np.max(fit.support_count)),
            "maximum_root_iterations": int(np.max(fit.root_iterations)),
            "converged": fit.converged,
        },
    }


def _pooled_supported_tables(
    tables: np.ndarray, support: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    values = np.asarray(tables, dtype=np.int64)
    observed = np.asarray(support, dtype=bool)
    comparison = np.asarray(mask, dtype=bool)
    if observed.shape != values.shape[:-2] or comparison.shape != values.shape[1:-2]:
        raise ValueError("pooled Poisson inputs have incompatible shapes")
    return _masked_tables(values, observed).sum(axis=0)[comparison]


def _fit_pooled_poisson(
    tables: np.ndarray, support: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    pooled = _pooled_supported_tables(tables, support, mask)
    if np.any(pooled <= 0):
        raise CouplingEstimationRefusal(
            "pooled saturated Poisson interaction has a zero cell"
        )
    selected = (
        np.log(pooled[:, 0, 0])
        + np.log(pooled[:, 1, 1])
        - np.log(pooled[:, 0, 1])
        - np.log(pooled[:, 1, 0])
    )
    if not np.isfinite(selected).all():
        raise CouplingEstimationRefusal("pooled Poisson interaction is nonfinite")
    maximum_cell_error = 0.0
    maximum_row_error = 0.0
    maximum_column_error = 0.0
    for table, log_odds in zip(pooled, selected):
        rows = table.sum(axis=1)
        columns = table.sum(axis=0)
        reconstructed = binary_table_from_helmert_coordinate(
            0.5 * float(log_odds), rows, columns
        )
        scale = max(1.0, float(table.sum()))
        maximum_cell_error = max(
            maximum_cell_error,
            float(np.max(np.abs(reconstructed - table))) / scale,
        )
        maximum_row_error = max(
            maximum_row_error,
            float(np.max(np.abs(reconstructed.sum(axis=1) - rows))) / scale,
        )
        maximum_column_error = max(
            maximum_column_error,
            float(np.max(np.abs(reconstructed.sum(axis=0) - columns))) / scale,
        )
    maximum_error = max(maximum_cell_error, maximum_row_error, maximum_column_error)
    if maximum_error > 1e-8:
        raise CouplingEstimationRefusal(
            "pooled Poisson saturated-table reconstruction failed"
        )
    population = np.zeros((MARKER_COUNT, MARKER_COUNT), dtype=float)
    population[mask] = selected
    return {
        "population_log_odds": population,
        "pooled_tables": pooled,
        "fit_certificate": {
            "coordinate_count": len(pooled),
            "pooled_tables_sha256": _array_sha256(pooled),
            "same_supported_donor_coordinate_tables_as_primary": True,
            "maximum_normalized_cell_error": maximum_cell_error,
            "maximum_normalized_row_margin_error": maximum_row_error,
            "maximum_normalized_column_margin_error": maximum_column_error,
            "maximum_normalized_reconstruction_error": maximum_error,
            "threshold": 1e-8,
            "passes": maximum_error <= 1e-8,
        },
    }


def _fit_residual(
    tables: np.ndarray, support: np.ndarray, mask: np.ndarray, family: str
) -> dict[str, Any]:
    if family not in RESIDUAL_FAMILIES:
        raise ValueError("unknown residual family")
    values = np.asarray(tables)
    observed = np.asarray(support, dtype=bool)
    selected_mask = np.asarray(mask, dtype=bool)
    coordinate = np.full((len(values), COORDINATE_COUNT), np.nan)
    support_flat = observed.reshape(len(values), COORDINATE_COUNT)
    tables_flat = values.reshape(len(values), COORDINATE_COUNT, 2, 2)
    selected_flat = selected_mask.ravel()
    statistic = (
        signed_pearson_coordinate if family == "pearson" else signed_deviance_coordinate
    )
    for donor, entity in np.argwhere(support_flat & selected_flat[None, :]):
        coordinate[donor, entity] = statistic(tables_flat[donor, entity]) / math.sqrt(
            CELL_BUDGET
        )
    selected_support = support_flat[:, selected_flat]
    if np.any(selected_support.sum(axis=0) < math.ceil(len(values) / 2)):
        raise CouplingEstimationRefusal(
            "classical residual lacks the frozen training support"
        )
    pooled_selected = np.nanmean(coordinate[:, selected_flat], axis=0)
    if not np.isfinite(pooled_selected).all():
        raise CouplingEstimationRefusal("classical residual is nonfinite")
    pooled = np.zeros(COORDINATE_COUNT, dtype=float)
    pooled[selected_flat] = pooled_selected
    return {
        "family": family,
        "pooled_coordinate": pooled.reshape(MARKER_COUNT, MARKER_COUNT),
        "fit_certificate": {
            "coordinate_count": int(np.count_nonzero(selected_mask)),
            "minimum_support_count": int(selected_support.sum(axis=0).min()),
            "maximum_support_count": int(selected_support.sum(axis=0).max()),
            "pooled_coordinate_sha256": _array_sha256(
                pooled.reshape(MARKER_COUNT, MARKER_COUNT)
            ),
            "normalization": f"signed one-df {family} / sqrt(128)",
        },
    }


def _fractional_pearson(table: np.ndarray) -> float:
    values = np.asarray(table, dtype=float)
    total = float(values.sum())
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
    coordinate: float, rows: np.ndarray, columns: np.ndarray, family: str
) -> np.ndarray:
    row = np.asarray(rows, dtype=float)
    column = np.asarray(columns, dtype=float)
    total = float(row.sum())
    lower = max(0.0, float(row[0] + column[0] - total))
    upper = min(float(row[0]), float(column[0]))
    if upper <= lower:
        return np.asarray(
            [
                [lower, row[0] - lower],
                [column[0] - lower, row[1] - column[0] + lower],
            ]
        )
    epsilon = min(1e-10, 0.25 * (upper - lower))
    left, right = lower + epsilon, upper - epsilon

    def table_at(value: float) -> np.ndarray:
        return np.asarray(
            [
                [value, row[0] - value],
                [column[0] - value, row[1] - column[0] + value],
            ]
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


def _evaluation_mask(record: dict[str, Any], mask: np.ndarray) -> np.ndarray:
    evaluation = np.asarray(mask, dtype=bool) & np.asarray(
        record["subject_support"], dtype=bool
    )
    count = int(np.count_nonzero(evaluation))
    if count < MINIMUM_SOURCE_COORDINATES:
        raise SourceGoRefusal(
            "a validation donor lacks the frozen coordinate support",
            {
                "batch": record["batch"],
                "supported_coordinates": count,
                "minimum": MINIMUM_SOURCE_COORDINATES,
            },
        )
    return evaluation


def _deviance_loss(
    observed: np.ndarray, predicted: np.ndarray, evaluation: np.ndarray
) -> float:
    truth = np.asarray(observed, dtype=float)[evaluation]
    estimate = np.asarray(predicted, dtype=float)[evaluation]
    if not np.allclose(truth.sum(axis=-1), estimate.sum(axis=-1)) or not np.allclose(
        truth.sum(axis=-2), estimate.sum(axis=-2)
    ):
        raise FloatingPointError("prediction changed a recipient margin")
    positive = truth > 0
    if np.any(estimate[positive] <= 0.0) or not np.isfinite(estimate).all():
        raise FloatingPointError("prediction assigns invalid mass")
    terms = np.zeros_like(truth)
    terms[positive] = truth[positive] * np.log(truth[positive] / estimate[positive])
    return float((2.0 * terms.sum(axis=(-2, -1)) / CELL_BUDGET).mean())


def _predict_log_odds(
    record: dict[str, Any], log_odds: np.ndarray, alpha: float, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    evaluation = _evaluation_mask(record, mask)
    observed = np.asarray(record["tables"])
    rows = observed.sum(axis=-1)
    columns = observed.sum(axis=-2)
    predicted = np.zeros_like(observed, dtype=float)
    for index in np.argwhere(evaluation):
        entity = tuple(map(int, index))
        predicted[entity] = expected_binary_table_from_log_odds(
            float(alpha) * float(log_odds[entity]), rows[entity], columns[entity]
        )
    return predicted, evaluation


def _population_loss(
    record: dict[str, Any], mask: np.ndarray, log_odds: np.ndarray, alpha: float
) -> float:
    predicted, evaluation = _predict_log_odds(record, log_odds, alpha, mask)
    return _deviance_loss(record["tables"], predicted, evaluation)


def _residual_loss(
    record: dict[str, Any], mask: np.ndarray, pooled: np.ndarray, config: ResidualConfig
) -> float:
    evaluation = _evaluation_mask(record, mask)
    observed = np.asarray(record["tables"])
    rows = observed.sum(axis=-1)
    columns = observed.sum(axis=-2)
    predicted = np.zeros_like(observed, dtype=float)
    for index in np.argwhere(evaluation):
        entity = tuple(map(int, index))
        coordinate = (
            config.transport_multiplier * float(pooled[entity]) * math.sqrt(CELL_BUDGET)
        )
        predicted[entity] = _residual_table(
            coordinate, rows[entity], columns[entity], config.family
        )
    return _deviance_loss(observed, predicted, evaluation)


def _independence_loss(record: dict[str, Any], mask: np.ndarray) -> float:
    zeros = np.zeros((MARKER_COUNT, MARKER_COUNT), dtype=float)
    return _population_loss(record, mask, zeros, 1.0)


def _batch_splits(
    records: dict[str, dict[str, Any]], donors: list[str]
) -> list[tuple[int, list[str], list[str]]]:
    batch_axis = sorted({int(records[donor]["batch"]) for donor in donors})
    if len(batch_axis) < 2:
        raise ValueError("leave-one-pool-out selection needs at least two pools")
    splits = []
    for held_batch in batch_axis:
        training = [
            donor for donor in donors if int(records[donor]["batch"]) != held_batch
        ]
        validation = [
            donor for donor in donors if int(records[donor]["batch"]) == held_batch
        ]
        if not training or not validation:
            raise ValueError("a leave-one-pool-out split is empty")
        splits.append((held_batch, training, validation))
    return splits


def _equal_batch_summary(
    values: np.ndarray, donors: list[str], batches: list[int]
) -> dict[str, Any]:
    losses = np.asarray(values, dtype=float)
    if losses.shape != (len(donors),) or len(batches) != len(donors):
        raise ValueError("loss and sample axes differ")
    complete = bool(np.isfinite(losses).all())
    batch_axis = sorted(set(batches))
    batch_array = np.asarray(batches, dtype=int)
    means = {
        str(batch): (float(np.mean(losses[batch_array == batch])) if complete else None)
        for batch in batch_axis
    }
    return {
        "complete": complete,
        "equal_batch_mean_loss": (
            float(np.mean(list(means.values()))) if complete else None
        ),
        "batch_mean_losses": means,
    }


def _curve_entry(
    config: Any,
    values: np.ndarray,
    donors: list[str],
    batches: list[int],
) -> dict[str, Any]:
    return {
        "configuration": (
            asdict(config) if hasattr(config, "__dataclass_fields__") else config
        ),
        "donor_axis": donors,
        "batch_axis": batches,
        "donor_losses": [
            float(value) if np.isfinite(value) else None for value in values
        ],
        **_equal_batch_summary(values, donors, batches),
    }


def _complete_configs(
    losses: dict[Any, np.ndarray], donors: list[str], batches: list[int]
) -> list[Any]:
    return [
        config
        for config, values in losses.items()
        if _equal_batch_summary(values, donors, batches)["complete"]
    ]


def _select_base(
    losses: dict[BaseConfig, np.ndarray], donors: list[str], batches: list[int]
) -> BaseConfig:
    complete = _complete_configs(losses, donors, batches)
    if not complete:
        raise SourceGoRefusal("no graph-zero configuration completed source CV", {})
    return min(
        complete,
        key=lambda config: (
            _equal_batch_summary(losses[config], donors, batches)[
                "equal_batch_mean_loss"
            ],
            config.heterogeneity_penalty,
            config.ridge_penalty,
            config.transport_multiplier,
        ),
    )


def _select_structured(
    losses: dict[StructuredConfig, np.ndarray],
    donors: list[str],
    batches: list[int],
) -> StructuredConfig:
    complete = _complete_configs(losses, donors, batches)
    if not complete:
        raise SourceGoRefusal(
            "no nonzero hypergraph configuration completed source CV", {}
        )
    return min(
        complete,
        key=lambda config: (
            _equal_batch_summary(losses[config], donors, batches)[
                "equal_batch_mean_loss"
            ],
            config.graph_penalty,
            config.graph_neighbors,
        ),
    )


def _select_transport(
    losses: dict[float, np.ndarray], donors: list[str], batches: list[int]
) -> float:
    complete = _complete_configs(losses, donors, batches)
    if not complete:
        raise SourceGoRefusal("no transport multiplier completed source CV", {})
    return min(
        complete,
        key=lambda alpha: (
            _equal_batch_summary(losses[alpha], donors, batches)[
                "equal_batch_mean_loss"
            ],
            alpha,
        ),
    )


def _select_residual(
    losses: dict[ResidualConfig, np.ndarray],
    selected_transports: dict[str, float],
    donors: list[str],
    batches: list[int],
) -> ResidualConfig:
    candidates = [
        ResidualConfig(family, selected_transports[family])
        for family in RESIDUAL_FAMILIES
    ]
    if any(config not in losses for config in candidates):
        raise ValueError("a source-selected residual configuration is missing")
    complete = [
        config
        for config in candidates
        if _equal_batch_summary(losses[config], donors, batches)["complete"]
    ]
    if not complete:
        raise SourceGoRefusal("no classical residual family completed source CV", {})
    return min(
        complete,
        key=lambda config: (
            _equal_batch_summary(losses[config], donors, batches)[
                "equal_batch_mean_loss"
            ],
            RESIDUAL_FAMILIES.index(config.family),
        ),
    )


def _validation_support_record(
    records: dict[str, dict[str, Any]], validation: list[str], mask: np.ndarray
) -> dict[str, int]:
    counts = {
        donor: int(np.count_nonzero(mask & records[donor]["subject_support"]))
        for donor in validation
    }
    if any(value < MINIMUM_SOURCE_COORDINATES for value in counts.values()):
        raise SourceGoRefusal(
            "a held source-pool donor lacks comparison support",
            {
                "validation_supported_coordinate_counts": counts,
                "minimum": MINIMUM_SOURCE_COORDINATES,
            },
        )
    return counts


def _select_models(
    records: dict[str, dict[str, Any]], donors: list[str]
) -> SelectionResult:
    """Select every model by leave-one-pool-out CV within ``donors``."""

    batches = [int(records[donor]["batch"]) for donor in donors]
    donor_index = {donor: index for index, donor in enumerate(donors)}
    splits = _batch_splits(records, donors)
    base_losses = {
        BaseConfig(eta, ridge, alpha): np.full(len(donors), np.nan)
        for eta, ridge, alpha in product(HETEROGENEITY_GRID, RIDGE_GRID, TRANSPORT_GRID)
    }
    common_losses = {alpha: np.full(len(donors), np.nan) for alpha in TRANSPORT_GRID}
    poisson_losses = {alpha: np.full(len(donors), np.nan) for alpha in TRANSPORT_GRID}
    residual_losses = {
        ResidualConfig(family, alpha): np.full(len(donors), np.nan)
        for family, alpha in product(RESIDUAL_FAMILIES, TRANSPORT_GRID)
    }
    independence_losses = np.full(len(donors), np.nan)
    fold_records: dict[int, dict[str, Any]] = {}
    refusals: list[dict[str, Any]] = []

    for held_batch, training, validation in splits:
        comparison_mask, mask_record = _training_mask(records, training)
        validation_support = _validation_support_record(
            records, validation, comparison_mask
        )
        tables, _, support = _fold_arrays(records, training)
        identity = np.eye(MARKER_COUNT, dtype=float)
        for eta, ridge in product(HETEROGENEITY_GRID, RIDGE_GRID):
            structural = StructuredConfig(2, eta, ridge, 0.0, 1.0)
            try:
                fit = _fit_structured(tables, support, identity, identity, structural)
            except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
                refusals.append(
                    {
                        "stage": "stage_a_graph_zero",
                        "held_batch": held_batch,
                        "configuration": asdict(structural),
                        "reason_code": type(error).__name__,
                        "reason": str(error),
                    }
                )
                continue
            for alpha in TRANSPORT_GRID:
                config = BaseConfig(eta, ridge, alpha)
                for donor in validation:
                    base_losses[config][donor_index[donor]] = _population_loss(
                        records[donor],
                        comparison_mask,
                        fit["population_log_odds"],
                        alpha,
                    )
        try:
            common = _fit_common(tables, support, comparison_mask)
            poisson = _fit_pooled_poisson(tables, support, comparison_mask)
            residuals = {
                family: _fit_residual(tables, support, comparison_mask, family)
                for family in RESIDUAL_FAMILIES
            }
        except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
            raise SourceGoRefusal(
                "a mandatory classical estimator failed source selection",
                {
                    "held_batch": held_batch,
                    "reason_code": type(error).__name__,
                    "reason": str(error),
                },
            ) from error
        for donor in validation:
            index = donor_index[donor]
            independence_losses[index] = _independence_loss(
                records[donor], comparison_mask
            )
            for alpha in TRANSPORT_GRID:
                common_losses[alpha][index] = _population_loss(
                    records[donor],
                    comparison_mask,
                    common["population_log_odds"],
                    alpha,
                )
                poisson_losses[alpha][index] = _population_loss(
                    records[donor],
                    comparison_mask,
                    poisson["population_log_odds"],
                    alpha,
                )
                for family in RESIDUAL_FAMILIES:
                    config = ResidualConfig(family, alpha)
                    residual_losses[config][index] = _residual_loss(
                        records[donor],
                        comparison_mask,
                        residuals[family]["pooled_coordinate"],
                        config,
                    )
        fold_records[held_batch] = {
            "training_donors": training,
            "validation_donors": validation,
            "comparison_mask": {
                **mask_record,
                "mask": comparison_mask.astype(np.uint8).tolist(),
            },
            "stage_a_profile_graph_constructed": False,
            "validation_supported_coordinate_counts": validation_support,
        }

    selected_base = _select_base(base_losses, donors, batches)
    structured_losses = {
        StructuredConfig(
            neighbors,
            selected_base.heterogeneity_penalty,
            selected_base.ridge_penalty,
            graph_penalty,
            selected_base.transport_multiplier,
        ): np.full(len(donors), np.nan)
        for neighbors, graph_penalty in product(NEIGHBOR_GRID, GRAPH_GRID)
    }
    for held_batch, training, validation in splits:
        tables, _, support = _fold_arrays(records, training)
        reference_mask = np.asarray(
            fold_records[held_batch]["comparison_mask"]["mask"], dtype=bool
        )
        for neighbors in NEIGHBOR_GRID:
            design = _training_design(records, training, neighbors)
            if not np.array_equal(reference_mask, design["mask"]):
                raise AssertionError("training mask changed with hypergraph k")
            geometry = fold_records[held_batch].setdefault("stage_b_geometry", {})
            geometry[str(neighbors)] = {
                "rna_normalization": design["rna_normalization"],
                "adt_normalization": design["adt_normalization"],
                "rna_incidence_sha256": design["rna_incidence_sha256"],
                "adt_incidence_sha256": design["adt_incidence_sha256"],
                "product_laplacian_sha256": design["product_laplacian_sha256"],
            }
            for graph_penalty in GRAPH_GRID:
                config = StructuredConfig(
                    neighbors,
                    selected_base.heterogeneity_penalty,
                    selected_base.ridge_penalty,
                    graph_penalty,
                    selected_base.transport_multiplier,
                )
                try:
                    fit = _fit_structured(
                        tables,
                        support,
                        design["rna_incidence"],
                        design["adt_incidence"],
                        config,
                    )
                    for donor in validation:
                        structured_losses[config][donor_index[donor]] = (
                            _population_loss(
                                records[donor],
                                design["mask"],
                                fit["population_log_odds"],
                                config.transport_multiplier,
                            )
                        )
                except (
                    ValueError,
                    FloatingPointError,
                    CouplingEstimationRefusal,
                ) as error:
                    refusals.append(
                        {
                            "stage": "stage_b_hypergraph",
                            "held_batch": held_batch,
                            "configuration": asdict(config),
                            "reason_code": type(error).__name__,
                            "reason": str(error),
                        }
                    )
    selected_primary = _select_structured(structured_losses, donors, batches)
    selected_common = _select_transport(common_losses, donors, batches)
    selected_poisson = _select_transport(poisson_losses, donors, batches)
    selected_residual_transports = {
        family: _select_transport(
            {
                config.transport_multiplier: values
                for config, values in residual_losses.items()
                if config.family == family
            },
            donors,
            batches,
        )
        for family in RESIDUAL_FAMILIES
    }
    selected_residual = _select_residual(
        residual_losses, selected_residual_transports, donors, batches
    )

    destroyed_losses = {alpha: np.full(len(donors), np.nan) for alpha in TRANSPORT_GRID}
    for held_batch, training, validation in splits:
        _, destroyed, support = _fold_arrays(records, training)
        design = _training_design(records, training, selected_primary.graph_neighbors)
        structural = StructuredConfig(
            selected_primary.graph_neighbors,
            selected_primary.heterogeneity_penalty,
            selected_primary.ridge_penalty,
            selected_primary.graph_penalty,
            1.0,
        )
        fit = _fit_structured(
            destroyed,
            support,
            design["rna_incidence"],
            design["adt_incidence"],
            structural,
        )
        for alpha in TRANSPORT_GRID:
            for donor in validation:
                destroyed_losses[alpha][donor_index[donor]] = _population_loss(
                    records[donor],
                    design["mask"],
                    fit["population_log_odds"],
                    alpha,
                )
    selected_destroyed = _select_transport(destroyed_losses, donors, batches)
    return SelectionResult(
        donors=donors,
        batches=batches,
        selected_base=selected_base,
        selected_primary=selected_primary,
        selected_common_transport=selected_common,
        selected_poisson_transport=selected_poisson,
        selected_residual=selected_residual,
        selected_residual_transports=selected_residual_transports,
        selected_destroyed_transport=selected_destroyed,
        base_losses=base_losses,
        structured_losses=structured_losses,
        common_losses=common_losses,
        poisson_losses=poisson_losses,
        residual_losses=residual_losses,
        destroyed_losses=destroyed_losses,
        independence_losses=independence_losses,
        fold_records=fold_records,
        refusals=refusals,
    )


def _selection_record(selection: SelectionResult) -> dict[str, Any]:
    donors = selection.donors
    batches = selection.batches
    return {
        "donor_axis": donors,
        "batch_axis": batches,
        "folds": selection.fold_records,
        "stage_a_graph_zero": {
            "selection_rule": ("minimum equal-pool mean loss; then eta, ridge, alpha"),
            "selected_configuration": asdict(selection.selected_base),
            "loss_curve": [
                _curve_entry(config, values, donors, batches)
                for config, values in sorted(selection.base_losses.items())
            ],
        },
        "stage_b_nonzero_hypergraph": {
            "fixed_from_stage_a": asdict(selection.selected_base),
            "selection_rule": (
                "minimum equal-pool mean loss; then graph penalty, then k"
            ),
            "selected_configuration": asdict(selection.selected_primary),
            "loss_curve": [
                _curve_entry(config, values, donors, batches)
                for config, values in sorted(selection.structured_losses.items())
            ],
        },
        "comparators": {
            "selected_common_transport": selection.selected_common_transport,
            "selected_poisson_transport": selection.selected_poisson_transport,
            "selected_residual": asdict(selection.selected_residual),
            "selected_residual_transports": selection.selected_residual_transports,
            "selected_destroyed_transport": selection.selected_destroyed_transport,
            "common_curve": [
                _curve_entry({"transport_multiplier": alpha}, values, donors, batches)
                for alpha, values in sorted(selection.common_losses.items())
            ],
            "pooled_poisson_curve": [
                _curve_entry({"transport_multiplier": alpha}, values, donors, batches)
                for alpha, values in sorted(selection.poisson_losses.items())
            ],
            "residual_curve": [
                _curve_entry(config, values, donors, batches)
                for config, values in sorted(selection.residual_losses.items())
            ],
            "destroyed_link_curve": [
                _curve_entry({"transport_multiplier": alpha}, values, donors, batches)
                for alpha, values in sorted(selection.destroyed_losses.items())
            ],
            "independence": _curve_entry(
                {"method": "recipient-margin independence"},
                selection.independence_losses,
                donors,
                batches,
            ),
        },
        "refusals": selection.refusals,
    }


def _outer_fit(
    records: dict[str, dict[str, Any]],
    training: list[str],
    selection: SelectionResult,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    primary_config = selection.selected_primary
    design = _training_design(records, training, primary_config.graph_neighbors)
    tables, destroyed, support = _fold_arrays(records, training)
    primary = _fit_structured(
        tables,
        support,
        design["rna_incidence"],
        design["adt_incidence"],
        primary_config,
    )
    zero_config = StructuredConfig(
        graph_neighbors=primary_config.graph_neighbors,
        heterogeneity_penalty=primary_config.heterogeneity_penalty,
        ridge_penalty=primary_config.ridge_penalty,
        graph_penalty=0.0,
        transport_multiplier=primary_config.transport_multiplier,
    )
    identity = np.eye(MARKER_COUNT, dtype=float)
    graph_zero = _fit_structured(tables, support, identity, identity, zero_config)
    common = _fit_common(tables, support, design["mask"])
    poisson = _fit_pooled_poisson(tables, support, design["mask"])
    residuals = {
        family: _fit_residual(tables, support, design["mask"], family)
        for family in RESIDUAL_FAMILIES
    }
    destroyed_config = StructuredConfig(
        graph_neighbors=primary_config.graph_neighbors,
        heterogeneity_penalty=primary_config.heterogeneity_penalty,
        ridge_penalty=primary_config.ridge_penalty,
        graph_penalty=primary_config.graph_penalty,
        transport_multiplier=selection.selected_destroyed_transport,
    )
    destroyed_fit = _fit_structured(
        destroyed,
        support,
        design["rna_incidence"],
        design["adt_incidence"],
        destroyed_config,
    )
    models = {
        "primary": {
            **primary,
            "configuration": asdict(primary_config),
        },
        "matched_graph_zero": {
            **graph_zero,
            "configuration": asdict(zero_config),
        },
        "common_effect_cmle": {
            **common,
            "configuration": {
                "transport_multiplier": selection.selected_common_transport
            },
        },
        "pooled_saturated_poisson": {
            **poisson,
            "configuration": {
                "transport_multiplier": selection.selected_poisson_transport
            },
        },
        "destroyed_link": {
            **destroyed_fit,
            "configuration": asdict(destroyed_config),
        },
    }
    for family, residual in residuals.items():
        models[f"{family}_residual"] = {
            **residual,
            "configuration": {
                "family": family,
                "transport_multiplier": selection.selected_residual_transports[family],
            },
        }
    models["primary_classical_residual"] = models[
        f"{selection.selected_residual.family}_residual"
    ]
    return models, design


def _score_outer_donor(
    record: dict[str, Any],
    mask: np.ndarray,
    models: dict[str, dict[str, Any]],
) -> dict[str, float]:
    primary = models["primary"]
    zero = models["matched_graph_zero"]
    common = models["common_effect_cmle"]
    poisson = models["pooled_saturated_poisson"]
    destroyed = models["destroyed_link"]
    losses = {
        "primary": _population_loss(
            record,
            mask,
            primary["population_log_odds"],
            primary["configuration"]["transport_multiplier"],
        ),
        "matched_graph_zero": _population_loss(
            record,
            mask,
            zero["population_log_odds"],
            zero["configuration"]["transport_multiplier"],
        ),
        "common_effect_cmle": _population_loss(
            record,
            mask,
            common["population_log_odds"],
            common["configuration"]["transport_multiplier"],
        ),
        "pooled_saturated_poisson": _population_loss(
            record,
            mask,
            poisson["population_log_odds"],
            poisson["configuration"]["transport_multiplier"],
        ),
        "destroyed_link": _population_loss(
            record,
            mask,
            destroyed["population_log_odds"],
            destroyed["configuration"]["transport_multiplier"],
        ),
        "independence": _independence_loss(record, mask),
    }
    for family in RESIDUAL_FAMILIES:
        model = models[f"{family}_residual"]
        config = ResidualConfig(
            family=family,
            transport_multiplier=model["configuration"]["transport_multiplier"],
        )
        losses[f"{family}_residual"] = _residual_loss(
            record, mask, model["pooled_coordinate"], config
        )
    selected = models["primary_classical_residual"]
    selected_config = ResidualConfig(**selected["configuration"])
    losses["primary_classical_residual"] = _residual_loss(
        record, mask, selected["pooled_coordinate"], selected_config
    )
    return losses


def _nested_source_predictions(
    data: SourceData, records: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    donors = data.donors
    batches = data.batches
    donor_index = {donor: index for index, donor in enumerate(donors)}
    method_names = (
        "primary",
        "matched_graph_zero",
        "common_effect_cmle",
        "pooled_saturated_poisson",
        "pearson_residual",
        "root_deviance_residual",
        "primary_classical_residual",
        "destroyed_link",
        "independence",
    )
    losses = {name: np.full(len(donors), np.nan) for name in method_names}
    outer_records: dict[str, Any] = {}
    for held_batch, training, validation in _batch_splits(records, donors):
        selection = _select_models(records, training)
        models, design = _outer_fit(records, training, selection)
        validation_support = _validation_support_record(
            records, validation, design["mask"]
        )
        for donor in validation:
            donor_losses = _score_outer_donor(records[donor], design["mask"], models)
            for name, value in donor_losses.items():
                losses[name][donor_index[donor]] = value
        outer_records[str(held_batch)] = {
            "training_donors": training,
            "validation_donors": validation,
            "inner_selection": _selection_record(selection),
            "outer_training_design": {
                "comparison_mask": {
                    **design["mask_record"],
                    "mask": design["mask"].astype(np.uint8).tolist(),
                },
                "rna_normalization": design["rna_normalization"],
                "adt_normalization": design["adt_normalization"],
                "rna_incidence_sha256": design["rna_incidence_sha256"],
                "adt_incidence_sha256": design["adt_incidence_sha256"],
                "product_laplacian_sha256": design["product_laplacian_sha256"],
            },
            "validation_supported_coordinate_counts": validation_support,
            "outer_fit_certificates": {
                name: model.get("fit_certificate")
                for name, model in models.items()
                if name != "primary_classical_residual"
            },
        }
    if any(not np.isfinite(values).all() for values in losses.values()):
        raise SourceGoRefusal("a mandatory nested outer prediction is incomplete", {})
    return {
        "donor_axis": donors,
        "batch_axis": batches,
        "losses": losses,
        "outer_folds": outer_records,
    }


def _bootstrap_seed(comparator: str) -> int:
    encoded = hashlib.sha256(f"{SOURCE_BOOTSTRAP_SALT}|{comparator}".encode()).digest()
    return int.from_bytes(encoded[:8], "little")


def _stratified_bootstrap(
    primary: np.ndarray,
    comparator: np.ndarray,
    batches: list[int],
    comparator_name: str,
) -> dict[str, Any]:
    first = np.asarray(primary, dtype=float)
    second = np.asarray(comparator, dtype=float)
    if first.shape != second.shape or first.shape != (len(batches),):
        raise ValueError("bootstrap loss axes differ")
    differences = first - second
    if not np.isfinite(differences).all():
        raise ValueError("bootstrap differences are nonfinite")
    batch_array = np.asarray(batches, dtype=int)
    generator = np.random.default_rng(_bootstrap_seed(comparator_name))
    batch_draws = []
    batch_indices: dict[str, list[int]] = {}
    for batch in sorted(set(batches)):
        indices = np.flatnonzero(batch_array == batch)
        batch_indices[str(batch)] = indices.tolist()
        draws = generator.integers(
            0, len(indices), size=(SOURCE_BOOTSTRAPS, len(indices))
        )
        batch_draws.append(differences[indices][draws].mean(axis=1))
    bootstrap = np.mean(np.asarray(batch_draws), axis=0)
    interval = np.quantile(bootstrap, (0.025, 0.975), method="linear")
    batch_means = {
        str(batch): float(np.mean(differences[batch_array == batch]))
        for batch in sorted(set(batches))
    }
    return {
        "comparator": comparator_name,
        "draws": SOURCE_BOOTSTRAPS,
        "unit": "donor resampled within physical pool",
        "equal_pool_weight": True,
        "batch_indices": batch_indices,
        "seed": _bootstrap_seed(comparator_name),
        "salt": SOURCE_BOOTSTRAP_SALT,
        "observed_equal_batch_mean_difference": float(
            np.mean(list(batch_means.values()))
        ),
        "batch_mean_differences": batch_means,
        "interval_95_percent": interval.tolist(),
        "quantile_method": "numpy linear",
    }


def _source_support_coverage(
    nested: dict[str, Any],
    final_selection: SelectionResult,
    all_source_mask: dict[str, Any],
) -> dict[str, Any]:
    masks: list[dict[str, Any]] = []
    validation_counts: list[dict[str, Any]] = []

    def add_mask(stage: str, mask: dict[str, Any]) -> None:
        checks = mask["checks"]
        masks.append(
            {
                "stage": stage,
                "coordinate_count": int(mask["coordinate_count"]),
                "mask_sha256": mask["mask_sha256"],
                "checks": checks,
            }
        )
        validation_counts.extend(
            {
                "stage": f"{stage}/training_support",
                "donor": donor,
                "supported_coordinates": int(count),
            }
            for donor, count in mask[
                "training_donor_supported_coordinate_counts"
            ].items()
        )

    def add_selection(prefix: str, selection: dict[str, Any]) -> None:
        for batch, fold in selection["folds"].items():
            mask = fold["comparison_mask"]
            add_mask(f"{prefix}/held_pool_{batch}", mask)
            validation_counts.extend(
                {
                    "stage": f"{prefix}/held_pool_{batch}",
                    "donor": donor,
                    "supported_coordinates": int(count),
                }
                for donor, count in fold[
                    "validation_supported_coordinate_counts"
                ].items()
            )

    for outer_batch, outer in nested["outer_folds"].items():
        add_selection(
            f"nested_outer_{outer_batch}/inner",
            outer["inner_selection"],
        )
        mask = outer["outer_training_design"]["comparison_mask"]
        add_mask(
            f"nested_outer_{outer_batch}/outer_training",
            mask,
        )
        validation_counts.extend(
            {
                "stage": f"nested_outer_{outer_batch}/outer_validation",
                "donor": donor,
                "supported_coordinates": int(count),
            }
            for donor, count in outer["validation_supported_coordinate_counts"].items()
        )
    add_selection("final_all_source_lopo", {"folds": final_selection.fold_records})
    add_mask("all_39_source_held_mask", all_source_mask)
    mask_floor = all(
        record["coordinate_count"] >= MINIMUM_SOURCE_COORDINATES
        and record["checks"]["at_least_232_coordinates"] is True
        and record["checks"]["every_training_donor_has_at_least_232_coordinates"]
        is True
        for record in masks
    )
    donor_floor = all(
        record["supported_coordinates"] >= MINIMUM_SOURCE_COORDINATES
        for record in validation_counts
    )
    return {
        "comparison_mask_floor_passes": mask_floor,
        "every_source_donor_support_floor_passes": donor_floor,
        "minimum_coordinates": MINIMUM_SOURCE_COORDINATES,
        "mask_certificate_count": len(masks),
        "donor_support_certificate_count": len(validation_counts),
        "minimum_mask_coordinates": min(record["coordinate_count"] for record in masks),
        "minimum_donor_supported_coordinates": min(
            record["supported_coordinates"] for record in validation_counts
        ),
        "masks": masks,
        "donor_support": validation_counts,
    }


def _source_gate(
    nested: dict[str, Any],
    final_selection: SelectionResult,
    support_coverage: dict[str, Any],
) -> dict[str, Any]:
    losses = nested["losses"]
    donors = nested["donor_axis"]
    batches = nested["batch_axis"]
    primary = np.asarray(losses["primary"], dtype=float)
    zero = np.asarray(losses["matched_graph_zero"], dtype=float)
    classical = np.asarray(losses["primary_classical_residual"], dtype=float)
    primary_summary = _equal_batch_summary(primary, donors, batches)
    zero_summary = _equal_batch_summary(zero, donors, batches)
    classical_summary = _equal_batch_summary(classical, donors, batches)
    relative = 1.0 - float(
        primary_summary["equal_batch_mean_loss"] / zero_summary["equal_batch_mean_loss"]
    )
    primary_batch = primary_summary["batch_mean_losses"]
    zero_batch = zero_summary["batch_mean_losses"]
    improved_pools = sum(
        float(primary_batch[key]) < float(zero_batch[key]) for key in primary_batch
    )
    favorable_donors = int(np.count_nonzero(primary < zero))
    zero_bootstrap = _stratified_bootstrap(primary, zero, batches, "matched_graph_zero")
    classical_bootstrap = _stratified_bootstrap(
        primary, classical, batches, "primary_classical_residual"
    )
    mandatory = (
        "primary",
        "matched_graph_zero",
        "common_effect_cmle",
        "pooled_saturated_poisson",
        "pearson_residual",
        "root_deviance_residual",
        "primary_classical_residual",
        "destroyed_link",
        "independence",
    )
    checks = {
        "selected_graph_penalty_is_nonzero": (
            final_selection.selected_primary.graph_penalty > 0.0
        ),
        "relative_equal_batch_reduction_vs_matched_zero_at_least_0_05": (
            relative >= 0.05
        ),
        "at_least_7_of_8_outer_pool_means_improve": improved_pools >= 7,
        "at_least_27_of_39_source_donors_improve": favorable_donors >= 27,
        "bootstrap_upper_95_vs_matched_zero_below_zero": (
            zero_bootstrap["interval_95_percent"][1] < 0.0
        ),
        "primary_mean_below_source_selected_classical": (
            primary_summary["equal_batch_mean_loss"]
            < classical_summary["equal_batch_mean_loss"]
        ),
        "bootstrap_upper_95_vs_source_selected_classical_below_zero": (
            classical_bootstrap["interval_95_percent"][1] < 0.0
        ),
        "all_mandatory_estimators_complete": all(
            name in losses and np.isfinite(losses[name]).all() for name in mandatory
        ),
        "comparison_mask_floor_passes": support_coverage[
            "comparison_mask_floor_passes"
        ],
        "every_source_donor_support_floor_passes": support_coverage[
            "every_source_donor_support_floor_passes"
        ],
    }
    return {
        "passes": all(checks.values()),
        "checks": checks,
        "relative_equal_batch_mean_loss_reduction_vs_matched_graph_zero": relative,
        "improved_outer_pool_means": improved_pools,
        "outer_pool_count": len(set(batches)),
        "favorable_source_donors": favorable_donors,
        "source_donor_count": len(donors),
        "primary_equal_batch_mean_loss": primary_summary["equal_batch_mean_loss"],
        "matched_graph_zero_equal_batch_mean_loss": zero_summary[
            "equal_batch_mean_loss"
        ],
        "classical_equal_batch_mean_loss": classical_summary["equal_batch_mean_loss"],
        "bootstrap": {
            "matched_graph_zero": zero_bootstrap,
            "primary_classical_residual": classical_bootstrap,
        },
        "support_coverage": support_coverage,
        "exact_sign_reference": {
            "at_least_7_of_8_pool_signs_one_sided_p": 9.0 / 256.0,
            "at_least_27_of_39_donor_signs_one_sided_p": (
                sum(math.comb(39, value) for value in range(27, 40)) / 2.0**39
            ),
        },
    }


def _permutation(control: int, axis: str) -> np.ndarray:
    if control < 0 or control >= TOPOLOGY_NULL_COUNT or axis not in {"rna", "adt"}:
        raise ValueError("topology-null index or axis is invalid")
    encoded = hashlib.sha256(f"{TOPOLOGY_NULL_SALT}|{control}|{axis}".encode()).digest()
    seed = int.from_bytes(encoded[:8], "little")
    permutation = np.random.default_rng(seed).permutation(MARKER_COUNT)
    if np.array_equal(permutation, np.arange(MARKER_COUNT)):
        permutation = np.roll(permutation, 1)
    return permutation


def _validate_topology_permutations() -> None:
    identity = np.arange(MARKER_COUNT)
    pairs = []
    for control in range(TOPOLOGY_NULL_COUNT):
        rna = _permutation(control, "rna")
        adt = _permutation(control, "adt")
        if np.array_equal(rna, identity) or np.array_equal(adt, identity):
            raise AssertionError("a topology null contains an identity permutation")
        pairs.append((tuple(map(int, rna)), tuple(map(int, adt))))
    if len(set(pairs)) != TOPOLOGY_NULL_COUNT:
        raise AssertionError("topology-null permutation pairs are not unique")


def _select_topology_null(
    records: dict[str, dict[str, Any]],
    donors: list[str],
    base: BaseConfig,
    control: int,
) -> dict[str, Any]:
    batches = [int(records[donor]["batch"]) for donor in donors]
    donor_index = {donor: index for index, donor in enumerate(donors)}
    rna_permutation = _permutation(control, "rna")
    adt_permutation = _permutation(control, "adt")
    losses = {
        StructuredConfig(
            neighbors,
            base.heterogeneity_penalty,
            base.ridge_penalty,
            graph_penalty,
            base.transport_multiplier,
        ): np.full(len(donors), np.nan)
        for neighbors, graph_penalty in product(NEIGHBOR_GRID, GRAPH_GRID)
    }
    refusals = []
    for held_batch, training, validation in _batch_splits(records, donors):
        tables, _, support = _fold_arrays(records, training)
        for neighbors in NEIGHBOR_GRID:
            design = _training_design(records, training, neighbors)
            first = design["rna_incidence"][rna_permutation]
            second = design["adt_incidence"][adt_permutation]
            for graph_penalty in GRAPH_GRID:
                config = StructuredConfig(
                    neighbors,
                    base.heterogeneity_penalty,
                    base.ridge_penalty,
                    graph_penalty,
                    base.transport_multiplier,
                )
                try:
                    fit = _fit_structured(tables, support, first, second, config)
                    for donor in validation:
                        losses[config][donor_index[donor]] = _population_loss(
                            records[donor],
                            design["mask"],
                            fit["population_log_odds"],
                            config.transport_multiplier,
                        )
                except (
                    ValueError,
                    FloatingPointError,
                    CouplingEstimationRefusal,
                ) as error:
                    refusals.append(
                        {
                            "held_batch": held_batch,
                            "configuration": asdict(config),
                            "reason_code": type(error).__name__,
                            "reason": str(error),
                        }
                    )
    selected = _select_structured(losses, donors, batches)
    design = _training_design(records, donors, selected.graph_neighbors)
    tables, _, support = _fold_arrays(records, donors)
    first = design["rna_incidence"][rna_permutation]
    second = design["adt_incidence"][adt_permutation]
    null_laplacian = product_hypergraph_laplacian(first, second)
    baseline_laplacian = product_hypergraph_laplacian(
        design["rna_incidence"], design["adt_incidence"]
    )
    joint_permutation = np.asarray(
        [
            rna_marker * MARKER_COUNT + adt_marker
            for rna_marker in rna_permutation
            for adt_marker in adt_permutation
        ],
        dtype=int,
    )
    expected_laplacian = baseline_laplacian[
        np.ix_(joint_permutation, joint_permutation)
    ]
    if not np.allclose(null_laplacian, expected_laplacian, rtol=0.0, atol=1e-12):
        raise AssertionError("topology-null row permutation changed the spectrum")
    fit = _fit_structured(tables, support, first, second, selected)
    return {
        "control_index": control,
        "rna_row_permutation": rna_permutation.tolist(),
        "adt_row_permutation": adt_permutation.tolist(),
        "selected_configuration": asdict(selected),
        "selection_rule": (
            "complete Stage-B k x graph source LOPO selection; minimum "
            "equal-pool mean loss, then graph penalty, then k"
        ),
        "selected_source_losses": losses[selected].tolist(),
        "loss_curve": [
            _curve_entry(config, values, donors, batches)
            for config, values in sorted(losses.items())
        ],
        "refusals": refusals,
        "rna_incidence_sha256": _array_sha256(first),
        "adt_incidence_sha256": _array_sha256(second),
        "product_laplacian_sha256": _array_sha256(null_laplacian),
        "same_k_unpermuted_product_laplacian_sha256": _array_sha256(baseline_laplacian),
        "population_log_odds": fit["population_log_odds"].tolist(),
        "population_log_odds_sha256": _array_sha256(fit["population_log_odds"]),
        "fit_certificate": fit["fit_certificate"],
    }


def _serialize_population_model(
    family: str, configuration: dict[str, Any], fit: dict[str, Any]
) -> dict[str, Any]:
    population = np.asarray(fit["population_log_odds"], dtype=float)
    return {
        "family": family,
        "configuration": configuration,
        "population_log_odds": population.tolist(),
        "population_log_odds_sha256": _array_sha256(population),
        "fit_certificate": fit["fit_certificate"],
    }


def _fit_final_candidate(
    data: SourceData,
    records: dict[str, dict[str, Any]],
    selection: SelectionResult,
    topology_nulls: list[dict[str, Any]],
) -> dict[str, Any]:
    models, design = _outer_fit(records, data.donors, selection)
    primary = models["primary"]
    zero = models["matched_graph_zero"]
    common = models["common_effect_cmle"]
    poisson = models["pooled_saturated_poisson"]
    destroyed = models["destroyed_link"]
    serialized_models: dict[str, Any] = {
        "primary": _serialize_population_model(
            "penalty-complete exact-conditional product-hypergraph",
            primary["configuration"],
            primary,
        ),
        "matched_graph_zero": _serialize_population_model(
            "penalty-complete exact-conditional graph-zero",
            zero["configuration"],
            zero,
        ),
        "common_effect_cmle": _serialize_population_model(
            "donor-stratified exact-conditional common interaction",
            common["configuration"],
            common,
        ),
        "pooled_saturated_poisson": {
            **_serialize_population_model(
                "unstratified pooled saturated 2x2 Poisson interaction",
                poisson["configuration"],
                poisson,
            ),
            "pooled_tables_sha256": poisson["fit_certificate"]["pooled_tables_sha256"],
        },
        "destroyed_link": _serialize_population_model(
            "primary estimator after deterministic within-donor link destruction",
            destroyed["configuration"],
            destroyed,
        ),
        "independence": {"family": "recipient-margin Poisson independence"},
    }
    for family in RESIDUAL_FAMILIES:
        model = models[f"{family}_residual"]
        coordinate = np.asarray(model["pooled_coordinate"], dtype=float)
        serialized_models[f"{family}_residual"] = {
            "family": f"signed one-df {family} interaction coordinate",
            "configuration": model["configuration"],
            "pooled_coordinate": coordinate.tolist(),
            "pooled_coordinate_sha256": _array_sha256(coordinate),
            "fit_certificate": model["fit_certificate"],
        }
    selected_family = selection.selected_residual.family
    serialized_models["primary_classical_residual"] = {
        **serialized_models[f"{selected_family}_residual"],
        "selection_role": "single source-CV-selected classical comparator",
        "model_reference": f"{selected_family}_residual",
    }
    coordinate_axis = [
        f"{rna.rna_gene}|{adt.adt_feature}" for rna in PANEL for adt in PANEL
    ]
    return {
        "canonical_configuration": {
            **asdict(selection.selected_primary),
            "estimator": (
                "penalty-complete exact-conditional product-hypergraph coupling field"
            ),
            "hyperedge_rule": (
                "each marker plus k nearest source-profile markers; duplicate "
                "memberships removed"
            ),
            "profile_metric": (
                "Euclidean after source physical-pool centering and pooled SD"
            ),
        },
        "coordinate_axis": coordinate_axis,
        "coordinate_axis_sha256": _axis_sha256(coordinate_axis),
        "comparison_mask": {
            **design["mask_record"],
            "mask": design["mask"].astype(np.uint8).tolist(),
        },
        "source_geometry": {
            "donor_axis": data.donors,
            "batch_axis": data.batches,
            "rna_jeffreys_logit_profiles": design["rna_raw"].tolist(),
            "adt_full96_clr_profiles": design["adt_raw"].tolist(),
            "rna_normalization": design["rna_normalization"],
            "adt_normalization": design["adt_normalization"],
            "rna_normalized_profiles": design["rna_normalized"].tolist(),
            "adt_normalized_profiles": design["adt_normalized"].tolist(),
            "rna_incidence": design["rna_incidence"].tolist(),
            "adt_incidence": design["adt_incidence"].tolist(),
            "rna_incidence_sha256": design["rna_incidence_sha256"],
            "adt_incidence_sha256": design["adt_incidence_sha256"],
            "product_laplacian_sha256": design["product_laplacian_sha256"],
        },
        "models": serialized_models,
        "topology_nulls": {
            "count": TOPOLOGY_NULL_COUNT,
            "selection_aware": True,
            "spectrum_preserving": True,
            "construction": (
                "independent deterministic nonidentity RNA and ADT incidence-row "
                "permutations; every null reruns complete Stage-B source selection"
            ),
            "held_empirical_p_formula": (
                "(1 + count(null equal-pool mean loss <= primary mean loss)) / 64"
            ),
            "controls": topology_nulls,
        },
        "held_access": {
            "internal_batches": [8, 9],
            "confirmation_batches": [10, 11],
            "authorized_by_source_artifact": False,
            "requires_public_candidate_freeze": True,
            "no_retuning_after_source": True,
        },
    }


def _nested_loss_record(nested: dict[str, Any]) -> dict[str, Any]:
    donors = nested["donor_axis"]
    batches = nested["batch_axis"]
    return {
        name: _curve_entry({"method": name}, np.asarray(values), donors, batches)
        for name, values in nested["losses"].items()
    }


def _develop(data: SourceData) -> dict[str, Any]:
    records = _records(data)
    nested = _nested_source_predictions(data, records)
    final_selection = _select_models(records, data.donors)
    _, all_source_mask_record = _training_mask(records, data.donors)
    support_coverage = _source_support_coverage(
        nested, final_selection, all_source_mask_record
    )
    gate = _source_gate(nested, final_selection, support_coverage)
    development = {
        "nested_outer_predictions": {
            "donor_axis": nested["donor_axis"],
            "batch_axis": nested["batch_axis"],
            "outer_folds": nested["outer_folds"],
            "losses": _nested_loss_record(nested),
        },
        "final_all_source_selection": _selection_record(final_selection),
        "source_go_gate": gate,
    }
    if not gate["passes"]:
        return {
            "status": "SOURCE_GO_GATE_FAILED",
            "source_statistical_go_gate_passed": False,
            "internal_numeric_access_authorized": False,
            "development": development,
            "candidate": None,
        }
    _validate_topology_permutations()
    topology_nulls = []
    for control in range(TOPOLOGY_NULL_COUNT):
        topology_nulls.append(
            _select_topology_null(
                records,
                data.donors,
                final_selection.selected_base,
                control,
            )
        )
        print(
            json.dumps(
                {
                    "completed_topology_null": control + 1,
                    "topology_null_count": TOPOLOGY_NULL_COUNT,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    candidate = _fit_final_candidate(data, records, final_selection, topology_nulls)
    return {
        "status": "SOURCE_GO_GATE_PASSED_CANDIDATE_FROZEN",
        "source_statistical_go_gate_passed": True,
        "internal_numeric_access_authorized": False,
        "internal_access_requires_public_candidate_freeze": True,
        "development": development,
        "candidate": candidate,
    }


def _artifact(path: Path, manifest_path: Path, data: SourceData) -> dict[str, Any]:
    attempt_path = ROOT / data.attempt_path
    reduction_terminal_path = ROOT / data.reduction_terminal_path
    input_snapshot = {
        "source_reduction_sha256": _sha256(path),
        "source_reduction_manifest_sha256": _sha256(manifest_path),
        "source_campaign_attempt_sha256": _sha256(attempt_path),
        "source_reduction_terminal_sha256": _sha256(reduction_terminal_path),
    }
    if (
        input_snapshot["source_reduction_sha256"] != data.manifest["output"]["sha256"]
        or input_snapshot["source_reduction_manifest_sha256"] != data.manifest_sha256
        or input_snapshot["source_campaign_attempt_sha256"] != data.attempt_sha256
        or input_snapshot["source_reduction_terminal_sha256"]
        != data.reduction_terminal_sha256
    ):
        raise PermissionError("loaded source bytes differ from their frozen manifest")
    implementation = _implementation_snapshot()
    try:
        result = _develop(data)
    except CouplingEstimationRefusal as error:
        result = {
            "status": "SOURCE_DEVELOPMENT_REFUSED",
            "source_statistical_go_gate_passed": False,
            "internal_numeric_access_authorized": False,
            "development": None,
            "candidate": None,
            "refusal": {
                "reason_code": type(error).__name__,
                "reason": str(error),
                "details": error.details if isinstance(error, SourceGoRefusal) else {},
            },
        }
    if input_snapshot != {
        "source_reduction_sha256": _sha256(path),
        "source_reduction_manifest_sha256": _sha256(manifest_path),
        "source_campaign_attempt_sha256": _sha256(attempt_path),
        "source_reduction_terminal_sha256": _sha256(reduction_terminal_path),
    }:
        raise PermissionError("source bytes changed during candidate development")
    if implementation != _implementation_snapshot():
        raise PermissionError("implementation bytes changed during development")
    return {
        "schema": "gse181897-b0-b7-source-candidate/1.0",
        "accession": "GSE181897",
        "numeric_batches_processed": list(SOURCE_BATCHES),
        "numeric_condition_processed": CONTROL_CONDITION,
        "forbidden_numeric_batches": list(FORBIDDEN_BATCHES),
        "source_reduction_path": _display_path(path),
        "source_reduction_sha256": input_snapshot["source_reduction_sha256"],
        "source_reduction_manifest_path": _display_path(manifest_path),
        "source_reduction_manifest_sha256": input_snapshot[
            "source_reduction_manifest_sha256"
        ],
        "source_reduction_manifest_schema": data.manifest["schema"],
        "source_campaign_attempt": {
            "path": data.attempt_path,
            "sha256": data.attempt_sha256,
        },
        "source_reduction_terminal": {
            "path": data.reduction_terminal_path,
            "sha256": data.reduction_terminal_sha256,
        },
        "donor_axis": data.donors,
        "free_id_axis": data.free_ids,
        "batch_axis": data.batches,
        "donor_axis_sha256": _axis_sha256(data.donors),
        "cell_budget_per_donor": CELL_BUDGET,
        "marker_count": MARKER_COUNT,
        "coordinate_count": COORDINATE_COUNT,
        "source_contract": {
            "nested_outer_batches": list(SOURCE_BATCHES),
            "internal_batches_numerically_unopened": [8, 9],
            "confirmation_batches_numerically_unopened": [10, 11],
            "condition": CONTROL_CONDITION,
            "binary_states": "raw count > 0 for both RNA and ADT",
            "marker_support": "4 <= positive cells <= 124",
            "coordinate_floor": MINIMUM_SOURCE_COORDINATES,
            "training_only_masks_profiles_and_hypergraphs": True,
            "source_validation_outcomes_used_for_selection": True,
            "held_internal_or_confirmation_outcomes_used_for_selection": False,
        },
        "implementation": {
            **implementation,
            "conditional_solver": (
                "mapreg.penalty_complete_conditional_coupling."
                "fit_hierarchical_conditional_log_odds"
            ),
        },
        **result,
    }


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())


def _replace_open_json(stream: Any, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    stream.seek(0)
    stream.truncate()
    stream.write(encoded)
    stream.flush()
    os.fsync(stream.fileno())


def _run_model_one_shot(args: argparse.Namespace) -> dict[str, Any]:
    args.model_terminal.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        args.model_terminal, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
    )
    with os.fdopen(descriptor, "w+") as stream:
        attempt_sha256 = _sha256(args.attempt) if args.attempt.is_file() else None
        _replace_open_json(
            stream,
            {
                "schema": "gse181897-source-model-terminal/1.0",
                "status": "SOURCE_MODEL_EXECUTION_STARTED_ATTEMPT_CONSUMED",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "attempt_path": _display_path(args.attempt),
                "attempt_sha256": attempt_sha256,
            },
        )
        try:
            _require_runtime()
            data = _load_source(
                args.input,
                args.manifest,
                args.attempt,
                args.output,
                args.model_terminal,
                args.reduction_terminal,
            )
            result = _artifact(args.input, args.manifest, data)
            _write_json_exclusive(args.output, result)
            payload = {
                "schema": "gse181897-source-model-terminal/1.0",
                "status": (
                    "SOURCE_MODEL_COMPLETE_STATISTICAL_GO_PASSED"
                    if result.get("source_statistical_go_gate_passed") is True
                    else "SOURCE_MODEL_COMPLETE_STATISTICAL_GO_NOT_PASSED"
                ),
                "attempt_path": _display_path(args.attempt),
                "attempt_sha256": attempt_sha256,
                "model_output_path": _display_path(args.output),
                "model_output_sha256": _sha256(args.output),
                "internal_numeric_access_authorized": False,
                "requires_public_candidate_freeze": True,
            }
        except BaseException as error:
            payload = {
                "schema": "gse181897-source-model-terminal/1.0",
                "status": "TERMINAL_SOURCE_MODEL_REFUSAL",
                "attempt_path": _display_path(args.attempt),
                "attempt_sha256": attempt_sha256,
                "reason_code": type(error).__name__,
                "reason": str(error)
                .replace(str(ROOT.resolve()), "<repository>")
                .replace(str(Path.home().resolve()), "<home>"),
                "rerun_forbidden": True,
            }
        _replace_open_json(stream, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--attempt", type=Path, default=DEFAULT_SOURCE_ATTEMPT)
    parser.add_argument(
        "--reduction-terminal", type=Path, default=DEFAULT_REDUCTION_TERMINAL
    )
    parser.add_argument("--model-terminal", type=Path, default=DEFAULT_MODEL_TERMINAL)
    args = parser.parse_args()
    terminal = _run_model_one_shot(args)
    print(json.dumps(terminal, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
