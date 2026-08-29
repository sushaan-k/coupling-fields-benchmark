"""Outcome-blind source reducer for the GSE181897 control CITE-seq panel.

The raw H5AD contains twelve physical pools.  This module freezes its axes in
one metadata-only artifact, then materializes counts only for the selected
control cells in development pools 0--7.  Pools 8--11 and every non-control
cell remain numerically unopened.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any
import urllib.request

import h5py
import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = Path.home() / "Library/Caches/coupling-fields/gse181897-source"
DEFAULT_OUTPUT_DIRECTORY = ROOT / "data/development/gse181897_source"
DEFAULT_PREFLIGHT = DEFAULT_OUTPUT_DIRECTORY / "axis_preflight_v2.json"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIRECTORY / "reduced_batches_0_7_control_v1.npz"
DEFAULT_MANIFEST = (
    DEFAULT_OUTPUT_DIRECTORY / "reduction_batches_0_7_control_manifest_v1.json"
)
DEFAULT_SOURCE_AUTHORIZATION = (
    DEFAULT_OUTPUT_DIRECTORY / "source_campaign_authorization_v1.json"
)
DEFAULT_SOURCE_ATTEMPT = DEFAULT_OUTPUT_DIRECTORY / "source_attempt_v1.json"
DEFAULT_REDUCTION_TERMINAL = (
    DEFAULT_OUTPUT_DIRECTORY / "source_reduction_terminal_v1.json"
)
DEFAULT_MODEL_OUTPUT = ROOT / "results/development/gse181897_source_candidate_v1.json"
DEFAULT_MODEL_TERMINAL = (
    ROOT / "results/development/gse181897_source_model_terminal_v1.json"
)
CANDIDATE_TAG = "gse181897-control-citeseq-v1-candidate"
IMPLEMENTATION_TAG = "gse181897-control-citeseq-v1-implementation"
AXIS_PREFLIGHT_TAG = "gse181897-control-citeseq-v1-axis-preflight-v2"
SOURCE_AUTHORIZATION_TAG = "gse181897-control-citeseq-v1-source-authorized"
PUBLIC_ORIGIN_URL = "https://github.com/sushaan-k/coupling-fields-benchmark.git"
CANDIDATE_PATH = (
    "data/confirmation/gse181897_control_citeseq/candidate_designation_v1.json"
)

SOURCE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE181nnn/GSE181897/suppl/"
    "GSE181897_concat.4.raw.h5ad.gz"
)
SOURCE_ARCHIVE_NAME = "GSE181897_concat.4.raw.h5ad.gz"
SOURCE_H5AD_NAME = "GSE181897_concat.4.raw.h5ad"
SOURCE_ARCHIVE_BYTES = 1_011_162_509
SOURCE_H5AD_BYTES = 3_063_713_137
SOURCE_ARCHIVE_SHA256 = (
    "7fe58432f2f238319e81c9218eb35b5f7fbdae6f10f3d87ce9a6044ee851675b"
)
SOURCE_H5AD_SHA256 = "183d7756c750fb0ca57f381512fe784df6249ec5a5478a9caf6a62df55cba56c"
EXPECTED_X_SHAPE = (136_142, 20_399)
EXPECTED_X_DATA_LENGTH = 292_741_570
EXPECTED_OBS_AXIS_SHA256 = (
    "24560e2df6a268b11509d2ab23ae898ae17cf699080ed604e710fa266db418c5"
)
EXPECTED_VAR_AXIS_SHA256 = (
    "7c7511ba42740fe6127fac90ebe181f49122ff07014908f14be11fafdc143e1e"
)
EXPECTED_BATCH_CATEGORIES = tuple(str(value) for value in range(12))
EXPECTED_CONDITION_CATEGORIES = ("0", "A", "B", "C", "G", "P", "R")
EXPECTED_DONOR_CATEGORIES = tuple(str(value) for value in range(64))
EXPECTED_CONDITION_ZERO_CELLS = 455
EXPECTED_CONTROL_CELLS = 22_732
EXPECTED_RNA_FEATURE_COUNT = 20_303
EXPECTED_PROTEIN_FEATURE_COUNT = 96

SOURCE_BATCHES = tuple(range(8))
HELD_BATCHES = tuple(range(8, 12))
CONTROL_CONDITION = "C"
EXCLUDED_DEVELOPMENT_DONORS = {"23": 6, "62": 2}
DEVELOPMENT_DONORS_BY_BATCH = {
    0: ("34", "35", "45", "57", "58"),
    1: ("18", "22", "24", "55", "59", "61"),
    2: ("13", "29", "31", "42", "43"),
    3: ("5", "12", "30", "36", "60", "63"),
    4: ("10", "14", "38", "47"),
    5: ("11", "15", "25", "39", "49"),
    6: ("1", "4", "7", "27"),
    7: ("9", "32", "37", "44"),
}
INTERNAL_DONORS_BY_BATCH = {
    8: ("3", "16", "19", "48", "50"),
    9: ("0", "2", "17", "33"),
}
CONFIRMATION_DONORS_BY_BATCH = {
    10: ("8", "21", "28", "41", "53", "56"),
    11: ("6", "20", "26", "40", "46", "54"),
}
EXCLUDED_CONTROL_DONORS = {"23": (6, 2), "51": (9, 70), "52": (8, 91), "62": (2, 64)}
CELL_BUDGET = 128
CELL_SELECTION_SALT = "GSE181897-CONTROL-CELL-BUDGET-v1"
MINIMUM_MARKER_POSITIVES = 4
MAXIMUM_MARKER_POSITIVES = CELL_BUDGET - MINIMUM_MARKER_POSITIVES
MINIMUM_SOURCE_COORDINATES = 232
MINIMUM_FREE_BYTES_AFTER_ACQUISITION = 512 << 20
SOURCE_ATTEMPT_SCHEMA = "gse181897-source-campaign-attempt/1.0"
SOURCE_ATTEMPT_STATUS = "CLAIMED_ONE_SHOT_SOURCE_CAMPAIGN"
SOURCE_AUTHORIZATION_SCHEMA = "gse181897-source-campaign-authorization/1.0"
SOURCE_AUTHORIZATION_STATUS = "PUBLIC_FREEZE_VERIFIED_SOURCE_NUMERIC_ACCESS_AUTHORIZED"
REQUIRED_THREAD_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
CAMPAIGN_IMPLEMENTATION_FILES = (
    "experiments/reduce_gse181897_source.py",
    "experiments/develop_gse181897_source_models.py",
    "experiments/confirm_gse181897_held.py",
    "tests/test_reduce_gse181897_source.py",
    "tests/test_develop_gse181897_source_models.py",
    "tests/test_confirm_gse181897_held.py",
    "tests/test_gse181897_protocol.py",
    "tests/test_gse181897_axis_preflight_artifact.py",
    "tests/test_penalty_complete_conditional_coupling.py",
    "tests/test_common_effect_conditional.py",
    "tests/test_heterogeneity_adaptive_coupling.py",
    "tests/test_classical_residuals_full.py",
    "tests/test_coupling_fields.py",
    "tests/test_table_prediction.py",
    "mapreg/penalty_complete_conditional_coupling.py",
    "mapreg/heterogeneity_adaptive_coupling.py",
    "mapreg/common_effect_conditional.py",
    "mapreg/classical_residuals.py",
    "mapreg/coupling_fields.py",
    "mapreg/table_prediction.py",
    "mapreg/__init__.py",
    "data/confirmation/gse181897_control_citeseq/candidate_designation_v1.json",
    "data/confirmation/gse181897_control_citeseq/protocol_v1.json",
    "docs/GSE181897_CONTROL_CITESEQ_SEQUENTIAL_CONFIRMATION_PROTOCOL_2026-08-29.md",
    "data/confirmation/gse181897_control_citeseq/internal_prepare_authorization_template_v1.json",
    "data/confirmation/gse181897_control_citeseq/internal_score_authorization_template_v1.json",
    "data/confirmation/gse181897_control_citeseq/confirmation_prepare_authorization_template_v1.json",
    "data/confirmation/gse181897_control_citeseq/confirmation_score_authorization_template_v1.json",
    "pyproject.toml",
    "requirements.txt",
)


@dataclass(frozen=True)
class Cognate:
    rna_gene: str
    rna_feature_id: str
    protein_label: str
    adt_feature: str


PANEL = (
    Cognate("CD1C", "ENSG00000158481", "CD1c", "CD1c|CD1C"),
    Cognate("CD2", "ENSG00000116824", "CD2", "CD2|CD2"),
    Cognate("CD4", "ENSG00000010610", "CD4", "CD4|CD4"),
    Cognate("CD7", "ENSG00000173762", "CD7", "CD7|CD7"),
    Cognate("CD8A", "ENSG00000153563", "CD8", "CD8|CD8A"),
    Cognate("ITGAM", "ENSG00000169896", "CD11b", "CD11b|ITGAM"),
    Cognate("ITGAX", "ENSG00000140678", "CD11c", "CD11c|ITGAX"),
    Cognate("CD14", "ENSG00000170458", "CD14", "CD14|CD14"),
    Cognate("MS4A1", "ENSG00000156738", "CD20", "CD20|MS4A1"),
    Cognate("CD27", "ENSG00000139193", "CD27", "CD27|CD27"),
    Cognate("CD33", "ENSG00000105383", "CD33", "CD33|CD33"),
    Cognate("CD34", "ENSG00000174059", "CD34", "CD34|CD34"),
    Cognate("CD38", "ENSG00000004468", "CD38", "CD38|CD38"),
    Cognate("CD69", "ENSG00000110848", "CD69", "CD69|CD69"),
    Cognate("CD80", "ENSG00000121594", "CD80", "CD80|CD80"),
    Cognate("CD86", "ENSG00000114013", "CD86", "CD86|CD86"),
    Cognate("CD163", "ENSG00000177575", "CD163", "CD163|CD163"),
)


@dataclass(frozen=True)
class SourcePlan:
    donor_axis: tuple[str, ...]
    free_id_axis: tuple[str, ...]
    batch_axis: tuple[int, ...]
    selected_rows: np.ndarray
    selected_barcodes: np.ndarray
    authorized_rows: np.ndarray
    donor_audit: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class AxisInspection:
    payload: dict[str, Any]
    plan: SourcePlan
    rna_columns: tuple[int, ...]
    adt_columns: tuple[int, ...]
    protein_columns: tuple[int, ...]
    protein_axis: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _axis_sha256(values: list[str] | tuple[str, ...] | np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _decode(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in {"O", "S", "U"}:
        return array
    return np.asarray(
        [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in array
        ],
        dtype=str,
    )


def _attribute_text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _frame_index(group: h5py.Group) -> np.ndarray:
    key = _attribute_text(group.attrs.get("_index", "_index"))
    if key != "_index" or key not in group or not isinstance(group[key], h5py.Dataset):
        raise ValueError(f"{group.name} lacks the frozen _index dataset")
    values = _decode(group[key][...])
    if values.ndim != 1 or values.dtype.kind != "U":
        raise ValueError(f"{group.name}/_index is not a string axis")
    return values


def _frame_column(group: h5py.Group, key: str) -> np.ndarray:
    if key not in group:
        raise ValueError(f"{group.name} lacks frozen column {key!r}")
    item = group[key]
    if isinstance(item, h5py.Group) and {"codes", "categories"}.issubset(item):
        codes = np.asarray(item["codes"][...], dtype=np.int64)
        categories = _decode(item["categories"][...])
        if np.any(codes < 0) or np.any(codes >= len(categories)):
            raise ValueError(f"{item.name} contains invalid categorical codes")
        return categories[codes]
    if not isinstance(item, h5py.Dataset):
        raise ValueError(f"{group.name}/{key} has an unsupported encoding")
    values = item[...]
    if "categories" in item.attrs:
        reference = item.attrs["categories"]
        if not isinstance(reference, h5py.Reference) or not reference:
            raise ValueError(f"{item.name} has a malformed category reference")
        category_item = group.file[reference]
        if not isinstance(category_item, h5py.Dataset):
            raise ValueError(f"{item.name} category reference is not a dataset")
        categories = _decode(category_item[...])
        codes = np.asarray(values, dtype=np.int64)
        if np.any(codes < 0) or np.any(codes >= len(categories)):
            raise ValueError(f"{item.name} contains invalid categorical codes")
        values = categories[codes]
    values = _decode(values)
    if values.ndim != 1:
        raise ValueError(f"{group.name}/{key} is not one-dimensional")
    return values


def _category_axis(group: h5py.Group, key: str) -> list[str]:
    item = group[key]
    if isinstance(item, h5py.Group):
        return _decode(item["categories"][...]).tolist()
    reference = item.attrs.get("categories")
    if not isinstance(reference, h5py.Reference) or not reference:
        raise ValueError(f"{item.name} lacks its frozen category reference")
    return _decode(group.file[reference][...]).tolist()


def _column_metadata(group: h5py.Group, key: str) -> dict[str, Any]:
    item = group[key]
    if not isinstance(item, h5py.Dataset):
        raise ValueError(f"{group.name}/{key} is not the frozen dataset encoding")
    reference = item.attrs.get("categories")
    if not isinstance(reference, h5py.Reference) or not reference:
        raise ValueError(f"{item.name} lacks its frozen category reference")
    category_item = group.file[reference]
    if not isinstance(category_item, h5py.Dataset):
        raise ValueError(f"{item.name} category reference is not a dataset")
    return {
        "path": item.name,
        "shape": list(item.shape),
        "dtype": item.dtype.name,
        "encoding": "legacy categorical codes with HDF5 category reference",
        "category_path": category_item.name,
        "category_shape": list(category_item.shape),
        "category_dtype": category_item.dtype.name,
    }


def _selection_hash(batch: int, donor: str, barcode: str) -> str:
    return hashlib.sha256(
        f"{CELL_SELECTION_SALT}|{batch}|{donor}|{barcode}".encode()
    ).hexdigest()


def _expected_donor_axis() -> tuple[str, ...]:
    return tuple(
        donor
        for batch in SOURCE_BATCHES
        for donor in DEVELOPMENT_DONORS_BY_BATCH[batch]
    )


def _build_source_plan(
    barcodes: np.ndarray,
    batches: np.ndarray,
    conditions: np.ndarray,
    exp_ids: np.ndarray,
    free_ids: np.ndarray,
) -> SourcePlan:
    length = len(barcodes)
    if not all(
        len(axis) == length for axis in (batches, conditions, exp_ids, free_ids)
    ):
        raise ValueError("GSE181897 obs columns have inconsistent lengths")
    if length == 0 or len(set(barcodes.tolist())) != length:
        raise ValueError("GSE181897 cell axis is empty or duplicated")
    observed_pairs = set(zip(exp_ids.tolist(), free_ids.tolist()))
    if (
        len(observed_pairs) != len(EXPECTED_DONOR_CATEGORIES)
        or len({pair[0] for pair in observed_pairs}) != len(EXPECTED_DONOR_CATEGORIES)
        or len({pair[1] for pair in observed_pairs}) != len(EXPECTED_DONOR_CATEGORIES)
    ):
        raise ValueError("exp_id and free_id are not a global 64-donor bijection")

    source_control = np.isin(batches, [str(batch) for batch in SOURCE_BATCHES]) & (
        conditions == CONTROL_CONDITION
    )
    observed_units = set(
        zip(batches[source_control].astype(int), exp_ids[source_control])
    )
    expected_units = {
        (batch, donor)
        for batch, donors in DEVELOPMENT_DONORS_BY_BATCH.items()
        for donor in donors
    } | {(batch, donor) for donor, batch in EXCLUDED_DEVELOPMENT_DONORS.items()}
    if observed_units != expected_units:
        raise ValueError(
            "source control batch/donor units differ from the frozen split"
        )

    authorized = source_control & ~np.isin(exp_ids, tuple(EXCLUDED_DEVELOPMENT_DONORS))
    donor_axis: list[str] = []
    free_axis: list[str] = []
    batch_axis: list[int] = []
    rows_by_donor: list[np.ndarray] = []
    barcodes_by_donor: list[np.ndarray] = []
    donor_audit: list[dict[str, Any]] = []
    for batch in SOURCE_BATCHES:
        for donor in DEVELOPMENT_DONORS_BY_BATCH[batch]:
            eligible = np.flatnonzero(
                authorized & (batches.astype(str) == str(batch)) & (exp_ids == donor)
            )
            if len(eligible) < CELL_BUDGET:
                raise ValueError(
                    f"batch {batch} donor {donor} has {len(eligible)} control cells; "
                    f"need {CELL_BUDGET}"
                )
            observed_free = sorted(set(free_ids[eligible].tolist()))
            if len(observed_free) != 1:
                raise ValueError(f"donor {donor} has a nonunique free_id")
            hash_ordered = sorted(
                eligible.tolist(),
                key=lambda row: (
                    _selection_hash(batch, donor, str(barcodes[row])),
                    str(barcodes[row]),
                    row,
                ),
            )[:CELL_BUDGET]
            selected = np.asarray(sorted(hash_ordered), dtype=np.int64)
            selected_barcodes = barcodes[selected].astype(str)
            if np.any(conditions[selected] != CONTROL_CONDITION) or np.any(
                ~np.isin(batches[selected].astype(int), SOURCE_BATCHES)
            ):
                raise PermissionError(
                    "cell selection crossed the numeric-access boundary"
                )
            donor_axis.append(donor)
            free_axis.append(observed_free[0])
            batch_axis.append(batch)
            rows_by_donor.append(selected)
            barcodes_by_donor.append(selected_barcodes)
            donor_audit.append(
                {
                    "batch": batch,
                    "exp_id": int(donor),
                    "free_id": int(observed_free[0]),
                    "eligible_control_cells": len(eligible),
                    "selected_cells": CELL_BUDGET,
                    "selected_row_indices_sha256": _array_sha256(selected),
                    "selected_cell_axis_sha256": _axis_sha256(selected_barcodes),
                }
            )
    if tuple(donor_axis) != _expected_donor_axis():
        raise AssertionError("development donor axis differs from the frozen donors")
    selected_rows = np.stack(rows_by_donor)
    selected_barcodes = np.stack(barcodes_by_donor)
    if len(set(selected_rows.ravel().tolist())) != selected_rows.size:
        raise ValueError("the source selection contains a duplicated cell row")
    if np.any(conditions[selected_rows.ravel()] == "0"):
        raise PermissionError("condition-0 mispool cells entered the source selection")
    return SourcePlan(
        donor_axis=tuple(donor_axis),
        free_id_axis=tuple(free_axis),
        batch_axis=tuple(batch_axis),
        selected_rows=selected_rows,
        selected_barcodes=selected_barcodes,
        authorized_rows=authorized,
        donor_audit=tuple(donor_audit),
    )


def _validate_control_allocation(
    batches: np.ndarray, conditions: np.ndarray, exp_ids: np.ndarray
) -> dict[str, dict[str, int]]:
    control = conditions == CONTROL_CONDITION
    if int(np.count_nonzero(control)) != EXPECTED_CONTROL_CELLS:
        raise ValueError("control condition no longer has the exact frozen cell count")
    if len(set(exp_ids[control].tolist())) != len(EXPECTED_DONOR_CATEGORIES):
        raise ValueError("control condition does not contain the exact donor count")
    expected_units = {
        (batch, donor)
        for allocation in (
            DEVELOPMENT_DONORS_BY_BATCH,
            INTERNAL_DONORS_BY_BATCH,
            CONFIRMATION_DONORS_BY_BATCH,
        )
        for batch, donors in allocation.items()
        for donor in donors
    } | {
        (batch, donor)
        for donor, (batch, expected_count) in EXCLUDED_CONTROL_DONORS.items()
        if expected_count > 0
    }
    observed_units = set(
        zip(batches[control].astype(int).tolist(), exp_ids[control].tolist())
    )
    if observed_units != expected_units:
        raise ValueError("control batch/donor allocation differs from the frozen split")
    counts: dict[str, dict[str, int]] = {}
    for batch, donor in sorted(
        expected_units, key=lambda unit: (unit[0], int(unit[1]))
    ):
        observed = int(
            np.count_nonzero(
                control & (batches.astype(str) == str(batch)) & (exp_ids == donor)
            )
        )
        counts.setdefault(str(batch), {})[donor] = observed
        if donor not in EXCLUDED_CONTROL_DONORS and observed < CELL_BUDGET:
            raise ValueError(
                f"designated batch {batch} donor {donor} has fewer than "
                f"{CELL_BUDGET} control cells"
            )
    for donor, (batch, expected_count) in EXCLUDED_CONTROL_DONORS.items():
        observed = counts[str(batch)][donor]
        if observed != expected_count:
            raise ValueError(f"excluded donor {donor} has the wrong control-cell count")
    return counts


def _matrix_metadata(matrix: h5py.Group) -> dict[str, Any]:
    if not isinstance(matrix, h5py.Group):
        raise ValueError("/X must be a sparse-matrix group")
    encoding = _attribute_text(matrix.attrs.get("encoding-type", ""))
    version = _attribute_text(matrix.attrs.get("encoding-version", ""))
    shape = tuple(int(value) for value in matrix.attrs.get("shape", ()))
    if encoding != "csr_matrix" or version != "0.1.0" or shape != EXPECTED_X_SHAPE:
        raise ValueError("/X differs from the frozen CSR v0.1.0 shape")
    if set(matrix.keys()) != {"data", "indices", "indptr"}:
        raise ValueError("/X sparse datasets differ from the frozen schema")
    data, indices, indptr = (matrix[name] for name in ("data", "indices", "indptr"))
    if not all(isinstance(value, h5py.Dataset) for value in (data, indices, indptr)):
        raise ValueError("/X sparse members must be datasets")
    exact = {
        "data": ((EXPECTED_X_DATA_LENGTH,), "float32"),
        "indices": ((EXPECTED_X_DATA_LENGTH,), "int32"),
        "indptr": ((EXPECTED_X_SHAPE[0] + 1,), "int32"),
    }
    for name, item in (("data", data), ("indices", indices), ("indptr", indptr)):
        expected_shape, expected_dtype = exact[name]
        if item.shape != expected_shape or item.dtype.name != expected_dtype:
            raise ValueError(f"/X/{name} differs from the frozen shape or dtype")
    return {
        "path": "/X",
        "encoding_type": encoding,
        "encoding_version": version,
        "shape": list(shape),
        "datasets": {
            name: {
                "path": f"/X/{name}",
                "shape": list(matrix[name].shape),
                "dtype": matrix[name].dtype.name,
                "chunks": list(matrix[name].chunks or ()),
                "compression": matrix[name].compression,
            }
            for name in ("data", "indices", "indptr")
        },
    }


def _feature_columns(
    names: np.ndarray,
    feature_ids: np.ndarray,
    genomes: np.ndarray,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[str, ...]]:
    if not (len(names) == len(feature_ids) == len(genomes) == EXPECTED_X_SHAPE[1]):
        raise ValueError("GSE181897 var axes have inconsistent lengths")
    rna_columns: list[int] = []
    adt_columns: list[int] = []
    for cognate in PANEL:
        rna = np.flatnonzero(
            (names == cognate.rna_gene)
            & (feature_ids == cognate.rna_feature_id)
            & (genomes == "GRCh38")
        )
        adt = np.flatnonzero(
            (names == cognate.adt_feature)
            & (feature_ids == cognate.adt_feature)
            & (genomes == "BD99AbSeq")
        )
        if len(rna) != 1 or len(adt) != 1:
            raise ValueError(
                f"cognate {cognate.rna_gene}/{cognate.adt_feature} is not unique"
            )
        rna_columns.append(int(rna[0]))
        adt_columns.append(int(adt[0]))
    protein_columns = np.flatnonzero(genomes == "BD99AbSeq").astype(int).tolist()
    protein_axis = names[protein_columns].astype(str).tolist()
    if (
        len(protein_columns) != EXPECTED_PROTEIN_FEATURE_COUNT
        or len(set(protein_axis)) != EXPECTED_PROTEIN_FEATURE_COUNT
        or any(column not in protein_columns for column in adt_columns)
    ):
        raise ValueError("BD99AbSeq denominator is not the exact 96-feature panel")
    return (
        tuple(rna_columns),
        tuple(adt_columns),
        tuple(protein_columns),
        tuple(protein_axis),
    )


def _require_unique_axis(values: np.ndarray, name: str) -> int:
    axis = np.asarray(values).astype(str)
    unique_count = len(set(axis.tolist()))
    if unique_count != len(axis):
        raise ValueError(f"GSE181897 {name} index is not unique")
    return unique_count


def inspect_axes(h5ad_path: Path) -> AxisInspection:
    """Inspect metadata and matrix structure without indexing any X dataset."""

    if not h5ad_path.is_file() or h5ad_path.is_symlink():
        raise ValueError("source H5AD must be a regular non-symlink file")
    if h5ad_path.stat().st_size != SOURCE_H5AD_BYTES:
        raise ValueError("source H5AD byte count differs from the frozen file")
    source_sha256 = _sha256(h5ad_path)
    if source_sha256 != SOURCE_H5AD_SHA256:
        raise ValueError("source H5AD SHA-256 differs from the frozen file")

    with h5py.File(h5ad_path, "r") as handle:
        if set(handle.keys()) != {"X", "obs", "obsm", "obsp", "uns", "var"}:
            raise ValueError("H5AD root groups differ from the frozen schema")
        if not isinstance(handle["obs"], h5py.Group) or not isinstance(
            handle["var"], h5py.Group
        ):
            raise ValueError("H5AD obs/var must be data-frame groups")
        x_metadata = _matrix_metadata(handle["X"])
        obs = handle["obs"]
        var = handle["var"]
        for name, group in (("obs", obs), ("var", var)):
            if (
                _attribute_text(group.attrs.get("encoding-type", "")) != "dataframe"
                or _attribute_text(group.attrs.get("encoding-version", "")) != "0.1.0"
                or _attribute_text(group.attrs.get("_index", "")) != "_index"
            ):
                raise ValueError(f"/{name} differs from the frozen dataframe encoding")
        obs_index = _frame_index(obs)
        var_index = _frame_index(var)
        obs_unique_count = _require_unique_axis(obs_index, "obs")
        var_unique_count = _require_unique_axis(var_index, "var")
        if _axis_sha256(obs_index) != EXPECTED_OBS_AXIS_SHA256:
            raise ValueError("GSE181897 obs index differs from the frozen axis")
        if _axis_sha256(var_index) != EXPECTED_VAR_AXIS_SHA256:
            raise ValueError("GSE181897 var index differs from the frozen axis")
        batches = _frame_column(obs, "batch").astype(str)
        conditions = _frame_column(obs, "cond").astype(str)
        exp_ids = _frame_column(obs, "exp_id").astype(str)
        free_ids = _frame_column(obs, "free_id").astype(str)
        exact_categories = {
            "batch": list(EXPECTED_BATCH_CATEGORIES),
            "cond": list(EXPECTED_CONDITION_CATEGORIES),
            "exp_id": list(EXPECTED_DONOR_CATEGORIES),
            "free_id": list(EXPECTED_DONOR_CATEGORIES),
        }
        for key, expected in exact_categories.items():
            if _category_axis(obs, key) != expected:
                raise ValueError(f"/obs/{key} category axis differs")
        if int(np.count_nonzero(conditions == "0")) != EXPECTED_CONDITION_ZERO_CELLS:
            raise ValueError("condition-0 mispool anomaly has the wrong cell count")
        control_cell_counts = _validate_control_allocation(batches, conditions, exp_ids)
        plan = _build_source_plan(obs_index, batches, conditions, exp_ids, free_ids)

        feature_ids = _frame_column(var, "gene_ids").astype(str)
        feature_types = _frame_column(var, "feature_types").astype(str)
        genomes = _frame_column(var, "genome").astype(str)
        feature_batches = _frame_column(var, "batch").astype(str)
        if (
            "categories" in var["gene_ids"].attrs
            or _category_axis(var, "feature_types") != ["Gene Expression"]
            or _category_axis(var, "genome") != ["BD99AbSeq", "GRCh38"]
            or _category_axis(var, "batch") != ["0", "1"]
            or np.count_nonzero(genomes == "GRCh38") != EXPECTED_RNA_FEATURE_COUNT
            or np.count_nonzero(genomes == "BD99AbSeq")
            != EXPECTED_PROTEIN_FEATURE_COUNT
            or set(genomes.tolist()) != {"GRCh38", "BD99AbSeq"}
            or set(feature_types.tolist()) != {"Gene Expression"}
            or np.count_nonzero(feature_batches == "0") != EXPECTED_RNA_FEATURE_COUNT
            or np.count_nonzero(feature_batches == "1")
            != EXPECTED_PROTEIN_FEATURE_COUNT
        ):
            raise ValueError("var modality axes differ from the frozen H5AD")
        rna_columns, adt_columns, protein_columns, protein_axis = _feature_columns(
            var_index, feature_ids, genomes
        )
        exp_to_free = [
            {"exp_id": int(exp_id), "free_id": int(free_id)}
            for exp_id, free_id in sorted(
                set(zip(exp_ids.tolist(), free_ids.tolist())),
                key=lambda pair: int(pair[0]),
            )
        ]
        payload = {
            "schema": "gse181897-axis-preflight/1.1",
            "status": "AXES_FROZEN_UNIQUE_X_NUMERIC_UNREAD",
            "accession": "GSE181897",
            "source": {
                "official_url": SOURCE_URL,
                "archive_name": SOURCE_ARCHIVE_NAME,
                "archive_bytes": SOURCE_ARCHIVE_BYTES,
                "archive_sha256": SOURCE_ARCHIVE_SHA256,
                "h5ad_name": SOURCE_H5AD_NAME,
                "h5ad_bytes": SOURCE_H5AD_BYTES,
                "h5ad_sha256": source_sha256,
            },
            "hdf5": {
                "root_paths": ["/X", "/obs", "/obsm", "/obsp", "/uns", "/var"],
                "matrix": x_metadata,
                "obs": {
                    "path": "/obs",
                    "encoding_type": _attribute_text(
                        obs.attrs.get("encoding-type", "")
                    ),
                    "encoding_version": _attribute_text(
                        obs.attrs.get("encoding-version", "")
                    ),
                    "index_path": "/obs/_index",
                    "rows": len(obs_index),
                    "unique_rows": obs_unique_count,
                    "index_is_unique": True,
                    "index_axis_sha256": _axis_sha256(obs_index),
                    "frozen_columns": ["batch", "cond", "exp_id", "free_id"],
                    "dataset_contracts": {
                        key: _column_metadata(obs, key)
                        for key in ("batch", "cond", "exp_id", "free_id")
                    },
                    "category_axes": exact_categories,
                    "category_axis_sha256": {
                        key: _axis_sha256(values)
                        for key, values in exact_categories.items()
                    },
                    "condition_counts": {
                        value: int(np.count_nonzero(conditions == value))
                        for value in exact_categories["cond"]
                    },
                    "batch_counts": {
                        value: int(np.count_nonzero(batches == value))
                        for value in exact_categories["batch"]
                    },
                    "control_cell_counts_by_batch_and_exp_id": control_cell_counts,
                    "exp_id_to_free_id": exp_to_free,
                },
                "var": {
                    "path": "/var",
                    "encoding_type": _attribute_text(
                        var.attrs.get("encoding-type", "")
                    ),
                    "encoding_version": _attribute_text(
                        var.attrs.get("encoding-version", "")
                    ),
                    "index_path": "/var/_index",
                    "rows": len(var_index),
                    "unique_rows": var_unique_count,
                    "index_is_unique": True,
                    "index_axis_sha256": _axis_sha256(var_index),
                    "gene_id_axis_sha256": _axis_sha256(feature_ids),
                    "genome_axis_sha256": _axis_sha256(genomes),
                    "feature_type_axis_sha256": _axis_sha256(feature_types),
                    "dataset_contracts": {
                        "batch": _column_metadata(var, "batch"),
                        "feature_types": _column_metadata(var, "feature_types"),
                        "genome": _column_metadata(var, "genome"),
                        "gene_ids": {
                            "path": var["gene_ids"].name,
                            "shape": list(var["gene_ids"].shape),
                            "dtype": var["gene_ids"].dtype.name,
                            "encoding": "string dataset",
                        },
                    },
                    "modality_rule": (
                        "var/genome is GRCh38 for RNA and BD99AbSeq for protein; "
                        "var/feature_types is not a modality discriminator"
                    ),
                    "grch38_features": EXPECTED_RNA_FEATURE_COUNT,
                    "bd99abseq_features": EXPECTED_PROTEIN_FEATURE_COUNT,
                    "bd99abseq_axis": list(protein_axis),
                    "bd99abseq_axis_sha256": _axis_sha256(protein_axis),
                },
            },
            "panel": [
                {
                    "rna_gene": cognate.rna_gene,
                    "rna_feature_id": cognate.rna_feature_id,
                    "rna_column_zero_based": rna_columns[index],
                    "protein_label": cognate.protein_label,
                    "adt_feature": cognate.adt_feature,
                    "adt_column_zero_based": adt_columns[index],
                }
                for index, cognate in enumerate(PANEL)
            ],
            "source_plan": {
                "numeric_batches": list(SOURCE_BATCHES),
                "held_batches": list(HELD_BATCHES),
                "condition": CONTROL_CONDITION,
                "excluded_development_exp_ids": sorted(
                    int(donor) for donor in EXCLUDED_DEVELOPMENT_DONORS
                ),
                "excluded_development_control_cells": {
                    donor: EXCLUDED_CONTROL_DONORS[donor][1]
                    for donor in sorted(EXCLUDED_DEVELOPMENT_DONORS, key=int)
                },
                "donor_count": len(plan.donor_axis),
                "donor_axis": [int(donor) for donor in plan.donor_axis],
                "free_id_axis": [int(donor) for donor in plan.free_id_axis],
                "batch_axis": list(plan.batch_axis),
                "cell_budget_per_donor": CELL_BUDGET,
                "selected_rows": int(plan.selected_rows.size),
                "selected_row_axis_sha256": _array_sha256(plan.selected_rows),
                "selected_cell_axis_sha256": _axis_sha256(
                    plan.selected_barcodes.ravel()
                ),
                "donors": list(plan.donor_audit),
                "selection_rule": (
                    "select the 128 smallest salted hashes of metadata-eligible "
                    "cells, then restore deposited obs order"
                ),
                "selection_salt": CELL_SELECTION_SALT,
            },
            "numeric_access": {
                "decoded_X_entries": 0,
                "matrix_datasets_indexed": [],
                "authorized_rows": (
                    "selected cond=C cells from batches 0-7, excluding exp_id 23 and 62"
                ),
                "held_batches_8_11": "metadata only; X numeric values unopened",
                "non_control_cells": "metadata only; X numeric values unopened",
                "condition_0_mispool_cells": EXPECTED_CONDITION_ZERO_CELLS,
            },
            "reducer_sha256": _sha256(Path(__file__)),
        }
    return AxisInspection(
        payload=payload,
        plan=plan,
        rna_columns=rna_columns,
        adt_columns=adt_columns,
        protein_columns=protein_columns,
        protein_axis=protein_axis,
    )


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())


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


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _campaign_runtime() -> dict[str, Any]:
    configuration = np.show_config(mode="dicts")
    blas = configuration.get("Build Dependencies", {}).get("blas", {})
    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "scipy": scipy.__version__,
        "anndata": None,
        "anndata_used": False,
        "blas": {
            "name": blas.get("name"),
            "version": blas.get("version"),
            "found": blas.get("found"),
            "detection_method": blas.get("detection method"),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "thread_environment": {
            key: os.environ.get(key) for key in REQUIRED_THREAD_ENVIRONMENT
        },
    }


def _campaign_implementation_hashes() -> dict[str, str]:
    return {
        relative: _sha256(ROOT / relative) for relative in CAMPAIGN_IMPLEMENTATION_FILES
    }


def _contains_pending(value: Any) -> bool:
    if isinstance(value, str):
        return "PENDING" in value.upper()
    if isinstance(value, dict):
        return any(_contains_pending(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_pending(item) for item in value)
    return False


def _freeze_node(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PermissionError(f"source authorization lacks {name}")
    if (
        not isinstance(value.get("tag"), str)
        or not value["tag"]
        or not isinstance(value.get("tag_object"), str)
        or len(value["tag_object"]) != 40
        or not isinstance(value.get("peeled_commit"), str)
        or len(value["peeled_commit"]) != 40
        or value.get("remote_tag_and_commit_match") is not True
    ):
        raise PermissionError(f"source authorization has an invalid {name}")
    return value


def _git_output(*arguments: str, text: bool = True) -> str | bytes:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=text,
    )
    return completed.stdout


def _verified_tag(tag: str) -> dict[str, str]:
    try:
        origin = str(_git_output("config", "--get", "remote.origin.url")).strip()
        if origin != PUBLIC_ORIGIN_URL:
            raise PermissionError("origin is not the frozen public repository")
        tag_object = str(_git_output("rev-parse", f"refs/tags/{tag}^{{tag}}")).strip()
        peeled_commit = str(_git_output("rev-parse", f"{tag}^{{}}")).strip()
        remote = str(
            _git_output(
                "ls-remote",
                "--tags",
                "origin",
                f"refs/tags/{tag}",
                f"refs/tags/{tag}^{{}}",
            )
        )
    except subprocess.CalledProcessError as error:
        raise PermissionError(f"cannot verify public annotated tag {tag}") from error
    observed: dict[str, str] = {}
    for line in remote.splitlines():
        fields = line.split()
        if len(fields) == 2:
            observed[fields[1]] = fields[0]
    if (
        observed.get(f"refs/tags/{tag}") != tag_object
        or observed.get(f"refs/tags/{tag}^{{}}") != peeled_commit
    ):
        raise PermissionError(f"local and public tag bytes differ for {tag}")
    return {
        "tag": tag,
        "tag_object": tag_object,
        "peeled_commit": peeled_commit,
        "remote_tag_and_commit_match": True,
    }


def _published_file_sha256(tag: str, relative: str) -> str:
    try:
        payload = _git_output("show", f"{tag}:{relative}", text=False)
    except subprocess.CalledProcessError as error:
        raise PermissionError(f"public tag {tag} lacks {relative}") from error
    if not isinstance(payload, bytes):
        raise TypeError("git blob verification did not return bytes")
    return hashlib.sha256(payload).hexdigest()


def _require_ancestor(first: str, second: str) -> None:
    completed = subprocess.run(
        ("git", "merge-base", "--is-ancestor", first, second),
        cwd=ROOT,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise PermissionError("source public-freeze ancestry is invalid")


def _validate_public_freeze_chain(
    authorization: dict[str, Any], authorization_path: Path
) -> None:
    expected_nodes = (
        ("candidate_freeze", CANDIDATE_TAG),
        ("implementation_freeze", IMPLEMENTATION_TAG),
        ("axis_freeze", AXIS_PREFLIGHT_TAG),
    )
    snapshots = []
    for key, tag in expected_nodes:
        node = _freeze_node(authorization.get(key), key.replace("_", " "))
        snapshot = _verified_tag(tag)
        if any(node.get(field) != snapshot[field] for field in snapshot):
            raise PermissionError(f"source authorization has the wrong {key} tag")
        snapshots.append(snapshot)
    authorization_snapshot = _verified_tag(SOURCE_AUTHORIZATION_TAG)
    lineage = [
        *(snapshot["peeled_commit"] for snapshot in snapshots),
        authorization_snapshot["peeled_commit"],
    ]
    for first, second in zip(lineage, lineage[1:]):
        _require_ancestor(first, second)
    candidate = authorization["candidate_freeze"]
    if (
        candidate.get("candidate_path") != CANDIDATE_PATH
        or candidate.get("candidate_sha256")
        != _published_file_sha256(CANDIDATE_TAG, CANDIDATE_PATH)
        or candidate.get("candidate_sha256") != _sha256(ROOT / CANDIDATE_PATH)
    ):
        raise PermissionError("candidate tag does not bind the exact candidate bytes")
    implementation = authorization["implementation_freeze"]
    files = implementation.get("files_sha256")
    if not isinstance(files, dict) or set(files) != set(CAMPAIGN_IMPLEMENTATION_FILES):
        raise PermissionError("implementation tag has the wrong exact file set")
    for relative, expected in files.items():
        if _published_file_sha256(IMPLEMENTATION_TAG, relative) != expected:
            raise PermissionError(
                f"implementation tag does not bind exact bytes: {relative}"
            )
    axis = authorization["axis_freeze"]
    if _published_file_sha256(AXIS_PREFLIGHT_TAG, axis["preflight_path"]) != axis.get(
        "preflight_sha256"
    ):
        raise PermissionError("axis tag does not bind the exact v2 preflight bytes")
    try:
        relative = authorization_path.resolve().relative_to(ROOT.resolve())
    except ValueError as error:
        raise PermissionError(
            "source authorization path is outside the repository"
        ) from error
    if _published_file_sha256(SOURCE_AUTHORIZATION_TAG, relative.as_posix()) != _sha256(
        authorization_path
    ):
        raise PermissionError("public authorization bytes differ from the local file")


def _validate_source_authorization(
    authorization_path: Path,
    preflight_path: Path,
    output_path: Path,
    manifest_path: Path,
    model_output_path: Path,
    model_terminal_path: Path,
    reduction_terminal_path: Path,
) -> dict[str, Any]:
    if (
        authorization_path.resolve() != DEFAULT_SOURCE_AUTHORIZATION.resolve()
        or preflight_path.resolve() != DEFAULT_PREFLIGHT.resolve()
    ):
        raise PermissionError("source authorization or axis-v2 path is not canonical")
    authorization = _read_json(authorization_path)
    if (
        authorization.get("schema") != SOURCE_AUTHORIZATION_SCHEMA
        or authorization.get("status") != SOURCE_AUTHORIZATION_STATUS
        or authorization.get("accession") != "GSE181897"
        or authorization.get("stage") != "source_development"
        or _contains_pending(authorization)
    ):
        raise PermissionError("source campaign authorization is not fully frozen")
    candidate = _freeze_node(authorization.get("candidate_freeze"), "candidate freeze")
    axis = _freeze_node(authorization.get("axis_freeze"), "axis freeze")
    implementation = _freeze_node(
        authorization.get("implementation_freeze"), "implementation freeze"
    )
    if (
        candidate.get("tag") != CANDIDATE_TAG
        or implementation.get("tag") != IMPLEMENTATION_TAG
        or axis.get("tag") != AXIS_PREFLIGHT_TAG
    ):
        raise PermissionError("source authorization uses an unexpected freeze tag")
    preflight = _read_json(preflight_path)
    if (
        axis.get("preflight_path") != _display_path(preflight_path)
        or axis.get("preflight_sha256") != _sha256(preflight_path)
        or axis.get("axis_reducer_sha256") != preflight.get("reducer_sha256")
    ):
        raise PermissionError("source authorization does not bind the axis artifact")
    if implementation.get("files_sha256") != _campaign_implementation_hashes():
        raise PermissionError(
            "source authorization does not bind the current implementation"
        )
    if axis.get("axis_reducer_sha256") != implementation["files_sha256"].get(
        "experiments/reduce_gse181897_source.py"
    ):
        raise PermissionError(
            "axis-v2 preflight and implementation use different reducer bytes"
        )
    expected_input = {
        "archive_bytes": SOURCE_ARCHIVE_BYTES,
        "archive_sha256": SOURCE_ARCHIVE_SHA256,
        "h5ad_bytes": SOURCE_H5AD_BYTES,
        "h5ad_sha256": SOURCE_H5AD_SHA256,
    }
    if authorization.get("source_input") != expected_input:
        raise PermissionError("source authorization has the wrong input binding")
    if authorization.get("attempt_path") != _display_path(DEFAULT_SOURCE_ATTEMPT):
        raise PermissionError("source authorization has the wrong attempt path")
    expected_outputs = {
        "source_reduction": _display_path(output_path),
        "source_reduction_manifest": _display_path(manifest_path),
        "source_model": _display_path(model_output_path),
        "source_model_terminal": _display_path(model_terminal_path),
        "source_reduction_terminal": _display_path(reduction_terminal_path),
    }
    if authorization.get("outputs") != expected_outputs:
        raise PermissionError("source authorization has the wrong output binding")
    if authorization.get("runtime") != _campaign_runtime():
        raise PermissionError("source authorization has the wrong runtime binding")
    one_shot = authorization.get("one_shot_policy")
    if (
        not isinstance(one_shot, dict)
        or one_shot.get("exclusive_attempt_before_numeric_x") is not True
        or one_shot.get("rerun_after_claim_forbidden") is not True
        or one_shot.get("failure_is_terminal") is not True
        or one_shot.get("model_must_bind_same_attempt") is not True
    ):
        raise PermissionError("source authorization lacks the one-shot policy")
    _validate_public_freeze_chain(authorization, authorization_path)
    return authorization


def claim_source_campaign(
    authorization_path: Path,
    attempt_path: Path,
    preflight_path: Path,
    output_path: Path,
    manifest_path: Path,
    model_output_path: Path,
    model_terminal_path: Path,
    reduction_terminal_path: Path,
) -> dict[str, Any]:
    if attempt_path.resolve() != DEFAULT_SOURCE_ATTEMPT.resolve():
        raise PermissionError("source campaign attempt path is not canonical")
    for path in (
        output_path,
        manifest_path,
        model_output_path,
        model_terminal_path,
        reduction_terminal_path,
    ):
        if path.exists():
            raise FileExistsError("a source campaign output already exists")
    authorization = _validate_source_authorization(
        authorization_path,
        preflight_path,
        output_path,
        manifest_path,
        model_output_path,
        model_terminal_path,
        reduction_terminal_path,
    )
    payload = {
        "schema": SOURCE_ATTEMPT_SCHEMA,
        "status": SOURCE_ATTEMPT_STATUS,
        "accession": "GSE181897",
        "stage": "source_development",
        "created_at_utc": _timestamp(),
        "one_shot": True,
        "attempt_consumed_even_on_interruption": True,
        "authorization": {
            "path": _display_path(authorization_path),
            "sha256": _sha256(authorization_path),
        },
        "binding": authorization,
    }
    _write_json_exclusive(attempt_path, payload)
    return payload


def validate_source_campaign_attempt(
    attempt_path: Path,
    preflight_path: Path,
    output_path: Path,
    manifest_path: Path,
    model_output_path: Path,
    model_terminal_path: Path,
    reduction_terminal_path: Path,
) -> tuple[dict[str, Any], str]:
    if attempt_path.resolve() != DEFAULT_SOURCE_ATTEMPT.resolve():
        raise PermissionError("source campaign attempt path is not canonical")
    claim_sha256 = _sha256(attempt_path)
    claim = _read_json(attempt_path)
    authorization_record = claim.get("authorization")
    if (
        claim.get("schema") != SOURCE_ATTEMPT_SCHEMA
        or claim.get("status") != SOURCE_ATTEMPT_STATUS
        or claim.get("accession") != "GSE181897"
        or claim.get("stage") != "source_development"
        or claim.get("one_shot") is not True
        or claim.get("attempt_consumed_even_on_interruption") is not True
        or not isinstance(authorization_record, dict)
        or not isinstance(authorization_record.get("path"), str)
        or not isinstance(authorization_record.get("sha256"), str)
    ):
        raise PermissionError("source campaign claim is absent or malformed")
    authorization_path = ROOT / authorization_record["path"]
    if (
        not authorization_path.is_file()
        or _sha256(authorization_path) != authorization_record["sha256"]
    ):
        raise PermissionError("source campaign authorization changed after claim")
    authorization = _validate_source_authorization(
        authorization_path,
        preflight_path,
        output_path,
        manifest_path,
        model_output_path,
        model_terminal_path,
        reduction_terminal_path,
    )
    if claim.get("binding") != authorization:
        raise PermissionError("source campaign claim differs from authorization")
    if _sha256(attempt_path) != claim_sha256:
        raise PermissionError("source campaign claim changed during validation")
    return claim, claim_sha256


def _available_bytes(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def _download_archive(path: Path) -> None:
    if path.is_file():
        if path.is_symlink() or path.stat().st_size != SOURCE_ARCHIVE_BYTES:
            raise ValueError("cached source archive has the wrong file contract")
        if _sha256(path) != SOURCE_ARCHIVE_SHA256:
            raise ValueError("cached source archive has the wrong SHA-256")
        return
    if (
        _available_bytes(path.parent)
        < SOURCE_ARCHIVE_BYTES + MINIMUM_FREE_BYTES_AFTER_ACQUISITION
    ):
        raise OSError("insufficient free space for the source archive")
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        with (
            urllib.request.urlopen(SOURCE_URL, timeout=120) as response,
            temporary.open("wb") as stream,
        ):
            while block := response.read(8 << 20):
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.stat().st_size != SOURCE_ARCHIVE_BYTES:
            raise ValueError("downloaded source archive has the wrong byte count")
        if _sha256(temporary) != SOURCE_ARCHIVE_SHA256:
            raise ValueError("downloaded source archive has the wrong SHA-256")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _decompress_archive(archive: Path, h5ad_path: Path) -> None:
    if (
        _available_bytes(h5ad_path.parent)
        < SOURCE_H5AD_BYTES + MINIMUM_FREE_BYTES_AFTER_ACQUISITION
    ):
        raise OSError("insufficient free space to decompress the source H5AD")
    temporary = h5ad_path.with_suffix(h5ad_path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        with gzip.open(archive, "rb") as source, temporary.open("wb") as target:
            while block := source.read(8 << 20):
                target.write(block)
            target.flush()
            os.fsync(target.fileno())
        if temporary.stat().st_size != SOURCE_H5AD_BYTES:
            raise ValueError("decompressed source H5AD has the wrong byte count")
        if _sha256(temporary) != SOURCE_H5AD_SHA256:
            raise ValueError("decompressed source H5AD has the wrong SHA-256")
        temporary.replace(h5ad_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def acquire_h5ad(
    cache: Path, *, keep_archive: bool = False
) -> tuple[Path, dict[str, bool]]:
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / SOURCE_ARCHIVE_NAME
    h5ad_path = cache / SOURCE_H5AD_NAME
    archive_present_at_start = archive.exists()
    archive_verified = False
    if archive.exists():
        _download_archive(archive)
        archive_verified = True
    if h5ad_path.is_file():
        if h5ad_path.is_symlink() or h5ad_path.stat().st_size != SOURCE_H5AD_BYTES:
            raise ValueError("cached source H5AD has the wrong file contract")
        if _sha256(h5ad_path) != SOURCE_H5AD_SHA256:
            raise ValueError("cached source H5AD has the wrong SHA-256")
    else:
        if not archive_verified:
            _download_archive(archive)
            archive_verified = True
        _decompress_archive(archive, h5ad_path)
    archive_removed = False
    if archive.exists() and not keep_archive:
        if archive.is_symlink() or archive.stat().st_size != SOURCE_ARCHIVE_BYTES:
            raise ValueError("refusing to remove an unverified source archive path")
        if _sha256(archive) != SOURCE_ARCHIVE_SHA256:
            raise ValueError("refusing to remove an unverified source archive")
        archive.unlink()
        archive_removed = True
    return h5ad_path, {
        "archive_present_at_start": archive_present_at_start,
        "archive_verified_in_this_run": archive_verified,
        "h5ad_verified_in_this_run": True,
        "archive_removed_after_verification": archive_removed,
    }


def _read_authorized_csr_columns(
    matrix: h5py.Group,
    plan: SourcePlan,
    columns: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    _matrix_metadata(matrix)
    if len(columns) != len(set(columns)) or any(
        column < 0 or column >= EXPECTED_X_SHAPE[1] for column in columns
    ):
        raise ValueError("requested numeric feature columns are invalid")
    rows = plan.selected_rows.ravel()
    if rows.size != len(plan.donor_axis) * CELL_BUDGET:
        raise ValueError("numeric row plan has the wrong size")
    if np.any(rows < 0) or np.any(rows >= EXPECTED_X_SHAPE[0]):
        raise ValueError("numeric row plan is out of bounds")
    if np.any(~plan.authorized_rows[rows]):
        raise PermissionError("numeric row plan includes an unauthorized cell")

    indptr = matrix["indptr"]
    endpoints = np.unique(np.concatenate((rows, rows + 1))).astype(np.int64)
    pointer_values = np.asarray(indptr[endpoints], dtype=np.int64)
    pointers = dict(zip(endpoints.tolist(), pointer_values.tolist()))
    lookup = np.full(EXPECTED_X_SHAPE[1], -1, dtype=np.int32)
    lookup[np.asarray(columns, dtype=np.int64)] = np.arange(
        len(columns), dtype=np.int32
    )
    output = np.zeros((len(rows), len(columns)), dtype=np.int32)
    decoded_values = 0
    scanned_indices = 0
    indices_dataset = matrix["indices"]
    data_dataset = matrix["data"]
    for output_row, source_row in enumerate(rows):
        start = pointers[int(source_row)]
        end = pointers[int(source_row + 1)]
        if start < 0 or end < start or end > EXPECTED_X_DATA_LENGTH:
            raise ValueError("selected CSR row has malformed pointers")
        indices = np.asarray(indices_dataset[start:end], dtype=np.int64)
        scanned_indices += len(indices)
        if (
            np.any(indices < 0)
            or np.any(indices >= EXPECTED_X_SHAPE[1])
            or len(np.unique(indices)) != len(indices)
        ):
            raise ValueError("selected CSR row has invalid or duplicate indices")
        target = lookup[indices]
        keep = np.flatnonzero(target >= 0)
        if not len(keep):
            continue
        values = np.asarray(data_dataset[start + keep], dtype=np.float64)
        if (
            not np.isfinite(values).all()
            or np.any(values < 0)
            or not np.array_equal(values, np.rint(values))
            or np.any(values > np.iinfo(np.int32).max)
        ):
            raise ValueError(
                "selected source entries are not nonnegative integer counts"
            )
        output[output_row, target[keep]] = values.astype(np.int32)
        decoded_values += len(values)
    return output, {
        "matrix_datasets_indexed": ["/X/indptr", "/X/indices", "/X/data"],
        "numeric_rows_decoded": len(rows),
        "numeric_row_axis_sha256": _array_sha256(plan.selected_rows),
        "numeric_feature_columns_decoded": len(columns),
        "numeric_feature_column_axis_sha256": _array_sha256(
            np.asarray(columns, dtype=np.int64)
        ),
        "stored_numeric_values_decoded": decoded_values,
        "csr_indptr_entries_decoded": len(endpoints),
        "csr_index_entries_scanned": scanned_indices,
        "requested_stored_data_entries_decoded": decoded_values,
        "unrequested_stored_data_entries_decoded": 0,
        "out_of_panel_index_positions_scanned": scanned_indices - decoded_values,
        "out_of_panel_indices_used_only_for_membership_filtering": True,
        "out_of_panel_featurewise_statistics_retained": 0,
        "out_of_panel_feature_signal_entering_model_outputs": 0,
        "implicit_zero_values_materialized": int(output.size - decoded_values),
        "held_batch_rows_decoded": 0,
        "non_control_rows_decoded": 0,
        "unselected_authorized_rows_decoded": 0,
    }


def _ordered_tables(rna: np.ndarray, adt: np.ndarray) -> np.ndarray:
    marker_count = len(PANEL)
    tables = np.empty((marker_count, marker_count, 2, 2), dtype=np.int16)
    for rna_index in range(marker_count):
        for adt_index in range(marker_count):
            code = 2 * rna[:, rna_index].astype(int) + adt[:, adt_index].astype(int)
            tables[rna_index, adt_index] = np.bincount(code, minlength=4).reshape(2, 2)
    return tables


def _availability_diagnostics(
    rna_counts: np.ndarray,
    adt_counts: np.ndarray,
    donor_axis: tuple[str, ...],
    batch_axis: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rna_states = rna_counts > 0
    adt_states = adt_counts > 0
    tables = np.asarray(
        [
            _ordered_tables(rna_states[index], adt_states[index])
            for index in range(len(donor_axis))
        ]
    )
    rna_positive = rna_states.sum(axis=1)
    adt_positive = adt_states.sum(axis=1)
    rna_support = (rna_positive >= MINIMUM_MARKER_POSITIVES) & (
        rna_positive <= MAXIMUM_MARKER_POSITIVES
    )
    adt_support = (adt_positive >= MINIMUM_MARKER_POSITIVES) & (
        adt_positive <= MAXIMUM_MARKER_POSITIVES
    )
    row_one = tables.sum(axis=-1)[..., 1]
    column_one = tables.sum(axis=-2)[..., 1]
    total = tables.sum(axis=(-2, -1))
    lower = np.maximum(0, row_one + column_one - total)
    upper = np.minimum(row_one, column_one)
    informative = lower < upper
    subject_support = rna_support[:, :, None] & adt_support[:, None, :] & informative
    donor_records = [
        {
            "exp_id": int(donor_axis[index]),
            "batch": batch_axis[index],
            "rna_positive_counts": rna_positive[index].astype(int).tolist(),
            "adt_positive_counts": adt_positive[index].astype(int).tolist(),
            "rna_supported_markers": int(np.count_nonzero(rna_support[index])),
            "adt_supported_markers": int(np.count_nonzero(adt_support[index])),
            "supported_coordinates": int(np.count_nonzero(subject_support[index])),
            "at_least_232_supported_coordinates": int(
                np.count_nonzero(subject_support[index])
            )
            >= MINIMUM_SOURCE_COORDINATES,
        }
        for index in range(len(donor_axis))
    ]

    def training_mask(indices: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        support_threshold = math.ceil(len(indices) / 2)
        support_counts = subject_support[indices].sum(axis=0)
        threshold_mask = support_counts >= support_threshold
        supported = subject_support[indices, :, :, None, None]
        pooled = np.sum(tables[indices] * supported, axis=0, dtype=np.int64)
        four_cell_positive = np.all(pooled > 0, axis=(-2, -1))
        observed_sum = np.sum(
            tables[indices, :, :, 1, 1] * subject_support[indices],
            axis=0,
            dtype=np.int64,
        )
        lower_sum = np.sum(
            lower[indices] * subject_support[indices], axis=0, dtype=np.int64
        )
        upper_sum = np.sum(
            upper[indices] * subject_support[indices], axis=0, dtype=np.int64
        )
        strict_interior = (observed_sum > lower_sum) & (observed_sum < upper_sum)
        mask = threshold_mask & four_cell_positive & strict_interior
        return mask, {
            "training_donors": len(indices),
            "minimum_training_donor_support": support_threshold,
            "coordinates_passing_support_count": int(np.count_nonzero(threshold_mask)),
            "coordinates_passing_pooled_four_cell_positivity": int(
                np.count_nonzero(four_cell_positive)
            ),
            "coordinates_passing_strict_fixed_margin_interior": int(
                np.count_nonzero(strict_interior)
            ),
            "mask_coordinates": int(np.count_nonzero(mask)),
            "mask_sha256": _array_sha256(mask.astype(np.uint8)),
            "mask_at_least_232_coordinates": int(np.count_nonzero(mask))
            >= MINIMUM_SOURCE_COORDINATES,
        }

    folds: list[dict[str, Any]] = []
    batches = np.asarray(batch_axis)
    for held_batch in SOURCE_BATCHES:
        training = np.flatnonzero(batches != held_batch)
        validation = np.flatnonzero(batches == held_batch)
        mask, certificate = training_mask(training)
        validation_support = {
            donor_axis[index]: int(np.count_nonzero(mask & subject_support[index]))
            for index in validation
        }
        folds.append(
            {
                "held_batch": held_batch,
                "validation_donors": [donor_axis[index] for index in validation],
                **certificate,
                "validation_supported_coordinates": validation_support,
                "every_validation_donor_at_least_232_coordinates": all(
                    count >= MINIMUM_SOURCE_COORDINATES
                    for count in validation_support.values()
                ),
            }
        )
    final_mask, final_certificate = training_mask(np.arange(len(donor_axis)))
    final_support = {
        donor_axis[index]: int(np.count_nonzero(final_mask & subject_support[index]))
        for index in range(len(donor_axis))
    }
    return (
        tables,
        final_mask,
        {
            "status": "SOURCE_SUPPORT_AND_TABLES_NO_MODEL_OR_LOSS_INSPECTED",
            "binary_state_rule": "count > 0 for both RNA and ADT",
            "marker_support_rule": (
                f"{MINIMUM_MARKER_POSITIVES} <= positive cells <= "
                f"{MAXIMUM_MARKER_POSITIVES} among {CELL_BUDGET} selected cells"
            ),
            "coordinate_support_rule": (
                "both cognate marker margins pass support and the donor's fixed "
                "margins admit more than one 2x2 table"
            ),
            "fold_support_rule": (
                "at least ceil(0.5 * training donors) support the coordinate and "
                "the pooled supported 2x2 table has all four cells positive and "
                "the summed observed n11 lies strictly inside the summed fixed-margin bounds"
            ),
            "minimum_coordinate_floor": MINIMUM_SOURCE_COORDINATES,
            "donors": donor_records,
            "leave_one_batch_out_folds": folds,
            "final_source_mask": {
                **final_certificate,
                "per_donor_supported_coordinates": final_support,
                "every_source_donor_at_least_232_coordinates": all(
                    count >= MINIMUM_SOURCE_COORDINATES
                    for count in final_support.values()
                ),
            },
        },
    )


def _adt_graph_profiles(
    all_protein_counts: np.ndarray,
    protein_axis: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    positions = [protein_axis.index(cognate.adt_feature) for cognate in PANEL]
    log_counts = np.log1p(all_protein_counts.astype(np.float64))
    cellwise_clr = log_counts - log_counts.mean(axis=2, keepdims=True)
    profiles = cellwise_clr.mean(axis=1)[:, positions]
    if (
        profiles.shape != (len(all_protein_counts), len(PANEL))
        or not np.isfinite(profiles).all()
    ):
        raise ValueError("ADT graph profiles are malformed")
    return profiles, {
        "cell_clr_formula": (
            f"log1p(count_{EXPECTED_PROTEIN_FEATURE_COUNT}) - "
            f"mean_j(log1p(count_{EXPECTED_PROTEIN_FEATURE_COUNT}))"
        ),
        "donor_profile_formula": (
            "mean across 128 selected cells, restricted to the 17 cognate positions"
        ),
        "denominator_rule": (
            f"all {EXPECTED_PROTEIN_FEATURE_COUNT} var/genome=BD99AbSeq features"
        ),
        "denominator_feature_count": EXPECTED_PROTEIN_FEATURE_COUNT,
        "denominator_feature_axis": list(protein_axis),
        "denominator_feature_axis_sha256": _axis_sha256(protein_axis),
        "cognate_positions_zero_based": positions,
        "cognate_feature_axis": [cognate.adt_feature for cognate in PANEL],
    }


def _write_npz_exclusive(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def write_preflight(
    h5ad_path: Path,
    preflight_path: Path,
    acquisition: dict[str, bool],
) -> dict[str, Any]:
    if not acquisition.get("archive_verified_in_this_run"):
        raise PermissionError(
            "axis preflight requires compressed-archive verification in this run"
        )
    inspection = inspect_axes(h5ad_path)
    inspection.payload["source"]["acquisition"] = acquisition
    _write_json_exclusive(preflight_path, inspection.payload)
    return inspection.payload


def _without_acquisition(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(payload, allow_nan=False))
    source = normalized.get("source")
    if isinstance(source, dict):
        source.pop("acquisition", None)
    return normalized


def _axis_contract_without_code_hash(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _without_acquisition(payload)
    normalized.pop("reducer_sha256", None)
    return normalized


def reduce_source(
    h5ad_path: Path,
    preflight_path: Path,
    output_path: Path,
    manifest_path: Path,
    attempt_path: Path,
    model_output_path: Path,
    model_terminal_path: Path,
    reduction_terminal_path: Path,
) -> dict[str, Any]:
    _, attempt_sha256 = validate_source_campaign_attempt(
        attempt_path,
        preflight_path,
        output_path,
        manifest_path,
        model_output_path,
        model_terminal_path,
        reduction_terminal_path,
    )
    inspection = inspect_axes(h5ad_path)
    frozen_preflight = _read_json(preflight_path)
    if _axis_contract_without_code_hash(
        frozen_preflight
    ) != _axis_contract_without_code_hash(inspection.payload):
        raise PermissionError("axis preflight differs; numeric access is refused")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("source reduction outputs are write-once")

    selected_columns = inspection.rna_columns + inspection.protein_columns
    with h5py.File(h5ad_path, "r") as handle:
        values, access_audit = _read_authorized_csr_columns(
            handle["X"], inspection.plan, selected_columns
        )
    if _sha256(h5ad_path) != SOURCE_H5AD_SHA256:
        raise PermissionError("source H5AD changed during numeric reduction")

    donor_count = len(inspection.plan.donor_axis)
    values = values.reshape(donor_count, CELL_BUDGET, len(selected_columns))
    rna_counts = values[:, :, : len(PANEL)]
    all_protein_counts = values[:, :, len(PANEL) :]
    protein_positions = [
        inspection.protein_axis.index(cognate.adt_feature) for cognate in PANEL
    ]
    adt_counts = all_protein_counts[:, :, protein_positions]
    adt_graph_profile, graph_specification = _adt_graph_profiles(
        all_protein_counts, inspection.protein_axis
    )
    tables, source_comparison_mask, availability = _availability_diagnostics(
        rna_counts,
        adt_counts,
        inspection.plan.donor_axis,
        inspection.plan.batch_axis,
    )
    arrays = {
        "donor_axis": np.asarray(inspection.plan.donor_axis, dtype=np.int16),
        "free_id_axis": np.asarray(inspection.plan.free_id_axis, dtype=np.int16),
        "batch_axis": np.asarray(inspection.plan.batch_axis, dtype=np.int8),
        "condition_axis": np.asarray([CONTROL_CONDITION] * donor_count),
        "rna_gene_axis": np.asarray([cognate.rna_gene for cognate in PANEL]),
        "rna_feature_id_axis": np.asarray(
            [cognate.rna_feature_id for cognate in PANEL]
        ),
        "adt_protein_axis": np.asarray([cognate.protein_label for cognate in PANEL]),
        "adt_feature_axis": np.asarray([cognate.adt_feature for cognate in PANEL]),
        "protein_denominator_axis": np.asarray(inspection.protein_axis),
        "coordinate_axis": np.asarray(
            [f"{rna.rna_gene}|{adt.adt_feature}" for rna in PANEL for adt in PANEL]
        ),
        "selected_barcodes": inspection.plan.selected_barcodes,
        "selected_row_indices": inspection.plan.selected_rows,
        "rna_counts": rna_counts.astype(np.int32, copy=False),
        "adt_counts": adt_counts.astype(np.int32, copy=False),
        "adt_graph_profile": adt_graph_profile.astype(np.float64, copy=False),
        "tables": tables.reshape(donor_count, len(PANEL) ** 2, 4),
        "source_comparison_mask": source_comparison_mask.reshape(-1).astype(np.uint8),
    }
    _write_npz_exclusive(output_path, arrays)
    manifest = {
        "schema": "gse181897-source-reduction/1.0",
        "status": "SOURCE_REDUCTION_COMPLETE",
        "accession": "GSE181897",
        "stage": "source_development",
        "numeric_batches_processed": list(SOURCE_BATCHES),
        "numeric_condition_processed": CONTROL_CONDITION,
        "held_batches_unopened": list(HELD_BATCHES),
        "non_control_rows_unopened": True,
        "condition_0_mispool_rows_unopened": EXPECTED_CONDITION_ZERO_CELLS,
        "excluded_development_exp_ids": sorted(
            int(donor) for donor in EXCLUDED_DEVELOPMENT_DONORS
        ),
        "donor_count": donor_count,
        "cell_budget_per_donor": CELL_BUDGET,
        "panel": inspection.payload["panel"],
        "adt_graph_profile": graph_specification,
        "cell_selection": inspection.payload["source_plan"],
        "source": inspection.payload["source"],
        "axis_preflight": {
            "path": str(preflight_path.resolve().relative_to(ROOT.resolve())),
            "sha256": _sha256(preflight_path),
            "schema": frozen_preflight["schema"],
            "status": frozen_preflight["status"],
            "axis_reducer_sha256": frozen_preflight["reducer_sha256"],
        },
        "source_campaign_attempt": {
            "path": _display_path(attempt_path),
            "sha256": attempt_sha256,
            "schema": SOURCE_ATTEMPT_SCHEMA,
            "status": SOURCE_ATTEMPT_STATUS,
        },
        "numeric_access": access_audit,
        "availability_diagnostics": availability,
        "output": {
            "path": str(output_path.resolve().relative_to(ROOT.resolve())),
            "bytes": output_path.stat().st_size,
            "sha256": _sha256(output_path),
            "members": sorted(arrays),
        },
        "reducer_sha256": _sha256(Path(__file__)),
    }
    try:
        _write_json_exclusive(manifest_path, manifest)
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    return manifest


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "claim", "reduce"))
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--authorization", type=Path, default=DEFAULT_SOURCE_AUTHORIZATION
    )
    parser.add_argument("--attempt", type=Path, default=DEFAULT_SOURCE_ATTEMPT)
    parser.add_argument(
        "--reduction-terminal", type=Path, default=DEFAULT_REDUCTION_TERMINAL
    )
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--model-terminal", type=Path, default=DEFAULT_MODEL_TERMINAL)
    parser.add_argument("--keep-archive", action="store_true")
    return parser


def _replace_open_json(stream: Any, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    stream.seek(0)
    stream.truncate()
    stream.write(encoded)
    stream.flush()
    os.fsync(stream.fileno())


def run_reduction_one_shot(args: argparse.Namespace) -> dict[str, Any]:
    args.reduction_terminal.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        args.reduction_terminal, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
    )
    with os.fdopen(descriptor, "w+") as stream:
        attempt_sha256 = _sha256(args.attempt) if args.attempt.is_file() else None
        started = {
            "schema": "gse181897-source-reduction-terminal/1.0",
            "status": "SOURCE_REDUCTION_EXECUTION_STARTED_ATTEMPT_CONSUMED",
            "created_at_utc": _timestamp(),
            "attempt_path": _display_path(args.attempt),
            "attempt_sha256": attempt_sha256,
        }
        _replace_open_json(stream, started)
        try:
            validate_source_campaign_attempt(
                args.attempt,
                args.preflight,
                args.output,
                args.manifest,
                args.model_output,
                args.model_terminal,
                args.reduction_terminal,
            )
            h5ad_path, _ = acquire_h5ad(args.cache, keep_archive=args.keep_archive)
            manifest = reduce_source(
                h5ad_path,
                args.preflight,
                args.output,
                args.manifest,
                args.attempt,
                args.model_output,
                args.model_terminal,
                args.reduction_terminal,
            )
            payload = {
                "schema": "gse181897-source-reduction-terminal/1.0",
                "status": "SOURCE_REDUCTION_COMPLETE_MODEL_PENDING",
                "created_at_utc": _timestamp(),
                "attempt_path": _display_path(args.attempt),
                "attempt_sha256": attempt_sha256,
                "source_reduction_path": _display_path(args.output),
                "source_reduction_sha256": manifest["output"]["sha256"],
                "source_manifest_path": _display_path(args.manifest),
                "source_manifest_sha256": _sha256(args.manifest),
                "model_output_path": _display_path(args.model_output),
                "model_terminal_path": _display_path(args.model_terminal),
                "model_numeric_access_authorized": False,
                "next_step": "run the write-once source model bound to this claim",
            }
        except BaseException as error:
            payload = {
                "schema": "gse181897-source-reduction-terminal/1.0",
                "status": "TERMINAL_SOURCE_REDUCTION_REFUSAL",
                "created_at_utc": _timestamp(),
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
    args = _argument_parser().parse_args()
    if args.command == "preflight":
        h5ad_path, acquisition = acquire_h5ad(
            args.cache, keep_archive=args.keep_archive
        )
        payload = write_preflight(h5ad_path, args.preflight, acquisition)
    elif args.command == "claim":
        payload = claim_source_campaign(
            args.authorization,
            args.attempt,
            args.preflight,
            args.output,
            args.manifest,
            args.model_output,
            args.model_terminal,
            args.reduction_terminal,
        )
    else:
        payload = run_reduction_one_shot(args)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
