"""Outcome-sealed reducer for the GSE299043 mesenteric-LN CITE-seq panel.

The GEO H5ADs are the authors' aligned, unnormalized 10x matrices.  This
module reads only the frozen RNA, ADT, and donor-HTO columns, reproduces the
authors' HashSolo call, and emits small per-library JSON pieces.  Those pieces
can be pooled after each source H5AD has been deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm


MARKERS = ("CD4", "CD7", "CD14", "CD19", "CD33", "CD38", "CD44", "CD47", "CD52")
RNA_FEATURE_IDS = (
    "ENSG00000010610",
    "ENSG00000173762",
    "ENSG00000170458",
    "ENSG00000177455",
    "ENSG00000105383",
    "ENSG00000004468",
    "ENSG00000026508",
    "ENSG00000196776",
    "ENSG00000169442",
)
ADT_FEATURE_IDS = (
    "C0072",
    "C0066",
    "C0081",
    "C0050",
    "C0052",
    "C0389",
    "C0073",
    "C0026",
    "C0033",
)

DEVELOPMENT_DONORS = (
    "591C",
    "621B",
    "637C",
    "640C",
    "647C",
    "689C",
    "694B",
    "759B",
    "768B",
    "778C",
)
HELD_DONORS = (
    "D512",
    "D520",
    "D523",
    "D528",
    "D529",
    "D533",
    "D534",
    "D543",
    "D564",
    "D570",
)
MLN_TAGS = {
    "591C": ("591C-MLN-1", "591C-MLN-2", "591C-MLN-3"),
    "621B": ("621B-MLN-87",),
    "637C": ("637C-MLN-105", "637C-MLN-115"),
    "640C": ("640C-MLN-123", "640C-MLN-132"),
    "647C": ("647C-MLN-140",),
    "689C": ("689C-MLN-181", "689C-MLN-189"),
    "694B": ("694B-MLN-206",),
    "759B": ("759B-MLN-263",),
    "768B": ("768B-MLN-274", "768B-MLN-282"),
    "778C": ("778C-MLN-297", "778C-MLN-305"),
    **{donor: (f"{donor}-MLN-1",) for donor in HELD_DONORS},
}
PATCHED_759B_LIBRARIES = {
    "CZI-IA12953908",
    "CZI-IA12953909",
    "CZI-IA12953910",
    "CZI-IA12953911",
}
SINGLE_TISSUE_ONE_HTO_DONOR = "694B"
SINGLE_TISSUE_ONE_HTO_FILENAME = "GSE299043_694B_001.CZI-IA11512689.v2.h5ad"
SINGLE_TISSUE_ONE_HTO_TAG = "694B-MLN-206"

HASH_SOLO_PRIORS = (0.05, 0.70, 0.25)
CELL_BUDGET = 512
CELL_SELECTION_SALT = "GSE299043-MLN-CELL-BUDGET-v1"
ADT_TIE_SALT = "GSE299043-MLN-ADT-v1"
FILENAME_PATTERN = re.compile(
    r"^GSE299043_(?P<donor>[A-Z0-9]+)_(?P<run>[0-9]{3})\."
    r"(?P<library>(?:CZI-IA|CZINY-)[0-9]+)\.v[0-9]+\.h5ad$"
)


@dataclass(frozen=True)
class HeldAccessPermit:
    """Bindings already validated by the commit-aware held-score runner."""

    prediction_sha256: str
    public_commit: str
    authorization_sha256: str
    terminal_attempt_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(serialized)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"{path.name} contains nonfinite JSON token {token}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _role(donor: str) -> str:
    if donor in DEVELOPMENT_DONORS:
        return "development"
    if donor in HELD_DONORS:
        return "held"
    raise ValueError(f"donor {donor!r} is not in the frozen split")


def _authorize_access(donor: str, phase: str, permit: HeldAccessPermit | None) -> str:
    role = _role(donor)
    if role == "development":
        if phase != "development" or permit is not None:
            raise PermissionError("development access must use the unpaired phase")
        return role
    if phase != "held_score_authorized" or permit is None:
        raise PermissionError(
            "held H5AD access requires a commit-bound score authorization"
        )
    bindings = (
        permit.prediction_sha256,
        permit.authorization_sha256,
        permit.terminal_attempt_sha256,
    )
    if not all(re.fullmatch(r"[0-9a-f]{64}", value) for value in bindings):
        raise PermissionError("held authorization contains a malformed SHA-256")
    if not re.fullmatch(r"[0-9a-f]{40}", permit.public_commit):
        raise PermissionError("held authorization contains a malformed commit")
    return role


def _decode(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind in {"S", "O", "U"}:
        return np.asarray(
            [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in array
            ],
            dtype=str,
        )
    return array


def _attribute_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _dataframe_index(group: h5py.Group) -> np.ndarray:
    key = _attribute_text(group.attrs.get("_index", "_index"))
    if key not in group or not isinstance(group[key], h5py.Dataset):
        raise ValueError("H5AD data frame lacks its index dataset")
    values = _decode(group[key][...])
    if values.ndim != 1 or values.dtype.kind != "U":
        raise ValueError("H5AD data-frame index is not a string axis")
    return values


def _dataframe_column(group: h5py.Group, key: str) -> np.ndarray:
    if key not in group:
        raise ValueError(f"H5AD var lacks frozen column {key!r}")
    item = group[key]
    if isinstance(item, h5py.Dataset):
        values = item[...]
    elif isinstance(item, h5py.Group) and {"codes", "categories"}.issubset(item):
        codes = np.asarray(item["codes"][...], dtype=np.int64)
        categories = _decode(item["categories"][...])
        if np.any(codes < 0) or np.any(codes >= len(categories)):
            raise ValueError(f"H5AD categorical column {key!r} has invalid codes")
        values = categories[codes]
    else:
        raise ValueError(f"H5AD var column {key!r} has an unsupported encoding")
    values = _decode(values)
    if values.ndim != 1:
        raise ValueError(f"H5AD var column {key!r} is not one-dimensional")
    return values


def _matrix_shape(matrix: h5py.Dataset | h5py.Group) -> tuple[int, int]:
    if isinstance(matrix, h5py.Dataset):
        if matrix.ndim != 2:
            raise ValueError("H5AD X is not two-dimensional")
        return int(matrix.shape[0]), int(matrix.shape[1])
    shape = matrix.attrs.get("shape")
    if shape is None:
        raise ValueError("sparse H5AD X lacks its shape attribute")
    parsed = tuple(int(value) for value in shape)
    if len(parsed) != 2:
        raise ValueError("sparse H5AD X shape is malformed")
    return parsed


def _matrix_columns(
    matrix: h5py.Dataset | h5py.Group,
    columns: list[int],
    expected_shape: tuple[int, int],
) -> np.ndarray:
    if len(columns) != len(set(columns)):
        raise ValueError("selected H5AD columns are not unique")
    if any(column < 0 or column >= expected_shape[1] for column in columns):
        raise ValueError("selected H5AD column is out of bounds")
    if _matrix_shape(matrix) != expected_shape:
        raise ValueError("H5AD X shape differs from the obs/var axes")
    if isinstance(matrix, h5py.Dataset):
        order = np.argsort(columns)
        sorted_columns = np.asarray(columns, dtype=np.int64)[order]
        sorted_values = np.asarray(matrix[:, sorted_columns])
        inverse = np.argsort(order)
        output = sorted_values[:, inverse]
    else:
        encoding = _attribute_text(matrix.attrs.get("encoding-type", ""))
        required = {"data", "indices", "indptr"}
        if encoding not in {"csr_matrix", "csc_matrix"} or not required.issubset(
            matrix
        ):
            raise ValueError("H5AD X is not a supported dense/CSR/CSC matrix")
        data = matrix["data"]
        indices = matrix["indices"]
        indptr = np.asarray(matrix["indptr"][...], dtype=np.int64)
        major = expected_shape[0] if encoding == "csr_matrix" else expected_shape[1]
        if (
            data.ndim != 1
            or indices.ndim != 1
            or len(data) != len(indices)
            or len(indptr) != major + 1
            or indptr[0] != 0
            or indptr[-1] != len(data)
            or np.any(np.diff(indptr) < 0)
        ):
            raise ValueError("H5AD sparse X structural arrays are invalid")
        output = np.zeros((expected_shape[0], len(columns)), dtype=np.float64)
        if encoding == "csr_matrix":
            lookup = np.full(expected_shape[1], -1, dtype=np.int64)
            lookup[columns] = np.arange(len(columns), dtype=np.int64)
            for first_row in range(0, expected_shape[0], 2048):
                last_row = min(first_row + 2048, expected_shape[0])
                start = int(indptr[first_row])
                end = int(indptr[last_row])
                source_columns = np.asarray(indices[start:end], dtype=np.int64)
                if np.any(source_columns < 0) or np.any(
                    source_columns >= expected_shape[1]
                ):
                    raise ValueError("H5AD CSR column index is out of bounds")
                target_columns = lookup[source_columns]
                keep = target_columns >= 0
                if not np.any(keep):
                    continue
                row_counts = np.diff(indptr[first_row : last_row + 1])
                source_rows = np.repeat(
                    np.arange(first_row, last_row, dtype=np.int64), row_counts
                )
                np.add.at(
                    output,
                    (source_rows[keep], target_columns[keep]),
                    np.asarray(data[start:end])[keep],
                )
        else:
            for target_column, source_column in enumerate(columns):
                start = int(indptr[source_column])
                end = int(indptr[source_column + 1])
                rows = np.asarray(indices[start:end], dtype=np.int64)
                if np.any(rows < 0) or np.any(rows >= expected_shape[0]):
                    raise ValueError("H5AD CSC row index is out of bounds")
                np.add.at(output[:, target_column], rows, np.asarray(data[start:end]))
    if (
        not np.isfinite(output).all()
        or np.any(output < 0)
        or not np.array_equal(output, np.rint(output))
    ):
        raise ValueError("H5AD X contains nonfinite, negative, or nonintegral counts")
    return output.astype(np.int64)


def _feature_columns(
    names: np.ndarray, gene_ids: np.ndarray, feature_types: np.ndarray, donor: str
) -> tuple[list[int], list[int], list[int], list[str]]:
    if not (len(names) == len(gene_ids) == len(feature_types)):
        raise ValueError("H5AD var columns have inconsistent lengths")
    rna_columns: list[int] = []
    adt_columns: list[int] = []
    for marker, rna_id, adt_id in zip(MARKERS, RNA_FEATURE_IDS, ADT_FEATURE_IDS):
        rna = np.flatnonzero(
            (gene_ids == rna_id) & (feature_types == "Gene Expression")
        )
        adt = np.flatnonzero(
            (gene_ids == adt_id) & (feature_types == "Antibody Capture")
        )
        if len(rna) != 1 or len(adt) != 1:
            raise ValueError(f"frozen marker {marker} lacks a unique RNA/ADT ID pair")
        rna_columns.append(int(rna[0]))
        adt_columns.append(int(adt[0]))
    hto_columns = [
        index
        for index, (feature_id, feature_type) in enumerate(zip(gene_ids, feature_types))
        if feature_type == "Antibody Capture"
        and str(feature_id).startswith(donor + "-")
    ]
    hto_ids = [str(gene_ids[index]) for index in hto_columns]
    if len(hto_ids) != len(set(hto_ids)):
        raise ValueError("H5AD donor HTO IDs are not unique")
    return rna_columns, adt_columns, hto_columns, hto_ids


def _normalize_hto_id(donor: str, library_id: str, hto_id: str) -> str:
    if (
        donor == "759B"
        and library_id in PATCHED_759B_LIBRARIES
        and hto_id == "759B-MLN-1"
    ):
        return "759B-MLN-263"
    return hto_id


def _uses_single_tissue_one_hto_exception(
    donor: str, filename: str, tags: list[str]
) -> bool:
    if len(tags) >= 2:
        return False
    if (
        donor == SINGLE_TISSUE_ONE_HTO_DONOR
        and filename == SINGLE_TISSUE_ONE_HTO_FILENAME
        and tags == [SINGLE_TISSUE_ONE_HTO_TAG]
    ):
        return True
    raise ValueError(
        "tissue assignment requires at least two donor HTO features outside the "
        "frozen single-tissue exception"
    )


def _hashsolo_log_likelihoods(
    counts: np.ndarray, tags: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(counts)
    if counts.ndim != 2 or counts.shape[1] != len(tags) or len(tags) < 2:
        raise ValueError("HashSolo requires at least two donor HTO columns")
    if (
        not np.isfinite(counts).all()
        or np.any(counts < 0)
        or not np.array_equal(counts, np.rint(counts))
    ):
        raise ValueError("HashSolo counts must be finite nonnegative integers")
    transformed = np.log1p(counts.astype(float))
    order = np.argsort(transformed, axis=1)
    sorted_counts = np.sort(transformed, axis=1)
    global_signal = sorted_counts[:, -1]
    global_noise = sorted_counts[:, :-1].ravel()
    global_signal_std = float(np.std(global_signal))
    global_noise_std = float(np.std(global_noise))
    if global_signal_std <= 0 or global_noise_std <= 0:
        raise ValueError("HashSolo global signal/noise variance is zero")

    def update(
        values: np.ndarray, prior_mean: float, prior_std: float
    ) -> tuple[float, float]:
        prior_precision = 1.0 / prior_std**2
        count = len(values)
        if count > 1:
            variance = float(np.var(values))
            if variance <= 0:
                raise ValueError("HashSolo tag-specific variance is zero")
            precision = 1.0 / variance
        else:
            precision = prior_precision
        posterior_precision = prior_precision + count * precision
        posterior_mean = (
            (float(np.mean(values)) * count * precision + prior_mean * prior_precision)
            / posterior_precision
            if count
            else prior_mean
        )
        return posterior_mean, math.sqrt((count + 1) / posterior_precision)

    signal_parameters: list[tuple[float, float]] = []
    noise_parameters: list[tuple[float, float]] = []
    for tag_index in range(len(tags)):
        values = transformed[:, tag_index]
        signal_parameters.append(
            update(
                values[order[:, -1] == tag_index],
                float(np.mean(global_signal)),
                global_signal_std,
            )
        )
        noise_parameters.append(
            update(
                values[np.any(order[:, :-1] == tag_index, axis=1)],
                float(np.mean(global_noise)),
                global_noise_std,
            )
        )

    log_likelihoods = np.empty((len(counts), 3), dtype=float)
    for cell in range(len(counts)):
        signal_index = int(order[cell, -1])
        noise_index = int(order[cell, -2])
        signal_value = transformed[cell, signal_index]
        noise_value = transformed[cell, noise_index]
        epsilon = 1e-15
        signal_signal = math.log(
            norm.pdf(signal_value, *signal_parameters[signal_index]) + epsilon
        )
        noise_signal = math.log(
            norm.pdf(noise_value, *signal_parameters[noise_index]) + epsilon
        )
        noise_noise = math.log(
            norm.pdf(noise_value, *noise_parameters[noise_index]) + epsilon
        )
        signal_noise = math.log(
            norm.pdf(signal_value, *noise_parameters[noise_index]) + epsilon
        )
        log_likelihoods[cell] = (
            noise_noise + signal_noise,
            noise_noise + signal_signal,
            noise_signal + signal_signal,
        )
    return log_likelihoods, order


def _hashsolo_classifications(counts: np.ndarray, tags: list[str]) -> np.ndarray:
    """Reproduce Scanpy 1.7.2 HashSolo with the authors' frozen call."""

    counts = np.asarray(counts)
    log_likelihoods, order = _hashsolo_log_likelihoods(counts, tags)
    log_posteriors = log_likelihoods + np.log(HASH_SOLO_PRIORS)
    if not np.isfinite(log_posteriors).all():
        raise FloatingPointError("HashSolo posterior is nonfinite")
    log_posteriors -= logsumexp(log_posteriors, axis=1, keepdims=True)
    hypotheses = np.argmax(log_posteriors, axis=1)
    classifications = np.full(len(counts), "Negative", dtype=object)
    classifications[hypotheses == 2] = "Doublet"
    singlets = hypotheses == 1
    winning_tags = order[singlets, -1]
    classifications[singlets] = np.asarray(tags, dtype=object)[winning_tags]
    return classifications.astype(str)


def _filename_binding(path: Path, donor: str) -> tuple[str, str]:
    match = FILENAME_PATTERN.fullmatch(path.name)
    if match is None or match.group("donor") != donor:
        raise ValueError("source H5AD filename differs from the frozen GEO contract")
    return match.group("library"), path.name


def _cell_selection_hash(donor: str, filename: str, barcode: str) -> str:
    return hashlib.sha256(
        f"{CELL_SELECTION_SALT}{donor}{filename}{barcode}".encode()
    ).hexdigest()


def reduce_library(
    h5ad_path: Path,
    donor: str,
    output_path: Path,
    *,
    phase: str = "development",
    permit: HeldAccessPermit | None = None,
) -> dict[str, Any]:
    """Reduce one authorized H5AD to at most 512 outcome-independent candidates."""

    role = _authorize_access(donor, phase, permit)
    if not h5ad_path.is_file() or h5ad_path.is_symlink():
        raise ValueError("source H5AD must be a regular non-symlink file")
    library_id, filename = _filename_binding(h5ad_path, donor)
    source_bytes = h5ad_path.stat().st_size
    source_sha256 = _sha256(h5ad_path)

    with h5py.File(h5ad_path, "r") as handle:
        if not {"X", "obs", "var"}.issubset(handle):
            raise ValueError("H5AD lacks X, obs, or var")
        if not isinstance(handle["obs"], h5py.Group) or not isinstance(
            handle["var"], h5py.Group
        ):
            raise ValueError("H5AD obs/var are not data-frame groups")
        barcodes = _dataframe_index(handle["obs"])
        names = _dataframe_index(handle["var"])
        gene_ids = _dataframe_column(handle["var"], "gene_ids")
        feature_types = _dataframe_column(handle["var"], "feature_types")
        if len(barcodes) == 0 or len(set(barcodes)) != len(barcodes):
            raise ValueError("H5AD barcode axis is empty or nonunique")
        rna_columns, adt_columns, hto_columns, raw_hto_ids = _feature_columns(
            names, gene_ids, feature_types, donor
        )
        normalized_hto_ids = [
            _normalize_hto_id(donor, library_id, tag) for tag in raw_hto_ids
        ]
        if len(normalized_hto_ids) != len(set(normalized_hto_ids)):
            raise ValueError("HTO normalization created duplicate donor tags")
        single_tissue_assignment = _uses_single_tissue_one_hto_exception(
            donor, filename, normalized_hto_ids
        )
        target_tags = set(MLN_TAGS[donor])
        present_targets = sorted(target_tags.intersection(normalized_hto_ids))
        if not present_targets:
            raise ValueError("MLN-manifest H5AD lacks an accepted MLN HTO ID")
        selected_columns = rna_columns + adt_columns + hto_columns
        values = _matrix_columns(
            handle["X"], selected_columns, (len(barcodes), len(names))
        )

    candidates: list[dict[str, Any]] = []
    target_singlets = 0
    hto_counts = values[:, 2 * len(MARKERS) :]
    classifications = (
        np.full(len(barcodes), normalized_hto_ids[0], dtype=object).astype(str)
        if single_tissue_assignment
        else _hashsolo_classifications(hto_counts, normalized_hto_ids)
    )
    retained = np.flatnonzero(np.isin(classifications, present_targets))
    target_singlets = len(retained)
    for cell in retained:
        barcode = str(barcodes[cell])
        candidates.append(
            {
                "filename": filename,
                "barcode": barcode,
                "assigned_mln_tag": str(classifications[cell]),
                "cell_selection_sha256": _cell_selection_hash(donor, filename, barcode),
                "rna_counts": values[cell, : len(MARKERS)].tolist(),
                "adt_counts": values[cell, len(MARKERS) : 2 * len(MARKERS)].tolist(),
            }
        )
    candidates.sort(
        key=lambda record: (
            record["cell_selection_sha256"],
            record["filename"],
            record["barcode"],
        )
    )
    candidates = candidates[:CELL_BUDGET]

    payload = {
        "schema": "gse299043-mln-library-reduction/1.0",
        "status": "TARGET_MLN_LIBRARY_REDUCED",
        "donor": donor,
        "role": role,
        "source_filename": filename,
        "library_id": library_id,
        "source_bytes": source_bytes,
        "source_sha256": source_sha256,
        "deposited_cells": len(barcodes),
        "raw_hto_ids": raw_hto_ids,
        "normalized_hto_ids": normalized_hto_ids,
        "present_mln_tags": present_targets,
        "target_mln_singlets": target_singlets,
        "retained_candidates": len(candidates),
        "markers": list(MARKERS),
        "rna_feature_ids": list(RNA_FEATURE_IDS),
        "adt_feature_ids": list(ADT_FEATURE_IDS),
        "hashsolo_priors": list(HASH_SOLO_PRIORS),
        "hashsolo_noise_barcodes": max(0, len(normalized_hto_ids) - 1),
        "cell_budget": CELL_BUDGET,
        "cell_selection_salt": CELL_SELECTION_SALT,
        "candidates": candidates,
        "access_audit": {
            "paired_rna_adt_columns_decoded": 18,
            "donor_hto_columns_decoded": len(normalized_hto_ids),
            "unselected_feature_columns_materialized": 0,
        },
    }
    _write_json_exclusive(output_path, payload)
    return payload


def _validated_candidates(
    paths: Iterable[Path], donor: str, role: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pieces: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    filenames: set[str] = set()
    cells: set[tuple[str, str]] = set()
    for path in paths:
        piece = _read_json(path)
        if (
            piece.get("schema") != "gse299043-mln-library-reduction/1.0"
            or piece.get("status") != "TARGET_MLN_LIBRARY_REDUCED"
            or piece.get("donor") != donor
            or piece.get("role") != role
            or piece.get("markers") != list(MARKERS)
            or piece.get("rna_feature_ids") != list(RNA_FEATURE_IDS)
            or piece.get("adt_feature_ids") != list(ADT_FEATURE_IDS)
            or piece.get("hashsolo_priors") != list(HASH_SOLO_PRIORS)
            or piece.get("cell_selection_salt") != CELL_SELECTION_SALT
            or piece.get("cell_budget") != CELL_BUDGET
        ):
            raise ValueError(f"library reduction contract differs: {path.name}")
        filename = piece.get("source_filename")
        if not isinstance(filename, str) or filename in filenames:
            raise ValueError("library reductions contain a duplicate source filename")
        match = FILENAME_PATTERN.fullmatch(filename)
        if (
            match is None
            or match.group("donor") != donor
            or match.group("library") != piece.get("library_id")
            or not re.fullmatch(r"[0-9a-f]{64}", str(piece.get("source_sha256")))
            or not isinstance(piece.get("source_bytes"), int)
            or piece["source_bytes"] <= 0
        ):
            raise ValueError("library reduction source binding is malformed")
        normalized_hto_ids = piece.get("normalized_hto_ids")
        present_mln_tags = piece.get("present_mln_tags")
        if (
            not isinstance(normalized_hto_ids, list)
            or len(normalized_hto_ids) != len(set(normalized_hto_ids))
            or not isinstance(present_mln_tags, list)
            or not present_mln_tags
            or not set(present_mln_tags).issubset(MLN_TAGS[donor])
            or piece.get("hashsolo_noise_barcodes")
            != max(0, len(normalized_hto_ids) - 1)
        ):
            raise ValueError("library reduction HTO binding is malformed")
        filenames.add(filename)
        records = piece.get("candidates")
        if (
            not isinstance(records, list)
            or len(records) > CELL_BUDGET
            or piece.get("retained_candidates") != len(records)
            or not isinstance(piece.get("target_mln_singlets"), int)
            or piece["target_mln_singlets"] < len(records)
        ):
            raise ValueError("library reduction candidate list is malformed")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("library reduction candidate is not an object")
            barcode = record.get("barcode")
            if not isinstance(barcode, str) or record.get("filename") != filename:
                raise ValueError("library reduction candidate identity differs")
            identity = (filename, barcode)
            if identity in cells:
                raise ValueError("library reductions contain a duplicate cell")
            cells.add(identity)
            if record.get("cell_selection_sha256") != _cell_selection_hash(
                donor, filename, barcode
            ):
                raise ValueError("library reduction candidate selection hash differs")
            for key in ("rna_counts", "adt_counts"):
                count = record.get(key)
                if (
                    not isinstance(count, list)
                    or len(count) != len(MARKERS)
                    or any(
                        not isinstance(value, int)
                        or isinstance(value, bool)
                        or value < 0
                        for value in count
                    )
                ):
                    raise ValueError(f"library reduction {key} are malformed")
            if record.get("assigned_mln_tag") not in MLN_TAGS[donor]:
                raise ValueError("library reduction retained a non-MLN classification")
            if (
                _normalize_hto_id(
                    donor, str(piece["library_id"]), str(record["assigned_mln_tag"])
                )
                != record["assigned_mln_tag"]
            ):
                raise ValueError("library reduction retained an unnormalized HTO ID")
            candidates.append(record)
        pieces.append(
            {
                "piece": path.name,
                "piece_sha256": _sha256(path),
                "source_filename": filename,
                "source_sha256": piece.get("source_sha256"),
                "status": piece.get("status"),
                "target_mln_singlets": piece.get("target_mln_singlets"),
            }
        )
    return pieces, candidates


def _adt_states(
    counts: np.ndarray, cells: list[dict[str, Any]], donor: str
) -> np.ndarray:
    states = np.ones_like(counts, dtype=np.uint8)
    for marker_index, marker in enumerate(MARKERS):
        tie_hashes = [
            hashlib.sha256(
                (
                    f"{ADT_TIE_SALT}{donor}{cell['filename']}{cell['barcode']}{marker}"
                ).encode()
            ).hexdigest()
            for cell in cells
        ]
        order = sorted(
            range(len(cells)),
            key=lambda cell: (int(counts[marker_index, cell]), tie_hashes[cell]),
        )
        states[marker_index, order[: CELL_BUDGET // 2]] = 0
    return states


def _ordered_tables(rna: np.ndarray, adt: np.ndarray) -> np.ndarray:
    tables = np.empty((len(MARKERS) ** 2, 2, 2), dtype=np.int64)
    entity = 0
    for rna_index in range(len(MARKERS)):
        for adt_index in range(len(MARKERS)):
            code = 2 * rna[rna_index].astype(int) + adt[adt_index].astype(int)
            tables[entity] = np.bincount(code, minlength=4).reshape(2, 2)
            entity += 1
    return tables


def finalize_donor(
    piece_paths: Iterable[Path],
    donor: str,
    output_path: Path,
    *,
    phase: str = "development",
    permit: HeldAccessPermit | None = None,
) -> dict[str, Any]:
    """Pool per-library pieces and materialize the frozen 512-cell donor map."""

    role = _authorize_access(donor, phase, permit)
    pieces, candidates = _validated_candidates(piece_paths, donor, role)
    candidates.sort(
        key=lambda record: (
            record["cell_selection_sha256"],
            record["filename"],
            record["barcode"],
        )
    )
    if len(candidates) < CELL_BUDGET:
        raise ValueError(f"{donor} has fewer than {CELL_BUDGET} MLN singlets")
    selected = candidates[:CELL_BUDGET]
    rna_counts = np.asarray([cell["rna_counts"] for cell in selected], dtype=np.int64).T
    adt_counts = np.asarray([cell["adt_counts"] for cell in selected], dtype=np.int64).T
    rna = (rna_counts > 0).astype(np.uint8)
    adt = _adt_states(adt_counts, selected, donor)
    if not np.all(adt.sum(axis=1) == CELL_BUDGET // 2):
        raise AssertionError("ADT midrank did not produce exact 256/256 states")
    tables = _ordered_tables(rna, adt)
    seed = int.from_bytes(
        hashlib.sha256(f"destroyed-v1{donor}".encode()).digest()[:8], "big"
    )
    destroyed = _ordered_tables(
        rna, adt[:, np.random.default_rng(seed).permutation(CELL_BUDGET)]
    )
    if not np.array_equal(
        tables.sum(axis=-1), destroyed.sum(axis=-1)
    ) or not np.array_equal(tables.sum(axis=-2), destroyed.sum(axis=-2)):
        raise AssertionError("destroyed-link control changed a table margin")
    panel_total = adt_counts.sum(axis=0, keepdims=True)
    composition = np.divide(
        100.0 * adt_counts,
        panel_total,
        out=np.zeros_like(adt_counts, dtype=float),
        where=panel_total > 0,
    )
    payload = {
        "schema": "gse299043-mln-reduced-donor/1.0",
        "status": "DONOR_REDUCTION_COMPLETE",
        "donor": donor,
        "role": role,
        "markers": list(MARKERS),
        "entity_count": len(MARKERS) ** 2,
        "cells": CELL_BUDGET,
        "candidate_mln_singlets_retained_across_pieces": len(candidates),
        "total_mln_singlets_before_per_library_budget": sum(
            int(piece["target_mln_singlets"]) for piece in pieces
        ),
        "cell_selection_salt": CELL_SELECTION_SALT,
        "adt_tie_salt": ADT_TIE_SALT,
        "selected_cell_axis_sha256": hashlib.sha256(
            (
                "\n".join(f"{cell['filename']}\t{cell['barcode']}" for cell in selected)
                + "\n"
            ).encode()
        ).hexdigest(),
        "rna_detection_prevalence": rna.mean(axis=1).tolist(),
        "adt_log_panel_fraction_mean": np.log1p(composition).mean(axis=1).tolist(),
        "tables": tables.reshape(len(MARKERS) ** 2, 4).tolist(),
        "destroyed_tables": destroyed.reshape(len(MARKERS) ** 2, 4).tolist(),
        "library_pieces": pieces,
        "access_audit": {
            "source_h5ad_required_during_pooling": False,
            "maximum_concurrent_source_h5ads": 1,
            "selected_cells": CELL_BUDGET,
        },
    }
    _write_json_exclusive(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    library = subparsers.add_parser("library")
    library.add_argument("--h5ad", type=Path, required=True)
    library.add_argument("--donor", choices=DEVELOPMENT_DONORS, required=True)
    library.add_argument("--output", type=Path, required=True)
    donor = subparsers.add_parser("donor")
    donor.add_argument("--piece", type=Path, action="append", required=True)
    donor.add_argument("--donor", choices=DEVELOPMENT_DONORS, required=True)
    donor.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "library":
        payload = reduce_library(args.h5ad, args.donor, args.output)
    else:
        payload = finalize_donor(args.piece, args.donor, args.output)
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
