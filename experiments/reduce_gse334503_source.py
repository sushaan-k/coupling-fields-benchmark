"""Reduce an explicitly selected GSE334503 stage to linked RNA-ADT cells."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import urllib.request

import numpy as np
from scipy import sparse
from scipy.io import mmread


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = (
    Path.home() / "Library/Caches/coupling-fields/gse334503-source"
)
DEFAULT_OUTPUT_DIRECTORY = ROOT / "data/development/gse334503_source"
HTODEMUX_SCRIPT = Path(__file__).with_name("gse334503_htodemux.R")
SERIES_BASE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE334nnn/GSE334503/suppl"
)
SAMPLE_BASE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9789nnn/{gsm}/suppl"
)
CELL_BUDGET = 512
MIN_DETECTED_GENES = 200
MAX_TOTAL_RNA = 70_000
CELL_SELECTION_SALT = "gse334503-source-day0-v1"
SEURAT_REFERENCE_COMMIT = "51b449b477fd3f63593f6c783daaac8d72a06dc6"
DEFAULT_BATCHES = (1, 2)

PANEL = (
    ("CD1C", "CD1c"),
    ("CD2", "CD2.1"),
    ("CD4", "CD4.1"),
    ("CD7", "CD7.1"),
    ("CD8A", "CD8"),
    ("ITGAM", "CD11b"),
    ("ITGAX", "CD11c"),
    ("CD14", "CD14.1"),
    ("CD19", "CD19.1"),
    ("MS4A1", "CD20"),
    ("CD22", "CD22.1"),
    ("CD27", "CD27.1"),
    ("CD33", "CD33.1"),
    ("CD36", "CD36.1"),
    ("CD38", "CD38.1"),
    ("CD40", "CD40.1"),
    ("CD47", "CD47.1"),
    ("FCGR1A", "CD64"),
    ("CD69", "CD69.1"),
    ("CD86", "CD86.1"),
    ("CD163", "CD163.1"),
    ("CX3CR1", "CX3CR1.1"),
)

SOURCE_BATCHES = {
    1: {"gex_gsm": "GSM9789808", "adt_gsm": "GSM9789809"},
    2: {"gex_gsm": "GSM9789810", "adt_gsm": "GSM9789811"},
    3: {"gex_gsm": "GSM9789812", "adt_gsm": "GSM9789813"},
}

EXPECTED_BYTES = {
    "GSE334503_feature_reference_ADT.csv.gz": 2_431,
    "GSE334503_feature_reference_HTO.csv.gz": 775,
    "GSM9789808_GEX_Batch1_barcodes.tsv.gz": 270_948,
    "GSM9789808_GEX_Batch1_features.tsv.gz": 126_983,
    "GSM9789808_GEX_Batch1_matrix.mtx.gz": 249_414_108,
    "GSM9789809_ADT_Batch1_barcodes.tsv.gz": 270_948,
    "GSM9789809_ADT_Batch1_features.tsv.gz": 930,
    "GSM9789809_ADT_Batch1_matrix.mtx.gz": 17_404_683,
    "GSM9789810_GEX_Batch2_barcodes.tsv.gz": 370_877,
    "GSM9789810_GEX_Batch2_features.tsv.gz": 126_983,
    "GSM9789810_GEX_Batch2_matrix.mtx.gz": 731_564_492,
    "GSM9789811_ADT_Batch2_barcodes.tsv.gz": 370_877,
    "GSM9789811_ADT_Batch2_features.tsv.gz": 933,
    "GSM9789811_ADT_Batch2_matrix.mtx.gz": 32_748_424,
    "GSM9789812_GEX_Batch3_barcodes.tsv.gz": 393_189,
    "GSM9789812_GEX_Batch3_features.tsv.gz": 126_983,
    "GSM9789812_GEX_Batch3_matrix.mtx.gz": 802_371_066,
    "GSM9789813_ADT_Batch3_barcodes.tsv.gz": 393_189,
    "GSM9789813_ADT_Batch3_features.tsv.gz": 931,
    "GSM9789813_ADT_Batch3_matrix.mtx.gz": 32_436_075,
}

EXPECTED_SHA256 = {
    "GSE334503_feature_reference_ADT.csv.gz": (
        "84eb07b4f5ae8aa1051c4c6b298d6a95e3416bc76f90268d16eb6b1466e7b612"
    ),
    "GSE334503_feature_reference_HTO.csv.gz": (
        "c229ed27299c7b49e5f85a2026286f763ba7bb5ae452589d639eb9a10cd53786"
    ),
    "GSM9789808_GEX_Batch1_features.tsv.gz": (
        "bac4cf7c2cb91a1688fcbfc09975fa7785fb17ac56f9859218c23741d928ffda"
    ),
    "GSM9789809_ADT_Batch1_features.tsv.gz": (
        "498f4ca459b5f18bf1c73158008ad2eb7465e3372a1375832d9c4bc4be3d5e83"
    ),
    "GSM9789810_GEX_Batch2_features.tsv.gz": (
        "1ab0b33d8524028586e32ef90ef12309c545f9d23ec1e72df48e81c20b6b4994"
    ),
    "GSM9789811_ADT_Batch2_features.tsv.gz": (
        "444e9c66acd20c15bda341d7badfd8d198e47f55cd4f2c7bbf229239527fa487"
    ),
    "GSM9789812_GEX_Batch3_features.tsv.gz": (
        "c346e88636f74a707326027f63951703a02dfb6987e441480637b1e5d204dbde"
    ),
    "GSM9789813_ADT_Batch3_features.tsv.gz": (
        "ea12d074f11d696a7f7ca7ed11d8d042a04174ace340ccc7af7d4ac06051f09a"
    ),
}

STAGES = {
    (1, 2): ("source_development", "batches_1_2"),
    (3,): ("sealed_internal_validation", "batch_3"),
    (1, 2, 3): ("post_validation_refit", "batches_1_2_3"),
}

ADT_REFERENCE_HEADER = (
    "id",
    "name",
    "read",
    "pattern",
    "sequence",
    "feature_type",
)
HTO_REFERENCE_HEADER = (
    "id",
    "multiplex_sample",
    "read",
    "pattern",
    "sequence",
    "feature_type",
    "Batch",
)
HTO_SAMPLE_PATTERN = re.compile(r"Donor(?P<donor>\d{3})_Day(?P<day>[07])postVax")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _axis_sha256(values: list[str] | tuple[str, ...]) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in values).encode()).hexdigest()


def _download(url: str, path: Path, expected_bytes: int) -> None:
    if path.is_file():
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"cached file has wrong byte count: {path.name}")
        if path.name in EXPECTED_SHA256 and _sha256(path) != EXPECTED_SHA256[path.name]:
            raise ValueError(f"cached file has wrong SHA-256: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open(
            "wb"
        ) as stream:
            while block := response.read(8 << 20):
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.stat().st_size != expected_bytes:
            raise ValueError(f"downloaded file has wrong byte count: {path.name}")
        if (
            path.name in EXPECTED_SHA256
            and _sha256(temporary) != EXPECTED_SHA256[path.name]
        ):
            raise ValueError(f"downloaded file has wrong SHA-256: {path.name}")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_gzip_axis(path: Path) -> list[str]:
    with gzip.open(path, "rt") as stream:
        values = [line.rstrip("\n\r") for line in stream]
    if not values or any(not value for value in values):
        raise ValueError(f"axis is empty or contains an empty value: {path.name}")
    return values


def _read_csv_reference(path: Path, header: tuple[str, ...]) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != header:
            raise ValueError(f"reference header differs: {path.name}")
        rows = [dict(row) for row in reader]
    if not rows or any(any(not row[field] for field in header) for row in rows):
        raise ValueError(f"reference contains an empty field: {path.name}")
    if any(
        row["read"] != "R2"
        or row["pattern"] != "5PNNNNNNNNNN(BC)"
        or row["feature_type"] != "Antibody Capture"
        for row in rows
    ):
        raise ValueError(f"reference assay schema differs: {path.name}")
    return rows


def _reference_paths(cache: Path) -> dict[str, Path]:
    names = {
        "adt": "GSE334503_feature_reference_ADT.csv.gz",
        "hto": "GSE334503_feature_reference_HTO.csv.gz",
    }
    paths = {role: cache / name for role, name in names.items()}
    for role, path in paths.items():
        del role
        _download(
            f"{SERIES_BASE_URL}/{path.name}", path, EXPECTED_BYTES[path.name]
        )
    return paths


def _batch_paths(batch: int, cache: Path) -> dict[str, Path]:
    if batch not in SOURCE_BATCHES:
        raise ValueError(f"unsupported source batch: {batch}")
    paths: dict[str, Path] = {}
    for assay, gsm_key in (("gex", "gex_gsm"), ("adt", "adt_gsm")):
        gsm = SOURCE_BATCHES[batch][gsm_key]
        for role, suffix in (
            ("barcodes", "barcodes.tsv.gz"),
            ("features", "features.tsv.gz"),
            ("matrix", "matrix.mtx.gz"),
        ):
            name = f"{gsm}_{assay.upper()}_Batch{batch}_{suffix}"
            path = cache / name
            url = f"{SAMPLE_BASE_URL.format(gsm=gsm)}/{name}"
            _download(url, path, EXPECTED_BYTES[name])
            paths[f"{assay}_{role}"] = path
    return paths


def _hto_identity(row: dict[str, str]) -> tuple[str, str, str]:
    if row["multiplex_sample"] == "Ctrl":
        return "Ctrl", "Control", "Ctrl"
    match = HTO_SAMPLE_PATTERN.fullmatch(row["multiplex_sample"])
    if match is None:
        raise ValueError("HTO multiplex_sample does not match the declared schema")
    donor_number = match.group("donor")
    day = f"Day{match.group('day')}"
    observed = f"{donor_number}-{'V1' if day == 'Day0' else 'V2'}"
    return f"Donor{donor_number}", day, observed


def _source_hto_rows(
    rows: list[dict[str, str]], batch: int
) -> list[dict[str, str]]:
    selected = [row for row in rows if row["Batch"] == f"Batch{batch}"]
    if len(selected) != 13:
        raise ValueError(f"Batch{batch} must declare 12 sample HTOs and one control")
    identities = [_hto_identity(row) for row in selected]
    if sum(day == "Control" for _, day, _ in identities) != 1:
        raise ValueError(f"Batch{batch} must declare exactly one HTO control")
    sample_identities = [identity for identity in identities if identity[1] != "Control"]
    if len(set(sample_identities)) != 12:
        raise ValueError(f"Batch{batch} sample HTO assignments are duplicated")
    if sum(day == "Day0" for _, day, _ in sample_identities) != 6:
        raise ValueError(f"Batch{batch} must declare six Day0 units")
    return selected


def _validate_feature_axes(
    batch: int,
    gex_features: list[str],
    adt_features: list[str],
    adt_reference: list[dict[str, str]],
    hto_rows: list[dict[str, str]],
) -> dict[str, object]:
    if len(adt_reference) != 137:
        raise ValueError("ADT feature reference must contain exactly 137 rows")
    if len(adt_features) != 150 or len(set(adt_features)) != 150:
        raise ValueError("ADT axis must contain 150 unique rows")
    gene_names = set(gex_features)
    expected_biological = [
        row["name"] + (".1" if row["name"] in gene_names else "")
        for row in adt_reference
    ]
    if adt_features[:137] != expected_biological:
        raise ValueError("first 137 ADT rows differ from the exact feature reference")
    expected_hto = [_hto_identity(row)[2] for row in hto_rows]
    if adt_features[137:] != expected_hto:
        raise ValueError(f"Batch{batch} trailing HTO axis differs from its reference")
    for gene, adt in PANEL:
        if gex_features.count(gene) != 1:
            raise ValueError(f"expected exactly one RNA feature for {gene}")
        if expected_biological.count(adt) != 1:
            raise ValueError(f"expected exactly one biological ADT feature for {adt}")
    mitochondrial_features = [
        name for name in gex_features if name.startswith("MT-")
    ]
    if mitochondrial_features:
        raise ValueError(
            "GSE334503 deposited GEX axis unexpectedly contains canonical "
            "mitochondrial features"
        )
    return {
        "gex_feature_count": len(gex_features),
        "gex_feature_axis_sha256": _axis_sha256(gex_features),
        "adt_feature_count": len(adt_features),
        "adt_feature_axis_sha256": _axis_sha256(adt_features),
        "biological_adt_rows": 137,
        "hto_rows": 13,
        "mitochondrial_qc_evidence": {
            "status": "UNAVAILABLE_NO_CANONICAL_MITOCHONDRIAL_FEATURES",
            "applied": False,
            "recognition_rule": "feature name starts with 'MT-'",
            "recognized_feature_count": 0,
            "recognized_feature_axis": [],
            "evidence_axis_sha256": _axis_sha256(gex_features),
        },
    }


def _read_integer_matrix(path: Path) -> tuple[sparse.csr_matrix, dict[str, object]]:
    with gzip.open(path, "rb") as stream:
        header = stream.readline().decode("ascii").rstrip("\n\r")
        if header != "%%MatrixMarket matrix coordinate integer general":
            raise ValueError(f"matrix is not coordinate integer general: {path.name}")
        stream.seek(0)
        matrix = mmread(stream)
    if not sparse.issparse(matrix):
        raise ValueError(f"matrix is not sparse: {path.name}")
    matrix = matrix.tocoo(copy=False)
    if (
        not np.all(np.isfinite(matrix.data))
        or np.any(matrix.data <= 0)
        or np.any(matrix.data != np.floor(matrix.data))
        or np.any(matrix.data > np.iinfo(np.int32).max)
    ):
        raise ValueError(f"matrix contains invalid counts: {path.name}")
    original_nnz = int(matrix.nnz)
    matrix.sum_duplicates()
    if matrix.nnz != original_nnz:
        raise ValueError(f"matrix contains duplicate coordinates: {path.name}")
    converted = matrix.astype(np.int32, copy=False).tocsr()
    return converted, {
        "shape": list(converted.shape),
        "nnz": int(converted.nnz),
        "matrix_market_field": "integer",
        "counts_nonnegative_integer": True,
        "duplicate_coordinates": False,
    }


def _parse_htodemux_output(path: Path) -> tuple[np.ndarray, list[dict[str, object]], dict[str, str]]:
    versions: dict[str, str] = {}
    lines: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith("# "):
            key, value = line[2:].split("\t", 1)
            versions[key] = value
        elif line:
            lines.append(line)
    reader = csv.DictReader(lines, delimiter="\t")
    rows = list(reader)
    if len(rows) != 12 or [int(row["tag_index"]) for row in rows] != list(range(12)):
        raise ValueError("HTODemux output does not contain the exact 12-tag axis")
    cutoffs = np.asarray([int(row["cutoff"]) for row in rows], dtype=np.int64)
    audit = [
        {
            "tag_index": int(row["tag_index"]),
            "negative_cluster": int(row["negative_cluster"]),
            "background_cells": int(row["background_cells"]),
            "size": float(row["size"]),
            "mu": float(row["mu"]),
            "cutoff": int(row["cutoff"]),
        }
        for row in rows
    ]
    if np.any(cutoffs < 0) or set(versions) != {
        "r_version",
        "cluster_version",
        "mass_version",
    }:
        raise ValueError("HTODemux output metadata is malformed")
    return cutoffs, audit, versions


def _seurat_htodemux(
    counts: np.ndarray, cache: Path, rscript: str
) -> tuple[np.ndarray, dict[str, object]]:
    counts = np.asarray(counts)
    if (
        counts.ndim != 2
        or counts.shape[1] != 12
        or len(counts) <= 13
        or not np.issubdtype(counts.dtype, np.integer)
        or np.any(counts < 0)
    ):
        raise ValueError("HTODemux input must be cells by 12 nonnegative integer HTOs")
    executable = shutil.which(rscript)
    if executable is None:
        raise RuntimeError(f"Rscript executable not found: {rscript}")
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="htodemux-", dir=cache) as temporary:
        temporary_path = Path(temporary)
        input_path = temporary_path / "counts.tsv"
        output_path = temporary_path / "cutoffs.tsv"
        np.savetxt(input_path, counts.T, fmt="%d", delimiter="\t")
        completed = subprocess.run(
            [executable, str(HTODEMUX_SCRIPT), str(input_path), str(output_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Seurat-compatible HTODemux failed: {message}")
        cutoffs, fits, versions = _parse_htodemux_output(output_path)
    positives = counts > cutoffs[None, :]
    positive_count = positives.sum(axis=1)
    denominators = np.exp(
        np.asarray(
            [
                np.log1p(column[column > 0]).sum() / len(column)
                for column in counts.T
            ]
        )
    )
    clr = np.log1p(counts / denominators[None, :])
    top_tag = np.argmax(clr, axis=1)
    assignment = np.full(len(counts), -1, dtype=np.int16)
    singlet = positive_count == 1
    assignment[singlet] = top_tag[singlet]
    return assignment, {
        "method": "Seurat_HTODemux_compatible",
        "seurat_reference_commit": SEURAT_REFERENCE_COMMIT,
        "clr_margin": 1,
        "clustering": "cluster::clara",
        "clusters": 13,
        "clara_samples": 100,
        "seed": 42,
        "positive_quantile": 0.99,
        "negative_binomial_fit": "MASS::fitdistr",
        "cutoffs": cutoffs.tolist(),
        "fits": fits,
        "classification_counts": {
            "negative": int(np.count_nonzero(positive_count == 0)),
            "singlet": int(np.count_nonzero(singlet)),
            "doublet": int(np.count_nonzero(positive_count > 1)),
        },
        "runtime": versions,
        "helper_sha256": _sha256(HTODEMUX_SCRIPT),
    }


def _selection_hash(batch: int, donor: str, barcode: str) -> str:
    return hashlib.sha256(
        f"{CELL_SELECTION_SALT}|Batch{batch}|{donor}|{barcode}".encode()
    ).hexdigest()


def _adt_graph_specification(
    adt_features: list[str],
    adt_reference: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    excluded_rows = [
        index
        for index, row in enumerate(adt_reference)
        if "isotype ctrl" in row["name"].casefold()
    ]
    if len(excluded_rows) != 7:
        raise ValueError("ADT reference must contain exactly seven isotype controls")
    denominator_rows = np.asarray(
        [index for index in range(137) if index not in set(excluded_rows)],
        dtype=np.int64,
    )
    denominator_axis = [adt_features[index] for index in denominator_rows]
    if len(denominator_axis) != 130 or len(set(denominator_axis)) != 130:
        raise ValueError("ADT graph denominator must contain 130 unique features")
    cognate_rows = [adt_features.index(protein) for _, protein in PANEL]
    row_to_position = {
        int(row): position for position, row in enumerate(denominator_rows)
    }
    try:
        cognate_positions = np.asarray(
            [row_to_position[row] for row in cognate_rows], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError("an ADT cognate was excluded from the graph denominator") from error
    return denominator_rows, cognate_positions, {
        "cell_clr_formula": (
            "clr_cell = log1p(count_130) - mean_j(log1p(count_130))"
        ),
        "donor_profile_formula": (
            "mean_cells(clr_cell), restricted to the 22 cognate positions"
        ),
        "denominator_rule": (
            "first 137 biological ADT rows excluding the seven feature-reference "
            "names containing 'isotype Ctrl' case-insensitively"
        ),
        "denominator_feature_count": 130,
        "denominator_adt_axis": denominator_axis,
        "denominator_adt_axis_sha256": _axis_sha256(denominator_axis),
        "denominator_adt_set_sha256": _axis_sha256(sorted(denominator_axis)),
        "excluded_isotype_reference_names": [
            adt_reference[index]["name"] for index in excluded_rows
        ],
        "cognate_positions_zero_based": cognate_positions.tolist(),
        "cognate_adt_axis": [protein for _, protein in PANEL],
    }


def _adt_graph_profile(
    adt: sparse.csr_matrix,
    selected: np.ndarray,
    denominator_rows: np.ndarray,
    cognate_positions: np.ndarray,
) -> np.ndarray:
    counts = adt[denominator_rows][:, selected].toarray().T.astype(np.float64)
    log_counts = np.log1p(counts)
    clr = log_counts - log_counts.mean(axis=1, keepdims=True)
    profile = clr.mean(axis=0)[cognate_positions]
    if profile.shape != (22,) or not np.all(np.isfinite(profile)):
        raise ValueError("ADT graph profile is malformed")
    return profile


def _rna_qc_mask(
    total_rna: np.ndarray,
    detected_genes: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    positive_total = total_rna > 0
    enough_genes = detected_genes >= MIN_DETECTED_GENES
    below_umi_ceiling = total_rna <= MAX_TOTAL_RNA
    accepted = positive_total & enough_genes & below_umi_ceiling
    return accepted, {
        "zero_total_rna": int(np.count_nonzero(~positive_total)),
        "below_minimum_detected_genes": int(np.count_nonzero(~enough_genes)),
        "above_maximum_total_rna": int(np.count_nonzero(~below_umi_ceiling)),
        "accepted": int(np.count_nonzero(accepted)),
    }


def _select_donor_cells(
    batch: int,
    donor: str,
    barcodes: list[str],
    assignment: np.ndarray,
    tag_index: int,
    rna_qc: np.ndarray,
) -> np.ndarray:
    eligible = np.flatnonzero((assignment == tag_index) & rna_qc)
    if len(eligible) < CELL_BUDGET:
        raise ValueError(
            f"Batch{batch} {donor} has {len(eligible)} HTO-singlet RNA-QC cells; "
            f"need {CELL_BUDGET}"
        )
    ordered = sorted(
        eligible,
        key=lambda index: (
            _selection_hash(batch, donor, barcodes[index]),
            barcodes[index],
        ),
    )
    return np.asarray(ordered[:CELL_BUDGET], dtype=np.int64)


def _file_audit(paths: dict[str, Path]) -> dict[str, dict[str, object]]:
    return {
        role: {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for role, path in sorted(paths.items())
    }


def _reduce_loaded_batch(
    batch: int,
    paths: dict[str, Path],
    adt_reference: list[dict[str, str]],
    hto_rows: list[dict[str, str]],
    cache: Path,
    rscript: str,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    gex_features = _read_gzip_axis(paths["gex_features"])
    adt_features = _read_gzip_axis(paths["adt_features"])
    feature_audit = _validate_feature_axes(
        batch, gex_features, adt_features, adt_reference, hto_rows
    )
    gex_barcodes = _read_gzip_axis(paths["gex_barcodes"])
    adt_barcodes = _read_gzip_axis(paths["adt_barcodes"])
    if (
        len(gex_barcodes) != len(set(gex_barcodes))
        or gex_barcodes != adt_barcodes
    ):
        raise ValueError(f"Batch{batch} GEX and ADT barcode axes differ or duplicate")

    gex, gex_matrix_audit = _read_integer_matrix(paths["gex_matrix"])
    if gex.shape != (len(gex_features), len(gex_barcodes)):
        raise ValueError(f"Batch{batch} GEX matrix differs from declared axes")
    total_rna = np.asarray(gex.sum(axis=0)).ravel().astype(np.int64)
    detected_genes = np.bincount(gex.indices, minlength=gex.shape[1])
    rna_qc, rna_qc_audit = _rna_qc_mask(total_rna, detected_genes)
    rna_rows = [gex_features.index(gene) for gene, _ in PANEL]
    panel_rna = gex[rna_rows].toarray().T.astype(np.int32)
    del gex

    adt, adt_matrix_audit = _read_integer_matrix(paths["adt_matrix"])
    if adt.shape != (len(adt_features), len(adt_barcodes)):
        raise ValueError(f"Batch{batch} ADT matrix differs from declared axes")
    adt_rows = [adt_features.index(protein) for _, protein in PANEL]
    panel_adt = adt[adt_rows].toarray().T.astype(np.int32)
    hto_counts = adt[137:149].toarray().T.astype(np.int32)
    control_counts = np.asarray(adt[149].todense()).ravel().astype(np.int32)
    denominator_rows, cognate_positions, graph_specification = (
        _adt_graph_specification(adt_features, adt_reference)
    )

    identities = [_hto_identity(row) for row in hto_rows]
    sample_identities = [identity for identity in identities if identity[1] != "Control"]
    if identities[-1][1] != "Control":
        raise ValueError(f"Batch{batch} HTO control must be the final ADT row")
    assignment, htodemux_audit = _seurat_htodemux(hto_counts, cache, rscript)

    donors: list[dict[str, object]] = []
    for tag_index, (donor, day, observed_hto) in enumerate(sample_identities):
        if day != "Day0":
            continue
        selected = _select_donor_cells(
            batch, donor, gex_barcodes, assignment, tag_index, rna_qc
        )
        donors.append(
            {
                "donor": donor,
                "day": day,
                "batch": f"Batch{batch}",
                "hto_feature": observed_hto,
                "hto_tag_index": tag_index,
                "selected_barcodes": np.asarray(
                    [gex_barcodes[index] for index in selected]
                ),
                "rna_counts": panel_rna[selected],
                "adt_counts": panel_adt[selected],
                "adt_graph_profile": _adt_graph_profile(
                    adt, selected, denominator_rows, cognate_positions
                ),
                "audit": {
                    "hto_singlets_before_rna_qc": int(
                        np.count_nonzero(assignment == tag_index)
                    ),
                    "hto_singlets_passing_rna_qc": int(
                        np.count_nonzero((assignment == tag_index) & rna_qc)
                    ),
                    "selected_cells": CELL_BUDGET,
                    "selected_total_rna_minimum": int(total_rna[selected].min()),
                    "selected_total_rna_maximum": int(total_rna[selected].max()),
                    "selected_detected_genes_minimum": int(
                        detected_genes[selected].min()
                    ),
                    "selected_control_hto_median": float(
                        np.median(control_counts[selected])
                    ),
                },
            }
        )
    if len(donors) != 6:
        raise ValueError(f"Batch{batch} did not yield the exact six Day0 donors")
    del adt
    return donors, {
        "batch": f"Batch{batch}",
        "cell_count": len(gex_barcodes),
        "barcode_axis_sha256": _axis_sha256(gex_barcodes),
        "rna_qc_pass_count": int(np.count_nonzero(rna_qc)),
        "rna_qc_counts": rna_qc_audit,
        "feature_axes": feature_audit,
        "gex_matrix": gex_matrix_audit,
        "adt_matrix": adt_matrix_audit,
        "hto_demultiplexing": htodemux_audit,
        "donors": [
            {"donor": donor["donor"], **donor["audit"]} for donor in donors
        ],
        "files": _file_audit(paths),
    }, graph_specification


def _parse_batches(value: str) -> tuple[int, ...]:
    try:
        batches = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("batches must be comma-separated integers") from error
    if batches not in STAGES:
        allowed = ", ".join("/".join(map(str, option)) for option in STAGES)
        raise argparse.ArgumentTypeError(f"allowed batch stages are {allowed}")
    return batches


def _output_paths(
    batches: tuple[int, ...], output: Path | None, manifest: Path | None
) -> tuple[Path, Path]:
    _, stem = STAGES[batches]
    return (
        output or DEFAULT_OUTPUT_DIRECTORY / f"reduced_{stem}_v1.npz",
        manifest or DEFAULT_OUTPUT_DIRECTORY / f"reduction_{stem}_manifest_v1.json",
    )


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batches",
        type=_parse_batches,
        default=DEFAULT_BATCHES,
        help="explicit stage: 1,2 (development), 3 (validation), or 1,2,3 (refit)",
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--rscript", default="Rscript")
    return parser


def main() -> None:
    args = _argument_parser().parse_args()
    output_path, manifest_path = _output_paths(
        args.batches, args.output, args.manifest
    )

    reference_paths = _reference_paths(args.cache)
    adt_reference = _read_csv_reference(
        reference_paths["adt"], ADT_REFERENCE_HEADER
    )
    hto_reference = _read_csv_reference(
        reference_paths["hto"], HTO_REFERENCE_HEADER
    )
    donors: list[dict[str, object]] = []
    batch_audits: list[dict[str, object]] = []
    graph_specifications: list[dict[str, object]] = []
    for batch in args.batches:
        paths = _batch_paths(batch, args.cache)
        reduced, audit, graph_specification = _reduce_loaded_batch(
            batch,
            paths,
            adt_reference,
            _source_hto_rows(hto_reference, batch),
            args.cache,
            args.rscript,
        )
        donors.extend(reduced)
        batch_audits.append(audit)
        graph_specifications.append(graph_specification)
        print(json.dumps({"completed_batch": batch, **audit}, sort_keys=True))

    if any(
        specification != graph_specifications[0]
        for specification in graph_specifications[1:]
    ):
        raise ValueError("ADT graph-profile specification differs across batches")

    donor_axis = [str(donor["donor"]) for donor in donors]
    if len(donor_axis) != 6 * len(args.batches) or len(set(donor_axis)) != len(
        donor_axis
    ):
        raise ValueError("reduction did not produce unique six-donor batch axes")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        donor_axis=np.asarray(donor_axis),
        day_axis=np.asarray([donor["day"] for donor in donors]),
        batch_axis=np.asarray([donor["batch"] for donor in donors]),
        hto_feature_axis=np.asarray([donor["hto_feature"] for donor in donors]),
        rna_gene_axis=np.asarray([gene for gene, _ in PANEL]),
        adt_protein_axis=np.asarray([protein for _, protein in PANEL]),
        selected_barcodes=np.asarray(
            [donor["selected_barcodes"] for donor in donors]
        ),
        rna_counts=np.asarray([donor["rna_counts"] for donor in donors], dtype=np.int32),
        adt_counts=np.asarray([donor["adt_counts"] for donor in donors], dtype=np.int32),
        adt_graph_profile=np.asarray(
            [donor["adt_graph_profile"] for donor in donors], dtype=np.float64
        ),
    )
    stage, _ = STAGES[args.batches]
    manifest = {
        "schema": "gse334503-source-reduction/1.0",
        "status": "COMPLETE",
        "accession": "GSE334503",
        "stage": stage,
        "numeric_batches_processed": [f"Batch{batch}" for batch in args.batches],
        "donor_count": len(donors),
        "cell_budget_per_donor": CELL_BUDGET,
        "panel": [
            {"rna_gene": gene, "adt_protein": protein} for gene, protein in PANEL
        ],
        "adt_graph_profile": graph_specifications[0],
        "cell_selection": {
            "visit": "Day0",
            "rna_qc": {
                "minimum_detected_genes": MIN_DETECTED_GENES,
                "maximum_total_rna": MAX_TOTAL_RNA,
                "mitochondrial_qc": {
                    "status": "UNAVAILABLE_NO_CANONICAL_MITOCHONDRIAL_FEATURES",
                    "applied": False,
                    "reason": (
                        "the exact deposited GEX feature axis contains zero "
                        "feature names beginning with 'MT-'"
                    ),
                },
            },
            "rule": "HTODemux singlet and RNA QC, then smallest salted barcode hashes",
            "salt": CELL_SELECTION_SALT,
        },
        "references": {
            "files": _file_audit(reference_paths),
            "adt_rows": len(adt_reference),
            "hto_rows_used": 13 * len(args.batches),
        },
        "batches": batch_audits,
        "output_path": _display_path(output_path),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": _sha256(output_path),
        "reducer_sha256": _sha256(Path(__file__)),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
