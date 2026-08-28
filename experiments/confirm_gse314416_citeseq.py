"""Pre-outcome GSE314416 held-pool CITE-seq confirmation.

The four commands are deliberately separated. ``preflight`` reads public
metadata only; ``develop`` reads DB1--DB2; ``predict`` reads held GEX margins;
and ``score`` is the first command permitted to read held ADT counts.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable

import h5py
import numpy as np

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
DATA_DIR = ROOT / "data/confirmation/gse314416_immunomicrobiome"
SOURCE_ROOT = DATA_DIR / "source_cache"
DEFAULT_METADATA = DATA_DIR / "GSE314416_cell_metadata_baseline.rds"
DEFAULT_CELL_MANIFEST = DATA_DIR / "baseline_cell_manifest.tsv"
DEFAULT_FEATURE_REFERENCE = DATA_DIR / "GSE314416_TotalSeqC_feature_ref.csv.gz"
DEFAULT_FILE_LIST = DATA_DIR / "GSE314416_filelist.txt"
DEFAULT_SOURCE_MANIFEST = DATA_DIR / "source_manifest_v1.json"
DEFAULT_DESIGNATION = DATA_DIR / "candidate_designation_v1.json"
DEFAULT_PREFLIGHT = (
    ROOT / "results/development/gse314416_citeseq_metadata_preflight.json"
)
DEFAULT_DEVELOPMENT = ROOT / "results/development/gse314416_citeseq_development.json"
DEFAULT_PREDICTION = ROOT / "results/gse314416_citeseq_predictions.json"
DEFAULT_SCORE = ROOT / "results/gse314416_citeseq_confirmation.json"
DEFAULT_PROTOCOL = (
    ROOT
    / "docs/GSE314416_IMMUNOMICROBIOME_HELD_POOL_CONFIRMATION_PROTOCOL_2026-08-28.md"
)

PUBLIC_ORIGIN = "https://github.com/sushaan-k/coupling-fields-benchmark.git"
PROTOCOL_TAG = "gse314416-citeseq-v1-protocol"
DEVELOPMENT_TAG = "gse314416-citeseq-v1-development"
PREDICTION_TAG = "gse314416-citeseq-v1-predictions"

MARKERS = ("CD4", "CD7", "CD14", "CD19", "CD33", "CD38", "CD44", "CD47", "CD52")
RNA_FEATURES = (
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
ADT_FEATURES = (
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

# Secondary, fixed before count access. It contains the full primary panel plus
# 15 lineage and activation markers with one unambiguous cognate gene target.
BROAD_MARKERS = (
    "CD3D",
    "CD4",
    "CD8A",
    "CD7",
    "CD14",
    "FCGR3A",
    "CD19",
    "MS4A1",
    "CD33",
    "ITGAX",
    "NCAM1",
    "CD38",
    "IL7R",
    "KLRB1",
    "HLA-DRA",
    "CD27",
    "CD44",
    "CD47",
    "CD52",
    "CD2",
    "CD69",
    "SELL",
    "ITGAM",
    "FCGR2A",
)
BROAD_RNA_FEATURES = (
    "ENSG00000167286",
    "ENSG00000010610",
    "ENSG00000153563",
    "ENSG00000173762",
    "ENSG00000170458",
    "ENSG00000203747",
    "ENSG00000177455",
    "ENSG00000156738",
    "ENSG00000105383",
    "ENSG00000140678",
    "ENSG00000149294",
    "ENSG00000004468",
    "ENSG00000168685",
    "ENSG00000111796",
    "ENSG00000204287",
    "ENSG00000139193",
    "ENSG00000026508",
    "ENSG00000196776",
    "ENSG00000169442",
    "ENSG00000116824",
    "ENSG00000110848",
    "ENSG00000188404",
    "ENSG00000169896",
    "ENSG00000143226",
)
BROAD_ADT_FEATURES = (
    "C0034",
    "C0072",
    "C0046",
    "C0066",
    "C0081",
    "C0083",
    "C0050",
    "C0100",
    "C0052",
    "C0053",
    "C0047",
    "C0389",
    "C0390",
    "C0149",
    "C0159",
    "C0154",
    "C0073",
    "C0026",
    "C0033",
    "C0367",
    "C0146",
    "C0147",
    "C0161",
    "C0142",
)
BROAD_HETEROGENEITY_PENALTY = 1.0
BROAD_RIDGE_PENALTY = 0.1
BROAD_TRANSPORT_MULTIPLIER = 1.0
BROAD_RESIDUAL_FAMILY = "deviance"
BROAD_MINIMUM_INFORMATIVE_ENTITIES = 432

CELL_BUDGET = 512
MINIMUM_INFORMATIVE_ENTITIES = 64
CELL_SELECTION_SALT = "GSE314416-CELL-BUDGET-v1"
CALIBRATION_DONOR_SALT = "GSE314416-CALIBRATION-DONOR-v1"
ADT_TIE_SALT = "GSE314416-ADT-MEDIAN-v1"
DESTROYED_LINK_SALT = "GSE314416-DESTROYED-LINK-v1"

CALIBRATION_POOL = "XHLT2-POOL-DB1"
PILOT_POOLS = ("XHLT2-POOL-DB1", "XHLT2-POOL-DB2")
HELD_POOLS = tuple(f"XHLT2-POOL-DB{index}" for index in range(3, 8))
EXPECTED_COUNTS = {"calibration": 12, "pilot": 20, "held": 77, "excluded": 1}

NEIGHBOR_GRID = (1, 2)
HETEROGENEITY_GRID = (0.1, 1.0, 10.0)
RIDGE_GRID = (0.01, 0.1)
GRAPH_GRID = (0.0, 0.1, 0.3, 1.0)
TRANSPORT_GRID = (0.5, 0.75, 1.0, 1.25)
RESIDUAL_FAMILIES = ("pearson", "deviance")
CLASSICAL_INVERSION_EPSILON = 1e-10
MAXIMUM_CONDITION_NUMBER = 1e12

BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 20260828
PILOT_REQUIRED_FAVORABLE = 15
HELD_REQUIRED_FAVORABLE = 58

PROTOCOL_BINDINGS = (
    "experiments/confirm_gse314416_citeseq.py",
    "experiments/export_gse314416_metadata.R",
    "tests/test_gse314416_citeseq_confirmation.py",
    "docs/GSE314416_IMMUNOMICROBIOME_HELD_POOL_CONFIRMATION_PROTOCOL_2026-08-28.md",
    "data/confirmation/gse314416_immunomicrobiome/source_manifest_v1.json",
    "data/confirmation/gse314416_immunomicrobiome/candidate_designation_v1.json",
    "data/confirmation/gse314416_immunomicrobiome/GSE314416_TotalSeqC_feature_ref.csv.gz",
    "data/confirmation/gse314416_immunomicrobiome/GSE314416_filelist.txt",
    "results/development/gse314416_citeseq_metadata_preflight.json",
    "mapreg/heterogeneity_adaptive_coupling.py",
    "mapreg/hierarchical_conditional_coupling.py",
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _require_public_tag(tag: str, paths: Iterable[str]) -> str:
    commit = subprocess.run(
        ["git", "rev-list", "-n", "1", tag],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    remote = subprocess.run(
        ["git", "ls-remote", PUBLIC_ORIGIN, f"refs/tags/{tag}^{{}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not remote:
        remote = subprocess.run(
            ["git", "ls-remote", PUBLIC_ORIGIN, f"refs/tags/{tag}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    if not remote or remote.split()[0] != commit:
        raise PermissionError(f"public tag {tag} does not resolve to local commit")
    for relative in paths:
        tagged = subprocess.run(
            ["git", "show", f"{tag}:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if tagged != (ROOT / relative).read_bytes():
            raise PermissionError(f"{relative} differs from public tag {tag}")
    return commit


def _export_metadata(metadata_path: Path, output_path: Path) -> None:
    if output_path.exists():
        return
    subprocess.run(
        [
            "Rscript",
            str(ROOT / "experiments/export_gse314416_metadata.R"),
            str(metadata_path),
            str(output_path),
        ],
        cwd=ROOT,
        check=True,
    )


def _metadata_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows or set(rows[0]) != {"cell_id", "well", "donor", "timepoint"}:
        raise ValueError("cell manifest schema differs from the frozen exporter")
    if len({row["cell_id"] for row in rows}) != len(rows):
        raise ValueError("cell manifest contains duplicate cell IDs")
    return rows


def _pool(well: str) -> str:
    match = re.fullmatch(r"(XHLT2-POOL-DB(?:[1-9]|1[0-4]))-SCG[1-4]", well)
    if match is None:
        raise ValueError(f"unexpected well identifier: {well}")
    return match.group(1)


def _designated_roles(
    rows: list[dict[str, str]], designation: dict[str, Any]
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, int], dict[str, str]]:
    cells: dict[str, list[str]] = {}
    donor_pool: dict[str, str] = {}
    for row in rows:
        if row["timepoint"] != "baseline" or not row["cell_id"].startswith(
            row["well"] + "_"
        ):
            raise ValueError("metadata row is not a baseline well-prefixed cell")
        cells.setdefault(row["donor"], []).append(row["cell_id"])
        pool = _pool(row["well"])
        previous = donor_pool.setdefault(row["donor"], pool)
        if previous != pool:
            raise ValueError("one physical donor appears in multiple baseline pools")

    counts = {donor: len(values) for donor, values in cells.items()}
    eligible = {donor for donor, count in counts.items() if count >= CELL_BUDGET}
    db1 = sorted(donor for donor in eligible if donor_pool[donor] == CALIBRATION_POOL)
    ordered = sorted(
        db1,
        key=lambda donor: (
            hashlib.sha256(f"{CALIBRATION_DONOR_SALT}|{donor}".encode()).hexdigest(),
            donor,
        ),
    )
    calibration = ordered[: EXPECTED_COUNTS["calibration"]]
    frozen_calibration = designation["roles"]["calibration"]["donors"]
    if calibration != frozen_calibration:
        raise PermissionError("calibration donor allocation differs from designation")
    pilot = sorted(
        donor
        for donor in eligible
        if donor_pool[donor] in PILOT_POOLS and donor not in calibration
    )
    held = sorted(donor for donor in eligible if donor_pool[donor] in HELD_POOLS)
    excluded = sorted(set(cells) - eligible)
    roles = {
        donor: role
        for role, donors in (
            ("calibration", calibration),
            ("pilot", pilot),
            ("held", held),
            ("excluded", excluded),
        )
        for donor in donors
    }
    observed = {
        "calibration": len(calibration),
        "pilot": len(pilot),
        "held": len(held),
        "excluded": len(excluded),
    }
    if observed != EXPECTED_COUNTS or len(roles) != len(cells):
        raise PermissionError(f"metadata role counts differ: {observed}")
    return roles, cells, counts, donor_pool


def _selected_cells(
    cells: dict[str, list[str]], donors: Iterable[str]
) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for donor in donors:
        ordered = sorted(
            cells[donor],
            key=lambda cell: (
                hashlib.sha256(
                    f"{CELL_SELECTION_SALT}|{donor}|{cell}".encode()
                ).hexdigest(),
                cell,
            ),
        )
        output[donor] = ordered[:CELL_BUDGET]
    return output


def _feature_reference() -> dict[str, dict[str, str]]:
    with gzip.open(DEFAULT_FEATURE_REFERENCE, "rt", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_id = {row["id"]: row for row in rows}
    for markers, rna_features, adt_features in (
        (MARKERS, RNA_FEATURES, ADT_FEATURES),
        (BROAD_MARKERS, BROAD_RNA_FEATURES, BROAD_ADT_FEATURES),
    ):
        for marker, rna, adt in zip(markers, rna_features, adt_features):
            row = by_id.get(adt)
            if (
                row is None
                or row.get("target_gene_id") != rna
                or row.get("target_gene_name") != marker
            ):
                raise PermissionError(f"feature reference differs for {marker}")
    return by_id


def run_preflight(
    metadata_path: Path, cell_manifest: Path, output_path: Path
) -> dict[str, Any]:
    source = _read_json(DEFAULT_SOURCE_MANIFEST)
    designation = _read_json(DEFAULT_DESIGNATION)
    for record in source["files"].values():
        if "path" not in record:
            continue
        path = ROOT / record["path"]
        if path.exists() and (
            path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]
        ):
            raise PermissionError(f"source digest differs for {path.name}")
    _export_metadata(metadata_path, cell_manifest)
    rows = _metadata_rows(cell_manifest)
    roles, cells, counts, donor_pool = _designated_roles(rows, designation)
    selected = _selected_cells(cells, sorted(roles))
    _feature_reference()
    file_list = DEFAULT_FILE_LIST.read_text()
    expected_wells = {
        f"XHLT2-POOL-DB{pool}-SCG{well}" for pool in range(1, 8) for well in range(1, 5)
    }
    for well in expected_wells:
        for modality in ("GEX", "ADT"):
            if f"_{well}_{modality}.h5" not in file_list:
                raise PermissionError(f"file list is missing {well} {modality}")

    donors = []
    for donor in sorted(roles):
        donors.append(
            {
                "donor": donor,
                "pool": donor_pool[donor],
                "role": roles[donor],
                "metadata_cells": counts[donor],
                "wells": sorted({row["well"] for row in rows if row["donor"] == donor}),
                "selected_cell_axis_sha256": _axis_sha256(selected[donor]),
            }
        )
    payload = {
        "schema": "gse314416-citeseq-metadata-preflight/1.0",
        "status": "PASS",
        "created_at_utc": _timestamp(),
        "accession": "GSE314416",
        "access_audit": {
            "adt_numeric_values_read": 0,
            "gex_numeric_values_read": 0,
            "h5_files_opened": 0,
        },
        "source_bindings": {
            "source_manifest_sha256": _sha256(DEFAULT_SOURCE_MANIFEST),
            "designation_sha256": _sha256(DEFAULT_DESIGNATION),
            "metadata_sha256": _sha256(metadata_path),
            "feature_reference_sha256": _sha256(DEFAULT_FEATURE_REFERENCE),
            "file_list_sha256": _sha256(DEFAULT_FILE_LIST),
        },
        "assay": {
            "included_pools": [
                CALIBRATION_POOL,
                *HELD_POOLS[:0],
                "XHLT2-POOL-DB2",
                *HELD_POOLS,
            ],
            "chemistry": "10X 5-prime Next GEM v1.1",
            "single_dataset_wide_antibody_reference": True,
            "population": "healthy baseline participants",
            "pool_is_prediction_outcome": False,
        },
        "markers": [
            {"marker": marker, "rna_feature": rna, "adt_feature": adt}
            for marker, rna, adt in zip(MARKERS, RNA_FEATURES, ADT_FEATURES)
        ],
        "secondary_broad_markers": [
            {"marker": marker, "rna_feature": rna, "adt_feature": adt}
            for marker, rna, adt in zip(
                BROAD_MARKERS, BROAD_RNA_FEATURES, BROAD_ADT_FEATURES
            )
        ],
        "role_counts": {
            role: sum(value == role for value in roles.values())
            for role in ("calibration", "pilot", "held", "excluded")
        },
        "donors": donors,
    }
    _write_json(output_path, payload)
    return payload


def _validated_preflight(
    cell_manifest: Path,
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, list[str]]]:
    payload = _read_json(DEFAULT_PREFLIGHT)
    if (
        payload.get("status") != "PASS"
        or payload.get("access_audit", {}).get("h5_files_opened") != 0
        or payload.get("source_bindings", {}).get("source_manifest_sha256")
        != _sha256(DEFAULT_SOURCE_MANIFEST)
        or payload.get("source_bindings", {}).get("designation_sha256")
        != _sha256(DEFAULT_DESIGNATION)
    ):
        raise PermissionError("metadata preflight differs from the frozen pass")
    _export_metadata(DEFAULT_METADATA, cell_manifest)
    rows = _metadata_rows(cell_manifest)
    roles, cells, _, _ = _designated_roles(rows, _read_json(DEFAULT_DESIGNATION))
    selected = _selected_cells(cells, sorted(roles))
    frozen = {row["donor"]: row for row in payload["donors"]}
    for donor in roles:
        if frozen[donor]["role"] != roles[donor] or frozen[donor][
            "selected_cell_axis_sha256"
        ] != _axis_sha256(selected[donor]):
            raise PermissionError(f"cell selection differs for {donor}")
    return roles, selected, {row["cell_id"]: row for row in rows}


def _decode(values: np.ndarray) -> list[str]:
    return [
        value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


def _discover_h5(source_root: Path) -> dict[tuple[str, str], Path]:
    pattern = re.compile(r"(?:GSM\d+_)?(XHLT2-POOL-DB[1-7]-SCG[1-4])_(GEX|ADT)\.h5$")
    output: dict[tuple[str, str], Path] = {}
    for path in source_root.rglob("*.h5"):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        key = (match.group(1), match.group(2))
        if key in output:
            raise ValueError(f"duplicate source matrix for {key}")
        output[key] = path
    expected = {
        (f"XHLT2-POOL-DB{pool}-SCG{well}", modality)
        for pool in range(1, 8)
        for well in range(1, 5)
        for modality in ("GEX", "ADT")
    }
    if set(output) != expected:
        missing = sorted(expected - set(output))
        raise FileNotFoundError(f"baseline H5 set is incomplete: {missing[:4]}")
    return output


def _read_10x_columns(
    path: Path, barcodes: list[str], features: tuple[str, ...]
) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]
        axis = _decode(matrix["barcodes"][:])
        by_barcode = {value: index for index, value in enumerate(axis)}
        if len(by_barcode) != len(axis) or any(
            value not in by_barcode for value in barcodes
        ):
            raise ValueError(f"selected barcode axis differs in {path.name}")
        feature_axis = _decode(matrix["features"]["id"][:])
        by_feature = {value: index for index, value in enumerate(feature_axis)}
        if any(value not in by_feature for value in features):
            raise ValueError(f"frozen feature panel differs in {path.name}")
        local = {by_feature[value]: index for index, value in enumerate(features)}
        indptr = np.asarray(matrix["indptr"][:], dtype=np.int64)
        output = np.zeros((len(barcodes), len(features)), dtype=np.int64)
        for out_index, barcode in enumerate(barcodes):
            column = by_barcode[barcode]
            start, stop = int(indptr[column]), int(indptr[column + 1])
            indices = np.asarray(matrix["indices"][start:stop], dtype=np.int64)
            values = np.asarray(matrix["data"][start:stop])
            if np.any(values < 0) or not np.array_equal(values, np.rint(values)):
                raise ValueError(
                    f"raw count matrix is not nonnegative integer in {path.name}"
                )
            for feature_index, value in zip(indices, values):
                marker = local.get(int(feature_index))
                if marker is not None:
                    output[out_index, marker] = int(value)
    return output


def _read_selected_counts(
    donors: list[str],
    selected: dict[str, list[str]],
    cell_rows: dict[str, dict[str, str]],
    files: dict[tuple[str, str], Path],
    *,
    read_gex: bool,
    read_adt: bool,
    rna_features: tuple[str, ...] = RNA_FEATURES,
    adt_features: tuple[str, ...] = ADT_FEATURES,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    donor_position = {
        donor: {cell: index for index, cell in enumerate(selected[donor])}
        for donor in donors
    }
    requests: dict[str, list[tuple[str, int, str]]] = {}
    for donor in donors:
        for cell in selected[donor]:
            row = cell_rows[cell]
            well = row["well"]
            barcode = cell[len(well) + 1 :]
            requests.setdefault(well, []).append(
                (donor, donor_position[donor][cell], barcode)
            )
    gex = {
        donor: np.zeros((CELL_BUDGET, len(rna_features)), dtype=np.int64)
        for donor in donors
    }
    adt = {
        donor: np.zeros((CELL_BUDGET, len(adt_features)), dtype=np.int64)
        for donor in donors
    }
    for well, values in requests.items():
        barcodes = [value[2] for value in values]
        if read_gex:
            counts = _read_10x_columns(files[(well, "GEX")], barcodes, rna_features)
            for index, (donor, position, _) in enumerate(values):
                gex[donor][position] = counts[index]
        if read_adt:
            counts = _read_10x_columns(files[(well, "ADT")], barcodes, adt_features)
            for index, (donor, position, _) in enumerate(values):
                adt[donor][position] = counts[index]
    return gex, adt


def _adt_states(counts: np.ndarray, cells: list[str], donor: str) -> np.ndarray:
    values = np.asarray(counts)
    if values.shape != (CELL_BUDGET, len(MARKERS)):
        raise ValueError("ADT panel differs from the frozen 512 by 9 design")
    states = np.zeros(values.shape, dtype=np.uint8)
    for marker_index, marker in enumerate(MARKERS):
        order = sorted(
            range(CELL_BUDGET),
            key=lambda cell_index: (
                int(values[cell_index, marker_index]),
                hashlib.sha256(
                    f"{ADT_TIE_SALT}|{donor}|{marker}|{cells[cell_index]}".encode()
                ).hexdigest(),
                cells[cell_index],
            ),
        )
        states[np.asarray(order[CELL_BUDGET // 2 :], dtype=int), marker_index] = 1
    if not np.all(states.sum(axis=0) == CELL_BUDGET // 2):
        raise AssertionError("ADT median split does not have fixed margins")
    return states


def _destroyed_adt(states: np.ndarray, cells: list[str], donor: str) -> np.ndarray:
    order = sorted(
        range(CELL_BUDGET),
        key=lambda index: (
            hashlib.sha256(
                f"{DESTROYED_LINK_SALT}|{donor}|{cells[index]}".encode()
            ).hexdigest(),
            cells[index],
        ),
    )
    return np.asarray(states)[np.asarray(order, dtype=int)]


def _binary_tables(rna: np.ndarray, adt: np.ndarray) -> np.ndarray:
    first = np.asarray(rna)
    second = np.asarray(adt)
    if first.shape != (CELL_BUDGET, len(MARKERS)) or second.shape != first.shape:
        raise ValueError("binary state panels differ")
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


def _reduce_records(
    donors: list[str],
    selected: dict[str, list[str]],
    cell_rows: dict[str, dict[str, str]],
    files: dict[tuple[str, str], Path],
) -> dict[str, dict[str, Any]]:
    gex, adt = _read_selected_counts(
        donors, selected, cell_rows, files, read_gex=True, read_adt=True
    )
    output: dict[str, dict[str, Any]] = {}
    for donor in donors:
        rna_states = (gex[donor] > 0).astype(np.uint8)
        adt_states = _adt_states(adt[donor], selected[donor], donor)
        tables = _binary_tables(rna_states, adt_states)
        destroyed = _binary_tables(
            rna_states, _destroyed_adt(adt_states, selected[donor], donor)
        )
        output[donor] = {
            "tables": tables,
            "destroyed_tables": destroyed,
            "rna_profiles": rna_states.mean(axis=0),
            "adt_profiles": _adt_profiles(adt[donor]),
            "table_sha256": _array_sha256(tables),
            "destroyed_table_sha256": _array_sha256(destroyed),
        }
    return output


def _panel_adt_states(
    counts: np.ndarray, cells: list[str], donor: str, markers: tuple[str, ...]
) -> np.ndarray:
    values = np.asarray(counts)
    if values.shape != (CELL_BUDGET, len(markers)):
        raise ValueError("secondary ADT panel differs from the frozen design")
    states = np.zeros(values.shape, dtype=np.uint8)
    for marker_index, marker in enumerate(markers):
        order = sorted(
            range(CELL_BUDGET),
            key=lambda cell_index: (
                int(values[cell_index, marker_index]),
                hashlib.sha256(
                    f"{ADT_TIE_SALT}|secondary|{donor}|{marker}|{cells[cell_index]}".encode()
                ).hexdigest(),
                cells[cell_index],
            ),
        )
        states[np.asarray(order[CELL_BUDGET // 2 :], dtype=int), marker_index] = 1
    return states


def _panel_binary_tables(rna: np.ndarray, adt: np.ndarray) -> np.ndarray:
    first = np.asarray(rna)
    second = np.asarray(adt)
    if first.shape != second.shape or first.shape[0] != CELL_BUDGET:
        raise ValueError("secondary binary state panels differ")
    marker_count = first.shape[1]
    tables = np.zeros((marker_count, marker_count, 2, 2), dtype=np.int64)
    for first_marker in range(marker_count):
        for second_marker in range(marker_count):
            code = 2 * first[:, first_marker] + second[:, second_marker]
            tables[first_marker, second_marker] = np.bincount(
                code, minlength=4
            ).reshape(2, 2)
    return tables


def _reduce_broad_records(
    donors: list[str],
    selected: dict[str, list[str]],
    cell_rows: dict[str, dict[str, str]],
    files: dict[tuple[str, str], Path],
) -> dict[str, dict[str, Any]]:
    gex, adt = _read_selected_counts(
        donors,
        selected,
        cell_rows,
        files,
        read_gex=True,
        read_adt=True,
        rna_features=BROAD_RNA_FEATURES,
        adt_features=BROAD_ADT_FEATURES,
    )
    output: dict[str, dict[str, Any]] = {}
    for donor in donors:
        rna_states = (gex[donor] > 0).astype(np.uint8)
        adt_states = _panel_adt_states(
            adt[donor], selected[donor], donor, BROAD_MARKERS
        )
        tables = _panel_binary_tables(rna_states, adt_states)
        output[donor] = {
            "tables": tables,
            "table_sha256": _array_sha256(tables),
        }
    return output


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
        raise CouplingEstimationRefusal("fold-local marker profile has zero variance")
    standardized = (values - values.mean(axis=1, keepdims=True)) / scale[:, None]
    edges: set[tuple[int, int]] = set()
    for marker in range(len(MARKERS)):
        candidates = np.asarray(
            [value for value in range(len(MARKERS)) if value != marker]
        )
        distances = np.asarray(
            [
                np.linalg.norm(standardized[marker] - standardized[value])
                for value in candidates
            ]
        )
        order = candidates[np.lexsort((candidates, distances))]
        edges.update(tuple(sorted((marker, int(value)))) for value in order[:neighbors])
    incidence = np.zeros((len(MARKERS), len(edges)), dtype=float)
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
        first = second = np.eye(len(MARKERS), dtype=float)
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
        "gradient_norm": fit.gradient_norm,
        "scaled_gradient_norm": fit.scaled_gradient_norm,
        "schur_condition_number": fit.schur_condition_number,
        "theta_curvature_condition_number": fit.theta_curvature_condition_number,
        "iterations": fit.iterations,
        "rna_incidence_sha256": _array_sha256(first),
        "adt_incidence_sha256": _array_sha256(second),
    }


def _fit_common_effect(tables: np.ndarray) -> dict[str, Any]:
    identity = np.eye(len(MARKERS), dtype=float)
    fit = fit_structured_conditional_log_odds(
        np.asarray(tables, dtype=np.int64),
        identity,
        identity,
        initial_log_odds=np.zeros((len(MARKERS), len(MARKERS)), dtype=float),
        ridge_penalty=0.0,
        graph_penalty=0.0,
        minimum_informative_donors=2,
        maximum_condition_number=MAXIMUM_CONDITION_NUMBER,
        tolerance=1e-9,
    )
    threshold = 1e-7 + 1e-10 * float(np.asarray(tables).sum(axis=(-2, -1)).max())
    if fit.gradient_norm > threshold:
        raise CouplingEstimationRefusal(
            "common-effect fit misses the gradient certificate"
        )
    return {
        "population_log_odds": fit.log_odds,
        "gradient_norm": fit.gradient_norm,
        "condition_number": fit.condition_number,
        "iterations": fit.iterations,
    }


def _margins(tables: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(tables)
    return values.sum(axis=-1), values.sum(axis=-2)


def _predict_log_odds(
    log_odds: np.ndarray, rows: np.ndarray, columns: np.ndarray, alpha: float
) -> np.ndarray:
    output = np.empty((len(MARKERS), len(MARKERS), 2, 2), dtype=float)
    for index in np.ndindex((len(MARKERS), len(MARKERS))):
        output[index] = expected_binary_table_from_log_odds(
            float(alpha) * float(log_odds[index]), rows[index], columns[index]
        )
    return output


def _donor_loss(observed: np.ndarray, predicted: np.ndarray) -> float:
    truth = np.asarray(observed, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    support = _informative(truth)
    if np.count_nonzero(support) < MINIMUM_INFORMATIVE_ENTITIES:
        raise CouplingEstimationRefusal("donor has fewer than 64 informative entities")
    truth = truth[support]
    estimate = estimate[support]
    if not np.allclose(truth.sum(axis=-1), estimate.sum(axis=-1)) or not np.allclose(
        truth.sum(axis=-2), estimate.sum(axis=-2)
    ):
        raise FloatingPointError("prediction changed a recipient margin")
    positive = truth > 0.0
    if np.any(estimate[positive] <= 0.0) or not np.isfinite(estimate).all():
        raise FloatingPointError("prediction assigns invalid mass")
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
    epsilon = min(CLASSICAL_INVERSION_EPSILON, 0.25 * (upper - lower))
    left = lower + epsilon
    right = upper - epsilon

    def table_at(value: float) -> np.ndarray:
        return np.asarray(
            [
                [value, row_margin[0] - value],
                [column_margin[0] - value, row_margin[1] - column_margin[0] + value],
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
        raise CouplingEstimationRefusal("residual comparator lacks donor coverage")
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
        raise CouplingEstimationRefusal("residual comparator is nonfinite")
    return pooled.reshape(len(MARKERS), len(MARKERS))


def _predict_residual(
    pooled: np.ndarray, rows: np.ndarray, columns: np.ndarray, config: ResidualConfig
) -> np.ndarray:
    output = np.empty((len(MARKERS), len(MARKERS), 2, 2), dtype=float)
    for index in np.ndindex((len(MARKERS), len(MARKERS))):
        coordinate = (
            config.transport_multiplier * float(pooled[index]) * math.sqrt(CELL_BUDGET)
        )
        output[index] = _classical_table(
            coordinate, rows[index], columns[index], config.family
        )
    return output


def _panel_residual_pool(tables: np.ndarray, family: str) -> np.ndarray:
    values = np.asarray(tables)
    marker_count = values.shape[1]
    flat = values.reshape(len(values), -1, 2, 2)
    support = _informative(flat)
    if np.any(support.sum(axis=0) < 2):
        raise CouplingEstimationRefusal("secondary residual lacks donor coverage")
    coordinates = np.full(support.shape, np.nan, dtype=float)
    statistic = (
        signed_pearson_coordinate if family == "pearson" else signed_deviance_coordinate
    )
    for donor, entity in np.argwhere(support):
        coordinates[donor, entity] = statistic(flat[donor, entity]) / math.sqrt(
            CELL_BUDGET
        )
    pooled = np.nanmean(coordinates, axis=0)
    if not np.isfinite(pooled).all():
        raise CouplingEstimationRefusal("secondary residual is nonfinite")
    return pooled.reshape(marker_count, marker_count)


def _fit_broad_models(
    records: dict[str, dict[str, Any]], donors: list[str]
) -> dict[str, Any]:
    tables = np.asarray([records[donor]["tables"] for donor in donors])
    identity = np.eye(len(BROAD_MARKERS), dtype=float)
    fit = fit_hierarchical_conditional_log_odds(
        tables,
        identity,
        identity,
        heterogeneity_penalty=BROAD_HETEROGENEITY_PENALTY,
        ridge_penalty=BROAD_RIDGE_PENALTY,
        graph_penalty=0.0,
        minimum_informative_donors=2,
        maximum_condition_number=MAXIMUM_CONDITION_NUMBER,
    )
    residual = _panel_residual_pool(tables, BROAD_RESIDUAL_FAMILY)
    return {
        "scope": "secondary_broad_cognate_panel",
        "markers": list(BROAD_MARKERS),
        "exact": {
            "configuration": {
                "heterogeneity_penalty": BROAD_HETEROGENEITY_PENALTY,
                "ridge_penalty": BROAD_RIDGE_PENALTY,
                "graph_penalty": 0.0,
                "transport_multiplier": BROAD_TRANSPORT_MULTIPLIER,
            },
            "population_log_odds": fit.population_log_odds.tolist(),
            "fit_certificate": {
                "gradient_norm": fit.gradient_norm,
                "scaled_gradient_norm": fit.scaled_gradient_norm,
                "schur_condition_number": fit.schur_condition_number,
                "theta_curvature_condition_number": fit.theta_curvature_condition_number,
                "iterations": fit.iterations,
            },
        },
        "residual": {
            "configuration": {
                "family": BROAD_RESIDUAL_FAMILY,
                "transport_multiplier": BROAD_TRANSPORT_MULTIPLIER,
                "inversion_epsilon": CLASSICAL_INVERSION_EPSILON,
            },
            "pooled_coordinate": residual.tolist(),
        },
    }


def _predict_broad_models(
    models: dict[str, Any], rows: np.ndarray, columns: np.ndarray
) -> dict[str, np.ndarray]:
    marker_count = len(BROAD_MARKERS)
    exact = np.empty((marker_count, marker_count, 2, 2), dtype=float)
    residual = np.empty_like(exact)
    log_odds = np.asarray(models["exact"]["population_log_odds"])
    pooled = np.asarray(models["residual"]["pooled_coordinate"])
    for index in np.ndindex((marker_count, marker_count)):
        exact[index] = expected_binary_table_from_log_odds(
            BROAD_TRANSPORT_MULTIPLIER * float(log_odds[index]),
            rows[index],
            columns[index],
        )
        coordinate = (
            BROAD_TRANSPORT_MULTIPLIER * float(pooled[index]) * math.sqrt(CELL_BUDGET)
        )
        residual[index] = _classical_table(
            coordinate, rows[index], columns[index], BROAD_RESIDUAL_FAMILY
        )
    return {"exact": exact, "residual": residual}


def _panel_donor_loss(
    observed: np.ndarray, predicted: np.ndarray, minimum_informative: int
) -> float:
    truth = np.asarray(observed, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    support = _informative(truth)
    if np.count_nonzero(support) < minimum_informative:
        raise CouplingEstimationRefusal(
            "secondary donor has too few informative entities"
        )
    truth = truth[support]
    estimate = estimate[support]
    if not np.allclose(truth.sum(axis=-1), estimate.sum(axis=-1)) or not np.allclose(
        truth.sum(axis=-2), estimate.sum(axis=-2)
    ):
        raise FloatingPointError("secondary prediction changed a recipient margin")
    positive = truth > 0.0
    if np.any(estimate[positive] <= 0.0) or not np.isfinite(estimate).all():
        raise FloatingPointError("secondary prediction assigns invalid mass")
    terms = np.zeros_like(truth)
    terms[positive] = truth[positive] * np.log(truth[positive] / estimate[positive])
    return float((2.0 * terms.sum(axis=(-2, -1)) / CELL_BUDGET).mean())


def _broad_panel_losses(
    records: dict[str, dict[str, Any]], donors: list[str], models: dict[str, Any]
) -> dict[str, np.ndarray]:
    losses = {"exact": np.empty(len(donors)), "residual": np.empty(len(donors))}
    for donor_index, donor in enumerate(donors):
        truth = np.asarray(records[donor]["tables"])
        rows, columns = _margins(truth)
        predictions = _predict_broad_models(models, rows, columns)
        for method in losses:
            losses[method][donor_index] = _panel_donor_loss(
                truth, predictions[method], BROAD_MINIMUM_INFORMATIVE_ENTITIES
            )
    return losses


def _primary_configs() -> list[PrimaryConfig]:
    configs = []
    for neighbors, heterogeneity, ridge, graph, transport in product(
        NEIGHBOR_GRID,
        HETEROGENEITY_GRID,
        RIDGE_GRID,
        GRAPH_GRID,
        TRANSPORT_GRID,
    ):
        if graph == 0.0 and neighbors != NEIGHBOR_GRID[0]:
            continue
        configs.append(PrimaryConfig(neighbors, heterogeneity, ridge, graph, transport))
    return configs


def _records_arrays(
    records: dict[str, dict[str, Any]], donors: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([records[donor]["tables"] for donor in donors]),
        np.asarray([records[donor]["destroyed_tables"] for donor in donors]),
        np.asarray([records[donor]["rna_profiles"] for donor in donors]),
        np.asarray([records[donor]["adt_profiles"] for donor in donors]),
    )


def _select_calibration(
    records: dict[str, dict[str, Any]], calibration: list[str]
) -> dict[str, Any]:
    primary_configs = _primary_configs()
    residual_configs = [
        ResidualConfig(family, alpha)
        for family, alpha in product(RESIDUAL_FAMILIES, TRANSPORT_GRID)
    ]
    primary_losses = {
        config: np.full(len(calibration), np.nan) for config in primary_configs
    }
    residual_losses = {
        config: np.full(len(calibration), np.nan) for config in residual_configs
    }
    refusals: list[dict[str, Any]] = []
    for fold, held in enumerate(calibration):
        training = [donor for donor in calibration if donor != held]
        tables, _, rna_profiles, adt_profiles = _records_arrays(records, training)
        truth = np.asarray(records[held]["tables"])
        rows, columns = _margins(truth)
        structural: dict[
            tuple[int, float, float, float], dict[str, Any] | Exception
        ] = {}
        for config in primary_configs:
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
            fit = structural[key]
            if isinstance(fit, Exception):
                refusals.append(
                    {"fold": held, "configuration": asdict(config), "reason": str(fit)}
                )
                continue
            prediction = _predict_log_odds(
                fit["population_log_odds"], rows, columns, config.transport_multiplier
            )
            primary_losses[config][fold] = _donor_loss(truth, prediction)
        for config in residual_configs:
            try:
                pooled = _residual_pool(tables, config.family)
                residual_losses[config][fold] = _donor_loss(
                    truth, _predict_residual(pooled, rows, columns, config)
                )
            except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
                refusals.append(
                    {
                        "fold": held,
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
    primary = min(
        available_primary,
        key=lambda config: (float(primary_losses[config].mean()), config),
    )
    residual = min(
        available_residual,
        key=lambda config: (float(residual_losses[config].mean()), config),
    )
    graph_zero = min(
        (config for config in available_primary if config.graph_penalty == 0.0),
        key=lambda config: (float(primary_losses[config].mean()), config),
    )
    return {
        "primary": asdict(primary),
        "best_residual": asdict(residual),
        "graph_zero": asdict(graph_zero),
        "primary_candidates": [
            {
                "configuration": asdict(config),
                "fold_losses": {
                    donor: float(value)
                    for donor, value in zip(calibration, primary_losses[config])
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
                    for donor, value in zip(calibration, residual_losses[config])
                },
                "mean_loss": float(residual_losses[config].mean()),
            }
            for config in available_residual
        ],
        "refusals": refusals,
    }


def _fit_models(
    records: dict[str, dict[str, Any]], donors: list[str], selection: dict[str, Any]
) -> dict[str, Any]:
    tables, destroyed, rna_profiles, adt_profiles = _records_arrays(records, donors)
    primary_config = PrimaryConfig(**selection["primary"])
    graph_zero_config = PrimaryConfig(**selection["graph_zero"])
    residual_config = ResidualConfig(**selection["best_residual"])
    primary = _fit_primary(tables, rna_profiles, adt_profiles, primary_config)
    destroyed_fit = _fit_primary(destroyed, rna_profiles, adt_profiles, primary_config)
    graph_zero = _fit_primary(tables, rna_profiles, adt_profiles, graph_zero_config)
    residual = _residual_pool(tables, residual_config.family)
    models = {
        "primary": {
            "configuration": asdict(primary_config),
            "population_log_odds": primary.pop("population_log_odds").tolist(),
            "fit_certificate": primary,
        },
        "destroyed_link": {
            "configuration": asdict(primary_config),
            "population_log_odds": destroyed_fit.pop("population_log_odds").tolist(),
            "fit_certificate": destroyed_fit,
        },
        "best_residual": {
            "configuration": asdict(residual_config),
            "pooled_coordinate": residual.tolist(),
        },
        "graph_zero": {
            "configuration": asdict(graph_zero_config),
            "population_log_odds": graph_zero.pop("population_log_odds").tolist(),
            "fit_certificate": graph_zero,
        },
    }
    try:
        common = _fit_common_effect(tables)
        models["common_effect_cmle"] = {
            "configuration": {
                "ridge_penalty": 0.0,
                "graph_penalty": 0.0,
                "transport_multiplier": 1.0,
            },
            "population_log_odds": common.pop("population_log_odds").tolist(),
            "fit_certificate": common,
        }
    except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
        models["common_effect_cmle_refusal"] = str(error)
    return models


def _predict_models(
    models: dict[str, Any], rows: np.ndarray, columns: np.ndarray
) -> dict[str, np.ndarray]:
    primary = models["primary"]
    primary_config = PrimaryConfig(**primary["configuration"])
    destroyed = models["destroyed_link"]
    residual = models["best_residual"]
    graph_zero = models["graph_zero"]
    graph_zero_config = PrimaryConfig(**graph_zero["configuration"])
    output = {
        "primary": _predict_log_odds(
            np.asarray(primary["population_log_odds"]),
            rows,
            columns,
            primary_config.transport_multiplier,
        ),
        "best_residual": _predict_residual(
            np.asarray(residual["pooled_coordinate"]),
            rows,
            columns,
            ResidualConfig(**residual["configuration"]),
        ),
        "destroyed_link": _predict_log_odds(
            np.asarray(destroyed["population_log_odds"]),
            rows,
            columns,
            primary_config.transport_multiplier,
        ),
        "graph_zero": _predict_log_odds(
            np.asarray(graph_zero["population_log_odds"]),
            rows,
            columns,
            graph_zero_config.transport_multiplier,
        ),
    }
    if "common_effect_cmle" in models:
        common = models["common_effect_cmle"]
        output["common_effect_cmle"] = _predict_log_odds(
            np.asarray(common["population_log_odds"]), rows, columns, 1.0
        )
    return output


def _panel_losses(
    records: dict[str, dict[str, Any]], donors: list[str], models: dict[str, Any]
) -> dict[str, np.ndarray]:
    methods = ["primary", "best_residual", "destroyed_link", "graph_zero"]
    if "common_effect_cmle" in models:
        methods.append("common_effect_cmle")
    losses = {method: np.empty(len(donors)) for method in methods}
    for donor_index, donor in enumerate(donors):
        truth = np.asarray(records[donor]["tables"])
        rows, columns = _margins(truth)
        predictions = _predict_models(models, rows, columns)
        for method in losses:
            losses[method][donor_index] = _donor_loss(truth, predictions[method])
    return losses


def _comparison(
    donors: list[str],
    primary: np.ndarray,
    comparator: np.ndarray,
    required: int,
    indices: np.ndarray,
) -> dict[str, Any]:
    difference = np.asarray(primary) - np.asarray(comparator)
    bootstrap = difference[indices].mean(axis=1)
    interval = np.quantile(bootstrap, [0.025, 0.975], method="linear")
    relative = 1.0 - float(np.mean(primary) / np.mean(comparator))
    favorable = int(np.count_nonzero(difference < 0.0))
    checks = {
        "relative_deviance_reduction_at_least_five_percent": bool(relative >= 0.05),
        "paired_bootstrap_upper_95_below_zero": bool(interval[1] < 0.0),
        "favorable_donor_count_reached": bool(favorable >= required),
    }
    return {
        "primary_mean_loss": float(np.mean(primary)),
        "comparator_mean_loss": float(np.mean(comparator)),
        "relative_deviance_reduction": relative,
        "mean_paired_difference": float(difference.mean()),
        "paired_bootstrap_95_interval": interval.tolist(),
        "favorable_donors": favorable,
        "required_favorable_donors": required,
        "checks": checks,
        "passes": all(checks.values()),
        "donor_differences": {
            donor: float(value) for donor, value in zip(donors, difference)
        },
    }


def _gate(
    donors: list[str], losses: dict[str, np.ndarray], required: int
) -> dict[str, Any]:
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(donors), size=(BOOTSTRAPS, len(donors))
    )
    residual = _comparison(
        donors, losses["primary"], losses["best_residual"], required, indices
    )
    destroyed = _comparison(
        donors, losses["primary"], losses["destroyed_link"], required, indices
    )
    return {
        "primary_vs_selected_residual": residual,
        "primary_vs_destroyed_link": destroyed,
        "passes": residual["passes"] and destroyed["passes"],
    }


def run_development(
    source_root: Path, cell_manifest: Path, output_path: Path
) -> dict[str, Any]:
    protocol_commit = _require_public_tag(PROTOCOL_TAG, PROTOCOL_BINDINGS)
    roles, selected, cell_rows = _validated_preflight(cell_manifest)
    calibration = sorted(
        donor for donor, role in roles.items() if role == "calibration"
    )
    pilot = sorted(donor for donor, role in roles.items() if role == "pilot")
    files = _discover_h5(source_root)
    records = _reduce_records(calibration + pilot, selected, cell_rows, files)
    broad_records = _reduce_broad_records(
        calibration + pilot, selected, cell_rows, files
    )
    selection = _select_calibration(records, calibration)
    calibration_models = _fit_models(records, calibration, selection)
    pilot_losses = _panel_losses(records, pilot, calibration_models)
    pilot_gate = _gate(pilot, pilot_losses, PILOT_REQUIRED_FAVORABLE)
    try:
        broad_calibration_models = _fit_broad_models(broad_records, calibration)
        broad_pilot_losses = _broad_panel_losses(
            broad_records, pilot, broad_calibration_models
        )
        broad_indices = np.random.default_rng(BOOTSTRAP_SEED).integers(
            0, len(pilot), size=(BOOTSTRAPS, len(pilot))
        )
        broad_pilot = {
            "status": "SECONDARY_EVALUATED",
            "comparison": _comparison(
                pilot,
                broad_pilot_losses["exact"],
                broad_pilot_losses["residual"],
                PILOT_REQUIRED_FAVORABLE,
                broad_indices,
            ),
            "losses": {
                method: {donor: float(value) for donor, value in zip(pilot, values)}
                for method, values in broad_pilot_losses.items()
            },
        }
    except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
        broad_pilot = {
            "status": "SECONDARY_REFUSAL",
            "reason": str(error),
        }
    payload: dict[str, Any] = {
        "schema": "gse314416-citeseq-development/1.0",
        "status": "PILOT_PASS" if pilot_gate["passes"] else "TERMINAL_PILOT_REFUSAL",
        "created_at_utc": _timestamp(),
        "protocol_commit": protocol_commit,
        "protocol_tag": PROTOCOL_TAG,
        "calibration_donors": calibration,
        "pilot_donors": pilot,
        "selection": selection,
        "pilot_losses": {
            method: {donor: float(value) for donor, value in zip(pilot, values)}
            for method, values in pilot_losses.items()
        },
        "pilot_gate": pilot_gate,
        "secondary_broad_panel": broad_pilot,
        "development_table_sha256": {
            donor: records[donor]["table_sha256"] for donor in calibration + pilot
        },
    }
    if pilot_gate["passes"]:
        payload["all_development_models"] = _fit_models(
            records, calibration + pilot, selection
        )
        try:
            payload["secondary_broad_all_development_models"] = _fit_broad_models(
                broad_records, calibration + pilot
            )
        except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
            payload["secondary_broad_all_development_refusal"] = str(error)
    _write_json(output_path, payload)
    return payload


def _panel_held_margins(
    rna_counts: np.ndarray, marker_count: int
) -> tuple[np.ndarray, np.ndarray]:
    positives = (np.asarray(rna_counts) > 0).sum(axis=0)
    rows = np.repeat(
        np.stack([CELL_BUDGET - positives, positives], axis=1)[:, None, :],
        marker_count,
        axis=1,
    )
    columns = np.broadcast_to(
        np.asarray([CELL_BUDGET // 2, CELL_BUDGET // 2]),
        (marker_count, marker_count, 2),
    ).copy()
    return rows, columns


def _held_margins(rna_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _panel_held_margins(rna_counts, len(MARKERS))


def run_prediction(
    source_root: Path, cell_manifest: Path, output_path: Path
) -> dict[str, Any]:
    development_commit = _require_public_tag(
        DEVELOPMENT_TAG, (*PROTOCOL_BINDINGS, _relative(DEFAULT_DEVELOPMENT))
    )
    development = _read_json(DEFAULT_DEVELOPMENT)
    if development.get("status") != "PILOT_PASS" or not development.get(
        "pilot_gate", {}
    ).get("passes"):
        raise PermissionError("development did not pass the frozen pilot gate")
    roles, selected, cell_rows = _validated_preflight(cell_manifest)
    held = sorted(donor for donor, role in roles.items() if role == "held")
    files = _discover_h5(source_root)
    gex, _ = _read_selected_counts(
        held, selected, cell_rows, files, read_gex=True, read_adt=False
    )
    broad_models = development.get("secondary_broad_all_development_models")
    broad_gex: dict[str, np.ndarray] = {}
    if broad_models is not None:
        broad_gex, _ = _read_selected_counts(
            held,
            selected,
            cell_rows,
            files,
            read_gex=True,
            read_adt=False,
            rna_features=BROAD_RNA_FEATURES,
            adt_features=BROAD_ADT_FEATURES,
        )
    samples = []
    for donor in held:
        rows, columns = _held_margins(gex[donor])
        predictions = _predict_models(
            development["all_development_models"], rows, columns
        )
        sample = {
            "donor": donor,
            "row_margins": rows.tolist(),
            "column_margins": columns.tolist(),
            "predicted_tables": {
                method: value.tolist() for method, value in predictions.items()
            },
            "prediction_sha256": {
                method: _array_sha256(value) for method, value in predictions.items()
            },
        }
        if broad_models is not None:
            broad_rows, broad_columns = _panel_held_margins(
                broad_gex[donor], len(BROAD_MARKERS)
            )
            broad_predictions = _predict_broad_models(
                broad_models, broad_rows, broad_columns
            )
            sample["secondary_broad"] = {
                "row_margins": broad_rows.tolist(),
                "column_margins": broad_columns.tolist(),
                "predicted_tables": {
                    method: value.tolist()
                    for method, value in broad_predictions.items()
                },
                "prediction_sha256": {
                    method: _array_sha256(value)
                    for method, value in broad_predictions.items()
                },
            }
        samples.append(sample)
    payload = {
        "schema": "gse314416-citeseq-held-predictions/1.0",
        "status": "HELD_PREDICTIONS_FROZEN_WITHOUT_ADT_ACCESS",
        "created_at_utc": _timestamp(),
        "development_commit": development_commit,
        "development_sha256": _sha256(DEFAULT_DEVELOPMENT),
        "held_donors": held,
        "held_access_audit": {
            "gex_numeric_values_read": "frozen primary and secondary marker rows for selected cells",
            "adt_h5_files_opened": 0,
            "adt_numeric_values_read": 0,
        },
        "samples": samples,
    }
    _write_json(output_path, payload)
    return payload


def run_score(
    source_root: Path, cell_manifest: Path, output_path: Path
) -> dict[str, Any]:
    prediction_commit = _require_public_tag(
        PREDICTION_TAG,
        (
            *PROTOCOL_BINDINGS,
            _relative(DEFAULT_DEVELOPMENT),
            _relative(DEFAULT_PREDICTION),
        ),
    )
    predictions = _read_json(DEFAULT_PREDICTION)
    if predictions.get("status") != "HELD_PREDICTIONS_FROZEN_WITHOUT_ADT_ACCESS":
        raise PermissionError("held predictions are not frozen")
    roles, selected, cell_rows = _validated_preflight(cell_manifest)
    held = sorted(donor for donor, role in roles.items() if role == "held")
    if held != predictions["held_donors"]:
        raise PermissionError("held donor axis differs from frozen predictions")
    files = _discover_h5(source_root)
    records = _reduce_records(held, selected, cell_rows, files)
    frozen = {sample["donor"]: sample for sample in predictions["samples"]}
    has_broad = all("secondary_broad" in frozen[donor] for donor in held)
    broad_records = (
        _reduce_broad_records(held, selected, cell_rows, files) if has_broad else {}
    )
    loss_methods = ["primary", "best_residual", "destroyed_link", "graph_zero"]
    if all("common_effect_cmle" in frozen[donor]["predicted_tables"] for donor in held):
        loss_methods.append("common_effect_cmle")
    losses = {method: np.empty(len(held)) for method in loss_methods}
    broad_losses = (
        {"exact": np.empty(len(held)), "residual": np.empty(len(held))}
        if has_broad
        else None
    )
    samples = []
    for donor_index, donor in enumerate(held):
        truth = np.asarray(records[donor]["tables"])
        rows, columns = _margins(truth)
        sample = frozen[donor]
        if (
            rows.tolist() != sample["row_margins"]
            or columns.tolist() != sample["column_margins"]
        ):
            raise PermissionError("held truth margins differ from frozen prediction")
        donor_losses = {}
        for method in losses:
            prediction = np.asarray(sample["predicted_tables"][method])
            if _array_sha256(prediction) != sample["prediction_sha256"][method]:
                raise PermissionError("held prediction hash differs")
            loss = _donor_loss(truth, prediction)
            losses[method][donor_index] = loss
            donor_losses[method] = float(loss)
        result_sample: dict[str, Any] = {
            "donor": donor,
            "losses": donor_losses,
            "truth_table_sha256": records[donor]["table_sha256"],
        }
        if has_broad and broad_losses is not None:
            broad_truth = np.asarray(broad_records[donor]["tables"])
            broad_rows, broad_columns = _margins(broad_truth)
            broad_frozen = sample["secondary_broad"]
            if (
                broad_rows.tolist() != broad_frozen["row_margins"]
                or broad_columns.tolist() != broad_frozen["column_margins"]
            ):
                raise PermissionError(
                    "secondary held truth margins differ from frozen prediction"
                )
            donor_broad_losses = {}
            for method in broad_losses:
                broad_prediction = np.asarray(broad_frozen["predicted_tables"][method])
                if (
                    _array_sha256(broad_prediction)
                    != broad_frozen["prediction_sha256"][method]
                ):
                    raise PermissionError("secondary held prediction hash differs")
                broad_loss = _panel_donor_loss(
                    broad_truth,
                    broad_prediction,
                    BROAD_MINIMUM_INFORMATIVE_ENTITIES,
                )
                broad_losses[method][donor_index] = broad_loss
                donor_broad_losses[method] = float(broad_loss)
            result_sample["secondary_broad"] = {
                "losses": donor_broad_losses,
                "truth_table_sha256": broad_records[donor]["table_sha256"],
            }
        samples.append(result_sample)
    gate = _gate(held, losses, HELD_REQUIRED_FAVORABLE)
    broad_result: dict[str, Any]
    if has_broad and broad_losses is not None:
        broad_indices = np.random.default_rng(BOOTSTRAP_SEED).integers(
            0, len(held), size=(BOOTSTRAPS, len(held))
        )
        broad_result = {
            "status": "SECONDARY_EVALUATED",
            "comparison": _comparison(
                held,
                broad_losses["exact"],
                broad_losses["residual"],
                HELD_REQUIRED_FAVORABLE,
                broad_indices,
            ),
            "losses": {
                method: {donor: float(value) for donor, value in zip(held, values)}
                for method, values in broad_losses.items()
            },
        }
    else:
        broad_result = {"status": "SECONDARY_NOT_PREDICTED_AFTER_DEVELOPMENT_REFUSAL"}
    payload = {
        "schema": "gse314416-citeseq-confirmation/1.0",
        "status": "CONFIRMATION_PASS" if gate["passes"] else "CONFIRMATION_FAIL",
        "created_at_utc": _timestamp(),
        "prediction_commit": prediction_commit,
        "prediction_sha256": _sha256(DEFAULT_PREDICTION),
        "held_donors": held,
        "held_pools": list(HELD_POOLS),
        "samples": samples,
        "losses": {
            method: {donor: float(value) for donor, value in zip(held, values)}
            for method, values in losses.items()
        },
        "gate": gate,
        "secondary_broad_panel": broad_result,
    }
    _write_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "develop", "predict", "score"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument(
            "--cell-manifest", type=Path, default=DEFAULT_CELL_MANIFEST
        )
        if command == "preflight":
            subparser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
            subparser.add_argument("--output", type=Path, default=DEFAULT_PREFLIGHT)
        else:
            subparser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
            default = {
                "develop": DEFAULT_DEVELOPMENT,
                "predict": DEFAULT_PREDICTION,
                "score": DEFAULT_SCORE,
            }[command]
            subparser.add_argument("--output", type=Path, default=default)
    args = parser.parse_args()
    if args.command == "preflight":
        payload = run_preflight(args.metadata, args.cell_manifest, args.output)
    elif args.command == "develop":
        payload = run_development(args.source_root, args.cell_manifest, args.output)
    elif args.command == "predict":
        payload = run_prediction(args.source_root, args.cell_manifest, args.output)
    else:
        payload = run_score(args.source_root, args.cell_manifest, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
