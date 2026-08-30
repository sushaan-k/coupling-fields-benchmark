"""Metadata-only preflight for the GSE252762 celiac CITE-seq holdout."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


ACCESSION = "GSE252762"
BASE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE252nnn/GSE252762/suppl"
)
CELL_BUDGET = 256
CELL_SELECTION_SALT = "GSE252762-CELIAC-CELL-v1"
MARKERS = (
    ("CD3D", "CD3"),
    ("CD4", "CD4"),
    ("CD8A", "CD8"),
    ("CD27", "CD27"),
    ("CD38", "CD38"),
    ("CD44", "CD44"),
    ("CD69", "CD69"),
    ("ITGAE", "CD103"),
    ("KLRB1", "CD161"),
)

CALIBRATION = (
    "B3:ACD 5",
    "B1:ACD 1",
    "B1:GFD 2",
    "B1:GFD 3",
    "B1:ACD 2",
    "B3:GFD 6",
    "B3:ACD 7",
    "B3:CONTROL 2",
    "B5:CONTROL 10",
)
PILOT = (
    "B2:ACD 4",
    "B3:GFD 5",
    "B3:ACD 8",
    "B3:ACD 6",
    "B3:GFD 7",
    "B5:CONTROL 9",
    "B4:CONTROL 5",
)
HELD = (
    "B6:ACD 10",
    "B6:ACD 9",
    "B6:CONTROL 12",
    "B6:CONTROL 13",
    "B6:CONTROL 15",
    "B6:CONTROL 16",
    "B6:GFD 10",
    "B6:GFD 11",
    "B6:GFD 12",
    "B6:GFD 13",
    "B6:GFD 14",
    "B6:GFD 8",
    "B6:GFD 9",
)

MATRIX_BYTES = {
    (1, "rna"): 25_037_581,
    (1, "cite"): 1_797_642,
    (2, "rna"): 2_298_154,
    (2, "cite"): 192_178,
    (3, "rna"): 21_764_331,
    (3, "cite"): 2_282_678,
    (4, "rna"): 22_264_316,
    (4, "cite"): 1_343_388,
    (5, "rna"): 13_302_986,
    (5, "cite"): 1_151_493,
    (6, "rna"): 89_728_625,
    (6, "cite"): 4_952_984,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_lines(path: Path) -> tuple[str, ...]:
    with gzip.open(path, "rt", newline="") as stream:
        return tuple(line.rstrip("\r\n") for line in stream)


def _read_features(path: Path) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(line.split("\t")) for line in _read_lines(path))


def _read_metadata(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"", "batch", "location", "sample", "condition"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path.name} lacks the required metadata columns")
    return rows


def _marker_index(
    features: tuple[tuple[str, ...], ...], marker: str, label: str
) -> int:
    matches = [
        index
        for index, row in enumerate(features, start=1)
        if marker in row[:2]
    ]
    if len(matches) != 1:
        raise ValueError(f"{label} marker {marker} is not unique")
    return matches[0]


def _selected_cells(
    rows: list[dict[str, str]], sample: str, batch: int
) -> tuple[tuple[str, int], ...]:
    candidates = [
        (row[""], index)
        for index, row in enumerate(rows, start=1)
        if row["sample"] == sample
    ]
    if len(candidates) < CELL_BUDGET:
        raise ValueError(f"B{batch}:{sample} has fewer than {CELL_BUDGET} cells")
    if len({barcode for barcode, _ in candidates}) != len(candidates):
        raise ValueError(f"B{batch}:{sample} contains duplicate barcodes")

    def key(item: tuple[str, int]) -> tuple[bytes, str]:
        barcode, _ = item
        payload = f"{CELL_SELECTION_SALT}\0B{batch}:{sample}\0{barcode}".encode()
        return hashlib.sha256(payload).digest(), barcode

    return tuple(sorted(candidates, key=key)[:CELL_BUDGET])


def build_preflight(axis_dir: Path) -> dict[str, Any]:
    roles = {
        **{sample: "calibration" for sample in CALIBRATION},
        **{sample: "pilot" for sample in PILOT},
        **{sample: "held" for sample in HELD},
    }
    if len(roles) != len(CALIBRATION) + len(PILOT) + len(HELD):
        raise AssertionError("sample roles overlap")

    batches: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    reference_rna_features: tuple[tuple[str, ...], ...] | None = None
    reference_cite_features: tuple[tuple[str, ...], ...] | None = None

    for batch in range(1, 7):
        prefix = f"GSE252762_batch{batch}"
        paths = {
            kind: axis_dir / f"{prefix}_{kind}.gz"
            for kind in (
                "metadata.csv",
                "rna_barcodes.tsv",
                "cite_barcodes.tsv",
                "rna_features.tsv",
                "cite_features.tsv",
            )
        }
        missing = [path.name for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(", ".join(missing))

        metadata = _read_metadata(paths["metadata.csv"])
        rna_barcodes = _read_lines(paths["rna_barcodes.tsv"])
        cite_barcodes = _read_lines(paths["cite_barcodes.tsv"])
        rna_features = _read_features(paths["rna_features.tsv"])
        cite_features = _read_features(paths["cite_features.tsv"])
        if rna_barcodes != cite_barcodes:
            raise ValueError(f"batch {batch} RNA and CITE barcodes differ")
        metadata_barcodes = tuple(row[""] for row in metadata)
        if metadata_barcodes != rna_barcodes:
            raise ValueError(f"batch {batch} metadata and matrix barcodes differ")
        if len(set(rna_barcodes)) != len(rna_barcodes):
            raise ValueError(f"batch {batch} barcodes are not unique")
        if reference_rna_features is None:
            reference_rna_features = rna_features
            reference_cite_features = cite_features
        elif (
            rna_features != reference_rna_features
            or cite_features != reference_cite_features
        ):
            raise ValueError("feature axes differ across batches")

        by_sample: dict[str, list[dict[str, str]]] = {}
        for row in metadata:
            by_sample.setdefault(row["sample"], []).append(row)
        for sample, current in sorted(by_sample.items()):
            identity = f"B{batch}:{sample}"
            conditions = {row["condition"] for row in current}
            locations = {row["location"] for row in current}
            eligible = (
                len(current) >= CELL_BUDGET
                and conditions <= {"ACD", "GFD", "CONTROL"}
                and locations == {"biopsy"}
            )
            if eligible != (identity in roles):
                raise ValueError(f"eligibility and frozen role disagree for {identity}")
            if not eligible:
                continue
            if len(conditions) != 1:
                raise ValueError(f"{identity} has inconsistent condition labels")
            chosen = _selected_cells(metadata, sample, batch)
            condition = next(iter(conditions))
            samples.append(
                {
                    "sample_id": identity,
                    "batch": batch,
                    "deposited_sample": sample,
                    "condition": condition,
                    "context": "CONTROL" if condition == "CONTROL" else "CELIAC",
                    "role": roles[identity],
                    "available_cells": len(current),
                    "selected_barcodes": [barcode for barcode, _ in chosen],
                    "selected_columns_1_based": [index for _, index in chosen],
                }
            )

        batch_files = []
        for kind, path in paths.items():
            batch_files.append(
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "url": f"{BASE_URL}/{path.name}",
                }
            )
        for modality in ("rna", "cite"):
            name = f"{prefix}_{modality}_matrix.mtx.gz"
            batch_files.append(
                {
                    "name": name,
                    "bytes": MATRIX_BYTES[(batch, modality)],
                    "sha256": None,
                    "url": f"{BASE_URL}/{name}",
                    "numeric_outcome": True,
                }
            )
        batches.append(
            {
                "batch": batch,
                "rna_shape": [len(rna_features), len(rna_barcodes)],
                "cite_shape": [len(cite_features), len(cite_barcodes)],
                "files": batch_files,
            }
        )

    observed = {sample["sample_id"] for sample in samples}
    if observed != set(roles):
        raise ValueError("frozen sample set does not match metadata eligibility")
    assert reference_rna_features is not None
    assert reference_cite_features is not None
    marker_rows = [
        {
            "rna": rna,
            "rna_row_1_based": _marker_index(reference_rna_features, rna, "RNA"),
            "adt": adt,
            "cite_row_1_based": _marker_index(reference_cite_features, adt, "CITE"),
        }
        for rna, adt in MARKERS
    ]

    role_order = {"calibration": 0, "pilot": 1, "held": 2}
    samples.sort(key=lambda row: (role_order[row["role"]], row["sample_id"]))
    return {
        "schema": "gse252762-celiac-metadata-preflight/1.0",
        "accession": ACCESSION,
        "status": "PASS",
        "numeric_matrix_gets": 0,
        "cell_budget": CELL_BUDGET,
        "cell_selection_salt": CELL_SELECTION_SALT,
        "metadata_columns_used": ["barcode", "batch", "location", "sample", "condition"],
        "ignored_metadata_fields_include": ["nCount_RNA", "nCount_CITE"],
        "markers": marker_rows,
        "role_counts": {"calibration": 9, "pilot": 7, "held": 13},
        "batches": batches,
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_preflight(args.axis_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
