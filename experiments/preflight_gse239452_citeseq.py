"""Metadata-only preflight for the prospective GSE239452 CITE-seq study."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/confirmation/gse239452_citeseq"
DEFAULT_MANIFEST = DATA_DIR / "source_manifest_v1.json"
DEFAULT_SOURCE_ROOT = DATA_DIR / "source_cache"
DEFAULT_PANEL = DEFAULT_SOURCE_ROOT / "GSE239449_TotalSeqC_Annotated.xlsx"
DEFAULT_OUTPUT = ROOT / "results/development/gse239452_citeseq_metadata_preflight.json"

EXPECTED_ROLES = {"calibration": 7, "pilot": 8, "held": 9, "excluded_metadata": 2}
EXPECTED_EXCLUSIONS = {"100", "78"}
EXPECTED_MARKERS = (
    "CD4",
    "CD7",
    "CD14",
    "CD19",
    "CD33",
    "CD38",
    "CD44",
    "CD47",
    "CD52",
)
CELL_BUDGET = 512


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


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _axis_sha256(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("source manifest must contain one JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(text)


def _archive_url(accession: str, filename: str, template: str) -> str:
    return template.format(accession=accession, filename=filename)


def _validate_manifest(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if payload.get("schema") != "gse239452-citeseq-source-manifest/1.0":
        raise ValueError("unexpected source-manifest schema")
    if (
        payload.get("accession") != "GSE239452"
        or payload.get("cell_budget") != CELL_BUDGET
    ):
        raise ValueError("source manifest changes the frozen study or cell budget")
    samples = payload.get("samples")
    markers = payload.get("markers")
    if not isinstance(samples, list) or not isinstance(markers, list):
        raise ValueError("source manifest lacks samples or markers")
    if tuple(row.get("marker") for row in markers) != EXPECTED_MARKERS:
        raise ValueError("marker order differs from the frozen panel")
    if len(samples) != 26 or len({row.get("donor") for row in samples}) != 26:
        raise ValueError("source manifest must enumerate 26 unique study donors")
    counts = {
        role: sum(row.get("role") == role for row in samples) for role in EXPECTED_ROLES
    }
    if counts != EXPECTED_ROLES:
        raise ValueError("source-manifest role counts differ from the freeze")
    if {
        row["donor"] for row in samples if row.get("role") == "excluded_metadata"
    } != EXPECTED_EXCLUSIONS:
        raise ValueError("prospective metadata exclusions differ from the freeze")
    for row in samples:
        if row.get("pregnancy") not in {"Pregnant", "NonPregnant"}:
            raise ValueError("invalid pregnancy label")
        if row.get("severity") not in {"Control", "Asymptomatic", "Severe"}:
            raise ValueError("invalid severity label")
        if not isinstance(row.get("gex"), dict):
            raise ValueError("every study donor must have one GEX record")
        if row["donor"] == "78":
            if row.get("adt") is not None:
                raise ValueError("donor 78 must remain excluded for missing paired ADT")
        elif not isinstance(row.get("adt"), dict):
            raise ValueError("every donor other than 78 must have one ADT record")
    eligible = [row for row in samples if row["role"] != "excluded_metadata"]
    held = [row for row in eligible if row["role"] == "held"]
    held_counts = {
        severity: sum(row["severity"] == severity for row in held)
        for severity in ("Control", "Asymptomatic", "Severe")
    }
    if held_counts != {"Control": 3, "Asymptomatic": 3, "Severe": 3}:
        raise ValueError("held pregnancy panel is not the frozen balanced 3/3/3 panel")
    return samples, markers


def _text_axis(handle: h5py.File, path: str, access_log: list[str]) -> list[str]:
    if path not in handle or not isinstance(handle[path], h5py.Dataset):
        raise ValueError(f"required text axis {path} is absent")
    dataset = handle[path]
    if h5py.check_string_dtype(dataset.dtype) is None:
        raise ValueError(f"required text axis {path} is not a string dataset")
    access_log.append(path)
    return [str(value) for value in dataset.asstr()[:].tolist()]


def _raw_shape(handle: h5py.File) -> tuple[int, int]:
    if "raw/X" not in handle:
        raise ValueError("H5AD lacks raw/X")
    matrix = handle["raw/X"]
    if isinstance(matrix, h5py.Dataset):
        if len(matrix.shape) != 2:
            raise ValueError("dense raw/X is not two-dimensional")
        return int(matrix.shape[0]), int(matrix.shape[1])
    encoding = matrix.attrs.get("encoding-type")
    if isinstance(encoding, bytes):
        encoding = encoding.decode()
    if encoding not in {"csr_matrix", "csc_matrix"}:
        raise ValueError("raw/X is not a supported AnnData matrix")
    shape = np.asarray(matrix.attrs.get("shape"), dtype=np.int64)
    if shape.shape != (2,) or np.any(shape <= 0):
        raise ValueError("raw/X has no valid declared shape")
    for name in ("data", "indices", "indptr"):
        if name not in matrix or not isinstance(matrix[name], h5py.Dataset):
            raise ValueError(f"raw/X lacks structural dataset {name}")
    expected_indptr = int(shape[0] + 1 if encoding == "csr_matrix" else shape[1] + 1)
    if matrix["indptr"].shape != (expected_indptr,):
        raise ValueError("raw/X indptr shape contradicts its declared layout")
    if matrix["data"].shape != matrix["indices"].shape:
        raise ValueError("raw/X data and indices shapes differ")
    return int(shape[0]), int(shape[1])


def _canonical_gex_barcode(value: str) -> str:
    nonpregnant = re.fullmatch(r"(.+-1)-[0-9]+-[01]", value)
    if nonpregnant is not None:
        return nonpregnant.group(1)
    if value.endswith("-Pregnant"):
        return value[: -len("-Pregnant")]
    raise ValueError("GEX barcode does not match either frozen deposited encoding")


def _feature_indices(axis: list[str], expected: list[str], label: str) -> list[int]:
    if len(axis) != len(set(axis)):
        raise ValueError(f"{label} feature axis is not unique")
    positions = {value: index for index, value in enumerate(axis)}
    missing = [value for value in expected if value not in positions]
    if missing:
        raise ValueError(f"{label} misses exact frozen features: {missing}")
    return [positions[value] for value in expected]


def _inspect_gex(
    path: Path, markers: list[dict[str, Any]], access_log: list[str]
) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        index = handle["obs"].attrs.get("_index") if "obs" in handle else None
        if isinstance(index, bytes):
            index = index.decode()
        if index not in {"barcodekey", "_index"}:
            raise ValueError("GEX obs index is not an official deposited barcode axis")
        barcodes = _text_axis(handle, f"obs/{index}", access_log)
        features = _text_axis(handle, "raw/var/featurekey", access_log)
        shape = _raw_shape(handle)
    if shape != (len(barcodes), len(features)):
        raise ValueError("GEX raw matrix shape differs from its text axes")
    canonical = [_canonical_gex_barcode(value) for value in barcodes]
    if len(canonical) != len(set(canonical)):
        raise ValueError("canonical GEX barcode axis is not unique")
    positions = _feature_indices(
        features, [str(row["rna_feature"]) for row in markers], "GEX raw"
    )
    return {
        "raw_shape": list(shape),
        "obs_index": index,
        "barcodes": canonical,
        "barcode_axis_sha256": _axis_sha256(canonical),
        "marker_indices": positions,
        "feature_axis_sha256": _axis_sha256(features),
    }


def _inspect_adt(
    path: Path, markers: list[dict[str, Any]], access_log: list[str]
) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        index = handle["obs"].attrs.get("_index") if "obs" in handle else None
        if isinstance(index, bytes):
            index = index.decode()
        if index != "_index":
            raise ValueError("ADT obs index is not the official _index axis")
        barcodes = _text_axis(handle, "obs/_index", access_log)
        features = _text_axis(handle, "raw/var/Featurekey", access_log)
        names = _text_axis(handle, "raw/var/NameInData", access_log)
        targets = _text_axis(handle, "raw/var/target-1", access_log)
        shape = _raw_shape(handle)
    if shape != (len(barcodes), len(features)) or not (
        len(features) == len(names) == len(targets)
    ):
        raise ValueError("ADT raw matrix shape differs from its text axes")
    if len(barcodes) != len(set(barcodes)):
        raise ValueError("ADT barcode axis is not unique")
    positions = _feature_indices(
        features, [str(row["adt_feature"]) for row in markers], "ADT raw"
    )
    for row, position in zip(markers, positions):
        if names[position] != row["adt_name"]:
            raise ValueError(f"ADT exact name differs for {row['marker']}")
        if not targets[position].startswith(f"{row['panel_id']} "):
            raise ValueError(f"ADT official panel ID differs for {row['marker']}")
        if str(row["marker"]).lower() not in targets[position].lower():
            raise ValueError(f"ADT official target differs for {row['marker']}")
    return {
        "raw_shape": list(shape),
        "barcodes": barcodes,
        "barcode_axis_sha256": _axis_sha256(barcodes),
        "marker_indices": positions,
        "feature_axis_sha256": _axis_sha256(features),
    }


def _file_record(
    source_root: Path, record: dict[str, Any], template: str
) -> dict[str, Any]:
    archive = source_root / record["filename"]
    h5ad = source_root / record["h5ad"]
    if not archive.is_file() or not h5ad.is_file():
        raise FileNotFoundError(
            f"source cache misses {record['filename']} or {record['h5ad']}"
        )
    if archive.stat().st_size != record["bytes"]:
        raise ValueError(f"archive size differs for {record['filename']}")
    return {
        "accession": record["accession"],
        "filename": record["filename"],
        "url": _archive_url(record["accession"], record["filename"], template),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": _sha256(archive),
        "h5ad": record["h5ad"],
        "h5ad_bytes": h5ad.stat().st_size,
        "h5ad_sha256": _sha256(h5ad),
    }


def run_preflight(
    manifest_path: Path,
    source_root: Path,
    panel_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    samples, markers = _validate_manifest(manifest)
    panel = manifest["official_archives"]["panel"]
    if (
        not panel_path.is_file()
        or panel_path.stat().st_size != panel["bytes"]
        or _sha256(panel_path) != panel["sha256"]
    ):
        raise ValueError("official TotalSeq-C workbook differs from the freeze")

    sample_results: list[dict[str, Any]] = []
    all_text_reads: set[str] = set()
    template = str(manifest["geo_sample_url_template"])
    for sample in samples:
        reads: list[str] = []
        gex_file = _file_record(source_root, sample["gex"], template)
        gex = _inspect_gex(source_root / sample["gex"]["h5ad"], markers, reads)
        result: dict[str, Any] = {
            "donor": sample["donor"],
            "role": sample["role"],
            "pregnancy": sample["pregnancy"],
            "severity": sample["severity"],
            "gex": gex_file,
            "gex_raw_shape": gex["raw_shape"],
            "gex_obs_index": gex["obs_index"],
            "gex_barcode_axis_sha256": gex["barcode_axis_sha256"],
            "gex_marker_indices": gex["marker_indices"],
        }
        if sample["adt"] is None:
            result.update(
                {
                    "adt": None,
                    "common_barcode_count": 0,
                    "common_barcode_axis_sha256": None,
                    "metadata_eligible": False,
                    "exclusion": sample["exclusion"],
                }
            )
        else:
            adt_file = _file_record(source_root, sample["adt"], template)
            adt = _inspect_adt(source_root / sample["adt"]["h5ad"], markers, reads)
            common = sorted(set(gex["barcodes"]) & set(adt["barcodes"]))
            result.update(
                {
                    "adt": adt_file,
                    "adt_raw_shape": adt["raw_shape"],
                    "adt_barcode_axis_sha256": adt["barcode_axis_sha256"],
                    "adt_marker_indices": adt["marker_indices"],
                    "common_barcode_count": len(common),
                    "common_barcode_axis_sha256": _axis_sha256(common),
                    "metadata_eligible": len(common) >= CELL_BUDGET,
                }
            )
            if sample["role"] == "excluded_metadata":
                result["exclusion"] = sample["exclusion"]
        if (
            sample["role"] in {"calibration", "pilot", "held"}
            and not result["metadata_eligible"]
        ):
            raise ValueError(
                f"frozen donor {sample['donor']} has fewer than {CELL_BUDGET} common barcodes"
            )
        if sample["donor"] == "100" and (
            result.get("adt_raw_shape", [None])[0] != 378
            or result["common_barcode_count"] != 378
        ):
            raise ValueError("donor 100 exclusion no longer matches deposited metadata")
        all_text_reads.update(reads)
        result["text_axes_read"] = sorted(set(reads))
        sample_results.append(result)

    payload = {
        "schema": "gse239452-citeseq-metadata-preflight/1.0",
        "status": "PASS",
        "created_at_utc": _timestamp(),
        "source_manifest_sha256": _sha256(manifest_path),
        "panel_workbook_sha256": _sha256(panel_path),
        "role_counts": EXPECTED_ROLES,
        "prospective_exclusions": [
            {"donor": row["donor"], "reason": row["exclusion"]}
            for row in samples
            if row["role"] == "excluded_metadata"
        ],
        "samples": sample_results,
        "access_audit": {
            "text_or_metadata_paths_read": sorted(all_text_reads),
            "raw_matrix_structural_metadata_read": True,
            "numeric_matrix_payload_values_read": 0,
            "raw_X_data_values_read": 0,
            "raw_X_indices_values_read": 0,
            "raw_X_indptr_values_read": 0,
            "normalized_X_values_read": 0,
            "layers_values_read": 0,
        },
    }
    payload["preflight_content_sha256"] = _canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "created_at_utc"}
    )
    _write_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run_preflight(args.manifest, args.source_root, args.panel, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
