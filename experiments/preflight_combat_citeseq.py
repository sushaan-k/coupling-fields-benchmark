"""Metadata-only structural preflight for the COMBAT CITE-seq H5AD."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np


MARKER_SPECS = (
    ("CD4", "ENSG00000010610", "AB_CD4"),
    ("CD7", "ENSG00000173762", "AB_CD7"),
    ("CD14", "ENSG00000170458", "AB_CD14"),
    ("CD19", "ENSG00000177455", "AB_CD19"),
    ("CD33", "ENSG00000105383", "AB_CD33"),
    ("CD38", "ENSG00000004468", "AB_CD38"),
    ("CD44", "ENSG00000026508", "AB_humanCD44"),
    ("CD47", "ENSG00000196776", "AB_CD47"),
    ("CD52", "ENSG00000169442", "AB_CD52"),
)
EXPECTED_MARKERS = tuple(marker for marker, _, _ in MARKER_SPECS)

ELIGIBLE_CELL_TYPES = (
    "B",
    "ERYTH",
    "HSC",
    "MNP",
    "NK",
    "PB",
    "PLT",
    "T",
)
MINIMUM_ELIGIBLE_CELLS_PER_SAMPLE = 512

CALIBRATION_BY_SOURCE = {
    "COVID_CRIT": ("S00024", "S00027"),
    "COVID_HCW_MILD": ("G05077", "G05171"),
    "COVID_MILD": ("S00002", "S00126"),
    "COVID_SEV": ("S00045", "S00148"),
    "HV": ("H00052", "H00054"),
    "Sepsis": ("N00032", "N00050"),
}

PILOT_BY_SOURCE = {
    "COVID_CRIT": ("S00008", "S00020", "S00040", "S00052"),
    "COVID_HCW_MILD": ("G05061", "G05097", "G05145", "G05164"),
    "COVID_MILD": ("S00063", "S00076", "S00104", "S00114"),
    "COVID_SEV": ("S00037", "S00042", "S00053", "S00134"),
    "HV": ("H00058", "H00064", "H00070", "H00072"),
    "Sepsis": ("N00006", "N00024", "N00025", "N00047"),
}

OXFORD_HELD_BY_SOURCE = {
    "COVID_CRIT": (
        "S00005",
        "S00007",
        "S00043",
        "S00050",
        "S00054",
        "S00065",
        "S00068",
        "S00094",
        "S00095",
        "S00099",
        "S00109",
        "S00124",
    ),
    "COVID_HCW_MILD": (
        "G05064",
        "G05073",
        "G05078",
        "G05105",
        "G05112",
        "G05153",
    ),
    "COVID_MILD": (
        "S00006",
        "S00016",
        "S00058",
        "S00059",
        "S00112",
        "S00113",
    ),
    "COVID_SEV": (
        "S00028",
        "S00034",
        "S00041",
        "S00048",
        "S00056",
        "S00057",
        "S00060",
        "S00061",
        "S00064",
        "S00067",
        "S00069",
        "S00077",
        "S00078",
        "S00106",
    ),
    "HV": ("H00049", "H00053", "H00067", "H00085"),
    "Sepsis": (
        "N00007",
        "N00012",
        "N00017",
        "N00021",
        "N00028",
        "N00033",
        "N00038",
        "N00040",
        "N00049",
    ),
}

ST_GEORGES_HELD_IDS = (
    "U00501",
    "U00502",
    "U00503",
    "U00505",
    "U00601",
    "U00605",
    "U00607",
    "U00617",
    "U00619",
    "U00701",
)

REQUIRED_OBS_FIELDS = (
    "COMBAT_ID",
    "scRNASeq_sample_ID",
    "Source",
    "Institute",
    "Annotation_cell_type",
)

COMPOSITION_TAR_SOURCE = {
    "filename": "CBD-KEY-CITESEQ-GEX-COMPOSITION.tar.gz",
    "content_url": (
        "https://zenodo.org/api/records/6120249/files/"
        "CBD-KEY-CITESEQ-GEX-COMPOSITION.tar.gz/content"
    ),
    "bytes": 212203,
    "md5": "fe077fc9f314419536d6d901855c9d84",
    "sha256": "cc6b50cb363b800f356aa79224240f96ca11046a9c2a1c5f4f78603531b3dae3",
}

COMPOSITION_CSV_SOURCE = {
    "archive_member": (
        "CBD-KEY-CITESEQ-GEX-COMPOSITION/"
        "COMBAT_CITEseq_Composition-PerSample_CellType_Counts_and_"
        "PercentFrequencies_out_of_all_PBMCs.csv"
    ),
    "bytes": 33391,
    "md5": "c4e635d7f16b3f7e3e66571a3359de3b",
    "sha256": "2ad7e92ab122ee52986d5748dbb23c335c02ec1f1f244943ce46fff94c585157",
}

COMPOSITION_COLUMNS = (
    "scRNASeq_sample_ID",
    "CellType",
    "CellType_Count",
    "TotalPBMC_Count",
    "Percentage",
)

COMPOSITION_CELL_TYPES = (
    "B",
    "ERYTH",
    "HSC",
    "MNP",
    "MNP|PLT",
    "NK",
    "PB",
    "PLT",
    "T",
)


def _file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    return _file_digest(path, "sha256")


def _md5(path: Path) -> str:
    return _file_digest(path, "md5")


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _attributes(item: h5py.Group | h5py.Dataset) -> dict[str, Any]:
    return {key: _json_value(item.attrs[key]) for key in sorted(item.attrs, key=str)}


def _object_metadata(item: h5py.Group | h5py.Dataset) -> dict[str, Any]:
    result: dict[str, Any] = {
        "kind": "dataset" if isinstance(item, h5py.Dataset) else "group",
        "attributes": _attributes(item),
    }
    if isinstance(item, h5py.Dataset):
        result["shape"] = list(item.shape)
        result["dtype"] = str(item.dtype)
    return result


def _inventory(handle: h5py.File) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {"/": _object_metadata(handle)}

    def visit(name: str, item: h5py.Group | h5py.Dataset) -> None:
        entries[f"/{name}"] = _object_metadata(item)

    handle.visititems(visit)
    return dict(sorted(entries.items()))


def _component_metadata(item: h5py.Group | h5py.Dataset) -> dict[str, Any]:
    result = _object_metadata(item)
    if isinstance(item, h5py.Group):
        result["members"] = {
            key: _object_metadata(item[key]) for key in sorted(item.keys())
        }
    return result


def _ordered_columns(group: h5py.Group, index_key: str | None) -> list[str]:
    declared = _json_value(group.attrs.get("column-order", []))
    if isinstance(declared, str):
        declared = [declared]
    if not isinstance(declared, list):
        declared = []
    columns = [str(value) for value in declared if str(value) in group]
    columns.extend(
        key for key in sorted(group.keys()) if key != index_key and key not in columns
    )
    return columns


def _row_count(item: h5py.Group | h5py.Dataset | None) -> int | None:
    if isinstance(item, h5py.Dataset) and item.ndim >= 1:
        return int(item.shape[0])
    if isinstance(item, h5py.Group):
        codes = item.get("codes")
        if isinstance(codes, h5py.Dataset) and codes.ndim == 1:
            return int(codes.shape[0])
    return None


def _dataframe_metadata(handle: h5py.File, path: str) -> dict[str, Any]:
    if path not in handle:
        return {"present": False}
    item = handle[path]
    if not isinstance(item, h5py.Group):
        result = _object_metadata(item)
        result["present"] = True
        result["warning"] = "expected an HDF5 group for a dataframe encoding"
        return result

    attrs = _attributes(item)
    raw_index = attrs.get("_index")
    index_key = raw_index if isinstance(raw_index, str) else None
    index = item.get(index_key) if index_key and index_key in item else None
    columns = _ordered_columns(item, index_key)
    return {
        "present": True,
        "attributes": attrs,
        "encoding_type": attrs.get("encoding-type"),
        "encoding_version": attrs.get("encoding-version"),
        "index_key": index_key,
        "rows": _row_count(index),
        "index": _component_metadata(index) if index is not None else None,
        "column_order": columns,
        "columns": {key: _component_metadata(item[key]) for key in columns},
    }


def _matrix_metadata(item: h5py.Group | h5py.Dataset) -> dict[str, Any]:
    result = _object_metadata(item)
    attrs = result["attributes"]
    result["encoding_type"] = attrs.get("encoding-type")
    result["encoding_version"] = attrs.get("encoding-version")
    if isinstance(item, h5py.Group):
        shape = attrs.get("shape")
        result["shape"] = shape if isinstance(shape, list) else None
        result["members"] = {
            key: _object_metadata(item[key]) for key in sorted(item.keys())
        }
    return result


def _matrix_report(handle: h5py.File) -> dict[str, Any]:
    result: dict[str, Any] = {
        "X": _matrix_metadata(handle["X"]) if "X" in handle else None,
        "layers": {},
        "raw_X": None,
    }
    layers = handle.get("layers")
    if isinstance(layers, h5py.Group):
        result["layers"] = {
            key: _matrix_metadata(layers[key]) for key in sorted(layers.keys())
        }
    raw = handle.get("raw")
    if isinstance(raw, h5py.Group) and "X" in raw:
        result["raw_X"] = _matrix_metadata(raw["X"])
    return result


def _matrix_payload_paths(handle: h5py.File) -> list[str]:
    paths: list[str] = []

    def add(path: str, item: h5py.Group | h5py.Dataset) -> None:
        if isinstance(item, h5py.Dataset):
            paths.append(path)
            return
        for key in ("data", "indices", "indptr"):
            if key in item:
                paths.append(f"{path}/{key}")

    if "X" in handle:
        add("/X", handle["X"])
    layers = handle.get("layers")
    if isinstance(layers, h5py.Group):
        for key in sorted(layers.keys()):
            add(f"/layers/{key}", layers[key])
    raw = handle.get("raw")
    if isinstance(raw, h5py.Group) and "X" in raw:
        add("/raw/X", raw["X"])
    return sorted(paths)


def _string_dataset_values(dataset: h5py.Dataset) -> list[str] | None:
    if dataset.ndim != 1 or h5py.check_string_dtype(dataset.dtype) is None:
        return None
    return [str(value) for value in dataset.asstr()[...].tolist()]


def _categorical_values(
    categories: h5py.Dataset, codes: h5py.Dataset
) -> list[str] | None:
    category_values = _string_dataset_values(categories)
    if (
        category_values is None
        or codes.ndim != 1
        or not np.issubdtype(codes.dtype, np.integer)
    ):
        return None
    code_values = np.asarray(codes[...], dtype=np.int64)
    return [
        category_values[code] if 0 <= code < len(category_values) else ""
        for code in code_values
    ]


def _encoded_values(item: h5py.Group | h5py.Dataset) -> list[str] | None:
    if isinstance(item, h5py.Dataset):
        return _string_dataset_values(item)
    categories = item.get("categories")
    codes = item.get("codes")
    if not isinstance(categories, h5py.Dataset) or not isinstance(codes, h5py.Dataset):
        return None
    return _categorical_values(categories, codes)


def _dataframe_values(frame: h5py.Group, key: str) -> list[str] | None:
    item = frame.get(key)
    if not isinstance(item, (h5py.Group, h5py.Dataset)):
        return None
    values = _encoded_values(item)
    if values is not None:
        return values

    legacy_categories = frame.get("__categories")
    if not isinstance(item, h5py.Dataset) or not isinstance(
        legacy_categories, h5py.Group
    ):
        return None
    categories = legacy_categories.get(key)
    if not isinstance(categories, h5py.Dataset):
        return None
    return _categorical_values(categories, item)


_ENSEMBL_ID = re.compile(r"^(ENSG[0-9]+)(?:\.[0-9]+)?$")


def _version_stripped_ensembl(value: str) -> str | None:
    match = _ENSEMBL_ID.fullmatch(value)
    return match.group(1) if match else None


def _marker_report(handle: h5py.File) -> dict[str, Any]:
    frame = handle.get("var")
    metadata_errors: list[str] = []
    if not isinstance(frame, h5py.Group):
        metadata_errors.append("/var is not a dataframe group")
        names = gene_ids = feature_types = None
        index_key = None
    else:
        raw_index = _json_value(frame.attrs.get("_index"))
        index_key = raw_index if isinstance(raw_index, str) else None
        names = _dataframe_values(frame, index_key) if index_key else None
        gene_ids = _dataframe_values(frame, "gene_ids")
        feature_types = _dataframe_values(frame, "feature_types")
        if names is None:
            metadata_errors.append("/var exact feature-name index is not decodable")
        if gene_ids is None:
            metadata_errors.append("/var/gene_ids is not decodable")
        if feature_types is None:
            metadata_errors.append("/var/feature_types is not decodable")

    lengths = {
        "feature_names": len(names) if names is not None else None,
        "gene_ids": len(gene_ids) if gene_ids is not None else None,
        "feature_types": len(feature_types) if feature_types is not None else None,
    }
    present_lengths = {value for value in lengths.values() if value is not None}
    if len(present_lengths) > 1:
        metadata_errors.append("var feature metadata vectors have unequal lengths")

    rows = []
    missing: list[str] = []
    duplicates: list[str] = []
    aligned = (
        names is not None
        and gene_ids is not None
        and feature_types is not None
        and len({len(names), len(gene_ids), len(feature_types)}) == 1
    )
    for marker, ensembl_id, adt_name in MARKER_SPECS:
        rna_matches: list[dict[str, Any]] = []
        adt_matches: list[dict[str, Any]] = []
        if aligned:
            for feature_index, (name, gene_id, feature_type) in enumerate(
                zip(names, gene_ids, feature_types)
            ):
                if (
                    name == marker
                    and _version_stripped_ensembl(gene_id) == ensembl_id
                    and feature_type == "Gene Expression"
                ):
                    rna_matches.append(
                        {
                            "source": "/var",
                            "feature_index": feature_index,
                            "symbol": name,
                            "gene_id": gene_id,
                            "feature_type": feature_type,
                        }
                    )
                if name == adt_name and feature_type == "Antibody Capture":
                    adt_matches.append(
                        {
                            "source": "/var",
                            "feature_index": feature_index,
                            "name": name,
                            "feature_type": feature_type,
                        }
                    )
        if not rna_matches:
            missing.append(marker)
        elif len(rna_matches) > 1:
            duplicates.append(marker)
        if not adt_matches:
            missing.append(adt_name)
        elif len(adt_matches) > 1:
            duplicates.append(adt_name)
        rows.append(
            {
                "marker": marker,
                "rna_expected": {
                    "symbol": marker,
                    "version_stripped_ensembl_id": ensembl_id,
                    "feature_type": "Gene Expression",
                },
                "rna_exact_matches": rna_matches,
                "adt_expected": {
                    "name": adt_name,
                    "feature_type": "Antibody Capture",
                },
                "adt_exact_matches": adt_matches,
            }
        )

    return {
        "expected_markers": list(EXPECTED_MARKERS),
        "exact_sources": {
            "feature_names": f"/var/{index_key}" if index_key else None,
            "gene_ids": "/var/gene_ids",
            "feature_types": "/var/feature_types",
        },
        "source_lengths": lengths,
        "markers": rows,
        "missing_exact_names": missing,
        "duplicate_exact_features": duplicates,
        "metadata_errors": metadata_errors,
        "fuzzy_fallback_used": False,
        "complete_exact_panel": not missing and not duplicates and not metadata_errors,
    }


def _flatten_by_source(values: dict[str, tuple[str, ...]]) -> list[str]:
    return [sample for samples in values.values() for sample in samples]


def _frozen_design_contract() -> tuple[dict[str, dict[str, str]], dict[str, list[str]]]:
    roles = {
        "calibration": _flatten_by_source(CALIBRATION_BY_SOURCE),
        "pilot_adaptive_development": _flatten_by_source(PILOT_BY_SOURCE),
        "oxford_held_confirmatory": _flatten_by_source(OXFORD_HELD_BY_SOURCE),
        "st_georges_held_confirmatory": list(ST_GEORGES_HELD_IDS),
    }
    expected: dict[str, dict[str, str]] = {}
    for role, by_source in (
        ("calibration", CALIBRATION_BY_SOURCE),
        ("pilot_adaptive_development", PILOT_BY_SOURCE),
        ("oxford_held_confirmatory", OXFORD_HELD_BY_SOURCE),
    ):
        for source, sample_ids in by_source.items():
            for sample_id in sample_ids:
                if sample_id in expected:
                    raise AssertionError(f"duplicate frozen COMBAT_ID: {sample_id}")
                expected[sample_id] = {
                    "role": role,
                    "source": source,
                    "institute": "Oxford",
                }
    for sample_id in ST_GEORGES_HELD_IDS:
        if sample_id in expected:
            raise AssertionError(f"duplicate frozen COMBAT_ID: {sample_id}")
        expected[sample_id] = {
            "role": "st_georges_held_confirmatory",
            "source": "Flu",
            "institute": "St_Georges",
        }
    return expected, roles


_COMPOSITION_SAMPLE_ID = re.compile(r"^([A-Z][0-9]{5})-[A-Za-z0-9]+-[A-Za-z0-9]+$")


def _composition_contract(
    composition_csv_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    size = composition_csv_path.stat().st_size
    md5 = _md5(composition_csv_path)
    sha256 = _sha256(composition_csv_path)
    errors: list[str] = []
    if size != COMPOSITION_CSV_SOURCE["bytes"]:
        errors.append("composition CSV byte count does not match the frozen source")
    if md5 != COMPOSITION_CSV_SOURCE["md5"]:
        errors.append("composition CSV MD5 does not match the frozen source")
    if sha256 != COMPOSITION_CSV_SOURCE["sha256"]:
        errors.append("composition CSV SHA-256 does not match the frozen source")

    with composition_csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COMPOSITION_COLUMNS:
            errors.append("composition CSV columns do not match the frozen schema")
        rows = list(reader)
    if len(rows) != 873:
        errors.append("composition CSV must contain exactly 873 data rows")

    rows_by_sample: dict[str, list[dict[str, str]]] = {}
    malformed_rows = 0
    for row in rows:
        if None in row or any(
            row.get(column) is None for column in COMPOSITION_COLUMNS
        ):
            malformed_rows += 1
            continue
        sample_id = row["scRNASeq_sample_ID"]
        rows_by_sample.setdefault(sample_id, []).append(row)
    if malformed_rows:
        errors.append("composition CSV contains malformed rows")

    pairs: dict[str, dict[str, Any]] = {}
    invalid_samples: list[str] = []
    for sample_id, sample_rows in sorted(rows_by_sample.items()):
        match = _COMPOSITION_SAMPLE_ID.fullmatch(sample_id)
        cell_types = [row["CellType"] for row in sample_rows]
        totals = {row["TotalPBMC_Count"] for row in sample_rows}
        valid_numeric = True
        try:
            counts = [int(row["CellType_Count"]) for row in sample_rows]
            percentages = [float(row["Percentage"]) for row in sample_rows]
            total_values = [int(value) for value in totals]
        except ValueError:
            valid_numeric = False
            counts = []
            percentages = []
            total_values = []
        valid_sample = (
            match is not None
            and len(sample_rows) == len(COMPOSITION_CELL_TYPES)
            and sorted(cell_types) == sorted(COMPOSITION_CELL_TYPES)
            and len(set(cell_types)) == len(COMPOSITION_CELL_TYPES)
            and len(totals) == 1
            and valid_numeric
            and all(count >= 0 for count in counts)
            and all(np.isfinite(percent) and percent >= 0.0 for percent in percentages)
            and len(total_values) == 1
            and total_values[0] > 0
        )
        if not valid_sample:
            invalid_samples.append(sample_id)
            continue
        combat_id = match.group(1)
        if combat_id in pairs:
            invalid_samples.extend((pairs[combat_id]["scRNASeq_sample_ID"], sample_id))
            continue
        pairs[combat_id] = {
            "scRNASeq_sample_ID": sample_id,
            "total_pbmc_count": total_values[0],
        }

    if invalid_samples:
        errors.append("composition CSV contains invalid or duplicate sample records")
    design, _ = _frozen_design_contract()
    missing_design_ids = sorted(set(design) - set(pairs))
    unexpected_composition_ids = sorted(set(pairs) - set(design))
    if missing_design_ids or unexpected_composition_ids or len(pairs) != 97:
        errors.append("composition CSV pair universe does not match the frozen 97 IDs")

    report = {
        "parent_archive": COMPOSITION_TAR_SOURCE,
        "frozen_csv_source": COMPOSITION_CSV_SOURCE,
        "input": {
            "filename": composition_csv_path.name,
            "bytes": size,
            "md5": md5,
            "sha256": sha256,
        },
        "columns": list(reader.fieldnames or ()),
        "data_rows": len(rows),
        "unique_sample_ids": len(rows_by_sample),
        "official_pairs_by_combat_id": pairs,
        "missing_frozen_combat_ids": missing_design_ids,
        "unexpected_combat_ids": unexpected_composition_ids,
        "invalid_sample_ids": sorted(set(invalid_samples)),
        "total_pbmc_count_use": "descriptive_only_no_h5ad_equality_assertion",
        "errors": errors,
        "valid": not errors,
    }
    return report, pairs


def _frozen_sample_contract(
    official_pairs: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    expected, roles = _frozen_design_contract()
    for combat_id, contract in expected.items():
        pair = official_pairs.get(combat_id)
        contract["scRNASeq_sample_ID"] = (
            pair["scRNASeq_sample_ID"] if pair is not None else None
        )
        contract["total_pbmc_count"] = (
            pair["total_pbmc_count"] if pair is not None else None
        )
    return expected, roles


def _obs_contract(
    handle: h5py.File,
    official_pairs: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    frame = handle.get("obs")
    schema_errors: list[str] = []
    decoded_paths: list[str] = []
    values: dict[str, list[str]] = {}
    if not isinstance(frame, h5py.Group):
        schema_errors.append("/obs is not a dataframe group")
    else:
        for field in REQUIRED_OBS_FIELDS:
            field_values = _dataframe_values(frame, field)
            if field_values is None:
                schema_errors.append(f"/obs/{field} is not decodable")
            else:
                values[field] = field_values
                decoded_paths.append(f"/obs/{field}")
                if (
                    isinstance(frame.get("__categories"), h5py.Group)
                    and field in frame["__categories"]
                ):
                    decoded_paths.append(f"/obs/__categories/{field}")

    row_lengths = {field: len(field_values) for field, field_values in values.items()}
    if len(set(row_lengths.values())) > 1:
        schema_errors.append("required obs metadata vectors have unequal lengths")

    expected, roles = _frozen_sample_contract(official_pairs)
    observed: dict[str, dict[str, Any]] = {
        combat_id: {
            "sources": set(),
            "institutes": set(),
            "raw_rows": 0,
            "eligible_cells": 0,
        }
        for combat_id in expected
    }
    outside_pair_rows: dict[str, dict[str, Any]] = {
        combat_id: {"rows": 0, "scRNASeq_sample_ID_values": set()}
        for combat_id in expected
    }
    allowed = set(ELIGIBLE_CELL_TYPES)
    observed_labels: set[str] = set()
    required_decoded = all(field in values for field in REQUIRED_OBS_FIELDS)
    aligned = required_decoded and len(set(row_lengths.values())) == 1
    if aligned:
        for combat_id, sample_id, source, institute, cell_type in zip(
            *(values[field] for field in REQUIRED_OBS_FIELDS)
        ):
            if combat_id not in observed:
                continue
            if sample_id != expected[combat_id]["scRNASeq_sample_ID"]:
                outside = outside_pair_rows[combat_id]
                outside["rows"] = int(outside["rows"]) + 1
                outside["scRNASeq_sample_ID_values"].add(sample_id)
                continue
            if cell_type not in {"", "nan"}:
                observed_labels.add(cell_type)
            entry = observed[combat_id]
            entry["sources"].add(source)
            entry["institutes"].add(institute)
            entry["raw_rows"] = int(entry["raw_rows"]) + 1
            if cell_type in allowed:
                entry["eligible_cells"] = int(entry["eligible_cells"]) + 1

    missing_ids = sorted(
        combat_id for combat_id, entry in observed.items() if entry["raw_rows"] == 0
    )
    missing_pairs = [
        {
            "combat_id": combat_id,
            "scRNASeq_sample_ID": expected[combat_id]["scRNASeq_sample_ID"],
        }
        for combat_id in missing_ids
    ]
    metadata_mismatches = []
    for combat_id, contract in expected.items():
        entry = observed[combat_id]
        if entry["raw_rows"] == 0:
            continue
        sources = sorted(entry["sources"])
        institutes = sorted(entry["institutes"])
        if sources != [contract["source"]] or institutes != [contract["institute"]]:
            metadata_mismatches.append(
                {
                    "combat_id": combat_id,
                    "scRNASeq_sample_ID": contract["scRNASeq_sample_ID"],
                    "expected_source": contract["source"],
                    "observed_sources": sources,
                    "expected_institute": contract["institute"],
                    "observed_institutes": institutes,
                }
            )

    sample_counts = {
        combat_id: {
            "raw_rows": int(entry["raw_rows"]),
            "eligible_cells": int(entry["eligible_cells"]),
            "composition_total_pbmc_count": expected[combat_id]["total_pbmc_count"],
        }
        for combat_id, entry in sorted(observed.items())
    }
    insufficient_eligible_cells = [
        {
            "combat_id": sample_id,
            "eligible_cells": counts["eligible_cells"],
            "required_minimum": MINIMUM_ELIGIBLE_CELLS_PER_SAMPLE,
        }
        for sample_id, counts in sample_counts.items()
        if counts["eligible_cells"] < MINIMUM_ELIGIBLE_CELLS_PER_SAMPLE
    ]
    present_ids = set(expected) - set(missing_ids)
    present_role_counts = {
        role: sum(combat_id in present_ids for combat_id in combat_ids)
        for role, combat_ids in roles.items()
    }
    expected_role_counts = {role: len(combat_ids) for role, combat_ids in roles.items()}
    role_counts_match = present_role_counts == expected_role_counts
    outside_pair_report = {
        combat_id: {
            "rows": int(entry["rows"]),
            "scRNASeq_sample_ID_values": sorted(entry["scRNASeq_sample_ID_values"]),
        }
        for combat_id, entry in sorted(outside_pair_rows.items())
        if entry["rows"]
    }
    sample_report = {
        "required_obs_fields": list(REQUIRED_OBS_FIELDS),
        "decoded_row_counts": row_lengths,
        "frozen_ids_by_role": roles,
        "expected_metadata_by_combat_id": expected,
        "expected_role_counts": expected_role_counts,
        "present_role_counts": present_role_counts,
        "missing_combat_ids": missing_ids,
        "missing_exact_pairs": missing_pairs,
        "outside_universe_rows_for_designated_combat_ids": outside_pair_report,
        "pair_selection_rule": "exact COMBAT_ID and scRNASeq_sample_ID equality",
        "source_institute_mismatches": metadata_mismatches,
        "designated_sample_counts": sample_counts,
        "minimum_eligible_cells_per_sample": MINIMUM_ELIGIBLE_CELLS_PER_SAMPLE,
        "insufficient_eligible_cells": insufficient_eligible_cells,
        "total_pbmc_count_equality_asserted": False,
        "schema_errors": schema_errors,
        "complete_frozen_cohorts": (
            aligned
            and role_counts_match
            and not missing_ids
            and not metadata_mismatches
            and not insufficient_eligible_cells
            and not schema_errors
        ),
    }

    unexpected = sorted(observed_labels - allowed)
    cell_type_report = {
        "field": "Annotation_cell_type",
        "exact_allowed_values": list(ELIGIBLE_CELL_TYPES),
        "observed_nonmissing_values": sorted(observed_labels),
        "observed_eligible_values": sorted(observed_labels & allowed),
        "allowed_but_unobserved_values": sorted(allowed - observed_labels),
        "unexpected_values": unexpected,
        "literal_nan_excluded": True,
        "fuzzy_fallback_used": False,
        "observed_values_are_allowed_subset": (
            aligned and not unexpected and bool(observed_labels)
        ),
    }
    return sample_report, cell_type_report, decoded_paths


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_loads(value: str) -> Any:
    return json.loads(value, object_pairs_hook=_unique_json_object)


def build_preflight(input_path: Path, composition_csv_path: Path) -> dict[str, Any]:
    """Inspect H5AD metadata without decoding any matrix payload."""

    input_size = input_path.stat().st_size
    input_hash = _sha256(input_path)
    composition_report, official_pairs = _composition_contract(composition_csv_path)
    with h5py.File(input_path, "r") as handle:
        inventory = _inventory(handle)
        obs = _dataframe_metadata(handle, "obs")
        var = _dataframe_metadata(handle, "var")
        raw_var = _dataframe_metadata(handle, "raw/var")
        matrices = _matrix_report(handle)
        payload_paths = _matrix_payload_paths(handle)
        marker_report = _marker_report(handle)
        sample_report, cell_type_report, obs_decoded_paths = _obs_contract(
            handle, official_pairs
        )
        feature_decoded_paths = [
            path
            for path in (
                marker_report["exact_sources"]["feature_names"],
                marker_report["exact_sources"]["gene_ids"],
                marker_report["exact_sources"]["feature_types"],
                "/var/__categories/feature_types",
            )
            if isinstance(path, str) and path.lstrip("/") in handle
        ]

    warnings: list[str] = []
    if not obs.get("present"):
        warnings.append("obs is absent")
    if not var.get("present"):
        warnings.append("var is absent")
    if matrices["X"] is None:
        warnings.append("X is absent")
    if "raw" not in matrices["layers"]:
        warnings.append("layers/raw is absent")
    if not composition_report["valid"]:
        warnings.append("the frozen composition CSV contract failed")
    if not marker_report["complete_exact_panel"]:
        warnings.append("the exact RNA/ADT feature contract failed")
    if not sample_report["complete_frozen_cohorts"]:
        warnings.append("the frozen sample metadata contract failed")
    if not cell_type_report["observed_values_are_allowed_subset"]:
        warnings.append("the exact Annotation_cell_type allowlist contract failed")

    contract_passes = (
        obs.get("present")
        and var.get("present")
        and matrices["X"] is not None
        and "raw" in matrices["layers"]
        and composition_report["valid"]
        and marker_report["complete_exact_panel"]
        and sample_report["complete_frozen_cohorts"]
        and cell_type_report["observed_values_are_allowed_subset"]
    )

    return {
        "schema_version": "combat_citeseq_metadata_preflight_v3",
        "status": (
            "PREFLIGHT_METADATA_CONTRACT_PASS"
            if contract_passes
            else "PREFLIGHT_METADATA_CONTRACT_FAIL"
        ),
        "input": {
            "filename": input_path.name,
            "bytes": input_size,
            "sha256": input_hash,
        },
        "hdf5": {
            "root_attributes": inventory["/"]["attributes"],
            "object_inventory": inventory,
        },
        "dataframes": {"obs": obs, "var": var, "raw_var": raw_var},
        "matrices": matrices,
        "composition_contract": composition_report,
        "marker_feature_candidates": marker_report,
        "frozen_sample_contract": sample_report,
        "cell_type_contract": cell_type_report,
        "warnings": warnings,
        "access_audit": {
            "opaque_complete_file_reads": ["SHA-256"],
            "decoded": [
                "HDF5 object names, attributes, shapes, and dtypes",
                "obs/var dataframe encoding metadata",
                "exact string or categorical metadata under obs and var",
                "official composition CSV rows and categorical metadata",
            ],
            "decoded_metadata_paths": sorted(
                set(obs_decoded_paths + feature_decoded_paths)
            ),
            "matrix_payload_reads": 0,
            "matrix_payload_paths_not_read": payload_paths,
        },
    }


def write_preflight(
    input_path: Path, composition_csv_path: Path, output_path: Path
) -> dict[str, Any]:
    if output_path.resolve() in {
        input_path.resolve(),
        composition_csv_path.resolve(),
    }:
        raise ValueError("input and output paths must differ")
    result = build_preflight(input_path, composition_csv_path)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    _strict_json_loads(serialized)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--composition-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_preflight(args.input, args.composition_csv, args.output)


if __name__ == "__main__":
    main()
