"""Sealed Stephenson E-MTAB-10026 CITE-seq confirmation.

The CLI deliberately separates metadata sealing, adaptive development, held
RNA-margin prediction, and held pairing access.  The public source manifest
uses the ArrayExpress SDRF individual identifier as the biological donor key;
the deposited ``patient_id`` field is never used for allocation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import unquote, urlparse
import urllib.request

import h5py
import numpy as np

from experiments import confirm_combat_citeseq as numerics
from mapreg.heterogeneity_adaptive_coupling import (
    CouplingEstimationRefusal,
    expected_binary_table_from_log_odds,
)
from mapreg.hierarchical_conditional_coupling import (
    fit_hierarchical_conditional_log_odds,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/confirmation/stephenson_citeseq"
DEFAULT_SDRF = DATA_DIR / "E-MTAB-10026.sdrf.txt"
DEFAULT_SOURCE = DATA_DIR / "source_manifest_v1.json"
DEFAULT_PREFLIGHT = (
    ROOT / "results/development/stephenson_citeseq_metadata_preflight.json"
)
DEFAULT_DESIGNATION = DATA_DIR / "candidate_designation_v1.json"
DEFAULT_PROTOCOL = (
    ROOT / "docs/STEPHENSON_CITESEQ_HELD_SITE_CONFIRMATION_PROTOCOL_2026-08-28.md"
)
DEFAULT_VERIFICATION = (
    ROOT / "docs/STEPHENSON_CITESEQ_PUBLIC_FREEZE_VERIFICATION_2026-08-28.json"
)
DEFAULT_DEVELOPMENT_AUTHORIZATION = DATA_DIR / "development_authorization_v1.json"
DEFAULT_DEVELOPMENT_ATTEMPT = DATA_DIR / "development_attempt_v1.json"
DEFAULT_DEVELOPMENT = (
    ROOT / "results/development/stephenson_citeseq_development.json"
)
DEFAULT_MARGIN_AUTHORIZATION = DATA_DIR / "held_rna_margin_authorization_v1.json"
DEFAULT_PREDICTION_ATTEMPT = DATA_DIR / "prediction_attempt_v1.json"
DEFAULT_PREDICTION = ROOT / "results/stephenson_citeseq_predictions.json"
DEFAULT_SCORE_AUTHORIZATION = DATA_DIR / "score_authorization_v1.json"
DEFAULT_SCORE_ATTEMPT = DATA_DIR / "score_attempt_v1.json"
DEFAULT_SCORE = ROOT / "results/stephenson_citeseq_confirmation.json"

PUBLIC_OWNER = "sushaan-k"
PUBLIC_REPOSITORY = "coupling-fields-benchmark"
PUBLIC_ORIGIN = f"https://github.com/{PUBLIC_OWNER}/{PUBLIC_REPOSITORY}.git"

OFFICIAL_H5AD_NAME = "covid_portal_210320_with_raw.h5ad"
OFFICIAL_H5AD_BYTES = 7_187_322_881
OFFICIAL_H5AD_URL = (
    "https://www.ebi.ac.uk/biostudies/files/E-MTAB-10026/"
    "covid_portal_210320_with_raw.h5ad"
)
OFFICIAL_SDRF_BYTES = 155_174
OFFICIAL_SDRF_SHA256 = (
    "68a27790e45b025f71f445c5ab6dbdc15d5fd74312f8d5366390759ff0580dc5"
)
OFFICIAL_SDRF_URL = (
    "https://www.ebi.ac.uk/biostudies/files/E-MTAB-10026/"
    "E-MTAB-10026.sdrf.txt"
)
SDRF_TO_H5AD_SAMPLE = {
    "BGCV06_CV0326": "BGCV13_CV0326",
    "BGCV13_CV0201": "BGCV06_CV0201",
}

MARKERS = numerics.MARKERS
RNA_ENSEMBL = dict(numerics.RNA_ENSEMBL)
ADT_FEATURE = {marker: f"AB_{marker}" for marker in MARKERS}
CELL_BUDGET = 512
CELL_TYPES = numerics.CELL_TYPES
INITIAL_CLUSTERS = (
    "B_cell",
    "CD4",
    "CD8",
    "CD14",
    "CD16",
    "DCs",
    "HSC",
    "Lymph_prolif",
    "MAIT",
    "Mono_prolif",
    "NK_16hi",
    "NK_56hi",
    "Plasmablast",
    "Platelets",
    "RBC",
    "Treg",
    "gdT",
    "pDC",
)
CLUSTER_TO_CELL_TYPE = {
    "B_cell": "B",
    "RBC": "ERYTH",
    "HSC": "HSC",
    "CD14": "MNP",
    "CD16": "MNP",
    "DCs": "MNP",
    "Mono_prolif": "MNP",
    "pDC": "MNP",
    "NK_16hi": "NK",
    "NK_56hi": "NK",
    "Plasmablast": "PB",
    "Platelets": "PLT",
    "CD4": "T",
    "CD8": "T",
    "Lymph_prolif": "T",
    "MAIT": "T",
    "Treg": "T",
    "gdT": "T",
}

SAMPLE_SELECTION_SALT = "STEPHENSON-CITESEQ-SAMPLE-v1"
CALIBRATION_SALT = "STEPHENSON-CAMBRIDGE-CALIBRATION-v1"
PILOT_SALT = "STEPHENSON-CAMBRIDGE-PILOT-v1"
CELL_SELECTION_SALT = "STEPHENSON-CITESEQ-CELL-BUDGET-v1"
ADT_TIE_SALT = "STEPHENSON-CITESEQ-ADT-v1"
DESTROYED_LINK_SALT = "STEPHENSON-CITESEQ-DESTROYED-LINK-v1"
EXPECTED_CALIBRATION = 12
EXPECTED_PILOT = 24
EXPECTED_HELD = 56
EXPECTED_CAMBRIDGE = 47
EXPECTED_NCL_UNSTIMULATED = 56
EXPECTED_SANGER = 11
PILOT_FAVORABLE = 19
HELD_FAVORABLE = 45
METHODS = (
    "primary",
    "best_residual",
    "destroyed_link",
    "hierarchical_graph_zero",
    "independence",
)
PROMOTION_COMPARATORS = ("best_residual", "destroyed_link")
CONDITIONAL_GRID = tuple(
    itertools.product((1, 2), (0.1, 1.0, 10.0), (0.01, 0.1), (0.0, 0.1, 1.0))
)
CLASSICAL_GRID = tuple(itertools.product(("pearson", "deviance"), (False, True)))
ALPHA_GRID = (0.5, 0.75, 1.0, 1.25)
MAXIMUM_CONDITION_NUMBER = 1e12

DEVELOPMENT_BINDING_PATHS = {
    "runner": "experiments/confirm_stephenson_citeseq.py",
    "runner_test": "tests/test_stephenson_citeseq_confirmation.py",
    "protocol": (
        "docs/STEPHENSON_CITESEQ_HELD_SITE_CONFIRMATION_PROTOCOL_2026-08-28.md"
    ),
    "designation": (
        "data/confirmation/stephenson_citeseq/candidate_designation_v1.json"
    ),
    "source_manifest": (
        "data/confirmation/stephenson_citeseq/source_manifest_v1.json"
    ),
    "sdrf": "data/confirmation/stephenson_citeseq/E-MTAB-10026.sdrf.txt",
    "metadata_preflight": (
        "results/development/stephenson_citeseq_metadata_preflight.json"
    ),
    "fresh_clone_verification": (
        "docs/STEPHENSON_CITESEQ_PUBLIC_FREEZE_VERIFICATION_2026-08-28.json"
    ),
    "combat_data_and_comparator_utility": "experiments/confirm_combat_citeseq.py",
    "combat_numerical_test": "tests/test_combat_citeseq_confirmation.py",
    "gse279451_evaluator": "experiments/evaluate_gse279451_sepsis_development.py",
    "gse279451_reducer": "experiments/reduce_gse279451_sepsis.py",
    "heterogeneity_adaptive_coupling": "mapreg/heterogeneity_adaptive_coupling.py",
    "hierarchical_conditional_coupling": (
        "mapreg/hierarchical_conditional_coupling.py"
    ),
    "coupling_fields": "mapreg/coupling_fields.py",
    "classical_residuals": "mapreg/classical_residuals.py",
    "table_prediction": "mapreg/table_prediction.py",
}


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


def _opaque_hashes(path: Path) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            sha256.update(block)
            md5.update(block)
    return sha256.hexdigest(), md5.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(), object_pairs_hook=_strict_object)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any], *, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("public artifact must be inside the repository") from error


def _bound_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise PermissionError("binding escapes repository") from error
    return path


def _immutable_public_bytes(relative: str, commit: str, label: str) -> bytes:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise PermissionError(f"{label} commit is not immutable")
    url = (
        f"https://raw.githubusercontent.com/{PUBLIC_OWNER}/{PUBLIC_REPOSITORY}/"
        f"{commit}/{relative}"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "coupling-fields/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
            for value in np.asarray(values)
        ],
        dtype=str,
    )


def _encoded_column(handle: h5py.File, group: h5py.Group, name: str) -> np.ndarray:
    if name not in group:
        raise ValueError(f"H5AD dataframe lacks {name}")
    item = group[name]
    if isinstance(item, h5py.Group):
        if "categories" not in item or "codes" not in item:
            raise ValueError(f"unsupported H5AD column encoding: {name}")
        categories = _decode_strings(item["categories"][...])
        codes = np.asarray(item["codes"][...], dtype=np.int64)
    elif isinstance(item, h5py.Dataset):
        if "categories" in item.attrs:
            categories = _decode_strings(handle[item.attrs["categories"]][...])
            codes = np.asarray(item[...], dtype=np.int64)
        elif h5py.check_string_dtype(item.dtype) is not None:
            return _decode_strings(item[...])
        else:
            return np.asarray(item[...])
    else:
        raise ValueError(f"unsupported H5AD column object: {name}")
    decoded = np.full(len(codes), "", dtype=object)
    valid = (codes >= 0) & (codes < len(categories))
    decoded[valid] = categories[codes[valid]]
    return np.asarray(decoded, dtype=str)


def _dataframe_index(handle: h5py.File, group: h5py.Group) -> np.ndarray:
    raw = group.attrs.get("_index", "_index")
    name = raw.decode() if isinstance(raw, bytes) else str(raw)
    if name not in group:
        raise ValueError("H5AD dataframe index is absent")
    return _decode_strings(group[name][...])


def _parse_sdrf(path: Path) -> list[dict[str, str]]:
    if path.stat().st_size != OFFICIAL_SDRF_BYTES or _sha256(path) != OFFICIAL_SDRF_SHA256:
        raise PermissionError("SDRF bytes differ from the official freeze")
    rows = list(csv.DictReader(io.StringIO(path.read_text()), delimiter="\t"))
    required = {
        "Source Name",
        "Characteristics[individual]",
        "Characteristics[disease]",
        "Characteristics[stimulus]",
    }
    if len(rows) != 143 or not rows or not required.issubset(rows[0]):
        raise ValueError("SDRF sample contract differs from E-MTAB-10026")
    result = []
    for row in rows:
        sample = row["Source Name"].strip()
        donor = row["Characteristics[individual]"].strip()
        disease = row["Characteristics[disease]"].strip()
        stimulus = " ".join(row["Characteristics[stimulus]"].split())
        if not sample or not donor or disease not in {"COVID-19", "normal"}:
            raise ValueError("SDRF contains an invalid sample row")
        result.append(
            {
                "sample": sample,
                "donor": donor,
                "disease": disease,
                "stimulus": stimulus,
            }
        )
    samples = [row["sample"] for row in result]
    if len(set(samples)) != len(samples):
        raise ValueError("SDRF Source Name is not unique")
    return result


def _sample_hash(donor: str, sample: str) -> str:
    return hashlib.sha256(
        "\0".join((SAMPLE_SELECTION_SALT, donor, sample)).encode()
    ).hexdigest()


def _role_hash(salt: str, disease: str, donor: str, sample: str) -> str:
    return hashlib.sha256(
        "\0".join((salt, disease, donor, sample)).encode()
    ).hexdigest()


def _assign_roles(records: list[dict[str, Any]]) -> dict[str, str]:
    cambridge = [record for record in records if record["site"] == "Cambridge"]
    held = [record for record in records if record["site"] == "Ncl"]
    if len(cambridge) != EXPECTED_CAMBRIDGE or len(held) != EXPECTED_HELD:
        raise PermissionError("eligible donor counts differ from the freeze")
    if {record["disease"] for record in cambridge} != {"COVID-19", "normal"}:
        raise PermissionError("Cambridge disease strata differ from the freeze")

    calibration: set[str] = set()
    pilot: set[str] = set()
    targets = {
        "COVID-19": (9, 18),
        "normal": (3, 6),
    }
    for disease, (calibration_count, pilot_count) in targets.items():
        group = [record for record in cambridge if record["disease"] == disease]
        first = sorted(
            group,
            key=lambda record: (
                _role_hash(
                    CALIBRATION_SALT,
                    disease,
                    record["donor"],
                    record["sample"],
                ),
                record["donor"],
            ),
        )
        chosen_calibration = first[:calibration_count]
        calibration.update(record["donor"] for record in chosen_calibration)
        remaining = first[calibration_count:]
        second = sorted(
            remaining,
            key=lambda record: (
                _role_hash(
                    PILOT_SALT,
                    disease,
                    record["donor"],
                    record["sample"],
                ),
                record["donor"],
            ),
        )
        pilot.update(record["donor"] for record in second[:pilot_count])

    roles = {}
    for record in records:
        donor = record["donor"]
        if record["site"] == "Ncl":
            roles[donor] = "held_site"
        elif donor in calibration:
            roles[donor] = "calibration"
        elif donor in pilot:
            roles[donor] = "pilot"
        else:
            roles[donor] = "unused_source"
    counts = {role: list(roles.values()).count(role) for role in set(roles.values())}
    expected = {
        "calibration": EXPECTED_CALIBRATION,
        "pilot": EXPECTED_PILOT,
        "unused_source": 11,
        "held_site": EXPECTED_HELD,
    }
    if counts != expected:
        raise AssertionError(f"role counts differ from freeze: {counts}")
    return roles


def _feature_columns(handle: h5py.File) -> dict[str, list[int]]:
    var = handle.get("var")
    if not isinstance(var, h5py.Group):
        raise ValueError("H5AD var dataframe is absent")
    names = _dataframe_index(handle, var)
    feature_types = _encoded_column(handle, var, "feature_types")
    if "gene_ids" in var:
        raise PermissionError("unexpected gene_ids column entered the symbol-only freeze")
    if not (len(names) == len(feature_types) == 24_929):
        raise ValueError("H5AD feature metadata axis differs from the freeze")
    result: dict[str, list[int]] = {"rna": [], "adt": []}
    for marker in MARKERS:
        rna = np.flatnonzero(
            (names == marker)
            & (feature_types == "Gene Expression")
        )
        adt = np.flatnonzero(
            (names == ADT_FEATURE[marker])
            & (feature_types == "Antibody Capture")
        )
        if len(rna) != 1 or len(adt) != 1:
            raise PermissionError(f"frozen feature pair is absent or duplicate: {marker}")
        result["rna"].append(int(rna[0]))
        result["adt"].append(int(adt[0]))
    return result


def _matrix_metadata(handle: h5py.File, path: str) -> dict[str, Any]:
    matrix = handle.get(path)
    if not isinstance(matrix, h5py.Group):
        raise ValueError(f"{path} is not a sparse matrix group")
    encoding = matrix.attrs.get("encoding-type")
    if isinstance(encoding, bytes):
        encoding = encoding.decode()
    shape = [int(value) for value in matrix.attrs.get("shape", ())]
    if encoding != "csr_matrix" or shape != [647_366, 24_929]:
        raise ValueError(f"{path} matrix encoding or shape differs")
    expected = {
        "data": 965_744_336 if path == "layers/raw" else 968_620_608,
        "indices": 965_744_336 if path == "layers/raw" else 968_620_608,
        "indptr": 647_367,
    }
    children = {}
    for name, length in expected.items():
        dataset = matrix.get(name)
        if not isinstance(dataset, h5py.Dataset) or dataset.shape != (length,):
            raise ValueError(f"{path}/{name} shape differs")
        children[name] = {"shape": [length], "dtype": str(dataset.dtype)}
    return {"encoding_type": encoding, "shape": shape, "datasets": children}


def _metadata_inventory(h5ad: Path, sdrf: Path) -> dict[str, Any]:
    if h5ad.name != OFFICIAL_H5AD_NAME or h5ad.stat().st_size != OFFICIAL_H5AD_BYTES:
        raise PermissionError("H5AD name or byte count differs from BioStudies")
    sdrf_rows = [
        {
            **row,
            "sdrf_sample": row["sample"],
            "sample": SDRF_TO_H5AD_SAMPLE.get(row["sample"], row["sample"]),
        }
        for row in _parse_sdrf(sdrf)
    ]
    by_sample = {row["sample"]: row for row in sdrf_rows}
    with h5py.File(h5ad, "r") as handle:
        obs = handle.get("obs")
        if not isinstance(obs, h5py.Group):
            raise ValueError("H5AD obs dataframe is absent")
        barcodes = _dataframe_index(handle, obs)
        sample_ids = _encoded_column(handle, obs, "sample_id")
        sites = _encoded_column(handle, obs, "Site")
        clusters = _encoded_column(handle, obs, "initial_clustering")
        if not (
            len(barcodes)
            == len(sample_ids)
            == len(sites)
            == len(clusters)
            == 647_366
        ):
            raise ValueError("H5AD observation metadata axes differ")
        if len(set(barcodes.tolist())) != len(barcodes):
            raise ValueError("H5AD observation names are not unique")
        if set(clusters) != set(INITIAL_CLUSTERS):
            raise PermissionError("initial_clustering vocabulary differs")
        feature_columns = _feature_columns(handle)
        matrix_metadata = {
            "X": _matrix_metadata(handle, "X"),
            "layers/raw": _matrix_metadata(handle, "layers/raw"),
        }

    sample_counts: dict[str, int] = {}
    sample_sites: dict[str, str] = {}
    for sample in sorted(set(sample_ids.tolist())):
        selected = sample_ids == sample
        observed_sites = set(sites[selected].tolist())
        if len(observed_sites) != 1:
            raise ValueError(f"sample has multiple sites: {sample}")
        sample_counts[sample] = int(np.count_nonzero(selected))
        sample_sites[sample] = next(iter(observed_sites))
    if set(sample_counts) != set(by_sample) or len(sample_counts) != 143:
        raise PermissionError("H5AD and SDRF sample universes differ")
    site_sample_counts = {
        site: sum(value == site for value in sample_sites.values())
        for site in sorted(set(sample_sites.values()))
    }
    if site_sample_counts != {"Cambridge": 60, "Ncl": 72, "Sanger": 11}:
        raise PermissionError("site sample counts differ from the freeze")

    donor_candidates: dict[tuple[str, str], list[dict[str, Any]]] = {}
    sanger_donors = set()
    for row in sdrf_rows:
        sample = row["sample"]
        site = sample_sites[sample]
        if site == "Sanger":
            sanger_donors.add(row["donor"])
            continue
        if site not in {"Cambridge", "Ncl"}:
            raise PermissionError("unexpected site entered source universe")
        if site == "Ncl" and row["stimulus"] == "LPS":
            continue
        if row["stimulus"] not in {"", "none"}:
            raise PermissionError("unexpected stimulus entered source universe")
        key = (site, row["donor"])
        donor_candidates.setdefault(key, []).append(
            {**row, "site": site, "eligible_pool_cells": sample_counts[sample]}
        )
    if len(sanger_donors) != EXPECTED_SANGER:
        raise PermissionError("Sanger donor count differs from metadata freeze")

    selected_records = []
    for (site, donor), candidates in sorted(donor_candidates.items()):
        disease = {row["disease"] for row in candidates}
        if len(disease) != 1:
            raise PermissionError(f"donor disease labels disagree: {donor}")
        eligible = [row for row in candidates if row["eligible_pool_cells"] >= CELL_BUDGET]
        if not eligible:
            raise PermissionError(f"donor lacks a 512-cell sample: {donor}")
        chosen = min(
            eligible,
            key=lambda row: (_sample_hash(donor, row["sample"]), row["sample"]),
        )
        selected_records.append(
            {
                "donor": donor,
                "sample": chosen["sample"],
                "sdrf_sample": chosen["sdrf_sample"],
                "site": site,
                "disease": next(iter(disease)),
                "eligible_pool_cells": chosen["eligible_pool_cells"],
                "eligible_sample_candidates": sorted(row["sample"] for row in eligible),
            }
        )
    roles = _assign_roles(selected_records)
    for record in selected_records:
        record["role"] = roles[record["donor"]]
    if len(selected_records) != EXPECTED_CAMBRIDGE + EXPECTED_HELD:
        raise AssertionError("selected donor universe has the wrong size")
    return {
        "samples": selected_records,
        "feature_columns": feature_columns,
        "matrix_metadata": matrix_metadata,
        "site_sample_counts": site_sample_counts,
        "h5ad_cells_by_site": {
            site: int(np.count_nonzero(sites == site))
            for site in sorted(set(sites.tolist()))
        },
        "sanger_donors_excluded": EXPECTED_SANGER,
    }


def seal_source(
    h5ad: Path,
    sdrf: Path,
    preflight_path: Path,
    source_path: Path,
) -> dict[str, Any]:
    """Hash opaque bytes and inspect metadata without touching matrix payloads."""

    for output in (preflight_path, source_path):
        if output.exists():
            raise FileExistsError(output)
    sha256, md5 = _opaque_hashes(h5ad)
    inventory = _metadata_inventory(h5ad, sdrf)
    preflight = {
        "schema": "stephenson-citeseq-metadata-preflight/1.0",
        "status": "PREFLIGHT_METADATA_CONTRACT_PASS",
        "created_at_utc": _timestamp(),
        "input": {
            "filename": h5ad.name,
            "bytes": h5ad.stat().st_size,
            "sha256": sha256,
            "md5": md5,
            "official_url": OFFICIAL_H5AD_URL,
        },
        "sdrf": {
            "filename": sdrf.name,
            "bytes": sdrf.stat().st_size,
            "sha256": _sha256(sdrf),
            "official_url": OFFICIAL_SDRF_URL,
        },
        "sdrf_to_h5ad_sample_corrections": SDRF_TO_H5AD_SAMPLE,
        "inventory": inventory,
        "access_audit": {
            "opaque_file_hashing_performed": True,
            "obs_metadata_values_read": True,
            "var_metadata_values_read": True,
            "matrix_shapes_and_encoding_metadata_read": True,
            "matrix_data_values_read": 0,
            "matrix_indices_values_read": 0,
            "matrix_indptr_values_read": 0,
        },
    }
    _write_json(preflight_path, preflight)
    source = {
        "schema": "stephenson-citeseq-source-manifest/1.0",
        "status": "SOURCE_SEALED_OUTCOME_ACCESS_DISABLED",
        "created_at_utc": _timestamp(),
        "preflight_sha256": _sha256(preflight_path),
        "h5ad": {
            "filename": h5ad.name,
            "bytes": h5ad.stat().st_size,
            "sha256": sha256,
            "md5": md5,
            "official_url": OFFICIAL_H5AD_URL,
            "raw_matrix_path": "layers/raw",
        },
        "sdrf": {
            "filename": sdrf.name,
            "bytes": sdrf.stat().st_size,
            "sha256": _sha256(sdrf),
            "official_url": OFFICIAL_SDRF_URL,
        },
        "markers": [
            {
                "marker": marker,
                "canonical_reference_ensembl": RNA_ENSEMBL[marker],
                "rna_feature": marker,
                "rna_feature_type": "Gene Expression",
                "adt_feature": ADT_FEATURE[marker],
                "adt_feature_type": "Antibody Capture",
            }
            for marker in MARKERS
        ],
        "cluster_to_cell_type": CLUSTER_TO_CELL_TYPE,
        "salts": {
            "sample_selection": SAMPLE_SELECTION_SALT,
            "calibration": CALIBRATION_SALT,
            "pilot": PILOT_SALT,
            "cell_selection": CELL_SELECTION_SALT,
            "adt_tie": ADT_TIE_SALT,
            "destroyed_link": DESTROYED_LINK_SALT,
        },
        "samples": inventory["samples"],
    }
    serialized = json.dumps(source, sort_keys=True, allow_nan=False)
    if "/Users/" in serialized or "file://" in serialized or '"path"' in serialized:
        raise ValueError("public source manifest contains a local path")
    _write_json(source_path, source)
    return source


def _resolved_h5ad(source: dict[str, Any]) -> Path:
    raw = os.environ.get("STEPHENSON_CITESEQ_H5AD")
    if not raw:
        raise ValueError("set STEPHENSON_CITESEQ_H5AD to the sealed H5AD")
    path = Path(raw).expanduser().resolve()
    record = source.get("h5ad")
    if not path.is_file() or not isinstance(record, dict):
        raise FileNotFoundError(path)
    if (
        path.name != record.get("filename")
        or path.stat().st_size != record.get("bytes")
        or record.get("bytes") != OFFICIAL_H5AD_BYTES
        or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256")))
        or not re.fullmatch(r"[0-9a-f]{32}", str(record.get("md5")))
    ):
        raise PermissionError("runtime H5AD differs from the source seal")
    return path


def _validated_source(source_path: Path, *, verify_hash: bool) -> dict[str, Any]:
    source = _read_json(source_path)
    if (
        source.get("schema") != "stephenson-citeseq-source-manifest/1.0"
        or source.get("status") != "SOURCE_SEALED_OUTCOME_ACCESS_DISABLED"
        or source.get("preflight_sha256") != _sha256(DEFAULT_PREFLIGHT)
        or source.get("sdrf", {}).get("sha256") != OFFICIAL_SDRF_SHA256
        or source.get("sdrf_to_h5ad_sample_corrections") != SDRF_TO_H5AD_SAMPLE
        or source.get("cluster_to_cell_type") != CLUSTER_TO_CELL_TYPE
    ):
        raise PermissionError("source manifest differs from the freeze")
    records = source.get("samples")
    if not isinstance(records, list) or len(records) != 103:
        raise PermissionError("source manifest must contain 103 selected donors")
    donors = [record.get("donor") for record in records if isinstance(record, dict)]
    samples = [record.get("sample") for record in records if isinstance(record, dict)]
    if len(set(donors)) != 103 or len(set(samples)) != 103:
        raise PermissionError("source donor or sample identifiers repeat")
    roles = _assign_roles(records)
    for record in records:
        if (
            set(record)
            != {
                "donor",
                "sample",
                "sdrf_sample",
                "site",
                "disease",
                "eligible_pool_cells",
                "eligible_sample_candidates",
                "role",
            }
            or record.get("role") != roles[record["donor"]]
            or not isinstance(record.get("eligible_pool_cells"), int)
            or record["eligible_pool_cells"] < CELL_BUDGET
        ):
            raise PermissionError("source sample record differs from the freeze")
    h5ad = _resolved_h5ad(source)
    if verify_hash:
        sha256, md5 = _opaque_hashes(h5ad)
        if sha256 != source["h5ad"]["sha256"] or md5 != source["h5ad"]["md5"]:
            raise PermissionError("runtime H5AD digest differs from the seal")
    return {
        "payload": source,
        "records": records,
        "by_sample": {record["sample"]: record for record in records},
        "roles": roles,
        "h5ad": h5ad,
        "source_sha256": _sha256(source_path),
    }


def verify_public_freeze(commit: str, output_path: Path) -> dict[str, Any]:
    """Verify every non-self freeze artifact from one immutable public commit."""

    if output_path != DEFAULT_VERIFICATION or output_path.exists():
        raise PermissionError("fresh-clone verification path is fixed and one-shot")
    bindings = {}
    for label, relative in DEVELOPMENT_BINDING_PATHS.items():
        if label == "fresh_clone_verification":
            continue
        local = _bound_path(relative)
        if not local.is_file():
            raise FileNotFoundError(local)
        public = _immutable_public_bytes(relative, commit, label)
        local_bytes = local.read_bytes()
        if public != local_bytes:
            raise PermissionError(f"public freeze bytes differ: {label}")
        bindings[label] = {"path": relative, "sha256": _sha256(local)}
    payload = {
        "schema": "stephenson-citeseq-public-freeze-verification/1.0",
        "status": "PASS",
        "created_at_utc": _timestamp(),
        "fresh_clone": True,
        "canonical_origin": PUBLIC_ORIGIN,
        "public_freeze_commit": commit,
        "planned_immutable_tag": "stephenson-citeseq-v1-protocol",
        "artifact_bindings": bindings,
        "all_bound_artifacts_match": True,
        "matrix_payload_reads": 0,
    }
    _write_json(output_path, payload)
    return payload


def _validated_development_authorization(
    authorization_path: Path,
    source_path: Path,
    authorization_commit: str,
) -> dict[str, Any]:
    authorization = _read_json(authorization_path)
    if (
        authorization.get("schema")
        != "stephenson-citeseq-development-authorization/1.0"
        or authorization.get("status") != "OUTCOME_ACCESS_AUTHORIZED"
    ):
        raise PermissionError("development outcome access is disabled")
    expected_keys = {
        "schema",
        "status",
        "created_at_utc",
        "public_freeze_commit",
        "public_verification_commit",
        "artifact_bindings",
    }
    if set(authorization) != expected_keys:
        raise PermissionError("development authorization fields differ")
    freeze_commit = authorization.get("public_freeze_commit")
    verification_commit = authorization.get("public_verification_commit")
    if not isinstance(freeze_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", freeze_commit
    ):
        raise PermissionError("development authorization freeze commit is invalid")
    if not isinstance(verification_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", verification_commit
    ):
        raise PermissionError("development verification commit is invalid")
    raw_bindings = authorization.get("artifact_bindings")
    if not isinstance(raw_bindings, dict) or set(raw_bindings) != set(
        DEVELOPMENT_BINDING_PATHS
    ):
        raise PermissionError("development authorization binding set differs")
    resolved = {}
    for label, relative in DEVELOPMENT_BINDING_PATHS.items():
        row = raw_bindings.get(label)
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "sha256"}
            or row.get("path") != relative
        ):
            raise PermissionError(f"development binding differs: {label}")
        local = _bound_path(relative)
        if not local.is_file() or row.get("sha256") != _sha256(local):
            raise PermissionError(f"development binding hash differs: {label}")
        public_commit = (
            verification_commit if label == "fresh_clone_verification" else freeze_commit
        )
        if _immutable_public_bytes(relative, public_commit, label) != local.read_bytes():
            raise PermissionError(f"immutable public binding differs: {label}")
        resolved[label] = row["sha256"]
    verification = _read_json(DEFAULT_VERIFICATION)
    nonself = {
        label: row
        for label, row in raw_bindings.items()
        if label != "fresh_clone_verification"
    }
    if (
        verification.get("schema")
        != "stephenson-citeseq-public-freeze-verification/1.0"
        or verification.get("status") != "PASS"
        or verification.get("fresh_clone") is not True
        or verification.get("canonical_origin") != PUBLIC_ORIGIN
        or verification.get("public_freeze_commit") != freeze_commit
        or verification.get("artifact_bindings") != nonself
        or verification.get("all_bound_artifacts_match") is not True
        or verification.get("matrix_payload_reads") != 0
    ):
        raise PermissionError("fresh-clone verification differs from authorization")
    if _sha256(source_path) != resolved["source_manifest"]:
        raise PermissionError("runtime source manifest differs from authorization")
    if _immutable_public_bytes(
        _relative(authorization_path), authorization_commit, "development authorization"
    ) != authorization_path.read_bytes():
        raise PermissionError("public development authorization bytes differ")
    return {
        "authorization_sha256": _sha256(authorization_path),
        "public_authorization_commit": authorization_commit,
        "public_freeze_commit": freeze_commit,
        "public_verification_commit": verification_commit,
        "binding_sha256": resolved,
    }


def _cell_hash(donor: str, sample: str, barcode: str) -> str:
    return hashlib.sha256(
        "\0".join((CELL_SELECTION_SALT, donor, sample, barcode)).encode()
    ).hexdigest()


def _selected_rows(
    h5ad: Path, records: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    with h5py.File(h5ad, "r") as handle:
        obs = handle.get("obs")
        if not isinstance(obs, h5py.Group):
            raise ValueError("H5AD obs dataframe is absent")
        barcodes = _dataframe_index(handle, obs)
        sample_ids = _encoded_column(handle, obs, "sample_id")
        sites = _encoded_column(handle, obs, "Site")
        clusters = _encoded_column(handle, obs, "initial_clustering")
    if len(set(barcodes.tolist())) != len(barcodes):
        raise ValueError("H5AD observation names are not unique")
    mapped = np.asarray([CLUSTER_TO_CELL_TYPE.get(value, "") for value in clusters])
    if np.any(mapped == ""):
        raise PermissionError("unmapped initial_clustering label entered selection")
    selected = {}
    for record in records:
        rows = np.flatnonzero(sample_ids == record["sample"])
        if not len(rows) or set(sites[rows].tolist()) != {record["site"]}:
            raise PermissionError(f"sample metadata differs: {record['sample']}")
        eligible = rows[mapped[rows] != ""]
        if len(eligible) != record["eligible_pool_cells"] or len(eligible) < CELL_BUDGET:
            raise PermissionError(f"eligible pool differs: {record['sample']}")
        ordered = sorted(
            eligible.tolist(),
            key=lambda row: (
                _cell_hash(record["donor"], record["sample"], barcodes[row]),
                barcodes[row],
            ),
        )[:CELL_BUDGET]
        matrix_rows = np.asarray(sorted(ordered), dtype=np.int64)
        chosen_barcodes = barcodes[matrix_rows]
        selected[record["sample"]] = {
            "rows": matrix_rows,
            "barcodes": chosen_barcodes,
            "cell_types": mapped[matrix_rows],
            "eligible_pool_cells": int(len(eligible)),
            "selected_barcode_sha256": hashlib.sha256(
                ("\n".join(sorted(chosen_barcodes.tolist())) + "\n").encode()
            ).hexdigest(),
        }
    return selected


def _read_modality(
    h5ad: Path,
    selections: dict[str, dict[str, Any]],
    samples: tuple[str, ...],
    modality: str,
) -> dict[str, np.ndarray]:
    if modality not in {"rna", "adt"}:
        raise ValueError("modality must be rna or adt")
    all_rows = np.concatenate([selections[sample]["rows"] for sample in samples])
    row_order = np.argsort(all_rows, kind="mergesort")
    sorted_rows = all_rows[row_order]
    if np.any(np.diff(sorted_rows) == 0):
        raise ValueError("selected sample rows overlap")
    with h5py.File(h5ad, "r") as handle:
        columns = np.asarray(_feature_columns(handle)[modality], dtype=np.int64)
        matrix = handle.get("layers/raw")
        if not isinstance(matrix, (h5py.Group, h5py.Dataset)):
            raise ValueError("raw matrix is absent")
        values = numerics._read_csr_feature_subset(matrix, sorted_rows, columns)
    unsorted = np.empty_like(values)
    unsorted[row_order] = values
    result = {}
    offset = 0
    for sample in samples:
        result[sample] = unsorted[offset : offset + CELL_BUDGET].T
        offset += CELL_BUDGET
    return result


def _adt_states(
    counts: np.ndarray,
    barcodes: np.ndarray,
    donor: str,
    sample: str,
) -> np.ndarray:
    values = numerics._integer_counts(counts, "ADT")
    barcode_values = _decode_strings(barcodes)
    if values.shape != (len(MARKERS), CELL_BUDGET):
        raise ValueError("ADT state input shape differs")
    states = np.ones_like(values, dtype=np.uint8)
    for marker_index, marker in enumerate(MARKERS):
        tie = np.asarray(
            [
                hashlib.sha256(
                    "\0".join(
                        (ADT_TIE_SALT, donor, sample, barcode, marker)
                    ).encode()
                ).hexdigest()
                for barcode in barcode_values
            ]
        )
        order = np.lexsort((tie, values[marker_index]))
        states[marker_index, order[: CELL_BUDGET // 2]] = 0
    if np.any(states.sum(axis=1) != CELL_BUDGET // 2):
        raise AssertionError("ADT midrank margins differ from 256/256")
    return states


def _destroyed_adt(states: np.ndarray, barcodes: np.ndarray, sample: str) -> np.ndarray:
    barcode_values = _decode_strings(barcodes)
    order = sorted(
        range(CELL_BUDGET),
        key=lambda index: (
            hashlib.sha256(
                "\0".join(
                    (DESTROYED_LINK_SALT, sample, barcode_values[index])
                ).encode()
            ).hexdigest(),
            barcode_values[index],
        ),
    )
    destroyed = np.empty_like(states)
    for index, destination in enumerate(order):
        destroyed[:, destination] = states[:, order[(index + 1) % CELL_BUDGET]]
    return destroyed


def _development_records(source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = [
        record
        for record in source["records"]
        if record["role"] in {"calibration", "pilot"}
    ]
    samples = tuple(record["sample"] for record in sorted(records, key=lambda row: row["donor"]))
    if len(samples) != EXPECTED_CALIBRATION + EXPECTED_PILOT:
        raise AssertionError("development sample count differs")
    selections = _selected_rows(source["h5ad"], records)
    rna_counts = _read_modality(source["h5ad"], selections, samples, "rna")
    rna_states = {
        sample: (numerics._integer_counts(rna_counts[sample], "RNA") > 0).astype(
            np.uint8
        )
        for sample in samples
    }
    adt_counts = _read_modality(source["h5ad"], selections, samples, "adt")
    by_sample = {record["sample"]: record for record in records}
    adt_states = {
        sample: _adt_states(
            adt_counts[sample],
            selections[sample]["barcodes"],
            by_sample[sample]["donor"],
            sample,
        )
        for sample in samples
    }
    tables = numerics._form_tables(rna_states, adt_states, list(samples))
    destroyed_states = {
        sample: _destroyed_adt(
            adt_states[sample], selections[sample]["barcodes"], sample
        )
        for sample in samples
    }
    destroyed = numerics._form_tables(rna_states, destroyed_states, list(samples))
    if not np.array_equal(tables.sum(axis=-1), destroyed.sum(axis=-1)) or not np.array_equal(
        tables.sum(axis=-2), destroyed.sum(axis=-2)
    ):
        raise AssertionError("destroyed-link control changed a fixed margin")
    output = {}
    for index, sample in enumerate(samples):
        support = numerics._informative(tables[index])
        if int(support.sum()) < numerics.MINIMUM_INFORMATIVE_ENTITIES:
            raise ValueError(f"sample has fewer than 64 informative entities: {sample}")
        record = by_sample[sample]
        output[sample] = {
            "sample": sample,
            "donor": record["donor"],
            "role": record["role"],
            "cells": CELL_BUDGET,
            "eligible_pool_cells": selections[sample]["eligible_pool_cells"],
            "selected_barcode_sha256": selections[sample][
                "selected_barcode_sha256"
            ],
            "strata": numerics._stratum_profiles(
                rna_states[sample],
                adt_counts[sample],
                selections[sample]["cell_types"],
            ),
            "tables": tables[index].reshape(len(MARKERS) ** 2, 4).tolist(),
            "destroyed_tables": destroyed[index]
            .reshape(len(MARKERS) ** 2, 4)
            .tolist(),
            "informative": support.reshape(-1).tolist(),
        }
    return output


def _development_samples(source: dict[str, Any], role: str) -> tuple[str, ...]:
    selected = [record for record in source["records"] if record["role"] == role]
    return tuple(record["sample"] for record in sorted(selected, key=lambda row: row["donor"]))


def _conditional_configuration(
    config: tuple[int, float, float, float], alpha: float,
) -> dict[str, Any]:
    neighbors, heterogeneity, ridge, graph = config
    return {
        "graph_neighbors": int(neighbors),
        "heterogeneity_penalty": float(heterogeneity),
        "ridge_penalty": float(ridge),
        "graph_penalty": float(graph),
        "transport_alpha": float(alpha),
        "minimum_informative_donors": 2,
        "maximum_condition_number": MAXIMUM_CONDITION_NUMBER,
        "gradient_tolerance": 1e-8,
    }


def _conditional_model(
    records: dict[str, dict[str, Any]],
    samples: tuple[str, ...],
    tables: np.ndarray,
    config: tuple[int, float, float, float],
    *,
    alpha: float,
    label: str,
) -> dict[str, Any]:
    neighbors, heterogeneity, ridge, graph_penalty = config
    first, second, graph = numerics._graphs(records, samples, neighbors)
    fit = fit_hierarchical_conditional_log_odds(
        tables,
        first,
        second,
        heterogeneity_penalty=heterogeneity,
        ridge_penalty=ridge,
        graph_penalty=graph_penalty,
        minimum_informative_donors=2,
        maximum_condition_number=MAXIMUM_CONDITION_NUMBER,
        tolerance=1e-8,
    )
    coordinate = np.asarray(fit.population_log_odds, dtype=float)
    if coordinate.shape != (len(MARKERS), len(MARKERS)) or not np.isfinite(
        coordinate
    ).all():
        raise CouplingEstimationRefusal("conditional population coordinate is invalid")
    return {
        "kind": "conditional_log_odds",
        "estimator": label,
        "source_coordinate": coordinate.reshape(-1).tolist(),
        "alpha": float(alpha),
        "selected_configuration": _conditional_configuration(config, alpha),
        "graph": graph,
        "certificate": {
            "optimizer": fit.optimizer,
            "converged": bool(fit.converged),
            "iterations": int(fit.iterations),
            "scaled_gradient_norm": float(fit.scaled_gradient_norm),
            "schur_condition_number": float(fit.schur_condition_number),
            "theta_curvature_condition_number": float(
                fit.theta_curvature_condition_number
            ),
            "support_count_range": [
                int(np.min(fit.support_count)),
                int(np.max(fit.support_count)),
            ],
            "prediction": "exact noncentral-hypergeometric expectation at recipient margins",
        },
    }


def _canonical_table(rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    total = int(rows.sum())
    upper_left = max(0, int(rows[0] + columns[0] - total))
    return np.asarray(
        [
            [upper_left, int(rows[0] - upper_left)],
            [int(columns[0] - upper_left), int(rows[1] - columns[0] + upper_left)],
        ],
        dtype=np.int64,
    )


def _classical_table(
    statistic: float,
    rows: np.ndarray,
    columns: np.ndarray,
    family: str,
) -> tuple[np.ndarray, bool]:
    total = float(rows.sum())
    lower = float(max(0, rows[0] + columns[0] - total))
    upper = float(min(rows[0], columns[0]))
    if upper <= lower:
        return _canonical_table(rows, columns).astype(float), False
    epsilon = min(1e-10, 0.25 * (upper - lower))
    clipped = False
    if family == "pearson":
        scale = math.sqrt(total / float(rows[0] * rows[1] * columns[0] * columns[1]))
        upper_left = (float(statistic) / scale + rows[0] * columns[0]) / total
    elif family == "deviance":
        expected = np.outer(rows, columns) / total

        def signed_root(value: float) -> float:
            table = np.asarray(
                [
                    [value, rows[0] - value],
                    [columns[0] - value, rows[1] - columns[0] + value],
                ],
                dtype=float,
            )
            positive = table > 0.0
            terms = np.zeros((2, 2), dtype=float)
            terms[positive] = table[positive] * np.log(
                table[positive] / expected[positive]
            )
            determinant = table[0, 0] * table[1, 1] - table[0, 1] * table[1, 0]
            return math.copysign(
                math.sqrt(max(2.0 * float(terms.sum()), 0.0)), determinant
            )

        left = lower + epsilon
        right = upper - epsilon
        left_statistic = signed_root(left)
        right_statistic = signed_root(right)
        target = min(max(float(statistic), left_statistic), right_statistic)
        clipped = target != float(statistic)
        for _ in range(80):
            midpoint = 0.5 * (left + right)
            if signed_root(midpoint) < target:
                left = midpoint
            else:
                right = midpoint
        upper_left = 0.5 * (left + right)
    else:
        raise ValueError("classical family must be pearson or deviance")
    bounded = min(max(float(upper_left), lower + epsilon), upper - epsilon)
    clipped = clipped or bounded != float(upper_left)
    return (
        np.asarray(
            [
                [bounded, rows[0] - bounded],
                [columns[0] - bounded, rows[1] - columns[0] + bounded],
            ]
        ),
        clipped,
    )


def _predict_method(
    model: dict[str, Any],
    row_margins: np.ndarray,
    column_margins: np.ndarray,
    *,
    boundary_flags: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    rows = np.asarray(row_margins, dtype=np.int64)
    columns = np.asarray(column_margins, dtype=np.int64)
    if rows.shape != (len(MARKERS), 2) or columns.shape != rows.shape:
        raise ValueError("recipient margins have invalid shape")
    kind = model.get("kind")
    output = np.empty((len(MARKERS), len(MARKERS), 2, 2), dtype=float)
    coordinate = None
    if kind != "independence":
        coordinate = np.asarray(model.get("source_coordinate"), dtype=float)
        if coordinate.shape != (len(MARKERS) ** 2,) or not np.isfinite(
            coordinate
        ).all():
            raise ValueError("frozen source coordinate is invalid")
        coordinate = coordinate.reshape(len(MARKERS), len(MARKERS))
    for first in range(len(MARKERS)):
        for second in range(len(MARKERS)):
            row = rows[first]
            column = columns[second]
            if row.sum() != column.sum() or row.sum() <= 0:
                raise ValueError("recipient margins are incompatible")
            if kind == "conditional_log_odds":
                output[first, second] = expected_binary_table_from_log_odds(
                    float(model.get("alpha")) * float(coordinate[first, second]),
                    row,
                    column,
                )
            elif kind == "classical_residual":
                family = model.get("family")
                if family not in {"pearson", "deviance"}:
                    raise ValueError("frozen classical family is invalid")
                statistic = (
                    float(model.get("alpha"))
                    * float(coordinate[first, second])
                    * math.sqrt(float(row.sum()))
                )
                if model.get("centered") is True:
                    null = numerics.centered_classical_coordinate(
                        _canonical_table(row, column), statistic=family
                    )
                    statistic += null.null_mean_coordinate
                table, clipped = _classical_table(statistic, row, column, family)
                output[first, second] = table
                if clipped and boundary_flags is not None:
                    boundary_flags.append(
                        {
                            "rna_marker": MARKERS[first],
                            "adt_marker": MARKERS[second],
                            "status": "classical_inversion_projected_to_feasible_boundary",
                        }
                    )
            elif kind == "independence":
                output[first, second] = np.outer(row, column) / float(row.sum())
            else:
                raise ValueError(f"unknown frozen model kind: {kind}")
            predicted = output[first, second]
            if (
                not np.isfinite(predicted).all()
                or np.any(predicted < 0.0)
                or not np.allclose(predicted.sum(axis=1), row, rtol=0.0, atol=1e-10)
                or not np.allclose(
                    predicted.sum(axis=0), column, rtol=0.0, atol=1e-10
                )
            ):
                raise FloatingPointError("prediction is not a valid margin-preserving table")
    return output


def _model_losses(model: dict[str, Any], target_tables: np.ndarray) -> np.ndarray:
    values = np.asarray(target_tables)
    losses = np.empty(len(values), dtype=float)
    for index, truth in enumerate(values):
        rows, columns = numerics._sample_margins(truth)
        prediction = _predict_method(model, rows, columns)
        losses[index] = numerics._donor_loss(
            truth, prediction, numerics._informative(truth).reshape(-1)
        )
    if not np.isfinite(losses).all():
        raise FloatingPointError("candidate produced a nonfinite pilot loss")
    return losses


def _fit_method_panel(
    records: dict[str, dict[str, Any]],
    samples: tuple[str, ...],
    selected_config: tuple[int, float, float, float, float],
    zero_config: tuple[int, float, float, float, float],
    residual_choice: tuple[str, bool, float],
) -> dict[str, dict[str, Any]]:
    tables = numerics._tables(records, samples, "tables")
    destroyed_tables = numerics._tables(records, samples, "destroyed_tables")
    return {
        "primary": _conditional_model(
            records,
            samples,
            tables,
            selected_config[:4],
            alpha=selected_config[4],
            label="hierarchical exact conditional full-log-odds field",
        ),
        "best_residual": {
            **numerics._classical_model(tables, *residual_choice[:2]),
            "alpha": residual_choice[2],
        },
        "destroyed_link": _conditional_model(
            records,
            samples,
            destroyed_tables,
            selected_config[:4],
            alpha=selected_config[4],
            label="destroyed-link hierarchical exact conditional full-log-odds field",
        ),
        "hierarchical_graph_zero": _conditional_model(
            records,
            samples,
            tables,
            zero_config[:4],
            alpha=zero_config[4],
            label="graph-zero hierarchical exact conditional full-log-odds field",
        ),
        "independence": {
            "kind": "independence",
            "estimator": "fixed-margin conditional independence",
        },
    }


def _pilot_analysis(
    records: dict[str, dict[str, Any]],
    calibration_samples: tuple[str, ...],
    pilot_samples: tuple[str, ...],
) -> dict[str, Any]:
    calibration_tables = numerics._tables(records, calibration_samples, "tables")
    pilot_tables = numerics._tables(records, pilot_samples, "tables")
    candidate_rows = []
    for config in CONDITIONAL_GRID:
        try:
            base_model = _conditional_model(
                records,
                calibration_samples,
                calibration_tables,
                config,
                alpha=1.0,
                label="hierarchical exact conditional full-log-odds field",
            )
        except (CouplingEstimationRefusal, ValueError, FloatingPointError) as error:
            for alpha in ALPHA_GRID:
                candidate_rows.append(
                    {
                        "configuration": _conditional_configuration(config, alpha),
                        "status": "REFUSED",
                        "reason": str(error),
                    }
                )
            continue
        for alpha in ALPHA_GRID:
            model = {
                **base_model,
                "alpha": alpha,
                "selected_configuration": _conditional_configuration(config, alpha),
            }
            try:
                losses = _model_losses(model, pilot_tables)
            except (ValueError, FloatingPointError) as error:
                candidate_rows.append(
                    {
                        "configuration": _conditional_configuration(config, alpha),
                        "status": "REFUSED",
                        "reason": str(error),
                    }
                )
            else:
                candidate_rows.append(
                    {
                        "configuration": _conditional_configuration(config, alpha),
                        "status": "EVALUATED",
                        "mean_pilot_deviance_per_cell": float(losses.mean()),
                        "pilot_losses": {
                            sample: float(value)
                            for sample, value in zip(pilot_samples, losses)
                        },
                    }
                )
    successful = [row for row in candidate_rows if row["status"] == "EVALUATED"]
    successful_zero = [
        row
        for row in successful
        if row["configuration"]["graph_penalty"] == 0.0
    ]

    classical_rows = []
    for family, centered in CLASSICAL_GRID:
        try:
            base_model = numerics._classical_model(
                calibration_tables, family, centered
            )
        except (ValueError, FloatingPointError) as error:
            for alpha in ALPHA_GRID:
                classical_rows.append(
                    {
                        "family": family,
                        "centered": centered,
                        "alpha": alpha,
                        "status": "REFUSED",
                        "reason": str(error),
                    }
                )
            continue
        for alpha in ALPHA_GRID:
            model = {**base_model, "alpha": alpha}
            try:
                losses = _model_losses(model, pilot_tables)
            except (ValueError, FloatingPointError) as error:
                classical_rows.append(
                    {
                        "family": family,
                        "centered": centered,
                        "alpha": alpha,
                        "status": "REFUSED",
                        "reason": str(error),
                    }
                )
            else:
                classical_rows.append(
                    {
                        "family": family,
                        "centered": centered,
                        "alpha": alpha,
                        "status": "EVALUATED",
                        "mean_pilot_deviance_per_cell": float(losses.mean()),
                        "pilot_losses": {
                            sample: float(value)
                            for sample, value in zip(pilot_samples, losses)
                        },
                    }
                )
    successful_classical = [
        row for row in classical_rows if row["status"] == "EVALUATED"
    ]
    if not successful or not successful_zero or not successful_classical:
        return {
            "status": "PILOT_FAIL",
            "terminal_failure": {
                "stage": "candidate_availability",
                "reason": "a frozen estimator family had no numerically valid candidate",
                "held_margin_access_authorized": False,
                "held_outcome_access_authorized": False,
            },
            "configuration_grid": [
                _conditional_configuration(config, alpha)
                for config in CONDITIONAL_GRID
                for alpha in ALPHA_GRID
            ],
            "primary_candidate_evaluations": candidate_rows,
            "classical_candidate_evaluations": classical_rows,
            "selection": None,
            "pilot_losses": {},
            "pilot_comparisons": {},
            "promotion_comparators": list(PROMOTION_COMPARATORS),
            "passes_pilot_gate": False,
            "frozen_source_models": None,
            "all_development_graph": None,
        }

    selected = min(
        successful,
        key=lambda row: (
            row["mean_pilot_deviance_per_cell"],
            row["configuration"]["graph_neighbors"],
            row["configuration"]["heterogeneity_penalty"],
            row["configuration"]["ridge_penalty"],
            row["configuration"]["graph_penalty"],
            row["configuration"]["transport_alpha"],
        ),
    )
    selected_zero = min(
        successful_zero,
        key=lambda row: (
            row["mean_pilot_deviance_per_cell"],
            row["configuration"]["graph_neighbors"],
            row["configuration"]["heterogeneity_penalty"],
            row["configuration"]["ridge_penalty"],
            row["configuration"]["transport_alpha"],
        ),
    )
    selected_classical = min(
        successful_classical,
        key=lambda row: (
            row["mean_pilot_deviance_per_cell"],
            row["family"],
            row["centered"],
            row["alpha"],
        ),
    )

    def frozen_config(
        row: dict[str, Any],
    ) -> tuple[int, float, float, float, float]:
        config = row["configuration"]
        return (
            int(config["graph_neighbors"]),
            float(config["heterogeneity_penalty"]),
            float(config["ridge_penalty"]),
            float(config["graph_penalty"]),
            float(config["transport_alpha"]),
        )

    selected_config = frozen_config(selected)
    zero_config = frozen_config(selected_zero)
    residual_choice = (
        str(selected_classical["family"]),
        bool(selected_classical["centered"]),
        float(selected_classical["alpha"]),
    )
    pilot_models = _fit_method_panel(
        records,
        calibration_samples,
        selected_config,
        zero_config,
        residual_choice,
    )
    pilot_losses = {
        name: _model_losses(model, pilot_tables)
        for name, model in pilot_models.items()
    }
    comparisons = {
        name: numerics._comparison(
            pilot_samples,
            pilot_losses["primary"],
            pilot_losses[name],
            favorable_required=PILOT_FAVORABLE if name in PROMOTION_COMPARATORS else None,
        )
        for name in METHODS[1:]
    }
    passes = all(comparisons[name]["passes"] for name in PROMOTION_COMPARATORS)
    frozen_models = None
    all_graph = None
    if passes:
        frozen_models = _fit_method_panel(
            records,
            calibration_samples + pilot_samples,
            selected_config,
            zero_config,
            residual_choice,
        )
        all_graph = frozen_models["primary"]["graph"]
    return {
        "status": "PILOT_PASS" if passes else "PILOT_FAIL",
        "terminal_failure": None,
        "configuration_grid": [
            _conditional_configuration(config, alpha)
            for config in CONDITIONAL_GRID
            for alpha in ALPHA_GRID
        ],
        "primary_candidate_evaluations": candidate_rows,
        "classical_candidate_evaluations": classical_rows,
        "selection": {
            "selected_primary_configuration": _conditional_configuration(
                selected_config[:4], selected_config[4]
            ),
            "selected_graph_zero_configuration": _conditional_configuration(
                zero_config[:4], zero_config[4]
            ),
            "selected_classical_residual": {
                "family": residual_choice[0],
                "centered": residual_choice[1],
                "transport_alpha": residual_choice[2],
            },
            "fit_samples": list(calibration_samples),
            "selection_samples": list(pilot_samples),
            "refit_samples_after_gate": list(calibration_samples + pilot_samples),
            "retuned_after_gate": False,
        },
        "pilot_losses": {
            name: {
                sample: float(value) for sample, value in zip(pilot_samples, losses)
            }
            for name, losses in pilot_losses.items()
        },
        "pilot_comparisons": comparisons,
        "promotion_comparators": list(PROMOTION_COMPARATORS),
        "passes_pilot_gate": passes,
        "frozen_source_models": frozen_models,
        "all_development_graph": all_graph,
    }


def run_development(
    source_path: Path,
    authorization_path: Path,
    authorization_commit: str,
    attempt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Run the authorized Cambridge calibration and adaptive pilot once."""

    if attempt_path.exists() or output_path.exists():
        raise FileExistsError("development is one-shot")
    authorization = _validated_development_authorization(
        authorization_path, source_path, authorization_commit
    )
    source = _validated_source(source_path, verify_hash=True)
    attempt = {
        "schema": "stephenson-citeseq-development-attempt/1.0",
        "status": "TERMINAL_ATTEMPT_STARTED",
        "created_at_utc": _timestamp(),
        "source_manifest_sha256": source["source_sha256"],
        "authorization_sha256": authorization["authorization_sha256"],
        "runner_sha256": _sha256(Path(__file__)),
        "numeric_access_begins_after_this_record": True,
        "held_matrix_rows_authorized": False,
        "rerun_permitted": False,
    }
    _write_json(attempt_path, attempt)
    records = _development_records(source)
    calibration = _development_samples(source, "calibration")
    pilot = _development_samples(source, "pilot")
    analysis = _pilot_analysis(records, calibration, pilot)
    payload = {
        "schema": "stephenson-citeseq-development/1.0",
        "created_at_utc": _timestamp(),
        "source_manifest_sha256": source["source_sha256"],
        "h5ad_sha256": source["payload"]["h5ad"]["sha256"],
        "development_authorization": authorization,
        "development_attempt": {
            "path": _relative(attempt_path),
            "sha256": _sha256(attempt_path),
        },
        "runner_sha256": _sha256(Path(__file__)),
        "combat_data_and_comparator_utility_sha256": _sha256(
            ROOT / DEVELOPMENT_BINDING_PATHS["combat_data_and_comparator_utility"]
        ),
        "markers": list(MARKERS),
        "calibration_samples": list(calibration),
        "pilot_samples": list(pilot),
        "access_audit": {
            "calibration_samples_read": EXPECTED_CALIBRATION,
            "pilot_samples_read": EXPECTED_PILOT,
            "held_site_matrix_rows_read": 0,
            "modalities_read_sequentially": ["rna", "adt"],
            "matrix_values_decoded_only_for_frozen_features": 18,
        },
        **analysis,
    }
    _write_json(output_path, payload)
    return payload


def _validated_development(
    path: Path, source_path: Path, *, require_pass: bool
) -> dict[str, Any]:
    payload = _read_json(path)
    source = _validated_source(source_path, verify_hash=False)
    if (
        payload.get("schema") != "stephenson-citeseq-development/1.0"
        or payload.get("source_manifest_sha256") != source["source_sha256"]
        or payload.get("h5ad_sha256") != source["payload"]["h5ad"]["sha256"]
        or payload.get("runner_sha256") != _sha256(Path(__file__))
        or payload.get("combat_data_and_comparator_utility_sha256")
        != _sha256(
            ROOT / DEVELOPMENT_BINDING_PATHS["combat_data_and_comparator_utility"]
        )
        or payload.get("markers") != list(MARKERS)
        or payload.get("calibration_samples")
        != list(_development_samples(source, "calibration"))
        or payload.get("pilot_samples") != list(_development_samples(source, "pilot"))
    ):
        raise PermissionError("development result differs from the freeze")
    attempt = payload.get("development_attempt")
    if (
        not isinstance(attempt, dict)
        or attempt.get("path") != _relative(DEFAULT_DEVELOPMENT_ATTEMPT)
        or not DEFAULT_DEVELOPMENT_ATTEMPT.is_file()
        or attempt.get("sha256") != _sha256(DEFAULT_DEVELOPMENT_ATTEMPT)
    ):
        raise PermissionError("development attempt binding differs")
    expected_models = set(METHODS)
    models = payload.get("frozen_source_models")
    if payload.get("status") == "PILOT_PASS":
        if not isinstance(models, dict) or set(models) != expected_models:
            raise PermissionError("passing development result lacks frozen models")
    elif payload.get("status") == "PILOT_FAIL":
        if models is not None:
            raise PermissionError("failed development result contains frozen models")
    else:
        raise PermissionError("development status is invalid")
    if require_pass and payload.get("status") != "PILOT_PASS":
        raise PermissionError("pilot did not authorize held access")
    return payload


def _validated_margin_authorization(
    authorization_path: Path,
    source_path: Path,
    development_path: Path,
    authorization_commit: str,
) -> dict[str, Any]:
    authorization = _read_json(authorization_path)
    development = _validated_development(
        development_path, source_path, require_pass=True
    )
    expected = {
        "schema",
        "status",
        "created_at_utc",
        "public_development_commit",
        "development_path",
        "development_sha256",
        "runner_sha256",
        "source_manifest_sha256",
    }
    if (
        set(authorization) != expected
        or authorization.get("schema")
        != "stephenson-citeseq-held-rna-authorization/1.0"
        or authorization.get("status") != "OUTCOME_ACCESS_AUTHORIZED"
        or authorization.get("development_path") != _relative(development_path)
        or authorization.get("development_sha256") != _sha256(development_path)
        or authorization.get("runner_sha256") != _sha256(Path(__file__))
        or authorization.get("source_manifest_sha256") != _sha256(source_path)
    ):
        raise PermissionError("held RNA-margin authorization differs")
    commit = authorization.get("public_development_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise PermissionError("held margin authorization commit is invalid")
    if _immutable_public_bytes(
        _relative(development_path), commit, "development result"
    ) != development_path.read_bytes():
        raise PermissionError("public development result bytes differ")
    if _immutable_public_bytes(
        _relative(authorization_path), authorization_commit, "margin authorization"
    ) != authorization_path.read_bytes():
        raise PermissionError("public margin authorization bytes differ")
    if development.get("status") != "PILOT_PASS":
        raise PermissionError("held margins are disabled after pilot failure")
    return {
        "authorization_sha256": _sha256(authorization_path),
        "public_authorization_commit": authorization_commit,
        "public_development_commit": commit,
    }


def _held_records(source: dict[str, Any]) -> list[dict[str, Any]]:
    records = [record for record in source["records"] if record["role"] == "held_site"]
    records.sort(key=lambda record: record["donor"])
    if len(records) != EXPECTED_HELD:
        raise PermissionError("held donor panel differs from the freeze")
    return records


def _held_rna_worker(
    connection: Any,
    h5ad: str,
    records: list[dict[str, Any]],
    authorization_path: str,
    authorization_sha256: str,
) -> None:
    try:
        permit = Path(authorization_path)
        if not permit.is_file() or _sha256(permit) != authorization_sha256:
            raise PermissionError("held RNA worker permit differs")
        path = Path(h5ad)
        selections = _selected_rows(path, records)
        samples = tuple(record["sample"] for record in records)
        counts = _read_modality(path, selections, samples, "rna")
        rows = []
        for record in records:
            sample = record["sample"]
            states = (numerics._integer_counts(counts[sample], "RNA") > 0).astype(
                np.uint8
            )
            margins = np.stack(
                (
                    CELL_BUDGET - states.sum(axis=1),
                    states.sum(axis=1),
                ),
                axis=1,
            ).astype(np.int64)
            rows.append(
                {
                    "donor": record["donor"],
                    "sample": sample,
                    "rna_margins": margins.tolist(),
                    "rna_margin_sha256": _array_sha256(margins),
                    "selected_barcode_sha256": selections[sample][
                        "selected_barcode_sha256"
                    ],
                    "eligible_pool_cells": selections[sample]["eligible_pool_cells"],
                }
            )
        connection.send({"status": "PASS", "rows": rows})
    except Exception as error:  # pragma: no cover - exercised through parent
        connection.send(
            {
                "status": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
    finally:
        connection.close()


def _extract_held_rna_margins(
    source: dict[str, Any],
    authorization_path: Path,
    authorization_sha256: str,
) -> list[dict[str, Any]]:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_held_rna_worker,
        args=(
            child,
            str(source["h5ad"]),
            _held_records(source),
            str(authorization_path),
            authorization_sha256,
        ),
    )
    process.start()
    child.close()
    result = parent.recv()
    parent.close()
    process.join()
    if process.exitcode != 0 or result.get("status") != "PASS":
        raise RuntimeError(
            f"held RNA worker refused: {result.get('error_type')}: {result.get('error')}"
        )
    rows = result.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_HELD:
        raise PermissionError("held RNA worker returned an invalid panel")
    return rows


def predict_held(
    source_path: Path,
    development_path: Path,
    authorization_path: Path,
    authorization_commit: str,
    attempt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Read only held RNA margins and freeze all method predictions."""

    if attempt_path.exists() or output_path.exists():
        raise FileExistsError("held prediction is one-shot")
    permit = _validated_margin_authorization(
        authorization_path, source_path, development_path, authorization_commit
    )
    source = _validated_source(source_path, verify_hash=True)
    development = _validated_development(
        development_path, source_path, require_pass=True
    )
    attempt = {
        "schema": "stephenson-citeseq-held-prediction-attempt/1.0",
        "status": "TERMINAL_ATTEMPT_STARTED",
        "created_at_utc": _timestamp(),
        "source_manifest_sha256": source["source_sha256"],
        "development_sha256": _sha256(development_path),
        "margin_authorization_sha256": permit["authorization_sha256"],
        "public_margin_authorization_commit": authorization_commit,
        "runner_sha256": _sha256(Path(__file__)),
        "held_margin_request_begins_after_this_record": True,
        "held_adt_numeric_access_authorized": False,
        "rerun_permitted": False,
    }
    _write_json(attempt_path, attempt)
    margin_rows = _extract_held_rna_margins(
        source, authorization_path, permit["authorization_sha256"]
    )
    models = development["frozen_source_models"]
    fixed_adt = np.tile(
        np.asarray([[CELL_BUDGET // 2, CELL_BUDGET // 2]], dtype=np.int64),
        (len(MARKERS), 1),
    )
    samples = []
    for row in margin_rows:
        rna = np.asarray(row["rna_margins"], dtype=np.int64)
        if rna.shape != (len(MARKERS), 2) or np.any(rna.sum(axis=1) != CELL_BUDGET):
            raise PermissionError("held RNA margins are malformed")
        predictions = {}
        boundaries = {}
        for name in METHODS:
            flags: list[dict[str, Any]] = []
            prediction = _predict_method(
                models[name], rna, fixed_adt, boundary_flags=flags
            )
            predictions[name] = prediction.reshape(len(MARKERS) ** 2, 4).tolist()
            boundaries[name] = flags
        samples.append(
            {
                **row,
                "adt_margins": fixed_adt.tolist(),
                "predictions": predictions,
                "boundary_tilts": boundaries,
            }
        )
    payload = {
        "schema": "stephenson-citeseq-held-predictions/1.0",
        "status": "FROZEN_HELD_PREDICTIONS",
        "created_at_utc": _timestamp(),
        "source_manifest_sha256": source["source_sha256"],
        "h5ad_sha256": source["payload"]["h5ad"]["sha256"],
        "development_sha256": _sha256(development_path),
        "runner_sha256": _sha256(Path(__file__)),
        "prediction_attempt": {
            "path": _relative(attempt_path),
            "sha256": _sha256(attempt_path),
        },
        "held_rna_margin_authorization": permit,
        "markers": list(MARKERS),
        "cells_per_donor": CELL_BUDGET,
        "samples": samples,
        "access_audit": {
            "held_rna_samples_read": EXPECTED_HELD,
            "held_adt_numeric_values_read": 0,
            "held_rna_adt_pairings_formed": 0,
            "held_truth_tables_formed": 0,
            "worker_output": "aggregate 9x2 RNA margins and digests only",
            "cell_vectors_serialized": False,
            "adt_margins": "fixed 256/256 by frozen midrank rule",
        },
    }
    _write_json(output_path, payload)
    return payload


def _validated_prediction(
    path: Path, source_path: Path, development_path: Path
) -> dict[str, Any]:
    payload = _read_json(path)
    source = _validated_source(source_path, verify_hash=False)
    development = _validated_development(
        development_path, source_path, require_pass=True
    )
    if (
        payload.get("schema") != "stephenson-citeseq-held-predictions/1.0"
        or payload.get("status") != "FROZEN_HELD_PREDICTIONS"
        or payload.get("source_manifest_sha256") != source["source_sha256"]
        or payload.get("h5ad_sha256") != source["payload"]["h5ad"]["sha256"]
        or payload.get("development_sha256") != _sha256(development_path)
        or payload.get("runner_sha256") != _sha256(Path(__file__))
        or payload.get("markers") != list(MARKERS)
        or payload.get("cells_per_donor") != CELL_BUDGET
    ):
        raise PermissionError("held prediction artifact differs from the freeze")
    rows = payload.get("samples")
    held = _held_records(source)
    if not isinstance(rows, list) or [row.get("sample") for row in rows] != [
        record["sample"] for record in held
    ]:
        raise PermissionError("held prediction donor order differs")
    fixed_adt = np.tile(
        np.asarray([[CELL_BUDGET // 2, CELL_BUDGET // 2]], dtype=np.int64),
        (len(MARKERS), 1),
    )
    models = development["frozen_source_models"]
    for row, record in zip(rows, held):
        if row.get("donor") != record["donor"]:
            raise PermissionError("held prediction donor differs")
        rna = np.asarray(row.get("rna_margins"), dtype=np.int64)
        adt = np.asarray(row.get("adt_margins"), dtype=np.int64)
        if (
            rna.shape != (len(MARKERS), 2)
            or np.any(rna.sum(axis=1) != CELL_BUDGET)
            or not np.array_equal(adt, fixed_adt)
            or row.get("rna_margin_sha256") != _array_sha256(rna)
            or set(row.get("predictions", {})) != set(METHODS)
            or set(row.get("boundary_tilts", {})) != set(METHODS)
        ):
            raise PermissionError("held prediction row differs")
        for name in METHODS:
            flags: list[dict[str, Any]] = []
            expected = _predict_method(
                models[name], rna, adt, boundary_flags=flags
            ).reshape(len(MARKERS) ** 2, 4)
            observed = np.asarray(row["predictions"][name], dtype=float)
            if observed.shape != expected.shape or not np.array_equal(observed, expected):
                raise PermissionError(f"held prediction changed: {name}")
            if row["boundary_tilts"][name] != flags:
                raise PermissionError(f"held boundary flags changed: {name}")
    return payload


def _validated_score_authorization(
    authorization_path: Path,
    prediction_path: Path,
    source_path: Path,
    development_path: Path,
    authorization_commit: str,
) -> dict[str, Any]:
    authorization = _read_json(authorization_path)
    _validated_prediction(prediction_path, source_path, development_path)
    expected_values = {
        "prediction_path": _relative(prediction_path),
        "prediction_sha256": _sha256(prediction_path),
        "prediction_bytes": prediction_path.stat().st_size,
        "runner_sha256": _sha256(Path(__file__)),
        "source_manifest_sha256": _sha256(source_path),
        "development_sha256": _sha256(development_path),
    }
    if (
        authorization.get("schema")
        != "stephenson-citeseq-score-authorization/1.0"
        or authorization.get("status") != "OUTCOME_ACCESS_AUTHORIZED"
        or set(authorization)
        != {
            "schema",
            "status",
            "created_at_utc",
            *expected_values,
            "public_prediction_commit",
            "public_prediction_url",
        }
    ):
        raise PermissionError("score authorization fields differ")
    for key, value in expected_values.items():
        if authorization.get(key) != value:
            raise PermissionError(f"score authorization differs: {key}")
    commit = authorization.get("public_prediction_commit")
    url = authorization.get("public_prediction_url")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise PermissionError("public prediction commit is invalid")
    if not isinstance(url, str):
        raise PermissionError("public prediction URL is absent")
    parsed = urlparse(url)
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    expected_tail = [
        "blob",
        commit,
        *expected_values["prediction_path"].split("/"),
    ]
    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or parsed.query
        or parsed.fragment
        or parts[:2] != [PUBLIC_OWNER, PUBLIC_REPOSITORY]
        or parts[2:] != expected_tail
    ):
        raise PermissionError("public prediction URL is not the pinned repository blob")
    if _immutable_public_bytes(
        expected_values["prediction_path"], commit, "held prediction"
    ) != prediction_path.read_bytes():
        raise PermissionError("public held prediction bytes differ")
    if _immutable_public_bytes(
        _relative(authorization_path), authorization_commit, "score authorization"
    ) != authorization_path.read_bytes():
        raise PermissionError("public score authorization bytes differ")
    return {
        "authorization_sha256": _sha256(authorization_path),
        "public_authorization_commit": authorization_commit,
        "public_prediction_commit": commit,
    }


def _exact_sign_test(values: np.ndarray) -> dict[str, Any]:
    difference = np.asarray(values, dtype=float)
    if difference.shape != (EXPECTED_HELD,) or not np.isfinite(difference).all():
        raise ValueError("sign test requires 56 finite donor differences")
    favorable = int(np.count_nonzero(difference < 0.0))
    numerator = sum(
        math.comb(EXPECTED_HELD, count)
        for count in range(favorable, EXPECTED_HELD + 1)
    )
    return {
        "one_sided_p": float(numerator / (1 << EXPECTED_HELD)),
        "favorable_donors": favorable,
        "donors": EXPECTED_HELD,
        "zeros_are_nonfavorable": True,
        "null": "independent favorable probability equals one half",
    }


def score_held(
    source_path: Path,
    development_path: Path,
    prediction_path: Path,
    authorization_path: Path,
    authorization_commit: str,
    attempt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Read held RNA--ADT pairings once after public prediction authorization."""

    if attempt_path.exists() or output_path.exists():
        raise FileExistsError("held scoring is one-shot")
    permit = _validated_score_authorization(
        authorization_path,
        prediction_path,
        source_path,
        development_path,
        authorization_commit,
    )
    source = _validated_source(source_path, verify_hash=True)
    prediction = _validated_prediction(
        prediction_path, source_path, development_path
    )
    attempt = {
        "schema": "stephenson-citeseq-score-attempt/1.0",
        "status": "TERMINAL_ATTEMPT_STARTED",
        "created_at_utc": _timestamp(),
        "source_manifest_sha256": source["source_sha256"],
        "prediction_sha256": _sha256(prediction_path),
        "score_authorization_sha256": permit["authorization_sha256"],
        "runner_sha256": _sha256(Path(__file__)),
        "held_pairing_request_begins_after_this_record": True,
        "rerun_permitted": False,
    }
    _write_json(attempt_path, attempt)
    held = _held_records(source)
    selections = _selected_rows(source["h5ad"], held)
    samples = tuple(record["sample"] for record in held)
    rna_counts = _read_modality(source["h5ad"], selections, samples, "rna")
    adt_counts = _read_modality(source["h5ad"], selections, samples, "adt")
    rna_states = {
        sample: (numerics._integer_counts(rna_counts[sample], "RNA") > 0).astype(
            np.uint8
        )
        for sample in samples
    }
    by_sample = {record["sample"]: record for record in held}
    adt_states = {
        sample: _adt_states(
            adt_counts[sample],
            selections[sample]["barcodes"],
            by_sample[sample]["donor"],
            sample,
        )
        for sample in samples
    }
    truth = numerics._form_tables(rna_states, adt_states, list(samples))
    prediction_rows = {row["sample"]: row for row in prediction["samples"]}
    losses = {name: np.empty(EXPECTED_HELD, dtype=float) for name in METHODS}
    donor_rows = []
    for index, (record, observed) in enumerate(zip(held, truth)):
        sample = record["sample"]
        support = numerics._informative(observed).reshape(-1)
        if int(support.sum()) < numerics.MINIMUM_INFORMATIVE_ENTITIES:
            raise ValueError(f"held donor has fewer than 64 informative pairs: {sample}")
        row_losses = {}
        for name in METHODS:
            fitted = np.asarray(
                prediction_rows[sample]["predictions"][name], dtype=float
            ).reshape(len(MARKERS), len(MARKERS), 2, 2)
            value = numerics._donor_loss(observed, fitted, support)
            losses[name][index] = value
            row_losses[name] = float(value)
        donor_rows.append(
            {
                "donor": record["donor"],
                "sample": sample,
                "informative_pairs": int(support.sum()),
                "losses": row_losses,
            }
        )
    comparisons = {}
    for name in METHODS[1:]:
        try:
            row = numerics._comparison(
                samples,
                losses["primary"],
                losses[name],
                favorable_required=(
                    HELD_FAVORABLE
                    if name in PROMOTION_COMPARATORS
                    else None
                ),
            )
        except ValueError as error:
            comparisons[name] = {
                "status": "NUMERICAL_FAILURE",
                "reason": str(error),
                "passes": False if name in PROMOTION_COMPARATORS else None,
            }
            continue
        if name in PROMOTION_COMPARATORS:
            sign_test = _exact_sign_test(losses["primary"] - losses[name])
            row["sign_test"] = sign_test
            row["passes"] = bool(row["passes"] and sign_test["one_sided_p"] <= 0.025)
        comparisons[name] = row
    field_transfer = all(
        comparisons[name].get("passes") is True
        for name in PROMOTION_COMPARATORS
    )
    graph_zero = comparisons.get("hierarchical_graph_zero", {})
    graph_specific = bool(
        isinstance(graph_zero.get("paired_difference_95_ci"), list)
        and graph_zero["paired_difference_95_ci"][1] < 0.0
    )
    full_pass = bool(field_transfer)
    payload = {
        "schema": "stephenson-citeseq-held-confirmation/1.0",
        "status": "CONFIRMATION_PASS" if full_pass else "CONFIRMATION_FAIL",
        "created_at_utc": _timestamp(),
        "source_manifest_sha256": source["source_sha256"],
        "development_sha256": _sha256(development_path),
        "prediction_sha256": _sha256(prediction_path),
        "score_authorization": permit,
        "score_attempt": {
            "path": _relative(attempt_path),
            "sha256": _sha256(attempt_path),
        },
        "runner_sha256": _sha256(Path(__file__)),
        "panel": "Newcastle held site",
        "donors": EXPECTED_HELD,
        "comparisons": comparisons,
        "passes_field_transfer": field_transfer,
        "supports_graph_specific_superiority": graph_specific,
        "passes_primary_method": full_pass,
        "donor_results": donor_rows,
        "access_audit": {
            "held_donors_scored": EXPECTED_HELD,
            "paired_truth_access_after_public_prediction": True,
            "terminal_attempt_preceded_first_truth_request": True,
            "cell_vectors_serialized": False,
            "rerun_permitted": False,
        },
    }
    _write_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    seal = subparsers.add_parser("seal-source")
    seal.add_argument("--h5ad", type=Path, required=True)
    seal.add_argument("--sdrf", type=Path, default=DEFAULT_SDRF)
    seal.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    seal.add_argument("--source", type=Path, default=DEFAULT_SOURCE)

    verify = subparsers.add_parser("verify-freeze")
    verify.add_argument("--commit", required=True)
    verify.add_argument("--output", type=Path, default=DEFAULT_VERIFICATION)

    development = subparsers.add_parser("run-development")
    development.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    development.add_argument(
        "--authorization", type=Path, default=DEFAULT_DEVELOPMENT_AUTHORIZATION
    )
    development.add_argument("--authorization-commit", required=True)
    development.add_argument("--attempt", type=Path, default=DEFAULT_DEVELOPMENT_ATTEMPT)
    development.add_argument("--output", type=Path, default=DEFAULT_DEVELOPMENT)

    predict = subparsers.add_parser("predict-held")
    predict.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    predict.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    predict.add_argument("--authorization", type=Path, default=DEFAULT_MARGIN_AUTHORIZATION)
    predict.add_argument("--authorization-commit", required=True)
    predict.add_argument("--attempt", type=Path, default=DEFAULT_PREDICTION_ATTEMPT)
    predict.add_argument("--output", type=Path, default=DEFAULT_PREDICTION)

    score = subparsers.add_parser("score-held")
    score.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    score.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    score.add_argument("--prediction", type=Path, default=DEFAULT_PREDICTION)
    score.add_argument("--authorization", type=Path, default=DEFAULT_SCORE_AUTHORIZATION)
    score.add_argument("--authorization-commit", required=True)
    score.add_argument("--attempt", type=Path, default=DEFAULT_SCORE_ATTEMPT)
    score.add_argument("--output", type=Path, default=DEFAULT_SCORE)

    args = parser.parse_args()
    if args.phase == "seal-source":
        seal_source(args.h5ad, args.sdrf, args.preflight, args.source)
    elif args.phase == "verify-freeze":
        verify_public_freeze(args.commit, args.output)
    elif args.phase == "run-development":
        run_development(
            args.source,
            args.authorization,
            args.authorization_commit,
            args.attempt,
            args.output,
        )
    elif args.phase == "predict-held":
        predict_held(
            args.source,
            args.development,
            args.authorization,
            args.authorization_commit,
            args.attempt,
            args.output,
        )
    else:
        score_held(
            args.source,
            args.development,
            args.prediction,
            args.authorization,
            args.authorization_commit,
            args.attempt,
            args.output,
        )


if __name__ == "__main__":
    main()
