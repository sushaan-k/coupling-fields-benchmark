"""Frozen held-batch confirmation on the GSE158769 linked RNA/ADT cohort.

The commands enforce four public barriers. ``preflight`` reads metadata only;
``acquire`` downloads the opaque gzip after the protocol tag; ``develop``
tokenizes calibration and pilot values only; ``predict`` tokenizes held RNA
values only; and ``score`` is the first command to tokenize held ADT values.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Iterator
import urllib.request

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import confirm_gse314416_citeseq as core  # noqa: E402
from mapreg.heterogeneity_adaptive_coupling import CouplingEstimationRefusal  # noqa: E402


ROOT = REPO_ROOT
DATA_DIR = ROOT / "data/confirmation/gse158769_citeseq"
SOURCE_CACHE = DATA_DIR / "source_cache"
DEFAULT_METADATA = SOURCE_CACHE / "GSE158769_meta_data.txt.gz"
DEFAULT_RAW = SOURCE_CACHE / "GSE158769_exprs_raw.tsv.gz"
DEFAULT_MANIFEST = DATA_DIR / "source_manifest_v1.json"
DEFAULT_DESIGNATION = DATA_DIR / "candidate_designation_v1.json"
DEFAULT_METADATA_ACCESS = (
    ROOT / "results/development/gse158769_metadata_access_record_v1.json"
)
DEFAULT_PREFLIGHT = ROOT / "results/development/gse158769_metadata_preflight_v1.json"
DEFAULT_SOURCE_ACCESS = ROOT / "results/development/gse158769_source_access_v1.json"
DEFAULT_DEVELOPMENT = ROOT / "results/development/gse158769_development_v1.json"
DEFAULT_PREDICTION = ROOT / "results/gse158769_held_predictions_v1.json"
DEFAULT_SCORE = ROOT / "results/gse158769_confirmation_v1.json"
DEFAULT_PROTOCOL = (
    ROOT / "docs/GSE158769_HELD_BATCH_CITESEQ_CONFIRMATION_PROTOCOL_2026-08-28.md"
)

PUBLIC_ORIGIN = "https://github.com/sushaan-k/coupling-fields-benchmark.git"
PROTOCOL_TAG = "gse158769-citeseq-v1-protocol"
SOURCE_TAG = "gse158769-citeseq-v1-source"
DEVELOPMENT_TAG = "gse158769-citeseq-v1-development"
PREDICTION_TAG = "gse158769-citeseq-v1-predictions"

MARKERS = ("CD3", "CD4", "CD5", "CD8", "CD161", "CD127", "CD27", "CD38", "CD26")
RNA_FEATURES = ("CD3E", "CD4", "CD5", "CD8A", "KLRB1", "IL7R", "CD27", "CD38", "DPP4")
ADT_ALIASES = (
    ("CD3", "CD3.1", "CD3 (CD3E)"),
    ("CD4", "CD4.1"),
    ("CD5", "CD5.1"),
    ("CD8a", "CD8a/CD8A"),
    ("CD161", "CD161 (KLRB1)"),
    ("CD127", "CD127/IL-7R", "CD127 (IL7R)"),
    ("CD27", "CD27.1"),
    ("CD38", "CD38.1"),
    ("CD26", "CD26 (DPP4)"),
)

CELL_BUDGET = 512
MINIMUM_INFORMATIVE_ENTITIES = 64
CELL_SELECTION_SALT = "GSE158769-CELL-BUDGET-v1"
ADT_TIE_SALT = "GSE158769-ADT-v1"
DESTROYED_LINK_SALT = "GSE158769-DESTROYED-LINK-v1"

CALIBRATION_BATCHES = (15, 42, 27, 36, 7, 2, 28, 10, 30, 40, 45, 12, 6, 37, 13, 32)
PILOT_BATCHES = (33, 41, 34, 38, 11, 29, 24, 18, 22, 9, 14, 17, 35, 4)
HELD_BATCHES = (21, 43, 25, 20, 1, 46, 16, 44, 39, 26, 19, 3, 8, 5, 23, 31)
REPEATED_DONORS = (
    "TB0763419",
    "TB0771071",
    "TB1285760",
    "TB2635526",
    "TB5063423",
    "TB5758296",
    "TB5867110",
    "TB6361266",
    "TB6448199",
    "TB6692076",
    "TB7289019",
    "TB9312329",
)
UNDER_BUDGET_DONORS = (
    "TB0414005",
    "TB0439275",
    "TB1535708",
    "TB2560544",
    "TB3350734",
    "TB4817029",
    "TB5484224",
    "TB5742418",
    "TB6257284",
    "TB6994841",
    "TB7640450",
    "TB8797251",
    "TB8805781",
)
EXPECTED_ROLE_COUNTS = {"calibration": 85, "pilot": 69, "held": 80}
EXPECTED_AVAILABLE_CELLS = {"calibration": 173764, "pilot": 130420, "held": 148228}
EXPECTED_CASE_CONTROL = {"calibration": (42, 43), "pilot": (31, 38), "held": (39, 41)}

NEIGHBOR_GRID = (1, 2)
HETEROGENEITY_GRID = (0.1, 1.0, 10.0)
RIDGE_GRID = (0.01, 0.1)
GRAPH_GRID = (0.0, 0.1, 1.0)
TRANSPORT_GRID = (0.5, 0.75, 1.0, 1.25)
RESIDUAL_FAMILIES = ("pearson", "deviance")
BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 20260828
PILOT_DONOR_REQUIRED = 56
PILOT_BATCH_REQUIRED = 12
HELD_DONOR_REQUIRED = 64
HELD_BATCH_REQUIRED = 13

PROTOCOL_BINDINGS = (
    "experiments/confirm_gse158769_citeseq.py",
    "tests/test_gse158769_citeseq_confirmation.py",
    "docs/GSE158769_HELD_BATCH_CITESEQ_CONFIRMATION_PROTOCOL_2026-08-28.md",
    "data/confirmation/gse158769_citeseq/source_manifest_v1.json",
    "data/confirmation/gse158769_citeseq/candidate_designation_v1.json",
    "results/development/gse158769_metadata_access_record_v1.json",
    "results/development/gse158769_metadata_preflight_v1.json",
    "experiments/confirm_gse314416_citeseq.py",
    "mapreg/heterogeneity_adaptive_coupling.py",
    "mapreg/hierarchical_conditional_coupling.py",
)


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


def _salted_hash(salt: str, *values: str) -> str:
    digest = hashlib.sha256(salt.encode())
    for value in values:
        digest.update(b"\0")
        digest.update(value.encode())
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one object")
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
        raise PermissionError(f"public tag {tag} does not resolve to the local commit")
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


def _validate_metadata_file(path: Path) -> None:
    record = _read_json(DEFAULT_MANIFEST)["files"]["metadata"]
    if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
        raise PermissionError("metadata bytes differ from the frozen source")


def _metadata_inventory(path: Path) -> dict[str, Any]:
    _validate_metadata_file(path)
    cells: dict[str, list[str]] = {}
    batches: dict[str, set[int]] = {}
    statuses: dict[str, set[str]] = {}
    seen_cells: set[str] = set()
    with gzip.open(path, "rt", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"cell_id", "batch", "TB_status", "donor"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError("metadata schema differs from the frozen fields")
        for row in reader:
            cell = row["cell_id"]
            donor = row["donor"]
            if cell in seen_cells:
                raise ValueError("metadata cell IDs are not globally unique")
            seen_cells.add(cell)
            cells.setdefault(donor, []).append(cell)
            batches.setdefault(donor, set()).add(int(row["batch"]))
            statuses.setdefault(donor, set()).add(row["TB_status"])
    if len(seen_cells) != 500089 or len(cells) != 259:
        raise PermissionError("metadata cohort size differs from the designation")
    if {donor for donor, values in batches.items() if len(values) > 1} != set(
        REPEATED_DONORS
    ):
        raise PermissionError("repeated-donor set differs from the designation")
    observed_under = {
        donor
        for donor, values in cells.items()
        if donor not in REPEATED_DONORS and len(values) < CELL_BUDGET
    }
    if observed_under != set(UNDER_BUDGET_DONORS):
        raise PermissionError("under-budget donor set differs from the designation")
    role_batches = {
        "calibration": set(CALIBRATION_BATCHES),
        "pilot": set(PILOT_BATCHES),
        "held": set(HELD_BATCHES),
    }
    if set.union(*role_batches.values()) != set(range(1, 47)) or any(
        role_batches[left] & role_batches[right]
        for left, right in (
            ("calibration", "pilot"),
            ("calibration", "held"),
            ("pilot", "held"),
        )
    ):
        raise AssertionError("processing-batch allocation is not a partition")
    roles: dict[str, str] = {}
    donor_batch: dict[str, int] = {}
    donor_status: dict[str, str] = {}
    for donor in cells:
        if donor in REPEATED_DONORS:
            roles[donor] = "excluded_repeated"
            continue
        if donor in UNDER_BUDGET_DONORS:
            roles[donor] = "excluded_under_budget"
            continue
        if len(batches[donor]) != 1 or len(statuses[donor]) != 1:
            raise PermissionError("eligible donor has inconsistent metadata")
        batch = next(iter(batches[donor]))
        role = next(name for name, values in role_batches.items() if batch in values)
        roles[donor] = role
        donor_batch[donor] = batch
        donor_status[donor] = next(iter(statuses[donor]))
    for role, expected in EXPECTED_ROLE_COUNTS.items():
        donors = [donor for donor, value in roles.items() if value == role]
        if (
            len(donors) != expected
            or sum(len(cells[donor]) for donor in donors)
            != EXPECTED_AVAILABLE_CELLS[role]
        ):
            raise PermissionError(f"{role} donor or cell count differs")
        case = sum(donor_status[donor] == "CASE" for donor in donors)
        control = sum(donor_status[donor] == "CONTROL" for donor in donors)
        if (case, control) != EXPECTED_CASE_CONTROL[role]:
            raise PermissionError(f"{role} case/control count differs")
    selected = {
        donor: sorted(
            cells[donor],
            key=lambda cell: (_salted_hash(CELL_SELECTION_SALT, donor, cell), cell),
        )[:CELL_BUDGET]
        for donor, role in roles.items()
        if role in EXPECTED_ROLE_COUNTS
    }
    return {
        "cells": cells,
        "batches": batches,
        "statuses": statuses,
        "roles": roles,
        "donor_batch": donor_batch,
        "donor_status": donor_status,
        "selected": selected,
    }


def run_preflight(metadata_path: Path, output_path: Path) -> dict[str, Any]:
    inventory = _metadata_inventory(metadata_path)
    donors = []
    for donor in sorted(inventory["roles"]):
        role = inventory["roles"][donor]
        row: dict[str, Any] = {
            "donor": donor,
            "role": role,
            "metadata_cells": len(inventory["cells"][donor]),
            "batches": sorted(inventory["batches"][donor]),
            "statuses": sorted(inventory["statuses"][donor]),
        }
        if donor in inventory["selected"]:
            row["selected_cell_axis_sha256"] = _axis_sha256(
                inventory["selected"][donor]
            )
        donors.append(row)
    payload = {
        "schema": "gse158769-metadata-preflight/1.0",
        "status": "PASS",
        "created_at_utc": _timestamp(),
        "accession": "GSE158769",
        "access_audit": {
            "raw_matrix_downloaded": False,
            "raw_matrix_decompressed": False,
            "rna_numeric_values_read": 0,
            "adt_numeric_values_read": 0,
        },
        "bindings": {
            "metadata_sha256": _sha256(metadata_path),
            "source_manifest_sha256": _sha256(DEFAULT_MANIFEST),
            "designation_sha256": _sha256(DEFAULT_DESIGNATION),
        },
        "role_counts": EXPECTED_ROLE_COUNTS,
        "role_batches": {
            "calibration": list(CALIBRATION_BATCHES),
            "pilot": list(PILOT_BATCHES),
            "held": list(HELD_BATCHES),
        },
        "markers": [
            {"marker": marker, "rna_feature": rna, "adt_aliases": list(adt)}
            for marker, rna, adt in zip(MARKERS, RNA_FEATURES, ADT_ALIASES)
        ],
        "donors": donors,
    }
    _write_json(output_path, payload)
    return payload


def _validated_inventory(metadata_path: Path) -> dict[str, Any]:
    frozen = _read_json(DEFAULT_PREFLIGHT)
    if (
        frozen.get("status") != "PASS"
        or frozen.get("access_audit", {}).get("adt_numeric_values_read") != 0
    ):
        raise PermissionError("metadata preflight is not an outcome-blind pass")
    if frozen.get("bindings", {}).get("source_manifest_sha256") != _sha256(
        DEFAULT_MANIFEST
    ):
        raise PermissionError("metadata preflight source binding differs")
    inventory = _metadata_inventory(metadata_path)
    by_donor = {row["donor"]: row for row in frozen["donors"]}
    for donor, role in inventory["roles"].items():
        if by_donor[donor]["role"] != role:
            raise PermissionError(f"role differs for {donor}")
        if donor in inventory["selected"] and by_donor[donor][
            "selected_cell_axis_sha256"
        ] != _axis_sha256(inventory["selected"][donor]):
            raise PermissionError(f"selected-cell axis differs for {donor}")
    return inventory


def acquire_raw(output_path: Path, record_path: Path) -> dict[str, Any]:
    protocol_commit = _require_public_tag(PROTOCOL_TAG, PROTOCOL_BINDINGS)
    source = _read_json(DEFAULT_MANIFEST)["files"]["raw_linked_matrix"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    partial = output_path.with_suffix(output_path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"remove or inspect incomplete download {partial}")
    digest = hashlib.sha256()
    size = 0
    with (
        urllib.request.urlopen(source["url"]) as response,
        partial.open("xb") as stream,
    ):
        while True:
            block = response.read(8 << 20)
            if not block:
                break
            stream.write(block)
            digest.update(block)
            size += len(block)
    if size != source["bytes"]:
        raise IOError(f"raw download has {size} bytes, expected {source['bytes']}")
    partial.replace(output_path)
    payload = {
        "schema": "gse158769-source-access/1.0",
        "status": "OPAQUE_GZIP_DOWNLOADED_NOT_DECOMPRESSED",
        "created_at_utc": _timestamp(),
        "protocol_tag": PROTOCOL_TAG,
        "protocol_commit": protocol_commit,
        "path": _relative(output_path),
        "bytes": size,
        "sha256": digest.hexdigest(),
        "decompression_performed": False,
        "numeric_values_read": 0,
    }
    _write_json(record_path, payload)
    return payload


def _validated_raw(path: Path) -> dict[str, Any]:
    _require_public_tag(
        SOURCE_TAG, (*PROTOCOL_BINDINGS, _relative(DEFAULT_SOURCE_ACCESS))
    )
    record = _read_json(DEFAULT_SOURCE_ACCESS)
    if record.get("status") != "OPAQUE_GZIP_DOWNLOADED_NOT_DECOMPRESSED":
        raise PermissionError("raw source was not acquired under the frozen protocol")
    if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
        raise PermissionError("raw source bytes differ from the public source record")
    return record


def _iter_tsv_fields(line: bytes) -> Iterator[tuple[int, bytes]]:
    start = 0
    index = 0
    while True:
        end = line.find(b"\t", start)
        if end < 0:
            value = line[start:].rstrip(b"\r\n")
            yield index, value
            return
        yield index, line[start:end]
        index += 1
        start = end + 1


def _selected_fields(line: bytes, wanted: list[int]) -> list[bytes]:
    output: list[bytes] = []
    cursor = 0
    start = 0
    field = 0
    while cursor < len(wanted):
        end = line.find(b"\t", start)
        if end < 0:
            end = len(line.rstrip(b"\r\n"))
        if field == wanted[cursor]:
            output.append(line[start:end])
            cursor += 1
        if end >= len(line) or line[end : end + 1] != b"\t":
            break
        field += 1
        start = end + 1
    if cursor != len(wanted):
        raise ValueError("selected TSV column is missing")
    return output


def _stream_panel(
    raw_path: Path,
    donors: list[str],
    selected: dict[str, list[str]],
    modalities: frozenset[str],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    if not modalities or not modalities <= {"rna", "adt"}:
        raise ValueError("modalities must be a nonempty RNA/ADT subset")
    requested_cells = {
        cell: (donor, offset)
        for donor in donors
        for offset, cell in enumerate(selected[donor])
    }
    if len(requested_cells) != len(donors) * CELL_BUDGET:
        raise ValueError("selected cell IDs are not unique")
    counts = {
        donor: {
            modality: np.zeros((CELL_BUDGET, len(MARKERS)), dtype=np.int64)
            for modality in modalities
        }
        for donor in donors
    }
    candidates: dict[tuple[str, int], list[tuple[int, np.ndarray]]] = {}
    parsed_rows = {"rna": 0, "adt": 0}
    skipped_adt_rows = 0
    with gzip.open(raw_path, "rb") as stream:
        header = stream.readline()
        if not header:
            raise ValueError("raw matrix is empty")
        located: dict[str, int] = {}
        header_cells = 0
        for field, value in _iter_tsv_fields(header):
            if field == 0:
                continue
            header_cells += 1
            cell = value.decode()
            if cell in requested_cells:
                if cell in located:
                    raise ValueError("requested cell appears twice in raw header")
                located[cell] = field
        if set(located) != set(requested_cells):
            missing = sorted(set(requested_cells) - set(located))[:3]
            raise ValueError(f"selected cells are absent from raw header: {missing}")
        ordered_cells = sorted(located, key=located.get)
        wanted = [located[cell] for cell in ordered_cells]
        targets = [requested_cells[cell] for cell in ordered_cells]
        for row_index, line in enumerate(stream, start=1):
            separator = line.find(b"\t")
            if separator < 0:
                raise ValueError("raw feature row has no value columns")
            label = line[:separator].decode()
            matches: list[tuple[str, int]] = []
            if "rna" in modalities:
                matches.extend(
                    ("rna", index)
                    for index, feature in enumerate(RNA_FEATURES)
                    if label == feature
                )
            adt_matches = [
                ("adt", index)
                for index, aliases in enumerate(ADT_ALIASES)
                if label in aliases
            ]
            if "adt" in modalities:
                matches.extend(adt_matches)
            elif adt_matches:
                skipped_adt_rows += 1
            # In RNA-only mode the first exact gene occurrence is frozen as RNA;
            # later duplicate antibody labels are skipped without tokenization.
            matches = [
                key for key in matches if not (key[0] == "rna" and key in candidates)
            ]
            if not matches:
                continue
            values = np.fromiter(
                (int(value) for value in _selected_fields(line, wanted)),
                dtype=np.int64,
                count=len(wanted),
            )
            if np.any(values < 0):
                raise ValueError("raw matrix contains a negative selected count")
            for key in matches:
                candidates.setdefault(key, []).append((row_index, values.copy()))
                parsed_rows[key[0]] += 1
    for marker in range(len(MARKERS)):
        for modality in modalities:
            key = (modality, marker)
            values = candidates.get(key, [])
            if not values:
                raise ValueError(f"missing {modality} feature for {MARKERS[marker]}")
            # RNA is the first exact occurrence; ADT is the last accepted
            # occurrence. Coincident RNA/ADT row ordinals are forbidden.
            row_index, vector = values[0] if modality == "rna" else values[-1]
            if modalities == {"rna", "adt"}:
                rna_row = candidates[("rna", marker)][0][0]
                adt_row = candidates[("adt", marker)][-1][0]
                if rna_row == adt_row:
                    raise ValueError(
                        f"RNA and ADT resolve to one row for {MARKERS[marker]}"
                    )
            for value, (donor, offset) in zip(vector, targets):
                counts[donor][modality][offset, marker] = value
    audit = {
        "header_cell_columns": header_cells,
        "selected_cell_columns": len(wanted),
        "rna_feature_rows_tokenized": parsed_rows["rna"],
        "adt_feature_rows_tokenized": parsed_rows["adt"],
        "adt_candidate_rows_skipped_without_tokenization": skipped_adt_rows,
        "full_tsv_materialized": False,
    }
    return counts, audit


def _adt_states(counts: np.ndarray, cells: list[str], donor: str) -> np.ndarray:
    values = np.asarray(counts)
    if values.shape != (CELL_BUDGET, len(MARKERS)):
        raise ValueError("ADT count panel differs from the frozen design")
    states = np.zeros(values.shape, dtype=np.uint8)
    for marker_index, marker in enumerate(MARKERS):
        order = sorted(
            range(CELL_BUDGET),
            key=lambda index: (
                int(values[index, marker_index]),
                _salted_hash(ADT_TIE_SALT, donor, cells[index], marker),
                cells[index],
            ),
        )
        states[np.asarray(order[CELL_BUDGET // 2 :], dtype=int), marker_index] = 1
    if not np.all(states.sum(axis=0) == CELL_BUDGET // 2):
        raise AssertionError("ADT midrank state does not have an exact half margin")
    return states


def _destroyed_adt(states: np.ndarray, cells: list[str], donor: str) -> np.ndarray:
    order = sorted(
        range(CELL_BUDGET),
        key=lambda index: (
            _salted_hash(DESTROYED_LINK_SALT, donor, cells[index]),
            cells[index],
        ),
    )
    output = np.empty_like(states)
    for position, target in enumerate(order):
        output[target] = states[order[(position + 1) % CELL_BUDGET]]
    return output


def _adt_profiles(counts: np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    total = values.sum(axis=1, keepdims=True)
    normalized = np.divide(
        100.0 * values, total, out=np.zeros_like(values), where=total > 0.0
    )
    return np.log1p(normalized).mean(axis=0)


@contextmanager
def _configured_core() -> Iterator[None]:
    settings = {
        "MARKERS": MARKERS,
        "CELL_BUDGET": CELL_BUDGET,
        "MINIMUM_INFORMATIVE_ENTITIES": MINIMUM_INFORMATIVE_ENTITIES,
        "NEIGHBOR_GRID": NEIGHBOR_GRID,
        "HETEROGENEITY_GRID": HETEROGENEITY_GRID,
        "RIDGE_GRID": RIDGE_GRID,
        "GRAPH_GRID": GRAPH_GRID,
        "TRANSPORT_GRID": TRANSPORT_GRID,
        "RESIDUAL_FAMILIES": RESIDUAL_FAMILIES,
        "BOOTSTRAPS": BOOTSTRAPS,
        "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
    }
    previous = {name: getattr(core, name) for name in settings}
    try:
        for name, value in settings.items():
            setattr(core, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(core, name, value)


def _records_from_counts(
    donors: list[str],
    selected: dict[str, list[str]],
    counts: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with _configured_core():
        for donor in donors:
            rna = (counts[donor]["rna"] > 0).astype(np.uint8)
            adt = _adt_states(counts[donor]["adt"], selected[donor], donor)
            tables = core._binary_tables(rna, adt)
            destroyed = core._binary_tables(
                rna, _destroyed_adt(adt, selected[donor], donor)
            )
            records[donor] = {
                "tables": tables,
                "destroyed_tables": destroyed,
                "rna_profiles": rna.mean(axis=0),
                "adt_profiles": _adt_profiles(counts[donor]["adt"]),
                "table_sha256": _array_sha256(tables),
                "destroyed_table_sha256": _array_sha256(destroyed),
            }
    return records


def _primary_configs() -> list[core.PrimaryConfig]:
    return [
        core.PrimaryConfig(neighbor, heterogeneity, ridge, graph, transport)
        for neighbor, heterogeneity, ridge, graph, transport in product(
            NEIGHBOR_GRID, HETEROGENEITY_GRID, RIDGE_GRID, GRAPH_GRID, TRANSPORT_GRID
        )
    ]


def _select_on_pilot(
    records: dict[str, dict[str, Any]], calibration: list[str], pilot: list[str]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    with _configured_core():
        tables, _, rna_profiles, adt_profiles = core._records_arrays(
            records, calibration
        )
        primary_losses: dict[core.PrimaryConfig, np.ndarray] = {}
        structural: dict[
            tuple[int, float, float, float], dict[str, Any] | Exception
        ] = {}
        for config in _primary_configs():
            key = (
                1 if config.graph_penalty == 0.0 else config.graph_neighbors,
                config.heterogeneity_penalty,
                config.ridge_penalty,
                config.graph_penalty,
            )
            if key not in structural:
                try:
                    structural[key] = core._fit_primary(
                        tables, rna_profiles, adt_profiles, config
                    )
                except (
                    ValueError,
                    FloatingPointError,
                    CouplingEstimationRefusal,
                ) as error:
                    structural[key] = error
            fit = structural[key]
            losses = np.full(len(pilot), np.nan)
            if not isinstance(fit, Exception):
                for index, donor in enumerate(pilot):
                    truth = np.asarray(records[donor]["tables"])
                    rows, columns = core._margins(truth)
                    prediction = core._predict_log_odds(
                        fit["population_log_odds"],
                        rows,
                        columns,
                        config.transport_multiplier,
                    )
                    losses[index] = core._donor_loss(truth, prediction)
            primary_losses[config] = losses
        residual_losses: dict[core.ResidualConfig, np.ndarray] = {}
        residual_pools: dict[str, np.ndarray] = {}
        for family in RESIDUAL_FAMILIES:
            residual_pools[family] = core._residual_pool(tables, family)
            for transport in TRANSPORT_GRID:
                config = core.ResidualConfig(family, transport)
                losses = np.empty(len(pilot))
                for index, donor in enumerate(pilot):
                    truth = np.asarray(records[donor]["tables"])
                    rows, columns = core._margins(truth)
                    losses[index] = core._donor_loss(
                        truth,
                        core._predict_residual(
                            residual_pools[family], rows, columns, config
                        ),
                    )
                residual_losses[config] = losses
        available = [
            config
            for config, losses in primary_losses.items()
            if np.isfinite(losses).all()
        ]
        if not available:
            raise CouplingEstimationRefusal(
                "no exact-field configuration completed the pilot"
            )
        primary = min(
            available, key=lambda config: (float(primary_losses[config].mean()), config)
        )
        graph_zero = min(
            (config for config in available if config.graph_penalty == 0.0),
            key=lambda config: (float(primary_losses[config].mean()), config),
        )
        residual = min(
            residual_losses,
            key=lambda config: (float(residual_losses[config].mean()), config),
        )
        selection = {
            "rule": "minimum donor-equal pilot deviance; lexicographic configuration tie break",
            "primary": asdict(primary),
            "graph_zero": asdict(graph_zero),
            "best_residual": asdict(residual),
            "primary_candidate_count": len(primary_losses),
            "primary_complete_count": len(available),
            "residual_candidate_count": len(residual_losses),
            "primary_candidates": [
                {
                    "configuration": asdict(config),
                    "mean_pilot_loss": float(primary_losses[config].mean()),
                }
                for config in available
            ],
            "residual_candidates": [
                {
                    "configuration": asdict(config),
                    "mean_pilot_loss": float(losses.mean()),
                }
                for config, losses in residual_losses.items()
            ],
        }
        models = core._fit_models(records, calibration, selection)
        losses = core._panel_losses(records, pilot, models)
    return selection, models, losses


def _batch_block_bootstrap(differences: np.ndarray, batches: np.ndarray) -> np.ndarray:
    unique = np.asarray(sorted(set(int(value) for value in batches)), dtype=int)
    sums = np.asarray([differences[batches == batch].sum() for batch in unique])
    counts = np.asarray([np.count_nonzero(batches == batch) for batch in unique])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output = np.empty(BOOTSTRAPS)
    chunk = 1000
    for start in range(0, BOOTSTRAPS, chunk):
        stop = min(start + chunk, BOOTSTRAPS)
        indices = rng.integers(0, len(unique), size=(stop - start, len(unique)))
        output[start:stop] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return output


def _exact_batch_sign_flip(batch_means: np.ndarray) -> dict[str, Any]:
    values = np.asarray(batch_means, dtype=float)
    observed = float(values.mean())
    if len(values) > 20:
        raise ValueError("exact sign-flip enumeration is limited to 20 batches")
    favorable = 0
    total = 1 << len(values)
    for mask in range(total):
        signed = sum(
            (-value if mask & (1 << index) else value)
            for index, value in enumerate(values)
        ) / len(values)
        favorable += signed <= observed + 1e-15
    return {
        "observed_batch_mean_difference": observed,
        "draws": total,
        "one_sided_p": favorable / total,
    }


def _comparison(
    donors: list[str],
    donor_batches: dict[str, int],
    primary: np.ndarray,
    comparator: np.ndarray,
    donor_required: int,
    batch_required: int,
    require_sign_flip: bool,
) -> dict[str, Any]:
    difference = np.asarray(primary) - np.asarray(comparator)
    batches = np.asarray([donor_batches[donor] for donor in donors], dtype=int)
    bootstrap = _batch_block_bootstrap(difference, batches)
    interval = np.quantile(bootstrap, [0.025, 0.975], method="linear")
    unique = sorted(set(batches.tolist()))
    batch_means = np.asarray([difference[batches == batch].mean() for batch in unique])
    relative = 1.0 - float(np.mean(primary) / np.mean(comparator))
    favorable_donors = int(np.count_nonzero(difference < 0.0))
    favorable_batches = int(np.count_nonzero(batch_means < 0.0))
    checks = {
        "relative_deviance_reduction_at_least_five_percent": bool(relative >= 0.05),
        "batch_block_bootstrap_upper_95_below_zero": bool(interval[1] < 0.0),
        "favorable_donor_count_reached": bool(favorable_donors >= donor_required),
        "favorable_batch_count_reached": bool(favorable_batches >= batch_required),
    }
    sign_flip = _exact_batch_sign_flip(batch_means) if require_sign_flip else None
    if sign_flip is not None:
        checks["exact_batch_sign_flip_p_at_most_0_025"] = bool(
            sign_flip["one_sided_p"] <= 0.025
        )
    return {
        "primary_mean_loss": float(np.mean(primary)),
        "comparator_mean_loss": float(np.mean(comparator)),
        "relative_deviance_reduction": relative,
        "mean_paired_difference": float(difference.mean()),
        "batch_block_bootstrap_95_interval": interval.tolist(),
        "bootstrap_draws": BOOTSTRAPS,
        "favorable_donors": favorable_donors,
        "required_favorable_donors": donor_required,
        "favorable_batches": favorable_batches,
        "required_favorable_batches": batch_required,
        "batch_mean_differences": {
            str(batch): float(value) for batch, value in zip(unique, batch_means)
        },
        "exact_batch_sign_flip": sign_flip,
        "checks": checks,
        "passes": all(checks.values()),
        "donor_differences": {
            donor: float(value) for donor, value in zip(donors, difference)
        },
    }


def _gate(
    donors: list[str],
    donor_batches: dict[str, int],
    losses: dict[str, np.ndarray],
    *,
    held: bool,
) -> dict[str, Any]:
    donor_required = HELD_DONOR_REQUIRED if held else PILOT_DONOR_REQUIRED
    batch_required = HELD_BATCH_REQUIRED if held else PILOT_BATCH_REQUIRED
    residual = _comparison(
        donors,
        donor_batches,
        losses["primary"],
        losses["best_residual"],
        donor_required,
        batch_required,
        held,
    )
    destroyed = _comparison(
        donors,
        donor_batches,
        losses["primary"],
        losses["destroyed_link"],
        donor_required,
        batch_required,
        held,
    )
    return {
        "primary_vs_selected_classical_residual": residual,
        "primary_vs_destroyed_link": destroyed,
        "passes": residual["passes"] and destroyed["passes"],
    }


def run_development(
    raw_path: Path, metadata_path: Path, output_path: Path
) -> dict[str, Any]:
    source_record = _validated_raw(raw_path)
    inventory = _validated_inventory(metadata_path)
    calibration = sorted(
        donor for donor, role in inventory["roles"].items() if role == "calibration"
    )
    pilot = sorted(
        donor for donor, role in inventory["roles"].items() if role == "pilot"
    )
    counts, stream_audit = _stream_panel(
        raw_path, calibration + pilot, inventory["selected"], frozenset({"rna", "adt"})
    )
    records = _records_from_counts(calibration + pilot, inventory["selected"], counts)
    selection, calibration_models, pilot_losses = _select_on_pilot(
        records, calibration, pilot
    )
    gate = _gate(pilot, inventory["donor_batch"], pilot_losses, held=False)
    payload: dict[str, Any] = {
        "schema": "gse158769-development/1.0",
        "status": "PILOT_PASS" if gate["passes"] else "TERMINAL_PILOT_FAILURE",
        "created_at_utc": _timestamp(),
        "source_tag": SOURCE_TAG,
        "source_sha256": source_record["sha256"],
        "calibration_donors": calibration,
        "pilot_donors": pilot,
        "calibration_batches": list(CALIBRATION_BATCHES),
        "pilot_batches": list(PILOT_BATCHES),
        "selection": selection,
        "pilot_gate": gate,
        "pilot_losses": {
            method: {donor: float(value) for donor, value in zip(pilot, values)}
            for method, values in pilot_losses.items()
        },
        "stream_access_audit": stream_audit,
        "development_table_sha256": {
            donor: records[donor]["table_sha256"] for donor in calibration + pilot
        },
    }
    if gate["passes"]:
        with _configured_core():
            payload["all_development_models"] = core._fit_models(
                records, calibration + pilot, selection
            )
    else:
        payload["terminal_rule"] = (
            "No held RNA or held ADT values may be tokenized after this pilot failure."
        )
    _write_json(output_path, payload)
    return payload


def run_prediction(
    raw_path: Path, metadata_path: Path, output_path: Path
) -> dict[str, Any]:
    _require_public_tag(
        DEVELOPMENT_TAG,
        (
            *PROTOCOL_BINDINGS,
            _relative(DEFAULT_SOURCE_ACCESS),
            _relative(DEFAULT_DEVELOPMENT),
        ),
    )
    development = _read_json(DEFAULT_DEVELOPMENT)
    if development.get("status") != "PILOT_PASS" or not development.get(
        "pilot_gate", {}
    ).get("passes"):
        raise PermissionError("the frozen pilot gate did not pass")
    _validated_raw(raw_path)
    inventory = _validated_inventory(metadata_path)
    held = sorted(donor for donor, role in inventory["roles"].items() if role == "held")
    counts, stream_audit = _stream_panel(
        raw_path, held, inventory["selected"], frozenset({"rna"})
    )
    if stream_audit["adt_feature_rows_tokenized"] != 0:
        raise PermissionError("prediction tokenized held ADT values")
    samples = []
    with _configured_core():
        for donor in held:
            rna = counts[donor]["rna"]
            positives = (rna > 0).sum(axis=0)
            rows = np.repeat(
                np.stack([CELL_BUDGET - positives, positives], axis=1)[:, None, :],
                len(MARKERS),
                axis=1,
            )
            columns = np.broadcast_to(
                np.asarray([CELL_BUDGET // 2, CELL_BUDGET // 2]),
                (len(MARKERS), len(MARKERS), 2),
            ).copy()
            predictions = core._predict_models(
                development["all_development_models"], rows, columns
            )
            samples.append(
                {
                    "donor": donor,
                    "batch": inventory["donor_batch"][donor],
                    "row_margins": rows.tolist(),
                    "column_margins": columns.tolist(),
                    "predicted_tables": {
                        name: values.tolist() for name, values in predictions.items()
                    },
                    "prediction_sha256": {
                        name: _array_sha256(values)
                        for name, values in predictions.items()
                    },
                    "held_rna_state_sha256": _array_sha256((rna > 0).astype(np.uint8)),
                }
            )
    payload = {
        "schema": "gse158769-held-predictions/1.0",
        "status": "HELD_PREDICTIONS_FROZEN_WITHOUT_ADT_TOKENIZATION",
        "created_at_utc": _timestamp(),
        "development_sha256": _sha256(DEFAULT_DEVELOPMENT),
        "held_donors": held,
        "stream_access_audit": stream_audit,
        "held_adt_values_serialized": 0,
        "samples": samples,
    }
    _write_json(output_path, payload)
    return payload


def run_score(raw_path: Path, metadata_path: Path, output_path: Path) -> dict[str, Any]:
    prediction_commit = _require_public_tag(
        PREDICTION_TAG,
        (
            *PROTOCOL_BINDINGS,
            _relative(DEFAULT_SOURCE_ACCESS),
            _relative(DEFAULT_DEVELOPMENT),
            _relative(DEFAULT_PREDICTION),
        ),
    )
    prediction = _read_json(DEFAULT_PREDICTION)
    if (
        prediction.get("status") != "HELD_PREDICTIONS_FROZEN_WITHOUT_ADT_TOKENIZATION"
        or prediction.get("stream_access_audit", {}).get("adt_feature_rows_tokenized")
        != 0
    ):
        raise PermissionError("held predictions are not a clean RNA-only freeze")
    _validated_raw(raw_path)
    inventory = _validated_inventory(metadata_path)
    held = sorted(donor for donor, role in inventory["roles"].items() if role == "held")
    if held != prediction["held_donors"]:
        raise PermissionError("held donor axis differs from frozen predictions")
    counts, stream_audit = _stream_panel(
        raw_path, held, inventory["selected"], frozenset({"rna", "adt"})
    )
    records = _records_from_counts(held, inventory["selected"], counts)
    frozen = {sample["donor"]: sample for sample in prediction["samples"]}
    methods = sorted(frozen[held[0]]["predicted_tables"])
    losses = {method: np.empty(len(held)) for method in methods}
    samples = []
    with _configured_core():
        for donor_index, donor in enumerate(held):
            truth = np.asarray(records[donor]["tables"])
            rows, columns = core._margins(truth)
            sample = frozen[donor]
            if (
                rows.tolist() != sample["row_margins"]
                or columns.tolist() != sample["column_margins"]
            ):
                raise PermissionError("held truth margins differ from frozen margins")
            donor_losses = {}
            for method in methods:
                estimate = np.asarray(sample["predicted_tables"][method])
                if _array_sha256(estimate) != sample["prediction_sha256"][method]:
                    raise PermissionError("held prediction hash differs")
                loss = core._donor_loss(truth, estimate)
                losses[method][donor_index] = loss
                donor_losses[method] = float(loss)
            samples.append(
                {
                    "donor": donor,
                    "batch": inventory["donor_batch"][donor],
                    "losses": donor_losses,
                    "truth_table_sha256": records[donor]["table_sha256"],
                }
            )
    gate = _gate(held, inventory["donor_batch"], losses, held=True)
    payload = {
        "schema": "gse158769-confirmation/1.0",
        "status": "CONFIRMATION_PASS" if gate["passes"] else "CONFIRMATION_FAIL",
        "created_at_utc": _timestamp(),
        "prediction_tag": PREDICTION_TAG,
        "prediction_commit": prediction_commit,
        "prediction_sha256": _sha256(DEFAULT_PREDICTION),
        "held_donors": held,
        "held_batches": list(HELD_BATCHES),
        "stream_access_audit": stream_audit,
        "losses": {
            method: {donor: float(value) for donor, value in zip(held, values)}
            for method, values in losses.items()
        },
        "gate": gate,
        "samples": samples,
    }
    _write_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    preflight.add_argument("--output", type=Path, default=DEFAULT_PREFLIGHT)
    acquire = subparsers.add_parser("acquire")
    acquire.add_argument("--output", type=Path, default=DEFAULT_RAW)
    acquire.add_argument("--record", type=Path, default=DEFAULT_SOURCE_ACCESS)
    for command in ("develop", "predict", "score"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
        subparser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
        default = {
            "develop": DEFAULT_DEVELOPMENT,
            "predict": DEFAULT_PREDICTION,
            "score": DEFAULT_SCORE,
        }[command]
        subparser.add_argument("--output", type=Path, default=default)
    args = parser.parse_args()
    if args.command == "preflight":
        payload = run_preflight(args.metadata, args.output)
    elif args.command == "acquire":
        payload = acquire_raw(args.output, args.record)
    elif args.command == "develop":
        payload = run_development(args.raw, args.metadata, args.output)
    elif args.command == "predict":
        payload = run_prediction(args.raw, args.metadata, args.output)
    else:
        payload = run_score(args.raw, args.metadata, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
