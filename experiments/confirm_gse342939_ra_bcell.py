"""One-shot GSE342939 longitudinal RA B-cell CITE-seq campaign."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import gzip
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import stat
import sys
import tempfile
from typing import Any, Iterable, Iterator
import urllib.request

import numpy as np
import scipy

from experiments import confirm_gse179221_bmmc as common
from experiments import confirm_gse309593_held_batches as table_engine
from mapreg.common_effect_conditional import fit_common_effect_conditional_log_odds
from mapreg.heterogeneity_adaptive_coupling import (
    CouplingEstimationRefusal,
    _fixed_margin_support,
    expected_binary_table_from_log_odds,
    signed_deviance_coordinate,
    signed_pearson_coordinate,
)
from mapreg.longitudinal_conditional_coupling import (
    fit_longitudinal_conditional_log_odds,
    fit_visit_agnostic_conditional_log_odds,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/confirmation/gse342939_ra_bcell"
DESIGNATION = DATA_DIR / "candidate_designation_v1.json"
MANIFEST = DATA_DIR / "metadata_access_manifest_v1.json"
AMENDMENT = DATA_DIR / "pre_access_implementation_amendment_v1.json"
STREAMING_CLARIFICATION = (
    DATA_DIR / "pre_access_streaming_reduction_clarification_v1.json"
)
PROTOCOL = (
    ROOT
    / "docs/GSE342939_RA_BCELL_CITESEQ_HELD_DONOR_PROTOCOL_2026-08-29.md"
)
RUNNER = ROOT / "experiments/confirm_gse342939_ra_bcell.py"
TEST = ROOT / "tests/test_gse342939_confirmation.py"
ENGINE = ROOT / "mapreg/longitudinal_conditional_coupling.py"

SOURCE_ATTEMPT = DATA_DIR / "source_attempt_v1.json"
SOURCE_CONSUMPTION = DATA_DIR / "source_consumption_v1.json"
SOURCE_RESULT = ROOT / "results/development/gse342939_ra_bcell_source_v1.json"
HELD_RNA_ATTEMPT = DATA_DIR / "held_rna_attempt_v1.json"
HELD_RNA_CONSUMPTION = DATA_DIR / "held_rna_consumption_v1.json"
HELD_RNA_MARGINS = ROOT / "results/gse342939_ra_bcell_held_rna_margins_v1.json"
HELD_ADT_ATTEMPT = DATA_DIR / "held_adt_attempt_v1.json"
HELD_ADT_CONSUMPTION = DATA_DIR / "held_adt_consumption_v1.json"
HELD_MARGINS = ROOT / "results/gse342939_ra_bcell_held_margins_v1.json"
HELD_PREDICTIONS = ROOT / "results/gse342939_ra_bcell_predictions_v1.json"
SCORE_AUTHORIZATION = DATA_DIR / "score_authorization_v1.json"
SCORE_ATTEMPT = DATA_DIR / "score_attempt_v1.json"
SCORE_RESULT = ROOT / "results/gse342939_ra_bcell_confirmation_v1.json"
PRIVATE_BARCODES = DATA_DIR / "private_held_selected_barcodes_v1.json"
PRIVATE_RNA = DATA_DIR / "private_held_rna_states_v1.npz"
PRIVATE_ADT = DATA_DIR / "private_held_adt_states_v1.npz"
DEFAULT_AXIS_CACHE = Path(
    os.environ.get(
        "GSE342939_AXIS_CACHE",
        "/private/tmp/gse342939-metadata-freeze-v1/axes",
    )
)
DEFAULT_SCRATCH = Path("/private/tmp/gse342939-ra-bcell-v1")

PUBLIC_ORIGIN = "https://github.com/sushaan-k/coupling-fields-benchmark.git"
CANDIDATE_TAG = "gse342939-ra-bcell-v1-candidate"
AMENDMENT_TAG = "gse342939-ra-bcell-v1-pre-access-amendment"
STREAMING_CLARIFICATION_TAG = (
    "gse342939-ra-bcell-v1-streaming-reduction-clarification"
)
IMPLEMENTATION_TAG = "gse342939-ra-bcell-v1-implementation"
SOURCE_ATTEMPT_TAG = "gse342939-ra-bcell-v1-source-attempt"
SOURCE_TAG = "gse342939-ra-bcell-v1-source"
HELD_RNA_ATTEMPT_TAG = "gse342939-ra-bcell-v1-held-rna-attempt"
HELD_RNA_MARGINS_TAG = "gse342939-ra-bcell-v1-held-rna-margins"
HELD_ADT_ATTEMPT_TAG = "gse342939-ra-bcell-v1-held-adt-attempt"
MARGINS_TAG = "gse342939-ra-bcell-v1-margins"
PREDICTION_TAG = "gse342939-ra-bcell-v1-prediction"
SCORE_AUTHORIZATION_TAG = "gse342939-ra-bcell-v1-score-authorization"

CANDIDATE_SHA256 = "26816d705f201227fb86fe1e6cb167fdc4879dbc9a02d24a3b00171c73911d9e"
MANIFEST_SHA256 = "14a8389a97da0f0ec2581d6b72aa59ab8932522f314fdf8a205c8a768f20dd53"
AMENDMENT_SHA256 = "583890f90f0c89190cdb74a47170832d987a8d90b193a6acc3a1c923b031f0ab"
STREAMING_CLARIFICATION_SHA256 = (
    "75d651d7f10b93e03f438fc35ea1420adca987a9f518afc8b9e1896c87c89ece"
)
MARKER_COUNT = 45
MINIMUM_CELLS = 128
MAXIMUM_CELLS = 512
MINIMUM_COORDINATES = 256
MINIMUM_DETECTED_GENES = 200
MAXIMUM_MITOCHONDRIAL_FRACTION = 0.10
MAXIMUM_RNA_UMIS = 70_000
CELL_SALT = "GSE342939-RA-B-CELL-v1"
ADT_TIE_SALT = "GSE342939-RA-B-ADT-v1"
BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 20260830
GRAPH_NEIGHBORS = 2
HETEROGENEITY_GRID = (0.1, 1.0, 10.0)
RIDGE_GRID = (0.01, 0.1)
GRAPH_GRID = (0.0, 0.03, 0.3)
TRANSPORT_GRID = (0.0, 0.5, 1.0, 1.5)
RESIDUAL_FAMILIES = ("pearson", "root_deviance")
VISITS = ("pre", "post")
MAXIMUM_AUTHORIZED_GEX_BARCODES = 19_683
GEX_FEATURE_ROWS = 36_617
MAXIMUM_PACKED_DETECTION_BITSET_BYTES = 90_091_552
MAXIMUM_COMPRESSED_MATRIX_BYTES = 102_854_757
MINIMUM_SCRATCH_FREE_BYTES = 1_176_596_581

IMPLEMENTATION_BINDINGS = (
    "experiments/confirm_gse342939_ra_bcell.py",
    "tests/test_gse342939_confirmation.py",
    "mapreg/longitudinal_conditional_coupling.py",
    "tests/test_longitudinal_conditional_coupling.py",
    "data/confirmation/gse342939_ra_bcell/"
    "pre_access_streaming_reduction_clarification_v1.json",
    "mapreg/common_effect_conditional.py",
    "mapreg/heterogeneity_adaptive_coupling.py",
    "experiments/confirm_gse179221_bmmc.py",
    "experiments/confirm_gse309593_held_batches.py",
)

ProtocolRefusal = common.ProtocolRefusal
_array_sha256 = common._array_sha256
_axis_sha256 = common._axis_sha256
_read_json = common._read_json
_require_ancestor = common._require_ancestor
_sha256 = common._sha256
_timestamp = common._timestamp
_write_json_x = common._write_json_x


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _require_canonical_path(observed: Path, expected: Path, label: str) -> None:
    if observed.resolve() != expected.resolve():
        raise PermissionError(f"{label} must use its canonical campaign path")


def _binding_hashes(paths: Iterable[str] = IMPLEMENTATION_BINDINGS) -> dict[str, str]:
    return {path: _sha256(ROOT / path) for path in paths}


def _runtime_record() -> dict[str, Any]:
    variables = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "resolved_executable": str(Path(sys.executable).resolve()),
        },
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "thread_environment": {name: os.environ.get(name) for name in variables},
    }


def _require_public_tag(tag: str, paths: Iterable[str]) -> str:
    previous = common.PUBLIC_ORIGIN
    common.PUBLIC_ORIGIN = PUBLIC_ORIGIN
    try:
        return common._require_public_tag(tag, paths)
    finally:
        common.PUBLIC_ORIGIN = previous


def _candidate(path: Path = DESIGNATION) -> dict[str, Any]:
    value = _read_json(path)
    if (
        _sha256(path) != CANDIDATE_SHA256
        or value.get("schema")
        != "gse342939-ra-bcell-citeseq-candidate-designation/1.0"
        or value.get("status")
        != "FROZEN_FROM_OFFICIAL_METADATA_AND_NONNUMERIC_AXES_BEFORE_ANY_NUMERIC_MATRIX_ACCESS"
        or len(value.get("source_donors", [])) != 7
        or len(value.get("held_donors", [])) != 6
        or len(value.get("ordered_cognate_panel", [])) != MARKER_COUNT
        or value.get("panel_rule", {}).get("coordinate_universe")
        != "all 2025 ordered RNA-to-ADT pairs"
        or value.get("numeric_matrix_inventory", {}).get("files") != 52
        or value.get("numeric_matrix_inventory", {}).get(
            "response_bodies_accessed_before_freeze"
        )
        != 0
    ):
        raise PermissionError("candidate designation differs from the frozen campaign")
    source = {record["donor"] for record in value["source_donors"]}
    held = {record["donor"] for record in value["held_donors"]}
    if source & held or source | held != {
        "NN1",
        "NN2",
        "NN3",
        "PC1",
        "PC2",
        "PC3",
        "PC4",
        "PC5",
        "PN1",
        "PN2",
        "PN3",
        "PN4",
        "PN5",
    }:
        raise PermissionError("candidate source and held donor axes differ")
    return value


def _manifest(path: Path = MANIFEST) -> dict[str, Any]:
    value = _read_json(path)
    boundary = value.get("access_boundary", {})
    if (
        _sha256(path) != MANIFEST_SHA256
        or value.get("schema")
        != "gse342939-ra-bcell-citeseq-metadata-access-manifest/1.0"
        or len(value.get("records", [])) != 107
        or boundary.get("numeric_matrix_body_gets") != 0
        or boundary.get("network_get_response_bytes") != 98_414_933
    ):
        raise PermissionError("metadata-access manifest differs from the freeze")
    return value


def _amendment(path: Path = AMENDMENT) -> dict[str, Any]:
    value = _read_json(path)
    primary = value.get("estimator_and_comparator_contract", {}).get("primary", {})
    firewall = value.get("access_firewall", {})
    if (
        _sha256(path) != AMENDMENT_SHA256
        or value.get("schema")
        != "gse342939-ra-bcell-citeseq-pre-access-implementation-amendment/1.0"
        or value.get("status") != "FROZEN_BEFORE_ANY_NUMERIC_MATRIX_ACCESS"
        or primary.get("heterogeneity_penalty_grid") != list(HETEROGENEITY_GRID)
        or primary.get("population_ridge_grid") != list(RIDGE_GRID)
        or primary.get("graph_penalty_grid") != list(GRAPH_GRID)
        or primary.get("baseline_transport_grid") != list(TRANSPORT_GRID)
        or primary.get("visit_change_transport_grid") != list(TRANSPORT_GRID)
        or firewall.get("future_cli_stages")
        != [
            "claim-source",
            "run-source",
            "claim-held-rna",
            "run-held-rna",
            "claim-held-adt",
            "run-held-adt",
            "predict-held",
            "authorize-score",
            "score-held",
        ]
    ):
        raise PermissionError("pre-access amendment differs from the frozen campaign")
    return value


def _streaming_clarification(
    path: Path = STREAMING_CLARIFICATION,
) -> dict[str, Any]:
    value = _read_json(path)
    contract = value.get("streaming_reduction_contract", {})
    if (
        _sha256(path) != STREAMING_CLARIFICATION_SHA256
        or value.get("schema")
        != "gse342939-ra-bcell-citeseq-pre-access-streaming-reduction-clarification/1.0"
        or value.get("status") != "FROZEN_BEFORE_ANY_NUMERIC_MATRIX_ACCESS"
        or value.get("candidate_sha256") != CANDIDATE_SHA256
        or value.get("metadata_access_manifest_sha256") != MANIFEST_SHA256
        or value.get("pre_access_implementation_amendment_sha256")
        != AMENDMENT_SHA256
        or contract.get("maximum_authorized_gex_barcodes")
        != MAXIMUM_AUTHORIZED_GEX_BARCODES
        or contract.get("gex_feature_rows") != GEX_FEATURE_ROWS
        or contract.get("maximum_packed_detection_bitset_bytes")
        != MAXIMUM_PACKED_DETECTION_BITSET_BYTES
        or contract.get("maximum_compressed_matrix_bytes")
        != MAXIMUM_COMPRESSED_MATRIX_BYTES
        or contract.get("minimum_scratch_free_bytes_before_attempt_claim")
        != MINIMUM_SCRATCH_FREE_BYTES
        or value.get("scientific_contract_changed") is not False
        or value.get("numeric_matrix_access_before_clarification") is not False
    ):
        raise PermissionError("streaming reduction clarification differs from freeze")
    return value


def _verify_public_freezes() -> dict[str, str]:
    _streaming_clarification()
    candidate_commit = _require_public_tag(
        CANDIDATE_TAG,
        (_relative(DESIGNATION), _relative(MANIFEST), _relative(PROTOCOL)),
    )
    amendment_commit = _require_public_tag(
        AMENDMENT_TAG,
        (
            _relative(DESIGNATION),
            _relative(MANIFEST),
            _relative(PROTOCOL),
            _relative(AMENDMENT),
        ),
    )
    _require_ancestor(candidate_commit, amendment_commit)
    clarification_commit = _require_public_tag(
        STREAMING_CLARIFICATION_TAG,
        (
            _relative(DESIGNATION),
            _relative(MANIFEST),
            _relative(AMENDMENT),
            _relative(STREAMING_CLARIFICATION),
        ),
    )
    _require_ancestor(amendment_commit, clarification_commit)
    implementation_commit = _require_public_tag(
        IMPLEMENTATION_TAG,
        (*IMPLEMENTATION_BINDINGS, _relative(AMENDMENT)),
    )
    _require_ancestor(clarification_commit, implementation_commit)
    return {
        "candidate_tag": CANDIDATE_TAG,
        "candidate_commit": candidate_commit,
        "amendment_tag": AMENDMENT_TAG,
        "amendment_commit": amendment_commit,
        "streaming_clarification_tag": STREAMING_CLARIFICATION_TAG,
        "streaming_clarification_commit": clarification_commit,
        "implementation_tag": IMPLEMENTATION_TAG,
        "implementation_commit": implementation_commit,
    }


def _require_public_result(
    tag: str, path: Path, expected_schema: str
) -> tuple[dict[str, Any], str]:
    public = _verify_public_freezes()
    commit = _require_public_tag(tag, (_relative(path),))
    _require_ancestor(public["implementation_commit"], commit)
    value = _read_json(path)
    if value.get("schema") != expected_schema:
        raise PermissionError(f"public {tag} artifact has the wrong schema")
    return value, commit


def _axis_records() -> dict[str, dict[str, Any]]:
    return {
        record["name"]: record
        for record in _manifest()["records"]
        if record["content_class"].startswith("nonnumeric_")
    }


def _decoded_axis(path: Path, expected: dict[str, Any]) -> list[str]:
    if not path.is_file():
        raise ProtocolRefusal("AXIS_CACHE_FILE_ABSENT", details={"name": path.name})
    if path.stat().st_size != expected["bytes"] or _sha256(path) != expected["sha256"]:
        raise ProtocolRefusal("AXIS_CACHE_COMPRESSED_IDENTITY_MISMATCH", details={"name": path.name})
    digest = hashlib.sha256()
    values: list[str] = []
    with gzip.open(path, "rb") as stream:
        for raw in stream:
            digest.update(raw)
            values.append(raw.decode("utf-8", errors="strict").rstrip("\r\n"))
    if (
        len(values) != expected["decoded_line_count"]
        or len(values) != len(set(values))
        or digest.hexdigest() != expected["decoded_sha256"]
    ):
        raise ProtocolRefusal("AXIS_CACHE_DECODED_IDENTITY_MISMATCH", details={"name": path.name})
    return values


def _validate_axis_cache(axis_cache: Path) -> dict[str, Any]:
    records = _axis_records()
    if len(records) != 104:
        raise PermissionError("metadata manifest does not bind 104 axis files")
    identities = []
    for name in sorted(records):
        values = _decoded_axis(axis_cache / name, records[name])
        identities.append(
            {
                "name": name,
                "bytes": records[name]["bytes"],
                "sha256": records[name]["sha256"],
                "decoded_line_count": len(values),
                "decoded_sha256": records[name]["decoded_sha256"],
            }
        )
    encoded = json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()
    return {
        "files": len(identities),
        "bytes": sum(record["bytes"] for record in identities),
        "identity_sha256": hashlib.sha256(encoded).hexdigest(),
        "metadata_manifest_sha256": MANIFEST_SHA256,
    }


def _exclusive_private_json(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(stat.S_IRUSR | stat.S_IWUSR)
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    finally:
        temporary.unlink(missing_ok=True)


def _exclusive_private_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            np.savez_compressed(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(stat.S_IRUSR | stat.S_IWUSR)
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    finally:
        temporary.unlink(missing_ok=True)


def _scratch_capacity_certificate(
    scratch: Path, matrices: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    records = list(matrices)
    maximum = max(int(record["expected_bytes"]) for record in records)
    if maximum > MAXIMUM_COMPRESSED_MATRIX_BYTES:
        raise PermissionError("stage matrix exceeds the frozen campaign maximum")
    resolved = scratch.resolve()
    if resolved == ROOT.resolve() or ROOT.resolve() in resolved.parents:
        raise PermissionError("scratch must remain outside the public repository")
    scratch.mkdir(parents=True, exist_ok=True, mode=0o700)
    if scratch.is_symlink() or any(scratch.iterdir()):
        raise PermissionError("scratch must be a real empty directory")
    scratch.chmod(stat.S_IRWXU)
    free = int(shutil.disk_usage(scratch).free)
    if free < MINIMUM_SCRATCH_FREE_BYTES:
        raise ProtocolRefusal(
            "SCRATCH_CAPACITY_BELOW_FROZEN_MINIMUM",
            details={
                "free_bytes": free,
                "minimum_free_bytes": MINIMUM_SCRATCH_FREE_BYTES,
            },
        )
    return {
        "scratch_path_disclosed": False,
        "scratch_empty_at_check": True,
        "filesystem_free_bytes": free,
        "maximum_stage_matrix_bytes": maximum,
        "maximum_campaign_matrix_bytes": MAXIMUM_COMPRESSED_MATRIX_BYTES,
        "maximum_packed_detection_bitset_bytes": (
            MAXIMUM_PACKED_DETECTION_BITSET_BYTES
        ),
        "minimum_free_bytes": MINIMUM_SCRATCH_FREE_BYTES,
        "temporary_coordinate_store_permitted": False,
        "passes": True,
    }


def _valid_scratch_capacity_certificate(
    value: Any, matrices: Iterable[dict[str, Any]]
) -> bool:
    records = list(matrices)
    maximum = max(int(record["expected_bytes"]) for record in records)
    if not isinstance(value, dict):
        return False
    free = value.get("filesystem_free_bytes")
    return bool(
        value.get("scratch_path_disclosed") is False
        and value.get("scratch_empty_at_check") is True
        and isinstance(free, int)
        and free >= MINIMUM_SCRATCH_FREE_BYTES
        and value.get("maximum_stage_matrix_bytes") == maximum
        and value.get("maximum_campaign_matrix_bytes")
        == MAXIMUM_COMPRESSED_MATRIX_BYTES
        and value.get("maximum_packed_detection_bitset_bytes")
        == MAXIMUM_PACKED_DETECTION_BITSET_BYTES
        and value.get("minimum_free_bytes") == MINIMUM_SCRATCH_FREE_BYTES
        and value.get("temporary_coordinate_store_permitted") is False
        and value.get("passes") is True
    )


def _require_scratch_capacity_certificate(
    value: Any, matrices: Iterable[dict[str, Any]], stage: str
) -> None:
    if not _valid_scratch_capacity_certificate(value, matrices):
        raise PermissionError(f"{stage} scratch capacity certificate is invalid")


def claim_source(
    attempt_path: Path = SOURCE_ATTEMPT,
    axis_cache: Path = DEFAULT_AXIS_CACHE,
    scratch: Path = DEFAULT_SCRATCH,
) -> dict[str, Any]:
    """Claim the source run before any numeric Matrix Market GET."""

    _require_canonical_path(attempt_path, SOURCE_ATTEMPT, "source attempt")
    forbidden = (
        SOURCE_CONSUMPTION,
        SOURCE_RESULT,
        HELD_RNA_ATTEMPT,
        HELD_RNA_MARGINS,
        HELD_ADT_ATTEMPT,
        HELD_MARGINS,
        HELD_PREDICTIONS,
        SCORE_AUTHORIZATION,
        SCORE_ATTEMPT,
        SCORE_RESULT,
    )
    if attempt_path.exists() or any(path.exists() for path in forbidden):
        raise FileExistsError("source campaign has already been claimed or advanced")
    candidate = _candidate()
    _amendment()
    tags = _verify_public_freezes()
    axis_certificate = _validate_axis_cache(axis_cache)
    source_files = [
        visit[assay]["matrix"]
        for donor in candidate["source_donors"]
        for visit in donor["visits"]
        for assay in ("gex", "cite")
    ]
    scratch_capacity = _scratch_capacity_certificate(scratch, source_files)
    payload = {
        "schema": "gse342939-ra-bcell-source-attempt/1.0",
        "status": "CLAIMED_BEFORE_FIRST_NUMERIC_MATRIX_GET",
        "created_at_utc": _timestamp(),
        "candidate_sha256": CANDIDATE_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "amendment_sha256": AMENDMENT_SHA256,
        "implementation_bindings": _binding_hashes(),
        "public_tags": tags,
        "runtime": _runtime_record(),
        "axis_cache_certificate": axis_certificate,
        "scratch_capacity_certificate": scratch_capacity,
        "source_numeric_urls": [record["url"] for record in source_files],
        "source_numeric_expected_bytes": sum(
            int(record["expected_bytes"]) for record in source_files
        ),
        "source_numeric_file_count": len(source_files),
        "held_numeric_access_authorized": False,
        "rerun_permitted": False,
    }
    _write_json_x(attempt_path, payload)
    return payload


def _validate_source_attempt(
    path: Path = SOURCE_ATTEMPT,
    axis_cache: Path = DEFAULT_AXIS_CACHE,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    _require_canonical_path(path, SOURCE_ATTEMPT, "source attempt")
    candidate = _candidate()
    _amendment()
    attempt = _read_json(path)
    public_tags = _verify_public_freezes()
    matrices = [
        visit[assay]["matrix"]
        for donor in candidate["source_donors"]
        for visit in donor["visits"]
        for assay in ("gex", "cite")
    ]
    if (
        attempt.get("schema") != "gse342939-ra-bcell-source-attempt/1.0"
        or attempt.get("status") != "CLAIMED_BEFORE_FIRST_NUMERIC_MATRIX_GET"
        or attempt.get("candidate_sha256") != CANDIDATE_SHA256
        or attempt.get("manifest_sha256") != MANIFEST_SHA256
        or attempt.get("amendment_sha256") != AMENDMENT_SHA256
        or attempt.get("implementation_bindings") != _binding_hashes()
        or attempt.get("public_tags") != public_tags
        or attempt.get("runtime") != _runtime_record()
        or attempt.get("axis_cache_certificate") != _validate_axis_cache(axis_cache)
        or not _valid_scratch_capacity_certificate(
            attempt.get("scratch_capacity_certificate"), matrices
        )
        or attempt.get("source_numeric_file_count") != 28
        or attempt.get("source_numeric_urls")
        != [record["url"] for record in matrices]
        or attempt.get("source_numeric_expected_bytes")
        != sum(int(record["expected_bytes"]) for record in matrices)
        or attempt.get("held_numeric_access_authorized") is not False
        or attempt.get("rerun_permitted") is not False
    ):
        raise PermissionError("source attempt differs from the frozen implementation")
    attempt_commit = _require_public_tag(SOURCE_ATTEMPT_TAG, (_relative(path),))
    _require_ancestor(public_tags["implementation_commit"], attempt_commit)
    return candidate, attempt, attempt_commit


def _validate_response_url(expected_url: str, observed_url: str) -> None:
    common._validate_frozen_response_url(expected_url, observed_url)


def _fetch_matrix(
    record: dict[str, Any],
    scratch: Path,
    identity: dict[str, Any],
) -> Path:
    url = str(record["url"])
    expected_bytes = int(record["expected_bytes"])
    scratch.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="gse342939-", suffix=".matrix.mtx.gz", dir=scratch
    )
    os.close(descriptor)
    path = Path(temporary_name)
    digest = hashlib.sha256()
    observed_bytes = 0
    identity.update(
        {
            "requested_url": url,
            "expected_bytes": expected_bytes,
            "request_started": True,
            "completed": False,
            "deleted": False,
        }
    )
    try:
        opener = urllib.request.build_opener(common._FrozenRedirectHandler(url))
        request = urllib.request.Request(url, headers={"User-Agent": "mapreg/1.0"})
        with opener.open(request, timeout=120) as response, path.open("wb") as output:
            final_url = response.geturl()
            identity["final_url"] = final_url
            _validate_response_url(url, final_url)
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                observed_bytes += len(block)
                digest.update(block)
                if observed_bytes > expected_bytes:
                    raise ProtocolRefusal("MATRIX_BYTE_COUNT_EXCEEDED")
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        identity["observed_bytes"] = observed_bytes
        identity["hashed_bytes"] = observed_bytes
        identity["sha256"] = digest.hexdigest()
        if observed_bytes != expected_bytes or path.stat().st_size != expected_bytes:
            raise ProtocolRefusal("MATRIX_BYTE_COUNT_MISMATCH")
        identity["completed"] = True
        return path
    except BaseException:
        identity["observed_bytes"] = observed_bytes
        identity["hashed_bytes"] = observed_bytes
        identity["partial_sha256"] = digest.hexdigest()
        path.unlink(missing_ok=True)
        identity["deleted"] = not path.exists()
        raise


def _axis_values(
    axis_cache: Path,
    record: dict[str, Any],
    records: dict[str, dict[str, Any]],
) -> list[str]:
    name = record["filename"]
    expected = records.get(name)
    if expected is None:
        raise PermissionError("candidate axis is absent from the metadata manifest")
    if (
        record.get("bytes") != expected["bytes"]
        or record.get("sha256") != expected["sha256"]
        or record.get("decoded_line_count") != expected["decoded_line_count"]
    ):
        raise PermissionError("candidate axis identity differs from the manifest")
    return _decoded_axis(axis_cache / name, expected)


def _visit_axes(
    candidate: dict[str, Any],
    visit: dict[str, Any],
    axis_cache: Path,
) -> dict[str, Any]:
    records = _axis_records()
    gex_features = _axis_values(
        axis_cache, visit["gex"]["features_axis"], records
    )
    gex_barcodes = _axis_values(
        axis_cache, visit["gex"]["barcodes_axis"], records
    )
    cite_features = _axis_values(
        axis_cache, visit["cite"]["features_axis"], records
    )
    cite_barcodes = _axis_values(
        axis_cache, visit["cite"]["barcodes_axis"], records
    )
    if len(gex_features) != 36_617 or any(
        len(line.split("\t")) != 3 for line in gex_features
    ):
        raise ProtocolRefusal("GEX_FEATURE_AXIS_SCHEMA_MISMATCH")
    if len(cite_features) != 63 or cite_features[-1] != "unmapped":
        raise ProtocolRefusal("CITE_FEATURE_AXIS_SCHEMA_MISMATCH")
    if any(not barcode.endswith("-1") for barcode in gex_barcodes):
        raise ProtocolRefusal("GEX_BARCODE_SUFFIX_MISMATCH")
    normalized = [barcode[:-2] for barcode in gex_barcodes]
    if len(normalized) != len(set(normalized)) or len(cite_barcodes) != len(
        set(cite_barcodes)
    ):
        raise ProtocolRefusal("NORMALIZED_BARCODE_AXIS_DUPLICATED")
    cite_set = set(cite_barcodes)
    intersection = [barcode for barcode in normalized if barcode in cite_set]
    if len(intersection) != visit["nonnumeric_axis_intersection_barcodes"]:
        raise ProtocolRefusal("FROZEN_BARCODE_INTERSECTION_MISMATCH")

    panel = candidate["ordered_cognate_panel"]
    parsed_gex = [line.split("\t") for line in gex_features]
    gex_symbols = [fields[1] for fields in parsed_gex]
    gene_expression_rows = np.asarray(
        [fields[2] == "Gene Expression" for fields in parsed_gex], dtype=bool
    )
    rna_rows = []
    adt_rows = []
    for marker in panel:
        rna = [
            index
            for index, symbol in enumerate(gex_symbols)
            if symbol == marker["rna"] and gene_expression_rows[index]
        ]
        adt = [
            index
            for index, feature in enumerate(cite_features)
            if feature == marker["adt_axis_exact"]
        ]
        if len(rna) != 1 or len(adt) != 1:
            raise ProtocolRefusal("FROZEN_PANEL_AXIS_NOT_EXACTLY_UNIQUE")
        rna_rows.append(rna[0])
        adt_rows.append(adt[0])
    return {
        "gex_features": gex_features,
        "gex_barcodes": gex_barcodes,
        "gex_normalized_barcodes": normalized,
        "cite_features": cite_features,
        "cite_barcodes": cite_barcodes,
        "intersection": intersection,
        "rna_rows": np.asarray(rna_rows, dtype=np.int64),
        "adt_rows": np.asarray(adt_rows, dtype=np.int64),
        "gene_expression_rows": gene_expression_rows,
    }


_INDEX_TOKEN = re.compile(r"[1-9][0-9]*\Z")
_COUNT_TOKEN = re.compile(r"\+?[0-9]+\Z")


@contextmanager
def _matrix_entries(
    path: Path,
    expected_shape: tuple[int, int],
    audit: dict[str, Any] | None = None,
) -> Iterator[tuple[Iterator[tuple[int, int, int]], dict[str, Any]]]:
    if audit is None:
        audit = {}
    elif audit:
        raise ValueError("matrix audit record must start empty")
    audit.update(
        {
            "parser_started": True,
            "expected_shape": list(expected_shape),
            "parse_completed": False,
            "entry_iteration_completed": False,
        }
    )
    stream = gzip.open(path, "rt", encoding="utf-8", errors="strict", newline="")
    try:
        header = stream.readline()
        if header.strip().lower() != "%%matrixmarket matrix coordinate integer general":
            raise ProtocolRefusal("MATRIX_MARKET_HEADER_MISMATCH")
        line_number = 1
        for line in stream:
            line_number += 1
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            tokens = stripped.split()
            if len(tokens) != 3 or any(_COUNT_TOKEN.fullmatch(token) is None for token in tokens):
                raise ProtocolRefusal("MATRIX_MARKET_DIMENSIONS_INVALID")
            rows, columns, declared = (int(token) for token in tokens)
            break
        else:
            raise ProtocolRefusal("MATRIX_MARKET_DIMENSIONS_MISSING")
        if (rows, columns) != expected_shape:
            raise ProtocolRefusal(
                "MATRIX_MARKET_AXIS_DIMENSION_MISMATCH",
                details={"observed": [rows, columns], "expected": list(expected_shape)},
            )
        audit.update(
            {
                "rows": rows,
                "columns": columns,
                "declared_entries": declared,
                "raw_entries_seen": 0,
                "raw_entries_yielded": 0,
                "maximum_raw_count": 0,
                "matrix_total_count": 0,
                "matrix_total_within_int64": False,
                "all_body_entries_validated": False,
                "gzip_eof_verified": False,
                "declared_entry_count_validated": False,
                "sorted_independent_duplicate_accumulation": True,
                "full_dense_matrix_materialized": False,
                "full_coordinate_set_materialized": False,
                "temporary_database_created": False,
                "parser_backend": "validated_single_pass_gzip_stream",
                "temporary_storage_bytes": 0,
                "peak_in_memory_entry_batch": 1,
            }
        )

        def iterator() -> Iterator[tuple[int, int, int]]:
            nonlocal line_number
            for body_line in stream:
                line_number += 1
                stripped_body = body_line.strip()
                if not stripped_body or stripped_body.startswith("%"):
                    continue
                fields = stripped_body.split()
                if (
                    len(fields) != 3
                    or _INDEX_TOKEN.fullmatch(fields[0]) is None
                    or _INDEX_TOKEN.fullmatch(fields[1]) is None
                    or _COUNT_TOKEN.fullmatch(fields[2]) is None
                ):
                    raise ProtocolRefusal("MATRIX_MARKET_ENTRY_INVALID")
                row = int(fields[0])
                column = int(fields[1])
                value = int(fields[2])
                if row > rows or column > columns or value > np.iinfo(np.int64).max:
                    raise ProtocolRefusal("MATRIX_MARKET_ENTRY_OUT_OF_RANGE")
                audit["raw_entries_seen"] += 1
                audit["raw_entries_yielded"] += 1
                audit["maximum_raw_count"] = max(audit["maximum_raw_count"], value)
                audit["matrix_total_count"] += value
                if audit["matrix_total_count"] > np.iinfo(np.int64).max:
                    raise ProtocolRefusal(
                        "MATRIX_MARKET_TOTAL_EXCEEDS_INT64_CERTIFICATE"
                    )
                audit["last_validated_entry"] = audit["raw_entries_seen"]
                audit["final_line_number"] = line_number
                yield row - 1, column - 1, value
            if audit["raw_entries_seen"] != declared:
                raise ProtocolRefusal("MATRIX_MARKET_DECLARED_ENTRY_COUNT_MISMATCH")
            audit["matrix_total_within_int64"] = True
            audit["all_body_entries_validated"] = True
            audit["gzip_eof_verified"] = True
            audit["declared_entry_count_validated"] = True
            audit["parse_completed"] = True
            audit["entry_iteration_completed"] = True

        yield iterator(), audit
        if not audit["entry_iteration_completed"]:
            raise ProtocolRefusal("MATRIX_MARKET_ITERATION_INCOMPLETE")
    finally:
        stream.close()


def _selected_cell_indices(
    donor: str,
    visit: str,
    barcodes: list[str],
    eligible: np.ndarray,
) -> list[int]:
    candidates = [int(value) for value in np.asarray(eligible, dtype=np.int64)]
    count = min(MAXIMUM_CELLS, 2 * (len(candidates) // 2))
    if count < MINIMUM_CELLS:
        raise ProtocolRefusal(
            "RNA_QC_PAIRED_SUPPORT_BELOW_128",
            details={"donor": donor, "visit": visit, "eligible": len(candidates)},
        )
    chosen = sorted(
        candidates,
        key=lambda index: (
            hashlib.sha256(
                f"{CELL_SALT}\0{donor}\0{visit}\0{barcodes[index]}".encode()
            ).hexdigest(),
            barcodes[index].encode(),
        ),
    )[:count]
    return sorted(chosen)


def _complete_matrix_audit(
    value: Any, expected_shape: tuple[int, int], assay: str
) -> bool:
    if not isinstance(value, dict):
        return False
    expected_reduction_kind = {
        "gex": "gex_qc_and_panel",
        "cite": "cite_selected_panel",
    }.get(assay)
    if expected_reduction_kind is None:
        return False
    declared = value.get("declared_entries")
    raw = value.get("raw_entries_seen")
    yielded = value.get("raw_entries_yielded")
    peak_batch = value.get("peak_in_memory_entry_batch")
    reduction_entries = value.get("reduction_entries_accumulated")
    reduction_duplicates = value.get("reduction_duplicate_coordinates")
    authorized = value.get("authorized_barcodes")
    materialized = value.get("selected_panel_entries_materialized")
    reduction_kind = value.get("reduction_kind")
    packed_bytes = value.get("packed_detection_bitset_bytes")
    selected_seen_bytes = value.get("selected_coordinate_seen_bytes")
    reduction_specific = False
    if reduction_kind == "gex_qc_and_panel" and isinstance(authorized, int):
        logical_bits = int(authorized) * expected_shape[0]
        expected_bitset_bytes = (logical_bits + 7) // 8
        reduction_specific = (
            packed_bytes == expected_bitset_bytes
            and value.get("packed_detection_bitset_rows") == authorized
            and value.get("packed_detection_bitset_feature_rows")
            == expected_shape[0]
            and value.get("packed_detection_bitset_logical_bits") == logical_bits
            and selected_seen_bytes is None
        )
    elif reduction_kind == "cite_selected_panel" and isinstance(authorized, int):
        reduction_specific = (
            selected_seen_bytes == int(authorized) * MARKER_COUNT
            and packed_bytes is None
        )
    return bool(
        value.get("parser_started") is True
        and value.get("expected_shape") == list(expected_shape)
        and value.get("rows") == expected_shape[0]
        and value.get("columns") == expected_shape[1]
        and isinstance(declared, int)
        and declared >= 0
        and raw == declared
        and yielded == raw
        and value.get("parse_completed") is True
        and value.get("entry_iteration_completed") is True
        and value.get("all_body_entries_validated") is True
        and value.get("gzip_eof_verified") is True
        and value.get("declared_entry_count_validated") is True
        and value.get("matrix_total_within_int64") is True
        and isinstance(value.get("matrix_total_count"), int)
        and 0 <= value.get("matrix_total_count") <= np.iinfo(np.int64).max
        and value.get("sorted_independent_duplicate_accumulation") is True
        and value.get("full_dense_matrix_materialized") is False
        and value.get("full_coordinate_set_materialized") is False
        and value.get("temporary_database_created") is False
        and value.get("parser_backend") == "validated_single_pass_gzip_stream"
        and value.get("aggregation_backend")
        == "single_pass_authorized_exact_integer_reduction"
        and reduction_kind == expected_reduction_kind
        and value.get("temporary_storage_bytes") == 0
        and peak_batch == 1
        and isinstance(authorized, int)
        and MINIMUM_CELLS <= authorized
        and isinstance(reduction_entries, int)
        and 0 <= reduction_entries <= raw
        and isinstance(reduction_duplicates, int)
        and 0 <= reduction_duplicates <= reduction_entries
        and isinstance(materialized, int)
        and 0 <= materialized <= reduction_entries
        and reduction_specific
        and value.get("reduction_completed") is True
    )


def _checked_count_sum(current: np.int64, increment: int) -> np.int64:
    value = int(current)
    if increment > np.iinfo(np.int64).max - value:
        raise ProtocolRefusal("MATRIX_MARKET_REDUCTION_SUM_OVERFLOW")
    return np.int64(value + increment)


def _reduce_gex_matrix(
    path: Path,
    donor: str,
    visit_name: str,
    axes: dict[str, Any],
    matrix_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = axes["gex_normalized_barcodes"]
    intersection = set(axes["intersection"])
    authorized = [index for index, barcode in enumerate(normalized) if barcode in intersection]
    output = {column: index for index, column in enumerate(authorized)}
    totals = np.zeros(len(authorized), dtype=np.int64)
    detected = np.zeros(len(authorized), dtype=np.int64)
    mitochondrial = np.zeros(len(authorized), dtype=np.int64)
    panel_counts = np.zeros((len(authorized), MARKER_COUNT), dtype=np.int64)
    feature_rows = len(axes["gex_features"])
    logical_detection_bits = len(authorized) * feature_rows
    detected_coordinates = bytearray((logical_detection_bits + 7) // 8)
    panel_rows = {int(row): index for index, row in enumerate(axes["rna_rows"])}
    symbols = [line.split("\t")[1] for line in axes["gex_features"]]
    gene_expression_rows = np.asarray(axes["gene_expression_rows"], dtype=bool)
    mitochondrial_rows = np.asarray(
        [
            symbol.startswith("MT-") and gene_expression_rows[index]
            for index, symbol in enumerate(symbols)
        ],
        dtype=bool,
    )
    with _matrix_entries(
        path,
        (len(axes["gex_features"]), len(axes["gex_barcodes"])),
        matrix_audit,
    ) as (entries, audit):
        materialized = 0
        reduction_entries = 0
        reduction_duplicates = 0
        for row, column, value in entries:
            if not gene_expression_rows[row]:
                continue
            cell = output.get(column)
            if cell is None:
                continue
            reduction_entries += 1
            totals[cell] = _checked_count_sum(totals[cell], value)
            if value > 0:
                bit_index = cell * feature_rows + row
                byte, shift = divmod(bit_index, 8)
                bit = 1 << shift
                if detected_coordinates[byte] & bit:
                    reduction_duplicates += 1
                else:
                    detected_coordinates[byte] |= bit
                    detected[cell] += 1
            if mitochondrial_rows[row]:
                mitochondrial[cell] = _checked_count_sum(
                    mitochondrial[cell], value
                )
            marker = panel_rows.get(row)
            if marker is not None:
                panel_counts[cell, marker] = _checked_count_sum(
                    panel_counts[cell, marker], value
                )
                materialized += 1
        audit["authorized_barcodes"] = len(authorized)
        audit["selected_panel_entries_materialized"] = materialized
        audit["reduction_kind"] = "gex_qc_and_panel"
        audit["aggregation_backend"] = (
            "single_pass_authorized_exact_integer_reduction"
        )
        audit["reduction_entries_accumulated"] = reduction_entries
        audit["reduction_duplicate_coordinates"] = reduction_duplicates
        audit["packed_detection_bitset_rows"] = len(authorized)
        audit["packed_detection_bitset_feature_rows"] = feature_rows
        audit["packed_detection_bitset_logical_bits"] = logical_detection_bits
        audit["packed_detection_bitset_bytes"] = len(detected_coordinates)
        audit["reduction_completed"] = True
    fraction = np.divide(
        mitochondrial,
        totals,
        out=np.ones(len(totals), dtype=float),
        where=totals > 0,
    )
    eligible = np.flatnonzero(
        (detected >= MINIMUM_DETECTED_GENES)
        & (fraction <= MAXIMUM_MITOCHONDRIAL_FRACTION)
        & (totals <= MAXIMUM_RNA_UMIS)
    )
    intersection_barcodes = [normalized[index] for index in authorized]
    selected = _selected_cell_indices(
        donor, visit_name, intersection_barcodes, eligible
    )
    selected_array = np.asarray(selected, dtype=np.int64)
    selected_counts = panel_counts[selected_array]
    states = (selected_counts > 0).astype(np.uint8)
    barcodes = [intersection_barcodes[index] for index in selected]
    return {
        "barcodes": barcodes,
        "states": states,
        "profile": states.mean(axis=0),
        "cell_count": len(barcodes),
        "matrix_audit": audit,
        "eligible_barcodes": len(eligible),
        "selected_barcode_sha256": _axis_sha256(barcodes),
        "state_sha256": _array_sha256(states),
    }


def _midrank_adt(
    counts: np.ndarray,
    barcodes: list[str],
    donor: str,
    visit_name: str,
) -> np.ndarray:
    values = np.asarray(counts, dtype=np.int64)
    if values.shape != (len(barcodes), MARKER_COUNT) or len(barcodes) % 2:
        raise ValueError("ADT panel must have an even selected-cell axis")
    states = np.zeros(values.shape, dtype=np.uint8)
    for marker in range(MARKER_COUNT):
        order = sorted(
            range(len(barcodes)),
            key=lambda index: (
                int(values[index, marker]),
                hashlib.sha256(
                    f"{ADT_TIE_SALT}\0{donor}\0{visit_name}\0{marker}\0{barcodes[index]}".encode()
                ).hexdigest(),
                barcodes[index].encode(),
            ),
        )
        states[np.asarray(order[len(order) // 2 :], dtype=np.int64), marker] = 1
    if not np.all(states.sum(axis=0) == len(barcodes) // 2):
        raise AssertionError("ADT midrank did not create equal halves")
    return states


def _reduce_cite_matrix(
    path: Path,
    donor: str,
    visit_name: str,
    axes: dict[str, Any],
    selected_barcodes: list[str],
    matrix_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    barcode_lookup = {barcode: index for index, barcode in enumerate(axes["cite_barcodes"])}
    if len(selected_barcodes) != len(set(selected_barcodes)) or any(
        barcode not in barcode_lookup for barcode in selected_barcodes
    ):
        raise ProtocolRefusal("SELECTED_BARCODE_NOT_EXACT_IN_CITE_AXIS")
    selected_columns = {
        barcode_lookup[barcode]: index for index, barcode in enumerate(selected_barcodes)
    }
    panel_rows = {int(row): index for index, row in enumerate(axes["adt_rows"])}
    counts = np.zeros((len(selected_barcodes), MARKER_COUNT), dtype=np.int64)
    seen_coordinates = np.zeros(counts.shape, dtype=bool)
    with _matrix_entries(
        path,
        (len(axes["cite_features"]), len(axes["cite_barcodes"])),
        matrix_audit,
    ) as (entries, audit):
        materialized = 0
        reduction_duplicates = 0
        for row, column, value in entries:
            cell = selected_columns.get(column)
            marker = panel_rows.get(row)
            if cell is not None and marker is not None:
                if seen_coordinates[cell, marker]:
                    reduction_duplicates += 1
                else:
                    seen_coordinates[cell, marker] = True
                counts[cell, marker] = _checked_count_sum(
                    counts[cell, marker], value
                )
                materialized += 1
        audit["authorized_barcodes"] = len(selected_barcodes)
        audit["selected_panel_entries_materialized"] = materialized
        audit["reduction_kind"] = "cite_selected_panel"
        audit["aggregation_backend"] = (
            "single_pass_authorized_exact_integer_reduction"
        )
        audit["reduction_entries_accumulated"] = materialized
        audit["reduction_duplicate_coordinates"] = reduction_duplicates
        audit["selected_coordinate_seen_bytes"] = int(seen_coordinates.nbytes)
        audit["reduction_completed"] = True
    states = _midrank_adt(counts, selected_barcodes, donor, visit_name)
    return {
        "states": states,
        "profile": np.log1p(counts).mean(axis=0),
        "cell_count": len(selected_barcodes),
        "matrix_audit": audit,
        "state_sha256": _array_sha256(states),
        "count_panel_sha256": _array_sha256(counts),
    }


def _destroyed_adt(
    states: np.ndarray,
    barcodes: list[str],
    donor: str,
    visit_name: str,
) -> np.ndarray:
    values = np.asarray(states, dtype=np.uint8)
    order = np.asarray(
        sorted(
            range(len(barcodes)),
            key=lambda index: (
                hashlib.sha256(
                    f"{CELL_SALT}\0{donor}\0{visit_name}\0{barcodes[index]}".encode()
                ).hexdigest(),
                barcodes[index].encode(),
            ),
        ),
        dtype=np.int64,
    )
    destroyed = np.empty_like(values)
    destroyed[order] = values[np.roll(order, 1)]
    if not np.array_equal(destroyed.sum(axis=0), values.sum(axis=0)):
        raise AssertionError("destroyed link changed an ADT margin")
    if sorted(map(tuple, destroyed.tolist())) != sorted(map(tuple, values.tolist())):
        raise AssertionError("destroyed link changed complete ADT vectors")
    return destroyed


def _joint_tables(rna: np.ndarray, adt: np.ndarray) -> np.ndarray:
    first = np.asarray(rna, dtype=np.uint8)
    second = np.asarray(adt, dtype=np.uint8)
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != MARKER_COUNT:
        raise ValueError("binary panels have incompatible shapes")
    output = np.zeros((MARKER_COUNT, MARKER_COUNT, 2, 2), dtype=np.int64)
    for row, column in np.ndindex((MARKER_COUNT, MARKER_COUNT)):
        output[row, column] = np.bincount(
            2 * first[:, row] + second[:, column], minlength=4
        ).reshape(2, 2)
    return output


def _margin_support(tables: np.ndarray) -> np.ndarray:
    values = np.asarray(tables)
    rows = values.sum(axis=-1)
    columns = values.sum(axis=-2)
    total = values.sum(axis=(-2, -1))
    lower = np.maximum(0, rows[..., 0] + columns[..., 0] - total)
    upper = np.minimum(rows[..., 0], columns[..., 0])
    return upper > lower


def _recipient_margin_support(rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    row = np.asarray(rows, dtype=float)
    column = np.asarray(columns, dtype=float)
    if row.shape != column.shape or row.shape[-1] != 2:
        raise ValueError("row and column margins must have matching binary axes")
    total = row.sum(axis=-1)
    if not np.allclose(total, column.sum(axis=-1), atol=1e-10, rtol=0.0):
        raise ValueError("row and column margins have different totals")
    lower = np.maximum(0.0, row[..., 0] + column[..., 0] - total)
    upper = np.minimum(row[..., 0], column[..., 0])
    return upper > lower


def _training_only_mask(tables: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(tables, dtype=np.int64)
    if values.ndim != 6 or values.shape[1:4] != (
        2,
        MARKER_COUNT,
        MARKER_COUNT,
    ):
        raise ValueError("training tables have the wrong longitudinal shape")
    support = _margin_support(values)
    mask = np.ones((MARKER_COUNT, MARKER_COUNT), dtype=bool)
    visit_support_counts = []
    interior_counts = []
    pooled_positive_counts = []
    for visit in range(2):
        visit_support = support[:, visit]
        support_count = visit_support.sum(axis=0)
        visit_support_counts.append(support_count)
        mask &= support_count >= 2
        observed_sum = np.zeros((MARKER_COUNT, MARKER_COUNT), dtype=float)
        lower_sum = np.zeros_like(observed_sum)
        upper_sum = np.zeros_like(observed_sum)
        for donor, first, second in np.argwhere(visit_support):
            counts, cells, _ = _fixed_margin_support(
                values[donor, visit, first, second]
            )
            observed_sum[first, second] += counts[0, 0]
            lower_sum[first, second] += cells[0, 0]
            upper_sum[first, second] += cells[-1, 0]
        interior = (observed_sum > lower_sum) & (observed_sum < upper_sum)
        interior_counts.append(interior)
        mask &= interior
        pooled = values[:, visit].sum(axis=0)
        pooled_positive = np.all(pooled > 0, axis=(-2, -1))
        pooled_positive_counts.append(pooled_positive)
        mask &= pooled_positive
    distinct = np.count_nonzero(np.any(support, axis=1), axis=0)
    mask &= distinct >= 3
    details = {
        "training_donors": values.shape[0],
        "coordinate_count": int(mask.sum()),
        "mask_sha256": _array_sha256(mask.astype(np.uint8)),
        "minimum_pre_support_count": int(np.min(visit_support_counts[0][mask]))
        if np.any(mask)
        else 0,
        "minimum_post_support_count": int(np.min(visit_support_counts[1][mask]))
        if np.any(mask)
        else 0,
        "minimum_distinct_donor_count": int(np.min(distinct[mask]))
        if np.any(mask)
        else 0,
        "pre_interior_count": int(np.count_nonzero(interior_counts[0])),
        "post_interior_count": int(np.count_nonzero(interior_counts[1])),
        "pre_pooled_four_positive_count": int(
            np.count_nonzero(pooled_positive_counts[0])
        ),
        "post_pooled_four_positive_count": int(
            np.count_nonzero(pooled_positive_counts[1])
        ),
    }
    return mask, details


def _fold_mask(
    training_tables: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mask, details = _training_only_mask(training_tables)
    recipient = _recipient_margin_support(rows, columns)
    scored = recipient & mask[None, ...]
    counts = [int(np.count_nonzero(scored[visit])) for visit in range(2)]
    details.update(
        {
            "recipient_margin_supported_counts": [
                int(np.count_nonzero(recipient[visit])) for visit in range(2)
            ],
            "scored_coordinate_counts": counts,
            "scored_mask_sha256": _array_sha256(scored.astype(np.uint8)),
            "validation_association_used_for_mask": False,
        }
    )
    if min(counts) < MINIMUM_COORDINATES:
        raise ProtocolRefusal(
            "FOLD_RECIPIENT_MASK_BELOW_256",
            details={"counts": counts},
        )
    return mask, scored, details


def _knn_laplacian(profiles: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    values = np.asarray(profiles, dtype=float).reshape(-1, MARKER_COUNT).T
    scale = values.std(axis=1, ddof=1)
    if not np.isfinite(values).all() or not np.isfinite(scale).all() or np.any(scale == 0):
        raise CouplingEstimationRefusal("source marker profile has zero variance")
    standardized = (values - values.mean(axis=1, keepdims=True)) / scale[:, None]
    edges: set[tuple[int, int]] = set()
    neighbors: list[list[int]] = []
    for marker in range(MARKER_COUNT):
        candidates = np.asarray(
            [candidate for candidate in range(MARKER_COUNT) if candidate != marker]
        )
        distances = np.linalg.norm(
            standardized[candidates] - standardized[marker], axis=1
        )
        selected = candidates[np.lexsort((candidates, distances))][
            :GRAPH_NEIGHBORS
        ]
        neighbors.append([int(value) for value in selected])
        edges.update(tuple(sorted((marker, int(value)))) for value in selected)
    laplacian = np.zeros((MARKER_COUNT, MARKER_COUNT), dtype=float)
    for left, right in sorted(edges):
        laplacian[left, left] += 1.0
        laplacian[right, right] += 1.0
        laplacian[left, right] -= 1.0
        laplacian[right, left] -= 1.0
    return laplacian, neighbors


@dataclass(frozen=True)
class PrimaryConfig:
    heterogeneity_penalty: float
    population_ridge: float
    graph_penalty: float
    baseline_transport: float
    visit_change_transport: float


@dataclass(frozen=True)
class ResidualConfig:
    family: str
    baseline_transport: float
    visit_change_transport: float


@dataclass(frozen=True)
class TransportConfig:
    baseline_transport: float
    visit_change_transport: float


def _fit_primary(
    tables: np.ndarray,
    rna_profiles: np.ndarray,
    adt_profiles: np.ndarray,
    mask: np.ndarray,
    configuration: PrimaryConfig,
) -> dict[str, Any]:
    values = np.asarray(tables, dtype=np.int64)
    support = _margin_support(values) & np.asarray(mask, dtype=bool)[None, None, ...]
    if configuration.graph_penalty == 0.0:
        rna_laplacian = np.zeros((MARKER_COUNT, MARKER_COUNT), dtype=float)
        adt_laplacian = np.zeros((MARKER_COUNT, MARKER_COUNT), dtype=float)
        rna_neighbors = None
        adt_neighbors = None
    else:
        rna_laplacian, rna_neighbors = _knn_laplacian(rna_profiles)
        adt_laplacian, adt_neighbors = _knn_laplacian(adt_profiles)
    fit = fit_longitudinal_conditional_log_odds(
        values,
        rna_laplacian,
        adt_laplacian,
        support_mask=support,
        heterogeneity_penalty=configuration.heterogeneity_penalty,
        population_ridge=configuration.population_ridge,
        graph_penalty=configuration.graph_penalty,
    )
    return {
        "population_mean": fit.population_mean,
        "population_change": fit.population_change,
        "fit_certificate": {
            "gradient_norm": fit.gradient_norm,
            "scaled_gradient_norm": fit.scaled_gradient_norm,
            "iterations": fit.iterations,
            "function_evaluations": fit.function_evaluations,
            "optimizer": fit.optimizer,
            "maximum_baseline_constraint_error": fit.maximum_baseline_constraint_error,
            "maximum_change_constraint_error": fit.maximum_change_constraint_error,
            "informative_table_count": fit.informative_table_count,
            "retained_coordinate_count": fit.retained_coordinate_count,
            "rna_laplacian_sha256": _array_sha256(rna_laplacian),
            "adt_laplacian_sha256": _array_sha256(adt_laplacian),
            "rna_neighbors": rna_neighbors,
            "adt_neighbors": adt_neighbors,
            "passes": True,
        },
    }


def _fit_visit_agnostic(
    tables: np.ndarray,
    rna_profiles: np.ndarray,
    adt_profiles: np.ndarray,
    mask: np.ndarray,
    configuration: PrimaryConfig,
) -> dict[str, Any]:
    values = np.asarray(tables, dtype=np.int64)
    support = _margin_support(values) & np.asarray(mask, dtype=bool)[None, None, ...]
    if configuration.graph_penalty == 0.0:
        rna_laplacian = np.zeros((MARKER_COUNT, MARKER_COUNT), dtype=float)
        adt_laplacian = np.zeros((MARKER_COUNT, MARKER_COUNT), dtype=float)
        rna_neighbors = None
        adt_neighbors = None
    else:
        rna_laplacian, rna_neighbors = _knn_laplacian(rna_profiles)
        adt_laplacian, adt_neighbors = _knn_laplacian(adt_profiles)
    fit = fit_visit_agnostic_conditional_log_odds(
        values,
        rna_laplacian,
        adt_laplacian,
        support_mask=support,
        heterogeneity_penalty=configuration.heterogeneity_penalty,
        population_ridge=configuration.population_ridge,
        graph_penalty=configuration.graph_penalty,
    )
    return {
        "population_mean": fit.population_mean,
        "population_change": fit.population_change,
        "fit_certificate": {
            "gradient_norm": fit.gradient_norm,
            "scaled_gradient_norm": fit.scaled_gradient_norm,
            "iterations": fit.iterations,
            "function_evaluations": fit.function_evaluations,
            "optimizer": fit.optimizer,
            "maximum_baseline_constraint_error": fit.maximum_baseline_constraint_error,
            "maximum_change_constraint_error": fit.maximum_change_constraint_error,
            "informative_table_count": fit.informative_table_count,
            "retained_coordinate_count": fit.retained_coordinate_count,
            "rna_laplacian_sha256": _array_sha256(rna_laplacian),
            "adt_laplacian_sha256": _array_sha256(adt_laplacian),
            "rna_neighbors": rna_neighbors,
            "adt_neighbors": adt_neighbors,
            "visit_change_constrained_to_zero": True,
            "passes": True,
        },
    }


def _conditional_missing_tables(tables: np.ndarray, support: np.ndarray) -> np.ndarray:
    return table_engine._conditional_missing_tables(tables, support)


def _fit_common(
    tables: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(tables, dtype=np.int64)
    selected = np.asarray(mask, dtype=bool)
    log_odds = np.zeros((2, MARKER_COUNT, MARKER_COUNT), dtype=float)
    certificates = []
    for visit in range(2):
        support = _margin_support(values[:, visit]) & selected[None, ...]
        encoded = _conditional_missing_tables(values[:, visit], support)[:, selected]
        fit = fit_common_effect_conditional_log_odds(
            encoded,
            minimum_informative_donors=2,
            tolerance=1e-10,
        )
        log_odds[visit][selected] = fit.log_odds
        certificates.append(
            {
                "gradient_norm": fit.gradient_norm,
                "scaled_gradient_norm": fit.scaled_gradient_norm,
                "minimum_data_precision": float(np.min(fit.data_precision)),
                "minimum_support_count": int(np.min(fit.support_count)),
                "maximum_root_iterations": int(np.max(fit.root_iterations)),
            }
        )
    return {
        "population_mean": 0.5 * (log_odds[0] + log_odds[1]),
        "population_change": log_odds[1] - log_odds[0],
        "fit_certificate": {"visits": certificates, "passes": True},
    }


def _table_log_odds(table: np.ndarray) -> float:
    values = np.asarray(table, dtype=float)
    if values.shape != (2, 2) or np.any(values <= 0.0):
        raise CouplingEstimationRefusal("fixed-interaction table lacks four positive cells")
    return float(
        math.log(values[0, 0])
        + math.log(values[1, 1])
        - math.log(values[0, 1])
        - math.log(values[1, 0])
    )


def _fixed_interaction_table(
    log_odds: float,
    rows: np.ndarray,
    columns: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    return common._fixed_interaction_table(log_odds, rows, columns)


def _fit_poisson(tables: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    values = np.asarray(tables, dtype=np.int64)
    selected = np.asarray(mask, dtype=bool)
    log_odds = np.zeros((2, MARKER_COUNT, MARKER_COUNT), dtype=float)
    maximum_cell_error = 0.0
    maximum_margin_error = 0.0
    maximum_log_odds_error = 0.0
    for visit in range(2):
        pooled = values[:, visit].sum(axis=0)
        if np.any(pooled[selected] <= 0):
            raise CouplingEstimationRefusal("pooled Poisson table contains zero cells")
        for index in zip(*np.nonzero(selected)):
            source = pooled[index]
            coordinate = _table_log_odds(source)
            log_odds[(visit, *index)] = coordinate
            reconstructed, certificate = _fixed_interaction_table(
                coordinate, source.sum(axis=1), source.sum(axis=0)
            )
            maximum_cell_error = max(
                maximum_cell_error,
                float(np.max(np.abs(reconstructed - source))) / float(source.sum()),
            )
            maximum_margin_error = max(
                maximum_margin_error,
                certificate["maximum_row_margin_error"],
                certificate["maximum_column_margin_error"],
            )
            maximum_log_odds_error = max(
                maximum_log_odds_error, certificate["absolute_log_odds_error"]
            )
    if maximum_cell_error > 1e-8:
        raise CouplingEstimationRefusal("pooled Poisson reconstruction certificate failed")
    return {
        "population_mean": 0.5 * (log_odds[0] + log_odds[1]),
        "population_change": log_odds[1] - log_odds[0],
        "fit_certificate": {
            "family": "visit-specific pooled saturated Poisson fixed interaction",
            "pooled_every_training_donor_including_degenerate_margins": True,
            "conditional_noncentral_hypergeometric_reconstruction": False,
            "maximum_normalized_source_cell_error": maximum_cell_error,
            "maximum_source_margin_error": maximum_margin_error,
            "maximum_source_log_odds_error": maximum_log_odds_error,
            "passes": True,
        },
    }


def _fit_residual(
    tables: np.ndarray,
    mask: np.ndarray,
    family: str,
) -> dict[str, Any]:
    values = np.asarray(tables, dtype=np.int64)
    selected = np.asarray(mask, dtype=bool)
    support = _margin_support(values) & selected[None, None, ...]
    statistic = (
        signed_pearson_coordinate
        if family == "pearson"
        else signed_deviance_coordinate
        if family == "root_deviance"
        else None
    )
    if statistic is None:
        raise ValueError("unknown residual family")
    coordinates = np.zeros((2, MARKER_COUNT, MARKER_COUNT), dtype=float)
    for visit in range(2):
        for index in zip(*np.nonzero(selected)):
            observed = []
            for donor in range(values.shape[0]):
                if support[(donor, visit, *index)]:
                    table = values[(donor, visit, *index)]
                    observed.append(statistic(table) / math.sqrt(float(table.sum())))
            if len(observed) < 2:
                raise CouplingEstimationRefusal("residual has too few training donors")
            coordinates[(visit, *index)] = float(np.mean(observed))
    return {
        "population_mean": 0.5 * (coordinates[0] + coordinates[1]),
        "population_change": coordinates[1] - coordinates[0],
        "fit_certificate": {
            "family": family,
            "raw_statistic_no_null_centering": True,
            "finite_on_mask": bool(np.isfinite(coordinates[:, selected]).all()),
            "passes": True,
        },
    }


def _transport_coordinates(
    model: dict[str, Any],
    baseline: float,
    change: float,
) -> np.ndarray:
    mean = np.asarray(model["population_mean"], dtype=float)
    delta = np.asarray(model["population_change"], dtype=float)
    return np.stack(
        [baseline * mean - 0.5 * change * delta, baseline * mean + 0.5 * change * delta]
    )


def _independence(rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    row = np.asarray(rows, dtype=float)
    column = np.asarray(columns, dtype=float)
    total = row.sum(axis=-1)
    return np.einsum("...i,...j->...ij", row, column) / total[..., None, None]


def _predict_nch(
    model: dict[str, Any],
    rows: np.ndarray,
    columns: np.ndarray,
    configuration: TransportConfig | PrimaryConfig,
    scored: np.ndarray,
) -> np.ndarray:
    coordinates = _transport_coordinates(
        model,
        configuration.baseline_transport,
        configuration.visit_change_transport,
    )
    prediction = _independence(rows, columns)
    for visit, first, second in np.argwhere(scored):
        prediction[visit, first, second] = expected_binary_table_from_log_odds(
            float(coordinates[visit, first, second]),
            rows[visit, first, second],
            columns[visit, first, second],
        )
    return prediction


def _predict_poisson(
    model: dict[str, Any],
    rows: np.ndarray,
    columns: np.ndarray,
    configuration: TransportConfig,
    scored: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    coordinates = _transport_coordinates(
        model,
        configuration.baseline_transport,
        configuration.visit_change_transport,
    )
    prediction = _independence(rows, columns)
    maximum_margin_error = 0.0
    maximum_log_odds_error = 0.0
    for visit, first, second in np.argwhere(scored):
        table, certificate = _fixed_interaction_table(
            float(coordinates[visit, first, second]),
            rows[visit, first, second],
            columns[visit, first, second],
        )
        prediction[visit, first, second] = table
        maximum_margin_error = max(
            maximum_margin_error,
            certificate["maximum_row_margin_error"],
            certificate["maximum_column_margin_error"],
        )
        maximum_log_odds_error = max(
            maximum_log_odds_error, certificate["absolute_log_odds_error"]
        )
    return prediction, {
        "maximum_margin_error": maximum_margin_error,
        "maximum_log_odds_error": maximum_log_odds_error,
        "conditional_noncentral_hypergeometric_reconstruction": False,
    }


def _predict_residual(
    model: dict[str, Any],
    rows: np.ndarray,
    columns: np.ndarray,
    configuration: ResidualConfig,
    scored: np.ndarray,
) -> np.ndarray:
    normalized = _transport_coordinates(
        model,
        configuration.baseline_transport,
        configuration.visit_change_transport,
    )
    prediction = _independence(rows, columns)
    for visit, first, second in np.argwhere(scored):
        total = float(rows[visit, first, second].sum())
        prediction[visit, first, second] = table_engine._residual_table(
            float(normalized[visit, first, second]) * math.sqrt(total),
            rows[visit, first, second],
            columns[visit, first, second],
            configuration.family,
        )
    return prediction


def _loss(
    observed: np.ndarray,
    predicted: np.ndarray,
    scored: np.ndarray,
) -> float:
    truth = np.asarray(observed, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    mask = np.asarray(scored, dtype=bool)
    visit_losses = []
    for visit in range(2):
        if np.count_nonzero(mask[visit]) < MINIMUM_COORDINATES:
            raise CouplingEstimationRefusal("recipient visit has too few scored coordinates")
        selected_truth = truth[visit][mask[visit]]
        selected_estimate = estimate[visit][mask[visit]]
        if not np.allclose(
            selected_truth.sum(axis=-1), selected_estimate.sum(axis=-1), atol=1e-8
        ) or not np.allclose(
            selected_truth.sum(axis=-2), selected_estimate.sum(axis=-2), atol=1e-8
        ):
            raise CouplingEstimationRefusal("prediction changed recipient margins")
        positive = selected_truth > 0
        if np.any(selected_estimate[positive] <= 0.0) or not np.isfinite(
            selected_estimate
        ).all():
            raise CouplingEstimationRefusal("prediction assigns invalid mass")
        terms = np.zeros_like(selected_truth)
        terms[positive] = selected_truth[positive] * np.log(
            selected_truth[positive] / selected_estimate[positive]
        )
        totals = selected_truth.sum(axis=(-2, -1))
        visit_losses.append(float(np.mean(2.0 * terms.sum(axis=(-2, -1)) / totals)))
    return float(np.mean(visit_losses))


def _select_complete(
    order: list[Any],
    losses: dict[Any, np.ndarray],
    label: str,
) -> tuple[Any, np.ndarray]:
    complete = [candidate for candidate in order if np.isfinite(losses[candidate]).all()]
    if not complete:
        raise ProtocolRefusal(f"NO_COMPLETE_{label.upper()}_CANDIDATE")
    selected = min(complete, key=lambda candidate: (float(losses[candidate].mean()), order.index(candidate)))
    return selected, losses[selected]


def _bootstrap_interval(delta: np.ndarray) -> tuple[float, float]:
    values = np.asarray(delta, dtype=float)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    draws = values[indices].mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975], method="linear")
    return float(lower), float(upper)


def _comparison(
    primary: np.ndarray,
    comparator: np.ndarray,
    label: str,
    stage: str,
) -> dict[str, Any]:
    first = np.asarray(primary, dtype=float)
    second = np.asarray(comparator, dtype=float)
    if first.shape != second.shape or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("comparison losses must be finite and matched")
    delta = first - second
    interval = _bootstrap_interval(delta)
    relative = float((second.mean() - first.mean()) / second.mean())
    favorable = int(np.count_nonzero(delta < 0.0))
    if stage == "source" and label in {"residual", "pooled_poisson", "destroyed"}:
        passes = relative >= 0.05 and interval[1] < 0.0 and favorable >= 6
    elif stage == "source" and label == "independence":
        passes = relative >= 0.05
    elif stage == "source" and label == "common_effect_cmle":
        passes = float(first.mean()) < float(second.mean())
    elif stage == "held" and label in {"residual", "pooled_poisson", "destroyed"}:
        passes = relative >= 0.05 and interval[1] < 0.0 and favorable == 6
    elif stage == "held" and label == "common_effect_cmle":
        passes = float(first.mean()) < float(second.mean())
    else:
        raise ValueError("unknown frozen comparison role")
    return {
        "label": label,
        "stage": stage,
        "primary_mean_loss": float(first.mean()),
        "comparator_mean_loss": float(second.mean()),
        "relative_reduction": relative,
        "paired_difference_mean": float(delta.mean()),
        "paired_bootstrap_95": list(interval),
        "favorable_physical_donors": favorable,
        "physical_donor_count": len(first),
        "passes_frozen_requirement": bool(passes),
    }


def _candidate_evaluations(
    order: list[Any], losses: dict[Any, np.ndarray]
) -> list[dict[str, Any]]:
    output = []
    for candidate in order:
        values = np.asarray(losses[candidate], dtype=float)
        complete = bool(np.isfinite(values).all())
        record = {
            "configuration": asdict(candidate)
            if hasattr(candidate, "__dataclass_fields__")
            else candidate,
            "complete": complete,
            "fold_losses": [float(value) if np.isfinite(value) else None for value in values],
        }
        if complete:
            record["mean_physical_donor_loss"] = float(values.mean())
        output.append(record)
    return output


def _source_arrays(
    records: dict[str, dict[str, Any]],
    donors: list[str],
    key: str = "tables",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tables = np.asarray([records[donor][key] for donor in donors], dtype=np.int64)
    rna = np.asarray([records[donor]["rna_profiles"] for donor in donors], dtype=float)
    adt = np.asarray([records[donor]["adt_profiles"] for donor in donors], dtype=float)
    return tables, rna, adt


def _evaluate_source(
    records: dict[str, dict[str, Any]],
    source_axis: list[str],
) -> dict[str, Any]:
    if len(source_axis) != 7 or set(source_axis) != set(records):
        raise ValueError("source evaluation requires the seven frozen donors")
    primary_order = [
        PrimaryConfig(heterogeneity, ridge, graph, baseline, change)
        for heterogeneity, ridge, graph, baseline, change in product(
            HETEROGENEITY_GRID,
            RIDGE_GRID,
            GRAPH_GRID,
            TRANSPORT_GRID,
            TRANSPORT_GRID,
        )
    ]
    visit_agnostic_order = [
        PrimaryConfig(heterogeneity, ridge, graph, baseline, 0.0)
        for heterogeneity, ridge, graph, baseline in product(
            HETEROGENEITY_GRID,
            RIDGE_GRID,
            GRAPH_GRID,
            TRANSPORT_GRID,
        )
    ]
    residual_order = [
        ResidualConfig(family, baseline, change)
        for family, baseline, change in product(
            RESIDUAL_FAMILIES, TRANSPORT_GRID, TRANSPORT_GRID
        )
    ]
    transport_order = [
        TransportConfig(baseline, change)
        for baseline, change in product(TRANSPORT_GRID, TRANSPORT_GRID)
    ]
    primary_losses = {
        configuration: np.full(len(source_axis), np.nan)
        for configuration in primary_order
    }
    visit_agnostic_losses = {
        configuration: np.full(len(source_axis), np.nan)
        for configuration in visit_agnostic_order
    }
    residual_losses = {
        configuration: np.full(len(source_axis), np.nan)
        for configuration in residual_order
    }
    common_losses = {
        configuration: np.full(len(source_axis), np.nan)
        for configuration in transport_order
    }
    poisson_losses = {
        configuration: np.full(len(source_axis), np.nan)
        for configuration in transport_order
    }
    independence_losses = np.full(len(source_axis), np.nan)
    fold_masks: dict[str, Any] = {}
    fold_certificates: dict[str, Any] = {}
    refusals: list[dict[str, Any]] = []

    for held_index, held in enumerate(source_axis):
        training_axis = [donor for donor in source_axis if donor != held]
        tables, rna_profiles, adt_profiles = _source_arrays(records, training_axis)
        truth = np.asarray(records[held]["tables"], dtype=np.int64)
        rows = truth.sum(axis=-1)
        columns = truth.sum(axis=-2)
        training_mask, scored, diagnostics = _fold_mask(tables, rows, columns)
        diagnostics["training_donors"] = training_axis
        diagnostics["validation_donor"] = held
        fold_masks[held] = diagnostics
        primary_base: dict[tuple[float, float, float], dict[str, Any]] = {}
        for heterogeneity, ridge, graph in product(
            HETEROGENEITY_GRID, RIDGE_GRID, GRAPH_GRID
        ):
            base = PrimaryConfig(heterogeneity, ridge, graph, 1.0, 1.0)
            try:
                fit = _fit_primary(
                    tables, rna_profiles, adt_profiles, training_mask, base
                )
                primary_base[(heterogeneity, ridge, graph)] = fit
                fold_certificates[
                    f"{held}:{heterogeneity}:{ridge}:{graph}"
                ] = fit["fit_certificate"]
            except (
                ValueError,
                FloatingPointError,
                CouplingEstimationRefusal,
                np.linalg.LinAlgError,
            ) as error:
                refusals.append(
                    {
                        "fold": held,
                        "family": "primary",
                        "heterogeneity_penalty": heterogeneity,
                        "population_ridge": ridge,
                        "graph_penalty": graph,
                        "reason": str(error),
                    }
                )
                continue
            for baseline, change in product(TRANSPORT_GRID, TRANSPORT_GRID):
                configuration = PrimaryConfig(
                    heterogeneity, ridge, graph, baseline, change
                )
                try:
                    primary_losses[configuration][held_index] = _loss(
                        truth,
                        _predict_nch(fit, rows, columns, configuration, scored),
                        scored,
                    )
                except (
                    ValueError,
                    FloatingPointError,
                    CouplingEstimationRefusal,
                ) as error:
                    refusals.append(
                        {
                            "fold": held,
                            "family": "primary_prediction",
                            "configuration": asdict(configuration),
                            "reason": str(error),
                        }
                    )

        for heterogeneity, ridge, graph in product(
            HETEROGENEITY_GRID, RIDGE_GRID, GRAPH_GRID
        ):
            base = PrimaryConfig(heterogeneity, ridge, graph, 1.0, 0.0)
            try:
                fit = _fit_visit_agnostic(
                    tables, rna_profiles, adt_profiles, training_mask, base
                )
            except (
                ValueError,
                FloatingPointError,
                CouplingEstimationRefusal,
                np.linalg.LinAlgError,
            ) as error:
                refusals.append(
                    {
                        "fold": held,
                        "family": "visit_agnostic_primary",
                        "heterogeneity_penalty": heterogeneity,
                        "population_ridge": ridge,
                        "graph_penalty": graph,
                        "reason": str(error),
                    }
                )
                continue
            for baseline in TRANSPORT_GRID:
                configuration = PrimaryConfig(
                    heterogeneity, ridge, graph, baseline, 0.0
                )
                try:
                    visit_agnostic_losses[configuration][held_index] = _loss(
                        truth,
                        _predict_nch(fit, rows, columns, configuration, scored),
                        scored,
                    )
                except (
                    ValueError,
                    FloatingPointError,
                    CouplingEstimationRefusal,
                ) as error:
                    refusals.append(
                        {
                            "fold": held,
                            "family": "visit_agnostic_prediction",
                            "configuration": asdict(configuration),
                            "reason": str(error),
                        }
                    )
        common = _fit_common(tables, training_mask)
        poisson = _fit_poisson(tables, training_mask)
        for configuration in transport_order:
            try:
                common_losses[configuration][held_index] = _loss(
                    truth,
                    _predict_nch(common, rows, columns, configuration, scored),
                    scored,
                )
                poisson_prediction, certificate = _predict_poisson(
                    poisson, rows, columns, configuration, scored
                )
                if (
                    certificate["maximum_margin_error"] > 1e-10
                    or certificate["maximum_log_odds_error"] > 1e-8
                ):
                    raise CouplingEstimationRefusal(
                        "pooled Poisson prediction missed its certificate"
                    )
                poisson_losses[configuration][held_index] = _loss(
                    truth, poisson_prediction, scored
                )
            except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
                refusals.append(
                    {
                        "fold": held,
                        "family": "common_or_poisson_prediction",
                        "configuration": asdict(configuration),
                        "reason": str(error),
                    }
                )
        for family in RESIDUAL_FAMILIES:
            try:
                residual = _fit_residual(tables, training_mask, family)
            except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
                refusals.append(
                    {
                        "fold": held,
                        "family": "residual",
                        "residual_family": family,
                        "reason": str(error),
                    }
                )
                continue
            for baseline, change in product(TRANSPORT_GRID, TRANSPORT_GRID):
                configuration = ResidualConfig(family, baseline, change)
                try:
                    residual_losses[configuration][held_index] = _loss(
                        truth,
                        _predict_residual(
                            residual, rows, columns, configuration, scored
                        ),
                        scored,
                    )
                except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
                    refusals.append(
                        {
                            "fold": held,
                            "family": "residual_prediction",
                            "configuration": asdict(configuration),
                            "reason": str(error),
                        }
                    )
        independence_losses[held_index] = _loss(
            truth, _independence(rows, columns), scored
        )

    selected_primary, primary = _select_complete(
        primary_order, primary_losses, "primary"
    )
    selected_residual, residual = _select_complete(
        residual_order, residual_losses, "residual"
    )
    selected_visit_agnostic, visit_agnostic_losses_selected = _select_complete(
        visit_agnostic_order,
        visit_agnostic_losses,
        "visit_agnostic_primary",
    )
    selected_common, common_losses_selected = _select_complete(
        transport_order, common_losses, "common_effect"
    )
    selected_poisson, poisson_losses_selected = _select_complete(
        transport_order, poisson_losses, "pooled_poisson"
    )

    destroyed_losses = np.full(len(source_axis), np.nan)
    destroyed_certificates: dict[str, Any] = {}
    for held_index, held in enumerate(source_axis):
        training_axis = [donor for donor in source_axis if donor != held]
        destroyed_tables, rna_profiles, adt_profiles = _source_arrays(
            records, training_axis, key="destroyed_tables"
        )
        real_tables = _source_arrays(records, training_axis)[0]
        truth = np.asarray(records[held]["tables"], dtype=np.int64)
        rows = truth.sum(axis=-1)
        columns = truth.sum(axis=-2)
        training_mask, scored, diagnostics = _fold_mask(
            real_tables, rows, columns
        )
        if diagnostics["scored_mask_sha256"] != fold_masks[held]["scored_mask_sha256"]:
            raise AssertionError("destroyed refit did not reuse the real fold mask")
        base = PrimaryConfig(
            selected_primary.heterogeneity_penalty,
            selected_primary.population_ridge,
            selected_primary.graph_penalty,
            1.0,
            1.0,
        )
        fit = _fit_primary(
            destroyed_tables,
            rna_profiles,
            adt_profiles,
            training_mask,
            base,
        )
        destroyed_certificates[held] = fit["fit_certificate"]
        destroyed_losses[held_index] = _loss(
            truth,
            _predict_nch(fit, rows, columns, selected_primary, scored),
            scored,
        )

    all_tables, all_rna_profiles, all_adt_profiles = _source_arrays(
        records, source_axis
    )
    final_mask, final_diagnostics = _training_only_mask(all_tables)
    if int(final_mask.sum()) < MINIMUM_COORDINATES:
        raise ProtocolRefusal(
            "FINAL_SOURCE_MASK_BELOW_256",
            details={"count": int(final_mask.sum())},
        )
    per_source_counts = {}
    for donor in source_axis:
        support = np.asarray(records[donor]["support"], dtype=bool)
        counts = [
            int(np.count_nonzero(final_mask & support[visit])) for visit in range(2)
        ]
        if min(counts) < MINIMUM_COORDINATES:
            raise ProtocolRefusal(
                "FINAL_SOURCE_RECIPIENT_MASK_BELOW_256",
                details={"donor": donor, "counts": counts},
            )
        per_source_counts[donor] = counts
    final_diagnostics["source_visit_margin_intersection_counts"] = per_source_counts
    primary_base_configuration = PrimaryConfig(
        selected_primary.heterogeneity_penalty,
        selected_primary.population_ridge,
        selected_primary.graph_penalty,
        1.0,
        1.0,
    )
    final_primary = _fit_primary(
        all_tables,
        all_rna_profiles,
        all_adt_profiles,
        final_mask,
        primary_base_configuration,
    )
    final_visit_agnostic = _fit_visit_agnostic(
        all_tables,
        all_rna_profiles,
        all_adt_profiles,
        final_mask,
        PrimaryConfig(
            selected_visit_agnostic.heterogeneity_penalty,
            selected_visit_agnostic.population_ridge,
            selected_visit_agnostic.graph_penalty,
            1.0,
            0.0,
        ),
    )
    final_common = _fit_common(all_tables, final_mask)
    final_poisson = _fit_poisson(all_tables, final_mask)
    final_residual = _fit_residual(all_tables, final_mask, selected_residual.family)
    all_destroyed = _source_arrays(records, source_axis, key="destroyed_tables")[0]
    final_destroyed = _fit_primary(
        all_destroyed,
        all_rna_profiles,
        all_adt_profiles,
        final_mask,
        primary_base_configuration,
    )

    comparisons = {
        "selected_residual": _comparison(primary, residual, "residual", "source"),
        "pooled_saturated_poisson": _comparison(
            primary, poisson_losses_selected, "pooled_poisson", "source"
        ),
        "destroyed_link": _comparison(
            primary, destroyed_losses, "destroyed", "source"
        ),
        "common_effect_cmle": _comparison(
            primary, common_losses_selected, "common_effect_cmle", "source"
        ),
        "independence": _comparison(
            primary, independence_losses, "independence", "source"
        ),
    }
    passes = all(
        comparison["passes_frozen_requirement"]
        for comparison in comparisons.values()
    )
    models = {
        "final_mask": final_mask.astype(np.uint8).tolist(),
        "final_mask_sha256": _array_sha256(final_mask.astype(np.uint8)),
        "primary": {
            "kind": "paired_longitudinal_exact_conditional_coupling_field",
            "configuration": asdict(selected_primary),
            "population_mean": final_primary["population_mean"].ravel().tolist(),
            "population_change": final_primary["population_change"].ravel().tolist(),
            "fit_certificate": final_primary["fit_certificate"],
        },
        "selected_residual": {
            "kind": "visit_aware_raw_signed_residual",
            "configuration": asdict(selected_residual),
            "population_mean": final_residual["population_mean"].ravel().tolist(),
            "population_change": final_residual["population_change"].ravel().tolist(),
            "fit_certificate": final_residual["fit_certificate"],
        },
        "common_effect_cmle": {
            "kind": "visit_specific_exact_conditional_common_log_odds",
            "configuration": asdict(selected_common),
            "population_mean": final_common["population_mean"].ravel().tolist(),
            "population_change": final_common["population_change"].ravel().tolist(),
            "fit_certificate": final_common["fit_certificate"],
        },
        "pooled_saturated_poisson": {
            "kind": "visit_specific_pooled_saturated_poisson_fixed_interaction",
            "configuration": asdict(selected_poisson),
            "population_mean": final_poisson["population_mean"].ravel().tolist(),
            "population_change": final_poisson["population_change"].ravel().tolist(),
            "fit_certificate": final_poisson["fit_certificate"],
        },
        "destroyed_link": {
            "kind": "destroyed_paired_longitudinal_exact_conditional_coupling_field",
            "configuration": asdict(selected_primary),
            "population_mean": final_destroyed["population_mean"].ravel().tolist(),
            "population_change": final_destroyed["population_change"].ravel().tolist(),
            "fit_certificate": final_destroyed["fit_certificate"],
        },
        "independence": {"kind": "recipient_margin_independence"},
        "visit_agnostic_primary": {
            "kind": "visit_agnostic_exact_conditional_coupling_field",
            "configuration": asdict(selected_visit_agnostic),
            "population_mean": final_visit_agnostic["population_mean"]
            .ravel()
            .tolist(),
            "population_change": final_visit_agnostic["population_change"]
            .ravel()
            .tolist(),
            "fit_certificate": final_visit_agnostic["fit_certificate"],
        },
    }
    return {
        "source_axis": source_axis,
        "passes_source_promotion_gate": bool(passes),
        "selected_primary": asdict(selected_primary),
        "selected_residual": asdict(selected_residual),
        "selected_visit_agnostic_primary": asdict(selected_visit_agnostic),
        "selected_common_effect": asdict(selected_common),
        "selected_pooled_poisson": asdict(selected_poisson),
        "fold_masks": fold_masks,
        "final_mask_diagnostics": final_diagnostics,
        "models": models,
        "comparisons": comparisons,
        "losses": {
            "primary": primary.tolist(),
            "selected_residual": residual.tolist(),
            "pooled_saturated_poisson": poisson_losses_selected.tolist(),
            "destroyed_link": destroyed_losses.tolist(),
            "common_effect_cmle": common_losses_selected.tolist(),
            "independence": independence_losses.tolist(),
            "visit_agnostic_primary": visit_agnostic_losses_selected.tolist(),
        },
        "candidate_evaluations": {
            "primary": _candidate_evaluations(primary_order, primary_losses),
            "visit_agnostic_primary": _candidate_evaluations(
                visit_agnostic_order, visit_agnostic_losses
            ),
            "residual": _candidate_evaluations(residual_order, residual_losses),
            "common_effect": _candidate_evaluations(transport_order, common_losses),
            "pooled_poisson": _candidate_evaluations(transport_order, poisson_losses),
        },
        "fit_certificates": {
            "primary_folds": fold_certificates,
            "destroyed_folds": destroyed_certificates,
        },
        "refusals": refusals,
    }


def _claim_matrix_consumption(path: Path, schema: str, attempt: Path) -> dict[str, Any]:
    payload = {
        "schema": schema,
        "status": "CONSUMED_EXCLUSIVELY_BEFORE_FIRST_NUMERIC_MATRIX_GET",
        "created_at_utc": _timestamp(),
        "attempt_sha256": _sha256(attempt),
        "process_id": os.getpid(),
        "rerun_permitted": False,
    }
    _write_json_x(path, payload)
    return payload


def _delete_download(path: Path | None, identity: dict[str, Any]) -> None:
    if path is not None:
        path.unlink(missing_ok=True)
        identity["deleted"] = not path.exists()


def _read_source_records(
    candidate: dict[str, Any],
    axis_cache: Path,
    scratch: Path,
    audit: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for donor_record in candidate["source_donors"]:
        donor = donor_record["donor"]
        visit_tables = []
        destroyed_tables = []
        rna_profiles = []
        adt_profiles = []
        visit_summaries = []
        for visit in donor_record["visits"]:
            visit_name = visit["timepoint"]
            axes = _visit_axes(candidate, visit, axis_cache)
            gex_identity = {
                "cohort": "source",
                "donor": donor,
                "visit": visit_name,
                "assay": "gex",
                "gsm": visit["gex"]["gsm"],
                "matrix_filename": visit["gex"]["matrix"]["filename"],
                "feature_axis_sha256": visit["gex"]["features_axis"]["sha256"],
                "barcode_axis_sha256": visit["gex"]["barcodes_axis"]["sha256"],
                "matrix_market": {},
            }
            audit["source_files"].append(gex_identity)
            gex_path: Path | None = None
            try:
                gex_path = _fetch_matrix(
                    visit["gex"]["matrix"], scratch, gex_identity
                )
                rna = _reduce_gex_matrix(
                    gex_path,
                    donor,
                    visit_name,
                    axes,
                    gex_identity["matrix_market"],
                )
                gex_identity["reduction_completed"] = True
            finally:
                _delete_download(gex_path, gex_identity)

            cite_identity = {
                "cohort": "source",
                "donor": donor,
                "visit": visit_name,
                "assay": "cite",
                "gsm": visit["cite"]["gsm"],
                "matrix_filename": visit["cite"]["matrix"]["filename"],
                "feature_axis_sha256": visit["cite"]["features_axis"]["sha256"],
                "barcode_axis_sha256": visit["cite"]["barcodes_axis"]["sha256"],
                "selected_barcode_sha256": rna["selected_barcode_sha256"],
                "matrix_market": {},
            }
            audit["source_files"].append(cite_identity)
            cite_path: Path | None = None
            try:
                cite_path = _fetch_matrix(
                    visit["cite"]["matrix"], scratch, cite_identity
                )
                adt = _reduce_cite_matrix(
                    cite_path,
                    donor,
                    visit_name,
                    axes,
                    rna["barcodes"],
                    cite_identity["matrix_market"],
                )
                cite_identity["reduction_completed"] = True
            finally:
                _delete_download(cite_path, cite_identity)

            if rna["cell_count"] != adt["cell_count"]:
                raise ProtocolRefusal("SOURCE_MODALITY_CELL_COUNT_MISMATCH")
            observed = _joint_tables(rna["states"], adt["states"])
            destroyed = _joint_tables(
                rna["states"],
                _destroyed_adt(
                    adt["states"], rna["barcodes"], donor, visit_name
                ),
            )
            if not np.array_equal(
                observed.sum(axis=-1), destroyed.sum(axis=-1)
            ) or not np.array_equal(
                observed.sum(axis=-2), destroyed.sum(axis=-2)
            ):
                raise AssertionError("destroyed source tables changed margins")
            visit_tables.append(observed)
            destroyed_tables.append(destroyed)
            rna_profiles.append(rna["profile"])
            adt_profiles.append(adt["profile"])
            visit_summaries.append(
                {
                    "visit": visit_name,
                    "cell_count": rna["cell_count"],
                    "eligible_barcodes": rna["eligible_barcodes"],
                    "selected_barcode_sha256": rna["selected_barcode_sha256"],
                    "rna_state_sha256": rna["state_sha256"],
                    "adt_state_sha256": adt["state_sha256"],
                    "adt_count_panel_sha256": adt["count_panel_sha256"],
                    "table_sha256": _array_sha256(observed),
                    "destroyed_table_sha256": _array_sha256(destroyed),
                }
            )
            del rna, adt, observed, destroyed
        tables = np.asarray(visit_tables, dtype=np.int64)
        destroyed_array = np.asarray(destroyed_tables, dtype=np.int64)
        records[donor] = {
            "stratum": donor_record["stratum"],
            "tables": tables,
            "destroyed_tables": destroyed_array,
            "rna_profiles": np.asarray(rna_profiles, dtype=float),
            "adt_profiles": np.asarray(adt_profiles, dtype=float),
            "support": _margin_support(tables),
            "visits": visit_summaries,
            "table_panel_sha256": _array_sha256(tables),
            "destroyed_table_panel_sha256": _array_sha256(destroyed_array),
        }
    return records


def _source_record_summaries(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        donor: {
            "stratum": record["stratum"],
            "visits": record["visits"],
            "table_panel_sha256": record["table_panel_sha256"],
            "destroyed_table_panel_sha256": record["destroyed_table_panel_sha256"],
            "rna_profile_sha256": _array_sha256(record["rna_profiles"]),
            "adt_profile_sha256": _array_sha256(record["adt_profiles"]),
            "support_sha256": _array_sha256(record["support"].astype(np.uint8)),
        }
        for donor, record in records.items()
    }


def run_source(
    attempt_path: Path = SOURCE_ATTEMPT,
    output_path: Path = SOURCE_RESULT,
    axis_cache: Path = DEFAULT_AXIS_CACHE,
    scratch: Path = DEFAULT_SCRATCH,
) -> dict[str, Any]:
    """Consume the public source attempt and publish one terminal source result."""

    _require_canonical_path(output_path, SOURCE_RESULT, "source result")
    _require_canonical_path(attempt_path, SOURCE_ATTEMPT, "source attempt")
    if output_path.exists() or SOURCE_CONSUMPTION.exists():
        raise FileExistsError("source attempt has already been consumed")
    if not attempt_path.is_file():
        raise FileNotFoundError("source attempt does not exist")
    candidate_for_capacity = _candidate()
    source_matrices = [
        visit[assay]["matrix"]
        for donor in candidate_for_capacity["source_donors"]
        for visit in donor["visits"]
        for assay in ("gex", "cite")
    ]
    capacity = _scratch_capacity_certificate(scratch, source_matrices)
    consumption = _claim_matrix_consumption(
        SOURCE_CONSUMPTION,
        "gse342939-ra-bcell-source-consumption/1.0",
        attempt_path,
    )
    audit: dict[str, Any] = {
        "source_files": [],
        "held_numeric_urls_requested": 0,
        "raw_tar_or_bcr_urls_requested": 0,
        "scratch_capacity_before_consumption": capacity,
    }
    attempt: dict[str, Any] = {}
    attempt_commit: str | None = None
    try:
        candidate, attempt, attempt_commit = _validate_source_attempt(
            attempt_path, axis_cache
        )
        records = _read_source_records(candidate, axis_cache, scratch, audit)
        source_axis = [record["donor"] for record in candidate["source_donors"]]
        evaluation = _evaluate_source(records, source_axis)
        passes = evaluation["passes_source_promotion_gate"]
        payload = {
            "schema": "gse342939-ra-bcell-source-result/1.0",
            "status": "SOURCE_PROMOTION_PASS" if passes else "SOURCE_PROMOTION_FAIL",
            "created_at_utc": _timestamp(),
            "candidate_sha256": CANDIDATE_SHA256,
            "manifest_sha256": MANIFEST_SHA256,
            "amendment_sha256": AMENDMENT_SHA256,
            "implementation_bindings": _binding_hashes(),
            "source_attempt_sha256": _sha256(attempt_path),
            "source_attempt_commit": attempt_commit,
            "source_consumption_sha256": _sha256(SOURCE_CONSUMPTION),
            "source_consumption": consumption,
            "runtime": attempt["runtime"],
            "source_files": audit["source_files"],
            "access_audit": audit,
            "source_records": _source_record_summaries(records),
            **evaluation,
            "held_numeric_access_authorized": bool(passes),
            "rerun_permitted": False,
        }
    except BaseException as error:
        payload = {
            "schema": "gse342939-ra-bcell-source-result/1.0",
            "status": "TERMINAL_SOURCE_EXECUTION_REFUSAL",
            "created_at_utc": _timestamp(),
            "candidate_sha256": CANDIDATE_SHA256,
            "manifest_sha256": MANIFEST_SHA256,
            "amendment_sha256": AMENDMENT_SHA256,
            "implementation_bindings": _binding_hashes(),
            "source_attempt_sha256": _sha256(attempt_path),
            "source_attempt_commit": attempt_commit,
            "source_consumption_sha256": _sha256(SOURCE_CONSUMPTION),
            "source_consumption": consumption,
            "source_files": audit["source_files"],
            "access_audit": audit,
            "reason_code": error.code
            if isinstance(error, ProtocolRefusal)
            else type(error).__name__,
            "reason": str(error),
            "reason_details": error.details
            if isinstance(error, ProtocolRefusal)
            else {},
            "held_numeric_access_authorized": False,
            "rerun_permitted": False,
        }
    _write_json_x(output_path, payload)
    return payload


def _finite_numeric_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return bool(np.isfinite(value))
    if isinstance(value, list):
        return all(_finite_numeric_tree(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _finite_numeric_tree(item) for key, item in value.items())
    return False


def _selected_published_candidate(
    evaluations: list[dict[str, Any]],
) -> tuple[dict[str, Any], np.ndarray]:
    complete: list[tuple[int, dict[str, Any], np.ndarray]] = []
    for index, record in enumerate(evaluations):
        raw_losses = record.get("fold_losses")
        if not isinstance(raw_losses, list) or len(raw_losses) != 7:
            raise PermissionError("published source fold losses are malformed")
        observed_complete = all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and np.isfinite(value)
            for value in raw_losses
        )
        if record.get("complete") is not observed_complete:
            raise PermissionError("published source completeness flag is inconsistent")
        if not observed_complete:
            if "mean_physical_donor_loss" in record:
                raise PermissionError("incomplete source candidate reports a mean loss")
            continue
        losses = np.asarray(raw_losses, dtype=float)
        reported_mean = record.get("mean_physical_donor_loss")
        if (
            isinstance(reported_mean, bool)
            or not isinstance(reported_mean, (int, float))
            or not np.isfinite(reported_mean)
            or not math.isclose(
                float(reported_mean),
                float(losses.mean()),
                rel_tol=1e-15,
                abs_tol=1e-15,
            )
        ):
            raise PermissionError("published source mean loss does not recompute")
        complete.append((index, record, losses))
    if not complete:
        raise PermissionError("published source family has no complete candidate")
    _, selected, losses = min(
        complete,
        key=lambda item: (float(item[1]["mean_physical_donor_loss"]), item[0]),
    )
    return selected["configuration"], losses


def _validate_source_pass_payload(source: dict[str, Any]) -> None:
    if (
        source.get("schema") != "gse342939-ra-bcell-source-result/1.0"
        or source.get("status") != "SOURCE_PROMOTION_PASS"
        or source.get("candidate_sha256") != CANDIDATE_SHA256
        or source.get("manifest_sha256") != MANIFEST_SHA256
        or source.get("amendment_sha256") != AMENDMENT_SHA256
        or source.get("implementation_bindings") != _binding_hashes()
        or source.get("passes_source_promotion_gate") is not True
        or source.get("held_numeric_access_authorized") is not True
        or source.get("rerun_permitted") is not False
        or not _finite_numeric_tree(source)
    ):
        raise PermissionError("source result is not a finite frozen promotion pass")
    candidate = _candidate()
    designated_files = [
        (donor, visit, assay)
        for donor in candidate["source_donors"]
        for visit in donor["visits"]
        for assay in ("gex", "cite")
    ]
    files = source.get("source_files", [])
    access = source.get("access_audit", {})
    source_matrices = [visit[assay]["matrix"] for _, visit, assay in designated_files]
    _require_scratch_capacity_certificate(
        access.get("scratch_capacity_before_consumption"),
        source_matrices,
        "source",
    )
    if (
        files != access.get("source_files")
        or access.get("held_numeric_urls_requested") != 0
        or access.get("raw_tar_or_bcr_urls_requested") != 0
    ):
        raise PermissionError("source download provenance is incomplete")
    _validate_mixed_download_provenance(files, designated_files, "source")
    evaluations = source.get("candidate_evaluations", {})
    expected_evaluations = {
        "primary": [
            asdict(PrimaryConfig(heterogeneity, ridge, graph, baseline, change))
            for heterogeneity, ridge, graph, baseline, change in product(
                HETEROGENEITY_GRID,
                RIDGE_GRID,
                GRAPH_GRID,
                TRANSPORT_GRID,
                TRANSPORT_GRID,
            )
        ],
        "visit_agnostic_primary": [
            asdict(PrimaryConfig(heterogeneity, ridge, graph, baseline, 0.0))
            for heterogeneity, ridge, graph, baseline in product(
                HETEROGENEITY_GRID,
                RIDGE_GRID,
                GRAPH_GRID,
                TRANSPORT_GRID,
            )
        ],
        "residual": [
            asdict(ResidualConfig(family, baseline, change))
            for family, baseline, change in product(
                RESIDUAL_FAMILIES, TRANSPORT_GRID, TRANSPORT_GRID
            )
        ],
        "common_effect": [
            asdict(TransportConfig(baseline, change))
            for baseline, change in product(TRANSPORT_GRID, TRANSPORT_GRID)
        ],
        "pooled_poisson": [
            asdict(TransportConfig(baseline, change))
            for baseline, change in product(TRANSPORT_GRID, TRANSPORT_GRID)
        ],
    }
    if set(evaluations) != set(expected_evaluations):
        raise PermissionError("published source candidate families are incomplete")
    for family, expected in expected_evaluations.items():
        observed = evaluations[family]
        if (
            not isinstance(observed, list)
            or len(observed) != len(expected)
            or [record.get("configuration") for record in observed] != expected
        ):
            raise PermissionError(f"published {family} grid differs from the freeze")
    selected_primary, primary_losses = _selected_published_candidate(
        evaluations.get("primary", [])
    )
    selected_residual, residual_losses = _selected_published_candidate(
        evaluations.get("residual", [])
    )
    selected_visit_agnostic, visit_agnostic_losses = _selected_published_candidate(
        evaluations.get("visit_agnostic_primary", [])
    )
    selected_common, common_losses = _selected_published_candidate(
        evaluations.get("common_effect", [])
    )
    selected_poisson, poisson_losses = _selected_published_candidate(
        evaluations.get("pooled_poisson", [])
    )
    losses = source.get("losses", {})
    destroyed_losses = np.asarray(losses.get("destroyed_link"), dtype=float)
    independence_losses = np.asarray(losses.get("independence"), dtype=float)
    if (
        selected_primary != source.get("selected_primary")
        or selected_residual != source.get("selected_residual")
        or selected_visit_agnostic
        != source.get("selected_visit_agnostic_primary")
        or selected_common != source.get("selected_common_effect")
        or selected_poisson != source.get("selected_pooled_poisson")
        or not np.array_equal(primary_losses, np.asarray(losses.get("primary")))
        or not np.array_equal(residual_losses, np.asarray(losses.get("selected_residual")))
        or not np.array_equal(poisson_losses, np.asarray(losses.get("pooled_saturated_poisson")))
        or not np.array_equal(common_losses, np.asarray(losses.get("common_effect_cmle")))
        or not np.array_equal(
            visit_agnostic_losses,
            np.asarray(losses.get("visit_agnostic_primary")),
        )
        or destroyed_losses.shape != (7,)
        or independence_losses.shape != (7,)
    ):
        raise PermissionError("source selected models do not match published evaluations")
    recomputed = {
        "selected_residual": _comparison(primary_losses, residual_losses, "residual", "source"),
        "pooled_saturated_poisson": _comparison(
            primary_losses, poisson_losses, "pooled_poisson", "source"
        ),
        "destroyed_link": _comparison(
            primary_losses, destroyed_losses, "destroyed", "source"
        ),
        "common_effect_cmle": _comparison(
            primary_losses, common_losses, "common_effect_cmle", "source"
        ),
        "independence": _comparison(
            primary_losses, independence_losses, "independence", "source"
        ),
    }
    if source.get("comparisons") != recomputed or not all(
        result["passes_frozen_requirement"] for result in recomputed.values()
    ):
        raise PermissionError("published source gate does not recompute")
    models = source.get("models", {})
    mask = np.asarray(models.get("final_mask"), dtype=np.uint8)
    if (
        set(models)
        != {
            "final_mask",
            "final_mask_sha256",
            "primary",
            "selected_residual",
            "common_effect_cmle",
            "pooled_saturated_poisson",
            "destroyed_link",
            "visit_agnostic_primary",
            "independence",
        }
        or mask.shape != (MARKER_COUNT, MARKER_COUNT)
        or not np.isin(mask, (0, 1)).all()
        or int(mask.sum()) < MINIMUM_COORDINATES
        or _array_sha256(mask) != models.get("final_mask_sha256")
        or models.get("primary", {}).get("configuration") != selected_primary
        or models.get("selected_residual", {}).get("configuration") != selected_residual
        or models.get("common_effect_cmle", {}).get("configuration") != selected_common
        or models.get("pooled_saturated_poisson", {}).get("configuration") != selected_poisson
        or models.get("destroyed_link", {}).get("configuration") != selected_primary
        or models.get("visit_agnostic_primary", {}).get("configuration")
        != selected_visit_agnostic
        or models.get("independence") != {"kind": "recipient_margin_independence"}
        or models.get("pooled_saturated_poisson", {})
        .get("fit_certificate", {})
        .get("conditional_noncentral_hypergeometric_reconstruction")
        is not False
    ):
        raise PermissionError("published source models differ from the frozen selections")
    expected_kinds = {
        "primary": "paired_longitudinal_exact_conditional_coupling_field",
        "selected_residual": "visit_aware_raw_signed_residual",
        "common_effect_cmle": "visit_specific_exact_conditional_common_log_odds",
        "pooled_saturated_poisson": (
            "visit_specific_pooled_saturated_poisson_fixed_interaction"
        ),
        "destroyed_link": (
            "destroyed_paired_longitudinal_exact_conditional_coupling_field"
        ),
        "visit_agnostic_primary": (
            "visit_agnostic_exact_conditional_coupling_field"
        ),
    }
    for name, kind in expected_kinds.items():
        model = models[name]
        mean = np.asarray(model.get("population_mean"), dtype=float)
        change = np.asarray(model.get("population_change"), dtype=float)
        if (
            model.get("kind") != kind
            or mean.shape != (MARKER_COUNT * MARKER_COUNT,)
            or change.shape != mean.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(change).all()
            or model.get("fit_certificate", {}).get("passes") is not True
        ):
            raise PermissionError("published source model field is incomplete")
    if not np.array_equal(
        np.asarray(models["visit_agnostic_primary"]["population_change"]),
        np.zeros(MARKER_COUNT * MARKER_COUNT),
    ):
        raise PermissionError("visit-agnostic source change is not exactly zero")
    source_axis = [record["donor"] for record in candidate["source_donors"]]
    fold_masks = source.get("fold_masks", {})
    if set(fold_masks) != set(source_axis):
        raise PermissionError("source fold-specific masks are incomplete")
    for held in source_axis:
        fold = fold_masks[held]
        if (
            fold.get("training_donors")
            != [donor for donor in source_axis if donor != held]
            or fold.get("validation_donor") != held
            or fold.get("validation_association_used_for_mask") is not False
            or min(fold.get("scored_coordinate_counts", [0, 0]))
            < MINIMUM_COORDINATES
        ):
            raise PermissionError("source fold mask is not training-donor-only")


def _require_passing_source() -> tuple[dict[str, Any], str]:
    source_commit = _require_public_tag(
        SOURCE_TAG,
        (_relative(SOURCE_RESULT), _relative(SOURCE_CONSUMPTION)),
    )
    source = _read_json(SOURCE_RESULT)
    _validate_source_pass_payload(source)
    _, _, attempt_commit = _validate_source_attempt(SOURCE_ATTEMPT)
    consumption = _read_json(SOURCE_CONSUMPTION)
    _require_ancestor(attempt_commit, source_commit)
    if (
        source.get("source_attempt_sha256") != _sha256(SOURCE_ATTEMPT)
        or source.get("source_attempt_commit") != attempt_commit
        or source.get("source_consumption_sha256") != _sha256(SOURCE_CONSUMPTION)
        or source.get("source_consumption") != consumption
        or consumption.get("schema")
        != "gse342939-ra-bcell-source-consumption/1.0"
        or consumption.get("status")
        != "CONSUMED_EXCLUSIVELY_BEFORE_FIRST_NUMERIC_MATRIX_GET"
        or consumption.get("attempt_sha256") != _sha256(SOURCE_ATTEMPT)
        or consumption.get("rerun_permitted") is not False
    ):
        raise PermissionError("source consumption differs from the public source result")
    return source, source_commit


def claim_held_rna(
    attempt_path: Path = HELD_RNA_ATTEMPT,
    axis_cache: Path = DEFAULT_AXIS_CACHE,
    scratch: Path = DEFAULT_SCRATCH,
) -> dict[str, Any]:
    """Authorize only held GEX matrices after a public source pass."""

    _require_canonical_path(attempt_path, HELD_RNA_ATTEMPT, "held-RNA attempt")
    forbidden = (
        HELD_RNA_CONSUMPTION,
        HELD_RNA_MARGINS,
        PRIVATE_BARCODES,
        PRIVATE_RNA,
        HELD_ADT_ATTEMPT,
        HELD_MARGINS,
        PRIVATE_ADT,
        HELD_PREDICTIONS,
        SCORE_AUTHORIZATION,
        SCORE_ATTEMPT,
        SCORE_RESULT,
    )
    if attempt_path.exists() or any(path.exists() for path in forbidden):
        raise FileExistsError("held-RNA stage has already been claimed or advanced")
    source, source_commit = _require_passing_source()
    candidate = _candidate()
    axis_certificate = _validate_axis_cache(axis_cache)
    matrices = [
        visit["gex"]["matrix"]
        for donor in candidate["held_donors"]
        for visit in donor["visits"]
    ]
    scratch_capacity = _scratch_capacity_certificate(scratch, matrices)
    payload = {
        "schema": "gse342939-ra-bcell-held-rna-attempt/1.0",
        "status": "CLAIMED_BEFORE_FIRST_HELD_GEX_MATRIX_GET",
        "created_at_utc": _timestamp(),
        "source_result_sha256": _sha256(SOURCE_RESULT),
        "source_commit": source_commit,
        "source_status": source["status"],
        "implementation_bindings": _binding_hashes(),
        "runtime": _runtime_record(),
        "axis_cache_certificate": axis_certificate,
        "scratch_capacity_certificate": scratch_capacity,
        "held_gex_urls": [record["url"] for record in matrices],
        "held_gex_expected_bytes": sum(
            int(record["expected_bytes"]) for record in matrices
        ),
        "held_gex_file_count": 12,
        "held_cite_access_authorized": False,
        "rna_state_pairing_authorized": False,
        "rerun_permitted": False,
    }
    _write_json_x(attempt_path, payload)
    return payload


def _validate_held_rna_attempt(
    path: Path,
    axis_cache: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    _require_canonical_path(path, HELD_RNA_ATTEMPT, "held-RNA attempt")
    source, source_commit = _require_passing_source()
    attempt = _read_json(path)
    candidate = _candidate()
    matrices = [
        visit["gex"]["matrix"]
        for donor in candidate["held_donors"]
        for visit in donor["visits"]
    ]
    if (
        attempt.get("schema") != "gse342939-ra-bcell-held-rna-attempt/1.0"
        or attempt.get("status") != "CLAIMED_BEFORE_FIRST_HELD_GEX_MATRIX_GET"
        or attempt.get("source_result_sha256") != _sha256(SOURCE_RESULT)
        or attempt.get("source_commit") != source_commit
        or attempt.get("source_status") != source["status"]
        or attempt.get("implementation_bindings") != _binding_hashes()
        or attempt.get("runtime") != _runtime_record()
        or attempt.get("axis_cache_certificate") != _validate_axis_cache(axis_cache)
        or not _valid_scratch_capacity_certificate(
            attempt.get("scratch_capacity_certificate"), matrices
        )
        or attempt.get("held_gex_file_count") != 12
        or attempt.get("held_gex_urls")
        != [record["url"] for record in matrices]
        or attempt.get("held_gex_expected_bytes")
        != sum(int(record["expected_bytes"]) for record in matrices)
        or attempt.get("held_cite_access_authorized") is not False
        or attempt.get("rna_state_pairing_authorized") is not False
        or attempt.get("rerun_permitted") is not False
    ):
        raise PermissionError("held-RNA attempt differs from the public source seal")
    attempt_commit = _require_public_tag(
        HELD_RNA_ATTEMPT_TAG, (_relative(path),)
    )
    _require_ancestor(source_commit, attempt_commit)
    return source, attempt, attempt_commit


def run_held_rna(
    attempt_path: Path = HELD_RNA_ATTEMPT,
    output_path: Path = HELD_RNA_MARGINS,
    private_barcodes: Path = PRIVATE_BARCODES,
    private_rna: Path = PRIVATE_RNA,
    axis_cache: Path = DEFAULT_AXIS_CACHE,
    scratch: Path = DEFAULT_SCRATCH,
) -> dict[str, Any]:
    """Reduce held GEX matrices without requesting CITE or forming joint tables."""

    _require_canonical_path(attempt_path, HELD_RNA_ATTEMPT, "held-RNA attempt")
    _require_canonical_path(output_path, HELD_RNA_MARGINS, "held-RNA margins")
    _require_canonical_path(private_barcodes, PRIVATE_BARCODES, "private barcodes")
    _require_canonical_path(private_rna, PRIVATE_RNA, "private RNA states")
    if (
        output_path.exists()
        or HELD_RNA_CONSUMPTION.exists()
        or private_barcodes.exists()
        or private_rna.exists()
    ):
        raise FileExistsError("held-RNA attempt has already been consumed")
    if not attempt_path.is_file():
        raise FileNotFoundError("held-RNA attempt does not exist")
    candidate_for_capacity = _candidate()
    held_gex_matrices = [
        visit["gex"]["matrix"]
        for donor in candidate_for_capacity["held_donors"]
        for visit in donor["visits"]
    ]
    capacity = _scratch_capacity_certificate(scratch, held_gex_matrices)
    consumption = _claim_matrix_consumption(
        HELD_RNA_CONSUMPTION,
        "gse342939-ra-bcell-held-rna-consumption/1.0",
        attempt_path,
    )
    audit: dict[str, Any] = {
        "held_gex_files": [],
        "held_cite_urls_requested": 0,
        "joint_table_calls": 0,
        "scratch_capacity_before_consumption": capacity,
    }
    arrays: dict[str, np.ndarray] = {}
    barcode_payload: dict[str, Any] = {
        "schema": "gse342939-ra-bcell-private-selected-barcodes/1.0",
        "source_result_sha256": _sha256(SOURCE_RESULT),
        "records": {},
    }
    margin_records: dict[str, Any] = {}
    source: dict[str, Any] = {}
    attempt_commit: str | None = None
    try:
        source, _, attempt_commit = _validate_held_rna_attempt(
            attempt_path, axis_cache
        )
        candidate = _candidate()
        for donor_record in candidate["held_donors"]:
            donor = donor_record["donor"]
            donor_visits = []
            for visit in donor_record["visits"]:
                visit_name = visit["timepoint"]
                axes = _visit_axes(candidate, visit, axis_cache)
                identity = {
                    "cohort": "held_rna",
                    "donor": donor,
                    "visit": visit_name,
                    "assay": "gex",
                    "gsm": visit["gex"]["gsm"],
                    "matrix_filename": visit["gex"]["matrix"]["filename"],
                    "feature_axis_sha256": visit["gex"]["features_axis"]["sha256"],
                    "barcode_axis_sha256": visit["gex"]["barcodes_axis"]["sha256"],
                    "matrix_market": {},
                }
                audit["held_gex_files"].append(identity)
                path: Path | None = None
                try:
                    path = _fetch_matrix(visit["gex"]["matrix"], scratch, identity)
                    reduced = _reduce_gex_matrix(
                        path,
                        donor,
                        visit_name,
                        axes,
                        identity["matrix_market"],
                    )
                    identity["reduction_completed"] = True
                finally:
                    _delete_download(path, identity)
                key = f"{donor}__{visit_name}"
                arrays[key] = reduced["states"]
                barcode_payload["records"][key] = reduced["barcodes"]
                positive = reduced["states"].sum(axis=0).astype(np.int64)
                margins = np.stack(
                    [reduced["cell_count"] - positive, positive], axis=1
                )
                donor_visits.append(
                    {
                        "visit": visit_name,
                        "cell_count": reduced["cell_count"],
                        "rna_margins": margins.tolist(),
                        "rna_margins_sha256": _array_sha256(margins),
                        "selected_barcode_sha256": reduced[
                            "selected_barcode_sha256"
                        ],
                        "rna_state_sha256": reduced["state_sha256"],
                    }
                )
                del reduced
            margin_records[donor] = {
                "stratum": donor_record["stratum"],
                "visits": donor_visits,
            }
        _exclusive_private_npz(private_rna, arrays)
        _exclusive_private_json(private_barcodes, barcode_payload)
        payload = {
            "schema": "gse342939-ra-bcell-held-rna-margins/1.0",
            "status": "HELD_RNA_MARGINS_COMPLETE",
            "created_at_utc": _timestamp(),
            "source_result_sha256": _sha256(SOURCE_RESULT),
            "source_status": source["status"],
            "held_rna_attempt_sha256": _sha256(attempt_path),
            "held_rna_attempt_commit": attempt_commit,
            "held_rna_consumption_sha256": _sha256(HELD_RNA_CONSUMPTION),
            "held_rna_consumption": consumption,
            "implementation_bindings": _binding_hashes(),
            "held_gex_files": audit["held_gex_files"],
            "access_audit": audit,
            "donors": margin_records,
            "private_selected_barcodes_path": _relative(private_barcodes),
            "private_selected_barcodes_sha256": _sha256(private_barcodes),
            "private_rna_states_path": _relative(private_rna),
            "private_rna_states_sha256": _sha256(private_rna),
            "held_cite_access_authorized": True,
            "joint_table_formation_authorized": False,
            "rerun_permitted": False,
        }
    except BaseException as error:
        private_rna.unlink(missing_ok=True)
        private_barcodes.unlink(missing_ok=True)
        payload = {
            "schema": "gse342939-ra-bcell-held-rna-margins/1.0",
            "status": "TERMINAL_HELD_RNA_EXECUTION_REFUSAL",
            "created_at_utc": _timestamp(),
            "source_result_sha256": _sha256(SOURCE_RESULT),
            "held_rna_attempt_sha256": _sha256(attempt_path),
            "held_rna_attempt_commit": attempt_commit,
            "held_rna_consumption_sha256": _sha256(HELD_RNA_CONSUMPTION),
            "held_rna_consumption": consumption,
            "implementation_bindings": _binding_hashes(),
            "held_gex_files": audit["held_gex_files"],
            "access_audit": audit,
            "reason_code": error.code
            if isinstance(error, ProtocolRefusal)
            else type(error).__name__,
            "reason": str(error),
            "reason_details": error.details
            if isinstance(error, ProtocolRefusal)
            else {},
            "held_cite_access_authorized": False,
            "joint_table_formation_authorized": False,
            "rerun_permitted": False,
        }
    _write_json_x(output_path, payload)
    return payload


def _held_matrix_records(
    candidate: dict[str, Any], assay: str
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return [
        (donor, visit)
        for donor in candidate["held_donors"]
        for visit in donor["visits"]
        if assay in visit
    ]


def _validate_mixed_download_provenance(
    observed: Any,
    designated: list[tuple[dict[str, Any], dict[str, Any], str]],
    cohort: str,
) -> None:
    if not isinstance(observed, list) or len(observed) != len(designated):
        raise PermissionError("matrix download provenance has the wrong length")
    for actual, (donor, visit, assay) in zip(observed, designated):
        expected = visit[assay]
        matrix = expected["matrix"]
        expected_shape = (
            int(expected["features_axis"]["decoded_line_count"]),
            int(expected["barcodes_axis"]["decoded_line_count"]),
        )
        if (
            actual.get("cohort") != cohort
            or actual.get("donor") != donor["donor"]
            or actual.get("visit") != visit["timepoint"]
            or actual.get("assay") != assay
            or actual.get("gsm") != expected["gsm"]
            or actual.get("matrix_filename") != matrix["filename"]
            or actual.get("feature_axis_sha256")
            != expected["features_axis"]["sha256"]
            or actual.get("barcode_axis_sha256")
            != expected["barcodes_axis"]["sha256"]
            or actual.get("requested_url") != matrix["url"]
            or actual.get("final_url") != matrix["url"]
            or actual.get("expected_bytes") != matrix["expected_bytes"]
            or actual.get("observed_bytes") != matrix["expected_bytes"]
            or actual.get("hashed_bytes") != matrix["expected_bytes"]
            or actual.get("completed") is not True
            or actual.get("deleted") is not True
            or actual.get("reduction_completed") is not True
            or re.fullmatch(r"[0-9a-f]{64}", str(actual.get("sha256"))) is None
            or not _complete_matrix_audit(
                actual.get("matrix_market"), expected_shape, assay
            )
        ):
            raise PermissionError("matrix download provenance is incomplete")


def _validate_download_provenance(
    observed: Any,
    designated: list[tuple[dict[str, Any], dict[str, Any]]],
    assay: str,
) -> None:
    _validate_mixed_download_provenance(
        observed,
        [(donor, visit, assay) for donor, visit in designated],
        {"gex": "held_rna", "cite": "held_adt"}[assay],
    )


def _validated_public_held_rna(
    axis_cache: Path = DEFAULT_AXIS_CACHE,
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    source, source_commit = _require_passing_source()
    _, _, attempt_commit = _validate_held_rna_attempt(HELD_RNA_ATTEMPT, axis_cache)
    margins, margins_commit = _require_public_result(
        HELD_RNA_MARGINS_TAG,
        HELD_RNA_MARGINS,
        "gse342939-ra-bcell-held-rna-margins/1.0",
    )
    _require_ancestor(attempt_commit, margins_commit)
    if (
        _require_public_tag(
            HELD_RNA_MARGINS_TAG, (_relative(HELD_RNA_CONSUMPTION),)
        )
        != margins_commit
    ):
        raise PermissionError("held-RNA consumption is on another public commit")
    consumption = _read_json(HELD_RNA_CONSUMPTION)
    candidate = _candidate()
    expected_donors = [record["donor"] for record in candidate["held_donors"]]
    access = margins.get("access_audit", {})
    held_gex_matrices = [
        visit["gex"]["matrix"]
        for donor in candidate["held_donors"]
        for visit in donor["visits"]
    ]
    _require_scratch_capacity_certificate(
        access.get("scratch_capacity_before_consumption"),
        held_gex_matrices,
        "held-RNA",
    )
    if (
        margins.get("status") != "HELD_RNA_MARGINS_COMPLETE"
        or margins.get("source_result_sha256") != _sha256(SOURCE_RESULT)
        or margins.get("source_status") != "SOURCE_PROMOTION_PASS"
        or margins.get("held_rna_attempt_sha256") != _sha256(HELD_RNA_ATTEMPT)
        or margins.get("held_rna_attempt_commit") != attempt_commit
        or margins.get("held_rna_consumption_sha256")
        != _sha256(HELD_RNA_CONSUMPTION)
        or margins.get("held_rna_consumption") != consumption
        or consumption.get("schema")
        != "gse342939-ra-bcell-held-rna-consumption/1.0"
        or consumption.get("status")
        != "CONSUMED_EXCLUSIVELY_BEFORE_FIRST_NUMERIC_MATRIX_GET"
        or consumption.get("attempt_sha256") != _sha256(HELD_RNA_ATTEMPT)
        or consumption.get("rerun_permitted") is not False
        or margins.get("implementation_bindings") != _binding_hashes()
        or set(margins.get("donors", {})) != set(expected_donors)
        or margins.get("private_selected_barcodes_path")
        != _relative(PRIVATE_BARCODES)
        or margins.get("private_rna_states_path") != _relative(PRIVATE_RNA)
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(margins.get("private_selected_barcodes_sha256")),
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}", str(margins.get("private_rna_states_sha256"))
        )
        is None
        or margins.get("held_cite_access_authorized") is not True
        or margins.get("joint_table_formation_authorized") is not False
        or margins.get("rerun_permitted") is not False
        or access.get("held_cite_urls_requested") != 0
        or access.get("joint_table_calls") != 0
    ):
        raise PermissionError("public held-RNA margins violate the frozen firewall")
    _validate_download_provenance(
        margins.get("held_gex_files"), _held_matrix_records(candidate, "gex"), "gex"
    )
    for donor_record in candidate["held_donors"]:
        published = margins["donors"][donor_record["donor"]]
        if (
            published.get("stratum") != donor_record["stratum"]
            or [record.get("visit") for record in published.get("visits", [])]
            != list(VISITS)
        ):
            raise PermissionError("held-RNA donor or visit axis changed")
        for visit in published["visits"]:
            raw = np.asarray(visit.get("rna_margins"))
            count = visit.get("cell_count")
            if (
                not isinstance(count, int)
                or count < MINIMUM_CELLS
                or count > MAXIMUM_CELLS
                or count % 2
                or raw.shape != (MARKER_COUNT, 2)
                or not np.issubdtype(raw.dtype, np.integer)
                or np.any(raw < 0)
                or not np.all(raw.sum(axis=1) == count)
                or visit.get("rna_margins_sha256") != _array_sha256(raw)
                or re.fullmatch(
                    r"[0-9a-f]{64}", str(visit.get("selected_barcode_sha256"))
                )
                is None
                or re.fullmatch(
                    r"[0-9a-f]{64}", str(visit.get("rna_state_sha256"))
                )
                is None
            ):
                raise PermissionError("held-RNA numeric margins are invalid")
    return margins, margins_commit, source, source_commit


def claim_held_adt(
    attempt_path: Path = HELD_ADT_ATTEMPT,
    axis_cache: Path = DEFAULT_AXIS_CACHE,
    scratch: Path = DEFAULT_SCRATCH,
) -> dict[str, Any]:
    """Authorize only held CITE matrices after public held-RNA margins."""

    _require_canonical_path(attempt_path, HELD_ADT_ATTEMPT, "held-ADT attempt")
    forbidden = (
        HELD_ADT_CONSUMPTION,
        HELD_MARGINS,
        PRIVATE_ADT,
        HELD_PREDICTIONS,
        SCORE_AUTHORIZATION,
        SCORE_ATTEMPT,
        SCORE_RESULT,
    )
    if attempt_path.exists() or any(path.exists() for path in forbidden):
        raise FileExistsError("held-ADT stage has already been claimed or advanced")
    rna, rna_commit, _, _ = _validated_public_held_rna(axis_cache)
    candidate = _candidate()
    matrices = [
        visit["cite"]["matrix"]
        for donor in candidate["held_donors"]
        for visit in donor["visits"]
    ]
    scratch_capacity = _scratch_capacity_certificate(scratch, matrices)
    payload = {
        "schema": "gse342939-ra-bcell-held-adt-attempt/1.0",
        "status": "CLAIMED_BEFORE_FIRST_HELD_CITE_MATRIX_GET",
        "created_at_utc": _timestamp(),
        "held_rna_margins_sha256": _sha256(HELD_RNA_MARGINS),
        "held_rna_margins_commit": rna_commit,
        "private_selected_barcodes_sha256": rna[
            "private_selected_barcodes_sha256"
        ],
        "implementation_bindings": _binding_hashes(),
        "runtime": _runtime_record(),
        "axis_cache_certificate": _validate_axis_cache(axis_cache),
        "scratch_capacity_certificate": scratch_capacity,
        "held_cite_urls": [record["url"] for record in matrices],
        "held_cite_expected_bytes": sum(
            int(record["expected_bytes"]) for record in matrices
        ),
        "held_cite_file_count": 12,
        "rna_state_access_authorized": False,
        "joint_table_formation_authorized": False,
        "rerun_permitted": False,
    }
    _write_json_x(attempt_path, payload)
    return payload


def _validate_held_adt_attempt(
    path: Path,
    axis_cache: Path,
) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    _require_canonical_path(path, HELD_ADT_ATTEMPT, "held-ADT attempt")
    rna, rna_commit, _, _ = _validated_public_held_rna(axis_cache)
    attempt = _read_json(path)
    candidate = _candidate()
    matrices = [
        visit["cite"]["matrix"]
        for donor in candidate["held_donors"]
        for visit in donor["visits"]
    ]
    if (
        attempt.get("schema") != "gse342939-ra-bcell-held-adt-attempt/1.0"
        or attempt.get("status") != "CLAIMED_BEFORE_FIRST_HELD_CITE_MATRIX_GET"
        or attempt.get("held_rna_margins_sha256") != _sha256(HELD_RNA_MARGINS)
        or attempt.get("held_rna_margins_commit") != rna_commit
        or attempt.get("private_selected_barcodes_sha256")
        != rna["private_selected_barcodes_sha256"]
        or attempt.get("implementation_bindings") != _binding_hashes()
        or attempt.get("runtime") != _runtime_record()
        or attempt.get("axis_cache_certificate") != _validate_axis_cache(axis_cache)
        or not _valid_scratch_capacity_certificate(
            attempt.get("scratch_capacity_certificate"), matrices
        )
        or attempt.get("held_cite_file_count") != 12
        or attempt.get("held_cite_urls")
        != [record["url"] for record in matrices]
        or attempt.get("held_cite_expected_bytes")
        != sum(int(record["expected_bytes"]) for record in matrices)
        or attempt.get("rna_state_access_authorized") is not False
        or attempt.get("joint_table_formation_authorized") is not False
        or attempt.get("rerun_permitted") is not False
    ):
        raise PermissionError("held-ADT attempt differs from public RNA margins")
    attempt_commit = _require_public_tag(
        HELD_ADT_ATTEMPT_TAG, (_relative(path),)
    )
    _require_ancestor(rna_commit, attempt_commit)
    return attempt, rna, attempt_commit, candidate


def _load_private_barcodes(
    path: Path, rna: dict[str, Any]
) -> dict[str, list[str]]:
    if _sha256(path) != rna["private_selected_barcodes_sha256"]:
        raise PermissionError("private selected barcodes differ from public RNA seal")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & (stat.S_IRWXG | stat.S_IRWXO) or not mode & stat.S_IRUSR:
        raise PermissionError("private selected barcodes are not owner-only")
    value = _read_json(path)
    records = value.get("records", {})
    expected = {
        f"{donor}__{visit}"
        for donor in rna["donors"]
        for visit in VISITS
    }
    if (
        value.get("schema")
        != "gse342939-ra-bcell-private-selected-barcodes/1.0"
        or value.get("source_result_sha256") != _sha256(SOURCE_RESULT)
        or set(records) != expected
    ):
        raise PermissionError("private barcode donor or visit axis changed")
    output: dict[str, list[str]] = {}
    for key, raw in records.items():
        barcodes = [str(value) for value in raw]
        donor, visit = key.split("__", 1)
        published = next(
            item
            for item in rna["donors"][donor]["visits"]
            if item["visit"] == visit
        )
        if (
            len(barcodes) != published["cell_count"]
            or len(barcodes) != len(set(barcodes))
            or _axis_sha256(barcodes) != published["selected_barcode_sha256"]
        ):
            raise PermissionError("private barcode axis differs from public RNA seal")
        output[key] = barcodes
    return output


def run_held_adt(
    attempt_path: Path = HELD_ADT_ATTEMPT,
    output_path: Path = HELD_MARGINS,
    private_barcodes: Path = PRIVATE_BARCODES,
    private_adt: Path = PRIVATE_ADT,
    axis_cache: Path = DEFAULT_AXIS_CACHE,
    scratch: Path = DEFAULT_SCRATCH,
) -> dict[str, Any]:
    """Reduce held CITE matrices without loading RNA states or joint tables."""

    _require_canonical_path(attempt_path, HELD_ADT_ATTEMPT, "held-ADT attempt")
    _require_canonical_path(output_path, HELD_MARGINS, "held margins")
    _require_canonical_path(private_barcodes, PRIVATE_BARCODES, "private barcodes")
    _require_canonical_path(private_adt, PRIVATE_ADT, "private ADT states")
    if output_path.exists() or HELD_ADT_CONSUMPTION.exists() or private_adt.exists():
        raise FileExistsError("held-ADT attempt has already been consumed")
    if not attempt_path.is_file():
        raise FileNotFoundError("held-ADT attempt does not exist")
    candidate_for_capacity = _candidate()
    held_cite_matrices = [
        visit["cite"]["matrix"]
        for donor in candidate_for_capacity["held_donors"]
        for visit in donor["visits"]
    ]
    capacity = _scratch_capacity_certificate(scratch, held_cite_matrices)
    consumption = _claim_matrix_consumption(
        HELD_ADT_CONSUMPTION,
        "gse342939-ra-bcell-held-adt-consumption/1.0",
        attempt_path,
    )
    audit: dict[str, Any] = {
        "held_cite_files": [],
        "rna_state_reads": 0,
        "joint_table_calls": 0,
        "network_gex_gets": 0,
        "scratch_capacity_before_consumption": capacity,
    }
    arrays: dict[str, np.ndarray] = {}
    public_records: dict[str, Any] = {}
    rna: dict[str, Any] = {}
    attempt_commit: str | None = None
    try:
        _, rna, attempt_commit, candidate = _validate_held_adt_attempt(
            attempt_path, axis_cache
        )
        selected_barcodes = _load_private_barcodes(private_barcodes, rna)
        for donor_record in candidate["held_donors"]:
            donor = donor_record["donor"]
            donor_visits = []
            rna_by_visit = {
                record["visit"]: record for record in rna["donors"][donor]["visits"]
            }
            for visit in donor_record["visits"]:
                visit_name = visit["timepoint"]
                axes = _visit_axes(candidate, visit, axis_cache)
                barcodes = selected_barcodes[f"{donor}__{visit_name}"]
                identity = {
                    "cohort": "held_adt",
                    "donor": donor,
                    "visit": visit_name,
                    "assay": "cite",
                    "gsm": visit["cite"]["gsm"],
                    "matrix_filename": visit["cite"]["matrix"]["filename"],
                    "feature_axis_sha256": visit["cite"]["features_axis"]["sha256"],
                    "barcode_axis_sha256": visit["cite"]["barcodes_axis"]["sha256"],
                    "selected_barcode_sha256": _axis_sha256(barcodes),
                    "matrix_market": {},
                }
                audit["held_cite_files"].append(identity)
                path: Path | None = None
                try:
                    path = _fetch_matrix(visit["cite"]["matrix"], scratch, identity)
                    reduced = _reduce_cite_matrix(
                        path,
                        donor,
                        visit_name,
                        axes,
                        barcodes,
                        identity["matrix_market"],
                    )
                    identity["reduction_completed"] = True
                finally:
                    _delete_download(path, identity)
                key = f"{donor}__{visit_name}"
                arrays[key] = reduced["states"]
                count = reduced["cell_count"]
                positive = reduced["states"].sum(axis=0).astype(np.int64)
                margins = np.stack([count - positive, positive], axis=1)
                rna_record = rna_by_visit[visit_name]
                if count != rna_record["cell_count"]:
                    raise ProtocolRefusal("HELD_MODALITY_CELL_COUNT_MISMATCH")
                donor_visits.append(
                    {
                        "visit": visit_name,
                        "cell_count": count,
                        "selected_barcode_sha256": _axis_sha256(barcodes),
                        "rna_margins": rna_record["rna_margins"],
                        "rna_margins_sha256": rna_record["rna_margins_sha256"],
                        "adt_margins": margins.tolist(),
                        "adt_margins_sha256": _array_sha256(margins),
                        "rna_state_sha256": rna_record["rna_state_sha256"],
                        "adt_state_sha256": reduced["state_sha256"],
                        "adt_count_panel_sha256": reduced["count_panel_sha256"],
                    }
                )
                del reduced
            public_records[donor] = {
                "stratum": donor_record["stratum"],
                "visits": donor_visits,
            }
        _exclusive_private_npz(private_adt, arrays)
        payload = {
            "schema": "gse342939-ra-bcell-held-margins/1.0",
            "status": "HELD_MARGINS_COMPLETE_WITHOUT_PAIRING",
            "created_at_utc": _timestamp(),
            "source_result_sha256": _sha256(SOURCE_RESULT),
            "held_rna_margins_sha256": _sha256(HELD_RNA_MARGINS),
            "held_adt_attempt_sha256": _sha256(attempt_path),
            "held_adt_attempt_commit": attempt_commit,
            "held_adt_consumption_sha256": _sha256(HELD_ADT_CONSUMPTION),
            "held_adt_consumption": consumption,
            "implementation_bindings": _binding_hashes(),
            "donors": public_records,
            "held_cite_files": audit["held_cite_files"],
            "access_audit": audit,
            "private_rna_states_path": _relative(PRIVATE_RNA),
            "private_rna_states_sha256": rna["private_rna_states_sha256"],
            "private_adt_states_path": _relative(private_adt),
            "private_adt_states_sha256": _sha256(private_adt),
            "prediction_stage_authorized": True,
            "joint_table_formation_authorized": False,
            "rerun_permitted": False,
        }
    except BaseException as error:
        private_adt.unlink(missing_ok=True)
        payload = {
            "schema": "gse342939-ra-bcell-held-margins/1.0",
            "status": "TERMINAL_HELD_ADT_EXECUTION_REFUSAL",
            "created_at_utc": _timestamp(),
            "source_result_sha256": _sha256(SOURCE_RESULT),
            "held_rna_margins_sha256": _sha256(HELD_RNA_MARGINS),
            "held_adt_attempt_sha256": _sha256(attempt_path),
            "held_adt_attempt_commit": attempt_commit,
            "held_adt_consumption_sha256": _sha256(HELD_ADT_CONSUMPTION),
            "held_adt_consumption": consumption,
            "implementation_bindings": _binding_hashes(),
            "completed_donors": public_records,
            "held_cite_files": audit["held_cite_files"],
            "access_audit": audit,
            "reason_code": error.code
            if isinstance(error, ProtocolRefusal)
            else type(error).__name__,
            "reason": str(error),
            "reason_details": error.details
            if isinstance(error, ProtocolRefusal)
            else {},
            "prediction_stage_authorized": False,
            "joint_table_formation_authorized": False,
            "rerun_permitted": False,
        }
    _write_json_x(output_path, payload)
    return payload


def _pair_margins(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    visits = record.get("visits", [])
    if [visit.get("visit") for visit in visits] != list(VISITS):
        raise PermissionError("held margin visit axis changed")
    rows = np.empty((2, MARKER_COUNT, MARKER_COUNT, 2), dtype=float)
    columns = np.empty_like(rows)
    for visit_index, visit in enumerate(visits):
        rna_raw = np.asarray(visit.get("rna_margins"))
        adt_raw = np.asarray(visit.get("adt_margins"))
        count = visit.get("cell_count")
        if (
            not isinstance(count, int)
            or count < MINIMUM_CELLS
            or count > MAXIMUM_CELLS
            or count % 2
            or rna_raw.shape != (MARKER_COUNT, 2)
            or adt_raw.shape != (MARKER_COUNT, 2)
            or not np.issubdtype(rna_raw.dtype, np.integer)
            or not np.issubdtype(adt_raw.dtype, np.integer)
            or np.any(rna_raw < 0)
            or np.any(adt_raw < 0)
            or not np.all(rna_raw.sum(axis=1) == count)
            or not np.all(adt_raw == count // 2)
            or visit.get("rna_margins_sha256") != _array_sha256(rna_raw)
            or visit.get("adt_margins_sha256") != _array_sha256(adt_raw)
            or re.fullmatch(
                r"[0-9a-f]{64}", str(visit.get("selected_barcode_sha256"))
            )
            is None
            or re.fullmatch(
                r"[0-9a-f]{64}", str(visit.get("rna_state_sha256"))
            )
            is None
            or re.fullmatch(
                r"[0-9a-f]{64}", str(visit.get("adt_state_sha256"))
            )
            is None
        ):
            raise PermissionError("held margins have invalid numeric shape or totals")
        rows[visit_index] = np.broadcast_to(
            rna_raw[:, None, :], (MARKER_COUNT, MARKER_COUNT, 2)
        )
        columns[visit_index] = np.broadcast_to(
            adt_raw[None, :, :], (MARKER_COUNT, MARKER_COUNT, 2)
        )
    return rows, columns


def _validated_public_margins(
    axis_cache: Path = DEFAULT_AXIS_CACHE,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    rna, rna_commit, source, _ = _validated_public_held_rna(axis_cache)
    _, _, attempt_commit, candidate = _validate_held_adt_attempt(
        HELD_ADT_ATTEMPT, axis_cache
    )
    margins, commit = _require_public_result(
        MARGINS_TAG,
        HELD_MARGINS,
        "gse342939-ra-bcell-held-margins/1.0",
    )
    _require_ancestor(attempt_commit, commit)
    if (
        _require_public_tag(MARGINS_TAG, (_relative(HELD_ADT_CONSUMPTION),))
        != commit
    ):
        raise PermissionError("held-ADT consumption is on another public commit")
    consumption = _read_json(HELD_ADT_CONSUMPTION)
    expected_donors = [record["donor"] for record in candidate["held_donors"]]
    access = margins.get("access_audit", {})
    held_cite_matrices = [
        visit["cite"]["matrix"]
        for donor in candidate["held_donors"]
        for visit in donor["visits"]
    ]
    _require_scratch_capacity_certificate(
        access.get("scratch_capacity_before_consumption"),
        held_cite_matrices,
        "held-ADT",
    )
    if (
        margins.get("status") != "HELD_MARGINS_COMPLETE_WITHOUT_PAIRING"
        or margins.get("source_result_sha256") != _sha256(SOURCE_RESULT)
        or margins.get("held_rna_margins_sha256") != _sha256(HELD_RNA_MARGINS)
        or margins.get("held_adt_attempt_sha256") != _sha256(HELD_ADT_ATTEMPT)
        or margins.get("held_adt_attempt_commit") != attempt_commit
        or margins.get("held_adt_consumption_sha256")
        != _sha256(HELD_ADT_CONSUMPTION)
        or margins.get("held_adt_consumption") != consumption
        or consumption.get("schema")
        != "gse342939-ra-bcell-held-adt-consumption/1.0"
        or consumption.get("status")
        != "CONSUMED_EXCLUSIVELY_BEFORE_FIRST_NUMERIC_MATRIX_GET"
        or consumption.get("attempt_sha256") != _sha256(HELD_ADT_ATTEMPT)
        or consumption.get("rerun_permitted") is not False
        or margins.get("implementation_bindings") != _binding_hashes()
        or set(margins.get("donors", {})) != set(expected_donors)
        or margins.get("private_rna_states_path") != _relative(PRIVATE_RNA)
        or margins.get("private_rna_states_sha256")
        != rna["private_rna_states_sha256"]
        or margins.get("private_adt_states_path") != _relative(PRIVATE_ADT)
        or re.fullmatch(
            r"[0-9a-f]{64}", str(margins.get("private_adt_states_sha256"))
        )
        is None
        or margins.get("prediction_stage_authorized") is not True
        or margins.get("joint_table_formation_authorized") is not False
        or margins.get("rerun_permitted") is not False
        or access.get("rna_state_reads") != 0
        or access.get("joint_table_calls") != 0
        or access.get("network_gex_gets") != 0
    ):
        raise PermissionError("public held margins violate the frozen firewall")
    _validate_download_provenance(
        margins.get("held_cite_files"),
        _held_matrix_records(candidate, "cite"),
        "cite",
    )
    for donor_record in candidate["held_donors"]:
        published = margins["donors"][donor_record["donor"]]
        if published.get("stratum") != donor_record["stratum"]:
            raise PermissionError("held disease stratum changed")
        _pair_margins(published)
        rna_visits = rna["donors"][donor_record["donor"]]["visits"]
        for combined, rna_visit in zip(published["visits"], rna_visits):
            if any(
                combined.get(field) != rna_visit.get(field)
                for field in (
                    "visit",
                    "cell_count",
                    "selected_barcode_sha256",
                    "rna_margins",
                    "rna_margins_sha256",
                    "rna_state_sha256",
                )
            ):
                raise PermissionError("held RNA seal changed during ADT reduction")
    return margins, commit, source


def _model_field(model: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "population_mean": np.asarray(model["population_mean"], dtype=float).reshape(
            MARKER_COUNT, MARKER_COUNT
        ),
        "population_change": np.asarray(
            model["population_change"], dtype=float
        ).reshape(MARKER_COUNT, MARKER_COUNT),
    }


def _predict_models(
    models: dict[str, Any], rows: np.ndarray, columns: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    source_mask = np.asarray(models["final_mask"], dtype=bool)
    scored = source_mask[None, ...] & _recipient_margin_support(rows, columns)
    counts = [int(np.count_nonzero(scored[visit])) for visit in range(2)]
    if min(counts) < MINIMUM_COORDINATES:
        raise ProtocolRefusal(
            "HELD_RECIPIENT_MASK_BELOW_256", details={"counts": counts}
        )
    primary = models["primary"]
    primary_config = PrimaryConfig(**primary["configuration"])
    residual = models["selected_residual"]
    residual_config = ResidualConfig(**residual["configuration"])
    common_effect = models["common_effect_cmle"]
    common_config = TransportConfig(**common_effect["configuration"])
    pooled = models["pooled_saturated_poisson"]
    pooled_config = TransportConfig(**pooled["configuration"])
    destroyed = models["destroyed_link"]
    destroyed_config = PrimaryConfig(**destroyed["configuration"])
    visit_agnostic = models["visit_agnostic_primary"]
    visit_agnostic_config = PrimaryConfig(**visit_agnostic["configuration"])
    predictions = {
        "primary": _predict_nch(
            _model_field(primary), rows, columns, primary_config, scored
        ),
        "selected_residual": _predict_residual(
            _model_field(residual), rows, columns, residual_config, scored
        ),
        "common_effect_cmle": _predict_nch(
            _model_field(common_effect), rows, columns, common_config, scored
        ),
        "destroyed_link": _predict_nch(
            _model_field(destroyed), rows, columns, destroyed_config, scored
        ),
        "visit_agnostic_primary": _predict_nch(
            _model_field(visit_agnostic),
            rows,
            columns,
            visit_agnostic_config,
            scored,
        ),
        "independence": _independence(rows, columns),
    }
    poisson_prediction, poisson_certificate = _predict_poisson(
        _model_field(pooled), rows, columns, pooled_config, scored
    )
    predictions["pooled_saturated_poisson"] = poisson_prediction
    if (
        poisson_certificate["maximum_margin_error"] > 1e-10
        or poisson_certificate["maximum_log_odds_error"] > 1e-8
        or poisson_certificate[
            "conditional_noncentral_hypergeometric_reconstruction"
        ]
        is not False
    ):
        raise CouplingEstimationRefusal(
            "held pooled Poisson prediction missed its certificate"
        )
    return scored, predictions, poisson_certificate


def run_prediction(output_path: Path = HELD_PREDICTIONS) -> dict[str, Any]:
    """Predict only from public source fields and public held margins."""

    _require_canonical_path(output_path, HELD_PREDICTIONS, "held prediction")
    if output_path.exists() or SCORE_AUTHORIZATION.exists() or SCORE_ATTEMPT.exists():
        raise FileExistsError("held prediction has already been published or advanced")
    margins, margins_commit, source = _validated_public_margins()
    records = []
    for donor, margin_record in margins["donors"].items():
        rows, columns = _pair_margins(margin_record)
        scored, predictions, poisson_certificate = _predict_models(
            source["models"], rows, columns
        )
        records.append(
            {
                "donor": donor,
                "stratum": margin_record["stratum"],
                "visits": margin_record["visits"],
                "comparison_masks": scored.astype(np.uint8).tolist(),
                "comparison_masks_sha256": _array_sha256(
                    scored.astype(np.uint8)
                ),
                "scored_coordinate_counts": [
                    int(np.count_nonzero(scored[visit])) for visit in range(2)
                ],
                "predictions": {
                    method: values.reshape(2, MARKER_COUNT * MARKER_COUNT, 4).tolist()
                    for method, values in predictions.items()
                },
                "pooled_poisson_prediction_certificate": poisson_certificate,
            }
        )
    payload = {
        "schema": "gse342939-ra-bcell-held-predictions/1.0",
        "status": "HELD_PREDICTIONS_FROZEN_BEFORE_PAIRING",
        "created_at_utc": _timestamp(),
        "source_result_sha256": _sha256(SOURCE_RESULT),
        "held_margins_sha256": _sha256(HELD_MARGINS),
        "held_margins_commit": margins_commit,
        "source_models_sha256": hashlib.sha256(
            json.dumps(source["models"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "held_records": records,
        "access_audit": {
            "private_rna_state_reads": 0,
            "private_adt_state_reads": 0,
            "joint_tables_formed": 0,
            "network_matrix_gets": 0,
        },
        "rerun_permitted": False,
    }
    _write_json_x(output_path, payload)
    return payload


def _validated_public_prediction(
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    margins, margins_commit, source = _validated_public_margins()
    prediction, prediction_commit = _require_public_result(
        PREDICTION_TAG,
        HELD_PREDICTIONS,
        "gse342939-ra-bcell-held-predictions/1.0",
    )
    _require_ancestor(margins_commit, prediction_commit)
    expected_axis = list(margins["donors"])
    observed_axis = [record.get("donor") for record in prediction.get("held_records", [])]
    expected_model_sha = hashlib.sha256(
        json.dumps(source["models"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    access = prediction.get("access_audit", {})
    if (
        prediction.get("status") != "HELD_PREDICTIONS_FROZEN_BEFORE_PAIRING"
        or prediction.get("source_result_sha256") != _sha256(SOURCE_RESULT)
        or prediction.get("held_margins_sha256") != _sha256(HELD_MARGINS)
        or prediction.get("held_margins_commit") != margins_commit
        or prediction.get("source_models_sha256") != expected_model_sha
        or observed_axis != expected_axis
        or access
        != {
            "private_rna_state_reads": 0,
            "private_adt_state_reads": 0,
            "joint_tables_formed": 0,
            "network_matrix_gets": 0,
        }
        or prediction.get("rerun_permitted") is not False
    ):
        raise PermissionError("public held prediction crossed the pairing firewall")
    for published in prediction["held_records"]:
        margin_record = margins["donors"][published["donor"]]
        rows, columns = _pair_margins(margin_record)
        scored, expected, certificate = _predict_models(
            source["models"], rows, columns
        )
        observed_mask = np.asarray(published.get("comparison_masks"), dtype=np.uint8)
        if (
            published.get("stratum") != margin_record["stratum"]
            or published.get("visits") != margin_record["visits"]
            or observed_mask.shape != (2, MARKER_COUNT, MARKER_COUNT)
            or not np.array_equal(observed_mask, scored.astype(np.uint8))
            or published.get("comparison_masks_sha256")
            != _array_sha256(scored.astype(np.uint8))
            or published.get("scored_coordinate_counts")
            != [int(np.count_nonzero(scored[visit])) for visit in range(2)]
            or set(published.get("predictions", {})) != set(expected)
            or published.get("pooled_poisson_prediction_certificate") != certificate
        ):
            raise PermissionError("public prediction record differs from replay")
        for method, expected_table in expected.items():
            observed = np.asarray(published["predictions"][method], dtype=float)
            desired = expected_table.reshape(
                2, MARKER_COUNT * MARKER_COUNT, 4
            )
            if (
                observed.shape != desired.shape
                or not np.isfinite(observed).all()
                or not np.allclose(observed, desired, rtol=1e-12, atol=1e-12)
            ):
                raise PermissionError("public prediction values differ from replay")
    return prediction, prediction_commit, margins, source


def authorize_score(output_path: Path = SCORE_AUTHORIZATION) -> dict[str, Any]:
    """Authorize one offline pairing after public prediction publication."""

    _require_canonical_path(output_path, SCORE_AUTHORIZATION, "score authorization")
    if output_path.exists() or SCORE_ATTEMPT.exists() or SCORE_RESULT.exists():
        raise FileExistsError("score stage has already been authorized or consumed")
    prediction, prediction_commit, margins, _ = _validated_public_prediction()
    payload = {
        "schema": "gse342939-ra-bcell-score-authorization/1.0",
        "status": "AUTHORIZED_AFTER_PUBLIC_PREDICTION_FREEZE",
        "created_at_utc": _timestamp(),
        "source_result_sha256": _sha256(SOURCE_RESULT),
        "held_margins_sha256": _sha256(HELD_MARGINS),
        "held_predictions_sha256": _sha256(HELD_PREDICTIONS),
        "held_predictions_commit": prediction_commit,
        "held_donor_axis": [record["donor"] for record in prediction["held_records"]],
        "private_rna_states_sha256": margins["private_rna_states_sha256"],
        "private_adt_states_sha256": margins["private_adt_states_sha256"],
        "private_state_reads_before_authorization": 0,
        "joint_tables_formed_before_authorization": 0,
        "score_pairing_authorized": True,
        "rerun_permitted": False,
    }
    _write_json_x(output_path, payload)
    return payload


def _validated_score_authorization(
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    prediction, prediction_commit, margins, _ = _validated_public_prediction()
    commit = _require_public_tag(
        SCORE_AUTHORIZATION_TAG, (_relative(SCORE_AUTHORIZATION),)
    )
    _require_ancestor(prediction_commit, commit)
    authorization = _read_json(SCORE_AUTHORIZATION)
    if (
        authorization.get("schema")
        != "gse342939-ra-bcell-score-authorization/1.0"
        or authorization.get("status")
        != "AUTHORIZED_AFTER_PUBLIC_PREDICTION_FREEZE"
        or authorization.get("source_result_sha256") != _sha256(SOURCE_RESULT)
        or authorization.get("held_margins_sha256") != _sha256(HELD_MARGINS)
        or authorization.get("held_predictions_sha256")
        != _sha256(HELD_PREDICTIONS)
        or authorization.get("held_predictions_commit") != prediction_commit
        or authorization.get("held_donor_axis")
        != [record["donor"] for record in prediction["held_records"]]
        or authorization.get("private_rna_states_sha256")
        != margins["private_rna_states_sha256"]
        or authorization.get("private_adt_states_sha256")
        != margins["private_adt_states_sha256"]
        or authorization.get("private_state_reads_before_authorization") != 0
        or authorization.get("joint_tables_formed_before_authorization") != 0
        or authorization.get("score_pairing_authorized") is not True
        or authorization.get("rerun_permitted") is not False
    ):
        raise PermissionError("public score authorization differs from prediction")
    return authorization, commit, prediction, margins


def _load_private_states(
    path: Path,
    expected_sha256: str,
    margins: dict[str, Any],
    modality: str,
) -> dict[str, np.ndarray]:
    if _sha256(path) != expected_sha256:
        raise PermissionError(f"private {modality} states differ from public margins")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & (stat.S_IRWXG | stat.S_IRWXO) or not mode & stat.S_IRUSR:
        raise PermissionError(f"private {modality} states are not owner-only")
    expected_keys = {
        f"{donor}__{visit}" for donor in margins["donors"] for visit in VISITS
    }
    output: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            raise PermissionError(f"private {modality} state axis changed")
        for donor, donor_record in margins["donors"].items():
            for visit_record in donor_record["visits"]:
                visit = visit_record["visit"]
                key = f"{donor}__{visit}"
                states = np.asarray(archive[key])
                expected_state_sha = visit_record[f"{modality}_state_sha256"]
                if (
                    states.shape != (visit_record["cell_count"], MARKER_COUNT)
                    or states.dtype.kind not in "biu"
                    or not np.isin(states, (0, 1)).all()
                    or _array_sha256(states) != expected_state_sha
                ):
                    raise PermissionError(f"private {modality} state array changed")
                output[key] = states.astype(np.uint8, copy=False)
    return output


def _visit_losses(
    observed: np.ndarray, predicted: np.ndarray, scored: np.ndarray
) -> np.ndarray:
    truth = np.asarray(observed, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    mask = np.asarray(scored, dtype=bool)
    values = np.empty(2, dtype=float)
    for visit in range(2):
        selected_truth = truth[visit][mask[visit]]
        selected_estimate = estimate[visit][mask[visit]]
        if len(selected_truth) < MINIMUM_COORDINATES:
            raise CouplingEstimationRefusal("held visit has too few scored coordinates")
        if not np.allclose(
            selected_truth.sum(axis=-1), selected_estimate.sum(axis=-1), atol=1e-8
        ) or not np.allclose(
            selected_truth.sum(axis=-2), selected_estimate.sum(axis=-2), atol=1e-8
        ):
            raise CouplingEstimationRefusal("held prediction changed margins")
        positive = selected_truth > 0
        if (
            np.any(selected_estimate[positive] <= 0.0)
            or not np.isfinite(selected_estimate).all()
        ):
            raise CouplingEstimationRefusal("held prediction assigns invalid mass")
        terms = np.zeros_like(selected_truth)
        terms[positive] = selected_truth[positive] * np.log(
            selected_truth[positive] / selected_estimate[positive]
        )
        totals = selected_truth.sum(axis=(-2, -1))
        values[visit] = float(
            np.mean(2.0 * terms.sum(axis=(-2, -1)) / totals)
        )
    return values


def _held_comparison(
    primary: np.ndarray, comparator: np.ndarray, label: str, confirmatory: bool
) -> dict[str, Any]:
    comparison = _comparison(primary, comparator, label, "held")
    favorable = comparison["favorable_physical_donors"]
    donor_count = comparison["physical_donor_count"]
    sign_p = float(
        sum(
            math.comb(donor_count, value)
            for value in range(favorable, donor_count + 1)
        )
        / (2**donor_count)
    )
    passes = (
        comparison["passes_frozen_requirement"] and sign_p <= 0.025
        if confirmatory
        else None
    )
    comparison.update(
        {
            "exact_one_sided_sign_test_p": sign_p,
            "confirmatory_comparator": confirmatory,
            "passes_frozen_confirmation_requirement": passes,
        }
    )
    return comparison


def _score_predictions(
    prediction: dict[str, Any],
    margins: dict[str, Any],
    rna: dict[str, np.ndarray],
    adt: dict[str, np.ndarray],
    access_audit: dict[str, int] | None = None,
) -> dict[str, Any]:
    methods = (
        "primary",
        "selected_residual",
        "pooled_saturated_poisson",
        "destroyed_link",
        "common_effect_cmle",
        "visit_agnostic_primary",
        "independence",
    )
    donor_count = len(prediction["held_records"])
    donor_losses = {method: np.empty(donor_count, dtype=float) for method in methods}
    visit_losses = {
        method: np.empty((donor_count, 2), dtype=float) for method in methods
    }
    truth_hashes: dict[str, list[str]] = {}
    records = []
    for donor_index, published in enumerate(prediction["held_records"]):
        donor = published["donor"]
        margin_record = margins["donors"][donor]
        rows, columns = _pair_margins(margin_record)
        truth = np.empty((2, MARKER_COUNT, MARKER_COUNT, 2, 2), dtype=np.int64)
        truth_hashes[donor] = []
        for visit_index, visit in enumerate(VISITS):
            key = f"{donor}__{visit}"
            truth[visit_index] = _joint_tables(rna[key], adt[key])
            if access_audit is not None:
                access_audit["joint_table_constructor_calls"] += 1
                access_audit["joint_tables_formed"] += MARKER_COUNT * MARKER_COUNT
            truth_hashes[donor].append(_array_sha256(truth[visit_index]))
        mask = np.asarray(published["comparison_masks"], dtype=bool)
        if (
            not np.array_equal(truth.sum(axis=-1), rows)
            or not np.array_equal(truth.sum(axis=-2), columns)
            or _array_sha256(mask.astype(np.uint8))
            != published["comparison_masks_sha256"]
            or set(published["predictions"]) != set(methods)
        ):
            raise PermissionError("held truth differs from frozen public margins")
        method_records: dict[str, Any] = {}
        for method in methods:
            estimate = np.asarray(published["predictions"][method], dtype=float).reshape(
                2, MARKER_COUNT, MARKER_COUNT, 2, 2
            )
            by_visit = _visit_losses(truth, estimate, mask)
            visit_losses[method][donor_index] = by_visit
            donor_losses[method][donor_index] = float(by_visit.mean())
            method_records[method] = {
                "pre": float(by_visit[0]),
                "post": float(by_visit[1]),
                "donor_equal": float(by_visit.mean()),
            }
        records.append(
            {
                "donor": donor,
                "stratum": published["stratum"],
                "losses": method_records,
            }
        )
    comparisons = {
        "selected_residual": _held_comparison(
            donor_losses["primary"], donor_losses["selected_residual"], "residual", True
        ),
        "pooled_saturated_poisson": _held_comparison(
            donor_losses["primary"],
            donor_losses["pooled_saturated_poisson"],
            "pooled_poisson",
            True,
        ),
        "destroyed_link": _held_comparison(
            donor_losses["primary"], donor_losses["destroyed_link"], "destroyed", True
        ),
        "common_effect_cmle": _held_comparison(
            donor_losses["primary"],
            donor_losses["common_effect_cmle"],
            "common_effect_cmle",
            False,
        ),
    }
    passes = all(
        comparisons[name]["passes_frozen_confirmation_requirement"] is True
        for name in (
            "selected_residual",
            "pooled_saturated_poisson",
            "destroyed_link",
        )
    ) and comparisons["common_effect_cmle"]["passes_frozen_requirement"]
    strata = sorted({record["stratum"] for record in records})
    return {
        "status": "CONFIRMATION_PASS"
        if passes
        else "COMPLETED_HELD_CONFIRMATION_NEGATIVE",
        "passes_frozen_confirmation_gate": bool(passes),
        "held_donor_records": records,
        "donor_equal_mean_losses": {
            method: float(values.mean()) for method, values in donor_losses.items()
        },
        "visit_mean_losses": {
            visit: {
                method: float(visit_losses[method][:, visit_index].mean())
                for method in methods
            }
            for visit_index, visit in enumerate(VISITS)
        },
        "stratum_mean_losses": {
            stratum: {
                method: float(
                    np.mean(
                        [
                            record["losses"][method]["donor_equal"]
                            for record in records
                            if record["stratum"] == stratum
                        ]
                    )
                )
                for method in methods
            }
            for stratum in strata
        },
        "comparisons": comparisons,
        "visit_agnostic_primary_ablation": {
            "primary_mean_loss": float(donor_losses["primary"].mean()),
            "visit_agnostic_mean_loss": float(
                donor_losses["visit_agnostic_primary"].mean()
            ),
            "excluded_from_confirmation_intersection": True,
        },
        "observed_table_sha256": truth_hashes,
        "joint_table_constructor_calls": donor_count * 2,
        "joint_tables_formed": donor_count * 2 * MARKER_COUNT * MARKER_COUNT,
    }


@contextmanager
def _network_disabled() -> Iterator[None]:
    original_open = urllib.request.urlopen
    original_builder = urllib.request.build_opener
    original_socket = socket.socket
    original_create_connection = socket.create_connection

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise PermissionError("network access is disabled during score-held")

    urllib.request.urlopen = forbidden
    urllib.request.build_opener = forbidden
    socket.socket = forbidden
    socket.create_connection = forbidden
    try:
        yield
    finally:
        urllib.request.urlopen = original_open
        urllib.request.build_opener = original_builder
        socket.socket = original_socket
        socket.create_connection = original_create_connection


def score_held(
    attempt_path: Path = SCORE_ATTEMPT,
    output_path: Path = SCORE_RESULT,
    private_rna: Path = PRIVATE_RNA,
    private_adt: Path = PRIVATE_ADT,
) -> dict[str, Any]:
    """Pair sealed held states once, offline, after public authorization."""

    _require_canonical_path(attempt_path, SCORE_ATTEMPT, "score attempt")
    _require_canonical_path(output_path, SCORE_RESULT, "score result")
    _require_canonical_path(private_rna, PRIVATE_RNA, "private RNA states")
    _require_canonical_path(private_adt, PRIVATE_ADT, "private ADT states")
    if attempt_path.exists() or output_path.exists():
        raise FileExistsError("score-held has already been claimed or completed")
    authorization, authorization_commit, prediction, margins = (
        _validated_score_authorization()
    )
    if not private_rna.is_file() or not private_adt.is_file():
        raise FileNotFoundError("sealed held state artifact is absent")
    attempt = {
        "schema": "gse342939-ra-bcell-score-attempt/1.0",
        "status": "CLAIMED_AFTER_PUBLIC_SCORE_AUTHORIZATION",
        "created_at_utc": _timestamp(),
        "score_authorization_sha256": _sha256(SCORE_AUTHORIZATION),
        "score_authorization_commit": authorization_commit,
        "held_predictions_sha256": _sha256(HELD_PREDICTIONS),
        "expected_private_rna_states_sha256": authorization[
            "private_rna_states_sha256"
        ],
        "expected_private_adt_states_sha256": authorization[
            "private_adt_states_sha256"
        ],
        "private_state_read_begins_after_this_record": True,
        "network_access_after_this_record": False,
        "rerun_permitted": False,
    }
    _write_json_x(attempt_path, attempt)
    access_audit = {
        "private_rna_state_reads": 0,
        "private_adt_state_reads": 0,
        "joint_table_constructor_calls": 0,
        "joint_tables_formed": 0,
        "network_calls_after_score_attempt": 0,
        "matrix_opens_after_score_attempt": 0,
    }
    try:
        with _network_disabled():
            rna = _load_private_states(
                private_rna,
                margins["private_rna_states_sha256"],
                margins,
                "rna",
            )
            access_audit["private_rna_state_reads"] = 1
            adt = _load_private_states(
                private_adt,
                margins["private_adt_states_sha256"],
                margins,
                "adt",
            )
            access_audit["private_adt_state_reads"] = 1
            payload = _score_predictions(
                prediction, margins, rna, adt, access_audit
            )
        payload.update(
            {
                "schema": "gse342939-ra-bcell-confirmation-result/1.0",
                "created_at_utc": _timestamp(),
                "score_attempt_sha256": _sha256(attempt_path),
                "score_authorization_sha256": _sha256(SCORE_AUTHORIZATION),
                "score_authorization_commit": authorization_commit,
                "held_predictions_sha256": _sha256(HELD_PREDICTIONS),
                "held_margins_sha256": _sha256(HELD_MARGINS),
                "private_state_sha256": {
                    "rna": margins["private_rna_states_sha256"],
                    "adt": margins["private_adt_states_sha256"],
                },
                "access_audit": access_audit,
                "rerun_permitted": False,
            }
        )
    except BaseException as error:
        payload = {
            "schema": "gse342939-ra-bcell-confirmation-result/1.0",
            "status": "TERMINAL_HELD_SCORE_EXECUTION_REFUSAL",
            "created_at_utc": _timestamp(),
            "passes_frozen_confirmation_gate": False,
            "reason_code": error.code
            if isinstance(error, ProtocolRefusal)
            else type(error).__name__,
            "reason": str(error),
            "reason_details": error.details
            if isinstance(error, ProtocolRefusal)
            else {},
            "score_attempt_sha256": _sha256(attempt_path),
            "score_authorization_sha256": _sha256(SCORE_AUTHORIZATION),
            "score_authorization_commit": authorization_commit,
            "held_predictions_sha256": _sha256(HELD_PREDICTIONS),
            "held_margins_sha256": _sha256(HELD_MARGINS),
            "access_audit": access_audit,
            "rerun_permitted": False,
        }
    _write_json_x(output_path, payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in (
        "claim-source",
        "run-source",
        "claim-held-rna",
        "run-held-rna",
        "claim-held-adt",
        "run-held-adt",
        "predict-held",
        "authorize-score",
        "score-held",
    ):
        commands.add_parser(command)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    action = {
        "claim-source": claim_source,
        "run-source": run_source,
        "claim-held-rna": claim_held_rna,
        "run-held-rna": run_held_rna,
        "claim-held-adt": claim_held_adt,
        "run-held-adt": run_held_adt,
        "predict-held": run_prediction,
        "authorize-score": authorize_score,
        "score-held": score_held,
    }[arguments.command]
    result = action()
    print(json.dumps({"status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
