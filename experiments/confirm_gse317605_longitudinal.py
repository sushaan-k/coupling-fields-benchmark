"""One-shot held-patient GSE317605 longitudinal CITE-seq confirmation.

Metadata and axes were inspected before designation; no count matrix was read.
Every numeric stage is separately claimed, durably journaled, and irreversible.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
import platform
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence
from urllib.request import Request, urlopen

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import gse317605_longitudinal_core as core  # noqa: E402
from mapreg.streamed_gzip_matrix_market import (  # noqa: E402
    GzipMatrixMarketValidationError,
    reduce_gzip_matrix_market,
)


DATA_DIR = ROOT / "data/confirmation/gse317605_longitudinal"
CANDIDATE = DATA_DIR / "candidate_designation_v1.json"
PROTOCOL = DATA_DIR / "protocol_v1.json"
MANIFEST = DATA_DIR / "sample_manifest_v1.json"
ACCESS_HISTORY = DATA_DIR / "axis_access_history_v1.json"
IMPLEMENTATION_FREEZE = DATA_DIR / "pre_access_implementation_v1.json"
PROTOCOL_DOC = ROOT / "docs/GSE317605_LONGITUDINAL_CITESEQ_PROTOCOL_2026-08-30.md"
RUNNER = ROOT / "experiments/confirm_gse317605_longitudinal.py"
CORE = ROOT / "experiments/gse317605_longitudinal_core.py"
TEST = ROOT / "tests/test_gse317605_longitudinal.py"

PUBLIC_ORIGIN = "https://github.com/sushaan-k/coupling-fields-benchmark.git"
PRIVATE_BRIDGE_ROOT = Path("/private/tmp/gse317605-longitudinal-v1-private")
CANDIDATE_TAG = "gse317605-longitudinal-v1-candidate"
IMPLEMENTATION_TAG = "gse317605-longitudinal-v1-implementation"
STAGES = ("calibration", "pilot_gex", "pilot_adt", "held_gex", "held_adt")
ATTEMPT_TAGS = {
    stage: f"gse317605-longitudinal-v1-{stage.replace('_', '-')}-attempt"
    for stage in STAGES
}
RESULT_TAGS = {
    stage: f"gse317605-longitudinal-v1-{stage.replace('_', '-')}-result"
    for stage in STAGES
}

CANDIDATE_SHA256 = "05ba040b3c44c6dc4e9c7e89319a13b23f1ea0d33df4a409e2e29141ffadf844"
PROTOCOL_SHA256 = "b48e96a9b5e6bd4932f972ddbd9e1e1b1662005412a92d1ee8c07df36e16a5e3"
MANIFEST_SHA256 = "2cdbd3182cf41306e6adb71d409049ba8aa07e3f9d3129ecbc444545673dff8a"
ACCESS_HISTORY_SHA256 = (
    "088fb812950310388dd80b27a672922d3b337208ed21407718893de0d20f3865"
)

IMPLEMENTATION_BINDINGS = (
    "data/confirmation/gse317605_longitudinal/candidate_designation_v1.json",
    "data/confirmation/gse317605_longitudinal/protocol_v1.json",
    "data/confirmation/gse317605_longitudinal/sample_manifest_v1.json",
    "data/confirmation/gse317605_longitudinal/axis_access_history_v1.json",
    "docs/GSE317605_LONGITUDINAL_CITESEQ_PROTOCOL_2026-08-30.md",
    "experiments/confirm_gse317605_longitudinal.py",
    "experiments/gse317605_longitudinal_core.py",
    "tests/test_gse317605_longitudinal.py",
    "mapreg/coupling_fields.py",
    "mapreg/heterogeneity_adaptive_coupling.py",
    "mapreg/poisson_loglinear.py",
    "mapreg/streamed_gzip_matrix_market.py",
    "pyproject.toml",
    "requirements.txt",
)

STAGE_PATHS = {
    stage: {
        "attempt": DATA_DIR / f"{stage}_attempt_v1.json",
        "consumption": DATA_DIR / f"{stage}_consumption_v1.json",
        "journal": DATA_DIR / f"{stage}_access_v1.jsonl",
        "result": DATA_DIR
        / {
            "calibration": "calibration_result_v1.json",
            "pilot_gex": "pilot_gex_predictions_v1.json",
            "pilot_adt": "pilot_result_v1.json",
            "held_gex": "held_gex_predictions_v1.json",
            "held_adt": "held_result_v1.json",
        }[stage],
    }
    for stage in STAGES
}
PRIVATE_GEX_BRIDGES = {
    "pilot_gex": PRIVATE_BRIDGE_ROOT / "pilot_gex_bridge_v1.json",
    "held_gex": PRIVATE_BRIDGE_ROOT / "held_gex_bridge_v1.json",
}

STAGE_ROLE = {
    "calibration": "calibration",
    "pilot_gex": "pilot",
    "pilot_adt": "pilot",
    "held_gex": "held",
    "held_adt": "held",
}
STAGE_MODALITIES = {
    "calibration": ("GEX", "ADT"),
    "pilot_gex": ("GEX",),
    "pilot_adt": ("ADT",),
    "held_gex": ("GEX",),
    "held_adt": ("ADT",),
}
EXPECTED_STAGE_PATIENTS = {
    "calibration": ("23", "16", "14", "11", "10", "12", "17"),
    "pilot_gex": ("13", "19", "26"),
    "pilot_adt": ("13", "19", "26"),
    "held_gex": ("24", "27", "22", "25", "15", "18", "20", "21"),
    "held_adt": ("24", "27", "22", "25", "15", "18", "20", "21"),
}
DEFAULT_SCRATCH = Path("/private/tmp/gse317605-longitudinal-v1")
STAGE_LOCK_DIRECTORY = Path("/private/tmp")
SERIES_ARCHIVE = "GSE317605_RAW.tar"
CELL_BUDGET = 192
ADT_HIGH_COUNT = 96
CELL_SALT = "GSE317605-CELL-v2"
ADT_TIE_SALT = "GSE317605-ADT-TIE-v2"
GEX_ROWS = 33_538
ADT_ROWS = 99
GEX_FEATURE_AXIS_SHA256 = (
    "6bb91dd583b8ed7e4d6ea2efb6cb9b103b229573fec7ba2c1f7ba583994a21b1"
)
ADT_FEATURE_AXIS_SHA256 = (
    "95797e25f128965db196858b0abf9a56487894431c5b792e54bbb53ccddfa1da"
)
GEX_SELECTED_ROWS = (
    18652,
    20240,
    4006,
    17796,
    9636,
    2206,
    18577,
    8783,
    20343,
    2300,
    20214,
    7358,
    25807,
    1641,
    17511,
    28465,
)
ADT_SELECTED_ROWS = (
    32, 31, 26, 7, 35, 50, 18, 27, 79, 46, 24, 21, 4, 28, 51, 40
)


class ProtocolRefusal(RuntimeError):
    """A frozen QC, access, or decision condition refused."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


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
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _barcode_axis_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8") + b"\n")
    return digest.hexdigest()


def _feature_axis_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8") + b"\n")
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{number} is not an object")
            rows.append(value)
    return rows


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_x(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)


def _write_private_json_x(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)


def _path_binding(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()


def _write_private_capability(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = secrets.token_bytes(32)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)
    return hashlib.sha256(payload).hexdigest()


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    line = (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        + b"\n"
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        with os.fdopen(descriptor, "ab", closefd=True) as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        pass


def _create_journal(path: Path, stage: str, created_at: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        header = {
            "schema": "gse317605-access-journal/1.0",
            "stage": stage,
            "event": "OPENED_BEFORE_ASSAY_ACCESS",
            "created_at_utc": created_at,
            "one_get_per_file": True,
            "range_requests_permitted": False,
            "automatic_retries_permitted": False,
            "series_archive_permitted": False,
        }
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(header, sort_keys=True, separators=(",", ":")))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _git(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def _require_public_origin() -> None:
    observed = _git("remote", "get-url", "origin").stdout.strip().removesuffix(".git")
    expected = PUBLIC_ORIGIN.removesuffix(".git")
    if observed != expected:
        raise PermissionError("origin differs from the frozen public repository")


def _require_ancestor(ancestor: str, descendant: str) -> None:
    try:
        _git("merge-base", "--is-ancestor", ancestor, descendant)
    except subprocess.CalledProcessError as error:
        raise PermissionError("public freeze ancestry differs") from error


def _remote_tag_ids(tag: str) -> tuple[str, str]:
    _require_public_origin()
    result = _git("ls-remote", "origin", f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}")
    values = {}
    for line in result.stdout.splitlines():
        commit, reference = line.split("\t", 1)
        values[reference] = commit
    tag_object = values.get(f"refs/tags/{tag}")
    peeled = values.get(f"refs/tags/{tag}^{{}}")
    if tag_object is None or peeled is None:
        raise PermissionError(f"annotated public tag {tag} is absent")
    return tag_object, peeled


def _require_public_tag(tag: str, required_paths: Sequence[Path]) -> str:
    if _git("cat-file", "-t", tag).stdout.strip() != "tag":
        raise PermissionError(f"{tag} is not an annotated tag")
    local_object = _git("rev-parse", f"refs/tags/{tag}").stdout.strip()
    local_commit = _git("rev-parse", f"{tag}^{{}}").stdout.strip()
    remote_object, remote_commit = _remote_tag_ids(tag)
    if (local_object, local_commit) != (remote_object, remote_commit):
        raise PermissionError(f"public tag {tag} differs from the local tag")
    for path in required_paths:
        relative = _relative(path)
        _git("cat-file", "-e", f"{local_commit}:{relative}")
        tagged = _git("show", f"{local_commit}:{relative}").stdout.encode("utf-8")
        if hashlib.sha256(tagged).hexdigest() != _sha256(path):
            raise PermissionError(f"{relative} differs from public tag {tag}")
    return local_commit


def _validate_contract() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    expected = {
        CANDIDATE: CANDIDATE_SHA256,
        PROTOCOL: PROTOCOL_SHA256,
        MANIFEST: MANIFEST_SHA256,
        ACCESS_HISTORY: ACCESS_HISTORY_SHA256,
    }
    for path, digest in expected.items():
        if _sha256(path) != digest:
            raise PermissionError(f"{path.name} differs from the frozen digest")
    candidate = _read_json(CANDIDATE)
    protocol = _read_json(PROTOCOL)
    manifest = _read_json(MANIFEST)
    access_history = _read_json(ACCESS_HISTORY)
    if (
        candidate.get("schema") != "gse317605-longitudinal-candidate/1.0"
        or candidate.get("status") != "DESIGNATED_BEFORE_ANY_COUNT_MATRIX_ACCESS"
        or protocol.get("schema") != "gse317605-longitudinal-protocol/1.0"
        or manifest.get("schema") != "gse317605-sample-manifest/1.0"
        or manifest.get("series_archive_permitted") is not False
        or access_history.get("schema") != "gse317605-axis-access-history/1.0"
        or protocol.get("candidate", {}).get("sha256") != CANDIDATE_SHA256
        or protocol.get("sample_manifest", {}).get("sha256") != MANIFEST_SHA256
        or protocol.get("axis_access_history", {}).get("sha256")
        != ACCESS_HISTORY_SHA256
    ):
        raise PermissionError("frozen contract schema or status differs")
    if protocol.get("stage_order") != list(STAGES):
        raise PermissionError("stage order differs")
    if tuple(candidate["time_axis"]) != core.TIMEPOINTS:
        raise PermissionError("time axis differs")
    if (
        candidate.get("cell_contract", {}).get("cell_selection_salt") != CELL_SALT
        or candidate.get("cell_contract", {}).get("adt_tie_salt") != ADT_TIE_SALT
    ):
        raise PermissionError("cell identity salts differ")
    if [row["rna_symbol"] for row in candidate["markers"]] != list(core.MARKERS):
        raise PermissionError("marker axis differs")
    if (
        candidate.get("feature_axes", {}).get("coordinate_convention")
        != "one-based deposited Matrix Market row index"
        or tuple(row["rna_row"] for row in candidate["markers"])
        != GEX_SELECTED_ROWS
        or tuple(row["adt_row"] for row in candidate["markers"])
        != ADT_SELECTED_ROWS
    ):
        raise PermissionError("deposited marker coordinates differ")
    mechanics = protocol.get("access_mechanics", {})
    if (
        "mode-0600" not in mechanics.get("private_bridge", "")
        or "no raw cell identifier" not in mechanics.get("public_privacy", "")
    ):
        raise PermissionError("private bridge or public privacy contract differs")
    patients = {record["patient_id"]: record for record in manifest["patients"]}
    if set(patients) != set(
        sum((list(value) for value in EXPECTED_STAGE_PATIENTS.values()), [])
    ):
        raise PermissionError("patient universe differs")
    replicate_count = 0
    visit_count = 0
    for record in manifest["patients"]:
        for visit in record["replicates"]:
            replicate_count += 1
            for modality in ("GEX", "ADT"):
                for file_record in visit[modality]["files"].values():
                    if (
                        SERIES_ARCHIVE in file_record["name"]
                        or ".tar" in file_record["name"].lower()
                        or ".tar" in file_record["url"].lower()
                    ):
                        raise PermissionError(
                            "bundled archive entered the file contract"
                        )
                    if not file_record["url"].startswith(
                        "https://ftp.ncbi.nlm.nih.gov/"
                    ):
                        raise PermissionError(
                            "file URL is outside the frozen NCBI origin"
                        )
            gex_axis = visit["GEX"]["files"]["barcodes"]
            adt_axis = visit["ADT"]["files"]["barcodes"]
            for axis in (gex_axis, adt_axis):
                if (
                    not isinstance(axis.get("rows"), int)
                    or axis["rows"] < 1
                    or not isinstance(axis.get("duplicate_rows"), int)
                    or not 0 <= axis["duplicate_rows"] < axis["rows"]
                    or len(str(axis.get("compressed_sha256", ""))) != 64
                    or len(str(axis.get("ordered_axis_sha256", ""))) != 64
                ):
                    raise PermissionError("frozen barcode-axis evidence differs")
            if (
                gex_axis["rows"] != adt_axis["rows"]
                or gex_axis["duplicate_rows"] != adt_axis["duplicate_rows"]
                or gex_axis["ordered_axis_sha256"]
                != adt_axis["ordered_axis_sha256"]
            ):
                raise PermissionError("paired barcode-axis evidence differs")
        for _, replicates in _visits(record):
            visit_count += 1
            if sum(
                int(replicate["GEX"]["files"]["barcodes"]["rows"])
                for replicate in replicates
            ) < CELL_BUDGET:
                raise PermissionError("a frozen visit has fewer than 192 paired cells")
    if replicate_count != 84 or visit_count != 68:
        raise PermissionError("frozen replicate or visit count differs")
    return candidate, protocol, manifest


def freeze_implementation() -> dict[str, Any]:
    _validate_contract()
    candidate_commit = _require_public_tag(
        CANDIDATE_TAG, (CANDIDATE, PROTOCOL, MANIFEST, ACCESS_HISTORY)
    )
    implementation_parent = _git("rev-parse", "HEAD").stdout.strip()
    _require_ancestor(candidate_commit, implementation_parent)
    if IMPLEMENTATION_FREEZE.exists():
        raise FileExistsError("implementation freeze already exists")
    if any(
        path.exists()
        for stage_paths in STAGE_PATHS.values()
        for path in stage_paths.values()
    ) or any(path.exists() for path in PRIVATE_GEX_BRIDGES.values()):
        raise PermissionError("a stage artifact predates the implementation freeze")
    missing = [
        relative
        for relative in IMPLEMENTATION_BINDINGS
        if not (ROOT / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"implementation bindings are absent: {missing}")
    payload = {
        "schema": "gse317605-implementation-freeze/1.0",
        "status": "FROZEN_BEFORE_FIRST_COUNT_MATRIX_ACCESS",
        "created_at_utc": _timestamp(),
        "candidate_tag": CANDIDATE_TAG,
        "candidate_public_commit": candidate_commit,
        "implementation_tag": IMPLEMENTATION_TAG,
        "implementation_parent_commit": implementation_parent,
        "runtime_versions": _runtime_versions(),
        "bindings": {
            relative: _sha256(ROOT / relative) for relative in IMPLEMENTATION_BINDINGS
        },
        "count_matrix_requests_before_freeze": 0,
        "series_archive_requests_before_freeze": 0,
        "rerun_permitted": False,
    }
    _write_json_x(IMPLEMENTATION_FREEZE, payload)
    return payload


def _verify_implementation() -> dict[str, Any]:
    _validate_contract()
    value = _read_json(IMPLEMENTATION_FREEZE)
    if (
        value.get("schema") != "gse317605-implementation-freeze/1.0"
        or value.get("status") != "FROZEN_BEFORE_FIRST_COUNT_MATRIX_ACCESS"
        or value.get("implementation_tag") != IMPLEMENTATION_TAG
        or value.get("candidate_tag") != CANDIDATE_TAG
        or value.get("count_matrix_requests_before_freeze") != 0
        or value.get("series_archive_requests_before_freeze") != 0
        or value.get("rerun_permitted") is not False
    ):
        raise PermissionError("implementation freeze differs")
    expected = {
        relative: _sha256(ROOT / relative) for relative in IMPLEMENTATION_BINDINGS
    }
    if value.get("bindings") != expected:
        raise PermissionError("implementation binding hashes differ")
    if value.get("runtime_versions") != _runtime_versions():
        raise PermissionError("runtime versions differ from the implementation freeze")
    candidate_commit = _require_public_tag(
        CANDIDATE_TAG, (CANDIDATE, PROTOCOL, MANIFEST, ACCESS_HISTORY)
    )
    if value.get("candidate_public_commit") != candidate_commit:
        raise PermissionError("candidate public commit differs")
    value["public_commit"] = _require_public_tag(
        IMPLEMENTATION_TAG,
        (
            IMPLEMENTATION_FREEZE,
            *map(lambda value: ROOT / value, IMPLEMENTATION_BINDINGS),
        ),
    )
    _require_ancestor(candidate_commit, value["public_commit"])
    return value


def _dependency(stage: str) -> tuple[Path, str, str] | None:
    mapping = {
        "pilot_gex": (
            STAGE_PATHS["calibration"]["result"],
            RESULT_TAGS["calibration"],
            "CALIBRATION_PASS",
        ),
        "pilot_adt": (
            STAGE_PATHS["pilot_gex"]["result"],
            RESULT_TAGS["pilot_gex"],
            "PREDICTIONS_FROZEN_BEFORE_PILOT_ADT_ACCESS",
        ),
        "held_gex": (
            STAGE_PATHS["pilot_adt"]["result"],
            RESULT_TAGS["pilot_adt"],
            "PILOT_PASS",
        ),
        "held_adt": (
            STAGE_PATHS["held_gex"]["result"],
            RESULT_TAGS["held_gex"],
            "PREDICTIONS_FROZEN_BEFORE_HELD_ADT_ACCESS",
        ),
    }
    return mapping.get(stage)


def _verify_dependency(stage: str) -> dict[str, Any] | None:
    dependency = _dependency(stage)
    if dependency is None:
        return None
    path, tag, status = dependency
    value = _read_json(path)
    if value.get("status") != status or value.get("rerun_permitted") is not False:
        raise PermissionError(f"{stage} dependency did not pass")
    _require_public_tag(tag, (path,))
    return value


def _require_token_outside_scratch(token_path: Path, scratch: Path) -> None:
    token = token_path.resolve()
    root = scratch.resolve()
    if token == root or root in token.parents:
        raise PermissionError("private capability must remain outside scratch")


@contextmanager
def _stage_lock(stage: str) -> Iterable[None]:
    STAGE_LOCK_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = STAGE_LOCK_DIRECTORY / f".gse317605-longitudinal-v1-{stage}.lock"
    with path.open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PermissionError(f"{stage} is already running") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def claim_stage(stage: str, token_path: Path, scratch: Path) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError("unknown stage")
    with _stage_lock(stage):
        implementation = _verify_implementation()
        dependency = _verify_dependency(stage)
        paths = STAGE_PATHS[stage]
        if any(path.exists() for path in paths.values()):
            raise FileExistsError(f"{stage} already has a durable artifact")
        if stage in PRIVATE_GEX_BRIDGES and PRIVATE_GEX_BRIDGES[stage].exists():
            raise FileExistsError(f"{stage} already has a private bridge")
        _require_token_outside_scratch(token_path, scratch)
        token_sha256 = _write_private_capability(token_path)
        created_at = _timestamp()
        attempt = {
            "schema": "gse317605-stage-attempt/1.0",
            "stage": stage,
            "status": "CLAIMED_BEFORE_FIRST_STAGE_FILE_GET",
            "created_at_utc": created_at,
            "implementation_public_commit": implementation["public_commit"],
            "dependency_sha256": (
                _sha256(_dependency(stage)[0]) if dependency is not None else None
            ),
            "token_sha256": token_sha256,
            "scratch_binding_sha256": _path_binding(scratch),
            "attempt_tag": ATTEMPT_TAGS[stage],
            "expected_patients": list(EXPECTED_STAGE_PATIENTS[stage]),
            "allowed_modalities": list(STAGE_MODALITIES[stage]),
            "rerun_permitted": False,
        }
        try:
            _write_json_x(paths["attempt"], attempt)
            _create_journal(paths["journal"], stage, created_at)
        except BaseException:
            token_path.unlink(missing_ok=True)
            paths["journal"].unlink(missing_ok=True)
            paths["attempt"].unlink(missing_ok=True)
            _fsync_directory(paths["attempt"].parent)
            _fsync_directory(token_path.parent)
            raise
        return attempt


def _consume_stage(stage: str, token_path: Path, scratch: Path) -> dict[str, Any]:
    paths = STAGE_PATHS[stage]
    attempt = _read_json(paths["attempt"])
    if paths["consumption"].exists() or paths["result"].exists():
        raise FileExistsError(f"{stage} was already consumed or completed")
    if _read_jsonl(paths["journal"]) != [_read_jsonl(paths["journal"])[0]]:
        raise PermissionError("stage journal contains access beyond its public header")
    _require_public_tag(ATTEMPT_TAGS[stage], (paths["attempt"], paths["journal"]))
    _require_token_outside_scratch(token_path, scratch)
    if _path_binding(scratch) != attempt.get("scratch_binding_sha256"):
        raise PermissionError("scratch differs from the claimed stage")
    if not token_path.is_file() or _sha256(token_path) != attempt.get("token_sha256"):
        raise PermissionError("private capability differs")
    consumption = {
        "schema": "gse317605-stage-consumption/1.0",
        "stage": stage,
        "created_at_utc": _timestamp(),
        "attempt_sha256": _sha256(paths["attempt"]),
        "token_sha256": attempt["token_sha256"],
        "scratch_binding_sha256": _path_binding(scratch),
        "consumed_before_first_file_get": True,
        "rerun_permitted": False,
    }
    _write_json_x(paths["consumption"], consumption)
    token_path.unlink()
    _fsync_directory(token_path.parent)
    _append_jsonl(
        paths["journal"],
        {
            "stage": stage,
            "event": "CAPABILITY_CONSUMED",
            "created_at_utc": _timestamp(),
        },
    )
    return consumption


def _open_url(request: Request):
    return urlopen(request, timeout=180)


def _download(
    stage: str,
    file_record: Mapping[str, Any],
    destination: Path,
    *,
    patient_id: str,
    timepoint: str,
    replicate: str,
    modality: str,
    kind: str,
) -> dict[str, Any]:
    name = str(file_record["name"])
    url = str(file_record["url"])
    expected_bytes = int(file_record["bytes"])
    if (
        SERIES_ARCHIVE in name
        or ".tar" in name.lower()
        or ".tar" in url.lower()
        or not url.startswith("https://ftp.ncbi.nlm.nih.gov/")
    ):
        raise PermissionError("file is outside the per-sample frozen contract")
    journal = STAGE_PATHS[stage]["journal"]
    event = {
        "stage": stage,
        "patient_id": patient_id,
        "timepoint": timepoint,
        "replicate": replicate,
        "modality": modality,
        "kind": kind,
        "name": name,
    }
    _append_jsonl(
        journal, {**event, "event": "FILE_GET_STARTED", "created_at_utc": _timestamp()}
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        url, headers={"User-Agent": "coupling-fields-gse317605-v1"}, method="GET"
    )
    digest = hashlib.sha256()
    observed_bytes = 0
    try:
        with _open_url(request) as response:
            status = response.getcode()
            final_url = response.geturl()
            if status != 200 or final_url != url:
                raise ProtocolRefusal("HTTP_IDENTITY_OR_STATUS_DIFFERS")
            descriptor = os.open(
                destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            with os.fdopen(descriptor, "wb") as output:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    output.write(block)
                    digest.update(block)
                    observed_bytes += len(block)
                output.flush()
                os.fsync(output.fileno())
        _fsync_directory(destination.parent)
        if observed_bytes != expected_bytes:
            raise ProtocolRefusal("FILE_BYTE_COUNT_DIFFERS")
    except BaseException as error:
        failure = {
            "exception_class": type(error).__name__,
            "bytes_received": observed_bytes,
        }
        if isinstance(error, ProtocolRefusal):
            failure["refusal_code"] = error.code
        if isinstance(error, OSError) and error.errno is not None:
            failure["errno"] = int(error.errno)
        _append_jsonl(
            journal,
            {
                **event,
                "event": "FILE_GET_FAILED",
                "created_at_utc": _timestamp(),
                **failure,
            },
        )
        raise
    audit = {
        "bytes": observed_bytes,
        "sha256": digest.hexdigest(),
        "http_status": 200,
        "final_url": url,
    }
    _append_jsonl(
        journal,
        {
            **event,
            "event": "FILE_GET_FINISHED",
            "created_at_utc": _timestamp(),
            **audit,
        },
    )
    return audit


def _delete_download(stage: str, path: Path, event: Mapping[str, Any]) -> None:
    existed = path.exists()
    path.unlink(missing_ok=True)
    _fsync_directory(path.parent)
    _append_jsonl(
        STAGE_PATHS[stage]["journal"],
        {
            "stage": stage,
            "event": "FILE_DELETED",
            "created_at_utc": _timestamp(),
            "name": path.name,
            "body_existed": existed,
            **event,
        },
    )


def _barcodes(path: Path) -> tuple[tuple[str, ...], dict[str, Any]]:
    compressed_sha256 = _sha256(path)
    with gzip.open(path, "rt", encoding="utf-8", newline=None) as stream:
        values = tuple(line.rstrip("\r\n") for line in stream)
    if not values or any(not value for value in values):
        raise ProtocolRefusal("BARCODE_AXIS_INVALID")
    multiplicity = Counter(values)
    return values, {
        "rows": len(values),
        "duplicate_rows": int(sum(count - 1 for count in multiplicity.values())),
        "duplicated_values": int(sum(count > 1 for count in multiplicity.values())),
        "compressed_sha256": compressed_sha256,
        "ordered_axis_sha256": _barcode_axis_sha256(values),
    }


def _features(path: Path, modality: str) -> dict[str, Any]:
    if modality not in {"GEX", "ADT"}:
        raise ValueError("unknown feature modality")
    with gzip.open(path, "rt", encoding="utf-8", newline=None) as stream:
        values = tuple(line.rstrip("\r\n") for line in stream)
    expected_rows = GEX_ROWS if modality == "GEX" else ADT_ROWS
    expected_sha256 = (
        GEX_FEATURE_AXIS_SHA256 if modality == "GEX" else ADT_FEATURE_AXIS_SHA256
    )
    observed_sha256 = _feature_axis_sha256(values)
    if (
        len(values) != expected_rows
        or any(not value for value in values)
        or observed_sha256 != expected_sha256
    ):
        raise ProtocolRefusal("FEATURE_AXIS_DIFFERS_FROM_FROZEN_REFERENCE")
    return {
        "rows": len(values),
        "compressed_sha256": _sha256(path),
        "ordered_axis_sha256": observed_sha256,
    }


def _cell_rank(
    patient: str, timepoint: str, replicate: str, column: int, barcode: str
) -> str:
    payload = "\0".join(
        (CELL_SALT, patient, timepoint, replicate, str(column), barcode)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _adt_tie_rank(
    patient: str,
    timepoint: str,
    marker: str,
    replicate: str,
    column: int,
    barcode: str,
) -> str:
    payload = "\0".join(
        (ADT_TIE_SALT, patient, timepoint, marker, replicate, str(column), barcode)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _selected_cells(
    patient: str,
    timepoint: str,
    axes: Mapping[str, Sequence[str]],
) -> tuple[tuple[str, int, str], ...]:
    candidates = [
        (replicate, column, barcode)
        for replicate, barcodes in axes.items()
        for column, barcode in enumerate(barcodes, start=1)
    ]
    if len(candidates) < CELL_BUDGET:
        raise ProtocolRefusal("VISIT_HAS_FEWER_THAN_192_PAIRED_CELLS")
    return tuple(
        sorted(
            candidates,
            key=lambda value: (
                _cell_rank(patient, timepoint, value[0], value[1], value[2]),
                value[0],
                value[1],
                value[2],
            ),
        )[:CELL_BUDGET]
    )


def _patient_records(manifest: Mapping[str, Any], role: str) -> list[dict[str, Any]]:
    order = {
        "calibration": EXPECTED_STAGE_PATIENTS["calibration"],
        "pilot": EXPECTED_STAGE_PATIENTS["pilot_gex"],
        "held": EXPECTED_STAGE_PATIENTS["held_gex"],
    }[role]
    by_id = {record["patient_id"]: record for record in manifest["patients"]}
    return [by_id[patient] for patient in order]


def _stage_file_records(
    stage: str, manifest: Mapping[str, Any]
) -> tuple[tuple[str, str, str, str, Mapping[str, Any]], ...]:
    if stage not in STAGES:
        raise ValueError("unknown stage")
    rows = []
    for patient in _patient_records(manifest, STAGE_ROLE[stage]):
        for timepoint, replicates in _visits(patient):
            for replicate in replicates:
                for modality in STAGE_MODALITIES[stage]:
                    for kind in ("features", "barcodes", "matrix"):
                        rows.append(
                            (
                                str(patient["patient_id"]),
                                timepoint,
                                str(replicate["replicate"]),
                                modality,
                                replicate[modality]["files"][kind],
                            )
                        )
    return tuple(rows)


def _visits(record: Mapping[str, Any]) -> list[tuple[str, list[Mapping[str, Any]]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for replicate in record["replicates"]:
        grouped.setdefault(str(replicate["timepoint"]), []).append(replicate)
    expected = tuple(record["timepoints"])
    if tuple(grouped) != expected:
        raise PermissionError("manifest visit order differs")
    return [(timepoint, grouped[timepoint]) for timepoint in expected]


def _reduce_visit_modality(
    stage: str,
    scratch: Path,
    patient: str,
    timepoint: str,
    replicates: Sequence[Mapping[str, Any]],
    modality: str,
    *,
    expected_selection: Sequence[Mapping[str, Any]] | None = None,
    expected_barcode_axes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    visit_root = scratch / stage / patient / timepoint / modality
    visit_root.mkdir(parents=True, exist_ok=False)
    feature_audits: dict[str, dict[str, Any]] = {}
    barcode_axes: dict[str, tuple[str, ...]] = {}
    barcode_audits: dict[str, dict[str, Any]] = {}
    try:
        for replicate_record in replicates:
            replicate = str(replicate_record["replicate"])
            sample = replicate_record[modality]
            feature_record = sample["files"]["features"]
            feature_path = visit_root / feature_record["name"]
            feature_event = {
                "patient_id": patient,
                "timepoint": timepoint,
                "replicate": replicate,
                "modality": modality,
                "kind": "features",
            }
            try:
                feature_download = _download(
                    stage,
                    feature_record,
                    feature_path,
                    patient_id=patient,
                    timepoint=timepoint,
                    replicate=replicate,
                    modality=modality,
                    kind="features",
                )
                feature_audits[replicate] = {
                    **_features(feature_path, modality),
                    "download": feature_download,
                }
            finally:
                _delete_download(stage, feature_path, feature_event)

            file_record = sample["files"]["barcodes"]
            path = visit_root / file_record["name"]
            event = {
                "patient_id": patient,
                "timepoint": timepoint,
                "replicate": replicate,
                "modality": modality,
                "kind": "barcodes",
            }
            try:
                download = _download(
                    stage,
                    file_record,
                    path,
                    patient_id=patient,
                    timepoint=timepoint,
                    replicate=replicate,
                    modality=modality,
                    kind="barcodes",
                )
                axis, audit = _barcodes(path)
                audit["download"] = download
                if (
                    audit["rows"] != file_record.get("rows")
                    or audit["duplicate_rows"] != file_record.get("duplicate_rows")
                    or audit["compressed_sha256"]
                    != file_record.get("compressed_sha256")
                    or audit["ordered_axis_sha256"]
                    != file_record.get("ordered_axis_sha256")
                ):
                    raise ProtocolRefusal(
                        "BARCODE_AXIS_DIFFERS_FROM_FROZEN_REFERENCE"
                    )
                barcode_axes[replicate] = axis
                barcode_audits[replicate] = audit
                if expected_barcode_axes is not None and audit[
                    "ordered_axis_sha256"
                ] != expected_barcode_axes.get(replicate):
                    raise ProtocolRefusal("GEX_ADT_BARCODE_AXIS_DIFFERS")
            finally:
                _delete_download(stage, path, event)

        if expected_selection is None:
            selected = _selected_cells(patient, timepoint, barcode_axes)
        else:
            selected = tuple(
                (
                    str(value["replicate"]),
                    int(value["column"]),
                    str(value["barcode"]),
                )
                for value in expected_selection
            )
            if len(selected) != CELL_BUDGET or len(set(selected)) != CELL_BUDGET:
                raise ProtocolRefusal("FROZEN_SELECTED_CELL_AXIS_INVALID")
            if any(
                column < 1
                or column > len(barcode_axes.get(replicate, ()))
                or barcode_axes[replicate][column - 1] != barcode
                for replicate, column, barcode in selected
            ):
                raise ProtocolRefusal("FROZEN_SELECTED_CELL_MISSING_FROM_MODALITY")
        selected_set = set(selected)
        rows = GEX_SELECTED_ROWS if modality == "GEX" else ADT_SELECTED_ROWS
        expected_rows = GEX_ROWS if modality == "GEX" else ADT_ROWS
        count_by_cell: dict[tuple[str, int, str], np.ndarray] = {}
        matrix_audits: dict[str, dict[str, Any]] = {}
        for replicate_record in replicates:
            replicate = str(replicate_record["replicate"])
            sample = replicate_record[modality]
            file_record = sample["files"]["matrix"]
            path = visit_root / file_record["name"]
            event = {
                "patient_id": patient,
                "timepoint": timepoint,
                "replicate": replicate,
                "modality": modality,
                "kind": "matrix",
            }
            current = [
                (index + 1, barcode)
                for index, barcode in enumerate(barcode_axes[replicate])
                if (replicate, index + 1, barcode) in selected_set
            ]
            parse_columns = [column for column, _ in current]
            if not parse_columns:
                # The full stream is still validated; this sentinel is discarded.
                parse_columns = [1]
            try:
                download = _download(
                    stage,
                    file_record,
                    path,
                    patient_id=patient,
                    timepoint=timepoint,
                    replicate=replicate,
                    modality=modality,
                    kind="matrix",
                )
                block, audit = reduce_gzip_matrix_market(
                    path,
                    expected_shape=(expected_rows, len(barcode_axes[replicate])),
                    selected_rows=rows,
                    selected_columns=parse_columns,
                    allow_integral_real=True,
                )
                for output_column, (input_column, barcode) in enumerate(current):
                    count_by_cell[(replicate, input_column, barcode)] = block[
                        :, output_column
                    ].copy()
                matrix_audits[replicate] = {**asdict(audit), "download": download}
            finally:
                _delete_download(stage, path, event)
        if set(count_by_cell) != selected_set:
            raise ProtocolRefusal("REDUCED_CELL_AXIS_DIFFERS")
        counts = np.asarray([count_by_cell[cell] for cell in selected], dtype=np.int64)
        if counts.shape != (CELL_BUDGET, len(core.MARKERS)):
            raise ProtocolRefusal("REDUCED_COUNT_BLOCK_SHAPE_DIFFERS")
        return {
            "patient_id": patient,
            "timepoint": timepoint,
            "selected_cells": [
                {"replicate": replicate, "column": column, "barcode": barcode}
                for replicate, column, barcode in selected
            ],
            "selected_cell_axis_sha256": _canonical_json_sha256(selected),
            "barcode_axes": {
                replicate: audit["ordered_axis_sha256"]
                for replicate, audit in barcode_audits.items()
            },
            "feature_audit": feature_audits,
            "barcode_audit": barcode_audits,
            "matrix_audit": matrix_audits,
            "counts": counts,
            "counts_sha256": _array_sha256(counts),
        }
    finally:
        shutil.rmtree(visit_root, ignore_errors=True)


def _acquire_modality(
    stage: str,
    scratch: Path,
    manifest: Mapping[str, Any],
    modality: str,
    *,
    frozen_gex_visits: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if modality not in STAGE_MODALITIES.get(stage, ()):
        raise PermissionError(f"{modality} is forbidden during {stage}")
    records = _patient_records(manifest, STAGE_ROLE[stage])
    visits = []
    for record in records:
        patient = str(record["patient_id"])
        for timepoint, replicates in _visits(record):
            expected = (
                frozen_gex_visits.get((patient, timepoint))
                if frozen_gex_visits is not None
                else None
            )
            visits.append(
                _reduce_visit_modality(
                    stage,
                    scratch,
                    patient,
                    timepoint,
                    replicates,
                    modality,
                    expected_selection=(
                        expected["selected_cells"] if expected else None
                    ),
                    expected_barcode_axes=(
                        expected["barcode_axes"] if expected else None
                    ),
                )
            )
    return visits


def _gex_private_visit(value: Mapping[str, Any]) -> dict[str, Any]:
    counts = np.asarray(value["counts"], dtype=np.int64)
    states = (counts > 0).astype(np.int8)
    return {
        key: value[key]
        for key in (
            "patient_id",
            "timepoint",
            "selected_cells",
            "selected_cell_axis_sha256",
            "barcode_axes",
            "feature_audit",
            "barcode_audit",
            "matrix_audit",
            "counts_sha256",
        )
    } | {
        "rna_states": states.tolist(),
        "rna_states_sha256": _array_sha256(states),
        "rna_high_margins": states.sum(axis=0).astype(int).tolist(),
    }


def _gex_public_visit(value: Mapping[str, Any]) -> dict[str, Any]:
    private = _gex_private_visit(value)
    return {
        key: value[key]
        for key in (
            "patient_id",
            "timepoint",
            "selected_cell_axis_sha256",
            "barcode_axes",
            "feature_audit",
            "barcode_audit",
            "matrix_audit",
            "counts_sha256",
        )
    } | {
        "rna_states_sha256": private["rna_states_sha256"],
        "rna_high_margins": private["rna_high_margins"],
    }


def _write_gex_bridge(
    stage: str, visits: Sequence[Mapping[str, Any]], models_sha256: str
) -> dict[str, Any]:
    if stage not in PRIVATE_GEX_BRIDGES:
        raise ValueError("only a GEX prediction stage has a private bridge")
    private_visits = [_gex_private_visit(value) for value in visits]
    payload = {
        "schema": "gse317605-private-gex-bridge/1.0",
        "stage": stage,
        "patient_order": list(EXPECTED_STAGE_PATIENTS[stage]),
        "models_sha256": models_sha256,
        "gex_visits": private_visits,
    }
    path = PRIVATE_GEX_BRIDGES[stage]
    _write_private_json_x(path, payload)
    return {
        "schema": payload["schema"],
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _load_gex_bridge(
    prediction: Mapping[str, Any], prediction_stage: str
) -> list[dict[str, Any]]:
    if prediction_stage not in PRIVATE_GEX_BRIDGES:
        raise ValueError("unknown private GEX bridge")
    path = PRIVATE_GEX_BRIDGES[prediction_stage]
    expected = prediction.get("private_gex_bridge")
    if (
        not isinstance(expected, Mapping)
        or expected.get("schema") != "gse317605-private-gex-bridge/1.0"
        or not path.is_file()
        or path.stat().st_size != expected.get("bytes")
        or _sha256(path) != expected.get("sha256")
    ):
        raise PermissionError("private GEX bridge differs from the public binding")
    value = _read_json(path)
    if (
        value.get("schema") != expected["schema"]
        or value.get("stage") != prediction_stage
        or value.get("patient_order")
        != list(EXPECTED_STAGE_PATIENTS[prediction_stage])
        or value.get("models_sha256") != prediction.get("models_sha256")
    ):
        raise PermissionError("private GEX bridge lineage differs")
    visits = value.get("gex_visits")
    if not isinstance(visits, list):
        raise PermissionError("private GEX visit payload is absent")
    for visit in visits:
        states = np.asarray(visit.get("rna_states"), dtype=np.int8)
        if (
            states.shape != (CELL_BUDGET, len(core.MARKERS))
            or not np.isin(states, (0, 1)).all()
            or _array_sha256(states) != visit.get("rna_states_sha256")
            or states.sum(axis=0).astype(int).tolist()
            != visit.get("rna_high_margins")
        ):
            raise PermissionError("private GEX state payload differs")
    return visits


def _adt_states(value: Mapping[str, Any]) -> np.ndarray:
    counts = np.asarray(value["counts"], dtype=np.int64)
    selected = value["selected_cells"]
    patient = str(value["patient_id"])
    timepoint = str(value["timepoint"])
    states = np.zeros_like(counts, dtype=np.int8)
    for marker_index, marker in enumerate(core.ADT_MARKERS):
        order = sorted(
            range(len(counts)),
            key=lambda index: (
                -int(counts[index, marker_index]),
                _adt_tie_rank(
                    patient,
                    timepoint,
                    marker,
                    str(selected[index]["replicate"]),
                    int(selected[index]["column"]),
                    str(selected[index]["barcode"]),
                ),
            ),
        )
        states[np.asarray(order[:ADT_HIGH_COUNT]), marker_index] = 1
    if not np.all(states.sum(axis=0) == ADT_HIGH_COUNT):
        raise AssertionError("ADT high-state count differs")
    return states


def _joint_panels(
    gex_visits: Sequence[Mapping[str, Any]], adt_visits: Sequence[Mapping[str, Any]]
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[dict[str, Any]]]:
    gex_by_key = {(row["patient_id"], row["timepoint"]): row for row in gex_visits}
    adt_by_key = {(row["patient_id"], row["timepoint"]): row for row in adt_visits}
    if tuple(gex_by_key) != tuple(adt_by_key):
        raise ProtocolRefusal("GEX_ADT_VISIT_AXIS_DIFFERS")
    tables = []
    destroyed = []
    patients = []
    timepoints = []
    audits = []
    for key, gex in gex_by_key.items():
        adt = adt_by_key[key]
        if (
            gex["selected_cells"] != adt["selected_cells"]
            or gex["selected_cell_axis_sha256"] != adt["selected_cell_axis_sha256"]
        ):
            raise ProtocolRefusal("GEX_ADT_SELECTED_CELL_AXIS_DIFFERS")
        rna = np.asarray(gex["rna_states"], dtype=np.int8)
        protein = _adt_states(adt)
        current = core.joint_binary_tables(rna, protein)
        shifted = core.destroyed_link_tables(rna, protein)
        tables.append(current)
        destroyed.append(shifted)
        patients.append(str(key[0]))
        timepoints.append(str(key[1]))
        audits.append(
            {
                "patient_id": key[0],
                "timepoint": key[1],
                "tables_sha256": _array_sha256(current),
                "destroyed_tables_sha256": _array_sha256(shifted),
                "rna_states_sha256": gex["rna_states_sha256"],
                "gex_counts_sha256": gex["counts_sha256"],
                "gex_feature_audit": gex["feature_audit"],
                "gex_barcode_audit": gex["barcode_audit"],
                "gex_matrix_audit": gex["matrix_audit"],
                "adt_states_sha256": _array_sha256(protein),
                "adt_counts_sha256": adt["counts_sha256"],
                "adt_feature_audit": adt["feature_audit"],
                "adt_barcode_audit": adt["barcode_audit"],
                "adt_matrix_audit": adt["matrix_audit"],
            }
        )
    return np.asarray(tables), np.asarray(destroyed), patients, timepoints, audits


def _fields(models: Mapping[str, Any]) -> dict[str, np.ndarray]:
    values = {
        method: np.asarray(field, dtype=float)
        for method, field in models["fields"].items()
    }
    if set(values) != set(core.MANDATORY_METHODS):
        raise PermissionError("model fields differ from the mandatory methods")
    for method, field in values.items():
        if core._array_sha256(field) != models["field_sha256"].get(method):
            raise PermissionError("model field hash differs")
    return values


def _descriptive_fields(models: Mapping[str, Any]) -> dict[str, np.ndarray]:
    values = {
        method: np.asarray(field, dtype=float)
        for method, field in models["descriptive_fields"].items()
    }
    if set(values) != set(core.DESCRIPTIVE_METHODS):
        raise PermissionError("descriptive model fields differ from the frozen methods")
    for method, field in values.items():
        if core._array_sha256(field) != models["descriptive_field_sha256"].get(method):
            raise PermissionError("descriptive model field hash differs")
    return values


def _terminal_refusal(stage: str, code: str) -> dict[str, Any]:
    paths = STAGE_PATHS[stage]
    if paths["result"].exists():
        return _read_json(paths["result"])
    payload = {
        "schema": "gse317605-stage-result/1.0",
        "stage": stage,
        "status": "TERMINAL_REFUSAL",
        "refusal_code": code,
        "created_at_utc": _timestamp(),
        "attempt_sha256": _sha256(paths["attempt"]),
        "consumption_sha256": _sha256(paths["consumption"]),
        "access_journal_sha256": _sha256(paths["journal"]),
        "rerun_permitted": False,
    }
    _write_json_x(paths["result"], payload)
    return payload


def _terminal_error_code(error: BaseException) -> str:
    if isinstance(error, ProtocolRefusal):
        return error.code
    if isinstance(error, GzipMatrixMarketValidationError):
        return "MATRIX_MARKET_VALIDATION_REFUSAL"
    if isinstance(error, core.EstimationRefusal):
        return "ESTIMATION_CERTIFICATE_REFUSAL"
    if isinstance(error, OSError) and error.errno is not None:
        return f"INFRASTRUCTURE_{type(error).__name__}_ERRNO_{int(error.errno)}"
    return f"UNEXPECTED_{type(error).__name__}"


def _run_calibration(scratch: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    gex_raw = _acquire_modality("calibration", scratch, manifest, "GEX")
    gex = [_gex_private_visit(value) for value in gex_raw]
    gex_by_key = {(row["patient_id"], row["timepoint"]): row for row in gex}
    adt = _acquire_modality(
        "calibration", scratch, manifest, "ADT", frozen_gex_visits=gex_by_key
    )
    tables, destroyed, patients, timepoints, audits = _joint_panels(gex, adt)
    selection = core.select_calibration_models(tables, destroyed, patients, timepoints)
    payload: dict[str, Any] = {
        "schema": "gse317605-calibration-result/1.0",
        "stage": "calibration",
        "status": "CALIBRATION_PASS"
        if selection["gate"]["passes"]
        else "CALIBRATION_FAIL",
        "created_at_utc": _timestamp(),
        "patient_order": list(EXPECTED_STAGE_PATIENTS["calibration"]),
        "visit_patient_axis": patients,
        "visit_timepoint_axis": timepoints,
        "tables": tables.tolist(),
        "destroyed_tables": destroyed.tolist(),
        "table_panel_sha256": _array_sha256(tables),
        "destroyed_panel_sha256": _array_sha256(destroyed),
        "reduction_audit": audits,
        "selection": selection,
        "pilot_matrix_requests": 0,
        "held_matrix_requests": 0,
        "rerun_permitted": False,
    }
    if selection["gate"]["passes"]:
        payload["models"] = core.fit_frozen_models(
            tables,
            destroyed,
            timepoints,
            selection["selected_primary"],
            selection["selected_graph_zero"],
            selection["selected_temporal_zero"],
            selection["selected_structure_zero"],
            selection["selected_poisson_ridge"],
            selection["selected_poisson_transport"],
        )
    return payload


def _prediction_payload(
    stage: str,
    scratch: Path,
    manifest: Mapping[str, Any],
    upstream: Mapping[str, Any],
) -> dict[str, Any]:
    models = upstream["models"] if stage == "pilot_gex" else upstream["promoted_models"]
    _fields(models)
    _descriptive_fields(models)
    models_sha256 = _canonical_json_sha256(models)
    raw_visits = _acquire_modality(stage, scratch, manifest, "GEX")
    private_bridge = _write_gex_bridge(stage, raw_visits, models_sha256)
    gex = [_gex_public_visit(value) for value in raw_visits]
    return {
        "schema": "gse317605-gex-predictions/1.0",
        "stage": stage,
        "status": (
            "PREDICTIONS_FROZEN_BEFORE_PILOT_ADT_ACCESS"
            if stage == "pilot_gex"
            else "PREDICTIONS_FROZEN_BEFORE_HELD_ADT_ACCESS"
        ),
        "created_at_utc": _timestamp(),
        "patient_order": list(EXPECTED_STAGE_PATIENTS[stage]),
        "gex_visits": gex,
        "models": models,
        "models_sha256": models_sha256,
        "private_gex_bridge": private_bridge,
        "public_artifact_contains_raw_cell_identifiers": False,
        "public_artifact_contains_cell_level_state_vectors": False,
        "adt_matrix_requests": 0,
        "rerun_permitted": False,
    }


def _run_pilot_adt(
    scratch: Path, manifest: Mapping[str, Any], prediction: Mapping[str, Any]
) -> dict[str, Any]:
    gex = _load_gex_bridge(prediction, "pilot_gex")
    gex_by_key = {(row["patient_id"], row["timepoint"]): row for row in gex}
    adt = _acquire_modality(
        "pilot_adt", scratch, manifest, "ADT", frozen_gex_visits=gex_by_key
    )
    tables, destroyed, patients, timepoints, audits = _joint_panels(gex, adt)
    losses = core.losses_from_fields(
        tables, patients, timepoints, _fields(prediction["models"])
    )
    descriptive_losses = core.losses_from_fields(
        tables, patients, timepoints, _descriptive_fields(prediction["models"])
    )
    gate = core.pilot_gate(losses, EXPECTED_STAGE_PATIENTS["pilot_adt"])
    payload: dict[str, Any] = {
        "schema": "gse317605-pilot-result/1.0",
        "stage": "pilot_adt",
        "status": "PILOT_PASS" if gate["passes"] else "PILOT_FAIL",
        "created_at_utc": _timestamp(),
        "patient_order": list(EXPECTED_STAGE_PATIENTS["pilot_adt"]),
        "visit_patient_axis": patients,
        "visit_timepoint_axis": timepoints,
        "tables": tables.tolist(),
        "destroyed_tables": destroyed.tolist(),
        "table_panel_sha256": _array_sha256(tables),
        "destroyed_panel_sha256": _array_sha256(destroyed),
        "reduction_audit": audits,
        "losses": losses,
        "descriptive_losses": descriptive_losses,
        "gate": gate,
        "held_matrix_requests": 0,
        "rerun_permitted": False,
    }
    if gate["passes"]:
        calibration = _read_json(STAGE_PATHS["calibration"]["result"])
        calibration_tables = np.asarray(calibration["tables"], dtype=np.int64)
        calibration_destroyed = np.asarray(
            calibration["destroyed_tables"], dtype=np.int64
        )
        selection = calibration["selection"]
        payload["promoted_models"] = core.fit_frozen_models(
            np.concatenate((calibration_tables, tables)),
            np.concatenate((calibration_destroyed, destroyed)),
            [*calibration["visit_timepoint_axis"], *timepoints],
            selection["selected_primary"],
            selection["selected_graph_zero"],
            selection["selected_temporal_zero"],
            selection["selected_structure_zero"],
            selection["selected_poisson_ridge"],
            selection["selected_poisson_transport"],
        )
        payload["promoted_source_patients"] = [
            *EXPECTED_STAGE_PATIENTS["calibration"],
            *EXPECTED_STAGE_PATIENTS["pilot_adt"],
        ]
    return payload


def _run_held_adt(
    scratch: Path, manifest: Mapping[str, Any], prediction: Mapping[str, Any]
) -> dict[str, Any]:
    gex = _load_gex_bridge(prediction, "held_gex")
    gex_by_key = {(row["patient_id"], row["timepoint"]): row for row in gex}
    adt = _acquire_modality(
        "held_adt", scratch, manifest, "ADT", frozen_gex_visits=gex_by_key
    )
    tables, destroyed, patients, timepoints, audits = _joint_panels(gex, adt)
    losses = core.losses_from_fields(
        tables, patients, timepoints, _fields(prediction["models"])
    )
    descriptive_losses = core.losses_from_fields(
        tables, patients, timepoints, _descriptive_fields(prediction["models"])
    )
    completeness = {
        "24": "complete",
        "27": "complete",
        "22": "complete",
        "25": "complete",
        "15": "complete",
        "18": "partial",
        "20": "partial",
        "21": "partial",
    }
    gate = core.held_gate(
        losses, EXPECTED_STAGE_PATIENTS["held_adt"], completeness
    )
    return {
        "schema": "gse317605-held-result/1.0",
        "stage": "held_adt",
        "status": "HELD_CONFIRMATION_PASS"
        if gate["passes"]
        else "HELD_CONFIRMATION_FAIL",
        "created_at_utc": _timestamp(),
        "patient_order": list(EXPECTED_STAGE_PATIENTS["held_adt"]),
        "visit_patient_axis": patients,
        "visit_timepoint_axis": timepoints,
        "table_panel_sha256": _array_sha256(tables),
        "destroyed_panel_sha256": _array_sha256(destroyed),
        "reduction_audit": audits,
        "losses": losses,
        "descriptive_losses": descriptive_losses,
        "gate": gate,
        "rerun_permitted": False,
    }


def run_stage(stage: str, token_path: Path, scratch: Path) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError("unknown stage")
    with _stage_lock(stage):
        _, _, manifest = _validate_contract()
        _verify_implementation()
        dependency = _verify_dependency(stage)
        try:
            _consume_stage(stage, token_path, scratch)
            scratch.mkdir(parents=True, exist_ok=True)
            if stage == "calibration":
                payload = _run_calibration(scratch, manifest)
            elif stage in {"pilot_gex", "held_gex"}:
                if dependency is None:
                    raise AssertionError("prediction stage lacks dependency")
                payload = _prediction_payload(stage, scratch, manifest, dependency)
            elif stage == "pilot_adt":
                if dependency is None:
                    raise AssertionError("pilot score lacks predictions")
                payload = _run_pilot_adt(scratch, manifest, dependency)
            else:
                if dependency is None:
                    raise AssertionError("held score lacks predictions")
                payload = _run_held_adt(scratch, manifest, dependency)
            events = _read_jsonl(STAGE_PATHS[stage]["journal"])
            _validate_access_ledger(stage, events, manifest, complete=True)
            expected_file_count = len(_expected_stage_file_keys(stage, manifest))
            payload["access_ledger"] = {
                "expected_files": expected_file_count,
                "started_files": sum(
                    event.get("event") == "FILE_GET_STARTED" for event in events
                ),
                "finished_files": sum(
                    event.get("event") == "FILE_GET_FINISHED" for event in events
                ),
                "failed_files": sum(
                    event.get("event") == "FILE_GET_FAILED" for event in events
                ),
                "deleted_files": sum(
                    event.get("event") == "FILE_DELETED" for event in events
                ),
                "exact_manifest_reconciliation_passes": True,
            }
            payload["attempt_sha256"] = _sha256(STAGE_PATHS[stage]["attempt"])
            payload["consumption_sha256"] = _sha256(STAGE_PATHS[stage]["consumption"])
            payload["access_journal_sha256"] = _sha256(STAGE_PATHS[stage]["journal"])
            _write_json_x(STAGE_PATHS[stage]["result"], payload)
            return payload
        except BaseException as error:
            if not STAGE_PATHS[stage]["consumption"].is_file():
                raise
            return _terminal_refusal(stage, _terminal_error_code(error))
        finally:
            shutil.rmtree(scratch / stage, ignore_errors=True)


def _replay_decision(stage: str, value: Mapping[str, Any]) -> dict[str, Any] | None:
    if stage == "calibration" and value.get("status") in {
        "CALIBRATION_PASS",
        "CALIBRATION_FAIL",
    }:
        return core.calibration_gate(
            value["selection"]["losses"], EXPECTED_STAGE_PATIENTS["calibration"]
        )
    if stage == "pilot_adt" and value.get("status") in {"PILOT_PASS", "PILOT_FAIL"}:
        return core.pilot_gate(value["losses"], EXPECTED_STAGE_PATIENTS["pilot_adt"])
    if stage == "held_adt" and value.get("status") in {
        "HELD_CONFIRMATION_PASS",
        "HELD_CONFIRMATION_FAIL",
    }:
        return core.held_gate(
            value["losses"],
            EXPECTED_STAGE_PATIENTS["held_adt"],
            {
                patient: (
                    "complete"
                    if patient in {"24", "27", "22", "25", "15"}
                    else "partial"
                )
                for patient in EXPECTED_STAGE_PATIENTS["held_adt"]
            },
        )
    return None


def _access_file_key(event: Mapping[str, Any]) -> tuple[str, ...]:
    fields = ("patient_id", "timepoint", "replicate", "modality", "kind", "name")
    if any(not isinstance(event.get(field), str) for field in fields):
        raise PermissionError("access journal file identity is incomplete")
    return tuple(str(event[field]) for field in fields)


def _expected_stage_file_keys(
    stage: str, manifest: Mapping[str, Any]
) -> set[tuple[str, ...]]:
    expected: set[tuple[str, ...]] = set()
    for patient in _patient_records(manifest, STAGE_ROLE[stage]):
        for timepoint, replicates in _visits(patient):
            for replicate in replicates:
                for modality in STAGE_MODALITIES[stage]:
                    for kind in ("features", "barcodes", "matrix"):
                        file_record = replicate[modality]["files"][kind]
                        key = (
                            str(patient["patient_id"]),
                            timepoint,
                            str(replicate["replicate"]),
                            modality,
                            kind,
                            str(file_record["name"]),
                        )
                        if key in expected:
                            raise PermissionError("manifest repeats a stage file identity")
                        expected.add(key)
    return expected


def _validate_access_ledger(
    stage: str,
    events: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    complete: bool,
) -> None:
    expected = _expected_stage_file_keys(stage, manifest)
    file_labels = (
        "FILE_GET_STARTED",
        "FILE_GET_FINISHED",
        "FILE_GET_FAILED",
        "FILE_DELETED",
    )
    observed: dict[tuple[str, ...], Counter[str]] = {}
    positions: dict[tuple[str, ...], dict[str, int]] = {}
    for position, event in enumerate(events):
        label = str(event.get("event", ""))
        if label not in file_labels:
            continue
        key = _access_file_key(event)
        if key not in expected:
            raise PermissionError("access journal contains an undesignated file")
        observed.setdefault(key, Counter())[label] += 1
        positions.setdefault(key, {}).setdefault(label, position)

    consumed_positions = [
        index
        for index, event in enumerate(events)
        if event.get("event") == "CAPABILITY_CONSUMED"
    ]
    if len(consumed_positions) != 1:
        raise PermissionError("access journal capability record differs")
    first_get = next(
        (
            index
            for index, event in enumerate(events)
            if event.get("event") == "FILE_GET_STARTED"
        ),
        len(events),
    )
    if consumed_positions[0] >= first_get:
        raise PermissionError("access began before capability consumption")

    for key, counts in observed.items():
        if any(counts[label] > 1 for label in file_labels):
            raise PermissionError("access journal repeats a file event")
        if (
            counts["FILE_GET_FINISHED"] + counts["FILE_GET_FAILED"]
            > counts["FILE_GET_STARTED"]
        ):
            raise PermissionError("access journal completes a file before its GET")
        order = positions[key]
        started = order.get("FILE_GET_STARTED", -1)
        for label in ("FILE_GET_FINISHED", "FILE_GET_FAILED"):
            if label in order and started >= order[label]:
                raise PermissionError("access journal file event order differs")
        terminal = max(
            order.get("FILE_GET_FINISHED", -1),
            order.get("FILE_GET_FAILED", -1),
        )
        if "FILE_DELETED" in order and terminal >= order["FILE_DELETED"]:
            raise PermissionError("access journal deletion order differs")

    if complete:
        if set(observed) != expected:
            raise PermissionError("successful stage did not access the manifest file set")
        for counts in observed.values():
            if (
                counts["FILE_GET_STARTED"] != 1
                or counts["FILE_GET_FINISHED"] != 1
                or counts["FILE_GET_FAILED"] != 0
                or counts["FILE_DELETED"] != 1
            ):
                raise PermissionError("successful stage file ledger is incomplete")


def validate_stage(stage: str) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError("unknown stage")
    _, _, manifest = _validate_contract()
    implementation = _verify_implementation()
    _verify_dependency(stage)
    paths = STAGE_PATHS[stage]
    for name in ("attempt", "consumption", "journal", "result"):
        if not paths[name].is_file():
            raise PermissionError(f"{stage} {name} artifact is absent")
    value = _read_json(paths["result"])
    if (
        value.get("stage") != stage
        or value.get("rerun_permitted") is not False
        or value.get("attempt_sha256") != _sha256(paths["attempt"])
        or value.get("consumption_sha256") != _sha256(paths["consumption"])
        or value.get("access_journal_sha256") != _sha256(paths["journal"])
    ):
        raise PermissionError("stage artifact hashes or identity differ")
    events = _read_jsonl(paths["journal"])
    if not events or events[0].get("event") != "OPENED_BEFORE_ASSAY_ACCESS":
        raise PermissionError("access journal header differs")
    if any(event.get("stage") != stage for event in events):
        raise PermissionError("access journal crosses stage boundaries")
    allowed = set(STAGE_MODALITIES[stage])
    if any(
        event.get("modality") not in allowed
        for event in events
        if event.get("event", "").startswith("FILE_")
    ):
        raise PermissionError("stage journal contains a forbidden modality")
    complete_statuses = {
        "CALIBRATION_PASS",
        "CALIBRATION_FAIL",
        "PREDICTIONS_FROZEN_BEFORE_PILOT_ADT_ACCESS",
        "PILOT_PASS",
        "PILOT_FAIL",
        "PREDICTIONS_FROZEN_BEFORE_HELD_ADT_ACCESS",
        "HELD_CONFIRMATION_PASS",
        "HELD_CONFIRMATION_FAIL",
    }
    _validate_access_ledger(
        stage,
        events,
        manifest,
        complete=value.get("status") in complete_statuses,
    )
    if value.get("status") in complete_statuses:
        expected_count = len(_expected_stage_file_keys(stage, manifest))
        expected_ledger = {
            "expected_files": expected_count,
            "started_files": expected_count,
            "finished_files": expected_count,
            "failed_files": 0,
            "deleted_files": expected_count,
            "exact_manifest_reconciliation_passes": True,
        }
        if value.get("access_ledger") != expected_ledger:
            raise PermissionError("serialized access-ledger certificate differs")
    replay = _replay_decision(stage, value)
    if stage == "calibration" and value.get("status") in {
        "CALIBRATION_PASS",
        "CALIBRATION_FAIL",
    }:
        core.replay_calibration_selection(value["selection"])
    if (
        replay is not None
        and replay != value.get("gate")
        and replay != value.get("selection", {}).get("gate")
    ):
        raise PermissionError("terminal decision does not replay")
    public_commit = _require_public_tag(
        RESULT_TAGS[stage],
        (paths["attempt"], paths["consumption"], paths["journal"], paths["result"]),
    )
    attempt_commit = _require_public_tag(ATTEMPT_TAGS[stage], (paths["attempt"],))
    _require_ancestor(implementation["public_commit"], public_commit)
    _require_ancestor(attempt_commit, public_commit)
    dependency = _dependency(stage)
    if dependency is not None:
        dependency_commit = _require_public_tag(dependency[1], (dependency[0],))
        _require_ancestor(dependency_commit, public_commit)
    return {
        "stage": stage,
        "status": value["status"],
        "events": len(events),
        "decision_replayed": replay is not None,
        "result_sha256": _sha256(paths["result"]),
        "public_result_commit": public_commit,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze-implementation")
    for stage in STAGES:
        claim = subparsers.add_parser(f"claim-{stage.replace('_', '-')}")
        claim.add_argument("--token", type=Path, required=True)
        claim.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
        run = subparsers.add_parser(f"run-{stage.replace('_', '-')}")
        run.add_argument("--token", type=Path, required=True)
        run.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
        subparsers.add_parser(f"validate-{stage.replace('_', '-')}")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "freeze-implementation":
        value = freeze_implementation()
    else:
        action, raw_stage = arguments.command.split("-", 1)
        stage = raw_stage.replace("-", "_")
        if action == "claim":
            value = claim_stage(stage, arguments.token, arguments.scratch)
        elif action == "run":
            value = run_stage(stage, arguments.token, arguments.scratch)
        else:
            value = validate_stage(stage)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
