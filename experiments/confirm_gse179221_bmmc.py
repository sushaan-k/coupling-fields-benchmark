"""One-shot GSE179221 BMMC CITE-seq held-donor campaign.

The source stage is the only executable stage until a public source result
passes its frozen gate. Source and held H5 files are fetched one donor at a
time, reduced through the 10x CSC representation, and deleted immediately.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Iterator
import urllib.parse
import urllib.request

import h5py
import numpy as np
import scipy
from scipy.optimize import brentq

from experiments import confirm_gse309593_held_batches as engine
from mapreg.heterogeneity_adaptive_coupling import (
    CouplingEstimationRefusal,
    expected_binary_table_from_log_odds,
    signed_deviance_coordinate,
    signed_pearson_coordinate,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/confirmation/gse179221_bmmc"
DESIGNATION = DATA_DIR / "candidate_designation_v1.json"
PROTOCOL = ROOT / "docs/GSE179221_BMMC_CITESEQ_HELD_DONOR_PROTOCOL_2026-08-29.md"
AMENDMENT = DATA_DIR / "pre_access_implementation_amendment_v1.json"
RUNNER = ROOT / "experiments/confirm_gse179221_bmmc.py"
TEST = ROOT / "tests/test_gse179221_bmmc_confirmation.py"

SOURCE_ATTEMPT = DATA_DIR / "source_attempt_v1.json"
SOURCE_CONSUMPTION = DATA_DIR / "source_consumption_v1.json"
SOURCE_RESULT = ROOT / "results/development/gse179221_bmmc_source_v1.json"
HELD_ATTEMPT = DATA_DIR / "held_margin_attempt_v1.json"
HELD_CONSUMPTION = DATA_DIR / "held_margin_consumption_v1.json"
HELD_MARGINS = ROOT / "results/gse179221_bmmc_held_margins_v1.json"
HELD_PREDICTIONS = ROOT / "results/gse179221_bmmc_predictions_v1.json"
SCORE_AUTHORIZATION = DATA_DIR / "score_authorization_v1.json"
SCORE_ATTEMPT = DATA_DIR / "score_attempt_v1.json"
SCORE_RESULT = ROOT / "results/gse179221_bmmc_confirmation_v1.json"
PRIVATE_RNA = DATA_DIR / "private_held_rna_states_v1.npz"
PRIVATE_ADT = DATA_DIR / "private_held_adt_states_v1.npz"
DEFAULT_SCRATCH = Path("/private/tmp/gse179221-bmmc-v1")

PUBLIC_ORIGIN = "https://github.com/sushaan-k/coupling-fields-benchmark.git"
CANDIDATE_TAG = "gse179221-bmmc-v1-candidate"
AMENDMENT_TAG = "gse179221-bmmc-v1-pre-access-amendment"
IMPLEMENTATION_TAG = "gse179221-bmmc-v1-implementation"
SOURCE_ATTEMPT_TAG = "gse179221-bmmc-v1-source-attempt"
SOURCE_TAG = "gse179221-bmmc-v1-source"
HELD_ATTEMPT_TAG = "gse179221-bmmc-v1-held-margin-attempt"
MARGINS_TAG = "gse179221-bmmc-v1-margins"
PREDICTION_TAG = "gse179221-bmmc-v1-prediction"
SCORE_AUTHORIZATION_TAG = "gse179221-bmmc-v1-score-authorization"

PANEL = (
    ("CD3D", "CD3"),
    ("NCAM1", "CD56"),
    ("CD19", "CD19"),
    ("CD14", "CD14"),
    ("FCGR3A", "CD16"),
    ("MS4A1", "CD20"),
    ("CD27", "CD27"),
    ("CD38", "CD38"),
    ("CD79B", "CD79b"),
)
MARKER_COUNT = len(PANEL)
CELL_BUDGET = 512
MINIMUM_COORDINATES = 64
MINIMUM_DETECTED_GENES = 200
MAXIMUM_MITOCHONDRIAL_FRACTION = 0.10
MAXIMUM_RNA_UMIS = 70_000
CELL_SALT = "GSE179221-BMMC-CELL-v1"
ADT_TIE_SALT = "GSE179221-BMMC-ADT-v1"
BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 20260830
GRAPH_NEIGHBORS = 2
HETEROGENEITY_GRID = (0.1, 1.0, 10.0)
RIDGE_GRID = (0.01, 0.1)
GRAPH_GRID = (0.0, 0.03, 0.3)
TRANSPORT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
RESIDUAL_FAMILIES = ("pearson", "root_deviance")
AMENDMENT_SHA256 = "04631dcb2cd89c6f29a2b1d24103f9d2740cc5f9ce78157a555618a738a3d25f"
ADT_ALIASES = (
    ("CD3",),
    ("NCAM", "CD56"),
    ("CD19",),
    ("CD14",),
    ("CD16",),
    ("CD20",),
    ("CD27",),
    ("CD38",),
    ("CD79b (Ig\u03b2)", "CD79b"),
)

IMPLEMENTATION_BINDINGS = (
    "experiments/confirm_gse179221_bmmc.py",
    "tests/test_gse179221_bmmc_confirmation.py",
    "experiments/confirm_gse309593_held_batches.py",
    "mapreg/hierarchical_conditional_coupling.py",
    "mapreg/common_effect_conditional.py",
    "mapreg/heterogeneity_adaptive_coupling.py",
)


class ProtocolRefusal(RuntimeError):
    """Terminal refusal carrying a stable machine-readable reason code."""

    def __init__(
        self, code: str, message: str | None = None, details: dict[str, Any] | None = None
    ):
        super().__init__(message or code)
        self.code = code
        self.details = details or {}


@contextmanager
def _engine_contract() -> Iterator[None]:
    previous = {
        "MARKER_COUNT": engine.MARKER_COUNT,
        "CELL_BUDGET": engine.CELL_BUDGET,
        "MINIMUM_INFORMATIVE_ENTITIES": engine.MINIMUM_INFORMATIVE_ENTITIES,
    }
    engine.MARKER_COUNT = MARKER_COUNT
    engine.CELL_BUDGET = CELL_BUDGET
    engine.MINIMUM_INFORMATIVE_ENTITIES = MINIMUM_COORDINATES
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(engine, name, value)


def _timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _write_json_x(path: Path, value: dict[str, Any]) -> None:
    serialized = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(
            stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
        )
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _claim_consumption(path: Path, schema: str, attempt_path: Path) -> dict[str, Any]:
    payload = {
        "schema": schema,
        "status": "CONSUMED_EXCLUSIVELY_BEFORE_FIRST_H5_GET",
        "created_at_utc": _timestamp(),
        "attempt_sha256": _sha256(attempt_path),
        "process_id": os.getpid(),
        "rerun_permitted": False,
    }
    _write_json_x(path, payload)
    return payload


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
        digest.update(str(value).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _require_canonical_path(observed: Path, expected: Path, label: str) -> None:
    if observed.resolve() != expected.resolve():
        raise PermissionError(f"{label} must use its canonical campaign path")


def _binding_hashes(paths: Iterable[str] = IMPLEMENTATION_BINDINGS) -> dict[str, str]:
    return {path: _sha256(ROOT / path) for path in paths}


def _runtime_record() -> dict[str, Any]:
    thread_variables = (
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
            "h5py": h5py.__version__,
        },
        "hdf5": {
            "runtime": h5py.version.hdf5_version,
            "built_against": ".".join(
                str(value) for value in h5py.version.hdf5_built_version_tuple
            ),
        },
        "thread_environment": {
            name: os.environ.get(name) for name in thread_variables
        },
    }


def _remote_tag_ids(tag: str) -> tuple[str, str]:
    lines = subprocess.run(
        [
            "git",
            "ls-remote",
            PUBLIC_ORIGIN,
            f"refs/tags/{tag}",
            f"refs/tags/{tag}^{{}}",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    refs = {
        fields[1]: fields[0]
        for line in lines
        if len(fields := line.split()) == 2
    }
    tag_object = refs.get(f"refs/tags/{tag}")
    commit = refs.get(f"refs/tags/{tag}^{{}}")
    if tag_object is None or commit is None:
        raise PermissionError(f"public annotated tag {tag} is absent")
    return tag_object, commit


def _require_public_tag(tag: str, paths: Iterable[str]) -> str:
    object_type = subprocess.run(
        ["git", "cat-file", "-t", tag],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if object_type != "tag":
        raise PermissionError(f"local tag {tag} is not annotated")
    local_object = subprocess.run(
        ["git", "rev-parse", f"refs/tags/{tag}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    local_commit = subprocess.run(
        ["git", "rev-parse", f"{tag}^{{}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    remote_object, remote_commit = _remote_tag_ids(tag)
    if (local_object, local_commit) != (remote_object, remote_commit):
        raise PermissionError(f"public tag {tag} differs from the local tag")
    for relative in paths:
        published = subprocess.run(
            ["git", "show", f"{tag}:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if published != (ROOT / relative).read_bytes():
            raise PermissionError(f"{relative} differs from public tag {tag}")
    return local_commit


def _require_ancestor(ancestor: str, descendant: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise PermissionError("public implementation does not descend from the candidate")


def _verify_public_freezes() -> dict[str, str]:
    candidate_commit = _require_public_tag(
        CANDIDATE_TAG,
        (_relative(DESIGNATION),),
    )
    amendment_commit = _require_public_tag(
        AMENDMENT_TAG,
        (_relative(DESIGNATION), _relative(PROTOCOL), _relative(AMENDMENT)),
    )
    _require_ancestor(candidate_commit, amendment_commit)
    implementation_paths = [*IMPLEMENTATION_BINDINGS, _relative(AMENDMENT)]
    implementation_commit = _require_public_tag(
        IMPLEMENTATION_TAG,
        implementation_paths,
    )
    _require_ancestor(amendment_commit, implementation_commit)
    return {
        "candidate_tag": CANDIDATE_TAG,
        "candidate_commit": candidate_commit,
        "amendment_tag": AMENDMENT_TAG,
        "amendment_commit": amendment_commit,
        "implementation_tag": IMPLEMENTATION_TAG,
        "implementation_commit": implementation_commit,
    }


def _candidate(path: Path = DESIGNATION) -> dict[str, Any]:
    value = _read_json(path)
    if (
        value.get("schema") != "gse179221-bmmc-citeseq-candidate-designation/1.0"
        or value.get("status")
        != "FROZEN_FROM_PUBLIC_METADATA_BEFORE_ANY_H5_BODY_ACCESS"
        or value.get("ordered_cognate_candidates")
        != [[rna, adt] for rna, adt in PANEL]
        or len(value.get("source_files", [])) != 8
        or len(value.get("held_files", [])) != 10
        or value.get("metadata_bindings", {}).get("all_donor_tar_forbidden") is not True
    ):
        raise PermissionError("candidate designation differs from the frozen campaign")
    source = {record["gsm"] for record in value["source_files"]}
    held = {record["gsm"] for record in value["held_files"]}
    if source & held or len(source | held) != 18:
        raise PermissionError("candidate source and held axes overlap")
    return value


def _amendment(path: Path = AMENDMENT) -> dict[str, Any]:
    value = _read_json(path)
    aliases = value.get("feature_contract", {}).get(
        "ordered_cognates_and_exact_adt_aliases"
    )
    observed = [
        (record.get("rna"), record.get("adt_target"), tuple(record.get("aliases", [])))
        for record in aliases or []
    ]
    expected = [
        (rna, adt, alias)
        for (rna, adt), alias in zip(PANEL, ADT_ALIASES)
    ]
    if (
        _sha256(path) != AMENDMENT_SHA256
        or value.get("schema")
        != "gse179221-bmmc-citeseq-pre-access-implementation-amendment/1.0"
        or value.get("status") != "FROZEN_BEFORE_ANY_H5_BODY_ACCESS"
        or observed != expected
        or value.get("graph_contract", {}).get("neighbors") != GRAPH_NEIGHBORS
        or value.get("fold_specific_comparison_masks", {}).get(
            "minimum_scored_coordinates_per_donor"
        )
        != MINIMUM_COORDINATES
    ):
        raise PermissionError("pre-access implementation amendment differs")
    return value


def claim_source(attempt_path: Path = SOURCE_ATTEMPT) -> dict[str, Any]:
    """Claim the source run before any donor H5 GET is permitted."""

    if attempt_path.exists() or SOURCE_RESULT.exists():
        raise FileExistsError("source attempt is already claimed or consumed")
    candidate = _candidate()
    _amendment()
    public = _verify_public_freezes()
    payload = {
        "schema": "gse179221-bmmc-source-attempt/1.0",
        "status": "CLAIMED_ONE_SHOT_BEFORE_ANY_H5_GET",
        "created_at_utc": _timestamp(),
        "public_freezes": public,
        "candidate_sha256": _sha256(DESIGNATION),
        "protocol_sha256": _sha256(PROTOCOL),
        "implementation_amendment_sha256": AMENDMENT_SHA256,
        "implementation_bindings": _binding_hashes(),
        "runtime": _runtime_record(),
        "source_gsm_axis": [record["gsm"] for record in candidate["source_files"]],
        "source_filename_axis": [
            record["filename"] for record in candidate["source_files"]
        ],
        "h5_get_begins_after_this_record": True,
        "only_eight_source_urls_authorized": True,
        "held_url_get_authorized": False,
        "all_donor_tar_get_authorized": False,
        "rerun_permitted": False,
    }
    _write_json_x(attempt_path, payload)
    return payload


def _validate_source_attempt(
    path: Path, enforce_current_runtime: bool = True
) -> tuple[dict[str, Any], dict[str, str]]:
    attempt = _read_json(path)
    candidate = _candidate()
    _amendment()
    public = _verify_public_freezes()
    if (
        attempt.get("schema") != "gse179221-bmmc-source-attempt/1.0"
        or attempt.get("status") != "CLAIMED_ONE_SHOT_BEFORE_ANY_H5_GET"
        or attempt.get("public_freezes") != public
        or attempt.get("candidate_sha256") != _sha256(DESIGNATION)
        or attempt.get("protocol_sha256") != _sha256(PROTOCOL)
        or attempt.get("implementation_amendment_sha256") != AMENDMENT_SHA256
        or attempt.get("implementation_bindings") != _binding_hashes()
        or not isinstance(attempt.get("runtime"), dict)
        or (
            enforce_current_runtime
            and attempt.get("runtime") != _runtime_record()
        )
        or attempt.get("source_gsm_axis")
        != [record["gsm"] for record in candidate["source_files"]]
        or attempt.get("source_filename_axis")
        != [record["filename"] for record in candidate["source_files"]]
        or attempt.get("h5_get_begins_after_this_record") is not True
        or attempt.get("only_eight_source_urls_authorized") is not True
        or attempt.get("held_url_get_authorized") is not False
        or attempt.get("all_donor_tar_get_authorized") is not False
        or attempt.get("rerun_permitted") is not False
    ):
        raise PermissionError("source attempt differs from the public frozen implementation")
    attempt_commit = _require_public_tag(SOURCE_ATTEMPT_TAG, (_relative(path),))
    _require_ancestor(public["implementation_commit"], attempt_commit)
    public = {
        **public,
        "source_attempt_tag": SOURCE_ATTEMPT_TAG,
        "source_attempt_commit": attempt_commit,
    }
    return candidate, public


def _decode_axis(dataset: h5py.Dataset) -> list[str]:
    values = np.asarray(dataset[()]).reshape(-1)
    decoded = [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]
    if any(not value for value in decoded):
        raise ProtocolRefusal("H5_AXIS_EMPTY_VALUE")
    return decoded


def _integer_values(values: np.ndarray, code: str) -> np.ndarray:
    numeric = np.asarray(values, dtype=float)
    if (
        not np.isfinite(numeric).all()
        or np.any(numeric < 0)
        or not np.array_equal(numeric, np.rint(numeric))
    ):
        raise ProtocolRefusal(code)
    return np.rint(numeric).astype(np.int64)


def _read_positions(dataset: h5py.Dataset, positions: np.ndarray) -> np.ndarray:
    axis = np.asarray(positions, dtype=np.int64)
    if axis.ndim != 1 or (len(axis) and np.any(axis[1:] <= axis[:-1])):
        raise ValueError("H5 positions must be unique and increasing")
    if len(axis) == 0:
        return np.empty(0, dtype=dataset.dtype)
    chunks: list[np.ndarray] = []
    start = int(axis[0])
    previous = start
    for value in axis[1:]:
        current = int(value)
        if current != previous + 1:
            chunks.append(np.asarray(dataset[start : previous + 1]))
            start = current
        previous = current
    chunks.append(np.asarray(dataset[start : previous + 1]))
    return np.concatenate(chunks)


def _selected_cells(donor: str, barcodes: list[str], eligible: np.ndarray) -> list[int]:
    if len(barcodes) != len(set(barcodes)):
        raise ProtocolRefusal("BARCODE_AXIS_DUPLICATED")
    candidates = [int(index) for index in np.asarray(eligible, dtype=np.int64)]
    if len(candidates) < CELL_BUDGET:
        raise ProtocolRefusal("RNA_QC_SUPPORT_BELOW_512")
    chosen = sorted(
        candidates,
        key=lambda index: (
            hashlib.sha256(
                f"{CELL_SALT}\0{donor}\0{barcodes[index]}".encode()
            ).hexdigest(),
            barcodes[index],
        ),
    )[:CELL_BUDGET]
    return sorted(chosen)


def _resolve_panel(
    names: list[str], feature_types: list[str]
) -> tuple[list[int], list[int]]:
    if len(names) != len(feature_types):
        raise ProtocolRefusal("TENX_FEATURE_AXIS_LENGTH_MISMATCH")

    def exact(names_allowed: tuple[str, ...], feature_type: str) -> int:
        matches = [
            index
            for index, (observed, observed_type) in enumerate(
                zip(names, feature_types)
            )
            if observed in names_allowed and observed_type == feature_type
        ]
        if len(matches) != 1:
            raise ProtocolRefusal("COGNATE_AXIS_NOT_EXACTLY_UNIQUE")
        return matches[0]

    return (
        [exact((rna,), "Gene Expression") for rna, _ in PANEL],
        [exact(aliases, "Antibody Capture") for aliases in ADT_ALIASES],
    )


def _midrank_adt(
    counts: np.ndarray, barcodes: list[str], donor: str
) -> np.ndarray:
    values = _integer_values(counts, "ADT_COUNTS_INVALID")
    if values.shape != (CELL_BUDGET, MARKER_COUNT):
        raise ValueError("ADT panel has the wrong shape")
    states = np.zeros(values.shape, dtype=np.uint8)
    for marker in range(MARKER_COUNT):
        order = sorted(
            range(CELL_BUDGET),
            key=lambda index: (
                int(values[index, marker]),
                hashlib.sha256(
                    f"{ADT_TIE_SALT}\0{donor}\0{marker}\0{barcodes[index]}".encode()
                ).hexdigest(),
                barcodes[index],
            ),
        )
        states[np.asarray(order[CELL_BUDGET // 2 :], dtype=np.int64), marker] = 1
    if not np.all(states.sum(axis=0) == CELL_BUDGET // 2):
        raise AssertionError("ADT midrank split changed the frozen 256/256 margin")
    return states


def _destroyed_adt(
    states: np.ndarray, barcodes: list[str], donor: str
) -> np.ndarray:
    values = np.asarray(states, dtype=np.uint8)
    if values.shape != (CELL_BUDGET, MARKER_COUNT):
        raise ValueError("ADT states have the wrong shape")
    order = np.asarray(
        sorted(
            range(CELL_BUDGET),
            key=lambda index: (
                hashlib.sha256(
                    f"{CELL_SALT}\0{donor}\0{barcodes[index]}".encode()
                ).hexdigest(),
                barcodes[index],
            ),
        ),
        dtype=np.int64,
    )
    destroyed = np.empty_like(values)
    destroyed[order] = values[np.roll(order, 1)]
    if not np.array_equal(destroyed.sum(axis=0), values.sum(axis=0)):
        raise AssertionError("destroyed link changed an ADT margin")
    if sorted(map(tuple, destroyed.tolist())) != sorted(map(tuple, values.tolist())):
        raise AssertionError("destroyed link changed complete ADT state profiles")
    return destroyed


def _joint_tables(rna: np.ndarray, adt: np.ndarray) -> np.ndarray:
    first = np.asarray(rna, dtype=np.uint8)
    second = np.asarray(adt, dtype=np.uint8)
    if first.shape != (CELL_BUDGET, MARKER_COUNT) or second.shape != first.shape:
        raise ValueError("binary state panels have the wrong shape")
    tables = np.empty((MARKER_COUNT, MARKER_COUNT, 2, 2), dtype=np.int64)
    for rna_index in range(MARKER_COUNT):
        for adt_index in range(MARKER_COUNT):
            tables[rna_index, adt_index] = np.bincount(
                2 * first[:, rna_index] + second[:, adt_index], minlength=4
            ).reshape(2, 2)
    return tables


def _margin_support(tables: np.ndarray) -> np.ndarray:
    values = np.asarray(tables)
    rows = values.sum(axis=-1)
    columns = values.sum(axis=-2)
    total = values.sum(axis=(-2, -1))
    lower = np.maximum(0, rows[..., 0] + columns[..., 0] - total)
    upper = np.minimum(rows[..., 0], columns[..., 0])
    return upper > lower


def _tenx_axes(
    matrix: h5py.Group,
    accessed: set[str] | None = None,
) -> tuple[list[str], list[str], list[str], list[int], list[int], np.ndarray]:
    decoded = accessed if accessed is not None else set()
    required = {"barcodes", "features", "data", "indices", "indptr", "shape"}
    if not required <= set(matrix):
        raise ProtocolRefusal("TENX_H5_SCHEMA_INCOMPLETE")
    features = matrix["features"]
    if not isinstance(features, h5py.Group) or not {"name", "feature_type"} <= set(
        features
    ):
        raise ProtocolRefusal("TENX_FEATURE_SCHEMA_INCOMPLETE")
    decoded.add("matrix/barcodes")
    barcodes = _decode_axis(matrix["barcodes"])
    decoded.add("matrix/features/name")
    names = _decode_axis(features["name"])
    decoded.add("matrix/features/feature_type")
    feature_types = _decode_axis(features["feature_type"])
    rna_features, adt_features = _resolve_panel(names, feature_types)
    decoded.add("matrix/shape")
    shape_values = _integer_values(np.asarray(matrix["shape"][()]), "TENX_SHAPE_INVALID")
    shape = tuple(int(value) for value in shape_values)
    if shape != (len(names), len(barcodes)):
        raise ProtocolRefusal("TENX_MATRIX_AXIS_LENGTH_MISMATCH")
    decoded.add("matrix/indptr")
    indptr = _integer_values(np.asarray(matrix["indptr"][()]), "CSC_INDPTR_INVALID")
    decoded.update(("matrix/data", "matrix/indices"))
    if (
        len(indptr) != shape[1] + 1
        or indptr[0] != 0
        or np.any(indptr[1:] < indptr[:-1])
        or int(indptr[-1]) != len(matrix["data"])
        or len(matrix["indices"]) != len(matrix["data"])
    ):
        raise ProtocolRefusal("CSC_STRUCTURE_INVALID")
    return barcodes, names, feature_types, rna_features, adt_features, indptr


def _column_rows(
    matrix: h5py.Group,
    indptr: np.ndarray,
    cell: int,
    feature_count: int,
    accessed: set[str] | None = None,
) -> tuple[int, np.ndarray]:
    start, stop = int(indptr[cell]), int(indptr[cell + 1])
    if accessed is not None:
        accessed.add("matrix/indices")
    rows = _integer_values(
        np.asarray(matrix["indices"][start:stop]), "CSC_INDEX_INVALID"
    )
    if (
        np.any(rows < 0)
        or np.any(rows >= feature_count)
        or len(rows) != len(set(rows.tolist()))
    ):
        raise ProtocolRefusal("CSC_INDEX_INVALID")
    return start, rows


def _selected_modality_counts(
    matrix: h5py.Group,
    indptr: np.ndarray,
    selected: list[int],
    feature_indices: list[int],
    feature_count: int,
    code: str,
    accessed: set[str] | None = None,
) -> tuple[np.ndarray, int]:
    lookup = {source: target for target, source in enumerate(feature_indices)}
    output = np.zeros((CELL_BUDGET, MARKER_COUNT), dtype=np.int64)
    decoded = 0
    for target_cell, source_cell in enumerate(selected):
        start, rows = _column_rows(
            matrix, indptr, source_cell, feature_count, accessed
        )
        offsets = np.asarray(
            [index for index, row in enumerate(rows) if int(row) in lookup],
            dtype=np.int64,
        )
        if accessed is not None:
            accessed.add("matrix/data")
        values = _integer_values(
            _read_positions(matrix["data"], start + offsets), code
        )
        for row, value in zip(rows[offsets], values):
            output[target_cell, lookup[int(row)]] = int(value)
        decoded += len(values)
    return output, decoded


def _reduce_held_rna(
    path: Path, donor: str, accessed: set[str] | None = None
) -> dict[str, Any]:
    """Select held cells and materialize only the RNA state object."""

    with h5py.File(path, "r") as handle:
        if "matrix" not in handle or not isinstance(handle["matrix"], h5py.Group):
            raise ProtocolRefusal("TENX_MATRIX_GROUP_ABSENT")
        matrix = handle["matrix"]
        barcodes, names, feature_types, rna_features, _, indptr = _tenx_axes(
            matrix, accessed
        )
        gene_mask = np.asarray(
            [value == "Gene Expression" for value in feature_types], dtype=bool
        )
        mitochondrial = np.asarray(
            [
                feature_types[index] == "Gene Expression" and name.startswith("MT-")
                for index, name in enumerate(names)
            ],
            dtype=bool,
        )
        detected = np.zeros(len(barcodes), dtype=np.int64)
        totals = np.zeros(len(barcodes), dtype=np.int64)
        mitochondrial_totals = np.zeros(len(barcodes), dtype=np.int64)
        qc_entries = 0
        for cell in range(len(barcodes)):
            start, rows = _column_rows(matrix, indptr, cell, len(names), accessed)
            offsets = np.flatnonzero(gene_mask[rows])
            if accessed is not None:
                accessed.add("matrix/data")
            values = _integer_values(
                _read_positions(matrix["data"], start + offsets),
                "RNA_COUNTS_INVALID",
            )
            gene_rows = rows[offsets]
            detected[cell] = int(np.count_nonzero(values > 0))
            totals[cell] = int(values.sum())
            mitochondrial_totals[cell] = int(values[mitochondrial[gene_rows]].sum())
            qc_entries += len(values)
        fraction = np.divide(
            mitochondrial_totals,
            totals,
            out=np.ones(len(barcodes), dtype=float),
            where=totals > 0,
        )
        eligible = np.flatnonzero(
            (detected >= MINIMUM_DETECTED_GENES)
            & (fraction <= MAXIMUM_MITOCHONDRIAL_FRACTION)
            & (totals <= MAXIMUM_RNA_UMIS)
        )
        selected = _selected_cells(donor, barcodes, eligible)
        selected_barcodes = [barcodes[index] for index in selected]
        rna_counts, panel_entries = _selected_modality_counts(
            matrix,
            indptr,
            selected,
            rna_features,
            len(names),
            "RNA_PANEL_COUNTS_INVALID",
            accessed,
        )
    states = (rna_counts > 0).astype(np.uint8)
    return {
        "donor": donor,
        "selected_indices": selected,
        "selected_barcodes": selected_barcodes,
        "selected_barcode_axis_sha256": _axis_sha256(selected_barcodes),
        "eligible_cell_count": int(len(eligible)),
        "states": states,
        "profile": states.mean(axis=0),
        "margins": np.stack(
            ((states == 0).sum(axis=0), (states == 1).sum(axis=0)), axis=1
        ),
        "access_certificate": {
            "storage": "10x_feature_by_barcode_csc",
            "full_matrix_dense_materialized": False,
            "rna_qc_used_only_gene_expression_numeric_entries": True,
            "adt_numeric_entries_read": 0,
            "rna_qc_numeric_entries_read": qc_entries,
            "rna_panel_numeric_entries_read": panel_entries,
            "selected_dense_panel_shape": [CELL_BUDGET, MARKER_COUNT],
            "mitochondrial_rule": "Gene Expression feature name starts with MT-",
            "profile": "mean raw detection state by donor",
            "decoded_h5_datasets": sorted(accessed or ()),
        },
    }


def _reduce_held_adt(
    path: Path,
    donor: str,
    selected: list[int],
    selected_barcodes: list[str],
    accessed: set[str] | None = None,
) -> dict[str, Any]:
    """Materialize only the ADT state object on an already frozen cell axis."""

    with h5py.File(path, "r") as handle:
        if "matrix" not in handle or not isinstance(handle["matrix"], h5py.Group):
            raise ProtocolRefusal("TENX_MATRIX_GROUP_ABSENT")
        matrix = handle["matrix"]
        barcodes, names, _, _, adt_features, indptr = _tenx_axes(matrix, accessed)
        if [barcodes[index] for index in selected] != selected_barcodes:
            raise PermissionError("ADT reduction cell axis differs from RNA selection")
        counts, panel_entries = _selected_modality_counts(
            matrix,
            indptr,
            selected,
            adt_features,
            len(names),
            "ADT_COUNTS_INVALID",
            accessed,
        )
    states = _midrank_adt(counts, selected_barcodes, donor)
    return {
        "donor": donor,
        "selected_barcode_axis_sha256": _axis_sha256(selected_barcodes),
        "states": states,
        "profile": np.log1p(counts).mean(axis=0),
        "margins": np.stack(
            ((states == 0).sum(axis=0), (states == 1).sum(axis=0)), axis=1
        ),
        "access_certificate": {
            "storage": "10x_feature_by_barcode_csc",
            "full_matrix_dense_materialized": False,
            "rna_numeric_entries_read": 0,
            "adt_panel_numeric_entries_read": panel_entries,
            "selected_dense_panel_shape": [CELL_BUDGET, MARKER_COUNT],
            "profile": "mean log1p raw count by donor",
            "decoded_h5_datasets": sorted(accessed or ()),
        },
    }


def _reduce_source_h5(
    path: Path, donor: str, accessed: set[str] | None = None
) -> dict[str, Any]:
    decoded = accessed if accessed is not None else set()
    rna = _reduce_held_rna(path, donor, decoded)
    adt = _reduce_held_adt(
        path, donor, rna["selected_indices"], rna["selected_barcodes"], decoded
    )
    rna_states = rna["states"]
    adt_states = adt["states"]
    tables = _joint_tables(rna_states, adt_states)
    destroyed = _joint_tables(
        rna_states,
        _destroyed_adt(adt_states, rna["selected_barcodes"], donor),
    )
    return {
        "donor": donor,
        "selected_barcodes": rna["selected_barcodes"],
        "selected_barcode_axis_sha256": rna["selected_barcode_axis_sha256"],
        "eligible_cell_count": rna["eligible_cell_count"],
        "rna_states": rna_states,
        "adt_states": adt_states,
        "rna_profile": rna["profile"],
        "adt_profile": adt["profile"],
        "rna_margins": rna["margins"],
        "adt_margins": adt["margins"],
        "tables": tables,
        "destroyed_tables": destroyed,
        "support": _margin_support(tables),
        "access_certificate": {
            "rna": rna["access_certificate"],
            "adt": adt["access_certificate"],
            "joint_tables_formed": True,
            "decoded_h5_datasets": sorted(decoded),
        },
    }


def _designated_url(candidate: dict[str, Any], record: dict[str, Any]) -> str:
    template = candidate["metadata_bindings"]["per_sample_url_template"]
    return template.format(
        gsm=record["gsm"], filename=urllib.parse.quote(record["filename"])
    )


def _validate_frozen_response_url(expected_url: str, observed_url: str) -> None:
    expected = urllib.parse.urlsplit(expected_url)
    observed = urllib.parse.urlsplit(observed_url)
    if (
        observed.scheme != "https"
        or observed.hostname != "ftp.ncbi.nlm.nih.gov"
        or observed.port not in (None, 443)
        or observed.username is not None
        or observed.password is not None
        or observed.path != expected.path
        or observed.query
        or observed.fragment
    ):
        raise PermissionError("H5 redirect crossed the frozen NCBI host/path boundary")


class _FrozenRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, expected_url: str):
        super().__init__()
        self.expected_url = expected_url

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        _validate_frozen_response_url(self.expected_url, newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _fetch_designated_file(
    candidate: dict[str, Any],
    record: dict[str, Any],
    scratch: Path,
    audit: dict[str, Any],
    cohort: str,
) -> tuple[Path, dict[str, Any]]:
    if cohort not in {"source", "held"}:
        raise ValueError("unknown H5 cohort")
    official = {
        (item["gsm"], item["filename"], int(item["bytes"]))
        for item in candidate[f"{cohort}_files"]
    }
    identity = (record["gsm"], record["filename"], int(record["bytes"]))
    if identity not in official:
        raise PermissionError(f"{cohort} fetch received a non-{cohort} donor")
    scratch.mkdir(parents=True, exist_ok=True)
    if list(scratch.glob("*.h5")):
        raise PermissionError("scratch already contains an H5; sequential access failed")
    target = scratch / record["filename"]
    url = _designated_url(candidate, record)
    audit["requested_urls"].append(url)
    audit[f"{cohort}_h5_get_count"] += 1
    identity: dict[str, Any] = {
        "gsm": record["gsm"],
        "donor": record["donor"],
        "stratum": record["stratum"],
        "filename": record["filename"],
        "expected_bytes": int(record["bytes"]),
        "requested_url": url,
        "download_status": "REQUESTED",
    }
    audit[f"{cohort}_files"].append(identity)
    request = urllib.request.Request(
        url, headers={"User-Agent": "coupling-fields-benchmark/2.0"}
    )
    digest = hashlib.sha256()
    size = 0
    final_url: str | None = None
    try:
        opener = urllib.request.build_opener(_FrozenRedirectHandler(url))
        with opener.open(request, timeout=300) as response, target.open(
            "xb"
        ) as stream:
            final_url = response.geturl()
            _validate_frozen_response_url(url, final_url)
            for block in iter(lambda: response.read(8 << 20), b""):
                size += len(block)
                digest.update(block)
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        sha256 = digest.hexdigest()
        identity.update(
            {
                "observed_bytes": size,
                "sha256": sha256,
                "final_url": final_url,
            }
        )
        if size != int(record["bytes"]):
            raise ProtocolRefusal(f"{cohort.upper()}_H5_BYTE_COUNT_MISMATCH")
        if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise AssertionError("downloaded H5 hash is invalid")
        identity["download_status"] = "COMPLETE"
        return target, identity
    except Exception as error:
        identity.update(
            {
                "observed_bytes": size,
                "sha256": digest.hexdigest(),
                "final_url": final_url,
                "download_status": "REFUSED",
                "download_reason_code": error.code
                if isinstance(error, ProtocolRefusal)
                else type(error).__name__,
            }
        )
        existed = target.exists()
        target.unlink(missing_ok=True)
        identity["deleted_after_download_failure"] = not target.exists()
        if existed:
            audit[f"{cohort}_h5_deleted_count"] += 1
        raise


def _fetch_source_file(
    candidate: dict[str, Any],
    record: dict[str, Any],
    scratch: Path,
    audit: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    return _fetch_designated_file(candidate, record, scratch, audit, "source")


@dataclass(frozen=True)
class ResidualConfig:
    family: str
    transport_multiplier: float


def _recipient_margin_support(rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    row = np.asarray(rows, dtype=float)
    column = np.asarray(columns, dtype=float)
    if row.shape != (MARKER_COUNT, MARKER_COUNT, 2) or column.shape != row.shape:
        raise ValueError("recipient margins have the wrong shape")
    total = row.sum(axis=-1)
    if not np.allclose(total, column.sum(axis=-1), rtol=0.0, atol=1e-10):
        raise ValueError("recipient row and column totals differ")
    lower = np.maximum(0.0, row[..., 0] + column[..., 0] - total)
    upper = np.minimum(row[..., 0], column[..., 0])
    return upper > lower


def _training_only_mask(tables: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Construct a mask from training associations only."""

    values = np.asarray(tables, dtype=np.int64)
    if values.ndim != 5 or values.shape[1:] != (MARKER_COUNT, MARKER_COUNT, 2, 2):
        raise ValueError("training table array has the wrong shape")
    support = _margin_support(values)
    rows = values.sum(axis=-1)
    columns = values.sum(axis=-2)
    total = values.sum(axis=(-2, -1))
    lower = np.maximum(0, rows[..., 0] + columns[..., 0] - total)
    upper = np.minimum(rows[..., 0], columns[..., 0])
    observed_sum = np.where(support, values[..., 0, 0], 0).sum(axis=0)
    lower_sum = np.where(support, lower, 0).sum(axis=0)
    upper_sum = np.where(support, upper, 0).sum(axis=0)
    pooled_all_training = values.sum(axis=0)
    mask = (
        (support.sum(axis=0) >= 2)
        & (observed_sum > lower_sum)
        & (observed_sum < upper_sum)
        & np.all(pooled_all_training > 0, axis=(-2, -1))
    )
    return mask, {
        "training_donor_count": len(values),
        "training_only_coordinate_count": int(mask.sum()),
        "training_only_mask_sha256": _array_sha256(mask.astype(np.uint8)),
        "minimum_informative_training_donors": int(
            support.sum(axis=0)[mask].min()
        )
        if np.any(mask)
        else 0,
        "pooled_poisson_includes_all_training_tables": True,
        "validation_association_used": False,
    }


def _fold_mask(
    training_tables: np.ndarray,
    validation_rows: np.ndarray,
    validation_columns: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    training, diagnostics = _training_only_mask(training_tables)
    recipient = _recipient_margin_support(validation_rows, validation_columns)
    mask = training & recipient
    diagnostics.update(
        {
            "recipient_margin_informative_count": int(recipient.sum()),
            "scored_coordinate_count": int(mask.sum()),
            "scored_mask_sha256": _array_sha256(mask.astype(np.uint8)),
            "recipient_paired_counts_used": False,
        }
    )
    if int(mask.sum()) < MINIMUM_COORDINATES:
        raise ProtocolRefusal("FOLD_COMPARISON_MASK_BELOW_64", details=diagnostics)
    return mask, diagnostics


def _independence(rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    row = np.asarray(rows, dtype=float)
    column = np.asarray(columns, dtype=float)
    total = row.sum(axis=-1)
    return row[..., :, None] * column[..., None, :] / total[..., None, None]


def _table_log_odds(table: np.ndarray) -> float:
    values = np.asarray(table, dtype=float)
    if values.shape != (2, 2) or np.any(values <= 0) or not np.isfinite(values).all():
        raise CouplingEstimationRefusal("pooled Poisson table is not strictly positive")
    return float(
        np.log(values[0, 0])
        + np.log(values[1, 1])
        - np.log(values[0, 1])
        - np.log(values[1, 0])
    )


def _fixed_interaction_table(
    log_odds: float, rows: np.ndarray, columns: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    """Solve the 2x2 log-linear nuisance refit and canonicalize from x00."""

    row = np.asarray(rows, dtype=float)
    column = np.asarray(columns, dtype=float)
    if (
        row.shape != (2,)
        or column.shape != (2,)
        or not np.isfinite(row).all()
        or not np.isfinite(column).all()
        or np.any(row <= 0)
        or np.any(column <= 0)
        or not np.isclose(row.sum(), column.sum(), rtol=0.0, atol=1e-10)
        or not np.isfinite(log_odds)
    ):
        raise CouplingEstimationRefusal("fixed-interaction margins are invalid")
    total = float(row.sum())
    lower = max(0.0, float(row[0] + column[0] - total))
    upper = min(float(row[0]), float(column[0]))
    if upper <= lower:
        raise CouplingEstimationRefusal("fixed-interaction margins are degenerate")

    def cells(x00: float) -> tuple[float, float, float]:
        x01 = float(upper - x00) if upper == row[0] else float(row[0] - x00)
        x10 = (
            float(upper - x00) if upper == column[0] else float(column[0] - x00)
        )
        x11 = (
            float(x00 - lower)
            if lower > 0.0
            else float(total - row[0] - column[0] + x00)
        )
        return x01, x10, x11

    def objective(x00: float) -> float:
        x01, x10, x11 = cells(x00)
        return math.log(x00) + math.log(x11) - math.log(x01) - math.log(x10) - log_odds

    left = np.nextafter(lower, upper)
    right = np.nextafter(upper, lower)
    x00 = float(brentq(objective, left, right, xtol=5e-15, rtol=1e-14, maxiter=200))
    x01, x10, x11 = cells(x00)
    table = np.asarray(
        [
            [x00, x01],
            [x10, x11],
        ]
    )
    observed_log_odds = _table_log_odds(table)
    row_error = float(np.max(np.abs(table.sum(axis=1) - row)))
    column_error = float(np.max(np.abs(table.sum(axis=0) - column)))
    log_odds_error = abs(observed_log_odds - log_odds)
    if max(row_error, column_error) > 1e-10 or log_odds_error > 1e-8:
        raise CouplingEstimationRefusal("fixed-interaction reconstruction certificate failed")
    return table, {
        "maximum_row_margin_error": row_error,
        "maximum_column_margin_error": column_error,
        "absolute_log_odds_error": log_odds_error,
    }


def _fit_pooled_poisson(tables: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    values = np.asarray(tables, dtype=np.int64)
    selected = np.asarray(mask, dtype=bool)
    pooled = values.sum(axis=0)
    if selected.shape != (MARKER_COUNT, MARKER_COUNT) or np.any(
        pooled[selected] <= 0
    ):
        raise CouplingEstimationRefusal(
            "pooled saturated Poisson interaction has a zero selected cell"
        )
    coordinates = np.zeros((MARKER_COUNT, MARKER_COUNT), dtype=float)
    maximum_cell_error = 0.0
    maximum_margin_error = 0.0
    maximum_log_odds_error = 0.0
    for index in zip(*np.nonzero(selected)):
        source = pooled[index]
        log_odds = _table_log_odds(source)
        coordinates[index] = log_odds
        reconstructed, certificate = _fixed_interaction_table(
            log_odds, source.sum(axis=1), source.sum(axis=0)
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
        raise CouplingEstimationRefusal("pooled Poisson source reconstruction failed")
    return {
        "population_log_odds": coordinates,
        "fit_certificate": {
            "family": "pooled saturated Poisson row-column-interaction",
            "pooled_every_training_donor_including_degenerate_margins": True,
            "recipient_reconstruction": "direct fixed-interaction nuisance refit",
            "conditional_noncentral_hypergeometric_reconstruction": False,
            "coordinate_count": int(selected.sum()),
            "maximum_normalized_source_cell_error": maximum_cell_error,
            "maximum_source_margin_error": maximum_margin_error,
            "maximum_source_log_odds_error": maximum_log_odds_error,
            "threshold": 1e-8,
            "passes": True,
        },
    }


def _predict_pooled_poisson(
    log_odds: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    alpha: float,
    mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    prediction = _independence(rows, columns)
    maximum_margin_error = 0.0
    maximum_log_odds_error = 0.0
    for index in zip(*np.nonzero(mask)):
        table, certificate = _fixed_interaction_table(
            float(alpha) * float(log_odds[index]), rows[index], columns[index]
        )
        prediction[index] = table
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
        "alpha_zero_is_independence": bool(
            alpha != 0.0
            or np.allclose(prediction, _independence(rows, columns), atol=1e-10)
        ),
    }


def _fit_residual(
    tables: np.ndarray, support: np.ndarray, mask: np.ndarray, family: str
) -> np.ndarray:
    values = np.asarray(tables, dtype=np.int64)
    informative = np.asarray(support, dtype=bool) & mask[None, ...]
    statistic = (
        signed_pearson_coordinate
        if family == "pearson"
        else signed_deviance_coordinate
        if family == "root_deviance"
        else None
    )
    if statistic is None:
        raise ValueError("unknown residual family")
    coordinates = np.full(informative.shape, np.nan)
    for donor, first, second in np.argwhere(informative):
        coordinates[donor, first, second] = (
            statistic(values[donor, first, second]) / math.sqrt(CELL_BUDGET)
        )
    pooled = np.zeros((MARKER_COUNT, MARKER_COUNT), dtype=float)
    pooled[mask] = np.nanmean(coordinates[:, mask], axis=0)
    if not np.isfinite(pooled[mask]).all():
        raise CouplingEstimationRefusal("residual coordinate pool is nonfinite")
    return pooled


def _predict_residual(
    pooled: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    configuration: ResidualConfig,
    mask: np.ndarray,
) -> np.ndarray:
    prediction = _independence(rows, columns)
    for index in zip(*np.nonzero(mask)):
        coordinate = (
            configuration.transport_multiplier
            * float(pooled[index])
            * math.sqrt(CELL_BUDGET)
        )
        prediction[index] = engine._residual_table(
            coordinate, rows[index], columns[index], configuration.family
        )
    return prediction


def _predict_conditional(
    log_odds: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    alpha: float,
    mask: np.ndarray,
) -> np.ndarray:
    prediction = _independence(rows, columns)
    for index in zip(*np.nonzero(mask)):
        prediction[index] = expected_binary_table_from_log_odds(
            float(alpha) * float(log_odds[index]), rows[index], columns[index]
        )
    return prediction


def _loss(observed: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> float:
    truth = np.asarray(observed, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    selected = np.asarray(mask, dtype=bool)
    if int(selected.sum()) < MINIMUM_COORDINATES:
        raise CouplingEstimationRefusal("donor comparison mask has fewer than 64 pairs")
    truth = truth[selected]
    estimate = estimate[selected]
    if not np.allclose(truth.sum(axis=-1), estimate.sum(axis=-1), atol=1e-8) or not np.allclose(
        truth.sum(axis=-2), estimate.sum(axis=-2), atol=1e-8
    ):
        raise FloatingPointError("prediction changed a recipient margin")
    positive = truth > 0
    if np.any(estimate[positive] <= 0) or not np.isfinite(estimate).all():
        raise FloatingPointError("prediction assigns invalid mass")
    terms = np.zeros_like(truth)
    terms[positive] = truth[positive] * np.log(truth[positive] / estimate[positive])
    return float((2.0 * terms.sum(axis=(-2, -1)) / CELL_BUDGET).mean())


def _select_complete(
    order: list[Any], losses: dict[Any, np.ndarray], family: str
) -> tuple[Any, np.ndarray]:
    complete = [configuration for configuration in order if np.isfinite(losses[configuration]).all()]
    if not complete:
        raise ProtocolRefusal(f"NO_COMPLETE_{family.upper()}_CONFIGURATION")
    rank = {configuration: index for index, configuration in enumerate(order)}
    selected = min(
        complete,
        key=lambda configuration: (
            float(np.mean(losses[configuration])),
            rank[configuration],
        ),
    )
    return selected, losses[selected]


def _bootstrap_comparison(
    primary: np.ndarray, comparator: np.ndarray, role: str
) -> dict[str, Any]:
    first = np.asarray(primary, dtype=float)
    second = np.asarray(comparator, dtype=float)
    difference = first - second
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws = generator.integers(0, len(first), size=(BOOTSTRAPS, len(first)))
    interval = np.quantile(difference[draws].mean(axis=1), [0.025, 0.975])
    relative = 1.0 - float(first.mean() / second.mean())
    favorable = int(np.count_nonzero(difference < 0.0))
    if role in {"residual", "pooled_poisson", "destroyed"}:
        passes = relative >= 0.05 and interval[1] < 0.0 and favorable >= 7
    elif role == "independence":
        passes = relative >= 0.05
    elif role == "common_effect_cmle":
        passes = float(first.mean()) < float(second.mean())
    else:
        raise ValueError("unknown comparison role")
    return {
        "role": role,
        "primary_mean_loss": float(first.mean()),
        "comparator_mean_loss": float(second.mean()),
        "relative_loss_reduction": relative,
        "paired_donor_difference_95_ci": interval.tolist(),
        "favorable_donors": favorable,
        "donor_count": len(first),
        "bootstrap_draws": BOOTSTRAPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "passes_frozen_source_requirement": bool(passes),
        "source_interval_is_selection_conditional": True,
    }


def _candidate_evaluations(order: list[Any], losses: dict[Any, np.ndarray]) -> list[dict[str, Any]]:
    output = []
    for configuration in order:
        values = losses[configuration]
        complete = bool(np.isfinite(values).all())
        record: dict[str, Any] = {
            "configuration": asdict(configuration)
            if hasattr(configuration, "__dataclass_fields__")
            else {"transport_multiplier": configuration},
            "status": "COMPLETE" if complete else "REFUSED",
            "completed_folds": int(np.isfinite(values).sum()),
        }
        if complete:
            record["fold_losses"] = values.tolist()
            record["mean_donor_equal_loss"] = float(values.mean())
        output.append(record)
    return output


def _source_arrays(
    records: dict[str, dict[str, Any]], axis: list[str], key: str = "tables"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tables = np.asarray([records[donor][key] for donor in axis], dtype=np.int64)
    rna_profiles = np.asarray([records[donor]["rna_profile"] for donor in axis])
    adt_profiles = np.asarray([records[donor]["adt_profile"] for donor in axis])
    support = _margin_support(tables)
    return tables, rna_profiles, adt_profiles, support


def _evaluate_source(
    records: dict[str, dict[str, Any]], source_axis: list[str]
) -> dict[str, Any]:
    if len(source_axis) != 8 or set(source_axis) != set(records):
        raise ValueError("source evaluation requires the eight frozen donors")

    primary_order = [
        engine.PrimaryConfig(
            GRAPH_NEIGHBORS,
            heterogeneity,
            ridge,
            graph,
            alpha,
        )
        for heterogeneity, ridge, graph, alpha in product(
            HETEROGENEITY_GRID, RIDGE_GRID, GRAPH_GRID, TRANSPORT_GRID
        )
    ]
    primary_losses = {
        configuration: np.full(len(source_axis), np.nan)
        for configuration in primary_order
    }
    residual_order = [
        ResidualConfig(family, alpha)
        for family, alpha in product(RESIDUAL_FAMILIES, TRANSPORT_GRID)
    ]
    residual_losses = {
        configuration: np.full(len(source_axis), np.nan)
        for configuration in residual_order
    }
    common_order = list(TRANSPORT_GRID)
    common_losses = {alpha: np.full(len(source_axis), np.nan) for alpha in common_order}
    poisson_order = list(TRANSPORT_GRID)
    poisson_losses = {
        alpha: np.full(len(source_axis), np.nan) for alpha in poisson_order
    }
    independence_losses = np.full(len(source_axis), np.nan)
    masks: dict[str, dict[str, Any]] = {}
    refusals: list[dict[str, Any]] = []
    primary_certificates: dict[tuple[str, float, float, float], dict[str, Any]] = {}
    common_certificates: dict[str, dict[str, Any]] = {}
    poisson_certificates: dict[str, dict[str, Any]] = {}

    for held_index, held in enumerate(source_axis):
        training_axis = [donor for donor in source_axis if donor != held]
        tables, rna_profiles, adt_profiles, support = _source_arrays(
            records, training_axis
        )
        truth = np.asarray(records[held]["tables"], dtype=np.int64)
        rows, columns = truth.sum(axis=-1), truth.sum(axis=-2)
        mask, diagnostics = _fold_mask(tables, rows, columns)
        diagnostics["training_donors"] = training_axis
        diagnostics["validation_donor"] = held
        masks[held] = diagnostics
        training_support = support & mask[None, ...]

        for heterogeneity, ridge, graph in product(
            HETEROGENEITY_GRID, RIDGE_GRID, GRAPH_GRID
        ):
            base = engine.PrimaryConfig(
                GRAPH_NEIGHBORS, heterogeneity, ridge, graph, 1.0
            )
            try:
                fit = engine._fit_primary(
                    tables,
                    rna_profiles,
                    adt_profiles,
                    base,
                    training_support,
                )
                primary_certificates[(held, heterogeneity, ridge, graph)] = fit[
                    "fit_certificate"
                ]
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
                        "configuration": asdict(base),
                        "reason": str(error),
                    }
                )
                continue
            for alpha in TRANSPORT_GRID:
                configuration = engine.PrimaryConfig(
                    GRAPH_NEIGHBORS, heterogeneity, ridge, graph, alpha
                )
                try:
                    prediction = _predict_conditional(
                        fit["population_log_odds"], rows, columns, alpha, mask
                    )
                    primary_losses[configuration][held_index] = _loss(
                        truth, prediction, mask
                    )
                except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
                    refusals.append(
                        {
                            "fold": held,
                            "family": "primary_prediction",
                            "configuration": asdict(configuration),
                            "reason": str(error),
                        }
                    )

        try:
            common = engine._fit_common_effect(tables, mask, training_support)
            common_certificates[held] = common["fit_certificate"]
        except (
            ValueError,
            FloatingPointError,
            CouplingEstimationRefusal,
            np.linalg.LinAlgError,
        ) as error:
            raise ProtocolRefusal(
                "MANDATORY_COMMON_EFFECT_FOLD_REFUSAL", str(error), {"fold": held}
            ) from error
        for alpha in TRANSPORT_GRID:
            try:
                common_losses[alpha][held_index] = _loss(
                    truth,
                    _predict_conditional(
                        common["population_log_odds"], rows, columns, alpha, mask
                    ),
                    mask,
                )
            except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
                refusals.append(
                    {
                        "fold": held,
                        "family": "common_effect_prediction",
                        "transport_multiplier": alpha,
                        "reason": str(error),
                    }
                )

        poisson = _fit_pooled_poisson(tables, mask)
        poisson_certificates[held] = poisson["fit_certificate"]
        for alpha in TRANSPORT_GRID:
            try:
                prediction, certificate = _predict_pooled_poisson(
                    poisson["population_log_odds"], rows, columns, alpha, mask
                )
                if not certificate["alpha_zero_is_independence"]:
                    raise CouplingEstimationRefusal(
                        "pooled Poisson alpha-zero certificate failed"
                    )
                poisson_losses[alpha][held_index] = _loss(truth, prediction, mask)
            except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
                refusals.append(
                    {
                        "fold": held,
                        "family": "pooled_poisson_prediction",
                        "transport_multiplier": alpha,
                        "reason": str(error),
                    }
                )

        for family in RESIDUAL_FAMILIES:
            try:
                pooled_residual = _fit_residual(
                    tables, training_support, mask, family
                )
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
            for alpha in TRANSPORT_GRID:
                configuration = ResidualConfig(family, alpha)
                try:
                    residual_losses[configuration][held_index] = _loss(
                        truth,
                        _predict_residual(
                            pooled_residual, rows, columns, configuration, mask
                        ),
                        mask,
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
            truth, _independence(rows, columns), mask
        )

    selected_primary, primary = _select_complete(
        primary_order, primary_losses, "primary"
    )
    selected_residual, residual = _select_complete(
        residual_order, residual_losses, "residual"
    )
    selected_common, common = _select_complete(
        common_order, common_losses, "common_effect"
    )
    selected_poisson, poisson = _select_complete(
        poisson_order, poisson_losses, "pooled_poisson"
    )

    destroyed_order = list(TRANSPORT_GRID)
    destroyed_losses = {
        alpha: np.full(len(source_axis), np.nan) for alpha in destroyed_order
    }
    destroyed_certificates: dict[str, dict[str, Any]] = {}
    for held_index, held in enumerate(source_axis):
        training_axis = [donor for donor in source_axis if donor != held]
        tables, rna_profiles, adt_profiles, support = _source_arrays(
            records, training_axis, key="destroyed_tables"
        )
        truth = np.asarray(records[held]["tables"], dtype=np.int64)
        rows, columns = truth.sum(axis=-1), truth.sum(axis=-2)
        mask = np.zeros((MARKER_COUNT, MARKER_COUNT), dtype=bool)
        expected_hash = masks[held]["scored_mask_sha256"]
        real_training = np.asarray(
            [records[donor]["tables"] for donor in training_axis], dtype=np.int64
        )
        mask, _ = _fold_mask(real_training, rows, columns)
        if _array_sha256(mask.astype(np.uint8)) != expected_hash:
            raise AssertionError("destroyed refit did not reuse the real comparison mask")
        configuration = engine.PrimaryConfig(
            GRAPH_NEIGHBORS,
            selected_primary.heterogeneity_penalty,
            selected_primary.ridge_penalty,
            selected_primary.graph_penalty,
            1.0,
        )
        fit = engine._fit_primary(
            tables,
            rna_profiles,
            adt_profiles,
            configuration,
            support & mask[None, ...],
        )
        destroyed_certificates[held] = fit["fit_certificate"]
        for alpha in TRANSPORT_GRID:
            destroyed_losses[alpha][held_index] = _loss(
                truth,
                _predict_conditional(
                    fit["population_log_odds"], rows, columns, alpha, mask
                ),
                mask,
            )
    selected_destroyed, destroyed = _select_complete(
        destroyed_order, destroyed_losses, "destroyed"
    )

    all_tables, rna_profiles, adt_profiles, all_support = _source_arrays(
        records, source_axis
    )
    final_mask, final_diagnostics = _training_only_mask(all_tables)
    per_source_counts = {}
    for donor in source_axis:
        count = int(np.count_nonzero(final_mask & records[donor]["support"]))
        per_source_counts[donor] = count
        if count < MINIMUM_COORDINATES:
            raise ProtocolRefusal(
                "FINAL_SOURCE_MASK_BELOW_64", details={"donor": donor, "count": count}
            )
    final_diagnostics["source_margin_intersection_counts"] = per_source_counts
    final_support = all_support & final_mask[None, ...]
    final_primary_configuration = engine.PrimaryConfig(
        GRAPH_NEIGHBORS,
        selected_primary.heterogeneity_penalty,
        selected_primary.ridge_penalty,
        selected_primary.graph_penalty,
        1.0,
    )
    final_primary = engine._fit_primary(
        all_tables,
        rna_profiles,
        adt_profiles,
        final_primary_configuration,
        final_support,
    )
    final_common = engine._fit_common_effect(all_tables, final_mask, final_support)
    final_poisson = _fit_pooled_poisson(all_tables, final_mask)
    final_residual = _fit_residual(
        all_tables, final_support, final_mask, selected_residual.family
    )
    destroyed_tables = np.asarray(
        [records[donor]["destroyed_tables"] for donor in source_axis], dtype=np.int64
    )
    final_destroyed = engine._fit_primary(
        destroyed_tables,
        rna_profiles,
        adt_profiles,
        final_primary_configuration,
        _margin_support(destroyed_tables) & final_mask[None, ...],
    )

    comparisons = {
        "selected_residual": _bootstrap_comparison(primary, residual, "residual"),
        "pooled_saturated_poisson": _bootstrap_comparison(
            primary, poisson, "pooled_poisson"
        ),
        "destroyed_link": _bootstrap_comparison(primary, destroyed, "destroyed"),
        "common_effect_cmle": _bootstrap_comparison(
            primary, common, "common_effect_cmle"
        ),
        "independence": _bootstrap_comparison(
            primary, independence_losses, "independence"
        ),
    }
    passes = all(
        comparison["passes_frozen_source_requirement"]
        for comparison in comparisons.values()
    )
    models = {
        "final_mask": final_mask.astype(np.uint8).tolist(),
        "final_mask_sha256": _array_sha256(final_mask.astype(np.uint8)),
        "primary": {
            "kind": "exact_conditional_log_odds",
            "configuration": asdict(selected_primary),
            "source_coordinate": final_primary["population_log_odds"].ravel().tolist(),
            "fit_certificate": final_primary["fit_certificate"],
        },
        "selected_residual": {
            "kind": "raw_signed_residual",
            "configuration": asdict(selected_residual),
            "source_coordinate": final_residual.ravel().tolist(),
            "fit_certificate": {
                "finite_on_final_mask": bool(np.isfinite(final_residual[final_mask]).all()),
                "raw_statistic_no_null_centering": True,
            },
        },
        "common_effect_cmle": {
            "kind": "exact_conditional_common_log_odds",
            "transport_multiplier": selected_common,
            "source_coordinate": final_common["population_log_odds"].ravel().tolist(),
            "fit_certificate": final_common["fit_certificate"],
        },
        "pooled_saturated_poisson": {
            "kind": "pooled_saturated_poisson_fixed_interaction",
            "transport_multiplier": selected_poisson,
            "source_coordinate": final_poisson["population_log_odds"].ravel().tolist(),
            "fit_certificate": final_poisson["fit_certificate"],
        },
        "destroyed_link": {
            "kind": "destroyed_exact_conditional_log_odds",
            "configuration": {
                **asdict(final_primary_configuration),
                "transport_multiplier": selected_destroyed,
            },
            "source_coordinate": final_destroyed[
                "population_log_odds"
            ].ravel().tolist(),
            "fit_certificate": final_destroyed["fit_certificate"],
        },
        "independence": {"kind": "recipient_margin_independence"},
    }
    return {
        "status": "SOURCE_PROMOTION_PASS"
        if passes
        else "TERMINAL_SOURCE_PROMOTION_REFUSAL",
        "passes_source_promotion_gate": bool(passes),
        "source_donor_axis": source_axis,
        "fold_specific_masks": masks,
        "final_source_only_mask": final_diagnostics,
        "selected_primary": asdict(selected_primary),
        "selected_residual": asdict(selected_residual),
        "selected_common_effect_transport": selected_common,
        "selected_pooled_poisson_transport": selected_poisson,
        "selected_destroyed_transport": selected_destroyed,
        "fold_losses": {
            "primary": primary.tolist(),
            "selected_residual": residual.tolist(),
            "common_effect_cmle": common.tolist(),
            "pooled_saturated_poisson": poisson.tolist(),
            "destroyed_link": destroyed.tolist(),
            "independence": independence_losses.tolist(),
        },
        "comparisons": comparisons,
        "candidate_evaluations": {
            "primary": _candidate_evaluations(primary_order, primary_losses),
            "residual": _candidate_evaluations(residual_order, residual_losses),
            "common_effect_cmle": _candidate_evaluations(
                common_order, common_losses
            ),
            "pooled_saturated_poisson": _candidate_evaluations(
                poisson_order, poisson_losses
            ),
            "destroyed_link": _candidate_evaluations(
                destroyed_order, destroyed_losses
            ),
        },
        "numerical_certificates": {
            "selected_primary_folds": {
                donor: primary_certificates[
                    (
                        donor,
                        selected_primary.heterogeneity_penalty,
                        selected_primary.ridge_penalty,
                        selected_primary.graph_penalty,
                    )
                ]
                for donor in source_axis
            },
            "common_effect_folds": common_certificates,
            "pooled_poisson_folds": poisson_certificates,
            "destroyed_folds": destroyed_certificates,
            "final_fits_complete": True,
        },
        "models": models if passes else None,
        "refusals": refusals,
        "held_h5_access_authorized": False,
        "held_h5_access_eligible_after_public_source_pass": bool(passes),
    }


def _read_source_records(
    candidate: dict[str, Any], scratch: Path, audit: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    records: dict[str, dict[str, Any]] = {}
    identities: list[dict[str, Any]] = []
    for sample in candidate["source_files"]:
        path: Path | None = None
        identity: dict[str, Any] | None = None
        accessed: set[str] = set()
        try:
            path, identity = _fetch_source_file(candidate, sample, scratch, audit)
            audit["maximum_simultaneous_h5_files"] = max(
                audit["maximum_simultaneous_h5_files"], len(list(scratch.glob("*.h5")))
            )
            reduced = _reduce_source_h5(path, sample["donor"], accessed)
            records[sample["donor"]] = reduced
            identity["reduction_status"] = "COMPLETE"
            identity["access_certificate"] = reduced["access_certificate"]
        except Exception as error:
            if identity is not None:
                identity["reduction_status"] = "REFUSED"
                identity["reduction_reason_code"] = (
                    error.code
                    if isinstance(error, ProtocolRefusal)
                    else type(error).__name__
                )
            raise
        finally:
            if path is not None:
                path.unlink(missing_ok=True)
                audit["source_h5_deleted_count"] += 1
                if identity is not None:
                    identity["deleted_after_reduction"] = not path.exists()
            if identity is not None:
                identity["decoded_h5_datasets"] = sorted(accessed)
                identities.append(identity)
    if len(records) != 8 or list(records) != [
        sample["donor"] for sample in candidate["source_files"]
    ]:
        raise AssertionError("source reduction did not preserve the frozen donor axis")
    if list(scratch.glob("*.h5")):
        raise PermissionError("source reduction left an H5 in scratch")
    return records, identities


def run_source(
    attempt_path: Path = SOURCE_ATTEMPT,
    output_path: Path = SOURCE_RESULT,
    scratch: Path = DEFAULT_SCRATCH,
) -> dict[str, Any]:
    """Consume the claimed source run exactly once."""

    _require_canonical_path(attempt_path, SOURCE_ATTEMPT, "source attempt")
    _require_canonical_path(output_path, SOURCE_RESULT, "source result")
    if output_path.exists():
        raise FileExistsError("source result already exists")
    if not attempt_path.exists():
        raise FileNotFoundError("source attempt does not exist")
    consumption = _claim_consumption(
        SOURCE_CONSUMPTION,
        "gse179221-bmmc-source-consumption/1.0",
        attempt_path,
    )
    audit: dict[str, Any] = {
        "source_attempt_existed_before_first_h5_get": True,
        "requested_urls": [],
        "source_h5_get_count": 0,
        "held_h5_get_count": 0,
        "all_donor_tar_get_count": 0,
        "source_h5_deleted_count": 0,
        "maximum_simultaneous_h5_files": 0,
        "source_files": [],
    }
    identities: list[dict[str, Any]] = []
    public: dict[str, str] = {}
    try:
        candidate, public = _validate_source_attempt(attempt_path)
        records, identities = _read_source_records(candidate, scratch, audit)
        with _engine_contract():
            result = _evaluate_source(
                records, [sample["donor"] for sample in candidate["source_files"]]
            )
        result.update(
            {
                "schema": "gse179221-bmmc-source-result/1.0",
                "created_at_utc": _timestamp(),
                "source_attempt_sha256": _sha256(attempt_path),
                "source_consumption_sha256": _sha256(SOURCE_CONSUMPTION),
                "source_consumption": consumption,
                "public_freezes": public,
                "implementation_amendment_sha256": AMENDMENT_SHA256,
                "ordered_cognates": [[rna, adt] for rna, adt in PANEL],
                "source_files": identities,
                "source_table_sha256": {
                    donor: _array_sha256(record["tables"])
                    for donor, record in records.items()
                },
                "source_destroyed_table_sha256": {
                    donor: _array_sha256(record["destroyed_tables"])
                    for donor, record in records.items()
                },
                "source_selected_cell_axis_sha256": {
                    donor: record["selected_barcode_axis_sha256"]
                    for donor, record in records.items()
                },
                "access_audit": audit,
            }
        )
    except Exception as error:
        result = {
            "schema": "gse179221-bmmc-source-result/1.0",
            "status": "TERMINAL_SOURCE_EXECUTION_REFUSAL",
            "created_at_utc": _timestamp(),
            "passes_source_promotion_gate": False,
            "reason_code": error.code
            if isinstance(error, ProtocolRefusal)
            else type(error).__name__,
            "reason": str(error),
            "reason_details": error.details
            if isinstance(error, ProtocolRefusal)
            else {},
            "source_attempt_sha256": _sha256(attempt_path),
            "source_consumption_sha256": _sha256(SOURCE_CONSUMPTION),
            "source_consumption": consumption,
            "public_freezes": public,
            "implementation_amendment_sha256": AMENDMENT_SHA256,
            "source_files": audit["source_files"],
            "access_audit": audit,
            "held_h5_access_authorized": False,
            "held_h5_access_eligible_after_public_source_pass": False,
        }
    if audit["held_h5_get_count"] != 0 or audit["all_donor_tar_get_count"] != 0:
        raise AssertionError("source execution reached a forbidden URL")
    _write_json_x(output_path, result)
    return result


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


def _finite_numeric_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_numeric_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_numeric_tree(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def _completed_download_records(
    observed: Any, designated: list[dict[str, Any]], candidate: dict[str, Any]
) -> bool:
    if not isinstance(observed, list) or len(observed) != len(designated):
        return False
    for actual, expected in zip(observed, designated):
        url = _designated_url(candidate, expected)
        if (
            actual.get("gsm") != expected["gsm"]
            or actual.get("donor") != expected["donor"]
            or actual.get("stratum") != expected["stratum"]
            or actual.get("filename") != expected["filename"]
            or actual.get("expected_bytes") != int(expected["bytes"])
            or actual.get("observed_bytes") != int(expected["bytes"])
            or actual.get("requested_url") != url
            or actual.get("final_url") != url
            or actual.get("download_status") != "COMPLETE"
            or actual.get("reduction_status") != "COMPLETE"
            or actual.get("deleted_after_reduction") is not True
            or re.fullmatch(r"[0-9a-f]{64}", actual.get("sha256", "")) is None
            or set(actual.get("decoded_h5_datasets", []))
            != {
                "matrix/barcodes",
                "matrix/features/name",
                "matrix/features/feature_type",
                "matrix/shape",
                "matrix/indptr",
                "matrix/indices",
                "matrix/data",
            }
            or not isinstance(actual.get("access_certificate"), dict)
        ):
            return False
    return True


def _selected_published_candidate(
    records: Any, expected_configurations: list[dict[str, Any]], family: str
) -> tuple[dict[str, Any], np.ndarray]:
    if not isinstance(records, list) or len(records) != len(expected_configurations):
        raise PermissionError(f"source {family} candidate grid is incomplete")
    complete: list[tuple[int, dict[str, Any], np.ndarray]] = []
    for index, (record, expected) in enumerate(zip(records, expected_configurations)):
        if record.get("configuration") != expected:
            raise PermissionError(f"source {family} candidate order changed")
        if record.get("status") == "COMPLETE":
            losses = np.asarray(record.get("fold_losses"), dtype=float)
            if (
                record.get("completed_folds") != 8
                or losses.shape != (8,)
                or not np.isfinite(losses).all()
                or not math.isclose(
                    float(losses.mean()),
                    float(record.get("mean_donor_equal_loss", math.nan)),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise PermissionError(f"source {family} complete candidate is invalid")
            complete.append((index, expected, losses))
        elif (
            record.get("status") != "REFUSED"
            or not isinstance(record.get("completed_folds"), int)
            or not 0 <= record["completed_folds"] < 8
            or "fold_losses" in record
            or "mean_donor_equal_loss" in record
        ):
            raise PermissionError(f"source {family} refused candidate is invalid")
    if not complete:
        raise PermissionError(f"source {family} has no complete candidate")
    _, configuration, losses = min(
        complete, key=lambda item: (float(item[2].mean()), item[0])
    )
    return configuration, losses


def _source_comparison_matches(
    observed: dict[str, Any], expected: dict[str, Any]
) -> bool:
    numeric_fields = (
        "primary_mean_loss",
        "comparator_mean_loss",
        "relative_loss_reduction",
    )
    for field in numeric_fields:
        if not math.isclose(
            float(observed.get(field, math.nan)),
            float(expected[field]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            return False
    if not np.allclose(
        np.asarray(observed.get("paired_donor_difference_95_ci"), dtype=float),
        np.asarray(expected["paired_donor_difference_95_ci"], dtype=float),
        rtol=1e-12,
        atol=1e-12,
    ):
        return False
    return all(
        observed.get(field) == expected[field]
        for field in (
            "role",
            "favorable_donors",
            "donor_count",
            "bootstrap_draws",
            "bootstrap_seed",
            "passes_frozen_source_requirement",
            "source_interval_is_selection_conditional",
        )
    )


def _validate_source_pass_payload(
    source: dict[str, Any], candidate: dict[str, Any]
) -> None:
    source_axis = [record["donor"] for record in candidate["source_files"]]
    comparisons = source.get("comparisons", {})
    expected_roles = {
        "selected_residual": "residual",
        "pooled_saturated_poisson": "pooled_poisson",
        "destroyed_link": "destroyed",
        "common_effect_cmle": "common_effect_cmle",
        "independence": "independence",
    }
    if set(comparisons) != set(expected_roles):
        raise PermissionError("source result omits a frozen comparison")
    for name, role in expected_roles.items():
        comparison = comparisons[name]
        if (
            comparison.get("role") != role
            or comparison.get("donor_count") != len(source_axis)
            or comparison.get("passes_frozen_source_requirement") is not True
            or not _finite_numeric_tree(comparison)
        ):
            raise PermissionError("source comparison does not pass its frozen gate")
        relative = float(comparison["relative_loss_reduction"])
        if name in {
            "selected_residual",
            "pooled_saturated_poisson",
            "destroyed_link",
        } and not (
            relative >= 0.05
            and comparison["paired_donor_difference_95_ci"][1] < 0.0
            and comparison["favorable_donors"] >= 7
            and comparison.get("bootstrap_draws") == BOOTSTRAPS
            and comparison.get("bootstrap_seed") == BOOTSTRAP_SEED
        ):
            raise PermissionError("source inferential comparison misses its frozen gate")
        if name == "independence" and relative < 0.05:
            raise PermissionError("source independence comparison misses its frozen gate")
        if name == "common_effect_cmle" and not (
            comparison["primary_mean_loss"] < comparison["comparator_mean_loss"]
        ):
            raise PermissionError("source common-effect point ablation misses its gate")

    models = source.get("models")
    expected_models = {
        "final_mask",
        "final_mask_sha256",
        "primary",
        "selected_residual",
        "common_effect_cmle",
        "pooled_saturated_poisson",
        "destroyed_link",
        "independence",
    }
    if not isinstance(models, dict) or set(models) != expected_models:
        raise PermissionError("source result has an incomplete frozen model payload")
    raw_mask = np.asarray(models["final_mask"])
    mask = raw_mask.astype(np.uint8)
    if (
        raw_mask.shape != (MARKER_COUNT, MARKER_COUNT)
        or not np.isin(raw_mask, (0, 1)).all()
        or int(mask.sum()) < MINIMUM_COORDINATES
        or models["final_mask_sha256"] != _array_sha256(mask)
        or source.get("final_source_only_mask", {}).get(
            "training_only_mask_sha256"
        )
        != models["final_mask_sha256"]
    ):
        raise PermissionError("source final mask violates its source-only seal")

    primary = models["primary"]
    primary_config = primary.get("configuration", {})
    if (
        primary.get("kind") != "exact_conditional_log_odds"
        or primary_config.get("graph_neighbors") != GRAPH_NEIGHBORS
        or primary_config.get("heterogeneity_penalty") not in HETEROGENEITY_GRID
        or primary_config.get("ridge_penalty") not in RIDGE_GRID
        or primary_config.get("graph_penalty") not in GRAPH_GRID
        or primary_config.get("transport_multiplier") not in TRANSPORT_GRID
    ):
        raise PermissionError("source primary configuration is outside the frozen grid")
    residual = models["selected_residual"]
    if (
        residual.get("kind") != "raw_signed_residual"
        or residual.get("configuration", {}).get("family") not in RESIDUAL_FAMILIES
        or residual.get("configuration", {}).get("transport_multiplier")
        not in TRANSPORT_GRID
        or residual.get("fit_certificate", {}).get("raw_statistic_no_null_centering")
        is not True
        or residual.get("fit_certificate", {}).get("finite_on_final_mask") is not True
    ):
        raise PermissionError("source residual model violates the frozen raw contract")
    common = models["common_effect_cmle"]
    poisson = models["pooled_saturated_poisson"]
    destroyed = models["destroyed_link"]
    if (
        common.get("kind") != "exact_conditional_common_log_odds"
        or common.get("transport_multiplier") not in TRANSPORT_GRID
        or poisson.get("kind") != "pooled_saturated_poisson_fixed_interaction"
        or poisson.get("transport_multiplier") not in TRANSPORT_GRID
        or poisson.get("fit_certificate", {}).get("passes") is not True
        or poisson.get("fit_certificate", {}).get(
            "conditional_noncentral_hypergeometric_reconstruction"
        )
        is not False
        or destroyed.get("kind") != "destroyed_exact_conditional_log_odds"
        or destroyed.get("configuration", {}).get("graph_neighbors")
        != GRAPH_NEIGHBORS
        or destroyed.get("configuration", {}).get("heterogeneity_penalty")
        not in HETEROGENEITY_GRID
        or destroyed.get("configuration", {}).get("ridge_penalty") not in RIDGE_GRID
        or destroyed.get("configuration", {}).get("graph_penalty") not in GRAPH_GRID
        or destroyed.get("configuration", {}).get("transport_multiplier")
        not in TRANSPORT_GRID
        or models["independence"] != {"kind": "recipient_margin_independence"}
    ):
        raise PermissionError("source ablation model payload violates the frozen grids")
    for model_name in (
        "primary",
        "selected_residual",
        "common_effect_cmle",
        "pooled_saturated_poisson",
        "destroyed_link",
    ):
        coordinate = np.asarray(models[model_name].get("source_coordinate"), dtype=float)
        if coordinate.shape != (MARKER_COUNT * MARKER_COUNT,) or not np.isfinite(
            coordinate
        ).all():
            raise PermissionError("source model coordinate is incomplete or nonfinite")
        if not _finite_numeric_tree(models[model_name].get("fit_certificate", {})):
            raise PermissionError("source final fit certificate is nonfinite")

    masks = source.get("fold_specific_masks", {})
    fold_losses = source.get("fold_losses", {})
    if (
        set(masks) != set(source_axis)
        or set(fold_losses)
        != {
            "primary",
            "selected_residual",
            "common_effect_cmle",
            "pooled_saturated_poisson",
            "destroyed_link",
            "independence",
        }
        or not _finite_numeric_tree(source.get("numerical_certificates", {}))
    ):
        raise PermissionError("source fold masks or numerical certificates are incomplete")
    if source.get("numerical_certificates", {}).get("final_fits_complete") is not True:
        raise PermissionError("source final numerical fits are not certified complete")
    for donor in source_axis:
        fold = masks[donor]
        if (
            fold.get("training_donors") != [
                candidate_donor
                for candidate_donor in source_axis
                if candidate_donor != donor
            ]
            or fold.get("validation_donor") != donor
            or fold.get("training_donor_count") != 7
            or fold.get("validation_association_used") is not False
            or fold.get("recipient_paired_counts_used") is not False
            or fold.get("scored_coordinate_count", 0) < MINIMUM_COORDINATES
        ):
            raise PermissionError("source fold mask is not seven-donor training-only")
    for values in fold_losses.values():
        array = np.asarray(values, dtype=float)
        if array.shape != (len(source_axis),) or not np.isfinite(array).all():
            raise PermissionError("source selected fold losses are incomplete")

    evaluations = source.get("candidate_evaluations", {})
    if set(evaluations) != {
        "primary",
        "residual",
        "common_effect_cmle",
        "pooled_saturated_poisson",
        "destroyed_link",
    }:
        raise PermissionError("source candidate evaluation families are incomplete")
    expected_primary = [
        asdict(
            engine.PrimaryConfig(
                GRAPH_NEIGHBORS, heterogeneity, ridge, graph, alpha
            )
        )
        for heterogeneity, ridge, graph, alpha in product(
            HETEROGENEITY_GRID, RIDGE_GRID, GRAPH_GRID, TRANSPORT_GRID
        )
    ]
    expected_residual = [
        asdict(ResidualConfig(family, alpha))
        for family, alpha in product(RESIDUAL_FAMILIES, TRANSPORT_GRID)
    ]
    expected_transport = [
        {"transport_multiplier": alpha} for alpha in TRANSPORT_GRID
    ]
    selected_primary, primary_losses = _selected_published_candidate(
        evaluations["primary"], expected_primary, "primary"
    )
    selected_residual, residual_losses = _selected_published_candidate(
        evaluations["residual"], expected_residual, "residual"
    )
    selected_common, common_losses = _selected_published_candidate(
        evaluations["common_effect_cmle"], expected_transport, "common effect"
    )
    selected_poisson, poisson_losses = _selected_published_candidate(
        evaluations["pooled_saturated_poisson"], expected_transport, "pooled Poisson"
    )
    selected_destroyed, destroyed_losses = _selected_published_candidate(
        evaluations["destroyed_link"], expected_transport, "destroyed link"
    )
    if (
        source.get("selected_primary") != selected_primary
        or primary.get("configuration") != selected_primary
        or source.get("selected_residual") != selected_residual
        or residual.get("configuration") != selected_residual
        or source.get("selected_common_effect_transport")
        != selected_common["transport_multiplier"]
        or common.get("transport_multiplier")
        != selected_common["transport_multiplier"]
        or source.get("selected_pooled_poisson_transport")
        != selected_poisson["transport_multiplier"]
        or poisson.get("transport_multiplier")
        != selected_poisson["transport_multiplier"]
        or source.get("selected_destroyed_transport")
        != selected_destroyed["transport_multiplier"]
        or destroyed.get("configuration", {}).get("transport_multiplier")
        != selected_destroyed["transport_multiplier"]
        or any(
            destroyed.get("configuration", {}).get(field)
            != primary_config.get(field)
            for field in (
                "graph_neighbors",
                "heterogeneity_penalty",
                "ridge_penalty",
                "graph_penalty",
            )
        )
    ):
        raise PermissionError("source selected candidates are not bound to final models")
    selected_losses = {
        "primary": primary_losses,
        "selected_residual": residual_losses,
        "common_effect_cmle": common_losses,
        "pooled_saturated_poisson": poisson_losses,
        "destroyed_link": destroyed_losses,
    }
    for name, expected_losses in selected_losses.items():
        if not np.allclose(
            np.asarray(fold_losses[name], dtype=float),
            expected_losses,
            rtol=0.0,
            atol=0.0,
        ):
            raise PermissionError("source selected fold losses differ from candidate grid")
    independence_losses = np.asarray(fold_losses["independence"], dtype=float)
    recomputed = {
        "selected_residual": _bootstrap_comparison(
            primary_losses, residual_losses, "residual"
        ),
        "pooled_saturated_poisson": _bootstrap_comparison(
            primary_losses, poisson_losses, "pooled_poisson"
        ),
        "destroyed_link": _bootstrap_comparison(
            primary_losses, destroyed_losses, "destroyed"
        ),
        "common_effect_cmle": _bootstrap_comparison(
            primary_losses, common_losses, "common_effect_cmle"
        ),
        "independence": _bootstrap_comparison(
            primary_losses, independence_losses, "independence"
        ),
    }
    if any(
        not _source_comparison_matches(comparisons[name], expected)
        for name, expected in recomputed.items()
    ) or not all(
        comparison["passes_frozen_source_requirement"]
        for comparison in recomputed.values()
    ):
        raise PermissionError("source promotion summaries fail deterministic replay")


def _require_passing_source() -> tuple[dict[str, Any], str]:
    source, commit = _require_public_result(
        SOURCE_TAG, SOURCE_RESULT, "gse179221-bmmc-source-result/1.0"
    )
    candidate, attempt_public = _validate_source_attempt(
        SOURCE_ATTEMPT, enforce_current_runtime=False
    )
    _require_ancestor(attempt_public["source_attempt_commit"], commit)
    if _require_public_tag(SOURCE_TAG, (_relative(SOURCE_CONSUMPTION),)) != commit:
        raise PermissionError("public source consumption record is on another commit")
    consumption = _read_json(SOURCE_CONSUMPTION)
    expected_axis = [record["donor"] for record in candidate["source_files"]]
    access = source.get("access_audit", {})
    if (
        source.get("status") != "SOURCE_PROMOTION_PASS"
        or source.get("passes_source_promotion_gate") is not True
        or not isinstance(source.get("models"), dict)
        or source.get("held_h5_access_eligible_after_public_source_pass") is not True
        or source.get("source_attempt_sha256") != _sha256(SOURCE_ATTEMPT)
        or source.get("source_consumption_sha256") != _sha256(SOURCE_CONSUMPTION)
        or source.get("source_consumption") != consumption
        or consumption.get("schema") != "gse179221-bmmc-source-consumption/1.0"
        or consumption.get("status")
        != "CONSUMED_EXCLUSIVELY_BEFORE_FIRST_H5_GET"
        or consumption.get("attempt_sha256") != _sha256(SOURCE_ATTEMPT)
        or consumption.get("rerun_permitted") is not False
        or source.get("public_freezes") != attempt_public
        or source.get("implementation_amendment_sha256") != AMENDMENT_SHA256
        or source.get("source_donor_axis") != expected_axis
        or len(source.get("source_files", [])) != len(expected_axis)
        or source.get("source_files") != access.get("source_files")
        or access.get("source_h5_get_count") != len(expected_axis)
        or access.get("source_h5_deleted_count") != len(expected_axis)
        or access.get("held_h5_get_count") != 0
        or access.get("all_donor_tar_get_count") != 0
        or access.get("maximum_simultaneous_h5_files") != 1
    ):
        raise PermissionError("source result did not promote; every held URL is disabled")
    if not _completed_download_records(
        source.get("source_files"), candidate["source_files"], candidate
    ):
        raise PermissionError("source input provenance is incomplete")
    _validate_source_pass_payload(source, candidate)
    return source, commit


def claim_held_margins(attempt_path: Path = HELD_ATTEMPT) -> dict[str, Any]:
    if attempt_path.exists() or HELD_MARGINS.exists():
        raise FileExistsError("held-margin attempt is already claimed or consumed")
    source, source_commit = _require_passing_source()
    candidate = _candidate()
    payload = {
        "schema": "gse179221-bmmc-held-margin-attempt/1.0",
        "status": "CLAIMED_HELD_MARGIN_RUN_BEFORE_ANY_HELD_H5_GET",
        "created_at_utc": _timestamp(),
        "source_result_path": _relative(SOURCE_RESULT),
        "source_result_sha256": _sha256(SOURCE_RESULT),
        "source_result_public_commit": source_commit,
        "source_model_sha256": hashlib.sha256(
            json.dumps(source["models"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "held_gsm_axis": [record["gsm"] for record in candidate["held_files"]],
        "held_filename_axis": [
            record["filename"] for record in candidate["held_files"]
        ],
        "held_h5_get_begins_after_this_record": True,
        "joint_table_formation_authorized": False,
        "rerun_permitted": False,
    }
    _write_json_x(attempt_path, payload)
    return payload


def _validate_held_attempt(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    attempt = _read_json(path)
    source, source_commit = _require_passing_source()
    candidate = _candidate()
    model_sha = hashlib.sha256(
        json.dumps(source["models"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        attempt.get("schema") != "gse179221-bmmc-held-margin-attempt/1.0"
        or attempt.get("status")
        != "CLAIMED_HELD_MARGIN_RUN_BEFORE_ANY_HELD_H5_GET"
        or attempt.get("source_result_sha256") != _sha256(SOURCE_RESULT)
        or attempt.get("source_result_public_commit") != source_commit
        or attempt.get("source_model_sha256") != model_sha
        or attempt.get("held_gsm_axis")
        != [record["gsm"] for record in candidate["held_files"]]
        or attempt.get("held_filename_axis")
        != [record["filename"] for record in candidate["held_files"]]
        or attempt.get("held_h5_get_begins_after_this_record") is not True
        or attempt.get("joint_table_formation_authorized") is not False
        or attempt.get("rerun_permitted") is not False
    ):
        raise PermissionError("held-margin attempt differs from the public source seal")
    attempt_commit = _require_public_tag(HELD_ATTEMPT_TAG, (_relative(path),))
    _require_ancestor(source_commit, attempt_commit)
    return candidate, source, source_commit, attempt_commit


def _write_private_npz_x(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            stat.S_IRUSR | stat.S_IWUSR,
        )
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            np.savez_compressed(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        path.unlink(missing_ok=True)
        raise


def _merge_private_parts(parts: list[Path], output: Path) -> None:
    arrays: dict[str, np.ndarray] = {}
    for part in parts:
        with np.load(part, allow_pickle=False) as archive:
            for key in archive.files:
                if key in arrays:
                    raise ValueError("private held-state key is duplicated")
                arrays[key] = np.asarray(archive[key])
    _write_private_npz_x(output, arrays)
    del arrays
    for part in parts:
        part.unlink()


def run_held_margins(
    attempt_path: Path = HELD_ATTEMPT,
    output_path: Path = HELD_MARGINS,
    private_rna_path: Path = PRIVATE_RNA,
    private_adt_path: Path = PRIVATE_ADT,
    scratch: Path = DEFAULT_SCRATCH,
) -> dict[str, Any]:
    """Reduce held RNA and ADT separately without forming a joint table."""

    _require_canonical_path(attempt_path, HELD_ATTEMPT, "held-margin attempt")
    _require_canonical_path(output_path, HELD_MARGINS, "held-margin result")
    _require_canonical_path(private_rna_path, PRIVATE_RNA, "private RNA state")
    _require_canonical_path(private_adt_path, PRIVATE_ADT, "private ADT state")
    if output_path.exists() or private_rna_path.exists() or private_adt_path.exists():
        raise FileExistsError("held margins or private held state already exist")
    if not attempt_path.exists():
        raise FileNotFoundError("held-margin attempt does not exist")
    consumption = _claim_consumption(
        HELD_CONSUMPTION,
        "gse179221-bmmc-held-margin-consumption/1.0",
        attempt_path,
    )
    audit: dict[str, Any] = {
        "requested_urls": [],
        "source_h5_get_count": 0,
        "held_h5_get_count": 0,
        "all_donor_tar_get_count": 0,
        "held_h5_deleted_count": 0,
        "maximum_simultaneous_h5_files": 0,
        "joint_table_function_calls": 0,
        "rna_state_cleared_before_adt_reduction": True,
        "held_files": [],
    }
    public_records: list[dict[str, Any]] = []
    rna_parts: list[Path] = []
    adt_parts: list[Path] = []
    source: dict[str, Any] | None = None
    source_commit: str | None = None
    held_attempt_commit: str | None = None
    try:
        candidate, source, source_commit, held_attempt_commit = _validate_held_attempt(
            attempt_path
        )
        for sample in candidate["held_files"]:
            path: Path | None = None
            identity: dict[str, Any] | None = None
            accessed: set[str] = set()
            try:
                path, identity = _fetch_designated_file(
                    candidate, sample, scratch, audit, "held"
                )
                audit["maximum_simultaneous_h5_files"] = max(
                    audit["maximum_simultaneous_h5_files"],
                    len(list(scratch.glob("*.h5"))),
                )
                rna = _reduce_held_rna(path, sample["donor"], accessed)
                selected = rna.pop("selected_indices")
                barcodes = rna.pop("selected_barcodes")
                rna_states = rna.pop("states")
                rna_part = scratch / f".{sample['gsm']}.rna.npz"
                _write_private_npz_x(
                    rna_part,
                    {
                        f"{sample['donor']}__states": rna_states,
                        f"{sample['donor']}__barcodes": np.asarray(barcodes),
                    },
                )
                rna_parts.append(rna_part)
                del rna_states
                rna_public = rna
                del rna

                adt = _reduce_held_adt(
                    path, sample["donor"], selected, barcodes, accessed
                )
                adt_states = adt.pop("states")
                adt_part = scratch / f".{sample['gsm']}.adt.npz"
                _write_private_npz_x(
                    adt_part,
                    {
                        f"{sample['donor']}__states": adt_states,
                        f"{sample['donor']}__barcodes": np.asarray(barcodes),
                    },
                )
                adt_parts.append(adt_part)
                del adt_states
                public_records.append(
                    {
                        "gsm": sample["gsm"],
                        "donor": sample["donor"],
                        "stratum": sample["stratum"],
                        "selected_barcode_axis_sha256": rna_public[
                            "selected_barcode_axis_sha256"
                        ],
                        "eligible_cell_count": rna_public["eligible_cell_count"],
                        "rna_margins": np.asarray(rna_public["margins"]).tolist(),
                        "adt_margins": np.asarray(adt["margins"]).tolist(),
                        "rna_access_certificate": rna_public["access_certificate"],
                        "adt_access_certificate": adt["access_certificate"],
                    }
                )
                identity["reduction_status"] = "COMPLETE"
                identity["selected_barcode_axis_sha256"] = rna_public[
                    "selected_barcode_axis_sha256"
                ]
                identity["access_certificate"] = {
                    "rna": rna_public["access_certificate"],
                    "adt": adt["access_certificate"],
                }
            except Exception as error:
                if identity is not None:
                    identity["reduction_status"] = "REFUSED"
                    identity["reduction_reason_code"] = (
                        error.code
                        if isinstance(error, ProtocolRefusal)
                        else type(error).__name__
                    )
                raise
            finally:
                if path is not None:
                    path.unlink(missing_ok=True)
                    audit["held_h5_deleted_count"] += 1
                    if identity is not None:
                        identity["deleted_after_reduction"] = not path.exists()
                if identity is not None:
                    identity["decoded_h5_datasets"] = sorted(accessed)
        _merge_private_parts(rna_parts, private_rna_path)
        rna_parts.clear()
        _merge_private_parts(adt_parts, private_adt_path)
        adt_parts.clear()
        payload = {
            "schema": "gse179221-bmmc-held-margins/1.0",
            "status": "HELD_MARGINS_FROZEN_WITHOUT_PAIRING",
            "created_at_utc": _timestamp(),
            "source_result_sha256": _sha256(SOURCE_RESULT),
            "source_result_public_commit": source_commit,
            "source_model_sha256": hashlib.sha256(
                json.dumps(source["models"], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "held_margin_attempt_sha256": _sha256(attempt_path),
            "held_margin_attempt_public_commit": held_attempt_commit,
            "held_margin_consumption_sha256": _sha256(HELD_CONSUMPTION),
            "held_margin_consumption": consumption,
            "held_records": public_records,
            "input_files": audit["held_files"],
            "private_rna_state_path": _relative(private_rna_path),
            "private_rna_state_sha256": _sha256(private_rna_path),
            "private_adt_state_path": _relative(private_adt_path),
            "private_adt_state_sha256": _sha256(private_adt_path),
            "access_audit": audit,
        }
    except Exception as error:
        for path in [*rna_parts, *adt_parts]:
            path.unlink(missing_ok=True)
        private_rna_path.unlink(missing_ok=True)
        private_adt_path.unlink(missing_ok=True)
        payload = {
            "schema": "gse179221-bmmc-held-margins/1.0",
            "status": "TERMINAL_HELD_MARGIN_EXECUTION_REFUSAL",
            "created_at_utc": _timestamp(),
            "reason_code": error.code
            if isinstance(error, ProtocolRefusal)
            else type(error).__name__,
            "reason": str(error),
            "reason_details": error.details
            if isinstance(error, ProtocolRefusal)
            else {},
            "source_result_sha256": _sha256(SOURCE_RESULT)
            if SOURCE_RESULT.exists()
            else None,
            "source_result_public_commit": source_commit,
            "source_model_sha256": hashlib.sha256(
                json.dumps(source["models"], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if isinstance(source, dict) and isinstance(source.get("models"), dict)
            else None,
            "held_margin_attempt_sha256": _sha256(attempt_path)
            if attempt_path.exists()
            else None,
            "held_margin_attempt_public_commit": held_attempt_commit,
            "held_margin_consumption_sha256": _sha256(HELD_CONSUMPTION),
            "held_margin_consumption": consumption,
            "completed_held_records": public_records,
            "input_files": audit["held_files"],
            "private_state_artifacts_deleted": not private_rna_path.exists()
            and not private_adt_path.exists(),
            "access_audit": audit,
            "prediction_stage_authorized": False,
            "score_stage_authorized": False,
        }
    if (
        audit["joint_table_function_calls"] != 0
        or audit["source_h5_get_count"] != 0
        or audit["all_donor_tar_get_count"] != 0
    ):
        payload.update(
            {
                "status": "TERMINAL_HELD_MARGIN_EXECUTION_REFUSAL",
                "reason_code": "HELD_MARGIN_FIREWALL_VIOLATION",
                "reason": "held margin process crossed a frozen access boundary",
                "prediction_stage_authorized": False,
                "score_stage_authorized": False,
            }
        )
        private_rna_path.unlink(missing_ok=True)
        private_adt_path.unlink(missing_ok=True)
    _write_json_x(output_path, payload)
    return payload


def _validated_public_margins() -> tuple[dict[str, Any], str, dict[str, Any]]:
    source, source_commit = _require_passing_source()
    candidate, _, held_source_commit, held_attempt_commit = _validate_held_attempt(
        HELD_ATTEMPT
    )
    if held_source_commit != source_commit:
        raise PermissionError("held-margin attempt binds a different source result")
    margins, commit = _require_public_result(
        MARGINS_TAG, HELD_MARGINS, "gse179221-bmmc-held-margins/1.0"
    )
    _require_ancestor(held_attempt_commit, commit)
    if _require_public_tag(MARGINS_TAG, (_relative(HELD_CONSUMPTION),)) != commit:
        raise PermissionError("public held-margin consumption record is on another commit")
    consumption = _read_json(HELD_CONSUMPTION)
    expected_axis = [
        (record["gsm"], record["donor"], record["stratum"])
        for record in candidate["held_files"]
    ]
    observed_axis = [
        (record.get("gsm"), record.get("donor"), record.get("stratum"))
        for record in margins.get("held_records", [])
    ]
    expected_urls = [
        _designated_url(candidate, record) for record in candidate["held_files"]
    ]
    access = margins.get("access_audit", {})
    expected_model_sha = hashlib.sha256(
        json.dumps(source["models"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        margins.get("status") != "HELD_MARGINS_FROZEN_WITHOUT_PAIRING"
        or margins.get("source_result_sha256") != _sha256(SOURCE_RESULT)
        or margins.get("source_result_public_commit") != source_commit
        or margins.get("source_model_sha256") != expected_model_sha
        or margins.get("held_margin_attempt_sha256") != _sha256(HELD_ATTEMPT)
        or margins.get("held_margin_attempt_public_commit") != held_attempt_commit
        or margins.get("held_margin_consumption_sha256")
        != _sha256(HELD_CONSUMPTION)
        or margins.get("held_margin_consumption") != consumption
        or consumption.get("schema")
        != "gse179221-bmmc-held-margin-consumption/1.0"
        or consumption.get("status")
        != "CONSUMED_EXCLUSIVELY_BEFORE_FIRST_H5_GET"
        or consumption.get("attempt_sha256") != _sha256(HELD_ATTEMPT)
        or consumption.get("rerun_permitted") is not False
        or margins.get("private_rna_state_path") != _relative(PRIVATE_RNA)
        or margins.get("private_adt_state_path") != _relative(PRIVATE_ADT)
        or re.fullmatch(r"[0-9a-f]{64}", margins.get("private_rna_state_sha256", ""))
        is None
        or re.fullmatch(r"[0-9a-f]{64}", margins.get("private_adt_state_sha256", ""))
        is None
        or observed_axis != expected_axis
        or access.get("joint_table_function_calls") != 0
        or access.get("source_h5_get_count") != 0
        or access.get("held_h5_get_count") != len(expected_axis)
        or access.get("all_donor_tar_get_count") != 0
        or access.get("held_h5_deleted_count") != len(expected_axis)
        or access.get("maximum_simultaneous_h5_files") != 1
        or access.get("requested_urls") != expected_urls
        or len(access.get("held_files", [])) != len(expected_axis)
        or margins.get("input_files") != access.get("held_files")
    ):
        raise PermissionError("public held margins differ from the prediction firewall")
    if not _completed_download_records(
        margins.get("input_files"), candidate["held_files"], candidate
    ):
        raise PermissionError("public held-margin input provenance is incomplete")
    expected_record_keys = {
        "gsm",
        "donor",
        "stratum",
        "selected_barcode_axis_sha256",
        "eligible_cell_count",
        "rna_margins",
        "adt_margins",
        "rna_access_certificate",
        "adt_access_certificate",
    }
    expected_datasets = {
        "matrix/barcodes",
        "matrix/features/name",
        "matrix/features/feature_type",
        "matrix/shape",
        "matrix/indptr",
        "matrix/indices",
        "matrix/data",
    }
    for record, identity in zip(margins["held_records"], margins["input_files"]):
        _pair_margins(record)
        rna_access = record.get("rna_access_certificate", {})
        adt_access = record.get("adt_access_certificate", {})
        if (
            set(record) != expected_record_keys
            or not isinstance(record.get("eligible_cell_count"), int)
            or record["eligible_cell_count"] < CELL_BUDGET
            or re.fullmatch(
                r"[0-9a-f]{64}", record.get("selected_barcode_axis_sha256", "")
            )
            is None
            or identity.get("selected_barcode_axis_sha256")
            != record["selected_barcode_axis_sha256"]
            or identity.get("access_certificate")
            != {"rna": rna_access, "adt": adt_access}
            or rna_access.get("storage") != "10x_feature_by_barcode_csc"
            or rna_access.get("full_matrix_dense_materialized") is not False
            or rna_access.get("rna_qc_used_only_gene_expression_numeric_entries")
            is not True
            or rna_access.get("adt_numeric_entries_read") != 0
            or rna_access.get("selected_dense_panel_shape")
            != [CELL_BUDGET, MARKER_COUNT]
            or rna_access.get("mitochondrial_rule")
            != "Gene Expression feature name starts with MT-"
            or rna_access.get("profile") != "mean raw detection state by donor"
            or set(rna_access.get("decoded_h5_datasets", [])) != expected_datasets
            or not isinstance(rna_access.get("rna_qc_numeric_entries_read"), int)
            or rna_access["rna_qc_numeric_entries_read"] < 0
            or not isinstance(rna_access.get("rna_panel_numeric_entries_read"), int)
            or rna_access["rna_panel_numeric_entries_read"] < 0
            or adt_access.get("storage") != "10x_feature_by_barcode_csc"
            or adt_access.get("full_matrix_dense_materialized") is not False
            or adt_access.get("rna_numeric_entries_read") != 0
            or adt_access.get("selected_dense_panel_shape")
            != [CELL_BUDGET, MARKER_COUNT]
            or adt_access.get("profile") != "mean log1p raw count by donor"
            or set(adt_access.get("decoded_h5_datasets", [])) != expected_datasets
            or not isinstance(adt_access.get("adt_panel_numeric_entries_read"), int)
            or adt_access["adt_panel_numeric_entries_read"] < 0
        ):
            raise PermissionError("public held-margin numeric/access contract failed")
    return margins, commit, source


def _pair_margins(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    raw_rna = np.asarray(record["rna_margins"])
    raw_adt = np.asarray(record["adt_margins"])
    rna = np.asarray(raw_rna, dtype=float)
    adt = np.asarray(raw_adt, dtype=float)
    if (
        rna.shape != (MARKER_COUNT, 2)
        or adt.shape != rna.shape
        or not np.isfinite(rna).all()
        or not np.isfinite(adt).all()
        or np.any(rna < 0)
        or np.any(adt < 0)
        or not np.array_equal(rna, np.rint(rna))
        or not np.array_equal(adt, np.rint(adt))
        or not np.all(rna.sum(axis=1) == CELL_BUDGET)
        or not np.all(adt == np.asarray([CELL_BUDGET // 2, CELL_BUDGET // 2]))
    ):
        raise ValueError("held marker margins have the wrong shape")
    rows = np.broadcast_to(rna[:, None, :], (MARKER_COUNT, MARKER_COUNT, 2)).copy()
    columns = np.broadcast_to(adt[None, :, :], rows.shape).copy()
    return rows, columns


def _predict_models(
    models: dict[str, Any], rows: np.ndarray, columns: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    source_mask = np.asarray(models["final_mask"], dtype=bool)
    mask = source_mask & _recipient_margin_support(rows, columns)
    if int(mask.sum()) < MINIMUM_COORDINATES:
        raise ProtocolRefusal("HELD_MARGIN_MASK_BELOW_64")
    primary = models["primary"]
    primary_coordinate = np.asarray(primary["source_coordinate"], dtype=float).reshape(
        MARKER_COUNT, MARKER_COUNT
    )
    residual = models["selected_residual"]
    residual_config = ResidualConfig(
        residual["configuration"]["family"],
        residual["configuration"]["transport_multiplier"],
    )
    common = models["common_effect_cmle"]
    poisson = models["pooled_saturated_poisson"]
    destroyed = models["destroyed_link"]
    predictions = {
        "primary": _predict_conditional(
            primary_coordinate,
            rows,
            columns,
            primary["configuration"]["transport_multiplier"],
            mask,
        ),
        "selected_residual": _predict_residual(
            np.asarray(residual["source_coordinate"], dtype=float).reshape(
                MARKER_COUNT, MARKER_COUNT
            ),
            rows,
            columns,
            residual_config,
            mask,
        ),
        "common_effect_cmle": _predict_conditional(
            np.asarray(common["source_coordinate"], dtype=float).reshape(
                MARKER_COUNT, MARKER_COUNT
            ),
            rows,
            columns,
            common["transport_multiplier"],
            mask,
        ),
        "destroyed_link": _predict_conditional(
            np.asarray(destroyed["source_coordinate"], dtype=float).reshape(
                MARKER_COUNT, MARKER_COUNT
            ),
            rows,
            columns,
            destroyed["configuration"]["transport_multiplier"],
            mask,
        ),
        "independence": _independence(rows, columns),
    }
    pooled, pooled_certificate = _predict_pooled_poisson(
        np.asarray(poisson["source_coordinate"], dtype=float).reshape(
            MARKER_COUNT, MARKER_COUNT
        ),
        rows,
        columns,
        poisson["transport_multiplier"],
        mask,
    )
    predictions["pooled_saturated_poisson"] = pooled
    return mask, predictions, pooled_certificate


def run_prediction(output_path: Path = HELD_PREDICTIONS) -> dict[str, Any]:
    """Predict from public source coordinates and public margins only."""

    _require_canonical_path(output_path, HELD_PREDICTIONS, "held prediction")
    if output_path.exists():
        raise FileExistsError("held prediction already exists")
    margins, margins_commit, source = _validated_public_margins()
    records = []
    for margin_record in margins["held_records"]:
        rows, columns = _pair_margins(margin_record)
        mask, predictions, pooled_certificate = _predict_models(
            source["models"], rows, columns
        )
        records.append(
            {
                "gsm": margin_record["gsm"],
                "donor": margin_record["donor"],
                "stratum": margin_record["stratum"],
                "selected_barcode_axis_sha256": margin_record[
                    "selected_barcode_axis_sha256"
                ],
                "rna_margins": margin_record["rna_margins"],
                "adt_margins": margin_record["adt_margins"],
                "comparison_mask": mask.astype(np.uint8).tolist(),
                "comparison_mask_sha256": _array_sha256(mask.astype(np.uint8)),
                "scored_coordinate_count": int(mask.sum()),
                "predictions": {
                    name: table.reshape(MARKER_COUNT * MARKER_COUNT, 4).tolist()
                    for name, table in predictions.items()
                },
                "pooled_poisson_prediction_certificate": pooled_certificate,
            }
        )
    payload = {
        "schema": "gse179221-bmmc-held-predictions/1.0",
        "status": "HELD_PREDICTIONS_FROZEN_BEFORE_PAIRING",
        "created_at_utc": _timestamp(),
        "source_result_sha256": _sha256(SOURCE_RESULT),
        "held_margins_sha256": _sha256(HELD_MARGINS),
        "held_margins_public_commit": margins_commit,
        "held_records": records,
        "access_audit": {
            "private_rna_state_reads": 0,
            "private_adt_state_reads": 0,
            "joint_tables_formed": 0,
            "network_h5_gets": 0,
        },
    }
    _write_json_x(output_path, payload)
    return payload


def _validated_public_prediction(
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    margins, margins_commit, source = _validated_public_margins()
    prediction, prediction_commit = _require_public_result(
        PREDICTION_TAG,
        HELD_PREDICTIONS,
        "gse179221-bmmc-held-predictions/1.0",
    )
    _require_ancestor(margins_commit, prediction_commit)
    margin_axis = [
        (record["gsm"], record["donor"], record["stratum"])
        for record in margins["held_records"]
    ]
    prediction_axis = [
        (record.get("gsm"), record.get("donor"), record.get("stratum"))
        for record in prediction.get("held_records", [])
    ]
    access = prediction.get("access_audit", {})
    if (
        prediction.get("status") != "HELD_PREDICTIONS_FROZEN_BEFORE_PAIRING"
        or prediction.get("source_result_sha256") != _sha256(SOURCE_RESULT)
        or prediction.get("held_margins_sha256") != _sha256(HELD_MARGINS)
        or prediction.get("held_margins_public_commit") != margins_commit
        or prediction_axis != margin_axis
        or access.get("private_rna_state_reads") != 0
        or access.get("private_adt_state_reads") != 0
        or access.get("joint_tables_formed") != 0
        or access.get("network_h5_gets") != 0
    ):
        raise PermissionError("public held predictions crossed the pairing firewall")
    for published, margin_record in zip(
        prediction["held_records"], margins["held_records"]
    ):
        rows, columns = _pair_margins(margin_record)
        expected_mask, expected_predictions, expected_poisson_certificate = (
            _predict_models(source["models"], rows, columns)
        )
        published_mask = np.asarray(published.get("comparison_mask"), dtype=np.uint8)
        if (
            published.get("selected_barcode_axis_sha256")
            != margin_record["selected_barcode_axis_sha256"]
            or published.get("rna_margins") != margin_record["rna_margins"]
            or published.get("adt_margins") != margin_record["adt_margins"]
            or published_mask.shape != (MARKER_COUNT, MARKER_COUNT)
            or not np.array_equal(published_mask, expected_mask.astype(np.uint8))
            or published.get("comparison_mask_sha256")
            != _array_sha256(expected_mask.astype(np.uint8))
            or published.get("scored_coordinate_count") != int(expected_mask.sum())
            or set(published.get("predictions", {})) != set(expected_predictions)
            or published.get("pooled_poisson_prediction_certificate")
            != expected_poisson_certificate
        ):
            raise PermissionError("public prediction record differs from deterministic replay")
        for method, expected_table in expected_predictions.items():
            observed_table = np.asarray(published["predictions"][method], dtype=float)
            if observed_table.shape != (MARKER_COUNT * MARKER_COUNT, 4) or not np.allclose(
                observed_table,
                expected_table.reshape(MARKER_COUNT * MARKER_COUNT, 4),
                rtol=1e-12,
                atol=1e-12,
            ):
                raise PermissionError(
                    "public prediction table differs from deterministic replay"
                )
    return prediction, prediction_commit, margins, source


def authorize_score(output_path: Path = SCORE_AUTHORIZATION) -> dict[str, Any]:
    """Bind the public predictions without reading either private state artifact."""

    _require_canonical_path(output_path, SCORE_AUTHORIZATION, "score authorization")
    if output_path.exists() or SCORE_ATTEMPT.exists() or SCORE_RESULT.exists():
        raise FileExistsError("score authorization is already claimed or consumed")
    prediction, prediction_commit, margins, _ = _validated_public_prediction()
    payload = {
        "schema": "gse179221-bmmc-score-authorization/1.0",
        "status": "AUTHORIZED_AFTER_PUBLIC_PREDICTION_FREEZE",
        "created_at_utc": _timestamp(),
        "source_result_sha256": _sha256(SOURCE_RESULT),
        "held_margins_sha256": _sha256(HELD_MARGINS),
        "held_predictions_sha256": _sha256(HELD_PREDICTIONS),
        "held_predictions_public_commit": prediction_commit,
        "held_donor_axis": [record["donor"] for record in prediction["held_records"]],
        "private_rna_state_sha256": margins["private_rna_state_sha256"],
        "private_adt_state_sha256": margins["private_adt_state_sha256"],
        "private_state_reads_before_authorization": 0,
        "joint_tables_formed_before_authorization": 0,
        "score_pairing_authorized": True,
        "rerun_permitted": False,
    }
    _write_json_x(output_path, payload)
    return payload


def _validated_score_authorization(
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    prediction, prediction_commit, margins, source = _validated_public_prediction()
    authorization_commit = _require_public_tag(
        SCORE_AUTHORIZATION_TAG, (_relative(SCORE_AUTHORIZATION),)
    )
    _require_ancestor(prediction_commit, authorization_commit)
    authorization = _read_json(SCORE_AUTHORIZATION)
    if (
        authorization.get("schema")
        != "gse179221-bmmc-score-authorization/1.0"
        or authorization.get("status")
        != "AUTHORIZED_AFTER_PUBLIC_PREDICTION_FREEZE"
        or authorization.get("source_result_sha256") != _sha256(SOURCE_RESULT)
        or authorization.get("held_margins_sha256") != _sha256(HELD_MARGINS)
        or authorization.get("held_predictions_sha256")
        != _sha256(HELD_PREDICTIONS)
        or authorization.get("held_predictions_public_commit") != prediction_commit
        or authorization.get("held_donor_axis")
        != [record["donor"] for record in prediction["held_records"]]
        or authorization.get("private_rna_state_sha256")
        != margins.get("private_rna_state_sha256")
        or authorization.get("private_adt_state_sha256")
        != margins.get("private_adt_state_sha256")
        or authorization.get("private_state_reads_before_authorization") != 0
        or authorization.get("joint_tables_formed_before_authorization") != 0
        or authorization.get("score_pairing_authorized") is not True
        or authorization.get("rerun_permitted") is not False
    ):
        raise PermissionError("public score authorization does not bind the frozen prediction")
    return authorization, authorization_commit, prediction, margins, source


def _load_private_states(
    path: Path, held_records: list[dict[str, Any]]
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & (stat.S_IRWXG | stat.S_IRWXO) or not mode & stat.S_IRUSR:
        raise PermissionError("private held state permissions are not owner-only")
    expected_keys = {
        f"{record['donor']}__{suffix}"
        for record in held_records
        for suffix in ("states", "barcodes")
    }
    states: dict[str, np.ndarray] = {}
    barcode_hashes: dict[str, str] = {}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            raise PermissionError("private held state donor axis differs from public margins")
        for record in held_records:
            donor = record["donor"]
            values = np.asarray(archive[f"{donor}__states"])
            barcodes = np.asarray(archive[f"{donor}__barcodes"])
            if (
                values.shape != (CELL_BUDGET, MARKER_COUNT)
                or values.dtype.kind not in "biu"
                or not np.isin(values, (0, 1)).all()
                or barcodes.shape != (CELL_BUDGET,)
            ):
                raise PermissionError("private held state array violates the frozen shape")
            barcode_axis = [str(value) for value in barcodes]
            barcode_sha = _axis_sha256(barcode_axis)
            if (
                len(barcode_axis) != len(set(barcode_axis))
                or barcode_sha != record["selected_barcode_axis_sha256"]
            ):
                raise PermissionError("private held barcode axis differs from public margins")
            states[donor] = values.astype(np.uint8, copy=False)
            barcode_hashes[donor] = barcode_sha
    return states, barcode_hashes


def _held_comparison(
    primary: np.ndarray, comparator: np.ndarray, role: str, confirmatory: bool
) -> dict[str, Any]:
    first = np.asarray(primary, dtype=float)
    second = np.asarray(comparator, dtype=float)
    if first.shape != (10,) or second.shape != first.shape or np.any(second <= 0):
        raise ValueError("held comparison requires ten finite positive comparator losses")
    difference = first - second
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws = generator.integers(0, len(first), size=(BOOTSTRAPS, len(first)))
    interval = np.quantile(difference[draws].mean(axis=1), [0.025, 0.975])
    favorable = int(np.count_nonzero(difference < 0.0))
    sign_p = float(
        sum(math.comb(len(first), count) for count in range(favorable, len(first) + 1))
        / (2 ** len(first))
    )
    relative = 1.0 - float(first.mean() / second.mean())
    criteria = {
        "relative_loss_reduction_at_least_0_05": bool(relative >= 0.05),
        "paired_bootstrap_upper_endpoint_below_zero": bool(interval[1] < 0.0),
        "at_least_nine_of_ten_donors_favorable": bool(favorable >= 9),
        "exact_one_sided_sign_test_p_at_most_0_025": bool(sign_p <= 0.025),
    }
    return {
        "role": role,
        "primary_mean_loss": float(first.mean()),
        "comparator_mean_loss": float(second.mean()),
        "relative_loss_reduction": relative,
        "paired_donor_difference_95_ci": interval.tolist(),
        "favorable_donors": favorable,
        "donor_count": len(first),
        "exact_one_sided_sign_test_p": sign_p,
        "bootstrap_draws": BOOTSTRAPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "confirmatory_comparator": confirmatory,
        "criteria": criteria,
        "passes_frozen_confirmation_requirement": bool(all(criteria.values()))
        if confirmatory
        else None,
    }


def _score_predictions(
    prediction: dict[str, Any],
    margins: dict[str, Any],
    rna_states: dict[str, np.ndarray],
    adt_states: dict[str, np.ndarray],
) -> dict[str, Any]:
    methods = (
        "primary",
        "selected_residual",
        "common_effect_cmle",
        "pooled_saturated_poisson",
        "destroyed_link",
        "independence",
    )
    losses = {method: np.full(10, np.nan) for method in methods}
    truth_hashes: dict[str, str] = {}
    donor_records: list[dict[str, Any]] = []
    margin_by_donor = {record["donor"]: record for record in margins["held_records"]}
    for donor_index, predicted_record in enumerate(prediction["held_records"]):
        donor = predicted_record["donor"]
        margin_record = margin_by_donor[donor]
        truth = _joint_tables(rna_states[donor], adt_states[donor])
        truth_hashes[donor] = _array_sha256(truth)
        mask = np.asarray(predicted_record["comparison_mask"], dtype=bool)
        rows, columns = _pair_margins(margin_record)
        if (
            mask.shape != (MARKER_COUNT, MARKER_COUNT)
            or _array_sha256(mask.astype(np.uint8))
            != predicted_record["comparison_mask_sha256"]
            or int(mask.sum()) != predicted_record["scored_coordinate_count"]
            or set(predicted_record.get("predictions", {})) != set(methods)
            or not np.array_equal(truth.sum(axis=-1), rows)
            or not np.array_equal(truth.sum(axis=-2), columns)
        ):
            raise PermissionError("held truth or prediction mask violates the frozen margins")
        donor_losses = {}
        for method in methods:
            table = np.asarray(predicted_record["predictions"][method], dtype=float)
            if table.shape != (MARKER_COUNT * MARKER_COUNT, 4):
                raise PermissionError("held prediction table has the wrong shape")
            value = _loss(
                truth,
                table.reshape(MARKER_COUNT, MARKER_COUNT, 2, 2),
                mask,
            )
            losses[method][donor_index] = value
            donor_losses[method] = value
        donor_records.append(
            {
                "gsm": predicted_record["gsm"],
                "donor": donor,
                "stratum": predicted_record["stratum"],
                "scored_coordinate_count": int(mask.sum()),
                "losses": donor_losses,
            }
        )

    comparisons = {
        "selected_residual": _held_comparison(
            losses["primary"], losses["selected_residual"], "residual", True
        ),
        "pooled_saturated_poisson": _held_comparison(
            losses["primary"],
            losses["pooled_saturated_poisson"],
            "pooled_poisson",
            True,
        ),
        "destroyed_link": _held_comparison(
            losses["primary"], losses["destroyed_link"], "destroyed", True
        ),
        "independence": _held_comparison(
            losses["primary"], losses["independence"], "independence", False
        ),
    }
    common_effect_pass = bool(
        losses["primary"].mean() < losses["common_effect_cmle"].mean()
    )
    passes = common_effect_pass and all(
        comparison["passes_frozen_confirmation_requirement"]
        for name, comparison in comparisons.items()
        if name != "independence"
    )
    strata = sorted({record["stratum"] for record in donor_records})
    stratum_means = {
        stratum: {
            method: float(
                np.mean(
                    [
                        record["losses"][method]
                        for record in donor_records
                        if record["stratum"] == stratum
                    ]
                )
            )
            for method in methods
        }
        for stratum in strata
    }
    leave_one_donor_out_means = {
        record["donor"]: {
            method: float(np.delete(losses[method], index).mean())
            for method in methods
        }
        for index, record in enumerate(donor_records)
    }
    return {
        "status": "CONFIRMATION_PASS"
        if passes
        else "COMPLETED_HELD_CONFIRMATION_NEGATIVE",
        "passes_frozen_confirmation_gate": passes,
        "donor_records": donor_records,
        "donor_equal_mean_losses": {
            method: float(values.mean()) for method, values in losses.items()
        },
        "comparisons": comparisons,
        "common_effect_point_ablation": {
            "primary_mean_loss": float(losses["primary"].mean()),
            "common_effect_mean_loss": float(losses["common_effect_cmle"].mean()),
            "primary_point_loss_is_lower": common_effect_pass,
        },
        "disease_stratum_mean_losses": stratum_means,
        "leave_one_donor_out_mean_losses": leave_one_donor_out_means,
        "observed_table_sha256": truth_hashes,
        "joint_table_count": len(donor_records) * MARKER_COUNT * MARKER_COUNT,
    }


def score_held(
    attempt_path: Path = SCORE_ATTEMPT,
    output_path: Path = SCORE_RESULT,
    private_rna_path: Path = PRIVATE_RNA,
    private_adt_path: Path = PRIVATE_ADT,
) -> dict[str, Any]:
    """Pair the separately sealed state artifacts only after public authorization."""

    _require_canonical_path(attempt_path, SCORE_ATTEMPT, "score attempt")
    _require_canonical_path(output_path, SCORE_RESULT, "held score")
    _require_canonical_path(private_rna_path, PRIVATE_RNA, "private RNA state")
    _require_canonical_path(private_adt_path, PRIVATE_ADT, "private ADT state")
    if attempt_path.exists() or output_path.exists():
        raise FileExistsError("held score is already claimed or consumed")
    authorization, authorization_commit, prediction, margins, _ = (
        _validated_score_authorization()
    )
    if (
        margins.get("private_rna_state_path") != _relative(private_rna_path)
        or margins.get("private_adt_state_path") != _relative(private_adt_path)
        or not private_rna_path.is_file()
        or not private_adt_path.is_file()
    ):
        raise PermissionError("private held states do not match the public margin seal")
    for private_path in (private_rna_path, private_adt_path):
        mode = stat.S_IMODE(private_path.stat().st_mode)
        if mode & (stat.S_IRWXG | stat.S_IRWXO) or not mode & stat.S_IRUSR:
            raise PermissionError("private held state permissions are not owner-only")
    attempt = {
        "schema": "gse179221-bmmc-score-attempt/1.0",
        "status": "CLAIMED_AFTER_PUBLIC_SCORE_AUTHORIZATION",
        "created_at_utc": _timestamp(),
        "score_authorization_sha256": _sha256(SCORE_AUTHORIZATION),
        "score_authorization_public_commit": authorization_commit,
        "held_predictions_sha256": _sha256(HELD_PREDICTIONS),
        "expected_private_rna_state_sha256": authorization[
            "private_rna_state_sha256"
        ],
        "expected_private_adt_state_sha256": authorization[
            "private_adt_state_sha256"
        ],
        "private_state_read_begins_after_this_record": True,
        "network_and_h5_access_after_this_record": False,
        "rerun_permitted": False,
    }
    _write_json_x(attempt_path, attempt)
    try:
        if (
            _sha256(private_rna_path) != margins["private_rna_state_sha256"]
            or _sha256(private_adt_path) != margins["private_adt_state_sha256"]
        ):
            raise PermissionError("private held state hash differs from public margins")
        rna_states, rna_barcode_hashes = _load_private_states(
            private_rna_path, margins["held_records"]
        )
        adt_states, adt_barcode_hashes = _load_private_states(
            private_adt_path, margins["held_records"]
        )
        if rna_barcode_hashes != adt_barcode_hashes:
            raise PermissionError("private RNA and ADT barcode axes differ")
        payload = _score_predictions(
            prediction, margins, rna_states, adt_states
        )
        payload.update(
            {
                "schema": "gse179221-bmmc-confirmation-result/1.0",
                "created_at_utc": _timestamp(),
                "score_attempt_sha256": _sha256(attempt_path),
                "score_authorization_sha256": _sha256(SCORE_AUTHORIZATION),
                "score_authorization_public_commit": authorization_commit,
                "held_predictions_sha256": _sha256(HELD_PREDICTIONS),
                "held_margins_sha256": _sha256(HELD_MARGINS),
                "private_state_sha256": {
                    "rna": margins["private_rna_state_sha256"],
                    "adt": margins["private_adt_state_sha256"],
                },
                "access_audit": {
                    "private_rna_state_reads": 1,
                    "private_adt_state_reads": 1,
                    "joint_table_constructor_calls": 10,
                    "joint_tables_formed": 810,
                    "network_calls_after_score_attempt": 0,
                    "h5_opens_after_score_attempt": 0,
                },
            }
        )
    except Exception as error:
        payload = {
            "schema": "gse179221-bmmc-confirmation-result/1.0",
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
            "score_authorization_public_commit": authorization_commit,
            "held_predictions_sha256": _sha256(HELD_PREDICTIONS),
            "held_margins_sha256": _sha256(HELD_MARGINS),
            "access_audit": {
                "network_calls_after_score_attempt": 0,
                "h5_opens_after_score_attempt": 0,
            },
        }
    _write_json_x(output_path, payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("claim-source")
    commands.add_parser("run-source")
    commands.add_parser("claim-held-margins")
    commands.add_parser("run-held-margins")
    commands.add_parser("predict-held")
    commands.add_parser("authorize-score")
    commands.add_parser("score-held")
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    actions = {
        "claim-source": claim_source,
        "run-source": run_source,
        "claim-held-margins": claim_held_margins,
        "run-held-margins": run_held_margins,
        "predict-held": run_prediction,
        "authorize-score": authorize_score,
        "score-held": score_held,
    }
    result = actions[arguments.command]()
    print(json.dumps({"status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
