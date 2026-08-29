"""One-shot held-batch confirmation in GSE309593 bone-marrow CITE-seq.

``source`` reads the 14 designated source H5/ADT pairs. ``predict`` reads only
allowlisted RNA datasets under ``matrix/`` in the nine held H5 files and freezes
fixed-margin predictions. ``score`` first reads the separate held ADT files and
first forms held RNA--ADT tables. Each public stage is claimed once and binds
all predecessor, protocol, implementation, and input bytes by SHA-256.
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
import os
import platform
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Optional, Union
from urllib.parse import quote
from urllib.request import urlopen

import h5py
import numpy as np
import scipy

from mapreg.heterogeneity_adaptive_coupling import (
    binary_table_from_helmert_coordinate,
    CouplingEstimationRefusal,
    expected_binary_table_from_log_odds,
    signed_deviance_coordinate,
    signed_pearson_coordinate,
)
from mapreg.common_effect_conditional import (
    fit_common_effect_conditional_log_odds,
)
from mapreg.hierarchical_conditional_coupling import (
    fit_hierarchical_conditional_log_odds,
)


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ORIGIN = "https://github.com/sushaan-k/coupling-fields-benchmark.git"
DATA_DIR = ROOT / "data/confirmation/gse309593_held_batches"
DEFAULT_DESIGNATION = DATA_DIR / "candidate_designation_v1.json"
DEFAULT_AMENDMENT = DATA_DIR / "candidate_amendment_v2.json"
DEFAULT_METADATA = (
    ROOT / "data/confirmation/gse309593_independent_study/candidate_designation_v1.json"
)
DEFAULT_PREFLIGHT = (
    ROOT / "results/development/gse309593_held_batches_axis_preflight_v1.json"
)
DEFAULT_PROTOCOL = DATA_DIR / "protocol_v1.json"
DEFAULT_RUNTIME = DATA_DIR / "runtime_environment_v1.json"
DEFAULT_AUTHORIZATION_TEMPLATE = DATA_DIR / "score_authorization_template_v1.json"
DEFAULT_SOURCE_ATTEMPT = DATA_DIR / "source_attempt_v1.json"
DEFAULT_PREDICTION_ATTEMPT = DATA_DIR / "prediction_attempt_v1.json"
DEFAULT_AUTHORIZATION_ATTEMPT = DATA_DIR / "score_authorization_attempt_v1.json"
DEFAULT_SCORE_ATTEMPT = DATA_DIR / "score_attempt_v1.json"
DEFAULT_SOURCE = ROOT / "results/development/gse309593_held_batches_source_v1.json"
DEFAULT_PREDICTION = ROOT / "results/gse309593_held_batches_predictions_v1.json"
DEFAULT_SCORE_AUTHORIZATION = (
    ROOT / "results/gse309593_held_batches_score_authorization_v1.json"
)
DEFAULT_SCORE = ROOT / "results/gse309593_held_batches_confirmation_v1.json"
DEFAULT_PRIVATE_RNA = DATA_DIR / "private_held_rna_states_v1.npz"
DEFAULT_SCRATCH = Path("/private/tmp/gse309593-held-batches")

CELL_BUDGET = 512
MARKER_COUNT = 24
SOURCE_BATCHES = ("B092", "B099", "B110", "B129")
HELD_BATCHES = ("B162", "B208", "B210")
CELL_SALT = "GSE309593-HELD-BATCH-CELL-BUDGET-v1"
ADT_TIE_SALT = "GSE309593-HELD-BATCH-ADT-MIDRANK-v1"
DESTROYED_SALT = "GSE309593-HELD-BATCH-DESTROYED-LINK-v1"
BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 20260829
SENSITIVITY_BOOTSTRAP_SEED = 20260830
MINIMUM_INFORMATIVE_ENTITIES = 64
MAXIMUM_CONDITION_NUMBER = 1e12
MINIMUM_DETECTED_GENES = 200
MAXIMUM_MITOCHONDRIAL_FRACTION = 0.10
MAXIMUM_RNA_UMIS = 70_000

NEIGHBOR_GRID = (2,)
HETEROGENEITY_GRID = (0.1, 1.0, 10.0)
RIDGE_GRID = (0.01, 0.1)
GRAPH_GRID = (0.0, 0.03, 0.3)
TRANSPORT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
RESIDUAL_FAMILIES = ("pearson", "root_deviance")


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


@dataclass(frozen=True, order=True)
class OddsConfig:
    method: str
    transport_multiplier: float


class SelectionRefusal(CouplingEstimationRefusal):
    def __init__(self, message: str, details: dict[str, Any]):
        super().__init__(message)
        self.details = details


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


def _newline_axis_sha256(values: Iterable[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _write_json_x(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _claim(path: Path, phase: str, bindings: dict[str, str]) -> None:
    _write_json_x(
        path,
        {
            "schema": "gse309593-held-batches-attempt/1.0",
            "status": "CLAIMED_ONE_SHOT",
            "phase": phase,
            "created_at_utc": _timestamp(),
            "bindings": bindings,
        },
    )


def _validate_attempt(path: Path, phase: str, bindings: dict[str, str]) -> None:
    attempt = _read_json(path)
    if (
        attempt.get("schema") != "gse309593-held-batches-attempt/1.0"
        or attempt.get("status") != "CLAIMED_ONE_SHOT"
        or attempt.get("phase") != phase
        or attempt.get("bindings") != bindings
    ):
        raise PermissionError(f"{phase} attempt is absent or bound to different bytes")


def _claimed_attempt_bindings(path: Path, phase: str) -> dict[str, str]:
    try:
        attempt = _read_json(path)
    except Exception:
        return {}
    bindings = attempt.get("bindings")
    if (
        attempt.get("schema") != "gse309593-held-batches-attempt/1.0"
        or attempt.get("status") != "CLAIMED_ONE_SHOT"
        or attempt.get("phase") != phase
        or not isinstance(bindings, dict)
        or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in bindings.items()
        )
    ):
        return {}
    return dict(bindings)


def _require_computed_bindings(
    computed: dict[str, str], claimed: dict[str, str]
) -> None:
    if computed != claimed:
        raise PermissionError("claimed stage bindings differ from validated bytes")


def _validate_runtime(path: Path) -> None:
    value = _read_json(path)
    if (
        value.get("schema") != "gse309593-held-batches-runtime-environment/1.0"
        or value.get("status") != "FROZEN_WITH_PROTOCOL"
    ):
        raise PermissionError("runtime contract is not frozen")
    required = value["required_runtime"]
    python = required["python"]
    packages = required["packages"]
    hdf5 = required["hdf5"]
    system = "macOS" if platform.system() == "Darwin" else platform.system()
    checks = {
        "python implementation": platform.python_implementation()
        == python["implementation"],
        "python version": platform.python_version() == python["version"],
        "python executable": str(Path(sys.executable).resolve())
        == python["resolved_executable"],
        "numpy": np.__version__ == packages["numpy"],
        "scipy": scipy.__version__ == packages["scipy"],
        "h5py": h5py.__version__ == packages["h5py"],
        "HDF5": list(h5py.version.hdf5_version_tuple) == hdf5["runtime_version_tuple"],
        "HDF5 built-against": list(h5py.version.hdf5_built_version_tuple)
        == hdf5["built_against_version_tuple"],
        "h5py API": list(h5py.version.api_version_tuple)
        == hdf5["h5py_api_version_tuple"],
        "operating system": system == required["platform"]["operating_system"],
        "architecture": platform.machine() == required["platform"]["architecture"],
    }
    for name, expected in required["thread_environment"].items():
        checks[f"environment {name}"] = os.environ.get(name) == expected
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise PermissionError("runtime differs at " + ", ".join(failures))


def _implementation_bindings(protocol: dict[str, Any]) -> dict[str, str]:
    specification = protocol.get("implementation_bindings", {})
    path_keys = {
        "gitignore_sha256": "gitignore_path",
        "runner_sha256": "runner_path",
        "test_sha256": "test_path",
        "common_effect_solver_sha256": "coordinatewise_common_effect_solver_path",
        "hierarchical_module_sha256": "hierarchical_solver_path",
        "coupling_module_sha256": "coupling_module_path",
        "classical_residuals_module_sha256": "classical_residuals_module_path",
        "coupling_fields_module_sha256": "coupling_fields_module_path",
        "table_prediction_module_sha256": "table_prediction_module_path",
        "mapreg_init_sha256": "mapreg_init_path",
        "factorial_coupling_module_sha256": "factorial_coupling_module_path",
    }
    hash_keys = {
        "gitignore_sha256": "gitignore_sha256",
        "runner_sha256": "runner_sha256",
        "test_sha256": "test_sha256",
        "common_effect_solver_sha256": "coordinatewise_common_effect_solver_sha256",
        "hierarchical_module_sha256": "hierarchical_solver_sha256",
        "coupling_module_sha256": "coupling_module_sha256",
        "classical_residuals_module_sha256": "classical_residuals_module_sha256",
        "coupling_fields_module_sha256": "coupling_fields_module_sha256",
        "table_prediction_module_sha256": "table_prediction_module_sha256",
        "mapreg_init_sha256": "mapreg_init_sha256",
        "factorial_coupling_module_sha256": "factorial_coupling_module_sha256",
    }
    output: dict[str, str] = {}
    for output_key, path_key in path_keys.items():
        relative = specification.get(path_key)
        expected = specification.get(hash_keys[output_key])
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise PermissionError("protocol implementation binding is incomplete")
        if (
            expected == "PENDING_PROTOCOL_FREEZE"
            or _sha256(ROOT / relative) != expected
        ):
            raise PermissionError(
                f"protocol implementation binding differs: {relative}"
            )
        output[output_key] = expected
    return output


def _base_bindings(
    designation_path: Path,
    amendment_path: Path,
    metadata_path: Path,
    preflight_path: Path,
    protocol_path: Path,
    runtime_path: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, str],
]:
    _require_official_base_paths(
        designation_path,
        amendment_path,
        metadata_path,
        preflight_path,
        protocol_path,
        runtime_path,
    )
    designation, source, held = _designation(
        designation_path, amendment_path, metadata_path, preflight_path
    )
    _validate_runtime(runtime_path)
    protocol = _read_json(protocol_path)
    if (
        protocol.get("schema") != "gse309593-held-batches-protocol/1.0"
        or protocol.get("status")
        != "FROZEN_AFTER_AXIS_ONLY_PREFLIGHT_BEFORE_NUMERIC_ASSAY_ACCESS"
    ):
        raise PermissionError("analysis protocol is not frozen pre-outcome")
    candidate_freeze = protocol.get("candidate_freeze", {})
    frozen_hashes = {
        "candidate_sha256": _sha256(designation_path),
        "amendment_sha256": _sha256(amendment_path),
        "axis_preflight_sha256": _sha256(preflight_path),
        "metadata_binding_sha256": _sha256(metadata_path),
    }
    protocol_keys = {
        "candidate_sha256": "candidate_v1_sha256",
        "amendment_sha256": "amendment_v2_sha256",
        "axis_preflight_sha256": "axis_preflight_sha256",
        "metadata_binding_sha256": "metadata_binding_sha256",
    }
    if any(
        candidate_freeze.get(protocol_keys[key]) != value
        for key, value in frozen_hashes.items()
    ):
        raise PermissionError("protocol does not bind candidate and preflight bytes")
    zero_access = (
        "numeric_rna_values_read_before_protocol_freeze",
        "numeric_adt_values_read_before_protocol_freeze",
        "barcodes_or_cell_identifiers_read_before_protocol_freeze",
        "joint_tables_or_losses_computed_before_protocol_freeze",
    )
    if any(candidate_freeze.get(key) != 0 for key in zero_access):
        raise PermissionError("protocol was not frozen before numeric assay access")
    code = _implementation_bindings(protocol)
    return (
        designation,
        source,
        held,
        {
            **frozen_hashes,
            "preflight_sha256": _sha256(preflight_path),
            "protocol_sha256": _sha256(protocol_path),
            "runtime_sha256": _sha256(runtime_path),
            "authorization_template_sha256": _sha256(DEFAULT_AUTHORIZATION_TEMPLATE),
            **code,
        },
    )


EXPECTED_PANEL = (
    ("CD1C", "CD1c"),
    ("CD2", "CD2"),
    ("CD4", "CD4"),
    ("CD7", "CD7"),
    ("CD8A", "CD8"),
    ("ITGAM", "CD11b"),
    ("ITGAX", "CD11c"),
    ("CD14", "CD14"),
    ("CD19", "CD19"),
    ("MS4A1", "CD20"),
    ("CD22", "CD22"),
    ("CD27", "CD27"),
    ("CD33", "CD33"),
    ("CD34", "CD34"),
    ("CD36", "CD36"),
    ("CD38", "CD38"),
    ("CD40", "CD40"),
    ("CD47", "CD47"),
    ("FCGR1A", "CD64"),
    ("CD69", "CD69"),
    ("CD80", "CD80"),
    ("CD86", "CD86"),
    ("CD163", "CD163"),
    ("CX3CR1", "CX3CR1"),
)


def _designation(
    path: Path,
    amendment_path: Path = DEFAULT_AMENDMENT,
    metadata_path: Path = DEFAULT_METADATA,
    preflight_path: Path = DEFAULT_PREFLIGHT,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    value = _read_json(path)
    if (
        value.get("schema") != "gse309593-held-batches-candidate-designation/1.0"
        or value.get("status") != "FROZEN_METADATA_SPLIT_BEFORE_NUMERIC_ASSAY_ACCESS"
    ):
        raise PermissionError("candidate designation is not frozen pre-outcome")
    amendment = _read_json(amendment_path)
    if (
        amendment.get("schema") != "gse309593-held-batches-candidate-amendment/2.0"
        or amendment.get("status")
        != "FROZEN_WORDING_CORRECTION_BEFORE_NUMERIC_ASSAY_ACCESS"
        or amendment.get("supersedes", {}).get("candidate_sha256") != _sha256(path)
    ):
        raise PermissionError("candidate wording correction is absent or unbound")
    correction = amendment.get("correction", {})
    if any(
        correction.get(key) is not False
        for key in (
            "allocation_changed",
            "subjects_changed",
            "files_changed",
            "decision_rules_changed",
        )
    ):
        raise PermissionError("candidate amendment changes the frozen allocation")

    preflight = _read_json(preflight_path)
    if (
        preflight.get("schema") != "gse309593-held-batches-axis-preflight/1.0"
        or preflight.get("status") != "PASS_AXIS_ONLY_BEFORE_NUMERIC_ASSAY_ACCESS"
    ):
        raise PermissionError("axis preflight is not frozen before assay access")
    audit = preflight.get("access_audit", {})
    forbidden = {
        "rna_numeric_matrix_values_read": 0,
        "adt_numeric_matrix_values_read": 0,
        "barcodes_or_cell_identifiers_read": 0,
        "same_cell_tables_formed": 0,
        "association_statistics_computed": 0,
        "losses_computed": 0,
    }
    if any(audit.get(key) != expected for key, expected in forbidden.items()):
        raise PermissionError("axis preflight accessed a forbidden outcome")
    markers = preflight.get("ordered_cognate_panel")
    if not isinstance(markers, list) or len(markers) != MARKER_COUNT:
        raise ValueError("axis preflight must freeze all 24 exact cognates")
    if (
        tuple(
            (marker.get("rna_symbol"), marker.get("adt_target")) for marker in markers
        )
        != EXPECTED_PANEL
    ):
        raise ValueError("ordered cognate panel differs from the frozen candidate axis")

    metadata = _read_json(metadata_path)
    if _sha256(metadata_path) != value["frozen_file_binding"]["metadata_source_sha256"]:
        raise PermissionError("bound GSE309593 metadata bytes differ")
    metadata_subjects = {
        row["subject_id"]: row for row in metadata["recipient_cohort"]["subjects"]
    }
    records = preflight.get("records")
    if not isinstance(records, list) or len(records) != 23:
        raise ValueError("axis preflight must bind 23 subjects")
    samples: list[dict[str, Any]] = []
    for record in records:
        subject = record.get("subject_id")
        meta = metadata_subjects.get(subject)
        if meta is None:
            raise ValueError("preflight subject is absent from frozen metadata")
        if (
            record.get("gsm") != meta.get("gsm")
            or record.get("batch") != meta.get("batch")
            or int(record.get("rna_h5_bytes", -1)) != int(meta["rna_h5"]["bytes"])
            or int(record.get("adt_csv_gz_bytes", -1))
            != int(meta["adt_csv_gz"]["bytes"])
            or record.get("all_24_cognates_resolve_exactly_once") is not True
        ):
            raise ValueError("preflight record differs from bound subject metadata")
        samples.append(
            {
                **record,
                "rna_h5": dict(meta["rna_h5"]),
                "adt_csv_gz": dict(meta["adt_csv_gz"]),
            }
        )
    source = [sample for sample in samples if sample.get("role") == "source"]
    held = [sample for sample in samples if sample.get("role") == "held"]
    if (
        len(source) != 14
        or len(held) != 9
        or tuple(sorted({sample["batch"] for sample in source})) != SOURCE_BATCHES
        or tuple(sorted({sample["batch"] for sample in held})) != HELD_BATCHES
        or len({sample["subject_id"] for sample in samples}) != 23
        or {sample["subject_id"] for sample in source}
        & {sample["subject_id"] for sample in held}
    ):
        raise ValueError("source or held allocation differs from the frozen split")
    if tuple(sample["subject_id"] for sample in source) != tuple(
        value["allocation"]["source_subjects"]
    ) or tuple(sample["subject_id"] for sample in held) != tuple(
        value["allocation"]["held_subjects"]
    ):
        raise ValueError("subject order differs from the candidate designation")
    value = {
        **value,
        "strict_cognates": markers,
        "url_template": metadata["recipient_cohort"]["per_sample_url_template"],
        "rna_axis_schema": preflight["axis_schemas"]["rna"],
        "adt_axis_schemas": preflight["axis_schemas"]["adt_csv"],
    }
    return value, source, held


def _integer_values(values: np.ndarray, message: str) -> np.ndarray:
    try:
        numeric = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(message) from error
    if (
        not np.isfinite(numeric).all()
        or np.any(numeric < 0)
        or not np.allclose(numeric, np.rint(numeric), rtol=0.0, atol=1e-8)
    ):
        raise ValueError(message)
    return np.rint(numeric).astype(np.int64)


def _decode(values: np.ndarray) -> list[str]:
    return [
        value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


H5_DATASET_ALLOWLIST = frozenset(
    {
        "matrix/barcodes",
        "matrix/features/name",
        "matrix/features/feature_type",
        "matrix/data",
        "matrix/indices",
        "matrix/indptr",
        "matrix/shape",
    }
)


class _AllowlistedH5:
    def __init__(self, handle: h5py.File, on_access: Optional[Any] = None):
        self.handle = handle
        self.accessed: list[str] = []
        self.on_access = on_access

    def _record_access(self, dataset: str) -> None:
        self.accessed.append(dataset)
        if self.on_access is not None:
            self.on_access(dataset)

    def read(self, dataset: str, key: Any = ()) -> np.ndarray:
        if dataset not in H5_DATASET_ALLOWLIST:
            raise PermissionError(f"H5 dataset is outside the allowlist: {dataset}")
        if dataset not in self.handle or not isinstance(
            self.handle[dataset], h5py.Dataset
        ):
            raise ValueError(f"allowlisted H5 dataset is absent: {dataset}")
        self._record_access(dataset)
        return np.asarray(self.handle[dataset][key])

    def length(self, dataset: str) -> int:
        if dataset not in H5_DATASET_ALLOWLIST:
            raise PermissionError(f"H5 dataset is outside the allowlist: {dataset}")
        if dataset not in self.handle or not isinstance(
            self.handle[dataset], h5py.Dataset
        ):
            raise ValueError(f"allowlisted H5 dataset is absent: {dataset}")
        self._record_access(dataset)
        return len(self.handle[dataset])


def _selected_cells(
    barcodes: list[str], subject: str, eligible: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    eligible = np.asarray(eligible, dtype=np.int64)
    if (
        len(barcodes) < CELL_BUDGET
        or len(set(barcodes)) != len(barcodes)
        or len(eligible) < CELL_BUDGET
        or len(set(eligible.tolist())) != len(eligible)
        or np.any(eligible < 0)
        or np.any(eligible >= len(barcodes))
    ):
        raise ValueError("RNA QC cell axis is too short, duplicated, or invalid")
    ranked = sorted(
        eligible.tolist(),
        key=lambda index: (
            hashlib.sha256(
                f"{CELL_SALT}|{subject}|{barcodes[index]}".encode()
            ).hexdigest(),
            barcodes[index],
        ),
    )[:CELL_BUDGET]
    indices = np.asarray(sorted(ranked), dtype=np.int64)
    return indices, [barcodes[index] for index in indices]


def _feature_indices(
    names: list[str], types: list[str], markers: list[dict[str, Any]]
) -> list[int]:
    output: list[int] = []
    for marker in markers:
        matches = [
            index
            for index, (name, kind) in enumerate(zip(names, types))
            if name == marker["rna_symbol"] and kind == "Gene Expression"
        ]
        if len(matches) != 1:
            raise ValueError("a frozen RNA feature does not resolve exactly once")
        output.append(matches[0])
    return output


def _read_rna_h5(
    path: Path,
    sample: dict[str, Any],
    markers: list[dict[str, Any]],
    rna_schema: dict[str, Any],
    access_audit: Optional[dict[str, Any]] = None,
    cohort: Optional[str] = None,
) -> dict[str, Any]:
    if access_audit is not None and cohort not in {"source", "held"}:
        raise ValueError("H5 access-audit cohort is required")

    def record_access(dataset: str) -> None:
        if access_audit is None:
            return
        key = "unique_h5_dataset_paths_with_started_read_or_length"
        observed = access_audit[key][cohort]
        access_audit[key][cohort] = sorted(set(observed) | {dataset})

    with h5py.File(path, "r") as handle:
        reader = _AllowlistedH5(handle, record_access)
        barcodes = _decode(reader.read("matrix/barcodes"))
        names = _decode(reader.read("matrix/features/name"))
        types = _decode(reader.read("matrix/features/feature_type"))
        if (
            len(names) != int(rna_schema["feature_count"])
            or _newline_axis_sha256(names) != rna_schema["feature_name_axis_sha256"]
            or _newline_axis_sha256(types) != rna_schema["feature_type_axis_sha256"]
        ):
            raise PermissionError("RNA feature axis differs from the axis preflight")
        shape = _integer_values(
            reader.read("matrix/shape"), "H5 sparse shape is invalid"
        )
        if shape.shape != (2,) or tuple(shape.tolist()) != (
            len(names),
            len(barcodes),
        ):
            raise ValueError("H5 matrix axes differ from its sparse shape")
        if len(types) != len(names) or any(kind != "Gene Expression" for kind in types):
            raise ValueError("matrix/ contains a non-RNA feature axis")
        feature_indices = _feature_indices(names, types, markers)
        indptr = _integer_values(reader.read("matrix/indptr"), "H5 indptr is invalid")
        if (
            len(indptr) != len(barcodes) + 1
            or indptr[0] != 0
            or np.any(indptr[1:] < indptr[:-1])
            or int(indptr[-1]) != reader.length("matrix/data")
            or int(indptr[-1]) != reader.length("matrix/indices")
        ):
            raise ValueError("H5 sparse structure is invalid")
        mitochondrial = {
            index for index, name in enumerate(names) if name.startswith("MT-")
        }
        detected = np.zeros(len(barcodes), dtype=np.int64)
        totals = np.zeros(len(barcodes), dtype=np.int64)
        mitochondrial_totals = np.zeros(len(barcodes), dtype=np.int64)
        decoded = 0
        for cell in range(len(barcodes)):
            start, stop = int(indptr[cell]), int(indptr[cell + 1])
            features = _integer_values(
                reader.read("matrix/indices", slice(start, stop)),
                "H5 indices are invalid",
            )
            values = _integer_values(
                reader.read("matrix/data", slice(start, stop)),
                "RNA matrix is not raw counts",
            )
            if np.any(features >= len(names)) or len(features) != len(
                set(features.tolist())
            ):
                raise ValueError("H5 sparse feature indices are invalid")
            detected[cell] = int(np.count_nonzero(values > 0))
            totals[cell] = int(values.sum())
            mitochondrial_totals[cell] = int(
                sum(
                    int(value)
                    for feature, value in zip(features, values)
                    if int(feature) in mitochondrial
                )
            )
            decoded += len(values)
        mitochondrial_fraction = np.divide(
            mitochondrial_totals,
            totals,
            out=np.ones_like(mitochondrial_totals, dtype=float),
            where=totals > 0,
        )
        eligible = np.flatnonzero(
            (detected >= MINIMUM_DETECTED_GENES)
            & (mitochondrial_fraction <= MAXIMUM_MITOCHONDRIAL_FRACTION)
            & (totals <= MAXIMUM_RNA_UMIS)
        )
        selected_indices, selected_barcodes = _selected_cells(
            barcodes, sample["subject_id"], eligible
        )
        lookup = {feature: index for index, feature in enumerate(feature_indices)}
        rna = np.zeros((CELL_BUDGET, MARKER_COUNT), dtype=np.int64)
        selected_decoded = 0
        for output_row, cell in enumerate(selected_indices):
            start, stop = int(indptr[cell]), int(indptr[cell + 1])
            features = _integer_values(
                reader.read("matrix/indices", slice(start, stop)),
                "H5 indices are invalid",
            )
            values = _integer_values(
                reader.read("matrix/data", slice(start, stop)),
                "RNA matrix is not raw counts",
            )
            for feature, value in zip(features, values):
                column = lookup.get(int(feature))
                if column is not None:
                    rna[output_row, column] = int(value)
            selected_decoded += len(values)
        accessed = sorted(set(reader.accessed))
    if accessed != sorted(H5_DATASET_ALLOWLIST):
        raise PermissionError("H5 reader dataset certificate differs from the allowlist")
    return {
        "rna": rna,
        "barcodes": selected_barcodes,
        "barcode_axis_sha256": _axis_sha256(barcodes),
        "selected_cell_axis_sha256": _axis_sha256(selected_barcodes),
        "accessed_h5_datasets": accessed,
        "qc_eligible_cells": int(len(eligible)),
        "rna_qc_numeric_values_decoded": int(decoded),
        "selected_sparse_values_decoded": int(selected_decoded),
    }


def _new_access_audit(phase: str) -> dict[str, Any]:
    counters = {}
    for cohort in ("source", "held"):
        for role in ("rna_h5", "adt_csv_gz"):
            for action in (
                "requests",
                "hashes_completed",
                "files_deleted",
                "decode_started",
                "reductions_completed",
            ):
                counters[f"{cohort}_{role}_{action}"] = 0
        for action in (
            "identifier_files_accessed",
            "numeric_assay_files_accessed",
            "state_panels_formed",
            "joint_table_panels_formed",
        ):
            counters[f"{cohort}_{action}"] = 0
    counters["h5_datasets_outside_allowlist_read"] = 0
    counters["embedded_h5_adt_datasets_read"] = 0
    return {
        "schema": "gse309593-held-batches-incremental-access-audit/1.0",
        "phase": phase,
        "events": [],
        "counters": counters,
        "unique_h5_dataset_paths_with_started_read_or_length": {
            "source": [],
            "held": [],
        },
        "contains_raw_identifiers_or_outcome_values": False,
        "contains_urls_or_local_paths": False,
    }


def _audit_increment(audit: dict[str, Any], field: str, value: int = 1) -> None:
    audit["counters"][field] += int(value)


def _audit_event(
    audit: dict[str, Any],
    event: str,
    sample: dict[str, Any],
    *,
    file_role: Optional[str] = None,
    sha256: Optional[str] = None,
    preflight_sha256_verified: Optional[bool] = None,
    h5_datasets: Optional[list[str]] = None,
) -> None:
    record: dict[str, Any] = {
        "sequence": len(audit["events"]),
        "event": event,
        "subject_id": sample["subject_id"],
        "gsm": sample["gsm"],
    }
    if file_role is not None:
        record.update(
            {
                "file_role": file_role,
                "designated_file_name": sample[file_role]["name"],
                "designated_file_bytes": int(sample[file_role]["bytes"]),
            }
        )
    if sha256 is not None:
        record["computed_sha256"] = sha256
    if preflight_sha256_verified is not None:
        record["preflight_sha256_verified"] = preflight_sha256_verified
    if h5_datasets is not None:
        unique = sorted(set(h5_datasets))
        record["completed_unique_h5_dataset_paths_decoded"] = unique
        cohort = "source" if sample["batch"] in SOURCE_BATCHES else "held"
        audit["unique_h5_dataset_paths_with_started_read_or_length"][cohort] = sorted(
            set(
                audit["unique_h5_dataset_paths_with_started_read_or_length"][cohort]
            )
            | set(unique)
        )
    audit["events"].append(record)


def _audit_file_deleted(
    audit: dict[str, Any], cohort: str, sample: dict[str, Any], file_role: str
) -> None:
    _audit_increment(audit, f"{cohort}_{file_role}_files_deleted")
    _audit_event(audit, "temporary_file_deleted", sample, file_role=file_role)


def _fetch_designated_file(
    url: str,
    expected_bytes: int,
    scratch: Path,
    suffix: str,
    access_audit: Optional[dict[str, Any]] = None,
    cohort: Optional[str] = None,
    sample: Optional[dict[str, Any]] = None,
    file_role: Optional[str] = None,
) -> tuple[Path, str]:
    scratch.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(
        dir=scratch, suffix=suffix, delete=False
    ) as stream:
        path = Path(stream.name)
        try:
            with urlopen(url) as response:
                total = 0
                while True:
                    block = response.read(8 << 20)
                    if not block:
                        break
                    total += len(block)
                    if total > expected_bytes:
                        raise PermissionError("designated file exceeds its byte count")
                    digest.update(block)
                    stream.write(block)
        except BaseException:
            path.unlink(missing_ok=True)
            if access_audit is not None:
                _audit_file_deleted(access_audit, cohort, sample, file_role)
            raise
    observed = digest.hexdigest()
    if access_audit is not None:
        _audit_increment(access_audit, f"{cohort}_{file_role}_hashes_completed")
        _audit_event(
            access_audit,
            "file_hash_completed",
            sample,
            file_role=file_role,
            sha256=observed,
        )
    if path.stat().st_size != expected_bytes:
        path.unlink(missing_ok=True)
        if access_audit is not None:
            _audit_file_deleted(access_audit, cohort, sample, file_role)
        raise PermissionError("designated file byte count differs")
    return path, observed


def _designated_url(
    designation: dict[str, Any], sample: dict[str, Any], key: str
) -> str:
    record = sample[key]
    return designation["url_template"].format(
        gsm=sample["gsm"], file=quote(record["name"], safe="")
    )


def _fetch_sample(
    designation: dict[str, Any],
    sample: dict[str, Any],
    key: str,
    scratch: Path,
    access_audit: Optional[dict[str, Any]] = None,
    cohort: Optional[str] = None,
) -> tuple[Path, str]:
    if access_audit is not None:
        if cohort not in {"source", "held"}:
            raise ValueError("access-audit cohort is required")
        _audit_increment(access_audit, f"{cohort}_{key}_requests")
        _audit_event(access_audit, "file_requested", sample, file_role=key)
    suffix = ".h5" if key == "rna_h5" else ".csv.gz"
    path, observed = _fetch_designated_file(
        _designated_url(designation, sample, key),
        int(sample[key]["bytes"]),
        scratch,
        suffix,
        access_audit,
        cohort,
        sample,
        key,
    )
    frozen = sample.get(f"{key}_sha256")
    if frozen is not None and observed != frozen:
        path.unlink(missing_ok=True)
        if access_audit is not None:
            _audit_file_deleted(access_audit, cohort, sample, key)
        raise PermissionError("designated file SHA-256 differs from preflight")
    if access_audit is not None:
        _audit_event(
            access_audit,
            "file_designation_verified",
            sample,
            file_role=key,
            sha256=observed,
            preflight_sha256_verified=frozen is not None,
        )
    return path, observed


def _read_adt_csv(
    path: Path,
    selected_barcodes: list[str],
    markers: list[dict[str, Any]],
    expected_schema: dict[str, Any],
) -> np.ndarray:
    labels = [marker["adt_target"] for marker in markers]
    selected = set(selected_barcodes)
    with gzip.open(path, "rt", newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("ADT CSV is empty") from error
        if len(header) < 2 or len(header) != len(set(header)):
            raise ValueError("ADT CSV header is invalid or duplicated")
        marker_headers = set(header[1:]) & set(labels)
        selected_headers = set(header[1:]) & selected
        if marker_headers and selected_headers:
            raise ValueError("ADT CSV orientation is ambiguous")
        if marker_headers:
            deposited_axis = header[1:]
            if (
                len(deposited_axis) != int(expected_schema["feature_count"])
                or _newline_axis_sha256(deposited_axis)
                != expected_schema["axis_sha256"]
            ):
                raise PermissionError("ADT header axis differs from the axis preflight")
            if marker_headers != set(labels):
                raise ValueError("ADT CSV lacks the complete frozen marker panel")
            columns = [header.index(label) for label in labels]
            rows: dict[str, np.ndarray] = {}
            all_identifiers: set[str] = set()
            for record in reader:
                if len(record) != len(header):
                    raise ValueError("ADT CSV row length differs")
                identifier = record[0]
                if not identifier or identifier in all_identifiers:
                    raise ValueError("ADT identifier axis is empty or duplicated")
                all_identifiers.add(identifier)
                if identifier in selected:
                    rows[identifier] = _integer_values(
                        np.asarray([record[index] for index in columns]),
                        "ADT matrix is not nonnegative integer counts",
                    )
            if set(rows) != selected:
                raise ValueError("a selected RNA barcode is missing from ADT")
            counts = np.stack([rows[barcode] for barcode in selected_barcodes])
        else:
            if selected_headers != selected:
                raise ValueError("ADT CSV cannot resolve all selected barcodes")
            columns = [header.index(barcode) for barcode in selected_barcodes]
            rows: dict[str, np.ndarray] = {}
            deposited_axis: list[str] = []
            for record in reader:
                if len(record) != len(header):
                    raise ValueError("ADT CSV row length differs")
                marker = record[0]
                deposited_axis.append(marker)
                if marker not in labels:
                    continue
                if marker in rows:
                    raise ValueError("ADT marker row is duplicated")
                rows[marker] = _integer_values(
                    np.asarray([record[index] for index in columns]),
                    "ADT matrix is not nonnegative integer counts",
                )
            if set(rows) != set(labels):
                raise ValueError("ADT CSV lacks the complete frozen marker panel")
            if (
                len(deposited_axis) != int(expected_schema["feature_count"])
                or _newline_axis_sha256(deposited_axis)
                != expected_schema["axis_sha256"]
            ):
                raise PermissionError(
                    "ADT feature axis differs from the axis preflight"
                )
            counts = np.stack([rows[label] for label in labels], axis=1)
    if counts.shape != (CELL_BUDGET, MARKER_COUNT):
        raise ValueError("ADT panel has the wrong shape")
    return counts


def _adt_states(counts: np.ndarray, barcodes: list[str], subject: str) -> np.ndarray:
    values = np.asarray(counts)
    if values.shape != (CELL_BUDGET, MARKER_COUNT):
        raise ValueError("ADT count panel has the wrong shape")
    states = np.zeros(values.shape, dtype=np.uint8)
    for marker in range(MARKER_COUNT):
        order = sorted(
            range(CELL_BUDGET),
            key=lambda index: (
                int(values[index, marker]),
                hashlib.sha256(
                    f"{ADT_TIE_SALT}|{subject}|{marker}|{barcodes[index]}".encode()
                ).hexdigest(),
                barcodes[index],
            ),
        )
        states[np.asarray(order[CELL_BUDGET // 2 :], dtype=int), marker] = 1
    if not np.all(states.sum(axis=0) == CELL_BUDGET // 2):
        raise AssertionError("deterministic ADT midrank split changed its margin")
    return states


def _destroyed_adt(states: np.ndarray, barcodes: list[str], subject: str) -> np.ndarray:
    order = sorted(
        range(CELL_BUDGET),
        key=lambda index: (
            hashlib.sha256(
                f"{DESTROYED_SALT}|{subject}|{barcodes[index]}".encode()
            ).hexdigest(),
            barcodes[index],
        ),
    )
    order_array = np.asarray(order, dtype=int)
    destroyed = np.empty_like(states)
    destroyed[order_array] = np.asarray(states)[np.roll(order_array, 1)]
    if not np.array_equal(destroyed.sum(axis=0), np.asarray(states).sum(axis=0)):
        raise AssertionError("destroyed-link control changed an ADT margin")
    if sorted(map(tuple, destroyed.tolist())) != sorted(
        map(tuple, np.asarray(states).tolist())
    ):
        raise AssertionError("destroyed-link control changed multivariate ADT states")
    return destroyed


def _tables(rna: np.ndarray, adt: np.ndarray) -> np.ndarray:
    first = np.asarray(rna)
    second = np.asarray(adt)
    if first.shape != (CELL_BUDGET, MARKER_COUNT) or second.shape != first.shape:
        raise ValueError("binary panels have the wrong shape")
    output = np.zeros((MARKER_COUNT, MARKER_COUNT, 2, 2), dtype=np.int64)
    for row, column in np.ndindex((MARKER_COUNT, MARKER_COUNT)):
        code = 2 * first[:, row] + second[:, column]
        output[row, column] = np.bincount(code, minlength=4).reshape(2, 2)
    return output


def _reduce_source(
    panels: dict[str, dict[str, Any]],
    samples: list[dict[str, Any]],
    access_audit: Optional[dict[str, Any]] = None,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for sample in samples:
        subject = sample["subject_id"]
        panel = panels[subject]
        rna = (panel["rna"] > 0).astype(np.uint8)
        adt_marker_support = _adt_variation_support(panel["adt"])
        adt = _adt_states(panel["adt"], panel["barcodes"], subject)
        if access_audit is not None:
            _audit_increment(access_audit, "source_state_panels_formed")
            _audit_event(access_audit, "source_adt_state_panel_formed", sample)
        observed = _tables(rna, adt)
        subject_support = _subject_support(observed, adt_marker_support)
        destroyed = _tables(rna, _destroyed_adt(adt, panel["barcodes"], subject))
        records[subject] = {
            "tables": observed,
            "destroyed_tables": destroyed,
            "rna_profile": rna.mean(axis=0),
            "adt_profile": np.log1p(panel["adt"]).mean(axis=0),
            "table_sha256": _array_sha256(observed),
            "destroyed_table_sha256": _array_sha256(destroyed),
            "barcode_axis_sha256": panel["barcode_axis_sha256"],
            "selected_cell_axis_sha256": panel["selected_cell_axis_sha256"],
            "adt_marker_support": adt_marker_support,
            "subject_support": subject_support,
            "pooled_support": _pooled_support(adt_marker_support),
            "informative_pair_count": int(np.count_nonzero(subject_support)),
        }
        if access_audit is not None:
            _audit_increment(access_audit, "source_joint_table_panels_formed", 2)
            _audit_event(access_audit, "source_joint_table_panels_formed", sample)
    return records


def _informative(tables: np.ndarray) -> np.ndarray:
    values = np.asarray(tables)
    rows = values.sum(axis=-1)
    columns = values.sum(axis=-2)
    total = values.sum(axis=(-2, -1))
    lower = np.maximum(0, rows[..., 0] + columns[..., 0] - total)
    upper = np.minimum(rows[..., 0], columns[..., 0])
    return upper > lower


def _adt_variation_support(counts: np.ndarray) -> np.ndarray:
    values = np.asarray(counts)
    if values.shape != (CELL_BUDGET, MARKER_COUNT):
        raise ValueError("ADT count panel has the wrong shape")
    return np.asarray(
        [len(np.unique(values[:, marker])) >= 2 for marker in range(MARKER_COUNT)],
        dtype=bool,
    )


def _subject_support(tables: np.ndarray, adt_marker_support: np.ndarray) -> np.ndarray:
    marker_support = np.asarray(adt_marker_support, dtype=bool)
    if marker_support.shape != (MARKER_COUNT,):
        raise ValueError("ADT marker support has the wrong shape")
    return _informative(tables) & marker_support[None, :]


def _masked_tables(tables: np.ndarray, support: np.ndarray) -> np.ndarray:
    values = np.asarray(tables, dtype=np.int64)
    observed = np.asarray(support, dtype=bool)
    if observed.shape != values.shape[:-2]:
        raise ValueError("table support mask has the wrong shape")
    return np.where(observed[..., None, None], values, 0)


def _conditional_missing_tables(
    tables: np.ndarray, support: np.ndarray
) -> np.ndarray:
    values = np.asarray(tables, dtype=np.int64)
    observed = np.asarray(support, dtype=bool)
    if observed.shape != values.shape[:-2]:
        raise ValueError("conditional support mask has the wrong shape")
    totals = values.sum(axis=(-2, -1))
    if np.any(totals <= 0):
        raise ValueError("conditional tables must have positive totals")
    encoded = np.zeros_like(values)
    encoded[..., 0, 0] = totals
    return np.where(observed[..., None, None], values, encoded)


def _pooled_support(adt_marker_support: np.ndarray) -> np.ndarray:
    marker_support = np.asarray(adt_marker_support, dtype=bool)
    if marker_support.shape != (MARKER_COUNT,):
        raise ValueError("ADT marker support has the wrong shape")
    return np.broadcast_to(marker_support[None, :], (MARKER_COUNT, MARKER_COUNT)).copy()


def _source_comparison_mask(
    records: dict[str, dict[str, Any]], samples: list[dict[str, Any]]
) -> tuple[np.ndarray, dict[str, Any]]:
    axes = [
        (
            f"lobo_without_{batch}",
            [sample for sample in samples if sample["batch"] != batch],
        )
        for batch in SOURCE_BATCHES
    ]
    axes.append(("final_all_source", list(samples)))
    intersection = np.ones((MARKER_COUNT, MARKER_COUNT), dtype=bool)
    diagnostics: dict[str, Any] = {}
    for name, training in axes:
        subjects = [sample["subject_id"] for sample in training]
        tables = np.asarray([records[value]["tables"] for value in subjects])
        support = np.asarray([records[value]["subject_support"] for value in subjects])
        pooled_support = np.asarray(
            [records[value]["pooled_support"] for value in subjects]
        )
        conditional = _masked_tables(tables, support)
        rows = conditional.sum(axis=-1)
        columns = conditional.sum(axis=-2)
        total = conditional.sum(axis=(-2, -1))
        lower = np.maximum(0, rows[..., 0] + columns[..., 0] - total)
        upper = np.minimum(rows[..., 0], columns[..., 0])
        observed_sum = conditional[..., 0, 0].sum(axis=0)
        lower_sum = lower.sum(axis=0)
        upper_sum = upper.sum(axis=0)
        pooled = _masked_tables(tables, pooled_support).sum(axis=0)
        eligible = (
            (support.sum(axis=0) >= 2)
            & (observed_sum > lower_sum)
            & (observed_sum < upper_sum)
            & np.all(pooled > 0, axis=(-2, -1))
        )
        intersection &= eligible
        diagnostics[name] = {
            "training_subjects": subjects,
            "eligible_coordinate_count": int(np.count_nonzero(eligible)),
            "eligible_mask_sha256": _array_sha256(eligible.astype(np.uint8)),
        }
    count = int(np.count_nonzero(intersection))
    per_subject = {
        sample["subject_id"]: int(
            np.count_nonzero(
                intersection & records[sample["subject_id"]]["subject_support"]
            )
        )
        for sample in samples
    }
    details = {
        "coordinate_count": count,
        "mask_sha256": _array_sha256(intersection.astype(np.uint8)),
        "hash_encoding": "dtype-string, int64 shape, then C-order uint8 mask bytes",
        "folds": diagnostics,
        "source_subject_supported_coordinate_counts": per_subject,
    }
    checks = {
        "at_least_288_coordinates_retained": count >= 288,
        "every_source_subject_has_at_least_64_supported_coordinates": all(
            value >= MINIMUM_INFORMATIVE_ENTITIES for value in per_subject.values()
        ),
    }
    details["checks"] = checks
    if not all(checks.values()):
        raise SelectionRefusal("source-only common comparison mask failed", details)
    return intersection, details


def _knn_incidence(profiles: np.ndarray, neighbors: int) -> np.ndarray:
    values = np.asarray(profiles, dtype=float).T
    scale = values.std(axis=1, ddof=1)
    if np.any(scale == 0.0) or not np.isfinite(scale).all():
        raise CouplingEstimationRefusal("source marker profile has zero variance")
    values = (values - values.mean(axis=1, keepdims=True)) / scale[:, None]
    edges: set[tuple[int, int]] = set()
    for marker in range(MARKER_COUNT):
        candidates = np.asarray(
            [value for value in range(MARKER_COUNT) if value != marker]
        )
        distances = np.asarray(
            [np.linalg.norm(values[marker] - values[value]) for value in candidates]
        )
        order = candidates[np.lexsort((candidates, distances))]
        edges.update(tuple(sorted((marker, int(value)))) for value in order[:neighbors])
    incidence = np.zeros((MARKER_COUNT, len(edges)), dtype=float)
    for edge, (left, right) in enumerate(sorted(edges)):
        incidence[left, edge] = 1.0
        incidence[right, edge] = 1.0
    return incidence


def _fit_primary(
    tables: np.ndarray,
    rna_profiles: np.ndarray,
    adt_profiles: np.ndarray,
    config: PrimaryConfig,
    subject_support: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    if config.graph_penalty == 0.0:
        first = np.eye(MARKER_COUNT, dtype=float)
        second = np.eye(MARKER_COUNT, dtype=float)
    else:
        first = _knn_incidence(rna_profiles, config.graph_neighbors)
        second = _knn_incidence(adt_profiles, config.graph_neighbors)
    fit = fit_hierarchical_conditional_log_odds(
        _conditional_missing_tables(
            np.asarray(tables, dtype=np.int64),
            _informative(tables) if subject_support is None else subject_support,
        ),
        first,
        second,
        heterogeneity_penalty=config.heterogeneity_penalty,
        ridge_penalty=config.ridge_penalty,
        graph_penalty=config.graph_penalty,
        minimum_informative_donors=1,
        maximum_condition_number=MAXIMUM_CONDITION_NUMBER,
    )
    return {
        "population_log_odds": fit.population_log_odds,
        "fit_certificate": {
            "gradient_norm": fit.gradient_norm,
            "scaled_gradient_norm": fit.scaled_gradient_norm,
            "schur_condition_number": fit.schur_condition_number,
            "theta_curvature_condition_number": fit.theta_curvature_condition_number,
            "iterations": fit.iterations,
            "rna_incidence_sha256": _array_sha256(first),
            "adt_incidence_sha256": _array_sha256(second),
        },
    }


def _fit_common_effect(
    tables: np.ndarray,
    comparison_mask: Optional[np.ndarray] = None,
    subject_support: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    mask = (
        np.ones((MARKER_COUNT, MARKER_COUNT), dtype=bool)
        if comparison_mask is None
        else np.asarray(comparison_mask, dtype=bool)
    )
    support = _informative(tables) if subject_support is None else subject_support
    selected_tables = _conditional_missing_tables(tables, support)[:, mask]
    fit = fit_common_effect_conditional_log_odds(
        selected_tables,
        minimum_informative_donors=2,
        tolerance=1e-10,
    )
    population_log_odds = np.zeros((MARKER_COUNT, MARKER_COUNT), dtype=float)
    population_log_odds[mask] = fit.log_odds
    return {
        "population_log_odds": population_log_odds,
        "fit_certificate": {
            "gradient_norm": fit.gradient_norm,
            "scaled_gradient_norm": fit.scaled_gradient_norm,
            "minimum_data_precision": float(np.min(fit.data_precision)),
            "maximum_data_precision": float(np.max(fit.data_precision)),
            "minimum_support_count": int(np.min(fit.support_count)),
            "maximum_root_iterations": int(np.max(fit.root_iterations)),
        },
    }


def _fit_pooled_poisson(
    tables: np.ndarray,
    comparison_mask: Optional[np.ndarray] = None,
    pooled_support: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    mask = (
        np.ones((MARKER_COUNT, MARKER_COUNT), dtype=bool)
        if comparison_mask is None
        else np.asarray(comparison_mask, dtype=bool)
    )
    support = (
        np.ones(np.asarray(tables).shape[:-2], dtype=bool)
        if pooled_support is None
        else np.asarray(pooled_support, dtype=bool)
    )
    pooled_selected = _masked_tables(tables, support).sum(axis=0)[mask]
    if np.any(pooled_selected <= 0):
        raise CouplingEstimationRefusal(
            "pooled saturated Poisson interaction has a zero cell"
        )
    selected_odds = np.log(pooled_selected[..., 0, 0]) + np.log(
        pooled_selected[..., 1, 1]
    )
    selected_odds -= np.log(pooled_selected[..., 0, 1]) + np.log(
        pooled_selected[..., 1, 0]
    )
    if not np.isfinite(selected_odds).all():
        raise CouplingEstimationRefusal("pooled Poisson interaction is nonfinite")
    maximum_cell_error = 0.0
    maximum_row_error = 0.0
    maximum_column_error = 0.0
    for table, log_odds in zip(pooled_selected, selected_odds):
        rows = table.sum(axis=1)
        columns = table.sum(axis=0)
        reconstructed = binary_table_from_helmert_coordinate(
            0.5 * float(log_odds), rows, columns
        )
        scale = max(1.0, float(table.sum()))
        maximum_cell_error = max(
            maximum_cell_error,
            float(np.max(np.abs(reconstructed - table))) / scale,
        )
        maximum_row_error = max(
            maximum_row_error,
            float(np.max(np.abs(reconstructed.sum(axis=1) - rows))) / scale,
        )
        maximum_column_error = max(
            maximum_column_error,
            float(np.max(np.abs(reconstructed.sum(axis=0) - columns))) / scale,
        )
    maximum_error = max(
        maximum_cell_error, maximum_row_error, maximum_column_error
    )
    if maximum_error > 1e-8:
        raise CouplingEstimationRefusal(
            "pooled Poisson saturated-table reconstruction failed"
        )
    odds = np.zeros((MARKER_COUNT, MARKER_COUNT), dtype=float)
    odds[mask] = selected_odds
    support_count = np.asarray(support, dtype=bool).sum(axis=0)[mask]
    support_array = np.asarray(support, dtype=bool)
    if not np.array_equal(
        support_array,
        np.broadcast_to(support_array[:, :1, :], support_array.shape),
    ):
        raise ValueError("pooled Poisson support must be ADT-marker-specific")
    marker_subject_counts = support_array[:, 0, :].sum(axis=0).astype(np.int64)
    return {
        "population_log_odds": odds,
        "pooled_tables_sha256": _array_sha256(pooled_selected),
        "fit_certificate": {
            "coordinate_count": int(np.count_nonzero(mask)),
            "minimum_pooled_subject_count": int(np.min(support_count)),
            "maximum_pooled_subject_count": int(np.max(support_count)),
            "ordered_adt_marker_pooled_subject_counts": marker_subject_counts.tolist(),
            "ordered_adt_marker_pooled_subject_counts_sha256": _array_sha256(
                marker_subject_counts
            ),
            "maximum_normalized_cell_error": maximum_cell_error,
            "maximum_normalized_row_margin_error": maximum_row_error,
            "maximum_normalized_column_margin_error": maximum_column_error,
            "maximum_normalized_reconstruction_error": maximum_error,
            "threshold": 1e-8,
            "passes": bool(maximum_error <= 1e-8),
        },
    }


def _residual_pool(
    tables: np.ndarray,
    family: str,
    comparison_mask: Optional[np.ndarray] = None,
    subject_support: Optional[np.ndarray] = None,
) -> np.ndarray:
    values = np.asarray(tables).reshape(len(tables), -1, 2, 2)
    mask = (
        np.ones((MARKER_COUNT, MARKER_COUNT), dtype=bool)
        if comparison_mask is None
        else np.asarray(comparison_mask, dtype=bool)
    ).ravel()
    support = (
        _informative(values)
        if subject_support is None
        else np.asarray(subject_support, dtype=bool).reshape(len(tables), -1)
    )
    support &= mask[None, :]
    if np.any(support[:, mask].sum(axis=0) < 2):
        raise CouplingEstimationRefusal("classical residual lacks source support")
    coordinates = np.full(support.shape, np.nan)
    statistic = (
        signed_pearson_coordinate
        if family == "pearson"
        else signed_deviance_coordinate
        if family == "root_deviance"
        else None
    )
    if statistic is None:
        raise ValueError("unknown residual family")
    for donor, entity in np.argwhere(support):
        coordinate = statistic(values[donor, entity])
        coordinates[donor, entity] = coordinate / math.sqrt(CELL_BUDGET)
    pooled_selected = np.nanmean(coordinates[:, mask], axis=0)
    if not np.isfinite(pooled_selected).all():
        raise CouplingEstimationRefusal("classical residual is nonfinite")
    pooled = np.zeros(MARKER_COUNT * MARKER_COUNT, dtype=float)
    pooled[mask] = pooled_selected
    return pooled.reshape(MARKER_COUNT, MARKER_COUNT)


def _fractional_pearson(table: np.ndarray) -> float:
    values = np.asarray(table, dtype=float)
    total = values.sum()
    rows = values.sum(axis=1)
    columns = values.sum(axis=0)
    determinant = values[0, 0] * values[1, 1] - values[0, 1] * values[1, 0]
    return float(determinant * math.sqrt(total / np.prod(rows) / np.prod(columns)))


def _fractional_deviance(table: np.ndarray) -> float:
    values = np.asarray(table, dtype=float)
    expected = np.outer(values.sum(axis=1), values.sum(axis=0)) / values.sum()
    positive = values > 0
    deviance = 2.0 * np.sum(
        values[positive] * np.log(values[positive] / expected[positive])
    )
    determinant = values[0, 0] * values[1, 1] - values[0, 1] * values[1, 0]
    return math.copysign(math.sqrt(max(float(deviance), 0.0)), float(determinant))


def _residual_table(
    coordinate: float,
    rows: np.ndarray,
    columns: np.ndarray,
    family: str,
) -> np.ndarray:
    row = np.asarray(rows, dtype=float)
    column = np.asarray(columns, dtype=float)
    total = row.sum()
    lower = max(0.0, row[0] + column[0] - total)
    upper = min(row[0], column[0])
    if upper <= lower:
        return np.asarray(
            [[lower, row[0] - lower], [column[0] - lower, row[1] - column[0] + lower]]
        )
    epsilon = min(1e-10, 0.25 * (upper - lower))
    left, right = lower + epsilon, upper - epsilon

    def table_at(value: float) -> np.ndarray:
        return np.asarray(
            [[value, row[0] - value], [column[0] - value, row[1] - column[0] + value]]
        )

    statistic = (
        _fractional_pearson
        if family == "pearson"
        else _fractional_deviance
        if family == "root_deviance"
        else None
    )
    if statistic is None:
        raise ValueError("unknown residual family")
    target = float(coordinate)
    target = min(max(target, statistic(table_at(left))), statistic(table_at(right)))
    for _ in range(80):
        midpoint = 0.5 * (left + right)
        if statistic(table_at(midpoint)) < target:
            left = midpoint
        else:
            right = midpoint
    return table_at(0.5 * (left + right))


def _margins(tables: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(tables)
    return values.sum(axis=-1), values.sum(axis=-2)


def _predict_odds(
    log_odds: np.ndarray, rows: np.ndarray, columns: np.ndarray, alpha: float
) -> np.ndarray:
    output = np.empty((MARKER_COUNT, MARKER_COUNT, 2, 2), dtype=float)
    for index in np.ndindex((MARKER_COUNT, MARKER_COUNT)):
        output[index] = expected_binary_table_from_log_odds(
            float(alpha) * float(log_odds[index]), rows[index], columns[index]
        )
    return output


def _predict_residual(
    pooled: np.ndarray, rows: np.ndarray, columns: np.ndarray, config: ResidualConfig
) -> np.ndarray:
    output = np.empty((MARKER_COUNT, MARKER_COUNT, 2, 2), dtype=float)
    for index in np.ndindex((MARKER_COUNT, MARKER_COUNT)):
        coordinate = (
            config.transport_multiplier * pooled[index] * math.sqrt(CELL_BUDGET)
        )
        output[index] = _residual_table(
            coordinate, rows[index], columns[index], config.family
        )
    return output


def _independence(rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    return _predict_odds(np.zeros((MARKER_COUNT, MARKER_COUNT)), rows, columns, 1.0)


def _loss(
    observed: np.ndarray,
    predicted: np.ndarray,
    evaluation_mask: Optional[np.ndarray] = None,
) -> float:
    truth = np.asarray(observed, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    support = _informative(truth)
    if evaluation_mask is not None:
        mask = np.asarray(evaluation_mask, dtype=bool)
        if mask.shape != support.shape:
            raise ValueError("evaluation mask has the wrong shape")
        support &= mask
    if np.count_nonzero(support) < MINIMUM_INFORMATIVE_ENTITIES:
        raise CouplingEstimationRefusal("sample has too few informative marker pairs")
    truth = truth[support]
    estimate = estimate[support]
    if not np.allclose(truth.sum(axis=-1), estimate.sum(axis=-1)) or not np.allclose(
        truth.sum(axis=-2), estimate.sum(axis=-2)
    ):
        raise FloatingPointError("prediction changed a fixed recipient margin")
    positive = truth > 0
    if np.any(estimate[positive] <= 0) or not np.isfinite(estimate).all():
        raise FloatingPointError("prediction assigns invalid mass")
    terms = np.zeros_like(truth)
    terms[positive] = truth[positive] * np.log(truth[positive] / estimate[positive])
    return float((2.0 * terms.sum(axis=(-2, -1)) / CELL_BUDGET).mean())


def _arrays(
    records: dict[str, dict[str, Any]], samples: list[dict[str, Any]]
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    axis = [sample["subject_id"] for sample in samples]
    return (
        np.asarray([records[gsm]["tables"] for gsm in axis]),
        np.asarray([records[gsm]["destroyed_tables"] for gsm in axis]),
        np.asarray([records[gsm]["rna_profile"] for gsm in axis]),
        np.asarray([records[gsm]["adt_profile"] for gsm in axis]),
        np.asarray([records[gsm]["subject_support"] for gsm in axis]),
        np.asarray([records[gsm]["pooled_support"] for gsm in axis]),
    )


def _primary_configs() -> list[PrimaryConfig]:
    return [
        PrimaryConfig(*values)
        for values in product(
            NEIGHBOR_GRID,
            HETEROGENEITY_GRID,
            RIDGE_GRID,
            GRAPH_GRID,
            TRANSPORT_GRID,
        )
    ]


def _equal_batch_mean(values: np.ndarray, samples: list[dict[str, Any]]) -> float:
    losses = np.asarray(values, dtype=float)
    means = [
        float(
            np.mean(
                [
                    losses[index]
                    for index, sample in enumerate(samples)
                    if sample["batch"] == batch
                ]
            )
        )
        for batch in SOURCE_BATCHES
    ]
    if not np.isfinite(means).all():
        return float("inf")
    return float(np.mean(means))


def _select_destroyed_transport(
    records: dict[str, dict[str, Any]],
    samples: list[dict[str, Any]],
    comparison_mask: np.ndarray,
    primary: PrimaryConfig,
) -> dict[str, Any]:
    sample_axis = [sample["subject_id"] for sample in samples]
    losses = {
        alpha: np.full(len(samples), np.nan, dtype=float) for alpha in TRANSPORT_GRID
    }
    refusals: list[dict[str, Any]] = []
    structural = PrimaryConfig(
        primary.graph_neighbors,
        primary.heterogeneity_penalty,
        primary.ridge_penalty,
        primary.graph_penalty,
        0.0,
    )
    for batch in SOURCE_BATCHES:
        training = [sample for sample in samples if sample["batch"] != batch]
        validation = [sample for sample in samples if sample["batch"] == batch]
        _, destroyed, rna_profiles, adt_profiles, training_support, _ = _arrays(
            records, training
        )
        try:
            fit = _fit_primary(
                destroyed,
                rna_profiles,
                adt_profiles,
                structural,
                training_support,
            )
        except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
            for alpha in TRANSPORT_GRID:
                refusals.append(
                    {
                        "batch": batch,
                        "method": "destroyed_link",
                        "configuration": {
                            "fixed_structure": asdict(structural),
                            "transport_multiplier": alpha,
                        },
                        "reason_code": type(error).__name__,
                        "reason": str(error),
                    }
                )
            continue
        for alpha in TRANSPORT_GRID:
            try:
                for sample in validation:
                    subject = sample["subject_id"]
                    truth = records[subject]["tables"]
                    rows, columns = _margins(truth)
                    losses[alpha][sample_axis.index(subject)] = _loss(
                        truth,
                        _predict_odds(fit["population_log_odds"], rows, columns, alpha),
                        comparison_mask & records[subject]["subject_support"],
                    )
            except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
                for sample in validation:
                    losses[alpha][sample_axis.index(sample["subject_id"])] = np.nan
                refusals.append(
                    {
                        "batch": batch,
                        "method": "destroyed_link",
                        "configuration": {
                            "fixed_structure": asdict(structural),
                            "transport_multiplier": alpha,
                        },
                        "reason_code": type(error).__name__,
                        "reason": str(error),
                    }
                )
    available = [alpha for alpha, values in losses.items() if np.isfinite(values).all()]
    if not available:
        raise SelectionRefusal(
            "destroyed-link alpha calibration has no complete candidate",
            {"refusals": refusals},
        )
    selected = min(
        available,
        key=lambda alpha: (_equal_batch_mean(losses[alpha], samples), alpha),
    )
    transport_curve = []
    for alpha in TRANSPORT_GRID:
        values = losses[alpha]
        complete = bool(np.isfinite(values).all())
        transport_curve.append(
            {
                "transport_multiplier": alpha,
                "status": "COMPLETE" if complete else "UNAVAILABLE",
                "sample_axis": sample_axis,
                "fold_losses": [
                    float(value) if np.isfinite(value) else None for value in values
                ],
                "equal_batch_mean_loss": (
                    _equal_batch_mean(values, samples) if complete else None
                ),
                "refusals": [
                    refusal
                    for refusal in refusals
                    if refusal["configuration"]["transport_multiplier"] == alpha
                ],
            }
        )
    return {
        "fixed_structure": asdict(structural),
        "selected_transport_multiplier": selected,
        "sample_axis": sample_axis,
        "selected_fold_losses": losses[selected].tolist(),
        "equal_batch_mean_loss": _equal_batch_mean(losses[selected], samples),
        "complete_transport_multipliers": available,
        "transport_curve": transport_curve,
        "refusals": refusals,
    }


def _select_source(
    records: dict[str, dict[str, Any]], samples: list[dict[str, Any]]
) -> dict[str, Any]:
    comparison_mask, comparison_mask_details = _source_comparison_mask(records, samples)
    source_support = comparison_mask_details[
        "source_subject_supported_coordinate_counts"
    ]
    primary_configs = _primary_configs()
    residual_configs = [
        ResidualConfig(*values) for values in product(RESIDUAL_FAMILIES, TRANSPORT_GRID)
    ]
    odds_configs = [
        OddsConfig(method, multiplier)
        for method in ("common_effect_cmle", "pooled_saturated_poisson")
        for multiplier in TRANSPORT_GRID
    ]
    sample_axis = [sample["subject_id"] for sample in samples]
    primary_losses = {
        config: np.full(len(samples), np.nan) for config in primary_configs
    }
    residual_losses = {
        config: np.full(len(samples), np.nan) for config in residual_configs
    }
    odds_losses = {config: np.full(len(samples), np.nan) for config in odds_configs}
    independence_losses = np.full(len(samples), np.nan)
    refusals: list[dict[str, Any]] = []
    pooled_poisson_fold_certificates: dict[str, dict[str, Any]] = {}
    for batch in SOURCE_BATCHES:
        training = [sample for sample in samples if sample["batch"] != batch]
        validation = [sample for sample in samples if sample["batch"] == batch]
        (
            tables,
            _,
            rna_profiles,
            adt_profiles,
            training_support,
            training_pooled_support,
        ) = _arrays(records, training)
        primary_fits: dict[
            tuple[int, float, float, float], Union[dict[str, Any], Exception]
        ] = {}
        for config in primary_configs:
            key = (
                config.graph_neighbors,
                config.heterogeneity_penalty,
                config.ridge_penalty,
                config.graph_penalty,
            )
            if key not in primary_fits:
                try:
                    primary_fits[key] = _fit_primary(
                        tables,
                        rna_profiles,
                        adt_profiles,
                        config,
                        training_support,
                    )
                except (
                    ValueError,
                    FloatingPointError,
                    CouplingEstimationRefusal,
                ) as error:
                    primary_fits[key] = error
            fit = primary_fits[key]
            if isinstance(fit, Exception):
                refusals.append(
                    {
                        "batch": batch,
                        "method": "primary",
                        "configuration": asdict(config),
                        "reason_code": type(fit).__name__,
                        "reason": str(fit),
                    }
                )
                continue
            try:
                for sample in validation:
                    truth = records[sample["subject_id"]]["tables"]
                    rows, columns = _margins(truth)
                    index = sample_axis.index(sample["subject_id"])
                    primary_losses[config][index] = _loss(
                        truth,
                        _predict_odds(
                            fit["population_log_odds"],
                            rows,
                            columns,
                            config.transport_multiplier,
                        ),
                        comparison_mask
                        & records[sample["subject_id"]]["subject_support"],
                    )
            except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
                for sample in validation:
                    primary_losses[config][sample_axis.index(sample["subject_id"])] = (
                        np.nan
                    )
                refusals.append(
                    {
                        "batch": batch,
                        "method": "primary",
                        "configuration": asdict(config),
                        "reason_code": type(error).__name__,
                        "reason": str(error),
                    }
                )
        for config in residual_configs:
            try:
                pooled = _residual_pool(
                    tables,
                    config.family,
                    comparison_mask,
                    training_support,
                )
                for sample in validation:
                    truth = records[sample["subject_id"]]["tables"]
                    rows, columns = _margins(truth)
                    index = sample_axis.index(sample["subject_id"])
                    residual_losses[config][index] = _loss(
                        truth,
                        _predict_residual(pooled, rows, columns, config),
                        comparison_mask
                        & records[sample["subject_id"]]["subject_support"],
                    )
            except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
                refusals.append(
                    {
                        "batch": batch,
                        "method": "residual",
                        "configuration": asdict(config),
                        "reason_code": type(error).__name__,
                        "reason": str(error),
                    }
                )
        odds: dict[str, Optional[np.ndarray]] = {}
        for method, fitter, fit_support in (
            ("common_effect_cmle", _fit_common_effect, training_support),
            (
                "pooled_saturated_poisson",
                _fit_pooled_poisson,
                training_pooled_support,
            ),
        ):
            try:
                fitted = fitter(tables, comparison_mask, fit_support)
                odds[method] = fitted["population_log_odds"]
                if method == "pooled_saturated_poisson":
                    pooled_poisson_fold_certificates[batch] = {
                        **fitted["fit_certificate"],
                        "pooled_tables_sha256": fitted["pooled_tables_sha256"],
                    }
            except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
                odds[method] = None
                for multiplier in TRANSPORT_GRID:
                    refusals.append(
                        {
                            "batch": batch,
                            "method": method,
                            "configuration": asdict(OddsConfig(method, multiplier)),
                            "reason_code": type(error).__name__,
                            "reason": str(error),
                        }
                    )
        for sample in validation:
            truth = records[sample["subject_id"]]["tables"]
            rows, columns = _margins(truth)
            index = sample_axis.index(sample["subject_id"])
            for config in odds_configs:
                if odds[config.method] is None:
                    continue
                try:
                    odds_losses[config][index] = _loss(
                        truth,
                        _predict_odds(
                            odds[config.method],
                            rows,
                            columns,
                            config.transport_multiplier,
                        ),
                        comparison_mask
                        & records[sample["subject_id"]]["subject_support"],
                    )
                except (
                    ValueError,
                    FloatingPointError,
                    CouplingEstimationRefusal,
                ) as error:
                    refusals.append(
                        {
                            "batch": batch,
                            "method": config.method,
                            "configuration": asdict(config),
                            "reason_code": type(error).__name__,
                            "reason": str(error),
                        }
                    )
            try:
                independence_losses[index] = _loss(
                    truth,
                    _independence(rows, columns),
                    comparison_mask & records[sample["subject_id"]]["subject_support"],
                )
            except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
                refusals.append(
                    {
                        "batch": batch,
                        "method": "independence",
                        "subject": sample["subject_id"],
                        "reason_code": type(error).__name__,
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
    available_odds = [
        config for config, values in odds_losses.items() if np.isfinite(values).all()
    ]
    eligibility = {
        "at_least_one_complete_primary_configuration": bool(available_primary),
        **{
            f"at_least_one_complete_{family}_residual_configuration": any(
                config.family == family for config in available_residual
            )
            for family in RESIDUAL_FAMILIES
        },
        "all_common_effect_transport_multipliers_complete": bool(
            all(
                config in available_odds
                for config in odds_configs
                if config.method == "common_effect_cmle"
            )
        ),
        "all_pooled_poisson_transport_multipliers_complete": bool(
            all(
                config in available_odds
                for config in odds_configs
                if config.method == "pooled_saturated_poisson"
            )
        ),
        "independence_complete_for_all_source_subjects": bool(
            np.isfinite(independence_losses).all()
        ),
    }
    if not all(eligibility.values()):
        raise SelectionRefusal(
            "leave-one-batch-out source eligibility requirements failed",
            {"eligibility": eligibility, "refusals": refusals},
        )
    primary = min(
        available_primary,
        key=lambda config: (_equal_batch_mean(primary_losses[config], samples), config),
    )
    try:
        destroyed_selection = _select_destroyed_transport(
            records, samples, comparison_mask, primary
        )
    except SelectionRefusal as error:
        raise SelectionRefusal(
            str(error),
            {
                "comparison_mask": comparison_mask_details,
                "eligibility": eligibility,
                "refusals": [*refusals, *error.details.get("refusals", [])],
            },
        ) from error
    eligibility["at_least_one_complete_destroyed_transport_multiplier"] = bool(
        destroyed_selection["complete_transport_multipliers"]
    )
    refusals.extend(destroyed_selection["refusals"])
    residual = min(
        available_residual,
        key=lambda config: (
            _equal_batch_mean(residual_losses[config], samples),
            RESIDUAL_FAMILIES.index(config.family),
            config.transport_multiplier,
        ),
    )
    residual_by_family = {
        family: min(
            (config for config in available_residual if config.family == family),
            key=lambda config: (
                _equal_batch_mean(residual_losses[config], samples),
                config,
            ),
        )
        for family in RESIDUAL_FAMILIES
    }
    selected_odds = {
        method: min(
            (config for config in available_odds if config.method == method),
            key=lambda config: (
                _equal_batch_mean(odds_losses[config], samples),
                config,
            ),
        )
        for method in ("common_effect_cmle", "pooled_saturated_poisson")
    }
    classical_losses = {
        "selected_residual": residual_losses[residual],
        **{method: odds_losses[config] for method, config in selected_odds.items()},
        "independence": independence_losses,
    }
    best_classical = min(
        classical_losses,
        key=lambda name: (_equal_batch_mean(classical_losses[name], samples), name),
    )
    primary_values = primary_losses[primary]
    residual_values = residual_losses[residual]
    primary_mean = _equal_batch_mean(primary_values, samples)
    residual_mean = _equal_batch_mean(residual_values, samples)
    common_mean = _equal_batch_mean(
        odds_losses[selected_odds["common_effect_cmle"]], samples
    )
    poisson_mean = _equal_batch_mean(
        odds_losses[selected_odds["pooled_saturated_poisson"]], samples
    )
    residual_batch_differences = {
        batch: float(
            np.mean(
                [
                    primary_values[index] - residual_values[index]
                    for index, sample in enumerate(samples)
                    if sample["batch"] == batch
                ]
            )
        )
        for batch in SOURCE_BATCHES
    }
    independence_mean = _equal_batch_mean(independence_losses, samples)
    independence_batch_differences = {
        batch: float(
            np.mean(
                [
                    primary_values[index] - independence_losses[index]
                    for index, sample in enumerate(samples)
                    if sample["batch"] == batch
                ]
            )
        )
        for batch in SOURCE_BATCHES
    }
    gate_checks = {
        "all_four_official_source_batches_held_out_once": True,
        "all_fourteen_source_subjects_have_at_least_64_informative_pairs": bool(
            all(
                value >= MINIMUM_INFORMATIVE_ENTITIES
                for value in source_support.values()
            )
        ),
        "mandatory_source_method_eligibility_passes": all(eligibility.values()),
        "primary_at_least_five_percent_below_calibrated_residual": bool(
            1.0 - primary_mean / residual_mean >= 0.05
        ),
        "primary_favorable_in_at_least_twelve_source_subjects": bool(
            np.count_nonzero(primary_values < residual_values) >= 12
        ),
        "primary_minus_residual_negative_in_every_source_batch": bool(
            all(value < 0.0 for value in residual_batch_differences.values())
        ),
        "primary_at_least_five_percent_below_independence": bool(
            1.0 - primary_mean / independence_mean >= 0.05
        ),
        "primary_favorable_vs_independence_in_at_least_twelve_source_subjects": bool(
            np.count_nonzero(primary_values < independence_losses) >= 12
        ),
        "primary_minus_independence_negative_in_every_source_batch": bool(
            all(value < 0.0 for value in independence_batch_differences.values())
        ),
        "primary_point_loss_below_tuned_common_effect_cmle": bool(
            primary_mean < common_mean
        ),
        "primary_point_loss_below_tuned_pooled_saturated_poisson": bool(
            primary_mean < poisson_mean
        ),
    }
    return {
        "primary": asdict(primary),
        "destroyed_link": destroyed_selection,
        "selected_residual": asdict(residual),
        "residual_by_family": {
            family: asdict(config) for family, config in residual_by_family.items()
        },
        "selected_odds": {
            method: asdict(config) for method, config in selected_odds.items()
        },
        "source_selected_best_classical": best_classical,
        "sample_axis": sample_axis,
        "selected_fold_losses": {
            "primary": primary_values.tolist(),
            "selected_residual": residual_losses[residual].tolist(),
            **{
                f"{family}_residual": residual_losses[config].tolist()
                for family, config in residual_by_family.items()
            },
            **{
                method: odds_losses[config].tolist()
                for method, config in selected_odds.items()
            },
            "independence": independence_losses.tolist(),
        },
        "equal_batch_mean_losses": {
            "primary": primary_mean,
            **{
                method: _equal_batch_mean(values, samples)
                for method, values in classical_losses.items()
            },
        },
        "primary_minus_residual_batch_means": residual_batch_differences,
        "primary_minus_independence_batch_means": independence_batch_differences,
        "source_gate": {"checks": gate_checks, "passes": all(gate_checks.values())},
        "source_informative_pair_counts": source_support,
        "comparison_mask": {
            **comparison_mask_details,
            "mask": comparison_mask.astype(np.uint8).tolist(),
            "retained_coordinates": [
                {
                    "rna_index": int(row),
                    "adt_index": int(column),
                    "rna_symbol": EXPECTED_PANEL[row][0],
                    "adt_target": EXPECTED_PANEL[column][1],
                }
                for row, column in np.argwhere(comparison_mask)
            ],
        },
        "pooled_poisson_lobo_fit_certificates": pooled_poisson_fold_certificates,
        "eligibility": eligibility,
        "candidate_counts": {
            "primary_frozen": len(primary_configs),
            "primary_complete": len(available_primary),
            "residual_frozen": len(residual_configs),
            "residual_complete": len(available_residual),
            "odds_frozen": len(odds_configs),
            "odds_complete": len(available_odds),
            "destroyed_transport_frozen": len(TRANSPORT_GRID),
            "destroyed_transport_complete": len(
                destroyed_selection["complete_transport_multipliers"]
            ),
        },
        "refusals": refusals,
    }


def _fit_models(
    records: dict[str, dict[str, Any]],
    samples: list[dict[str, Any]],
    selection: dict[str, Any],
) -> dict[str, Any]:
    (
        tables,
        destroyed,
        rna_profiles,
        adt_profiles,
        subject_support,
        pooled_support,
    ) = _arrays(records, samples)
    comparison_mask = np.asarray(selection["comparison_mask"]["mask"], dtype=bool)
    primary_config = PrimaryConfig(**selection["primary"])
    residual_config = ResidualConfig(**selection["selected_residual"])
    primary = _fit_primary(
        tables, rna_profiles, adt_profiles, primary_config, subject_support
    )
    destroyed_structure = PrimaryConfig(
        **selection["destroyed_link"]["fixed_structure"]
    )
    destroyed_config = PrimaryConfig(
        destroyed_structure.graph_neighbors,
        destroyed_structure.heterogeneity_penalty,
        destroyed_structure.ridge_penalty,
        destroyed_structure.graph_penalty,
        selection["destroyed_link"]["selected_transport_multiplier"],
    )
    destroyed_fit = _fit_primary(
        destroyed,
        rna_profiles,
        adt_profiles,
        destroyed_structure,
        subject_support,
    )
    common = _fit_common_effect(tables, comparison_mask, subject_support)
    poisson = _fit_pooled_poisson(tables, comparison_mask, pooled_support)
    common_config = OddsConfig(**selection["selected_odds"]["common_effect_cmle"])
    poisson_config = OddsConfig(
        **selection["selected_odds"]["pooled_saturated_poisson"]
    )
    models = {
        "primary": {
            "family": "hierarchical_graph_regularized_exact_conditional",
            "configuration": asdict(primary_config),
            "population_log_odds": primary["population_log_odds"].tolist(),
            "fit_certificate": primary["fit_certificate"],
        },
        "selected_residual": {
            "family": f"poisson_independence_signed_{residual_config.family}",
            "configuration": asdict(residual_config),
            "pooled_coordinate": _residual_pool(
                tables,
                residual_config.family,
                comparison_mask,
                subject_support,
            ).tolist(),
        },
        "common_effect_cmle": {
            "family": "unpenalized_common_effect_exact_conditional",
            "configuration": asdict(common_config),
            "population_log_odds": common["population_log_odds"].tolist(),
            "fit_certificate": common["fit_certificate"],
        },
        "pooled_saturated_poisson": {
            "family": "pooled_saturated_poisson_log_linear_interaction",
            "configuration": asdict(poisson_config),
            "population_log_odds": poisson["population_log_odds"].tolist(),
            "pooled_tables_sha256": poisson["pooled_tables_sha256"],
            "fit_certificate": poisson["fit_certificate"],
        },
        "independence": {"family": "poisson_row_plus_column_independence"},
        "destroyed_link": {
            "family": "hierarchical_exact_conditional_after_within_sample_link_destruction",
            "configuration": asdict(destroyed_config),
            "population_log_odds": destroyed_fit["population_log_odds"].tolist(),
            "fit_certificate": destroyed_fit["fit_certificate"],
        },
    }
    for family, raw_config in selection["residual_by_family"].items():
        config = ResidualConfig(**raw_config)
        models[f"{family}_residual"] = {
            "family": f"poisson_independence_signed_{family}",
            "configuration": asdict(config),
            "pooled_coordinate": _residual_pool(
                tables, family, comparison_mask, subject_support
            ).tolist(),
            "comparison_mask_sha256": selection["comparison_mask"]["mask_sha256"],
        }
    models["comparison_mask"] = {
        "mask": comparison_mask.astype(np.uint8).tolist(),
        "mask_sha256": selection["comparison_mask"]["mask_sha256"],
        "coordinate_count": selection["comparison_mask"]["coordinate_count"],
    }
    return models


def _predict_models(
    models: dict[str, Any], rows: np.ndarray, columns: np.ndarray
) -> dict[str, np.ndarray]:
    primary_config = PrimaryConfig(**models["primary"]["configuration"])
    residual_config = ResidualConfig(**models["selected_residual"]["configuration"])
    common_config = OddsConfig(**models["common_effect_cmle"]["configuration"])
    poisson_config = OddsConfig(**models["pooled_saturated_poisson"]["configuration"])
    destroyed_config = PrimaryConfig(**models["destroyed_link"]["configuration"])
    output = {
        "primary": _predict_odds(
            np.asarray(models["primary"]["population_log_odds"]),
            rows,
            columns,
            primary_config.transport_multiplier,
        ),
        "selected_residual": _predict_residual(
            np.asarray(models["selected_residual"]["pooled_coordinate"]),
            rows,
            columns,
            residual_config,
        ),
        "common_effect_cmle": _predict_odds(
            np.asarray(models["common_effect_cmle"]["population_log_odds"]),
            rows,
            columns,
            common_config.transport_multiplier,
        ),
        "pooled_saturated_poisson": _predict_odds(
            np.asarray(models["pooled_saturated_poisson"]["population_log_odds"]),
            rows,
            columns,
            poisson_config.transport_multiplier,
        ),
        "independence": _independence(rows, columns),
        "destroyed_link": _predict_odds(
            np.asarray(models["destroyed_link"]["population_log_odds"]),
            rows,
            columns,
            destroyed_config.transport_multiplier,
        ),
    }
    for family in RESIDUAL_FAMILIES:
        name = f"{family}_residual"
        model = models[name]
        output[name] = _predict_residual(
            np.asarray(model["pooled_coordinate"]),
            rows,
            columns,
            ResidualConfig(**model["configuration"]),
        )
    return output


def _held_margins(rna_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positives = (np.asarray(rna_counts) > 0).sum(axis=0)
    rows = np.repeat(
        np.stack([CELL_BUDGET - positives, positives], axis=1)[:, None, :],
        MARKER_COUNT,
        axis=1,
    )
    columns = np.broadcast_to(
        np.asarray([CELL_BUDGET // 2, CELL_BUDGET // 2]),
        (MARKER_COUNT, MARKER_COUNT, 2),
    ).copy()
    return rows, columns


def _informative_margin_count(
    rows: np.ndarray,
    columns: np.ndarray,
    comparison_mask: Optional[np.ndarray] = None,
) -> int:
    row = np.asarray(rows)
    column = np.asarray(columns)
    if row.shape != (MARKER_COUNT, MARKER_COUNT, 2) or column.shape != row.shape:
        raise ValueError("recipient margins have the wrong shape")
    if not np.array_equal(row.sum(axis=-1), column.sum(axis=-1)):
        raise ValueError("recipient row and column totals differ")
    total = row.sum(axis=-1)
    lower = np.maximum(0, row[..., 0] + column[..., 0] - total)
    upper = np.minimum(row[..., 0], column[..., 0])
    support = upper > lower
    if comparison_mask is not None:
        mask = np.asarray(comparison_mask, dtype=bool)
        if mask.shape != support.shape:
            raise ValueError("comparison mask has the wrong shape")
        support &= mask
    return int(np.count_nonzero(support))


def _comparison(
    units: list[str],
    unit_batches: list[str],
    primary: np.ndarray,
    comparator: np.ndarray,
    seed: int,
    *,
    formal_transfer: bool,
) -> dict[str, Any]:
    difference = np.asarray(primary) - np.asarray(comparator)
    generator = np.random.default_rng(seed)
    batch_indices = [
        np.flatnonzero(np.asarray(unit_batches) == batch) for batch in HELD_BATCHES
    ]
    expected_batch_sizes = (3, 2, 4)
    if tuple(len(indices) for indices in batch_indices) != expected_batch_sizes:
        raise ValueError("held batch sizes differ from the frozen 3/2/4 allocation")
    draws = np.concatenate(
        [
            indices[
                generator.integers(0, len(indices), size=(BOOTSTRAPS, len(indices)))
            ]
            for indices in batch_indices
        ],
        axis=1,
    )
    bootstrap = difference[draws].mean(axis=1)
    interval = np.quantile(bootstrap, [0.025, 0.975], method="linear")
    sensitivity_generator = np.random.default_rng(SENSITIVITY_BOOTSTRAP_SEED)
    sensitivity_indices = sensitivity_generator.integers(
        0, len(units), size=(BOOTSTRAPS, len(units))
    )
    sensitivity_interval = np.quantile(
        difference[sensitivity_indices].mean(axis=1),
        [0.025, 0.975],
        method="linear",
    )
    nonzero = difference[difference != 0]
    favorable = int(np.count_nonzero(nonzero < 0))
    sign_p = (
        float(
            sum(
                math.comb(len(nonzero), value)
                for value in range(favorable, len(nonzero) + 1)
            )
            / (2 ** len(nonzero))
        )
        if len(nonzero)
        else 1.0
    )
    reduction = 1.0 - float(np.mean(primary) / np.mean(comparator))
    batch_differences = {
        batch: float(
            np.mean(
                [
                    difference[index]
                    for index, observed in enumerate(unit_batches)
                    if observed == batch
                ]
            )
        )
        for batch in HELD_BATCHES
    }
    classical_checks = {
        "primary_point_loss_below_comparator": bool(difference.mean() < 0.0),
        "batch_stratified_paired_bootstrap_upper_95_below_zero": bool(
            interval[1] < 0.0
        ),
    }
    transfer_checks = {
        **classical_checks,
        "relative_deviance_reduction_at_least_five_percent": bool(reduction >= 0.05),
        "at_least_eight_of_nine_donors_favorable": bool(favorable >= 8),
        "exact_one_sided_sign_test_p_at_most_0_025": bool(sign_p <= 0.025),
        "primary_minus_comparator_negative_in_every_held_batch": bool(
            all(value < 0.0 for value in batch_differences.values())
        ),
    }
    checks = transfer_checks if formal_transfer else classical_checks
    return {
        "unit_count": len(units),
        "primary_mean_loss": float(np.mean(primary)),
        "comparator_mean_loss": float(np.mean(comparator)),
        "relative_deviance_reduction": reduction,
        "mean_paired_difference": float(difference.mean()),
        "batch_stratified_paired_bootstrap_95_interval": interval.tolist(),
        "unstratified_paired_bootstrap_95_interval_sensitivity": sensitivity_interval.tolist(),
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_seed": seed,
        "sensitivity_bootstrap_seed": SENSITIVITY_BOOTSTRAP_SEED,
        "favorable_non_tied_units": favorable,
        "non_tied_units": int(len(nonzero)),
        "exact_one_sided_sign_test_p": sign_p,
        "held_batch_mean_differences": batch_differences,
        "comparison_gate": "formal_transfer"
        if formal_transfer
        else "classical_increment",
        "checks": checks,
        "passes": all(checks.values()),
        "unit_differences": dict(zip(units, difference.tolist())),
    }


def _aggregate_unit_losses(
    samples: list[dict[str, Any]],
    sample_losses: dict[str, dict[str, float]],
) -> tuple[list[str], dict[str, np.ndarray]]:
    units = [sample["subject_id"] for sample in samples]
    if len(units) != len(set(units)):
        raise ValueError("held subject axis is duplicated")
    output = {
        method: np.asarray([values[unit] for unit in units], dtype=float)
        for method, values in sample_losses.items()
    }
    return units, output


def _held_unit_batches(samples: list[dict[str, Any]], units: list[str]) -> list[str]:
    output: list[str] = []
    for unit in units:
        batches = {
            sample["batch"] for sample in samples if sample["subject_id"] == unit
        }
        if len(batches) != 1:
            raise ValueError("a held biological unit spans official batches")
        output.append(next(iter(batches)))
    if set(output) != set(HELD_BATCHES):
        raise ValueError("held biological units do not cover all frozen batches")
    return output


def _confirmation_status(transfer_pass: bool, classical_increment: bool) -> str:
    if transfer_pass and classical_increment:
        return "CONFIRMATION_PASS_WITH_CLASSICAL_INCREMENT"
    if transfer_pass:
        return "TRANSFER_PASS_WITHOUT_CLASSICAL_INCREMENT"
    return "CONFIRMATION_FAIL"


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _require_official_paths(
    paths: dict[str, tuple[Path, Path]],
) -> None:
    for name, (observed, expected) in paths.items():
        if observed.resolve() != expected.resolve():
            raise PermissionError(f"{name} differs from the frozen protocol path")


def _require_official_base_paths(
    designation_path: Path,
    amendment_path: Path,
    metadata_path: Path,
    preflight_path: Path,
    protocol_path: Path,
    runtime_path: Path,
) -> None:
    _require_official_paths(
        {
            "candidate designation": (designation_path, DEFAULT_DESIGNATION),
            "candidate amendment": (amendment_path, DEFAULT_AMENDMENT),
            "metadata binding": (metadata_path, DEFAULT_METADATA),
            "axis preflight": (preflight_path, DEFAULT_PREFLIGHT),
            "protocol": (protocol_path, DEFAULT_PROTOCOL),
            "runtime": (runtime_path, DEFAULT_RUNTIME),
        }
    )


def _remote_tag_ids(tag: str) -> tuple[str, str]:
    result = subprocess.run(
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
    references = {
        fields[1]: fields[0]
        for line in result
        if len(fields := line.split()) == 2
    }
    tag_object = references.get(f"refs/tags/{tag}")
    commit = references.get(f"refs/tags/{tag}^{{}}")
    if tag_object is None or commit is None:
        raise PermissionError(f"public annotated tag {tag} is absent")
    return tag_object, commit


def _remote_tag_commit(tag: str) -> str:
    return _remote_tag_ids(tag)[1]


def _remote_tag_object_if_present(tag: str) -> Optional[str]:
    result = subprocess.run(
        ["git", "ls-remote", PUBLIC_ORIGIN, f"refs/tags/{tag}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return result.split()[0] if result else None


def _require_public_tag(
    tag: str,
    paths: Iterable[str],
    *,
    expected_tag_object: Optional[str] = None,
    expected_commit: Optional[str] = None,
) -> str:
    object_type = subprocess.run(
        ["git", "cat-file", "-t", tag],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if object_type != "tag":
        raise PermissionError(f"local tag {tag} is not annotated")
    tag_object = subprocess.run(
        ["git", "rev-parse", f"refs/tags/{tag}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "rev-parse", f"{tag}^{{}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    remote_object, remote_commit = _remote_tag_ids(tag)
    if (
        not tag_object
        or not commit
        or remote_object != tag_object
        or remote_commit != commit
        or (expected_tag_object is not None and tag_object != expected_tag_object)
        or (expected_commit is not None and commit != expected_commit)
    ):
        raise PermissionError(f"public tag {tag} does not match the local tag")
    for relative in paths:
        published = subprocess.run(
            ["git", "show", f"{tag}:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if published != (ROOT / relative).read_bytes():
            raise PermissionError(f"{relative} differs from public tag {tag}")
    return commit


def _require_ancestor(ancestor: str, descendant: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise PermissionError("public stage ancestry differs")


def _public_stage_specification() -> tuple[dict[str, str], list[str]]:
    protocol = _read_json(DEFAULT_PROTOCOL)
    tags = protocol.get("public_execution", {}).get("tags", {})
    required = {
        "protocol",
        "source_attempt",
        "source",
        "prediction_attempt",
        "prediction",
        "score_authorization_attempt",
        "score_authorization",
        "score_attempt",
        "result",
    }
    if not required <= set(tags) or any(
        not isinstance(tags[key], str) for key in required
    ):
        raise PermissionError("protocol public tag specification is incomplete")
    bound_paths = protocol.get("bound_paths")
    if not isinstance(bound_paths, list) or any(
        not isinstance(path, str) for path in bound_paths
    ):
        raise PermissionError("protocol bound path list is invalid")
    return tags, bound_paths


def _require_public_freeze_chain() -> str:
    protocol = _read_json(DEFAULT_PROTOCOL)
    chain = protocol.get("public_freeze_chain", {})
    candidate_freeze = protocol.get("candidate_freeze", {})
    nodes = (
        ("candidate", "candidate_v1"),
        ("amendment", "amendment_v2"),
        ("axis_preflight", "axis_preflight"),
    )
    previous = ""
    for node_name, frozen_prefix in nodes:
        node = chain.get(node_name, {})
        tag = node.get("tag")
        tag_object = node.get("annotated_tag_object")
        commit = node.get("peeled_commit")
        paths = node.get("required_paths")
        if (
            not isinstance(tag, str)
            or not isinstance(tag_object, str)
            or not isinstance(commit, str)
            or not isinstance(paths, list)
            or not paths
            or any(not isinstance(path, str) for path in paths)
            or candidate_freeze.get(f"{frozen_prefix}_tag") != tag
            or candidate_freeze.get(f"{frozen_prefix}_annotated_tag_object")
            != tag_object
            or candidate_freeze.get(f"{frozen_prefix}_peeled_commit") != commit
            or candidate_freeze.get(f"{frozen_prefix}_path") != paths[-1]
        ):
            raise PermissionError("public freeze-chain node differs from frozen fields")
        observed = _require_public_tag(
            tag,
            paths,
            expected_tag_object=tag_object,
            expected_commit=commit,
        )
        if previous:
            _require_ancestor(previous, observed)
        previous = observed
    tags, bound_paths = _public_stage_specification()
    protocol_node = chain.get("protocol", {})
    if (
        protocol_node.get("tag") != tags["protocol"]
        or protocol_node.get("peeled_commit")
        != "DERIVE_FROM_VERIFIED_ANNOTATED_TAG"
        or protocol_node.get("required_paths")
        != "Use the complete bound_paths array in this protocol."
    ):
        raise PermissionError("protocol freeze-chain node differs")
    protocol_commit = _require_public_tag(tags["protocol"], bound_paths)
    _require_ancestor(previous, protocol_commit)
    return protocol_commit


def _completion_artifact(phase: str) -> tuple[str, Path]:
    return {
        "source": ("source", DEFAULT_SOURCE),
        "prediction": ("prediction", DEFAULT_PREDICTION),
        "score-authorization": (
            "score_authorization",
            DEFAULT_SCORE_AUTHORIZATION,
        ),
    }[phase]


def _attempt_artifact(phase: str) -> tuple[str, Path]:
    return {
        "source": ("source_attempt", DEFAULT_SOURCE_ATTEMPT),
        "prediction": ("prediction_attempt", DEFAULT_PREDICTION_ATTEMPT),
        "score-authorization": (
            "score_authorization_attempt",
            DEFAULT_AUTHORIZATION_ATTEMPT,
        ),
        "score": ("score_attempt", DEFAULT_SCORE_ATTEMPT),
    }[phase]


def _public_predecessor_chain(phase: str) -> list[tuple[str, list[str]]]:
    tags, bound_paths = _public_stage_specification()
    chain: list[tuple[str, list[str]]] = []
    predecessors = {
        "source": (),
        "prediction": ("source",),
        "score-authorization": ("source", "prediction"),
        "score": ("source", "prediction", "score-authorization"),
    }[phase]
    for predecessor in predecessors:
        attempt_key, attempt = _attempt_artifact(predecessor)
        chain.append((tags[attempt_key], [*bound_paths, _relative(attempt)]))
        tag_key, artifact = _completion_artifact(predecessor)
        chain.append((tags[tag_key], [*bound_paths, _relative(artifact)]))
    return chain


def _require_public_prerequisites(phase: str) -> str:
    previous = _require_public_freeze_chain()
    for tag, paths in _public_predecessor_chain(phase):
        commit = _require_public_tag(tag, paths)
        _require_ancestor(previous, commit)
        previous = commit
    return previous


def _require_remote_completion_absent(phase: str) -> None:
    tags, _ = _public_stage_specification()
    tag_key = {
        "source": "source",
        "prediction": "prediction",
        "score-authorization": "score_authorization",
        "score": "result",
    }[phase]
    if _remote_tag_object_if_present(tags[tag_key]) is not None:
        raise PermissionError(f"public {phase} completion tag already exists")


def _require_public_attempt(phase: str, attempt_path: Path) -> str:
    previous = _require_public_prerequisites(phase)
    tags, bound_paths = _public_stage_specification()
    tag_key, expected_attempt = _attempt_artifact(phase)
    if attempt_path.resolve() != expected_attempt.resolve():
        raise PermissionError("numeric stage attempt path differs from protocol")
    commit = _require_public_tag(tags[tag_key], [*bound_paths, _relative(attempt_path)])
    _require_ancestor(previous, commit)
    return commit


def _sanitized_reason(error: BaseException) -> str:
    message = str(error)
    replacements = {
        str(ROOT.resolve()): "<repository>",
        str(Path.home().resolve()): "<home>",
        str(DEFAULT_SCRATCH): "<scratch>",
    }
    for private, public in replacements.items():
        message = message.replace(private, public)
    return message or type(error).__name__


def _read_source_panels(
    designation: dict[str, Any],
    samples: list[dict[str, Any]],
    scratch: Path,
    access_audit: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    markers = designation["strict_cognates"]
    panels: dict[str, dict[str, Any]] = {}
    files: dict[str, dict[str, Any]] = {}
    for sample in samples:
        subject = sample["subject_id"]
        rna_path, rna_sha256 = _fetch_sample(
            designation, sample, "rna_h5", scratch, access_audit, "source"
        )
        try:
            _audit_increment(access_audit, "source_rna_h5_decode_started")
            _audit_increment(access_audit, "source_identifier_files_accessed")
            _audit_increment(access_audit, "source_numeric_assay_files_accessed")
            _audit_event(
                access_audit, "rna_h5_decode_started", sample, file_role="rna_h5"
            )
            rna = _read_rna_h5(
                rna_path,
                sample,
                markers,
                designation["rna_axis_schema"],
                access_audit,
                "source",
            )
            _audit_increment(access_audit, "source_rna_h5_reductions_completed")
            _audit_event(
                access_audit,
                "rna_h5_reduction_completed",
                sample,
                file_role="rna_h5",
                h5_datasets=rna["accessed_h5_datasets"],
            )
        finally:
            rna_path.unlink(missing_ok=True)
            _audit_file_deleted(access_audit, "source", sample, "rna_h5")
        adt_path, adt_sha256 = _fetch_sample(
            designation,
            sample,
            "adt_csv_gz",
            scratch,
            access_audit,
            "source",
        )
        try:
            _audit_increment(access_audit, "source_adt_csv_gz_decode_started")
            _audit_increment(access_audit, "source_identifier_files_accessed")
            _audit_increment(access_audit, "source_numeric_assay_files_accessed")
            _audit_event(
                access_audit,
                "adt_csv_decode_started",
                sample,
                file_role="adt_csv_gz",
            )
            adt = _read_adt_csv(
                adt_path,
                rna["barcodes"],
                markers,
                designation["adt_axis_schemas"][sample["adt_csv_schema"]],
            )
            _audit_increment(access_audit, "source_adt_csv_gz_reductions_completed")
            _audit_event(
                access_audit,
                "adt_csv_reduction_completed",
                sample,
                file_role="adt_csv_gz",
            )
        finally:
            adt_path.unlink(missing_ok=True)
            _audit_file_deleted(access_audit, "source", sample, "adt_csv_gz")
        panels[subject] = {**rna, "adt": adt}
        files[subject] = {
            "rna_h5_name": sample["rna_h5"]["name"],
            "rna_h5_bytes": int(sample["rna_h5"]["bytes"]),
            "rna_h5_sha256": rna_sha256,
            "adt_csv_gz_name": sample["adt_csv_gz"]["name"],
            "adt_csv_gz_bytes": int(sample["adt_csv_gz"]["bytes"]),
            "adt_csv_gz_sha256": adt_sha256,
            "h5_datasets_read": rna["accessed_h5_datasets"],
            "qc_eligible_cells": rna["qc_eligible_cells"],
        }
    return panels, files


def _read_held_rna_panels(
    designation: dict[str, Any],
    samples: list[dict[str, Any]],
    scratch: Path,
    access_audit: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    markers = designation["strict_cognates"]
    panels: dict[str, dict[str, Any]] = {}
    files: dict[str, dict[str, Any]] = {}
    for sample in samples:
        subject = sample["subject_id"]
        path, file_sha256 = _fetch_sample(
            designation, sample, "rna_h5", scratch, access_audit, "held"
        )
        try:
            _audit_increment(access_audit, "held_rna_h5_decode_started")
            _audit_increment(access_audit, "held_identifier_files_accessed")
            _audit_increment(access_audit, "held_numeric_assay_files_accessed")
            _audit_event(
                access_audit, "rna_h5_decode_started", sample, file_role="rna_h5"
            )
            panel = _read_rna_h5(
                path,
                sample,
                markers,
                designation["rna_axis_schema"],
                access_audit,
                "held",
            )
            _audit_increment(access_audit, "held_rna_h5_reductions_completed")
            _audit_event(
                access_audit,
                "rna_h5_reduction_completed",
                sample,
                file_role="rna_h5",
                h5_datasets=panel["accessed_h5_datasets"],
            )
        finally:
            path.unlink(missing_ok=True)
            _audit_file_deleted(access_audit, "held", sample, "rna_h5")
        panels[subject] = panel
        files[subject] = {
            "rna_h5_name": sample["rna_h5"]["name"],
            "rna_h5_bytes": int(sample["rna_h5"]["bytes"]),
            "rna_h5_sha256": file_sha256,
            "h5_datasets_read": panel["accessed_h5_datasets"],
            "qc_eligible_cells": panel["qc_eligible_cells"],
        }
    return panels, files


def _one_shot(
    phase: str,
    attempt_path: Path,
    output_path: Path,
    bindings: dict[str, str],
    body: Any,
    access_audit: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    expected_attempt = {
        "source": DEFAULT_SOURCE_ATTEMPT,
        "prediction": DEFAULT_PREDICTION_ATTEMPT,
        "score-authorization": DEFAULT_AUTHORIZATION_ATTEMPT,
        "score": DEFAULT_SCORE_ATTEMPT,
    }[phase]
    expected_output = {
        "source": DEFAULT_SOURCE,
        "prediction": DEFAULT_PREDICTION,
        "score-authorization": DEFAULT_SCORE_AUTHORIZATION,
        "score": DEFAULT_SCORE,
    }[phase]
    _require_official_paths(
        {
            "attempt path": (attempt_path, expected_attempt),
            "output path": (output_path, expected_output),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        output_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    audit = access_audit or _new_access_audit(phase)
    with os.fdopen(descriptor, "w") as stream:
        try:
            _validate_attempt(attempt_path, phase, bindings)
            _require_remote_completion_absent(phase)
            _require_public_attempt(phase, attempt_path)
            payload = body()
        except BaseException as error:
            payload = {
                "schema": f"gse309593-held-batches-{phase}/1.0",
                "status": f"TERMINAL_{phase.upper().replace('-', '_')}_REFUSAL",
                "created_at_utc": _timestamp(),
                "reason_code": type(error).__name__,
                "reason": _sanitized_reason(error),
                "bindings": bindings,
            }
            details = getattr(error, "details", None)
            if isinstance(details, dict):
                payload["details"] = details
        try:
            if not isinstance(payload, dict):
                raise TypeError("stage body did not return one JSON object")
            payload = {**payload, "incremental_nonnumeric_access_audit": audit}
            encoded = json.dumps(
                payload, indent=2, sort_keys=True, allow_nan=False
            ) + "\n"
        except BaseException as error:
            payload = {
                "schema": f"gse309593-held-batches-{phase}/1.0",
                "status": f"TERMINAL_{phase.upper().replace('-', '_')}_REFUSAL",
                "created_at_utc": _timestamp(),
                "reason_code": "UnpublishablePayload",
                "reason": _sanitized_reason(error),
                "bindings": bindings,
                "incremental_nonnumeric_access_audit": audit,
            }
            encoded = json.dumps(
                payload, indent=2, sort_keys=True, allow_nan=False
            ) + "\n"
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return payload


def _require_bindings(result: dict[str, Any], bindings: dict[str, str]) -> None:
    if any(result.get(key) != value for key, value in bindings.items()):
        raise PermissionError("predecessor artifact bindings differ")


def _private_rna_mode_is_0600(path: Path) -> bool:
    return stat.S_IMODE(path.stat().st_mode) == 0o600


def _write_private_rna(
    path: Path,
    samples: list[dict[str, Any]],
    panels: dict[str, dict[str, Any]],
) -> tuple[str, int, dict[str, str]]:
    subjects = [sample["subject_id"] for sample in samples]
    states = np.asarray(
        [(panels[subject]["rna"] > 0).astype(np.uint8) for subject in subjects],
        dtype=np.uint8,
    )
    barcodes = np.asarray([panels[subject]["barcodes"] for subject in subjects])
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        np.savez_compressed(
            stream,
            subject_axis=np.asarray(subjects),
            barcodes=barcodes,
            states=states,
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o600)
    if not _private_rna_mode_is_0600(path):
        raise PermissionError("private held RNA artifact is not mode 0600")
    return (
        _sha256(path),
        path.stat().st_size,
        {
            subject: _array_sha256(states[index])
            for index, subject in enumerate(subjects)
        },
    )


def _read_private_rna(
    path: Path, samples: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    if not _private_rna_mode_is_0600(path):
        raise PermissionError("private held RNA artifact is not mode 0600")
    with np.load(path, allow_pickle=False) as archive:
        subjects = [str(value) for value in archive["subject_axis"]]
        barcodes = np.asarray(archive["barcodes"])
        states = np.asarray(archive["states"], dtype=np.uint8)
    expected = [sample["subject_id"] for sample in samples]
    if (
        subjects != expected
        or barcodes.shape != (len(samples), CELL_BUDGET)
        or states.shape != (len(samples), CELL_BUDGET, MARKER_COUNT)
        or np.any((states != 0) & (states != 1))
    ):
        raise PermissionError("private held RNA axes or states differ")
    return {
        subject: {
            "barcodes": [str(value) for value in barcodes[index]],
            "states": states[index],
        }
        for index, subject in enumerate(subjects)
    }


def _validate_integer_certificate(
    certificate: dict[str, Any], expected: dict[str, int], name: str
) -> None:
    if any(
        type(certificate.get(field)) is not int or certificate[field] != value
        for field, value in expected.items()
    ):
        raise PermissionError(f"{name} access certificate differs")


def _frozen_sample_axis(samples: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "subject_id": sample["subject_id"],
            "gsm": sample["gsm"],
            "batch": sample["batch"],
        }
        for sample in samples
    ]


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _matches_optional_frozen_sha256(observed: Any, frozen: Any) -> bool:
    return _valid_sha256(observed) and (frozen is None or observed == frozen)


def _require_finite_array_field(
    record: dict[str, Any], field: str, shape: tuple[int, ...], context: str
) -> np.ndarray:
    try:
        values = np.asarray(record[field], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise PermissionError(f"{context} field {field} is invalid") from error
    if values.shape != shape or not np.isfinite(values).all():
        raise PermissionError(f"{context} field {field} is invalid")
    return values


def _validate_source_access_certificate(
    source: dict[str, Any], expected_samples: list[dict[str, Any]]
) -> None:
    expected_axis = _frozen_sample_axis(expected_samples)
    subjects = source.get("source_subjects")
    files = source.get("input_files")
    if (
        source.get("schema") != "gse309593-held-batches-source/1.0"
        or source.get("source_samples") != expected_axis
        or subjects != [sample["subject_id"] for sample in expected_axis]
        or source.get("source_gsms") != [sample["gsm"] for sample in expected_axis]
        or not isinstance(files, dict)
        or set(files) != set(subjects)
    ):
        raise PermissionError("source frozen sample axis differs")
    for sample in expected_samples:
        subject = sample["subject_id"]
        record = files.get(subject)
        if (
            not isinstance(record, dict)
            or record.get("rna_h5_name") != sample["rna_h5"]["name"]
            or type(record.get("rna_h5_bytes")) is not int
            or record["rna_h5_bytes"] != int(sample["rna_h5"]["bytes"])
            or not _matches_optional_frozen_sha256(
                record.get("rna_h5_sha256"), sample.get("rna_h5_sha256")
            )
            or record.get("adt_csv_gz_name") != sample["adt_csv_gz"]["name"]
            or type(record.get("adt_csv_gz_bytes")) is not int
            or record["adt_csv_gz_bytes"] != int(sample["adt_csv_gz"]["bytes"])
            or not _matches_optional_frozen_sha256(
                record.get("adt_csv_gz_sha256"), sample.get("adt_csv_gz_sha256")
            )
            or record.get("h5_datasets_read") != sorted(H5_DATASET_ALLOWLIST)
        ):
            raise PermissionError("source per-subject file certificate differs")
    _validate_integer_certificate(
        source.get("access", {}),
        {
            "source_rna_h5_files_requested": 14,
            "source_rna_h5_files_read": 14,
            "source_rna_h5_reductions_completed": 14,
            "source_adt_csv_files_requested": 14,
            "source_adt_csv_files_read": 14,
            "source_adt_csv_reductions_completed": 14,
            "source_embedded_h5_adt_datasets_read": 0,
            "held_rna_h5_files_requested": 0,
            "held_rna_h5_files_read": 0,
            "held_adt_csv_files_requested": 0,
            "held_adt_csv_files_opened": 0,
            "held_adt_csv_hashes_completed": 0,
            "held_adt_identifiers_read": 0,
            "held_adt_numeric_values_read": 0,
            "held_adt_states_formed": 0,
            "held_joint_tables_formed": 0,
        },
        "source",
    )
    selection = source.get("selection")
    models = source.get("models")
    if not isinstance(selection, dict) or not isinstance(models, dict):
        raise PermissionError("source selection or models are absent")
    gate = selection.get("source_gate", {})
    checks = gate.get("checks")
    if (
        gate.get("passes") is not True
        or not isinstance(checks, dict)
        or not checks
        or any(value is not True for value in checks.values())
    ):
        raise PermissionError("source gate certificate does not pass every check")
    mask_record = selection.get("comparison_mask", {})
    try:
        mask_values = np.asarray(mask_record["mask"], dtype=np.uint8)
    except (KeyError, TypeError, ValueError) as error:
        raise PermissionError("source comparison mask is invalid") from error
    if (
        mask_values.shape != (MARKER_COUNT, MARKER_COUNT)
        or np.any((mask_values != 0) & (mask_values != 1))
        or mask_record.get("mask_sha256") != _array_sha256(mask_values)
        or type(mask_record.get("coordinate_count")) is not int
        or mask_record["coordinate_count"] != int(mask_values.sum())
        or mask_record["coordinate_count"] < 288
    ):
        raise PermissionError("source comparison mask certificate differs")
    required_models = {
        "primary",
        "selected_residual",
        "pearson_residual",
        "root_deviance_residual",
        "common_effect_cmle",
        "pooled_saturated_poisson",
        "independence",
        "destroyed_link",
        "comparison_mask",
    }
    if set(models) != required_models:
        raise PermissionError("source model family set differs")
    copied_mask = models["comparison_mask"]
    if (
        copied_mask.get("mask") != mask_record["mask"]
        or copied_mask.get("mask_sha256") != mask_record["mask_sha256"]
        or copied_mask.get("coordinate_count") != mask_record["coordinate_count"]
    ):
        raise PermissionError("source model comparison mask differs")
    for method in ("primary", "common_effect_cmle", "pooled_saturated_poisson", "destroyed_link"):
        record = models.get(method)
        if not isinstance(record, dict) or not isinstance(
            record.get("configuration"), dict
        ):
            raise PermissionError(f"source {method} configuration is absent")
        _require_finite_array_field(
            record,
            "population_log_odds",
            (MARKER_COUNT, MARKER_COUNT),
            f"source {method}",
        )
    for method in ("selected_residual", "pearson_residual", "root_deviance_residual"):
        record = models.get(method)
        if not isinstance(record, dict) or not isinstance(
            record.get("configuration"), dict
        ):
            raise PermissionError(f"source {method} configuration is absent")
        _require_finite_array_field(
            record,
            "pooled_coordinate",
            (MARKER_COUNT, MARKER_COUNT),
            f"source {method}",
        )


def _validate_prediction_access_certificate(
    prediction: dict[str, Any],
    expected_samples: list[dict[str, Any]],
    source: dict[str, Any],
) -> None:
    expected_axis = _frozen_sample_axis(expected_samples)
    samples = prediction.get("samples")
    if (
        prediction.get("schema") != "gse309593-held-batches-prediction/1.0"
        or prediction.get("status") != "HELD_MARGIN_ONLY_PREDICTIONS_FROZEN"
        or prediction.get("held_batches") != list(HELD_BATCHES)
        or not isinstance(samples, list)
        or len(samples) != len(expected_axis)
        or not all(isinstance(sample, dict) for sample in samples)
        or [
            {
                "subject_id": sample.get("subject_id"),
                "gsm": sample.get("gsm"),
                "batch": sample.get("batch"),
            }
            for sample in samples
        ]
        != expected_axis
        or type(prediction.get("held_subject_count")) is not int
        or prediction["held_subject_count"] != len(expected_axis)
    ):
        raise PermissionError("prediction frozen sample axis differs")

    source_selection = source.get("selection")
    if not isinstance(source_selection, dict):
        raise PermissionError("prediction source selection is absent")
    source_mask_record = source_selection.get("comparison_mask")
    if not isinstance(source_mask_record, dict):
        raise PermissionError("prediction source comparison mask is absent")
    try:
        source_mask = np.asarray(source_mask_record["mask"], dtype=np.uint8)
    except (KeyError, TypeError, ValueError) as error:
        raise PermissionError("prediction source comparison mask is invalid") from error
    source_mask_sha256 = _array_sha256(source_mask)
    source_mask_count = int(np.count_nonzero(source_mask))
    if (
        source_mask.shape != (MARKER_COUNT, MARKER_COUNT)
        or np.any((source_mask != 0) & (source_mask != 1))
        or source_mask_record.get("mask_sha256") != source_mask_sha256
        or source_mask_record.get("coordinate_count") != source_mask_count
        or prediction.get("comparison_mask_sha256") != source_mask_sha256
        or prediction.get("comparison_mask_coordinate_count") != source_mask_count
    ):
        raise PermissionError("prediction comparison mask certificate differs")
    if prediction.get("source_selected_best_classical") != source_selection.get(
        "source_selected_best_classical"
    ):
        raise PermissionError("prediction selected classical comparator differs")
    if (
        not _valid_sha256(prediction.get("private_rna_sha256"))
        or type(prediction.get("private_rna_bytes")) is not int
        or prediction["private_rna_bytes"] <= 0
    ):
        raise PermissionError("prediction private RNA certificate differs")

    expected_methods = {
        "primary",
        "selected_residual",
        *(f"{family}_residual" for family in RESIDUAL_FAMILIES),
        "common_effect_cmle",
        "pooled_saturated_poisson",
        "independence",
        "destroyed_link",
    }
    expected_support: dict[str, int] = {}
    expected_columns = np.full(
        (MARKER_COUNT, MARKER_COUNT, 2), CELL_BUDGET // 2, dtype=np.int64
    )
    for frozen, sample in zip(expected_samples, samples):
        if (
            sample.get("rna_h5_name") != frozen["rna_h5"]["name"]
            or type(sample.get("rna_h5_bytes")) is not int
            or sample["rna_h5_bytes"] != int(frozen["rna_h5"]["bytes"])
            or not _matches_optional_frozen_sha256(
                sample.get("rna_h5_sha256"), frozen.get("rna_h5_sha256")
            )
            or sample.get("h5_datasets_read") != sorted(H5_DATASET_ALLOWLIST)
        ):
            raise PermissionError("prediction per-subject H5 certificate differs")
        if any(
            not _valid_sha256(sample.get(field))
            for field in (
                "barcode_axis_sha256",
                "selected_cell_axis_sha256",
                "rna_state_sha256",
            )
        ):
            raise PermissionError("prediction per-subject axis certificate differs")
        try:
            rows = np.asarray(sample["row_margins"], dtype=float)
            columns = np.asarray(sample["column_margins"], dtype=float)
        except (KeyError, TypeError, ValueError) as error:
            raise PermissionError("prediction per-subject margins are invalid") from error
        if (
            rows.shape != (MARKER_COUNT, MARKER_COUNT, 2)
            or columns.shape != rows.shape
            or not np.isfinite(rows).all()
            or not np.isfinite(columns).all()
            or np.any(rows < 0)
            or np.any(columns < 0)
            or not np.array_equal(rows, np.rint(rows))
            or not np.array_equal(columns, np.rint(columns))
        ):
            raise PermissionError("prediction per-subject margins are invalid")
        integer_rows = rows.astype(np.int64)
        integer_columns = columns.astype(np.int64)
        if (
            not np.array_equal(
                integer_rows.sum(axis=-1), integer_columns.sum(axis=-1)
            )
            or not np.all(integer_rows.sum(axis=-1) == CELL_BUDGET)
            or not np.array_equal(integer_columns, expected_columns)
            or not np.array_equal(
                integer_rows,
                np.broadcast_to(integer_rows[:, :1, :], integer_rows.shape),
            )
        ):
            raise PermissionError("prediction per-subject margins are inconsistent")
        informative_count = _informative_margin_count(
            integer_rows, integer_columns, source_mask.astype(bool)
        )
        if (
            type(sample.get("informative_margin_pair_count")) is not int
            or sample["informative_margin_pair_count"] != informative_count
            or informative_count < MINIMUM_INFORMATIVE_ENTITIES
        ):
            raise PermissionError("prediction per-subject margin support differs")
        expected_support[str(sample["subject_id"])] = informative_count

        tables = sample.get("predicted_tables")
        table_hashes = sample.get("prediction_sha256")
        if (
            not isinstance(tables, dict)
            or not isinstance(table_hashes, dict)
            or set(tables) != expected_methods
            or set(table_hashes) != expected_methods
        ):
            raise PermissionError("prediction method family set differs")
        for method in sorted(expected_methods):
            try:
                predicted = np.asarray(tables[method], dtype=float)
            except (TypeError, ValueError) as error:
                raise PermissionError(
                    f"prediction table for {method} is invalid"
                ) from error
            predicted_rows, predicted_columns = _margins(predicted)
            if (
                predicted.shape != (MARKER_COUNT, MARKER_COUNT, 2, 2)
                or not np.isfinite(predicted).all()
                or np.any(predicted < 0)
                or not np.allclose(predicted_rows, rows, rtol=0.0, atol=1e-8)
                or not np.allclose(predicted_columns, columns, rtol=0.0, atol=1e-8)
                or not _valid_sha256(table_hashes[method])
                or table_hashes[method] != _array_sha256(predicted)
            ):
                raise PermissionError(f"prediction table for {method} differs")
    if prediction.get("held_informative_margin_pair_counts") != expected_support:
        raise PermissionError("prediction held margin support certificate differs")
    access = prediction.get("access", {})
    if (
        access.get("held_h5_dataset_allowlist") != sorted(H5_DATASET_ALLOWLIST)
        or access.get("held_h5_unique_decoded_dataset_set")
        != sorted(H5_DATASET_ALLOWLIST)
    ):
        raise PermissionError("prediction H5 dataset-set certificate differs")
    _validate_integer_certificate(
        access,
        {
            "held_rna_h5_files_requested": 9,
            "held_rna_h5_files_read": 9,
            "held_rna_h5_reductions_completed": 9,
            "held_h5_datasets_outside_allowlist_read": 0,
            "held_embedded_h5_adt_datasets_read": 0,
            "held_adt_csv_files_requested": 0,
            "held_adt_csv_files_opened": 0,
            "held_adt_csv_hashes_completed": 0,
            "held_adt_identifiers_read": 0,
            "held_adt_numeric_values_read": 0,
            "held_adt_states_formed": 0,
            "held_joint_tables_formed": 0,
        },
        "prediction",
    )


def _validate_score_authorization(
    authorization: dict[str, Any],
    bindings: dict[str, str],
    source_path: Path,
    prediction_path: Path,
    private_rna_path: Path,
) -> None:
    prediction_tag = "gse309593-held-batches-v1-prediction"
    expected = {
        "schema": "gse309593-held-batches-score-authorization/1.0",
        "status": "SCORE_AUTHORIZED_WITHOUT_OUTCOME_ACCESS",
        "held_adt_numeric_access_authorized": True,
        "outcome_access_authorized": True,
        "held_adt_files_opened_before_authorization": 0,
        "held_adt_numeric_values_read_before_authorization": 0,
        "held_joint_tables_formed_before_authorization": 0,
        "prediction_tag": prediction_tag,
        "prediction_commit": _remote_tag_commit(prediction_tag),
        "prediction_path": _relative(prediction_path),
        "prediction_sha256": _sha256(prediction_path),
        "prediction_bytes": prediction_path.stat().st_size,
        "private_rna_state_sha256": _sha256(private_rna_path),
        "private_rna_state_bytes": private_rna_path.stat().st_size,
        "source_output_sha256": _sha256(source_path),
        "protocol_sha256": bindings["protocol_sha256"],
        "runtime_environment_sha256": bindings["runtime_sha256"],
        "runner_sha256": bindings["runner_sha256"],
        "test_sha256": bindings["test_sha256"],
        "coordinatewise_common_effect_solver_sha256": bindings[
            "common_effect_solver_sha256"
        ],
        "hierarchical_solver_sha256": bindings["hierarchical_module_sha256"],
        "coupling_module_sha256": bindings["coupling_module_sha256"],
        "transitive_bindings": bindings,
        "bindings": bindings,
    }
    if any(authorization.get(key) != value for key, value in expected.items()):
        raise PermissionError("score authorization certificate differs")
    if (
        authorization.get("held_adt_numeric_access_authorized") is not True
        or authorization.get("outcome_access_authorized") is not True
    ):
        raise PermissionError("score authorization boolean certificate differs")
    integer_fields = (
        "held_adt_files_opened_before_authorization",
        "held_adt_numeric_values_read_before_authorization",
        "held_joint_tables_formed_before_authorization",
        "prediction_bytes",
        "private_rna_state_bytes",
    )
    if any(type(authorization.get(field)) is not int for field in integer_fields):
        raise PermissionError("score authorization integer certificate differs")


def _stage_bindings(
    phase: str,
    base: dict[str, str],
    source_path: Path,
    prediction_path: Path,
    private_rna_path: Path,
    authorization_path: Path,
    expected_source: list[dict[str, Any]],
    expected_held: list[dict[str, Any]],
) -> dict[str, str]:
    bindings = dict(base)
    if phase in {"prediction", "score-authorization", "score"}:
        source = _read_json(source_path)
        if source.get("status") != "SOURCE_GATE_PASS":
            raise PermissionError("source gate did not pass")
        _require_bindings(source, base)
        _validate_source_access_certificate(source, expected_source)
        bindings["source_sha256"] = _sha256(source_path)
    if phase in {"score-authorization", "score"}:
        prediction = _read_json(prediction_path)
        if prediction.get("status") != "HELD_MARGIN_ONLY_PREDICTIONS_FROZEN":
            raise PermissionError("held predictions are not frozen")
        _require_bindings(prediction, bindings)
        _validate_prediction_access_certificate(prediction, expected_held, source)
        if not private_rna_path.is_file() or not _private_rna_mode_is_0600(
            private_rna_path
        ):
            raise PermissionError("private held RNA artifact is absent or exposed")
        private_sha256 = _sha256(private_rna_path)
        if prediction.get("private_rna_sha256") != private_sha256:
            raise PermissionError("private held RNA artifact differs from prediction")
        bindings["prediction_sha256"] = _sha256(prediction_path)
        bindings["private_rna_sha256"] = private_sha256
    if phase == "score":
        authorization = _read_json(authorization_path)
        _validate_score_authorization(
            authorization,
            bindings,
            source_path,
            prediction_path,
            private_rna_path,
        )
        bindings["score_authorization_sha256"] = _sha256(authorization_path)
    return bindings


def claim_stage(
    phase: str,
    designation_path: Path = DEFAULT_DESIGNATION,
    amendment_path: Path = DEFAULT_AMENDMENT,
    metadata_path: Path = DEFAULT_METADATA,
    preflight_path: Path = DEFAULT_PREFLIGHT,
    protocol_path: Path = DEFAULT_PROTOCOL,
    runtime_path: Path = DEFAULT_RUNTIME,
    source_path: Path = DEFAULT_SOURCE,
    prediction_path: Path = DEFAULT_PREDICTION,
    private_rna_path: Path = DEFAULT_PRIVATE_RNA,
    authorization_path: Path = DEFAULT_SCORE_AUTHORIZATION,
    attempt_path: Optional[Path] = None,
) -> dict[str, Any]:
    if phase not in {"source", "prediction", "score-authorization", "score"}:
        raise ValueError("unknown claim phase")
    default = {
        "source": DEFAULT_SOURCE_ATTEMPT,
        "prediction": DEFAULT_PREDICTION_ATTEMPT,
        "score-authorization": DEFAULT_AUTHORIZATION_ATTEMPT,
        "score": DEFAULT_SCORE_ATTEMPT,
    }[phase]
    actual_attempt = attempt_path or default
    _require_official_paths(
        {
            "source artifact": (source_path, DEFAULT_SOURCE),
            "prediction artifact": (prediction_path, DEFAULT_PREDICTION),
            "private RNA artifact": (private_rna_path, DEFAULT_PRIVATE_RNA),
            "score authorization": (
                authorization_path,
                DEFAULT_SCORE_AUTHORIZATION,
            ),
            "attempt path": (actual_attempt, default),
        }
    )
    _require_official_base_paths(
        designation_path,
        amendment_path,
        metadata_path,
        preflight_path,
        protocol_path,
        runtime_path,
    )
    _require_public_prerequisites(phase)
    _, source, held, base = _base_bindings(
        designation_path,
        amendment_path,
        metadata_path,
        preflight_path,
        protocol_path,
        runtime_path,
    )
    bindings = _stage_bindings(
        phase,
        base,
        source_path,
        prediction_path,
        private_rna_path,
        authorization_path,
        source,
        held,
    )
    output = {
        "source": source_path,
        "prediction": prediction_path,
        "score-authorization": authorization_path,
        "score": DEFAULT_SCORE,
    }[phase]
    if output.exists():
        raise FileExistsError(f"{phase} output already exists")
    _claim(actual_attempt, phase, bindings)
    return _read_json(actual_attempt)


def run_source(
    designation_path: Path = DEFAULT_DESIGNATION,
    amendment_path: Path = DEFAULT_AMENDMENT,
    metadata_path: Path = DEFAULT_METADATA,
    preflight_path: Path = DEFAULT_PREFLIGHT,
    protocol_path: Path = DEFAULT_PROTOCOL,
    runtime_path: Path = DEFAULT_RUNTIME,
    attempt_path: Path = DEFAULT_SOURCE_ATTEMPT,
    output_path: Path = DEFAULT_SOURCE,
    scratch: Path = DEFAULT_SCRATCH,
) -> dict[str, Any]:
    _require_official_paths(
        {
            "source attempt": (attempt_path, DEFAULT_SOURCE_ATTEMPT),
            "source output": (output_path, DEFAULT_SOURCE),
            "scratch directory": (scratch, DEFAULT_SCRATCH),
        }
    )
    claimed_bindings = _claimed_attempt_bindings(attempt_path, "source")
    access_audit = _new_access_audit("source")

    def body() -> dict[str, Any]:
        designation, source, _, bindings = _base_bindings(
            designation_path,
            amendment_path,
            metadata_path,
            preflight_path,
            protocol_path,
            runtime_path,
        )
        _require_computed_bindings(bindings, claimed_bindings)
        panels, file_audit = _read_source_panels(
            designation, source, scratch, access_audit
        )
        records = _reduce_source(panels, source, access_audit)
        selection = _select_source(records, source)
        passes = bool(selection["source_gate"]["passes"])
        payload: dict[str, Any] = {
            "schema": "gse309593-held-batches-source/1.0",
            "status": "SOURCE_GATE_PASS" if passes else "TERMINAL_SOURCE_GATE_REFUSAL",
            "created_at_utc": _timestamp(),
            **bindings,
            "source_batches": list(SOURCE_BATCHES),
            "source_samples": _frozen_sample_axis(source),
            "source_subjects": [sample["subject_id"] for sample in source],
            "source_gsms": [sample["gsm"] for sample in source],
            "rna_marker_axis": [
                marker["rna_symbol"] for marker in designation["strict_cognates"]
            ],
            "adt_marker_axis": [
                marker["adt_target"] for marker in designation["strict_cognates"]
            ],
            "selection": selection,
            "source_informative_pair_counts": selection[
                "source_informative_pair_counts"
            ],
            "source_table_sha256": {
                subject: record["table_sha256"] for subject, record in records.items()
            },
            "source_destroyed_table_sha256": {
                subject: record["destroyed_table_sha256"]
                for subject, record in records.items()
            },
            "source_selected_cell_axis_sha256": {
                subject: record["selected_cell_axis_sha256"]
                for subject, record in records.items()
            },
            "source_adt_marker_support": {
                subject: record["adt_marker_support"].astype(np.uint8).tolist()
                for subject, record in records.items()
            },
            "source_subject_support_sha256": {
                subject: _array_sha256(record["subject_support"].astype(np.uint8))
                for subject, record in records.items()
            },
            "input_files": file_audit,
            "access": {
                "source_rna_h5_files_requested": len(source),
                "source_rna_h5_files_read": len(source),
                "source_rna_h5_reductions_completed": len(source),
                "source_adt_csv_files_requested": len(source),
                "source_adt_csv_files_read": len(source),
                "source_adt_csv_reductions_completed": len(source),
                "source_embedded_h5_adt_datasets_read": 0,
                "held_rna_h5_files_requested": 0,
                "held_rna_h5_files_read": 0,
                "held_adt_csv_files_requested": 0,
                "held_adt_csv_files_opened": 0,
                "held_adt_csv_hashes_completed": 0,
                "held_adt_identifiers_read": 0,
                "held_adt_numeric_values_read": 0,
                "held_adt_states_formed": 0,
                "held_joint_tables_formed": 0,
            },
        }
        if passes:
            payload["models"] = _fit_models(records, source, selection)
        return payload

    return _one_shot(
        "source", attempt_path, output_path, claimed_bindings, body, access_audit
    )


def run_prediction(
    designation_path: Path = DEFAULT_DESIGNATION,
    amendment_path: Path = DEFAULT_AMENDMENT,
    metadata_path: Path = DEFAULT_METADATA,
    preflight_path: Path = DEFAULT_PREFLIGHT,
    protocol_path: Path = DEFAULT_PROTOCOL,
    runtime_path: Path = DEFAULT_RUNTIME,
    source_path: Path = DEFAULT_SOURCE,
    attempt_path: Path = DEFAULT_PREDICTION_ATTEMPT,
    output_path: Path = DEFAULT_PREDICTION,
    private_rna_path: Path = DEFAULT_PRIVATE_RNA,
    scratch: Path = DEFAULT_SCRATCH,
) -> dict[str, Any]:
    _require_official_paths(
        {
            "source artifact": (source_path, DEFAULT_SOURCE),
            "prediction attempt": (attempt_path, DEFAULT_PREDICTION_ATTEMPT),
            "prediction output": (output_path, DEFAULT_PREDICTION),
            "private RNA artifact": (private_rna_path, DEFAULT_PRIVATE_RNA),
            "scratch directory": (scratch, DEFAULT_SCRATCH),
        }
    )
    claimed_bindings = _claimed_attempt_bindings(attempt_path, "prediction")
    access_audit = _new_access_audit("prediction")

    def body() -> dict[str, Any]:
        designation, source, held, base = _base_bindings(
            designation_path,
            amendment_path,
            metadata_path,
            preflight_path,
            protocol_path,
            runtime_path,
        )
        bindings = _stage_bindings(
            "prediction",
            base,
            source_path,
            output_path,
            private_rna_path,
            DEFAULT_SCORE_AUTHORIZATION,
            source,
            held,
        )
        _require_computed_bindings(bindings, claimed_bindings)
        source_result = _read_json(source_path)
        panels, file_audit = _read_held_rna_panels(
            designation, held, scratch, access_audit
        )
        comparison_mask = np.asarray(
            source_result["selection"]["comparison_mask"]["mask"], dtype=bool
        )
        if (
            _array_sha256(comparison_mask.astype(np.uint8))
            != source_result["selection"]["comparison_mask"]["mask_sha256"]
        ):
            raise PermissionError("source comparison mask bytes changed")
        support_counts = {}
        for sample in held:
            subject = sample["subject_id"]
            rows, columns = _held_margins(panels[subject]["rna"])
            support_counts[subject] = _informative_margin_count(
                rows, columns, comparison_mask
            )
        if any(
            value < MINIMUM_INFORMATIVE_ENTITIES for value in support_counts.values()
        ):
            raise SelectionRefusal(
                "a held subject has fewer than 64 informative margin pairs",
                {"held_informative_margin_pair_counts": support_counts},
            )
        private_sha256, private_bytes, state_hashes = _write_private_rna(
            private_rna_path, held, panels
        )
        _audit_increment(access_audit, "held_state_panels_formed", len(held))
        for sample in held:
            _audit_event(access_audit, "held_rna_state_panel_formed", sample)
        samples = []
        for sample in held:
            subject = sample["subject_id"]
            panel = panels[subject]
            rows, columns = _held_margins(panel["rna"])
            predictions = _predict_models(source_result["models"], rows, columns)
            samples.append(
                {
                    "subject_id": subject,
                    "gsm": sample["gsm"],
                    "batch": sample["batch"],
                    "row_margins": rows.tolist(),
                    "column_margins": columns.tolist(),
                    "predicted_tables": {
                        name: values.tolist() for name, values in predictions.items()
                    },
                    "prediction_sha256": {
                        name: _array_sha256(values)
                        for name, values in predictions.items()
                    },
                    "rna_h5_name": file_audit[subject]["rna_h5_name"],
                    "rna_h5_bytes": file_audit[subject]["rna_h5_bytes"],
                    "rna_h5_sha256": file_audit[subject]["rna_h5_sha256"],
                    "h5_datasets_read": file_audit[subject]["h5_datasets_read"],
                    "barcode_axis_sha256": panel["barcode_axis_sha256"],
                    "selected_cell_axis_sha256": panel["selected_cell_axis_sha256"],
                    "rna_state_sha256": state_hashes[subject],
                    "informative_margin_pair_count": support_counts[subject],
                }
            )
        return {
            "schema": "gse309593-held-batches-prediction/1.0",
            "status": "HELD_MARGIN_ONLY_PREDICTIONS_FROZEN",
            "created_at_utc": _timestamp(),
            **bindings,
            "held_batches": list(HELD_BATCHES),
            "held_subject_count": len(held),
            "private_rna_sha256": private_sha256,
            "private_rna_bytes": private_bytes,
            "source_selected_best_classical": source_result["selection"][
                "source_selected_best_classical"
            ],
            "samples": samples,
            "held_informative_margin_pair_counts": support_counts,
            "comparison_mask_sha256": source_result["selection"]["comparison_mask"][
                "mask_sha256"
            ],
            "comparison_mask_coordinate_count": int(np.count_nonzero(comparison_mask)),
            "access": {
                "held_rna_h5_files_requested": len(held),
                "held_rna_h5_files_read": len(held),
                "held_rna_h5_reductions_completed": len(held),
                "held_h5_dataset_allowlist": sorted(H5_DATASET_ALLOWLIST),
                "held_h5_unique_decoded_dataset_set": sorted(
                    H5_DATASET_ALLOWLIST
                ),
                "held_h5_datasets_outside_allowlist_read": 0,
                "held_embedded_h5_adt_datasets_read": 0,
                "held_adt_csv_files_requested": 0,
                "held_adt_csv_files_opened": 0,
                "held_adt_csv_hashes_completed": 0,
                "held_adt_identifiers_read": 0,
                "held_adt_numeric_values_read": 0,
                "held_adt_states_formed": 0,
                "held_joint_tables_formed": 0,
            },
        }

    return _one_shot(
        "prediction",
        attempt_path,
        output_path,
        claimed_bindings,
        body,
        access_audit,
    )


def _validate_authorization_template(path: Path) -> dict[str, Any]:
    template = _read_json(path)
    if (
        template.get("schema") != "gse309593-held-batches-score-authorization/1.0"
        or template.get("status") != "DISABLED_TEMPLATE"
        or template.get("held_adt_numeric_access_authorized") is not False
        or template.get("outcome_access_authorized") is not False
        or template.get("held_adt_files_opened_before_authorization") != 0
        or template.get("held_adt_numeric_values_read_before_authorization") != 0
        or template.get("held_joint_tables_formed_before_authorization") != 0
    ):
        raise PermissionError("score authorization template is not disabled")
    null_fields = (
        "prediction_commit",
        "prediction_sha256",
        "prediction_bytes",
        "private_rna_state_sha256",
        "private_rna_state_bytes",
        "source_output_sha256",
        "protocol_sha256",
        "runtime_environment_sha256",
        "runner_sha256",
        "test_sha256",
        "coordinatewise_common_effect_solver_sha256",
        "hierarchical_solver_sha256",
        "coupling_module_sha256",
        "transitive_bindings",
    )
    if any(template.get(field) is not None for field in null_fields):
        raise PermissionError("score authorization template is already populated")
    return template


def authorize_score(
    designation_path: Path = DEFAULT_DESIGNATION,
    amendment_path: Path = DEFAULT_AMENDMENT,
    metadata_path: Path = DEFAULT_METADATA,
    preflight_path: Path = DEFAULT_PREFLIGHT,
    protocol_path: Path = DEFAULT_PROTOCOL,
    runtime_path: Path = DEFAULT_RUNTIME,
    template_path: Path = DEFAULT_AUTHORIZATION_TEMPLATE,
    source_path: Path = DEFAULT_SOURCE,
    prediction_path: Path = DEFAULT_PREDICTION,
    private_rna_path: Path = DEFAULT_PRIVATE_RNA,
    attempt_path: Path = DEFAULT_AUTHORIZATION_ATTEMPT,
    output_path: Path = DEFAULT_SCORE_AUTHORIZATION,
) -> dict[str, Any]:
    _require_official_paths(
        {
            "authorization template": (
                template_path,
                DEFAULT_AUTHORIZATION_TEMPLATE,
            ),
            "source artifact": (source_path, DEFAULT_SOURCE),
            "prediction artifact": (prediction_path, DEFAULT_PREDICTION),
            "private RNA artifact": (private_rna_path, DEFAULT_PRIVATE_RNA),
            "authorization attempt": (
                attempt_path,
                DEFAULT_AUTHORIZATION_ATTEMPT,
            ),
            "authorization output": (
                output_path,
                DEFAULT_SCORE_AUTHORIZATION,
            ),
        }
    )
    claimed_bindings = _claimed_attempt_bindings(
        attempt_path, "score-authorization"
    )
    access_audit = _new_access_audit("score-authorization")

    def body() -> dict[str, Any]:
        _, source, held, base = _base_bindings(
            designation_path,
            amendment_path,
            metadata_path,
            preflight_path,
            protocol_path,
            runtime_path,
        )
        bindings = _stage_bindings(
            "score-authorization",
            base,
            source_path,
            prediction_path,
            private_rna_path,
            output_path,
            source,
            held,
        )
        _require_computed_bindings(bindings, claimed_bindings)
        template = _validate_authorization_template(template_path)
        prediction = _read_json(prediction_path)
        if template.get("prediction_tag") != "gse309593-held-batches-v1-prediction":
            raise PermissionError("authorization template prediction tag differs")
        if template.get("prediction_path") != _relative(prediction_path):
            raise PermissionError("authorization template prediction path differs")
        if prediction.get("access", {}).get("held_adt_numeric_values_read") != 0:
            raise PermissionError("held ADT values were accessed before authorization")
        return {
            "schema": "gse309593-held-batches-score-authorization/1.0",
            "status": "SCORE_AUTHORIZED_WITHOUT_OUTCOME_ACCESS",
            "created_at_utc": _timestamp(),
            "held_adt_numeric_access_authorized": True,
            "outcome_access_authorized": True,
            "held_adt_files_opened_before_authorization": 0,
            "held_adt_numeric_values_read_before_authorization": 0,
            "held_joint_tables_formed_before_authorization": 0,
            "prediction_tag": template["prediction_tag"],
            "prediction_commit": _remote_tag_commit(template["prediction_tag"]),
            "prediction_path": template["prediction_path"],
            "prediction_sha256": _sha256(prediction_path),
            "prediction_bytes": prediction_path.stat().st_size,
            "private_rna_state_sha256": _sha256(private_rna_path),
            "private_rna_state_bytes": private_rna_path.stat().st_size,
            "source_output_sha256": _sha256(source_path),
            "protocol_sha256": bindings["protocol_sha256"],
            "runtime_environment_sha256": bindings["runtime_sha256"],
            "runner_sha256": bindings["runner_sha256"],
            "test_sha256": bindings["test_sha256"],
            "coordinatewise_common_effect_solver_sha256": bindings[
                "common_effect_solver_sha256"
            ],
            "hierarchical_solver_sha256": bindings["hierarchical_module_sha256"],
            "coupling_module_sha256": bindings["coupling_module_sha256"],
            "transitive_bindings": bindings,
            "bindings": bindings,
        }

    return _one_shot(
        "score-authorization",
        attempt_path,
        output_path,
        claimed_bindings,
        body,
        access_audit,
    )


def _frozen_prediction_axis(
    prediction: dict[str, Any], held: list[dict[str, Any]]
) -> list[str]:
    expected = [sample["subject_id"] for sample in held]
    observed = [sample.get("subject_id") for sample in prediction.get("samples", [])]
    if observed != expected or len(set(observed)) != len(observed):
        raise PermissionError("held prediction subject axis differs from designation")
    return expected


def _verify_private_prediction(
    private_sample: dict[str, Any], frozen_sample: dict[str, Any]
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    barcodes = [str(value) for value in private_sample["barcodes"]]
    rna = np.asarray(private_sample["states"], dtype=np.uint8)
    if (
        len(barcodes) != CELL_BUDGET
        or len(set(barcodes)) != CELL_BUDGET
        or rna.shape != (CELL_BUDGET, MARKER_COUNT)
        or np.any((rna != 0) & (rna != 1))
        or _axis_sha256(barcodes) != frozen_sample["selected_cell_axis_sha256"]
        or _array_sha256(rna) != frozen_sample["rna_state_sha256"]
    ):
        raise PermissionError("private held RNA state or barcode axis changed")
    rows, columns = _held_margins(rna)
    if (
        rows.tolist() != frozen_sample["row_margins"]
        or columns.tolist() != frozen_sample["column_margins"]
    ):
        raise PermissionError("private held RNA margins changed")
    return rna, barcodes, rows, columns


def run_score(
    designation_path: Path = DEFAULT_DESIGNATION,
    amendment_path: Path = DEFAULT_AMENDMENT,
    metadata_path: Path = DEFAULT_METADATA,
    preflight_path: Path = DEFAULT_PREFLIGHT,
    protocol_path: Path = DEFAULT_PROTOCOL,
    runtime_path: Path = DEFAULT_RUNTIME,
    source_path: Path = DEFAULT_SOURCE,
    prediction_path: Path = DEFAULT_PREDICTION,
    private_rna_path: Path = DEFAULT_PRIVATE_RNA,
    authorization_path: Path = DEFAULT_SCORE_AUTHORIZATION,
    attempt_path: Path = DEFAULT_SCORE_ATTEMPT,
    output_path: Path = DEFAULT_SCORE,
    scratch: Path = DEFAULT_SCRATCH,
) -> dict[str, Any]:
    _require_official_paths(
        {
            "source artifact": (source_path, DEFAULT_SOURCE),
            "prediction artifact": (prediction_path, DEFAULT_PREDICTION),
            "private RNA artifact": (private_rna_path, DEFAULT_PRIVATE_RNA),
            "score authorization": (
                authorization_path,
                DEFAULT_SCORE_AUTHORIZATION,
            ),
            "score attempt": (attempt_path, DEFAULT_SCORE_ATTEMPT),
            "score output": (output_path, DEFAULT_SCORE),
            "scratch directory": (scratch, DEFAULT_SCRATCH),
        }
    )
    claimed_bindings = _claimed_attempt_bindings(attempt_path, "score")
    access_audit = _new_access_audit("score")

    def body() -> dict[str, Any]:
        designation, source, held, base = _base_bindings(
            designation_path,
            amendment_path,
            metadata_path,
            preflight_path,
            protocol_path,
            runtime_path,
        )
        bindings = _stage_bindings(
            "score",
            base,
            source_path,
            prediction_path,
            private_rna_path,
            authorization_path,
            source,
            held,
        )
        _require_computed_bindings(bindings, claimed_bindings)
        prediction = _read_json(prediction_path)
        source_result = _read_json(source_path)
        _frozen_prediction_axis(prediction, held)
        private = _read_private_rna(private_rna_path, held)
        comparison_mask = np.asarray(
            source_result["selection"]["comparison_mask"]["mask"], dtype=bool
        )
        mask_sha256 = _array_sha256(comparison_mask.astype(np.uint8))
        if (
            mask_sha256 != source_result["selection"]["comparison_mask"]["mask_sha256"]
            or prediction.get("comparison_mask_sha256") != mask_sha256
        ):
            raise PermissionError("frozen source comparison mask changed")
        frozen = {sample["subject_id"]: sample for sample in prediction["samples"]}
        verified_private = {
            sample["subject_id"]: _verify_private_prediction(
                private[sample["subject_id"]], frozen[sample["subject_id"]]
            )
            for sample in held
        }
        methods = (
            "primary",
            "selected_residual",
            *(f"{family}_residual" for family in RESIDUAL_FAMILIES),
            "common_effect_cmle",
            "pooled_saturated_poisson",
            "independence",
            "destroyed_link",
        )
        sample_losses = {method: {} for method in methods}
        truth_hashes: dict[str, str] = {}
        adt_state_hashes: dict[str, str] = {}
        adt_file_audit: dict[str, dict[str, Any]] = {}
        held_support_counts: dict[str, int] = {}
        held_evaluation_masks: dict[str, list[list[int]]] = {}
        held_evaluation_mask_hashes: dict[str, str] = {}
        for sample in held:
            subject = sample["subject_id"]
            frozen_sample = frozen[subject]
            rna, barcodes, rows, columns = verified_private[subject]
            adt_path, adt_sha256 = _fetch_sample(
                designation,
                sample,
                "adt_csv_gz",
                scratch,
                access_audit,
                "held",
            )
            try:
                _audit_increment(access_audit, "held_adt_csv_gz_decode_started")
                _audit_increment(access_audit, "held_identifier_files_accessed")
                _audit_increment(access_audit, "held_numeric_assay_files_accessed")
                _audit_event(
                    access_audit,
                    "adt_csv_decode_started",
                    sample,
                    file_role="adt_csv_gz",
                )
                adt_counts = _read_adt_csv(
                    adt_path,
                    barcodes,
                    designation["strict_cognates"],
                    designation["adt_axis_schemas"][sample["adt_csv_schema"]],
                )
                _audit_increment(
                    access_audit, "held_adt_csv_gz_reductions_completed"
                )
                _audit_event(
                    access_audit,
                    "adt_csv_reduction_completed",
                    sample,
                    file_role="adt_csv_gz",
                )
            finally:
                adt_path.unlink(missing_ok=True)
                _audit_file_deleted(access_audit, "held", sample, "adt_csv_gz")
            adt = _adt_states(adt_counts, barcodes, subject)
            _audit_increment(access_audit, "held_state_panels_formed")
            _audit_event(access_audit, "held_adt_state_panel_formed", sample)
            adt_marker_support = _adt_variation_support(adt_counts)
            truth = _tables(rna, adt)
            _audit_increment(access_audit, "held_joint_table_panels_formed")
            _audit_event(access_audit, "held_joint_table_panel_formed", sample)
            evaluation_mask = comparison_mask & _subject_support(
                truth, adt_marker_support
            )
            support_count = int(np.count_nonzero(evaluation_mask))
            if support_count < MINIMUM_INFORMATIVE_ENTITIES:
                raise SelectionRefusal(
                    "a held subject has fewer than 64 comparison-supported coordinates",
                    {
                        "subject_id": subject,
                        "supported_coordinate_count": support_count,
                        "adt_marker_support": adt_marker_support.astype(
                            np.uint8
                        ).tolist(),
                    },
                )
            held_support_counts[subject] = support_count
            evaluation_mask_uint8 = evaluation_mask.astype(np.uint8)
            held_evaluation_masks[subject] = evaluation_mask_uint8.tolist()
            held_evaluation_mask_hashes[subject] = _array_sha256(evaluation_mask_uint8)
            observed_rows, observed_columns = _margins(truth)
            if not np.array_equal(observed_rows, rows) or not np.array_equal(
                observed_columns, columns
            ):
                raise PermissionError("held truth margins differ from predictions")
            truth_hashes[subject] = _array_sha256(truth)
            adt_state_hashes[subject] = _array_sha256(adt)
            adt_file_audit[subject] = {
                "adt_csv_gz_name": sample["adt_csv_gz"]["name"],
                "adt_csv_gz_sha256": adt_sha256,
                "selected_identifier_count": CELL_BUDGET,
                "selected_adt_values_read": CELL_BUDGET * MARKER_COUNT,
                "adt_marker_support": adt_marker_support.astype(np.uint8).tolist(),
                "adt_supported_marker_count": int(np.count_nonzero(adt_marker_support)),
                "comparison_supported_coordinate_count": support_count,
            }
            for method in methods:
                predicted = np.asarray(
                    frozen_sample["predicted_tables"][method], dtype=float
                )
                if (
                    _array_sha256(predicted)
                    != frozen_sample["prediction_sha256"][method]
                ):
                    raise PermissionError("a frozen held prediction changed")
                sample_losses[method][subject] = _loss(
                    truth, predicted, evaluation_mask
                )
        units, unit_losses = _aggregate_unit_losses(held, sample_losses)
        unit_batches = _held_unit_batches(held, units)
        comparisons = {
            f"primary_vs_{method}": _comparison(
                units,
                unit_batches,
                unit_losses["primary"],
                unit_losses[method],
                BOOTSTRAP_SEED,
                formal_transfer=method
                in {"selected_residual", "independence", "destroyed_link"},
            )
            for method in methods[1:]
        }
        transfer_required = (
            "primary_vs_selected_residual",
            "primary_vs_independence",
            "primary_vs_destroyed_link",
        )
        classical_required = (
            "primary_vs_common_effect_cmle",
            "primary_vs_pooled_saturated_poisson",
        )
        transfer_pass = all(comparisons[name]["passes"] for name in transfer_required)
        classical_increment = all(
            comparisons[name]["passes"] for name in classical_required
        )
        return {
            "schema": "gse309593-held-batches-score/1.0",
            "status": _confirmation_status(transfer_pass, classical_increment),
            "created_at_utc": _timestamp(),
            **bindings,
            "held_batches": list(HELD_BATCHES),
            "held_subjects": units,
            "held_subject_batches": dict(zip(units, unit_batches)),
            "inference_unit": "one distinct biological subject",
            "source_selected_best_classical": prediction[
                "source_selected_best_classical"
            ],
            "sample_losses": sample_losses,
            "unit_losses": {
                method: dict(zip(units, values.tolist()))
                for method, values in unit_losses.items()
            },
            "comparisons": comparisons,
            "formal_transfer_comparisons": list(transfer_required),
            "classical_increment_comparisons": list(classical_required),
            "transfer_pass": transfer_pass,
            "classical_increment_pass": classical_increment,
            "passes": transfer_pass and classical_increment,
            "truth_table_sha256": truth_hashes,
            "adt_state_sha256": adt_state_hashes,
            "held_adt_files": adt_file_audit,
            "comparison_mask_sha256": mask_sha256,
            "held_comparison_supported_coordinate_counts": held_support_counts,
            "held_evaluation_masks": held_evaluation_masks,
            "held_evaluation_mask_sha256": held_evaluation_mask_hashes,
            "access": {
                "held_rna_h5_files_opened_during_score": 0,
                "held_embedded_h5_adt_datasets_read": 0,
                "held_adt_csv_files_first_opened_during_score": len(held),
                "held_adt_csv_hashes_completed_during_score": len(held),
                "held_joint_tables_first_formed_during_score": len(held),
            },
        }

    return _one_shot(
        "score", attempt_path, output_path, claimed_bindings, body, access_audit
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "claim-source",
            "source",
            "claim-prediction",
            "predict",
            "claim-score-authorization",
            "authorize-score",
            "claim-score",
            "score",
        ),
    )
    parser.add_argument("--designation", type=Path, default=DEFAULT_DESIGNATION)
    parser.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument(
        "--authorization-template",
        type=Path,
        default=DEFAULT_AUTHORIZATION_TEMPLATE,
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--prediction", type=Path, default=DEFAULT_PREDICTION)
    parser.add_argument("--private-rna", type=Path, default=DEFAULT_PRIVATE_RNA)
    parser.add_argument(
        "--authorization", type=Path, default=DEFAULT_SCORE_AUTHORIZATION
    )
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--attempt", type=Path)
    args = parser.parse_args()
    common = {
        "designation_path": args.designation,
        "amendment_path": args.amendment,
        "metadata_path": args.metadata,
        "preflight_path": args.preflight,
        "protocol_path": args.protocol,
        "runtime_path": args.runtime,
    }
    if args.phase.startswith("claim-"):
        claimed = args.phase.removeprefix("claim-")
        result = claim_stage(
            phase=claimed,
            **common,
            source_path=args.source,
            prediction_path=args.prediction,
            private_rna_path=args.private_rna,
            authorization_path=args.authorization,
            attempt_path=args.attempt,
        )
    elif args.phase == "source":
        result = run_source(
            **common,
            attempt_path=args.attempt or DEFAULT_SOURCE_ATTEMPT,
            output_path=args.output or DEFAULT_SOURCE,
            scratch=args.scratch,
        )
    elif args.phase == "predict":
        result = run_prediction(
            **common,
            source_path=args.source,
            attempt_path=args.attempt or DEFAULT_PREDICTION_ATTEMPT,
            output_path=args.output or DEFAULT_PREDICTION,
            private_rna_path=args.private_rna,
            scratch=args.scratch,
        )
    elif args.phase == "authorize-score":
        result = authorize_score(
            **common,
            template_path=args.authorization_template,
            source_path=args.source,
            prediction_path=args.prediction,
            private_rna_path=args.private_rna,
            attempt_path=args.attempt or DEFAULT_AUTHORIZATION_ATTEMPT,
            output_path=args.output or args.authorization,
        )
    else:
        result = run_score(
            **common,
            source_path=args.source,
            prediction_path=args.prediction,
            private_rna_path=args.private_rna,
            authorization_path=args.authorization,
            attempt_path=args.attempt or DEFAULT_SCORE_ATTEMPT,
            output_path=args.output or DEFAULT_SCORE,
            scratch=args.scratch,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
