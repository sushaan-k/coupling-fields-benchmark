"""Prospective held-batch validation for GSE202150 CITE-seq.

The protocol has three numeric stages. ``source`` may form paired RNA/ADT
tables only for acute subjects in IOF1 and IOF2. ``predict`` decodes HTO and RNA
counts for IOF3 and IOF4, freezes same-margin predictions, and serializes RNA
states only to a byte-bound private artifact outside the public repository.
The combined sparse support is visible, but ``score`` is the first stage
allowed to decode held ADT counts or form held joint tables. Every numeric
stage needs a separately published STARTED ledger and is one-shot, including
terminal refusals.
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
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Iterator, Literal

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mapreg.heterogeneity_adaptive_coupling import (  # noqa: E402
    CouplingEstimationRefusal,
    expected_binary_table_from_log_odds,
    signed_deviance_coordinate,
)
from mapreg.hierarchical_conditional_coupling import (  # noqa: E402
    fit_hierarchical_conditional_log_odds,
)
from mapreg.heterogeneity_adaptive_coupling import (  # noqa: E402
    fit_structured_conditional_log_odds,
)


ROOT = REPO_ROOT
DATA_DIR = ROOT / "data/confirmation/gse202150_citeseq"
DEFAULT_CANDIDATE = DATA_DIR / "candidate_designation_v1.json"
DEFAULT_MANIFEST = DATA_DIR / "source_manifest_v1.json"
DEFAULT_RUNTIME = DATA_DIR / "runtime_environment_v1.json"
DEFAULT_PREFLIGHT = ROOT / "results/development/gse202150_metadata_preflight_v1.json"
DEFAULT_SOURCE = ROOT / "results/development/gse202150_source_development_v1.json"
DEFAULT_PREDICTION = ROOT / "results/gse202150_held_predictions_v1.json"
DEFAULT_SCORE_AUTHORIZATION = DATA_DIR / "score_authorization_v1.json"
DEFAULT_SCORE = ROOT / "results/gse202150_confirmation_v1.json"
DEFAULT_PROTOCOL = (
    ROOT / "docs/GSE202150_ACUTE_INFECTION_HELD_BATCH_CONFIRMATION_PROTOCOL_2026-08-29.md"
)
DEFAULT_TEST = ROOT / "tests/test_gse202150_citeseq_confirmation.py"
ATTEMPT_PATHS = {
    "source": DATA_DIR / "source_attempt_v1.jsonl",
    "predict": DATA_DIR / "prediction_attempt_v1.jsonl",
    "score": DATA_DIR / "score_attempt_v1.jsonl",
}
EXECUTION_CLAIM_PATHS = {
    "source": DATA_DIR / "source_execution_consumed_v1.json",
    "predict": DATA_DIR / "prediction_execution_consumed_v1.json",
    "score": DATA_DIR / "score_execution_consumed_v1.json",
}
STAGE_OUTPUTS = {
    "source": DEFAULT_SOURCE,
    "predict": DEFAULT_PREDICTION,
    "score": DEFAULT_SCORE,
}

PUBLIC_ORIGIN = "https://github.com/sushaan-k/coupling-fields-benchmark.git"
PROTOCOL_TAG = "gse202150-citeseq-v1-protocol"
SOURCE_ATTEMPT_TAG = "gse202150-citeseq-v1-source-attempt"
SOURCE_RESULT_TAG = "gse202150-citeseq-v1-source"
PREDICTION_ATTEMPT_TAG = "gse202150-citeseq-v1-prediction-attempt"
PREDICTION_TAG = "gse202150-citeseq-v1-predictions"
SCORE_AUTHORIZATION_TAG = "gse202150-citeseq-v1-score-authorized"
SCORE_ATTEMPT_TAG = "gse202150-citeseq-v1-score-attempt"
RESULT_TAG = "gse202150-citeseq-v1-result"
ATTEMPT_TAGS = {
    "source": SOURCE_ATTEMPT_TAG,
    "predict": PREDICTION_ATTEMPT_TAG,
    "score": SCORE_ATTEMPT_TAG,
}

BATCHES = ("IOF1", "IOF2", "IOF3", "IOF4")
SOURCE_BATCHES = ("IOF1", "IOF2")
HELD_BATCHES = ("IOF3", "IOF4")
ROLE_BY_BATCH = {
    "IOF1": "source_calibration",
    "IOF2": "source_pilot",
    "IOF3": "held",
    "IOF4": "held",
}
EXPECTED_SUBJECTS_BY_BATCH = {"IOF1": 8, "IOF2": 8, "IOF3": 8, "IOF4": 9}
BRIDGE_SAMPLES = ("HD105_Control", "HD108_Control")

RNA_SYMBOLS = (
    "CD3E",
    "CD4",
    "CD8A",
    "MS4A1",
    "CD19",
    "CD27",
    "CD38",
    "CD14",
    "ITGAM",
    "ITGAX",
    "CD33",
    "NCAM1",
    "KLRB1",
)
ADT_TARGETS = (
    "CD3",
    "CD4",
    "CD8",
    "CD20",
    "CD19",
    "CD27",
    "CD38",
    "CD14",
    "CD11b",
    "CD11c",
    "CD33",
    "CD56",
    "CD161",
)
MARKER_LABELS = tuple(
    f"{rna}/{adt}" for rna, adt in zip(RNA_SYMBOLS, ADT_TARGETS)
)
PANEL_SIZE = len(RNA_SYMBOLS)
CELL_BUDGET = 384
MINIMUM_DETECTED_GENES = 200
MAXIMUM_MITOCHONDRIAL_FRACTION = 0.10
MAXIMUM_RNA_UMIS = 70_000
MINIMUM_INFORMATIVE_PAIRS = 100
CELL_SELECTION_SALT = "GSE202150-CELL-BUDGET-v1"
ADT_TIE_SALT = "GSE202150-ADT-MIDRANK-v1"
DESTROYED_LINK_SALT = "GSE202150-DESTROYED-LINK-v1"
BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 20260829

HETEROGENEITY_GRID = (0.1, 1.0, 10.0)
RIDGE_GRID = (0.01, 0.1)
GRAPH_GRID = (0.0, 0.1, 1.0)
TRANSPORT_GRID = (0.5, 0.75, 1.0, 1.25)
GRAPH_NEIGHBORS = 2
MAXIMUM_CONDITION_NUMBER = 1e12
CLASSICAL_ORDER = (
    "poisson_independence_deviance_residual",
    "common_effect_stratified_cmle",
    "pooled_saturated_poisson",
)
RAW_RESIDUAL_METHOD = "poisson_independence_deviance_residual_raw"
ALL_CLASSICAL_METHODS = (RAW_RESIDUAL_METHOD,) + CLASSICAL_ORDER

PROTOCOL_BINDINGS = (
    ".gitattributes",
    "experiments/confirm_gse202150_citeseq.py",
    "tests/test_gse202150_citeseq_confirmation.py",
    "docs/GSE202150_ACUTE_INFECTION_HELD_BATCH_CONFIRMATION_PROTOCOL_2026-08-29.md",
    "data/confirmation/gse202150_citeseq/candidate_designation_v1.json",
    "data/confirmation/gse202150_citeseq/source_manifest_v1.json",
    "data/confirmation/gse202150_citeseq/runtime_environment_v1.json",
    "results/development/gse202150_metadata_preflight_v1.json",
    "mapreg/coupling_fields.py",
    "mapreg/heterogeneity_adaptive_coupling.py",
    "mapreg/hierarchical_conditional_coupling.py",
    "mapreg/table_prediction.py",
    "requirements.txt",
    "pyproject.toml",
)


class ProtocolRefusal(RuntimeError):
    """A prespecified terminal support, integrity, or authorization refusal."""

    def __init__(self, code: str):
        if not code or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for character in code):
            raise ValueError("refusal codes must be nonempty uppercase identifiers")
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, order=True)
class PrimaryConfig:
    heterogeneity_penalty: float
    ridge_penalty: float
    graph_penalty: float
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


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _salted_hash(salt: str, *values: str) -> str:
    digest = hashlib.sha256(salt.encode())
    for value in values:
        digest.update(b"\0")
        digest.update(value.encode())
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = json.load(stream, object_pairs_hook=_strict_object)
    if not isinstance(value, dict):
        raise ValueError("top-level JSON value must be an object")
    return value


def _write_json_x(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_jsonl(path: Path, payload: dict[str, Any], *, create: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_APPEND
    flags |= os.O_CREAT | os.O_EXCL if create else 0
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "ab") as stream:
        stream.write(
            (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        )
        stream.flush()
        os.fsync(stream.fileno())


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _private_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        return resolved
    raise PermissionError("private state artifacts must be outside the public repository")


def _validate_public_payload(value: Any, key: str | None = None) -> None:
    forbidden_keys = {"states", "barcodes", "cell_ids", "selected_cell_ids"}
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key in forbidden_keys:
                raise PermissionError(f"public payload contains private key {child_key}")
            _validate_public_payload(child, child_key)
    elif isinstance(value, list):
        for child in value:
            _validate_public_payload(child, key)
    elif isinstance(value, str):
        is_hdf5_identifier = (
            key is not None
            and key.endswith(("datasets_opened", "datasets_not_opened"))
            and value.startswith("/matrix/")
        )
        if (value.startswith("/") or value.startswith("~")) and not is_hdf5_identifier:
            raise PermissionError(f"public payload contains a local path in {key}")


def _remote_tag_commit(tag: str) -> str:
    output = subprocess.run(
        ["git", "ls-remote", "--tags", PUBLIC_ORIGIN, f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    commits = [line.split()[0] for line in output if line.strip()]
    if not commits:
        raise PermissionError(f"required public tag is absent: {tag}")
    return commits[-1]


def _require_public_tag(tag: str, paths: Iterable[str]) -> str:
    commit = _remote_tag_commit(tag)
    for relative in paths:
        local = ROOT / relative
        if not local.is_file():
            raise PermissionError(f"bound path is absent: {relative}")
        published = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(published).hexdigest() != _sha256(local):
            raise PermissionError(f"public tag does not bind local bytes: {relative}")
    return commit


def _require_tagged_artifact(tag: str, path: Path) -> tuple[dict[str, Any], str]:
    commit = _require_public_tag(tag, (_relative(path),))
    return _read_json(path), commit


def _require_completed_stage_artifact(
    tag: str, path: Path, stage: str
) -> tuple[dict[str, Any], str]:
    attempt = ATTEMPT_PATHS[stage]
    execution_claim = EXECUTION_CLAIM_PATHS[stage]
    commit = _require_public_tag(
        tag,
        (_relative(path), _relative(attempt), _relative(execution_claim)),
    )
    payload = _read_json(path)
    consumed = _read_json(execution_claim)
    with attempt.open() as stream:
        rows = [
            json.loads(line, object_pairs_hook=_strict_object)
            for line in stream
            if line.strip()
        ]
    statuses = [row.get("status") for row in rows]
    if statuses != ["STARTED", "EXECUTING_CONSUMED", "FINISHED"]:
        raise PermissionError("completed stage ledger is not the frozen three-state sequence")
    if consumed != rows[1]:
        raise PermissionError("atomic execution claim differs from the stage ledger")
    protocol_commit = _require_public_tag(PROTOCOL_TAG, PROTOCOL_BINDINGS)
    runtime = _require_runtime_environment()
    if any(
        row.get("stage") != stage
        or row.get("protocol_commit") != protocol_commit
        or row.get("runtime_environment") != runtime
        for row in rows
    ):
        raise PermissionError("completed stage ledger differs from protocol or runtime")
    if (
        payload.get("protocol_commit") != protocol_commit
        or payload.get("runtime_environment") != runtime
        or rows[-1].get("terminal_status") != payload.get("status")
        or rows[-1].get("output_sha256") != _sha256(path)
    ):
        raise PermissionError("completed stage artifact differs from its ledger")
    return payload, commit


def _candidate() -> dict[str, Any]:
    value = _read_json(DEFAULT_CANDIDATE)
    if value.get("schema") != "gse202150-candidate-designation/1.0":
        raise PermissionError("candidate designation schema differs")
    samples = value.get("acute_samples")
    if not isinstance(samples, list) or len(samples) != 33:
        raise PermissionError("candidate designation must contain 33 acute samples")
    if len({row.get("subject") for row in samples}) != 33:
        raise PermissionError("candidate physical subjects are not unique")
    counts = {batch: sum(row.get("batch") == batch for row in samples) for batch in BATCHES}
    if counts != EXPECTED_SUBJECTS_BY_BATCH:
        raise PermissionError("candidate batch counts differ")
    for row in samples:
        batch = row.get("batch")
        if row.get("role") != ROLE_BY_BATCH.get(batch):
            raise PermissionError("candidate role differs from frozen batch role")
        tags = row.get("hto_tags")
        if not isinstance(tags, list) or len(tags) != 2 or len(set(tags)) != 2:
            raise PermissionError("candidate HTO pair is invalid")
    return value


def _manifest() -> dict[str, Any]:
    value = _read_json(DEFAULT_MANIFEST)
    if value.get("schema") != "gse202150-source-manifest/1.0":
        raise PermissionError("source manifest schema differs")
    files = value.get("h5_files")
    if not isinstance(files, list) or len(files) != 19:
        raise PermissionError("source manifest must bind 19 H5 files")
    if len({row.get("filename") for row in files}) != 19:
        raise PermissionError("source manifest H5 names are not unique")
    if any(
        not isinstance(row.get("sha256"), str) or len(row["sha256"]) != 64
        for row in files
    ):
        raise PermissionError("source manifest must bind every H5 SHA-256")
    return value


def _runtime_environment() -> dict[str, Any]:
    return {
        "python": {
            "implementation": sys.implementation.name,
            "version": ".".join(str(value) for value in sys.version_info[:3]),
        },
        "packages": {
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "h5py": h5py.__version__,
        },
        "hdf5": {
            "runtime_version": h5py.version.hdf5_version,
            "runtime_version_tuple": list(h5py.version.hdf5_version_tuple),
            "built_against_version_tuple": list(h5py.version.hdf5_built_version_tuple),
            "h5py_api_version": h5py.version.api_version,
        },
    }


def _require_runtime_environment() -> dict[str, Any]:
    specification = _read_json(DEFAULT_RUNTIME)
    if specification.get("schema") != "gse202150-runtime-environment/1.0":
        raise PermissionError("runtime specification schema differs")
    observed = _runtime_environment()
    if specification.get("required_runtime") != observed:
        raise PermissionError("runtime environment differs from the frozen specification")
    return observed


def _metadata_paths(metadata_root: Path) -> dict[str, Path]:
    manifest = _manifest()
    output: dict[str, Path] = {}
    for key, record in manifest["metadata"].items():
        path = metadata_root / record["filename"]
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise PermissionError(f"official metadata bytes differ: {key}")
        output[key] = path
    return output


def _h5_paths(h5_root: Path, batches: Iterable[str]) -> dict[str, Path]:
    requested = set(batches)
    output: dict[str, Path] = {}
    for row in _manifest()["h5_files"]:
        if row["batch"] not in requested:
            continue
        path = h5_root / row["filename"]
        if path.stat().st_size != row["bytes"] or _sha256(path) != row["sha256"]:
            raise PermissionError(f"H5 bytes differ: {row['filename']}")
        output[row["library"]] = path
    expected = 10 if requested == set(SOURCE_BATCHES) else 9 if requested == set(HELD_BATCHES) else None
    if expected is not None and len(output) != expected:
        raise PermissionError("H5 library set differs from frozen batch selection")
    return output


def _open_csv_gz(path: Path) -> Iterator[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise PermissionError("official metadata header is absent or duplicated")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise PermissionError("official metadata row has a shifted field count")
            yield {str(key): str(value).strip() for key, value in row.items()}


def _libraries(value: str) -> tuple[str, ...]:
    libraries = tuple(part.strip() for part in value.split(",") if part.strip())
    if not libraries or len(libraries) != len(set(libraries)):
        raise PermissionError("sample library list is empty or duplicated")
    return libraries


def _hto_matrix_name(official_name: str) -> str:
    prefix, separator, suffix = official_name.partition("_")
    if (
        not separator
        or suffix != "h-3-IH-A"
        or not prefix.startswith("HTO-")
        or not prefix[4:].isdigit()
    ):
        raise PermissionError("official HTO name cannot be mapped exactly to the H5 feature")
    return prefix


def _batch_from_library(library: str) -> str:
    pieces = library.split("_")
    if len(pieces) != 3 or pieces[0] != "ADRA03" or pieces[1] not in BATCHES:
        raise PermissionError("library name does not encode one frozen loading batch")
    return pieces[1]


def _official_metadata(metadata_root: Path) -> dict[str, Any]:
    paths = _metadata_paths(metadata_root)
    with paths["filelist"].open() as stream:
        filelist_rows = [
            line.rstrip("\n").split("\t")
            for line in stream
            if line.strip() and not line.startswith("#")
        ]
    if any(len(row) != 5 for row in filelist_rows):
        raise PermissionError("official file list has an unexpected field count")
    official_h5 = {
        row[1]: int(row[3])
        for row in filelist_rows
        if row[0] == "File" and row[4] == "H5"
    }
    frozen_h5 = {
        row["filename"]: row["bytes"] for row in _manifest()["h5_files"]
    }
    if official_h5 != frozen_h5:
        raise PermissionError("source manifest differs from the official H5 file list")
    additional_rows = list(_open_csv_gz(paths["sample_annotations"]))
    hashing_rows = list(_open_csv_gz(paths["sample_hashing_map"]))
    feature_rows = list(_open_csv_gz(paths["feature_reference"]))
    additional = {row[""]: row for row in additional_rows}
    hashing: dict[str, list[dict[str, str]]] = {}
    for row in hashing_rows:
        hashing.setdefault(row["Sample"], []).append(row)
    if len(additional) != len(additional_rows):
        raise PermissionError("official sample annotation contains duplicate names")
    if any(
        len(rows) != (4 if sample in BRIDGE_SAMPLES else 1)
        for sample, rows in hashing.items()
    ):
        raise PermissionError("official hashing-map multiplicity differs")
    if set(additional) != set(hashing):
        raise PermissionError("official sample annotation and HTO axes differ")

    libraries = sorted(
        {
            library
            for row in hashing_rows
            for library in _libraries(row["Libraries"])
        }
    )
    if len(libraries) != 19:
        raise PermissionError("official sample metadata must name 19 libraries")
    expected_libraries = sorted(row["library"] for row in _manifest()["h5_files"])
    if libraries != expected_libraries:
        raise PermissionError("official sample libraries differ from source manifest")

    assignments: dict[str, dict[frozenset[str], str]] = {batch: {} for batch in BATCHES}
    records: dict[str, dict[str, Any]] = {}
    for sample in sorted(hashing):
        hto_rows = hashing[sample]
        annotation = additional[sample]
        tag_sets = {
            tuple(
                sorted(
                    part.strip()
                    for part in row["Hashing BC"].split(";")
                    if part.strip()
                )
            )
            for row in hto_rows
        }
        if len(tag_sets) != 1:
            raise PermissionError("bridge HTO identity changes across batches")
        tags = next(iter(tag_sets))
        if len(tags) != 2 or len(set(tags)) != 2:
            raise PermissionError("official HTO assignment is not one unique pair")
        row_libraries = tuple(
            library
            for row in hto_rows
            for library in _libraries(row["Libraries"])
        )
        if len(row_libraries) != len(set(row_libraries)):
            raise PermissionError("sample library membership is duplicated")
        batches = sorted({_batch_from_library(library) for library in row_libraries})
        declared_loading = annotation["Loading Sample"]
        if sample in BRIDGE_SAMPLES:
            if batches != list(BATCHES) or declared_loading != "IOF1/IOF2/IOF3/IOF4":
                raise PermissionError("bridge donor batch declaration differs")
        elif len(batches) != 1 or declared_loading != batches[0]:
            raise PermissionError("sample loading batch differs between official tables")
        for batch in batches:
            pair = frozenset(_hto_matrix_name(tag) for tag in tags)
            if pair in assignments[batch]:
                raise PermissionError("HTO pair is duplicated within one loading batch")
            assignments[batch][pair] = sample
        records[sample] = {
            "sample": sample,
            "subject": annotation["Subject"],
            "pathogen": annotation["Pathogen"],
            "timepoint": annotation["Timepoint"],
            "batches": batches,
            "libraries": list(row_libraries),
            "hto_tags": list(tags),
        }

    acute = [row for row in records.values() if row["timepoint"] == "acute"]
    if len(acute) != 33 or len({row["subject"] for row in acute}) != 33:
        raise PermissionError("official metadata does not define 33 unique acute subjects")
    counts = {
        batch: sum(row["batches"] == [batch] for row in acute) for batch in BATCHES
    }
    if counts != EXPECTED_SUBJECTS_BY_BATCH:
        raise PermissionError("official acute-subject batch counts differ")
    designated = {
        (row["sample"], row["subject"], row["pathogen"], row["batches"][0], tuple(row["hto_tags"]))
        for row in acute
    }
    frozen = {
        (row["sample"], row["subject"], row["pathogen"], row["batch"], tuple(row["hto_tags"]))
        for row in _candidate()["acute_samples"]
    }
    if designated != frozen:
        raise PermissionError("candidate designation differs from official metadata")
    return {
        "records": records,
        "assignments": assignments,
        "feature_rows": feature_rows,
        "libraries": libraries,
    }


def _decode_strings(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


def _feature_schema(path: Path) -> dict[str, list[str]]:
    with h5py.File(path, "r") as handle:
        group = handle["matrix/features"]
        required = ("id", "name", "feature_type")
        if any(key not in group for key in required):
            raise PermissionError("10x feature schema is incomplete")
        schema = {key: _decode_strings(group[key][...]) for key in required}
    if len({len(values) for values in schema.values()}) != 1:
        raise PermissionError("10x feature schema axes differ")
    return schema


def _schema_sha256(schema: dict[str, list[str]]) -> str:
    return _canonical_json_sha256(schema)


def _resolve_feature_panel(schema: dict[str, list[str]]) -> dict[str, Any]:
    names = schema["name"]
    types = schema["feature_type"]

    def unique(name: str, feature_type: str) -> int:
        found = [
            index
            for index, (observed_name, observed_type) in enumerate(zip(names, types))
            if observed_name == name and observed_type == feature_type
        ]
        if len(found) != 1:
            raise PermissionError(f"feature mapping is absent or nonunique: {name}/{feature_type}")
        return found[0]

    rna = [unique(symbol, "Gene Expression") for symbol in RNA_SYMBOLS]
    custom = [unique(target, "Custom") for target in ADT_TARGETS]
    hto_features = [
        (name, index)
        for index, (name, feature_type) in enumerate(zip(names, types))
        if feature_type == "Custom" and name.startswith("HTO-")
    ]
    if len({name for name, _ in hto_features}) != len(hto_features):
        raise PermissionError("HTO feature names are duplicated")
    hto = dict(hto_features)
    if len(hto) not in {7, 10}:
        raise PermissionError("HTO feature schema must contain seven or ten unique tags")
    if any(name == "CD45" and feature_type == "Custom" for name, feature_type in zip(names, types)):
        raise PermissionError("unexpected total-CD45 feature changes the frozen 13-marker panel")
    if len(set(rna)) != PANEL_SIZE or len(set(custom)) != PANEL_SIZE:
        raise PermissionError("frozen cognate mappings are not one-to-one")
    return {
        "rna": rna,
        "adt": custom,
        "hto": hto,
        "gene_expression": [index for index, value in enumerate(types) if value == "Gene Expression"],
        "mitochondrial": [
            index
            for index, (name, value) in enumerate(zip(names, types))
            if value == "Gene Expression" and name.startswith("MT-")
        ],
    }


def run_preflight(
    metadata_root: Path,
    h5_root: Path,
    output_path: Path = DEFAULT_PREFLIGHT,
) -> dict[str, Any]:
    if output_path.resolve() != DEFAULT_PREFLIGHT.resolve():
        raise PermissionError("preflight output path is fixed")
    runtime = _require_runtime_environment()
    official = _official_metadata(metadata_root)
    library_paths = _h5_paths(h5_root, BATCHES)
    schema_hashes: dict[str, str] = {}
    hto_counts: dict[str, int] = {}
    for library in official["libraries"]:
        schema = _feature_schema(library_paths[library])
        resolved = _resolve_feature_panel(schema)
        schema_hashes[library] = _schema_sha256(schema)
        hto_counts[library] = len(resolved["hto"])
        batch = _batch_from_library(library)
        expected = 10 if batch == "IOF4" else 7
        if hto_counts[library] != expected:
            raise ProtocolRefusal("HTO_FEATURE_COUNT_DIFFERS_BY_BATCH")
        official_tags = {
            tag for pair in official["assignments"][batch] for tag in pair
        }
        if set(resolved["hto"]) != official_tags:
            raise ProtocolRefusal("HTO_FEATURE_SET_DIFFERS_FROM_OFFICIAL_SAMPLE_MAP")
    early_hashes = {
        schema_hashes[library]
        for library in official["libraries"]
        if _batch_from_library(library) in ("IOF1", "IOF2", "IOF3")
    }
    late_hashes = {
        schema_hashes[library]
        for library in official["libraries"]
        if _batch_from_library(library) == "IOF4"
    }
    if len(early_hashes) != 1 or len(late_hashes) != 1 or early_hashes == late_hashes:
        raise ProtocolRefusal("FEATURE_SCHEMAS_DO_NOT_HAVE_EXPECTED_V1_V3_SPLIT")
    payload = {
        "schema": "gse202150-metadata-preflight/1.0",
        "status": "PREFLIGHT_PASS_NO_ASSAY_MATRIX_VALUES_OPENED",
        "created_at_utc": _timestamp(),
        "official_metadata_sha256": {
            key: _sha256(path) for key, path in _metadata_paths(metadata_root).items()
        },
        "h5_file_sha256": {
            row["library"]: row["sha256"] for row in _manifest()["h5_files"]
        },
        "runtime_environment": runtime,
        "libraries": official["libraries"],
        "feature_schema_sha256": schema_hashes,
        "hto_feature_counts": hto_counts,
        "acute_subjects": 33,
        "batch_subject_counts": EXPECTED_SUBJECTS_BY_BATCH,
        "source_subjects": 16,
        "held_subjects": 17,
        "bridge_samples_excluded_from_inference": list(BRIDGE_SAMPLES),
        "panel": list(MARKER_LABELS),
        "total_cd45_substituted": False,
        "cd45ra_excluded": True,
        "preflight_runner_datasets_opened": [
            "/matrix/features/id",
            "/matrix/features/name",
            "/matrix/features/feature_type",
        ],
        "preflight_runner_datasets_not_opened": [
            "/matrix/barcodes",
            "/matrix/data",
            "/matrix/indices",
            "/matrix/indptr",
            "/matrix/shape",
        ],
        "whole_file_sha256_computed_without_hdf5_dataset_decoding": True,
        "assay_numeric_values_read": 0,
    }
    _validate_public_payload(payload)
    _write_json_x(output_path, payload)
    return payload


def _integer_values(values: np.ndarray, code: str) -> np.ndarray:
    observed = np.asarray(values)
    if not np.isfinite(observed).all() or np.any(observed < 0.0):
        raise ProtocolRefusal(code)
    rounded = np.rint(observed)
    if not np.array_equal(observed, rounded):
        raise ProtocolRefusal(code)
    return rounded.astype(np.int64)


def _sparse_structure(handle: h5py.File) -> tuple[list[str], np.ndarray, np.ndarray, int]:
    matrix = handle["matrix"]
    barcodes = _decode_strings(matrix["barcodes"][...])
    indptr = np.asarray(matrix["indptr"][...], dtype=np.int64)
    indices = np.asarray(matrix["indices"][...], dtype=np.int64)
    shape = np.asarray(matrix["shape"][...], dtype=np.int64)
    if shape.shape != (2,) or tuple(shape) != (len(matrix["features/id"]), len(barcodes)):
        raise ProtocolRefusal("SPARSE_SHAPE_DIFFERS")
    if len(indptr) != len(barcodes) + 1 or indptr[0] != 0 or np.any(np.diff(indptr) < 0):
        raise ProtocolRefusal("SPARSE_POINTERS_INVALID")
    if indptr[-1] != len(indices) or indptr[-1] != len(matrix["data"]):
        raise ProtocolRefusal("SPARSE_TERMINAL_POINTER_INVALID")
    if np.any(indices < 0) or np.any(indices >= shape[0]):
        raise ProtocolRefusal("SPARSE_FEATURE_INDEX_INVALID")
    for column in range(len(barcodes)):
        start, stop = int(indptr[column]), int(indptr[column + 1])
        if np.any(np.diff(indices[start:stop]) <= 0):
            raise ProtocolRefusal("SPARSE_COLUMN_INDICES_NOT_STRICTLY_INCREASING")
    if len(barcodes) != len(set(barcodes)):
        raise ProtocolRefusal("LIBRARY_BARCODES_NOT_UNIQUE")
    return barcodes, indptr, indices, int(shape[0])


def _read_sparse_positions(
    dataset: h5py.Dataset,
    positions: np.ndarray,
    *,
    code: str,
) -> np.ndarray:
    selected = np.asarray(positions, dtype=np.int64)
    if selected.ndim != 1 or (len(selected) and np.any(np.diff(selected) <= 0)):
        raise ValueError("sparse data positions must be strictly increasing")
    if not len(selected):
        return np.empty(0, dtype=np.int64)
    return _integer_values(dataset[selected], code)


def _column_values(
    data: h5py.Dataset,
    indices: np.ndarray,
    start: int,
    stop: int,
    allowed_rows: set[int],
    *,
    code: str,
) -> tuple[dict[int, int], int]:
    column_rows = indices[start:stop]
    local = np.flatnonzero(np.fromiter((int(row) in allowed_rows for row in column_rows), bool))
    positions = local.astype(np.int64) + start
    values = _read_sparse_positions(data, positions, code=code)
    return {int(column_rows[offset]): int(value) for offset, value in zip(local, values)}, len(values)


def _demultiplex(
    hto_values: dict[int, int],
    hto_rows: dict[str, int],
    assignments: dict[frozenset[str], str],
) -> str | None:
    ranked = sorted(
        ((int(hto_values.get(row, 0)), tag) for tag, row in hto_rows.items()),
        key=lambda item: (-item[0], item[1]),
    )
    if len(ranked) < 2 or ranked[1][0] <= 0:
        return None
    third = ranked[2][0] if len(ranked) > 2 else -1
    if ranked[0][0] == ranked[1][0] or ranked[1][0] == third:
        return None
    return assignments.get(frozenset((ranked[0][1], ranked[1][1])))


def _scan_library(
    path: Path,
    library: str,
    official: dict[str, Any],
    *,
    modalities: frozenset[Literal["rna", "adt"]],
    selected_ids: set[str] | None = None,
    scope: Literal["acute", "bridges"] = "acute",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not modalities or not modalities <= {"rna", "adt"}:
        raise ValueError("one or both response modalities must be requested")
    batch = _batch_from_library(library)
    with h5py.File(path, "r") as handle:
        schema = {
            key: _decode_strings(handle[f"matrix/features/{key}"][...])
            for key in ("id", "name", "feature_type")
        }
        resolved = _resolve_feature_panel(schema)
        barcodes, indptr, indices, _ = _sparse_structure(handle)
        data = handle["matrix/data"]
        gene_rows = set(resolved["gene_expression"])
        mitochondrial = set(resolved["mitochondrial"])
        adt_panel_rows = set(resolved["adt"])
        hto_rows = set(resolved["hto"].values())
        records: list[dict[str, Any]] = []
        audit = {
            "library": library,
            "batch": batch,
            "cells_seen": len(barcodes),
            "cells_unassigned_or_ambiguous_by_hto": 0,
            "cells_assigned_by_official_hto_pair": 0,
            "cells_matching_requested_scope": 0,
            "cells_failing_rna_qc": 0,
            "cells_returned": 0,
            "hto_values_read": 0,
            "rna_values_read": 0,
            "adt_values_read": 0,
            "co_resident_adt_values_decoded": False,
        }
        for column, barcode in enumerate(barcodes):
            composite = f"{library}|{barcode}"
            start, stop = int(indptr[column]), int(indptr[column + 1])
            hto, count = _column_values(
                data,
                indices,
                start,
                stop,
                hto_rows,
                code="HTO_COUNTS_INVALID",
            )
            audit["hto_values_read"] += count
            sample = _demultiplex(
                hto, resolved["hto"], official["assignments"][batch]
            )
            if sample is None:
                audit["cells_unassigned_or_ambiguous_by_hto"] += 1
                continue
            audit["cells_assigned_by_official_hto_pair"] += 1
            metadata = official["records"][sample]
            if scope == "acute":
                if metadata["timepoint"] != "acute" or sample in BRIDGE_SAMPLES:
                    continue
                subject = metadata["subject"]
            elif scope == "bridges":
                if sample not in BRIDGE_SAMPLES:
                    continue
                subject = f"{metadata['subject']}@{batch}"
            else:
                raise ValueError("unknown sample scope")
            audit["cells_matching_requested_scope"] += 1
            if selected_ids is not None and composite not in selected_ids:
                continue
            record: dict[str, Any] = {
                "cell_id": composite,
                "sample": sample,
                "subject": subject,
                "pathogen": metadata["pathogen"],
                "batch": batch,
            }
            if "rna" in modalities:
                rna, count = _column_values(
                    data,
                    indices,
                    start,
                    stop,
                    gene_rows,
                    code="RNA_COUNTS_INVALID",
                )
                audit["rna_values_read"] += count
                detected = len(rna)
                total = sum(rna.values())
                mitochondrial_total = sum(
                    value for row, value in rna.items() if row in mitochondrial
                )
                fraction = mitochondrial_total / total if total else 1.0
                if (
                    detected < MINIMUM_DETECTED_GENES
                    or fraction > MAXIMUM_MITOCHONDRIAL_FRACTION
                    or total > MAXIMUM_RNA_UMIS
                ):
                    audit["cells_failing_rna_qc"] += 1
                    continue
                record["rna_state"] = np.asarray(
                    [int(rna.get(row, 0) > 0) for row in resolved["rna"]],
                    dtype=np.int8,
                )
            if "adt" in modalities:
                adt, count = _column_values(
                    data,
                    indices,
                    start,
                    stop,
                    adt_panel_rows,
                    code="ADT_COUNTS_INVALID",
                )
                audit["adt_values_read"] += count
                audit["co_resident_adt_values_decoded"] = count > 0 or audit[
                    "co_resident_adt_values_decoded"
                ]
                record["adt_counts"] = np.asarray(
                    [adt.get(row, 0) for row in resolved["adt"]], dtype=np.int64
                )
            records.append(record)
            audit["cells_returned"] += 1
    if "adt" not in modalities and (
        audit["adt_values_read"] != 0 or audit["co_resident_adt_values_decoded"]
    ):
        raise AssertionError("RNA firewall decoded an ADT data position")
    return records, audit


def _axis_sha256(subject: str, cells: Iterable[str]) -> str:
    digest = hashlib.sha256(CELL_SELECTION_SALT.encode())
    digest.update(b"\0")
    digest.update(subject.encode())
    for cell in cells:
        encoded = cell.encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _select_subject_cells(
    records: list[dict[str, Any]], expected_subjects: list[str]
) -> dict[str, list[dict[str, Any]]]:
    grouped = {subject: [] for subject in expected_subjects}
    seen: set[str] = set()
    for record in records:
        cell = record["cell_id"]
        if cell in seen:
            raise ProtocolRefusal("COMPOSITE_CELL_IDENTIFIER_DUPLICATED")
        seen.add(cell)
        if record["subject"] in grouped:
            grouped[record["subject"]].append(record)
    selected: dict[str, list[dict[str, Any]]] = {}
    for subject in expected_subjects:
        values = grouped[subject]
        if len(values) < CELL_BUDGET:
            raise ProtocolRefusal("ACUTE_SUBJECT_HAS_FEWER_THAN_384_ELIGIBLE_CELLS")
        values.sort(
            key=lambda row: (
                _salted_hash(CELL_SELECTION_SALT, subject, row["cell_id"]),
                row["cell_id"],
            )
        )
        selected[subject] = values[:CELL_BUDGET]
    return selected


def _adt_states(counts: np.ndarray, cells: list[str], subject: str) -> np.ndarray:
    values = np.asarray(counts, dtype=np.int64)
    if values.shape != (CELL_BUDGET, PANEL_SIZE) or np.any(values < 0):
        raise ValueError("ADT matrix shape or values differ")
    output = np.zeros_like(values, dtype=np.int8)
    for marker in range(PANEL_SIZE):
        order = sorted(
            range(CELL_BUDGET),
            key=lambda index: (
                -int(values[index, marker]),
                _salted_hash(ADT_TIE_SALT, subject, ADT_TARGETS[marker], cells[index]),
                cells[index],
            ),
        )
        output[order[: CELL_BUDGET // 2], marker] = 1
    if not np.all(output.sum(axis=0) == CELL_BUDGET // 2):
        raise AssertionError("ADT midrank did not produce an exact half margin")
    return output


def _destroyed_adt(states: np.ndarray, cells: list[str], subject: str) -> np.ndarray:
    order = np.asarray(
        sorted(
            range(CELL_BUDGET),
            key=lambda index: (
                _salted_hash(DESTROYED_LINK_SALT, subject, cells[index]),
                cells[index],
            ),
        ),
        dtype=np.int64,
    )
    source = np.roll(order, 1)
    output = np.empty_like(states)
    output[order] = states[source]
    if not np.array_equal(output.sum(axis=0), states.sum(axis=0)):
        raise AssertionError("destroyed-link rotation changed ADT margins")
    return output


def _binary_tables(rna: np.ndarray, adt: np.ndarray) -> np.ndarray:
    first = np.asarray(rna, dtype=np.int8)
    second = np.asarray(adt, dtype=np.int8)
    if first.shape != (CELL_BUDGET, PANEL_SIZE) or second.shape != first.shape:
        raise ValueError("state matrices differ from the frozen dimensions")
    output = np.empty((PANEL_SIZE, PANEL_SIZE, 2, 2), dtype=np.int64)
    for row in range(PANEL_SIZE):
        for column in range(PANEL_SIZE):
            output[row, column] = np.bincount(
                2 * first[:, row] + second[:, column], minlength=4
            ).reshape(2, 2)
    return output


def _subject_records(
    selected: dict[str, list[dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for subject, rows in selected.items():
        cells = [row["cell_id"] for row in rows]
        rna = np.asarray([row["rna_state"] for row in rows], dtype=np.int8)
        adt_counts = np.asarray([row["adt_counts"] for row in rows], dtype=np.int64)
        adt = _adt_states(adt_counts, cells, subject)
        output[subject] = {
            "subject": subject,
            "sample": rows[0]["sample"],
            "pathogen": rows[0]["pathogen"],
            "batch": rows[0]["batch"],
            "cells": cells,
            "axis_sha256": _axis_sha256(subject, cells),
            "rna_states": rna,
            "adt_states": adt,
            "tables": _binary_tables(rna, adt),
            "destroyed_tables": _binary_tables(rna, _destroyed_adt(adt, cells, subject)),
            "rna_profile": rna.mean(axis=0),
            "adt_profile": np.log1p(adt_counts).mean(axis=0),
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


def _margins(tables: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(tables)
    return values.sum(axis=-1), values.sum(axis=-2)


def _knn_incidence(profiles: np.ndarray) -> np.ndarray:
    values = np.asarray(profiles, dtype=float).T
    if values.shape[0] != PANEL_SIZE or values.shape[1] < 3:
        raise ValueError("marker profile dimensions differ")
    scale = values.std(axis=1, ddof=1)
    if np.any(scale <= 0.0) or not np.isfinite(scale).all():
        raise CouplingEstimationRefusal("source marker profile has zero variance")
    standardized = (values - values.mean(axis=1, keepdims=True)) / scale[:, None]
    edges: set[tuple[int, int]] = set()
    for marker in range(PANEL_SIZE):
        candidates = np.asarray([value for value in range(PANEL_SIZE) if value != marker])
        distances = np.linalg.norm(
            standardized[candidates] - standardized[marker][None, :], axis=1
        )
        order = candidates[np.lexsort((candidates, distances))]
        edges.update(
            tuple(sorted((marker, int(neighbor))))
            for neighbor in order[:GRAPH_NEIGHBORS]
        )
    incidence = np.zeros((PANEL_SIZE, len(edges)), dtype=float)
    for column, (left, right) in enumerate(sorted(edges)):
        incidence[left, column] = 1.0
        incidence[right, column] = 1.0
    return incidence


def _fit_primary(
    tables: np.ndarray,
    rna_profiles: np.ndarray,
    adt_profiles: np.ndarray,
    config: PrimaryConfig,
) -> dict[str, Any]:
    if config.graph_penalty == 0.0:
        first = second = np.eye(PANEL_SIZE, dtype=float)
    else:
        first = _knn_incidence(rna_profiles)
        second = _knn_incidence(adt_profiles)
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
        "family": "exact_fixed_margin_hierarchical_conditional_coupling",
        "configuration": asdict(config),
        "population_log_odds": fit.population_log_odds,
        "support_count": fit.support_count,
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


def _fit_residual(tables: np.ndarray) -> dict[str, Any]:
    values = np.asarray(tables, dtype=np.int64)
    support = _informative(values)
    if np.any(support.sum(axis=0) < 2):
        raise CouplingEstimationRefusal("deviance residual lacks source support")
    coordinates = np.full(support.shape, np.nan, dtype=float)
    for donor, row, column in np.argwhere(support):
        coordinates[donor, row, column] = (
            signed_deviance_coordinate(values[donor, row, column])
            / math.sqrt(CELL_BUDGET)
        )
    pooled = np.nanmean(coordinates, axis=0)
    if not np.isfinite(pooled).all():
        raise CouplingEstimationRefusal("deviance residual pool is nonfinite")
    return {
        "family": "row_plus_column_poisson_independence_signed_deviance_residual",
        "transport_multiplier": 1.0,
        "pooled_coordinate": pooled,
        "support_count": support.sum(axis=0),
        "source_donor_weighting": "equal among informative donors for each ordered pair",
    }


def _fit_common_effect(tables: np.ndarray) -> dict[str, Any]:
    identity = np.eye(PANEL_SIZE, dtype=float)
    fit = fit_structured_conditional_log_odds(
        np.asarray(tables, dtype=np.int64),
        identity,
        identity,
        initial_log_odds=np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=float),
        ridge_penalty=0.0,
        graph_penalty=0.0,
        minimum_informative_donors=2,
        maximum_condition_number=MAXIMUM_CONDITION_NUMBER,
        tolerance=1e-9,
    )
    return {
        "family": "common_effect_stratified_conditional_maximum_likelihood",
        "transport_multiplier": 1.0,
        "population_log_odds": fit.log_odds,
        "support_count": fit.support_count,
        "fit_certificate": {
            "gradient_norm": fit.gradient_norm,
            "condition_number": fit.condition_number,
            "iterations": fit.iterations,
        },
    }


def _fit_pooled_poisson(tables: np.ndarray) -> dict[str, Any]:
    pooled = np.asarray(tables, dtype=np.int64).sum(axis=0)
    if np.any(pooled <= 0):
        raise CouplingEstimationRefusal("pooled saturated Poisson interaction has a zero cell")
    log_odds = np.log(pooled[..., 0, 0]) + np.log(pooled[..., 1, 1])
    log_odds -= np.log(pooled[..., 0, 1]) + np.log(pooled[..., 1, 0])
    if not np.isfinite(log_odds).all():
        raise CouplingEstimationRefusal("pooled saturated Poisson interaction is nonfinite")
    return {
        "family": "pooled_saturated_poisson_log_linear_interaction",
        "transport_multiplier": 1.0,
        "population_log_odds": log_odds,
        "pooled_tables_sha256": _array_sha256(pooled),
        "source_donor_weighting": "pooled cell counts; reported separately from donor-equal scoring",
    }


def _fit_classical(method: str, tables: np.ndarray) -> dict[str, Any]:
    if method == "poisson_independence_deviance_residual":
        return _fit_residual(tables)
    if method == "common_effect_stratified_cmle":
        return _fit_common_effect(tables)
    if method == "pooled_saturated_poisson":
        return _fit_pooled_poisson(tables)
    raise ValueError(f"unknown classical method: {method}")


def _fractional_signed_deviance(table: np.ndarray) -> float:
    values = np.asarray(table, dtype=float)
    expected = np.outer(values.sum(axis=1), values.sum(axis=0)) / values.sum()
    positive = values > 0.0
    deviance = 2.0 * float(
        np.sum(values[positive] * np.log(values[positive] / expected[positive]))
    )
    determinant = float(values[0, 0] * values[1, 1] - values[0, 1] * values[1, 0])
    return math.copysign(math.sqrt(max(deviance, 0.0)), determinant)


def _residual_table(
    coordinate: float, row_margin: np.ndarray, column_margin: np.ndarray
) -> np.ndarray:
    rows = np.asarray(row_margin, dtype=float)
    columns = np.asarray(column_margin, dtype=float)
    total = float(rows.sum())
    lower = float(max(0.0, rows[0] + columns[0] - total))
    upper = float(min(rows[0], columns[0]))

    def table_at(value: float) -> np.ndarray:
        return np.asarray(
            [
                [value, rows[0] - value],
                [columns[0] - value, rows[1] - columns[0] + value],
            ],
            dtype=float,
        )

    if upper <= lower:
        return table_at(lower)
    epsilon = min(1e-8, 0.25 * (upper - lower))
    left, right = lower + epsilon, upper - epsilon
    target = min(
        max(float(coordinate), _fractional_signed_deviance(table_at(left))),
        _fractional_signed_deviance(table_at(right)),
    )
    for _ in range(96):
        midpoint = 0.5 * (left + right)
        if _fractional_signed_deviance(table_at(midpoint)) < target:
            left = midpoint
        else:
            right = midpoint
    return table_at(0.5 * (left + right))


def _predict_model(
    model: dict[str, Any], rows: np.ndarray, columns: np.ndarray
) -> np.ndarray:
    row_values = np.asarray(rows, dtype=np.int64)
    column_values = np.asarray(columns, dtype=np.int64)
    output = np.empty((PANEL_SIZE, PANEL_SIZE, 2, 2), dtype=float)
    alpha = float(model.get("transport_multiplier", model.get("configuration", {}).get("transport_multiplier", 1.0)))
    if "pooled_coordinate" in model:
        field = np.asarray(model["pooled_coordinate"], dtype=float)
        for index in np.ndindex((PANEL_SIZE, PANEL_SIZE)):
            output[index] = _residual_table(
                alpha * float(field[index]) * math.sqrt(CELL_BUDGET),
                row_values[index],
                column_values[index],
            )
    else:
        field = np.asarray(model["population_log_odds"], dtype=float)
        for index in np.ndindex((PANEL_SIZE, PANEL_SIZE)):
            output[index] = expected_binary_table_from_log_odds(
                alpha * float(field[index]), row_values[index], column_values[index]
            )
    return output


def _donor_loss(observed: np.ndarray, predicted: np.ndarray) -> float:
    truth = np.asarray(observed, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    support = _informative(truth)
    if np.count_nonzero(support) < MINIMUM_INFORMATIVE_PAIRS:
        raise CouplingEstimationRefusal("subject has fewer than 100 informative ordered pairs")
    truth = truth[support]
    estimate = estimate[support]
    if not np.allclose(truth.sum(axis=-1), estimate.sum(axis=-1)) or not np.allclose(
        truth.sum(axis=-2), estimate.sum(axis=-2)
    ):
        raise FloatingPointError("prediction changed recipient margins")
    positive = truth > 0.0
    if np.any(estimate[positive] <= 0.0) or not np.isfinite(estimate).all():
        raise FloatingPointError("prediction assigns invalid mass")
    terms = np.zeros_like(truth)
    terms[positive] = truth[positive] * np.log(truth[positive] / estimate[positive])
    return float((2.0 * terms.sum(axis=(-2, -1)) / CELL_BUDGET).mean())


def _losses(
    records: dict[str, dict[str, Any]], subjects: list[str], model: dict[str, Any]
) -> np.ndarray:
    output = np.empty(len(subjects), dtype=float)
    for index, subject in enumerate(subjects):
        truth = records[subject]["tables"]
        rows, columns = _margins(truth)
        output[index] = _donor_loss(truth, _predict_model(model, rows, columns))
    return output


def _json_model(model: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in model.items():
        output[key] = value.tolist() if isinstance(value, np.ndarray) else value
    return output


def _primary_configs() -> list[PrimaryConfig]:
    return [
        PrimaryConfig(heterogeneity, ridge, graph, transport)
        for heterogeneity, ridge, graph, transport in product(
            HETEROGENEITY_GRID, RIDGE_GRID, GRAPH_GRID, TRANSPORT_GRID
        )
    ]


def _select_primary_loocv(
    records: dict[str, dict[str, Any]], calibration: list[str]
) -> dict[str, Any]:
    candidates = _primary_configs()
    losses = {config: np.full(len(calibration), np.nan) for config in candidates}
    refusals: list[dict[str, Any]] = []
    for fold, held in enumerate(calibration):
        training = [subject for subject in calibration if subject != held]
        tables = np.asarray([records[subject]["tables"] for subject in training])
        rna_profiles = np.asarray([records[subject]["rna_profile"] for subject in training])
        adt_profiles = np.asarray([records[subject]["adt_profile"] for subject in training])
        truth = records[held]["tables"]
        rows, columns = _margins(truth)
        structural: dict[tuple[float, float, float], dict[str, Any] | Exception] = {}
        for config in candidates:
            key = (
                config.heterogeneity_penalty,
                config.ridge_penalty,
                config.graph_penalty,
            )
            if key not in structural:
                try:
                    structural[key] = _fit_primary(
                        tables, rna_profiles, adt_profiles, config
                    )
                except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
                    structural[key] = error
            fit = structural[key]
            if isinstance(fit, Exception):
                refusals.append(
                    {"held_subject": held, "configuration": asdict(config), "reason": str(fit)}
                )
                continue
            candidate_model = dict(fit)
            candidate_model["configuration"] = asdict(config)
            losses[config][fold] = _donor_loss(
                truth, _predict_model(candidate_model, rows, columns)
            )
    complete = [config for config, values in losses.items() if np.isfinite(values).all()]
    if not complete:
        raise ProtocolRefusal("PRIMARY_CALIBRATION_LOOCV_HAS_NO_COMPLETE_CONFIGURATION")
    selected = min(complete, key=lambda config: (float(losses[config].mean()), config))
    return {
        "selected": selected,
        "selected_fold_losses": losses[selected],
        "complete_candidates": len(complete),
        "refusals": refusals,
    }


def _select_classical_transport_loocv(
    records: dict[str, dict[str, Any]], calibration: list[str], method: str
) -> dict[str, Any]:
    losses = {
        multiplier: np.full(len(calibration), np.nan, dtype=float)
        for multiplier in TRANSPORT_GRID
    }
    for fold, held in enumerate(calibration):
        training = [subject for subject in calibration if subject != held]
        tables = np.asarray([records[subject]["tables"] for subject in training])
        model = _fit_classical(method, tables)
        truth = records[held]["tables"]
        rows, columns = _margins(truth)
        for multiplier in TRANSPORT_GRID:
            candidate = dict(model)
            candidate["transport_multiplier"] = multiplier
            losses[multiplier][fold] = _donor_loss(
                truth, _predict_model(candidate, rows, columns)
            )
    selected = min(
        TRANSPORT_GRID,
        key=lambda multiplier: (float(losses[multiplier].mean()), multiplier),
    )
    return {
        "selected_transport_multiplier": selected,
        "selected_fold_losses": losses[selected],
    }


def _designated_rows(batches: Iterable[str]) -> list[dict[str, Any]]:
    requested = set(batches)
    rows = [row for row in _candidate()["acute_samples"] if row["batch"] in requested]
    return sorted(rows, key=lambda row: (BATCHES.index(row["batch"]), row["subject"]))


def _lock_classical(pilot_losses: dict[str, list[float]]) -> str:
    estimable = [method for method in ALL_CLASSICAL_METHODS if method in pilot_losses]
    if not estimable:
        raise ProtocolRefusal("NO_CLASSICAL_COMPARATOR_ESTIMABLE")
    return min(
        estimable,
        key=lambda method: (
            float(np.mean(pilot_losses[method])),
            ALL_CLASSICAL_METHODS.index(method),
        ),
    )


def _load_inference_cells(
    metadata_root: Path,
    h5_root: Path,
    batches: tuple[str, ...],
    *,
    modalities: frozenset[Literal["rna", "adt"]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, str]]:
    official = _official_metadata(metadata_root)
    paths = _h5_paths(h5_root, batches)
    file_hashes = {library: _sha256(path) for library, path in sorted(paths.items())}
    all_records: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for library, path in sorted(paths.items()):
        records, audit = _scan_library(
            path, library, official, modalities=modalities
        )
        all_records.extend(records)
        audits.append(audit)
    expected = [row["subject"] for row in _designated_rows(batches)]
    selected = _select_subject_cells(all_records, expected)
    return selected, audits, file_hashes


def _source_body(metadata_root: Path, h5_root: Path) -> dict[str, Any]:
    selected, audits, file_hashes = _load_inference_cells(
        metadata_root,
        h5_root,
        SOURCE_BATCHES,
        modalities=frozenset(("rna", "adt")),
    )
    records = _subject_records(selected)
    calibration = [row["subject"] for row in _designated_rows(("IOF1",))]
    pilot = [row["subject"] for row in _designated_rows(("IOF2",))]
    all_source = calibration + pilot
    if set(records) != set(all_source) or len(calibration) != 8 or len(pilot) != 8:
        raise ProtocolRefusal("SOURCE_SUBJECT_AXIS_DIFFERS")

    selection = _select_primary_loocv(records, calibration)
    selected_config: PrimaryConfig = selection["selected"]
    calibration_tables = np.asarray([records[subject]["tables"] for subject in calibration])
    all_tables = np.asarray([records[subject]["tables"] for subject in all_source])
    calibration_rna = np.asarray([records[subject]["rna_profile"] for subject in calibration])
    calibration_adt = np.asarray([records[subject]["adt_profile"] for subject in calibration])
    all_rna = np.asarray([records[subject]["rna_profile"] for subject in all_source])
    all_adt = np.asarray([records[subject]["adt_profile"] for subject in all_source])

    primary_calibration = _fit_primary(
        calibration_tables, calibration_rna, calibration_adt, selected_config
    )
    primary_full = _fit_primary(all_tables, all_rna, all_adt, selected_config)
    destroyed_tables = np.asarray(
        [records[subject]["destroyed_tables"] for subject in all_source]
    )
    destroyed_full = _fit_primary(
        destroyed_tables, all_rna, all_adt, selected_config
    )

    classical_calibration: dict[str, dict[str, Any]] = {}
    classical_full: dict[str, dict[str, Any]] = {}
    classical_selection: dict[str, dict[str, Any]] = {}
    refusals: dict[str, dict[str, str]] = {}
    pilot_losses: dict[str, list[float]] = {}
    try:
        raw_calibration = _fit_residual(calibration_tables)
        raw_full = _fit_residual(all_tables)
    except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
        raise ProtocolRefusal("MANDATORY_POISSON_DEVIANCE_RESIDUAL_REFUSED") from error
    classical_calibration[RAW_RESIDUAL_METHOD] = raw_calibration
    classical_full[RAW_RESIDUAL_METHOD] = raw_full
    pilot_losses[RAW_RESIDUAL_METHOD] = _losses(
        records, pilot, raw_calibration
    ).tolist()
    for method in CLASSICAL_ORDER:
        try:
            transport_selection = _select_classical_transport_loocv(
                records, calibration, method
            )
            calibration_model = _fit_classical(method, calibration_tables)
            full_model = _fit_classical(method, all_tables)
            multiplier = transport_selection["selected_transport_multiplier"]
            calibration_model["transport_multiplier"] = multiplier
            full_model["transport_multiplier"] = multiplier
            classical_calibration[method] = calibration_model
            classical_full[method] = full_model
            classical_selection[method] = {
                "selected_transport_multiplier": multiplier,
                "fold_losses": {
                    subject: float(loss)
                    for subject, loss in zip(
                        calibration,
                        transport_selection["selected_fold_losses"],
                    )
                },
            }
            pilot_losses[method] = _losses(records, pilot, calibration_model).tolist()
        except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
            refusals[method] = {
                "status": "REFUSED",
                "reason": str(error),
            }
    if (
        RAW_RESIDUAL_METHOD not in classical_full
        or "poisson_independence_deviance_residual" not in classical_full
    ):
        raise ProtocolRefusal("MANDATORY_POISSON_DEVIANCE_RESIDUAL_REFUSED")
    locked_classical = _lock_classical(pilot_losses)
    primary_pilot = _losses(records, pilot, primary_calibration)
    locked_pilot = np.asarray(pilot_losses[locked_classical], dtype=float)
    primary_beats_pilot = float(primary_pilot.mean()) < float(locked_pilot.mean())
    interaction_estimable = any(
        method in classical_full
        for method in ("common_effect_stratified_cmle", "pooled_saturated_poisson")
    )
    broad_ready = bool(interaction_estimable and primary_beats_pilot)

    models = {
        "primary": _json_model(primary_full),
        "destroyed_link": _json_model(destroyed_full),
        **{method: _json_model(model) for method, model in classical_full.items()},
    }
    payload = {
        "schema": "gse202150-source-development/1.0",
        "status": "SOURCE_READY_FOR_HELD_RNA_PREDICTION",
        "created_at_utc": _timestamp(),
        "source_batches": list(SOURCE_BATCHES),
        "source_calibration_subjects": calibration,
        "source_pilot_subjects": pilot,
        "held_batches_opened": [],
        "source_h5_sha256": file_hashes,
        "cell_budget_per_subject": CELL_BUDGET,
        "panel": list(MARKER_LABELS),
        "ordered_pairs": PANEL_SIZE**2,
        "primary_selection": {
            "rule": "minimum donor-equal IOF1 leave-one-subject-out mean deviance; lexicographic configuration tie break",
            "selected_configuration": asdict(selected_config),
            "fold_losses": {
                subject: float(loss)
                for subject, loss in zip(calibration, selection["selected_fold_losses"])
            },
            "complete_candidates": selection["complete_candidates"],
            "refusal_count": len(selection["refusals"]),
        },
        "classical_lock": {
            "rule": "minimum donor-equal IOF2 mean deviance among the raw residual and IOF1-LOOCV-calibrated classical methods estimable on the full source; deterministic ALL_CLASSICAL_METHODS tie break",
            "locked_method": locked_classical,
            "method_order": list(ALL_CLASSICAL_METHODS),
            "source_only_transport_selection": classical_selection,
            "pilot_losses": {
                method: {
                    subject: float(loss)
                    for subject, loss in zip(pilot, values)
                }
                for method, values in pilot_losses.items()
            },
            "refusals": refusals,
        },
        "source_pilot_diagnostic": {
            "primary_losses": {
                subject: float(loss) for subject, loss in zip(pilot, primary_pilot)
            },
            "locked_classical_losses": {
                subject: float(loss) for subject, loss in zip(pilot, locked_pilot)
            },
            "primary_mean_loss": float(primary_pilot.mean()),
            "locked_classical_mean_loss": float(locked_pilot.mean()),
            "primary_lower": primary_beats_pilot,
            "performance_gate_for_held_access": False,
        },
        "broad_gain_over_classical_claim_prerequisites": {
            "direct_residual_estimable": True,
            "at_least_one_interaction_fit_estimable": interaction_estimable,
            "primary_beats_locked_classical_on_source_pilot": primary_beats_pilot,
            "passes": broad_ready,
            "failure_withholds_broad_claim_but_not_held_access": True,
        },
        "models": models,
        "available_methods": list(models),
        "locked_classical_method": locked_classical,
        "source_subject_certificates": [
            {
                "subject": subject,
                "sample": records[subject]["sample"],
                "pathogen": records[subject]["pathogen"],
                "batch": records[subject]["batch"],
                "selected_axis_sha256": records[subject]["axis_sha256"],
                "informative_pairs": int(np.count_nonzero(_informative(records[subject]["tables"]))),
            }
            for subject in all_source
        ],
        "access_audit": audits,
        "bridge_samples_used_for_estimation": [],
        "nonacute_samples_used_for_estimation": [],
        "target_values_used_for_selection": 0,
    }
    _validate_public_payload(payload)
    return payload


def _prerequisite_for_claim(stage: str) -> tuple[str, tuple[str, ...]]:
    if stage == "source":
        return PROTOCOL_TAG, PROTOCOL_BINDINGS
    if stage == "predict":
        return SOURCE_RESULT_TAG, (_relative(DEFAULT_SOURCE),)
    if stage == "score":
        return SCORE_AUTHORIZATION_TAG, (_relative(DEFAULT_SCORE_AUTHORIZATION),)
    raise ValueError("unknown stage")


def claim_stage(stage: str) -> dict[str, Any]:
    if stage not in ATTEMPT_PATHS:
        raise ValueError("stage must be source, predict, or score")
    attempt = ATTEMPT_PATHS[stage]
    execution_claim = EXECUTION_CLAIM_PATHS[stage]
    output = STAGE_OUTPUTS[stage]
    if attempt.exists() or execution_claim.exists() or output.exists():
        raise PermissionError("stage attempt or output already exists; rerun is forbidden")
    runtime = _require_runtime_environment()
    protocol_commit = _require_public_tag(PROTOCOL_TAG, PROTOCOL_BINDINGS)
    prerequisite_tag, paths = _prerequisite_for_claim(stage)
    prerequisite_commit = _require_public_tag(prerequisite_tag, paths)
    payload = {
        "schema": "gse202150-stage-attempt/1.0",
        "stage": stage,
        "status": "STARTED",
        "created_at_utc": _timestamp(),
        "prerequisite_tag": prerequisite_tag,
        "prerequisite_commit": prerequisite_commit,
        "protocol_commit": protocol_commit,
        "runtime_environment": runtime,
        "one_shot": True,
    }
    _append_jsonl(attempt, payload, create=True)
    return payload


def _attempt_started(stage: str) -> dict[str, Any]:
    runtime = _require_runtime_environment()
    path = ATTEMPT_PATHS[stage]
    with path.open() as stream:
        rows = [json.loads(line, object_pairs_hook=_strict_object) for line in stream if line.strip()]
    if len(rows) != 1 or rows[0].get("status") != "STARTED" or rows[0].get("stage") != stage:
        raise PermissionError("numeric stage requires one unclosed STARTED attempt")
    _require_public_tag(ATTEMPT_TAGS[stage], (_relative(path),))
    protocol_commit = _require_public_tag(PROTOCOL_TAG, PROTOCOL_BINDINGS)
    prerequisite_tag, prerequisite_paths = _prerequisite_for_claim(stage)
    prerequisite_commit = _require_public_tag(prerequisite_tag, prerequisite_paths)
    if (
        rows[0].get("protocol_commit") != protocol_commit
        or rows[0].get("prerequisite_tag") != prerequisite_tag
        or rows[0].get("prerequisite_commit") != prerequisite_commit
        or rows[0].get("runtime_environment") != runtime
    ):
        raise PermissionError("stage attempt differs from the current frozen lineage")
    return rows[0]


def _run_claimed_stage(
    stage: str,
    body: Any,
) -> dict[str, Any]:
    output = STAGE_OUTPUTS[stage]
    attempt = ATTEMPT_PATHS[stage]
    if output.exists():
        raise PermissionError("stage output already exists; rerun is forbidden")
    started = _attempt_started(stage)
    executing = {
        "schema": "gse202150-stage-attempt/1.0",
        "stage": stage,
        "status": "EXECUTING_CONSUMED",
        "created_at_utc": _timestamp(),
        "protocol_commit": started["protocol_commit"],
        "runtime_environment": started["runtime_environment"],
        "interruption_consumes_stage": True,
    }
    _write_json_x(EXECUTION_CLAIM_PATHS[stage], executing)
    _append_jsonl(attempt, executing, create=False)
    try:
        payload = dict(body())
        payload["protocol_commit"] = started["protocol_commit"]
        payload["runtime_environment"] = started["runtime_environment"]
        terminal_status = payload["status"]
        _validate_public_payload(payload)
        _write_json_x(output, payload)
    except BaseException as error:
        if output.exists():
            raise
        code = error.code if isinstance(error, ProtocolRefusal) else f"UNEXPECTED_{type(error).__name__.upper()}"
        payload = {
            "schema": f"gse202150-{stage}-terminal/1.0",
            "status": f"TERMINAL_{stage.upper()}_REFUSAL",
            "created_at_utc": _timestamp(),
            "refusal_code": code,
            "attempt_created_at_utc": started["created_at_utc"],
            "protocol_commit": started["protocol_commit"],
            "runtime_environment": started["runtime_environment"],
            "rerun_or_rescue_permitted": False,
        }
        terminal_status = payload["status"]
        _validate_public_payload(payload)
        _write_json_x(output, payload)
    _append_jsonl(
        attempt,
        {
            "schema": "gse202150-stage-attempt/1.0",
            "stage": stage,
            "status": "FINISHED",
            "created_at_utc": _timestamp(),
            "terminal_status": terminal_status,
            "output_sha256": _sha256(output),
            "protocol_commit": started["protocol_commit"],
            "runtime_environment": started["runtime_environment"],
        },
        create=False,
    )
    return payload


def run_source(metadata_root: Path, h5_root: Path) -> dict[str, Any]:
    return _run_claimed_stage(
        "source", lambda: _source_body(metadata_root, h5_root)
    )


def _validated_source() -> tuple[dict[str, Any], str]:
    source, commit = _require_completed_stage_artifact(
        SOURCE_RESULT_TAG, DEFAULT_SOURCE, "source"
    )
    if source.get("schema") != "gse202150-source-development/1.0" or source.get(
        "status"
    ) != "SOURCE_READY_FOR_HELD_RNA_PREDICTION":
        raise PermissionError("public source result is not a valid pass")
    locked = source.get("locked_classical_method")
    models = source.get("models")
    if (
        locked not in ALL_CLASSICAL_METHODS
        or not isinstance(models, dict)
        or "primary" not in models
        or "destroyed_link" not in models
        or locked not in models
        or RAW_RESIDUAL_METHOD not in models
        or "poisson_independence_deviance_residual" not in models
    ):
        raise PermissionError("public source model contract differs")
    return source, commit


def _prediction_body(
    metadata_root: Path, h5_root: Path, private_rna_path: Path
) -> dict[str, Any]:
    source, source_commit = _validated_source()
    private_path = _private_path(private_rna_path)
    if private_path.exists():
        raise PermissionError("private RNA state artifact already exists")
    selected, audits, file_hashes = _load_inference_cells(
        metadata_root,
        h5_root,
        HELD_BATCHES,
        modalities=frozenset(("rna",)),
    )
    if any(audit["adt_values_read"] != 0 for audit in audits):
        raise AssertionError("held RNA firewall decoded ADT values")
    held_rows = _designated_rows(HELD_BATCHES)
    held_subjects = [row["subject"] for row in held_rows]
    methods = [method for method in source["available_methods"] if method in source["models"]]
    predictions: list[dict[str, Any]] = []
    private_subjects: list[dict[str, Any]] = []
    for designation in held_rows:
        subject = designation["subject"]
        rows = selected[subject]
        cells = [row["cell_id"] for row in rows]
        rna = np.asarray([row["rna_state"] for row in rows], dtype=np.int8)
        row_margins = np.empty((PANEL_SIZE, PANEL_SIZE, 2), dtype=np.int64)
        column_margins = np.empty_like(row_margins)
        for row in range(PANEL_SIZE):
            margin = np.bincount(rna[:, row], minlength=2)
            for column in range(PANEL_SIZE):
                row_margins[row, column] = margin
                column_margins[row, column] = (CELL_BUDGET // 2, CELL_BUDGET // 2)
        informative = int(
            np.count_nonzero(
                np.all(row_margins > 0, axis=-1)
                & np.all(column_margins > 0, axis=-1)
            )
        )
        if informative < MINIMUM_INFORMATIVE_PAIRS:
            raise ProtocolRefusal("HELD_SUBJECT_HAS_FEWER_THAN_100_INFORMATIVE_PAIRS")
        method_predictions = {
            method: _predict_model(
                source["models"][method], row_margins, column_margins
            )
            for method in methods
        }
        axis_hash = _axis_sha256(subject, cells)
        predictions.append(
            {
                "subject": subject,
                "sample": designation["sample"],
                "pathogen": designation["pathogen"],
                "batch": designation["batch"],
                "selected_axis_sha256": axis_hash,
                "informative_pairs_from_rna_margins": informative,
                "row_margins": row_margins.tolist(),
                "column_margins": column_margins.tolist(),
                "predicted_tables": {
                    method: table.tolist() for method, table in method_predictions.items()
                },
                "prediction_sha256": {
                    method: _array_sha256(table)
                    for method, table in method_predictions.items()
                },
            }
        )
        private_subjects.append(
            {
                "subject": subject,
                "sample": designation["sample"],
                "batch": designation["batch"],
                "cell_ids": cells,
                "selected_axis_sha256": axis_hash,
                "states": {
                    symbol: rna[:, index].astype(int).tolist()
                    for index, symbol in enumerate(RNA_SYMBOLS)
                },
            }
        )
    if len(predictions) != 17 or set(held_subjects) != {
        row["subject"] for row in predictions
    }:
        raise ProtocolRefusal("HELD_SUBJECT_AXIS_DIFFERS")
    private_payload = {
        "schema": "gse202150-private-held-rna-states/1.0",
        "created_at_utc": _timestamp(),
        "subjects": private_subjects,
    }
    _write_json_x(private_path, private_payload)
    payload = {
        "schema": "gse202150-held-predictions/1.0",
        "status": "PREDICTIONS_FROZEN_BEFORE_HELD_ADT_COUNT_DECODING",
        "created_at_utc": _timestamp(),
        "source_commit": source_commit,
        "source_sha256": _sha256(DEFAULT_SOURCE),
        "held_batches": list(HELD_BATCHES),
        "held_subjects": held_subjects,
        "held_h5_sha256": file_hashes,
        "panel": list(MARKER_LABELS),
        "available_methods": methods,
        "locked_classical_method": source["locked_classical_method"],
        "predictions": predictions,
        "private_rna_artifact": {
            "bytes": private_path.stat().st_size,
            "sha256": _sha256(private_path),
        },
        "access_audit": audits,
        "held_adt_count_values_decoded": 0,
        "held_sparse_support_indices_read": True,
        "whole_file_sha256_computed": True,
        "held_joint_tables_formed": 0,
        "sparse_h5_co_residence_disclosed": True,
        "ignored_adt_values_returned_or_serialized": 0,
    }
    _validate_public_payload(payload)
    return payload


def run_prediction(
    metadata_root: Path, h5_root: Path, private_rna_path: Path
) -> dict[str, Any]:
    return _run_claimed_stage(
        "predict",
        lambda: _prediction_body(metadata_root, h5_root, private_rna_path),
    )


def _validated_prediction() -> tuple[dict[str, Any], str]:
    prediction, commit = _require_completed_stage_artifact(
        PREDICTION_TAG, DEFAULT_PREDICTION, "predict"
    )
    _, source_commit = _validated_source()
    if prediction.get("schema") != "gse202150-held-predictions/1.0" or prediction.get(
        "status"
    ) != "PREDICTIONS_FROZEN_BEFORE_HELD_ADT_COUNT_DECODING":
        raise PermissionError("public prediction artifact is not a valid freeze")
    if prediction.get("held_adt_count_values_decoded") != 0 or prediction.get(
        "held_joint_tables_formed"
    ) != 0:
        raise PermissionError("public prediction artifact crossed the ADT firewall")
    if (
        prediction.get("source_commit") != source_commit
        or prediction.get("source_sha256") != _sha256(DEFAULT_SOURCE)
    ):
        raise PermissionError("public prediction source lineage differs")
    return prediction, commit


def authorize_score() -> dict[str, Any]:
    if DEFAULT_SCORE_AUTHORIZATION.exists():
        raise PermissionError("score authorization already exists")
    prediction, commit = _validated_prediction()
    payload = {
        "schema": "gse202150-score-authorization/1.0",
        "status": "AUTHORIZED_ONCE",
        "created_at_utc": _timestamp(),
        "prediction_tag": PREDICTION_TAG,
        "prediction_commit": commit,
        "prediction_sha256": _sha256(DEFAULT_PREDICTION),
        "held_subjects": prediction["held_subjects"],
        "authorized_score_stage": "one-shot paired held ADT decoding and scoring",
        "rescue_or_rerun_permitted": False,
    }
    _write_json_x(DEFAULT_SCORE_AUTHORIZATION, payload)
    return payload


def _validated_score_authorization() -> tuple[dict[str, Any], str]:
    authorization, commit = _require_tagged_artifact(
        SCORE_AUTHORIZATION_TAG, DEFAULT_SCORE_AUTHORIZATION
    )
    prediction, prediction_commit = _validated_prediction()
    if (
        authorization.get("schema") != "gse202150-score-authorization/1.0"
        or authorization.get("status") != "AUTHORIZED_ONCE"
        or authorization.get("prediction_commit") != prediction_commit
        or authorization.get("prediction_sha256") != _sha256(DEFAULT_PREDICTION)
        or authorization.get("held_subjects") != prediction.get("held_subjects")
    ):
        raise PermissionError("score authorization differs from public prediction freeze")
    return authorization, commit


def _private_rna(path: Path, prediction: dict[str, Any]) -> dict[str, Any]:
    private_path = _private_path(path)
    expected = prediction["private_rna_artifact"]
    if private_path.stat().st_size != expected["bytes"] or _sha256(private_path) != expected["sha256"]:
        raise PermissionError("private RNA artifact differs from public byte commitment")
    value = _read_json(private_path)
    if value.get("schema") != "gse202150-private-held-rna-states/1.0":
        raise PermissionError("private RNA state schema differs")
    return value


def _truth_from_held_adt(
    metadata_root: Path,
    h5_root: Path,
    private: dict[str, Any],
    expected_file_hashes: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    official = _official_metadata(metadata_root)
    paths = _h5_paths(h5_root, HELD_BATCHES)
    file_hashes = {library: _sha256(path) for library, path in sorted(paths.items())}
    if file_hashes != expected_file_hashes:
        raise PermissionError("held H5 bytes differ from the public prediction commitment")
    private_rows = {row["subject"]: row for row in private["subjects"]}
    selected_ids = {
        cell for row in private_rows.values() for cell in row["cell_ids"]
    }
    if len(selected_ids) != 17 * CELL_BUDGET:
        raise PermissionError("private held cell axis is duplicated or incomplete")
    observed: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for library, path in sorted(paths.items()):
        records, audit = _scan_library(
            path,
            library,
            official,
            modalities=frozenset(("adt",)),
            selected_ids=selected_ids,
        )
        observed.extend(records)
        audits.append(audit)
    by_cell = {row["cell_id"]: row for row in observed}
    if len(by_cell) != len(observed) or set(by_cell) != selected_ids:
        raise ProtocolRefusal("HELD_SELECTED_CELL_AXIS_NOT_RECOVERED_EXACTLY")
    output: dict[str, dict[str, Any]] = {}
    for subject, row in private_rows.items():
        cells = row["cell_ids"]
        if row["selected_axis_sha256"] != _axis_sha256(subject, cells):
            raise PermissionError("private held axis hash differs")
        adt_counts = np.asarray([by_cell[cell]["adt_counts"] for cell in cells], dtype=np.int64)
        adt = _adt_states(adt_counts, cells, subject)
        rna = np.column_stack(
            [np.asarray(row["states"][symbol], dtype=np.int8) for symbol in RNA_SYMBOLS]
        )
        truth = _binary_tables(rna, adt)
        output[subject] = {
            "tables": truth,
            "pathogen": by_cell[cells[0]]["pathogen"],
            "batch": by_cell[cells[0]]["batch"],
            "axis_sha256": row["selected_axis_sha256"],
        }
    return output, audits, file_hashes


def _bootstrap_interval(differences: np.ndarray, seed_offset: int = 0) -> list[float]:
    values = np.asarray(differences, dtype=float)
    generator = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    means = np.empty(BOOTSTRAPS, dtype=float)
    for start in range(0, BOOTSTRAPS, 1000):
        stop = min(start + 1000, BOOTSTRAPS)
        indices = generator.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return np.quantile(means, [0.025, 0.975], method="linear").tolist()


def _exact_donor_sign_test(differences: np.ndarray) -> dict[str, Any]:
    values = np.asarray(differences, dtype=float)
    favorable = int(np.count_nonzero(values < 0.0))
    unfavorable = int(np.count_nonzero(values > 0.0))
    ties = int(np.count_nonzero(values == 0.0))
    non_ties = favorable + unfavorable
    probability = (
        sum(math.comb(non_ties, count) for count in range(favorable, non_ties + 1))
        / 2**non_ties
        if non_ties
        else 1.0
    )
    return {
        "favorable": favorable,
        "unfavorable": unfavorable,
        "exact_ties": ties,
        "non_tied_subjects": non_ties,
        "one_sided_p": probability,
    }


def _pathogen_sign_flip(
    differences: np.ndarray, pathogens: list[str]
) -> dict[str, Any]:
    values = np.asarray(differences, dtype=float)
    axis = np.asarray(pathogens, dtype=object)
    strata = sorted(set(pathogens))
    if len(strata) < 6:
        raise ProtocolRefusal("HELD_HAS_FEWER_THAN_SIX_PATHOGEN_STRATA")
    means = np.asarray([values[axis == pathogen].mean() for pathogen in strata])
    observed = float(means.mean())
    draws = 1 << len(strata)
    favorable = 0
    for mask in range(draws):
        signed = np.asarray(
            [
                -value if mask & (1 << index) else value
                for index, value in enumerate(means)
            ]
        )
        favorable += float(signed.mean()) <= observed + 1e-15
    return {
        "method": "exact_pathogen_stratum_sign_flip",
        "strata": strata,
        "stratum_mean_differences": {
            pathogen: float(value) for pathogen, value in zip(strata, means)
        },
        "draws": draws,
        "observed_equal_stratum_mean_difference": observed,
        "one_sided_p": favorable / draws,
    }


def _held_comparison(
    subjects: list[str],
    batches: list[str],
    pathogens: list[str],
    primary: np.ndarray,
    comparator: np.ndarray,
    *,
    gating: bool,
    seed_offset: int,
) -> dict[str, Any]:
    primary_values = np.asarray(primary, dtype=float)
    comparator_values = np.asarray(comparator, dtype=float)
    if not (
        len(subjects)
        == len(batches)
        == len(pathogens)
        == len(primary_values)
        == len(comparator_values)
        == 17
    ):
        raise ValueError("held comparison requires the frozen 17-subject axis")
    difference = primary_values - comparator_values
    interval = _bootstrap_interval(difference, seed_offset)
    sign = _exact_donor_sign_test(difference)
    pathogen_flip = _pathogen_sign_flip(difference, pathogens)
    batch_axis = np.asarray(batches, dtype=object)
    pathogen_axis = np.asarray(pathogens, dtype=object)
    batch_means = {
        batch: float(difference[batch_axis == batch].mean())
        for batch in sorted(set(batches))
    }
    leave_one_pathogen_out = {
        pathogen: float(difference[pathogen_axis != pathogen].mean())
        for pathogen in sorted(set(pathogens))
    }
    comparator_mean = float(comparator_values.mean())
    relative = (
        1.0 - float(primary_values.mean()) / comparator_mean
        if comparator_mean > 0.0
        else None
    )
    favorable = int(np.count_nonzero(difference < 0.0))
    if gating:
        checks = {
            "relative_loss_reduction_at_least_five_percent": relative is not None and relative >= 0.05,
            "donor_bootstrap_97_5_percentile_below_zero": interval[1] < 0.0,
            "favorable_subject_fraction_at_least_0_8": favorable >= math.ceil(0.8 * 17),
            "donor_exact_sign_one_sided_p_at_most_0_025": sign["one_sided_p"] <= 0.025,
            "both_held_batch_means_negative": set(batch_means) == set(HELD_BATCHES) and all(value < 0.0 for value in batch_means.values()),
            "every_leave_one_pathogen_out_mean_negative": all(value < 0.0 for value in leave_one_pathogen_out.values()),
        }
    else:
        checks = {
            "primary_mean_loss_lower": float(difference.mean()) < 0.0,
            "donor_bootstrap_97_5_percentile_below_zero": interval[1] < 0.0,
        }
    return {
        "subjects": 17,
        "primary_mean_loss": float(primary_values.mean()),
        "comparator_mean_loss": comparator_mean,
        "relative_loss_reduction": relative,
        "mean_paired_difference": float(difference.mean()),
        "donor_bootstrap_95_interval": interval,
        "bootstrap_draws": BOOTSTRAPS,
        "favorable_subjects": favorable,
        "required_favorable_subjects": math.ceil(0.8 * 17),
        "donor_exact_sign_test": sign,
        "batch_mean_differences": batch_means,
        "pathogen_sign_flip": pathogen_flip,
        "pathogen_sign_flip_role": "prespecified heterogeneity sensitivity; not a primary gate",
        "leave_one_pathogen_out_mean_differences": leave_one_pathogen_out,
        "checks": checks,
        "passes": all(checks.values()),
        "subject_differences": {
            subject: float(value) for subject, value in zip(subjects, difference)
        },
    }


def _prediction_records(prediction: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = prediction.get("predictions")
    if not isinstance(rows, list) or len(rows) != 17:
        raise PermissionError("prediction subject records differ")
    mapped = {row.get("subject"): row for row in rows}
    if len(mapped) != 17 or None in mapped:
        raise PermissionError("prediction subject axis is duplicated")
    return mapped


def _bridge_qc(metadata_root: Path, h5_root: Path) -> dict[str, Any]:
    try:
        official = _official_metadata(metadata_root)
        paths = _h5_paths(h5_root, BATCHES)
        observed: list[dict[str, Any]] = []
        audits: list[dict[str, Any]] = []
        for library, path in sorted(paths.items()):
            records, audit = _scan_library(
                path,
                library,
                official,
                modalities=frozenset(("rna", "adt")),
                scope="bridges",
            )
            observed.extend(records)
            audits.append(audit)
        expected = [f"{donor}@{batch}" for donor in ("HD105", "HD108") for batch in BATCHES]
        selected = _select_subject_cells(observed, expected)
        records = _subject_records(selected)
        correlations: dict[str, dict[str, float]] = {}
        for donor in ("HD105", "HD108"):
            profiles = {
                batch: np.concatenate(
                    (
                        records[f"{donor}@{batch}"]["rna_profile"],
                        records[f"{donor}@{batch}"]["adt_profile"],
                    )
                )
                for batch in BATCHES
            }
            correlations[donor] = {
                f"{left}-{right}": float(np.corrcoef(profiles[left], profiles[right])[0, 1])
                for left_index, left in enumerate(BATCHES)
                for right in BATCHES[left_index + 1 :]
            }
        return {
            "status": "DESCRIPTIVE_BRIDGE_QC_COMPLETED_AFTER_CONFIRMATORY_SCORING",
            "inference_or_model_selection_use": False,
            "bridge_donor_batch_units": 8,
            "profile_correlations": correlations,
            "access_audit": audits,
        }
    except Exception as error:
        return {
            "status": "DESCRIPTIVE_BRIDGE_QC_REFUSED",
            "inference_or_model_selection_use": False,
            "reason_type": type(error).__name__,
        }


def _score_body(
    metadata_root: Path,
    h5_root: Path,
    private_rna_path: Path,
) -> dict[str, Any]:
    _validated_score_authorization()
    source, source_commit = _validated_source()
    prediction, prediction_commit = _validated_prediction()
    private = _private_rna(private_rna_path, prediction)
    truth, audits, file_hashes = _truth_from_held_adt(
        metadata_root,
        h5_root,
        private,
        prediction["held_h5_sha256"],
    )
    prediction_rows = _prediction_records(prediction)
    subjects = prediction["held_subjects"]
    if set(subjects) != set(truth):
        raise ProtocolRefusal("HELD_TRUTH_SUBJECT_AXIS_DIFFERS")
    methods = prediction["available_methods"]
    scored_methods = [method for method in methods if method != "destroyed_link"]
    losses = {method: np.empty(17, dtype=float) for method in scored_methods}
    batches: list[str] = []
    pathogens: list[str] = []
    truth_hashes: dict[str, str] = {}
    for subject_index, subject in enumerate(subjects):
        observed = truth[subject]["tables"]
        truth_hashes[subject] = _array_sha256(observed)
        batches.append(truth[subject]["batch"])
        pathogens.append(truth[subject]["pathogen"])
        record = prediction_rows[subject]
        if record["selected_axis_sha256"] != truth[subject]["axis_sha256"]:
            raise PermissionError("held truth and prediction cell axes differ")
        for method in scored_methods:
            predicted = np.asarray(record["predicted_tables"][method], dtype=float)
            if _array_sha256(predicted) != record["prediction_sha256"][method]:
                raise PermissionError("held prediction table hash differs")
            losses[method][subject_index] = _donor_loss(observed, predicted)
    locked = source["locked_classical_method"]
    if locked not in losses or "primary" not in losses:
        raise PermissionError("locked held comparison is absent")
    primary_comparison = _held_comparison(
        subjects,
        batches,
        pathogens,
        losses["primary"],
        losses[locked],
        gating=True,
        seed_offset=0,
    )
    classical_comparisons = {
        method: _held_comparison(
            subjects,
            batches,
            pathogens,
            losses["primary"],
            losses[method],
            gating=False,
            seed_offset=10 + index,
        )
        for index, method in enumerate(ALL_CLASSICAL_METHODS)
        if method in losses
    }
    if primary_comparison["passes"]:
        destroyed_losses = np.empty(17, dtype=float)
        for subject_index, subject in enumerate(subjects):
            record = prediction_rows[subject]
            predicted = np.asarray(
                record["predicted_tables"]["destroyed_link"], dtype=float
            )
            if _array_sha256(predicted) != record["prediction_sha256"]["destroyed_link"]:
                raise PermissionError("destroyed-link prediction table hash differs")
            destroyed_losses[subject_index] = _donor_loss(
                truth[subject]["tables"], predicted
            )
        losses["destroyed_link"] = destroyed_losses
        destroyed_comparison: dict[str, Any] = _held_comparison(
            subjects,
            batches,
            pathogens,
            losses["primary"],
            destroyed_losses,
            gating=False,
            seed_offset=30,
        )
        destroyed_comparison["serial_secondary_evaluated_after_primary_gate"] = True
    else:
        destroyed_comparison = {
            "status": "NOT_EVALUATED_PRIMARY_HELD_VALIDATION_FAILED",
            "serial_secondary_evaluated_after_primary_gate": False,
        }
    broad_prerequisites = source["broad_gain_over_classical_claim_prerequisites"]
    gain_over_all = bool(
        classical_comparisons
        and all(value["passes"] for value in classical_comparisons.values())
    )
    broad_supported = bool(
        primary_comparison["passes"]
        and broad_prerequisites["passes"]
        and gain_over_all
    )
    bridge_qc = _bridge_qc(metadata_root, h5_root)
    passed = bool(primary_comparison["passes"])
    payload = {
        "schema": "gse202150-held-validation/1.0",
        "status": (
            "COMPLETED_HELD_VALIDATION_PASS"
            if passed
            else "COMPLETED_HELD_VALIDATION_FAIL"
        ),
        "created_at_utc": _timestamp(),
        "source_commit": source_commit,
        "prediction_commit": prediction_commit,
        "held_subjects": subjects,
        "held_batches": list(HELD_BATCHES),
        "pathogen_strata": sorted(set(pathogens)),
        "held_h5_sha256": file_hashes,
        "truth_table_sha256": truth_hashes,
        "locked_classical_method": locked,
        "method_losses": {
            method: {
                subject: float(loss) for subject, loss in zip(subjects, values)
            }
            for method, values in losses.items()
        },
        "primary_held_validation": primary_comparison,
        "classical_head_to_head": classical_comparisons,
        "destroyed_link_serial_secondary": destroyed_comparison,
        "broad_gain_over_classical_claim_supported": broad_supported,
        "broad_claim_contract": {
            "source_prerequisites_pass": broad_prerequisites["passes"],
            "primary_beats_every_estimable_classical_on_held_with_bootstrap_support": gain_over_all,
            "primary_held_validation_pass": primary_comparison["passes"],
        },
        "bridge_batch_qc": bridge_qc,
        "held_score_access_audit": audits,
        "held_adt_count_values_decoded_only_after_public_prediction_and_score_authorization": True,
        "rescue_or_rerun_permitted": False,
    }
    _validate_public_payload(payload)
    return payload


def run_score(
    metadata_root: Path, h5_root: Path, private_rna_path: Path
) -> dict[str, Any]:
    return _run_claimed_stage(
        "score", lambda: _score_body(metadata_root, h5_root, private_rna_path)
    )


def verify_public_result() -> dict[str, Any]:
    result, commit = _require_completed_stage_artifact(
        RESULT_TAG, DEFAULT_SCORE, "score"
    )
    if result.get("schema") != "gse202150-held-validation/1.0" or result.get(
        "status"
    ) not in {
        "COMPLETED_HELD_VALIDATION_PASS",
        "COMPLETED_HELD_VALIDATION_FAIL",
    }:
        raise PermissionError("public held-validation result schema or status differs")
    return {
        "schema": "gse202150-public-result-verification/1.0",
        "status": "PUBLIC_RESULT_LINEAGE_VERIFIED",
        "result_commit": commit,
        "result_sha256": _sha256(DEFAULT_SCORE),
        "held_validation_status": result["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--metadata-root", type=Path, required=True)
    preflight.add_argument("--h5-root", type=Path, required=True)
    claim = subparsers.add_parser("claim")
    claim.add_argument("stage", choices=tuple(ATTEMPT_PATHS))
    source = subparsers.add_parser("source")
    source.add_argument("--metadata-root", type=Path, required=True)
    source.add_argument("--h5-root", type=Path, required=True)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--metadata-root", type=Path, required=True)
    predict.add_argument("--h5-root", type=Path, required=True)
    predict.add_argument("--private-rna", type=Path, required=True)
    subparsers.add_parser("authorize-score")
    score = subparsers.add_parser("score")
    score.add_argument("--metadata-root", type=Path, required=True)
    score.add_argument("--h5-root", type=Path, required=True)
    score.add_argument("--private-rna", type=Path, required=True)
    subparsers.add_parser("verify-result")
    arguments = parser.parse_args()
    if arguments.command == "preflight":
        payload = run_preflight(arguments.metadata_root, arguments.h5_root)
    elif arguments.command == "claim":
        payload = claim_stage(arguments.stage)
    elif arguments.command == "source":
        payload = run_source(arguments.metadata_root, arguments.h5_root)
    elif arguments.command == "predict":
        payload = run_prediction(
            arguments.metadata_root, arguments.h5_root, arguments.private_rna
        )
    elif arguments.command == "authorize-score":
        payload = authorize_score()
    elif arguments.command == "score":
        payload = run_score(
            arguments.metadata_root, arguments.h5_root, arguments.private_rna
        )
    else:
        payload = verify_public_result()
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
