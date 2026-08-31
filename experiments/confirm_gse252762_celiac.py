"""Sealed GSE252762 celiac CITE-seq held-batch confirmation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import secrets
import subprocess
from typing import Any, Iterator, Mapping, Sequence
import urllib.request

import numpy as np
import scipy

from experiments import gse252762_celiac_core as core
from mapreg.streamed_gzip_matrix_market import reduce_gzip_matrix_market


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/confirmation/gse252762_celiac"
PREFLIGHT = ROOT / "results/development/gse252762_celiac_metadata_preflight_v1.json"
DESIGNATION = DATA_DIR / "candidate_designation_v2.json"
PROTOCOL = ROOT / "docs/GSE252762_CELIAC_CITESEQ_HELD_BATCH_PROTOCOL_V2_2026-08-30.md"

SOURCE_AUTHORIZATION = DATA_DIR / "source_authorization_v2.json"
SOURCE_ATTEMPT = DATA_DIR / "source_attempt_v2.json"
SOURCE_CONSUMPTION = DATA_DIR / "source_consumption_v2.json"
SOURCE_JOURNAL = DATA_DIR / "source_access_v2.jsonl"
SOURCE_REDUCED = ROOT / "data/development/gse252762_celiac/source_reduced_v2.json"
SOURCE_RESULT = ROOT / "results/development/gse252762_celiac_source_v2.json"

PREDICTION_AUTHORIZATION = DATA_DIR / "held_rna_authorization_v2.json"
PREDICTION_ATTEMPT = DATA_DIR / "prediction_attempt_v2.json"
PREDICTION_CONSUMPTION = DATA_DIR / "prediction_consumption_v2.json"
PREDICTION_JOURNAL = DATA_DIR / "held_rna_access_v2.jsonl"
PREDICTION_REDUCED = (
    ROOT / "data/confirmation/gse252762_celiac/held_rna_reduced_v2.json"
)
PREDICTIONS = ROOT / "results/gse252762_celiac_predictions_v2.json"

SCORE_AUTHORIZATION = DATA_DIR / "score_authorization_v2.json"
SCORE_ATTEMPT = DATA_DIR / "score_attempt_v2.json"
SCORE_CONSUMPTION = DATA_DIR / "score_consumption_v2.json"
SCORE_JOURNAL = DATA_DIR / "held_cite_access_v2.jsonl"
SCORE_REDUCED = ROOT / "data/confirmation/gse252762_celiac/held_cite_reduced_v2.json"
SCORE_RESULT = ROOT / "results/gse252762_celiac_confirmation_v2.json"

SOURCE_SELECTED_SIDECARS = tuple(
    DATA_DIR / f"source_batch{batch}_{modality}_selected_v2.json"
    for batch in range(1, 6)
    for modality in ("rna", "cite")
)
PREDICTION_SELECTED_SIDECARS = (DATA_DIR / "held_rna_selected_v2.json",)
SCORE_SELECTED_SIDECARS = (DATA_DIR / "held_cite_selected_v2.json",)

PUBLIC_ORIGIN = "https://github.com/sushaan-k/coupling-fields-benchmark.git"
CANDIDATE_TAG = "gse252762-celiac-v2-candidate"
CANDIDATE_COMMIT = "fd84891c9c4be03e7faeeffd09838a98f2f1bda1"
IMPLEMENTATION_TAG = "gse252762-celiac-v2-implementation"
SOURCE_AUTHORIZATION_TAG = "gse252762-celiac-v2-source-authorization"
SOURCE_ATTEMPT_TAG = "gse252762-celiac-v2-source-attempt"
SOURCE_CONSUMPTION_TAG = "gse252762-celiac-v2-source-consumption"
SOURCE_RESULT_TAG = "gse252762-celiac-v2-source-result"
PREDICTION_AUTHORIZATION_TAG = "gse252762-celiac-v2-held-rna-authorization"
PREDICTION_ATTEMPT_TAG = "gse252762-celiac-v2-prediction-attempt"
PREDICTION_CONSUMPTION_TAG = "gse252762-celiac-v2-prediction-consumption"
PREDICTIONS_TAG = "gse252762-celiac-v2-predictions"
SCORE_AUTHORIZATION_TAG = "gse252762-celiac-v2-score-authorization"
SCORE_ATTEMPT_TAG = "gse252762-celiac-v2-score-attempt"
SCORE_CONSUMPTION_TAG = "gse252762-celiac-v2-score-consumption"
SCORE_RESULT_TAG = "gse252762-celiac-v2-held-result"

IMPLEMENTATION_BINDINGS = (
    "experiments/confirm_gse252762_celiac.py",
    "experiments/gse217494_heart_core.py",
    "experiments/gse252762_celiac_core.py",
    "experiments/preflight_gse252762_celiac.py",
    "tests/test_gse252762_celiac_confirmation.py",
    "tests/test_gse252762_celiac_core.py",
    "tests/test_gse252762_celiac_preflight.py",
    "tests/test_poisson_loglinear.py",
    "docs/GSE252762_CELIAC_CITESEQ_HELD_BATCH_PROTOCOL_V2_2026-08-30.md",
    "docs/GSE252762_CELIAC_EXECUTION_CONTRACT_V2.md",
    "data/confirmation/gse252762_celiac/candidate_designation_v2.json",
    "results/development/gse252762_celiac_metadata_preflight_v1.json",
    "mapreg/__init__.py",
    "mapreg/classical_residuals.py",
    "mapreg/common_effect_conditional.py",
    "mapreg/context_conditional_coupling.py",
    "mapreg/coupling_fields.py",
    "mapreg/factorial_coupling.py",
    "mapreg/heterogeneity_adaptive_coupling.py",
    "mapreg/hierarchical_conditional_coupling.py",
    "mapreg/longitudinal_conditional_coupling.py",
    "mapreg/penalty_complete_conditional_coupling.py",
    "mapreg/poisson_loglinear.py",
    "mapreg/streamed_gzip_matrix_market.py",
    "mapreg/streamed_matrix_market.py",
    "mapreg/structured_context_conditional.py",
    "mapreg/table_prediction.py",
    "pyproject.toml",
    "requirements.txt",
)

STAGE_PATHS = {
    "source": {
        "authorization": SOURCE_AUTHORIZATION,
        "authorization_tag": SOURCE_AUTHORIZATION_TAG,
        "authorization_schema": "gse252762-celiac-source-authorization/2.0",
        "authorization_status": "SOURCE_AUTHORIZED",
        "attempt": SOURCE_ATTEMPT,
        "attempt_tag": SOURCE_ATTEMPT_TAG,
        "consumption": SOURCE_CONSUMPTION,
        "consumption_tag": SOURCE_CONSUMPTION_TAG,
        "journal": SOURCE_JOURNAL,
        "sidecars": SOURCE_SELECTED_SIDECARS,
        "checkpoint": SOURCE_REDUCED,
        "result": SOURCE_RESULT,
        "matrix_gets": 10,
    },
    "prediction": {
        "authorization": PREDICTION_AUTHORIZATION,
        "authorization_tag": PREDICTION_AUTHORIZATION_TAG,
        "authorization_schema": "gse252762-celiac-prediction-authorization/2.0",
        "authorization_status": "HELD_RNA_AUTHORIZED",
        "attempt": PREDICTION_ATTEMPT,
        "attempt_tag": PREDICTION_ATTEMPT_TAG,
        "consumption": PREDICTION_CONSUMPTION,
        "consumption_tag": PREDICTION_CONSUMPTION_TAG,
        "journal": PREDICTION_JOURNAL,
        "sidecars": PREDICTION_SELECTED_SIDECARS,
        "checkpoint": PREDICTION_REDUCED,
        "result": PREDICTIONS,
        "matrix_gets": 1,
    },
    "score": {
        "authorization": SCORE_AUTHORIZATION,
        "authorization_tag": SCORE_AUTHORIZATION_TAG,
        "authorization_schema": "gse252762-celiac-score-authorization/2.0",
        "authorization_status": "HELD_CITE_SCORE_AUTHORIZED",
        "attempt": SCORE_ATTEMPT,
        "attempt_tag": SCORE_ATTEMPT_TAG,
        "consumption": SCORE_CONSUMPTION,
        "consumption_tag": SCORE_CONSUMPTION_TAG,
        "journal": SCORE_JOURNAL,
        "sidecars": SCORE_SELECTED_SIDECARS,
        "checkpoint": SCORE_REDUCED,
        "result": SCORE_RESULT,
        "matrix_gets": 1,
    },
}
STAGE_ORDER = ("source", "prediction", "score")

CHECKPOINT_BINDINGS = {
    "source": ("source_reduced_path", "source_reduced_sha256"),
    "prediction": ("held_rna_reduced_path", "held_rna_reduced_sha256"),
    "score": ("held_cite_reduced_path", "held_cite_reduced_sha256"),
}

FROZEN_RUNTIME = {
    "python": "3.9.6",
    "numpy": "2.0.2",
    "scipy": "1.13.1",
    "platform": "macOS-26.0.1-arm64-arm-64bit",
    "machine": "arm64",
    "blas": "accelerate",
    "lapack": "accelerate",
    "thread_environment": {
        "OMP_NUM_THREADS": None,
        "OPENBLAS_NUM_THREADS": None,
        "MKL_NUM_THREADS": None,
        "VECLIB_MAXIMUM_THREADS": None,
        "NUMEXPR_NUM_THREADS": None,
    },
}

FAILURE_CODES = {
    "source": {
        "source_reduction": "SOURCE_MATRIX_REDUCTION_FAILED",
        "source_selection": "SOURCE_SELECTION_FAILED",
        "recovery": "PROCESS_INTERRUPTED_AFTER_SOURCE_CONSUMPTION",
    },
    "prediction": {
        "held_rna_reduction": "HELD_RNA_REDUCTION_FAILED",
        "held_prediction": "HELD_PREDICTION_FAILED",
        "recovery": "PROCESS_INTERRUPTED_AFTER_PREDICTION_CONSUMPTION",
    },
    "score": {
        "held_cite_reduction": "HELD_CITE_REDUCTION_FAILED",
        "held_scoring": "HELD_SCORING_FAILED",
        "recovery": "PROCESS_INTERRUPTED_AFTER_SCORE_CONSUMPTION",
    },
}


class ProtocolRefusal(RuntimeError):
    """A frozen access or analysis condition was not met."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


_URL_OPENER = urllib.request.build_opener(_NoRedirect)


def _timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _numpy_dependency_name(dependency: str) -> str:
    build_dependencies = np.__config__.CONFIG.get("Build Dependencies", {})
    record = build_dependencies.get(dependency, {})
    name = record.get("name") if isinstance(record, Mapping) else None
    return name if isinstance(name, str) else "unknown"


def _runtime() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "blas": _numpy_dependency_name("blas"),
        "lapack": _numpy_dependency_name("lapack"),
        "thread_environment": {
            name: os.environ.get(name) for name in FROZEN_RUNTIME["thread_environment"]
        },
    }


def _require_runtime() -> dict[str, Any]:
    runtime = _runtime()
    if runtime != FROZEN_RUNTIME:
        raise ProtocolRefusal("runtime differs from the frozen numerical environment")
    return runtime


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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if not isinstance(payload, dict):
        raise ProtocolRefusal(f"{path.name} must contain one JSON object")
    return payload


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{secrets.token_hex(16)}.tmp")


def _write_temp_bytes(path: Path, payload: bytes) -> Path:
    temporary = _temporary_path(path)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("temporary artifact write made no progress")
            offset += written
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    os.close(descriptor)
    return temporary


def _atomic_create_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _write_temp_bytes(path, payload)
    try:
        os.link(temporary, path)
        _fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _write_temp_bytes(path, payload)
    try:
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        + "\n"
    )
    _atomic_create_bytes(path, text.encode("utf-8"))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_value(payload: Any) -> Any:
    return json.loads(
        json.dumps(
            payload,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
    )


def _append_journal(path: Path, payload: Mapping[str, Any]) -> None:
    _read_jsonl(path)
    current = path.read_bytes()
    record = (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode()
    _atomic_replace_bytes(path, current + record)


def _write_journal_header(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
    _atomic_create_bytes(path, text.encode("utf-8"))


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ProtocolRefusal("public artifact escapes the repository") from error


def _git(*arguments: str, check: bool = True, text: bool = True) -> Any:
    environment = dict(os.environ)
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=check,
        capture_output=True,
        text=text,
        env=environment,
    )


def _require_git_integrity() -> None:
    replacements = _git("for-each-ref", "--format=%(refname)", "refs/replace").stdout
    grafts = Path(_git("rev-parse", "--git-path", "info/grafts").stdout.strip())
    if not grafts.is_absolute():
        grafts = ROOT / grafts
    rewrites = _git(
        "config",
        "--show-origin",
        "--get-regexp",
        r"^url\..*\.(insteadof|pushinsteadof)$",
        check=False,
    )
    if replacements.strip() or (grafts.is_file() and grafts.stat().st_size > 0):
        raise ProtocolRefusal("Git replacement or graft state is not permitted")
    if rewrites.returncode not in {0, 1} or rewrites.stdout.strip():
        raise ProtocolRefusal("Git URL rewriting is not permitted")


def _remote_tag_ids(tag: str) -> tuple[str, str]:
    _require_git_integrity()
    lines = _git(
        "ls-remote",
        PUBLIC_ORIGIN,
        f"refs/tags/{tag}",
        f"refs/tags/{tag}^{{}}",
    ).stdout.splitlines()
    refs = {fields[1]: fields[0] for line in lines if len(fields := line.split()) == 2}
    tag_object = refs.get(f"refs/tags/{tag}")
    commit = refs.get(f"refs/tags/{tag}^{{}}")
    if tag_object is None or commit is None:
        raise ProtocolRefusal(f"public annotated tag {tag} is absent")
    return tag_object, commit


def _stage_artifact_paths(stage: str, fields: Sequence[str]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for field in fields:
        value = STAGE_PATHS[stage][field]
        if field == "sidecars":
            paths.extend(value)
        else:
            paths.append(value)
    return tuple(paths)


def _campaign_artifact_paths() -> tuple[Path, ...]:
    fields = (
        "authorization",
        "attempt",
        "consumption",
        "journal",
        "sidecars",
        "checkpoint",
        "result",
    )
    return tuple(
        path for stage in STAGE_ORDER for path in _stage_artifact_paths(stage, fields)
    )


def _forbidden_at_boundary(stage: str, boundary: str) -> tuple[Path, ...]:
    if stage not in STAGE_ORDER or boundary not in {
        "authorization",
        "attempt",
        "consumption",
        "result",
    }:
        raise ValueError("unknown campaign boundary")
    current_fields = {
        "authorization": (
            "attempt",
            "consumption",
            "journal",
            "sidecars",
            "checkpoint",
            "result",
        ),
        "attempt": ("consumption", "journal", "sidecars", "checkpoint", "result"),
        "consumption": ("journal", "sidecars", "checkpoint", "result"),
        "result": (),
    }[boundary]
    paths = list(_stage_artifact_paths(stage, current_fields))
    for later in STAGE_ORDER[STAGE_ORDER.index(stage) + 1 :]:
        paths.extend(
            _stage_artifact_paths(
                later,
                (
                    "authorization",
                    "attempt",
                    "consumption",
                    "journal",
                    "sidecars",
                    "checkpoint",
                    "result",
                ),
            )
        )
    return tuple(paths)


def _require_public_tag(
    tag: str,
    paths: Sequence[Path],
    *,
    absent_paths: Sequence[Path] = (),
) -> str:
    _require_git_integrity()
    if _git("cat-file", "-t", tag).stdout.strip() != "tag":
        raise ProtocolRefusal(f"local tag {tag} is not annotated")
    local_object = _git("rev-parse", f"refs/tags/{tag}").stdout.strip()
    local_commit = _git("rev-parse", f"{tag}^{{}}").stdout.strip()
    if (local_object, local_commit) != _remote_tag_ids(tag):
        raise ProtocolRefusal(f"public tag {tag} differs from the local tag")
    for path in paths:
        published = _git("show", f"{tag}:{_relative(path)}", text=False).stdout
        if published != path.read_bytes():
            raise ProtocolRefusal(f"{_relative(path)} differs from public tag {tag}")
    for path in absent_paths:
        if (
            _git("cat-file", "-e", f"{tag}:{_relative(path)}", check=False).returncode
            == 0
        ):
            raise ProtocolRefusal(
                f"{_relative(path)} existed prematurely at public tag {tag}"
            )
    return local_commit


def _require_ancestor(ancestor: str, descendant: str) -> None:
    if (
        ancestor == descendant
        or _git(
            "merge-base", "--is-ancestor", ancestor, descendant, check=False
        ).returncode
    ):
        raise ProtocolRefusal("campaign tags do not form the required ancestry")


def _candidate_commit() -> str:
    commit = _require_public_tag(
        CANDIDATE_TAG,
        (DESIGNATION, PREFLIGHT, PROTOCOL),
        absent_paths=_campaign_artifact_paths(),
    )
    if commit != CANDIDATE_COMMIT:
        raise ProtocolRefusal("candidate tag differs from the frozen commit")
    return commit


def _implementation_commit() -> str:
    candidate = _candidate_commit()
    paths = tuple(ROOT / relative for relative in IMPLEMENTATION_BINDINGS)
    commit = _require_public_tag(
        IMPLEMENTATION_TAG,
        paths,
        absent_paths=_campaign_artifact_paths(),
    )
    _require_ancestor(candidate, commit)
    return commit


def _binding_sha256() -> dict[str, str]:
    bindings = {}
    for relative in IMPLEMENTATION_BINDINGS:
        path = ROOT / relative
        if not path.is_file():
            raise ProtocolRefusal(f"implementation file is absent: {relative}")
        bindings[relative] = _sha256(path)
    return bindings


def _verify_implementation_payload(payload: Mapping[str, Any]) -> str:
    implementation = _implementation_commit()
    if payload.get("candidate_commit") != CANDIDATE_COMMIT:
        raise ProtocolRefusal("authorization candidate commit differs")
    if payload.get("public_implementation_commit") != implementation:
        raise ProtocolRefusal("authorization implementation commit differs")
    if payload.get("binding_sha256") != _binding_sha256():
        raise ProtocolRefusal("authorization implementation bindings differ")
    if (
        payload.get("protocol_sha256") != _sha256(PROTOCOL)
        or payload.get("designation_sha256") != _sha256(DESIGNATION)
        or payload.get("metadata_preflight_sha256") != _sha256(PREFLIGHT)
    ):
        raise ProtocolRefusal("authorization frozen artifact bindings differ")
    return implementation


def _verify_authorization(stage: str) -> tuple[dict[str, Any], str]:
    config = STAGE_PATHS[stage]
    path = config["authorization"]
    payload = _read_json(path)
    if (
        payload.get("schema") != config["authorization_schema"]
        or payload.get("status") != config["authorization_status"]
    ):
        raise ProtocolRefusal("authorization schema or status differs")
    commit = _require_public_tag(
        config["authorization_tag"],
        (path,),
        absent_paths=_forbidden_at_boundary(stage, "authorization"),
    )
    implementation = _verify_implementation_payload(payload)
    _require_ancestor(implementation, commit)
    return payload, commit


def _preflight() -> dict[str, Any]:
    payload = _read_json(PREFLIGHT)
    designation = _read_json(DESIGNATION)
    if (
        payload.get("schema") != "gse252762-celiac-metadata-preflight/1.0"
        or payload.get("status") != "PASS"
        or payload.get("numeric_matrix_gets") != 0
        or designation.get("schema") != "gse252762-celiac-candidate-designation/2.0"
        or designation.get("status") != "CANDIDATE_FROZEN_BEFORE_NUMERIC_MATRIX_ACCESS"
        or designation.get("accession") != "GSE252762"
        or designation.get("numeric_matrix_gets_before_designation") != 0
        or designation.get("v1_numeric_matrix_gets_before_supersession") != 0
        or designation.get("metadata_preflight_path")
        != "results/development/gse252762_celiac_metadata_preflight_v1.json"
        or designation.get("protocol_path")
        != "docs/GSE252762_CELIAC_CITESEQ_HELD_BATCH_PROTOCOL_V2_2026-08-30.md"
        or designation.get("metadata_preflight_sha256") != _sha256(PREFLIGHT)
        or designation.get("protocol_sha256") != _sha256(PROTOCOL)
    ):
        raise ProtocolRefusal("candidate or metadata preflight differs from the freeze")
    _validate_preflight_semantics(payload)
    return payload


def _samples(
    preflight: Mapping[str, Any], roles: set[str], *, batch: int | None = None
) -> list[dict[str, Any]]:
    raw = preflight.get("samples")
    if not isinstance(raw, list):
        raise ProtocolRefusal("preflight sample table is absent")
    selected = [
        sample
        for sample in raw
        if isinstance(sample, dict)
        and sample.get("role") in roles
        and (batch is None or sample.get("batch") == batch)
    ]
    if any(
        len(sample.get("selected_columns_1_based", [])) != 256 for sample in selected
    ):
        raise ProtocolRefusal("a selected sample lacks 256 frozen columns")
    return selected


def _batch_record(preflight: Mapping[str, Any], batch: int) -> dict[str, Any]:
    records = [
        record
        for record in preflight.get("batches", [])
        if isinstance(record, dict) and record.get("batch") == batch
    ]
    if len(records) != 1:
        raise ProtocolRefusal("preflight batch record is not unique")
    return records[0]


def _matrix_record(batch_record: Mapping[str, Any], modality: str) -> dict[str, Any]:
    suffix = f"_{modality}_matrix.mtx.gz"
    records = [
        record
        for record in batch_record.get("files", [])
        if isinstance(record, dict) and str(record.get("name", "")).endswith(suffix)
    ]
    if len(records) != 1 or records[0].get("sha256") is not None:
        raise ProtocolRefusal("numeric matrix record differs from the unopened freeze")
    return records[0]


def _validate_preflight_semantics(preflight: Mapping[str, Any]) -> None:
    markers = preflight.get("markers")
    if (
        preflight.get("accession") != "GSE252762"
        or preflight.get("cell_budget") != 256
        or preflight.get("cell_selection_salt") != core.CELL_SALT
        or preflight.get("role_counts") != {"calibration": 9, "pilot": 7, "held": 13}
        or not isinstance(markers, list)
        or [(row.get("rna"), row.get("adt")) for row in markers]
        != list(core.MARKER_PAIRS)
    ):
        raise ProtocolRefusal("preflight cohort or marker contract differs")
    batches = preflight.get("batches")
    if not isinstance(batches, list) or [row.get("batch") for row in batches] != list(
        range(1, 7)
    ):
        raise ProtocolRefusal("preflight batch axis differs")
    samples = preflight.get("samples")
    if not isinstance(samples, list) or len(samples) != 29:
        raise ProtocolRefusal("preflight donor axis differs")
    sample_ids = [sample.get("sample_id") for sample in samples]
    if len(set(sample_ids)) != 29 or any(
        not isinstance(value, str) for value in sample_ids
    ):
        raise ProtocolRefusal("preflight donor identifiers are not unique strings")
    role_counts = {role: 0 for role in ("calibration", "pilot", "held")}
    selected_by_batch: dict[int, set[int]] = {batch: set() for batch in range(1, 7)}
    shape_by_batch = {record["batch"]: record for record in batches}
    for sample in samples:
        role = sample.get("role")
        batch = sample.get("batch")
        condition = sample.get("condition")
        columns = sample.get("selected_columns_1_based")
        barcodes = sample.get("selected_barcodes")
        if (
            role not in role_counts
            or batch not in shape_by_batch
            or condition not in core.DEPOSITED_CONDITIONS
            or sample.get("context") != core.diagnosis_context(condition)
            or (role == "held") != (batch == 6)
            or not isinstance(columns, list)
            or len(columns) != 256
            or len(set(columns)) != 256
            or not isinstance(barcodes, list)
            or len(barcodes) != 256
            or len(set(barcodes)) != 256
            or min(columns) < 1
            or max(columns) > shape_by_batch[batch]["rna_shape"][1]
        ):
            raise ProtocolRefusal("preflight sample selection differs")
        if selected_by_batch[batch].intersection(columns):
            raise ProtocolRefusal("preflight selected columns overlap between donors")
        selected_by_batch[batch].update(columns)
        role_counts[role] += 1
    if role_counts != {"calibration": 9, "pilot": 7, "held": 13}:
        raise ProtocolRefusal("preflight role counts differ")
    for batch in batches:
        if (
            batch.get("rna_shape", [None])[0] != 36_601
            or batch.get("cite_shape", [None])[0] != 204
            or batch.get("rna_shape", [None, None])[1]
            != batch.get("cite_shape", [None, None])[1]
        ):
            raise ProtocolRefusal("preflight matrix shapes differ")
        for modality in ("rna", "cite"):
            record = _matrix_record(batch, modality)
            expected_name = f"GSE252762_batch{batch['batch']}_{modality}_matrix.mtx.gz"
            if (
                record.get("name") != expected_name
                or record.get("url")
                != "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE252nnn/"
                f"GSE252762/suppl/{expected_name}"
                or not isinstance(record.get("bytes"), int)
                or record["bytes"] <= 0
            ):
                raise ProtocolRefusal("preflight numeric matrix record differs")


def _open_url(request: urllib.request.Request, timeout: int) -> Any:
    return _URL_OPENER.open(request, timeout=timeout)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            raise ProtocolRefusal("access journal contains a blank record")
        value = json.loads(
            line,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
        if not isinstance(value, dict):
            raise ProtocolRefusal("access journal record is not an object")
        records.append(value)
    if not records:
        raise ProtocolRefusal("access journal is empty")
    return records


def _digest_is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _expected_matrix_accesses(
    preflight: Mapping[str, Any], stage: str
) -> list[tuple[int, str, list[dict[str, Any]]]]:
    if stage == "source":
        return [
            (
                batch,
                modality,
                _samples(preflight, {"calibration", "pilot"}, batch=batch),
            )
            for batch in range(1, 6)
            for modality in ("rna", "cite")
        ]
    if stage == "prediction":
        return [(6, "rna", _held_samples(preflight))]
    if stage == "score":
        return [(6, "cite", _held_samples(preflight))]
    raise ValueError("unknown stage")


@dataclass
class _AccessLease:
    stage: str
    execution_id: str
    expected: tuple[tuple[int, str], ...]
    position: int = 0
    active: bool = True


def _sidecar_specs(
    preflight: Mapping[str, Any], stage: str
) -> list[tuple[int, str, list[dict[str, Any]], Path]]:
    accesses = _expected_matrix_accesses(preflight, stage)
    paths = tuple(STAGE_PATHS[stage]["sidecars"])
    if len(paths) != len(accesses):
        raise ProtocolRefusal("selected-count sidecar axis differs from access budget")
    return [
        (batch, modality, samples, path)
        for (batch, modality, samples), path in zip(accesses, paths)
    ]


def _sidecar_spec(
    preflight: Mapping[str, Any], stage: str, batch: int, modality: str
) -> tuple[int, list[dict[str, Any]], Path]:
    matches = [
        (index, samples, path)
        for index, (current_batch, current_modality, samples, path) in enumerate(
            _sidecar_specs(preflight, stage)
        )
        if (current_batch, current_modality) == (batch, modality)
    ]
    if len(matches) != 1:
        raise ProtocolRefusal("matrix access lacks one selected-count sidecar")
    return matches[0]


def _sidecar_payload(
    preflight: Mapping[str, Any],
    lease: _AccessLease,
    batch: int,
    modality: str,
    selected: np.ndarray,
    audit: Mapping[str, Any],
    journal: Path,
) -> tuple[Path, dict[str, Any]]:
    index, _samples_for_access, path = _sidecar_spec(
        preflight, lease.stage, batch, modality
    )
    if index != lease.position - 1:
        raise ProtocolRefusal("selected-count sidecar order differs from access lease")
    prefix = _read_jsonl(journal)
    expected_prefix_records = 2 * index + 2
    if len(prefix) != expected_prefix_records:
        raise ProtocolRefusal("selected-count sidecar journal prefix differs")
    prefix_bytes = b"".join(
        (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        for record in prefix
    )
    return path, {
        "schema": "gse252762-celiac-selected-count-sidecar/2.0",
        "stage": lease.stage,
        "batch": batch,
        "modality": modality,
        "execution_id": lease.execution_id,
        "selected_block": selected.tolist(),
        "selected_block_sha256": _array_sha256(selected),
        "matrix_audit": dict(audit),
        "access_journal_path": _relative(journal),
        "access_journal_prefix_records": expected_prefix_records,
        "access_journal_prefix_sha256": _sha256_bytes(prefix_bytes),
    }


def _validate_sidecar(
    preflight: Mapping[str, Any], stage: str, index: int, path: Path
) -> tuple[np.ndarray, dict[str, Any]]:
    batch, modality, samples, expected_path = _sidecar_specs(preflight, stage)[index]
    if path != expected_path or not path.is_file():
        raise ProtocolRefusal("selected-count sidecar path differs")
    payload = _read_json(path)
    journal = STAGE_PATHS[stage]["journal"]
    raw = np.asarray(payload.get("selected_block"))
    selected = np.asarray(payload.get("selected_block"), dtype=np.int64)
    audit = payload.get("matrix_audit")
    expected_prefix_records = 2 * index + 2
    if (
        payload.get("schema") != "gse252762-celiac-selected-count-sidecar/2.0"
        or payload.get("stage") != stage
        or payload.get("batch") != batch
        or payload.get("modality") != modality
        or payload.get("execution_id")
        != _read_json(STAGE_PATHS[stage]["consumption"]).get("execution_id")
        or raw.dtype.kind not in "iu"
        or selected.shape != (9, 256 * len(samples))
        or np.any(selected < 0)
        or payload.get("selected_block_sha256") != _array_sha256(selected)
        or not isinstance(audit, Mapping)
        or payload.get("access_journal_path") != _relative(journal)
        or payload.get("access_journal_prefix_records") != expected_prefix_records
        or not journal.is_file()
    ):
        raise ProtocolRefusal("selected-count sidecar contract differs")
    records = _read_jsonl(journal)
    if len(records) < expected_prefix_records:
        raise ProtocolRefusal("selected-count sidecar journal prefix is absent")
    prefix_bytes = b"".join(
        (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        for record in records[:expected_prefix_records]
    )
    if payload.get("access_journal_prefix_sha256") != _sha256_bytes(prefix_bytes):
        raise ProtocolRefusal("selected-count sidecar journal prefix differs")
    _validate_audit_certificate(audit, preflight, batch, modality, samples)
    if audit.get("selected_block_sha256") != _array_sha256(selected) or audit.get(
        "selected_value_sum"
    ) != int(selected.sum(dtype=np.int64)):
        raise ProtocolRefusal("selected-count sidecar audit does not bind raw counts")
    return selected, dict(audit)


def _existing_sidecar_bindings(stage: str) -> list[dict[str, str]]:
    bindings = []
    absent_seen = False
    for path in STAGE_PATHS[stage]["sidecars"]:
        if path.is_file():
            if absent_seen:
                raise ProtocolRefusal("selected-count sidecars are not a prefix")
            bindings.append({"path": _relative(path), "sha256": _sha256(path)})
        else:
            absent_seen = True
    return bindings


def _complete_sidecar_bindings(stage: str) -> list[dict[str, str]]:
    bindings = _existing_sidecar_bindings(stage)
    if len(bindings) != len(STAGE_PATHS[stage]["sidecars"]):
        raise ProtocolRefusal("successful stage lacks selected-count sidecars")
    return bindings


def _validate_sidecar_bindings(
    preflight: Mapping[str, Any],
    stage: str,
    payload: Mapping[str, Any],
    *,
    require_all: bool,
) -> tuple[Path, ...]:
    expected = _existing_sidecar_bindings(stage)
    if payload.get("selected_count_sidecars") != expected:
        raise ProtocolRefusal("selected-count sidecar bindings differ")
    paths = tuple(STAGE_PATHS[stage]["sidecars"][: len(expected)])
    if require_all and len(paths) != len(STAGE_PATHS[stage]["sidecars"]):
        raise ProtocolRefusal("successful stage lacks selected-count sidecars")
    for index, path in enumerate(paths):
        _validate_sidecar(preflight, stage, index, path)
    return paths


def _assert_access_lease(
    lease: _AccessLease,
    preflight: Mapping[str, Any],
    batch: int,
    modality: str,
    journal: Path,
) -> None:
    if not lease.active or lease.stage not in STAGE_PATHS:
        raise ProtocolRefusal("matrix access lacks an active stage lease")
    config = STAGE_PATHS[lease.stage]
    if journal != config["journal"] or lease.position >= len(lease.expected):
        raise ProtocolRefusal("matrix access exceeds its stage lease")
    if lease.expected[lease.position] != (batch, modality):
        raise ProtocolRefusal("matrix access order differs from its stage lease")
    consumption = _read_json(config["consumption"])
    if consumption.get("execution_id") != lease.execution_id:
        raise ProtocolRefusal("matrix access lease differs from consumption")
    blocked = list(_forbidden_at_boundary(lease.stage, "consumption"))
    blocked.remove(config["journal"])
    for path in config["sidecars"]:
        blocked.remove(path)
    if any(path.exists() for path in blocked):
        raise ProtocolRefusal("matrix access found a premature outcome artifact")
    for index, path in enumerate(config["sidecars"]):
        if path.exists() != (index < lease.position):
            raise ProtocolRefusal(
                "selected-count sidecar prefix differs from access lease"
            )
    records = _read_jsonl(journal)
    if records[0].get("execution_id") != lease.execution_id:
        raise ProtocolRefusal("matrix access journal differs from its lease")
    if any(record.get("event") == "MATRIX_ACCESS_FAILED" for record in records[1:]):
        raise ProtocolRefusal("matrix access cannot continue after a failed request")
    _validate_access_journal(
        lease.stage,
        preflight,
        terminal_failure=True,
    )
    lease.position += 1


def _assert_access_complete(lease: _AccessLease) -> None:
    if not lease.active or lease.position != len(lease.expected):
        raise ProtocolRefusal("stage did not consume its exact matrix-access budget")


def _validate_audit_certificate(
    audit: Mapping[str, Any],
    preflight: Mapping[str, Any],
    batch: int,
    modality: str,
    samples: Sequence[Mapping[str, Any]],
) -> None:
    batch_record = _batch_record(preflight, batch)
    matrix_record = _matrix_record(batch_record, modality)
    row_field = "rna_row_1_based" if modality == "rna" else "cite_row_1_based"
    rows = [marker[row_field] for marker in preflight["markers"]]
    columns = [
        column for sample in samples for column in sample["selected_columns_1_based"]
    ]
    integer_fields = (
        "declared_nnz",
        "parsed_nnz",
        "comment_lines",
        "blank_lines",
        "zero_value_entries",
        "selected_entries",
        "selected_distinct_coordinates",
        "selected_duplicate_entries",
        "global_value_sum",
        "selected_value_sum",
        "compressed_bytes",
        "decompressed_bytes",
    )
    if (
        audit.get("banner")
        not in {
            "%%MatrixMarket matrix coordinate integer general",
            "%%MatrixMarket matrix coordinate real general",
        }
        or audit.get("matrix_shape") != batch_record[f"{modality}_shape"]
        or audit.get("selected_rows") != rows
        or audit.get("selected_columns") != columns
        or any(
            isinstance(audit.get(field), bool)
            or not isinstance(audit.get(field), int)
            or audit[field] < 0
            for field in integer_fields
        )
        or audit.get("declared_nnz") != audit.get("parsed_nnz")
        or audit.get("selected_entries", 0) > audit.get("parsed_nnz", -1)
        or audit.get("zero_value_entries", 0) > audit.get("parsed_nnz", -1)
        or audit.get("selected_entries", -1)
        < audit.get("selected_distinct_coordinates", 0)
        or audit.get("selected_distinct_coordinates", 0) > len(rows) * len(columns)
        or audit.get("selected_duplicate_entries")
        != audit.get("selected_entries", 0)
        - audit.get("selected_distinct_coordinates", 0)
        or audit.get("selected_value_sum", 0) > audit.get("global_value_sum", -1)
        or audit.get("compressed_bytes") != matrix_record["bytes"]
        or audit.get("decompressed_bytes", 0) <= 0
        or not _digest_is_sha256(audit.get("compressed_sha256"))
        or not _digest_is_sha256(audit.get("decompressed_sha256"))
        or not _digest_is_sha256(audit.get("selected_block_sha256"))
        or audit.get("compressed_source_exhausted") is not True
        or audit.get("gzip_stream_exhausted") is not True
        or audit.get("output_dtype") != "int64"
    ):
        raise ProtocolRefusal("matrix reduction certificate differs")


def _validate_access_journal(
    stage: str,
    preflight: Mapping[str, Any],
    *,
    audits: Sequence[Mapping[str, Any]] | None = None,
    terminal_failure: bool = False,
) -> None:
    config = STAGE_PATHS[stage]
    if not config["journal"].is_file():
        if terminal_failure:
            return
        raise ProtocolRefusal("successful stage lacks an access journal")
    records = _read_jsonl(config["journal"])
    header = records[0]
    if (
        header.get("schema") != "gse252762-celiac-access-journal/2.0"
        or header.get("stage") != stage
        or header.get("event") != "OPENED_BEFORE_MATRIX_ACCESS"
        or header.get("attempt_sha256") != _sha256(config["attempt"])
        or header.get("consumption_path") != _relative(config["consumption"])
        or header.get("consumption_sha256") != _sha256(config["consumption"])
        or header.get("execution_id")
        != _read_json(config["consumption"]).get("execution_id")
        or not _git_sha_is_valid(header.get("public_consumption_commit"))
        or header.get("automatic_retries") is not False
        or header.get("http_redirects") is not False
    ):
        raise ProtocolRefusal("access journal header differs")
    expected = _expected_matrix_accesses(preflight, stage)
    events = records[1:]
    finished_audits = []
    position = 0
    access_index = 0
    saw_failure = False
    while position < len(events):
        if access_index >= len(expected):
            raise ProtocolRefusal("access journal exceeds the frozen GET budget")
        batch, modality, samples = expected[access_index]
        matrix = _matrix_record(_batch_record(preflight, batch), modality)
        started = events[position]
        if (
            started.get("event") != "MATRIX_ACCESS_STARTED"
            or started.get("batch") != batch
            or started.get("modality") != modality
            or started.get("url") != matrix["url"]
            or started.get("execution_id") != header.get("execution_id")
        ):
            raise ProtocolRefusal("access journal matrix order differs")
        position += 1
        access_index += 1
        if position == len(events):
            if terminal_failure:
                break
            raise ProtocolRefusal("successful access journal ends after a GET start")
        terminal = events[position]
        if any(
            terminal.get(field) != started.get(field)
            for field in ("batch", "modality", "url", "execution_id")
        ):
            raise ProtocolRefusal(
                "access journal terminal event differs from its start"
            )
        if terminal.get("event") == "MATRIX_REDUCTION_FINISHED":
            audit = terminal.get("audit")
            if not isinstance(audit, Mapping):
                raise ProtocolRefusal("access journal lacks a matrix certificate")
            sidecar_path = STAGE_PATHS[stage]["sidecars"][access_index - 1]
            if (
                terminal.get("selected_sidecar_path") != _relative(sidecar_path)
                or not sidecar_path.is_file()
                or terminal.get("selected_sidecar_sha256") != _sha256(sidecar_path)
            ):
                raise ProtocolRefusal("access journal sidecar binding differs")
            _selected, sidecar_audit = _validate_sidecar(
                preflight, stage, access_index - 1, sidecar_path
            )
            if dict(audit) != sidecar_audit:
                raise ProtocolRefusal("access journal audit differs from raw sidecar")
            _validate_audit_certificate(audit, preflight, batch, modality, samples)
            finished_audits.append(dict(audit))
        elif terminal.get("event") == "MATRIX_ACCESS_FAILED" and terminal_failure:
            if not isinstance(terminal.get("exception_class"), str):
                raise ProtocolRefusal("access failure lacks an exception class")
            if STAGE_PATHS[stage]["sidecars"][access_index - 1].exists():
                raise ProtocolRefusal("failed matrix access has a raw-count sidecar")
            saw_failure = True
        else:
            raise ProtocolRefusal("access journal terminal event differs")
        position += 1
        if saw_failure and position != len(events):
            raise ProtocolRefusal("matrix access continued after a terminal failure")
    if not terminal_failure and access_index != len(expected):
        raise ProtocolRefusal("successful access journal is incomplete")
    if audits is not None:
        normalized = [
            {
                key: value
                for key, value in audit.items()
                if key not in {"batch", "modality"}
            }
            for audit in audits
        ]
        if normalized != finished_audits:
            raise ProtocolRefusal(
                "published matrix audits differ from the access journal"
            )


def _reduce_matrix(
    preflight: Mapping[str, Any],
    batch: int,
    modality: str,
    samples: Sequence[Mapping[str, Any]],
    journal: Path,
    lease: _AccessLease,
) -> tuple[np.ndarray, dict[str, Any]]:
    batch_record = _batch_record(preflight, batch)
    record = _matrix_record(batch_record, modality)
    markers = preflight.get("markers")
    if not isinstance(markers, list) or len(markers) != 9:
        raise ProtocolRefusal("frozen marker axis is absent")
    row_field = "rna_row_1_based" if modality == "rna" else "cite_row_1_based"
    rows = [marker[row_field] for marker in markers]
    columns = [
        column for sample in samples for column in sample["selected_columns_1_based"]
    ]
    _assert_access_lease(lease, preflight, batch, modality, journal)
    event = {
        "timestamp_utc": _timestamp(),
        "batch": batch,
        "modality": modality,
        "url": record["url"],
        "execution_id": lease.execution_id,
    }
    _append_journal(journal, {**event, "event": "MATRIX_ACCESS_STARTED"})
    try:
        request = urllib.request.Request(
            record["url"],
            headers={
                "User-Agent": "coupling-fields/1",
                "Accept-Encoding": "identity",
            },
        )
        with _open_url(request, timeout=180) as response:
            encoding = response.headers.get("Content-Encoding")
            if response.geturl() != record["url"] or response.status != 200:
                raise ProtocolRefusal("matrix response URL or status differs")
            if encoding not in (None, "identity"):
                raise ProtocolRefusal("matrix response content encoding differs")
            length = response.headers.get("Content-Length")
            if length is None or int(length) != int(record["bytes"]):
                raise ProtocolRefusal("matrix response length differs from the freeze")
            block, audit = reduce_gzip_matrix_market(
                response,
                expected_shape=batch_record[f"{modality}_shape"],
                selected_rows=rows,
                selected_columns=columns,
                allow_integral_real=True,
            )
        if audit.compressed_bytes != int(record["bytes"]):
            raise ProtocolRefusal("complete matrix byte count differs")
    except BaseException as error:
        partial = getattr(error, "partial_audit", None)
        _append_journal(
            journal,
            {
                **event,
                "event": "MATRIX_ACCESS_FAILED",
                "exception_class": type(error).__name__,
                "partial_audit": asdict(partial) if partial is not None else None,
            },
        )
        raise
    summary = _json_value(asdict(audit))
    summary["selected_block_sha256"] = _array_sha256(block)
    if summary["selected_value_sum"] != int(block.sum(dtype=np.int64)):
        raise ProtocolRefusal("matrix certificate selected sum differs from its block")
    sidecar_path, sidecar = _sidecar_payload(
        preflight, lease, batch, modality, block, summary, journal
    )
    _write_json(sidecar_path, sidecar)
    _validate_sidecar(preflight, lease.stage, lease.position - 1, sidecar_path)
    _append_journal(
        journal,
        {
            **event,
            "event": "MATRIX_REDUCTION_FINISHED",
            "audit": summary,
            "selected_sidecar_path": _relative(sidecar_path),
            "selected_sidecar_sha256": _sha256(sidecar_path),
        },
    )
    return block, summary


def _reduce_source(preflight: Mapping[str, Any], lease: _AccessLease) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for batch in range(1, 6):
        batch_samples = _samples(preflight, {"calibration", "pilot"}, batch=batch)
        if not batch_samples:
            continue
        rna, rna_audit = _reduce_matrix(
            preflight, batch, "rna", batch_samples, SOURCE_JOURNAL, lease
        )
        cite, cite_audit = _reduce_matrix(
            preflight, batch, "cite", batch_samples, SOURCE_JOURNAL, lease
        )
        audits.extend(
            [
                {"batch": batch, "modality": "rna", **rna_audit},
                {"batch": batch, "modality": "cite", **cite_audit},
            ]
        )
        for index, sample in enumerate(batch_samples):
            start = 256 * index
            stop = start + 256
            rna_counts = rna[:, start:stop].T
            cite_counts = cite[:, start:stop].T
            truth, destroyed = core.sample_tables(
                rna_counts,
                cite_counts,
                sample["selected_barcodes"],
                sample["sample_id"],
            )
            records.append(
                {
                    "sample_id": sample["sample_id"],
                    "role": sample["role"],
                    "condition": sample["condition"],
                    "context": sample["context"],
                    "rna_counts": rna_counts.tolist(),
                    "rna_counts_sha256": _array_sha256(rna_counts),
                    "cite_counts": cite_counts.tolist(),
                    "cite_counts_sha256": _array_sha256(cite_counts),
                    "truth_tables": truth.tolist(),
                    "destroyed_tables": destroyed.tolist(),
                    "truth_sha256": _array_sha256(truth),
                    "destroyed_sha256": _array_sha256(destroyed),
                }
            )
    expected = {
        sample["sample_id"] for sample in _samples(preflight, {"calibration", "pilot"})
    }
    if {record["sample_id"] for record in records} != expected or len(records) != 16:
        raise ProtocolRefusal("source reduction did not produce the frozen 16 samples")
    role_order = {"calibration": 0, "pilot": 1}
    records.sort(key=lambda record: (role_order[record["role"]], record["sample_id"]))
    return {
        "schema": "gse252762-celiac-source-reduced/2.0",
        "status": "SOURCE_REDUCTION_COMPLETE",
        "samples": records,
        "matrix_audits": audits,
        "access_journal_path": _relative(SOURCE_JOURNAL),
        "access_journal_sha256": _sha256(SOURCE_JOURNAL),
    }


def _arrays(
    reduced: Mapping[str, Any], roles: set[str]
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    records = [record for record in reduced["samples"] if record["role"] in roles]
    return (
        np.asarray([record["truth_tables"] for record in records], dtype=np.int64),
        np.asarray([record["destroyed_tables"] for record in records], dtype=np.int64),
        [record["context"] for record in records],
        [record["condition"] for record in records],
        [record["sample_id"] for record in records],
    )


def _validate_source_reduced(
    reduced: Mapping[str, Any], preflight: Mapping[str, Any]
) -> None:
    if (
        reduced.get("schema") != "gse252762-celiac-source-reduced/2.0"
        or reduced.get("status") != "SOURCE_REDUCTION_COMPLETE"
        or reduced.get("access_journal_path") != _relative(SOURCE_JOURNAL)
        or not SOURCE_JOURNAL.is_file()
        or reduced.get("access_journal_sha256") != _sha256(SOURCE_JOURNAL)
    ):
        raise ProtocolRefusal("source reduction header or journal binding differs")
    frozen = _samples(preflight, {"calibration", "pilot"})
    role_order = {"calibration": 0, "pilot": 1}
    frozen.sort(key=lambda row: (role_order[row["role"]], row["sample_id"]))
    records = reduced.get("samples")
    if not isinstance(records, list) or len(records) != 16:
        raise ProtocolRefusal("source reduction sample count differs")
    for record, sample in zip(records, frozen):
        if any(
            record.get(field) != sample.get(field)
            for field in ("sample_id", "role", "condition", "context")
        ):
            raise ProtocolRefusal("source reduction sample metadata differs")
        truth_raw = np.asarray(record.get("truth_tables"))
        destroyed_raw = np.asarray(record.get("destroyed_tables"))
        rna_raw = np.asarray(record.get("rna_counts"))
        cite_raw = np.asarray(record.get("cite_counts"))
        truth = np.asarray(record.get("truth_tables"), dtype=np.int64)
        destroyed = np.asarray(record.get("destroyed_tables"), dtype=np.int64)
        rna = np.asarray(record.get("rna_counts"), dtype=np.int64)
        cite = np.asarray(record.get("cite_counts"), dtype=np.int64)
        replay_truth, replay_destroyed = core.sample_tables(
            rna,
            cite,
            sample["selected_barcodes"],
            sample["sample_id"],
        )
        if (
            rna.shape != (256, 9)
            or cite.shape != (256, 9)
            or rna_raw.dtype.kind not in "iu"
            or cite_raw.dtype.kind not in "iu"
            or np.any(rna < 0)
            or np.any(cite < 0)
            or _array_sha256(rna) != record.get("rna_counts_sha256")
            or _array_sha256(cite) != record.get("cite_counts_sha256")
            or truth.shape != (9, 9, 2, 2)
            or destroyed.shape != truth.shape
            or truth_raw.dtype.kind not in "iu"
            or destroyed_raw.dtype.kind not in "iu"
            or _array_sha256(truth) != record.get("truth_sha256")
            or _array_sha256(destroyed) != record.get("destroyed_sha256")
            or np.any(truth < 0)
            or np.any(destroyed < 0)
            or not np.array_equal(truth, replay_truth)
            or not np.array_equal(destroyed, replay_destroyed)
            or np.any(truth.sum(axis=(-2, -1)) != 256)
            or not np.array_equal(truth.sum(axis=-1), destroyed.sum(axis=-1))
            or not np.array_equal(truth.sum(axis=-2), destroyed.sum(axis=-2))
        ):
            raise ProtocolRefusal("source reduction table contract differs")
    audits = reduced.get("matrix_audits")
    expected_audits = []
    for batch in range(1, 6):
        batch_samples = _samples(preflight, {"calibration", "pilot"}, batch=batch)
        if not batch_samples:
            continue
        batch_record = _batch_record(preflight, batch)
        for modality in ("rna", "cite"):
            marker_field = (
                "rna_row_1_based" if modality == "rna" else "cite_row_1_based"
            )
            expected_audits.append(
                {
                    "batch": batch,
                    "modality": modality,
                    "rows": [row[marker_field] for row in preflight["markers"]],
                    "columns": [
                        column
                        for sample in batch_samples
                        for column in sample["selected_columns_1_based"]
                    ],
                    "shape": batch_record[f"{modality}_shape"],
                    "bytes": _matrix_record(batch_record, modality)["bytes"],
                }
            )
    if not isinstance(audits, list) or len(audits) != len(expected_audits):
        raise ProtocolRefusal("source matrix audit count differs")
    record_by_id = {record["sample_id"]: record for record in records}
    for audit, expected in zip(audits, expected_audits):
        if (
            audit.get("batch") != expected["batch"]
            or audit.get("modality") != expected["modality"]
        ):
            raise ProtocolRefusal("source matrix audit differs from the frozen axes")
        _validate_audit_certificate(
            audit,
            preflight,
            expected["batch"],
            expected["modality"],
            _samples(
                preflight,
                {"calibration", "pilot"},
                batch=expected["batch"],
            ),
        )
        batch_samples = _samples(
            preflight,
            {"calibration", "pilot"},
            batch=expected["batch"],
        )
        count_field = f"{expected['modality']}_counts"
        selected_block = np.concatenate(
            [
                np.asarray(
                    record_by_id[sample["sample_id"]][count_field], dtype=np.int64
                ).T
                for sample in batch_samples
            ],
            axis=1,
        )
        if audit.get("selected_block_sha256") != _array_sha256(
            selected_block
        ) or audit.get("selected_value_sum") != int(selected_block.sum(dtype=np.int64)):
            raise ProtocolRefusal("source matrix audit does not bind selected counts")
    _validate_access_journal("source", preflight, audits=audits)


def _source_selection(reduced: Mapping[str, Any]) -> dict[str, Any]:
    calibration, calibration_destroyed, calibration_contexts, _, calibration_ids = (
        _arrays(reduced, {"calibration"})
    )
    pilot, pilot_destroyed, pilot_contexts, _, pilot_ids = _arrays(reduced, {"pilot"})
    if len(calibration_ids) != 9 or len(pilot_ids) != 7:
        raise ProtocolRefusal(
            "source split differs from nine calibration and seven pilot"
        )
    selection = _json_value(
        core.select_source(
            calibration,
            calibration_destroyed,
            calibration_contexts,
            pilot,
            pilot_destroyed,
            pilot_contexts,
        )
    )
    _validate_v2_selection(selection)
    return selection


def _validate_v2_selection(selection: Mapping[str, Any]) -> None:
    alphas = selection.get("selected_comparator_alphas")
    models = selection.get("calibration_models")
    pilot_losses = selection.get("pilot_sample_losses")
    gate = selection.get("pilot_promotion_gate")
    strongest = selection.get("strongest_benchmark")
    if (
        selection.get("schema") != "gse252762-celiac-source-selection/2.0"
        or strongest not in core.BENCHMARK_TIE_ORDER
        or not isinstance(alphas, Mapping)
        or set(alphas) != set(core.CLASSICAL_METHODS)
        or "donor_stratified_ridge_poisson" not in alphas
        or not isinstance(models, Mapping)
        or models.get("strongest_benchmark") != strongest
        or "donor_stratified_ridge_poisson_field" not in models
        or not isinstance(
            models.get("donor_stratified_ridge_poisson_certificate"), Mapping
        )
        or not isinstance(pilot_losses, Mapping)
        or not set(core.MANDATORY_METHODS).issubset(pilot_losses)
        or not isinstance(gate, Mapping)
        or gate.get("strongest_benchmark") != strongest
    ):
        raise ProtocolRefusal(
            "source selection omits the frozen v2 benchmark interface"
        )


def _replay_source_result(
    preflight: Mapping[str, Any],
    reduced: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_source_reduced(reduced, preflight)
    _validate_sidecar_bindings(preflight, "source", result, require_all=True)
    selection = _source_selection(reduced)
    expected_status = (
        "SOURCE_PASS"
        if selection.get("status") == "PROMOTED"
        else "TERMINAL_SOURCE_GATE_FAIL"
    )
    if (
        result.get("schema") != "gse252762-celiac-source-result/2.0"
        or result.get("status") != expected_status
        or result.get("source_reduced_path") != _relative(SOURCE_REDUCED)
        or result.get("source_reduced_sha256") != _sha256(SOURCE_REDUCED)
        or result.get("source_selection") != selection
        or result.get("access_journal_path") != _relative(SOURCE_JOURNAL)
        or result.get("access_journal_sha256") != _sha256(SOURCE_JOURNAL)
        or result.get("rerun_permitted") is not False
    ):
        raise ProtocolRefusal(
            "source result does not replay from public reduced tables"
        )
    return selection


def _verify_source_public(
    *, require_pass: bool
) -> tuple[dict[str, Any], dict[str, Any], str]:
    _attempt, consumption_commit, _authorization, _consumption = (
        _verify_consumption_public("source")
    )
    commit = _require_public_tag(
        SOURCE_RESULT_TAG,
        (
            SOURCE_ATTEMPT,
            SOURCE_CONSUMPTION,
            SOURCE_JOURNAL,
            *STAGE_PATHS["source"]["sidecars"],
            SOURCE_REDUCED,
            SOURCE_RESULT,
        ),
        absent_paths=_forbidden_at_boundary("source", "result"),
    )
    _require_ancestor(consumption_commit, commit)
    preflight = _preflight()
    reduced = _read_json(SOURCE_REDUCED)
    result = _read_json(SOURCE_RESULT)
    selection = _replay_source_result(preflight, reduced, result)
    if require_pass and result.get("status") != "SOURCE_PASS":
        raise ProtocolRefusal("source did not pass the frozen pilot gate")
    return result, selection, commit


def _held_samples(preflight: Mapping[str, Any]) -> list[dict[str, Any]]:
    samples = _samples(preflight, {"held"}, batch=6)
    if len(samples) != 13:
        raise ProtocolRefusal("held panel requires 13 frozen batch-6 samples")
    return samples


def _checkpoint_binding(stage: str) -> dict[str, str]:
    path_field, hash_field = CHECKPOINT_BINDINGS[stage]
    checkpoint = STAGE_PATHS[stage]["checkpoint"]
    if not checkpoint.is_file():
        raise ProtocolRefusal(f"{stage} reduction checkpoint is absent")
    return {
        path_field: _relative(checkpoint),
        hash_field: _sha256(checkpoint),
    }


def _held_checkpoint_payload(
    preflight: Mapping[str, Any],
    stage: str,
    modality: str,
    selected: np.ndarray,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    if (stage, modality) not in {("prediction", "rna"), ("score", "cite")}:
        raise ValueError("unknown held reduction checkpoint")
    held = _held_samples(preflight)
    records = []
    for index, sample in enumerate(held):
        counts = selected[:, 256 * index : 256 * (index + 1)].T
        records.append(
            {
                "sample_id": sample["sample_id"],
                "condition": sample["condition"],
                "context": sample["context"],
                "counts": counts.tolist(),
                "counts_sha256": _array_sha256(counts),
            }
        )
    return {
        "schema": "gse252762-celiac-held-count-checkpoint/2.0",
        "status": f"HELD_{modality.upper()}_REDUCTION_COMPLETE",
        "stage": stage,
        "modality": modality,
        "samples": records,
        "matrix_audit": dict(audit),
        "access_journal_path": _relative(STAGE_PATHS[stage]["journal"]),
        "access_journal_sha256": _sha256(STAGE_PATHS[stage]["journal"]),
    }


def _validate_held_checkpoint(
    preflight: Mapping[str, Any], stage: str, payload: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    modality = {"prediction": "rna", "score": "cite"}.get(stage)
    if modality is None:
        raise ValueError("source does not use a held-count checkpoint")
    held = _held_samples(preflight)
    records = payload.get("samples")
    journal = STAGE_PATHS[stage]["journal"]
    if (
        payload.get("schema") != "gse252762-celiac-held-count-checkpoint/2.0"
        or payload.get("status") != f"HELD_{modality.upper()}_REDUCTION_COMPLETE"
        or payload.get("stage") != stage
        or payload.get("modality") != modality
        or payload.get("access_journal_path") != _relative(journal)
        or not journal.is_file()
        or payload.get("access_journal_sha256") != _sha256(journal)
        or not isinstance(records, list)
        or len(records) != 13
    ):
        raise ProtocolRefusal("held reduction checkpoint header differs")
    blocks = []
    for record, sample in zip(records, held):
        raw = np.asarray(record.get("counts"))
        counts = np.asarray(record.get("counts"), dtype=np.int64)
        if (
            any(
                record.get(field) != sample.get(field)
                for field in ("sample_id", "condition", "context")
            )
            or raw.dtype.kind not in "iu"
            or counts.shape != (256, 9)
            or np.any(counts < 0)
            or record.get("counts_sha256") != _array_sha256(counts)
        ):
            raise ProtocolRefusal("held reduction checkpoint counts differ")
        blocks.append(counts)
    audit = payload.get("matrix_audit")
    if not isinstance(audit, Mapping):
        raise ProtocolRefusal("held reduction checkpoint audit is absent")
    _validate_audit_certificate(audit, preflight, 6, modality, held)
    selected = np.concatenate([counts.T for counts in blocks], axis=1)
    if audit.get("selected_block_sha256") != _array_sha256(selected) or audit.get(
        "selected_value_sum"
    ) != int(selected.sum(dtype=np.int64)):
        raise ProtocolRefusal("held reduction checkpoint audit does not bind counts")
    _validate_access_journal(stage, preflight, audits=(audit,))
    return np.asarray(blocks, dtype=np.int64), dict(audit)


def _validate_checkpoint_binding(
    preflight: Mapping[str, Any], stage: str, payload: Mapping[str, Any]
) -> bool:
    path_field, hash_field = CHECKPOINT_BINDINGS[stage]
    checkpoint = STAGE_PATHS[stage]["checkpoint"]
    bound = payload.get(path_field) is not None or payload.get(hash_field) is not None
    if not bound:
        if checkpoint.exists():
            raise ProtocolRefusal(f"terminal {stage} checkpoint is not bound")
        return False
    if (
        payload.get(path_field) != _relative(checkpoint)
        or not checkpoint.is_file()
        or payload.get(hash_field) != _sha256(checkpoint)
    ):
        raise ProtocolRefusal(f"terminal {stage} checkpoint binding differs")
    if stage == "source":
        _validate_source_reduced(_read_json(checkpoint), preflight)
    else:
        _validate_held_checkpoint(preflight, stage, _read_json(checkpoint))
    return True


def _all_source(reduced: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    tables, destroyed, contexts, _, sample_ids = _arrays(
        reduced, {"calibration", "pilot"}
    )
    if len(sample_ids) != 16:
        raise ProtocolRefusal("all-source refit requires 16 frozen samples")
    return tables, destroyed, contexts


def _prediction_payload(
    preflight: Mapping[str, Any],
    reduced: Mapping[str, Any],
    selection: Mapping[str, Any],
    rna: np.ndarray,
    audit: Mapping[str, Any],
    source_commit: str,
) -> dict[str, Any]:
    held = _held_samples(preflight)
    states = []
    rows = []
    columns = []
    count_blocks = []
    for index, _sample in enumerate(held):
        counts = rna[:, 256 * index : 256 * (index + 1)].T
        state = core.rna_detection_states(counts)
        row, column = core.rna_margin_tables(counts)
        states.append(state)
        rows.append(row)
        columns.append(column)
        count_blocks.append(counts)
    source, destroyed, contexts = _all_source(reduced)
    prediction_result = core.predict_from_source(
        source,
        destroyed,
        contexts,
        selection,
        np.asarray(rows),
        np.asarray(columns),
        [sample["context"] for sample in held],
        return_fit_report=True,
    )
    if not isinstance(prediction_result, tuple):
        raise AssertionError("all-source prediction omitted its fit report")
    predictions, fit_report = prediction_result
    if (
        not set(core.MANDATORY_METHODS).issubset(predictions)
        or "donor_stratified_ridge_poisson" not in predictions
        or not isinstance(fit_report, Mapping)
        or fit_report.get("strongest_benchmark") != selection.get("strongest_benchmark")
        or not isinstance(
            fit_report.get("donor_stratified_ridge_poisson_certificate"), Mapping
        )
    ):
        raise ProtocolRefusal("all-source fit omits the frozen v2 benchmark outputs")
    sample_payload = []
    for index, sample in enumerate(held):
        sample_payload.append(
            {
                "sample_id": sample["sample_id"],
                "condition": sample["condition"],
                "context": sample["context"],
                "rna_counts": count_blocks[index].tolist(),
                "rna_counts_sha256": _array_sha256(count_blocks[index]),
                "rna_states": states[index].tolist(),
                "rna_states_sha256": _array_sha256(states[index]),
                "row_margins": rows[index].tolist(),
                "column_margins": columns[index].tolist(),
                "predictions": {
                    method: values[index].tolist()
                    for method, values in predictions.items()
                },
            }
        )
    return {
        "schema": "gse252762-celiac-held-predictions/2.0",
        "status": "PREDICTIONS_FROZEN_BEFORE_HELD_CITE_ACCESS",
        "source_result_path": _relative(SOURCE_RESULT),
        "source_result_sha256": _sha256(SOURCE_RESULT),
        "public_source_result_commit": source_commit,
        "held_cite_matrix_gets": 0,
        "samples": sample_payload,
        "all_source_fit_report": _json_value(fit_report),
        "held_rna_audit": dict(audit),
        "access_journal_path": _relative(PREDICTION_JOURNAL),
        "access_journal_sha256": _sha256(PREDICTION_JOURNAL),
    }


def _replay_predictions(
    preflight: Mapping[str, Any],
    reduced: Mapping[str, Any],
    source_result: Mapping[str, Any],
    source_commit: str,
    predictions: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    selection = _replay_source_result(preflight, reduced, source_result)
    if source_result.get("status") != "SOURCE_PASS":
        raise ProtocolRefusal("held prediction replay requires a promoted source")
    held = _held_samples(preflight)
    records = predictions.get("samples")
    if (
        predictions.get("schema") != "gse252762-celiac-held-predictions/2.0"
        or predictions.get("status") != "PREDICTIONS_FROZEN_BEFORE_HELD_CITE_ACCESS"
        or predictions.get("source_result_path") != _relative(SOURCE_RESULT)
        or predictions.get("source_result_sha256") != _sha256(SOURCE_RESULT)
        or predictions.get("public_source_result_commit") != source_commit
        or predictions.get("held_cite_matrix_gets") != 0
        or predictions.get("access_journal_path") != _relative(PREDICTION_JOURNAL)
        or not PREDICTION_JOURNAL.is_file()
        or predictions.get("access_journal_sha256") != _sha256(PREDICTION_JOURNAL)
        or predictions.get("held_rna_reduced_path") != _relative(PREDICTION_REDUCED)
        or not PREDICTION_REDUCED.is_file()
        or predictions.get("held_rna_reduced_sha256") != _sha256(PREDICTION_REDUCED)
        or not isinstance(records, list)
        or len(records) != 13
    ):
        raise ProtocolRefusal("held prediction header or source binding differs")
    _validate_sidecar_bindings(preflight, "prediction", predictions, require_all=True)
    checkpoint_counts, checkpoint_audit = _validate_held_checkpoint(
        preflight, "prediction", _read_json(PREDICTION_REDUCED)
    )
    rows = []
    columns = []
    count_blocks = []
    for index, (record, sample) in enumerate(zip(records, held)):
        if any(
            record.get(field) != sample.get(field)
            for field in ("sample_id", "condition", "context")
        ):
            raise ProtocolRefusal("held prediction sample metadata differs")
        count_raw = np.asarray(record.get("rna_counts"))
        count = np.asarray(record.get("rna_counts"), dtype=np.int64)
        state_raw = np.asarray(record.get("rna_states"))
        state = np.asarray(record.get("rna_states"), dtype=np.uint8)
        if (
            count.shape != (256, 9)
            or count_raw.dtype.kind not in "iu"
            or np.any(count < 0)
            or _array_sha256(count) != record.get("rna_counts_sha256")
            or not np.array_equal(count, checkpoint_counts[index])
            or state_raw.shape != (256, 9)
            or state_raw.dtype.kind not in "iu"
            or not np.isin(state_raw, (0, 1)).all()
            or not np.array_equal(state, core.rna_detection_states(count))
            or _array_sha256(state) != record.get("rna_states_sha256")
        ):
            raise ProtocolRefusal("held RNA state payload differs")
        expected_rows, expected_columns = core.rna_margin_tables(state)
        row_raw = np.asarray(record.get("row_margins"))
        column_raw = np.asarray(record.get("column_margins"))
        row = np.asarray(record.get("row_margins"), dtype=np.int64)
        column = np.asarray(record.get("column_margins"), dtype=np.int64)
        if (
            not np.array_equal(row, expected_rows)
            or not np.array_equal(column, expected_columns)
            or row_raw.dtype.kind not in "iu"
            or column_raw.dtype.kind not in "iu"
        ):
            raise ProtocolRefusal("held prediction margins do not follow RNA states")
        rows.append(row)
        columns.append(column)
        count_blocks.append(count.T)
    audit = predictions.get("held_rna_audit")
    if not isinstance(audit, Mapping) or dict(audit) != checkpoint_audit:
        raise ProtocolRefusal("held RNA audit is absent")
    _validate_held_audit(preflight, audit, "rna")
    selected_block = np.concatenate(count_blocks, axis=1)
    if audit.get("selected_block_sha256") != _array_sha256(selected_block) or audit.get(
        "selected_value_sum"
    ) != int(selected_block.sum(dtype=np.int64)):
        raise ProtocolRefusal("held RNA audit does not bind published counts")
    _validate_access_journal("prediction", preflight, audits=(audit,))
    source, destroyed, contexts = _all_source(reduced)
    replay_result = core.predict_from_source(
        source,
        destroyed,
        contexts,
        selection,
        np.asarray(rows),
        np.asarray(columns),
        [sample["context"] for sample in held],
        return_fit_report=True,
    )
    if not isinstance(replay_result, tuple):
        raise AssertionError("all-source replay omitted its fit report")
    replay, fit_report = replay_result
    if (
        not set(core.MANDATORY_METHODS).issubset(replay)
        or "donor_stratified_ridge_poisson" not in replay
        or not isinstance(fit_report, Mapping)
        or fit_report.get("strongest_benchmark") != selection.get("strongest_benchmark")
        or not isinstance(
            fit_report.get("donor_stratified_ridge_poisson_certificate"), Mapping
        )
    ):
        raise ProtocolRefusal("all-source replay omits the frozen v2 benchmark outputs")
    if predictions.get("all_source_fit_report") != _json_value(fit_report):
        raise ProtocolRefusal("all-source fit report does not replay")
    for index, record in enumerate(records):
        stored = record.get("predictions")
        if not isinstance(stored, dict) or set(stored) != set(replay):
            raise ProtocolRefusal("held prediction method set differs")
        for method, values in replay.items():
            if not np.array_equal(
                np.asarray(stored[method], dtype=float), values[index]
            ):
                raise ProtocolRefusal(
                    "held predictions do not replay from source tables"
                )
    return replay


def _verify_predictions_public() -> tuple[dict[str, Any], dict[str, Any], str, str]:
    _attempt, consumption_commit, _authorization, _consumption = (
        _verify_consumption_public("prediction")
    )
    predictions_commit = _require_public_tag(
        PREDICTIONS_TAG,
        (
            PREDICTION_ATTEMPT,
            PREDICTION_CONSUMPTION,
            PREDICTION_JOURNAL,
            *STAGE_PATHS["prediction"]["sidecars"],
            PREDICTION_REDUCED,
            PREDICTIONS,
        ),
        absent_paths=_forbidden_at_boundary("prediction", "result"),
    )
    _require_ancestor(consumption_commit, predictions_commit)
    source_result, _selection, source_commit = _verify_source_public(require_pass=True)
    preflight = _preflight()
    reduced = _read_json(SOURCE_REDUCED)
    predictions = _read_json(PREDICTIONS)
    _replay_predictions(preflight, reduced, source_result, source_commit, predictions)
    return predictions, source_result, predictions_commit, source_commit


def _require_authorization_slot_clean(stage: str) -> None:
    stage_order = ("source", "prediction", "score")
    paths = []
    for current in stage_order[stage_order.index(stage) :]:
        paths.extend(
            _stage_artifact_paths(
                current,
                (
                    "authorization",
                    "attempt",
                    "consumption",
                    "journal",
                    "sidecars",
                    "checkpoint",
                    "result",
                ),
            )
        )
    if any(path.exists() for path in paths):
        raise ProtocolRefusal(f"{stage} authorization slot is not clean")


def authorize_source() -> dict[str, Any]:
    _require_authorization_slot_clean("source")
    candidate = _candidate_commit()
    implementation = _implementation_commit()
    _preflight()
    payload = {
        "schema": "gse252762-celiac-source-authorization/2.0",
        "status": "SOURCE_AUTHORIZED",
        "candidate_commit": candidate,
        "public_implementation_commit": implementation,
        "binding_sha256": _binding_sha256(),
        "protocol_sha256": _sha256(PROTOCOL),
        "designation_sha256": _sha256(DESIGNATION),
        "metadata_preflight_sha256": _sha256(PREFLIGHT),
        "source_matrix_gets_authorized": 10,
        "held_rna_matrix_gets_authorized": 0,
        "held_cite_matrix_gets_authorized": 0,
        "created_at_utc": _timestamp(),
    }
    _write_json(SOURCE_AUTHORIZATION, payload)
    return payload


def authorize_prediction() -> dict[str, Any]:
    _require_authorization_slot_clean("prediction")
    source, _selection, source_commit = _verify_source_public(require_pass=True)
    implementation = _implementation_commit()
    payload = {
        "schema": "gse252762-celiac-prediction-authorization/2.0",
        "status": "HELD_RNA_AUTHORIZED",
        "candidate_commit": CANDIDATE_COMMIT,
        "public_implementation_commit": implementation,
        "binding_sha256": _binding_sha256(),
        "protocol_sha256": _sha256(PROTOCOL),
        "designation_sha256": _sha256(DESIGNATION),
        "metadata_preflight_sha256": _sha256(PREFLIGHT),
        "source_result_path": _relative(SOURCE_RESULT),
        "source_result_sha256": _sha256(SOURCE_RESULT),
        "source_reduced_path": _relative(SOURCE_REDUCED),
        "source_reduced_sha256": _sha256(SOURCE_REDUCED),
        "public_source_result_commit": source_commit,
        "source_status": source["status"],
        "held_rna_matrix_gets_authorized": 1,
        "held_cite_matrix_gets_authorized": 0,
        "created_at_utc": _timestamp(),
    }
    _write_json(PREDICTION_AUTHORIZATION, payload)
    return payload


def authorize_score() -> dict[str, Any]:
    _require_authorization_slot_clean("score")
    predictions, _source, predictions_commit, source_commit = (
        _verify_predictions_public()
    )
    implementation = _implementation_commit()
    payload = {
        "schema": "gse252762-celiac-score-authorization/2.0",
        "status": "HELD_CITE_SCORE_AUTHORIZED",
        "candidate_commit": CANDIDATE_COMMIT,
        "public_implementation_commit": implementation,
        "binding_sha256": _binding_sha256(),
        "protocol_sha256": _sha256(PROTOCOL),
        "designation_sha256": _sha256(DESIGNATION),
        "metadata_preflight_sha256": _sha256(PREFLIGHT),
        "predictions_path": _relative(PREDICTIONS),
        "predictions_sha256": _sha256(PREDICTIONS),
        "public_predictions_commit": predictions_commit,
        "prediction_status": predictions["status"],
        "public_source_result_commit": source_commit,
        "held_rna_matrix_gets_authorized": 0,
        "held_cite_matrix_gets_authorized": 1,
        "created_at_utc": _timestamp(),
    }
    _write_json(SCORE_AUTHORIZATION, payload)
    return payload


def _verify_stage_prerequisites(stage: str) -> tuple[dict[str, Any], str]:
    authorization, authorization_commit = _verify_authorization(stage)
    if stage == "source":
        if (
            authorization.get("source_matrix_gets_authorized") != 10
            or authorization.get("held_rna_matrix_gets_authorized") != 0
            or authorization.get("held_cite_matrix_gets_authorized") != 0
        ):
            raise ProtocolRefusal("source authorization matrix count differs")
        _preflight()
    elif stage == "prediction":
        source, _selection, source_commit = _verify_source_public(require_pass=True)
        if (
            authorization.get("source_result_path") != _relative(SOURCE_RESULT)
            or authorization.get("source_result_sha256") != _sha256(SOURCE_RESULT)
            or authorization.get("source_reduced_path") != _relative(SOURCE_REDUCED)
            or authorization.get("source_reduced_sha256") != _sha256(SOURCE_REDUCED)
            or authorization.get("public_source_result_commit") != source_commit
            or authorization.get("source_status") != source.get("status")
            or authorization.get("held_rna_matrix_gets_authorized") != 1
            or authorization.get("held_cite_matrix_gets_authorized") != 0
        ):
            raise ProtocolRefusal("held RNA authorization predecessor differs")
        _require_ancestor(source_commit, authorization_commit)
    else:
        predictions, _source, predictions_commit, source_commit = (
            _verify_predictions_public()
        )
        if (
            authorization.get("predictions_path") != _relative(PREDICTIONS)
            or authorization.get("predictions_sha256") != _sha256(PREDICTIONS)
            or authorization.get("public_predictions_commit") != predictions_commit
            or authorization.get("prediction_status") != predictions.get("status")
            or authorization.get("public_source_result_commit") != source_commit
            or authorization.get("held_rna_matrix_gets_authorized") != 0
            or authorization.get("held_cite_matrix_gets_authorized") != 1
        ):
            raise ProtocolRefusal("held CITE authorization predecessor differs")
        _require_ancestor(predictions_commit, authorization_commit)
    return authorization, authorization_commit


def claim(stage: str) -> dict[str, Any]:
    if stage not in STAGE_PATHS:
        raise ValueError("unknown stage")
    config = STAGE_PATHS[stage]
    stage_order = ("source", "prediction", "score")
    protected = tuple(
        path
        for current in stage_order[stage_order.index(stage) :]
        for path in _stage_artifact_paths(
            current,
            ("attempt", "consumption", "journal", "sidecars", "checkpoint", "result"),
        )
    )
    protected += tuple(
        STAGE_PATHS[current]["authorization"]
        for current in stage_order[stage_order.index(stage) + 1 :]
    )
    if any(path.exists() for path in protected):
        raise ProtocolRefusal(f"{stage} already has a durable artifact")
    _authorization, authorization_commit = _verify_stage_prerequisites(stage)
    payload = {
        "schema": "gse252762-celiac-stage-attempt/2.0",
        "stage": stage,
        "status": "CLAIMED_BEFORE_MATRIX_ACCESS",
        "authorization_tag": config["authorization_tag"],
        "authorization_commit": authorization_commit,
        "implementation_commit": _implementation_commit(),
        "matrix_gets_authorized": config["matrix_gets"],
        "runtime": _require_runtime(),
        "rerun_permitted": False,
        "created_at_utc": _timestamp(),
    }
    _write_json(config["attempt"], payload)
    return payload


def _verify_claim(stage: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
    config = STAGE_PATHS[stage]
    authorization, authorization_commit = _verify_stage_prerequisites(stage)
    attempt = _read_json(config["attempt"])
    attempt_commit = _require_public_tag(
        config["attempt_tag"],
        (config["attempt"],),
        absent_paths=_forbidden_at_boundary(stage, "attempt"),
    )
    if (
        attempt.get("schema") != "gse252762-celiac-stage-attempt/2.0"
        or attempt.get("stage") != stage
        or attempt.get("status") != "CLAIMED_BEFORE_MATRIX_ACCESS"
        or attempt.get("authorization_tag") != config["authorization_tag"]
        or attempt.get("authorization_commit") != authorization_commit
        or attempt.get("implementation_commit") != _implementation_commit()
        or attempt.get("matrix_gets_authorized") != config["matrix_gets"]
        or attempt.get("runtime") != _require_runtime()
        or attempt.get("rerun_permitted") is not False
    ):
        raise ProtocolRefusal("public stage claim differs from the frozen contract")
    _require_ancestor(authorization_commit, attempt_commit)
    return attempt, attempt_commit, authorization


def _begin_consumption(stage: str, attempt_commit: str) -> dict[str, Any]:
    config = STAGE_PATHS[stage]
    blocked = [config["consumption"], *_forbidden_at_boundary(stage, "consumption")]
    if any(path.exists() for path in blocked):
        raise ProtocolRefusal(f"{stage} was already consumed or completed")
    payload = {
        "schema": "gse252762-celiac-stage-consumption/2.0",
        "stage": stage,
        "status": "CONSUMED_BEFORE_FIRST_MATRIX_REQUEST",
        "attempt_path": _relative(config["attempt"]),
        "attempt_sha256": _sha256(config["attempt"]),
        "public_attempt_commit": attempt_commit,
        "execution_id": secrets.token_hex(16),
        "runtime": _require_runtime(),
        "rerun_permitted": False,
        "consumed_at_utc": _timestamp(),
    }
    _write_json(config["consumption"], payload)
    return payload


def consume(stage: str) -> dict[str, Any]:
    if stage not in STAGE_PATHS:
        raise ValueError("unknown stage")
    _attempt, attempt_commit, _authorization = _verify_claim(stage)
    return _begin_consumption(stage, attempt_commit)


def _verify_consumption_public(
    stage: str,
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    config = STAGE_PATHS[stage]
    attempt, attempt_commit, authorization = _verify_claim(stage)
    consumption = _read_json(config["consumption"])
    consumption_commit = _require_public_tag(
        config["consumption_tag"],
        (config["attempt"], config["consumption"]),
        absent_paths=_forbidden_at_boundary(stage, "consumption"),
    )
    _require_ancestor(attempt_commit, consumption_commit)
    _attempt_payload, validated = _validate_consumption(stage)
    if validated.get("public_attempt_commit") != attempt_commit:
        raise ProtocolRefusal("consumption public attempt commit differs")
    if config["journal"].is_file():
        header = _read_jsonl(config["journal"])[0]
        if header.get("public_consumption_commit") != consumption_commit:
            raise ProtocolRefusal("journal public consumption commit differs")
    return attempt, consumption_commit, authorization, consumption


@contextmanager
def _stage_access(
    stage: str,
) -> Iterator[tuple[_AccessLease, dict[str, Any]]]:
    _attempt, consumption_commit, authorization, consumption = (
        _verify_consumption_public(stage)
    )
    config = STAGE_PATHS[stage]
    handle = config["consumption"].open("rb")
    lease: _AccessLease | None = None
    locked = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except BlockingIOError as error:
            raise ProtocolRefusal("stage consumption is already active") from error
        if any(path.exists() for path in _forbidden_at_boundary(stage, "consumption")):
            raise ProtocolRefusal("stage has a journal, result, or downstream artifact")
        preflight = _preflight()
        expected = tuple(
            (batch, modality)
            for batch, modality, _samples_for_access in _expected_matrix_accesses(
                preflight, stage
            )
        )
        _write_journal_header(
            config["journal"],
            {
                "schema": "gse252762-celiac-access-journal/2.0",
                "stage": stage,
                "event": "OPENED_BEFORE_MATRIX_ACCESS",
                "attempt_sha256": _sha256(config["attempt"]),
                "consumption_path": _relative(config["consumption"]),
                "consumption_sha256": _sha256(config["consumption"]),
                "public_consumption_commit": consumption_commit,
                "execution_id": consumption["execution_id"],
                "automatic_retries": False,
                "http_redirects": False,
                "timestamp_utc": _timestamp(),
            },
        )
        lease = _AccessLease(stage, consumption["execution_id"], expected)
        yield lease, authorization
    finally:
        if lease is not None:
            lease.active = False
        if locked:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _failure(
    stage: str, phase: str, error: BaseException, *, reason_code: str | None = None
) -> dict[str, Any]:
    config = STAGE_PATHS[stage]
    output = config["result"]
    if output.exists():
        existing = _read_json(output)
        _validate_failure(stage, existing)
        return existing
    if not config["consumption"].is_file():
        raise ProtocolRefusal("terminal failure requires a consumed stage")
    code = reason_code or FAILURE_CODES[stage].get(phase)
    if code is None:
        code = f"{stage.upper()}_EXECUTION_FAILED"
    payload = {
        "schema": "gse252762-celiac-terminal-result/2.0",
        "status": "TERMINAL_EXECUTION_FAILURE",
        "stage": stage,
        "phase": phase,
        "reason_code": code,
        "exception_class": type(error).__name__,
        "attempt_path": _relative(config["attempt"]),
        "attempt_sha256": _sha256(config["attempt"]),
        "consumption_path": _relative(config["consumption"]),
        "consumption_sha256": _sha256(config["consumption"]),
        "access_journal_path": _relative(config["journal"]),
        "access_journal_sha256": (
            _sha256(config["journal"]) if config["journal"].is_file() else None
        ),
        "rerun_permitted": False,
        "timestamp_utc": _timestamp(),
        "selected_count_sidecars": _existing_sidecar_bindings(stage),
    }
    if config["checkpoint"].is_file():
        payload.update(_checkpoint_binding(stage))
    _write_json(output, payload)
    return payload


def run_source() -> dict[str, Any]:
    phase = "source_reduction"
    with _stage_access("source") as (lease, _authorization):
        try:
            preflight = _preflight()
            reduced = _reduce_source(preflight, lease)
            _assert_access_complete(lease)
            _write_json(SOURCE_REDUCED, reduced)
            phase = "source_selection"
            selection = _source_selection(reduced)
            payload = {
                "schema": "gse252762-celiac-source-result/2.0",
                "status": (
                    "SOURCE_PASS"
                    if selection["status"] == "PROMOTED"
                    else "TERMINAL_SOURCE_GATE_FAIL"
                ),
                "source_reduced_path": _relative(SOURCE_REDUCED),
                "source_reduced_sha256": _sha256(SOURCE_REDUCED),
                "source_selection": selection,
                "selected_count_sidecars": _complete_sidecar_bindings("source"),
                "access_journal_path": _relative(SOURCE_JOURNAL),
                "access_journal_sha256": _sha256(SOURCE_JOURNAL),
                "rerun_permitted": False,
            }
            _write_json(SOURCE_RESULT, payload)
            return payload
        except BaseException as error:
            _failure("source", phase, error)
            raise


def run_predict() -> dict[str, Any]:
    phase = "held_rna_reduction"
    with _stage_access("prediction") as (lease, authorization):
        try:
            preflight = _preflight()
            held = _held_samples(preflight)
            rna, audit = _reduce_matrix(
                preflight,
                6,
                "rna",
                held,
                PREDICTION_JOURNAL,
                lease,
            )
            _assert_access_complete(lease)
            checkpoint = _held_checkpoint_payload(
                preflight, "prediction", "rna", rna, audit
            )
            _write_json(PREDICTION_REDUCED, checkpoint)
            phase = "held_prediction"
            reduced = _read_json(SOURCE_REDUCED)
            source_result = _read_json(SOURCE_RESULT)
            selection = _replay_source_result(preflight, reduced, source_result)
            payload = _prediction_payload(
                preflight,
                reduced,
                selection,
                rna,
                audit,
                authorization["public_source_result_commit"],
            )
            payload.update(_checkpoint_binding("prediction"))
            payload["selected_count_sidecars"] = _complete_sidecar_bindings(
                "prediction"
            )
            _write_json(PREDICTIONS, payload)
            return payload
        except BaseException as error:
            _failure("prediction", phase, error)
            raise


def _held_truth_from_counts(
    preflight: Mapping[str, Any],
    predictions: Mapping[str, Any],
    cite_counts: np.ndarray,
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    held = _held_samples(preflight)
    truth = []
    conditions = []
    contexts = []
    sample_ids = []
    for index, (sample, record) in enumerate(zip(held, predictions["samples"])):
        states = np.asarray(record["rna_states"], dtype=np.uint8)
        counts = cite_counts[index]
        adt = core.adt_top_states(
            counts,
            sample["selected_barcodes"],
            sample["sample_id"],
        )
        truth.append(core.joint_binary_tables(states, adt))
        conditions.append(sample["condition"])
        contexts.append(sample["context"])
        sample_ids.append(sample["sample_id"])
    return (
        np.asarray(truth),
        conditions,
        contexts,
        sample_ids,
    )


def _loss_payload(
    losses: Mapping[str, np.ndarray], sample_ids: Sequence[str]
) -> dict[str, Any]:
    return {
        method: {
            "mean": float(np.mean(values)),
            "by_sample": {
                sample: float(loss) for sample, loss in zip(sample_ids, values)
            },
        }
        for method, values in losses.items()
    }


def _score_payload(
    truth: np.ndarray,
    predictions: Mapping[str, Any],
    conditions: Sequence[str],
    sample_ids: Sequence[str],
    strongest: str,
) -> dict[str, Any]:
    records = {record["sample_id"]: record for record in predictions["samples"]}
    methods = tuple(records[sample_ids[0]]["predictions"])
    losses = {method: np.empty(len(sample_ids), dtype=float) for method in methods}
    entity_losses = {
        method: np.empty((len(sample_ids), 9, 9), dtype=float) for method in methods
    }
    for sample_index, sample_id in enumerate(sample_ids):
        record = records[sample_id]
        if tuple(record["predictions"]) != methods:
            raise ProtocolRefusal("prediction methods changed between held samples")
        for method in methods:
            current = core.entity_deviance(
                truth[sample_index],
                np.asarray(record["predictions"][method], dtype=float),
            )
            entity_losses[method][sample_index] = current
            losses[method][sample_index] = float(current.mean())
    gate = core.held_confirmation_gate(losses, conditions, strongest)
    diagonal = np.arange(9)
    return {
        "losses": _loss_payload(losses, sample_ids),
        "condition_mean_losses": {
            condition: {
                method: float(np.mean(values[np.asarray(conditions) == condition]))
                for method, values in losses.items()
            }
            for condition in core.DEPOSITED_CONDITIONS
        },
        "entity_mean_losses": {
            method: values.mean(axis=0).tolist()
            for method, values in entity_losses.items()
        },
        "cognate_diagonal_mean_losses": {
            method: float(values[:, diagonal, diagonal].mean())
            for method, values in entity_losses.items()
        },
        "held_gate": _json_value(gate),
    }


def _validate_held_audit(
    preflight: Mapping[str, Any], audit: Mapping[str, Any], modality: str
) -> None:
    held = _held_samples(preflight)
    _validate_audit_certificate(audit, preflight, 6, modality, held)


def _replay_score_result(
    preflight: Mapping[str, Any],
    source: Mapping[str, Any],
    predictions: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    held = _held_samples(preflight)
    sample_ids = [sample["sample_id"] for sample in held]
    conditions = [sample["condition"] for sample in held]
    contexts = [sample["context"] for sample in held]
    cite_raw = np.asarray(result.get("cite_counts"))
    cite = np.asarray(result.get("cite_counts"), dtype=np.int64)
    truth_raw = np.asarray(result.get("truth_tables"))
    truth = np.asarray(result.get("truth_tables"), dtype=np.int64)
    if (
        result.get("schema") != "gse252762-celiac-held-result/2.0"
        or result.get("status") not in {"CONFIRMATION_PASS", "CONFIRMATION_FAIL"}
        or result.get("predictions_path") != _relative(PREDICTIONS)
        or result.get("predictions_sha256") != _sha256(PREDICTIONS)
        or result.get("sample_ids") != sample_ids
        or result.get("conditions") != conditions
        or result.get("contexts") != contexts
        or cite_raw.dtype.kind not in "iu"
        or cite.shape != (13, 256, 9)
        or np.any(cite < 0)
        or _array_sha256(cite) != result.get("cite_counts_sha256")
        or truth_raw.dtype.kind not in "iu"
        or truth.shape != (13, 9, 9, 2, 2)
        or _array_sha256(truth) != result.get("truth_tables_sha256")
        or result.get("access_journal_path") != _relative(SCORE_JOURNAL)
        or not SCORE_JOURNAL.is_file()
        or result.get("access_journal_sha256") != _sha256(SCORE_JOURNAL)
        or result.get("held_cite_reduced_path") != _relative(SCORE_REDUCED)
        or not SCORE_REDUCED.is_file()
        or result.get("held_cite_reduced_sha256") != _sha256(SCORE_REDUCED)
        or result.get("rerun_permitted") is not False
    ):
        raise ProtocolRefusal("held result header, axes, or digest differs")
    _validate_sidecar_bindings(preflight, "score", result, require_all=True)
    checkpoint_counts, checkpoint_audit = _validate_held_checkpoint(
        preflight, "score", _read_json(SCORE_REDUCED)
    )
    if not np.array_equal(cite, checkpoint_counts):
        raise ProtocolRefusal("held result counts differ from reduction checkpoint")
    audit = result.get("held_cite_audit")
    if not isinstance(audit, Mapping) or dict(audit) != checkpoint_audit:
        raise ProtocolRefusal("held CITE audit is absent")
    _validate_held_audit(preflight, audit, "cite")
    selected_block = cite.reshape(13 * 256, 9).T
    if audit.get("selected_block_sha256") != _array_sha256(selected_block) or audit.get(
        "selected_value_sum"
    ) != int(selected_block.sum(dtype=np.int64)):
        raise ProtocolRefusal("held CITE audit does not bind published counts")
    replay_truth = []
    for index, (sample, prediction) in enumerate(zip(held, predictions["samples"])):
        rna_states = np.asarray(prediction["rna_states"], dtype=np.uint8)
        adt_states = core.adt_top_states(
            cite[index],
            sample["selected_barcodes"],
            sample["sample_id"],
        )
        replay_truth.append(core.joint_binary_tables(rna_states, adt_states))
    if not np.array_equal(truth, np.asarray(replay_truth)):
        raise ProtocolRefusal("held truth does not replay from published counts")
    _validate_access_journal("score", preflight, audits=(audit,))
    summary = _score_payload(
        truth,
        predictions,
        conditions,
        sample_ids,
        source["source_selection"]["strongest_benchmark"],
    )
    expected = {
        key: result[key]
        for key in (
            "losses",
            "condition_mean_losses",
            "entity_mean_losses",
            "cognate_diagonal_mean_losses",
            "held_gate",
        )
    }
    if _json_value(summary) != expected:
        raise ProtocolRefusal("held result does not replay from frozen truth")
    expected_status = (
        "CONFIRMATION_PASS" if summary["held_gate"]["passes"] else "CONFIRMATION_FAIL"
    )
    if result.get("status") != expected_status:
        raise ProtocolRefusal("held terminal status differs from the frozen gate")


def run_score() -> dict[str, Any]:
    phase = "held_cite_reduction"
    with _stage_access("score") as (lease, authorization):
        try:
            preflight = _preflight()
            reduced = _read_json(SOURCE_REDUCED)
            source_result = _read_json(SOURCE_RESULT)
            predictions = _read_json(PREDICTIONS)
            replay = _replay_predictions(
                preflight,
                reduced,
                source_result,
                authorization["public_source_result_commit"],
                predictions,
            )
            if not replay:
                raise ProtocolRefusal("held predictions are absent")
            held = _held_samples(preflight)
            cite, audit = _reduce_matrix(
                preflight, 6, "cite", held, SCORE_JOURNAL, lease
            )
            _assert_access_complete(lease)
            checkpoint = _held_checkpoint_payload(
                preflight, "score", "cite", cite, audit
            )
            _write_json(SCORE_REDUCED, checkpoint)
            phase = "held_scoring"
            cite_counts, checkpoint_audit = _validate_held_checkpoint(
                preflight, "score", checkpoint
            )
            truth, conditions, contexts, sample_ids = _held_truth_from_counts(
                preflight, predictions, cite_counts
            )
            strongest = source_result["source_selection"]["strongest_benchmark"]
            summary = _score_payload(
                truth, predictions, conditions, sample_ids, strongest
            )
            gate = summary["held_gate"]
            payload = {
                "schema": "gse252762-celiac-held-result/2.0",
                "status": (
                    "CONFIRMATION_PASS" if gate["passes"] else "CONFIRMATION_FAIL"
                ),
                "predictions_path": _relative(PREDICTIONS),
                "predictions_sha256": _sha256(PREDICTIONS),
                "public_predictions_commit": authorization["public_predictions_commit"],
                "sample_ids": sample_ids,
                "conditions": conditions,
                "contexts": contexts,
                "cite_counts": cite_counts.tolist(),
                "cite_counts_sha256": _array_sha256(cite_counts),
                "truth_tables": truth.tolist(),
                "truth_tables_sha256": _array_sha256(truth),
                **summary,
                "held_cite_audit": checkpoint_audit,
                "selected_count_sidecars": _complete_sidecar_bindings("score"),
                "access_journal_path": _relative(SCORE_JOURNAL),
                "access_journal_sha256": _sha256(SCORE_JOURNAL),
                "rerun_permitted": False,
            }
            payload.update(_checkpoint_binding("score"))
            _write_json(SCORE_RESULT, payload)
            return payload
        except BaseException as error:
            _failure("score", phase, error)
            raise


def _validate_failure(stage: str, payload: Mapping[str, Any]) -> None:
    config = STAGE_PATHS[stage]
    _validate_consumption(stage)
    phase = payload.get("phase")
    if (
        payload.get("schema") != "gse252762-celiac-terminal-result/2.0"
        or payload.get("status") != "TERMINAL_EXECUTION_FAILURE"
        or payload.get("stage") != stage
        or phase not in FAILURE_CODES[stage]
        or payload.get("reason_code") != FAILURE_CODES[stage].get(phase)
        or not isinstance(payload.get("exception_class"), str)
        or not payload["exception_class"].isidentifier()
        or payload.get("attempt_path") != _relative(config["attempt"])
        or payload.get("attempt_sha256") != _sha256(config["attempt"])
        or payload.get("consumption_path") != _relative(config["consumption"])
        or payload.get("consumption_sha256") != _sha256(config["consumption"])
        or payload.get("access_journal_path") != _relative(config["journal"])
        or payload.get("access_journal_sha256")
        != (_sha256(config["journal"]) if config["journal"].is_file() else None)
        or payload.get("rerun_permitted") is not False
    ):
        raise ProtocolRefusal("terminal execution failure bindings differ")
    preflight = _preflight()
    _validate_sidecar_bindings(preflight, stage, payload, require_all=False)
    checkpoint_bound = _validate_checkpoint_binding(preflight, stage, payload)
    post_reduction_phase = phase in {
        "source_selection",
        "held_prediction",
        "held_scoring",
    }
    if post_reduction_phase and not checkpoint_bound:
        raise ProtocolRefusal(f"{phase} failure lacks completed reduction")
    if not checkpoint_bound:
        _validate_access_journal(
            stage,
            preflight,
            terminal_failure=True,
        )


def _runtime_payload_is_frozen(runtime: Any) -> bool:
    return isinstance(runtime, Mapping) and dict(runtime) == FROZEN_RUNTIME


def _git_sha_is_valid(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_consumption(stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
    config = STAGE_PATHS[stage]
    attempt = _read_json(config["attempt"])
    consumption = _read_json(config["consumption"])
    if (
        attempt.get("schema") != "gse252762-celiac-stage-attempt/2.0"
        or attempt.get("stage") != stage
        or attempt.get("status") != "CLAIMED_BEFORE_MATRIX_ACCESS"
        or attempt.get("matrix_gets_authorized") != config["matrix_gets"]
        or not _runtime_payload_is_frozen(attempt.get("runtime"))
        or attempt.get("rerun_permitted") is not False
        or not _git_sha_is_valid(attempt.get("authorization_commit"))
        or not _git_sha_is_valid(attempt.get("implementation_commit"))
        or consumption.get("schema") != "gse252762-celiac-stage-consumption/2.0"
        or consumption.get("stage") != stage
        or consumption.get("status") != "CONSUMED_BEFORE_FIRST_MATRIX_REQUEST"
        or consumption.get("attempt_path") != _relative(config["attempt"])
        or consumption.get("attempt_sha256") != _sha256(config["attempt"])
        or not _git_sha_is_valid(consumption.get("public_attempt_commit"))
        or not isinstance(consumption.get("execution_id"), str)
        or len(consumption["execution_id"]) != 32
        or any(
            character not in "0123456789abcdef"
            for character in consumption["execution_id"]
        )
        or consumption.get("runtime") != attempt.get("runtime")
        or consumption.get("rerun_permitted") is not False
    ):
        raise ProtocolRefusal("stage attempt or consumption contract differs")
    return attempt, consumption


def _verify_terminal_failure_public(stage: str) -> str:
    config = STAGE_PATHS[stage]
    _attempt, consumption_commit, _authorization, _consumption = (
        _verify_consumption_public(stage)
    )
    tag = {
        "source": SOURCE_RESULT_TAG,
        "prediction": PREDICTIONS_TAG,
        "score": SCORE_RESULT_TAG,
    }[stage]
    result = _read_json(config["result"])
    _validate_failure(stage, result)
    paths = [config["attempt"], config["consumption"], config["result"]]
    absent = list(_forbidden_at_boundary(stage, "result"))
    if config["journal"].is_file():
        paths.append(config["journal"])
    else:
        absent.append(config["journal"])
    bound_sidecars = tuple(
        config["sidecars"][: len(result.get("selected_count_sidecars", []))]
    )
    paths.extend(bound_sidecars)
    absent.extend(path for path in config["sidecars"] if path not in bound_sidecars)
    path_field, _hash_field = CHECKPOINT_BINDINGS[stage]
    if result.get(path_field) is not None:
        paths.append(config["checkpoint"])
    else:
        absent.append(config["checkpoint"])
    commit = _require_public_tag(
        tag,
        tuple(paths),
        absent_paths=tuple(absent),
    )
    _require_ancestor(consumption_commit, commit)
    return commit


def _verify_score_public() -> str:
    _attempt, consumption_commit, authorization, _consumption = (
        _verify_consumption_public("score")
    )
    commit = _require_public_tag(
        SCORE_RESULT_TAG,
        (
            SCORE_ATTEMPT,
            SCORE_CONSUMPTION,
            SCORE_JOURNAL,
            *STAGE_PATHS["score"]["sidecars"],
            SCORE_REDUCED,
            SCORE_RESULT,
        ),
        absent_paths=_forbidden_at_boundary("score", "result"),
    )
    _require_ancestor(consumption_commit, commit)
    predictions, source, predictions_commit, _source_commit = (
        _verify_predictions_public()
    )
    result = _read_json(SCORE_RESULT)
    if (
        result.get("public_predictions_commit") != predictions_commit
        or authorization.get("public_predictions_commit") != predictions_commit
    ):
        raise ProtocolRefusal("held result public prediction commit differs")
    _replay_score_result(_preflight(), source, predictions, result)
    return commit


def recover(stage: str) -> dict[str, Any]:
    if stage not in STAGE_PATHS:
        raise ValueError("unknown stage")
    config = STAGE_PATHS[stage]
    if config["result"].is_file():
        result = _read_json(config["result"])
        report = validate(require_public=False)
        key = {
            "source": "source_valid",
            "prediction": "predictions_valid",
            "score": "held_result_valid",
        }[stage]
        if report.get(key) is not True:
            raise ProtocolRefusal("existing stage result is not locally replayable")
        return result
    _verify_consumption_public(stage)
    handle = config["consumption"].open("rb")
    locked = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except BlockingIOError as error:
            raise ProtocolRefusal("cannot recover an active stage") from error
        return _failure(
            stage,
            "recovery",
            RuntimeError("interrupted stage"),
            reason_code=FAILURE_CODES[stage]["recovery"],
        )
    finally:
        if locked:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def validate(*, require_public: bool = True) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": "gse252762-celiac-validation/2.0",
        "preflight_valid": False,
        "artifact_checks": 0,
        "public_validation_requested": require_public,
    }
    preflight = _preflight()
    report["preflight_valid"] = len(_held_samples(preflight)) == 13
    source = _read_json(SOURCE_RESULT) if SOURCE_RESULT.is_file() else None
    predictions = _read_json(PREDICTIONS) if PREDICTIONS.is_file() else None
    held_result = _read_json(SCORE_RESULT) if SCORE_RESULT.is_file() else None
    source_status = source.get("status") if source is not None else None
    prediction_status = predictions.get("status") if predictions is not None else None
    held_status = held_result.get("status") if held_result is not None else None
    if source is not None and source_status not in {
        "SOURCE_PASS",
        "TERMINAL_SOURCE_GATE_FAIL",
        "TERMINAL_EXECUTION_FAILURE",
    }:
        raise ProtocolRefusal("source result has an unknown status")
    if predictions is not None and prediction_status not in {
        "PREDICTIONS_FROZEN_BEFORE_HELD_CITE_ACCESS",
        "TERMINAL_EXECUTION_FAILURE",
    }:
        raise ProtocolRefusal("prediction result has an unknown status")
    if held_result is not None and held_status not in {
        "CONFIRMATION_PASS",
        "CONFIRMATION_FAIL",
        "TERMINAL_EXECUTION_FAILURE",
    }:
        raise ProtocolRefusal("held result has an unknown status")

    fields = (
        "authorization",
        "attempt",
        "consumption",
        "journal",
        "sidecars",
        "checkpoint",
        "result",
    )
    later_prediction_artifacts = any(
        path.exists()
        for stage in ("prediction", "score")
        for path in _stage_artifact_paths(stage, fields)
    )
    later_score_artifacts = any(
        path.exists() for path in _stage_artifact_paths("score", fields)
    )
    if (
        source_status in {"TERMINAL_SOURCE_GATE_FAIL", "TERMINAL_EXECUTION_FAILURE"}
        and later_prediction_artifacts
    ):
        raise ProtocolRefusal("later artifacts follow a terminal source stage")
    if prediction_status == "TERMINAL_EXECUTION_FAILURE" and later_score_artifacts:
        raise ProtocolRefusal("score artifacts follow a terminal prediction stage")
    if predictions is not None and source_status != "SOURCE_PASS":
        raise ProtocolRefusal("prediction artifact lacks a replayable SOURCE_PASS")
    if (
        held_result is not None
        and prediction_status != "PREDICTIONS_FROZEN_BEFORE_HELD_CITE_ACCESS"
    ):
        raise ProtocolRefusal("held result lacks replayable frozen predictions")
    for stage in STAGE_PATHS:
        config = STAGE_PATHS[stage]
        if config["consumption"].exists() and not config["result"].exists():
            raise ProtocolRefusal(f"consumed {stage} stage lacks a terminal result")

    if SOURCE_RESULT.is_file():
        assert source is not None
        report["artifact_checks"] += 1
        report["source_status"] = source_status
        if source_status == "TERMINAL_EXECUTION_FAILURE":
            _validate_failure("source", source)
            report["source_valid"] = True
        else:
            _validate_consumption("source")
            selection = _replay_source_result(
                preflight, _read_json(SOURCE_REDUCED), source
            )
            report["source_valid"] = True
            report["pilot_gate_replayed"] = (
                selection.get("pilot_promotion_gate")
                == source["source_selection"]["pilot_promotion_gate"]
            )
    if PREDICTIONS.is_file():
        assert predictions is not None and source is not None
        report["artifact_checks"] += 1
        report["prediction_status"] = prediction_status
        if prediction_status == "TERMINAL_EXECUTION_FAILURE":
            _validate_failure("prediction", predictions)
            report["predictions_valid"] = True
        else:
            _validate_consumption("prediction")
            source_commit = predictions.get("public_source_result_commit")
            if not isinstance(source_commit, str):
                raise ProtocolRefusal("prediction source commit is absent")
            _replay_predictions(
                preflight,
                _read_json(SOURCE_REDUCED),
                source,
                source_commit,
                predictions,
            )
            report["predictions_valid"] = True
            report["predictions_replayed"] = True
    if SCORE_RESULT.is_file():
        assert (
            held_result is not None and source is not None and predictions is not None
        )
        report["artifact_checks"] += 1
        report["held_status"] = held_status
        if held_status == "TERMINAL_EXECUTION_FAILURE":
            _validate_failure("score", held_result)
            report["held_result_valid"] = True
        else:
            _validate_consumption("score")
            _replay_score_result(preflight, source, predictions, held_result)
            report["held_result_valid"] = True
            report["held_gate_replayed"] = True
    checks = [
        value for key, value in report.items() if key.endswith(("_valid", "_replayed"))
    ]
    report["local_artifact_chain_valid"] = bool(report["artifact_checks"]) and all(
        value is True for value in checks
    )
    source_terminal = report.get("source_status") in {
        "TERMINAL_SOURCE_GATE_FAIL",
        "TERMINAL_EXECUTION_FAILURE",
    }
    prediction_terminal = (
        report.get("prediction_status") == "TERMINAL_EXECUTION_FAILURE"
    )
    held_terminal = report.get("held_status") in {
        "CONFIRMATION_PASS",
        "CONFIRMATION_FAIL",
        "TERMINAL_EXECUTION_FAILURE",
    }
    report["campaign_complete"] = bool(
        source_terminal or prediction_terminal or held_terminal
    )
    report["public_chain_valid"] = None
    if require_public and report["campaign_complete"]:
        if source_status == "TERMINAL_EXECUTION_FAILURE":
            _verify_terminal_failure_public("source")
        elif source_status == "TERMINAL_SOURCE_GATE_FAIL":
            _verify_source_public(require_pass=False)
        elif prediction_status == "TERMINAL_EXECUTION_FAILURE":
            _verify_terminal_failure_public("prediction")
        elif held_status == "TERMINAL_EXECUTION_FAILURE":
            _verify_terminal_failure_public("score")
        elif held_status in {"CONFIRMATION_PASS", "CONFIRMATION_FAIL"}:
            _verify_score_public()
        else:
            raise ProtocolRefusal("completed campaign has no public terminal stage")
        report["public_chain_valid"] = True
    report["valid"] = bool(
        report["local_artifact_chain_valid"]
        and report["campaign_complete"]
        and report["public_chain_valid"] is True
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("authorize-source", "authorize-prediction", "authorize-score"):
        subparsers.add_parser(command)
    for stage in STAGE_PATHS:
        subparsers.add_parser(f"claim-{stage}")
        subparsers.add_parser(f"consume-{stage}")
        subparsers.add_parser(f"run-{stage}")
    recovery = subparsers.add_parser("recover")
    recovery.add_argument("--stage", choices=tuple(STAGE_PATHS), required=True)
    subparsers.add_parser("validate")
    subparsers.add_parser("validate-local")
    args = parser.parse_args()
    if args.command == "authorize-source":
        payload = authorize_source()
    elif args.command == "authorize-prediction":
        payload = authorize_prediction()
    elif args.command == "authorize-score":
        payload = authorize_score()
    elif args.command.startswith("claim-"):
        payload = claim(args.command.removeprefix("claim-"))
    elif args.command.startswith("consume-"):
        payload = consume(args.command.removeprefix("consume-"))
    elif args.command == "run-source":
        payload = run_source()
    elif args.command == "run-prediction":
        payload = run_predict()
    elif args.command == "run-score":
        payload = run_score()
    elif args.command == "recover":
        payload = recover(args.stage)
    elif args.command == "validate-local":
        payload = validate(require_public=False)
    else:
        payload = validate(require_public=True)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
