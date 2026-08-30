"""One-shot held-patient confirmation for GSE313642 HCC CITE-seq.

Axis preflight is local and matrix-free.  Source and held matrix stages require
separate public claims, consume a private capability before access, issue one
streaming GET per selected matrix with no retry, and delete each matrix body.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import secrets
import shutil
import subprocess
import tempfile
from typing import Any, BinaryIO, Iterable, Mapping, Sequence
import urllib.parse
import urllib.request

import numpy as np
import scipy

from experiments.gse313642_hcc_core import (
    ADT_HIGH_COUNT,
    CELL_BUDGET,
    COHORTS,
    MARKERS,
    PrimaryConfig,
    SOURCE_GATE_COMPARATORS,
    adt_midrank_states,
    deserialize_models,
    fit_models,
    held_gate,
    patient_tables,
    predict_serialized_at_margins,
    rna_detection_states,
    select_barcodes,
    select_comparator_alphas,
    select_primary_configuration,
    serialize_models,
    serialized_panel_losses,
)
from experiments.gse217494_heart_core import entity_deviance, joint_binary_tables
from mapreg.streamed_gzip_matrix_market import reduce_gzip_matrix_market


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/confirmation/gse313642_hcc"
DESIGNATION = DATA_DIR / "candidate_designation_v2.json"
SOURCE_MANIFEST = DATA_DIR / "source_manifest_v2.json"
AXIS_PREFLIGHT = ROOT / "results/development/gse313642_hcc_axis_preflight_v2.json"
V1_DESIGNATION = DATA_DIR / "candidate_designation_v1.json"
V1_SOURCE_MANIFEST = DATA_DIR / "source_manifest_v1.json"
V1_AXIS_ACCESS = DATA_DIR / "axis_access_v1.jsonl"
V1_AXIS_REFUSAL = ROOT / "results/development/gse313642_hcc_axis_preflight_v1.json"
AXIS_ACCESS = V1_AXIS_ACCESS
CALIBRATION_ATTEMPT = DATA_DIR / "calibration_attempt_v2.json"
CALIBRATION_CONSUMPTION = DATA_DIR / "calibration_consumption_v2.json"
CALIBRATION_ACCESS = DATA_DIR / "calibration_access_v2.jsonl"
CALIBRATION_SELECTION = (
    ROOT / "results/development/gse313642_hcc_calibration_selection_v2.json"
)
PILOT_AUTHORIZATION = DATA_DIR / "pilot_authorization_v2.json"
SOURCE_ATTEMPT = DATA_DIR / "source_attempt_v2.json"
SOURCE_CONSUMPTION = DATA_DIR / "source_consumption_v2.json"
SOURCE_ACCESS = DATA_DIR / "source_access_v2.jsonl"
SOURCE_RESULT = ROOT / "results/development/gse313642_hcc_source_v2.json"
PREDICTION_AUTHORIZATION = DATA_DIR / "prediction_authorization_v2.json"
PREDICTION_ATTEMPT = DATA_DIR / "prediction_attempt_v2.json"
PREDICTION_CONSUMPTION = DATA_DIR / "prediction_consumption_v2.json"
PREDICTION_ACCESS = DATA_DIR / "prediction_access_v2.jsonl"
PREDICTION_RESULT = ROOT / "results/gse313642_hcc_predictions_v2.json"
SCORE_AUTHORIZATION = DATA_DIR / "score_authorization_v2.json"
SCORE_ATTEMPT = DATA_DIR / "score_attempt_v2.json"
SCORE_CONSUMPTION = DATA_DIR / "score_consumption_v2.json"
SCORE_ACCESS = DATA_DIR / "score_access_v2.jsonl"
SCORE_RESULT = ROOT / "results/gse313642_hcc_confirmation_v2.json"
DEFAULT_SCRATCH = Path("/private/tmp/gse313642-hcc-v2")
PROTOCOL = ROOT / "docs/GSE313642_HCC_CITESEQ_HELD_PATIENT_PROTOCOL_V2_2026-08-30.md"
RUNNER = ROOT / "experiments/confirm_gse313642_hcc.py"
CORE = ROOT / "experiments/gse313642_hcc_core.py"
RUNNER_TEST = ROOT / "tests/test_gse313642_hcc_confirmation.py"
IMPLEMENTATION_FREEZE = DATA_DIR / "pre_access_implementation_v2.json"

PUBLIC_ORIGIN = "https://github.com/sushaan-k/coupling-fields-benchmark.git"
CANDIDATE_TAG = "gse313642-hcc-v2-candidate"
IMPLEMENTATION_TAG = "gse313642-hcc-v2-implementation"
PREFLIGHT_TAG = "gse313642-hcc-v2-axis-preflight"
CALIBRATION_TAG = "gse313642-hcc-v2-calibration"
CALIBRATION_ATTEMPT_TAG = "gse313642-hcc-v2-calibration-attempt"
PILOT_AUTHORIZATION_TAG = "gse313642-hcc-v2-pilot-authorization"
SOURCE_TAG = "gse313642-hcc-v2-source"
SOURCE_ATTEMPT_TAG = "gse313642-hcc-v2-source-attempt"
PREDICTION_AUTHORIZATION_TAG = "gse313642-hcc-v2-prediction-authorization"
PREDICTION_TAG = "gse313642-hcc-v2-predictions"
PREDICTION_ATTEMPT_TAG = "gse313642-hcc-v2-prediction-attempt"
SCORE_AUTHORIZATION_TAG = "gse313642-hcc-v2-score-authorization"
SCORE_ATTEMPT_TAG = "gse313642-hcc-v2-score-attempt"
SCORE_TAG = "gse313642-hcc-v2-score"
V1_TERMINAL_TAG = "gse313642-hcc-v1-terminal-axis-refusal"
V1_TERMINAL_COMMIT = "b814c6c33a068ae341f983d8d82ea40dddb36207"
V1_AXIS_ACCESS_SHA256 = (
    "2619458df75a219f1f2b21b8f00e7dc2c096b7309a06dc22caa8d6d5ca201555"
)
V1_AXIS_REFUSAL_SHA256 = (
    "94840b4b8756e8745caaa7738a060001ab0aea7311ddedd740aa5b6477ddba86"
)

IMPLEMENTATION_BINDINGS = (
    "data/confirmation/gse313642_hcc/candidate_designation_v2.json",
    "data/confirmation/gse313642_hcc/source_manifest_v2.json",
    "data/confirmation/gse313642_hcc/candidate_designation_v1.json",
    "data/confirmation/gse313642_hcc/source_manifest_v1.json",
    "data/confirmation/gse313642_hcc/axis_access_v1.jsonl",
    "results/development/gse313642_hcc_axis_preflight_v1.json",
    "docs/GSE313642_HCC_CITESEQ_HELD_PATIENT_PROTOCOL_V2_2026-08-30.md",
    "experiments/confirm_gse313642_hcc.py",
    "experiments/gse313642_hcc_core.py",
    "tests/test_gse313642_hcc_confirmation.py",
    "experiments/gse217494_heart_core.py",
    "mapreg/classical_residuals.py",
    "mapreg/common_effect_conditional.py",
    "mapreg/context_conditional_coupling.py",
    "mapreg/coupling_fields.py",
    "mapreg/heterogeneity_adaptive_coupling.py",
    "mapreg/poisson_loglinear.py",
    "mapreg/streamed_gzip_matrix_market.py",
    "mapreg/structured_context_conditional.py",
    "mapreg/table_prediction.py",
    "pyproject.toml",
)

BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9371nnn"
ROLE_COUNTS = {"calibration": 11, "pilot": 11, "held": 12}
MATRIX_MEMBER = "matrix.mtx.gz"
AXIS_MEMBERS = ("barcodes.tsv.gz", "features.tsv.gz")
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
REQUIRED_RUNTIME = {
    "python": {"implementation": "CPython", "version": "3.9.6"},
    "operating_system": {"system": "Darwin", "machine": "arm64"},
    "packages": {"numpy": "2.0.2", "scipy": "1.13.1"},
    "thread_environment": {name: "1" for name in THREAD_VARIABLES},
}
FB_MARKER_DESCRIPTIONS = {
    "CD4": "0072 anti-human CD4",
    "CD7": "0066 anti-human CD7",
    "CD14": "0081 anti-human CD14",
    "CD19": "0050 anti-human CD19",
    "CD33": "0052 anti-human CD33",
    "CD38": "0389 anti-human CD38",
    "CD44": "0073 anti-mouse/human CD44",
    "CD47": "0026 anti-human CD47",
    "CD52": "0033 anti-human CD52",
}
FROZEN_FEATURE_SCHEMA = {
    "columns_per_row": 3,
    "gex": {
        "marker_column_1based": 2,
        "type_column_1based": 3,
        "required_type": "Gene Expression",
    },
    "fb": {
        "marker_column_1based": 1,
        "description_column_1based": 2,
        "type_column_1based": 3,
        "required_type": "Antibody Capture",
        "exact_marker_descriptions": FB_MARKER_DESCRIPTIONS,
    },
}


class ProtocolRefusal(RuntimeError):
    """A stable terminal protocol refusal."""

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


def _runtime() -> dict[str, Any]:
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "operating_system": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "packages": {"numpy": np.__version__, "scipy": scipy.__version__},
        "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
    }


def _require_runtime() -> dict[str, Any]:
    observed = _runtime()
    if observed != REQUIRED_RUNTIME:
        raise PermissionError("runtime differs from the frozen execution environment")
    return observed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _axis_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8") + b"\0")
    return digest.hexdigest()


def _set_sha256(values: Iterable[str]) -> str:
    return _axis_sha256(sorted(values))


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
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


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _git(*arguments: str, check: bool = True, text: bool = True) -> Any:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=check,
        capture_output=True,
        text=text,
    )


def _remote_tag_ids(tag: str) -> tuple[str, str]:
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
        raise PermissionError(f"public annotated tag {tag} is absent")
    return tag_object, commit


def _require_public_tag(tag: str, paths: Sequence[Path]) -> str:
    if _git("cat-file", "-t", tag).stdout.strip() != "tag":
        raise PermissionError(f"local tag {tag} is not annotated")
    local_object = _git("rev-parse", f"refs/tags/{tag}").stdout.strip()
    local_commit = _git("rev-parse", f"{tag}^{{}}").stdout.strip()
    if (local_object, local_commit) != _remote_tag_ids(tag):
        raise PermissionError(f"public tag {tag} differs from the local tag")
    for path in paths:
        published = _git("show", f"{tag}:{_relative(path)}", text=False).stdout
        if published != path.read_bytes():
            raise PermissionError(f"{_relative(path)} differs from public tag {tag}")
    return local_commit


def _published_bytes(tag: str, path: Path) -> bytes:
    return _git("show", f"{tag}:{_relative(path)}", text=False).stdout


def _require_ancestor(ancestor: str, descendant: str) -> None:
    if _git(
        "merge-base", "--is-ancestor", ancestor, descendant, check=False
    ).returncode:
        raise PermissionError("campaign tags do not form the required ancestry")


def _verify_v1_terminal_ancestry(candidate_commit: str) -> dict[str, str]:
    terminal_commit = _require_public_tag(
        V1_TERMINAL_TAG,
        (V1_DESIGNATION, V1_SOURCE_MANIFEST, V1_AXIS_ACCESS, V1_AXIS_REFUSAL),
    )
    if terminal_commit != V1_TERMINAL_COMMIT:
        raise PermissionError("v1 terminal refusal commit differs")
    _require_ancestor(terminal_commit, candidate_commit)
    return {
        "terminal_refusal_tag": V1_TERMINAL_TAG,
        "terminal_refusal_commit": terminal_commit,
    }


def freeze_implementation() -> dict[str, Any]:
    """Write the manifest that must be committed and publicly tagged before access."""

    if IMPLEMENTATION_FREEZE.exists():
        raise FileExistsError("implementation freeze already exists")
    candidate_commit = _require_public_tag(
        CANDIDATE_TAG, (DESIGNATION, SOURCE_MANIFEST, PROTOCOL)
    )
    terminal = _verify_v1_terminal_ancestry(candidate_commit)
    missing = [
        relative
        for relative in IMPLEMENTATION_BINDINGS
        if not (ROOT / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"implementation bindings are absent: {', '.join(missing)}"
        )
    payload = {
        "schema": "gse313642-hcc-pre-access-implementation/2.0",
        "status": "FROZEN_BEFORE_ANY_NUMERIC_MATRIX_ACCESS",
        "created_at_utc": _timestamp(),
        "candidate_tag": CANDIDATE_TAG,
        "candidate_commit": candidate_commit,
        "required_implementation_tag": IMPLEMENTATION_TAG,
        "implementation_files_sha256": {
            relative: _sha256(ROOT / relative) for relative in IMPLEMENTATION_BINDINGS
        },
        "required_runtime": REQUIRED_RUNTIME,
        "v1_terminal_axis_refusal": terminal,
        "matrix_body_or_header_requested": False,
        "rerun_permitted": False,
    }
    _write_json_x(IMPLEMENTATION_FREEZE, payload)
    return payload


def _verify_implementation() -> dict[str, str]:
    value = _read_json(IMPLEMENTATION_FREEZE)
    expected = {
        relative: _sha256(ROOT / relative) for relative in IMPLEMENTATION_BINDINGS
    }
    if (
        value.get("schema") != "gse313642-hcc-pre-access-implementation/2.0"
        or value.get("status") != "FROZEN_BEFORE_ANY_NUMERIC_MATRIX_ACCESS"
        or value.get("candidate_tag") != CANDIDATE_TAG
        or value.get("implementation_files_sha256") != expected
        or value.get("required_implementation_tag") != IMPLEMENTATION_TAG
        or value.get("required_runtime") != REQUIRED_RUNTIME
        or value.get("matrix_body_or_header_requested") is not False
        or value.get("rerun_permitted") is not False
    ):
        raise PermissionError("implementation freeze differs")
    candidate_commit = _require_public_tag(
        CANDIDATE_TAG, (DESIGNATION, SOURCE_MANIFEST, PROTOCOL)
    )
    terminal = _verify_v1_terminal_ancestry(candidate_commit)
    if (
        value.get("candidate_commit") != candidate_commit
        or value.get("v1_terminal_axis_refusal") != terminal
    ):
        raise PermissionError("implementation freeze terminal ancestry differs")
    implementation_commit = _require_public_tag(
        IMPLEMENTATION_TAG,
        tuple(ROOT / relative for relative in IMPLEMENTATION_BINDINGS)
        + (IMPLEMENTATION_FREEZE,),
    )
    _require_ancestor(candidate_commit, implementation_commit)
    return {
        "candidate_commit": candidate_commit,
        "implementation_commit": implementation_commit,
    }


def _write_json_x(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _append_jsonl(
    path: Path, value: Mapping[str, Any], *, create: bool = False
) -> None:
    encoded = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_APPEND | (os.O_CREAT | os.O_EXCL if create else 0)
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "ab") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        value = json.loads(
            line,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
        if not isinstance(value, dict):
            raise PermissionError("access journal row is not an object")
        rows.append(value)
    if not rows:
        raise PermissionError("access journal is empty")
    return rows


def _v1_designation() -> dict[str, Any]:
    value = _read_json(V1_DESIGNATION)
    patients = value.get("patients")
    if (
        value.get("schema") != "gse313642-hcc-citeseq-candidate-designation/1.0"
        or value.get("status") != "METADATA_ONLY_DESIGNATED_OUTCOME_DISABLED"
        or not isinstance(patients, list)
        or len(patients) != 35
        or len({record.get("patient_id") for record in patients}) != 35
    ):
        raise PermissionError("v1 candidate designation differs")
    a33 = [record for record in patients if record.get("patient_id") == "A33"]
    if (
        len(a33) != 1
        or a33[0].get("deposited_patient_id") != "A-33"
        or a33[0].get("pair_stem") != "A-33-01_01"
        or a33[0].get("role") != "pilot"
        or a33[0].get("rank") != 6
    ):
        raise PermissionError("v1 A33 designation differs")
    return value


def _designation() -> dict[str, Any]:
    value = _read_json(DESIGNATION)
    patients = value.get("patients")
    disclosure = value.get("metadata_provenance", {}).get(
        "post_v1_pre_v2_axis_gate_disclosure"
    )
    if (
        value.get("schema") != "gse313642-hcc-citeseq-candidate-designation/2.0"
        or value.get("status") != "METADATA_ONLY_DESIGNATED_OUTCOME_DISABLED"
        or value.get("markers") != list(MARKERS)
        or value.get("entity_count") != 81
        or value.get("feature_schema") != FROZEN_FEATURE_SCHEMA
        or value.get("numeric_assay_matrix_entry_accessed_before_designation")
        is not False
        or not isinstance(patients, list)
        or len(patients) != 34
        or value.get("split_rule", {}).get("counts") != ROLE_COUNTS
        or value.get("remote_access", {}).get("series_raw_tar") != "FORBIDDEN"
        or not isinstance(disclosure, dict)
        or disclosure.get("terminal_refusal_tag") != V1_TERMINAL_TAG
        or disclosure.get("terminal_refusal_commit") != V1_TERMINAL_COMMIT
        or disclosure.get("terminal_refusal_path") != _relative(V1_AXIS_REFUSAL)
        or disclosure.get("axis_access_journal") != _relative(V1_AXIS_ACCESS)
        or disclosure.get("axis_access_journal_sha256") != V1_AXIS_ACCESS_SHA256
        or disclosure.get("axis_gets_completed") != 140
        or disclosure.get("retained_axis_files_already_accessed") != 136
        or disclosure.get("numeric_matrix_body_or_header_requested") is not False
        or disclosure.get("used_for_v2_input_exclusion") is not True
        or disclosure.get("used_for_split_marker_gate_or_hyperparameter_selection")
        is not False
    ):
        raise PermissionError("candidate designation differs from the frozen contract")

    v1 = _v1_designation()
    v1_by_id = {record["patient_id"]: record for record in v1["patients"]}
    patient_ids = [record.get("patient_id") for record in patients]
    if (
        len(set(patient_ids)) != 34
        or "A33" in patient_ids
        or set(patient_ids) != set(v1_by_id) - {"A33"}
        or any(record != v1_by_id[record["patient_id"]] for record in patients)
    ):
        raise PermissionError(
            "v2 must equal the v1 patient panel with only A33 removed"
        )

    by_role = {role: [] for role in ROLE_COUNTS}
    salt = value.get("split_rule", {}).get("salt")
    if not isinstance(salt, str) or salt != v1.get("split_rule", {}).get("salt"):
        raise PermissionError("split salt differs from v1")
    for record in patients:
        role = record.get("role")
        expected_hash = hashlib.sha256(
            (salt + "\0" + record["deposited_patient_id"]).encode("utf-8")
        ).hexdigest()
        if (
            role not in by_role
            or record.get("group") not in COHORTS
            or record.get("split_sha256") != expected_hash
        ):
            raise PermissionError("split rank, hash, or role differs")
        for modality in ("gex", "fb"):
            expected_bytes = record.get(f"{modality}_expected_bytes")
            if (
                not isinstance(expected_bytes, dict)
                or set(expected_bytes) != set(AXIS_MEMBERS + (MATRIX_MEMBER,))
                or any(
                    isinstance(size, bool) or not isinstance(size, int) or size <= 0
                    for size in expected_bytes.values()
                )
            ):
                raise PermissionError("official filelist byte binding differs")
        by_role[role].append(record)

    role_order = value.get("role_order", {})
    expected_order = {
        role: [patient for patient in v1["role_order"][role] if patient != "A33"]
        for role in ROLE_COUNTS
    }
    for role, count in ROLE_COUNTS.items():
        if (
            len(by_role[role]) != count
            or role_order.get(role) != expected_order[role]
            or set(role_order[role])
            != {record["patient_id"] for record in by_role[role]}
        ):
            raise PermissionError("patient role order differs")
    return value


def _v1_axis_manifest() -> tuple[
    dict[str, Any], dict[tuple[str, str, str], dict[str, Any]]
]:
    designation = _v1_designation()
    value = _read_json(V1_SOURCE_MANIFEST)
    files = value.get("files")
    if (
        value.get("schema") != "gse313642-hcc-source-manifest/1.0"
        or value.get("status")
        != "METADATA_ONLY_FROZEN_NO_NUMERIC_MATRIX_CONTENT_ACCESSED"
        or value.get("designation_sha256") != _sha256(V1_DESIGNATION)
        or value.get("selected_patient_count") != 35
        or value.get("selected_file_count") != 210
        or value.get("axis_file_count") != 140
        or value.get("matrix_file_count") != 70
        or not isinstance(files, list)
        or len(files) != 210
    ):
        raise PermissionError("v1 source manifest differs")
    mapped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for file_record in files:
        key = (
            file_record.get("patient_id"),
            file_record.get("modality"),
            file_record.get("member"),
        )
        if key in mapped:
            raise PermissionError("v1 source manifest contains duplicate files")
        mapped[key] = dict(file_record)
    expected = {
        (patient["patient_id"], modality, member)
        for patient in designation["patients"]
        for modality in ("GEX", "FB")
        for member in AXIS_MEMBERS + (MATRIX_MEMBER,)
    }
    if set(mapped) != expected:
        raise PermissionError("v1 source manifest file axis differs")
    for patient in designation["patients"]:
        for modality in ("GEX", "FB"):
            for member in AXIS_MEMBERS:
                record = mapped[(patient["patient_id"], modality, member)]
                if (
                    record.get("allowed_stage") != "axis_preflight"
                    or record.get("deposited_patient_id")
                    != patient["deposited_patient_id"]
                    or record.get("gsm")
                    != patient["gex_gsm" if modality == "GEX" else "fb_gsm"]
                    or record.get("filename") != _filename(patient, modality, member)
                    or record.get("expected_bytes")
                    != patient[
                        "gex_expected_bytes"
                        if modality == "GEX"
                        else "fb_expected_bytes"
                    ][member]
                ):
                    raise PermissionError("v1 source manifest axis binding differs")
    return designation, mapped


def _v1_terminal_refusal() -> dict[str, Any]:
    if (
        _sha256(V1_AXIS_ACCESS) != V1_AXIS_ACCESS_SHA256
        or _sha256(V1_AXIS_REFUSAL) != V1_AXIS_REFUSAL_SHA256
    ):
        raise PermissionError("v1 terminal evidence hash differs")
    value = _read_json(V1_AXIS_REFUSAL)
    failure = value.get("failure")
    access = value.get("axis_access")
    if (
        value.get("schema") != "gse313642-hcc-axis-preflight-refusal/1.0"
        or value.get("status") != "TERMINAL_AXIS_REFUSAL"
        or value.get("refusal_code") != "BARCODE_AXIS_NOT_UNIQUE"
        or value.get("matrix_body_or_header_requested") is not False
        or value.get("calibration_claimed") is not False
        or value.get("prediction_written") is not False
        or value.get("outcome_scored") is not False
        or value.get("rerun_permitted") is not False
        or not isinstance(failure, dict)
        or failure.get("patient_id") != "A33"
        or failure.get("deposited_patient_id") != "A-33"
        or failure.get("pair_stem") != "A-33-01_01"
        or failure.get("role") != "pilot"
        or not isinstance(access, dict)
        or access.get("journal") != _relative(V1_AXIS_ACCESS)
        or access.get("journal_sha256") != V1_AXIS_ACCESS_SHA256
        or access.get("axis_file_count") != 140
        or access.get("get_started") != 140
        or access.get("get_completed") != 140
        or access.get("gzip_parse_succeeded") != 140
        or access.get("failed_gets") != 0
        or access.get("matrix_requests") != 0
    ):
        raise PermissionError("v1 terminal refusal differs")
    return {
        "tag": V1_TERMINAL_TAG,
        "commit": V1_TERMINAL_COMMIT,
        "artifact_sha256": V1_AXIS_REFUSAL_SHA256,
        "refusal_code": "BARCODE_AXIS_NOT_UNIQUE",
        "excluded_patient_id": "A33",
    }


def _source_manifest() -> dict[tuple[str, str, str], dict[str, Any]]:
    value = _read_json(SOURCE_MANIFEST)
    files = value.get("files")
    access = value.get("post_v1_pre_v2_axis_payload_access")
    disclosure = value.get("post_v1_pre_v2_axis_gate_disclosure")
    if (
        value.get("schema") != "gse313642-hcc-source-manifest/2.0"
        or value.get("status")
        != "METADATA_ONLY_FROZEN_NO_NUMERIC_MATRIX_CONTENT_ACCESSED"
        or value.get("designation_sha256") != _sha256(DESIGNATION)
        or value.get("numeric_matrix_content_accessed_before_manifest") is not False
        or value.get("matrix_market_header_accessed_before_manifest") is not False
        or value.get("selected_patient_count") != 34
        or value.get("selected_file_count") != 204
        or value.get("axis_file_count") != 136
        or value.get("matrix_file_count") != 68
        or value.get("series_raw_tar") != "FORBIDDEN"
        or value.get("individual_url_template")
        != _designation()["remote_access"]["individual_url_template"]
        or value.get("official_filelist")
        != {
            "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE313nnn/GSE313642/suppl/filelist.txt",
            "bytes": 43340,
            "sha256": "8fb88cd52c30eb91a8da21caff1352c1307a47dc7ac0ddcc3fdd9960c9141aa9",
        }
        or value.get("request_contract")
        != {
            "only_individual_supplementary_urls": True,
            "expected_content_length_must_equal_official_filelist_bytes": True,
            "http_redirects": False,
            "automatic_retries": False,
            "observed_bytes_and_sha256_recorded_before_parse": True,
            "axis_files_reused_from_disclosed_v1_access": True,
            "one_get_per_unopened_matrix_per_permitted_stage": True,
        }
        or not isinstance(access, dict)
        or access.get("journal") != _relative(V1_AXIS_ACCESS)
        or access.get("journal_sha256") != V1_AXIS_ACCESS_SHA256
        or access.get("axis_gets_completed") != 140
        or access.get("retained_axis_files") != 136
        or access.get("excluded_axis_files") != 4
        or access.get("numeric_matrix_values_accessed") is not False
        or access.get("matrix_market_header_accessed") is not False
        or access.get("used_for_v2_input_exclusion") is not True
        or access.get("used_for_split_marker_gate_or_hyperparameter_selection")
        is not False
        or not isinstance(disclosure, dict)
        or disclosure.get("terminal_refusal_tag") != V1_TERMINAL_TAG
        or disclosure.get("terminal_refusal_commit") != V1_TERMINAL_COMMIT
        or disclosure.get("axis_access_journal") != _relative(V1_AXIS_ACCESS)
        or disclosure.get("axis_access_journal_sha256") != V1_AXIS_ACCESS_SHA256
        or disclosure.get("retained_axis_files_already_accessed") != 136
        or disclosure.get("numeric_matrix_body_or_header_requested") is not False
        or not isinstance(files, list)
        or len(files) != 204
    ):
        raise PermissionError("source manifest differs from the frozen contract")
    mapped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in files:
        key = (record.get("patient_id"), record.get("modality"), record.get("member"))
        if (
            key in mapped
            or key[1] not in ("GEX", "FB")
            or key[2] not in AXIS_MEMBERS + (MATRIX_MEMBER,)
            or isinstance(record.get("expected_bytes"), bool)
            or not isinstance(record.get("expected_bytes"), int)
            or record["expected_bytes"] <= 0
            or not isinstance(record.get("filename"), str)
            or not isinstance(record.get("gsm"), str)
        ):
            raise PermissionError("source manifest file record differs")
        mapped[key] = dict(record)
    designation = _designation()
    expected_keys = {
        (record["patient_id"], modality, member)
        for record in designation["patients"]
        for modality in ("GEX", "FB")
        for member in AXIS_MEMBERS + (MATRIX_MEMBER,)
    }
    if set(mapped) != expected_keys:
        raise PermissionError("source manifest file axis differs")
    prior_hashes = designation["metadata_provenance"]["pre_v1_designation_axis_access"]
    for patient in designation["patients"]:
        for modality in ("GEX", "FB"):
            for member in AXIS_MEMBERS + (MATRIX_MEMBER,):
                record = mapped[(patient["patient_id"], modality, member)]
                expected_stage = (
                    "v2_reuse_only_no_get"
                    if member in AXIS_MEMBERS
                    else "source_calibration"
                    if patient["role"] == "calibration"
                    else "source_pilot_after_calibration_freeze"
                    if patient["role"] == "pilot"
                    else "held_gex_prediction"
                    if modality == "GEX"
                    else "held_fb_score_after_prediction_freeze"
                )
                if record.get("allowed_stage") != expected_stage:
                    raise PermissionError("source manifest stage boundary differs")
                expected_gsm = patient["gex_gsm" if modality == "GEX" else "fb_gsm"]
                expected_access = (
                    "v1_axis_preflight" if member in AXIS_MEMBERS else "none"
                )
                expected_prior_hash = None
                if patient["patient_id"] == "A07" and member in AXIS_MEMBERS:
                    axis_name = (
                        "barcodes" if member.startswith("barcodes") else "features"
                    )
                    expected_prior_hash = prior_hashes[
                        f"{modality.lower()}_{axis_name}_gzip_sha256"
                    ]
                if (
                    record.get("patient_id") != patient["patient_id"]
                    or record.get("deposited_patient_id")
                    != patient["deposited_patient_id"]
                    or record.get("role") != patient["role"]
                    or record.get("group") != patient["group"]
                    or record.get("gsm") != expected_gsm
                    or record.get("filename") != _filename(patient, modality, member)
                    or record.get("expected_bytes")
                    != patient[
                        "gex_expected_bytes"
                        if modality == "GEX"
                        else "fb_expected_bytes"
                    ][member]
                    or record.get("pre_manifest_payload_access") != expected_access
                    or record.get("observed_gzip_sha256") != expected_prior_hash
                    or record.get("v2_get_authorized") != (member == MATRIX_MEMBER)
                ):
                    raise PermissionError("source manifest file binding differs")
    return mapped


def _manifest_file(
    record: Mapping[str, Any], modality: str, member: str
) -> dict[str, Any]:
    manifest_record = _source_manifest()[(record["patient_id"], modality, member)]
    expected_filename = _filename(record, modality, member)
    expected_bytes = record[
        "gex_expected_bytes" if modality == "GEX" else "fb_expected_bytes"
    ][member]
    if (
        manifest_record["filename"] != expected_filename
        or manifest_record["expected_bytes"] != expected_bytes
        or manifest_record["gsm"]
        != record["gex_gsm" if modality == "GEX" else "fb_gsm"]
        or manifest_record["role"] != record["role"]
        or manifest_record["group"] != record["group"]
        or manifest_record["deposited_patient_id"] != record["deposited_patient_id"]
    ):
        raise PermissionError("designation and source manifest differ")
    return manifest_record


def _filename(record: Mapping[str, Any], modality: str, member: str) -> str:
    gsm = str(record["gex_gsm" if modality == "GEX" else "fb_gsm"])
    return f"{gsm}_{record['pair_stem']}_{modality}_{member}"


def _url(record: Mapping[str, Any], modality: str, member: str) -> str:
    file_record = _manifest_file(record, modality, member)
    return _manifest_url(file_record)


def _manifest_url(file_record: Mapping[str, Any]) -> str:
    gsm = file_record["gsm"]
    filename = urllib.parse.quote(file_record["filename"], safe="-_.")
    return f"{BASE_URL}/{gsm}/suppl/{filename}"


def _decode_gzip_lines(path: Path) -> tuple[bytes, list[str]]:
    compressed_sha = _sha256(path)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        lines = [line.rstrip("\r\n") for line in stream]
    if not lines or any(not line for line in lines):
        raise ProtocolRefusal("AXIS_EMPTY_OR_BLANK")
    return bytes.fromhex(compressed_sha), lines


def _feature_axis(
    lines: Sequence[str], modality: str
) -> tuple[tuple[str, ...], str, dict[str, str]]:
    output = []
    selected_descriptions = {}
    expected_type = "Gene Expression" if modality == "GEX" else "Antibody Capture"
    for line in lines:
        fields = line.split("\t")
        if len(fields) != 3 or fields[2] != expected_type:
            raise ProtocolRefusal("FEATURE_TYPE_OR_SCHEMA_MISMATCH")
        name = fields[1] if modality == "GEX" else fields[0]
        if (
            modality == "FB"
            and name in FB_MARKER_DESCRIPTIONS
            and fields[1] != FB_MARKER_DESCRIPTIONS[name]
        ):
            raise ProtocolRefusal("FB_REAGENT_DESCRIPTION_MISMATCH")
        if modality == "FB" and name in FB_MARKER_DESCRIPTIONS:
            selected_descriptions[name] = fields[1]
        output.append(name)
    return tuple(output), expected_type, selected_descriptions


def _marker_rows(names: Sequence[str]) -> list[int]:
    rows = []
    for marker in MARKERS:
        matches = [index + 1 for index, value in enumerate(names) if value == marker]
        if len(matches) != 1:
            raise ProtocolRefusal("MARKER_NOT_UNIQUE")
        rows.append(matches[0])
    return rows


def _axis_paths(
    axis_root: Path, record: Mapping[str, Any], modality: str
) -> tuple[Path, Path]:
    return (
        axis_root / _filename(record, modality, "barcodes.tsv.gz"),
        axis_root / _filename(record, modality, "features.tsv.gz"),
    )


def _inspect_pair_axes(axis_root: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    axes: dict[str, dict[str, Any]] = {}
    barcode_values: dict[str, tuple[str, ...]] = {}
    for modality in ("GEX", "FB"):
        barcode_path, feature_path = _axis_paths(axis_root, record, modality)
        if not barcode_path.is_file() or not feature_path.is_file():
            raise ProtocolRefusal("AXIS_FILE_MISSING")
        barcode_manifest = _manifest_file(record, modality, "barcodes.tsv.gz")
        feature_manifest = _manifest_file(record, modality, "features.tsv.gz")
        if (
            barcode_path.stat().st_size != barcode_manifest["expected_bytes"]
            or feature_path.stat().st_size != feature_manifest["expected_bytes"]
        ):
            raise ProtocolRefusal("AXIS_OFFICIAL_SIZE_MISMATCH")
        _, barcodes = _decode_gzip_lines(barcode_path)
        _, feature_lines = _decode_gzip_lines(feature_path)
        if len(set(barcodes)) != len(barcodes):
            raise ProtocolRefusal("BARCODE_AXIS_NOT_UNIQUE")
        names, feature_type, descriptions = _feature_axis(feature_lines, modality)
        barcode_sha256 = _sha256(barcode_path)
        feature_sha256 = _sha256(feature_path)
        if barcode_manifest.get("observed_gzip_sha256") not in (
            None,
            barcode_sha256,
        ) or feature_manifest.get("observed_gzip_sha256") not in (None, feature_sha256):
            raise ProtocolRefusal("DISCLOSED_AXIS_HASH_MISMATCH")
        barcode_values[modality] = tuple(barcodes)
        axes[modality] = {
            "barcode_file": barcode_path.name,
            "barcode_url": _url(record, modality, "barcodes.tsv.gz"),
            "barcode_expected_bytes": barcode_manifest["expected_bytes"],
            "barcode_observed_bytes": barcode_path.stat().st_size,
            "barcode_gzip_sha256": barcode_sha256,
            "barcode_count": len(barcodes),
            "barcode_axis_sha256": _axis_sha256(barcodes),
            "barcode_set_sha256": _set_sha256(barcodes),
            "feature_file": feature_path.name,
            "feature_url": _url(record, modality, "features.tsv.gz"),
            "feature_expected_bytes": feature_manifest["expected_bytes"],
            "feature_observed_bytes": feature_path.stat().st_size,
            "feature_gzip_sha256": feature_sha256,
            "feature_count": len(names),
            "feature_axis_sha256": _axis_sha256(names),
            "feature_set_sha256": _set_sha256(names),
            "feature_type": feature_type,
            "marker_rows_1based": _marker_rows(names),
            "selected_marker_descriptions": descriptions,
        }
    if set(barcode_values["GEX"]) != set(barcode_values["FB"]):
        raise ProtocolRefusal("PAIRED_BARCODE_SETS_DIFFER")
    if len(barcode_values["GEX"]) < CELL_BUDGET:
        raise ProtocolRefusal("FEWER_THAN_512_SHARED_BARCODES")
    selected = select_barcodes(
        barcode_values["GEX"],
        str(record["deposited_patient_id"]),
        count=CELL_BUDGET,
    )
    return {
        "patient_id": record["patient_id"],
        "role": record["role"],
        "group": record["group"],
        "pair_stem": record["pair_stem"],
        "axes": axes,
        "barcode_sets_exactly_equal": True,
        "barcode_orders_equal": barcode_values["GEX"] == barcode_values["FB"],
        "selected_barcode_axis_sha256": _axis_sha256(selected),
        "selected_barcode_count": CELL_BUDGET,
    }


def _validate_axis_access(
    axis_root: Path, designation: Mapping[str, Any]
) -> dict[str, Any]:
    v1_designation, v1_manifest = _v1_axis_manifest()
    terminal = _v1_terminal_refusal()
    rows = _read_jsonl(V1_AXIS_ACCESS)
    header = rows[0]
    if (
        header.get("schema") != "gse313642-hcc-axis-access/1.0"
        or header.get("stage") != "axis_acquisition"
        or header.get("event") != "OPENED_BEFORE_FIRST_AXIS_GET"
        or header.get("matrix_requests") != 0
        or header.get("series_tar_used") is not False
    ):
        raise PermissionError("axis access header differs")
    expected = [
        (record, modality, member)
        for record in v1_designation["patients"]
        for modality in ("GEX", "FB")
        for member in AXIS_MEMBERS
    ]
    retained_keys = [
        (record["patient_id"], modality, member)
        for record in designation["patients"]
        for modality in ("GEX", "FB")
        for member in AXIS_MEMBERS
    ]
    retained_set = set(retained_keys)
    if len(expected) != 140 or len(rows) != 421:
        raise PermissionError("axis access did not use exactly 140 one-shot GETs")
    historical_files: dict[tuple[str, str, str], dict[str, Any]] = {}
    for index, (record, modality, member) in enumerate(expected):
        started = rows[1 + 3 * index]
        completed = rows[2 + 3 * index]
        parsed = rows[3 + 3 * index]
        key = (record["patient_id"], modality, member)
        path = axis_root / _filename(record, modality, member)
        manifest = v1_manifest[(record["patient_id"], modality, member)]
        url = _manifest_url(manifest)
        decoded = None
        if key in retained_set and path.is_file():
            _, decoded = _decode_gzip_lines(path)
        digest = completed.get("sha256")
        if (
            started.get("event") != "GET_STARTED"
            or started.get("stage") != "axis_acquisition"
            or started.get("patient_id") != record["patient_id"]
            or started.get("modality") != modality
            or started.get("member") != member
            or started.get("url") != url
            or started.get("expected_bytes") != manifest["expected_bytes"]
            or completed.get("event") != "GET_COMPLETED"
            or completed.get("stage") != "axis_acquisition"
            or completed.get("patient_id") != record["patient_id"]
            or completed.get("modality") != modality
            or completed.get("member") != member
            or completed.get("bytes") != manifest["expected_bytes"]
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or manifest.get("observed_gzip_sha256") not in (None, digest)
            or parsed.get("event") != "GZIP_PARSE_SUCCEEDED"
            or parsed.get("stage") != "axis_acquisition"
            or parsed.get("patient_id") != record["patient_id"]
            or parsed.get("modality") != modality
            or parsed.get("member") != member
            or parsed.get("download_sha256") != digest
            or not isinstance(parsed.get("line_count"), int)
            or parsed.get("line_count") <= 0
            or (
                key in retained_set
                and (
                    not path.is_file()
                    or completed.get("bytes") != path.stat().st_size
                    or digest != _sha256(path)
                    or decoded is None
                    or parsed.get("line_count") != len(decoded)
                )
            )
        ):
            raise PermissionError("axis access file binding differs")
        historical_files[key] = {
            "patient_id": record["patient_id"],
            "modality": modality,
            "member": member,
            "url": url,
            "bytes": completed["bytes"],
            "sha256": completed["sha256"],
        }

    excluded = set(historical_files) - retained_set
    if (
        len(retained_keys) != 136
        or len(retained_set) != 136
        or {key[0] for key in retained_set}
        != {record["patient_id"] for record in designation["patients"]}
        or "A33" in {key[0] for key in retained_set}
        or excluded
        != {
            ("A33", modality, member)
            for modality in ("GEX", "FB")
            for member in AXIS_MEMBERS
        }
    ):
        raise PermissionError("v2 retained axis set does not exclude exactly A33")
    current_manifest = _source_manifest()
    files = []
    for key in retained_keys:
        historical = historical_files[key]
        current = current_manifest[key]
        if (
            current.get("allowed_stage") != "v2_reuse_only_no_get"
            or current.get("v2_get_authorized") is not False
            or current.get("expected_bytes") != historical["bytes"]
            or _manifest_url(current) != historical["url"]
        ):
            raise PermissionError("retained v2 axis differs from v1 access")
        files.append(historical)
    return {
        "v1_journal_rows": len(rows),
        "v1_axis_gets": len(expected),
        "v2_axis_gets": 0,
        "retained_axis_files": len(files),
        "excluded_axis_files": 4,
        "matrix_gets": 0,
        "journal_sha256": _sha256(V1_AXIS_ACCESS),
        "terminal_refusal": terminal,
        "files": files,
    }


def preflight(axis_root: Path, *, output: Path = AXIS_PREFLIGHT) -> dict[str, Any]:
    """Validate all supplied axes before any matrix request."""

    tags = _verify_implementation()
    terminal = {
        "terminal_refusal_tag": V1_TERMINAL_TAG,
        "terminal_refusal_commit": V1_TERMINAL_COMMIT,
    }
    root = axis_root.expanduser().resolve()
    designation = _designation()
    access_audit = _validate_axis_access(root, designation)
    if output.exists() or any(
        path.exists() for path in (SOURCE_ATTEMPT, SOURCE_RESULT)
    ):
        raise FileExistsError("axis preflight or a downstream source artifact exists")
    records = [_inspect_pair_axes(root, record) for record in designation["patients"]]
    payload = {
        "schema": "gse313642-hcc-axis-preflight/2.0",
        "status": "PASS_BEFORE_ANY_MATRIX_REQUEST",
        "created_at_utc": _timestamp(),
        "designation_sha256": _sha256(DESIGNATION),
        "implementation_commit": tags["implementation_commit"],
        "patient_count": 34,
        "matrix_body_or_header_requested": False,
        "series_tar_used": False,
        "axis_access": access_audit,
        "v1_terminal_axis_refusal": terminal,
        "required_runtime": REQUIRED_RUNTIME,
        "patients": records,
    }
    _write_json_x(output, payload)
    return payload


def _validate_preflight(axis_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    designation = _designation()
    access_audit = _validate_axis_access(axis_root, designation)
    preflight_value = _read_json(AXIS_PREFLIGHT)
    if (
        preflight_value.get("schema") != "gse313642-hcc-axis-preflight/2.0"
        or preflight_value.get("status") != "PASS_BEFORE_ANY_MATRIX_REQUEST"
        or preflight_value.get("designation_sha256") != _sha256(DESIGNATION)
        or not isinstance(preflight_value.get("implementation_commit"), str)
        or len(preflight_value["implementation_commit"]) != 40
        or preflight_value.get("patient_count") != 34
        or preflight_value.get("matrix_body_or_header_requested") is not False
        or preflight_value.get("axis_access") != access_audit
        or preflight_value.get("v1_terminal_axis_refusal")
        != {
            "terminal_refusal_tag": V1_TERMINAL_TAG,
            "terminal_refusal_commit": V1_TERMINAL_COMMIT,
        }
    ):
        raise PermissionError("axis preflight differs")
    observed = [
        _inspect_pair_axes(axis_root, record) for record in designation["patients"]
    ]
    if observed != preflight_value.get("patients"):
        raise PermissionError("supplied axes differ from the frozen preflight")
    return designation, preflight_value


def _token_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        return resolved
    raise PermissionError("claim token must be outside the repository")


def _private_state_path(path: Path, scratch: Path) -> Path:
    resolved = _token_path(path)
    try:
        resolved.relative_to(scratch.expanduser().resolve())
    except ValueError:
        return resolved
    raise PermissionError("private state must be outside the scratch directory")


def _access_header(
    stage: str, created_at_utc: str, runtime: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema": "gse313642-hcc-access/2.0",
        "stage": stage,
        "event": "OPENED_BEFORE_MATRIX_ACCESS",
        "created_at_utc": created_at_utc,
        "runtime": dict(runtime),
        "one_streaming_get_per_matrix": True,
        "automatic_retries": False,
        "http_redirects": False,
    }


def _claim(stage: str, token_path: Path) -> dict[str, Any]:
    runtime = _require_runtime()
    if stage == "calibration":
        tags = _verify_implementation()
        preflight_commit = _require_public_tag(
            PREFLIGHT_TAG, (V1_AXIS_ACCESS, V1_AXIS_REFUSAL, AXIS_PREFLIGHT)
        )
        _require_ancestor(tags["implementation_commit"], preflight_commit)
        attempt, result, access = (
            CALIBRATION_ATTEMPT,
            CALIBRATION_SELECTION,
            CALIBRATION_ACCESS,
        )
        if not AXIS_PREFLIGHT.exists():
            raise PermissionError("axis preflight is absent")
        prerequisite = {
            "axis_preflight_sha256": _sha256(AXIS_PREFLIGHT),
            "implementation_commit": tags["implementation_commit"],
            "axis_preflight_commit": preflight_commit,
        }
    elif stage == "source":
        attempt, result, access = SOURCE_ATTEMPT, SOURCE_RESULT, SOURCE_ACCESS
        authorization = _read_json(PILOT_AUTHORIZATION)
        if (
            authorization.get("schema") != "gse313642-hcc-pilot-authorization/2.0"
            or authorization.get("status")
            != "AUTHORIZED_AFTER_PUBLIC_CALIBRATION_FREEZE"
            or authorization.get("calibration_selection_sha256")
            != _sha256(CALIBRATION_SELECTION)
            or authorization.get("pilot_matrix_gets_authorized") != 22
        ):
            raise PermissionError("pilot authorization differs from calibration")
        pilot_authorization_commit = _require_public_tag(
            PILOT_AUTHORIZATION_TAG, (PILOT_AUTHORIZATION,)
        )
        _require_ancestor(
            authorization["calibration_commit"], pilot_authorization_commit
        )
        prerequisite = {
            "pilot_authorization_sha256": _sha256(PILOT_AUTHORIZATION),
            "pilot_authorization_commit": pilot_authorization_commit,
        }
    elif stage == "prediction":
        attempt, result, access = (
            PREDICTION_ATTEMPT,
            PREDICTION_RESULT,
            PREDICTION_ACCESS,
        )
        authorization = _read_json(PREDICTION_AUTHORIZATION)
        if (
            authorization.get("schema") != "gse313642-hcc-prediction-authorization/2.0"
            or authorization.get("status") != "AUTHORIZED_AFTER_PUBLIC_SOURCE_PASS"
            or authorization.get("source_result_sha256") != _sha256(SOURCE_RESULT)
            or authorization.get("held_gex_gets_authorized") != 12
            or authorization.get("held_fb_gets_authorized") != 0
        ):
            raise PermissionError("prediction authorization differs from source")
        prediction_authorization_commit = _require_public_tag(
            PREDICTION_AUTHORIZATION_TAG, (PREDICTION_AUTHORIZATION,)
        )
        _require_ancestor(
            authorization["source_commit"], prediction_authorization_commit
        )
        prerequisite = {
            "prediction_authorization_sha256": _sha256(PREDICTION_AUTHORIZATION),
            "prediction_authorization_commit": prediction_authorization_commit,
        }
    elif stage == "score":
        attempt, result, access = SCORE_ATTEMPT, SCORE_RESULT, SCORE_ACCESS
        authorization = _read_json(SCORE_AUTHORIZATION)
        if (
            authorization.get("schema") != "gse313642-hcc-score-authorization/2.0"
            or authorization.get("status")
            != "AUTHORIZED_AFTER_PUBLIC_PREDICTION_FREEZE"
            or authorization.get("prediction_result_sha256")
            != _sha256(PREDICTION_RESULT)
            or authorization.get("held_fb_gets_authorized") != 12
            or authorization.get("held_gex_gets_authorized") != 0
        ):
            raise PermissionError("score authorization differs from predictions")
        score_authorization_commit = _require_public_tag(
            SCORE_AUTHORIZATION_TAG, (SCORE_AUTHORIZATION,)
        )
        _require_ancestor(
            authorization["prediction_commit"], score_authorization_commit
        )
        prerequisite = {
            "score_authorization_sha256": _sha256(SCORE_AUTHORIZATION),
            "score_authorization_commit": score_authorization_commit,
        }
    else:
        raise ValueError("unknown stage")
    if any(path.exists() for path in (attempt, result, access)):
        raise FileExistsError(f"{stage} stage was already claimed or completed")
    token = _token_path(token_path)
    token.parent.mkdir(parents=True, exist_ok=True)
    raw = secrets.token_bytes(32)
    with token.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    created_at_utc = _timestamp()
    payload = {
        "schema": f"gse313642-hcc-{stage}-attempt/2.0",
        "status": "CLAIMED_BEFORE_MATRIX_ACCESS",
        "created_at_utc": created_at_utc,
        "designation_sha256": _sha256(DESIGNATION),
        "claim_token_sha256": hashlib.sha256(raw).hexdigest(),
        "one_get_per_matrix": True,
        "automatic_retries": False,
        "series_tar_used": False,
        "runtime": runtime,
        "rerun_permitted": False,
        **prerequisite,
    }
    _write_json_x(attempt, payload)
    _append_jsonl(
        access,
        _access_header(stage, created_at_utc, runtime),
        create=True,
    )
    return payload


def claim_source(token_path: Path) -> dict[str, Any]:
    return _claim("source", token_path)


def claim_calibration(token_path: Path) -> dict[str, Any]:
    return _claim("calibration", token_path)


def claim_prediction(token_path: Path) -> dict[str, Any]:
    return _claim("prediction", token_path)


def claim_score(token_path: Path) -> dict[str, Any]:
    return _claim("score", token_path)


def _require_public_attempt(stage: str) -> str:
    configuration = {
        "calibration": (
            CALIBRATION_ATTEMPT_TAG,
            CALIBRATION_ATTEMPT,
            CALIBRATION_ACCESS,
            "axis_preflight_commit",
        ),
        "source": (
            SOURCE_ATTEMPT_TAG,
            SOURCE_ATTEMPT,
            SOURCE_ACCESS,
            "pilot_authorization_commit",
        ),
        "prediction": (
            PREDICTION_ATTEMPT_TAG,
            PREDICTION_ATTEMPT,
            PREDICTION_ACCESS,
            "prediction_authorization_commit",
        ),
        "score": (
            SCORE_ATTEMPT_TAG,
            SCORE_ATTEMPT,
            SCORE_ACCESS,
            "score_authorization_commit",
        ),
    }
    if stage not in configuration:
        raise ValueError("unknown stage")
    tag, attempt_path, access_path, upstream_key = configuration[stage]
    attempt = _read_json(attempt_path)
    upstream_commit = attempt.get(upstream_key)
    if not isinstance(upstream_commit, str) or len(upstream_commit) != 40:
        raise PermissionError("attempt upstream commit is absent")
    expected_header = _access_header(
        stage, str(attempt.get("created_at_utc")), _require_runtime()
    )
    attempt_commit = _require_public_tag(tag, (attempt_path,))
    encoded_header = (
        json.dumps(
            expected_header,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()
    if _published_bytes(
        tag, access_path
    ) != encoded_header or not access_path.read_bytes().startswith(encoded_header):
        raise PermissionError("public attempt does not bind the access header")
    prerequisite_paths = {
        "calibration": ("axis_preflight_sha256", AXIS_PREFLIGHT),
        "source": ("pilot_authorization_sha256", PILOT_AUTHORIZATION),
        "prediction": (
            "prediction_authorization_sha256",
            PREDICTION_AUTHORIZATION,
        ),
        "score": ("score_authorization_sha256", SCORE_AUTHORIZATION),
    }
    hash_key, prerequisite_path = prerequisite_paths[stage]
    token_hash = attempt.get("claim_token_sha256")
    if (
        attempt.get("schema") != f"gse313642-hcc-{stage}-attempt/2.0"
        or attempt.get("status") != "CLAIMED_BEFORE_MATRIX_ACCESS"
        or attempt.get("designation_sha256") != _sha256(DESIGNATION)
        or attempt.get(hash_key) != _sha256(prerequisite_path)
        or attempt.get("runtime") != _require_runtime()
        or attempt.get("one_get_per_matrix") is not True
        or attempt.get("automatic_retries") is not False
        or attempt.get("series_tar_used") is not False
        or attempt.get("rerun_permitted") is not False
        or not isinstance(token_hash, str)
        or len(token_hash) != 64
        or any(character not in "0123456789abcdef" for character in token_hash)
    ):
        raise PermissionError("public attempt differs from the frozen contract")
    _require_ancestor(upstream_commit, attempt_commit)
    return attempt_commit


def _consume(
    stage: str,
    token_path: Path,
    scratch: Path,
    private_state: Path | None = None,
) -> None:
    paths = {
        "calibration": (CALIBRATION_ATTEMPT, CALIBRATION_CONSUMPTION),
        "source": (SOURCE_ATTEMPT, SOURCE_CONSUMPTION),
        "prediction": (PREDICTION_ATTEMPT, PREDICTION_CONSUMPTION),
        "score": (SCORE_ATTEMPT, SCORE_CONSUMPTION),
    }
    if stage not in paths:
        raise ValueError("unknown stage")
    if (stage in {"prediction", "score"}) != (private_state is not None):
        raise ValueError("private state binding differs from the stage")
    attempt_path, consumption = paths[stage]
    attempt = _read_json(attempt_path)
    token = _token_path(token_path)
    raw = token.read_bytes()
    if hashlib.sha256(raw).hexdigest() != attempt.get("claim_token_sha256"):
        raise PermissionError("claim token differs")
    payload = {
        "schema": f"gse313642-hcc-{stage}-consumption/2.0",
        "status": "CONSUMED_BEFORE_FIRST_MATRIX_REQUEST",
        "consumed_at_utc": _timestamp(),
        "attempt_sha256": _sha256(attempt_path),
        "scratch_identity_sha256": hashlib.sha256(
            str(scratch.expanduser().resolve()).encode("utf-8")
        ).hexdigest(),
        "rerun_permitted": False,
    }
    if private_state is not None:
        payload["private_state_identity_sha256"] = hashlib.sha256(
            str(private_state).encode("utf-8")
        ).hexdigest()
        if stage == "prediction":
            if private_state.exists():
                raise FileExistsError("private prediction state already exists")
            payload["private_state_expected_absent_before_prediction"] = True
        else:
            if not private_state.is_file():
                raise FileNotFoundError("private prediction state is absent")
            payload["private_state_sha256"] = _sha256(private_state)
            payload["private_state_bytes"] = private_state.stat().st_size
    _write_json_x(consumption, payload)
    token.unlink()


@contextmanager
def _stage_lock(scratch: Path, stage: str) -> Iterable[None]:
    scratch.parent.mkdir(parents=True, exist_ok=True)
    lock_path = scratch.parent / f".{scratch.name}.{stage}.lock"
    with lock_path.open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PermissionError("stage is already running") from error
        yield


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        return None


def _open_url(request: urllib.request.Request) -> BinaryIO:
    return urllib.request.build_opener(_NoRedirect).open(request, timeout=180)


def _download_matrix(
    record: Mapping[str, Any],
    modality: str,
    destination: Path,
    journal: Path,
    stage: str,
) -> dict[str, Any]:
    url = _url(record, modality, MATRIX_MEMBER)
    file_record = _manifest_file(record, modality, MATRIX_MEMBER)
    expected_stage = {
        ("calibration", "calibration"): "source_calibration",
        ("source", "pilot"): "source_pilot_after_calibration_freeze",
        ("prediction", "held"): "held_gex_prediction",
        ("score", "held"): "held_fb_score_after_prediction_freeze",
    }.get((stage, record["role"]))
    if (
        file_record["allowed_stage"] != expected_stage
        or file_record.get("v2_get_authorized") is not True
    ):
        raise PermissionError("matrix file is not authorized for this stage")
    expected_bytes = int(file_record["expected_bytes"])
    _append_jsonl(
        journal,
        {
            "stage": stage,
            "event": "GET_STARTED",
            "patient_id": record["patient_id"],
            "modality": modality,
            "url": url,
            "expected_bytes": expected_bytes,
        },
    )
    digest = hashlib.sha256()
    byte_count = 0
    request = urllib.request.Request(
        url, method="GET", headers={"Accept-Encoding": "identity"}
    )
    try:
        with _open_url(request) as response, destination.open("xb") as stream:
            final_url = response.geturl()
            status = getattr(response, "status", response.getcode())
            if status != 200 or final_url != url:
                raise ProtocolRefusal("MATRIX_RESPONSE_OR_REDIRECT_REFUSAL")
            declared = response.headers.get("Content-Length")
            if (
                declared is None
                or not declared.isdigit()
                or int(declared) != expected_bytes
            ):
                raise ProtocolRefusal("MATRIX_CONTENT_LENGTH_REFUSAL")
            for block in iter(lambda: response.read(8 << 20), b""):
                stream.write(block)
                digest.update(block)
                byte_count += len(block)
            stream.flush()
            os.fsync(stream.fileno())
            if byte_count != expected_bytes:
                raise ProtocolRefusal("MATRIX_SIZE_MISMATCH")
    except Exception:
        _append_jsonl(
            journal,
            {
                "stage": stage,
                "event": "GET_FAILED_TERMINALLY",
                "patient_id": record["patient_id"],
                "modality": modality,
            },
        )
        raise
    payload = {"bytes": byte_count, "sha256": digest.hexdigest()}
    _append_jsonl(
        journal,
        {
            "stage": stage,
            "event": "GET_COMPLETED",
            "patient_id": record["patient_id"],
            "modality": modality,
            **payload,
        },
    )
    return payload


def _axes_for_patient(
    axis_root: Path, record: Mapping[str, Any], preflight_record: Mapping[str, Any]
) -> tuple[dict[str, tuple[str, ...]], dict[str, list[int]]]:
    barcode_axes = {}
    marker_rows = {}
    for modality in ("GEX", "FB"):
        barcode_path, feature_path = _axis_paths(axis_root, record, modality)
        _, barcodes = _decode_gzip_lines(barcode_path)
        _, feature_lines = _decode_gzip_lines(feature_path)
        names, feature_type, descriptions = _feature_axis(feature_lines, modality)
        observed = {
            "barcode_url": _url(record, modality, "barcodes.tsv.gz"),
            "barcode_expected_bytes": barcode_path.stat().st_size,
            "barcode_observed_bytes": barcode_path.stat().st_size,
            "barcode_gzip_sha256": _sha256(barcode_path),
            "barcode_count": len(barcodes),
            "barcode_axis_sha256": _axis_sha256(barcodes),
            "barcode_set_sha256": _set_sha256(barcodes),
            "feature_url": _url(record, modality, "features.tsv.gz"),
            "feature_expected_bytes": feature_path.stat().st_size,
            "feature_observed_bytes": feature_path.stat().st_size,
            "feature_gzip_sha256": _sha256(feature_path),
            "feature_count": len(names),
            "feature_axis_sha256": _axis_sha256(names),
            "feature_type": feature_type,
            "marker_rows_1based": _marker_rows(names),
            "selected_marker_descriptions": descriptions,
        }
        frozen = preflight_record["axes"][modality]
        if any(observed[key] != frozen[key] for key in observed):
            raise ProtocolRefusal("AXIS_CHANGED_AFTER_PREFLIGHT")
        barcode_axes[modality] = tuple(barcodes)
        marker_rows[modality] = observed["marker_rows_1based"]
    if set(barcode_axes["GEX"]) != set(barcode_axes["FB"]):
        raise ProtocolRefusal("PAIRED_BARCODE_SETS_DIFFER")
    return barcode_axes, marker_rows


def _reduce_patient(
    gex_matrix: Path,
    fb_matrix: Path,
    axis_root: Path,
    record: Mapping[str, Any],
    preflight_record: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    barcode_axes, marker_rows = _axes_for_patient(axis_root, record, preflight_record)
    selected = select_barcodes(
        barcode_axes["GEX"],
        str(record["deposited_patient_id"]),
        count=CELL_BUDGET,
    )
    selected_columns = {
        modality: [barcode_axes[modality].index(barcode) + 1 for barcode in selected]
        for modality in ("GEX", "FB")
    }
    blocks = {}
    audits = {}
    for modality, matrix_path in (("GEX", gex_matrix), ("FB", fb_matrix)):
        frozen = preflight_record["axes"][modality]
        block, audit = reduce_gzip_matrix_market(
            matrix_path,
            expected_shape=(frozen["feature_count"], frozen["barcode_count"]),
            selected_rows=marker_rows[modality],
            selected_columns=selected_columns[modality],
        )
        blocks[modality] = block.T
        audits[modality] = {
            "parsed_nnz": audit.parsed_nnz,
            "declared_nnz": audit.declared_nnz,
            "selected_value_sum": audit.selected_value_sum,
            "decompressed_bytes": audit.decompressed_bytes,
        }
    tables, destroyed = patient_tables(
        blocks["GEX"],
        blocks["FB"],
        selected,
        str(record["deposited_patient_id"]),
    )
    return (
        tables,
        destroyed,
        {
            "patient_id": record["patient_id"],
            "role": record["role"],
            "group": record["group"],
            "selected_barcode_axis_sha256": _axis_sha256(selected),
            "tables_sha256": _array_sha256(tables),
            "destroyed_tables_sha256": _array_sha256(destroyed),
            "matrix_market": audits,
        },
    )


def _reduce_one_modality(
    matrix_path: Path,
    axis_root: Path,
    record: Mapping[str, Any],
    preflight_record: Mapping[str, Any],
    modality: str,
    selected: Sequence[str] | None = None,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
    barcode_axes, marker_rows = _axes_for_patient(axis_root, record, preflight_record)
    selected_axis = (
        select_barcodes(
            barcode_axes["GEX"],
            str(record["deposited_patient_id"]),
            count=CELL_BUDGET,
        )
        if selected is None
        else tuple(selected)
    )
    if (
        len(selected_axis) != CELL_BUDGET
        or len(set(selected_axis)) != CELL_BUDGET
        or any(barcode not in set(barcode_axes[modality]) for barcode in selected_axis)
    ):
        raise ProtocolRefusal("FROZEN_SELECTED_BARCODE_AXIS_DIFFERS")
    columns = [barcode_axes[modality].index(barcode) + 1 for barcode in selected_axis]
    frozen = preflight_record["axes"][modality]
    block, audit = reduce_gzip_matrix_market(
        matrix_path,
        expected_shape=(frozen["feature_count"], frozen["barcode_count"]),
        selected_rows=marker_rows[modality],
        selected_columns=columns,
    )
    return (
        block.T,
        selected_axis,
        {
            "parsed_nnz": audit.parsed_nnz,
            "declared_nnz": audit.declared_nnz,
            "selected_value_sum": audit.selected_value_sum,
            "decompressed_bytes": audit.decompressed_bytes,
        },
    )


def _acquire_role(
    role: str,
    axis_root: Path,
    scratch: Path,
    designation: Mapping[str, Any],
    preflight_value: Mapping[str, Any],
    journal: Path,
    stage: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[dict[str, Any]]]:
    records_by_id = {record["patient_id"]: record for record in designation["patients"]}
    preflight_by_id = {
        record["patient_id"]: record for record in preflight_value["patients"]
    }
    patient_ids = designation["role_order"][role]
    tables = []
    destroyed = []
    audits = []
    cohorts = []
    for patient_id in patient_ids:
        record = records_by_id[patient_id]
        patient_dir = scratch / stage / patient_id
        patient_dir.mkdir(parents=True, exist_ok=False)
        paths = {
            modality: patient_dir / _filename(record, modality, MATRIX_MEMBER)
            for modality in ("GEX", "FB")
        }
        downloads = {}
        try:
            for modality in ("GEX", "FB"):
                downloads[modality] = _download_matrix(
                    record, modality, paths[modality], journal, stage
                )
            current, shifted, audit = _reduce_patient(
                paths["GEX"],
                paths["FB"],
                axis_root,
                record,
                preflight_by_id[patient_id],
            )
            audit["downloads"] = downloads
            tables.append(current)
            destroyed.append(shifted)
            audits.append(audit)
            cohorts.append(record["group"])
        finally:
            for modality, path in paths.items():
                existed = path.exists()
                path.unlink(missing_ok=True)
                _append_jsonl(
                    journal,
                    {
                        "stage": stage,
                        "event": "MATRIX_DELETED",
                        "patient_id": patient_id,
                        "modality": modality,
                        "body_existed": existed,
                    },
                )
            shutil.rmtree(patient_dir, ignore_errors=True)
    return np.asarray(tables), np.asarray(destroyed), list(patient_ids), cohorts, audits


def _loss_payload(
    losses: Mapping[str, np.ndarray], patient_ids: Sequence[str]
) -> dict[str, Any]:
    return {
        method: {
            "mean": float(np.mean(values)),
            "by_patient": {
                patient: float(loss) for patient, loss in zip(patient_ids, values)
            },
        }
        for method, values in losses.items()
    }


def _source_gate(
    losses: Mapping[str, np.ndarray], cohorts: Sequence[str]
) -> dict[str, Any]:
    primary = np.asarray(losses["primary"], dtype=float)
    labels = np.asarray(tuple(cohorts), dtype=object)
    if (
        primary.shape != (11,)
        or labels.shape != primary.shape
        or int(np.count_nonzero(labels == "A")) != 4
        or int(np.count_nonzero(labels == "B")) != 7
        or not np.isfinite(primary).all()
    ):
        raise ValueError("source gate requires eleven finite pilot losses (4 A, 7 B)")
    comparisons = {}
    for method in SOURCE_GATE_COMPARATORS:
        comparator = np.asarray(losses[method], dtype=float)
        if comparator.shape != primary.shape or not np.isfinite(comparator).all():
            raise ValueError("source comparator losses differ from the pilot panel")
        difference = primary - comparator
        checks = {
            "primary_mean_strictly_lower": float(difference.mean()) < 0.0,
            "at_least_eight_of_eleven_favorable": int(
                np.count_nonzero(difference < 0.0)
            )
            >= 8,
            "A_mean_improvement_strictly_positive": float(
                difference[labels == "A"].mean()
            )
            < 0.0,
            "B_mean_improvement_strictly_positive": float(
                difference[labels == "B"].mean()
            )
            < 0.0,
        }
        comparisons[method] = {
            "primary_mean_loss": float(primary.mean()),
            "comparator_mean_loss": float(comparator.mean()),
            "mean_difference": float(difference.mean()),
            "favorable_patients": int(np.count_nonzero(difference < 0.0)),
            "checks": checks,
            "passes": all(checks.values()),
        }
    return {
        "comparisons": comparisons,
        "passes": all(record["passes"] for record in comparisons.values()),
    }


def _failure(stage: str, code: str) -> dict[str, Any]:
    results = {
        "calibration": CALIBRATION_SELECTION,
        "source": SOURCE_RESULT,
        "prediction": PREDICTION_RESULT,
        "score": SCORE_RESULT,
    }
    if stage not in results:
        raise ValueError("unknown stage")
    result = results[stage]
    payload = {
        "schema": f"gse313642-hcc-{stage}-result/2.0",
        "status": "TERMINAL_REFUSAL",
        "refusal_code": code,
        "created_at_utc": _timestamp(),
        "rerun_permitted": False,
    }
    if not result.exists():
        _write_json_x(result, payload)
    return payload


def _integer_table_panel(value: Any, count: int, label: str) -> np.ndarray:
    raw = np.asarray(value)
    numeric = np.asarray(value, dtype=float)
    expected = (count, len(MARKERS), len(MARKERS), 2, 2)
    if (
        raw.shape != expected
        or not np.isfinite(numeric).all()
        or np.any(numeric < 0.0)
        or not np.array_equal(numeric, np.rint(numeric))
        or np.any(numeric.sum(axis=(-2, -1)) != CELL_BUDGET)
    ):
        raise PermissionError(f"{label} differs from the frozen table contract")
    return numeric.astype(np.int64)


def _configuration_payload(config: PrimaryConfig) -> dict[str, float]:
    return {
        "donor_deviation_penalty": config.donor_deviation_penalty,
        "coefficient_ridge_penalty": config.coefficient_ridge_penalty,
        "graph_penalty": config.graph_penalty,
        "transport_multiplier": config.transport_multiplier,
    }


def _primary_loss_payload(
    losses: Mapping[PrimaryConfig, np.ndarray],
) -> dict[str, list[float]]:
    return {
        json.dumps(
            {
                "eta": config.donor_deviation_penalty,
                "alpha": config.transport_multiplier,
            },
            sort_keys=True,
        ): losses[config].tolist()
        for config in sorted(losses)
    }


def _comparator_loss_payload(
    losses: Mapping[str, Mapping[float, np.ndarray]],
) -> dict[str, dict[str, list[float]]]:
    return {
        method: {str(alpha): losses[method][alpha].tolist() for alpha in (0.75, 1.0)}
        for method in ("cohort_poisson", "cohort_signed_deviance")
    }


def _panel_hashes(patient_ids: Sequence[str], tables: np.ndarray) -> dict[str, str]:
    return {
        patient_id: _array_sha256(tables[index])
        for index, patient_id in enumerate(patient_ids)
    }


def _validate_reduction_hashes(
    audits: Any,
    patient_ids: Sequence[str],
    tables: np.ndarray,
    destroyed: np.ndarray,
) -> None:
    if not isinstance(audits, list) or [
        row.get("patient_id") for row in audits
    ] != list(patient_ids):
        raise PermissionError("reduction audit patient axis differs")
    for index, row in enumerate(audits):
        if row.get("tables_sha256") != _array_sha256(tables[index]) or row.get(
            "destroyed_tables_sha256"
        ) != _array_sha256(destroyed[index]):
            raise PermissionError("reduction table hash differs")


def _validate_calibration_selection(
    selection: Mapping[str, Any], designation: Mapping[str, Any]
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
    PrimaryConfig,
    dict[str, float],
    dict[str, Any],
]:
    patient_ids = list(designation["role_order"]["calibration"])
    by_id = {record["patient_id"]: record for record in designation["patients"]}
    cohorts = [by_id[patient_id]["group"] for patient_id in patient_ids]
    tables = _integer_table_panel(
        selection.get("calibration_tables"), 11, "calibration tables"
    )
    destroyed = _integer_table_panel(
        selection.get("calibration_destroyed_tables"),
        11,
        "calibration destroyed tables",
    )
    selected, primary_losses = select_primary_configuration(tables, cohorts)
    comparator_alphas, comparator_losses = select_comparator_alphas(tables, cohorts)
    models = serialize_models(
        fit_models(tables, destroyed, cohorts, selected, comparator_alphas)
    )
    if (
        selection.get("schema") != "gse313642-hcc-calibration-selection/2.0"
        or selection.get("status") != "FROZEN_BEFORE_ANY_PILOT_MATRIX_REQUEST"
        or selection.get("rerun_permitted") is not False
        or selection.get("calibration_patient_order") != patient_ids
        or selection.get("calibration_cohorts") != dict(zip(patient_ids, cohorts))
        or selection.get("selected_configuration") != _configuration_payload(selected)
        or selection.get("primary_calibration_lopo_losses")
        != _primary_loss_payload(primary_losses)
        or selection.get("matched_comparator_alphas") != comparator_alphas
        or selection.get("matched_comparator_calibration_lopo_losses")
        != _comparator_loss_payload(comparator_losses)
        or selection.get("calibration_models") != models
        or selection.get("calibration_table_hashes")
        != _panel_hashes(patient_ids, tables)
        or selection.get("calibration_destroyed_table_hashes")
        != _panel_hashes(patient_ids, destroyed)
        or selection.get("pilot_matrix_requests") != 0
    ):
        raise PermissionError("calibration selection does not replay exactly")
    _validate_reduction_hashes(
        selection.get("reduction_audit"), patient_ids, tables, destroyed
    )
    return (
        tables,
        destroyed,
        patient_ids,
        cohorts,
        selected,
        comparator_alphas,
        models,
    )


def run_calibration(
    token_path: Path, axis_root: Path, *, scratch: Path = DEFAULT_SCRATCH
) -> dict[str, Any]:
    with _stage_lock(scratch, "calibration"):
        try:
            _require_runtime()
            _require_public_attempt("calibration")
            _consume("calibration", token_path, scratch)
            designation, preflight_value = _validate_preflight(axis_root)
            scratch.mkdir(parents=True, exist_ok=True)
            (
                tables,
                destroyed,
                patient_ids,
                cohorts,
                audits,
            ) = _acquire_role(
                "calibration",
                axis_root,
                scratch,
                designation,
                preflight_value,
                CALIBRATION_ACCESS,
                "calibration",
            )
            selected, cv_losses = select_primary_configuration(tables, cohorts)
            comparator_alphas, comparator_losses = select_comparator_alphas(
                tables, cohorts
            )
            calibration_models = fit_models(
                tables,
                destroyed,
                cohorts,
                selected,
                comparator_alphas,
            )
            payload = {
                "schema": "gse313642-hcc-calibration-selection/2.0",
                "status": "FROZEN_BEFORE_ANY_PILOT_MATRIX_REQUEST",
                "created_at_utc": _timestamp(),
                "rerun_permitted": False,
                "calibration_patient_order": patient_ids,
                "calibration_cohorts": dict(zip(patient_ids, cohorts)),
                "selected_configuration": _configuration_payload(selected),
                "primary_calibration_lopo_losses": _primary_loss_payload(cv_losses),
                "matched_comparator_alphas": comparator_alphas,
                "matched_comparator_calibration_lopo_losses": _comparator_loss_payload(
                    comparator_losses
                ),
                "calibration_models": serialize_models(calibration_models),
                "calibration_tables": tables.tolist(),
                "calibration_destroyed_tables": destroyed.tolist(),
                "calibration_table_hashes": {
                    record["patient_id"]: record["tables_sha256"] for record in audits
                },
                "calibration_destroyed_table_hashes": {
                    record["patient_id"]: record["destroyed_tables_sha256"]
                    for record in audits
                },
                "reduction_audit": audits,
                "pilot_matrix_requests": 0,
            }
            _write_json_x(CALIBRATION_SELECTION, payload)
            return payload
        except Exception as error:
            code = (
                error.code
                if isinstance(error, ProtocolRefusal)
                else "CALIBRATION_EXECUTION_FAILURE"
            )
            return _failure("calibration", code)


def authorize_source() -> dict[str, Any]:
    _require_runtime()
    selection = _read_json(CALIBRATION_SELECTION)
    tags = _verify_implementation()
    preflight_commit = _require_public_tag(
        PREFLIGHT_TAG, (V1_AXIS_ACCESS, V1_AXIS_REFUSAL, AXIS_PREFLIGHT)
    )
    calibration_commit = validate("calibration")["final_commit"]
    _require_ancestor(tags["implementation_commit"], preflight_commit)
    _validate_calibration_selection(selection, _designation())
    if any(
        path.exists() for path in (PILOT_AUTHORIZATION, SOURCE_ATTEMPT, SOURCE_RESULT)
    ):
        raise FileExistsError("pilot stage is already authorized or consumed")
    payload = {
        "schema": "gse313642-hcc-pilot-authorization/2.0",
        "status": "AUTHORIZED_AFTER_PUBLIC_CALIBRATION_FREEZE",
        "created_at_utc": _timestamp(),
        "calibration_selection_sha256": _sha256(CALIBRATION_SELECTION),
        "calibration_commit": calibration_commit,
        "pilot_matrix_gets_authorized": 22,
    }
    _write_json_x(PILOT_AUTHORIZATION, payload)
    return payload


def run_source(
    token_path: Path, axis_root: Path, *, scratch: Path = DEFAULT_SCRATCH
) -> dict[str, Any]:
    with _stage_lock(scratch, "source"):
        try:
            _require_runtime()
            _require_public_attempt("source")
            _consume("source", token_path, scratch)
            designation, preflight_value = _validate_preflight(axis_root)
            selection = _read_json(CALIBRATION_SELECTION)
            authorization = _read_json(PILOT_AUTHORIZATION)
            if (
                selection.get("status") != "FROZEN_BEFORE_ANY_PILOT_MATRIX_REQUEST"
                or authorization.get("schema")
                != "gse313642-hcc-pilot-authorization/2.0"
                or authorization.get("status")
                != "AUTHORIZED_AFTER_PUBLIC_CALIBRATION_FREEZE"
                or authorization.get("calibration_selection_sha256")
                != _sha256(CALIBRATION_SELECTION)
                or authorization.get("pilot_matrix_gets_authorized") != 22
            ):
                raise PermissionError("pilot prerequisites differ")
            (
                calibration_tables,
                calibration_destroyed,
                calibration_ids,
                calibration_cohorts,
                selected,
                comparator_alphas,
                calibration_models_payload,
            ) = _validate_calibration_selection(selection, designation)
            scratch.mkdir(parents=True, exist_ok=True)
            (
                pilot_tables,
                pilot_destroyed,
                pilot_ids,
                pilot_cohorts,
                pilot_audit,
            ) = _acquire_role(
                "pilot",
                axis_root,
                scratch,
                designation,
                preflight_value,
                SOURCE_ACCESS,
                "source",
            )
            calibration_models = deserialize_models(calibration_models_payload)
            pilot_losses = serialized_panel_losses(
                calibration_models, pilot_tables, pilot_cohorts
            )
            gate = _source_gate(pilot_losses, pilot_cohorts)
            payload: dict[str, Any] = {
                "schema": "gse313642-hcc-source-result/2.0",
                "status": "SOURCE_PASS_REFIT_22"
                if gate["passes"]
                else "TERMINAL_SOURCE_GATE_FAIL",
                "created_at_utc": _timestamp(),
                "rerun_permitted": False,
                "calibration_selection_sha256": _sha256(CALIBRATION_SELECTION),
                "selected_configuration": selection["selected_configuration"],
                "matched_comparator_alphas": comparator_alphas,
                "pilot_gate": gate,
                "pilot_losses": _loss_payload(pilot_losses, pilot_ids),
                "patient_order": {
                    "calibration": calibration_ids,
                    "pilot": pilot_ids,
                },
                "cohorts": {
                    **dict(zip(calibration_ids, calibration_cohorts)),
                    **dict(zip(pilot_ids, pilot_cohorts)),
                },
                "reduction_audit": pilot_audit,
                "pilot_tables": pilot_tables.tolist(),
                "pilot_destroyed_tables": pilot_destroyed.tolist(),
                "pilot_table_hashes": _panel_hashes(pilot_ids, pilot_tables),
                "pilot_destroyed_table_hashes": _panel_hashes(
                    pilot_ids, pilot_destroyed
                ),
            }
            if gate["passes"]:
                refit = fit_models(
                    np.concatenate((calibration_tables, pilot_tables)),
                    np.concatenate((calibration_destroyed, pilot_destroyed)),
                    calibration_cohorts + pilot_cohorts,
                    selected,
                    comparator_alphas,
                )
                payload["source_patient_count"] = 22
                payload["source_models"] = serialize_models(refit)
            _write_json_x(SOURCE_RESULT, payload)
            return payload
        except Exception as error:
            code = (
                error.code
                if isinstance(error, ProtocolRefusal)
                else "SOURCE_EXECUTION_FAILURE"
            )
            return _failure("source", code)


def _validate_source_result(
    source: Mapping[str, Any], designation: Mapping[str, Any]
) -> dict[str, Any]:
    selection = _read_json(CALIBRATION_SELECTION)
    (
        calibration_tables,
        calibration_destroyed,
        calibration_ids,
        calibration_cohorts,
        selected,
        comparator_alphas,
        calibration_models_payload,
    ) = _validate_calibration_selection(selection, designation)
    pilot_ids = list(designation["role_order"]["pilot"])
    by_id = {record["patient_id"]: record for record in designation["patients"]}
    pilot_cohorts = [by_id[patient_id]["group"] for patient_id in pilot_ids]
    pilot_tables = _integer_table_panel(source.get("pilot_tables"), 11, "pilot tables")
    pilot_destroyed = _integer_table_panel(
        source.get("pilot_destroyed_tables"), 11, "pilot destroyed tables"
    )
    calibration_models = deserialize_models(calibration_models_payload)
    pilot_losses = serialized_panel_losses(
        calibration_models, pilot_tables, pilot_cohorts
    )
    gate = _source_gate(pilot_losses, pilot_cohorts)
    expected_models = serialize_models(
        fit_models(
            np.concatenate((calibration_tables, pilot_tables)),
            np.concatenate((calibration_destroyed, pilot_destroyed)),
            calibration_cohorts + pilot_cohorts,
            selected,
            comparator_alphas,
        )
    )
    all_ids = calibration_ids + pilot_ids
    all_cohorts = calibration_cohorts + pilot_cohorts
    if (
        source.get("schema") != "gse313642-hcc-source-result/2.0"
        or source.get("status") != "SOURCE_PASS_REFIT_22"
        or source.get("rerun_permitted") is not False
        or source.get("calibration_selection_sha256") != _sha256(CALIBRATION_SELECTION)
        or source.get("selected_configuration") != _configuration_payload(selected)
        or source.get("matched_comparator_alphas") != comparator_alphas
        or source.get("pilot_gate") != gate
        or not gate["passes"]
        or source.get("pilot_losses") != _loss_payload(pilot_losses, pilot_ids)
        or source.get("patient_order")
        != {"calibration": calibration_ids, "pilot": pilot_ids}
        or source.get("cohorts") != dict(zip(all_ids, all_cohorts))
        or source.get("pilot_table_hashes") != _panel_hashes(pilot_ids, pilot_tables)
        or source.get("pilot_destroyed_table_hashes")
        != _panel_hashes(pilot_ids, pilot_destroyed)
        or source.get("source_patient_count") != 22
        or source.get("source_models") != expected_models
    ):
        raise PermissionError("source promotion and refit do not replay exactly")
    _validate_reduction_hashes(
        source.get("reduction_audit"), pilot_ids, pilot_tables, pilot_destroyed
    )
    return expected_models


def authorize_prediction() -> dict[str, Any]:
    _require_runtime()
    source = _read_json(SOURCE_RESULT)
    source_commit = validate("source")["final_commit"]
    _validate_source_result(source, _designation())
    if any(
        path.exists()
        for path in (PREDICTION_AUTHORIZATION, PREDICTION_ATTEMPT, PREDICTION_RESULT)
    ):
        raise FileExistsError("prediction stage is already authorized or consumed")
    payload = {
        "schema": "gse313642-hcc-prediction-authorization/2.0",
        "status": "AUTHORIZED_AFTER_PUBLIC_SOURCE_PASS",
        "created_at_utc": _timestamp(),
        "source_result_sha256": _sha256(SOURCE_RESULT),
        "source_commit": source_commit,
        "held_gex_gets_authorized": 12,
        "held_fb_gets_authorized": 0,
    }
    _write_json_x(PREDICTION_AUTHORIZATION, payload)
    return payload


def run_prediction(
    token_path: Path,
    axis_root: Path,
    state_path: Path,
    *,
    scratch: Path = DEFAULT_SCRATCH,
) -> dict[str, Any]:
    with _stage_lock(scratch, "prediction"):
        private_state: Path | None = None
        consumed = False
        try:
            _require_runtime()
            _require_public_attempt("prediction")
            private_state = _private_state_path(state_path, scratch)
            if private_state.exists():
                raise FileExistsError("private held GEX state already exists")
            _consume("prediction", token_path, scratch, private_state)
            consumed = True
            designation, preflight_value = _validate_preflight(axis_root)
            source = _read_json(SOURCE_RESULT)
            authorization = _read_json(PREDICTION_AUTHORIZATION)
            if source.get("status") != "SOURCE_PASS_REFIT_22" or authorization.get(
                "source_result_sha256"
            ) != _sha256(SOURCE_RESULT):
                raise PermissionError("prediction prerequisites differ")
            source_models = _validate_source_result(source, designation)
            scratch.mkdir(parents=True, exist_ok=True)
            records_by_id = {
                record["patient_id"]: record for record in designation["patients"]
            }
            preflight_by_id = {
                record["patient_id"]: record for record in preflight_value["patients"]
            }
            models = deserialize_models(source_models)
            public_patients = []
            private_patients = []
            for patient_id in designation["role_order"]["held"]:
                record = records_by_id[patient_id]
                patient_dir = scratch / "prediction" / patient_id
                patient_dir.mkdir(parents=True, exist_ok=False)
                matrix_path = patient_dir / _filename(record, "GEX", MATRIX_MEMBER)
                try:
                    download = _download_matrix(
                        record,
                        "GEX",
                        matrix_path,
                        PREDICTION_ACCESS,
                        "prediction",
                    )
                    counts, selected, matrix_audit = _reduce_one_modality(
                        matrix_path,
                        axis_root,
                        record,
                        preflight_by_id[patient_id],
                        "GEX",
                    )
                    rna = rna_detection_states(counts)
                    positive = rna.sum(axis=0).astype(np.int64)
                    marker_rows = np.stack((CELL_BUDGET - positive, positive), axis=1)
                    rows = np.repeat(marker_rows[:, None, :], len(MARKERS), axis=1)
                    columns = np.broadcast_to(
                        np.asarray((ADT_HIGH_COUNT, ADT_HIGH_COUNT), dtype=np.int64),
                        rows.shape,
                    ).copy()
                    predictions = predict_serialized_at_margins(
                        models, rows, columns, record["group"]
                    )
                    public_patients.append(
                        {
                            "patient_id": patient_id,
                            "group": record["group"],
                            "row_margins": rows.tolist(),
                            "column_margins": columns.tolist(),
                            "predictions": {
                                method: values.tolist()
                                for method, values in predictions.items()
                            },
                            "selected_barcode_axis_sha256": _axis_sha256(selected),
                            "gex_download": download,
                            "gex_matrix_market": matrix_audit,
                        }
                    )
                    private_patients.append(
                        {
                            "patient_id": patient_id,
                            "deposited_patient_id": record["deposited_patient_id"],
                            "group": record["group"],
                            "selected_barcodes": list(selected),
                            "rna_states": rna.tolist(),
                        }
                    )
                finally:
                    existed = matrix_path.exists()
                    matrix_path.unlink(missing_ok=True)
                    _append_jsonl(
                        PREDICTION_ACCESS,
                        {
                            "stage": "prediction",
                            "event": "MATRIX_DELETED",
                            "patient_id": patient_id,
                            "modality": "GEX",
                            "body_existed": existed,
                        },
                    )
                    shutil.rmtree(patient_dir, ignore_errors=True)
            private_payload = {
                "schema": "gse313642-hcc-private-held-gex-state/2.0",
                "prediction_attempt_sha256": _sha256(PREDICTION_ATTEMPT),
                "patients": private_patients,
            }
            _write_json_x(private_state, private_payload)
            payload = {
                "schema": "gse313642-hcc-predictions/2.0",
                "status": "PREDICTIONS_FROZEN_BEFORE_ANY_HELD_FB_ACCESS",
                "created_at_utc": _timestamp(),
                "rerun_permitted": False,
                "source_result_sha256": _sha256(SOURCE_RESULT),
                "source_models_sha256": _json_sha256(source_models),
                "prediction_attempt_sha256": _sha256(PREDICTION_ATTEMPT),
                "selected_configuration": source["selected_configuration"],
                "matched_comparator_alphas": source["matched_comparator_alphas"],
                "marker_order": list(MARKERS),
                "coordinate_order": {
                    "rows": ["RNA-negative", "RNA-positive"],
                    "columns": ["FB-low", "FB-high"],
                },
                "private_state_sha256": _sha256(private_state),
                "private_state_bytes": private_state.stat().st_size,
                "held_gex_gets": 12,
                "held_fb_gets": 0,
                "held_fb_numeric_values_read": 0,
                "held_rna_fb_pairings_read": 0,
                "fb_margin_rule": "exactly 256 low and 256 high cells per marker",
                "patients": public_patients,
            }
            _write_json_x(PREDICTION_RESULT, payload)
            return payload
        except Exception as error:
            if consumed and private_state is not None:
                private_state.unlink(missing_ok=True)
            code = (
                error.code
                if isinstance(error, ProtocolRefusal)
                else "PREDICTION_EXECUTION_FAILURE"
            )
            return _failure("prediction", code)


def authorize_score() -> dict[str, Any]:
    _require_runtime()
    predictions = _read_json(PREDICTION_RESULT)
    prediction_commit = validate("prediction")["final_commit"]
    source_models = _validate_source_result(_read_json(SOURCE_RESULT), _designation())
    _validate_public_predictions(predictions, _designation(), source_models)
    if any(
        path.exists() for path in (SCORE_AUTHORIZATION, SCORE_ATTEMPT, SCORE_RESULT)
    ):
        raise FileExistsError("score stage is already authorized or consumed")
    payload = {
        "schema": "gse313642-hcc-score-authorization/2.0",
        "status": "AUTHORIZED_AFTER_PUBLIC_PREDICTION_FREEZE",
        "created_at_utc": _timestamp(),
        "prediction_result_sha256": _sha256(PREDICTION_RESULT),
        "prediction_commit": prediction_commit,
        "private_state_sha256": predictions["private_state_sha256"],
        "held_fb_gets_authorized": 12,
        "held_gex_gets_authorized": 0,
    }
    _write_json_x(SCORE_AUTHORIZATION, payload)
    return payload


def _validate_public_predictions(
    predictions: Mapping[str, Any],
    designation: Mapping[str, Any],
    source_models: Mapping[str, Any],
) -> None:
    patient_ids = designation["role_order"]["held"]
    public_rows = predictions.get("patients")
    if (
        predictions.get("schema") != "gse313642-hcc-predictions/2.0"
        or predictions.get("status") != "PREDICTIONS_FROZEN_BEFORE_ANY_HELD_FB_ACCESS"
        or predictions.get("rerun_permitted") is not False
        or predictions.get("source_result_sha256") != _sha256(SOURCE_RESULT)
        or predictions.get("source_models_sha256") != _json_sha256(source_models)
        or not isinstance(predictions.get("prediction_attempt_sha256"), str)
        or len(predictions["prediction_attempt_sha256"]) != 64
        or predictions.get("selected_configuration") != source_models["configuration"]
        or predictions.get("matched_comparator_alphas")
        != source_models["comparator_alphas"]
        or predictions.get("marker_order") != list(MARKERS)
        or predictions.get("coordinate_order")
        != {
            "rows": ["RNA-negative", "RNA-positive"],
            "columns": ["FB-low", "FB-high"],
        }
        or predictions.get("held_gex_gets") != 12
        or predictions.get("held_fb_gets") != 0
        or predictions.get("held_fb_numeric_values_read") != 0
        or predictions.get("held_rna_fb_pairings_read") != 0
        or not isinstance(public_rows, list)
        or [record.get("patient_id") for record in public_rows] != patient_ids
    ):
        raise ProtocolRefusal("FROZEN_PREDICTION_PATIENT_AXIS_DIFFERS")
    designated = {record["patient_id"]: record for record in designation["patients"]}
    models = deserialize_models(source_models)
    for public in public_rows:
        patient_id = public["patient_id"]
        cohort = designated[patient_id]["group"]
        if public.get("group") != cohort:
            raise ProtocolRefusal("FROZEN_PREDICTION_METADATA_DIFFERS")
        rows = np.asarray(public.get("row_margins"))
        columns = np.asarray(public.get("column_margins"))
        if (
            rows.shape != (len(MARKERS), len(MARKERS), 2)
            or not np.isfinite(rows).all()
            or np.any(rows < 0)
            or not np.array_equal(rows, np.rint(rows))
            or np.any(rows.sum(axis=-1) != CELL_BUDGET)
            or columns.shape != rows.shape
            or not np.all(columns == np.asarray((ADT_HIGH_COUNT, ADT_HIGH_COUNT)))
        ):
            raise ProtocolRefusal("FROZEN_HELD_MARGINS_DIFFER")
        methods = public.get("predictions")
        expected = predict_serialized_at_margins(models, rows, columns, cohort)
        if not isinstance(methods, dict) or set(methods) != set(expected):
            raise ProtocolRefusal("FROZEN_PREDICTION_METHOD_SET_DIFFERS")
        for method, values in methods.items():
            estimate = np.asarray(values, dtype=float)
            structural_zero = (rows[..., :, None] == 0) | (columns[..., None, :] == 0)
            if (
                estimate.shape != (len(MARKERS), len(MARKERS), 2, 2)
                or not np.isfinite(estimate).all()
                or np.any(estimate < 0.0)
                or np.any(estimate[structural_zero] != 0.0)
                or np.any(estimate[~structural_zero] <= 0.0)
                or not np.allclose(estimate.sum(axis=-1), rows, atol=1e-8)
                or not np.allclose(estimate.sum(axis=-2), columns, atol=1e-8)
                or not np.array_equal(estimate, expected[method])
            ):
                raise ProtocolRefusal("FROZEN_PREDICTION_TABLE_DIFFERS")


def _validate_prediction_state(
    predictions: Mapping[str, Any],
    state: Mapping[str, Any],
    designation: Mapping[str, Any],
    source_models: Mapping[str, Any],
) -> None:
    _validate_public_predictions(predictions, designation, source_models)
    patient_ids = designation["role_order"]["held"]
    public_rows = predictions["patients"]
    private_rows = state.get("patients")
    if (
        state.get("schema") != "gse313642-hcc-private-held-gex-state/2.0"
        or state.get("prediction_attempt_sha256")
        != predictions.get("prediction_attempt_sha256")
        or not isinstance(private_rows, list)
        or [record.get("patient_id") for record in private_rows] != patient_ids
    ):
        raise ProtocolRefusal("PRIVATE_HELD_GEX_STATE_DIFFERS")
    designated = {record["patient_id"]: record for record in designation["patients"]}
    for public, private in zip(public_rows, private_rows):
        patient_id = public["patient_id"]
        if (
            private.get("patient_id") != patient_id
            or private.get("group") != designated[patient_id]["group"]
            or private.get("deposited_patient_id")
            != designated[patient_id]["deposited_patient_id"]
        ):
            raise ProtocolRefusal("FROZEN_PREDICTION_METADATA_DIFFERS")
        barcodes = private.get("selected_barcodes")
        rna = np.asarray(private.get("rna_states"))
        if (
            not isinstance(barcodes, list)
            or len(barcodes) != CELL_BUDGET
            or len(set(barcodes)) != CELL_BUDGET
            or any(not isinstance(barcode, str) or not barcode for barcode in barcodes)
            or rna.shape != (CELL_BUDGET, len(MARKERS))
            or not np.isin(rna, (0, 1)).all()
            or _axis_sha256(barcodes) != public.get("selected_barcode_axis_sha256")
        ):
            raise ProtocolRefusal("PRIVATE_HELD_GEX_STATE_DIFFERS")
        positive = rna.sum(axis=0).astype(np.int64)
        marker_rows = np.stack((CELL_BUDGET - positive, positive), axis=1)
        expected_rows = np.repeat(marker_rows[:, None, :], len(MARKERS), axis=1)
        if not np.array_equal(np.asarray(public["row_margins"]), expected_rows):
            raise ProtocolRefusal("FROZEN_HELD_MARGINS_DIFFER")


def _validate_score_result(
    value: Mapping[str, Any], designation: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    if (
        value.get("schema") != "gse313642-hcc-score-result/2.0"
        or value.get("rerun_permitted") is not False
    ):
        raise PermissionError("score result schema or durability differs")
    if value.get("status") == "TERMINAL_REFUSAL":
        refusal = value.get("refusal_code")
        if not isinstance(refusal, str) or not refusal:
            raise PermissionError("terminal score refusal code is absent")
        return {}

    patient_ids = list(designation["role_order"]["held"])
    records = {record["patient_id"]: record for record in designation["patients"]}
    cohorts = [records[patient_id]["group"] for patient_id in patient_ids]
    expected_cohorts = dict(zip(patient_ids, cohorts))
    predictions = _read_json(PREDICTION_RESULT)
    prediction_patients = predictions.get("patients")
    if not isinstance(prediction_patients, list) or len(prediction_patients) != len(
        patient_ids
    ):
        raise PermissionError("frozen prediction patient panel differs")
    if [record.get("patient_id") for record in prediction_patients] != patient_ids:
        raise PermissionError("frozen prediction patient order differs")
    method_sets = [
        set(record.get("predictions", {}))
        if isinstance(record.get("predictions"), dict)
        else set()
        for record in prediction_patients
    ]
    expected_methods = method_sets[0] if method_sets else set()
    if not expected_methods or any(
        methods != expected_methods for methods in method_sets
    ):
        raise PermissionError("frozen prediction method set differs")
    if (
        value.get("status") not in {"CONFIRMATION_PASS", "COMPLETED_CONFIRMATION_FAIL"}
        or value.get("prediction_result_sha256") != _sha256(PREDICTION_RESULT)
        or value.get("held_patient_order") != patient_ids
        or value.get("held_cohorts") != expected_cohorts
        or value.get("held_gex_gets_during_score") != 0
        or value.get("predictions_reconstructed_after_fb_access") is not False
    ):
        raise PermissionError("score result frozen design binding differs")

    loss_payload = value.get("held_losses")
    if not isinstance(loss_payload, dict) or set(loss_payload) != expected_methods:
        raise PermissionError("score result method losses differ")
    losses: dict[str, np.ndarray] = {}
    for method in sorted(expected_methods):
        record = loss_payload.get(method)
        if not isinstance(record, dict) or not isinstance(
            record.get("by_patient"), dict
        ):
            raise PermissionError("score result patient losses are absent")
        by_patient = record["by_patient"]
        if list(by_patient) != patient_ids:
            raise PermissionError("score result patient loss order differs")
        current = np.asarray(
            [by_patient[patient] for patient in patient_ids], dtype=float
        )
        if (
            current.shape != (len(patient_ids),)
            or not np.isfinite(current).all()
            or np.any(current < 0.0)
            or record.get("mean") != float(current.mean())
        ):
            raise PermissionError("score result patient loss values differ")
        losses[method] = current

    expected_gate = held_gate(losses, cohorts)
    expected_status = (
        "CONFIRMATION_PASS"
        if expected_gate["passes"]
        else "COMPLETED_CONFIRMATION_FAIL"
    )
    if (
        value.get("status") != expected_status
        or not isinstance(value.get("held_gate"), dict)
        or _json_sha256(value["held_gate"]) != _json_sha256(expected_gate)
    ):
        raise PermissionError("score result gate does not recompute")

    audits = value.get("reduction_audit")
    if not isinstance(audits, list) or [
        (record.get("patient_id"), record.get("group")) for record in audits
    ] != list(zip(patient_ids, cohorts)):
        raise PermissionError("score reduction audit order differs")
    for record in audits:
        digest = record.get("truth_tables_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise PermissionError("score truth-table hash differs")
    return losses


def run_score(
    token_path: Path,
    axis_root: Path,
    state_path: Path,
    *,
    scratch: Path = DEFAULT_SCRATCH,
) -> dict[str, Any]:
    with _stage_lock(scratch, "score"):
        private_state: Path | None = None
        consumed = False
        try:
            _require_runtime()
            _require_public_attempt("score")
            private_state = _private_state_path(state_path, scratch)
            _consume("score", token_path, scratch, private_state)
            consumed = True
            designation, preflight_value = _validate_preflight(axis_root)
            predictions = _read_json(PREDICTION_RESULT)
            source_models = _validate_source_result(
                _read_json(SOURCE_RESULT), designation
            )
            authorization = _read_json(SCORE_AUTHORIZATION)
            if (
                authorization.get("prediction_result_sha256")
                != _sha256(PREDICTION_RESULT)
                or authorization.get("private_state_sha256") != _sha256(private_state)
                or predictions.get("private_state_sha256") != _sha256(private_state)
            ):
                raise PermissionError("score prerequisites differ")
            state = _read_json(private_state)
            _validate_prediction_state(predictions, state, designation, source_models)
            state_by_id = {record["patient_id"]: record for record in state["patients"]}
            prediction_by_id = {
                record["patient_id"]: record for record in predictions["patients"]
            }
            records_by_id = {
                record["patient_id"]: record for record in designation["patients"]
            }
            preflight_by_id = {
                record["patient_id"]: record for record in preflight_value["patients"]
            }
            patient_ids = designation["role_order"]["held"]
            losses: dict[str, np.ndarray] = {}
            audits = []
            cohorts = []
            scratch.mkdir(parents=True, exist_ok=True)
            for patient_index, patient_id in enumerate(patient_ids):
                record = records_by_id[patient_id]
                private = state_by_id[patient_id]
                frozen = prediction_by_id[patient_id]
                patient_dir = scratch / "score" / patient_id
                patient_dir.mkdir(parents=True, exist_ok=False)
                matrix_path = patient_dir / _filename(record, "FB", MATRIX_MEMBER)
                try:
                    download = _download_matrix(
                        record, "FB", matrix_path, SCORE_ACCESS, "score"
                    )
                    counts, selected, matrix_audit = _reduce_one_modality(
                        matrix_path,
                        axis_root,
                        record,
                        preflight_by_id[patient_id],
                        "FB",
                        private["selected_barcodes"],
                    )
                    if _axis_sha256(selected) != frozen["selected_barcode_axis_sha256"]:
                        raise ProtocolRefusal("SCORE_BARCODE_AXIS_DIFFERS")
                    rna = np.asarray(private["rna_states"], dtype=np.uint8)
                    fb = adt_midrank_states(
                        counts, selected, record["deposited_patient_id"]
                    )
                    truth = joint_binary_tables(rna, fb)
                    current_predictions = {
                        method: np.asarray(values, dtype=float)
                        for method, values in frozen["predictions"].items()
                    }
                    if patient_index == 0:
                        losses = {
                            method: np.empty(len(patient_ids), dtype=float)
                            for method in current_predictions
                        }
                    elif set(current_predictions) != set(losses):
                        raise ProtocolRefusal("FROZEN_PREDICTION_METHOD_SET_DIFFERS")
                    for method, estimate in current_predictions.items():
                        losses[method][patient_index] = float(
                            np.mean(entity_deviance(truth, estimate))
                        )
                    audits.append(
                        {
                            "patient_id": patient_id,
                            "group": record["group"],
                            "truth_tables_sha256": _array_sha256(truth),
                            "fb_download": download,
                            "fb_matrix_market": matrix_audit,
                        }
                    )
                    cohorts.append(record["group"])
                finally:
                    existed = matrix_path.exists()
                    matrix_path.unlink(missing_ok=True)
                    _append_jsonl(
                        SCORE_ACCESS,
                        {
                            "stage": "score",
                            "event": "MATRIX_DELETED",
                            "patient_id": patient_id,
                            "modality": "FB",
                            "body_existed": existed,
                        },
                    )
                    shutil.rmtree(patient_dir, ignore_errors=True)
            gate = held_gate(losses, cohorts)
            payload = {
                "schema": "gse313642-hcc-score-result/2.0",
                "status": "CONFIRMATION_PASS"
                if gate["passes"]
                else "COMPLETED_CONFIRMATION_FAIL",
                "created_at_utc": _timestamp(),
                "rerun_permitted": False,
                "prediction_result_sha256": _sha256(PREDICTION_RESULT),
                "held_patient_order": patient_ids,
                "held_cohorts": dict(zip(patient_ids, cohorts)),
                "held_gate": gate,
                "held_losses": _loss_payload(losses, patient_ids),
                "reduction_audit": audits,
                "held_gex_gets_during_score": 0,
                "predictions_reconstructed_after_fb_access": False,
            }
            _write_json_x(SCORE_RESULT, payload)
            private_state.unlink()
            return payload
        except Exception as error:
            if consumed and private_state is not None:
                private_state.unlink(missing_ok=True)
            code = (
                error.code
                if isinstance(error, ProtocolRefusal)
                else "SCORE_EXECUTION_FAILURE"
            )
            return _failure("score", code)


def _stage_paths(stage: str) -> tuple[Path, Path, Path]:
    paths = {
        "calibration": (
            CALIBRATION_CONSUMPTION,
            CALIBRATION_SELECTION,
            CALIBRATION_ACCESS,
        ),
        "source": (SOURCE_CONSUMPTION, SOURCE_RESULT, SOURCE_ACCESS),
        "prediction": (
            PREDICTION_CONSUMPTION,
            PREDICTION_RESULT,
            PREDICTION_ACCESS,
        ),
        "score": (SCORE_CONSUMPTION, SCORE_RESULT, SCORE_ACCESS),
    }
    if stage not in paths:
        raise ValueError("unknown stage")
    return paths[stage]


def recover(
    stage: str,
    *,
    scratch: Path = DEFAULT_SCRATCH,
    state_path: Path | None = None,
) -> dict[str, Any]:
    consumption, result, access = _stage_paths(stage)
    if result.exists() and stage != "score":
        return _read_json(result)
    if not consumption.exists():
        raise PermissionError("stage was not consumed; recovery is unavailable")
    consumed = _read_json(consumption)
    identity = hashlib.sha256(
        str(scratch.expanduser().resolve()).encode("utf-8")
    ).hexdigest()
    if consumed.get("scratch_identity_sha256") != identity:
        raise PermissionError("recovery scratch differs from the consumed stage")
    if stage in {"prediction", "score"}:
        if state_path is None:
            raise PermissionError("private state path is required for recovery")
        private_state = _private_state_path(state_path, scratch)
        state_identity = hashlib.sha256(str(private_state).encode("utf-8")).hexdigest()
        if consumed.get("private_state_identity_sha256") != state_identity:
            raise PermissionError("recovery private state path differs")
        private_state.unlink(missing_ok=True)
    elif state_path is not None:
        raise PermissionError("this stage has no private state")
    if result.exists():
        shutil.rmtree(scratch / stage, ignore_errors=True)
        return _read_json(result)
    designation = _designation()
    records = {record["patient_id"]: record for record in designation["patients"]}
    events = _read_jsonl(access)
    completed = {
        (event.get("patient_id"), event.get("modality"))
        for event in events
        if event.get("event") == "GET_COMPLETED"
    }
    deleted = {
        (event.get("patient_id"), event.get("modality"))
        for event in events
        if event.get("event") == "MATRIX_DELETED"
    }
    role, modalities = {
        "calibration": ("calibration", ("GEX", "FB")),
        "source": ("pilot", ("GEX", "FB")),
        "prediction": ("held", ("GEX",)),
        "score": ("held", ("FB",)),
    }[stage]
    expected = [
        (patient_id, modality)
        for patient_id in designation["role_order"][role]
        for modality in modalities
    ]
    for patient_id, modality in expected:
        if (patient_id, modality) not in completed - deleted:
            continue
        matrix_path = (
            scratch
            / stage
            / str(patient_id)
            / _filename(records[str(patient_id)], str(modality), MATRIX_MEMBER)
        )
        existed = matrix_path.exists()
        matrix_path.unlink(missing_ok=True)
        _append_jsonl(
            access,
            {
                "stage": stage,
                "event": "MATRIX_DELETED",
                "patient_id": patient_id,
                "modality": modality,
                "body_existed": existed,
                "recovered_after_crash": True,
            },
        )
    shutil.rmtree(scratch / stage, ignore_errors=True)
    return _failure(stage, "CRASH_RECOVERY")


def validate(stage: str) -> dict[str, Any]:
    consumption, result, access = _stage_paths(stage)
    value = _read_json(result)
    if (
        not consumption.exists()
        or not access.exists()
        or value.get("rerun_permitted") is not False
    ):
        raise PermissionError("stage durability artifacts are incomplete")
    attempt_commit = _require_public_attempt(stage)
    events = _read_jsonl(access)
    completed = [event for event in events if event.get("event") == "GET_COMPLETED"]
    started = [event for event in events if event.get("event") == "GET_STARTED"]
    deleted = [event for event in events if event.get("event") == "MATRIX_DELETED"]
    failures = [
        event for event in events if event.get("event") == "GET_FAILED_TERMINALLY"
    ]
    designation = _designation()
    if stage == "score":
        _validate_score_result(value, designation)
    role, modalities = {
        "calibration": ("calibration", ("GEX", "FB")),
        "source": ("pilot", ("GEX", "FB")),
        "prediction": ("held", ("GEX",)),
        "score": ("held", ("FB",)),
    }[stage]
    expected = [
        (patient, modality)
        for patient in designation["role_order"][role]
        for modality in modalities
    ]
    records = {record["patient_id"]: record for record in designation["patients"]}
    started_keys = [
        (event.get("patient_id"), event.get("modality")) for event in started
    ]
    completed_keys = [
        (event.get("patient_id"), event.get("modality")) for event in completed
    ]
    deleted_keys = [
        (event.get("patient_id"), event.get("modality")) for event in deleted
    ]
    failure_keys = [
        (event.get("patient_id"), event.get("modality")) for event in failures
    ]
    allowed_events = {
        "OPENED_BEFORE_MATRIX_ACCESS",
        "GET_STARTED",
        "GET_COMPLETED",
        "GET_FAILED_TERMINALLY",
        "MATRIX_DELETED",
    }
    if (
        any(event.get("event") not in allowed_events for event in events)
        or started_keys != expected[: len(started_keys)]
        or completed_keys != expected[: len(completed_keys)]
        or deleted_keys != expected[: len(deleted_keys)]
        or len(started_keys) != len(set(started_keys))
        or len(completed_keys) != len(set(completed_keys))
        or len(deleted_keys) != len(set(deleted_keys))
        or len(failure_keys) > 1
        or any(key not in started_keys for key in completed_keys + failure_keys)
        or any(key not in deleted_keys for key in completed_keys)
        or any(
            event.get("body_existed") is not True
            and event.get("recovered_after_crash") is not True
            for event in deleted
            if (event.get("patient_id"), event.get("modality")) in completed_keys
        )
    ):
        raise PermissionError("stage access is not a unique allowed prefix")
    for event, key in zip(started, started_keys):
        record = records[key[0]]
        manifest = _manifest_file(record, key[1], MATRIX_MEMBER)
        if (
            event.get("stage") != stage
            or event.get("url") != _url(record, key[1], MATRIX_MEMBER)
            or event.get("expected_bytes") != manifest["expected_bytes"]
        ):
            raise PermissionError("matrix GET start binding differs")
    for event, key in zip(completed, completed_keys):
        manifest = _manifest_file(records[key[0]], key[1], MATRIX_MEMBER)
        digest = event.get("sha256")
        if (
            event.get("stage") != stage
            or event.get("bytes") != manifest["expected_bytes"]
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise PermissionError("completed matrix GET binding differs")
    if failures and failure_keys != started_keys[-1:]:
        raise PermissionError("terminal GET failure is not the final request")
    if value.get("status") != "TERMINAL_REFUSAL" and (
        started_keys != expected
        or completed_keys != expected
        or deleted_keys != expected
        or failures
    ):
        raise PermissionError("completed stage has an invalid exact access set")
    final_tags = {
        "calibration": (CALIBRATION_ATTEMPT_TAG, CALIBRATION_ATTEMPT, CALIBRATION_TAG),
        "source": (SOURCE_ATTEMPT_TAG, SOURCE_ATTEMPT, SOURCE_TAG),
        "prediction": (
            PREDICTION_ATTEMPT_TAG,
            PREDICTION_ATTEMPT,
            PREDICTION_TAG,
        ),
        "score": (SCORE_ATTEMPT_TAG, SCORE_ATTEMPT, SCORE_TAG),
    }
    _, attempt_path, final_tag = final_tags[stage]
    consumed = _read_json(consumption)
    if (
        consumed.get("schema") != f"gse313642-hcc-{stage}-consumption/2.0"
        or consumed.get("status") != "CONSUMED_BEFORE_FIRST_MATRIX_REQUEST"
        or consumed.get("attempt_sha256") != _sha256(attempt_path)
        or consumed.get("rerun_permitted") is not False
    ):
        raise PermissionError("stage consumption differs")
    private_identity = consumed.get("private_state_identity_sha256")
    if stage in {"prediction", "score"}:
        if (
            not isinstance(private_identity, str)
            or len(private_identity) != 64
            or any(
                character not in "0123456789abcdef" for character in private_identity
            )
        ):
            raise PermissionError("private state consumption binding differs")
        if stage == "prediction" and (
            consumed.get("private_state_expected_absent_before_prediction") is not True
            or "private_state_sha256" in consumed
        ):
            raise PermissionError("prediction private state boundary differs")
        if stage == "score":
            state_digest = consumed.get("private_state_sha256")
            if (
                not isinstance(state_digest, str)
                or len(state_digest) != 64
                or consumed.get("private_state_bytes", 0) <= 0
                or state_digest
                != _read_json(SCORE_AUTHORIZATION).get("private_state_sha256")
            ):
                raise PermissionError("score private state binding differs")
    elif private_identity is not None:
        raise PermissionError("unexpected private state consumption binding")
    final_commit = _require_public_tag(
        final_tag, (attempt_path, consumption, access, result)
    )
    _require_ancestor(attempt_commit, final_commit)
    return {
        "stage": stage,
        "status": value.get("status"),
        "result_sha256": _sha256(result),
        "completed_matrix_gets": len(completed),
        "matrix_deletions": len(deleted),
        "terminal_get_failures": len(failures),
        "final_commit": final_commit,
        "passes": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--axis-root", type=Path, required=True)
    subparsers.add_parser("freeze-implementation")
    for command in (
        "claim-calibration",
        "claim-source",
        "claim-prediction",
        "claim-score",
    ):
        current = subparsers.add_parser(command)
        current.add_argument("--token", type=Path, required=True)
    for command in ("run-calibration", "run-source"):
        current = subparsers.add_parser(command)
        current.add_argument("--token", type=Path, required=True)
        current.add_argument("--axis-root", type=Path, required=True)
        current.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    for command in ("run-prediction", "run-score"):
        current = subparsers.add_parser(command)
        current.add_argument("--token", type=Path, required=True)
        current.add_argument("--axis-root", type=Path, required=True)
        current.add_argument("--state", type=Path, required=True)
        current.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    subparsers.add_parser("authorize-source")
    subparsers.add_parser("authorize-prediction")
    subparsers.add_parser("authorize-score")
    for stage in ("calibration", "source", "prediction", "score"):
        recovery = subparsers.add_parser(f"recover-{stage}")
        recovery.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
        if stage in {"prediction", "score"}:
            recovery.add_argument("--state", type=Path, required=True)
        subparsers.add_parser(f"validate-{stage}")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "preflight":
        output = preflight(arguments.axis_root)
    elif arguments.command == "freeze-implementation":
        output = freeze_implementation()
    elif arguments.command == "claim-calibration":
        output = claim_calibration(arguments.token)
    elif arguments.command == "run-calibration":
        output = run_calibration(
            arguments.token, arguments.axis_root, scratch=arguments.scratch
        )
    elif arguments.command == "authorize-source":
        output = authorize_source()
    elif arguments.command == "claim-source":
        output = claim_source(arguments.token)
    elif arguments.command == "run-source":
        output = run_source(
            arguments.token, arguments.axis_root, scratch=arguments.scratch
        )
    elif arguments.command == "authorize-prediction":
        output = authorize_prediction()
    elif arguments.command == "claim-prediction":
        output = claim_prediction(arguments.token)
    elif arguments.command == "run-prediction":
        output = run_prediction(
            arguments.token,
            arguments.axis_root,
            arguments.state,
            scratch=arguments.scratch,
        )
    elif arguments.command == "authorize-score":
        output = authorize_score()
    elif arguments.command == "claim-score":
        output = claim_score(arguments.token)
    elif arguments.command == "run-score":
        output = run_score(
            arguments.token,
            arguments.axis_root,
            arguments.state,
            scratch=arguments.scratch,
        )
    elif arguments.command.startswith("recover-"):
        output = recover(
            arguments.command.removeprefix("recover-"),
            scratch=arguments.scratch,
            state_path=getattr(arguments, "state", None),
        )
    else:
        output = validate(arguments.command.removeprefix("validate-"))
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
