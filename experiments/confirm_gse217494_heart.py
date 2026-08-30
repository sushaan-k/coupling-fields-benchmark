"""One-shot held-heart confirmation for GSE217494 cardiac CITE-seq.

The command-line stages are deliberately separate. Claim commands create a
public attempt and a private capability without opening an assay body. Run
commands consume that capability, durably journal every request, and can never
retry a matrix. Recovery commands publish a terminal crash record without
reopening any remote object.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import tempfile
from typing import Any, BinaryIO, Iterable, Mapping, Sequence
import urllib.parse
import urllib.request

import numpy as np
import scipy

from experiments.gse217494_heart_core import (
    BOOTSTRAP_SEED,
    CELL_BUDGET,
    CLASSICAL_COMPARATORS,
    ETIOLOGIES,
    MANDATORY_COMPARATORS,
    MarkerSupportRefusal,
    NoCompleteConfigurationError,
    ConditionalFieldConfig,
    adt_high_states,
    adt_mean_profile,
    benjamini_hochberg,
    conditional_field_configurations,
    context_log_odds,
    destroy_adt_vectors,
    entity_deviance,
    evaluable_modules,
    evaluate_held_gate,
    evaluate_source_gate,
    exact_paired_sign_permutation,
    fit_standardized_pearson,
    joint_binary_tables,
    marker_knn_graph,
    module_knn_graph,
    neighbor_overlap_permutation,
    one_hot_context,
    predict_conditional_tables,
    predict_standardized_pearson,
    protein_fast_product_laplacian,
    rna_detection_profile,
    select_cv_configuration,
    select_fold_markers,
    select_strongest_classical,
    selected_cell_indices,
    stratified_paired_bootstrap,
)
from mapreg.common_effect_conditional import (
    fit_common_effect_conditional_log_odds,
)
from mapreg.heterogeneity_adaptive_coupling import CouplingEstimationRefusal
from mapreg.poisson_loglinear import (
    PoissonLoglinearFit,
    PoissonLoglinearRefusal,
    fit_poisson_loglinear_interaction,
    reconstruct_poisson_tables,
)
from mapreg.streamed_gzip_matrix_market import (
    GzipMatrixMarketValidationError,
    reduce_gzip_matrix_market,
)
from mapreg.structured_context_conditional import (
    StructuredContextConditionalFit,
    fit_structured_context_conditional_log_odds,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/confirmation/gse217494_heart"
DESIGNATION = DATA_DIR / "candidate_designation_v1.json"
HARDENING = DATA_DIR / "pre_access_hardening_v1.json"
IMPLEMENTATION_FREEZE = DATA_DIR / "pre_access_implementation_v1.json"
COGNATE_AXIS = DATA_DIR / "cognate_axis_v1.tsv"
PROTOCOL = ROOT / "docs/GSE217494_CARDIAC_CITESEQ_HELD_DONOR_PROTOCOL_2026-08-30.md"
IMPLEMENTATION_CLARIFICATIONS = ROOT / (
    "docs/GSE217494_CARDIAC_CITESEQ_IMPLEMENTATION_CLARIFICATIONS_2026-08-30.md"
)
RUNNER = ROOT / "experiments/confirm_gse217494_heart.py"
RUNNER_TEST = ROOT / "tests/test_gse217494_confirmation.py"

SOURCE_ATTEMPT = DATA_DIR / "source_attempt_v1.json"
SOURCE_CONSUMPTION = DATA_DIR / "source_consumption_v1.json"
SOURCE_ACCESS = DATA_DIR / "source_access_v1.jsonl"
SOURCE_RESULT = ROOT / "results/development/gse217494_heart_source_v1.json"
SCORE_AUTHORIZATION = DATA_DIR / "score_authorization_v1.json"
SCORE_ATTEMPT = DATA_DIR / "score_attempt_v1.json"
SCORE_CONSUMPTION = DATA_DIR / "score_consumption_v1.json"
SCORE_ACCESS = DATA_DIR / "score_access_v1.jsonl"
SCORE_RESULT = ROOT / "results/gse217494_heart_confirmation_v1.json"
DEFAULT_SCRATCH = Path("/private/tmp/gse217494-heart-v1")
STAGE_LOCK_DIRECTORY = DEFAULT_SCRATCH.parent

PUBLIC_ORIGIN = "https://github.com/sushaan-k/coupling-fields-benchmark.git"
CANDIDATE_TAG = "gse217494-heart-v1-candidate"
HARDENING_TAG = "gse217494-heart-v1-pre-access-hardening"
IMPLEMENTATION_TAG = "gse217494-heart-v1-implementation"
SOURCE_ATTEMPT_TAG = "gse217494-heart-v1-source-attempt"
SOURCE_TAG = "gse217494-heart-v1-source"
SCORE_AUTHORIZATION_TAG = "gse217494-heart-v1-score-authorization"
SCORE_ATTEMPT_TAG = "gse217494-heart-v1-score-attempt"
SCORE_TAG = "gse217494-heart-v1-score"

CANDIDATE_SHA256 = "bd35539fc61ffa771ea77df628eb4c918c44043896bcd2b058869e5d86032ba3"
HARDENING_SHA256 = "e7816874f7863f92b22c5510a329e57491832033a7c706a47acfd3dea3d6aef6"
COGNATE_SHA256 = "b12478eab2498a47fc7d3fe98aadf7fe5aef4caa167d1180abfab9d806f3b9a9"
PROTOCOL_SHA256 = "a37f2e9140e71a3d0a250cab7736df03fc49a446a537d6fef3a57a70d79f63ac"
IMPLEMENTATION_CLARIFICATIONS_SHA256 = (
    "5df207ca5884be96adbd2fb2c42556a357f08e1c213c337bd98a3897fbfbdbe4"
)

SOURCE_ORDER = (
    "sample2",
    "sample4",
    "sample7",
    "sample8",
    "sample13",
    "sample15",
    "sample17",
    "sample27",
    "sample28",
    "sample29",
    "sample30",
    "sample32",
    "sample33",
    "sample41",
)
HELD_ORDER = (
    "sample1",
    "sample5",
    "sample6",
    "sample9",
    "sample12",
    "sample34",
    "sample39",
    "sample42",
)
DEVIATION_GRID = (0.3, 3.0)
RIDGE_GRID = (0.1, 1.0)
GRAPH_GRID = (0.0, 0.01, 0.1, 1.0)
TRANSPORT_GRID = (0.75, 1.0, 1.25)
BOOTSTRAPS = 20_000
NEIGHBOR_PERMUTATIONS = 10_000
NEIGHBOR_SEED = 21_749_402
FEATURE_COUNT = 33_817
RNA_FEATURE_COUNT = 33_538
ADT_FEATURE_COUNT = 279
MINIMUM_FREE_BYTES = 1 << 30
TERMINAL_REFUSAL_CODES = {
    "ACQUISITION_HASH_MISMATCH",
    "ACQUISITION_RANGE_REFUSAL",
    "ACQUISITION_REDIRECT_REFUSAL",
    "ACQUISITION_RESPONSE_REFUSAL",
    "ACQUISITION_SIZE_MISMATCH",
    "AXIS_VALIDATION_FAILURE",
    "CLASSICAL_COMPARATOR_REFUSAL",
    "CRASH_RECOVERY",
    "DESTROYED_SELECTION_INVARIANCE_FAILURE",
    "MARKER_SUPPORT_REFUSAL",
    "MATRIX_VALIDATION_FAILURE",
    "NO_COMPLETE_SOURCE_CONFIGURATION",
    "NUMERICAL_REFUSAL",
    "SCRATCH_STATE_FAILURE",
    "SOURCE_PANEL_MISMATCH",
    "UNEXPECTED_EXCEPTION",
}

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

IMPLEMENTATION_BINDINGS = (
    "data/confirmation/gse217494_heart/candidate_designation_v1.json",
    "data/confirmation/gse217494_heart/pre_access_hardening_v1.json",
    "data/confirmation/gse217494_heart/pre_access_implementation_v1.json",
    "data/confirmation/gse217494_heart/cognate_axis_v1.tsv",
    "docs/GSE217494_CARDIAC_CITESEQ_HELD_DONOR_PROTOCOL_2026-08-30.md",
    "docs/GSE217494_CARDIAC_CITESEQ_IMPLEMENTATION_CLARIFICATIONS_2026-08-30.md",
    "experiments/confirm_gse217494_heart.py",
    "experiments/gse217494_heart_core.py",
    "tests/test_gse217494_confirmation.py",
    "tests/test_gse217494_heart_core.py",
    "mapreg/__init__.py",
    "mapreg/classical_residuals.py",
    "mapreg/coupling_fields.py",
    "mapreg/structured_context_conditional.py",
    "tests/test_structured_context_conditional.py",
    "mapreg/poisson_loglinear.py",
    "tests/test_poisson_loglinear.py",
    "mapreg/streamed_gzip_matrix_market.py",
    "tests/test_streamed_gzip_matrix_market.py",
    "mapreg/context_conditional_coupling.py",
    "mapreg/factorial_coupling.py",
    "mapreg/heterogeneity_adaptive_coupling.py",
    "mapreg/common_effect_conditional.py",
    "mapreg/table_prediction.py",
    "pyproject.toml",
)


class ProtocolRefusal(RuntimeError):
    """Terminal refusal with a stable public code and sanitized details."""

    def __init__(self, code: str, details: Mapping[str, Any] | None = None):
        super().__init__(code)
        self.code = code
        self.details = dict(details or {})


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


def _axis_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _write_json_x(path: Path, value: dict[str, Any]) -> None:
    _validate_public_payload(value)
    encoded = (
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
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _append_jsonl(path: Path, value: dict[str, Any], *, create: bool = False) -> None:
    _validate_public_payload(value)
    encoded = _canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_APPEND
    if create:
        flags |= os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "ab") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_public_payload(value: Any, key: str | None = None) -> None:
    forbidden_keys = {
        "claim_token",
        "claim_token_bytes",
        "barcodes",
        "selected_barcodes",
        "rna_counts",
        "adt_counts",
        "rna_states",
        "adt_states",
    }
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key in forbidden_keys:
                raise PermissionError(
                    f"public payload contains private key {child_key}"
                )
            _validate_public_payload(child, child_key)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_public_payload(child, key)
    elif isinstance(value, str) and (value.startswith("/") or value.startswith("~")):
        raise PermissionError(f"public payload contains a local path in {key}")


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


def _require_public_tag(tag: str, paths: Iterable[Path | str]) -> str:
    if _git("cat-file", "-t", tag).stdout.strip() != "tag":
        raise PermissionError(f"local tag {tag} is not annotated")
    local_object = _git("rev-parse", f"refs/tags/{tag}").stdout.strip()
    local_commit = _git("rev-parse", f"{tag}^{{}}").stdout.strip()
    remote_object, remote_commit = _remote_tag_ids(tag)
    if (local_object, local_commit) != (remote_object, remote_commit):
        raise PermissionError(f"public tag {tag} differs from the local tag")
    for supplied in paths:
        path = ROOT / supplied if isinstance(supplied, str) else supplied
        relative = _relative(path)
        published = _git("show", f"{tag}:{relative}", text=False).stdout
        if published != path.read_bytes():
            raise PermissionError(f"{relative} differs from public tag {tag}")
    return local_commit


def _require_ancestor(ancestor: str, descendant: str) -> None:
    result = _git("merge-base", "--is-ancestor", ancestor, descendant, check=False)
    if result.returncode != 0:
        raise PermissionError("public campaign tags do not form the required ancestry")


def _implementation_freeze() -> dict[str, Any]:
    value = _read_json(IMPLEMENTATION_FREEZE)
    expected_files = {
        relative: _sha256(ROOT / relative)
        for relative in IMPLEMENTATION_BINDINGS
        if ROOT / relative != IMPLEMENTATION_FREEZE
    }
    if (
        value.get("schema") != "gse217494-heart-pre-access-implementation/1.0"
        or value.get("status")
        != "FROZEN_BEFORE_ANY_NUMERIC_GSE217494_MATRIX_ACCESS"
        or value.get("required_cli_invocation")
        != "python3 -m experiments.confirm_gse217494_heart"
        or value.get("required_implementation_tag") != IMPLEMENTATION_TAG
        or value.get("implementation_files_sha256") != expected_files
        or value.get("required_runtime") != REQUIRED_RUNTIME
        or value.get("candidate_sha256") != CANDIDATE_SHA256
        or value.get("hardening_sha256") != HARDENING_SHA256
        or value.get("protocol_sha256") != PROTOCOL_SHA256
        or value.get("implementation_clarifications_sha256")
        != IMPLEMENTATION_CLARIFICATIONS_SHA256
        or value.get("cognate_axis_sha256") != COGNATE_SHA256
        or value.get("numeric_matrix_body_header_or_entry_read") is not False
        or value.get("held_matrix_body_header_or_entry_requested") is not False
        or value.get("rerun_permitted") is not False
    ):
        raise PermissionError("pre-access implementation freeze differs")
    clarifications = value.get("clarifications", {})
    if (
        clarifications.get("destroyed_selection")
        != "rerun full selection and graphs; assert invariance to complete-vector permutation"
        or clarifications.get("secondary_module_graph_neighbors")
        != "min(3, module_marker_count - 1)"
        or clarifications.get("exact_common_unavailability")
        != "publish unavailable; residual and independence remain eligible"
    ):
        raise PermissionError("implementation clarifications differ")
    return value


def _verify_implementation_freeze() -> dict[str, str]:
    if (
        _sha256(DESIGNATION) != CANDIDATE_SHA256
        or _sha256(HARDENING) != HARDENING_SHA256
        or _sha256(PROTOCOL) != PROTOCOL_SHA256
        or _sha256(IMPLEMENTATION_CLARIFICATIONS)
        != IMPLEMENTATION_CLARIFICATIONS_SHA256
        or _sha256(COGNATE_AXIS) != COGNATE_SHA256
    ):
        raise PermissionError("frozen campaign bytes differ")
    candidate_commit = _require_public_tag(CANDIDATE_TAG, (DESIGNATION,))
    hardening_commit = _require_public_tag(
        HARDENING_TAG, (DESIGNATION, HARDENING, PROTOCOL, COGNATE_AXIS)
    )
    _require_ancestor(candidate_commit, hardening_commit)
    _implementation_freeze()
    implementation_commit = _require_public_tag(
        IMPLEMENTATION_TAG, IMPLEMENTATION_BINDINGS
    )
    _require_ancestor(hardening_commit, implementation_commit)
    return {
        "candidate_tag": CANDIDATE_TAG,
        "candidate_commit": candidate_commit,
        "hardening_tag": HARDENING_TAG,
        "hardening_commit": hardening_commit,
        "implementation_tag": IMPLEMENTATION_TAG,
        "implementation_commit": implementation_commit,
    }


def _binding_hashes() -> dict[str, str]:
    return {relative: _sha256(ROOT / relative) for relative in IMPLEMENTATION_BINDINGS}


def _contract() -> dict[str, Any]:
    candidate = _read_json(DESIGNATION)
    hardening = _read_json(HARDENING)
    if (
        candidate.get("schema") != "gse217494-heart-citeseq-candidate/1.0"
        or candidate.get("source_samples") != list(SOURCE_ORDER)
        or candidate.get("held_samples") != list(HELD_ORDER)
        or candidate.get("source_donor_count") != len(SOURCE_ORDER)
        or candidate.get("held_donor_count") != len(HELD_ORDER)
        or candidate.get("primary_cell_budget_per_donor") != CELL_BUDGET
        or candidate.get("rerun_permitted") is not False
        or hardening.get("schema") != "gse217494-heart-pre-access-hardening/1.0"
        or hardening.get("rerun_permitted") is not False
    ):
        raise PermissionError("candidate or hardening contract differs")
    samples = candidate.get("samples", [])
    by_sample = {record.get("sample"): dict(record) for record in samples}
    if set(by_sample) != set(SOURCE_ORDER + HELD_ORDER):
        raise PermissionError("frozen sample axis differs")
    barcode = {record["sample"]: record for record in hardening["barcode_axes"]}
    feature = {record["sample"]: record for record in hardening["feature_axes"]}
    if set(barcode) != set(by_sample) or set(feature) != set(by_sample):
        raise PermissionError("frozen feature or barcode inventory differs")
    base_url = candidate.get("files", {}).get("base_url")
    if base_url != "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE217nnn/GSE217494/suppl":
        raise PermissionError("frozen GEO base URL differs")
    for sample, record in by_sample.items():
        role = "source" if sample in SOURCE_ORDER else "held"
        if (
            record.get("role") != role
            or barcode[sample].get("role") != role
            or feature[sample].get("role") != role
        ):
            raise PermissionError("frozen sample roles differ")
        record["barcode_axis"] = barcode[sample]
        record["feature_axis"] = feature[sample]
        record["urls"] = {
            kind: f"{base_url}/GSE217494_{sample}.{suffix}"
            for kind, suffix in (
                ("barcodes", "barcodes.tsv.gz"),
                ("features", "features.tsv.gz"),
                ("matrix", "matrix.mtx.gz"),
            )
        }
    return {
        "candidate": candidate,
        "hardening": hardening,
        "source": [by_sample[sample] for sample in SOURCE_ORDER],
        "held": [by_sample[sample] for sample in HELD_ORDER],
        "base_url": base_url,
    }


def _cognates() -> list[dict[str, Any]]:
    lines = COGNATE_AXIS.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].split("\t") != [
        "symbol",
        "rna_row_1based",
        "rna_feature_id",
        "adt_row_1based",
        "adt_feature_id",
    ]:
        raise PermissionError("cognate axis header differs")
    records: list[dict[str, Any]] = []
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != 5:
            raise PermissionError("cognate axis row differs")
        records.append(
            {
                "symbol": fields[0],
                "rna_row_1based": int(fields[1]),
                "rna_feature_id": fields[2],
                "adt_row_1based": int(fields[3]),
                "adt_feature_id": fields[4],
            }
        )
    if (
        len(records) != 249
        or len({record["symbol"] for record in records}) != 249
        or any(
            not (1 <= record["rna_row_1based"] <= RNA_FEATURE_COUNT)
            for record in records
        )
        or any(
            not (RNA_FEATURE_COUNT < record["adt_row_1based"] <= FEATURE_COUNT)
            for record in records
        )
    ):
        raise PermissionError("cognate axis contents differ")
    return records


def _outside_repository(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        return resolved
    raise PermissionError(f"{label} must be outside the repository")


def _prepare_scratch(path: Path) -> Path:
    scratch = _outside_repository(path, "scratch directory")
    scratch.mkdir(parents=True, exist_ok=True, mode=0o700)
    scratch.chmod(0o700)
    if any(scratch.iterdir()):
        raise PermissionError("scratch directory must be empty")
    if shutil.disk_usage(scratch).free < MINIMUM_FREE_BYTES:
        raise PermissionError("scratch directory has less than one GiB free")
    return scratch


def _require_token_outside_scratch(token: Path, scratch: Path) -> None:
    token_path = _outside_repository(token, "claim token")
    scratch_path = _outside_repository(scratch, "scratch directory")
    try:
        token_path.relative_to(scratch_path)
    except ValueError:
        return
    raise PermissionError("claim token must be outside the scratch directory")


def _scratch_identity(path: Path) -> str:
    resolved = _outside_repository(path, "scratch directory")
    return hashlib.sha256(os.fsencode(str(resolved))).hexdigest()


def _create_claim_token(path: Path) -> str:
    token_path = _outside_repository(path, "claim token")
    token_path.parent.mkdir(parents=True, exist_ok=True)
    payload = os.urandom(32)
    descriptor = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(token_path.parent)
    return hashlib.sha256(payload).hexdigest()


def _validate_token(path: Path | None, expected_sha256: str) -> Path:
    if path is None:
        raise PermissionError("--claim-token is required")
    token_path = _outside_repository(path, "claim token")
    payload = token_path.read_bytes()
    if len(payload) != 32 or hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise PermissionError("private claim token differs from the public attempt")
    return token_path


def _consume_token(path: Path) -> None:
    path.unlink()
    _fsync_directory(path.parent)


@contextmanager
def _stage_lock(scratch: Path, stage: str) -> Iterable[None]:
    if stage not in {"source", "score"}:
        raise ValueError("stage lock requires source or score")
    _outside_repository(scratch, "scratch directory")
    lock_path = STAGE_LOCK_DIRECTORY / f".gse217494-heart-v1.{stage}.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PermissionError(f"{stage} execution is still active") from error
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        raise ProtocolRefusal("ACQUISITION_REDIRECT_REFUSAL")


def _validate_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "ftp.ncbi.nlm.nih.gov"
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/geo/series/GSE217nnn/GSE217494/suppl/")
    ):
        raise PermissionError("download URL differs from the frozen GEO origin")


def _open_url(request: urllib.request.Request) -> BinaryIO:
    return urllib.request.build_opener(_NoRedirectHandler()).open(request, timeout=120)


def _journal_event(path: Path, stage: str, event: str, **fields: Any) -> None:
    _append_jsonl(
        path,
        {
            "schema": "gse217494-heart-access-journal/1.0",
            "stage": stage,
            "event": event,
            "created_at_utc": _timestamp(),
            **fields,
        },
    )


def _download(
    *,
    stage: str,
    sample: str,
    kind: str,
    url: str,
    destination: Path,
    journal: Path,
    expected_bytes: int,
    expected_sha256: str | None,
) -> dict[str, Any]:
    """Issue one non-retrying streaming GET and independently hash its bytes."""

    _validate_url(url)
    if destination.exists():
        raise ProtocolRefusal("SCRATCH_STATE_FAILURE", {"sample": sample, "kind": kind})
    _journal_event(
        journal,
        stage,
        "REQUEST_STARTED",
        sample=sample,
        kind=kind,
        url=url,
        filename=destination.name,
        expected_bytes=int(expected_bytes),
        expected_sha256=expected_sha256,
        method="GET",
        range_header=None,
        automatic_retry_count=0,
        streaming=True,
    )
    request = urllib.request.Request(
        url,
        method="GET",
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": "coupling-fields-benchmark/1",
        },
    )
    digest = hashlib.sha256()
    observed = 0
    try:
        with _open_url(request) as response:
            status_code = int(getattr(response, "status", response.getcode()))
            response_url = str(response.geturl())
            if status_code != 200 or response_url != url:
                raise ProtocolRefusal(
                    "ACQUISITION_RESPONSE_REFUSAL",
                    {"sample": sample, "kind": kind, "status_code": status_code},
                )
            if response.headers.get("Content-Range") is not None:
                raise ProtocolRefusal(
                    "ACQUISITION_RANGE_REFUSAL", {"sample": sample, "kind": kind}
                )
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
                    observed += len(block)
                output.flush()
                os.fsync(output.fileno())
        _fsync_directory(destination.parent)
        observed_sha256 = digest.hexdigest()
        _journal_event(
            journal,
            stage,
            "REQUEST_FINISHED",
            sample=sample,
            kind=kind,
            filename=destination.name,
            observed_bytes=observed,
            observed_sha256=observed_sha256,
            status_code=200,
            response_url=url,
        )
        if observed != int(expected_bytes):
            raise ProtocolRefusal(
                "ACQUISITION_SIZE_MISMATCH",
                {"sample": sample, "kind": kind, "observed_bytes": observed},
            )
        if expected_sha256 is not None and observed_sha256 != expected_sha256:
            raise ProtocolRefusal(
                "ACQUISITION_HASH_MISMATCH", {"sample": sample, "kind": kind}
            )
        return {"observed_bytes": observed, "observed_sha256": observed_sha256}
    except BaseException as error:
        _journal_event(
            journal,
            stage,
            "REQUEST_FAILED",
            sample=sample,
            kind=kind,
            filename=destination.name,
            observed_bytes=observed,
            partial_sha256=digest.hexdigest(),
            reason_code=(
                error.code
                if isinstance(error, ProtocolRefusal)
                else "UNEXPECTED_EXCEPTION"
            ),
        )
        raise


def _delete_download(
    path: Path, *, stage: str, sample: str, kind: str, journal: Path
) -> None:
    existed = path.exists()
    path.unlink(missing_ok=True)
    if existed:
        _fsync_directory(path.parent)
    _journal_event(
        journal,
        stage,
        "DOWNLOAD_DELETED",
        sample=sample,
        kind=kind,
        filename=path.name,
        existed_before_delete=existed,
        deleted=not path.exists(),
    )


def _decode_gzip_axis(
    path: Path, *, expected_sha256: str, expected_count: int
) -> tuple[bytes, list[str]]:
    try:
        raw = gzip.decompress(path.read_bytes())
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ProtocolRefusal("AXIS_VALIDATION_FAILURE") from error
    values = text.splitlines()
    if (
        hashlib.sha256(raw).hexdigest() != expected_sha256
        or len(values) != expected_count
        or any(not value for value in values)
    ):
        raise ProtocolRefusal("AXIS_VALIDATION_FAILURE")
    return raw, values


def _validate_feature_axis(raw: bytes, cognates: Sequence[Mapping[str, Any]]) -> None:
    rows = raw.decode("utf-8").splitlines()
    parsed = [row.split("\t") for row in rows]
    if any(len(row) < 3 for row in parsed):
        raise ProtocolRefusal("AXIS_VALIDATION_FAILURE")
    if any(row[2] != "Gene Expression" for row in parsed[:RNA_FEATURE_COUNT]) or any(
        row[2] != "Antibody Capture" for row in parsed[RNA_FEATURE_COUNT:]
    ):
        raise ProtocolRefusal("AXIS_VALIDATION_FAILURE")
    for record in cognates:
        rna = parsed[int(record["rna_row_1based"]) - 1]
        adt = parsed[int(record["adt_row_1based"]) - 1]
        if (
            rna[0] != record["rna_feature_id"]
            or rna[1] != record["symbol"]
            or adt[0] != record["adt_feature_id"]
            or adt[1] != record["symbol"]
        ):
            raise ProtocolRefusal("AXIS_VALIDATION_FAILURE")


def _reduce_sample(
    sample: Mapping[str, Any],
    *,
    stage: str,
    scratch: Path,
    journal: Path,
    cognates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Download one triplet, reduce selected rows/cells, then delete it."""

    sample_id = str(sample["sample"])
    files: dict[str, Path] = {}
    downloads: dict[str, dict[str, Any]] = {}
    try:
        for kind in ("features", "barcodes"):
            suffix = "features.tsv.gz" if kind == "features" else "barcodes.tsv.gz"
            path = scratch / f"GSE217494_{sample_id}.{suffix}"
            files[kind] = path
            axis = sample[f"{kind[:-1] if kind.endswith('s') else kind}_axis"]
            downloads[kind] = _download(
                stage=stage,
                sample=sample_id,
                kind=kind,
                url=sample["urls"][kind],
                destination=path,
                journal=journal,
                expected_bytes=int(axis["bytes"]),
                expected_sha256=str(axis["gzip_sha256"]),
            )
        feature_raw, _ = _decode_gzip_axis(
            files["features"],
            expected_sha256=str(sample["feature_axis"]["axis_sha256"]),
            expected_count=FEATURE_COUNT,
        )
        _validate_feature_axis(feature_raw, cognates)
        barcode_raw, barcodes = _decode_gzip_axis(
            files["barcodes"],
            expected_sha256=str(sample["barcode_axis"]["axis_sha256"]),
            expected_count=int(sample["barcode_axis"]["count"]),
        )
        if len(set(barcodes)) != len(barcodes):
            raise ProtocolRefusal("AXIS_VALIDATION_FAILURE")
        selected_columns_zero = selected_cell_indices(barcodes, sample_id)
        selected_barcodes = tuple(barcodes[index] for index in selected_columns_zero)
        for kind in ("features", "barcodes"):
            _delete_download(
                files[kind], stage=stage, sample=sample_id, kind=kind, journal=journal
            )
            del files[kind]

        matrix_path = scratch / f"GSE217494_{sample_id}.matrix.mtx.gz"
        files["matrix"] = matrix_path
        downloads["matrix"] = _download(
            stage=stage,
            sample=sample_id,
            kind="matrix",
            url=sample["urls"]["matrix"],
            destination=matrix_path,
            journal=journal,
            expected_bytes=int(sample["matrix_bytes"]),
            expected_sha256=None,
        )
        selected_rows = [int(record["rna_row_1based"]) for record in cognates]
        selected_rows.extend(range(RNA_FEATURE_COUNT + 1, FEATURE_COUNT + 1))
        try:
            block, parser_audit = reduce_gzip_matrix_market(
                matrix_path,
                expected_shape=(FEATURE_COUNT, len(barcodes)),
                selected_rows=selected_rows,
                selected_columns=(selected_columns_zero + 1).tolist(),
            )
        except GzipMatrixMarketValidationError as error:
            partial = (
                asdict(error.partial_audit) if error.partial_audit is not None else None
            )
            _journal_event(
                journal,
                stage,
                "MATRIX_PARSE_FAILED",
                sample=sample_id,
                kind="matrix",
                partial_audit=partial,
            )
            raise ProtocolRefusal(
                "MATRIX_VALIDATION_FAILURE", {"sample": sample_id}
            ) from error
        _journal_event(
            journal,
            stage,
            "MATRIX_PARSE_FINISHED",
            sample=sample_id,
            kind="matrix",
            parser_audit=asdict(parser_audit),
            independent_compressed_bytes=downloads["matrix"]["observed_bytes"],
            independent_compressed_sha256=downloads["matrix"]["observed_sha256"],
        )
        if block.shape != (len(cognates) + ADT_FEATURE_COUNT, CELL_BUDGET):
            raise ProtocolRefusal("MATRIX_VALIDATION_FAILURE", {"sample": sample_id})
        candidate_rna = block[: len(cognates)].T.copy()
        all_adt = block[len(cognates) :].T.copy()
        adt_indices = np.asarray(
            [
                int(record["adt_row_1based"]) - RNA_FEATURE_COUNT - 1
                for record in cognates
            ],
            dtype=np.int64,
        )
        candidate_adt = all_adt[:, adt_indices]
        symbols = [str(record["symbol"]) for record in cognates]
        rna_states = (candidate_rna > 0).astype(np.uint8)
        adt_states = adt_high_states(
            candidate_adt, selected_barcodes, sample_id, symbols
        )
        destroyed_all_adt = destroy_adt_vectors(all_adt, selected_barcodes, sample_id)
        destroyed_candidate_adt = destroyed_all_adt[:, adt_indices]
        destroyed_adt_states = adt_high_states(
            destroyed_candidate_adt, selected_barcodes, sample_id, symbols
        )
        result = {
            "sample": sample_id,
            "etiology": str(sample["etiology"]),
            "selected_barcodes_private": selected_barcodes,
            "selected_cell_indices": selected_columns_zero,
            "rna_counts_private": candidate_rna,
            "all_adt_counts_private": all_adt,
            "adt_counts_private": candidate_adt,
            "destroyed_adt_counts_private": destroyed_candidate_adt,
            "rna_profile": rna_detection_profile(candidate_rna),
            "adt_profile": adt_mean_profile(all_adt, adt_indices),
            "destroyed_adt_profile": adt_mean_profile(destroyed_all_adt, adt_indices),
            "tables": joint_binary_tables(rna_states, adt_states),
            "destroyed_tables": joint_binary_tables(rna_states, destroyed_adt_states),
            "public_record": {
                "sample": sample_id,
                "etiology": str(sample["etiology"]),
                "selected_cells": CELL_BUDGET,
                "selected_cell_indices_sha256": _array_sha256(selected_columns_zero),
                "selected_cell_axis_sha256": _axis_sha256(selected_barcodes),
                "full_barcode_axis_sha256": hashlib.sha256(barcode_raw).hexdigest(),
                "feature_axis_sha256": hashlib.sha256(feature_raw).hexdigest(),
                "compressed_downloads": downloads,
                "matrix_parser_audit_sha256": _canonical_json_sha256(
                    asdict(parser_audit)
                ),
            },
        }
        return result
    finally:
        for kind, path in tuple(files.items()):
            _delete_download(
                path, stage=stage, sample=sample_id, kind=kind, journal=journal
            )


def _selection(
    records: Sequence[Mapping[str, Any]],
    cognates: Sequence[Mapping[str, Any]],
    *,
    destroyed: bool,
) -> tuple[Any, Any, Any, np.ndarray]:
    symbols = [str(record["symbol"]) for record in cognates]
    rna_counts = np.asarray(
        [record["rna_counts_private"] for record in records], dtype=np.int64
    )
    adt_key = "destroyed_adt_counts_private" if destroyed else "adt_counts_private"
    profile_key = "destroyed_adt_profile" if destroyed else "adt_profile"
    table_key = "destroyed_tables" if destroyed else "tables"
    adt_counts = np.asarray([record[adt_key] for record in records], dtype=np.int64)
    rna_profiles = np.asarray(
        [record["rna_profile"] for record in records], dtype=float
    )
    adt_profiles = np.asarray([record[profile_key] for record in records], dtype=float)
    tables = np.asarray([record[table_key] for record in records], dtype=np.int64)
    selected = select_fold_markers(
        symbols,
        np.count_nonzero(rna_counts > 0, axis=1),
        adt_counts,
        rna_profiles,
        adt_profiles,
        tables,
    )
    indices = np.asarray(selected.indices, dtype=np.int64)
    rna_graph = marker_knn_graph(rna_profiles[:, indices], selected.symbols)
    adt_graph = marker_knn_graph(adt_profiles[:, indices], selected.symbols)
    laplacian = protein_fast_product_laplacian(rna_graph.laplacian, adt_graph.laplacian)
    return selected, rna_graph, adt_graph, laplacian


def _assert_destroyed_invariance(
    records: Sequence[Mapping[str, Any]],
    cognates: Sequence[Mapping[str, Any]],
    real: tuple[Any, Any, Any, np.ndarray],
    destroyed: tuple[Any, Any, Any, np.ndarray],
) -> dict[str, Any]:
    real_selection, real_rna, real_adt, real_product = real
    shifted_selection, shifted_rna, shifted_adt, shifted_product = destroyed
    real_profiles = np.asarray(
        [record["adt_profile"] for record in records], dtype=float
    )
    shifted_profiles = np.asarray(
        [record["destroyed_adt_profile"] for record in records], dtype=float
    )
    if (
        real_selection.indices != shifted_selection.indices
        or real_selection.symbols != shifted_selection.symbols
        or not np.allclose(real_profiles, shifted_profiles, rtol=0.0, atol=1e-12)
        or not np.array_equal(real_rna.adjacency, shifted_rna.adjacency)
        or not np.array_equal(real_adt.adjacency, shifted_adt.adjacency)
        or not np.allclose(real_product, shifted_product, rtol=0.0, atol=1e-12)
    ):
        raise ProtocolRefusal("DESTROYED_SELECTION_INVARIANCE_FAILURE")
    return {
        "marker_axis_identical": True,
        "rna_graph_identical": True,
        "adt_graph_identical": True,
        "maximum_absolute_adt_profile_difference": float(
            np.max(np.abs(real_profiles - shifted_profiles))
        ),
        "product_laplacian_identical_within_1e_12": True,
    }


def _selected_tables(
    records: Sequence[Mapping[str, Any]], indices: Sequence[int], *, destroyed: bool
) -> np.ndarray:
    key = "destroyed_tables" if destroyed else "tables"
    selected = np.asarray(tuple(indices), dtype=np.int64)
    return np.asarray([record[key] for record in records], dtype=np.int64)[:, selected][
        :, :, selected
    ]


def _source_fold_partition(
    records: Sequence[Mapping[str, Any]], validation: int
) -> tuple[list[Mapping[str, Any]], np.ndarray]:
    """Return training records and the untouched validation truth table."""

    index = int(validation)
    if index < 0 or index >= len(records):
        raise ValueError("validation index is outside the source donor axis")
    training = [record for donor, record in enumerate(records) if donor != index]
    truth = np.asarray(records[index]["tables"], dtype=np.int64).copy()
    return training, truth


def _source_fold_artifacts(
    records: Sequence[Mapping[str, Any]],
    cognates: Sequence[Mapping[str, Any]],
    validation: int,
) -> dict[str, Any]:
    training, validation_truth = _source_fold_partition(records, validation)
    real_selection = _selection(training, cognates, destroyed=False)
    destroyed_selection = _selection(training, cognates, destroyed=True)
    invariance = _assert_destroyed_invariance(
        training, cognates, real_selection, destroyed_selection
    )
    selected, rna_graph, adt_graph, product_laplacian = real_selection
    _, _, _, destroyed_laplacian = destroyed_selection
    selected_indices = np.asarray(selected.indices, dtype=np.int64)
    return {
        "training": training,
        "training_labels": [str(record["etiology"]) for record in training],
        "selected": selected,
        "selected_indices": selected_indices,
        "rna_graph": rna_graph,
        "adt_graph": adt_graph,
        "product_laplacian": product_laplacian,
        "destroyed_laplacian": destroyed_laplacian,
        "training_tables": _selected_tables(
            training, selected_indices, destroyed=False
        ),
        "shifted_tables": _selected_tables(training, selected_indices, destroyed=True),
        "truth": validation_truth[selected_indices][:, selected_indices],
        "destroyed_invariance": invariance,
    }


def _margins(tables: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(tables)
    return values.sum(axis=-1), values.sum(axis=-2)


def _fit_structured(
    tables: np.ndarray,
    etiologies: Sequence[str],
    laplacian: np.ndarray,
    configuration: ConditionalFieldConfig,
) -> StructuredContextConditionalFit:
    return fit_structured_context_conditional_log_odds(
        tables,
        one_hot_context(etiologies),
        graph_laplacian=laplacian,
        donor_deviation_penalty=configuration.donor_deviation_penalty,
        coefficient_ridge_penalty=configuration.coefficient_ridge_penalty,
        graph_penalty=configuration.graph_penalty,
        minimum_informative_donors=len(tables),
        maximum_condition_number=1e12,
        maximum_iterations=100,
        tolerance=1e-8,
    )


def _structured_prediction(
    fit: StructuredContextConditionalFit,
    etiology: str,
    transport: float,
    truth: np.ndarray,
) -> np.ndarray:
    rows, columns = _margins(truth)
    field = context_log_odds(
        fit.coefficient, (etiology,), transport_multiplier=transport
    )[0]
    return predict_conditional_tables(field, rows, columns)


def _poisson_field(fit: PoissonLoglinearFit, etiology: str | None) -> np.ndarray:
    label: object = "__pooled__" if etiology is None else etiology
    try:
        index = fit.group_labels.index(label)
    except ValueError as error:
        raise ProtocolRefusal("CLASSICAL_COMPARATOR_REFUSAL") from error
    return fit.log_odds[index]


def _poisson_prediction(
    fit: PoissonLoglinearFit,
    etiology: str | None,
    transport: float,
    truth: np.ndarray,
) -> np.ndarray:
    rows, columns = _margins(truth)
    return reconstruct_poisson_tables(
        _poisson_field(fit, etiology),
        rows,
        columns,
        transport_scale=transport,
    ).table


def _poisson_reconstruction_certificate(reconstruction: Any) -> dict[str, Any]:
    nullable = np.empty(reconstruction.reconstructed_log_odds.shape, dtype=object)
    nullable[...] = None
    mask = np.asarray(reconstruction.informative_margin_mask, dtype=bool)
    nullable[mask] = reconstruction.reconstructed_log_odds[mask]
    return {
        "informative_margin_count": reconstruction.informative_margin_count,
        "degenerate_margin_count": reconstruction.degenerate_margin_count,
        "informative_margin_mask_sha256": _array_sha256(mask.astype(np.uint8)),
        "reconstructed_log_odds": nullable.tolist(),
        "maximum_absolute_row_margin_error": reconstruction.maximum_absolute_row_margin_error,
        "maximum_absolute_column_margin_error": reconstruction.maximum_absolute_column_margin_error,
        "maximum_absolute_log_odds_error_on_informative_margins": reconstruction.maximum_absolute_log_odds_error,
    }


def _pearson_prediction(
    coordinate: np.ndarray,
    etiology: str,
    transport: float,
    truth: np.ndarray,
) -> np.ndarray:
    rows, columns = _margins(truth)
    field = context_log_odds(coordinate, (etiology,), transport_multiplier=1.0)[0]
    return predict_standardized_pearson(
        field, rows, columns, transport_multiplier=transport
    )


def _common_prediction(
    log_odds: np.ndarray, transport: float, truth: np.ndarray
) -> np.ndarray:
    rows, columns = _margins(truth)
    return predict_conditional_tables(transport * log_odds, rows, columns)


def _independence_prediction(truth: np.ndarray) -> np.ndarray:
    rows, columns = _margins(truth)
    return predict_standardized_pearson(
        np.zeros(truth.shape[:-2], dtype=float), rows, columns
    )


def _fit_certificate(fit: StructuredContextConditionalFit) -> dict[str, Any]:
    return {
        "optimizer": fit.optimizer,
        "converged": fit.converged,
        "iterations": fit.iterations,
        "objective": fit.objective,
        "gradient_norm": fit.gradient_norm,
        "scaled_gradient_norm": fit.scaled_gradient_norm,
        "schur_condition_number": fit.schur_condition_number,
        "donor_curvature_condition_number": fit.donor_curvature_condition_number,
        "minimum_schur_eigenvalue": fit.minimum_schur_eigenvalue,
        "maximum_schur_eigenvalue": fit.maximum_schur_eigenvalue,
        "minimum_donor_curvature": fit.minimum_donor_curvature,
        "maximum_donor_curvature": fit.maximum_donor_curvature,
        "graph_symmetry_residual": fit.graph_symmetry_residual,
        "graph_minimum_eigenvalue": fit.graph_minimum_eigenvalue,
        "graph_maximum_eigenvalue": fit.graph_maximum_eigenvalue,
        "graph_nullity": fit.graph_nullity,
        "graph_source": fit.graph_source,
        "maximum_condition_number": fit.maximum_condition_number,
        "gradient_tolerance": fit.gradient_tolerance,
    }


def _poisson_certificate(fit: PoissonLoglinearFit) -> dict[str, Any]:
    return {
        "estimator": fit.estimator,
        "converged": fit.converged,
        "maximum_absolute_score": fit.maximum_absolute_score,
        "maximum_scaled_score": fit.maximum_scaled_score,
        "maximum_absolute_row_margin_error": fit.maximum_absolute_row_margin_error,
        "maximum_absolute_column_margin_error": fit.maximum_absolute_column_margin_error,
        "maximum_absolute_log_odds_error": fit.maximum_absolute_log_odds_error,
        "minimum_positive_fitted_mean": fit.minimum_positive_fitted_mean,
        "score_tolerance": fit.score_tolerance,
        "certificate_tolerance": fit.certificate_tolerance,
        "pseudocount": fit.pseudocount,
        "included_table_count_sha256": _array_sha256(fit.included_table_count),
        "degenerate_table_count_sha256": _array_sha256(fit.degenerate_table_count),
    }


def _graph_payload(graph: Any) -> dict[str, Any]:
    return {
        "symbols": list(graph.symbols),
        "neighbors": [list(values) for values in graph.neighbors],
        "adjacency": graph.adjacency.astype(np.uint8).tolist(),
        "adjacency_sha256": _array_sha256(graph.adjacency.astype(np.uint8)),
        "laplacian": graph.laplacian.tolist(),
        "laplacian_sha256": _array_sha256(graph.laplacian),
        "mean_diagonal": float(np.mean(np.diag(graph.laplacian))),
    }


def _configuration_payload(configuration: ConditionalFieldConfig) -> dict[str, float]:
    return {
        "donor_deviation_penalty": configuration.donor_deviation_penalty,
        "coefficient_ridge_penalty": configuration.coefficient_ridge_penalty,
        "graph_penalty": configuration.graph_penalty,
        "transport_multiplier": configuration.transport_multiplier,
    }


def _complete_grid_payload(
    losses: Mapping[Any, np.ndarray], serializer: Any
) -> list[dict[str, Any]]:
    output = []
    for configuration in sorted(losses):
        values = np.asarray(losses[configuration], dtype=float)
        complete = bool(np.isfinite(values).all())
        output.append(
            {
                "configuration": serializer(configuration),
                "complete": complete,
                "donor_losses": values.tolist() if complete else None,
            }
        )
    return output


def _select_transport(losses: Mapping[float, np.ndarray]) -> tuple[float, np.ndarray]:
    complete = [
        (float(transport), np.asarray(values, dtype=float))
        for transport, values in losses.items()
        if np.isfinite(values).all()
    ]
    if not complete:
        raise ProtocolRefusal("NO_COMPLETE_SOURCE_CONFIGURATION")
    return min(complete, key=lambda item: (float(item[1].mean()), item[0]))


def _select_final_classical(
    pearson_losses: np.ndarray,
    common_losses: np.ndarray | None,
    independence_losses: np.ndarray,
    *,
    final_common_available: bool,
) -> tuple[Any, np.ndarray]:
    eligible_common = common_losses if final_common_available else None
    candidates = {
        "standardized_fixed_margin_pearson": pearson_losses,
        "exact_common_effect_conditional_field": eligible_common,
        "fixed_margin_independence": independence_losses,
    }
    selected = select_strongest_classical(candidates)
    selected_losses = candidates[selected.selected]
    if selected_losses is None:
        raise ProtocolRefusal("CLASSICAL_COMPARATOR_REFUSAL")
    return selected, np.asarray(selected_losses, dtype=float)


def _model_array(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ProtocolRefusal("NUMERICAL_REFUSAL")
    return {
        "values": array.tolist(),
        "shape": list(array.shape),
        "sha256": _array_sha256(array),
    }


def _fit_source_models(
    records: Sequence[Mapping[str, Any]], cognates: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Run the complete source-only LOHO selection, fits, and promotion gate."""

    donor_count = len(records)
    if donor_count != len(SOURCE_ORDER):
        raise ProtocolRefusal("SOURCE_PANEL_MISMATCH")
    labels = [str(record["etiology"]) for record in records]
    configurations = conditional_field_configurations()
    primary_losses = {
        configuration: np.full(donor_count, np.nan) for configuration in configurations
    }
    destroyed_losses = {
        configuration: np.full(donor_count, np.nan) for configuration in configurations
    }
    pooled_losses = {
        transport: np.full(donor_count, np.nan) for transport in TRANSPORT_GRID
    }
    etiology_poisson_losses = {
        transport: np.full(donor_count, np.nan) for transport in TRANSPORT_GRID
    }
    pearson_losses = {
        transport: np.full(donor_count, np.nan) for transport in TRANSPORT_GRID
    }
    common_losses = {
        transport: np.full(donor_count, np.nan) for transport in TRANSPORT_GRID
    }
    independence_losses = np.full(donor_count, np.nan)
    fold_records: list[dict[str, Any]] = []
    marker_counts: list[int] = []
    common_available = True

    for validation in range(donor_count):
        artifacts = _source_fold_artifacts(records, cognates, validation)
        training = artifacts["training"]
        training_labels = artifacts["training_labels"]
        selected = artifacts["selected"]
        selected_indices = artifacts["selected_indices"]
        rna_graph = artifacts["rna_graph"]
        adt_graph = artifacts["adt_graph"]
        product_laplacian = artifacts["product_laplacian"]
        destroyed_laplacian = artifacts["destroyed_laplacian"]
        marker_counts.append(len(selected_indices))
        training_tables = artifacts["training_tables"]
        shifted_tables = artifacts["shifted_tables"]
        truth = artifacts["truth"]
        fold = {
            "validation_sample": records[validation]["sample"],
            "training_samples": [record["sample"] for record in training],
            "selected_symbols": list(selected.symbols),
            "eligible_marker_count": selected.eligible_count,
            "destroyed_invariance": artifacts["destroyed_invariance"],
            "rna_graph_sha256": _array_sha256(rna_graph.adjacency.astype(np.uint8)),
            "adt_graph_sha256": _array_sha256(adt_graph.adjacency.astype(np.uint8)),
            "fit_certificates": {"primary": [], "destroyed": []},
        }

        for deviation in DEVIATION_GRID:
            for ridge in RIDGE_GRID:
                for graph_penalty in GRAPH_GRID:
                    base = ConditionalFieldConfig(deviation, ridge, graph_penalty, 1.0)
                    try:
                        fit = _fit_structured(
                            training_tables, training_labels, product_laplacian, base
                        )
                        fold["fit_certificates"]["primary"].append(
                            {
                                "base_configuration": _configuration_payload(base),
                                **_fit_certificate(fit),
                            }
                        )
                        for transport in TRANSPORT_GRID:
                            configuration = ConditionalFieldConfig(
                                deviation, ridge, graph_penalty, transport
                            )
                            primary_losses[configuration][validation] = float(
                                np.mean(
                                    entity_deviance(
                                        truth,
                                        _structured_prediction(
                                            fit, labels[validation], transport, truth
                                        ),
                                    )
                                )
                            )
                    except (ValueError, FloatingPointError, CouplingEstimationRefusal):
                        pass
                    try:
                        shifted_fit = _fit_structured(
                            shifted_tables,
                            training_labels,
                            destroyed_laplacian,
                            base,
                        )
                        fold["fit_certificates"]["destroyed"].append(
                            {
                                "base_configuration": _configuration_payload(base),
                                **_fit_certificate(shifted_fit),
                            }
                        )
                        for transport in TRANSPORT_GRID:
                            configuration = ConditionalFieldConfig(
                                deviation, ridge, graph_penalty, transport
                            )
                            destroyed_losses[configuration][validation] = float(
                                np.mean(
                                    entity_deviance(
                                        truth,
                                        _structured_prediction(
                                            shifted_fit,
                                            labels[validation],
                                            transport,
                                            truth,
                                        ),
                                    )
                                )
                            )
                    except (ValueError, FloatingPointError, CouplingEstimationRefusal):
                        pass

        try:
            pooled = fit_poisson_loglinear_interaction(training_tables)
            etiology_poisson = fit_poisson_loglinear_interaction(
                training_tables, np.asarray(training_labels, dtype=object)
            )
        except (ValueError, FloatingPointError, PoissonLoglinearRefusal) as error:
            raise ProtocolRefusal(
                "CLASSICAL_COMPARATOR_REFUSAL",
                {"fold": validation, "method": "poisson"},
            ) from error
        fold["poisson_certificates"] = {
            "pooled": _poisson_certificate(pooled),
            "etiology_specific": _poisson_certificate(etiology_poisson),
        }
        fold["poisson_validation_reconstruction"] = []
        rows, columns = _margins(truth)
        for transport in TRANSPORT_GRID:
            pooled_reconstruction = reconstruct_poisson_tables(
                _poisson_field(pooled, None),
                rows,
                columns,
                transport_scale=transport,
            )
            etiology_reconstruction = reconstruct_poisson_tables(
                _poisson_field(etiology_poisson, labels[validation]),
                rows,
                columns,
                transport_scale=transport,
            )
            pooled_losses[transport][validation] = float(
                np.mean(
                    entity_deviance(
                        truth,
                        pooled_reconstruction.table,
                    )
                )
            )
            etiology_poisson_losses[transport][validation] = float(
                np.mean(
                    entity_deviance(
                        truth,
                        etiology_reconstruction.table,
                    )
                )
            )
            fold["poisson_validation_reconstruction"].append(
                {
                    "transport_multiplier": transport,
                    "pooled": _poisson_reconstruction_certificate(
                        pooled_reconstruction
                    ),
                    "etiology_specific": _poisson_reconstruction_certificate(
                        etiology_reconstruction
                    ),
                }
            )

        pearson = fit_standardized_pearson(training_tables, training_labels)
        for transport in TRANSPORT_GRID:
            pearson_losses[transport][validation] = float(
                np.mean(
                    entity_deviance(
                        truth,
                        _pearson_prediction(
                            pearson, labels[validation], transport, truth
                        ),
                    )
                )
            )
        if common_available:
            try:
                common = fit_common_effect_conditional_log_odds(
                    training_tables, minimum_informative_donors=len(training_tables)
                )
                fold["common_effect_certificate"] = {
                    "converged": common.converged,
                    "objective": common.objective,
                    "gradient_norm": common.gradient_norm,
                    "scaled_gradient_norm": common.scaled_gradient_norm,
                    "support_count_sha256": _array_sha256(common.support_count),
                }
                for transport in TRANSPORT_GRID:
                    common_losses[transport][validation] = float(
                        np.mean(
                            entity_deviance(
                                truth,
                                _common_prediction(common.log_odds, transport, truth),
                            )
                        )
                    )
            except (ValueError, FloatingPointError, CouplingEstimationRefusal):
                common_available = False
        independence_losses[validation] = float(
            np.mean(entity_deviance(truth, _independence_prediction(truth)))
        )
        fold_records.append(fold)

    selected_primary, selected_primary_losses = select_cv_configuration(primary_losses)
    selected_graph_zero, selected_graph_zero_losses = select_cv_configuration(
        {
            configuration: values
            for configuration, values in primary_losses.items()
            if configuration.graph_penalty == 0.0
        }
    )
    selected_destroyed, selected_destroyed_losses = select_cv_configuration(
        destroyed_losses
    )
    pooled_transport, selected_pooled_losses = _select_transport(pooled_losses)
    etiology_poisson_transport, selected_etiology_poisson_losses = _select_transport(
        etiology_poisson_losses
    )
    pearson_transport, selected_pearson_losses = _select_transport(pearson_losses)
    common_transport: float | None = None
    selected_common_losses: np.ndarray | None = None
    if common_available:
        try:
            common_transport, selected_common_losses = _select_transport(common_losses)
        except ProtocolRefusal:
            common_available = False
            selected_common_losses = None
    real_final = _selection(records, cognates, destroyed=False)
    shifted_final = _selection(records, cognates, destroyed=True)
    final_invariance = _assert_destroyed_invariance(
        records, cognates, real_final, shifted_final
    )
    selected, rna_graph, adt_graph, product_laplacian = real_final
    _, shifted_rna_graph, shifted_adt_graph, shifted_laplacian = shifted_final
    marker_counts.append(len(selected.indices))
    selected_indices = np.asarray(selected.indices, dtype=np.int64)
    source_tables = _selected_tables(records, selected_indices, destroyed=False)
    shifted_tables = _selected_tables(records, selected_indices, destroyed=True)
    primary_fit = _fit_structured(
        source_tables, labels, product_laplacian, selected_primary
    )
    graph_zero_fit = _fit_structured(
        source_tables, labels, product_laplacian, selected_graph_zero
    )
    destroyed_fit = _fit_structured(
        shifted_tables, labels, shifted_laplacian, selected_destroyed
    )
    pooled_fit = fit_poisson_loglinear_interaction(source_tables)
    etiology_poisson_fit = fit_poisson_loglinear_interaction(
        source_tables, np.asarray(labels, dtype=object)
    )
    pearson_fit = fit_standardized_pearson(source_tables, labels)
    common_fit = None
    if common_available:
        try:
            common_fit = fit_common_effect_conditional_log_odds(
                source_tables, minimum_informative_donors=len(source_tables)
            )
        except (ValueError, FloatingPointError, CouplingEstimationRefusal):
            common_available = False
            common_transport = None
            selected_common_losses = None
    classical, selected_classical_losses = _select_final_classical(
        selected_pearson_losses,
        selected_common_losses,
        independence_losses,
        final_common_available=common_fit is not None,
    )

    comparator_losses = {
        "pooled_fixed_interaction_poisson": selected_pooled_losses,
        "etiology_specific_fixed_interaction_poisson": selected_etiology_poisson_losses,
        "strongest_remaining_classical_comparator": selected_classical_losses,
        "destroyed_links": selected_destroyed_losses,
    }
    source_gate = evaluate_source_gate(
        selected_primary_losses,
        comparator_losses,
        labels,
        marker_counts,
        all_reductions_and_fits_complete=True,
    )
    models: dict[str, Any] = {
        "primary": {
            "coefficient": _model_array(primary_fit.coefficient),
            "configuration": _configuration_payload(selected_primary),
            "fit_certificate": _fit_certificate(primary_fit),
        },
        "graph_zero": {
            "coefficient": _model_array(graph_zero_fit.coefficient),
            "configuration": _configuration_payload(selected_graph_zero),
            "fit_certificate": _fit_certificate(graph_zero_fit),
        },
        "destroyed_links": {
            "coefficient": _model_array(destroyed_fit.coefficient),
            "configuration": _configuration_payload(selected_destroyed),
            "fit_certificate": _fit_certificate(destroyed_fit),
        },
        "pooled_fixed_interaction_poisson": {
            "log_odds": _model_array(pooled_fit.log_odds[0]),
            "transport_multiplier": pooled_transport,
            "fit_certificate": _poisson_certificate(pooled_fit),
        },
        "etiology_specific_fixed_interaction_poisson": {
            "log_odds": _model_array(
                np.asarray(
                    [
                        etiology_poisson_fit.log_odds[
                            etiology_poisson_fit.group_labels.index(level)
                        ]
                        for level in ETIOLOGIES
                    ]
                )
            ),
            "context_levels": list(ETIOLOGIES),
            "transport_multiplier": etiology_poisson_transport,
            "fit_certificate": _poisson_certificate(etiology_poisson_fit),
        },
        "standardized_fixed_margin_pearson": {
            "coordinate": _model_array(pearson_fit),
            "context_levels": list(ETIOLOGIES),
            "transport_multiplier": pearson_transport,
        },
        "exact_common_effect_conditional_field": None,
        "fixed_margin_independence": {"kind": "recipient_fixed_margin_independence"},
    }
    if common_fit is not None and common_transport is not None:
        models["exact_common_effect_conditional_field"] = {
            "log_odds": _model_array(common_fit.log_odds),
            "transport_multiplier": common_transport,
            "fit_certificate": {
                "converged": common_fit.converged,
                "objective": common_fit.objective,
                "gradient_norm": common_fit.gradient_norm,
                "scaled_gradient_norm": common_fit.scaled_gradient_norm,
                "support_count_sha256": _array_sha256(common_fit.support_count),
            },
        }

    modules: dict[str, Any] = {}
    selected_symbols = tuple(selected.symbols)
    for name, members in evaluable_modules(selected_symbols).items():
        positions = np.asarray(
            [selected_symbols.index(symbol) for symbol in members], dtype=np.int64
        )
        module_tables = source_tables[:, positions][:, :, positions]
        shifted_module_tables = shifted_tables[:, positions][:, :, positions]
        rna_profiles = np.asarray([record["rna_profile"] for record in records])[
            :, selected_indices[positions]
        ]
        adt_profiles = np.asarray([record["adt_profile"] for record in records])[
            :, selected_indices[positions]
        ]
        shifted_adt_profiles = np.asarray(
            [record["destroyed_adt_profile"] for record in records]
        )[:, selected_indices[positions]]
        module_rna_graph = module_knn_graph(rna_profiles, members)
        module_adt_graph = module_knn_graph(adt_profiles, members)
        shifted_module_adt_graph = module_knn_graph(shifted_adt_profiles, members)
        if not np.array_equal(
            module_adt_graph.adjacency, shifted_module_adt_graph.adjacency
        ):
            raise ProtocolRefusal("DESTROYED_SELECTION_INVARIANCE_FAILURE")
        module_laplacian = protein_fast_product_laplacian(
            module_rna_graph.laplacian, module_adt_graph.laplacian
        )
        shifted_module_laplacian = protein_fast_product_laplacian(
            module_rna_graph.laplacian, shifted_module_adt_graph.laplacian
        )
        module_primary = _fit_structured(
            module_tables, labels, module_laplacian, selected_primary
        )
        module_destroyed = _fit_structured(
            shifted_module_tables, labels, shifted_module_laplacian, selected_destroyed
        )
        modules[name] = {
            "members": list(members),
            "positions": positions.tolist(),
            "graph_neighbors": min(3, len(members) - 1),
            "primary_coefficient": _model_array(module_primary.coefficient),
            "destroyed_coefficient": _model_array(module_destroyed.coefficient),
            "primary_fit_certificate": _fit_certificate(module_primary),
            "destroyed_fit_certificate": _fit_certificate(module_destroyed),
            "rna_graph": _graph_payload(module_rna_graph),
            "adt_graph": _graph_payload(module_adt_graph),
            "destroyed_adt_graph": _graph_payload(shifted_module_adt_graph),
            "classical_fields_reused_by_entitywise_invariance": True,
        }

    return {
        "selected_marker_source_indices": selected_indices.tolist(),
        "selected_symbols": list(selected.symbols),
        "selected_marker_axis_sha256": _axis_sha256(selected.symbols),
        "eligible_marker_count": selected.eligible_count,
        "selection_minimum_balance": list(selected.minimum_balance),
        "selection_median_balance": list(selected.median_balance),
        "graphs": {
            "rna": _graph_payload(rna_graph),
            "adt": _graph_payload(adt_graph),
            "destroyed_rna": _graph_payload(shifted_rna_graph),
            "destroyed_adt": _graph_payload(shifted_adt_graph),
            "product_laplacian_sha256": _array_sha256(product_laplacian),
            "destroyed_product_laplacian_sha256": _array_sha256(shifted_laplacian),
            "destroyed_invariance": final_invariance,
        },
        "source_cross_validation": {
            "folds": fold_records,
            "selected_primary_configuration": _configuration_payload(selected_primary),
            "selected_graph_zero_configuration": _configuration_payload(
                selected_graph_zero
            ),
            "selected_destroyed_configuration": _configuration_payload(
                selected_destroyed
            ),
            "selected_pooled_poisson_transport": pooled_transport,
            "selected_etiology_poisson_transport": etiology_poisson_transport,
            "selected_pearson_transport": pearson_transport,
            "selected_common_effect_transport": common_transport,
            "common_effect_available": common_fit is not None,
            "strongest_remaining_classical": asdict(classical),
            "selected_losses": {
                "primary": selected_primary_losses.tolist(),
                "graph_zero": selected_graph_zero_losses.tolist(),
                "destroyed_links": selected_destroyed_losses.tolist(),
                "pooled_fixed_interaction_poisson": selected_pooled_losses.tolist(),
                "etiology_specific_fixed_interaction_poisson": selected_etiology_poisson_losses.tolist(),
                "standardized_fixed_margin_pearson": selected_pearson_losses.tolist(),
                "exact_common_effect_conditional_field": (
                    selected_common_losses.tolist()
                    if selected_common_losses is not None
                    else None
                ),
                "fixed_margin_independence": independence_losses.tolist(),
                "strongest_remaining_classical_comparator": selected_classical_losses.tolist(),
            },
            "complete_loss_grids": {
                "primary": _complete_grid_payload(
                    primary_losses, _configuration_payload
                ),
                "destroyed_links": _complete_grid_payload(
                    destroyed_losses, _configuration_payload
                ),
                "pooled_fixed_interaction_poisson": _complete_grid_payload(
                    pooled_losses, float
                ),
                "etiology_specific_fixed_interaction_poisson": _complete_grid_payload(
                    etiology_poisson_losses, float
                ),
                "standardized_fixed_margin_pearson": _complete_grid_payload(
                    pearson_losses, float
                ),
                "exact_common_effect_conditional_field": _complete_grid_payload(
                    common_losses, float
                ),
            },
            "selection_unit": "physical-heart-equal mean multinomial deviance per cell",
            "marker_selection_scope": "repeated within every 13-heart training fold",
            "tie_break": "ascending mean loss then frozen parameter tuple",
        },
        "source_gate": source_gate,
        "models": models,
        "secondary_module_models": modules,
    }


def _attempt_bindings(tags: Mapping[str, str]) -> dict[str, Any]:
    return {
        "candidate_sha256": CANDIDATE_SHA256,
        "hardening_sha256": HARDENING_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_clarifications_sha256": IMPLEMENTATION_CLARIFICATIONS_SHA256,
        "cognate_axis_sha256": COGNATE_SHA256,
        "public_tags": dict(tags),
        "implementation_file_sha256": _sha256(IMPLEMENTATION_FREEZE),
        "implementation_bindings": _binding_hashes(),
    }


def _access_header(
    stage: str, attempt_created_at: str, runtime: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema": "gse217494-heart-access-journal/1.0",
        "stage": stage,
        "event": "OPENED_BEFORE_ASSAY_ACCESS",
        "created_at_utc": attempt_created_at,
        "runtime": dict(runtime),
        "one_streaming_get_per_matrix": True,
        "range_requests_permitted": False,
        "automatic_retries_permitted": False,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(
                    line,
                    parse_constant=lambda token: (_ for _ in ()).throw(
                        ValueError(token)
                    ),
                )
                if not isinstance(value, dict):
                    raise PermissionError("access journal row is not an object")
                rows.append(value)
    if not rows:
        raise PermissionError("access journal is empty")
    return rows


def _published_bytes(tag: str, path: Path) -> bytes:
    return _git("show", f"{tag}:{_relative(path)}", text=False).stdout


def _require_public_attempt_prefix(
    tag: str,
    attempt_path: Path,
    journal_path: Path,
    expected_header: Mapping[str, Any],
) -> str:
    commit = _require_public_tag(tag, (attempt_path,))
    published = _published_bytes(tag, journal_path)
    expected = _canonical_json_bytes(expected_header) + b"\n"
    if published != expected or not journal_path.read_bytes().startswith(expected):
        raise PermissionError(
            "public attempt tag does not bind the access-journal header"
        )
    return commit


def _validate_claim_token_hash(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PermissionError("public claim-token hash is invalid")
    return value


def _source_downstream_paths() -> tuple[Path, ...]:
    return (
        SOURCE_ATTEMPT,
        SOURCE_CONSUMPTION,
        SOURCE_ACCESS,
        SOURCE_RESULT,
        SCORE_AUTHORIZATION,
        SCORE_ATTEMPT,
        SCORE_CONSUMPTION,
        SCORE_ACCESS,
        SCORE_RESULT,
    )


def claim_source(
    *, claim_token: Path, scratch: Path = DEFAULT_SCRATCH
) -> dict[str, Any]:
    """Create and freeze the source attempt without issuing an assay GET."""

    if any(path.exists() for path in _source_downstream_paths()):
        raise FileExistsError("GSE217494 source campaign has already been claimed")
    _require_token_outside_scratch(claim_token, scratch)
    _prepare_scratch(scratch)
    runtime = _require_runtime()
    tags = _verify_implementation_freeze()
    contract = _contract()
    _cognates()
    claim_sha256 = _create_claim_token(claim_token)
    created = _timestamp()
    attempt = {
        "schema": "gse217494-heart-source-attempt/1.0",
        "status": "CLAIMED_BEFORE_FIRST_SOURCE_MATRIX_GET",
        "created_at_utc": created,
        "bindings": _attempt_bindings(tags),
        "source_samples": list(SOURCE_ORDER),
        "held_samples": list(HELD_ORDER),
        "source_matrix_bytes": int(
            sum(int(sample["matrix_bytes"]) for sample in contract["source"])
        ),
        "claim_token_sha256": claim_sha256,
        "runtime": runtime,
        "held_matrix_access_authorized": False,
        "rerun_permitted": False,
    }
    _write_json_x(SOURCE_ATTEMPT, attempt)
    _append_jsonl(
        SOURCE_ACCESS, _access_header("source", created, runtime), create=True
    )
    return attempt


def _validate_source_attempt(
    *, allow_advanced_journal: bool = False
) -> tuple[dict[str, Any], str]:
    tags = _verify_implementation_freeze()
    runtime = _require_runtime()
    attempt = _read_json(SOURCE_ATTEMPT)
    expected_header = _access_header(
        "source", str(attempt.get("created_at_utc")), runtime
    )
    attempt_commit = _require_public_attempt_prefix(
        SOURCE_ATTEMPT_TAG, SOURCE_ATTEMPT, SOURCE_ACCESS, expected_header
    )
    _require_ancestor(tags["implementation_commit"], attempt_commit)
    contract = _contract()
    if (
        attempt.get("schema") != "gse217494-heart-source-attempt/1.0"
        or attempt.get("status") != "CLAIMED_BEFORE_FIRST_SOURCE_MATRIX_GET"
        or attempt.get("bindings") != _attempt_bindings(tags)
        or attempt.get("source_samples") != list(SOURCE_ORDER)
        or attempt.get("held_samples") != list(HELD_ORDER)
        or attempt.get("source_matrix_bytes")
        != sum(int(sample["matrix_bytes"]) for sample in contract["source"])
        or attempt.get("runtime") != runtime
        or attempt.get("held_matrix_access_authorized") is not False
        or attempt.get("rerun_permitted") is not False
    ):
        raise PermissionError("public source attempt differs from the frozen contract")
    _validate_claim_token_hash(attempt.get("claim_token_sha256"))
    rows = _read_jsonl(SOURCE_ACCESS)
    if not allow_advanced_journal and rows != [expected_header]:
        raise PermissionError("source access began before the consuming run")
    return attempt, attempt_commit


def _consumption_payload(
    stage: str,
    attempt_path: Path,
    runtime: Mapping[str, Any],
    scratch: Path,
) -> dict[str, Any]:
    return {
        "schema": "gse217494-heart-consumption/1.0",
        "stage": stage,
        "status": "CONSUMED_EXCLUSIVELY_BEFORE_FIRST_MATRIX_GET",
        "created_at_utc": _timestamp(),
        "attempt_sha256": _sha256(attempt_path),
        "scratch_identity_sha256": _scratch_identity(scratch),
        "runtime": dict(runtime),
        "interruption_consumes_stage": True,
        "rerun_permitted": False,
    }


def _public_failure(
    *,
    stage: str,
    schema: str,
    error: BaseException,
    bindings: Mapping[str, Any],
    journal: Path,
) -> dict[str, Any]:
    if isinstance(error, ProtocolRefusal):
        code = error.code
        details = {
            key: value
            for key, value in error.details.items()
            if key
            in {"sample", "kind", "fold", "method", "observed_bytes", "status_code"}
            and isinstance(value, (str, int, float, bool, type(None)))
        }
    elif isinstance(error, (GzipMatrixMarketValidationError,)):
        code = "MATRIX_VALIDATION_FAILURE"
        details = {}
    elif isinstance(error, MarkerSupportRefusal):
        code = "MARKER_SUPPORT_REFUSAL"
        details = {}
    elif isinstance(error, NoCompleteConfigurationError):
        code = "NO_COMPLETE_SOURCE_CONFIGURATION"
        details = {}
    elif isinstance(error, (CouplingEstimationRefusal, PoissonLoglinearRefusal)):
        code = "NUMERICAL_REFUSAL"
        details = {}
    else:
        code = "UNEXPECTED_EXCEPTION"
        details = {}
    return {
        "schema": schema,
        "status": f"TERMINAL_{stage.upper()}_REFUSAL",
        "created_at_utc": _timestamp(),
        "refusal_code": code,
        "details": details,
        "bindings": dict(bindings),
        "access_journal_sha256": _sha256(journal),
        "held_matrix_requested": any(
            row.get("event") == "REQUEST_STARTED"
            and row.get("kind") == "matrix"
            and row.get("sample") in HELD_ORDER
            for row in _read_jsonl(journal)
        ),
        "rerun_permitted": False,
    }


def run_source(
    *, claim_token: Path | None, scratch: Path = DEFAULT_SCRATCH
) -> dict[str, Any]:
    """Run the consumed source stage under an interprocess exclusion lock."""

    with _stage_lock(scratch, "source"):
        return _run_source_locked(claim_token=claim_token, scratch=scratch)


def _run_source_locked(
    *, claim_token: Path | None, scratch: Path = DEFAULT_SCRATCH
) -> dict[str, Any]:
    """Consume the source capability, acquire 14 hearts, and decide promotion."""

    attempt, attempt_commit = _validate_source_attempt()
    if (
        SOURCE_RESULT.exists()
        or SOURCE_CONSUMPTION.exists()
        or any(
            path.exists()
            for path in (
                SCORE_AUTHORIZATION,
                SCORE_ATTEMPT,
                SCORE_CONSUMPTION,
                SCORE_ACCESS,
                SCORE_RESULT,
            )
        )
    ):
        raise FileExistsError("GSE217494 source campaign has already advanced")
    scratch_path = _prepare_scratch(scratch)
    token_path = _validate_token(claim_token, attempt["claim_token_sha256"])
    runtime = _require_runtime()
    consumption = _consumption_payload("source", SOURCE_ATTEMPT, runtime, scratch)
    _write_json_x(SOURCE_CONSUMPTION, consumption)
    _journal_event(
        SOURCE_ACCESS,
        "source",
        "CONSUMPTION_COMMITTED",
        consumption_sha256=_sha256(SOURCE_CONSUMPTION),
    )
    _consume_token(token_path)
    _journal_event(SOURCE_ACCESS, "source", "PRIVATE_CAPABILITY_CONSUMED")
    bindings = {
        **attempt["bindings"],
        "source_attempt_tag": SOURCE_ATTEMPT_TAG,
        "source_attempt_commit": attempt_commit,
        "source_attempt_sha256": _sha256(SOURCE_ATTEMPT),
        "source_consumption_sha256": _sha256(SOURCE_CONSUMPTION),
        "claim_token_sha256": attempt["claim_token_sha256"],
    }
    try:
        contract = _contract()
        cognates = _cognates()
        records = [
            _reduce_sample(
                sample,
                stage="source",
                scratch=scratch_path,
                journal=SOURCE_ACCESS,
                cognates=cognates,
            )
            for sample in contract["source"]
        ]
        if any(scratch_path.iterdir()):
            raise ProtocolRefusal("SCRATCH_STATE_FAILURE")
        fitted = _fit_source_models(records, cognates)
        access_audit = _success_access_audit(
            stage="source", journal=SOURCE_ACCESS, samples=SOURCE_ORDER
        )
        promoted = bool(fitted["source_gate"]["passes"])
        result = {
            "schema": "gse217494-heart-source-result/1.0",
            "status": (
                "SOURCE_PROMOTED" if promoted else "TERMINAL_SOURCE_PROMOTION_REFUSAL"
            ),
            "created_at_utc": _timestamp(),
            "bindings": bindings,
            "runtime": runtime,
            "source_records": [record["public_record"] for record in records],
            **fitted,
            "access_audit": access_audit,
            "held_matrix_requested": False,
            "held_matrix_access_authorized": False,
            "access_journal_sha256": _sha256(SOURCE_ACCESS),
            "rerun_permitted": False,
        }
    except BaseException as error:
        result = _public_failure(
            stage="source",
            schema="gse217494-heart-source-result/1.0",
            error=error,
            bindings=bindings,
            journal=SOURCE_ACCESS,
        )
    _write_json_x(SOURCE_RESULT, result)
    return result


def _recover(
    *,
    stage: str,
    scratch: Path,
    attempt_path: Path,
    consumption_path: Path,
    journal_path: Path,
    result_path: Path,
    schema: str,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    with _stage_lock(scratch, stage):
        return _recover_locked(
            stage=stage,
            scratch=scratch,
            attempt_path=attempt_path,
            consumption_path=consumption_path,
            journal_path=journal_path,
            result_path=result_path,
            schema=schema,
            bindings=bindings,
        )


def _recover_locked(
    *,
    stage: str,
    scratch: Path,
    attempt_path: Path,
    consumption_path: Path,
    journal_path: Path,
    result_path: Path,
    schema: str,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    if result_path.exists():
        raise FileExistsError(f"{stage} result already exists")
    if not consumption_path.exists():
        raise PermissionError(f"{stage} was not durably consumed")
    consumption = _read_json(consumption_path)
    if (
        consumption.get("schema") != "gse217494-heart-consumption/1.0"
        or consumption.get("stage") != stage
        or consumption.get("status")
        != "CONSUMED_EXCLUSIVELY_BEFORE_FIRST_MATRIX_GET"
        or consumption.get("attempt_sha256") != _sha256(attempt_path)
        or consumption.get("scratch_identity_sha256") != _scratch_identity(scratch)
        or consumption.get("rerun_permitted") is not False
    ):
        raise PermissionError(f"{stage} consumption or scratch identity differs")
    scratch_path = _outside_repository(scratch, "scratch directory")
    _journal_event(journal_path, stage, "CRASH_RECOVERY_STARTED")
    deleted_count = 0
    unexpected_count = 0
    if scratch_path.exists():
        for path in sorted(scratch_path.iterdir(), key=lambda value: value.name):
            if path.is_file() and path.name.startswith("GSE217494_"):
                path.unlink()
                deleted_count += 1
            else:
                unexpected_count += 1
        _fsync_directory(scratch_path)
    _journal_event(
        journal_path,
        stage,
        "CRASH_RECOVERY_FINISHED",
        deleted_campaign_file_count=deleted_count,
        unexpected_remaining_entry_count=unexpected_count,
    )
    if unexpected_count:
        refusal_code = "SCRATCH_STATE_FAILURE"
    else:
        refusal_code = "CRASH_RECOVERY"
    result = {
        "schema": schema,
        "status": f"TERMINAL_{stage.upper()}_CRASH_UNEVALUABLE",
        "created_at_utc": _timestamp(),
        "refusal_code": refusal_code,
        "bindings": dict(bindings),
        "attempt_sha256": _sha256(attempt_path),
        "consumption_sha256": _sha256(consumption_path),
        "access_journal_sha256": _sha256(journal_path),
        "recovered_without_remote_access": True,
        "rerun_permitted": False,
    }
    _write_json_x(result_path, result)
    return result


def recover_source(*, scratch: Path = DEFAULT_SCRATCH) -> dict[str, Any]:
    attempt, attempt_commit = _validate_source_attempt(allow_advanced_journal=True)
    bindings = {
        **attempt["bindings"],
        "source_attempt_tag": SOURCE_ATTEMPT_TAG,
        "source_attempt_commit": attempt_commit,
        "source_attempt_sha256": _sha256(SOURCE_ATTEMPT),
        "source_consumption_sha256": _sha256(SOURCE_CONSUMPTION),
        "claim_token_sha256": attempt["claim_token_sha256"],
    }
    return _recover(
        stage="source",
        scratch=scratch,
        attempt_path=SOURCE_ATTEMPT,
        consumption_path=SOURCE_CONSUMPTION,
        journal_path=SOURCE_ACCESS,
        result_path=SOURCE_RESULT,
        schema="gse217494-heart-source-result/1.0",
        bindings=bindings,
    )


def _success_access_audit(
    *, stage: str, journal: Path, samples: Sequence[str]
) -> dict[str, Any]:
    rows = _read_jsonl(journal)
    allowed_events = {
        "OPENED_BEFORE_ASSAY_ACCESS",
        "CONSUMPTION_COMMITTED",
        "PRIVATE_CAPABILITY_CONSUMED",
        "REQUEST_STARTED",
        "REQUEST_FINISHED",
        "REQUEST_FAILED",
        "MATRIX_PARSE_FINISHED",
        "DOWNLOAD_DELETED",
    }
    headers = [
        index
        for index, row in enumerate(rows)
        if row.get("event") == "OPENED_BEFORE_ASSAY_ACCESS"
    ]
    if headers != [0] or any(
        row.get("schema") != "gse217494-heart-access-journal/1.0"
        or row.get("stage") != stage
        or row.get("event") not in allowed_events
        for row in rows
    ):
        raise PermissionError("successful access journal has an invalid stage or event")
    header = rows[0]
    if (
        header.get("runtime") != REQUIRED_RUNTIME
        or header.get("one_streaming_get_per_matrix") is not True
        or header.get("range_requests_permitted") is not False
        or header.get("automatic_retries_permitted") is not False
    ):
        raise PermissionError("successful access journal header differs")
    if any(row.get("event") == "REQUEST_FAILED" for row in rows):
        raise PermissionError("successful access journal contains a failed request")
    consumption_indices = [
        index
        for index, row in enumerate(rows)
        if row.get("event") == "CONSUMPTION_COMMITTED"
    ]
    capability_indices = [
        index
        for index, row in enumerate(rows)
        if row.get("event") == "PRIVATE_CAPABILITY_CONSUMED"
    ]
    if len(consumption_indices) != 1 or len(capability_indices) != 1:
        raise PermissionError("successful access journal lacks exclusive consumption")
    expected_identities = [
        (sample, kind)
        for sample in samples
        for kind in ("features", "barcodes", "matrix")
    ]
    started = [
        (row.get("sample"), row.get("kind"), index)
        for index, row in enumerate(rows)
        if row.get("event") == "REQUEST_STARTED"
    ]
    if [(sample, kind) for sample, kind, _ in started] != expected_identities:
        raise PermissionError(
            "access journal request order differs from the frozen order"
        )
    if (
        sum(row.get("event") == "REQUEST_FINISHED" for row in rows)
        != len(expected_identities)
        or sum(row.get("event") == "DOWNLOAD_DELETED" for row in rows)
        != len(expected_identities)
        or sum(row.get("event") == "MATRIX_PARSE_FINISHED" for row in rows)
        != len(samples)
    ):
        raise PermissionError("access journal contains extra completion events")
    if not (
        consumption_indices[0]
        < capability_indices[0]
        < (started[0][2] if started else len(rows))
    ):
        raise PermissionError("consumption or private capability order differs")
    contract = _contract()
    sample_contract = {
        record["sample"]: record
        for record in (*contract["source"], *contract["held"])
    }
    if any(sample not in sample_contract for sample in samples):
        raise PermissionError("access journal sample is outside the frozen contract")
    for sample, kind, start_index in started:
        matching_finished = [
            index
            for index, row in enumerate(rows)
            if row.get("event") == "REQUEST_FINISHED"
            and row.get("sample") == sample
            and row.get("kind") == kind
        ]
        matching_deleted = [
            index
            for index, row in enumerate(rows)
            if row.get("event") == "DOWNLOAD_DELETED"
            and row.get("sample") == sample
            and row.get("kind") == kind
            and row.get("deleted") is True
        ]
        expected = sample_contract[str(sample)]
        if kind == "features":
            expected_bytes = int(expected["feature_axis"]["bytes"])
            expected_sha256 = str(expected["feature_axis"]["gzip_sha256"])
            suffix = "features.tsv.gz"
        elif kind == "barcodes":
            expected_bytes = int(expected["barcode_axis"]["bytes"])
            expected_sha256 = str(expected["barcode_axis"]["gzip_sha256"])
            suffix = "barcodes.tsv.gz"
        else:
            expected_bytes = int(expected["matrix_bytes"])
            expected_sha256 = None
            suffix = "matrix.mtx.gz"
        expected_filename = f"GSE217494_{sample}.{suffix}"
        started_record = rows[start_index]
        finished_record = rows[matching_finished[0]] if matching_finished else {}
        deleted_record = rows[matching_deleted[0]] if matching_deleted else {}
        observed_sha256 = finished_record.get("observed_sha256")
        valid_observed_sha256 = (
            isinstance(observed_sha256, str)
            and len(observed_sha256) == 64
            and all(character in "0123456789abcdef" for character in observed_sha256)
        )
        if (
            len(matching_finished) != 1
            or len(matching_deleted) != 1
            or not (start_index < matching_finished[0] < matching_deleted[0])
            or started_record.get("url") != expected["urls"][kind]
            or started_record.get("filename") != expected_filename
            or started_record.get("expected_bytes") != expected_bytes
            or started_record.get("expected_sha256") != expected_sha256
            or finished_record.get("filename") != expected_filename
            or finished_record.get("observed_bytes") != expected_bytes
            or finished_record.get("status_code") != 200
            or finished_record.get("response_url") != expected["urls"][kind]
            or not valid_observed_sha256
            or (
                expected_sha256 is not None
                and observed_sha256 != expected_sha256
            )
            or deleted_record.get("filename") != expected_filename
            or deleted_record.get("existed_before_delete") is not True
            or deleted_record.get("deleted") is not True
        ):
            raise PermissionError(
                "access journal request identity or certificate differs"
            )
        if (
            started_record.get("method") != "GET"
            or started_record.get("range_header") is not None
            or started_record.get("automatic_retry_count") != 0
            or started_record.get("streaming") is not True
        ):
            raise PermissionError("access journal records a Range request or retry")
        if kind == "matrix":
            parsed = [
                row
                for row in rows
                if row.get("event") == "MATRIX_PARSE_FINISHED"
                and row.get("sample") == sample
            ]
            if (
                len(parsed) != 1
                or parsed[0].get("kind") != "matrix"
                or not (
                    matching_finished[0]
                    < rows.index(parsed[0])
                    < matching_deleted[0]
                )
                or parsed[0].get("parser_audit", {}).get("gzip_stream_exhausted")
                is not True
                or parsed[0].get("parser_audit", {}).get("banner")
                != "%%MatrixMarket matrix coordinate integer general"
                or parsed[0].get("parser_audit", {}).get("matrix_shape")
                != [FEATURE_COUNT, int(expected["barcode_axis"]["count"])]
                or parsed[0].get("parser_audit", {}).get("declared_nnz")
                != parsed[0].get("parser_audit", {}).get("parsed_nnz")
                or not isinstance(
                    parsed[0].get("parser_audit", {}).get("decompressed_bytes"),
                    int,
                )
                or parsed[0].get("parser_audit", {}).get("decompressed_bytes", 0)
                <= 0
                or not isinstance(
                    parsed[0].get("parser_audit", {}).get("decompressed_sha256"),
                    str,
                )
                or len(
                    parsed[0].get("parser_audit", {}).get("decompressed_sha256", "")
                )
                != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in parsed[0]
                    .get("parser_audit", {})
                    .get("decompressed_sha256", "")
                )
                or parsed[0].get("independent_compressed_bytes")
                != finished_record.get("observed_bytes")
                or parsed[0].get("independent_compressed_sha256")
                != finished_record.get("observed_sha256")
            ):
                raise PermissionError(
                    "matrix compressed/decompressed certificates differ"
                )
    matrix_starts = {
        sample: index for sample, kind, index in started if kind == "matrix"
    }
    matrix_deletes = {
        row["sample"]: index
        for index, row in enumerate(rows)
        if row.get("event") == "DOWNLOAD_DELETED" and row.get("kind") == "matrix"
    }
    for previous, following in zip(samples, samples[1:]):
        if matrix_deletes[previous] >= matrix_starts[following]:
            raise PermissionError(
                "a matrix was not deleted before the next donor request"
            )
    return {
        "journal_sha256": _sha256(journal),
        "request_count": len(started),
        "matrix_streaming_get_count": len(samples),
        "range_request_count": 0,
        "automatic_retry_count": 0,
        "all_downloads_deleted": True,
    }


def _model_array_from(record: Mapping[str, Any], shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(record.get("values"), dtype=float)
    if (
        values.shape != shape
        or record.get("shape") != list(shape)
        or not np.isfinite(values).all()
        or record.get("sha256") != _array_sha256(values)
    ):
        raise PermissionError("published model field differs from its certificate")
    return values


def _source_field_manifest(source: Mapping[str, Any]) -> dict[str, Any]:
    models = source["models"]
    modules = source.get("secondary_module_models", {})
    return {
        "selected_marker_axis_sha256": source["selected_marker_axis_sha256"],
        "graphs_sha256": _canonical_json_sha256(source["graphs"]),
        "primary_coefficient_sha256": models["primary"]["coefficient"]["sha256"],
        "primary_configuration_sha256": _canonical_json_sha256(
            models["primary"]["configuration"]
        ),
        "graph_zero_coefficient_sha256": models["graph_zero"]["coefficient"]["sha256"],
        "destroyed_coefficient_sha256": models["destroyed_links"]["coefficient"][
            "sha256"
        ],
        "pooled_poisson_sha256": models["pooled_fixed_interaction_poisson"]["log_odds"][
            "sha256"
        ],
        "etiology_poisson_sha256": models[
            "etiology_specific_fixed_interaction_poisson"
        ]["log_odds"]["sha256"],
        "pearson_sha256": models["standardized_fixed_margin_pearson"]["coordinate"][
            "sha256"
        ],
        "common_effect_sha256": (
            models["exact_common_effect_conditional_field"]["log_odds"]["sha256"]
            if models["exact_common_effect_conditional_field"] is not None
            else None
        ),
        "strongest_remaining_classical": source["source_cross_validation"][
            "strongest_remaining_classical"
        ]["selected"],
        "module_fields_sha256": _canonical_json_sha256(modules),
        "implementation_bindings": _binding_hashes(),
        "runtime": source["runtime"],
    }


def _validate_source_result(
    *, require_public: bool = True, require_promoted: bool = True
) -> tuple[dict[str, Any], str]:
    attempt, attempt_commit = _validate_source_attempt(allow_advanced_journal=True)
    if require_public:
        commit = _require_public_tag(
            SOURCE_TAG,
            (SOURCE_ATTEMPT, SOURCE_CONSUMPTION, SOURCE_ACCESS, SOURCE_RESULT),
        )
        _require_ancestor(attempt_commit, commit)
    else:
        commit = _git("rev-parse", "HEAD").stdout.strip()
    source = _read_json(SOURCE_RESULT)
    expected_bindings = {
        **attempt["bindings"],
        "source_attempt_tag": SOURCE_ATTEMPT_TAG,
        "source_attempt_commit": attempt_commit,
        "source_attempt_sha256": _sha256(SOURCE_ATTEMPT),
        "source_consumption_sha256": _sha256(SOURCE_CONSUMPTION),
        "claim_token_sha256": attempt["claim_token_sha256"],
    }
    if (
        source.get("schema") != "gse217494-heart-source-result/1.0"
        or source.get("status")
        not in {"SOURCE_PROMOTED", "TERMINAL_SOURCE_PROMOTION_REFUSAL"}
        or source.get("bindings") != expected_bindings
        or source.get("runtime") != _require_runtime()
        or source.get("held_matrix_requested") is not False
        or source.get("held_matrix_access_authorized") is not False
        or source.get("rerun_permitted") is not False
        or source.get("access_journal_sha256") != _sha256(SOURCE_ACCESS)
    ):
        raise PermissionError(
            "public source result is not a promoted frozen field artifact"
        )
    access = _success_access_audit(
        stage="source", journal=SOURCE_ACCESS, samples=SOURCE_ORDER
    )
    if source.get("access_audit") != access or access[
        "matrix_streaming_get_count"
    ] != len(SOURCE_ORDER):
        raise PermissionError("public source matrix access count differs")
    symbols = source.get("selected_symbols")
    indices = source.get("selected_marker_source_indices")
    if (
        not isinstance(symbols, list)
        or not isinstance(indices, list)
        or not (9 <= len(symbols) <= 12)
        or len(indices) != len(symbols)
        or len(set(symbols)) != len(symbols)
        or source.get("selected_marker_axis_sha256") != _axis_sha256(symbols)
    ):
        raise PermissionError("public source marker axis differs")
    count = len(symbols)
    models = source.get("models")
    required_models = {
        "primary",
        "graph_zero",
        "destroyed_links",
        "pooled_fixed_interaction_poisson",
        "etiology_specific_fixed_interaction_poisson",
        "standardized_fixed_margin_pearson",
        "exact_common_effect_conditional_field",
        "fixed_margin_independence",
    }
    if not isinstance(models, dict) or set(models) != required_models:
        raise PermissionError("public source models are incomplete")
    _model_array_from(models["primary"]["coefficient"], (4, count, count))
    _model_array_from(models["graph_zero"]["coefficient"], (4, count, count))
    _model_array_from(models["destroyed_links"]["coefficient"], (4, count, count))
    _model_array_from(
        models["pooled_fixed_interaction_poisson"]["log_odds"], (count, count)
    )
    _model_array_from(
        models["etiology_specific_fixed_interaction_poisson"]["log_odds"],
        (4, count, count),
    )
    _model_array_from(
        models["standardized_fixed_margin_pearson"]["coordinate"],
        (4, count, count),
    )
    common = models["exact_common_effect_conditional_field"]
    if common is not None:
        _model_array_from(common["log_odds"], (count, count))
    cross_validation = source.get("source_cross_validation", {})
    losses = cross_validation.get("selected_losses", {})
    labels = [sample["etiology"] for sample in _contract()["source"]]
    primary_losses = np.asarray(losses.get("primary"), dtype=float)
    strongest_losses = np.asarray(
        losses.get("strongest_remaining_classical_comparator"), dtype=float
    )
    comparator_losses = {
        "pooled_fixed_interaction_poisson": np.asarray(
            losses.get("pooled_fixed_interaction_poisson"), dtype=float
        ),
        "etiology_specific_fixed_interaction_poisson": np.asarray(
            losses.get("etiology_specific_fixed_interaction_poisson"), dtype=float
        ),
        "strongest_remaining_classical_comparator": strongest_losses,
        "destroyed_links": np.asarray(losses.get("destroyed_links"), dtype=float),
    }
    folds = cross_validation.get("folds", [])
    marker_counts = [len(fold.get("selected_symbols", [])) for fold in folds] + [count]
    expected_gate = evaluate_source_gate(
        primary_losses,
        comparator_losses,
        labels,
        marker_counts,
        all_reductions_and_fits_complete=True,
    )
    if (
        source.get("source_gate") != expected_gate
        or source.get("status")
        != (
            "SOURCE_PROMOTED"
            if expected_gate["passes"]
            else "TERMINAL_SOURCE_PROMOTION_REFUSAL"
        )
        or (require_promoted and expected_gate["passes"] is not True)
    ):
        raise PermissionError("public source promotion arithmetic does not reproduce")
    strongest = cross_validation.get("strongest_remaining_classical", {})
    if strongest.get("selected") not in CLASSICAL_COMPARATORS:
        raise PermissionError("public strongest classical comparator differs")
    modules = source.get("secondary_module_models", {})
    if not isinstance(modules, dict):
        raise PermissionError("public secondary module fields differ")
    for module in modules.values():
        members = module.get("members", [])
        module_count = len(members)
        if module_count < 3 or module.get("graph_neighbors") != min(
            3, module_count - 1
        ):
            raise PermissionError("public module graph contract differs")
        _model_array_from(
            module["primary_coefficient"], (4, module_count, module_count)
        )
        _model_array_from(
            module["destroyed_coefficient"], (4, module_count, module_count)
        )
        if module.get("classical_fields_reused_by_entitywise_invariance") is not True:
            raise PermissionError("public module comparator semantics differ")
    _source_field_manifest(source)
    return source, commit


def _validate_terminal_result(stage: str) -> tuple[dict[str, Any], str]:
    if stage == "source":
        attempt, attempt_commit = _validate_source_attempt(
            allow_advanced_journal=True
        )
        tag = SOURCE_TAG
        paths = (SOURCE_ATTEMPT, SOURCE_CONSUMPTION, SOURCE_ACCESS, SOURCE_RESULT)
        result_path = SOURCE_RESULT
        journal = SOURCE_ACCESS
        schema = "gse217494-heart-source-result/1.0"
        statuses = {
            "TERMINAL_SOURCE_REFUSAL",
            "TERMINAL_SOURCE_CRASH_UNEVALUABLE",
        }
        expected_bindings = {
            **attempt["bindings"],
            "source_attempt_tag": SOURCE_ATTEMPT_TAG,
            "source_attempt_commit": attempt_commit,
            "source_attempt_sha256": _sha256(SOURCE_ATTEMPT),
            "source_consumption_sha256": _sha256(SOURCE_CONSUMPTION),
            "claim_token_sha256": attempt["claim_token_sha256"],
        }
    elif stage == "score":
        attempt, _, attempt_commit = _validate_score_attempt(
            allow_advanced_journal=True
        )
        tag = SCORE_TAG
        paths = (SCORE_ATTEMPT, SCORE_CONSUMPTION, SCORE_ACCESS, SCORE_RESULT)
        result_path = SCORE_RESULT
        journal = SCORE_ACCESS
        schema = "gse217494-heart-confirmation-result/1.0"
        statuses = {
            "TERMINAL_SCORE_REFUSAL",
            "TERMINAL_SCORE_CRASH_UNEVALUABLE",
        }
        expected_bindings = {
            **attempt["bindings"],
            "score_attempt_tag": SCORE_ATTEMPT_TAG,
            "score_attempt_commit": attempt_commit,
            "score_attempt_sha256": _sha256(SCORE_ATTEMPT),
            "score_consumption_sha256": _sha256(SCORE_CONSUMPTION),
            "claim_token_sha256": attempt["claim_token_sha256"],
        }
    else:
        raise ValueError("terminal validation requires source or score")
    commit = _require_public_tag(tag, paths)
    _require_ancestor(attempt_commit, commit)
    result = _read_json(result_path)
    rows = _read_jsonl(journal)
    held_requested = any(
        row.get("event") == "REQUEST_STARTED"
        and row.get("kind") == "matrix"
        and row.get("sample") in HELD_ORDER
        for row in rows
    )
    if (
        result.get("schema") != schema
        or result.get("status") not in statuses
        or result.get("bindings") != expected_bindings
        or result.get("access_journal_sha256") != _sha256(journal)
        or result.get("held_matrix_requested") is not held_requested
        or result.get("refusal_code") not in TERMINAL_REFUSAL_CODES
        or result.get("rerun_permitted") is not False
    ):
        raise PermissionError("public terminal result differs from the frozen stage")
    if result["status"].endswith("CRASH_UNEVALUABLE"):
        if (
            result.get("recovered_without_remote_access") is not True
            or result.get("attempt_sha256")
            != _sha256(SOURCE_ATTEMPT if stage == "source" else SCORE_ATTEMPT)
            or result.get("consumption_sha256")
            != _sha256(
                SOURCE_CONSUMPTION if stage == "source" else SCORE_CONSUMPTION
            )
        ):
            raise PermissionError("public crash recovery certificate differs")
    elif not isinstance(result.get("details"), dict):
        raise PermissionError("public terminal refusal details differ")
    return result, commit


def _validate_source_outcome() -> tuple[dict[str, Any], str]:
    status = _read_json(SOURCE_RESULT).get("status")
    if status in {"SOURCE_PROMOTED", "TERMINAL_SOURCE_PROMOTION_REFUSAL"}:
        return _validate_source_result(
            require_public=True, require_promoted=False
        )
    return _validate_terminal_result("source")


def authorize_score() -> dict[str, Any]:
    """Bind the public source fields before any held matrix is opened."""

    source, source_commit = _validate_source_result(require_public=True)
    if any(
        path.exists()
        for path in (
            SCORE_AUTHORIZATION,
            SCORE_ATTEMPT,
            SCORE_CONSUMPTION,
            SCORE_ACCESS,
            SCORE_RESULT,
        )
    ):
        raise FileExistsError("GSE217494 held score has already advanced")
    authorization = {
        "schema": "gse217494-heart-score-authorization/1.0",
        "status": "AUTHORIZED_AFTER_PUBLIC_SOURCE_FIELDS",
        "created_at_utc": _timestamp(),
        "source_tag": SOURCE_TAG,
        "source_commit": source_commit,
        "source_sha256": _sha256(SOURCE_RESULT),
        "source_field_manifest": _source_field_manifest(source),
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_clarifications_sha256": IMPLEMENTATION_CLARIFICATIONS_SHA256,
        "held_samples": list(HELD_ORDER),
        "held_matrix_access_authorized": True,
        "rerun_permitted": False,
    }
    _write_json_x(SCORE_AUTHORIZATION, authorization)
    return authorization


def _validate_score_authorization(
    source: Mapping[str, Any], source_commit: str
) -> tuple[dict[str, Any], str]:
    commit = _require_public_tag(
        SCORE_AUTHORIZATION_TAG, (SOURCE_RESULT, SCORE_AUTHORIZATION)
    )
    _require_ancestor(source_commit, commit)
    value = _read_json(SCORE_AUTHORIZATION)
    if (
        value.get("schema") != "gse217494-heart-score-authorization/1.0"
        or value.get("status") != "AUTHORIZED_AFTER_PUBLIC_SOURCE_FIELDS"
        or value.get("source_tag") != SOURCE_TAG
        or value.get("source_commit") != source_commit
        or value.get("source_sha256") != _sha256(SOURCE_RESULT)
        or value.get("source_field_manifest") != _source_field_manifest(source)
        or value.get("protocol_sha256") != PROTOCOL_SHA256
        or value.get("implementation_clarifications_sha256")
        != IMPLEMENTATION_CLARIFICATIONS_SHA256
        or value.get("held_samples") != list(HELD_ORDER)
        or value.get("held_matrix_access_authorized") is not True
        or value.get("rerun_permitted") is not False
    ):
        raise PermissionError("public score authorization differs from source fields")
    return value, commit


def claim_score(
    *, claim_token: Path, scratch: Path = DEFAULT_SCRATCH
) -> dict[str, Any]:
    """Create the held score attempt without issuing a held assay GET."""

    source, source_commit = _validate_source_result(require_public=True)
    authorization, authorization_commit = _validate_score_authorization(
        source, source_commit
    )
    if any(
        path.exists()
        for path in (SCORE_ATTEMPT, SCORE_CONSUMPTION, SCORE_ACCESS, SCORE_RESULT)
    ):
        raise FileExistsError("GSE217494 held score has already been claimed")
    _require_token_outside_scratch(claim_token, scratch)
    _prepare_scratch(scratch)
    runtime = _require_runtime()
    claim_sha256 = _create_claim_token(claim_token)
    created = _timestamp()
    bindings = {
        "source_tag": SOURCE_TAG,
        "source_commit": source_commit,
        "source_sha256": _sha256(SOURCE_RESULT),
        "source_field_manifest": _source_field_manifest(source),
        "score_authorization_tag": SCORE_AUTHORIZATION_TAG,
        "score_authorization_commit": authorization_commit,
        "score_authorization_sha256": _sha256(SCORE_AUTHORIZATION),
        "implementation_bindings": _binding_hashes(),
    }
    attempt = {
        "schema": "gse217494-heart-score-attempt/1.0",
        "status": "CLAIMED_AFTER_AUTHORIZATION_BEFORE_FIRST_HELD_MATRIX_GET",
        "created_at_utc": created,
        "bindings": bindings,
        "held_samples": list(HELD_ORDER),
        "claim_token_sha256": claim_sha256,
        "runtime": runtime,
        "held_matrix_access_authorized": authorization["held_matrix_access_authorized"],
        "rerun_permitted": False,
    }
    _write_json_x(SCORE_ATTEMPT, attempt)
    _append_jsonl(SCORE_ACCESS, _access_header("score", created, runtime), create=True)
    return attempt


def _validate_score_attempt(
    *, allow_advanced_journal: bool = False
) -> tuple[dict[str, Any], dict[str, Any], str]:
    source, source_commit = _validate_source_result(require_public=True)
    _, authorization_commit = _validate_score_authorization(source, source_commit)
    runtime = _require_runtime()
    attempt = _read_json(SCORE_ATTEMPT)
    expected_header = _access_header(
        "score", str(attempt.get("created_at_utc")), runtime
    )
    attempt_commit = _require_public_attempt_prefix(
        SCORE_ATTEMPT_TAG, SCORE_ATTEMPT, SCORE_ACCESS, expected_header
    )
    _require_ancestor(authorization_commit, attempt_commit)
    expected_bindings = {
        "source_tag": SOURCE_TAG,
        "source_commit": source_commit,
        "source_sha256": _sha256(SOURCE_RESULT),
        "source_field_manifest": _source_field_manifest(source),
        "score_authorization_tag": SCORE_AUTHORIZATION_TAG,
        "score_authorization_commit": authorization_commit,
        "score_authorization_sha256": _sha256(SCORE_AUTHORIZATION),
        "implementation_bindings": _binding_hashes(),
    }
    if (
        attempt.get("schema") != "gse217494-heart-score-attempt/1.0"
        or attempt.get("status")
        != "CLAIMED_AFTER_AUTHORIZATION_BEFORE_FIRST_HELD_MATRIX_GET"
        or attempt.get("bindings") != expected_bindings
        or attempt.get("held_samples") != list(HELD_ORDER)
        or attempt.get("runtime") != runtime
        or attempt.get("held_matrix_access_authorized") is not True
        or attempt.get("rerun_permitted") is not False
    ):
        raise PermissionError("public score attempt differs from authorized fields")
    _validate_claim_token_hash(attempt.get("claim_token_sha256"))
    rows = _read_jsonl(SCORE_ACCESS)
    if not allow_advanced_journal and rows != [expected_header]:
        raise PermissionError("held access began before the consuming score run")
    return attempt, source, attempt_commit


def _model_predictions(
    source: Mapping[str, Any], record: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    truth = np.asarray(record["tables"], dtype=np.int64)
    count = truth.shape[0]
    etiology = str(record["etiology"])
    rows, columns = _margins(truth)
    models = source["models"]
    predictions: dict[str, np.ndarray] = {}
    primary_coefficient = _model_array_from(
        models["primary"]["coefficient"], (4, count, count)
    )
    graph_zero_coefficient = _model_array_from(
        models["graph_zero"]["coefficient"], (4, count, count)
    )
    destroyed_coefficient = _model_array_from(
        models["destroyed_links"]["coefficient"], (4, count, count)
    )
    for name, coefficient in (
        ("primary", primary_coefficient),
        ("graph_zero", graph_zero_coefficient),
        ("destroyed_links", destroyed_coefficient),
    ):
        transport = float(models[name]["configuration"]["transport_multiplier"])
        field = context_log_odds(
            coefficient, (etiology,), transport_multiplier=transport
        )[0]
        predictions[name] = predict_conditional_tables(field, rows, columns)

    pooled_field = _model_array_from(
        models["pooled_fixed_interaction_poisson"]["log_odds"], (count, count)
    )
    pooled_reconstruction = reconstruct_poisson_tables(
        pooled_field,
        rows,
        columns,
        transport_scale=float(
            models["pooled_fixed_interaction_poisson"]["transport_multiplier"]
        ),
    )
    predictions["pooled_fixed_interaction_poisson"] = pooled_reconstruction.table
    etiology_fields = _model_array_from(
        models["etiology_specific_fixed_interaction_poisson"]["log_odds"],
        (4, count, count),
    )
    etiology_index = ETIOLOGIES.index(etiology)
    etiology_reconstruction = reconstruct_poisson_tables(
        etiology_fields[etiology_index],
        rows,
        columns,
        transport_scale=float(
            models["etiology_specific_fixed_interaction_poisson"][
                "transport_multiplier"
            ]
        ),
    )
    predictions["etiology_specific_fixed_interaction_poisson"] = (
        etiology_reconstruction.table
    )
    pearson = _model_array_from(
        models["standardized_fixed_margin_pearson"]["coordinate"],
        (4, count, count),
    )
    predictions["standardized_fixed_margin_pearson"] = predict_standardized_pearson(
        pearson[etiology_index],
        rows,
        columns,
        transport_multiplier=float(
            models["standardized_fixed_margin_pearson"]["transport_multiplier"]
        ),
    )
    common = models["exact_common_effect_conditional_field"]
    if common is not None:
        common_field = _model_array_from(common["log_odds"], (count, count))
        predictions["exact_common_effect_conditional_field"] = (
            predict_conditional_tables(
                float(common["transport_multiplier"]) * common_field, rows, columns
            )
        )
    predictions["fixed_margin_independence"] = predict_standardized_pearson(
        np.zeros((count, count), dtype=float), rows, columns
    )
    diagnostics = {
        "pooled_fixed_interaction_poisson": _poisson_reconstruction_certificate(
            pooled_reconstruction
        ),
        "etiology_specific_fixed_interaction_poisson": _poisson_reconstruction_certificate(
            etiology_reconstruction
        ),
    }
    return predictions, diagnostics


def _fixed_margin_standardized_field(tables: np.ndarray) -> np.ndarray:
    values = np.asarray(tables, dtype=float)
    rna_positive = values[..., 1, :].sum(axis=-1)
    adt_high = values[..., :, 1].sum(axis=-1)
    observed = values[..., 1, 1]
    expected = rna_positive * adt_high / CELL_BUDGET
    variance = (
        rna_positive
        * adt_high
        * (CELL_BUDGET - rna_positive)
        * (CELL_BUDGET - adt_high)
        / (CELL_BUDGET**2 * (CELL_BUDGET - 1))
    )
    return np.divide(
        observed - expected,
        np.sqrt(variance),
        out=np.zeros_like(expected),
        where=variance > 0.0,
    )


def _neighbor_overlap(
    truth_fields: Sequence[np.ndarray],
    predicted_fields: Sequence[np.ndarray],
    symbols: Sequence[str],
) -> dict[str, Any]:
    result = dict(
        neighbor_overlap_permutation(
            np.asarray(predicted_fields, dtype=float),
            np.asarray(truth_fields, dtype=float),
            symbols,
            neighbors=3,
            permutations=NEIGHBOR_PERMUTATIONS,
            seed=NEIGHBOR_SEED,
        )
    )
    result["exploratory"] = True
    return result


def _module_results(
    module_losses: Mapping[str, Mapping[str, np.ndarray]],
    labels: Sequence[str],
) -> dict[str, Any]:
    pending: list[tuple[str, str, dict[str, Any]]] = []
    output: dict[str, Any] = {}
    for module_name in sorted(module_losses):
        losses = module_losses[module_name]
        primary = np.asarray(losses["primary"], dtype=float)
        comparisons: dict[str, Any] = {}
        for comparator in MANDATORY_COMPARATORS:
            comparator_values = np.asarray(losses[comparator], dtype=float)
            difference = primary - comparator_values
            permutation = exact_paired_sign_permutation(difference)
            bootstrap = stratified_paired_bootstrap(difference, labels)
            record = {
                "primary_mean_deviance": float(primary.mean()),
                "comparator_mean_deviance": float(comparator_values.mean()),
                "primary_minus_comparator_mean": float(difference.mean()),
                "stratified_paired_bootstrap_95_interval": list(bootstrap["interval"]),
                "bootstrap_draws": bootstrap["draws"],
                "bootstrap_seed": bootstrap["seed"],
                "exact_sign_permutation_assignments": permutation["assignments"],
                "exact_one_sided_p": permutation["one_sided_p"],
                "benjamini_hochberg_q": None,
            }
            comparisons[comparator] = record
            pending.append((module_name, comparator, record))
        output[module_name] = {
            "donor_losses": {
                method: np.asarray(values, dtype=float).tolist()
                for method, values in losses.items()
            },
            "comparisons": comparisons,
        }
    if pending:
        adjusted = benjamini_hochberg(
            np.asarray([record["exact_one_sided_p"] for _, _, record in pending])
        )
        for (_, _, record), q_value in zip(pending, adjusted):
            record["benjamini_hochberg_q"] = float(q_value)
    return output


def run_score(
    *, claim_token: Path | None, scratch: Path = DEFAULT_SCRATCH
) -> dict[str, Any]:
    """Run held scoring under the same exclusion used by crash recovery."""

    with _stage_lock(scratch, "score"):
        return _run_score_locked(claim_token=claim_token, scratch=scratch)


def _run_score_locked(
    *, claim_token: Path | None, scratch: Path = DEFAULT_SCRATCH
) -> dict[str, Any]:
    """Consume held authorization, acquire eight hearts once, and score fields."""

    attempt, source, attempt_commit = _validate_score_attempt()
    if SCORE_RESULT.exists() or SCORE_CONSUMPTION.exists():
        raise FileExistsError("GSE217494 held score has already advanced")
    scratch_path = _prepare_scratch(scratch)
    token_path = _validate_token(claim_token, attempt["claim_token_sha256"])
    runtime = _require_runtime()
    consumption = _consumption_payload("score", SCORE_ATTEMPT, runtime, scratch)
    _write_json_x(SCORE_CONSUMPTION, consumption)
    _journal_event(
        SCORE_ACCESS,
        "score",
        "CONSUMPTION_COMMITTED",
        consumption_sha256=_sha256(SCORE_CONSUMPTION),
    )
    _consume_token(token_path)
    _journal_event(SCORE_ACCESS, "score", "PRIVATE_CAPABILITY_CONSUMED")
    bindings = {
        **attempt["bindings"],
        "score_attempt_tag": SCORE_ATTEMPT_TAG,
        "score_attempt_commit": attempt_commit,
        "score_attempt_sha256": _sha256(SCORE_ATTEMPT),
        "score_consumption_sha256": _sha256(SCORE_CONSUMPTION),
        "claim_token_sha256": attempt["claim_token_sha256"],
    }
    try:
        contract = _contract()
        all_cognates = _cognates()
        selected_indices = np.asarray(
            source["selected_marker_source_indices"], dtype=np.int64
        )
        selected_cognates = [all_cognates[index] for index in selected_indices]
        records = [
            _reduce_sample(
                sample,
                stage="score",
                scratch=scratch_path,
                journal=SCORE_ACCESS,
                cognates=selected_cognates,
            )
            for sample in contract["held"]
        ]
        if any(scratch_path.iterdir()):
            raise ProtocolRefusal("SCRATCH_STATE_FAILURE")
        method_losses: dict[str, np.ndarray] = {}
        prediction_diagnostics: list[dict[str, Any]] = []
        truth_hashes: list[dict[str, Any]] = []
        labels = [str(record["etiology"]) for record in records]
        per_donor_predictions: list[dict[str, np.ndarray]] = []
        for donor, record in enumerate(records):
            truth = np.asarray(record["tables"], dtype=np.int64)
            predictions, diagnostics = _model_predictions(source, record)
            per_donor_predictions.append(predictions)
            prediction_diagnostics.append(
                {
                    "sample": record["sample"],
                    "poisson_reconstruction": diagnostics,
                }
            )
            truth_hashes.append(
                {
                    "sample": record["sample"],
                    "truth_table_shape": list(truth.shape),
                    "truth_table_sha256": _array_sha256(truth),
                }
            )
            for method, prediction in predictions.items():
                method_losses.setdefault(method, np.full(len(records), np.nan))[
                    donor
                ] = float(np.mean(entity_deviance(truth, prediction)))
        strongest_name = source["source_cross_validation"][
            "strongest_remaining_classical"
        ]["selected"]
        if strongest_name not in method_losses:
            raise ProtocolRefusal(
                "CLASSICAL_COMPARATOR_REFUSAL", {"method": strongest_name}
            )
        mandatory = {
            "pooled_fixed_interaction_poisson": method_losses[
                "pooled_fixed_interaction_poisson"
            ],
            "etiology_specific_fixed_interaction_poisson": method_losses[
                "etiology_specific_fixed_interaction_poisson"
            ],
            "strongest_remaining_classical_comparator": method_losses[strongest_name],
            "destroyed_links": method_losses["destroyed_links"],
        }
        held_gate = evaluate_held_gate(method_losses["primary"], mandatory, labels)
        graph_difference = method_losses["primary"] - method_losses["graph_zero"]
        graph_bootstrap = stratified_paired_bootstrap(graph_difference, labels)
        primary_configuration = source["models"]["primary"]["configuration"]
        source_losses = source["source_cross_validation"]["selected_losses"]
        graph_gain = {
            "selected_graph_penalty_positive": primary_configuration["graph_penalty"]
            > 0.0,
            "source_cv_primary_below_graph_zero": float(
                np.mean(source_losses["primary"])
            )
            < float(np.mean(source_losses["graph_zero"])),
            "held_bootstrap_upper_97_5_below_zero": graph_bootstrap["interval"][1]
            < 0.0,
            "primary_minus_graph_zero_mean": float(graph_difference.mean()),
            "stratified_paired_bootstrap_95_interval": list(
                graph_bootstrap["interval"]
            ),
            "bootstrap_draws": graph_bootstrap["draws"],
            "bootstrap_seed": graph_bootstrap["seed"],
        }
        graph_gain["claimed"] = bool(
            graph_gain["selected_graph_penalty_positive"]
            and graph_gain["source_cv_primary_below_graph_zero"]
            and graph_gain["held_bootstrap_upper_97_5_below_zero"]
        )

        modules = source.get("secondary_module_models", {})
        module_losses: dict[str, dict[str, np.ndarray]] = {}
        selected_symbols = list(source["selected_symbols"])
        for module_name, module in modules.items():
            positions = np.asarray(module["positions"], dtype=np.int64)
            module_count = len(positions)
            primary_coefficient = _model_array_from(
                module["primary_coefficient"], (4, module_count, module_count)
            )
            destroyed_coefficient = _model_array_from(
                module["destroyed_coefficient"], (4, module_count, module_count)
            )
            losses = {
                "primary": np.full(len(records), np.nan),
                **{
                    comparator: np.full(len(records), np.nan)
                    for comparator in MANDATORY_COMPARATORS
                },
            }
            for donor, (record, predictions) in enumerate(
                zip(records, per_donor_predictions)
            ):
                truth = np.asarray(record["tables"], dtype=np.int64)[positions][
                    :, positions
                ]
                rows, columns = _margins(truth)
                etiology = str(record["etiology"])
                primary_field = context_log_odds(
                    primary_coefficient,
                    (etiology,),
                    transport_multiplier=float(
                        source["models"]["primary"]["configuration"][
                            "transport_multiplier"
                        ]
                    ),
                )[0]
                destroyed_field = context_log_odds(
                    destroyed_coefficient,
                    (etiology,),
                    transport_multiplier=float(
                        source["models"]["destroyed_links"]["configuration"][
                            "transport_multiplier"
                        ]
                    ),
                )[0]
                losses["primary"][donor] = float(
                    np.mean(
                        entity_deviance(
                            truth,
                            predict_conditional_tables(primary_field, rows, columns),
                        )
                    )
                )
                subset_predictions = {
                    "destroyed_links": predict_conditional_tables(
                        destroyed_field, rows, columns
                    ),
                    "pooled_fixed_interaction_poisson": predictions[
                        "pooled_fixed_interaction_poisson"
                    ][positions][:, positions],
                    "etiology_specific_fixed_interaction_poisson": predictions[
                        "etiology_specific_fixed_interaction_poisson"
                    ][positions][:, positions],
                    "strongest_remaining_classical_comparator": predictions[
                        strongest_name
                    ][positions][:, positions],
                }
                for comparator, prediction in subset_predictions.items():
                    losses[comparator][donor] = float(
                        np.mean(entity_deviance(truth, prediction))
                    )
            module_losses[module_name] = losses
        secondary = _module_results(module_losses, labels)
        non_evaluable = sorted(
            set(("endothelial", "fibroblast_fibrosis", "myeloid")) - set(modules)
        )
        truth_fields = [
            _fixed_margin_standardized_field(np.asarray(record["tables"]))
            for record in records
        ]
        predicted_fields = [
            _fixed_margin_standardized_field(predictions["primary"])
            for predictions in per_donor_predictions
        ]
        relational = _neighbor_overlap(truth_fields, predicted_fields, selected_symbols)
        access_audit = _success_access_audit(
            stage="score", journal=SCORE_ACCESS, samples=HELD_ORDER
        )
        passes = bool(held_gate["passes"])
        result = {
            "schema": "gse217494-heart-confirmation-result/1.0",
            "status": "CONFIRMATION_PASS" if passes else "CONFIRMATION_FAIL",
            "created_at_utc": _timestamp(),
            "bindings": bindings,
            "runtime": runtime,
            "decision": {
                "passes": passes,
                "mandatory_comparators": list(MANDATORY_COMPARATORS),
                "strongest_remaining_classical_method": strongest_name,
            },
            "held_gate": held_gate,
            "donor_losses": {
                **{method: values.tolist() for method, values in method_losses.items()},
                "strongest_remaining_classical_comparator": method_losses[
                    strongest_name
                ].tolist(),
            },
            "donor_mean_deviance": {
                **{
                    method: float(values.mean())
                    for method, values in method_losses.items()
                },
                "strongest_remaining_classical_comparator": float(
                    method_losses[strongest_name].mean()
                ),
            },
            "graph_specific_gain": graph_gain,
            "secondary_modules": secondary,
            "non_evaluable_modules": non_evaluable,
            "exploratory_relational_summary": relational,
            "truth_hashes": truth_hashes,
            "prediction_diagnostics": prediction_diagnostics,
            "held_records": [record["public_record"] for record in records],
            "access_audit": access_audit,
            "access_journal_sha256": _sha256(SCORE_ACCESS),
            "held_matrix_requested": True,
            "held_hearts_scored": len(records),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAPS,
            "rerun_permitted": False,
        }
    except BaseException as error:
        result = _public_failure(
            stage="score",
            schema="gse217494-heart-confirmation-result/1.0",
            error=error,
            bindings=bindings,
            journal=SCORE_ACCESS,
        )
    _write_json_x(SCORE_RESULT, result)
    return result


def recover_score(*, scratch: Path = DEFAULT_SCRATCH) -> dict[str, Any]:
    attempt, _, attempt_commit = _validate_score_attempt(allow_advanced_journal=True)
    bindings = {
        **attempt["bindings"],
        "score_attempt_tag": SCORE_ATTEMPT_TAG,
        "score_attempt_commit": attempt_commit,
        "score_attempt_sha256": _sha256(SCORE_ATTEMPT),
        "score_consumption_sha256": _sha256(SCORE_CONSUMPTION),
        "claim_token_sha256": attempt["claim_token_sha256"],
    }
    return _recover(
        stage="score",
        scratch=scratch,
        attempt_path=SCORE_ATTEMPT,
        consumption_path=SCORE_CONSUMPTION,
        journal_path=SCORE_ACCESS,
        result_path=SCORE_RESULT,
        schema="gse217494-heart-confirmation-result/1.0",
        bindings=bindings,
    )


def _validate_score_result() -> tuple[dict[str, Any], str]:
    attempt, source, attempt_commit = _validate_score_attempt(
        allow_advanced_journal=True
    )
    commit = _require_public_tag(
        SCORE_TAG,
        (SCORE_ATTEMPT, SCORE_CONSUMPTION, SCORE_ACCESS, SCORE_RESULT),
    )
    _require_ancestor(attempt_commit, commit)
    result = _read_json(SCORE_RESULT)
    expected_bindings = {
        **attempt["bindings"],
        "score_attempt_tag": SCORE_ATTEMPT_TAG,
        "score_attempt_commit": attempt_commit,
        "score_attempt_sha256": _sha256(SCORE_ATTEMPT),
        "score_consumption_sha256": _sha256(SCORE_CONSUMPTION),
        "claim_token_sha256": attempt["claim_token_sha256"],
    }
    if (
        result.get("schema") != "gse217494-heart-confirmation-result/1.0"
        or result.get("status") not in {"CONFIRMATION_PASS", "CONFIRMATION_FAIL"}
        or result.get("bindings") != expected_bindings
        or result.get("runtime") != _require_runtime()
        or result.get("held_matrix_requested") is not True
        or result.get("held_hearts_scored") != len(HELD_ORDER)
        or result.get("bootstrap_seed") != BOOTSTRAP_SEED
        or result.get("bootstrap_draws") != BOOTSTRAPS
        or result.get("access_journal_sha256") != _sha256(SCORE_ACCESS)
        or result.get("rerun_permitted") is not False
    ):
        raise PermissionError("public score result differs from the frozen contract")
    access = _success_access_audit(
        stage="score", journal=SCORE_ACCESS, samples=HELD_ORDER
    )
    if result.get("access_audit") != access:
        raise PermissionError("public held access audit differs")
    losses_object = result.get("donor_losses", {})
    strongest_name = source["source_cross_validation"][
        "strongest_remaining_classical"
    ]["selected"]
    expected_methods = {
        "primary",
        "graph_zero",
        "destroyed_links",
        "pooled_fixed_interaction_poisson",
        "etiology_specific_fixed_interaction_poisson",
        "standardized_fixed_margin_pearson",
        "fixed_margin_independence",
    }
    if source["models"]["exact_common_effect_conditional_field"] is not None:
        expected_methods.add("exact_common_effect_conditional_field")
    if set(losses_object) != expected_methods | {
        "strongest_remaining_classical_comparator"
    }:
        raise PermissionError("public held method panel differs")
    all_losses = {
        method: np.asarray(losses_object[method], dtype=float)
        for method in expected_methods
    }
    if any(
        values.shape != (len(HELD_ORDER),)
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        for values in all_losses.values()
    ):
        raise PermissionError("public held losses are invalid")
    if strongest_name not in all_losses or not np.array_equal(
        np.asarray(losses_object["strongest_remaining_classical_comparator"]),
        all_losses[strongest_name],
    ):
        raise PermissionError("public strongest classical loss alias differs")
    required_losses = {
        "primary",
        "pooled_fixed_interaction_poisson",
        "etiology_specific_fixed_interaction_poisson",
        "strongest_remaining_classical_comparator",
        "destroyed_links",
    }
    if not required_losses <= set(losses_object):
        raise PermissionError("public held losses are incomplete")
    losses = {
        method: np.asarray(losses_object[method], dtype=float)
        for method in required_losses
    }
    expected_means = {
        **{method: float(values.mean()) for method, values in all_losses.items()},
        "strongest_remaining_classical_comparator": float(
            all_losses[strongest_name].mean()
        ),
    }
    if result.get("donor_mean_deviance") != expected_means:
        raise PermissionError("public held donor means do not reproduce")
    labels = [sample["etiology"] for sample in _contract()["held"]]
    expected_gate = evaluate_held_gate(
        losses["primary"],
        {comparator: losses[comparator] for comparator in MANDATORY_COMPARATORS},
        labels,
    )
    passes = bool(expected_gate["passes"])
    if (
        result.get("held_gate") != expected_gate
        or result.get("decision", {}).get("passes") is not passes
        or result.get("decision", {}).get("mandatory_comparators")
        != list(MANDATORY_COMPARATORS)
        or result["status"] != ("CONFIRMATION_PASS" if passes else "CONFIRMATION_FAIL")
        or result.get("decision", {}).get("strongest_remaining_classical_method")
        != strongest_name
    ):
        raise PermissionError("public held decision arithmetic does not reproduce")
    graph_difference = all_losses["primary"] - all_losses["graph_zero"]
    graph_bootstrap = stratified_paired_bootstrap(graph_difference, labels)
    source_losses = source["source_cross_validation"]["selected_losses"]
    expected_graph = {
        "selected_graph_penalty_positive": source["models"]["primary"][
            "configuration"
        ]["graph_penalty"]
        > 0.0,
        "source_cv_primary_below_graph_zero": float(
            np.mean(source_losses["primary"])
        )
        < float(np.mean(source_losses["graph_zero"])),
        "held_bootstrap_upper_97_5_below_zero": graph_bootstrap["interval"][1]
        < 0.0,
        "primary_minus_graph_zero_mean": float(graph_difference.mean()),
        "stratified_paired_bootstrap_95_interval": list(
            graph_bootstrap["interval"]
        ),
        "bootstrap_draws": graph_bootstrap["draws"],
        "bootstrap_seed": graph_bootstrap["seed"],
    }
    expected_graph["claimed"] = bool(
        expected_graph["selected_graph_penalty_positive"]
        and expected_graph["source_cv_primary_below_graph_zero"]
        and expected_graph["held_bootstrap_upper_97_5_below_zero"]
    )
    if result.get("graph_specific_gain") != expected_graph:
        raise PermissionError("public graph-specific gain does not reproduce")

    published_modules = result.get("secondary_modules")
    if not isinstance(published_modules, dict) or set(published_modules) != set(
        source.get("secondary_module_models", {})
    ):
        raise PermissionError("public secondary module panel differs")
    module_losses = {}
    for name, module in published_modules.items():
        donor_losses = module.get("donor_losses", {})
        if set(donor_losses) != {"primary", *MANDATORY_COMPARATORS}:
            raise PermissionError("public secondary module losses differ")
        module_losses[name] = {
            method: np.asarray(values, dtype=float)
            for method, values in donor_losses.items()
        }
        if any(
            values.shape != (len(HELD_ORDER),)
            or not np.isfinite(values).all()
            or np.any(values < 0.0)
            for values in module_losses[name].values()
        ):
            raise PermissionError("public secondary module losses are invalid")
    if published_modules != _module_results(module_losses, labels):
        raise PermissionError("public secondary module inference does not reproduce")
    expected_non_evaluable = sorted(
        set(("endothelial", "fibroblast_fibrosis", "myeloid"))
        - set(published_modules)
    )
    if result.get("non_evaluable_modules") != expected_non_evaluable:
        raise PermissionError("public non-evaluable module list differs")

    relational = result.get("exploratory_relational_summary", {})
    relational_numbers = (
        relational.get("mean_top_k_jaccard"),
        relational.get("null_mean"),
        *(relational.get("null_interval") or (None, None)),
        relational.get("one_sided_monte_carlo_p"),
    )
    if (
        relational.get("neighbors") != 3
        or relational.get("donors") != len(HELD_ORDER)
        or relational.get("markers") != len(source["selected_symbols"])
        or relational.get("permutations") != NEIGHBOR_PERMUTATIONS
        or relational.get("seed") != NEIGHBOR_SEED
        or relational.get("exploratory") is not True
        or relational.get("joint_permutation")
        != "one RNA-marker relabeling shared across held donors"
        or any(
            not isinstance(value, (int, float)) or not np.isfinite(value)
            for value in relational_numbers
        )
        or not all(0.0 <= float(value) <= 1.0 for value in relational_numbers)
    ):
        raise PermissionError("public exploratory relational summary differs")
    return result, commit


def _validate_score_outcome() -> tuple[dict[str, Any], str]:
    status = _read_json(SCORE_RESULT).get("status")
    if status in {"CONFIRMATION_PASS", "CONFIRMATION_FAIL"}:
        return _validate_score_result()
    return _validate_terminal_result("score")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for name in ("claim-source", "source", "claim-score", "score"):
        command = subparsers.add_parser(name)
        command.add_argument("--claim-token", type=Path, required=True)
        command.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    for name in ("recover-source", "recover-score"):
        command = subparsers.add_parser(name)
        command.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    subparsers.add_parser("authorize-score")
    subparsers.add_parser("validate-source")
    subparsers.add_parser("validate-score")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.stage == "claim-source":
        result = claim_source(
            claim_token=arguments.claim_token, scratch=arguments.scratch
        )
    elif arguments.stage == "source":
        result = run_source(
            claim_token=arguments.claim_token, scratch=arguments.scratch
        )
    elif arguments.stage == "recover-source":
        result = recover_source(scratch=arguments.scratch)
    elif arguments.stage == "authorize-score":
        result = authorize_score()
    elif arguments.stage == "claim-score":
        result = claim_score(
            claim_token=arguments.claim_token, scratch=arguments.scratch
        )
    elif arguments.stage == "score":
        result = run_score(claim_token=arguments.claim_token, scratch=arguments.scratch)
    elif arguments.stage == "recover-score":
        result = recover_score(scratch=arguments.scratch)
    elif arguments.stage == "validate-source":
        result, _ = _validate_source_outcome()
    else:
        result, _ = _validate_score_outcome()
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    return 1 if str(result["status"]).startswith("TERMINAL_") else 0


if __name__ == "__main__":
    raise SystemExit(main())
