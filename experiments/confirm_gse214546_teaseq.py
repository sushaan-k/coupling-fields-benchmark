"""One-shot GSE214546 TEA-seq held-donor confirmation campaign.

The three CLI stages are deliberately separate. ``source`` may access only the
eight amended source donors. ``predict`` may access held metadata and frozen
RNA rows, but its HDF5 allowlist excludes every ADT sparse dataset. ``score``
requires a publicly frozen prediction and an explicit public authorization
before it can read held ADT values.

The access journal covers failures catchable by Python. External process loss
after token consumption is permanently ``CRASH_UNEVALUABLE`` and cannot be
rerun; the in-memory journal is not power-loss durable.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import platform
import stat
import subprocess
import tempfile
from typing import Any, Iterable
import urllib.parse
import urllib.request

import h5py
import numpy as np
import scipy
from scipy.optimize import brentq
from scipy.special import gammaln, logsumexp

from mapreg.context_conditional_coupling import (
    ContextConditionalCouplingFit,
    fit_context_conditional_log_odds,
    predict_context_log_odds,
)
from mapreg.common_effect_conditional import (
    fit_common_effect_conditional_log_odds,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/confirmation/gse214546_teaseq"
DESIGNATION = DATA_DIR / "candidate_designation_v1.json"
AMENDMENT = DATA_DIR / "pre_access_schema_amendment_v1.json"
IMPLEMENTATION_CLARIFICATION = (
    DATA_DIR / "pre_access_implementation_clarification_v1.json"
)
CV_AVAILABILITY_CLARIFICATION = (
    DATA_DIR / "pre_access_cv_availability_clarification_v1.json"
)
NORMALIZATION_CORRECTION = DATA_DIR / "pre_access_normalization_correction_v1.json"
SPARSE_ACCESS_CLARIFICATION = (
    DATA_DIR / "pre_access_sparse_access_clarification_v1.json"
)
CRASH_SEMANTICS_CLARIFICATION = (
    DATA_DIR / "pre_access_crash_semantics_clarification_v1.json"
)
PROTOCOL = ROOT / "docs/GSE214546_TEASEQ_HELD_DONOR_CONFIRMATION_PROTOCOL_2026-08-30.md"
RUNNER = ROOT / "experiments/confirm_gse214546_teaseq.py"
TEST = ROOT / "tests/test_gse214546_confirmation.py"
COUPLING_MODULE = ROOT / "mapreg/context_conditional_coupling.py"
CONTEXT_TEST = ROOT / "tests/test_context_conditional_coupling.py"
COMMON_EFFECT_MODULE = ROOT / "mapreg/common_effect_conditional.py"
HETEROGENEITY_MODULE = ROOT / "mapreg/heterogeneity_adaptive_coupling.py"
COMMON_EFFECT_TEST = ROOT / "tests/test_common_effect_conditional.py"
HETEROGENEITY_TEST = ROOT / "tests/test_heterogeneity_adaptive_coupling.py"

SOURCE_ATTEMPT = DATA_DIR / "source_attempt_v1.json"
SOURCE_RESULT = ROOT / "results/development/gse214546_teaseq_source_v1.json"
PREDICTION_ATTEMPT = DATA_DIR / "prediction_attempt_v1.json"
PREDICTION_RESULT = ROOT / "results/gse214546_teaseq_predictions_v1.json"
SCORE_AUTHORIZATION = DATA_DIR / "score_authorization_v1.json"
SCORE_ATTEMPT = DATA_DIR / "score_attempt_v1.json"
SCORE_RESULT = ROOT / "results/gse214546_teaseq_confirmation_v1.json"
DEFAULT_SCRATCH = Path("/private/tmp/gse214546-teaseq-v1")

PUBLIC_ORIGIN = "https://github.com/sushaan-k/coupling-fields-benchmark.git"
CANDIDATE_TAG = "gse214546-teaseq-v1-candidate"
AMENDMENT_TAG = "gse214546-teaseq-v1-pre-access-amendment"
IMPLEMENTATION_CLARIFICATION_TAG = (
    "gse214546-teaseq-v1-implementation-clarification"
)
CV_AVAILABILITY_TAG = "gse214546-teaseq-v1-cv-availability"
NORMALIZATION_CORRECTION_TAG = "gse214546-teaseq-v1-normalization-correction"
SPARSE_ACCESS_CLARIFICATION_TAG = (
    "gse214546-teaseq-v1-sparse-access-clarification"
)
CRASH_SEMANTICS_CLARIFICATION_TAG = (
    "gse214546-teaseq-v1-crash-semantics-clarification"
)
IMPLEMENTATION_TAG = "gse214546-teaseq-v1-implementation"
SOURCE_ATTEMPT_TAG = "gse214546-teaseq-v1-source-attempt"
SOURCE_TAG = "gse214546-teaseq-v1-source"
PREDICTION_ATTEMPT_TAG = "gse214546-teaseq-v1-prediction-attempt"
PREDICTION_TAG = "gse214546-teaseq-v1-prediction"
SCORE_AUTHORIZATION_TAG = "gse214546-teaseq-v1-score-authorization"
SCORE_ATTEMPT_TAG = "gse214546-teaseq-v1-score-attempt"
SCORE_TAG = "gse214546-teaseq-v1-score"

CANDIDATE_SHA256 = "0d86092f047a40c2b58120185c477e8588d0f4b66346ba81bd1f28f6c24042f2"
AMENDMENT_SHA256 = "e675caf330934671cf28de3bc5795713b685fb75e8c82653ab6eb1424fd9cd58"
IMPLEMENTATION_CLARIFICATION_SHA256 = (
    "54304d0829f74c684b4b10d41806bc2651832ca35b6d0328ecfeefabfe41ecea"
)
CV_AVAILABILITY_SHA256 = (
    "cb8eb8285f2ac2a58ab0ede774966b40cc217f55024db524a49dabaed7bdb281"
)
NORMALIZATION_CORRECTION_SHA256 = (
    "b469bd0e482c49a81537dfcc20930a7dfd25c5a9276119d4a4622403fc0e5938"
)
SPARSE_ACCESS_CLARIFICATION_SHA256 = (
    "1bda50b7990da0e9b76485f184628df88e83a0d7f2a564a5f440c4024bdda628"
)
CRASH_SEMANTICS_CLARIFICATION_SHA256 = (
    "fdb38aee98cd7bd4a65aafcb2566f109ece6c865461096943f2e8abb0a14709a"
)
PROTOCOL_SHA256 = "bbe21165fd034129bc1606243ba202deb50c3625fa55c3594967e15e0f62b0a5"

CELL_BUDGET = 512
PROTEIN_HIGH_COUNT = 256
MINIMUM_MARKERS = 20
MAXIMUM_MARKERS = 24
CELL_SALT = "GSE214546-CELL-v1"
ADT_TIE_SALT = "GSE214546-ADT-TIE-v1"
DESTROY_SALT = "GSE214546-DESTROY-v1"
BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 21_454_601
DEVIATION_GRID = (0.1, 1.0, 10.0)
AGE_RIDGE_GRID = (0.1, 1.0, 10.0)
PRIMARY_TRANSPORT_GRID = (0.75, 1.0)
POISSON_TRANSPORT_GRID = (0.75, 1.0, 1.25)
INTERCEPT_RIDGE = 0.01
MAXIMUM_CONDITION_NUMBER = 1e12
GRADIENT_TOLERANCE = 1e-8

METHODS = (
    "primary",
    "pooled_fixed_interaction_poisson",
    "age_stratified_fixed_interaction_poisson",
    "destroyed_link",
    "common_effect_exact_conditional",
    "signed_root_deviance",
    "independence",
)

PREDICT_H5_ALLOWLIST = frozenset(
    {
        "matrix/barcodes",
        "matrix/features/name",
        "matrix/features/feature_type",
        "matrix/shape",
        "matrix/indptr",
        "matrix/indices",
        "matrix/data",
    }
)
LINKED_H5_ALLOWLIST = PREDICT_H5_ALLOWLIST | frozenset(
    {
        "ADT/barcodes",
        "ADT/features/id",
        "ADT/shape",
        "ADT/indptr",
        "ADT/indices",
        "ADT/data",
    }
)


class ProtocolRefusal(RuntimeError):
    """A terminal outcome fixed by the public campaign protocol."""

    def __init__(self, reason_code: str, details: dict[str, Any] | None = None):
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.details = {} if details is None else details


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _axis_sha256(values: Iterable[str]) -> str:
    return _bytes_sha256("".join(f"{value}\n" for value in values).encode())


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise PermissionError(f"JSON root must be an object: {path.name}")
    return value


def _write_json_x(path: Path, value: dict[str, Any]) -> None:
    payload = _canonical_json_bytes(value)
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
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments), cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def _tag_commit(tag: str) -> str:
    local = _git("rev-list", "-n", "1", tag)
    rows = _git("ls-remote", "--tags", PUBLIC_ORIGIN, f"refs/tags/{tag}^{{}}").splitlines()
    if len(rows) != 1 or rows[0].split()[0] != local:
        raise PermissionError(f"public annotated tag is absent or differs: {tag}")
    return local


def _require_ancestor(ancestor: str, descendant: str) -> None:
    completed = subprocess.run(
        ("git", "merge-base", "--is-ancestor", ancestor, descendant),
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise PermissionError("public campaign tags are not an ordered ancestry chain")


def _require_tag_paths(tag: str, paths: Iterable[Path]) -> str:
    commit = _tag_commit(tag)
    for path in paths:
        relative = _relative(path)
        frozen = subprocess.check_output(
            ("git", "show", f"{commit}:{relative}"), cwd=ROOT
        )
        if _bytes_sha256(frozen) != _sha256(path):
            raise PermissionError(f"working file differs from {tag}: {relative}")
    return commit


def _runtime() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "h5py": h5py.__version__,
    }


def _implementation_bindings() -> dict[str, str]:
    return {
        "runner": _sha256(RUNNER),
        "test": _sha256(TEST),
        "context_conditional_coupling": _sha256(COUPLING_MODULE),
        "context_conditional_coupling_test": _sha256(CONTEXT_TEST),
        "common_effect_conditional": _sha256(COMMON_EFFECT_MODULE),
        "heterogeneity_adaptive_coupling": _sha256(HETEROGENEITY_MODULE),
        "common_effect_conditional_test": _sha256(COMMON_EFFECT_TEST),
        "heterogeneity_adaptive_coupling_test": _sha256(HETEROGENEITY_TEST),
    }


def _contract() -> dict[str, Any]:
    if (
        _sha256(DESIGNATION) != CANDIDATE_SHA256
        or _sha256(AMENDMENT) != AMENDMENT_SHA256
        or _sha256(IMPLEMENTATION_CLARIFICATION)
        != IMPLEMENTATION_CLARIFICATION_SHA256
        or _sha256(CV_AVAILABILITY_CLARIFICATION) != CV_AVAILABILITY_SHA256
        or _sha256(NORMALIZATION_CORRECTION) != NORMALIZATION_CORRECTION_SHA256
        or _sha256(SPARSE_ACCESS_CLARIFICATION)
        != SPARSE_ACCESS_CLARIFICATION_SHA256
        or _sha256(CRASH_SEMANTICS_CLARIFICATION)
        != CRASH_SEMANTICS_CLARIFICATION_SHA256
        or _sha256(PROTOCOL) != PROTOCOL_SHA256
    ):
        raise PermissionError("GSE214546 public contract bytes differ from the freeze")
    candidate = _read_json(DESIGNATION)
    amendment = _read_json(AMENDMENT)
    implementation_clarification = _read_json(IMPLEMENTATION_CLARIFICATION)
    cv_availability = _read_json(CV_AVAILABILITY_CLARIFICATION)
    normalization = _read_json(NORMALIZATION_CORRECTION)
    sparse_access = _read_json(SPARSE_ACCESS_CLARIFICATION)
    crash_semantics = _read_json(CRASH_SEMANTICS_CLARIFICATION)
    if (
        candidate.get("schema") != "gse214546-teaseq-candidate/1.0"
        or amendment.get("schema")
        != "gse214546-teaseq-pre-access-schema-amendment/1.0"
        or amendment.get("candidate_designation_sha256") != CANDIDATE_SHA256
        or amendment.get("protocol_sha256") != PROTOCOL_SHA256
        or amendment.get("numeric_count_or_sparse_value_dataset_read") is not False
        or amendment.get("held_h5_requested_or_opened") is not False
        or implementation_clarification.get("schema")
        != "gse214546-teaseq-pre-access-implementation-clarification/1.0"
        or implementation_clarification.get("schema_amendment_sha256")
        != AMENDMENT_SHA256
        or implementation_clarification.get("numeric_count_or_sparse_value_dataset_read")
        is not False
        or implementation_clarification.get("held_h5_requested_or_opened") is not False
        or cv_availability.get("schema")
        != "gse214546-teaseq-pre-access-cv-availability-clarification/1.0"
        or cv_availability.get("prior_clarification_sha256")
        != IMPLEMENTATION_CLARIFICATION_SHA256
        or cv_availability.get("numeric_count_or_sparse_value_dataset_read") is not False
        or cv_availability.get("held_h5_requested_or_opened") is not False
        or normalization.get("schema")
        != "gse214546-teaseq-pre-access-normalization-correction/1.0"
        or normalization.get("corrected_context_estimator_sha256")
        != _sha256(COUPLING_MODULE)
        or normalization.get("corrected_context_estimator_test_sha256")
        != _sha256(CONTEXT_TEST)
        or normalization.get("numeric_count_or_sparse_value_dataset_read") is not False
        or normalization.get("held_h5_requested_or_opened") is not False
        or sparse_access.get("schema")
        != "gse214546-teaseq-pre-access-sparse-access-clarification/1.0"
        or sparse_access.get("prior_normalization_correction_sha256")
        != NORMALIZATION_CORRECTION_SHA256
        or sparse_access.get("numeric_count_or_sparse_value_dataset_read") is not False
        or sparse_access.get("held_h5_requested_or_opened") is not False
        or crash_semantics.get("schema")
        != "gse214546-teaseq-pre-access-crash-semantics-clarification/1.0"
        or crash_semantics.get("prior_sparse_access_clarification_sha256")
        != SPARSE_ACCESS_CLARIFICATION_SHA256
        or crash_semantics.get("numeric_count_or_sparse_value_dataset_read") is not False
        or crash_semantics.get("held_h5_requested_or_opened") is not False
        or crash_semantics.get("estimator_axis_grid_or_gate_changed") is not False
        or crash_semantics.get("firewall_changed") is not False
        or crash_semantics.get("rerun_permitted") is not False
    ):
        raise PermissionError("GSE214546 contract is not frozen before assay access")
    source_ids = amendment["amended_split"]["source"]
    held_ids = amendment["amended_split"]["held"]
    by_gsm = {sample["gsm"]: dict(sample) for sample in candidate["samples"]}
    if (
        len(source_ids) != 8
        or len(held_ids) != 8
        or len(set(source_ids + held_ids)) != 16
        or set(source_ids + held_ids) != set(by_gsm)
    ):
        raise PermissionError("amended source and held split differs from the freeze")
    cmv = amendment["amended_split"]["cmv_status"]

    def records(identifiers: list[str], role: str) -> list[dict[str, Any]]:
        output = []
        for gsm in identifiers:
            record = {**by_gsm[gsm], "role": role}
            record["cmv_status"] = cmv[record["donor"]]
            template = candidate["ftp_url_template"]
            record["h5_url"] = template.format(
                gsm=gsm, filename=urllib.parse.quote(record["h5_filename"])
            )
            record["metadata_url"] = template.format(
                gsm=gsm, filename=urllib.parse.quote(record["metadata_filename"])
            )
            output.append(record)
        return output

    markers = amendment.get("candidate_markers")
    if (
        not isinstance(markers, list)
        or len(markers) != 53
        or len({marker["protein"] for marker in markers}) != 53
        or len({marker["rna"] for marker in markers}) != 53
        or amendment["amendment"].get("minimum_retained_markers") != MINIMUM_MARKERS
        or amendment["amendment"].get("maximum_retained_markers") != MAXIMUM_MARKERS
    ):
        raise PermissionError("amended marker contract differs from the freeze")
    source = records(source_ids, "source")
    held = records(held_ids, "held")
    for arm in (source, held):
        if [sum(sample["age_group"] == group for sample in arm) for group in ("adult", "pediatric")] != [4, 4]:
            raise PermissionError("amended age balance differs from the freeze")
    return {
        "candidate": candidate,
        "amendment": amendment,
        "implementation_clarification": implementation_clarification,
        "cv_availability_clarification": cv_availability,
        "normalization_correction": normalization,
        "sparse_access_clarification": sparse_access,
        "crash_semantics_clarification": crash_semantics,
        "source": source,
        "held": held,
        "markers": [dict(marker) for marker in markers],
    }


def _verify_implementation_freeze() -> dict[str, str]:
    _contract()
    candidate = _require_tag_paths(CANDIDATE_TAG, (DESIGNATION, PROTOCOL))
    amendment = _require_tag_paths(AMENDMENT_TAG, (DESIGNATION, PROTOCOL, AMENDMENT))
    clarification = _require_tag_paths(
        IMPLEMENTATION_CLARIFICATION_TAG,
        (DESIGNATION, PROTOCOL, AMENDMENT, IMPLEMENTATION_CLARIFICATION),
    )
    cv_availability = _require_tag_paths(
        CV_AVAILABILITY_TAG,
        (
            IMPLEMENTATION_CLARIFICATION,
            CV_AVAILABILITY_CLARIFICATION,
        ),
    )
    normalization = _require_tag_paths(
        NORMALIZATION_CORRECTION_TAG,
        (
            IMPLEMENTATION_CLARIFICATION,
            CV_AVAILABILITY_CLARIFICATION,
            NORMALIZATION_CORRECTION,
            COUPLING_MODULE,
        ),
    )
    sparse_access = _require_tag_paths(
        SPARSE_ACCESS_CLARIFICATION_TAG,
        (NORMALIZATION_CORRECTION, SPARSE_ACCESS_CLARIFICATION),
    )
    crash_semantics = _require_tag_paths(
        CRASH_SEMANTICS_CLARIFICATION_TAG,
        (SPARSE_ACCESS_CLARIFICATION, CRASH_SEMANTICS_CLARIFICATION),
    )
    implementation = _require_tag_paths(
        IMPLEMENTATION_TAG,
        (
            DESIGNATION,
            PROTOCOL,
            AMENDMENT,
            IMPLEMENTATION_CLARIFICATION,
            CV_AVAILABILITY_CLARIFICATION,
            NORMALIZATION_CORRECTION,
            SPARSE_ACCESS_CLARIFICATION,
            CRASH_SEMANTICS_CLARIFICATION,
            RUNNER,
            TEST,
            COUPLING_MODULE,
            CONTEXT_TEST,
            COMMON_EFFECT_MODULE,
            HETEROGENEITY_MODULE,
            COMMON_EFFECT_TEST,
            HETEROGENEITY_TEST,
        ),
    )
    _require_ancestor(candidate, amendment)
    _require_ancestor(amendment, clarification)
    _require_ancestor(clarification, cv_availability)
    _require_ancestor(cv_availability, normalization)
    _require_ancestor(normalization, sparse_access)
    _require_ancestor(sparse_access, crash_semantics)
    _require_ancestor(crash_semantics, implementation)
    return {
        "candidate_tag": CANDIDATE_TAG,
        "candidate_commit": candidate,
        "amendment_tag": AMENDMENT_TAG,
        "amendment_commit": amendment,
        "implementation_clarification_tag": IMPLEMENTATION_CLARIFICATION_TAG,
        "implementation_clarification_commit": clarification,
        "cv_availability_tag": CV_AVAILABILITY_TAG,
        "cv_availability_commit": cv_availability,
        "normalization_correction_tag": NORMALIZATION_CORRECTION_TAG,
        "normalization_correction_commit": normalization,
        "sparse_access_clarification_tag": SPARSE_ACCESS_CLARIFICATION_TAG,
        "sparse_access_clarification_commit": sparse_access,
        "crash_semantics_clarification_tag": CRASH_SEMANTICS_CLARIFICATION_TAG,
        "crash_semantics_clarification_commit": crash_semantics,
        "implementation_tag": IMPLEMENTATION_TAG,
        "implementation_commit": implementation,
    }


def _scratch(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == ROOT.resolve() or ROOT.resolve() in resolved.parents:
        raise PermissionError("scratch must be outside the public repository")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or any(path.iterdir()):
        raise PermissionError("scratch must be a real empty directory")
    path.chmod(stat.S_IRWXU)
    return path


def _outside_repository(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if resolved == ROOT.resolve() or ROOT.resolve() in resolved.parents:
        raise PermissionError(f"{label} must be outside the public repository")
    return resolved


def _create_claim_token(path: Path) -> str:
    resolved = _outside_repository(path, "claim token")
    resolved.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    token = os.urandom(32)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(resolved, flags, 0o600)
    try:
        os.write(descriptor, token)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    resolved.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return _bytes_sha256(token)


def _consume_claim_token(path: Path | None, expected_sha256: str) -> str:
    if path is None:
        raise PermissionError("--claim-token is required before campaign consumption")
    if path.is_symlink():
        raise PermissionError("claim token file must not be a symbolic link")
    resolved = _outside_repository(path, "claim token")
    if not resolved.is_file() or resolved.is_symlink():
        raise PermissionError("claim token file is absent or not a regular file")
    mode = resolved.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise PermissionError("claim token permissions must be owner-only")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(resolved, flags)
    try:
        token = os.read(descriptor, 33)
    finally:
        os.close(descriptor)
    observed = _bytes_sha256(token)
    if len(token) != 32 or not hmac.compare_digest(observed, expected_sha256):
        raise PermissionError("claim token preimage differs from the public attempt")
    resolved.unlink()
    directory = os.open(resolved.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return observed


class _FrozenRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, expected_url: str):
        self.expected_url = expected_url

    def redirect_request(self, request: Any, fp: Any, code: int, message: str, headers: Any, newurl: str) -> Any:
        expected = urllib.parse.urlparse(self.expected_url)
        observed = urllib.parse.urlparse(newurl)
        if expected.hostname != observed.hostname or urllib.parse.unquote(expected.path) != urllib.parse.unquote(observed.path):
            raise PermissionError("download redirected outside the frozen GEO object")
        return super().redirect_request(request, fp, code, message, headers, newurl)


def _fetch(
    url: str,
    expected_bytes: int,
    filename: str,
    scratch: Path,
    audit: list[dict[str, Any]],
    *,
    expected_sha256: str | None = None,
) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(prefix="gse214546-", suffix=Path(filename).suffix, dir=scratch)
    os.close(descriptor)
    path = Path(temporary_name)
    record: dict[str, Any] = {
        "url": url,
        "filename": filename,
        "expected_bytes": int(expected_bytes),
        "request_started": True,
        "completed": False,
        "deleted": False,
        "datasets_read": [],
    }
    audit.append(record)
    digest = hashlib.sha256()
    observed = 0
    try:
        opener = urllib.request.build_opener(_FrozenRedirectHandler(url))
        request = urllib.request.Request(url, headers={"User-Agent": "mapreg/1.0"})
        with opener.open(request, timeout=180) as response, path.open("wb") as output:
            final_url = response.geturl()
            record["final_url"] = final_url
            expected = urllib.parse.urlparse(url)
            final = urllib.parse.urlparse(final_url)
            if expected.hostname != final.hostname or urllib.parse.unquote(expected.path) != urllib.parse.unquote(final.path):
                raise PermissionError("download response differs from the frozen GEO object")
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                observed += len(block)
                if observed > expected_bytes:
                    raise ProtocolRefusal("DOWNLOAD_BYTE_COUNT_EXCEEDED")
                digest.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        record["observed_bytes"] = observed
        record["sha256"] = digest.hexdigest()
        if observed != expected_bytes or path.stat().st_size != expected_bytes:
            raise ProtocolRefusal("DOWNLOAD_BYTE_COUNT_MISMATCH")
        if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
            raise ProtocolRefusal("DOWNLOAD_SHA256_MISMATCH")
        record["completed"] = True
        return path
    except BaseException:
        record["observed_bytes"] = observed
        record["partial_sha256"] = digest.hexdigest()
        path.unlink(missing_ok=True)
        record["deleted"] = not path.exists()
        raise


def _delete_download(path: Path, record: dict[str, Any]) -> None:
    path.unlink(missing_ok=True)
    record["deleted"] = not path.exists()


def _eligible_metadata(path: Path, sample: dict[str, Any]) -> tuple[set[str], dict[str, Any]]:
    rows = 0
    singlets: set[str] = set()
    seen: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "barcodes" not in reader.fieldnames or "singlet" not in reader.fieldnames:
            raise ProtocolRefusal("METADATA_SCHEMA_MISMATCH")
        for row in reader:
            rows += 1
            barcode = row.get("barcodes", "")
            if not barcode or barcode in seen:
                raise ProtocolRefusal("METADATA_BARCODE_MISSING_OR_DUPLICATED")
            seen.add(barcode)
            if row.get("singlet") == "TRUE":
                singlets.add(barcode)
    if len(singlets) < CELL_BUDGET:
        raise ProtocolRefusal("FEWER_THAN_512_METADATA_SINGLETS")
    if sample["gsm"] == "GSM6611363" and (rows != 11_191 or len(singlets) != 10_295):
        raise ProtocolRefusal("INVENTORY_METADATA_COUNTS_MISMATCH")
    return singlets, {
        "rows": rows,
        "unique_barcodes": len(seen),
        "literal_true_singlets": len(singlets),
        "barcode_column": "barcodes",
        "singlet_column": "singlet",
        "singlet_value": "TRUE",
    }


def _decode_axis(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


def _integer_array(values: np.ndarray, label: str) -> np.ndarray:
    numeric = np.asarray(values, dtype=float)
    if (
        not np.isfinite(numeric).all()
        or np.any(numeric < 0.0)
        or not np.array_equal(numeric, np.rint(numeric))
        or np.any(numeric > np.iinfo(np.int64).max)
    ):
        raise ProtocolRefusal(f"{label}_NOT_NONNEGATIVE_INTEGER")
    return np.rint(numeric).astype(np.int64)


def _selection_provenance(key: Any) -> dict[str, Any]:
    if isinstance(key, tuple) and not key:
        return {"kind": "all"}
    if isinstance(key, slice):
        return {"kind": "slice", "start": key.start, "stop": key.stop, "step": key.step}
    positions = np.asarray(key, dtype=np.int64)
    return {
        "kind": "fancy_positions",
        "count": int(positions.size),
        "minimum": None if positions.size == 0 else int(positions.min()),
        "maximum": None if positions.size == 0 else int(positions.max()),
        "sha256": _array_sha256(positions),
    }


def _dataset(
    handle: h5py.File,
    path: str,
    allowlist: frozenset[str],
    audit_record: dict[str, Any],
) -> h5py.Dataset:
    if path not in allowlist:
        raise PermissionError(f"HDF5 dataset is forbidden in this stage: {path}")
    opened = audit_record.setdefault("datasets_opened", [])
    if path not in opened:
        opened.append(path)
        opened.sort()
    link = handle.get(path, getlink=True)
    if isinstance(link, h5py.ExternalLink):
        raise PermissionError("external HDF5 links are forbidden")
    item = handle.get(path)
    if not isinstance(item, h5py.Dataset):
        raise ProtocolRefusal("HDF5_DATASET_ABSENT", {"dataset": path})
    return item


def _read_dataset(
    dataset: h5py.Dataset,
    path: str,
    key: Any,
    audit_record: dict[str, Any],
) -> np.ndarray:
    event = {
        "dataset": path,
        "selection": _selection_provenance(key),
        "started": True,
        "completed": False,
    }
    audit_record.setdefault("dataset_access_events", []).append(event)
    datasets = audit_record.setdefault("datasets_read", [])
    if path not in datasets:
        datasets.append(path)
        datasets.sort()
    try:
        values = np.asarray(dataset[key])
    except BaseException as error:
        event["error_type"] = type(error).__name__
        raise
    event["completed"] = True
    event["returned_shape"] = list(values.shape)
    event["returned_dtype"] = values.dtype.str
    event["returned_elements"] = int(values.size)
    return values


def _selected_cells(barcodes: list[str], eligible: set[str], gsm: str) -> tuple[np.ndarray, list[str]]:
    if len(barcodes) != len(set(barcodes)):
        raise ProtocolRefusal("H5_BARCODE_AXIS_DUPLICATED")
    positions = {barcode: index for index, barcode in enumerate(barcodes)}
    matched = sorted(
        (barcode for barcode in eligible if barcode in positions),
        key=lambda barcode: (
            hashlib.sha256(f"{CELL_SALT}|{gsm}|{barcode}".encode()).hexdigest(),
            barcode,
        ),
    )
    if len(matched) < CELL_BUDGET:
        raise ProtocolRefusal("FEWER_THAN_512_MATCHED_SINGLETS")
    selected = matched[:CELL_BUDGET]
    return np.asarray([positions[barcode] for barcode in selected], dtype=np.int64), selected


def _marker_rows(axis: list[str], markers: list[dict[str, Any]], key: str) -> list[int]:
    output = []
    for marker in markers:
        matches = [index for index, value in enumerate(axis) if value == marker[key]]
        if len(matches) != 1:
            raise ProtocolRefusal("MARKER_AXIS_RESOLUTION_NOT_UNIQUE", {"marker": marker[key], "axis": key})
        output.append(matches[0])
    return output


def _csc_rows(
    handle: h5py.File,
    group: str,
    shape: tuple[int, int],
    selected_cells: np.ndarray,
    selected_rows: list[int],
    allowlist: frozenset[str],
    audit_record: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    indptr_path = f"{group}/indptr"
    indices_path = f"{group}/indices"
    data_path = f"{group}/data"
    indptr = _dataset(handle, indptr_path, allowlist, audit_record)
    indices = _dataset(handle, indices_path, allowlist, audit_record)
    data = _dataset(handle, data_path, allowlist, audit_record)
    if indptr.shape != (shape[1] + 1,) or indices.ndim != 1 or data.ndim != 1 or indices.shape != data.shape:
        raise ProtocolRefusal("CSC_DATASET_SHAPES_MISMATCH")
    row_lookup = {row: output for output, row in enumerate(selected_rows)}
    output = np.zeros((len(selected_cells), len(selected_rows)), dtype=np.int64)
    span_digest = hashlib.sha256()
    position_digest = hashlib.sha256()
    decoded_indices = 0
    decoded_values = 0
    summary_key = "gex_access" if group == "matrix" else "adt_access"
    summary: dict[str, Any] = {
        "orientation": "feature_by_cell_csc",
        "selected_columns": len(selected_cells),
        "selected_rows": len(selected_rows),
        "selected_column_indices_sha256": _array_sha256(selected_cells),
        "selected_feature_rows_sha256": _array_sha256(
            np.asarray(selected_rows, dtype=np.int64)
        ),
        "indptr_values_decoded": 0,
        "indices_decoded": 0,
        "selected_data_values_decoded": 0,
        "slice_contract": "indptr[cell:cell+2], indices[start:stop], data[matched absolute positions]",
        "column_span_sha256": span_digest.hexdigest(),
        "data_positions_sha256": position_digest.hexdigest(),
        "full_sparse_data_read": False,
        "completed": False,
    }
    audit_record[summary_key] = summary
    for output_cell, cell in enumerate(selected_cells.tolist()):
        pointers = _integer_array(
            _read_dataset(
                indptr, indptr_path, slice(cell, cell + 2), audit_record
            ),
            "CSC_INDPTR",
        )
        summary["indptr_values_decoded"] += 2
        if pointers.shape != (2,):
            raise ProtocolRefusal("CSC_INDPTR_SLICE_MISMATCH")
        start, stop = map(int, pointers)
        if start > stop or stop > len(data):
            raise ProtocolRefusal("CSC_POINTER_RANGE_INVALID")
        rows = _integer_array(
            _read_dataset(indices, indices_path, slice(start, stop), audit_record),
            "CSC_INDICES",
        )
        if np.any(rows >= shape[0]) or len(rows) != len(set(rows.tolist())):
            raise ProtocolRefusal("CSC_ROW_INDEX_INVALID_OR_DUPLICATED")
        decoded_indices += len(rows)
        span_digest.update(np.asarray((cell, start, stop), dtype=np.int64).tobytes())
        summary["indices_decoded"] = decoded_indices
        summary["column_span_sha256"] = span_digest.hexdigest()
        local = np.asarray([offset for offset, row in enumerate(rows.tolist()) if row in row_lookup], dtype=np.int64)
        if local.size:
            absolute = local + start
            values = _integer_array(
                _read_dataset(data, data_path, absolute, audit_record), "CSC_DATA"
            )
            decoded_values += len(values)
            position_digest.update(np.asarray(absolute, dtype=np.int64).tobytes())
            summary["selected_data_values_decoded"] = decoded_values
            summary["data_positions_sha256"] = position_digest.hexdigest()
            for row, value in zip(rows[local].tolist(), values.tolist()):
                output[output_cell, row_lookup[row]] = value
    summary["completed"] = True
    return output, dict(summary)


def _read_h5(
    path: Path,
    eligible: set[str],
    sample: dict[str, Any],
    markers: list[dict[str, Any]],
    stage: str,
    audit_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if stage not in {"source", "predict", "score"}:
        raise ValueError("unknown HDF5 access stage")
    allowlist = PREDICT_H5_ALLOWLIST if stage == "predict" else LINKED_H5_ALLOWLIST
    record = {} if audit_record is None else audit_record
    record["h5_stage"] = stage
    record["h5_open_started"] = True
    record["h5_open_completed"] = False
    record["h5_reduction_completed"] = False
    record.setdefault("datasets_opened", [])
    record.setdefault("datasets_read", [])
    record.setdefault("dataset_access_events", [])
    with h5py.File(path, "r") as handle:
        record["h5_open_completed"] = True
        gex_barcodes = _decode_axis(
            _read_dataset(
                _dataset(handle, "matrix/barcodes", allowlist, record),
                "matrix/barcodes",
                (),
                record,
            )
        )
        gex_names = _decode_axis(
            _read_dataset(
                _dataset(handle, "matrix/features/name", allowlist, record),
                "matrix/features/name",
                (),
                record,
            )
        )
        gex_types = _decode_axis(
            _read_dataset(
                _dataset(handle, "matrix/features/feature_type", allowlist, record),
                "matrix/features/feature_type",
                (),
                record,
            )
        )
        gex_shape_values = _integer_array(
            _read_dataset(
                _dataset(handle, "matrix/shape", allowlist, record),
                "matrix/shape",
                (),
                record,
            ),
            "GEX_SHAPE",
        )
        if gex_shape_values.shape != (2,) or tuple(gex_shape_values.tolist()) != (len(gex_names), len(gex_barcodes)) or len(gex_types) != len(gex_names):
            raise ProtocolRefusal("GEX_AXIS_SHAPE_MISMATCH")
        gex_rows = _marker_rows(gex_names, markers, "rna")
        if any(gex_types[row] != "Gene Expression" for row in gex_rows):
            raise ProtocolRefusal("GEX_MARKER_FEATURE_TYPE_MISMATCH")
        selected_indices, selected_barcodes = _selected_cells(gex_barcodes, eligible, sample["gsm"])
        record["selected_cells"] = len(selected_barcodes)
        record["selected_cell_axis_sha256"] = _axis_sha256(selected_barcodes)
        record["selected_cell_indices_sha256"] = _array_sha256(selected_indices)
        record["authorized_marker_count"] = len(markers)
        record["authorized_marker_axis_sha256"] = _bytes_sha256(
            _canonical_json_bytes(markers)
        )
        gex, gex_audit = _csc_rows(
            handle,
            "matrix",
            (len(gex_names), len(gex_barcodes)),
            selected_indices,
            gex_rows,
            allowlist,
            record,
        )
        output: dict[str, Any] = {
            "rna_counts": gex,
            "selected_barcodes": selected_barcodes,
            "selected_cell_axis_sha256": _axis_sha256(selected_barcodes),
            "selected_cell_indices_sha256": _array_sha256(selected_indices),
            "matched_metadata_barcodes": len(set(gex_barcodes) & eligible),
            "gex_feature_axis_sha256": _axis_sha256(gex_names),
            "gex_access": gex_audit,
        }
        if stage != "predict":
            adt_barcodes = _decode_axis(
                _read_dataset(
                    _dataset(handle, "ADT/barcodes", allowlist, record),
                    "ADT/barcodes",
                    (),
                    record,
                )
            )
            adt_names = _decode_axis(
                _read_dataset(
                    _dataset(handle, "ADT/features/id", allowlist, record),
                    "ADT/features/id",
                    (),
                    record,
                )
            )
            adt_shape_values = _integer_array(
                _read_dataset(
                    _dataset(handle, "ADT/shape", allowlist, record),
                    "ADT/shape",
                    (),
                    record,
                ),
                "ADT_SHAPE",
            )
            if adt_barcodes != gex_barcodes or adt_shape_values.shape != (2,) or tuple(adt_shape_values.tolist()) != (len(adt_names), len(adt_barcodes)):
                raise ProtocolRefusal("ADT_AXIS_SHAPE_OR_CELL_ORDER_MISMATCH")
            adt_rows = _marker_rows(adt_names, markers, "protein")
            adt, adt_audit = _csc_rows(
                handle,
                "ADT",
                (len(adt_names), len(adt_barcodes)),
                selected_indices,
                adt_rows,
                allowlist,
                record,
            )
            output["adt_counts"] = adt
            output["adt_feature_axis_sha256"] = _axis_sha256(adt_names)
            output["adt_access"] = adt_audit
        output["datasets_read"] = list(record["datasets_read"])
    if stage == "predict" and any(path.startswith("ADT/") for path in output["datasets_read"]):
        raise PermissionError("prediction stage accessed an ADT dataset")
    record["h5_reduction_completed"] = True
    return output


def _adt_states(counts: np.ndarray, barcodes: list[str], gsm: str, proteins: list[str]) -> np.ndarray:
    values = np.asarray(counts, dtype=np.int64)
    if values.shape != (CELL_BUDGET, len(proteins)) or len(barcodes) != CELL_BUDGET:
        raise ValueError("ADT state inputs have the wrong shape")
    states = np.zeros_like(values, dtype=np.uint8)
    for marker, protein in enumerate(proteins):
        order = sorted(
            range(CELL_BUDGET),
            key=lambda cell: (
                -int(values[cell, marker]),
                hashlib.sha256(f"{ADT_TIE_SALT}|{gsm}|{protein}|{barcodes[cell]}".encode()).hexdigest(),
                barcodes[cell],
            ),
        )
        states[np.asarray(order[:PROTEIN_HIGH_COUNT]), marker] = 1
    if not np.array_equal(states.sum(axis=0), np.full(len(proteins), PROTEIN_HIGH_COUNT)):
        raise AssertionError("ADT rank state failed its exact margin")
    return states


def _destroyed_states(states: np.ndarray, barcodes: list[str], gsm: str) -> np.ndarray:
    order = np.asarray(
        sorted(
            range(CELL_BUDGET),
            key=lambda cell: (
                hashlib.sha256(f"{DESTROY_SALT}|{gsm}|{barcodes[cell]}".encode()).hexdigest(),
                barcodes[cell],
            ),
        ),
        dtype=np.int64,
    )
    output = np.empty_like(states)
    output[order] = states[np.roll(order, 1)]
    if not np.array_equal(output.sum(axis=0), states.sum(axis=0)):
        raise AssertionError("destroyed-link shift changed a protein margin")
    return output


def _joint_tables(rna_states: np.ndarray, adt_states: np.ndarray) -> np.ndarray:
    first = np.asarray(rna_states, dtype=np.int64)
    second = np.asarray(adt_states, dtype=np.int64)
    if first.ndim != 2 or second.ndim != 2 or first.shape[0] != second.shape[0]:
        raise ValueError("RNA and ADT states must share a cell axis")
    n11 = first.T @ second
    row_one = first.sum(axis=0)[:, None]
    column_one = second.sum(axis=0)[None, :]
    output = np.empty((first.shape[1], second.shape[1], 2, 2), dtype=np.int64)
    output[..., 1, 1] = n11
    output[..., 1, 0] = row_one - n11
    output[..., 0, 1] = column_one - n11
    output[..., 0, 0] = first.shape[0] - row_one - column_one + n11
    if np.any(output < 0) or not np.all(output.sum(axis=(-2, -1)) == first.shape[0]):
        raise AssertionError("binary table construction failed")
    return output


def _poisson_pool_masks(age_groups: list[str]) -> list[np.ndarray]:
    ages = np.asarray(age_groups)
    donors = len(ages)
    masks = [np.ones(donors, dtype=bool), ages == "adult", ages == "pediatric"]
    for held in range(donors):
        training = np.arange(donors) != held
        masks.extend((training, training & (ages == "adult"), training & (ages == "pediatric")))
    return [mask for mask in masks if np.any(mask)]


def _select_markers(
    source_rna_states: np.ndarray,
    source_tables: np.ndarray,
    markers: list[dict[str, Any]],
    age_groups: list[str],
) -> tuple[list[int], list[dict[str, Any]]]:
    rna = np.asarray(source_rna_states, dtype=np.uint8)
    tables = np.asarray(source_tables, dtype=np.int64)
    positives = rna.sum(axis=1)
    eligible = np.all((positives >= 4) & (positives <= 508), axis=0)
    balance = np.min(np.minimum(positives, CELL_BUDGET - positives), axis=0)
    order = sorted(
        np.flatnonzero(eligible).tolist(),
        key=lambda index: (-int(balance[index]), markers[index]["protein"]),
    )
    pools = _poisson_pool_masks(age_groups)
    selected: list[int] = []
    decisions: list[dict[str, Any]] = []
    for index in order:
        proposed = [*selected, index]
        available = all(
            np.all(tables[mask][:, proposed][:, :, proposed].sum(axis=0) > 0)
            for mask in pools
        )
        decisions.append(
            {
                "protein": markers[index]["protein"],
                "rna": markers[index]["rna"],
                "minimum_source_rna_balance": int(balance[index]),
                "accepted": bool(available),
            }
        )
        if available:
            selected.append(index)
        if len(selected) == MAXIMUM_MARKERS:
            break
    if len(selected) < MINIMUM_MARKERS:
        raise ProtocolRefusal(
            "FEWER_THAN_20_POISSON_AVAILABLE_MARKERS",
            {"eligible_before_poisson_availability": int(eligible.sum()), "selected": len(selected)},
        )
    selected_tables = tables[:, selected][:, :, selected]
    if not all(np.all(selected_tables[mask].sum(axis=0) > 0) for mask in pools):
        raise AssertionError("selected axis misses a required Poisson pool")
    return selected, decisions


def _fixed_margins(table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(table, dtype=float)
    return values.sum(axis=-1), values.sum(axis=-2)


def _unique_margin_table(rows: np.ndarray, columns: np.ndarray) -> np.ndarray | None:
    row = np.asarray(rows, dtype=float)
    column = np.asarray(columns, dtype=float)
    total = float(row.sum())
    if row.shape != (2,) or column.shape != (2,) or not np.isclose(total, column.sum()):
        raise ValueError("binary margins are invalid")
    if total <= 0.0 or np.any(row < 0.0) or np.any(column < 0.0):
        raise ValueError("binary margins must be nonnegative with positive total")
    if row[0] == 0.0:
        return np.asarray([[0.0, 0.0], [column[0], column[1]]])
    if row[1] == 0.0:
        return np.asarray([[column[0], column[1]], [0.0, 0.0]])
    if column[0] == 0.0:
        return np.asarray([[0.0, row[0]], [0.0, row[1]]])
    if column[1] == 0.0:
        return np.asarray([[row[0], 0.0], [row[1], 0.0]])
    return None


def _expected_conditional_table(log_odds: float, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Exact noncentral-hypergeometric mean table at integer margins."""

    if not np.isfinite(log_odds):
        raise ProtocolRefusal("NONFINITE_CONDITIONAL_LOG_ODDS")
    row = _integer_array(rows, "ROW_MARGIN")
    column = _integer_array(columns, "COLUMN_MARGIN")
    unique = _unique_margin_table(row, column)
    if unique is not None:
        return unique
    total = int(row.sum())
    lower = max(0, int(row[0] + column[0] - total))
    upper = min(int(row[0]), int(column[0]))
    support = np.arange(lower, upper + 1, dtype=float)
    log_weights = (
        gammaln(column[0] + 1)
        - gammaln(support + 1)
        - gammaln(column[0] - support + 1)
        + gammaln(total - column[0] + 1)
        - gammaln(row[0] - support + 1)
        - gammaln(total - column[0] - row[0] + support + 1)
        + support * float(log_odds)
    )
    probability = np.exp(log_weights - logsumexp(log_weights))
    x00 = float(probability @ support)
    table = np.asarray(
        [
            [x00, float(row[0]) - x00],
            [float(column[0]) - x00, float(total - row[0] - column[0]) + x00],
        ]
    )
    if (
        not np.isfinite(table).all()
        or np.any(table <= 0.0)
        or not np.allclose(table.sum(axis=1), row, atol=1e-10)
        or not np.allclose(table.sum(axis=0), column, atol=1e-10)
    ):
        raise ProtocolRefusal("CONDITIONAL_RECONSTRUCTION_CERTIFICATE_FAILED")
    return table


def _table_log_odds(table: np.ndarray) -> float:
    values = np.asarray(table, dtype=float)
    if values.shape != (2, 2) or np.any(values <= 0.0) or not np.isfinite(values).all():
        raise ProtocolRefusal("POISSON_POOL_NOT_STRICTLY_POSITIVE")
    return float(
        math.log(values[0, 0])
        + math.log(values[1, 1])
        - math.log(values[0, 1])
        - math.log(values[1, 0])
    )


def _fixed_interaction_table(log_odds: float, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Profile Poisson row and column terms at one fixed interaction."""

    row = np.asarray(rows, dtype=float)
    column = np.asarray(columns, dtype=float)
    unique = _unique_margin_table(row, column)
    if unique is not None:
        return unique
    if not np.isfinite(log_odds):
        raise ProtocolRefusal("NONFINITE_POISSON_LOG_ODDS")
    total = float(row.sum())
    lower = max(0.0, float(row[0] + column[0] - total))
    upper = min(float(row[0]), float(column[0]))

    def objective(x00: float) -> float:
        x01 = row[0] - x00
        x10 = column[0] - x00
        x11 = total - row[0] - column[0] + x00
        return math.log(x00) + math.log(x11) - math.log(x01) - math.log(x10) - log_odds

    left = np.nextafter(lower, upper)
    right = np.nextafter(upper, lower)
    x00 = float(brentq(objective, left, right, xtol=5e-15, rtol=1e-14, maxiter=256))
    table = np.asarray(
        [[x00, row[0] - x00], [column[0] - x00, total - row[0] - column[0] + x00]]
    )
    if (
        np.any(table <= 0.0)
        or not np.allclose(table.sum(axis=1), row, atol=1e-10)
        or not np.allclose(table.sum(axis=0), column, atol=1e-10)
        or abs(_table_log_odds(table) - log_odds) > 1e-8
    ):
        raise ProtocolRefusal("POISSON_RECONSTRUCTION_CERTIFICATE_FAILED")
    return table


def _independence_table(rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    row = np.asarray(rows, dtype=float)
    column = np.asarray(columns, dtype=float)
    return np.outer(row, column) / float(row.sum())


def _signed_root_deviance(table: np.ndarray) -> float:
    values = np.asarray(table, dtype=float)
    expected = _independence_table(values.sum(axis=1), values.sum(axis=0))
    positive = values > 0.0
    deviance = 2.0 * float(
        np.sum(
            values[positive]
            * (np.log(values[positive]) - np.log(expected[positive]))
        )
    )
    determinant = float(values[0, 0] * values[1, 1] - values[0, 1] * values[1, 0])
    return float(np.sign(determinant) * math.sqrt(max(0.0, deviance)))


def _signed_root_table(target: float, rows: np.ndarray, columns: np.ndarray) -> tuple[np.ndarray, bool]:
    row = np.asarray(rows, dtype=float)
    column = np.asarray(columns, dtype=float)
    unique = _unique_margin_table(row, column)
    if unique is not None:
        return unique, False
    total = float(row.sum())
    lower = max(0.0, float(row[0] + column[0] - total))
    upper = min(float(row[0]), float(column[0]))
    left = np.nextafter(lower, upper)
    right = np.nextafter(upper, lower)

    def table(x00: float) -> np.ndarray:
        return np.asarray(
            [[x00, row[0] - x00], [column[0] - x00, total - row[0] - column[0] + x00]]
        )

    left_value = _signed_root_deviance(table(left))
    right_value = _signed_root_deviance(table(right))
    clipped = float(np.clip(target, left_value, right_value))
    saturated = clipped != float(target)
    if abs(clipped) <= 1e-14:
        return _independence_table(row, column), saturated
    root = float(
        brentq(
            lambda x00: _signed_root_deviance(table(x00)) - clipped,
            left,
            right,
            xtol=5e-15,
            rtol=1e-14,
            maxiter=256,
        )
    )
    return table(root), saturated


def _loss(observed: np.ndarray, predicted: np.ndarray) -> float:
    truth = np.asarray(observed, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    if truth.shape != estimate.shape or truth.shape[-2:] != (2, 2):
        raise ValueError("truth and prediction shapes differ")
    if (
        not np.isfinite(estimate).all()
        or np.any(estimate < 0.0)
        or not np.allclose(truth.sum(axis=-1), estimate.sum(axis=-1), atol=1e-8)
        or not np.allclose(truth.sum(axis=-2), estimate.sum(axis=-2), atol=1e-8)
    ):
        raise ProtocolRefusal("PREDICTION_MARGIN_OR_FINITE_CERTIFICATE_FAILED")
    positive = truth > 0.0
    if np.any(estimate[positive] <= 0.0):
        return math.inf
    terms = np.zeros_like(truth)
    terms[positive] = truth[positive] * np.log(truth[positive] / estimate[positive])
    return float(np.mean(2.0 * terms.sum(axis=(-2, -1)) / CELL_BUDGET))


def _fit_certificate(fit: ContextConditionalCouplingFit) -> dict[str, Any]:
    maximum_schur = float(np.max(fit.schur_condition_number))
    maximum_donor = float(np.max(fit.donor_curvature_condition_number))
    passes = bool(
        fit.converged
        and fit.scaled_gradient_norm <= GRADIENT_TOLERANCE
        and max(maximum_schur, maximum_donor) <= MAXIMUM_CONDITION_NUMBER
    )
    if not passes:
        raise ProtocolRefusal("CONTEXT_FIT_NUMERICAL_CERTIFICATE_FAILED")
    return {
        "optimizer": fit.optimizer,
        "objective": fit.objective,
        "gradient_norm": fit.gradient_norm,
        "scaled_gradient_norm": fit.scaled_gradient_norm,
        "gradient_tolerance": GRADIENT_TOLERANCE,
        "maximum_schur_condition_number": maximum_schur,
        "maximum_donor_curvature_condition_number": maximum_donor,
        "maximum_condition_number": MAXIMUM_CONDITION_NUMBER,
        "maximum_iterations_used": int(np.max(fit.iterations)),
        "passes": True,
    }


def _context_fit(
    tables: np.ndarray,
    pediatric: np.ndarray,
    deviation_penalty: float,
    age_ridge: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    contexts = np.column_stack((np.ones(len(pediatric)), pediatric.astype(float)))
    fit = fit_context_conditional_log_odds(
        np.asarray(tables, dtype=np.int64),
        contexts,
        donor_deviation_penalty=deviation_penalty,
        coefficient_ridge_penalty=np.asarray((INTERCEPT_RIDGE, age_ridge)),
        minimum_informative_donors=1,
        maximum_condition_number=MAXIMUM_CONDITION_NUMBER,
        tolerance=GRADIENT_TOLERANCE,
    )
    return np.asarray(fit.coefficient, dtype=float), _fit_certificate(fit)


def _predict_conditional(coefficient: np.ndarray, pediatric: bool, transport: float, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    log_odds = predict_context_log_odds(
        coefficient, np.asarray((1.0, float(pediatric)))
    )
    output = np.empty((*log_odds.shape, 2, 2), dtype=float)
    for index in np.ndindex(log_odds.shape):
        output[index] = _expected_conditional_table(
            transport * float(log_odds[index]), rows[index], columns[index]
        )
    return output


def _fit_poisson(tables: np.ndarray) -> np.ndarray:
    pooled = np.asarray(tables, dtype=np.int64).sum(axis=0)
    if np.any(pooled <= 0):
        raise ProtocolRefusal("POISSON_POOL_NOT_STRICTLY_POSITIVE")
    output = np.empty(pooled.shape[:-2], dtype=float)
    for index in np.ndindex(output.shape):
        output[index] = _table_log_odds(pooled[index])
    return output


def _predict_poisson(log_odds: np.ndarray, transport: float, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    output = np.empty((*log_odds.shape, 2, 2), dtype=float)
    for index in np.ndindex(log_odds.shape):
        output[index] = _fixed_interaction_table(
            transport * float(log_odds[index]), rows[index], columns[index]
        )
    return output


def _fit_common_effect(tables: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    fit = fit_common_effect_conditional_log_odds(
        tables,
        minimum_informative_donors=2,
        tolerance=1e-10,
    )
    certificate = {
        "estimator": "unregularized_delta_free_exact_conditional_cmle",
        "objective": fit.objective,
        "gradient_norm": fit.gradient_norm,
        "scaled_gradient_norm": fit.scaled_gradient_norm,
        "gradient_tolerance": 1e-10,
        "minimum_data_precision": float(np.min(fit.data_precision)),
        "minimum_support_count": int(np.min(fit.support_count)),
        "maximum_root_iterations": int(np.max(fit.root_iterations)),
        "passes": bool(fit.converged and fit.scaled_gradient_norm <= 1e-10),
    }
    if not certificate["passes"]:
        raise ProtocolRefusal("COMMON_EFFECT_NUMERICAL_CERTIFICATE_FAILED")
    return np.asarray(fit.log_odds, dtype=float), certificate


def _fit_signed_root(tables: np.ndarray) -> np.ndarray:
    values = np.asarray(tables, dtype=np.int64)
    output = np.empty(values.shape[:-2], dtype=float)
    for index in np.ndindex(output.shape):
        output[index] = _signed_root_deviance(values[index]) / math.sqrt(CELL_BUDGET)
    return output.mean(axis=0)


def _predict_signed_root(coordinate: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> tuple[np.ndarray, int]:
    output = np.empty((*coordinate.shape, 2, 2), dtype=float)
    saturated = 0
    for index in np.ndindex(coordinate.shape):
        output[index], clipped = _signed_root_table(
            float(coordinate[index]) * math.sqrt(CELL_BUDGET), rows[index], columns[index]
        )
        saturated += int(clipped)
    return output, saturated


def _rows_columns_from_positive(positive: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(positive, dtype=np.int64)
    rows = np.stack((CELL_BUDGET - counts, counts), axis=-1)
    columns = np.broadcast_to(
        np.asarray((CELL_BUDGET - PROTEIN_HIGH_COUNT, PROTEIN_HIGH_COUNT)),
        (*counts.shape, 2),
    ).copy()
    return rows, columns


def _select_lowest(losses: dict[tuple[float, ...], np.ndarray]) -> tuple[tuple[float, ...], np.ndarray]:
    complete = [(configuration, values) for configuration, values in losses.items() if np.isfinite(values).all()]
    if not complete:
        raise ProtocolRefusal("NO_COMPLETE_SOURCE_CONFIGURATION")
    selected, values = min(complete, key=lambda item: (float(np.mean(item[1])), item[0]))
    return selected, values


def _loss_grid_payload(
    losses: dict[tuple[float, ...], np.ndarray], names: tuple[str, ...]
) -> list[dict[str, Any]]:
    return [
        {
            "configuration": dict(zip(names, configuration)),
            "donor_losses": np.asarray(losses[configuration], dtype=float).tolist(),
        }
        for configuration in sorted(losses)
    ]


def _source_models(tables: np.ndarray, destroyed: np.ndarray, age_groups: list[str]) -> dict[str, Any]:
    values = np.asarray(tables, dtype=np.int64)
    destroyed_values = np.asarray(destroyed, dtype=np.int64)
    pediatric = np.asarray([group == "pediatric" for group in age_groups], dtype=bool)
    donors = len(values)
    primary_losses = {
        (deviation, age_ridge, transport): np.full(donors, np.nan)
        for deviation in DEVIATION_GRID
        for age_ridge in AGE_RIDGE_GRID
        for transport in PRIMARY_TRANSPORT_GRID
    }
    poisson_losses = {
        (transport,): np.full(donors, np.nan) for transport in POISSON_TRANSPORT_GRID
    }
    age_poisson_losses = {
        (transport,): np.full(donors, np.nan) for transport in POISSON_TRANSPORT_GRID
    }
    fold_certificates: list[dict[str, Any]] = []
    for held in range(donors):
        training = np.arange(donors) != held
        truth = values[held]
        rows, columns = _fixed_margins(truth)
        fold_certificate: dict[str, Any] = {"held_source_donor_index": held, "primary": []}
        for deviation in DEVIATION_GRID:
            for age_ridge in AGE_RIDGE_GRID:
                coefficient, certificate = _context_fit(
                    values[training], pediatric[training], deviation, age_ridge
                )
                fold_certificate["primary"].append(
                    {"deviation_penalty": deviation, "age_ridge": age_ridge, **certificate}
                )
                for transport in PRIMARY_TRANSPORT_GRID:
                    prediction = _predict_conditional(
                        coefficient, bool(pediatric[held]), transport, rows, columns
                    )
                    primary_losses[(deviation, age_ridge, transport)][held] = _loss(truth, prediction)
        poisson = _fit_poisson(values[training])
        for transport in POISSON_TRANSPORT_GRID:
            poisson_losses[(transport,)][held] = _loss(
                truth, _predict_poisson(poisson, transport, rows, columns)
            )
        age_stratum = training & (pediatric == pediatric[held])
        age_poisson = _fit_poisson(values[age_stratum])
        for transport in POISSON_TRANSPORT_GRID:
            age_poisson_losses[(transport,)][held] = _loss(
                truth, _predict_poisson(age_poisson, transport, rows, columns)
            )
        fold_certificates.append(fold_certificate)

    primary_configuration, selected_primary_losses = _select_lowest(primary_losses)
    poisson_configuration, selected_poisson_losses = _select_lowest(poisson_losses)
    age_poisson_configuration, selected_age_poisson_losses = _select_lowest(
        age_poisson_losses
    )
    deviation, age_ridge, primary_transport = primary_configuration
    poisson_transport = poisson_configuration[0]
    age_poisson_transport = age_poisson_configuration[0]
    destroyed_losses = np.full(donors, np.nan)
    for held in range(donors):
        training = np.arange(donors) != held
        truth = values[held]
        rows, columns = _fixed_margins(truth)
        coefficient, _ = _context_fit(
            destroyed_values[training], pediatric[training], deviation, age_ridge
        )
        destroyed_losses[held] = _loss(
            truth,
            _predict_conditional(
                coefficient, bool(pediatric[held]), primary_transport, rows, columns
            ),
        )

    primary_mean = float(np.mean(selected_primary_losses))
    poisson_mean = float(np.mean(selected_poisson_losses))
    destroyed_mean = float(np.mean(destroyed_losses))
    promotion = {
        "primary_mean_deviance": primary_mean,
        "pooled_poisson_mean_deviance": poisson_mean,
        "destroyed_link_mean_deviance": destroyed_mean,
        "primary_favorable_donors_vs_pooled_poisson": int(np.sum(selected_primary_losses < selected_poisson_losses)),
        "primary_favorable_donors_vs_destroyed_link": int(np.sum(selected_primary_losses < destroyed_losses)),
        "primary_below_pooled_poisson": bool(primary_mean < poisson_mean),
        "primary_below_destroyed_link": bool(primary_mean < destroyed_mean),
    }
    promotion["passes"] = bool(
        promotion["primary_below_pooled_poisson"]
        and promotion["primary_below_destroyed_link"]
        and promotion["primary_favorable_donors_vs_pooled_poisson"] >= 6
        and promotion["primary_favorable_donors_vs_destroyed_link"] >= 6
    )
    if not promotion["passes"]:
        raise ProtocolRefusal("SOURCE_PROMOTION_GATE_FAILED", promotion)

    primary_coefficient, primary_certificate = _context_fit(values, pediatric, deviation, age_ridge)
    destroyed_coefficient, destroyed_certificate = _context_fit(
        destroyed_values, pediatric, deviation, age_ridge
    )
    pooled_poisson = _fit_poisson(values)
    age_poisson = {
        "adult": _fit_poisson(values[~pediatric]),
        "pediatric": _fit_poisson(values[pediatric]),
    }
    common, common_certificate = _fit_common_effect(values)
    signed_root = _fit_signed_root(values)
    models = {
        "primary": {
            "coefficient": primary_coefficient.tolist(),
            "coefficient_sha256": _array_sha256(primary_coefficient),
            "deviation_penalty": deviation,
            "intercept_ridge": INTERCEPT_RIDGE,
            "age_ridge": age_ridge,
            "transport": primary_transport,
            "fit_certificate": primary_certificate,
        },
        "destroyed_link": {
            "coefficient": destroyed_coefficient.tolist(),
            "coefficient_sha256": _array_sha256(destroyed_coefficient),
            "deviation_penalty": deviation,
            "intercept_ridge": INTERCEPT_RIDGE,
            "age_ridge": age_ridge,
            "transport": primary_transport,
            "fit_certificate": destroyed_certificate,
        },
        "pooled_fixed_interaction_poisson": {
            "log_odds": pooled_poisson.tolist(),
            "log_odds_sha256": _array_sha256(pooled_poisson),
            "transport": poisson_transport,
            "pseudocount": None,
        },
        "age_stratified_fixed_interaction_poisson": {
            "transport": age_poisson_transport,
            **{
                group: {
                    "log_odds": coordinate.tolist(),
                    "log_odds_sha256": _array_sha256(coordinate),
                }
                for group, coordinate in age_poisson.items()
            },
        },
        "common_effect_exact_conditional": {
            "log_odds": common.tolist(),
            "log_odds_sha256": _array_sha256(common),
            "transport": 1.0,
            "fit_certificate": common_certificate,
        },
        "signed_root_deviance": {
            "coordinate_per_sqrt_n": signed_root.tolist(),
            "coordinate_sha256": _array_sha256(signed_root),
            "boundary_rule": "saturate at nearest closed attainable fixed-margin endpoint",
        },
        "independence": {"kind": "recipient_fixed_margin_independence"},
    }
    return {
        "models": models,
        "promotion": promotion,
        "source_cross_validation": {
            "selected_primary_configuration": {
                "deviation_penalty": deviation,
                "age_ridge": age_ridge,
                "transport": primary_transport,
            },
            "selected_poisson_transport": poisson_transport,
            "selected_age_stratified_poisson_transport": age_poisson_transport,
            "losses": {
                "primary": selected_primary_losses.tolist(),
                "pooled_fixed_interaction_poisson": selected_poisson_losses.tolist(),
                "age_stratified_fixed_interaction_poisson": selected_age_poisson_losses.tolist(),
                "destroyed_link": destroyed_losses.tolist(),
            },
            "complete_loss_grids": {
                "primary": _loss_grid_payload(
                    primary_losses,
                    ("deviation_penalty", "age_ridge", "transport"),
                ),
                "pooled_fixed_interaction_poisson": _loss_grid_payload(
                    poisson_losses, ("transport",)
                ),
                "age_stratified_fixed_interaction_poisson": _loss_grid_payload(
                    age_poisson_losses, ("transport",)
                ),
            },
            "fold_fit_certificates": fold_certificates,
            "selection_unit": "donor_equal mean multinomial deviance per cell on the full-source support-screened axis",
            "cross_validation_scope": "transductive support-screened LOODO configuration selection; the marker axis is fixed once from all eight source donors before folds",
            "tie_break": "lexicographic_parameter_order",
        },
    }


def _reduce_sample(
    sample: dict[str, Any],
    markers: list[dict[str, Any]],
    stage: str,
    scratch: Path,
    audit: list[dict[str, Any]],
    *,
    expected_metadata_sha256: str | None = None,
    expected_h5_sha256: str | None = None,
) -> dict[str, Any]:
    inventory = sample["gsm"] == "GSM6611363"
    metadata_path = _fetch(
        sample["metadata_url"],
        int(sample["metadata_bytes"]),
        sample["metadata_filename"],
        scratch,
        audit,
        expected_sha256=(
            "9152caa317ed1d2eeb50dc6e36034de969f6a82ed24bd987eba31764d5d4eab4"
            if inventory
            else expected_metadata_sha256
        ),
    )
    metadata_record = audit[-1]
    try:
        metadata_record["decode_started"] = True
        metadata_record["decode_completed"] = False
        eligible, metadata_audit = _eligible_metadata(metadata_path, sample)
        metadata_record["decode"] = metadata_audit
        metadata_record["decode_completed"] = True
    finally:
        _delete_download(metadata_path, metadata_record)

    h5_path = _fetch(
        sample["h5_url"],
        int(sample["h5_bytes"]),
        sample["h5_filename"],
        scratch,
        audit,
        expected_sha256=(
            "fb7b1fddf5f21e8a7e0377911dc86b28c69c2505650de9061f64eb1871f9a9dd"
            if inventory
            else expected_h5_sha256
        ),
    )
    h5_record = audit[-1]
    try:
        reduced = _read_h5(
            h5_path, eligible, sample, markers, stage, audit_record=h5_record
        )
    finally:
        _delete_download(h5_path, h5_record)
    if any(scratch.iterdir()):
        raise PermissionError("sequential reduction left a scratch file behind")
    reduced["metadata_download_sha256"] = metadata_record["sha256"]
    reduced["metadata_download_bytes"] = metadata_record["observed_bytes"]
    reduced["h5_download_sha256"] = h5_record["sha256"]
    reduced["h5_download_bytes"] = h5_record["observed_bytes"]
    return reduced


def _failure_payload(
    schema: str,
    status: str,
    stage: str,
    error: BaseException,
    audit: list[dict[str, Any]],
    bindings: dict[str, Any],
) -> dict[str, Any]:
    reason = error.reason_code if isinstance(error, ProtocolRefusal) else type(error).__name__
    details = error.details if isinstance(error, ProtocolRefusal) else {"message": str(error)}
    return {
        "schema": schema,
        "status": status,
        "stage": stage,
        "created_at_utc": _timestamp(),
        "reason_code": reason,
        "details": details,
        "bindings": bindings,
        "access_audit": audit,
        "held_h5_requested": any(
            record.get("request_started")
            and any(sample["gsm"] in record.get("filename", "") for sample in _contract()["held"])
            for record in audit
        ),
        "rerun_permitted": False,
    }


def _downstream_paths() -> tuple[Path, ...]:
    return (
        SOURCE_RESULT,
        PREDICTION_ATTEMPT,
        PREDICTION_RESULT,
        SCORE_AUTHORIZATION,
        SCORE_ATTEMPT,
        SCORE_RESULT,
    )


def _frozen_contract_bindings(tags: dict[str, str]) -> dict[str, Any]:
    return {
        "candidate_sha256": CANDIDATE_SHA256,
        "amendment_sha256": AMENDMENT_SHA256,
        "implementation_clarification_sha256": IMPLEMENTATION_CLARIFICATION_SHA256,
        "cv_availability_clarification_sha256": CV_AVAILABILITY_SHA256,
        "normalization_correction_sha256": NORMALIZATION_CORRECTION_SHA256,
        "sparse_access_clarification_sha256": SPARSE_ACCESS_CLARIFICATION_SHA256,
        "crash_semantics_clarification_sha256": CRASH_SEMANTICS_CLARIFICATION_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation": _implementation_bindings(),
        "public_tags": tags,
    }


def _validate_claim_token_hash(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PermissionError("public attempt claim-token hash is invalid")
    return value


def claim_source(*, claim_token: Path) -> dict[str, Any]:
    """Create the public source attempt before any source-file request."""

    if SOURCE_ATTEMPT.exists() or any(path.exists() for path in _downstream_paths()):
        raise FileExistsError("GSE214546 source campaign has already been claimed or advanced")
    contract = _contract()
    tags = _verify_implementation_freeze()
    token_sha256 = _create_claim_token(claim_token)
    attempt = {
        "schema": "gse214546-teaseq-source-attempt/1.0",
        "status": "CLAIMED_BEFORE_FIRST_SOURCE_FILE_GET",
        "created_at_utc": _timestamp(),
        "bindings": _frozen_contract_bindings(tags),
        "source_gsms": [sample["gsm"] for sample in contract["source"]],
        "held_gsms": [sample["gsm"] for sample in contract["held"]],
        "source_h5_bytes": sum(int(sample["h5_bytes"]) for sample in contract["source"]),
        "claim_token_sha256": token_sha256,
        "held_numeric_access_authorized": False,
        "runtime": _runtime(),
        "rerun_permitted": False,
    }
    _write_json_x(SOURCE_ATTEMPT, attempt)
    return attempt


def _validate_source_attempt() -> tuple[dict[str, Any], str]:
    contract = _contract()
    tags = _verify_implementation_freeze()
    commit = _require_tag_paths(SOURCE_ATTEMPT_TAG, (SOURCE_ATTEMPT,))
    _require_ancestor(tags["implementation_commit"], commit)
    attempt = _read_json(SOURCE_ATTEMPT)
    if (
        attempt.get("schema") != "gse214546-teaseq-source-attempt/1.0"
        or attempt.get("status") != "CLAIMED_BEFORE_FIRST_SOURCE_FILE_GET"
        or attempt.get("bindings") != _frozen_contract_bindings(tags)
        or attempt.get("source_gsms")
        != [sample["gsm"] for sample in contract["source"]]
        or attempt.get("held_gsms") != [sample["gsm"] for sample in contract["held"]]
        or attempt.get("source_h5_bytes")
        != sum(int(sample["h5_bytes"]) for sample in contract["source"])
        or attempt.get("held_numeric_access_authorized") is not False
        or attempt.get("rerun_permitted") is not False
    ):
        raise PermissionError("public source attempt differs from the frozen contract")
    _validate_claim_token_hash(attempt.get("claim_token_sha256"))
    return attempt, commit


def run_source(
    *, scratch: Path = DEFAULT_SCRATCH, claim_token: Path | None = None
) -> dict[str, Any]:
    """Acquire, reduce, fit, and terminally decide the frozen source stage."""

    attempt, attempt_commit = _validate_source_attempt()
    if any(path.exists() for path in _downstream_paths()):
        raise FileExistsError("GSE214546 source campaign has already advanced")
    contract = _contract()
    scratch_path = _scratch(scratch)
    bindings = {
        **attempt["bindings"],
        "source_attempt_tag": SOURCE_ATTEMPT_TAG,
        "source_attempt_commit": attempt_commit,
        "source_attempt_sha256": _sha256(SOURCE_ATTEMPT),
        "source_attempt_bytes": SOURCE_ATTEMPT.stat().st_size,
        "claim_token_sha256": attempt["claim_token_sha256"],
    }
    _consume_claim_token(claim_token, attempt["claim_token_sha256"])
    audit: list[dict[str, Any]] = []
    try:
        reductions = [
            _reduce_sample(sample, contract["markers"], "source", scratch_path, audit)
            for sample in contract["source"]
        ]
        source_rna = np.asarray(
            [reduction["rna_counts"] > 0 for reduction in reductions], dtype=np.uint8
        )
        source_adt = np.asarray(
            [
                _adt_states(
                    reduction["adt_counts"],
                    reduction["selected_barcodes"],
                    sample["gsm"],
                    [marker["protein"] for marker in contract["markers"]],
                )
                for sample, reduction in zip(contract["source"], reductions)
            ],
            dtype=np.uint8,
        )
        destroyed_adt = np.asarray(
            [
                _destroyed_states(
                    states, reduction["selected_barcodes"], sample["gsm"]
                )
                for sample, reduction, states in zip(contract["source"], reductions, source_adt)
            ],
            dtype=np.uint8,
        )
        all_tables = np.asarray(
            [_joint_tables(rna, adt) for rna, adt in zip(source_rna, source_adt)]
        )
        selected, axis_decisions = _select_markers(
            source_rna,
            all_tables,
            contract["markers"],
            [sample["age_group"] for sample in contract["source"]],
        )
        selected_markers = [contract["markers"][index] for index in selected]
        tables = all_tables[:, selected][:, :, selected]
        destroyed_tables = np.asarray(
            [
                _joint_tables(rna[:, selected], adt[:, selected])
                for rna, adt in zip(source_rna, destroyed_adt)
            ]
        )
        fitted = _source_models(
            tables,
            destroyed_tables,
            [sample["age_group"] for sample in contract["source"]],
        )
        source_records = []
        for sample, reduction in zip(contract["source"], reductions):
            selected_positive = (reduction["rna_counts"][:, selected] > 0).sum(axis=0)
            source_records.append(
                {
                    "gsm": sample["gsm"],
                    "donor": sample["donor"],
                    "age_group": sample["age_group"],
                    "batch": sample["batch"],
                    "cmv_status": sample["cmv_status"],
                    "selected_cell_axis_sha256": reduction["selected_cell_axis_sha256"],
                    "selected_cell_indices_sha256": reduction["selected_cell_indices_sha256"],
                    "rna_positive_counts": selected_positive.tolist(),
                    "rna_positive_counts_sha256": _array_sha256(selected_positive),
                    "metadata_download_sha256": reduction["metadata_download_sha256"],
                    "metadata_download_bytes": reduction["metadata_download_bytes"],
                    "h5_download_sha256": reduction["h5_download_sha256"],
                    "h5_download_bytes": reduction["h5_download_bytes"],
                    "selected_cells": CELL_BUDGET,
                }
            )
        result = {
            "schema": "gse214546-teaseq-source-result/1.0",
            "status": "SOURCE_PROMOTED",
            "created_at_utc": _timestamp(),
            "bindings": bindings,
            "runtime": _runtime(),
            "selected_markers": selected_markers,
            "selected_marker_count": len(selected),
            "selected_marker_source_indices": selected,
            "selected_marker_axis_sha256": _bytes_sha256(_canonical_json_bytes(selected_markers)),
            "axis_selection": {
                "rule": "source support then ranked greedy complete-cross-product Poisson availability",
                "poisson_pools": "full and every LOODO training global/adult/pediatric pool",
                "pseudocount": None,
                "decisions": axis_decisions,
            },
            "source_records": source_records,
            **fitted,
            "access_audit": audit,
            "held_h5_requested": False,
            "held_adt_value_access_authorized": False,
            "rerun_permitted": False,
        }
    except BaseException as error:
        result = _failure_payload(
            "gse214546-teaseq-source-result/1.0",
            "TERMINAL_SOURCE_REFUSAL",
            "source",
            error,
            audit,
            bindings,
        )
    _write_json_x(SOURCE_RESULT, result)
    return result


def _public_result(
    tag: str, path: Path, predecessor: str, *, bound_paths: tuple[Path, ...] = ()
) -> tuple[dict[str, Any], str]:
    commit = _require_tag_paths(tag, (*bound_paths, path))
    _require_ancestor(predecessor, commit)
    return _read_json(path), commit


def _validate_model_array(record: dict[str, Any], value_key: str, hash_key: str, shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(record.get(value_key), dtype=float)
    if values.shape != shape or not np.isfinite(values).all() or record.get(hash_key) != _array_sha256(values):
        raise PermissionError(f"frozen model array differs: {value_key}")
    return values


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_fit_certificate(value: Any) -> None:
    if not isinstance(value, dict):
        raise PermissionError("fit certificate is absent")
    condition_numbers = (
        value.get("maximum_schur_condition_number"),
        value.get("maximum_donor_curvature_condition_number"),
    )
    if (
        value.get("optimizer") != "coordinate_block_newton_schur_backtracking"
        or value.get("passes") is not True
        or value.get("gradient_tolerance") != GRADIENT_TOLERANCE
        or value.get("maximum_condition_number") != MAXIMUM_CONDITION_NUMBER
        or not math.isfinite(float(value.get("scaled_gradient_norm", math.inf)))
        or float(value["scaled_gradient_norm"]) > GRADIENT_TOLERANCE
        or any(
            not math.isfinite(float(number))
            or float(number) > MAXIMUM_CONDITION_NUMBER
            for number in condition_numbers
        )
    ):
        raise PermissionError("fit certificate does not pass the frozen tolerances")


def _grid_records(
    value: Any, names: tuple[str, ...], configurations: list[tuple[float, ...]]
) -> dict[tuple[float, ...], np.ndarray]:
    if not isinstance(value, list) or len(value) != len(configurations):
        raise PermissionError("complete source CV grid is absent")
    observed: dict[tuple[float, ...], np.ndarray] = {}
    for record in value:
        if not isinstance(record, dict) or set(record) != {"configuration", "donor_losses"}:
            raise PermissionError("source CV grid record has an invalid schema")
        configuration_object = record["configuration"]
        if not isinstance(configuration_object, dict) or set(configuration_object) != set(names):
            raise PermissionError("source CV grid parameters differ")
        configuration = tuple(float(configuration_object[name]) for name in names)
        losses = np.asarray(record["donor_losses"], dtype=float)
        if configuration in observed or losses.shape != (8,) or not np.isfinite(losses).all():
            raise PermissionError("source CV grid losses are incomplete or nonfinite")
        observed[configuration] = losses
    if set(observed) != set(configurations):
        raise PermissionError("source CV grid configurations differ from the freeze")
    return observed


def _validate_download_audit(
    audit: Any,
    samples: list[dict[str, Any]],
    stage: str,
    markers: list[dict[str, Any]],
    *,
    expected_metadata_sha256: dict[str, str] | None = None,
    expected_h5_sha256: dict[str, str] | None = None,
    expected_cell_axis_sha256: dict[str, str] | None = None,
) -> None:
    if not isinstance(audit, list) or len(audit) != 2 * len(samples):
        raise PermissionError("download audit does not contain one sequential file pair per donor")
    allowed = PREDICT_H5_ALLOWLIST if stage == "predict" else LINKED_H5_ALLOWLIST
    marker_count = len(markers)
    marker_axis_sha256 = _bytes_sha256(_canonical_json_bytes(markers))
    for donor_index, sample in enumerate(samples):
        metadata, h5_record = audit[2 * donor_index : 2 * donor_index + 2]
        for record, kind in ((metadata, "metadata"), (h5_record, "h5")):
            expected_filename = sample[f"{kind}_filename"]
            expected_bytes = int(sample[f"{kind}_bytes"])
            if (
                record.get("filename") != expected_filename
                or record.get("url") != sample[f"{kind}_url"]
                or record.get("expected_bytes") != expected_bytes
                or record.get("observed_bytes") != expected_bytes
                or record.get("completed") is not True
                or record.get("deleted") is not True
                or not _is_sha256(record.get("sha256"))
                or "partial_sha256" in record
            ):
                raise PermissionError("download audit is incomplete or differs from the frozen object")
        if (
            metadata.get("decode_started") is not True
            or metadata.get("decode_completed") is not True
            or metadata.get("datasets_read") != []
            or metadata.get("decode", {}).get("barcode_column") != "barcodes"
            or metadata.get("decode", {}).get("singlet_column") != "singlet"
            or metadata.get("decode", {}).get("singlet_value") != "TRUE"
        ):
            raise PermissionError("metadata audit differs from the literal singlet rule")
        if (
            expected_metadata_sha256 is not None
            and metadata.get("sha256") != expected_metadata_sha256[sample["gsm"]]
        ):
            raise PermissionError("held metadata bytes differ from the prediction freeze")
        if expected_h5_sha256 is not None and h5_record.get("sha256") != expected_h5_sha256[sample["gsm"]]:
            raise PermissionError("held H5 bytes differ from the prediction freeze")
        if (
            h5_record.get("h5_stage") != stage
            or h5_record.get("h5_open_started") is not True
            or h5_record.get("h5_open_completed") is not True
            or h5_record.get("h5_reduction_completed") is not True
            or h5_record.get("selected_cells") != CELL_BUDGET
            or not _is_sha256(h5_record.get("selected_cell_axis_sha256"))
            or not _is_sha256(h5_record.get("selected_cell_indices_sha256"))
            or h5_record.get("authorized_marker_count") != marker_count
            or h5_record.get("authorized_marker_axis_sha256") != marker_axis_sha256
            or set(h5_record.get("datasets_opened", [])) != set(allowed)
        ):
            raise PermissionError("HDF5 audit is incomplete or violates the stage allowlist")
        if (
            expected_cell_axis_sha256 is not None
            and h5_record.get("selected_cell_axis_sha256")
            != expected_cell_axis_sha256[sample["gsm"]]
        ):
            raise PermissionError("held cell axis differs from the prediction freeze")
        events = h5_record.get("dataset_access_events")
        if not isinstance(events, list) or not events or any(
            event.get("started") is not True or event.get("completed") is not True
            for event in events
        ):
            raise PermissionError("HDF5 access event journal is incomplete")
        for event in events:
            path = event.get("dataset")
            selection = event.get("selection", {})
            if path not in allowed:
                raise PermissionError("HDF5 audit records a forbidden dataset")
            if path.endswith("/indptr"):
                if selection.get("kind") != "slice" or selection.get("step") is not None or selection.get("stop") - selection.get("start") != 2:
                    raise PermissionError("CSC indptr access was not a selected-column pair")
            elif path.endswith("/indices"):
                if selection.get("kind") != "slice" or selection.get("step") is not None:
                    raise PermissionError("CSC indices access was not a selected-column span")
            elif path.endswith("/data"):
                if selection.get("kind") != "fancy_positions":
                    raise PermissionError("CSC data access included unselected sparse values")
            elif selection.get("kind") != "all":
                raise PermissionError("HDF5 axis access has an unexpected selection")
        summaries = (
            (("gex_access", "matrix/data"),)
            if stage == "predict"
            else (("gex_access", "matrix/data"), ("adt_access", "ADT/data"))
        )
        expected_read = set(allowed)
        for key, data_path in summaries:
            summary = h5_record.get(key, {})
            if (
                summary.get("completed") is not True
                or summary.get("selected_columns") != CELL_BUDGET
                or summary.get("selected_rows") != marker_count
                or summary.get("indptr_values_decoded") != 2 * CELL_BUDGET
                or summary.get("full_sparse_data_read") is not False
                or not _is_sha256(summary.get("selected_column_indices_sha256"))
                or not _is_sha256(summary.get("selected_feature_rows_sha256"))
            ):
                raise PermissionError("sparse selected-row audit is incomplete")
            if summary.get("selected_data_values_decoded") == 0:
                expected_read.remove(data_path)
            elif not isinstance(summary.get("selected_data_values_decoded"), int) or summary["selected_data_values_decoded"] < 0:
                raise PermissionError("sparse selected-value audit is invalid")
        if set(h5_record.get("datasets_read", [])) != expected_read:
            raise PermissionError("HDF5 datasets read do not match selected sparse values")
        if stage == "predict" and (
            "adt_access" in h5_record
            or any(path.startswith("ADT/") for path in h5_record.get("datasets_opened", []))
        ):
            raise PermissionError("prediction audit crossed the held ADT firewall")


def _validate_source_cross_validation(source: dict[str, Any]) -> None:
    cross_validation = source.get("source_cross_validation", {})
    if (
        cross_validation.get("selection_unit")
        != "donor_equal mean multinomial deviance per cell on the full-source support-screened axis"
        or cross_validation.get("cross_validation_scope")
        != "transductive support-screened LOODO configuration selection; the marker axis is fixed once from all eight source donors before folds"
        or cross_validation.get("tie_break") != "lexicographic_parameter_order"
    ):
        raise PermissionError("source CV scope or tie rule differs from the freeze")
    primary_configurations = [
        (deviation, age_ridge, transport)
        for deviation in DEVIATION_GRID
        for age_ridge in AGE_RIDGE_GRID
        for transport in PRIMARY_TRANSPORT_GRID
    ]
    poisson_configurations = [(transport,) for transport in POISSON_TRANSPORT_GRID]
    grids = cross_validation.get("complete_loss_grids", {})
    primary = _grid_records(
        grids.get("primary"),
        ("deviation_penalty", "age_ridge", "transport"),
        primary_configurations,
    )
    pooled = _grid_records(
        grids.get("pooled_fixed_interaction_poisson"),
        ("transport",),
        poisson_configurations,
    )
    age = _grid_records(
        grids.get("age_stratified_fixed_interaction_poisson"),
        ("transport",),
        poisson_configurations,
    )
    selected_primary, primary_losses = _select_lowest(primary)
    selected_pooled, pooled_losses = _select_lowest(pooled)
    selected_age, age_losses = _select_lowest(age)
    selected_object = cross_validation.get("selected_primary_configuration", {})
    if (
        selected_object
        != dict(zip(("deviation_penalty", "age_ridge", "transport"), selected_primary))
        or cross_validation.get("selected_poisson_transport") != selected_pooled[0]
        or cross_validation.get("selected_age_stratified_poisson_transport") != selected_age[0]
    ):
        raise PermissionError("selected source configuration is not the lexicographic CV optimum")
    selected_losses = cross_validation.get("losses", {})
    destroyed_losses = np.asarray(selected_losses.get("destroyed_link"), dtype=float)
    if (
        not np.array_equal(np.asarray(selected_losses.get("primary"), dtype=float), primary_losses)
        or not np.array_equal(np.asarray(selected_losses.get("pooled_fixed_interaction_poisson"), dtype=float), pooled_losses)
        or not np.array_equal(np.asarray(selected_losses.get("age_stratified_fixed_interaction_poisson"), dtype=float), age_losses)
        or destroyed_losses.shape != (8,)
        or not np.isfinite(destroyed_losses).all()
    ):
        raise PermissionError("selected source losses do not match the complete grids")
    promotion = source.get("promotion", {})
    expected = {
        "primary_mean_deviance": float(primary_losses.mean()),
        "pooled_poisson_mean_deviance": float(pooled_losses.mean()),
        "destroyed_link_mean_deviance": float(destroyed_losses.mean()),
        "primary_favorable_donors_vs_pooled_poisson": int(np.sum(primary_losses < pooled_losses)),
        "primary_favorable_donors_vs_destroyed_link": int(np.sum(primary_losses < destroyed_losses)),
        "primary_below_pooled_poisson": bool(primary_losses.mean() < pooled_losses.mean()),
        "primary_below_destroyed_link": bool(primary_losses.mean() < destroyed_losses.mean()),
    }
    expected["passes"] = bool(
        expected["primary_below_pooled_poisson"]
        and expected["primary_below_destroyed_link"]
        and expected["primary_favorable_donors_vs_pooled_poisson"] >= 6
        and expected["primary_favorable_donors_vs_destroyed_link"] >= 6
    )
    if promotion != expected or expected["passes"] is not True:
        raise PermissionError("source promotion arithmetic does not reproduce")
    certificates = cross_validation.get("fold_fit_certificates")
    if not isinstance(certificates, list) or len(certificates) != 8:
        raise PermissionError("source LOODO fit certificates are incomplete")
    expected_pairs = {(deviation, age_ridge) for deviation in DEVIATION_GRID for age_ridge in AGE_RIDGE_GRID}
    for held, certificate in enumerate(certificates):
        fits = certificate.get("primary") if isinstance(certificate, dict) else None
        if certificate.get("held_source_donor_index") != held or not isinstance(fits, list) or len(fits) != len(expected_pairs):
            raise PermissionError("source LOODO fit certificate order differs")
        if {(fit.get("deviation_penalty"), fit.get("age_ridge")) for fit in fits} != expected_pairs:
            raise PermissionError("source LOODO fit certificate grid differs")
        for fit in fits:
            _validate_fit_certificate(fit)


def _validate_source_result(*, require_public: bool = True) -> tuple[dict[str, Any], str]:
    contract = _contract()
    attempt, attempt_commit = _validate_source_attempt()
    if require_public:
        source, commit = _public_result(
            SOURCE_TAG,
            SOURCE_RESULT,
            attempt_commit,
            bound_paths=(SOURCE_ATTEMPT,),
        )
    else:
        source, commit = _read_json(SOURCE_RESULT), attempt_commit
    expected_bindings = {
        **attempt["bindings"],
        "source_attempt_tag": SOURCE_ATTEMPT_TAG,
        "source_attempt_commit": attempt_commit,
        "source_attempt_sha256": _sha256(SOURCE_ATTEMPT),
        "source_attempt_bytes": SOURCE_ATTEMPT.stat().st_size,
        "claim_token_sha256": attempt["claim_token_sha256"],
    }
    if (
        source.get("schema") != "gse214546-teaseq-source-result/1.0"
        or source.get("status") != "SOURCE_PROMOTED"
        or source.get("bindings") != expected_bindings
        or source.get("promotion", {}).get("passes") is not True
        or source.get("held_h5_requested") is not False
        or source.get("held_adt_value_access_authorized") is not False
        or source.get("rerun_permitted") is not False
    ):
        raise PermissionError("public source result is not a promoted frozen result")
    markers = source.get("selected_markers")
    count = source.get("selected_marker_count")
    indices = source.get("selected_marker_source_indices")
    if (
        not isinstance(markers, list)
        or not isinstance(count, int)
        or not MINIMUM_MARKERS <= count <= MAXIMUM_MARKERS
        or len(markers) != count
        or not isinstance(indices, list)
        or len(indices) != count
        or len(set(indices)) != count
        or any(not isinstance(index, int) or index < 0 or index >= len(contract["markers"]) for index in indices)
        or markers != [contract["markers"][index] for index in indices]
        or source.get("selected_marker_axis_sha256")
        != _bytes_sha256(_canonical_json_bytes(markers))
    ):
        raise PermissionError("public source marker axis is invalid")
    axis_selection = source.get("axis_selection", {})
    decisions = axis_selection.get("decisions")
    decision_indices = []
    if isinstance(decisions, list):
        candidate_pairs = [
            (marker["protein"], marker["rna"]) for marker in contract["markers"]
        ]
        decision_pairs = [
            (decision.get("protein"), decision.get("rna"))
            for decision in decisions
            if isinstance(decision, dict)
        ]
        if len(decision_pairs) == len(decisions) and all(
            pair in candidate_pairs for pair in decision_pairs
        ):
            decision_indices = [candidate_pairs.index(pair) for pair in decision_pairs]
    if (
        axis_selection.get("rule")
        != "source support then ranked greedy complete-cross-product Poisson availability"
        or axis_selection.get("poisson_pools")
        != "full and every LOODO training global/adult/pediatric pool"
        or axis_selection.get("pseudocount") is not None
        or not isinstance(decisions, list)
        or len(decision_indices) != len(decisions)
        or len(set(decision_indices)) != len(decision_indices)
        or [index for index, decision in zip(decision_indices, decisions) if decision.get("accepted") is True]
        != indices
        or any(
            not isinstance(decision.get("minimum_source_rna_balance"), int)
            or decision["minimum_source_rna_balance"] < 4
            or decision["minimum_source_rna_balance"] > 256
            or not isinstance(decision.get("accepted"), bool)
            for decision in decisions
        )
    ):
        raise PermissionError("public source marker selection trace differs")
    records = source.get("source_records")
    if not isinstance(records, list) or len(records) != len(contract["source"]):
        raise PermissionError("public source donor records are incomplete")
    for donor_index, (sample, record) in enumerate(zip(contract["source"], records)):
        positives = np.asarray(record.get("rna_positive_counts"), dtype=np.int64)
        metadata_audit, h5_audit = source["access_audit"][2 * donor_index : 2 * donor_index + 2]
        if (
            record.get("gsm") != sample["gsm"]
            or record.get("donor") != sample["donor"]
            or record.get("age_group") != sample["age_group"]
            or record.get("batch") != sample["batch"]
            or record.get("cmv_status") != sample["cmv_status"]
            or record.get("selected_cells") != CELL_BUDGET
            or positives.shape != (count,)
            or np.any(positives < 4)
            or np.any(positives > 508)
            or record.get("rna_positive_counts_sha256") != _array_sha256(positives)
            or record.get("selected_cell_axis_sha256")
            != h5_audit.get("selected_cell_axis_sha256")
            or record.get("selected_cell_indices_sha256")
            != h5_audit.get("selected_cell_indices_sha256")
            or record.get("metadata_download_sha256") != metadata_audit.get("sha256")
            or record.get("metadata_download_bytes") != int(sample["metadata_bytes"])
            or record.get("h5_download_sha256") != h5_audit.get("sha256")
            or record.get("h5_download_bytes") != int(sample["h5_bytes"])
        ):
            raise PermissionError("public source donor record differs from its audit")
    _validate_download_audit(
        source.get("access_audit"), contract["source"], "source", contract["markers"]
    )
    shape = (count, count)
    models = source.get("models", {})
    if set(models) != set(METHODS):
        raise PermissionError("public source methods differ from the freeze")
    _validate_model_array(models["primary"], "coefficient", "coefficient_sha256", (2, *shape))
    _validate_model_array(models["destroyed_link"], "coefficient", "coefficient_sha256", (2, *shape))
    _validate_model_array(models["pooled_fixed_interaction_poisson"], "log_odds", "log_odds_sha256", shape)
    for group in ("adult", "pediatric"):
        _validate_model_array(models["age_stratified_fixed_interaction_poisson"][group], "log_odds", "log_odds_sha256", shape)
    _validate_model_array(models["common_effect_exact_conditional"], "log_odds", "log_odds_sha256", shape)
    _validate_model_array(models["signed_root_deviance"], "coordinate_per_sqrt_n", "coordinate_sha256", shape)
    if (
        models["primary"].get("transport") not in PRIMARY_TRANSPORT_GRID
        or models["destroyed_link"].get("transport")
        != models["primary"].get("transport")
        or models["pooled_fixed_interaction_poisson"].get("transport")
        not in POISSON_TRANSPORT_GRID
        or models["age_stratified_fixed_interaction_poisson"].get("transport")
        not in POISSON_TRANSPORT_GRID
        or models["common_effect_exact_conditional"]
        .get("fit_certificate", {})
        .get("estimator")
        != "unregularized_delta_free_exact_conditional_cmle"
    ):
        raise PermissionError("public source transport or exact comparator differs")
    _validate_fit_certificate(models["primary"].get("fit_certificate"))
    _validate_fit_certificate(models["destroyed_link"].get("fit_certificate"))
    for key in ("primary", "destroyed_link"):
        if (
            models[key].get("deviation_penalty") not in DEVIATION_GRID
            or models[key].get("age_ridge") not in AGE_RIDGE_GRID
            or models[key].get("intercept_ridge") != INTERCEPT_RIDGE
        ):
            raise PermissionError("public context model hyperparameters differ")
    common_certificate = models["common_effect_exact_conditional"].get("fit_certificate", {})
    if (
        common_certificate.get("passes") is not True
        or common_certificate.get("gradient_tolerance") != 1e-10
        or float(common_certificate.get("scaled_gradient_norm", math.inf)) > 1e-10
        or models["common_effect_exact_conditional"].get("transport") != 1.0
        or models["pooled_fixed_interaction_poisson"].get("pseudocount") is not None
    ):
        raise PermissionError("public exact/common or Poisson model certificate differs")
    _validate_source_cross_validation(source)
    selected_configuration = source["source_cross_validation"]["selected_primary_configuration"]
    if (
        models["primary"].get("deviation_penalty") != selected_configuration["deviation_penalty"]
        or models["primary"].get("age_ridge") != selected_configuration["age_ridge"]
        or models["primary"].get("transport") != selected_configuration["transport"]
        or any(
            models["destroyed_link"].get(key) != models["primary"].get(key)
            for key in ("deviation_penalty", "age_ridge", "transport")
        )
        or models["pooled_fixed_interaction_poisson"].get("transport")
        != source["source_cross_validation"]["selected_poisson_transport"]
        or models["age_stratified_fixed_interaction_poisson"].get("transport")
        != source["source_cross_validation"]["selected_age_stratified_poisson_transport"]
    ):
        raise PermissionError("public final models do not use the selected source configurations")
    return source, commit


def _method_predictions(
    source: dict[str, Any], age_group: str, rna_positive: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    count = int(source["selected_marker_count"])
    positives = np.asarray(rna_positive, dtype=np.int64)
    if positives.shape != (count,) or np.any(positives < 0) or np.any(positives > CELL_BUDGET):
        raise ProtocolRefusal("HELD_RNA_MARGIN_INVALID")
    pediatric = age_group == "pediatric"
    marker_rows, marker_columns = _rows_columns_from_positive(positives)
    rows = np.broadcast_to(marker_rows[:, None, :], (count, count, 2)).copy()
    columns = np.broadcast_to(marker_columns[None, :, :], (count, count, 2)).copy()
    models = source["models"]
    predictions: dict[str, np.ndarray] = {}
    predictions["primary"] = _predict_conditional(
        np.asarray(models["primary"]["coefficient"], dtype=float),
        pediatric,
        float(models["primary"]["transport"]),
        rows,
        columns,
    )
    predictions["destroyed_link"] = _predict_conditional(
        np.asarray(models["destroyed_link"]["coefficient"], dtype=float),
        pediatric,
        float(models["destroyed_link"]["transport"]),
        rows,
        columns,
    )
    predictions["pooled_fixed_interaction_poisson"] = _predict_poisson(
        np.asarray(models["pooled_fixed_interaction_poisson"]["log_odds"], dtype=float),
        float(models["pooled_fixed_interaction_poisson"]["transport"]),
        rows,
        columns,
    )
    age_model = models["age_stratified_fixed_interaction_poisson"][age_group]
    predictions["age_stratified_fixed_interaction_poisson"] = _predict_poisson(
        np.asarray(age_model["log_odds"], dtype=float),
        float(models["age_stratified_fixed_interaction_poisson"]["transport"]),
        rows,
        columns,
    )
    common = np.asarray(models["common_effect_exact_conditional"]["log_odds"], dtype=float)
    predictions["common_effect_exact_conditional"] = np.empty((count, count, 2, 2), dtype=float)
    for index in np.ndindex((count, count)):
        predictions["common_effect_exact_conditional"][index] = _expected_conditional_table(
            float(common[index]), rows[index], columns[index]
        )
    predictions["signed_root_deviance"], saturated = _predict_signed_root(
        np.asarray(models["signed_root_deviance"]["coordinate_per_sqrt_n"], dtype=float),
        rows,
        columns,
    )
    predictions["independence"] = np.empty((count, count, 2, 2), dtype=float)
    for index in np.ndindex((count, count)):
        predictions["independence"][index] = _independence_table(rows[index], columns[index])
    if set(predictions) != set(METHODS):
        raise AssertionError("prediction method set is incomplete")
    return predictions, {"signed_root_boundary_saturations": saturated}


def claim_prediction(*, claim_token: Path) -> dict[str, Any]:
    """Create the public RNA-only held-prediction attempt before any held GET."""

    source, source_commit = _validate_source_result(require_public=True)
    if (
        PREDICTION_ATTEMPT.exists()
        or PREDICTION_RESULT.exists()
        or SCORE_AUTHORIZATION.exists()
        or SCORE_ATTEMPT.exists()
        or SCORE_RESULT.exists()
    ):
        raise FileExistsError("GSE214546 prediction stage has already been claimed or advanced")
    contract = _contract()
    token_sha256 = _create_claim_token(claim_token)
    bindings = {
        "source_tag": SOURCE_TAG,
        "source_commit": source_commit,
        "source_sha256": _sha256(SOURCE_RESULT),
        "source_bytes": SOURCE_RESULT.stat().st_size,
        "crash_semantics_clarification_sha256": CRASH_SEMANTICS_CLARIFICATION_SHA256,
        "implementation": _implementation_bindings(),
    }
    attempt = {
        "schema": "gse214546-teaseq-prediction-attempt/1.0",
        "status": "CLAIMED_BEFORE_FIRST_HELD_FILE_GET",
        "created_at_utc": _timestamp(),
        "bindings": bindings,
        "held_gsms": [sample["gsm"] for sample in contract["held"]],
        "claim_token_sha256": token_sha256,
        "held_adt_value_access_authorized": False,
        "rerun_permitted": False,
    }
    _write_json_x(PREDICTION_ATTEMPT, attempt)
    return attempt


def _validate_prediction_attempt() -> tuple[dict[str, Any], dict[str, Any], str]:
    source, source_commit = _validate_source_result(require_public=True)
    contract = _contract()
    commit = _require_tag_paths(PREDICTION_ATTEMPT_TAG, (PREDICTION_ATTEMPT,))
    _require_ancestor(source_commit, commit)
    attempt = _read_json(PREDICTION_ATTEMPT)
    expected_bindings = {
        "source_tag": SOURCE_TAG,
        "source_commit": source_commit,
        "source_sha256": _sha256(SOURCE_RESULT),
        "source_bytes": SOURCE_RESULT.stat().st_size,
        "crash_semantics_clarification_sha256": CRASH_SEMANTICS_CLARIFICATION_SHA256,
        "implementation": _implementation_bindings(),
    }
    if (
        attempt.get("schema") != "gse214546-teaseq-prediction-attempt/1.0"
        or attempt.get("status") != "CLAIMED_BEFORE_FIRST_HELD_FILE_GET"
        or attempt.get("bindings") != expected_bindings
        or attempt.get("held_gsms") != [sample["gsm"] for sample in contract["held"]]
        or attempt.get("held_adt_value_access_authorized") is not False
        or attempt.get("rerun_permitted") is not False
    ):
        raise PermissionError("public prediction attempt differs from the frozen source")
    _validate_claim_token_hash(attempt.get("claim_token_sha256"))
    return attempt, source, commit


def run_prediction(
    *, scratch: Path = DEFAULT_SCRATCH, claim_token: Path | None = None
) -> dict[str, Any]:
    """Freeze held margins and predictions without reading any held ADT value."""

    attempt, source, attempt_commit = _validate_prediction_attempt()
    if PREDICTION_RESULT.exists() or SCORE_AUTHORIZATION.exists() or SCORE_ATTEMPT.exists() or SCORE_RESULT.exists():
        raise FileExistsError("GSE214546 prediction stage has already advanced")
    contract = _contract()
    scratch_path = _scratch(scratch)
    bindings = {
        **attempt["bindings"],
        "prediction_attempt_tag": PREDICTION_ATTEMPT_TAG,
        "prediction_attempt_commit": attempt_commit,
        "prediction_attempt_sha256": _sha256(PREDICTION_ATTEMPT),
        "prediction_attempt_bytes": PREDICTION_ATTEMPT.stat().st_size,
        "claim_token_sha256": attempt["claim_token_sha256"],
    }
    _consume_claim_token(claim_token, attempt["claim_token_sha256"])
    audit: list[dict[str, Any]] = []
    try:
        held_records = []
        markers = source["selected_markers"]
        for sample in contract["held"]:
            reduced = _reduce_sample(sample, markers, "predict", scratch_path, audit)
            if "adt_counts" in reduced or any(
                dataset.startswith("ADT/") for dataset in reduced["datasets_read"]
            ):
                raise PermissionError("prediction stage crossed the held ADT firewall")
            positive = (reduced["rna_counts"] > 0).sum(axis=0).astype(np.int64)
            predictions, diagnostics = _method_predictions(
                source, sample["age_group"], positive
            )
            held_records.append(
                {
                    "gsm": sample["gsm"],
                    "donor": sample["donor"],
                    "age_group": sample["age_group"],
                    "batch": sample["batch"],
                    "cmv_status": sample["cmv_status"],
                    "selected_cell_axis_sha256": reduced["selected_cell_axis_sha256"],
                    "selected_cell_indices_sha256": reduced["selected_cell_indices_sha256"],
                    "selected_cells": CELL_BUDGET,
                    "held_metadata_sha256": reduced["metadata_download_sha256"],
                    "held_metadata_bytes": reduced["metadata_download_bytes"],
                    "held_h5_sha256": reduced["h5_download_sha256"],
                    "held_h5_bytes": reduced["h5_download_bytes"],
                    "rna_positive_counts": positive.tolist(),
                    "rna_positive_counts_sha256": _array_sha256(positive),
                    "protein_high_counts": [PROTEIN_HIGH_COUNT] * len(markers),
                    "predictions": {name: values.tolist() for name, values in predictions.items()},
                    "prediction_sha256": {name: _array_sha256(values) for name, values in predictions.items()},
                    "diagnostics": diagnostics,
                }
            )
        result = {
            "schema": "gse214546-teaseq-prediction-result/1.0",
            "status": "PREDICTIONS_FROZEN_WITHOUT_HELD_ADT_VALUES",
            "created_at_utc": _timestamp(),
            "bindings": bindings,
            "selected_markers": markers,
            "selected_marker_axis_sha256": source["selected_marker_axis_sha256"],
            "methods": list(METHODS),
            "held_records": held_records,
            "access_audit": audit,
            "held_h5_requested": True,
            "held_adt_value_datasets_read": 0,
            "held_truth_tables_formed": 0,
            "rerun_permitted": False,
        }
    except BaseException as error:
        result = _failure_payload(
            "gse214546-teaseq-prediction-result/1.0",
            "TERMINAL_PREDICTION_REFUSAL",
            "predict",
            error,
            audit,
            bindings,
        )
    _write_json_x(PREDICTION_RESULT, result)
    return result


def _validate_prediction_result() -> tuple[dict[str, Any], dict[str, Any], str]:
    attempt, source, attempt_commit = _validate_prediction_attempt()
    prediction, commit = _public_result(
        PREDICTION_TAG,
        PREDICTION_RESULT,
        attempt_commit,
        bound_paths=(PREDICTION_ATTEMPT,),
    )
    contract = _contract()
    expected_bindings = {
        **attempt["bindings"],
        "prediction_attempt_tag": PREDICTION_ATTEMPT_TAG,
        "prediction_attempt_commit": attempt_commit,
        "prediction_attempt_sha256": _sha256(PREDICTION_ATTEMPT),
        "prediction_attempt_bytes": PREDICTION_ATTEMPT.stat().st_size,
        "claim_token_sha256": attempt["claim_token_sha256"],
    }
    if (
        prediction.get("schema") != "gse214546-teaseq-prediction-result/1.0"
        or prediction.get("status") != "PREDICTIONS_FROZEN_WITHOUT_HELD_ADT_VALUES"
        or prediction.get("bindings") != expected_bindings
        or prediction.get("held_adt_value_datasets_read") != 0
        or prediction.get("held_truth_tables_formed") != 0
        or prediction.get("held_h5_requested") is not True
        or prediction.get("methods") != list(METHODS)
        or prediction.get("selected_markers") != source["selected_markers"]
        or prediction.get("selected_marker_axis_sha256")
        != source["selected_marker_axis_sha256"]
        or prediction.get("rerun_permitted") is not False
    ):
        raise PermissionError("public prediction is not the frozen RNA-only artifact")
    records = prediction.get("held_records")
    if not isinstance(records, list) or [record.get("gsm") for record in records] != [sample["gsm"] for sample in contract["held"]]:
        raise PermissionError("public prediction held donor order differs")
    count = int(source["selected_marker_count"])
    expected_h5_hashes: dict[str, str] = {}
    for donor_index, (sample, record) in enumerate(zip(contract["held"], records)):
        positive = np.asarray(record.get("rna_positive_counts"), dtype=np.int64)
        metadata_audit, h5_audit = prediction["access_audit"][
            2 * donor_index : 2 * donor_index + 2
        ]
        if (
            record.get("gsm") != sample["gsm"]
            or record.get("donor") != sample["donor"]
            or record.get("age_group") != sample["age_group"]
            or record.get("batch") != sample["batch"]
            or record.get("cmv_status") != sample["cmv_status"]
            or record.get("selected_cells") != CELL_BUDGET
            or record.get("selected_cell_axis_sha256")
            != h5_audit.get("selected_cell_axis_sha256")
            or record.get("selected_cell_indices_sha256")
            != h5_audit.get("selected_cell_indices_sha256")
            or record.get("held_metadata_sha256") != metadata_audit.get("sha256")
            or record.get("held_metadata_bytes") != int(sample["metadata_bytes"])
            or record.get("held_h5_sha256") != h5_audit.get("sha256")
            or record.get("held_h5_bytes") != int(sample["h5_bytes"])
            or positive.shape != (count,)
            or np.any(positive < 0)
            or np.any(positive > CELL_BUDGET)
            or record.get("rna_positive_counts_sha256") != _array_sha256(positive)
            or record.get("protein_high_counts") != [PROTEIN_HIGH_COUNT] * count
        ):
            raise PermissionError("public held RNA margins differ")
        expected_h5_hashes[sample["gsm"]] = record["held_h5_sha256"]
        expected, expected_diagnostics = _method_predictions(
            source, sample["age_group"], positive
        )
        if (
            set(record.get("predictions", {})) != set(METHODS)
            or set(record.get("prediction_sha256", {})) != set(METHODS)
            or record.get("diagnostics") != expected_diagnostics
        ):
            raise PermissionError("public held predictions are incomplete")
        for method in METHODS:
            observed = np.asarray(record["predictions"][method], dtype=float)
            if observed.shape != (count, count, 2, 2) or record["prediction_sha256"].get(method) != _array_sha256(observed) or not np.array_equal(observed, expected[method]):
                raise PermissionError(f"public held prediction differs: {method}")
    _validate_download_audit(
        prediction.get("access_audit"),
        contract["held"],
        "predict",
        source["selected_markers"],
        expected_h5_sha256=expected_h5_hashes,
    )
    return prediction, source, commit


def authorize_score() -> dict[str, Any]:
    """Freeze explicit held-ADT authorization after the public prediction."""

    _, _, prediction_commit = _validate_prediction_result()
    if SCORE_AUTHORIZATION.exists() or SCORE_ATTEMPT.exists() or SCORE_RESULT.exists():
        raise FileExistsError("GSE214546 score stage has already been authorized or advanced")
    authorization = {
        "schema": "gse214546-teaseq-score-authorization/1.0",
        "status": "AUTHORIZED_AFTER_PUBLIC_PREDICTION",
        "created_at_utc": _timestamp(),
        "prediction_tag": PREDICTION_TAG,
        "prediction_commit": prediction_commit,
        "prediction_sha256": _sha256(PREDICTION_RESULT),
        "prediction_bytes": PREDICTION_RESULT.stat().st_size,
        "held_adt_value_access_authorized": True,
        "rerun_permitted": False,
    }
    _write_json_x(SCORE_AUTHORIZATION, authorization)
    return authorization


def _validate_score_authorization(
    prediction_commit: str, authorization_path: Path = SCORE_AUTHORIZATION
) -> tuple[dict[str, Any], str]:
    if authorization_path.resolve() != SCORE_AUTHORIZATION.resolve():
        raise PermissionError("score authorization must be the frozen repository artifact")
    authorization_commit = _require_tag_paths(
        SCORE_AUTHORIZATION_TAG, (authorization_path, PREDICTION_RESULT)
    )
    _require_ancestor(prediction_commit, authorization_commit)
    value = _read_json(authorization_path)
    if (
        value.get("schema") != "gse214546-teaseq-score-authorization/1.0"
        or value.get("status") != "AUTHORIZED_AFTER_PUBLIC_PREDICTION"
        or value.get("prediction_tag") != PREDICTION_TAG
        or value.get("prediction_commit") != prediction_commit
        or value.get("prediction_sha256") != _sha256(PREDICTION_RESULT)
        or value.get("prediction_bytes") != PREDICTION_RESULT.stat().st_size
        or value.get("held_adt_value_access_authorized") is not True
        or value.get("rerun_permitted") is not False
    ):
        raise PermissionError("public score authorization differs from the frozen prediction")
    return value, authorization_commit


def claim_score(
    *, claim_token: Path, authorization_path: Path = SCORE_AUTHORIZATION
) -> dict[str, Any]:
    """Create the public score attempt after public authorization and before truth GET."""

    _, _, prediction_commit = _validate_prediction_result()
    authorization, authorization_commit = _validate_score_authorization(
        prediction_commit, authorization_path
    )
    if SCORE_ATTEMPT.exists() or SCORE_RESULT.exists():
        raise FileExistsError("GSE214546 score stage has already been claimed")
    token_sha256 = _create_claim_token(claim_token)
    bindings = {
        "prediction_tag": PREDICTION_TAG,
        "prediction_commit": prediction_commit,
        "prediction_sha256": _sha256(PREDICTION_RESULT),
        "prediction_bytes": PREDICTION_RESULT.stat().st_size,
        "authorization_tag": SCORE_AUTHORIZATION_TAG,
        "authorization_commit": authorization_commit,
        "authorization_sha256": _sha256(authorization_path),
        "authorization_bytes": authorization_path.stat().st_size,
        "crash_semantics_clarification_sha256": CRASH_SEMANTICS_CLARIFICATION_SHA256,
        "implementation": _implementation_bindings(),
    }
    attempt = {
        "schema": "gse214546-teaseq-score-attempt/1.0",
        "status": "CLAIMED_AFTER_AUTHORIZATION_BEFORE_FIRST_HELD_TRUTH_GET",
        "created_at_utc": _timestamp(),
        "bindings": bindings,
        "claim_token_sha256": token_sha256,
        "held_adt_value_access_authorized": authorization[
            "held_adt_value_access_authorized"
        ],
        "rerun_permitted": False,
    }
    _write_json_x(SCORE_ATTEMPT, attempt)
    return attempt


def _validate_score_attempt(
    authorization_path: Path = SCORE_AUTHORIZATION,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    prediction, source, prediction_commit = _validate_prediction_result()
    authorization, authorization_commit = _validate_score_authorization(
        prediction_commit, authorization_path
    )
    commit = _require_tag_paths(SCORE_ATTEMPT_TAG, (SCORE_ATTEMPT,))
    _require_ancestor(authorization_commit, commit)
    attempt = _read_json(SCORE_ATTEMPT)
    expected_bindings = {
        "prediction_tag": PREDICTION_TAG,
        "prediction_commit": prediction_commit,
        "prediction_sha256": _sha256(PREDICTION_RESULT),
        "prediction_bytes": PREDICTION_RESULT.stat().st_size,
        "authorization_tag": SCORE_AUTHORIZATION_TAG,
        "authorization_commit": authorization_commit,
        "authorization_sha256": _sha256(authorization_path),
        "authorization_bytes": authorization_path.stat().st_size,
        "crash_semantics_clarification_sha256": CRASH_SEMANTICS_CLARIFICATION_SHA256,
        "implementation": _implementation_bindings(),
    }
    if (
        attempt.get("schema") != "gse214546-teaseq-score-attempt/1.0"
        or attempt.get("status")
        != "CLAIMED_AFTER_AUTHORIZATION_BEFORE_FIRST_HELD_TRUTH_GET"
        or attempt.get("bindings") != expected_bindings
        or attempt.get("held_adt_value_access_authorized") is not True
        or attempt.get("rerun_permitted") is not False
    ):
        raise PermissionError("public score attempt differs from the authorization")
    _validate_claim_token_hash(attempt.get("claim_token_sha256"))
    return attempt, prediction, source, commit


def _comparison(primary: np.ndarray, comparator: np.ndarray, ages: list[str]) -> dict[str, Any]:
    first = np.asarray(primary, dtype=float)
    second = np.asarray(comparator, dtype=float)
    difference = first - second
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws = difference[generator.integers(0, len(difference), size=(BOOTSTRAPS, len(difference)))].mean(axis=1)
    interval = np.quantile(draws, (0.025, 0.975))
    favorable = int(np.sum(difference < 0.0))
    sign_probability = float(
        sum(math.comb(len(difference), count) for count in range(favorable, len(difference) + 1))
        / (2 ** len(difference))
    )
    strata = {
        group: float(np.mean(difference[np.asarray(ages) == group]))
        for group in ("adult", "pediatric")
    }
    relative = float(1.0 - first.mean() / second.mean())
    checks = {
        "mean_deviance_at_least_five_percent_lower": relative >= 0.05,
        "paired_bootstrap_upper_endpoint_below_zero": float(interval[1]) < 0.0,
        "at_least_seven_of_eight_donors_favorable": favorable >= 7,
        "adult_mean_difference_below_zero": strata["adult"] < 0.0,
        "pediatric_mean_difference_below_zero": strata["pediatric"] < 0.0,
    }
    return {
        "primary_mean_deviance": float(first.mean()),
        "comparator_mean_deviance": float(second.mean()),
        "relative_reduction": relative,
        "primary_minus_comparator_mean": float(difference.mean()),
        "paired_donor_bootstrap_95_ci": interval.tolist(),
        "bootstrap_draws": BOOTSTRAPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "favorable_donors": favorable,
        "exact_one_sided_sign_probability": sign_probability,
        "stratum_mean_differences": strata,
        "checks": checks,
        "passes": bool(all(checks.values())),
    }


def _score_access_counts(audit: list[dict[str, Any]]) -> dict[str, int]:
    if not isinstance(audit, list) or any(not isinstance(record, dict) for record in audit):
        raise PermissionError("score access audit is not a record list")
    return {
        "held_adt_value_datasets_read": sum(
            "ADT/data" in record.get("datasets_read", []) for record in audit
        ),
        "held_adt_modalities_reduced": sum(
            record.get("adt_access", {}).get("completed") is True for record in audit
        ),
    }


def run_score(
    *,
    scratch: Path = DEFAULT_SCRATCH,
    authorization_path: Path = SCORE_AUTHORIZATION,
    claim_token: Path | None = None,
) -> dict[str, Any]:
    """Read held truth only after public prediction and authorization freezes."""

    attempt, prediction, source, attempt_commit = _validate_score_attempt(
        authorization_path
    )
    if SCORE_RESULT.exists():
        raise FileExistsError("GSE214546 score stage has already advanced")
    contract = _contract()
    scratch_path = _scratch(scratch)
    bindings = {
        **attempt["bindings"],
        "score_attempt_tag": SCORE_ATTEMPT_TAG,
        "score_attempt_commit": attempt_commit,
        "score_attempt_sha256": _sha256(SCORE_ATTEMPT),
        "score_attempt_bytes": SCORE_ATTEMPT.stat().st_size,
        "claim_token_sha256": attempt["claim_token_sha256"],
    }
    _consume_claim_token(claim_token, attempt["claim_token_sha256"])
    audit: list[dict[str, Any]] = []
    try:
        losses = {method: np.full(len(contract["held"]), np.nan) for method in METHODS}
        truth_hashes = []
        for donor_index, (sample, frozen) in enumerate(
            zip(contract["held"], prediction["held_records"])
        ):
            reduced = _reduce_sample(
                sample,
                source["selected_markers"],
                "score",
                scratch_path,
                audit,
                expected_metadata_sha256=frozen["held_metadata_sha256"],
                expected_h5_sha256=frozen["held_h5_sha256"],
            )
            positive = (reduced["rna_counts"] > 0).sum(axis=0).astype(np.int64)
            if (
                reduced["selected_cell_axis_sha256"] != frozen["selected_cell_axis_sha256"]
                or frozen["rna_positive_counts_sha256"] != _array_sha256(positive)
            ):
                raise PermissionError("held score cell axis or RNA margins differ from prediction")
            rna_states = (reduced["rna_counts"] > 0).astype(np.uint8)
            adt_states = _adt_states(
                reduced["adt_counts"],
                reduced["selected_barcodes"],
                sample["gsm"],
                [marker["protein"] for marker in source["selected_markers"]],
            )
            truth = _joint_tables(rna_states, adt_states)
            truth_hashes.append(
                {
                    "gsm": sample["gsm"],
                    "truth_table_sha256": _array_sha256(truth),
                    "truth_table_shape": list(truth.shape),
                }
            )
            for method in METHODS:
                estimate = np.asarray(frozen["predictions"][method], dtype=float)
                losses[method][donor_index] = _loss(truth, estimate)
        comparisons = {
            method: _comparison(
                losses["primary"],
                losses[method],
                [sample["age_group"] for sample in contract["held"]],
            )
            for method in METHODS
            if method != "primary"
        }
        passes = bool(
            comparisons["pooled_fixed_interaction_poisson"]["passes"]
            and comparisons["destroyed_link"]["passes"]
        )
        result = {
            "schema": "gse214546-teaseq-confirmation-result/1.0",
            "status": "CONFIRMATION_PASS" if passes else "CONFIRMATION_FAIL",
            "created_at_utc": _timestamp(),
            "bindings": bindings,
            "decision": {
                "passes": passes,
                "confirmatory_comparators": [
                    "pooled_fixed_interaction_poisson",
                    "destroyed_link",
                ],
                "age_stratified_poisson_role": "reported sensitivity outside frozen confirmatory gate",
            },
            "donor_losses": {method: values.tolist() for method, values in losses.items()},
            "donor_mean_deviance": {
                method: float(values.mean()) for method, values in losses.items()
            },
            "comparisons": comparisons,
            "truth_hashes": truth_hashes,
            "access_audit": audit,
            **_score_access_counts(audit),
            "held_donors_scored": len(contract["held"]),
            "rerun_permitted": False,
        }
    except BaseException as error:
        result = _failure_payload(
            "gse214546-teaseq-confirmation-result/1.0",
            "TERMINAL_SCORE_REFUSAL",
            "score",
            error,
            audit,
            bindings,
        )
    _write_json_x(SCORE_RESULT, result)
    return result


def _validate_score_result() -> tuple[dict[str, Any], str]:
    attempt, prediction, source, attempt_commit = _validate_score_attempt()
    result, commit = _public_result(
        SCORE_TAG,
        SCORE_RESULT,
        attempt_commit,
        bound_paths=(SCORE_ATTEMPT,),
    )
    expected_bindings = {
        **attempt["bindings"],
        "score_attempt_tag": SCORE_ATTEMPT_TAG,
        "score_attempt_commit": attempt_commit,
        "score_attempt_sha256": _sha256(SCORE_ATTEMPT),
        "score_attempt_bytes": SCORE_ATTEMPT.stat().st_size,
        "claim_token_sha256": attempt["claim_token_sha256"],
    }
    decision = result.get("decision", {})
    access_counts = _score_access_counts(result.get("access_audit", []))
    if (
        result.get("schema") != "gse214546-teaseq-confirmation-result/1.0"
        or result.get("status") not in {"CONFIRMATION_PASS", "CONFIRMATION_FAIL"}
        or result.get("bindings") != expected_bindings
        or decision.get("confirmatory_comparators")
        != ["pooled_fixed_interaction_poisson", "destroyed_link"]
        or decision.get("age_stratified_poisson_role")
        != "reported sensitivity outside frozen confirmatory gate"
        or result.get("held_adt_value_datasets_read")
        != access_counts["held_adt_value_datasets_read"]
        or result.get("held_adt_modalities_reduced")
        != access_counts["held_adt_modalities_reduced"]
        or result.get("held_adt_modalities_reduced") != len(_contract()["held"])
        or result.get("held_donors_scored") != len(_contract()["held"])
        or result.get("rerun_permitted") is not False
    ):
        raise PermissionError("public score result differs from the frozen decision contract")
    losses_object = result.get("donor_losses")
    if not isinstance(losses_object, dict) or set(losses_object) != set(METHODS):
        raise PermissionError("public score losses are incomplete")
    losses: dict[str, np.ndarray] = {}
    for method in METHODS:
        values = np.asarray(losses_object[method], dtype=float)
        if values.shape != (8,) or not np.isfinite(values).all() or np.any(values < 0.0):
            raise PermissionError("public score donor losses are invalid")
        losses[method] = values
        if result.get("donor_mean_deviance", {}).get(method) != float(values.mean()):
            raise PermissionError("public score donor mean does not reproduce")
    ages = [sample["age_group"] for sample in _contract()["held"]]
    expected_comparisons = {
        method: _comparison(losses["primary"], losses[method], ages)
        for method in METHODS
        if method != "primary"
    }
    if result.get("comparisons") != expected_comparisons:
        raise PermissionError("public score comparisons do not reproduce")
    passes = bool(
        expected_comparisons["pooled_fixed_interaction_poisson"]["passes"]
        and expected_comparisons["destroyed_link"]["passes"]
    )
    if (
        decision.get("passes") is not passes
        or result["status"] != ("CONFIRMATION_PASS" if passes else "CONFIRMATION_FAIL")
    ):
        raise PermissionError("public score decision arithmetic does not reproduce")
    truths = result.get("truth_hashes")
    if (
        not isinstance(truths, list)
        or [record.get("gsm") for record in truths]
        != [sample["gsm"] for sample in _contract()["held"]]
        or any(
            record.get("truth_table_shape")
            != [source["selected_marker_count"], source["selected_marker_count"], 2, 2]
            or not _is_sha256(record.get("truth_table_sha256"))
            for record in truths
        )
    ):
        raise PermissionError("public held truth hashes are incomplete")
    expected_h5_hashes = {
        record["gsm"]: record["held_h5_sha256"]
        for record in prediction["held_records"]
    }
    expected_metadata_hashes = {
        record["gsm"]: record["held_metadata_sha256"]
        for record in prediction["held_records"]
    }
    _validate_download_audit(
        result.get("access_audit"),
        _contract()["held"],
        "score",
        source["selected_markers"],
        expected_metadata_sha256=expected_metadata_hashes,
        expected_h5_sha256=expected_h5_hashes,
        expected_cell_axis_sha256={
            record["gsm"]: record["selected_cell_axis_sha256"]
            for record in prediction["held_records"]
        },
    )
    return result, commit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for stage in ("claim-source", "claim-predict", "claim-score"):
        command = subparsers.add_parser(stage)
        command.add_argument("--claim-token", type=Path, required=True)
    subparsers.add_parser("authorize-score")
    subparsers.add_parser("validate-score")
    for stage in ("source", "predict", "score"):
        command = subparsers.add_parser(stage)
        command.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
        command.add_argument("--claim-token", type=Path, required=True)
        if stage == "score":
            command.add_argument(
                "--authorization", type=Path, default=SCORE_AUTHORIZATION
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.stage == "claim-source":
        result = claim_source(claim_token=arguments.claim_token)
    elif arguments.stage == "source":
        result = run_source(
            scratch=arguments.scratch, claim_token=arguments.claim_token
        )
    elif arguments.stage == "claim-predict":
        result = claim_prediction(claim_token=arguments.claim_token)
    elif arguments.stage == "predict":
        result = run_prediction(
            scratch=arguments.scratch, claim_token=arguments.claim_token
        )
    elif arguments.stage == "authorize-score":
        result = authorize_score()
    elif arguments.stage == "claim-score":
        result = claim_score(claim_token=arguments.claim_token)
    elif arguments.stage == "score":
        result = run_score(
            scratch=arguments.scratch,
            authorization_path=arguments.authorization,
            claim_token=arguments.claim_token,
        )
    else:
        result, _ = _validate_score_result()
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    return 0 if result["status"] not in {
        "TERMINAL_SOURCE_REFUSAL",
        "TERMINAL_PREDICTION_REFUSAL",
        "TERMINAL_SCORE_REFUSAL",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
