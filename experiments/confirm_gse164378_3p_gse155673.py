"""One-shot cross-study CITE-seq coupling-field confirmation.

GSE164378 3-prime day-0 donors form the source map.  GSE155673 donors are
held until source fitting has closed publicly.  Held RNA and ADT values are
reduced in separate stages; prediction sees only their margins, and score is
the first stage allowed to join the private state artifacts.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import hashlib
from http.client import IncompleteRead
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable, Iterable
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np

from mapreg.classical_residuals import poisson_independence_residuals
from mapreg.heterogeneity_adaptive_coupling import (
    CouplingEstimationRefusal,
    centered_haldane_log_odds,
    expected_binary_table_from_log_odds,
    fit_structured_conditional_log_odds,
    paule_mandel_pool,
)
from mapreg.hierarchical_conditional_coupling import (
    fit_hierarchical_conditional_log_odds,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/confirmation/gse164378_3p_gse155673"
DEFAULT_CANDIDATE = DATA_DIR / "candidate_designation_v1.json"
DEFAULT_PROTOCOL = DATA_DIR / "protocol_v1.json"
DEFAULT_RUNTIME = DATA_DIR / "runtime_environment_v1.json"
DEFAULT_MANIFEST = DATA_DIR / "source_manifest_v1.json"
DEFAULT_SCORE_AUTHORIZATION_TEMPLATE = DATA_DIR / "score_authorization_template_v1.json"
DEFAULT_METADATA_PREFLIGHT = (
    ROOT / "results/development/gse164378_3p_gse155673_metadata_preflight_v1.json"
)
DEFAULT_TESTS = ROOT / "tests/test_gse164378_3p_gse155673_confirmation.py"

DEFAULT_SOURCE = ROOT / "results/development/gse164378_3p_gse155673_source_v1.json"
DEFAULT_RNA = ROOT / "results/development/gse164378_3p_gse155673_rna_v1.json"
DEFAULT_ADT = ROOT / "results/development/gse164378_3p_gse155673_adt_v1.json"
DEFAULT_PREDICTION = ROOT / "results/gse164378_3p_gse155673_predictions_v1.json"
DEFAULT_SCORE_AUTHORIZATION = DATA_DIR / "score_authorization_v1.json"
DEFAULT_SCORE = ROOT / "results/gse164378_3p_gse155673_confirmation_v1.json"

PROTOCOL_TAG = "gse164378-3p-gse155673-v1-protocol"
SOURCE_ATTEMPT_TAG = "gse164378-3p-gse155673-v1-source-attempt"
SOURCE_TAG = "gse164378-3p-gse155673-v1-source"
RNA_ATTEMPT_TAG = "gse164378-3p-gse155673-v1-rna-attempt"
RNA_TAG = "gse164378-3p-gse155673-v1-rna"
ADT_ATTEMPT_TAG = "gse164378-3p-gse155673-v1-adt-attempt"
ADT_TAG = "gse164378-3p-gse155673-v1-adt"
PREDICTION_ATTEMPT_TAG = "gse164378-3p-gse155673-v1-prediction-attempt"
PREDICTION_TAG = "gse164378-3p-gse155673-v1-prediction"
SCORE_AUTHORIZATION_TAG = "gse164378-3p-gse155673-v1-score-authorization"
SCORE_ATTEMPT_TAG = "gse164378-3p-gse155673-v1-score-attempt"
RESULT_TAG = "gse164378-3p-gse155673-v1-result"
PUBLIC_ORIGIN = "https://github.com/sushaan-k/coupling-fields-benchmark.git"

CELL_BUDGET = 384
SOURCE_DONORS = ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8")
CALIBRATION_DONORS = ("P1", "P3", "P5", "P7")
VALIDATION_DONORS = ("P2", "P4", "P6", "P8")
VALIDATION_COMPONENTS = (("P2", "P4"), ("P6", "P8"))
HELD_DONORS = (
    "cov01",
    "cov02",
    "cov03",
    "cov04",
    "cov07",
    "cov08",
    "cov09",
    "cov10",
    "cov11",
    "cov12",
    "cov17",
    "cov18",
)
HEALTHY_DONORS = ("cov07", "cov08", "cov09", "cov17", "cov18")
MODERATE_DONORS = ("cov02", "cov03", "cov12")
SEVERE_DONORS = ("cov01", "cov04", "cov10", "cov11")
FROZEN_MARKER_AXIS_SHA256 = (
    "1883286191dabe544372ec6d1695fe5cf54be4c62b716fdd9e82cba9cdaafeca"
)
FROZEN_SOURCE_DONOR_AXIS_SHA256 = (
    "30434af29e158e471969d008f5ff7f5c0a64f7d2ef1a69b8a9360194ce42901c"
)
FROZEN_HELD_DONOR_AXIS_SHA256 = (
    "86b7f6bc9b6d348bab1b59dff0a19f021e0188fd1f8b1f863d8563cc952d13f2"
)
MINIMUM_SOURCE_LOCKED_MARKERS = 16
MINIMUM_HELD_SUPPORTED_DONORS = 10
MINIMUM_HELD_PAIR_FRACTION = 0.80
MINIMUM_RNA_PREVALENCE = 0.05
MAXIMUM_RNA_PREVALENCE = 0.95
MAXIMUM_ADT_EQUAL_VALUE_FRACTION = 0.90
SOURCE_VALIDATION_MINIMUM_RELATIVE_GAIN = 0.0
HETEROGENEITY_GRID = (0.1, 1.0, 10.0)
RIDGE_GRID = (0.01, 0.1, 1.0)
GRAPH_GRID = (0.1, 1.0)
TRANSPORT_GRID = (0.5, 0.75, 1.0)
GRAPH_NEIGHBORS = 2
MAXIMUM_CONDITION_NUMBER = 1e12
BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 20260829
ALPHA = 0.0125
DOWNLOAD_ATTEMPTS = 3
CELL_SELECTION_SALT = "GSE164378-3P-GSE155673-CELL-BUDGET-v1"
ADT_TIE_SALT = "GSE164378-3P-GSE155673-ADT-v1"
DESTROYED_LINK_SALT = "GSE164378-3P-GSE155673-DESTROY-v1"

RAW_RESIDUAL_METHOD = "poisson_independence_signed_deviance_residual_raw"
CLASSICAL_ORDER = (
    "poisson_independence_signed_deviance_residual",
    "common_effect_stratified_cmle",
    "pooled_saturated_poisson",
    "paule_mandel_random_effects_log_odds",
)
ALL_CLASSICAL_METHODS = (RAW_RESIDUAL_METHOD,) + CLASSICAL_ORDER
INDEPENDENCE_METHOD = "target_margin_independence"

PROTOCOL_BINDINGS = (
    ".gitattributes",
    "experiments/confirm_gse164378_3p_gse155673.py",
    "tests/test_gse164378_3p_gse155673_confirmation.py",
    "data/confirmation/gse164378_3p_gse155673/candidate_designation_v1.json",
    "data/confirmation/gse164378_3p_gse155673/protocol_v1.json",
    "data/confirmation/gse164378_3p_gse155673/runtime_environment_v1.json",
    "data/confirmation/gse164378_3p_gse155673/source_manifest_v1.json",
    "data/confirmation/gse164378_3p_gse155673/score_authorization_template_v1.json",
    "results/development/gse164378_3p_gse155673_metadata_preflight_v1.json",
    "docs/GSE164378_3P_TO_GSE155673_EXTERNAL_STUDY_PROTOCOL_2026-08-29.md",
    "mapreg/heterogeneity_adaptive_coupling.py",
    "mapreg/hierarchical_conditional_coupling.py",
    "mapreg/classical_residuals.py",
    "mapreg/coupling_fields.py",
    "mapreg/table_prediction.py",
    "requirements.txt",
    "pyproject.toml",
)

STAGE_PATHS = {
    "source": DEFAULT_SOURCE,
    "rna": DEFAULT_RNA,
    "adt": DEFAULT_ADT,
    "prediction": DEFAULT_PREDICTION,
    "score": DEFAULT_SCORE,
}
ATTEMPT_PATHS = {
    stage: ROOT / f"results/development/gse164378_3p_gse155673_{stage}_attempt_v1.jsonl"
    for stage in STAGE_PATHS
}
EXECUTION_CLAIM_PATHS = {
    stage: ROOT
    / f"results/development/gse164378_3p_gse155673_{stage}_execution_consumed_v1.json"
    for stage in STAGE_PATHS
}
ACCESS_JOURNAL_PATHS = {
    stage: ROOT / f"results/development/gse164378_3p_gse155673_{stage}_access_v1.jsonl"
    for stage in ("source", "rna", "adt")
}
ATTEMPT_TAGS = {
    "source": SOURCE_ATTEMPT_TAG,
    "rna": RNA_ATTEMPT_TAG,
    "adt": ADT_ATTEMPT_TAG,
    "prediction": PREDICTION_ATTEMPT_TAG,
    "score": SCORE_ATTEMPT_TAG,
}
COMPLETION_TAGS = {
    "source": SOURCE_TAG,
    "rna": RNA_TAG,
    "adt": ADT_TAG,
    "prediction": PREDICTION_TAG,
    "score": RESULT_TAG,
}


class ConfirmationRefusal(RuntimeError):
    """A prespecified terminal refusal."""

    def __init__(self, code: str, details: dict[str, Any] | None = None):
        if not code or any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
            for character in code
        ):
            raise ValueError("refusal codes must be uppercase identifiers")
        super().__init__(code)
        self.code = code
        self.details = details


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
        raise PermissionError(f"{path.name} must contain one JSON object")
    return value


def _write_json_x(
    path: Path, payload: dict[str, Any], *, mode: int = 0o644
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"one-shot output already exists: {path}")
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _jsonl_record_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _append_jsonl(path: Path, payload: dict[str, Any], *, create: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_APPEND
    if create:
        flags |= os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "ab") as stream:
        stream.write(_jsonl_record_bytes(payload))
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
    raise PermissionError(
        "private assay-derived artifacts must remain outside the repository"
    )


def _validate_public_payload(value: Any, key: str | None = None) -> None:
    forbidden = {
        "states",
        "barcodes",
        "cell_ids",
        "selected",
        "selected_barcodes",
        "pool_by_cell",
    }
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key in forbidden:
                raise PermissionError(
                    f"public payload contains private key {child_key}"
                )
            _validate_public_payload(child, child_key)
    elif isinstance(value, list):
        for child in value:
            _validate_public_payload(child, key)
    elif isinstance(value, str) and (value.startswith("/") or value.startswith("~")):
        raise PermissionError(f"public payload contains a local path in {key}")


def _runtime_environment() -> dict[str, Any]:
    return {
        "python": {
            "implementation": sys.implementation.name,
            "version": ".".join(str(value) for value in sys.version_info[:3]),
        },
        "packages": {
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
        },
    }


def _require_runtime_environment() -> dict[str, Any]:
    specification = _read_json(DEFAULT_RUNTIME)
    observed = _runtime_environment()
    if specification.get("required_runtime") != observed:
        raise PermissionError("runtime environment differs from frozen specification")
    return observed


def _remote_tag_commit(tag: str) -> str:
    output = subprocess.run(
        [
            "git",
            "ls-remote",
            "--tags",
            PUBLIC_ORIGIN,
            f"refs/tags/{tag}",
            f"refs/tags/{tag}^{{}}",
        ],
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
        published = _published_path_bytes(commit, relative)
        if hashlib.sha256(published).hexdigest() != _sha256(local):
            raise PermissionError(f"public tag does not bind local bytes: {relative}")
    return commit


def _published_path_bytes(commit: str, relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _require_commit_ancestor(ancestor: str, descendant: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise PermissionError(
            f"public execution lineage is not descendant: {ancestor} -> {descendant}"
        )


def _require_public_attempt_snapshot(
    stage: str, started: dict[str, Any], protocol_commit: str
) -> str:
    attempt_commit = _remote_tag_commit(ATTEMPT_TAGS[stage])
    expected = {
        _relative(ATTEMPT_PATHS[stage]): _jsonl_record_bytes(started),
    }
    if stage in ACCESS_JOURNAL_PATHS:
        records = _access_records(stage)
        if not records:
            raise PermissionError("assay access journal is empty")
        expected[_relative(ACCESS_JOURNAL_PATHS[stage])] = _jsonl_record_bytes(
            records[0]
        )
    for relative, content in expected.items():
        if _published_path_bytes(attempt_commit, relative) != content:
            raise PermissionError(
                f"public attempt tag does not bind its pre-execution snapshot: {relative}"
            )
    _require_commit_ancestor(protocol_commit, attempt_commit)
    for key, commit in started.get("prerequisites", {}).items():
        if key.endswith("_commit"):
            _require_commit_ancestor(str(commit), attempt_commit)
    return attempt_commit


def _marker_record(row: dict[str, Any]) -> dict[str, str]:
    axis = row.get("adt_axis")
    if not isinstance(axis, dict):
        raise PermissionError("each marker requires one ADT axis")
    output = {
        "marker_id": row.get("marker_id"),
        "rna_symbol": row.get("rna_symbol"),
        "rna_ensembl_id": row.get("held_rna_ensembl_id"),
        "axis_id": axis.get("axis_id"),
        "catalog": str(axis.get("catalog")),
        "sequence": axis.get("sequence"),
        "clone": str(axis.get("clone")),
        "source_adt_feature": axis.get("source_feature_label"),
        "held_adt_feature": axis.get("held_feature_id"),
        "held_adt_label": axis.get("held_feature_label"),
    }
    if any(not isinstance(value, str) or not value for value in output.values()):
        raise PermissionError("marker mapping contains an empty field")
    if output["held_adt_feature"] != f"TotalSeq-{output['catalog']}":
        raise PermissionError("held ADT feature does not bind the exact catalog")
    return output


def _candidate() -> dict[str, Any]:
    value = _read_json(DEFAULT_CANDIDATE)
    if value.get("schema") != "gse164378-3p-gse155673-candidate-designation/1.0":
        raise PermissionError("candidate designation schema differs")
    contract = value.get("marker_contract")
    markers = contract.get("markers") if isinstance(contract, dict) else None
    if (
        not isinstance(markers, list)
        or len(markers) != 24
        or contract.get("marker_count") != 24
        or contract.get("ordered_rna_by_adt_pairs_before_support") != 24**2
    ):
        raise PermissionError("candidate must freeze exactly 24 cognates")
    normalized = [_marker_record(row) for row in markers]
    if _canonical_json_sha256(normalized) != FROZEN_MARKER_AXIS_SHA256:
        raise PermissionError("ordered frozen marker records differ")
    for key in (
        "marker_id",
        "rna_symbol",
        "rna_ensembl_id",
        "axis_id",
        "catalog",
        "source_adt_feature",
        "held_adt_feature",
    ):
        if len({row[key] for row in normalized}) != 24:
            raise PermissionError(f"candidate marker {key} must be one-to-one")
    source = value.get("source_donors")
    held = value.get("held_donors")
    if not isinstance(source, list) or not isinstance(held, list):
        raise PermissionError("candidate donor axes are missing")
    if _canonical_json_sha256(source) != FROZEN_SOURCE_DONOR_AXIS_SHA256:
        raise PermissionError("ordered frozen source donor records differ")
    if _canonical_json_sha256(held) != FROZEN_HELD_DONOR_AXIS_SHA256:
        raise PermissionError("ordered frozen held donor records differ")
    by_source = {row.get("donor_id"): row for row in source}
    if set(by_source) != set(SOURCE_DONORS):
        raise PermissionError("source donor axis differs")
    expected_roles = {
        donor: "calibration" if donor in CALIBRATION_DONORS else "validation"
        for donor in SOURCE_DONORS
    }
    if any(
        by_source[donor].get("frozen_role") != role
        for donor, role in expected_roles.items()
    ):
        raise PermissionError("source donor split differs")
    if [_held_id(row) for row in held] != list(HELD_DONORS):
        raise PermissionError("held donor axis or order differs")
    _held_groups(value)
    if value.get("cell_budget_per_donor") != CELL_BUDGET:
        raise PermissionError("cell budget differs")
    split = value.get("source_split")
    if (
        not isinstance(split, dict)
        or split.get("calibration") != list(CALIBRATION_DONORS)
        or split.get("validation") != list(VALIDATION_DONORS)
        or split.get("validation_batch_components")
        != [list(component) for component in VALIDATION_COMPONENTS]
    ):
        raise PermissionError("source split arrays or validation components differ")
    if value.get("cell_selection_salt") != CELL_SELECTION_SALT:
        raise PermissionError("cell-selection salt differs")
    return value


def _protocol() -> dict[str, Any]:
    value = _read_json(DEFAULT_PROTOCOL)
    if value.get("schema") != "gse164378-3p-gse155673-external-study-protocol/1.0":
        raise PermissionError("protocol schema differs")
    source = value.get("source_design", {})
    feature = value.get("cell_and_feature_contract", {})
    lock = feature.get("source_marker_lock", {})
    held = feature.get("held_support", {})
    grid = value.get("primary_estimator", {}).get("grid", {})
    inference = value.get("held_inference", {})
    transport = value.get("transport_contract", {})
    expected_tags = {
        "protocol": PROTOCOL_TAG,
        "source_attempt": SOURCE_ATTEMPT_TAG,
        "source": SOURCE_TAG,
        "rna_attempt": RNA_ATTEMPT_TAG,
        "rna": RNA_TAG,
        "adt_attempt": ADT_ATTEMPT_TAG,
        "adt": ADT_TAG,
        "prediction_attempt": PREDICTION_ATTEMPT_TAG,
        "prediction": PREDICTION_TAG,
        "score_authorization": SCORE_AUTHORIZATION_TAG,
        "score_attempt": SCORE_ATTEMPT_TAG,
        "result": RESULT_TAG,
    }
    if (
        source.get("calibration_donors") != list(CALIBRATION_DONORS)
        or source.get("validation_donors") != list(VALIDATION_DONORS)
        or source.get("validation_batch_components")
        != [list(component) for component in VALIDATION_COMPONENTS]
        or feature.get("candidate_markers") != 24
        or feature.get("cell_budget_per_donor") != CELL_BUDGET
        or lock.get("minimum_locked_markers") != MINIMUM_SOURCE_LOCKED_MARKERS
        or held.get("minimum_donors_per_marker_per_modality")
        != MINIMUM_HELD_SUPPORTED_DONORS
        or held.get("minimum_pair_fraction_per_donor") != MINIMUM_HELD_PAIR_FRACTION
        or grid.get("heterogeneity_penalty") != list(HETEROGENEITY_GRID)
        or grid.get("ridge_penalty") != list(RIDGE_GRID)
        or grid.get("graph_penalty") != list(GRAPH_GRID)
        or grid.get("transport_multiplier") != list(TRANSPORT_GRID)
        or inference.get("bootstrap_draws") != BOOTSTRAPS
        or inference.get("bootstrap_seed") != BOOTSTRAP_SEED
        or inference.get("minimum_relative_loss_reduction") != 0.05
        or inference.get("minimum_favorable_donors") != 11
        or inference.get("directional_alpha") != ALPHA
        or transport.get("download_attempts") != DOWNLOAD_ATTEMPTS
        or transport.get("partial_file_rule") != "delete after each failed attempt"
        or transport.get("journal_rule")
        != "append DOWNLOADED_AND_HASHED only after a transport attempt completes and observed bytes and SHA-256 are computed; then compare to the manifest; record and terminate on mismatch"
        or transport.get("exhaustion") != "terminal stage failure"
        or value.get("public_tags") != expected_tags
        or value.get("public_tag_sequence") != list(expected_tags)
    ):
        raise PermissionError("protocol constants differ from the runner")
    if DESTROYED_LINK_SALT not in value.get("primary_estimator", {}).get(
        "destroyed_link_control", ""
    ):
        raise PermissionError("destroyed-link salt differs")
    return value


def _markers(candidate: dict[str, Any]) -> list[dict[str, str]]:
    return [_marker_record(row) for row in candidate["marker_contract"]["markers"]]


def _manifest() -> dict[str, Any]:
    value = _read_json(DEFAULT_MANIFEST)
    if value.get("schema") != "gse164378-3p-gse155673-source-manifest/1.0":
        raise PermissionError("source manifest schema differs")
    return value


def _held_id(row: dict[str, Any]) -> str:
    value = row.get("donor_file_id")
    if not isinstance(value, str) or not value:
        raise PermissionError("held donor identifier differs")
    return value


def _held_groups(candidate: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    phenotype: dict[str, str] = {}
    severity: dict[str, str] = {}
    for row in candidate["held_donors"]:
        donor = _held_id(row)
        raw_phenotype = row.get("phenotype")
        if raw_phenotype not in {"Healthy", "COVID-19"}:
            raise PermissionError("held phenotype label differs")
        phenotype[donor] = "healthy" if raw_phenotype == "Healthy" else "covid"
        raw_severity = row.get("severity")
        if raw_severity == "Moderate":
            severity[donor] = "moderate"
        elif raw_severity == "Severe":
            severity[donor] = "severe"
        elif raw_severity is not None:
            raise PermissionError("held severity label differs")
    expected_phenotype = {
        donor: "healthy" if donor in HEALTHY_DONORS else "covid"
        for donor in HELD_DONORS
    }
    expected_severity = {
        **{donor: "moderate" for donor in MODERATE_DONORS},
        **{donor: "severe" for donor in SEVERE_DONORS},
    }
    if phenotype != expected_phenotype:
        raise PermissionError("held phenotype strata differ")
    if severity != expected_severity:
        raise PermissionError("held severity strata differ")
    return phenotype, severity


def _url_filename(url: str) -> str:
    name = Path(urlparse(url).path).name
    if not name:
        raise ConfirmationRefusal("URL_HAS_NO_FILENAME")
    return name


def _file_records(
    value: Any, path: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], dict[str, Any]]]:
    output: list[tuple[tuple[str, ...], dict[str, Any]]] = []
    if isinstance(value, dict):
        if isinstance(value.get("url"), str) and isinstance(value.get("bytes"), int):
            output.append((path, value))
        for key, child in value.items():
            output.extend(_file_records(child, (*path, str(key))))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            output.extend(_file_records(child, (*path, str(index))))
    return output


def _record_name(record: dict[str, Any]) -> str:
    for key in ("filename", "name", "path"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return Path(value).name
    return _url_filename(record["url"])


def _manifest_record(
    manifest: dict[str, Any], reference: str | dict[str, Any]
) -> dict[str, Any]:
    if isinstance(reference, dict):
        record = reference
    else:
        matches = []
        for path, candidate in _file_records(manifest):
            identities = {
                _record_name(candidate),
                str(candidate.get("file_id", "")),
                str(candidate.get("id", "")),
                str(candidate.get("role", "")),
                path[-1] if path else "",
            }
            if reference in identities:
                matches.append(candidate)
        if len(matches) != 1:
            raise PermissionError(f"manifest reference is not unique: {reference}")
        record = matches[0]
    name = _record_name(record)
    if (
        not isinstance(record.get("url"), str)
        or not isinstance(record.get("bytes"), int)
        or record["bytes"] <= 0
        or name != _url_filename(record["url"])
    ):
        raise PermissionError("manifest file record differs")
    digest = record.get("sha256")
    if digest is not None and (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise PermissionError("manifest SHA-256 is malformed")
    return {**record, "filename": name}


def _source_file(manifest: dict[str, Any], filename: str) -> dict[str, Any]:
    return _manifest_record(manifest, filename)


def _access_records(stage: str) -> list[dict[str, Any]]:
    if stage not in ACCESS_JOURNAL_PATHS:
        return []
    with ACCESS_JOURNAL_PATHS[stage].open() as stream:
        rows = [
            json.loads(line, object_pairs_hook=_strict_object)
            for line in stream
            if line.strip()
        ]
    if not rows or rows[0].get("status") != "OPENED_BEFORE_ASSAY_ACCESS":
        raise PermissionError("assay access journal header differs")
    seen: set[tuple[str, str]] = set()
    for row in rows[1:]:
        identity = (str(row.get("modality")), str(row.get("unit_id")))
        if (
            row.get("schema") != "gse164378-3p-gse155673-assay-access/1.0"
            or row.get("stage") != stage
            or row.get("status") != "DOWNLOADED_AND_HASHED"
            or identity in seen
            or not isinstance(row.get("observed_bytes"), int)
            or not isinstance(row.get("observed_sha256"), str)
            or len(row["observed_sha256"]) != 64
        ):
            raise PermissionError("assay access journal record differs")
        seen.add(identity)
    return rows


def _append_assay_access(
    stage: str,
    modality: str,
    unit_id: str,
    record: dict[str, Any],
    destination: Path,
) -> None:
    existing = _access_records(stage)
    if any(
        row.get("modality") == modality and row.get("unit_id") == unit_id
        for row in existing[1:]
    ):
        raise PermissionError("assay access journal repeats a unit modality")
    payload = {
        "schema": "gse164378-3p-gse155673-assay-access/1.0",
        "stage": stage,
        "status": "DOWNLOADED_AND_HASHED",
        "created_at_utc": _timestamp(),
        "unit_id": unit_id,
        "modality": modality,
        "url": record["url"],
        "filename": destination.name,
        "expected_bytes": record["bytes"],
        "expected_sha256": record.get("sha256"),
        "observed_bytes": destination.stat().st_size,
        "observed_sha256": _sha256(destination),
    }
    _validate_public_payload(payload)
    _append_jsonl(ACCESS_JOURNAL_PATHS[stage], payload, create=False)


def _fetch(
    record: dict[str, Any],
    destination: Path,
    *,
    stage: str | None = None,
    modality: str | None = None,
    unit_id: str | None = None,
) -> Path:
    frozen = _manifest_record({"record": record}, record)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.exists():
        raise FileExistsError(f"scratch file already exists: {destination}")
    temporary = destination.with_suffix(destination.suffix + ".part")
    for _ in range(DOWNLOAD_ATTEMPTS):
        temporary.unlink(missing_ok=True)
        try:
            with urlopen(frozen["url"]) as response:
                descriptor = os.open(
                    temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                )
                with os.fdopen(descriptor, "wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
        except (OSError, EOFError, IncompleteRead):
            temporary.unlink(missing_ok=True)
            continue
        temporary.replace(destination)
        break
    else:
        raise ConfirmationRefusal("DOWNLOAD_ATTEMPTS_EXHAUSTED")
    if stage is not None:
        if modality is None or unit_id is None or stage not in ACCESS_JOURNAL_PATHS:
            raise PermissionError("journaled access lacks a stage, modality, or unit")
        _append_assay_access(stage, modality, unit_id, frozen, destination)
    if destination.stat().st_size != frozen["bytes"]:
        raise ConfirmationRefusal("DOWNLOADED_FILE_SIZE_DIFFERS")
    digest = frozen.get("sha256")
    if digest is not None and _sha256(destination) != digest:
        raise ConfirmationRefusal("DOWNLOADED_FILE_HASH_DIFFERS")
    return destination


def _open_text(path: Path):
    return (
        gzip.open(path, "rt", newline="")
        if path.suffix == ".gz"
        else path.open("r", newline="")
    )


def _read_tsv(path: Path) -> list[list[str]]:
    with _open_text(path) as stream:
        rows = [line.rstrip("\r\n").split("\t") for line in stream]
    if not rows or any(not row for row in rows):
        raise ConfirmationRefusal("TABULAR_AXIS_IS_EMPTY_OR_MALFORMED")
    return rows


def _read_barcodes(path: Path) -> list[str]:
    rows = _read_tsv(path)
    barcodes = [row[0] for row in rows]
    if any(len(row) != 1 for row in rows) or len(set(barcodes)) != len(barcodes):
        raise ConfirmationRefusal("BARCODE_AXIS_IS_NOT_UNIQUE")
    return barcodes


def _deterministic_selection(values: list[str], donor: str) -> list[str]:
    if len(values) < CELL_BUDGET:
        raise ConfirmationRefusal("DONOR_HAS_FEWER_THAN_384_CELLS")
    return sorted(
        values,
        key=lambda value: (
            hashlib.sha256(
                f"{CELL_SELECTION_SALT}|{donor}|{value}".encode()
            ).hexdigest(),
            value,
        ),
    )[:CELL_BUDGET]


def _column_lookup(axis: list[str], selected: list[str]) -> dict[int, int]:
    lookup = {value: index for index, value in enumerate(axis)}
    if len(lookup) != len(axis) or any(value not in lookup for value in selected):
        raise ConfirmationRefusal("SELECTED_CELL_AXIS_DOES_NOT_MATCH_BARCODES")
    columns = [lookup[value] for value in selected]
    if len(set(columns)) != len(columns):
        raise ConfirmationRefusal("SELECTED_CELL_AXIS_COLLAPSES")
    return {source: target for target, source in enumerate(columns)}


def _resolve_feature_rows(
    features: list[list[str]],
    markers: list[dict[str, str]],
    *,
    modality: str,
    held: bool,
) -> dict[int, int]:
    output: dict[int, int] = {}
    for marker_index, marker in enumerate(markers):
        if modality == "rna":
            if held:
                matches = [
                    index
                    for index, row in enumerate(features)
                    if len(row) >= 3
                    and row[0] == marker["rna_ensembl_id"]
                    and row[1] == marker["rna_symbol"]
                    and row[2] == "Gene Expression"
                ]
            else:
                matches = [
                    index
                    for index, row in enumerate(features)
                    if len(row) >= 2
                    and row[0] == marker["rna_symbol"]
                    and row[1] == marker["rna_symbol"]
                ]
        else:
            if held:
                matches = [
                    index
                    for index, row in enumerate(features)
                    if len(row) >= 3
                    and row[0] == marker["held_adt_feature"]
                    and row[1] == marker["held_adt_label"]
                    and row[2] == "Antibody Capture"
                ]
            else:
                matches = [
                    index
                    for index, row in enumerate(features)
                    if len(row) >= 2
                    and row[0] == marker["source_adt_feature"]
                    and row[1] == marker["source_adt_feature"]
                ]
        if len(matches) != 1 or matches[0] in output:
            raise ConfirmationRefusal("FEATURE_MAPPING_IS_NOT_EXACTLY_ONE_TO_ONE")
        output[matches[0]] = marker_index
    return output


def _gene_expression_rows(features: list[list[str]]) -> set[int]:
    rows = {
        index
        for index, row in enumerate(features)
        if len(row) >= 3 and row[2] == "Gene Expression"
    }
    if not rows:
        raise ConfirmationRefusal("GENE_EXPRESSION_FEATURE_AXIS_IS_EMPTY")
    return rows


def _read_matrix_market_subset(
    path: Path,
    *,
    expected_rows: int,
    expected_columns: int,
    selected_columns: dict[int, int],
    retained_rows: dict[int, int],
    authorized_rows: set[int],
    collect_totals: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read only authorized row values for selected columns.

    Row and column indices are validated for every coordinate.  The numeric
    token is deliberately not converted when its row or column is outside the
    stage authorization, which is the held combined-file firewall.
    """

    if not set(retained_rows) <= authorized_rows:
        raise PermissionError("retained rows exceed the modality authorization")
    values = np.zeros(
        (len(selected_columns), len(set(retained_rows.values()))), dtype=np.int64
    )
    totals = np.zeros(len(selected_columns), dtype=np.int64) if collect_totals else None
    seen: set[tuple[int, int]] = set()
    dimensions: tuple[int, int, int] | None = None
    observed = 0
    with gzip.open(path, "rt", newline="") as stream:
        if (
            stream.readline().strip()
            != "%%MatrixMarket matrix coordinate integer general"
        ):
            raise ConfirmationRefusal("MATRIX_MARKET_HEADER_DIFFERS")
        for raw in stream:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            tokens = line.split()
            if len(tokens) != 3:
                raise ConfirmationRefusal("MATRIX_MARKET_DIMENSIONS_ARE_MALFORMED")
            dimensions = tuple(int(token) for token in tokens)
            break
        if dimensions is None:
            raise ConfirmationRefusal("MATRIX_MARKET_DIMENSIONS_ARE_MISSING")
        rows, columns, declared = dimensions
        if rows != expected_rows or columns != expected_columns or declared < 0:
            raise ConfirmationRefusal("MATRIX_MARKET_DIMENSIONS_DIFFER")
        for raw in stream:
            tokens = raw.split()
            if len(tokens) != 3:
                raise ConfirmationRefusal("MATRIX_MARKET_ENTRY_IS_MALFORMED")
            try:
                source_row = int(tokens[0]) - 1
                source_column = int(tokens[1]) - 1
            except ValueError as error:
                raise ConfirmationRefusal("MATRIX_MARKET_INDEX_IS_INVALID") from error
            if not (0 <= source_row < rows and 0 <= source_column < columns):
                raise ConfirmationRefusal("MATRIX_MARKET_INDEX_IS_OUT_OF_RANGE")
            observed += 1
            target_column = selected_columns.get(source_column)
            if target_column is None or source_row not in authorized_rows:
                continue
            try:
                numeric = int(tokens[2])
            except ValueError as error:
                raise ConfirmationRefusal(
                    "AUTHORIZED_MATRIX_VALUE_IS_NOT_INTEGER"
                ) from error
            if numeric < 0:
                raise ConfirmationRefusal("AUTHORIZED_MATRIX_VALUE_IS_NEGATIVE")
            identity = (source_row, source_column)
            if identity in seen:
                raise ConfirmationRefusal("AUTHORIZED_MATRIX_COORDINATE_IS_DUPLICATED")
            seen.add(identity)
            if totals is not None:
                totals[target_column] += numeric
            target_row = retained_rows.get(source_row)
            if target_row is not None:
                values[target_column, target_row] += numeric
    if observed != declared:
        raise ConfirmationRefusal("MATRIX_MARKET_NONZERO_COUNT_DIFFERS")
    if totals is not None and np.any(totals <= 0):
        raise ConfirmationRefusal("SELECTED_CELL_HAS_ZERO_RNA_LIBRARY_SIZE")
    return values, totals


def _protocol_paths() -> tuple[str, ...]:
    return PROTOCOL_BINDINGS


def _completion_paths(stage: str) -> tuple[str, ...]:
    paths = (
        _relative(STAGE_PATHS[stage]),
        _relative(ATTEMPT_PATHS[stage]),
        _relative(EXECUTION_CLAIM_PATHS[stage]),
    )
    if stage in ACCESS_JOURNAL_PATHS:
        paths += (_relative(ACCESS_JOURNAL_PATHS[stage]),)
    return paths


def _stage_prerequisites(stage: str) -> dict[str, Any]:
    _protocol()
    if stage == "source":
        commit = _require_public_tag(PROTOCOL_TAG, _protocol_paths())
        _require_commit_ancestor(commit, "HEAD")
        return {"protocol_tag": PROTOCOL_TAG, "protocol_commit": commit}
    previous = {
        "rna": ("source", SOURCE_TAG),
        "adt": ("rna", RNA_TAG),
        "prediction": ("adt", ADT_TAG),
    }
    if stage in previous:
        prior, tag = previous[stage]
        _, commit = _require_completed_stage_artifact(
            tag, STAGE_PATHS[prior], prior, require_success=True
        )
        _require_commit_ancestor(commit, "HEAD")
        return {
            f"{prior}_tag": tag,
            f"{prior}_commit": commit,
            f"{prior}_sha256": _sha256(STAGE_PATHS[prior]),
        }
    if stage == "score":
        _, prediction_commit = _require_completed_stage_artifact(
            PREDICTION_TAG,
            DEFAULT_PREDICTION,
            "prediction",
            require_success=True,
        )
        authorization_commit = _require_public_tag(
            SCORE_AUTHORIZATION_TAG, (_relative(DEFAULT_SCORE_AUTHORIZATION),)
        )
        _require_commit_ancestor(prediction_commit, authorization_commit)
        _require_commit_ancestor(authorization_commit, "HEAD")
        _require_score_authorization()
        return {
            "prediction_tag": PREDICTION_TAG,
            "prediction_commit": prediction_commit,
            "prediction_sha256": _sha256(DEFAULT_PREDICTION),
            "score_authorization_tag": SCORE_AUTHORIZATION_TAG,
            "score_authorization_commit": authorization_commit,
            "score_authorization_sha256": _sha256(DEFAULT_SCORE_AUTHORIZATION),
        }
    raise ValueError(f"unknown stage: {stage}")


def _access_summary(stage: str) -> dict[str, Any] | None:
    if stage not in ACCESS_JOURNAL_PATHS:
        return None
    records = _access_records(stage)
    return {
        "journal_path": _relative(ACCESS_JOURNAL_PATHS[stage]),
        "journal_sha256": _sha256(ACCESS_JOURNAL_PATHS[stage]),
        "downloaded_and_hashed_files": records[1:],
    }


def _attempt_records(stage: str) -> list[dict[str, Any]]:
    with ATTEMPT_PATHS[stage].open() as stream:
        rows = [
            json.loads(line, object_pairs_hook=_strict_object)
            for line in stream
            if line.strip()
        ]
    if not all(isinstance(row, dict) for row in rows):
        raise PermissionError("attempt ledger is malformed")
    return rows


def _validate_started_attempt(
    stage: str,
    started: dict[str, Any],
    runtime: dict[str, Any],
    protocol_commit: str,
) -> None:
    if (
        started.get("schema") != "gse164378-3p-gse155673-stage-attempt/1.0"
        or started.get("stage") != stage
        or started.get("status") != "STARTED"
        or started.get("attempt_tag_required_before_assay_access")
        != ATTEMPT_TAGS[stage]
        or started.get("prerequisites") != _stage_prerequisites(stage)
        or started.get("protocol_commit") != protocol_commit
        or started.get("runtime_environment") != runtime
        or started.get("one_shot") is not True
    ):
        raise PermissionError("STARTED event differs from the public stage contract")


def claim_stage(stage: str) -> dict[str, Any]:
    if stage not in STAGE_PATHS:
        raise ValueError(f"unknown stage: {stage}")
    occupied = [STAGE_PATHS[stage], ATTEMPT_PATHS[stage], EXECUTION_CLAIM_PATHS[stage]]
    if stage in ACCESS_JOURNAL_PATHS:
        occupied.append(ACCESS_JOURNAL_PATHS[stage])
    if any(path.exists() for path in occupied):
        raise PermissionError(f"stage has already been claimed: {stage}")
    runtime = _require_runtime_environment()
    protocol_commit = _require_public_tag(PROTOCOL_TAG, _protocol_paths())
    payload = {
        "schema": "gse164378-3p-gse155673-stage-attempt/1.0",
        "stage": stage,
        "status": "STARTED",
        "created_at_utc": _timestamp(),
        "attempt_tag_required_before_assay_access": ATTEMPT_TAGS[stage],
        "prerequisites": _stage_prerequisites(stage),
        "protocol_commit": protocol_commit,
        "runtime_environment": runtime,
        "one_shot": True,
    }
    if stage in ACCESS_JOURNAL_PATHS:
        header = {
            "schema": "gse164378-3p-gse155673-assay-access/1.0",
            "stage": stage,
            "status": "OPENED_BEFORE_ASSAY_ACCESS",
            "created_at_utc": payload["created_at_utc"],
            "protocol_commit": protocol_commit,
            "runtime_environment": runtime,
        }
        _append_jsonl(ACCESS_JOURNAL_PATHS[stage], header, create=True)
    _append_jsonl(ATTEMPT_PATHS[stage], payload, create=True)
    return payload


def _require_public_attempt(stage: str) -> dict[str, Any]:
    rows = _attempt_records(stage)
    if len(rows) != 1:
        raise PermissionError("attempt ledger must contain one STARTED event")
    runtime = _require_runtime_environment()
    protocol_commit = _require_public_tag(PROTOCOL_TAG, _protocol_paths())
    _validate_started_attempt(stage, rows[0], runtime, protocol_commit)
    paths = [_relative(ATTEMPT_PATHS[stage])]
    if stage in ACCESS_JOURNAL_PATHS:
        records = _access_records(stage)
        if (
            len(records) != 1
            or records[0].get("created_at_utc") != rows[0]["created_at_utc"]
            or records[0].get("protocol_commit") != protocol_commit
            or records[0].get("runtime_environment") != runtime
        ):
            raise PermissionError("assay access began before the public attempt")
        paths.append(_relative(ACCESS_JOURNAL_PATHS[stage]))
    attempt_commit = _require_public_tag(ATTEMPT_TAGS[stage], paths)
    _require_commit_ancestor(protocol_commit, attempt_commit)
    for key, commit in rows[0]["prerequisites"].items():
        if key.endswith("_commit"):
            _require_commit_ancestor(str(commit), attempt_commit)
    return rows[0]


def _run_claimed_stage(
    stage: str, body: Callable[[], dict[str, Any]]
) -> dict[str, Any]:
    if STAGE_PATHS[stage].exists() or EXECUTION_CLAIM_PATHS[stage].exists():
        raise PermissionError(f"one-shot stage is already consumed: {stage}")
    started = _require_public_attempt(stage)
    executing = {
        "schema": "gse164378-3p-gse155673-stage-attempt/1.0",
        "stage": stage,
        "status": "EXECUTING_CONSUMED",
        "created_at_utc": _timestamp(),
        "protocol_commit": started["protocol_commit"],
        "runtime_environment": started["runtime_environment"],
        "interruption_consumes_stage": True,
    }
    _write_json_x(EXECUTION_CLAIM_PATHS[stage], executing)
    _append_jsonl(ATTEMPT_PATHS[stage], executing, create=False)
    try:
        payload = dict(body())
        payload["protocol_commit"] = started["protocol_commit"]
        payload["runtime_environment"] = started["runtime_environment"]
        summary = _access_summary(stage)
        if summary is not None:
            payload["assay_access"] = summary
        terminal_status = str(payload["status"])
        _validate_public_payload(payload)
        _write_json_x(STAGE_PATHS[stage], payload)
    except BaseException as error:
        if STAGE_PATHS[stage].exists():
            raise
        if isinstance(error, ConfirmationRefusal):
            code = error.code
            refusal_details = error.details
        elif isinstance(error, CouplingEstimationRefusal):
            code = "COUPLING_ESTIMATION_REFUSAL"
            refusal_details = None
        else:
            code = f"UNEXPECTED_{type(error).__name__.upper()}"
            refusal_details = None
        payload = {
            "schema": f"gse164378-3p-gse155673-{stage}-terminal/1.0",
            "status": f"TERMINAL_{stage.upper()}_REFUSAL",
            "created_at_utc": _timestamp(),
            "refusal_code": code,
            "attempt_created_at_utc": started["created_at_utc"],
            "protocol_commit": started["protocol_commit"],
            "runtime_environment": started["runtime_environment"],
            "rerun_or_rescue_permitted": False,
        }
        if refusal_details is not None:
            payload["refusal_details"] = refusal_details
        summary = _access_summary(stage)
        if summary is not None:
            payload["assay_access"] = summary
        terminal_status = payload["status"]
        _validate_public_payload(payload)
        _write_json_x(STAGE_PATHS[stage], payload)
    finished = {
        "schema": "gse164378-3p-gse155673-stage-attempt/1.0",
        "stage": stage,
        "status": "FINISHED",
        "created_at_utc": _timestamp(),
        "terminal_status": terminal_status,
        "output_sha256": _sha256(STAGE_PATHS[stage]),
        "protocol_commit": started["protocol_commit"],
        "runtime_environment": started["runtime_environment"],
    }
    _append_jsonl(ATTEMPT_PATHS[stage], finished, create=False)
    return payload


def _require_completed(stage: str, *, require_success: bool = True) -> dict[str, Any]:
    if not all(
        path.exists()
        for path in (
            STAGE_PATHS[stage],
            ATTEMPT_PATHS[stage],
            EXECUTION_CLAIM_PATHS[stage],
        )
    ):
        raise PermissionError(f"stage is not complete: {stage}")
    rows = _attempt_records(stage)
    if len(rows) != 3 or rows[1] != _read_json(EXECUTION_CLAIM_PATHS[stage]):
        raise PermissionError(f"stage ledger is incomplete: {stage}")
    runtime = _require_runtime_environment()
    protocol_commit = _require_public_tag(PROTOCOL_TAG, _protocol_paths())
    _validate_started_attempt(stage, rows[0], runtime, protocol_commit)
    if (
        rows[1].get("status") != "EXECUTING_CONSUMED"
        or rows[2].get("status") != "FINISHED"
        or rows[2].get("output_sha256") != _sha256(STAGE_PATHS[stage])
    ):
        raise PermissionError(f"stage execution lineage differs: {stage}")
    value = _read_json(STAGE_PATHS[stage])
    if (
        value.get("protocol_commit") != protocol_commit
        or value.get("runtime_environment") != runtime
        or rows[2].get("terminal_status") != value.get("status")
        or (require_success and str(value.get("status", "")).startswith("TERMINAL_"))
    ):
        raise PermissionError(f"stage output differs from its ledger: {stage}")
    if stage in ACCESS_JOURNAL_PATHS and value.get("assay_access") != _access_summary(
        stage
    ):
        raise PermissionError(f"stage output differs from access journal: {stage}")
    return value


def _require_completed_stage_artifact(
    tag: str,
    path: Path,
    stage: str,
    *,
    require_success: bool,
) -> tuple[dict[str, Any], str]:
    if path != STAGE_PATHS[stage]:
        raise ValueError("stage path differs")
    commit = _require_public_tag(tag, _completion_paths(stage))
    result = _require_completed(stage, require_success=require_success)
    started = _attempt_records(stage)[0]
    attempt_commit = _require_public_attempt_snapshot(
        stage, started, str(result["protocol_commit"])
    )
    _require_commit_ancestor(attempt_commit, commit)
    _require_commit_ancestor(commit, "HEAD")
    return result, commit


def verify_stage(stage: str) -> dict[str, Any]:
    result, commit = _require_completed_stage_artifact(
        COMPLETION_TAGS[stage], STAGE_PATHS[stage], stage, require_success=False
    )
    return {
        "stage": stage,
        "status": "PUBLIC_STAGE_LINEAGE_VERIFIED",
        "commit": commit,
        "result": result,
    }


def _adt_states(counts: np.ndarray, barcodes: list[str], donor: str) -> np.ndarray:
    values = np.asarray(counts, dtype=np.int64)
    if (
        values.ndim != 2
        or values.shape[0] != CELL_BUDGET
        or len(barcodes) != CELL_BUDGET
    ):
        raise ValueError("ADT counts and barcode axis differ from the cell budget")
    states = np.zeros(values.shape, dtype=np.uint8)
    for marker in range(values.shape[1]):
        order = sorted(
            range(CELL_BUDGET),
            key=lambda index: (
                values[index, marker],
                hashlib.sha256(
                    f"{ADT_TIE_SALT}|{donor}|{marker}|{barcodes[index]}".encode()
                ).hexdigest(),
                barcodes[index],
            ),
        )
        states[np.asarray(order[CELL_BUDGET // 2 :], dtype=int), marker] = 1
    return states


def _rna_axis_quality(counts: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(counts)
    if values.ndim != 2 or values.shape[0] != CELL_BUDGET:
        raise ValueError("RNA counts differ from the cell budget")
    prevalence = np.mean(values > 0, axis=0)
    return {
        "prevalence": prevalence,
        "valid": (prevalence >= MINIMUM_RNA_PREVALENCE)
        & (prevalence <= MAXIMUM_RNA_PREVALENCE),
    }


def _adt_axis_quality(counts: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(counts)
    if values.ndim != 2 or values.shape[0] != CELL_BUDGET:
        raise ValueError("ADT counts differ from the cell budget")
    distinct = np.empty(values.shape[1], dtype=np.int64)
    largest = np.empty(values.shape[1], dtype=float)
    for marker in range(values.shape[1]):
        _, frequencies = np.unique(values[:, marker], return_counts=True)
        distinct[marker] = len(frequencies)
        largest[marker] = float(frequencies.max() / CELL_BUDGET)
    return {
        "distinct_values": distinct,
        "largest_equal_value_fraction": largest,
        "valid": (distinct >= 2) & (largest <= MAXIMUM_ADT_EQUAL_VALUE_FRACTION),
    }


def _destroyed_adt(states: np.ndarray, barcodes: list[str], donor: str) -> np.ndarray:
    order = np.asarray(
        sorted(
            range(CELL_BUDGET),
            key=lambda index: (
                hashlib.sha256(
                    f"{DESTROYED_LINK_SALT}|{donor}|{barcodes[index]}".encode()
                ).hexdigest(),
                barcodes[index],
            ),
        ),
        dtype=int,
    )
    output = np.empty_like(states)
    output[order] = np.asarray(states)[np.roll(order, 1)]
    if not np.array_equal(output.sum(axis=0), np.asarray(states).sum(axis=0)):
        raise AssertionError("destroyed-link rotation changed ADT margins")
    return output


def _binary_tables(rna: np.ndarray, adt: np.ndarray) -> np.ndarray:
    first = np.asarray(rna, dtype=np.uint8)
    second = np.asarray(adt, dtype=np.uint8)
    if first.shape != second.shape or first.shape[0] != CELL_BUDGET:
        raise ValueError("RNA and ADT state panels differ")
    size = first.shape[1]
    output = np.empty((size, size, 2, 2), dtype=np.int64)
    for row in range(size):
        for column in range(size):
            output[row, column] = np.bincount(
                2 * first[:, row] + second[:, column], minlength=4
            ).reshape(2, 2)
    return output


def _informative(tables: np.ndarray) -> np.ndarray:
    values = np.asarray(tables)
    rows = values.sum(axis=-1)
    columns = values.sum(axis=-2)
    total = values.sum(axis=(-2, -1))
    return np.minimum(rows[..., 0], columns[..., 0]) > np.maximum(
        0, rows[..., 0] + columns[..., 0] - total
    )


def _margins(tables: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(tables)
    return values.sum(axis=-1), values.sum(axis=-2)


def _signed_poisson_deviance(table: np.ndarray) -> float:
    values = np.asarray(table, dtype=float)
    residual = poisson_independence_residuals(values, residual="deviance")
    determinant = float(values[0, 0] * values[1, 1] - values[0, 1] * values[1, 0])
    return float(np.sign(determinant) * np.linalg.norm(residual))


def _fractional_signed_deviance(table: np.ndarray) -> float:
    values = np.asarray(table, dtype=float)
    expected = np.outer(values.sum(axis=1), values.sum(axis=0)) / values.sum()
    positive = values > 0
    deviance = 2.0 * float(
        np.sum(values[positive] * np.log(values[positive] / expected[positive]))
    )
    determinant = float(values[0, 0] * values[1, 1] - values[0, 1] * values[1, 0])
    return float(np.sign(determinant) * math.sqrt(max(deviance, 0.0)))


def _classical_table(
    coordinate: float, rows: np.ndarray, columns: np.ndarray
) -> np.ndarray:
    row = np.asarray(rows, dtype=float)
    column = np.asarray(columns, dtype=float)
    total = float(row.sum())
    lower = max(0.0, row[0] + column[0] - total)
    upper = min(row[0], column[0])

    def at(value: float) -> np.ndarray:
        return np.asarray(
            [[value, row[0] - value], [column[0] - value, row[1] - column[0] + value]]
        )

    if upper <= lower:
        return at(lower)
    epsilon = min(1e-10, 0.25 * (upper - lower))
    left, right = lower + epsilon, upper - epsilon
    target = min(
        max(float(coordinate), _fractional_signed_deviance(at(left))),
        _fractional_signed_deviance(at(right)),
    )
    for _ in range(96):
        middle = (left + right) / 2.0
        if _fractional_signed_deviance(at(middle)) < target:
            left = middle
        else:
            right = middle
    return at((left + right) / 2.0)


def _knn_incidence(profiles: np.ndarray) -> np.ndarray:
    values = np.asarray(profiles, dtype=float).T
    if (
        values.ndim != 2
        or values.shape[0] < MINIMUM_SOURCE_LOCKED_MARKERS
        or values.shape[1] < 3
    ):
        raise ValueError("source marker profile dimensions differ")
    scale = values.std(axis=1, ddof=1)
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise CouplingEstimationRefusal("source marker profile has zero variance")
    standardized = (values - values.mean(axis=1, keepdims=True)) / scale[:, None]
    edges: set[tuple[int, int]] = set()
    for marker in range(values.shape[0]):
        candidates = np.asarray(
            [index for index in range(values.shape[0]) if index != marker]
        )
        distances = np.linalg.norm(
            standardized[candidates] - standardized[marker][None, :], axis=1
        )
        order = candidates[np.lexsort((candidates, distances))]
        edges.update(
            tuple(sorted((marker, int(neighbor))))
            for neighbor in order[:GRAPH_NEIGHBORS]
        )
    incidence = np.zeros((values.shape[0], len(edges)), dtype=float)
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
    values = np.asarray(tables, dtype=np.int64)
    size = values.shape[1]
    if config.graph_penalty == 0:
        first = second = np.eye(size, dtype=float)
    else:
        first = _knn_incidence(rna_profiles)
        second = _knn_incidence(adt_profiles)
    fit = fit_hierarchical_conditional_log_odds(
        values,
        first,
        second,
        heterogeneity_penalty=config.heterogeneity_penalty,
        ridge_penalty=config.ridge_penalty,
        graph_penalty=config.graph_penalty,
        minimum_informative_donors=2,
        maximum_condition_number=MAXIMUM_CONDITION_NUMBER,
    )
    if not fit.converged:
        raise CouplingEstimationRefusal("hierarchical fit did not converge")
    return {
        "family": "graph_regularized_exact_fixed_margin_hierarchical_coupling",
        "configuration": asdict(config),
        "transport_multiplier": config.transport_multiplier,
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


def _residual_pool(tables: np.ndarray) -> np.ndarray:
    values = np.asarray(tables)
    support = _informative(values)
    coordinates = np.full(support.shape, np.nan)
    for donor, row, column in np.argwhere(support):
        coordinates[donor, row, column] = _signed_poisson_deviance(
            values[donor, row, column]
        ) / math.sqrt(CELL_BUDGET)
    if np.any(support.sum(axis=0) < 2):
        raise CouplingEstimationRefusal("Poisson residual lacks two source donors")
    result = np.nanmean(coordinates, axis=0)
    if not np.isfinite(result).all():
        raise CouplingEstimationRefusal("Poisson residual is nonfinite")
    return result


def _fit_residual(tables: np.ndarray) -> dict[str, Any]:
    return {
        "family": "poisson_independence_signed_deviance_residual",
        "transport_multiplier": 1.0,
        "pooled_coordinate": _residual_pool(tables),
        "support_count": _informative(tables).sum(axis=0),
    }


def _fit_common_effect(tables: np.ndarray) -> dict[str, Any]:
    values = np.asarray(tables, dtype=np.int64)
    size = values.shape[1]
    fit = fit_structured_conditional_log_odds(
        values,
        np.eye(size),
        np.eye(size),
        initial_log_odds=np.zeros((size, size)),
        ridge_penalty=0,
        graph_penalty=0,
        minimum_informative_donors=2,
        maximum_condition_number=MAXIMUM_CONDITION_NUMBER,
        tolerance=1e-9,
    )
    return {
        "family": "common_effect_stratified_conditional_maximum_likelihood",
        "transport_multiplier": 1.0,
        "population_log_odds": fit.log_odds,
        "support_count": fit.support_count,
    }


def _fit_pooled_poisson(tables: np.ndarray) -> dict[str, Any]:
    pooled = np.asarray(tables, dtype=np.int64).sum(axis=0)
    if np.any(pooled <= 0):
        raise CouplingEstimationRefusal(
            "pooled saturated Poisson table has a zero cell"
        )
    log_odds = np.log(pooled[..., 0, 0]) + np.log(pooled[..., 1, 1])
    log_odds -= np.log(pooled[..., 0, 1]) + np.log(pooled[..., 1, 0])
    return {
        "family": "pooled_saturated_poisson_log_linear_interaction",
        "transport_multiplier": 1.0,
        "population_log_odds": log_odds,
    }


def _fit_paule_mandel(tables: np.ndarray) -> dict[str, Any]:
    values = np.asarray(tables, dtype=np.int64)
    coordinates = np.zeros(values.shape[:3])
    variances = np.zeros_like(coordinates)
    support = np.zeros(values.shape[:3], dtype=bool)
    for index in np.ndindex(values.shape[:3]):
        estimate = centered_haldane_log_odds(values[index])
        coordinates[index] = estimate.observed_log_odds
        variances[index] = estimate.sampling_variance
        support[index] = estimate.supported
    pooled = paule_mandel_pool(
        coordinates, variances, support=support, minimum_donors=2
    )
    if not pooled.supported.all():
        raise CouplingEstimationRefusal("Paule-Mandel log odds lacks source support")
    return {
        "family": "haldane_log_odds_paule_mandel_random_effects",
        "transport_multiplier": 1.0,
        "population_log_odds": pooled.mean,
        "support_count": pooled.support_count,
        "tau_squared": pooled.tau_squared,
    }


def _fit_classical(method: str, tables: np.ndarray) -> dict[str, Any]:
    if method in {RAW_RESIDUAL_METHOD, "poisson_independence_signed_deviance_residual"}:
        return _fit_residual(tables)
    if method == "common_effect_stratified_cmle":
        return _fit_common_effect(tables)
    if method == "pooled_saturated_poisson":
        return _fit_pooled_poisson(tables)
    if method == "paule_mandel_random_effects_log_odds":
        return _fit_paule_mandel(tables)
    raise ValueError(f"unknown classical method: {method}")


def _predict_log_odds(
    log_odds: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    multiplier: float,
) -> np.ndarray:
    field = np.asarray(log_odds, dtype=float)
    output = np.empty((*field.shape, 2, 2))
    for index in np.ndindex(field.shape):
        output[index] = expected_binary_table_from_log_odds(
            float(multiplier) * field[index], rows[index], columns[index]
        )
    return output


def _predict_residual(
    pool: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    multiplier: float,
) -> np.ndarray:
    coordinates = np.asarray(pool, dtype=float)
    output = np.empty((*coordinates.shape, 2, 2))
    for index in np.ndindex(coordinates.shape):
        output[index] = _classical_table(
            float(multiplier) * coordinates[index] * math.sqrt(CELL_BUDGET),
            rows[index],
            columns[index],
        )
    return output


def _predict_model(
    model: dict[str, Any], rows: np.ndarray, columns: np.ndarray
) -> np.ndarray:
    row_values = np.asarray(rows, dtype=np.int64)
    column_values = np.asarray(columns, dtype=np.int64)
    if row_values.shape != column_values.shape:
        raise ValueError("row and column margins differ")
    if model.get("family") == "target_margin_independence":
        total = row_values.sum(axis=-1)
        return (
            row_values[..., :, None]
            * column_values[..., None, :]
            / total[..., None, None]
        )
    multiplier = float(model.get("transport_multiplier", 1.0))
    if "pooled_coordinate" in model:
        return _predict_residual(
            np.asarray(model["pooled_coordinate"]),
            row_values,
            column_values,
            multiplier,
        )
    return _predict_log_odds(
        np.asarray(model["population_log_odds"]), row_values, column_values, multiplier
    )


def _donor_loss(truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    observed = np.asarray(truth, dtype=float)
    estimate = np.asarray(prediction, dtype=float)
    support = np.asarray(mask, dtype=bool)
    if support.shape != observed.shape[:-2] or np.any(
        support & ~_informative(observed)
    ):
        raise PermissionError("scoring mask includes a noninformative table")
    if not np.allclose(observed.sum(axis=-1), estimate.sum(axis=-1)) or not np.allclose(
        observed.sum(axis=-2), estimate.sum(axis=-2)
    ):
        raise PermissionError("prediction changed a target margin")
    observed = observed[support]
    estimate = estimate[support]
    positive = observed > 0
    if np.any(estimate[positive] <= 0) or not np.isfinite(estimate).all():
        raise FloatingPointError("prediction assigns invalid mass")
    terms = np.zeros_like(observed)
    terms[positive] = observed[positive] * np.log(
        observed[positive] / estimate[positive]
    )
    return float((2 * terms.sum(axis=(-2, -1)) / CELL_BUDGET).mean())


def _json_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in model.items()
    }


def _component_equal_mean(losses: dict[str, float]) -> float:
    if set(losses) != set(VALIDATION_DONORS):
        raise ValueError("validation donor loss axis differs")
    return float(
        np.mean(
            [
                np.mean([losses[donor] for donor in component])
                for component in VALIDATION_COMPONENTS
            ]
        )
    )


def _model_losses(
    records: dict[str, dict[str, Any]], donors: Iterable[str], model: dict[str, Any]
) -> dict[str, float]:
    output = {}
    for donor in donors:
        truth = records[donor]["tables"]
        rows, columns = _margins(truth)
        output[donor] = _donor_loss(
            truth,
            _predict_model(model, rows, columns),
            np.ones(truth.shape[:2], dtype=bool),
        )
    return output


def _primary_configs(graph_values: tuple[float, ...]) -> list[PrimaryConfig]:
    return [
        PrimaryConfig(heterogeneity, ridge, graph, transport)
        for heterogeneity, ridge, graph, transport in itertools.product(
            HETEROGENEITY_GRID, RIDGE_GRID, graph_values, TRANSPORT_GRID
        )
    ]


def _record_arrays(
    records: dict[str, dict[str, Any]], donors: Iterable[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = list(donors)
    return (
        np.asarray([records[donor]["tables"] for donor in axis]),
        np.asarray([records[donor]["rna_profile"] for donor in axis]),
        np.asarray([records[donor]["adt_profile"] for donor in axis]),
    )


def _select_primary(
    records: dict[str, dict[str, Any]], graph_values: tuple[float, ...]
) -> dict[str, Any]:
    tables, rna, adt = _record_arrays(records, CALIBRATION_DONORS)
    losses: dict[PrimaryConfig, dict[str, float]] = {}
    structural: dict[tuple[float, float, float], dict[str, Any] | BaseException] = {}
    for config in _primary_configs(graph_values):
        key = (config.heterogeneity_penalty, config.ridge_penalty, config.graph_penalty)
        if key not in structural:
            try:
                structural[key] = _fit_primary(
                    tables, rna, adt, PrimaryConfig(*key, 1.0)
                )
            except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
                structural[key] = error
        base = structural[key]
        if isinstance(base, BaseException):
            continue
        model = dict(base)
        model["transport_multiplier"] = config.transport_multiplier
        losses[config] = _model_losses(records, VALIDATION_DONORS, model)
    if not losses:
        raise ConfirmationRefusal("PRIMARY_SOURCE_CV_HAS_NO_COMPLETE_CONFIGURATION")
    selected = min(
        losses,
        key=lambda config: (_component_equal_mean(losses[config]), config),
    )
    return {
        "selected": selected,
        "losses": losses[selected],
        "component_equal_loss": _component_equal_mean(losses[selected]),
        "complete_candidates": len(losses),
    }


def _select_classical(
    records: dict[str, dict[str, Any]], method: str
) -> tuple[dict[str, Any], dict[str, float], float]:
    calibration_tables = np.asarray(
        [records[donor]["tables"] for donor in CALIBRATION_DONORS]
    )
    base = _fit_classical(method, calibration_tables)
    multipliers = (1.0,) if method == RAW_RESIDUAL_METHOD else TRANSPORT_GRID
    choices = []
    for multiplier in multipliers:
        model = dict(base)
        model["transport_multiplier"] = multiplier
        losses = _model_losses(records, VALIDATION_DONORS, model)
        choices.append((_component_equal_mean(losses), multiplier, model, losses))
    _, multiplier, model, losses = min(choices, key=lambda row: (row[0], row[1]))
    return model, losses, multiplier


def _source_tuning(
    records: dict[str, dict[str, Any]], marker_panel: list[dict[str, str]]
) -> dict[str, Any]:
    primary_selection = _select_primary(records, GRAPH_GRID)
    graph_zero_selection = _select_primary(records, (0.0,))
    selected: PrimaryConfig = primary_selection["selected"]
    selected_zero: PrimaryConfig = graph_zero_selection["selected"]
    calibration_tables, calibration_rna, calibration_adt = _record_arrays(
        records, CALIBRATION_DONORS
    )
    primary_calibration = _fit_primary(
        calibration_tables, calibration_rna, calibration_adt, selected
    )
    validation_primary = _model_losses(records, VALIDATION_DONORS, primary_calibration)
    classical_validation: dict[str, dict[str, float]] = {}
    classical_multipliers: dict[str, float] = {}
    refusals: dict[str, dict[str, str]] = {}
    for method in ALL_CLASSICAL_METHODS:
        try:
            _, losses, multiplier = _select_classical(records, method)
            classical_validation[method] = losses
            classical_multipliers[method] = multiplier
        except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
            refusals[method] = {
                "exception": type(error).__name__,
                "reason": str(error),
            }
    if RAW_RESIDUAL_METHOD not in classical_validation:
        raise ConfirmationRefusal(
            "UNTUNED_RAW_POISSON_SOURCE_COMPARATOR_REFUSED",
            {"classical_refusals": refusals},
        )
    estimable = [method for method in CLASSICAL_ORDER if method in classical_validation]
    if not estimable:
        raise ConfirmationRefusal(
            "NO_SOURCE_CLASSICAL_COMPARATOR_IS_ESTIMABLE",
            {
                "classical_refusals": refusals,
                "untuned_raw_poisson_validation_losses": classical_validation[
                    RAW_RESIDUAL_METHOD
                ],
            },
        )
    fitted_classical = [
        method for method in ALL_CLASSICAL_METHODS if method in classical_validation
    ]
    locked = min(
        estimable,
        key=lambda method: (
            _component_equal_mean(classical_validation[method]),
            ALL_CLASSICAL_METHODS.index(method),
        ),
    )
    component_means: dict[str, dict[str, float | bool]] = {}
    source_eligible = True
    for component in VALIDATION_COMPONENTS:
        primary_mean = float(
            np.mean([validation_primary[donor] for donor in component])
        )
        locked_mean = float(
            np.mean([classical_validation[locked][donor] for donor in component])
        )
        raw_mean = float(
            np.mean(
                [
                    classical_validation[RAW_RESIDUAL_METHOD][donor]
                    for donor in component
                ]
            )
        )
        passed = primary_mean < locked_mean and primary_mean < raw_mean
        component_means["_".join(component)] = {
            "primary": primary_mean,
            "locked_classical": locked_mean,
            "untuned_raw_poisson": raw_mean,
            "passed": passed,
        }
        source_eligible &= passed
    if not source_eligible:
        raise ConfirmationRefusal(
            "SOURCE_PRIMARY_NOT_BETTER_THAN_LOCKED_AND_RAW_IN_EVERY_BATCH",
            {
                "locked_classical_method": locked,
                "selected_primary_configuration": asdict(selected),
                "component_means": component_means,
                "primary_validation_losses": validation_primary,
                "locked_classical_validation_losses": classical_validation[locked],
                "untuned_raw_poisson_validation_losses": classical_validation[
                    RAW_RESIDUAL_METHOD
                ],
                "all_classical_validation_losses": classical_validation,
                "classical_refusals": refusals,
            },
        )
    all_tables, all_rna, all_adt = _record_arrays(records, SOURCE_DONORS)
    final_primary = _fit_primary(all_tables, all_rna, all_adt, selected)
    final_zero = _fit_primary(all_tables, all_rna, all_adt, selected_zero)
    destroyed_tables = np.asarray(
        [records[donor]["destroyed_tables"] for donor in SOURCE_DONORS]
    )
    final_destroyed = _fit_primary(destroyed_tables, all_rna, all_adt, selected)
    models: dict[str, dict[str, Any]] = {
        "primary": _json_model(final_primary),
        "graph_zero_ablation": _json_model(final_zero),
        "destroyed_link": _json_model(final_destroyed),
    }
    for method in fitted_classical:
        model = _fit_classical(method, all_tables)
        model["transport_multiplier"] = classical_multipliers[method]
        models[method] = _json_model(model)
    models[INDEPENDENCE_METHOD] = {"family": "target_margin_independence"}
    return {
        "marker_panel": marker_panel,
        "calibration_donors": list(CALIBRATION_DONORS),
        "validation_donors": list(VALIDATION_DONORS),
        "validation_components": [
            list(component) for component in VALIDATION_COMPONENTS
        ],
        "primary_selection": {
            "selected_configuration": asdict(selected),
            "validation_losses": validation_primary,
            "component_equal_loss": _component_equal_mean(validation_primary),
            "complete_candidates": primary_selection["complete_candidates"],
        },
        "graph_zero_selection": {
            "selected_configuration": asdict(selected_zero),
            "validation_losses": graph_zero_selection["losses"],
        },
        "classical_validation_losses": classical_validation,
        "classical_transport_multipliers": classical_multipliers,
        "classical_refusals": refusals,
        "source_eligibility_passed": True,
        "source_eligibility_rule": "primary mean loss is strictly lower than locked and raw Poisson in both validation batches",
        "locked_classical_method": locked,
        "available_methods": list(models),
        "models": models,
        "final_graph_profiles_use_all_eight_source_donors": True,
    }


def _download_nonassay(manifest: dict[str, Any], filename: str, scratch: Path) -> Path:
    record = _source_file(manifest, filename)
    if record.get("sha256") is None:
        raise PermissionError("non-assay input lacks a frozen SHA-256")
    return _fetch(record, scratch / filename)


def _source_selected_cells(metadata_path: Path) -> dict[str, list[str]]:
    by_donor = {donor: [] for donor in SOURCE_DONORS}
    batches = {
        donor: "Batch1" if donor in {"P1", "P2", "P3", "P4"} else "Batch2"
        for donor in SOURCE_DONORS
    }
    with _open_text(metadata_path) as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"donor", "time", "Batch"} <= set(
            reader.fieldnames
        ):
            raise ConfirmationRefusal("SOURCE_METADATA_SCHEMA_DIFFERS")
        cell_field = reader.fieldnames[0]
        for row in reader:
            donor = row["donor"]
            if donor not in by_donor or str(row["time"]) != "0":
                continue
            if row["Batch"] != batches[donor]:
                raise ConfirmationRefusal("SOURCE_METADATA_BATCH_DIFFERS")
            cell = row[cell_field]
            if not cell:
                raise ConfirmationRefusal("SOURCE_METADATA_CELL_IDENTIFIER_IS_EMPTY")
            by_donor[donor].append(cell)
    if any(len(set(values)) != len(values) for values in by_donor.values()):
        raise ConfirmationRefusal("SOURCE_METADATA_CELL_IDENTIFIER_IS_DUPLICATED")
    return {
        donor: _deterministic_selection(by_donor[donor], donor)
        for donor in SOURCE_DONORS
    }


def _source_counts(
    scratch_dir: Path,
) -> tuple[
    dict[str, list[str]],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, dict[str, Any]],
]:
    candidate = _candidate()
    manifest = _manifest()
    markers = _markers(candidate)
    scratch = _private_path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=False, mode=0o700)
    metadata = _download_nonassay(manifest, "GSE164378_sc.meta.data_3P.csv.gz", scratch)
    rna_barcodes_path = _download_nonassay(
        manifest, "GSM5008737_RNA_3P-barcodes.tsv.gz", scratch
    )
    adt_barcodes_path = _download_nonassay(
        manifest, "GSM5008738_ADT_3P-barcodes.tsv.gz", scratch
    )
    rna_features_path = _download_nonassay(
        manifest, "GSM5008737_RNA_3P-features.tsv.gz", scratch
    )
    adt_features_path = _download_nonassay(
        manifest, "GSM5008738_ADT_3P-features.tsv.gz", scratch
    )
    rna_record = _source_file(manifest, "GSM5008737_RNA_3P-matrix.mtx.gz")
    adt_record = _source_file(manifest, "GSM5008738_ADT_3P-matrix.mtx.gz")
    rna_matrix = _fetch(
        rna_record,
        scratch / rna_record["filename"],
        stage="source",
        modality="rna",
        unit_id="GSE164378_3P_day0",
    )
    adt_matrix = _fetch(
        adt_record,
        scratch / adt_record["filename"],
        stage="source",
        modality="adt",
        unit_id="GSE164378_3P_day0",
    )
    rna_barcodes = _read_barcodes(rna_barcodes_path)
    adt_barcodes = _read_barcodes(adt_barcodes_path)
    if rna_barcodes != adt_barcodes:
        raise ConfirmationRefusal("SOURCE_RNA_ADT_BARCODE_AXES_DIFFER")
    selected = _source_selected_cells(metadata)
    flattened = [cell for donor in SOURCE_DONORS for cell in selected[donor]]
    selected_columns = _column_lookup(rna_barcodes, flattened)
    destinations = {
        target: (donor, index)
        for donor in SOURCE_DONORS
        for index, target in enumerate(
            range(
                SOURCE_DONORS.index(donor) * CELL_BUDGET,
                (SOURCE_DONORS.index(donor) + 1) * CELL_BUDGET,
            )
        )
    }
    rna_features = _read_tsv(rna_features_path)
    adt_features = _read_tsv(adt_features_path)
    rna_rows = _resolve_feature_rows(rna_features, markers, modality="rna", held=False)
    adt_rows = _resolve_feature_rows(adt_features, markers, modality="adt", held=False)
    rna_all, totals_all = _read_matrix_market_subset(
        rna_matrix,
        expected_rows=len(rna_features),
        expected_columns=len(rna_barcodes),
        selected_columns=selected_columns,
        retained_rows=rna_rows,
        authorized_rows=set(range(len(rna_features))),
        collect_totals=True,
    )
    adt_all, _ = _read_matrix_market_subset(
        adt_matrix,
        expected_rows=len(adt_features),
        expected_columns=len(adt_barcodes),
        selected_columns=selected_columns,
        retained_rows=adt_rows,
        authorized_rows=set(adt_rows),
        collect_totals=False,
    )
    if totals_all is None:
        raise AssertionError("RNA totals were not collected")
    rna: dict[str, np.ndarray] = {}
    adt: dict[str, np.ndarray] = {}
    totals: dict[str, np.ndarray] = {}
    for target, (donor, index) in destinations.items():
        rna.setdefault(donor, np.empty((CELL_BUDGET, len(markers)), dtype=np.int64))[
            index
        ] = rna_all[target]
        adt.setdefault(donor, np.empty((CELL_BUDGET, len(markers)), dtype=np.int64))[
            index
        ] = adt_all[target]
        totals.setdefault(donor, np.empty(CELL_BUDGET, dtype=np.int64))[index] = (
            totals_all[target]
        )
    observed = {
        "rna": {
            "filename": rna_matrix.name,
            "bytes": rna_matrix.stat().st_size,
            "sha256": _sha256(rna_matrix),
        },
        "adt": {
            "filename": adt_matrix.name,
            "bytes": adt_matrix.stat().st_size,
            "sha256": _sha256(adt_matrix),
        },
    }
    shutil.rmtree(scratch)
    return selected, rna, adt, totals, observed


def _source_records(
    selected: dict[str, list[str]],
    rna: dict[str, np.ndarray],
    adt: dict[str, np.ndarray],
    totals: dict[str, np.ndarray],
) -> tuple[list[int], dict[str, dict[str, Any]]]:
    rna_quality = {donor: _rna_axis_quality(rna[donor]) for donor in SOURCE_DONORS}
    adt_quality = {donor: _adt_axis_quality(adt[donor]) for donor in SOURCE_DONORS}
    locked = [
        marker
        for marker in range(rna[SOURCE_DONORS[0]].shape[1])
        if all(
            rna_quality[donor]["valid"][marker] and adt_quality[donor]["valid"][marker]
            for donor in SOURCE_DONORS
        )
    ]
    if len(locked) < MINIMUM_SOURCE_LOCKED_MARKERS:
        raise ConfirmationRefusal(
            "FEWER_THAN_16_SOURCE_LOCKED_COGNATES",
            {
                "source_locked_marker_indices": locked,
                "source_locked_marker_count": len(locked),
                "required_source_locked_marker_count": MINIMUM_SOURCE_LOCKED_MARKERS,
                "axis_validity": {
                    donor: {
                        "rna": rna_quality[donor]["valid"].tolist(),
                        "adt": adt_quality[donor]["valid"].tolist(),
                        "rna_detection_prevalence": rna_quality[donor][
                            "prevalence"
                        ].tolist(),
                        "adt_distinct_values": adt_quality[donor][
                            "distinct_values"
                        ].tolist(),
                        "adt_largest_equal_value_fraction": adt_quality[donor][
                            "largest_equal_value_fraction"
                        ].tolist(),
                    }
                    for donor in SOURCE_DONORS
                },
            },
        )
    records: dict[str, dict[str, Any]] = {}
    for donor in SOURCE_DONORS:
        rna_counts = rna[donor][:, locked]
        adt_counts = adt[donor][:, locked]
        rna_states = (rna_counts > 0).astype(np.uint8)
        adt_states = _adt_states(adt_counts, selected[donor], donor)
        normalized = np.log1p(10_000 * rna_counts / totals[donor][:, None])
        records[donor] = {
            "tables": _binary_tables(rna_states, adt_states),
            "destroyed_tables": _binary_tables(
                rna_states, _destroyed_adt(adt_states, selected[donor], donor)
            ),
            "rna_profile": normalized.mean(axis=0),
            "adt_profile": np.log1p(adt_counts).mean(axis=0),
            "rna_prevalence": rna_quality[donor]["prevalence"],
            "adt_distinct_values": adt_quality[donor]["distinct_values"],
            "adt_largest_equal_value_fraction": adt_quality[donor][
                "largest_equal_value_fraction"
            ],
        }
    return locked, records


def _source_stage_body(scratch_dir: Path) -> dict[str, Any]:
    candidate = _candidate()
    full_markers = _markers(candidate)
    selected, rna, adt, totals, observed = _source_counts(scratch_dir)
    locked_indices, records = _source_records(selected, rna, adt, totals)
    marker_panel = [full_markers[index] for index in locked_indices]
    tuning = _source_tuning(records, marker_panel)
    return {
        "schema": "gse164378-3p-gse155673-source/1.0",
        "status": "SOURCE_ELIGIBILITY_PASS_AND_MODELS_FROZEN",
        "created_at_utc": _timestamp(),
        "candidate_sha256": _sha256(DEFAULT_CANDIDATE),
        "protocol_sha256": _sha256(DEFAULT_PROTOCOL),
        "source_donors": list(SOURCE_DONORS),
        "cell_budget": CELL_BUDGET,
        "source_locked_marker_indices": locked_indices,
        "source_locked_marker_count": len(marker_panel),
        "marker_panel": marker_panel,
        "selected_cell_axis_sha256": {
            donor: _canonical_json_sha256(selected[donor]) for donor in SOURCE_DONORS
        },
        "source_table_sha256": {
            donor: _array_sha256(records[donor]["tables"]) for donor in SOURCE_DONORS
        },
        "source_axis_quality": {
            donor: {
                "rna_detection_prevalence": records[donor]["rna_prevalence"].tolist(),
                "adt_distinct_values": records[donor]["adt_distinct_values"].tolist(),
                "adt_largest_equal_value_fraction": records[donor][
                    "adt_largest_equal_value_fraction"
                ].tolist(),
            }
            for donor in SOURCE_DONORS
        },
        "observed_source_assay_files": observed,
        "model": tuning,
        "held_matrix_files_opened": 0,
        "held_numeric_values_read": 0,
    }


def _held_file_records(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = manifest.get("held_donor_files")
    if not isinstance(rows, list) or len(rows) != 12:
        raise PermissionError("held donor manifest axis differs")
    output = {row.get("donor_file_id"): row for row in rows}
    if len(output) != 12:
        raise PermissionError("held donor manifest identifiers differ")
    return output


def _write_private(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    destination = _private_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _write_json_x(destination, payload, mode=0o600)
    return {"sha256": _sha256(destination), "bytes": destination.stat().st_size}


def _private_artifact(path: Path, certificate: dict[str, Any]) -> dict[str, Any]:
    destination = _private_path(path)
    if destination.stat().st_size != certificate.get("bytes") or _sha256(
        destination
    ) != certificate.get("sha256"):
        raise PermissionError("private artifact differs from its public certificate")
    return _read_json(destination)


def _held_shared_features(
    manifest: dict[str, Any], scratch: Path
) -> tuple[Path, list[list[str]]]:
    record = _source_file(manifest, "GSE155673_features.tsv.gz")
    path = _fetch(record, scratch / record["filename"])
    return path, _read_tsv(path)


def _rna_stage_body(
    scratch_dir: Path,
    selection_bridge_path: Path,
    rna_states_path: Path,
) -> dict[str, Any]:
    source = _require_completed("source")
    candidate = _candidate()
    manifest = _manifest()
    marker_panel = source["marker_panel"]
    scratch = _private_path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=False, mode=0o700)
    feature_path, features = _held_shared_features(manifest, scratch)
    retained = _resolve_feature_rows(features, marker_panel, modality="rna", held=True)
    authorized = set(retained)
    manifest_rows = _held_file_records(manifest)
    selected: dict[str, list[str]] = {}
    states: dict[str, np.ndarray] = {}
    quality: dict[str, dict[str, np.ndarray]] = {}
    observed: dict[str, dict[str, Any]] = {}
    for donor_row in candidate["held_donors"]:
        donor = _held_id(donor_row)
        files = manifest_rows[donor]
        barcode_record = _manifest_record(manifest, files["barcode"])
        matrix_record = _manifest_record(manifest, files["matrix"])
        if (
            barcode_record["filename"] != donor_row["barcode_file"]
            or matrix_record["filename"] != donor_row["matrix_file"]
        ):
            raise PermissionError("held candidate and manifest filenames differ")
        barcode_path = _fetch(barcode_record, scratch / barcode_record["filename"])
        barcodes = _read_barcodes(barcode_path)
        selected[donor] = _deterministic_selection(barcodes, donor)
        columns = _column_lookup(barcodes, selected[donor])
        matrix_path = _fetch(
            matrix_record,
            scratch / matrix_record["filename"],
            stage="rna",
            modality="rna",
            unit_id=donor,
        )
        counts, _ = _read_matrix_market_subset(
            matrix_path,
            expected_rows=len(features),
            expected_columns=len(barcodes),
            selected_columns=columns,
            retained_rows=retained,
            authorized_rows=authorized,
            collect_totals=False,
        )
        states[donor] = (counts > 0).astype(np.uint8)
        quality[donor] = _rna_axis_quality(counts)
        observed[donor] = {
            "filename": matrix_path.name,
            "bytes": matrix_path.stat().st_size,
            "sha256": _sha256(matrix_path),
        }
        barcode_path.unlink()
        matrix_path.unlink()
    feature_path.unlink()
    scratch.rmdir()
    bridge = _write_private(
        selection_bridge_path,
        {
            "schema": "gse164378-3p-gse155673-private-selection/1.0",
            "selected": selected,
        },
    )
    private = _write_private(
        rna_states_path,
        {
            "schema": "gse164378-3p-gse155673-private-rna-states/1.0",
            "states": {donor: value.tolist() for donor, value in states.items()},
            "selected_axis_sha256": {
                donor: _canonical_json_sha256(selected[donor]) for donor in selected
            },
        },
    )
    return {
        "schema": "gse164378-3p-gse155673-rna/1.0",
        "status": "HELD_RNA_REDUCED_WITHOUT_ADT_VALUE_CONVERSION",
        "created_at_utc": _timestamp(),
        "source_sha256": _sha256(DEFAULT_SOURCE),
        "held_donors": list(selected),
        "marker_panel": marker_panel,
        "selected_axis_sha256": {
            donor: _canonical_json_sha256(selected[donor]) for donor in selected
        },
        "row_margins": {
            donor: [
                [
                    int(CELL_BUDGET - states[donor][:, marker].sum()),
                    int(states[donor][:, marker].sum()),
                ]
                for marker in range(len(marker_panel))
            ]
            for donor in selected
        },
        "rna_axis_quality": {
            donor: {
                "detection_prevalence": quality[donor]["prevalence"].tolist(),
                "axis_valid": quality[donor]["valid"].tolist(),
            }
            for donor in selected
        },
        "observed_combined_matrix_files": observed,
        "selection_bridge": bridge,
        "rna_states": private,
        "combined_file_firewall": "only Gene Expression row values at selected columns were converted",
        "held_adt_values_converted": 0,
        "held_joint_tables_formed": 0,
    }


def _adt_stage_body(
    scratch_dir: Path,
    selection_bridge_path: Path,
    adt_states_path: Path,
) -> dict[str, Any]:
    source = _require_completed("source")
    rna = _require_completed("rna")
    candidate = _candidate()
    manifest = _manifest()
    bridge = _private_artifact(selection_bridge_path, rna["selection_bridge"])
    selected = bridge["selected"]
    marker_panel = source["marker_panel"]
    scratch = _private_path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=False, mode=0o700)
    feature_path, features = _held_shared_features(manifest, scratch)
    retained = _resolve_feature_rows(features, marker_panel, modality="adt", held=True)
    authorized = set(retained)
    manifest_rows = _held_file_records(manifest)
    states: dict[str, np.ndarray] = {}
    quality: dict[str, dict[str, np.ndarray]] = {}
    observed: dict[str, dict[str, Any]] = {}
    for donor_row in candidate["held_donors"]:
        donor = _held_id(donor_row)
        files = manifest_rows[donor]
        barcode_record = _manifest_record(manifest, files["barcode"])
        matrix_record = _manifest_record(manifest, files["matrix"])
        barcode_path = _fetch(barcode_record, scratch / barcode_record["filename"])
        barcodes = _read_barcodes(barcode_path)
        if selected[donor] != _deterministic_selection(barcodes, donor):
            raise PermissionError("held selection bridge differs from barcode axis")
        columns = _column_lookup(barcodes, selected[donor])
        matrix_path = _fetch(
            matrix_record,
            scratch / matrix_record["filename"],
            stage="adt",
            modality="adt",
            unit_id=donor,
        )
        current = {
            "filename": matrix_path.name,
            "bytes": matrix_path.stat().st_size,
            "sha256": _sha256(matrix_path),
        }
        if current != rna["observed_combined_matrix_files"][donor]:
            raise PermissionError(
                "combined matrix bytes differ between RNA and ADT stages"
            )
        counts, _ = _read_matrix_market_subset(
            matrix_path,
            expected_rows=len(features),
            expected_columns=len(barcodes),
            selected_columns=columns,
            retained_rows=retained,
            authorized_rows=authorized,
            collect_totals=False,
        )
        states[donor] = _adt_states(counts, selected[donor], donor)
        quality[donor] = _adt_axis_quality(counts)
        observed[donor] = current
        barcode_path.unlink()
        matrix_path.unlink()
    feature_path.unlink()
    scratch.rmdir()
    private = _write_private(
        adt_states_path,
        {
            "schema": "gse164378-3p-gse155673-private-adt-states/1.0",
            "states": {donor: value.tolist() for donor, value in states.items()},
            "selected_axis_sha256": rna["selected_axis_sha256"],
        },
    )
    return {
        "schema": "gse164378-3p-gse155673-adt/1.0",
        "status": "HELD_ADT_REDUCED_SEPARATELY_WITHOUT_RNA_VALUE_CONVERSION",
        "created_at_utc": _timestamp(),
        "rna_sha256": _sha256(DEFAULT_RNA),
        "held_donors": list(selected),
        "marker_panel": marker_panel,
        "selected_axis_sha256": rna["selected_axis_sha256"],
        "column_margins": {
            donor: [[CELL_BUDGET // 2, CELL_BUDGET // 2] for _ in marker_panel]
            for donor in selected
        },
        "adt_axis_quality": {
            donor: {
                "distinct_values": quality[donor]["distinct_values"].tolist(),
                "largest_equal_value_fraction": quality[donor][
                    "largest_equal_value_fraction"
                ].tolist(),
                "axis_valid": quality[donor]["valid"].tolist(),
            }
            for donor in selected
        },
        "observed_combined_matrix_files": observed,
        "adt_states": private,
        "combined_file_firewall": "only frozen Antibody Capture row values at selected columns were converted",
        "held_rna_state_artifact_opened": False,
        "held_joint_tables_formed": 0,
    }


def _held_margin_arrays(
    row_margins: list[list[int]], column_margins: list[list[int]]
) -> tuple[np.ndarray, np.ndarray]:
    row_axis = np.asarray(row_margins, dtype=np.int64)
    column_axis = np.asarray(column_margins, dtype=np.int64)
    if (
        row_axis.ndim != 2
        or row_axis.shape[1] != 2
        or column_axis.shape != row_axis.shape
        or np.any(row_axis < 0)
        or np.any(column_axis < 0)
        or np.any(row_axis.sum(axis=1) != CELL_BUDGET)
        or np.any(column_axis.sum(axis=1) != CELL_BUDGET)
    ):
        raise PermissionError("held margin axes differ from the frozen contract")
    size = len(row_axis)
    rows = np.repeat(row_axis[:, None, :], size, axis=1)
    columns = np.repeat(column_axis[None, :, :], size, axis=0)
    return rows, columns


def _prediction_stage_body() -> dict[str, Any]:
    source = _require_completed("source")
    rna = _require_completed("rna")
    adt = _require_completed("adt")
    candidate = _candidate()
    donors = [_held_id(row) for row in candidate["held_donors"]]
    full_markers = _markers(candidate)
    locked_indices = source.get("source_locked_marker_indices")
    if (
        not isinstance(locked_indices, list)
        or len(locked_indices) < MINIMUM_SOURCE_LOCKED_MARKERS
        or len(set(locked_indices)) != len(locked_indices)
        or any(
            not isinstance(index, int) or not 0 <= index < 24
            for index in locked_indices
        )
    ):
        raise PermissionError("source-locked marker indices differ")
    marker_panel = [full_markers[index] for index in locked_indices]
    if (
        source.get("marker_panel") != marker_panel
        or rna.get("marker_panel") != marker_panel
        or adt.get("marker_panel") != marker_panel
        or rna.get("held_donors") != donors
        or adt.get("held_donors") != donors
        or rna.get("selected_axis_sha256") != adt.get("selected_axis_sha256")
    ):
        raise PermissionError("held public axes differ")

    models = source.get("model", {}).get("models", {})
    methods = source.get("model", {}).get("available_methods")
    locked = source.get("model", {}).get("locked_classical_method")
    selected_graph_penalty = (
        source.get("model", {})
        .get("primary_selection", {})
        .get("selected_configuration", {})
        .get("graph_penalty")
    )
    required = {
        "primary",
        "graph_zero_ablation",
        "destroyed_link",
        RAW_RESIDUAL_METHOD,
        INDEPENDENCE_METHOD,
    }
    if (
        not isinstance(models, dict)
        or not isinstance(methods, list)
        or len(methods) != len(set(methods))
        or set(methods) != set(models)
        or not required <= set(methods)
        or locked not in CLASSICAL_ORDER
        or locked not in methods
        or selected_graph_penalty not in GRAPH_GRID
    ):
        raise PermissionError("source method contract differs")

    size = len(marker_panel)
    rna_valid = np.empty((len(donors), size), dtype=bool)
    adt_valid = np.empty_like(rna_valid)
    for donor_index, donor in enumerate(donors):
        rna_valid[donor_index] = np.asarray(
            rna["rna_axis_quality"][donor]["axis_valid"], dtype=bool
        )
        adt_valid[donor_index] = np.asarray(
            adt["adt_axis_quality"][donor]["axis_valid"], dtype=bool
        )
    rna_support = rna_valid.sum(axis=0)
    adt_support = adt_valid.sum(axis=0)
    if np.any(rna_support < MINIMUM_HELD_SUPPORTED_DONORS):
        raise ConfirmationRefusal(
            "HELD_RNA_MARKER_FAILS_TEN_OF_TWELVE_SUPPORT",
            {
                "marker_ids": [marker["marker_id"] for marker in marker_panel],
                "supported_donors_per_marker": rna_support.tolist(),
                "required_supported_donors": MINIMUM_HELD_SUPPORTED_DONORS,
            },
        )
    if np.any(adt_support < MINIMUM_HELD_SUPPORTED_DONORS):
        raise ConfirmationRefusal(
            "HELD_ADT_MARKER_FAILS_TEN_OF_TWELVE_SUPPORT",
            {
                "marker_ids": [marker["marker_id"] for marker in marker_panel],
                "supported_donors_per_marker": adt_support.tolist(),
                "required_supported_donors": MINIMUM_HELD_SUPPORTED_DONORS,
            },
        )

    minimum_pairs = math.ceil(MINIMUM_HELD_PAIR_FRACTION * size**2)
    samples = []
    for donor_index, donor in enumerate(donors):
        rows, columns = _held_margin_arrays(
            rna["row_margins"][donor], adt["column_margins"][donor]
        )
        mask = np.outer(rna_valid[donor_index], adt_valid[donor_index])
        supported_pairs = int(mask.sum())
        if supported_pairs < minimum_pairs:
            raise ConfirmationRefusal(
                "HELD_DONOR_FAILS_EIGHTY_PERCENT_PAIR_SUPPORT",
                {
                    "donor_id": donor,
                    "valid_rna_axes": int(rna_valid[donor_index].sum()),
                    "valid_adt_axes": int(adt_valid[donor_index].sum()),
                    "valid_ordered_pairs": supported_pairs,
                    "required_valid_ordered_pairs": minimum_pairs,
                },
            )
        estimates = {
            method: _predict_model(models[method], rows, columns) for method in methods
        }
        samples.append(
            {
                "donor_id": donor,
                "phenotype": _held_groups(candidate)[0][donor],
                "severity": _held_groups(candidate)[1].get(donor),
                "selected_axis_sha256": rna["selected_axis_sha256"][donor],
                "row_margins": rows.tolist(),
                "column_margins": columns.tolist(),
                "valid_ordered_pair_mask": mask.tolist(),
                "valid_ordered_pair_mask_sha256": _array_sha256(mask),
                "valid_rna_axes": int(rna_valid[donor_index].sum()),
                "valid_adt_axes": int(adt_valid[donor_index].sum()),
                "valid_ordered_pairs": supported_pairs,
                "required_valid_ordered_pairs": minimum_pairs,
                "predicted_tables": {
                    method: estimate.tolist() for method, estimate in estimates.items()
                },
                "prediction_sha256": {
                    method: _array_sha256(estimate)
                    for method, estimate in estimates.items()
                },
            }
        )
    return {
        "schema": "gse164378-3p-gse155673-predictions/1.0",
        "status": "HELD_PREDICTIONS_FROZEN_BEFORE_RNA_ADT_PAIRING",
        "created_at_utc": _timestamp(),
        "source_sha256": _sha256(DEFAULT_SOURCE),
        "rna_sha256": _sha256(DEFAULT_RNA),
        "adt_sha256": _sha256(DEFAULT_ADT),
        "candidate_sha256": _sha256(DEFAULT_CANDIDATE),
        "protocol_sha256": _sha256(DEFAULT_PROTOCOL),
        "marker_panel": marker_panel,
        "methods": methods,
        "locked_classical_method": locked,
        "global_rna_supported_donors_per_marker": rna_valid.sum(axis=0).tolist(),
        "global_adt_supported_donors_per_marker": adt_valid.sum(axis=0).tolist(),
        "minimum_supported_donors_per_marker_per_modality": MINIMUM_HELD_SUPPORTED_DONORS,
        "samples": samples,
        "held_private_state_artifacts_opened": 0,
        "held_joint_tables_formed": 0,
    }


def make_score_authorization() -> dict[str, Any]:
    prediction, prediction_commit = _require_completed_stage_artifact(
        PREDICTION_TAG,
        DEFAULT_PREDICTION,
        "prediction",
        require_success=True,
    )
    if DEFAULT_SCORE_AUTHORIZATION.exists():
        raise FileExistsError("score authorization already exists")
    payload = {
        **_score_authorization_bindings(prediction, prediction_commit),
        "schema": "gse164378-3p-gse155673-score-authorization/1.0",
        "status": "AUTHORIZED_AFTER_PUBLIC_PREDICTION_FREEZE",
        "created_at_utc": _timestamp(),
    }
    _validate_public_payload(payload)
    _write_json_x(DEFAULT_SCORE_AUTHORIZATION, payload)
    return payload


def _require_score_authorization() -> dict[str, Any]:
    value = _read_json(DEFAULT_SCORE_AUTHORIZATION)
    prediction, prediction_commit = _require_completed_stage_artifact(
        PREDICTION_TAG,
        DEFAULT_PREDICTION,
        "prediction",
        require_success=True,
    )
    expected = {
        **_score_authorization_bindings(prediction, prediction_commit),
        "schema": "gse164378-3p-gse155673-score-authorization/1.0",
        "status": "AUTHORIZED_AFTER_PUBLIC_PREDICTION_FREEZE",
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise PermissionError(f"score authorization differs at {key}")
    _require_public_tag(
        SCORE_AUTHORIZATION_TAG, (_relative(DEFAULT_SCORE_AUTHORIZATION),)
    )
    _validate_public_payload(value)
    return value


def _score_authorization_bindings(
    prediction: dict[str, Any], prediction_commit: str
) -> dict[str, Any]:
    prediction_path = _relative(DEFAULT_PREDICTION)
    return {
        "outcome_access_authorized": True,
        "prediction_tag": PREDICTION_TAG,
        "prediction_path": prediction_path,
        "prediction_sha256": _sha256(DEFAULT_PREDICTION),
        "prediction_bytes": DEFAULT_PREDICTION.stat().st_size,
        "prediction_public_commit": prediction_commit,
        "prediction_public_url": (
            f"https://github.com/sushaan-k/coupling-fields-benchmark/blob/"
            f"{prediction_commit}/{prediction_path}"
        ),
        "prediction_payload_sha256": _canonical_json_sha256(prediction),
        "protocol_sha256": _sha256(DEFAULT_PROTOCOL),
        "candidate_designation_sha256": _sha256(DEFAULT_CANDIDATE),
        "source_manifest_sha256": _sha256(DEFAULT_MANIFEST),
        "metadata_preflight_sha256": _sha256(DEFAULT_METADATA_PREFLIGHT),
        "runtime_environment_sha256": _sha256(DEFAULT_RUNTIME),
        "score_authorization_template_sha256": _sha256(
            DEFAULT_SCORE_AUTHORIZATION_TEMPLATE
        ),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "tests_sha256": _sha256(DEFAULT_TESTS),
        "source_result_sha256": _sha256(DEFAULT_SOURCE),
        "rna_result_sha256": _sha256(DEFAULT_RNA),
        "adt_result_sha256": _sha256(DEFAULT_ADT),
        "recipient_joint_tables_formed_before_authorization": 0,
        "held_rna_and_adt_states_opened_together_before_authorization": False,
        "publication_required_before_score": True,
        "required_public_tag": SCORE_AUTHORIZATION_TAG,
        "protocol_commit": _require_public_tag(PROTOCOL_TAG, _protocol_paths()),
        "runtime_environment": _require_runtime_environment(),
    }


def _disease_stratified_bootstrap(
    primary: np.ndarray,
    comparator: np.ndarray,
    donors: list[str],
    phenotype: dict[str, str],
) -> dict[str, Any]:
    first = np.asarray(primary, dtype=float)
    second = np.asarray(comparator, dtype=float)
    if first.shape != (len(donors),) or second.shape != first.shape:
        raise ValueError("paired held loss axes differ")
    labels = np.asarray([phenotype[donor] for donor in donors], dtype=object)
    groups = [np.flatnonzero(labels == label) for label in ("healthy", "covid")]
    if [len(group) for group in groups] != [5, 7]:
        raise PermissionError("held disease strata differ")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = np.concatenate(
        [
            group[generator.integers(0, len(group), size=(BOOTSTRAPS, len(group)))]
            for group in groups
        ],
        axis=1,
    )
    sampled_primary = first[sampled].mean(axis=1)
    sampled_comparator = second[sampled].mean(axis=1)
    differences = sampled_primary - sampled_comparator
    positive = sampled_comparator > 0
    reductions = (
        1.0 - sampled_primary / sampled_comparator if positive.all() else None
    )
    quantiles = (ALPHA, 1.0 - ALPHA)
    return {
        "draws": BOOTSTRAPS,
        "seed": BOOTSTRAP_SEED,
        "resampling_unit": "donors within the frozen healthy and COVID-19 strata at observed 5:7 weights",
        "mean_difference_97_5_percent_interval": np.quantile(
            differences, quantiles, method="linear"
        ).tolist(),
        "mean_difference_98_75th_percentile": float(
            np.quantile(differences, 1.0 - ALPHA, method="linear")
        ),
        "relative_loss_reduction_97_5_percent_interval": (
            np.quantile(reductions, quantiles, method="linear").tolist()
            if reductions is not None
            else None
        ),
        "relative_loss_reduction_undefined_draws": int(
            np.count_nonzero(~positive)
        ),
    }


def _exact_donor_sign_p(differences: np.ndarray) -> dict[str, Any]:
    values = np.asarray(differences, dtype=float)
    favorable = int(np.count_nonzero(values < 0))
    donors = len(values)
    p_value = sum(
        math.comb(donors, count) for count in range(favorable, donors + 1)
    ) / (2**donors)
    return {
        "donors": donors,
        "favorable_donors": favorable,
        "zero_donors_counted_as_unfavorable": int(np.count_nonzero(values == 0)),
        "one_sided_p": p_value,
    }


def _exact_donor_sign_flip(differences: np.ndarray) -> dict[str, Any]:
    values = np.asarray(differences, dtype=float)
    if values.shape != (12,):
        raise ValueError("exact sign-flip requires the 12 frozen donors")
    assignments = np.arange(1 << len(values), dtype=np.uint16)[:, None]
    bits = (assignments >> np.arange(len(values), dtype=np.uint16)) & 1
    statistics = ((1.0 - 2.0 * bits) * np.abs(values)).mean(axis=1)
    observed = float(values.mean())
    p_value = float(np.count_nonzero(statistics <= observed + 1e-15) / len(statistics))
    return {
        "method": "exact",
        "donors": len(values),
        "assignments": len(statistics),
        "observed_mean_difference": observed,
        "one_sided_p": p_value,
    }


def _comparison(
    donors: list[str],
    phenotype: dict[str, str],
    severity: dict[str, str],
    primary: np.ndarray,
    comparator: np.ndarray,
) -> dict[str, Any]:
    first = np.asarray(primary, dtype=float)
    second = np.asarray(comparator, dtype=float)
    if (
        len(donors) != 12
        or first.shape != (12,)
        or second.shape != (12,)
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
    ):
        raise ValueError("comparison requires 12 finite paired donor losses")
    difference = first - second
    comparator_mean = float(second.mean())
    reduction = (
        1.0 - float(first.mean()) / comparator_mean if comparator_mean > 0 else None
    )
    bootstrap = _disease_stratified_bootstrap(first, second, donors, phenotype)
    sign = _exact_donor_sign_p(difference)
    sign_flip = _exact_donor_sign_flip(difference)
    phenotype_means = {
        label: float(
            np.mean(
                [
                    difference[index]
                    for index, donor in enumerate(donors)
                    if phenotype[donor] == label
                ]
            )
        )
        for label in ("healthy", "covid")
    }
    severity_means = {
        label: float(
            np.mean(
                [
                    difference[index]
                    for index, donor in enumerate(donors)
                    if severity.get(donor) == label
                ]
            )
        )
        for label in ("moderate", "severe")
    }
    leave_one_out = {
        donor: float(np.delete(difference, index).mean())
        for index, donor in enumerate(donors)
    }
    favorable = int(np.count_nonzero(difference < 0))
    checks = {
        "relative_loss_reduction_at_least_five_percent": reduction is not None
        and reduction >= 0.05,
        "disease_stratified_bootstrap_upper_98_75_below_zero": bootstrap[
            "mean_difference_98_75th_percentile"
        ]
        < 0,
        "at_least_eleven_of_twelve_donors_favorable": favorable >= 11,
        "exact_one_sided_sign_p_at_most_0_0125": sign["one_sided_p"] <= ALPHA,
        "exact_4096_sign_flip_p_at_most_0_0125": sign_flip["one_sided_p"] <= ALPHA,
        "healthy_and_covid_means_negative": all(
            value < 0 for value in phenotype_means.values()
        ),
        "moderate_and_severe_means_negative": all(
            value < 0 for value in severity_means.values()
        ),
        "every_leave_one_donor_out_mean_negative": all(
            value < 0 for value in leave_one_out.values()
        ),
    }
    return {
        "donors": len(donors),
        "primary_mean_loss": float(first.mean()),
        "comparator_mean_loss": comparator_mean,
        "relative_loss_reduction": reduction,
        "mean_paired_difference": float(difference.mean()),
        "disease_stratified_bootstrap": bootstrap,
        "favorable_donors": favorable,
        "required_favorable_donors": 11,
        "exact_donor_sign_test": sign,
        "exact_paired_donor_sign_flip": sign_flip,
        "phenotype_mean_differences": phenotype_means,
        "severity_mean_differences": severity_means,
        "leave_one_donor_out_mean_differences": leave_one_out,
        "checks": checks,
        "passes": all(checks.values()),
        "donor_differences": dict(zip(donors, map(float, difference))),
    }


def _score_stage_body(rna_states_path: Path, adt_states_path: Path) -> dict[str, Any]:
    prediction = _require_completed("prediction")
    source = _require_completed("source")
    rna_public = _require_completed("rna")
    adt_public = _require_completed("adt")
    _require_score_authorization()
    rna_private = _private_artifact(rna_states_path, rna_public["rna_states"])
    adt_private = _private_artifact(adt_states_path, adt_public["adt_states"])
    if (
        rna_private.get("schema") != "gse164378-3p-gse155673-private-rna-states/1.0"
        or adt_private.get("schema") != "gse164378-3p-gse155673-private-adt-states/1.0"
        or rna_private.get("selected_axis_sha256")
        != rna_public.get("selected_axis_sha256")
        or adt_private.get("selected_axis_sha256")
        != adt_public.get("selected_axis_sha256")
    ):
        raise PermissionError("private held state contract differs")
    candidate = _candidate()
    donors = [_held_id(row) for row in candidate["held_donors"]]
    phenotype, severity = _held_groups(candidate)
    frozen_rows = prediction.get("samples")
    if not isinstance(frozen_rows, list):
        raise PermissionError("frozen prediction samples differ")
    frozen = {row.get("donor_id"): row for row in frozen_rows}
    methods = prediction.get("methods")
    locked = source.get("model", {}).get("locked_classical_method")
    required = {"primary", locked, RAW_RESIDUAL_METHOD, INDEPENDENCE_METHOD}
    if (
        set(frozen) != set(donors)
        or len(frozen_rows) != len(donors)
        or not isinstance(methods, list)
        or not required <= set(methods)
        or prediction.get("locked_classical_method") != locked
    ):
        raise PermissionError("held prediction method or donor contract differs")
    marker_count = len(prediction["marker_panel"])
    minimum_pairs = math.ceil(MINIMUM_HELD_PAIR_FRACTION * marker_count**2)
    truths: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    losses: dict[str, np.ndarray] = {}
    samples: list[dict[str, Any]] = []

    def score_method(method: str) -> None:
        if method in losses:
            return
        if method not in methods:
            raise PermissionError("required frozen prediction method is absent")
        values = np.empty(len(donors), dtype=float)
        for donor_index, donor in enumerate(donors):
            record = frozen[donor]
            estimate = np.asarray(record["predicted_tables"][method], dtype=float)
            if _array_sha256(estimate) != record["prediction_sha256"][method]:
                raise PermissionError("frozen prediction hash differs")
            values[donor_index] = _donor_loss(truths[donor], estimate, masks[donor])
            samples[donor_index]["losses"][method] = float(values[donor_index])
        losses[method] = values

    for donor_index, donor in enumerate(donors):
        rna_state = np.asarray(rna_private["states"][donor], dtype=np.uint8)
        adt_state = np.asarray(adt_private["states"][donor], dtype=np.uint8)
        if (
            rna_state.shape != (CELL_BUDGET, marker_count)
            or adt_state.shape != rna_state.shape
            or np.any((rna_state != 0) & (rna_state != 1))
            or np.any((adt_state != 0) & (adt_state != 1))
        ):
            raise PermissionError("private held state axes differ")
        truth = _binary_tables(rna_state, adt_state)
        rows, columns = _margins(truth)
        record = frozen[donor]
        mask = np.asarray(record["valid_ordered_pair_mask"], dtype=bool)
        rna_valid = np.asarray(
            rna_public["rna_axis_quality"][donor]["axis_valid"], dtype=bool
        )
        adt_valid = np.asarray(
            adt_public["adt_axis_quality"][donor]["axis_valid"], dtype=bool
        )
        expected_mask = np.outer(rna_valid, adt_valid)
        if (
            rows.tolist() != record["row_margins"]
            or columns.tolist() != record["column_margins"]
            or not np.array_equal(mask, expected_mask)
            or _array_sha256(mask) != record["valid_ordered_pair_mask_sha256"]
            or int(mask.sum()) != record["valid_ordered_pairs"]
            or record["required_valid_ordered_pairs"] != minimum_pairs
            or int(mask.sum()) < minimum_pairs
            or np.any(mask & ~_informative(truth))
        ):
            raise PermissionError("held truth differs from frozen margins or support")
        truths[donor] = truth
        masks[donor] = mask
        samples.append(
            {
                "donor_id": donor,
                "phenotype": phenotype[donor],
                "severity": severity.get(donor),
                "truth_table_sha256": _array_sha256(truth),
                "valid_ordered_pairs": int(mask.sum()),
                "losses": {},
            }
        )

    for method in ("primary", locked, RAW_RESIDUAL_METHOD, INDEPENDENCE_METHOD):
        score_method(method)
    for method in methods:
        if method in ALL_CLASSICAL_METHODS:
            score_method(method)
    primary = losses["primary"]
    locked_comparison = _comparison(
        donors, phenotype, severity, primary, losses[locked]
    )
    locked_comparison["inferential_role"] = "primary_confirmatory_endpoint"
    raw_comparison = _comparison(
        donors, phenotype, severity, primary, losses[RAW_RESIDUAL_METHOD]
    )
    raw_comparison["serial_gate_status"] = (
        "CONFIRMATORY_EVALUATED_AFTER_LOCKED_GATE_PASS"
        if locked_comparison["passes"]
        else "DESCRIPTIVE_ONLY_LOCKED_GATE_FAILED"
    )
    raw_comparison["inferential_role"] = "serial_confirmatory_gate_two"
    independence_comparison = _comparison(
        donors, phenotype, severity, primary, losses[INDEPENDENCE_METHOD]
    )
    broad_classical_support = bool(
        locked_comparison["passes"] and raw_comparison["passes"]
    )

    graph_zero: dict[str, Any] = {
        "status": "NOT_EVALUATED_CLASSICAL_GATE_FAILED",
        "serial_gate_position": 3,
    }
    destroyed: dict[str, Any] = {
        "status": "NOT_EVALUATED_CLASSICAL_OR_GRAPH_ZERO_GATE_FAILED",
        "serial_gate_position": 4,
    }
    classical_comparisons: dict[str, Any] = {}
    for method in methods:
        if method not in ALL_CLASSICAL_METHODS:
            continue
        if method == locked:
            comparison = locked_comparison
        elif method == RAW_RESIDUAL_METHOD:
            comparison = raw_comparison
        else:
            comparison = _comparison(
                donors, phenotype, severity, primary, losses[method]
            )
            comparison["inferential_role"] = "descriptive_nonlocked_classical"
        classical_comparisons[method] = comparison
    if broad_classical_support:
        score_method("graph_zero_ablation")
        graph_zero = {
            **_comparison(
                donors,
                phenotype,
                severity,
                primary,
                losses["graph_zero_ablation"],
            ),
            "status": "EVALUATED_AFTER_BOTH_CLASSICAL_GATES_PASS",
            "serial_gate_position": 3,
        }
        if graph_zero["passes"]:
            score_method("destroyed_link")
            destroyed = {
                **_comparison(
                    donors,
                    phenotype,
                    severity,
                    primary,
                    losses["destroyed_link"],
                ),
                "status": "EVALUATED_AFTER_GRAPH_ZERO_GATE_PASS",
                "serial_gate_position": 4,
            }

    selected_graph_penalty = float(
        source["model"]["primary_selection"]["selected_configuration"]["graph_penalty"]
    )
    graph_structure_supported = bool(
        broad_classical_support
        and selected_graph_penalty > 0
        and graph_zero.get("passes", False)
    )
    coupling_link_supported = bool(
        graph_structure_supported and destroyed.get("passes", False)
    )
    return {
        "schema": "gse164378-3p-gse155673-held-validation/1.0",
        "status": (
            "COMPLETED_PRIMARY_ESTIMATOR_CONFIRMATION_PASS"
            if locked_comparison["passes"]
            else "COMPLETED_PRIMARY_ESTIMATOR_CONFIRMATION_FAIL"
        ),
        "created_at_utc": _timestamp(),
        "prediction_sha256": _sha256(DEFAULT_PREDICTION),
        "score_authorization_sha256": _sha256(DEFAULT_SCORE_AUTHORIZATION),
        "locked_classical_method": locked,
        "selected_graph_penalty": selected_graph_penalty,
        "primary_vs_locked_classical": locked_comparison,
        "primary_vs_untuned_raw_poisson": raw_comparison,
        "target_margin_independence_head_to_head": independence_comparison,
        "target_margin_independence_inferential_role": "descriptive outside the serial confirmatory chain",
        "classical_head_to_head": classical_comparisons,
        "graph_zero_serial_secondary": graph_zero,
        "destroyed_link_serial_secondary": destroyed,
        "confirmatory_gate_order": [
            "primary_vs_locked_classical",
            "primary_vs_untuned_raw_poisson",
            "primary_vs_graph_zero_ablation",
            "primary_vs_destroyed_link",
        ],
        "primary_estimator_confirmed": bool(locked_comparison["passes"]),
        "broad_classical_support": broad_classical_support,
        "graph_structure_supported": graph_structure_supported,
        "coupling_link_supported": coupling_link_supported,
        "structured_coupling_field_supported": coupling_link_supported,
        "samples": samples,
        "outcome_failure_rule": "A completed score that misses a criterion is a negative result, not a QC refusal.",
    }


def verify_public_result() -> dict[str, Any]:
    result, commit = _require_completed_stage_artifact(
        RESULT_TAG, DEFAULT_SCORE, "score", require_success=False
    )
    status = str(result.get("status", ""))
    if status.startswith("TERMINAL_SCORE_REFUSAL"):
        if result.get("schema") != "gse164378-3p-gse155673-score-terminal/1.0":
            raise PermissionError("terminal result schema differs")
    elif result.get(
        "schema"
    ) != "gse164378-3p-gse155673-held-validation/1.0" or status not in {
        "COMPLETED_PRIMARY_ESTIMATOR_CONFIRMATION_PASS",
        "COMPLETED_PRIMARY_ESTIMATOR_CONFIRMATION_FAIL",
    }:
        raise PermissionError("public held-validation result schema or status differs")
    return {
        "schema": "gse164378-3p-gse155673-public-result-verification/1.0",
        "status": "PUBLIC_RESULT_LINEAGE_VERIFIED",
        "result_commit": commit,
        "result_sha256": _sha256(DEFAULT_SCORE),
        "held_validation_status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    claim = commands.add_parser("claim")
    claim.add_argument("stage", choices=tuple(STAGE_PATHS))
    source = commands.add_parser("run-source")
    source.add_argument("--scratch-dir", type=Path, required=True)
    rna = commands.add_parser("run-rna")
    rna.add_argument("--scratch-dir", type=Path, required=True)
    rna.add_argument("--selection-bridge", type=Path, required=True)
    rna.add_argument("--rna-states", type=Path, required=True)
    adt = commands.add_parser("run-adt")
    adt.add_argument("--scratch-dir", type=Path, required=True)
    adt.add_argument("--selection-bridge", type=Path, required=True)
    adt.add_argument("--adt-states", type=Path, required=True)
    commands.add_parser("run-prediction")
    commands.add_parser("make-score-authorization")
    score = commands.add_parser("run-score")
    score.add_argument("--rna-states", type=Path, required=True)
    score.add_argument("--adt-states", type=Path, required=True)
    verify = commands.add_parser("verify-stage")
    verify.add_argument("stage", choices=tuple(STAGE_PATHS))
    commands.add_parser("verify-public-result", aliases=["verify-result"])
    args = parser.parse_args()
    if args.command == "claim":
        payload = claim_stage(args.stage)
    elif args.command == "run-source":
        payload = _run_claimed_stage(
            "source", lambda: _source_stage_body(args.scratch_dir)
        )
    elif args.command == "run-rna":
        payload = _run_claimed_stage(
            "rna",
            lambda: _rna_stage_body(
                args.scratch_dir, args.selection_bridge, args.rna_states
            ),
        )
    elif args.command == "run-adt":
        payload = _run_claimed_stage(
            "adt",
            lambda: _adt_stage_body(
                args.scratch_dir, args.selection_bridge, args.adt_states
            ),
        )
    elif args.command == "run-prediction":
        payload = _run_claimed_stage("prediction", _prediction_stage_body)
    elif args.command == "make-score-authorization":
        payload = make_score_authorization()
    elif args.command == "run-score":
        payload = _run_claimed_stage(
            "score", lambda: _score_stage_body(args.rna_states, args.adt_states)
        )
    elif args.command == "verify-stage":
        payload = verify_stage(args.stage)
    else:
        payload = verify_public_result()
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
