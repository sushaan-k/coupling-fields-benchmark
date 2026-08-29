"""One-shot GSE288020 to GSE309593 independent-study confirmation.

The four stages are claimed and published before execution. RNA and ADT are
reduced by disjoint code paths; their private state arrays first meet in the
authorized score stage. No command in this module is allowed to tune a model,
marker, threshold, or comparator on GSE309593 values.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterable
from urllib.parse import quote
from urllib.request import urlopen

import h5py
import numpy as np
import scipy

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mapreg.heterogeneity_adaptive_coupling import (  # noqa: E402
    expected_binary_table_from_log_odds,
)
from mapreg.table_prediction import multinomial_deviance_per_observation  # noqa: E402


ROOT = REPO_ROOT
DATA_DIR = ROOT / "data/confirmation/gse309593_independent_study"
DEFAULT_CANDIDATE = DATA_DIR / "candidate_designation_v1.json"
DEFAULT_AMENDMENT = DATA_DIR / "source_split_amendment_v1.json"
DEFAULT_PROTOCOL = DATA_DIR / "protocol_v1.json"
DEFAULT_RUNTIME_SPEC = DATA_DIR / "runtime_environment_v1.json"
DEFAULT_SOURCE_AUTH_TEMPLATE = DATA_DIR / "source_authorization_template_v1.json"
DEFAULT_SOURCE_AUTHORIZATION = DATA_DIR / "source_authorization_v1.json"
DEFAULT_SCORE_AUTH_TEMPLATE = DATA_DIR / "score_authorization_template_v1.json"
DEFAULT_SCORE_AUTHORIZATION = DATA_DIR / "score_authorization_v1.json"

DEFAULT_RNA_ATTEMPT = DATA_DIR / "rna_attempt_v1.jsonl"
DEFAULT_ADT_ATTEMPT = DATA_DIR / "adt_attempt_v1.jsonl"
DEFAULT_PREDICTION_ATTEMPT = DATA_DIR / "prediction_attempt_v1.jsonl"
DEFAULT_SCORE_ATTEMPT = DATA_DIR / "score_attempt_v1.jsonl"

DEFAULT_RNA = ROOT / "results/gse309593_independent_study_rna_v1.json"
DEFAULT_ADT = ROOT / "results/gse309593_independent_study_adt_v1.json"
DEFAULT_PREDICTION = ROOT / "results/gse309593_independent_study_predictions_v1.json"
DEFAULT_SCORE = ROOT / "results/gse309593_independent_study_confirmation_v1.json"

DEFAULT_PROTOCOL_DOC = (
    ROOT / "docs/GSE309593_INDEPENDENT_STUDY_CONFIRMATION_PROTOCOL_2026-08-29.md"
)
DEFAULT_CHECKLIST = ROOT / "docs/GSE309593_PUBLIC_AUTHORIZATION_CHECKLIST_2026-08-29.md"
DEFAULT_TEST = ROOT / "tests/test_gse309593_independent_study.py"

PUBLIC_ORIGIN = "https://github.com/sushaan-k/coupling-fields-benchmark.git"
CANDIDATE_TAG = "gse309593-independent-study-v1-candidate"
CANDIDATE_SHA256 = "07c1979bcee8009db14265a2360f2f46527f674ac5835cf54f098f2f10bdc3e9"
GSE288020_PROTOCOL_TAG = "gse288020-citeseq-v1-protocol"
GSE288020_DEVELOPMENT_TAG = "gse288020-citeseq-v1-development"
PROTOCOL_TAG = "gse309593-independent-study-v1-protocol"
SOURCE_AUTHORIZATION_TAG = "gse309593-independent-study-v1-source-authorized"
RNA_ATTEMPT_TAG = "gse309593-independent-study-v1-rna-attempt"
RNA_TAG = "gse309593-independent-study-v1-rna"
ADT_ATTEMPT_TAG = "gse309593-independent-study-v1-adt-attempt"
ADT_TAG = "gse309593-independent-study-v1-adt"
PREDICTION_ATTEMPT_TAG = "gse309593-independent-study-v1-prediction-attempt"
PREDICTION_TAG = "gse309593-independent-study-v1-predictions"
SCORE_AUTHORIZATION_TAG = "gse309593-independent-study-v1-score-authorized"
SCORE_ATTEMPT_TAG = "gse309593-independent-study-v1-score-attempt"
RESULT_TAG = "gse309593-independent-study-v1-result"

CELL_BUDGET = 512
MINIMUM_DETECTED_GENES = 200
MAXIMUM_MITOCHONDRIAL_FRACTION = 0.10
MAXIMUM_RNA_UMIS = 70_000
MAXIMUM_ADT_BOUNDARY_TIE_FRACTION = 0.25
MINIMUM_COGNATES = 9
MINIMUM_INFORMATIVE_PAIRS = 64
MINIMUM_ELIGIBLE_SUBJECTS = 18
BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 20260829
CELL_SELECTION_SALT = "GSE309593-CELL-BUDGET-v1"
ADT_TIE_SALT = "GSE309593-ADT-MIDRANK-v1"

EXPECTED_SOURCE_SUBJECTS = (
    "R001",
    "R005",
    "R008",
    "R009",
    "R010",
    "R013",
    "R014",
    "R003",
    "R006",
    "R015",
    "R016",
    "R020",
    "R023",
    "R024",
)
EXPECTED_SOURCE_CALIBRATION = EXPECTED_SOURCE_SUBJECTS[:7]
EXPECTED_SOURCE_PILOT = EXPECTED_SOURCE_SUBJECTS[7:]
CORE_METHODS = (
    "primary",
    "matched_deviance_residual",
    "destroyed_link",
)
CLASSICAL_METHODS = (
    "common_effect_cmle",
    "pooled_saturated_poisson",
)
REQUIRED_METHODS = CORE_METHODS + CLASSICAL_METHODS
SOURCE_TRANSPORT_GRID = (0.5, 0.75, 1.0, 1.25)
CLAIM_SCOPE = (
    "composition-inclusive cross-condition transfer among source-QC-matched "
    "selected cell mixtures; not cell-type-conditional invariance or "
    "same-population causal transport"
)
COMPLETED_SCORE_STATUSES = {
    "CONFIRMATION_PASS_WITH_GAIN_OVER_BOTH_CLASSICAL_INTERACTIONS",
    "CONFIRMATION_PASS_WITHOUT_GAIN_OVER_BOTH_CLASSICAL_INTERACTIONS",
    "COMPLETED_CONFIRMATION_FAIL",
}

GSE288020_PROTOCOL_BINDINGS = (
    ".gitattributes",
    ".gitignore",
    "experiments/confirm_gse288020_citeseq.py",
    "experiments/reduce_gse288020_held_rna.py",
    "tests/test_gse288020_citeseq_confirmation.py",
    "docs/GSE288020_MGUS_TO_MYELOMA_CITESEQ_CONFIRMATION_PROTOCOL_2026-08-28.md",
    "docs/FINAL_PUBLIC_EVIDENCE_LEDGER.md",
    "data/confirmation/gse288020_citeseq/source_manifest_v1.json",
    "data/confirmation/gse288020_citeseq/candidate_designation_v1.json",
    "data/confirmation/gse288020_citeseq/metadata/filelist.txt",
    "results/development/gse288020_schema_access_v1.json",
    "results/development/gse288020_schema_preflight_v1.json",
    "results/development/gse288020_runtime_environment_v1.json",
    "experiments/confirm_gse158769_citeseq.py",
    "experiments/confirm_gse314416_citeseq.py",
    "mapreg/__init__.py",
    "mapreg/classical_residuals.py",
    "mapreg/coupling_fields.py",
    "mapreg/factorial_coupling.py",
    "mapreg/heterogeneity_adaptive_coupling.py",
    "mapreg/hierarchical_conditional_coupling.py",
    "mapreg/table_prediction.py",
    "requirements.txt",
    "pyproject.toml",
)

PROTOCOL_BINDINGS = (
    ".gitattributes",
    "experiments/confirm_gse309593_independent_study.py",
    "tests/test_gse309593_independent_study.py",
    "docs/GSE309593_INDEPENDENT_STUDY_CONFIRMATION_PROTOCOL_2026-08-29.md",
    "docs/GSE309593_PUBLIC_AUTHORIZATION_CHECKLIST_2026-08-29.md",
    "docs/GSE309593_INDEPENDENT_STUDY_CANDIDATE_DESIGNATION_2026-08-29.md",
    "data/confirmation/gse309593_independent_study/candidate_designation_v1.json",
    "data/confirmation/gse309593_independent_study/source_split_amendment_v1.json",
    "data/confirmation/gse309593_independent_study/protocol_v1.json",
    "data/confirmation/gse309593_independent_study/runtime_environment_v1.json",
    "data/confirmation/gse309593_independent_study/source_authorization_template_v1.json",
    "data/confirmation/gse309593_independent_study/score_authorization_template_v1.json",
    "mapreg/__init__.py",
    "mapreg/coupling_fields.py",
    "mapreg/heterogeneity_adaptive_coupling.py",
    "mapreg/hierarchical_conditional_coupling.py",
    "mapreg/table_prediction.py",
    "requirements.txt",
    "pyproject.toml",
)


class ProtocolRefusal(RuntimeError):
    """A prespecified terminal QC, support, or authorization refusal."""

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


def _runtime_environment() -> dict[str, Any]:
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
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
    specification = _read_json(DEFAULT_RUNTIME_SPEC)
    if specification.get("schema") != "gse309593-runtime-environment/1.0":
        raise PermissionError("runtime specification schema differs")
    observed = _runtime_environment()
    if specification.get("required_runtime") != observed:
        raise PermissionError(
            "runtime environment differs from the frozen specification"
        )
    return observed


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


def _axis_sha256(subject: str, values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    digest.update(CELL_SELECTION_SALT.encode())
    digest.update(b"\0")
    digest.update(subject.encode())
    for value in values:
        encoded = value.encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _salted_hash(salt: str, *values: str) -> str:
    digest = hashlib.sha256(salt.encode())
    for value in values:
        digest.update(b"\0")
        digest.update(value.encode())
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return payload


def _write_json_x(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_jsonl(path: Path, payload: dict[str, Any], *, create: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if create else "a") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _private_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        return resolved
    raise PermissionError("private artifacts must be outside the public repository")


def _validate_public_payload(value: Any, key: str | None = None) -> None:
    if key in {
        "selected_ids",
        "cell_ids",
        "barcodes",
        "states",
        "selection_bridge_path",
        "rna_states_path",
        "adt_states_path",
    }:
        raise PermissionError(f"public payload contains forbidden private field {key}")
    if isinstance(value, dict):
        for child_key, child in value.items():
            _validate_public_payload(child, str(child_key))
    elif isinstance(value, list):
        for child in value:
            _validate_public_payload(child, key)
    elif isinstance(value, str) and str(ROOT.resolve()) in value:
        raise PermissionError("public payload contains a local repository path")


def _binding_hashes(paths: Iterable[str] = PROTOCOL_BINDINGS) -> dict[str, str]:
    return {relative: _sha256(ROOT / relative) for relative in paths}


def _remote_tag_commit(tag: str) -> str:
    result = subprocess.run(
        ["git", "ls-remote", PUBLIC_ORIGIN, f"refs/tags/{tag}^{{}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not result:
        raise PermissionError(f"public annotated tag {tag} is absent")
    return result.split()[0]


def _require_remote_tag_commit(tag: str, expected: str) -> None:
    if _remote_tag_commit(tag) != expected:
        raise PermissionError(f"public tag {tag} has a different commit")


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
    commit = subprocess.run(
        ["git", "rev-list", "-n", "1", tag],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not commit or _remote_tag_commit(tag) != commit:
        raise PermissionError(f"public tag {tag} does not match the local tag")
    for relative in paths:
        tagged = subprocess.run(
            ["git", "show", f"{tag}:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if tagged != (ROOT / relative).read_bytes():
            raise PermissionError(f"{relative} differs from public tag {tag}")
    return commit


def _candidate() -> dict[str, Any]:
    if _sha256(DEFAULT_CANDIDATE) != CANDIDATE_SHA256:
        raise PermissionError("historical candidate designation bytes differ")
    payload = _read_json(DEFAULT_CANDIDATE)
    subjects = payload.get("recipient_cohort", {}).get("subjects", [])
    if len(subjects) != 23:
        raise PermissionError("candidate cohort does not contain 23 subjects")
    subject_ids = [row.get("subject_id") for row in subjects]
    gsms = [row.get("gsm") for row in subjects]
    if len(set(subject_ids)) != 23 or len(set(gsms)) != 23:
        raise PermissionError("candidate subject or GSM identifiers are not unique")
    if payload.get("study_design", {}).get("target_study_tuning") != "forbidden":
        raise PermissionError("candidate does not forbid target-study tuning")
    return payload


def _expected_source_mappings() -> list[dict[str, str]]:
    values = _read_json(DEFAULT_PROTOCOL)["expected_source_marker_mappings"]
    return [dict(value) for value in values]


def _source_marker_mappings(model: dict[str, Any]) -> list[dict[str, str]]:
    mappings = _expected_source_mappings()
    marker_order = model.get("source_support", {}).get("marker_order")
    if marker_order != [row["adt_target"] for row in mappings]:
        raise PermissionError("source marker axis differs from the frozen contract")
    return mappings


def _transport_multiplier(record: dict[str, Any]) -> float:
    configuration = record.get("configuration", {})
    value = configuration.get(
        "transport_multiplier", record.get("transport_multiplier")
    )
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise PermissionError("source transport multiplier is invalid")
    return float(value)


def _validate_source_model(model: dict[str, Any]) -> None:
    if (
        set(model)
        != {
            "schema",
            "scope",
            "public_export_location",
            "source_accession",
            "source_condition",
            "selection_provenance",
            "refit_axis",
            "external_study_eligibility",
            "declared_method_order",
            "methods",
            "method_refusals",
            "source_support",
            "numerical_certificate",
        }
        or model.get("schema") != "gse288020-independent-study-source-model/1.0"
        or model.get("scope")
        != (
            "fixed GSE288020 MGUS source model for a separately frozen external "
            "study; not a GSE288020 held-MM result"
        )
        or model.get("public_export_location")
        != (
            "results/development/gse288020_development_v1.json#/"
            "source_only_external_study_model"
        )
        or model.get("source_accession") != "GSE288020"
        or model.get("source_condition") != "MGUS"
    ):
        raise PermissionError("source model identity differs")
    selection = model.get("selection_provenance")
    if (
        not isinstance(selection, dict)
        or set(selection)
        != {
            "split_salt",
            "designated_calibration_donors",
            "designated_pilot_donors",
            "retained_calibration_donors",
            "retained_pilot_donors",
            "primary_configuration",
            "deviance_residual_configuration",
            "gse288020_mm_values_used",
            "gse309593_values_used",
        }
        or selection.get("split_salt") != "GSE288020-MGUS-SPLIT-v1"
        or selection.get("designated_calibration_donors")
        != list(EXPECTED_SOURCE_CALIBRATION)
        or selection.get("designated_pilot_donors") != list(EXPECTED_SOURCE_PILOT)
        or selection.get("retained_calibration_donors")
        != list(EXPECTED_SOURCE_CALIBRATION)
        or selection.get("retained_pilot_donors") != list(EXPECTED_SOURCE_PILOT)
    ):
        raise PermissionError("source configuration-selection split differs")
    if (
        selection.get("gse288020_mm_values_used") is not False
        or selection.get("gse309593_values_used") is not False
    ):
        raise PermissionError("source artifact used forbidden outcome values")
    primary_configuration = selection.get("primary_configuration")
    residual_configuration = selection.get("deviance_residual_configuration")
    if (
        not isinstance(primary_configuration, dict)
        or set(primary_configuration)
        != {
            "graph_neighbors",
            "heterogeneity_penalty",
            "ridge_penalty",
            "graph_penalty",
            "transport_multiplier",
        }
        or primary_configuration["graph_neighbors"] not in (1, 2)
        or primary_configuration["heterogeneity_penalty"] not in (0.1, 1.0, 10.0)
        or primary_configuration["ridge_penalty"] not in (0.01, 0.1)
        or primary_configuration["graph_penalty"] not in (0.0, 0.1, 0.3, 1.0)
        or primary_configuration["transport_multiplier"] not in SOURCE_TRANSPORT_GRID
        or (
            primary_configuration["graph_penalty"] == 0.0
            and primary_configuration["graph_neighbors"] != 1
        )
        or residual_configuration
        not in (
            {"family": "deviance", "transport_multiplier": value}
            for value in SOURCE_TRANSPORT_GRID
        )
    ):
        raise PermissionError("source selected configuration differs")
    refit = model.get("refit_axis")
    if (
        not isinstance(refit, dict)
        or set(refit)
        != {
            "designated_mgus_donors",
            "retained_mgus_donors",
            "retained_mgus_donor_count",
        }
        or refit.get("designated_mgus_donors") != list(EXPECTED_SOURCE_SUBJECTS)
        or refit.get("retained_mgus_donors") != list(EXPECTED_SOURCE_SUBJECTS)
        or refit.get("retained_mgus_donor_count") != 14
    ):
        raise PermissionError("source final-refit subject axis differs")
    eligibility = model.get("external_study_eligibility")
    if eligibility != {
        "internal_gse288020_refit_allows_12_to_14_retained_mgus_donors": True,
        "external_study_requires_all_14_designated_mgus_donors": True,
        "external_study_requires_7_calibration_and_7_pilot_donors": True,
        "external_study_requires_all_five_methods_valid": True,
        "target_assay_access_requires_external_study_ready": True,
    }:
        raise PermissionError("source external-study eligibility contract differs")
    if model.get("declared_method_order") != list(REQUIRED_METHODS):
        raise PermissionError("source declared method order differs")

    support = model.get("source_support")
    if not isinstance(support, dict) or set(support) != {
        "marker_order",
        "ordered_pair_axis",
        "informative_source_donors_per_marker",
        "informative_source_donors_per_ordered_pair",
        "informative_source_donors_per_ordered_pair_sha256",
        "retained_ordered_pair_support_mask",
        "retained_ordered_pair_count",
    }:
        raise PermissionError("source support certificate is absent")
    mappings = _source_marker_mappings(model)
    size = len(mappings)
    pair_support = np.asarray(
        support.get("informative_source_donors_per_ordered_pair"), dtype=np.int64
    )
    marker_support = np.asarray(
        support.get("informative_source_donors_per_marker"), dtype=np.int64
    )
    support_mask = np.asarray(
        support.get("retained_ordered_pair_support_mask"), dtype=bool
    )
    if (
        support.get("ordered_pair_axis") != "RNA-major, then ADT marker order"
        or marker_support.shape != (size,)
        or np.any(marker_support < 2)
        or np.any(marker_support > len(EXPECTED_SOURCE_SUBJECTS))
        or pair_support.shape != (size * size,)
        or not np.array_equal(pair_support, np.repeat(marker_support, size))
        or np.any(pair_support < 2)
        or support_mask.shape != (size * size,)
        or not support_mask.all()
        or support.get("retained_ordered_pair_count") != size * size
        or support.get("informative_source_donors_per_ordered_pair_sha256")
        != _array_sha256(pair_support)
    ):
        raise PermissionError("source pair-support certificate differs")

    numerical = model.get("numerical_certificate")
    if not isinstance(numerical, dict) or set(numerical) != {
        "finite_coordinate_checks",
        "core_checks",
        "classical_checks",
        "core_passes",
        "classical_head_to_head_ready",
        "external_study_ready",
        "passes",
    }:
        raise PermissionError("source numerical certificate fields differ")
    checks = numerical.get("core_checks", {}) if isinstance(numerical, dict) else {}
    required_checks = (
        "exactly_14_retained_mgus_donors",
        "exactly_7_retained_calibration_and_7_retained_pilot",
        "all_256_pairs_have_at_least_2_informative_source_donors",
        "primary_residual_destroyed_coordinates_are_finite",
        "primary_optimizer_certificate_pass",
        "destroyed_optimizer_certificate_pass",
        "no_mm_or_gse309593_values_used",
    )
    finite = (
        numerical.get("finite_coordinate_checks", {})
        if isinstance(numerical, dict)
        else {}
    )
    if (
        set(checks) != set(required_checks)
        or any(checks.get(key) is not True for key in required_checks)
        or any(finite.get(method) is not True for method in CORE_METHODS)
    ):
        raise PermissionError("source core numerical certificate differs")
    if any(
        numerical.get(key) is not True
        for key in (
            "core_passes",
            "classical_head_to_head_ready",
            "external_study_ready",
            "passes",
        )
    ):
        raise PermissionError("source external-study numerical certificate failed")

    methods = model.get("methods")
    refusals = model.get("method_refusals")
    if not isinstance(methods, dict) or set(methods) != set(REQUIRED_METHODS):
        raise PermissionError("source method set differs from the frozen contract")
    if refusals != {}:
        raise PermissionError("a source method refusal blocks target-study access")
    for name in REQUIRED_METHODS:
        record = methods[name]
        if record.get("status") != "VALID":
            raise PermissionError(f"source {name} is not valid")
        transport_multiplier = _transport_multiplier(record)
        if transport_multiplier not in SOURCE_TRANSPORT_GRID:
            raise PermissionError(f"source {name} transport multiplier differs")
        key = (
            "pooled_coordinate"
            if name == "matched_deviance_residual"
            else "population_log_odds"
        )
        values = np.asarray(record.get(key), dtype=float)
        if values.shape != (size, size) or not np.isfinite(values).all():
            raise PermissionError(f"source {name} coordinate field differs")
        if finite.get(name) is not True:
            raise PermissionError(f"source {name} finite-coordinate check differs")
    if set(finite) != set(REQUIRED_METHODS):
        raise PermissionError("source finite-coordinate method axis differs")
    if (
        methods["primary"].get("configuration") != primary_configuration
        or methods["destroyed_link"].get("configuration") != primary_configuration
    ):
        raise PermissionError("source primary configuration provenance differs")
    residual = methods["matched_deviance_residual"]
    if residual.get("configuration") != residual_configuration:
        raise PermissionError("matched residual semantics differ")
    classical_checks = numerical.get("classical_checks", {})
    if set(classical_checks) != {
        "both_classical_coordinates_are_finite",
        "common_cmle_gradient_and_condition_certificate_pass",
        "pooled_poisson_no_structural_zero_pass",
    }:
        raise PermissionError("source classical certificate fields differ")
    if (
        "common_effect_cmle" in methods
        and classical_checks.get("common_cmle_gradient_and_condition_certificate_pass")
        is not True
    ):
        raise PermissionError("source common-effect certificate failed")
    if (
        "pooled_saturated_poisson" in methods
        and classical_checks.get("pooled_poisson_no_structural_zero_pass") is not True
    ):
        raise PermissionError("source pooled-Poisson certificate failed")
    if classical_checks.get("both_classical_coordinates_are_finite") is not True:
        raise PermissionError("source classical coordinate certificate failed")


def _validate_gse288020_development_lineage(
    authorization: dict[str, Any],
    contract: dict[str, Any],
    source_path: Path,
    development: dict[str, Any],
) -> None:
    attempt_path = ROOT / contract["canonical_attempt_path"]
    runtime_path = ROOT / contract["canonical_runtime_path"]
    expected_files = (
        (attempt_path, "source_development_attempt"),
        (runtime_path, "source_runtime"),
    )
    for path, prefix in expected_files:
        if path.stat().st_size != authorization.get(f"{prefix}_bytes") or _sha256(
            path
        ) != authorization.get(f"{prefix}_sha256"):
            raise PermissionError(f"{prefix} bytes differ from authorization")

    protocol_commit = authorization["gse288020_protocol_commit"]
    development_commit = authorization["gse288020_development_commit"]
    if (
        _require_public_tag(GSE288020_PROTOCOL_TAG, GSE288020_PROTOCOL_BINDINGS)
        != protocol_commit
    ):
        raise PermissionError("GSE288020 protocol lineage differs")
    if (
        _require_public_tag(
            GSE288020_DEVELOPMENT_TAG,
            (
                *GSE288020_PROTOCOL_BINDINGS,
                _relative(attempt_path),
                _relative(source_path),
            ),
        )
        != development_commit
    ):
        raise PermissionError("GSE288020 development lineage differs")

    runtime_spec = _read_json(runtime_path)
    if runtime_spec.get(
        "schema"
    ) != "gse288020-runtime-environment/1.0" or not isinstance(
        runtime_spec.get("required_runtime"), dict
    ):
        raise PermissionError("GSE288020 runtime specification differs")
    runtime = runtime_spec["required_runtime"]
    try:
        records = [
            json.loads(line) for line in attempt_path.read_text().splitlines() if line
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise PermissionError("GSE288020 development ledger is unreadable") from error
    if len(records) != 2:
        raise PermissionError("GSE288020 development ledger length differs")
    started, finished = records
    if set(started) != {
        "authorization_commit",
        "created_at_utc",
        "event",
        "output",
        "runtime_environment",
        "stage",
    } or set(finished) != {
        "authorization_commit",
        "created_at_utc",
        "event",
        "output",
        "output_sha256",
        "runtime_environment",
        "stage",
        "status",
    }:
        raise PermissionError("GSE288020 development ledger fields differ")
    if started["event"] != "STARTED" or finished["event"] != "FINISHED":
        raise PermissionError("GSE288020 development ledger events differ")
    expected_output = _relative(source_path)
    for record in records:
        if (
            record["authorization_commit"] != protocol_commit
            or record["stage"] != "development"
            or record["output"] != expected_output
            or record["runtime_environment"] != runtime
        ):
            raise PermissionError("GSE288020 development ledger lineage differs")
    allowed_status = {"PILOT_PROMOTION_PASS", "TERMINAL_PILOT_PROMOTION_FAILURE"}
    if (
        development.get("schema") != "gse288020-development/1.0"
        or development.get("status") not in allowed_status
        or development.get("protocol_commit") != protocol_commit
        or development.get("authorization_commit") != protocol_commit
        or development.get("stage") != "development"
        or development.get("runtime_environment") != runtime
        or finished["status"] != development.get("status")
        or finished["output_sha256"] != _sha256(source_path)
    ):
        raise PermissionError("GSE288020 development result lineage differs")


def _validate_source_model_against_development(
    development: dict[str, Any], model: dict[str, Any]
) -> None:
    calibration = list(EXPECTED_SOURCE_CALIBRATION)
    pilot = list(EXPECTED_SOURCE_PILOT)
    retained = list(EXPECTED_SOURCE_SUBJECTS)
    support = development.get("support", {})
    if (
        development.get("calibration_donors") != calibration
        or development.get("pilot_donors") != pilot
        or development.get("retained_development_donors") != retained
        or development.get("retained_development_donor_count") != len(retained)
        or support.get("calibration", {}).get("eligible_donors") != calibration
        or support.get("pilot", {}).get("eligible_donors") != pilot
        or model["refit_axis"]["retained_mgus_donors"] != retained
    ):
        raise PermissionError("source outer and nested donor axes differ")

    selection = development.get("selection")
    nested_selection = model["selection_provenance"]
    if (
        not isinstance(selection, dict)
        or selection.get("primary") != nested_selection["primary_configuration"]
    ):
        raise PermissionError("source outer and nested primary selection differ")
    deviance_candidates = [
        candidate
        for candidate in selection.get("residual_candidates", [])
        if candidate.get("configuration", {}).get("family") == "deviance"
        and isinstance(candidate.get("mean_pilot_loss"), (int, float))
        and math.isfinite(candidate["mean_pilot_loss"])
    ]
    if not deviance_candidates:
        raise PermissionError("source outer deviance selection is absent")
    selected_deviance = min(
        deviance_candidates,
        key=lambda candidate: (
            candidate["mean_pilot_loss"],
            tuple(sorted(candidate["configuration"].items())),
        ),
    )["configuration"]
    if selected_deviance != nested_selection["deviance_residual_configuration"]:
        raise PermissionError("source outer and nested residual selection differ")

    classical = development.get("classical_selection")
    classical_methods = classical.get("methods", {}) if isinstance(classical, dict) else {}
    if (
        not isinstance(classical, dict)
        or classical.get("refusals") != {}
        or set(classical_methods)
        != {
            "common_effect_stratified_cmle",
            "pooled_saturated_poisson_interaction",
        }
    ):
        raise PermissionError("source outer classical selection differs")
    classical_mapping = {
        "common_effect_cmle": "common_effect_stratified_cmle",
        "pooled_saturated_poisson": "pooled_saturated_poisson_interaction",
    }
    for nested_name, outer_name in classical_mapping.items():
        if (
            classical_methods[outer_name].get("transport_multiplier")
            != _transport_multiplier(model["methods"][nested_name])
        ):
            raise PermissionError("source outer and nested classical transport differ")

    frozen_models = development.get("gse288020_held_prediction_models", {})
    core_models = frozen_models.get("primary_residual_destroyed_models", {})
    for name in ("primary", "destroyed_link"):
        nested = dict(model["methods"][name])
        nested.pop("status", None)
        if core_models.get(name) != nested:
            raise PermissionError("source outer and nested refit models differ")
    classical_models = frozen_models.get("classical_models", {}).get("models", {})
    for nested_name, outer_name in classical_mapping.items():
        nested = dict(model["methods"][nested_name])
        nested.pop("status", None)
        if classical_models.get(outer_name) != nested:
            raise PermissionError("source outer and nested classical models differ")


def _require_source_authorization() -> tuple[dict[str, Any], dict[str, Any], str]:
    authorization = _read_json(DEFAULT_SOURCE_AUTHORIZATION)
    if (
        set(authorization)
        != {
            "schema",
            "status",
            "recipient_rna_access_authorized",
            "gse309593_assay_identifier_or_barcode_access_before_authorization",
            "gse288020_protocol_tag",
            "gse288020_protocol_commit",
            "gse288020_development_tag",
            "gse288020_development_commit",
            "gse288020_mm_diagnostic_outcome_is_not_an_access_gate",
            "gse288020_mm_internal_test_outcome_values_used",
            "source_development_path",
            "source_development_sha256",
            "source_development_bytes",
            "source_development_attempt_sha256",
            "source_development_attempt_bytes",
            "source_runtime_sha256",
            "source_runtime_bytes",
            "source_model_sha256",
            "protocol_sha256",
            "candidate_designation_sha256",
            "source_split_amendment_sha256",
            "runner_sha256",
            "transitive_bindings",
        }
        or authorization.get("schema")
        != "gse309593-independent-study-source-authorization/1.0"
        or authorization.get("status")
        != "SOURCE_MODEL_AND_RECIPIENT_RNA_ACCESS_AUTHORIZED"
        or authorization.get("recipient_rna_access_authorized") is not True
        or authorization.get(
            "gse309593_assay_identifier_or_barcode_access_before_authorization"
        )
        != 0
    ):
        raise PermissionError(
            "source authorization is not an outcome-blind access release"
        )
    expected = {
        "protocol_sha256": _sha256(DEFAULT_PROTOCOL),
        "candidate_designation_sha256": _sha256(DEFAULT_CANDIDATE),
        "source_split_amendment_sha256": _sha256(DEFAULT_AMENDMENT),
        "runner_sha256": _sha256(Path(__file__)),
        "transitive_bindings": _binding_hashes(),
    }
    for key, value in expected.items():
        if authorization.get(key) != value:
            raise PermissionError(f"source authorization {key} differs")
    source_path = ROOT / authorization.get("source_development_path", "")
    contract = _read_json(DEFAULT_PROTOCOL)["source_model_contract"]
    canonical = contract["canonical_artifact_path"]
    if _relative(source_path) != canonical:
        raise PermissionError("source development path differs from the contract")
    if source_path.stat().st_size != authorization.get(
        "source_development_bytes"
    ) or _sha256(source_path) != authorization.get("source_development_sha256"):
        raise PermissionError("source development bytes differ from authorization")
    if (
        authorization.get("gse288020_mm_diagnostic_outcome_is_not_an_access_gate")
        is not True
    ):
        raise PermissionError(
            "source authorization improperly gates on the MM diagnostic"
        )
    if authorization.get("gse288020_mm_internal_test_outcome_values_used") != 0:
        raise PermissionError("source authorization permits MM outcome use")
    if (
        authorization.get("gse288020_protocol_tag") != GSE288020_PROTOCOL_TAG
        or authorization.get("gse288020_development_tag") != GSE288020_DEVELOPMENT_TAG
    ):
        raise PermissionError("source authorization uses an unregistered GSE288020 tag")
    for prefix in ("gse288020_protocol", "gse288020_development"):
        tag = authorization.get(f"{prefix}_tag")
        commit = authorization.get(f"{prefix}_commit")
        if not isinstance(tag, str) or not isinstance(commit, str):
            raise PermissionError(f"source authorization lacks {prefix} binding")
        _require_remote_tag_commit(tag, commit)
    development = _read_json(source_path)
    model = development.get("source_only_external_study_model")
    if not isinstance(model, dict):
        raise PermissionError(
            "source development artifact lacks the nested source model"
        )
    if contract.get("json_pointer") != "/source_only_external_study_model":
        raise PermissionError("source model JSON pointer differs from the contract")
    if _canonical_json_sha256(model) != authorization.get("source_model_sha256"):
        raise PermissionError("nested source model differs from authorization")
    _validate_source_model(model)
    _validate_gse288020_development_lineage(
        authorization, contract, source_path, development
    )
    _validate_source_model_against_development(development, model)
    commit = _require_public_tag(
        SOURCE_AUTHORIZATION_TAG,
        (
            *PROTOCOL_BINDINGS,
            _relative(DEFAULT_SOURCE_AUTHORIZATION),
            contract["canonical_attempt_path"],
            contract["canonical_runtime_path"],
            _relative(source_path),
        ),
    )
    return authorization, model, commit


STAGE_PATHS = {
    "rna": (DEFAULT_RNA_ATTEMPT, DEFAULT_RNA, RNA_ATTEMPT_TAG),
    "adt": (DEFAULT_ADT_ATTEMPT, DEFAULT_ADT, ADT_ATTEMPT_TAG),
    "prediction": (
        DEFAULT_PREDICTION_ATTEMPT,
        DEFAULT_PREDICTION,
        PREDICTION_ATTEMPT_TAG,
    ),
    "score": (DEFAULT_SCORE_ATTEMPT, DEFAULT_SCORE, SCORE_ATTEMPT_TAG),
}
STAGE_RESULT_TAGS = {
    "rna": RNA_TAG,
    "adt": ADT_TAG,
    "prediction": PREDICTION_TAG,
    "score": RESULT_TAG,
}


def _validated_public_stage(
    stage: str,
    tag: str,
    attempt: Path,
    output: Path,
    expected_status: str | set[str],
) -> tuple[dict[str, Any], str]:
    commit = _require_public_tag(
        tag,
        (*PROTOCOL_BINDINGS, _relative(attempt), _relative(output)),
    )
    payload = _read_json(output)
    runtime = _require_runtime_environment()
    allowed_statuses = (
        {expected_status} if isinstance(expected_status, str) else expected_status
    )
    lines = [json.loads(line) for line in attempt.read_text().splitlines() if line]
    if len(lines) != 3:
        raise PermissionError(f"public {stage} attempt ledger differs")
    started, execution, finished = lines
    if (
        set(started)
        != {
            "schema",
            "status",
            "event",
            "stage",
            "created_at_utc",
            "output",
            "target_files_opened_before_this_record",
            "target_identifiers_read_before_this_record",
            "target_assay_values_read_before_this_record",
            "prerequisites",
            "runtime_environment",
        }
        or set(execution)
        != {
            "created_at_utc",
            "event",
            "attempt_tag",
            "attempt_commit",
            "stage",
            "runtime_environment",
        }
        or set(finished)
        != {
            "created_at_utc",
            "event",
            "attempt_tag",
            "attempt_commit",
            "output_sha256",
            "stage",
            "status",
            "runtime_environment",
        }
    ):
        raise PermissionError(f"public {stage} ledger fields differ")
    if (
        started["schema"] != "gse309593-independent-study-attempt/1.0"
        or started["status"] != "ONE_SHOT_ATTEMPT_CLAIMED"
        or started["event"] != "STARTED"
        or started["output"] != _relative(output)
        or any(
            started[key] != 0
            for key in (
                "target_files_opened_before_this_record",
                "target_identifiers_read_before_this_record",
                "target_assay_values_read_before_this_record",
            )
        )
        or started["prerequisites"] != _prerequisites(stage)
        or execution["event"] != "EXECUTION_BEGINS_AFTER_PUBLIC_ATTEMPT"
        or finished["event"] != "SUCCEEDED"
        or finished["status"] not in allowed_statuses
        or finished["output_sha256"] != _sha256(output)
        or execution["attempt_tag"] != STAGE_PATHS[stage][2]
        or finished["attempt_tag"] != STAGE_PATHS[stage][2]
        or execution["attempt_commit"] != finished["attempt_commit"]
        or any(record["stage"] != stage for record in lines)
        or any(record["runtime_environment"] != runtime for record in lines)
        or payload.get("status") not in allowed_statuses
        or payload.get("stage") != stage
        or payload.get("runtime_environment") != runtime
        or payload.get("attempt_tag") != STAGE_PATHS[stage][2]
        or payload.get("attempt_commit") != execution["attempt_commit"]
    ):
        raise PermissionError(f"public {stage} lineage differs")
    _validate_attempt_tag_snapshot(
        STAGE_PATHS[stage][2], attempt, started, execution["attempt_commit"]
    )
    _validate_stage_payload(stage, payload)
    return payload, commit


def _validated_terminal_stage(stage: str) -> tuple[dict[str, Any], str]:
    attempt, output, attempt_tag = STAGE_PATHS[stage]
    result_tag = STAGE_RESULT_TAGS[stage]
    commit = _require_public_tag(
        result_tag,
        (*PROTOCOL_BINDINGS, _relative(attempt), _relative(output)),
    )
    payload = _read_json(output)
    runtime = _require_runtime_environment()
    lines = [json.loads(line) for line in attempt.read_text().splitlines() if line]
    if len(lines) != 3:
        raise PermissionError(f"public terminal {stage} attempt ledger differs")
    started, execution, finished = lines
    started_keys = {
        "schema",
        "status",
        "event",
        "stage",
        "created_at_utc",
        "output",
        "target_files_opened_before_this_record",
        "target_identifiers_read_before_this_record",
        "target_assay_values_read_before_this_record",
        "prerequisites",
        "runtime_environment",
    }
    execution_keys = {
        "created_at_utc",
        "event",
        "attempt_tag",
        "attempt_commit",
        "stage",
        "runtime_environment",
    }
    terminal_keys = {
        "created_at_utc",
        "event",
        "attempt_tag",
        "attempt_commit",
        "output_sha256",
        "reason_code",
        "stage",
        "runtime_environment",
    }
    payload_keys = {
        "schema",
        "status",
        "stage",
        "created_at_utc",
        "attempt_started_at_utc",
        "reason_code",
        "adaptive_rescue_permitted",
        "runtime_environment",
        "attempt_tag",
        "attempt_commit",
    }
    if set(started) != started_keys or set(execution) != execution_keys or set(
        finished
    ) != terminal_keys:
        raise PermissionError(f"public terminal {stage} ledger fields differ")
    if set(payload) not in (payload_keys, payload_keys | {"exception_class"}):
        raise PermissionError(f"public terminal {stage} payload fields differ")
    expected_status = f"TERMINAL_{stage.upper()}_REFUSAL"
    if (
        started["schema"] != "gse309593-independent-study-attempt/1.0"
        or started["status"] != "ONE_SHOT_ATTEMPT_CLAIMED"
        or started["event"] != "STARTED"
        or started["output"] != _relative(output)
        or any(
            started[key] != 0
            for key in (
                "target_files_opened_before_this_record",
                "target_identifiers_read_before_this_record",
                "target_assay_values_read_before_this_record",
            )
        )
        or started["prerequisites"] != _prerequisites(stage)
        or execution["event"] != "EXECUTION_BEGINS_AFTER_PUBLIC_ATTEMPT"
        or finished["event"] != "TERMINAL_REFUSAL"
        or finished["output_sha256"] != _sha256(output)
        or finished["reason_code"] != payload.get("reason_code")
        or execution["attempt_tag"] != attempt_tag
        or finished["attempt_tag"] != attempt_tag
        or execution["attempt_commit"] != finished["attempt_commit"]
        or any(record["stage"] != stage for record in lines)
        or any(record["runtime_environment"] != runtime for record in lines)
        or payload.get("schema")
        != "gse309593-independent-study-terminal-refusal/1.0"
        or payload.get("status") != expected_status
        or payload.get("stage") != stage
        or payload.get("attempt_started_at_utc") != started["created_at_utc"]
        or payload.get("adaptive_rescue_permitted") is not False
        or payload.get("runtime_environment") != runtime
        or payload.get("attempt_tag") != attempt_tag
        or payload.get("attempt_commit") != execution["attempt_commit"]
        or (
            payload.get("reason_code") == "UNEXPECTED_IMPLEMENTATION_FAILURE"
        )
        != ("exception_class" in payload)
    ):
        raise PermissionError(f"public terminal {stage} lineage differs")
    _validate_attempt_tag_snapshot(
        attempt_tag, attempt, started, execution["attempt_commit"]
    )
    return payload, commit


def _verify_public_stage(stage: str) -> tuple[dict[str, Any], str]:
    _, output, _ = STAGE_PATHS[stage]
    payload = _read_json(output)
    if payload.get("status") == f"TERMINAL_{stage.upper()}_REFUSAL":
        return _validated_terminal_stage(stage)
    verifiers = {
        "rna": _require_rna_stage,
        "adt": _require_adt_stage,
        "prediction": _require_prediction_stage,
        "score": _require_score_stage,
    }
    return verifiers[stage]()


def _require_rna_stage() -> tuple[dict[str, Any], str]:
    return _validated_public_stage(
        "rna",
        RNA_TAG,
        DEFAULT_RNA_ATTEMPT,
        DEFAULT_RNA,
        "RNA_STAGE_FROZEN_WITHOUT_ADT_ACCESS",
    )


def _require_adt_stage() -> tuple[dict[str, Any], str]:
    return _validated_public_stage(
        "adt",
        ADT_TAG,
        DEFAULT_ADT_ATTEMPT,
        DEFAULT_ADT,
        "ADT_STAGE_FROZEN_WITHOUT_RNA_STATE_ACCESS",
    )


def _require_prediction_stage() -> tuple[dict[str, Any], str]:
    return _validated_public_stage(
        "prediction",
        PREDICTION_TAG,
        DEFAULT_PREDICTION_ATTEMPT,
        DEFAULT_PREDICTION,
        "PREDICTIONS_FROZEN_BEFORE_RECIPIENT_PAIRING",
    )


def _require_score_stage() -> tuple[dict[str, Any], str]:
    return _validated_public_stage(
        "score",
        RESULT_TAG,
        DEFAULT_SCORE_ATTEMPT,
        DEFAULT_SCORE,
        COMPLETED_SCORE_STATUSES,
    )


def _require_score_authorization() -> tuple[dict[str, Any], str]:
    source, _, _ = _require_source_authorization()
    rna, _ = _require_rna_stage()
    adt, _ = _require_adt_stage()
    prediction, prediction_commit = _require_prediction_stage()
    authorization = _read_json(DEFAULT_SCORE_AUTHORIZATION)
    if (
        set(authorization)
        != {
            "schema",
            "status",
            "outcome_access_authorized",
            "recipient_joint_tables_formed_before_authorization",
            "prediction_tag",
            "prediction_commit",
            "prediction_path",
            "prediction_sha256",
            "prediction_bytes",
            "rna_stage_sha256",
            "adt_stage_sha256",
            "source_authorization_sha256",
            "protocol_sha256",
            "runner_sha256",
            "transitive_bindings",
        }
        or authorization.get("schema")
        != "gse309593-independent-study-score-authorization/1.0"
        or authorization.get("status") != "JOINT_SCORING_AUTHORIZED"
        or authorization.get("outcome_access_authorized") is not True
        or authorization.get("recipient_joint_tables_formed_before_authorization") != 0
    ):
        raise PermissionError("joint scoring is not authorized")
    expected = {
        "prediction_tag": PREDICTION_TAG,
        "prediction_commit": prediction_commit,
        "prediction_path": _relative(DEFAULT_PREDICTION),
        "prediction_sha256": _sha256(DEFAULT_PREDICTION),
        "prediction_bytes": DEFAULT_PREDICTION.stat().st_size,
        "rna_stage_sha256": _sha256(DEFAULT_RNA),
        "adt_stage_sha256": _sha256(DEFAULT_ADT),
        "source_authorization_sha256": _sha256(DEFAULT_SOURCE_AUTHORIZATION),
        "protocol_sha256": _sha256(DEFAULT_PROTOCOL),
        "runner_sha256": _sha256(Path(__file__)),
        "transitive_bindings": _binding_hashes(),
    }
    for key, value in expected.items():
        if authorization.get(key) != value:
            raise PermissionError(f"score authorization {key} differs")
    if prediction.get("source_model_sha256") != source.get("source_model_sha256"):
        raise PermissionError("prediction source binding differs")
    commit = _require_public_tag(
        SCORE_AUTHORIZATION_TAG,
        (
            *PROTOCOL_BINDINGS,
            _relative(DEFAULT_SOURCE_AUTHORIZATION),
            _relative(DEFAULT_RNA_ATTEMPT),
            _relative(DEFAULT_RNA),
            _relative(DEFAULT_ADT_ATTEMPT),
            _relative(DEFAULT_ADT),
            _relative(DEFAULT_PREDICTION_ATTEMPT),
            _relative(DEFAULT_PREDICTION),
            _relative(DEFAULT_SCORE_AUTHORIZATION),
        ),
    )
    return authorization, commit


def _prerequisites(stage: str) -> dict[str, Any]:
    runtime = _require_runtime_environment()
    protocol_commit = _require_public_tag(PROTOCOL_TAG, PROTOCOL_BINDINGS)
    source, _, source_commit = _require_source_authorization()
    result: dict[str, Any] = {
        "protocol_commit": protocol_commit,
        "source_authorization_commit": source_commit,
        "source_authorization_sha256": _sha256(DEFAULT_SOURCE_AUTHORIZATION),
        "source_model_sha256": source["source_model_sha256"],
        "runtime_spec_sha256": _sha256(DEFAULT_RUNTIME_SPEC),
        "runtime_environment": runtime,
    }
    if stage in {"adt", "prediction", "score"}:
        _, commit = _require_rna_stage()
        result.update(rna_commit=commit, rna_sha256=_sha256(DEFAULT_RNA))
    if stage in {"prediction", "score"}:
        _, commit = _require_adt_stage()
        result.update(adt_commit=commit, adt_sha256=_sha256(DEFAULT_ADT))
    if stage == "score":
        _, commit = _require_prediction_stage()
        _, authorization_commit = _require_score_authorization()
        result.update(
            prediction_commit=commit,
            prediction_sha256=_sha256(DEFAULT_PREDICTION),
            score_authorization_commit=authorization_commit,
            score_authorization_sha256=_sha256(DEFAULT_SCORE_AUTHORIZATION),
        )
    return result


def claim_stage(stage: str) -> dict[str, Any]:
    if stage not in STAGE_PATHS:
        raise ValueError("unknown stage")
    attempt, output, _ = STAGE_PATHS[stage]
    if attempt.exists() or output.exists():
        raise PermissionError(f"{stage} already has an attempt or output")
    prerequisites = _prerequisites(stage)
    runtime = _require_runtime_environment()
    payload = {
        "schema": "gse309593-independent-study-attempt/1.0",
        "status": "ONE_SHOT_ATTEMPT_CLAIMED",
        "event": "STARTED",
        "stage": stage,
        "created_at_utc": _timestamp(),
        "output": _relative(output),
        "target_files_opened_before_this_record": 0,
        "target_identifiers_read_before_this_record": 0,
        "target_assay_values_read_before_this_record": 0,
        "prerequisites": prerequisites,
        "runtime_environment": runtime,
    }
    _append_jsonl(attempt, payload, create=True)
    return payload


def _require_public_attempt(stage: str) -> tuple[dict[str, Any], str]:
    attempt, output, tag = STAGE_PATHS[stage]
    if output.exists():
        raise PermissionError(f"{stage} output already exists")
    lines = attempt.read_text().splitlines()
    if len(lines) != 1:
        raise PermissionError(f"{stage} attempt is no longer runnable")
    payload = json.loads(lines[0])
    if (
        payload.get("schema") != "gse309593-independent-study-attempt/1.0"
        or payload.get("status") != "ONE_SHOT_ATTEMPT_CLAIMED"
        or payload.get("event") != "STARTED"
        or payload.get("stage") != stage
        or payload.get("output") != _relative(output)
        or payload.get("prerequisites") != _prerequisites(stage)
        or payload.get("runtime_environment") != _require_runtime_environment()
        or set(payload)
        != {
            "schema",
            "status",
            "event",
            "stage",
            "created_at_utc",
            "output",
            "target_files_opened_before_this_record",
            "target_identifiers_read_before_this_record",
            "target_assay_values_read_before_this_record",
            "prerequisites",
            "runtime_environment",
        }
    ):
        raise PermissionError(f"{stage} attempt binding differs")
    commit = _require_public_tag(tag, (*PROTOCOL_BINDINGS, _relative(attempt)))
    return payload, commit


def _validate_attempt_tag_snapshot(
    tag: str, attempt: Path, started: dict[str, Any], commit: str
) -> None:
    object_type = subprocess.run(
        ["git", "cat-file", "-t", tag],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    local_commit = subprocess.run(
        ["git", "rev-list", "-n", "1", tag],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if object_type != "tag" or local_commit != commit:
        raise PermissionError(f"public {tag} attempt snapshot differs")
    _require_remote_tag_commit(tag, commit)
    tagged = subprocess.run(
        ["git", "show", f"{tag}:{_relative(attempt)}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    expected = (json.dumps(started, sort_keys=True, allow_nan=False) + "\n").encode()
    if tagged != expected:
        raise PermissionError(f"public {tag} does not contain the sole STARTED record")


def _run_claimed_stage(
    stage: str, function: Callable[[], dict[str, Any]]
) -> dict[str, Any]:
    attempt, output, _ = STAGE_PATHS[stage]
    claim, attempt_commit = _require_public_attempt(stage)
    attempt_tag = STAGE_PATHS[stage][2]
    runtime = _require_runtime_environment()
    _append_jsonl(
        attempt,
        {
            "created_at_utc": _timestamp(),
            "event": "EXECUTION_BEGINS_AFTER_PUBLIC_ATTEMPT",
            "attempt_tag": attempt_tag,
            "attempt_commit": attempt_commit,
            "stage": stage,
            "runtime_environment": runtime,
        },
        create=False,
    )
    try:
        payload = function()
        payload = dict(payload)
        payload["stage"] = stage
        payload["runtime_environment"] = runtime
        payload["attempt_tag"] = attempt_tag
        payload["attempt_commit"] = attempt_commit
        _validate_public_payload(payload)
        _validate_stage_payload(stage, payload)
        _write_json_x(output, payload)
    except ProtocolRefusal as error:
        payload = {
            "schema": "gse309593-independent-study-terminal-refusal/1.0",
            "status": f"TERMINAL_{stage.upper()}_REFUSAL",
            "stage": stage,
            "created_at_utc": _timestamp(),
            "attempt_started_at_utc": claim["created_at_utc"],
            "reason_code": error.code,
            "adaptive_rescue_permitted": False,
            "runtime_environment": runtime,
            "attempt_tag": attempt_tag,
            "attempt_commit": attempt_commit,
        }
        _write_json_x(output, payload)
        _append_jsonl(
            attempt,
            {
                "created_at_utc": _timestamp(),
                "event": "TERMINAL_REFUSAL",
                "attempt_tag": attempt_tag,
                "attempt_commit": attempt_commit,
                "output_sha256": _sha256(output),
                "reason_code": error.code,
                "stage": stage,
                "runtime_environment": runtime,
            },
            create=False,
        )
        return payload
    except Exception as error:
        payload = {
            "schema": "gse309593-independent-study-terminal-refusal/1.0",
            "status": f"TERMINAL_{stage.upper()}_REFUSAL",
            "stage": stage,
            "created_at_utc": _timestamp(),
            "attempt_started_at_utc": claim["created_at_utc"],
            "reason_code": "UNEXPECTED_IMPLEMENTATION_FAILURE",
            "exception_class": type(error).__name__,
            "adaptive_rescue_permitted": False,
            "runtime_environment": runtime,
            "attempt_tag": attempt_tag,
            "attempt_commit": attempt_commit,
        }
        _write_json_x(output, payload)
        _append_jsonl(
            attempt,
            {
                "created_at_utc": _timestamp(),
                "event": "TERMINAL_REFUSAL",
                "attempt_tag": attempt_tag,
                "attempt_commit": attempt_commit,
                "output_sha256": _sha256(output),
                "reason_code": "UNEXPECTED_IMPLEMENTATION_FAILURE",
                "stage": stage,
                "runtime_environment": runtime,
            },
            create=False,
        )
        raise
    _append_jsonl(
        attempt,
        {
            "created_at_utc": _timestamp(),
            "event": "SUCCEEDED",
            "attempt_tag": attempt_tag,
            "attempt_commit": attempt_commit,
            "output_sha256": _sha256(output),
            "stage": stage,
            "status": payload["status"],
            "runtime_environment": runtime,
        },
        create=False,
    )
    return payload


def _decode_strings(values: np.ndarray) -> list[str]:
    output = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            output.append(value.decode("utf-8"))
        else:
            output.append(str(value))
    return output


def _h5_axis(node: h5py.Dataset | h5py.Group) -> list[str]:
    if isinstance(node, h5py.Dataset):
        return _decode_strings(np.asarray(node[()]))
    if "categories" in node and "codes" in node:
        categories = _decode_strings(np.asarray(node["categories"][()]))
        codes = np.asarray(node["codes"][()], dtype=int)
        if np.any(codes < 0) or np.any(codes >= len(categories)):
            raise ProtocolRefusal("H5_CATEGORICAL_AXIS_INVALID")
        return [categories[index] for index in codes]
    raise ProtocolRefusal("H5_AXIS_ENCODING_UNSUPPORTED")


def _h5_sparse_shape(group: h5py.Group) -> tuple[int, int]:
    if "shape" in group:
        shape = np.asarray(group["shape"][()])
    elif "shape" in group.attrs:
        shape = np.asarray(group.attrs["shape"])
    else:
        raise ProtocolRefusal("H5_SPARSE_SHAPE_ABSENT")
    try:
        numeric = np.asarray(shape, dtype=float)
    except (TypeError, ValueError) as error:
        raise ProtocolRefusal("H5_SPARSE_SHAPE_INVALID") from error
    if (
        shape.ndim != 1
        or shape.shape != (2,)
        or not np.isfinite(numeric).all()
        or np.any(numeric <= 0)
        or not np.array_equal(numeric, np.rint(numeric))
    ):
        raise ProtocolRefusal("H5_SPARSE_SHAPE_INVALID")
    return int(numeric[0]), int(numeric[1])


def _integer_values(values: np.ndarray, code: str) -> np.ndarray:
    array = np.asarray(values)
    try:
        numeric = np.asarray(array, dtype=float)
    except (TypeError, ValueError) as error:
        raise ProtocolRefusal(code) from error
    if not np.issubdtype(numeric.dtype, np.number):
        raise ProtocolRefusal(code)
    if (
        not np.isfinite(numeric).all()
        or np.any(numeric < 0)
        or not np.allclose(numeric, np.rint(numeric), rtol=0.0, atol=1e-8)
    ):
        raise ProtocolRefusal(code)
    return np.rint(numeric).astype(np.int64)


def _exact_feature_indices(
    feature_names: list[str],
    candidates: list[str],
    feature_types: list[str] | None = None,
) -> tuple[list[str], list[int]]:
    if feature_types is not None and len(feature_types) != len(feature_names):
        raise ProtocolRefusal("RNA_FEATURE_TYPE_AXIS_LENGTH_MISMATCH")
    supported: list[str] = []
    indices: list[int] = []
    for symbol in candidates:
        matches = [
            index
            for index, value in enumerate(feature_names)
            if value == symbol
            and (feature_types is None or feature_types[index] == "Gene Expression")
        ]
        if len(matches) > 1:
            raise ProtocolRefusal("RNA_FEATURE_MAPPING_AMBIGUOUS")
        if matches:
            supported.append(symbol)
            indices.append(matches[0])
    return supported, indices


def _selected_cells(
    subject: str,
    identifiers: list[str],
    eligible_indices: Iterable[int] | None = None,
) -> tuple[list[int], list[str]]:
    if not identifiers or any(not value for value in identifiers):
        raise ProtocolRefusal("RNA_IDENTIFIER_AXIS_EMPTY")
    if len(identifiers) != len(set(identifiers)):
        raise ProtocolRefusal("RNA_IDENTIFIER_AXIS_DUPLICATED")
    eligible = (
        list(range(len(identifiers)))
        if eligible_indices is None
        else list(eligible_indices)
    )
    if len(eligible) != len(set(eligible)) or any(
        index < 0 or index >= len(identifiers) for index in eligible
    ):
        raise ProtocolRefusal("RNA_QC_ELIGIBLE_AXIS_INVALID")
    if len(eligible) < CELL_BUDGET:
        raise ProtocolRefusal("RNA_QC_SUPPORT_BELOW_512")
    order = sorted(
        eligible,
        key=lambda index: (
            _salted_hash(CELL_SELECTION_SALT, subject, identifiers[index]),
            identifiers[index],
        ),
    )[:CELL_BUDGET]
    return order, [identifiers[index] for index in order]


def _sparse_structure(
    group: h5py.Group, shape: tuple[int, int], encoding: str
) -> np.ndarray:
    if not {"data", "indices", "indptr"} <= set(group):
        raise ProtocolRefusal("H5_SPARSE_DATASETS_ABSENT")
    if any(group[name].ndim != 1 for name in ("data", "indices", "indptr")):
        raise ProtocolRefusal("H5_SPARSE_STRUCTURE_INVALID")
    raw_indptr = np.asarray(group["indptr"][()])
    raw_indices = np.asarray(group["indices"][()])
    try:
        numeric_indptr = np.asarray(raw_indptr, dtype=float)
        numeric_indices = np.asarray(raw_indices, dtype=float)
    except (TypeError, ValueError) as error:
        raise ProtocolRefusal("H5_SPARSE_STRUCTURE_INVALID") from error
    if (
        not np.isfinite(numeric_indptr).all()
        or not np.isfinite(numeric_indices).all()
        or not np.array_equal(numeric_indptr, np.rint(numeric_indptr))
        or not np.array_equal(numeric_indices, np.rint(numeric_indices))
    ):
        raise ProtocolRefusal("H5_SPARSE_STRUCTURE_INVALID")
    indptr = np.rint(numeric_indptr).astype(np.int64)
    indices = np.rint(numeric_indices).astype(np.int64)
    data_length = len(group["data"])
    index_length = len(indices)
    expected_pointers = (
        shape[1] + 1
        if encoding in {"feature_by_cell_csc", "cell_by_feature_csc"}
        else shape[0] + 1
    )
    if (
        len(indptr) != expected_pointers
        or len(indptr) == 0
        or indptr[0] != 0
        or np.any(indptr[1:] < indptr[:-1])
        or int(indptr[-1]) != data_length
        or data_length != index_length
    ):
        raise ProtocolRefusal("H5_SPARSE_STRUCTURE_INVALID")
    index_bound = shape[0] if encoding != "cell_by_feature_csr" else shape[1]
    if np.any(indices < 0) or np.any(indices >= index_bound):
        raise ProtocolRefusal("H5_SPARSE_INDEX_OUT_OF_RANGE")
    if any(
        len(segment) != len(set(segment.tolist()))
        for start, stop in zip(indptr[:-1], indptr[1:])
        for segment in (indices[int(start) : int(stop)],)
    ):
        raise ProtocolRefusal("H5_SPARSE_DUPLICATE_INDEX")
    return indptr


def _sparse_rna_qc(
    group: h5py.Group,
    shape: tuple[int, int],
    encoding: str,
    mitochondrial_features: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    indptr = _sparse_structure(group, shape, encoding)
    cells = shape[1] if encoding == "feature_by_cell_csc" else shape[0]
    detected = np.zeros(cells, dtype=np.int64)
    totals = np.zeros(cells, dtype=np.int64)
    mitochondrial = np.zeros(cells, dtype=np.int64)
    decoded = 0

    if encoding in {"feature_by_cell_csc", "cell_by_feature_csr"}:
        feature_bound = shape[0] if encoding == "feature_by_cell_csc" else shape[1]
        for cell in range(cells):
            start, stop = int(indptr[cell]), int(indptr[cell + 1])
            features = np.asarray(group["indices"][start:stop], dtype=np.int64)
            values = _integer_values(
                np.asarray(group["data"][start:stop]), "RNA_MATRIX_IS_NOT_RAW_COUNTS"
            )
            if np.any(features < 0) or np.any(features >= feature_bound):
                raise ProtocolRefusal("H5_SPARSE_INDEX_OUT_OF_RANGE")
            if len(features) != len(set(features.tolist())):
                raise ProtocolRefusal("H5_SPARSE_DUPLICATE_INDEX")
            positive = values > 0
            detected[cell] = int(np.count_nonzero(positive))
            totals[cell] = int(values.sum())
            mitochondrial[cell] = int(
                sum(
                    int(value)
                    for feature, value in zip(features, values)
                    if int(feature) in mitochondrial_features
                )
            )
            decoded += len(values)
    elif encoding == "cell_by_feature_csc":
        for feature in range(shape[1]):
            start, stop = int(indptr[feature]), int(indptr[feature + 1])
            cells_axis = np.asarray(group["indices"][start:stop], dtype=np.int64)
            values = _integer_values(
                np.asarray(group["data"][start:stop]), "RNA_MATRIX_IS_NOT_RAW_COUNTS"
            )
            if np.any(cells_axis < 0) or np.any(cells_axis >= shape[0]):
                raise ProtocolRefusal("H5_SPARSE_INDEX_OUT_OF_RANGE")
            if len(cells_axis) != len(set(cells_axis.tolist())):
                raise ProtocolRefusal("H5_SPARSE_DUPLICATE_INDEX")
            np.add.at(detected, cells_axis[values > 0], 1)
            np.add.at(totals, cells_axis, values)
            if feature in mitochondrial_features:
                np.add.at(mitochondrial, cells_axis, values)
            decoded += len(values)
    else:
        raise ProtocolRefusal("H5_SPARSE_ENCODING_UNSUPPORTED")
    return detected, totals, mitochondrial, decoded


def _qc_eligible_indices(
    detected: np.ndarray, totals: np.ndarray, mitochondrial: np.ndarray
) -> np.ndarray:
    fraction = np.divide(
        mitochondrial,
        totals,
        out=np.ones_like(mitochondrial, dtype=float),
        where=totals > 0,
    )
    return np.flatnonzero(
        (detected >= MINIMUM_DETECTED_GENES)
        & (fraction <= MAXIMUM_MITOCHONDRIAL_FRACTION)
        & (totals <= MAXIMUM_RNA_UMIS)
    )


def _sparse_subset(
    group: h5py.Group,
    shape: tuple[int, int],
    encoding: str,
    selected_cells: list[int],
    feature_indices: list[int],
) -> tuple[np.ndarray, int]:
    indptr = _sparse_structure(group, shape, encoding)
    output = np.zeros((len(selected_cells), len(feature_indices)), dtype=np.int64)
    feature_lookup = {value: index for index, value in enumerate(feature_indices)}
    cell_lookup = {value: index for index, value in enumerate(selected_cells)}
    decoded = 0

    if encoding == "feature_by_cell_csc":
        for out_cell, cell in enumerate(selected_cells):
            start, stop = int(indptr[cell]), int(indptr[cell + 1])
            rows = np.asarray(group["indices"][start:stop], dtype=np.int64)
            if np.any(rows < 0) or np.any(rows >= shape[0]):
                raise ProtocolRefusal("H5_SPARSE_INDEX_OUT_OF_RANGE")
            values = _integer_values(
                np.asarray(group["data"][start:stop]), "RNA_MATRIX_IS_NOT_RAW_COUNTS"
            )
            decoded += len(values)
            for row, value in zip(rows, values):
                target = feature_lookup.get(int(row))
                if target is not None:
                    output[out_cell, target] += int(value)
    elif encoding == "cell_by_feature_csr":
        for out_cell, cell in enumerate(selected_cells):
            start, stop = int(indptr[cell]), int(indptr[cell + 1])
            columns = np.asarray(group["indices"][start:stop], dtype=np.int64)
            if np.any(columns < 0) or np.any(columns >= shape[1]):
                raise ProtocolRefusal("H5_SPARSE_INDEX_OUT_OF_RANGE")
            values = _integer_values(
                np.asarray(group["data"][start:stop]), "RNA_MATRIX_IS_NOT_RAW_COUNTS"
            )
            decoded += len(values)
            for column, value in zip(columns, values):
                target = feature_lookup.get(int(column))
                if target is not None:
                    output[out_cell, target] += int(value)
    elif encoding == "cell_by_feature_csc":
        for out_feature, feature in enumerate(feature_indices):
            start, stop = int(indptr[feature]), int(indptr[feature + 1])
            rows = np.asarray(group["indices"][start:stop], dtype=np.int64)
            if np.any(rows < 0) or np.any(rows >= shape[0]):
                raise ProtocolRefusal("H5_SPARSE_INDEX_OUT_OF_RANGE")
            values = _integer_values(
                np.asarray(group["data"][start:stop]), "RNA_MATRIX_IS_NOT_RAW_COUNTS"
            )
            decoded += len(values)
            for row, value in zip(rows, values):
                target = cell_lookup.get(int(row))
                if target is not None:
                    output[target, out_feature] += int(value)
    else:
        raise ProtocolRefusal("H5_SPARSE_ENCODING_UNSUPPORTED")
    return output, decoded


def _anndata_feature_axis(handle: h5py.File, prefix: str) -> tuple[list[str], str]:
    roots = [f"{prefix}/var"] if prefix else ["var"]
    axis_names = ("gene_symbol", "feature_name", "_index", "index")
    for root in roots:
        for name in axis_names:
            path = f"{root}/{name}"
            if path not in handle:
                continue
            values = _h5_axis(handle[path])
            if values:
                return values, path
    raise ProtocolRefusal("RNA_EXACT_SYMBOL_AXIS_UNRESOLVED")


def _anndata_feature_types(
    handle: h5py.File, prefix: str, feature_count: int
) -> list[str] | None:
    root = f"{prefix}/var" if prefix else "var"
    for name in ("feature_types", "feature_type"):
        path = f"{root}/{name}"
        if path not in handle:
            continue
        values = _h5_axis(handle[path])
        if len(values) != feature_count:
            raise ProtocolRefusal("RNA_FEATURE_TYPE_AXIS_LENGTH_MISMATCH")
        if any(value != "Gene Expression" for value in values):
            raise ProtocolRefusal("RNA_H5_CONTAINS_NON_RNA_MATRIX_VALUES")
        return values
    return None


def _reduce_rna_h5(
    path: Path, subject: str, candidate_symbols: list[str]
) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        if "matrix" in handle and isinstance(handle["matrix"], h5py.Group):
            matrix = handle["matrix"]
            required = {"barcodes", "features", "data", "indices", "indptr", "shape"}
            if not required <= set(matrix):
                raise ProtocolRefusal("TENX_H5_SCHEMA_INCOMPLETE")
            identifiers = _h5_axis(matrix["barcodes"])
            features = matrix["features"]
            if not isinstance(features, h5py.Group) or "name" not in features:
                raise ProtocolRefusal("TENX_FEATURE_AXIS_ABSENT")
            names = _h5_axis(features["name"])
            types = (
                _h5_axis(features["feature_type"])
                if "feature_type" in features
                else None
            )
            if types is not None and any(value != "Gene Expression" for value in types):
                raise ProtocolRefusal("RNA_H5_CONTAINS_NON_RNA_MATRIX_VALUES")
            supported, feature_indices = _exact_feature_indices(
                names, candidate_symbols, types
            )
            shape = _h5_sparse_shape(matrix)
            if shape != (len(names), len(identifiers)):
                raise ProtocolRefusal("TENX_MATRIX_AXIS_LENGTH_MISMATCH")
            mitochondrial = {
                index for index, name in enumerate(names) if name.startswith("MT-")
            }
            detected, totals, mitochondrial_totals, qc_decoded = _sparse_rna_qc(
                matrix,
                shape,
                "feature_by_cell_csc",
                mitochondrial,
            )
            eligible_indices = _qc_eligible_indices(
                detected, totals, mitochondrial_totals
            )
            selected_indices, selected_ids = _selected_cells(
                subject, identifiers, eligible_indices
            )
            counts, decoded = _sparse_subset(
                matrix,
                shape,
                "feature_by_cell_csc",
                selected_indices,
                feature_indices,
            )
            encoding = "10x_feature_by_cell_csc"
            matrix_path = "matrix"
            symbol_path = "matrix/features/name"
            matrix_shape = shape
            matrix_storage_entries = len(matrix["data"])
            sparse_structure_validated = True
        else:
            if "obs/_index" in handle:
                identifier_path = "obs/_index"
            elif "obs/index" in handle:
                identifier_path = "obs/index"
            else:
                raise ProtocolRefusal("ANNDATA_IDENTIFIER_AXIS_ABSENT")
            identifiers = _h5_axis(handle[identifier_path])
            matrix_path = ""
            prefix = ""
            for candidate_path, candidate_prefix in (
                ("layers/counts", ""),
                ("raw/X", "raw"),
                ("X", ""),
            ):
                if candidate_path in handle:
                    matrix_path = candidate_path
                    prefix = candidate_prefix
                    break
            if not matrix_path:
                raise ProtocolRefusal("ANNDATA_COUNT_MATRIX_ABSENT")
            names, symbol_path = _anndata_feature_axis(handle, prefix)
            feature_types = _anndata_feature_types(handle, prefix, len(names))
            supported, feature_indices = _exact_feature_indices(
                names, candidate_symbols, feature_types
            )
            mitochondrial = {
                index for index, name in enumerate(names) if name.startswith("MT-")
            }
            node = handle[matrix_path]
            if isinstance(node, h5py.Dataset):
                if node.ndim != 2 or node.shape != (len(identifiers), len(names)):
                    raise ProtocolRefusal("ANNDATA_DENSE_AXIS_LENGTH_MISMATCH")
                detected = np.empty(len(identifiers), dtype=np.int64)
                totals = np.empty(len(identifiers), dtype=np.int64)
                mitochondrial_totals = np.empty(len(identifiers), dtype=np.int64)
                for cell in range(len(identifiers)):
                    values = _integer_values(
                        np.asarray(node[cell, :]), "RNA_MATRIX_IS_NOT_RAW_COUNTS"
                    )
                    detected[cell] = int(np.count_nonzero(values > 0))
                    totals[cell] = int(values.sum())
                    mitochondrial_totals[cell] = int(
                        sum(int(values[index]) for index in mitochondrial)
                    )
                qc_decoded = len(identifiers) * len(names)
                eligible_indices = _qc_eligible_indices(
                    detected, totals, mitochondrial_totals
                )
                selected_indices, selected_ids = _selected_cells(
                    subject, identifiers, eligible_indices
                )
                counts = np.empty(
                    (len(selected_indices), len(feature_indices)), dtype=np.int64
                )
                for row, cell in enumerate(selected_indices):
                    values = np.asarray(
                        [node[cell, feature] for feature in feature_indices]
                    )
                    counts[row] = _integer_values(
                        values, "RNA_MATRIX_IS_NOT_RAW_COUNTS"
                    )
                decoded = counts.size
                encoding = "anndata_cell_by_feature_dense"
                matrix_shape = tuple(int(value) for value in node.shape)
                matrix_storage_entries = int(np.prod(node.shape))
                sparse_structure_validated = False
            elif isinstance(node, h5py.Group):
                shape = _h5_sparse_shape(node)
                if shape != (len(identifiers), len(names)):
                    raise ProtocolRefusal("ANNDATA_SPARSE_AXIS_LENGTH_MISMATCH")
                raw_encoding = node.attrs.get("encoding-type", "")
                if isinstance(raw_encoding, bytes):
                    raw_encoding = raw_encoding.decode()
                if raw_encoding == "csr_matrix":
                    sparse_encoding = "cell_by_feature_csr"
                elif raw_encoding == "csc_matrix":
                    sparse_encoding = "cell_by_feature_csc"
                else:
                    pointer_length = len(node["indptr"])
                    if (
                        pointer_length == shape[0] + 1
                        and pointer_length != shape[1] + 1
                    ):
                        sparse_encoding = "cell_by_feature_csr"
                    elif (
                        pointer_length == shape[1] + 1
                        and pointer_length != shape[0] + 1
                    ):
                        sparse_encoding = "cell_by_feature_csc"
                    else:
                        raise ProtocolRefusal("ANNDATA_SPARSE_ENCODING_AMBIGUOUS")
                detected, totals, mitochondrial_totals, qc_decoded = _sparse_rna_qc(
                    node,
                    shape,
                    sparse_encoding,
                    mitochondrial,
                )
                eligible_indices = _qc_eligible_indices(
                    detected, totals, mitochondrial_totals
                )
                selected_indices, selected_ids = _selected_cells(
                    subject, identifiers, eligible_indices
                )
                counts, decoded = _sparse_subset(
                    node,
                    shape,
                    sparse_encoding,
                    selected_indices,
                    feature_indices,
                )
                encoding = f"anndata_{sparse_encoding}"
                matrix_shape = shape
                matrix_storage_entries = len(node["data"])
                sparse_structure_validated = True
            else:
                raise ProtocolRefusal("ANNDATA_COUNT_MATRIX_UNSUPPORTED")

    if len(supported) < MINIMUM_COGNATES:
        raise ProtocolRefusal("RNA_SCHEMA_HAS_FEWER_THAN_NINE_CANDIDATES")
    states = (counts > 0).astype(np.uint8)
    return {
        "selected_ids": selected_ids,
        "selected_axis_sha256": _axis_sha256(subject, selected_ids),
        "available_rna_symbols": supported,
        "states": {symbol: states[:, index] for index, symbol in enumerate(supported)},
        "row_margins": {
            symbol: [
                CELL_BUDGET - int(states[:, index].sum()),
                int(states[:, index].sum()),
            ]
            for index, symbol in enumerate(supported)
        },
        "audit": {
            "format": encoding,
            "matrix_path": matrix_path,
            "symbol_axis_path": symbol_path,
            "matrix_shape": list(matrix_shape),
            "matrix_storage_entries": matrix_storage_entries,
            "sparse_structure_validated": sparse_structure_validated,
            "identifier_values_read": len(identifiers),
            "qc_eligible_identifiers": len(eligible_indices),
            "rna_qc_thresholds": {
                "minimum_detected_genes": MINIMUM_DETECTED_GENES,
                "maximum_mitochondrial_fraction": MAXIMUM_MITOCHONDRIAL_FRACTION,
                "maximum_rna_umis": MAXIMUM_RNA_UMIS,
            },
            "selected_identifiers": CELL_BUDGET,
            "rna_qc_numeric_values_decoded": qc_decoded,
            "selected_marker_numeric_values_decoded": decoded,
            "adt_files_opened": 0,
            "adt_values_read": 0,
        },
    }


def _fetch_designated_file(
    url: str, expected_bytes: int, scratch_dir: Path, suffix: str
) -> tuple[Path, str]:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(
        dir=scratch_dir, suffix=suffix, delete=False
    ) as temporary:
        path = Path(temporary.name)
        try:
            with urlopen(url) as response:
                total = 0
                while True:
                    block = response.read(8 << 20)
                    if not block:
                        break
                    total += len(block)
                    if total > expected_bytes:
                        raise ProtocolRefusal("DESIGNATED_FILE_BYTE_COUNT_EXCEEDED")
                    digest.update(block)
                    temporary.write(block)
        except Exception:
            path.unlink(missing_ok=True)
            raise
    if path.stat().st_size != expected_bytes:
        path.unlink(missing_ok=True)
        raise ProtocolRefusal("DESIGNATED_FILE_BYTE_COUNT_MISMATCH")
    return path, digest.hexdigest()


def _source_supported_candidates(
    candidate: dict[str, Any], source_model: dict[str, Any]
) -> list[dict[str, str]]:
    source_symbols = {
        row["rna_symbol"] for row in _source_marker_mappings(source_model)
    }
    values = [
        dict(row)
        for row in candidate["marker_panel"]["ordered_candidates"]
        if row["rna_symbol"] in source_symbols
    ]
    if len({row["rna_symbol"] for row in values}) != len(values):
        raise PermissionError("candidate RNA symbols are not unique")
    if len({row["adt_target"] for row in values}) != len(values):
        raise PermissionError("candidate ADT targets are not unique")
    return values


def _read_adt_csv(
    path: Path,
    subject: str,
    selected_ids: list[str],
    candidates: list[str],
) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ProtocolRefusal("ADT_CSV_EMPTY") from error
        if len(header) < 2 or len(header) != len(set(header)):
            raise ProtocolRefusal("ADT_CSV_HEADER_INVALID_OR_DUPLICATED")
        marker_headers = set(header[1:]) & set(candidates)
        selected_headers = set(header[1:]) & set(selected_ids)
        if marker_headers and selected_headers:
            raise ProtocolRefusal("ADT_CSV_ORIENTATION_AMBIGUOUS")

        selected_set = set(selected_ids)
        if marker_headers:
            orientation = "cells_by_markers"
            available = [value for value in candidates if value in marker_headers]
            columns = {value: header.index(value) for value in available}
            rows: dict[str, list[int]] = {}
            all_identifiers: set[str] = set()
            identifiers_seen = 0
            numeric_values = 0
            for record in reader:
                if len(record) != len(header):
                    raise ProtocolRefusal("ADT_CSV_ROW_LENGTH_MISMATCH")
                identifier = record[0]
                if not identifier or identifier in all_identifiers:
                    raise ProtocolRefusal("ADT_IDENTIFIER_AXIS_EMPTY_OR_DUPLICATED")
                all_identifiers.add(identifier)
                if identifier not in selected_set:
                    continue
                if identifier in rows:
                    raise ProtocolRefusal("ADT_SELECTED_IDENTIFIER_DUPLICATED")
                values = _integer_values(
                    np.asarray([record[columns[value]] for value in available]),
                    "ADT_MATRIX_IS_NOT_NONNEGATIVE_COUNTS",
                )
                rows[identifier] = values.tolist()
                identifiers_seen += 1
                numeric_values += len(values)
            missing = selected_set - set(rows)
            if missing:
                raise ProtocolRefusal("ADT_SELECTED_IDENTIFIER_MISSING")
            counts = np.asarray([rows[value] for value in selected_ids], dtype=np.int64)
        else:
            if len(selected_headers) != CELL_BUDGET:
                raise ProtocolRefusal("ADT_CSV_ORIENTATION_UNRESOLVED")
            orientation = "markers_by_cells"
            selected_columns = [header.index(value) for value in selected_ids]
            rows_by_marker: dict[str, np.ndarray] = {}
            identifiers_seen = len(selected_headers)
            numeric_values = 0
            for record in reader:
                if len(record) != len(header):
                    raise ProtocolRefusal("ADT_CSV_ROW_LENGTH_MISMATCH")
                marker = record[0]
                if marker not in candidates:
                    continue
                if marker in rows_by_marker:
                    raise ProtocolRefusal("ADT_TARGET_ROW_DUPLICATED")
                values = _integer_values(
                    np.asarray([record[index] for index in selected_columns]),
                    "ADT_MATRIX_IS_NOT_NONNEGATIVE_COUNTS",
                )
                rows_by_marker[marker] = values
                numeric_values += len(values)
            available = [value for value in candidates if value in rows_by_marker]
            counts = np.stack([rows_by_marker[value] for value in available], axis=1)

    if len(available) < MINIMUM_COGNATES:
        raise ProtocolRefusal("ADT_SCHEMA_HAS_FEWER_THAN_NINE_CANDIDATES")
    states = np.zeros_like(counts, dtype=np.uint8)
    variation_qc: dict[str, dict[str, Any]] = {}
    marker_support: dict[str, bool] = {}
    for marker_index, marker in enumerate(available):
        ordered_values = np.sort(counts[:, marker_index])
        lower_value = int(ordered_values[CELL_BUDGET // 2 - 1])
        upper_value = int(ordered_values[CELL_BUDGET // 2])
        boundary_tie_cells = (
            int(np.count_nonzero(ordered_values == lower_value))
            if lower_value == upper_value
            else 0
        )
        supported = boundary_tie_cells <= int(
            CELL_BUDGET * MAXIMUM_ADT_BOUNDARY_TIE_FRACTION
        )
        marker_support[marker] = supported
        variation_qc[marker] = {
            "distinct_raw_values": int(len(np.unique(ordered_values))),
            "lower_boundary_value": lower_value,
            "upper_boundary_value": upper_value,
            "boundary_tie_cells": boundary_tie_cells,
            "maximum_boundary_tie_cells": int(
                CELL_BUDGET * MAXIMUM_ADT_BOUNDARY_TIE_FRACTION
            ),
            "passes": supported,
        }
        if not supported:
            continue
        order = sorted(
            range(CELL_BUDGET),
            key=lambda index: (
                int(counts[index, marker_index]),
                _salted_hash(ADT_TIE_SALT, subject, marker, selected_ids[index]),
                selected_ids[index],
            ),
        )
        states[np.asarray(order[CELL_BUDGET // 2 :]), marker_index] = 1
        if int(states[:, marker_index].sum()) != CELL_BUDGET // 2:
            raise ProtocolRefusal("ADT_MIDRANK_MARGIN_INVALID")
    return {
        "available_adt_targets": available,
        "states": {marker: states[:, index] for index, marker in enumerate(available)},
        "column_margins": {
            marker: (
                [CELL_BUDGET // 2, CELL_BUDGET // 2]
                if marker_support[marker]
                else [CELL_BUDGET, 0]
            )
            for marker in available
        },
        "marker_support": marker_support,
        "variation_qc": variation_qc,
        "audit": {
            "orientation": orientation,
            "selected_identifier_values_read": identifiers_seen,
            "selected_adt_numeric_values_read": numeric_values,
            "rna_state_files_opened": 0,
            "rna_state_values_read": 0,
        },
    }


def _designated_url(
    candidate: dict[str, Any], subject: dict[str, Any], key: str
) -> str:
    record = subject[key]
    return candidate["recipient_cohort"]["per_sample_url_template"].format(
        gsm=subject["gsm"], file=quote(record["name"], safe="")
    )


def _ordered_schema_intersection(
    current: list[str] | None, observed: list[str]
) -> list[str]:
    if len(observed) != len(set(observed)):
        raise ProtocolRefusal("RECIPIENT_FEATURE_SCHEMA_DUPLICATED")
    if current is None:
        return list(observed)
    observed_set = set(observed)
    return [value for value in current if value in observed_set]


def _require_common_schema(axis: list[str] | None, code: str) -> list[str]:
    if axis is None or len(axis) < MINIMUM_COGNATES:
        raise ProtocolRefusal(code)
    return axis


def _rna_stage_body(
    scratch_dir: Path, selection_bridge_path: Path, rna_states_path: Path
) -> dict[str, Any]:
    scratch_dir = _private_path(scratch_dir)
    source_authorization, source_model, source_commit = _require_source_authorization()
    candidate = _candidate()
    supported = _source_supported_candidates(candidate, source_model)
    candidate_symbols = [row["rna_symbol"] for row in supported]
    selection_bridge_path = _private_path(selection_bridge_path)
    rna_states_path = _private_path(rna_states_path)
    if selection_bridge_path.exists() or rna_states_path.exists():
        raise PermissionError("private RNA outputs already exist")

    public_subjects: list[dict[str, Any]] = []
    bridge_subjects: dict[str, Any] = {}
    state_subjects: dict[str, Any] = {}
    common_schema: list[str] | None = None
    for row in candidate["recipient_cohort"]["subjects"]:
        subject = row["subject_id"]
        expected_bytes = int(row["rna_h5"]["bytes"])
        path, file_sha256 = _fetch_designated_file(
            _designated_url(candidate, row, "rna_h5"),
            expected_bytes,
            scratch_dir,
            ".h5",
        )
        try:
            reduced = _reduce_rna_h5(path, subject, candidate_symbols)
        finally:
            path.unlink(missing_ok=True)
        available = reduced["available_rna_symbols"]
        common_schema = _ordered_schema_intersection(common_schema, available)
        bridge_subjects[subject] = {
            "selected_ids": reduced["selected_ids"],
            "selected_axis_sha256": reduced["selected_axis_sha256"],
        }
        state_subjects[subject] = {
            "selected_axis_sha256": reduced["selected_axis_sha256"],
            "states": {
                symbol: values.astype(int).tolist()
                for symbol, values in reduced["states"].items()
            },
        }
        public_subjects.append(
            {
                "subject_id": subject,
                "gsm": row["gsm"],
                "batch": row["batch"],
                "rna_h5_name": row["rna_h5"]["name"],
                "rna_h5_bytes": expected_bytes,
                "rna_h5_sha256": file_sha256,
                "selected_axis_sha256": reduced["selected_axis_sha256"],
                "row_margins": reduced["row_margins"],
                "rna_state_sha256": {
                    symbol: _array_sha256(values)
                    for symbol, values in reduced["states"].items()
                },
                "access_audit": reduced["audit"],
            }
        )
    common_schema = _require_common_schema(
        common_schema, "RNA_COMMON_SCHEMA_HAS_FEWER_THAN_NINE_CANDIDATES"
    )
    for record in public_subjects:
        for key in ("row_margins", "rna_state_sha256"):
            record[key] = {symbol: record[key][symbol] for symbol in common_schema}
    for record in state_subjects.values():
        record["states"] = {
            symbol: record["states"][symbol] for symbol in common_schema
        }

    bridge = {
        "schema": "gse309593-private-selection-bridge/1.0",
        "subjects": bridge_subjects,
    }
    states = {
        "schema": "gse309593-private-rna-states/1.0",
        "available_rna_symbols": common_schema,
        "subjects": state_subjects,
    }
    _write_json_x(selection_bridge_path, bridge)
    _write_json_x(rna_states_path, states)
    return {
        "schema": "gse309593-independent-study-rna-stage/1.0",
        "status": "RNA_STAGE_FROZEN_WITHOUT_ADT_ACCESS",
        "created_at_utc": _timestamp(),
        "source_authorization_commit": source_commit,
        "source_authorization_sha256": _sha256(DEFAULT_SOURCE_AUTHORIZATION),
        "source_model_sha256": source_authorization["source_model_sha256"],
        "recipient_subjects": [row["subject_id"] for row in public_subjects],
        "available_rna_symbols": common_schema,
        "subjects": public_subjects,
        "private_artifacts": {
            "selection_bridge_sha256": _sha256(selection_bridge_path),
            "selection_bridge_bytes": selection_bridge_path.stat().st_size,
            "rna_states_sha256": _sha256(rna_states_path),
            "rna_states_bytes": rna_states_path.stat().st_size,
            "paths_serialized": 0,
        },
        "access_boundary": {
            "adt_files_requested": 0,
            "adt_files_opened": 0,
            "adt_numeric_values_read": 0,
            "cell_selection_used_rna_identifiers_only": True,
            "adt_identifier_fallback_or_resampling_permitted": False,
        },
    }


def _adt_stage_body(
    scratch_dir: Path, selection_bridge_path: Path, adt_states_path: Path
) -> dict[str, Any]:
    scratch_dir = _private_path(scratch_dir)
    _, source_model, _ = _require_source_authorization()
    rna, rna_commit = _require_rna_stage()
    candidate = _candidate()
    supported = _source_supported_candidates(candidate, source_model)
    candidate_targets = [row["adt_target"] for row in supported]
    selection_bridge_path = _private_path(selection_bridge_path)
    adt_states_path = _private_path(adt_states_path)
    if adt_states_path.exists():
        raise PermissionError("private ADT output already exists")
    if (
        selection_bridge_path.stat().st_size
        != rna["private_artifacts"]["selection_bridge_bytes"]
        or _sha256(selection_bridge_path)
        != rna["private_artifacts"]["selection_bridge_sha256"]
    ):
        raise PermissionError("private selection bridge differs from RNA freeze")
    bridge = _read_json(selection_bridge_path)
    if bridge.get("schema") != "gse309593-private-selection-bridge/1.0":
        raise PermissionError("private selection bridge schema differs")

    public_subjects: list[dict[str, Any]] = []
    state_subjects: dict[str, Any] = {}
    common_schema: list[str] | None = None
    for row in candidate["recipient_cohort"]["subjects"]:
        subject = row["subject_id"]
        bridge_subject = bridge.get("subjects", {}).get(subject)
        if not isinstance(bridge_subject, dict):
            raise ProtocolRefusal("ADT_SELECTION_BRIDGE_SUBJECT_MISSING")
        selected_ids = bridge_subject.get("selected_ids")
        if not isinstance(selected_ids, list) or len(selected_ids) != CELL_BUDGET:
            raise ProtocolRefusal("ADT_SELECTION_BRIDGE_AXIS_INVALID")
        if _axis_sha256(subject, selected_ids) != bridge_subject.get(
            "selected_axis_sha256"
        ):
            raise ProtocolRefusal("ADT_SELECTION_BRIDGE_AXIS_HASH_MISMATCH")
        expected_bytes = int(row["adt_csv_gz"]["bytes"])
        path, file_sha256 = _fetch_designated_file(
            _designated_url(candidate, row, "adt_csv_gz"),
            expected_bytes,
            scratch_dir,
            ".csv.gz",
        )
        try:
            reduced = _read_adt_csv(path, subject, selected_ids, candidate_targets)
        finally:
            path.unlink(missing_ok=True)
        available = reduced["available_adt_targets"]
        common_schema = _ordered_schema_intersection(common_schema, available)
        state_subjects[subject] = {
            "selected_axis_sha256": bridge_subject["selected_axis_sha256"],
            "states": {
                marker: values.astype(int).tolist()
                for marker, values in reduced["states"].items()
            },
        }
        public_subjects.append(
            {
                "subject_id": subject,
                "gsm": row["gsm"],
                "batch": row["batch"],
                "adt_csv_name": row["adt_csv_gz"]["name"],
                "adt_csv_bytes": expected_bytes,
                "adt_csv_sha256": file_sha256,
                "selected_axis_sha256": bridge_subject["selected_axis_sha256"],
                "column_margins": reduced["column_margins"],
                "adt_marker_support": reduced["marker_support"],
                "adt_variation_qc": reduced["variation_qc"],
                "adt_state_sha256": {
                    marker: _array_sha256(values)
                    for marker, values in reduced["states"].items()
                },
                "access_audit": reduced["audit"],
            }
        )
    common_schema = _require_common_schema(
        common_schema, "ADT_COMMON_SCHEMA_HAS_FEWER_THAN_NINE_CANDIDATES"
    )
    for record in public_subjects:
        for key in (
            "column_margins",
            "adt_marker_support",
            "adt_variation_qc",
            "adt_state_sha256",
        ):
            record[key] = {marker: record[key][marker] for marker in common_schema}
    for record in state_subjects.values():
        record["states"] = {
            marker: record["states"][marker] for marker in common_schema
        }
    states = {
        "schema": "gse309593-private-adt-states/1.0",
        "available_adt_targets": common_schema,
        "subjects": state_subjects,
    }
    _write_json_x(adt_states_path, states)
    return {
        "schema": "gse309593-independent-study-adt-stage/1.0",
        "status": "ADT_STAGE_FROZEN_WITHOUT_RNA_STATE_ACCESS",
        "created_at_utc": _timestamp(),
        "rna_stage_commit": rna_commit,
        "rna_stage_sha256": _sha256(DEFAULT_RNA),
        "source_model_sha256": rna["source_model_sha256"],
        "recipient_subjects": [row["subject_id"] for row in public_subjects],
        "available_adt_targets": common_schema,
        "subjects": public_subjects,
        "private_artifacts": {
            "adt_states_sha256": _sha256(adt_states_path),
            "adt_states_bytes": adt_states_path.stat().st_size,
            "paths_serialized": 0,
        },
        "access_boundary": {
            "selection_bridge_read": True,
            "rna_state_artifact_path_received": False,
            "rna_state_files_opened": 0,
            "rna_state_values_read": 0,
            "all_512_rna_selected_identifiers_required_exactly_once": True,
            "fallback_or_resampling_permitted": False,
        },
    }


def _final_panel(
    candidate: dict[str, Any],
    source_model: dict[str, Any],
    available_rna: list[str],
    available_adt: list[str],
) -> list[dict[str, Any]]:
    source_by_symbol = {
        row["rna_symbol"]: (index, row["adt_target"])
        for index, row in enumerate(_source_marker_mappings(source_model))
    }
    rna_set = set(available_rna)
    adt_set = set(available_adt)
    panel = []
    for row in candidate["marker_panel"]["ordered_candidates"]:
        symbol = row["rna_symbol"]
        target = row["adt_target"]
        if (
            symbol not in source_by_symbol
            or symbol not in rna_set
            or target not in adt_set
        ):
            continue
        source_index, source_target = source_by_symbol[symbol]
        panel.append(
            {
                "rna_symbol": symbol,
                "recipient_adt_target": target,
                "source_adt_target": source_target,
                "source_index": source_index,
            }
        )
    if len(panel) < MINIMUM_COGNATES:
        raise ProtocolRefusal("FINAL_PANEL_HAS_FEWER_THAN_NINE_COGNATES")
    return panel


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


def _predicted_tables(
    source_model: dict[str, Any],
    panel: list[dict[str, Any]],
    rows: np.ndarray,
    columns: np.ndarray,
) -> dict[str, np.ndarray]:
    size = len(panel)
    source_indices = [int(row["source_index"]) for row in panel]
    output: dict[str, np.ndarray] = {}
    for method in REQUIRED_METHODS:
        record = source_model["methods"][method]
        alpha = _transport_multiplier(record)
        prediction = np.empty((size, size, 2, 2), dtype=float)
        if method == "matched_deviance_residual":
            field = np.asarray(record["pooled_coordinate"], dtype=float)
            for row_index, source_row in enumerate(source_indices):
                for column_index, source_column in enumerate(source_indices):
                    coordinate = (
                        alpha
                        * float(field[source_row, source_column])
                        * math.sqrt(CELL_BUDGET)
                    )
                    prediction[row_index, column_index] = _residual_table(
                        coordinate,
                        rows[row_index, column_index],
                        columns[row_index, column_index],
                    )
        else:
            field = np.asarray(record["population_log_odds"], dtype=float)
            for row_index, source_row in enumerate(source_indices):
                for column_index, source_column in enumerate(source_indices):
                    prediction[row_index, column_index] = (
                        expected_binary_table_from_log_odds(
                            alpha * float(field[source_row, source_column]),
                            rows[row_index, column_index],
                            columns[row_index, column_index],
                        )
                    )
        output[method] = prediction
    return output


def _subject_records(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = payload.get("subjects")
    if not isinstance(records, list):
        raise PermissionError("stage subject records are absent")
    mapped = {row.get("subject_id"): row for row in records}
    if len(mapped) != len(records) or None in mapped:
        raise PermissionError("stage subject records are duplicated")
    return mapped


def _require_sha256(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PermissionError(f"{label} is not a SHA-256 digest")


def _validate_rna_stage_payload(payload: dict[str, Any]) -> None:
    expected_keys = {
        "schema",
        "status",
        "created_at_utc",
        "source_authorization_commit",
        "source_authorization_sha256",
        "source_model_sha256",
        "recipient_subjects",
        "available_rna_symbols",
        "subjects",
        "private_artifacts",
        "access_boundary",
        "stage",
        "runtime_environment",
        "attempt_tag",
        "attempt_commit",
    }
    if set(payload) != expected_keys or payload.get("schema") != (
        "gse309593-independent-study-rna-stage/1.0"
    ):
        raise PermissionError("RNA stage payload fields differ")
    authorization, model, commit = _require_source_authorization()
    candidate = _candidate()
    expected_subjects = [
        row["subject_id"] for row in candidate["recipient_cohort"]["subjects"]
    ]
    supported = _source_supported_candidates(candidate, model)
    allowed_symbols = [row["rna_symbol"] for row in supported]
    symbols = payload.get("available_rna_symbols")
    if (
        payload.get("source_authorization_commit") != commit
        or payload.get("source_authorization_sha256")
        != _sha256(DEFAULT_SOURCE_AUTHORIZATION)
        or payload.get("source_model_sha256") != authorization["source_model_sha256"]
        or payload.get("recipient_subjects") != expected_subjects
        or not isinstance(symbols, list)
        or len(symbols) < MINIMUM_COGNATES
        or symbols != [value for value in allowed_symbols if value in set(symbols)]
    ):
        raise PermissionError("RNA stage source, subject, or feature axis differs")
    candidate_by_subject = {
        row["subject_id"]: row for row in candidate["recipient_cohort"]["subjects"]
    }
    records = payload.get("subjects")
    if not isinstance(records, list) or [row.get("subject_id") for row in records] != (
        expected_subjects
    ):
        raise PermissionError("RNA stage subject records differ")
    for record in records:
        expected = candidate_by_subject[record["subject_id"]]
        if set(record) != {
            "subject_id",
            "gsm",
            "batch",
            "rna_h5_name",
            "rna_h5_bytes",
            "rna_h5_sha256",
            "selected_axis_sha256",
            "row_margins",
            "rna_state_sha256",
            "access_audit",
        } or any(
            record[key] != value
            for key, value in {
                "gsm": expected["gsm"],
                "batch": expected["batch"],
                "rna_h5_name": expected["rna_h5"]["name"],
                "rna_h5_bytes": expected["rna_h5"]["bytes"],
            }.items()
        ):
            raise PermissionError("RNA stage subject metadata differs")
        _require_sha256(record["rna_h5_sha256"], "RNA file hash")
        _require_sha256(record["selected_axis_sha256"], "RNA selected-axis hash")
        if set(record["row_margins"]) != set(symbols) or set(
            record["rna_state_sha256"]
        ) != set(symbols):
            raise PermissionError("RNA stage marker margins or hashes differ")
        for symbol in symbols:
            margin = record["row_margins"][symbol]
            if (
                not isinstance(margin, list)
                or len(margin) != 2
                or any(not isinstance(value, int) or value < 0 for value in margin)
                or sum(margin) != CELL_BUDGET
            ):
                raise PermissionError("RNA stage margin differs")
            _require_sha256(record["rna_state_sha256"][symbol], "RNA state hash")
        audit = record["access_audit"]
        if (
            set(audit)
            != {
                "format",
                "matrix_path",
                "symbol_axis_path",
                "matrix_shape",
                "matrix_storage_entries",
                "sparse_structure_validated",
                "identifier_values_read",
                "qc_eligible_identifiers",
                "rna_qc_thresholds",
                "selected_identifiers",
                "rna_qc_numeric_values_decoded",
                "selected_marker_numeric_values_decoded",
                "adt_files_opened",
                "adt_values_read",
            }
            or audit["rna_qc_thresholds"]
            != {
                "minimum_detected_genes": MINIMUM_DETECTED_GENES,
                "maximum_mitochondrial_fraction": MAXIMUM_MITOCHONDRIAL_FRACTION,
                "maximum_rna_umis": MAXIMUM_RNA_UMIS,
            }
            or audit["qc_eligible_identifiers"] < CELL_BUDGET
            or not isinstance(audit["matrix_shape"], list)
            or len(audit["matrix_shape"]) != 2
            or any(
                not isinstance(value, int) or value <= 0
                for value in audit["matrix_shape"]
            )
            or not isinstance(audit["matrix_storage_entries"], int)
            or audit["matrix_storage_entries"] < 0
            or not isinstance(audit["sparse_structure_validated"], bool)
            or any(
                audit[key] != value
                for key, value in {
                    "selected_identifiers": CELL_BUDGET,
                    "adt_files_opened": 0,
                    "adt_values_read": 0,
                }.items()
            )
        ):
            raise PermissionError("RNA stage access or QC audit differs")
    private = payload.get("private_artifacts")
    if (
        set(private or {})
        != {
            "selection_bridge_sha256",
            "selection_bridge_bytes",
            "rna_states_sha256",
            "rna_states_bytes",
            "paths_serialized",
        }
        or private["paths_serialized"] != 0
    ):
        raise PermissionError("RNA private-artifact certificate differs")
    _require_sha256(private["selection_bridge_sha256"], "selection bridge hash")
    _require_sha256(private["rna_states_sha256"], "RNA private-state hash")
    if payload.get("access_boundary") != {
        "adt_files_requested": 0,
        "adt_files_opened": 0,
        "adt_numeric_values_read": 0,
        "cell_selection_used_rna_identifiers_only": True,
        "adt_identifier_fallback_or_resampling_permitted": False,
    }:
        raise PermissionError("RNA access boundary differs")


def _validate_adt_stage_payload(payload: dict[str, Any]) -> None:
    expected_keys = {
        "schema",
        "status",
        "created_at_utc",
        "rna_stage_commit",
        "rna_stage_sha256",
        "source_model_sha256",
        "recipient_subjects",
        "available_adt_targets",
        "subjects",
        "private_artifacts",
        "access_boundary",
        "stage",
        "runtime_environment",
        "attempt_tag",
        "attempt_commit",
    }
    if set(payload) != expected_keys or payload.get("schema") != (
        "gse309593-independent-study-adt-stage/1.0"
    ):
        raise PermissionError("ADT stage payload fields differ")
    rna, rna_commit = _require_rna_stage()
    candidate = _candidate()
    expected_subjects = [
        row["subject_id"] for row in candidate["recipient_cohort"]["subjects"]
    ]
    targets = payload.get("available_adt_targets")
    candidate_targets = [
        row["adt_target"] for row in candidate["marker_panel"]["ordered_candidates"]
    ]
    if (
        payload.get("rna_stage_commit") != rna_commit
        or payload.get("rna_stage_sha256") != _sha256(DEFAULT_RNA)
        or payload.get("source_model_sha256") != rna["source_model_sha256"]
        or payload.get("recipient_subjects") != expected_subjects
        or not isinstance(targets, list)
        or len(targets) < MINIMUM_COGNATES
        or targets != [value for value in candidate_targets if value in set(targets)]
    ):
        raise PermissionError("ADT stage lineage, subject, or feature axis differs")
    candidate_by_subject = {
        row["subject_id"]: row for row in candidate["recipient_cohort"]["subjects"]
    }
    rna_records = _subject_records(rna)
    records = payload.get("subjects")
    if not isinstance(records, list) or [row.get("subject_id") for row in records] != (
        expected_subjects
    ):
        raise PermissionError("ADT stage subject records differ")
    for record in records:
        expected = candidate_by_subject[record["subject_id"]]
        if set(record) != {
            "subject_id",
            "gsm",
            "batch",
            "adt_csv_name",
            "adt_csv_bytes",
            "adt_csv_sha256",
            "selected_axis_sha256",
            "column_margins",
            "adt_marker_support",
            "adt_variation_qc",
            "adt_state_sha256",
            "access_audit",
        } or any(
            record[key] != value
            for key, value in {
                "gsm": expected["gsm"],
                "batch": expected["batch"],
                "adt_csv_name": expected["adt_csv_gz"]["name"],
                "adt_csv_bytes": expected["adt_csv_gz"]["bytes"],
            }.items()
        ):
            raise PermissionError("ADT stage subject metadata differs")
        if (
            record["selected_axis_sha256"]
            != rna_records[record["subject_id"]]["selected_axis_sha256"]
        ):
            raise PermissionError("ADT selected axis differs from the RNA freeze")
        _require_sha256(record["adt_csv_sha256"], "ADT file hash")
        _require_sha256(record["selected_axis_sha256"], "ADT selected-axis hash")
        if any(
            set(record[key]) != set(targets)
            for key in (
                "column_margins",
                "adt_marker_support",
                "adt_variation_qc",
                "adt_state_sha256",
            )
        ):
            raise PermissionError("ADT marker certificates differ")
        for target in targets:
            supported = record["adt_marker_support"][target]
            if not isinstance(supported, bool) or record["column_margins"][target] != (
                [256, 256] if supported else [512, 0]
            ):
                raise PermissionError("ADT fixed margin differs")
            variation = record["adt_variation_qc"][target]
            boundary_tie_cells = variation.get("boundary_tie_cells", -1)
            maximum_tie_cells = variation.get("maximum_boundary_tie_cells")
            if (
                set(variation)
                != {
                    "distinct_raw_values",
                    "lower_boundary_value",
                    "upper_boundary_value",
                    "boundary_tie_cells",
                    "maximum_boundary_tie_cells",
                    "passes",
                }
                or not isinstance(variation["distinct_raw_values"], int)
                or variation["distinct_raw_values"] < 1
                or not isinstance(variation["lower_boundary_value"], int)
                or not isinstance(variation["upper_boundary_value"], int)
                or variation["lower_boundary_value"] > variation["upper_boundary_value"]
                or maximum_tie_cells
                != int(CELL_BUDGET * MAXIMUM_ADT_BOUNDARY_TIE_FRACTION)
                or not isinstance(boundary_tie_cells, int)
                or not 0 <= boundary_tie_cells <= CELL_BUDGET
                or variation.get("passes") is not supported
                or (boundary_tie_cells <= maximum_tie_cells) != supported
                or (
                    variation["lower_boundary_value"]
                    < variation["upper_boundary_value"]
                    and boundary_tie_cells != 0
                )
                or (
                    variation["lower_boundary_value"]
                    == variation["upper_boundary_value"]
                    and boundary_tie_cells < 2
                )
            ):
                raise PermissionError("ADT variation certificate differs")
            _require_sha256(record["adt_state_sha256"][target], "ADT state hash")
        audit = record["access_audit"]
        if set(audit) != {
            "orientation",
            "selected_identifier_values_read",
            "selected_adt_numeric_values_read",
            "rna_state_files_opened",
            "rna_state_values_read",
        } or any(
            audit[key] != value
            for key, value in {
                "selected_identifier_values_read": CELL_BUDGET,
                "rna_state_files_opened": 0,
                "rna_state_values_read": 0,
            }.items()
        ):
            raise PermissionError("ADT access audit differs")
    private = payload.get("private_artifacts")
    if (
        set(private or {})
        != {
            "adt_states_sha256",
            "adt_states_bytes",
            "paths_serialized",
        }
        or private["paths_serialized"] != 0
    ):
        raise PermissionError("ADT private-artifact certificate differs")
    _require_sha256(private["adt_states_sha256"], "ADT private-state hash")
    if payload.get("access_boundary") != {
        "selection_bridge_read": True,
        "rna_state_artifact_path_received": False,
        "rna_state_files_opened": 0,
        "rna_state_values_read": 0,
        "all_512_rna_selected_identifiers_required_exactly_once": True,
        "fallback_or_resampling_permitted": False,
    }:
        raise PermissionError("ADT access boundary differs")


def _validate_prediction_stage_payload(payload: dict[str, Any]) -> None:
    expected_keys = {
        "schema",
        "status",
        "created_at_utc",
        "source_authorization_commit",
        "source_authorization_sha256",
        "source_model_sha256",
        "rna_stage_commit",
        "rna_stage_sha256",
        "adt_stage_commit",
        "adt_stage_sha256",
        "panel",
        "ordered_rna_to_adt_pairs",
        "panel_selection_inputs",
        "target_numeric_values_used_to_choose_panel",
        "available_methods",
        "classical_head_to_head_ready",
        "eligible_subjects",
        "ineligible_subjects",
        "all_seven_batches_represented",
        "predictions",
        "recipient_joint_tables_formed",
        "recipient_rna_and_adt_state_artifacts_opened_together",
        "stage",
        "runtime_environment",
        "attempt_tag",
        "attempt_commit",
    }
    if set(payload) != expected_keys or payload.get("schema") != (
        "gse309593-independent-study-predictions/1.0"
    ):
        raise PermissionError("prediction stage payload fields differ")
    authorization, model, source_commit = _require_source_authorization()
    rna, rna_commit = _require_rna_stage()
    adt, adt_commit = _require_adt_stage()
    candidate = _candidate()
    panel = _final_panel(
        candidate,
        model,
        list(rna["available_rna_symbols"]),
        list(adt["available_adt_targets"]),
    )
    if any(
        (
            payload.get("source_authorization_commit") != source_commit,
            payload.get("source_authorization_sha256")
            != _sha256(DEFAULT_SOURCE_AUTHORIZATION),
            payload.get("source_model_sha256") != authorization["source_model_sha256"],
            payload.get("rna_stage_commit") != rna_commit,
            payload.get("rna_stage_sha256") != _sha256(DEFAULT_RNA),
            payload.get("adt_stage_commit") != adt_commit,
            payload.get("adt_stage_sha256") != _sha256(DEFAULT_ADT),
            payload.get("panel") != panel,
            payload.get("ordered_rna_to_adt_pairs") != len(panel) ** 2,
            payload.get("panel_selection_inputs")
            != "feature schemas and frozen source axis only",
            payload.get("available_methods") != list(REQUIRED_METHODS),
            payload.get("classical_head_to_head_ready") is not True,
            payload.get("target_numeric_values_used_to_choose_panel") != 0,
            payload.get("recipient_joint_tables_formed") != 0,
            payload.get("recipient_rna_and_adt_state_artifacts_opened_together")
            is not False,
        )
    ):
        raise PermissionError("prediction lineage or frozen contract differs")
    candidate_subjects = [
        row["subject_id"] for row in candidate["recipient_cohort"]["subjects"]
    ]
    eligible = payload.get("eligible_subjects")
    ineligible = payload.get("ineligible_subjects")
    if (
        not isinstance(eligible, list)
        or not isinstance(ineligible, dict)
        or len(eligible) < MINIMUM_ELIGIBLE_SUBJECTS
        or len(eligible) != len(set(eligible))
        or set(eligible) | set(ineligible) != set(candidate_subjects)
        or set(eligible) & set(ineligible)
        or any(
            value != "FEWER_THAN_64_INFORMATIVE_MARGIN_PAIRS"
            for value in ineligible.values()
        )
    ):
        raise PermissionError("prediction eligibility axis differs")
    batches = {
        row["subject_id"]: row["batch"]
        for row in candidate["recipient_cohort"]["subjects"]
    }
    if {batches[subject] for subject in eligible} != set(
        candidate["recipient_cohort"]["expected_batches"]
    ) or payload.get("all_seven_batches_represented") is not True:
        raise PermissionError("prediction does not retain all seven batches")
    records = payload.get("predictions")
    if not isinstance(records, list) or [row.get("subject_id") for row in records] != (
        candidate_subjects
    ):
        raise PermissionError("prediction subject records differ")
    size = len(panel)
    rna_records = _subject_records(rna)
    adt_records = _subject_records(adt)
    derived_eligible: list[str] = []
    derived_ineligible: dict[str, str] = {}
    for record in records:
        if (
            set(record)
            != {
                "subject_id",
                "batch",
                "selected_axis_sha256",
                "informative_pairs_from_margins",
                "row_margins",
                "column_margins",
                "predicted_tables",
                "prediction_sha256",
            }
            or record["batch"] != batches[record["subject_id"]]
        ):
            raise PermissionError("prediction subject metadata differs")
        rows = np.asarray(record["row_margins"], dtype=np.int64)
        columns = np.asarray(record["column_margins"], dtype=np.int64)
        expected_rows = np.empty((size, size, 2), dtype=np.int64)
        expected_columns = np.empty_like(expected_rows)
        for row_index, row_marker in enumerate(panel):
            for column_index, column_marker in enumerate(panel):
                expected_rows[row_index, column_index] = rna_records[
                    record["subject_id"]
                ]["row_margins"][row_marker["rna_symbol"]]
                expected_columns[row_index, column_index] = adt_records[
                    record["subject_id"]
                ]["column_margins"][column_marker["recipient_adt_target"]]
        informative = np.all(rows > 0, axis=-1) & np.all(columns > 0, axis=-1)
        if (
            rows.shape != (size, size, 2)
            or columns.shape != (size, size, 2)
            or np.any(rows < 0)
            or np.any(columns < 0)
            or not np.all(rows.sum(axis=-1) == CELL_BUDGET)
            or not np.all(columns.sum(axis=-1) == CELL_BUDGET)
            or not np.array_equal(rows, expected_rows)
            or not np.array_equal(columns, expected_columns)
            or record["selected_axis_sha256"]
            != rna_records[record["subject_id"]]["selected_axis_sha256"]
            or record["selected_axis_sha256"]
            != adt_records[record["subject_id"]]["selected_axis_sha256"]
            or record["informative_pairs_from_margins"]
            != int(np.count_nonzero(informative))
            or set(record["predicted_tables"]) != set(REQUIRED_METHODS)
            or set(record["prediction_sha256"]) != set(REQUIRED_METHODS)
        ):
            raise PermissionError("prediction margins or method axis differs")
        if record["informative_pairs_from_margins"] >= MINIMUM_INFORMATIVE_PAIRS:
            derived_eligible.append(record["subject_id"])
        else:
            derived_ineligible[record["subject_id"]] = (
                "FEWER_THAN_64_INFORMATIVE_MARGIN_PAIRS"
            )
        for method in REQUIRED_METHODS:
            estimate = np.asarray(record["predicted_tables"][method], dtype=float)
            if (
                estimate.shape != (size, size, 2, 2)
                or not np.isfinite(estimate).all()
                or np.any(estimate < 0)
                or not np.allclose(estimate.sum(axis=-1), rows)
                or not np.allclose(estimate.sum(axis=-2), columns)
                or _array_sha256(estimate) != record["prediction_sha256"][method]
            ):
                raise PermissionError("prediction table certificate differs")
    if eligible != derived_eligible or ineligible != derived_ineligible:
        raise PermissionError("prediction eligibility differs from frozen margins")


def _validate_score_stage_payload(payload: dict[str, Any]) -> None:
    expected_keys = {
        "schema",
        "status",
        "created_at_utc",
        "prediction_tag",
        "prediction_commit",
        "prediction_sha256",
        "score_authorization_commit",
        "score_authorization_sha256",
        "source_model_sha256",
        "available_methods",
        "classical_head_to_head_ready",
        "eligible_subjects",
        "eligible_subject_count",
        "all_seven_batches_represented",
        "panel",
        "ordered_rna_to_adt_pairs",
        "losses",
        "comparisons",
        "transfer_confirmation_pass",
        "gain_over_both_classical_interactions",
        "samples",
        "outcome_failure_rule",
        "claim_scope",
        "stage",
        "runtime_environment",
        "attempt_tag",
        "attempt_commit",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema") != "gse309593-independent-study-confirmation/1.0"
        or payload.get("status") not in COMPLETED_SCORE_STATUSES
    ):
        raise PermissionError("score stage payload fields differ")
    prediction, prediction_commit = _require_prediction_stage()
    authorization, authorization_commit = _require_score_authorization()
    eligible = list(prediction["eligible_subjects"])
    candidate = _candidate()
    batches = {
        row["subject_id"]: row["batch"]
        for row in candidate["recipient_cohort"]["subjects"]
    }
    if any(
        (
            payload.get("prediction_tag") != PREDICTION_TAG,
            payload.get("prediction_commit") != prediction_commit,
            payload.get("prediction_sha256") != _sha256(DEFAULT_PREDICTION),
            payload.get("score_authorization_commit") != authorization_commit,
            payload.get("score_authorization_sha256")
            != _sha256(DEFAULT_SCORE_AUTHORIZATION),
            payload.get("source_model_sha256") != prediction["source_model_sha256"],
            payload.get("available_methods") != list(REQUIRED_METHODS),
            payload.get("classical_head_to_head_ready") is not True,
            payload.get("eligible_subjects") != eligible,
            payload.get("eligible_subject_count") != len(eligible),
            payload.get("all_seven_batches_represented") is not True,
            payload.get("panel") != prediction["panel"],
            payload.get("ordered_rna_to_adt_pairs")
            != prediction["ordered_rna_to_adt_pairs"],
            payload.get("claim_scope") != CLAIM_SCOPE,
            authorization.get("outcome_access_authorized") is not True,
        )
    ):
        raise PermissionError("score stage lineage or scope differs")
    if {batches[subject] for subject in eligible} != set(
        candidate["recipient_cohort"]["expected_batches"]
    ):
        raise PermissionError("score stage omits a processing batch")

    losses = payload.get("losses")
    if not isinstance(losses, dict) or list(losses) != list(REQUIRED_METHODS):
        raise PermissionError("score loss method axis differs")
    arrays: dict[str, np.ndarray] = {}
    for method in REQUIRED_METHODS:
        values = losses[method]
        if not isinstance(values, dict) or list(values) != eligible:
            raise PermissionError("score loss subject axis differs")
        array = np.asarray([values[subject] for subject in eligible], dtype=float)
        if not np.isfinite(array).all() or np.any(array < 0.0):
            raise PermissionError("score losses are invalid")
        arrays[method] = array

    expected_comparisons = {
        "primary_vs_matched_deviance_residual": _held_comparison(
            eligible,
            batches,
            arrays["primary"],
            arrays["matched_deviance_residual"],
            classical=False,
            seed_offset=0,
        ),
        "primary_vs_destroyed_link": _held_comparison(
            eligible,
            batches,
            arrays["primary"],
            arrays["destroyed_link"],
            classical=False,
            seed_offset=1,
        ),
        "primary_vs_common_effect_cmle": _held_comparison(
            eligible,
            batches,
            arrays["primary"],
            arrays["common_effect_cmle"],
            classical=True,
            seed_offset=2,
        ),
        "primary_vs_pooled_saturated_poisson": _held_comparison(
            eligible,
            batches,
            arrays["primary"],
            arrays["pooled_saturated_poisson"],
            classical=True,
            seed_offset=3,
        ),
    }
    if payload.get("comparisons") != expected_comparisons:
        raise PermissionError("score comparisons do not reproduce from frozen losses")
    transfer_pass = (
        expected_comparisons["primary_vs_matched_deviance_residual"]["passes"]
        and expected_comparisons["primary_vs_destroyed_link"]["passes"]
    )
    classical_gain = (
        expected_comparisons["primary_vs_common_effect_cmle"]["passes"]
        and expected_comparisons["primary_vs_pooled_saturated_poisson"]["passes"]
    )
    expected_status = (
        "CONFIRMATION_PASS_WITH_GAIN_OVER_BOTH_CLASSICAL_INTERACTIONS"
        if transfer_pass and classical_gain
        else (
            "CONFIRMATION_PASS_WITHOUT_GAIN_OVER_BOTH_CLASSICAL_INTERACTIONS"
            if transfer_pass
            else "COMPLETED_CONFIRMATION_FAIL"
        )
    )
    if (
        payload.get("transfer_confirmation_pass") is not transfer_pass
        or payload.get("gain_over_both_classical_interactions") is not classical_gain
        or payload.get("status") != expected_status
        or payload.get("outcome_failure_rule")
        != (
            "A supported result that misses a criterion is a completed negative, "
            "not a QC refusal."
        )
    ):
        raise PermissionError("score decision does not reproduce")

    prediction_by_subject = {
        row["subject_id"]: row for row in prediction["predictions"]
    }
    samples = payload.get("samples")
    if not isinstance(samples, list) or [row.get("subject_id") for row in samples] != (
        eligible
    ):
        raise PermissionError("score sample axis differs")
    for row in samples:
        subject = row["subject_id"]
        if (
            set(row)
            != {
                "subject_id",
                "batch",
                "informative_pairs",
                "truth_table_sha256",
                "losses",
            }
            or row["batch"] != batches[subject]
            or row["informative_pairs"]
            != prediction_by_subject[subject]["informative_pairs_from_margins"]
            or row["losses"]
            != {method: losses[method][subject] for method in REQUIRED_METHODS}
        ):
            raise PermissionError("score sample certificate differs")
        _require_sha256(row["truth_table_sha256"], "truth-table hash")


def _validate_stage_payload(stage: str, payload: dict[str, Any]) -> None:
    validators = {
        "rna": _validate_rna_stage_payload,
        "adt": _validate_adt_stage_payload,
        "prediction": _validate_prediction_stage_payload,
        "score": _validate_score_stage_payload,
    }
    if stage not in validators:
        raise PermissionError(f"no semantic validator exists for prior stage {stage}")
    validators[stage](payload)


def _prediction_stage_body() -> dict[str, Any]:
    source_authorization, source_model, source_commit = _require_source_authorization()
    rna, rna_commit = _require_rna_stage()
    adt, adt_commit = _require_adt_stage()
    candidate = _candidate()
    panel = _final_panel(
        candidate,
        source_model,
        list(rna["available_rna_symbols"]),
        list(adt["available_adt_targets"]),
    )
    rna_records = _subject_records(rna)
    adt_records = _subject_records(adt)
    expected_subjects = [
        row["subject_id"] for row in candidate["recipient_cohort"]["subjects"]
    ]
    if (
        list(rna["recipient_subjects"]) != expected_subjects
        or list(adt["recipient_subjects"]) != expected_subjects
    ):
        raise ProtocolRefusal("RECIPIENT_SUBJECT_AXIS_DIFFERS")

    predictions: list[dict[str, Any]] = []
    available_methods = list(REQUIRED_METHODS)
    eligible: list[str] = []
    ineligible: dict[str, str] = {}
    eligible_batches: set[str] = set()
    for subject_row in candidate["recipient_cohort"]["subjects"]:
        subject = subject_row["subject_id"]
        rna_row = rna_records[subject]
        adt_row = adt_records[subject]
        if rna_row["selected_axis_sha256"] != adt_row["selected_axis_sha256"]:
            raise ProtocolRefusal("RNA_ADT_SELECTED_AXIS_HASH_MISMATCH")
        size = len(panel)
        rows = np.empty((size, size, 2), dtype=np.int64)
        columns = np.empty_like(rows)
        for row_index, row_marker in enumerate(panel):
            row_margin = np.asarray(
                rna_row["row_margins"][row_marker["rna_symbol"]], dtype=np.int64
            )
            for column_index, column_marker in enumerate(panel):
                rows[row_index, column_index] = row_margin
                columns[row_index, column_index] = np.asarray(
                    adt_row["column_margins"][column_marker["recipient_adt_target"]],
                    dtype=np.int64,
                )
        informative = int(
            np.count_nonzero(np.all(rows > 0, axis=-1) & np.all(columns > 0, axis=-1))
        )
        estimates = _predicted_tables(source_model, panel, rows, columns)
        if informative >= MINIMUM_INFORMATIVE_PAIRS:
            eligible.append(subject)
            eligible_batches.add(subject_row["batch"])
        else:
            ineligible[subject] = "FEWER_THAN_64_INFORMATIVE_MARGIN_PAIRS"
        predictions.append(
            {
                "subject_id": subject,
                "batch": subject_row["batch"],
                "selected_axis_sha256": rna_row["selected_axis_sha256"],
                "informative_pairs_from_margins": informative,
                "row_margins": rows.tolist(),
                "column_margins": columns.tolist(),
                "predicted_tables": {
                    method: values.tolist() for method, values in estimates.items()
                },
                "prediction_sha256": {
                    method: _array_sha256(values)
                    for method, values in estimates.items()
                },
            }
        )
    required_batches = set(candidate["recipient_cohort"]["expected_batches"].keys())
    if len(eligible) < MINIMUM_ELIGIBLE_SUBJECTS:
        raise ProtocolRefusal("FEWER_THAN_18_ELIGIBLE_RECIPIENT_SUBJECTS")
    if eligible_batches != required_batches:
        raise ProtocolRefusal("ELIGIBLE_RECIPIENTS_DO_NOT_COVER_ALL_SEVEN_BATCHES")
    return {
        "schema": "gse309593-independent-study-predictions/1.0",
        "status": "PREDICTIONS_FROZEN_BEFORE_RECIPIENT_PAIRING",
        "created_at_utc": _timestamp(),
        "source_authorization_commit": source_commit,
        "source_authorization_sha256": _sha256(DEFAULT_SOURCE_AUTHORIZATION),
        "source_model_sha256": source_authorization["source_model_sha256"],
        "rna_stage_commit": rna_commit,
        "rna_stage_sha256": _sha256(DEFAULT_RNA),
        "adt_stage_commit": adt_commit,
        "adt_stage_sha256": _sha256(DEFAULT_ADT),
        "panel": panel,
        "ordered_rna_to_adt_pairs": len(panel) ** 2,
        "panel_selection_inputs": "feature schemas and frozen source axis only",
        "target_numeric_values_used_to_choose_panel": 0,
        "available_methods": available_methods,
        "classical_head_to_head_ready": True,
        "eligible_subjects": eligible,
        "ineligible_subjects": ineligible,
        "all_seven_batches_represented": True,
        "predictions": predictions,
        "recipient_joint_tables_formed": 0,
        "recipient_rna_and_adt_state_artifacts_opened_together": False,
    }


def _private_artifact(
    path: Path, expected: dict[str, Any], prefix: str
) -> dict[str, Any]:
    path = _private_path(path)
    if (
        path.stat().st_size != expected[f"{prefix}_bytes"]
        or _sha256(path) != expected[f"{prefix}_sha256"]
    ):
        raise PermissionError(f"private {prefix} artifact differs from public freeze")
    return _read_json(path)


def _truth_tables(
    panel: list[dict[str, Any]],
    rna_subject: dict[str, Any],
    adt_subject: dict[str, Any],
) -> np.ndarray:
    size = len(panel)
    output = np.empty((size, size, 2, 2), dtype=np.int64)
    for row_index, row_marker in enumerate(panel):
        rna_state = np.asarray(
            rna_subject["states"][row_marker["rna_symbol"]], dtype=np.int64
        )
        if rna_state.shape != (CELL_BUDGET,) or np.any(
            (rna_state < 0) | (rna_state > 1)
        ):
            raise PermissionError("private RNA state vector is invalid")
        for column_index, column_marker in enumerate(panel):
            adt_state = np.asarray(
                adt_subject["states"][column_marker["recipient_adt_target"]],
                dtype=np.int64,
            )
            if adt_state.shape != (CELL_BUDGET,) or np.any(
                (adt_state < 0) | (adt_state > 1)
            ):
                raise PermissionError("private ADT state vector is invalid")
            output[row_index, column_index] = np.bincount(
                2 * rna_state + adt_state, minlength=4
            ).reshape(2, 2)
    return output


def _donor_bootstrap_interval(values: np.ndarray, seed_offset: int = 0) -> list[float]:
    differences = np.asarray(values, dtype=float)
    generator = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    means = np.empty(BOOTSTRAPS, dtype=float)
    for start in range(0, BOOTSTRAPS, 1000):
        stop = min(start + 1000, BOOTSTRAPS)
        indices = generator.integers(
            0, len(differences), size=(stop - start, len(differences))
        )
        means[start:stop] = differences[indices].mean(axis=1)
    return np.quantile(means, [0.025, 0.975], method="linear").tolist()


def _batch_bootstrap_interval(
    values: np.ndarray, batch_labels: list[str], seed_offset: int = 0
) -> list[float]:
    differences = np.asarray(values, dtype=float)
    axis = np.asarray(batch_labels, dtype=object)
    batches = np.asarray(sorted(set(batch_labels)), dtype=object)
    sums = np.asarray([differences[axis == batch].sum() for batch in batches])
    counts = np.asarray([np.count_nonzero(axis == batch) for batch in batches])
    generator = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    means = np.empty(BOOTSTRAPS, dtype=float)
    for start in range(0, BOOTSTRAPS, 1000):
        stop = min(start + 1000, BOOTSTRAPS)
        indices = generator.integers(0, len(batches), size=(stop - start, len(batches)))
        means[start:stop] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return np.quantile(means, [0.025, 0.975], method="linear").tolist()


def _batch_stratified_donor_bootstrap_interval(
    values: np.ndarray, batch_labels: list[str], seed_offset: int = 0
) -> list[float]:
    differences = np.asarray(values, dtype=float)
    axis = np.asarray(batch_labels, dtype=object)
    groups = [differences[axis == batch] for batch in sorted(set(batch_labels))]
    generator = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    means = np.empty(BOOTSTRAPS, dtype=float)
    for draw in range(BOOTSTRAPS):
        total = 0.0
        count = 0
        for group in groups:
            indices = generator.integers(0, len(group), size=len(group))
            total += float(group[indices].sum())
            count += len(group)
        means[draw] = total / count
    return np.quantile(means, [0.025, 0.975], method="linear").tolist()


def _leave_one_batch_jackknife_t(
    values: np.ndarray, batch_labels: list[str]
) -> dict[str, Any]:
    differences = np.asarray(values, dtype=float)
    axis = np.asarray(batch_labels, dtype=object)
    batches = sorted(set(batch_labels))
    estimates = np.asarray(
        [float(differences[axis != batch].mean()) for batch in batches]
    )
    center = float(estimates.mean())
    standard_error = math.sqrt(
        (len(batches) - 1) / len(batches) * float(np.sum((estimates - center) ** 2))
    )
    estimate = float(differences.mean())
    critical = 2.4469118511449692
    return {
        "degrees_of_freedom": 6,
        "t_0_975": critical,
        "standard_error": standard_error,
        "interval": [
            estimate - critical * standard_error,
            estimate + critical * standard_error,
        ],
    }


def _batch_sign_flip(
    batch_means: dict[str, float], batch_counts: dict[str, int], seed_offset: int = 0
) -> dict[str, Any]:
    keys = sorted(batch_means)
    if set(keys) != set(batch_counts) or any(batch_counts[key] <= 0 for key in keys):
        raise ValueError("batch means and counts differ")
    values = np.asarray([batch_means[key] for key in keys], dtype=float)
    weights = np.asarray([batch_counts[key] for key in keys], dtype=float)
    observed = float(np.average(values, weights=weights))
    if len(values) <= 20:
        draws = 1 << len(values)
        favorable = 0
        for mask in range(draws):
            signed = np.asarray(
                [
                    -value if mask & (1 << index) else value
                    for index, value in enumerate(values)
                ]
            )
            statistic = float(np.average(signed, weights=weights))
            favorable += statistic <= observed + 1e-15
        method = "exact"
    else:
        draws = 100_000
        generator = np.random.default_rng(BOOTSTRAP_SEED + 100 + seed_offset)
        signs = generator.choice((-1.0, 1.0), size=(draws, len(values)))
        statistics = (signs * values * weights).sum(axis=1) / weights.sum()
        favorable = int(np.count_nonzero(statistics <= observed))
        method = "monte_carlo"
    return {
        "method": method,
        "batch_units": len(values),
        "draws": draws,
        "observed_donor_equal_mean_difference": observed,
        "one_sided_p": favorable / draws,
    }


def _held_comparison(
    subjects: list[str],
    batches: dict[str, str],
    primary: np.ndarray,
    comparator: np.ndarray,
    *,
    classical: bool,
    seed_offset: int,
) -> dict[str, Any]:
    primary_values = np.asarray(primary, dtype=float)
    comparator_values = np.asarray(comparator, dtype=float)
    if (
        len(subjects) != len(primary_values)
        or len(subjects) != len(comparator_values)
        or len({batches[subject] for subject in subjects}) != 7
    ):
        raise ValueError("held comparison requires matched donors in all seven batches")
    difference = primary_values - comparator_values
    batch_labels = [batches[subject] for subject in subjects]
    primary_interval = _batch_stratified_donor_bootstrap_interval(
        difference, batch_labels, seed_offset
    )
    batch_interval = _batch_bootstrap_interval(
        difference, batch_labels, 25 + seed_offset
    )
    donor_interval = _donor_bootstrap_interval(difference, 50 + seed_offset)
    jackknife = _leave_one_batch_jackknife_t(difference, batch_labels)
    comparator_mean = float(comparator_values.mean())
    relative = (
        1.0 - float(primary_values.mean()) / comparator_mean
        if comparator_mean > 0.0
        else None
    )
    favorable = int(np.count_nonzero(difference < 0.0))
    required = math.ceil(0.8 * len(subjects))
    batch_axis = np.asarray(batch_labels, dtype=object)
    batch_means = {
        batch: float(difference[batch_axis == batch].mean())
        for batch in sorted(set(batch_axis))
    }
    batch_counts = {
        batch: int(np.count_nonzero(batch_axis == batch))
        for batch in sorted(set(batch_axis))
    }
    sign_flip = _batch_sign_flip(batch_means, batch_counts, seed_offset)
    leave_one_batch_out = {
        batch: float(difference[batch_axis != batch].mean())
        for batch in sorted(set(batch_axis))
    }
    if classical:
        checks = {
            "primary_point_loss_lower": float(difference.mean()) < 0.0,
            "batch_stratified_donor_bootstrap_upper_95_below_zero": (
                primary_interval[1] < 0.0
            ),
        }
    else:
        checks = {
            "relative_deviance_reduction_at_least_five_percent": (
                relative is not None and relative >= 0.05
            ),
            "batch_stratified_donor_bootstrap_upper_95_below_zero": (
                primary_interval[1] < 0.0
            ),
            "favorable_subject_fraction_at_least_0_8": favorable >= required,
            "every_batch_mean_difference_negative": all(
                value < 0.0 for value in batch_means.values()
            ),
            "batch_sign_flip_one_sided_p_at_most_0_025": (
                sign_flip["one_sided_p"] <= 0.025
            ),
            "every_leave_one_batch_out_mean_negative": all(
                value < 0.0 for value in leave_one_batch_out.values()
            ),
        }
    return {
        "classical_increment_rule": classical,
        "subjects": len(subjects),
        "primary_mean_loss": float(primary_values.mean()),
        "comparator_mean_loss": comparator_mean,
        "relative_deviance_reduction": relative,
        "mean_paired_difference": float(difference.mean()),
        "batch_stratified_donor_bootstrap_95_interval": primary_interval,
        "batch_block_bootstrap_sensitivity_95_interval": batch_interval,
        "unstratified_donor_bootstrap_sensitivity_95_interval": donor_interval,
        "leave_one_batch_jackknife_t_sensitivity": jackknife,
        "bootstrap_draws": BOOTSTRAPS,
        "favorable_subjects": favorable,
        "required_favorable_subjects": required,
        "batch_mean_differences": batch_means,
        "batch_subject_counts": batch_counts,
        "batch_sign_flip": sign_flip,
        "leave_one_batch_out_mean_differences": leave_one_batch_out,
        "checks": checks,
        "passes": all(checks.values()),
        "subject_differences": {
            subject: float(value) for subject, value in zip(subjects, difference)
        },
    }


def _score_stage_body(rna_states_path: Path, adt_states_path: Path) -> dict[str, Any]:
    _, source_model, _ = _require_source_authorization()
    rna, _ = _require_rna_stage()
    adt, _ = _require_adt_stage()
    prediction, prediction_commit = _require_prediction_stage()
    authorization, authorization_commit = _require_score_authorization()
    private_rna = _private_artifact(
        rna_states_path, rna["private_artifacts"], "rna_states"
    )
    private_adt = _private_artifact(
        adt_states_path, adt["private_artifacts"], "adt_states"
    )
    if private_rna.get("schema") != "gse309593-private-rna-states/1.0":
        raise PermissionError("private RNA state schema differs")
    if private_adt.get("schema") != "gse309593-private-adt-states/1.0":
        raise PermissionError("private ADT state schema differs")
    candidate = _candidate()
    panel = _final_panel(
        candidate,
        source_model,
        list(rna["available_rna_symbols"]),
        list(adt["available_adt_targets"]),
    )
    if prediction.get("panel") != panel:
        raise PermissionError(
            "prediction panel differs from frozen schema intersection"
        )
    prediction_by_subject = {
        row["subject_id"]: row for row in prediction["predictions"]
    }
    rna_public = _subject_records(rna)
    adt_public = _subject_records(adt)
    eligible = list(prediction["eligible_subjects"])
    available_methods = list(prediction.get("available_methods", []))
    if available_methods != list(REQUIRED_METHODS):
        raise PermissionError("prediction does not contain all five methods in order")
    if (
        prediction.get("classical_head_to_head_ready")
        is not source_model["numerical_certificate"]["classical_head_to_head_ready"]
    ):
        raise PermissionError("prediction classical readiness differs from source")
    batches = {
        row["subject_id"]: row["batch"]
        for row in candidate["recipient_cohort"]["subjects"]
    }
    losses = {
        method: np.empty(len(eligible), dtype=float) for method in available_methods
    }
    samples: list[dict[str, Any]] = []
    for subject_index, subject in enumerate(eligible):
        rna_subject = private_rna.get("subjects", {}).get(subject)
        adt_subject = private_adt.get("subjects", {}).get(subject)
        if not isinstance(rna_subject, dict) or not isinstance(adt_subject, dict):
            raise PermissionError("private subject state is absent")
        frozen = prediction_by_subject[subject]
        axis = frozen["selected_axis_sha256"]
        if not all(
            value == axis
            for value in (
                rna_subject.get("selected_axis_sha256"),
                adt_subject.get("selected_axis_sha256"),
                rna_public[subject]["selected_axis_sha256"],
                adt_public[subject]["selected_axis_sha256"],
            )
        ):
            raise PermissionError("private and public subject axes differ")
        for marker in panel:
            symbol = marker["rna_symbol"]
            target = marker["recipient_adt_target"]
            if (
                _array_sha256(np.asarray(rna_subject["states"][symbol], dtype=np.uint8))
                != rna_public[subject]["rna_state_sha256"][symbol]
            ):
                raise PermissionError("private RNA state hash differs")
            if (
                _array_sha256(np.asarray(adt_subject["states"][target], dtype=np.uint8))
                != adt_public[subject]["adt_state_sha256"][target]
            ):
                raise PermissionError("private ADT state hash differs")
            adt_values = np.asarray(adt_subject["states"][target], dtype=np.uint8)
            supported = adt_public[subject]["adt_marker_support"][target]
            if int(adt_values.sum()) != (CELL_BUDGET // 2 if supported else 0):
                raise PermissionError("private ADT support state differs")
        truth = _truth_tables(panel, rna_subject, adt_subject)
        rows = truth.sum(axis=-1)
        columns = truth.sum(axis=-2)
        if (
            rows.tolist() != frozen["row_margins"]
            or columns.tolist() != frozen["column_margins"]
        ):
            raise PermissionError("joint table margins differ from prediction freeze")
        informative = np.all(rows > 0, axis=-1) & np.all(columns > 0, axis=-1)
        if (
            int(np.count_nonzero(informative))
            != frozen["informative_pairs_from_margins"]
        ):
            raise PermissionError("informative-pair support differs from margin freeze")
        sample_losses = {}
        for method in available_methods:
            estimate = np.asarray(frozen["predicted_tables"][method], dtype=float)
            if _array_sha256(estimate) != frozen["prediction_sha256"][method]:
                raise PermissionError("frozen prediction hash differs")
            values = [
                multinomial_deviance_per_observation(truth[index], estimate[index])
                for index in zip(*np.nonzero(informative))
            ]
            loss = float(np.mean(values))
            losses[method][subject_index] = loss
            sample_losses[method] = loss
        samples.append(
            {
                "subject_id": subject,
                "batch": batches[subject],
                "informative_pairs": int(np.count_nonzero(informative)),
                "truth_table_sha256": _array_sha256(truth),
                "losses": sample_losses,
            }
        )

    comparisons: dict[str, Any] = {
        "primary_vs_matched_deviance_residual": _held_comparison(
            eligible,
            batches,
            losses["primary"],
            losses["matched_deviance_residual"],
            classical=False,
            seed_offset=0,
        ),
        "primary_vs_destroyed_link": _held_comparison(
            eligible,
            batches,
            losses["primary"],
            losses["destroyed_link"],
            classical=False,
            seed_offset=1,
        ),
    }
    comparisons["primary_vs_common_effect_cmle"] = _held_comparison(
        eligible,
        batches,
        losses["primary"],
        losses["common_effect_cmle"],
        classical=True,
        seed_offset=2,
    )
    comparisons["primary_vs_pooled_saturated_poisson"] = _held_comparison(
        eligible,
        batches,
        losses["primary"],
        losses["pooled_saturated_poisson"],
        classical=True,
        seed_offset=3,
    )
    transfer_pass = (
        comparisons["primary_vs_matched_deviance_residual"]["passes"]
        and comparisons["primary_vs_destroyed_link"]["passes"]
    )
    classical_gain = (
        prediction.get("classical_head_to_head_ready") is True
        and set(CLASSICAL_METHODS) <= set(available_methods)
        and comparisons["primary_vs_common_effect_cmle"]["passes"]
        and comparisons["primary_vs_pooled_saturated_poisson"]["passes"]
    )
    if transfer_pass and classical_gain:
        status = "CONFIRMATION_PASS_WITH_GAIN_OVER_BOTH_CLASSICAL_INTERACTIONS"
    elif transfer_pass:
        status = "CONFIRMATION_PASS_WITHOUT_GAIN_OVER_BOTH_CLASSICAL_INTERACTIONS"
    else:
        status = "COMPLETED_CONFIRMATION_FAIL"
    return {
        "schema": "gse309593-independent-study-confirmation/1.0",
        "status": status,
        "created_at_utc": _timestamp(),
        "prediction_tag": PREDICTION_TAG,
        "prediction_commit": prediction_commit,
        "prediction_sha256": _sha256(DEFAULT_PREDICTION),
        "score_authorization_commit": authorization_commit,
        "score_authorization_sha256": _sha256(DEFAULT_SCORE_AUTHORIZATION),
        "source_model_sha256": prediction["source_model_sha256"],
        "available_methods": available_methods,
        "classical_head_to_head_ready": prediction.get("classical_head_to_head_ready"),
        "eligible_subjects": eligible,
        "eligible_subject_count": len(eligible),
        "all_seven_batches_represented": len({batches[value] for value in eligible})
        == 7,
        "panel": panel,
        "ordered_rna_to_adt_pairs": len(panel) ** 2,
        "losses": {
            method: {subject: float(value) for subject, value in zip(eligible, values)}
            for method, values in losses.items()
        },
        "comparisons": comparisons,
        "transfer_confirmation_pass": transfer_pass,
        "gain_over_both_classical_interactions": classical_gain,
        "samples": samples,
        "outcome_failure_rule": (
            "A supported result that misses a criterion is a completed negative, not a QC refusal."
        ),
        "claim_scope": CLAIM_SCOPE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    claim = subparsers.add_parser("claim")
    claim.add_argument("stage", choices=tuple(STAGE_PATHS))

    rna = subparsers.add_parser("run-rna")
    rna.add_argument("--scratch-dir", type=Path, required=True)
    rna.add_argument("--selection-bridge", type=Path, required=True)
    rna.add_argument("--rna-states", type=Path, required=True)

    adt = subparsers.add_parser("run-adt")
    adt.add_argument("--scratch-dir", type=Path, required=True)
    adt.add_argument("--selection-bridge", type=Path, required=True)
    adt.add_argument("--adt-states", type=Path, required=True)

    subparsers.add_parser("run-prediction")
    verify = subparsers.add_parser("verify-stage")
    verify.add_argument("stage", choices=tuple(STAGE_PATHS))
    subparsers.add_parser("verify-result")

    score = subparsers.add_parser("run-score")
    score.add_argument("--rna-states", type=Path, required=True)
    score.add_argument("--adt-states", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "claim":
        payload = claim_stage(args.stage)
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
    elif args.command == "verify-stage":
        payload, _ = _verify_public_stage(args.stage)
    elif args.command == "verify-result":
        payload, _ = _verify_public_stage("score")
    else:
        payload = _run_claimed_stage(
            "score", lambda: _score_stage_body(args.rna_states, args.adt_states)
        )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
