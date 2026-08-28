"""Acquire only the preregistered nonheld GSE279451 source members."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any

from experiments import reduce_gse279451_sepsis as reducer
from experiments.confirm_gse279451_sepsis import _assert_family_available


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "data/confirmation/gse279451_sepsis/source_manifest_template_v1.json"
OUTPUT = ROOT / "data/confirmation/gse279451_sepsis/source_manifest_v1.json"
MEMBER_DIR = ROOT / "data/development/gse279451_sepsis/source_members"
REDUCED_OUTPUT = ROOT / "data/development/gse279451_sepsis/reduced_development_v1.json"
DEVELOPMENT_ATTEMPT = (
    ROOT / "data/development/gse279451_sepsis/development_attempt_v1.json"
)
DEVELOPMENT_REFUSAL = (
    ROOT / "results/development/gse279451_sepsis_development_acquisition_refusal.json"
)
PROTOCOL = ROOT / "docs/GSE279451_SEPSIS_HELD_DONOR_CONFIRMATION_PROTOCOL_2026-08-28.md"
PREFLIGHT = ROOT / "data/development/gse279451_sepsis/metadata_preflight_v1.json"
DESIGNATION = ROOT / "data/confirmation/gse279451_sepsis/candidate_designation_v1.json"
FAMILY_POLICY = ROOT / "data/confirmation/gse279451_sepsis/family_policy_v1.json"
REDUCER = ROOT / "experiments/reduce_gse279451_sepsis.py"
EVALUATOR = ROOT / "experiments/evaluate_gse279451_sepsis_development.py"
RUNNER = ROOT / "experiments/confirm_gse279451_sepsis.py"
TESTS = ROOT / "tests/test_gse279451_sepsis_confirmation.py"
EVALUATOR_TESTS = ROOT / "tests/test_evaluate_gse279451_sepsis_development.py"
TRANSITIVE_ARTIFACTS = {
    "hierarchical_estimator_sha256": ROOT
    / "mapreg/hierarchical_conditional_coupling.py",
    "heterogeneity_estimator_sha256": ROOT
    / "mapreg/heterogeneity_adaptive_coupling.py",
    "classical_residuals_sha256": ROOT / "mapreg/classical_residuals.py",
    "coupling_fields_sha256": ROOT / "mapreg/coupling_fields.py",
    "table_prediction_sha256": ROOT / "mapreg/table_prediction.py",
    "hierarchical_estimator_tests_sha256": ROOT
    / "tests/test_hierarchical_conditional_coupling.py",
    "heterogeneity_estimator_tests_sha256": ROOT
    / "tests/test_heterogeneity_adaptive_coupling.py",
    "classical_residuals_tests_sha256": ROOT / "tests/test_classical_residuals_full.py",
    "coupling_fields_tests_sha256": ROOT / "tests/test_coupling_fields.py",
    "table_prediction_tests_sha256": ROOT / "tests/test_table_prediction.py",
}

BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8571nnn"
MINIMUM_FREE_AFTER_ACQUISITION = 512 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"{path.name} contains nonfinite JSON token {token}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _serialized_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    serialized = _serialized_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(serialized)


def _filename(donor: dict[str, Any], kind: str) -> str:
    suffix = {
        "barcodes": "barcodes.tsv.gz",
        "features": "features.tsv.gz",
        "matrix": "matrix.mtx.gz",
    }[kind]
    return f"{donor['accession']}_{donor['sample']}.{suffix}"


def _member_plan(donors: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str]]:
    plan = []
    for donor in donors:
        kinds = ["barcodes", "features"]
        if donor.get("role") == "development":
            kinds.append("matrix")
        plan.extend((donor, kind) for kind in kinds)
    return plan


def _download(
    url: str,
    destination: Path,
    expected_bytes: int,
    expected_sha256: str | None,
) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    digest = hashlib.sha256()
    observed = 0
    request = urllib.request.Request(url, headers={"User-Agent": "coupling-fields/1.0"})
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            temporary.open("xb") as output,
        ):
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                if int(content_length) != expected_bytes:
                    raise PermissionError(
                        "remote byte count differs from the frozen manifest"
                    )
            for block in iter(lambda: response.read(1024 * 1024), b""):
                output.write(block)
                digest.update(block)
                observed += len(block)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if observed != expected_bytes:
        temporary.unlink(missing_ok=True)
        raise PermissionError("downloaded byte count differs from the frozen manifest")
    observed_hash = digest.hexdigest()
    if expected_sha256 is not None and observed_hash != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise PermissionError(
            "downloaded SHA-256 differs from the frozen axis preflight"
        )
    os.replace(temporary, destination)
    return observed, observed_hash


def _artifact_bindings() -> dict[str, str]:
    artifacts = {
        "source_template_sha256": TEMPLATE,
        "candidate_designation_sha256": DESIGNATION,
        "family_policy_sha256": FAMILY_POLICY,
        "protocol_sha256": PROTOCOL,
        "metadata_preflight_sha256": PREFLIGHT,
        "acquisition_sha256": Path(__file__),
        "reducer_sha256": REDUCER,
        "development_evaluator_sha256": EVALUATOR,
        "runner_sha256": RUNNER,
        "tests_sha256": TESTS,
        "development_evaluator_tests_sha256": EVALUATOR_TESTS,
        **TRANSITIVE_ARTIFACTS,
    }
    missing = [path for path in artifacts.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "bound artifact is absent before network access: "
            + ", ".join(path.name for path in missing)
        )
    return {name: _sha256(path) for name, path in artifacts.items()}


def _member_record(
    donor: dict[str, Any], kind: str, observed: int, digest: str, destination: Path
) -> dict[str, Any]:
    return {
        "accession": donor["accession"],
        "kind": kind,
        "url": f"{BASE_URL}/{donor['accession']}/suppl/{_filename(donor, kind)}",
        "bytes": observed,
        "local_path": destination.relative_to(ROOT).as_posix(),
        "sha256": digest,
        "retained": True,
    }


def acquire() -> dict[str, Any]:
    if any(
        path.exists()
        for path in (OUTPUT, REDUCED_OUTPUT, DEVELOPMENT_ATTEMPT, DEVELOPMENT_REFUSAL)
    ):
        raise FileExistsError("a GSE279451 development acquisition artifact exists")
    _assert_family_available()
    template = _read_json(TEMPLATE)
    if template.get("status") != "SOURCE_UNAVAILABLE" or template.get("members") != []:
        raise PermissionError("source template is not the disabled public freeze")
    donors = template.get("donors")
    if not isinstance(donors, list) or len(donors) != 40:
        raise ValueError("source template must bind exactly 40 donors")
    development = [donor for donor in donors if donor.get("role") == "development"]
    held = [donor for donor in donors if donor.get("role") == "held"]
    if len(development) != 19 or len(held) != 21:
        raise ValueError("source template donor split differs")
    if {donor["accession"] for donor in development} != set(
        reducer.DEVELOPMENT_DONORS
    ) or {donor["accession"] for donor in held} != set(reducer.HELD_DONORS):
        raise ValueError("source template donor accessions differ")
    bindings = _artifact_bindings()
    required_bytes = sum(
        int(donor["barcode_bytes"]) + int(donor["feature_bytes"]) for donor in donors
    ) + max(int(donor["matrix_bytes"]) for donor in development)
    free_bytes = shutil.disk_usage(ROOT).free
    if free_bytes - required_bytes < MINIMUM_FREE_AFTER_ACQUISITION:
        raise OSError("insufficient disk for the nonheld source acquisition")

    members: list[dict[str, Any]] = []
    for donor, kind in _member_plan(donors):
        if kind == "matrix":
            continue
        filename = _filename(donor, kind)
        url = f"{BASE_URL}/{donor['accession']}/suppl/{filename}"
        destination = MEMBER_DIR / filename
        singular = kind[:-1]
        observed, digest = _download(
            url,
            destination,
            int(donor[f"{singular}_bytes"]),
            str(donor[f"{singular}_sha256"]),
        )
        members.append(_member_record(donor, kind, observed, digest, destination))

    source = copy.deepcopy(template)
    source["status"] = "NONHELD_SOURCE_ACCESS_AUTHORIZED"
    source["members"] = members
    source["bindings"] = bindings
    axis_members_sha256 = hashlib.sha256(
        json.dumps(
            members, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    _write_json_exclusive(
        DEVELOPMENT_ATTEMPT,
        {
            "schema": "gse279451-sepsis-development-attempt/1.0",
            "status": "TERMINAL_DEVELOPMENT_ATTEMPT_STARTED",
            "source_template_sha256": _sha256(TEMPLATE),
            "axis_members_sha256": axis_members_sha256,
            "artifact_bindings": bindings,
            "first_development_matrix_acquisition_starts_after_this_write": True,
            "held_matrix_members_acquired": 0,
        },
    )
    reduced_donors = []
    try:
        for donor in development:
            kind = "matrix"
            filename = _filename(donor, kind)
            url = f"{BASE_URL}/{donor['accession']}/suppl/{filename}"
            destination = MEMBER_DIR / filename
            observed, digest = _download(
                url, destination, int(donor["matrix_bytes"]), None
            )
            record = _member_record(donor, kind, observed, digest, destination)
            source["members"].append(record)
            try:
                reduced_donors.append(
                    reducer.reduce_donor(
                        source, donor["accession"], phase="development"
                    )
                )
            finally:
                destination.unlink(missing_ok=True)
                record["local_path"] = None
                record["retained"] = False

        source["access_audit"] = {
            "matrix_entries_decoded_before_template_freeze": 0,
            "held_matrix_bytes_read_before_public_prediction_authorization": 0,
            "held_matrix_members_acquired": 0,
            "development_matrix_members_acquired": 19,
            "development_attempt_sha256": _sha256(DEVELOPMENT_ATTEMPT),
            "sanger_path_or_content_accessed": False,
        }
        reducer._validate_manifest_shape(source)
        source_hash = hashlib.sha256(_serialized_json(source).encode()).hexdigest()
        reduced = {
            "schema": "gse279451-sepsis-reduced-development/1.0",
            "status": "NONHELD_REDUCTION_COMPLETE",
            "source_manifest_sha256": source_hash,
            "development_attempt_sha256": _sha256(DEVELOPMENT_ATTEMPT),
            "development_donors": list(reducer.DEVELOPMENT_DONORS),
            "held_donors": list(reducer.HELD_DONORS),
            "markers": list(reducer.MARKERS),
            "entity_count": 81,
            "primary_cells_per_donor": reducer.CELL_BUDGET,
            "cell_selection_salt": reducer.CELL_SELECTION_SALT,
            "all_cells_sensitivity_included": False,
            "donors": reduced_donors,
            "access_audit": {
                "development_matrix_members_decoded": len(reduced_donors),
                "held_matrix_members_opened": 0,
                "held_matrix_entries_decoded": 0,
                "maximum_concurrent_donor_matrices": 1,
            },
        }
        _write_json_exclusive(OUTPUT, source)
        _write_json_exclusive(REDUCED_OUTPUT, reduced)
        return source
    except Exception as error:
        if not DEVELOPMENT_REFUSAL.exists():
            _write_json_exclusive(
                DEVELOPMENT_REFUSAL,
                {
                    "schema": "gse279451-sepsis-development-refusal/1.0",
                    "status": "TERMINAL_DEVELOPMENT_ACQUISITION_REFUSAL",
                    "error_type": type(error).__name__,
                    "reason": "development acquisition or reduction refused after the terminal attempt",
                    "development_attempt_sha256": _sha256(DEVELOPMENT_ATTEMPT),
                    "held_matrix_members_acquired": 0,
                    "rerun_permitted": False,
                },
            )
        raise


if __name__ == "__main__":
    print(json.dumps(acquire(), indent=2, sort_keys=True, allow_nan=False))
