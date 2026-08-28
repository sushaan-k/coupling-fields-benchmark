"""Acquire and reduce only the frozen GSE299043 development H5ADs."""

from __future__ import annotations

import copy
import csv
import hashlib
import http.client
import json
import os
import ssl
import shutil
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments import reduce_gse299043_mln as reducer


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "data/confirmation/gse299043_mln/source_manifest_template_v1.json"
OUTPUT = ROOT / "data/confirmation/gse299043_mln/source_manifest_v1.json"
PREFLIGHT = ROOT / "data/development/gse299043_mln/metadata_preflight_v1.tsv"
PROTOCOL = ROOT / "docs/GSE299043_MLN_HELD_SITE_CONFIRMATION_PROTOCOL_2026-08-28.md"
DESIGNATION = ROOT / "data/confirmation/gse299043_mln/candidate_designation_v1.json"
FAMILY_POLICY = ROOT / "data/confirmation/gse299043_mln/family_policy_v1.json"
AUTH_TEMPLATE = (
    ROOT / "data/confirmation/gse299043_mln/score_authorization_template_v1.json"
)
AUTH_PUBLICATION_TEMPLATE = (
    ROOT
    / "data/confirmation/gse299043_mln/score_authorization_publication_template_v1.json"
)
REDUCER = ROOT / "experiments/reduce_gse299043_mln.py"
EVALUATOR = ROOT / "experiments/evaluate_gse299043_mln_development.py"
RUNNER = ROOT / "experiments/confirm_gse299043_mln.py"

MEMBER_DIR = ROOT / "data/development/gse299043_mln/source_members"
PIECE_DIR = ROOT / "data/development/gse299043_mln/library_pieces"
DONOR_DIR = ROOT / "data/development/gse299043_mln/reduced_donors"
REDUCED_OUTPUT = ROOT / "data/development/gse299043_mln/reduced_development_v1.json"
DEVELOPMENT_ATTEMPT = (
    ROOT / "data/development/gse299043_mln/development_attempt_v1.json"
)
DEVELOPMENT_REFUSAL = (
    ROOT / "results/development/gse299043_mln_development_acquisition_refusal.json"
)

PRIOR_REFUSAL = ROOT / "results/development/gse279451_sepsis_evaluation_refusal.json"
MINIMUM_FREE_AFTER_ACQUISITION = 512 * 1024 * 1024
DEVELOPMENT_MEMBER_COUNT = 56
DEVELOPMENT_BYTES = 2_991_542_178
HELD_MEMBER_COUNT = 151
HELD_BYTES = 4_766_004_153
DOWNLOAD_ATTEMPTS = 3
FROZEN_TEMPLATE_SHA256 = (
    "83a36f5eeb2641012e763652c21630c0b29d1230fbcbfd721f46a6828009c43c"
)
FROZEN_PREFLIGHT_SHA256 = (
    "dfc929364a40895620c39897a671542670f1cf1e89058cf4ff02f51d16c86933"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


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
    with os.fdopen(descriptor, "w") as stream:
        stream.write(serialized)


def _assert_family_available() -> None:
    policy = _read_json(FAMILY_POLICY)
    expected = {
        "schema": "gse299043-mln-family-policy/1.0",
        "status": "ELIGIBLE_AFTER_PRIOR_TERMINAL_REFUSAL",
        "prior_candidate": "GSE279451",
        "required_terminal_artifact": (
            "results/development/gse279451_sepsis_evaluation_refusal.json"
        ),
        "required_terminal_status": "TERMINAL_DEVELOPMENT_EVALUATION_REFUSAL",
        "prior_held_matrix_members_opened": 0,
        "prior_rerun_permitted": False,
        "gse299043_development_h5ad_access_before_public_freeze": False,
        "gse299043_held_h5ad_access_before_public_prediction_authorization": False,
        "maximum_development_attempts": 1,
        "maximum_held_score_attempts": 1,
    }
    for key, value in expected.items():
        if policy.get(key) != value:
            raise PermissionError(f"family policy differs at {key}")
    if not PRIOR_REFUSAL.is_file() or PRIOR_REFUSAL.is_symlink():
        raise PermissionError("the required prior terminal refusal is absent")
    if _sha256(PRIOR_REFUSAL) != policy.get("required_terminal_artifact_sha256"):
        raise PermissionError("the required prior terminal refusal hash differs")
    prior = _read_json(PRIOR_REFUSAL)
    if (
        prior.get("status") != policy["required_terminal_status"]
        or prior.get("rerun_permitted") is not False
        or prior.get("held_matrix_members_opened") != 0
    ):
        raise PermissionError("the prior candidate is not terminally outcome-sealed")


def _artifact_bindings() -> dict[str, str]:
    artifacts = {
        "source_template_sha256": TEMPLATE,
        "metadata_preflight_sha256": PREFLIGHT,
        "protocol_sha256": PROTOCOL,
        "candidate_designation_sha256": DESIGNATION,
        "family_policy_sha256": FAMILY_POLICY,
        "score_authorization_template_sha256": AUTH_TEMPLATE,
        "score_authorization_publication_template_sha256": AUTH_PUBLICATION_TEMPLATE,
        "acquisition_sha256": Path(__file__),
        "reducer_sha256": REDUCER,
        "development_evaluator_sha256": EVALUATOR,
        "runner_sha256": RUNNER,
    }
    missing = [path for path in artifacts.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "bound artifact is absent before network access: "
            + ", ".join(path.name for path in missing)
        )
    bindings = {name: _sha256(path) for name, path in artifacts.items()}
    from experiments import evaluate_gse299043_mln_development as evaluator

    helper = getattr(evaluator, "_transitive_bindings", None)
    if helper is not None:
        transitive = helper()
        if not isinstance(transitive, dict) or not all(
            isinstance(key, str) and isinstance(value, str) and len(value) == 64
            for key, value in transitive.items()
        ):
            raise ValueError("development evaluator transitive bindings are malformed")
        bindings.update(transitive)
    return bindings


def _preflight_rows() -> list[dict[str, str]]:
    with PREFLIGHT.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        expected_fields = [
            "donor",
            "site",
            "role",
            "gsm",
            "seq_run",
            "gex_library",
            "version",
            "tissue",
            "source",
            "library_protocol",
            "raw_matrix_assays",
            "filename",
            "url",
            "content_length_bytes",
        ]
        if reader.fieldnames != expected_fields:
            raise ValueError("metadata preflight columns differ from the frozen schema")
        return list(reader)


def _member_preflight_record(member: dict[str, Any]) -> dict[str, str]:
    return {
        "donor": str(member.get("donor")),
        "site": str(member.get("site")),
        "role": str(member.get("role")),
        "gsm": str(member.get("gsm")),
        "seq_run": str(member.get("seq_run")),
        "gex_library": str(member.get("gex_library")),
        "version": str(member.get("version")),
        "tissue": str(member.get("tissue_metadata")),
        "source": str(member.get("source_program")),
        "library_protocol": str(member.get("library_protocol")),
        "raw_matrix_assays": str(member.get("raw_matrix_assays")),
        "filename": str(member.get("filename")),
        "url": str(member.get("url")),
        "content_length_bytes": str(member.get("bytes")),
    }


def _validate_member(member: Any) -> dict[str, Any]:
    if not isinstance(member, dict):
        raise ValueError("source manifest member is not an object")
    required = {
        "bytes",
        "donor",
        "filename",
        "gex_library",
        "gsm",
        "library_protocol",
        "local_path",
        "raw_matrix_assays",
        "retained",
        "role",
        "seq_run",
        "sha256",
        "site",
        "source_program",
        "tissue_metadata",
        "url",
        "version",
    }
    if set(member) != required:
        raise ValueError("source manifest member fields differ")
    donor = member["donor"]
    role = member["role"]
    expected_role = (
        "development"
        if donor in reducer.DEVELOPMENT_DONORS
        else "held"
        if donor in reducer.HELD_DONORS
        else None
    )
    match = reducer.FILENAME_PATTERN.fullmatch(str(member["filename"]))
    if (
        role != expected_role
        or match is None
        or match.group("donor") != donor
        or match.group("run") != member["seq_run"]
        or match.group("library") != member["gex_library"]
        or member["url"]
        != (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE299nnn/"
            f"GSE299043/suppl/{member['filename']}"
        )
        or not isinstance(member["bytes"], int)
        or isinstance(member["bytes"], bool)
        or member["bytes"] <= 0
        or member["sha256"] is not None
        or member["local_path"] is not None
        or member["retained"] is not False
        or member["library_protocol"] != "10x 5' v2"
        or member["raw_matrix_assays"] != "snRNA-seq, CITE-seq"
        or (role == "development" and member["site"] != "UK")
        or (role == "held" and member["site"] != "NY")
    ):
        raise ValueError("source manifest member differs from the frozen contract")
    return member


def _validated_template() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if _sha256(TEMPLATE) != FROZEN_TEMPLATE_SHA256:
        raise PermissionError("source template SHA-256 differs from the frozen code")
    if _sha256(PREFLIGHT) != FROZEN_PREFLIGHT_SHA256:
        raise PermissionError("metadata preflight SHA-256 differs from the frozen code")
    template = _read_json(TEMPLATE)
    metadata = template.get("metadata_manifest")
    design = template.get("design")
    hashsolo = template.get("hashsolo_contract")
    markers = template.get("markers")
    contract = template.get("member_contract")
    if (
        template.get("schema") != "gse299043-mln-source/1.0"
        or template.get("status") != "SOURCE_UNAVAILABLE_OUTCOME_ACCESS_DISABLED"
        or template.get("accession") != "GSE299043"
        or template.get("bioproject") != "PRJNA1215450"
        or template.get("study_doi") != "10.1038/s41590-025-02241-4"
        or template.get("bindings") != {}
        or not isinstance(metadata, dict)
        or metadata.get("source_path")
        != "data/development/gse299043_mln/metadata_preflight_v1.tsv"
        or metadata.get("sha256") != FROZEN_PREFLIGHT_SHA256
        or metadata.get("rows") != DEVELOPMENT_MEMBER_COUNT + HELD_MEMBER_COUNT
        or metadata.get("development_members") != DEVELOPMENT_MEMBER_COUNT
        or metadata.get("development_bytes") != DEVELOPMENT_BYTES
        or metadata.get("held_members") != HELD_MEMBER_COUNT
        or metadata.get("held_bytes") != HELD_BYTES
        or not isinstance(design, dict)
        or design.get("development_donors") != list(reducer.DEVELOPMENT_DONORS)
        or design.get("held_donors") != list(reducer.HELD_DONORS)
        or design.get("cells_per_donor") != reducer.CELL_BUDGET
        or design.get("cell_selection_salt") != reducer.CELL_SELECTION_SALT
        or design.get("adt_tie_salt") != reducer.ADT_TIE_SALT
        or not isinstance(hashsolo, dict)
        or hashsolo.get("single_tissue_one_hto_exception")
        != {
            "donor": reducer.SINGLE_TISSUE_ONE_HTO_DONOR,
            "filename": reducer.SINGLE_TISSUE_ONE_HTO_FILENAME,
            "metadata_preflight_sha256": FROZEN_PREFLIGHT_SHA256,
            "metadata_preflight_tissue": "pooled:mesenteric lymph node",
            "normalized_hto_id": reducer.SINGLE_TISSUE_ONE_HTO_TAG,
            "rule": (
                "assign every cell only when this exact member contains exactly "
                "this sole normalized donor HTO; every other member requires at "
                "least two donor HTOs"
            ),
        }
        or not isinstance(markers, dict)
        or markers.get("order") != list(reducer.MARKERS)
        or markers.get("rna_feature_ids") != list(reducer.RNA_FEATURE_IDS)
        or markers.get("adt_feature_ids") != list(reducer.ADT_FEATURE_IDS)
        or markers.get("ordered_entities") != len(reducer.MARKERS) ** 2
        or contract
        != {
            "delete_after_reduction": True,
            "development_acquisition_may_open_roles": ["development"],
            "download_verify_before_open": True,
            "held_acquisition_requires_public_prediction_authorization": True,
            "members": DEVELOPMENT_MEMBER_COUNT + HELD_MEMBER_COUNT,
            "stream_one_member_at_a_time": True,
        }
    ):
        raise PermissionError("source template differs from the frozen contract")

    members = template.get("members")
    if not isinstance(members, list):
        raise ValueError("source template members are absent")
    validated = [_validate_member(member) for member in members]
    if len(validated) != DEVELOPMENT_MEMBER_COUNT + HELD_MEMBER_COUNT:
        raise ValueError("source template must bind exactly 207 members")
    filenames = [member["filename"] for member in validated]
    urls = [member["url"] for member in validated]
    if len(set(filenames)) != len(filenames) or len(set(urls)) != len(urls):
        raise ValueError("source template contains duplicate members")
    if [_member_preflight_record(member) for member in validated] != _preflight_rows():
        raise PermissionError("source template differs from the metadata preflight")

    development = [member for member in validated if member["role"] == "development"]
    held = [member for member in validated if member["role"] == "held"]
    if (
        len(development) != DEVELOPMENT_MEMBER_COUNT
        or sum(member["bytes"] for member in development) != DEVELOPMENT_BYTES
        or len(held) != HELD_MEMBER_COUNT
        or sum(member["bytes"] for member in held) != HELD_BYTES
        or {member["donor"] for member in development}
        != set(reducer.DEVELOPMENT_DONORS)
        or {member["donor"] for member in held} != set(reducer.HELD_DONORS)
    ):
        raise ValueError("source template member split differs")
    development.sort(
        key=lambda member: (
            reducer.DEVELOPMENT_DONORS.index(member["donor"]),
            member["filename"],
        )
    )
    return template, development


def _retryable_transport_error(error: Exception) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code in {408, 429, 500, 502, 503, 504}
    return isinstance(
        error,
        (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.IncompleteRead,
            ssl.SSLError,
        ),
    )


def _download(url: str, destination: Path, expected_bytes: int) -> tuple[int, str]:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"download destination already exists: {destination.name}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"partial download already exists: {temporary.name}")
    request = urllib.request.Request(url, headers={"User-Agent": "coupling-fields/1.0"})
    for attempt in range(DOWNLOAD_ATTEMPTS):
        temporary.unlink(missing_ok=True)
        digest = hashlib.sha256()
        observed = 0
        try:
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                temporary.open("xb") as output,
            ):
                content_length = response.headers.get("Content-Length")
                if content_length is None or int(content_length) != expected_bytes:
                    raise PermissionError(
                        "remote byte count differs from the frozen source manifest"
                    )
                for block in iter(lambda: response.read(8 << 20), b""):
                    output.write(block)
                    digest.update(block)
                    observed += len(block)
            if observed != expected_bytes:
                raise PermissionError(
                    "downloaded byte count differs from the frozen source manifest"
                )
            os.replace(temporary, destination)
            return observed, digest.hexdigest()
        except Exception as error:
            temporary.unlink(missing_ok=True)
            if attempt + 1 == DOWNLOAD_ATTEMPTS or not _retryable_transport_error(
                error
            ):
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable download retry state")


def _download_member(member: dict[str, Any], destination: Path) -> tuple[int, str]:
    if member.get("role") != "development" or member.get("donor") not in set(
        reducer.DEVELOPMENT_DONORS
    ):
        raise PermissionError("development acquisition forbids every held member")
    expected_destination = MEMBER_DIR / str(member.get("filename"))
    if destination != expected_destination:
        raise PermissionError("development member destination differs")
    return _download(member["url"], destination, member["bytes"])


def _terminal_artifact_exists() -> bool:
    direct = (OUTPUT, REDUCED_OUTPUT, DEVELOPMENT_ATTEMPT, DEVELOPMENT_REFUSAL)
    if any(path.exists() or path.is_symlink() for path in direct):
        return True
    for directory in (MEMBER_DIR, PIECE_DIR, DONOR_DIR):
        if directory.is_symlink():
            return True
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            return True
    return False


def _validate_library_reduction(
    payload: dict[str, Any], member: dict[str, Any], digest: str
) -> None:
    if (
        payload.get("schema") != "gse299043-mln-library-reduction/1.0"
        or payload.get("status") != "TARGET_MLN_LIBRARY_REDUCED"
        or payload.get("donor") != member["donor"]
        or payload.get("role") != "development"
        or payload.get("source_filename") != member["filename"]
        or payload.get("source_bytes") != member["bytes"]
        or payload.get("source_sha256") != digest
    ):
        raise PermissionError("library reduction differs from its downloaded source")


def acquire() -> dict[str, Any]:
    if _terminal_artifact_exists():
        raise FileExistsError("a GSE299043 development acquisition artifact exists")
    _assert_family_available()
    template, development = _validated_template()
    bindings = _artifact_bindings()
    required_bytes = max(member["bytes"] for member in development)
    free_bytes = shutil.disk_usage(ROOT).free
    if free_bytes - required_bytes < MINIMUM_FREE_AFTER_ACQUISITION:
        raise OSError("insufficient disk for one streamed development H5AD")

    plan_hash = _json_sha256(development)
    _write_json_exclusive(
        DEVELOPMENT_ATTEMPT,
        {
            "schema": "gse299043-mln-development-attempt/1.0",
            "status": "TERMINAL_DEVELOPMENT_ATTEMPT_STARTED",
            "source_template_sha256": _sha256(TEMPLATE),
            "metadata_preflight_sha256": _sha256(PREFLIGHT),
            "development_member_plan_sha256": plan_hash,
            "axis_members_sha256": plan_hash,
            "artifact_bindings": bindings,
            "development_members_planned": DEVELOPMENT_MEMBER_COUNT,
            "development_bytes_planned": DEVELOPMENT_BYTES,
            "first_network_request_starts_after_this_write": True,
            "held_h5ad_members_requested": 0,
            "rerun_permitted": False,
        },
    )

    active = copy.deepcopy(template)
    active["status"] = "NONHELD_SOURCE_ACCESS_AUTHORIZED"
    active["bindings"] = bindings
    member_by_filename = {member["filename"]: member for member in active["members"]}
    pieces_by_donor: dict[str, list[Path]] = defaultdict(list)
    donor_payloads: list[dict[str, Any]] = []
    donor_artifacts: list[dict[str, Any]] = []
    try:
        for member in development:
            destination = MEMBER_DIR / member["filename"]
            piece = PIECE_DIR / f"{member['filename']}.json"
            try:
                observed, digest = _download_member(member, destination)
                payload = reducer.reduce_library(
                    destination, member["donor"], piece, phase="development"
                )
                _validate_library_reduction(payload, member, digest)
                pieces_by_donor[member["donor"]].append(piece)
                active_member = member_by_filename[member["filename"]]
                active_member["sha256"] = digest
                active_member["local_path"] = None
                active_member["retained"] = False
                if observed != active_member["bytes"]:
                    raise AssertionError("verified member byte count changed")
            finally:
                destination.unlink(missing_ok=True)
                destination.with_name(destination.name + ".part").unlink(
                    missing_ok=True
                )

        for donor in reducer.DEVELOPMENT_DONORS:
            output = DONOR_DIR / f"{donor}.json"
            payload = reducer.finalize_donor(
                pieces_by_donor[donor], donor, output, phase="development"
            )
            if (
                payload.get("schema") != "gse299043-mln-reduced-donor/1.0"
                or payload.get("status") != "DONOR_REDUCTION_COMPLETE"
                or payload.get("donor") != donor
                or payload.get("role") != "development"
                or payload.get("cells") != reducer.CELL_BUDGET
                or payload.get("entity_count") != len(reducer.MARKERS) ** 2
                or _read_json(output) != payload
            ):
                raise PermissionError("final donor reduction differs from its contract")
            donor_payloads.append(payload)
            donor_artifacts.append(
                {
                    "donor": donor,
                    "path": output.relative_to(ROOT).as_posix(),
                    "sha256": _sha256(output),
                }
            )

        active["access_audit"] = {
            "development_members_opened_before_public_freeze": 0,
            "h5ad_members_opened_before_template_freeze": 0,
            "held_members_opened_before_public_prediction_authorization": 0,
            "matrix_entries_decoded_before_template_freeze": 0,
            "development_attempt_sha256": _sha256(DEVELOPMENT_ATTEMPT),
            "development_h5ad_members_requested": DEVELOPMENT_MEMBER_COUNT,
            "development_h5ad_members_decoded": DEVELOPMENT_MEMBER_COUNT,
            "held_h5ad_members_requested": 0,
            "held_h5ad_members_opened": 0,
            "held_h5ad_members_decoded": 0,
            "maximum_concurrent_source_h5ads": 1,
        }
        _write_json_exclusive(OUTPUT, active)
        reduced = {
            "schema": "gse299043-mln-reduced-development/1.0",
            "status": "NONHELD_REDUCTION_COMPLETE",
            "source_manifest_sha256": _sha256(OUTPUT),
            "development_attempt_sha256": _sha256(DEVELOPMENT_ATTEMPT),
            "development_donors": list(reducer.DEVELOPMENT_DONORS),
            "held_donors": list(reducer.HELD_DONORS),
            "markers": list(reducer.MARKERS),
            "rna_feature_ids": list(reducer.RNA_FEATURE_IDS),
            "adt_feature_ids": list(reducer.ADT_FEATURE_IDS),
            "entity_count": len(reducer.MARKERS) ** 2,
            "primary_cells_per_donor": reducer.CELL_BUDGET,
            "cell_selection_salt": reducer.CELL_SELECTION_SALT,
            "adt_tie_salt": reducer.ADT_TIE_SALT,
            "donors": donor_payloads,
            "donor_artifacts": donor_artifacts,
            "access_audit": {
                "development_h5ad_members_decoded": DEVELOPMENT_MEMBER_COUNT,
                "held_h5ad_members_opened": 0,
                "held_h5ad_members_decoded": 0,
                "maximum_concurrent_source_h5ads": 1,
            },
        }
        _write_json_exclusive(REDUCED_OUTPUT, reduced)
        return active
    except Exception as error:
        if not DEVELOPMENT_REFUSAL.exists():
            _write_json_exclusive(
                DEVELOPMENT_REFUSAL,
                {
                    "schema": "gse299043-mln-development-refusal/1.0",
                    "status": "TERMINAL_DEVELOPMENT_ACQUISITION_REFUSAL",
                    "error_type": type(error).__name__,
                    "reason": (
                        "development acquisition or reduction refused after the "
                        "terminal attempt"
                    ),
                    "development_attempt_sha256": _sha256(DEVELOPMENT_ATTEMPT),
                    "held_h5ad_members_requested": 0,
                    "held_h5ad_members_opened": 0,
                    "rerun_permitted": False,
                },
            )
        raise


if __name__ == "__main__":
    print(json.dumps(acquire(), indent=2, sort_keys=True, allow_nan=False))
