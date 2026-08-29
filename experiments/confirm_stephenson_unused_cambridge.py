"""Prospective score of the 11 unused Stephenson Cambridge donors.

The CLI has three phases. ``verify-preaccess`` reads public JSON artifacts only.
``predict`` implements the single replacement attempt allowed by the append-only
recovery amendment. It verifies source identity, writes a terminal attempt, then
reads RNA margins only. ``score`` requires public frozen predictions, verifies
source identity, writes a second terminal attempt, and only then reads linked
RNA and ADT values once.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterator
import urllib.request

import fsspec
import h5py
import numpy as np

from experiments import confirm_stephenson_citeseq as stephenson


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/confirmation/stephenson_unused_cambridge"
DEFAULT_SOURCE = ROOT / "data/confirmation/stephenson_citeseq/source_manifest_v1.json"
DEFAULT_DEVELOPMENT = ROOT / "results/development/stephenson_citeseq_development.json"
DEFAULT_CLASSICAL = (
    ROOT / "data/development/stephenson_unused_cambridge/classical_fields_v1.json"
)
DEFAULT_CLASSICAL_AUDIT = (
    ROOT / "results/development/classical_interaction_baselines_posthoc.json"
)
DEFAULT_DESIGNATION = DATA_DIR / "candidate_designation_v1.json"
DEFAULT_PROTOCOL = (
    ROOT / "docs/STEPHENSON_UNUSED_CAMBRIDGE_HELD_DONOR_PROTOCOL_2026-08-29.md"
)
DEFAULT_PREACCESS = (
    ROOT / "results/development/stephenson_unused_cambridge_preaccess_v1.json"
)
DEFAULT_INITIAL_PREDICTION_AUTHORIZATION = DATA_DIR / "prediction_authorization_v1.json"
DEFAULT_INITIAL_PREDICTION_ATTEMPT = DATA_DIR / "prediction_attempt_v1.json"
DEFAULT_INITIAL_PREDICTION = (
    ROOT / "results/stephenson_unused_cambridge_predictions_v1.json"
)
DEFAULT_INITIAL_SCORE_AUTHORIZATION = DATA_DIR / "score_authorization_v1.json"
DEFAULT_INITIAL_SCORE_ATTEMPT = DATA_DIR / "score_attempt_v1.json"
DEFAULT_INITIAL_SCORE = (
    ROOT / "results/stephenson_unused_cambridge_confirmation_v1.json"
)
DEFAULT_RECOVERY_AMENDMENT = DATA_DIR / "prediction_recovery_amendment_v1_1.json"
DEFAULT_RECOVERY_PROTOCOL = (
    ROOT
    / "docs/STEPHENSON_UNUSED_CAMBRIDGE_PREDICTION_RECOVERY_AMENDMENT_2026-08-29.md"
)
DEFAULT_PREDICTION_AUTHORIZATION = DATA_DIR / "prediction_authorization_v1_1.json"
DEFAULT_PREDICTION_ATTEMPT = DATA_DIR / "prediction_attempt_v1_1.json"
DEFAULT_PREDICTION = ROOT / "results/stephenson_unused_cambridge_predictions_v1_1.json"
DEFAULT_SCORE_AUTHORIZATION = DATA_DIR / "score_authorization_v1_1.json"
DEFAULT_SCORE_ATTEMPT = DATA_DIR / "score_attempt_v1_1.json"
DEFAULT_SCORE = ROOT / "results/stephenson_unused_cambridge_confirmation_v1_1.json"

OFFICIAL_H5AD_NAME = "covid_portal_210320_with_raw.h5ad"
OFFICIAL_H5AD_BYTES = 7_187_322_881
OFFICIAL_H5AD_SHA256 = (
    "ec48f328f2e884c23376c8aa1f26041e11625762be5c30b0bd0869aa8bb1a334"
)
OFFICIAL_H5AD_URL = (
    "https://www.ebi.ac.uk/biostudies/files/E-MTAB-10026/"
    "covid_portal_210320_with_raw.h5ad"
)
OFFICIAL_H5AD_RESOLVED_URL = (
    "https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-/026/E-MTAB-10026/"
    "Files/covid_portal_210320_with_raw.h5ad"
)
OFFICIAL_H5AD_ETAG = '"1ac65d801-60a1af9b3c6c0"'
OFFICIAL_H5AD_LAST_MODIFIED = "Tue, 14 Nov 2023 11:20:35 GMT"
EXPECTED_SOURCE_SHA256 = (
    "431e8a370bc9b08a207ab0ff8d3581f80abaf0f36b55eba4accedc5685a3d3cd"
)
EXPECTED_DEVELOPMENT_SHA256 = (
    "2a8b535a0581b51a2a1eff8c764038a11cac7db08b67217c9b9dae22157e3e72"
)
EXPECTED_CLASSICAL_SHA256 = (
    "23dd6540fa1c133b5d5ff09af1a35ab212e4e7de5d2a255ee018f6d4e7b07a77"
)
EXPECTED_CLASSICAL_AUDIT_SHA256 = (
    "bc6efbb2ffe3404a294eae26b51e214054718113a8deeaf6b9f4e73ebf05f305"
)
EXPECTED_ORIGINAL_PREDICTION_SHA256 = (
    "7a183d4a2f55922dc07f8cbcdca904893710fe008e9169fdf618532ba06ea6e2"
)
EXPECTED_ORIGINAL_SCORE_SHA256 = (
    "5eb5fd2b41df7f4f7d822a92765ffe69854dcbe5f572f2db35cf433d7dd0adb1"
)
EXPECTED_INITIAL_PREDICTION_AUTHORIZATION_SHA256 = (
    "58c168f52b8e51a92c6a83785d400117c74af6767f21f3ef7e4201b1556f0250"
)
EXPECTED_INITIAL_PREDICTION_ATTEMPT_SHA256 = (
    "9600c70f423588e144bde0f4c5fbe85773a89c694ecb004ed462baee1d131186"
)
EXPECTED_RECOVERY_AMENDMENT_SHA256 = (
    "49929929e8f07149b6bc1ecbc7b8848980e0297b921a02087d9623cf208c05ff"
)
EXPECTED_DESIGNATION_SHA256 = (
    "b6bcc8049f688b6c5c5297c348a1f425068d56afb797fe6e8e55fd8f833bcd60"
)
EXPECTED_PROTOCOL_SHA256 = (
    "a4d302c5c7c0fd8f28b5c3d7d8a142120816d89eb13ba82a801cc0c23f8b3c61"
)

UNUSED = (
    ("CV0073", "BGCV08_CV0073"),
    ("CV0094", "BGCV07_CV0094"),
    ("CV0100", "BGCV04_CV0100"),
    ("CV0134", "BGCV07_CV0134"),
    ("CV0176", "BGCV15_CV0176"),
    ("CV0180", "BGCV11_CV0180"),
    ("CV0200", "BGCV03_CV0200"),
    ("CV0201", "BGCV06_CV0201"),
    ("CV0257", "BGCV11_CV0257"),
    ("CV0911", "BGCV04_CV0911"),
    ("CV0940", "BGCV14_CV0940"),
)
METHODS = (
    "primary",
    "best_residual",
    "pooled_poisson_loglinear_interaction",
    "common_effect_exact_cmle",
    "destroyed_link",
)
PRIMARY_COMPARATORS = (
    "best_residual",
    "common_effect_exact_cmle",
)
VALIDITY_CONTROL = "destroyed_link"
REPORTED_SECONDARY = "pooled_poisson_loglinear_interaction"
BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 2_026_082_902
PUBLIC_OWNER = "sushaan-k"
PUBLIC_REPOSITORY = "coupling-fields-benchmark"
PUBLIC_ORIGIN = f"https://github.com/{PUBLIC_OWNER}/{PUBLIC_REPOSITORY}.git"

BINDING_PATHS = {
    "runner": "experiments/confirm_stephenson_unused_cambridge.py",
    "runner_test": "tests/test_stephenson_unused_cambridge.py",
    "protocol": ("docs/STEPHENSON_UNUSED_CAMBRIDGE_HELD_DONOR_PROTOCOL_2026-08-29.md"),
    "designation": (
        "data/confirmation/stephenson_unused_cambridge/candidate_designation_v1.json"
    ),
    "preaccess": ("results/development/stephenson_unused_cambridge_preaccess_v1.json"),
    "initial_prediction_authorization": (
        "data/confirmation/stephenson_unused_cambridge/prediction_authorization_v1.json"
    ),
    "initial_prediction_attempt": (
        "data/confirmation/stephenson_unused_cambridge/prediction_attempt_v1.json"
    ),
    "prediction_recovery_amendment": (
        "data/confirmation/stephenson_unused_cambridge/"
        "prediction_recovery_amendment_v1_1.json"
    ),
    "prediction_recovery_protocol": (
        "docs/STEPHENSON_UNUSED_CAMBRIDGE_PREDICTION_RECOVERY_AMENDMENT_2026-08-29.md"
    ),
    "classical_fields": (
        "data/development/stephenson_unused_cambridge/classical_fields_v1.json"
    ),
    "classical_audit": (
        "results/development/classical_interaction_baselines_posthoc.json"
    ),
    "source_manifest": ("data/confirmation/stephenson_citeseq/source_manifest_v1.json"),
    "development_result": ("results/development/stephenson_citeseq_development.json"),
    "stephenson_runner": "experiments/confirm_stephenson_citeseq.py",
    "combat_numerics": "experiments/confirm_combat_citeseq.py",
    "conditional_solver": "mapreg/hierarchical_conditional_coupling.py",
    "table_prediction": "mapreg/table_prediction.py",
}


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


def _read_json(path: Path) -> dict[str, Any]:
    return stephenson._read_json(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    stephenson._write_json(path, payload, exclusive=True)


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise PermissionError(
            "public artifact must be inside the repository"
        ) from error


def _bound_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise PermissionError("artifact binding escapes the repository") from error
    return path


def _require_fixed_paths(
    stage: str, bindings: tuple[tuple[str, Path, Path], ...]
) -> None:
    for label, supplied, expected in bindings:
        if supplied.resolve() != expected.resolve():
            raise PermissionError(f"{stage} {label} path is not fixed")


def _immutable_public_bytes(relative: str, commit: str, label: str) -> bytes:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise PermissionError(f"{label} commit is not immutable")
    url = (
        f"https://raw.githubusercontent.com/{PUBLIC_OWNER}/{PUBLIC_REPOSITORY}/"
        f"{commit}/{relative}"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "coupling-fields/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _validate_binding_set(
    authorization: dict[str, Any], implementation_commit: str
) -> None:
    expected = {}
    for label, relative in BINDING_PATHS.items():
        path = _bound_path(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        row = {"path": relative, "sha256": _sha256(path)}
        expected[label] = row
        if (
            _immutable_public_bytes(relative, implementation_commit, label)
            != path.read_bytes()
        ):
            raise PermissionError(f"public implementation differs: {label}")
    if authorization.get("artifact_bindings") != expected:
        raise PermissionError("authorization binding set differs")


def _validate_prediction_authorization(
    path: Path, authorization_commit: str
) -> dict[str, Any]:
    authorization = _read_json(path)
    implementation_commit = authorization.get("public_implementation_commit")
    if (
        authorization.get("schema")
        != "stephenson-unused-cambridge-prediction-authorization/1.1"
        or authorization.get("status") != "ONE_REPLACEMENT_RNA_MARGIN_ACCESS_AUTHORIZED"
        or authorization.get("canonical_origin") != PUBLIC_ORIGIN
        or authorization.get("prediction_attempt_path")
        != _relative(DEFAULT_PREDICTION_ATTEMPT)
        or authorization.get("prediction_output_path") != _relative(DEFAULT_PREDICTION)
        or authorization.get("attempt_kind") != "REPLACEMENT_AFTER_HOST_INTERRUPTION"
        or authorization.get("replacement_ordinal") != 1
        or authorization.get("maximum_replacement_attempts") != 1
        or authorization.get("initial_attempt_sha256")
        != EXPECTED_INITIAL_PREDICTION_ATTEMPT_SHA256
        or authorization.get("recovery_amendment_sha256")
        != EXPECTED_RECOVERY_AMENDMENT_SHA256
        or authorization.get("scientific_design_changed") is not False
        or authorization.get("adt_numeric_access_authorized") is not False
        or authorization.get("rerun_permitted") is not False
        or not isinstance(implementation_commit, str)
    ):
        raise PermissionError("prediction authorization differs")
    _validate_binding_set(authorization, implementation_commit)
    if (
        _immutable_public_bytes(
            _relative(path), authorization_commit, "prediction authorization"
        )
        != path.read_bytes()
    ):
        raise PermissionError("public prediction authorization bytes differ")
    return authorization


def _validate_score_authorization(
    path: Path, prediction_path: Path, authorization_commit: str
) -> dict[str, Any]:
    authorization = _read_json(path)
    prediction_commit = authorization.get("public_prediction_commit")
    if (
        authorization.get("schema")
        != "stephenson-unused-cambridge-score-authorization/1.1"
        or authorization.get("status") != "RECOVERY_AMENDED_PAIRING_ACCESS_AUTHORIZED"
        or authorization.get("canonical_origin") != PUBLIC_ORIGIN
        or authorization.get("prediction_path") != _relative(prediction_path)
        or authorization.get("prediction_sha256") != _sha256(prediction_path)
        or authorization.get("score_attempt_path") != _relative(DEFAULT_SCORE_ATTEMPT)
        or authorization.get("score_output_path") != _relative(DEFAULT_SCORE)
        or authorization.get("recovery_amended") is not True
        or authorization.get("replacement_ordinal") != 1
        or authorization.get("recovery_amendment_sha256")
        != EXPECTED_RECOVERY_AMENDMENT_SHA256
        or authorization.get("rerun_permitted") is not False
        or not isinstance(prediction_commit, str)
    ):
        raise PermissionError("score authorization differs")
    if (
        _immutable_public_bytes(
            _relative(prediction_path), prediction_commit, "frozen prediction"
        )
        != prediction_path.read_bytes()
        or _immutable_public_bytes(
            _relative(path), authorization_commit, "score authorization"
        )
        != path.read_bytes()
    ):
        raise PermissionError("public score boundary bytes differ")
    return authorization


def _validated_source_manifest(path: Path) -> dict[str, Any]:
    if _sha256(path) != EXPECTED_SOURCE_SHA256:
        raise PermissionError("source manifest digest differs")
    payload = _read_json(path)
    if (
        payload.get("schema") != "stephenson-citeseq-source-manifest/1.1"
        or payload.get("status") != "SOURCE_SEALED_OUTCOME_ACCESS_DISABLED"
        or payload.get("h5ad", {}).get("bytes") != OFFICIAL_H5AD_BYTES
        or payload.get("h5ad", {}).get("sha256") != OFFICIAL_H5AD_SHA256
        or payload.get("h5ad", {}).get("official_url") != OFFICIAL_H5AD_URL
    ):
        raise PermissionError("source manifest differs")
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != 103:
        raise PermissionError("source sample panel differs")
    roles = stephenson._assign_roles(samples)
    if any(row.get("role") != roles[row["donor"]] for row in samples):
        raise PermissionError("source roles differ from their frozen allocation")
    return payload


def _unused_records(source: dict[str, Any]) -> list[dict[str, Any]]:
    rows = sorted(
        (row for row in source["samples"] if row["role"] == "unused_source"),
        key=lambda row: row["donor"],
    )
    if tuple((row["donor"], row["sample"]) for row in rows) != UNUSED or any(
        row["site"] != "Cambridge" for row in rows
    ):
        raise PermissionError("unused-Cambridge donor panel differs")
    return rows


def _validated_development(path: Path) -> dict[str, Any]:
    if _sha256(path) != EXPECTED_DEVELOPMENT_SHA256:
        raise PermissionError("development result digest differs")
    payload = _read_json(path)
    if (
        payload.get("schema") != "stephenson-citeseq-development/1.0"
        or payload.get("status") != "PILOT_PASS"
        or payload.get("passes_pilot_gate") is not True
        or payload.get("source_manifest_sha256") != EXPECTED_SOURCE_SHA256
    ):
        raise PermissionError("development result differs")
    models = payload.get("frozen_source_models")
    if not isinstance(models, dict) or not {
        "primary",
        "best_residual",
        "destroyed_link",
    }.issubset(models):
        raise PermissionError("development source fields are absent")
    return payload


def _validated_classical(path: Path, development: dict[str, Any]) -> dict[str, Any]:
    if _sha256(path) != EXPECTED_CLASSICAL_SHA256:
        raise PermissionError("compact classical-field digest differs")
    payload = _read_json(path)
    if (
        payload.get("schema") != "stephenson-unused-cambridge-classical-fields/1.0"
        or payload.get("status") != "FROZEN_SOURCE_ONLY_FIELDS"
        or payload.get("calibration_samples") != development["calibration_samples"]
        or payload.get("pilot_samples") != development["pilot_samples"]
    ):
        raise PermissionError("compact classical fields differ")
    fields = payload.get("fields")
    if not isinstance(fields, dict) or set(fields) != {
        "common_effect_exact_cmle",
        "pooled_poisson_loglinear_interaction",
    }:
        raise PermissionError("classical method panel differs")
    for row in fields.values():
        coordinate = np.asarray(row.get("source_coordinate"), dtype=float)
        if (
            row.get("kind") != "conditional_log_odds"
            or row.get("alpha") != 1.0
            or row.get("boundary_entities") != 0
            or coordinate.shape != (81,)
            or not np.isfinite(coordinate).all()
        ):
            raise PermissionError("a compact classical field is invalid")
    if _sha256(DEFAULT_CLASSICAL_AUDIT) != EXPECTED_CLASSICAL_AUDIT_SHA256:
        raise PermissionError("public classical audit digest differs")
    audit = _read_json(DEFAULT_CLASSICAL_AUDIT)
    study = audit.get("studies", {}).get("stephenson_newcastle_held_site", {})
    if (
        audit.get("schema") != "classical-interaction-baseline-audit/1.0"
        or audit.get("status") != "POST_HOC_NONCONFIRMATORY_BASELINE_AUDIT"
        or audit.get("confirmatory") is not False
        or payload.get("source_audit", {}).get("artifact_sha256")
        != EXPECTED_CLASSICAL_AUDIT_SHA256
        or study.get("splits", {}).get("calibration")
        != payload["calibration_samples"]
        or study.get("splits", {}).get("selection_or_pilot")
        != payload["pilot_samples"]
    ):
        raise PermissionError("public classical audit provenance differs")
    for name, row in fields.items():
        fitted = study.get("fitted_fields", {}).get(name, {})
        selected = study.get("selection", {}).get(name, {})
        log_odds = np.asarray(fitted.get("log_odds"), dtype=float)
        boundary = np.asarray(fitted.get("boundary"), dtype=np.int8)
        if (
            log_odds.shape != (9, 9)
            or boundary.shape != (9, 9)
            or row["source_coordinate"] != log_odds.reshape(-1).tolist()
            or row["boundary_entities"] != int(np.count_nonzero(boundary))
            or row["alpha"] != selected.get("selected_alpha")
        ):
            raise PermissionError(f"compact classical field differs from audit: {name}")
    return payload


def _validated_recovery_lineage(stage: str) -> dict[str, Any]:
    expected = {
        DEFAULT_DESIGNATION: EXPECTED_DESIGNATION_SHA256,
        DEFAULT_PROTOCOL: EXPECTED_PROTOCOL_SHA256,
        DEFAULT_INITIAL_PREDICTION_AUTHORIZATION: (
            EXPECTED_INITIAL_PREDICTION_AUTHORIZATION_SHA256
        ),
        DEFAULT_INITIAL_PREDICTION_ATTEMPT: (
            EXPECTED_INITIAL_PREDICTION_ATTEMPT_SHA256
        ),
        DEFAULT_RECOVERY_AMENDMENT: EXPECTED_RECOVERY_AMENDMENT_SHA256,
    }
    if any(not path.is_file() for path in expected) or any(
        _sha256(path) != digest for path, digest in expected.items()
    ):
        raise PermissionError("prediction-recovery artifact digest differs")
    initial_authorization = _read_json(DEFAULT_INITIAL_PREDICTION_AUTHORIZATION)
    initial_attempt = _read_json(DEFAULT_INITIAL_PREDICTION_ATTEMPT)
    amendment = _read_json(DEFAULT_RECOVERY_AMENDMENT)
    bindings = amendment.get("bindings", {})
    interruption = amendment.get("interruption", {})
    access = interruption.get("conservative_access_bound", {})
    frozen = amendment.get("frozen_scientific_choices", {})
    scope = amendment.get("recovery_scope", {})
    if (
        initial_authorization.get("schema")
        != "stephenson-unused-cambridge-prediction-authorization/1.0"
        or initial_authorization.get("status") != "RNA_MARGIN_ACCESS_AUTHORIZED"
        or initial_authorization.get("prediction_attempt_path")
        != _relative(DEFAULT_INITIAL_PREDICTION_ATTEMPT)
        or initial_authorization.get("prediction_output_path")
        != _relative(DEFAULT_INITIAL_PREDICTION)
        or initial_authorization.get("adt_numeric_access_authorized") is not False
        or initial_authorization.get("rerun_permitted") is not False
        or initial_attempt.get("schema")
        != "stephenson-unused-cambridge-prediction-attempt/1.0"
        or initial_attempt.get("status") != "TERMINAL_ATTEMPT_STARTED"
        or initial_attempt.get("authorization_sha256")
        != EXPECTED_INITIAL_PREDICTION_AUTHORIZATION_SHA256
        or initial_attempt.get("adt_numeric_access_authorized") is not False
        or initial_attempt.get("rerun_permitted") is not False
        or amendment.get("schema")
        != "stephenson-unused-cambridge-prediction-recovery-amendment/1.1"
        or amendment.get("status") != "OUTCOME_BLIND_SINGLE_REPLACEMENT_ELIGIBLE"
        or bindings.get("candidate_designation_v1_sha256")
        != EXPECTED_DESIGNATION_SHA256
        or bindings.get("protocol_v1_sha256") != EXPECTED_PROTOCOL_SHA256
        or bindings.get("initial_prediction_authorization_sha256")
        != EXPECTED_INITIAL_PREDICTION_AUTHORIZATION_SHA256
        or bindings.get("initial_prediction_attempt_sha256")
        != EXPECTED_INITIAL_PREDICTION_ATTEMPT_SHA256
        or interruption.get("reason_code") != "EXECUTION_HOST_TURN_ABORT"
        or access.get("unused_adt_numeric_values_read") != 0
        or access.get("unused_rna_adt_pairings_formed") != 0
        or access.get("unused_truth_tables_formed") != 0
        or frozen.get("changed_after_initial_attempt") is not False
        or scope.get("attempt_kind") != "REPLACEMENT_AFTER_HOST_INTERRUPTION"
        or scope.get("attempt_ordinal") != 2
        or scope.get("replacement_ordinal") != 1
        or scope.get("maximum_replacement_attempts") != 1
        or scope.get("initial_attempt_rerun_permitted") is not False
        or scope.get("replacement_rerun_permitted") is not False
    ):
        raise PermissionError("prediction-recovery lineage differs")

    initial_forbidden = (
        DEFAULT_INITIAL_PREDICTION,
        DEFAULT_INITIAL_SCORE_AUTHORIZATION,
        DEFAULT_INITIAL_SCORE_ATTEMPT,
        DEFAULT_INITIAL_SCORE,
    )
    if any(path.exists() for path in initial_forbidden):
        raise PermissionError("an initial outcome artifact exists")
    if stage == "predict":
        forbidden = (
            DEFAULT_PREDICTION_ATTEMPT,
            DEFAULT_PREDICTION,
            DEFAULT_SCORE_AUTHORIZATION,
            DEFAULT_SCORE_ATTEMPT,
            DEFAULT_SCORE,
        )
        if any(path.exists() for path in forbidden):
            raise PermissionError("a replacement or score artifact already exists")
    elif stage == "score":
        if not DEFAULT_PREDICTION_ATTEMPT.is_file() or not DEFAULT_PREDICTION.is_file():
            raise PermissionError("replacement prediction lineage is incomplete")
        replacement = _read_json(DEFAULT_PREDICTION_ATTEMPT)
        if (
            replacement.get("schema")
            != "stephenson-unused-cambridge-prediction-attempt/1.1"
            or replacement.get("status") != "TERMINAL_REPLACEMENT_ATTEMPT_STARTED"
            or replacement.get("attempt_kind") != "REPLACEMENT_AFTER_HOST_INTERRUPTION"
            or replacement.get("attempt_ordinal") != 2
            or replacement.get("replacement_ordinal") != 1
            or replacement.get("maximum_replacement_attempts") != 1
            or replacement.get("initial_attempt_sha256")
            != EXPECTED_INITIAL_PREDICTION_ATTEMPT_SHA256
            or replacement.get("recovery_amendment_sha256")
            != EXPECTED_RECOVERY_AMENDMENT_SHA256
            or replacement.get("authorization_sha256")
            != _sha256(DEFAULT_PREDICTION_AUTHORIZATION)
            or replacement.get("scientific_design_changed") is not False
            or replacement.get("rna_request_begins_after_this_record") is not True
            or replacement.get("adt_numeric_access_authorized") is not False
            or replacement.get("rerun_permitted") is not False
        ):
            raise PermissionError("replacement prediction attempt differs")
        if DEFAULT_SCORE_ATTEMPT.exists() or DEFAULT_SCORE.exists():
            raise PermissionError("a replacement score artifact already exists")
    else:
        raise ValueError("recovery stage must be predict or score")
    return amendment


@dataclass(frozen=True)
class H5ADInput:
    mode: str
    value: str


def _source_input(local: Path | None, remote: str | None) -> H5ADInput:
    if (local is None) == (remote is None):
        raise ValueError("supply exactly one of local H5AD or official H5AD URL")
    if local is not None:
        path = local.expanduser().resolve()
        if not path.is_file() or path.name != OFFICIAL_H5AD_NAME:
            raise FileNotFoundError(path)
        return H5ADInput("local", str(path))
    if remote != OFFICIAL_H5AD_URL:
        raise PermissionError("remote H5AD URL differs from the freeze")
    return H5ADInput("remote", remote)


def _verify_source_bytes(source: H5ADInput) -> dict[str, Any]:
    if source.mode == "local":
        digest = hashlib.sha256()
        count = 0
        with Path(source.value).open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
                count += len(block)
        observed = digest.hexdigest()
        if count != OFFICIAL_H5AD_BYTES or observed != OFFICIAL_H5AD_SHA256:
            raise PermissionError("local H5AD bytes differ from the source seal")
        return {
            "mode": "local",
            "bytes": count,
            "sha256": observed,
            "sha256_provenance": "recomputed_full_local_stream",
        }
    if source.mode != "remote" or source.value != OFFICIAL_H5AD_URL:
        raise PermissionError("remote H5AD URL differs from the source seal")
    request = urllib.request.Request(
        source.value,
        method="HEAD",
        headers={"User-Agent": "coupling-fields/1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        resolved_url = response.geturl()
        content_length = response.headers.get("Content-Length")
        accept_ranges = response.headers.get("Accept-Ranges")
        etag = response.headers.get("ETag")
        last_modified = response.headers.get("Last-Modified")
    if (
        resolved_url != OFFICIAL_H5AD_RESOLVED_URL
        or content_length is None
        or int(content_length) != OFFICIAL_H5AD_BYTES
        or accept_ranges is None
        or accept_ranges.strip().lower() != "bytes"
        or etag != OFFICIAL_H5AD_ETAG
        or last_modified != OFFICIAL_H5AD_LAST_MODIFIED
    ):
        raise PermissionError("remote H5AD identity differs from the source seal")
    return {
        "mode": "remote",
        "request_url": source.value,
        "resolved_url": resolved_url,
        "bytes": OFFICIAL_H5AD_BYTES,
        "accept_ranges": "bytes",
        "etag": etag,
        "last_modified": last_modified,
        "sha256": OFFICIAL_H5AD_SHA256,
        "sha256_provenance": "checksum-bound source manifest",
    }


@contextmanager
def _open_h5ad(source: H5ADInput) -> Iterator[h5py.File]:
    if source.mode == "local":
        with h5py.File(source.value, "r") as handle:
            yield handle
        return
    remote = fsspec.open(
        OFFICIAL_H5AD_RESOLVED_URL,
        mode="rb",
        block_size=8 << 20,
        cache_type="readahead",
    )
    with remote as stream, h5py.File(stream, "r") as handle:
        yield handle


def _selected_rows_from_handle(
    handle: h5py.File, records: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    obs = handle.get("obs")
    if not isinstance(obs, h5py.Group):
        raise ValueError("H5AD obs dataframe is absent")
    barcodes = stephenson._dataframe_index(handle, obs)
    sample_ids = stephenson._encoded_column(handle, obs, "sample_id")
    sites = stephenson._encoded_column(handle, obs, "Site")
    clusters = stephenson._encoded_column(handle, obs, "initial_clustering")
    if len(set(barcodes.tolist())) != len(barcodes):
        raise ValueError("H5AD observation names are not unique")
    mapped = np.asarray(
        [stephenson.CLUSTER_TO_CELL_TYPE.get(value, "") for value in clusters]
    )
    if np.any(mapped == ""):
        raise PermissionError("unmapped cluster label entered selection")
    selected = {}
    for record in records:
        rows = np.flatnonzero(sample_ids == record["sample"])
        if not len(rows) or set(sites[rows].tolist()) != {record["site"]}:
            raise PermissionError(f"sample metadata differs: {record['sample']}")
        eligible = rows[mapped[rows] != ""]
        if (
            len(eligible) != record["eligible_pool_cells"]
            or len(eligible) < stephenson.CELL_BUDGET
        ):
            raise PermissionError(f"eligible pool differs: {record['sample']}")
        ordered = sorted(
            eligible.tolist(),
            key=lambda row: (
                stephenson._cell_hash(record["donor"], record["sample"], barcodes[row]),
                barcodes[row],
            ),
        )[: stephenson.CELL_BUDGET]
        matrix_rows = np.asarray(sorted(ordered), dtype=np.int64)
        chosen = barcodes[matrix_rows]
        selected[record["sample"]] = {
            "rows": matrix_rows,
            "barcodes": chosen,
            "selected_barcode_sha256": hashlib.sha256(
                ("\n".join(sorted(chosen.tolist())) + "\n").encode()
            ).hexdigest(),
        }
    return selected


def _read_modality_from_handle(
    handle: h5py.File,
    selections: dict[str, dict[str, Any]],
    samples: tuple[str, ...],
    modality: str,
) -> dict[str, np.ndarray]:
    if modality not in {"rna", "adt"}:
        raise ValueError("modality must be rna or adt")
    all_rows = np.concatenate([selections[sample]["rows"] for sample in samples])
    row_order = np.argsort(all_rows, kind="mergesort")
    sorted_rows = all_rows[row_order]
    if np.any(np.diff(sorted_rows) == 0):
        raise ValueError("selected sample rows overlap")
    columns = np.asarray(stephenson._feature_columns(handle)[modality], dtype=np.int64)
    matrix = handle.get("layers/raw")
    if not isinstance(matrix, (h5py.Group, h5py.Dataset)):
        raise ValueError("raw matrix is absent")
    values = stephenson.numerics._read_csr_feature_subset(matrix, sorted_rows, columns)
    unsorted = np.empty_like(values)
    unsorted[row_order] = values
    result = {}
    offset = 0
    for sample in samples:
        result[sample] = unsorted[offset : offset + stephenson.CELL_BUDGET].T
        offset += stephenson.CELL_BUDGET
    return result


def _models(
    development: dict[str, Any], classical: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    frozen = development["frozen_source_models"]
    models = {
        "primary": frozen["primary"],
        "best_residual": frozen["best_residual"],
        "destroyed_link": frozen["destroyed_link"],
    }
    for name, row in classical["fields"].items():
        models[name] = {
            "kind": row["kind"],
            "estimator": row["estimator"],
            "source_coordinate": row["source_coordinate"],
            "alpha": row["alpha"],
        }
    if set(models) != set(METHODS):
        raise PermissionError("frozen method panel differs")
    return models


def verify_preaccess(
    output_path: Path = DEFAULT_PREACCESS, check_existing: bool = False
) -> dict[str, Any]:
    _require_fixed_paths("preaccess", (("output", output_path, DEFAULT_PREACCESS),))
    if output_path.exists() and not check_existing:
        raise FileExistsError("preaccess result is one-shot")
    source = _validated_source_manifest(DEFAULT_SOURCE)
    records = _unused_records(source)
    development = _validated_development(DEFAULT_DEVELOPMENT)
    classical = _validated_classical(DEFAULT_CLASSICAL, development)
    original_prediction = _read_json(stephenson.DEFAULT_PREDICTION)
    original_score = _read_json(stephenson.DEFAULT_SCORE)
    if (
        _sha256(stephenson.DEFAULT_PREDICTION) != EXPECTED_ORIGINAL_PREDICTION_SHA256
        or _sha256(stephenson.DEFAULT_SCORE) != EXPECTED_ORIGINAL_SCORE_SHA256
    ):
        raise PermissionError("original held artifacts differ")
    unused_samples = {row["sample"] for row in records}
    prior_samples = set(development["calibration_samples"])
    prior_samples.update(development["pilot_samples"])
    prior_samples.update(row["sample"] for row in original_prediction["samples"])
    if unused_samples & prior_samples:
        raise PermissionError("a prior numeric panel contains an unused donor")
    if (
        unused_samples & set(classical["calibration_samples"])
        or unused_samples & set(classical["pilot_samples"])
        or original_score.get("panel") != "Newcastle held site"
        or original_score.get("donors") != 56
    ):
        raise PermissionError("prior result roles differ")
    payload = {
        "schema": "stephenson-unused-cambridge-preaccess/1.0",
        "status": "PASS_H5AD_UNOPENED",
        "created_at_utc": _timestamp(),
        "donors": [row["donor"] for row in records],
        "samples": [row["sample"] for row in records],
        "bindings": {
            "source_manifest_sha256": EXPECTED_SOURCE_SHA256,
            "development_sha256": EXPECTED_DEVELOPMENT_SHA256,
            "classical_fields_sha256": EXPECTED_CLASSICAL_SHA256,
            "classical_audit_sha256": EXPECTED_CLASSICAL_AUDIT_SHA256,
            "original_prediction_sha256": EXPECTED_ORIGINAL_PREDICTION_SHA256,
            "original_score_sha256": EXPECTED_ORIGINAL_SCORE_SHA256,
        },
        "prior_access_audits": {
            "development": development["access_audit"],
            "prediction": original_prediction["access_audit"],
            "score": original_score["access_audit"],
        },
        "matrix_values_read_by_this_check": 0,
        "h5ad_opened_by_this_check": False,
        "unused_rna_values_previously_recorded": 0,
        "unused_adt_values_previously_recorded": 0,
        "unused_pairings_previously_recorded": 0,
    }
    if check_existing:
        existing = _read_json(output_path)
        expected = dict(payload)
        expected["created_at_utc"] = existing.get("created_at_utc")
        if existing != expected:
            raise PermissionError("committed preaccess result differs")
        return existing
    _write_json(output_path, payload)
    return payload


def _prediction_rows(
    records: list[dict[str, Any]],
    selections: dict[str, dict[str, Any]],
    rna_counts: dict[str, np.ndarray],
    models: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    fixed_adt = np.tile(np.asarray([[256, 256]], dtype=np.int64), (9, 1))
    rows = []
    for record in records:
        sample = record["sample"]
        states = (
            stephenson.numerics._integer_counts(rna_counts[sample], "RNA") > 0
        ).astype(np.uint8)
        rna = np.stack((512 - states.sum(axis=1), states.sum(axis=1)), axis=1)
        rows.append(
            {
                "donor": record["donor"],
                "sample": sample,
                "rna_margins": rna.tolist(),
                "adt_margins": fixed_adt.tolist(),
                "selected_barcode_sha256": selections[sample][
                    "selected_barcode_sha256"
                ],
                "predictions": {
                    name: stephenson._predict_method(model, rna, fixed_adt)
                    .reshape(81, 4)
                    .tolist()
                    for name, model in models.items()
                },
            }
        )
    return rows


def predict(
    source_input: H5ADInput,
    authorization_path: Path,
    authorization_commit: str,
    attempt_path: Path = DEFAULT_PREDICTION_ATTEMPT,
    output_path: Path = DEFAULT_PREDICTION,
) -> dict[str, Any]:
    _require_fixed_paths(
        "prediction",
        (
            ("authorization", authorization_path, DEFAULT_PREDICTION_AUTHORIZATION),
            ("attempt", attempt_path, DEFAULT_PREDICTION_ATTEMPT),
            ("output", output_path, DEFAULT_PREDICTION),
        ),
    )
    if (
        attempt_path.exists()
        or output_path.exists()
        or any(
            path.exists()
            for path in (
                DEFAULT_SCORE_AUTHORIZATION,
                DEFAULT_SCORE_ATTEMPT,
                DEFAULT_SCORE,
            )
        )
    ):
        raise FileExistsError("replacement prediction campaign is one-shot")
    recovery = _validated_recovery_lineage("predict")
    authorization = _validate_prediction_authorization(
        authorization_path, authorization_commit
    )
    source_audit = _verify_source_bytes(source_input)
    source = _validated_source_manifest(DEFAULT_SOURCE)
    records = _unused_records(source)
    development = _validated_development(DEFAULT_DEVELOPMENT)
    classical = _validated_classical(DEFAULT_CLASSICAL, development)
    _write_json(
        attempt_path,
        {
            "schema": "stephenson-unused-cambridge-prediction-attempt/1.1",
            "status": "TERMINAL_REPLACEMENT_ATTEMPT_STARTED",
            "created_at_utc": _timestamp(),
            "authorization_sha256": _sha256(authorization_path),
            "attempt_kind": "REPLACEMENT_AFTER_HOST_INTERRUPTION",
            "attempt_ordinal": 2,
            "replacement_ordinal": 1,
            "maximum_replacement_attempts": 1,
            "initial_attempt_sha256": EXPECTED_INITIAL_PREDICTION_ATTEMPT_SHA256,
            "recovery_amendment_sha256": EXPECTED_RECOVERY_AMENDMENT_SHA256,
            "scientific_design_changed": False,
            "source_audit": source_audit,
            "rna_request_begins_after_this_record": True,
            "adt_numeric_access_authorized": False,
            "rerun_permitted": False,
        },
    )
    with _open_h5ad(source_input) as handle:
        selections = _selected_rows_from_handle(handle, records)
    samples = tuple(row["sample"] for row in records)
    with _open_h5ad(source_input) as handle:
        rna_counts = _read_modality_from_handle(handle, selections, samples, "rna")
    rows = _prediction_rows(
        records, selections, rna_counts, _models(development, classical)
    )
    payload = {
        "schema": "stephenson-unused-cambridge-predictions/1.1",
        "status": "FROZEN_PREDICTIONS",
        "created_at_utc": _timestamp(),
        "authorization_sha256": _sha256(authorization_path),
        "public_authorization_commit": authorization_commit,
        "public_implementation_commit": authorization["public_implementation_commit"],
        "attempt_sha256": _sha256(attempt_path),
        "attempt_kind": "REPLACEMENT_AFTER_HOST_INTERRUPTION",
        "attempt_ordinal": 2,
        "replacement_ordinal": 1,
        "maximum_replacement_attempts": 1,
        "initial_attempt_sha256": EXPECTED_INITIAL_PREDICTION_ATTEMPT_SHA256,
        "recovery_amendment_sha256": EXPECTED_RECOVERY_AMENDMENT_SHA256,
        "scientific_design_changed": False,
        "source_audit": source_audit,
        "source_manifest_sha256": EXPECTED_SOURCE_SHA256,
        "development_sha256": EXPECTED_DEVELOPMENT_SHA256,
        "classical_fields_sha256": EXPECTED_CLASSICAL_SHA256,
        "classical_audit_sha256": EXPECTED_CLASSICAL_AUDIT_SHA256,
        "runner_sha256": _sha256(Path(__file__)),
        "protocol_sha256": _sha256(DEFAULT_PROTOCOL),
        "recovery_protocol_sha256": _sha256(DEFAULT_RECOVERY_PROTOCOL),
        "recovery_status": recovery["status"],
        "methods": list(METHODS),
        "donors": 11,
        "samples": rows,
        "access_audit": {
            "metadata_handles_opened": 1,
            "rna_handles_opened": 1,
            "adt_handles_opened": 0,
            "requested_rna_coordinates_materialized": 11 * 9 * 512,
            "requested_adt_coordinates_materialized": 0,
            "rna_adt_pairings_formed": 0,
            "truth_tables_formed": 0,
            "cell_vectors_serialized": False,
        },
    }
    _write_json(output_path, payload)
    return payload


def _validate_prediction(
    path: Path, development: dict[str, Any], classical: dict[str, Any]
) -> dict[str, Any]:
    payload = _read_json(path)
    if (
        payload.get("schema") != "stephenson-unused-cambridge-predictions/1.1"
        or payload.get("status") != "FROZEN_PREDICTIONS"
        or payload.get("attempt_sha256") != _sha256(DEFAULT_PREDICTION_ATTEMPT)
        or payload.get("source_manifest_sha256") != EXPECTED_SOURCE_SHA256
        or payload.get("development_sha256") != EXPECTED_DEVELOPMENT_SHA256
        or payload.get("classical_fields_sha256") != EXPECTED_CLASSICAL_SHA256
        or payload.get("classical_audit_sha256") != EXPECTED_CLASSICAL_AUDIT_SHA256
        or payload.get("runner_sha256") != _sha256(Path(__file__))
        or payload.get("protocol_sha256") != _sha256(DEFAULT_PROTOCOL)
        or payload.get("recovery_protocol_sha256") != _sha256(DEFAULT_RECOVERY_PROTOCOL)
        or payload.get("recovery_status") != "OUTCOME_BLIND_SINGLE_REPLACEMENT_ELIGIBLE"
        or payload.get("attempt_kind") != "REPLACEMENT_AFTER_HOST_INTERRUPTION"
        or payload.get("attempt_ordinal") != 2
        or payload.get("replacement_ordinal") != 1
        or payload.get("maximum_replacement_attempts") != 1
        or payload.get("initial_attempt_sha256")
        != EXPECTED_INITIAL_PREDICTION_ATTEMPT_SHA256
        or payload.get("recovery_amendment_sha256")
        != EXPECTED_RECOVERY_AMENDMENT_SHA256
        or payload.get("scientific_design_changed") is not False
        or payload.get("methods") != list(METHODS)
        or payload.get("donors") != 11
        or payload.get("access_audit", {}).get("adt_handles_opened") != 0
        or payload.get("access_audit", {}).get("rna_adt_pairings_formed") != 0
        or payload.get("access_audit", {}).get("truth_tables_formed") != 0
    ):
        raise PermissionError("frozen prediction header differs")
    rows = payload.get("samples")
    if not isinstance(rows, list) or len(rows) != 11:
        raise PermissionError("frozen prediction panel differs")
    models = _models(development, classical)
    fixed_adt = np.tile(np.asarray([[256, 256]], dtype=np.int64), (9, 1))
    for row, (donor, sample) in zip(rows, UNUSED):
        if row.get("donor") != donor or row.get("sample") != sample:
            raise PermissionError("frozen prediction donor order differs")
        selected_digest = row.get("selected_barcode_sha256")
        if not isinstance(selected_digest, str) or not re.fullmatch(
            r"[0-9a-f]{64}", selected_digest
        ):
            raise PermissionError("frozen selected-cell digest differs")
        rna = np.asarray(row.get("rna_margins"), dtype=np.int64)
        adt = np.asarray(row.get("adt_margins"), dtype=np.int64)
        if (
            rna.shape != (9, 2)
            or np.any(rna.sum(axis=1) != 512)
            or not np.array_equal(adt, fixed_adt)
        ):
            raise PermissionError("frozen prediction margins differ")
        predicted = row.get("predictions")
        if not isinstance(predicted, dict) or set(predicted) != set(METHODS):
            raise PermissionError("frozen prediction methods differ")
        for name, model in models.items():
            expected = stephenson._predict_method(model, rna, fixed_adt).reshape(81, 4)
            observed = np.asarray(predicted[name], dtype=float)
            if observed.shape != expected.shape or not np.allclose(
                observed, expected, rtol=0.0, atol=1e-12
            ):
                raise PermissionError(f"frozen prediction changed: {name}")
    return payload


def _comparison(
    donors: list[str], primary: np.ndarray, comparator: np.ndarray, label: str
) -> dict[str, Any]:
    first = np.asarray(primary, dtype=float)
    second = np.asarray(comparator, dtype=float)
    if (
        first.shape != (11,)
        or second.shape != first.shape
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
        or second.mean() <= 0
    ):
        raise ValueError("paired donor losses are invalid")
    difference = first - second
    seed = int.from_bytes(
        hashlib.sha256(f"{BOOTSTRAP_SEED}|{label}".encode()).digest()[:8],
        "little",
    )
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, 11, size=(BOOTSTRAPS, 11))
    interval = np.quantile(difference[indices].mean(axis=1), [0.025, 0.975])
    favorable = int(np.count_nonzero(difference < 0))
    bits = (np.arange(2**11, dtype=np.uint16)[:, None] >> np.arange(11)) & 1
    signs = np.where(bits == 0, 1.0, -1.0)
    observed_mean = float(difference.mean())
    permuted_means = (signs * difference).mean(axis=1)
    sign_flip_p = float(np.count_nonzero(permuted_means <= observed_mean)) / float(
        2**11
    )
    relative = 1.0 - float(first.mean() / second.mean())
    passes_effect_and_ci = bool(relative >= 0.05 and interval[1] < 0)
    passes_primary_gate = bool(passes_effect_and_ci and sign_flip_p <= 0.05)
    return {
        "primary_mean_loss": float(first.mean()),
        "comparator_mean_loss": float(second.mean()),
        "relative_reduction": relative,
        "paired_difference_95_ci": interval.tolist(),
        "bootstrap_draws": BOOTSTRAPS,
        "bootstrap_seed": seed,
        "bootstrap_unit": "physical donor",
        "favorable_donors": favorable,
        "exact_one_sided_paired_sign_flip_p": sign_flip_p,
        "passes_effect_and_ci": passes_effect_and_ci,
        "passes_primary_gate": passes_primary_gate,
        "donor_differences_primary_minus_comparator": dict(
            zip(donors, difference.tolist())
        ),
    }


def _confirmation_gates(
    comparisons: dict[str, dict[str, Any]],
) -> dict[str, bool]:
    field_pass = bool(
        comparisons["best_residual"]["passes_primary_gate"]
        and comparisons[VALIDITY_CONTROL]["passes_effect_and_ci"]
    )
    hierarchy_pass = bool(
        comparisons["common_effect_exact_cmle"]["passes_primary_gate"]
    )
    return {
        "passes_field_transfer": field_pass,
        "passes_hierarchical_increment": hierarchy_pass,
        "passes_full_confirmation": field_pass and hierarchy_pass,
    }


def score(
    source_input: H5ADInput,
    authorization_path: Path,
    authorization_commit: str,
    prediction_path: Path = DEFAULT_PREDICTION,
    attempt_path: Path = DEFAULT_SCORE_ATTEMPT,
    output_path: Path = DEFAULT_SCORE,
) -> dict[str, Any]:
    _require_fixed_paths(
        "score",
        (
            ("authorization", authorization_path, DEFAULT_SCORE_AUTHORIZATION),
            ("prediction", prediction_path, DEFAULT_PREDICTION),
            ("attempt", attempt_path, DEFAULT_SCORE_ATTEMPT),
            ("output", output_path, DEFAULT_SCORE),
        ),
    )
    if attempt_path.exists() or output_path.exists():
        raise FileExistsError("score campaign is one-shot")
    recovery = _validated_recovery_lineage("score")
    authorization = _validate_score_authorization(
        authorization_path, prediction_path, authorization_commit
    )
    source_audit = _verify_source_bytes(source_input)
    source = _validated_source_manifest(DEFAULT_SOURCE)
    records = _unused_records(source)
    development = _validated_development(DEFAULT_DEVELOPMENT)
    classical = _validated_classical(DEFAULT_CLASSICAL, development)
    prediction = _validate_prediction(prediction_path, development, classical)
    _write_json(
        attempt_path,
        {
            "schema": "stephenson-unused-cambridge-score-attempt/1.1",
            "status": "TERMINAL_ATTEMPT_STARTED",
            "created_at_utc": _timestamp(),
            "authorization_sha256": _sha256(authorization_path),
            "prediction_sha256": _sha256(prediction_path),
            "prediction_attempt_sha256": _sha256(DEFAULT_PREDICTION_ATTEMPT),
            "recovery_amended": True,
            "replacement_ordinal": 1,
            "initial_attempt_sha256": EXPECTED_INITIAL_PREDICTION_ATTEMPT_SHA256,
            "recovery_amendment_sha256": EXPECTED_RECOVERY_AMENDMENT_SHA256,
            "source_audit": source_audit,
            "pairing_request_begins_after_this_record": True,
            "rerun_permitted": False,
        },
    )
    with _open_h5ad(source_input) as handle:
        selections = _selected_rows_from_handle(handle, records)
    samples = tuple(row["sample"] for row in records)
    with _open_h5ad(source_input) as handle:
        rna_counts = _read_modality_from_handle(handle, selections, samples, "rna")
    with _open_h5ad(source_input) as handle:
        adt_counts = _read_modality_from_handle(handle, selections, samples, "adt")
    rna_states = {
        sample: (
            stephenson.numerics._integer_counts(rna_counts[sample], "RNA") > 0
        ).astype(np.uint8)
        for sample in samples
    }
    by_sample = {row["sample"]: row for row in records}
    adt_states = {
        sample: stephenson._adt_states(
            adt_counts[sample],
            selections[sample]["barcodes"],
            by_sample[sample]["donor"],
            sample,
        )
        for sample in samples
    }
    truth = stephenson.numerics._form_tables(rna_states, adt_states, list(samples))
    frozen = {row["sample"]: row for row in prediction["samples"]}
    losses = {name: np.empty(11, dtype=float) for name in METHODS}
    donor_rows = []
    for index, (record, observed) in enumerate(zip(records, truth)):
        row = frozen[record["sample"]]
        if (
            row["selected_barcode_sha256"]
            != selections[record["sample"]]["selected_barcode_sha256"]
            or not np.array_equal(observed.sum(axis=-1)[:, 0, :], row["rna_margins"])
            or not np.array_equal(observed.sum(axis=-2)[0, :, :], row["adt_margins"])
        ):
            raise PermissionError("scored donor differs from frozen prediction")
        support = stephenson.numerics._informative(observed).reshape(-1)
        if int(support.sum()) < 64:
            raise ValueError(f"fewer than 64 informative pairs: {record['donor']}")
        row_losses = {}
        for name in METHODS:
            fitted = np.asarray(row["predictions"][name], dtype=float).reshape(
                9, 9, 2, 2
            )
            value = stephenson.numerics._donor_loss(observed, fitted, support)
            losses[name][index] = value
            row_losses[name] = float(value)
        donor_rows.append(
            {
                "donor": record["donor"],
                "sample": record["sample"],
                "informative_pairs": int(support.sum()),
                "losses": row_losses,
            }
        )
    donors = [row["donor"] for row in records]
    comparisons = {
        name: _comparison(donors, losses["primary"], losses[name], name)
        for name in METHODS[1:]
    }
    gates = _confirmation_gates(comparisons)
    full_pass = gates["passes_full_confirmation"]
    payload = {
        "schema": "stephenson-unused-cambridge-confirmation/1.1",
        "status": "CONFIRMATION_PASS" if full_pass else "CONFIRMATION_FAIL",
        "created_at_utc": _timestamp(),
        "authorization_sha256": _sha256(authorization_path),
        "public_authorization_commit": authorization_commit,
        "public_prediction_commit": authorization["public_prediction_commit"],
        "recovery_amended": True,
        "recovery_status": recovery["status"],
        "replacement_ordinal": 1,
        "initial_attempt_sha256": EXPECTED_INITIAL_PREDICTION_ATTEMPT_SHA256,
        "recovery_amendment_sha256": EXPECTED_RECOVERY_AMENDMENT_SHA256,
        "prediction_sha256": _sha256(prediction_path),
        "prediction_attempt_sha256": _sha256(DEFAULT_PREDICTION_ATTEMPT),
        "attempt_sha256": _sha256(attempt_path),
        "source_audit": source_audit,
        "comparisons": comparisons,
        "mandatory_reported_secondary": REPORTED_SECONDARY,
        **gates,
        "donor_results": donor_rows,
        "access_audit": {
            "metadata_handles_opened": 1,
            "rna_handles_opened": 1,
            "adt_handles_opened": 1,
            "donors_scored": 11,
            "paired_truth_access_after_public_prediction": True,
            "terminal_attempt_preceded_first_h5ad_open": True,
            "cell_vectors_serialized": False,
            "rerun_permitted": False,
        },
    }
    _write_json(output_path, payload)
    return payload


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--h5ad", type=Path)
    source.add_argument("--h5ad-url")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    preaccess = subparsers.add_parser("verify-preaccess")
    preaccess.add_argument("--output", type=Path, default=DEFAULT_PREACCESS)
    preaccess.add_argument("--check-existing", action="store_true")
    prediction = subparsers.add_parser("predict")
    _add_source_arguments(prediction)
    prediction.add_argument(
        "--authorization", type=Path, default=DEFAULT_PREDICTION_AUTHORIZATION
    )
    prediction.add_argument("--authorization-commit", required=True)
    prediction.add_argument("--attempt", type=Path, default=DEFAULT_PREDICTION_ATTEMPT)
    prediction.add_argument("--output", type=Path, default=DEFAULT_PREDICTION)
    scoring = subparsers.add_parser("score")
    _add_source_arguments(scoring)
    scoring.add_argument(
        "--authorization", type=Path, default=DEFAULT_SCORE_AUTHORIZATION
    )
    scoring.add_argument("--authorization-commit", required=True)
    scoring.add_argument("--prediction", type=Path, default=DEFAULT_PREDICTION)
    scoring.add_argument("--attempt", type=Path, default=DEFAULT_SCORE_ATTEMPT)
    scoring.add_argument("--output", type=Path, default=DEFAULT_SCORE)
    args = parser.parse_args()
    if args.phase == "verify-preaccess":
        payload = verify_preaccess(args.output, args.check_existing)
    elif args.phase == "predict":
        payload = predict(
            _source_input(args.h5ad, args.h5ad_url),
            args.authorization,
            args.authorization_commit,
            args.attempt,
            args.output,
        )
    else:
        payload = score(
            _source_input(args.h5ad, args.h5ad_url),
            args.authorization,
            args.authorization_commit,
            args.prediction,
            args.attempt,
            args.output,
        )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
