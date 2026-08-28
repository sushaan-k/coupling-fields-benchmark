"""Fail-closed held-donor confirmation for SCMMIB BMMC RNA--ADT.

``predict`` packages an independently audited non-held development result.
``score`` is a hash-authorized one-shot held-outcome operation. The disabled
source manifest shipped with this repository makes both phases refuse before
feature-level access until the versioned complete CITE H5AD is acquired and
bound.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/SCMMIB_BMMC_HELD_DONOR_CONFIRMATION_PROTOCOL_2026-08-28.md"
PREFLIGHT = ROOT / "results/development/scmmib_bmmc_metadata_preflight.json"
SOURCE_TEMPLATE = (
    ROOT / "data/confirmation/scmmib_bmmc/source_manifest_template_v1.json"
)
SOURCE_MANIFEST = ROOT / "data/confirmation/scmmib_bmmc/source_manifest_v1.json"
DEVELOPMENT_RESULT = ROOT / "results/development/scmmib_bmmc_exact_development.json"
PREDICTION = ROOT / "results/scmmib_bmmc_exact_predictions.json"
AUTH_TEMPLATE = (
    ROOT / "data/confirmation/scmmib_bmmc/score_authorization_template_v1.json"
)
AUTHORIZATION = ROOT / "data/confirmation/scmmib_bmmc/score_authorization_v1.json"
SCORE_ATTEMPT = ROOT / "data/confirmation/scmmib_bmmc/score_attempt_v1.json"
OUTPUT = ROOT / "results/scmmib_bmmc_exact_confirmation.json"
REFUSAL = ROOT / "results/scmmib_bmmc_exact_score_refusal.json"
SANGER_TERMINAL_ARTIFACTS = (
    ROOT / "data/confirmation/scmmib_sanger/score_attempt_v1.json",
    ROOT / "results/scmmib_sanger_exact_confirmation.json",
    ROOT / "results/scmmib_sanger_exact_score_refusal.json",
)

FIT_DONORS = ("11466", "19593")
DEVELOPMENT_DONORS = ("15078",)
HELD_DONORS = ("10886", "12710", "13272", "16710", "18303", "28045")
MARKERS = ("CD4", "CD7", "CD14", "CD19", "CD33", "CD38", "CD44", "CD47", "CD52", "CD93")
REQUIRED_METHODS = (
    "primary",
    "best_residual",
    "destroyed_link",
    "hierarchical_ridge_only",
    "common_effect_graph",
    "common_effect_ridge_only",
    "label_permuted_graph",
    "independence",
)
GATE_COMPARATORS = (
    "best_residual",
    "destroyed_link",
    "hierarchical_ridge_only",
    "common_effect_graph",
)
COMPLETE_CITE_BYTES = 624_797_386
COMPLETE_CITE_BUCKET = "openproblems-bio"
COMPLETE_CITE_KEY = (
    "public/phase2-private-data/common/openproblems_bmmc_complete/"
    "openproblems_bmmc_cite_complete.h5ad"
)
COMPLETE_CITE_VERSION = "kyN5dZPIsYJ0NC8Y5ECK55TuiebegCII"
COMPLETE_CITE_ETAG = "15d0db3fb12efb77160e293cdeb98e11-75"
METADATA_SHA256 = "b267d4a820b062d0a05227c9cab61d389dcf924c3a6e062fb2389ce1be2f6e4f"
MINIMUM_INFORMATIVE_ENTITIES = 80
BOOTSTRAPS = 20_000
SEED = 20260828


@dataclass(frozen=True)
class _ScorePermit:
    prediction_sha256: str
    public_commit: str


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _bound_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PermissionError(f"{label} path is not bound")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PermissionError(f"{label} path must be repository-relative")
    resolved = (ROOT / candidate).resolve()
    resolved.relative_to(ROOT.resolve())
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} file is absent")
    return resolved


def _validate_hash(path: Path, expected: object, label: str) -> None:
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise PermissionError(f"{label} SHA-256 is not bound")
    if _sha256(path) != expected:
        raise PermissionError(f"{label} SHA-256 differs")


def _validated_source() -> tuple[dict[str, object], dict[str, Path]]:
    if not SOURCE_MANIFEST.is_file():
        raise PermissionError(
            "active source manifest is absent; complete CITE H5AD remains unavailable"
        )
    source = _read_json(SOURCE_MANIFEST)
    if source.get("schema") != "scmmib-bmmc-source/1.0":
        raise PermissionError("source manifest schema differs")
    if source.get("status") != "NONHELD_SOURCE_ACCESS_AUTHORIZED":
        raise PermissionError("non-held source access is disabled")
    if source.get("held_feature_rows_may_be_read") is not False:
        raise PermissionError("source manifest does not forbid held feature rows")
    complete_record = source.get("complete_cite_h5ad")
    assays = source.get("internal_assays")
    metadata_record = source.get("metadata")
    preflight_record = source.get("preflight")
    if not all(
        isinstance(value, dict)
        for value in (complete_record, assays, metadata_record, preflight_record)
    ):
        raise PermissionError("source manifest records are incomplete")
    complete = _bound_path(complete_record.get("local_path"), "complete CITE H5AD")
    identity = {
        "bytes": COMPLETE_CITE_BYTES,
        "s3_bucket": COMPLETE_CITE_BUCKET,
        "s3_key": COMPLETE_CITE_KEY,
        "version_id": COMPLETE_CITE_VERSION,
        "etag": COMPLETE_CITE_ETAG,
    }
    if complete.stat().st_size != COMPLETE_CITE_BYTES or any(
        complete_record.get(key) != value for key, value in identity.items()
    ):
        raise PermissionError("complete CITE H5AD object identity differs")
    _validate_hash(complete, complete_record.get("sha256"), "complete CITE H5AD")
    paths = {"complete_cite_h5ad": complete}
    for modality in ("rna", "adt"):
        record = assays.get(modality)
        if (
            not isinstance(record, dict)
            or record.get("matrix_is_raw_counts") is not True
        ):
            raise PermissionError(f"{modality} is not certified as raw counts")
        internal = (
            record.get("obs_index_hdf5_path"),
            record.get("feature_index_hdf5_path"),
            record.get("matrix_hdf5_path"),
        )
        if any(not isinstance(value, str) or not value for value in internal):
            raise PermissionError(f"{modality} internal HDF5 paths are not bound")
    metadata = _bound_path(metadata_record.get("local_path"), "metadata")
    if metadata_record.get("sha256") != METADATA_SHA256:
        raise PermissionError("metadata is not bound to the preflight source")
    _validate_hash(metadata, metadata_record.get("sha256"), "metadata")
    preflight = _bound_path(preflight_record.get("path"), "preflight")
    _validate_hash(preflight, preflight_record.get("sha256"), "preflight")
    paths.update(metadata=metadata, preflight=preflight)
    return source, paths


def _metadata_roles(path: Path) -> tuple[list[dict[str, str]], dict[str, list[str]]]:
    with gzip.open(path, "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"barcode", "DonorID", "Site", "batch", "cell_type.l1", "is_train"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("BMMC metadata lacks required columns")
    if len({row["barcode"] for row in rows}) != len(rows):
        raise ValueError("BMMC metadata barcodes are not unique")
    roles = {
        "fit": [row["barcode"] for row in rows if row["DonorID"] in FIT_DONORS],
        "development": [
            row["barcode"] for row in rows if row["DonorID"] in DEVELOPMENT_DONORS
        ],
        "held": [row["barcode"] for row in rows if row["DonorID"] in HELD_DONORS],
    }
    assigned = set().union(*(set(value) for value in roles.values()))
    if len(assigned) != len(rows) or sum(map(len, roles.values())) != len(rows):
        raise ValueError("physical-donor roles do not partition metadata")
    donors = {
        role: sorted(
            {row["DonorID"] for row in rows if row["barcode"] in set(barcodes)}
        )
        for role, barcodes in roles.items()
    }
    if donors != {
        "fit": list(FIT_DONORS),
        "development": list(DEVELOPMENT_DONORS),
        "held": list(HELD_DONORS),
    }:
        raise ValueError("metadata does not reproduce the locked donor split")
    return rows, roles


def _decode(values: np.ndarray) -> list[str]:
    return [
        value.decode() if isinstance(value, bytes) else str(value) for value in values
    ]


def _axis(path: Path, assay: dict[str, object]) -> dict[str, object]:
    obs_path = str(assay["obs_index_hdf5_path"])
    feature_path = str(assay["feature_index_hdf5_path"])
    matrix_path = str(assay["matrix_hdf5_path"])
    with h5py.File(path, "r") as handle:
        barcodes = _decode(handle[obs_path][:])
        features = _decode(handle[feature_path][:])
        matrix = handle[matrix_path]
        if matrix.attrs.get("encoding-type") != "csr_matrix":
            raise ValueError("top-level H5AD X must be CSR")
        shape = tuple(int(value) for value in matrix.attrs["shape"])
    if shape != (len(barcodes), len(features)) or len(set(barcodes)) != len(barcodes):
        raise ValueError("H5AD axes and X shape are inconsistent")
    missing = sorted(set(MARKERS) - set(features))
    if missing:
        raise ValueError(f"locked markers are absent: {missing}")
    return {"barcodes": barcodes, "features": features, "shape": shape}


def _row_vectors(
    axis: dict[str, object], roles: dict[str, list[str]]
) -> dict[str, np.ndarray]:
    lookup = {barcode: index for index, barcode in enumerate(axis["barcodes"])}
    missing = sorted(
        set().union(*(set(value) for value in roles.values())) - set(lookup)
    )
    if missing:
        raise ValueError("metadata barcodes are absent from an assay")
    return {
        role: np.sort(np.asarray([lookup[value] for value in barcodes], dtype=int))
        for role, barcodes in roles.items()
    }


def _validate_numerical_certificate(value: object, name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} lacks a numerical certificate")
    required = (
        value.get("converged") is True,
        isinstance(value.get("scaled_gradient_norm"), (int, float)),
        isinstance(value.get("gradient_tolerance"), (int, float)),
        isinstance(value.get("schur_condition_number"), (int, float)),
        isinstance(value.get("theta_curvature_condition_number"), (int, float)),
    )
    if not all(required):
        raise ValueError(f"{name} numerical certificate is incomplete")
    if float(value["scaled_gradient_norm"]) > float(value["gradient_tolerance"]):
        raise ValueError(f"{name} misses its gradient certificate")
    if (
        max(
            float(value["schur_condition_number"]),
            float(value["theta_curvature_condition_number"]),
        )
        > 1e12
    ):
        raise ValueError(f"{name} exceeds the condition-number limit")


def _assert_confirmation_family_available() -> None:
    if any(path.exists() for path in SANGER_TERMINAL_ARTIFACTS):
        raise PermissionError(
            "BMMC backup is disabled because a Sanger terminal artifact exists"
        )


def _validated_development(source_hash: str) -> dict[str, object]:
    if not DEVELOPMENT_RESULT.is_file():
        raise PermissionError("audited non-held development result is absent")
    result = _read_json(DEVELOPMENT_RESULT)
    if (
        result.get("status") != "DEVELOPMENT_PASS"
        or result.get("gate", {}).get("passes_all") is not True
    ):
        raise PermissionError("non-held development has not passed")
    if result.get("source_manifest_sha256") != source_hash:
        raise PermissionError(
            "development result is bound to a different source manifest"
        )
    split = result.get("split")
    expected = {
        "fit_donors": list(FIT_DONORS),
        "development_donors": list(DEVELOPMENT_DONORS),
        "held_donors": list(HELD_DONORS),
        "original_is_train_used": False,
        "physical_donor_disjoint": True,
        "site_disjoint": False,
    }
    if (
        split != expected
        or result.get("markers") != list(MARKERS)
        or result.get("entity_count") != 100
    ):
        raise ValueError("development result differs from the locked design")
    audit = result.get("access_audit", {})
    if (
        audit.get("held_feature_rows_read") != 0
        or audit.get("held_tables_formed") != 0
        or audit.get("raw_x_opened") is not False
    ):
        raise PermissionError("development records held feature access")
    if result.get("software_audit", {}).get("hierarchical_tests_passed") is not True:
        raise PermissionError("hierarchical estimator software audit is absent")
    methods = result.get("frozen_source_model", {}).get("methods", {})
    if set(methods) != set(REQUIRED_METHODS):
        raise ValueError("frozen method set differs")
    if (
        methods["primary"].get("kind") != "conditional_log_odds"
        or methods["best_residual"].get("kind") != "classical_residual"
    ):
        raise ValueError("primary or classical comparator kind differs")
    for name in REQUIRED_METHODS:
        if name == "independence":
            continue
        coordinate = np.asarray(methods[name].get("source_coordinate"), dtype=float)
        if coordinate.shape != (100,) or not np.isfinite(coordinate).all():
            raise ValueError(f"{name} source coordinate is invalid")
        if methods[name].get("kind") == "conditional_log_odds":
            _validate_numerical_certificate(
                methods[name].get("numerical_certificate"), name
            )
    residual = methods["best_residual"]
    if (
        residual.get("family") not in {"pearson", "deviance"}
        or residual.get("sample_size_normalized") is not True
    ):
        raise ValueError(
            "classical comparator is not matched sqrt(n)-normalized transfer"
        )
    graph = result.get("frozen_source_model", {}).get("graph", {})
    if (
        graph.get("built_from_fit_donors_only") is not True
        or graph.get("development_or_held_outcomes_used") is not False
    ):
        raise PermissionError("hypergraph provenance differs from the protocol")
    return result


def predict() -> dict[str, object]:
    """Package a frozen non-held model without decoding held assay rows."""

    _assert_confirmation_family_available()
    if any(
        path.exists()
        for path in (PREDICTION, AUTHORIZATION, SCORE_ATTEMPT, OUTPUT, REFUSAL)
    ):
        raise FileExistsError("a BMMC confirmation artifact already exists")
    source, paths = _validated_source()
    rows, roles = _metadata_roles(paths["metadata"])
    complete = paths["complete_cite_h5ad"]
    axes = {
        name: _axis(complete, source["internal_assays"][name])
        for name in ("rna", "adt")
    }
    role_rows = {name: _row_vectors(axis, roles) for name, axis in axes.items()}
    if any(
        np.intersect1d(value["held"], np.r_[value["fit"], value["development"]]).size
        for value in role_rows.values()
    ):
        raise PermissionError("held and non-held H5AD rows overlap")
    source_hash = _sha256(SOURCE_MANIFEST)
    development = _validated_development(source_hash)
    payload = {
        "schema": "scmmib-bmmc-exact-prediction/1.0",
        "status": "FROZEN_OUTCOME_ACCESS_DISABLED",
        "created_at_utc": _timestamp(),
        "design": {
            "fit_donors": list(FIT_DONORS),
            "development_donors": list(DEVELOPMENT_DONORS),
            "held_donors": list(HELD_DONORS),
            "held_unit": "physical_donor",
            "site_disjoint": False,
            "original_is_train_used": False,
            "markers": list(MARKERS),
            "ordered_entities": 100,
        },
        "bindings": {
            "runner_sha256": _sha256(Path(__file__)),
            "protocol_sha256": _sha256(PROTOCOL),
            "preflight_sha256": _sha256(PREFLIGHT),
            "source_manifest_sha256": source_hash,
            "development_result_sha256": _sha256(DEVELOPMENT_RESULT),
        },
        "source_summary": {
            "complete_cite_h5ad_sha256": source["complete_cite_h5ad"]["sha256"],
            "metadata_rows": len(rows),
            "rna_shape": axes["rna"]["shape"],
            "adt_shape": axes["adt"]["shape"],
        },
        "frozen_source_model": development["frozen_source_model"],
        "development_gate": development["gate"],
        "held_access_audit": {
            "held_feature_rows_decoded": 0,
            "held_modality_margins_computed": 0,
            "held_tables_formed": 0,
            "raw_x_opened": False,
            "held_row_counts": {
                name: len(value["held"]) for name, value in role_rows.items()
            },
        },
    }
    _write_json_exclusive(PREDICTION, payload)
    if (
        not AUTH_TEMPLATE.is_file()
        or _read_json(AUTH_TEMPLATE).get("status") != "OUTCOME_ACCESS_DISABLED"
    ):
        raise PermissionError("disabled authorization template is absent")
    return payload


def _validated_authorization() -> tuple[dict[str, object], _ScorePermit]:
    if not PREDICTION.is_file() or not AUTHORIZATION.is_file():
        raise PermissionError("prediction or active authorization is absent")
    prediction = _read_json(PREDICTION)
    authorization = _read_json(AUTHORIZATION)
    if (
        prediction.get("status") != "FROZEN_OUTCOME_ACCESS_DISABLED"
        or authorization.get("status") != "OUTCOME_ACCESS_AUTHORIZED"
    ):
        raise PermissionError("held outcome access is disabled")
    expected = {
        "prediction_path": _relative(PREDICTION),
        "prediction_sha256": _sha256(PREDICTION),
        "runner_sha256": _sha256(Path(__file__)),
        "protocol_sha256": _sha256(PROTOCOL),
        "source_manifest_sha256": _sha256(SOURCE_MANIFEST),
        "development_result_sha256": _sha256(DEVELOPMENT_RESULT),
    }
    for key, value in expected.items():
        if authorization.get(key) != value:
            raise PermissionError(f"authorization {key} differs")
    commit = authorization.get("public_prediction_commit")
    url = authorization.get("public_prediction_url")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise PermissionError("public prediction commit is not immutable")
    if not isinstance(url, str):
        raise PermissionError("public prediction URL is absent")
    parsed = urlparse(url)
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    expected_tail = ["blob", commit, *_relative(PREDICTION).split("/")]
    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or len(parts) < len(expected_tail) + 2
        or parts[-len(expected_tail) :] != expected_tail
    ):
        raise PermissionError("public prediction URL is not the bound GitHub blob")
    return prediction, _ScorePermit(expected["prediction_sha256"], commit)


def _score_held_once(
    prediction: dict[str, object], permit: _ScorePermit
) -> dict[str, object]:
    """Held scoring engine; deliberately unreachable in the metadata-only package."""

    del prediction, permit
    raise RuntimeError(
        "held scoring is disabled until the complete-H5AD reader is audited"
    )


def score() -> dict[str, object]:
    """Run the terminal one-shot score after an immutable public authorization."""

    _assert_confirmation_family_available()
    if any(path.exists() for path in (SCORE_ATTEMPT, OUTPUT, REFUSAL)):
        raise FileExistsError("a terminal BMMC scoring artifact already exists")
    prediction, permit = _validated_authorization()
    _write_json_exclusive(
        SCORE_ATTEMPT,
        {
            "schema": "scmmib-bmmc-score-attempt/1.0",
            "status": "TERMINAL_ATTEMPT_STARTED",
            "started_at_utc": _timestamp(),
            "prediction_sha256": permit.prediction_sha256,
            "public_prediction_commit": permit.public_commit,
            "held_feature_access_started_after_this_write": True,
        },
    )
    try:
        result = _score_held_once(prediction, permit)
        if result.get("status") not in {"CONFIRMATION_PASS", "CONFIRMATION_FAIL"}:
            raise ValueError("held scorer did not return a terminal decision")
        _write_json_exclusive(OUTPUT, result)
        return result
    except Exception as error:
        _write_json_exclusive(
            REFUSAL,
            {
                "schema": "scmmib-bmmc-score-refusal/1.0",
                "status": "TERMINAL_SCORE_REFUSAL",
                "created_at_utc": _timestamp(),
                "error_type": type(error).__name__,
                "reason": str(error),
                "prediction_sha256": permit.prediction_sha256,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("predict", "score"))
    args = parser.parse_args()
    payload = predict() if args.phase == "predict" else score()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
