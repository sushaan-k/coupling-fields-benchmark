"""Selective one-donor-at-a-time reducer for GSE279451 RNA--ADT counts.

The shipped source template is disabled, so no GEO matrix is reachable. The
reader is deliberately small: it validates a bound member before opening it,
streams only the 18 locked feature rows, emits 81 2x2 tables and unpaired
marker-level marginal summaries, and discards the cell-level vectors before
advancing to another donor.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GSE279451_SEPSIS_HELD_DONOR_CONFIRMATION_PROTOCOL_2026-08-28.md"
PREFLIGHT = ROOT / "data/development/gse279451_sepsis/metadata_preflight_v1.json"
DESIGNATION = ROOT / "data/confirmation/gse279451_sepsis/candidate_designation_v1.json"
FAMILY_POLICY = ROOT / "data/confirmation/gse279451_sepsis/family_policy_v1.json"
SOURCE_TEMPLATE = (
    ROOT / "data/confirmation/gse279451_sepsis/source_manifest_template_v1.json"
)
SOURCE_MANIFEST = ROOT / "data/confirmation/gse279451_sepsis/source_manifest_v1.json"
OUTPUT = ROOT / "data/development/gse279451_sepsis/reduced_development_v1.json"
DEVELOPMENT_ATTEMPT = (
    ROOT / "data/development/gse279451_sepsis/development_attempt_v1.json"
)
SCORE_ATTEMPT = ROOT / "data/confirmation/gse279451_sepsis/score_attempt_v1.json"
ACQUISITION = ROOT / "experiments/acquire_gse279451_nonheld.py"
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

MARKERS = ("CD4", "CD7", "CD14", "CD19", "CD33", "CD38", "CD44", "CD47", "CD52")
DEVELOPMENT_DONORS = (
    "GSM8571043",
    "GSM8571044",
    "GSM8571047",
    "GSM8571048",
    "GSM8571049",
    "GSM8571052",
    "GSM8571055",
    "GSM8571056",
    "GSM8571060",
    "GSM8571061",
    "GSM8571065",
    "GSM8571068",
    "GSM8571072",
    "GSM8571073",
    "GSM8571074",
    "GSM8571075",
    "GSM8571077",
    "GSM8571079",
    "GSM8571081",
)
HELD_DONORS = (
    "GSM8571042",
    "GSM8571045",
    "GSM8571046",
    "GSM8571050",
    "GSM8571051",
    "GSM8571053",
    "GSM8571054",
    "GSM8571057",
    "GSM8571058",
    "GSM8571059",
    "GSM8571062",
    "GSM8571063",
    "GSM8571064",
    "GSM8571066",
    "GSM8571067",
    "GSM8571069",
    "GSM8571070",
    "GSM8571071",
    "GSM8571076",
    "GSM8571078",
    "GSM8571080",
)
FEATURE_AXIS_SHA256 = "ff6a914dd33b3a3c2dd913ed439ed4b150fd8ab210595dec2447a283eb9b417b"
ADT_TIE_SALT = "GSE279451-ADT-v1"
CELL_SELECTION_SALT = "GSE279451-CELL-BUDGET-v1"
CELL_BUDGET = 1024


@dataclass(frozen=True)
class HeldAccessPermit:
    prediction_sha256: str
    public_commit: str
    attempt_path: Path


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


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(serialized)


def _bound_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PermissionError(f"{label} path is not bound")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PermissionError(f"{label} path must be repository-relative")
    resolved = (ROOT / candidate).resolve(strict=True)
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise PermissionError(f"{label} path escapes the repository") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is absent")
    return resolved


def _expected_role(accession: str) -> str:
    if accession in DEVELOPMENT_DONORS:
        return "development"
    if accession in HELD_DONORS:
        return "held"
    raise ValueError(f"unrecognized donor accession {accession}")


def _sample_name(source: dict[str, Any], accession: str) -> str:
    matches = [
        record.get("sample")
        for record in source.get("donors", [])
        if isinstance(record, dict) and record.get("accession") == accession
    ]
    if len(matches) != 1 or not isinstance(matches[0], str) or not matches[0]:
        raise ValueError(f"{accession} sample name is not uniquely bound")
    return matches[0]


def _validate_manifest_shape(source: dict[str, Any]) -> None:
    if source.get("schema") != "gse279451-sepsis-source/1.0":
        raise PermissionError("source manifest schema differs")
    if source.get("status") != "NONHELD_SOURCE_ACCESS_AUTHORIZED":
        raise PermissionError("non-held source access is disabled")
    template = _read_json(SOURCE_TEMPLATE)
    if template.get("status") != "SOURCE_UNAVAILABLE" or template.get("members") != []:
        raise PermissionError("source template is not the disabled public freeze")
    frozen_keys = (
        "accession",
        "raw_tar",
        "member_rule",
        "axis_contract",
        "primary_cell_budget",
        "matrix_contract",
        "donor_identity_contract",
        "donors",
    )
    if any(source.get(key) != template.get(key) for key in frozen_keys):
        raise PermissionError("source manifest differs from the frozen source template")
    donors = source.get("donors")
    if not isinstance(donors, list) or len(donors) != 40:
        raise ValueError("source manifest must bind 40 donors")
    observed = {
        str(record.get("accession")): str(record.get("role"))
        for record in donors
        if isinstance(record, dict)
    }
    expected = {
        accession: _expected_role(accession)
        for accession in (*DEVELOPMENT_DONORS, *HELD_DONORS)
    }
    if observed != expected:
        raise ValueError("source donor roles differ from the designation")
    donor_records = {
        str(record["accession"]): record
        for record in donors
        if isinstance(record, dict)
    }
    members = source.get("members")
    if not isinstance(members, list):
        raise ValueError("source members are absent")
    member_keys = [
        (record.get("accession"), record.get("kind"))
        for record in members
        if isinstance(record, dict)
    ]
    expected_keys = {
        *((accession, "barcodes") for accession in expected),
        *((accession, "features") for accession in expected),
        *((accession, "matrix") for accession in DEVELOPMENT_DONORS),
    }
    if len(member_keys) != len(expected_keys) or set(member_keys) != expected_keys:
        raise PermissionError(
            "non-held manifest must bind all axes and only development matrices"
        )
    for record in members:
        if not isinstance(record, dict) or set(record) != {
            "accession",
            "kind",
            "url",
            "bytes",
            "local_path",
            "sha256",
            "retained",
        }:
            raise PermissionError(
                "source member fields differ from the frozen contract"
            )
        accession = record["accession"]
        kind = record["kind"]
        donor = donor_records[accession]
        suffix = {
            "barcodes": "barcodes.tsv.gz",
            "features": "features.tsv.gz",
            "matrix": "matrix.mtx.gz",
        }[kind]
        filename = f"{accession}_{donor['sample']}.{suffix}"
        expected_url = (
            "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8571nnn/"
            f"{accession}/suppl/{filename}"
        )
        if record.get("url") != expected_url:
            raise PermissionError(f"{accession} {kind} URL differs")
        expected_bytes = (
            donor.get(f"{kind[:-1]}_bytes")
            if kind != "matrix"
            else donor.get("matrix_bytes")
        )
        if record.get("bytes") != expected_bytes:
            raise PermissionError(f"{accession} {kind} byte count differs")
        if not isinstance(record.get("sha256"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", record["sha256"]
        ):
            raise PermissionError(f"{accession} {kind} SHA-256 is not bound")
        if kind != "matrix" and record.get("sha256") != donor.get(
            f"{kind[:-1]}_sha256"
        ):
            raise PermissionError(f"{accession} {kind} SHA-256 differs from preflight")
        retained = record.get("retained")
        if kind == "matrix":
            if retained is not False or record.get("local_path") is not None:
                raise PermissionError(
                    f"{accession} development matrix was not discarded after reduction"
                )
        elif retained is not True:
            raise PermissionError(f"{accession} {kind} axis is not retained")
    bindings = source.get("bindings", {})
    expected_bindings = {
        "source_template_sha256": _sha256(SOURCE_TEMPLATE),
        "candidate_designation_sha256": _sha256(DESIGNATION),
        "family_policy_sha256": _sha256(FAMILY_POLICY),
        "protocol_sha256": _sha256(PROTOCOL),
        "metadata_preflight_sha256": _sha256(PREFLIGHT),
        "acquisition_sha256": _sha256(ACQUISITION),
        "reducer_sha256": _sha256(Path(__file__)),
        "development_evaluator_sha256": _sha256(EVALUATOR),
        "runner_sha256": _sha256(RUNNER),
        "tests_sha256": _sha256(TESTS),
        "development_evaluator_tests_sha256": _sha256(EVALUATOR_TESTS),
        **{name: _sha256(path) for name, path in TRANSITIVE_ARTIFACTS.items()},
    }
    if bindings != expected_bindings:
        raise PermissionError("source manifest artifact bindings differ")
    audit = source.get("access_audit", {})
    if (
        not isinstance(audit, dict)
        or audit.get("held_matrix_bytes_read_before_public_prediction_authorization")
        != 0
        or not DEVELOPMENT_ATTEMPT.is_file()
        or audit.get("development_attempt_sha256") != _sha256(DEVELOPMENT_ATTEMPT)
        or audit.get("sanger_path_or_content_accessed") is not False
    ):
        raise PermissionError("source access audit is not clean")


def _validated_member(
    source: dict[str, Any],
    accession: str,
    kind: str,
    *,
    phase: str,
    permit: HeldAccessPermit | None = None,
) -> Path:
    role = _expected_role(accession)
    if phase == "development" and role != "development":
        raise PermissionError("development reducer forbids every held accession")
    if phase not in {"development", "held_score_authorized"}:
        raise PermissionError("unrecognized source-access phase")
    if role == "held" and kind == "matrix":
        if phase != "held_score_authorized" or permit is None:
            raise PermissionError(
                "held MTX access requires public prediction authorization"
            )
        _validate_held_permit(permit)
    members = source.get("members")
    if not isinstance(members, list):
        raise ValueError("source members are absent")
    matches = [
        record
        for record in members
        if isinstance(record, dict)
        and record.get("accession") == accession
        and record.get("kind") == kind
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {accession} {kind} member")
    record = matches[0]
    if record.get("retained") is not True:
        raise PermissionError(f"{accession} {kind} member is not retained")
    path = _bound_path(record.get("local_path"), f"{accession} {kind}")
    expected_bytes = record.get("bytes")
    expected_hash = record.get("sha256")
    if not isinstance(expected_bytes, int) or path.stat().st_size != expected_bytes:
        raise PermissionError(f"{accession} {kind} byte count differs")
    if not isinstance(expected_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_hash
    ):
        raise PermissionError(f"{accession} {kind} SHA-256 is not bound")
    if _sha256(path) != expected_hash:
        raise PermissionError(f"{accession} {kind} SHA-256 differs")
    return path


def _validate_held_permit(permit: HeldAccessPermit) -> None:
    if permit.attempt_path.resolve() != SCORE_ATTEMPT.resolve():
        raise PermissionError("held permit is bound to a different attempt marker")
    if not SCORE_ATTEMPT.is_file():
        raise PermissionError("terminal score-attempt marker is absent")
    attempt = _read_json(SCORE_ATTEMPT)
    if (
        attempt.get("status") != "TERMINAL_ATTEMPT_STARTED"
        or attempt.get("prediction_sha256") != permit.prediction_sha256
        or attempt.get("public_prediction_commit") != permit.public_commit
        or not re.fullmatch(r"[0-9a-f]{64}", permit.prediction_sha256)
        or not re.fullmatch(r"[0-9a-f]{40}", permit.public_commit)
    ):
        raise PermissionError("held permit differs from the terminal attempt")


def _axes(
    source: dict[str, Any],
    accession: str,
    *,
    phase: str,
    permit: HeldAccessPermit | None = None,
) -> tuple[list[str], list[tuple[str, str, str]]]:
    barcode_path = _validated_member(
        source, accession, "barcodes", phase=phase, permit=permit
    )
    feature_path = _validated_member(
        source, accession, "features", phase=phase, permit=permit
    )
    with gzip.open(barcode_path, "rt") as handle:
        barcodes = [line.rstrip("\n") for line in handle]
    if not barcodes or len(set(barcodes)) != len(barcodes):
        raise ValueError(f"{accession} barcode axis is empty or nonunique")
    donor = next(
        record
        for record in source["donors"]
        if isinstance(record, dict) and record.get("accession") == accession
    )
    if len(barcodes) != donor.get("barcode_cells"):
        raise ValueError(f"{accession} barcode count differs from preflight")
    with gzip.open(feature_path, "rb") as handle:
        raw_features = handle.read()
    if hashlib.sha256(raw_features).hexdigest() != FEATURE_AXIS_SHA256:
        raise ValueError(f"{accession} decompressed feature axis differs")
    features = []
    for line in raw_features.decode().splitlines():
        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError(f"{accession} feature row is malformed")
        features.append((fields[0], fields[1], fields[2]))
    return barcodes, features


def _marker_rows(features: list[tuple[str, str, str]]) -> dict[str, list[int]]:
    rows: dict[str, list[int]] = {"rna": [], "adt": []}
    for marker in MARKERS:
        rna = [
            index
            for index, (_, name, kind) in enumerate(features)
            if kind == "Gene Expression" and name == marker
        ]
        adt = [
            index
            for index, (_, name, kind) in enumerate(features)
            if kind == "Antibody Capture"
            and re.fullmatch(r"C[0-9]+-(.+)", name)
            and name.split("-", 1)[1] == marker
        ]
        if len(rna) != 1 or len(adt) != 1:
            raise ValueError(f"locked marker {marker} lacks a unique RNA/ADT pair")
        rows["rna"].append(rna[0])
        rows["adt"].append(adt[0])
    return rows


def _budgeted_cells(
    barcodes: list[str], accession: str, sample: str
) -> tuple[list[int], list[str]]:
    if len(barcodes) < CELL_BUDGET:
        raise ValueError(f"{accession} has fewer than {CELL_BUDGET} barcodes")
    order = sorted(
        range(len(barcodes)),
        key=lambda index: hashlib.sha256(
            f"{CELL_SELECTION_SALT}{accession}{sample}{barcodes[index]}".encode()
        ).hexdigest(),
    )[:CELL_BUDGET]
    return order, [barcodes[index] for index in order]


def _stream_selected_rows(
    matrix_path: Path,
    selected_rows: list[int],
    selected_cells: list[int],
    *,
    expected_features: int,
    expected_cells: int,
) -> np.ndarray:
    """Decode selected zero-based feature rows from a 10x Matrix Market gzip."""

    lookup = {row + 1: index for index, row in enumerate(selected_rows)}
    if len(lookup) != len(selected_rows):
        raise ValueError("selected Matrix Market rows are not unique")
    cell_lookup = {cell + 1: index for index, cell in enumerate(selected_cells)}
    if len(cell_lookup) != len(selected_cells):
        raise ValueError("selected Matrix Market cells are not unique")
    values = np.zeros((len(selected_rows), len(selected_cells)), dtype=np.int64)
    with gzip.open(matrix_path, "rt") as handle:
        banner = handle.readline().strip()
        if banner != "%%MatrixMarket matrix coordinate integer general":
            raise ValueError("matrix is not a general integer coordinate matrix")
        dimensions = ""
        for line in handle:
            if not line.startswith("%"):
                dimensions = line
                break
        fields = dimensions.split()
        if len(fields) != 3:
            raise ValueError("Matrix Market dimensions are absent")
        n_features, n_cells, declared_entries = map(int, fields)
        if (n_features, n_cells) != (expected_features, expected_cells):
            raise ValueError("Matrix Market shape differs from its axes")
        observed_entries = 0
        for line in handle:
            fields = line.split()
            if len(fields) != 3:
                raise ValueError("Matrix Market entry is malformed")
            feature, cell, count = map(int, fields)
            observed_entries += 1
            if not 1 <= feature <= n_features or not 1 <= cell <= n_cells:
                raise ValueError("Matrix Market coordinate is out of bounds")
            if count < 0:
                raise ValueError("raw count is negative")
            output_row = lookup.get(feature)
            output_cell = cell_lookup.get(cell)
            if output_row is not None and output_cell is not None:
                values[output_row, output_cell] += count
        if observed_entries != declared_entries:
            raise ValueError("Matrix Market entry count differs")
    return values


def _adt_states(counts: np.ndarray, barcodes: list[str], accession: str) -> np.ndarray:
    states = np.empty_like(counts, dtype=np.uint8)
    lower = len(barcodes) // 2
    for marker_index, marker in enumerate(MARKERS):
        tie_hashes = [
            hashlib.sha256(
                f"{ADT_TIE_SALT}{accession}{barcode}{marker}".encode()
            ).hexdigest()
            for barcode in barcodes
        ]
        order = sorted(
            range(len(barcodes)),
            key=lambda cell: (int(counts[marker_index, cell]), tie_hashes[cell]),
        )
        states[marker_index] = 1
        states[marker_index, order[:lower]] = 0
    return states


def _ordered_tables(rna: np.ndarray, adt: np.ndarray) -> np.ndarray:
    tables = np.empty((81, 2, 2), dtype=np.int64)
    entity = 0
    for rna_index in range(len(MARKERS)):
        for adt_index in range(len(MARKERS)):
            code = 2 * rna[rna_index].astype(int) + adt[adt_index].astype(int)
            tables[entity] = np.bincount(code, minlength=4).reshape(2, 2)
            entity += 1
    return tables


def _destroyed_adt(adt: np.ndarray, accession: str) -> np.ndarray:
    seed = int.from_bytes(
        hashlib.sha256(f"destroyed-v1{accession}".encode()).digest()[:8], "big"
    )
    permutation = np.random.default_rng(seed).permutation(adt.shape[1])
    return adt[:, permutation]


def reduce_donor(
    source: dict[str, Any],
    accession: str,
    *,
    phase: str = "development",
    permit: HeldAccessPermit | None = None,
    predictions_materialized_sha256: str | None = None,
) -> dict[str, Any]:
    """Reduce one authorized donor; callers must discard the returned arrays."""

    if _expected_role(accession) == "held" and (
        not isinstance(predictions_materialized_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", predictions_materialized_sha256)
    ):
        raise PermissionError("held truth requires materialized prediction binding")
    barcodes, features = _axes(source, accession, phase=phase, permit=permit)
    sample = _sample_name(source, accession)
    selected_cells, selected_barcodes = _budgeted_cells(barcodes, accession, sample)
    marker_rows = _marker_rows(features)
    matrix_path = _validated_member(
        source, accession, "matrix", phase=phase, permit=permit
    )
    selected = marker_rows["rna"] + marker_rows["adt"]
    values = _stream_selected_rows(
        matrix_path,
        selected,
        selected_cells,
        expected_features=len(features),
        expected_cells=len(barcodes),
    )
    rna = (values[: len(MARKERS)] > 0).astype(np.uint8)
    adt_counts = values[len(MARKERS) :]
    adt = _adt_states(adt_counts, selected_barcodes, accession)
    tables = _ordered_tables(rna, adt)
    destroyed_tables = _ordered_tables(rna, _destroyed_adt(adt, accession))
    if not np.array_equal(tables.sum(axis=-1), destroyed_tables.sum(axis=-1)):
        raise AssertionError("destroyed-link control changed an RNA margin")
    if not np.array_equal(tables.sum(axis=-2), destroyed_tables.sum(axis=-2)):
        raise AssertionError("destroyed-link control changed an ADT margin")
    panel_total = adt_counts.sum(axis=0, keepdims=True)
    adt_composition = np.divide(
        100.0 * adt_counts,
        panel_total,
        out=np.zeros_like(adt_counts, dtype=float),
        where=panel_total > 0,
    )
    informative = [
        bool(
            table[0].sum()
            and table[1].sum()
            and table[:, 0].sum()
            and table[:, 1].sum()
        )
        for table in tables
    ]
    if sum(informative) < 64:
        raise ValueError(f"{accession} has fewer than 64 margin-informative entities")
    if not np.isfinite(adt_composition).all() or not math.isfinite(
        float(np.log1p(adt_composition).mean())
    ):
        raise FloatingPointError(f"{accession} graph profiles are nonfinite")
    return {
        "accession": accession,
        "sample": sample,
        "role": _expected_role(accession),
        "deposited_cells": len(barcodes),
        "cells": CELL_BUDGET,
        "cell_selection_salt": CELL_SELECTION_SALT,
        "selected_barcode_axis_sha256": hashlib.sha256(
            ("\n".join(selected_barcodes) + "\n").encode()
        ).hexdigest(),
        "markers": list(MARKERS),
        "entity_count": 81,
        "rna_detection_prevalence": rna.mean(axis=1).tolist(),
        "adt_log_panel_fraction_mean": np.log1p(adt_composition).mean(axis=1).tolist(),
        "tables": tables.reshape(81, 4).tolist(),
        "destroyed_tables": destroyed_tables.reshape(81, 4).tolist(),
        "informative": informative,
        "matrix_sha256": _sha256(matrix_path),
        "predictions_materialized_sha256": predictions_materialized_sha256,
    }


def reduce_development(
    source_path: Path = SOURCE_MANIFEST, output_path: Path = OUTPUT
) -> dict[str, Any]:
    """Reduce exactly the 19 development donors and no held matrix."""

    if not source_path.is_file():
        raise PermissionError("active source manifest is absent")
    source = _read_json(source_path)
    _validate_manifest_shape(source)
    reduced = []
    for accession in DEVELOPMENT_DONORS:
        reduced.append(reduce_donor(source, accession, phase="development"))
    payload = {
        "schema": "gse279451-sepsis-reduced-development/1.0",
        "status": "NONHELD_REDUCTION_COMPLETE",
        "source_manifest_sha256": _sha256(source_path),
        "development_donors": list(DEVELOPMENT_DONORS),
        "held_donors": list(HELD_DONORS),
        "markers": list(MARKERS),
        "entity_count": 81,
        "primary_cells_per_donor": CELL_BUDGET,
        "cell_selection_salt": CELL_SELECTION_SALT,
        "all_cells_sensitivity_included": False,
        "donors": reduced,
        "access_audit": {
            "development_matrix_members_decoded": len(reduced),
            "held_matrix_members_opened": 0,
            "held_matrix_entries_decoded": 0,
            "maximum_concurrent_donor_matrices": 1,
        },
    }
    _write_json_exclusive(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(
        json.dumps(
            reduce_development(args.source, args.output), indent=2, allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
