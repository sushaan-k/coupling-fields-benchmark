"""Fail-closed prospective runner for the GSE279451 held-donor confirmation.

``predict`` packages a passing 19-donor development model without opening a
held MTX. ``score`` requires an immutable public prediction authorization and
writes a terminal attempt marker before the one-shot held engine can acquire a
held member.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np

from experiments import reduce_gse279451_sepsis as reducer
from experiments.reduce_gse279451_sepsis import _validate_manifest_shape


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GSE279451_SEPSIS_HELD_DONOR_CONFIRMATION_PROTOCOL_2026-08-28.md"
PREFLIGHT = ROOT / "data/development/gse279451_sepsis/metadata_preflight_v1.json"
DESIGNATION = ROOT / "data/confirmation/gse279451_sepsis/candidate_designation_v1.json"
FAMILY_POLICY = ROOT / "data/confirmation/gse279451_sepsis/family_policy_v1.json"
SOURCE_TEMPLATE = (
    ROOT / "data/confirmation/gse279451_sepsis/source_manifest_template_v1.json"
)
SOURCE_MANIFEST = ROOT / "data/confirmation/gse279451_sepsis/source_manifest_v1.json"
REDUCER = ROOT / "experiments/reduce_gse279451_sepsis.py"
EVALUATOR = ROOT / "experiments/evaluate_gse279451_sepsis_development.py"
REDUCED_DEVELOPMENT = (
    ROOT / "data/development/gse279451_sepsis/reduced_development_v1.json"
)
DEVELOPMENT_ATTEMPT = (
    ROOT / "data/development/gse279451_sepsis/development_attempt_v1.json"
)
DEVELOPMENT_RESULT = (
    ROOT / "results/development/gse279451_sepsis_exact_development.json"
)
EVALUATION_ATTEMPT = (
    ROOT / "data/development/gse279451_sepsis/evaluation_attempt_v1.json"
)
PREDICTION = ROOT / "results/gse279451_sepsis_exact_predictions.json"
AUTH_TEMPLATE = (
    ROOT / "data/confirmation/gse279451_sepsis/score_authorization_template_v1.json"
)
AUTHORIZATION = ROOT / "data/confirmation/gse279451_sepsis/score_authorization_v1.json"
SCORE_ATTEMPT = ROOT / "data/confirmation/gse279451_sepsis/score_attempt_v1.json"
OUTPUT = ROOT / "results/gse279451_sepsis_exact_confirmation.json"
REFUSAL = ROOT / "results/gse279451_sepsis_exact_score_refusal.json"
HELD_MEMBER_DIR = ROOT / "data/confirmation/gse279451_sepsis/held_source_work"
HELD_PREDICTION_DIR = (
    ROOT / "data/confirmation/gse279451_sepsis/held_prediction_materializations"
)
BMMC_TERMINAL = (
    ROOT
    / "results/development/scmmib_bmmc_exact_development_attempt_3_terminal_refusal.json"
)

SANGER_TERMINAL_ARTIFACTS = (
    ROOT / "data/confirmation/scmmib_sanger/score_attempt_v1.json",
    ROOT / "results/scmmib_sanger_exact_confirmation.json",
    ROOT / "results/scmmib_sanger_exact_score_refusal.json",
)
BMMC_REVIVAL_ARTIFACTS = (
    ROOT / "results/development/scmmib_bmmc_exact_development.json",
    ROOT / "results/scmmib_bmmc_exact_predictions.json",
    ROOT / "data/confirmation/scmmib_bmmc/score_attempt_v1.json",
    ROOT / "results/scmmib_bmmc_exact_confirmation.json",
    ROOT / "results/scmmib_bmmc_exact_score_refusal.json",
)

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
DEVELOPMENT_GATE_COMPARATORS = (
    "best_residual",
    "destroyed_link",
    "hierarchical_ridge_only",
    "common_effect_graph",
)
HELD_GATE_COMPARATORS = ("best_residual", "destroyed_link")
CV_GRID = {
    "graph_neighborhood": [1, 2, 3],
    "heterogeneity_penalty": [0.1, 1.0, 10.0],
    "ridge_penalty": [0.01, 0.1],
    "graph_penalty": [0.1, 0.3, 1.0],
    "transport_multiplier": [0.75, 1.0, 1.25],
}
CELL_BUDGET = 1024
MINIMUM_INFORMATIVE_ENTITIES = 64
BMMC_TERMINAL_SHA256 = (
    "caf920719694487ba228dc64ac14ed4a6579619349f496f7154372920f3e128c"
)
BOOTSTRAPS = 20_000
SEED = 20260828
_LAST_HELD_AUDIT: dict[str, Any] | None = None


@dataclass(frozen=True)
class _ScorePermit:
    prediction_sha256: str
    public_commit: str
    authorization_sha256: str
    public_prediction_url: str
    remote_prediction_sha256: str


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _transitive_bindings() -> dict[str, str]:
    return {name: _sha256(path) for name, path in reducer.TRANSITIVE_ARTIFACTS.items()}


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


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _assert_family_available() -> None:
    if any(path.exists() for path in SANGER_TERMINAL_ARTIFACTS):
        raise PermissionError("GSE279451 is disabled by a Sanger terminal artifact")
    if any(path.exists() for path in BMMC_REVIVAL_ARTIFACTS):
        raise PermissionError("BMMC revival artifact violates the closed family")
    if not FAMILY_POLICY.is_file() or not BMMC_TERMINAL.is_file():
        raise PermissionError("family policy or terminal BMMC closure is absent")
    policy = _read_json(FAMILY_POLICY)
    terminal = _read_json(BMMC_TERMINAL)
    bmmc = policy.get("bmmc", {})
    if (
        policy.get("status") != "OUTCOME_ACCESS_DISABLED"
        or not isinstance(bmmc, dict)
        or bmmc.get("status") != "TERMINAL_CLOSED_CANNOT_BE_REVIVED"
        or bmmc.get("terminal_artifact_sha256") != BMMC_TERMINAL_SHA256
        or _sha256(BMMC_TERMINAL) != BMMC_TERMINAL_SHA256
        or terminal.get("status") != "TERMINAL_NUMERICAL_EQUIVALENCE_RETRY_REFUSAL"
    ):
        raise PermissionError("terminal BMMC closure differs")


def _validated_source() -> dict[str, Any]:
    if not SOURCE_MANIFEST.is_file():
        raise PermissionError("active source manifest is absent")
    source = _read_json(SOURCE_MANIFEST)
    if (
        source.get("schema") != "gse279451-sepsis-source/1.0"
        or source.get("status") != "NONHELD_SOURCE_ACCESS_AUTHORIZED"
    ):
        raise PermissionError("non-held source access is disabled")
    _validate_manifest_shape(source)
    audit = source.get("access_audit", {})
    if (
        not isinstance(audit, dict)
        or audit.get("held_matrix_bytes_read_before_public_prediction_authorization")
        != 0
        or audit.get("development_attempt_sha256")
        != (_sha256(DEVELOPMENT_ATTEMPT) if DEVELOPMENT_ATTEMPT.is_file() else None)
        or audit.get("sanger_path_or_content_accessed") is not False
    ):
        raise PermissionError("source manifest records forbidden access")
    return source


def _validate_certificate(value: object, name: str) -> None:
    if not isinstance(value, dict) or value.get("converged") is not True:
        raise ValueError(f"{name} numerical certificate is incomplete")
    numeric = (
        "scaled_gradient_norm",
        "gradient_tolerance",
        "schur_condition_number",
        "theta_curvature_condition_number",
    )
    if any(
        isinstance(value.get(key), bool)
        or not isinstance(value.get(key), (int, float))
        or not math.isfinite(float(value[key]))
        for key in numeric
    ):
        raise ValueError(f"{name} numerical certificate is incomplete")
    if (
        float(value["scaled_gradient_norm"]) < 0.0
        or float(value["gradient_tolerance"]) <= 0.0
        or float(value["schur_condition_number"]) <= 0.0
        or float(value["theta_curvature_condition_number"]) <= 0.0
    ):
        raise ValueError(f"{name} numerical certificate has an invalid range")
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


def _finite_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _gate_comparison(
    donors: tuple[str, ...],
    primary: np.ndarray,
    comparator: np.ndarray,
    bootstrap_indices: np.ndarray,
    *,
    favorable_required: int,
) -> dict[str, Any]:
    primary_values = np.asarray(primary, dtype=float)
    comparator_values = np.asarray(comparator, dtype=float)
    if (
        primary_values.shape != (len(donors),)
        or comparator_values.shape != primary_values.shape
        or not np.isfinite(primary_values).all()
        or not np.isfinite(comparator_values).all()
        or float(comparator_values.mean()) <= 0.0
    ):
        raise ValueError("gate requires one finite loss per physical donor")
    difference = primary_values - comparator_values
    bootstrap = difference[bootstrap_indices].mean(axis=1)
    interval = np.quantile(bootstrap, [0.025, 0.975], method="linear")
    relative = 1.0 - float(primary_values.mean() / comparator_values.mean())
    favorable = int(np.count_nonzero(difference < 0.0))
    return {
        "primary_mean_loss": float(primary_values.mean()),
        "comparator_mean_loss": float(comparator_values.mean()),
        "relative_reduction": relative,
        "bootstrap_95_ci": interval.tolist(),
        "bootstrap_upper_95": float(interval[1]),
        "bootstrap_draws": BOOTSTRAPS,
        "bootstrap_seed": SEED,
        "bootstrap_quantile_method": "linear",
        "bootstrap_unit": "physical donor",
        "favorable_donors": favorable,
        "required_favorable_donors": favorable_required,
        "donor_differences_primary_minus_comparator": {
            donor: float(value) for donor, value in zip(donors, difference)
        },
        "passes_all": bool(
            relative >= 0.05 and interval[1] < 0.0 and favorable >= favorable_required
        ),
    }


def _validate_reduced_donors(reduced: dict[str, Any]) -> None:
    records = reduced.get("donors")
    if not isinstance(records, list) or [
        record.get("accession") if isinstance(record, dict) else None
        for record in records
    ] != list(DEVELOPMENT_DONORS):
        raise PermissionError("reduced development donor records differ")
    for record in records:
        if (
            record.get("role") != "development"
            or record.get("cells") != CELL_BUDGET
            or record.get("markers") != list(MARKERS)
            or record.get("entity_count") != 81
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(record.get("selected_barcode_axis_sha256"))
            )
            or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("matrix_sha256")))
        ):
            raise PermissionError("reduced development donor contract differs")
        raw_tables = np.asarray(record.get("tables"), dtype=object)
        raw_destroyed = np.asarray(record.get("destroyed_tables"), dtype=object)
        if raw_tables.shape != (81, 4) or raw_destroyed.shape != (81, 4):
            raise ValueError("reduced donor tables have the wrong shape")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (*raw_tables.flat, *raw_destroyed.flat)
        ):
            raise ValueError("reduced donor tables must be nonnegative integers")
        tables = np.asarray(raw_tables, dtype=np.int64).reshape(81, 2, 2)
        destroyed = np.asarray(raw_destroyed, dtype=np.int64).reshape(81, 2, 2)
        if not np.all(tables.sum(axis=(1, 2)) == CELL_BUDGET):
            raise ValueError("reduced donor table totals differ")
        if not np.array_equal(
            tables.sum(axis=-1), destroyed.sum(axis=-1)
        ) or not np.array_equal(tables.sum(axis=-2), destroyed.sum(axis=-2)):
            raise ValueError("destroyed-link tables changed a fixed margin")
        informative = [
            bool(
                table[0].sum()
                and table[1].sum()
                and table[:, 0].sum()
                and table[:, 1].sum()
            )
            for table in tables
        ]
        if record.get("informative") != informative or sum(informative) < 64:
            raise ValueError("reduced donor support gate differs")
        for key in ("rna_detection_prevalence", "adt_log_panel_fraction_mean"):
            profile = np.asarray(record.get(key), dtype=float)
            if profile.shape != (9,) or not np.isfinite(profile).all():
                raise ValueError(f"reduced donor {key} is invalid")


def _validate_disabled_authorization_template() -> None:
    if not AUTH_TEMPLATE.is_file():
        raise PermissionError("disabled authorization template is absent")
    template = _read_json(AUTH_TEMPLATE)
    if (
        template.get("schema") != "gse279451-sepsis-score-authorization/1.0"
        or template.get("status") != "OUTCOME_ACCESS_DISABLED"
        or template.get("prediction_path") != _relative(PREDICTION)
    ):
        raise PermissionError("disabled authorization template differs")
    bound_later = {
        "prediction_sha256",
        "runner_sha256",
        "reducer_sha256",
        "protocol_sha256",
        "candidate_designation_sha256",
        "family_policy_sha256",
        "source_manifest_sha256",
        "development_result_sha256",
        "public_prediction_commit",
        "public_prediction_url",
        *_transitive_bindings(),
    }
    if set(template) != {"schema", "status", "prediction_path", *bound_later} or any(
        template.get(key) is not None for key in bound_later
    ):
        raise PermissionError("disabled authorization template is already populated")


def _validated_development(source_hash: str, source: dict[str, Any]) -> dict[str, Any]:
    if not all(
        path.is_file()
        for path in (
            DEVELOPMENT_ATTEMPT,
            EVALUATION_ATTEMPT,
            REDUCED_DEVELOPMENT,
            DEVELOPMENT_RESULT,
        )
    ):
        raise PermissionError("reduced development input or result is absent")
    attempt = _read_json(DEVELOPMENT_ATTEMPT)
    evaluation_attempt = _read_json(EVALUATION_ATTEMPT)
    reduced = _read_json(REDUCED_DEVELOPMENT)
    result = _read_json(DEVELOPMENT_RESULT)
    axis_members = [
        member
        for member in source.get("members", [])
        if isinstance(member, dict) and member.get("kind") != "matrix"
    ]
    source_matrix_hashes = {
        member.get("accession"): member.get("sha256")
        for member in source.get("members", [])
        if isinstance(member, dict) and member.get("kind") == "matrix"
    }
    if (
        attempt.get("status") != "TERMINAL_DEVELOPMENT_ATTEMPT_STARTED"
        or attempt.get("source_template_sha256") != _sha256(SOURCE_TEMPLATE)
        or attempt.get("artifact_bindings") != source.get("bindings")
        or attempt.get("axis_members_sha256") != _json_sha256(axis_members)
        or evaluation_attempt.get("status") != "TERMINAL_DEVELOPMENT_EVALUATION_STARTED"
        or evaluation_attempt.get("reduced_development_sha256")
        != _sha256(REDUCED_DEVELOPMENT)
        or evaluation_attempt.get("development_attempt_sha256")
        != _sha256(DEVELOPMENT_ATTEMPT)
        or evaluation_attempt.get("evaluator_sha256") != _sha256(EVALUATOR)
        or evaluation_attempt.get("protocol_sha256") != _sha256(PROTOCOL)
        or evaluation_attempt.get("candidate_designation_sha256")
        != _sha256(DESIGNATION)
        or evaluation_attempt.get("family_policy_sha256") != _sha256(FAMILY_POLICY)
        or evaluation_attempt.get("transitive_bindings") != _transitive_bindings()
        or reduced.get("status") != "NONHELD_REDUCTION_COMPLETE"
        or reduced.get("development_attempt_sha256") != _sha256(DEVELOPMENT_ATTEMPT)
        or reduced.get("source_manifest_sha256") != source_hash
        or reduced.get("development_donors") != list(DEVELOPMENT_DONORS)
        or reduced.get("held_donors") != list(HELD_DONORS)
        or reduced.get("primary_cells_per_donor") != CELL_BUDGET
        or reduced.get("all_cells_sensitivity_included") is not False
        or reduced.get("access_audit", {}).get("held_matrix_members_opened") != 0
    ):
        raise PermissionError("reduced development input violates the access seal")
    _validate_reduced_donors(reduced)
    for record in reduced["donors"]:
        if record.get("matrix_sha256") != source_matrix_hashes.get(
            record.get("accession")
        ):
            raise PermissionError("reduced donor matrix hash differs from source")
    if (
        result.get("status") != "DEVELOPMENT_PASS"
        or result.get("evaluation_attempt_sha256") != _sha256(EVALUATION_ATTEMPT)
        or result.get("source_manifest_sha256") != source_hash
        or result.get("reduced_development_sha256") != _sha256(REDUCED_DEVELOPMENT)
        or result.get("markers") != list(MARKERS)
        or result.get("entity_count") != 81
        or result.get("cell_budget_per_donor") != CELL_BUDGET
        or result.get("all_cells_sensitivity_used") is not False
        or result.get("evaluator_sha256") != _sha256(EVALUATOR)
        or result.get("protocol_sha256") != _sha256(PROTOCOL)
        or result.get("candidate_designation_sha256") != _sha256(DESIGNATION)
        or result.get("family_policy_sha256") != _sha256(FAMILY_POLICY)
        or result.get("reducer_sha256") != _sha256(REDUCER)
        or result.get("transitive_bindings") != _transitive_bindings()
    ):
        raise PermissionError("development result differs from the designation")
    selection = result.get("selection", {})
    if (
        not isinstance(selection, dict)
        or selection.get("folds") != 19
        or selection.get("held_one_donor_per_fold") is not True
        or selection.get("fold_donors") != list(DEVELOPMENT_DONORS)
        or selection.get("grid") != CV_GRID
        or selection.get("final_refit_donors") != list(DEVELOPMENT_DONORS)
    ):
        raise ValueError("development selection is not locked 19-fold LOODO")
    gates = result.get("gate", {})
    comparisons = gates.get("comparisons", {}) if isinstance(gates, dict) else {}
    if gates.get("passes_all") is not True or set(comparisons) != set(
        DEVELOPMENT_GATE_COMPARATORS
    ):
        raise PermissionError("development gate is incomplete")
    development_losses = result.get("development_losses")
    if not isinstance(development_losses, dict) or set(
        DEVELOPMENT_GATE_COMPARATORS
    ).union({"primary"}) - set(development_losses):
        raise ValueError("development donor-labeled losses are incomplete")
    loss_vectors: dict[str, np.ndarray] = {}
    for method in ("primary", *DEVELOPMENT_GATE_COMPARATORS):
        labeled = development_losses.get(method)
        if not isinstance(labeled, dict) or list(labeled) != list(DEVELOPMENT_DONORS):
            raise ValueError(f"development {method} donor labels differ")
        loss_vectors[method] = np.asarray(
            [
                _finite_number(labeled[donor], f"{method} {donor} loss")
                for donor in DEVELOPMENT_DONORS
            ]
        )
    generator = np.random.default_rng(SEED)
    indices = generator.integers(
        0,
        len(DEVELOPMENT_DONORS),
        size=(BOOTSTRAPS, len(DEVELOPMENT_DONORS)),
        endpoint=False,
    )
    for comparator in DEVELOPMENT_GATE_COMPARATORS:
        comparison = comparisons[comparator]
        if not isinstance(comparison, dict):
            raise ValueError(f"development comparison {comparator} is invalid")
        relative = _finite_number(
            comparison.get("relative_reduction"), f"{comparator} relative reduction"
        )
        upper = _finite_number(
            comparison.get("bootstrap_upper_95"), f"{comparator} bootstrap endpoint"
        )
        favorable = comparison.get("favorable_donors")
        if (
            not 0.0 <= relative <= 1.0
            or not isinstance(favorable, int)
            or isinstance(favorable, bool)
            or not 0 <= favorable <= 19
        ):
            raise ValueError(f"development comparison {comparator} has invalid ranges")
        if relative < 0.05 or upper >= 0 or favorable < 15:
            raise PermissionError(f"development gate fails against {comparator}")
        recomputed = _gate_comparison(
            DEVELOPMENT_DONORS,
            loss_vectors["primary"],
            loss_vectors[comparator],
            indices,
            favorable_required=15,
        )
        for key in (
            "relative_reduction",
            "bootstrap_upper_95",
            "favorable_donors",
            "donor_differences_primary_minus_comparator",
            "passes_all",
        ):
            if comparison.get(key) != recomputed[key]:
                raise PermissionError(
                    f"development gate {comparator} {key} was not recomputed"
                )
    methods = result.get("frozen_source_model", {}).get("methods", {})
    if set(methods) != set(REQUIRED_METHODS):
        raise ValueError("frozen method set differs")
    for name, method in methods.items():
        if name == "independence":
            continue
        coordinate = np.asarray(method.get("source_coordinate"), dtype=float)
        if coordinate.shape != (81,) or not np.isfinite(coordinate).all():
            raise ValueError(f"{name} source coordinate is invalid")
        if method.get("kind") == "conditional_log_odds":
            _validate_certificate(method.get("numerical_certificate"), name)
    residual = methods["best_residual"]
    if (
        residual.get("kind") != "classical_residual"
        or residual.get("family") not in {"pearson", "deviance"}
        or residual.get("sample_size_normalized") is not True
        or residual.get("normalization") != "source/sqrt(n), recipient*sqrt(m)"
    ):
        raise ValueError("classical residual is not the matched head-to-head")
    return result


def _expected_prediction_semantics(
    development: dict[str, Any], source_hash: str, source: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema": "gse279451-sepsis-exact-prediction/1.0",
        "status": "FROZEN_OUTCOME_ACCESS_DISABLED",
        "design": {
            "development_donors": list(DEVELOPMENT_DONORS),
            "held_donors": list(HELD_DONORS),
            "markers": list(MARKERS),
            "ordered_entities": 81,
            "cells_per_donor": CELL_BUDGET,
            "minimum_informative_entities": MINIMUM_INFORMATIVE_ENTITIES,
            "all_cells_sensitivity_permitted_before_decision": False,
        },
        "selection": development["selection"],
        "development_gate": development["gate"],
        "frozen_source_model": development["frozen_source_model"],
        "bindings": {
            "runner_sha256": _sha256(Path(__file__)),
            "reducer_sha256": _sha256(REDUCER),
            "development_evaluator_sha256": _sha256(EVALUATOR),
            "protocol_sha256": _sha256(PROTOCOL),
            "preflight_sha256": _sha256(PREFLIGHT),
            "candidate_designation_sha256": _sha256(DESIGNATION),
            "family_policy_sha256": _sha256(FAMILY_POLICY),
            "source_manifest_sha256": source_hash,
            "development_attempt_sha256": _sha256(DEVELOPMENT_ATTEMPT),
            "evaluation_attempt_sha256": _sha256(EVALUATION_ATTEMPT),
            "reduced_development_sha256": _sha256(REDUCED_DEVELOPMENT),
            "development_result_sha256": _sha256(DEVELOPMENT_RESULT),
            **_transitive_bindings(),
        },
        "source_summary": {
            "accession": source.get("accession"),
            "development_matrix_members": len(DEVELOPMENT_DONORS),
            "held_matrix_members": len(HELD_DONORS),
        },
        "held_access_audit": {
            "held_matrix_bytes_read": 0,
            "held_matrix_entries_decoded": 0,
            "held_margins_computed": 0,
            "held_tables_formed": 0,
            "all_cells_sensitivity_run": False,
        },
    }


def _validate_prediction_semantics(
    prediction: dict[str, Any],
    development: dict[str, Any],
    source_hash: str,
    source: dict[str, Any],
) -> None:
    expected = _expected_prediction_semantics(development, source_hash, source)
    if set(prediction) != {*expected, "created_at_utc"}:
        raise PermissionError("prediction fields differ from the frozen contract")
    if (
        not isinstance(prediction.get("created_at_utc"), str)
        or not prediction["created_at_utc"]
    ):
        raise PermissionError("prediction timestamp is absent")
    for key, value in expected.items():
        if prediction.get(key) != value:
            raise PermissionError(f"prediction {key} differs from development freeze")


def predict() -> dict[str, Any]:
    """Freeze the passing non-held model while held MTX access stays disabled."""

    _assert_family_available()
    if any(
        path.exists()
        for path in (PREDICTION, AUTHORIZATION, SCORE_ATTEMPT, OUTPUT, REFUSAL)
    ):
        raise FileExistsError("a GSE279451 confirmation artifact already exists")
    _validate_disabled_authorization_template()
    source = _validated_source()
    source_hash = _sha256(SOURCE_MANIFEST)
    development = _validated_development(source_hash, source)
    payload = {
        **_expected_prediction_semantics(development, source_hash, source),
        "created_at_utc": _timestamp(),
    }
    _validate_prediction_semantics(payload, development, source_hash, source)
    _write_json_exclusive(PREDICTION, payload)
    return payload


def _validated_authorization() -> tuple[dict[str, Any], _ScorePermit]:
    if not PREDICTION.is_file() or not AUTHORIZATION.is_file():
        raise PermissionError("prediction or active authorization is absent")
    prediction = _read_json(PREDICTION)
    authorization = _read_json(AUTHORIZATION)
    source = _validated_source()
    source_hash = _sha256(SOURCE_MANIFEST)
    development = _validated_development(source_hash, source)
    _validate_prediction_semantics(prediction, development, source_hash, source)
    if (
        prediction.get("status") != "FROZEN_OUTCOME_ACCESS_DISABLED"
        or authorization.get("status") != "OUTCOME_ACCESS_AUTHORIZED"
    ):
        raise PermissionError("held outcome access is disabled")
    expected = {
        "prediction_path": _relative(PREDICTION),
        "prediction_sha256": _sha256(PREDICTION),
        "runner_sha256": _sha256(Path(__file__)),
        "reducer_sha256": _sha256(REDUCER),
        "protocol_sha256": _sha256(PROTOCOL),
        "candidate_designation_sha256": _sha256(DESIGNATION),
        "family_policy_sha256": _sha256(FAMILY_POLICY),
        "source_manifest_sha256": _sha256(SOURCE_MANIFEST),
        "development_result_sha256": _sha256(DEVELOPMENT_RESULT),
        **_transitive_bindings(),
    }
    expected_fields = {
        "schema",
        "status",
        *expected,
        "public_prediction_commit",
        "public_prediction_url",
    }
    if (
        authorization.get("schema") != "gse279451-sepsis-score-authorization/1.0"
        or set(authorization) != expected_fields
    ):
        raise PermissionError("authorization fields differ")
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
    owner, repository = parts[0], parts[1]
    relative_prediction = "/".join(_relative(PREDICTION).split("/"))
    raw_url = (
        f"https://raw.githubusercontent.com/{owner}/{repository}/{commit}/"
        f"{relative_prediction}"
    )
    request = urllib.request.Request(
        raw_url, headers={"User-Agent": "coupling-fields/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            remote_bytes = response.read()
    except Exception as error:
        raise PermissionError("immutable public prediction fetch failed") from error
    local_bytes = PREDICTION.read_bytes()
    remote_hash = hashlib.sha256(remote_bytes).hexdigest()
    if remote_bytes != local_bytes or remote_hash != expected["prediction_sha256"]:
        raise PermissionError("immutable public prediction bytes differ")
    return prediction, _ScorePermit(
        expected["prediction_sha256"],
        commit,
        _sha256(AUTHORIZATION),
        url,
        remote_hash,
    )


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sanitized_error_message(error: Exception) -> str:
    message = str(error).replace(str(ROOT), "<repository>")
    message = re.sub(
        r"/(?:Users|home|private|tmp|var/folders)/[^\s'\"]+",
        "<local-path>",
        message,
    )
    message = re.sub(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "<email>",
        message,
    )
    return message[:500] or "held scorer refused without a message"


def _download_held_matrix(url: str, destination: Path, expected_bytes: int) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("held matrix destination already exists")
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    observed = 0
    request = urllib.request.Request(url, headers={"User-Agent": "coupling-fields/1.0"})
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            temporary.open("xb") as output,
        ):
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) != expected_bytes:
                raise PermissionError("held member remote byte count differs")
            for block in iter(lambda: response.read(1024 * 1024), b""):
                output.write(block)
                digest.update(block)
                observed += len(block)
        if observed != expected_bytes:
            raise PermissionError("held member byte count differs")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return digest.hexdigest()


def _canonical_margin_tables(rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    output = np.empty((81, 2, 2), dtype=np.int64)
    for entity in range(81):
        total = int(rows[entity].sum())
        upper_left = max(0, int(rows[entity, 0] + columns[entity, 0] - total))
        output[entity] = (
            (upper_left, int(rows[entity, 0] - upper_left)),
            (
                int(columns[entity, 0] - upper_left),
                int(rows[entity, 1] - columns[entity, 0] + upper_left),
            ),
        )
    return output


def _predict_from_margins(
    methods: dict[str, Any], margin_tables: np.ndarray
) -> dict[str, np.ndarray]:
    from experiments import evaluate_gse279451_sepsis_development as evaluator

    recipient = evaluator._conditional_support(margin_tables)
    predicted: dict[str, np.ndarray] = {}
    for name in REQUIRED_METHODS:
        method = methods[name]
        kind = method.get("kind")
        if kind == "conditional_log_odds":
            predicted[name] = evaluator._conditional_expected_tables(
                np.asarray(method["source_coordinate"], dtype=float), recipient
            )
        elif kind == "classical_residual":
            family = method["family"]
            predicted[name] = evaluator._predict_residual(
                np.asarray(method["source_coordinate"], dtype=float),
                margin_tables,
                family=family,
                centered=bool(method["centered"]),
                alpha=1.0,
                target_null=evaluator._target_null_mean(margin_tables, family),
            )
        elif kind == "independence":
            values = np.asarray(margin_tables, dtype=float)
            rows = values.sum(axis=-1)
            columns = values.sum(axis=-2)
            predicted[name] = rows[:, :, None] * columns[:, None, :] / CELL_BUDGET
        else:
            raise ValueError(f"frozen method {name} has an unknown kind")
    return predicted


def _exact_sign_flip_p(difference: np.ndarray) -> float:
    values = np.asarray(difference, dtype=float)
    if values.shape != (len(HELD_DONORS),) or not np.isfinite(values).all():
        raise ValueError("sign-flip test requires 21 finite paired differences")
    observed_sum = float(values.sum())
    inclusive = 0
    total = 1 << len(values)
    for start in range(0, total, 65536):
        codes = np.arange(start, min(start + 65536, total), dtype=np.uint32)
        bits = ((codes[:, None] >> np.arange(len(values))) & 1).astype(float)
        signed_sums = (2.0 * bits - 1.0) @ values
        inclusive += int(np.count_nonzero(signed_sums <= observed_sum))
    return inclusive / total


def _held_gate(losses: dict[str, list[float]]) -> dict[str, Any]:
    generator = np.random.default_rng(SEED)
    indices = generator.integers(
        0,
        len(HELD_DONORS),
        size=(BOOTSTRAPS, len(HELD_DONORS)),
        endpoint=False,
    )
    comparisons = {}
    for comparator in HELD_GATE_COMPARATORS:
        row = _gate_comparison(
            HELD_DONORS,
            np.asarray(losses["primary"]),
            np.asarray(losses[comparator]),
            indices,
            favorable_required=16,
        )
        difference = np.asarray(losses["primary"]) - np.asarray(losses[comparator])
        row["exact_one_sided_sign_flip_p"] = _exact_sign_flip_p(difference)
        row["sign_flip_statistic"] = "mean(primary minus comparator donor loss)"
        row["sign_flip_tail"] = "inclusive lower tail over all 2^21 sign vectors"
        row["sign_flip_zero_rule"] = "zero differences duplicate tied sign assignments"
        row["passes_all"] = bool(
            row["passes_all"] and row["exact_one_sided_sign_flip_p"] <= 0.025
        )
        comparisons[comparator] = row
    return {
        "comparisons": comparisons,
        "passes_all": all(row["passes_all"] for row in comparisons.values()),
    }


def _entity_support_report(informative: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(informative, dtype=bool)
    if mask.shape != (len(MARKERS) ** 2,):
        raise ValueError("held informative-entity mask has the wrong shape")
    excluded = []
    for index in np.flatnonzero(~mask):
        excluded.append(
            {
                "entity_index": int(index),
                "rna_marker": MARKERS[int(index) // len(MARKERS)],
                "adt_marker": MARKERS[int(index) % len(MARKERS)],
            }
        )
    return {
        "informative_entities": int(mask.sum()),
        "informative_entity_mask": mask.tolist(),
        "excluded_entities": excluded,
    }


def _score_held_once(
    prediction: dict[str, Any], permit: _ScorePermit
) -> dict[str, Any]:
    """Acquire, predict, and score one held donor at a time after attempt write."""

    global _LAST_HELD_AUDIT

    _LAST_HELD_AUDIT = {
        "held_donors_completed": [],
        "matrix_members_hashed": [],
        "prediction_materializations": [],
        "current_donor": None,
        "current_source_matrix_deleted": True,
        "cell_vectors_retained": False,
    }
    if not SCORE_ATTEMPT.is_file():
        raise PermissionError("held scorer requires the terminal attempt marker")
    source = _validated_source()
    template = _read_json(SOURCE_TEMPLATE)
    donor_records = {
        donor["accession"]: donor
        for donor in template["donors"]
        if isinstance(donor, dict)
    }
    methods = prediction.get("frozen_source_model", {}).get("methods", {})
    if set(methods) != set(REQUIRED_METHODS):
        raise ValueError("prediction method set differs")
    held_permit = reducer.HeldAccessPermit(
        prediction_sha256=permit.prediction_sha256,
        public_commit=permit.public_commit,
        attempt_path=SCORE_ATTEMPT,
    )
    losses: dict[str, list[float]] = {name: [] for name in REQUIRED_METHODS}
    donor_payloads = []
    for accession in HELD_DONORS:
        _LAST_HELD_AUDIT["current_donor"] = accession
        _LAST_HELD_AUDIT["current_source_matrix_deleted"] = True
        donor = donor_records[accession]
        sample = donor["sample"]
        filename = f"{accession}_{sample}.matrix.mtx.gz"
        url = (
            "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8571nnn/"
            f"{accession}/suppl/{filename}"
        )
        destination = HELD_MEMBER_DIR / filename
        matrix_hash = _download_held_matrix(
            url, destination, int(donor["matrix_bytes"])
        )
        _LAST_HELD_AUDIT["current_source_matrix_deleted"] = False
        _LAST_HELD_AUDIT["matrix_members_hashed"].append(
            {
                "accession": accession,
                "bytes": int(donor["matrix_bytes"]),
                "sha256": matrix_hash,
            }
        )
        try:
            working_source = copy.deepcopy(source)
            working_source["members"].append(
                {
                    "accession": accession,
                    "kind": "matrix",
                    "url": url,
                    "bytes": int(donor["matrix_bytes"]),
                    "local_path": destination.relative_to(ROOT).as_posix(),
                    "sha256": matrix_hash,
                    "retained": True,
                }
            )
            barcodes, features = reducer._axes(
                working_source,
                accession,
                phase="held_score_authorized",
                permit=held_permit,
            )
            selected_cells, selected_barcodes = reducer._budgeted_cells(
                barcodes, accession, sample
            )
            marker_rows = reducer._marker_rows(features)
            matrix_path = reducer._validated_member(
                working_source,
                accession,
                "matrix",
                phase="held_score_authorized",
                permit=held_permit,
            )

            rna_counts = reducer._stream_selected_rows(
                matrix_path,
                marker_rows["rna"],
                selected_cells,
                expected_features=len(features),
                expected_cells=len(barcodes),
            )
            rna_positive = (rna_counts > 0).sum(axis=1)
            del rna_counts
            rows = np.repeat(
                np.stack((CELL_BUDGET - rna_positive, rna_positive), axis=1),
                len(MARKERS),
                axis=0,
            )

            adt_counts = reducer._stream_selected_rows(
                matrix_path,
                marker_rows["adt"],
                selected_cells,
                expected_features=len(features),
                expected_cells=len(barcodes),
            )
            adt_states = reducer._adt_states(adt_counts, selected_barcodes, accession)
            adt_positive = adt_states.sum(axis=1)
            del adt_counts, adt_states
            columns = np.tile(
                np.stack((CELL_BUDGET - adt_positive, adt_positive), axis=1),
                (len(MARKERS), 1),
            )
            margin_tables = _canonical_margin_tables(rows, columns)
            predicted = _predict_from_margins(methods, margin_tables)
            prediction_payload = {
                "schema": "gse279451-sepsis-held-margin-predictions/1.0",
                "accession": accession,
                "prediction_sha256": permit.prediction_sha256,
                "matrix_sha256": matrix_hash,
                "selected_barcode_axis_sha256": hashlib.sha256(
                    ("\n".join(selected_barcodes) + "\n").encode()
                ).hexdigest(),
                "methods": {
                    name: values.reshape(81, 4).tolist()
                    for name, values in predicted.items()
                },
            }
            materialized_hash = _json_sha256(prediction_payload)
            prediction_payload["materialized_content_sha256"] = materialized_hash
            materialized_path = HELD_PREDICTION_DIR / f"{accession}.json"
            _write_json_exclusive(materialized_path, prediction_payload)
            reread_prediction = _read_json(materialized_path)
            embedded_hash = reread_prediction.pop("materialized_content_sha256", None)
            if (
                embedded_hash != materialized_hash
                or _json_sha256(reread_prediction) != materialized_hash
            ):
                raise PermissionError("held prediction materialization differs")
            _LAST_HELD_AUDIT["prediction_materializations"].append(
                {
                    "accession": accession,
                    "path": _relative(materialized_path),
                    "sha256": _sha256(materialized_path),
                    "content_sha256": materialized_hash,
                }
            )

            values = reducer._stream_selected_rows(
                matrix_path,
                marker_rows["rna"] + marker_rows["adt"],
                selected_cells,
                expected_features=len(features),
                expected_cells=len(barcodes),
            )
            rna = (values[: len(MARKERS)] > 0).astype(np.uint8)
            adt = reducer._adt_states(
                values[len(MARKERS) :], selected_barcodes, accession
            )
            truth = reducer._ordered_tables(rna, adt)
            del values, rna, adt
            if not np.array_equal(truth.sum(axis=-1), rows) or not np.array_equal(
                truth.sum(axis=-2), columns
            ):
                raise AssertionError("held truth changed a materialized margin")
            informative = np.asarray(
                [
                    bool(
                        table[0].sum()
                        and table[1].sum()
                        and table[:, 0].sum()
                        and table[:, 1].sum()
                    )
                    for table in truth
                ]
            )
            if int(informative.sum()) < MINIMUM_INFORMATIVE_ENTITIES:
                raise ValueError("held donor misses the prespecified support gate")
            support_report = _entity_support_report(informative)
            from experiments import evaluate_gse279451_sepsis_development as evaluator

            donor_losses = {}
            for name, fitted in predicted.items():
                loss = evaluator._donor_loss(truth, fitted, informative)
                losses[name].append(loss)
                donor_losses[name] = loss
            truth_hash = _json_sha256(truth.reshape(81, 4).tolist())
            donor_payloads.append(
                {
                    "accession": accession,
                    "sample": sample,
                    "cells": CELL_BUDGET,
                    **support_report,
                    "member_hashes": {
                        "barcodes": donor["barcode_sha256"],
                        "features": donor["feature_sha256"],
                        "matrix": matrix_hash,
                    },
                    "prediction_materialization_path": _relative(materialized_path),
                    "prediction_materialization_sha256": _sha256(materialized_path),
                    "prediction_content_sha256": materialized_hash,
                    "truth_tables_sha256": truth_hash,
                    "donor_equal_deviance": donor_losses,
                }
            )
            _LAST_HELD_AUDIT["held_donors_completed"].append(accession)
            del truth, predicted, margin_tables
        finally:
            destination.unlink(missing_ok=True)
            _LAST_HELD_AUDIT["current_source_matrix_deleted"] = not destination.exists()

    gate = _held_gate(losses)
    decision = "CONFIRMATION_PASS" if gate["passes_all"] else "CONFIRMATION_FAIL"
    return {
        "schema": "gse279451-sepsis-exact-confirmation/1.0",
        "status": decision,
        "created_at_utc": _timestamp(),
        "prediction_sha256": permit.prediction_sha256,
        "public_prediction_commit": permit.public_commit,
        "bindings": {
            "score_attempt_sha256": _sha256(SCORE_ATTEMPT),
            "authorization_sha256": _sha256(AUTHORIZATION),
            "runner_sha256": _sha256(Path(__file__)),
            "reducer_sha256": _sha256(REDUCER),
            "protocol_sha256": _sha256(PROTOCOL),
            "source_template_sha256": _sha256(SOURCE_TEMPLATE),
            "source_manifest_sha256": _sha256(SOURCE_MANIFEST),
            "development_attempt_sha256": _sha256(DEVELOPMENT_ATTEMPT),
            "evaluation_attempt_sha256": _sha256(EVALUATION_ATTEMPT),
            "development_result_sha256": _sha256(DEVELOPMENT_RESULT),
            **_transitive_bindings(),
        },
        "held_donors": list(HELD_DONORS),
        "inference_unit": "physical donor",
        "primary_cells_per_donor": CELL_BUDGET,
        "all_cells_sensitivity_used_for_decision": False,
        "donor_results": donor_payloads,
        "held_losses": {
            name: {donor: value for donor, value in zip(HELD_DONORS, values)}
            for name, values in losses.items()
        },
        "gate": gate,
        "access_audit": {
            "attempt_marker_preceded_first_held_download": True,
            "held_matrix_members_acquired": len(HELD_DONORS),
            "maximum_held_matrix_members_retained": 1,
            "all_source_matrices_deleted": True,
            "predictions_hashed_before_truth_per_donor": True,
            "cell_vectors_retained": False,
            "all_cells_sensitivity_run": False,
        },
    }


def score() -> dict[str, Any]:
    """Start the terminal one-shot held phase after immutable authorization."""

    global _LAST_HELD_AUDIT

    _LAST_HELD_AUDIT = None
    _assert_family_available()
    if any(path.exists() for path in (SCORE_ATTEMPT, OUTPUT, REFUSAL)):
        raise FileExistsError("a terminal GSE279451 scoring artifact already exists")
    prediction, permit = _validated_authorization()
    _write_json_exclusive(
        SCORE_ATTEMPT,
        {
            "schema": "gse279451-sepsis-score-attempt/1.0",
            "status": "TERMINAL_ATTEMPT_STARTED",
            "started_at_utc": _timestamp(),
            "prediction_sha256": permit.prediction_sha256,
            "public_prediction_commit": permit.public_commit,
            "authorization_sha256": permit.authorization_sha256,
            "public_prediction_url": permit.public_prediction_url,
            "remote_prediction_sha256": permit.remote_prediction_sha256,
            "transitive_bindings": _transitive_bindings(),
            "held_member_hashing_and_decode_start_after_this_write": True,
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
                "schema": "gse279451-sepsis-score-refusal/1.0",
                "status": "TERMINAL_SCORE_REFUSAL",
                "created_at_utc": _timestamp(),
                "error_type": type(error).__name__,
                "sanitized_error_message": _sanitized_error_message(error),
                "prediction_sha256": permit.prediction_sha256,
                "partial_audit": copy.deepcopy(_LAST_HELD_AUDIT)
                if _LAST_HELD_AUDIT is not None
                else {
                    "held_donors_completed": [],
                    "matrix_members_hashed": [],
                    "prediction_materializations": [],
                    "current_donor": None,
                    "current_source_matrix_deleted": True,
                    "cell_vectors_retained": False,
                },
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("predict", "score"))
    args = parser.parse_args()
    payload = predict() if args.phase == "predict" else score()
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
