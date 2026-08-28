"""Fail-closed prospective runner for the GSE299043 held-site confirmation.

``predict`` freezes a passing development model without opening a held H5AD.
``score`` accepts held access only after that prediction is available byte for
byte at an immutable public commit.  Scoring separates HTO census, RNA margin,
ADT margin, prediction materialization, and paired-truth access.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import json
import math
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse

import h5py
import numpy as np

from experiments import reduce_gse299043_mln as reducer


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_GITHUB_OWNER = "sushaan-k"
PUBLIC_GITHUB_REPOSITORY = "coupling-fields-benchmark"
PROTOCOL = ROOT / "docs/GSE299043_MLN_HELD_SITE_CONFIRMATION_PROTOCOL_2026-08-28.md"
DESIGNATION = ROOT / "data/confirmation/gse299043_mln/candidate_designation_v1.json"
FAMILY_POLICY = ROOT / "data/confirmation/gse299043_mln/family_policy_v1.json"
SOURCE_TEMPLATE = (
    ROOT / "data/confirmation/gse299043_mln/source_manifest_template_v1.json"
)
SOURCE_MANIFEST = ROOT / "data/confirmation/gse299043_mln/source_manifest_v1.json"
REDUCER = ROOT / "experiments/reduce_gse299043_mln.py"
EVALUATOR = ROOT / "experiments/evaluate_gse299043_mln_development.py"
DEVELOPMENT_ATTEMPT = (
    ROOT / "data/development/gse299043_mln/development_attempt_v1.json"
)
EVALUATION_ATTEMPT = ROOT / "data/development/gse299043_mln/evaluation_attempt_v1.json"
REDUCED_DEVELOPMENT = (
    ROOT / "data/development/gse299043_mln/reduced_development_v1.json"
)
DEVELOPMENT_RESULT = ROOT / "results/development/gse299043_mln_exact_development.json"
DEVELOPMENT_REFUSAL = ROOT / "results/development/gse299043_mln_evaluation_refusal.json"
DEVELOPMENT_ACQUISITION_REFUSAL = (
    ROOT / "results/development/gse299043_mln_development_acquisition_refusal.json"
)
PREDICTION = ROOT / "results/gse299043_mln_exact_predictions.json"
AUTH_TEMPLATE = (
    ROOT / "data/confirmation/gse299043_mln/score_authorization_template_v1.json"
)
AUTHORIZATION = ROOT / "data/confirmation/gse299043_mln/score_authorization_v1.json"
AUTH_PUBLICATION_TEMPLATE = (
    ROOT
    / "data/confirmation/gse299043_mln/score_authorization_publication_template_v1.json"
)
AUTH_PUBLICATION = (
    ROOT / "data/confirmation/gse299043_mln/score_authorization_publication_v1.json"
)
SCORE_ATTEMPT = ROOT / "data/confirmation/gse299043_mln/score_attempt_v1.json"
OUTPUT = ROOT / "results/gse299043_mln_exact_confirmation.json"
REFUSAL = ROOT / "results/gse299043_mln_exact_score_refusal.json"
HELD_MEMBER_DIR = ROOT / "data/confirmation/gse299043_mln/held_source_work"
HELD_PREDICTION_DIR = (
    ROOT / "data/confirmation/gse299043_mln/held_prediction_materializations"
)

MARKERS = reducer.MARKERS
DEVELOPMENT_DONORS = reducer.DEVELOPMENT_DONORS
HELD_DONORS = reducer.HELD_DONORS
CELL_BUDGET = reducer.CELL_BUDGET
MINIMUM_INFORMATIVE_ENTITIES = 64
REQUIRED_METHODS = (
    "primary",
    "best_residual",
    "destroyed_link",
    "hierarchical_ridge_only",
    "independence",
)
OPTIONAL_METHODS = ("label_permuted_graph",)
DEVELOPMENT_GATE_COMPARATORS = (
    "best_residual",
    "destroyed_link",
    "hierarchical_ridge_only",
)
HELD_GATE_COMPARATORS = ("best_residual", "destroyed_link")
CV_GRID = {
    "graph_neighborhood": [1, 2, 3],
    "heterogeneity_penalty": [0.1, 1.0, 10.0],
    "ridge_penalty": [0.01, 0.1],
    "graph_penalty": [0.1, 0.3, 1.0],
    "transport_multiplier": [0.75, 1.0, 1.25],
    "classical_residual": {
        "statistic": ["pearson", "deviance"],
        "exact_null_centered": [False, True],
    },
}
BOOTSTRAPS = 20_000
SEED = 20260828
DOWNLOAD_ATTEMPTS = 3
_LAST_HELD_AUDIT: dict[str, Any] | None = None


@dataclass(frozen=True)
class _ScorePermit:
    prediction_sha256: str
    public_commit: str
    authorization_sha256: str
    public_prediction_url: str
    remote_prediction_sha256: str
    authorization_publication_sha256: str = ""
    public_authorization_commit: str = ""
    public_authorization_url: str = ""
    remote_authorization_sha256: str = ""


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"JSON object contains duplicate key {key!r}")
        value[key] = item
    return value


def _decode_json_object(raw: str | bytes, label: str) -> dict[str, Any]:
    value = json.loads(
        raw,
        object_pairs_hook=_unique_json_object,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"{label} contains nonfinite JSON token {token}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return _decode_json_object(path.read_bytes(), path.name)


def _read_json_with_hash(path: Path) -> tuple[dict[str, Any], str, bytes]:
    raw = path.read_bytes()
    return _decode_json_object(raw, path.name), hashlib.sha256(raw).hexdigest(), raw


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    parent_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _binding(payload: dict[str, Any], key: str) -> object:
    if key in payload:
        return payload[key]
    bindings = payload.get("bindings")
    return bindings.get(key) if isinstance(bindings, dict) else None


def _transitive_bindings() -> dict[str, str]:
    from experiments import evaluate_gse299043_mln_development as evaluator

    bindings = evaluator._transitive_bindings()
    if not isinstance(bindings, dict) or not all(
        isinstance(key, str)
        and isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value)
        for key, value in bindings.items()
    ):
        raise PermissionError("evaluator transitive bindings are malformed")
    return bindings


def _finite_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _validate_certificate(value: object, name: str) -> None:
    if not isinstance(value, dict) or value.get("converged") is not True:
        raise ValueError(f"{name} numerical certificate is incomplete")
    numeric = (
        "scaled_gradient_norm",
        "gradient_tolerance",
        "schur_condition_number",
        "theta_curvature_condition_number",
    )
    parsed = {key: _finite_number(value.get(key), f"{name} {key}") for key in numeric}
    if (
        parsed["scaled_gradient_norm"] < 0.0
        or parsed["gradient_tolerance"] <= 0.0
        or parsed["scaled_gradient_norm"] > parsed["gradient_tolerance"]
        or parsed["schur_condition_number"] <= 0.0
        or parsed["theta_curvature_condition_number"] <= 0.0
        or max(
            parsed["schur_condition_number"],
            parsed["theta_curvature_condition_number"],
        )
        > 1e12
    ):
        raise ValueError(f"{name} numerical certificate misses its frozen limits")


def _assert_family_available() -> None:
    if not FAMILY_POLICY.is_file():
        raise PermissionError("GSE299043 family policy is absent")
    policy = _read_json(FAMILY_POLICY)
    terminal_path = ROOT / str(policy.get("required_terminal_artifact", ""))
    expected_hash = policy.get("required_terminal_artifact_sha256")
    terminal = _read_json(terminal_path) if terminal_path.is_file() else {}
    if (
        policy.get("schema") != "gse299043-mln-family-policy/1.0"
        or policy.get("status") != "ELIGIBLE_AFTER_PRIOR_TERMINAL_REFUSAL"
        or policy.get("prior_candidate") != "GSE279451"
        or policy.get("required_terminal_status")
        != "TERMINAL_DEVELOPMENT_EVALUATION_REFUSAL"
        or policy.get("prior_held_matrix_members_opened") != 0
        or policy.get("prior_rerun_permitted") is not False
        or policy.get("maximum_development_attempts") != 1
        or policy.get("maximum_held_score_attempts") != 1
        or not isinstance(expected_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
        or not terminal_path.is_file()
        or _sha256(terminal_path) != expected_hash
        or terminal.get("status") != policy.get("required_terminal_status")
        or terminal.get("held_matrix_members_opened") != 0
        or terminal.get("rerun_permitted") is not False
    ):
        raise PermissionError("prior-candidate terminal closure differs")


def _validate_member_static(active: dict[str, Any], frozen: dict[str, Any]) -> None:
    if set(active) != set(frozen):
        raise PermissionError("active source member fields differ from the freeze")
    for key, value in frozen.items():
        if key == "sha256":
            continue
        if active.get(key) != value:
            raise PermissionError(f"active source member {key} differs from the freeze")
    role = frozen.get("role")
    digest = active.get("sha256")
    if role == "development":
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise PermissionError("development source member lacks its observed hash")
    elif role == "held":
        if digest is not None:
            raise PermissionError("held member was hashed before score authorization")
    else:
        raise PermissionError("source member has an unknown role")


def _validated_source() -> dict[str, Any]:
    if not SOURCE_TEMPLATE.is_file() or not SOURCE_MANIFEST.is_file():
        raise PermissionError("active source manifest is absent")
    frozen = _read_json(SOURCE_TEMPLATE)
    source = _read_json(SOURCE_MANIFEST)
    if (
        frozen.get("schema") != "gse299043-mln-source/1.0"
        or frozen.get("status") != "SOURCE_UNAVAILABLE_OUTCOME_ACCESS_DISABLED"
        or source.get("schema") != frozen.get("schema")
        or source.get("status") != "NONHELD_SOURCE_ACCESS_AUTHORIZED"
    ):
        raise PermissionError("source manifest status differs from the freeze")
    frozen_members = frozen.get("members")
    active_members = source.get("members")
    if (
        not isinstance(frozen_members, list)
        or not isinstance(active_members, list)
        or len(frozen_members) != 207
        or len(active_members) != len(frozen_members)
    ):
        raise PermissionError("source manifest member count differs")
    identities: set[tuple[str, str]] = set()
    for active, template in zip(active_members, frozen_members):
        if not isinstance(active, dict) or not isinstance(template, dict):
            raise PermissionError("source member is not an object")
        _validate_member_static(active, template)
        identity = (str(active.get("donor")), str(active.get("filename")))
        if identity in identities:
            raise PermissionError("source manifest contains a duplicate member")
        identities.add(identity)
    for key in (
        "accession",
        "bioproject",
        "design",
        "h5ad_contract",
        "hashsolo_contract",
        "markers",
        "member_contract",
        "metadata_manifest",
        "study_doi",
    ):
        if source.get(key) != frozen.get(key):
            raise PermissionError(f"active source {key} differs from the freeze")
    audit = source.get("access_audit")
    bindings = source.get("bindings")
    expected_audit = {
        "development_members_opened_before_public_freeze": 0,
        "h5ad_members_opened_before_template_freeze": 0,
        "held_members_opened_before_public_prediction_authorization": 0,
        "matrix_entries_decoded_before_template_freeze": 0,
        "development_attempt_sha256": _sha256(DEVELOPMENT_ATTEMPT),
        "development_h5ad_members_requested": 56,
        "development_h5ad_members_decoded": 56,
        "held_h5ad_members_requested": 0,
        "held_h5ad_members_opened": 0,
        "held_h5ad_members_decoded": 0,
        "maximum_concurrent_source_h5ads": 1,
    }
    if not isinstance(audit, dict) or audit != expected_audit:
        raise PermissionError("source manifest records forbidden held access")
    expected_bindings = {
        "source_template_sha256": _sha256(SOURCE_TEMPLATE),
        "protocol_sha256": _sha256(PROTOCOL),
        "candidate_designation_sha256": _sha256(DESIGNATION),
        "family_policy_sha256": _sha256(FAMILY_POLICY),
        "score_authorization_publication_template_sha256": _sha256(
            AUTH_PUBLICATION_TEMPLATE
        ),
        "reducer_sha256": _sha256(REDUCER),
        "development_evaluator_sha256": _sha256(EVALUATOR),
        "runner_sha256": _sha256(Path(__file__)),
        **_transitive_bindings(),
    }
    if not isinstance(bindings, dict) or any(
        bindings.get(key) != value for key, value in expected_bindings.items()
    ):
        raise PermissionError("active source artifact bindings differ")
    return source


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
        or np.any(primary_values < 0.0)
        or np.any(comparator_values < 0.0)
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
        "bootstrap_indices_shared_across_comparisons": True,
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


def _validate_method_set(methods: object) -> dict[str, Any]:
    if (
        not isinstance(methods, dict)
        or not set(REQUIRED_METHODS).issubset(methods)
        or set(methods) - {*REQUIRED_METHODS, *OPTIONAL_METHODS}
    ):
        raise ValueError("frozen method set differs")
    for name, method in methods.items():
        if not isinstance(method, dict):
            raise ValueError(f"frozen method {name} is not an object")
        kind = method.get("kind")
        if name == "independence":
            if kind != "independence":
                raise ValueError("independence model kind differs")
            continue
        coordinate = np.asarray(method.get("source_coordinate"), dtype=float)
        if coordinate.shape != (81,) or not np.isfinite(coordinate).all():
            raise ValueError(f"{name} source coordinate is invalid")
        if name == "best_residual":
            if (
                kind != "classical_residual"
                or method.get("family") not in {"pearson", "deviance"}
                or not isinstance(method.get("centered"), bool)
                or method.get("sample_size_normalized") is not True
                or method.get("normalization") != "source/sqrt(n), recipient*sqrt(m)"
                or method.get("donor_equal_pooling") is not True
                or method.get("target_margin_inversion") is not True
                or method.get("target_null_restored") is not method.get("centered")
            ):
                raise ValueError("classical residual is not the matched head-to-head")
        elif kind != "conditional_log_odds":
            raise ValueError(f"frozen conditional method {name} kind differs")
        else:
            _validate_certificate(method.get("numerical_certificate"), name)
    return methods


def _validated_development(source_hash: str) -> dict[str, Any]:
    required = (
        DEVELOPMENT_ATTEMPT,
        EVALUATION_ATTEMPT,
        REDUCED_DEVELOPMENT,
        DEVELOPMENT_RESULT,
        REDUCER,
        EVALUATOR,
        PROTOCOL,
        DESIGNATION,
        FAMILY_POLICY,
    )
    if (
        DEVELOPMENT_REFUSAL.exists()
        or DEVELOPMENT_ACQUISITION_REFUSAL.exists()
        or not all(path.is_file() for path in required)
    ):
        raise PermissionError("passing development result is absent")
    attempt = _read_json(DEVELOPMENT_ATTEMPT)
    evaluation_attempt = _read_json(EVALUATION_ATTEMPT)
    reduced = _read_json(REDUCED_DEVELOPMENT)
    result = _read_json(DEVELOPMENT_RESULT)
    if (
        attempt.get("status") != "TERMINAL_DEVELOPMENT_ATTEMPT_STARTED"
        or attempt.get("source_template_sha256") != _sha256(SOURCE_TEMPLATE)
        or reduced.get("schema") != "gse299043-mln-reduced-development/1.0"
        or reduced.get("status") != "NONHELD_REDUCTION_COMPLETE"
        or reduced.get("source_manifest_sha256") != source_hash
        or reduced.get("development_attempt_sha256") != _sha256(DEVELOPMENT_ATTEMPT)
        or reduced.get("development_donors") != list(DEVELOPMENT_DONORS)
        or reduced.get("held_donors") != list(HELD_DONORS)
        or reduced.get("access_audit", {}).get("held_h5ad_members_opened") != 0
        or evaluation_attempt.get("status") != "TERMINAL_DEVELOPMENT_EVALUATION_STARTED"
        or _binding(evaluation_attempt, "reduced_development_sha256")
        != _sha256(REDUCED_DEVELOPMENT)
        or _binding(evaluation_attempt, "development_attempt_sha256")
        != _sha256(DEVELOPMENT_ATTEMPT)
        or _binding(evaluation_attempt, "evaluator_sha256") != _sha256(EVALUATOR)
        or _binding(evaluation_attempt, "gse299043_reducer_sha256") != _sha256(REDUCER)
        or _binding(evaluation_attempt, "protocol_sha256") != _sha256(PROTOCOL)
        or _binding(evaluation_attempt, "candidate_designation_sha256")
        != _sha256(DESIGNATION)
        or _binding(evaluation_attempt, "family_policy_sha256")
        != _sha256(FAMILY_POLICY)
        or _binding(evaluation_attempt, "source_manifest_sha256") != source_hash
        or any(
            _binding(evaluation_attempt, key) != value
            for key, value in _transitive_bindings().items()
        )
    ):
        raise PermissionError("development access or evaluation seal differs")
    if (
        result.get("schema") != "gse299043-mln-exact-development/1.0"
        or result.get("status") != "DEVELOPMENT_PASS"
        or result.get("markers") != list(MARKERS)
        or result.get("entity_count") != 81
        or result.get("cell_budget_per_donor") != CELL_BUDGET
        or _binding(result, "evaluation_attempt_sha256") != _sha256(EVALUATION_ATTEMPT)
        or _binding(result, "development_attempt_sha256")
        != _sha256(DEVELOPMENT_ATTEMPT)
        or _binding(result, "reduced_development_sha256")
        != _sha256(REDUCED_DEVELOPMENT)
        or _binding(result, "source_manifest_sha256") != source_hash
        or _binding(result, "evaluator_sha256") != _sha256(EVALUATOR)
        or _binding(result, "gse299043_reducer_sha256") != _sha256(REDUCER)
        or _binding(result, "protocol_sha256") != _sha256(PROTOCOL)
        or _binding(result, "candidate_designation_sha256") != _sha256(DESIGNATION)
        or _binding(result, "family_policy_sha256") != _sha256(FAMILY_POLICY)
        or any(
            _binding(result, key) != value
            for key, value in _transitive_bindings().items()
        )
    ):
        raise PermissionError("development result differs from the frozen experiment")
    selection = result.get("selection")
    if (
        not isinstance(selection, dict)
        or selection.get("folds") != len(DEVELOPMENT_DONORS)
        or selection.get("held_one_physical_donor_per_fold") is not True
        or selection.get("fold_donors") != list(DEVELOPMENT_DONORS)
        or selection.get("final_refit_donors") != list(DEVELOPMENT_DONORS)
        or selection.get("grid") != CV_GRID
    ):
        raise ValueError("development selection is not frozen ten-fold LOODO")
    gate = result.get("gate")
    comparisons = gate.get("comparisons") if isinstance(gate, dict) else None
    losses = result.get("development_losses")
    if (
        not isinstance(comparisons, dict)
        or set(comparisons) != set(DEVELOPMENT_GATE_COMPARATORS)
        or gate.get("passes_all") is not True
        or not isinstance(losses, dict)
    ):
        raise PermissionError("development gate is incomplete")
    vectors: dict[str, np.ndarray] = {}
    for method in ("primary", *DEVELOPMENT_GATE_COMPARATORS):
        labeled = losses.get(method)
        if not isinstance(labeled, dict) or list(labeled) != list(DEVELOPMENT_DONORS):
            raise ValueError(f"development {method} donor labels differ")
        vectors[method] = np.asarray(
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
        recomputed = _gate_comparison(
            DEVELOPMENT_DONORS,
            vectors["primary"],
            vectors[comparator],
            indices,
            favorable_required=8,
        )
        actual = comparisons[comparator]
        if not isinstance(actual, dict) or any(
            actual.get(key) != recomputed[key]
            for key in (
                "relative_reduction",
                "bootstrap_upper_95",
                "favorable_donors",
                "donor_differences_primary_minus_comparator",
                "passes_all",
            )
        ):
            raise PermissionError(f"development gate {comparator} was not recomputed")
    _validate_method_set(result.get("frozen_source_model", {}).get("methods"))
    return result


def _validate_disabled_authorization_template() -> None:
    if not AUTH_TEMPLATE.is_file():
        raise PermissionError("disabled authorization template is absent")
    template = _read_json(AUTH_TEMPLATE)
    bound_later = {
        "prediction_sha256",
        "public_prediction_commit",
        "public_prediction_url",
        "runner_sha256",
        "reducer_sha256",
        "development_evaluator_sha256",
        "protocol_sha256",
        "candidate_designation_sha256",
        "family_policy_sha256",
        "authorization_publication_template_sha256",
        "source_manifest_sha256",
        "development_result_sha256",
        *_transitive_bindings(),
    }
    if (
        template.get("schema") != "gse299043-mln-score-authorization/1.0"
        or template.get("status") != "OUTCOME_ACCESS_DISABLED"
        or template.get("prediction_path") != _relative(PREDICTION)
        or set(template) != {"schema", "status", "prediction_path", *bound_later}
        or any(template.get(key) is not None for key in bound_later)
    ):
        raise PermissionError("disabled authorization template differs")


def _validate_disabled_authorization_publication_template() -> None:
    if not AUTH_PUBLICATION_TEMPLATE.is_file():
        raise PermissionError("authorization publication template is absent")
    template = _read_json(AUTH_PUBLICATION_TEMPLATE)
    if template != {
        "schema": "gse299043-mln-score-authorization-publication/1.0",
        "status": "PUBLIC_AUTHORIZATION_UNAVAILABLE",
        "authorization_path": _relative(AUTHORIZATION),
        "authorization_sha256": None,
        "public_authorization_commit": None,
        "public_authorization_url": None,
    }:
        raise PermissionError("authorization publication template differs")


def _expected_prediction(
    development: dict[str, Any], source_hash: str, source: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema": "gse299043-mln-exact-prediction/1.0",
        "status": "FROZEN_OUTCOME_ACCESS_DISABLED",
        "design": {
            "development_donors": list(DEVELOPMENT_DONORS),
            "held_donors": list(HELD_DONORS),
            "markers": list(MARKERS),
            "ordered_entities": 81,
            "cells_per_donor": CELL_BUDGET,
            "minimum_informative_entities": MINIMUM_INFORMATIVE_ENTITIES,
        },
        "selection": development["selection"],
        "development_gate": development["gate"],
        "frozen_source_model": development["frozen_source_model"],
        "bindings": {
            "runner_sha256": _sha256(Path(__file__)),
            "reducer_sha256": _sha256(REDUCER),
            "development_evaluator_sha256": _sha256(EVALUATOR),
            "protocol_sha256": _sha256(PROTOCOL),
            "candidate_designation_sha256": _sha256(DESIGNATION),
            "family_policy_sha256": _sha256(FAMILY_POLICY),
            "authorization_publication_template_sha256": _sha256(
                AUTH_PUBLICATION_TEMPLATE
            ),
            "source_template_sha256": _sha256(SOURCE_TEMPLATE),
            "source_manifest_sha256": source_hash,
            "development_attempt_sha256": _sha256(DEVELOPMENT_ATTEMPT),
            "evaluation_attempt_sha256": _sha256(EVALUATION_ATTEMPT),
            "reduced_development_sha256": _sha256(REDUCED_DEVELOPMENT),
            "development_result_sha256": _sha256(DEVELOPMENT_RESULT),
            **_transitive_bindings(),
        },
        "source_summary": {
            "accession": source.get("accession"),
            "development_members": sum(
                member.get("role") == "development" for member in source["members"]
            ),
            "held_members": sum(
                member.get("role") == "held" for member in source["members"]
            ),
        },
        "held_access_audit": {
            "held_h5ad_members_requested": 0,
            "held_h5ad_members_opened": 0,
            "held_matrix_entries_decoded": 0,
            "held_predictions_materialized": 0,
            "held_truth_tables_formed": 0,
        },
    }


def _validate_prediction(
    prediction: dict[str, Any],
    development: dict[str, Any],
    source_hash: str,
    source: dict[str, Any],
) -> None:
    expected = _expected_prediction(development, source_hash, source)
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
    _validate_method_set(prediction["frozen_source_model"].get("methods"))


def predict() -> dict[str, Any]:
    """Freeze a passing development model without opening a held H5AD."""

    _assert_family_available()
    if any(
        path.exists()
        for path in (
            PREDICTION,
            AUTHORIZATION,
            AUTH_PUBLICATION,
            SCORE_ATTEMPT,
            OUTPUT,
            REFUSAL,
        )
    ):
        raise FileExistsError("a GSE299043 confirmation artifact already exists")
    _validate_disabled_authorization_template()
    _validate_disabled_authorization_publication_template()
    source = _validated_source()
    source_hash = _sha256(SOURCE_MANIFEST)
    development = _validated_development(source_hash)
    payload = {
        **_expected_prediction(development, source_hash, source),
        "created_at_utc": _timestamp(),
    }
    _validate_prediction(payload, development, source_hash, source)
    _write_json_exclusive(PREDICTION, payload)
    return payload


def _validated_authorization() -> tuple[dict[str, Any], _ScorePermit]:
    if (
        not PREDICTION.is_file()
        or not AUTHORIZATION.is_file()
        or not AUTH_PUBLICATION.is_file()
    ):
        raise PermissionError(
            "prediction, active authorization, or publication sidecar is absent"
        )
    _validate_disabled_authorization_template()
    _validate_disabled_authorization_publication_template()
    source = _validated_source()
    source_hash = _sha256(SOURCE_MANIFEST)
    development = _validated_development(source_hash)
    prediction, prediction_hash, local_bytes = _read_json_with_hash(PREDICTION)
    authorization, authorization_hash, local_authorization = _read_json_with_hash(
        AUTHORIZATION
    )
    _validate_prediction(prediction, development, source_hash, source)
    expected = {
        "prediction_path": _relative(PREDICTION),
        "prediction_sha256": prediction_hash,
        "runner_sha256": _sha256(Path(__file__)),
        "reducer_sha256": _sha256(REDUCER),
        "development_evaluator_sha256": _sha256(EVALUATOR),
        "protocol_sha256": _sha256(PROTOCOL),
        "candidate_designation_sha256": _sha256(DESIGNATION),
        "family_policy_sha256": _sha256(FAMILY_POLICY),
        "authorization_publication_template_sha256": _sha256(AUTH_PUBLICATION_TEMPLATE),
        "source_manifest_sha256": source_hash,
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
        prediction.get("status") != "FROZEN_OUTCOME_ACCESS_DISABLED"
        or authorization.get("schema") != "gse299043-mln-score-authorization/1.0"
        or authorization.get("status") != "OUTCOME_ACCESS_AUTHORIZED"
        or set(authorization) != expected_fields
    ):
        raise PermissionError("held outcome authorization fields differ")
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
        or parsed.query
        or parsed.fragment
        or len(parts) != len(expected_tail) + 2
        or parts[:2] != [PUBLIC_GITHUB_OWNER, PUBLIC_GITHUB_REPOSITORY]
        or parts[-len(expected_tail) :] != expected_tail
    ):
        raise PermissionError("public prediction URL is not the bound GitHub blob")
    owner, repository = parts[0], parts[1]
    raw_url = (
        f"https://raw.githubusercontent.com/{owner}/{repository}/{commit}/"
        f"{_relative(PREDICTION)}"
    )
    request = urllib.request.Request(
        raw_url, headers={"User-Agent": "coupling-fields/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            remote_bytes = response.read()
    except Exception as error:
        raise PermissionError("immutable public prediction fetch failed") from error
    remote_hash = hashlib.sha256(remote_bytes).hexdigest()
    if remote_bytes != local_bytes or remote_hash != expected["prediction_sha256"]:
        raise PermissionError("immutable public prediction bytes differ")
    publication, publication_hash, _ = _read_json_with_hash(AUTH_PUBLICATION)
    publication_commit = publication.get("public_authorization_commit")
    publication_url = publication.get("public_authorization_url")
    if (
        set(publication)
        != {
            "schema",
            "status",
            "authorization_path",
            "authorization_sha256",
            "public_authorization_commit",
            "public_authorization_url",
        }
        or publication.get("schema")
        != "gse299043-mln-score-authorization-publication/1.0"
        or publication.get("status") != "PUBLIC_AUTHORIZATION_AVAILABLE"
        or publication.get("authorization_path") != _relative(AUTHORIZATION)
        or publication.get("authorization_sha256") != authorization_hash
        or not isinstance(publication_commit, str)
        or not re.fullmatch(r"[0-9a-f]{40}", publication_commit)
        or not isinstance(publication_url, str)
    ):
        raise PermissionError("public authorization sidecar differs")
    publication_parsed = urlparse(publication_url)
    publication_parts = [
        unquote(part) for part in publication_parsed.path.split("/") if part
    ]
    publication_tail = [
        "blob",
        publication_commit,
        *_relative(AUTHORIZATION).split("/"),
    ]
    if (
        publication_parsed.scheme != "https"
        or publication_parsed.netloc != "github.com"
        or publication_parsed.query
        or publication_parsed.fragment
        or len(publication_parts) != len(publication_tail) + 2
        or publication_parts[:2] != [PUBLIC_GITHUB_OWNER, PUBLIC_GITHUB_REPOSITORY]
        or publication_parts[-len(publication_tail) :] != publication_tail
    ):
        raise PermissionError("public authorization URL is not the bound GitHub blob")
    publication_owner, publication_repository = publication_parts[:2]
    raw_authorization_url = (
        "https://raw.githubusercontent.com/"
        f"{publication_owner}/{publication_repository}/{publication_commit}/"
        f"{_relative(AUTHORIZATION)}"
    )
    authorization_request = urllib.request.Request(
        raw_authorization_url, headers={"User-Agent": "coupling-fields/1.0"}
    )
    try:
        with urllib.request.urlopen(authorization_request, timeout=120) as response:
            remote_authorization = response.read()
    except Exception as error:
        raise PermissionError("immutable public authorization fetch failed") from error
    remote_authorization_hash = hashlib.sha256(remote_authorization).hexdigest()
    if (
        remote_authorization != local_authorization
        or remote_authorization_hash != authorization_hash
    ):
        raise PermissionError("immutable public authorization bytes differ")
    return prediction, _ScorePermit(
        prediction_sha256=expected["prediction_sha256"],
        public_commit=commit,
        authorization_sha256=authorization_hash,
        public_prediction_url=url,
        remote_prediction_sha256=remote_hash,
        authorization_publication_sha256=publication_hash,
        public_authorization_commit=publication_commit,
        public_authorization_url=publication_url,
        remote_authorization_sha256=remote_authorization_hash,
    )


def _score_attempt_payload(permit: _ScorePermit, started_at_utc: str) -> dict[str, Any]:
    return {
        "schema": "gse299043-mln-score-attempt/1.0",
        "status": "TERMINAL_ATTEMPT_STARTED",
        "started_at_utc": started_at_utc,
        "prediction_sha256": permit.prediction_sha256,
        "public_prediction_commit": permit.public_commit,
        "authorization_sha256": permit.authorization_sha256,
        "authorization_publication_sha256": permit.authorization_publication_sha256,
        "public_prediction_url": permit.public_prediction_url,
        "remote_prediction_sha256": permit.remote_prediction_sha256,
        "public_authorization_commit": permit.public_authorization_commit,
        "public_authorization_url": permit.public_authorization_url,
        "remote_authorization_sha256": permit.remote_authorization_sha256,
        "held_member_request_and_decode_start_after_this_write": True,
    }


def _validated_score_attempt(permit: _ScorePermit) -> str:
    if not all(
        path.is_file()
        for path in (SCORE_ATTEMPT, PREDICTION, AUTHORIZATION, AUTH_PUBLICATION)
    ):
        raise PermissionError("terminal score seal is incomplete")
    attempt, attempt_hash, _ = _read_json_with_hash(SCORE_ATTEMPT)
    expected = _score_attempt_payload(permit, str(attempt.get("started_at_utc", "")))
    publication = _read_json(AUTH_PUBLICATION)
    expected_publication = {
        "schema": "gse299043-mln-score-authorization-publication/1.0",
        "status": "PUBLIC_AUTHORIZATION_AVAILABLE",
        "authorization_path": _relative(AUTHORIZATION),
        "authorization_sha256": permit.authorization_sha256,
        "public_authorization_commit": permit.public_authorization_commit,
        "public_authorization_url": permit.public_authorization_url,
    }
    if (
        not isinstance(attempt.get("started_at_utc"), str)
        or not attempt["started_at_utc"]
        or attempt != expected
        or permit.remote_prediction_sha256 != permit.prediction_sha256
        or permit.remote_authorization_sha256 != permit.authorization_sha256
        or _sha256(PREDICTION) != permit.prediction_sha256
        or _sha256(AUTHORIZATION) != permit.authorization_sha256
        or _sha256(AUTH_PUBLICATION) != permit.authorization_publication_sha256
        or publication != expected_publication
    ):
        raise PermissionError("terminal score seal differs from public authorization")
    return attempt_hash


def _held_members(
    source: dict[str, Any], donor: str, permit: reducer.HeldAccessPermit
) -> list[dict[str, Any]]:
    reducer._authorize_access(donor, "held_score_authorized", permit)
    records = source.get("members")
    if not isinstance(records, list):
        raise PermissionError("source member list is unavailable")
    selected: list[dict[str, Any]] = []
    filenames: set[str] = set()
    for member in records:
        if not isinstance(member, dict) or member.get("donor") != donor:
            continue
        filename = member.get("filename")
        url = member.get("url")
        expected_bytes = member.get("bytes")
        if (
            member.get("role") != "held"
            or not isinstance(filename, str)
            or Path(filename).name != filename
            or filename in filenames
            or not isinstance(url, str)
            or unquote(urlparse(url).path).split("/")[-1] != filename
            or not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes <= 0
            or member.get("sha256") is not None
            or member.get("local_path") is not None
            or member.get("retained") is not False
        ):
            raise PermissionError("held source member role or identity differs")
        filenames.add(filename)
        selected.append(member)
    if not selected:
        raise PermissionError(f"held donor {donor} has no frozen source members")
    return selected


def _validate_live_held_permit(donor: str, permit: reducer.HeldAccessPermit) -> None:
    if not SCORE_ATTEMPT.is_file():
        raise PermissionError("terminal score attempt must precede held access")
    reducer._authorize_access(donor, "held_score_authorized", permit)
    if (
        not PREDICTION.is_file()
        or not AUTHORIZATION.is_file()
        or not AUTH_PUBLICATION.is_file()
    ):
        raise PermissionError(
            "held prediction, authorization, or publication sidecar disappeared"
        )
    attempt = _read_json(SCORE_ATTEMPT)
    authorization = _read_json(AUTHORIZATION)
    publication = _read_json(AUTH_PUBLICATION)
    expected_attempt_fields = {
        "schema",
        "status",
        "started_at_utc",
        "prediction_sha256",
        "public_prediction_commit",
        "authorization_sha256",
        "authorization_publication_sha256",
        "public_prediction_url",
        "remote_prediction_sha256",
        "public_authorization_commit",
        "public_authorization_url",
        "remote_authorization_sha256",
        "held_member_request_and_decode_start_after_this_write",
    }
    expected_publication = {
        "schema": "gse299043-mln-score-authorization-publication/1.0",
        "status": "PUBLIC_AUTHORIZATION_AVAILABLE",
        "authorization_path": _relative(AUTHORIZATION),
        "authorization_sha256": permit.authorization_sha256,
        "public_authorization_commit": attempt.get("public_authorization_commit"),
        "public_authorization_url": attempt.get("public_authorization_url"),
    }
    if (
        permit.prediction_sha256 != _sha256(PREDICTION)
        or permit.authorization_sha256 != _sha256(AUTHORIZATION)
        or permit.terminal_attempt_sha256 != _sha256(SCORE_ATTEMPT)
        or set(attempt) != expected_attempt_fields
        or attempt.get("schema") != "gse299043-mln-score-attempt/1.0"
        or attempt.get("status") != "TERMINAL_ATTEMPT_STARTED"
        or not isinstance(attempt.get("started_at_utc"), str)
        or not attempt["started_at_utc"]
        or attempt.get("prediction_sha256") != permit.prediction_sha256
        or attempt.get("public_prediction_commit") != permit.public_commit
        or attempt.get("authorization_sha256") != permit.authorization_sha256
        or attempt.get("authorization_publication_sha256") != _sha256(AUTH_PUBLICATION)
        or attempt.get("remote_prediction_sha256") != permit.prediction_sha256
        or attempt.get("remote_authorization_sha256") != permit.authorization_sha256
        or attempt.get("held_member_request_and_decode_start_after_this_write")
        is not True
        or publication != expected_publication
        or authorization.get("status") != "OUTCOME_ACCESS_AUTHORIZED"
    ):
        raise PermissionError("live held-access permit differs from the terminal seal")


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


def _download_held_member(
    member: dict[str, Any],
    destination: Path,
    permit: reducer.HeldAccessPermit,
) -> str:
    _validate_live_held_permit(str(member.get("donor")), permit)
    if destination.parent != HELD_MEMBER_DIR or destination.name != member["filename"]:
        raise PermissionError("held member destination escapes the work directory")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("held member destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        member["url"], headers={"User-Agent": "coupling-fields/1.0"}
    )
    for attempt in range(DOWNLOAD_ATTEMPTS):
        temporary.unlink(missing_ok=True)
        if attempt:
            _validate_live_held_permit(str(member.get("donor")), permit)
        digest = hashlib.sha256()
        observed = 0
        try:
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                temporary.open("xb") as output,
            ):
                content_length = response.headers.get("Content-Length")
                if (
                    content_length is not None
                    and int(content_length) != member["bytes"]
                ):
                    raise PermissionError("held member remote byte count differs")
                for block in iter(lambda: response.read(1 << 20), b""):
                    output.write(block)
                    digest.update(block)
                    observed += len(block)
            if observed != member["bytes"]:
                raise PermissionError("held member byte count differs")
            os.replace(temporary, destination)
            return digest.hexdigest()
        except Exception as error:
            temporary.unlink(missing_ok=True)
            if attempt + 1 == DOWNLOAD_ATTEMPTS or not _retryable_transport_error(
                error
            ):
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable held-download retry state")


def _visit_member(
    member: dict[str, Any],
    pass_name: str,
    audit: dict[str, Any],
    reader: Callable[[Path], Any],
    permit: reducer.HeldAccessPermit,
) -> Any:
    destination = HELD_MEMBER_DIR / member["filename"]
    digest = _download_held_member(member, destination, permit)
    audit["current_member"] = member["filename"]
    audit["current_member_deleted"] = False
    first_hash = audit["member_content_sha256"].setdefault(member["filename"], digest)
    if first_hash != digest:
        destination.unlink(missing_ok=True)
        audit["current_member_deleted"] = True
        raise PermissionError("held source member changed between sealed passes")
    audit["member_passes"].append(
        {"filename": member["filename"], "pass": pass_name, "sha256": digest}
    )
    try:
        return reader(destination)
    finally:
        destination.unlink(missing_ok=True)
        audit["current_member_deleted"] = not destination.exists()


def _matrix_selected_values(
    matrix: h5py.Dataset | h5py.Group,
    rows: list[int],
    columns: list[int],
    expected_shape: tuple[int, int],
) -> np.ndarray:
    if len(rows) != len(set(rows)) or len(columns) != len(set(columns)):
        raise ValueError("selected H5AD rows and columns must be unique")
    if any(row < 0 or row >= expected_shape[0] for row in rows) or any(
        column < 0 or column >= expected_shape[1] for column in columns
    ):
        raise ValueError("selected H5AD matrix index is out of bounds")
    if reducer._matrix_shape(matrix) != expected_shape:
        raise ValueError("H5AD X shape differs from obs/var")
    output = np.zeros((len(rows), len(columns)), dtype=np.float64)
    if isinstance(matrix, h5py.Dataset):
        column_order = np.argsort(columns)
        sorted_columns = np.asarray(columns, dtype=np.int64)[column_order]
        inverse = np.argsort(column_order)
        for target_row, source_row in enumerate(rows):
            output[target_row] = np.asarray(matrix[source_row, sorted_columns])[inverse]
    else:
        encoding = reducer._attribute_text(matrix.attrs.get("encoding-type", ""))
        if encoding not in {"csr_matrix", "csc_matrix"} or not {
            "data",
            "indices",
            "indptr",
        }.issubset(matrix):
            raise ValueError("H5AD X is not a supported dense/CSR/CSC matrix")
        data = matrix["data"]
        indices = matrix["indices"]
        indptr = np.asarray(matrix["indptr"][...], dtype=np.int64)
        major = expected_shape[0] if encoding == "csr_matrix" else expected_shape[1]
        if (
            data.ndim != 1
            or indices.ndim != 1
            or len(data) != len(indices)
            or len(indptr) != major + 1
            or indptr[0] != 0
            or indptr[-1] != len(data)
            or np.any(np.diff(indptr) < 0)
        ):
            raise ValueError("H5AD sparse X structural arrays are invalid")
        if encoding == "csr_matrix":
            lookup = {column: target for target, column in enumerate(columns)}
            for target_row, source_row in enumerate(rows):
                start, end = int(indptr[source_row]), int(indptr[source_row + 1])
                source_columns = np.asarray(indices[start:end], dtype=np.int64)
                if np.any(source_columns < 0) or np.any(
                    source_columns >= expected_shape[1]
                ):
                    raise ValueError("H5AD CSR column index is out of bounds")
                for offset, source_column in enumerate(source_columns):
                    target_column = lookup.get(int(source_column))
                    if target_column is not None:
                        output[target_row, target_column] += data[start + offset]
        else:
            row_lookup = {row: target for target, row in enumerate(rows)}
            for target_column, source_column in enumerate(columns):
                start, end = int(indptr[source_column]), int(indptr[source_column + 1])
                source_rows = np.asarray(indices[start:end], dtype=np.int64)
                if np.any(source_rows < 0) or np.any(source_rows >= expected_shape[0]):
                    raise ValueError("H5AD CSC row index is out of bounds")
                for offset, source_row in enumerate(source_rows):
                    target_row = row_lookup.get(int(source_row))
                    if target_row is not None:
                        output[target_row, target_column] += data[start + offset]
    if (
        not np.isfinite(output).all()
        or np.any(output < 0)
        or not np.array_equal(output, np.rint(output))
    ):
        raise ValueError("selected H5AD values are not finite nonnegative integers")
    return output.astype(np.int64)


def _h5ad_axes(
    handle: h5py.File,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not {"X", "obs", "var"}.issubset(handle):
        raise ValueError("H5AD lacks X, obs, or var")
    if not isinstance(handle["obs"], h5py.Group) or not isinstance(
        handle["var"], h5py.Group
    ):
        raise ValueError("H5AD obs/var are not data-frame groups")
    barcodes = reducer._dataframe_index(handle["obs"])
    names = reducer._dataframe_index(handle["var"])
    gene_ids = reducer._dataframe_column(handle["var"], "gene_ids")
    feature_types = reducer._dataframe_column(handle["var"], "feature_types")
    if len(barcodes) == 0 or len(set(barcodes)) != len(barcodes):
        raise ValueError("H5AD barcode axis is empty or nonunique")
    if not (len(names) == len(gene_ids) == len(feature_types)):
        raise ValueError("H5AD feature axes differ")
    return barcodes, names, gene_ids, feature_types


def _hto_candidates(
    path: Path,
    member: dict[str, Any],
    donor: str,
    permit: reducer.HeldAccessPermit,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    _validate_live_held_permit(donor, permit)
    library, filename = reducer._filename_binding(path, donor)
    if filename != member["filename"] or library != member["gex_library"]:
        raise PermissionError("held H5AD filename differs from its source member")
    with h5py.File(path, "r") as handle:
        barcodes, names, gene_ids, feature_types = _h5ad_axes(handle)
        hto_columns = [
            index
            for index, (feature_id, feature_type) in enumerate(
                zip(gene_ids, feature_types)
            )
            if feature_type == "Antibody Capture"
            and str(feature_id).startswith(donor + "-")
        ]
        raw_tags = [str(gene_ids[index]) for index in hto_columns]
        if not hto_columns or len(raw_tags) != len(set(raw_tags)):
            raise ValueError("held H5AD donor HTO axis is empty or nonunique")
        normalized = [
            reducer._normalize_hto_id(donor, library, tag) for tag in raw_tags
        ]
        if len(normalized) != len(set(normalized)):
            raise ValueError("HTO normalization created duplicate donor tags")
        single_tissue_assignment = reducer._uses_single_tissue_one_hto_exception(
            donor, filename, normalized
        )
        target_tags = set(reducer.MLN_TAGS[donor])
        present_targets = sorted(target_tags.intersection(normalized))
        if not present_targets:
            raise ValueError("held H5AD lacks the frozen MLN HTO")
        counts = reducer._matrix_columns(
            handle["X"], hto_columns, (len(barcodes), len(names))
        )
    classifications = (
        np.full(len(barcodes), normalized[0], dtype=object).astype(str)
        if single_tissue_assignment
        else reducer._hashsolo_classifications(counts, normalized)
    )
    del counts
    retained = np.flatnonzero(np.isin(classifications, present_targets))
    candidates = [
        {
            "filename": filename,
            "barcode": str(barcodes[cell]),
            "assigned_mln_tag": str(classifications[cell]),
            "cell_selection_sha256": reducer._cell_selection_hash(
                donor, filename, str(barcodes[cell])
            ),
        }
        for cell in retained
    ]
    candidates.sort(
        key=lambda record: (
            record["cell_selection_sha256"],
            record["filename"],
            record["barcode"],
        )
    )
    return candidates[:CELL_BUDGET], {
        "filename": filename,
        "deposited_cells": len(barcodes),
        "target_mln_singlets": len(retained),
        "hto_features": len(normalized),
    }


def _selected_marker_counts(
    path: Path,
    member: dict[str, Any],
    donor: str,
    selected_barcodes: list[str],
    modality: str,
    permit: reducer.HeldAccessPermit,
) -> np.ndarray:
    _validate_live_held_permit(donor, permit)
    library, filename = reducer._filename_binding(path, donor)
    if filename != member["filename"] or library != member["gex_library"]:
        raise PermissionError("held H5AD filename differs from its source member")
    if modality == "rna":
        specifications = ((reducer.RNA_FEATURE_IDS, "Gene Expression", "rna"),)
    elif modality == "adt":
        specifications = ((reducer.ADT_FEATURE_IDS, "Antibody Capture", "adt"),)
    elif modality == "paired":
        specifications = (
            (reducer.RNA_FEATURE_IDS, "Gene Expression", "rna"),
            (reducer.ADT_FEATURE_IDS, "Antibody Capture", "adt"),
        )
    else:
        raise ValueError("unknown held marker pass")
    with h5py.File(path, "r") as handle:
        barcodes, names, gene_ids, feature_types = _h5ad_axes(handle)
        columns = []
        for identifiers, expected_type, label in specifications:
            for marker, identifier in zip(MARKERS, identifiers):
                matches = np.flatnonzero(
                    (gene_ids == identifier) & (feature_types == expected_type)
                )
                if len(matches) != 1:
                    raise ValueError(f"held marker {marker} lacks one exact {label} ID")
                columns.append(int(matches[0]))
        barcode_lookup = {str(barcode): index for index, barcode in enumerate(barcodes)}
        if len(barcode_lookup) != len(barcodes) or any(
            barcode not in barcode_lookup for barcode in selected_barcodes
        ):
            raise PermissionError("selected cell is absent from a repeated held pass")
        rows = [barcode_lookup[barcode] for barcode in selected_barcodes]
        values = _matrix_selected_values(
            handle["X"], rows, columns, (len(barcodes), len(names))
        )
    return values.T


def _census_donor(
    donor: str,
    members: list[dict[str, Any]],
    audit: dict[str, Any],
    permit: reducer.HeldAccessPermit,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    candidates: list[dict[str, str]] = []
    member_summaries = []
    cells: set[tuple[str, str]] = set()
    for member in members:
        records, summary = _visit_member(
            member,
            "hto_census",
            audit,
            lambda path, current=member: _hto_candidates(path, current, donor, permit),
            permit,
        )
        for record in records:
            identity = (record["filename"], record["barcode"])
            if identity in cells:
                raise PermissionError("HTO census contains a duplicate cell identity")
            cells.add(identity)
        candidates.extend(records)
        member_summaries.append(summary)
    candidates.sort(
        key=lambda record: (
            record["cell_selection_sha256"],
            record["filename"],
            record["barcode"],
        )
    )
    if len(candidates) < CELL_BUDGET:
        raise ValueError(f"{donor} has fewer than {CELL_BUDGET} MLN singlets")
    selected = candidates[:CELL_BUDGET]
    axis_hash = hashlib.sha256(
        (
            "\n".join(
                f"{record['filename']}\t{record['barcode']}" for record in selected
            )
            + "\n"
        ).encode()
    ).hexdigest()
    return selected, {
        "source_members": len(members),
        "candidate_mln_singlets_retained": len(candidates),
        "selected_cells": CELL_BUDGET,
        "selected_cell_axis_sha256": axis_hash,
        "member_census": member_summaries,
    }


def _selected_by_member(
    selected: list[dict[str, str]],
) -> dict[str, list[tuple[int, str]]]:
    grouped: dict[str, list[tuple[int, str]]] = {}
    for index, record in enumerate(selected):
        grouped.setdefault(record["filename"], []).append((index, record["barcode"]))
    return grouped


def _marker_pass(
    donor: str,
    members: list[dict[str, Any]],
    selected: list[dict[str, str]],
    modality: str,
    audit: dict[str, Any],
    permit: reducer.HeldAccessPermit,
) -> np.ndarray:
    grouped = _selected_by_member(selected)
    values = np.zeros(
        ((18 if modality == "paired" else 9), CELL_BUDGET), dtype=np.int64
    )
    seen = np.zeros(CELL_BUDGET, dtype=bool)
    for member in members:
        selected_rows = grouped.get(member["filename"], [])
        if not selected_rows:
            continue
        positions = [position for position, _ in selected_rows]
        barcodes = [barcode for _, barcode in selected_rows]
        current = _visit_member(
            member,
            f"{modality}_pass",
            audit,
            lambda path, current_member=member: _selected_marker_counts(
                path, current_member, donor, barcodes, modality, permit
            ),
            permit,
        )
        expected_shape = (values.shape[0], len(positions))
        if current.shape != expected_shape:
            raise ValueError(f"held {modality} pass returned the wrong shape")
        values[:, positions] = current
        seen[positions] = True
        del current
    if not np.all(seen):
        raise PermissionError(f"held {modality} pass missed a selected cell")
    return values


def _margin_tables(rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    row_values = np.asarray(rows, dtype=np.int64)
    column_values = np.asarray(columns, dtype=np.int64)
    if (
        row_values.shape != (81, 2)
        or column_values.shape != (81, 2)
        or np.any(row_values < 0)
        or np.any(column_values < 0)
        or not np.all(row_values.sum(axis=1) == CELL_BUDGET)
        or not np.all(column_values.sum(axis=1) == CELL_BUDGET)
    ):
        raise ValueError("held margins are malformed")
    tables = np.empty((81, 2, 2), dtype=np.int64)
    for entity in range(81):
        upper_left = max(
            0,
            int(row_values[entity, 0] + column_values[entity, 0] - CELL_BUDGET),
        )
        tables[entity] = (
            (upper_left, int(row_values[entity, 0] - upper_left)),
            (
                int(column_values[entity, 0] - upper_left),
                int(row_values[entity, 1] - column_values[entity, 0] + upper_left),
            ),
        )
    return tables


def _rna_margins(
    donor: str,
    members: list[dict[str, Any]],
    selected: list[dict[str, str]],
    audit: dict[str, Any],
    permit: reducer.HeldAccessPermit,
) -> np.ndarray:
    counts = _marker_pass(donor, members, selected, "rna", audit, permit)
    positives = (counts > 0).sum(axis=1)
    del counts
    marker_rows = np.stack((CELL_BUDGET - positives, positives), axis=1)
    return np.repeat(marker_rows, len(MARKERS), axis=0)


def _adt_margins(
    donor: str,
    members: list[dict[str, Any]],
    selected: list[dict[str, str]],
    audit: dict[str, Any],
    permit: reducer.HeldAccessPermit,
) -> np.ndarray:
    counts = _marker_pass(donor, members, selected, "adt", audit, permit)
    states = reducer._adt_states(counts, selected, donor)
    del counts
    positives = states.sum(axis=1)
    del states
    if not np.all(positives == CELL_BUDGET // 2):
        raise AssertionError("held ADT midrank did not produce exact 256/256 margins")
    marker_columns = np.stack((CELL_BUDGET - positives, positives), axis=1)
    return np.tile(marker_columns, (len(MARKERS), 1))


def _predict_from_margins(
    methods: dict[str, Any], rows: np.ndarray, columns: np.ndarray
) -> dict[str, np.ndarray]:
    from experiments import evaluate_gse299043_mln_development as evaluator

    predicted: dict[str, np.ndarray] = {}
    for name, method in methods.items():
        if method["kind"] == "conditional_log_odds":
            fitted = evaluator.predict_conditional_from_margins(
                np.asarray(method["source_coordinate"], dtype=float), rows, columns
            )
        elif method["kind"] == "classical_residual":
            fitted = evaluator.predict_residual_from_margins(
                np.asarray(method["source_coordinate"], dtype=float),
                rows,
                columns,
                family=method["family"],
                centered=method["centered"],
            )
        elif method["kind"] == "independence":
            fitted = (
                rows[:, :, None].astype(float)
                * columns[:, None, :].astype(float)
                / CELL_BUDGET
            )
        else:
            raise ValueError(f"frozen method {name} has an unknown kind")
        fitted = np.asarray(fitted, dtype=float)
        if (
            fitted.shape != (81, 2, 2)
            or not np.isfinite(fitted).all()
            or np.any(fitted < -1e-9)
            or not np.allclose(fitted.sum(axis=-1), rows, rtol=0.0, atol=1e-9)
            or not np.allclose(fitted.sum(axis=-2), columns, rtol=0.0, atol=1e-9)
        ):
            raise FloatingPointError(f"held {name} prediction changed a margin")
        predicted[name] = np.maximum(fitted, 0.0)
    return predicted


def _materialize_predictions(
    donor: str,
    prediction_sha256: str,
    axis_sha256: str,
    rows: np.ndarray,
    columns: np.ndarray,
    predicted: dict[str, np.ndarray],
) -> dict[str, str]:
    content = {
        "schema": "gse299043-mln-held-margin-predictions/1.0",
        "donor": donor,
        "prediction_sha256": prediction_sha256,
        "selected_cell_axis_sha256": axis_sha256,
        "row_margins_sha256": _json_sha256(rows.tolist()),
        "column_margins_sha256": _json_sha256(columns.tolist()),
        "methods": {
            name: values.reshape(81, 4).tolist() for name, values in predicted.items()
        },
    }
    content_hash = _json_sha256(content)
    path = HELD_PREDICTION_DIR / f"{donor}.json"
    _write_json_exclusive(
        path, {**content, "materialized_content_sha256": content_hash}
    )
    reread = _read_json(path)
    embedded = reread.pop("materialized_content_sha256", None)
    if embedded != content_hash or _json_sha256(reread) != content_hash:
        raise PermissionError("held prediction materialization differs after write")
    return {
        "path": _relative(path),
        "sha256": _sha256(path),
        "content_sha256": content_hash,
    }


def _read_materialized(
    donor: str,
    prediction_sha256: str,
    axis_sha256: str,
    rows: np.ndarray,
    columns: np.ndarray,
    expected_methods: tuple[str, ...],
) -> dict[str, np.ndarray]:
    path = HELD_PREDICTION_DIR / f"{donor}.json"
    payload = _read_json(path)
    embedded = payload.pop("materialized_content_sha256", None)
    if (
        not isinstance(embedded, str)
        or _json_sha256(payload) != embedded
        or payload.get("donor") != donor
        or payload.get("prediction_sha256") != prediction_sha256
        or payload.get("selected_cell_axis_sha256") != axis_sha256
        or payload.get("row_margins_sha256") != _json_sha256(rows.tolist())
        or payload.get("column_margins_sha256") != _json_sha256(columns.tolist())
    ):
        raise PermissionError("held prediction materialization binding differs")
    methods = payload.get("methods")
    if not isinstance(methods, dict) or set(methods) != set(expected_methods):
        raise PermissionError("held materialized method set differs")
    output = {}
    for name, values in methods.items():
        fitted = np.asarray(values, dtype=float)
        tables = fitted.reshape(81, 2, 2) if fitted.shape == (81, 4) else None
        if (
            tables is None
            or not np.isfinite(tables).all()
            or np.any(tables < -1e-9)
            or not np.allclose(tables.sum(axis=-1), rows, rtol=0.0, atol=1e-9)
            or not np.allclose(tables.sum(axis=-2), columns, rtol=0.0, atol=1e-9)
        ):
            raise ValueError("held materialized prediction is malformed")
        output[name] = np.maximum(tables, 0.0)
    return output


def _entity_support_report(informative: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(informative, dtype=bool)
    if mask.shape != (81,):
        raise ValueError("held informative-entity mask has the wrong shape")
    excluded = [
        {
            "entity_index": int(index),
            "rna_marker": MARKERS[int(index) // len(MARKERS)],
            "adt_marker": MARKERS[int(index) % len(MARKERS)],
        }
        for index in np.flatnonzero(~mask)
    ]
    return {
        "informative_entities": int(mask.sum()),
        "informative_entity_mask": mask.tolist(),
        "excluded_entities": excluded,
    }


def _truth_and_losses(
    donor: str,
    members: list[dict[str, Any]],
    selected: list[dict[str, str]],
    axis_sha256: str,
    rows: np.ndarray,
    columns: np.ndarray,
    prediction_sha256: str,
    expected_methods: tuple[str, ...],
    audit: dict[str, Any],
    permit: reducer.HeldAccessPermit,
) -> tuple[dict[str, float], dict[str, Any]]:
    if len(list(HELD_PREDICTION_DIR.glob("*.json"))) != len(HELD_DONORS):
        raise PermissionError("all held predictions must be materialized before truth")
    paired = _marker_pass(donor, members, selected, "paired", audit, permit)
    rna = (paired[: len(MARKERS)] > 0).astype(np.uint8)
    adt = reducer._adt_states(paired[len(MARKERS) :], selected, donor)
    del paired
    truth = reducer._ordered_tables(rna, adt)
    del rna, adt
    if not np.array_equal(truth.sum(axis=-1), rows) or not np.array_equal(
        truth.sum(axis=-2), columns
    ):
        raise AssertionError("paired held truth changed a frozen margin")
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
    predicted = _read_materialized(
        donor, prediction_sha256, axis_sha256, rows, columns, expected_methods
    )
    from experiments import evaluate_gse299043_mln_development as evaluator

    donor_losses = {
        name: evaluator.donor_loss(truth, fitted, informative)
        for name, fitted in predicted.items()
    }
    truth_hash = _json_sha256(truth.reshape(81, 4).tolist())
    del truth, predicted
    return donor_losses, {
        **_entity_support_report(informative),
        "truth_sha256": truth_hash,
    }


def _exact_sign_flip_p(difference: np.ndarray) -> float:
    values = np.asarray(difference, dtype=float)
    if values.shape != (len(HELD_DONORS),) or not np.isfinite(values).all():
        raise ValueError(
            f"sign-flip test requires {len(HELD_DONORS)} finite donor differences"
        )
    observed = float(values.sum())
    codes = np.arange(1 << len(values), dtype=np.uint32)
    bits = ((codes[:, None] >> np.arange(len(values))) & 1).astype(float)
    signed_sums = (2.0 * bits - 1.0) @ values
    return float(np.count_nonzero(signed_sums <= observed) / len(codes))


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
            favorable_required=8,
        )
        difference = np.asarray(losses["primary"]) - np.asarray(losses[comparator])
        row["exact_one_sided_sign_flip_p"] = _exact_sign_flip_p(difference)
        row["sign_flip_statistic"] = "mean(primary minus comparator donor loss)"
        row["sign_flip_tail"] = "inclusive lower tail over all 2^10 sign vectors"
        row["sign_flip_zero_rule"] = "zero differences duplicate tied assignments"
        row["passes_all"] = bool(
            row["passes_all"] and row["exact_one_sided_sign_flip_p"] <= 0.025
        )
        comparisons[comparator] = row
    return {
        "comparisons": comparisons,
        "passes_all": all(row["passes_all"] for row in comparisons.values()),
    }


def _score_held_once(
    prediction: dict[str, Any], permit: _ScorePermit
) -> dict[str, Any]:
    global _LAST_HELD_AUDIT

    attempt_hash = _validated_score_attempt(permit)
    held_permit = reducer.HeldAccessPermit(
        prediction_sha256=permit.prediction_sha256,
        public_commit=permit.public_commit,
        authorization_sha256=permit.authorization_sha256,
        terminal_attempt_sha256=attempt_hash,
    )
    _LAST_HELD_AUDIT = {
        "phase": "authorized_before_hto_census",
        "held_donors_completed": [],
        "current_donor": None,
        "current_member": None,
        "current_member_deleted": True,
        "member_content_sha256": {},
        "member_passes": [],
        "prediction_materializations": [],
        "predictions_all_materialized_before_truth": False,
        "cell_count_vectors_serialized": False,
    }
    source = _validated_source()
    methods = _validate_method_set(
        prediction.get("frozen_source_model", {}).get("methods")
    )
    donors: dict[str, dict[str, Any]] = {}
    _LAST_HELD_AUDIT["phase"] = "hto_census_and_margin_passes"
    for donor in HELD_DONORS:
        _LAST_HELD_AUDIT["current_donor"] = donor
        members = _held_members(source, donor, held_permit)
        selected, census = _census_donor(donor, members, _LAST_HELD_AUDIT, held_permit)
        rows = _rna_margins(donor, members, selected, _LAST_HELD_AUDIT, held_permit)
        columns = _adt_margins(donor, members, selected, _LAST_HELD_AUDIT, held_permit)
        predicted = _predict_from_margins(methods, rows, columns)
        materialization = _materialize_predictions(
            donor,
            permit.prediction_sha256,
            census["selected_cell_axis_sha256"],
            rows,
            columns,
            predicted,
        )
        _LAST_HELD_AUDIT["prediction_materializations"].append(
            {"donor": donor, **materialization}
        )
        donors[donor] = {
            "members": members,
            "selected": selected,
            "census": census,
            "rows": rows,
            "columns": columns,
            "materialization": materialization,
        }
        del predicted
    materialized_paths = [
        HELD_PREDICTION_DIR / f"{donor}.json" for donor in HELD_DONORS
    ]
    if not all(path.is_file() for path in materialized_paths):
        raise PermissionError("held prediction materialization set is incomplete")
    _LAST_HELD_AUDIT["predictions_all_materialized_before_truth"] = True
    _LAST_HELD_AUDIT["phase"] = "paired_truth_after_all_predictions"

    method_order = tuple(methods)
    losses: dict[str, list[float]] = {name: [] for name in method_order}
    donor_results = []
    for donor in HELD_DONORS:
        record = donors[donor]
        donor_losses, support = _truth_and_losses(
            donor,
            record["members"],
            record["selected"],
            record["census"]["selected_cell_axis_sha256"],
            record["rows"],
            record["columns"],
            permit.prediction_sha256,
            method_order,
            _LAST_HELD_AUDIT,
            held_permit,
        )
        for name, value in donor_losses.items():
            losses[name].append(value)
        donor_results.append(
            {
                "donor": donor,
                "cells": CELL_BUDGET,
                **support,
                "census": record["census"],
                "prediction_materialization": record["materialization"],
                "donor_equal_deviance": donor_losses,
            }
        )
        _LAST_HELD_AUDIT["held_donors_completed"].append(donor)
        del record["selected"], record["rows"], record["columns"]
    gate = _held_gate(losses)
    status = "CONFIRMATION_PASS" if gate["passes_all"] else "CONFIRMATION_FAIL"
    return {
        "schema": "gse299043-mln-exact-confirmation/1.0",
        "status": status,
        "created_at_utc": _timestamp(),
        "prediction_sha256": permit.prediction_sha256,
        "public_prediction_commit": permit.public_commit,
        "public_authorization_commit": permit.public_authorization_commit,
        "bindings": {
            "score_attempt_sha256": attempt_hash,
            "authorization_sha256": _sha256(AUTHORIZATION),
            "authorization_publication_template_sha256": _sha256(
                AUTH_PUBLICATION_TEMPLATE
            ),
            "authorization_publication_sha256": _sha256(AUTH_PUBLICATION),
            "runner_sha256": _sha256(Path(__file__)),
            "reducer_sha256": _sha256(REDUCER),
            "development_evaluator_sha256": _sha256(EVALUATOR),
            "protocol_sha256": _sha256(PROTOCOL),
            "candidate_designation_sha256": _sha256(DESIGNATION),
            "family_policy_sha256": _sha256(FAMILY_POLICY),
            "source_template_sha256": _sha256(SOURCE_TEMPLATE),
            "source_manifest_sha256": _sha256(SOURCE_MANIFEST),
            "development_result_sha256": _sha256(DEVELOPMENT_RESULT),
            **_transitive_bindings(),
        },
        "held_donors": list(HELD_DONORS),
        "inference_unit": "physical organ donor",
        "cells_per_donor": CELL_BUDGET,
        "ordered_entities": 81,
        "classical_head_to_head": {
            key: methods["best_residual"][key]
            for key in (
                "family",
                "centered",
                "sample_size_normalized",
                "normalization",
            )
        },
        "donor_results": donor_results,
        "held_losses": {
            name: {donor: value for donor, value in zip(HELD_DONORS, values)}
            for name, values in losses.items()
        },
        "held_member_sha256": dict(_LAST_HELD_AUDIT["member_content_sha256"]),
        "gate": gate,
        "access_audit": {
            "attempt_marker_preceded_first_held_request": True,
            "held_source_members_requested": len(_LAST_HELD_AUDIT["member_passes"]),
            "distinct_held_source_members_hashed": len(
                _LAST_HELD_AUDIT["member_content_sha256"]
            ),
            "maximum_source_h5ads_retained": 1,
            "all_source_h5ads_deleted": _LAST_HELD_AUDIT["current_member_deleted"],
            "hto_only_census_preceded_margin_access": True,
            "rna_and_adt_margins_used_separate_passes": True,
            "all_predictions_hashed_before_any_paired_truth": True,
            "cell_count_vectors_serialized": False,
        },
    }


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


def score() -> dict[str, Any]:
    """Run the one terminal held-site score after immutable authorization."""

    global _LAST_HELD_AUDIT

    _LAST_HELD_AUDIT = None
    _assert_family_available()
    if any(path.exists() for path in (SCORE_ATTEMPT, OUTPUT, REFUSAL)):
        raise FileExistsError("a terminal GSE299043 scoring artifact already exists")
    if any(
        directory.exists() and any(directory.iterdir())
        for directory in (HELD_MEMBER_DIR, HELD_PREDICTION_DIR)
    ):
        raise FileExistsError("held work directory is not empty before scoring")
    prediction, permit = _validated_authorization()
    _write_json_exclusive(
        SCORE_ATTEMPT,
        _score_attempt_payload(permit, _timestamp()),
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
                "schema": "gse299043-mln-score-refusal/1.0",
                "status": "TERMINAL_SCORE_REFUSAL",
                "created_at_utc": _timestamp(),
                "error_type": type(error).__name__,
                "sanitized_error_message": _sanitized_error_message(error),
                "prediction_sha256": permit.prediction_sha256,
                "partial_audit": copy.deepcopy(_LAST_HELD_AUDIT)
                if _LAST_HELD_AUDIT is not None
                else {
                    "phase": "authorized_before_hto_census",
                    "held_donors_completed": [],
                    "current_donor": None,
                    "current_member": None,
                    "current_member_deleted": True,
                    "member_content_sha256": {},
                    "member_passes": [],
                    "prediction_materializations": [],
                    "predictions_all_materialized_before_truth": False,
                    "cell_count_vectors_serialized": False,
                },
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("predict", "score"))
    args = parser.parse_args()
    payload = predict() if args.command == "predict" else score()
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
