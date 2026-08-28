"""Outcome-disabled development evaluation for the GSE299043 MLN study.

Only the reduced ten-donor JSON is read. Every candidate is selected by
leave-one-donor-out prediction, and held-site files are neither located nor
opened by this module.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from itertools import product
import json
import multiprocessing
import os
from pathlib import Path
import re
from typing import Any

import numpy as np

from experiments import evaluate_gse279451_sepsis_development as numerical
from experiments import reduce_gse299043_mln as reducer
from mapreg.heterogeneity_adaptive_coupling import CouplingEstimationRefusal


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/development/gse299043_mln/reduced_development_v1.json"
DEVELOPMENT_ATTEMPT = (
    ROOT / "data/development/gse299043_mln/development_attempt_v1.json"
)
EVALUATION_ATTEMPT = (
    ROOT / "data/development/gse299043_mln/evaluation_attempt_v1.json"
)
OUTPUT = ROOT / "results/development/gse299043_mln_exact_development.json"
EVALUATION_REFUSAL = (
    ROOT / "results/development/gse299043_mln_evaluation_refusal.json"
)
PROTOCOL = ROOT / "docs/GSE299043_MLN_HELD_SITE_CONFIRMATION_PROTOCOL_2026-08-28.md"
DESIGNATION = ROOT / "data/confirmation/gse299043_mln/candidate_designation_v1.json"
FAMILY_POLICY = ROOT / "data/confirmation/gse299043_mln/family_policy_v1.json"
SOURCE_TEMPLATE = (
    ROOT / "data/confirmation/gse299043_mln/source_manifest_template_v1.json"
)
SOURCE_MANIFEST = ROOT / "data/confirmation/gse299043_mln/source_manifest_v1.json"
METADATA_PREFLIGHT = ROOT / "data/development/gse299043_mln/metadata_preflight_v1.tsv"
PRIOR_REFUSAL = ROOT / "results/development/gse279451_sepsis_evaluation_refusal.json"
EVALUATOR_TESTS = ROOT / "tests/test_evaluate_gse299043_mln_development.py"

TRANSITIVE_ARTIFACTS = {
    "gse279451_numerical_core_sha256": Path(numerical.__file__),
    "gse299043_reducer_sha256": Path(reducer.__file__),
    "hierarchical_estimator_sha256": ROOT
    / "mapreg/hierarchical_conditional_coupling.py",
    "heterogeneity_estimator_sha256": ROOT
    / "mapreg/heterogeneity_adaptive_coupling.py",
    "classical_residuals_sha256": ROOT / "mapreg/classical_residuals.py",
    "evaluator_tests_sha256": EVALUATOR_TESTS,
}

MARKERS = reducer.MARKERS
DEVELOPMENT_DONORS = reducer.DEVELOPMENT_DONORS
HELD_DONORS = reducer.HELD_DONORS
CELL_BUDGET = reducer.CELL_BUDGET
MINIMUM_INFORMATIVE_ENTITIES = 64
DEVELOPMENT_MEMBER_COUNT = 56
NEIGHBOR_GRID = (1, 2, 3)
HETEROGENEITY_GRID = (0.1, 1.0, 10.0)
RIDGE_GRID = (0.01, 0.1)
GRAPH_GRID = (0.1, 0.3, 1.0)
ALPHA_GRID = (0.75, 1.0, 1.25)
CV_GRID = {
    "graph_neighborhood": list(NEIGHBOR_GRID),
    "heterogeneity_penalty": list(HETEROGENEITY_GRID),
    "ridge_penalty": list(RIDGE_GRID),
    "graph_penalty": list(GRAPH_GRID),
    "transport_multiplier": list(ALPHA_GRID),
    "classical_residual": {
        "statistic": ["pearson", "deviance"],
        "exact_null_centered": [False, True],
    },
}
REQUIRED_FAMILIES = (
    "primary",
    "best_residual",
    "destroyed_link",
    "hierarchical_ridge_only",
)
OPTIONAL_FAMILIES = ("label_permuted_graph",)
GATE_COMPARATORS = (
    "best_residual",
    "destroyed_link",
    "hierarchical_ridge_only",
)
BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 20260828
MAXIMUM_WORKERS = min(8, os.cpu_count() or 1, len(DEVELOPMENT_DONORS))

# These candidate-independent numerical routines are already sealed and tested
# by the preceding confirmation protocol. The marker axis and grids are exact
# matches; this evaluator supplies a new cohort contract and decision rule.
GraphConstructionRefusal = numerical.GraphConstructionRefusal
_CandidateBook = numerical._CandidateBook
_array_sha256 = numerical._array_sha256
_conditional_expected_tables = numerical._conditional_expected_tables
_conditional_support = numerical._conditional_support
_donor_loss = numerical._donor_loss
_fit_hierarchical = numerical._fit_hierarchical
_fold_graphs = numerical._fold_graphs
_graph_payload = numerical._graph_payload
_independence_prediction = numerical._independence_prediction
_informative = numerical._informative
_knn_edge_incidence = numerical._knn_edge_incidence
_predict_conditional = numerical._predict_conditional
_predict_residual = numerical._predict_residual
_record_conditional_alphas = numerical._record_conditional_alphas
_refuse_alphas = numerical._refuse_alphas
_residual_pool = numerical._residual_pool
_target_null_mean = numerical._target_null_mean


class DevelopmentEvaluationRefusal(RuntimeError):
    """The one permitted development evaluation could not be completed."""

    def __init__(self, detail: dict[str, Any]) -> None:
        super().__init__("development evaluation could not complete as declared")
        self.detail = detail


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"{path.name} contains nonfinite JSON token {token}")
        ),
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(serialized)


def _transitive_bindings() -> dict[str, str]:
    missing = [name for name, path in TRANSITIVE_ARTIFACTS.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "transitive evaluator artifact is absent: " + ", ".join(sorted(missing))
        )
    return {name: _sha256(path) for name, path in TRANSITIVE_ARTIFACTS.items()}


def _artifact_bindings() -> dict[str, str]:
    required = {
        "protocol_sha256": PROTOCOL,
        "candidate_designation_sha256": DESIGNATION,
        "family_policy_sha256": FAMILY_POLICY,
        "source_template_sha256": SOURCE_TEMPLATE,
        "source_manifest_sha256": SOURCE_MANIFEST,
        "metadata_preflight_sha256": METADATA_PREFLIGHT,
        **TRANSITIVE_ARTIFACTS,
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "frozen evaluator artifact is absent: " + ", ".join(sorted(missing))
        )
    return {name: _sha256(path) for name, path in required.items()}


def _validate_family_policy() -> dict[str, Any]:
    policy = _read_json(FAMILY_POLICY)
    if not PRIOR_REFUSAL.is_file():
        raise PermissionError("the prior candidate lacks its terminal refusal")
    prior = _read_json(PRIOR_REFUSAL)
    expected_hash = policy.get("required_terminal_artifact_sha256")
    if (
        policy.get("schema") != "gse299043-mln-family-policy/1.0"
        or policy.get("status") != "ELIGIBLE_AFTER_PRIOR_TERMINAL_REFUSAL"
        or policy.get("required_terminal_status")
        != "TERMINAL_DEVELOPMENT_EVALUATION_REFUSAL"
        or expected_hash != _sha256(PRIOR_REFUSAL)
        or prior.get("status") != policy.get("required_terminal_status")
        or prior.get("held_matrix_members_opened") != 0
        or policy.get("prior_held_matrix_members_opened") != 0
        or policy.get("prior_rerun_permitted") is not False
        or policy.get("maximum_development_attempts") != 1
    ):
        raise PermissionError("the backup-family policy is not satisfied")
    return policy


def _validate_active_source(
    source: dict[str, Any], attempt_bindings: object
) -> set[str]:
    audit = source.get("access_audit")
    members = source.get("members")
    if (
        source.get("schema") != "gse299043-mln-source/1.0"
        or source.get("status") != "NONHELD_SOURCE_ACCESS_AUTHORIZED"
        or source.get("bindings") != attempt_bindings
        or not isinstance(audit, dict)
        or audit.get("development_h5ad_members_decoded")
        != DEVELOPMENT_MEMBER_COUNT
        or audit.get("held_h5ad_members_requested") != 0
        or audit.get("held_h5ad_members_opened") != 0
        or audit.get("held_h5ad_members_decoded") != 0
        or audit.get("maximum_concurrent_source_h5ads") != 1
        or not isinstance(members, list)
        or len(members) != 207
    ):
        raise PermissionError("active source manifest violates the nonheld seal")
    development: set[str] = set()
    held = 0
    for member in members:
        if not isinstance(member, dict):
            raise ValueError("active source member is not an object")
        filename = member.get("filename")
        if (
            not isinstance(filename, str)
            or member.get("local_path") is not None
            or member.get("retained") is not False
        ):
            raise PermissionError("active source member retention state differs")
        if member.get("role") == "development":
            if (
                filename in development
                or member.get("donor") not in DEVELOPMENT_DONORS
                or not re.fullmatch(r"[0-9a-f]{64}", str(member.get("sha256")))
            ):
                raise PermissionError("development source member binding is malformed")
            development.add(filename)
        elif member.get("role") == "held":
            if member.get("donor") not in HELD_DONORS or member.get("sha256") is not None:
                raise PermissionError("a held source member was opened or relabeled")
            held += 1
        else:
            raise PermissionError("active source member has an unknown role")
    if len(development) != DEVELOPMENT_MEMBER_COUNT or held != 151:
        raise PermissionError("active source member split differs from the freeze")
    return development


def _validated_reduced(path: Path = INPUT) -> dict[str, Any]:
    payload = _read_json(path)
    if not DEVELOPMENT_ATTEMPT.is_file():
        raise PermissionError("development acquisition attempt marker is absent")
    attempt = _read_json(DEVELOPMENT_ATTEMPT)
    source = _read_json(SOURCE_MANIFEST)
    audit = payload.get("access_audit")
    source_hash = payload.get("source_manifest_sha256")
    attempt_bindings = attempt.get("artifact_bindings")
    current_attempt_bindings = {
        "development_evaluator_sha256": _sha256(Path(__file__)),
        "reducer_sha256": _sha256(Path(reducer.__file__)),
        **_transitive_bindings(),
    }
    if (
        payload.get("schema") != "gse299043-mln-reduced-development/1.0"
        or payload.get("status") != "NONHELD_REDUCTION_COMPLETE"
        or attempt.get("status") != "TERMINAL_DEVELOPMENT_ATTEMPT_STARTED"
        or payload.get("development_attempt_sha256") != _sha256(DEVELOPMENT_ATTEMPT)
        or source_hash != _sha256(SOURCE_MANIFEST)
        or attempt.get("source_template_sha256") != _sha256(SOURCE_TEMPLATE)
        or attempt.get("metadata_preflight_sha256") != _sha256(METADATA_PREFLIGHT)
        or attempt.get("development_members_planned") != DEVELOPMENT_MEMBER_COUNT
        or attempt.get("held_h5ad_members_requested") != 0
        or attempt.get("first_network_request_starts_after_this_write") is not True
        or attempt.get("rerun_permitted") is not False
        or not isinstance(attempt_bindings, dict)
        or any(
            attempt_bindings.get(name) != digest
            for name, digest in current_attempt_bindings.items()
        )
        or payload.get("development_donors") != list(DEVELOPMENT_DONORS)
        or payload.get("held_donors") != list(HELD_DONORS)
        or payload.get("markers") != list(MARKERS)
        or payload.get("entity_count") != len(MARKERS) ** 2
        or payload.get("primary_cells_per_donor") != CELL_BUDGET
        or payload.get("cell_selection_salt") != reducer.CELL_SELECTION_SALT
        or payload.get("adt_tie_salt") != reducer.ADT_TIE_SALT
        or not isinstance(audit, dict)
        or audit.get("development_h5ad_members_decoded")
        != DEVELOPMENT_MEMBER_COUNT
        or audit.get("held_h5ad_members_opened") != 0
        or audit.get("held_h5ad_members_decoded") != 0
        or audit.get("maximum_concurrent_source_h5ads") != 1
    ):
        raise PermissionError("reduced development input violates the frozen seal")
    active_development_members = _validate_active_source(source, attempt_bindings)

    records = payload.get("donors")
    if not isinstance(records, list) or len(records) != len(DEVELOPMENT_DONORS):
        raise ValueError("reduced input must contain exactly ten development donors")
    if [record.get("donor") for record in records] != list(DEVELOPMENT_DONORS):
        raise PermissionError("reduced donor order differs from the frozen split")
    if any(record.get("donor") in HELD_DONORS for record in records):
        raise PermissionError("a held donor entered the development reduction")

    tables = np.asarray([record.get("tables") for record in records])
    destroyed = np.asarray([record.get("destroyed_tables") for record in records])
    expected_shape = (len(DEVELOPMENT_DONORS), len(MARKERS) ** 2, 4)
    for name, values in (("tables", tables), ("destroyed tables", destroyed)):
        if (
            values.shape != expected_shape
            or not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 0)
        ):
            raise ValueError(f"{name} must be nonnegative integer 2-by-2 tables")
    tables = tables.reshape(len(DEVELOPMENT_DONORS), 9, 9, 2, 2).astype(np.int64)
    destroyed = destroyed.reshape(tables.shape).astype(np.int64)
    if np.any(tables.sum(axis=(-2, -1)) != CELL_BUDGET) or np.any(
        destroyed.sum(axis=(-2, -1)) != CELL_BUDGET
    ):
        raise ValueError("every reduced table must use exactly 512 cells")
    rows = tables.sum(axis=-1)
    columns = tables.sum(axis=-2)
    if not np.array_equal(rows, np.broadcast_to(rows[:, :, :1], rows.shape)):
        raise ValueError("RNA endpoint margins differ across ADT entities")
    if not np.array_equal(
        columns, np.broadcast_to(columns[:, :1, :], columns.shape)
    ):
        raise ValueError("ADT endpoint margins differ across RNA entities")
    if np.any(columns != CELL_BUDGET // 2):
        raise ValueError("an ADT endpoint is not split into exactly 256 and 256 cells")
    if not np.array_equal(rows, destroyed.sum(axis=-1)) or not np.array_equal(
        columns, destroyed.sum(axis=-2)
    ):
        raise ValueError("destroyed-link tables changed a fixed margin")

    support = _informative(tables).reshape(len(DEVELOPMENT_DONORS), -1)
    support_counts = support.sum(axis=1)
    if np.any(support_counts < MINIMUM_INFORMATIVE_ENTITIES):
        raise ValueError("a development donor misses the fixed support floor")
    rna = np.asarray([record.get("rna_detection_prevalence") for record in records])
    adt = np.asarray(
        [record.get("adt_log_panel_fraction_mean") for record in records]
    )
    if (
        rna.shape != (len(DEVELOPMENT_DONORS), len(MARKERS))
        or adt.shape != rna.shape
        or not np.isfinite(rna).all()
        or not np.isfinite(adt).all()
        or np.any((rna < 0.0) | (rna > 1.0))
        or np.any(adt < 0.0)
    ):
        raise ValueError("fold-local marginal marker profiles are invalid")
    expected_rna = tables[:, :, 0, 1, :].sum(axis=-1) / CELL_BUDGET
    if not np.allclose(rna, expected_rna, rtol=0.0, atol=0.0):
        raise ValueError("RNA graph profiles differ from table margins")

    source_members: set[str] = set()
    for record in records:
        if (
            record.get("schema") != "gse299043-mln-reduced-donor/1.0"
            or record.get("status") != "DONOR_REDUCTION_COMPLETE"
            or record.get("role") != "development"
            or record.get("cells") != CELL_BUDGET
            or record.get("markers") != list(MARKERS)
            or record.get("entity_count") != len(MARKERS) ** 2
            or record.get("cell_selection_salt") != reducer.CELL_SELECTION_SALT
            or record.get("adt_tie_salt") != reducer.ADT_TIE_SALT
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(record.get("selected_cell_axis_sha256"))
            )
        ):
            raise PermissionError("a donor record differs from the reduction seal")
        pieces = record.get("library_pieces")
        if not isinstance(pieces, list) or not pieces:
            raise ValueError("a reduced donor has no source-library ledger")
        for piece in pieces:
            filename = piece.get("source_filename")
            if (
                not isinstance(filename, str)
                or filename in source_members
                or not re.fullmatch(r"[0-9a-f]{64}", str(piece.get("source_sha256")))
                or not re.fullmatch(r"[0-9a-f]{64}", str(piece.get("piece_sha256")))
            ):
                raise ValueError("source-library ledger is malformed or duplicated")
            source_members.add(filename)
    if len(source_members) != DEVELOPMENT_MEMBER_COUNT:
        raise ValueError("source-library ledger does not contain all 56 members")
    if source_members != active_development_members:
        raise PermissionError("reduced and active development member axes differ")

    return {
        "source_manifest_sha256": source_hash,
        "donors": list(DEVELOPMENT_DONORS),
        "tables": tables,
        "destroyed_tables": destroyed,
        "rna_profiles": rna.astype(float),
        "adt_profiles": adt.astype(float),
        "support_counts": support_counts.astype(int),
    }


def _configuration(family: str, config: tuple[Any, ...]) -> dict[str, Any]:
    return numerical._configuration(family, config)


def _candidate_books(donors: list[str]) -> dict[str, _CandidateBook]:
    hierarchical = list(
        product(
            NEIGHBOR_GRID,
            HETEROGENEITY_GRID,
            RIDGE_GRID,
            GRAPH_GRID,
            ALPHA_GRID,
        )
    )
    residual = [
        (family, centered, alpha)
        for family, centered in (
            ("pearson", False),
            ("pearson", True),
            ("deviance", False),
            ("deviance", True),
        )
        for alpha in ALPHA_GRID
    ]
    return {
        "primary": _CandidateBook("primary", hierarchical, donors),
        "destroyed_link": _CandidateBook("destroyed_link", hierarchical.copy(), donors),
        "label_permuted_graph": _CandidateBook(
            "label_permuted_graph", hierarchical.copy(), donors
        ),
        "hierarchical_ridge_only": _CandidateBook(
            "hierarchical_ridge_only",
            list(product(HETEROGENEITY_GRID, RIDGE_GRID, ALPHA_GRID)),
            donors,
        ),
        "best_residual": _CandidateBook("best_residual", residual, donors),
    }


def _cross_validate_serial(
    data: dict[str, Any], folds: tuple[int, ...] | None = None
) -> dict[str, Any]:
    donors = data["donors"]
    books = _candidate_books(donors)
    independence = np.full(len(donors), np.nan, dtype=float)
    graph_audit: list[dict[str, Any]] = []
    identity = np.eye(len(MARKERS), dtype=float)
    selected_folds = tuple(range(len(donors))) if folds is None else folds

    for fold in selected_folds:
        omitted = donors[fold]
        training = np.arange(len(donors)) != fold
        training_donors = [
            donor for index, donor in enumerate(donors) if training[index]
        ]
        target_tables = data["tables"][fold]
        recipient = _conditional_support(target_tables)
        independence[fold] = _donor_loss(
            target_tables,
            _independence_prediction(target_tables),
            recipient["informative"],
        )
        try:
            graphs, permuted, audit = _fold_graphs(
                data["rna_profiles"][training],
                data["adt_profiles"][training],
                training_donors,
            )
        except GraphConstructionRefusal as error:
            graph_audit.append(
                {
                    "omitted_donor": omitted,
                    "training_donors": training_donors,
                    "status": "REFUSED",
                    "reason": str(error),
                }
            )
            for family in ("primary", "destroyed_link", "label_permuted_graph"):
                for config in books[family].configs:
                    books[family].refuse(config, fold, str(error))
            graphs = {}
            permuted = {}
        else:
            graph_audit.append({"omitted_donor": omitted, "status": "OK", **audit})

        for neighbors in NEIGHBOR_GRID:
            if neighbors not in graphs:
                continue
            first, second = graphs[neighbors]
            permuted_first, permuted_second = permuted[neighbors]
            for heterogeneity in HETEROGENEITY_GRID:
                for ridge in RIDGE_GRID:
                    for graph in GRAPH_GRID:
                        prefix = (neighbors, heterogeneity, ridge, graph)
                        for family, source_tables, incidence in (
                            (
                                "primary",
                                data["tables"][training],
                                (first, second),
                            ),
                            (
                                "destroyed_link",
                                data["destroyed_tables"][training],
                                (first, second),
                            ),
                            (
                                "label_permuted_graph",
                                data["tables"][training],
                                (permuted_first, permuted_second),
                            ),
                        ):
                            try:
                                coordinate, certificate = _fit_hierarchical(
                                    source_tables,
                                    *incidence,
                                    heterogeneity,
                                    ridge,
                                    graph,
                                )
                            except (
                                CouplingEstimationRefusal,
                                FloatingPointError,
                                np.linalg.LinAlgError,
                            ) as error:
                                _refuse_alphas(books[family], prefix, fold, error)
                            else:
                                _record_conditional_alphas(
                                    books[family],
                                    prefix,
                                    fold,
                                    coordinate,
                                    certificate,
                                    recipient,
                                )

        for heterogeneity in HETEROGENEITY_GRID:
            for ridge in RIDGE_GRID:
                prefix = (heterogeneity, ridge)
                try:
                    coordinate, certificate = _fit_hierarchical(
                        data["tables"][training],
                        identity,
                        identity,
                        heterogeneity,
                        ridge,
                        0.0,
                    )
                except (
                    CouplingEstimationRefusal,
                    FloatingPointError,
                    np.linalg.LinAlgError,
                ) as error:
                    _refuse_alphas(
                        books["hierarchical_ridge_only"], prefix, fold, error
                    )
                else:
                    _record_conditional_alphas(
                        books["hierarchical_ridge_only"],
                        prefix,
                        fold,
                        coordinate,
                        certificate,
                        recipient,
                    )

        target_null = {
            family: _target_null_mean(target_tables, family)
            for family in ("pearson", "deviance")
        }
        for family, centered in (
            ("pearson", False),
            ("pearson", True),
            ("deviance", False),
            ("deviance", True),
        ):
            try:
                pooled, certificate = _residual_pool(
                    data["tables"][training], family, centered
                )
            except (CouplingEstimationRefusal, FloatingPointError) as error:
                _refuse_alphas(books["best_residual"], (family, centered), fold, error)
                continue
            for alpha in ALPHA_GRID:
                config = (family, centered, alpha)
                try:
                    prediction = _predict_residual(
                        pooled,
                        target_tables,
                        family=family,
                        centered=centered,
                        alpha=alpha,
                        target_null=target_null[family],
                    )
                    loss = _donor_loss(
                        target_tables, prediction, recipient["informative"]
                    )
                except (FloatingPointError, ValueError) as error:
                    books["best_residual"].refuse(
                        config, fold, f"{type(error).__name__}: {error}"
                    )
                else:
                    books["best_residual"].record(config, fold, loss, certificate)

    return {
        "books": books,
        "independence": independence,
        "fold_graph_audit": graph_audit,
    }


def _fold_worker(payload: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    fold, data = payload
    return fold, _cross_validate_serial(data, (fold,))


def _cross_validate(
    data: dict[str, Any], workers: int = MAXIMUM_WORKERS
) -> dict[str, Any]:
    count = int(workers)
    if not 1 <= count <= MAXIMUM_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAXIMUM_WORKERS}")
    if count == 1:
        return _cross_validate_serial(data)
    folds = list(range(len(data["donors"])))
    with ProcessPoolExecutor(
        max_workers=count, mp_context=multiprocessing.get_context("fork")
    ) as executor:
        results = list(executor.map(_fold_worker, ((fold, data) for fold in folds)))
    results.sort(key=lambda item: item[0])
    donors = data["donors"]
    books = _candidate_books(donors)
    independence = np.full(len(donors), np.nan, dtype=float)
    graph_audit: list[dict[str, Any]] = []
    for fold, result in results:
        donor = donors[fold]
        independence[fold] = result["independence"][fold]
        if len(result["fold_graph_audit"]) != 1:
            raise AssertionError("fold worker returned an incomplete graph audit")
        graph_audit.append(result["fold_graph_audit"][0])
        for family, target in books.items():
            source = result["books"][family]
            for config in target.configs:
                value = source.losses[config][fold]
                refusal = source.refusals[config].get(donor)
                if np.isfinite(value) and refusal is None:
                    target.record(
                        config,
                        fold,
                        float(value),
                        source.certificates[config].get(donor),
                    )
                elif refusal is not None and not np.isfinite(value):
                    target.refuse(config, fold, refusal)
                else:
                    raise AssertionError(
                        "fold worker did not return exactly one candidate outcome"
                    )
    if not np.isfinite(independence).all():
        raise AssertionError("fold workers did not return every independence loss")
    return {
        "books": books,
        "independence": independence,
        "fold_graph_audit": graph_audit,
    }


def _comparison(
    donors: list[str],
    primary: np.ndarray,
    comparator: np.ndarray,
    label: str,
    bootstrap_indices: np.ndarray | None = None,
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
        or comparator_values.mean() <= 0.0
    ):
        raise ValueError("gate requires finite paired losses and a positive comparator")
    difference = primary_values - comparator_values
    if bootstrap_indices is None:
        indices = np.random.default_rng(BOOTSTRAP_SEED).integers(
            0, len(donors), size=(BOOTSTRAPS, len(donors)), endpoint=False
        )
    else:
        indices = np.asarray(bootstrap_indices)
        if indices.shape != (BOOTSTRAPS, len(donors)) or np.any(
            (indices < 0) | (indices >= len(donors))
        ):
            raise ValueError("shared bootstrap index matrix is invalid")
    bootstrap = difference[indices].mean(axis=1)
    interval = np.quantile(bootstrap, [0.025, 0.975], method="linear")
    relative = 1.0 - float(primary_values.mean() / comparator_values.mean())
    favorable = int(np.count_nonzero(difference < 0.0))
    passes = {
        "relative_reduction_at_least_five_percent": bool(relative >= 0.05),
        "bootstrap_upper_95_below_zero": bool(interval[1] < 0.0),
        "at_least_eight_favorable_donors": bool(favorable >= 8),
    }
    return {
        "comparator": label,
        "primary_mean_loss": float(primary_values.mean()),
        "comparator_mean_loss": float(comparator_values.mean()),
        "relative_reduction": relative,
        "bootstrap_95_ci": interval.tolist(),
        "bootstrap_upper_95": float(interval[1]),
        "bootstrap_draws": BOOTSTRAPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_quantile_method": "linear",
        "bootstrap_indices_shared_across_comparisons": True,
        "bootstrap_unit": "physical donor",
        "favorable_donors": favorable,
        "required_favorable_donors": 8,
        "donor_differences_primary_minus_comparator": {
            donor: float(value) for donor, value in zip(donors, difference)
        },
        "passes": passes,
        "passes_all": all(passes.values()),
    }


def _tables_from_margins(
    row_margins: np.ndarray, column_margins: np.ndarray
) -> np.ndarray:
    rows = np.asarray(row_margins)
    columns = np.asarray(column_margins)
    expected = (len(MARKERS) ** 2, 2)
    if (
        rows.shape != expected
        or columns.shape != expected
        or not np.issubdtype(rows.dtype, np.integer)
        or not np.issubdtype(columns.dtype, np.integer)
        or np.any(rows < 0)
        or np.any(columns < 0)
        or not np.array_equal(rows.sum(axis=1), columns.sum(axis=1))
        or np.any(rows.sum(axis=1) <= 0)
    ):
        raise ValueError("recipient margins must be positive integer 81-by-2 arrays")
    tables = np.empty((len(rows), 2, 2), dtype=np.int64)
    for entity, (row, column) in enumerate(zip(rows, columns)):
        tables[entity] = numerical._canonical_table(row, column)
    return tables


def predict_conditional_from_margins(
    source_coordinate: np.ndarray,
    row_margins: np.ndarray,
    column_margins: np.ndarray,
) -> np.ndarray:
    """Reconstruct expected recipient tables without reading paired outcomes."""

    tables = _tables_from_margins(row_margins, column_margins)
    return _conditional_expected_tables(
        np.asarray(source_coordinate, dtype=float), _conditional_support(tables)
    )


def predict_residual_from_margins(
    source_coordinate: np.ndarray,
    row_margins: np.ndarray,
    column_margins: np.ndarray,
    *,
    family: str,
    centered: bool,
) -> np.ndarray:
    """Invert a frozen classical interaction coordinate at recipient margins."""

    tables = _tables_from_margins(row_margins, column_margins)
    return _predict_residual(
        np.asarray(source_coordinate, dtype=float),
        tables,
        family=family,
        centered=bool(centered),
        alpha=1.0,
        target_null=_target_null_mean(tables, family),
    )


def donor_loss(
    truth: np.ndarray,
    prediction: np.ndarray,
    informative: np.ndarray | None = None,
) -> float:
    """Return the protocol's informative-entity deviance per cell."""

    return _donor_loss(truth, prediction, informative)


def _conditional_method(
    coordinate: np.ndarray,
    alpha: float,
    config: dict[str, Any],
    certificate: dict[str, Any],
    estimator: str,
    graph: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "kind": "conditional_log_odds",
        "estimator": estimator,
        "source_coordinate": (float(alpha) * coordinate).tolist(),
        "unscaled_source_coordinate": coordinate.tolist(),
        "transport_multiplier": float(alpha),
        "selected_configuration": config,
        "numerical_certificate": certificate,
        "recipient_reconstruction": "exact conditional expected table at target margins",
    }
    if graph is not None:
        payload["graph"] = graph
    return payload


def _fit_frozen_models(
    data: dict[str, Any], selections: dict[str, tuple[Any, ...]]
) -> dict[str, Any]:
    graphs, permuted, graph_audit = _fold_graphs(
        data["rna_profiles"], data["adt_profiles"], data["donors"]
    )
    identity = np.eye(len(MARKERS), dtype=float)
    methods: dict[str, Any] = {}
    for family, source_tables, graph_source in (
        ("primary", data["tables"], graphs),
        ("destroyed_link", data["destroyed_tables"], graphs),
    ):
        neighbors, heterogeneity, ridge, graph, alpha = selections[family]
        first, second = graph_source[int(neighbors)]
        coordinate, certificate = _fit_hierarchical(
            source_tables, first, second, heterogeneity, ridge, graph
        )
        methods[family] = _conditional_method(
            coordinate,
            alpha,
            _configuration(family, selections[family]),
            certificate,
            "donor-heterogeneity-aware exact conditional log odds",
            _graph_payload(first, second),
        )

    heterogeneity, ridge, alpha = selections["hierarchical_ridge_only"]
    coordinate, certificate = _fit_hierarchical(
        data["tables"], identity, identity, heterogeneity, ridge, 0.0
    )
    methods["hierarchical_ridge_only"] = _conditional_method(
        coordinate,
        alpha,
        _configuration(
            "hierarchical_ridge_only", selections["hierarchical_ridge_only"]
        ),
        certificate,
        "donor-heterogeneity-aware ridge-only exact conditional log odds",
        None,
    )

    family, centered, alpha = selections["best_residual"]
    coordinate, certificate = _residual_pool(data["tables"], family, centered)
    methods["best_residual"] = {
        "kind": "classical_residual",
        "family": family,
        "centered": bool(centered),
        "source_coordinate": (float(alpha) * coordinate).tolist(),
        "unscaled_source_coordinate": coordinate.tolist(),
        "transport_multiplier": float(alpha),
        "selected_configuration": _configuration(
            "best_residual", selections["best_residual"]
        ),
        "sample_size_normalized": True,
        "normalization": "source/sqrt(n), recipient*sqrt(m)",
        "donor_equal_pooling": True,
        "target_null_restored": bool(centered),
        "target_margin_inversion": True,
        "numerical_certificate": certificate,
    }

    optional_refit_error = None
    optional = selections.get("label_permuted_graph")
    if optional is not None:
        try:
            neighbors, heterogeneity, ridge, graph, alpha = optional
            first, second = permuted[int(neighbors)]
            coordinate, certificate = _fit_hierarchical(
                data["tables"], first, second, heterogeneity, ridge, graph
            )
            methods["label_permuted_graph"] = _conditional_method(
                coordinate,
                alpha,
                _configuration("label_permuted_graph", optional),
                certificate,
                "label-permuted graph conditional log odds",
                _graph_payload(first, second),
            )
        except (
            CouplingEstimationRefusal,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as error:
            optional_refit_error = f"{type(error).__name__}: {error}"
    methods["independence"] = {"kind": "independence"}
    if not set(REQUIRED_FAMILIES).issubset(methods):
        raise AssertionError("frozen required method set is incomplete")
    return {
        "schema": "gse299043-mln-frozen-source-model/1.0",
        "entity_order": "row-major RNA marker by ADT marker",
        "entity_count": len(MARKERS) ** 2,
        "methods": methods,
        "optional_label_permuted_refit_error": optional_refit_error,
        "all_development_graph_audit": graph_audit,
    }


def _completed_development_status(
    gate_pass: bool,
    unavailable_required: list[str],
    final_refit_error: str | None,
    diagnostics: dict[str, Any] | None = None,
) -> str:
    if unavailable_required or final_refit_error is not None:
        detail: dict[str, Any] = {
            "unavailable_required_candidate_families": list(unavailable_required),
            "final_refit_error": final_refit_error,
        }
        if diagnostics is not None:
            detail["diagnostics"] = diagnostics
        raise DevelopmentEvaluationRefusal(detail)
    return "DEVELOPMENT_PASS" if gate_pass else "DEVELOPMENT_FAIL"


def _run_development_after_attempt(
    data: dict[str, Any], workers: int, bindings: dict[str, str]
) -> dict[str, Any]:
    evaluated = _cross_validate(data, workers=workers)
    books: dict[str, _CandidateBook] = evaluated["books"]
    selections = {family: book.selected() for family, book in books.items()}
    unavailable_required = [
        family for family in REQUIRED_FAMILIES if selections[family] is None
    ]
    unavailable_optional = [
        family for family in OPTIONAL_FAMILIES if selections[family] is None
    ]
    diagnostics = {family: book.diagnostics() for family, book in books.items()}
    selected_losses = {
        family: books[family].losses[config]
        for family, config in selections.items()
        if config is not None
    }
    selected_losses["independence"] = evaluated["independence"]

    refusal_diagnostics = {
        "selected_settings": {
            family: (
                _configuration(family, config) if config is not None else None
            )
            for family, config in selections.items()
        },
        "candidate_diagnostics": diagnostics,
        "fold_graph_audit": evaluated["fold_graph_audit"],
    }
    _completed_development_status(
        False, unavailable_required, None, refusal_diagnostics
    )

    primary = selected_losses["primary"]
    bootstrap_indices = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0,
        len(data["donors"]),
        size=(BOOTSTRAPS, len(data["donors"])),
        endpoint=False,
    )
    comparisons = {
        comparator: _comparison(
            data["donors"],
            primary,
            selected_losses[comparator],
            comparator,
            bootstrap_indices,
        )
        for comparator in GATE_COMPARATORS
    }
    gate_pass = all(row["passes_all"] for row in comparisons.values())

    frozen: dict[str, Any] | None = None
    final_refit_error: str | None = None
    if gate_pass:
        try:
            frozen = _fit_frozen_models(
                data,
                {family: config for family, config in selections.items() if config},
            )
        except (
            CouplingEstimationRefusal,
            GraphConstructionRefusal,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as error:
            final_refit_error = f"{type(error).__name__}: {error}"
    status = _completed_development_status(
        gate_pass,
        [],
        final_refit_error,
        {**refusal_diagnostics, "comparisons": comparisons},
    )
    payload = {
        "schema": "gse299043-mln-exact-development/1.0",
        "status": status,
        "evaluation_attempt_sha256": _sha256(EVALUATION_ATTEMPT),
        "development_attempt_sha256": _sha256(DEVELOPMENT_ATTEMPT),
        "reduced_development_sha256": _sha256(INPUT),
        "evaluator_sha256": _sha256(Path(__file__)),
        **bindings,
        "markers": list(MARKERS),
        "entity_count": len(MARKERS) ** 2,
        "cell_budget_per_donor": CELL_BUDGET,
        "development_donors": data["donors"],
        "selection": {
            "folds": len(DEVELOPMENT_DONORS),
            "held_one_physical_donor_per_fold": True,
            "fold_donors": data["donors"],
            "grid": CV_GRID,
            "selection_loss": (
                "multinomial deviance per cell, averaged over informative "
                "entities within donor and equally over physical donors"
            ),
            "candidate_tie_rule": "first tuple in the declared grid order",
            "final_refit_donors": data["donors"] if frozen is not None else [],
        },
        "selected_settings": refusal_diagnostics["selected_settings"],
        "development_losses": {
            family: {
                donor: float(value) for donor, value in zip(data["donors"], losses)
            }
            for family, losses in selected_losses.items()
        },
        "candidate_diagnostics": diagnostics,
        "selected_fold_numerical_certificates": {
            family: books[family].certificates[config]
            for family, config in selections.items()
            if config is not None
        },
        "fold_graph_audit": evaluated["fold_graph_audit"],
        "gate": {
            "required_families": list(REQUIRED_FAMILIES),
            "required_comparators": list(GATE_COMPARATORS),
            "required_relative_reduction": 0.05,
            "bootstrap_draws": BOOTSTRAPS,
            "minimum_favorable_donors": 8,
            "comparisons": comparisons,
            "unavailable_required_candidate_families": [],
            "unavailable_optional_candidate_families": unavailable_optional,
            "final_refit_error": None,
            "passes_all": gate_pass,
        },
        "frozen_source_model": frozen,
        "access_audit": {
            "reduced_development_donors_read": len(DEVELOPMENT_DONORS),
            "held_h5ad_members_opened": 0,
            "held_h5ad_members_decoded": 0,
            "held_margins_computed": 0,
            "held_tables_formed": 0,
            "held_outcomes_used_for_selection": False,
            "raw_h5ad_or_count_matrix_opened": False,
        },
    }
    _write_json_exclusive(OUTPUT, payload)
    return payload


def _public_error_message(error: Exception) -> str:
    return str(error).replace(str(ROOT), "<repository>")


def run_development(workers: int = MAXIMUM_WORKERS) -> dict[str, Any]:
    if any(path.exists() for path in (OUTPUT, EVALUATION_ATTEMPT, EVALUATION_REFUSAL)):
        raise FileExistsError("a GSE299043 development evaluation artifact exists")
    if not INPUT.is_file() or not DEVELOPMENT_ATTEMPT.is_file():
        raise FileNotFoundError("reduced input or acquisition attempt is absent")
    _validate_family_policy()
    bindings = _artifact_bindings()
    reduced_hash = _sha256(INPUT)
    acquisition_hash = _sha256(DEVELOPMENT_ATTEMPT)
    _write_json_exclusive(
        EVALUATION_ATTEMPT,
        {
            "schema": "gse299043-mln-evaluation-attempt/1.0",
            "status": "TERMINAL_DEVELOPMENT_EVALUATION_STARTED",
            "reduced_development_sha256": reduced_hash,
            "development_attempt_sha256": acquisition_hash,
            "evaluator_sha256": _sha256(Path(__file__)),
            **bindings,
            "required_candidate_families": list(REQUIRED_FAMILIES),
            "optional_candidate_families": list(OPTIONAL_FAMILIES),
            "numerical_evaluation_starts_after_this_write": True,
            "held_h5ad_members_opened": 0,
        },
    )
    try:
        data = _validated_reduced(INPUT)
        return _run_development_after_attempt(data, workers, bindings)
    except Exception as error:
        if not EVALUATION_REFUSAL.exists():
            refusal: dict[str, Any] = {
                "schema": "gse299043-mln-evaluation-refusal/1.0",
                "status": "TERMINAL_DEVELOPMENT_EVALUATION_REFUSAL",
                "error_type": type(error).__name__,
                "error_message": _public_error_message(error),
                "reason": "development evaluation refused after the terminal attempt",
                "evaluation_attempt_sha256": _sha256(EVALUATION_ATTEMPT),
                "development_attempt_sha256": acquisition_hash,
                "reduced_development_sha256": reduced_hash,
                "bindings": bindings,
                "held_h5ad_members_opened": 0,
                "rerun_permitted": False,
            }
            if isinstance(error, DevelopmentEvaluationRefusal):
                refusal["evaluation_detail"] = error.detail
            _write_json_exclusive(EVALUATION_REFUSAL, refusal)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=MAXIMUM_WORKERS)
    args = parser.parse_args()
    print(
        json.dumps(
            run_development(workers=args.workers),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
