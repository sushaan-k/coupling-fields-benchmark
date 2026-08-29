"""Freeze the GSE334503 Batch1--Batch2 coupling-field candidate.

This stage reads only the reduced Day0 donors from Batch1 and Batch2. Batch3
is reserved for the subsequent internal validation gate and is not an input to
this program.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
from itertools import product
import json
import os
import platform
from pathlib import Path
import sys
from typing import Any, Iterator

import numpy as np
import scipy

from experiments import confirm_gse309593_held_batches as engine
from experiments.reduce_gse334503_source import PANEL
from mapreg.heterogeneity_adaptive_coupling import (
    CouplingEstimationRefusal,
    product_hypergraph_laplacian,
)
from mapreg.penalty_complete_conditional_coupling import (
    fit_hierarchical_conditional_log_odds,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/development/gse334503_source/reduced_batches_1_2_v1.npz"
DEFAULT_MANIFEST = (
    ROOT / "data/development/gse334503_source/reduction_batches_1_2_manifest_v1.json"
)
DEFAULT_OUTPUT = ROOT / "results/development/gse334503_source_candidate_v1.json"

MARKER_COUNT = 22
CELL_BUDGET = 512
SOURCE_BATCHES = ("Batch1", "Batch2")
EXPECTED_DONORS = {
    "Batch1": (
        "Donor001",
        "Donor002",
        "Donor005",
        "Donor008",
        "Donor015",
        "Donor025",
    ),
    "Batch2": (
        "Donor004",
        "Donor006",
        "Donor012",
        "Donor017",
        "Donor027",
        "Donor030",
    ),
}
HELD_DONORS = {
    "Batch3": (
        "Donor010",
        "Donor011",
        "Donor021",
        "Donor024",
        "Donor032",
        "Donor036",
    ),
    "Batch4": (
        "Donor013",
        "Donor014",
        "Donor028",
        "Donor034",
        "Donor037",
        "Donor041",
    ),
    "Batch5": (
        "Donor018",
        "Donor022",
        "Donor035",
        "Donor040",
        "Donor050",
        "Donor051",
    ),
}

HETEROGENEITY_GRID = (0.1, 1.0, 10.0)
RIDGE_GRID = (0.01, 0.1)
TRANSPORT_GRID = (0.5, 0.75, 1.0, 1.25, 1.5)
NEIGHBOR_GRID = (2, 3)
GRAPH_GRID = (0.01, 0.03, 0.1, 0.3)
TOPOLOGY_NULL_COUNT = 63
TOPOLOGY_NULL_SALT = "GSE334503-B1-B2-TOPOLOGY-NULL-v1"
DESTROYED_LINK_SALT = "GSE334503-B1-B2-DESTROYED-LINK-v1"
MINIMUM_MASK_COORDINATES = 440
MINIMUM_DONOR_COORDINATES = 390
MINIMUM_BINARY_POSITIVES = 8
MAXIMUM_BINARY_POSITIVES = CELL_BUDGET - MINIMUM_BINARY_POSITIVES
MAXIMUM_CONDITION_NUMBER = 1e12
HELD_BOOTSTRAPS = 20_000
HELD_BOOTSTRAP_SALT = "GSE334503-HELD-DONOR-BOOTSTRAP-v1"
REQUIRED_THREAD_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

IMPLEMENTATION_FILES = (
    "experiments/develop_gse334503_source_models.py",
    "experiments/confirm_gse309593_held_batches.py",
    "experiments/reduce_gse334503_source.py",
    "mapreg/penalty_complete_conditional_coupling.py",
    "mapreg/heterogeneity_adaptive_coupling.py",
    "mapreg/common_effect_conditional.py",
    "mapreg/classical_residuals.py",
    "mapreg/coupling_fields.py",
    "mapreg/factorial_coupling.py",
    "mapreg/__init__.py",
    "mapreg/table_prediction.py",
)


@dataclass(frozen=True, order=True)
class BaseConfig:
    heterogeneity_penalty: float
    ridge_penalty: float
    transport_multiplier: float


@dataclass(frozen=True, order=True)
class StructuredConfig:
    graph_neighbors: int
    heterogeneity_penalty: float
    ridge_penalty: float
    graph_penalty: float
    transport_multiplier: float


@dataclass(frozen=True, order=True)
class ResidualConfig:
    family: str
    transport_multiplier: float


@dataclass
class SourceData:
    donors: list[str]
    batches: list[str]
    barcodes: list[list[str]]
    rna_counts: np.ndarray
    adt_counts: np.ndarray
    adt_clr_profile: np.ndarray
    profile_input_key: str
    manifest_sha256: str
    manifest: dict[str, Any]


class SourceGoRefusal(CouplingEstimationRefusal):
    def __init__(self, message: str, details: dict[str, Any]):
        super().__init__(message)
        self.details = details


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


def _axis_sha256(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _newline_axis_sha256(values: list[str]) -> str:
    return hashlib.sha256(
        "".join(f"{value}\n" for value in values).encode()
    ).hexdigest()


def _implementation_snapshot() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "thread_environment": {
            name: os.environ.get(name) for name in REQUIRED_THREAD_ENVIRONMENT
        },
        "files_sha256": {
            relative: _sha256(ROOT / relative) for relative in IMPLEMENTATION_FILES
        },
    }


def _require_runtime() -> None:
    mismatches = [
        name
        for name, expected in REQUIRED_THREAD_ENVIRONMENT.items()
        if os.environ.get(name) != expected
    ]
    if mismatches:
        raise PermissionError(
            "required single-thread environment is absent for " + ", ".join(mismatches)
        )


def _validated_manifest(path: Path, source_path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("source reduction manifest is unreadable") from error
    if not isinstance(manifest, dict):
        raise ValueError("source reduction manifest must contain one JSON object")
    exact = {
        "schema": "gse334503-source-reduction/1.0",
        "status": "COMPLETE",
        "accession": "GSE334503",
        "stage": "source_development",
        "numeric_batches_processed": list(SOURCE_BATCHES),
        "donor_count": 12,
        "cell_budget_per_donor": CELL_BUDGET,
        "output_path": _display_path(source_path),
        "output_bytes": source_path.stat().st_size,
        "output_sha256": _sha256(source_path),
        "reducer_sha256": _sha256(ROOT / "experiments/reduce_gse334503_source.py"),
    }
    for key, expected in exact.items():
        if manifest.get(key) != expected:
            raise ValueError(f"source reduction manifest has wrong {key}")
    expected_panel = [
        {"rna_gene": gene, "adt_protein": protein} for gene, protein in PANEL
    ]
    if manifest.get("panel") != expected_panel:
        raise ValueError("source reduction manifest has wrong cognate panel")
    selection = manifest.get("cell_selection")
    if not isinstance(selection, dict) or selection.get("visit") != "Day0":
        raise ValueError("source reduction manifest does not bind Day0 selection")
    profile = manifest.get("adt_graph_profile")
    if not isinstance(profile, dict):
        raise ValueError("source reduction manifest lacks ADT CLR provenance")
    denominator_axis = profile.get("denominator_adt_axis")
    positions = profile.get("cognate_positions_zero_based")
    profile_checks = {
        "cell_clr_formula": ("clr_cell = log1p(count_130) - mean_j(log1p(count_130))"),
        "donor_profile_formula": (
            "mean_cells(clr_cell), restricted to the 22 cognate positions"
        ),
        "denominator_feature_count": 130,
        "cognate_adt_axis": [protein for _, protein in PANEL],
    }
    for key, expected in profile_checks.items():
        if profile.get(key) != expected:
            raise ValueError(f"source reduction manifest has wrong ADT profile {key}")
    if (
        not isinstance(denominator_axis, list)
        or len(denominator_axis) != 130
        or len(set(denominator_axis)) != 130
        or profile.get("denominator_adt_axis_sha256")
        != _newline_axis_sha256([str(value) for value in denominator_axis])
        or profile.get("denominator_adt_set_sha256")
        != _newline_axis_sha256(sorted(str(value) for value in denominator_axis))
    ):
        raise ValueError("source reduction manifest has invalid ADT denominator axis")
    if (
        not isinstance(positions, list)
        or len(positions) != MARKER_COUNT
        or len(set(positions)) != MARKER_COUNT
        or any(
            not isinstance(value, int) or value < 0 or value >= 130
            for value in positions
        )
        or [denominator_axis[value] for value in positions]
        != [protein for _, protein in PANEL]
    ):
        raise ValueError(
            "source reduction manifest does not bind CLR profiles to cognates"
        )
    return manifest


@contextmanager
def _engine_contract() -> Iterator[None]:
    names = (
        "MARKER_COUNT",
        "CELL_BUDGET",
        "MINIMUM_INFORMATIVE_ENTITIES",
        "DESTROYED_SALT",
    )
    previous = {name: getattr(engine, name) for name in names}
    engine.MARKER_COUNT = MARKER_COUNT
    engine.CELL_BUDGET = CELL_BUDGET
    engine.MINIMUM_INFORMATIVE_ENTITIES = MINIMUM_DONOR_COORDINATES
    engine.DESTROYED_SALT = DESTROYED_LINK_SALT
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(engine, name, value)


def _load_source(path: Path, manifest_path: Path) -> SourceData:
    source_sha256 = _sha256(path)
    manifest_sha256 = _sha256(manifest_path)
    manifest = _validated_manifest(manifest_path, path)
    with np.load(path, allow_pickle=False) as data:
        expected_members = {
            "donor_axis",
            "day_axis",
            "batch_axis",
            "hto_feature_axis",
            "rna_gene_axis",
            "adt_protein_axis",
            "selected_barcodes",
            "rna_counts",
            "adt_counts",
            "adt_graph_profile",
        }
        if set(data.files) != expected_members:
            raise ValueError("source reduction has the wrong exact NPZ member set")
        donors = [str(value) for value in data["donor_axis"]]
        days = [str(value) for value in data["day_axis"]]
        batches = [str(value) for value in data["batch_axis"]]
        hto_features = [str(value) for value in data["hto_feature_axis"]]
        genes = [str(value) for value in data["rna_gene_axis"]]
        proteins = [str(value) for value in data["adt_protein_axis"]]
        barcodes = [[str(value) for value in row] for row in data["selected_barcodes"]]
        rna_counts = np.asarray(data["rna_counts"])
        adt_counts = np.asarray(data["adt_counts"])
        adt_clr_profile = np.asarray(data["adt_graph_profile"])

    expected_donors = [
        donor for batch in SOURCE_BATCHES for donor in EXPECTED_DONORS[batch]
    ]
    expected_batches = [
        batch for batch in SOURCE_BATCHES for _ in EXPECTED_DONORS[batch]
    ]
    if donors != expected_donors or batches != expected_batches:
        raise ValueError(
            "source donor or batch axis differs from the frozen B1-B2 axis"
        )
    expected_hto = [f"{donor.removeprefix('Donor')}-V1" for donor in donors]
    if days != ["Day0"] * len(donors) or hto_features != expected_hto:
        raise ValueError("source Day0 or HTO axis differs from the frozen B1-B2 axis")
    if genes != [gene for gene, _ in PANEL] or proteins != [
        protein for _, protein in PANEL
    ]:
        raise ValueError("source marker axes differ from the frozen 22 cognates")
    expected_shape = (len(donors), CELL_BUDGET, MARKER_COUNT)
    if rna_counts.shape != expected_shape or adt_counts.shape != expected_shape:
        raise ValueError("source count panels have the wrong shape")
    if adt_clr_profile.shape != (len(donors), MARKER_COUNT):
        raise ValueError("ADT CLR profiles have the wrong shape")
    int32 = np.iinfo(np.int32)
    for name, values in (("RNA", rna_counts), ("ADT", adt_counts)):
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"source {name} count panel must contain integers")
        if np.any(values < 0) or int(values.max(initial=0)) > int32.max:
            raise ValueError(f"source {name} count panel is outside int32 range")
    if adt_clr_profile.dtype != np.dtype(np.float64):
        raise ValueError("ADT CLR profiles must be float64")
    if not np.isfinite(adt_clr_profile).all():
        raise ValueError("ADT CLR profiles contain nonfinite values")
    if any(
        len(axis) != CELL_BUDGET or len(set(axis)) != CELL_BUDGET for axis in barcodes
    ):
        raise ValueError("a selected barcode axis is not unique and complete")
    linked = [
        (batch, barcode) for batch, axis in zip(batches, barcodes) for barcode in axis
    ]
    if len(set(linked)) != len(linked):
        raise ValueError("a selected cell occurs in two donors from the same batch")
    if _sha256(path) != source_sha256 or _sha256(manifest_path) != manifest_sha256:
        raise PermissionError("source or reduction manifest changed while being loaded")
    return SourceData(
        donors=donors,
        batches=batches,
        barcodes=barcodes,
        rna_counts=rna_counts.astype(np.int32, copy=False),
        adt_counts=adt_counts.astype(np.int32, copy=False),
        adt_clr_profile=adt_clr_profile.astype(np.float64, copy=False),
        profile_input_key="adt_graph_profile",
        manifest_sha256=manifest_sha256,
        manifest=manifest,
    )


def _binary_marker_support(states: np.ndarray) -> np.ndarray:
    positives = np.asarray(states, dtype=np.uint8).sum(axis=0)
    return (positives >= MINIMUM_BINARY_POSITIVES) & (
        positives <= MAXIMUM_BINARY_POSITIVES
    )


def _records(data: SourceData) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with _engine_contract():
        for index, subject in enumerate(data.donors):
            rna = (data.rna_counts[index] > 0).astype(np.uint8)
            positives = rna.sum(axis=0)
            rna_support = _binary_marker_support(rna)
            adt = (data.adt_counts[index] > 0).astype(np.uint8)
            adt_support = _binary_marker_support(adt)
            tables = engine._tables(rna, adt)
            support = (
                engine._informative(tables)
                & rna_support[:, None]
                & adt_support[None, :]
            )
            destroyed = engine._tables(
                rna,
                engine._destroyed_adt(adt, data.barcodes[index], subject),
            )
            rna_logit = np.log(
                (positives.astype(float) + 0.5)
                / (CELL_BUDGET - positives.astype(float) + 0.5)
            )
            records[subject] = {
                "batch": data.batches[index],
                "tables": tables,
                "destroyed_tables": destroyed,
                "rna_profile": rna_logit,
                "adt_profile": data.adt_clr_profile[index],
                "rna_marker_support": rna_support,
                "adt_marker_support": adt_support,
                "subject_support": support,
                "pooled_support": np.broadcast_to(
                    adt_support[None, :], (MARKER_COUNT, MARKER_COUNT)
                ).copy(),
                "table_sha256": _array_sha256(tables),
                "destroyed_table_sha256": _array_sha256(destroyed),
                "selected_cell_axis_sha256": _axis_sha256(data.barcodes[index]),
            }
    return records


def _training_mask(
    records: dict[str, dict[str, Any]], subjects: list[str]
) -> tuple[np.ndarray, dict[str, Any]]:
    tables = np.asarray([records[subject]["tables"] for subject in subjects])
    support = np.asarray([records[subject]["subject_support"] for subject in subjects])
    pooled_support = np.asarray(
        [records[subject]["pooled_support"] for subject in subjects]
    )
    with _engine_contract():
        conditional = engine._masked_tables(tables, support)
        rows = conditional.sum(axis=-1)
        columns = conditional.sum(axis=-2)
        total = conditional.sum(axis=(-2, -1))
        lower = np.maximum(0, rows[..., 0] + columns[..., 0] - total)
        upper = np.minimum(rows[..., 0], columns[..., 0])
        observed_sum = conditional[..., 0, 0].sum(axis=0)
        pooled = engine._masked_tables(tables, pooled_support).sum(axis=0)
    mask = (
        (support.sum(axis=0) >= 2)
        & (observed_sum > lower.sum(axis=0))
        & (observed_sum < upper.sum(axis=0))
        & np.all(pooled > 0, axis=(-2, -1))
    )
    count = int(np.count_nonzero(mask))
    per_subject = {
        subject: int(np.count_nonzero(mask & records[subject]["subject_support"]))
        for subject in subjects
    }
    checks = {
        "at_least_440_coordinates": count >= MINIMUM_MASK_COORDINATES,
        "every_training_donor_has_at_least_390_coordinates": all(
            value >= MINIMUM_DONOR_COORDINATES for value in per_subject.values()
        ),
    }
    details = {
        "training_subjects": subjects,
        "coordinate_count": count,
        "mask_sha256": _array_sha256(mask.astype(np.uint8)),
        "training_donor_supported_coordinate_counts": per_subject,
        "checks": checks,
    }
    if not all(checks.values()):
        raise SourceGoRefusal("training-only comparison mask failed", details)
    return mask, details


def _within_batch_normalize(
    profiles: np.ndarray, batches: list[str]
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(profiles, dtype=float)
    if values.ndim != 2 or values.shape[1] != MARKER_COUNT:
        raise ValueError("marker profiles have the wrong shape")
    if len(batches) != len(values):
        raise ValueError("profile batch axis has the wrong length")
    unique_batches = sorted(set(batches))
    centered = np.empty_like(values)
    centers: dict[str, list[float]] = {}
    for batch in unique_batches:
        indices = np.flatnonzero(np.asarray(batches) == batch)
        if len(indices) < 2:
            raise SourceGoRefusal(
                "profile normalization has fewer than two donors in a batch",
                {"batch": batch, "donor_count": len(indices)},
            )
        center = values[indices].mean(axis=0)
        centered[indices] = values[indices] - center
        centers[batch] = center.tolist()
    degrees_of_freedom = len(values) - len(unique_batches)
    scale = np.sqrt(np.square(centered).sum(axis=0) / degrees_of_freedom)
    if np.any(scale <= 0.0) or not np.isfinite(scale).all():
        raise SourceGoRefusal(
            "a source marker profile has zero or nonfinite within-batch variance",
            {"invalid_marker_indices": np.flatnonzero(~(scale > 0.0)).tolist()},
        )
    normalized = centered / scale
    return normalized, {
        "method": "acquisition-batch centering and pooled within-batch SD",
        "batch_axis": unique_batches,
        "batch_centers": centers,
        "pooled_within_batch_scale": scale.tolist(),
        "degrees_of_freedom": degrees_of_freedom,
        "normalized_profile_sha256": _array_sha256(normalized),
    }


def _marker_hyperedges(profiles: np.ndarray, neighbors: int) -> np.ndarray:
    if neighbors not in NEIGHBOR_GRID:
        raise ValueError("hypergraph neighbor count is outside the frozen grid")
    values = np.asarray(profiles, dtype=float)
    if values.ndim != 2 or values.shape[1] != MARKER_COUNT:
        raise ValueError("normalized marker profiles have the wrong shape")
    marker_profiles = values.T
    memberships: set[tuple[int, ...]] = set()
    for marker in range(MARKER_COUNT):
        candidates = np.asarray(
            [candidate for candidate in range(MARKER_COUNT) if candidate != marker]
        )
        distances = np.linalg.norm(
            marker_profiles[candidates] - marker_profiles[marker], axis=1
        )
        order = candidates[np.lexsort((candidates, distances))]
        memberships.add(tuple(sorted((marker, *map(int, order[:neighbors])))))
    ordered = sorted(memberships)
    incidence = np.zeros((MARKER_COUNT, len(ordered)), dtype=float)
    for column, members in enumerate(ordered):
        incidence[np.asarray(members), column] = 1.0
    if not np.all(incidence.sum(axis=0) == neighbors + 1):
        raise AssertionError("a marker-centered hyperedge has the wrong size")
    return incidence


def _training_design(
    records: dict[str, dict[str, Any]], subjects: list[str], neighbors: int
) -> dict[str, Any]:
    mask, mask_record = _training_mask(records, subjects)
    batches = [str(records[subject]["batch"]) for subject in subjects]
    rna_raw = np.asarray([records[subject]["rna_profile"] for subject in subjects])
    adt_raw = np.asarray([records[subject]["adt_profile"] for subject in subjects])
    rna_normalized, rna_normalization = _within_batch_normalize(rna_raw, batches)
    adt_normalized, adt_normalization = _within_batch_normalize(adt_raw, batches)
    rna_incidence = _marker_hyperedges(rna_normalized, neighbors)
    adt_incidence = _marker_hyperedges(adt_normalized, neighbors)
    laplacian = product_hypergraph_laplacian(rna_incidence, adt_incidence)
    return {
        "subjects": subjects,
        "mask": mask,
        "mask_record": mask_record,
        "rna_raw": rna_raw,
        "adt_raw": adt_raw,
        "rna_normalized": rna_normalized,
        "adt_normalized": adt_normalized,
        "rna_normalization": rna_normalization,
        "adt_normalization": adt_normalization,
        "rna_incidence": rna_incidence,
        "adt_incidence": adt_incidence,
        "product_laplacian_sha256": _array_sha256(laplacian),
    }


def _fit_structured(
    tables: np.ndarray,
    support: np.ndarray,
    first_incidence: np.ndarray,
    second_incidence: np.ndarray,
    config: StructuredConfig,
) -> dict[str, Any]:
    with _engine_contract():
        fit = fit_hierarchical_conditional_log_odds(
            engine._conditional_missing_tables(tables, support),
            first_incidence,
            second_incidence,
            heterogeneity_penalty=config.heterogeneity_penalty,
            ridge_penalty=config.ridge_penalty,
            graph_penalty=config.graph_penalty,
            minimum_informative_donors=0,
            maximum_condition_number=MAXIMUM_CONDITION_NUMBER,
        )
    certificate = {
        name: getattr(fit, name)
        for name in (
            "objective",
            "gradient_norm",
            "scaled_gradient_norm",
            "schur_condition_number",
            "theta_curvature_condition_number",
            "minimum_theta_curvature",
            "maximum_theta_curvature",
            "minimum_schur_eigenvalue",
            "maximum_schur_eigenvalue",
            "iterations",
            "converged",
            "optimizer",
            "heterogeneity_penalty_scale",
            "population_penalty_scale",
            "effective_heterogeneity_penalty",
            "effective_ridge_penalty",
            "effective_graph_penalty",
        )
    }
    certificate.update(
        {
            "minimum_support_count": int(fit.support_count.min()),
            "maximum_support_count": int(fit.support_count.max()),
            "support_count_sha256": _array_sha256(fit.support_count),
            "rna_incidence_sha256": _array_sha256(first_incidence),
            "adt_incidence_sha256": _array_sha256(second_incidence),
        }
    )
    return {
        "population_log_odds": fit.population_log_odds,
        "fit_certificate": certificate,
    }


def _fold_arrays(
    records: dict[str, dict[str, Any]], subjects: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([records[subject]["tables"] for subject in subjects]),
        np.asarray([records[subject]["destroyed_tables"] for subject in subjects]),
        np.asarray([records[subject]["subject_support"] for subject in subjects]),
        np.asarray([records[subject]["pooled_support"] for subject in subjects]),
    )


def _population_loss(
    record: dict[str, Any], mask: np.ndarray, population: np.ndarray, alpha: float
) -> float:
    evaluation = mask & record["subject_support"]
    with _engine_contract():
        rows, columns = engine._margins(record["tables"])
        prediction = engine._predict_odds(population, rows, columns, alpha)
        return engine._loss(record["tables"], prediction, evaluation)


def _residual_loss(
    record: dict[str, Any],
    mask: np.ndarray,
    pooled: np.ndarray,
    config: ResidualConfig,
) -> float:
    evaluation = mask & record["subject_support"]
    with _engine_contract():
        rows, columns = engine._margins(record["tables"])
        prediction = engine._predict_residual(
            pooled,
            rows,
            columns,
            engine.ResidualConfig(config.family, config.transport_multiplier),
        )
        return engine._loss(record["tables"], prediction, evaluation)


def _independence_loss(record: dict[str, Any], mask: np.ndarray) -> float:
    evaluation = mask & record["subject_support"]
    with _engine_contract():
        rows, columns = engine._margins(record["tables"])
        return engine._loss(
            record["tables"], engine._independence(rows, columns), evaluation
        )


def _equal_batch_summary(values: np.ndarray, batches: list[str]) -> dict[str, Any]:
    losses = np.asarray(values, dtype=float)
    complete = bool(np.isfinite(losses).all())
    means = {
        batch: (
            float(np.mean(losses[np.asarray(batches) == batch])) if complete else None
        )
        for batch in SOURCE_BATCHES
    }
    return {
        "complete": complete,
        "equal_batch_mean_loss": (
            float(np.mean(list(means.values()))) if complete else None
        ),
        "fold_mean_losses": means,
    }


def _curve_entry(
    config: Any, values: np.ndarray, donors: list[str], batches: list[str]
) -> dict[str, Any]:
    summary = _equal_batch_summary(values, batches)
    return {
        "configuration": asdict(config)
        if hasattr(config, "__dataclass_fields__")
        else config,
        "sample_axis": donors,
        "fold_losses": [
            float(value) if np.isfinite(value) else None for value in values
        ],
        **summary,
    }


def _complete_configs(losses: dict[Any, np.ndarray], batches: list[str]) -> list[Any]:
    return [
        config
        for config, values in losses.items()
        if _equal_batch_summary(values, batches)["complete"]
    ]


def _select_base(
    losses: dict[BaseConfig, np.ndarray], batches: list[str]
) -> BaseConfig:
    complete = _complete_configs(losses, batches)
    if not complete:
        raise SourceGoRefusal("no graph-zero configuration completed both folds", {})
    return min(
        complete,
        key=lambda config: (
            _equal_batch_summary(losses[config], batches)["equal_batch_mean_loss"],
            config.heterogeneity_penalty,
            config.ridge_penalty,
            config.transport_multiplier,
        ),
    )


def _select_structured(
    losses: dict[StructuredConfig, np.ndarray], batches: list[str]
) -> StructuredConfig:
    complete = _complete_configs(losses, batches)
    if not complete:
        raise SourceGoRefusal(
            "no nonzero hypergraph configuration completed both folds", {}
        )
    return min(
        complete,
        key=lambda config: (
            _equal_batch_summary(losses[config], batches)["equal_batch_mean_loss"],
            config.graph_penalty,
            config.graph_neighbors,
        ),
    )


def _select_transport(losses: dict[float, np.ndarray], batches: list[str]) -> float:
    complete = _complete_configs(losses, batches)
    if not complete:
        raise SourceGoRefusal("no transport configuration completed both folds", {})
    return min(
        complete,
        key=lambda alpha: (
            _equal_batch_summary(losses[alpha], batches)["equal_batch_mean_loss"],
            alpha,
        ),
    )


def _select_residual_family(
    losses: dict[ResidualConfig, np.ndarray],
    selected_transport: dict[str, float],
    batches: list[str],
) -> ResidualConfig:
    candidates = [
        ResidualConfig(family, selected_transport[family])
        for family in engine.RESIDUAL_FAMILIES
    ]
    if any(config not in losses for config in candidates):
        raise ValueError("a source-selected residual configuration is missing")
    return min(
        candidates,
        key=lambda config: (
            _equal_batch_summary(losses[config], batches)["equal_batch_mean_loss"],
            config.family,
        ),
    )


def _source_gate(
    primary_config: StructuredConfig,
    primary: np.ndarray,
    matched_zero: np.ndarray,
    batches: list[str],
) -> dict[str, Any]:
    primary_summary = _equal_batch_summary(primary, batches)
    zero_summary = _equal_batch_summary(matched_zero, batches)
    if not primary_summary["complete"] or not zero_summary["complete"]:
        raise ValueError("source gate received incomplete loss vectors")
    relative_reduction = 1.0 - float(
        primary_summary["equal_batch_mean_loss"] / zero_summary["equal_batch_mean_loss"]
    )
    fold_improvements = {
        batch: bool(
            primary_summary["fold_mean_losses"][batch]
            < zero_summary["fold_mean_losses"][batch]
        )
        for batch in SOURCE_BATCHES
    }
    favorable = int(np.count_nonzero(np.asarray(primary) < np.asarray(matched_zero)))
    checks = {
        "graph_penalty_is_nonzero": primary_config.graph_penalty > 0.0,
        "relative_reduction_at_least_0_05": relative_reduction >= 0.05,
        "both_directional_fold_means_improve": all(fold_improvements.values()),
        "at_least_10_of_12_donors_improve": favorable >= 10,
    }
    return {
        "passes": all(checks.values()),
        "checks": checks,
        "relative_loss_reduction_vs_matched_graph_zero": relative_reduction,
        "favorable_donors": favorable,
        "donor_count": len(primary),
        "fold_improvements": fold_improvements,
        "primary_equal_batch_mean_loss": primary_summary["equal_batch_mean_loss"],
        "matched_zero_equal_batch_mean_loss": zero_summary["equal_batch_mean_loss"],
    }


def _matched_zero_config(config: StructuredConfig) -> StructuredConfig:
    return StructuredConfig(
        graph_neighbors=config.graph_neighbors,
        heterogeneity_penalty=config.heterogeneity_penalty,
        ridge_penalty=config.ridge_penalty,
        graph_penalty=0.0,
        transport_multiplier=config.transport_multiplier,
    )


def _permutation(control: int, axis: str) -> np.ndarray:
    if control < 0 or control >= TOPOLOGY_NULL_COUNT or axis not in {"rna", "adt"}:
        raise ValueError("topology-null index or axis is invalid")
    encoded = hashlib.sha256(f"{TOPOLOGY_NULL_SALT}|{control}|{axis}".encode()).digest()
    seed = int.from_bytes(encoded[:8], "little")
    permutation = np.random.default_rng(seed).permutation(MARKER_COUNT)
    if np.array_equal(permutation, np.arange(MARKER_COUNT)):
        permutation = np.roll(permutation, 1)
    return permutation


def _select_topology_nulls(
    data: SourceData,
    records: dict[str, dict[str, Any]],
    base: BaseConfig,
) -> list[dict[str, Any]]:
    donor_index = {donor: index for index, donor in enumerate(data.donors)}
    selections = []
    for control in range(TOPOLOGY_NULL_COUNT):
        losses = {
            StructuredConfig(
                neighbors,
                base.heterogeneity_penalty,
                base.ridge_penalty,
                graph,
                base.transport_multiplier,
            ): np.full(len(data.donors), np.nan)
            for neighbors, graph in product(NEIGHBOR_GRID, GRAPH_GRID)
        }
        refusals = []
        rna_permutation = _permutation(control, "rna")
        adt_permutation = _permutation(control, "adt")
        for held_batch in SOURCE_BATCHES:
            training = [
                donor
                for donor, batch in zip(data.donors, data.batches)
                if batch != held_batch
            ]
            validation = [
                donor
                for donor, batch in zip(data.donors, data.batches)
                if batch == held_batch
            ]
            tables, _, support, _ = _fold_arrays(records, training)
            for neighbors in NEIGHBOR_GRID:
                design = _training_design(records, training, neighbors)
                first = design["rna_incidence"][rna_permutation]
                second = design["adt_incidence"][adt_permutation]
                for graph in GRAPH_GRID:
                    config = StructuredConfig(
                        neighbors,
                        base.heterogeneity_penalty,
                        base.ridge_penalty,
                        graph,
                        base.transport_multiplier,
                    )
                    try:
                        fit = _fit_structured(tables, support, first, second, config)
                        for donor in validation:
                            losses[config][donor_index[donor]] = _population_loss(
                                records[donor],
                                design["mask"],
                                fit["population_log_odds"],
                                config.transport_multiplier,
                            )
                    except (
                        ValueError,
                        FloatingPointError,
                        CouplingEstimationRefusal,
                    ) as error:
                        refusals.append(
                            {
                                "held_batch": held_batch,
                                "configuration": asdict(config),
                                "reason_code": type(error).__name__,
                                "reason": str(error),
                            }
                        )
        selected = _select_structured(losses, data.batches)
        selections.append(
            {
                "control_index": control,
                "rna_row_permutation": rna_permutation.tolist(),
                "adt_row_permutation": adt_permutation.tolist(),
                "selected_configuration": asdict(selected),
                "selected_fold_losses": losses[selected].tolist(),
                "selection_rule": (
                    "complete Stage-B k x graph-penalty source CV; minimum "
                    "equal-direction mean loss, then graph penalty, then k"
                ),
                "loss_curve": [
                    _curve_entry(config, values, data.donors, data.batches)
                    for config, values in sorted(losses.items())
                ],
                "refusals": refusals,
            }
        )
        print(
            json.dumps(
                {
                    "completed_topology_null_selection": control + 1,
                    "topology_null_count": TOPOLOGY_NULL_COUNT,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return selections


def _serialize_model(
    family: str, configuration: dict[str, Any], fit: dict[str, Any]
) -> dict[str, Any]:
    population = np.asarray(fit["population_log_odds"], dtype=float)
    return {
        "family": family,
        "configuration": configuration,
        "population_log_odds": population.tolist(),
        "population_log_odds_sha256": _array_sha256(population),
        "fit_certificate": fit["fit_certificate"],
    }


def _held_gate_contract() -> dict[str, Any]:
    comparators = [
        "matched_graph_zero",
        "common_effect_cmle",
        "pooled_saturated_poisson",
        "primary_classical_residual",
        "independence",
        "destroyed_link",
    ]
    return {
        "status": "FROZEN_IN_SOURCE_CANDIDATE_BEFORE_BATCH3_NUMERIC_ACCESS",
        "ordered_donor_axes": {
            batch: list(donors) for batch, donors in HELD_DONORS.items()
        },
        "stage_order": ["Batch3", "Batch4", "Batch5"],
        "stage_roles": {
            "Batch3": "internal validation gate",
            "Batch4": "primary held-donor confirmation",
            "Batch5": "independent held-donor replication",
        },
        "sequential_access": {
            "Batch3": "requires a public protocol that binds this source candidate",
            "Batch4": "requires Batch3 to pass every frozen gate",
            "Batch5": "requires Batch4 to pass every frozen gate",
        },
        "no_retuning": {
            "after_source_candidate": True,
            "frozen_components": [
                "comparison mask",
                "hypergraph memberships",
                "model family",
                "all penalties",
                "transport multipliers",
                "classical coordinate family",
                "topology-null configurations",
                "decision thresholds",
            ],
            "held_batches_must_not_be_pooled_for_selection_or_rescue": True,
        },
        "evaluation": {
            "mask": (
                "frozen all-source comparison mask intersected separately with "
                "each held donor's informative RNA-ADT coordinate support"
            ),
            "minimum_supported_coordinates_per_donor": MINIMUM_DONOR_COORDINATES,
            "required_valid_donors_per_batch": 6,
            "donor_loss": (
                "mean multinomial deviance per cell across the donor's supported "
                "frozen coordinates"
            ),
            "batch_statistic": "unweighted mean of the six donor losses",
            "support_failure_action": "terminal refusal; do not open the next stage",
        },
        "paired_bootstrap": {
            "unit": "donor",
            "draws": HELD_BOOTSTRAPS,
            "sampling": "six donors with replacement within the evaluated batch",
            "difference": "primary donor loss minus comparator donor loss",
            "interval": "2.5th and 97.5th percentiles",
            "quantile_method": "numpy linear",
            "seed_formula": (
                "little-endian integer from the first eight bytes of "
                "SHA256(salt|batch|comparator)"
            ),
            "salt": HELD_BOOTSTRAP_SALT,
        },
        "mandatory_comparators": comparators,
        "comparator_interpretation": {
            "common_effect_cmle": "donor-stratified exact conditional common interaction",
            "pooled_saturated_poisson": (
                "unstratified pooled 2x2 Poisson log-linear interaction"
            ),
            "primary_classical_residual": (
                "source-CV-selected signed one-degree-of-freedom Pearson or "
                "root-deviance interaction coordinate; not a cellwise residual vector"
            ),
            "independence": "recipient margins with zero transferred interaction",
        },
        "pass_rules_applied_separately_to_each_batch": {
            "nonzero_graph_penalty": True,
            "relative_mean_loss_reduction_vs_matched_graph_zero_at_least": 0.05,
            "primary_better_than_matched_graph_zero_for_all_six_donors": True,
            "matched_graph_zero_one_sided_sign_test_p_at_most": 1.0 / 64.0,
            "paired_bootstrap_upper_95_percent_bound_below_zero_for_each_comparator": comparators,
            "topology_empirical_p_at_most": 0.05,
            "relative_mean_loss_reduction_vs_median_topology_null_at_least": 0.03,
        },
        "topology_control": {
            "null_count": TOPOLOGY_NULL_COUNT,
            "statistic": "unweighted mean donor loss within the evaluated batch",
            "empirical_p_formula": (
                "(1 + count(null mean loss <= primary mean loss)) / 64"
            ),
            "inference_label": (
                "selection-aware deterministic empirical topology control, not an "
                "exact randomization test"
            ),
        },
        "failure_policy": (
            "publish the failed stage and stop; do not pool stages, retune, replace "
            "donors, change thresholds, or rerun the terminal stage"
        ),
    }


def _fit_candidate(
    data: SourceData,
    records: dict[str, dict[str, Any]],
    primary_config: StructuredConfig,
    comparator_selection: dict[str, Any],
    destroyed_alpha: float,
    topology_null_selections: list[dict[str, Any]],
) -> dict[str, Any]:
    design = _training_design(records, data.donors, primary_config.graph_neighbors)
    tables, destroyed, support, pooled_support = _fold_arrays(records, data.donors)
    primary = _fit_structured(
        tables,
        support,
        design["rna_incidence"],
        design["adt_incidence"],
        primary_config,
    )
    zero_config = _matched_zero_config(primary_config)
    identity = np.eye(MARKER_COUNT, dtype=float)
    graph_zero = _fit_structured(tables, support, identity, identity, zero_config)
    with _engine_contract():
        common = engine._fit_common_effect(tables, design["mask"], support)
        poisson = engine._fit_pooled_poisson(tables, design["mask"], pooled_support)
        residuals = {
            family: engine._residual_pool(tables, family, design["mask"], support)
            for family in engine.RESIDUAL_FAMILIES
        }
    destroyed_config = StructuredConfig(
        graph_neighbors=primary_config.graph_neighbors,
        heterogeneity_penalty=primary_config.heterogeneity_penalty,
        ridge_penalty=primary_config.ridge_penalty,
        graph_penalty=primary_config.graph_penalty,
        transport_multiplier=destroyed_alpha,
    )
    destroyed_fit = _fit_structured(
        destroyed,
        support,
        design["rna_incidence"],
        design["adt_incidence"],
        destroyed_config,
    )

    topology_nulls = []
    if [selection["control_index"] for selection in topology_null_selections] != list(
        range(TOPOLOGY_NULL_COUNT)
    ):
        raise ValueError("topology-null selection axis is incomplete or reordered")
    for selection in topology_null_selections:
        control = int(selection["control_index"])
        null_config = StructuredConfig(**selection["selected_configuration"])
        null_design = _training_design(
            records, data.donors, null_config.graph_neighbors
        )
        rna_permutation = np.asarray(selection["rna_row_permutation"], dtype=int)
        adt_permutation = np.asarray(selection["adt_row_permutation"], dtype=int)
        if not np.array_equal(rna_permutation, _permutation(control, "rna")) or not (
            np.array_equal(adt_permutation, _permutation(control, "adt"))
        ):
            raise ValueError("topology-null permutation differs from source selection")
        first = null_design["rna_incidence"][rna_permutation]
        second = null_design["adt_incidence"][adt_permutation]
        null_laplacian = product_hypergraph_laplacian(first, second)
        baseline_laplacian = product_hypergraph_laplacian(
            null_design["rna_incidence"], null_design["adt_incidence"]
        )
        joint_permutation = np.asarray(
            [
                first_marker * MARKER_COUNT + second_marker
                for first_marker in rna_permutation
                for second_marker in adt_permutation
            ]
        )
        expected_laplacian = baseline_laplacian[
            np.ix_(joint_permutation, joint_permutation)
        ]
        if not np.allclose(null_laplacian, expected_laplacian, rtol=0.0, atol=1e-12):
            raise AssertionError("topology-null permutation changed the spectrum")
        fit = _fit_structured(tables, support, first, second, null_config)
        topology_nulls.append(
            {
                **selection,
                "rna_incidence_sha256": _array_sha256(first),
                "adt_incidence_sha256": _array_sha256(second),
                "product_laplacian_sha256": _array_sha256(null_laplacian),
                "same_k_unpermuted_product_laplacian_sha256": _array_sha256(
                    baseline_laplacian
                ),
                **_serialize_model(
                    "spectrum_preserving_membership_permutation",
                    asdict(null_config),
                    fit,
                ),
            }
        )
        print(
            json.dumps(
                {
                    "completed_topology_null": control + 1,
                    "topology_null_count": TOPOLOGY_NULL_COUNT,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    models = {
        "primary": _serialize_model(
            "penalty_complete_exact_conditional_product_hypergraph",
            asdict(primary_config),
            primary,
        ),
        "matched_graph_zero": _serialize_model(
            "penalty_complete_exact_conditional_graph_zero",
            asdict(zero_config),
            graph_zero,
        ),
        "common_effect_cmle": _serialize_model(
            "unpenalized_common_effect_exact_conditional",
            {"transport_multiplier": comparator_selection["common_effect_cmle"]},
            common,
        ),
        "pooled_saturated_poisson": {
            **_serialize_model(
                "unstratified_pooled_saturated_poisson_log_linear_interaction",
                {
                    "transport_multiplier": comparator_selection[
                        "pooled_saturated_poisson"
                    ]
                },
                poisson,
            ),
            "pooled_tables_sha256": poisson["pooled_tables_sha256"],
        },
        "independence": {"family": "poisson_row_plus_column_independence"},
        "destroyed_link": _serialize_model(
            "penalty_complete_exact_conditional_after_within_donor_link_destruction",
            asdict(destroyed_config),
            destroyed_fit,
        ),
    }
    for family, coordinate in residuals.items():
        config = ResidualConfig(
            family,
            comparator_selection[f"{family}_residual"],
        )
        models[f"{family}_residual"] = {
            "family": f"poisson_independence_signed_{family}_coordinate",
            "configuration": asdict(config),
            "pooled_coordinate": coordinate.tolist(),
            "pooled_coordinate_sha256": _array_sha256(coordinate),
        }
    primary_residual = comparator_selection["primary_classical_residual"]
    selected_family = str(primary_residual["family"])
    models["primary_classical_residual"] = {
        **models[f"{selected_family}_residual"],
        "selection_role": (
            "single source-CV-selected classical interaction coordinate for held gates"
        ),
        "model_reference": f"{selected_family}_residual",
    }

    return {
        "canonical_candidate_configuration": {
            **asdict(primary_config),
            "estimator": (
                "penalty-complete exact-conditional product-hypergraph coupling field"
            ),
            "hyperedge_rule": "each marker plus its k nearest markers; duplicate memberships removed",
            "profile_metric": "Euclidean distance after acquisition-batch normalization",
        },
        "comparison_mask": {
            **design["mask_record"],
            "mask": design["mask"].astype(np.uint8).tolist(),
        },
        "source_geometry": {
            "sample_axis": data.donors,
            "sample_batches": data.batches,
            "rna_jeffreys_logit_profiles": design["rna_raw"].tolist(),
            "adt_clr_profiles": design["adt_raw"].tolist(),
            "rna_normalization": design["rna_normalization"],
            "adt_normalization": design["adt_normalization"],
            "rna_normalized_profiles": design["rna_normalized"].tolist(),
            "adt_normalized_profiles": design["adt_normalized"].tolist(),
            "rna_incidence": design["rna_incidence"].tolist(),
            "adt_incidence": design["adt_incidence"].tolist(),
            "rna_incidence_sha256": _array_sha256(design["rna_incidence"]),
            "adt_incidence_sha256": _array_sha256(design["adt_incidence"]),
            "product_laplacian_sha256": design["product_laplacian_sha256"],
        },
        "models": models,
        "topology_nulls": {
            "count": TOPOLOGY_NULL_COUNT,
            "construction": (
                "independent deterministic row permutations of RNA and ADT "
                "incidence matrices, each rerunning the complete Stage-B k x "
                "graph-penalty source selection before final refit"
            ),
            "spectrum_preserved_against_same_k_unpermuted_candidate": True,
            "selection_aware": True,
            "held_empirical_p_formula": "(1 + count(null_mean_loss <= primary_mean_loss)) / 64",
            "controls": topology_nulls,
        },
        "held_gate_contract": _held_gate_contract(),
    }


def _develop(data: SourceData) -> dict[str, Any]:
    records = _records(data)
    donors = data.donors
    batches = data.batches
    donor_index = {donor: index for index, donor in enumerate(donors)}
    fold_designs: dict[str, dict[str, Any]] = {}
    refusals: list[dict[str, Any]] = []
    base_losses = {
        BaseConfig(eta, ridge, alpha): np.full(len(donors), np.nan)
        for eta, ridge, alpha in product(HETEROGENEITY_GRID, RIDGE_GRID, TRANSPORT_GRID)
    }
    common_losses = {alpha: np.full(len(donors), np.nan) for alpha in TRANSPORT_GRID}
    poisson_losses = {alpha: np.full(len(donors), np.nan) for alpha in TRANSPORT_GRID}
    residual_losses = {
        ResidualConfig(family, alpha): np.full(len(donors), np.nan)
        for family, alpha in product(engine.RESIDUAL_FAMILIES, TRANSPORT_GRID)
    }
    independence_losses = np.full(len(donors), np.nan)

    for held_batch in SOURCE_BATCHES:
        training = [
            donor for donor, batch in zip(donors, batches) if batch != held_batch
        ]
        validation = [
            donor for donor, batch in zip(donors, batches) if batch == held_batch
        ]
        design = _training_design(records, training, NEIGHBOR_GRID[0])
        validation_support = {
            donor: int(
                np.count_nonzero(design["mask"] & records[donor]["subject_support"])
            )
            for donor in validation
        }
        if any(
            count < MINIMUM_DONOR_COORDINATES for count in validation_support.values()
        ):
            raise SourceGoRefusal(
                "a directional validation donor lacks comparison support",
                {
                    "held_batch": held_batch,
                    "validation_supported_coordinate_counts": validation_support,
                    "minimum": MINIMUM_DONOR_COORDINATES,
                },
            )
        fold_designs[held_batch] = {
            "training_subjects": training,
            "validation_subjects": validation,
            "comparison_mask": {
                **design["mask_record"],
                "mask": design["mask"].astype(np.uint8).tolist(),
            },
            "rna_normalization": design["rna_normalization"],
            "adt_normalization": design["adt_normalization"],
            "validation_supported_coordinate_counts": validation_support,
        }
        tables, _, support, pooled_support = _fold_arrays(records, training)
        identity = np.eye(MARKER_COUNT, dtype=float)
        for eta, ridge in product(HETEROGENEITY_GRID, RIDGE_GRID):
            structural = StructuredConfig(2, eta, ridge, 0.0, 1.0)
            try:
                fit = _fit_structured(tables, support, identity, identity, structural)
            except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
                refusals.append(
                    {
                        "stage": "matched_graph_zero",
                        "held_batch": held_batch,
                        "configuration": asdict(structural),
                        "reason_code": type(error).__name__,
                        "reason": str(error),
                    }
                )
                continue
            for alpha in TRANSPORT_GRID:
                config = BaseConfig(eta, ridge, alpha)
                for donor in validation:
                    base_losses[config][donor_index[donor]] = _population_loss(
                        records[donor],
                        design["mask"],
                        fit["population_log_odds"],
                        alpha,
                    )
        with _engine_contract():
            common = engine._fit_common_effect(tables, design["mask"], support)
            poisson = engine._fit_pooled_poisson(tables, design["mask"], pooled_support)
            residuals = {
                family: engine._residual_pool(tables, family, design["mask"], support)
                for family in engine.RESIDUAL_FAMILIES
            }
        for donor in validation:
            index = donor_index[donor]
            independence_losses[index] = _independence_loss(
                records[donor], design["mask"]
            )
            for alpha in TRANSPORT_GRID:
                common_losses[alpha][index] = _population_loss(
                    records[donor],
                    design["mask"],
                    common["population_log_odds"],
                    alpha,
                )
                poisson_losses[alpha][index] = _population_loss(
                    records[donor],
                    design["mask"],
                    poisson["population_log_odds"],
                    alpha,
                )
                for family in engine.RESIDUAL_FAMILIES:
                    config = ResidualConfig(family, alpha)
                    residual_losses[config][index] = _residual_loss(
                        records[donor],
                        design["mask"],
                        residuals[family],
                        config,
                    )
        print(
            json.dumps(
                {
                    "completed_stage_a_fold": held_batch,
                    "training_donors": len(training),
                    "validation_donors": len(validation),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    selected_base = _select_base(base_losses, batches)
    structured_losses = {
        StructuredConfig(
            neighbors,
            selected_base.heterogeneity_penalty,
            selected_base.ridge_penalty,
            graph,
            selected_base.transport_multiplier,
        ): np.full(len(donors), np.nan)
        for neighbors, graph in product(NEIGHBOR_GRID, GRAPH_GRID)
    }
    for held_batch in SOURCE_BATCHES:
        training = [
            donor for donor, batch in zip(donors, batches) if batch != held_batch
        ]
        validation = [
            donor for donor, batch in zip(donors, batches) if batch == held_batch
        ]
        tables, _, support, _ = _fold_arrays(records, training)
        mask = np.asarray(
            fold_designs[held_batch]["comparison_mask"]["mask"], dtype=bool
        )
        for neighbors in NEIGHBOR_GRID:
            design = _training_design(records, training, neighbors)
            if not np.array_equal(mask, design["mask"]):
                raise AssertionError(
                    "fold mask changed with hypergraph neighborhood size"
                )
            for graph in GRAPH_GRID:
                config = StructuredConfig(
                    neighbors,
                    selected_base.heterogeneity_penalty,
                    selected_base.ridge_penalty,
                    graph,
                    selected_base.transport_multiplier,
                )
                try:
                    fit = _fit_structured(
                        tables,
                        support,
                        design["rna_incidence"],
                        design["adt_incidence"],
                        config,
                    )
                    for donor in validation:
                        structured_losses[config][donor_index[donor]] = (
                            _population_loss(
                                records[donor],
                                mask,
                                fit["population_log_odds"],
                                config.transport_multiplier,
                            )
                        )
                except (
                    ValueError,
                    FloatingPointError,
                    CouplingEstimationRefusal,
                ) as error:
                    refusals.append(
                        {
                            "stage": "nonzero_hypergraph",
                            "held_batch": held_batch,
                            "configuration": asdict(config),
                            "reason_code": type(error).__name__,
                            "reason": str(error),
                        }
                    )
        print(
            json.dumps({"completed_stage_b_fold": held_batch}, sort_keys=True),
            flush=True,
        )

    selected_primary = _select_structured(structured_losses, batches)
    selected_zero = BaseConfig(
        selected_primary.heterogeneity_penalty,
        selected_primary.ridge_penalty,
        selected_primary.transport_multiplier,
    )
    primary_values = structured_losses[selected_primary]
    zero_values = base_losses[selected_zero]
    gate = _source_gate(selected_primary, primary_values, zero_values, batches)

    selected_common = _select_transport(common_losses, batches)
    selected_poisson = _select_transport(poisson_losses, batches)
    selected_residuals = {
        family: _select_transport(
            {
                config.transport_multiplier: values
                for config, values in residual_losses.items()
                if config.family == family
            },
            batches,
        )
        for family in engine.RESIDUAL_FAMILIES
    }
    primary_residual = _select_residual_family(
        residual_losses, selected_residuals, batches
    )

    destroyed_losses = {alpha: np.full(len(donors), np.nan) for alpha in TRANSPORT_GRID}
    for held_batch in SOURCE_BATCHES:
        training = [
            donor for donor, batch in zip(donors, batches) if batch != held_batch
        ]
        validation = [
            donor for donor, batch in zip(donors, batches) if batch == held_batch
        ]
        _, destroyed, support, _ = _fold_arrays(records, training)
        design = _training_design(records, training, selected_primary.graph_neighbors)
        structural = StructuredConfig(
            selected_primary.graph_neighbors,
            selected_primary.heterogeneity_penalty,
            selected_primary.ridge_penalty,
            selected_primary.graph_penalty,
            1.0,
        )
        fit = _fit_structured(
            destroyed,
            support,
            design["rna_incidence"],
            design["adt_incidence"],
            structural,
        )
        for alpha in TRANSPORT_GRID:
            for donor in validation:
                destroyed_losses[alpha][donor_index[donor]] = _population_loss(
                    records[donor],
                    design["mask"],
                    fit["population_log_odds"],
                    alpha,
                )
    selected_destroyed = _select_transport(destroyed_losses, batches)
    comparator_selection = {
        "common_effect_cmle": selected_common,
        "pooled_saturated_poisson": selected_poisson,
        "primary_classical_residual": asdict(primary_residual),
        **{f"{family}_residual": alpha for family, alpha in selected_residuals.items()},
    }

    development = {
        "folds": fold_designs,
        "stage_a_matched_graph_zero": {
            "selection_rule": "minimum equal-direction mean loss; then eta, ridge, alpha",
            "selected_configuration": asdict(selected_base),
            "loss_curve": [
                _curve_entry(config, values, donors, batches)
                for config, values in sorted(base_losses.items())
            ],
        },
        "stage_b_nonzero_hypergraph": {
            "fixed_from_stage_a": asdict(selected_base),
            "selection_rule": "minimum equal-direction mean loss; then graph penalty, then k",
            "selected_configuration": asdict(selected_primary),
            "loss_curve": [
                _curve_entry(config, values, donors, batches)
                for config, values in sorted(structured_losses.items())
            ],
        },
        "source_go_gate": gate,
        "comparators": {
            "selected_transport": comparator_selection,
            "common_effect_cmle_curve": [
                _curve_entry({"transport_multiplier": alpha}, values, donors, batches)
                for alpha, values in sorted(common_losses.items())
            ],
            "pooled_saturated_poisson_curve": [
                _curve_entry({"transport_multiplier": alpha}, values, donors, batches)
                for alpha, values in sorted(poisson_losses.items())
            ],
            "residual_curve": [
                _curve_entry(config, values, donors, batches)
                for config, values in sorted(residual_losses.items())
            ],
            "independence": _curve_entry(
                {"method": "independence"},
                independence_losses,
                donors,
                batches,
            ),
            "destroyed_link_selected_transport": selected_destroyed,
            "destroyed_link_curve": [
                _curve_entry({"transport_multiplier": alpha}, values, donors, batches)
                for alpha, values in sorted(destroyed_losses.items())
            ],
        },
        "selected_primary_losses": primary_values.tolist(),
        "matched_graph_zero_losses": zero_values.tolist(),
        "refusals": refusals,
    }
    if not gate["passes"]:
        return {
            "status": "SOURCE_GO_GATE_FAILED",
            "b3_numeric_access_gate_passed": False,
            "development": development,
            "candidate": None,
        }

    try:
        topology_null_selections = _select_topology_nulls(data, records, selected_base)
        candidate = _fit_candidate(
            data,
            records,
            selected_primary,
            comparator_selection,
            selected_destroyed,
            topology_null_selections,
        )
    except CouplingEstimationRefusal as error:
        return {
            "status": "SOURCE_CANDIDATE_FREEZE_REFUSED",
            "b3_numeric_access_gate_passed": False,
            "development": development,
            "candidate": None,
            "refusal": {
                "reason_code": type(error).__name__,
                "reason": str(error),
                "details": error.details if isinstance(error, SourceGoRefusal) else {},
            },
        }
    return {
        "status": "SOURCE_GO_GATE_PASSED_CANDIDATE_FROZEN",
        "b3_numeric_access_gate_passed": True,
        "development": development,
        "candidate": candidate,
    }


def _artifact(path: Path, manifest_path: Path, data: SourceData) -> dict[str, Any]:
    input_snapshot = {
        "source_reduction_sha256": _sha256(path),
        "source_reduction_manifest_sha256": _sha256(manifest_path),
    }
    if (
        input_snapshot["source_reduction_sha256"] != data.manifest["output_sha256"]
        or input_snapshot["source_reduction_manifest_sha256"] != data.manifest_sha256
    ):
        raise PermissionError("loaded source bytes differ from their frozen manifest")
    implementation = _implementation_snapshot()
    try:
        result = _develop(data)
    except CouplingEstimationRefusal as error:
        result = {
            "status": "SOURCE_DEVELOPMENT_REFUSED",
            "b3_numeric_access_gate_passed": False,
            "development": None,
            "candidate": None,
            "refusal": {
                "reason_code": type(error).__name__,
                "reason": str(error),
                "details": error.details if isinstance(error, SourceGoRefusal) else {},
            },
        }
    if input_snapshot != {
        "source_reduction_sha256": _sha256(path),
        "source_reduction_manifest_sha256": _sha256(manifest_path),
    }:
        raise PermissionError("source bytes changed during candidate development")
    if implementation != _implementation_snapshot():
        raise PermissionError(
            "implementation bytes changed during candidate development"
        )
    return {
        "schema": "gse334503-b1-b2-source-candidate/1.0",
        "accession": "GSE334503",
        "numeric_batches_processed": list(SOURCE_BATCHES),
        "forbidden_numeric_batches": ["Batch3", "Batch4", "Batch5"],
        "source_reduction_path": _display_path(path),
        "source_reduction_sha256": input_snapshot["source_reduction_sha256"],
        "source_reduction_manifest_path": _display_path(manifest_path),
        "source_reduction_manifest_sha256": input_snapshot[
            "source_reduction_manifest_sha256"
        ],
        "source_reduction_manifest_schema": data.manifest["schema"],
        "source_reduction_manifest_output_sha256": data.manifest["output_sha256"],
        "source_profile_input_key": data.profile_input_key,
        "sample_axis": data.donors,
        "sample_batches": data.batches,
        "sample_axis_sha256": _axis_sha256(data.donors),
        "panel": [
            {"rna_gene": gene, "adt_protein": protein} for gene, protein in PANEL
        ],
        "panel_axis_sha256": _axis_sha256(
            [f"{gene}\t{protein}" for gene, protein in PANEL]
        ),
        "cell_budget_per_donor": CELL_BUDGET,
        "source_contract": {
            "source_development": "Batch1 and Batch2 directional folds only",
            "internal_validation": "Batch3 remains numerically untouched until this artifact is frozen",
            "held_confirmations": ["Batch4", "Batch5"],
            "rna_profile": "Jeffreys-smoothed donor detection logit",
            "adt_profile": "provided donor mean of cellwise 130-feature CLR",
            "binary_estimand": {
                "rna_state": "observed count > 0",
                "adt_state": "observed count > 0",
                "marker_support": "8 <= positive cells <= 504 within each donor",
                "destroyed_link": (
                    "deterministic cyclic permutation of complete binary ADT rows "
                    "within each donor"
                ),
                "midrank_assignment_used": False,
            },
            "training_only_fold_masks_and_geometry": True,
            "b3_numeric_access_requires_external_candidate_freeze": True,
            "comparison_mask_floor": {
                "minimum_coordinates": MINIMUM_MASK_COORDINATES,
                "minimum_per_donor_coordinates": MINIMUM_DONOR_COORDINATES,
                "rationale": (
                    "A source-only marginal-support preflight, performed before model "
                    "fitting or loss inspection, found directional training-mask counts "
                    "of 458 and 480 under count-positive RNA and ADT states, with "
                    "per-training-donor support of 416--480, directional-validation "
                    "support of 397--459, and 483 coordinates in the all-source mask. "
                    "The availability floors were therefore frozen at 440 overall and "
                    "390 per donor before any performance result was inspected; the "
                    "per-donor floor retains more than 80% of the 484-coordinate field "
                    "and applies unchanged to source, Batch3, Batch4, and Batch5."
                ),
                "performance_gate_unchanged_by_feasibility_review": True,
            },
        },
        "b3_numeric_access_authorized": False,
        "implementation": {
            **implementation,
            "conditional_solver": (
                "mapreg.penalty_complete_conditional_coupling."
                "fit_hierarchical_conditional_log_odds"
            ),
        },
        **result,
    }


def _write_json_x(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    _require_runtime()
    data = _load_source(args.input, args.manifest)
    result = _artifact(args.input, args.manifest, data)
    _write_json_x(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
