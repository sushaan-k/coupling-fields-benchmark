"""Retrospective nonheld head-to-head for exact conditional coupling fields."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import gammaln, logsumexp

from experiments import confirm_combat_citeseq as combat
from experiments import confirm_scmmib_bmmc as bmmc_lock
from experiments import evaluate_gse279451_sepsis_development as classical
from experiments import evaluate_scmmib_bmmc_exact_development as bmmc_io
from mapreg.heterogeneity_adaptive_coupling import (
    CouplingEstimationRefusal,
    product_hypergraph_laplacian,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMBAT = ROOT / "data/development/combat_citeseq/reduced_v1.json"
DEFAULT_BMMC_SOURCE = (
    ROOT.parents[1] / "data/confirmation/scmmib_bmmc/source_manifest_v1.json"
)
DEFAULT_OUTPUT = ROOT / "results/development/exact_logodds_head_to_head_v1.json"

COMBAT_SHA256 = "06ebfad44c339d763a451662462d7c9dc60684e792d5c229a42eefd232302b2f"
BMMC_SOURCE_SHA256 = "0869c3db2a5c22d7e211f31b75d66ee85c63de6bce74782934a17ec090d83e37"
BMMC_H5AD_SHA256 = "a0d193bb30f01a280c219c3ac40fd4b0a46b7c3bcb89223a1053f0f9a4cef434"
BMMC_METADATA_SHA256 = (
    "b267d4a820b062d0a05227c9cab61d389dcf924c3a6e062fb2389ce1be2f6e4f"
)

HETEROGENEITY_GRID = (0.1, 1.0, 10.0)
RIDGE_GRID = (0.01, 0.1)
GRAPH_GRID = (0.0, 0.1, 0.3, 1.0)
ALPHA_GRID = (0.5, 0.75, 1.0, 1.25)
BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 20260828
MAXIMUM_CONDITION_NUMBER = 1e12
TOLERANCE = 1e-8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    serialized = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(serialized)


@dataclass(frozen=True)
class ExactFit:
    population_log_odds: np.ndarray
    donor_log_odds: np.ndarray
    objective: float
    scaled_gradient_norm: float
    schur_condition_number: float
    theta_condition_number: float
    iterations: int


@dataclass(frozen=True)
class ConditionalWorkspace:
    donors: int
    entity_shape: tuple[int, int]
    observed: np.ndarray
    support: np.ndarray
    valid: np.ndarray
    log_probability: np.ndarray
    informative: np.ndarray
    null_precision: np.ndarray


def _conditional_workspace(tables: np.ndarray) -> ConditionalWorkspace:
    values = np.asarray(tables)
    if (
        values.ndim != 5
        or values.shape[-2:] != (2, 2)
        or values.shape[0] < 2
        or not np.issubdtype(values.dtype, np.integer)
        or np.any(values < 0)
    ):
        raise ValueError(
            "tables must be nonnegative integer donor x entity x entity x 2 x 2"
        )
    donors = values.shape[0]
    entity_shape = values.shape[1:3]
    flat = values.reshape(-1, 2, 2).astype(float)
    rows = flat.sum(axis=-1)
    columns = flat.sum(axis=-2)
    total = flat.sum(axis=(-2, -1))
    if np.any(total <= 0) or np.any(total != np.rint(total)):
        raise ValueError("every table must have a positive integer total")
    lower = np.maximum(0, rows[:, 0] + columns[:, 0] - total).astype(int)
    upper = np.minimum(rows[:, 0], columns[:, 0]).astype(int)
    width = int(np.max(upper - lower + 1))
    support = lower[:, None] + np.arange(width)[None, :]
    valid = support <= upper[:, None]
    x = support.astype(float)
    c0 = columns[:, 0, None]
    c1 = columns[:, 1, None]
    r0 = rows[:, 0, None]
    log_weight = (
        gammaln(c0 + 1.0)
        - gammaln(x + 1.0)
        - gammaln(c0 - x + 1.0)
        + gammaln(c1 + 1.0)
        - gammaln(r0 - x + 1.0)
        - gammaln(c1 - r0 + x + 1.0)
    )
    log_weight[~valid] = -np.inf
    log_probability = log_weight - logsumexp(log_weight, axis=1, keepdims=True)
    probability = np.exp(log_probability)
    probability[~valid] = 0.0
    expected = np.sum(probability * x, axis=1)
    null_precision = np.sum(probability * np.square(x - expected[:, None]), axis=1)
    informative = upper > lower
    null_precision[~informative] = 0.0
    support_count = informative.reshape(donors, -1).sum(axis=0)
    if np.any(support_count < 2):
        raise CouplingEstimationRefusal(
            "too few informative source units for an entity"
        )
    return ConditionalWorkspace(
        donors=donors,
        entity_shape=entity_shape,
        observed=flat[:, 0, 0],
        support=x,
        valid=valid,
        log_probability=log_probability,
        informative=informative,
        null_precision=null_precision,
    )


def _likelihood_statistics(
    workspace: ConditionalWorkspace, donor_log_odds: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    theta = np.asarray(donor_log_odds, dtype=float).reshape(-1)
    log_mass = (
        workspace.log_probability
        + (workspace.support - workspace.observed[:, None]) * theta[:, None]
    )
    normalizer = logsumexp(log_mass, axis=1)
    probability = np.exp(log_mass - normalizer[:, None])
    probability[~workspace.valid] = 0.0
    expected = np.sum(probability * workspace.support, axis=1)
    precision = np.sum(
        probability * np.square(workspace.support - expected[:, None]), axis=1
    )
    score = expected - workspace.observed
    score[~workspace.informative] = 0.0
    precision[~workspace.informative] = 0.0
    if (
        not np.isfinite(normalizer).all()
        or not np.isfinite(score).all()
        or np.any(precision[workspace.informative] <= 0.0)
    ):
        raise CouplingEstimationRefusal(
            "exact conditional likelihood left finite support"
        )
    return float(normalizer.sum()), score, precision


def _fit_exact_hierarchical(
    tables: np.ndarray,
    first_incidence: np.ndarray,
    second_incidence: np.ndarray,
    *,
    heterogeneity: float,
    ridge: float,
    graph: float,
) -> ExactFit:
    workspace = _conditional_workspace(tables)
    donors = workspace.donors
    entities = int(np.prod(workspace.entity_shape))
    null = workspace.null_precision.reshape(donors, entities)
    support = workspace.informative.reshape(donors, entities)
    heterogeneity_scale = float(np.median(null[support]))
    entity_precision = null.sum(axis=0) / donors
    population_scale = float(np.median(entity_precision[entity_precision > 0.0]))
    eta = float(heterogeneity) * heterogeneity_scale / donors
    ridge_effective = float(ridge) * population_scale
    graph_effective = float(graph) * population_scale
    laplacian = product_hypergraph_laplacian(first_incidence, second_incidence)
    theta = np.zeros((donors, entities), dtype=float)
    mu = np.zeros(entities, dtype=float)
    donor_scale = max(1.0, heterogeneity_scale) / donors
    population_gradient_scale = max(1.0, population_scale)

    def evaluate(
        current_theta: np.ndarray, current_mu: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        likelihood, raw_score, raw_precision = _likelihood_statistics(
            workspace, current_theta
        )
        score = raw_score.reshape(donors, entities) / donors
        precision = raw_precision.reshape(donors, entities) / donors
        deviation = current_theta - current_mu[None, :]
        graph_action = laplacian @ current_mu
        objective = likelihood / donors
        objective += 0.5 * eta * float(np.sum(np.square(deviation)))
        objective += 0.5 * ridge_effective * float(np.sum(np.square(current_mu)))
        objective += 0.5 * graph_effective * float(current_mu @ graph_action)
        donor_gradient = score + eta * deviation
        population_gradient = (
            -eta * deviation.sum(axis=0)
            + ridge_effective * current_mu
            + graph_effective * graph_action
        )
        return objective, donor_gradient, population_gradient, precision

    for iteration in range(101):
        objective, donor_gradient, population_gradient, precision = evaluate(theta, mu)
        scaled_gradient = max(
            float(np.max(np.abs(donor_gradient))) / donor_scale,
            float(np.max(np.abs(population_gradient))) / population_gradient_scale,
        )
        if scaled_gradient <= TOLERANCE:
            break
        if iteration == 100:
            raise CouplingEstimationRefusal(
                "vectorized exact optimizer missed its gradient certificate"
            )
        curvature = precision + eta
        transmitted = eta * precision / curvature
        schur = graph_effective * laplacian.copy()
        diagonal = np.arange(entities)
        schur[diagonal, diagonal] += transmitted.sum(axis=0) + ridge_effective
        population_penalty_gradient = ridge_effective * mu + graph_effective * (
            laplacian @ mu
        )
        data_score = donor_gradient - eta * (theta - mu[None, :])
        right_hand_side = -population_penalty_gradient + np.sum(
            transmitted * (theta - mu[None, :]) - (eta / curvature) * data_score,
            axis=0,
        )
        population_step = np.linalg.solve(schur, right_hand_side)
        donor_step = (-donor_gradient + eta * population_step[None, :]) / curvature
        directional = float(
            np.sum(donor_gradient * donor_step)
            + np.sum(population_gradient * population_step)
        )
        if not np.isfinite(directional) or directional >= 0.0:
            raise CouplingEstimationRefusal("exact Newton direction is not descending")
        step = 1.0
        for _ in range(48):
            candidate_theta = theta + step * donor_step
            candidate_mu = mu + step * population_step
            try:
                candidate_objective = evaluate(candidate_theta, candidate_mu)[0]
            except CouplingEstimationRefusal:
                candidate_objective = np.inf
            if candidate_objective <= objective + 1e-4 * step * directional:
                theta, mu = candidate_theta, candidate_mu
                break
            step *= 0.5
        else:
            raise CouplingEstimationRefusal("exact Newton line search refused")

    objective, donor_gradient, population_gradient, precision = evaluate(theta, mu)
    scaled_gradient = max(
        float(np.max(np.abs(donor_gradient))) / donor_scale,
        float(np.max(np.abs(population_gradient))) / population_gradient_scale,
    )
    curvature = precision + eta
    transmitted = eta * precision / curvature
    schur = graph_effective * laplacian.copy()
    diagonal = np.arange(entities)
    schur[diagonal, diagonal] += transmitted.sum(axis=0) + ridge_effective
    schur_eigenvalues = np.linalg.eigvalsh(schur)
    if schur_eigenvalues[0] <= 0.0 or np.min(curvature) <= 0.0:
        raise CouplingEstimationRefusal(
            "exact conditional Hessian is not positive definite"
        )
    schur_condition = float(schur_eigenvalues[-1] / schur_eigenvalues[0])
    theta_condition = float(np.max(curvature) / np.min(curvature))
    if max(schur_condition, theta_condition) > MAXIMUM_CONDITION_NUMBER:
        raise CouplingEstimationRefusal(
            "exact conditional solve exceeds condition limit"
        )
    return ExactFit(
        population_log_odds=mu.reshape(workspace.entity_shape),
        donor_log_odds=theta.reshape((donors, *workspace.entity_shape)),
        objective=float(objective),
        scaled_gradient_norm=float(scaled_gradient),
        schur_condition_number=schur_condition,
        theta_condition_number=theta_condition,
        iterations=int(iteration),
    )


def _target_losses(
    target_tables: np.ndarray, source_log_odds: np.ndarray, alpha: float
) -> np.ndarray:
    losses = []
    for tables in target_tables:
        recipient = classical._conditional_support(tables)
        prediction = classical._predict_conditional(source_log_odds, alpha, recipient)
        losses.append(classical._donor_loss(tables, prediction))
    return np.asarray(losses)


def _field_candidates(
    source_tables: np.ndarray,
    target_tables: np.ndarray,
    graphs: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    cache: dict[tuple[Any, ...], ExactFit | Exception] = {}
    for neighbors, heterogeneity, ridge, graph in itertools.product(
        sorted(graphs), HETEROGENEITY_GRID, RIDGE_GRID, GRAPH_GRID
    ):
        cache_key = (
            ("ridge", heterogeneity, ridge)
            if graph == 0.0
            else (neighbors, heterogeneity, ridge, graph)
        )
        if cache_key not in cache:
            try:
                cache[cache_key] = _fit_exact_hierarchical(
                    source_tables,
                    *graphs[neighbors],
                    heterogeneity=heterogeneity,
                    ridge=ridge,
                    graph=graph,
                )
            except (CouplingEstimationRefusal, FloatingPointError) as error:
                cache[cache_key] = error
        fit = cache[cache_key]
        if isinstance(fit, Exception):
            candidates.append(
                {
                    "neighbors": neighbors,
                    "heterogeneity": heterogeneity,
                    "ridge": ridge,
                    "graph": graph,
                    "status": "REFUSED",
                    "reason": str(fit),
                }
            )
            continue
        for alpha in ALPHA_GRID:
            losses = _target_losses(target_tables, fit.population_log_odds, alpha)
            candidates.append(
                {
                    "neighbors": neighbors,
                    "heterogeneity": heterogeneity,
                    "ridge": ridge,
                    "graph": graph,
                    "alpha": alpha,
                    "status": "OK",
                    "mean_loss": float(losses.mean()),
                    "unit_losses": losses.tolist(),
                    "certificate": {
                        "objective": fit.objective,
                        "scaled_gradient_norm": fit.scaled_gradient_norm,
                        "schur_condition_number": fit.schur_condition_number,
                        "theta_condition_number": fit.theta_condition_number,
                        "iterations": fit.iterations,
                    },
                }
            )
    valid = [row for row in candidates if row["status"] == "OK"]
    if not valid:
        raise CouplingEstimationRefusal("all exact conditional candidates refused")
    selected = min(
        valid,
        key=lambda row: (
            row["mean_loss"],
            row["neighbors"],
            row["heterogeneity"],
            row["ridge"],
            row["graph"],
            row["alpha"],
        ),
    )
    return selected, candidates


def _residual_candidates(
    source_tables: np.ndarray, target_tables: np.ndarray
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = []
    support_count = bmmc_io._informative(source_tables).sum(axis=0)
    for family, centered, alpha in itertools.product(
        ("pearson", "deviance"), (False, True), ALPHA_GRID
    ):
        pooled = bmmc_io._residual_coordinate(source_tables, family, centered)
        losses = bmmc_io._losses(
            target_tables,
            bmmc_io._predict_residual(
                pooled,
                target_tables,
                family=family,
                centered=centered,
                alpha=alpha,
            ),
        )
        candidates.append(
            {
                "family": family,
                "centered": centered,
                "alpha": alpha,
                "mean_loss": float(losses.mean()),
                "unit_losses": losses.tolist(),
                "pool_audit": {
                    "support_count_range": [
                        int(support_count.min()),
                        int(support_count.max()),
                    ],
                    "sample_size_normalized": True,
                    "donor_equal_pooling": True,
                    "source_coordinate_divisor": "sqrt(source table total)",
                    "target_coordinate_multiplier": "sqrt(target table total)",
                    "direct_feasible_inversion": True,
                    "boundary_clamp": "epsilon inside the closed feasible table interval",
                },
            }
        )
    return min(
        candidates,
        key=lambda row: (
            row["mean_loss"],
            row["family"],
            row["centered"],
            row["alpha"],
        ),
    ), candidates


def _comparison(
    primary: np.ndarray, comparator: np.ndarray, seed: int
) -> dict[str, Any]:
    first = np.asarray(primary, dtype=float)
    second = np.asarray(comparator, dtype=float)
    if first.shape != second.shape or first.ndim != 1 or len(first) < 2:
        raise ValueError("paired unit losses are invalid")
    difference = first - second
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(first), size=(BOOTSTRAPS, len(first)))
    bootstrap = difference[indices].mean(axis=1)
    return {
        "primary_mean_deviance": float(first.mean()),
        "comparator_mean_deviance": float(second.mean()),
        "relative_reduction": float(1.0 - first.mean() / second.mean()),
        "paired_difference_primary_minus_comparator": difference.tolist(),
        "paired_unit_bootstrap_95_ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "favorable_units": int(np.count_nonzero(difference < 0.0)),
        "total_units": int(len(first)),
        "bootstrap_draws": BOOTSTRAPS,
        "bootstrap_seed": int(seed),
    }


def _combat_panel(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if _sha256(path) != COMBAT_SHA256:
        raise PermissionError("COMBAT development reduction SHA-256 differs")
    payload = json.loads(path.read_text())
    records = payload.get("samples")
    audit = payload.get("access_audit", {})
    if (
        payload.get("schema") != "combat-citeseq-reduced-development/1.0"
        or payload.get("status") != "DEVELOPMENT_REDUCTION_COMPLETE"
        or payload.get("markers") != list(combat.MARKERS)
        or not isinstance(records, list)
        or len(records) != 36
        or audit.get("calibration_samples_read") != 12
        or audit.get("pilot_samples_read") != 24
        or audit.get("held_donor_matrix_rows_read") != 0
        or audit.get("held_site_matrix_rows_read") != 0
    ):
        raise PermissionError("COMBAT reduction violates its nonheld seal")
    by_sample = {record["sample"]: record for record in records}
    if len(by_sample) != 36 or any(
        record.get("role") not in {"calibration", "pilot"} for record in records
    ):
        raise PermissionError("COMBAT reduction contains a held or duplicate sample")
    calibration = tuple(
        sorted(
            record["sample"] for record in records if record["role"] == "calibration"
        )
    )
    pilot = tuple(
        sorted(record["sample"] for record in records if record["role"] == "pilot")
    )
    if len(calibration) != 12 or len(pilot) != 24:
        raise ValueError("COMBAT role counts differ")
    source_tables = combat._tables(by_sample, calibration, "tables")
    destroyed_tables = combat._tables(by_sample, calibration, "destroyed_tables")
    target_tables = combat._tables(by_sample, pilot, "tables")
    graphs = {
        neighbors: combat._graphs(by_sample, calibration, neighbors)[:2]
        for neighbors in (1, 2)
    }
    return (
        {
            "name": "COMBAT Oxford calibration-to-pilot",
            "unit_labels": list(pilot),
            "source_tables": source_tables,
            "destroyed_tables": destroyed_tables,
            "target_tables": target_tables,
            "graphs": graphs,
        },
        {
            "reduction_sha256": COMBAT_SHA256,
            "source_manifest_sha256": payload.get("source_manifest_sha256"),
            "h5ad_sha256": payload.get("h5ad_sha256"),
            "calibration_records_read": 12,
            "pilot_records_read": 24,
            "held_donor_records_read": 0,
            "held_site_records_read": 0,
            "matrix_payload_reopened": False,
        },
    )


def _resolved_source_path(source_manifest: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise PermissionError("BMMC source path is not repository-relative")
    root = source_manifest.resolve().parents[3]
    path = (root / relative).resolve()
    path.relative_to(root)
    if not path.is_file():
        raise FileNotFoundError(path.name)
    return path


def _destroy_bmmc_adt(adt: np.ndarray, cells: list[dict[str, str]]) -> np.ndarray:
    destroyed = np.empty_like(adt)
    strata: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(cells):
        strata.setdefault((row["DonorID"], row["cell_type.l1"]), []).append(index)
    for (donor, lineage), members in sorted(strata.items()):
        selected = np.asarray(members, dtype=int)
        generator = np.random.default_rng(bmmc_io._seed("destroy", donor, lineage))
        destroyed[selected] = adt[generator.permutation(selected)]
    return destroyed


def _bmmc_panel(source_manifest: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if _sha256(source_manifest) != BMMC_SOURCE_SHA256:
        raise PermissionError("BMMC authorized source manifest SHA-256 differs")
    source = json.loads(source_manifest.read_text())
    complete = source.get("complete_cite_h5ad", {})
    metadata_record = source.get("metadata", {})
    preflight_record = source.get("preflight", {})
    if (
        source.get("schema") != "scmmib-bmmc-source/1.0"
        or source.get("status") != "NONHELD_SOURCE_ACCESS_AUTHORIZED"
        or source.get("held_feature_rows_may_be_read") is not False
        or complete.get("sha256") != BMMC_H5AD_SHA256
        or metadata_record.get("sha256") != BMMC_METADATA_SHA256
    ):
        raise PermissionError("BMMC source authorization differs")
    h5ad = _resolved_source_path(source_manifest, complete.get("local_path"))
    metadata_path = _resolved_source_path(
        source_manifest, metadata_record.get("local_path")
    )
    preflight = _resolved_source_path(source_manifest, preflight_record.get("path"))
    if (
        h5ad.stat().st_size != int(complete.get("bytes", -1))
        or _sha256(h5ad) != BMMC_H5AD_SHA256
        or _sha256(metadata_path) != BMMC_METADATA_SHA256
        or _sha256(preflight) != preflight_record.get("sha256")
    ):
        raise PermissionError("BMMC authorized source bytes differ")
    metadata = bmmc_io._metadata(metadata_path)
    _, roles = bmmc_lock._metadata_roles(metadata_path)
    axis = bmmc_lock._axis(h5ad, source["combined_assay"])
    role_rows = bmmc_lock._row_vectors(axis, roles)
    selected_rows = np.sort(np.r_[role_rows["fit"], role_rows["development"]])
    forbidden_rows = role_rows["held"]
    columns = np.asarray(
        axis["marker_indices"]["rna"] + axis["marker_indices"]["adt"], dtype=int
    )
    counts, read_audit = bmmc_io._read_csr_marker_counts(
        h5ad,
        matrix_path=source["combined_assay"]["matrix_hdf5_path"],
        selected_rows=selected_rows,
        forbidden_rows=forbidden_rows,
        selected_columns=columns,
    )
    metadata_by_barcode = {row["barcode"]: row for row in metadata}
    cells = [metadata_by_barcode[axis["barcodes"][index]] for index in selected_rows]
    if any(row["DonorID"] in bmmc_lock.HELD_DONORS for row in cells):
        raise PermissionError("BMMC held donor entered the nonheld benchmark")
    rna = (counts[:, :10] > 0).astype(np.int8)
    adt = bmmc_io._rank_binary_adt(counts[:, 10:], cells)
    donors = np.asarray([row["DonorID"] for row in cells])
    fit_mask = np.isin(donors, bmmc_lock.FIT_DONORS)
    bridge_mask = donors == bmmc_lock.DEVELOPMENT_DONORS[0]
    fit_cells = [row for row, keep in zip(cells, fit_mask) if keep]
    bridge_cells = [row for row, keep in zip(cells, bridge_mask) if keep]
    fit_labels, source_tables = bmmc_io._tables(
        rna[fit_mask], adt[fit_mask], fit_cells, "DonorID"
    )
    unit_labels, target_tables = bmmc_io._tables(
        rna[bridge_mask], adt[bridge_mask], bridge_cells, "batch"
    )
    destroyed_adt = _destroy_bmmc_adt(adt[fit_mask], fit_cells)
    _, destroyed_tables = bmmc_io._tables(
        rna[fit_mask], destroyed_adt, fit_cells, "DonorID"
    )
    if fit_labels != list(bmmc_lock.FIT_DONORS) or unit_labels != [
        "s1d1",
        "s2d1",
        "s3d1",
        "s4d1",
    ]:
        raise ValueError("BMMC fit or bridge axes differ")
    graphs = {
        neighbors: (
            bmmc_io._prevalence_incidence(rna[fit_mask], fit_cells, neighbors),
            bmmc_io._prevalence_incidence(adt[fit_mask], fit_cells, neighbors),
        )
        for neighbors in (1, 2, 3)
    }
    return (
        {
            "name": "SCMMIB BMMC fit-to-bridge",
            "unit_labels": unit_labels,
            "source_tables": source_tables,
            "destroyed_tables": destroyed_tables,
            "target_tables": target_tables,
            "graphs": graphs,
        },
        {
            "source_manifest_sha256": BMMC_SOURCE_SHA256,
            "complete_h5ad_sha256": BMMC_H5AD_SHA256,
            "metadata_sha256": BMMC_METADATA_SHA256,
            "complete_h5ad_opaque_hash_verified": True,
            "fit_feature_rows_decoded": int(np.count_nonzero(fit_mask)),
            "bridge_feature_rows_decoded": int(np.count_nonzero(bridge_mask)),
            "held_feature_rows_forbidden": int(len(forbidden_rows)),
            "held_feature_rows_decoded": 0,
            "held_tables_formed": 0,
            "held_outcomes_used": False,
            "full_observation_axis_read": True,
            **read_audit,
        },
    )


def _evaluate_panel(panel: dict[str, Any], seed: int) -> dict[str, Any]:
    selected, field_candidates = _field_candidates(
        panel["source_tables"], panel["target_tables"], panel["graphs"]
    )
    residual, residual_candidates = _residual_candidates(
        panel["source_tables"], panel["target_tables"]
    )
    first, second = panel["graphs"][int(selected["neighbors"])]
    destroyed_fit = _fit_exact_hierarchical(
        panel["destroyed_tables"],
        first,
        second,
        heterogeneity=float(selected["heterogeneity"]),
        ridge=float(selected["ridge"]),
        graph=float(selected["graph"]),
    )
    destroyed_losses = _target_losses(
        panel["target_tables"], destroyed_fit.population_log_odds, selected["alpha"]
    )
    primary_losses = np.asarray(selected["unit_losses"], dtype=float)
    residual_losses = np.asarray(residual["unit_losses"], dtype=float)
    valid_field = [row for row in field_candidates if row["status"] == "OK"]
    best_graph_zero = min(
        (row for row in valid_field if row["graph"] == 0.0),
        key=lambda row: (
            row["mean_loss"],
            row["neighbors"],
            row["heterogeneity"],
            row["ridge"],
            row["alpha"],
        ),
    )
    best_positive_graph = min(
        (row for row in valid_field if row["graph"] > 0.0),
        key=lambda row: (
            row["mean_loss"],
            row["neighbors"],
            row["heterogeneity"],
            row["ridge"],
            row["graph"],
            row["alpha"],
        ),
    )
    return {
        "panel": panel["name"],
        "status": "RETROSPECTIVE_ADAPTIVE_DEVELOPMENT_ONLY",
        "unit_labels": panel["unit_labels"],
        "unit_count": len(panel["unit_labels"]),
        "entity_count": int(np.prod(panel["source_tables"].shape[1:3])),
        "selection": {
            "same_nonheld_units_used_for_selection_and_summary": True,
            "confirmatory_inference": False,
            "primary": {
                key: selected[key]
                for key in (
                    "neighbors",
                    "heterogeneity",
                    "ridge",
                    "graph",
                    "alpha",
                    "mean_loss",
                    "certificate",
                )
            },
            "best_residual": {
                key: residual[key]
                for key in ("family", "centered", "alpha", "mean_loss", "pool_audit")
            },
        },
        "unit_losses": {
            "primary": dict(zip(panel["unit_labels"], primary_losses.tolist())),
            "best_residual": dict(zip(panel["unit_labels"], residual_losses.tolist())),
            "destroyed_link": dict(
                zip(panel["unit_labels"], destroyed_losses.tolist())
            ),
        },
        "comparisons": {
            "primary_vs_best_residual": _comparison(
                primary_losses, residual_losses, seed
            ),
            "primary_vs_destroyed_link": _comparison(
                primary_losses, destroyed_losses, seed + 1
            ),
        },
        "graph_diagnostic": {
            "primary_grid_included_graph_zero": True,
            "selected_graph_penalty": selected["graph"],
            "best_graph_zero_mean_deviance": best_graph_zero["mean_loss"],
            "best_positive_graph_mean_deviance": best_positive_graph["mean_loss"],
            "best_graph_zero_settings": {
                key: best_graph_zero[key]
                for key in ("neighbors", "heterogeneity", "ridge", "graph", "alpha")
            },
            "best_positive_graph_settings": {
                key: best_positive_graph[key]
                for key in ("neighbors", "heterogeneity", "ridge", "graph", "alpha")
            },
            "relative_reduction_positive_graph_vs_graph_zero": 1.0
            - best_positive_graph["mean_loss"] / best_graph_zero["mean_loss"],
            "graph_selected": bool(selected["graph"] > 0.0),
        },
        "grid_evaluation": {
            "primary_candidates": len(field_candidates),
            "primary_valid": len(valid_field),
            "primary_refused": len(field_candidates) - len(valid_field),
            "primary_evaluations_sha256": _canonical_sha256(field_candidates),
            "residual_candidates": len(residual_candidates),
            "residual_evaluations_sha256": _canonical_sha256(residual_candidates),
        },
        "destroyed_link": {
            "source": "pre-existing deterministic within-stratum pairing destruction",
            "refit_at_selected_primary_setting": True,
            "source_table_sha256": _array_sha256(panel["destroyed_tables"]),
        },
    }


def run(combat_path: Path, bmmc_source: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("head-to-head result already exists")
    combat_panel, combat_access = _combat_panel(combat_path)
    bmmc_panel, bmmc_access = _bmmc_panel(bmmc_source)
    payload = {
        "schema": "exact-logodds-head-to-head-development/1.0",
        "status": "RETROSPECTIVE_ADAPTIVE_DEVELOPMENT_ONLY",
        "interpretation": "descriptive nonheld development evidence; not held confirmation or prospective inference",
        "estimator": "donor-heterogeneous exact fixed-margin conditional log-odds field with product-graph penalty",
        "comparator": "development-selected donor-equal signed Pearson or Poisson-deviance residual transfer; direct target-feasible inversion clamps boundary excess",
        "grid": {
            "heterogeneity": list(HETEROGENEITY_GRID),
            "ridge": list(RIDGE_GRID),
            "graph": list(GRAPH_GRID),
            "transport_alpha": list(ALPHA_GRID),
            "combat_neighbors": [1, 2],
            "bmmc_neighbors": [1, 2, 3],
        },
        "panels": {
            "combat": _evaluate_panel(combat_panel, BOOTSTRAP_SEED),
            "bmmc": _evaluate_panel(bmmc_panel, BOOTSTRAP_SEED + 10),
        },
        "bindings": {
            "evaluator_sha256": _sha256(Path(__file__)),
            "combat_reduction_sha256": COMBAT_SHA256,
            "bmmc_source_manifest_sha256": BMMC_SOURCE_SHA256,
            "bmmc_complete_h5ad_sha256": BMMC_H5AD_SHA256,
            "hierarchical_reference_module_sha256": _sha256(
                ROOT / "mapreg/hierarchical_conditional_coupling.py"
            ),
            "conditional_prediction_module_sha256": _sha256(
                ROOT / "experiments/evaluate_gse279451_sepsis_development.py"
            ),
            "classical_and_bmmc_reader_module_sha256": _sha256(
                ROOT / "experiments/evaluate_scmmib_bmmc_exact_development.py"
            ),
        },
        "access_audit": {
            "combat": combat_access,
            "bmmc": bmmc_access,
            "combat_held_access": 0,
            "bmmc_held_numeric_access": 0,
            "held_predictions_formed": 0,
        },
    }
    _write_json_exclusive(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combat-reduction", type=Path, default=DEFAULT_COMBAT)
    parser.add_argument(
        "--bmmc-source-manifest", type=Path, default=DEFAULT_BMMC_SOURCE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.combat_reduction, args.bmmc_source_manifest, args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
