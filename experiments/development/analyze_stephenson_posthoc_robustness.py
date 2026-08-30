"""Post-hoc Stephenson lineage and destroyed-link robustness analyses.

The analyses reuse the frozen Cambridge/Newcastle donor split, marker panel,
whole-cohort hyperparameters, and fixed-margin prediction rule. They do not
modify or replace any confirmatory artifact.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from experiments import audit_classical_interaction_baselines as classical_audit
from experiments import confirm_stephenson_citeseq as stephenson
from mapreg.hierarchical_conditional_coupling import (
    fit_hierarchical_conditional_log_odds,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_H5AD = (
    ROOT
    / "data/confirmation/stephenson_citeseq/"
    "covid_portal_210320_with_raw.h5ad"
)
DEFAULT_LINEAGE_JSON = (
    ROOT / "results/development/stephenson_lineage_sensitivity_v1.json"
)
DEFAULT_LINEAGE_TSV = (
    ROOT / "results/development/stephenson_lineage_sensitivity_v1.tsv"
)
DEFAULT_PERMUTATION_JSON = (
    ROOT
    / "results/development/"
    "stephenson_destroyed_link_permutation_robustness_v1.json"
)
DEFAULT_PERMUTATION_TSV = (
    ROOT
    / "results/development/"
    "stephenson_destroyed_link_permutation_robustness_v1.tsv"
)

LINEAGES = {
    "T_cells": ("T",),
    "B_cells": ("B",),
    "myeloid_cells": ("MNP",),
}
LINEAGE_CELL_BUDGET = 64
LINEAGE_CELL_SALT = "STEPHENSON-LINEAGE-SENSITIVITY-v1"
PERMUTATION_SALT = "STEPHENSON-DESTROYED-LINK-REPEATS-v1"
INDEPENDENT_PERMUTATIONS = 32
BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 20260829


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


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _source(h5ad: Path) -> dict[str, Any]:
    old = os.environ.get("STEPHENSON_CITESEQ_H5AD")
    os.environ["STEPHENSON_CITESEQ_H5AD"] = str(h5ad.resolve())
    try:
        return stephenson._validated_source(
            stephenson.DEFAULT_SOURCE, verify_hash=False
        )
    finally:
        if old is None:
            os.environ.pop("STEPHENSON_CITESEQ_H5AD", None)
        else:
            os.environ["STEPHENSON_CITESEQ_H5AD"] = old


def _observation_metadata(
    h5ad: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(h5ad, "r") as handle:
        obs = handle["obs"]
        barcodes = stephenson._dataframe_index(handle, obs)
        samples = stephenson._encoded_column(handle, obs, "sample_id")
        sites = stephenson._encoded_column(handle, obs, "Site")
        clusters = stephenson._encoded_column(handle, obs, "initial_clustering")
    cell_types = np.asarray(
        [stephenson.CLUSTER_TO_CELL_TYPE.get(value, "") for value in clusters],
        dtype=str,
    )
    if np.any(cell_types == ""):
        raise PermissionError("unmapped cell type entered the post-hoc analysis")
    return barcodes, samples, sites, cell_types


def _lineage_selections(
    records: list[dict[str, Any]],
    barcodes: np.ndarray,
    sample_ids: np.ndarray,
    sites: np.ndarray,
    cell_types: np.ndarray,
    lineage: str,
    labels: tuple[str, ...],
    budget: int = LINEAGE_CELL_BUDGET,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    selections: dict[str, np.ndarray] = {}
    inventory = []
    allowed = np.isin(cell_types, labels)
    for record in records:
        sample = record["sample"]
        rows = np.flatnonzero((sample_ids == sample) & allowed)
        if len(rows) and set(sites[rows].tolist()) != {record["site"]}:
            raise PermissionError(f"site labels differ for {sample}")
        status = "ELIGIBLE" if len(rows) >= budget else "EXCLUDED_CELL_SUPPORT"
        selected_rows = np.asarray([], dtype=np.int64)
        selected_hash = None
        if status == "ELIGIBLE":
            ordered = sorted(
                rows.tolist(),
                key=lambda row: (
                    hashlib.sha256(
                        "\0".join(
                            (
                                LINEAGE_CELL_SALT,
                                lineage,
                                record["donor"],
                                sample,
                                str(barcodes[row]),
                            )
                        ).encode()
                    ).hexdigest(),
                    str(barcodes[row]),
                ),
            )[:budget]
            selected_rows = np.asarray(sorted(ordered), dtype=np.int64)
            selections[sample] = selected_rows
            selected_hash = hashlib.sha256(
                ("\n".join(sorted(barcodes[selected_rows].tolist())) + "\n").encode()
            ).hexdigest()
        inventory.append(
            {
                "donor": record["donor"],
                "sample": sample,
                "site": record["site"],
                "role": record["role"],
                "available_lineage_cells": int(len(rows)),
                "selected_cells": int(len(selected_rows)),
                "selected_barcode_sha256": selected_hash,
                "status": status,
            }
        )
    return selections, inventory


def _read_union_counts(
    h5ad: Path,
    selections: list[dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    row_sets = [rows for panel in selections for rows in panel.values()]
    rows = np.unique(np.concatenate(row_sets)).astype(np.int64)
    with h5py.File(h5ad, "r") as handle:
        features = stephenson._feature_columns(handle)
        rna_columns = np.asarray(features["rna"], dtype=np.int64)
        adt_columns = np.asarray(features["adt"], dtype=np.int64)
        values = classical_audit._read_csr_columns_fast(
            handle["layers/raw"],
            rows,
            np.concatenate((rna_columns, adt_columns)),
        )
    marker_count = len(stephenson.MARKERS)
    row_lookup = {int(row): index for index, row in enumerate(rows)}
    return values[:, :marker_count], values[:, marker_count:], row_lookup


def _counts_for(
    rows: np.ndarray,
    values: np.ndarray,
    row_lookup: dict[int, int],
) -> np.ndarray:
    indices = np.asarray([row_lookup[int(row)] for row in rows], dtype=np.int64)
    return values[indices].T


def _midrank_states(
    counts: np.ndarray,
    barcodes: np.ndarray,
    donor: str,
    sample: str,
) -> np.ndarray:
    values = stephenson.numerics._integer_counts(counts, "ADT")
    budget = values.shape[1]
    if values.shape != (len(stephenson.MARKERS), budget) or budget % 2:
        raise ValueError("ADT count panel must have an even cell budget")
    states = np.ones_like(values, dtype=np.uint8)
    for marker_index, marker in enumerate(stephenson.MARKERS):
        ties = np.asarray(
            [
                hashlib.sha256(
                    "\0".join(
                        (
                            stephenson.ADT_TIE_SALT,
                            donor,
                            sample,
                            str(barcode),
                            marker,
                        )
                    ).encode()
                ).hexdigest()
                for barcode in barcodes
            ]
        )
        order = np.lexsort((ties, values[marker_index]))
        states[marker_index, order[: budget // 2]] = 0
    if np.any(states.sum(axis=1) != budget // 2):
        raise AssertionError("midrank ADT margins are not balanced")
    return states


def _form_tables(rna: np.ndarray, adt: np.ndarray) -> np.ndarray:
    first = np.asarray(rna, dtype=np.uint8)
    second = np.asarray(adt, dtype=np.uint8)
    if first.shape != second.shape or first.shape[0] != len(stephenson.MARKERS):
        raise ValueError("RNA and ADT states must share marker by cell shape")
    output = np.empty(
        (len(stephenson.MARKERS), len(stephenson.MARKERS), 2, 2),
        dtype=np.int64,
    )
    for rna_index in range(len(stephenson.MARKERS)):
        for adt_index in range(len(stephenson.MARKERS)):
            code = 2 * first[rna_index].astype(np.int64) + second[
                adt_index
            ].astype(np.int64)
            output[rna_index, adt_index] = np.bincount(
                code, minlength=4
            ).reshape(2, 2)
    return output


def _state_panel(
    records: list[dict[str, Any]],
    selections: dict[str, np.ndarray],
    barcodes: np.ndarray,
    rna_values: np.ndarray,
    adt_values: np.ndarray,
    row_lookup: dict[int, int],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    rna_states = {}
    adt_states = {}
    tables = {}
    for record in records:
        sample = record["sample"]
        if sample not in selections:
            continue
        rows = selections[sample]
        rna = (
            stephenson.numerics._integer_counts(
                _counts_for(rows, rna_values, row_lookup), "RNA"
            )
            > 0
        ).astype(np.uint8)
        adt = _midrank_states(
            _counts_for(rows, adt_values, row_lookup),
            barcodes[rows],
            record["donor"],
            sample,
        )
        rna_states[sample] = rna
        adt_states[sample] = adt
        tables[sample] = _form_tables(rna, adt)
    return rna_states, adt_states, tables


def _frozen_configuration(development: dict[str, Any]) -> dict[str, float]:
    selected = development["selection"]["selected_primary_configuration"]
    expected = development["selection"]["selected_graph_zero_configuration"]
    if selected != expected or selected["graph_penalty"] != 0.0:
        raise PermissionError("Stephenson frozen graph-zero selection differs")
    return selected


def _fit_conditional(
    tables: np.ndarray,
    development: dict[str, Any],
    *,
    tolerance: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _frozen_configuration(development)
    graph = development["all_development_graph"]
    fit = fit_hierarchical_conditional_log_odds(
        tables,
        np.asarray(graph["rna_incidence"], dtype=float),
        np.asarray(graph["adt_incidence"], dtype=float),
        heterogeneity_penalty=float(config["heterogeneity_penalty"]),
        ridge_penalty=float(config["ridge_penalty"]),
        graph_penalty=float(config["graph_penalty"]),
        minimum_informative_donors=int(config["minimum_informative_donors"]),
        maximum_condition_number=float(config["maximum_condition_number"]),
        maximum_iterations=100,
        tolerance=(
            float(config["gradient_tolerance"])
            if tolerance is None
            else float(tolerance)
        ),
    )
    model = {
        "kind": "conditional_log_odds",
        "source_coordinate": fit.population_log_odds.reshape(-1).tolist(),
        "alpha": float(config["transport_alpha"]),
    }
    certificate = {
        "converged": bool(fit.converged),
        "iterations": int(fit.iterations),
        "scaled_gradient_norm": float(fit.scaled_gradient_norm),
        "schur_condition_number": float(fit.schur_condition_number),
        "support_count_range": [
            int(fit.support_count.min()),
            int(fit.support_count.max()),
        ],
        "gradient_tolerance": float(fit.gradient_tolerance),
    }
    return model, certificate


def _held_losses(
    model: dict[str, Any],
    held_tables: dict[str, np.ndarray],
    held_records: list[dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]], int]:
    losses = []
    rows = []
    boundary_count = 0
    for record in held_records:
        sample = record["sample"]
        if sample not in held_tables:
            continue
        truth = held_tables[sample]
        support = stephenson.numerics._informative(truth).reshape(-1)
        if int(support.sum()) < stephenson.numerics.MINIMUM_INFORMATIVE_ENTITIES:
            rows.append(
                {
                    "donor": record["donor"],
                    "sample": sample,
                    "informative_pairs": int(support.sum()),
                    "status": "EXCLUDED_PAIR_SUPPORT",
                }
            )
            continue
        margins = stephenson.numerics._sample_margins(truth)
        flags: list[dict[str, Any]] = []
        prediction = stephenson._predict_method(
            model, *margins, boundary_flags=flags
        )
        loss = stephenson.numerics._donor_loss(truth, prediction, support)
        losses.append(float(loss))
        boundary_count += len(flags)
        rows.append(
            {
                "donor": record["donor"],
                "sample": sample,
                "informative_pairs": int(support.sum()),
                "loss": float(loss),
                "status": "SCORED",
            }
        )
    return np.asarray(losses, dtype=float), rows, boundary_count


def _paired_comparison(
    donors: list[str],
    primary: np.ndarray,
    comparator: np.ndarray,
    label: str,
) -> dict[str, Any]:
    first = np.asarray(primary, dtype=float)
    second = np.asarray(comparator, dtype=float)
    if first.shape != second.shape or first.shape != (len(donors),):
        raise ValueError("paired losses and donor labels differ")
    difference = first - second
    seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0, len(donors), size=(BOOTSTRAPS, len(donors)), endpoint=False
    )
    boot_difference = difference[indices].mean(axis=1)
    boot_relative = 1.0 - (
        first[indices].mean(axis=1) / second[indices].mean(axis=1)
    )
    favorable = int(np.count_nonzero(difference < 0.0))
    sign_p = sum(
        math.comb(len(donors), count)
        for count in range(favorable, len(donors) + 1)
    ) / float(2 ** len(donors))
    return {
        "donors": len(donors),
        "primary_mean_deviance_per_cell": float(first.mean()),
        "residual_mean_deviance_per_cell": float(second.mean()),
        "relative_loss_reduction": 1.0 - float(first.mean() / second.mean()),
        "relative_loss_reduction_paired_bootstrap_95_ci": np.quantile(
            boot_relative, [0.025, 0.975]
        ).tolist(),
        "paired_difference_primary_minus_residual": float(difference.mean()),
        "paired_difference_95_ci": np.quantile(
            boot_difference, [0.025, 0.975]
        ).tolist(),
        "favorable_donors": favorable,
        "exact_one_sided_sign_p": float(sign_p),
        "bootstrap_draws": BOOTSTRAPS,
        "bootstrap_seed": seed,
        "bootstrap_unit": "physical donor",
        "post_hoc_inference": True,
    }


def _lineage_analysis(
    lineage_selections: dict[str, dict[str, np.ndarray]],
    lineage_inventory: dict[str, list[dict[str, Any]]],
    records: list[dict[str, Any]],
    barcodes: np.ndarray,
    rna_values: np.ndarray,
    adt_values: np.ndarray,
    row_lookup: dict[int, int],
    development: dict[str, Any],
) -> dict[str, Any]:
    source_records = [
        row for row in records if row["role"] in {"calibration", "pilot"}
    ]
    held_records = [row for row in records if row["role"] == "held_site"]
    source_records.sort(key=lambda row: row["donor"])
    held_records.sort(key=lambda row: row["donor"])
    output = {}
    for lineage in LINEAGES:
        selections = lineage_selections[lineage]
        _, _, tables = _state_panel(
            records,
            selections,
            barcodes,
            rna_values,
            adt_values,
            row_lookup,
        )
        source_samples = [
            row["sample"] for row in source_records if row["sample"] in tables
        ]
        source_tables = np.asarray([tables[sample] for sample in source_samples])
        primary, certificate = _fit_conditional(source_tables, development)
        residual = {
            **stephenson.numerics._classical_model(
                source_tables, "deviance", False
            ),
            "alpha": 0.75,
        }
        primary_loss, primary_rows, primary_boundaries = _held_losses(
            primary, tables, held_records
        )
        residual_loss, residual_rows, residual_boundaries = _held_losses(
            residual, tables, held_records
        )
        primary_by_sample = {
            row["sample"]: row for row in primary_rows if row["status"] == "SCORED"
        }
        residual_by_sample = {
            row["sample"]: row for row in residual_rows if row["status"] == "SCORED"
        }
        scored_samples = [
            row["sample"]
            for row in held_records
            if row["sample"] in primary_by_sample
            and row["sample"] in residual_by_sample
        ]
        if len(scored_samples) != len(primary_loss) or len(scored_samples) != len(
            residual_loss
        ):
            raise AssertionError("lineage comparator support sets differ")
        donor_by_sample = {row["sample"]: row["donor"] for row in held_records}
        comparison = _paired_comparison(
            [donor_by_sample[sample] for sample in scored_samples],
            np.asarray([primary_by_sample[sample]["loss"] for sample in scored_samples]),
            np.asarray([residual_by_sample[sample]["loss"] for sample in scored_samples]),
            f"stephenson|{lineage}|primary_vs_residual|v1",
        )
        inventory = lineage_inventory[lineage]
        output[lineage] = {
            "status": "POST_HOC_LINEAGE_SENSITIVITY",
            "cell_type_labels": list(LINEAGES[lineage]),
            "cells_per_donor": LINEAGE_CELL_BUDGET,
            "candidate_budget_per_method": 1,
            "hyperparameter_selection": (
                "one frozen whole-cohort configuration per method; "
                "no lineage-specific tuning"
            ),
            "source_cambridge_donors_eligible": len(source_samples),
            "source_cambridge_eligible_by_role": {
                role: sum(
                    row["role"] == role and row["status"] == "ELIGIBLE"
                    for row in inventory
                )
                for role in ("calibration", "pilot")
            },
            "held_newcastle_donors_cell_eligible": sum(
                row["role"] == "held_site" and row["status"] == "ELIGIBLE"
                for row in inventory
            ),
            "held_newcastle_donors_scored": len(scored_samples),
            "held_newcastle_donors_pair_support_excluded": (
                sum(
                    row["role"] == "held_site" and row["status"] == "ELIGIBLE"
                    for row in inventory
                )
                - len(scored_samples)
            ),
            "source_fit_certificate": certificate,
            "primary_prediction_boundary_events": primary_boundaries,
            "residual_prediction_boundary_events": residual_boundaries,
            "comparison": comparison,
            "donor_inventory": inventory,
            "held_primary_results": primary_rows,
            "held_residual_results": residual_rows,
        }
    return output


def _permutation(
    sample: str, draw: int, cells: int = stephenson.CELL_BUDGET
) -> tuple[np.ndarray, int]:
    seed = int.from_bytes(
        hashlib.sha256(
            "\0".join((PERMUTATION_SALT, str(draw), sample)).encode()
        ).digest()[:8],
        "little",
    )
    return np.random.default_rng(seed).permutation(cells), seed


def _permuted_tables(
    source_records: list[dict[str, Any]],
    rna_states: dict[str, np.ndarray],
    adt_states: dict[str, np.ndarray],
    draw: int,
) -> tuple[np.ndarray, str, list[int]]:
    tables = []
    seeds = []
    digest = hashlib.sha256()
    for record in source_records:
        sample = record["sample"]
        permutation, seed = _permutation(sample, draw)
        seeds.append(seed)
        digest.update(sample.encode())
        digest.update(np.asarray(permutation, dtype=np.int64).tobytes())
        permuted = adt_states[sample][:, permutation]
        if not np.array_equal(
            permuted.sum(axis=1), adt_states[sample].sum(axis=1)
        ):
            raise AssertionError("destroyed-link permutation changed ADT margins")
        tables.append(_form_tables(rna_states[sample], permuted))
    return np.asarray(tables), digest.hexdigest(), seeds


def _predict_losses(
    model: dict[str, Any],
    held_records: list[dict[str, Any]],
    held_tables: dict[str, np.ndarray],
) -> np.ndarray:
    values = []
    for record in held_records:
        truth = held_tables[record["sample"]]
        support = stephenson.numerics._informative(truth).reshape(-1)
        prediction = stephenson._predict_method(
            model, *stephenson.numerics._sample_margins(truth)
        )
        values.append(
            stephenson.numerics._donor_loss(truth, prediction, support)
        )
    return np.asarray(values, dtype=float)


def _nested_bootstrap_interval(differences: np.ndarray) -> dict[str, Any]:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("nested bootstrap requires permutation by donor values")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = np.empty(BOOTSTRAPS, dtype=float)
    chunk = 250
    for left in range(0, BOOTSTRAPS, chunk):
        right = min(left + chunk, BOOTSTRAPS)
        draw_index = generator.integers(
            0, values.shape[0], size=(right - left, values.shape[0])
        )
        donor_index = generator.integers(
            0, values.shape[1], size=(right - left, values.shape[1])
        )
        for offset in range(right - left):
            bootstrap[left + offset] = values[
                np.ix_(draw_index[offset], donor_index[offset])
            ].mean()
    return {
        "mean_intact_minus_destroyed": float(values.mean()),
        "nested_permutation_donor_bootstrap_95_ci": np.quantile(
            bootstrap, [0.025, 0.975]
        ).tolist(),
        "bootstrap_draws": BOOTSTRAPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "resampled_axes": ["fixed independent source permutation", "held donor"],
    }


def _destroyed_link_analysis(
    full_selections: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    barcodes: np.ndarray,
    rna_values: np.ndarray,
    adt_values: np.ndarray,
    row_lookup: dict[int, int],
    development: dict[str, Any],
) -> dict[str, Any]:
    source_records = [
        row for row in records if row["role"] in {"calibration", "pilot"}
    ]
    held_records = [row for row in records if row["role"] == "held_site"]
    source_records.sort(key=lambda row: row["donor"])
    held_records.sort(key=lambda row: row["donor"])
    rna_states, adt_states, tables = _state_panel(
        records,
        full_selections,
        barcodes,
        rna_values,
        adt_values,
        row_lookup,
    )
    held_truth = {row["sample"]: tables[row["sample"]] for row in held_records}
    frozen_predictions = json.loads(stephenson.DEFAULT_PREDICTION.read_text())
    prediction_by_sample = {
        row["sample"]: row["predictions"]["primary"]
        for row in frozen_predictions["samples"]
    }
    intact_losses = []
    for record in held_records:
        truth = held_truth[record["sample"]]
        prediction = np.asarray(
            prediction_by_sample[record["sample"]], dtype=float
        ).reshape(len(stephenson.MARKERS), len(stephenson.MARKERS), 2, 2)
        intact_losses.append(
            stephenson.numerics._donor_loss(
                truth,
                prediction,
                stephenson.numerics._informative(truth).reshape(-1),
            )
        )
    intact = np.asarray(intact_losses, dtype=float)
    frozen_score = json.loads(stephenson.DEFAULT_SCORE.read_text())
    frozen_intact = np.asarray(
        [row["losses"]["primary"] for row in frozen_score["donor_results"]]
    )
    intact_replay_error = float(np.max(np.abs(intact - frozen_intact)))
    if intact_replay_error > 1e-12:
        raise PermissionError("intact held loss does not replay the frozen result")

    original_tables = []
    for record in source_records:
        sample = record["sample"]
        destroyed = stephenson._destroyed_adt(
            adt_states[sample], barcodes[full_selections[sample]], sample
        )
        original_tables.append(_form_tables(rna_states[sample], destroyed))
    original_model, original_certificate = _fit_conditional(
        np.asarray(original_tables), development
    )
    frozen_destroyed_coordinate = np.asarray(
        development["frozen_source_models"]["destroyed_link"]["source_coordinate"]
    )
    original_coordinate_error = float(
        np.max(
            np.abs(
                np.asarray(original_model["source_coordinate"])
                - frozen_destroyed_coordinate
            )
        )
    )
    if original_coordinate_error > 1e-12:
        raise PermissionError("prespecified destroyed source fit does not replay")
    original_loss = _predict_losses(original_model, held_records, held_truth)
    frozen_original = np.asarray(
        [row["losses"]["destroyed_link"] for row in frozen_score["donor_results"]]
    )
    original_loss_error = float(np.max(np.abs(original_loss - frozen_original)))
    if original_loss_error > 1e-12:
        raise PermissionError("prespecified destroyed held loss does not replay")

    draws = []
    successful_losses = []
    for draw in range(INDEPENDENT_PERMUTATIONS):
        source_tables, permutation_sha256, seeds = _permuted_tables(
            source_records, rna_states, adt_states, draw
        )
        try:
            model, certificate = _fit_conditional(
                source_tables, development, tolerance=1e-7
            )
        except stephenson.CouplingEstimationRefusal as error:
            draws.append(
                {
                    "draw": draw,
                    "status": "REFUSED_NUMERICAL_CERTIFICATE",
                    "reason": str(error),
                    "permutation_sha256": permutation_sha256,
                    "per_donor_seed_sha256": _array_sha256(
                        np.asarray(seeds, dtype=np.uint64)
                    ),
                }
            )
            continue
        losses = _predict_losses(model, held_records, held_truth)
        successful_losses.append(losses)
        difference = intact - losses
        draws.append(
            {
                "draw": draw,
                "status": "EVALUATED",
                "permutation_sha256": permutation_sha256,
                "per_donor_seed_sha256": _array_sha256(
                    np.asarray(seeds, dtype=np.uint64)
                ),
                "mean_destroyed_loss": float(losses.mean()),
                "mean_intact_minus_destroyed": float(difference.mean()),
                "relative_loss_reduction_intact_vs_destroyed": (
                    1.0 - float(intact.mean() / losses.mean())
                ),
                "favorable_held_donors": int(np.count_nonzero(difference < 0.0)),
                "held_destroyed_losses": {
                    record["donor"]: float(value)
                    for record, value in zip(held_records, losses)
                },
                "fit_certificate": certificate,
            }
        )
    if len({row["permutation_sha256"] for row in draws}) != len(draws):
        raise AssertionError("independent destroyed-link permutations repeat")
    if len(successful_losses) < 20:
        raise RuntimeError("fewer than 20 independent permutations were evaluable")
    draw_losses = np.asarray(successful_losses, dtype=float)
    difference = intact[None, :] - draw_losses
    mean_destroyed = draw_losses.mean(axis=1)
    mean_difference = difference.mean(axis=1)
    relative = 1.0 - intact.mean() / mean_destroyed
    empirical = {
        "independent_permutations_attempted": INDEPENDENT_PERMUTATIONS,
        "independent_permutations_evaluated": len(successful_losses),
        "independent_permutations_refused": (
            INDEPENDENT_PERMUTATIONS - len(successful_losses)
        ),
        "mean_destroyed_loss_quantiles_2_5_50_97_5": np.quantile(
            mean_destroyed, [0.025, 0.5, 0.975]
        ).tolist(),
        "mean_intact_minus_destroyed_quantiles_2_5_50_97_5": np.quantile(
            mean_difference, [0.025, 0.5, 0.975]
        ).tolist(),
        "relative_loss_reduction_quantiles_2_5_50_97_5": np.quantile(
            relative, [0.025, 0.5, 0.975]
        ).tolist(),
        "permutations_with_lower_intact_mean_loss": int(
            np.count_nonzero(mean_difference < 0.0)
        ),
        "original_destroyed_mean_loss_percentile_among_independent_draws": float(
            100.0 * np.mean(mean_destroyed <= original_loss.mean())
        ),
    }
    mean_control = draw_losses.mean(axis=0)
    donor_ids = [row["donor"] for row in held_records]
    return {
        "schema": "stephenson-destroyed-link-permutation-robustness/1.0",
        "status": "POST_HOC_ROBUSTNESS_COMPLETE",
        "created_at_utc": _timestamp(),
        "confirmatory": False,
        "source_site": "Cambridge",
        "held_site": "Newcastle",
        "source_donors": len(source_records),
        "held_donors": len(held_records),
        "cells_per_donor": stephenson.CELL_BUDGET,
        "permutation_unit": "within-source-donor whole-cell ADT vector",
        "permutation_invariant": (
            "each draw preserves every source donor's RNA and ADT margins "
            "and the full within-ADT multivariate vector"
        ),
        "independent_fit_gradient_tolerance": 1e-7,
        "prespecified_fit_gradient_tolerance": float(
            _frozen_configuration(development)["gradient_tolerance"]
        ),
        "intact_mean_loss": float(intact.mean()),
        "intact_held_losses": {
            donor: float(value) for donor, value in zip(donor_ids, intact)
        },
        "prespecified_v1_permutation": {
            "mean_destroyed_loss": float(original_loss.mean()),
            "mean_intact_minus_destroyed": float(
                (intact - original_loss).mean()
            ),
            "relative_loss_reduction_intact_vs_destroyed": (
                1.0 - float(intact.mean() / original_loss.mean())
            ),
            "favorable_held_donors": int(
                np.count_nonzero(intact - original_loss < 0.0)
            ),
            "held_destroyed_losses": {
                donor: float(value)
                for donor, value in zip(donor_ids, original_loss)
            },
            "fit_certificate": original_certificate,
        },
        "independent_draws": draws,
        "independent_draw_distribution": empirical,
        "intact_vs_mean_independent_destroyed_control": _paired_comparison(
            donor_ids,
            intact,
            mean_control,
            "stephenson|intact_vs_mean_independent_destroyed|v1",
        ),
        "nested_uncertainty": _nested_bootstrap_interval(difference),
        "replay_checks": {
            "maximum_absolute_intact_held_loss_error": intact_replay_error,
            "maximum_absolute_prespecified_source_coordinate_error": (
                original_coordinate_error
            ),
            "maximum_absolute_prespecified_held_loss_error": original_loss_error,
        },
    }


def _lineage_tsv(payload: dict[str, Any]) -> str:
    stream = io.StringIO()
    fields = (
        "lineage",
        "source_cambridge_donors",
        "held_newcastle_cell_eligible",
        "held_newcastle_scored",
        "primary_mean_loss",
        "residual_mean_loss",
        "relative_loss_reduction",
        "relative_ci_low",
        "relative_ci_high",
        "difference_ci_low",
        "difference_ci_high",
        "favorable_donors",
        "exact_one_sided_sign_p",
        "status",
    )
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for lineage, result in payload["lineages"].items():
        comparison = result["comparison"]
        relative_ci = comparison[
            "relative_loss_reduction_paired_bootstrap_95_ci"
        ]
        difference_ci = comparison["paired_difference_95_ci"]
        writer.writerow(
            {
                "lineage": lineage,
                "source_cambridge_donors": result[
                    "source_cambridge_donors_eligible"
                ],
                "held_newcastle_cell_eligible": result[
                    "held_newcastle_donors_cell_eligible"
                ],
                "held_newcastle_scored": result[
                    "held_newcastle_donors_scored"
                ],
                "primary_mean_loss": f"{comparison['primary_mean_deviance_per_cell']:.12g}",
                "residual_mean_loss": f"{comparison['residual_mean_deviance_per_cell']:.12g}",
                "relative_loss_reduction": f"{comparison['relative_loss_reduction']:.12g}",
                "relative_ci_low": f"{relative_ci[0]:.12g}",
                "relative_ci_high": f"{relative_ci[1]:.12g}",
                "difference_ci_low": f"{difference_ci[0]:.12g}",
                "difference_ci_high": f"{difference_ci[1]:.12g}",
                "favorable_donors": comparison["favorable_donors"],
                "exact_one_sided_sign_p": f"{comparison['exact_one_sided_sign_p']:.12g}",
                "status": "post_hoc_nonconfirmatory",
            }
        )
    return stream.getvalue()


def _permutation_tsv(payload: dict[str, Any]) -> str:
    stream = io.StringIO()
    fields = (
        "draw",
        "status",
        "draw_type",
        "mean_destroyed_loss",
        "mean_intact_minus_destroyed",
        "relative_loss_reduction_intact_vs_destroyed",
        "favorable_held_donors",
        "permutation_sha256",
    )
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    original = payload["prespecified_v1_permutation"]
    writer.writerow(
        {
            "draw": "prespecified_v1",
            "status": "EVALUATED",
            "draw_type": "prespecified_original",
            "mean_destroyed_loss": f"{original['mean_destroyed_loss']:.12g}",
            "mean_intact_minus_destroyed": f"{original['mean_intact_minus_destroyed']:.12g}",
            "relative_loss_reduction_intact_vs_destroyed": f"{original['relative_loss_reduction_intact_vs_destroyed']:.12g}",
            "favorable_held_donors": original["favorable_held_donors"],
            "permutation_sha256": "prespecified_v1",
        }
    )
    for row in payload["independent_draws"]:
        if row["status"] != "EVALUATED":
            writer.writerow(
                {
                    "draw": row["draw"],
                    "status": row["status"],
                    "draw_type": "fixed_independent",
                    "permutation_sha256": row["permutation_sha256"],
                }
            )
            continue
        writer.writerow(
            {
                "draw": row["draw"],
                "status": row["status"],
                "draw_type": "fixed_independent",
                "mean_destroyed_loss": f"{row['mean_destroyed_loss']:.12g}",
                "mean_intact_minus_destroyed": f"{row['mean_intact_minus_destroyed']:.12g}",
                "relative_loss_reduction_intact_vs_destroyed": f"{row['relative_loss_reduction_intact_vs_destroyed']:.12g}",
                "favorable_held_donors": row["favorable_held_donors"],
                "permutation_sha256": row["permutation_sha256"],
            }
        )
    return stream.getvalue()


def run(
    h5ad: Path,
    lineage_json: Path,
    lineage_tsv: Path,
    permutation_json: Path,
    permutation_tsv: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for path in (lineage_json, lineage_tsv, permutation_json, permutation_tsv):
        if path.exists():
            raise FileExistsError(path)
    source = _source(h5ad)
    records = [
        row
        for row in source["records"]
        if row["role"] in {"calibration", "pilot", "held_site"}
    ]
    barcodes, sample_ids, sites, cell_types = _observation_metadata(h5ad)
    full = stephenson._selected_rows(h5ad, records)
    full_selections = {sample: row["rows"] for sample, row in full.items()}
    lineage_selections = {}
    lineage_inventory = {}
    for lineage, labels in LINEAGES.items():
        selections, inventory = _lineage_selections(
            records,
            barcodes,
            sample_ids,
            sites,
            cell_types,
            lineage,
            labels,
        )
        lineage_selections[lineage] = selections
        lineage_inventory[lineage] = inventory
    rna_values, adt_values, row_lookup = _read_union_counts(
        h5ad, [full_selections, *lineage_selections.values()]
    )
    development = json.loads(stephenson.DEFAULT_DEVELOPMENT.read_text())
    lineages = _lineage_analysis(
        lineage_selections,
        lineage_inventory,
        records,
        barcodes,
        rna_values,
        adt_values,
        row_lookup,
        development,
    )
    provenance = {
        "source_manifest_sha256": _sha256(stephenson.DEFAULT_SOURCE),
        "sealed_h5ad_sha256": source["payload"]["h5ad"]["sha256"],
        "frozen_development_sha256": _sha256(stephenson.DEFAULT_DEVELOPMENT),
        "frozen_prediction_sha256": _sha256(stephenson.DEFAULT_PREDICTION),
        "frozen_confirmation_sha256": _sha256(stephenson.DEFAULT_SCORE),
        "script_sha256": _sha256(Path(__file__)),
        "numeric_matrix_rows_read": len(row_lookup),
        "numeric_matrix_columns_read": 2 * len(stephenson.MARKERS),
        "full_h5ad_matrix_loaded": False,
        "cell_vectors_serialized": False,
    }
    lineage_payload = {
        "schema": "stephenson-major-lineage-sensitivity/1.0",
        "status": "POST_HOC_ROBUSTNESS_COMPLETE",
        "created_at_utc": _timestamp(),
        "confirmatory": False,
        "source_site": "Cambridge",
        "held_site": "Newcastle",
        "donor_split": "frozen 12 calibration plus 24 pilot source donors versus 56 held-site donors",
        "common_cell_budget": LINEAGE_CELL_BUDGET,
        "lineages_requested": list(LINEAGES),
        "lineages_reported": list(lineages),
        "fixed_primary_configuration": _frozen_configuration(development),
        "fixed_residual_configuration": {
            "family": "signed-root Poisson deviance",
            "centered": False,
            "transport_alpha": 0.75,
        },
        "lineages": lineages,
        "provenance": provenance,
    }
    permutation_payload = _destroyed_link_analysis(
        full_selections,
        records,
        barcodes,
        rna_values,
        adt_values,
        row_lookup,
        development,
    )
    permutation_payload["provenance"] = provenance
    _write_json(lineage_json, lineage_payload)
    lineage_tsv.parent.mkdir(parents=True, exist_ok=True)
    lineage_tsv.write_text(_lineage_tsv(lineage_payload))
    _write_json(permutation_json, permutation_payload)
    permutation_tsv.parent.mkdir(parents=True, exist_ok=True)
    permutation_tsv.write_text(_permutation_tsv(permutation_payload))
    return lineage_payload, permutation_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--lineage-json", type=Path, default=DEFAULT_LINEAGE_JSON)
    parser.add_argument("--lineage-tsv", type=Path, default=DEFAULT_LINEAGE_TSV)
    parser.add_argument("--permutation-json", type=Path, default=DEFAULT_PERMUTATION_JSON)
    parser.add_argument("--permutation-tsv", type=Path, default=DEFAULT_PERMUTATION_TSV)
    args = parser.parse_args()
    run(
        args.h5ad,
        args.lineage_json,
        args.lineage_tsv,
        args.permutation_json,
        args.permutation_tsv,
    )


if __name__ == "__main__":
    main()
