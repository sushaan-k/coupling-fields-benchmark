"""Held-technical-date PerturbFate conditional-field benchmark.

The four deposited dates are technical units. They are used only to test
technical reproducibility and are never relabelled as biological replicates.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from mapreg.coupling_fields import (
    conditional_association_coordinates,
    fit_structured_coupling_fields,
    helmert_contrast,
    inverse_permutation_variance_weights,
    normalized_hypergraph_laplacian,
)

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "results/development/perturbfate_paired_state_features.csv.gz"
ENCODER_PATH = ROOT / "results/development/perturbfate_ntc_frozen_state_encoder.rds"
SOURCE_MANIFEST = ROOT / "data/development/perturbfate_gse291147/manifest.json"
SCGPT_EMBEDDING = ROOT / "data/scgpt_gene_embeddings.npz"
OUTPUT = ROOT / "results/development/perturbfate_conditional_fields.json"

CONTROL = "NO-TARGET"
CONDITIONS = ("DMSO_treated_cells", "PLX_treated_cells")
STATE_COUNT = 3
MINIMUM_ARM_CELLS = 20
PERMUTATIONS = 64
PSEUDOCOUNT = 0.5
NUCLEAR_FRACTION = 0.1
GRAPH_PENALTY = 5.0
RIDGE_PENALTY = 0.1
HYPERGRAPH_NEIGHBORS = 5
BOOTSTRAPS = 2_000
SEED = 291_147


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_states(path: Path) -> dict[str, np.ndarray]:
    columns = {
        "target": [],
        "condition": [],
        "technical_date": [],
        "old_state": [],
        "new_state": [],
    }
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            columns["target"].append(row["target"])
            columns["condition"].append(row["condition"])
            columns["technical_date"].append(row["technical_date"])
            columns["old_state"].append(int(row["old_state"]) - 1)
            columns["new_state"].append(int(row["new_state"]) - 1)
    return {
        "target": np.asarray(columns["target"]),
        "condition": np.asarray(columns["condition"]),
        "technical_date": np.asarray(columns["technical_date"]),
        "first": np.asarray(columns["old_state"], dtype=int),
        "second": np.asarray(columns["new_state"], dtype=int),
    }


def _arm_statistics(
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray | int]:
    indices = np.flatnonzero(mask)
    if len(indices) < MINIMUM_ARM_CELLS:
        raise ValueError("each target/date/treatment arm needs at least 20 cells")
    left = first[indices]
    right = second[indices]
    table = np.bincount(left * STATE_COUNT + right, minlength=STATE_COUNT**2).reshape(
        STATE_COUNT, STATE_COUNT
    )
    probability = table / table.sum()
    basis = helmert_contrast(STATE_COUNT)
    centered = probability - np.outer(probability.sum(axis=1), probability.sum(axis=0))
    estimate = conditional_association_coordinates(
        left,
        right,
        first_levels=STATE_COUNT,
        second_levels=STATE_COUNT,
        pseudocount=PSEUDOCOUNT,
        permutations=PERMUTATIONS,
        seed=seed,
    )
    return {
        "field": estimate.coordinates.ravel(),
        "variance": estimate.null_variance_coordinates.ravel(),
        "destroyed": estimate.destroyed_coordinates.ravel(),
        "covariance": (basis.T @ centered @ basis).ravel(),
        "endpoint": np.concatenate(
            (
                basis.T @ probability.sum(axis=1),
                basis.T @ probability.sum(axis=0),
            )
        ),
        "cells": len(indices),
    }


def _factorial_contrast(
    target_vehicle: np.ndarray,
    target_challenge: np.ndarray,
    control_vehicle: np.ndarray,
    control_challenge: np.ndarray,
) -> np.ndarray:
    return target_challenge - target_vehicle - control_challenge + control_vehicle


def _eligible_targets(
    data: dict[str, np.ndarray], dates: list[str]
) -> tuple[list[str], list[dict[str, object]]]:
    targets = sorted(set(data["target"]) - {CONTROL})
    retained = []
    attrition = []
    for name in targets:
        counts = []
        for date in dates:
            for condition in CONDITIONS:
                counts.append(
                    int(
                        np.count_nonzero(
                            (data["target"] == name)
                            & (data["technical_date"] == date)
                            & (data["condition"] == condition)
                        )
                    )
                )
        passed = min(counts) >= MINIMUM_ARM_CELLS
        attrition.append(
            {
                "target": name,
                "retained": passed,
                "date_by_treatment_cells": counts,
            }
        )
        if passed:
            retained.append(name)
    return retained, attrition


def _build_arrays(
    data: dict[str, np.ndarray], dates: list[str], targets: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    coordinate_count = (STATE_COUNT - 1) ** 2
    arrays = {
        name: np.empty((len(dates), len(targets), coordinate_count), dtype=float)
        for name in ("field", "variance", "destroyed", "covariance", "endpoint")
    }
    target_support = np.empty((len(dates), len(targets), len(CONDITIONS)), dtype=int)
    control_support = np.empty((len(dates), len(CONDITIONS)), dtype=int)
    statistic_names = ("field", "destroyed", "covariance", "endpoint")
    for date_index, date in enumerate(dates):
        control_statistics = []
        for condition_index, condition in enumerate(CONDITIONS):
            mask = (
                (data["target"] == CONTROL)
                & (data["technical_date"] == date)
                & (data["condition"] == condition)
            )
            statistics = _arm_statistics(
                data["first"],
                data["second"],
                mask,
                SEED + 100_000 * date_index + 10_000 * condition_index,
            )
            control_statistics.append(statistics)
            control_support[date_index, condition_index] = int(statistics["cells"])
        for target_index, target in enumerate(targets):
            target_statistics = []
            for condition_index, condition in enumerate(CONDITIONS):
                mask = (
                    (data["target"] == target)
                    & (data["technical_date"] == date)
                    & (data["condition"] == condition)
                )
                statistics = _arm_statistics(
                    data["first"],
                    data["second"],
                    mask,
                    SEED
                    + 100_000 * date_index
                    + 1_000 * (target_index + 1)
                    + 10 * condition_index,
                )
                target_statistics.append(statistics)
                target_support[date_index, target_index, condition_index] = int(
                    statistics["cells"]
                )
            for name in statistic_names:
                arrays[name][date_index, target_index] = _factorial_contrast(
                    np.asarray(target_statistics[0][name]),
                    np.asarray(target_statistics[1][name]),
                    np.asarray(control_statistics[0][name]),
                    np.asarray(control_statistics[1][name]),
                )
            arrays["variance"][date_index, target_index] = sum(
                np.asarray(statistics["variance"])
                for statistics in (*target_statistics, *control_statistics)
            )
    return arrays, {
        "target": target_support,
        "control": control_support,
    }


def _embedding_laplacian(
    targets: list[str], path: Path
) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as archive:
        names = archive["gene_names"].astype(str)
        embedding = np.asarray(archive["embedding"], dtype=float)
    lookup = {name.upper(): index for index, name in enumerate(names)}
    covered = [index for index, name in enumerate(targets) if name.upper() in lookup]
    incidence = np.zeros((len(targets), len(covered)), dtype=float)
    if covered:
        values = np.asarray(
            [embedding[lookup[targets[index].upper()]] for index in covered]
        )
        norm = np.linalg.norm(values, axis=1)
        values = np.divide(
            values,
            norm[:, None],
            out=np.zeros_like(values),
            where=norm[:, None] > 0.0,
        )
        similarity = values @ values.T
        if not np.isfinite(similarity).all():
            raise FloatingPointError("scGPT similarities are non-finite")
        for column, row in enumerate(range(len(covered))):
            order = np.argsort(similarity[row])[::-1]
            members = [
                covered[index]
                for index in order[: min(HYPERGRAPH_NEIGHBORS + 1, len(order))]
            ]
            incidence[members, column] = 1.0
        laplacian = normalized_hypergraph_laplacian(incidence)
    else:
        laplacian = np.zeros((len(targets), len(targets)), dtype=float)
    return laplacian, {
        "embedding": str(path.relative_to(ROOT)),
        "embedding_sha256": _sha256(path),
        "covered_targets": len(covered),
        "total_targets": len(targets),
        "hyperedges": int(incidence.shape[1]),
        "neighbors_excluding_self": HYPERGRAPH_NEIGHBORS,
    }


def _structured(
    values: np.ndarray,
    *,
    laplacian: np.ndarray | None = None,
    observation_weight: np.ndarray | None = None,
    nuclear: bool,
    graph: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    singular_maximum = float(np.linalg.svd(values, compute_uv=False)[0])
    fit = fit_structured_coupling_fields(
        values,
        observation_weight=observation_weight,
        graph_laplacian=laplacian if graph else None,
        nuclear_penalty=NUCLEAR_FRACTION * singular_maximum if nuclear else 0.0,
        graph_penalty=GRAPH_PENALTY if graph else 0.0,
        tolerance=1e-9,
    )
    return fit.coefficient, {
        "converged": fit.converged,
        "iterations": fit.iterations,
        "effective_rank": fit.effective_rank,
        "nuclear_penalty": fit.nuclear_penalty,
        "graph_penalty": fit.graph_penalty,
    }


def _metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float | None]:
    estimate_flat = estimate.ravel()
    truth_flat = truth.ravel()
    correlation = (
        float(np.corrcoef(estimate_flat, truth_flat)[0, 1])
        if np.std(estimate_flat) > 0.0 and np.std(truth_flat) > 0.0
        else None
    )
    truth_scale = max(float(np.mean(truth_flat**2)), 1e-12)
    estimate_norm = np.linalg.norm(estimate, axis=-1)
    truth_norm = np.linalg.norm(truth, axis=-1)
    cosine = np.divide(
        np.sum(estimate * truth, axis=-1),
        estimate_norm * truth_norm,
        out=np.zeros_like(estimate_norm),
        where=estimate_norm * truth_norm > 0.0,
    )
    return {
        "pooled_pearson": correlation,
        "standardized_rmse": float(
            np.sqrt(np.mean((estimate_flat - truth_flat) ** 2) / truth_scale)
        ),
        "macro_target_cosine": float(cosine.mean()),
    }


def _bootstrap_metrics(
    estimate: np.ndarray,
    truth: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict[str, list[float] | None]:
    samples = {name: [] for name in _metrics(estimate, truth)}
    for indices in bootstrap_indices:
        metrics = _metrics(estimate[:, indices], truth[:, indices])
        for name, value in metrics.items():
            if value is not None:
                samples[name].append(value)
    return {
        name: (
            [float(value) for value in np.quantile(values, [0.025, 0.975])]
            if values
            else None
        )
        for name, values in samples.items()
    }


def _bootstrap_mse_difference(
    first: np.ndarray,
    second: np.ndarray,
    truth: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict[str, float | list[float]]:
    difference = np.mean((first - truth) ** 2, axis=(0, 2)) - np.mean(
        (second - truth) ** 2, axis=(0, 2)
    )
    draws = difference[bootstrap_indices].mean(axis=1)
    return {
        "mean_first_minus_second_target_mse": float(difference.mean()),
        "target_bootstrap_95_ci": [
            float(value) for value in np.quantile(draws, [0.025, 0.975])
        ],
    }


def _support_records(
    support: dict[str, np.ndarray], dates: list[str], targets: list[str]
) -> dict[str, list[dict[str, object]]]:
    target_rows = []
    for date_index, date in enumerate(dates):
        for target_index, target in enumerate(targets):
            for condition_index, condition in enumerate(CONDITIONS):
                target_rows.append(
                    {
                        "technical_date": date,
                        "target": target,
                        "condition": condition,
                        "cells": int(
                            support["target"][date_index, target_index, condition_index]
                        ),
                    }
                )
    control_rows = []
    for date_index, date in enumerate(dates):
        for condition_index, condition in enumerate(CONDITIONS):
            control_rows.append(
                {
                    "technical_date": date,
                    "target": CONTROL,
                    "condition": condition,
                    "cells": int(support["control"][date_index, condition_index]),
                }
            )
    return {"target_arms": target_rows, "control_arms": control_rows}


def run(path: Path, output: Path) -> dict[str, object]:
    data = _read_states(path)
    dates = sorted(set(data["technical_date"]))
    if len(dates) != 4:
        raise ValueError("PerturbFate benchmark requires exactly four technical dates")
    targets, attrition = _eligible_targets(data, dates)
    if not targets:
        raise ValueError("no targets meet the frozen support rule")
    arrays, support = _build_arrays(data, dates, targets)
    for name, values in arrays.items():
        if not np.isfinite(values).all():
            raise FloatingPointError(f"{name} contains non-finite values")
    if np.any(arrays["variance"] <= 0.0):
        raise FloatingPointError("factorial permutation variance must be positive")

    laplacian, graph = _embedding_laplacian(targets, SCGPT_EMBEDDING)
    permutation = np.random.default_rng(SEED + 700_000).permutation(len(targets))
    shuffled_laplacian = laplacian[np.ix_(permutation, permutation)]
    method_names = (
        "zero",
        "direct_mean_remaining_dates",
        "nuclear_only",
        "graph_only",
        "variance_weighted_nuclear_scgpt_hypergraph",
        "endpoint_margin_ridge",
        "linear_cross_covariance",
        "shuffled_hypergraph",
        "destroyed_links",
    )
    predictions: dict[str, list[np.ndarray]] = {name: [] for name in method_names}
    truths = []
    fold_fits = []
    for held in range(len(dates)):
        training = [index for index in range(len(dates)) if index != held]
        direct = arrays["field"][training].mean(axis=0)
        mean_variance = arrays["variance"][training].sum(axis=0) / len(training) ** 2
        weights = inverse_permutation_variance_weights(mean_variance)
        truths.append(arrays["field"][held])
        predictions["zero"].append(np.zeros_like(direct))
        predictions["direct_mean_remaining_dates"].append(direct)
        nuclear, nuclear_fit = _structured(direct, nuclear=True, graph=False)
        predictions["nuclear_only"].append(nuclear)
        graph_only, graph_fit = _structured(
            direct, laplacian=laplacian, nuclear=False, graph=True
        )
        predictions["graph_only"].append(graph_only)
        final, final_fit = _structured(
            direct,
            laplacian=laplacian,
            observation_weight=weights,
            nuclear=True,
            graph=True,
        )
        predictions["variance_weighted_nuclear_scgpt_hypergraph"].append(final)
        model = Ridge(alpha=RIDGE_PENALTY).fit(
            arrays["endpoint"][training].reshape(-1, arrays["endpoint"].shape[-1]),
            arrays["field"][training].reshape(-1, arrays["field"].shape[-1]),
        )
        predictions["endpoint_margin_ridge"].append(
            model.predict(arrays["endpoint"][held])
        )
        predictions["linear_cross_covariance"].append(
            arrays["covariance"][training].mean(axis=0)
        )
        shuffled, shuffled_fit = _structured(
            direct,
            laplacian=shuffled_laplacian,
            observation_weight=weights,
            nuclear=True,
            graph=True,
        )
        predictions["shuffled_hypergraph"].append(shuffled)
        predictions["destroyed_links"].append(
            arrays["destroyed"][training].mean(axis=0)
        )
        fold_fits.append(
            {
                "held_technical_date": dates[held],
                "nuclear_only": nuclear_fit,
                "graph_only": graph_fit,
                "final_weighted_structured": final_fit,
                "shuffled_hypergraph": shuffled_fit,
            }
        )

    truth = np.asarray(truths)
    prediction_arrays = {
        name: np.asarray(values) for name, values in predictions.items()
    }
    generator = np.random.default_rng(SEED + 800_000)
    bootstrap_indices = generator.integers(
        0, len(targets), size=(BOOTSTRAPS, len(targets))
    )
    methods = {}
    for name, estimate in prediction_arrays.items():
        methods[name] = {
            "metrics": _metrics(estimate, truth),
            "target_bootstrap_95_ci": _bootstrap_metrics(
                estimate, truth, bootstrap_indices
            ),
            "held_date_metrics": {
                date: _metrics(estimate[index], truth[index])
                for index, date in enumerate(dates)
            },
        }

    primary = "variance_weighted_nuclear_scgpt_hypergraph"
    comparisons = {
        name: _bootstrap_mse_difference(
            prediction_arrays[primary], estimate, truth, bootstrap_indices
        )
        for name, estimate in prediction_arrays.items()
        if name != primary
    }
    direct_vs_destroyed = _bootstrap_mse_difference(
        prediction_arrays["direct_mean_remaining_dates"],
        prediction_arrays["destroyed_links"],
        truth,
        bootstrap_indices,
    )
    direct_correlation_interval = methods["direct_mean_remaining_dates"][
        "target_bootstrap_95_ci"
    ]["pooled_pearson"]
    primary_intervals = methods[primary]["target_bootstrap_95_ci"]
    primary_metrics = methods[primary]["metrics"]
    pairing_positive = bool(
        direct_correlation_interval is not None
        and direct_correlation_interval[0] > 0.0
        and direct_vs_destroyed["target_bootstrap_95_ci"][1] < 0.0
    )
    estimator_comparators = (
        "zero",
        "direct_mean_remaining_dates",
        "nuclear_only",
        "graph_only",
        "endpoint_margin_ridge",
        "linear_cross_covariance",
        "shuffled_hypergraph",
        "destroyed_links",
    )
    estimator_positive = bool(
        primary_intervals["pooled_pearson"] is not None
        and primary_intervals["pooled_pearson"][0] > 0.0
        and primary_metrics["standardized_rmse"] < 1.0
        and primary_intervals["standardized_rmse"][1] < 1.0
        and all(
            comparisons[name]["target_bootstrap_95_ci"][1] < 0.0
            for name in estimator_comparators
        )
    )

    result = {
        "schema": "v2p2r.perturbfate-conditional-fields/1",
        "panel": "GSE291147 PerturbFate pre-existing RNA--nascent RNA",
        "observable": (
            "same-cell conditional association between pre-existing and one-hour "
            "5-EU-labelled RNA states at the common day-6 endpoint"
        ),
        "factorial_contrast": (
            "(target PLX - target DMSO) - (NO-TARGET PLX - NO-TARGET DMSO)"
        ),
        "technical_dates": dates,
        "unit_type": "deposited technical date",
        "biological_replicate_unit": None,
        "targets": len(targets),
        "target_names": targets,
        "field_coordinates": int(truth.shape[-1]),
        "analysis_cells": len(data["target"]),
        "minimum_target_arm_cells": int(support["target"].min()),
        "median_target_arm_cells": float(np.median(support["target"])),
        "minimum_control_arm_cells": int(support["control"].min()),
        "methods": methods,
        "primary": primary,
        "primary_minus_comparator_target_mse": comparisons,
        "pairing_signal_comparison": {
            "comparison": "paired direct mean versus source destroyed links",
            **direct_vs_destroyed,
        },
        "decisions": {
            "pairing_specific_technical_reproducibility": (
                "POSITIVE" if pairing_positive else "REFUSE"
            ),
            "estimator_superiority": "POSITIVE" if estimator_positive else "REFUSE",
            "biological_replication": "REFUSE",
            "biological_replication_reason": (
                "The four deposited dates are technical partitions; no public "
                "biological-screen label is available."
            ),
        },
        "decision_rules_fixed_before_execution": {
            "pairing_specific_technical_reproducibility": (
                "lower 95% target-bootstrap bound for direct held-date correlation "
                "> 0 and upper 95% bound for direct-minus-destroyed target MSE < 0"
            ),
            "estimator_superiority": (
                "lower 95% target-bootstrap bound for primary correlation > 0; "
                "primary standardized RMSE and its upper 95% bound < 1; and the "
                "upper 95% bound for primary-minus-comparator target MSE < 0 for "
                "every declared comparator"
            ),
            "biological_replication": (
                "requires an independently executed biological screen and therefore "
                "cannot pass on technical dates"
            ),
        },
        "fixed_settings": {
            "state_labels": "frozen NTC-trained three-state labels supplied in input",
            "minimum_cells_per_target_date_treatment_arm": MINIMUM_ARM_CELLS,
            "permutations": PERMUTATIONS,
            "permutation_scope": "unrestricted within each target/date/treatment arm",
            "pseudocount": PSEUDOCOUNT,
            "factorial_variance": "sum of the four arm permutation variances",
            "training_date_mean_variance": "sum of three date variances divided by 9",
            "weighting": "clipped median-normalized inverse permutation variance",
            "nuclear_fraction_of_leading_singular_value": NUCLEAR_FRACTION,
            "scgpt_hypergraph_penalty": GRAPH_PENALTY,
            "scgpt_neighbors_excluding_self": HYPERGRAPH_NEIGHBORS,
            "endpoint_margin_ridge_alpha": RIDGE_PENALTY,
            "target_bootstraps": BOOTSTRAPS,
            "seed": SEED,
        },
        "scgpt_hypergraph": graph,
        "shuffled_hypergraph_target_order": [targets[index] for index in permutation],
        "fold_fits": fold_fits,
        "support": _support_records(support, dates, targets),
        "attrition": attrition,
        "provenance": {
            "state_table": str(path.relative_to(ROOT)),
            "state_table_sha256": _sha256(path),
            "frozen_state_encoder": str(ENCODER_PATH.relative_to(ROOT)),
            "frozen_state_encoder_sha256": _sha256(ENCODER_PATH),
            "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
            "source_manifest_sha256": _sha256(SOURCE_MANIFEST),
            "preprocessing_script": (
                "experiments/development/prepare_perturbfate_route_states.R"
            ),
            "preprocessing_script_sha256": _sha256(
                ROOT / "experiments/development/prepare_perturbfate_route_states.R"
            ),
            "estimator": "mapreg/coupling_fields.py",
            "estimator_sha256": _sha256(ROOT / "mapreg/coupling_fields.py"),
            "benchmark": str(Path(__file__).relative_to(ROOT)),
            "benchmark_sha256": _sha256(Path(__file__)),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, default=STATE_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        result = run(arguments.states, arguments.output)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "targets": result["targets"],
                "primary": result["primary"],
                "decisions": result["decisions"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
