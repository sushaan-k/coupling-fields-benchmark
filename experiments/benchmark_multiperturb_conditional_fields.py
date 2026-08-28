"""Held-guide RNA-ATAC confirmation of conditional coupling fields."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import zlib
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from experiments.benchmark_public_coupling_fields import _embedding_laplacian
from mapreg.coupling_fields import (
    conditional_association_coordinates,
    fit_structured_coupling_fields,
    helmert_contrast,
    inverse_permutation_variance_weights,
)


ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = (
    ROOT / "data/development/public_coupling_atlas/multiperturb_states_v1.tsv.gz"
)
OUTPUT = ROOT / "results/multiperturb_conditional_fields.json"
SEED = 277747
PERMUTATIONS = 64
BOOTSTRAPS = 2_000
NUCLEAR_FRACTION = 0.1
GRAPH_PENALTY = 5.0
RIDGE_PENALTY = 0.1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_mod(values: np.ndarray, modulus: int, salt: str) -> np.ndarray:
    return np.asarray(
        [zlib.crc32((salt + "|" + value).encode()) % modulus for value in values],
        dtype=int,
    )


def _load_states(path: Path) -> dict[str, np.ndarray]:
    rows: list[list[str]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        rows.extend(reader)
    columns = {name: index for index, name in enumerate(header)}
    return {
        "cell": np.asarray([row[columns["cell"]] for row in rows]),
        "guide": np.asarray([row[columns["guide"]] for row in rows]),
        "target": np.asarray([row[columns["target"]] for row in rows]),
        "calibration": np.asarray(
            [int(row[columns["calibration"]]) for row in rows], dtype=bool
        ),
        "first": np.asarray(
            [int(row[columns["rna_state"]]) for row in rows], dtype=int
        ),
        "second": np.asarray(
            [int(row[columns["atac_state"]]) for row in rows], dtype=int
        ),
    }


def _statistics(
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray | int]:
    indices = np.flatnonzero(mask)
    if len(indices) < 30:
        raise ValueError("each held-guide table requires at least 30 nuclei")
    left = first[indices]
    right = second[indices]
    table = np.bincount(left * 3 + right, minlength=9).reshape(3, 3)
    probability = table / table.sum()
    basis = helmert_contrast(3)
    covariance = probability - np.outer(
        probability.sum(axis=1), probability.sum(axis=0)
    )
    endpoint = np.concatenate(
        (basis.T @ probability.sum(axis=1), basis.T @ probability.sum(axis=0))
    )
    estimate = conditional_association_coordinates(
        left,
        right,
        first_levels=3,
        second_levels=3,
        pseudocount=0.5,
        permutations=PERMUTATIONS,
        seed=seed,
    )
    return {
        "field": estimate.coordinates.ravel(),
        "variance": estimate.null_variance_coordinates.ravel(),
        "destroyed": estimate.destroyed_coordinates.ravel(),
        "covariance": (basis.T @ covariance @ basis).ravel(),
        "endpoint": endpoint,
        "cells": len(indices),
    }


def _build_arrays(data: dict[str, np.ndarray]):
    eligible = ~data["calibration"]
    control = eligible & (data["target"] == "non-targeting (human)")
    control_unit = _stable_mod(data["cell"], 2, "multiperturb-control")
    guides: dict[str, list[str]] = {}
    for target in sorted(set(data["target"]) - {"non-targeting (human)"}):
        counts = []
        for guide in sorted(set(data["guide"][eligible & (data["target"] == target)])):
            count = int(
                np.count_nonzero(
                    eligible & (data["target"] == target) & (data["guide"] == guide)
                )
            )
            if count >= 30:
                counts.append((count, guide))
        if len(counts) >= 2:
            guides[target] = [guide for _, guide in sorted(counts, reverse=True)[:2]]
    targets = sorted(guides)
    arrays = {
        key: np.empty((2, len(targets), 4), dtype=float)
        for key in ("field", "destroyed", "covariance", "endpoint", "variance")
    }
    support = np.empty((2, len(targets)), dtype=int)
    for unit in range(2):
        control_stats = _statistics(
            data["first"],
            data["second"],
            control & (control_unit == unit),
            SEED + 1_000 + unit,
        )
        for target_index, target in enumerate(targets):
            target_stats = _statistics(
                data["first"],
                data["second"],
                eligible
                & (data["target"] == target)
                & (data["guide"] == guides[target][unit]),
                SEED + 10_000 * unit + target_index,
            )
            for key in arrays:
                if key == "variance":
                    arrays[key][unit, target_index] = (
                        target_stats[key] + control_stats[key]
                    )
                else:
                    arrays[key][unit, target_index] = (
                        target_stats[key] - control_stats[key]
                    )
            support[unit, target_index] = int(target_stats["cells"])
    return arrays, targets, guides, support


def _structured(values: np.ndarray, laplacian: np.ndarray) -> np.ndarray:
    singular = np.linalg.svd(values, compute_uv=False)
    threshold = NUCLEAR_FRACTION * float(singular[0])
    return fit_structured_coupling_fields(
        values,
        graph_laplacian=laplacian,
        nuclear_penalty=threshold,
        graph_penalty=GRAPH_PENALTY,
        tolerance=1e-9,
    ).coefficient


def _reliability_shrink(values: np.ndarray, variance: np.ndarray) -> np.ndarray:
    noise = np.maximum(np.asarray(variance, dtype=float), 1e-12)
    signal_variance = np.maximum(
        np.mean(values**2, axis=0) - np.mean(noise, axis=0), 0.0
    )
    reliability = np.divide(
        signal_variance[None, :],
        signal_variance[None, :] + noise,
        out=np.zeros_like(noise),
        where=(signal_variance[None, :] + noise) > 0,
    )
    return reliability * values


def _weighted_structured(
    values: np.ndarray, variance: np.ndarray, laplacian: np.ndarray
) -> np.ndarray:
    weights = inverse_permutation_variance_weights(variance)
    singular = np.linalg.svd(values, compute_uv=False)
    threshold = NUCLEAR_FRACTION * float(singular[0])
    return fit_structured_coupling_fields(
        values,
        observation_weight=weights,
        graph_laplacian=laplacian,
        nuclear_penalty=threshold,
        graph_penalty=GRAPH_PENALTY,
        tolerance=1e-9,
    ).coefficient


def _nuclear(values: np.ndarray) -> np.ndarray:
    singular = np.linalg.svd(values, compute_uv=False)
    return fit_structured_coupling_fields(
        values,
        nuclear_penalty=NUCLEAR_FRACTION * float(singular[0]),
        tolerance=1e-9,
    ).coefficient


def _graph(values: np.ndarray, laplacian: np.ndarray) -> np.ndarray:
    return fit_structured_coupling_fields(
        values,
        graph_laplacian=laplacian,
        graph_penalty=GRAPH_PENALTY,
        tolerance=1e-9,
    ).coefficient


def _metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float | None]:
    flat_estimate = estimate.ravel()
    flat_truth = truth.ravel()
    correlation = (
        float(np.corrcoef(flat_estimate, flat_truth)[0, 1])
        if np.std(flat_estimate) > 0 and np.std(flat_truth) > 0
        else None
    )
    denominator = float(np.mean(flat_truth**2))
    rmse = float(np.sqrt(np.mean((flat_estimate - flat_truth) ** 2) / denominator))
    estimate_norm = np.linalg.norm(estimate, axis=-1)
    truth_norm = np.linalg.norm(truth, axis=-1)
    cosine = np.divide(
        np.sum(estimate * truth, axis=-1),
        estimate_norm * truth_norm,
        out=np.zeros_like(estimate_norm),
        where=(estimate_norm * truth_norm) > 0,
    )
    return {
        "pooled_pearson": correlation,
        "standardized_rmse": rmse,
        "macro_target_cosine": float(np.mean(cosine)),
    }


def _bootstrap_metrics(
    estimate: np.ndarray, truth: np.ndarray, seed: int
) -> dict[str, list[float] | None]:
    generator = np.random.default_rng(seed)
    draws = {name: [] for name in _metrics(estimate, truth)}
    for _ in range(BOOTSTRAPS):
        indices = generator.integers(0, truth.shape[1], size=truth.shape[1])
        values = _metrics(estimate[:, indices], truth[:, indices])
        for name, value in values.items():
            if value is not None:
                draws[name].append(value)
    return {
        name: (
            [float(value) for value in np.quantile(values, [0.025, 0.975])]
            if values
            else None
        )
        for name, values in draws.items()
    }


def run(path: Path, output: Path) -> dict[str, object]:
    data = _load_states(path)
    arrays, targets, guides, support = _build_arrays(data)
    scgpt_laplacian, scgpt = _embedding_laplacian(
        targets, ROOT / "data/scgpt_gene_embeddings.npz"
    )
    gene2vec_laplacian, gene2vec = _embedding_laplacian(
        targets, ROOT / "data/gene2vec_embeddings.npz"
    )
    permutation = np.random.default_rng(SEED).permutation(len(targets))
    shuffled = scgpt_laplacian[np.ix_(permutation, permutation)]
    predictions: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "zero",
            "direct",
            "nuclear",
            "scgpt_hypergraph",
            "conditional_nuclear_scgpt_hypergraph",
            "permutation_reliability_shrinkage",
            "variance_weighted_nuclear_scgpt_hypergraph",
            "conditional_nuclear_gene2vec_hypergraph",
            "shuffled_hypergraph",
            "endpoint_ridge",
            "endpoint_plus_structured_residual",
            "linear_cross_covariance",
            "destroyed_links",
        )
    }
    truths = []
    for held in range(2):
        source = 1 - held
        source_field = arrays["field"][source]
        truth = arrays["field"][held]
        truths.append(truth)
        predictions["zero"].append(np.zeros_like(source_field))
        predictions["direct"].append(source_field)
        predictions["nuclear"].append(_nuclear(source_field))
        predictions["scgpt_hypergraph"].append(
            _graph(source_field, scgpt_laplacian)
        )
        predictions["conditional_nuclear_scgpt_hypergraph"].append(
            _structured(source_field, scgpt_laplacian)
        )
        predictions["permutation_reliability_shrinkage"].append(
            _reliability_shrink(source_field, arrays["variance"][source])
        )
        predictions["variance_weighted_nuclear_scgpt_hypergraph"].append(
            _weighted_structured(
                source_field, arrays["variance"][source], scgpt_laplacian
            )
        )
        predictions["conditional_nuclear_gene2vec_hypergraph"].append(
            _structured(source_field, gene2vec_laplacian)
        )
        predictions["shuffled_hypergraph"].append(
            _structured(source_field, shuffled)
        )
        endpoint_model = Ridge(alpha=RIDGE_PENALTY).fit(
            arrays["endpoint"][source], source_field
        )
        endpoint_prediction = endpoint_model.predict(arrays["endpoint"][held])
        predictions["endpoint_ridge"].append(endpoint_prediction)
        residual = source_field - endpoint_model.predict(arrays["endpoint"][source])
        predictions["endpoint_plus_structured_residual"].append(
            endpoint_prediction + _structured(residual, scgpt_laplacian)
        )
        predictions["linear_cross_covariance"].append(
            arrays["covariance"][source]
        )
        predictions["destroyed_links"].append(arrays["destroyed"][source])
    truth_array = np.asarray(truths)
    methods = {}
    for index, (name, values) in enumerate(predictions.items()):
        estimate = np.asarray(values)
        if not np.isfinite(estimate).all():
            raise FloatingPointError(f"{name} produced non-finite predictions")
        methods[name] = {
            "metrics": _metrics(estimate, truth_array),
            "target_bootstrap_95_ci": _bootstrap_metrics(
                estimate, truth_array, SEED + 1_000 * (index + 1)
            ),
        }
    primary = "variance_weighted_nuclear_scgpt_hypergraph"
    competitor_names = [
        name
        for name in methods
        if name not in {primary, "destroyed_links", "shuffled_hypergraph"}
    ]
    best = min(
        competitor_names,
        key=lambda name: methods[name]["metrics"]["standardized_rmse"],
    )
    primary_prediction = np.asarray(predictions[primary])
    best_prediction = np.asarray(predictions[best])
    generator = np.random.default_rng(SEED + 90_000)
    loss_difference = []
    for _ in range(BOOTSTRAPS):
        indices = generator.integers(0, len(targets), size=len(targets))
        primary_loss = np.mean(
            (primary_prediction[:, indices] - truth_array[:, indices]) ** 2
        )
        best_loss = np.mean(
            (best_prediction[:, indices] - truth_array[:, indices]) ** 2
        )
        loss_difference.append(float(primary_loss - best_loss))
    loss_interval = [
        float(value) for value in np.quantile(loss_difference, [0.025, 0.975])
    ]
    result = {
        "schema": "v2p2r.multiperturb-conditional-fields/1",
        "panel": "GSE277747 MultiPerturb-seq RNA-ATAC",
        "targets": len(targets),
        "units": "two sequence-distinct guides per target",
        "minimum_guide_cells": int(support.min()),
        "median_guide_cells": float(np.median(support)),
        "field_coordinates": 4,
        "targets_and_guides": guides,
        "methods": methods,
        "primary": primary,
        "best_matched_competitor": best,
        "primary_minus_best_squared_error_bootstrap_95_ci": loss_interval,
        "success": bool(
            methods[primary]["metrics"]["standardized_rmse"]
            < methods[best]["metrics"]["standardized_rmse"]
            and methods[primary]["metrics"]["pooled_pearson"] > 0
            and loss_interval[1] < 0
        ),
        "fixed_settings": {
            "permutations": PERMUTATIONS,
            "pseudocount": 0.5,
            "nuclear_fraction": NUCLEAR_FRACTION,
            "graph_penalty": GRAPH_PENALTY,
            "ridge_penalty": RIDGE_PENALTY,
            "target_bootstraps": BOOTSTRAPS,
        },
        "embedding_hypergraphs": {"scgpt": scgpt, "gene2vec": gene2vec},
        "provenance": {
            "state_table": str(path.relative_to(ROOT)),
            "state_table_sha256": _sha256(path),
            "geo_tar_sha256": _sha256(
                ROOT / "data/external/multiperturb_gse277747/GSE277747_RAW.tar"
            ),
            "preprocessing_sha256": _sha256(
                ROOT / "experiments/preprocess_multiperturb_states.R"
            ),
            "estimator_sha256": _sha256(ROOT / "mapreg/coupling_fields.py"),
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
                "best_competitor": result["best_matched_competitor"],
                "success": result["success"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
