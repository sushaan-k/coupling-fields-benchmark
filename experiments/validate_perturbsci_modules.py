"""Strict module and neighborhood validation for PerturbSci coupling fields.

This script is a locked secondary analysis. It reuses the final field builder,
held-guide folds, variance weights, and structured estimator from
``benchmark_public_coupling_fields.py`` without changing them.

The module rule is fixed before execution: a target must lie in the top field-
norm quintile in every held truth and corresponding primary prediction. A
target selected by the identical all-fold rule on destroyed-link truth and
predictions is excluded. Reactome and CORUM are tested by one-sided Fisher
tests with Benjamini-Hochberg correction within family.

The neighborhood rule is also fixed: cosine 5-nearest-neighbor recovery is
scored against held truth, with targets as bootstrap units. Destroyed-link
predictions are the matched control. A 10,000-draw target-label permutation of
the primary neighbor graph supplies the random baseline. Reactome and CORUM
co-membership of predicted neighbor edges is evaluated by the same target
bootstrap and label-permutation null.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

from experiments.benchmark_public_coupling_fields import (
    DATA,
    ROOT,
    _build_perturbsci_fields,
    _embedding_laplacian,
    _field_metrics,
    _fixed_weighted_prediction,
)


OUTPUT = ROOT / "results/perturbsci_module_validation.json"
MAIN_RESULT = ROOT / "results/public_coupling_atlas_benchmark_v4_final_estimator.json"
SOURCE = DATA / "development/perturbsci_kinetics_gse218566/frozen_states_v1.npz"
RUNNER = ROOT / "experiments/validate_perturbsci_modules.py"
TEST_FILE = ROOT / "tests/test_validate_perturbsci_modules.py"
SEED = 218_566
TOP_FRACTION = 0.20
NEIGHBORS = 5
TARGET_BOOTSTRAPS = 2_000
LABEL_PERMUTATIONS = 10_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values."""

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("p_values must be a finite vector")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must lie in [0, 1]")
    if len(values) == 0:
        return values.copy()
    order = np.argsort(values, kind="mergesort")
    adjusted = np.empty_like(values)
    running = 1.0
    for position in range(len(order) - 1, -1, -1):
        index = order[position]
        running = min(running, values[index] * len(values) / (position + 1))
        adjusted[index] = running
    return adjusted


def _ranked_top(scores: np.ndarray, names: list[str], count: int) -> set[int]:
    values = np.asarray(scores, dtype=float)
    if values.shape != (len(names),) or not np.isfinite(values).all():
        raise ValueError("scores must be one finite value per target")
    if count < 1 or count > len(names):
        raise ValueError("count is outside the target range")
    order = sorted(range(len(names)), key=lambda index: (-values[index], names[index]))
    return set(order[:count])


def strict_replicated_selection(
    truth: np.ndarray,
    prediction: np.ndarray,
    destroyed_truth: np.ndarray,
    destroyed_prediction: np.ndarray,
    names: list[str],
    *,
    top_fraction: float = TOP_FRACTION,
) -> dict[str, object]:
    """Apply the locked every-fold high-norm rule and matched null exclusion."""

    arrays = [
        np.asarray(values, dtype=float)
        for values in (truth, prediction, destroyed_truth, destroyed_prediction)
    ]
    if any(values.shape != arrays[0].shape for values in arrays):
        raise ValueError("all field arrays must have the same shape")
    if arrays[0].ndim != 3 or arrays[0].shape[1] != len(names):
        raise ValueError("fields must have shape fold x target x coordinate")
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError("field arrays must be finite")
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must lie in (0, 1]")

    selected_count = max(1, int(np.ceil(top_fraction * len(names))))
    fold_records = []
    primary_sets = []
    destroyed_sets = []
    for fold in range(arrays[0].shape[0]):
        truth_top = _ranked_top(
            np.linalg.norm(arrays[0][fold], axis=1), names, selected_count
        )
        prediction_top = _ranked_top(
            np.linalg.norm(arrays[1][fold], axis=1), names, selected_count
        )
        destroyed_truth_top = _ranked_top(
            np.linalg.norm(arrays[2][fold], axis=1), names, selected_count
        )
        destroyed_prediction_top = _ranked_top(
            np.linalg.norm(arrays[3][fold], axis=1), names, selected_count
        )
        primary_sets.extend((truth_top, prediction_top))
        destroyed_sets.extend((destroyed_truth_top, destroyed_prediction_top))
        fold_records.append(
            {
                "held_unit": fold,
                "truth_top_targets": sorted(names[index] for index in truth_top),
                "primary_top_targets": sorted(
                    names[index] for index in prediction_top
                ),
                "destroyed_truth_top_targets": sorted(
                    names[index] for index in destroyed_truth_top
                ),
                "destroyed_prediction_top_targets": sorted(
                    names[index] for index in destroyed_prediction_top
                ),
            }
        )
    replicated = set.intersection(*primary_sets)
    destroyed_replicated = set.intersection(*destroyed_sets)
    retained = replicated - destroyed_replicated
    return {
        "top_fraction": top_fraction,
        "targets_per_top_set": selected_count,
        "rule": (
            "top field-norm quintile in every held truth and corresponding "
            "primary prediction; exclude targets passing the identical rule "
            "for destroyed-link truths and predictions"
        ),
        "selected_before_destroyed_exclusion": sorted(
            names[index] for index in replicated
        ),
        "selected_by_destroyed_links": sorted(
            names[index] for index in destroyed_replicated
        ),
        "excluded_destroyed_overlap": sorted(
            names[index] for index in replicated & destroyed_replicated
        ),
        "selected_targets": sorted(names[index] for index in retained),
        "folds": fold_records,
    }


def _annotation_sets(
    names: list[str], family: str
) -> list[tuple[str, set[str]]]:
    universe = set(names)
    records: list[tuple[str, set[str]]] = []
    if family == "Reactome":
        with (DATA / "ReactomePathways.gmt").open() as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                members = set(fields[2:]) & universe
                if 3 <= len(members) <= 30:
                    records.append((fields[0], members))
    elif family == "CORUM":
        with (DATA / "corum_humanComplexes_current.txt").open() as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                if row.get("organism") != "Human":
                    continue
                members = set(row["subunits_gene_name"].split(";")) & universe
                if 2 <= len(members) <= 20:
                    records.append((row["complex_name"], members))
    else:
        raise ValueError(f"unknown annotation family: {family}")

    unique = {}
    for label, members in records:
        key = (label, tuple(sorted(members)))
        unique[key] = (label, members)
    return [unique[key] for key in sorted(unique)]


def module_enrichment(
    selected: set[str],
    universe: set[str],
    annotations: list[tuple[str, set[str]]],
) -> dict[str, object]:
    """Test over-representation of annotations in a locked selected set."""

    if not selected <= universe:
        raise ValueError("selected targets must belong to the universe")
    rows = []
    for label, members in annotations:
        annotated = members & universe
        overlap = selected & annotated
        table = np.asarray(
            [
                [len(overlap), len(selected - annotated)],
                [len(annotated - selected), len(universe - selected - annotated)],
            ]
        )
        if len(selected) == 0:
            odds_ratio, p_value = 0.0, 1.0
        else:
            test = fisher_exact(table, alternative="greater")
            odds_ratio, p_value = float(test.statistic), float(test.pvalue)
        rows.append(
            {
                "annotation": label,
                "members_in_universe": len(annotated),
                "selected_members": sorted(overlap),
                "odds_ratio": odds_ratio if np.isfinite(odds_ratio) else None,
                "p_value": p_value,
            }
        )
    adjusted = benjamini_hochberg(
        np.asarray([row["p_value"] for row in rows], dtype=float)
    )
    for row, q_value in zip(rows, adjusted):
        row["q_value"] = float(q_value)
    rows.sort(key=lambda row: (row["q_value"], row["p_value"], row["annotation"]))
    return {
        "tested_annotations": len(rows),
        "bh_significant": sum(row["q_value"] < 0.05 for row in rows),
        "significant_results": [row for row in rows if row["q_value"] < 0.05],
        "all_results": rows,
    }


def cosine_neighbors(values: np.ndarray, names: list[str], k: int) -> np.ndarray:
    """Return deterministic cosine-nearest-neighbor indices for each target."""

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(names):
        raise ValueError("values must have shape target x coordinate")
    if not np.isfinite(matrix).all() or not 1 <= k < len(names):
        raise ValueError("invalid neighbor input")
    norm = np.linalg.norm(matrix, axis=1)
    normalized = np.divide(
        matrix,
        norm[:, None],
        out=np.zeros_like(matrix),
        where=norm[:, None] > 0.0,
    )
    similarity = normalized @ normalized.T
    result = np.empty((len(names), k), dtype=int)
    for target in range(len(names)):
        candidates = [index for index in range(len(names)) if index != target]
        candidates.sort(key=lambda index: (-similarity[target, index], names[index]))
        result[target] = candidates[:k]
    return result


def neighbor_recovery_by_target(
    truth_neighbors: np.ndarray, candidate_neighbors: np.ndarray
) -> np.ndarray:
    """Return fold-averaged neighbor recall for each target."""

    truth = np.asarray(truth_neighbors, dtype=int)
    candidate = np.asarray(candidate_neighbors, dtype=int)
    if truth.shape != candidate.shape or truth.ndim != 3:
        raise ValueError("neighbor arrays must share fold x target x k shape")
    per_fold = np.empty(truth.shape[:2], dtype=float)
    for fold in range(truth.shape[0]):
        for target in range(truth.shape[1]):
            per_fold[fold, target] = len(
                set(truth[fold, target]) & set(candidate[fold, target])
            ) / truth.shape[2]
    return per_fold.mean(axis=0)


def _bootstrap_mean(values: np.ndarray, seed: int) -> dict[str, object]:
    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1 or len(samples) < 2 or not np.isfinite(samples).all():
        raise ValueError("bootstrap values must be a finite vector")
    generator = np.random.default_rng(seed)
    draws = np.empty(TARGET_BOOTSTRAPS)
    for draw in range(TARGET_BOOTSTRAPS):
        indices = generator.integers(0, len(samples), size=len(samples))
        draws[draw] = samples[indices].mean()
    return {
        "mean": float(samples.mean()),
        "target_bootstrap_95_ci": [
            float(value) for value in np.quantile(draws, [0.025, 0.975])
        ],
    }


def relabel_neighbor_graph(neighbors: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    """Randomly assign a fixed neighbor graph to target labels."""

    graph = np.asarray(neighbors, dtype=int)
    labels = np.asarray(permutation, dtype=int)
    if graph.ndim != 2 or labels.shape != (graph.shape[0],):
        raise ValueError("incompatible graph and permutation")
    if set(labels.tolist()) != set(range(len(labels))):
        raise ValueError("permutation must contain every target index once")
    inverse = np.empty_like(labels)
    inverse[labels] = np.arange(len(labels))
    return inverse[graph[labels]]


def _shared_annotation_matrix(
    names: list[str], annotations: list[tuple[str, set[str]]]
) -> np.ndarray:
    index = {name: position for position, name in enumerate(names)}
    shared = np.zeros((len(names), len(names)), dtype=bool)
    for _, members in annotations:
        positions = [index[name] for name in members if name in index]
        shared[np.ix_(positions, positions)] = True
    np.fill_diagonal(shared, False)
    return shared


def neighbor_annotation_by_target(
    neighbors: np.ndarray, shared: np.ndarray
) -> np.ndarray:
    """Return fold-averaged annotated-edge fraction for each target."""

    graph = np.asarray(neighbors, dtype=int)
    matrix = np.asarray(shared, dtype=bool)
    if graph.ndim != 3 or matrix.shape != (graph.shape[1], graph.shape[1]):
        raise ValueError("incompatible neighbor graph and annotation matrix")
    target = np.arange(graph.shape[1])[:, None]
    per_fold = np.asarray(
        [matrix[target, graph[fold]].mean(axis=1) for fold in range(graph.shape[0])]
    )
    return per_fold.mean(axis=0)


def permutation_null(
    truth_neighbors: np.ndarray,
    primary_neighbors: np.ndarray,
    shared_matrices: dict[str, np.ndarray],
    *,
    draws: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Generate target-label permutation nulls for recovery and annotation edges."""

    truth = np.asarray(truth_neighbors, dtype=int)
    primary = np.asarray(primary_neighbors, dtype=int)
    if truth.shape != primary.shape or truth.ndim != 3:
        raise ValueError("neighbor arrays must share fold x target x k shape")
    generator = np.random.default_rng(seed)
    recovery = np.empty(draws)
    annotation = {family: np.empty(draws) for family in shared_matrices}
    target = np.arange(truth.shape[1])[:, None]
    for draw in range(draws):
        fold_recovery = []
        fold_annotation = {family: [] for family in shared_matrices}
        for fold in range(truth.shape[0]):
            relabeled = relabel_neighbor_graph(
                primary[fold], generator.permutation(truth.shape[1])
            )
            fold_recovery.extend(
                len(set(truth[fold, index]) & set(relabeled[index])) / truth.shape[2]
                for index in range(truth.shape[1])
            )
            for family, shared in shared_matrices.items():
                fold_annotation[family].append(
                    float(shared[target, relabeled].mean())
                )
        recovery[draw] = float(np.mean(fold_recovery))
        for family in shared_matrices:
            annotation[family][draw] = float(np.mean(fold_annotation[family]))
    return {"recovery": recovery, **annotation}


def _permutation_summary(observed: float, null: np.ndarray) -> dict[str, object]:
    values = np.asarray(null, dtype=float)
    return {
        "null_mean": float(values.mean()),
        "null_95_interval": [
            float(value) for value in np.quantile(values, [0.025, 0.975])
        ],
        "one_sided_p_value": float(
            (1 + np.count_nonzero(values >= observed)) / (len(values) + 1)
        ),
    }


def _neighbor_records(
    names: list[str],
    truth: np.ndarray,
    primary: np.ndarray,
    destroyed: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for fold in range(truth.shape[0]):
        for target, name in enumerate(names):
            rows.append(
                {
                    "held_unit": fold,
                    "target": name,
                    "truth_neighbors": [names[index] for index in truth[fold, target]],
                    "primary_neighbors": [
                        names[index] for index in primary[fold, target]
                    ],
                    "destroyed_neighbors": [
                        names[index] for index in destroyed[fold, target]
                    ],
                }
            )
    return rows


def _replicated_interpretable_edges(
    names: list[str],
    truth: np.ndarray,
    primary: np.ndarray,
    annotation_membership: dict[str, list[tuple[str, set[str]]]],
) -> list[dict[str, object]]:
    rows = []
    for target, name in enumerate(names):
        for neighbor_index, neighbor in enumerate(names):
            if target == neighbor_index:
                continue
            folds = sum(
                neighbor_index in truth[fold, target]
                and neighbor_index in primary[fold, target]
                for fold in range(truth.shape[0])
            )
            if folds != truth.shape[0]:
                continue
            shared = {}
            for family, annotation_records in annotation_membership.items():
                labels = [
                    label
                    for label, members in annotation_records
                    if name in members and neighbor in members
                ]
                if labels:
                    shared[family] = labels
            if shared:
                rows.append(
                    {
                        "target": name,
                        "neighbor": neighbor,
                        "recovered_in_all_held_units": True,
                        "shared_annotations": shared,
                    }
                )
    return rows


def _build_final_fold_arrays() -> tuple[dict[str, np.ndarray], list[str], dict[str, object]]:
    arrays, names, source = _build_perturbsci_fields()
    laplacian, graph = _embedding_laplacian(
        names, DATA / "scgpt_gene_embeddings.npz"
    )
    truth = []
    primary = []
    destroyed_truth = []
    destroyed_prediction = []
    for held in range(3):
        training = [unit for unit in range(3) if unit != held]
        first, second = training
        train = 0.5 * (arrays["field"][first] + arrays["field"][second])
        variance = 0.25 * (arrays["variance"][first] + arrays["variance"][second])
        prediction, _ = _fixed_weighted_prediction(train, variance, laplacian)
        truth.append(arrays["field"][held])
        primary.append(prediction)
        destroyed_truth.append(arrays["destroyed"][held])
        destroyed_prediction.append(
            0.5 * (arrays["destroyed"][first] + arrays["destroyed"][second])
        )
    return {
        "truth": np.asarray(truth),
        "primary": np.asarray(primary),
        "destroyed_truth": np.asarray(destroyed_truth),
        "destroyed_prediction": np.asarray(destroyed_prediction),
    }, names, {"source": source, "scgpt_hypergraph": graph}


def run(output: Path = OUTPUT) -> dict[str, object]:
    """Execute the locked PerturbSci module and neighborhood validation."""

    fields, names, source = _build_final_fold_arrays()
    with MAIN_RESULT.open() as handle:
        main = json.load(handle)
    main_panel = next(
        panel for panel in main["panels"] if panel["panel"] == "PerturbSci-Kinetics"
    )
    reconstructed = _field_metrics(fields["primary"], fields["truth"])
    per_held_guide_metrics = [
        {
            "held_guide_rotation": held,
            **_field_metrics(fields["primary"][held], fields["truth"][held]),
        }
        for held in range(fields["truth"].shape[0])
    ]
    recorded = main_panel["methods"][
        "variance_weighted_nuclear_scgpt_hypergraph"
    ]["metrics"]
    for metric, value in reconstructed.items():
        if value != recorded[metric]:
            raise RuntimeError(f"reconstructed {metric} differs from the final benchmark")

    selection = strict_replicated_selection(
        fields["truth"],
        fields["primary"],
        fields["destroyed_truth"],
        fields["destroyed_prediction"],
        names,
    )
    selected = set(selection["selected_targets"])
    annotation_membership = {
        family: _annotation_sets(names, family) for family in ("Reactome", "CORUM")
    }
    enrichment = {
        family: module_enrichment(selected, set(names), annotations)
        for family, annotations in annotation_membership.items()
    }

    truth_neighbors = np.asarray(
        [cosine_neighbors(values, names, NEIGHBORS) for values in fields["truth"]]
    )
    primary_neighbors = np.asarray(
        [cosine_neighbors(values, names, NEIGHBORS) for values in fields["primary"]]
    )
    destroyed_neighbors = np.asarray(
        [
            cosine_neighbors(values, names, NEIGHBORS)
            for values in fields["destroyed_prediction"]
        ]
    )
    primary_recovery = neighbor_recovery_by_target(truth_neighbors, primary_neighbors)
    destroyed_recovery = neighbor_recovery_by_target(
        truth_neighbors, destroyed_neighbors
    )
    shared_matrices = {
        family: _shared_annotation_matrix(names, annotations)
        for family, annotations in annotation_membership.items()
    }
    null = permutation_null(
        truth_neighbors,
        primary_neighbors,
        shared_matrices,
        draws=LABEL_PERMUTATIONS,
        seed=SEED + 9_000,
    )
    recovery = {
        "primary": _bootstrap_mean(primary_recovery, SEED + 1_000),
        "destroyed_links": _bootstrap_mean(destroyed_recovery, SEED + 2_000),
        "primary_minus_destroyed": _bootstrap_mean(
            primary_recovery - destroyed_recovery, SEED + 3_000
        ),
        "primary_vs_random_label_permutation": _permutation_summary(
            float(primary_recovery.mean()), null["recovery"]
        ),
    }

    annotation_results = {}
    random_p_values = []
    for family, shared in shared_matrices.items():
        primary_values = neighbor_annotation_by_target(primary_neighbors, shared)
        destroyed_values = neighbor_annotation_by_target(destroyed_neighbors, shared)
        random_summary = _permutation_summary(
            float(primary_values.mean()), null[family]
        )
        random_p_values.append(random_summary["one_sided_p_value"])
        annotation_results[family] = {
            "primary": _bootstrap_mean(primary_values, SEED + 4_000),
            "destroyed_links": _bootstrap_mean(destroyed_values, SEED + 5_000),
            "primary_minus_destroyed": _bootstrap_mean(
                primary_values - destroyed_values, SEED + 6_000
            ),
            "primary_vs_random_label_permutation": random_summary,
        }
    adjusted = benjamini_hochberg(np.asarray(random_p_values))
    for family, q_value in zip(annotation_results, adjusted):
        annotation_results[family]["random_label_permutation_bh_q_value"] = float(
            q_value
        )

    module_positive = any(
        family["bh_significant"] > 0 for family in enrichment.values()
    )
    recovery_positive = (
        recovery["primary_minus_destroyed"]["target_bootstrap_95_ci"][0] > 0.0
        and recovery["primary_vs_random_label_permutation"]["one_sided_p_value"]
        <= 0.05
    )
    annotation_positive = {
        family: bool(
            result["primary_minus_destroyed"]["target_bootstrap_95_ci"][0] > 0.0
            and result["random_label_permutation_bh_q_value"] <= 0.05
        )
        for family, result in annotation_results.items()
    }
    result = {
        "schema": "v2p2r.perturbsci-module-validation/1",
        "panel": "GSE218566 PerturbSci-Kinetics",
        "preanalysis_contract": {
            "module_selection": selection["rule"],
            "module_top_fraction": TOP_FRACTION,
            "annotation_tests": (
                "one-sided Fisher over the 85-target universe; BH within Reactome "
                "and CORUM separately"
            ),
            "nearest_neighbor_k": NEIGHBORS,
            "nearest_neighbor_metric": "cosine",
            "recovery_uncertainty": (
                "2,000 target bootstraps; primary compared with destroyed links"
            ),
            "random_control": (
                "10,000 target-label permutations of the fixed primary neighbor graph"
            ),
            "neighbor_annotation_test": (
                "fraction of directed kNN edges sharing at least one annotation; "
                "target bootstrap vs destroyed links and label permutation vs random; "
                "BH across the two declared annotation families"
            ),
        },
        "targets": len(names),
        "held_units": 3,
        "same_final_fold_check": {
            "passed": True,
            "reconstructed_metrics": reconstructed,
            "recorded_metrics": recorded,
            "per_held_guide_metrics": per_held_guide_metrics,
        },
        "strict_module_selection": selection,
        "module_enrichment": enrichment,
        "nearest_neighbor_validation": {
            "k": NEIGHBORS,
            "recovery": recovery,
            "annotation_enrichment": annotation_results,
            "replicated_interpretable_edges": _replicated_interpretable_edges(
                names, truth_neighbors, primary_neighbors, annotation_membership
            ),
            "all_neighbor_records": _neighbor_records(
                names, truth_neighbors, primary_neighbors, destroyed_neighbors
            ),
        },
        "decisions": {
            "strict_module_validation": "PROMOTE" if module_positive else "REFUSE",
            "nearest_neighbor_recovery": "PROMOTE" if recovery_positive else "REFUSE",
            "neighbor_annotation_enrichment": {
                family: "PROMOTE" if positive else "REFUSE"
                for family, positive in annotation_positive.items()
            },
        },
        "provenance": {
            "source": str(SOURCE.relative_to(ROOT)),
            "source_sha256": _sha256(SOURCE),
            "main_benchmark": str(MAIN_RESULT.relative_to(ROOT)),
            "main_benchmark_sha256": _sha256(MAIN_RESULT),
            "field_builder": "experiments/benchmark_public_coupling_fields.py",
            "field_builder_sha256": _sha256(
                ROOT / "experiments/benchmark_public_coupling_fields.py"
            ),
            "estimator": "mapreg/coupling_fields.py",
            "estimator_sha256": _sha256(ROOT / "mapreg/coupling_fields.py"),
            "runner": str(RUNNER.relative_to(ROOT)),
            "runner_sha256": _sha256(RUNNER),
            "tests": str(TEST_FILE.relative_to(ROOT)),
            "tests_sha256": _sha256(TEST_FILE),
            "reactome_sha256": _sha256(DATA / "ReactomePathways.gmt"),
            "corum_sha256": _sha256(DATA / "corum_humanComplexes_current.txt"),
            "source_metadata": source,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    result = run(arguments.output)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "selected_targets": result["strict_module_selection"][
                    "selected_targets"
                ],
                "decisions": result["decisions"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
