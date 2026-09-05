"""Fixed-split biological sensitivities for the Stephenson marker panel.

No recipient outcome selects a threshold, pair, penalty, or permutation. These
analyses are post hoc and do not replace the original confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from experiments import audit_classical_interaction_baselines as classical
from experiments import confirm_stephenson_citeseq as stephenson
from experiments.development import analyze_stephenson_posthoc_robustness as posthoc
from mapreg.heterogeneity_adaptive_coupling import (
    CouplingEstimationRefusal,
    expected_binary_table_from_log_odds,
)
from mapreg.hierarchical_conditional_coupling import (
    fit_hierarchical_conditional_log_odds,
)


PERMUTATIONS = 8
ROOT = Path(__file__).resolve().parents[2]


def validate_data(data: Mapping[str, np.ndarray]) -> None:
    counts = np.asarray(data["rna_counts"])
    if counts.ndim != 3 or counts.shape[1] != 9 or counts.shape[2] % 2:
        raise ValueError("counts must have shape donor x 9 x even cell budget")
    donors, _, cells = counts.shape
    for name in ("rna_counts", "adt_counts"):
        values = np.asarray(data[name])
        if (values.shape != counts.shape or not np.issubdtype(values.dtype, np.integer)
                or np.any(values < 0)):
            raise ValueError(f"{name} must be matching nonnegative integer counts")
    for name in ("cell_types", "barcodes"):
        values = np.asarray(data[name])
        if values.shape != (donors, cells) or values.dtype.kind not in "US":
            raise ValueError(f"{name} must contain donor x cell strings")
        if np.any(values == ""):
            raise ValueError(f"{name} contains empty labels")
    for name in ("donor_ids", "sample_ids", "roles"):
        values = np.asarray(data[name])
        if values.shape != (donors,) or values.dtype.kind not in "US":
            raise ValueError(f"{name} must contain one string per donor")
    if len(set(data["donor_ids"])) != donors:
        raise ValueError("physical donors must be unique")
    if len(set(data["sample_ids"])) != donors:
        raise ValueError("sample identifiers must be unique")
    if any(len(set(row)) != cells for row in data["barcodes"]):
        raise ValueError("barcodes must be unique within each donor")
    roles = np.asarray(data["roles"])
    if not set(roles) <= {"calibration", "pilot", "held_site"}:
        raise ValueError("roles must be calibration, pilot, or held_site")
    if np.count_nonzero(roles != "held_site") < 2 or not np.any(roles == "held_site"):
        raise ValueError("at least two source donors and one held donor are required")
    if list(data["markers"]) != list(stephenson.MARKERS):
        raise ValueError("marker order differs from the original panel")


def source_median_thresholds(adt_counts: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Pool equal cell budgets from source donors; never inspect held values."""
    return np.median(np.asarray(adt_counts)[source], axis=(0, 2))


def permute_within_cell_types(
    states: np.ndarray,
    cell_types: np.ndarray,
    barcodes: np.ndarray,
    donor: str,
    repeat: int,
) -> np.ndarray:
    """Shuffle intact protein vectors within lineage, invariant to cell order."""
    output = np.asarray(states).copy()
    for label in sorted(set(cell_types)):
        indices = np.flatnonzero(cell_types == label)
        indices = indices[np.argsort(barcodes[indices], kind="stable")]
        key = f"fixed-margin-lineage-null-v1\0{donor}\0{repeat}\0{label}"
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
        order = np.random.default_rng(seed).permutation(indices)
        output[:, indices] = states[:, order]
    return output


def make_tables(rna: np.ndarray, adt: np.ndarray) -> np.ndarray:
    return np.stack([posthoc._form_tables(x, y) for x, y in zip(rna, adt)])


def _rank_states(data: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.stack([
        posthoc._midrank_states(counts, bars, str(donor), str(sample))
        for counts, bars, donor, sample in zip(
            data["adt_counts"], data["barcodes"], data["donor_ids"], data["sample_ids"]
        )
    ])


def tables_from_counts(data: Mapping[str, np.ndarray], adt_rule: str = "median_rank") -> np.ndarray:
    """Return donor x RNA marker x ADT marker x 2 x 2 integer tables."""
    validate_data(data)
    if adt_rule == "median_rank":
        states = _rank_states(data)
    elif adt_rule == "source_median":
        source = np.asarray(data["roles"]) != "held_site"
        threshold = source_median_thresholds(data["adt_counts"], source)
        states = (data["adt_counts"] > threshold[None, :, None]).astype(np.uint8)
    else:
        raise ValueError(f"unknown ADT rule: {adt_rule}")
    return make_tables((data["rna_counts"] > 0).astype(np.uint8), states)


def sufficient_tables(data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Retain only aggregate tables needed to replay these two sensitivities."""
    validate_data(data)
    source = data["roles"] != "held_site"
    rna = (data["rna_counts"] > 0).astype(np.uint8)
    adt = _rank_states(data)
    ranked = make_tables(rna, adt)
    threshold = source_median_thresholds(data["adt_counts"], source)
    thresholded = make_tables(rna, data["adt_counts"] > threshold[None, :, None])
    nulls = []
    for repeat in range(PERMUTATIONS):
        shuffled = np.stack([
            permute_within_cell_types(adt[index], data["cell_types"][index],
                                     data["barcodes"][index], str(data["donor_ids"][index]), repeat)
            for index in np.flatnonzero(source)
        ])
        nulls.append(make_tables(rna[source], shuffled))
    return {
        "source_rank_tables": ranked[source], "held_rank_tables": ranked[~source],
        "source_threshold_tables": thresholded[source],
        "held_threshold_tables": thresholded[~source],
        "source_lineage_null_tables": np.stack(nulls),
        "source_pair_mask": informative(ranked[source]).sum(axis=0) >= 2,
        "threshold_pair_mask": informative(thresholded[source]).sum(axis=0) >= 2,
        "thresholds": threshold, "markers": np.asarray(data["markers"]),
        "source_donor_ids": data["donor_ids"][source],
        "held_donor_ids": data["donor_ids"][~source],
        "source_sample_ids": data["sample_ids"][source],
        "held_sample_ids": data["sample_ids"][~source],
    }


def informative(tables: np.ndarray) -> np.ndarray:
    total = tables.sum(axis=(-1, -2))
    row = tables[..., 0, :].sum(axis=-1)
    column = tables[..., :, 0].sum(axis=-1)
    return np.minimum(row, column) > np.maximum(0, row + column - total)


def fit_field(tables: np.ndarray, pair_mask: np.ndarray, method: str) -> dict:
    """Fit the frozen graph-zero hierarchy or extended common conditional MLE."""
    selected = tables[:, pair_mask][:, :, None]
    if not pair_mask.any():
        return {"status": "NO_SOURCE_SUPPORTED_PAIRS"}
    try:
        if method == "hierarchical":
            fit = fit_hierarchical_conditional_log_odds(
                selected, np.eye(selected.shape[1]), np.ones((1, 1)),
                heterogeneity_penalty=0.1, ridge_penalty=0.01, graph_penalty=0.0,
                minimum_informative_donors=2, maximum_condition_number=1e12,
                maximum_iterations=100, tolerance=1e-8,
            )
            return {
                "status": "FITTED", "log_odds": fit.population_log_odds.ravel(),
                "boundary": np.zeros(selected.shape[1], dtype=int),
                "certificate": {"iterations": fit.iterations,
                                "scaled_gradient_norm": fit.scaled_gradient_norm,
                                "converged": bool(fit.converged)},
            }
        if method != "common_conditional":
            raise ValueError(f"unknown method: {method}")
        fit = classical._common_effect_exact_cmle(selected)
        return {"status": "FITTED", "log_odds": fit.log_odds.ravel(),
                "boundary": fit.boundary.ravel(), "certificate": fit.certificate}
    except (CouplingEstimationRefusal, FloatingPointError, np.linalg.LinAlgError) as error:
        return {"status": "FIT_FAILED", "error_type": type(error).__name__,
                "reason": str(error)}


def score_field(fit: dict, tables: np.ndarray, pair_mask: np.ndarray,
                donor_ids: np.ndarray) -> list[dict]:
    results = []
    for donor, panel in zip(donor_ids, tables):
        selected = panel[pair_mask]
        support = informative(selected)
        row = {"donor": str(donor), "informative_pairs": int(support.sum()),
               "source_supported_pairs": int(pair_mask.sum()), "loss": None}
        if fit["status"] != "FITTED":
            row["status"] = fit["status"]
        elif not support.any():
            row["status"] = "NO_INFORMATIVE_PAIRS"
        else:
            losses = []
            for table, theta, boundary, active in zip(
                selected, fit["log_odds"], fit["boundary"], support
            ):
                if not active:
                    losses.append(None)
                    continue
                margins = (table.sum(axis=1), table.sum(axis=0))
                predicted = (classical._boundary_table(int(boundary), *margins)
                             if boundary else expected_binary_table_from_log_odds(
                                 float(theta), *margins))
                positive = table > 0
                if np.any(predicted[positive] <= 0):
                    losses.append(float("inf"))
                else:
                    losses.append(float(2 / table.sum() * np.sum(
                        table[positive] * np.log(table[positive] / predicted[positive])
                    )))
            finite = [loss for loss in losses if loss is not None]
            if np.isfinite(finite).all():
                row.update(status="SCORED", loss=float(np.mean(finite)), pair_losses=losses)
            else:
                row.update(status="INFINITE_DEVIANCE",
                           infinite_pairs=int(np.count_nonzero(np.isinf(finite))))
        results.append(row)
    return results


def compare_donors(primary: list[dict], comparator: list[dict], label: str) -> dict:
    if [row["donor"] for row in primary] != [row["donor"] for row in comparator]:
        raise ValueError("donor orders differ")
    paired = [(a, b) for a, b in zip(primary, comparator)
              if a["status"] == b["status"] == "SCORED"]
    if not paired:
        return {"status": "NO_SCORABLE_PAIRED_DONORS", "donors": 0}
    donors = [a["donor"] for a, _ in paired]
    result = posthoc._paired_comparison(
        donors, np.asarray([a["loss"] for a, _ in paired]),
        np.asarray([b["loss"] for _, b in paired]), label,
    )
    result["excluded_donors"] = len(primary) - len(paired)
    return result


def _aggregate_null(intact: list[dict], repeats: list[dict]) -> dict:
    rows = [intact] + [repeat["donor_results"] for repeat in repeats]
    if any(row["status"] != "SCORED" for panel in rows for row in panel):
        return {"status": "INCOMPLETE_REPETITIONS",
                "reason": "Aggregate requires all eight refits and all held donors."}
    losses = np.asarray([[row["loss"] for row in panel] for panel in rows])
    average = [{**row, "loss": float(mean)}
               for row, mean in zip(intact, losses[1:].mean(axis=0))]
    return {
        "status": "SCORED", "permutation_repetitions": len(repeats),
        "comparison_to_mean_null": compare_donors(intact, average, "lineage-null-mean"),
        "nested_uncertainty": posthoc._nested_bootstrap_interval(losses[:1] - losses[1:]),
    }


def _evaluate(source_tables: np.ndarray, held_tables: np.ndarray,
              mask: np.ndarray, donors: np.ndarray, method: str) -> dict:
    fit = fit_field(source_tables, mask, method)
    rows = score_field(fit, held_tables, mask, donors)
    serial = {key: value.tolist() if isinstance(value, np.ndarray) else value
              for key, value in fit.items()}
    return {"fit": serial, "donor_results": rows}


def analyze(data: Mapping[str, np.ndarray]) -> dict:
    """Run the two fixed sensitivities from selected counts, with all failures retained."""
    validate_data(data)
    source = np.asarray(data["roles"]) != "held_site"
    held = ~source
    donor_ids = np.asarray(data["donor_ids"])[held]
    rna = (np.asarray(data["rna_counts"]) > 0).astype(np.uint8)
    adt = _rank_states(data)
    tables = make_tables(rna, adt)
    mask = informative(tables[source]).sum(axis=0) >= 2
    intact = _evaluate(tables[source], tables[held], mask, donor_ids, "hierarchical")
    repeats = []
    for repeat in range(PERMUTATIONS):
        shuffled = np.stack([
            permute_within_cell_types(adt[index], data["cell_types"][index],
                                     data["barcodes"][index], str(data["donor_ids"][index]), repeat)
            for index in np.flatnonzero(source)
        ])
        null_tables = make_tables(rna[source], shuffled)
        result = _evaluate(null_tables, tables[held], mask, donor_ids, "hierarchical")
        result.update(repeat=repeat, comparison=compare_donors(
            intact["donor_results"], result["donor_results"], f"lineage-null-{repeat}"
        ))
        repeats.append(result)
    thresholds = source_median_thresholds(data["adt_counts"], source)
    threshold_states = (data["adt_counts"] > thresholds[None, :, None]).astype(np.uint8)
    threshold_tables = make_tables(rna, threshold_states)
    threshold_mask = informative(threshold_tables[source]).sum(axis=0) >= 2
    threshold_methods = {
        method: _evaluate(threshold_tables[source], threshold_tables[held],
                          threshold_mask, donor_ids, method)
        for method in ("hierarchical", "common_conditional")
    }
    return {
        "schema": "fixed-margin-biological-sensitivities-v1", "confirmatory": False,
        "source_donors": int(source.sum()), "held_donors": int(held.sum()),
        "cells_per_donor": int(rna.shape[-1]), "markers": list(data["markers"]),
        "donor_ids": list(data["donor_ids"]), "sample_ids": list(data["sample_ids"]),
        "roles": list(data["roles"]),
        "composition_preserving_null": {
            "source_pair_mask": mask.tolist(), "intact": intact, "repeats": repeats,
            "attempted_repeats": PERMUTATIONS,
            "fitted_repeats": sum(row["fit"]["status"] == "FITTED" for row in repeats),
            "aggregate": _aggregate_null(intact["donor_results"], repeats),
        },
        "source_median_threshold": {
            "thresholds": thresholds.tolist(), "state_rule": "raw ADT count > source pooled median",
            "source_pair_mask": threshold_mask.tolist(), "methods": threshold_methods,
            "adt_positive_counts_by_donor": threshold_states.sum(axis=2).tolist(),
            "adt_threshold_ties_by_donor": np.sum(
                data["adt_counts"] == thresholds[None, :, None], axis=2
            ).tolist(),
            "comparison": compare_donors(
                threshold_methods["hierarchical"]["donor_results"],
                threshold_methods["common_conditional"]["donor_results"], "source-threshold",
            ),
        },
    }


def main() -> None:
    from experiments.development.reanalyze_stephenson_prediction import verify_original

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT /
                        "results/development/stephenson_biological_reanalysis.json")
    parser.add_argument("--sufficient-tables", type=Path, default=ROOT /
                        "data/development/stephenson_biological_sufficient_tables.npz")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Output already exists; preserve completed analyses")
    with np.load(args.counts, allow_pickle=False) as archive:
        data = dict(archive)
    tables = tables_from_counts(data)
    source = data["roles"] != "held_site"
    mask = informative(tables[source]).sum(axis=0) >= 2
    intact = _evaluate(tables[source], tables[~source], mask,
                       data["donor_ids"][~source], "hierarchical")
    error = verify_original(intact["donor_results"])
    print(f"Original donor losses reproduced: max error {error:.3g}", flush=True)
    aggregates = sufficient_tables(data)
    args.sufficient_tables.parent.mkdir(parents=True, exist_ok=True)
    if args.sufficient_tables.exists():
        with np.load(args.sufficient_tables, allow_pickle=False) as previous:
            if set(previous.files) != set(aggregates) or any(
                not np.array_equal(previous[key], value) for key, value in aggregates.items()
            ):
                raise ValueError("existing sufficient tables differ; preserving both analyses")
    else:
        np.savez_compressed(args.sufficient_tables, **aggregates)
    result = analyze(data)
    result["original_loss_max_absolute_error"] = error
    result["bindings"] = {
        "counts_sha256": hashlib.sha256(args.counts.read_bytes()).hexdigest(),
        "sufficient_tables_sha256": hashlib.sha256(args.sufficient_tables.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(
            (ROOT / "docs/FIXED_MARGIN_PREDICTIVE_REANALYSIS.md").read_bytes()
        ).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"composition": result["composition_preserving_null"]["aggregate"],
                      "threshold": result["source_median_threshold"]["comparison"]}, indent=2))


if __name__ == "__main__":
    main()
