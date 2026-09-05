"""Post hoc assay-availability sensitivity using unchanged published predictions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.development import fixed_margin_biology as biology
from experiments.development import analyze_stephenson_posthoc_robustness as posthoc

ROOT = Path(__file__).resolve().parents[2]
EXCLUDED = {"C-8914": 3, "C-8939": 0}
CONFIRMATION = ROOT / "results/stephenson_citeseq_confirmation.json"
CLASSICAL = ROOT / "results/development/classical_interaction_baselines_posthoc.json"


def summarize_counts(data):
    biology.validate_data(data)
    adt = data["adt_counts"]
    donors = []
    for index, donor in enumerate(data["donor_ids"]):
        row = {"donor": str(donor), "sample": str(data["sample_ids"][index]),
               "role": str(data["roles"][index]), "total_adt_counts": int(adt[index].sum())}
        if donor in EXCLUDED:
            row["markers"] = [
                {"marker": str(marker), "minimum": int(values.min()),
                 "maximum": int(values.max()), "total_counts": int(values.sum()),
                 "nonzero_cells": int(np.count_nonzero(values)),
                 "nonzero_fraction": float(np.mean(values > 0))}
                for marker, values in zip(data["markers"], adt[index])
            ]
            row["cells_with_any_panel_adt"] = int(np.count_nonzero(adt[index].sum(axis=0)))
        donors.append(row)
    return {"schema": "stephenson-selected-panel-adt-diagnostics-v1",
            "cells_per_donor": int(adt.shape[-1]), "markers": list(data["markers"]),
            "donors": donors}


def analyze(diagnostics, confirmation, classical):
    inventory = {row["donor"]: row for row in diagnostics["donors"]}
    if len(inventory) != 92 or diagnostics["cells_per_donor"] != 512:
        raise ValueError("diagnostics differ from the original 92-donor selected panel")
    if {donor: inventory[donor]["total_adt_counts"] for donor in EXCLUDED} != EXCLUDED:
        raise ValueError("the two fixed assay-QC diagnoses do not match")
    held = confirmation["donor_results"]
    if len(held) != 56 or len({row["donor"] for row in held}) != 56:
        raise ValueError("the original held panel must contain 56 unique donors")
    kept = [row for row in held if row["donor"] not in EXCLUDED]
    if len(kept) != 54:
        raise ValueError("the fixed sensitivity must retain exactly 54 donors")
    fitted = classical["studies"]["stephenson_newcastle_held_site"]["held_losses"]
    names = {"signed_statistic": "best_residual",
             "common_conditional": "common_effect_exact_cmle",
             "pooled_conditional": "pooled_poisson_loglinear_interaction"}
    rows = []
    for row in kept:
        donor = row["donor"]
        if fitted["primary"][donor] != row["losses"]["primary"]:
            raise ValueError("published primary predictions differ between comparison artifacts")
        losses = {"hierarchical": row["losses"]["primary"]}
        for method, key in names.items():
            losses[method] = (row["losses"][key] if method == "signed_statistic"
                              else fitted[key][donor])
        rows.append({"donor": donor, "sample": row["sample"], "losses": losses})
    comparisons = {
        method: posthoc._paired_comparison(
            [row["donor"] for row in rows],
            np.array([row["losses"]["hierarchical"] for row in rows]),
            np.array([row["losses"][method] for row in rows]), f"assay-qc-{method}",
        ) for method in names
    }
    return {
        "schema": "stephenson-posthoc-assay-qc-sensitivity-v1", "confirmatory": False,
        "analysis_role": "post_hoc_assay_availability_sensitivity",
        "selection_reason": "Two donors identified after threshold sensitivity had 3 and 0 selected-panel ADT counts.",
        "original_confirmation_unchanged": True, "models_refitted": False,
        "predictions_retuned": False, "original_held_donors": 56, "retained_held_donors": 54,
        "excluded_donors": [inventory[donor] for donor in EXCLUDED],
        "next_lowest_panel_count_among_other_donors": min(
            row["total_adt_counts"] for donor, row in inventory.items() if donor not in EXCLUDED
        ),
        "adt_diagnostics": diagnostics, "donor_results": rows, "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--counts", type=Path)
    inputs.add_argument("--diagnostics", type=Path)
    parser.add_argument("--save-diagnostics", type=Path, default=ROOT /
                        "data/development/stephenson_assay_qc_diagnostics.json")
    parser.add_argument("--output", type=Path, default=ROOT /
                        "results/development/stephenson_assay_qc_sensitivity.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserving the completed sensitivity")
    if args.counts:
        with np.load(args.counts, allow_pickle=False) as archive:
            diagnostics = summarize_counts(dict(archive))
        args.save_diagnostics.parent.mkdir(parents=True, exist_ok=True)
        with args.save_diagnostics.open("x") as stream:
            json.dump(diagnostics, stream, indent=2, allow_nan=False)
            stream.write("\n")
    else:
        diagnostics = json.loads(args.diagnostics.read_text())
    result = analyze(diagnostics, json.loads(CONFIRMATION.read_text()), json.loads(CLASSICAL.read_text()))
    paths = {"diagnostics": args.diagnostics or args.save_diagnostics,
             "confirmation": CONFIRMATION, "classical": CLASSICAL, "runner": Path(__file__)}
    result["bindings"] = {name + "_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                          for name, path in paths.items()}
    if args.counts:
        result["bindings"]["selected_counts_sha256"] = hashlib.sha256(args.counts.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["comparisons"], indent=2))


if __name__ == "__main__":
    main()
