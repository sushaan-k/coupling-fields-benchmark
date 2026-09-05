"""Summarize the fixed predictions on all donors and the published QC subset."""

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.development.reanalyze_stephenson_prediction import compare


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results/development/stephenson_predictive_reanalysis.json"
OUTPUT = ROOT / "results/development/stephenson_predictive_reanalysis_summary.json"
FROZEN = ROOT / "results/stephenson_citeseq_confirmation.json"
QC_EXCLUDED = {"C-8914", "C-8939"}


def summarize(result):
    summaries = {}
    frozen = {r["donor"]: r for r in json.loads(FROZEN.read_text())["donor_results"]}
    for cohort, excluded in (("original_56", set()), ("assay_qc_54", QC_EXCLUDED)):
        methods = {}
        for name, method in result["methods"].items():
            rows = [r for r in method.get("donor_results", []) if r["donor"] not in excluded]
            if not rows or any(r["status"] != "SCORED" for r in rows):
                methods[name] = {"status": "UNAVAILABLE"}
                continue
            rng = np.random.default_rng(20260904)
            draws = rng.integers(0, len(rows), size=(20000, len(rows)))
            summary = {"status": "SCORED", "donors": len(rows), "alpha": method["alpha"]}
            for field in ("loss", "coverage", "mean_interval_width"):
                if field not in rows[0]:
                    continue
                values = np.array([r[field] for r in rows])
                summary[field] = {"mean": float(values.mean()),
                                  "bootstrap_95_ci": np.quantile(values[draws].mean(axis=1), [.025, .975]).tolist()}
            methods[name] = summary
        comparisons = {}
        original = [r for r in result["methods"]["original_hierarchy"]["donor_results"]
                    if r["donor"] not in excluded]
        for name, method in result["methods"].items():
            if name != "original_hierarchy" and "donor_results" in method:
                rows = [r for r in method["donor_results"] if r["donor"] not in excluded]
                comparisons[f"original_vs_{name}"] = compare(original, rows, f"{cohort}-{name}")
        for suffix in ("", "_alpha1"):
            rows = [[r for r in result["methods"][f"random_effects_{kind}{suffix}"].get("donor_results", [])
                     if r["donor"] not in excluded] for kind in ("mixture", "plugin")]
            if all(rows):
                comparisons[f"mixture_vs_plugin{suffix}"] = compare(*rows, f"{cohort}-mixture{suffix}")
        mixture = [r for r in result["methods"]["random_effects_mixture"].get("donor_results", [])
                   if r["donor"] not in excluded]
        if mixture:
            comparisons["mixture_vs_original"] = compare(mixture, original, f"{cohort}-mixture-original")
            common = [r for r in result["methods"]["common_ridge"].get("donor_results", [])
                      if r["donor"] not in excluded]
            if common:
                comparisons["mixture_vs_common_ridge"] = compare(mixture, common, f"{cohort}-mixture-ridge")
            statistic = [{"donor": r["donor"], "status": "SCORED",
                          "loss": frozen[r["donor"]]["losses"]["best_residual"]} for r in mixture]
            comparisons["mixture_vs_original_signed_statistic"] = compare(
                mixture, statistic, f"{cohort}-mixture-frozen-statistic")
        if cohort == "original_56":
            comparisons.update(result["comparisons"])
        summaries[cohort] = {"excluded_donors": sorted(excluded), "methods": methods,
                             "comparisons": comparisons}
    return {"confirmatory": False, "source_refitted_for_qc": False,
            "bootstrap_draws": 20000, "bootstrap_unit": "recipient donor; source fit fixed",
            "cohorts": summaries}


if __name__ == "__main__":
    if OUTPUT.exists():
        raise FileExistsError("Preserve completed summary")
    payload = summarize(json.loads(INPUT.read_text()))
    payload["bindings"] = {"predictions_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
                           "frozen_original_sha256": hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
                           "summary_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
