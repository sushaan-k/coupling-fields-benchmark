"""Post-hoc, fixed-split comparison of conditional predictive estimators."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy.optimize import brentq

from experiments import audit_classical_interaction_baselines as classical
from experiments.development.fixed_margin_biology import informative
from mapreg.heterogeneity_adaptive_coupling import (
    CouplingEstimationRefusal,
    evaluate_conditional_log_odds,
    fit_structured_conditional_log_odds,
)
from mapreg.hierarchical_conditional_coupling import fit_hierarchical_conditional_log_odds
from mapreg.predictive_conditional import normal_mixture_prediction


ROOT = Path(__file__).resolve().parents[2]
ALPHAS = (0.5, 0.75, 1.0, 1.25)
BOOTSTRAPS = 20_000
DEFAULT_OUTPUT = ROOT / "results/development/stephenson_predictive_reanalysis.json"
R_SCRIPT = ROOT / "experiments/development/fit_conditional_random_effects.R"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_tables(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in ("tables", "donor_ids", "sample_ids", "roles")}
    tables = data["tables"]
    if (tables.shape != (92, 9, 9, 2, 2)
            or not np.issubdtype(tables.dtype, np.integer) or np.any(tables < 0)
            or not np.all(tables.sum(axis=(-1, -2)) == 512)):
        raise ValueError("Expected the original 92-donor integer 512-cell panel")
    for name in ("donor_ids", "sample_ids", "roles"):
        if data[name].shape != (92,) or data[name].dtype.kind not in "US":
            raise ValueError(f"Invalid {name}")
    if len(set(data["donor_ids"])) != 92 or len(set(data["sample_ids"])) != 92:
        raise ValueError("Donors and samples must be unique")
    manifest = json.loads((ROOT / "data/confirmation/stephenson_citeseq/source_manifest_v1.json").read_text())
    expected = {(row["donor"], row["sample"], row["role"]) for row in manifest["samples"]
                if row["role"] in {"calibration", "pilot", "held_site"}}
    if set(zip(data["donor_ids"], data["sample_ids"], data["roles"])) != expected:
        raise ValueError("Donor allocation differs from the original manifest")
    if not np.all(tables.sum(axis=-2) == 256):
        raise ValueError("Original median-rank protein margins must equal 256")
    if np.any(informative(tables).sum(axis=(1, 2)) < 64):
        raise ValueError("An original donor has fewer than 64 informative pairs")
    return data


def fit_common_ridge_by_score(tables: np.ndarray) -> dict:
    """Solve the same graph-zero penalized likelihood by its monotone scores."""
    null = evaluate_conditional_log_odds(np.zeros((9, 9)), tables,
                                        minimum_informative_donors=2)
    scale = float(np.median(null.data_precision[null.data_precision > 0]))
    ridge = 0.01 * scale
    coordinates = []
    for panel in np.moveaxis(tables.reshape(len(tables), 81, 2, 2), 1, 0):
        def score(value):
            return float(evaluate_conditional_log_odds(
                np.asarray(value), panel, ridge_penalty=ridge).gradient)

        low, high = -1.0, 1.0
        while score(low) > 0:
            low *= 2
        while score(high) < 0:
            high *= 2
        coordinates.append(brentq(score, low, high, xtol=1e-12, rtol=1e-14))
    mu = np.array(coordinates).reshape(9, 9)
    final = evaluate_conditional_log_odds(mu, tables, ridge_penalty=ridge,
                                         minimum_informative_donors=2)
    gradient = float(np.max(np.abs(final.gradient)))
    condition = float(np.linalg.cond(final.hessian))
    if gradient > 1e-8 or condition > 1e12:
        raise CouplingEstimationRefusal("Common-ridge score-root certificate failed")
    return {"status": "FITTED", "mu": mu, "boundary": np.zeros((9, 9), dtype=int),
            "certificate": {"solver": "separable_score_brentq", "converged": True,
                            "gradient_norm": gradient, "condition_number": condition,
                            "penalty_scale": scale, "effective_ridge": ridge}}


def fit_conditional(tables: np.ndarray, method: str) -> dict:
    try:
        if method == "common_unpenalized":
            fitted = classical._common_effect_exact_cmle(tables)
            return {"status": "FITTED", "mu": fitted.log_odds,
                    "boundary": fitted.boundary, "certificate": fitted.certificate}
        if method == "hierarchical":
            fitted = fit_hierarchical_conditional_log_odds(
                tables, np.eye(9), np.eye(9), heterogeneity_penalty=0.1,
                ridge_penalty=0.01, graph_penalty=0.0, minimum_informative_donors=2,
                maximum_condition_number=1e12, maximum_iterations=200, tolerance=1e-8,
            )
            mu = fitted.population_log_odds
        elif method == "common_ridge":
            fitted = fit_structured_conditional_log_odds(
                tables, np.eye(9), np.eye(9), ridge_penalty=0.01, graph_penalty=0.0,
                minimum_informative_donors=2, maximum_condition_number=1e12,
                maximum_iterations=500, tolerance=1e-8,
            )
            mu = fitted.log_odds
        else:
            raise ValueError(f"Unknown method: {method}")
        return {"status": "FITTED", "mu": mu, "boundary": np.zeros((9, 9), dtype=int),
                "certificate": {"iterations": fitted.iterations,
                                "converged": bool(fitted.converged),
                                "gradient_norm": fitted.gradient_norm}}
    except (CouplingEstimationRefusal, FloatingPointError, np.linalg.LinAlgError) as error:
        if method == "common_ridge" and str(error) == "conditional-likelihood optimizer did not converge":
            recovered = fit_common_ridge_by_score(tables)
            recovered["certificate"]["initial_failure"] = str(error)
            return recovered
        return {"status": "FIT_FAILED", "error": str(error)}


def _source_csv(tables: np.ndarray, donors: np.ndarray, pairs=None) -> str:
    selected = set(range(81)) if pairs is None else set(pairs)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["donor", "pair", "a", "b", "c", "d"])
    for donor, panel in zip(donors, tables):
        for pair, table in enumerate(panel.reshape(81, 2, 2)):
            if pair in selected:
                writer.writerow([str(donor), pair, *table.ravel().tolist()])
    return stream.getvalue()


def fit_random_effects(tables: np.ndarray, donors: np.ndarray, cache: Path) -> dict:
    """Cache complete R output rows, including failures, without mixing inputs."""
    cache.mkdir(parents=True, exist_ok=True)
    content = _source_csv(tables, donors)
    binding = {"input_sha256": hashlib.sha256(content.encode()).hexdigest(),
               "r_script_sha256": sha256(R_SCRIPT), "integration_tolerance": 1e-8}
    binding_path = cache / "binding.json"
    if binding_path.exists():
        if json.loads(binding_path.read_text()) != binding:
            raise ValueError("Random-effects cache belongs to a different input or script")
    else:
        binding_path.write_text(json.dumps(binding, indent=2) + "\n")
        (cache / "source_tables.csv").write_text(content)
    rows = {}
    existing = sorted(cache.glob("fits_*.csv"))
    for path in existing:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                pair = int(row["pair"])
                if pair in rows:
                    raise ValueError("Duplicate pair in random-effects cache")
                rows[pair] = row
    missing = set(range(81)) - set(rows)
    if missing:
        rscript = shutil.which("Rscript")
        if not rscript:
            return {"status": "FIT_FAILED", "error": "Rscript is unavailable"}
        processes = []
        workers = min(4, len(missing))
        for worker in range(workers):
            chunk = len(existing) + worker
            input_path, output_path = cache / f"input_{chunk}.csv", cache / f"fits_{chunk}.csv"
            input_path.write_text(_source_csv(tables, donors, sorted(missing)[worker::workers]))
            log = (cache / f"worker_{chunk}.log").open("w")
            process = subprocess.Popen([rscript, str(R_SCRIPT), str(input_path), str(output_path)],
                                       stdout=log, stderr=subprocess.STDOUT)
            processes.append((process, output_path, log))
        print(f"Fitting {len(missing)} remaining pairs with {workers} R workers in {cache}", flush=True)
        return_codes = []
        try:
            for process, output_path, log in processes:
                return_codes.append(process.wait())
                log.close()
                if output_path.exists():
                    with output_path.open(newline="") as stream:
                        rows.update({int(row["pair"]): row for row in csv.DictReader(stream)})
        finally:
            for process, _, log in processes:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
                log.close()
        if any(return_codes):
            return {"status": "FIT_FAILED", "error": f"R workers exited with {return_codes}",
                    "pair_fits": [rows[pair] for pair in sorted(rows)]}
    failed = [pair for pair in range(81) if pair not in rows or rows[pair]["status"] != "OK"]
    result = {"status": "FIT_FAILED" if failed else "FITTED",
              "failed_pairs": failed, "pair_fits": [rows[pair] for pair in sorted(rows)],
              "binding": binding}
    imported_provenance = cache / "imported_provenance.json"
    if imported_provenance.exists():
        result["imported_fit_provenance"] = json.loads(imported_provenance.read_text())
    if not failed:
        result.update(mu=np.array([float(rows[p]["mu"]) for p in range(81)]).reshape(9, 9),
                      tau2=np.array([float(rows[p]["tau2"]) if rows[p]["tau2"] else np.nan
                                     for p in range(81)]).reshape(9, 9),
                      boundary=np.array([int(rows[p]["boundary"]) for p in range(81)]).reshape(9, 9))
    return result


def score_model(fitted: dict, tables: np.ndarray, donors: np.ndarray, alpha: float,
                *, mixture: bool = False, intervals: bool = False) -> list[dict]:
    rows = []
    cache = {}
    for donor, panel in zip(donors, tables):
        mask = informative(panel).ravel()
        row = {"donor": str(donor), "informative_pairs": int(mask.sum()), "loss": None}
        if fitted["status"] != "FITTED":
            rows.append({**row, "status": "FIT_FAILED"})
            continue
        losses, covered, widths = [], [], []
        failures = []
        for pair, (table, active) in enumerate(zip(panel.reshape(-1, 2, 2), mask)):
            if not active:
                losses.append(None)
                covered.append(None)
                widths.append(None)
                continue
            mu = float(np.asarray(fitted["mu"]).ravel()[pair]) * alpha
            tau2 = float(np.asarray(fitted["tau2"]).ravel()[pair]) * alpha**2 if mixture else 0.0
            boundary = int(np.asarray(fitted["boundary"]).ravel()[pair])
            margins = (table.sum(axis=1), table.sum(axis=0))
            try:
                if boundary:
                    prediction = classical._boundary_table(boundary, *margins)
                    distribution = None
                else:
                    key = (pair, int(margins[0][0]), int(margins[1][0]), int(table.sum()))
                    if key not in cache:
                        cache[key] = normal_mixture_prediction(mu, tau2, *margins)
                    distribution = cache[key]
                    prediction = distribution.mean_table
                positive = table > 0
                if np.any(prediction[positive] <= 0):
                    loss = float("inf")
                else:
                    loss = float(2 / table.sum() * np.sum(
                        table[positive] * np.log(table[positive] / prediction[positive])))
                losses.append(loss)
                if intervals:
                    low, high = ((int(prediction[0, 0]), int(prediction[0, 0]))
                                 if boundary else distribution.count_interval())
                    covered.append(bool(low <= table[0, 0] <= high))
                    widths.append(high - low)
                else:
                    covered.append(None)
                    widths.append(None)
            except (CouplingEstimationRefusal, FloatingPointError) as error:
                failures.append({"pair": pair, "error": str(error)})
                losses.append(None)
                covered.append(None)
                widths.append(None)
        row.update(pair_losses=losses, interval_covered=covered, interval_width=widths)
        if failures:
            row.update(status="PREDICTION_FAILED", failures=failures)
        elif not mask.any():
            row.update(status="NO_INFORMATIVE_PAIRS")
        else:
            loss = float(np.mean([x for x in losses if x is not None]))
            row.update(status="SCORED" if np.isfinite(loss) else "INFINITE_DEVIANCE", loss=loss)
            if intervals:
                row.update(coverage=float(np.mean([x for x in covered if x is not None])),
                           mean_interval_width=float(np.mean([x for x in widths if x is not None])))
        rows.append(row)
    return rows


def select_alpha(fitted: dict, pilot: np.ndarray, donors: np.ndarray, *, mixture=False) -> dict:
    candidates = []
    for alpha in ALPHAS:
        scored = score_model(fitted, pilot, donors, alpha, mixture=mixture)
        valid = all(row["status"] == "SCORED" for row in scored)
        candidates.append({"alpha": alpha, "status": "SCORED" if valid else "UNAVAILABLE",
                           "mean_loss": float(np.mean([row["loss"] for row in scored])) if valid else None})
    valid = [row for row in candidates if row["status"] == "SCORED"]
    if not valid:
        return {"status": "UNAVAILABLE", "candidates": candidates}
    selected = min(valid, key=lambda row: (row["mean_loss"], row["alpha"]))
    return {"status": "SELECTED", "alpha": selected["alpha"], "candidates": candidates}


def compare(primary: list[dict], comparator: list[dict], label: str) -> dict:
    if [r["donor"] for r in primary] != [r["donor"] for r in comparator]:
        raise ValueError("Unpaired donor comparison")
    if any(row["status"] == "INFINITE_DEVIANCE" for row in primary + comparator):
        return {"status": "INFINITE_DEVIANCE", "bootstrap_performed": False,
                "primary_infinite_donors": [r["donor"] for r in primary if r["status"] == "INFINITE_DEVIANCE"],
                "comparator_infinite_donors": [r["donor"] for r in comparator if r["status"] == "INFINITE_DEVIANCE"]}
    if any(row["status"] != "SCORED" for row in primary + comparator):
        return {"status": "UNAVAILABLE", "reason": "Requires every original held donor and pair"}
    result = classical._comparison(
        [row["donor"] for row in primary], np.array([row["loss"] for row in primary]),
        np.array([row["loss"] for row in comparator]), label=label,
    )
    return {"status": "SCORED", **result}


def verify_original(rows: list[dict]) -> float:
    reference = json.loads((ROOT / "results/stephenson_citeseq_confirmation.json").read_text())
    expected = {r["donor"]: r for r in reference["donor_results"]}
    if set(expected) != {r["donor"] for r in rows} or any(r["status"] != "SCORED" for r in rows):
        raise ValueError("Original held-donor scores are incomplete")
    error = max(abs(row["loss"] - expected[row["donor"]]["losses"]["primary"]) for row in rows)
    if error > 2e-8:
        raise ValueError(f"Recovered original loss differs from frozen result: {error}")
    if any(row["informative_pairs"] != expected[row["donor"]]["informative_pairs"] for row in rows):
        raise ValueError("Original informative masks differ")
    return error


def analyze(data: dict[str, np.ndarray], cache: Path) -> dict:
    tables, donors, roles = data["tables"], data["donor_ids"], data["roles"]
    calibration, pilot, source, held = roles == "calibration", roles == "pilot", roles != "held_site", roles == "held_site"
    fits = {phase: {} for phase in ("calibration", "source")}
    for phase, mask in (("calibration", calibration), ("source", source)):
        for family in ("hierarchical", "common_unpenalized", "common_ridge"):
            print(f"Fitting {family} on {phase}", flush=True)
            fits[phase][family] = fit_conditional(tables[mask], family)
    original = score_model(fits["source"]["hierarchical"], tables[held], donors[held], 1.0)
    original_error = verify_original(original)
    print(f"Original donor losses reproduced: max error {original_error:.3g}", flush=True)
    for phase, mask in (("calibration", calibration), ("source", source)):
        fits[phase]["random_effects"] = fit_random_effects(tables[mask], donors[mask], cache / phase)
    selections, methods = {}, {"original_hierarchy": {"alpha": 1.0, "donor_results": original}}
    families = [("hierarchy_selected", "hierarchical", False),
                ("common_unpenalized", "common_unpenalized", False),
                ("common_ridge", "common_ridge", False),
                ("random_effects_plugin", "random_effects", False),
                ("random_effects_mixture", "random_effects", True)]
    for name, family, mixture in families:
        print(f"Selecting and scoring {name}", flush=True)
        selection = select_alpha(fits["calibration"][family], tables[pilot], donors[pilot], mixture=mixture)
        selections[name] = selection
        if selection["status"] == "SELECTED":
            alpha = selection["alpha"]
            scores = score_model(fits["source"][family], tables[held], donors[held], alpha,
                                 mixture=mixture, intervals=family == "random_effects")
            methods[name] = {"alpha": alpha, "donor_results": scores}
        else:
            methods[name] = {"status": "UNAVAILABLE", "reason": "Calibration selection failed"}
    for mixture in (False, True):
        name = "random_effects_mixture_alpha1" if mixture else "random_effects_plugin_alpha1"
        methods[name] = {"alpha": 1.0, "donor_results": score_model(
            fits["source"]["random_effects"], tables[held], donors[held], 1.0,
            mixture=mixture, intervals=True)}
    for method in methods.values():
        scored = method.get("donor_results", [])
        if len(scored) == int(held.sum()) and all(r["status"] == "SCORED" for r in scored):
            method.update(status="SCORED", mean_loss=float(np.mean([r["loss"] for r in scored])))
            if "coverage" in scored[0]:
                method.update(donor_equal_coverage=float(np.mean([r["coverage"] for r in scored])),
                              donor_equal_interval_width=float(np.mean([r["mean_interval_width"] for r in scored])))
        elif len(scored) == int(held.sum()) and all(r["status"] in {"SCORED", "INFINITE_DEVIANCE"} for r in scored):
            method.update(status="INFINITE_DEVIANCE", mean_loss=float("inf"))
        else:
            method["status"] = "UNAVAILABLE"
    comparisons = {}
    for name, method in methods.items():
        if name != "original_hierarchy" and "donor_results" in method:
            comparisons[f"original_vs_{name}"] = compare(original, method["donor_results"], f"stephenson-reanalysis-{name}")
    for suffix in ("", "_alpha1"):
        plugin, mixture = methods[f"random_effects_plugin{suffix}"], methods[f"random_effects_mixture{suffix}"]
        if "donor_results" in plugin and "donor_results" in mixture:
            comparisons[f"mixture_vs_plugin{suffix}"] = compare(
                mixture["donor_results"], plugin["donor_results"], f"stephenson-mixture-plugin{suffix}")
    return {"schema": "stephenson-predictive-reanalysis-v1", "confirmatory": False,
            "original_loss_max_absolute_error": original_error,
            "roles": dict(zip(*np.unique(roles, return_counts=True))),
            "alpha_grid": ALPHAS, "bootstrap_draws": BOOTSTRAPS,
            "predictive_interval": "95% equal-tail upper-left count; fitted source parameters fixed",
            "fits": fits, "pilot_selection": selections, "methods": methods, "comparisons": comparisons}


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [serializable(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        return float(value) if np.isfinite(value) else ("+Infinity" if value > 0 else "-Infinity")
    return value


def write_results(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(serializable(payload), indent=2, allow_nan=False) + "\n")
    with output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "donor", "pair", "status", "deviance", "interval_covered", "interval_width"])
        for method, result in payload["methods"].items():
            for row in result.get("donor_results", []):
                for pair, loss in enumerate(row.get("pair_losses", [None] * 81)):
                    writer.writerow([method, row["donor"], pair, row["status"], loss,
                                     row.get("interval_covered", [None] * 81)[pair],
                                     row.get("interval_width", [None] * 81)[pair]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/stephenson-r-fit-cache"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Output already exists; preserve completed analyses")
    data = load_tables(args.tables)
    result = analyze(data, args.cache)
    result["bindings"] = {"tables_sha256": sha256(args.tables), "runner_sha256": sha256(Path(__file__)),
                          "protocol_sha256": sha256(ROOT / "docs/FIXED_MARGIN_PREDICTIVE_REANALYSIS.md")}
    write_results(result, args.output)
    print(json.dumps({name: {key: row.get(key) for key in ("status", "mean_loss", "donor_equal_coverage")}
                      for name, row in result["methods"].items()}, indent=2))


if __name__ == "__main__":
    main()
