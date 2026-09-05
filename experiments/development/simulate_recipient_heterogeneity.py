"""Test population plug-in predictions when recipient donors also vary.

The source panels and fixed fitting settings match the original simulation.
Known-parameter predictions diagnose the decision rule; they are not fitted
competitors. Recipient outcomes are used only for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import gammaln, logsumexp

from experiments.development import simulate_exact_conditional_heterogeneity as original


METHODS = (
    "hierarchical_exact",
    "common_effect_exact",
    "oracle_plugin",
    "oracle_mixture",
)
ROOT = Path(__file__).resolve().parents[2]


def fit_source(source):
    """Repeat the two original fits without computing unused statistic baselines."""
    hierarchical = original.fit_hierarchical_conditional_log_odds(
        source,
        original.INCIDENCE,
        original.INCIDENCE,
        heterogeneity_penalty=original.HETEROGENEITY_PENALTY,
        ridge_penalty=original.RIDGE_PENALTY,
        graph_penalty=original.GRAPH_PENALTY,
        minimum_informative_donors=2,
        maximum_iterations=200,
        tolerance=original.HIERARCHICAL_TOLERANCE,
    )
    common = original.fit_structured_conditional_log_odds(
        source,
        original.INCIDENCE,
        original.INCIDENCE,
        ridge_penalty=original.RIDGE_PENALTY,
        graph_penalty=original.GRAPH_PENALTY,
        minimum_informative_donors=2,
        maximum_iterations=500,
        tolerance=original.COMMON_TOLERANCE,
    )
    refined, gradient, condition, iterations = original._certify_common_effect(
        source,
        common.log_odds,
        common.ridge_penalty * common.penalty_scale,
    )
    return {
        "coordinates": {
            "hierarchical_exact": hierarchical.population_log_odds,
            "common_effect_exact": refined,
        },
        "fit_diagnostics": {
            "hierarchical_scaled_gradient_norm": hierarchical.scaled_gradient_norm,
            "hierarchical_iterations": hierarchical.iterations,
            "common_gradient_norm": gradient,
            "common_condition_number": condition,
            "common_refinement_iterations": iterations,
        },
    }


def mixture_mean(mu, sd, rows, columns, order=40):
    """Integrate exact conditional means over known normal donor variation."""
    rows, columns = np.asarray(rows), np.asarray(columns)
    total = rows.sum(axis=-1)
    lower = np.maximum(0, rows[..., 0] + columns[..., 0] - total)
    upper = np.minimum(rows[..., 0], columns[..., 0])
    support = np.arange(int(total.max()) + 1)
    a = support[None, :]
    r = rows.reshape(-1, 2)[:, 0, None]
    c = columns.reshape(-1, 2)[:, 0, None]
    n = total.reshape(-1, 1)
    feasible = (a >= lower.reshape(-1, 1)) & (a <= upper.reshape(-1, 1))
    weights = (
        gammaln(c + 1)
        - gammaln(a + 1)
        - gammaln(c - a + 1)
        + gammaln(n - c + 1)
        - gammaln(r - a + 1)
        - gammaln(n - c - r + a + 1)
    )
    weights = np.where(feasible, weights, -np.inf)
    means = np.broadcast_to(mu, total.shape).reshape(-1)
    nodes, quadrature = hermgauss(order)
    expected = np.zeros(means.shape)
    for node, weight in zip(nodes, quadrature / np.sqrt(np.pi)):
        log_probability = weights + (means + np.sqrt(2) * sd * node)[:, None] * a
        probability = np.exp(
            log_probability - logsumexp(log_probability, axis=1)[:, None]
        )
        expected += weight * (probability * support).sum(axis=1)
    expected = expected.reshape(total.shape)
    return np.stack(
        (
            expected,
            rows[..., 0] - expected,
            columns[..., 0] - expected,
            total - rows[..., 0] - columns[..., 0] + expected,
        ),
        axis=-1,
    ).reshape(*total.shape, 2, 2)


def donor_losses(truth, prediction):
    terms = np.zeros_like(prediction, dtype=float)
    positive = truth > 0
    terms[positive] = truth[positive] * np.log(truth[positive] / prediction[positive])
    losses = 2 * terms.sum(axis=(-2, -1)) / truth.sum(axis=(-2, -1))
    return losses.reshape(len(truth), -1).mean(axis=1)


def summarize(records, draws, seed):
    if not records:
        return None
    values = np.array([[r["losses"][method] for method in METHODS] for r in records])
    indices = np.random.default_rng(seed).integers(
        len(records), size=(draws, len(records))
    )
    bootstrap = values[indices].mean(axis=1)
    output = {
        method: {
            "mean": float(values[:, i].mean()),
            "ci95": np.quantile(bootstrap[:, i], [0.025, 0.975]).tolist(),
        }
        for i, method in enumerate(METHODS)
    }
    comparisons = {}
    for primary, comparator in ((0, 1), (3, 2)):
        difference = bootstrap[:, primary] - bootstrap[:, comparator]
        reduction = 1 - bootstrap[:, primary] / bootstrap[:, comparator]
        comparisons[f"{METHODS[primary]}_vs_{METHODS[comparator]}"] = {
            "difference": float((values[:, primary] - values[:, comparator]).mean()),
            "difference_ci95": np.quantile(difference, [0.025, 0.975]).tolist(),
            "relative_reduction": float(
                1 - values[:, primary].mean() / values[:, comparator].mean()
            ),
            "relative_reduction_ci95": np.quantile(reduction, [0.025, 0.975]).tolist(),
        }
    return {"methods": output, "paired_comparisons": comparisons}


def run(replicates=128, bootstraps=20_000, seed=original.DEFAULT_SEED):
    scenarios = {}
    source_failures = []
    for label, source_sd in (("homogeneous", 0.0), ("heterogeneous", 1.2)):
        for recipient_sd in (0.0, 1.2):
            for shifted in (False, True):
                name = f"{label}_recipient_sd_{recipient_sd}_{'shifted' if shifted else 'same'}"
                scenarios[name] = {
                    "source_sd": source_sd,
                    "recipient_sd": recipient_sd,
                    "shifted_margins": shifted,
                    "records": [],
                    "failures": [],
                }
        for replicate in range(replicates):
            source = original._panel(
                original.POPULATION_LOG_ODDS,
                source_sd,
                original.SOURCE_ROW_PROPORTIONS,
                original.SOURCE_COLUMN_PROPORTIONS,
                original.SOURCE_DONORS,
                np.random.default_rng(
                    original._seed(f"{label}|{replicate}|source", seed)
                ),
            )
            try:
                fitted = fit_source(source)
            except (
                original.CouplingEstimationRefusal,
                FloatingPointError,
                np.linalg.LinAlgError,
            ) as error:
                failure = {
                    "replicate": replicate,
                    "source": label,
                    "reason": str(error),
                    "error_type": type(error).__name__,
                }
                source_failures.append(failure)
                for scenario in scenarios.values():
                    if scenario["source_sd"] == source_sd:
                        scenario["failures"].append(failure)
                continue
            for recipient_sd in (0.0, 1.2):
                for shifted in (False, True):
                    name = f"{label}_recipient_sd_{recipient_sd}_{'shifted' if shifted else 'same'}"
                    scenario = scenarios[name]
                    target = original._panel(
                        original.POPULATION_LOG_ODDS,
                        recipient_sd,
                        original.SHIFTED_ROW_PROPORTIONS
                        if shifted
                        else original.SOURCE_ROW_PROPORTIONS,
                        original.SHIFTED_COLUMN_PROPORTIONS
                        if shifted
                        else original.SOURCE_COLUMN_PROPORTIONS,
                        original.RECIPIENT_DONORS,
                        np.random.default_rng(
                            original._seed(f"{label}|{replicate}|target", seed)
                        ),
                    )
                    predictions = {
                        method: original._predict_exact(
                            fitted["coordinates"][method], target
                        )
                        for method in METHODS[:2]
                    }
                    predictions["oracle_plugin"] = original._predict_exact(
                        original.POPULATION_LOG_ODDS, target
                    )
                    rows, columns = target.sum(axis=-1), target.sum(axis=-2)
                    coarse = mixture_mean(
                        original.POPULATION_LOG_ODDS, recipient_sd, rows, columns, 40
                    )
                    fine = mixture_mean(
                        original.POPULATION_LOG_ODDS, recipient_sd, rows, columns, 80
                    )
                    error = float(np.max(np.abs(coarse - fine)))
                    if not np.isfinite(error) or error > 1e-8:
                        scenario["failures"].append(
                            {
                                "replicate": replicate,
                                "reason": "quadrature_nonconvergence",
                                "max_count_difference": error,
                            }
                        )
                        continue
                    predictions["oracle_mixture"] = fine
                    by_donor = {
                        method: donor_losses(target, value).tolist()
                        for method, value in predictions.items()
                    }
                    scenario["records"].append(
                        {
                            "replicate": replicate,
                            "losses": {
                                method: float(np.mean(value))
                                for method, value in by_donor.items()
                            },
                            "donor_losses": by_donor,
                            "quadrature_max_count_difference": error,
                            "fit_diagnostics": fitted["fit_diagnostics"],
                        }
                    )
            if (replicate + 1) % 16 == 0:
                print(f"{label}: {replicate + 1}/{replicates}", flush=True)
    for name, scenario in scenarios.items():
        scenario["summary"] = summarize(
            scenario["records"], bootstraps, original._seed(name, seed)
        )
        scenario["requested_replicates"] = replicates
        scenario["successful_replicates"] = len(scenario["records"])
    return {
        "analysis_role": "post_review_simulation_extension",
        "seed": seed,
        "replicates_per_scenario": replicates,
        "bootstrap_draws": bootstraps,
        "source_donors": original.SOURCE_DONORS,
        "recipient_donors": original.RECIPIENT_DONORS,
        "table_total": original.TABLE_TOTAL,
        "pairs": 16,
        "source_generator": "unchanged original seeds, source panels, and fixed fitting settings",
        "recipient_generator": "independent normal effects; common streams across recipient SD and margin scenarios",
        "oracle_status": "known population mean and recipient SD; diagnostic, not learned method evidence",
        "quadrature": {"orders": [40, 80], "absolute_count_tolerance": 1e-8},
        "source_fit_failures": source_failures,
        "scenarios": scenarios,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=128)
    parser.add_argument("--bootstraps", type=int, default=20_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/development/recipient_heterogeneity_reanalysis.json",
    )
    args = parser.parse_args()
    if args.replicates < 1 or args.bootstraps < 1:
        parser.error("replicates and bootstraps must be positive")
    result = run(args.replicates, args.bootstraps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with args.output.with_suffix(".tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "scenario",
                "source_sd",
                "recipient_sd",
                "shifted_margins",
                "successful_replicates",
                "failed_replicates",
                "method",
                "mean_deviance",
                "ci_low",
                "ci_high",
            ]
        )
        for name, scenario in result["scenarios"].items():
            if scenario["summary"] is None:
                continue
            for method, summary in scenario["summary"]["methods"].items():
                writer.writerow(
                    [
                        name,
                        scenario["source_sd"],
                        scenario["recipient_sd"],
                        scenario["shifted_margins"],
                        scenario["successful_replicates"],
                        len(scenario["failures"]),
                        method,
                        summary["mean"],
                        *summary["ci95"],
                    ]
                )
    print(args.output)


if __name__ == "__main__":
    main()
