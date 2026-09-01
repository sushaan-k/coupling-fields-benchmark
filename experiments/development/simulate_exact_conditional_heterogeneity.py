"""Ground-truth ablation for binary conditional coupling transfer.

The simulation targets the population log odds in a panel of independent
fixed-margin 2x2 tables. The sum over entity pairs is therefore a pairwise
composite objective, not a coherent multivariate likelihood for shared cells.
Source donors have known Normal log-odds heterogeneity. Recipient tables are
independent draws at the population field and expose only their margins to the
estimators.

Three fixed estimator families are compared without outcome-based tuning:

* the production donor-heterogeneity-aware exact conditional estimator;
* a common-effect exact conditional estimator with the same ridge penalty;
* donor-equal signed-root Poisson-deviance residual transfer in raw and
  exact-null-centered form.

The 2x2 tables are sampled exactly from Fisher's noncentral hypergeometric
law. A 2x2 factorial ablation crosses donor heterogeneity with recipient-margin
shift. Results include population-field RMSE, held recipient-table deviance,
paired simulation-replicate bootstrap intervals, and every fit failure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import gammaln, logsumexp

from mapreg.heterogeneity_adaptive_coupling import (
    CouplingEstimationRefusal,
    centered_classical_coordinate,
    evaluate_conditional_log_odds,
    expected_binary_table_from_log_odds,
    fit_structured_conditional_log_odds,
    signed_deviance_coordinate,
)
from mapreg.hierarchical_conditional_coupling import (
    fit_hierarchical_conditional_log_odds,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = (
    ROOT
    / "results/development/exact_conditional_heterogeneity_simulation_v1.json"
)
DEFAULT_TSV = (
    ROOT
    / "results/development/exact_conditional_heterogeneity_simulation_v1.tsv"
)

DEFAULT_SEED = 20260828
DEFAULT_REPLICATES = 128
DEFAULT_BOOTSTRAPS = 20_000
SOURCE_DONORS = 12
RECIPIENT_DONORS = 12
TABLE_TOTAL = 128
MARGIN_JITTER_SD = 0.035
HETEROGENEITY_SD = 1.2
HETEROGENEITY_PENALTY = 0.1
RIDGE_PENALTY = 0.01
GRAPH_PENALTY = 0.0
HIERARCHICAL_TOLERANCE = 1e-7
COMMON_TOLERANCE = 1e-5
COMMON_CERTIFICATE_TOLERANCE = 1e-10

POPULATION_LOG_ODDS = np.asarray(
    [
        [-1.20, -0.75, -0.30, 0.20],
        [-0.85, -0.35, 0.15, 0.55],
        [-0.40, 0.05, 0.50, 0.90],
        [-0.10, 0.35, 0.80, 1.20],
    ],
    dtype=float,
)
SOURCE_ROW_PROPORTIONS = np.asarray(
    [
        [0.42, 0.45, 0.48, 0.51],
        [0.44, 0.47, 0.50, 0.53],
        [0.46, 0.49, 0.52, 0.55],
        [0.48, 0.51, 0.54, 0.58],
    ],
    dtype=float,
)
SOURCE_COLUMN_PROPORTIONS = np.asarray(
    [
        [0.58, 0.55, 0.52, 0.49],
        [0.56, 0.53, 0.50, 0.47],
        [0.54, 0.51, 0.48, 0.45],
        [0.52, 0.49, 0.46, 0.42],
    ],
    dtype=float,
)
SHIFTED_ROW_PROPORTIONS = np.asarray(
    [
        [0.16, 0.24, 0.72, 0.84],
        [0.22, 0.78, 0.18, 0.76],
        [0.74, 0.20, 0.82, 0.26],
        [0.86, 0.70, 0.28, 0.14],
    ],
    dtype=float,
)
SHIFTED_COLUMN_PROPORTIONS = np.asarray(
    [
        [0.82, 0.25, 0.75, 0.18],
        [0.28, 0.80, 0.20, 0.72],
        [0.76, 0.22, 0.84, 0.16],
        [0.14, 0.70, 0.30, 0.86],
    ],
    dtype=float,
)
INCIDENCE = np.eye(POPULATION_LOG_ODDS.shape[0], dtype=float)

METHODS = (
    "hierarchical_exact",
    "common_effect_exact",
    "poisson_deviance_raw",
    "poisson_deviance_centered",
)
EXACT_METHODS = ("hierarchical_exact", "common_effect_exact")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed(label: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}|{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _conditional_sample(
    log_odds: float,
    row_zero: int,
    column_zero: int,
    total: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """Draw one exact fixed-margin 2x2 table."""

    lower = max(0, row_zero + column_zero - total)
    upper = min(row_zero, column_zero)
    support = np.arange(lower, upper + 1, dtype=int)
    log_weight = (
        gammaln(column_zero + 1)
        - gammaln(support + 1)
        - gammaln(column_zero - support + 1)
        + gammaln(total - column_zero + 1)
        - gammaln(row_zero - support + 1)
        - gammaln(total - column_zero - row_zero + support + 1)
        + float(log_odds) * support
    )
    probability = np.exp(log_weight - logsumexp(log_weight))
    upper_left = int(generator.choice(support, p=probability))
    return np.asarray(
        [
            [upper_left, row_zero - upper_left],
            [
                column_zero - upper_left,
                total - row_zero - column_zero + upper_left,
            ],
        ],
        dtype=np.int64,
    )


def _panel(
    population_log_odds: np.ndarray,
    heterogeneity_sd: float,
    row_proportions: np.ndarray,
    column_proportions: np.ndarray,
    donors: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """Generate independent entity-pair tables for a donor panel."""

    tables = np.empty(
        (donors, *population_log_odds.shape, 2, 2), dtype=np.int64
    )
    for donor in range(donors):
        donor_log_odds = population_log_odds + heterogeneity_sd * generator.normal(
            size=population_log_odds.shape
        )
        for entity in np.ndindex(population_log_odds.shape):
            row_probability = row_proportions[entity] + generator.normal(
                0.0, MARGIN_JITTER_SD
            )
            column_probability = column_proportions[entity] + generator.normal(
                0.0, MARGIN_JITTER_SD
            )
            row_zero = int(
                np.clip(np.rint(TABLE_TOTAL * row_probability), 12, TABLE_TOTAL - 12)
            )
            column_zero = int(
                np.clip(
                    np.rint(TABLE_TOTAL * column_probability),
                    12,
                    TABLE_TOTAL - 12,
                )
            )
            tables[(donor, *entity)] = _conditional_sample(
                donor_log_odds[entity],
                row_zero,
                column_zero,
                TABLE_TOTAL,
                generator,
            )
    return tables


def _canonical_table(rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    total = int(rows.sum())
    upper_left = max(0, int(rows[0] + columns[0] - total))
    return np.asarray(
        [
            [upper_left, int(rows[0] - upper_left)],
            [
                int(columns[0] - upper_left),
                int(rows[1] - columns[0] + upper_left),
            ],
        ],
        dtype=np.int64,
    )


def _classical_table(
    coordinate: float,
    rows: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    """Invert one signed-root independence coordinate at fixed margins."""

    total = float(rows.sum())
    lower = float(max(0, rows[0] + columns[0] - total))
    upper = float(min(rows[0], columns[0]))
    if upper <= lower:
        return _canonical_table(rows, columns).astype(float)
    epsilon = min(1e-10, 0.25 * (upper - lower))
    expected = np.outer(rows, columns) / total

    def statistic(value: float) -> float:
        table = np.asarray(
            [
                [value, rows[0] - value],
                [columns[0] - value, rows[1] - columns[0] + value],
            ]
        )
        positive = table > 0.0
        terms = np.zeros((2, 2), dtype=float)
        terms[positive] = table[positive] * np.log(
            table[positive] / expected[positive]
        )
        determinant = table[0, 0] * table[1, 1] - table[0, 1] * table[1, 0]
        return math.copysign(
            math.sqrt(max(2.0 * float(terms.sum()), 0.0)), determinant
        )

    left = lower + epsilon
    right = upper - epsilon
    target = min(max(float(coordinate), statistic(left)), statistic(right))
    for _ in range(80):
        midpoint = 0.5 * (left + right)
        if statistic(midpoint) < target:
            left = midpoint
        else:
            right = midpoint
    upper_left = 0.5 * (left + right)
    upper_left = min(max(float(upper_left), lower + epsilon), upper - epsilon)
    return np.asarray(
        [
            [upper_left, rows[0] - upper_left],
            [columns[0] - upper_left, rows[1] - columns[0] + upper_left],
        ]
    )


def _pooled_residual_coordinate(tables: np.ndarray, centered: bool) -> np.ndarray:
    values = np.empty(tables.shape[:3], dtype=float)
    for index in np.ndindex(tables.shape[:3]):
        table = tables[index]
        coordinate = (
            centered_classical_coordinate(
                table, statistic="deviance"
            ).centered_coordinate
            if centered
            else signed_deviance_coordinate(table)
        )
        values[index] = coordinate / math.sqrt(float(table.sum()))
    return values.mean(axis=0)


def _predict_exact(log_odds: np.ndarray, target: np.ndarray) -> np.ndarray:
    predicted = np.empty(target.shape, dtype=float)
    rows = target.sum(axis=-1)
    columns = target.sum(axis=-2)
    for index in np.ndindex(target.shape[:-2]):
        predicted[index] = expected_binary_table_from_log_odds(
            float(log_odds[index[-2:]]), rows[index], columns[index]
        )
    return predicted


def _predict_residual(
    pooled_coordinate: np.ndarray,
    target: np.ndarray,
    centered: bool,
) -> np.ndarray:
    predicted = np.empty(target.shape, dtype=float)
    rows = target.sum(axis=-1)
    columns = target.sum(axis=-2)
    for index in np.ndindex(target.shape[:-2]):
        coordinate = pooled_coordinate[index[-2:]] * math.sqrt(
            float(target[index].sum())
        )
        if centered:
            coordinate += centered_classical_coordinate(
                _canonical_table(rows[index], columns[index]), statistic="deviance"
            ).null_mean_coordinate
        predicted[index] = _classical_table(
            float(coordinate), rows[index], columns[index]
        )
    return predicted


def _mean_deviance(truth: np.ndarray, prediction: np.ndarray) -> float:
    positive = truth > 0
    terms = np.zeros_like(prediction, dtype=float)
    terms[positive] = truth[positive] * np.log(
        truth[positive] / prediction[positive]
    )
    per_table = 2.0 * terms.sum(axis=(-2, -1)) / truth.sum(axis=(-2, -1))
    return float(per_table.mean())


def _certify_common_effect(
    source: np.ndarray, initial: np.ndarray, effective_ridge: float
) -> tuple[np.ndarray, float, float, int]:
    """Refine the common-effect fit to an explicit gradient certificate."""

    log_odds = np.asarray(initial, dtype=float).copy()
    for iteration in range(21):
        evaluation = evaluate_conditional_log_odds(
            log_odds,
            source,
            ridge_penalty=effective_ridge,
            graph_penalty=0.0,
            minimum_informative_donors=2,
        )
        gradient_norm = float(np.max(np.abs(evaluation.gradient)))
        if gradient_norm <= COMMON_CERTIFICATE_TOLERANCE:
            condition_number = float(np.linalg.cond(evaluation.hessian))
            return log_odds, gradient_norm, condition_number, iteration
        step = -np.linalg.solve(
            evaluation.hessian, evaluation.gradient.ravel(order="C")
        ).reshape(log_odds.shape)
        directional_derivative = float(np.sum(evaluation.gradient * step))
        step_size = 1.0
        for _ in range(48):
            candidate = log_odds + step_size * step
            candidate_evaluation = evaluate_conditional_log_odds(
                candidate,
                source,
                ridge_penalty=effective_ridge,
                graph_penalty=0.0,
                minimum_informative_donors=2,
            )
            candidate_gradient = float(
                np.max(np.abs(candidate_evaluation.gradient))
            )
            armijo = candidate_evaluation.objective <= (
                evaluation.objective + 1e-4 * step_size * directional_derivative
            )
            numerical_descent = (
                candidate_gradient < gradient_norm
                and candidate_evaluation.objective
                <= evaluation.objective
                + 1e-12 * max(1.0, abs(evaluation.objective))
            )
            if armijo or numerical_descent:
                log_odds = candidate
                break
            step_size *= 0.5
        else:
            raise CouplingEstimationRefusal(
                "common-effect Newton refinement missed a descent step"
            )
    raise CouplingEstimationRefusal(
        "common-effect exact fit missed its gradient certificate"
    )


def _fit_source(source: np.ndarray) -> dict[str, Any]:
    hierarchical = fit_hierarchical_conditional_log_odds(
        source,
        INCIDENCE,
        INCIDENCE,
        heterogeneity_penalty=HETEROGENEITY_PENALTY,
        ridge_penalty=RIDGE_PENALTY,
        graph_penalty=GRAPH_PENALTY,
        minimum_informative_donors=2,
        maximum_iterations=200,
        tolerance=HIERARCHICAL_TOLERANCE,
    )
    common = fit_structured_conditional_log_odds(
        source,
        INCIDENCE,
        INCIDENCE,
        ridge_penalty=RIDGE_PENALTY,
        graph_penalty=GRAPH_PENALTY,
        minimum_informative_donors=2,
        maximum_iterations=500,
        tolerance=COMMON_TOLERANCE,
    )
    common_log_odds, common_gradient, common_condition, refinement_iterations = (
        _certify_common_effect(
            source,
            common.log_odds,
            common.ridge_penalty * common.penalty_scale,
        )
    )
    return {
        "coordinates": {
            "hierarchical_exact": hierarchical.population_log_odds,
            "common_effect_exact": common_log_odds,
            "poisson_deviance_raw": _pooled_residual_coordinate(source, False),
            "poisson_deviance_centered": _pooled_residual_coordinate(source, True),
        },
        "field_rmse": {
            "hierarchical_exact": float(
                np.sqrt(
                    np.mean(
                        np.square(
                            hierarchical.population_log_odds - POPULATION_LOG_ODDS
                        )
                    )
                )
            ),
            "common_effect_exact": float(
                np.sqrt(np.mean(np.square(common_log_odds - POPULATION_LOG_ODDS)))
            ),
        },
        "fit_diagnostics": {
            "hierarchical_scaled_gradient_norm": hierarchical.scaled_gradient_norm,
            "hierarchical_iterations": hierarchical.iterations,
            "hierarchical_schur_condition_number": (
                hierarchical.schur_condition_number
            ),
            "common_gradient_norm": common_gradient,
            "common_initializer_iterations": common.iterations,
            "common_refinement_iterations": refinement_iterations,
            "common_condition_number": common_condition,
        },
    }


def _score_target(fitted: dict[str, Any], target: np.ndarray) -> dict[str, float]:
    coordinates = fitted["coordinates"]
    predictions = {
        "hierarchical_exact": _predict_exact(
            coordinates["hierarchical_exact"], target
        ),
        "common_effect_exact": _predict_exact(
            coordinates["common_effect_exact"], target
        ),
        "poisson_deviance_raw": _predict_residual(
            coordinates["poisson_deviance_raw"], target, False
        ),
        "poisson_deviance_centered": _predict_residual(
            coordinates["poisson_deviance_centered"], target, True
        ),
    }
    return {
        method: _mean_deviance(target, predictions[method]) for method in METHODS
    }


def _bootstrap_summary(
    replicate_metrics: list[dict[str, Any]],
    metric: str,
    methods: tuple[str, ...],
    draws: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    values = {
        method: np.asarray(
            [record[metric][method] for record in replicate_metrics], dtype=float
        )
        for method in methods
    }
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0, len(replicate_metrics), size=(draws, len(replicate_metrics))
    )
    summaries: dict[str, Any] = {}
    for method, observed in values.items():
        bootstrap = observed[indices].mean(axis=1)
        summaries[method] = {
            "mean": float(observed.mean()),
            "simulation_replicate_bootstrap_95_ci": np.quantile(
                bootstrap, [0.025, 0.975]
            ).tolist(),
        }

    paired: dict[str, Any] = {}
    primary = values["hierarchical_exact"]
    for method, comparator in values.items():
        if method == "hierarchical_exact":
            continue
        difference = primary - comparator
        bootstrap_difference = difference[indices].mean(axis=1)
        ratio_bootstrap = 1.0 - (
            primary[indices].mean(axis=1) / comparator[indices].mean(axis=1)
        )
        paired[method] = {
            "mean_difference_hierarchical_minus_comparator": float(
                difference.mean()
            ),
            "paired_simulation_replicate_bootstrap_95_ci": np.quantile(
                bootstrap_difference, [0.025, 0.975]
            ).tolist(),
            "relative_reduction_hierarchical_vs_comparator": float(
                1.0 - primary.mean() / comparator.mean()
            ),
            "relative_reduction_bootstrap_95_ci": np.quantile(
                ratio_bootstrap, [0.025, 0.975]
            ).tolist(),
            "favorable_replicates": int(np.count_nonzero(difference < 0.0)),
            "total_replicates": int(len(difference)),
        }
    return summaries, paired


def _summarize_scenario(
    name: str,
    heterogeneity_sd: float,
    margin_shift: bool,
    records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    requested_replicates: int,
    bootstraps: int,
    base_seed: int,
) -> dict[str, Any]:
    if not records:
        raise RuntimeError(f"all simulation replicates failed for {name}")

    field_summary, field_paired = _bootstrap_summary(
        records,
        "field_rmse",
        EXACT_METHODS,
        bootstraps,
        _seed(f"{name}|field-rmse-bootstrap", base_seed),
    )
    deviance_summary, deviance_paired = _bootstrap_summary(
        records,
        "recipient_deviance",
        METHODS,
        bootstraps,
        _seed(f"{name}|deviance-bootstrap", base_seed),
    )
    return {
        "heterogeneity_sd": heterogeneity_sd,
        "recipient_margin_shift": margin_shift,
        "recipient_field": "population_log_odds_without_recipient_random_effect",
        "requested_replicates": requested_replicates,
        "successful_replicates": len(records),
        "failed_replicates": len(failures),
        "failures": failures,
        "summary": {
            "field_rmse": field_summary,
            "field_rmse_paired_comparisons": field_paired,
            "recipient_deviance": deviance_summary,
            "recipient_deviance_paired_comparisons": deviance_paired,
        },
        "replicate_metrics": records,
    }


def _world(
    heterogeneity_label: str,
    heterogeneity_sd: float,
    replicates: int,
    bootstraps: int,
    base_seed: int,
) -> dict[str, Any]:
    records = {False: [], True: []}
    failures = {False: [], True: []}
    for replicate in range(replicates):
        source_generator = np.random.default_rng(
            _seed(f"{heterogeneity_label}|{replicate}|source", base_seed)
        )
        source = _panel(
            POPULATION_LOG_ODDS,
            heterogeneity_sd,
            SOURCE_ROW_PROPORTIONS,
            SOURCE_COLUMN_PROPORTIONS,
            SOURCE_DONORS,
            source_generator,
        )
        try:
            fitted = _fit_source(source)
        except (CouplingEstimationRefusal, FloatingPointError) as error:
            failure = {
                "replicate": replicate,
                "stage": "source_fit",
                "error_type": type(error).__name__,
                "reason": str(error),
            }
            failures[False].append(failure)
            failures[True].append(failure)
            continue
        for margin_shift in (False, True):
            target_generator = np.random.default_rng(
                _seed(f"{heterogeneity_label}|{replicate}|target", base_seed)
            )
            target = _panel(
                POPULATION_LOG_ODDS,
                0.0,
                SHIFTED_ROW_PROPORTIONS
                if margin_shift
                else SOURCE_ROW_PROPORTIONS,
                SHIFTED_COLUMN_PROPORTIONS
                if margin_shift
                else SOURCE_COLUMN_PROPORTIONS,
                RECIPIENT_DONORS,
                target_generator,
            )
            try:
                recipient_deviance = _score_target(fitted, target)
            except FloatingPointError as error:
                failures[margin_shift].append(
                    {
                        "replicate": replicate,
                        "stage": "recipient_score",
                        "error_type": type(error).__name__,
                        "reason": str(error),
                    }
                )
                continue
            records[margin_shift].append(
                {
                    "replicate": replicate,
                    "field_rmse": fitted["field_rmse"],
                    "recipient_deviance": recipient_deviance,
                    "fit_diagnostics": fitted["fit_diagnostics"],
                }
            )
    return {
        f"{heterogeneity_label}_same_margins": _summarize_scenario(
            f"{heterogeneity_label}_same_margins",
            heterogeneity_sd,
            False,
            records[False],
            failures[False],
            replicates,
            bootstraps,
            base_seed,
        ),
        f"{heterogeneity_label}_shifted_margins": _summarize_scenario(
            f"{heterogeneity_label}_shifted_margins",
            heterogeneity_sd,
            True,
            records[True],
            failures[True],
            replicates,
            bootstraps,
            base_seed,
        ),
    }


def _summary_tsv(payload: dict[str, Any]) -> str:
    columns = (
        "scenario",
        "heterogeneity_sd",
        "recipient_margin_shift",
        "method",
        "method_class",
        "successful_replicates",
        "failed_replicates",
        "field_rmse_mean",
        "field_rmse_ci_low",
        "field_rmse_ci_high",
        "recipient_deviance_mean",
        "recipient_deviance_ci_low",
        "recipient_deviance_ci_high",
        "deviance_difference_hierarchical_minus_method",
        "deviance_difference_ci_low",
        "deviance_difference_ci_high",
        "relative_deviance_reduction_hierarchical_vs_method",
        "relative_deviance_reduction_ci_low",
        "relative_deviance_reduction_ci_high",
        "hierarchical_favorable_replicates",
    )
    handle = io.StringIO()
    writer = csv.DictWriter(
        handle, fieldnames=columns, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    for scenario_name, scenario in payload["scenarios"].items():
        field = scenario["summary"]["field_rmse"]
        deviance = scenario["summary"]["recipient_deviance"]
        paired = scenario["summary"]["recipient_deviance_paired_comparisons"]
        for method in METHODS:
            field_record = field.get(method)
            comparison = paired.get(method)
            writer.writerow(
                {
                    "scenario": scenario_name,
                    "heterogeneity_sd": scenario["heterogeneity_sd"],
                    "recipient_margin_shift": str(
                        scenario["recipient_margin_shift"]
                    ).lower(),
                    "method": method,
                    "method_class": (
                        "exact_conditional" if method in EXACT_METHODS else "classical_residual"
                    ),
                    "successful_replicates": scenario["successful_replicates"],
                    "failed_replicates": scenario["failed_replicates"],
                    "field_rmse_mean": (
                        field_record["mean"] if field_record is not None else ""
                    ),
                    "field_rmse_ci_low": (
                        field_record["simulation_replicate_bootstrap_95_ci"][0]
                        if field_record is not None
                        else ""
                    ),
                    "field_rmse_ci_high": (
                        field_record["simulation_replicate_bootstrap_95_ci"][1]
                        if field_record is not None
                        else ""
                    ),
                    "recipient_deviance_mean": deviance[method]["mean"],
                    "recipient_deviance_ci_low": deviance[method][
                        "simulation_replicate_bootstrap_95_ci"
                    ][0],
                    "recipient_deviance_ci_high": deviance[method][
                        "simulation_replicate_bootstrap_95_ci"
                    ][1],
                    "deviance_difference_hierarchical_minus_method": (
                        comparison[
                            "mean_difference_hierarchical_minus_comparator"
                        ]
                        if comparison is not None
                        else 0.0
                    ),
                    "deviance_difference_ci_low": (
                        comparison[
                            "paired_simulation_replicate_bootstrap_95_ci"
                        ][0]
                        if comparison is not None
                        else 0.0
                    ),
                    "deviance_difference_ci_high": (
                        comparison[
                            "paired_simulation_replicate_bootstrap_95_ci"
                        ][1]
                        if comparison is not None
                        else 0.0
                    ),
                    "relative_deviance_reduction_hierarchical_vs_method": (
                        comparison[
                            "relative_reduction_hierarchical_vs_comparator"
                        ]
                        if comparison is not None
                        else 0.0
                    ),
                    "relative_deviance_reduction_ci_low": (
                        comparison["relative_reduction_bootstrap_95_ci"][0]
                        if comparison is not None
                        else 0.0
                    ),
                    "relative_deviance_reduction_ci_high": (
                        comparison["relative_reduction_bootstrap_95_ci"][1]
                        if comparison is not None
                        else 0.0
                    ),
                    "hierarchical_favorable_replicates": (
                        comparison["favorable_replicates"]
                        if comparison is not None
                        else scenario["successful_replicates"]
                    ),
                }
            )
    return handle.getvalue()


def run_simulation(
    *,
    replicates: int = DEFAULT_REPLICATES,
    bootstraps: int = DEFAULT_BOOTSTRAPS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    if replicates < 2:
        raise ValueError("replicates must be at least two")
    if bootstraps < 100:
        raise ValueError("bootstraps must be at least 100")
    results = {
        **_world(
            "heterogeneous",
            HETEROGENEITY_SD,
            replicates,
            bootstraps,
            seed,
        ),
        **_world("homogeneous", 0.0, replicates, bootstraps, seed),
    }
    return {
        "schema": "exact-conditional-heterogeneity-simulation/1.0",
        "status": "GROUND_TRUTH_ESTIMATOR_SIMULATION",
        "seed": seed,
        "bootstrap_draws": bootstraps,
        "estimand": "population log odds in independent fixed-margin binary tables",
        "objective_scope": (
            "pairwise composite over independent entity-pair 2x2 tables; no shared-pair "
            "dependence or coherent multivariate cell-level joint law is simulated"
        ),
        "recipient_evaluation": (
            "independent held tables sampled at the population field; estimators receive "
            "only recipient row and column margins"
        ),
        "selection": (
            "all hyperparameters and residual variants are fixed across scenarios; no "
            "outcome-based method or multiplier selection"
        ),
        "configuration": {
            "source_donors": SOURCE_DONORS,
            "recipient_donors": RECIPIENT_DONORS,
            "entity_shape": list(POPULATION_LOG_ODDS.shape),
            "tables_per_panel": int(POPULATION_LOG_ODDS.size),
            "table_total": TABLE_TOTAL,
            "margin_jitter_sd": MARGIN_JITTER_SD,
            "population_log_odds": POPULATION_LOG_ODDS.tolist(),
            "source_row_proportions": SOURCE_ROW_PROPORTIONS.tolist(),
            "source_column_proportions": SOURCE_COLUMN_PROPORTIONS.tolist(),
            "shifted_row_proportions": SHIFTED_ROW_PROPORTIONS.tolist(),
            "shifted_column_proportions": SHIFTED_COLUMN_PROPORTIONS.tolist(),
            "heterogeneity_sd": HETEROGENEITY_SD,
            "hierarchical_heterogeneity_penalty": HETEROGENEITY_PENALTY,
            "exact_ridge_penalty": RIDGE_PENALTY,
            "graph_penalty": GRAPH_PENALTY,
            "transport_multiplier": 1.0,
            "residual_pooling": "donor-equal after square-root-total normalization",
            "paired_random_streams_across_margin_ablation": True,
        },
        "methods": {
            "hierarchical_exact": (
                "donor-specific exact conditional log odds shrunk to a population field"
            ),
            "common_effect_exact": (
                "one common exact conditional log odds per entity pair"
            ),
            "poisson_deviance_raw": (
                "raw signed-root Poisson-deviance coordinate transfer"
            ),
            "poisson_deviance_centered": (
                "exact-null-centered signed-root Poisson-deviance transfer"
            ),
        },
        "scenarios": results,
        "bindings": {
            "evaluator_sha256": _sha256(Path(__file__)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-tsv", type=Path, default=DEFAULT_TSV)
    args = parser.parse_args()
    payload = run_simulation(
        replicates=args.replicates,
        bootstraps=args.bootstraps,
        seed=args.seed,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    args.output_tsv.write_text(_summary_tsv(payload))
    print(json.dumps({
        "json": str(args.output_json),
        "tsv": str(args.output_tsv),
        "scenario_failures": {
            name: result["failed_replicates"]
            for name, result in payload["scenarios"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
