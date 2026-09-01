"""Ground-truth numerical stress test for the fixed-margin transfer-risk bound.

The source field is fixed at its known population value so that the experiment
isolates the two quantities in the proposition: transport error induced by
``alpha`` and recipient drift. Recipient drift is a bounded, symmetric
two-point distribution with the requested variance. For each informative
margin configuration, the script computes

* a seeded finite-sample estimate of excess conditional log loss;
* the exact expected conditional KL risk;
* the curvature-integral representation of that risk; and
* the Fisher-information and finite-support envelopes from the theorem.

The exact calculations enumerate the finite noncentral-hypergeometric support.
No fitted model or asymptotic approximation enters the bound checks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gammaln, logsumexp


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = ROOT / "results/development/transfer_risk_bound_stress_v1.json"
DEFAULT_TSV = ROOT / "results/development/transfer_risk_bound_stress_v1.tsv"

DEFAULT_SEED = 20260828
DEFAULT_DRAWS_PER_DRIFT_STATE = 1_000_000
TRUE_POPULATION_LOG_ODDS = 0.8
SOURCE_FIELD_ESTIMATE = TRUE_POPULATION_LOG_ODDS
TRANSPORT_MULTIPLIERS = (0.5, 0.75, 1.0, 1.25)
DRIFT_STANDARD_DEVIATIONS = (0.0, 0.25, 0.5, 1.0)
MARGINS = (
    {"id": "balanced", "row_zero": 48, "column_zero": 48, "total": 96},
    {"id": "row_sparse", "row_zero": 24, "column_zero": 66, "total": 96},
    {"id": "overlap_shifted", "row_zero": 76, "column_zero": 31, "total": 96},
)
EXTREMA_GRID_SIZE = 8193
INTEGRATION_ABSOLUTE_TOLERANCE = 1e-12
INTEGRATION_RELATIVE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ConditionalFamily:
    """One finite fixed-margin noncentral-hypergeometric family."""

    support: np.ndarray
    log_base_weight: np.ndarray
    total: int
    lower: int
    upper: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed(label: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}|{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _conditional_family(
    row_zero: int,
    column_zero: int,
    total: int,
) -> ConditionalFamily:
    row = int(row_zero)
    column = int(column_zero)
    size = int(total)
    if size < 2 or not (0 < row < size) or not (0 < column < size):
        raise ValueError("margins must be positive and smaller than the total")
    lower = max(0, row + column - size)
    upper = min(row, column)
    if upper <= lower:
        raise ValueError("margins must have at least two feasible tables")
    support = np.arange(lower, upper + 1, dtype=float)
    log_base_weight = (
        gammaln(column + 1)
        - gammaln(support + 1)
        - gammaln(column - support + 1)
        + gammaln(size - column + 1)
        - gammaln(row - support + 1)
        - gammaln(size - column - row + support + 1)
    )
    return ConditionalFamily(
        support=support,
        log_base_weight=log_base_weight,
        total=size,
        lower=lower,
        upper=upper,
    )


def _distribution(family: ConditionalFamily, theta: float) -> np.ndarray:
    log_mass = family.log_base_weight + float(theta) * family.support
    return np.exp(log_mass - logsumexp(log_mass))


def _moments(
    family: ConditionalFamily, theta: float
) -> tuple[float, float, float]:
    probability = _distribution(family, theta)
    mean = float(np.sum(probability * family.support))
    centered = family.support - mean
    variance = float(np.sum(probability * np.square(centered)))
    third_central = float(np.sum(probability * np.power(centered, 3)))
    return mean, variance, third_central


def _conditional_kl(
    family: ConditionalFamily,
    true_theta: float,
    predicted_theta: float,
) -> float:
    true_probability = _distribution(family, true_theta)
    true_log_mass = family.log_base_weight + true_theta * family.support
    predicted_log_mass = (
        family.log_base_weight + predicted_theta * family.support
    )
    true_log_probability = true_log_mass - logsumexp(true_log_mass)
    predicted_log_probability = predicted_log_mass - logsumexp(
        predicted_log_mass
    )
    return float(
        np.sum(true_probability * (true_log_probability - predicted_log_probability))
    )


def _curvature_integral_risk(
    family: ConditionalFamily,
    true_theta: float,
    predicted_theta: float,
) -> float:
    displacement = float(predicted_theta - true_theta)
    if displacement == 0.0:
        return 0.0

    integral, _ = quad(
        lambda t: (1.0 - t)
        * _moments(family, true_theta + t * displacement)[1],
        0.0,
        1.0,
        epsabs=INTEGRATION_ABSOLUTE_TOLERANCE,
        epsrel=INTEGRATION_RELATIVE_TOLERANCE,
        limit=100,
    )
    return float(displacement**2 * integral)


def _information_extrema(
    family: ConditionalFamily,
    interval_low: float,
    interval_high: float,
) -> dict[str, Any]:
    """Find information extrema through roots of its third cumulant.

    In a one-parameter exponential family, the derivative of Fisher
    information is the third central moment. A dense deterministic grid
    brackets its roots; the returned extrema include a floating-point padding
    so they form conservative numerical envelopes.
    """

    low = float(min(interval_low, interval_high))
    high = float(max(interval_low, interval_high))
    if low == high:
        information = _moments(family, low)[1]
        padding = 64.0 * np.finfo(float).eps * max(1.0, information)
        return {
            "interval": [low, high],
            "lower": max(0.0, information - padding),
            "upper": information + padding,
            "stationary_points": [],
            "numerical_padding": padding,
        }

    grid = np.linspace(low, high, EXTREMA_GRID_SIZE)
    derivative = np.asarray([_moments(family, value)[2] for value in grid])
    roots: list[float] = []
    for index in range(len(grid) - 1):
        left_value = derivative[index]
        right_value = derivative[index + 1]
        if left_value == 0.0:
            roots.append(float(grid[index]))
        elif left_value * right_value < 0.0:
            roots.append(
                float(
                    brentq(
                        lambda value: _moments(family, value)[2],
                        float(grid[index]),
                        float(grid[index + 1]),
                        xtol=1e-14,
                        rtol=4.0 * np.finfo(float).eps,
                    )
                )
            )
    if derivative[-1] == 0.0:
        roots.append(high)

    unique_roots: list[float] = []
    for root in sorted(roots):
        if not unique_roots or abs(root - unique_roots[-1]) > 1e-10:
            unique_roots.append(root)
    candidates = [low, high, *unique_roots]
    information = np.asarray(
        [_moments(family, value)[1] for value in candidates], dtype=float
    )
    raw_lower = float(information.min())
    raw_upper = float(information.max())
    padding = 64.0 * np.finfo(float).eps * max(1.0, raw_upper)
    return {
        "interval": [low, high],
        "lower": max(0.0, raw_lower - padding),
        "upper": raw_upper + padding,
        "stationary_points": unique_roots,
        "numerical_padding": padding,
    }


def _drift_distribution(standard_deviation: float) -> tuple[np.ndarray, np.ndarray]:
    sigma = float(standard_deviation)
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("drift standard deviation must be finite and nonnegative")
    if sigma == 0.0:
        return np.asarray([0.0]), np.asarray([1.0])
    return np.asarray([-sigma, sigma]), np.asarray([0.5, 0.5])


def _finite_sample_excess_loss(
    family: ConditionalFamily,
    true_thetas: np.ndarray,
    drift_weights: np.ndarray,
    predicted_theta: float,
    draws_per_state: int,
    generator: np.random.Generator,
) -> tuple[float, float]:
    if draws_per_state < 1:
        raise ValueError("draws_per_state must be positive")
    empirical = 0.0
    variance = 0.0
    for true_theta, weight in zip(true_thetas, drift_weights):
        true_probability = _distribution(family, float(true_theta))
        predicted_probability = _distribution(family, predicted_theta)
        score = np.log(true_probability) - np.log(predicted_probability)
        counts = generator.multinomial(draws_per_state, true_probability)
        empirical += float(weight * np.dot(counts, score) / draws_per_state)
        exact_mean = float(np.dot(true_probability, score))
        state_variance = float(
            np.dot(true_probability, np.square(score - exact_mean))
        )
        variance += float(weight**2 * state_variance / draws_per_state)
    return empirical, math.sqrt(max(variance, 0.0))


def _evaluate_condition(
    margin: dict[str, Any],
    alpha: float,
    drift_standard_deviation: float,
    draws_per_state: int,
    seed: int,
) -> dict[str, Any]:
    family = _conditional_family(
        margin["row_zero"], margin["column_zero"], margin["total"]
    )
    drift, drift_weight = _drift_distribution(drift_standard_deviation)
    true_theta = TRUE_POPULATION_LOG_ODDS + drift
    predicted_theta = float(alpha * SOURCE_FIELD_ESTIMATE)

    exact_components = np.asarray(
        [
            _conditional_kl(family, float(theta), predicted_theta)
            for theta in true_theta
        ]
    )
    integral_components = np.asarray(
        [
            _curvature_integral_risk(family, float(theta), predicted_theta)
            for theta in true_theta
        ]
    )
    exact_risk = float(np.dot(drift_weight, exact_components))
    integral_risk = float(np.dot(drift_weight, integral_components))

    squared_error = float(
        np.dot(drift_weight, np.square(predicted_theta - true_theta))
    )
    bias = predicted_theta - TRUE_POPULATION_LOG_ODDS
    decomposed_squared_error = float(
        bias**2 + drift_standard_deviation**2
    )
    if not math.isclose(
        squared_error, decomposed_squared_error, rel_tol=1e-13, abs_tol=1e-13
    ):
        raise AssertionError("drift moments do not match the risk decomposition")

    interval_low = float(min(predicted_theta, float(true_theta.min())))
    interval_high = float(max(predicted_theta, float(true_theta.max())))
    extrema = _information_extrema(family, interval_low, interval_high)
    lower_bound = float(0.5 * extrema["lower"] * squared_error)
    upper_bound = float(0.5 * extrema["upper"] * squared_error)
    support_upper_bound = float(
        (family.upper - family.lower) ** 2 * squared_error / 8.0
    )
    tolerance = 2e-10 * max(1.0, exact_risk, upper_bound)
    within_information_envelope = bool(
        exact_risk >= lower_bound - tolerance
        and exact_risk <= upper_bound + tolerance
    )
    within_support_envelope = bool(exact_risk <= support_upper_bound + tolerance)

    condition_seed = _seed(
        f"{margin['id']}|{alpha:.12g}|{drift_standard_deviation:.12g}", seed
    )
    empirical_risk, empirical_standard_error = _finite_sample_excess_loss(
        family,
        true_theta,
        drift_weight,
        predicted_theta,
        draws_per_state,
        np.random.default_rng(condition_seed),
    )
    if empirical_standard_error == 0.0:
        empirical_z = 0.0 if empirical_risk == exact_risk else math.inf
    else:
        empirical_z = (empirical_risk - exact_risk) / empirical_standard_error

    return {
        "margin_id": margin["id"],
        "total": family.total,
        "row_zero": int(margin["row_zero"]),
        "column_zero": int(margin["column_zero"]),
        "support_lower": family.lower,
        "support_upper": family.upper,
        "support_size": int(len(family.support)),
        "true_population_log_odds": TRUE_POPULATION_LOG_ODDS,
        "source_field_estimate": SOURCE_FIELD_ESTIMATE,
        "transport_multiplier": float(alpha),
        "predicted_log_odds": predicted_theta,
        "transport_bias": bias,
        "drift_standard_deviation": float(drift_standard_deviation),
        "drift_variance": float(drift_standard_deviation**2),
        "drift_support": drift.tolist(),
        "drift_weights": drift_weight.tolist(),
        "mean_squared_canonical_error": squared_error,
        "bias_squared_plus_drift_variance": decomposed_squared_error,
        "empirical_excess_log_loss": empirical_risk,
        "empirical_monte_carlo_standard_error": empirical_standard_error,
        "empirical_normal_95_ci": [
            empirical_risk - 1.959963984540054 * empirical_standard_error,
            empirical_risk + 1.959963984540054 * empirical_standard_error,
        ],
        "empirical_minus_exact_z": empirical_z,
        "exact_expected_excess_log_loss": exact_risk,
        "curvature_integral_risk": integral_risk,
        "exact_minus_integral": exact_risk - integral_risk,
        "information_interval": extrema["interval"],
        "information_lower": extrema["lower"],
        "information_upper": extrema["upper"],
        "information_stationary_points": extrema["stationary_points"],
        "information_numerical_padding": extrema["numerical_padding"],
        "theorem_lower_bound": lower_bound,
        "theorem_upper_bound": upper_bound,
        "finite_support_upper_bound": support_upper_bound,
        "lower_bound_slack": exact_risk - lower_bound,
        "upper_bound_slack": upper_bound - exact_risk,
        "finite_support_upper_bound_slack": support_upper_bound - exact_risk,
        "within_information_envelope": within_information_envelope,
        "within_finite_support_envelope": within_support_envelope,
        "exact_expected_excess_log_loss_per_cell": exact_risk / family.total,
        "curvature_integral_risk_per_cell": integral_risk / family.total,
        "empirical_excess_log_loss_per_cell": empirical_risk / family.total,
        "condition_seed": condition_seed,
    }


def _summary_tsv(payload: dict[str, Any]) -> str:
    columns = (
        "margin_id",
        "total",
        "row_zero",
        "column_zero",
        "support_lower",
        "support_upper",
        "support_size",
        "transport_multiplier",
        "predicted_log_odds",
        "transport_bias",
        "drift_standard_deviation",
        "drift_variance",
        "mean_squared_canonical_error",
        "empirical_excess_log_loss",
        "empirical_monte_carlo_standard_error",
        "exact_expected_excess_log_loss",
        "curvature_integral_risk",
        "exact_minus_integral",
        "information_lower",
        "information_upper",
        "theorem_lower_bound",
        "theorem_upper_bound",
        "finite_support_upper_bound",
        "within_information_envelope",
        "within_finite_support_envelope",
        "exact_expected_excess_log_loss_per_cell",
    )
    handle = io.StringIO()
    writer = csv.DictWriter(
        handle, fieldnames=columns, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    for record in payload["conditions"]:
        writer.writerow(
            {
                column: (
                    str(record[column]).lower()
                    if isinstance(record[column], bool)
                    else record[column]
                )
                for column in columns
            }
        )
    return handle.getvalue()


def run_stress_test(
    *,
    draws_per_state: int = DEFAULT_DRAWS_PER_DRIFT_STATE,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    if draws_per_state < 1:
        raise ValueError("draws_per_state must be positive")
    conditions = [
        _evaluate_condition(margin, alpha, sigma, draws_per_state, seed)
        for margin in MARGINS
        for alpha in TRANSPORT_MULTIPLIERS
        for sigma in DRIFT_STANDARD_DEVIATIONS
    ]
    exact_integral_errors = np.abs(
        np.asarray([record["exact_minus_integral"] for record in conditions])
    )
    empirical_errors = np.abs(
        np.asarray(
            [
                record["empirical_excess_log_loss"]
                - record["exact_expected_excess_log_loss"]
                for record in conditions
            ]
        )
    )
    empirical_z = np.abs(
        np.asarray([record["empirical_minus_exact_z"] for record in conditions])
    )
    lower_slack = np.asarray(
        [record["lower_bound_slack"] for record in conditions]
    )
    upper_slack = np.asarray(
        [record["upper_bound_slack"] for record in conditions]
    )

    drift_floor = {}
    for margin in MARGINS:
        relevant = [
            record
            for record in conditions
            if record["margin_id"] == margin["id"]
            and record["transport_multiplier"] == 1.0
        ]
        drift_floor[margin["id"]] = {
            str(record["drift_standard_deviation"]): record[
                "exact_expected_excess_log_loss"
            ]
            for record in relevant
        }

    return {
        "schema": "transfer-risk-bound-stress/1.0",
        "status": "GROUND_TRUTH_TRANSFER_RISK_STRESS_TEST",
        "seed": int(seed),
        "estimand": (
            "conditional excess log loss for a finite fixed-margin 2x2 "
            "noncentral-hypergeometric family"
        ),
        "design": (
            "known source population field; bounded mean-zero two-point "
            "recipient drift; fixed transport multipliers; exact finite-support "
            "risk and seeded conditional outcome sampling"
        ),
        "scope": (
            "numerical validation of the fixed-margin transfer-risk identity and "
            "its curvature envelopes; not an estimator-performance comparison"
        ),
        "configuration": {
            "true_population_log_odds": TRUE_POPULATION_LOG_ODDS,
            "source_field_estimate": SOURCE_FIELD_ESTIMATE,
            "transport_multipliers": list(TRANSPORT_MULTIPLIERS),
            "drift_standard_deviations": list(DRIFT_STANDARD_DEVIATIONS),
            "drift_family": "symmetric two-point; support {-sigma,+sigma}",
            "margins": list(MARGINS),
            "draws_per_drift_state": int(draws_per_state),
            "information_extrema_grid_size": EXTREMA_GRID_SIZE,
            "quadrature_absolute_tolerance": INTEGRATION_ABSOLUTE_TOLERANCE,
            "quadrature_relative_tolerance": INTEGRATION_RELATIVE_TOLERANCE,
        },
        "summary": {
            "condition_count": len(conditions),
            "all_exact_risks_within_information_envelopes": all(
                record["within_information_envelope"] for record in conditions
            ),
            "all_exact_risks_within_finite_support_envelopes": all(
                record["within_finite_support_envelope"] for record in conditions
            ),
            "maximum_absolute_exact_vs_curvature_integral_error": float(
                exact_integral_errors.max()
            ),
            "maximum_absolute_empirical_vs_exact_error": float(
                empirical_errors.max()
            ),
            "maximum_absolute_empirical_vs_exact_z": float(empirical_z.max()),
            "empirical_normal_95_ci_coverage_count": int(
                sum(
                    record["empirical_normal_95_ci"][0]
                    <= record["exact_expected_excess_log_loss"]
                    <= record["empirical_normal_95_ci"][1]
                    for record in conditions
                )
            ),
            "minimum_information_lower_bound_slack": float(lower_slack.min()),
            "minimum_information_upper_bound_slack": float(upper_slack.min()),
            "alpha_one_drift_floor_by_margin": drift_floor,
        },
        "conditions": conditions,
        "bindings": {"evaluator_sha256": _sha256(Path(__file__))},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--draws-per-state", type=int, default=DEFAULT_DRAWS_PER_DRIFT_STATE
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-tsv", type=Path, default=DEFAULT_TSV)
    args = parser.parse_args()
    payload = run_stress_test(
        draws_per_state=args.draws_per_state,
        seed=args.seed,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    args.output_tsv.write_text(_summary_tsv(payload))
    print(
        json.dumps(
            {
                "json": str(args.output_json),
                "tsv": str(args.output_tsv),
                "summary": payload["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
