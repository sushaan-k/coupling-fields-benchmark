from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import evaluate_scmmib_bmmc_exact_development as reference
from experiments.development import simulate_exact_conditional_heterogeneity as simulation


ROOT = Path(__file__).resolve().parents[1]
RESULT_JSON = (
    ROOT
    / "results/development/exact_conditional_heterogeneity_simulation_v1.json"
)
RESULT_TSV = (
    ROOT
    / "results/development/exact_conditional_heterogeneity_simulation_v1.tsv"
)


def _synthetic_tables() -> np.ndarray:
    return np.asarray(
        [
            [
                [
                    [[18, 10], [12, 24]],
                    [[12, 20], [16, 16]],
                ]
            ],
            [
                [
                    [[16, 12], [14, 22]],
                    [[10, 22], [18, 14]],
                ]
            ],
            [
                [
                    [[19, 9], [11, 25]],
                    [[13, 19], [15, 17]],
                ]
            ],
        ],
        dtype=np.int64,
    )


def test_exact_sampler_is_deterministic_and_preserves_fixed_margins() -> None:
    first = simulation._conditional_sample(
        0.7, 43, 51, 96, np.random.default_rng(19)
    )
    second = simulation._conditional_sample(
        0.7, 43, 51, 96, np.random.default_rng(19)
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first.sum(axis=1), [43, 53])
    np.testing.assert_array_equal(first.sum(axis=0), [51, 45])


def test_residual_transfer_matches_the_benchmark_implementation() -> None:
    source = _synthetic_tables()
    target = source[[1, 2]]
    for centered in (False, True):
        observed_coordinate = simulation._pooled_residual_coordinate(
            source, centered
        )
        expected_coordinate = reference._residual_coordinate(
            source, "deviance", centered
        )
        np.testing.assert_allclose(observed_coordinate, expected_coordinate)
        observed_prediction = simulation._predict_residual(
            observed_coordinate, target, centered
        )
        expected_prediction = reference._predict_residual(
            expected_coordinate,
            target,
            family="deviance",
            centered=centered,
            alpha=1.0,
        )
        np.testing.assert_allclose(observed_prediction, expected_prediction)


def test_small_run_pairs_source_fits_across_margin_ablation() -> None:
    result = simulation.run_simulation(replicates=2, bootstraps=100, seed=31)
    for label in ("heterogeneous", "homogeneous"):
        same = result["scenarios"][f"{label}_same_margins"]
        shifted = result["scenarios"][f"{label}_shifted_margins"]
        assert same["failed_replicates"] == shifted["failed_replicates"] == 0
        for method in simulation.EXACT_METHODS:
            assert (
                same["summary"]["field_rmse"][method]["mean"]
                == shifted["summary"]["field_rmse"][method]["mean"]
            )
        assert (
            shifted["summary"]["recipient_deviance"][
                "poisson_deviance_raw"
            ]["mean"]
            > same["summary"]["recipient_deviance"][
                "poisson_deviance_raw"
            ]["mean"]
        )


def test_public_result_and_tsv_are_bound_and_internally_consistent() -> None:
    result = json.loads(RESULT_JSON.read_text())
    assert result["schema"] == "exact-conditional-heterogeneity-simulation/1.0"
    assert result["status"] == "GROUND_TRUTH_ESTIMATOR_SIMULATION"
    assert result["configuration"]["graph_penalty"] == 0.0
    assert result["configuration"]["transport_multiplier"] == 1.0
    assert "pairwise composite" in result["objective_scope"]
    assert result["bindings"]["evaluator_sha256"] == hashlib.sha256(
        (ROOT / "experiments/development/simulate_exact_conditional_heterogeneity.py").read_bytes()
    ).hexdigest()

    for scenario in result["scenarios"].values():
        assert scenario["requested_replicates"] == simulation.DEFAULT_REPLICATES
        assert scenario["successful_replicates"] == simulation.DEFAULT_REPLICATES
        assert scenario["failed_replicates"] == 0

    heterogeneous = result["scenarios"]["heterogeneous_shifted_margins"]
    field_comparison = heterogeneous["summary"][
        "field_rmse_paired_comparisons"
    ]["common_effect_exact"]
    residual_comparison = heterogeneous["summary"][
        "recipient_deviance_paired_comparisons"
    ]["poisson_deviance_raw"]
    assert field_comparison["paired_simulation_replicate_bootstrap_95_ci"][1] < 0.0
    assert residual_comparison["paired_simulation_replicate_bootstrap_95_ci"][1] < 0.0

    homogeneous = result["scenarios"]["homogeneous_same_margins"]
    assert (
        homogeneous["summary"]["field_rmse"]["common_effect_exact"]["mean"]
        < homogeneous["summary"]["field_rmse"]["hierarchical_exact"]["mean"]
    )

    with RESULT_TSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == len(result["scenarios"]) * len(simulation.METHODS)
    for row in rows:
        scenario = result["scenarios"][row["scenario"]]
        method = row["method"]
        observed = float(row["recipient_deviance_mean"])
        expected = scenario["summary"]["recipient_deviance"][method]["mean"]
        assert observed == expected
