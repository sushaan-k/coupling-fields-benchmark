from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.development import stress_transfer_risk_bound as stress


ROOT = Path(__file__).resolve().parents[1]
RESULT_JSON = ROOT / "results/development/transfer_risk_bound_stress_v1.json"
RESULT_TSV = ROOT / "results/development/transfer_risk_bound_stress_v1.tsv"


def test_conditional_family_is_normalized_and_informative() -> None:
    family = stress._conditional_family(24, 66, 96)
    probability = stress._distribution(family, 0.8)
    assert family.lower == 0
    assert family.upper == 24
    np.testing.assert_allclose(probability.sum(), 1.0, atol=1e-14)
    mean, information, third_central = stress._moments(family, 0.8)
    assert family.lower < mean < family.upper
    assert information > 0.0
    assert np.isfinite(third_central)


def test_curvature_integral_equals_exact_conditional_kl() -> None:
    family = stress._conditional_family(76, 31, 96)
    for true_theta, predicted_theta in ((-0.2, 0.4), (0.8, 0.8), (1.8, 1.0)):
        exact = stress._conditional_kl(family, true_theta, predicted_theta)
        integral = stress._curvature_integral_risk(
            family, true_theta, predicted_theta
        )
        np.testing.assert_allclose(exact, integral, rtol=1e-11, atol=2e-14)


def test_information_extrema_enclose_dense_interval_evaluation() -> None:
    family = stress._conditional_family(48, 48, 96)
    extrema = stress._information_extrema(family, -0.2, 1.8)
    information = np.asarray(
        [
            stress._moments(family, theta)[1]
            for theta in np.linspace(-0.2, 1.8, 4001)
        ]
    )
    assert extrema["lower"] <= float(information.min())
    assert extrema["upper"] >= float(information.max())


def test_condition_obeys_risk_decomposition_and_both_envelopes() -> None:
    record = stress._evaluate_condition(
        stress.MARGINS[0],
        alpha=0.75,
        drift_standard_deviation=0.5,
        draws_per_state=20_000,
        seed=19,
    )
    expected_squared_error = (
        0.75 * stress.SOURCE_FIELD_ESTIMATE - stress.TRUE_POPULATION_LOG_ODDS
    ) ** 2 + 0.5**2
    np.testing.assert_allclose(
        record["mean_squared_canonical_error"], expected_squared_error
    )
    assert record["within_information_envelope"]
    assert record["within_finite_support_envelope"]
    assert (
        record["theorem_lower_bound"]
        <= record["exact_expected_excess_log_loss"]
        <= record["theorem_upper_bound"]
    )


def test_small_run_is_deterministic_and_contains_drift_floor(monkeypatch) -> None:
    monkeypatch.setattr(stress, "MARGINS", stress.MARGINS[:1])
    monkeypatch.setattr(stress, "TRANSPORT_MULTIPLIERS", (1.0,))
    first = stress.run_stress_test(draws_per_state=500, seed=31)
    second = stress.run_stress_test(draws_per_state=500, seed=31)
    assert first == second
    assert first["summary"]["condition_count"] == (
        len(stress.MARGINS)
        * len(stress.TRANSPORT_MULTIPLIERS)
        * len(stress.DRIFT_STANDARD_DEVIATIONS)
    )
    assert first["summary"]["all_exact_risks_within_information_envelopes"]
    assert first["summary"]["all_exact_risks_within_finite_support_envelopes"]
    for risk_by_sigma in first["summary"]["alpha_one_drift_floor_by_margin"].values():
        assert risk_by_sigma["0.0"] == 0.0
        assert risk_by_sigma["0.25"] > risk_by_sigma["0.0"]
        assert risk_by_sigma["0.5"] > risk_by_sigma["0.25"]
        assert risk_by_sigma["1.0"] > risk_by_sigma["0.5"]


def test_public_result_is_bound_and_tsv_matches_json() -> None:
    payload = json.loads(RESULT_JSON.read_text())
    assert payload["schema"] == "transfer-risk-bound-stress/1.0"
    assert payload["status"] == "GROUND_TRUTH_TRANSFER_RISK_STRESS_TEST"
    assert payload["configuration"]["draws_per_drift_state"] == (
        stress.DEFAULT_DRAWS_PER_DRIFT_STATE
    )
    assert payload["summary"]["all_exact_risks_within_information_envelopes"]
    assert payload["summary"]["all_exact_risks_within_finite_support_envelopes"]
    assert (
        payload["summary"][
            "maximum_absolute_exact_vs_curvature_integral_error"
        ]
        < 1e-11
    )
    assert payload["bindings"]["evaluator_sha256"] == hashlib.sha256(
        (
            ROOT
            / "experiments/development/stress_transfer_risk_bound.py"
        ).read_bytes()
    ).hexdigest()

    with RESULT_TSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == payload["summary"]["condition_count"]
    for row, record in zip(rows, payload["conditions"]):
        assert row["margin_id"] == record["margin_id"]
        assert float(row["transport_multiplier"]) == record[
            "transport_multiplier"
        ]
        assert float(row["drift_variance"]) == record["drift_variance"]
        assert float(row["exact_expected_excess_log_loss"]) == record[
            "exact_expected_excess_log_loss"
        ]
