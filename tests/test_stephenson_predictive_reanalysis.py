import json

import numpy as np
import pytest

from experiments.development import reanalyze_stephenson_prediction as reanalysis


def panel(values):
    return np.array([np.tile([[a, 20 - a], [20 - a, a]], (9, 9, 1, 1)) for a in values])


@pytest.mark.parametrize("family", ["hierarchical", "common_unpenalized", "common_ridge"])
def test_conditional_family_smoke(family):
    fitted = reanalysis.fit_conditional(panel([8, 10, 14]), family)
    assert fitted["status"] == "FITTED", fitted
    rows = reanalysis.score_model(fitted, panel([10, 12]), np.array(["d1", "d2"]), 1)
    assert all(row["status"] == "SCORED" for row in rows)
    assert all(row["informative_pairs"] == 81 for row in rows)


def test_common_ridge_score_root_matches_the_existing_penalized_fit():
    tables = panel([8, 10, 14])
    expected = reanalysis.fit_conditional(tables, "common_ridge")
    recovered = reanalysis.fit_common_ridge_by_score(tables)
    assert np.max(np.abs(expected["mu"] - recovered["mu"])) < 1e-7
    assert recovered["certificate"]["gradient_norm"] < 1e-8


def test_common_ridge_optimizer_failure_keeps_the_same_objective(monkeypatch):
    def fail(*args, **kwargs):
        raise reanalysis.CouplingEstimationRefusal("conditional-likelihood optimizer did not converge")

    monkeypatch.setattr(reanalysis, "fit_structured_conditional_log_odds", fail)
    recovered = reanalysis.fit_conditional(panel([8, 10, 14]), "common_ridge")
    assert recovered["status"] == "FITTED"
    assert recovered["certificate"]["solver"] == "separable_score_brentq"
    assert recovered["certificate"]["initial_failure"] == "conditional-likelihood optimizer did not converge"


def test_original_pair_masks_are_preserved():
    tables = panel([12])
    tables[0, 0, 0] = [[0, 0], [20, 20]]
    fitted = {"status": "FITTED", "mu": np.ones((9, 9)),
              "tau2": np.ones((9, 9)), "boundary": np.zeros((9, 9))}
    row = reanalysis.score_model(fitted, tables, np.array(["d1"]), 1,
                                 mixture=True, intervals=True)[0]
    assert row["informative_pairs"] == 80
    assert row["pair_losses"][0] is None
    assert row["interval_covered"][0] is None
    assert len(row["pair_losses"]) == 81
    assert row["coverage"] == pytest.approx(np.mean(row["interval_covered"][1:]))


def test_failed_fit_does_not_produce_partial_scores_or_comparison():
    unavailable = reanalysis.score_model({"status": "FIT_FAILED"}, panel([12, 13]),
                                          np.array(["d1", "d2"]), 1)
    assert len(unavailable) == 2
    assert all(row["loss"] is None for row in unavailable)
    primary = [dict(row, status="SCORED", loss=0.1) for row in unavailable]
    assert reanalysis.compare(primary, unavailable, "failed")["status"] == "UNAVAILABLE"


def test_alpha_selection_uses_all_pilot_donors_and_deterministic_tie_break(monkeypatch):
    calls = []

    def score(fit, tables, donors, alpha, **kwargs):
        calls.append((tables.copy(), donors.copy(), alpha))
        return [{"donor": str(d), "status": "SCORED", "loss": (alpha - 0.9)**2} for d in donors]

    monkeypatch.setattr(reanalysis, "score_model", score)
    pilot = panel([8, 9])
    selected = reanalysis.select_alpha({}, pilot, np.array(["pilot1", "pilot2"]))
    assert selected["alpha"] == 1.0
    assert len(calls) == 4
    assert all(np.array_equal(tables, pilot) for tables, _, _ in calls)
    assert all(list(donors) == ["pilot1", "pilot2"] for _, donors, _ in calls)


def test_result_serialization_retains_boundary_status_without_nonfinite_json():
    data = {"mu": np.array([[np.inf, -np.inf]]), "boundary": np.array([[1, -1]]),
            "count": np.int64(2), "converged": np.bool_(True)}
    serialized = reanalysis.serializable(data)
    assert serialized["mu"] == [["+Infinity", "-Infinity"]]
    assert serialized["boundary"] == [[1, -1]]
    json.dumps(serialized, allow_nan=False)


def test_comparison_is_donor_paired_not_pair_pseudoreplication():
    primary = [{"donor": str(i), "status": "SCORED", "loss": value}
               for i, value in enumerate([0.1, 0.2, 0.3])]
    comparator = [dict(row, loss=row["loss"] + 0.1) for row in primary]
    comparison = reanalysis.compare(primary, comparator, "unit-test")
    assert comparison["units"] == 3
    assert comparison["bootstrap_draws"] == 20_000
    assert comparison["favorable_donors"] == 3
    assert comparison["exact_one_sided_sign_p"] == pytest.approx(0.125)


def test_extended_endpoint_predictions_preserve_infinite_loss_and_point_intervals():
    fitted = {"status": "FITTED", "mu": np.full((9, 9), np.inf),
              "tau2": np.full((9, 9), np.nan), "boundary": np.ones((9, 9), dtype=int)}
    rows = reanalysis.score_model(fitted, panel([12, 20]), np.array(["interior", "endpoint"]),
                                   0.5, mixture=True, intervals=True)
    assert rows[0]["status"] == "INFINITE_DEVIANCE"
    assert rows[0]["loss"] == np.inf
    assert rows[0]["coverage"] == 0
    assert rows[0]["mean_interval_width"] == 0
    assert rows[1]["status"] == "SCORED"
    assert rows[1]["loss"] == pytest.approx(0)
    assert rows[1]["coverage"] == 1
    assert reanalysis.serializable(rows[0])["pair_losses"][0] == "+Infinity"
    comparison = reanalysis.compare(rows, rows, "infinite")
    assert comparison["status"] == "INFINITE_DEVIANCE"
    assert comparison["bootstrap_performed"] is False
