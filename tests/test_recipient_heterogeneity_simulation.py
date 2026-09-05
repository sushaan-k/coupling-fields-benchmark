import numpy as np

from experiments.development.simulate_recipient_heterogeneity import (
    donor_losses,
    mixture_mean,
    summarize,
)
from mapreg.heterogeneity_adaptive_coupling import expected_binary_table_from_log_odds
from mapreg.predictive_conditional import normal_mixture_prediction


def test_zero_drift_matches_exact_prediction():
    rows, columns = np.array([[50, 78], [92, 36]]), np.array([[61, 67], [26, 102]])
    result = mixture_mean(np.array([-0.8, 1.1]), 0, rows, columns)
    for index, mu in enumerate([-0.8, 1.1]):
        expected = expected_binary_table_from_log_odds(mu, rows[index], columns[index])
        np.testing.assert_allclose(result[index], expected, atol=1e-11)


def test_normal_mixture_mean_matches_independent_adaptive_predictor():
    rows, columns = np.array([[64, 64], [22, 106]]), np.array([[64, 64], [97, 31]])
    result = mixture_mean(np.array([1.2, -0.7]), 1.2, rows, columns, 80)
    for index, mu in enumerate([1.2, -0.7]):
        expected = normal_mixture_prediction(mu, 1.2**2, rows[index], columns[index])
        np.testing.assert_allclose(result[index], expected.mean_table, atol=1e-9)
    np.testing.assert_allclose(result.sum(axis=-1), rows, atol=1e-11)
    np.testing.assert_allclose(result.sum(axis=-2), columns, atol=1e-11)


def test_loss_averages_pairs_inside_each_donor():
    truth = np.array([[[[2, 1], [1, 2]]], [[[1, 2], [2, 1]]]])
    np.testing.assert_allclose(donor_losses(truth, truth.astype(float)), [0, 0])


def test_paired_bootstrap_uses_replicates():
    methods = [
        "hierarchical_exact",
        "common_effect_exact",
        "oracle_plugin",
        "oracle_mixture",
    ]
    records = [{"losses": dict(zip(methods, [1.0, 2.0, 3.0, 2.0]))} for _ in range(4)]
    result = summarize(records, 20, 0)
    comparison = result["paired_comparisons"]["oracle_mixture_vs_oracle_plugin"]
    np.testing.assert_allclose(comparison["difference_ci95"], [-1, -1])
    assert summarize([], 20, 0) is None


def test_failed_source_fits_remain_in_every_recipient_scenario(monkeypatch):
    from experiments.development import simulate_recipient_heterogeneity as simulation

    def refuse(source):
        raise simulation.original.CouplingEstimationRefusal("test refusal")

    monkeypatch.setattr(simulation, "fit_source", refuse)
    result = simulation.run(replicates=1, bootstraps=5)
    assert len(result["source_fit_failures"]) == 2
    assert len(result["scenarios"]) == 8
    for scenario in result["scenarios"].values():
        assert scenario["successful_replicates"] == 0
        assert len(scenario["failures"]) == 1
        assert scenario["summary"] is None
