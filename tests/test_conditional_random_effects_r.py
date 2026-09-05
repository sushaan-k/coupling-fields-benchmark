"""Optional integration checks against the external metafor implementation."""

import csv
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq, minimize
from scipy.special import gammaln, logsumexp
from scipy.stats import norm


RSCRIPT = shutil.which("Rscript")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def r_environment():
    if not RSCRIPT:
        pytest.skip("Rscript is unavailable")
    environment = os.environ.copy()
    library = environment.get("COUPLING_R_LIBRARY", "")
    check = subprocess.run(
        [RSCRIPT, "-e", (f'.libPaths(c("{library}",.libPaths())); ' if library else "") +
         'quit(status=if(all(sapply(c("metafor","lme4","BiasedUrn"),'
         'requireNamespace,quietly=TRUE))) 0 else 1)'],
        capture_output=True, env=environment, timeout=30,
    )
    if check.returncode:
        pytest.skip("metafor exact-conditional dependencies are unavailable")
    return environment


def run_fixture(tmp_path, counts, environment):
    source, output = tmp_path / "input.csv", tmp_path / "output.csv"
    with source.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["donor", "pair", "a", "b", "c", "d"])
        writer.writerows((i, "pair", a, 20 - a, 20 - a, a) for i, a in enumerate(counts))
    subprocess.run(
        [RSCRIPT, str(ROOT / "experiments/development/fit_conditional_random_effects.R"),
         str(source), str(output)],
        capture_output=True, check=True, timeout=300, env=environment,
    )
    with output.open() as stream:
        return next(csv.DictReader(stream))


def test_official_fit_matches_independent_integrated_likelihood(tmp_path, r_environment):
    observed = [15, 10, 13, 7, 17, 9, 12, 15]
    result = run_fixture(tmp_path, observed, r_environment)
    assert result["status"] == "OK", result
    support = np.arange(21)
    log_weight = 2 * (gammaln(21) - gammaln(support + 1) - gammaln(21 - support))

    def objective(parameters):
        mu, log_sd = parameters

        def probability(z, observed_count):
            logits = log_weight + (mu + np.exp(log_sd) * z) * support
            return np.exp(logits[observed_count] - logsumexp(logits)) * norm.pdf(z)

        return -sum(np.log(quad(probability, -12, 12, args=(a,),
                                epsabs=1e-12, epsrel=1e-12)[0]) for a in observed)

    independent = minimize(objective, [1, np.log(1.2)], method="BFGS", options={"gtol": 1e-7})
    assert independent.success
    assert float(result["mu"]) == pytest.approx(independent.x[0], abs=1e-5)
    assert float(result["tau2"]) == pytest.approx(np.exp(2 * independent.x[1]), abs=1e-5)
    assert float(result["random_loglik"]) == pytest.approx(-independent.fun, abs=1e-7)


def test_identical_tables_have_certified_global_zero_variance_optimum(tmp_path, r_environment):
    result = run_fixture(tmp_path, [12] * 8, r_environment)
    assert result["status"] == "OK"
    assert result["fit_route"] == "analytic_identical_table_boundary"
    assert float(result["tau2"]) == 0
    assert float(result["boundary_score_abs"]) < 1e-7
    support = np.arange(21)
    log_weight = 2 * (gammaln(21) - gammaln(support + 1) - gammaln(21 - support))

    def probability(theta):
        logits = log_weight + theta * support
        return np.exp(logits - logsumexp(logits))

    optimum = brentq(lambda theta: probability(theta) @ support - 12, -5, 5)
    assert float(result["mu"]) == pytest.approx(optimum, abs=1e-7)
    boundary_loglik = 8 * np.log(probability(optimum)[12])
    assert float(result["random_loglik"]) == pytest.approx(boundary_loglik, abs=1e-7)
    for mu in (-2, optimum, 2):
        for sd in (0.1, 0.5, 1, 2):
            marginal = quad(lambda z: probability(mu + sd * z)[12] * norm.pdf(z),
                            -12, 12, epsabs=1e-12, epsrel=1e-12)[0]
            assert 8 * np.log(marginal) <= boundary_loglik + 1e-9


def test_nonidentical_zero_variance_candidate_requires_score_and_profile_checks(tmp_path, r_environment):
    result = run_fixture(tmp_path, [11, 12] * 4, r_environment)
    assert result["status"] == "OK", result
    assert result["fit_route"] == "profile_checked_zero_variance_boundary"
    assert float(result["tau2"]) == 0
    assert float(result["boundary_variance_score"]) < 0
    assert len(result["variance_profile"].split(" | ")) == 12
    assert abs(float(result["independent_loglik_difference"])) < 1e-6
    assert "nlminb[1000]" in result["optimizer_attempts"]
    assert "BFGS[1000]" in result["optimizer_attempts"]


@pytest.mark.parametrize("count,direction", [(0, -1), (20, 1)])
def test_shared_endpoint_is_extended_mle_with_unidentified_variance(tmp_path, r_environment, count, direction):
    result = run_fixture(tmp_path, [count] * 8, r_environment)
    assert result["status"] == "OK"
    assert result["fit_route"] == "extended_common_support_endpoint"
    assert int(result["boundary"]) == direction
    assert float(result["mu"]) == direction * np.inf
    assert result["tau2"] == ""
    assert float(result["random_loglik"]) == 0


def test_failed_approximate_initializer_retries_the_same_exact_likelihood(tmp_path, r_environment):
    counts = [
        [254, 243, 2, 13], [255, 240, 1, 16], [255, 228, 1, 28],
        [254, 242, 2, 14], [252, 235, 4, 21], [255, 232, 1, 24],
        [252, 241, 4, 15], [254, 196, 2, 60], [254, 242, 2, 14],
        [256, 232, 0, 24], [254, 212, 2, 44], [254, 227, 2, 29],
    ]
    source, output = tmp_path / "input.csv", tmp_path / "output.csv"
    with source.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["donor", "pair", "a", "b", "c", "d"])
        writer.writerows([i, "pair", *table] for i, table in enumerate(counts))
    subprocess.run(
        [RSCRIPT, str(ROOT / "experiments/development/fit_conditional_random_effects.R"),
         str(source), str(output)],
        capture_output=True, check=True, timeout=300, env=r_environment,
    )
    with output.open() as stream:
        result = next(csv.DictReader(stream))
    assert result["status"] == "OK", result
    assert result["fit_route"].endswith("initializer_retry")
    assert float(result["mu"]) == pytest.approx(2.6993542, abs=1e-5)
    assert float(result["tau2"]) == pytest.approx(0.2214448, abs=1e-5)
    assert float(result["random_loglik"]) == pytest.approx(-21.3897839, abs=1e-5)
    assert abs(float(result["independent_loglik_difference"])) < 1e-6


def test_zero_information_rows_are_constant_factors_not_fit_failures(tmp_path, r_environment):
    observed = [15, 10, 13, 7, 17, 9, 12, 15]
    counts = [[a, 20 - a, 20 - a, a] for a in observed] + [[20, 20, 0, 0]] * 3
    source, output = tmp_path / "input.csv", tmp_path / "output.csv"
    with source.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["donor", "pair", "a", "b", "c", "d"])
        writer.writerows([i, "pair", *table] for i, table in enumerate(counts))
    subprocess.run(
        [RSCRIPT, str(ROOT / "experiments/development/fit_conditional_random_effects.R"),
         str(source), str(output)], capture_output=True, check=True, timeout=300, env=r_environment,
    )
    with output.open() as stream:
        result = next(csv.DictReader(stream))
    assert result["status"] == "OK", result
    assert int(result["k"]) == 8
    assert result["informative_donors"].split(" | ") == [str(i) for i in range(8)]
    assert result["uninformative_donors"].split(" | ") == ["8", "9", "10"]
    assert float(result["mu"]) == pytest.approx(0.971209810553, abs=1e-5)
    assert float(result["tau2"]) == pytest.approx(1.50704916898, abs=1e-5)
    assert abs(float(result["independent_loglik_difference"])) < 1e-6
