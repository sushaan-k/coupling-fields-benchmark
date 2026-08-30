from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import audit_classical_interaction_baselines as audit


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/development/classical_interaction_baselines_posthoc.json"
TABLE = ROOT / "results/development/classical_interaction_baselines_posthoc.tsv"


def _counterexample() -> np.ndarray:
    return np.asarray(
        [
            [[[1, 9], [11, 3]]],
            [[[8, 2], [1, 9]]],
        ],
        dtype=np.int64,
    ).reshape(2, 1, 1, 2, 2)


def test_pooled_poisson_interaction_is_exactly_the_pooled_sample_log_odds() -> None:
    tables = _counterexample()
    fitted = audit._pooled_poisson_interaction(tables)
    pooled = tables.sum(axis=0)[0, 0]
    expected = np.log(pooled[0, 0] * pooled[1, 1]) - np.log(
        pooled[0, 1] * pooled[1, 0]
    )
    assert fitted.log_odds[0, 0] == pytest.approx(expected, abs=1e-14)
    assert fitted.certificate["closed_form"] == (
        "beta = log(N00*N11/(N01*N10))"
    )


def test_stratified_exact_cmle_is_not_a_renamed_pooled_poisson_fit() -> None:
    tables = _counterexample()
    exact = audit._common_effect_exact_cmle(tables)
    poisson = audit._pooled_poisson_interaction(tables)
    assert exact.log_odds[0, 0] == pytest.approx(
        -0.17715362640729543, abs=1e-12
    )
    assert poisson.log_odds[0, 0] == pytest.approx(
        -0.20067069546215066, abs=1e-12
    )
    assert abs(exact.log_odds[0, 0] - poisson.log_odds[0, 0]) > 0.02
    assert exact.certificate["maximum_absolute_finite_score"] < 1e-10


def test_both_classical_fields_reconstruct_the_requested_margins() -> None:
    rows = np.asarray([[10, 14]])
    columns = np.asarray([[12, 12]])
    for field in (
        audit._common_effect_exact_cmle(_counterexample()),
        audit._pooled_poisson_interaction(_counterexample()),
    ):
        prediction = audit._predict_field(field, rows, columns, alpha=0.75)
        np.testing.assert_allclose(prediction.sum(axis=-1), rows[:, None, :])
        np.testing.assert_allclose(prediction.sum(axis=-2), columns[None, :, :])


def test_paired_donor_bootstrap_is_deterministic() -> None:
    primary = np.asarray([1.0, 1.5, 2.0, 1.25])
    comparator = np.asarray([1.2, 2.0, 1.9, 1.5])
    first = audit._comparison(
        ["a", "b", "c", "d"], primary, comparator, label="test"
    )
    second = audit._comparison(
        ["a", "b", "c", "d"], primary, comparator, label="test"
    )
    assert first == second
    assert first["favorable_donors"] == 3
    assert first["post_hoc_inference"] is True


def test_public_posthoc_artifacts_bind_frozen_predictions_and_report_no_duplicate() -> None:
    result = json.loads(RESULT.read_text())
    assert result["status"] == "POST_HOC_NONCONFIRMATORY_BASELINE_AUDIT"
    assert result["confirmatory"] is False
    assert result["bindings"]["runner"]["sha256"] == hashlib.sha256(
        (ROOT / "experiments/audit_classical_interaction_baselines.py").read_bytes()
    ).hexdigest()

    boundary = result["equivalence_boundary"]
    assert boundary["poisson_equals_pooled_sample_log_odds"]["consequence"].endswith(
        "is not reported."
    )
    assert boundary["poisson_is_not_the_stratified_exact_cmle"][
        "absolute_difference"
    ] > 0.02

    stephenson_confirmation = json.loads(
        (ROOT / "results/stephenson_citeseq_confirmation.json").read_text()
    )
    corrected_gse = json.loads(
        (ROOT / "results/gse239452_citeseq_post_access_correction.json").read_text()
    )
    expected_primary = {
        "stephenson_newcastle_held_site": {
            row["donor"]: row["losses"]["primary"]
            for row in stephenson_confirmation["donor_results"]
        },
        "gse239452_held_cohort_post_access_correction": corrected_gse["held"][
            "losses"
        ]["primary"],
    }
    for study, row in result["studies"].items():
        assert row["status"] == "POST_HOC_BASELINE_AUDIT"
        assert row["confirmatory"] is False
        assert row["held_losses"]["primary"] == expected_primary[study]
        assert set(row["selection"]) == {
            "common_effect_exact_cmle",
            "pooled_poisson_loglinear_interaction",
        }
        for selection in row["selection"].values():
            assert selection["selected_alpha"] in audit.ALPHA_GRID
        for comparison in row["comparisons"].values():
            assert comparison["unit"] == "physical donor"
            assert comparison["bootstrap_draws"] == audit.BOOTSTRAPS
            assert len(comparison["paired_difference_95_ci"]) == 2
            assert comparison["post_hoc_inference"] is True

    with TABLE.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 6
    assert {row["analysis_status"] for row in rows} == {
        "post_hoc_nonconfirmatory"
    }
