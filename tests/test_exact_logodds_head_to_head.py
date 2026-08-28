from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import evaluate_exact_logodds_head_to_head as benchmark
from mapreg.hierarchical_conditional_coupling import (
    fit_hierarchical_conditional_log_odds,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/development/exact_logodds_head_to_head_v1.json"


def _synthetic_tables() -> np.ndarray:
    generator = np.random.default_rng(41)
    tables = np.empty((3, 2, 2, 2, 2), dtype=int)
    for index in np.ndindex(3, 2, 2):
        total = 40
        row_zero = int(generator.integers(8, 32))
        column_zero = int(generator.integers(8, 32))
        lower = max(0, row_zero + column_zero - total)
        upper = min(row_zero, column_zero)
        upper_left = int(generator.integers(lower, upper + 1))
        tables[index] = [
            [upper_left, row_zero - upper_left],
            [column_zero - upper_left, total - row_zero - column_zero + upper_left],
        ]
    return tables


def test_vectorized_exact_fit_matches_reference_estimator() -> None:
    tables = _synthetic_tables()
    incidence = np.asarray([[1.0], [1.0]])
    observed = benchmark._fit_exact_hierarchical(
        tables,
        incidence,
        incidence,
        heterogeneity=1.0,
        ridge=0.1,
        graph=0.3,
    )
    reference = fit_hierarchical_conditional_log_odds(
        tables,
        incidence,
        incidence,
        heterogeneity_penalty=1.0,
        ridge_penalty=0.1,
        graph_penalty=0.3,
        minimum_informative_donors=2,
        tolerance=benchmark.TOLERANCE,
    )
    np.testing.assert_allclose(
        observed.population_log_odds,
        reference.population_log_odds,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        observed.donor_log_odds, reference.donor_log_odds, rtol=1e-12, atol=1e-12
    )
    assert observed.objective == pytest.approx(reference.objective, abs=1e-12)
    assert observed.scaled_gradient_norm <= benchmark.TOLERANCE


def test_combat_loader_binds_the_authorized_reduction_bytes(tmp_path: Path) -> None:
    altered = tmp_path / "reduced.json"
    altered.write_text("{}\n")
    with pytest.raises(PermissionError, match="SHA-256 differs"):
        benchmark._combat_panel(altered)


def test_bmmc_destroyed_link_is_deterministic_and_stratum_preserving() -> None:
    adt = np.arange(48).reshape(8, 6)
    cells = [
        {"DonorID": donor, "cell_type.l1": lineage}
        for donor in ("fit-a", "fit-b")
        for lineage in ("B", "T")
        for _ in range(2)
    ]
    first = benchmark._destroy_bmmc_adt(adt, cells)
    second = benchmark._destroy_bmmc_adt(adt, cells)
    np.testing.assert_array_equal(first, second)
    for donor in ("fit-a", "fit-b"):
        for lineage in ("B", "T"):
            members = [
                index
                for index, row in enumerate(cells)
                if row == {"DonorID": donor, "cell_type.l1": lineage}
            ]
            assert Counter(map(tuple, first[members])) == Counter(
                map(tuple, adt[members])
            )


def test_paired_bootstrap_is_deterministic_and_counts_units() -> None:
    primary = np.asarray([1.0, 2.0, 4.0, 3.0])
    comparator = np.asarray([2.0, 2.5, 3.0, 4.0])
    first = benchmark._comparison(primary, comparator, seed=17)
    second = benchmark._comparison(primary, comparator, seed=17)
    assert first == second
    assert first["favorable_units"] == 3
    assert first["total_units"] == 4
    assert len(first["paired_difference_primary_minus_comparator"]) == 4


def test_public_result_reproduces_both_nonheld_development_directions() -> None:
    result = json.loads(RESULT.read_text())
    assert result["status"] == "RETROSPECTIVE_ADAPTIVE_DEVELOPMENT_ONLY"
    assert (
        result["bindings"]["evaluator_sha256"]
        == hashlib.sha256(
            (ROOT / "experiments/evaluate_exact_logodds_head_to_head.py").read_bytes()
        ).hexdigest()
    )

    combat = result["panels"]["combat"]
    combat_primary = combat["selection"]["primary"]
    for key, value in {
        "alpha": 1.0,
        "graph": 0.0,
        "heterogeneity": 10.0,
        "mean_loss": 0.011557968134894311,
        "neighbors": 1,
        "ridge": 0.01,
    }.items():
        assert combat_primary[key] == pytest.approx(value)
    assert combat["selection"]["best_residual"]["family"] == "deviance"
    assert combat["selection"]["best_residual"]["alpha"] == 0.75
    assert combat["selection"]["best_residual"]["mean_loss"] == pytest.approx(
        0.015605178347566027, abs=1e-14
    )
    assert combat["comparisons"]["primary_vs_best_residual"][
        "relative_reduction"
    ] == pytest.approx(0.2593504619127258, abs=1e-12)

    bmmc = result["panels"]["bmmc"]
    bmmc_primary = bmmc["selection"]["primary"]
    for key, value in {
        "alpha": 1.0,
        "graph": 0.0,
        "heterogeneity": 0.1,
        "mean_loss": 0.010850708394108175,
        "neighbors": 1,
        "ridge": 0.01,
    }.items():
        assert bmmc_primary[key] == pytest.approx(value)
    assert bmmc["selection"]["best_residual"]["family"] == "deviance"
    assert bmmc["selection"]["best_residual"]["centered"] is True
    assert bmmc["selection"]["best_residual"]["alpha"] == 0.75
    assert bmmc["selection"]["best_residual"]["mean_loss"] == pytest.approx(
        0.013147523553882618, abs=1e-14
    )
    assert bmmc["comparisons"]["primary_vs_best_residual"][
        "relative_reduction"
    ] == pytest.approx(0.17469564898373324, abs=1e-12)

    for panel in (combat, bmmc):
        assert panel["status"] == "RETROSPECTIVE_ADAPTIVE_DEVELOPMENT_ONLY"
        assert (
            panel["selection"]["same_nonheld_units_used_for_selection_and_summary"]
            is True
        )
        assert panel["selection"]["confirmatory_inference"] is False
        assert panel["graph_diagnostic"]["primary_grid_included_graph_zero"] is True
        assert panel["graph_diagnostic"]["graph_selected"] is False
        assert (
            panel["comparisons"]["primary_vs_best_residual"][
                "paired_unit_bootstrap_95_ci"
            ][1]
            < 0.0
        )
        assert (
            panel["comparisons"]["primary_vs_destroyed_link"][
                "paired_unit_bootstrap_95_ci"
            ][1]
            < 0.0
        )

    audit = result["access_audit"]
    assert audit["combat_held_access"] == 0
    assert audit["bmmc_held_numeric_access"] == 0
    assert audit["held_predictions_formed"] == 0
    assert audit["bmmc"]["held_feature_rows_decoded"] == 0
    assert audit["bmmc"]["held_tables_formed"] == 0
