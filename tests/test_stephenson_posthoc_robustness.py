import json
from pathlib import Path

import numpy as np

from experiments.development import analyze_stephenson_posthoc_robustness as posthoc


ROOT = Path(__file__).resolve().parents[1]


def test_generic_tables_preserve_binary_endpoint_margins():
    generator = np.random.default_rng(7)
    rna = generator.integers(0, 2, size=(9, 64), dtype=np.uint8)
    adt = generator.integers(0, 2, size=(9, 64), dtype=np.uint8)
    tables = posthoc._form_tables(rna, adt)
    assert tables.shape == (9, 9, 2, 2)
    assert np.all(tables.sum(axis=(-2, -1)) == 64)
    assert np.array_equal(
        tables.sum(axis=-1),
        np.broadcast_to(tables[:, :1].sum(axis=-1), (9, 9, 2)),
    )
    assert np.array_equal(
        tables.sum(axis=-2),
        np.broadcast_to(tables[:1].sum(axis=-2), (9, 9, 2)),
    )


def test_fixed_permutations_are_reproducible_distinct_and_margin_preserving():
    first, first_seed = posthoc._permutation("sample-a", 0, cells=64)
    replay, replay_seed = posthoc._permutation("sample-a", 0, cells=64)
    second, second_seed = posthoc._permutation("sample-a", 1, cells=64)
    other, other_seed = posthoc._permutation("sample-b", 0, cells=64)
    assert first_seed == replay_seed
    assert np.array_equal(first, replay)
    assert len(set(first.tolist())) == 64
    assert not np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert len({first_seed, second_seed, other_seed}) == 3
    states = np.arange(9 * 64).reshape(9, 64) % 2
    assert np.array_equal(states[:, first].sum(axis=1), states.sum(axis=1))


def test_lineage_selection_is_deterministic_and_reports_ineligible_donors():
    records = [
        {
            "donor": "d1",
            "sample": "s1",
            "site": "Cambridge",
            "role": "calibration",
        },
        {
            "donor": "d2",
            "sample": "s2",
            "site": "Ncl",
            "role": "held_site",
        },
    ]
    barcodes = np.asarray([f"bc-{index:03d}" for index in range(90)])
    samples = np.asarray(["s1"] * 70 + ["s2"] * 20)
    sites = np.asarray(["Cambridge"] * 70 + ["Ncl"] * 20)
    cell_types = np.asarray(["T"] * 90)
    selected, inventory = posthoc._lineage_selections(
        records,
        barcodes,
        samples,
        sites,
        cell_types,
        "T_cells",
        ("T",),
        budget=64,
    )
    replay, replay_inventory = posthoc._lineage_selections(
        records,
        barcodes,
        samples,
        sites,
        cell_types,
        "T_cells",
        ("T",),
        budget=64,
    )
    assert set(selected) == {"s1"}
    assert len(selected["s1"]) == 64
    assert np.array_equal(selected["s1"], replay["s1"])
    assert inventory == replay_inventory
    assert [row["status"] for row in inventory] == [
        "ELIGIBLE",
        "EXCLUDED_CELL_SUPPORT",
    ]


def test_paired_and_nested_bootstraps_have_fixed_finite_intervals():
    primary = np.asarray([0.7, 0.8, 0.9, 0.95])
    comparator = np.asarray([1.0, 1.0, 1.0, 1.0])
    result = posthoc._paired_comparison(
        ["a", "b", "c", "d"], primary, comparator, "test"
    )
    assert result["favorable_donors"] == 4
    assert result["relative_loss_reduction"] > 0.0
    assert result["paired_difference_95_ci"][1] <= 0.0
    differences = np.tile(primary - comparator, (5, 1))
    nested = posthoc._nested_bootstrap_interval(differences)
    assert nested["mean_intact_minus_destroyed"] < 0.0
    assert nested["nested_permutation_donor_bootstrap_95_ci"][1] <= 0.0


def test_posthoc_artifacts_report_all_lineages_and_all_permutations():
    lineage_path = (
        ROOT / "results/development/stephenson_lineage_sensitivity_v1.json"
    )
    permutation_path = (
        ROOT
        / "results/development/"
        "stephenson_destroyed_link_permutation_robustness_v1.json"
    )
    if not lineage_path.exists() or not permutation_path.exists():
        return
    lineage = json.loads(lineage_path.read_text())
    permutations = json.loads(permutation_path.read_text())
    assert lineage["lineages_requested"] == list(posthoc.LINEAGES)
    assert lineage["lineages_reported"] == list(posthoc.LINEAGES)
    assert set(lineage["lineages"]) == set(posthoc.LINEAGES)
    for row in lineage["lineages"].values():
        assert row["candidate_budget_per_method"] == 1
        assert row["held_newcastle_donors_scored"] > 0
        assert row["comparison"]["bootstrap_draws"] == posthoc.BOOTSTRAPS
    draws = permutations["independent_draws"]
    assert len(draws) == posthoc.INDEPENDENT_PERMUTATIONS
    assert len({row["permutation_sha256"] for row in draws}) == len(draws)
    assert sum(row["status"] == "EVALUATED" for row in draws) >= 20
    assert sum(row["status"] != "EVALUATED" for row in draws) == permutations[
        "independent_draw_distribution"
    ]["independent_permutations_refused"]
    assert max(permutations["replay_checks"].values()) < 1e-12
