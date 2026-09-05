import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from experiments.development import fixed_margin_biology as biology


def panel():
    rng = np.random.default_rng(41)
    donors, cells = 6, 64
    return {
        "rna_counts": rng.poisson(0.7, (donors, 9, cells)),
        "adt_counts": rng.poisson(5, (donors, 9, cells)),
        "cell_types": np.tile(np.repeat(["T", "B"], cells // 2), (donors, 1)),
        "barcodes": np.tile([f"cell-{i:03d}" for i in range(cells)], (donors, 1)),
        "donor_ids": np.array([f"donor-{i}" for i in range(donors)]),
        "sample_ids": np.array([f"sample-{i}" for i in range(donors)]),
        "roles": np.array(["calibration"] * 2 + ["pilot"] * 2 + ["held_site"] * 2),
        "markers": np.array(biology.stephenson.MARKERS),
    }


def test_data_contract_rejects_ambiguous_donors_and_roles():
    data = panel()
    biology.validate_data(data)
    data["donor_ids"][1] = data["donor_ids"][0]
    with pytest.raises(ValueError, match="physical donors"):
        biology.validate_data(data)
    data = panel()
    data["roles"][0] = "unknown"
    with pytest.raises(ValueError, match="roles"):
        biology.validate_data(data)


def test_lineage_permutation_preserves_complete_vectors_and_is_order_invariant():
    data = panel()
    states = data["adt_counts"][0]
    labels, barcodes = data["cell_types"][0], data["barcodes"][0]
    shuffled = biology.permute_within_cell_types(states, labels, barcodes, "d", 0)
    assert not np.array_equal(states, shuffled)
    for label in set(labels):
        indices = labels == label
        assert sorted(map(tuple, states[:, indices].T)) == sorted(map(tuple, shuffled[:, indices].T))
    order = np.random.default_rng(7).permutation(len(labels))
    reordered = biology.permute_within_cell_types(
        states[:, order], labels[order], barcodes[order], "d", 0
    )
    assert np.array_equal(reordered, shuffled[:, order])
    assert not np.array_equal(shuffled, biology.permute_within_cell_types(
        states, labels, barcodes, "d", 1
    ))


def test_lineage_only_association_survives_the_null():
    labels = np.array(["T"] * 16 + ["B"] * 16)
    states = np.tile(labels == "T", (9, 1)).astype(np.uint8)
    shuffled = biology.permute_within_cell_types(
        states, labels, np.array([str(i) for i in range(32)]), "donor", 0
    )
    assert np.array_equal(states, shuffled)
    assert np.array_equal(biology.make_tables(states[None], states[None]),
                          biology.make_tables(states[None], shuffled[None]))


def test_source_thresholds_and_pair_support_do_not_use_held_values():
    data = panel()
    source = data["roles"] != "held_site"
    threshold = biology.source_median_thresholds(data["adt_counts"], source)
    states = data["adt_counts"] > threshold[None, :, None]
    tables = biology.make_tables(data["rna_counts"] > 0, states)
    support = biology.informative(tables[source]).sum(axis=0) >= 2
    data["adt_counts"][~source] = 1_000_000
    data["rna_counts"][~source] = 0
    new = biology.source_median_thresholds(data["adt_counts"], source)
    new_tables = biology.make_tables(data["rna_counts"] > 0,
                                    data["adt_counts"] > new[None, :, None])
    assert np.array_equal(new, threshold)
    assert np.array_equal(support, biology.informative(new_tables[source]).sum(axis=0) >= 2)


def test_rank_tables_match_original_512_cell_state_construction():
    data = panel()
    for name in ("rna_counts", "adt_counts", "cell_types"):
        data[name] = np.tile(data[name], (1, 1, 8) if data[name].ndim == 3 else (1, 8))
    data["barcodes"] = np.tile([f"cell-{i:03d}" for i in range(512)], (6, 1))
    observed = biology.tables_from_counts(data)
    expected = biology.make_tables(data["rna_counts"] > 0, np.stack([
        biology.stephenson._adt_states(counts, bars, str(donor), str(sample))
        for counts, bars, donor, sample in zip(
            data["adt_counts"], data["barcodes"], data["donor_ids"], data["sample_ids"]
        )
    ]))
    np.testing.assert_array_equal(observed, expected)


def test_sufficient_table_export_contains_no_cell_vectors_and_preserves_margins():
    data = panel()
    result = biology.sufficient_tables(data)
    assert not set(result) & {"barcodes", "cell_types", "rna_counts", "adt_counts"}
    source = data["roles"] != "held_site"
    assert result["source_lineage_null_tables"].shape == (8, 4, 9, 9, 2, 2)
    np.testing.assert_array_equal(result["held_rank_tables"], biology.tables_from_counts(data)[~source])
    np.testing.assert_array_equal(result["held_threshold_tables"],
                                  biology.tables_from_counts(data, "source_median")[~source])
    for null in result["source_lineage_null_tables"]:
        for axis in (-1, -2):
            np.testing.assert_array_equal(null.sum(axis=axis),
                                          result["source_rank_tables"].sum(axis=axis))


def test_real_estimators_fit_and_all_held_donors_are_reported():
    data = panel()
    tables = biology.make_tables(data["rna_counts"] > 0, data["adt_counts"] > 5)
    mask = np.zeros((9, 9), dtype=bool)
    mask[0, :2] = True
    for method in ("hierarchical", "common_conditional"):
        fit = biology.fit_field(tables[:4], mask, method)
        assert fit["status"] == "FITTED"
        held = tables[4:].copy()
        held[1] = 0
        held[1, :, :, 0, 0] = 64
        rows = biology.score_field(fit, held, mask, data["donor_ids"][4:])
        assert len(rows) == 2
        assert rows[0]["status"] == "SCORED"
        assert np.isfinite(rows[0]["loss"])
        assert rows[1]["status"] == "NO_INFORMATIVE_PAIRS"


def test_every_failed_null_fit_is_retained_and_output_is_json_safe(monkeypatch):
    calls = []

    def fail_fit(tables, mask, method):
        calls.append(method)
        return {"status": "FIT_FAILED", "reason": "synthetic convergence failure"}

    monkeypatch.setattr(biology, "fit_field", fail_fit)
    result = biology.analyze(panel())
    json.dumps(result, allow_nan=False)
    null = result["composition_preserving_null"]
    assert null["attempted_repeats"] == 8
    assert null["fitted_repeats"] == 0
    assert null["aggregate"]["status"] == "INCOMPLETE_REPETITIONS"
    assert len(null["repeats"]) == 8
    assert all(len(row["donor_results"]) == 2 for row in null["repeats"])
    assert len(calls) == 11
    assert result["source_median_threshold"]["comparison"]["donors"] == 0


def test_counts_and_aggregate_entrypoints_share_the_same_analysis(monkeypatch):
    def fit_independence(tables, mask, method):
        return {"status": "FITTED", "log_odds": np.zeros(int(mask.sum())),
                "boundary": np.zeros(int(mask.sum()), dtype=int)}

    monkeypatch.setattr(biology, "fit_field", fit_independence)
    data = panel()
    archive = biology.sufficient_tables(data)
    original = biology.analyze(data)
    replay = biology.analyze_tables(archive)
    assert original["composition_preserving_null"] == replay["composition_preserving_null"]
    for key, value in replay["source_median_threshold"].items():
        assert original["source_median_threshold"][key] == value
    archive["source_pair_mask"][0, 0] = False
    with pytest.raises(ValueError, match="source-only eligibility"):
        biology.analyze_tables(archive)


def test_boundary_prediction_with_positive_truth_is_not_silently_dropped():
    fit = {"status": "FITTED", "log_odds": np.array([0.0]), "boundary": np.array([1])}
    tables = np.ones((1, 1, 1, 2, 2), dtype=int)
    rows = biology.score_field(fit, tables, np.ones((1, 1), bool), np.array(["d"]))
    assert rows == [{"donor": "d", "informative_pairs": 1,
                     "source_supported_pairs": 1, "loss": None,
                     "status": "INFINITE_DEVIANCE", "infinite_pairs": 1}]


def test_released_aggregate_replay_matches_every_original_scientific_result():
    root = Path(__file__).resolve().parents[1]
    original = json.loads((root / "results/development/stephenson_biological_reanalysis.json").read_text())
    replay = json.loads((root / "results/development/stephenson_biological_reanalysis_replay.json").read_text())
    assert original["composition_preserving_null"] == replay["composition_preserving_null"]
    for key, value in replay["source_median_threshold"].items():
        assert original["source_median_threshold"][key] == value
    digest = hashlib.sha256(
        (root / "data/development/stephenson_biological_sufficient_tables.npz").read_bytes()
    ).hexdigest()
    assert original["bindings"]["sufficient_tables_sha256"] == digest
    assert replay["bindings"]["sufficient_tables_sha256"] == digest
    assert original["original_loss_max_absolute_error"] < 1e-10
