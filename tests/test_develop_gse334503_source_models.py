from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments import develop_gse334503_source_models as subject
from mapreg.heterogeneity_adaptive_coupling import product_hypergraph_laplacian


def _source_data(seed: int = 17) -> subject.SourceData:
    generator = np.random.default_rng(seed)
    donors = [
        donor
        for batch in subject.SOURCE_BATCHES
        for donor in subject.EXPECTED_DONORS[batch]
    ]
    batches = [
        batch
        for batch in subject.SOURCE_BATCHES
        for _ in subject.EXPECTED_DONORS[batch]
    ]
    rna = generator.binomial(
        1, 0.45, size=(len(donors), subject.CELL_BUDGET, subject.MARKER_COUNT)
    ).astype(np.int32)
    adt_detected = generator.binomial(
        1, 0.4, size=(len(donors), subject.CELL_BUDGET, subject.MARKER_COUNT)
    )
    adt = (
        adt_detected
        * generator.integers(
            1,
            8,
            size=(len(donors), subject.CELL_BUDGET, subject.MARKER_COUNT),
        )
    ).astype(np.int32)
    profile = generator.normal(size=(len(donors), subject.MARKER_COUNT))
    profile[6:] += np.linspace(-0.4, 0.6, subject.MARKER_COUNT)
    barcodes = [
        [f"{batch}-{donor}-{cell:04d}" for cell in range(subject.CELL_BUDGET)]
        for donor, batch in zip(donors, batches)
    ]
    return subject.SourceData(
        donors=donors,
        batches=batches,
        barcodes=barcodes,
        rna_counts=rna,
        adt_counts=adt,
        adt_clr_profile=profile,
        profile_input_key="adt_graph_profile",
        manifest_sha256="fixture",
        manifest={},
    )


def _write_source(path: Path, data: subject.SourceData) -> None:
    np.savez_compressed(
        path,
        donor_axis=np.asarray(data.donors),
        day_axis=np.asarray(["Day0"] * len(data.donors)),
        batch_axis=np.asarray(data.batches),
        hto_feature_axis=np.asarray(
            [f"{donor.removeprefix('Donor')}-V1" for donor in data.donors]
        ),
        rna_gene_axis=np.asarray([gene for gene, _ in subject.PANEL]),
        adt_protein_axis=np.asarray([protein for _, protein in subject.PANEL]),
        selected_barcodes=np.asarray(data.barcodes),
        rna_counts=data.rna_counts,
        adt_counts=data.adt_counts,
        adt_graph_profile=data.adt_clr_profile,
    )


def _write_manifest(path: Path, source_path: Path) -> None:
    proteins = [protein for _, protein in subject.PANEL]
    denominator = [*proteins, *(f"protein-{index}" for index in range(108))]
    payload = {
        "schema": "gse334503-source-reduction/1.0",
        "status": "COMPLETE",
        "accession": "GSE334503",
        "stage": "source_development",
        "numeric_batches_processed": list(subject.SOURCE_BATCHES),
        "donor_count": 12,
        "cell_budget_per_donor": subject.CELL_BUDGET,
        "output_path": subject._display_path(source_path),
        "output_bytes": source_path.stat().st_size,
        "output_sha256": subject._sha256(source_path),
        "reducer_sha256": subject._sha256(
            subject.ROOT / "experiments/reduce_gse334503_source.py"
        ),
        "panel": [
            {"rna_gene": gene, "adt_protein": protein}
            for gene, protein in subject.PANEL
        ],
        "cell_selection": {"visit": "Day0"},
        "adt_graph_profile": {
            "cell_clr_formula": (
                "clr_cell = log1p(count_130) - mean_j(log1p(count_130))"
            ),
            "donor_profile_formula": (
                "mean_cells(clr_cell), restricted to the 22 cognate positions"
            ),
            "denominator_feature_count": 130,
            "denominator_adt_axis": denominator,
            "denominator_adt_axis_sha256": subject._newline_axis_sha256(denominator),
            "denominator_adt_set_sha256": subject._newline_axis_sha256(
                sorted(denominator)
            ),
            "cognate_positions_zero_based": list(range(subject.MARKER_COUNT)),
            "cognate_adt_axis": proteins,
        },
    }
    path.write_text(json.dumps(payload))


def test_loader_enforces_exact_b1_b2_axis_and_profile_contract(tmp_path: Path) -> None:
    path = tmp_path / "source.npz"
    manifest = tmp_path / "manifest.json"
    expected = _source_data()
    _write_source(path, expected)
    _write_manifest(manifest, path)
    observed = subject._load_source(path, manifest)
    assert observed.donors == expected.donors
    assert observed.batches == ["Batch1"] * 6 + ["Batch2"] * 6
    assert observed.profile_input_key == "adt_graph_profile"
    np.testing.assert_array_equal(observed.rna_counts, expected.rna_counts)

    forbidden = _source_data()
    forbidden.batches[-1] = "Batch3"
    forbidden_path = path.with_name("forbidden.npz")
    forbidden_manifest = path.with_name("forbidden-manifest.json")
    _write_source(forbidden_path, forbidden)
    _write_manifest(forbidden_manifest, forbidden_path)
    with pytest.raises(ValueError, match="frozen B1-B2 axis"):
        subject._load_source(forbidden_path, forbidden_manifest)


def test_loader_rejects_manifest_or_npz_drift_before_cast(tmp_path: Path) -> None:
    path = tmp_path / "source.npz"
    manifest = tmp_path / "manifest.json"
    data = _source_data()
    _write_source(path, data)
    _write_manifest(manifest, path)

    payload = json.loads(manifest.read_text())
    payload["output_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="wrong output_sha256"):
        subject._load_source(path, manifest)

    overflow = _source_data()
    overflow.rna_counts = overflow.rna_counts.astype(np.int64)
    overflow.rna_counts[0, 0, 0] = np.iinfo(np.int32).max + 1
    overflow_path = tmp_path / "overflow.npz"
    overflow_manifest = tmp_path / "overflow-manifest.json"
    _write_source(overflow_path, overflow)
    _write_manifest(overflow_manifest, overflow_path)
    with pytest.raises(ValueError, match="outside int32 range"):
        subject._load_source(overflow_path, overflow_manifest)


def test_fold_mask_profiles_and_hypergraphs_ignore_validation_batch() -> None:
    first = _source_data(23)
    second = _source_data(23)
    second.rna_counts[6:] = 1 - second.rna_counts[6:]
    second.adt_counts[6:] = second.adt_counts[6:, ::-1]
    second.adt_clr_profile[6:] += 1000.0

    first_records = subject._records(first)
    second_records = subject._records(second)
    training = first.donors[:6]
    first_design = subject._training_design(first_records, training, 2)
    second_design = subject._training_design(second_records, training, 2)

    np.testing.assert_array_equal(first_design["mask"], second_design["mask"])
    np.testing.assert_array_equal(
        first_design["rna_normalized"], second_design["rna_normalized"]
    )
    np.testing.assert_array_equal(
        first_design["adt_normalized"], second_design["adt_normalized"]
    )
    np.testing.assert_array_equal(
        first_design["rna_incidence"], second_design["rna_incidence"]
    )
    np.testing.assert_array_equal(
        first_design["adt_incidence"], second_design["adt_incidence"]
    )


def test_adt_estimand_is_observed_detection_without_midrank_assignment() -> None:
    data = _source_data(29)
    data.adt_counts[0, :, 0] = 0
    data.adt_counts[0, :8, 0] = 9
    data.adt_counts[0, :, 1] = 0
    data.adt_counts[0, :7, 1] = 9
    record = subject._records(data)[data.donors[0]]
    assert record["adt_marker_support"][:2].tolist() == [True, False]
    np.testing.assert_array_equal(
        record["tables"][0, 0].sum(axis=0), np.asarray([504, 8])
    )
    np.testing.assert_array_equal(
        record["destroyed_tables"][0, 0].sum(axis=0), np.asarray([504, 8])
    )


@pytest.mark.parametrize("neighbors", subject.NEIGHBOR_GRID)
def test_marker_centered_hyperedges_are_deterministic_and_genuine(
    neighbors: int,
) -> None:
    profiles = np.random.default_rng(31).normal(size=(6, subject.MARKER_COUNT))
    first = subject._marker_hyperedges(profiles, neighbors)
    second = subject._marker_hyperedges(profiles.copy(), neighbors)
    np.testing.assert_array_equal(first, second)
    assert np.all(first.sum(axis=0) == neighbors + 1)
    assert np.all(first.sum(axis=0) > 2)
    assert len({tuple(column) for column in first.T}) == first.shape[1]


def test_matched_graph_zero_changes_only_structural_penalty() -> None:
    primary = subject.StructuredConfig(3, 1.0, 0.1, 0.3, 1.25)
    matched = subject._matched_zero_config(primary)
    assert matched.graph_penalty == 0.0
    assert matched.graph_neighbors == primary.graph_neighbors
    assert matched.heterogeneity_penalty == primary.heterogeneity_penalty
    assert matched.ridge_penalty == primary.ridge_penalty
    assert matched.transport_multiplier == primary.transport_multiplier


def test_structural_selection_and_source_gate_are_frozen() -> None:
    batches = ["Batch1"] * 6 + ["Batch2"] * 6
    weak = subject.StructuredConfig(2, 1.0, 0.1, 0.01, 1.0)
    strong = subject.StructuredConfig(3, 1.0, 0.1, 0.1, 1.0)
    losses = {
        weak: np.full(12, 0.96),
        strong: np.full(12, 0.80),
    }
    assert subject._select_structured(losses, batches) == strong

    gate = subject._source_gate(strong, np.full(12, 0.80), np.full(12, 1.0), batches)
    assert gate["passes"]
    assert gate["favorable_donors"] == 12

    only_nine = np.asarray([0.8] * 9 + [1.1] * 3)
    failed = subject._source_gate(strong, only_nine, np.ones(12), batches)
    assert not failed["passes"]
    assert not failed["checks"]["at_least_10_of_12_donors_improve"]

    residual_losses = {
        subject.ResidualConfig("pearson", 1.0): np.full(12, 0.7),
        subject.ResidualConfig("root_deviance", 0.75): np.full(12, 0.8),
    }
    selected_residual = subject._select_residual_family(
        residual_losses,
        {"pearson": 1.0, "root_deviance": 0.75},
        batches,
    )
    assert selected_residual == subject.ResidualConfig("pearson", 1.0)


def test_topology_nulls_are_deterministic_and_spectrum_preserving() -> None:
    profiles = np.random.default_rng(43).normal(size=(12, subject.MARKER_COUNT))
    first = subject._marker_hyperedges(profiles, 3)
    second = subject._marker_hyperedges(profiles[:, ::-1], 3)
    baseline = np.linalg.eigvalsh(product_hypergraph_laplacian(first, second))

    rna_permutation = subject._permutation(0, "rna")
    adt_permutation = subject._permutation(0, "adt")
    np.testing.assert_array_equal(rna_permutation, subject._permutation(0, "rna"))
    assert not np.array_equal(rna_permutation, np.arange(subject.MARKER_COUNT))
    permuted = np.linalg.eigvalsh(
        product_hypergraph_laplacian(first[rna_permutation], second[adt_permutation])
    )
    np.testing.assert_allclose(permuted, baseline, rtol=0.0, atol=1e-10)


def test_all_topology_null_permutation_pairs_are_unique_and_nonidentity() -> None:
    identity = tuple(range(subject.MARKER_COUNT))
    rna_permutations = {
        tuple(subject._permutation(control, "rna"))
        for control in range(subject.TOPOLOGY_NULL_COUNT)
    }
    adt_permutations = {
        tuple(subject._permutation(control, "adt"))
        for control in range(subject.TOPOLOGY_NULL_COUNT)
    }
    pairs = {
        (
            tuple(subject._permutation(control, "rna")),
            tuple(subject._permutation(control, "adt")),
        )
        for control in range(subject.TOPOLOGY_NULL_COUNT)
    }
    assert identity not in rna_permutations
    assert identity not in adt_permutations
    assert len(rna_permutations) == subject.TOPOLOGY_NULL_COUNT
    assert len(adt_permutations) == subject.TOPOLOGY_NULL_COUNT
    assert rna_permutations.isdisjoint(adt_permutations)
    assert len(pairs) == subject.TOPOLOGY_NULL_COUNT


def test_each_topology_null_reruns_complete_stage_b_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _source_data(47)
    records = subject._records(data)
    calls = []

    def fake_fit(tables, support, first, second, config):
        del tables, support, first, second
        calls.append(config)
        value = config.graph_neighbors + config.graph_penalty
        return {
            "population_log_odds": np.full(
                (subject.MARKER_COUNT, subject.MARKER_COUNT), value
            ),
            "fit_certificate": {"converged": True},
        }

    monkeypatch.setattr(subject, "TOPOLOGY_NULL_COUNT", 1)
    monkeypatch.setattr(subject, "_fit_structured", fake_fit)
    monkeypatch.setattr(
        subject,
        "_population_loss",
        lambda record, mask, population, alpha: float(population[0, 0]),
    )
    selected = subject._select_topology_nulls(
        data, records, subject.BaseConfig(1.0, 0.1, 1.0)
    )
    assert len(calls) == 2 * len(subject.NEIGHBOR_GRID) * len(subject.GRAPH_GRID)
    assert selected[0]["selected_configuration"] == {
        "graph_neighbors": 2,
        "heterogeneity_penalty": 1.0,
        "ridge_penalty": 0.1,
        "graph_penalty": 0.01,
        "transport_multiplier": 1.0,
    }
    assert len(selected[0]["loss_curve"]) == len(subject.NEIGHBOR_GRID) * len(
        subject.GRAPH_GRID
    )


def test_model_and_candidate_json_serialization_are_hash_bound(tmp_path: Path) -> None:
    population = np.arange(subject.MARKER_COUNT**2, dtype=float).reshape(
        subject.MARKER_COUNT, subject.MARKER_COUNT
    )
    model = subject._serialize_model(
        "fixture",
        {"graph_penalty": 0.1},
        {
            "population_log_odds": population,
            "fit_certificate": {"converged": True},
        },
    )
    payload = {
        "status": "SOURCE_GO_GATE_PASSED_CANDIDATE_FROZEN",
        "candidate": {
            "canonical_candidate_configuration": {"graph_penalty": 0.1},
            "models": {"primary": model},
        },
    }
    path = tmp_path / "candidate.json"
    subject._write_json_x(path, payload)
    restored = json.loads(path.read_text())
    assert restored == payload
    assert restored["candidate"]["models"]["primary"][
        "population_log_odds_sha256"
    ] == subject._array_sha256(population)
    with pytest.raises(FileExistsError):
        subject._write_json_x(path, payload)


def test_artifact_binds_the_versioned_penalty_complete_solver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source.npz"
    manifest_path = tmp_path / "manifest.json"
    source_path.write_bytes(b"source")
    manifest_path.write_text("{}")
    data = _source_data()
    data.manifest = {
        "schema": "gse334503-source-reduction/1.0",
        "output_sha256": subject._sha256(source_path),
    }
    data.manifest_sha256 = subject._sha256(manifest_path)
    monkeypatch.setattr(
        subject,
        "_develop",
        lambda _: {
            "status": "SOURCE_GO_GATE_FAILED",
            "b3_numeric_access_gate_passed": False,
            "development": {},
            "candidate": None,
        },
    )
    artifact = subject._artifact(source_path, manifest_path, data)
    solver_path = subject.ROOT / "mapreg/penalty_complete_conditional_coupling.py"
    assert artifact["implementation"]["conditional_solver"] == (
        "mapreg.penalty_complete_conditional_coupling."
        "fit_hierarchical_conditional_log_odds"
    )
    assert artifact["implementation"]["files_sha256"][
        "mapreg/penalty_complete_conditional_coupling.py"
    ] == subject._sha256(solver_path)
    assert set(artifact["implementation"]["files_sha256"]) == set(
        subject.IMPLEMENTATION_FILES
    )


def test_held_gate_is_fully_frozen_before_batch3_access() -> None:
    contract = subject._held_gate_contract()
    assert contract["ordered_donor_axes"] == {
        batch: list(donors) for batch, donors in subject.HELD_DONORS.items()
    }
    assert contract["stage_order"] == ["Batch3", "Batch4", "Batch5"]
    assert contract["evaluation"]["required_valid_donors_per_batch"] == 6
    assert contract["paired_bootstrap"]["draws"] == 20_000
    required = set(contract["mandatory_comparators"])
    assert required == {
        "matched_graph_zero",
        "common_effect_cmle",
        "pooled_saturated_poisson",
        "primary_classical_residual",
        "independence",
        "destroyed_link",
    }
    rules = contract["pass_rules_applied_separately_to_each_batch"]
    assert rules["relative_mean_loss_reduction_vs_matched_graph_zero_at_least"] == 0.05
    assert rules["matched_graph_zero_one_sided_sign_test_p_at_most"] == 1 / 64
    assert rules["topology_empirical_p_at_most"] == 0.05
    assert (
        rules["relative_mean_loss_reduction_vs_median_topology_null_at_least"] == 0.03
    )


def test_artifact_refuses_input_or_implementation_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source.npz"
    manifest_path = tmp_path / "manifest.json"
    source_path.write_bytes(b"source")
    manifest_path.write_text("{}")
    data = _source_data()
    data.manifest = {
        "schema": "gse334503-source-reduction/1.0",
        "output_sha256": subject._sha256(source_path),
    }
    data.manifest_sha256 = subject._sha256(manifest_path)

    def mutate_source(_: subject.SourceData) -> dict:
        source_path.write_bytes(b"changed")
        return {}

    monkeypatch.setattr(subject, "_develop", mutate_source)
    with pytest.raises(PermissionError, match="source bytes changed"):
        subject._artifact(source_path, manifest_path, data)

    source_path.write_bytes(b"source")
    snapshots = iter([{"state": 1}, {"state": 2}])
    monkeypatch.setattr(subject, "_develop", lambda _: {})
    monkeypatch.setattr(subject, "_implementation_snapshot", lambda: next(snapshots))
    with pytest.raises(PermissionError, match="implementation bytes changed"):
        subject._artifact(source_path, manifest_path, data)


def test_runtime_requires_the_frozen_single_thread_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, expected in subject.REQUIRED_THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, expected)
    subject._require_runtime()
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(PermissionError, match="OMP_NUM_THREADS"):
        subject._require_runtime()
