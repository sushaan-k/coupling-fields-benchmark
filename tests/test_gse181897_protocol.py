from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = (
    ROOT / "data/confirmation/gse181897_control_citeseq/candidate_designation_v1.json"
)
PROTOCOL_PATH = ROOT / "data/confirmation/gse181897_control_citeseq/protocol_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _role_axis(candidate: dict, role: str) -> list[tuple[int, int]]:
    return [
        (record["batch"], exp_id)
        for record in candidate["allocation"][role]["batches"]
        for exp_id in record["exp_ids"]
    ]


def test_candidate_is_metadata_only_and_protocol_binds_exact_bytes() -> None:
    candidate = _load(CANDIDATE_PATH)
    protocol = _load(PROTOCOL_PATH)
    created = datetime.fromisoformat(candidate["created_at_utc"].replace("Z", "+00:00"))
    assert created.isoformat().endswith("+00:00")
    assert candidate["status"] == (
        "FROZEN_METADATA_ONLY_PENDING_PUBLIC_TAG_AND_AXIS_PREFLIGHT"
    )
    assert candidate["input"]["numeric_matrix_values_read_at_designation"] == 0
    assert candidate["outcome_access"]["ann_data_x_entries_read_at_designation"] == 0
    assert (
        candidate["outcome_access"]["internal_values_may_be_read_before_source_go"]
        is False
    )
    assert (
        candidate["outcome_access"][
            "confirmation_values_may_be_read_before_internal_pass"
        ]
        is False
    )
    assert protocol["candidate_binding"]["sha256"] == _sha256(CANDIDATE_PATH)
    assert protocol["status"] == (
        "FROZEN_METADATA_ONLY_NUMERIC_ACCESS_DISABLED_PENDING_PUBLIC_BINDINGS"
    )


def test_exact_control_cohorts_are_disjoint_and_ordered() -> None:
    candidate = _load(CANDIDATE_PATH)
    expected = {
        "source_development": [
            (0, 34),
            (0, 35),
            (0, 45),
            (0, 57),
            (0, 58),
            (1, 18),
            (1, 22),
            (1, 24),
            (1, 55),
            (1, 59),
            (1, 61),
            (2, 13),
            (2, 29),
            (2, 31),
            (2, 42),
            (2, 43),
            (3, 5),
            (3, 12),
            (3, 30),
            (3, 36),
            (3, 60),
            (3, 63),
            (4, 10),
            (4, 14),
            (4, 38),
            (4, 47),
            (5, 11),
            (5, 15),
            (5, 25),
            (5, 39),
            (5, 49),
            (6, 1),
            (6, 4),
            (6, 7),
            (6, 27),
            (7, 9),
            (7, 32),
            (7, 37),
            (7, 44),
        ],
        "internal_validation": [
            (8, 3),
            (8, 16),
            (8, 19),
            (8, 48),
            (8, 50),
            (9, 0),
            (9, 2),
            (9, 17),
            (9, 33),
        ],
        "primary_confirmation": [
            (10, 8),
            (10, 21),
            (10, 28),
            (10, 41),
            (10, 53),
            (10, 56),
            (11, 6),
            (11, 20),
            (11, 26),
            (11, 40),
            (11, 46),
            (11, 54),
        ],
    }
    axes = {role: _role_axis(candidate, role) for role in expected}
    assert axes == expected
    donor_sets = [{donor for _, donor in axis} for axis in axes.values()]
    assert [len(axis) for axis in axes.values()] == [39, 9, 12]
    assert all(
        left.isdisjoint(right)
        for left in donor_sets
        for right in donor_sets
        if left is not right
    )
    assert set.union(*donor_sets) == set(range(64)) - {23, 51, 52, 62}
    assert (9, 0) in axes["internal_validation"]


def test_control_selector_and_whole_file_boundary_are_explicit() -> None:
    candidate = _load(CANDIDATE_PATH)
    metadata = candidate["metadata_contract"]
    allocation = candidate["allocation"]
    assert metadata["control_selector"] == {
        "column": "cond",
        "value": "C",
        "exact_equality": True,
    }
    assert metadata["documented_mispool_anomaly"]["value"] == "0"
    assert metadata["documented_mispool_anomaly"]["metadata_cells"] == 455
    assert metadata["cell_budget_per_retained_donor"] == 128
    assert allocation["all_excluded_exp_ids"] == [23, 51, 52, 62]
    assert allocation["source_development"]["excluded_control_cell_counts"] == {
        "23": 2,
        "62": 64,
    }
    assert allocation["internal_validation"]["excluded_control_cell_counts"] == {
        "51": 70,
        "52": 91,
    }
    assert allocation["donor_disjoint"] is True
    assert allocation["physical_batch_disjoint"] is True
    assert allocation["whole_file_disjoint"] is False
    assert "non-control" in allocation["whole_file_boundary"]


def test_panel_binds_exact_composite_protein_features_and_genome_modalities() -> None:
    candidate = _load(CANDIDATE_PATH)
    protocol = _load(PROTOCOL_PATH)
    expected = [
        ("CD1C", "CD1c", "CD1c|CD1C"),
        ("CD2", "CD2", "CD2|CD2"),
        ("CD4", "CD4", "CD4|CD4"),
        ("CD7", "CD7", "CD7|CD7"),
        ("CD8A", "CD8", "CD8|CD8A"),
        ("ITGAM", "CD11b", "CD11b|ITGAM"),
        ("ITGAX", "CD11c", "CD11c|ITGAX"),
        ("CD14", "CD14", "CD14|CD14"),
        ("MS4A1", "CD20", "CD20|MS4A1"),
        ("CD27", "CD27", "CD27|CD27"),
        ("CD33", "CD33", "CD33|CD33"),
        ("CD34", "CD34", "CD34|CD34"),
        ("CD38", "CD38", "CD38|CD38"),
        ("CD69", "CD69", "CD69|CD69"),
        ("CD80", "CD80", "CD80|CD80"),
        ("CD86", "CD86", "CD86|CD86"),
        ("CD163", "CD163", "CD163|CD163"),
    ]
    observed = [
        (item["rna"], item["protein_target"], item["protein_exact_feature"])
        for item in candidate["panel"]["ordered_cognates"]
    ]
    assert observed == expected
    assert (
        protocol["panel"]["ordered_cognates"] == candidate["panel"]["ordered_cognates"]
    )
    assert candidate["panel"]["ordered_rna_by_protein_coordinates"] == 289
    assert "var/genome == GRCh38" in candidate["panel"]["modality_rule"]
    assert "var/genome == BD99AbSeq" in candidate["panel"]["modality_rule"]
    assert "Do not use var/feature_types" in candidate["panel"]["modality_rule"]


def test_axis_and_input_placeholders_are_closed_numeric_gates() -> None:
    candidate = _load(CANDIDATE_PATH)
    protocol = _load(PROTOCOL_PATH)
    assert candidate["input"]["compressed_sha256"] == "PENDING_SOURCE_ACQUISITION"
    assert candidate["input"]["uncompressed_sha256"] == "PENDING_AXIS_PREFLIGHT"
    assert "CSR encoding-version 0.1.0" in candidate["input"]["numeric_matrix_schema"]
    assert (
        protocol["input_binding"]["axis_preflight_artifact"] == "PENDING_AXIS_PREFLIGHT"
    )
    assert "closed gate" in candidate["freeze_requirements"]["placeholder_rule"]
    assert (
        "No source numeric access"
        in protocol["implementation_freeze"]["placeholder_rule"]
    )
    forbidden = " ".join(protocol["axis_only_preflight"]["forbidden"])
    assert "numeric matrix" in forbidden
    assert "joint tables" in forbidden


def test_cell_states_geometry_and_training_masks_match_audited_contract() -> None:
    protocol = _load(PROTOCOL_PATH)
    states = protocol["cell_and_state_construction"]
    mask = protocol["source_only_comparison_mask"]
    assert states["cell_budget_per_donor"] == 128
    assert states["cell_selection"].count("str(") == 2
    assert "'|' + obs_index" in states["cell_selection"]
    assert (
        states["rna_state"]
        == "Raw RNA count greater than zero is state 1; zero is state 0."
    )
    assert (
        states["protein_state"]
        == "Raw protein count greater than zero is state 1; zero is state 0."
    )
    assert "4 through 124" in states["marker_validity"]
    assert (
        "pooled within-pool standard deviation"
        in states["profile_construction"]["normalization"]
    )
    assert "never replaced" in states["profile_construction"]["normalization"]
    assert mask["minimum_retained_coordinates"] == 232
    assert mask["minimum_scored_coordinates_per_donor"] == 232
    assert "ceil(D/2)" in mask["construction"]
    assert "pooled observed n11" in mask["construction"]
    assert "Never intersect masks across folds" in mask["construction"]
    assert "Only the all-39-source mask" in mask["construction"]


def test_nested_source_selection_and_source_gate_are_exact() -> None:
    protocol = _load(PROTOCOL_PATH)
    primary = protocol["primary_estimator"]
    gate = protocol["source_go_gate"]
    assert (
        "Nested leave-one-physical-pool-out"
        in protocol["allocation"]["source_development"]["cross_validation"]
    )
    assert primary["stage_a_grid"] == {
        "heterogeneity_penalty": [0.1, 1.0, 10.0],
        "ridge_penalty": [0.01, 0.1],
        "transport_multiplier": [0.5, 0.75, 1.0, 1.25, 1.5],
        "graph_penalty": [0.0],
    }
    assert primary["stage_b_grid"] == {
        "neighbors": [2, 3],
        "graph_penalty": [0.01, 0.03, 0.1, 0.3],
    }
    checks = gate["required_checks"]
    assert (
        checks[
            "relative_equal_batch_mean_loss_reduction_vs_matched_graph_zero_at_least"
        ]
        == 0.05
    )
    assert checks["minimum_improved_outer_pool_means_vs_matched_graph_zero"] == 7
    assert checks["outer_pool_count"] == 8
    assert checks["minimum_favorable_source_donors_vs_matched_graph_zero"] == 27
    assert checks["source_donor_count"] == 39
    assert (
        checks[
            "within_pool_bootstrap_upper_95_percent_endpoint_vs_matched_graph_zero_below_zero"
        ]
        is True
    )
    assert (
        checks[
            "within_pool_bootstrap_upper_95_percent_endpoint_vs_source_selected_classical_coordinate_below_zero"
        ]
        is True
    )
    assert protocol["comparators"]["topology_nulls"]["count"] == 63
    assert protocol["comparators"]["topology_nulls"]["selection_aware"] is True


def test_classical_comparators_and_held_gates_are_not_overstated() -> None:
    protocol = _load(PROTOCOL_PATH)
    comparators = protocol["comparators"]
    assert "donor-stratified" in comparators["common_effect_cmle"]["family"]
    assert "unstratified" in comparators["unstratified_pooled_poisson"]["family"]
    assert comparators["classical_residual_coordinate"]["families"] == [
        "signed Pearson",
        "signed-root Poisson deviance",
    ]
    for key, donors, favorable in [
        ("internal_validation_gate", 9, 8),
        ("primary_confirmation_gate", 12, 10),
    ]:
        gate = protocol[key]
        checks = gate["required_checks"]
        assert gate["required_valid_donors"] == donors
        assert (
            checks[
                "relative_equal_batch_mean_loss_reduction_vs_matched_graph_zero_at_least"
            ]
            == 0.05
        )
        assert checks["minimum_favorable_donors_vs_matched_graph_zero"] == favorable
        assert (
            checks["maximum_one_sided_exact_sign_test_p_vs_matched_graph_zero"] == 0.025
        )
        assert checks["both_physical_batch_means_improve_vs_matched_graph_zero"] is True
        assert (
            checks[
                "equal_batch_mean_difference_vs_each_classical_comparator_below_zero"
            ]
            is True
        )
        assert (
            checks[
                "paired_bootstrap_upper_95_percent_endpoint_vs_each_classical_comparator_below_zero"
            ]
            is True
        )
        assert (
            checks[
                "both_physical_batch_means_improve_vs_source_selected_classical_residual_coordinate"
            ]
            is True
        )
        assert checks["topology_empirical_p_at_most"] == 0.05
        assert (
            checks[
                "relative_equal_batch_mean_loss_reduction_vs_median_topology_null_at_least"
            ]
            == 0.03
        )
        assert (
            checks[
                "paired_bootstrap_upper_95_percent_endpoint_vs_destroyed_link_below_zero"
            ]
            is True
        )
        assert gate["mandatory_reported_control"] == "independence"


def test_sequential_access_is_one_shot_and_source_joint_tables_are_disclosed() -> None:
    protocol = _load(PROTOCOL_PATH)
    stages = protocol["sequential_access"]["stages"]
    assert [stage["index"] for stage in stages] == list(range(8))
    assert [stage["name"] for stage in stages] == [
        "candidate_public_freeze",
        "axis_only_preflight",
        "implementation_and_runtime_freeze",
        "source_development",
        "internal_margin_and_prediction",
        "internal_score",
        "confirmation_margin_and_prediction",
        "confirmation_score",
    ]
    assert "same-cell source joint tables" in stages[3]["numeric_x_access"]
    assert "same-cell joint tables forbidden" in stages[4]["numeric_x_access"]
    assert "one authorized construction" in stages[5]["numeric_x_access"]
    assert "same-cell joint tables forbidden" in stages[6]["numeric_x_access"]
    assert "one authorized construction" in stages[7]["numeric_x_access"]
    no_rescue = protocol["sequential_access"]["no_rescue"]
    assert "cannot be pooled" in no_rescue
