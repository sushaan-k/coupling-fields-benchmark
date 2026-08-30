from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pytest

from experiments import gse217494_heart_core as core


def _digest(*parts: str) -> bytes:
    return hashlib.sha256("|".join(parts).encode()).digest()


def _table(rna_positive: int, adt_high: int, n11: int) -> np.ndarray:
    return np.asarray(
        [
            [core.CELL_BUDGET - rna_positive - adt_high + n11, adt_high - n11],
            [rna_positive - n11, n11],
        ],
        dtype=np.int64,
    )


def _training_fixture(donors: int = 6, markers: int = 13):
    symbols = tuple(f"M{marker:02d}" for marker in range(markers))
    positives = np.empty((donors, markers), dtype=np.int64)
    for donor in range(donors):
        positives[donor] = 244 - np.arange(markers) + donor % 3
    raw_adt = np.empty((donors, core.CELL_BUDGET, markers), dtype=np.int64)
    cells = np.arange(core.CELL_BUDGET)
    for donor in range(donors):
        for marker in range(markers):
            raw_adt[donor, :, marker] = (cells + donor + marker) % 3
    rna_profiles = positives / core.CELL_BUDGET
    adt_profiles = (
        np.arange(donors, dtype=float)[:, None]
        + 0.01 * np.arange(markers, dtype=float)[None, :]
    )
    tables = np.empty((donors, markers, markers, 2, 2), dtype=np.int64)
    for donor, rna_marker in np.ndindex(donors, markers):
        rna_positive = int(positives[donor, rna_marker])
        tables[donor, rna_marker, :] = _table(
            rna_positive, core.ADT_HIGH_COUNT, rna_positive // 2
        )
    return symbols, positives, raw_adt, rna_profiles, adt_profiles, tables


def _mandatory(values: np.ndarray) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(values, dtype=float).copy()
        for name in core.MANDATORY_COMPARATORS
    }


def test_cell_selection_is_salted_deterministic_and_input_order_invariant():
    barcodes = tuple(f"BC{index:04d}" for index in range(700))
    expected = tuple(
        sorted(
            barcodes,
            key=lambda barcode: (
                _digest(core.CELL_SALT, "sample2", barcode),
                barcode,
            ),
        )[: core.CELL_BUDGET]
    )
    assert core.selected_cell_barcodes(barcodes, "sample2") == expected

    permutation = np.random.default_rng(7).permutation(len(barcodes))
    shuffled = tuple(barcodes[index] for index in permutation)
    assert core.selected_cell_barcodes(shuffled, "sample2") == expected
    assert len(set(core.selected_cell_indices(barcodes, "sample2"))) == core.CELL_BUDGET


def test_adt_high_ties_use_hash_then_barcode_and_are_row_order_invariant():
    barcodes = tuple(f"BC{index:04d}" for index in range(core.CELL_BUDGET))
    counts = np.zeros((core.CELL_BUDGET, 1), dtype=np.int64)
    states = core.adt_high_states(counts, barcodes, "sample2", ("CD14",))[:, 0]
    expected = set(
        sorted(
            barcodes,
            key=lambda barcode: (
                _digest(core.ADT_TIE_SALT, "sample2", "CD14", barcode),
                barcode,
            ),
        )[: core.ADT_HIGH_COUNT]
    )
    assert {barcode for barcode, state in zip(barcodes, states) if state} == expected

    permutation = np.random.default_rng(11).permutation(core.CELL_BUDGET)
    shuffled_states = core.adt_high_states(
        counts[permutation],
        tuple(barcodes[index] for index in permutation),
        "sample2",
        ("CD14",),
    )[:, 0]
    observed = {
        barcodes[index] for index, state in zip(permutation, shuffled_states) if state
    }
    assert observed == expected

    ranked_counts = counts.copy()
    ranked_counts[: core.ADT_HIGH_COUNT, 0] = 1
    ranked_states = core.adt_high_states(ranked_counts, barcodes, "sample2", ("CD14",))[
        :, 0
    ]
    np.testing.assert_array_equal(
        np.flatnonzero(ranked_states), np.arange(core.ADT_HIGH_COUNT)
    )


def test_destroyed_link_shifts_whole_vectors_by_exactly_256_salted_positions():
    barcodes = tuple(f"BC{index:04d}" for index in range(core.CELL_BUDGET))
    vectors = np.column_stack(
        (np.arange(core.CELL_BUDGET), 10_000 + np.arange(core.CELL_BUDGET))
    )
    destroyed = core.destroy_adt_vectors(vectors, barcodes, "sample2")
    order = sorted(
        range(core.CELL_BUDGET),
        key=lambda cell: (
            _digest(core.DESTROY_SALT, "sample2", barcodes[cell]),
            barcodes[cell],
        ),
    )
    for position, target in enumerate(order):
        source = order[(position + core.ADT_HIGH_COUNT) % core.CELL_BUDGET]
        np.testing.assert_array_equal(destroyed[target], vectors[source])
    assert {tuple(value) for value in destroyed} == {tuple(value) for value in vectors}


def test_joint_table_orientation_is_rna_state_by_adt_state():
    rna = np.zeros((core.CELL_BUDGET, 1), dtype=np.uint8)
    adt = np.zeros_like(rna)
    rna[:100] = 1
    adt[: core.ADT_HIGH_COUNT] = 1
    tables = core.joint_binary_tables(rna, adt)
    np.testing.assert_array_equal(tables[0, 0], [[256, 156], [0, 100]])
    assert tables[0, 0].sum(axis=1).tolist() == [412, 100]
    assert tables[0, 0].sum(axis=0).tolist() == [256, 256]


def test_profiles_use_positive_rna_and_all_adt_rows_in_the_denominator():
    rna = np.zeros((core.CELL_BUDGET, 2), dtype=np.int64)
    rna[:100, 0] = 2
    rna[:300, 1] = 1
    np.testing.assert_allclose(core.rna_detection_profile(rna), [100 / 512, 300 / 512])

    adt = np.tile(np.asarray([1, 1, 8]), (core.CELL_BUDGET, 1))
    np.testing.assert_allclose(
        core.adt_mean_profile(adt, (0, 2)),
        [np.log1p(1000.0), np.log1p(8000.0)],
    )


def test_fold_marker_selection_uses_only_training_arrays_and_fixed_ranking():
    fixture = _training_fixture()
    selection = core.select_fold_markers(*fixture)
    assert selection.symbols == tuple(f"M{marker:02d}" for marker in range(12))
    assert selection.eligible_count == 13
    assert all(value >= 16 for value in selection.minimum_balance)

    signature = inspect.signature(core.select_fold_markers)
    assert not {"validation", "held", "validation_tables"} & set(signature.parameters)
    symbols, positives, raw_adt, rna_profiles, adt_profiles, tables = fixture
    validation = np.zeros((1, len(symbols)), dtype=np.int64)
    assert validation[0, 0] == 0
    training_only = core.select_fold_markers(
        symbols,
        positives,
        raw_adt,
        rna_profiles,
        adt_profiles,
        tables,
    )
    assert training_only == selection


def test_fold_marker_selection_is_candidate_order_equivariant_and_refuses_low_support():
    symbols, positives, raw_adt, rna_profiles, adt_profiles, tables = (
        _training_fixture()
    )
    expected = core.select_fold_markers(
        symbols, positives, raw_adt, rna_profiles, adt_profiles, tables
    )
    permutation = np.random.default_rng(23).permutation(len(symbols))
    observed = core.select_fold_markers(
        tuple(symbols[index] for index in permutation),
        positives[:, permutation],
        raw_adt[:, :, permutation],
        rna_profiles[:, permutation],
        adt_profiles[:, permutation],
        tables[:, permutation][:, :, permutation],
    )
    assert observed.symbols == expected.symbols

    raw_adt[:, :, 8:] = 0
    with pytest.raises(ValueError, match="fewer than 9 markers"):
        core.select_fold_markers(
            symbols, positives, raw_adt, rna_profiles, adt_profiles, tables
        )


def test_fold_marker_selection_refuses_a_table_inconsistent_with_training_margins():
    symbols, positives, raw_adt, rna_profiles, adt_profiles, tables = (
        _training_fixture()
    )
    tables[0, 0, 1] = np.asarray([[512, 0], [0, 0]])
    with pytest.raises(ValueError, match="RNA margins differ"):
        core.select_fold_markers(
            symbols, positives, raw_adt, rna_profiles, adt_profiles, tables
        )
    assert not core.informative_fixed_margin_support(np.asarray([[512, 0], [0, 0]]))


def test_knn_graph_uses_symbol_ties_union_edges_and_mean_diagonal_normalization():
    base = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
    profiles = np.column_stack(
        (base, base, base, -base, np.asarray([0.0, 1.0, -1.0, 1.0, -1.0]))
    )
    symbols = ("Z", "B", "A", "Y", "X")
    graph = core.marker_knn_graph(profiles, symbols, neighbors=1)
    assert graph.neighbors[0] == ("A",)
    np.testing.assert_array_equal(graph.adjacency, graph.adjacency.T)
    np.testing.assert_allclose(graph.laplacian, graph.laplacian.T)
    np.testing.assert_allclose(graph.laplacian.sum(axis=1), 0.0, atol=1e-15)
    assert np.isclose(np.diag(graph.laplacian).mean(), 1.0)
    assert np.linalg.eigvalsh(graph.laplacian).min() >= -1e-12

    duplicated = core.marker_knn_graph(
        np.repeat(profiles, 2, axis=0), symbols, neighbors=1
    )
    np.testing.assert_array_equal(duplicated.adjacency, graph.adjacency)


def test_knn_graph_is_marker_order_equivariant():
    generator = np.random.default_rng(29)
    profiles = generator.normal(size=(8, 6))
    symbols = tuple(f"P{index}" for index in range(6))
    expected = core.marker_knn_graph(profiles, symbols)
    assert all(len(values) == core.KNN_NEIGHBORS for values in expected.neighbors)
    permutation = np.asarray([3, 0, 5, 1, 4, 2])
    observed = core.marker_knn_graph(
        profiles[:, permutation], tuple(symbols[index] for index in permutation)
    )
    original_positions = np.argsort(permutation)
    np.testing.assert_array_equal(
        observed.adjacency[np.ix_(original_positions, original_positions)],
        expected.adjacency,
    )
    np.testing.assert_allclose(
        observed.laplacian[np.ix_(original_positions, original_positions)],
        expected.laplacian,
    )


def test_product_laplacian_has_protein_fast_order_and_unit_mean_diagonal():
    base = np.asarray([[1.0, -1.0], [-1.0, 1.0]])
    product_laplacian = core.protein_fast_product_laplacian(base, base)
    assert product_laplacian.shape == (4, 4)
    assert product_laplacian[0, 1] == -0.5
    assert product_laplacian[0, 2] == -0.5
    assert product_laplacian[0, 3] == 0.0
    assert np.isclose(np.diag(product_laplacian).mean(), 1.0)


def test_secondary_module_graph_reduces_k_only_for_a_three_marker_panel():
    base = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
    profiles = np.column_stack((base, -base, np.asarray([0.0, 1.0, -1.0, 1.0, -1.0])))
    with pytest.raises(ValueError, match="smaller than marker count"):
        core.marker_knn_graph(profiles, ("A", "B", "C"))
    graph = core.module_knn_graph(profiles, ("A", "B", "C"))
    assert all(len(values) == 2 for values in graph.neighbors)
    assert np.isclose(np.diag(graph.laplacian).mean(), 1.0)


def test_one_hot_context_and_conditional_fields_have_no_reference_level():
    labels = ("NICM", "Donor", "AMI", "ICM")
    design = core.one_hot_context(labels)
    np.testing.assert_array_equal(design.sum(axis=1), np.ones(4))
    assert np.argmax(design, axis=1).tolist() == [3, 0, 1, 2]

    coefficient = np.arange(8, dtype=float).reshape(4, 2)
    observed = core.context_log_odds(coefficient, labels, transport_multiplier=0.75)
    np.testing.assert_allclose(observed, 0.75 * coefficient[[3, 0, 1, 2]])


def test_conditional_prediction_preserves_margins_and_positive_field_increases_n11():
    field = np.asarray([0.0, 2.0, -2.0])
    rows = np.asarray([[300, 212], [300, 212], [512, 0]])
    columns = np.asarray([[256, 256], [256, 256], [256, 256]])
    prediction = core.predict_conditional_tables(field, rows, columns)
    np.testing.assert_allclose(prediction.sum(axis=-1), rows)
    np.testing.assert_allclose(prediction.sum(axis=-2), columns)
    assert prediction[1, 1, 1] > prediction[0, 1, 1]
    np.testing.assert_array_equal(prediction[2], [[256, 256], [0, 0]])
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(FloatingPointError, match="finite and nonnegative"),
    ):
        core.predict_conditional_tables(
            np.asarray([np.finfo(float).max]),
            np.asarray([[300, 212]]),
            np.asarray([[256, 256]]),
        )


def test_standardized_pearson_fit_is_donor_order_invariant_and_predicts_at_margins():
    labels = tuple(value for value in core.ETIOLOGIES for _ in range(2))
    z_by_context = {"Donor": 118, "AMI": 123, "ICM": 133, "NICM": 140}
    tables = np.asarray([[_table(256, 256, z_by_context[label])] for label in labels])
    fit = core.fit_standardized_pearson(tables, labels)
    variance = 256**4 / (core.CELL_BUDGET**2 * (core.CELL_BUDGET - 1))
    expected = np.asarray(
        [(z_by_context[label] - 128) / np.sqrt(variance) for label in core.ETIOLOGIES]
    )
    np.testing.assert_allclose(fit[:, 0], expected)

    permutation = np.asarray([6, 1, 4, 3, 0, 7, 2, 5])
    np.testing.assert_allclose(
        core.fit_standardized_pearson(
            tables[permutation], tuple(labels[i] for i in permutation)
        ),
        fit,
    )
    rows = np.asarray([[300, 212]])
    columns = np.asarray([[256, 256]])
    prediction = core.predict_standardized_pearson(fit[3], rows, columns)
    np.testing.assert_allclose(prediction.sum(axis=-1), rows)
    np.testing.assert_allclose(prediction.sum(axis=-2), columns)


def test_deviance_is_zero_at_truth_positive_off_truth_and_donor_equal():
    truth = _table(256, 256, 120)
    independence = np.full((2, 2), 128.0)
    assert core.donor_deviance(truth, truth) == pytest.approx(0.0)
    assert core.donor_deviance(truth, independence) > 0.0
    observed = np.asarray([[truth], [truth]])
    predicted = np.asarray([[truth], [independence]])
    losses = core.panel_deviances(observed, predicted)
    assert losses[0] == pytest.approx(0.0)
    assert losses[1] > 0.0
    with pytest.raises(FloatingPointError, match="changed a recipient margin"):
        core.entity_deviance(truth, independence + np.asarray([[1, 0], [0, 0]]))


def test_cv_selection_uses_complete_folds_and_lexicographic_ties():
    first = core.ConditionalFieldConfig(0.3, 0.1, 0.0, 0.75)
    second = core.ConditionalFieldConfig(0.3, 0.1, 0.01, 0.75)
    incomplete = core.ConditionalFieldConfig(3.0, 1.0, 1.0, 1.25)
    selected, losses = core.select_cv_configuration(
        {
            second: np.asarray([1.0, 2.0]),
            first: np.asarray([1.5, 1.5]),
            incomplete: np.asarray([0.0, np.nan]),
        }
    )
    assert selected == first
    np.testing.assert_array_equal(losses, [1.5, 1.5])
    assert len(core.conditional_field_configurations()) == 48
    splits = core.leave_one_out_training_indices(4)
    assert all(validation not in splits[validation] for validation in range(4))


def test_classical_selection_uses_the_frozen_tie_order():
    losses = {name: np.ones(4) for name in core.CLASSICAL_COMPARATORS}
    selection = core.select_strongest_classical(losses)
    assert selection.selected == "standardized_fixed_margin_pearson"
    assert selection.ineligible == ()
    losses["exact_common_effect_conditional_field"] = np.full(4, 0.9)
    assert (
        core.select_strongest_classical(losses).selected
        == "exact_common_effect_conditional_field"
    )


def test_classical_selection_records_boundary_common_effect_as_ineligible():
    losses = {
        "standardized_fixed_margin_pearson": np.full(4, 0.9),
        "exact_common_effect_conditional_field": None,
        "fixed_margin_independence": np.ones(4),
    }
    selection = core.select_strongest_classical(losses)
    assert selection.selected == "standardized_fixed_margin_pearson"
    assert selection.ineligible == ("exact_common_effect_conditional_field",)
    assert selection.eligible == (
        "standardized_fixed_margin_pearson",
        "fixed_margin_independence",
    )


def test_stratified_bootstrap_is_frozen_and_sign_probability_excludes_zero_ties():
    labels = tuple(value for value in core.ETIOLOGIES for _ in range(2))
    differences = np.repeat(np.asarray([-0.1, -0.2, -0.3, -0.4]), 2)
    first = core.stratified_paired_bootstrap(differences, labels)
    second = core.stratified_paired_bootstrap(differences, labels)
    assert first == second
    assert first["draws"] == 20_000
    assert first["seed"] == 21_749_401
    np.testing.assert_allclose(first["interval"], [-0.25, -0.25])
    assert core.exact_one_sided_sign_probability(
        np.asarray([-1, -1, -1, 0, 1])
    ) == pytest.approx(5 / 16)


def test_source_gate_requires_all_four_controls_fold_support_and_each_etiology():
    labels = (
        *("Donor",) * 4,
        *("AMI",) * 2,
        *("ICM",) * 4,
        *("NICM",) * 4,
    )
    primary = np.full(14, 0.8)
    result = core.evaluate_source_gate(
        primary,
        _mandatory(np.ones(14)),
        labels,
        [9] * 15,
        all_reductions_and_fits_complete=True,
    )
    assert result["passes"] is True
    assert all(value["passes"] for value in result["comparisons"].values())

    refused = core.evaluate_source_gate(
        primary,
        _mandatory(np.ones(14)),
        labels,
        [9] * 14 + [8],
        all_reductions_and_fits_complete=True,
    )
    assert refused["passes"] is False

    with pytest.raises(ValueError, match="marker cap"):
        core.evaluate_source_gate(
            primary,
            _mandatory(np.ones(14)),
            labels,
            [13] * 15,
            all_reductions_and_fits_complete=True,
        )


def test_gates_reject_negative_deviance_values():
    source_labels = (
        *("Donor",) * 4,
        *("AMI",) * 2,
        *("ICM",) * 4,
        *("NICM",) * 4,
    )
    primary = np.full(14, 0.8)
    primary[0] = -0.1
    with pytest.raises(ValueError, match="frozen donor panel"):
        core.evaluate_source_gate(
            primary,
            _mandatory(np.ones(14)),
            source_labels,
            [9] * 15,
            all_reductions_and_fits_complete=True,
        )

    comparators = _mandatory(np.ones(14))
    comparators[core.MANDATORY_COMPARATORS[0]][0] = -0.1
    with pytest.raises(ValueError, match="nonnegative finite vectors"):
        core.evaluate_source_gate(
            np.full(14, 0.8),
            comparators,
            source_labels,
            [9] * 15,
            all_reductions_and_fits_complete=True,
        )


def test_held_gate_applies_all_four_criteria_and_reports_exact_uncertainty():
    labels = tuple(value for value in core.ETIOLOGIES for _ in range(2))
    primary = np.full(8, 0.8)
    passed = core.evaluate_held_gate(primary, _mandatory(np.ones(8)), labels)
    assert passed["passes"] is True
    for comparison in passed["comparisons"].values():
        assert comparison["passes"] is True
        assert comparison["favorable_hearts"] == 8
        assert comparison["exact_one_sided_sign_probability"] == pytest.approx(1 / 256)
        assert comparison["stratified_paired_bootstrap"]["interval"][1] < 0.0

    failed_primary = primary.copy()
    failed_primary[:2] = 1.1
    failed = core.evaluate_held_gate(failed_primary, _mandatory(np.ones(8)), labels)
    assert failed["passes"] is False
    assert all(
        not comparison["checks"]["every_etiology_mean_improvement_positive"]
        for comparison in failed["comparisons"].values()
    )


def test_module_helpers_use_complete_ordered_pairs_and_exact_multiple_testing():
    selected = ("KDR", "PECAM1", "CDH5", "CD14", "CSF1R", "OTHER")
    modules = core.evaluable_modules(selected)
    assert modules == {"endothelial": ("PECAM1", "CDH5", "KDR")}
    mask = core.module_pair_mask(selected, modules["endothelial"])
    assert mask.shape == (6, 6)
    assert np.count_nonzero(mask) == 9
    assert mask[0, 1] and mask[1, 0] and mask[2, 2]

    permutation = core.exact_paired_sign_permutation(-np.ones(8))
    assert permutation["assignments"] == 256
    assert permutation["one_sided_p"] == pytest.approx(1 / 256)
    np.testing.assert_allclose(
        core.benjamini_hochberg(np.asarray([0.01, 0.04, 0.03])),
        [0.03, 0.04, 0.04],
    )


def test_neighbor_ranking_uses_symbol_ties_and_joint_permutation_is_frozen():
    symbols = ("Z", "B", "A", "Y", "X")
    fields = np.asarray(
        [
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [3.0, 0.0],
                [6.0, 0.0],
            ],
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [3.0, 1.0],
                [6.0, 1.0],
            ],
        ]
    )
    neighbors = core.nearest_neighbor_indices(fields, symbols, neighbors=1)
    assert neighbors[:, 0, 0].tolist() == [2, 2]

    first = core.neighbor_overlap_permutation(
        fields, fields, symbols, neighbors=1, permutations=200, seed=91
    )
    second = core.neighbor_overlap_permutation(
        fields, fields, symbols, neighbors=1, permutations=200, seed=91
    )
    assert first == second
    assert first["mean_top_k_jaccard"] == 1.0
    assert first["permutations"] == 200
    assert first["joint_permutation"].startswith("one RNA-marker relabeling")
    assert 0.0 < first["one_sided_monte_carlo_p"] <= 1.0
