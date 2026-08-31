from __future__ import annotations

from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import gse252762_celiac_core as core


def _panel(count: int, table: np.ndarray | None = None) -> np.ndarray:
    cell = np.asarray([[90, 38], [38, 90]] if table is None else table, dtype=np.int64)
    return np.broadcast_to(cell, (count, 9, 9, 2, 2)).copy()


def _contexts(count: int) -> list[str]:
    split = count // 2
    return ["CELIAC"] * split + ["CONTROL"] * (count - split)


def _fake_fit(value: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        coefficient=np.full((2, 9, 9), value),
        converged=True,
        iterations=2,
        objective=1.0,
        scaled_gradient_norm=1e-10,
        schur_condition_number=2.0,
        donor_curvature_condition_number=2.5,
        minimum_schur_eigenvalue=0.5,
        maximum_schur_eigenvalue=1.0,
        minimum_donor_curvature=0.4,
        maximum_donor_curvature=1.0,
        support_count=np.full(81, 4),
        gradient_tolerance=1e-8,
        maximum_condition_number=1e14,
        graph_source="entity_laplacian",
        graph_nullity=1,
    )


def _held_losses() -> dict[str, np.ndarray]:
    return {
        "primary": np.full(13, 0.80),
        "donor_stratified_ridge_poisson": np.full(13, 1.00),
        "bias_reduced_context_poisson": np.full(13, 1.10),
        "context_signed_deviance": np.full(13, 1.20),
        "destroyed_links": np.full(13, 1.30),
        "independence": np.full(13, 1.40),
    }


def _pilot_losses() -> dict[str, np.ndarray]:
    return {
        "primary": np.full(7, 0.80),
        "donor_stratified_ridge_poisson": np.full(7, 1.00),
        "bias_reduced_context_poisson": np.full(7, 1.10),
        "context_signed_deviance": np.full(7, 1.20),
        "destroyed_links": np.full(7, 1.30),
        "independence": np.full(7, 1.40),
    }


def _held_conditions() -> list[str]:
    # Frozen metadata-preflight order: two ACD, four controls, then seven GFD.
    return ["ACD"] * 2 + ["CONTROL"] * 4 + ["GFD"] * 7


def test_frozen_numerical_constants_are_literal_and_ordered() -> None:
    assert core.MARKER_PAIRS == (
        ("CD3D", "CD3"),
        ("CD4", "CD4"),
        ("CD8A", "CD8"),
        ("CD27", "CD27"),
        ("CD38", "CD38"),
        ("CD44", "CD44"),
        ("CD69", "CD69"),
        ("ITGAE", "CD103"),
        ("KLRB1", "CD161"),
    )
    assert core.PRIMARY_TRANSPORT_GRID == (0.0, 0.75, 1.0)
    assert core.COMPARATOR_TRANSPORT_GRID == (0.0, 0.5, 0.75, 1.0, 1.25)
    assert core.COEFFICIENT_RIDGE_PENALTY == 0.01
    assert core.DONOR_PROFILED_RIDGE_PENALTY == 0.01
    assert core.CONFIGURATIONS == tuple(
        core.PrimaryConfig(deviation, graph, transport, 0.01)
        for deviation in (0.1, 1.0, 10.0)
        for graph in (0.0, 0.05, 0.2)
        for transport in (0.0, 0.75, 1.0)
    )
    assert core.CLASSICAL_METHODS == (
        "donor_stratified_ridge_poisson",
        "bias_reduced_context_poisson",
        "context_signed_deviance",
    )
    assert core.BENCHMARK_TIE_ORDER == (
        "independence",
        "donor_stratified_ridge_poisson",
        "bias_reduced_context_poisson",
        "context_signed_deviance",
    )
    assert core.MANDATORY_METHODS == (
        "primary",
        *core.CLASSICAL_METHODS,
        "destroyed_links",
        "independence",
    )
    assert core.DIFFERENCE_TOLERANCE == 1e-12
    assert core.BOOTSTRAPS == 20_000
    assert core.BOOTSTRAP_SEED == 25_276_201


def test_frozen_axes_and_product_graph_are_exact() -> None:
    assert core.MARKER_PAIRS[0] == ("CD3D", "CD3")
    assert core.MARKER_PAIRS[-1] == ("KLRB1", "CD161")
    assert len(core.MARKER_PAIRS) == 9
    graph = core.product_graph_laplacian()
    assert graph.shape == (81, 81)
    np.testing.assert_allclose(graph, graph.T, atol=1e-14)
    np.testing.assert_allclose(graph.sum(axis=1), 0.0, atol=1e-14)
    np.testing.assert_allclose(np.diag(graph), 1.0, atol=1e-14)
    assert np.linalg.eigvalsh(graph)[0] > -1e-12
    assert graph[0, 1] == pytest.approx(-1.0 / 16.0)
    assert graph[0, 9] == pytest.approx(-1.0 / 16.0)
    assert graph[0, 10] == 0.0
    graph[0, 0] = 99.0
    assert core.product_graph_laplacian()[0, 0] == pytest.approx(1.0)


def test_cell_selection_is_order_invariant_and_rna_margins_need_no_adt() -> None:
    barcodes = [f"CELL-{index:04d}" for index in range(300)]
    first = core.select_barcodes(barcodes, "SAMPLE")
    second = core.select_barcodes(list(reversed(barcodes)), "SAMPLE")
    assert first == second
    assert len(first) == 256

    counts = np.zeros((256, 9), dtype=np.int64)
    counts[:73, 0] = 1
    counts[20:200, 1:] = 2
    rows, columns = core.rna_margin_tables(counts)
    assert rows.shape == columns.shape == (9, 9, 2)
    np.testing.assert_array_equal(rows[0, :, 1], 73)
    np.testing.assert_array_equal(rows[1:, :, 1], 180)
    np.testing.assert_array_equal(columns, 128)


def test_adt_top_rank_and_destroyed_half_cycle_are_deterministic() -> None:
    barcodes = [f"CELL-{index:04d}" for index in range(256)]
    counts = np.zeros((256, 9), dtype=np.int64)
    counts[:, 1] = np.arange(256)
    first = core.adt_top_states(counts, barcodes, "SAMPLE")
    second = core.adt_top_states(counts, barcodes, "SAMPLE")
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first.sum(axis=0), 128)
    assert np.all(first[128:, 1] == 1)

    destroyed = core.destroy_adt_states(first, barcodes, "SAMPLE")
    np.testing.assert_array_equal(destroyed.sum(axis=0), first.sum(axis=0))
    assert sorted(map(tuple, destroyed.tolist())) == sorted(map(tuple, first.tolist()))
    assert not np.array_equal(destroyed, first)


def test_sample_tables_have_81_ordered_pairs_and_preserve_every_margin() -> None:
    barcodes = [f"CELL-{index:04d}" for index in range(256)]
    rna = np.zeros((256, 9), dtype=np.int64)
    adt = np.zeros((256, 9), dtype=np.int64)
    for marker in range(9):
        rna[(np.arange(256) + marker) % 3 == 0, marker] = 1
        adt[:, marker] = (np.arange(256) + 7 * marker) % 19
    truth, destroyed = core.sample_tables(rna, adt, barcodes, "SAMPLE")
    assert truth.shape == destroyed.shape == (9, 9, 2, 2)
    np.testing.assert_array_equal(truth.sum(axis=(-2, -1)), 256)
    np.testing.assert_array_equal(truth.sum(axis=-2), 128)
    np.testing.assert_array_equal(truth.sum(axis=-1), destroyed.sum(axis=-1))
    np.testing.assert_array_equal(truth.sum(axis=-2), destroyed.sum(axis=-2))


def test_exact_and_classical_predictions_preserve_recipient_margins() -> None:
    rows = np.full((9, 9, 2), 128, dtype=np.int64)
    columns = np.full((9, 9, 2), 128, dtype=np.int64)
    exact = core.predict_conditional_tables(np.zeros((9, 9)), rows, columns)
    np.testing.assert_allclose(exact, 64.0)
    np.testing.assert_allclose(exact.sum(axis=-1), rows)
    np.testing.assert_allclose(exact.sum(axis=-2), columns)

    residual = core._predict_residual_at_margins(np.full((9, 9), 1.0), rows, columns)
    poisson = core._predict_poisson_at_margins(
        np.full((9, 9), 0.5), rows, columns, 1.25
    )
    for prediction in (residual, poisson):
        np.testing.assert_allclose(prediction.sum(axis=-1), rows, atol=1e-8)
        np.testing.assert_allclose(prediction.sum(axis=-2), columns, atol=1e-8)


def test_smoothed_loglinear_is_finite_at_boundaries_and_profile_status_is_granular() -> (
    None
):
    boundary = _panel(4, np.asarray([[128, 0], [0, 128]]))
    labels = ["CELIAC", "CELIAC", "CONTROL", "CONTROL"]
    smoothed = core._fit_bias_reduced_context_poisson(boundary, labels)
    assert np.isfinite(smoothed).all()

    field, status = core._profiled_poisson_report(boundary, labels)
    assert field is None
    assert status["coordinate_status_counts"] == {
        "FINITE": 0,
        "BOUNDARY": 162,
        "NO_INFORMATION": 0,
    }
    assert status["coordinates"][0]["status"] == "BOUNDARY"

    no_information = _panel(4, np.asarray([[128, 128], [0, 0]]))
    field, status = core._profiled_poisson_report(no_information, labels)
    assert field is None
    assert status["coordinate_status_counts"]["NO_INFORMATION"] == 162


def test_primary_tie_break_uses_frozen_grid_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core, "_fit_primary", lambda *_args, **_kwargs: _fake_fit())
    selected, losses = core.select_primary_configuration(
        _panel(4), ["CELIAC", "CELIAC", "CONTROL", "CONTROL"]
    )
    assert selected == core.CONFIGURATIONS[0]
    assert all(np.isfinite(losses[config]).all() for config in core.CONFIGURATIONS)


def test_comparator_selection_includes_matched_ridge_and_prefers_null_ties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ridge_calls: list[int] = []
    zero_field = np.zeros((2, 9, 9), dtype=float)

    def fit_ridge(tables, _contexts):
        ridge_calls.append(len(tables))
        return zero_field.copy(), {"converged": True}

    monkeypatch.setattr(core, "_fit_donor_stratified_ridge_poisson", fit_ridge)
    monkeypatch.setattr(
        core, "_fit_bias_reduced_context_poisson", lambda *_args: zero_field.copy()
    )
    monkeypatch.setattr(
        core, "_fit_context_signed_deviance", lambda *_args: zero_field.copy()
    )
    monkeypatch.setattr(core, "sample_loss", lambda *_args: 1.0)
    selected, losses, independence = core.select_comparator_alphas(
        _panel(4), ["CELIAC", "CELIAC", "CONTROL", "CONTROL"]
    )
    assert ridge_calls == [3, 3, 3, 3]
    assert set(selected) == set(core.CLASSICAL_METHODS)
    assert set(losses) == set(core.CLASSICAL_METHODS)
    assert all(alpha == 0.0 for alpha in selected.values())
    for method in core.CLASSICAL_METHODS:
        np.testing.assert_allclose(losses[method][0.0], independence)


def test_zero_transport_is_exact_recipient_margin_independence() -> None:
    shape = (2, 9, 9)
    models = {
        "configuration": core.PrimaryConfig(0.1, 0.0, 0.0),
        "context_order": core.CONTEXTS,
        "comparator_alphas": {method: 0.0 for method in core.CLASSICAL_METHODS},
        "strongest_benchmark": "independence",
        "primary_field": np.full(shape, 2.0),
        "destroyed_field": np.full(shape, -2.0),
        "donor_stratified_ridge_poisson_field": np.full(shape, 1.5),
        "bias_reduced_context_poisson_field": np.full(shape, -1.5),
        "context_signed_deviance_field": np.full(shape, 3.0),
        "profiled_poisson_field": None,
    }
    rows = np.broadcast_to(np.asarray([80, 176]), (9, 9, 2)).copy()
    columns = np.full((9, 9, 2), 128, dtype=np.int64)
    predictions = core.predict_models_at_margins(models, rows, columns, "CELIAC")
    independence = predictions["independence"]
    for method in core.MANDATORY_METHODS:
        np.testing.assert_array_equal(predictions[method], independence)


def test_calibration_loso_alone_freezes_strongest_benchmark() -> None:
    selected = {method: 0.5 for method in core.CLASSICAL_METHODS}
    winning = "donor_stratified_ridge_poisson"
    losses = {
        method: {
            alpha: np.full(9, 0.7 if method == winning and alpha == 0.5 else 1.0)
            for alpha in core.COMPARATOR_TRANSPORT_GRID
        }
        for method in core.CLASSICAL_METHODS
    }
    assert (
        core.strongest_benchmark_from_calibration(selected, losses, np.full(9, 0.8))
        == winning
    )


@pytest.mark.parametrize(
    ("means", "expected"),
    (
        ((1.0, 1.0, 1.0, 1.0), "independence"),
        ((2.0, 1.0, 1.0, 1.0), "donor_stratified_ridge_poisson"),
        ((2.0, 2.0, 1.0, 1.0), "bias_reduced_context_poisson"),
        ((2.0, 2.0, 2.0, 1.0), "context_signed_deviance"),
    ),
)
def test_benchmark_ties_use_the_frozen_family_order(
    means: tuple[float, float, float, float], expected: str
) -> None:
    selected = {method: 0.0 for method in core.CLASSICAL_METHODS}
    by_method = dict(zip(core.BENCHMARK_TIE_ORDER, means))
    losses = {
        method: {
            alpha: np.full(9, by_method[method])
            for alpha in core.COMPARATOR_TRANSPORT_GRID
        }
        for method in core.CLASSICAL_METHODS
    }
    assert (
        core.strongest_benchmark_from_calibration(
            selected, losses, np.full(9, by_method["independence"])
        )
        == expected
    )


def test_pilot_gate_passes_only_with_all_v2_controls() -> None:
    result = core.pilot_promotion_gate(
        _pilot_losses(), "donor_stratified_ridge_poisson"
    )
    assert result["passes"] is True
    assert result["strongest_benchmark"] == "donor_stratified_ridge_poisson"
    assert set(result["comparisons"]) == {
        *core.CLASSICAL_METHODS,
        "destroyed_links",
        "independence",
    }


@pytest.mark.parametrize(
    "method", (*core.CLASSICAL_METHODS, "destroyed_links", "independence")
)
def test_pilot_gate_requires_strict_mean_dominance_over_every_control(
    method: str,
) -> None:
    losses = _pilot_losses()
    losses[method][:] = losses["primary"]
    result = core.pilot_promotion_gate(losses, "donor_stratified_ridge_poisson")
    assert result["passes"] is False
    if method in core.CLASSICAL_METHODS:
        assert result["checks"]["primary_mean_below_each_classical"] is False
    else:
        assert result["checks"][f"primary_mean_below_{method}"] is False


@pytest.mark.parametrize(
    ("comparison", "strongest", "key"),
    (
        (
            "independence",
            "bias_reduced_context_poisson",
            "at_least_five_of_seven_favorable_vs_independence",
        ),
        (
            "donor_stratified_ridge_poisson",
            "bias_reduced_context_poisson",
            "at_least_five_of_seven_favorable_vs_donor_stratified_ridge_poisson",
        ),
        (
            "bias_reduced_context_poisson",
            "bias_reduced_context_poisson",
            "at_least_five_of_seven_favorable_vs_strongest_benchmark",
        ),
        (
            "destroyed_links",
            "bias_reduced_context_poisson",
            "at_least_five_of_seven_favorable_vs_destroyed_links",
        ),
    ),
)
@pytest.mark.parametrize(("favorable", "expected"), ((4, False), (5, True)))
def test_pilot_five_of_seven_threshold_is_exact(
    comparison: str,
    strongest: str,
    key: str,
    favorable: int,
    expected: bool,
) -> None:
    losses = {method: np.full(7, 1.3) for method in core.MANDATORY_METHODS}
    losses["primary"][:] = 1.0
    losses[comparison][:] = 0.999
    losses[comparison][:favorable] = 1.2
    result = core.pilot_promotion_gate(losses, strongest)
    assert result["checks"][key] is expected


@pytest.mark.parametrize(
    ("comparison", "key"),
    (
        ("independence", "independence_relative_reduction_at_least_five_percent"),
        (
            "donor_stratified_ridge_poisson",
            "donor_stratified_ridge_poisson_relative_reduction_at_least_five_percent",
        ),
        ("destroyed_links", "destroyed_link_relative_reduction_at_least_five_percent"),
    ),
)
def test_pilot_five_percent_threshold_includes_the_endpoint(
    comparison: str, key: str
) -> None:
    losses = {method: np.full(7, 23.0) for method in core.MANDATORY_METHODS}
    losses["primary"][:] = 19.0
    losses[comparison][:] = 20.0
    assert core.pilot_promotion_gate(losses, "bias_reduced_context_poisson")["checks"][
        key
    ]
    losses["primary"][:] = 19.0001
    assert not core.pilot_promotion_gate(losses, "bias_reduced_context_poisson")[
        "checks"
    ][key]


def test_tolerance_endpoint_is_a_tie_and_next_float_is_not() -> None:
    tolerance = core.DIFFERENCE_TOLERANCE
    beyond = np.nextafter(tolerance, np.inf)
    differences = np.asarray([-tolerance, -beyond, tolerance, beyond])
    assert core._favorable_count(differences) == 1
    assert core.exact_one_sided_sign_test(differences) == {
        "favorable": 1,
        "unfavorable": 1,
        "ties": 2,
        "nonzero_pairs": 2,
        "one_sided_probability": 0.75,
    }


@pytest.mark.parametrize("stage", ("pilot", "held"))
@pytest.mark.parametrize(
    "comparison", ("independence", "donor_stratified_ridge_poisson", "destroyed_links")
)
def test_zero_comparator_mean_fails_gate_without_execution_error(
    stage: str,
    comparison: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    count = 7 if stage == "pilot" else 13
    losses = {method: np.full(count, 1.0) for method in core.MANDATORY_METHODS}
    losses["primary"][:] = 0.0
    losses[comparison][:] = 0.0
    if stage == "pilot":
        result = core.pilot_promotion_gate(losses, "bias_reduced_context_poisson")
    else:
        monkeypatch.setattr(
            core,
            "paired_bootstrap_intervals",
            lambda differences, _conditions: {
                method: (-0.2, -0.1) for method in differences
            },
        )
        result = core.held_confirmation_gate(
            losses, _held_conditions(), "bias_reduced_context_poisson"
        )
    assert result["comparisons"][comparison]["relative_reduction"] is None
    assert result["passes"] is False
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "method", (*core.CLASSICAL_METHODS, "destroyed_links", "independence")
)
def test_held_gate_requires_strict_mean_dominance_over_every_control(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    losses = _held_losses()
    losses[method][:] = losses["primary"]
    monkeypatch.setattr(
        core,
        "paired_bootstrap_intervals",
        lambda differences, _conditions: {
            comparison: (-0.2, -0.1) for comparison in differences
        },
    )
    result = core.held_confirmation_gate(
        losses, _held_conditions(), "donor_stratified_ridge_poisson"
    )
    assert result["passes"] is False
    if method in core.CLASSICAL_METHODS:
        assert result["checks"]["primary_mean_below_each_classical"] is False
    else:
        assert result["checks"][f"primary_mean_below_{method}"] is False


def test_held_gate_cannot_reselect_strongest_from_held_ranking() -> None:
    first = core.held_confirmation_gate(
        _held_losses(), _held_conditions(), "bias_reduced_context_poisson"
    )
    mutated = _held_losses()
    mutated["context_signed_deviance"][:] = 0.81
    second = core.held_confirmation_gate(
        mutated, _held_conditions(), "bias_reduced_context_poisson"
    )
    assert first["passes"] is True
    assert second["passes"] is True
    assert first["strongest_benchmark"] == second["strongest_benchmark"]
    assert (
        first["comparisons"]["bias_reduced_context_poisson"]
        == second["comparisons"]["bias_reduced_context_poisson"]
    )
    assert first["bootstrap"] == {
        "draws": 20_000,
        "seed": 25_276_201,
        "stratification": ["ACD", "GFD", "CONTROL"],
        "generator": "numpy.random.default_rng/PCG64",
        "same_resample_indices_for_every_comparator": True,
        "within_stratum_donor_order": "input/preflight order",
    }


@pytest.mark.parametrize(
    ("comparison", "favorable_key", "sign_key"),
    (
        (
            "independence",
            "at_least_ten_of_thirteen_favorable_vs_independence",
            "independence_one_sided_sign_probability_at_most_0_05",
        ),
        (
            "donor_stratified_ridge_poisson",
            "at_least_ten_of_thirteen_favorable_vs_donor_stratified_ridge_poisson",
            "donor_stratified_ridge_poisson_one_sided_sign_probability_at_most_0_05",
        ),
        (
            "bias_reduced_context_poisson",
            "at_least_ten_of_thirteen_favorable_vs_strongest_benchmark",
            "strongest_benchmark_one_sided_sign_probability_at_most_0_05",
        ),
    ),
)
@pytest.mark.parametrize(("favorable", "expected"), ((9, False), (10, True)))
def test_held_ten_of_thirteen_and_sign_thresholds_are_exact(
    comparison: str,
    favorable_key: str,
    sign_key: str,
    favorable: int,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    losses = {method: np.full(13, 1.3) for method in core.MANDATORY_METHODS}
    losses["primary"][:] = 1.0
    losses[comparison][:] = 0.999
    losses[comparison][:favorable] = 1.2
    monkeypatch.setattr(
        core,
        "paired_bootstrap_intervals",
        lambda differences, _conditions: {
            method: (-0.2, -0.1) for method in differences
        },
    )
    result = core.held_confirmation_gate(
        losses, _held_conditions(), "bias_reduced_context_poisson"
    )
    assert result["checks"][favorable_key] is expected
    assert result["checks"][sign_key] is expected


@pytest.mark.parametrize(
    ("method", "key"),
    (
        ("independence", "independence_bootstrap_upper_below_zero"),
        (
            "donor_stratified_ridge_poisson",
            "donor_stratified_ridge_poisson_bootstrap_upper_below_zero",
        ),
        (
            "bias_reduced_context_poisson",
            "strongest_benchmark_bootstrap_upper_below_zero",
        ),
        ("destroyed_links", "destroyed_link_bootstrap_upper_below_zero"),
    ),
)
def test_held_bootstrap_upper_endpoint_must_be_strictly_below_zero(
    method: str, key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def intervals(differences, _conditions):
        output = {name: (-0.2, -0.1) for name in differences}
        output[method] = (-0.2, 0.0)
        return output

    monkeypatch.setattr(core, "paired_bootstrap_intervals", intervals)
    checks = core.held_confirmation_gate(
        _held_losses(), _held_conditions(), "bias_reduced_context_poisson"
    )["checks"]
    assert checks[key] is False


def test_held_sign_probability_endpoint_is_inclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        core,
        "paired_bootstrap_intervals",
        lambda differences, _conditions: {
            method: (-0.2, -0.1) for method in differences
        },
    )
    monkeypatch.setattr(
        core,
        "exact_one_sided_sign_test",
        lambda *_args: {
            "favorable": 13,
            "unfavorable": 0,
            "ties": 0,
            "nonzero_pairs": 13,
            "one_sided_probability": 0.05,
        },
    )
    checks = core.held_confirmation_gate(
        _held_losses(), _held_conditions(), "bias_reduced_context_poisson"
    )["checks"]
    assert checks["independence_one_sided_sign_probability_at_most_0_05"]
    assert checks[
        "donor_stratified_ridge_poisson_one_sided_sign_probability_at_most_0_05"
    ]
    assert checks["strongest_benchmark_one_sided_sign_probability_at_most_0_05"]


@pytest.mark.parametrize(
    ("comparison", "key"),
    (
        ("independence", "independence_relative_reduction_at_least_five_percent"),
        (
            "donor_stratified_ridge_poisson",
            "donor_stratified_ridge_poisson_relative_reduction_at_least_five_percent",
        ),
        ("destroyed_links", "destroyed_link_relative_reduction_at_least_five_percent"),
    ),
)
def test_held_five_percent_threshold_includes_the_endpoint(
    comparison: str, key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        core,
        "paired_bootstrap_intervals",
        lambda differences, _conditions: {
            method: (-0.2, -0.1) for method in differences
        },
    )
    losses = {method: np.full(13, 23.0) for method in core.MANDATORY_METHODS}
    losses["primary"][:] = 19.0
    losses[comparison][:] = 20.0
    assert core.held_confirmation_gate(
        losses, _held_conditions(), "bias_reduced_context_poisson"
    )["checks"][key]
    losses["primary"][:] = 19.0001
    assert not core.held_confirmation_gate(
        losses, _held_conditions(), "bias_reduced_context_poisson"
    )["checks"][key]


def test_exact_sign_test_and_condition_stratified_bootstrap() -> None:
    differences = np.asarray([-1.0] * 10 + [1.0] * 3)
    sign = core.exact_one_sided_sign_test(differences)
    assert sign["favorable"] == 10
    assert sign["one_sided_probability"] == pytest.approx(0.046142578125)
    interval = core.paired_bootstrap_interval(
        np.full(13, -0.2), _held_conditions(), draws=100
    )
    assert interval == pytest.approx((-0.2, -0.2))

    with_tie = core.exact_one_sided_sign_test(
        np.asarray([-1.0] * 10 + [1.0] * 2 + [0.0])
    )
    assert with_tie == {
        "favorable": 10,
        "unfavorable": 2,
        "ties": 1,
        "nonzero_pairs": 12,
        "one_sided_probability": pytest.approx(79 / 4096),
    }


def test_bootstrap_reuses_indices_and_retains_preflight_order_within_strata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, tuple[int, int]]] = []
    quantile_inputs: list[np.ndarray] = []

    class Generator:
        def integers(self, low: int, high: int, *, size: tuple[int, int]) -> np.ndarray:
            assert low == 0
            calls.append((high, size))
            return np.vstack(
                (
                    np.zeros(size[1], dtype=np.int64),
                    np.full(size[1], high - 1, dtype=np.int64),
                )
            )

    def generator(seed):
        assert seed == 25_276_201
        return Generator()

    monkeypatch.setattr(core.np.random, "default_rng", generator)

    def quantile(values, probabilities, *, method):
        assert probabilities == (0.025, 0.975)
        assert method == "linear"
        quantile_inputs.append(np.asarray(values).copy())
        return np.asarray([np.min(values), np.max(values)])

    monkeypatch.setattr(core.np, "quantile", quantile)
    first = np.arange(13, dtype=float)
    second = 10.0 * first + 1.0
    intervals = core.paired_bootstrap_intervals(
        {"first": first, "second": second},
        _held_conditions(),
        seed=25_276_201,
        draws=2,
    )
    expected_first = np.asarray([50.0 / 13.0, 106.0 / 13.0])
    np.testing.assert_allclose(quantile_inputs[0], expected_first)
    np.testing.assert_allclose(quantile_inputs[1], 10.0 * expected_first + 1.0)
    assert calls == [(2, (2, 2)), (7, (2, 7)), (4, (2, 4))]
    assert intervals["first"] == pytest.approx(
        (expected_first.min(), expected_first.max())
    )


def test_fit_serialization_preserves_frozen_comparator_and_report_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core, "_fit_primary", lambda *_args, **_kwargs: _fake_fit(0.2))
    ridge_certificate = {
        "estimator": "donor_stratified_mean_profile_ridge_poisson",
        "converged": True,
        "status_counts": {"FINITE": 162, "NO_INFORMATION": 0},
    }
    monkeypatch.setattr(
        core,
        "_fit_donor_stratified_ridge_poisson",
        lambda *_args, **_kwargs: (
            np.full((2, 9, 9), 0.3),
            ridge_certificate,
        ),
    )
    monkeypatch.setattr(
        core,
        "_profiled_poisson_report",
        lambda *_args, **_kwargs: (
            None,
            {
                "status": "PARTIAL",
                "coordinate_status_counts": {
                    "FINITE": 1,
                    "BOUNDARY": 161,
                    "NO_INFORMATION": 0,
                },
            },
        ),
    )
    tables = _panel(4)
    models = core.fit_models(
        tables,
        _panel(4, np.asarray([[64, 64], [64, 64]])),
        ["CELIAC", "CELIAC", "CONTROL", "CONTROL"],
        core.CONFIGURATIONS[0],
        {method: 0.0 for method in core.CLASSICAL_METHODS},
        strongest_benchmark="donor_stratified_ridge_poisson",
    )
    payload = core.serialize_models(models)
    restored = core.deserialize_models(payload)
    assert restored["strongest_benchmark"] == "donor_stratified_ridge_poisson"
    np.testing.assert_allclose(restored["donor_stratified_ridge_poisson_field"], 0.3)
    assert restored["donor_stratified_ridge_poisson_certificate"] == ridge_certificate
    assert restored["profiled_poisson_status"]["status"] == "PARTIAL"
    json.dumps(payload, allow_nan=False)

    off_grid = json.loads(json.dumps(payload))
    off_grid["comparator_alphas"]["donor_stratified_ridge_poisson"] = 0.25
    with pytest.raises(ValueError, match="comparator transports"):
        core.deserialize_models(off_grid)


def test_gate_payloads_are_strict_json() -> None:
    result = core.held_confirmation_gate(
        _held_losses(), _held_conditions(), "bias_reduced_context_poisson"
    )
    json.dumps(result, allow_nan=False)


def test_held_gate_requires_the_frozen_deposited_condition_composition() -> None:
    wrong_composition = ["ACD"] * 4 + ["GFD"] * 5 + ["CONTROL"] * 4
    with pytest.raises(ValueError, match="held conditions"):
        core.held_confirmation_gate(
            _held_losses(), wrong_composition, "bias_reduced_context_poisson"
        )


def test_source_selection_uses_only_nine_calibration_donors_before_pilot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, int] = {}

    def select_primary(tables, contexts, *, expected_sample_count):
        observed["primary_selection_count"] = len(tables)
        assert len(contexts) == 9
        assert expected_sample_count == 9
        return core.CONFIGURATIONS[0], {
            config: np.full(9, 0.8) for config in core.CONFIGURATIONS
        }

    def select_comparators(tables, contexts, *, expected_sample_count):
        observed["comparator_selection_count"] = len(tables)
        assert len(contexts) == 9
        assert expected_sample_count == 9
        return (
            {method: 0.0 for method in core.CLASSICAL_METHODS},
            {
                method: {
                    alpha: np.full(9, 1.0) for alpha in core.COMPARATOR_TRANSPORT_GRID
                }
                for method in core.CLASSICAL_METHODS
            },
            np.full(9, 1.1),
        )

    monkeypatch.setattr(core, "select_primary_configuration", select_primary)
    monkeypatch.setattr(core, "select_comparator_alphas", select_comparators)
    monkeypatch.setattr(
        core,
        "strongest_benchmark_from_calibration",
        lambda _alphas, losses, independence: (
            observed.update(
                benchmark_classical_count=len(next(iter(losses.values()))[0.0]),
                benchmark_independence_count=len(independence),
            )
            or "donor_stratified_ridge_poisson"
        ),
    )
    monkeypatch.setattr(core, "fit_models", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(core, "serialize_models", lambda _models: {})
    monkeypatch.setattr(
        core,
        "panel_losses",
        lambda _models, tables, _contexts: {
            method: values[: len(tables)] for method, values in _pilot_losses().items()
        },
    )
    calibration = _panel(9)
    pilot = _panel(7)
    result = core.select_source(
        calibration,
        calibration.copy(),
        _contexts(9),
        pilot,
        pilot.copy(),
        _contexts(7),
    )
    assert observed == {
        "primary_selection_count": 9,
        "comparator_selection_count": 9,
        "benchmark_classical_count": 9,
        "benchmark_independence_count": 9,
    }
    assert result["status"] == "PROMOTED"
    assert result["strongest_benchmark"] == "donor_stratified_ridge_poisson"
    assert result["schema"] == "gse252762-celiac-source-selection/2.0"


@pytest.mark.parametrize("calibration_count", (8, 10))
def test_source_selection_rejects_any_calibration_count_other_than_nine(
    calibration_count: int,
) -> None:
    with pytest.raises(ValueError, match="nine|9|calibration"):
        core.select_source(
            _panel(calibration_count),
            _panel(calibration_count),
            _contexts(calibration_count),
            _panel(7),
            _panel(7),
            _contexts(7),
        )


def test_held_prediction_refits_promoted_estimators_on_all_sixteen_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fit(tables, destroyed, contexts, *_args, **kwargs):
        observed["source_count"] = len(tables)
        observed["destroyed_count"] = len(destroyed)
        observed["context_count"] = len(contexts)
        observed["strongest_benchmark"] = kwargs["strongest_benchmark"]
        return {"models": True}

    monkeypatch.setattr(core, "fit_models", fit)
    monkeypatch.setattr(
        core,
        "predict_models_at_margins",
        lambda *_args: {
            method: np.full((9, 9, 2, 2), 64.0) for method in core.MANDATORY_METHODS
        },
    )
    selection = {
        "status": "PROMOTED",
        "selected_configuration": asdict(core.CONFIGURATIONS[0]),
        "selected_comparator_alphas": {
            method: 1.0 for method in core.CLASSICAL_METHODS
        },
        "strongest_benchmark": "bias_reduced_context_poisson",
    }
    rows = np.full((13, 9, 9, 2), 128, dtype=np.int64)
    columns = np.full((13, 9, 9, 2), 128, dtype=np.int64)
    predictions = core.predict_from_source(
        _panel(16),
        _panel(16),
        _contexts(16),
        selection,
        rows,
        columns,
        ["CELIAC"] * 2 + ["CONTROL"] * 4 + ["CELIAC"] * 7,
    )
    assert observed == {
        "source_count": 16,
        "destroyed_count": 16,
        "context_count": 16,
        "strongest_benchmark": "bias_reduced_context_poisson",
    }
    assert set(predictions) == set(core.MANDATORY_METHODS)
    assert all(values.shape == (13, 9, 9, 2, 2) for values in predictions.values())


@pytest.mark.parametrize("source_count", (15, 17))
def test_held_prediction_rejects_any_source_count_other_than_sixteen(
    source_count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(core, "fit_models", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        core,
        "predict_models_at_margins",
        lambda *_args: {
            method: np.full((9, 9, 2, 2), 64.0) for method in core.MANDATORY_METHODS
        },
    )
    selection = {
        "status": "PROMOTED",
        "selected_configuration": asdict(core.CONFIGURATIONS[0]),
        "selected_comparator_alphas": {
            method: 0.0 for method in core.CLASSICAL_METHODS
        },
        "strongest_benchmark": "independence",
    }
    rows = np.full((13, 9, 9, 2), 128, dtype=np.int64)
    columns = np.full((13, 9, 9, 2), 128, dtype=np.int64)
    with pytest.raises(ValueError, match="sixteen|16|source"):
        core.predict_from_source(
            _panel(source_count),
            _panel(source_count),
            _contexts(source_count),
            selection,
            rows,
            columns,
            ["CELIAC"] * 2 + ["CONTROL"] * 4 + ["CELIAC"] * 7,
        )


def test_held_score_uses_the_v2_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    truth = _panel(13)
    predictions = {
        method: truth.astype(float, copy=True) for method in core.MANDATORY_METHODS
    }
    monkeypatch.setattr(
        core,
        "held_confirmation_gate",
        lambda *_args: {"passes": True},
    )
    result = core.score_held(
        truth,
        predictions,
        _held_conditions(),
        "independence",
    )
    assert result["schema"] == "gse252762-celiac-held-score/2.0"
