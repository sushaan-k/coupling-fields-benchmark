from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from experiments import confirm_gse326573_lung as confirmation


def _markers() -> list[dict[str, object]]:
    return [
        {
            "rna": f"GENE{index}",
            "adt_canonical": f"ADT{index}",
            "labels": [f"ADT{index}_totalSeqC", f"ADT{index}_totalSeq-C"],
        }
        for index in range(confirmation.MARKER_COUNT)
    ]


def _write_h5(path: Path, cells: int = 520) -> np.ndarray:
    marker_count = confirmation.MARKER_COUNT
    values = np.zeros((2 * marker_count, cells), dtype=np.int64)
    for marker in range(marker_count):
        values[marker] = (np.arange(cells) + marker) % 4
        values[marker_count + marker] = 10 + ((np.arange(cells) * 3 + marker) % 7)
    data: list[int] = []
    indices: list[int] = []
    indptr = [0]
    for column in range(cells):
        for row in range(2 * marker_count):
            if values[row, column] != 0:
                indices.append(row)
                data.append(int(values[row, column]))
        indptr.append(len(data))
    with h5py.File(path, "w") as handle:
        matrix = handle.create_group("matrix")
        matrix.create_dataset(
            "barcodes",
            data=np.asarray([f"cell-{index}".encode() for index in range(cells)]),
        )
        features = matrix.create_group("features")
        features.create_dataset(
            "name",
            data=np.asarray(
                [f"GENE{index}".encode() for index in range(marker_count)]
                + [f"ADT{index}_totalSeqC".encode() for index in range(marker_count)]
            ),
        )
        features.create_dataset(
            "feature_type",
            data=np.asarray(
                [b"Gene Expression"] * marker_count
                + [b"Antibody Capture"] * marker_count
            ),
        )
        matrix.create_dataset("data", data=np.asarray(data, dtype=np.int32))
        matrix.create_dataset("indices", data=np.asarray(indices, dtype=np.int32))
        matrix.create_dataset("indptr", data=np.asarray(indptr, dtype=np.int64))
        matrix.create_dataset(
            "shape", data=np.asarray([2 * marker_count, cells], dtype=np.int64)
        )
    return values


def test_frozen_candidate_has_intended_independent_split() -> None:
    designation, source, held = confirmation._designation(
        confirmation.DEFAULT_DESIGNATION
    )
    confirmation._validate_preflight(confirmation.DEFAULT_PREFLIGHT, designation)
    assert len(source) == 20
    assert len(held) == 10
    assert len({sample["biological_unit"] for sample in held}) == 9
    assert {sample["batch"] for sample in source} == set(confirmation.SOURCE_BATCHES)
    assert {sample["batch"] for sample in held} == set(confirmation.HELD_BATCHES)
    assert not (
        {sample["biological_unit"] for sample in source}
        & {sample["biological_unit"] for sample in held}
    )
    assert designation["split"]["excluded_overlap_units"] == ["CTD-ILD_2", "IPF_1"]


def test_frozen_runtime_contract_is_enforced(monkeypatch) -> None:
    confirmation._validate_runtime(confirmation.DEFAULT_RUNTIME)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(PermissionError, match="environment OMP_NUM_THREADS"):
        confirmation._validate_runtime(confirmation.DEFAULT_RUNTIME)


def test_hash_cell_selection_is_exact_and_deterministic() -> None:
    barcodes = [f"cell-{index}" for index in range(800)]
    first_indices, first_cells = confirmation._selected_cells(barcodes, "GSM")
    second_indices, second_cells = confirmation._selected_cells(barcodes, "GSM")
    assert len(first_cells) == confirmation.CELL_BUDGET
    np.testing.assert_array_equal(first_indices, second_indices)
    assert first_cells == second_cells
    assert np.all(np.diff(first_indices) > 0)


def test_sparse_reader_requests_only_rna_when_adt_is_forbidden(tmp_path: Path) -> None:
    path = tmp_path / "matrix.h5"
    dense = _write_h5(path)
    sample = {"gsm": "GSM", "cells": dense.shape[1]}
    panel = confirmation._read_panel(path, sample, _markers(), read_adt=False)
    selected, _ = confirmation._selected_cells(
        [f"cell-{index}" for index in range(dense.shape[1])], "GSM"
    )
    np.testing.assert_array_equal(
        panel["rna"], dense[: confirmation.MARKER_COUNT, selected].T
    )
    assert panel["adt"] is None


def test_midrank_states_and_destroyed_link_preserve_exact_margins() -> None:
    generator = np.random.default_rng(8)
    counts = generator.integers(
        0, 8, size=(confirmation.CELL_BUDGET, confirmation.MARKER_COUNT)
    )
    barcodes = [f"cell-{index}" for index in range(confirmation.CELL_BUDGET)]
    first = confirmation._adt_states(counts, barcodes, "GSM")
    second = confirmation._adt_states(counts, barcodes, "GSM")
    destroyed = confirmation._destroyed_adt(first, barcodes, "GSM")
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(
        first.sum(axis=0), np.full(confirmation.MARKER_COUNT, 256)
    )
    np.testing.assert_array_equal(first.sum(axis=0), destroyed.sum(axis=0))
    assert sorted(map(tuple, first.tolist())) == sorted(map(tuple, destroyed.tolist()))
    assert not np.array_equal(first, destroyed)


def test_destroyed_link_is_exactly_one_cyclic_shift_on_salted_order() -> None:
    states = np.arange(
        confirmation.CELL_BUDGET * confirmation.MARKER_COUNT, dtype=np.int64
    ).reshape(confirmation.CELL_BUDGET, confirmation.MARKER_COUNT)
    barcodes = [f"cell-{index}" for index in range(confirmation.CELL_BUDGET)]
    order = sorted(
        range(confirmation.CELL_BUDGET),
        key=lambda index: (
            confirmation.hashlib.sha256(
                f"{confirmation.DESTROYED_SALT}|GSM|{barcodes[index]}".encode()
            ).hexdigest(),
            barcodes[index],
        ),
    )
    expected = np.empty_like(states)
    expected[np.asarray(order)] = states[np.roll(np.asarray(order), 1)]
    np.testing.assert_array_equal(
        confirmation._destroyed_adt(states, barcodes, "GSM"), expected
    )


def test_binary_tables_and_predictions_keep_both_margins() -> None:
    generator = np.random.default_rng(9)
    rna = generator.integers(
        0, 2, size=(confirmation.CELL_BUDGET, confirmation.MARKER_COUNT)
    )
    adt = np.zeros_like(rna)
    adt[:256] = 1
    tables = confirmation._tables(rna, adt)
    rows, columns = confirmation._margins(tables)
    predicted = confirmation._predict_odds(
        np.full((confirmation.MARKER_COUNT, confirmation.MARKER_COUNT), 0.4),
        rows,
        columns,
        1.0,
    )
    assert tables.shape == (11, 11, 2, 2)
    np.testing.assert_allclose(predicted.sum(axis=-1), rows)
    np.testing.assert_allclose(predicted.sum(axis=-2), columns)
    assert np.isfinite(confirmation._loss(tables, predicted))


def test_zero_transport_nests_independence_for_every_residual_family() -> None:
    rows = np.broadcast_to(
        np.asarray([350, 162]),
        (confirmation.MARKER_COUNT, confirmation.MARKER_COUNT, 2),
    ).copy()
    columns = np.broadcast_to(
        np.asarray([48, 464]),
        (confirmation.MARKER_COUNT, confirmation.MARKER_COUNT, 2),
    ).copy()
    pooled = np.linspace(-2.0, 2.0, confirmation.MARKER_COUNT**2).reshape(11, 11)
    independence = confirmation._independence(rows, columns)
    for family in confirmation.RESIDUAL_FAMILIES:
        predicted = confirmation._predict_residual(
            pooled,
            rows,
            columns,
            confirmation.ResidualConfig(family, 0.0),
        )
        np.testing.assert_allclose(predicted, independence, atol=1e-8)


def test_pooled_poisson_is_the_unpenalized_saturated_interaction() -> None:
    table = np.asarray([[120, 80], [70, 242]], dtype=np.int64)
    tables = np.broadcast_to(
        table,
        (4, confirmation.MARKER_COUNT, confirmation.MARKER_COUNT, 2, 2),
    ).copy()
    fitted = confirmation._fit_pooled_poisson(tables)
    expected = np.log(120 * 242 / (80 * 70))
    np.testing.assert_allclose(fitted["population_log_odds"], expected)


def test_graph_zero_never_constructs_a_profile_graph(monkeypatch) -> None:
    monkeypatch.setattr(
        confirmation,
        "_knn_incidence",
        lambda *_args: (_ for _ in ()).throw(AssertionError("graph was constructed")),
    )
    monkeypatch.setattr(
        confirmation,
        "fit_hierarchical_conditional_log_odds",
        lambda *_args, **_kwargs: SimpleNamespace(
            population_log_odds=np.zeros((11, 11)),
            gradient_norm=0.0,
            scaled_gradient_norm=0.0,
            schur_condition_number=1.0,
            theta_curvature_condition_number=1.0,
            iterations=1,
        ),
    )
    result = confirmation._fit_primary(
        np.zeros((2, 11, 11, 2, 2), dtype=np.int64),
        np.zeros((2, 11)),
        np.zeros((2, 11)),
        confirmation.PrimaryConfig(2, 1.0, 0.1, 0.0, 1.0),
    )
    assert result["fit_certificate"][
        "rna_incidence_sha256"
    ] == confirmation._array_sha256(np.eye(11))


def test_leave_one_batch_out_selection_never_trains_on_held_batch(monkeypatch) -> None:
    batch_sizes = {"Batch1": 8, "Batch2": 8, "Batch3": 4}
    samples = []
    records = {}
    for batch, count in batch_sizes.items():
        for index in range(count):
            gsm = f"{batch}-{index}"
            samples.append({"gsm": gsm, "batch": batch})
            records[gsm] = {
                "tables": np.zeros((11, 11, 2, 2), dtype=np.int64),
                "destroyed_tables": np.zeros((11, 11, 2, 2), dtype=np.int64),
                "rna_profile": np.full(11, index + 1.0),
                "adt_profile": np.full(11, index + 2.0),
            }
    training_sizes: list[int] = []

    def fit_primary(tables, *_args):
        training_sizes.append(len(tables))
        return {"population_log_odds": np.ones((11, 11)), "fit_certificate": {}}

    monkeypatch.setattr(confirmation, "_fit_primary", fit_primary)
    monkeypatch.setattr(
        confirmation,
        "_fit_common_effect",
        lambda _tables: {"population_log_odds": np.full((11, 11), 2.0)},
    )
    monkeypatch.setattr(
        confirmation,
        "_fit_pooled_poisson",
        lambda _tables: {"population_log_odds": np.full((11, 11), 3.0)},
    )
    monkeypatch.setattr(
        confirmation, "_residual_pool", lambda _tables, _family: np.zeros((11, 11))
    )
    monkeypatch.setattr(
        confirmation,
        "_predict_odds",
        lambda odds, _rows, _columns, alpha: np.full(
            (11, 11, 2, 2), np.mean(odds) * alpha
        ),
    )
    monkeypatch.setattr(
        confirmation,
        "_predict_residual",
        lambda *_args: np.full((11, 11, 2, 2), 4.0),
    )
    monkeypatch.setattr(
        confirmation,
        "_independence",
        lambda *_args: np.full((11, 11, 2, 2), 5.0),
    )
    monkeypatch.setattr(
        confirmation,
        "_loss",
        lambda _truth, predicted: (
            5.0 if float(np.mean(predicted)) == 0.0 else float(np.mean(predicted))
        ),
    )
    selection = confirmation._select_source(records, samples)
    assert set(training_sizes) == {12, 16}
    assert training_sizes.count(12) == 36
    assert training_sizes.count(16) == 18
    assert selection["source_gate"]["passes"]
    assert (
        selection["selected_odds"]["common_effect_cmle"]["transport_multiplier"] == 0.25
    )
    assert (
        selection["selected_odds"]["pooled_saturated_poisson"]["transport_multiplier"]
        == 0.25
    )
    assert all(selection["source_gate"]["checks"].values())


def test_source_objective_weights_official_batches_equally() -> None:
    samples = [
        *[{"batch": "Batch1"} for _ in range(8)],
        *[{"batch": "Batch2"} for _ in range(8)],
        *[{"batch": "Batch3"} for _ in range(4)],
    ]
    losses = np.asarray([1.0] * 16 + [10.0] * 4)
    assert losses.mean() == pytest.approx(2.8)
    assert confirmation._equal_batch_mean(losses, samples) == pytest.approx(4.0)


def test_replicate_samples_are_averaged_before_donor_inference() -> None:
    samples = [
        {"gsm": "a", "biological_unit": "CTD-ILD_3"},
        {"gsm": "b", "biological_unit": "CTD-ILD_3"},
        {"gsm": "c", "biological_unit": "CONTROL_9"},
    ]
    units, losses = confirmation._aggregate_unit_losses(
        samples,
        {"primary": {"a": 1.0, "b": 3.0, "c": 7.0}},
    )
    assert units == ["CONTROL_9", "CTD-ILD_3"]
    np.testing.assert_allclose(losses["primary"], [7.0, 2.0])


def test_held_comparison_uses_20k_donor_bootstraps_and_exact_sign_test() -> None:
    units = [f"donor-{index}" for index in range(9)]
    batches = ["Batch4"] * 3 + ["Batch5"] * 3 + ["Batch6"] * 3
    result = confirmation._comparison(
        units,
        batches,
        np.full(9, 0.5),
        np.full(9, 1.0),
        confirmation.BOOTSTRAP_SEED,
        formal_transfer=True,
    )
    assert result["bootstrap_replicates"] == 20_000
    assert result["bootstrap_seed"] == confirmation.BOOTSTRAP_SEED
    assert result["exact_one_sided_sign_test_p"] == pytest.approx(1 / 512)
    assert result["passes"]


def test_formal_transfer_requires_eight_donors_and_every_held_batch() -> None:
    units = [f"donor-{index}" for index in range(9)]
    batches = ["Batch4"] * 3 + ["Batch5"] * 3 + ["Batch6"] * 3
    comparator = np.ones(9)
    seven_favorable = confirmation._comparison(
        units,
        batches,
        np.asarray([0.5] * 7 + [1.1, 1.1]),
        comparator,
        confirmation.BOOTSTRAP_SEED,
        formal_transfer=True,
    )
    assert not seven_favorable["checks"]["at_least_eight_of_nine_donors_favorable"]
    batch_failure = confirmation._comparison(
        units,
        batches,
        np.asarray([0.5] * 6 + [2.0, 1.5, 0.5]),
        comparator,
        confirmation.BOOTSTRAP_SEED,
        formal_transfer=True,
    )
    assert not batch_failure["checks"][
        "primary_minus_comparator_negative_in_every_held_batch"
    ]


def test_confirmation_status_distinguishes_classical_increment() -> None:
    assert (
        confirmation._confirmation_status(True, True)
        == "CONFIRMATION_PASS_WITH_CLASSICAL_INCREMENT"
    )
    assert (
        confirmation._confirmation_status(True, False)
        == "TRANSFER_PASS_WITHOUT_CLASSICAL_INCREMENT"
    )
    assert confirmation._confirmation_status(False, True) == "CONFIRMATION_FAIL"


def test_one_shot_records_terminal_failure_and_cannot_be_reused(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt.json"
    output = tmp_path / "output.json"

    def fail():
        raise RuntimeError("frozen failure")

    confirmation._claim(attempt, "source", {"x": "y"})
    result = confirmation._one_shot("source", attempt, output, {"x": "y"}, fail)
    assert result["status"] == "TERMINAL_SOURCE_REFUSAL"
    assert json.loads(attempt.read_text())["status"] == "CLAIMED_ONE_SHOT"
    with pytest.raises(FileExistsError):
        confirmation._one_shot("source", attempt, output, {"x": "y"}, fail)


def test_one_shot_requires_a_preexisting_exactly_bound_claim(tmp_path: Path) -> None:
    called = False

    def body():
        nonlocal called
        called = True
        return {}

    with pytest.raises(FileNotFoundError):
        confirmation._one_shot(
            "source",
            tmp_path / "attempt.json",
            tmp_path / "output.json",
            {"frozen": "bytes"},
            body,
        )
    assert not called
    confirmation._claim(tmp_path / "attempt.json", "source", {"frozen": "other"})
    with pytest.raises(PermissionError, match="different bytes"):
        confirmation._one_shot(
            "source",
            tmp_path / "attempt.json",
            tmp_path / "output.json",
            {"frozen": "bytes"},
            body,
        )
    assert not called


def test_prediction_refuses_before_held_reader_when_source_gate_failed(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"status": "TERMINAL_SOURCE_GATE_REFUSAL"}))
    monkeypatch.setattr(
        confirmation,
        "_designation",
        lambda _path: (
            {"strict_cognates": _markers()},
            [],
            [{"gsm": "held", "biological_unit": "held", "batch": "Batch4"}],
        ),
    )
    monkeypatch.setattr(
        confirmation, "_base_bindings", lambda *_args: ({}, {"candidate": "hash"})
    )
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("held reader should not run")

    monkeypatch.setattr(confirmation, "_read_samples", forbidden)
    with pytest.raises(PermissionError, match="source gate"):
        confirmation.run_prediction(
            archive_path=tmp_path / "archive.tar",
            designation_path=tmp_path / "candidate.json",
            preflight_path=tmp_path / "preflight.json",
            protocol_path=tmp_path / "protocol.json",
            runtime_path=tmp_path / "runtime.json",
            source_path=source,
            attempt_path=tmp_path / "attempt.json",
            output_path=tmp_path / "prediction.json",
        )
    assert not called


def test_private_rna_hash_rejects_same_margin_state_mutation(tmp_path: Path) -> None:
    generator = np.random.default_rng(19)
    counts = generator.integers(
        0, 2, size=(confirmation.CELL_BUDGET, confirmation.MARKER_COUNT)
    )
    samples = [{"gsm": "GSM"}]
    panels = {"GSM": {"rna": counts}}
    path = tmp_path / "held.npz"
    private_sha, state_hashes = confirmation._write_private_rna(path, samples, panels)
    assert private_sha == confirmation._sha256(path)
    assert path.stat().st_mode & 0o777 == 0o600
    private = confirmation._read_private_rna(path, samples)["GSM"]
    frozen = {
        "rna_state_sha256": state_hashes["GSM"],
        "selected_cell_axis_sha256": "selected",
        "barcode_axis_sha256": "all",
    }
    panel = {
        "rna": counts,
        "selected_cell_axis_sha256": "selected",
        "barcode_axis_sha256": "all",
    }
    confirmation._verify_frozen_rna(panel, frozen, private)
    mutated = private.copy()
    one = int(np.flatnonzero(mutated[:, 0] == 1)[0])
    zero = int(np.flatnonzero(mutated[:, 0] == 0)[0])
    mutated[one, 0], mutated[zero, 0] = 0, 1
    assert mutated[:, 0].sum() == private[:, 0].sum()
    panel["rna"] = mutated
    with pytest.raises(PermissionError, match="RNA state"):
        confirmation._verify_frozen_rna(panel, frozen, private)


def test_prediction_axis_rejects_duplicates_and_extras() -> None:
    held = [{"gsm": "a"}, {"gsm": "b"}]
    assert confirmation._frozen_prediction_axis(
        {"samples": [{"gsm": "a"}, {"gsm": "b"}]}, held
    ) == ["a", "b"]
    with pytest.raises(PermissionError, match="sample axis"):
        confirmation._frozen_prediction_axis(
            {"samples": [{"gsm": "a"}, {"gsm": "a"}, {"gsm": "b"}]}, held
        )


def test_score_cannot_read_held_data_before_authorization_and_claim(
    tmp_path: Path, monkeypatch
) -> None:
    base = {"frozen": "base"}
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps({"status": "SOURCE_GATE_PASS", **base}))
    source_sha = confirmation._sha256(source_path)
    private_path = tmp_path / "held.npz"
    private_path.write_bytes(b"private-state")
    private_path.chmod(0o600)
    prediction_path = tmp_path / "prediction.json"
    prediction_path.write_text(
        json.dumps(
            {
                "status": "HELD_MARGIN_ONLY_PREDICTIONS_FROZEN",
                **base,
                "source_sha256": source_sha,
                "private_rna_sha256": confirmation._sha256(private_path),
                "samples": [],
            }
        )
    )
    authorization_path = tmp_path / "authorization.json"
    attempt_path = tmp_path / "attempt.json"
    output_path = tmp_path / "score.json"
    monkeypatch.setattr(
        confirmation,
        "_designation",
        lambda _path: ({"strict_cognates": _markers()}, [], []),
    )
    monkeypatch.setattr(confirmation, "_base_bindings", lambda *_args: ({}, dict(base)))
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("held reader crossed the public boundary")

    monkeypatch.setattr(confirmation, "_read_samples", forbidden)
    arguments = {
        "archive_path": tmp_path / "archive.tar",
        "designation_path": tmp_path / "candidate.json",
        "preflight_path": tmp_path / "preflight.json",
        "protocol_path": tmp_path / "protocol.json",
        "runtime_path": tmp_path / "runtime.json",
        "source_path": source_path,
        "prediction_path": prediction_path,
        "private_rna_path": private_path,
        "authorization_path": authorization_path,
        "attempt_path": attempt_path,
        "output_path": output_path,
    }
    with pytest.raises(FileNotFoundError):
        confirmation.run_score(**arguments)
    assert not called
    authorization_bindings = {
        **base,
        "source_sha256": source_sha,
        "prediction_sha256": confirmation._sha256(prediction_path),
        "private_rna_sha256": confirmation._sha256(private_path),
    }
    authorization_path.write_text(
        json.dumps(
            {
                "schema": "gse326573-lung-score-authorization/1.0",
                "status": "SCORE_AUTHORIZED",
                "bindings": authorization_bindings,
            }
        )
    )
    with pytest.raises(FileNotFoundError):
        confirmation.run_score(**arguments)
    assert not called
