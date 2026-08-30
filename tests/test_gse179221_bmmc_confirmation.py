from __future__ import annotations

import inspect
import hashlib
import json
from pathlib import Path
import stat

import h5py
import numpy as np
import pytest
from scipy import sparse

from experiments import confirm_gse179221_bmmc as confirmation
from mapreg.heterogeneity_adaptive_coupling import (
    expected_binary_table_from_log_odds,
    signed_pearson_coordinate,
)


def _write_tenx_h5(path: Path, cells: int = 520) -> None:
    rna_names = [rna for rna, _ in confirmation.PANEL]
    filler_names = [f"GENE{index:03d}" for index in range(200)]
    gene_names = rna_names + ["MT-ND1", *filler_names]
    adt_names = [
        "CD3",
        "CD56",
        "CD19",
        "CD14",
        "CD16",
        "CD20",
        "CD27",
        "CD38",
        "CD79b (Ig\u03b2)",
    ]
    names = [*gene_names, *adt_names, "sample-tag"]
    feature_types = [
        *(["Gene Expression"] * len(gene_names)),
        *(["Antibody Capture"] * len(adt_names)),
        "Multiplexing Capture",
    ]
    values = np.ones((len(names), cells), dtype=np.int32)
    cell_axis = np.arange(cells)
    for marker in range(confirmation.MARKER_COUNT):
        values[marker] = (cell_axis + marker) % 3
        values[len(gene_names) + marker] = (3 * cell_axis + marker) % 11
    values[-1] = cell_axis % 2
    matrix_values = sparse.csc_matrix(values)
    with h5py.File(path, "w") as handle:
        matrix = handle.create_group("matrix")
        matrix.create_dataset(
            "barcodes",
            data=np.asarray([f"cell-{index:04d}".encode() for index in range(cells)]),
        )
        features = matrix.create_group("features")
        features.create_dataset(
            "name", data=np.asarray([value.encode() for value in names])
        )
        features.create_dataset(
            "feature_type",
            data=np.asarray([value.encode() for value in feature_types]),
        )
        matrix.create_dataset("data", data=matrix_values.data)
        matrix.create_dataset("indices", data=matrix_values.indices)
        matrix.create_dataset("indptr", data=matrix_values.indptr)
        matrix.create_dataset(
            "shape", data=np.asarray(matrix_values.shape, dtype=np.int64)
        )


def _binary_tables(table: np.ndarray, donors: int = 7) -> np.ndarray:
    values = np.zeros(
        (donors, confirmation.MARKER_COUNT, confirmation.MARKER_COUNT, 2, 2),
        dtype=np.int64,
    )
    values[:] = np.asarray(table, dtype=np.int64)
    return values


def test_public_amendment_and_literal_estimator_contract_are_bound() -> None:
    amendment = confirmation._amendment()
    assert confirmation._sha256(confirmation.AMENDMENT) == confirmation.AMENDMENT_SHA256
    assert amendment["graph_contract"]["neighbors"] == 2
    assert confirmation.RESIDUAL_FAMILIES == ("pearson", "root_deviance")
    assert not hasattr(confirmation, "RESIDUAL_CENTERING")
    assert confirmation.ADT_ALIASES[-1] == ("CD79b (Ig\u03b2)", "CD79b")


def test_sparse_tenx_reduction_obeys_qc_hash_and_modality_contract(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.h5"
    _write_tenx_h5(path)
    reduced = confirmation._reduce_source_h5(path, "HD1")
    selected = reduced["selected_barcodes"]
    selected_indices = [int(value.split("-")[1]) for value in selected]
    assert len(selected) == confirmation.CELL_BUDGET
    assert selected_indices == sorted(selected_indices)
    assert reduced["rna_states"].shape == (512, 9)
    assert reduced["adt_states"].shape == (512, 9)
    np.testing.assert_array_equal(reduced["adt_states"].sum(axis=0), np.full(9, 256))
    assert reduced["access_certificate"]["rna"]["adt_numeric_entries_read"] == 0
    assert reduced["access_certificate"]["adt"]["rna_numeric_entries_read"] == 0
    assert not reduced["access_certificate"]["rna"]["full_matrix_dense_materialized"]
    assert not reduced["access_certificate"]["adt"]["full_matrix_dense_materialized"]
    np.testing.assert_array_equal(
        reduced["tables"].sum(axis=-1),
        np.broadcast_to(reduced["rna_margins"][:, None, :], (9, 9, 2)),
    )
    np.testing.assert_array_equal(
        reduced["tables"].sum(axis=-2),
        np.broadcast_to(reduced["adt_margins"][None, :, :], (9, 9, 2)),
    )


def test_panel_resolution_ignores_unrelated_feature_types_and_refuses_ambiguity() -> None:
    names = [rna for rna, _ in confirmation.PANEL] + [
        aliases[0] for aliases in confirmation.ADT_ALIASES
    ]
    types = ["Gene Expression"] * 9 + ["Antibody Capture"] * 9
    resolved = confirmation._resolve_panel(
        [*names, "hash-tag"], [*types, "Multiplexing Capture"]
    )
    assert resolved == (list(range(9)), list(range(9, 18)))
    with pytest.raises(confirmation.ProtocolRefusal, match="COGNATE_AXIS"):
        confirmation._resolve_panel(
            [*names, "CD3"], [*types, "Antibody Capture"]
        )


def test_cell_selection_and_adt_ties_are_deterministic_and_restore_deposit_order() -> None:
    barcodes = [f"cell-{index:04d}" for index in range(700)]
    selected = confirmation._selected_cells("HD1", barcodes, np.arange(700))
    expected = sorted(
        sorted(
            range(700),
            key=lambda index: (
                hashlib.sha256(
                    f"{confirmation.CELL_SALT}\0HD1\0{barcodes[index]}".encode()
                ).hexdigest(),
                barcodes[index],
            ),
        )[:512]
    )
    assert len(selected) == 512
    assert selected == expected
    counts = np.zeros((512, 9), dtype=np.int64)
    selected_barcodes = [barcodes[index] for index in selected]
    first = confirmation._midrank_adt(counts, selected_barcodes, "HD1")
    second = confirmation._midrank_adt(counts, selected_barcodes, "HD1")
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first.sum(axis=0), np.full(9, 256))
    for marker in (0, 8):
        order = sorted(
            range(512),
            key=lambda index: (
                0,
                hashlib.sha256(
                    f"{confirmation.ADT_TIE_SALT}\0HD1\0{marker}\0"
                    f"{selected_barcodes[index]}".encode()
                ).hexdigest(),
                selected_barcodes[index],
            ),
        )
        expected_states = np.zeros(512, dtype=np.uint8)
        expected_states[np.asarray(order[256:])] = 1
        np.testing.assert_array_equal(first[:, marker], expected_states)


def test_destroyed_link_is_one_complete_profile_shift_and_preserves_margins() -> None:
    generator = np.random.default_rng(19)
    states = generator.integers(0, 2, size=(512, 9), dtype=np.uint8)
    barcodes = [f"cell-{index:04d}" for index in range(512)]
    destroyed = confirmation._destroyed_adt(states, barcodes, "MGUS1")
    np.testing.assert_array_equal(states.sum(axis=0), destroyed.sum(axis=0))
    assert sorted(map(tuple, states.tolist())) == sorted(map(tuple, destroyed.tolist()))
    assert not np.array_equal(states, destroyed)
    order = sorted(
        range(512),
        key=lambda index: (
            hashlib.sha256(
                f"{confirmation.CELL_SALT}\0MGUS1\0{barcodes[index]}".encode()
            ).hexdigest(),
            barcodes[index],
        ),
    )
    for index in range(512):
        np.testing.assert_array_equal(
            destroyed[order[(index + 1) % 512]], states[order[index]]
        )


def test_fold_mask_uses_training_association_and_validation_margins_only() -> None:
    training = _binary_tables(np.asarray([[180, 70], [70, 192]]))
    first_validation = _binary_tables(np.asarray([[180, 70], [70, 192]]), donors=1)[0]
    second_validation = _binary_tables(np.asarray([[120, 130], [130, 132]]), donors=1)[0]
    first_mask, first_audit = confirmation._fold_mask(
        training, first_validation.sum(axis=-1), first_validation.sum(axis=-2)
    )
    second_mask, second_audit = confirmation._fold_mask(
        training, second_validation.sum(axis=-1), second_validation.sum(axis=-2)
    )
    np.testing.assert_array_equal(first_mask, second_mask)
    assert first_audit["training_donor_count"] == 7
    assert first_audit["recipient_paired_counts_used"] is False
    assert first_audit["scored_coordinate_count"] == 81
    assert first_audit["scored_mask_sha256"] == second_audit["scored_mask_sha256"]


def test_raw_residual_pool_matches_frozen_signed_statistic_without_centering() -> None:
    tables = _binary_tables(np.asarray([[180, 70], [70, 192]]))
    support = confirmation._margin_support(tables)
    mask = np.ones((9, 9), dtype=bool)
    pooled = confirmation._fit_residual(tables, support, mask, "pearson")
    expected = signed_pearson_coordinate(tables[0, 0, 0]) / np.sqrt(512)
    np.testing.assert_allclose(pooled, expected)


def test_true_pooled_poisson_reconstruction_has_exact_margins_and_differs_from_nch() -> None:
    rows = np.asarray([379.0, 133.0])
    columns = np.asarray([301.0, 211.0])
    log_odds = 1.35
    direct, certificate = confirmation._fixed_interaction_table(log_odds, rows, columns)
    conditional = expected_binary_table_from_log_odds(log_odds, rows, columns)
    np.testing.assert_allclose(direct.sum(axis=1), rows, atol=1e-10)
    np.testing.assert_allclose(direct.sum(axis=0), columns, atol=1e-10)
    assert certificate["absolute_log_odds_error"] <= 1e-8
    assert not np.allclose(direct, conditional, rtol=0.0, atol=1e-4)
    independence, zero_certificate = confirmation._predict_pooled_poisson(
        np.full((9, 9), log_odds),
        np.broadcast_to(rows, (9, 9, 2)),
        np.broadcast_to(columns, (9, 9, 2)),
        0.0,
        np.ones((9, 9), dtype=bool),
    )
    np.testing.assert_allclose(
        independence,
        confirmation._independence(
            np.broadcast_to(rows, (9, 9, 2)),
            np.broadcast_to(columns, (9, 9, 2)),
        ),
        atol=1e-10,
    )
    assert zero_certificate["alpha_zero_is_independence"]


def test_true_poisson_solver_handles_positive_lower_support_boundary_without_cancellation() -> None:
    source = np.asarray([[50.0, 20.0], [20.0, 50.0]])
    log_odds = confirmation._table_log_odds(source)
    rows = np.asarray([70.0, 30.0])
    columns = np.asarray([40.0, 60.0])
    table, certificate = confirmation._fixed_interaction_table(log_odds, rows, columns)
    np.testing.assert_allclose(table.sum(axis=1), rows, atol=1e-10)
    np.testing.assert_allclose(table.sum(axis=0), columns, atol=1e-10)
    assert certificate["absolute_log_odds_error"] <= 1e-8
    assert not np.allclose(
        table,
        expected_binary_table_from_log_odds(log_odds, rows, columns),
        rtol=0.0,
        atol=1e-4,
    )


def test_pooled_poisson_includes_a_conditionally_degenerate_training_table() -> None:
    tables = np.zeros((2, 9, 9, 2, 2), dtype=np.int64)
    tables[0, 0, 0] = np.asarray([[200, 312], [0, 0]])
    tables[1, 0, 0] = np.asarray([[120, 80], [90, 222]])
    mask = np.zeros((9, 9), dtype=bool)
    mask[0, 0] = True
    fitted = confirmation._fit_pooled_poisson(tables, mask)
    expected = confirmation._table_log_odds(tables[:, 0, 0].sum(axis=0))
    assert fitted["population_log_odds"][0, 0] == pytest.approx(expected)
    assert fitted["fit_certificate"][
        "pooled_every_training_donor_including_degenerate_margins"
    ]


def test_redirect_validation_rejects_ports_userinfo_queries_and_other_hosts() -> None:
    frozen = "https://ftp.ncbi.nlm.nih.gov/geo/sample.h5"
    confirmation._validate_frozen_response_url(frozen, frozen)
    for observed in (
        "https://ftp.ncbi.nlm.nih.gov:444/geo/sample.h5",
        "https://user@ftp.ncbi.nlm.nih.gov/geo/sample.h5",
        "https://ftp.ncbi.nlm.nih.gov/geo/sample.h5?token=x",
        "https://example.org/geo/sample.h5",
    ):
        with pytest.raises(PermissionError):
            confirmation._validate_frozen_response_url(frozen, observed)


def test_exclusive_consumption_record_prevents_a_second_campaign_process(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt.json"
    consumption = tmp_path / "consumption.json"
    attempt.write_text("{}", encoding="utf-8")
    first = confirmation._claim_consumption(
        consumption, "test-consumption/1.0", attempt
    )
    assert first["rerun_permitted"] is False
    with pytest.raises(FileExistsError):
        confirmation._claim_consumption(
            consumption, "test-consumption/1.0", attempt
        )


def test_failed_h5_schema_records_only_the_datasets_reached_before_refusal(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid-shape.h5"
    _write_tenx_h5(path)
    with h5py.File(path, "r+") as handle:
        shape = np.asarray(handle["matrix/shape"])
        shape[1] += 1
        handle["matrix/shape"][...] = shape
    accessed: set[str] = set()
    with pytest.raises(confirmation.ProtocolRefusal, match="AXIS_LENGTH"):
        confirmation._reduce_held_rna(path, "HD1", accessed)
    assert accessed == {
        "matrix/barcodes",
        "matrix/features/name",
        "matrix/features/feature_type",
        "matrix/shape",
    }


def test_public_source_attempt_tag_is_required_before_source_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "attempt.json"
    candidate = {
        "source_files": [
            {"gsm": "GSM1", "filename": "one.h5", "donor": "D1"}
        ]
    }
    public = {
        "implementation_commit": "a" * 40,
        "implementation_tag": confirmation.IMPLEMENTATION_TAG,
    }
    monkeypatch.setattr(confirmation, "_candidate", lambda: candidate)
    monkeypatch.setattr(confirmation, "_amendment", lambda: {})
    monkeypatch.setattr(confirmation, "_verify_public_freezes", lambda: public)
    monkeypatch.setattr(confirmation, "_binding_hashes", lambda: {"runner": "hash"})
    monkeypatch.setattr(confirmation, "_relative", lambda path: Path(path).name)
    tags: list[str] = []
    monkeypatch.setattr(
        confirmation,
        "_require_public_tag",
        lambda tag, paths: tags.append(tag) or "b" * 40,
    )
    monkeypatch.setattr(confirmation, "_require_ancestor", lambda *_: None)
    payload = {
        "schema": "gse179221-bmmc-source-attempt/1.0",
        "status": "CLAIMED_ONE_SHOT_BEFORE_ANY_H5_GET",
        "public_freezes": public,
        "candidate_sha256": confirmation._sha256(confirmation.DESIGNATION),
        "protocol_sha256": confirmation._sha256(confirmation.PROTOCOL),
        "implementation_amendment_sha256": confirmation.AMENDMENT_SHA256,
        "implementation_bindings": {"runner": "hash"},
        "runtime": confirmation._runtime_record(),
        "source_gsm_axis": ["GSM1"],
        "source_filename_axis": ["one.h5"],
        "h5_get_begins_after_this_record": True,
        "only_eight_source_urls_authorized": True,
        "held_url_get_authorized": False,
        "all_donor_tar_get_authorized": False,
        "rerun_permitted": False,
    }
    attempt.write_text(json.dumps(payload), encoding="utf-8")
    _, validated = confirmation._validate_source_attempt(attempt)
    assert tags == [confirmation.SOURCE_ATTEMPT_TAG]
    assert validated["source_attempt_commit"] == "b" * 40


def test_source_midcampaign_refusal_preserves_prior_and_failing_file_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "source-attempt.json"
    output = tmp_path / "source-result.json"
    consumption = tmp_path / "source-consumption.json"
    scratch = tmp_path / "scratch"
    attempt.write_text("{}", encoding="utf-8")
    samples = [
        {"gsm": "GSM1", "donor": "D1", "filename": "one.h5"},
        {"gsm": "GSM2", "donor": "D2", "filename": "two.h5"},
    ]
    monkeypatch.setattr(confirmation, "SOURCE_ATTEMPT", attempt)
    monkeypatch.setattr(confirmation, "SOURCE_RESULT", output)
    monkeypatch.setattr(confirmation, "SOURCE_CONSUMPTION", consumption)
    monkeypatch.setattr(
        confirmation,
        "_validate_source_attempt",
        lambda _: ({"source_files": samples}, {"source_attempt_commit": "c"}),
    )

    def fetch(_candidate, sample, folder, audit):
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / sample["filename"]
        path.write_bytes(b"h5")
        identity = {
            **sample,
            "download_status": "COMPLETE",
            "sha256": "1" * 64,
        }
        audit["source_files"].append(identity)
        audit["source_h5_get_count"] += 1
        return path, identity

    calls = 0

    def reduce(_path, _donor, _accessed=None):
        nonlocal calls
        calls += 1
        if calls == 2:
            _accessed.update(("matrix/barcodes", "matrix/features/name"))
            raise confirmation.ProtocolRefusal("SYNTHETIC_REDUCTION_FAILURE")
        return {"access_certificate": {"synthetic": True}}

    monkeypatch.setattr(confirmation, "_fetch_source_file", fetch)
    monkeypatch.setattr(confirmation, "_reduce_source_h5", reduce)
    result = confirmation.run_source(attempt, output, scratch)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "TERMINAL_SOURCE_EXECUTION_REFUSAL"
    assert len(persisted["source_files"]) == 2
    assert persisted["source_files"][0]["reduction_status"] == "COMPLETE"
    assert persisted["source_files"][1]["reduction_status"] == "REFUSED"
    assert persisted["source_files"][1]["decoded_h5_datasets"] == [
        "matrix/barcodes",
        "matrix/features/name",
    ]
    assert all(record["deleted_after_reduction"] for record in persisted["source_files"])


def test_held_source_refusal_cannot_reach_any_h5_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def network_trap(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("network must be unreachable")

    monkeypatch.setattr(
        confirmation,
        "_require_passing_source",
        lambda: (_ for _ in ()).throw(PermissionError("source refused")),
    )
    monkeypatch.setattr(confirmation.urllib.request, "build_opener", network_trap)
    with pytest.raises(PermissionError, match="source refused"):
        confirmation.claim_held_margins(tmp_path / "held-attempt.json")
    assert calls == 0


def test_held_midcampaign_refusal_is_terminal_and_deletes_private_parts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "held-attempt.json"
    output = tmp_path / "held-result.json"
    private_rna = tmp_path / "rna.npz"
    private_adt = tmp_path / "adt.npz"
    consumption = tmp_path / "held-consumption.json"
    scratch = tmp_path / "scratch"
    attempt.write_text("{}", encoding="utf-8")
    samples = [
        {"gsm": "GSM1", "donor": "D1", "stratum": "HD", "filename": "one.h5"},
        {"gsm": "GSM2", "donor": "D2", "stratum": "HD", "filename": "two.h5"},
    ]
    monkeypatch.setattr(confirmation, "HELD_ATTEMPT", attempt)
    monkeypatch.setattr(confirmation, "HELD_MARGINS", output)
    monkeypatch.setattr(confirmation, "PRIVATE_RNA", private_rna)
    monkeypatch.setattr(confirmation, "PRIVATE_ADT", private_adt)
    monkeypatch.setattr(confirmation, "HELD_CONSUMPTION", consumption)
    monkeypatch.setattr(
        confirmation,
        "_validate_held_attempt",
        lambda _: ({"held_files": samples}, {"models": {}}, "s" * 40, "a" * 40),
    )

    def fetch(_candidate, sample, folder, audit, _cohort):
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / sample["filename"]
        path.write_bytes(b"h5")
        identity = {**sample, "download_status": "COMPLETE", "sha256": "2" * 64}
        audit["held_files"].append(identity)
        audit["held_h5_get_count"] += 1
        return path, identity

    calls = 0

    def rna_reduce(_path, donor, _accessed=None):
        nonlocal calls
        calls += 1
        if calls == 2:
            _accessed.add("matrix/barcodes")
            raise confirmation.ProtocolRefusal("SYNTHETIC_HELD_FAILURE")
        barcodes = [f"{donor}-cell-{index}" for index in range(512)]
        states = np.zeros((512, 9), dtype=np.uint8)
        return {
            "selected_indices": list(range(512)),
            "selected_barcodes": barcodes,
            "states": states,
            "selected_barcode_axis_sha256": confirmation._axis_sha256(barcodes),
            "eligible_cell_count": 512,
            "profile": np.zeros(9),
            "margins": np.asarray([[512, 0]] * 9),
            "access_certificate": {"rna": True},
        }

    def adt_reduce(_path, donor, _selected, barcodes, _accessed=None):
        states = np.zeros((512, 9), dtype=np.uint8)
        states[256:] = 1
        return {
            "states": states,
            "profile": np.zeros(9),
            "margins": np.asarray([[256, 256]] * 9),
            "access_certificate": {"adt": True, "donor": donor, "cells": len(barcodes)},
        }

    monkeypatch.setattr(confirmation, "_fetch_designated_file", fetch)
    monkeypatch.setattr(confirmation, "_reduce_held_rna", rna_reduce)
    monkeypatch.setattr(confirmation, "_reduce_held_adt", adt_reduce)
    result = confirmation.run_held_margins(
        attempt, output, private_rna, private_adt, scratch
    )
    assert result["status"] == "TERMINAL_HELD_MARGIN_EXECUTION_REFUSAL"
    assert len(result["input_files"]) == 2
    assert result["input_files"][1]["reduction_status"] == "REFUSED"
    assert result["input_files"][1]["decoded_h5_datasets"] == ["matrix/barcodes"]
    assert result["private_state_artifacts_deleted"]
    assert not list(scratch.glob("*.npz"))


def test_private_state_writer_is_owner_only_and_unlinks_partial_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    complete = tmp_path / "complete.npz"
    confirmation._write_private_npz_x(complete, {"x": np.arange(4)})
    assert stat.S_IMODE(complete.stat().st_mode) == 0o600
    partial = tmp_path / "partial.npz"
    monkeypatch.setattr(
        confirmation.np,
        "savez_compressed",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("write failed")),
    )
    with pytest.raises(RuntimeError, match="write failed"):
        confirmation._write_private_npz_x(partial, {"x": np.arange(4)})
    assert not partial.exists()


def test_score_attempt_is_written_before_private_load_or_joint_construction() -> None:
    source = inspect.getsource(confirmation.score_held)
    attempt_write = source.index("_write_json_x(attempt_path, attempt)")
    assert attempt_write < source.index("_load_private_states")
    assert attempt_write < source.index("_score_predictions")
    held_margin_source = inspect.getsource(confirmation.run_held_margins)
    assert "_joint_tables(" not in held_margin_source


def test_held_score_uses_donors_for_inference_and_reports_strata_and_lodo() -> None:
    generator = np.random.default_rng(23)
    rna_states = {}
    adt_states = {}
    margin_records = []
    prediction_records = []
    for index in range(10):
        donor = f"D{index}"
        rna = generator.integers(0, 2, size=(512, 9), dtype=np.uint8)
        adt = np.zeros((512, 9), dtype=np.uint8)
        adt[256:] = 1
        rna_states[donor] = rna
        adt_states[donor] = adt
        truth = confirmation._joint_tables(rna, adt)
        rows, columns = truth.sum(axis=-1), truth.sum(axis=-2)
        independence = confirmation._independence(rows, columns)
        margins = {
            "gsm": f"GSM{index}",
            "donor": donor,
            "stratum": "HD" if index < 2 else "MGUS" if index < 6 else "WM",
            "rna_margins": rows[:, 0, :].tolist(),
            "adt_margins": columns[0, :, :].tolist(),
        }
        margin_records.append(margins)
        predictions = {
            "primary": truth.reshape(81, 4).tolist(),
            "selected_residual": independence.reshape(81, 4).tolist(),
            "common_effect_cmle": independence.reshape(81, 4).tolist(),
            "pooled_saturated_poisson": independence.reshape(81, 4).tolist(),
            "destroyed_link": independence.reshape(81, 4).tolist(),
            "independence": independence.reshape(81, 4).tolist(),
        }
        mask = np.ones((9, 9), dtype=np.uint8)
        prediction_records.append(
            {
                "gsm": margins["gsm"],
                "donor": donor,
                "stratum": margins["stratum"],
                "comparison_mask": mask.tolist(),
                "comparison_mask_sha256": confirmation._array_sha256(mask),
                "scored_coordinate_count": 81,
                "predictions": predictions,
            }
        )
    result = confirmation._score_predictions(
        {"held_records": prediction_records},
        {"held_records": margin_records},
        rna_states,
        adt_states,
    )
    assert result["status"] == "CONFIRMATION_PASS"
    assert result["passes_frozen_confirmation_gate"]
    assert set(result["disease_stratum_mean_losses"]) == {"HD", "MGUS", "WM"}
    assert len(result["leave_one_donor_out_mean_losses"]) == 10
    comparison = result["comparisons"]["selected_residual"]
    assert comparison["favorable_donors"] == 10
    assert comparison["exact_one_sided_sign_test_p"] == pytest.approx(1 / 1024)


def test_held_sign_test_requires_nine_of_ten_physical_donors() -> None:
    nine = confirmation._held_comparison(
        np.asarray([0.8] * 9 + [1.0]), np.ones(10), "residual", True
    )
    assert nine["favorable_donors"] == 9
    assert nine["exact_one_sided_sign_test_p"] == pytest.approx(11 / 1024)
    assert nine["passes_frozen_confirmation_requirement"]
    eight = confirmation._held_comparison(
        np.asarray([0.8] * 8 + [1.0] * 2), np.ones(10), "residual", True
    )
    assert eight["favorable_donors"] == 8
    assert eight["exact_one_sided_sign_test_p"] == pytest.approx(56 / 1024)
    assert not eight["passes_frozen_confirmation_requirement"]
