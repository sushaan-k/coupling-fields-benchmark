from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments import confirm_gse239452_citeseq as confirmation


def _synthetic_tables(donors: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tables = []
    rna_profiles = []
    adt_profiles = []
    for donor in range(donors):
        generator = np.random.default_rng(100 + donor)
        latent = generator.normal(size=(confirmation.CELL_BUDGET, 3))
        rna = np.column_stack(
            [
                latent[:, marker % 3]
                + generator.normal(scale=0.8, size=confirmation.CELL_BUDGET)
                > 0.0
                for marker in range(len(confirmation.MARKERS))
            ]
        ).astype(np.uint8)
        adt = np.column_stack(
            [
                latent[:, marker % 3]
                + generator.normal(scale=0.8, size=confirmation.CELL_BUDGET)
                > 0.0
                for marker in range(len(confirmation.MARKERS))
            ]
        ).astype(np.uint8)
        tables.append(confirmation._binary_tables(rna, adt))
        rna_profiles.append(rna.mean(axis=0))
        adt_profiles.append(adt.mean(axis=0))
    return np.asarray(tables), np.asarray(rna_profiles), np.asarray(adt_profiles)


def test_transport_alpha_grid_is_shared_and_includes_one_half():
    assert confirmation.ALPHA_GRID == (0.5, 0.75, 1.0, 1.25)
    assert {
        config.transport_multiplier for config in confirmation._primary_configs()
    } == set(confirmation.ALPHA_GRID)
    residual = {
        confirmation.ResidualConfig(family, alpha).transport_multiplier
        for family in confirmation.RESIDUAL_FAMILIES
        for alpha in confirmation.ALPHA_GRID
    }
    assert residual == set(confirmation.ALPHA_GRID)


def test_runner_validates_the_real_frozen_source_manifest_and_exact_split():
    manifest = confirmation._sample_manifest(confirmation.DEFAULT_SOURCE)
    assert tuple(manifest) == (
        *confirmation.CALIBRATION,
        *confirmation.PILOT,
        *confirmation.HELD,
        "100",
        "78",
    )
    assert all(
        manifest[donor]["role"] == "calibration" for donor in confirmation.CALIBRATION
    )
    assert all(manifest[donor]["role"] == "pilot" for donor in confirmation.PILOT)
    assert all(manifest[donor]["role"] == "held" for donor in confirmation.HELD)
    assert {manifest[donor]["role"] for donor in ("100", "78")} == {"excluded_metadata"}


def test_runner_refuses_a_role_swap_that_preserves_manifest_role_counts(tmp_path):
    payload = json.loads(confirmation.DEFAULT_SOURCE.read_text())
    donor47 = next(row for row in payload["samples"] if row["donor"] == "47")
    donor94 = next(row for row in payload["samples"] if row["donor"] == "94")
    donor47["role"], donor94["role"] = donor94["role"], donor47["role"]
    path = tmp_path / "source_manifest.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(PermissionError, match="frozen execution contract"):
        confirmation._sample_manifest(path)


def test_common_cell_selection_is_hash_deterministic_and_exactly_512():
    common = [f"cell-{index:04d}" for index in range(700)]
    first = confirmation._selected_barcodes(common, "donor")
    second = confirmation._selected_barcodes(list(reversed(common)), "donor")
    assert first == second
    assert len(first) == confirmation.CELL_BUDGET
    assert len(set(first)) == confirmation.CELL_BUDGET


def test_adt_median_state_has_exact_fixed_margins_even_under_ties():
    counts = np.zeros((confirmation.CELL_BUDGET, len(confirmation.MARKERS)), dtype=int)
    barcodes = [f"cell-{index:04d}" for index in range(confirmation.CELL_BUDGET)]
    first = confirmation._adt_states(counts, barcodes, "donor")
    second = confirmation._adt_states(counts, barcodes, "donor")
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(
        first.sum(axis=0), np.full(len(confirmation.MARKERS), 256)
    )


def test_destroyed_link_preserves_adt_margins_and_changes_pairing():
    barcodes = [f"cell-{index:04d}" for index in range(confirmation.CELL_BUDGET)]
    states = np.tile(np.arange(confirmation.CELL_BUDGET)[:, None] % 2, (1, 9))
    destroyed = confirmation._destroyed_adt(states, barcodes, "donor")
    np.testing.assert_array_equal(destroyed.sum(axis=0), states.sum(axis=0))
    assert not np.array_equal(destroyed, states)


def test_full_log_odds_reconstruction_is_finite_and_margin_exact():
    odds = np.linspace(-2.0, 2.0, 81).reshape(9, 9)
    rows = np.empty((9, 9, 2), dtype=int)
    rows[..., 0] = np.arange(100, 109)[:, None]
    rows[..., 1] = confirmation.CELL_BUDGET - rows[..., 0]
    columns = np.broadcast_to(np.asarray([256, 256]), rows.shape).copy()
    predicted = confirmation._predict_log_odds(odds, rows, columns, 1.25)
    assert np.isfinite(predicted).all()
    np.testing.assert_allclose(predicted.sum(axis=-1), rows)
    np.testing.assert_allclose(predicted.sum(axis=-2), columns)
    assert np.all(predicted > 0.0)


def test_classical_residual_reconstruction_uses_frozen_alpha_and_margins():
    pooled = np.full((9, 9), 0.05)
    rows = np.broadcast_to(np.asarray([300, 212]), (9, 9, 2)).copy()
    columns = np.broadcast_to(np.asarray([256, 256]), (9, 9, 2)).copy()
    half = confirmation._predict_residual(
        pooled, rows, columns, confirmation.ResidualConfig("deviance", 0.5)
    )
    full = confirmation._predict_residual(
        pooled, rows, columns, confirmation.ResidualConfig("deviance", 1.0)
    )
    np.testing.assert_allclose(half.sum(axis=-1), rows)
    np.testing.assert_allclose(half.sum(axis=-2), columns)
    assert not np.allclose(half, full)


def test_hierarchical_exact_fit_returns_finite_full_log_odds_with_graph_zero():
    tables, rna_profiles, adt_profiles = _synthetic_tables()
    config = confirmation.PrimaryConfig(1, 1.0, 0.1, 0.0, 1.0)
    fitted = confirmation._fit_primary(tables, rna_profiles, adt_profiles, config)
    assert fitted["population_log_odds"].shape == (9, 9)
    assert np.isfinite(fitted["population_log_odds"]).all()
    assert fitted["scaled_gradient_norm"] <= 1e-8


def test_pilot_and_held_gate_thresholds_are_donor_paired():
    donors = tuple(f"donor-{index}" for index in range(8))
    primary = np.asarray([0.8] * 7 + [1.1])
    comparator = np.ones(8)
    generator = np.random.default_rng(confirmation.BOOTSTRAP_SEED)
    indices = generator.integers(0, 8, size=(confirmation.BOOTSTRAPS, 8))
    report = confirmation._comparison(
        donors, primary, comparator, required_favorable=7, bootstrap_indices=indices
    )
    assert report["favorable_donors"] == 7
    assert report["relative_deviance_reduction"] > 0.05
    assert report["bootstrap_unit"] == "physical donor"


def test_held_margins_are_available_without_adt_numeric_access():
    rows, columns = confirmation._held_margin_arrays([100] * 9)
    assert rows.shape == (9, 9, 2)
    np.testing.assert_array_equal(columns[..., 0], np.full((9, 9), 256))
    np.testing.assert_array_equal(columns[..., 1], np.full((9, 9), 256))


def test_csr_reader_returns_only_requested_columns_in_requested_row_order(tmp_path):
    path = tmp_path / "matrix.h5ad"
    dense = np.asarray([[0, 2, 0, 4], [5, 0, 6, 0], [0, 8, 9, 0]])
    with h5py.File(path, "w") as handle:
        raw = handle.create_group("raw")
        matrix = raw.create_group("X")
        matrix.attrs["encoding-type"] = "csr_matrix"
        matrix.attrs["shape"] = dense.shape
        indices = []
        data = []
        indptr = [0]
        for row in dense:
            nonzero = np.flatnonzero(row)
            indices.extend(nonzero.tolist())
            data.extend(row[nonzero].tolist())
            indptr.append(len(indices))
        matrix.create_dataset("indices", data=np.asarray(indices, dtype=np.int32))
        matrix.create_dataset("data", data=np.asarray(data, dtype=np.float32))
        matrix.create_dataset("indptr", data=np.asarray(indptr, dtype=np.int32))
    observed, audit = confirmation._read_csr_rows_columns(path, [2, 0], [2, 1])
    np.testing.assert_array_equal(observed, dense[[2, 0]][:, [2, 1]])
    assert audit["raw_indptr_values_read"] == 4


def test_artifact_authorization_binds_local_and_public_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(confirmation, "ROOT", tmp_path)
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n")
    authorization = tmp_path / "authorization.json"
    commit = "a" * 40
    payload = {
        "schema": "gse239452-artifact-authorization/1.0",
        "status": "HELD_GEX_ACCESS_AUTHORIZED",
        "pilot_result": confirmation._relative(artifact),
        "pilot_result_sha256": confirmation._sha256(artifact),
        "public_pilot_result_commit": commit,
    }
    authorization.write_text(json.dumps(payload) + "\n")

    def public_bytes(relative: str, observed_commit: str) -> bytes:
        assert observed_commit == commit
        if relative == confirmation._relative(artifact):
            return artifact.read_bytes()
        return authorization.read_bytes()

    monkeypatch.setattr(confirmation, "_immutable_public_bytes", public_bytes)
    confirmation._validated_artifact_authorization(
        authorization,
        commit,
        status="HELD_GEX_ACCESS_AUTHORIZED",
        artifact_path=artifact,
        artifact_field="pilot_result",
    )


def test_development_attempt_follows_authorization_and_precedes_numeric_access(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.json"
    preflight = tmp_path / "preflight.json"
    authorization = tmp_path / "authorization.json"
    attempt = tmp_path / "attempt.json"
    reduced = tmp_path / "reduced.json"
    for path in (source, preflight, authorization):
        path.write_text("{}\n")

    monkeypatch.setattr(confirmation, "ROOT", tmp_path)
    monkeypatch.setattr(confirmation, "DEFAULT_SOURCE", source)
    monkeypatch.setattr(confirmation, "DEFAULT_PREFLIGHT", preflight)
    monkeypatch.setattr(
        confirmation, "DEFAULT_DEVELOPMENT_AUTHORIZATION", authorization
    )
    monkeypatch.setattr(confirmation, "DEFAULT_DEVELOPMENT_ATTEMPT", attempt)
    monkeypatch.setattr(confirmation, "DEFAULT_REDUCED", reduced)
    monkeypatch.setattr(
        confirmation, "DEFAULT_TERMINAL_REFUSAL", tmp_path / "terminal.json"
    )
    events = []

    def authorize(*_args):
        events.append("public_authorization")
        return {"public_protocol_commit": "a" * 40}

    def manifest(*_args):
        events.append("manifest_validation")
        return {}

    def metadata_preflight(*_args):
        events.append("preflight_validation")
        return {}

    def write(path, payload, *, exclusive=True):
        del exclusive
        events.append("attempt_record" if path == attempt else "reduced_output")
        path.write_text(json.dumps(payload))

    def numeric_read(donor, *_args, **_kwargs):
        events.append(f"numeric_read:{donor}")
        return {"donor": donor}

    monkeypatch.setattr(confirmation, "_validated_protocol_authorization", authorize)
    monkeypatch.setattr(confirmation, "_sample_manifest", manifest)
    monkeypatch.setattr(confirmation, "_preflight_records", metadata_preflight)
    monkeypatch.setattr(confirmation, "_write_json", write)
    monkeypatch.setattr(confirmation, "_reduce_one", numeric_read)

    confirmation.reduce_development(
        source,
        preflight,
        authorization,
        "b" * 40,
        tmp_path,
        attempt,
        reduced,
    )
    first_numeric = next(
        index for index, value in enumerate(events) if value.startswith("numeric_read:")
    )
    assert events[:4] == [
        "public_authorization",
        "manifest_validation",
        "preflight_validation",
        "attempt_record",
    ]
    assert first_numeric > events.index("attempt_record")


def test_reduced_validator_replays_authorization_and_attempt_and_refuses_tampering(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.json"
    preflight = tmp_path / "preflight.json"
    authorization = tmp_path / "authorization.json"
    attempt = tmp_path / "attempt.json"
    reduced = tmp_path / "reduced.json"
    for path in (source, preflight, authorization):
        path.write_text("{}\n")
    monkeypatch.setattr(confirmation, "ROOT", tmp_path)
    monkeypatch.setattr(confirmation, "DEFAULT_SOURCE", source)
    monkeypatch.setattr(confirmation, "DEFAULT_PREFLIGHT", preflight)
    monkeypatch.setattr(
        confirmation, "DEFAULT_DEVELOPMENT_AUTHORIZATION", authorization
    )
    monkeypatch.setattr(confirmation, "DEFAULT_DEVELOPMENT_ATTEMPT", attempt)

    authorization_commit = "d" * 40
    protocol_commit = "e" * 40
    attempt_payload = {
        "schema": "gse239452-development-attempt/1.0",
        "status": "TERMINAL_ATTEMPT_STARTED",
        "created_at_utc": "2026-08-28T00:00:00Z",
        "authorization_sha256": confirmation._sha256(authorization),
        "public_authorization_commit": authorization_commit,
        "numeric_development_access_begins_after_this_record": True,
        "held_numeric_values_read": 0,
    }
    attempt.write_text(json.dumps(attempt_payload))
    table = np.full((9, 9, 2, 2), 128, dtype=np.int64)
    records = []
    manifest = {}
    for donor in (*confirmation.CALIBRATION, *confirmation.PILOT):
        role = "calibration" if donor in confirmation.CALIBRATION else "pilot"
        manifest[donor] = {"role": role}
        records.append(
            {
                "donor": donor,
                "role": role,
                "cells": confirmation.CELL_BUDGET,
                "tables": table.tolist(),
                "destroyed_tables": table.tolist(),
                "table_sha256": confirmation._array_sha256(table),
                "destroyed_table_sha256": confirmation._array_sha256(table),
                "adt_high_counts": [256] * 9,
                "selected_barcode_axis_sha256": "a" * 64,
            }
        )
    payload = {
        "schema": "gse239452-reduced-development/1.0",
        "status": "DEVELOPMENT_REDUCTION_COMPLETE",
        "created_at_utc": "2026-08-28T00:00:01Z",
        "source_manifest_sha256": confirmation._sha256(source),
        "metadata_preflight_sha256": confirmation._sha256(preflight),
        "development_authorization": {
            "path": confirmation._relative(authorization),
            "sha256": confirmation._sha256(authorization),
            "public_commit": authorization_commit,
            "public_protocol_commit": protocol_commit,
        },
        "development_attempt": {
            "path": confirmation._relative(attempt),
            "sha256": confirmation._sha256(attempt),
        },
        "runner_sha256": confirmation._sha256(Path(confirmation.__file__)),
        "calibration_donors": list(confirmation.CALIBRATION),
        "pilot_donors": list(confirmation.PILOT),
        "samples": records,
        "access_audit": {
            "calibration_samples_read": len(confirmation.CALIBRATION),
            "pilot_samples_read": len(confirmation.PILOT),
            "held_gex_numeric_values_read": 0,
            "held_adt_numeric_values_read": 0,
        },
    }
    monkeypatch.setattr(confirmation, "_sample_manifest", lambda *_args: manifest)
    monkeypatch.setattr(
        confirmation,
        "_validated_protocol_authorization",
        lambda *_args, **_kwargs: {"public_protocol_commit": protocol_commit},
    )
    reduced.write_text(json.dumps(payload))
    assert set(confirmation._validated_reduced(reduced)) == set(manifest)

    payload["development_authorization"]["path"] = "alternate/auth.json"
    reduced.write_text(json.dumps(payload))
    with pytest.raises(PermissionError, match="does not replay"):
        confirmation._validated_reduced(reduced)


def test_prediction_validator_recomputes_tables_and_refuses_tampering(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.json"
    preflight = tmp_path / "preflight.json"
    pilot = tmp_path / "pilot.json"
    authorization = tmp_path / "held_gex_authorization.json"
    attempt = tmp_path / "prediction_attempt.json"
    for path in (source, preflight, pilot, authorization):
        path.write_text("{}\n")
    monkeypatch.setattr(confirmation, "ROOT", tmp_path)
    monkeypatch.setattr(confirmation, "DEFAULT_SOURCE", source)
    monkeypatch.setattr(confirmation, "DEFAULT_PREFLIGHT", preflight)
    monkeypatch.setattr(confirmation, "DEFAULT_PILOT", pilot)
    monkeypatch.setattr(confirmation, "DEFAULT_HELD_GEX_AUTHORIZATION", authorization)
    monkeypatch.setattr(confirmation, "DEFAULT_PREDICTION_ATTEMPT", attempt)
    config = confirmation.PrimaryConfig(1, 1.0, 0.1, 0.0, 1.0)
    residual = confirmation.ResidualConfig("deviance", 1.0)
    zeros = np.zeros((9, 9))
    models = {
        "primary": {
            "configuration": vars(config),
            "population_log_odds": zeros.tolist(),
            "fit_certificate": {},
        },
        "destroyed_link": {
            "configuration": vars(config),
            "population_log_odds": zeros.tolist(),
            "fit_certificate": {},
        },
        "best_residual": {
            "configuration": vars(residual),
            "pooled_coordinate": zeros.tolist(),
        },
        "graph_zero_diagnostic": {
            "configuration": vars(config),
            "population_log_odds": zeros.tolist(),
            "fit_certificate": {},
        },
    }
    pilot_models = json.loads(json.dumps(models))
    rows, columns = confirmation._held_margin_arrays([100] * 9)
    predicted = confirmation._predict_panel(models, rows, columns)
    common_axis_sha256 = "b" * 64
    sample = {
        "donor": None,
        "severity": "Control",
        "cells": confirmation.CELL_BUDGET,
        "selected_barcode_axis_sha256": "a" * 64,
        "common_barcode_axis_sha256": common_axis_sha256,
        "common_barcode_count": 600,
        "rna_positive_counts": [100] * 9,
        "row_margins": rows.tolist(),
        "column_margins": columns.tolist(),
        "predicted_tables": {
            method: value.tolist() for method, value in predicted.items()
        },
        "gex_access": {
            "raw_data_values_decoded": 10,
            "raw_indices_values_decoded": 10,
            "raw_indptr_values_read": 1024,
        },
        "adt_numeric_values_read": 0,
    }
    samples = []
    for donor in confirmation.HELD:
        row = dict(sample)
        row["donor"] = donor
        samples.append(row)
    commit = "c" * 40
    attempt_payload = {
        "schema": "gse239452-prediction-attempt/1.0",
        "status": "TERMINAL_ATTEMPT_STARTED",
        "created_at_utc": "2026-08-28T00:00:00Z",
        "pilot_result_sha256": confirmation._sha256(pilot),
        "authorization_sha256": confirmation._sha256(authorization),
        "public_authorization_commit": commit,
        "held_gex_numeric_access_begins_after_this_record": True,
        "held_adt_numeric_values_read": 0,
    }
    attempt.write_text(json.dumps(attempt_payload))
    payload = {
        "schema": "gse239452-held-predictions/1.0",
        "status": "PREDICTIONS_FROZEN",
        "created_at_utc": "2026-08-28T00:00:01Z",
        "runner_sha256": confirmation._sha256(Path(confirmation.__file__)),
        "source_manifest_sha256": confirmation._sha256(source),
        "metadata_preflight_sha256": confirmation._sha256(preflight),
        "pilot_result_sha256": confirmation._sha256(pilot),
        "held_gex_authorization": {
            "path": confirmation._relative(authorization),
            "sha256": confirmation._sha256(authorization),
            "public_commit": commit,
        },
        "prediction_attempt": {
            "path": confirmation._relative(attempt),
            "sha256": confirmation._sha256(attempt),
        },
        "held_donors": list(confirmation.HELD),
        "models": models,
        "samples": samples,
        "reconstruction": "noncentral-hypergeometric expectation at frozen finite full log odds; no clipping",
        "access_audit": {
            "held_gex_samples_read": len(confirmation.HELD),
            "held_adt_barcode_axes_read": len(confirmation.HELD),
            "held_adt_files_opaque_sha256_hashed": len(confirmation.HELD),
            "held_adt_numeric_values_read": 0,
            "held_pairings_formed": 0,
            "held_truth_tables_formed": 0,
        },
    }
    manifest = {donor: {"severity": "Control"} for donor in confirmation.HELD}
    preflight_records = {
        donor: {
            "common_barcode_axis_sha256": common_axis_sha256,
            "common_barcode_count": 600,
        }
        for donor in confirmation.HELD
    }
    monkeypatch.setattr(
        confirmation,
        "_validated_pilot",
        lambda *_args, **_kwargs: {"all_development_models": pilot_models},
    )
    monkeypatch.setattr(
        confirmation, "_validated_artifact_authorization", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(confirmation, "_sample_manifest", lambda *_args: manifest)
    monkeypatch.setattr(
        confirmation, "_preflight_records", lambda *_args: preflight_records
    )
    prediction_path = tmp_path / "predictions.json"
    prediction_path.write_text(json.dumps(payload))
    confirmation._validated_prediction(prediction_path)

    payload["models"]["primary"]["population_log_odds"][0][0] = 0.5
    prediction_path.write_text(json.dumps(payload))
    with pytest.raises(PermissionError, match="models differ"):
        confirmation._validated_prediction(prediction_path)
    payload["models"] = json.loads(json.dumps(pilot_models))

    payload["undeclared_held_truth"] = [[1, 2], [3, 4]]
    prediction_path.write_text(json.dumps(payload))
    with pytest.raises(PermissionError, match="frozen runner"):
        confirmation._validated_prediction(prediction_path)
    del payload["undeclared_held_truth"]

    payload["samples"][0]["predicted_tables"]["primary"][0][0][0][0] += 1.0
    prediction_path.write_text(json.dumps(payload))
    with pytest.raises(PermissionError, match="does not recompute exactly"):
        confirmation._validated_prediction(prediction_path)


def test_terminal_artifact_blocks_all_later_phases(tmp_path, monkeypatch):
    terminal = tmp_path / "terminal.json"
    terminal.write_text("{}\n")
    monkeypatch.setattr(confirmation, "DEFAULT_TERMINAL_REFUSAL", terminal)
    with pytest.raises(PermissionError, match="permanently closes"):
        confirmation._require_open()


def test_terminal_wrapper_refuses_noncanonical_attempt_before_operation(
    tmp_path, monkeypatch
):
    canonical = tmp_path / "canonical_attempt.json"
    monkeypatch.setattr(confirmation, "DEFAULT_PREDICTION_ATTEMPT", canonical)
    called = False

    def operation():
        nonlocal called
        called = True
        return {}

    with pytest.raises(PermissionError, match="path differs"):
        confirmation._terminal_wrapper(
            "held_prediction", tmp_path / "alternate_attempt.json", operation
        )
    assert called is False
