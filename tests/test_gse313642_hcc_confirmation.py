from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import confirm_gse313642_hcc as runner
from experiments import gse313642_hcc_core as core
from mapreg.heterogeneity_adaptive_coupling import CouplingEstimationRefusal
from mapreg.poisson_loglinear import PoissonLoglinearRefusal


def _panel(count: int) -> tuple[np.ndarray, np.ndarray]:
    real = np.empty((count, 9, 9, 2, 2), dtype=np.int64)
    destroyed = np.empty_like(real)
    for patient in range(count):
        diagonal = 160 + patient % 5
        real[patient, ...] = np.asarray(
            [[diagonal, 256 - diagonal], [256 - diagonal, diagonal]]
        )
        destroyed[patient, ...] = np.asarray([[129, 127], [127, 129]])
    return real, destroyed


@pytest.fixture(scope="module")
def serialized_models() -> tuple[dict[str, object], dict[str, object]]:
    tables, destroyed = _panel(11)
    cohorts = ["A"] * 5 + ["B"] * 6
    fitted = core.fit_models(
        tables,
        destroyed,
        cohorts,
        core.PrimaryConfig(1.0, 1.0),
        {"cohort_poisson": 0.75, "cohort_signed_deviance": 1.0},
    )
    serialized = core.serialize_models(fitted)
    return serialized, core.deserialize_models(serialized)


@pytest.fixture(scope="module")
def calibration_artifact() -> tuple[dict[str, object], dict[str, object]]:
    designation = runner._designation()
    patient_ids = designation["role_order"]["calibration"]
    by_id = {record["patient_id"]: record for record in designation["patients"]}
    cohorts = [by_id[patient_id]["group"] for patient_id in patient_ids]
    tables, destroyed = _panel(11)
    selected, primary_losses = core.select_primary_configuration(tables, cohorts)
    alphas, comparator_losses = core.select_comparator_alphas(tables, cohorts)
    models = core.serialize_models(
        core.fit_models(tables, destroyed, cohorts, selected, alphas)
    )
    audits = [
        {
            "patient_id": patient_id,
            "tables_sha256": runner._array_sha256(tables[index]),
            "destroyed_tables_sha256": runner._array_sha256(destroyed[index]),
        }
        for index, patient_id in enumerate(patient_ids)
    ]
    payload = {
        "schema": "gse313642-hcc-calibration-selection/1.0",
        "status": "FROZEN_BEFORE_ANY_PILOT_MATRIX_REQUEST",
        "rerun_permitted": False,
        "calibration_patient_order": patient_ids,
        "calibration_cohorts": dict(zip(patient_ids, cohorts)),
        "selected_configuration": runner._configuration_payload(selected),
        "primary_calibration_lopo_losses": runner._primary_loss_payload(primary_losses),
        "matched_comparator_alphas": alphas,
        "matched_comparator_calibration_lopo_losses": runner._comparator_loss_payload(
            comparator_losses
        ),
        "calibration_models": models,
        "calibration_tables": tables.tolist(),
        "calibration_destroyed_tables": destroyed.tolist(),
        "calibration_table_hashes": runner._panel_hashes(patient_ids, tables),
        "calibration_destroyed_table_hashes": runner._panel_hashes(
            patient_ids, destroyed
        ),
        "reduction_audit": audits,
        "pilot_matrix_requests": 0,
    }
    return payload, designation


def _write_axis(path: Path, lines: list[str]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        for line in lines:
            stream.write(line + "\n")


def _write_mtx(path: Path, matrix: np.ndarray) -> None:
    rows, columns = np.nonzero(matrix)
    with gzip.open(path, "wt", encoding="ascii", newline="") as stream:
        stream.write("%%MatrixMarket matrix coordinate integer general\n")
        stream.write(f"{matrix.shape[0]} {matrix.shape[1]} {len(rows)}\n")
        for row, column in zip(rows, columns):
            stream.write(f"{row + 1} {column + 1} {int(matrix[row, column])}\n")


def _prediction_artifacts(
    serialized: dict[str, object],
    models: dict[str, object],
    designation: dict[str, object],
    *,
    degenerate: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    by_id = {record["patient_id"]: record for record in designation["patients"]}
    public_rows = []
    private_rows = []
    for patient_id in designation["role_order"]["held"]:
        record = by_id[patient_id]
        barcodes = [f"{patient_id}-CELL-{index:04d}" for index in range(512)]
        rna = np.zeros((512, 9), dtype=np.uint8)
        if degenerate:
            rna[:212, 1:] = 1
        else:
            rna[:212] = 1
        marker_rows = np.stack((512 - rna.sum(axis=0), rna.sum(axis=0)), axis=1)
        rows = np.repeat(marker_rows[:, None, :], 9, axis=1)
        columns = np.broadcast_to(np.asarray((256, 256)), rows.shape).copy()
        predictions = core.predict_serialized_at_margins(
            models, rows, columns, record["group"]
        )
        public_rows.append(
            {
                "patient_id": patient_id,
                "group": record["group"],
                "row_margins": rows.tolist(),
                "column_margins": columns.tolist(),
                "predictions": {
                    method: values.tolist() for method, values in predictions.items()
                },
                "selected_barcode_axis_sha256": runner._axis_sha256(barcodes),
            }
        )
        private_rows.append(
            {
                "patient_id": patient_id,
                "deposited_patient_id": record["deposited_patient_id"],
                "group": record["group"],
                "selected_barcodes": barcodes,
                "rna_states": rna.tolist(),
            }
        )
    attempt_sha256 = "a" * 64
    return (
        {
            "schema": "gse313642-hcc-predictions/1.0",
            "status": "PREDICTIONS_FROZEN_BEFORE_ANY_HELD_FB_ACCESS",
            "rerun_permitted": False,
            "source_result_sha256": runner._sha256(runner.SOURCE_RESULT),
            "source_models_sha256": runner._json_sha256(serialized),
            "prediction_attempt_sha256": attempt_sha256,
            "selected_configuration": serialized["configuration"],
            "matched_comparator_alphas": serialized["comparator_alphas"],
            "marker_order": list(core.MARKERS),
            "coordinate_order": {
                "rows": ["RNA-negative", "RNA-positive"],
                "columns": ["FB-low", "FB-high"],
            },
            "held_gex_gets": 12,
            "held_fb_gets": 0,
            "held_fb_numeric_values_read": 0,
            "held_rna_fb_pairings_read": 0,
            "patients": public_rows,
        },
        {
            "schema": "gse313642-hcc-private-held-gex-state/1.0",
            "prediction_attempt_sha256": attempt_sha256,
            "patients": private_rows,
        },
    )


def _one_pair_axes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, object], dict[str, object]]:
    record = copy.deepcopy(runner._designation()["patients"][0])
    barcodes = [f"CELL-{index:04d}" for index in range(520)]
    gex_features = [
        f"ENSG{index:05d}\t{marker}\tGene Expression"
        for index, marker in enumerate(core.MARKERS)
    ] + ["ENSG99999\tBACKGROUND\tGene Expression"]
    fb_features = [
        f"{marker}\t{runner.FB_MARKER_DESCRIPTIONS[marker]}\tAntibody Capture"
        for marker in core.MARKERS
    ]
    for modality, axis, features in (
        ("GEX", barcodes, gex_features),
        ("FB", list(reversed(barcodes)), fb_features),
    ):
        barcode_path, feature_path = runner._axis_paths(tmp_path, record, modality)
        _write_axis(barcode_path, axis)
        _write_axis(feature_path, features)

    def manifest(
        current: dict[str, object], modality: str, member: str
    ) -> dict[str, object]:
        path = tmp_path / runner._filename(current, modality, member)
        return {
            "expected_bytes": path.stat().st_size,
            "observed_gzip_sha256": None,
        }

    monkeypatch.setattr(runner, "_manifest_file", manifest)
    monkeypatch.setattr(
        runner,
        "_url",
        lambda record, modality, member: (
            f"https://example.test/{record['patient_id']}/{modality}/{member}"
        ),
    )
    return record, runner._inspect_pair_axes(tmp_path, record)


def test_current_candidate_and_manifest_recompute_the_frozen_split() -> None:
    candidate = runner._designation()
    manifest = runner._source_manifest()
    assert len(candidate["patients"]) == 35
    assert len(manifest) == 210
    assert candidate["role_order"]["held"][:5] == [
        "A05",
        "A02",
        "A07",
        "A21",
        "A36",
    ]


def test_rank_or_hash_mutation_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    value = json.loads(runner.DESIGNATION.read_text())
    value["patients"][0]["split_sha256"] = "0" * 64
    monkeypatch.setattr(runner, "_read_json", lambda path: value)
    with pytest.raises(PermissionError, match="split rank, hash, or role"):
        runner._designation()


def test_modality_specific_feature_alias_and_type_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, inspected = _one_pair_axes(tmp_path, monkeypatch)
    assert inspected["axes"]["GEX"]["marker_rows_1based"] == list(range(1, 10))
    assert inspected["axes"]["FB"]["marker_rows_1based"] == list(range(1, 10))
    assert inspected["axes"]["GEX"]["feature_type"] == "Gene Expression"
    assert inspected["axes"]["FB"]["feature_type"] == "Antibody Capture"
    assert inspected["axes"]["FB"]["feature_set_sha256"] == runner._set_sha256(
        core.MARKERS
    )
    assert inspected["axes"]["FB"]["selected_marker_descriptions"] == (
        runner.FB_MARKER_DESCRIPTIONS
    )
    assert inspected["barcode_sets_exactly_equal"] is True
    assert inspected["barcode_orders_equal"] is False

    _, feature_path = runner._axis_paths(tmp_path, record, "FB")
    lines = gzip.open(feature_path, "rt").read().splitlines()
    lines[0] = lines[0].replace("Antibody Capture", "Gene Expression")
    _write_axis(feature_path, lines)
    with pytest.raises(runner.ProtocolRefusal, match="FEATURE_TYPE_OR_SCHEMA_MISMATCH"):
        runner._inspect_pair_axes(tmp_path, record)

    lines[0] = "CD4\tnot-the-frozen-reagent\tAntibody Capture"
    _write_axis(feature_path, lines)
    with pytest.raises(runner.ProtocolRefusal, match="FB_REAGENT_DESCRIPTION_MISMATCH"):
        runner._inspect_pair_axes(tmp_path, record)


def test_axis_gate_rejects_size_disclosed_hash_and_barcode_set_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, inspected = _one_pair_axes(tmp_path, monkeypatch)
    assert (
        inspected["axes"]["GEX"]["barcode_set_sha256"]
        == inspected["axes"]["FB"]["barcode_set_sha256"]
    )
    manifest = runner._manifest_file

    def wrong_size(
        current: dict[str, object], modality: str, member: str
    ) -> dict[str, object]:
        value = dict(manifest(current, modality, member))
        if modality == "GEX" and member == "barcodes.tsv.gz":
            value["expected_bytes"] = int(value["expected_bytes"]) + 1
        return value

    monkeypatch.setattr(runner, "_manifest_file", wrong_size)
    with pytest.raises(runner.ProtocolRefusal, match="AXIS_OFFICIAL_SIZE_MISMATCH"):
        runner._inspect_pair_axes(tmp_path, record)

    def wrong_hash(
        current: dict[str, object], modality: str, member: str
    ) -> dict[str, object]:
        value = dict(manifest(current, modality, member))
        if modality == "GEX" and member == "barcodes.tsv.gz":
            value["observed_gzip_sha256"] = "0" * 64
        return value

    monkeypatch.setattr(runner, "_manifest_file", wrong_hash)
    with pytest.raises(runner.ProtocolRefusal, match="DISCLOSED_AXIS_HASH_MISMATCH"):
        runner._inspect_pair_axes(tmp_path, record)

    monkeypatch.setattr(runner, "_manifest_file", manifest)
    fb_barcodes = runner._axis_paths(tmp_path, record, "FB")[0]
    values = gzip.open(fb_barcodes, "rt").read().splitlines()
    values[0] = "DIFFERENT-BARCODE"
    _write_axis(fb_barcodes, values)
    with pytest.raises(runner.ProtocolRefusal, match="PAIRED_BARCODE_SETS_DIFFER"):
        runner._inspect_pair_axes(tmp_path, record)


def test_axis_access_requires_exact_one_shot_url_and_size_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, _ = _one_pair_axes(tmp_path, monkeypatch)
    access = tmp_path / "axis-access.jsonl"
    monkeypatch.setattr(runner, "AXIS_ACCESS", access)
    rows = [
        {
            "schema": "gse313642-hcc-axis-access/1.0",
            "stage": "axis_acquisition",
            "event": "OPENED_BEFORE_FIRST_AXIS_GET",
            "matrix_requests": 0,
            "series_tar_used": False,
        }
    ]
    for modality in ("GEX", "FB"):
        for member in runner.AXIS_MEMBERS:
            path = tmp_path / runner._filename(record, modality, member)
            url = runner._url(record, modality, member)
            rows.extend(
                (
                    {
                        "stage": "axis_acquisition",
                        "event": "GET_STARTED",
                        "patient_id": record["patient_id"],
                        "modality": modality,
                        "member": member,
                        "url": url,
                        "expected_bytes": path.stat().st_size,
                    },
                    {
                        "stage": "axis_acquisition",
                        "event": "GET_COMPLETED",
                        "patient_id": record["patient_id"],
                        "modality": modality,
                        "member": member,
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    },
                    {
                        "stage": "axis_acquisition",
                        "event": "GZIP_PARSE_SUCCEEDED",
                        "patient_id": record["patient_id"],
                        "modality": modality,
                        "member": member,
                        "download_sha256": hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest(),
                        "line_count": len(gzip.open(path, "rt").read().splitlines()),
                    },
                )
            )
    access.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert (
        runner._validate_axis_access(tmp_path, {"patients": [record]})["axis_gets"] == 4
    )

    rows[1]["url"] = "https://wrong.test/file"
    access.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(PermissionError, match="axis access file binding differs"):
        runner._validate_axis_access(tmp_path, {"patients": [record]})
    rows[1]["url"] = runner._url(record, "GEX", runner.AXIS_MEMBERS[0])
    rows[2]["bytes"] = int(rows[2]["bytes"]) + 1
    access.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(PermissionError, match="axis access file binding differs"):
        runner._validate_axis_access(tmp_path, {"patients": [record]})


def test_matrix_reduction_joins_barcode_ids_not_positions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, preflight = _one_pair_axes(tmp_path, monkeypatch)
    gex_barcodes = (
        gzip.open(runner._axis_paths(tmp_path, record, "GEX")[0], "rt")
        .read()
        .splitlines()
    )
    fb_barcodes = list(reversed(gex_barcodes))
    gex = np.zeros((10, 520), dtype=np.int64)
    fb = np.zeros((9, 520), dtype=np.int64)
    for marker in range(9):
        for column, barcode in enumerate(gex_barcodes):
            identifier = int(barcode.split("-")[1])
            gex[marker, column] = int((identifier + marker) % 3 == 0)
        for column, barcode in enumerate(fb_barcodes):
            identifier = int(barcode.split("-")[1])
            fb[marker, column] = (identifier + 3 * marker) % 11
    gex_path = tmp_path / "gex.mtx.gz"
    fb_path = tmp_path / "fb.mtx.gz"
    _write_mtx(gex_path, gex)
    _write_mtx(fb_path, fb)
    tables, destroyed, audit = runner._reduce_patient(
        gex_path, fb_path, tmp_path, record, preflight
    )
    assert tables.shape == (9, 9, 2, 2)
    assert destroyed.shape == tables.shape
    assert np.all(tables.sum(axis=(-2, -1)) == 512)
    assert np.all(tables.sum(axis=-2)[..., 1] == 256)
    assert (
        audit["selected_barcode_axis_sha256"]
        == preflight["selected_barcode_axis_sha256"]
    )


def test_calibration_only_selects_comparator_alphas() -> None:
    tables, _ = _panel(11)
    cohorts = ["A"] * 5 + ["B"] * 6
    selected, losses = core.select_comparator_alphas(tables, cohorts)
    assert set(selected) == {"cohort_poisson", "cohort_signed_deviance"}
    assert set(selected.values()) <= {0.75, 1.0}
    assert all(
        losses[method][alpha].shape == (11,)
        for method in losses
        for alpha in (0.75, 1.0)
    )


def test_selected_alphas_survive_source_serialization(
    serialized_models: tuple[dict[str, object], dict[str, object]],
) -> None:
    serialized, restored = serialized_models
    assert serialized["comparator_alphas"] == {
        "cohort_poisson": 0.75,
        "cohort_signed_deviance": 1.0,
    }
    assert restored["comparator_alphas"] == serialized["comparator_alphas"]


def test_calibration_artifact_is_fully_replayed_before_pilot(
    calibration_artifact: tuple[dict[str, object], dict[str, object]],
) -> None:
    payload, designation = calibration_artifact
    replayed = runner._validate_calibration_selection(payload, designation)
    assert replayed[2] == designation["role_order"]["calibration"]
    corrupted = copy.deepcopy(payload)
    corrupted["calibration_models"]["primary"]["cohort_log_odds"][0][0][0] += 0.01
    with pytest.raises(PermissionError, match="does not replay exactly"):
        runner._validate_calibration_selection(corrupted, designation)


def test_source_refit_identity_is_replayed_before_held_gex(
    calibration_artifact: tuple[dict[str, object], dict[str, object]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration_payload, designation = calibration_artifact
    replayed = runner._validate_calibration_selection(calibration_payload, designation)
    (
        calibration_tables,
        calibration_destroyed,
        calibration_ids,
        calibration_cohorts,
        selected,
        alphas,
        calibration_models_payload,
    ) = replayed
    pilot_tables, pilot_destroyed = _panel(12)
    pilot_ids = designation["role_order"]["pilot"]
    by_id = {record["patient_id"]: record for record in designation["patients"]}
    pilot_cohorts = [by_id[patient_id]["group"] for patient_id in pilot_ids]
    pilot_losses = core.serialized_panel_losses(
        core.deserialize_models(calibration_models_payload),
        pilot_tables,
        pilot_cohorts,
    )
    gate = {"passes": True, "synthetic_replay_gate": True}
    monkeypatch.setattr(runner, "source_gate", lambda losses, cohorts: gate)
    selection_path = tmp_path / "calibration.json"
    selection_path.write_text("{}\n")
    monkeypatch.setattr(runner, "CALIBRATION_SELECTION", selection_path)
    monkeypatch.setattr(
        runner,
        "_validate_calibration_selection",
        lambda selection, current_designation: replayed,
    )
    expected_models = core.serialize_models(
        core.fit_models(
            np.concatenate((calibration_tables, pilot_tables)),
            np.concatenate((calibration_destroyed, pilot_destroyed)),
            calibration_cohorts + pilot_cohorts,
            selected,
            alphas,
        )
    )
    audits = [
        {
            "patient_id": patient_id,
            "tables_sha256": runner._array_sha256(pilot_tables[index]),
            "destroyed_tables_sha256": runner._array_sha256(pilot_destroyed[index]),
        }
        for index, patient_id in enumerate(pilot_ids)
    ]
    source = {
        "schema": "gse313642-hcc-source-result/1.0",
        "status": "SOURCE_PASS_REFIT_23",
        "rerun_permitted": False,
        "calibration_selection_sha256": runner._sha256(selection_path),
        "selected_configuration": runner._configuration_payload(selected),
        "matched_comparator_alphas": alphas,
        "pilot_gate": gate,
        "pilot_losses": runner._loss_payload(pilot_losses, pilot_ids),
        "patient_order": {
            "calibration": calibration_ids,
            "pilot": pilot_ids,
        },
        "cohorts": dict(
            zip(
                calibration_ids + pilot_ids,
                calibration_cohorts + pilot_cohorts,
            )
        ),
        "reduction_audit": audits,
        "pilot_tables": pilot_tables.tolist(),
        "pilot_destroyed_tables": pilot_destroyed.tolist(),
        "pilot_table_hashes": runner._panel_hashes(pilot_ids, pilot_tables),
        "pilot_destroyed_table_hashes": runner._panel_hashes(
            pilot_ids, pilot_destroyed
        ),
        "source_patient_count": 23,
        "source_models": expected_models,
    }
    assert runner._validate_source_result(source, designation) == expected_models
    source["source_models"]["primary"]["cohort_log_odds"][0][0][0] += 0.01
    with pytest.raises(PermissionError, match="do not replay exactly"):
        runner._validate_source_result(source, designation)


def test_report_only_boundary_does_not_abort_mandatory_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tables, destroyed = _panel(4)
    sentinel = object()
    monkeypatch.setattr(core, "_fit_primary", lambda *args, **kwargs: sentinel)
    monkeypatch.setattr(core, "_fit_cohort_poisson", lambda *args, **kwargs: sentinel)
    monkeypatch.setattr(
        core,
        "_fit_cohort_residual",
        lambda *args, **kwargs: {"A": np.zeros((9, 9)), "B": np.zeros((9, 9))},
    )
    monkeypatch.setattr(
        core,
        "fit_poisson_loglinear_interaction",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            PoissonLoglinearRefusal("boundary")
        ),
    )
    monkeypatch.setattr(
        core,
        "_fit_cohort_cmle",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            CouplingEstimationRefusal("boundary")
        ),
    )
    fitted = core.fit_models(
        tables,
        destroyed,
        ["A", "A", "B", "B"],
        core.PrimaryConfig(1.0, 1.0),
        {"cohort_poisson": 1.0, "cohort_signed_deviance": 1.0},
    )
    assert fitted["cohort_poisson"] is sentinel
    assert fitted["pooled_poisson"] is None
    assert fitted["cohort_exact_cmle"] is None
    assert fitted["pooled_poisson_refusal"] == "FINITE_POOLED_POISSON_UNAVAILABLE"


def test_gex_only_predictions_have_exact_frozen_fb_margins(
    serialized_models: tuple[dict[str, object], dict[str, object]],
) -> None:
    _, models = serialized_models
    marker_rows = np.tile(np.asarray((300, 212)), (9, 1))
    rows = np.repeat(marker_rows[:, None, :], 9, axis=1)
    columns = np.broadcast_to(np.asarray((256, 256)), rows.shape).copy()
    predictions = core.predict_serialized_at_margins(models, rows, columns, "A")
    assert {"cohort_exact_cmle", "independence"} <= set(predictions)
    for estimate in predictions.values():
        assert np.allclose(estimate.sum(axis=-1), rows)
        assert np.allclose(estimate.sum(axis=-2), columns)


def test_score_preflight_rejects_corrupted_private_state_before_fb(
    serialized_models: tuple[dict[str, object], dict[str, object]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serialized, models = serialized_models
    source = tmp_path / "source.json"
    source.write_text("{}\n")
    monkeypatch.setattr(runner, "SOURCE_RESULT", source)
    designation = runner._designation()
    public, private = _prediction_artifacts(serialized, models, designation)
    runner._validate_prediction_state(public, private, designation, serialized)
    private["patients"][0]["rna_states"][0][0] = 2
    with pytest.raises(runner.ProtocolRefusal, match="PRIVATE_HELD_GEX_STATE_DIFFERS"):
        runner._validate_prediction_state(public, private, designation, serialized)


def test_score_preflight_recomputes_every_frozen_prediction(
    serialized_models: tuple[dict[str, object], dict[str, object]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serialized, models = serialized_models
    source = tmp_path / "source.json"
    source.write_text("{}\n")
    monkeypatch.setattr(runner, "SOURCE_RESULT", source)
    designation = runner._designation()
    public, private = _prediction_artifacts(serialized, models, designation)
    estimate = np.asarray(public["patients"][0]["predictions"]["primary"])
    estimate[0, 0] += np.asarray([[0.01, -0.01], [-0.01, 0.01]])
    public["patients"][0]["predictions"]["primary"] = estimate.tolist()
    with pytest.raises(runner.ProtocolRefusal, match="FROZEN_PREDICTION_TABLE_DIFFERS"):
        runner._validate_prediction_state(public, private, designation, serialized)


def test_score_preflight_accepts_margin_forced_structural_zeros(
    serialized_models: tuple[dict[str, object], dict[str, object]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serialized, models = serialized_models
    source = tmp_path / "source.json"
    source.write_text("{}\n")
    monkeypatch.setattr(runner, "SOURCE_RESULT", source)
    designation = runner._designation()
    public, private = _prediction_artifacts(
        serialized, models, designation, degenerate=True
    )
    assert all(
        np.all(np.asarray(values)[0, :, 1, :] == 0.0)
        for values in public["patients"][0]["predictions"].values()
    )
    runner._validate_prediction_state(
        public,
        private,
        designation,
        serialized,
    )


def test_gate_scope_shared_bootstrap_and_within_cohort_requirement() -> None:
    cohorts = ["A"] * 5 + ["B"] * 7
    primary = np.ones(12)
    losses = {"primary": primary}
    for method in core.SOURCE_GATE_COMPARATORS:
        losses[method] = np.full(12, 1.25)
    losses["cohort_exact_cmle"] = np.full(12, 0.5)
    losses["independence"] = np.full(12, 2.0)
    gate = core.held_gate(losses, cohorts)
    assert gate["passes"] is True
    assert set(gate["comparisons"]) == set(core.SOURCE_GATE_COMPARATORS)
    assert gate["shared_stratified_bootstrap"] == {
        "draws": 20_000,
        "seed": core.BOOTSTRAP_SEED,
        "cohort_order": ["A", "B"],
        "same_draw_index_tensor_for_every_comparator": True,
    }

    bad = dict(losses)
    bad["cohort_poisson"] = np.asarray([0.9] * 5 + [1.6] * 7)
    failed = core.held_gate(bad, cohorts)
    assert (
        failed["comparisons"]["cohort_poisson"]["checks"][
            "A_mean_improvement_strictly_positive"
        ]
        is False
    )
    assert failed["passes"] is False


def test_score_result_is_recomputed_from_patient_losses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    designation = runner._designation()
    patient_ids = designation["role_order"]["held"]
    by_id = {record["patient_id"]: record for record in designation["patients"]}
    cohorts = [by_id[patient_id]["group"] for patient_id in patient_ids]
    methods = (
        "primary",
        *core.HELD_GATE_COMPARATORS,
        "cohort_exact_cmle",
        "independence",
    )
    prediction = tmp_path / "predictions.json"
    prediction.write_text(
        json.dumps(
            {
                "patients": [
                    {
                        "patient_id": patient_id,
                        "predictions": {method: [] for method in methods},
                    }
                    for patient_id in patient_ids
                ]
            }
        )
        + "\n"
    )
    monkeypatch.setattr(runner, "PREDICTION_RESULT", prediction)
    losses = {"primary": np.full(12, 1.0)}
    losses.update({method: np.full(12, 1.25) for method in core.HELD_GATE_COMPARATORS})
    losses["cohort_exact_cmle"] = np.full(12, 1.1)
    losses["independence"] = np.full(12, 1.5)
    gate = core.held_gate(losses, cohorts)
    value = {
        "schema": "gse313642-hcc-score-result/1.0",
        "status": "CONFIRMATION_PASS",
        "rerun_permitted": False,
        "prediction_result_sha256": runner._sha256(prediction),
        "held_patient_order": patient_ids,
        "held_cohorts": dict(zip(patient_ids, cohorts)),
        "held_gate": gate,
        "held_losses": runner._loss_payload(losses, patient_ids),
        "reduction_audit": [
            {
                "patient_id": patient_id,
                "group": cohort,
                "truth_tables_sha256": f"{index:064x}",
            }
            for index, (patient_id, cohort) in enumerate(zip(patient_ids, cohorts), 1)
        ],
        "held_gex_gets_during_score": 0,
        "predictions_reconstructed_after_fb_access": False,
    }
    assert set(runner._validate_score_result(value, designation)) == set(methods)

    bad_mean = copy.deepcopy(value)
    bad_mean["held_losses"]["primary"]["mean"] += 0.01
    with pytest.raises(PermissionError, match="patient loss values"):
        runner._validate_score_result(bad_mean, designation)

    bad_status = copy.deepcopy(value)
    bad_status["status"] = "COMPLETED_CONFIRMATION_FAIL"
    with pytest.raises(PermissionError, match="gate does not recompute"):
        runner._validate_score_result(bad_status, designation)


def test_claim_calibration_creates_no_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight = tmp_path / "preflight.json"
    preflight.write_text("{}\n")
    monkeypatch.setattr(runner, "AXIS_PREFLIGHT", preflight)
    monkeypatch.setattr(runner, "CALIBRATION_ATTEMPT", tmp_path / "attempt.json")
    monkeypatch.setattr(runner, "CALIBRATION_SELECTION", tmp_path / "selection.json")
    monkeypatch.setattr(runner, "CALIBRATION_ACCESS", tmp_path / "access.jsonl")
    monkeypatch.setattr(
        runner,
        "_verify_implementation",
        lambda: {"implementation_commit": "a" * 40},
    )
    monkeypatch.setattr(runner, "_require_public_tag", lambda *args: "b" * 40)
    monkeypatch.setattr(runner, "_require_ancestor", lambda *args: None)
    monkeypatch.setattr(runner, "_require_runtime", lambda: runner.REQUIRED_RUNTIME)
    monkeypatch.setattr(
        runner,
        "_open_url",
        lambda request: (_ for _ in ()).throw(AssertionError("unexpected GET")),
    )
    token = tmp_path / "token"
    payload = runner.claim_calibration(token)
    assert payload["status"] == "CLAIMED_BEFORE_MATRIX_ACCESS"
    assert token.is_file()
    rows = [
        json.loads(line) for line in runner.CALIBRATION_ACCESS.read_text().splitlines()
    ]
    assert [row["event"] for row in rows] == ["OPENED_BEFORE_MATRIX_ACCESS"]


class _Response(io.BytesIO):
    def __init__(self, body: bytes, url: str):
        super().__init__(body)
        self.status = 200
        self.headers = {"Content-Length": str(len(body))}
        self._url = url

    def geturl(self) -> str:
        return self._url

    def getcode(self) -> int:
        return self.status

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def test_axis_get_is_journaled_before_gzip_crc_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = runner._designation()["patients"][0]
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb") as stream:
        stream.write(b"CELL-1\nCELL-2\n")
    body = buffer.getvalue()
    journal = tmp_path / "axis-access.jsonl"
    journal.write_text(json.dumps({"event": "HEADER"}) + "\n")
    destination = tmp_path / "barcodes.tsv.gz"
    monkeypatch.setattr(runner, "AXIS_ACCESS", journal)
    monkeypatch.setattr(
        runner,
        "_manifest_file",
        lambda *args: {
            "allowed_stage": "axis_preflight",
            "expected_bytes": len(body),
        },
    )
    monkeypatch.setattr(runner, "_url", lambda *args: "https://example.test/axis")
    monkeypatch.setattr(
        runner,
        "_open_url",
        lambda request: _Response(body, request.full_url),
    )
    decode = runner._decode_gzip_lines

    def assert_download_was_recorded(path: Path) -> tuple[bytes, list[str]]:
        events = [
            json.loads(line)["event"] for line in journal.read_text().splitlines()
        ]
        assert events[-1] == "GET_COMPLETED"
        return decode(path)

    monkeypatch.setattr(runner, "_decode_gzip_lines", assert_download_was_recorded)
    result = runner._download_axis(record, "GEX", "barcodes.tsv.gz", destination)
    assert result["sha256"] == hashlib.sha256(body).hexdigest()
    assert [json.loads(line)["event"] for line in journal.read_text().splitlines()] == [
        "HEADER",
        "GET_STARTED",
        "GET_COMPLETED",
        "GZIP_PARSE_SUCCEEDED",
    ]


def test_matrix_get_has_one_attempt_exact_size_and_no_redirect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = runner._designation()["patients"][0]
    body = b"abcd"
    file_record = {
        "allowed_stage": "held_gex_prediction",
        "expected_bytes": len(body),
    }
    monkeypatch.setattr(runner, "_manifest_file", lambda *args: file_record)
    monkeypatch.setattr(runner, "_url", lambda *args: "https://example.test/matrix")
    calls = []
    monkeypatch.setattr(
        runner,
        "_open_url",
        lambda request: calls.append(request) or _Response(body, request.full_url),
    )
    journal = tmp_path / "access.jsonl"
    runner._append_jsonl(journal, {"event": "HEADER"}, create=True)
    result = runner._download_matrix(
        record, "GEX", tmp_path / "matrix.gz", journal, "prediction"
    )
    assert result["bytes"] == len(body)
    assert len(calls) == 1

    redirect_calls = []
    monkeypatch.setattr(
        runner,
        "_open_url",
        lambda request: (
            redirect_calls.append(request)
            or _Response(body, "https://redirect.test/matrix")
        ),
    )
    with pytest.raises(
        runner.ProtocolRefusal, match="MATRIX_RESPONSE_OR_REDIRECT_REFUSAL"
    ):
        runner._download_matrix(
            record, "GEX", tmp_path / "redirect.gz", journal, "prediction"
        )
    assert len(redirect_calls) == 1


def test_cli_enforces_four_separate_numeric_stages() -> None:
    choices = runner._parser()._subparsers._group_actions[0].choices
    required = {
        "run-calibration",
        "authorize-source",
        "run-source",
        "authorize-prediction",
        "run-prediction",
        "authorize-score",
        "run-score",
    }
    assert required <= set(choices)
    assert "run-held" not in choices


def test_validation_requires_exact_unique_get_and_delete_sets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    consumption = tmp_path / "consumption.json"
    result = tmp_path / "predictions.json"
    access = tmp_path / "access.jsonl"
    attempt = tmp_path / "attempt.json"
    attempt.write_text("{}\n")
    consumption.write_text(
        json.dumps(
            {
                "schema": "gse313642-hcc-prediction-consumption/1.0",
                "status": "CONSUMED_BEFORE_FIRST_MATRIX_REQUEST",
                "attempt_sha256": runner._sha256(attempt),
                "scratch_identity_sha256": "x",
                "private_state_identity_sha256": "f" * 64,
                "private_state_expected_absent_before_prediction": True,
                "rerun_permitted": False,
            }
        )
        + "\n"
    )
    result.write_text(
        json.dumps(
            {
                "status": "PREDICTIONS_FROZEN_BEFORE_ANY_HELD_FB_ACCESS",
                "rerun_permitted": False,
            }
        )
        + "\n"
    )
    monkeypatch.setattr(runner, "PREDICTION_CONSUMPTION", consumption)
    monkeypatch.setattr(runner, "PREDICTION_RESULT", result)
    monkeypatch.setattr(runner, "PREDICTION_ACCESS", access)
    monkeypatch.setattr(runner, "PREDICTION_ATTEMPT", attempt)
    monkeypatch.setattr(runner, "_require_public_attempt", lambda stage: "a" * 40)
    monkeypatch.setattr(runner, "_require_public_tag", lambda *args: "a" * 40)
    monkeypatch.setattr(runner, "_require_ancestor", lambda *args: None)
    rows = [{"event": "OPENED_BEFORE_MATRIX_ACCESS"}]
    by_id = {
        record["patient_id"]: record for record in runner._designation()["patients"]
    }
    for patient in runner._designation()["role_order"]["held"]:
        record = by_id[patient]
        manifest = runner._manifest_file(record, "GEX", runner.MATRIX_MEMBER)
        rows.append(
            {
                "stage": "prediction",
                "event": "GET_STARTED",
                "patient_id": patient,
                "modality": "GEX",
                "url": runner._url(record, "GEX", runner.MATRIX_MEMBER),
                "expected_bytes": manifest["expected_bytes"],
            }
        )
        rows.append(
            {
                "stage": "prediction",
                "event": "GET_COMPLETED",
                "patient_id": patient,
                "modality": "GEX",
                "bytes": manifest["expected_bytes"],
                "sha256": "0" * 64,
            }
        )
        rows.append(
            {
                "stage": "prediction",
                "event": "MATRIX_DELETED",
                "patient_id": patient,
                "modality": "GEX",
                "body_existed": True,
            }
        )
    access.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert runner.validate("prediction")["completed_matrix_gets"] == 12
    rows.insert(4, copy.deepcopy(rows[2]))
    access.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(PermissionError, match="unique allowed prefix"):
        runner.validate("prediction")


def test_terminal_validation_accepts_only_an_expected_request_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    consumption = tmp_path / "consumption.json"
    result = tmp_path / "result.json"
    access = tmp_path / "access.jsonl"
    attempt = tmp_path / "attempt.json"
    attempt.write_text("{}\n")
    consumption.write_text(
        json.dumps(
            {
                "schema": "gse313642-hcc-prediction-consumption/1.0",
                "status": "CONSUMED_BEFORE_FIRST_MATRIX_REQUEST",
                "attempt_sha256": runner._sha256(attempt),
                "scratch_identity_sha256": "x",
                "private_state_identity_sha256": "f" * 64,
                "private_state_expected_absent_before_prediction": True,
                "rerun_permitted": False,
            }
        )
        + "\n"
    )
    result.write_text(
        json.dumps({"status": "TERMINAL_REFUSAL", "rerun_permitted": False}) + "\n"
    )
    monkeypatch.setattr(runner, "PREDICTION_CONSUMPTION", consumption)
    monkeypatch.setattr(runner, "PREDICTION_RESULT", result)
    monkeypatch.setattr(runner, "PREDICTION_ACCESS", access)
    monkeypatch.setattr(runner, "PREDICTION_ATTEMPT", attempt)
    monkeypatch.setattr(runner, "_require_public_attempt", lambda stage: "a" * 40)
    monkeypatch.setattr(runner, "_require_public_tag", lambda *args: "a" * 40)
    monkeypatch.setattr(runner, "_require_ancestor", lambda *args: None)
    designation = runner._designation()
    by_id = {record["patient_id"]: record for record in designation["patients"]}
    first, second = designation["role_order"]["held"][:2]
    rows = [{"event": "OPENED_BEFORE_MATRIX_ACCESS"}]
    for index, patient in enumerate((first, second)):
        record = by_id[patient]
        manifest = runner._manifest_file(record, "GEX", runner.MATRIX_MEMBER)
        rows.append(
            {
                "stage": "prediction",
                "event": "GET_STARTED",
                "patient_id": patient,
                "modality": "GEX",
                "url": runner._url(record, "GEX", runner.MATRIX_MEMBER),
                "expected_bytes": manifest["expected_bytes"],
            }
        )
        if index == 0:
            rows.append(
                {
                    "stage": "prediction",
                    "event": "GET_COMPLETED",
                    "patient_id": patient,
                    "modality": "GEX",
                    "bytes": manifest["expected_bytes"],
                    "sha256": "0" * 64,
                }
            )
        else:
            rows.append(
                {
                    "stage": "prediction",
                    "event": "GET_FAILED_TERMINALLY",
                    "patient_id": patient,
                    "modality": "GEX",
                }
            )
        rows.append(
            {
                "stage": "prediction",
                "event": "MATRIX_DELETED",
                "patient_id": patient,
                "modality": "GEX",
                "body_existed": True,
            }
        )
    access.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert runner.validate("prediction")["completed_matrix_gets"] == 1
    rows[-2]["patient_id"] = designation["role_order"]["calibration"][0]
    access.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(PermissionError, match="unique allowed prefix"):
        runner.validate("prediction")


def test_prediction_recovery_removes_bound_private_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    state = tmp_path / "private-state.json"
    state.write_text("private\n")
    consumption = tmp_path / "consumption.json"
    result = tmp_path / "result.json"
    access = tmp_path / "access.jsonl"
    access.write_text(json.dumps({"event": "OPENED_BEFORE_MATRIX_ACCESS"}) + "\n")
    consumption.write_text(
        json.dumps(
            {
                "scratch_identity_sha256": hashlib.sha256(
                    str(scratch.resolve()).encode()
                ).hexdigest(),
                "private_state_identity_sha256": hashlib.sha256(
                    str(state.resolve()).encode()
                ).hexdigest(),
            }
        )
        + "\n"
    )
    (scratch / "prediction").mkdir(parents=True)
    (scratch / "prediction" / "partial").write_text("matrix bytes")
    monkeypatch.setattr(runner, "PREDICTION_CONSUMPTION", consumption)
    monkeypatch.setattr(runner, "PREDICTION_RESULT", result)
    monkeypatch.setattr(runner, "PREDICTION_ACCESS", access)
    recovered = runner.recover("prediction", scratch=scratch, state_path=state)
    assert recovered["status"] == "TERMINAL_REFUSAL"
    assert not state.exists()
    assert not (scratch / "prediction").exists()


def test_accidental_second_prediction_run_preserves_prior_private_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    state = tmp_path / "private-state.json"
    state.write_text("prior successful state\n")
    consumption = tmp_path / "consumption.json"
    consumption.write_text("{}\n")
    result = tmp_path / "result.json"
    monkeypatch.setattr(runner, "PREDICTION_CONSUMPTION", consumption)
    monkeypatch.setattr(runner, "PREDICTION_RESULT", result)
    monkeypatch.setattr(runner, "_require_runtime", lambda: runner.REQUIRED_RUNTIME)
    monkeypatch.setattr(runner, "_require_public_attempt", lambda stage: "a" * 40)
    output = runner.run_prediction(
        tmp_path / "unused-token", tmp_path / "unused-axes", state, scratch=scratch
    )
    assert output["status"] == "TERMINAL_REFUSAL"
    assert state.read_text() == "prior successful state\n"


def test_accidental_second_score_run_preserves_prior_private_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    state = tmp_path / "private-state.json"
    state.write_text("prior successful state\n")
    token = tmp_path / "score-token"
    token.write_text("token\n")
    attempt = tmp_path / "attempt.json"
    attempt.write_text(
        json.dumps(
            {"claim_token_sha256": hashlib.sha256(token.read_bytes()).hexdigest()}
        )
        + "\n"
    )
    consumption = tmp_path / "consumption.json"
    consumption.write_text("{}\n")
    result = tmp_path / "result.json"
    monkeypatch.setattr(runner, "SCORE_ATTEMPT", attempt)
    monkeypatch.setattr(runner, "SCORE_CONSUMPTION", consumption)
    monkeypatch.setattr(runner, "SCORE_RESULT", result)
    monkeypatch.setattr(runner, "_require_runtime", lambda: runner.REQUIRED_RUNTIME)
    monkeypatch.setattr(runner, "_require_public_attempt", lambda stage: "a" * 40)
    output = runner.run_score(token, tmp_path / "unused-axes", state, scratch=scratch)
    assert output["status"] == "TERMINAL_REFUSAL"
    assert state.read_text() == "prior successful state\n"


def test_score_recovery_finishes_private_state_cleanup_after_result_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    state = tmp_path / "private-state.json"
    state.write_text("scored state\n")
    consumption = tmp_path / "consumption.json"
    consumption.write_text(
        json.dumps(
            {
                "scratch_identity_sha256": hashlib.sha256(
                    str(scratch.resolve()).encode()
                ).hexdigest(),
                "private_state_identity_sha256": hashlib.sha256(
                    str(state.resolve()).encode()
                ).hexdigest(),
            }
        )
        + "\n"
    )
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"status": "CONFIRMATION_PASS"}) + "\n")
    access = tmp_path / "access.jsonl"
    monkeypatch.setattr(runner, "SCORE_CONSUMPTION", consumption)
    monkeypatch.setattr(runner, "SCORE_RESULT", result)
    monkeypatch.setattr(runner, "SCORE_ACCESS", access)
    recovered = runner.recover("score", scratch=scratch, state_path=state)
    assert recovered["status"] == "CONFIRMATION_PASS"
    assert not state.exists()


def test_divergent_public_tag_ancestry_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = type("GitResult", (), {"returncode": 1})()
    monkeypatch.setattr(runner, "_git", lambda *args, **kwargs: result)
    with pytest.raises(PermissionError, match="required ancestry"):
        runner._require_ancestor("a" * 40, "b" * 40)


def test_numeric_runtime_requires_single_thread_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in runner.THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")
    assert runner._require_runtime() == runner.REQUIRED_RUNTIME
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(PermissionError, match="runtime differs"):
        runner._require_runtime()
