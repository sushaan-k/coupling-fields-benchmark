from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

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


def _matrix_audit(
    banner: str = "%%MatrixMarket matrix coordinate real general",
) -> dict[str, object]:
    return {
        "banner": banner,
        "parsed_nnz": 1,
        "declared_nnz": 1,
        "selected_value_sum": 1,
        "compressed_bytes": 80,
        "compressed_sha256": "c" * 64,
        "compressed_source_exhausted": True,
        "decompressed_bytes": 100,
        "decompressed_sha256": "d" * 64,
        "gzip_stream_exhausted": True,
    }


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
    tables, destroyed = _panel(10)
    selected, primary_losses = core.select_primary_configuration(
        tables, cohorts, expected_patient_count=10
    )
    alphas, comparator_losses = core.select_comparator_alphas(
        tables, cohorts, expected_patient_count=10
    )
    models = core.serialize_models(
        core.fit_models(tables, destroyed, cohorts, selected, alphas)
    )
    audits = [
        {
            "patient_id": patient_id,
            "tables_sha256": runner._array_sha256(tables[index]),
            "destroyed_tables_sha256": runner._array_sha256(destroyed[index]),
            "matrix_market": {
                "GEX": _matrix_audit(),
                "FB": _matrix_audit("%%MatrixMarket matrix coordinate integer general"),
            },
        }
        for index, patient_id in enumerate(patient_ids)
    ]
    payload = {
        "schema": "gse313642-hcc-calibration-selection/3.0",
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


def _write_mtx(path: Path, matrix: np.ndarray, *, real_field: bool = False) -> None:
    rows, columns = np.nonzero(matrix)
    with gzip.open(path, "wt", encoding="ascii", newline="") as stream:
        field = "real" if real_field else "integer"
        stream.write(f"%%MatrixMarket matrix coordinate {field} general\n")
        stream.write(f"{matrix.shape[0]} {matrix.shape[1]} {len(rows)}\n")
        for row, column in zip(rows, columns):
            value = (
                f"{int(matrix[row, column])}.0000000e+00"
                if real_field
                else str(int(matrix[row, column]))
            )
            stream.write(f"{row + 1} {column + 1} {value}\n")


def test_reduce_matrix_journals_canonical_real_parser_audit(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.mtx.gz"
    matrix = np.asarray([[3, 0], [0, 7]], dtype=np.int64)
    _write_mtx(matrix_path, matrix, real_field=True)
    compressed = matrix_path.read_bytes()
    journal = tmp_path / "access.jsonl"
    journal.write_text(json.dumps({"event": "OPENED_BEFORE_MATRIX_ACCESS"}) + "\n")
    record = {"patient_id": "TEST"}

    block, audit = runner._reduce_matrix(
        matrix_path,
        expected_shape=(2, 2),
        selected_rows=(1, 2),
        selected_columns=(1, 2),
        record=record,
        modality="GEX",
        journal=journal,
        stage="prediction",
    )

    assert np.array_equal(block, matrix)
    finished = runner._read_jsonl(journal)[-1]
    assert finished["event"] == "MATRIX_PARSE_FINISHED"
    assert finished["parser_audit"] == runner._matrix_audit(audit)
    assert finished["parser_audit"]["compressed_bytes"] == len(compressed)
    assert (
        finished["parser_audit"]["compressed_sha256"]
        == hashlib.sha256(compressed).hexdigest()
    )


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
                "gex_download": {"bytes": 80, "sha256": "c" * 64},
                "gex_matrix_market": _matrix_audit(),
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
            "schema": "gse313642-hcc-predictions/3.0",
            "status": "PREDICTIONS_FROZEN_BEFORE_ANY_HELD_FB_ACCESS",
            "rerun_permitted": False,
            "source_result_sha256": runner._sha256(runner.SOURCE_RESULT),
            "source_models_sha256": runner._json_sha256(serialized),
            "prediction_attempt_sha256": attempt_sha256,
            "private_state_sha256": "f" * 64,
            "private_state_bytes": 1,
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
            "schema": "gse313642-hcc-private-held-gex-state/3.0",
            "prediction_attempt_sha256": attempt_sha256,
            "patients": private_rows,
        },
    )


def _bind_prediction_attempt(
    public: dict[str, object],
    private: dict[str, object] | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt = tmp_path / "prediction-attempt.json"
    attempt.write_text("{}\n")
    digest = runner._sha256(attempt)
    public["prediction_attempt_sha256"] = digest
    if private is not None:
        private["prediction_attempt_sha256"] = digest
    monkeypatch.setattr(runner, "PREDICTION_ATTEMPT", attempt)


def _held_truth_panel(predictions: dict[str, object]) -> np.ndarray:
    patients = predictions["patients"]
    tables = np.empty((len(patients), 9, 9, 2, 2), dtype=np.int64)
    for patient_index, patient in enumerate(patients):
        rows = np.asarray(patient["row_margins"], dtype=np.int64)
        columns = np.asarray(patient["column_margins"], dtype=np.int64)
        for row_index in range(9):
            for column_index in range(9):
                rna_positive = int(rows[row_index, column_index, 1])
                adt_high = int(columns[row_index, column_index, 1])
                lower = max(0, rna_positive + adt_high - 512)
                upper = min(rna_positive, adt_high)
                n11 = lower + (3 * (upper - lower)) // 4
                if upper > lower:
                    n11 = min(
                        upper, n11 + (patient_index + row_index + column_index) % 2
                    )
                tables[patient_index, row_index, column_index] = np.asarray(
                    [
                        [512 - rna_positive - adt_high + n11, adt_high - n11],
                        [rna_positive - n11, n11],
                    ]
                )
    return tables


def _score_result_payload(
    predictions: dict[str, object],
    prediction_path: Path,
    truth_tables: np.ndarray,
    designation: dict[str, object],
) -> dict[str, object]:
    patient_ids = designation["role_order"]["held"]
    by_id = {record["patient_id"]: record for record in designation["patients"]}
    cohorts = [by_id[patient_id]["group"] for patient_id in patient_ids]
    methods = tuple(predictions["patients"][0]["predictions"])
    losses = {method: np.empty(len(patient_ids), dtype=float) for method in methods}
    for patient_index, (truth, patient) in enumerate(
        zip(truth_tables, predictions["patients"])
    ):
        for method in methods:
            losses[method][patient_index] = float(
                np.mean(
                    runner.entity_deviance(
                        truth,
                        np.asarray(patient["predictions"][method], dtype=float),
                    )
                )
            )
    gate = core.held_gate(losses, cohorts)
    hashes = runner._panel_hashes(patient_ids, truth_tables)
    return {
        "schema": "gse313642-hcc-score-result/3.0",
        "status": (
            "CONFIRMATION_PASS" if gate["passes"] else "COMPLETED_CONFIRMATION_FAIL"
        ),
        "rerun_permitted": False,
        "prediction_result_sha256": runner._sha256(prediction_path),
        "held_patient_order": patient_ids,
        "held_cohorts": dict(zip(patient_ids, cohorts)),
        "held_gate": gate,
        "held_losses": runner._loss_payload(losses, patient_ids),
        "held_truth_tables": truth_tables.tolist(),
        "held_truth_table_hashes": hashes,
        "reduction_audit": [
            {
                "patient_id": patient_id,
                "group": cohort,
                "truth_tables_sha256": hashes[patient_id],
                "fb_matrix_market": _matrix_audit(),
            }
            for patient_id, cohort in zip(patient_ids, cohorts)
        ],
        "held_gex_gets_during_score": 0,
        "predictions_reconstructed_after_fb_access": False,
    }


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
    assert len(candidate["patients"]) == 33
    assert len(manifest) == 198
    assert {record["patient_id"] for record in candidate["patients"]}.isdisjoint(
        {"A30", "A33"}
    )
    assert candidate["role_order"]["calibration"] == [
        "A12",
        "A08",
        "A32",
        "A34",
        "B15",
        "B23",
        "B17",
        "B04",
        "B08",
        "B03",
    ]
    assert candidate["role_order"]["pilot"] == [
        "A04",
        "A03",
        "A35",
        "A31",
        "B02",
        "B07",
        "B06",
        "B05",
        "B21",
        "B01",
        "B12",
    ]
    axes = [
        record
        for record in manifest.values()
        if record["member"] in runner.AXIS_MEMBERS
    ]
    matrices = [
        record
        for record in manifest.values()
        if record["member"] == runner.MATRIX_MEMBER
    ]
    assert len(axes) == 132
    assert len(matrices) == 66
    assert all(
        record["allowed_stage"] == "v3_reuse_only_no_get"
        and record["v3_get_authorized"] is False
        for record in axes
    )
    assert all(record["v3_get_authorized"] is True for record in matrices)
    assert sum(record["allowed_stage"] == "calibration" for record in matrices) == 20
    assert sum(record["allowed_stage"] == "source" for record in matrices) == 22
    assert sum(record["allowed_stage"] == "prediction" for record in matrices) == 12
    assert sum(record["allowed_stage"] == "score" for record in matrices) == 12
    assert candidate["role_order"]["held"][:5] == [
        "A05",
        "A02",
        "A07",
        "A21",
        "A36",
    ]


def test_clean_runner_import_closure_is_frozen() -> None:
    script = """
import json
from pathlib import Path
import sys
root = Path.cwd().resolve()
import experiments.confirm_gse313642_hcc  # noqa: F401
paths = []
for module in sys.modules.values():
    filename = getattr(module, "__file__", None)
    if not filename:
        continue
    path = Path(filename).resolve()
    if not path.is_file():
        continue
    try:
        paths.append(path.relative_to(root).as_posix())
    except ValueError:
        pass
print(json.dumps(sorted(set(paths))))
"""
    observed = set(
        json.loads(
            subprocess.run(
                [sys.executable, "-c", script],
                cwd=runner.ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    )
    assert observed <= set(runner.IMPLEMENTATION_BINDINGS)
    assert {"mapreg/__init__.py", "mapreg/factorial_coupling.py"} <= observed


def test_preflight_replays_exact_implementation_runtime_and_network_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    designation = runner._designation()
    implementation_commit = "a" * 40
    access_audit = {"frozen_axis_audit": True}
    patients = [
        {"patient_id": record["patient_id"]} for record in designation["patients"]
    ]
    preflight = tmp_path / "preflight.json"
    payload = {
        "schema": "gse313642-hcc-axis-preflight/3.0",
        "status": "PASS_BEFORE_ANY_MATRIX_REQUEST",
        "designation_sha256": runner._sha256(runner.DESIGNATION),
        "implementation_commit": implementation_commit,
        "patient_count": 33,
        "v3_matrix_body_or_header_requested": False,
        "series_tar_used": False,
        "axis_access": access_audit,
        "v1_terminal_axis_refusal": {
            "terminal_refusal_tag": runner.V1_TERMINAL_TAG,
            "terminal_refusal_commit": runner.V1_TERMINAL_COMMIT,
        },
        "required_runtime": runner.REQUIRED_RUNTIME,
        "patients": patients,
    }
    preflight.write_text(json.dumps(payload) + "\n")
    monkeypatch.setattr(runner, "AXIS_PREFLIGHT", preflight)
    monkeypatch.setattr(
        runner, "_validate_axis_access", lambda axis_root, current: access_audit
    )
    monkeypatch.setattr(
        runner,
        "_inspect_pair_axes",
        lambda axis_root, record: {"patient_id": record["patient_id"]},
    )
    runner._validate_preflight(
        tmp_path / "axes", implementation_commit=implementation_commit
    )

    payload["series_tar_used"] = True
    preflight.write_text(json.dumps(payload) + "\n")
    with pytest.raises(PermissionError, match="axis preflight differs"):
        runner._validate_preflight(
            tmp_path / "axes", implementation_commit=implementation_commit
        )


def test_freeze_and_preflight_cannot_be_regenerated_after_downstream_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    implementation_freeze = tmp_path / "implementation.json"
    preflight = tmp_path / "preflight.json"
    score_result = tmp_path / "score.json"
    score_result.write_text("{}\n")
    monkeypatch.setattr(runner, "IMPLEMENTATION_FREEZE", implementation_freeze)
    monkeypatch.setattr(runner, "AXIS_PREFLIGHT", preflight)
    monkeypatch.setattr(runner, "SCORE_RESULT", score_result)
    monkeypatch.setattr(
        runner,
        "_verify_implementation",
        lambda: (_ for _ in ()).throw(AssertionError("verification was reached")),
    )
    monkeypatch.setattr(
        runner,
        "_require_public_tag",
        lambda *args: (_ for _ in ()).throw(AssertionError("tag lookup was reached")),
    )

    assert score_result in runner._numeric_stage_artifacts()
    with pytest.raises(FileExistsError, match="downstream artifact"):
        runner.freeze_implementation()
    with pytest.raises(FileExistsError, match="downstream numeric artifact"):
        runner.preflight(tmp_path / "axes", output=preflight)
    assert not implementation_freeze.exists()
    assert not preflight.exists()


def test_v1_terminal_refusal_and_immutable_journal_are_bound() -> None:
    terminal = runner._v1_terminal_refusal()
    assert terminal == {
        "tag": runner.V1_TERMINAL_TAG,
        "commit": runner.V1_TERMINAL_COMMIT,
        "artifact_sha256": runner.V1_AXIS_REFUSAL_SHA256,
        "refusal_code": "BARCODE_AXIS_NOT_UNIQUE",
        "excluded_patient_id": "A33",
    }
    assert len(runner._read_jsonl(runner.V1_AXIS_ACCESS)) == 421


def test_rank_or_hash_mutation_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    value = json.loads(runner.BASE_DESIGNATION.read_text())
    value["patients"][0]["split_sha256"] = "0" * 64
    read_json = runner._read_json
    monkeypatch.setattr(
        runner,
        "_read_json",
        lambda path: value if path == runner.BASE_DESIGNATION else read_json(path),
    )
    with pytest.raises(PermissionError, match="v2 must equal the v1 patient panel"):
        runner._designation()


def test_v3_overlay_base_hash_and_stage_mapping_mutations_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    designation = json.loads(runner.DESIGNATION.read_text())
    designation["base_designation"]["sha256"] = "0" * 64
    read_json = runner._read_json
    monkeypatch.setattr(
        runner,
        "_read_json",
        lambda path: designation if path == runner.DESIGNATION else read_json(path),
    )
    with pytest.raises(PermissionError, match="v3 candidate overlay"):
        runner._designation()

    manifest = json.loads(runner.SOURCE_MANIFEST.read_text())
    manifest["allowed_stage_mapping"]["source_calibration"] = "source"
    monkeypatch.setattr(
        runner,
        "_read_json",
        lambda path: manifest if path == runner.SOURCE_MANIFEST else read_json(path),
    )
    with pytest.raises(PermissionError, match="v3 source manifest overlay"):
        runner._source_manifest()


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
    v1 = json.loads(runner.V1_DESIGNATION.read_text())
    retained = [
        record
        for record in v1["patients"]
        if record["patient_id"] not in {"A30", "A33"}
    ]
    access = tmp_path / "axis-access.jsonl"
    monkeypatch.setattr(runner, "V1_AXIS_ACCESS", access)
    monkeypatch.setattr(runner, "AXIS_ACCESS", access)
    monkeypatch.setattr(
        runner,
        "_open_url",
        lambda request: (_ for _ in ()).throw(AssertionError("unexpected GET")),
    )
    rows = [
        {
            "schema": "gse313642-hcc-axis-access/1.0",
            "stage": "axis_acquisition",
            "event": "OPENED_BEFORE_FIRST_AXIS_GET",
            "matrix_requests": 0,
            "series_tar_used": False,
        }
    ]
    historical_manifest = {}
    current_manifest = {}
    for record in v1["patients"]:
        for modality in ("GEX", "FB"):
            for member in runner.AXIS_MEMBERS:
                key = (record["patient_id"], modality, member)
                path = tmp_path / runner._filename(record, modality, member)
                if record["patient_id"] in {"A30", "A33"}:
                    expected_bytes = 123
                    digest = hashlib.sha256(
                        f"excluded-{record['patient_id']}-axis".encode()
                    ).hexdigest()
                else:
                    _write_axis(path, ["one-axis-row"])
                    expected_bytes = path.stat().st_size
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                manifest = {
                    "patient_id": record["patient_id"],
                    "deposited_patient_id": record["deposited_patient_id"],
                    "modality": modality,
                    "member": member,
                    "gsm": record["gex_gsm" if modality == "GEX" else "fb_gsm"],
                    "filename": path.name,
                    "expected_bytes": expected_bytes,
                    "allowed_stage": "axis_preflight",
                }
                historical_manifest[key] = manifest
                if record["patient_id"] not in {"A30", "A33"}:
                    current_manifest[key] = {
                        **manifest,
                        "allowed_stage": "v3_reuse_only_no_get",
                        "v3_get_authorized": False,
                    }
                url = runner._manifest_url(manifest)
                rows.extend(
                    (
                        {
                            "stage": "axis_acquisition",
                            "event": "GET_STARTED",
                            "patient_id": record["patient_id"],
                            "modality": modality,
                            "member": member,
                            "url": url,
                            "expected_bytes": expected_bytes,
                        },
                        {
                            "stage": "axis_acquisition",
                            "event": "GET_COMPLETED",
                            "patient_id": record["patient_id"],
                            "modality": modality,
                            "member": member,
                            "bytes": expected_bytes,
                            "sha256": digest,
                        },
                        {
                            "stage": "axis_acquisition",
                            "event": "GZIP_PARSE_SUCCEEDED",
                            "patient_id": record["patient_id"],
                            "modality": modality,
                            "member": member,
                            "download_sha256": digest,
                            "line_count": 1,
                        },
                    )
                )
    monkeypatch.setattr(runner, "_v1_axis_manifest", lambda: (v1, historical_manifest))
    monkeypatch.setattr(
        runner,
        "_v1_terminal_refusal",
        lambda: {"excluded_patient_id": "A33"},
    )
    monkeypatch.setattr(runner, "_source_manifest", lambda: current_manifest)
    access.write_text("".join(json.dumps(row) + "\n" for row in rows))
    audit = runner._validate_axis_access(tmp_path, {"patients": retained})
    assert audit["v1_journal_rows"] == 421
    assert audit["v1_axis_gets"] == 140
    assert audit["v3_axis_gets"] == 0
    assert audit["retained_axis_files"] == 132
    assert audit["excluded_axis_files"] == 8
    assert audit["excluded_patient_ids"] == ["A30", "A33"]
    assert len(audit["files"]) == 132
    assert not list(tmp_path.glob("*A-30-01_01*"))
    assert not list(tmp_path.glob("*A-33-01_01*"))

    rows[1]["url"] = "https://wrong.test/file"
    access.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(PermissionError, match="axis access file binding differs"):
        runner._validate_axis_access(tmp_path, {"patients": retained})
    rows[1]["url"] = runner._manifest_url(
        historical_manifest[("A02", "GEX", runner.AXIS_MEMBERS[0])]
    )
    rows[2]["bytes"] = int(rows[2]["bytes"]) + 1
    access.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(PermissionError, match="axis access file binding differs"):
        runner._validate_axis_access(tmp_path, {"patients": retained})


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
    _write_mtx(gex_path, gex, real_field=True)
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
    assert audit["matrix_market"]["GEX"]["banner"].endswith("real general")
    assert audit["matrix_market"]["FB"]["banner"].endswith("integer general")

    gex_counts, selected, gex_audit = runner._reduce_one_modality(
        gex_path, tmp_path, record, preflight, "GEX"
    )
    fb_counts, joined, fb_audit = runner._reduce_one_modality(
        fb_path, tmp_path, record, preflight, "FB", selected
    )
    assert gex_counts.shape == fb_counts.shape == (512, 9)
    assert joined == selected
    assert gex_audit["banner"].endswith("real general")
    assert fb_audit["banner"].endswith("integer general")


def test_real_parse_failure_is_structured_before_deletion_and_publicly_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, preflight = _one_pair_axes(tmp_path, monkeypatch)
    matrix_path = tmp_path / "invalid-real.mtx.gz"
    with gzip.open(matrix_path, "wt", encoding="ascii", newline="") as stream:
        stream.write("%%MatrixMarket matrix coordinate real general\n")
        stream.write("10 520 1\n")
        stream.write("10 520 1.5\n")
    journal = tmp_path / "calibration-access-v3.jsonl"
    runner._append_jsonl(journal, {"event": "OPENED_BEFORE_MATRIX_ACCESS"}, create=True)

    with pytest.raises(runner.ProtocolRefusal, match="MATRIX_PARSE_VALIDATION_REFUSAL"):
        runner._reduce_one_modality(
            matrix_path,
            tmp_path,
            record,
            preflight,
            "GEX",
            journal=journal,
            stage="calibration",
        )
    matrix_path.unlink()
    runner._append_jsonl(
        journal,
        {
            "stage": "calibration",
            "event": "MATRIX_DELETED",
            "patient_id": record["patient_id"],
            "modality": "GEX",
            "body_existed": True,
        },
    )
    events = runner._read_jsonl(journal)
    assert [event["event"] for event in events] == [
        "OPENED_BEFORE_MATRIX_ACCESS",
        "MATRIX_PARSE_STARTED",
        "MATRIX_PARSE_FAILED",
        "MATRIX_DELETED",
    ]
    failed = events[2]
    assert failed["exception_class"] == "GzipMatrixMarketValidationError"
    assert failed["refusal_code"] == "MATRIX_PARSE_VALIDATION_REFUSAL"
    assert "finite nonnegative integral real" in failed["message"]
    assert failed["partial_audit"]["declared_nnz"] == 1
    assert failed["partial_audit"]["parsed_nnz"] == 0

    attempt = tmp_path / "calibration-attempt-v3.json"
    consumption = tmp_path / "calibration-consumption-v3.json"
    result = tmp_path / "calibration-result-v3.json"
    attempt.write_text("{}\n")
    consumption.write_text("{}\n")
    monkeypatch.setattr(runner, "CALIBRATION_ATTEMPT", attempt)
    monkeypatch.setattr(runner, "CALIBRATION_CONSUMPTION", consumption)
    monkeypatch.setattr(runner, "CALIBRATION_ACCESS", journal)
    monkeypatch.setattr(runner, "CALIBRATION_SELECTION", result)
    refusal = runner._failure("calibration", failed["refusal_code"])
    assert refusal["refusal_code"] == failed["refusal_code"]
    assert refusal["access_journal"] == journal.name
    assert refusal["access_journal_sha256"] == runner._sha256(journal)


def test_v3_mutable_artifacts_tags_and_scratch_are_isolated() -> None:
    mutable_paths = (
        runner.AXIS_PREFLIGHT,
        runner.IMPLEMENTATION_FREEZE,
        runner.CALIBRATION_ATTEMPT,
        runner.CALIBRATION_CONSUMPTION,
        runner.CALIBRATION_ACCESS,
        runner.CALIBRATION_SELECTION,
        runner.PILOT_AUTHORIZATION,
        runner.SOURCE_ATTEMPT,
        runner.SOURCE_CONSUMPTION,
        runner.SOURCE_ACCESS,
        runner.SOURCE_RESULT,
        runner.PREDICTION_AUTHORIZATION,
        runner.PREDICTION_ATTEMPT,
        runner.PREDICTION_CONSUMPTION,
        runner.PREDICTION_ACCESS,
        runner.PREDICTION_RESULT,
        runner.SCORE_AUTHORIZATION,
        runner.SCORE_ATTEMPT,
        runner.SCORE_CONSUMPTION,
        runner.SCORE_ACCESS,
        runner.SCORE_RESULT,
    )
    assert all("v3" in path.name for path in mutable_paths)
    tags = (
        runner.CANDIDATE_TAG,
        runner.IMPLEMENTATION_TAG,
        runner.PREFLIGHT_TAG,
        runner.CALIBRATION_ATTEMPT_TAG,
        runner.CALIBRATION_TAG,
        runner.PILOT_AUTHORIZATION_TAG,
        runner.SOURCE_ATTEMPT_TAG,
        runner.SOURCE_TAG,
        runner.PREDICTION_AUTHORIZATION_TAG,
        runner.PREDICTION_ATTEMPT_TAG,
        runner.PREDICTION_TAG,
        runner.SCORE_AUTHORIZATION_TAG,
        runner.SCORE_ATTEMPT_TAG,
        runner.SCORE_TAG,
    )
    assert all("-v3-" in tag for tag in tags)
    assert runner.DEFAULT_SCRATCH.name == "gse313642-hcc-v3"
    assert "v2" in runner.BASE_DESIGNATION.name
    assert "v2" in runner.V2_CALIBRATION_RESULT.name
    assert "v1" in runner.V1_AXIS_REFUSAL.name


def test_calibration_only_selects_comparator_alphas() -> None:
    tables, _ = _panel(10)
    cohorts = ["A"] * 4 + ["B"] * 6
    selected, losses = core.select_comparator_alphas(
        tables, cohorts, expected_patient_count=10
    )
    assert set(selected) == {"cohort_poisson", "cohort_signed_deviance"}
    assert set(selected.values()) <= {0.75, 1.0}
    assert all(
        losses[method][alpha].shape == (10,)
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


def test_source_pass_and_terminal_gate_results_are_replayed_before_held_gex(
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
    pilot_tables, pilot_destroyed = _panel(11)
    pilot_ids = designation["role_order"]["pilot"]
    by_id = {record["patient_id"]: record for record in designation["patients"]}
    pilot_cohorts = [by_id[patient_id]["group"] for patient_id in pilot_ids]
    pilot_losses = core.serialized_panel_losses(
        core.deserialize_models(calibration_models_payload),
        pilot_tables,
        pilot_cohorts,
    )
    gate = {"passes": True, "synthetic_replay_gate": True}
    monkeypatch.setattr(runner, "_source_gate", lambda losses, cohorts: gate)
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
            "matrix_market": {
                "GEX": _matrix_audit(),
                "FB": _matrix_audit("%%MatrixMarket matrix coordinate integer general"),
            },
        }
        for index, patient_id in enumerate(pilot_ids)
    ]
    source = {
        "schema": "gse313642-hcc-source-result/3.0",
        "status": "SOURCE_PASS_REFIT_21",
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
        "source_patient_count": 21,
        "source_models": expected_models,
    }
    failed_source = copy.deepcopy(source)
    assert runner._validate_source_result(source, designation) == expected_models
    source["source_models"]["primary"]["cohort_log_odds"][0][0][0] += 0.01
    with pytest.raises(PermissionError, match="do not replay exactly"):
        runner._validate_source_result(source, designation)

    failed_gate = {"passes": False, "synthetic_replay_gate": True}
    monkeypatch.setattr(runner, "_source_gate", lambda losses, cohorts: failed_gate)
    failed_source["status"] = "TERMINAL_SOURCE_GATE_FAIL"
    failed_source["pilot_gate"] = failed_gate
    del failed_source["source_patient_count"]
    del failed_source["source_models"]
    assert runner._validate_source_result(failed_source, designation) == {}

    for key, value in (
        ("source_patient_count", 21),
        ("source_models", expected_models),
    ):
        corrupted = copy.deepcopy(failed_source)
        corrupted[key] = value
        with pytest.raises(PermissionError, match="contains a refit"):
            runner._validate_source_result(corrupted, designation)

    corrupted = copy.deepcopy(failed_source)
    corrupted["pilot_gate"] = {"passes": True}
    with pytest.raises(PermissionError, match="do not replay exactly"):
        runner._validate_source_result(corrupted, designation)
    corrupted = copy.deepcopy(failed_source)
    corrupted["reduction_audit"] = []
    with pytest.raises(PermissionError, match="reduction audit"):
        runner._validate_source_result(corrupted, designation)


def test_v3_inherits_source_gate_eight_of_eleven_and_both_cohorts() -> None:
    cohorts = ["A"] * 4 + ["B"] * 7
    difference = np.asarray(
        [-0.2, -0.2, -0.2, 0.1, -0.2, -0.2, -0.2, -0.2, -0.2, 0.1, 0.1]
    )
    losses = {"primary": 1.0 + difference}
    losses.update({method: np.ones(11) for method in core.SOURCE_GATE_COMPARATORS})
    gate = runner._source_gate(losses, cohorts)
    assert gate["passes"] is True
    assert all(
        record["favorable_patients"] == 8
        and record["checks"]["at_least_eight_of_eleven_favorable"] is True
        for record in gate["comparisons"].values()
    )

    failed_losses = dict(losses)
    failed_difference = difference.copy()
    failed_difference[0] = 0.1
    failed_losses["primary"] = 1.0 + failed_difference
    failed = runner._source_gate(failed_losses, cohorts)
    assert failed["passes"] is False
    assert all(
        record["checks"]["at_least_eight_of_eleven_favorable"] is False
        for record in failed["comparisons"].values()
    )


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
    _bind_prediction_attempt(public, private, tmp_path, monkeypatch)
    runner._validate_prediction_state(public, private, designation, serialized)
    wrong_attempt = copy.deepcopy(public)
    wrong_attempt["prediction_attempt_sha256"] = "0" * 64
    with pytest.raises(runner.ProtocolRefusal, match="FROZEN_PREDICTION_PATIENT_AXIS"):
        runner._validate_prediction_state(
            wrong_attempt, private, designation, serialized
        )
    bad_private_digest = copy.deepcopy(public)
    bad_private_digest["private_state_sha256"] = "not-a-digest"
    with pytest.raises(runner.ProtocolRefusal, match="FROZEN_PREDICTION_PATIENT_AXIS"):
        runner._validate_prediction_state(
            bad_private_digest, private, designation, serialized
        )
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
    _bind_prediction_attempt(public, private, tmp_path, monkeypatch)
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
    _bind_prediction_attempt(public, private, tmp_path, monkeypatch)
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


def test_score_result_replays_truth_predictions_losses_hashes_and_status(
    serialized_models: tuple[dict[str, object], dict[str, object]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serialized, models = serialized_models
    designation = runner._designation()
    patient_ids = designation["role_order"]["held"]
    source = tmp_path / "source.json"
    source.write_text("{}\n")
    monkeypatch.setattr(runner, "SOURCE_RESULT", source)
    monkeypatch.setattr(
        runner,
        "_validate_source_result",
        lambda source_value, current_designation: serialized,
    )
    predictions, _ = _prediction_artifacts(serialized, models, designation)
    _bind_prediction_attempt(predictions, None, tmp_path, monkeypatch)
    prediction = tmp_path / "predictions.json"
    prediction.write_text(json.dumps(predictions) + "\n")
    monkeypatch.setattr(runner, "PREDICTION_RESULT", prediction)
    truth_tables = _held_truth_panel(predictions)
    value = _score_result_payload(predictions, prediction, truth_tables, designation)
    methods = set(predictions["patients"][0]["predictions"])
    assert set(runner._validate_score_result(value, designation)) == methods

    forged_truth = copy.deepcopy(value)
    forged_truth["held_truth_tables"][0][0][0][0][0] += 1
    forged_truth["held_truth_tables"][0][0][0][0][1] -= 1
    forged_truth["held_truth_tables"][0][0][0][1][0] -= 1
    forged_truth["held_truth_tables"][0][0][0][1][1] += 1
    with pytest.raises(PermissionError, match="truth-table hashes"):
        runner._validate_score_result(forged_truth, designation)

    forged_hash = copy.deepcopy(value)
    forged_hash["held_truth_table_hashes"][patient_ids[0]] = "0" * 64
    with pytest.raises(PermissionError, match="truth-table hashes"):
        runner._validate_score_result(forged_hash, designation)

    forged_reduction_hash = copy.deepcopy(value)
    forged_reduction_hash["reduction_audit"][0]["truth_tables_sha256"] = "0" * 64
    with pytest.raises(PermissionError, match="reduction truth-table hash"):
        runner._validate_score_result(forged_reduction_hash, designation)

    forged_loss = copy.deepcopy(value)
    forged_loss["held_losses"]["primary"]["by_patient"][patient_ids[0]] += 0.01
    with pytest.raises(PermissionError, match="losses do not replay"):
        runner._validate_score_result(forged_loss, designation)

    forged_status = copy.deepcopy(value)
    forged_status["status"] = (
        "COMPLETED_CONFIRMATION_FAIL"
        if value["status"] == "CONFIRMATION_PASS"
        else "CONFIRMATION_PASS"
    )
    with pytest.raises(PermissionError, match="gate does not recompute"):
        runner._validate_score_result(forged_status, designation)

    forged_prediction = copy.deepcopy(predictions)
    estimate = np.asarray(
        forged_prediction["patients"][0]["predictions"]["primary"], dtype=float
    )
    estimate[0, 0] += np.asarray([[0.01, -0.01], [-0.01, 0.01]])
    forged_prediction["patients"][0]["predictions"]["primary"] = estimate.tolist()
    prediction.write_text(json.dumps(forged_prediction) + "\n")
    forged_prediction_binding = copy.deepcopy(value)
    forged_prediction_binding["prediction_result_sha256"] = runner._sha256(prediction)
    with pytest.raises(runner.ProtocolRefusal, match="FROZEN_PREDICTION_TABLE_DIFFERS"):
        runner._validate_score_result(forged_prediction_binding, designation)


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


def test_claim_source_rejects_an_older_upstream_commit_before_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = tmp_path / "calibration-selection-v3.json"
    selection.write_text(
        json.dumps(
            {
                "schema": "gse313642-hcc-calibration-selection/3.0",
                "status": "FROZEN_BEFORE_ANY_PILOT_MATRIX_REQUEST",
            }
        )
        + "\n"
    )
    authorization = tmp_path / "pilot-authorization-v3.json"
    authorization.write_text(
        json.dumps(
            {
                "schema": "gse313642-hcc-pilot-authorization/3.0",
                "status": "AUTHORIZED_AFTER_PUBLIC_CALIBRATION_FREEZE",
                "calibration_selection_sha256": runner._sha256(selection),
                "calibration_commit": "b" * 40,
                "pilot_matrix_gets_authorized": 22,
            }
        )
        + "\n"
    )
    attempt = tmp_path / "source-attempt-v3.json"
    access = tmp_path / "source-access-v3.jsonl"
    token = tmp_path / "source-capability-v3"
    monkeypatch.setattr(runner, "CALIBRATION_SELECTION", selection)
    monkeypatch.setattr(runner, "PILOT_AUTHORIZATION", authorization)
    monkeypatch.setattr(runner, "SOURCE_ATTEMPT", attempt)
    monkeypatch.setattr(runner, "SOURCE_RESULT", tmp_path / "source-result-v3.json")
    monkeypatch.setattr(runner, "SOURCE_ACCESS", access)
    monkeypatch.setattr(runner, "_require_runtime", lambda: runner.REQUIRED_RUNTIME)
    monkeypatch.setattr(
        runner,
        "_verify_implementation",
        lambda: {"implementation_commit": "a" * 40},
    )
    monkeypatch.setattr(runner, "validate", lambda stage: {"final_commit": "c" * 40})
    with pytest.raises(PermissionError, match="exact|does not bind calibration"):
        runner.claim_source(token)
    assert not token.exists()
    assert not attempt.exists()
    assert not access.exists()


def test_source_attempt_revalidates_public_calibration_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = tmp_path / "calibration.json"
    selection.write_text(
        json.dumps(
            {
                "schema": "gse313642-hcc-calibration-selection/3.0",
                "status": "FROZEN_BEFORE_ANY_PILOT_MATRIX_REQUEST",
            }
        )
        + "\n"
    )
    authorization = tmp_path / "authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "schema": "gse313642-hcc-pilot-authorization/3.0",
                "status": "AUTHORIZED_AFTER_PUBLIC_CALIBRATION_FREEZE",
                "calibration_selection_sha256": runner._sha256(selection),
                "calibration_commit": "c" * 40,
                "pilot_matrix_gets_authorized": 22,
            }
        )
        + "\n"
    )
    monkeypatch.setattr(runner, "CALIBRATION_SELECTION", selection)
    monkeypatch.setattr(runner, "PILOT_AUTHORIZATION", authorization)
    monkeypatch.setattr(runner, "validate", lambda stage: {"final_commit": "c" * 40})
    monkeypatch.setattr(runner, "_require_ancestor", lambda *args: None)
    changed = json.loads(selection.read_text())
    changed["post_attempt_mutation"] = True
    selection.write_text(json.dumps(changed) + "\n")

    with pytest.raises(PermissionError, match="authorization chain"):
        runner._revalidate_upstream_publication("source", "d" * 40, "a" * 40)


def test_score_attempt_rejects_private_state_rebinding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps(
            {
                "schema": "gse313642-hcc-predictions/3.0",
                "status": "PREDICTIONS_FROZEN_BEFORE_ANY_HELD_FB_ACCESS",
                "private_state_sha256": "f" * 64,
            }
        )
        + "\n"
    )
    authorization = tmp_path / "score-authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "schema": "gse313642-hcc-score-authorization/3.0",
                "status": "AUTHORIZED_AFTER_PUBLIC_PREDICTION_FREEZE",
                "prediction_result_sha256": runner._sha256(predictions),
                "prediction_commit": "c" * 40,
                "private_state_sha256": "0" * 64,
                "held_fb_gets_authorized": 12,
                "held_gex_gets_authorized": 0,
            }
        )
        + "\n"
    )
    monkeypatch.setattr(runner, "PREDICTION_RESULT", predictions)
    monkeypatch.setattr(runner, "SCORE_AUTHORIZATION", authorization)
    monkeypatch.setattr(runner, "validate", lambda stage: {"final_commit": "c" * 40})
    monkeypatch.setattr(runner, "_require_ancestor", lambda *args: None)
    with pytest.raises(PermissionError, match="score authorization chain"):
        runner._revalidate_upstream_publication("score", "d" * 40, "a" * 40)


@pytest.mark.parametrize("stage", ("source", "prediction", "score"))
def test_claim_rejects_a_rebound_nonpassing_predecessor_before_capability(
    stage: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predecessor = tmp_path / f"{stage}-predecessor-v3.json"
    authorization = tmp_path / f"{stage}-authorization-v3.json"
    attempt = tmp_path / f"{stage}-attempt-v3.json"
    result = tmp_path / f"{stage}-result-v3.json"
    access = tmp_path / f"{stage}-access-v3.jsonl"
    token = tmp_path / f"{stage}-capability-v3"
    predecessor_commit = "c" * 40

    if stage == "source":
        predecessor_payload = {
            "schema": "gse313642-hcc-calibration-selection/3.0",
            "status": "FROZEN_BEFORE_ANY_PILOT_MATRIX_REQUEST",
        }
        failed_status = "TERMINAL_REFUSAL"
        predecessor_name = "CALIBRATION_SELECTION"
        authorization_name = "PILOT_AUTHORIZATION"
        attempt_name, result_name, access_name = (
            "SOURCE_ATTEMPT",
            "SOURCE_RESULT",
            "SOURCE_ACCESS",
        )
        authorization_payload = {
            "schema": "gse313642-hcc-pilot-authorization/3.0",
            "status": "AUTHORIZED_AFTER_PUBLIC_CALIBRATION_FREEZE",
            "calibration_commit": predecessor_commit,
            "pilot_matrix_gets_authorized": 22,
        }
        digest_name = "calibration_selection_sha256"
    elif stage == "prediction":
        predecessor_payload = {
            "schema": "gse313642-hcc-source-result/3.0",
            "status": "SOURCE_PASS_REFIT_21",
        }
        failed_status = "TERMINAL_SOURCE_GATE_FAIL"
        predecessor_name = "SOURCE_RESULT"
        authorization_name = "PREDICTION_AUTHORIZATION"
        attempt_name, result_name, access_name = (
            "PREDICTION_ATTEMPT",
            "PREDICTION_RESULT",
            "PREDICTION_ACCESS",
        )
        authorization_payload = {
            "schema": "gse313642-hcc-prediction-authorization/3.0",
            "status": "AUTHORIZED_AFTER_PUBLIC_SOURCE_PASS",
            "source_commit": predecessor_commit,
            "held_gex_gets_authorized": 12,
            "held_fb_gets_authorized": 0,
        }
        digest_name = "source_result_sha256"
    else:
        predecessor_payload = {
            "schema": "gse313642-hcc-predictions/3.0",
            "status": "PREDICTIONS_FROZEN_BEFORE_ANY_HELD_FB_ACCESS",
        }
        failed_status = "TERMINAL_REFUSAL"
        predecessor_name = "PREDICTION_RESULT"
        authorization_name = "SCORE_AUTHORIZATION"
        attempt_name, result_name, access_name = (
            "SCORE_ATTEMPT",
            "SCORE_RESULT",
            "SCORE_ACCESS",
        )
        authorization_payload = {
            "schema": "gse313642-hcc-score-authorization/3.0",
            "status": "AUTHORIZED_AFTER_PUBLIC_PREDICTION_FREEZE",
            "prediction_commit": predecessor_commit,
            "held_fb_gets_authorized": 12,
            "held_gex_gets_authorized": 0,
        }
        digest_name = "prediction_result_sha256"

    predecessor.write_text(json.dumps(predecessor_payload) + "\n")
    predecessor_payload["status"] = failed_status
    predecessor.write_text(json.dumps(predecessor_payload) + "\n")
    authorization_payload[digest_name] = runner._sha256(predecessor)
    authorization.write_text(json.dumps(authorization_payload) + "\n")

    monkeypatch.setattr(runner, predecessor_name, predecessor)
    monkeypatch.setattr(runner, authorization_name, authorization)
    monkeypatch.setattr(runner, attempt_name, attempt)
    monkeypatch.setattr(runner, result_name, result)
    monkeypatch.setattr(runner, access_name, access)
    monkeypatch.setattr(runner, "_require_runtime", lambda: runner.REQUIRED_RUNTIME)
    monkeypatch.setattr(
        runner,
        "validate",
        lambda current: (_ for _ in ()).throw(AssertionError("unexpected validation")),
    )
    monkeypatch.setattr(
        runner,
        "_require_public_tag",
        lambda *args: (_ for _ in ()).throw(AssertionError("unexpected tag check")),
    )
    monkeypatch.setattr(
        runner,
        "_verify_implementation",
        lambda: {"implementation_commit": "a" * 40},
    )

    with pytest.raises(PermissionError, match="authorization differs"):
        getattr(runner, f"claim_{stage}")(token)
    assert authorization_payload[digest_name] == runner._sha256(predecessor)
    assert not token.exists()
    assert not attempt.exists()
    assert not access.exists()


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


def test_matrix_get_has_one_attempt_exact_size_and_no_redirect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = runner._designation()["patients"][0]
    body = b"abcd"
    file_record = {
        "allowed_stage": "prediction",
        "expected_bytes": len(body),
        "v3_get_authorized": True,
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
    assert "acquire-axes" not in choices
    assert "run-held" not in choices


def test_validation_requires_exact_unique_get_and_delete_sets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    consumption = tmp_path / "consumption.json"
    result = tmp_path / "predictions.json"
    source = tmp_path / "source.json"
    source.write_text("{}\n")
    access = tmp_path / "access.jsonl"
    attempt = tmp_path / "attempt.json"
    attempt.write_text("{}\n")
    consumption.write_text(
        json.dumps(
            {
                "schema": "gse313642-hcc-prediction-consumption/3.0",
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
    monkeypatch.setattr(runner, "PREDICTION_CONSUMPTION", consumption)
    monkeypatch.setattr(runner, "PREDICTION_RESULT", result)
    monkeypatch.setattr(runner, "SOURCE_RESULT", source)
    monkeypatch.setattr(runner, "PREDICTION_ACCESS", access)
    monkeypatch.setattr(runner, "PREDICTION_ATTEMPT", attempt)
    monkeypatch.setattr(runner, "_require_public_attempt", lambda stage: "a" * 40)
    monkeypatch.setattr(runner, "_require_public_tag", lambda *args: "a" * 40)
    monkeypatch.setattr(runner, "_require_ancestor", lambda *args: None)
    monkeypatch.setattr(runner, "_validate_source_result", lambda *args: {})
    monkeypatch.setattr(runner, "_validate_public_predictions", lambda *args: None)
    rows = [{"event": "OPENED_BEFORE_MATRIX_ACCESS"}]
    by_id = {
        record["patient_id"]: record for record in runner._designation()["patients"]
    }
    evidence = []
    for patient in runner._designation()["role_order"]["held"]:
        record = by_id[patient]
        manifest = runner._manifest_file(record, "GEX", runner.MATRIX_MEMBER)
        audit = _matrix_audit()
        audit["compressed_bytes"] = manifest["expected_bytes"]
        audit["compressed_sha256"] = "0" * 64
        evidence.append(
            {
                "gex_download": {
                    "bytes": manifest["expected_bytes"],
                    "sha256": "0" * 64,
                },
                "gex_matrix_market": audit,
            }
        )
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
                "event": "MATRIX_PARSE_STARTED",
                "patient_id": patient,
                "modality": "GEX",
                "allow_integral_real": True,
            }
        )
        rows.append(
            {
                "stage": "prediction",
                "event": "MATRIX_PARSE_FINISHED",
                "patient_id": patient,
                "modality": "GEX",
                "parser_audit": audit,
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
    result.write_text(
        json.dumps(
            {
                "status": "PREDICTIONS_FROZEN_BEFORE_ANY_HELD_FB_ACCESS",
                "rerun_permitted": False,
                "patients": evidence,
            }
        )
        + "\n"
    )
    access.write_text("".join(json.dumps(row) + "\n" for row in rows))
    validated = runner.validate("prediction")
    assert validated["completed_matrix_gets"] == 12
    assert validated["completed_matrix_parses"] == 12
    intact_result = result.read_text()
    changed_result = json.loads(intact_result)
    changed_result["patients"][0]["gex_matrix_market"]["compressed_sha256"] = "1" * 64
    result.write_text(json.dumps(changed_result) + "\n")
    with pytest.raises(PermissionError, match="download, parser, and result"):
        runner.validate("prediction")
    result.write_text(intact_result)
    rows.insert(5, copy.deepcopy(rows[2]))
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
                "schema": "gse313642-hcc-prediction-consumption/3.0",
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
            audit = _matrix_audit()
            audit["compressed_bytes"] = manifest["expected_bytes"]
            audit["compressed_sha256"] = "0" * 64
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
                    "event": "MATRIX_PARSE_STARTED",
                    "patient_id": patient,
                    "modality": "GEX",
                    "allow_integral_real": True,
                }
            )
            rows.append(
                {
                    "stage": "prediction",
                    "event": "MATRIX_PARSE_FINISHED",
                    "patient_id": patient,
                    "modality": "GEX",
                    "parser_audit": audit,
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
    result.write_text(
        json.dumps(
            {
                "schema": "gse313642-hcc-prediction-result/3.0",
                "status": "TERMINAL_REFUSAL",
                "refusal_code": "MATRIX_RESPONSE_OR_REDIRECT_REFUSAL",
                "attempt_sha256": runner._sha256(attempt),
                "consumption_sha256": runner._sha256(consumption),
                "access_journal": access.name,
                "access_journal_sha256": runner._sha256(access),
                "rerun_permitted": False,
            }
        )
        + "\n"
    )
    assert runner.validate("prediction")["completed_matrix_gets"] == 1

    missing_outcome = [
        row for row in rows if row.get("event") != "GET_FAILED_TERMINALLY"
    ]
    access.write_text("".join(json.dumps(row) + "\n" for row in missing_outcome))
    terminal = json.loads(result.read_text())
    terminal["access_journal_sha256"] = runner._sha256(access)
    result.write_text(json.dumps(terminal) + "\n")
    with pytest.raises(PermissionError, match="unique allowed prefix"):
        runner.validate("prediction")

    parse_failure_rows = copy.deepcopy(rows[:6])
    first_manifest = runner._manifest_file(by_id[first], "GEX", runner.MATRIX_MEMBER)
    parse_failure_rows[4] = {
        "stage": "prediction",
        "event": "MATRIX_PARSE_FAILED",
        "patient_id": first,
        "modality": "GEX",
        "exception_class": "GzipMatrixMarketValidationError",
        "refusal_code": "MATRIX_PARSE_VALIDATION_REFUSAL",
        "message": "frozen parser refusal",
        "partial_audit": {
            "declared_nnz": 1,
            "parsed_nnz": 0,
            "compressed_bytes": first_manifest["expected_bytes"],
            "compressed_sha256": "0" * 64,
            "compressed_source_exhausted": True,
            "decompressed_bytes": 64,
            "decompressed_sha256": "d" * 64,
            "gzip_stream_exhausted": False,
        },
    }
    access.write_text("".join(json.dumps(row) + "\n" for row in parse_failure_rows))
    terminal["refusal_code"] = "MATRIX_PARSE_VALIDATION_REFUSAL"
    terminal["access_journal_sha256"] = runner._sha256(access)
    result.write_text(json.dumps(terminal) + "\n")
    assert runner.validate("prediction")["completed_matrix_gets"] == 1

    parse_failure_rows[4]["partial_audit"]["compressed_sha256"] = "1" * 64
    access.write_text("".join(json.dumps(row) + "\n" for row in parse_failure_rows))
    terminal["access_journal_sha256"] = runner._sha256(access)
    result.write_text(json.dumps(terminal) + "\n")
    with pytest.raises(PermissionError, match="partial audit differs from"):
        runner.validate("prediction")

    parse_failure_rows[4]["partial_audit"].update(
        {
            "compressed_source_exhausted": False,
            "compressed_bytes": first_manifest["expected_bytes"],
        }
    )
    access.write_text("".join(json.dumps(row) + "\n" for row in parse_failure_rows))
    terminal["access_journal_sha256"] = runner._sha256(access)
    result.write_text(json.dumps(terminal) + "\n")
    with pytest.raises(PermissionError, match="partial audit differs from"):
        runner.validate("prediction")

    parse_failure_rows[4]["partial_audit"].update(
        {
            "compressed_sha256": "0" * 64,
            "compressed_source_exhausted": False,
            "compressed_bytes": first_manifest["expected_bytes"] + 1,
        }
    )
    access.write_text("".join(json.dumps(row) + "\n" for row in parse_failure_rows))
    terminal["access_journal_sha256"] = runner._sha256(access)
    result.write_text(json.dumps(terminal) + "\n")
    with pytest.raises(PermissionError, match="partial audit differs from"):
        runner.validate("prediction")

    access.write_text("".join(json.dumps(row) + "\n" for row in rows))
    terminal["refusal_code"] = "MATRIX_RESPONSE_OR_REDIRECT_REFUSAL"
    terminal["access_journal_sha256"] = runner._sha256(access)
    result.write_text(json.dumps(terminal) + "\n")
    rows[-2]["patient_id"] = designation["role_order"]["calibration"][0]
    access.write_text("".join(json.dumps(row) + "\n" for row in rows))
    terminal = json.loads(result.read_text())
    terminal["access_journal_sha256"] = runner._sha256(access)
    result.write_text(json.dumps(terminal) + "\n")
    with pytest.raises(PermissionError, match="unique allowed prefix"):
        runner.validate("prediction")


@pytest.mark.parametrize(
    ("tail", "expected_code"),
    (
        ((), "CRASH_RECOVERY"),
        (("GET_STARTED",), "CRASH_RECOVERY"),
        (("GET_STARTED", "GET_FAILED_TERMINALLY"), "CRASH_RECOVERY"),
        (("GET_STARTED", "GET_COMPLETED"), "CRASH_RECOVERY"),
        (
            ("GET_STARTED", "GET_COMPLETED", "MATRIX_PARSE_STARTED"),
            "CRASH_RECOVERY",
        ),
        (
            (
                "GET_STARTED",
                "GET_COMPLETED",
                "MATRIX_PARSE_STARTED",
                "MATRIX_PARSE_FINISHED",
            ),
            "CRASH_RECOVERY",
        ),
        (
            (
                "GET_STARTED",
                "GET_COMPLETED",
                "MATRIX_PARSE_STARTED",
                "MATRIX_PARSE_FAILED",
            ),
            "MATRIX_PARSE_VALIDATION_REFUSAL",
        ),
        (
            (
                "GET_STARTED",
                "GET_COMPLETED",
                "MATRIX_PARSE_STARTED",
                "MATRIX_PARSE_FINISHED",
                "MATRIX_DELETED",
            ),
            "CRASH_RECOVERY",
        ),
    ),
    ids=(
        "before-get",
        "during-get",
        "after-get-failure",
        "before-parse",
        "during-parse",
        "after-parse",
        "after-parse-refusal",
        "after-deletion",
    ),
)
def test_prediction_recovery_closes_every_durable_boundary_and_validates(
    tail: tuple[str, ...],
    expected_code: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = tmp_path / "scratch"
    state = tmp_path / "private-state.json"
    state.write_text("private\n")
    attempt = tmp_path / "attempt.json"
    attempt.write_text("{}\n")
    consumption = tmp_path / "consumption.json"
    result = tmp_path / "result.json"
    access = tmp_path / "access.jsonl"
    consumption.write_text(
        json.dumps(
            {
                "schema": "gse313642-hcc-prediction-consumption/3.0",
                "status": "CONSUMED_BEFORE_FIRST_MATRIX_REQUEST",
                "attempt_sha256": runner._sha256(attempt),
                "scratch_identity_sha256": hashlib.sha256(
                    str(scratch.resolve()).encode()
                ).hexdigest(),
                "private_state_identity_sha256": hashlib.sha256(
                    str(state.resolve()).encode()
                ).hexdigest(),
                "private_state_expected_absent_before_prediction": True,
                "rerun_permitted": False,
            }
        )
        + "\n"
    )
    designation = runner._designation()
    patient_id = designation["role_order"]["held"][0]
    record = next(
        record
        for record in designation["patients"]
        if record["patient_id"] == patient_id
    )
    manifest = runner._manifest_file(record, "GEX", runner.MATRIX_MEMBER)
    common = {
        "stage": "prediction",
        "patient_id": patient_id,
        "modality": "GEX",
    }
    completed_audit = _matrix_audit()
    completed_audit["compressed_bytes"] = manifest["expected_bytes"]
    completed_audit["compressed_sha256"] = "0" * 64
    event_payloads = {
        "GET_STARTED": {
            **common,
            "event": "GET_STARTED",
            "url": runner._url(record, "GEX", runner.MATRIX_MEMBER),
            "expected_bytes": manifest["expected_bytes"],
        },
        "GET_COMPLETED": {
            **common,
            "event": "GET_COMPLETED",
            "bytes": manifest["expected_bytes"],
            "sha256": "0" * 64,
        },
        "GET_FAILED_TERMINALLY": {
            **common,
            "event": "GET_FAILED_TERMINALLY",
        },
        "MATRIX_PARSE_STARTED": {
            **common,
            "event": "MATRIX_PARSE_STARTED",
            "allow_integral_real": True,
        },
        "MATRIX_PARSE_FINISHED": {
            **common,
            "event": "MATRIX_PARSE_FINISHED",
            "parser_audit": completed_audit,
        },
        "MATRIX_PARSE_FAILED": {
            **common,
            "event": "MATRIX_PARSE_FAILED",
            "exception_class": "GzipMatrixMarketValidationError",
            "refusal_code": "MATRIX_PARSE_VALIDATION_REFUSAL",
            "message": "frozen parser refusal",
            "partial_audit": None,
        },
        "MATRIX_DELETED": {
            **common,
            "event": "MATRIX_DELETED",
            "body_existed": True,
        },
    }
    rows = [{"event": "OPENED_BEFORE_MATRIX_ACCESS"}]
    rows.extend(copy.deepcopy(event_payloads[event]) for event in tail)
    access.write_text("".join(json.dumps(row) + "\n" for row in rows))
    if tail and tail[-1] != "MATRIX_DELETED":
        matrix = (
            scratch
            / "prediction"
            / patient_id
            / runner._filename(record, "GEX", runner.MATRIX_MEMBER)
        )
        matrix.parent.mkdir(parents=True)
        matrix.write_text("matrix bytes")
    monkeypatch.setattr(runner, "PREDICTION_ATTEMPT", attempt)
    monkeypatch.setattr(runner, "PREDICTION_CONSUMPTION", consumption)
    monkeypatch.setattr(runner, "PREDICTION_RESULT", result)
    monkeypatch.setattr(runner, "PREDICTION_ACCESS", access)
    monkeypatch.setattr(runner, "_require_public_attempt", lambda stage: "a" * 40)
    monkeypatch.setattr(runner, "_require_public_tag", lambda *args: "b" * 40)
    monkeypatch.setattr(runner, "_require_ancestor", lambda *args: None)

    recovered = runner.recover("prediction", scratch=scratch, state_path=state)
    recovered_events = runner._read_jsonl(access)
    assert recovered["status"] == "TERMINAL_REFUSAL"
    assert recovered["refusal_code"] == expected_code
    assert recovered["access_journal_sha256"] == runner._sha256(access)
    assert not state.exists()
    assert not (scratch / "prediction").exists()
    assert sum(
        event.get("event") == "MATRIX_DELETED" for event in recovered_events
    ) == (1 if tail else 0)
    if "MATRIX_PARSE_FAILED" in tail:
        refusal = next(
            event
            for event in recovered_events
            if event.get("event") == "MATRIX_PARSE_FAILED"
        )
        assert refusal == event_payloads["MATRIX_PARSE_FAILED"]
    validation = runner.validate("prediction")
    assert validation["valid"] is True
    assert validation["stage_passed"] is False

    event_count = len(recovered_events)
    assert runner.recover("prediction", scratch=scratch, state_path=state) == recovered
    assert len(runner._read_jsonl(access)) == event_count

    recovered_refusals = [
        event
        for event in recovered_events
        if event.get("event") == "MATRIX_PARSE_FAILED"
        and event.get("recovered_after_crash") is True
    ]
    if recovered_refusals:
        assert recovered_refusals == [
            {
                **common,
                "event": "MATRIX_PARSE_FAILED",
                "exception_class": "CrashRecovery",
                "refusal_code": "CRASH_RECOVERY",
                "message": runner.CRASH_PARSE_MESSAGE,
                "partial_audit": None,
                "recovered_after_crash": True,
            }
        ]
        recovered_refusals[0]["message"] = "tampered recovery reason"
        access.write_text(
            "".join(json.dumps(event) + "\n" for event in recovered_events)
        )
        terminal = json.loads(result.read_text())
        terminal["access_journal_sha256"] = runner._sha256(access)
        result.write_text(json.dumps(terminal) + "\n")
        with pytest.raises(PermissionError, match="parse failure diagnostic"):
            runner.validate("prediction")


def test_accidental_second_prediction_run_preserves_prior_private_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    state = tmp_path / "private-state.json"
    state.write_text("prior successful state\n")
    attempt = tmp_path / "attempt.json"
    attempt.write_text(json.dumps({"implementation_commit": "a" * 40}) + "\n")
    consumption = tmp_path / "consumption.json"
    consumption.write_text("{}\n")
    result = tmp_path / "result.json"
    monkeypatch.setattr(runner, "PREDICTION_CONSUMPTION", consumption)
    monkeypatch.setattr(runner, "PREDICTION_RESULT", result)
    monkeypatch.setattr(runner, "PREDICTION_ATTEMPT", attempt)
    monkeypatch.setattr(runner, "_require_runtime", lambda: runner.REQUIRED_RUNTIME)
    monkeypatch.setattr(runner, "_require_public_attempt", lambda stage: "a" * 40)
    with pytest.raises(FileExistsError, match="private held GEX state"):
        runner.run_prediction(
            tmp_path / "unused-token", tmp_path / "unused-axes", state, scratch=scratch
        )
    assert state.read_text() == "prior successful state\n"
    assert not result.exists()


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
            {
                "claim_token_sha256": hashlib.sha256(token.read_bytes()).hexdigest(),
                "implementation_commit": "a" * 40,
            }
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
    with pytest.raises(FileExistsError):
        runner.run_score(token, tmp_path / "unused-axes", state, scratch=scratch)
    assert state.read_text() == "prior successful state\n"
    assert not result.exists()


def test_recovery_records_and_closes_one_torn_journal_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    state = tmp_path / "private-state.json"
    state.write_text("partial state\n")
    attempt = tmp_path / "attempt.json"
    attempt.write_text("{}\n")
    consumption = tmp_path / "consumption.json"
    consumption.write_text(
        json.dumps(
            {
                "schema": "gse313642-hcc-prediction-consumption/3.0",
                "status": "CONSUMED_BEFORE_FIRST_MATRIX_REQUEST",
                "attempt_sha256": runner._sha256(attempt),
                "scratch_identity_sha256": hashlib.sha256(
                    str(scratch.resolve()).encode()
                ).hexdigest(),
                "private_state_identity_sha256": hashlib.sha256(
                    str(state.resolve()).encode()
                ).hexdigest(),
                "private_state_expected_absent_before_prediction": True,
                "rerun_permitted": False,
            }
        )
        + "\n"
    )
    access = tmp_path / "access.jsonl"
    header = json.dumps({"event": "OPENED_BEFORE_MATRIX_ACCESS"}).encode() + b"\n"
    torn = b'{"stage":"prediction","event":"GET_STARTED"'
    access.write_bytes(header + torn)
    result = tmp_path / "result.json"
    monkeypatch.setattr(runner, "PREDICTION_ATTEMPT", attempt)
    monkeypatch.setattr(runner, "PREDICTION_CONSUMPTION", consumption)
    monkeypatch.setattr(runner, "PREDICTION_ACCESS", access)
    monkeypatch.setattr(runner, "PREDICTION_RESULT", result)
    monkeypatch.setattr(runner, "_require_public_attempt", lambda stage: "a" * 40)
    monkeypatch.setattr(runner, "_require_public_tag", lambda *args: "b" * 40)
    monkeypatch.setattr(runner, "_require_ancestor", lambda *args: None)

    recovered = runner.recover("prediction", scratch=scratch, state_path=state)
    events = runner._read_jsonl(access)
    repair = events[1]
    assert recovered["refusal_code"] == "CRASH_RECOVERY"
    assert repair == {
        "stage": "prediction",
        "event": "JOURNAL_TORN_TAIL_RECOVERED",
        "torn_bytes": len(torn),
        "torn_sha256": hashlib.sha256(torn).hexdigest(),
        "recovered_after_crash": True,
    }
    validation = runner.validate("prediction")
    assert validation["valid"] is True
    assert validation["stage_passed"] is False


def test_recovery_cannot_race_a_live_stage(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    with runner._stage_lock(scratch, "prediction"):
        with pytest.raises(PermissionError, match="already running"):
            runner.recover(
                "prediction",
                scratch=scratch,
                state_path=tmp_path / "state.json",
            )


def test_recovery_lock_uses_canonical_scratch_path(tmp_path: Path) -> None:
    canonical_parent = tmp_path / "canonical"
    canonical_parent.mkdir()
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(canonical_parent, target_is_directory=True)
    canonical_scratch = canonical_parent / "scratch"
    alias_scratch = alias_parent / "scratch"

    with runner._stage_lock(alias_scratch, "prediction"):
        with pytest.raises(PermissionError, match="already running"):
            runner.recover(
                "prediction",
                scratch=canonical_scratch,
                state_path=tmp_path / "state.json",
            )


def test_prediction_recovery_preserves_successful_private_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    state = tmp_path / "private-state.json"
    state.write_text("frozen prediction state\n")
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
    result.write_text(
        json.dumps({"status": "PREDICTIONS_FROZEN_BEFORE_ANY_HELD_FB_ACCESS"}) + "\n"
    )
    access = tmp_path / "access.jsonl"
    access.write_text(json.dumps({"event": "OPENED_BEFORE_MATRIX_ACCESS"}) + "\n")
    monkeypatch.setattr(runner, "PREDICTION_CONSUMPTION", consumption)
    monkeypatch.setattr(runner, "PREDICTION_RESULT", result)
    monkeypatch.setattr(runner, "PREDICTION_ACCESS", access)

    recovered = runner.recover("prediction", scratch=scratch, state_path=state)

    assert recovered["status"] == "PREDICTIONS_FROZEN_BEFORE_ANY_HELD_FB_ACCESS"
    assert state.read_text() == "frozen prediction state\n"


@pytest.mark.parametrize("stage", ("calibration", "source", "prediction", "score"))
def test_preconsumption_failure_leaves_stage_nonterminal(
    stage: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = tmp_path / f"{stage}-result.json"
    consumption = tmp_path / f"{stage}-consumption.json"
    monkeypatch.setattr(
        runner,
        {
            "calibration": "CALIBRATION_SELECTION",
            "source": "SOURCE_RESULT",
            "prediction": "PREDICTION_RESULT",
            "score": "SCORE_RESULT",
        }[stage],
        result,
    )
    monkeypatch.setattr(
        runner,
        f"{stage.upper()}_CONSUMPTION",
        consumption,
    )
    monkeypatch.setattr(runner, "_require_runtime", lambda: runner.REQUIRED_RUNTIME)
    monkeypatch.setattr(
        runner,
        "_require_public_attempt",
        lambda current: (_ for _ in ()).throw(PermissionError("attempt unavailable")),
    )
    arguments = [tmp_path / "token", tmp_path / "axes"]
    if stage in {"prediction", "score"}:
        arguments.append(tmp_path / "private-state.json")

    with pytest.raises(PermissionError, match="attempt unavailable"):
        getattr(runner, f"run_{stage}")(*arguments, scratch=tmp_path / "scratch")
    assert not consumption.exists()
    assert not result.exists()


def test_public_attempt_rejects_changed_frozen_implementation_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "_verify_implementation",
        lambda: (_ for _ in ()).throw(PermissionError("implementation bytes changed")),
    )
    with pytest.raises(PermissionError, match="implementation bytes changed"):
        runner._require_public_attempt("calibration")


def test_failure_refuses_to_publish_without_consumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "attempt.json"
    attempt.write_text("{}\n")
    access = tmp_path / "access.jsonl"
    access.write_text(json.dumps({"event": "OPENED_BEFORE_MATRIX_ACCESS"}) + "\n")
    result = tmp_path / "result.json"
    monkeypatch.setattr(runner, "CALIBRATION_ATTEMPT", attempt)
    monkeypatch.setattr(runner, "CALIBRATION_CONSUMPTION", tmp_path / "missing.json")
    monkeypatch.setattr(runner, "CALIBRATION_ACCESS", access)
    monkeypatch.setattr(runner, "CALIBRATION_SELECTION", result)

    with pytest.raises(PermissionError, match="requires a consumed stage"):
        runner._failure("calibration", "CALIBRATION_EXECUTION_FAILURE")
    assert not result.exists()


def test_consumption_requires_header_only_and_absent_prior_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = tmp_path / "token"
    token.write_bytes(b"one-shot capability")
    attempt = tmp_path / "attempt.json"
    created = "2026-08-30T00:00:00Z"
    attempt.write_text(
        json.dumps(
            {
                "created_at_utc": created,
                "runtime": runner.REQUIRED_RUNTIME,
                "claim_token_sha256": hashlib.sha256(token.read_bytes()).hexdigest(),
            }
        )
        + "\n"
    )
    access = tmp_path / "access.jsonl"
    header = runner._access_header("calibration", created, runner.REQUIRED_RUNTIME)
    access.write_text(
        json.dumps(header) + "\n" + json.dumps({"event": "GET_STARTED"}) + "\n"
    )
    consumption = tmp_path / "consumption.json"
    result = tmp_path / "result.json"
    monkeypatch.setattr(runner, "CALIBRATION_ATTEMPT", attempt)
    monkeypatch.setattr(runner, "CALIBRATION_ACCESS", access)
    monkeypatch.setattr(runner, "CALIBRATION_CONSUMPTION", consumption)
    monkeypatch.setattr(runner, "CALIBRATION_SELECTION", result)

    with pytest.raises(PermissionError, match="header only"):
        runner._consume("calibration", token, tmp_path / "scratch")
    assert token.exists()
    assert not consumption.exists()

    access.write_text(json.dumps(header) + "\n")
    result.write_text("{}\n")
    with pytest.raises(FileExistsError, match="already consumed or completed"):
        runner._consume("calibration", token, tmp_path / "scratch")
    assert token.exists()
    assert not consumption.exists()


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
    access.write_text(json.dumps({"event": "OPENED_BEFORE_MATRIX_ACCESS"}) + "\n")
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


def test_equal_publication_commits_are_not_sequential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "_git",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("git called")),
    )
    with pytest.raises(PermissionError, match="required ancestry"):
        runner._require_ancestor("a" * 40, "a" * 40)


def test_numeric_runtime_requires_single_thread_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in runner.THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")
    assert runner._require_runtime() == runner.REQUIRED_RUNTIME
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(PermissionError, match="runtime differs"):
        runner._require_runtime()
