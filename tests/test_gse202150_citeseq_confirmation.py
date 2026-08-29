from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import inspect
import json
from pathlib import Path
import threading

import h5py
import numpy as np
import pytest

from experiments import confirm_gse202150_citeseq as confirmation


def _schema(*, hto_count: int = 7, include_total_cd45: bool = False):
    genes = list(confirmation.RNA_SYMBOLS)
    genes += ["MT-SYNTHETIC"]
    genes += [f"GENE{index}" for index in range(201 - len(genes))]
    custom = list(confirmation.ADT_TARGETS)
    if include_total_cd45:
        custom.append("CD45")
    custom += [f"HTO-{index}" for index in range(1, hto_count + 1)]
    names = genes + custom
    return {
        "id": [f"ID{index}" for index in range(len(names))],
        "name": names,
        "feature_type": ["Gene Expression"] * len(genes)
        + ["Custom"] * len(custom),
    }


def _write_h5(path: Path, *, poison_adt: bool = False) -> None:
    schema = _schema()
    names = schema["name"]
    rows = []
    values = []
    for index, feature_type in enumerate(schema["feature_type"]):
        if feature_type == "Gene Expression":
            rows.append(index)
            values.append(1.0)
    for target in confirmation.ADT_TARGETS:
        rows.append(names.index(target, 201))
        values.append(np.nan if poison_adt else 2.0)
    for tag, value in (("HTO-1", 11.0), ("HTO-2", 7.0), ("HTO-3", 1.0)):
        rows.append(names.index(tag, 201))
        values.append(value)
    order = np.argsort(rows)
    rows = np.asarray(rows, dtype=np.int64)[order]
    values = np.asarray(values, dtype=float)[order]
    string = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as handle:
        matrix = handle.create_group("matrix")
        features = matrix.create_group("features")
        for key, axis in schema.items():
            features.create_dataset(key, data=np.asarray(axis, dtype=string))
        matrix.create_dataset("barcodes", data=np.asarray(["CELL-1"], dtype=string))
        matrix.create_dataset("indices", data=rows)
        matrix.create_dataset("data", data=values)
        matrix.create_dataset("indptr", data=np.asarray([0, len(rows)], dtype=np.int64))
        matrix.create_dataset("shape", data=np.asarray([len(names), 1], dtype=np.int64))


def _official():
    return {
        "assignments": {
            "IOF1": {},
            "IOF2": {},
            "IOF3": {frozenset(("HTO-1", "HTO-2")): "ACUTE_SAMPLE"},
            "IOF4": {},
        },
        "records": {
            "ACUTE_SAMPLE": {
                "sample": "ACUTE_SAMPLE",
                "subject": "SUBJECT-1",
                "pathogen": "WNV",
                "timepoint": "acute",
                "batches": ["IOF3"],
                "libraries": ["ADRA03_IOF3_1"],
                "hto_tags": ["HTO-1", "HTO-2"],
            }
        },
    }


def test_candidate_has_33_disjoint_physical_subjects_and_frozen_8_8_8_9_split():
    candidate = confirmation._candidate()
    rows = candidate["acute_samples"]
    assert len(rows) == len({row["subject"] for row in rows}) == 33
    assert {
        batch: sum(row["batch"] == batch for row in rows)
        for batch in confirmation.BATCHES
    } == {"IOF1": 8, "IOF2": 8, "IOF3": 8, "IOF4": 9}
    assert all(row["role"] == confirmation.ROLE_BY_BATCH[row["batch"]] for row in rows)
    assert not ({"HD105", "HD108"} & {row["subject"] for row in rows})
    assert candidate["exclusions"]["longitudinal_nonacute_samples"] == 45


def test_panel_has_exactly_13_unique_cognates_and_does_not_substitute_cd45ra():
    candidate = confirmation._candidate()
    assert len(candidate["panel"]) == 13
    assert len({row["rna_symbol"] for row in candidate["panel"]}) == 13
    assert len({row["adt_target"] for row in candidate["panel"]}) == 13
    assert "PTPRC" not in confirmation.RNA_SYMBOLS
    assert "CD45RA" not in confirmation.ADT_TARGETS
    resolved = confirmation._resolve_feature_panel(_schema())
    assert len(resolved["rna"]) == len(resolved["adt"]) == 13
    with pytest.raises(PermissionError, match="total-CD45"):
        confirmation._resolve_feature_panel(_schema(include_total_cd45=True))


def test_manifest_binds_19_h5_names_urls_and_byte_counts():
    manifest = confirmation._manifest()
    files = manifest["h5_files"]
    assert len(files) == len({row["filename"] for row in files}) == 19
    assert all(row["url"].startswith("https://ftp.ncbi.nlm.nih.gov/") for row in files)
    assert all(isinstance(row["bytes"], int) and row["bytes"] > 0 for row in files)
    assert all(len(row["sha256"]) == 64 for row in files)
    assert {
        batch: sum(row["batch"] == batch for row in files)
        for batch in confirmation.BATCHES
    } == {"IOF1": 5, "IOF2": 5, "IOF3": 5, "IOF4": 4}
    assert all(len(row["sha256"]) == 64 for row in manifest["metadata"].values())


def test_feature_schema_reader_never_opens_poison_matrix_payload(tmp_path: Path):
    path = tmp_path / "schema.h5"
    schema = _schema()
    string = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as handle:
        matrix = handle.create_group("matrix")
        features = matrix.create_group("features")
        for key, values in schema.items():
            features.create_dataset(key, data=np.asarray(values, dtype=string))
        matrix["data"] = h5py.ExternalLink("poison.h5", "/data")
        matrix["indices"] = h5py.ExternalLink("poison.h5", "/indices")
        matrix["indptr"] = h5py.ExternalLink("poison.h5", "/indptr")
        matrix["barcodes"] = h5py.ExternalLink("poison.h5", "/barcodes")
        matrix["shape"] = h5py.ExternalLink("poison.h5", "/shape")
    assert confirmation._feature_schema(path) == schema


def test_public_payload_allows_hdf5_identifiers_but_rejects_local_paths():
    confirmation._validate_public_payload(
        {"preflight_runner_datasets_opened": ["/matrix/features/id"]}
    )
    with pytest.raises(PermissionError, match="local path"):
        confirmation._validate_public_payload({"artifact": "/tmp/private.json"})


def test_hto_demultiplexing_requires_two_positive_uniquely_ranked_official_tags():
    rows = {"HTO-1": 1, "HTO-2": 2, "HTO-3": 3}
    assignments = {frozenset(("HTO-1", "HTO-2")): "SAMPLE"}
    assert confirmation._demultiplex({1: 9, 2: 4, 3: 1}, rows, assignments) == "SAMPLE"
    assert confirmation._demultiplex({1: 9, 2: 4, 3: 4}, rows, assignments) is None
    assert confirmation._demultiplex({1: 9, 2: 9, 3: 1}, rows, assignments) is None
    assert confirmation._demultiplex({1: 9, 2: 0, 3: 0}, rows, assignments) is None
    assert confirmation._demultiplex({1: 9, 2: 3, 3: 1}, rows, {}) is None


def test_official_hto_unique_name_maps_exactly_to_matrix_feature_name():
    assert confirmation._hto_matrix_name("HTO-10_h-3-IH-A") == "HTO-10"
    for invalid in ("HTO-10", "HTO-10_wrong", "HASH-10_h-3-IH-A"):
        with pytest.raises(PermissionError, match="mapped exactly"):
            confirmation._hto_matrix_name(invalid)


def test_held_rna_scan_reads_only_rna_and_hto_positions_not_poison_adt(tmp_path: Path):
    path = tmp_path / "held.h5"
    _write_h5(path, poison_adt=True)
    records, audit = confirmation._scan_library(
        path,
        "ADRA03_IOF3_1",
        _official(),
        modalities=frozenset(("rna",)),
    )
    assert len(records) == 1
    assert records[0]["rna_state"].tolist() == [1] * 13
    assert audit["rna_values_read"] == 201
    assert audit["adt_values_read"] == 0
    assert not audit["co_resident_adt_values_decoded"]
    with pytest.raises(confirmation.ProtocolRefusal, match="ADT_COUNTS_INVALID"):
        confirmation._scan_library(
            path,
            "ADRA03_IOF3_1",
            _official(),
            modalities=frozenset(("adt",)),
        )


def test_rna_firewall_uses_sparse_positions_not_whole_mixed_columns():
    source = inspect.getsource(confirmation._scan_library)
    assert "_column_values" in source
    assert "data[start:stop]" not in source
    assert "modalities=frozenset((\"rna\",))" in inspect.getsource(
        confirmation._prediction_body
    )


def test_composite_cell_selection_is_deterministic_and_never_splits_subject(monkeypatch):
    monkeypatch.setattr(confirmation, "CELL_BUDGET", 4)
    records = [
        {"cell_id": f"LIB-{library}|BC-{index}", "subject": "S1"}
        for library in (1, 2)
        for index in range(4)
    ]
    first = confirmation._select_subject_cells(records, ["S1"])["S1"]
    second = confirmation._select_subject_cells(list(reversed(records)), ["S1"])["S1"]
    assert [row["cell_id"] for row in first] == [row["cell_id"] for row in second]
    assert len({row["cell_id"] for row in first}) == 4
    with pytest.raises(confirmation.ProtocolRefusal, match="FEWER_THAN_384"):
        confirmation._select_subject_cells(records[:3], ["S1"])


def test_adt_midrank_and_destroyed_rotation_preserve_exact_margins(monkeypatch):
    monkeypatch.setattr(confirmation, "CELL_BUDGET", 4)
    counts = np.ones((4, 13), dtype=np.int64)
    cells = [f"L|C{index}" for index in range(4)]
    states = confirmation._adt_states(counts, cells, "S1")
    assert np.all(states.sum(axis=0) == 2)
    destroyed = confirmation._destroyed_adt(states, cells, "S1")
    np.testing.assert_array_equal(destroyed.sum(axis=0), states.sum(axis=0))


def test_classical_lock_uses_iof2_loss_and_frozen_tie_order():
    losses = {
        "pooled_saturated_poisson": [0.5] * 8,
        "common_effect_stratified_cmle": [0.5] * 8,
        "poisson_independence_deviance_residual": [0.5] * 8,
    }
    assert (
        confirmation._lock_classical(losses)
        == "poisson_independence_deviance_residual"
    )
    losses["pooled_saturated_poisson"] = [0.4] * 8
    assert confirmation._lock_classical(losses) == "pooled_saturated_poisson"


def test_classical_transport_gets_the_same_source_only_multiplier_grid(monkeypatch):
    records = {
        f"S{index}": {
            "tables": np.ones((13, 13, 2, 2), dtype=np.int64)
        }
        for index in range(4)
    }
    monkeypatch.setattr(
        confirmation,
        "_fit_classical",
        lambda method, tables: {"population_log_odds": np.zeros((13, 13))},
    )
    monkeypatch.setattr(
        confirmation,
        "_predict_model",
        lambda model, rows, columns: np.full(
            (13, 13, 2, 2), model["transport_multiplier"], dtype=float
        ),
    )
    monkeypatch.setattr(
        confirmation,
        "_donor_loss",
        lambda truth, predicted: abs(float(predicted[0, 0, 0, 0]) - 0.75),
    )
    selected = confirmation._select_classical_transport_loocv(
        records, list(records), "pooled_saturated_poisson"
    )
    assert selected["selected_transport_multiplier"] == 0.75


def test_pooled_saturated_poisson_has_exact_log_odds_and_refuses_zero_cells():
    tables = np.tile(np.asarray([[[[[4, 2], [1, 3]]]]], dtype=np.int64), (3, 13, 13, 1, 1))
    fit = confirmation._fit_pooled_poisson(tables)
    np.testing.assert_allclose(fit["population_log_odds"], np.log(6.0))
    tables[..., 0, 0] = 0
    with pytest.raises(confirmation.CouplingEstimationRefusal, match="zero cell"):
        confirmation._fit_pooled_poisson(tables)


def test_all_prediction_families_restore_identical_recipient_margins():
    rows = np.tile(np.asarray([200, 184]), (13, 13, 1))
    columns = np.tile(np.asarray([192, 192]), (13, 13, 1))
    primary = {
        "configuration": {"transport_multiplier": 1.0},
        "population_log_odds": np.full((13, 13), 0.5),
    }
    residual = {
        "transport_multiplier": 1.0,
        "pooled_coordinate": np.full((13, 13), 0.02),
    }
    for model in (primary, residual):
        predicted = confirmation._predict_model(model, rows, columns)
        np.testing.assert_allclose(predicted.sum(axis=-1), rows)
        np.testing.assert_allclose(predicted.sum(axis=-2), columns)


def test_frozen_held_gate_uses_17_subjects_two_batch_robustness_and_seven_pathogens():
    subjects = [f"S{index}" for index in range(17)]
    batches = ["IOF3"] * 8 + ["IOF4"] * 9
    pathogen_axis = ["BK", "CMV", "Denv", "Lyme", "Malaria", "TB", "WNV"]
    pathogens = [pathogen_axis[index % 7] for index in range(17)]
    result = confirmation._held_comparison(
        subjects,
        batches,
        pathogens,
        np.full(17, 0.5),
        np.ones(17),
        gating=True,
        seed_offset=0,
    )
    assert result["passes"]
    assert result["favorable_subjects"] == 17
    assert result["donor_exact_sign_test"]["one_sided_p"] == 1 / 2**17
    assert result["pathogen_sign_flip"]["one_sided_p"] == 1 / 2**7
    assert set(result["batch_mean_differences"]) == {"IOF3", "IOF4"}
    assert not any("batch_sign" in key for key in result["checks"])


def test_pathogen_heterogeneity_can_fail_gate_even_when_global_mean_is_negative():
    subjects = [f"S{index}" for index in range(17)]
    batches = ["IOF3"] * 8 + ["IOF4"] * 9
    pathogens = ["BK"] * 2 + ["CMV"] * 3 + ["Denv"] * 2 + ["Lyme"] * 2 + ["Malaria"] * 2 + ["TB"] * 3 + ["WNV"] * 3
    comparator = np.ones(17)
    difference = np.full(17, -0.2)
    difference[-3:] = 0.4
    result = confirmation._held_comparison(
        subjects,
        batches,
        pathogens,
        comparator + difference,
        comparator,
        gating=True,
        seed_offset=1,
    )
    assert result["mean_paired_difference"] < 0.0
    assert not result["passes"]
    assert result["pathogen_sign_flip_role"].endswith("not a primary gate")
    assert result["pathogen_sign_flip"]["one_sided_p"] > 0.025


def test_numeric_stage_terminalizes_refusal_and_forbids_rerun(monkeypatch, tmp_path: Path):
    attempt = tmp_path / "attempt.jsonl"
    output = tmp_path / "result.json"
    attempt.write_text(
        json.dumps({"schema": "gse202150-stage-attempt/1.0", "stage": "source", "status": "STARTED", "created_at_utc": "T"}) + "\n"
    )
    monkeypatch.setitem(confirmation.ATTEMPT_PATHS, "source", attempt)
    monkeypatch.setitem(
        confirmation.EXECUTION_CLAIM_PATHS, "source", tmp_path / "execution.json"
    )
    monkeypatch.setitem(confirmation.STAGE_OUTPUTS, "source", output)
    monkeypatch.setattr(
        confirmation,
        "_attempt_started",
        lambda stage: {
            "created_at_utc": "T",
            "protocol_commit": "a" * 40,
            "runtime_environment": {"runtime": "frozen"},
        },
    )
    result = confirmation._run_claimed_stage(
        "source",
        lambda: (_ for _ in ()).throw(confirmation.ProtocolRefusal("FROZEN_FAILURE")),
    )
    assert result["status"] == "TERMINAL_SOURCE_REFUSAL"
    assert result["refusal_code"] == "FROZEN_FAILURE"
    assert [json.loads(line).get("status") for line in attempt.read_text().splitlines()] == [
        "STARTED",
        "EXECUTING_CONSUMED",
        "FINISHED",
    ]
    with pytest.raises(PermissionError, match="already exists"):
        confirmation._run_claimed_stage("source", lambda: {})


def test_unexpected_terminal_refusal_does_not_serialize_exception_text(monkeypatch, tmp_path: Path):
    attempt = tmp_path / "attempt.jsonl"
    output = tmp_path / "result.json"
    attempt.write_text("{}\n")
    monkeypatch.setitem(confirmation.ATTEMPT_PATHS, "predict", attempt)
    monkeypatch.setitem(
        confirmation.EXECUTION_CLAIM_PATHS, "predict", tmp_path / "execution.json"
    )
    monkeypatch.setitem(confirmation.STAGE_OUTPUTS, "predict", output)
    monkeypatch.setattr(
        confirmation,
        "_attempt_started",
        lambda stage: {
            "created_at_utc": "T",
            "protocol_commit": "a" * 40,
            "runtime_environment": {"runtime": "frozen"},
        },
    )
    result = confirmation._run_claimed_stage(
        "predict", lambda: (_ for _ in ()).throw(RuntimeError("/private/path"))
    )
    assert result["refusal_code"] == "UNEXPECTED_RUNTIMEERROR"
    assert "/private/path" not in output.read_text()


def test_keyboard_interrupt_consumes_numeric_stage(monkeypatch, tmp_path: Path):
    attempt = tmp_path / "attempt.jsonl"
    output = tmp_path / "result.json"
    attempt.write_text("{}\n")
    monkeypatch.setitem(confirmation.ATTEMPT_PATHS, "score", attempt)
    monkeypatch.setitem(
        confirmation.EXECUTION_CLAIM_PATHS, "score", tmp_path / "execution.json"
    )
    monkeypatch.setitem(confirmation.STAGE_OUTPUTS, "score", output)
    monkeypatch.setattr(
        confirmation,
        "_attempt_started",
        lambda stage: {
            "created_at_utc": "T",
            "protocol_commit": "a" * 40,
            "runtime_environment": {"runtime": "frozen"},
        },
    )
    result = confirmation._run_claimed_stage(
        "score", lambda: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    assert result["refusal_code"] == "UNEXPECTED_KEYBOARDINTERRUPT"
    assert [json.loads(line).get("status") for line in attempt.read_text().splitlines()] == [
        None,
        "EXECUTING_CONSUMED",
        "FINISHED",
    ]
    with pytest.raises(PermissionError, match="already exists"):
        confirmation._run_claimed_stage("score", lambda: {})


def test_atomic_execution_claim_allows_only_one_concurrent_numeric_body(
    monkeypatch, tmp_path: Path
):
    attempt = tmp_path / "attempt.jsonl"
    output = tmp_path / "result.json"
    execution = tmp_path / "execution.json"
    attempt.write_text("{}\n")
    monkeypatch.setitem(confirmation.ATTEMPT_PATHS, "source", attempt)
    monkeypatch.setitem(confirmation.EXECUTION_CLAIM_PATHS, "source", execution)
    monkeypatch.setitem(confirmation.STAGE_OUTPUTS, "source", output)
    barrier = threading.Barrier(2)
    started = {
        "created_at_utc": "T",
        "protocol_commit": "a" * 40,
        "runtime_environment": {"runtime": "frozen"},
    }

    def attempt_started(stage):
        barrier.wait()
        return started

    monkeypatch.setattr(confirmation, "_attempt_started", attempt_started)
    body_count = 0
    lock = threading.Lock()

    def body():
        nonlocal body_count
        with lock:
            body_count += 1
        return {"status": "SOURCE_READY_FOR_HELD_RNA_PREDICTION"}

    def invoke():
        try:
            return confirmation._run_claimed_stage("source", body)
        except (FileExistsError, PermissionError):
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: invoke(), range(2)))
    assert body_count == 1
    assert sum(result is not None for result in results) == 1
    assert json.loads(execution.read_text())["status"] == "EXECUTING_CONSUMED"


def test_claim_writes_started_only_after_public_prerequisite(monkeypatch, tmp_path: Path):
    attempt = tmp_path / "attempt.jsonl"
    output = tmp_path / "output.json"
    monkeypatch.setitem(confirmation.ATTEMPT_PATHS, "source", attempt)
    monkeypatch.setitem(
        confirmation.EXECUTION_CLAIM_PATHS, "source", tmp_path / "execution.json"
    )
    monkeypatch.setitem(confirmation.STAGE_OUTPUTS, "source", output)
    monkeypatch.setattr(confirmation, "_prerequisite_for_claim", lambda stage: ("TAG", ("FILE",)))
    monkeypatch.setattr(confirmation, "_require_public_tag", lambda tag, paths: "a" * 40)
    monkeypatch.setattr(
        confirmation, "_require_runtime_environment", lambda: {"runtime": "frozen"}
    )
    result = confirmation.claim_stage("source")
    assert result["status"] == "STARTED"
    assert json.loads(attempt.read_text())["prerequisite_commit"] == "a" * 40
    assert json.loads(attempt.read_text())["protocol_commit"] == "a" * 40
    with pytest.raises(PermissionError, match="rerun is forbidden"):
        confirmation.claim_stage("source")


def test_completed_result_requires_bound_three_state_ledger(monkeypatch, tmp_path: Path):
    runtime = {"runtime": "frozen"}
    protocol_commit = "a" * 40
    result = tmp_path / "result.json"
    attempt = tmp_path / "attempt.jsonl"
    result.write_text(
        json.dumps(
            {
                "status": "SOURCE_READY_FOR_HELD_RNA_PREDICTION",
                "protocol_commit": protocol_commit,
                "runtime_environment": runtime,
            }
        )
        + "\n"
    )
    rows = [
        {
            "stage": "source",
            "status": "STARTED",
            "protocol_commit": protocol_commit,
            "runtime_environment": runtime,
        },
        {
            "stage": "source",
            "status": "EXECUTING_CONSUMED",
            "protocol_commit": protocol_commit,
            "runtime_environment": runtime,
        },
        {
            "stage": "source",
            "status": "FINISHED",
            "protocol_commit": protocol_commit,
            "runtime_environment": runtime,
            "terminal_status": "SOURCE_READY_FOR_HELD_RNA_PREDICTION",
            "output_sha256": confirmation._sha256(result),
        },
    ]
    attempt.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setitem(confirmation.ATTEMPT_PATHS, "source", attempt)
    execution_claim = tmp_path / "execution.json"
    execution_claim.write_text(json.dumps(rows[1]) + "\n")
    monkeypatch.setitem(
        confirmation.EXECUTION_CLAIM_PATHS, "source", execution_claim
    )
    monkeypatch.setattr(confirmation, "_relative", lambda path: path.name)
    monkeypatch.setattr(
        confirmation, "_require_public_tag", lambda tag, paths: protocol_commit
    )
    monkeypatch.setattr(
        confirmation, "_require_runtime_environment", lambda: runtime
    )
    payload, _ = confirmation._require_completed_stage_artifact(
        "RESULT_TAG", result, "source"
    )
    assert payload["status"] == "SOURCE_READY_FOR_HELD_RNA_PREDICTION"
    rows[1]["status"] = "STARTED"
    attempt.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(PermissionError, match="three-state"):
        confirmation._require_completed_stage_artifact(
            "RESULT_TAG", result, "source"
        )


def test_public_result_verifier_binds_score_output_ledger_and_execution_claim(
    monkeypatch, tmp_path: Path
):
    runtime = {"runtime": "frozen"}
    protocol_commit = "a" * 40
    score = tmp_path / "score.json"
    attempt = tmp_path / "attempt.jsonl"
    execution = tmp_path / "execution.json"
    score.write_text(
        json.dumps(
            {
                "schema": "gse202150-held-validation/1.0",
                "status": "COMPLETED_HELD_VALIDATION_PASS",
                "protocol_commit": protocol_commit,
                "runtime_environment": runtime,
            }
        )
        + "\n"
    )
    executing = {
        "stage": "score",
        "status": "EXECUTING_CONSUMED",
        "protocol_commit": protocol_commit,
        "runtime_environment": runtime,
    }
    rows = [
        {
            "stage": "score",
            "status": "STARTED",
            "protocol_commit": protocol_commit,
            "runtime_environment": runtime,
        },
        executing,
        {
            "stage": "score",
            "status": "FINISHED",
            "protocol_commit": protocol_commit,
            "runtime_environment": runtime,
            "terminal_status": "COMPLETED_HELD_VALIDATION_PASS",
            "output_sha256": confirmation._sha256(score),
        },
    ]
    attempt.write_text("".join(json.dumps(row) + "\n" for row in rows))
    execution.write_text(json.dumps(executing) + "\n")
    monkeypatch.setattr(confirmation, "DEFAULT_SCORE", score)
    monkeypatch.setitem(confirmation.ATTEMPT_PATHS, "score", attempt)
    monkeypatch.setitem(confirmation.EXECUTION_CLAIM_PATHS, "score", execution)
    monkeypatch.setattr(confirmation, "_relative", lambda path: path.name)
    observed = []

    def public_tag(tag, paths):
        observed.append((tag, tuple(paths)))
        return protocol_commit

    monkeypatch.setattr(confirmation, "_require_public_tag", public_tag)
    monkeypatch.setattr(
        confirmation, "_require_runtime_environment", lambda: runtime
    )
    result = confirmation.verify_public_result()
    assert observed[0] == (
        confirmation.RESULT_TAG,
        (score.name, attempt.name, execution.name),
    )
    assert result["status"] == "PUBLIC_RESULT_LINEAGE_VERIFIED"


def test_private_artifacts_and_public_payloads_enforce_separation(tmp_path: Path):
    assert confirmation._private_path(tmp_path / "states.json").is_absolute()
    with pytest.raises(PermissionError, match="outside"):
        confirmation._private_path(confirmation.ROOT / "states.json")
    with pytest.raises(PermissionError, match="private key"):
        confirmation._validate_public_payload({"states": [0, 1]})
    with pytest.raises(PermissionError, match="local path"):
        confirmation._validate_public_payload({"artifact": "/tmp/private.json"})


def test_h5_identity_uses_sha256_not_only_byte_count(monkeypatch, tmp_path: Path):
    path = tmp_path / "input.h5"
    path.write_bytes(b"abcd")
    record = {
        "batch": "IOF1",
        "bytes": 4,
        "filename": path.name,
        "library": "ADRA03_IOF1_1",
        "sha256": confirmation._sha256(path),
    }
    monkeypatch.setattr(confirmation, "_manifest", lambda: {"h5_files": [record]})
    assert confirmation._h5_paths(tmp_path, ("IOF1",))["ADRA03_IOF1_1"] == path
    path.write_bytes(b"wxyz")
    with pytest.raises(PermissionError, match="H5 bytes differ"):
        confirmation._h5_paths(tmp_path, ("IOF1",))


def test_destroyed_link_loss_is_deferred_until_after_primary_gate():
    source = inspect.getsource(confirmation._score_body)
    assert 'scored_methods = [method for method in methods if method != "destroyed_link"]' in source
    assert source.index('if primary_comparison["passes"]') < source.index(
        'record["predicted_tables"]["destroyed_link"]'
    )


def test_protocol_binds_runner_test_designation_manifest_preflight_and_estimators():
    required = {
        "experiments/confirm_gse202150_citeseq.py",
        "tests/test_gse202150_citeseq_confirmation.py",
        "docs/GSE202150_ACUTE_INFECTION_HELD_BATCH_CONFIRMATION_PROTOCOL_2026-08-29.md",
        "data/confirmation/gse202150_citeseq/candidate_designation_v1.json",
        "data/confirmation/gse202150_citeseq/source_manifest_v1.json",
        "data/confirmation/gse202150_citeseq/runtime_environment_v1.json",
        "results/development/gse202150_metadata_preflight_v1.json",
        "mapreg/hierarchical_conditional_coupling.py",
        "mapreg/heterogeneity_adaptive_coupling.py",
    }
    assert required <= set(confirmation.PROTOCOL_BINDINGS)
    assert len(confirmation.PROTOCOL_TAG) > 0
    assert len({confirmation.PROTOCOL_TAG, confirmation.SOURCE_RESULT_TAG, confirmation.PREDICTION_TAG, confirmation.RESULT_TAG}) == 4
