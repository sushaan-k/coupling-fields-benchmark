from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import gzip
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from experiments import confirm_gse252762_celiac as runner


class _Response(io.BytesIO):
    def __init__(
        self,
        payload: bytes,
        url: str,
        *,
        content_encoding: str | None = None,
    ) -> None:
        super().__init__(payload)
        self.status = 200
        self.headers = {"Content-Length": str(len(payload))}
        if content_encoding is not None:
            self.headers["Content-Encoding"] = content_encoding
        self._url = url

    def geturl(self) -> str:
        return self._url

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _matrix(rows: int = 9, columns: int = 256) -> bytes:
    entries = [
        f"{row} {column} {(row + column) % 5 + 1}"
        for row in range(1, rows + 1)
        for column in range(1, columns + 1)
    ]
    text = (
        "%%MatrixMarket matrix coordinate integer general\n"
        f"{rows} {columns} {len(entries)}\n" + "\n".join(entries) + "\n"
    )
    return gzip.compress(text.encode(), mtime=0)


def _held_preflight(sample_count: int = 13) -> dict[str, Any]:
    payload = _matrix(columns=256 * sample_count)
    url = "https://example.test/GSE252762_batch6_rna_matrix.mtx.gz"
    conditions = (
        runner.core.HELD_CONDITIONS
        if sample_count == len(runner.core.HELD_CONDITIONS)
        else tuple(
            ("ACD", "GFD", "CONTROL")[index % 3] for index in range(sample_count)
        )
    )
    samples = [
        {
            "sample_id": f"B6:S{index}",
            "batch": 6,
            "role": "held",
            "condition": conditions[index],
            "context": "CONTROL" if conditions[index] == "CONTROL" else "CELIAC",
            "selected_barcodes": [f"B{index}-{cell}" for cell in range(256)],
            "selected_columns_1_based": [256 * index + cell + 1 for cell in range(256)],
        }
        for index in range(sample_count)
    ]
    return {
        "markers": [
            {
                "rna": rna,
                "adt": adt,
                "rna_row_1_based": index + 1,
                "cite_row_1_based": index + 1,
            }
            for index, (rna, adt) in enumerate(runner.core.MARKER_PAIRS)
        ],
        "samples": samples,
        "batches": [
            {
                "batch": 6,
                "rna_shape": [9, 256 * sample_count],
                "cite_shape": [9, 256 * sample_count],
                "files": [
                    {
                        "name": "GSE252762_batch6_rna_matrix.mtx.gz",
                        "url": url,
                        "bytes": len(payload),
                        "sha256": None,
                    },
                    {
                        "name": "GSE252762_batch6_cite_matrix.mtx.gz",
                        "url": url.replace("rna", "cite"),
                        "bytes": len(payload),
                        "sha256": None,
                    },
                ],
            }
        ],
        "_payload": payload,
    }


def _redirect_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    path_names = (
        "SOURCE_AUTHORIZATION",
        "SOURCE_ATTEMPT",
        "SOURCE_CONSUMPTION",
        "SOURCE_JOURNAL",
        "SOURCE_REDUCED",
        "SOURCE_RESULT",
        "PREDICTION_AUTHORIZATION",
        "PREDICTION_ATTEMPT",
        "PREDICTION_CONSUMPTION",
        "PREDICTION_JOURNAL",
        "PREDICTION_REDUCED",
        "PREDICTIONS",
        "SCORE_AUTHORIZATION",
        "SCORE_ATTEMPT",
        "SCORE_CONSUMPTION",
        "SCORE_JOURNAL",
        "SCORE_REDUCED",
        "SCORE_RESULT",
    )
    paths = {}
    for name in path_names:
        path = tmp_path / f"{name.lower()}.json"
        monkeypatch.setattr(runner, name, path)
        paths[name] = path
    stage_paths = deepcopy(runner.STAGE_PATHS)
    for stage, prefix in (
        ("source", "SOURCE"),
        ("prediction", "PREDICTION"),
        ("score", "SCORE"),
    ):
        stage_paths[stage]["authorization"] = paths[f"{prefix}_AUTHORIZATION"]
        stage_paths[stage]["attempt"] = paths[f"{prefix}_ATTEMPT"]
        stage_paths[stage]["consumption"] = paths[f"{prefix}_CONSUMPTION"]
        stage_paths[stage]["journal"] = paths[f"{prefix}_JOURNAL"]
        stage_paths[stage]["sidecars"] = tuple(
            tmp_path / f"{prefix.lower()}_selected_{index}.json"
            for index in range(stage_paths[stage]["matrix_gets"])
        )
        stage_paths[stage]["checkpoint"] = paths[f"{prefix}_REDUCED"]
    stage_paths["source"]["result"] = paths["SOURCE_RESULT"]
    stage_paths["prediction"]["result"] = paths["PREDICTIONS"]
    stage_paths["score"]["result"] = paths["SCORE_RESULT"]
    monkeypatch.setattr(runner, "STAGE_PATHS", stage_paths)


def _audit_certificate(
    preflight: dict[str, Any],
    batch: int,
    modality: str,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    batch_record = runner._batch_record(preflight, batch)
    matrix = runner._matrix_record(batch_record, modality)
    row_field = "rna_row_1_based" if modality == "rna" else "cite_row_1_based"
    return {
        "banner": "%%MatrixMarket matrix coordinate integer general",
        "matrix_shape": batch_record[f"{modality}_shape"],
        "selected_rows": [marker[row_field] for marker in preflight["markers"]],
        "selected_columns": [
            column
            for sample in samples
            for column in sample["selected_columns_1_based"]
        ],
        "declared_nnz": 0,
        "parsed_nnz": 0,
        "comment_lines": 0,
        "blank_lines": 0,
        "zero_value_entries": 0,
        "selected_entries": 0,
        "selected_distinct_coordinates": 0,
        "selected_duplicate_entries": 0,
        "global_value_sum": 0,
        "selected_value_sum": 0,
        "compressed_bytes": matrix["bytes"],
        "decompressed_bytes": 1,
        "compressed_sha256": "a" * 64,
        "decompressed_sha256": "b" * 64,
        "selected_block_sha256": "c" * 64,
        "compressed_source_exhausted": True,
        "gzip_stream_exhausted": True,
        "output_dtype": "int64",
    }


def _write_attempt_consumption_and_header(stage: str) -> None:
    config = runner.STAGE_PATHS[stage]
    runtime = runner._runtime()
    attempt = {
        "schema": "gse252762-celiac-stage-attempt/2.0",
        "stage": stage,
        "status": "CLAIMED_BEFORE_MATRIX_ACCESS",
        "authorization_commit": "a" * 40,
        "implementation_commit": "b" * 40,
        "matrix_gets_authorized": config["matrix_gets"],
        "runtime": runtime,
        "rerun_permitted": False,
    }
    config["attempt"].write_text(json.dumps(attempt) + "\n")
    consumption = {
        "schema": "gse252762-celiac-stage-consumption/2.0",
        "stage": stage,
        "status": "CONSUMED_BEFORE_FIRST_MATRIX_REQUEST",
        "attempt_path": runner._relative(config["attempt"]),
        "attempt_sha256": runner._sha256(config["attempt"]),
        "public_attempt_commit": "c" * 40,
        "execution_id": "d" * 32,
        "runtime": runtime,
        "rerun_permitted": False,
    }
    config["consumption"].write_text(json.dumps(consumption) + "\n")
    header = {
        "schema": "gse252762-celiac-access-journal/2.0",
        "stage": stage,
        "event": "OPENED_BEFORE_MATRIX_ACCESS",
        "attempt_sha256": runner._sha256(config["attempt"]),
        "consumption_path": runner._relative(config["consumption"]),
        "consumption_sha256": runner._sha256(config["consumption"]),
        "public_consumption_commit": "e" * 40,
        "execution_id": consumption["execution_id"],
        "automatic_retries": False,
        "http_redirects": False,
    }
    config["journal"].write_text(json.dumps(header) + "\n")


def _source_reduced_records() -> dict[str, Any]:
    table = np.full((9, 9, 2, 2), 64, dtype=np.int64)
    records = []
    for index in range(16):
        role = "calibration" if index < 9 else "pilot"
        records.append(
            {
                "sample_id": f"S{index}",
                "role": role,
                "condition": "ACD" if index % 2 == 0 else "CONTROL",
                "context": "CELIAC" if index % 2 == 0 else "CONTROL",
                "truth_tables": table.tolist(),
                "destroyed_tables": table.tolist(),
                "truth_sha256": runner._array_sha256(table),
                "destroyed_sha256": runner._array_sha256(table),
            }
        )
    return {"samples": records}


def _independence_predictions(preflight: dict[str, Any]) -> dict[str, Any]:
    states = np.zeros((256, 9), dtype=np.uint8)
    states[:128] = 1
    rows, columns = runner.core.rna_margin_tables(states)
    tables = np.full((9, 9, 2, 2), 64.0)
    return {
        "samples": [
            {
                "sample_id": sample["sample_id"],
                "condition": sample["condition"],
                "context": sample["context"],
                "rna_states": states.tolist(),
                "rna_states_sha256": runner._array_sha256(states),
                "row_margins": rows.tolist(),
                "column_margins": columns.tolist(),
                "predictions": {
                    method: tables.tolist() for method in runner.core.MANDATORY_METHODS
                },
            }
            for sample in preflight["samples"]
        ]
    }


def test_strict_stream_reduction_binds_complete_matrix_axes_and_identity_encoding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    preflight = _held_preflight(13)
    payload = preflight.pop("_payload")
    url = preflight["batches"][0]["files"][0]["url"]
    seen_headers: dict[str, str] = {}

    def open_url(request, timeout):
        assert timeout == 180
        seen_headers.update(dict(request.header_items()))
        return _Response(payload, url)

    monkeypatch.setattr(runner, "_open_url", open_url)
    _redirect_paths(monkeypatch, tmp_path)
    _write_attempt_consumption_and_header("prediction")
    journal = runner.PREDICTION_JOURNAL
    lease = runner._AccessLease("prediction", "d" * 32, ((6, "rna"),))
    block, audit = runner._reduce_matrix(
        preflight, 6, "rna", preflight["samples"], journal, lease
    )
    assert block.shape == (9, 13 * 256)
    assert audit["compressed_bytes"] == len(payload)
    assert audit["selected_rows"] == list(range(1, 10))
    assert audit["selected_columns"] == list(range(1, 13 * 256 + 1))
    assert seen_headers["Accept-encoding"] == "identity"
    events = [json.loads(line)["event"] for line in journal.read_text().splitlines()]
    assert events == [
        "OPENED_BEFORE_MATRIX_ACCESS",
        "MATRIX_ACCESS_STARTED",
        "MATRIX_REDUCTION_FINISHED",
    ]
    assert runner.STAGE_PATHS["prediction"]["sidecars"][0].is_file()


@pytest.mark.parametrize(
    "mutation",
    (
        {
            "declared_nnz": 0,
            "parsed_nnz": 0,
            "selected_entries": 1,
            "selected_distinct_coordinates": 1,
            "selected_duplicate_entries": 0,
        },
        {"declared_nnz": 0, "parsed_nnz": 0, "zero_value_entries": 1},
        {
            "declared_nnz": 29_953,
            "parsed_nnz": 29_953,
            "selected_entries": 29_953,
            "selected_distinct_coordinates": 29_953,
            "selected_duplicate_entries": 0,
        },
        {"global_value_sum": 0, "selected_value_sum": 1},
    ),
)
def test_matrix_audit_rejects_impossible_counter_algebra(
    mutation: dict[str, int],
) -> None:
    preflight = _held_preflight(13)
    preflight.pop("_payload")
    samples = preflight["samples"]
    audit = _audit_certificate(preflight, 6, "rna", samples)
    audit.update(mutation)
    with pytest.raises(runner.ProtocolRefusal, match="certificate differs"):
        runner._validate_audit_certificate(audit, preflight, 6, "rna", samples)


def test_bad_matrix_response_is_journaled_without_free_text_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    preflight = _held_preflight(13)
    payload = preflight.pop("_payload")
    monkeypatch.setattr(
        runner,
        "_open_url",
        lambda *_args, **_kwargs: _Response(payload, "https://wrong.test/file"),
    )
    _redirect_paths(monkeypatch, tmp_path)
    _write_attempt_consumption_and_header("prediction")
    journal = runner.PREDICTION_JOURNAL
    lease = runner._AccessLease("prediction", "d" * 32, ((6, "rna"),))
    with pytest.raises(runner.ProtocolRefusal, match="URL or status"):
        runner._reduce_matrix(preflight, 6, "rna", preflight["samples"], journal, lease)
    event = json.loads(journal.read_text().splitlines()[-1])
    assert event["event"] == "MATRIX_ACCESS_FAILED"
    assert event["exception_class"] == "ProtocolRefusal"
    assert "message" not in event


def test_redirect_handler_refuses_redirects_before_following() -> None:
    handler = runner._NoRedirect()
    assert handler.redirect_request(None, None, 302, "Found", {}, "x", "y") is None


def test_public_boundary_forbids_premature_outcome_artifacts() -> None:
    source_authorization = set(runner._forbidden_at_boundary("source", "authorization"))
    source_attempt = set(runner._forbidden_at_boundary("source", "attempt"))
    source_result = set(runner._forbidden_at_boundary("source", "result"))
    assert runner.SOURCE_ATTEMPT in source_authorization
    assert runner.SOURCE_REDUCED in source_authorization
    assert runner.SOURCE_REDUCED in source_attempt
    assert runner.STAGE_PATHS["source"]["sidecars"][0] in source_attempt
    assert runner.PREDICTION_AUTHORIZATION in source_result
    assert runner.SCORE_RESULT in source_result
    assert runner.SOURCE_RESULT not in source_result


def test_public_tag_rejects_forbidden_file_already_in_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    required = tmp_path / "required.json"
    forbidden = tmp_path / "forbidden.json"
    required.write_bytes(b"required\n")

    def git(*arguments: str, check: bool = True, text: bool = True):
        if arguments[:1] == ("for-each-ref",):
            return SimpleNamespace(stdout="", returncode=0)
        if arguments[:3] == ("rev-parse", "--git-path", "info/grafts"):
            return SimpleNamespace(stdout=str(tmp_path / "grafts") + "\n", returncode=0)
        if arguments[:1] == ("config",):
            return SimpleNamespace(stdout="", returncode=1)
        if arguments[:2] == ("cat-file", "-t"):
            return SimpleNamespace(stdout="tag\n", returncode=0)
        if arguments[:1] == ("rev-parse",):
            value = "a" * 40 if "^{}" not in arguments[1] else "b" * 40
            return SimpleNamespace(stdout=value + "\n", returncode=0)
        if arguments[:1] == ("ls-remote",):
            tag = arguments[2].removeprefix("refs/tags/")
            return SimpleNamespace(
                stdout=(
                    f"{'a' * 40}\trefs/tags/{tag}\n{'b' * 40}\trefs/tags/{tag}^{{}}\n"
                ),
                returncode=0,
            )
        if arguments[:1] == ("show",):
            return SimpleNamespace(stdout=required.read_bytes(), returncode=0)
        if arguments[:2] == ("cat-file", "-e"):
            return SimpleNamespace(stdout="", returncode=0)
        raise AssertionError(arguments)

    monkeypatch.setattr(runner, "_git", git)
    with pytest.raises(runner.ProtocolRefusal, match="prematurely"):
        runner._require_public_tag(
            "example-tag", (required,), absent_paths=(forbidden,)
        )


def test_public_tag_verification_against_bare_remote_and_replace_ref(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    remote = tmp_path / "remote.git"
    subprocess.run(
        ["git", "init", "--bare", str(remote)], check=True, capture_output=True
    )
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    for key, value in (("user.name", "Test"), ("user.email", "test@example.org")):
        subprocess.run(
            ["git", "config", key, value], cwd=repo, check=True, capture_output=True
        )
    required = repo / "required.json"
    forbidden = repo / "forbidden.json"
    required.write_text('{"frozen":true}\n')
    subprocess.run(["git", "add", "required.json"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "freeze"], cwd=repo, check=True)
    subprocess.run(
        ["git", "tag", "-a", "freeze-v1", "-m", "freeze"], cwd=repo, check=True
    )
    subprocess.run(
        ["git", "remote", "add", "origin", str(remote)], cwd=repo, check=True
    )
    subprocess.run(["git", "push", "origin", "freeze-v1"], cwd=repo, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(runner, "ROOT", repo)
    monkeypatch.setattr(runner, "PUBLIC_ORIGIN", str(remote))
    assert (
        runner._require_public_tag("freeze-v1", (required,), absent_paths=(forbidden,))
        == commit
    )

    required.write_text('{"frozen":false}\n')
    subprocess.run(["git", "add", "required.json"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "replacement"], cwd=repo, check=True)
    replacement = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(["git", "replace", commit, replacement], cwd=repo, check=True)
    with pytest.raises(runner.ProtocolRefusal, match="replacement or graft"):
        runner._require_public_tag("freeze-v1", (required,))


def test_strict_json_rejects_nonfinite_constants(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"value": NaN}\n')
    with pytest.raises(ValueError, match="NaN"):
        runner._read_json(path)


def test_atomic_json_create_never_overwrites_or_exposes_a_torn_final(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "artifact.json"
    runner._write_json(path, {"version": 1})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        runner._write_json(path, {"version": 2})
    assert path.read_bytes() == original

    torn = tmp_path / "torn.json"
    original_write = runner.os.write
    writes = 0

    def partial_then_fail(descriptor: int, payload: bytes) -> int:
        nonlocal writes
        writes += 1
        if writes == 1:
            return original_write(descriptor, payload[: max(1, len(payload) // 2)])
        raise OSError("simulated interrupted write")

    with monkeypatch.context() as patch:
        patch.setattr(runner.os, "write", partial_then_fail)
        with pytest.raises(OSError, match="interrupted"):
            runner._write_json(torn, {"payload": "x" * 100})
    assert not torn.exists()
    assert not list(tmp_path.glob(f".{torn.name}.*.tmp"))


def test_journal_append_failure_preserves_complete_previous_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    journal = tmp_path / "access.jsonl"
    runner._write_journal_header(journal, {"event": "OPENED"})
    original = journal.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(
            runner.os,
            "replace",
            lambda *_args: (_ for _ in ()).throw(OSError("simulated replace failure")),
        )
        with pytest.raises(OSError, match="replace failure"):
            runner._append_journal(journal, {"event": "STARTED"})
    assert journal.read_bytes() == original
    assert not list(tmp_path.glob(f".{journal.name}.*.tmp"))


def test_recovery_after_finished_reduction_binds_replayable_raw_sidecar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    preflight = _held_preflight(13)
    matrix = preflight.pop("_payload")
    url = preflight["batches"][0]["files"][0]["url"]
    _redirect_paths(monkeypatch, tmp_path)
    _write_attempt_consumption_and_header("prediction")
    monkeypatch.setattr(
        runner,
        "_open_url",
        lambda *_args, **_kwargs: _Response(matrix, url),
    )
    lease = runner._AccessLease("prediction", "d" * 32, ((6, "rna"),))
    runner._reduce_matrix(
        preflight,
        6,
        "rna",
        preflight["samples"],
        runner.PREDICTION_JOURNAL,
        lease,
    )
    assert not runner.PREDICTION_REDUCED.exists()
    monkeypatch.setattr(runner, "_preflight", lambda: preflight)
    monkeypatch.setattr(
        runner,
        "_verify_consumption_public",
        lambda _stage: ({}, "e" * 40, {}, {}),
    )
    failure = runner.recover("prediction")
    assert failure["phase"] == "recovery"
    assert len(failure["selected_count_sidecars"]) == 1
    runner._validate_failure("prediction", failure)

    sidecar = runner.STAGE_PATHS["prediction"]["sidecars"][0]
    changed = runner._read_json(sidecar)
    changed["selected_block"][0][0] += 1
    sidecar.write_text(json.dumps(changed) + "\n")
    with pytest.raises(runner.ProtocolRefusal, match="sidecar bindings"):
        runner._validate_failure("prediction", failure)


def test_public_terminal_failure_proves_optional_current_artifacts_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _redirect_paths(monkeypatch, tmp_path)
    _write_attempt_consumption_and_header("source")
    runner.SOURCE_JOURNAL.unlink()
    runner._failure("source", "source_reduction", RuntimeError("synthetic"))
    monkeypatch.setattr(
        runner,
        "_verify_consumption_public",
        lambda _stage: ({}, "a" * 40, {}, {}),
    )
    observed: dict[str, tuple[Path, ...]] = {}

    def public_tag(_tag, _paths, *, absent_paths=()):
        observed["absent"] = tuple(absent_paths)
        return "b" * 40

    monkeypatch.setattr(runner, "_require_public_tag", public_tag)
    monkeypatch.setattr(runner, "_require_ancestor", lambda *_args: None)
    runner._verify_terminal_failure_public("source")
    absent = set(observed["absent"])
    assert runner.SOURCE_JOURNAL in absent
    assert runner.SOURCE_REDUCED in absent
    assert set(runner.STAGE_PATHS["source"]["sidecars"]) <= absent


def test_runtime_contract_rejects_python_and_cross_stage_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frozen = deepcopy(runner.FROZEN_RUNTIME)
    changed = deepcopy(frozen)
    changed["python"] = "3.10.0"
    assert runner._runtime_payload_is_frozen(frozen)
    assert not runner._runtime_payload_is_frozen(changed)

    monkeypatch.setattr(runner, "_runtime", lambda: changed)
    with pytest.raises(runner.ProtocolRefusal, match="runtime differs"):
        runner._require_runtime()

    _redirect_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "_runtime", lambda: frozen)
    _write_attempt_consumption_and_header("source")
    _write_attempt_consumption_and_header("prediction")
    runner._validate_consumption("source")

    prediction = runner._read_json(runner.PREDICTION_ATTEMPT)
    prediction["runtime"] = changed
    runner.PREDICTION_ATTEMPT.write_text(json.dumps(prediction) + "\n")
    consumption = runner._read_json(runner.PREDICTION_CONSUMPTION)
    consumption["attempt_sha256"] = runner._sha256(runner.PREDICTION_ATTEMPT)
    consumption["runtime"] = changed
    runner.PREDICTION_CONSUMPTION.write_text(json.dumps(consumption) + "\n")
    with pytest.raises(runner.ProtocolRefusal, match="contract differs"):
        runner._validate_consumption("prediction")


def test_complete_transitive_binding_includes_table_prediction() -> None:
    assert "mapreg/table_prediction.py" in runner.IMPLEMENTATION_BINDINGS
    assert "mapreg/poisson_loglinear.py" in runner.IMPLEMENTATION_BINDINGS
    assert "tests/test_poisson_loglinear.py" in runner.IMPLEMENTATION_BINDINGS
    assert (
        "docs/GSE252762_CELIAC_CITESEQ_HELD_BATCH_PROTOCOL_V2_2026-08-30.md"
        in runner.IMPLEMENTATION_BINDINGS
    )
    assert "docs/GSE252762_CELIAC_EXECUTION_CONTRACT_V2.md" in (
        runner.IMPLEMENTATION_BINDINGS
    )
    assert "requirements.txt" in runner.IMPLEMENTATION_BINDINGS


def test_v2_campaign_paths_tags_and_schemas_cannot_collide_with_v1() -> None:
    assert runner.CANDIDATE_COMMIT == "fd84891c9c4be03e7faeeffd09838a98f2f1bda1"
    assert runner.CANDIDATE_TAG == "gse252762-celiac-v2-candidate"
    assert runner.DESIGNATION.name == "candidate_designation_v2.json"
    assert runner.PROTOCOL.name == (
        "GSE252762_CELIAC_CITESEQ_HELD_BATCH_PROTOCOL_V2_2026-08-30.md"
    )
    assert all(
        "v2" in path.name and "v1" not in path.name
        for path in (runner._campaign_artifact_paths())
    )
    assert all(
        config["authorization_schema"].endswith("/2.0")
        for config in runner.STAGE_PATHS.values()
    )
    tags = (
        runner.IMPLEMENTATION_TAG,
        runner.SOURCE_AUTHORIZATION_TAG,
        runner.SOURCE_ATTEMPT_TAG,
        runner.SOURCE_CONSUMPTION_TAG,
        runner.SOURCE_RESULT_TAG,
        runner.PREDICTION_AUTHORIZATION_TAG,
        runner.PREDICTION_ATTEMPT_TAG,
        runner.PREDICTION_CONSUMPTION_TAG,
        runner.PREDICTIONS_TAG,
        runner.SCORE_AUTHORIZATION_TAG,
        runner.SCORE_ATTEMPT_TAG,
        runner.SCORE_CONSUMPTION_TAG,
        runner.SCORE_RESULT_TAG,
    )
    assert all("-v2-" in tag and "-v1-" not in tag for tag in tags)


def test_v2_selection_interface_requires_matched_ridge_poisson() -> None:
    selection = {
        "schema": "gse252762-celiac-source-selection/2.0",
        "selected_comparator_alphas": {
            method: 0.0 for method in runner.core.CLASSICAL_METHODS
        },
        "strongest_benchmark": "independence",
        "calibration_models": {
            "strongest_benchmark": "independence",
            "donor_stratified_ridge_poisson_field": [],
            "donor_stratified_ridge_poisson_certificate": {"status": "FINITE"},
        },
        "pilot_sample_losses": {
            method: [1.0] * 7 for method in runner.core.MANDATORY_METHODS
        },
        "pilot_promotion_gate": {"strongest_benchmark": "independence"},
    }
    runner._validate_v2_selection(selection)
    selection["schema"] = "gse252762-celiac-source-selection/1.0"
    with pytest.raises(runner.ProtocolRefusal, match="v2 benchmark interface"):
        runner._validate_v2_selection(selection)
    selection["schema"] = "gse252762-celiac-source-selection/2.0"
    del selection["calibration_models"]["donor_stratified_ridge_poisson_field"]
    with pytest.raises(runner.ProtocolRefusal, match="v2 benchmark interface"):
        runner._validate_v2_selection(selection)


def test_claim_is_matrix_free_and_creates_only_public_attempt_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _redirect_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runner, "_verify_stage_prerequisites", lambda _stage: ({}, "a" * 40)
    )
    monkeypatch.setattr(runner, "_implementation_commit", lambda: "b" * 40)
    touched = []
    monkeypatch.setattr(
        runner,
        "_reduce_matrix",
        lambda *_args, **_kwargs: touched.append("matrix"),
    )
    payload = runner.claim("source")
    assert payload["status"] == "CLAIMED_BEFORE_MATRIX_ACCESS"
    assert runner.SOURCE_ATTEMPT.is_file()
    assert not runner.SOURCE_CONSUMPTION.exists()
    assert not runner.SOURCE_JOURNAL.exists()
    assert touched == []


def test_claim_refuses_any_existing_current_or_downstream_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _redirect_paths(monkeypatch, tmp_path)
    runner.PREDICTIONS.write_text("{}\n")
    with pytest.raises(runner.ProtocolRefusal, match="durable artifact"):
        runner.claim("source")
    assert not runner.SOURCE_ATTEMPT.exists()


def test_execution_cannot_consume_or_get_before_public_claim_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _redirect_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runner,
        "_verify_claim",
        lambda _stage: (_ for _ in ()).throw(runner.ProtocolRefusal("not public")),
    )
    touched = []
    monkeypatch.setattr(
        runner,
        "_reduce_matrix",
        lambda *_args, **_kwargs: touched.append("matrix"),
    )
    with pytest.raises(runner.ProtocolRefusal, match="not public"):
        runner.run_predict()
    assert not runner.PREDICTION_CONSUMPTION.exists()
    assert not runner.PREDICTIONS.exists()
    assert touched == []


def test_consumption_is_exclusive_and_precedes_access_journal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _redirect_paths(monkeypatch, tmp_path)
    runner.SOURCE_ATTEMPT.write_text('{"claim": true}\n')
    runner._begin_consumption("source", "c" * 40)
    assert runner.SOURCE_CONSUMPTION.is_file()
    assert not runner.SOURCE_JOURNAL.exists()
    with pytest.raises(runner.ProtocolRefusal, match="already consumed"):
        runner._begin_consumption("source", "c" * 40)


def test_active_stage_lease_refuses_a_concurrent_invocation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _redirect_paths(monkeypatch, tmp_path)
    preflight = _held_preflight(13)
    preflight.pop("_payload")
    _write_attempt_consumption_and_header("prediction")
    runner.PREDICTION_JOURNAL.unlink()
    consumption = runner._read_json(runner.PREDICTION_CONSUMPTION)
    monkeypatch.setattr(runner, "_preflight", lambda: preflight)
    monkeypatch.setattr(
        runner,
        "_verify_consumption_public",
        lambda _stage: ({}, "e" * 40, {}, consumption),
    )
    with runner._stage_access("prediction"):
        with pytest.raises(runner.ProtocolRefusal, match="already active"):
            with runner._stage_access("prediction"):
                pass


def test_inactive_or_wrong_stage_lease_blocks_before_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    preflight = _held_preflight(13)
    preflight.pop("_payload")
    touched = []
    monkeypatch.setattr(
        runner, "_open_url", lambda *_args, **_kwargs: touched.append("network")
    )
    lease = runner._AccessLease("prediction", "d" * 32, ((6, "rna"),), active=False)
    with pytest.raises(runner.ProtocolRefusal, match="active stage lease"):
        runner._reduce_matrix(
            preflight,
            6,
            "rna",
            preflight["samples"],
            tmp_path / "journal.jsonl",
            lease,
        )
    assert touched == []


def test_frozen_preflight_semantics_and_role_context_mutation() -> None:
    payload = runner._read_json(runner.PREFLIGHT)
    runner._validate_preflight_semantics(payload)
    changed = deepcopy(payload)
    changed["samples"][0]["context"] = "CONTROL"
    with pytest.raises(runner.ProtocolRefusal, match="sample selection"):
        runner._validate_preflight_semantics(changed)


def test_source_reduced_validator_binds_samples_tables_and_matrix_axes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    preflight = runner._read_json(runner.PREFLIGHT)
    _redirect_paths(monkeypatch, tmp_path)
    _write_attempt_consumption_and_header("source")
    source_samples = [
        sample
        for sample in preflight["samples"]
        if sample["role"] in {"calibration", "pilot"}
    ]
    source_samples.sort(
        key=lambda sample: (
            {"calibration": 0, "pilot": 1}[sample["role"]],
            sample["sample_id"],
        )
    )
    rna_counts = np.zeros((256, 9), dtype=np.int64)
    rna_counts[:128] = 1
    cite_counts = np.zeros((256, 9), dtype=np.int64)
    cite_counts[64:192] = 1
    records = []
    for sample in source_samples:
        truth, destroyed = runner.core.sample_tables(
            rna_counts,
            cite_counts,
            sample["selected_barcodes"],
            sample["sample_id"],
        )
        records.append(
            {
                "sample_id": sample["sample_id"],
                "role": sample["role"],
                "condition": sample["condition"],
                "context": sample["context"],
                "rna_counts": rna_counts.tolist(),
                "rna_counts_sha256": runner._array_sha256(rna_counts),
                "cite_counts": cite_counts.tolist(),
                "cite_counts_sha256": runner._array_sha256(cite_counts),
                "truth_tables": truth.tolist(),
                "destroyed_tables": destroyed.tolist(),
                "truth_sha256": runner._array_sha256(truth),
                "destroyed_sha256": runner._array_sha256(destroyed),
            }
        )
    record_by_id = {record["sample_id"]: record for record in records}
    audits = []
    for batch_number in range(1, 6):
        batch_samples = [
            sample for sample in source_samples if sample["batch"] == batch_number
        ]
        for modality in ("rna", "cite"):
            audit = _audit_certificate(preflight, batch_number, modality, batch_samples)
            count_field = f"{modality}_counts"
            selected_block = np.concatenate(
                [
                    np.asarray(
                        record_by_id[sample["sample_id"]][count_field],
                        dtype=np.int64,
                    ).T
                    for sample in batch_samples
                ],
                axis=1,
            )
            audit["selected_block_sha256"] = runner._array_sha256(selected_block)
            audit["selected_value_sum"] = int(selected_block.sum())
            audit["global_value_sum"] = int(selected_block.sum())
            audit["selected_entries"] = int(np.count_nonzero(selected_block))
            audit["selected_distinct_coordinates"] = int(
                np.count_nonzero(selected_block)
            )
            audit["declared_nnz"] = int(np.count_nonzero(selected_block))
            audit["parsed_nnz"] = int(np.count_nonzero(selected_block))
            audits.append({"batch": batch_number, "modality": modality, **audit})
    for index, audit in enumerate(audits):
        matrix = runner._matrix_record(
            runner._batch_record(preflight, audit["batch"]), audit["modality"]
        )
        event = {
            "batch": audit["batch"],
            "modality": audit["modality"],
            "url": matrix["url"],
            "execution_id": "d" * 32,
        }
        runner._append_journal(
            runner.SOURCE_JOURNAL, {**event, "event": "MATRIX_ACCESS_STARTED"}
        )
        audit_payload = {
            key: value
            for key, value in audit.items()
            if key not in {"batch", "modality"}
        }
        batch_samples = [
            sample for sample in source_samples if sample["batch"] == audit["batch"]
        ]
        selected_block = np.concatenate(
            [
                np.asarray(
                    record_by_id[sample["sample_id"]][f"{audit['modality']}_counts"],
                    dtype=np.int64,
                ).T
                for sample in batch_samples
            ],
            axis=1,
        )
        lease = runner._AccessLease("source", "d" * 32, (), position=index + 1)
        sidecar_path, sidecar = runner._sidecar_payload(
            preflight,
            lease,
            audit["batch"],
            audit["modality"],
            selected_block,
            audit_payload,
            runner.SOURCE_JOURNAL,
        )
        runner._write_json(sidecar_path, sidecar)
        runner._append_journal(
            runner.SOURCE_JOURNAL,
            {
                **event,
                "event": "MATRIX_REDUCTION_FINISHED",
                "audit": audit_payload,
                "selected_sidecar_path": runner._relative(sidecar_path),
                "selected_sidecar_sha256": runner._sha256(sidecar_path),
            },
        )
    reduced = {
        "schema": "gse252762-celiac-source-reduced/2.0",
        "status": "SOURCE_REDUCTION_COMPLETE",
        "samples": records,
        "matrix_audits": audits,
        "access_journal_path": runner._relative(runner.SOURCE_JOURNAL),
        "access_journal_sha256": runner._sha256(runner.SOURCE_JOURNAL),
    }
    runner._validate_source_reduced(reduced, preflight)
    changed = deepcopy(reduced)
    changed["samples"][0]["truth_tables"][0][0][0][0] += 1
    with pytest.raises(runner.ProtocolRefusal, match="table contract"):
        runner._validate_source_reduced(changed, preflight)


def test_prediction_payload_refits_all_sixteen_source_donors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _redirect_paths(monkeypatch, tmp_path)
    preflight = _held_preflight(13)
    preflight.pop("_payload")
    reduced = _source_reduced_records()
    runner.SOURCE_RESULT.write_text('{"status":"SOURCE_PASS"}\n')
    runner.PREDICTION_JOURNAL.write_text("{}\n")
    observed = {}

    def predict(
        source,
        destroyed,
        contexts,
        selection,
        rows,
        columns,
        held_contexts,
        *,
        return_fit_report,
    ):
        observed["source_count"] = len(source)
        observed["destroyed_count"] = len(destroyed)
        observed["context_count"] = len(contexts)
        assert selection["status"] == "PROMOTED"
        assert rows.shape == columns.shape == (13, 9, 9, 2)
        assert len(held_contexts) == 13
        assert return_fit_report is True
        predictions = {
            method: np.full((13, 9, 9, 2, 2), 64.0)
            for method in runner.core.MANDATORY_METHODS
        }
        return predictions, {
            "strongest_benchmark": "independence",
            "primary_fit_certificate": {"converged": True},
            "donor_stratified_ridge_poisson_certificate": {"status": "FINITE"},
        }

    monkeypatch.setattr(runner.core, "predict_from_source", predict)
    rna = np.concatenate(
        [np.full((9, 256), sample, dtype=np.int64) for sample in range(13)],
        axis=1,
    )
    payload = runner._prediction_payload(
        preflight,
        reduced,
        {"status": "PROMOTED", "strongest_benchmark": "independence"},
        rna,
        {"audit": True},
        "d" * 40,
    )
    assert observed == {
        "source_count": 16,
        "destroyed_count": 16,
        "context_count": 16,
    }
    assert len(payload["samples"]) == 13
    assert payload["samples"][0]["rna_counts"] != payload["samples"][-1]["rna_counts"]
    assert payload["all_source_fit_report"]["primary_fit_certificate"]["converged"]
    assert payload["held_cite_matrix_gets"] == 0


def test_prediction_stage_accesses_only_held_rna(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _redirect_paths(monkeypatch, tmp_path)
    preflight = _held_preflight(13)
    preflight.pop("_payload")
    runner.SOURCE_REDUCED.write_text(json.dumps(_source_reduced_records()))
    runner.SOURCE_RESULT.write_text('{"status":"SOURCE_PASS"}\n')
    monkeypatch.setattr(runner, "_preflight", lambda: preflight)
    monkeypatch.setattr(
        runner, "_replay_source_result", lambda *_args: {"status": "PROMOTED"}
    )
    monkeypatch.setattr(
        runner,
        "_prediction_payload",
        lambda *_args: {
            "status": "PREDICTIONS_FROZEN_BEFORE_HELD_CITE_ACCESS",
            "samples": [{}] * 13,
        },
    )
    seen = []

    @contextmanager
    def access(stage):
        assert stage == "prediction"
        runner.PREDICTION_JOURNAL.write_text("{}\n")
        yield (
            runner._AccessLease("prediction", "d" * 32, ((6, "rna"),)),
            {"public_source_result_commit": "d" * 40},
        )

    def reduce(_preflight, _batch, modality, samples, _journal, lease):
        seen.append(modality)
        runner._append_journal(_journal, {"event": "MATRIX_ACCESS_STARTED"})
        lease.position += 1
        counts = np.zeros((9, 256 * len(samples)), dtype=np.int64)
        audit = _audit_certificate(_preflight, 6, modality, samples)
        audit["selected_block_sha256"] = runner._array_sha256(counts)
        path, sidecar = runner._sidecar_payload(
            _preflight, lease, 6, modality, counts, audit, _journal
        )
        runner._write_json(path, sidecar)
        runner._append_journal(
            _journal,
            {
                "event": "MATRIX_REDUCTION_FINISHED",
                "selected_sidecar_path": runner._relative(path),
                "selected_sidecar_sha256": runner._sha256(path),
            },
        )
        return counts, audit

    monkeypatch.setattr(runner, "_stage_access", access)
    monkeypatch.setattr(runner, "_reduce_matrix", reduce)
    result = runner.run_predict()
    assert result["status"] == "PREDICTIONS_FROZEN_BEFORE_HELD_CITE_ACCESS"
    assert seen == ["rna"]


def test_score_precheck_failure_occurs_before_consumption_and_cite_get(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _redirect_paths(monkeypatch, tmp_path)
    touched = []
    monkeypatch.setattr(
        runner,
        "_verify_claim",
        lambda _stage: (_ for _ in ()).throw(
            runner.ProtocolRefusal("prediction replay mismatch")
        ),
    )
    monkeypatch.setattr(
        runner,
        "_reduce_matrix",
        lambda *_args, **_kwargs: touched.append("cite"),
    )
    with pytest.raises(runner.ProtocolRefusal, match="replay mismatch"):
        runner.run_score()
    assert not runner.SCORE_CONSUMPTION.exists()
    assert touched == []


def test_score_stage_accesses_only_held_cite_and_writes_terminal_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _redirect_paths(monkeypatch, tmp_path)
    preflight = _held_preflight(13)
    preflight.pop("_payload")
    predictions = _independence_predictions(preflight)
    source = {
        "source_selection": {"strongest_benchmark": "bias_reduced_context_poisson"}
    }
    runner.SOURCE_REDUCED.write_text("{}\n")
    runner.SOURCE_RESULT.write_text(json.dumps(source))
    runner.PREDICTIONS.write_text(json.dumps(predictions))
    monkeypatch.setattr(runner, "_preflight", lambda: preflight)
    monkeypatch.setattr(
        runner,
        "_replay_predictions",
        lambda *_args: {"primary": np.zeros((13, 9, 9, 2, 2))},
    )
    seen = []

    @contextmanager
    def access(stage):
        assert stage == "score"
        runner.SCORE_JOURNAL.write_text("{}\n")
        yield (
            runner._AccessLease("score", "d" * 32, ((6, "cite"),)),
            {
                "public_source_result_commit": "d" * 40,
                "public_predictions_commit": "e" * 40,
            },
        )

    def reduce(_preflight, _batch, modality, samples, _journal, lease):
        seen.append(modality)
        runner._append_journal(_journal, {"event": "MATRIX_ACCESS_STARTED"})
        lease.position += 1
        counts = np.tile(np.arange(256), (9, len(samples))).reshape(
            9, 256 * len(samples), order="F"
        )
        selected_entries = int(np.count_nonzero(counts))
        selected_value_sum = int(counts.sum(dtype=np.int64))
        audit = _audit_certificate(_preflight, 6, modality, samples)
        audit.update(
            {
                "declared_nnz": selected_entries,
                "parsed_nnz": selected_entries,
                "selected_entries": selected_entries,
                "selected_distinct_coordinates": selected_entries,
                "global_value_sum": selected_value_sum,
                "selected_value_sum": selected_value_sum,
                "selected_block_sha256": runner._array_sha256(counts),
            }
        )
        path, sidecar = runner._sidecar_payload(
            _preflight, lease, 6, modality, counts, audit, _journal
        )
        runner._write_json(path, sidecar)
        runner._append_journal(
            _journal,
            {
                "event": "MATRIX_REDUCTION_FINISHED",
                "selected_sidecar_path": runner._relative(path),
                "selected_sidecar_sha256": runner._sha256(path),
            },
        )
        return counts, audit

    monkeypatch.setattr(runner, "_stage_access", access)
    monkeypatch.setattr(runner, "_reduce_matrix", reduce)
    monkeypatch.setattr(
        runner, "_validate_access_journal", lambda *_args, **_kwargs: None
    )
    result = runner.run_score()
    assert seen == ["cite"]
    assert result["status"] == "CONFIRMATION_FAIL"
    assert len(result["truth_tables"]) == 13
    cite = np.asarray(result["cite_counts"], dtype=np.int64)
    assert cite.shape == (13, 256, 9)
    assert result["cite_counts_sha256"] == runner._array_sha256(cite)


def test_terminal_failure_has_stable_code_and_no_local_error_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _redirect_paths(monkeypatch, tmp_path)
    _write_attempt_consumption_and_header("source")
    payload = runner._failure(
        "source", "source_reduction", RuntimeError("/private/user/path")
    )
    assert payload["reason_code"] == "SOURCE_MATRIX_REDUCTION_FAILED"
    assert payload["exception_class"] == "RuntimeError"
    assert "message" not in payload
    assert "/private" not in json.dumps(payload)
    runner._validate_failure("source", payload)


@pytest.mark.parametrize(
    ("stage", "phase"),
    (
        ("source", "source_selection"),
        ("prediction", "held_prediction"),
        ("score", "held_scoring"),
    ),
)
def test_post_reduction_failure_requires_completed_reduction_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage: str,
    phase: str,
) -> None:
    _redirect_paths(monkeypatch, tmp_path)
    _write_attempt_consumption_and_header(stage)
    payload = runner._failure(stage, phase, RuntimeError("synthetic"))
    with pytest.raises(runner.ProtocolRefusal):
        runner._validate_failure(stage, payload)


def test_recover_refuses_a_malformed_existing_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _redirect_paths(monkeypatch, tmp_path)
    runner.SOURCE_RESULT.write_text("{}\n")
    with pytest.raises(runner.ProtocolRefusal, match="unknown status"):
        runner.recover("source")


def test_preflight_only_validation_is_not_a_complete_valid_campaign(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _redirect_paths(monkeypatch, tmp_path)
    preflight = _held_preflight(13)
    preflight.pop("_payload")
    monkeypatch.setattr(runner, "_preflight", lambda: preflight)
    report = runner.validate(require_public=False)
    assert report["preflight_valid"] is True
    assert report["artifact_checks"] == 0
    assert report["local_artifact_chain_valid"] is False
    assert report["campaign_complete"] is False
    assert report["valid"] is False


def test_strict_json_normalization_makes_tuple_intervals_replayable() -> None:
    assert runner._json_value({"interval": (-1.0, 0.0), "passes": np.bool_(False)}) == {
        "interval": [-1.0, 0.0],
        "passes": False,
    }
