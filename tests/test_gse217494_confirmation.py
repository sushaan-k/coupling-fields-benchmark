from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import confirm_gse217494_heart as confirmation


@pytest.fixture(autouse=True)
def _isolated_stage_locks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(confirmation, "STAGE_LOCK_DIRECTORY", tmp_path)


def _patch_stage_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    names = (
        "SOURCE_ATTEMPT",
        "SOURCE_CONSUMPTION",
        "SOURCE_ACCESS",
        "SOURCE_RESULT",
        "SCORE_AUTHORIZATION",
        "SCORE_ATTEMPT",
        "SCORE_CONSUMPTION",
        "SCORE_ACCESS",
        "SCORE_RESULT",
    )
    for name in names:
        suffix = ".jsonl" if name.endswith("ACCESS") else ".json"
        monkeypatch.setattr(confirmation, name, tmp_path / f"{name.lower()}{suffix}")


class _Response:
    def __init__(self, payload: bytes, url: str):
        self.payload = payload
        self.url = url
        self.position = 0
        self.status = 200
        self.headers: dict[str, str] = {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def getcode(self) -> int:
        return self.status

    def geturl(self) -> str:
        return self.url

    def read(self, count: int) -> bytes:
        block = self.payload[self.position : self.position + count]
        self.position += len(block)
        return block


def _journal_header(stage: str = "source") -> dict[str, object]:
    return {
        "schema": "gse217494-heart-access-journal/1.0",
        "stage": stage,
        "event": "OPENED_BEFORE_ASSAY_ACCESS",
        "created_at_utc": "2026-08-30T00:00:00Z",
        "runtime": confirmation.REQUIRED_RUNTIME,
        "one_streaming_get_per_matrix": True,
        "range_requests_permitted": False,
        "automatic_retries_permitted": False,
    }


def test_implementation_binding_covers_clarification_and_transitive_modules() -> None:
    required = {
        "docs/GSE217494_CARDIAC_CITESEQ_IMPLEMENTATION_CLARIFICATIONS_2026-08-30.md",
        "experiments/confirm_gse217494_heart.py",
        "experiments/gse217494_heart_core.py",
        "mapreg/__init__.py",
        "mapreg/classical_residuals.py",
        "mapreg/coupling_fields.py",
        "mapreg/structured_context_conditional.py",
        "mapreg/poisson_loglinear.py",
        "mapreg/streamed_gzip_matrix_market.py",
        "mapreg/context_conditional_coupling.py",
        "mapreg/factorial_coupling.py",
        "mapreg/heterogeneity_adaptive_coupling.py",
        "mapreg/common_effect_conditional.py",
        "mapreg/table_prediction.py",
    }
    assert required <= set(confirmation.IMPLEMENTATION_BINDINGS)
    assert len(confirmation.IMPLEMENTATION_BINDINGS) == len(
        set(confirmation.IMPLEMENTATION_BINDINGS)
    )
    assert all(
        (confirmation.ROOT / relative).is_file()
        for relative in confirmation.IMPLEMENTATION_BINDINGS
    )
    assert confirmation.REQUIRED_RUNTIME["thread_environment"] == {
        name: "1" for name in confirmation.THREAD_VARIABLES
    }


def test_final_boundary_common_effect_falls_back_to_an_eligible_classical() -> None:
    pearson = np.asarray([0.4, 0.5, 0.6])
    common = np.asarray([0.1, 0.1, 0.1])
    independence = np.asarray([0.8, 0.8, 0.8])

    selected, losses = confirmation._select_final_classical(
        pearson,
        common,
        independence,
        final_common_available=False,
    )

    assert selected.selected == "standardized_fixed_margin_pearson"
    assert selected.ineligible == ("exact_common_effect_conditional_field",)
    assert np.array_equal(losses, pearson)


def test_claim_source_creates_private_capability_and_no_get(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_stage_paths(monkeypatch, tmp_path)
    scratch = tmp_path / "scratch"
    token = tmp_path / "private" / "token.bin"
    runtime = confirmation.REQUIRED_RUNTIME
    monkeypatch.setattr(confirmation, "_require_runtime", lambda: runtime)
    monkeypatch.setattr(
        confirmation,
        "_verify_implementation_freeze",
        lambda: {"implementation_commit": "a" * 40},
    )
    monkeypatch.setattr(
        confirmation,
        "_attempt_bindings",
        lambda tags: {"implementation_commit": tags["implementation_commit"]},
    )
    monkeypatch.setattr(
        confirmation,
        "_contract",
        lambda: {
            "source": [{"matrix_bytes": 10}] * len(confirmation.SOURCE_ORDER),
            "held": [],
        },
    )
    monkeypatch.setattr(confirmation, "_cognates", lambda: [{}] * 249)
    monkeypatch.setattr(
        confirmation,
        "_open_url",
        lambda request: (_ for _ in ()).throw(AssertionError("GET during claim")),
    )

    result = confirmation.claim_source(claim_token=token, scratch=scratch)

    assert result["status"] == "CLAIMED_BEFORE_FIRST_SOURCE_MATRIX_GET"
    assert token.is_file() and token.stat().st_size == 32
    assert confirmation.SOURCE_ATTEMPT.is_file()
    assert confirmation._read_jsonl(confirmation.SOURCE_ACCESS) == [
        confirmation._access_header("source", result["created_at_utc"], runtime)
    ]
    with pytest.raises(FileExistsError):
        confirmation.claim_source(claim_token=tmp_path / "second", scratch=scratch)


def test_claim_token_must_not_be_inside_scratch(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    with pytest.raises(PermissionError, match="outside the scratch"):
        confirmation._require_token_outside_scratch(
            scratch / "private-token.bin", scratch
        )


def test_stage_lock_blocks_concurrent_run_or_recovery(tmp_path: Path) -> None:
    scratch_a = tmp_path / "scratch-a"
    scratch_b = tmp_path / "scratch-b"
    with confirmation._stage_lock(scratch_a, "source"):
        with pytest.raises(PermissionError, match="still active"):
            with confirmation._stage_lock(scratch_b, "source"):
                pass
    with confirmation._stage_lock(scratch_b, "source"):
        pass


def test_public_tag_requires_annotated_remote_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_git(*arguments, **_):
        if arguments[:2] == ("cat-file", "-t"):
            return SimpleNamespace(stdout="tag\n", returncode=0)
        if arguments[:2] == ("rev-parse", "refs/tags/example"):
            return SimpleNamespace(stdout="1" * 40 + "\n", returncode=0)
        if arguments[:2] == ("rev-parse", "example^{}"):
            return SimpleNamespace(stdout="2" * 40 + "\n", returncode=0)
        raise AssertionError(arguments)

    monkeypatch.setattr(confirmation, "_git", fake_git)
    monkeypatch.setattr(
        confirmation, "_remote_tag_ids", lambda tag: ("9" * 40, "2" * 40)
    )
    with pytest.raises(PermissionError, match="differs"):
        confirmation._require_public_tag("example", ())


def test_download_is_one_streaming_get_with_independent_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = b"compressed bytes" * 100
    url = (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE217nnn/"
        "GSE217494/suppl/GSE217494_sample2.matrix.mtx.gz"
    )
    calls = []
    monkeypatch.setattr(
        confirmation,
        "_open_url",
        lambda request: calls.append(request) or _Response(payload, url),
    )
    journal = tmp_path / "access.jsonl"
    confirmation._append_jsonl(journal, _journal_header(), create=True)
    destination = tmp_path / "matrix.mtx.gz"

    record = confirmation._download(
        stage="source",
        sample="sample2",
        kind="matrix",
        url=url,
        destination=destination,
        journal=journal,
        expected_bytes=len(payload),
        expected_sha256=None,
    )

    assert len(calls) == 1
    assert calls[0].get_header("Range") is None
    assert calls[0].get_method() == "GET"
    assert record == {
        "observed_bytes": len(payload),
        "observed_sha256": hashlib.sha256(payload).hexdigest(),
    }
    assert destination.read_bytes() == payload
    events = confirmation._read_jsonl(journal)
    assert [row["event"] for row in events[-2:]] == [
        "REQUEST_STARTED",
        "REQUEST_FINISHED",
    ]
    assert events[-2]["automatic_retry_count"] == 0
    assert events[-2]["range_header"] is None


def test_download_failure_is_not_retried_and_is_durably_journaled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    url = (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE217nnn/"
        "GSE217494/suppl/GSE217494_sample2.matrix.mtx.gz"
    )
    calls = 0

    def fail(_):
        nonlocal calls
        calls += 1
        raise OSError("network text must not enter public result")

    monkeypatch.setattr(confirmation, "_open_url", fail)
    journal = tmp_path / "access.jsonl"
    confirmation._append_jsonl(journal, _journal_header(), create=True)
    with pytest.raises(OSError):
        confirmation._download(
            stage="source",
            sample="sample2",
            kind="matrix",
            url=url,
            destination=tmp_path / "matrix.gz",
            journal=journal,
            expected_bytes=1,
            expected_sha256=None,
        )
    assert calls == 1
    failure = confirmation._read_jsonl(journal)[-1]
    assert failure["event"] == "REQUEST_FAILED"
    assert failure["reason_code"] == "UNEXPECTED_EXCEPTION"
    assert "OSError" not in journal.read_text(encoding="utf-8")


def _write_success_journal(path: Path, samples: tuple[str, ...]) -> None:
    contract = confirmation._contract()
    by_sample = {
        record["sample"]: record
        for record in (*contract["source"], *contract["held"])
    }
    confirmation._append_jsonl(path, _journal_header(), create=True)
    confirmation._journal_event(
        path,
        "source",
        "CONSUMPTION_COMMITTED",
        consumption_sha256="c" * 64,
    )
    confirmation._journal_event(path, "source", "PRIVATE_CAPABILITY_CONSUMED")
    for sample in samples:
        for kind in ("features", "barcodes", "matrix"):
            record = by_sample[sample]
            if kind == "features":
                expected_bytes = record["feature_axis"]["bytes"]
                expected_sha256 = record["feature_axis"]["gzip_sha256"]
                suffix = "features.tsv.gz"
            elif kind == "barcodes":
                expected_bytes = record["barcode_axis"]["bytes"]
                expected_sha256 = record["barcode_axis"]["gzip_sha256"]
                suffix = "barcodes.tsv.gz"
            else:
                expected_bytes = record["matrix_bytes"]
                expected_sha256 = None
                suffix = "matrix.mtx.gz"
            observed_sha256 = expected_sha256 or "a" * 64
            base = {
                "sample": sample,
                "kind": kind,
                "filename": f"GSE217494_{sample}.{suffix}",
            }
            confirmation._journal_event(
                path,
                "source",
                "REQUEST_STARTED",
                **base,
                url=record["urls"][kind],
                expected_bytes=expected_bytes,
                expected_sha256=expected_sha256,
                method="GET",
                range_header=None,
                automatic_retry_count=0,
                streaming=True,
            )
            confirmation._journal_event(
                path,
                "source",
                "REQUEST_FINISHED",
                **base,
                observed_bytes=expected_bytes,
                observed_sha256=observed_sha256,
                status_code=200,
                response_url=record["urls"][kind],
            )
            if kind == "matrix":
                confirmation._journal_event(
                    path,
                    "source",
                    "MATRIX_PARSE_FINISHED",
                    sample=sample,
                    kind=kind,
                    parser_audit={
                        "banner": "%%MatrixMarket matrix coordinate integer general",
                        "matrix_shape": [
                            confirmation.FEATURE_COUNT,
                            record["barcode_axis"]["count"],
                        ],
                        "gzip_stream_exhausted": True,
                        "declared_nnz": 10,
                        "parsed_nnz": 10,
                        "decompressed_bytes": 100,
                        "decompressed_sha256": "b" * 64,
                    },
                    independent_compressed_bytes=expected_bytes,
                    independent_compressed_sha256=observed_sha256,
                )
            confirmation._journal_event(
                path,
                "source",
                "DOWNLOAD_DELETED",
                **base,
                existed_before_delete=True,
                deleted=True,
            )


def test_success_audit_enforces_order_deletion_and_zero_failures(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "success.jsonl"
    _write_success_journal(journal, ("sample2", "sample4"))
    audit = confirmation._success_access_audit(
        stage="source", journal=journal, samples=("sample2", "sample4")
    )
    assert audit["matrix_streaming_get_count"] == 2
    assert audit["all_downloads_deleted"] is True

    failed = tmp_path / "failed.jsonl"
    _write_success_journal(failed, ("sample2",))
    confirmation._journal_event(
        failed,
        "source",
        "REQUEST_FAILED",
        sample="sample2",
        kind="matrix",
    )
    with pytest.raises(PermissionError, match="failed request"):
        confirmation._success_access_audit(
            stage="source", journal=failed, samples=("sample2",)
        )


def test_consumption_is_written_before_private_token_unlink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_stage_paths(monkeypatch, tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    token = tmp_path / "token.bin"
    token.write_bytes(b"x" * 32)
    claim_hash = hashlib.sha256(b"x" * 32).hexdigest()
    attempt = {
        "bindings": {},
        "claim_token_sha256": claim_hash,
    }
    confirmation.SOURCE_ATTEMPT.write_text("{}\n", encoding="utf-8")
    confirmation._append_jsonl(
        confirmation.SOURCE_ACCESS, _journal_header(), create=True
    )
    monkeypatch.setattr(
        confirmation, "_validate_source_attempt", lambda: (attempt, "c" * 40)
    )
    monkeypatch.setattr(confirmation, "_prepare_scratch", lambda path: scratch)
    monkeypatch.setattr(
        confirmation, "_require_runtime", lambda: confirmation.REQUIRED_RUNTIME
    )
    monkeypatch.setattr(
        confirmation,
        "_contract",
        lambda: {"source": [{"sample": "sample2"}]},
    )
    monkeypatch.setattr(confirmation, "_cognates", lambda: [])
    monkeypatch.setattr(
        confirmation,
        "_reduce_sample",
        lambda *args, **kwargs: {"public_record": {}, "etiology": "Donor"},
    )
    monkeypatch.setattr(
        confirmation,
        "_fit_source_models",
        lambda records, cognates: {"source_gate": {"passes": True}},
    )
    monkeypatch.setattr(
        confirmation,
        "_success_access_audit",
        lambda **kwargs: {"matrix_streaming_get_count": 1},
    )
    order = []
    original_write = confirmation._write_json_x

    def observed_write(path, payload):
        if path == confirmation.SOURCE_CONSUMPTION:
            order.append("consumption")
        original_write(path, payload)

    original_consume = confirmation._consume_token

    def observed_consume(path):
        assert confirmation.SOURCE_CONSUMPTION.exists()
        order.append("token")
        original_consume(path)

    monkeypatch.setattr(confirmation, "_write_json_x", observed_write)
    monkeypatch.setattr(confirmation, "_consume_token", observed_consume)

    result = confirmation.run_source(claim_token=token, scratch=scratch)

    assert order[:2] == ["consumption", "token"]
    assert result["status"] == "SOURCE_PROMOTED"
    assert not token.exists()


def _synthetic_fold_panel() -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    marker_count = 12
    symbols = [f"M{marker:02d}" for marker in range(marker_count)]
    cognates = [{"symbol": symbol} for symbol in symbols]
    cells = np.arange(confirmation.CELL_BUDGET)[:, None]
    markers = np.arange(marker_count)[None, :]
    records: list[dict[str, object]] = []
    for donor in range(14):
        positive = 160 + ((donor * 17 + np.arange(marker_count) * 11) % 160)
        rna = (
            (cells + donor * 7 + markers * 29) % confirmation.CELL_BUDGET
            < positive[None, :]
        ).astype(np.int64)
        adt_state = (
            (cells + donor * 19 + markers * 23) % confirmation.CELL_BUDGET
            < confirmation.CELL_BUDGET // 2
        ).astype(np.int64)
        adt = adt_state * (np.arange(marker_count)[None, :] + 2) + cells % 3
        destroyed_adt = np.roll(adt, confirmation.CELL_BUDGET // 2, axis=0)
        destroyed_state = np.roll(
            adt_state, confirmation.CELL_BUDGET // 2, axis=0
        )
        records.append(
            {
                "sample": f"sample-{donor}",
                "etiology": confirmation.ETIOLOGIES[donor % 4],
                "rna_counts_private": rna,
                "adt_counts_private": adt,
                "destroyed_adt_counts_private": destroyed_adt,
                "rna_profile": rna.mean(axis=0),
                "adt_profile": confirmation.adt_mean_profile(adt),
                "destroyed_adt_profile": confirmation.adt_mean_profile(
                    destroyed_adt
                ),
                "tables": confirmation.joint_binary_tables(rna, adt_state),
                "destroyed_tables": confirmation.joint_binary_tables(
                    rna, destroyed_state
                ),
            }
        )
    return records, cognates


def test_fold_artifacts_are_invariant_to_every_validation_assay_value() -> None:
    records, cognates = _synthetic_fold_panel()
    validation = 6
    before = confirmation._source_fold_artifacts(records, cognates, validation)
    protected = {
        "training_samples": [record["sample"] for record in before["training"]],
        "training_labels": before["training_labels"],
        "selected": before["selected"],
        "rna_graph": confirmation._array_sha256(before["rna_graph"].adjacency),
        "adt_graph": confirmation._array_sha256(before["adt_graph"].adjacency),
        "product": confirmation._array_sha256(before["product_laplacian"]),
        "destroyed_product": confirmation._array_sha256(
            before["destroyed_laplacian"]
        ),
        "training_tables": confirmation._array_sha256(before["training_tables"]),
        "shifted_tables": confirmation._array_sha256(before["shifted_tables"]),
        "invariance": before["destroyed_invariance"],
    }

    held = records[validation]
    for key in (
        "rna_counts_private",
        "adt_counts_private",
        "destroyed_adt_counts_private",
        "rna_profile",
        "adt_profile",
        "destroyed_adt_profile",
    ):
        np.asarray(held[key])[...] = 0
    np.asarray(held["tables"])[...] = 0
    np.asarray(held["tables"])[..., 0, 0] = confirmation.CELL_BUDGET
    np.asarray(held["destroyed_tables"])[...] = 0
    np.asarray(held["destroyed_tables"])[..., 1, 1] = confirmation.CELL_BUDGET

    after = confirmation._source_fold_artifacts(records, cognates, validation)
    observed = {
        "training_samples": [record["sample"] for record in after["training"]],
        "training_labels": after["training_labels"],
        "selected": after["selected"],
        "rna_graph": confirmation._array_sha256(after["rna_graph"].adjacency),
        "adt_graph": confirmation._array_sha256(after["adt_graph"].adjacency),
        "product": confirmation._array_sha256(after["product_laplacian"]),
        "destroyed_product": confirmation._array_sha256(
            after["destroyed_laplacian"]
        ),
        "training_tables": confirmation._array_sha256(after["training_tables"]),
        "shifted_tables": confirmation._array_sha256(after["shifted_tables"]),
        "invariance": after["destroyed_invariance"],
    }
    assert observed == protected
    selected = after["selected_indices"]
    expected_truth = np.asarray(held["tables"])[selected][:, selected]
    destroyed_truth = np.asarray(held["destroyed_tables"])[selected][:, selected]
    assert np.array_equal(after["truth"], expected_truth)
    assert not np.array_equal(after["truth"], destroyed_truth)


def test_source_gate_requires_all_four_comparators_and_fifteen_support_counts() -> None:
    primary = np.full(14, 0.5)
    comparators = {
        name: np.full(14, 1.0) for name in confirmation.MANDATORY_COMPARATORS
    }
    labels = ["Donor"] * 4 + ["AMI"] * 2 + ["ICM"] * 4 + ["NICM"] * 4
    gate = confirmation.evaluate_source_gate(
        primary,
        comparators,
        labels,
        [9] * 15,
        all_reductions_and_fits_complete=True,
    )
    assert gate["passes"] is True
    with pytest.raises(ValueError, match="missing mandatory"):
        confirmation.evaluate_source_gate(
            primary,
            {
                key: value
                for key, value in comparators.items()
                if key != "destroyed_links"
            },
            labels,
            [9] * 15,
            all_reductions_and_fits_complete=True,
        )
    with pytest.raises(ValueError, match="14 folds"):
        confirmation.evaluate_source_gate(
            primary,
            comparators,
            labels,
            [9] * 14,
            all_reductions_and_fits_complete=True,
        )


def test_recovery_never_publishes_scratch_names_or_reopens_remote(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "GSE217494_sample2.matrix.mtx.gz").write_bytes(b"partial")
    (scratch / "private-secret-name.txt").write_text("keep", encoding="utf-8")
    attempt = tmp_path / "attempt.json"
    consumption = tmp_path / "consumption.json"
    journal = tmp_path / "access.jsonl"
    output = tmp_path / "result.json"
    attempt.write_text("{}\n", encoding="utf-8")
    confirmation._write_json_x(
        consumption,
        confirmation._consumption_payload(
            "source", attempt, confirmation.REQUIRED_RUNTIME, scratch
        ),
    )
    confirmation._append_jsonl(journal, _journal_header(), create=True)
    monkeypatch.setattr(
        confirmation,
        "_open_url",
        lambda request: (_ for _ in ()).throw(AssertionError("recovery issued GET")),
    )

    result = confirmation._recover(
        stage="source",
        scratch=scratch,
        attempt_path=attempt,
        consumption_path=consumption,
        journal_path=journal,
        result_path=output,
        schema="test/1.0",
        bindings={},
    )

    serialized = output.read_text(encoding="utf-8")
    assert result["refusal_code"] == "SCRATCH_STATE_FAILURE"
    assert "private-secret-name" not in serialized
    assert not (scratch / "GSE217494_sample2.matrix.mtx.gz").exists()
    assert (scratch / "private-secret-name.txt").exists()


def test_recovery_refuses_a_different_scratch_identity(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt.json"
    attempt.write_text("{}\n", encoding="utf-8")
    actual_scratch = tmp_path / "actual-scratch"
    wrong_scratch = tmp_path / "wrong-scratch"
    consumption = tmp_path / "consumption.json"
    confirmation._write_json_x(
        consumption,
        confirmation._consumption_payload(
            "source", attempt, confirmation.REQUIRED_RUNTIME, actual_scratch
        ),
    )
    journal = tmp_path / "access.jsonl"
    confirmation._append_jsonl(journal, _journal_header(), create=True)
    with pytest.raises(PermissionError, match="scratch identity differs"):
        confirmation._recover(
            stage="source",
            scratch=wrong_scratch,
            attempt_path=attempt,
            consumption_path=consumption,
            journal_path=journal,
            result_path=tmp_path / "result.json",
            schema="test/1.0",
            bindings={},
        )


def test_unexpected_failure_uses_fixed_code_and_omits_exception_text(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "access.jsonl"
    confirmation._append_jsonl(journal, _journal_header(), create=True)
    result = confirmation._public_failure(
        stage="source",
        schema="test/1.0",
        error=ValueError("/private/secret/path"),
        bindings={},
        journal=journal,
    )
    serialized = json.dumps(result)
    assert result["refusal_code"] == "UNEXPECTED_EXCEPTION"
    assert "ValueError" not in serialized
    assert "secret" not in serialized


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (confirmation.MarkerSupportRefusal("support"), "MARKER_SUPPORT_REFUSAL"),
        (
            confirmation.NoCompleteConfigurationError("grid"),
            "NO_COMPLETE_SOURCE_CONFIGURATION",
        ),
    ],
)
def test_expected_source_refusals_keep_scientific_codes(
    tmp_path: Path, error: Exception, expected_code: str
) -> None:
    journal = tmp_path / "access.jsonl"
    confirmation._append_jsonl(journal, _journal_header(), create=True)
    result = confirmation._public_failure(
        stage="source",
        schema="test/1.0",
        error=error,
        bindings={},
        journal=journal,
    )
    assert result["refusal_code"] == expected_code


def _model(values: np.ndarray) -> dict[str, object]:
    return confirmation._model_array(np.asarray(values, dtype=float))


def _zero_source_models(marker_count: int) -> dict[str, object]:
    zero_context = np.zeros((4, marker_count, marker_count))
    zero_field = np.zeros((marker_count, marker_count))
    base = {
        "donor_deviation_penalty": 0.3,
        "coefficient_ridge_penalty": 0.1,
        "transport_multiplier": 1.0,
    }
    return {
        "primary": {
            "coefficient": _model(zero_context),
            "configuration": {**base, "graph_penalty": 0.1},
        },
        "graph_zero": {
            "coefficient": _model(zero_context),
            "configuration": {**base, "graph_penalty": 0.0},
        },
        "destroyed_links": {
            "coefficient": _model(zero_context),
            "configuration": {**base, "graph_penalty": 0.1},
        },
        "pooled_fixed_interaction_poisson": {
            "log_odds": _model(zero_field),
            "transport_multiplier": 1.0,
        },
        "etiology_specific_fixed_interaction_poisson": {
            "log_odds": _model(np.zeros((4, marker_count, marker_count))),
            "transport_multiplier": 1.0,
        },
        "standardized_fixed_margin_pearson": {
            "coordinate": _model(np.zeros((4, marker_count, marker_count))),
            "transport_multiplier": 1.0,
        },
        "exact_common_effect_conditional_field": None,
        "fixed_margin_independence": {
            "kind": "recipient_fixed_margin_independence"
        },
    }


def test_all_models_return_unique_table_for_degenerate_recipient_margins() -> None:
    source = {"models": _zero_source_models(1)}
    truth = np.asarray([[[[0, 0], [256, 256]]]], dtype=np.int64)
    predictions, diagnostics = confirmation._model_predictions(
        source, {"tables": truth, "etiology": "Donor"}
    )
    assert all(np.array_equal(prediction, truth) for prediction in predictions.values())
    for value in diagnostics.values():
        assert value["informative_margin_count"] == 0
        assert value["degenerate_margin_count"] == 1
        assert value["reconstructed_log_odds"] == [[None]]
    assert "NaN" not in json.dumps(diagnostics, allow_nan=False)


def test_predictions_depend_only_on_sealed_recipient_margins() -> None:
    marker_count = 2
    first = np.empty((marker_count, marker_count, 2, 2), dtype=np.int64)
    second = np.empty_like(first)
    for rna in range(marker_count):
        for protein in range(marker_count):
            row = 180 + 20 * rna
            column = 220 + 15 * protein
            for output, n11 in ((first, 90 + rna + protein), (second, 110 + rna + protein)):
                output[rna, protein] = np.asarray(
                    [
                        [confirmation.CELL_BUDGET - row - column + n11, column - n11],
                        [row - n11, n11],
                    ]
                )
    source = {"models": _zero_source_models(marker_count)}
    first_predictions, _ = confirmation._model_predictions(
        source, {"tables": first, "etiology": "ICM"}
    )
    second_predictions, _ = confirmation._model_predictions(
        source, {"tables": second, "etiology": "ICM"}
    )
    assert set(first_predictions) == set(second_predictions)
    assert all(
        np.array_equal(first_predictions[name], second_predictions[name])
        for name in first_predictions
    )


def test_secondary_module_tests_use_all_four_comparators_and_one_bh_family() -> None:
    primary = np.asarray([0.1] * 8)
    module_losses = {
        "endothelial": {
            "primary": primary,
            **{
                comparator: np.asarray([0.2] * 8)
                for comparator in confirmation.MANDATORY_COMPARATORS
            },
        },
        "myeloid": {
            "primary": primary,
            **{
                comparator: np.asarray([0.3] * 8)
                for comparator in confirmation.MANDATORY_COMPARATORS
            },
        },
    }
    labels = ["Donor", "Donor", "AMI", "AMI", "ICM", "ICM", "NICM", "NICM"]
    result = confirmation._module_results(module_losses, labels)
    q_values = [
        comparison["benjamini_hochberg_q"]
        for module in result.values()
        for comparison in module["comparisons"].values()
    ]
    assert all(
        set(module["comparisons"]) == set(confirmation.MANDATORY_COMPARATORS)
        for module in result.values()
    )
    assert len(q_values) == 8
    assert all(0.0 <= value <= 1.0 for value in q_values)


def test_eight_heart_score_smoke_reaches_every_frozen_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_stage_paths(monkeypatch, tmp_path)
    marker_count = 4
    symbols = [f"M{index}" for index in range(marker_count)]
    source = {
        "selected_marker_source_indices": list(range(marker_count)),
        "selected_symbols": symbols,
        "models": _zero_source_models(marker_count),
        "source_cross_validation": {
            "strongest_remaining_classical": {
                "selected": "fixed_margin_independence"
            },
            "selected_losses": {
                "primary": [0.2] * 14,
                "graph_zero": [0.3] * 14,
            },
        },
        "secondary_module_models": {},
    }
    etiologies = [
        "Donor",
        "Donor",
        "AMI",
        "AMI",
        "ICM",
        "ICM",
        "NICM",
        "NICM",
    ]
    samples = [
        {"sample": f"held-{index}", "etiology": etiology}
        for index, etiology in enumerate(etiologies)
    ]
    records = {}
    for donor, sample in enumerate(samples):
        tables = np.empty((marker_count, marker_count, 2, 2), dtype=np.int64)
        for rna in range(marker_count):
            for protein in range(marker_count):
                row = 170 + 12 * rna
                column = 210 + 9 * protein
                n11 = int(round(row * column / confirmation.CELL_BUDGET)) + 12 + donor % 3
                tables[rna, protein] = np.asarray(
                    [
                        [confirmation.CELL_BUDGET - row - column + n11, column - n11],
                        [row - n11, n11],
                    ]
                )
        records[sample["sample"]] = {
            "sample": sample["sample"],
            "etiology": sample["etiology"],
            "tables": tables,
            "public_record": {"sample": sample["sample"]},
        }
    token_payload = b"t" * 32
    token = tmp_path / "token.bin"
    token.write_bytes(token_payload)
    attempt = {
        "bindings": {"frozen": True},
        "claim_token_sha256": hashlib.sha256(token_payload).hexdigest(),
    }
    confirmation.SCORE_ATTEMPT.write_text("{}\n", encoding="utf-8")
    confirmation._append_jsonl(
        confirmation.SCORE_ACCESS, _journal_header("score"), create=True
    )
    monkeypatch.setattr(
        confirmation,
        "_validate_score_attempt",
        lambda: (attempt, source, "a" * 40),
    )
    monkeypatch.setattr(
        confirmation, "_require_runtime", lambda: confirmation.REQUIRED_RUNTIME
    )
    monkeypatch.setattr(
        confirmation, "_contract", lambda: {"held": samples, "source": []}
    )
    monkeypatch.setattr(
        confirmation,
        "_cognates",
        lambda: [{"symbol": symbol} for symbol in symbols],
    )
    monkeypatch.setattr(
        confirmation,
        "_reduce_sample",
        lambda sample, **_: records[sample["sample"]],
    )
    monkeypatch.setattr(
        confirmation,
        "_success_access_audit",
        lambda **_: {
            "journal_sha256": "f" * 64,
            "request_count": 24,
            "matrix_streaming_get_count": 8,
            "range_request_count": 0,
            "automatic_retry_count": 0,
            "all_downloads_deleted": True,
        },
    )

    result = confirmation.run_score(
        claim_token=token, scratch=tmp_path / "score-scratch"
    )

    assert result["status"] == "CONFIRMATION_FAIL"
    assert set(confirmation.MANDATORY_COMPARATORS) <= set(result["donor_losses"])
    assert all(
        len(result["donor_losses"][name]) == 8
        for name in confirmation.MANDATORY_COMPARATORS
    )
    assert result["graph_specific_gain"]["bootstrap_draws"] == 20_000
    assert result["secondary_modules"] == {}
    assert len(result["non_evaluable_modules"]) == 3
    assert (
        result["exploratory_relational_summary"]["permutations"]
        == confirmation.NEIGHBOR_PERMUTATIONS
    )
    assert len(result["prediction_diagnostics"]) == 8
    assert "NaN" not in json.dumps(result, allow_nan=False)
    assert confirmation.SCORE_RESULT.is_file()
    assert not token.exists()


def test_claim_score_cannot_open_held_matrix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_stage_paths(monkeypatch, tmp_path)
    source = {"runtime": confirmation.REQUIRED_RUNTIME}
    authorization = {"held_matrix_access_authorized": True}
    monkeypatch.setattr(
        confirmation,
        "_validate_source_result",
        lambda require_public=True: (source, "s" * 40),
    )
    monkeypatch.setattr(
        confirmation,
        "_validate_score_authorization",
        lambda source, commit: (authorization, "a" * 40),
    )
    monkeypatch.setattr(
        confirmation,
        "_source_field_manifest",
        lambda source: {"field": "f" * 64},
    )
    monkeypatch.setattr(confirmation, "_binding_hashes", lambda: {})
    monkeypatch.setattr(
        confirmation, "_require_runtime", lambda: confirmation.REQUIRED_RUNTIME
    )
    monkeypatch.setattr(
        confirmation,
        "_open_url",
        lambda request: (_ for _ in ()).throw(AssertionError("GET during claim")),
    )
    confirmation.SCORE_AUTHORIZATION.write_text("{}\n", encoding="utf-8")
    confirmation.SOURCE_RESULT.parent.mkdir(parents=True, exist_ok=True)
    confirmation.SOURCE_RESULT.write_text("{}\n", encoding="utf-8")

    result = confirmation.claim_score(
        claim_token=tmp_path / "token", scratch=tmp_path / "scratch"
    )

    assert result["status"].startswith("CLAIMED_AFTER_AUTHORIZATION")
    assert result["held_matrix_access_authorized"] is True
    assert confirmation.SCORE_ACCESS.exists()


def test_cli_exposes_only_frozen_stage_commands() -> None:
    parser = confirmation._parser()
    subparsers_action = next(
        action
        for action in parser._actions
        if hasattr(action, "choices") and isinstance(action.choices, dict)
    )
    assert set(subparsers_action.choices) == {
        "claim-source",
        "source",
        "recover-source",
        "authorize-score",
        "claim-score",
        "score",
        "recover-score",
        "validate-source",
        "validate-score",
    }


def test_gzip_axis_validation_binds_decompressed_bytes_and_count(
    tmp_path: Path,
) -> None:
    raw = b"a\nb\n"
    path = tmp_path / "axis.tsv.gz"
    path.write_bytes(gzip.compress(raw))
    decoded, values = confirmation._decode_gzip_axis(
        path,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        expected_count=2,
    )
    assert decoded == raw
    assert values == ["a", "b"]
    with pytest.raises(confirmation.ProtocolRefusal):
        confirmation._decode_gzip_axis(path, expected_sha256="0" * 64, expected_count=2)


def test_synthetic_mex_reduction_uses_bound_axes_and_streamed_matrix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rna_features = [
        "rna0\tM00\tGene Expression",
        "rna1\tOTHER\tGene Expression",
    ]
    adt_features = ["adt0\tM00\tAntibody Capture"] + [
        f"adt{index}\tP{index:03d}\tAntibody Capture"
        for index in range(1, confirmation.ADT_FEATURE_COUNT)
    ]
    feature_raw = ("\n".join(rna_features + adt_features) + "\n").encode()
    barcodes = [f"BC{index:04d}-1" for index in range(confirmation.CELL_BUDGET)]
    barcode_raw = ("\n".join(barcodes) + "\n").encode()
    entries = []
    entries.extend((1, column, 1) for column in range(1, 201))
    entries.extend(
        (3, column, 1 + column % 5)
        for column in range(1, confirmation.CELL_BUDGET + 1)
    )
    entries.extend(
        (4, column, 1) for column in range(1, confirmation.CELL_BUDGET + 1)
    )
    matrix_raw = (
        "%%MatrixMarket matrix coordinate integer general\n"
        "% synthetic reducer fixture\n"
        f"281 {confirmation.CELL_BUDGET} {len(entries)}\n"
        + "".join(f"{row} {column} {value}\n" for row, column, value in entries)
    ).encode()
    payloads = {
        "features": gzip.compress(feature_raw),
        "barcodes": gzip.compress(barcode_raw),
        "matrix": gzip.compress(matrix_raw),
    }

    def fake_download(*, kind, destination, **_):
        payload = payloads[kind]
        destination.write_bytes(payload)
        return {
            "observed_bytes": len(payload),
            "observed_sha256": hashlib.sha256(payload).hexdigest(),
        }

    monkeypatch.setattr(confirmation, "RNA_FEATURE_COUNT", 2)
    monkeypatch.setattr(confirmation, "FEATURE_COUNT", 281)
    monkeypatch.setattr(confirmation, "_download", fake_download)
    sample = {
        "sample": "sample2",
        "etiology": "Donor",
        "matrix_bytes": len(payloads["matrix"]),
        "urls": {kind: f"https://example.invalid/{kind}" for kind in payloads},
        "feature_axis": {
            "bytes": len(payloads["features"]),
            "gzip_sha256": hashlib.sha256(payloads["features"]).hexdigest(),
            "axis_sha256": hashlib.sha256(feature_raw).hexdigest(),
        },
        "barcode_axis": {
            "bytes": len(payloads["barcodes"]),
            "gzip_sha256": hashlib.sha256(payloads["barcodes"]).hexdigest(),
            "axis_sha256": hashlib.sha256(barcode_raw).hexdigest(),
            "count": confirmation.CELL_BUDGET,
        },
    }
    cognates = [
        {
            "symbol": "M00",
            "rna_row_1based": 1,
            "rna_feature_id": "rna0",
            "adt_row_1based": 3,
            "adt_feature_id": "adt0",
        }
    ]
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    journal = tmp_path / "access.jsonl"
    confirmation._append_jsonl(journal, _journal_header(), create=True)

    record = confirmation._reduce_sample(
        sample,
        stage="source",
        scratch=scratch,
        journal=journal,
        cognates=cognates,
    )

    assert record["tables"].shape == (1, 1, 2, 2)
    assert int(record["tables"].sum()) == confirmation.CELL_BUDGET
    assert record["rna_counts_private"].shape == (confirmation.CELL_BUDGET, 1)
    assert record["all_adt_counts_private"].shape == (
        confirmation.CELL_BUDGET,
        confirmation.ADT_FEATURE_COUNT,
    )
    assert not any(scratch.iterdir())
    parse = [
        row
        for row in confirmation._read_jsonl(journal)
        if row.get("event") == "MATRIX_PARSE_FINISHED"
    ]
    assert len(parse) == 1
    assert parse[0]["parser_audit"]["gzip_stream_exhausted"] is True
    assert parse[0]["parser_audit"]["declared_nnz"] == len(entries)
