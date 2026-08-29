from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import fsspec
import h5py
import numpy as np
import pytest

from experiments import confirm_stephenson_unused_cambridge as campaign


def _records() -> list[dict[str, object]]:
    return [
        {
            "donor": donor,
            "sample": sample,
            "site": "Cambridge",
            "role": "unused_source",
        }
        for donor, sample in campaign.UNUSED
    ]


def _selections() -> dict[str, dict[str, object]]:
    barcodes = np.asarray([f"cell-{index}" for index in range(512)])
    return {
        sample: {
            "rows": np.arange(512),
            "barcodes": barcodes,
            "selected_barcode_sha256": hashlib.sha256(sample.encode()).hexdigest(),
        }
        for _, sample in campaign.UNUSED
    }


def _independence_models() -> dict[str, dict[str, str]]:
    return {
        name: {"kind": "independence", "estimator": "test"} for name in campaign.METHODS
    }


def _patch_prediction_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path, Path, Path]:
    authorization = tmp_path / "prediction_authorization.json"
    attempt = tmp_path / "prediction_attempt.json"
    output = tmp_path / "predictions.json"
    score_attempt = tmp_path / "score_attempt.json"
    score_authorization = tmp_path / "score_authorization.json"
    score_output = tmp_path / "score.json"
    authorization.write_text("{}\n")
    monkeypatch.setattr(campaign, "DEFAULT_PREDICTION_AUTHORIZATION", authorization)
    monkeypatch.setattr(campaign, "DEFAULT_PREDICTION_ATTEMPT", attempt)
    monkeypatch.setattr(campaign, "DEFAULT_PREDICTION", output)
    monkeypatch.setattr(campaign, "DEFAULT_SCORE_AUTHORIZATION", score_authorization)
    monkeypatch.setattr(campaign, "DEFAULT_SCORE_ATTEMPT", score_attempt)
    monkeypatch.setattr(campaign, "DEFAULT_SCORE", score_output)
    return authorization, attempt, output, score_attempt


def _patch_score_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path, Path, Path]:
    authorization = tmp_path / "score_authorization.json"
    prediction = tmp_path / "predictions.json"
    prediction_attempt = tmp_path / "prediction_attempt.json"
    attempt = tmp_path / "score_attempt.json"
    output = tmp_path / "score.json"
    authorization.write_text("{}\n")
    prediction.write_text("{}\n")
    prediction_attempt.write_text("{}\n")
    monkeypatch.setattr(campaign, "DEFAULT_SCORE_AUTHORIZATION", authorization)
    monkeypatch.setattr(campaign, "DEFAULT_PREDICTION", prediction)
    monkeypatch.setattr(campaign, "DEFAULT_PREDICTION_ATTEMPT", prediction_attempt)
    monkeypatch.setattr(campaign, "DEFAULT_SCORE_ATTEMPT", attempt)
    monkeypatch.setattr(campaign, "DEFAULT_SCORE", output)
    return authorization, prediction, attempt, output


def test_real_preaccess_check_never_opens_the_h5ad(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "preaccess.json"
    monkeypatch.setattr(campaign, "DEFAULT_PREACCESS", output)
    monkeypatch.setattr(
        campaign,
        "_open_h5ad",
        lambda *_: pytest.fail("preaccess must not open the H5AD"),
    )
    monkeypatch.setattr(
        campaign,
        "_verify_source_bytes",
        lambda *_: pytest.fail("preaccess must not stream the H5AD"),
    )

    result = campaign.verify_preaccess(output)
    replayed = campaign.verify_preaccess(output, check_existing=True)

    assert result["status"] == "PASS_H5AD_UNOPENED"
    assert replayed == result
    assert result["matrix_values_read_by_this_check"] == 0
    assert result["donors"] == [donor for donor, _ in campaign.UNUSED]


def test_compact_classical_fields_are_bound_and_well_formed() -> None:
    development = campaign._validated_development(campaign.DEFAULT_DEVELOPMENT)
    classical = campaign._validated_classical(campaign.DEFAULT_CLASSICAL, development)

    assert campaign._sha256(campaign.DEFAULT_CLASSICAL) == (
        campaign.EXPECTED_CLASSICAL_SHA256
    )
    assert campaign._sha256(campaign.DEFAULT_CLASSICAL_AUDIT) == (
        campaign.EXPECTED_CLASSICAL_AUDIT_SHA256
    )
    assert set(classical["fields"]) == {
        "common_effect_exact_cmle",
        "pooled_poisson_loglinear_interaction",
    }
    for field in classical["fields"].values():
        assert len(field["source_coordinate"]) == 81
        assert field["alpha"] == 1.0


def test_prediction_reads_rna_only_after_terminal_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization, attempt, output, _ = _patch_prediction_paths(monkeypatch, tmp_path)
    records = _records()
    selections = _selections()
    events: list[str] = []
    monkeypatch.setattr(
        campaign,
        "_validate_prediction_authorization",
        lambda *_: {"public_implementation_commit": "a" * 40},
    )
    monkeypatch.setattr(
        campaign,
        "_validated_recovery_lineage",
        lambda *_: {"status": "OUTCOME_BLIND_SINGLE_REPLACEMENT_ELIGIBLE"},
    )
    monkeypatch.setattr(
        campaign,
        "_verify_source_bytes",
        lambda *_: {"bytes": 1, "sha256": "b" * 64},
    )
    monkeypatch.setattr(campaign, "_validated_source_manifest", lambda *_: {})
    monkeypatch.setattr(campaign, "_unused_records", lambda *_: records)
    monkeypatch.setattr(campaign, "_validated_development", lambda *_: {})
    monkeypatch.setattr(campaign, "_validated_classical", lambda *_: {})
    monkeypatch.setattr(campaign, "_models", lambda *_: _independence_models())

    @contextmanager
    def open_h5ad(_source: campaign.H5ADInput):
        assert attempt.is_file(), "HDF5 open preceded the terminal attempt"
        events.append("open")
        yield object()

    def select(_handle: object, _records: list[dict[str, object]]):
        assert attempt.is_file()
        events.append("metadata")
        return selections

    def read(
        _handle: object,
        _selections: dict[str, dict[str, object]],
        samples: tuple[str, ...],
        modality: str,
    ) -> dict[str, np.ndarray]:
        assert attempt.is_file(), "numeric read preceded the terminal attempt"
        if modality != "rna":
            raise AssertionError("prediction attempted an ADT read")
        events.append(modality)
        return {sample: np.zeros((9, 512), dtype=np.int64) for sample in samples}

    monkeypatch.setattr(campaign, "_open_h5ad", open_h5ad)
    monkeypatch.setattr(campaign, "_selected_rows_from_handle", select)
    monkeypatch.setattr(campaign, "_read_modality_from_handle", read)

    result = campaign.predict(
        campaign.H5ADInput("remote", "memory://unused-fixture"),
        authorization,
        "c" * 40,
        attempt,
        output,
    )

    assert events == ["open", "metadata", "open", "rna"]
    attempt_payload = json.loads(attempt.read_text())
    assert attempt_payload["status"] == "TERMINAL_REPLACEMENT_ATTEMPT_STARTED"
    assert attempt_payload["replacement_ordinal"] == 1
    assert attempt_payload["maximum_replacement_attempts"] == 1
    assert attempt_payload["scientific_design_changed"] is False
    assert result["access_audit"]["adt_handles_opened"] == 0
    assert result["access_audit"]["rna_adt_pairings_formed"] == 0


def test_score_opens_metadata_rna_and_adt_only_after_terminal_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization, prediction_path, attempt, output = _patch_score_paths(
        monkeypatch, tmp_path
    )
    records = _records()
    selections = _selections()
    rna_margins = np.tile(np.asarray([[1, 511]], dtype=np.int64), (9, 1))
    adt_margins = np.tile(np.asarray([[256, 256]], dtype=np.int64), (9, 1))
    expected = campaign.stephenson._predict_method(
        {"kind": "independence"}, rna_margins, adt_margins
    ).reshape(81, 4)
    prediction = {
        "samples": [
            {
                "donor": donor,
                "sample": sample,
                "rna_margins": rna_margins.tolist(),
                "adt_margins": adt_margins.tolist(),
                "selected_barcode_sha256": selections[sample][
                    "selected_barcode_sha256"
                ],
                "predictions": {name: expected.tolist() for name in campaign.METHODS},
            }
            for donor, sample in campaign.UNUSED
        ]
    }
    events: list[str] = []
    monkeypatch.setattr(
        campaign,
        "_validate_score_authorization",
        lambda *_: {"public_prediction_commit": "a" * 40},
    )
    monkeypatch.setattr(
        campaign,
        "_validated_recovery_lineage",
        lambda *_: {"status": "OUTCOME_BLIND_SINGLE_REPLACEMENT_ELIGIBLE"},
    )
    monkeypatch.setattr(
        campaign,
        "_verify_source_bytes",
        lambda *_: {"bytes": 1, "sha256": "b" * 64},
    )
    monkeypatch.setattr(campaign, "_validated_source_manifest", lambda *_: {})
    monkeypatch.setattr(campaign, "_unused_records", lambda *_: records)
    monkeypatch.setattr(campaign, "_validated_development", lambda *_: {})
    monkeypatch.setattr(campaign, "_validated_classical", lambda *_: {})
    monkeypatch.setattr(campaign, "_validate_prediction", lambda *_: prediction)

    @contextmanager
    def open_h5ad(_source: campaign.H5ADInput):
        assert attempt.is_file(), "HDF5 open preceded the terminal attempt"
        events.append("open")
        yield object()

    def select(_handle: object, _records: list[dict[str, object]]):
        assert attempt.is_file()
        events.append("metadata")
        return selections

    state_counts = np.tile(np.arange(512, dtype=np.int64), (9, 1))

    def read(
        _handle: object,
        _selections: dict[str, dict[str, object]],
        samples: tuple[str, ...],
        modality: str,
    ) -> dict[str, np.ndarray]:
        assert attempt.is_file(), "numeric read preceded the terminal attempt"
        events.append(modality)
        return {sample: state_counts.copy() for sample in samples}

    monkeypatch.setattr(campaign, "_open_h5ad", open_h5ad)
    monkeypatch.setattr(campaign, "_selected_rows_from_handle", select)
    monkeypatch.setattr(campaign, "_read_modality_from_handle", read)

    result = campaign.score(
        campaign.H5ADInput("remote", "memory://unused-fixture"),
        authorization,
        "c" * 40,
        prediction_path,
        attempt,
        output,
    )

    assert events == [
        "open",
        "metadata",
        "open",
        "rna",
        "open",
        "adt",
    ]
    assert result["status"] == "CONFIRMATION_FAIL"
    assert result["access_audit"]["terminal_attempt_preceded_first_h5ad_open"]


def test_existing_attempt_refuses_before_source_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization, attempt, output, _ = _patch_prediction_paths(monkeypatch, tmp_path)
    attempt.write_text("{}\n")
    monkeypatch.setattr(
        campaign,
        "_validate_prediction_authorization",
        lambda *_: pytest.fail("authorization validation was reached"),
    )

    with pytest.raises(FileExistsError, match="one-shot"):
        campaign.predict(
            campaign.H5ADInput("remote", "memory://unused-fixture"),
            authorization,
            "a" * 40,
            attempt,
            output,
        )


def test_altered_initial_attempt_refuses_before_source_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization, attempt, output, _ = _patch_prediction_paths(monkeypatch, tmp_path)
    real_sha256 = campaign._sha256
    monkeypatch.setattr(
        campaign,
        "_sha256",
        lambda path: (
            "0" * 64
            if path == campaign.DEFAULT_INITIAL_PREDICTION_ATTEMPT
            else real_sha256(path)
        ),
    )
    monkeypatch.setattr(
        campaign,
        "_validate_prediction_authorization",
        lambda *_: pytest.fail("authorization validation was reached"),
    )
    monkeypatch.setattr(
        campaign,
        "_verify_source_bytes",
        lambda *_: pytest.fail("source verification was reached"),
    )

    with pytest.raises(PermissionError, match="artifact digest"):
        campaign.predict(
            campaign.H5ADInput("remote", "memory://unused-fixture"),
            authorization,
            "a" * 40,
            attempt,
            output,
        )


def test_altered_recovery_amendment_refuses_before_source_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization, attempt, output, _ = _patch_prediction_paths(monkeypatch, tmp_path)
    real_sha256 = campaign._sha256
    monkeypatch.setattr(
        campaign,
        "_sha256",
        lambda path: (
            "0" * 64
            if path == campaign.DEFAULT_RECOVERY_AMENDMENT
            else real_sha256(path)
        ),
    )
    monkeypatch.setattr(
        campaign,
        "_validate_prediction_authorization",
        lambda *_: pytest.fail("authorization validation was reached"),
    )
    monkeypatch.setattr(
        campaign,
        "_verify_source_bytes",
        lambda *_: pytest.fail("source verification was reached"),
    )

    with pytest.raises(PermissionError, match="artifact digest"):
        campaign.predict(
            campaign.H5ADInput("remote", "memory://unused-fixture"),
            authorization,
            "a" * 40,
            attempt,
            output,
        )


def test_initial_score_artifact_blocks_replacement_before_source_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization, attempt, output, _ = _patch_prediction_paths(monkeypatch, tmp_path)
    initial_score_authorization = tmp_path / "initial_score_authorization.json"
    initial_score_authorization.write_text("{}\n")
    monkeypatch.setattr(
        campaign, "DEFAULT_INITIAL_SCORE_AUTHORIZATION", initial_score_authorization
    )
    monkeypatch.setattr(
        campaign,
        "_validate_prediction_authorization",
        lambda *_: pytest.fail("authorization validation was reached"),
    )
    monkeypatch.setattr(
        campaign,
        "_verify_source_bytes",
        lambda *_: pytest.fail("source verification was reached"),
    )

    with pytest.raises(PermissionError, match="initial outcome artifact"):
        campaign.predict(
            campaign.H5ADInput("remote", "memory://unused-fixture"),
            authorization,
            "a" * 40,
            attempt,
            output,
        )


def test_exact_sign_flip_has_no_favorable_count_gate() -> None:
    donors = [f"d{index}" for index in range(11)]
    comparator = np.ones(11)
    primary = np.asarray([0.2] * 9 + [1.01, 1.01])

    comparison = campaign._comparison(donors, primary, comparator, "nine")

    assert comparison["favorable_donors"] == 9
    assert comparison["exact_one_sided_paired_sign_flip_p"] == pytest.approx(4 / 2048)
    assert comparison["passes_primary_gate"]


def test_pooled_poisson_is_reported_but_does_not_veto_confirmation() -> None:
    comparisons = {
        "best_residual": {"passes_primary_gate": True},
        "common_effect_exact_cmle": {"passes_primary_gate": True},
        "destroyed_link": {"passes_effect_and_ci": True},
        "pooled_poisson_loglinear_interaction": {
            "passes_effect_and_ci": False,
            "passes_primary_gate": False,
        },
    }

    assert campaign._confirmation_gates(comparisons) == {
        "passes_field_transfer": True,
        "passes_hierarchical_increment": True,
        "passes_full_confirmation": True,
    }


def test_remote_fsspec_handle_is_seekable_by_h5py(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "fixture.h5ad"
    with h5py.File(local, "w") as handle:
        handle.create_dataset("sentinel", data=np.asarray([3, 5, 8]))
    remote = f"memory://stephenson-test/{tmp_path.name}.h5ad"
    with fsspec.open(remote, "wb") as stream:
        stream.write(local.read_bytes())
    monkeypatch.setattr(campaign, "OFFICIAL_H5AD_RESOLVED_URL", remote)

    with campaign._open_h5ad(campaign.H5ADInput("remote", remote)) as handle:
        assert handle["sentinel"][:].tolist() == [3, 5, 8]


def test_remote_identity_check_uses_head_without_reading_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HeadResponse:
        headers = {
            "Content-Length": str(campaign.OFFICIAL_H5AD_BYTES),
            "Accept-Ranges": "bytes",
            "ETag": campaign.OFFICIAL_H5AD_ETAG,
            "Last-Modified": campaign.OFFICIAL_H5AD_LAST_MODIFIED,
        }

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def geturl(self) -> str:
            return campaign.OFFICIAL_H5AD_RESOLVED_URL

        def read(self, *_args: object) -> bytes:
            raise AssertionError("remote identity check read payload bytes")

    monkeypatch.setattr(
        campaign.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: HeadResponse(),
    )

    result = campaign._verify_source_bytes(
        campaign.H5ADInput("remote", campaign.OFFICIAL_H5AD_URL)
    )

    assert result["bytes"] == campaign.OFFICIAL_H5AD_BYTES
    assert result["sha256"] == campaign.OFFICIAL_H5AD_SHA256
    assert result["sha256_provenance"] == "checksum-bound source manifest"


def test_local_source_verification_hashes_the_full_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"opaque-source-fixture" * 17
    path = tmp_path / "source.h5ad"
    path.write_bytes(payload)
    monkeypatch.setattr(campaign, "OFFICIAL_H5AD_BYTES", len(payload))
    monkeypatch.setattr(
        campaign, "OFFICIAL_H5AD_SHA256", hashlib.sha256(payload).hexdigest()
    )

    result = campaign._verify_source_bytes(campaign.H5ADInput("local", str(path)))

    assert result["bytes"] == len(payload)
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["sha256_provenance"] == "recomputed_full_local_stream"


def test_frozen_prediction_tampering_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "prediction.json"
    attempt = tmp_path / "prediction_attempt.json"
    attempt.write_text("{}\n")
    monkeypatch.setattr(campaign, "DEFAULT_PREDICTION_ATTEMPT", attempt)
    models = _independence_models()
    rna = np.tile(np.asarray([[256, 256]], dtype=np.int64), (9, 1))
    adt = rna.copy()
    expected = (
        campaign.stephenson._predict_method({"kind": "independence"}, rna, adt)
        .reshape(81, 4)
        .tolist()
    )
    payload = {
        "schema": "stephenson-unused-cambridge-predictions/1.1",
        "status": "FROZEN_PREDICTIONS",
        "attempt_sha256": campaign._sha256(attempt),
        "source_manifest_sha256": campaign.EXPECTED_SOURCE_SHA256,
        "development_sha256": campaign.EXPECTED_DEVELOPMENT_SHA256,
        "classical_fields_sha256": campaign.EXPECTED_CLASSICAL_SHA256,
        "classical_audit_sha256": campaign.EXPECTED_CLASSICAL_AUDIT_SHA256,
        "runner_sha256": campaign._sha256(Path(campaign.__file__)),
        "protocol_sha256": campaign._sha256(campaign.DEFAULT_PROTOCOL),
        "recovery_protocol_sha256": campaign._sha256(
            campaign.DEFAULT_RECOVERY_PROTOCOL
        ),
        "recovery_status": "OUTCOME_BLIND_SINGLE_REPLACEMENT_ELIGIBLE",
        "attempt_kind": "REPLACEMENT_AFTER_HOST_INTERRUPTION",
        "attempt_ordinal": 2,
        "replacement_ordinal": 1,
        "maximum_replacement_attempts": 1,
        "initial_attempt_sha256": (campaign.EXPECTED_INITIAL_PREDICTION_ATTEMPT_SHA256),
        "recovery_amendment_sha256": campaign.EXPECTED_RECOVERY_AMENDMENT_SHA256,
        "scientific_design_changed": False,
        "methods": list(campaign.METHODS),
        "donors": 11,
        "access_audit": {
            "adt_handles_opened": 0,
            "rna_adt_pairings_formed": 0,
            "truth_tables_formed": 0,
        },
        "samples": [
            {
                "donor": donor,
                "sample": sample,
                "rna_margins": rna.tolist(),
                "adt_margins": adt.tolist(),
                "selected_barcode_sha256": "a" * 64,
                "predictions": {name: expected for name in campaign.METHODS},
            }
            for donor, sample in campaign.UNUSED
        ],
    }
    path.write_text(json.dumps(payload) + "\n")
    monkeypatch.setattr(campaign, "_models", lambda *_: models)
    assert campaign._validate_prediction(path, {}, {})["donors"] == 11

    payload["attempt_sha256"] = "0" * 64
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(PermissionError, match="header differs"):
        campaign._validate_prediction(path, {}, {})

    payload["attempt_sha256"] = campaign._sha256(attempt)
    payload["recovery_amendment_sha256"] = "0" * 64
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(PermissionError, match="header differs"):
        campaign._validate_prediction(path, {}, {})

    payload["recovery_amendment_sha256"] = campaign.EXPECTED_RECOVERY_AMENDMENT_SHA256
    del payload["samples"][0]["selected_barcode_sha256"]
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(PermissionError, match="selected-cell digest"):
        campaign._validate_prediction(path, {}, {})
