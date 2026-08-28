from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments import confirm_stephenson_citeseq as stephenson


def _records() -> list[dict[str, object]]:
    records = []
    for index in range(36):
        records.append(
            {
                "donor": f"CV-C-{index:02d}",
                "sample": f"BGCV-C-{index:02d}",
                "site": "Cambridge",
                "disease": "COVID-19",
                "eligible_pool_cells": 600,
                "eligible_sample_candidates": [f"BGCV-C-{index:02d}"],
            }
        )
    for index in range(11):
        records.append(
            {
                "donor": f"CV-H-{index:02d}",
                "sample": f"BGCV-H-{index:02d}",
                "site": "Cambridge",
                "disease": "normal",
                "eligible_pool_cells": 600,
                "eligible_sample_candidates": [f"BGCV-H-{index:02d}"],
            }
        )
    for index in range(56):
        records.append(
            {
                "donor": f"NC-{index:02d}",
                "sample": f"MH-{index:02d}",
                "site": "Ncl",
                "disease": "COVID-19" if index < 39 else "normal",
                "eligible_pool_cells": 600,
                "eligible_sample_candidates": [f"MH-{index:02d}"],
            }
        )
    roles = stephenson._assign_roles(records)
    for record in records:
        record["role"] = roles[record["donor"]]
    return records


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def _patch_entry_defaults(
    monkeypatch: pytest.MonkeyPatch, **bindings: Path
) -> None:
    for name, path in bindings.items():
        monkeypatch.setattr(stephenson, name, path)


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"status":"A","status":"B"}\n')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        stephenson._read_json(path)


def test_development_authorization_rejects_tampered_v11_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze_commit = "a" * 40
    verification_commit = "b" * 40
    authorization_commit = "c" * 40
    authorization_path = tmp_path / "authorization.json"
    verification_path = tmp_path / "verification.json"
    bindings = {
        label: {
            "path": relative,
            "sha256": stephenson._sha256(stephenson._bound_path(relative)),
        }
        for label, relative in stephenson.DEVELOPMENT_BINDING_PATHS.items()
    }
    authorization = {
        "schema": "stephenson-citeseq-development-authorization/1.0",
        "status": "OUTCOME_ACCESS_AUTHORIZED",
        "created_at_utc": "2026-08-28T00:00:00Z",
        "public_freeze_commit": freeze_commit,
        "public_verification_commit": verification_commit,
        "artifact_bindings": bindings,
    }
    verification = {
        "schema": "stephenson-citeseq-public-freeze-verification/1.0",
        "status": "PASS",
        "fresh_clone": True,
        "canonical_origin": stephenson.PUBLIC_ORIGIN,
        "public_freeze_commit": freeze_commit,
        "planned_immutable_tag": stephenson.PLANNED_IMMUTABLE_TAG,
        "artifact_bindings": {
            label: row
            for label, row in bindings.items()
            if label != "fresh_clone_verification"
        },
        "all_bound_artifacts_match": True,
        "matrix_payload_reads": 0,
    }
    _write_json(authorization_path, authorization)
    _write_json(verification_path, verification)
    monkeypatch.setattr(stephenson, "DEFAULT_VERIFICATION", verification_path)
    monkeypatch.setattr(stephenson, "_relative", lambda path: path.name)

    def public_bytes(relative: str, _commit: str, label: str) -> bytes:
        if label == "development authorization":
            return authorization_path.read_bytes()
        return stephenson._bound_path(relative).read_bytes()

    monkeypatch.setattr(stephenson, "_immutable_public_bytes", public_bytes)
    assert stephenson._validated_development_authorization(
        authorization_path,
        stephenson.DEFAULT_SOURCE,
        authorization_commit,
    )["public_freeze_commit"] == freeze_commit

    verification["planned_immutable_tag"] = "stephenson-citeseq-wrong-tag"
    _write_json(verification_path, verification)
    with pytest.raises(PermissionError, match="fresh-clone verification"):
        stephenson._validated_development_authorization(
            authorization_path,
            stephenson.DEFAULT_SOURCE,
            authorization_commit,
        )


def test_role_assignment_is_deterministic_and_disjoint() -> None:
    records = _records()
    roles = {record["donor"]: record["role"] for record in records}
    assert list(roles.values()).count("calibration") == 12
    assert list(roles.values()).count("pilot") == 24
    assert list(roles.values()).count("unused_source") == 11
    assert list(roles.values()).count("held_site") == 56
    assert stephenson._assign_roles(list(reversed(records))) == roles


def test_role_assignment_rejects_missing_held_donor() -> None:
    records = _records()[:-1]
    with pytest.raises(PermissionError, match="eligible donor counts"):
        stephenson._assign_roles(records)


def test_frozen_real_source_manifest_v1_1_validates_without_outcome_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = stephenson._read_json(stephenson.DEFAULT_SOURCE)
    assert stephenson._sha256(stephenson.DEFAULT_SOURCE) == (
        "431e8a370bc9b08a207ab0ff8d3581f80abaf0f36b55eba4accedc5685a3d3cd"
    )
    assert manifest["schema"] == "stephenson-citeseq-source-manifest/1.1"
    assert (
        manifest["sdrf_to_h5ad_sample_corrections"]
        == stephenson.SDRF_TO_H5AD_SAMPLE
    )
    sentinel = Path("outcome-access-disabled.h5ad")
    monkeypatch.setattr(stephenson, "_resolved_h5ad", lambda _: sentinel)
    monkeypatch.setattr(
        stephenson,
        "_opaque_hashes",
        lambda _: pytest.fail("source validation must not hash the H5AD"),
    )

    validated = stephenson._validated_source(
        stephenson.DEFAULT_SOURCE, verify_hash=False
    )

    assert validated["payload"] == manifest
    assert validated["h5ad"] == sentinel


def test_seal_source_serializes_exact_sample_corrections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h5ad = tmp_path / stephenson.OFFICIAL_H5AD_NAME
    sdrf = tmp_path / "source.sdrf.txt"
    h5ad.write_bytes(b"opaque fixture")
    sdrf.write_text("metadata fixture\n")
    monkeypatch.setattr(stephenson, "_opaque_hashes", lambda _: ("a" * 64, "b" * 32))
    monkeypatch.setattr(
        stephenson,
        "_metadata_inventory",
        lambda *_: {"samples": []},
    )

    source = stephenson.seal_source(
        h5ad,
        sdrf,
        tmp_path / "preflight.json",
        tmp_path / "source.json",
    )

    assert source["schema"] == "stephenson-citeseq-source-manifest/1.1"
    assert source["sdrf_to_h5ad_sample_corrections"] == stephenson.SDRF_TO_H5AD_SAMPLE


def test_legacy_anndata_categorical_decodes(tmp_path: Path) -> None:
    path = tmp_path / "legacy.h5ad"
    string = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as handle:
        obs = handle.create_group("obs")
        categories = obs.create_dataset("site_categories", data=["A", "B"], dtype=string)
        codes = obs.create_dataset("Site", data=np.asarray([0, 1, 0], dtype=np.int8))
        codes.attrs["categories"] = categories.ref
    with h5py.File(path, "r") as handle:
        observed = stephenson._encoded_column(handle, handle["obs"], "Site")
    assert observed.tolist() == ["A", "B", "A"]


def test_matrix_metadata_never_indexes_payload_datasets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "matrix.h5ad"
    with h5py.File(path, "w") as handle:
        matrix = handle.create_group("layers/raw")
        matrix.attrs["encoding-type"] = "csr_matrix"
        matrix.attrs["shape"] = np.asarray([647_366, 24_929], dtype=np.int64)
        matrix.create_dataset(
            "data", shape=(965_744_336,), dtype=np.float32, chunks=(1024,)
        )
        matrix.create_dataset(
            "indices", shape=(965_744_336,), dtype=np.int32, chunks=(1024,)
        )
        matrix.create_dataset(
            "indptr", shape=(647_367,), dtype=np.int32, chunks=(1024,)
        )
    original = h5py.Dataset.__getitem__

    def poison(dataset: h5py.Dataset, key: object) -> object:
        if dataset.name.rsplit("/", 1)[-1] in {"data", "indices", "indptr"}:
            raise AssertionError("matrix payload was indexed")
        return original(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", poison)
    with h5py.File(path, "r") as handle:
        metadata = stephenson._matrix_metadata(handle, "layers/raw")
    assert metadata["shape"] == [647_366, 24_929]


def test_csr_subset_reads_only_requested_feature_values(tmp_path: Path) -> None:
    path = tmp_path / "small.h5ad"
    with h5py.File(path, "w") as handle:
        matrix = handle.create_group("raw")
        matrix.attrs["encoding-type"] = "csr_matrix"
        matrix.attrs["shape"] = np.asarray([3, 5], dtype=np.int64)
        matrix.create_dataset("indptr", data=np.asarray([0, 2, 3, 5]))
        matrix.create_dataset("indices", data=np.asarray([0, 3, 2, 1, 4]))
        matrix.create_dataset("data", data=np.asarray([2, 7, 5, 11, 13]))
    with h5py.File(path, "r") as handle:
        observed = stephenson.numerics._read_csr_feature_subset(
            handle["raw"], np.asarray([0, 2]), np.asarray([1, 3])
        )
    assert observed.tolist() == [[0.0, 7.0], [11.0, 0.0]]


def test_entrypoints_reject_every_nondefault_path_before_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        stephenson,
        "_validated_development_authorization",
        lambda *_args, **_kwargs: pytest.fail("development validator was reached"),
    )
    monkeypatch.setattr(
        stephenson,
        "_validated_margin_authorization",
        lambda *_args, **_kwargs: pytest.fail("margin validator was reached"),
    )
    monkeypatch.setattr(
        stephenson,
        "_validated_score_authorization",
        lambda *_args, **_kwargs: pytest.fail("score validator was reached"),
    )
    cases = (
        (
            stephenson.run_development,
            [
                stephenson.DEFAULT_SOURCE,
                stephenson.DEFAULT_DEVELOPMENT_AUTHORIZATION,
                "a" * 40,
                stephenson.DEFAULT_DEVELOPMENT_ATTEMPT,
                stephenson.DEFAULT_DEVELOPMENT,
            ],
            (0, 1, 3, 4),
        ),
        (
            stephenson.predict_held,
            [
                stephenson.DEFAULT_SOURCE,
                stephenson.DEFAULT_DEVELOPMENT,
                stephenson.DEFAULT_MARGIN_AUTHORIZATION,
                "a" * 40,
                stephenson.DEFAULT_PREDICTION_ATTEMPT,
                stephenson.DEFAULT_PREDICTION,
            ],
            (0, 1, 2, 4, 5),
        ),
        (
            stephenson.score_held,
            [
                stephenson.DEFAULT_SOURCE,
                stephenson.DEFAULT_DEVELOPMENT,
                stephenson.DEFAULT_PREDICTION,
                stephenson.DEFAULT_SCORE_AUTHORIZATION,
                "a" * 40,
                stephenson.DEFAULT_SCORE_ATTEMPT,
                stephenson.DEFAULT_SCORE,
            ],
            (0, 1, 2, 3, 5, 6),
        ),
    )
    for entrypoint, arguments, path_indices in cases:
        for index in path_indices:
            tampered = list(arguments)
            tampered[index] = tmp_path / f"wrong-{entrypoint.__name__}-{index}"
            with pytest.raises(PermissionError, match="path is not fixed"):
                entrypoint(*tampered)


def test_development_attempt_precedes_first_numeric_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source.json"
    authorization = tmp_path / "authorization.json"
    attempt = tmp_path / "attempt.json"
    output = tmp_path / "development.json"
    source_path.write_text("{}")
    authorization.write_text("{}")
    _patch_entry_defaults(
        monkeypatch,
        DEFAULT_SOURCE=source_path,
        DEFAULT_DEVELOPMENT_AUTHORIZATION=authorization,
        DEFAULT_DEVELOPMENT_ATTEMPT=attempt,
        DEFAULT_DEVELOPMENT=output,
    )
    records = _records()
    source = {
        "records": records,
        "payload": {"h5ad": {"sha256": "a" * 64}},
        "source_sha256": stephenson._sha256(source_path),
    }
    monkeypatch.setattr(
        stephenson,
        "_validated_development_authorization",
        lambda *_: {
            "authorization_sha256": "b" * 64,
            "public_authorization_commit": "c" * 40,
        },
    )
    monkeypatch.setattr(stephenson, "_validated_source", lambda *_args, **_kw: source)

    def numeric(_: object) -> dict[str, object]:
        assert attempt.is_file()
        return {record["sample"]: {} for record in records if record["role"] in {"calibration", "pilot"}}

    monkeypatch.setattr(stephenson, "_development_records", numeric)
    monkeypatch.setattr(stephenson, "_relative", lambda path: path.name)
    monkeypatch.setattr(
        stephenson,
        "_pilot_analysis",
        lambda *_: {
            "status": "PILOT_FAIL",
            "passes_pilot_gate": False,
            "frozen_source_models": None,
        },
    )
    stephenson.run_development(
        source_path, authorization, "d" * 40, attempt, output
    )
    assert attempt.is_file()
    assert json.loads(output.read_text())["status"] == "PILOT_FAIL"


def test_development_rerun_rejected_before_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "attempt.json"
    source = tmp_path / "source"
    authorization = tmp_path / "auth"
    output = tmp_path / "output"
    attempt.write_text("{}")
    _patch_entry_defaults(
        monkeypatch,
        DEFAULT_SOURCE=source,
        DEFAULT_DEVELOPMENT_AUTHORIZATION=authorization,
        DEFAULT_DEVELOPMENT_ATTEMPT=attempt,
        DEFAULT_DEVELOPMENT=output,
    )
    monkeypatch.setattr(
        stephenson,
        "_validated_development_authorization",
        lambda *_: pytest.fail("validation must not run on rerun"),
    )
    with pytest.raises(FileExistsError, match="one-shot"):
        stephenson.run_development(
            source,
            authorization,
            "a" * 40,
            attempt,
            output,
        )


def test_pilot_failure_blocks_held_margin_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auth = tmp_path / "auth.json"
    development = tmp_path / "development.json"
    source = tmp_path / "source.json"
    for path in (auth, development, source):
        path.write_text("{}")
    monkeypatch.setattr(
        stephenson,
        "_validated_development",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            PermissionError("pilot did not authorize held access")
        ),
    )
    with pytest.raises(PermissionError, match="pilot"):
        stephenson._validated_margin_authorization(
            auth, source, development, "a" * 40
        )


def test_development_result_rejects_gate_comparator_and_model_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "development-attempt.json"
    attempt.write_text("{}")
    monkeypatch.setattr(stephenson, "DEFAULT_DEVELOPMENT_ATTEMPT", attempt)
    monkeypatch.setattr(stephenson, "_relative", lambda path: path.name)
    source = {
        "source_sha256": "a" * 64,
        "payload": {"h5ad": {"sha256": "b" * 64}},
    }
    monkeypatch.setattr(
        stephenson, "_validated_source", lambda *_args, **_kwargs: source
    )
    monkeypatch.setattr(
        stephenson,
        "_development_samples",
        lambda _source, role: ("calibration",) if role == "calibration" else ("pilot",),
    )
    models = {
        name: {"kind": kind} for name, kind in stephenson.METHOD_KINDS.items()
    }
    payload = {
        "schema": "stephenson-citeseq-development/1.0",
        "status": "PILOT_PASS",
        "source_manifest_sha256": source["source_sha256"],
        "h5ad_sha256": source["payload"]["h5ad"]["sha256"],
        "runner_sha256": stephenson._sha256(Path(stephenson.__file__)),
        "combat_data_and_comparator_utility_sha256": stephenson._sha256(
            stephenson.ROOT
            / stephenson.DEVELOPMENT_BINDING_PATHS[
                "combat_data_and_comparator_utility"
            ]
        ),
        "markers": list(stephenson.MARKERS),
        "calibration_samples": ["calibration"],
        "pilot_samples": ["pilot"],
        "development_attempt": {
            "path": attempt.name,
            "sha256": stephenson._sha256(attempt),
        },
        "promotion_comparators": list(stephenson.PROMOTION_COMPARATORS),
        "passes_pilot_gate": True,
        "pilot_comparisons": {
            name: {"passes": True} for name in stephenson.PROMOTION_COMPARATORS
        },
        "frozen_source_models": models,
    }
    baseline = tmp_path / "development-pass.json"
    _write_json(baseline, payload)
    assert stephenson._validated_development(
        baseline, tmp_path / "source.json", require_pass=True
    )["status"] == "PILOT_PASS"

    variants = []
    tampered = json.loads(json.dumps(payload))
    tampered["passes_pilot_gate"] = False
    variants.append(tampered)
    tampered = json.loads(json.dumps(payload))
    tampered["promotion_comparators"].reverse()
    variants.append(tampered)
    tampered = json.loads(json.dumps(payload))
    tampered["pilot_comparisons"]["best_residual"]["passes"] = False
    variants.append(tampered)
    tampered = json.loads(json.dumps(payload))
    del tampered["pilot_comparisons"]["destroyed_link"]
    variants.append(tampered)
    tampered = json.loads(json.dumps(payload))
    del tampered["frozen_source_models"]["primary"]
    variants.append(tampered)
    tampered = json.loads(json.dumps(payload))
    tampered["frozen_source_models"]["extra"] = {"kind": "independence"}
    variants.append(tampered)
    tampered = json.loads(json.dumps(payload))
    tampered["frozen_source_models"]["best_residual"]["kind"] = "independence"
    variants.append(tampered)

    for index, candidate in enumerate(variants):
        path = tmp_path / f"tampered-development-{index}.json"
        _write_json(path, candidate)
        with pytest.raises(PermissionError):
            stephenson._validated_development(
                path, tmp_path / "source.json", require_pass=True
            )


def test_prediction_attempt_precedes_worker_and_serializes_no_vectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "prediction_attempt.json"
    output = tmp_path / "predictions.json"
    source_path = tmp_path / "source.json"
    development_path = tmp_path / "development.json"
    authorization_path = tmp_path / "authorization.json"
    for path in (source_path, development_path, authorization_path):
        path.write_text("{}")
    _patch_entry_defaults(
        monkeypatch,
        DEFAULT_SOURCE=source_path,
        DEFAULT_DEVELOPMENT=development_path,
        DEFAULT_MARGIN_AUTHORIZATION=authorization_path,
        DEFAULT_PREDICTION_ATTEMPT=attempt,
        DEFAULT_PREDICTION=output,
    )
    held = [record for record in _records() if record["role"] == "held_site"]
    source = {
        "records": held,
        "payload": {"h5ad": {"sha256": "a" * 64}},
        "source_sha256": stephenson._sha256(source_path),
    }
    models = {
        name: {"kind": "independence", "estimator": "test"}
        for name in stephenson.METHODS
    }
    monkeypatch.setattr(
        stephenson,
        "_validated_margin_authorization",
        lambda *_: {
            "authorization_sha256": "b" * 64,
            "public_authorization_commit": "c" * 40,
            "public_development_commit": "d" * 40,
        },
    )
    monkeypatch.setattr(stephenson, "_validated_source", lambda *_args, **_kw: source)
    monkeypatch.setattr(
        stephenson,
        "_validated_development",
        lambda *_args, **_kw: {"frozen_source_models": models},
    )
    monkeypatch.setattr(stephenson, "_relative", lambda path: path.name)

    def margins(*_: object) -> list[dict[str, object]]:
        assert attempt.is_file()
        values = np.tile(np.asarray([[256, 256]]), (len(stephenson.MARKERS), 1))
        return [
            {
                "donor": record["donor"],
                "sample": record["sample"],
                "rna_margins": values.tolist(),
                "rna_margin_sha256": stephenson._array_sha256(values),
                "selected_barcode_sha256": "e" * 64,
                "eligible_pool_cells": 600,
            }
            for record in held
        ]

    monkeypatch.setattr(stephenson, "_extract_held_rna_margins", margins)
    stephenson.predict_held(
        source_path,
        development_path,
        authorization_path,
        "f" * 40,
        attempt,
        output,
    )
    payload = json.loads(output.read_text())
    assert payload["access_audit"]["held_adt_numeric_values_read"] == 0
    assert payload["access_audit"]["cell_vectors_serialized"] is False


def test_conditional_prediction_uses_exact_target_margin_expectation() -> None:
    coordinate = np.linspace(-0.8, 0.8, len(stephenson.MARKERS) ** 2)
    model = {
        "kind": "conditional_log_odds",
        "alpha": 1.0,
        "source_coordinate": coordinate.tolist(),
    }
    rows = np.tile(np.asarray([[190, 322]], dtype=np.int64), (9, 1))
    columns = np.tile(np.asarray([[256, 256]], dtype=np.int64), (9, 1))
    observed = stephenson._predict_method(model, rows, columns)
    expected = stephenson.expected_binary_table_from_log_odds(
        coordinate[0], rows[0], columns[0]
    )
    np.testing.assert_allclose(observed[0, 0], expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        observed.sum(axis=-1),
        np.broadcast_to(rows[:, None], (9, 9, 2)),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        observed.sum(axis=-2),
        np.broadcast_to(columns[None, :], (9, 9, 2)),
        atol=1e-10,
    )


def test_transport_alpha_scales_conditional_and_classical_coordinates() -> None:
    rows = np.tile(np.asarray([[190, 322]], dtype=np.int64), (9, 1))
    columns = np.tile(np.asarray([[256, 256]], dtype=np.int64), (9, 1))
    coordinate = np.full(81, 0.8)
    conditional = stephenson._predict_method(
        {
            "kind": "conditional_log_odds",
            "alpha": 0.5,
            "source_coordinate": coordinate.tolist(),
        },
        rows,
        columns,
    )
    np.testing.assert_allclose(
        conditional[0, 0],
        stephenson.expected_binary_table_from_log_odds(0.4, rows[0], columns[0]),
        atol=1e-12,
    )
    classical = stephenson._predict_method(
        {
            "kind": "classical_residual",
            "family": "pearson",
            "centered": False,
            "alpha": 0.5,
            "source_coordinate": coordinate.tolist(),
        },
        rows,
        columns,
    )
    expected, _ = stephenson._classical_table(
        0.4 * np.sqrt(512.0), rows[0], columns[0], "pearson"
    )
    np.testing.assert_allclose(classical[0, 0], expected, atol=1e-12)


def test_frozen_grid_contains_all_transport_alpha_values() -> None:
    configurations = [
        stephenson._conditional_configuration(config, alpha)
        for config in stephenson.CONDITIONAL_GRID
        for alpha in stephenson.ALPHA_GRID
    ]
    assert len(configurations) == 144
    assert {row["transport_alpha"] for row in configurations} == {
        0.5,
        0.75,
        1.0,
        1.25,
    }


def test_score_attempt_precedes_first_held_truth_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source.json"
    development_path = tmp_path / "development.json"
    prediction_path = tmp_path / "prediction.json"
    authorization_path = tmp_path / "authorization.json"
    attempt_path = tmp_path / "attempt.json"
    output_path = tmp_path / "score.json"
    for path in (source_path, development_path, prediction_path, authorization_path):
        path.write_text("{}")
    _patch_entry_defaults(
        monkeypatch,
        DEFAULT_SOURCE=source_path,
        DEFAULT_DEVELOPMENT=development_path,
        DEFAULT_PREDICTION=prediction_path,
        DEFAULT_SCORE_AUTHORIZATION=authorization_path,
        DEFAULT_SCORE_ATTEMPT=attempt_path,
        DEFAULT_SCORE=output_path,
    )
    source = {
        "source_sha256": stephenson._sha256(source_path),
        "h5ad": tmp_path / "opaque.h5ad",
    }
    monkeypatch.setattr(
        stephenson,
        "_validated_score_authorization",
        lambda *_: {"authorization_sha256": "a" * 64},
    )
    monkeypatch.setattr(stephenson, "_validated_source", lambda *_args, **_kw: source)
    monkeypatch.setattr(stephenson, "_validated_prediction", lambda *_: {"samples": []})
    monkeypatch.setattr(stephenson, "_held_records", lambda *_: [])
    monkeypatch.setattr(stephenson, "_selected_rows", lambda *_: {})

    def refuse_after_attempt(*_: object) -> dict[str, np.ndarray]:
        assert attempt_path.is_file()
        raise RuntimeError("truth read sentinel")

    monkeypatch.setattr(stephenson, "_read_modality", refuse_after_attempt)
    with pytest.raises(RuntimeError, match="truth read sentinel"):
        stephenson.score_held(
            source_path,
            development_path,
            prediction_path,
            authorization_path,
            "b" * 40,
            attempt_path,
            output_path,
        )


def test_score_authorization_rejects_arbitrary_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prediction = tmp_path / "prediction.json"
    source = tmp_path / "source.json"
    development = tmp_path / "development.json"
    authorization = tmp_path / "authorization.json"
    prediction.write_text("{}")
    source.write_text("{}")
    development.write_text("{}")
    commit = "a" * 40
    payload = {
        "schema": "stephenson-citeseq-score-authorization/1.0",
        "status": "OUTCOME_ACCESS_AUTHORIZED",
        "created_at_utc": "2026-08-28T00:00:00Z",
        "prediction_path": prediction.name,
        "prediction_sha256": stephenson._sha256(prediction),
        "prediction_bytes": prediction.stat().st_size,
        "runner_sha256": stephenson._sha256(Path(stephenson.__file__)),
        "source_manifest_sha256": stephenson._sha256(source),
        "development_sha256": stephenson._sha256(development),
        "public_prediction_commit": commit,
        "public_prediction_url": (
            f"https://github.com/other/repository/blob/{commit}/{prediction.name}"
        ),
    }
    _write_json(authorization, payload)
    monkeypatch.setattr(stephenson, "_validated_prediction", lambda *_: {})
    monkeypatch.setattr(stephenson, "_relative", lambda path: path.name)
    with pytest.raises(PermissionError, match="pinned repository"):
        stephenson._validated_score_authorization(
            authorization,
            prediction,
            source,
            development,
            "b" * 40,
        )


def test_exact_binomial_sign_test_uses_frozen_56_donors() -> None:
    values = np.asarray([-1.0] * 45 + [1.0] * 11)
    result = stephenson._exact_sign_test(values)
    assert result["favorable_donors"] == 45
    assert result["one_sided_p"] <= 0.025
    with pytest.raises(ValueError, match="56"):
        stephenson._exact_sign_test(values[:-1])
