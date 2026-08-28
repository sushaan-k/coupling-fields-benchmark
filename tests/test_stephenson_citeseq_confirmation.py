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


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"status":"A","status":"B"}\n')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        stephenson._read_json(path)


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


def test_development_attempt_precedes_first_numeric_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source.json"
    authorization = tmp_path / "authorization.json"
    attempt = tmp_path / "attempt.json"
    output = tmp_path / "development.json"
    source_path.write_text("{}")
    authorization.write_text("{}")
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
    attempt.write_text("{}")
    monkeypatch.setattr(
        stephenson,
        "_validated_development_authorization",
        lambda *_: pytest.fail("validation must not run on rerun"),
    )
    with pytest.raises(FileExistsError, match="one-shot"):
        stephenson.run_development(
            tmp_path / "source",
            tmp_path / "auth",
            "a" * 40,
            attempt,
            tmp_path / "output",
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
