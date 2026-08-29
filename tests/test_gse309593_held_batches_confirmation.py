from __future__ import annotations

import csv
import gzip
import inspect
import json
from pathlib import Path
import platform
import stat
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from experiments import confirm_gse309593_held_batches as subject
from mapreg.heterogeneity_adaptive_coupling import CouplingEstimationRefusal


def _markers() -> list[dict[str, str]]:
    return [
        {"rna_symbol": rna, "adt_target": adt} for rna, adt in subject.EXPECTED_PANEL
    ]


def _samples() -> list[dict[str, str]]:
    batches = (
        ("B092", 3),
        ("B099", 4),
        ("B110", 5),
        ("B129", 2),
    )
    output = []
    index = 0
    for batch, count in batches:
        for _ in range(count):
            index += 1
            output.append({"subject_id": f"S{index:02d}", "batch": batch})
    return output


def _write_synthetic_h5(path: Path) -> tuple[dict[str, object], list[str]]:
    marker_names = [marker[0] for marker in subject.EXPECTED_PANEL]
    names = marker_names + [f"GENE{index:03d}" for index in range(196)]
    types = ["Gene Expression"] * len(names)
    barcodes = [f"cell-{index:04d}" for index in range(520)]
    features_per_cell = len(names)
    indices = np.tile(np.arange(features_per_cell, dtype=np.int32), len(barcodes))
    data = np.ones(len(indices), dtype=np.int16)
    indptr = np.arange(0, len(indices) + 1, features_per_cell, dtype=np.int64)
    with h5py.File(path, "w") as handle:
        matrix = handle.create_group("matrix")
        matrix.create_dataset("barcodes", data=np.asarray(barcodes, dtype="S"))
        features = matrix.create_group("features")
        features.create_dataset("name", data=np.asarray(names, dtype="S"))
        features.create_dataset("feature_type", data=np.asarray(types, dtype="S"))
        matrix.create_dataset("data", data=data)
        matrix.create_dataset("indices", data=indices)
        matrix.create_dataset("indptr", data=indptr)
        matrix.create_dataset(
            "shape", data=np.asarray([len(names), len(barcodes)], dtype=np.int64)
        )
        embedded = handle.create_group("ADT")
        embedded.create_dataset("data", data=np.full((4, 4), 99, dtype=np.int64))
    schema = {
        "feature_count": len(names),
        "feature_name_axis_sha256": subject._newline_axis_sha256(names),
        "feature_type_axis_sha256": subject._newline_axis_sha256(types),
    }
    return schema, barcodes


def _write_adt_csv(path: Path, barcodes: list[str]) -> dict[str, object]:
    labels = [marker[1] for marker in subject.EXPECTED_PANEL]
    with gzip.open(path, "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["barcode", *labels])
        for index, barcode in enumerate(barcodes):
            writer.writerow([barcode, *[(index + marker) % 7 for marker in range(24)]])
    return {
        "feature_count": len(labels),
        "axis_sha256": subject._newline_axis_sha256(labels),
    }


def test_candidate_and_axis_preflight_freeze_all_24_exact_cognates() -> None:
    designation, source_samples, held_samples = subject._designation(
        subject.DEFAULT_DESIGNATION
    )

    assert len(designation["strict_cognates"]) == 24
    assert (
        tuple(
            (marker["rna_symbol"], marker["adt_target"])
            for marker in designation["strict_cognates"]
        )
        == subject.EXPECTED_PANEL
    )
    assert [sample["batch"] for sample in source_samples].count("B092") == 3
    assert len(source_samples) == 14
    assert len(held_samples) == 9
    assert {sample["subject_id"] for sample in source_samples}.isdisjoint(
        {sample["subject_id"] for sample in held_samples}
    )


def test_held_h5_reader_uses_only_allowlisted_matrix_datasets(tmp_path: Path) -> None:
    path = tmp_path / "sample.h5"
    schema, barcodes = _write_synthetic_h5(path)
    reduced = subject._read_rna_h5(
        path,
        {"subject_id": "FHX"},
        _markers(),
        schema,
    )

    assert set(reduced["accessed_h5_datasets"]) == subject.H5_DATASET_ALLOWLIST
    assert all(
        dataset.startswith("matrix/") for dataset in reduced["accessed_h5_datasets"]
    )
    assert reduced["rna"].shape == (512, 24)
    assert reduced["qc_eligible_cells"] == 520
    ranked = sorted(
        range(len(barcodes)),
        key=lambda index: (
            subject.hashlib.sha256(
                f"{subject.CELL_SALT}|FHX|{barcodes[index]}".encode()
            ).hexdigest(),
            barcodes[index],
        ),
    )[:512]
    assert reduced["barcodes"] == [barcodes[index] for index in sorted(ranked)]


def test_h5_reader_rejects_out_of_allowlist_and_axis_drift(tmp_path: Path) -> None:
    path = tmp_path / "sample.h5"
    schema, _ = _write_synthetic_h5(path)
    with h5py.File(path, "r") as handle:
        reader = subject._AllowlistedH5(handle)
        with pytest.raises(PermissionError, match="outside the allowlist"):
            reader.read("ADT/data")

    drifted = dict(schema)
    drifted["feature_name_axis_sha256"] = "0" * 64
    audit = subject._new_access_audit("prediction")
    with pytest.raises(PermissionError, match="RNA feature axis differs"):
        subject._read_rna_h5(
            path,
            {"subject_id": "FHX"},
            _markers(),
            drifted,
            audit,
            "held",
        )
    assert audit["unique_h5_dataset_paths_with_started_read_or_length"]["held"] == [
        "matrix/barcodes",
        "matrix/features/feature_type",
        "matrix/features/name",
    ]


def test_runtime_contract_checks_exact_hdf5_built_against_tuple(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "runtime.json"
    monkeypatch.setattr(subject.sys, "executable", str(Path(subject.sys.executable)))
    runtime = {
        "schema": "gse309593-held-batches-runtime-environment/1.0",
        "status": "FROZEN_WITH_PROTOCOL",
        "required_runtime": {
            "python": {
                "implementation": platform.python_implementation(),
                "version": platform.python_version(),
                "resolved_executable": str(Path(subject.sys.executable).resolve()),
            },
            "packages": {
                "numpy": np.__version__,
                "scipy": subject.scipy.__version__,
                "h5py": h5py.__version__,
            },
            "hdf5": {
                "runtime_version_tuple": list(h5py.version.hdf5_version_tuple),
                "built_against_version_tuple": list(
                    h5py.version.hdf5_built_version_tuple
                ),
                "h5py_api_version_tuple": list(h5py.version.api_version_tuple),
            },
            "platform": {
                "operating_system": (
                    "macOS" if platform.system() == "Darwin" else platform.system()
                ),
                "architecture": platform.machine(),
            },
            "thread_environment": {},
        },
    }
    path.write_text(json.dumps(runtime))
    subject._validate_runtime(path)

    runtime["required_runtime"]["hdf5"]["built_against_version_tuple"] = [0, 0, 0]
    path.write_text(json.dumps(runtime))
    with pytest.raises(PermissionError, match="HDF5 built-against"):
        subject._validate_runtime(path)


def test_interrupted_download_removes_partial_assay_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class InterruptedResponse:
        reads = 0

        def __enter__(self) -> "InterruptedResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, size: int) -> bytes:
            del size
            self.reads += 1
            if self.reads == 1:
                return b"partial-assay-bytes"
            raise KeyboardInterrupt("synthetic interruption")

    monkeypatch.setattr(subject, "urlopen", lambda url: InterruptedResponse())
    with pytest.raises(KeyboardInterrupt, match="synthetic interruption"):
        subject._fetch_designated_file(
            "https://example.invalid/assay",
            100,
            tmp_path,
            ".h5",
        )

    assert list(tmp_path.iterdir()) == []


def test_digest_mismatch_audit_never_labels_computed_hash_as_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"synthetic-public-file"

    class Response:
        def __enter__(self) -> "Response":
            self.done = False
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, size: int) -> bytes:
            del size
            if self.done:
                return b""
            self.done = True
            return payload

    sample = {
        "subject_id": "FHX",
        "gsm": "GSMX",
        "batch": "B162",
        "rna_h5": {"name": "sample.h5", "bytes": len(payload)},
        "rna_h5_sha256": "0" * 64,
    }
    designation = {"url_template": "https://example.invalid/{gsm}/{file}"}
    audit = subject._new_access_audit("prediction")
    monkeypatch.setattr(subject, "urlopen", lambda url: Response())

    with pytest.raises(PermissionError, match="SHA-256 differs"):
        subject._fetch_sample(
            designation, sample, "rna_h5", tmp_path, audit, "held"
        )

    completed = [
        event for event in audit["events"] if event["event"] == "file_hash_completed"
    ]
    assert len(completed) == 1
    assert completed[0]["computed_sha256"] == subject.hashlib.sha256(payload).hexdigest()
    assert "verified_sha256" not in completed[0]
    assert not any(
        event["event"] == "file_designation_verified" for event in audit["events"]
    )
    assert audit["counters"]["held_rna_h5_files_deleted"] == 1


def test_adt_csv_axis_and_selected_identifiers_are_exact(tmp_path: Path) -> None:
    barcodes = [f"cell-{index:04d}" for index in range(512)]
    path = tmp_path / "adt.csv.gz"
    schema = _write_adt_csv(path, barcodes)

    counts = subject._read_adt_csv(path, barcodes, _markers(), schema)
    assert counts.shape == (512, 24)
    assert counts[17, 3] == (17 + 3) % 7

    drifted = dict(schema)
    drifted["axis_sha256"] = "f" * 64
    with pytest.raises(PermissionError, match="ADT header axis differs"):
        subject._read_adt_csv(path, barcodes, _markers(), drifted)
    with pytest.raises(ValueError, match="selected RNA barcode is missing"):
        subject._read_adt_csv(
            path,
            [*barcodes[:-1], "missing-cell"],
            _markers(),
            schema,
        )


def test_midrank_is_exact_and_deterministic_even_when_every_count_ties() -> None:
    barcodes = [f"cell-{index:04d}" for index in range(512)]
    counts = np.zeros((512, 24), dtype=np.int64)
    first = subject._adt_states(counts, barcodes, "FHX")
    second = subject._adt_states(counts, barcodes, "FHX")

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first.sum(axis=0), np.full(24, 256))
    destroyed = subject._destroyed_adt(first, barcodes, "FHX")
    np.testing.assert_array_equal(destroyed.sum(axis=0), first.sum(axis=0))
    assert sorted(map(tuple, destroyed.tolist())) == sorted(map(tuple, first.tolist()))


def test_raw_adt_variation_support_excludes_tied_marker_columns() -> None:
    counts = np.zeros((512, 24), dtype=np.int64)
    counts[256:, 1:] = 1
    support = subject._adt_variation_support(counts)

    assert not support[0]
    assert support[1:].all()
    tables = np.tile(np.asarray([[10, 5], [5, 10]]), (24, 24, 1, 1))
    pair_support = subject._subject_support(tables, support)
    assert not pair_support[:, 0].any()
    assert pair_support[:, 1:].all()


def test_source_comparison_mask_removes_boundary_coordinate_in_every_fold() -> None:
    samples = _samples()
    interior = np.tile(np.asarray([[10, 5], [5, 10]], dtype=np.int64), (24, 24, 1, 1))
    records = {}
    for sample in samples:
        tables = interior.copy()
        tables[0, 0] = np.asarray([[20, 0], [0, 20]])
        records[sample["subject_id"]] = {
            "tables": tables,
            "subject_support": np.ones((24, 24), dtype=bool),
            "pooled_support": np.ones((24, 24), dtype=bool),
        }

    mask, details = subject._source_comparison_mask(records, samples)

    assert mask.shape == (24, 24)
    assert not mask[0, 0]
    assert np.count_nonzero(mask) == 575
    assert details["coordinate_count"] == 575
    assert details["mask_sha256"] == subject._array_sha256(mask.astype(np.uint8))
    assert set(details["folds"]) == {
        "lobo_without_B092",
        "lobo_without_B099",
        "lobo_without_B110",
        "lobo_without_B129",
        "final_all_source",
    }


def test_common_effect_adapter_uses_coordinatewise_solver_and_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = SimpleNamespace(
        log_odds=np.zeros(24 * 24),
        gradient_norm=1e-12,
        scaled_gradient_norm=2e-13,
        data_precision=np.full((24, 24), 3.0),
        support_count=np.full((24, 24), 12),
        root_iterations=np.full((24, 24), 7),
    )
    calls = []

    def fit(tables: np.ndarray, **kwargs: object) -> object:
        calls.append((tables.shape, kwargs))
        return fake

    monkeypatch.setattr(subject, "fit_common_effect_conditional_log_odds", fit)
    tables = np.tile(
        np.asarray([[10, 5], [5, 10]], dtype=int), (14, 24, 24, 1, 1)
    )
    result = subject._fit_common_effect(tables)

    assert calls == [
        ((14, 576, 2, 2), {"minimum_informative_donors": 2, "tolerance": 1e-10})
    ]
    assert result["fit_certificate"] == {
        "gradient_norm": 1e-12,
        "scaled_gradient_norm": 2e-13,
        "minimum_data_precision": 3.0,
        "maximum_data_precision": 3.0,
        "minimum_support_count": 12,
        "maximum_root_iterations": 7,
    }
    assert "fit_structured_conditional_log_odds" not in inspect.getsource(
        subject._fit_common_effect
    )


def test_exact_fits_accept_mixed_informative_and_missing_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "MARKER_COUNT", 2)
    tables = np.tile(
        np.asarray([[10, 5], [5, 10]], dtype=np.int64), (3, 2, 2, 1, 1)
    )
    support = np.ones((3, 2, 2), dtype=bool)
    support[0, 0, 0] = False

    encoded = subject._conditional_missing_tables(tables, support)
    np.testing.assert_array_equal(encoded[0, 0, 0], [[30, 0], [0, 0]])
    common = subject._fit_common_effect(
        tables, np.ones((2, 2), dtype=bool), support
    )
    primary = subject._fit_primary(
        tables,
        np.zeros((3, 2)),
        np.zeros((3, 2)),
        subject.PrimaryConfig(2, 0.1, 0.01, 0.0, 0.0),
        support,
    )

    assert np.isfinite(common["population_log_odds"]).all()
    assert common["fit_certificate"]["minimum_support_count"] == 2
    assert np.isfinite(primary["population_log_odds"]).all()


def test_pooled_poisson_includes_raw_varying_margin_degenerate_subjects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "MARKER_COUNT", 2)
    tables = np.tile(
        np.asarray([[10, 5], [5, 10]], dtype=np.int64), (3, 2, 2, 1, 1)
    )
    tables[0] = np.asarray([[5, 0], [5, 0]], dtype=np.int64)
    pooled_support = np.ones((3, 2, 2), dtype=bool)
    mask = np.ones((2, 2), dtype=bool)

    result = subject._fit_pooled_poisson(tables, mask, pooled_support)
    expected_pooled = tables.sum(axis=0)[mask]
    certificate = result["fit_certificate"]

    assert result["pooled_tables_sha256"] == subject._array_sha256(expected_pooled)
    assert certificate["ordered_adt_marker_pooled_subject_counts"] == [3, 3]
    counts = np.asarray([3, 3], dtype=np.int64)
    assert certificate["ordered_adt_marker_pooled_subject_counts_sha256"] == (
        subject._array_sha256(counts)
    )
    assert certificate["maximum_normalized_reconstruction_error"] <= 1e-8
    assert certificate["passes"] is True


def _patch_lightweight_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    valid = subject.PrimaryConfig(2, 0.1, 0.01, 0.0, 0.0)
    invalid = subject.PrimaryConfig(2, 0.1, 0.1, 0.0, 0.0)
    monkeypatch.setattr(subject, "TRANSPORT_GRID", (0.0,))
    monkeypatch.setattr(subject, "_primary_configs", lambda: [valid, invalid])

    def fit_primary(
        tables: np.ndarray,
        rna_profiles: np.ndarray,
        adt_profiles: np.ndarray,
        config: subject.PrimaryConfig,
        support: np.ndarray,
    ) -> dict[str, np.ndarray]:
        if config.ridge_penalty == 0.1:
            raise CouplingEstimationRefusal("prespecified invalid configuration")
        return {"population_log_odds": np.zeros((24, 24))}

    monkeypatch.setattr(subject, "_fit_primary", fit_primary)
    monkeypatch.setattr(
        subject,
        "_residual_pool",
        lambda tables, family, mask, support: np.zeros((24, 24)),
    )
    monkeypatch.setattr(
        subject,
        "_predict_residual",
        lambda pooled, rows, columns, config: np.full((24, 24, 2, 2), 1.0),
    )
    monkeypatch.setattr(
        subject,
        "_fit_common_effect",
        lambda tables, mask, support: {"population_log_odds": np.full((24, 24), 2.0)},
    )
    def fit_pooled(
        tables: np.ndarray, mask: np.ndarray, support: np.ndarray
    ) -> dict[str, object]:
        marker_counts = np.asarray(support, dtype=bool)[:, 0, :].sum(axis=0).astype(
            np.int64
        )
        return {
            "population_log_odds": np.full((24, 24), 3.0),
            "pooled_tables_sha256": "b" * 64,
            "fit_certificate": {
                "ordered_adt_marker_pooled_subject_counts": marker_counts.tolist(),
                "ordered_adt_marker_pooled_subject_counts_sha256": subject._array_sha256(
                    marker_counts
                ),
                "maximum_normalized_reconstruction_error": 0.0,
                "passes": True,
            },
        }

    monkeypatch.setattr(subject, "_fit_pooled_poisson", fit_pooled)

    def predict_odds(
        log_odds: np.ndarray, rows: np.ndarray, columns: np.ndarray, alpha: float
    ) -> np.ndarray:
        value = float(np.asarray(log_odds).flat[0])
        return np.full((24, 24, 2, 2), 0.1 if value == 0.0 else value)

    monkeypatch.setattr(subject, "_predict_odds", predict_odds)
    monkeypatch.setattr(
        subject,
        "_independence",
        lambda rows, columns: np.full((24, 24, 2, 2), 4.0),
    )
    monkeypatch.setattr(
        subject,
        "_loss",
        lambda truth, prediction, evaluation_mask=None: float(prediction.flat[0]),
    )
    full_mask = np.ones((24, 24), dtype=bool)
    monkeypatch.setattr(
        subject,
        "_source_comparison_mask",
        lambda records, samples: (
            full_mask,
            {
                "coordinate_count": 576,
                "mask_sha256": subject._array_sha256(full_mask.astype(np.uint8)),
                "hash_encoding": "test",
                "folds": {},
                "source_subject_supported_coordinate_counts": {
                    sample["subject_id"]: 576 for sample in samples
                },
                "checks": {
                    "at_least_288_coordinates_retained": True,
                    "every_source_subject_has_at_least_64_supported_coordinates": True,
                },
            },
        ),
    )


def _selection_records() -> dict[str, dict[str, np.ndarray]]:
    records = {}
    for sample in _samples():
        records[sample["subject_id"]] = {
            "tables": np.zeros((24, 24, 2, 2), dtype=int),
            "destroyed_tables": np.zeros((24, 24, 2, 2), dtype=int),
            "rna_profile": np.zeros(24),
            "adt_profile": np.zeros(24),
            "informative_pair_count": 576,
            "subject_support": np.ones((24, 24), dtype=bool),
            "pooled_support": np.ones((24, 24), dtype=bool),
        }
    return records


def _source_access() -> dict[str, int]:
    return {
        "source_rna_h5_files_requested": 14,
        "source_rna_h5_files_read": 14,
        "source_rna_h5_reductions_completed": 14,
        "source_adt_csv_files_requested": 14,
        "source_adt_csv_files_read": 14,
        "source_adt_csv_reductions_completed": 14,
        "source_embedded_h5_adt_datasets_read": 0,
        "held_rna_h5_files_requested": 0,
        "held_rna_h5_files_read": 0,
        "held_adt_csv_files_requested": 0,
        "held_adt_csv_files_opened": 0,
        "held_adt_csv_hashes_completed": 0,
        "held_adt_identifiers_read": 0,
        "held_adt_numeric_values_read": 0,
        "held_adt_states_formed": 0,
        "held_joint_tables_formed": 0,
    }


def _prediction_access() -> dict[str, object]:
    return {
        "held_rna_h5_files_requested": 9,
        "held_rna_h5_files_read": 9,
        "held_rna_h5_reductions_completed": 9,
        "held_h5_dataset_allowlist": sorted(subject.H5_DATASET_ALLOWLIST),
        "held_h5_unique_decoded_dataset_set": sorted(subject.H5_DATASET_ALLOWLIST),
        "held_h5_datasets_outside_allowlist_read": 0,
        "held_embedded_h5_adt_datasets_read": 0,
        "held_adt_csv_files_requested": 0,
        "held_adt_csv_files_opened": 0,
        "held_adt_csv_hashes_completed": 0,
        "held_adt_identifiers_read": 0,
        "held_adt_numeric_values_read": 0,
        "held_adt_states_formed": 0,
        "held_joint_tables_formed": 0,
    }


def _valid_source_artifact(samples: list[dict[str, object]]) -> dict[str, object]:
    mask = np.ones((24, 24), dtype=np.uint8)
    mask_record = {
        "mask": mask.tolist(),
        "mask_sha256": subject._array_sha256(mask),
        "coordinate_count": 576,
    }
    log_odds = np.zeros((24, 24)).tolist()
    residual = np.zeros((24, 24)).tolist()
    models: dict[str, object] = {
        method: {"configuration": {}, "population_log_odds": log_odds}
        for method in (
            "primary",
            "common_effect_cmle",
            "pooled_saturated_poisson",
            "destroyed_link",
        )
    }
    models.update(
        {
            method: {"configuration": {}, "pooled_coordinate": residual}
            for method in (
                "selected_residual",
                "pearson_residual",
                "root_deviance_residual",
            )
        }
    )
    models["independence"] = {"family": "independence"}
    models["comparison_mask"] = dict(mask_record)
    return {
        "schema": "gse309593-held-batches-source/1.0",
        "status": "SOURCE_GATE_PASS",
        "source_samples": subject._frozen_sample_axis(samples),
        "source_subjects": [sample["subject_id"] for sample in samples],
        "source_gsms": [sample["gsm"] for sample in samples],
        "input_files": {
            sample["subject_id"]: {
                "rna_h5_name": sample["rna_h5"]["name"],
                "rna_h5_bytes": sample["rna_h5"]["bytes"],
                "rna_h5_sha256": sample.get("rna_h5_sha256", "a" * 64),
                "adt_csv_gz_name": sample["adt_csv_gz"]["name"],
                "adt_csv_gz_bytes": sample["adt_csv_gz"]["bytes"],
                "adt_csv_gz_sha256": sample.get("adt_csv_gz_sha256", "b" * 64),
                "h5_datasets_read": sorted(subject.H5_DATASET_ALLOWLIST),
            }
            for sample in samples
        },
        "access": _source_access(),
        "selection": {
            "source_gate": {"passes": True, "checks": {"all": True}},
            "comparison_mask": mask_record,
            "source_selected_best_classical": "pooled_saturated_poisson",
        },
        "models": models,
    }


def _valid_prediction_artifact(
    samples: list[dict[str, object]], source_artifact: dict[str, object]
) -> dict[str, object]:
    states = np.tile(
        np.concatenate(
            [np.zeros(256, dtype=np.uint8), np.ones(256, dtype=np.uint8)]
        )[:, None],
        (1, 24),
    )
    rows, columns = subject._held_margins(states)
    predicted = subject._independence(rows, columns)
    methods = {
        "primary",
        "selected_residual",
        "pearson_residual",
        "root_deviance_residual",
        "common_effect_cmle",
        "pooled_saturated_poisson",
        "independence",
        "destroyed_link",
    }
    mask_record = source_artifact["selection"]["comparison_mask"]
    support = subject._informative_margin_count(
        rows, columns, np.asarray(mask_record["mask"], dtype=bool)
    )
    return {
        "schema": "gse309593-held-batches-prediction/1.0",
        "status": "HELD_MARGIN_ONLY_PREDICTIONS_FROZEN",
        "held_batches": list(subject.HELD_BATCHES),
        "held_subject_count": len(samples),
        "private_rna_sha256": "d" * 64,
        "private_rna_bytes": 1,
        "source_selected_best_classical": "pooled_saturated_poisson",
        "comparison_mask_sha256": mask_record["mask_sha256"],
        "comparison_mask_coordinate_count": mask_record["coordinate_count"],
        "held_informative_margin_pair_counts": {
            sample["subject_id"]: support for sample in samples
        },
        "samples": [
            {
                "subject_id": sample["subject_id"],
                "gsm": sample["gsm"],
                "batch": sample["batch"],
                "rna_h5_name": sample["rna_h5"]["name"],
                "rna_h5_bytes": sample["rna_h5"]["bytes"],
                "rna_h5_sha256": sample.get("rna_h5_sha256", "c" * 64),
                "h5_datasets_read": sorted(subject.H5_DATASET_ALLOWLIST),
                "barcode_axis_sha256": "e" * 64,
                "selected_cell_axis_sha256": "f" * 64,
                "rna_state_sha256": "1" * 64,
                "row_margins": rows.tolist(),
                "column_margins": columns.tolist(),
                "informative_margin_pair_count": support,
                "predicted_tables": {
                    method: predicted.tolist() for method in methods
                },
                "prediction_sha256": {
                    method: subject._array_sha256(predicted) for method in methods
                },
            }
            for sample in samples
        ],
        "access": _prediction_access(),
    }


def test_invalid_primary_grid_point_is_ineligible_not_study_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_lightweight_selection(monkeypatch)
    result = subject._select_source(_selection_records(), _samples())

    assert result["source_gate"]["passes"]
    assert result["candidate_counts"]["primary_frozen"] == 2
    assert result["candidate_counts"]["primary_complete"] == 1
    assert result["candidate_counts"]["destroyed_transport_frozen"] == 1
    assert result["destroyed_link"]["selected_transport_multiplier"] == 0.0
    assert result["destroyed_link"]["transport_curve"] == [
        {
            "transport_multiplier": 0.0,
            "status": "COMPLETE",
            "sample_axis": [sample["subject_id"] for sample in _samples()],
            "fold_losses": [0.1] * 14,
            "equal_batch_mean_loss": 0.1,
            "refusals": [],
        }
    ]
    certificates = result["pooled_poisson_lobo_fit_certificates"]
    assert set(certificates) == set(subject.SOURCE_BATCHES)
    assert all(
        certificate["pooled_tables_sha256"] == "b" * 64
        for certificate in certificates.values()
    )
    assert len(result["refusals"]) == 4
    assert all(refusal["method"] == "primary" for refusal in result["refusals"])


def test_mandatory_common_effect_failure_has_structured_refusal_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_lightweight_selection(monkeypatch)
    monkeypatch.setattr(
        subject,
        "_fit_common_effect",
        lambda tables, mask, support: (_ for _ in ()).throw(
            CouplingEstimationRefusal("boundary")
        ),
    )

    with pytest.raises(subject.SelectionRefusal) as captured:
        subject._select_source(_selection_records(), _samples())
    assert not captured.value.details["eligibility"][
        "all_common_effect_transport_multipliers_complete"
    ]
    assert any(
        refusal["method"] == "common_effect_cmle"
        for refusal in captured.value.details["refusals"]
    )


def test_predecessor_access_validators_require_exact_frozen_axes_and_optional_hashes() -> None:
    _, source_samples, held_samples = subject._designation(subject.DEFAULT_DESIGNATION)
    source_artifact = _valid_source_artifact(source_samples)
    prediction_artifact = _valid_prediction_artifact(held_samples, source_artifact)

    subject._validate_source_access_certificate(source_artifact, source_samples)
    subject._validate_prediction_access_certificate(
        prediction_artifact, held_samples, source_artifact
    )

    reordered = dict(source_artifact)
    reordered["source_samples"] = list(reversed(source_artifact["source_samples"]))
    with pytest.raises(PermissionError, match="frozen sample axis"):
        subject._validate_source_access_certificate(reordered, source_samples)

    substituted = dict(prediction_artifact)
    substituted["samples"] = [dict(sample) for sample in prediction_artifact["samples"]]
    substituted["samples"][0]["gsm"] = "SUBSTITUTED"
    with pytest.raises(PermissionError, match="frozen sample axis"):
        subject._validate_prediction_access_certificate(
            substituted, held_samples, source_artifact
        )

    extra = dict(prediction_artifact)
    extra["samples"] = [*prediction_artifact["samples"], "ignored-extra"]
    with pytest.raises(PermissionError, match="frozen sample axis"):
        subject._validate_prediction_access_certificate(extra, held_samples, source_artifact)

    corrupted = dict(prediction_artifact)
    corrupted["samples"] = [dict(sample) for sample in prediction_artifact["samples"]]
    corrupted["samples"][-1]["prediction_sha256"] = dict(
        corrupted["samples"][-1]["prediction_sha256"]
    )
    corrupted["samples"][-1]["prediction_sha256"]["primary"] = "0" * 64
    with pytest.raises(PermissionError, match="prediction table for primary differs"):
        subject._validate_prediction_access_certificate(
            corrupted, held_samples, source_artifact
        )


def test_margin_support_is_certified_before_held_adt_access() -> None:
    unsupported = np.zeros((512, 24), dtype=np.uint8)
    rows, columns = subject._held_margins(unsupported)
    assert subject._informative_margin_count(rows, columns) == 0

    supported = np.tile(
        np.concatenate([np.zeros(256, dtype=np.uint8), np.ones(256, dtype=np.uint8)])[
            :, None
        ],
        (1, 24),
    )
    rows, columns = subject._held_margins(supported)
    assert subject._informative_margin_count(rows, columns) == 576
    assert "_read_adt_csv" not in inspect.getsource(subject.run_prediction)


def test_private_state_stores_barcodes_and_rejects_state_or_axis_mutation(
    tmp_path: Path,
) -> None:
    samples = [{"subject_id": f"H{index}"} for index in range(9)]
    panels = {
        sample["subject_id"]: {
            "rna": np.tile(np.arange(512, dtype=np.uint16)[:, None] % 2, (1, 24)),
            "barcodes": [
                f"{sample['subject_id']}-cell-{cell:04d}" for cell in range(512)
            ],
        }
        for sample in samples
    }
    path = tmp_path / "private.npz"
    digest, byte_count, state_hashes = subject._write_private_rna(path, samples, panels)
    restored = subject._read_private_rna(path, samples)

    assert len(digest) == 64
    assert byte_count == path.stat().st_size
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    first = samples[0]["subject_id"]
    assert restored[first]["barcodes"] == panels[first]["barcodes"]
    rows, columns = subject._held_margins(panels[first]["rna"])
    frozen = {
        "selected_cell_axis_sha256": subject._axis_sha256(panels[first]["barcodes"]),
        "rna_state_sha256": state_hashes[first],
        "row_margins": rows.tolist(),
        "column_margins": columns.tolist(),
    }
    subject._verify_private_prediction(restored[first], frozen)
    mutated = {
        "barcodes": list(restored[first]["barcodes"]),
        "states": restored[first]["states"].copy(),
    }
    mutated["barcodes"][0] = "changed"
    with pytest.raises(PermissionError, match="state or barcode axis changed"):
        subject._verify_private_prediction(mutated, frozen)
    mutated = {
        "barcodes": list(restored[first]["barcodes"]),
        "states": restored[first]["states"].copy(),
    }
    mutated["states"][0, 0] ^= 1
    with pytest.raises(PermissionError, match="state or barcode axis changed"):
        subject._verify_private_prediction(mutated, frozen)


def test_batch_stratified_inference_uses_frozen_three_two_four_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "BOOTSTRAPS", 200)
    units = [f"H{index}" for index in range(9)]
    batches = ["B162"] * 3 + ["B208"] * 2 + ["B210"] * 4
    result = subject._comparison(
        units,
        batches,
        np.zeros(9),
        np.ones(9),
        17,
        formal_transfer=True,
    )

    assert result["passes"]
    assert result["favorable_non_tied_units"] == 9
    assert result["exact_one_sided_sign_test_p"] == pytest.approx(1 / 512)
    with pytest.raises(ValueError, match="3/2/4"):
        subject._comparison(
            units,
            ["B162"] * 3 + ["B208"] * 3 + ["B210"] * 3,
            np.zeros(9),
            np.ones(9),
            17,
            formal_transfer=True,
        )


def test_one_shot_consumes_keyboard_interrupt_and_keeps_paths_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "attempt.json"
    output = tmp_path / "output.json"
    alternate = tmp_path / "alternate.json"
    bindings = {"runner_sha256": "a" * 64}
    monkeypatch.setattr(subject, "DEFAULT_SOURCE_ATTEMPT", attempt)
    monkeypatch.setattr(subject, "DEFAULT_SOURCE", output)
    subject._claim(attempt, "source", bindings)
    monkeypatch.setattr(
        subject, "_require_public_attempt", lambda phase, path: "c" * 40
    )
    monkeypatch.setattr(subject, "_require_remote_completion_absent", lambda phase: None)

    def interrupted() -> dict[str, object]:
        raise KeyboardInterrupt(str(subject.ROOT / "private-input"))

    result = subject._one_shot("source", attempt, output, bindings, interrupted)

    assert result["status"] == "TERMINAL_SOURCE_REFUSAL"
    assert "<repository>" in result["reason"]
    assert str(subject.ROOT) not in output.read_text()
    assert "os.O_EXCL" in inspect.getsource(subject._one_shot)
    with pytest.raises(FileExistsError):
        subject._one_shot("source", attempt, output, bindings, interrupted)
    with pytest.raises(PermissionError, match="output path"):
        subject._one_shot("source", attempt, alternate, bindings, interrupted)


def test_selection_refusal_details_are_published_by_one_shot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "attempt.json"
    output = tmp_path / "output.json"
    bindings = {"runner_sha256": "a" * 64}
    monkeypatch.setattr(subject, "DEFAULT_SOURCE_ATTEMPT", attempt)
    monkeypatch.setattr(subject, "DEFAULT_SOURCE", output)
    subject._claim(attempt, "source", bindings)
    monkeypatch.setattr(
        subject, "_require_public_attempt", lambda phase, path: "c" * 40
    )
    monkeypatch.setattr(subject, "_require_remote_completion_absent", lambda phase: None)

    def refused() -> dict[str, object]:
        raise subject.SelectionRefusal(
            "mandatory eligibility failed",
            {"eligibility": {"common": False}, "refusals": [{"method": "common"}]},
        )

    result = subject._one_shot("source", attempt, output, bindings, refused)
    assert result["details"]["eligibility"] == {"common": False}
    assert result["details"]["refusals"] == [{"method": "common"}]


def test_claimed_stage_bootstrap_failure_writes_terminal_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "source-attempt.json"
    output = tmp_path / "source.json"
    bindings = {"runner_sha256": "a" * 64}
    monkeypatch.setattr(subject, "DEFAULT_SOURCE_ATTEMPT", attempt)
    monkeypatch.setattr(subject, "DEFAULT_SOURCE", output)
    subject._claim(attempt, "source", bindings)
    monkeypatch.setattr(
        subject, "_require_public_attempt", lambda phase, path: "c" * 40
    )
    monkeypatch.setattr(subject, "_require_remote_completion_absent", lambda phase: None)

    def refuse_bootstrap(*args: object) -> object:
        raise PermissionError("runtime binding changed")

    monkeypatch.setattr(subject, "_base_bindings", refuse_bootstrap)

    result = subject.run_source(attempt_path=attempt, output_path=output)

    assert result["status"] == "TERMINAL_SOURCE_REFUSAL"
    assert result["reason_code"] == "PermissionError"
    assert result["bindings"] == bindings
    assert output.is_file()
    assert result["incremental_nonnumeric_access_audit"]["events"] == []


@pytest.mark.parametrize(
    ("runner", "argument"),
    (
        (subject.run_source, "output_path"),
        (subject.run_prediction, "source_path"),
        (subject.authorize_score, "template_path"),
        (subject.run_score, "private_rna_path"),
    ),
)
def test_numeric_stages_reject_unbound_alternate_paths_before_access(
    tmp_path: Path, runner: object, argument: str
) -> None:
    with pytest.raises(PermissionError, match="frozen protocol path"):
        runner(**{argument: tmp_path / "alternate"})


def test_authorization_uses_verified_prediction_tag_not_current_head() -> None:
    source = inspect.getsource(subject.authorize_score)
    assert '_remote_tag_commit(template["prediction_tag"])' in source
    assert "_git_head" not in source
    template = subject._validate_authorization_template(
        subject.DEFAULT_AUTHORIZATION_TEMPLATE
    )
    assert template["held_adt_numeric_access_authorized"] is False


def test_score_authorization_is_validated_field_by_field_before_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source.json"
    prediction_path = tmp_path / "prediction.json"
    private_path = tmp_path / "private.npz"
    source_path.write_bytes(b"source")
    prediction_path.write_bytes(b"prediction")
    private_path.write_bytes(b"private")
    prediction_commit = "c" * 40
    monkeypatch.setattr(subject, "_remote_tag_commit", lambda tag: prediction_commit)
    monkeypatch.setattr(subject, "_relative", lambda path: path.name)
    bindings = {
        "protocol_sha256": "1" * 64,
        "runtime_sha256": "2" * 64,
        "runner_sha256": "3" * 64,
        "test_sha256": "4" * 64,
        "common_effect_solver_sha256": "5" * 64,
        "hierarchical_module_sha256": "6" * 64,
        "coupling_module_sha256": "7" * 64,
    }
    authorization = {
        "schema": "gse309593-held-batches-score-authorization/1.0",
        "status": "SCORE_AUTHORIZED_WITHOUT_OUTCOME_ACCESS",
        "held_adt_numeric_access_authorized": True,
        "outcome_access_authorized": True,
        "held_adt_files_opened_before_authorization": 0,
        "held_adt_numeric_values_read_before_authorization": 0,
        "held_joint_tables_formed_before_authorization": 0,
        "prediction_tag": "gse309593-held-batches-v1-prediction",
        "prediction_commit": prediction_commit,
        "prediction_path": prediction_path.name,
        "prediction_sha256": subject._sha256(prediction_path),
        "prediction_bytes": prediction_path.stat().st_size,
        "private_rna_state_sha256": subject._sha256(private_path),
        "private_rna_state_bytes": private_path.stat().st_size,
        "source_output_sha256": subject._sha256(source_path),
        "protocol_sha256": bindings["protocol_sha256"],
        "runtime_environment_sha256": bindings["runtime_sha256"],
        "runner_sha256": bindings["runner_sha256"],
        "test_sha256": bindings["test_sha256"],
        "coordinatewise_common_effect_solver_sha256": bindings[
            "common_effect_solver_sha256"
        ],
        "hierarchical_solver_sha256": bindings["hierarchical_module_sha256"],
        "coupling_module_sha256": bindings["coupling_module_sha256"],
        "transitive_bindings": bindings,
        "bindings": bindings,
    }
    subject._validate_score_authorization(
        authorization, bindings, source_path, prediction_path, private_path
    )

    mutated = dict(authorization)
    mutated["held_joint_tables_formed_before_authorization"] = 1
    with pytest.raises(PermissionError, match="authorization certificate"):
        subject._validate_score_authorization(
            mutated, bindings, source_path, prediction_path, private_path
        )


def test_score_reads_only_private_rna_and_separate_adt_csv() -> None:
    score_source = inspect.getsource(subject.run_score)
    assert "_read_rna_h5" not in score_source
    assert '"adt_csv_gz"' in score_source
    assert "_read_adt_csv" in score_source


def test_score_prevalidates_every_private_subject_before_first_adt_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    held = []
    for batch, count in (("B162", 3), ("B208", 2), ("B210", 4)):
        for _ in range(count):
            index = len(held)
            held.append(
                {"subject_id": f"H{index}", "gsm": f"G{index}", "batch": batch}
            )
    private = {}
    frozen_samples = []
    for sample in held:
        subject_id = sample["subject_id"]
        barcodes = [f"{subject_id}-cell-{index:04d}" for index in range(512)]
        states = np.zeros((512, 24), dtype=np.uint8)
        rows, columns = subject._held_margins(states)
        private[subject_id] = {"barcodes": barcodes, "states": states}
        frozen_samples.append(
            {
                **sample,
                "selected_cell_axis_sha256": subject._axis_sha256(barcodes),
                "rna_state_sha256": subject._array_sha256(states),
                "row_margins": rows.tolist(),
                "column_margins": columns.tolist(),
            }
        )
    frozen_samples[-1]["rna_state_sha256"] = "f" * 64
    mask = np.ones((24, 24), dtype=np.uint8)
    mask_sha256 = subject._array_sha256(mask)
    prediction = {
        "samples": frozen_samples,
        "comparison_mask_sha256": mask_sha256,
    }
    source_result = {
        "selection": {
            "comparison_mask": {
                "mask": mask.tolist(),
                "mask_sha256": mask_sha256,
            }
        }
    }
    monkeypatch.setattr(
        subject,
        "_base_bindings",
        lambda *args: ({}, [], held, {}),
    )
    monkeypatch.setattr(
        subject,
        "_read_json",
        lambda path: (
            prediction if path == subject.DEFAULT_PREDICTION else source_result
        ),
    )
    monkeypatch.setattr(subject, "_stage_bindings", lambda *args: {})
    monkeypatch.setattr(subject, "_read_private_rna", lambda path, samples: private)
    monkeypatch.setattr(
        subject, "_one_shot", lambda phase, attempt, output, bindings, body, audit: body()
    )
    fetch_calls = []

    def fetch(*args: object) -> object:
        fetch_calls.append(args)
        raise AssertionError("held ADT fetch occurred before global private validation")

    monkeypatch.setattr(subject, "_fetch_sample", fetch)

    with pytest.raises(PermissionError, match="state or barcode axis changed"):
        subject.run_score()
    assert fetch_calls == []


def test_held_evaluation_mask_bytes_and_hash_are_reproducible() -> None:
    comparison_mask = np.ones((24, 24), dtype=bool)
    comparison_mask[0, 0] = False
    truth = np.tile(np.asarray([[10, 5], [5, 10]], dtype=np.int64), (24, 24, 1, 1))
    adt_marker_support = np.ones(24, dtype=bool)
    adt_marker_support[3] = False

    evaluation_mask = comparison_mask & subject._subject_support(
        truth, adt_marker_support
    )
    serialized = evaluation_mask.astype(np.uint8).tolist()
    restored = np.asarray(serialized, dtype=np.uint8)

    np.testing.assert_array_equal(restored.astype(bool), evaluation_mask)
    assert subject._array_sha256(restored) == subject._array_sha256(
        evaluation_mask.astype(np.uint8)
    )
    assert not restored[:, 3].any()
    score_source = inspect.getsource(subject.run_score)
    assert '"held_evaluation_masks"' in score_source
    assert '"held_evaluation_mask_sha256"' in score_source


def test_public_tag_chain_includes_every_predecessor() -> None:
    assert len(subject._public_predecessor_chain("source")) == 0
    assert len(subject._public_predecessor_chain("prediction")) == 2
    assert len(subject._public_predecessor_chain("score-authorization")) == 4
    assert len(subject._public_predecessor_chain("score")) == 6


def test_public_freeze_chain_checks_exact_objects_commits_paths_and_ancestry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = json.loads(subject.DEFAULT_PROTOCOL.read_text())
    chain = protocol["public_freeze_chain"]
    calls = []

    def require_tag(
        tag: str,
        paths: list[str],
        *,
        expected_tag_object: str | None = None,
        expected_commit: str | None = None,
    ) -> str:
        calls.append((tag, paths, expected_tag_object, expected_commit))
        return expected_commit or "f" * 40

    ancestry = []
    monkeypatch.setattr(subject, "_require_public_tag", require_tag)
    monkeypatch.setattr(
        subject,
        "_require_ancestor",
        lambda ancestor, descendant: ancestry.append((ancestor, descendant)),
    )

    protocol_commit = subject._require_public_freeze_chain()

    for index, node_name in enumerate(("candidate", "amendment", "axis_preflight")):
        node = chain[node_name]
        assert calls[index] == (
            node["tag"],
            node["required_paths"],
            node["annotated_tag_object"],
            node["peeled_commit"],
        )
    assert calls[3][0] == chain["protocol"]["tag"]
    assert calls[3][2:] == (None, None)
    assert protocol_commit == "f" * 40
    assert ancestry == [
        (chain["candidate"]["peeled_commit"], chain["amendment"]["peeled_commit"]),
        (
            chain["amendment"]["peeled_commit"],
            chain["axis_preflight"]["peeled_commit"],
        ),
        (chain["axis_preflight"]["peeled_commit"], "f" * 40),
    ]


def test_remote_completion_tag_blocks_replay_before_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "attempt.json"
    output = tmp_path / "output.json"
    bindings = {"runner_sha256": "a" * 64}
    monkeypatch.setattr(subject, "DEFAULT_SOURCE_ATTEMPT", attempt)
    monkeypatch.setattr(subject, "DEFAULT_SOURCE", output)
    subject._claim(attempt, "source", bindings)
    monkeypatch.setattr(
        subject,
        "_require_remote_completion_absent",
        lambda phase: (_ for _ in ()).throw(
            PermissionError("public source completion tag already exists")
        ),
    )
    monkeypatch.setattr(subject, "_require_public_attempt", lambda phase, path: "c")
    body_called = False

    def body() -> dict[str, object]:
        nonlocal body_called
        body_called = True
        return {"status": "SHOULD_NOT_RUN"}

    result = subject._one_shot("source", attempt, output, bindings, body)

    assert body_called is False
    assert result["status"] == "TERMINAL_SOURCE_REFUSAL"
    assert "completion tag already exists" in result["reason"]


def test_one_shot_sanitizes_nan_and_unserializable_refusal_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subject, "_require_public_attempt", lambda phase, path: "c")
    monkeypatch.setattr(subject, "_require_remote_completion_absent", lambda phase: None)
    bindings = {"runner_sha256": "a" * 64}

    for index, body in enumerate(
        (
            lambda: {"status": "INVALID_SUCCESS", "value": float("nan")},
            lambda: (_ for _ in ()).throw(
                subject.SelectionRefusal("invalid details", {"bad": object()})
            ),
        )
    ):
        attempt = tmp_path / f"attempt-{index}.json"
        output = tmp_path / f"output-{index}.json"
        monkeypatch.setattr(subject, "DEFAULT_SOURCE_ATTEMPT", attempt)
        monkeypatch.setattr(subject, "DEFAULT_SOURCE", output)
        subject._claim(attempt, "source", bindings)

        result = subject._one_shot("source", attempt, output, bindings, body)
        published = json.loads(output.read_text())

        assert result == published
        assert result["status"] == "TERMINAL_SOURCE_REFUSAL"
        assert result["reason_code"] == "UnpublishablePayload"
        assert "incremental_nonnumeric_access_audit" in result
