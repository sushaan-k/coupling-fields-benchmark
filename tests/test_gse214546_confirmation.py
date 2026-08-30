from __future__ import annotations

from __future__ import annotations

import gzip
import hashlib
import inspect
import math
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy import sparse

from experiments import confirm_gse214546_teaseq as subject


def _strings(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=h5py.string_dtype("utf-8"))


def _write_csc(group: h5py.Group, values: np.ndarray) -> None:
    matrix = sparse.csc_matrix(values)
    group.create_dataset("data", data=matrix.data.astype(np.int32))
    group.create_dataset("indices", data=matrix.indices.astype(np.int32))
    group.create_dataset("indptr", data=matrix.indptr.astype(np.int64))
    group.create_dataset("shape", data=np.asarray(matrix.shape, dtype=np.int64))


def _synthetic_files(tmp_path: Path) -> tuple[Path, Path, dict, list[dict]]:
    markers = subject._contract()["markers"]
    cells = 600
    barcodes = [f"cell-{index:04d}" for index in range(cells)]
    gex_names = [marker["rna"] for marker in markers] + [f"FILLER{index}" for index in range(7)]
    adt_names = [marker["protein"] for marker in markers] + [
        "IgG1_K_Isotype_Control",
        "CD45RO",
    ]
    gex = np.zeros((len(gex_names), cells), dtype=np.int32)
    adt = np.zeros((len(adt_names), cells), dtype=np.int32)
    for marker in range(len(markers)):
        gex[marker] = ((np.arange(cells) + marker) % (marker % 7 + 3) == 0).astype(int)
        adt[marker] = (np.arange(cells) * (marker % 5 + 1) + marker) % 11
    h5_path = tmp_path / "synthetic.h5"
    with h5py.File(h5_path, "w") as handle:
        matrix = handle.create_group("matrix")
        matrix.create_dataset("barcodes", data=_strings(barcodes))
        features = matrix.create_group("features")
        features.create_dataset("name", data=_strings(gex_names))
        features.create_dataset(
            "feature_type", data=_strings(["Gene Expression"] * len(gex_names))
        )
        _write_csc(matrix, gex)
        proteins = handle.create_group("ADT")
        proteins.create_dataset("barcodes", data=_strings(barcodes))
        protein_features = proteins.create_group("features")
        protein_features.create_dataset("id", data=_strings(adt_names))
        _write_csc(proteins, adt)
    metadata_path = tmp_path / "metadata.csv.gz"
    with gzip.open(metadata_path, "wt") as stream:
        stream.write("barcodes,singlet,unused\n")
        for barcode in barcodes:
            stream.write(f"{barcode},TRUE,x\n")
    sample = {"gsm": "SYNTHETIC", "donor": "D", "age_group": "adult"}
    return h5_path, metadata_path, sample, markers


def test_frozen_contract_uses_amended_split_and_marker_panel() -> None:
    contract = subject._contract()
    assert [sample["gsm"] for sample in contract["source"]][-1] == "GSM6611377"
    assert "GSM6611376" in [sample["gsm"] for sample in contract["held"]]
    assert len(contract["markers"]) == 53
    assert (
        contract["normalization_correction"]["corrected_context_estimator_sha256"]
        == subject._sha256(subject.COUPLING_MODULE)
    )
    assert (
        contract["crash_semantics_clarification"][
            "prior_sparse_access_clarification_sha256"
        ]
        == subject.SPARSE_ACCESS_CLARIFICATION_SHA256
    )
    assert "CRASH_UNEVALUABLE" in contract["crash_semantics_clarification"][
        "clarification"
    ]["external_process_loss"]
    assert {sample["age_group"] for sample in contract["source"]} == {
        "adult",
        "pediatric",
    }


def test_implementation_bindings_cover_all_imported_numerical_dependencies() -> None:
    bindings = subject._implementation_bindings()
    expected = {
        "context_conditional_coupling": subject.COUPLING_MODULE,
        "context_conditional_coupling_test": subject.CONTEXT_TEST,
        "common_effect_conditional": subject.COMMON_EFFECT_MODULE,
        "heterogeneity_adaptive_coupling": subject.HETEROGENEITY_MODULE,
        "common_effect_conditional_test": subject.COMMON_EFFECT_TEST,
        "heterogeneity_adaptive_coupling_test": subject.HETEROGENEITY_TEST,
    }
    for key, path in expected.items():
        assert bindings[key] == subject._sha256(path)
    implementation_freeze = inspect.getsource(subject._verify_implementation_freeze)
    for path_name in (
        "COUPLING_MODULE",
        "CONTEXT_TEST",
        "COMMON_EFFECT_MODULE",
        "HETEROGENEITY_MODULE",
        "COMMON_EFFECT_TEST",
        "HETEROGENEITY_TEST",
        "CRASH_SEMANTICS_CLARIFICATION",
    ):
        assert path_name in implementation_freeze


def test_synthetic_h5_matches_bound_csc_schema_and_exact_cell_hash(tmp_path: Path) -> None:
    h5_path, metadata_path, sample, markers = _synthetic_files(tmp_path)
    eligible, metadata = subject._eligible_metadata(metadata_path, sample)
    reduced = subject._read_h5(h5_path, eligible, sample, markers, "source")

    assert metadata["literal_true_singlets"] == 600
    assert reduced["rna_counts"].shape == (512, 53)
    assert reduced["adt_counts"].shape == (512, 53)
    assert len(reduced["selected_barcodes"]) == 512
    expected = sorted(
        eligible,
        key=lambda barcode: (
            hashlib.sha256(
                f"{subject.CELL_SALT}|SYNTHETIC|{barcode}".encode()
            ).hexdigest(),
            barcode,
        ),
    )[:512]
    assert reduced["selected_barcodes"] == expected
    assert reduced["selected_cell_axis_sha256"] == subject._axis_sha256(expected)
    assert reduced["gex_access"]["full_sparse_data_read"] is False
    assert reduced["adt_access"]["full_sparse_data_read"] is False


def test_prediction_h5_firewall_never_requests_adt_sparse_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h5_path, metadata_path, sample, markers = _synthetic_files(tmp_path)
    eligible, _ = subject._eligible_metadata(metadata_path, sample)
    original = h5py.Dataset.__getitem__

    def guarded(dataset: h5py.Dataset, key):
        if dataset.name in {"/ADT/data", "/ADT/indices", "/ADT/indptr"}:
            raise AssertionError("prediction touched an ADT sparse dataset")
        return original(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)
    reduced = subject._read_h5(h5_path, eligible, sample, markers[:24], "predict")
    assert "adt_counts" not in reduced
    assert all(not path.startswith("ADT/") for path in reduced["datasets_read"])
    assert set(reduced["datasets_read"]) == subject.PREDICT_H5_ALLOWLIST


def test_adt_rank_is_exact_deterministic_and_uses_hash_ties() -> None:
    barcodes = [f"cell-{index}" for index in range(512)]
    counts = np.zeros((512, 2), dtype=np.int64)
    first = subject._adt_states(counts, barcodes, "GSM", ["A", "B"])
    second = subject._adt_states(counts, barcodes, "GSM", ["A", "B"])
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first.sum(axis=0), [256, 256])
    assert not np.array_equal(first[:, 0], first[:, 1])


def test_destroyed_link_is_a_deterministic_profile_shift_with_fixed_margins() -> None:
    barcodes = [f"cell-{index}" for index in range(512)]
    states = np.column_stack(
        ((np.arange(512) % 2), (np.arange(512) % 3 == 0))
    ).astype(np.uint8)
    first = subject._destroyed_states(states, barcodes, "GSM")
    second = subject._destroyed_states(states, barcodes, "GSM")
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first.sum(axis=0), states.sum(axis=0))
    assert not np.array_equal(first, states)


def test_conditional_reconstruction_matches_direct_finite_support_mean() -> None:
    rows = np.asarray([7, 13])
    columns = np.asarray([9, 11])
    theta = 0.73
    observed = subject._expected_conditional_table(theta, rows, columns)
    support = np.arange(0, 8)
    weights = np.asarray(
        [
            math.comb(9, int(value))
            * math.comb(11, 7 - int(value))
            * np.exp(theta * value)
            for value in support
        ]
    )
    expected_x00 = float(support @ weights / weights.sum())
    assert observed[0, 0] == pytest.approx(expected_x00, abs=1e-12)
    np.testing.assert_allclose(observed.sum(axis=1), rows)
    np.testing.assert_allclose(observed.sum(axis=0), columns)


def test_fixed_interaction_poisson_reconstructs_odds_without_pseudocount() -> None:
    rows = np.asarray([17.0, 23.0])
    columns = np.asarray([19.0, 21.0])
    theta = -1.2
    table = subject._fixed_interaction_table(theta, rows, columns)
    assert subject._table_log_odds(table) == pytest.approx(theta, abs=1e-10)
    np.testing.assert_allclose(table.sum(axis=1), rows)
    np.testing.assert_allclose(table.sum(axis=0), columns)


def test_age_stratified_poisson_selects_and_uses_its_own_transport() -> None:
    source_code = inspect.getsource(subject._source_models)
    assert "age_poisson_configuration, selected_age_poisson_losses = _select_lowest" in source_code
    source = {
        "selected_marker_count": 1,
        "models": {
            "primary": {"coefficient": [[[0.0]], [[0.0]]], "transport": 1.0},
            "destroyed_link": {
                "coefficient": [[[0.0]], [[0.0]]],
                "transport": 1.0,
            },
            "pooled_fixed_interaction_poisson": {
                "log_odds": [[2.0]],
                "transport": 0.75,
            },
            "age_stratified_fixed_interaction_poisson": {
                "transport": 1.25,
                "adult": {"log_odds": [[0.4]]},
                "pediatric": {"log_odds": [[0.8]]},
            },
            "common_effect_exact_conditional": {"log_odds": [[0.0]]},
            "signed_root_deviance": {"coordinate_per_sqrt_n": [[0.0]]},
            "independence": {"kind": "recipient_fixed_margin_independence"},
        },
    }
    predictions, _ = subject._method_predictions(
        source, "adult", np.asarray([256])
    )
    assert subject._table_log_odds(
        predictions["pooled_fixed_interaction_poisson"][0, 0]
    ) == pytest.approx(1.5)
    assert subject._table_log_odds(
        predictions["age_stratified_fixed_interaction_poisson"][0, 0]
    ) == pytest.approx(0.5)


def test_complete_source_cv_grids_replay_selected_models_and_promotion() -> None:
    adult = np.asarray([[200, 56], [56, 200]], dtype=np.int64)
    pediatric = np.asarray([[56, 200], [200, 56]], dtype=np.int64)
    independence = np.asarray([[128, 128], [128, 128]], dtype=np.int64)
    tables = np.asarray([adult] * 4 + [pediatric] * 4)[:, None, None]
    destroyed = np.asarray([independence] * 8)[:, None, None]

    fitted = subject._source_models(
        tables, destroyed, ["adult"] * 4 + ["pediatric"] * 4
    )

    subject._validate_source_cross_validation(fitted)
    assert fitted["promotion"]["passes"] is True
    assert (
        fitted["models"]["age_stratified_fixed_interaction_poisson"]["transport"]
        == fitted["source_cross_validation"][
            "selected_age_stratified_poisson_transport"
        ]
    )
    assert len(
        fitted["source_cross_validation"]["complete_loss_grids"]["primary"]
    ) == len(subject.DEVIATION_GRID) * len(subject.AGE_RIDGE_GRID) * len(
        subject.PRIMARY_TRANSPORT_GRID
    )


def test_common_effect_control_is_delta_free_exact_conditional_cmle() -> None:
    tables = np.asarray(
        [
            [[[[8, 4], [3, 9]]]],
            [[[[7, 5], [4, 8]]]],
            [[[[9, 3], [5, 7]]]],
        ]
    )
    coordinate, certificate = subject._fit_common_effect(tables)
    assert coordinate.shape == (1, 1)
    assert np.isfinite(coordinate).all()
    assert (
        certificate["estimator"]
        == "unregularized_delta_free_exact_conditional_cmle"
    )
    assert certificate["passes"]


def test_ranked_greedy_axis_skips_poisson_unavailable_marker() -> None:
    generator = np.random.default_rng(214546)
    donors, cells, markers_count = 8, 512, 25
    rna = np.empty((donors, cells, markers_count), dtype=np.uint8)
    adt = np.empty_like(rna)
    for donor in range(donors):
        for marker in range(markers_count):
            rna[donor, :, marker] = generator.permutation(
                np.r_[np.ones(256, dtype=np.uint8), np.zeros(256, dtype=np.uint8)]
            )
            adt[donor, :, marker] = generator.permutation(
                np.r_[np.ones(256, dtype=np.uint8), np.zeros(256, dtype=np.uint8)]
            )
        adt[donor, :, 0] = rna[donor, :, 0]
    tables = np.asarray(
        [subject._joint_tables(first, second) for first, second in zip(rna, adt)]
    )
    markers = [
        {"protein": f"P{index:02d}", "rna": f"R{index:02d}"}
        for index in range(markers_count)
    ]
    selected, decisions = subject._select_markers(
        rna, tables, markers, ["adult"] * 4 + ["pediatric"] * 4
    )
    assert 0 not in selected
    assert len(selected) == 24
    assert decisions[0]["protein"] == "P00"
    assert decisions[0]["accepted"] is False


def test_claim_stages_are_separate_from_consuming_runs() -> None:
    assert "_write_json_x(SOURCE_ATTEMPT" in inspect.getsource(subject.claim_source)
    assert "_validate_source_attempt" in inspect.getsource(subject.run_source)
    assert "_write_json_x(PREDICTION_ATTEMPT" in inspect.getsource(
        subject.claim_prediction
    )
    assert "_validate_prediction_attempt" in inspect.getsource(subject.run_prediction)
    assert "_write_json_x(SCORE_ATTEMPT" in inspect.getsource(subject.claim_score)
    assert "_validate_score_attempt" in inspect.getsource(subject.run_score)
    for claim in (subject.claim_source, subject.claim_prediction, subject.claim_score):
        assert "_fetch(" not in inspect.getsource(claim)


def test_score_authorization_and_attempt_precede_held_truth_reduction() -> None:
    source = inspect.getsource(subject.run_score)
    assert source.index("_validate_score_attempt") < source.index(
        "_consume_claim_token"
    )
    assert source.index("_consume_claim_token") < source.index("_reduce_sample(")
    assert "expected_h5_sha256=frozen[\"held_h5_sha256\"]" in source


def test_source_claim_creates_private_capability_without_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt_path = tmp_path / "repo-artifacts" / "attempt.json"
    token_path = tmp_path / "private" / "source.token"
    monkeypatch.setattr(subject, "SOURCE_ATTEMPT", attempt_path)
    monkeypatch.setattr(subject, "_downstream_paths", lambda: ())
    monkeypatch.setattr(
        subject,
        "_contract",
        lambda: {
            "source": [{"gsm": "S", "h5_bytes": 11}],
            "held": [{"gsm": "H"}],
        },
    )
    monkeypatch.setattr(
        subject,
        "_verify_implementation_freeze",
        lambda: {"implementation_commit": "a" * 40},
    )
    monkeypatch.setattr(
        subject, "_fetch", lambda *args, **kwargs: pytest.fail("claim performed GET")
    )

    attempt = subject.claim_source(claim_token=token_path)

    token = token_path.read_bytes()
    assert len(token) == 32
    assert token_path.stat().st_mode & 0o777 == 0o600
    assert attempt["claim_token_sha256"] == hashlib.sha256(token).hexdigest()
    assert token not in attempt_path.read_bytes()


@pytest.mark.parametrize("provided", [None, b"wrong-token-preimage-is-32-bytes!"])
def test_source_run_refuses_missing_or_wrong_capability_before_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provided: bytes | None,
) -> None:
    expected = hashlib.sha256(b"correct-token-preimage-is-32-byte").hexdigest()
    attempt_path = tmp_path / "attempt.json"
    attempt_path.write_text("{}\n")
    monkeypatch.setattr(subject, "SOURCE_ATTEMPT", attempt_path)
    monkeypatch.setattr(
        subject,
        "_validate_source_attempt",
        lambda: ({"claim_token_sha256": expected, "bindings": {}}, "b" * 40),
    )
    monkeypatch.setattr(subject, "_downstream_paths", lambda: ())
    monkeypatch.setattr(subject, "_contract", lambda: {})
    monkeypatch.setattr(
        subject, "_fetch", lambda *args, **kwargs: pytest.fail("run performed GET")
    )
    token_path = None
    if provided is not None:
        token_path = tmp_path / "wrong.token"
        token_path.write_bytes(provided[:32].ljust(32, b"!"))
        token_path.chmod(0o600)
    with pytest.raises(PermissionError):
        subject.run_source(claim_token=token_path, scratch=tmp_path / "scratch")


@pytest.mark.parametrize("failure_path", ["/ADT/barcodes", "/ADT/data"])
def test_h5_audit_journals_catchable_mid_read_failure_truthfully(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_path: str,
) -> None:
    h5_path, metadata_path, sample, markers = _synthetic_files(tmp_path)
    eligible, _ = subject._eligible_metadata(metadata_path, sample)
    audit: dict = {}
    original = h5py.Dataset.__getitem__

    def failing(dataset: h5py.Dataset, key):
        if dataset.name == failure_path:
            raise OSError("injected read failure")
        return original(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", failing)
    with pytest.raises(OSError, match="injected"):
        subject._read_h5(
            h5_path, eligible, sample, markers[:24], "score", audit_record=audit
        )
    assert audit["h5_open_started"] is True
    assert audit["h5_open_completed"] is True
    assert audit["h5_reduction_completed"] is False
    assert audit["gex_access"]["completed"] is True
    failed = [
        event
        for event in audit["dataset_access_events"]
        if event["dataset"] == failure_path.lstrip("/")
        and event["completed"] is False
    ]
    assert len(failed) == 1
    assert failed[0]["error_type"] == "OSError"
    if failure_path == "/ADT/data":
        assert audit["adt_access"]["completed"] is False


def test_successful_prediction_audit_validates_sparse_firewall(
    tmp_path: Path,
) -> None:
    h5_path, metadata_path, sample, markers = _synthetic_files(tmp_path)
    sample.update(
        {
            "metadata_filename": metadata_path.name,
            "metadata_url": "https://example.test/metadata.csv.gz",
            "metadata_bytes": metadata_path.stat().st_size,
            "h5_filename": h5_path.name,
            "h5_url": "https://example.test/synthetic.h5",
            "h5_bytes": h5_path.stat().st_size,
        }
    )
    eligible, metadata_decode = subject._eligible_metadata(metadata_path, sample)
    metadata_record = {
        "filename": sample["metadata_filename"],
        "url": sample["metadata_url"],
        "expected_bytes": sample["metadata_bytes"],
        "observed_bytes": sample["metadata_bytes"],
        "completed": True,
        "deleted": True,
        "sha256": subject._sha256(metadata_path),
        "datasets_read": [],
        "decode_started": True,
        "decode_completed": True,
        "decode": metadata_decode,
    }
    h5_record = {
        "filename": sample["h5_filename"],
        "url": sample["h5_url"],
        "expected_bytes": sample["h5_bytes"],
        "observed_bytes": sample["h5_bytes"],
        "completed": True,
        "deleted": True,
        "sha256": subject._sha256(h5_path),
    }
    subject._read_h5(
        h5_path,
        eligible,
        sample,
        markers[:24],
        "predict",
        audit_record=h5_record,
    )
    subject._validate_download_audit(
        [metadata_record, h5_record], [sample], "predict", markers[:24]
    )
    data_event = next(
        event
        for event in h5_record["dataset_access_events"]
        if event["dataset"] == "matrix/data"
    )
    data_event["selection"]["kind"] = "all"
    with pytest.raises(PermissionError, match="unselected sparse values"):
        subject._validate_download_audit(
            [metadata_record, h5_record], [sample], "predict", markers[:24]
        )


def test_prediction_audit_allows_unread_data_only_for_all_zero_selected_rows(
    tmp_path: Path,
) -> None:
    h5_path, metadata_path, sample, markers = _synthetic_files(tmp_path)
    with h5py.File(h5_path, "r+") as handle:
        matrix = handle["matrix"]
        shape = tuple(matrix["shape"][:].tolist())
        for name in ("data", "indices", "indptr", "shape"):
            del matrix[name]
        _write_csc(matrix, np.zeros(shape, dtype=np.int32))
    sample.update(
        {
            "metadata_filename": metadata_path.name,
            "metadata_url": "https://example.test/metadata.csv.gz",
            "metadata_bytes": metadata_path.stat().st_size,
            "h5_filename": h5_path.name,
            "h5_url": "https://example.test/synthetic.h5",
            "h5_bytes": h5_path.stat().st_size,
        }
    )
    eligible, metadata_decode = subject._eligible_metadata(metadata_path, sample)
    metadata_record = {
        "filename": sample["metadata_filename"],
        "url": sample["metadata_url"],
        "expected_bytes": sample["metadata_bytes"],
        "observed_bytes": sample["metadata_bytes"],
        "completed": True,
        "deleted": True,
        "sha256": subject._sha256(metadata_path),
        "datasets_read": [],
        "decode_started": True,
        "decode_completed": True,
        "decode": metadata_decode,
    }
    h5_record = {
        "filename": sample["h5_filename"],
        "url": sample["h5_url"],
        "expected_bytes": sample["h5_bytes"],
        "observed_bytes": sample["h5_bytes"],
        "completed": True,
        "deleted": True,
        "sha256": subject._sha256(h5_path),
    }
    reduced = subject._read_h5(
        h5_path,
        eligible,
        sample,
        markers[:24],
        "predict",
        audit_record=h5_record,
    )

    assert not reduced["rna_counts"].any()
    assert "matrix/data" in h5_record["datasets_opened"]
    assert "matrix/data" not in h5_record["datasets_read"]
    assert h5_record["gex_access"]["selected_data_values_decoded"] == 0
    subject._validate_download_audit(
        [metadata_record, h5_record], [sample], "predict", markers[:24]
    )


def test_score_audit_counts_completed_all_zero_adt_without_data_read(
    tmp_path: Path,
) -> None:
    h5_path, metadata_path, sample, markers = _synthetic_files(tmp_path)
    with h5py.File(h5_path, "r+") as handle:
        for group_name in ("matrix", "ADT"):
            group = handle[group_name]
            shape = tuple(group["shape"][:].tolist())
            for name in ("data", "indices", "indptr", "shape"):
                del group[name]
            _write_csc(group, np.zeros(shape, dtype=np.int32))
    sample.update(
        {
            "metadata_filename": metadata_path.name,
            "metadata_url": "https://example.test/metadata.csv.gz",
            "metadata_bytes": metadata_path.stat().st_size,
            "h5_filename": h5_path.name,
            "h5_url": "https://example.test/synthetic.h5",
            "h5_bytes": h5_path.stat().st_size,
        }
    )
    eligible, metadata_decode = subject._eligible_metadata(metadata_path, sample)
    metadata_record = {
        "filename": sample["metadata_filename"],
        "url": sample["metadata_url"],
        "expected_bytes": sample["metadata_bytes"],
        "observed_bytes": sample["metadata_bytes"],
        "completed": True,
        "deleted": True,
        "sha256": subject._sha256(metadata_path),
        "datasets_read": [],
        "decode_started": True,
        "decode_completed": True,
        "decode": metadata_decode,
    }
    h5_record = {
        "filename": sample["h5_filename"],
        "url": sample["h5_url"],
        "expected_bytes": sample["h5_bytes"],
        "observed_bytes": sample["h5_bytes"],
        "completed": True,
        "deleted": True,
        "sha256": subject._sha256(h5_path),
    }
    reduced = subject._read_h5(
        h5_path,
        eligible,
        sample,
        markers[:24],
        "score",
        audit_record=h5_record,
    )

    assert not reduced["rna_counts"].any()
    assert not reduced["adt_counts"].any()
    assert "matrix/data" not in h5_record["datasets_read"]
    assert "ADT/data" not in h5_record["datasets_read"]
    subject._validate_download_audit(
        [metadata_record, h5_record], [sample], "score", markers[:24]
    )
    repeated = [metadata_record, h5_record] * 8
    assert subject._score_access_counts(repeated) == {
        "held_adt_value_datasets_read": 0,
        "held_adt_modalities_reduced": 8,
    }


def test_prediction_allowlist_has_no_adt_path() -> None:
    assert all(not path.startswith("ADT/") for path in subject.PREDICT_H5_ALLOWLIST)
    assert {"ADT/data", "ADT/indices", "ADT/indptr"}.issubset(
        subject.LINKED_H5_ALLOWLIST
    )
