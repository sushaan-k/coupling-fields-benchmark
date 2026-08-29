from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from scipy import sparse

from experiments import reduce_gse181897_source as subject


def test_claim_rejects_an_alternate_attempt_path(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="attempt path is not canonical"):
        subject.claim_source_campaign(
            tmp_path / "authorization.json",
            tmp_path / "alternate-attempt.json",
            tmp_path / "preflight.json",
            tmp_path / "source.npz",
            tmp_path / "manifest.json",
            tmp_path / "model.json",
            tmp_path / "model-terminal.json",
            tmp_path / "reduction-terminal.json",
        )


def test_public_freeze_chain_rejects_an_unexpected_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze = {
        "tag_object": "1" * 40,
        "peeled_commit": "2" * 40,
        "remote_tag_and_commit_match": True,
    }
    authorization = {
        "candidate_freeze": {"tag": "wrong-candidate", **freeze},
        "implementation_freeze": {"tag": subject.IMPLEMENTATION_TAG, **freeze},
        "axis_freeze": {"tag": subject.AXIS_PREFLIGHT_TAG, **freeze},
    }
    monkeypatch.setattr(
        subject,
        "_verified_tag",
        lambda tag: {"tag": tag, **freeze},
    )
    with pytest.raises(PermissionError, match="wrong candidate_freeze tag"):
        subject._validate_public_freeze_chain(authorization, Path("unused.json"))


def test_public_freeze_chain_rejects_a_tagged_blob_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze = {
        "tag_object": "1" * 40,
        "peeled_commit": "2" * 40,
        "remote_tag_and_commit_match": True,
    }
    candidate_sha256 = subject._sha256(subject.ROOT / subject.CANDIDATE_PATH)
    authorization = {
        "candidate_freeze": {
            "tag": subject.CANDIDATE_TAG,
            **freeze,
            "candidate_path": subject.CANDIDATE_PATH,
            "candidate_sha256": candidate_sha256,
        },
        "implementation_freeze": {
            "tag": subject.IMPLEMENTATION_TAG,
            **freeze,
            "files_sha256": {"implementation.py": "a" * 64},
        },
        "axis_freeze": {
            "tag": subject.AXIS_PREFLIGHT_TAG,
            **freeze,
            "preflight_path": "axis.json",
            "preflight_sha256": "b" * 64,
        },
    }
    monkeypatch.setattr(
        subject, "CAMPAIGN_IMPLEMENTATION_FILES", ("implementation.py",)
    )
    monkeypatch.setattr(
        subject,
        "_verified_tag",
        lambda tag: {"tag": tag, **freeze},
    )
    monkeypatch.setattr(subject, "_require_ancestor", lambda *_args: None)

    def published(_tag: str, relative: str) -> str:
        if relative == subject.CANDIDATE_PATH:
            return candidate_sha256
        return "f" * 64

    monkeypatch.setattr(subject, "_published_file_sha256", published)
    with pytest.raises(PermissionError, match="implementation tag"):
        subject._validate_public_freeze_chain(authorization, Path("unused.json"))


def test_reduction_interruption_is_terminal_and_blocks_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "attempt.json"
    attempt.write_text("{}\n")
    terminal = tmp_path / "terminal.json"
    args = SimpleNamespace(
        reduction_terminal=terminal,
        attempt=attempt,
        preflight=tmp_path / "preflight.json",
        output=tmp_path / "source.npz",
        manifest=tmp_path / "manifest.json",
        model_output=tmp_path / "model.json",
        model_terminal=tmp_path / "model-terminal.json",
        cache=tmp_path / "cache",
        keep_archive=False,
    )
    monkeypatch.setattr(
        subject,
        "validate_source_campaign_attempt",
        lambda *_args: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    result = subject.run_reduction_one_shot(args)
    assert result["status"] == "TERMINAL_SOURCE_REDUCTION_REFUSAL"
    assert result["reason_code"] == "KeyboardInterrupt"
    with pytest.raises(FileExistsError):
        subject.run_reduction_one_shot(args)


def _strings(values: list[str] | tuple[str, ...] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=h5py.string_dtype("utf-8"))


def _write_legacy_frame(
    handle: h5py.File,
    name: str,
    index: list[str],
    columns: dict[str, list[str]],
    *,
    category_axes: dict[str, list[str]] | None = None,
    plain: tuple[str, ...] = (),
) -> h5py.Group:
    group = handle.create_group(name)
    group.attrs["encoding-type"] = "dataframe"
    group.attrs["encoding-version"] = "0.1.0"
    group.attrs["_index"] = "_index"
    group.attrs["column-order"] = np.asarray(list(columns), dtype=object)
    group.create_dataset("_index", data=_strings(index))
    categories = group.create_group("__categories")
    for key, values in columns.items():
        if key in plain:
            group.create_dataset(key, data=_strings(values))
            continue
        axis = (category_axes or {}).get(key, list(dict.fromkeys(values)))
        category = categories.create_dataset(key, data=_strings(axis))
        item = group.create_dataset(
            key, data=np.asarray([axis.index(value) for value in values], dtype=np.int8)
        )
        item.attrs["categories"] = category.ref
    return group


def _write_csr(group: h5py.Group, matrix: sparse.csr_matrix) -> None:
    matrix = matrix.astype(np.float32)
    group.attrs["encoding-type"] = "csr_matrix"
    group.attrs["encoding-version"] = "0.1.0"
    group.attrs["shape"] = matrix.shape
    group.create_dataset("data", data=matrix.data.astype(np.float32))
    group.create_dataset("indices", data=matrix.indices.astype(np.int32))
    group.create_dataset("indptr", data=matrix.indptr.astype(np.int32))


def _production_like_metadata() -> tuple[np.ndarray, ...]:
    rows: list[tuple[str, str, str, str, str]] = []
    free_map = {
        donor: str((int(donor) + 7) % 64) for donor in subject.EXPECTED_DONOR_CATEGORIES
    }
    for batch, donors in subject.DEVELOPMENT_DONORS_BY_BATCH.items():
        for donor in donors:
            for cell in range(subject.CELL_BUDGET + 3):
                rows.append(
                    (
                        f"source-{batch}-{donor}-{cell:03d}",
                        str(batch),
                        "C",
                        donor,
                        free_map[donor],
                    )
                )
    for donor, batch in subject.EXCLUDED_DEVELOPMENT_DONORS.items():
        for cell in range(subject.EXCLUDED_CONTROL_DONORS[donor][1]):
            rows.append(
                (
                    f"excluded-{batch}-{donor}-{cell:03d}",
                    str(batch),
                    "C",
                    donor,
                    free_map[donor],
                )
            )
    for allocation in (
        subject.INTERNAL_DONORS_BY_BATCH,
        subject.CONFIRMATION_DONORS_BY_BATCH,
    ):
        for batch, donors in allocation.items():
            for donor in donors:
                rows.append(
                    (f"held-{batch}-{donor}", str(batch), "C", donor, free_map[donor])
                )
    for donor in ("51", "52"):
        batch, unused_count = subject.EXCLUDED_CONTROL_DONORS[donor]
        del unused_count
        rows.append((f"held-{batch}-{donor}", str(batch), "C", donor, free_map[donor]))
    rows.append(("mispool", "0", "0", "34", free_map["34"]))
    return tuple(np.asarray([row[index] for row in rows]) for index in range(5))


def test_frozen_panel_axes_and_selection_hash_are_exact() -> None:
    assert len(subject.PANEL) == 17
    assert len({cognate.rna_gene for cognate in subject.PANEL}) == 17
    assert len({cognate.adt_feature for cognate in subject.PANEL}) == 17
    assert subject.PANEL[0].adt_feature == "CD1c|CD1C"
    assert subject.PANEL[-1].adt_feature == "CD163|CD163"
    assert subject.CELL_SELECTION_SALT == "GSE181897-CONTROL-CELL-BUDGET-v1"
    expected = hashlib.sha256(
        b"GSE181897-CONTROL-CELL-BUDGET-v1|3|12|cell-7"
    ).hexdigest()
    assert subject._selection_hash(3, "12", "cell-7") == expected
    assert subject.EXPECTED_OBS_AXIS_SHA256 == (
        "24560e2df6a268b11509d2ab23ae898ae17cf699080ed604e710fa266db418c5"
    )
    assert subject.EXPECTED_VAR_AXIS_SHA256 == (
        "7c7511ba42740fe6127fac90ebe181f49122ff07014908f14be11fafdc143e1e"
    )


@pytest.mark.parametrize("axis_name", ["obs", "var"])
def test_axis_uniqueness_certificate_rejects_duplicate_values(
    axis_name: str,
) -> None:
    with pytest.raises(ValueError, match=f"{axis_name} index is not unique"):
        subject._require_unique_axis(
            np.asarray([f"{axis_name}-0", f"{axis_name}-0"]), axis_name
        )
    assert (
        subject._require_unique_axis(
            np.asarray([f"{axis_name}-0", f"{axis_name}-1"]), axis_name
        )
        == 2
    )


def test_source_plan_is_deterministic_deposited_order_and_firewalled() -> None:
    barcodes, batches, conditions, exp_ids, free_ids = _production_like_metadata()
    plan = subject._build_source_plan(barcodes, batches, conditions, exp_ids, free_ids)
    assert plan.donor_axis == subject._expected_donor_axis()
    assert plan.selected_rows.shape == (39, 128)
    assert np.all(np.diff(plan.selected_rows, axis=1) > 0)
    assert np.all(conditions[plan.selected_rows] == "C")
    assert np.all(np.isin(batches[plan.selected_rows], [str(i) for i in range(8)]))
    assert not set(plan.donor_axis).intersection({"23", "62"})
    assert "mispool" not in set(plan.selected_barcodes.ravel())

    first_eligible = np.flatnonzero(
        (batches == "0") & (conditions == "C") & (exp_ids == "34")
    )
    expected = sorted(
        sorted(
            first_eligible,
            key=lambda row: (
                subject._selection_hash(0, "34", str(barcodes[row])),
                str(barcodes[row]),
                row,
            ),
        )[:128]
    )
    assert plan.selected_rows[0].tolist() == expected


def test_source_plan_rejects_exp_free_nonbijection() -> None:
    barcodes, batches, conditions, exp_ids, free_ids = _production_like_metadata()
    free_ids = free_ids.copy()
    free_ids[0] = "63"
    with pytest.raises(ValueError, match="bijection"):
        subject._build_source_plan(barcodes, batches, conditions, exp_ids, free_ids)


def test_guarded_csr_reader_never_decodes_unauthorized_poison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = sparse.csr_matrix(
        np.asarray(
            [
                [1, 0, 2, 0, 0],
                [np.nan, 0, -4, 0, 0],
                [0, 3, 4, 0, 0],
                [-8, 0, np.nan, 0, 0],
            ],
            dtype=np.float32,
        )
    )
    monkeypatch.setattr(subject, "EXPECTED_X_SHAPE", values.shape)
    monkeypatch.setattr(subject, "EXPECTED_X_DATA_LENGTH", values.nnz)
    monkeypatch.setattr(subject, "CELL_BUDGET", 1)
    path = tmp_path / "guard.h5"
    with h5py.File(path, "w") as handle:
        _write_csr(handle.create_group("X"), values)

    plan = subject.SourcePlan(
        donor_axis=("a", "b"),
        free_id_axis=("x", "y"),
        batch_axis=(0, 0),
        selected_rows=np.asarray([[0], [2]], dtype=np.int64),
        selected_barcodes=np.asarray([["c0"], ["c2"]]),
        authorized_rows=np.asarray([True, False, True, False]),
        donor_audit=({}, {}),
    )
    with h5py.File(path, "r") as handle:
        observed, audit = subject._read_authorized_csr_columns(
            handle["X"], plan, (0, 2)
        )
    np.testing.assert_array_equal(observed, [[1, 2], [0, 4]])
    assert audit["held_batch_rows_decoded"] == 0
    assert audit["non_control_rows_decoded"] == 0
    assert audit["unselected_authorized_rows_decoded"] == 0
    assert audit["csr_index_entries_scanned"] == values[[0, 2]].nnz
    assert audit["requested_stored_data_entries_decoded"] == 3
    assert audit["unrequested_stored_data_entries_decoded"] == 0
    assert audit["out_of_panel_index_positions_scanned"] == 1
    assert audit["out_of_panel_indices_used_only_for_membership_filtering"] is True
    assert audit["out_of_panel_featurewise_statistics_retained"] == 0
    assert audit["out_of_panel_feature_signal_entering_model_outputs"] == 0

    unauthorized = subject.SourcePlan(
        donor_axis=("a", "b"),
        free_id_axis=("x", "y"),
        batch_axis=(0, 1),
        selected_rows=np.asarray([[0], [1]], dtype=np.int64),
        selected_barcodes=np.asarray([["c0"], ["c1"]]),
        authorized_rows=plan.authorized_rows,
        donor_audit=({}, {}),
    )
    with (
        h5py.File(path, "r") as handle,
        pytest.raises(PermissionError, match="unauthorized"),
    ):
        subject._read_authorized_csr_columns(handle["X"], unauthorized, (0, 2))


def test_axis_inspection_does_not_index_poisoned_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subject, "SOURCE_BATCHES", (0,))
    monkeypatch.setattr(subject, "HELD_BATCHES", (1,))
    monkeypatch.setattr(subject, "DEVELOPMENT_DONORS_BY_BATCH", {0: ("1",)})
    monkeypatch.setattr(subject, "INTERNAL_DONORS_BY_BATCH", {1: ("2",)})
    monkeypatch.setattr(subject, "CONFIRMATION_DONORS_BY_BATCH", {})
    monkeypatch.setattr(subject, "EXCLUDED_DEVELOPMENT_DONORS", {})
    monkeypatch.setattr(subject, "EXCLUDED_CONTROL_DONORS", {})
    monkeypatch.setattr(subject, "EXPECTED_BATCH_CATEGORIES", ("0", "1"))
    monkeypatch.setattr(subject, "EXPECTED_CONDITION_CATEGORIES", ("0", "C"))
    monkeypatch.setattr(subject, "EXPECTED_DONOR_CATEGORIES", ("1", "2"))
    monkeypatch.setattr(subject, "EXPECTED_CONDITION_ZERO_CELLS", 1)
    monkeypatch.setattr(subject, "EXPECTED_CONTROL_CELLS", 4)
    monkeypatch.setattr(subject, "EXPECTED_RNA_FEATURE_COUNT", 17)
    monkeypatch.setattr(subject, "EXPECTED_PROTEIN_FEATURE_COUNT", 17)
    monkeypatch.setattr(subject, "CELL_BUDGET", 2)
    monkeypatch.setattr(subject, "MINIMUM_MARKER_POSITIVES", 1)
    monkeypatch.setattr(subject, "MAXIMUM_MARKER_POSITIVES", 1)

    obs_index = ["source-0", "source-1", "mispool", "held-0", "held-1"]
    var_index = [cognate.rna_gene for cognate in subject.PANEL] + [
        cognate.adt_feature for cognate in subject.PANEL
    ]
    feature_ids = [cognate.rna_feature_id for cognate in subject.PANEL] + [
        cognate.adt_feature for cognate in subject.PANEL
    ]
    path = tmp_path / "fixture.h5ad"
    with h5py.File(path, "w") as handle:
        _write_legacy_frame(
            handle,
            "obs",
            obs_index,
            {
                "batch": ["0", "0", "0", "1", "1"],
                "cond": ["C", "C", "0", "C", "C"],
                "exp_id": ["1", "1", "1", "2", "2"],
                "free_id": ["2", "2", "2", "1", "1"],
            },
            category_axes={
                "batch": ["0", "1"],
                "cond": ["0", "C"],
                "exp_id": ["1", "2"],
                "free_id": ["1", "2"],
            },
        )
        _write_legacy_frame(
            handle,
            "var",
            var_index,
            {
                "gene_ids": feature_ids,
                "feature_types": ["Gene Expression"] * 34,
                "genome": ["GRCh38"] * 17 + ["BD99AbSeq"] * 17,
                "batch": ["0"] * 17 + ["1"] * 17,
            },
            category_axes={
                "feature_types": ["Gene Expression"],
                "genome": ["BD99AbSeq", "GRCh38"],
                "batch": ["0", "1"],
            },
            plain=("gene_ids",),
        )
        matrix = handle.create_group("X")
        matrix.attrs["encoding-type"] = "csr_matrix"
        matrix.attrs["encoding-version"] = "0.1.0"
        matrix.attrs["shape"] = (5, 34)
        matrix.create_dataset("data", data=np.asarray([np.nan], dtype=np.float32))
        matrix.create_dataset("indices", data=np.asarray([-1], dtype=np.int32))
        matrix.create_dataset("indptr", data=np.asarray([7] * 6, dtype=np.int32))
        handle.create_group("obsm")
        handle.create_group("obsp")
        handle.create_group("uns")
    monkeypatch.setattr(subject, "EXPECTED_X_SHAPE", (5, 34))
    monkeypatch.setattr(subject, "EXPECTED_X_DATA_LENGTH", 1)
    monkeypatch.setattr(subject, "SOURCE_H5AD_BYTES", path.stat().st_size)
    monkeypatch.setattr(subject, "SOURCE_H5AD_SHA256", subject._sha256(path))
    monkeypatch.setattr(
        subject, "EXPECTED_OBS_AXIS_SHA256", subject._axis_sha256(obs_index)
    )
    monkeypatch.setattr(
        subject, "EXPECTED_VAR_AXIS_SHA256", subject._axis_sha256(var_index)
    )

    inspection = subject.inspect_axes(path)
    assert inspection.payload["numeric_access"]["decoded_X_entries"] == 0
    assert inspection.payload["numeric_access"]["matrix_datasets_indexed"] == []
    assert inspection.plan.selected_rows.tolist() == [[0, 1]]


def test_strict_fixed_margin_interior_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "SOURCE_BATCHES", (0, 1))
    donors = ("1", "2", "3", "4")
    batches = (0, 0, 1, 1)
    rna = np.zeros((4, 128, 17), dtype=np.int32)
    adt = np.zeros_like(rna)
    rna[0, :64] = 1
    adt[0, :32] = 1
    rna[1, :32] = 1
    adt[1, :64] = 1
    rna[2] = rna[0]
    adt[2] = adt[0]
    rna[3] = rna[1]
    adt[3] = adt[1]
    _, mask, audit = subject._availability_diagnostics(rna, adt, donors, batches)
    final = audit["final_source_mask"]
    assert final["coordinates_passing_pooled_four_cell_positivity"] == 289
    assert final["coordinates_passing_strict_fixed_margin_interior"] == 0
    assert np.count_nonzero(mask) == 0

    rna[3] = 0
    adt[3] = 0
    rna[3, :32] = 1
    adt[3, 16:80] = 1
    _, mask, audit = subject._availability_diagnostics(rna, adt, donors, batches)
    assert (
        audit["final_source_mask"]["coordinates_passing_strict_fixed_margin_interior"]
        == 289
    )
    assert np.count_nonzero(mask) == 289


def test_acquire_validates_existing_archive_even_when_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / subject.SOURCE_ARCHIVE_NAME
    h5ad = tmp_path / subject.SOURCE_H5AD_NAME
    archive.write_bytes(b"archive")
    h5ad.write_bytes(b"h5ad")
    monkeypatch.setattr(subject, "SOURCE_ARCHIVE_BYTES", len(b"archive"))
    monkeypatch.setattr(subject, "SOURCE_H5AD_BYTES", len(b"h5ad"))
    monkeypatch.setattr(
        subject, "SOURCE_ARCHIVE_SHA256", hashlib.sha256(b"archive").hexdigest()
    )
    monkeypatch.setattr(
        subject, "SOURCE_H5AD_SHA256", hashlib.sha256(b"h5ad").hexdigest()
    )
    observed, audit = subject.acquire_h5ad(tmp_path, keep_archive=True)
    assert observed == h5ad
    assert audit == {
        "archive_present_at_start": True,
        "archive_verified_in_this_run": True,
        "h5ad_verified_in_this_run": True,
        "archive_removed_after_verification": False,
    }
    archive.write_bytes(b"drifted")
    monkeypatch.setattr(subject, "SOURCE_ARCHIVE_BYTES", len(b"drifted"))
    with pytest.raises(ValueError, match="SHA-256"):
        subject.acquire_h5ad(tmp_path, keep_archive=True)


def test_preflight_refuses_unverified_compressed_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subject,
        "inspect_axes",
        lambda unused: (_ for _ in ()).throw(AssertionError("must not inspect")),
    )
    with pytest.raises(PermissionError, match="archive verification"):
        subject.write_preflight(
            Path("not-opened.h5ad"),
            Path("not-written.json"),
            {"archive_verified_in_this_run": False},
        )
