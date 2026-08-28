from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments import evaluate_scmmib_bmmc_exact_development as candidate
from mapreg.heterogeneity_adaptive_coupling import (
    fit_structured_conditional_log_odds,
)


def _csr_fixture(path: Path) -> np.ndarray:
    dense = np.asarray(
        [
            [1, 0, 2, 0],
            [999, 999, 999, 999],
            [0, 3, 0, 4],
            [888, 888, 888, 888],
            [5, 0, 6, 0],
            [777, 777, 777, 777],
        ],
        dtype=np.int64,
    )
    indices = []
    values = []
    indptr = [0]
    for row in dense:
        nonzero = np.flatnonzero(row)
        indices.extend(nonzero.tolist())
        values.extend(row[nonzero].tolist())
        indptr.append(len(indices))
    with h5py.File(path, "w") as handle:
        layers = handle.create_group("layers")
        matrix = layers.create_group("counts")
        matrix.attrs["encoding-type"] = "csr_matrix"
        matrix.attrs["shape"] = dense.shape
        matrix.create_dataset("data", data=np.asarray(values, dtype=np.int64))
        matrix.create_dataset("indices", data=np.asarray(indices, dtype=np.int32))
        matrix.create_dataset("indptr", data=np.asarray(indptr, dtype=np.int64))
        handle["X"] = h5py.ExternalLink("poisoned-X.h5", "/X")
        handle["raw"] = h5py.ExternalLink("poisoned-raw.h5", "/raw")
    return dense


def test_selective_csr_reader_never_spans_held_rows_or_opens_x_raw(
    tmp_path: Path,
) -> None:
    path = tmp_path / "combined.h5ad"
    dense = _csr_fixture(path)
    selected = np.asarray([0, 2, 4])
    forbidden = np.asarray([1, 3, 5])

    observed, audit = candidate._read_csr_marker_counts(
        path,
        matrix_path="layers/counts",
        selected_rows=selected,
        forbidden_rows=forbidden,
        selected_columns=np.asarray([0, 2]),
    )

    np.testing.assert_array_equal(observed, dense[selected][:, [0, 2]])
    assert audit["rows_read"] == 3
    assert audit["contiguous_data_slices"] == 3
    assert audit["held_rows_read"] == 0
    assert audit["full_indptr_read"] is False
    assert audit["permitted_indptr_slices"] == 3
    assert audit["x_opened"] is False
    assert audit["raw_x_opened"] is False


def test_intersecting_held_row_refuses_before_hdf5_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "unopened.h5ad"
    path.write_text("not HDF5")

    def forbidden_open(*args, **kwargs):
        raise AssertionError("HDF5 opened before held-row rejection")

    monkeypatch.setattr(candidate.h5py, "File", forbidden_open)
    with pytest.raises(PermissionError, match="include a held donor"):
        candidate._read_csr_marker_counts(
            path,
            matrix_path="layers/counts",
            selected_rows=np.asarray([0, 2]),
            forbidden_rows=np.asarray([2, 3]),
            selected_columns=np.asarray([0]),
        )


def test_csr_reader_requests_only_permitted_run_indptr_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dense = np.asarray([[1, 0], [9, 9], [0, 2], [8, 8]])
    indices = np.asarray([0, 0, 1, 1, 1, 1, 0, 1], dtype=np.int32)
    data = np.asarray([1, 9, 9, 2, 8, 8, 8, 8])
    indptr_values = np.asarray([0, 1, 3, 4, 8], dtype=np.int64)

    class Dataset:
        def __init__(self, values):
            self.values = values
            self.shape = values.shape
            self.requests = []

        def __getitem__(self, key):
            self.requests.append(key)
            return self.values[key]

    indptr = Dataset(indptr_values)
    matrix = {
        "indptr": indptr,
        "indices": Dataset(indices),
        "data": Dataset(data),
    }

    class Matrix(dict):
        attrs = {"encoding-type": "csr_matrix", "shape": dense.shape}

    class Handle(dict):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    handle = Handle({"layers/counts": Matrix(matrix)})
    monkeypatch.setattr(candidate.h5py, "File", lambda *args, **kwargs: handle)

    observed, _ = candidate._read_csr_marker_counts(
        Path("unused.h5ad"),
        matrix_path="layers/counts",
        selected_rows=np.asarray([0, 2]),
        forbidden_rows=np.asarray([1, 3]),
        selected_columns=np.asarray([0, 1]),
    )

    np.testing.assert_array_equal(observed, dense[[0, 2]])
    assert indptr.requests == [slice(0, 2, None), slice(2, 4, None)]
    assert all(request != slice(None, None, None) for request in indptr.requests)


def test_deterministic_adt_midrank_is_balanced_within_donor_batch() -> None:
    cells = []
    for donor, batch, size in (("A", "x", 7), ("B", "y", 8)):
        for index in range(size):
            cells.append(
                {
                    "DonorID": donor,
                    "batch": batch,
                    "barcode": f"{donor}-{index}",
                }
            )
    counts = np.ones((15, 10), dtype=int)
    first = candidate._rank_binary_adt(counts, cells)
    second = candidate._rank_binary_adt(counts, cells)
    np.testing.assert_array_equal(first, second)
    for donor, size in (("A", 7), ("B", 8)):
        mask = np.asarray([row["DonorID"] == donor for row in cells])
        assert np.all((first[mask] == 0).sum(axis=0) == size // 2)


def test_sqrt_n_normalization_matches_scaled_classical_tables() -> None:
    small = np.asarray([[[[[12, 8], [8, 12]]]]])
    large = 4 * small
    small_coordinate = candidate._residual_coordinate(small, "deviance", False)
    large_coordinate = candidate._residual_coordinate(large, "deviance", False)
    np.testing.assert_allclose(small_coordinate, large_coordinate, atol=1e-12)


def test_zero_initializer_converges_to_same_common_effect_optimum() -> None:
    tables = np.asarray(
        [
            [[[[12, 8], [8, 12]], [[9, 11], [11, 9]]]],
            [[[[11, 9], [9, 11]], [[8, 12], [12, 8]]]],
            [[[[14, 6], [6, 14]], [[10, 10], [10, 10]]]],
        ]
    )
    first = np.ones((1, 1))
    second = np.ones((2, 1))
    repaired = candidate._common_fit(tables, first, second, ridge=0.1, graph=0.0)
    reference = fit_structured_conditional_log_odds(
        tables,
        first,
        second,
        initial_log_odds=np.full((1, 2), 0.2),
        ridge_penalty=0.1,
        graph_penalty=0.0,
        minimum_informative_donors=2,
        tolerance=1e-9,
    )
    assert repaired.converged and reference.converged
    np.testing.assert_allclose(repaired.log_odds, reference.log_odds, atol=1e-8)
    assert repaired.objective == pytest.approx(reference.objective, abs=1e-10)


def test_gate_serializes_batch_differences_and_declared_threshold() -> None:
    primary = np.asarray([0.80, 0.82, 0.78, 0.81])
    comparator = np.asarray([1.00, 1.01, 0.99, 1.02])
    result = candidate._comparison(
        ["s1d1", "s2d1", "s3d1", "s4d1"],
        primary,
        comparator,
        required_favorable=4,
    )
    assert result["passes"]
    assert result["relative_reduction"] >= 0.05
    assert result["favorable_batches"] == 4
    assert set(result["batch_differences_primary_minus_comparator"]) == {
        "s1d1",
        "s2d1",
        "s3d1",
        "s4d1",
    }


def test_actual_combined_axis_matches_locked_schema_without_count_access() -> None:
    source, paths = candidate.lock._validated_source()
    axis = candidate.lock._axis(paths["complete_cite_h5ad"], source["combined_assay"])
    assert axis["shape"] == (90261, 14087)
    assert axis["feature_type_categories"] == ["ADT", "GEX"]
    assert len(axis["marker_indices"]["rna"]) == 10
    assert len(axis["marker_indices"]["adt"]) == 10
    assert not set(axis["marker_indices"]["rna"]) & set(axis["marker_indices"]["adt"])
