import json
from itertools import product
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy import sparse
from scipy.stats import norm

from experiments.reduce_gse299043_mln import (
    ADT_FEATURE_IDS,
    CELL_BUDGET,
    DEVELOPMENT_DONORS,
    HASH_SOLO_PRIORS,
    MARKERS,
    MLN_TAGS,
    RNA_FEATURE_IDS,
    HeldAccessPermit,
    _cell_selection_hash,
    _hashsolo_classifications,
    _hashsolo_log_likelihoods,
    _normalize_hto_id,
    finalize_donor,
    reduce_library,
)


def _strings(values):
    return np.asarray(values, dtype=h5py.string_dtype("utf-8"))


def _write_frame(handle, name, index, columns, *, categorical=()):
    group = handle.create_group(name)
    group.attrs["encoding-type"] = "dataframe"
    group.attrs["encoding-version"] = "0.1.0"
    group.attrs["_index"] = "_index"
    group.create_dataset("_index", data=_strings(index))
    for key, values in columns.items():
        if key in categorical:
            categories = list(dict.fromkeys(values))
            item = group.create_group(key)
            item.attrs["encoding-type"] = "categorical"
            item.create_dataset("categories", data=_strings(categories))
            item.create_dataset(
                "codes",
                data=np.asarray([categories.index(value) for value in values]),
            )
        else:
            group.create_dataset(key, data=_strings(values))


def _write_matrix(handle, matrix, layout):
    value = (
        sparse.csr_matrix(matrix)
        if layout == "csr_matrix"
        else sparse.csc_matrix(matrix)
    )
    group = handle.create_group("X")
    group.attrs["encoding-type"] = layout
    group.attrs["encoding-version"] = "0.1.0"
    group.attrs["shape"] = value.shape
    group.create_dataset("data", data=value.data.astype(np.float32))
    group.create_dataset("indices", data=value.indices)
    group.create_dataset("indptr", data=value.indptr)


def _filename(donor="621B", library="CZI-IA10244331", version="v1"):
    run = "001" if donor != "591C" else "003"
    return f"GSE299043_{donor}_{run}.{library}.{version}.h5ad"


def _make_h5ad(
    root: Path,
    *,
    donor="621B",
    library="CZI-IA10244331",
    version="v1",
    tags=None,
    cells=620,
    layout="csr_matrix",
    adt_permutation=None,
    duplicate_rna=False,
    wrong_adt_name=False,
    poison_value=None,
):
    default_tags = tags is None
    tags = [MLN_TAGS[donor][0], f"{donor}-SPL-fixture"] if default_tags else list(tags)
    path = root / _filename(donor, library, version)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(MARKERS) + [f"{marker}-1" for marker in MARKERS] + tags
    if wrong_adt_name:
        names[len(MARKERS)] = "not-CD4"
    gene_ids = list(RNA_FEATURE_IDS) + list(ADT_FEATURE_IDS) + tags
    feature_types = ["Gene Expression"] * len(MARKERS) + ["Antibody Capture"] * (
        len(MARKERS) + len(tags)
    )
    if duplicate_rna:
        names.append("CD4-2")
        gene_ids.append(RNA_FEATURE_IDS[0])
        feature_types.append("Gene Expression")

    cell_index = np.arange(cells)
    rna = np.column_stack(
        [(cell_index + marker_index) % 4 for marker_index in range(len(MARKERS))]
    )
    adt = np.column_stack(
        [
            1 + (cell_index * (marker_index + 3)) % 23
            for marker_index in range(len(MARKERS))
        ]
    )
    if adt_permutation is not None:
        adt = adt[np.asarray(adt_permutation)]
    hto = np.full((cells, len(tags)), 25, dtype=int)
    if default_tags:
        hto[:, 0] = 30 + cell_index % 11
        hto[:, 1] = 1 + cell_index % 3
    elif len(tags) > 1:
        hto.fill(1)
        for cell in range(cells):
            winner = cell % len(tags)
            hto[cell, winner] = 30 + cell % 11
            hto[cell] += (cell + np.arange(len(tags))) % 3
    matrix = np.column_stack((rna, adt, hto)).astype(np.float32)
    if duplicate_rna:
        matrix = np.column_stack((matrix, rna[:, 0]))
    if poison_value is not None:
        matrix[0, 0] = poison_value
    barcodes = [f"cell-{index:04d}" for index in range(cells)]
    with h5py.File(path, "w") as handle:
        handle.attrs["encoding-type"] = "anndata"
        handle.attrs["encoding-version"] = "0.1.0"
        _write_frame(handle, "obs", barcodes, {})
        _write_frame(
            handle,
            "var",
            names,
            {"gene_ids": gene_ids, "feature_types": feature_types},
            categorical=("feature_types",),
        )
        _write_matrix(handle, matrix, layout)
    return path, barcodes


@pytest.mark.parametrize("layout", ["csr_matrix", "csc_matrix"])
def test_reducer_pools_after_source_deletion_and_has_exact_margins(tmp_path, layout):
    source, barcodes = _make_h5ad(tmp_path / "source", layout=layout)
    piece = tmp_path / "piece.json"
    payload = reduce_library(source, "621B", piece)
    assert payload["hashsolo_priors"] == list(HASH_SOLO_PRIORS)
    assert payload["hashsolo_noise_barcodes"] == 1
    assert payload["target_mln_singlets"] == len(barcodes)
    expected = sorted(
        barcodes,
        key=lambda barcode: _cell_selection_hash("621B", source.name, barcode),
    )[:CELL_BUDGET]
    assert [record["barcode"] for record in payload["candidates"]] == expected

    source.unlink()
    output = tmp_path / "donor.json"
    reduced = finalize_donor([piece], "621B", output)
    assert reduced["cells"] == CELL_BUDGET
    assert reduced["candidate_mln_singlets_retained_across_pieces"] == CELL_BUDGET
    assert reduced["total_mln_singlets_before_per_library_budget"] == len(barcodes)
    tables = np.asarray(reduced["tables"]).reshape(81, 2, 2)
    assert np.all(tables.sum(axis=(1, 2)) == CELL_BUDGET)
    assert np.all(tables.sum(axis=1) == CELL_BUDGET // 2)
    assert reduced["access_audit"]["source_h5ad_required_during_pooling"] is False
    with pytest.raises(FileExistsError):
        finalize_donor([piece], "621B", output)


def test_hashsolo_keeps_strong_singlets_and_excludes_other_tissue():
    generator = np.random.default_rng(17)
    tags = ["591C-MLN-1", "591C-MLN-2", "591C-SPL-4"]
    counts = generator.poisson(2, size=(900, 3))
    truth = np.arange(len(counts)) % len(tags)
    counts[np.arange(len(counts)), truth] += 35 + np.arange(len(counts)) % 13
    observed = _hashsolo_classifications(counts, tags)
    expected = np.asarray(tags)[truth]
    assert np.mean(observed == expected) > 0.98
    retained = np.isin(observed, MLN_TAGS["591C"])
    assert np.all(observed[retained] != "591C-SPL-4")


def _scanpy_172_reference(counts, tags):
    data = np.log1p(np.asarray(counts, dtype=float))
    order = np.argsort(data, axis=1)
    sorted_data = np.sort(data, axis=1)
    global_signal = sorted_data[:, -1]
    global_noise = sorted_data[:, :-1].ravel()

    def update(values, prior_mean, prior_std):
        prior_precision = 1 / prior_std**2
        count = len(values)
        precision = 1 / np.var(values) if count > 1 else prior_precision
        posterior_precision = prior_precision + count * precision
        posterior_mean = (
            (np.mean(values) * count * precision + prior_mean * prior_precision)
            / posterior_precision
            if count
            else prior_mean
        )
        return posterior_mean, np.sqrt((count + 1) / posterior_precision)

    signal = {}
    noise = {}
    for tag in range(len(tags)):
        values = data[:, tag]
        signal[tag] = update(
            values[np.where(order[:, -1] == tag)],
            np.mean(global_signal),
            np.std(global_signal),
        )
        noise[tag] = update(
            values[np.where(order[:, :-1] == tag)[0]],
            np.mean(global_noise),
            np.std(global_noise),
        )

    likelihoods = np.zeros((len(data), 3))
    for noise_tag, signal_tag in product(range(len(tags)), repeat=2):
        subset = (order[:, -1] == signal_tag) & (order[:, -2] == noise_tag)
        if not np.any(subset):
            continue
        values = data[subset]
        signal_signal = np.log(
            norm.pdf(values[:, signal_tag], *signal[signal_tag]) + 1e-15
        )
        noise_signal = np.log(
            norm.pdf(values[:, noise_tag], *signal[noise_tag]) + 1e-15
        )
        noise_noise = np.log(norm.pdf(values[:, noise_tag], *noise[noise_tag]) + 1e-15)
        signal_noise = np.log(
            norm.pdf(values[:, signal_tag], *noise[noise_tag]) + 1e-15
        )
        likelihoods[subset] = np.column_stack(
            (
                noise_noise + signal_noise,
                noise_noise + signal_signal,
                noise_signal + signal_signal,
            )
        )
    weighted = np.exp(likelihoods) * np.asarray(HASH_SOLO_PRIORS)
    hypotheses = np.argmax(weighted / weighted.sum(axis=1, keepdims=True), axis=1)
    classifications = np.full(len(data), "Negative", dtype=object)
    classifications[hypotheses == 2] = "Doublet"
    singlets = hypotheses == 1
    classifications[singlets] = np.asarray(tags, dtype=object)[order[singlets, -1]]
    return likelihoods, classifications.astype(str)


def test_hashsolo_matches_scanpy_172_reference_likelihoods_and_calls():
    generator = np.random.default_rng(241)
    tags = ["A", "B", "C", "D"]
    counts = generator.poisson(3, size=(800, len(tags)))
    winners = np.arange(len(counts)) % len(tags)
    counts[np.arange(len(counts)), winners] += 20 + np.arange(len(counts)) % 19
    expected_likelihoods, expected_calls = _scanpy_172_reference(counts, tags)
    observed_likelihoods, _ = _hashsolo_log_likelihoods(counts, tags)
    np.testing.assert_allclose(observed_likelihoods, expected_likelihoods, rtol=1e-12)
    np.testing.assert_array_equal(
        _hashsolo_classifications(counts, tags), expected_calls
    )


def test_adt_pairing_cannot_change_cell_selection(tmp_path):
    first, _ = _make_h5ad(tmp_path / "first")
    permutation = np.random.default_rng(9).permutation(620)
    second, _ = _make_h5ad(tmp_path / "second", adt_permutation=permutation)
    first_piece = tmp_path / "first.json"
    second_piece = tmp_path / "second.json"
    first_result = reduce_library(first, "621B", first_piece)
    second_result = reduce_library(second, "621B", second_piece)
    first_cells = [
        (record["barcode"], record["cell_selection_sha256"])
        for record in first_result["candidates"]
    ]
    second_cells = [
        (record["barcode"], record["cell_selection_sha256"])
        for record in second_result["candidates"]
    ]
    assert first_cells == second_cells
    assert any(
        left["adt_counts"] != right["adt_counts"]
        for left, right in zip(first_result["candidates"], second_result["candidates"])
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"duplicate_rna": True}, "unique RNA/ADT ID pair"),
        ({"poison_value": -1}, "negative"),
        ({"poison_value": 1.5}, "nonintegral"),
        ({"poison_value": np.nan}, "nonfinite"),
    ],
)
def test_schema_and_count_poisons_fail_closed(tmp_path, kwargs, message):
    source, _ = _make_h5ad(tmp_path / "source", **kwargs)
    with pytest.raises(ValueError, match=message):
        reduce_library(source, "621B", tmp_path / "piece.json")
    assert not (tmp_path / "piece.json").exists()


def test_feature_identity_uses_ids_not_display_names(tmp_path):
    source, _ = _make_h5ad(tmp_path / "source", wrong_adt_name=True)
    payload = reduce_library(source, "621B", tmp_path / "piece.json")
    assert payload["status"] == "TARGET_MLN_LIBRARY_REDUCED"


def test_missing_mln_tag_is_terminal_schema_refusal(tmp_path):
    source, _ = _make_h5ad(tmp_path / "source", tags=["621B-SPL-86", "621B-BLO-88"])
    with pytest.raises(ValueError, match="lacks an accepted MLN HTO"):
        reduce_library(source, "621B", tmp_path / "piece.json")


def test_one_hto_refuses_outside_the_exact_single_tissue_exception(tmp_path):
    source, _ = _make_h5ad(tmp_path / "source", tags=["621B-MLN-87"])
    output = tmp_path / "piece.json"
    with pytest.raises(ValueError, match="frozen single-tissue exception"):
        reduce_library(source, "621B", output)
    assert not output.exists()


def test_exact_694b_single_tissue_one_hto_exception_assigns_all_cells(tmp_path):
    source, barcodes = _make_h5ad(
        tmp_path / "source",
        donor="694B",
        library="CZI-IA11512689",
        version="v2",
        tags=["694B-MLN-206"],
    )
    payload = reduce_library(source, "694B", tmp_path / "piece.json")
    assert source.name == "GSE299043_694B_001.CZI-IA11512689.v2.h5ad"
    assert payload["target_mln_singlets"] == len(barcodes)
    assert payload["hashsolo_noise_barcodes"] == 0
    assert {record["assigned_mln_tag"] for record in payload["candidates"]} == {
        "694B-MLN-206"
    }


@pytest.mark.parametrize(
    ("version", "tag"),
    [("v1", "694B-MLN-206"), ("v2", "694B-SPL-206")],
)
def test_694b_one_hto_exception_refuses_wrong_filename_or_tag(tmp_path, version, tag):
    source, _ = _make_h5ad(
        tmp_path / "source",
        donor="694B",
        library="CZI-IA11512689",
        version=version,
        tags=[tag],
    )
    with pytest.raises(ValueError, match="frozen single-tissue exception"):
        reduce_library(source, "694B", tmp_path / "piece.json")


def test_759b_patch_is_exactly_limited_to_four_libraries():
    assert MLN_TAGS["759B"] == ("759B-MLN-263",)
    for library in (
        "CZI-IA12953908",
        "CZI-IA12953909",
        "CZI-IA12953910",
        "CZI-IA12953911",
    ):
        assert _normalize_hto_id("759B", library, "759B-MLN-1") == "759B-MLN-263"
    assert _normalize_hto_id("759B", "CZI-IA12953912", "759B-MLN-1") == "759B-MLN-1"
    assert _normalize_hto_id("694B", "CZI-IA12953908", "694B-MLN-1") == "694B-MLN-1"
    assert _normalize_hto_id("621B", "CZI-IA10244331", "621B-TLN-87") == "621B-TLN-87"


def test_held_path_is_not_touched_before_authorization(tmp_path):
    held_path = tmp_path / _filename("D512", "CZINY-0161")
    with pytest.raises(PermissionError, match="commit-bound"):
        reduce_library(held_path, "D512", tmp_path / "piece.json")
    permit = HeldAccessPermit(
        prediction_sha256="a" * 64,
        public_commit="b" * 40,
        authorization_sha256="c" * 64,
        terminal_attempt_sha256="d" * 64,
    )
    with pytest.raises(ValueError, match="regular non-symlink"):
        reduce_library(
            held_path,
            "D512",
            tmp_path / "piece.json",
            phase="held_score_authorized",
            permit=permit,
        )


def test_tampered_piece_is_rejected_without_output(tmp_path):
    source, _ = _make_h5ad(tmp_path / "source")
    piece = tmp_path / "piece.json"
    reduce_library(source, "621B", piece)
    value = json.loads(piece.read_text())
    value["candidates"][0]["cell_selection_sha256"] = "0" * 64
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(value))
    output = tmp_path / "donor.json"
    with pytest.raises(ValueError, match="selection hash differs"):
        finalize_donor([tampered], "621B", output)
    assert not output.exists()


def test_frozen_split_and_marker_contract_are_complete():
    assert len(DEVELOPMENT_DONORS) == 10
    assert len(MARKERS) == len(RNA_FEATURE_IDS) == len(ADT_FEATURE_IDS) == 9
    assert set(DEVELOPMENT_DONORS).issubset(MLN_TAGS)
