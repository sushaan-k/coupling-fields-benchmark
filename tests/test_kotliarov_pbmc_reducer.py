import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from experiments.reduce_kotliarov_pbmc import (
    ADT_FEATURE_KEY,
    ADT_MATRIX_KEY,
    ASSET_MANIFEST_KEY,
    DEVELOPMENT,
    HELD,
    LINEAGES,
    METADATA_KEY,
    PREDICTION_OUTPUTS,
    RNA_FEATURE_KEY,
    RNA_MATRIX_KEY,
    SCORE_OUTPUTS,
    read_npy_gzip,
    reduce,
)


def _hashes(path: Path) -> tuple[int, str, str]:
    value = path.read_bytes()
    return len(value), hashlib.md5(value).hexdigest(), hashlib.sha256(value).hexdigest()


def _bytes(values) -> np.ndarray:
    encoded = [str(value).encode() for value in values]
    width = max(1, max(map(len, encoded), default=1))
    return np.asarray(encoded, dtype=f"S{width}")


def _write_frame(path: Path, row_names, columns: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("data_frame")
        group.attrs["row-count"] = len(row_names)
        group.create_dataset("row_names", data=_bytes(row_names))
        group.create_dataset("column_names", data=_bytes(columns))
        data = group.create_group("data")
        for index, values in enumerate(columns.values()):
            array = np.asarray(values)
            if array.dtype.kind in {"U", "O"}:
                array = _bytes(array)
            data.create_dataset(str(index), data=array)


def _write_sparse(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = sparse.csc_matrix(matrix)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("compressed_sparse_matrix")
        group.attrs["layout"] = "CSC"
        group.attrs["type"] = "integer"
        group.create_dataset("shape", data=np.asarray(value.shape, dtype=np.uint64))
        group.create_dataset("data", data=value.data.astype(np.int32))
        group.create_dataset("indices", data=value.indices.astype(np.uint64))
        group.create_dataset("indptr", data=value.indptr.astype(np.uint64))


def _write_dense(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("dense_array")
        group.attrs["type"] = "integer"
        group.attrs["transposed"] = 1
        group.create_dataset("data", data=matrix.T.astype(np.int32))


def _lineage_definitions() -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    return {
        "B": (("MS4A1", "CD79A", "CD74", "CD37", "CD19"), ("NKG7", "LST1", "S100A8")),
        "CD4 T": (("CD3D", "CD3E", "TRAC", "IL7R", "LTB"), ("NKG7", "GNLY", "LST1")),
        "CD8 T": (("CD3D", "CD3E", "TRAC", "CD8A", "CD8B"), ("IL7R", "LST1", "S100A8")),
        "NK": (("NKG7", "GNLY", "KLRD1", "PRF1", "FCER1G"), ("CD3D", "CD79A", "IL7R")),
        "Monocyte": (("LST1", "FCER1G", "CTSS", "S100A8", "S100A9", "TYROBP"), ("CD3D", "CD79A", "GNLY")),
    }


def _make_fixture(root: Path, *, permute_held_adt: bool = False) -> dict[str, Path]:
    metadata_root = root / "metadata"
    rna_path = root / "matrix.h5"
    adt_path = root / "array.h5"
    source_path = root / "source_manifest_v1.json"
    aliases_path = root / "aliases.tsv"
    lineage_path = root / "lineages.tsv"

    definitions = _lineage_definitions()
    marker_genes = list(
        dict.fromkeys(
            gene
            for positive, negative in definitions.values()
            for gene in (*positive, *negative)
        )
    )
    alias_genes = [f"GENE{index:02d}" for index in range(16)]
    rna_names = alias_genes + marker_genes
    adt_targets = [f"ADT{index:02d}" for index in range(83)]
    adt_names = [f"{target}_PROT" for target in adt_targets]

    cells = []
    designed_lineage = []
    local_index = []
    for donor in DEVELOPMENT + HELD:
        batch = "1" if donor in DEVELOPMENT else "2"
        for lineage in LINEAGES:
            for local in range(60):
                cells.append((f"{donor}_{lineage}_{local}", donor, batch))
                designed_lineage.append(lineage)
                local_index.append(local)
    for local in range(10):
        cells.append((f"209_excluded_{local}", "209", "1" if local < 5 else "2"))
        designed_lineage.append("B")
        local_index.append(local)

    n_cells = len(cells)
    n_gene = 1_000 + np.arange(n_cells) % 5
    n_umi = np.full(n_cells, 10_000)
    pct_mt = 0.04 + (np.arange(n_cells) % 5) / 1_000
    n_gene[0] = 100_000
    columns = {
        "nGene": n_gene.astype(np.int32),
        "nUMI": n_umi.astype(float),
        "pctMT": pct_mt.astype(float),
        "batch": _bytes([cell[2] for cell in cells]),
        "sampleid": _bytes([cell[1] for cell in cells]),
        "joint_classification_global": _bytes(["SNG_Singlet"] * n_cells),
        "dmx_hto_match": _bytes(["1"] * n_cells),
        "timepoint": _bytes(["d0"] * n_cells),
    }
    metadata_path = metadata_root / METADATA_KEY
    _write_frame(metadata_path, [cell[0] for cell in cells], columns)

    rna = np.empty((len(rna_names), n_cells), dtype=np.int32)
    for gene_index in range(len(rna_names)):
        rna[gene_index] = 1 + (np.arange(n_cells) + gene_index) % 3
    gene_lookup = {gene: index for index, gene in enumerate(rna_names)}
    for cell_index, lineage in enumerate(designed_lineage):
        for gene in definitions[lineage][0]:
            rna[gene_lookup[gene], cell_index] += 30
        for gene in alias_genes:
            rna[gene_lookup[gene], cell_index] = 2 + 10 * (
                local_index[cell_index] % 3
            )
    _write_sparse(rna_path, rna)

    adt = np.empty((83, n_cells), dtype=np.int32)
    adt.fill(1)
    for marker_index in range(16):
        adt[marker_index] = 2 + 10 * (np.asarray(local_index) % 3)
    if permute_held_adt:
        donors = np.asarray([cell[1] for cell in cells])
        lineages = np.asarray(designed_lineage)
        for donor in HELD:
            for lineage in LINEAGES:
                indices = np.flatnonzero((donors == donor) & (lineages == lineage))
                adt[:, indices] = adt[:, indices[::-1]]
    _write_dense(adt_path, adt)

    rna_features_path = metadata_root / RNA_FEATURE_KEY
    adt_features_path = metadata_root / ADT_FEATURE_KEY
    _write_frame(rna_features_path, rna_names, {})
    _write_frame(
        adt_features_path,
        adt_names,
        {"target": _bytes(adt_targets), "isotype": np.zeros(83, dtype=np.int32)},
    )
    aliases = pd.DataFrame(
        {
            "adt_target": adt_targets[:16],
            "gene_symbol": alias_genes,
            "module": [f"module{index % 4}" for index in range(16)],
        }
    )
    aliases_path.write_text(aliases.to_csv(sep="\t", index=False))
    lineage_rows = []
    for lineage in LINEAGES:
        positive, negative = definitions[lineage]
        lineage_rows.append(
            {
                "lineage": lineage,
                "positive_markers": ",".join(positive),
                "negative_markers": ",".join(negative),
            }
        )
    lineage_path.write_text(pd.DataFrame(lineage_rows).to_csv(sep="\t", index=False))

    files = {
        RNA_MATRIX_KEY: rna_path,
        ADT_MATRIX_KEY: adt_path,
        METADATA_KEY: metadata_path,
        RNA_FEATURE_KEY: rna_features_path,
        ADT_FEATURE_KEY: adt_features_path,
    }
    asset_manifest = {}
    for key, path in files.items():
        size, md5, _ = _hashes(path)
        asset_manifest[key] = {"size": size, "md5sum": md5}
    asset_path = metadata_root / ASSET_MANIFEST_KEY
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    asset_path.write_text(json.dumps(asset_manifest, sort_keys=True))

    metadata_files = []
    for key, path in {
        ASSET_MANIFEST_KEY: asset_path,
        METADATA_KEY: metadata_path,
        RNA_FEATURE_KEY: rna_features_path,
        ADT_FEATURE_KEY: adt_features_path,
    }.items():
        metadata_files.append(
            {
                "path": f"data/development/kotliarov_pbmc_metadata_v1/{key}",
                "sha256": _hashes(path)[2],
            }
        )
    rna_size, rna_md5, _ = _hashes(rna_path)
    adt_size, adt_md5, _ = _hashes(adt_path)
    source = {
        "schema": "kotliarov-pbmc-gypsum-source/1.0",
        "files": [
            {
                "key": "rna_matrix",
                "name": "matrix.h5",
                "bytes": rna_size,
                "md5": rna_md5,
                "sha256": None,
                "url": "https://example.test/matrix.h5",
            },
            {
                "key": "adt_matrix",
                "name": "array.h5",
                "bytes": adt_size,
                "md5": adt_md5,
                "sha256": None,
                "url": "https://example.test/array.h5",
            },
        ],
        "metadata_files": metadata_files,
    }
    source_path.write_text(json.dumps(source, sort_keys=True))
    return {
        "rna": rna_path,
        "adt": adt_path,
        "metadata_root": metadata_root,
        "source": source_path,
        "aliases": aliases_path,
        "lineages": lineage_path,
    }


def _run(paths: dict[str, Path], root: Path) -> tuple[Path, Path]:
    prediction = root / "prediction"
    score = root / "score"
    reduce(
        rna_matrix_path=paths["rna"],
        adt_matrix_path=paths["adt"],
        metadata_path=paths["metadata_root"] / METADATA_KEY,
        rna_features_path=paths["metadata_root"] / RNA_FEATURE_KEY,
        adt_features_path=paths["metadata_root"] / ADT_FEATURE_KEY,
        asset_manifest_path=paths["metadata_root"] / ASSET_MANIFEST_KEY,
        source_manifest_path=paths["source"],
        aliases_path=paths["aliases"],
        lineage_markers_path=paths["lineages"],
        prediction_output=prediction,
        score_output=score,
    )
    return prediction, score


@pytest.fixture(scope="module")
def reduced(tmp_path_factory):
    root = tmp_path_factory.mktemp("kotliarov")
    paths = _make_fixture(root / "source")
    prediction, score = _run(paths, root / "run")
    return root, paths, prediction, score


def test_reducer_excludes_209_and_separates_held_pairing(reduced):
    _, _, prediction, score = reduced
    assert {path.name for path in prediction.iterdir()} == set(PREDICTION_OUTPUTS) | {
        "prediction_manifest.tsv"
    }
    assert {path.name for path in score.iterdir()} == set(SCORE_OUTPUTS) | {
        "score_manifest.tsv"
    }
    cells = pd.read_csv(prediction / "cells.tsv.gz", sep="\t", dtype={"donor": str})
    assert "209" not in set(cells["donor"])
    assert set(cells.loc[cells["split"] == "development", "donor"]) == set(DEVELOPMENT)
    assert set(cells.loc[cells["split"] == "held", "donor"]) == set(HELD)
    assert set(cells["lineage"]) == set(LINEAGES)
    assert len(cells[(cells["donor"] == "200") & (cells["lineage"] == "B")]) == 59
    assert not any("held_adt_state" in path.name for path in prediction.iterdir())
    assert not any("held_cell" in path.name for path in prediction.iterdir())


def test_exact_marker_and_entity_axes_are_explicit(reduced):
    _, _, prediction, _ = reduced
    markers = pd.read_csv(prediction / "markers.tsv", sep="\t")
    entities = pd.read_csv(prediction / "entities.tsv", sep="\t")
    assert len(markers) == 16
    assert list(markers["marker_index"]) == list(range(16))
    assert list(entities.columns) == [
        "entity_id",
        "marker_index",
        "marker_id",
        "gene_symbol",
        "adt_target",
        "module",
        "lineage",
        "eligible",
    ]
    assert len(entities) >= 32
    assert entities["marker_id"].nunique() == 16
    assert entities["eligible"].eq(1).all()
    ordering = [
        (int(row.marker_index), LINEAGES.index(row.lineage))
        for row in entities.itertuples(index=False)
    ]
    assert ordering == sorted(ordering)
    cells = pd.read_csv(prediction / "cells.tsv.gz", sep="\t")
    assert read_npy_gzip(prediction / "rna_values.npy.gz").shape == (16, len(cells))
    assert read_npy_gzip(prediction / "rna_states.npy.gz").shape == (16, len(cells))


def test_held_margins_match_score_only_states(reduced):
    _, _, prediction, score = reduced
    markers = pd.read_csv(prediction / "markers.tsv", sep="\t")
    margins = pd.read_csv(prediction / "held_adt_marginals.tsv", sep="\t", dtype={"donor": str})
    held_cells = pd.read_csv(score / "held_cells.tsv.gz", sep="\t", dtype={"donor": str})
    states = read_npy_gzip(score / "held_adt_states.npy.gz")
    assert states.shape == (len(markers), len(held_cells))
    for row in margins.itertuples(index=False):
        block = (held_cells["donor"] == row.donor) & (
            held_cells["lineage"] == row.lineage
        )
        observed = int(np.sum(states[int(row.marker_index), block.to_numpy()] == row.state))
        assert observed == row.count


def test_prediction_payload_is_invariant_to_held_whole_cell_permutation(reduced, tmp_path):
    _, _, prediction, _ = reduced
    paths = _make_fixture(tmp_path / "source", permute_held_adt=True)
    permuted_prediction, permuted_score = _run(paths, tmp_path / "run")
    statistical_files = set(PREDICTION_OUTPUTS) - {"source_acquisition.json"}
    for relative in statistical_files:
        assert (prediction / relative).read_bytes() == (
            permuted_prediction / relative
        ).read_bytes()
    assert (prediction / "source_acquisition.json").read_bytes() != (
        permuted_prediction / "source_acquisition.json"
    ).read_bytes()
    assert (permuted_score / "held_adt_states.npy.gz").read_bytes() != (
        reduced[3] / "held_adt_states.npy.gz"
    ).read_bytes()


def test_reducer_is_byte_deterministic(reduced, tmp_path):
    _, paths, prediction, score = reduced
    second_prediction, second_score = _run(paths, tmp_path / "second")
    for first, second in ((prediction, second_prediction), (score, second_score)):
        assert {path.name for path in first.iterdir()} == {
            path.name for path in second.iterdir()
        }
        for path in first.iterdir():
            assert path.read_bytes() == (second / path.name).read_bytes()


def test_source_byte_md5_mismatch_refuses_before_reduction(tmp_path):
    paths = _make_fixture(tmp_path / "source")
    with paths["rna"].open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="byte count or MD5 differs"):
        _run(paths, tmp_path / "run")
    assert not (tmp_path / "run" / "prediction").exists()
    assert not (tmp_path / "run" / "score").exists()


def test_duplicate_exact_alias_refuses(tmp_path):
    paths = _make_fixture(tmp_path / "source")
    aliases = pd.read_csv(paths["aliases"], sep="\t")
    aliases.loc[1, "gene_symbol"] = aliases.loc[0, "gene_symbol"]
    paths["aliases"].write_text(aliases.to_csv(sep="\t", index=False))
    with pytest.raises(ValueError, match="exactly one-to-one"):
        _run(paths, tmp_path / "run")
