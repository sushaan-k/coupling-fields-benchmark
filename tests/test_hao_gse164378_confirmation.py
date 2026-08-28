import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.io import mmwrite

import experiments.confirm_hao_gse164378 as hao
from experiments.confirm_hao_gse164378 import (
    _field_table,
    _margin_stats,
    _reducer_artifact_bundle,
    _require_reducer_artifact_bundle,
    _require_score_authorization,
    _require_score_release,
    _residual_table,
    _sha256,
    _table_stats,
    summarize,
)
from experiments.reduce_hao_gse164378 import (
    REQUIRED_SOURCE_FILES,
    _read_matrix_subset,
    reduce,
)


def _states(table):
    first, second = [], []
    for row, column in np.ndindex(table.shape):
        first.extend([row] * int(table[row, column]))
        second.extend([column] * int(table[row, column]))
    return np.asarray(first), np.asarray(second)


def test_disabled_lock_freezes_split_aliases_and_family_reporting():
    designation = json.loads(hao.DESIGNATION.read_text())
    assert designation["schema"].endswith("/1.0")
    assert designation["status"] == "OUTCOME_ACCESS_DISABLED"
    assert designation["outcome_access_authorized"] is False
    assert designation["development_units"] == list(hao.DEVELOPMENT)
    assert designation["held_units"] == list(hao.HELD)
    assert designation["excluded_units"] == ["P6"]
    assert designation["lineages"] == list(hao.LINEAGES)
    assert designation["primary_blocks"] == ["day3", "day7"]
    assert "DC" not in designation["lineages"]
    assert designation["score_output"] == "results/hao_gse164378_confirmation.json"
    assert designation["score_arrays"].endswith("_confirmation_arrays.npz")
    assert designation["score_release"].endswith("score_release_v1.json")
    assert designation["confirmatory_family"]["execute_all_candidates"] is True
    assert designation["confirmatory_family"]["scoreable_candidates"] == [
        "Lawlor HCA PBMC",
        "Hao GSE164378",
    ]
    with pytest.raises(PermissionError, match="not SEALED"):
        hao.preflight(require_sealed=True)


def test_exact_alias_table_has_no_fallback_or_ambiguous_cd16():
    aliases = pd.read_csv(hao.ALIASES, sep="\t")
    assert len(aliases) == 33
    assert not aliases["adt_feature"].duplicated().any()
    assert not aliases["gene_symbol"].str.upper().duplicated().any()
    assert "CD16" not in set(aliases["adt_feature"])
    reducer = hao.REDUCER.read_text()
    assert "fallback" not in reducer.lower()
    assert "missing or duplicated" in reducer
    assert ".upper()" not in reducer


def test_metadata_support_is_outcome_free_and_covers_every_frozen_block():
    support = json.loads(hao.SUPPORT.read_text())
    assert support["outcome_matrices_accessed"] is False
    assert support["source_time_code_map"] == {"0": "day0", "2": "day3", "7": "day7"}
    assert support["broad_lineages"] == list(hao.LINEAGES)
    assert support["minimum_cells_in_any_retained_donor_time_lineage"] >= hao.MINIMUM_CELLS
    assert support["minimum_cells_in_any_held_donor_time_lineage"] >= 71
    assert len(support["post_vaccine_blocks"]) == len(hao.BLOCKS) * len(hao.LINEAGES)


def test_source_lock_uses_only_six_direct_5p_files():
    source = json.loads(hao.SOURCE_MANIFEST.read_text())
    by_name = {record["name"]: record for record in source["files"]}
    locked = [by_name[name] for name in REQUIRED_SOURCE_FILES]
    assert [record["name"] for record in locked] == list(REQUIRED_SOURCE_FILES)
    assert sum(record["bytes"] for record in locked) == 226_047_108
    assert source["required_5p_source_bytes"] == 226_047_108
    pinned = {
        "GSM5008740_RNA_5P-barcodes.tsv.gz": "d3d316a3973ec745d9768e18124fb6518f9a434b404147fcb05ba25c9d68c7b7",
        "GSM5008740_RNA_5P-features.tsv.gz": "9f3a435942cab5fa1bf41aac342d17ec4818d8c20139b5bcb3713cdedfdee7d1",
        "GSM5008741_ADT_5P-barcodes.tsv.gz": "d3d316a3973ec745d9768e18124fb6518f9a434b404147fcb05ba25c9d68c7b7",
        "GSM5008741_ADT_5P-features.tsv.gz": "979b9f96fdcc951eaf7f34df67b0ba0cb2ab696ee75ef6064a6a0bb70654e53a",
    }
    for record in locked:
        assert record["url"].startswith("https://ftp.ncbi.nlm.nih.gov/geo/samples/")
        assert record["url"].endswith("/suppl/" + record["name"])
        assert record["sha256"] == pinned.get(record["name"])
    assert [record["accessed_before_seal"] for record in locked] == [
        True,
        True,
        False,
        True,
        True,
        False,
    ]
    assert source["outcome_matrix_values_accessed_before_seal"] is False
    combined = hao.REDUCER.read_text() + Path(hao.__file__).read_text()
    assert "--archive" not in combined
    assert "GSE164378_RAW.tar" not in combined


def test_marker_lineage_support_filter_uses_separate_held_margins():
    rows = []
    values = []
    for donor in hao.DEVELOPMENT + hao.HELD:
        for day in (0, *hao.DAYS):
            for lineage in hao.LINEAGES:
                for rank in range(hao.MINIMUM_CELLS):
                    rows.append(
                        {
                            "cell_id": f"{donor}-{day}-{lineage}-{rank}",
                            "donor": donor,
                            "day": day,
                            "cell_type": lineage,
                        }
                    )
                    values.append(float(rank))
    cells = pd.DataFrame(rows)
    markers = pd.DataFrame(
        {
            "marker_id": [f"ADT{i}::GENE{i}" for i in range(12)],
            "gene_symbol": [f"GENE{i}" for i in range(12)],
        }
    )
    rna = np.tile(np.asarray(values), (len(markers), 1))
    adt = rna.copy()
    unsupported = (
        (cells["donor"] == "P5")
        & (cells["day"] == 3)
        & (cells["cell_type"] == "B")
    ).to_numpy()
    adt[0, unsupported] = 0.0
    entity_ids, _, _, excluded, rna_state, adt_state = hao._state_thresholds(
        cells, markers, rna, adt
    )
    assert "ADT0::GENE0@@B" in excluded
    assert "ADT0::GENE0@@CD4 T" in entity_ids
    assert len(entity_ids) == len(markers) * len(hao.LINEAGES) - 1
    assert rna_state.shape == adt_state.shape == (len(entity_ids), len(cells))
    with pytest.raises(ValueError, match="fewer than 12 unique cognate markers"):
        hao._state_thresholds(
            cells,
            markers.iloc[:11].reset_index(drop=True),
            rna[:11],
            adt[:11],
        )


def test_membership_permutation_preserves_represented_lineage_hyperedges(
    tmp_path, monkeypatch
):
    embedding_path = tmp_path / "embedding.npz"
    cluster_genes = [f"G{index}" for index in range(8)]
    np.savez(
        embedding_path,
        gene_names=np.asarray(cluster_genes),
        embedding=np.random.default_rng(4).normal(size=(8, 5)),
    )
    monkeypatch.setattr(hao, "ROOT", tmp_path)
    monkeypatch.setattr(hao, "SCGPT", embedding_path)
    genes = [gene for gene in cluster_genes for _ in range(2)]
    lineages = ["B", "CD4 T"] * len(cluster_genes)
    original, original_metadata = hao._embedding_laplacian(genes, lineages)
    permutation = np.arange(len(cluster_genes))[::-1]
    permuted, permuted_metadata = hao._embedding_laplacian(
        genes,
        lineages,
        gene_membership_permutation=permutation,
    )
    assert original_metadata["lineage_hyperedges"] == ["B", "CD4 T"]
    assert (
        original_metadata["lineage_memberships"]
        == permuted_metadata["lineage_memberships"]
    )
    assert permuted_metadata["gene_membership_permutation"] == permutation.tolist()
    assert not np.allclose(original, permuted)


def test_margin_null_reference_is_independent_of_pairing_order():
    first = np.repeat(np.arange(3), [20, 30, 40])
    second = np.repeat(np.arange(3), [25, 35, 30])
    generator = np.random.default_rng(9)
    original = _margin_stats(first, second, seed=44)
    permuted = _margin_stats(
        first[generator.permutation(len(first))],
        second[generator.permutation(len(second))],
        seed=44,
    )
    for family in ("field", "pearson", "deviance"):
        np.testing.assert_allclose(
            original[f"{family}_null"], permuted[f"{family}_null"]
        )


def test_absolute_field_and_residual_reconstruction_recover_table():
    table = np.array([[18, 3, 4], [5, 20, 5], [7, 6, 22]])
    stats = _table_stats(*_states(table), seed=11)
    predicted = _field_table(
        stats["field"], stats["field_null"], stats["rows"], stats["columns"]
    )
    np.testing.assert_allclose(predicted.sum(axis=1), table.sum(axis=1))
    np.testing.assert_allclose(predicted.sum(axis=0), table.sum(axis=0))
    np.testing.assert_allclose(
        predicted,
        hao.field_coordinates_to_table(
            stats["field_raw"], stats["rows"], stats["columns"]
        ),
    )
    for family in ("pearson", "deviance"):
        predicted = _residual_table(
            stats[family],
            stats[f"{family}_null"],
            stats["total"],
            stats["rows"],
            stats["columns"],
            family,
        )
        np.testing.assert_allclose(predicted, table, atol=1e-7)


def test_prediction_inputs_are_invariant_to_held_within_block_pairing(monkeypatch):
    rows = []
    for donor in hao.DEVELOPMENT + hao.HELD:
        for day in (0, *hao.DAYS):
            for lineage in hao.LINEAGES:
                rows.extend(
                    {"donor": donor, "day": day, "cell_type": lineage}
                    for _ in range(hao.MINIMUM_CELLS + 1)
                )
    cells = pd.DataFrame(rows)
    state = np.resize(np.arange(3), len(cells))[None, :]
    calls = []

    def margins(first, second, seed):
        del seed
        row = np.bincount(first, minlength=3).astype(float)
        column = np.bincount(second, minlength=3).astype(float)
        return {
            "rows": row,
            "columns": column,
            "total": float(len(first)),
            "endpoint": np.zeros(4),
            "field_null": np.zeros((2, 2)),
            "field_destroyed": np.zeros((2, 2)),
            "field_variance": np.ones((2, 2)),
            "pearson_null": np.zeros((3, 3)),
            "pearson_destroyed": np.zeros((3, 3)),
            "pearson_variance": np.ones((3, 3)),
            "deviance_null": np.zeros((3, 3)),
            "deviance_destroyed": np.zeros((3, 3)),
            "deviance_variance": np.ones((3, 3)),
        }

    def table_stats(first, second, seed):
        calls.append(seed)
        result = margins(first, second, seed)
        return {
            **result,
            "table": hao._table(first, second),
            "field": np.zeros((2, 2)),
            "field_raw": np.zeros((2, 2)),
            "covariance": np.zeros((2, 2)),
            "pearson": np.zeros((3, 3)),
            "pearson_raw": np.zeros((3, 3)),
            "deviance": np.zeros((3, 3)),
            "deviance_raw": np.zeros((3, 3)),
        }

    monkeypatch.setattr(hao, "_margin_stats", margins)
    monkeypatch.setattr(hao, "_table_stats", table_stats)
    result = hao._build_fields(
        cells, ["marker@@B"], ["B"], state, state, open_held_pairing=False
    )
    assert len(calls) == len(hao.DEVELOPMENT) * (1 + len(hao.BLOCKS))
    permuted_adt = state.copy()
    rng = np.random.default_rng(18)
    for donor in hao.HELD:
        for day in (0, *hao.DAYS):
            mask = (
                (cells["donor"] == donor)
                & (cells["day"] == day)
                & (cells["cell_type"] == "B")
            ).to_numpy()
            permuted_adt[0, mask] = rng.permutation(permuted_adt[0, mask])
    permuted = hao._build_fields(
        cells,
        ["marker@@B"],
        ["B"],
        state,
        permuted_adt,
        open_held_pairing=False,
    )
    for name in result:
        np.testing.assert_allclose(result[name], permuted[name], equal_nan=True)
    held = slice(len(hao.DEVELOPMENT), None)
    assert np.isnan(result["field"][held]).all()
    assert np.isnan(result["table"][held]).all()
    assert np.isnan(result["baseline_field"][held]).all()
    assert np.isfinite(result["rows"][held]).all()


def _write_reducer_bundle(reduced: Path) -> None:
    reduced.mkdir()
    rows = []
    for index, relative in enumerate(hao.REDUCER_OUTPUTS):
        path = reduced / relative
        path.write_bytes(f"artifact-{index}\n".encode())
        rows.append(f"{relative}\t{path.stat().st_size}\t{_sha256(path)}")
    (reduced / hao.REDUCER_MANIFEST).write_text(
        "path\tbytes\tsha256\n" + "\n".join(rows) + "\n"
    )


def test_reducer_bundle_binds_every_output_manifest_and_rejects_extra(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(hao, "ROOT", tmp_path)
    reduced = tmp_path / "data" / "reduced"
    reduced.parent.mkdir()
    _write_reducer_bundle(reduced)
    bundle = _reducer_artifact_bundle(reduced)
    assert [record["path"] for record in bundle["artifacts"]] == [
        *hao.REDUCER_OUTPUTS,
        hao.REDUCER_MANIFEST,
    ]
    assert _require_reducer_artifact_bundle(reduced, bundle) == bundle
    (reduced / "extra.txt").write_text("unexpected\n")
    with pytest.raises(PermissionError, match="frozen prediction provenance"):
        _require_reducer_artifact_bundle(reduced, bundle)


def test_score_requires_exact_hash_and_40_hex_github_blob(tmp_path, monkeypatch):
    monkeypatch.setattr(hao, "ROOT", tmp_path)
    prediction = tmp_path / "results" / "prediction.json"
    prediction.parent.mkdir()
    prediction.write_text('{"frozen": true}\n')
    authorization = tmp_path / "authorization.json"
    commit = "0123456789abcdef0123456789abcdef01234567"
    provenance = {}
    record = {
        "schema": "hao-gse164378-score-authorization/1.0",
        "status": "SEALED",
        "outcome_access_authorized": True,
        "prediction_path": "results/prediction.json",
        "prediction_sha256": _sha256(prediction),
        "prediction_bytes": prediction.stat().st_size,
        "prediction_public_url": (
            "https://github.com/example/benchmark/blob/"
            f"{commit}/results/prediction.json"
        ),
        "prediction_public_commit": commit,
        "runner_sha256": _sha256(Path(hao.__file__)),
        "protocol_sha256": _sha256(hao.PROTOCOL),
        "frozen_provenance": provenance,
    }
    authorization.write_text(json.dumps(record))
    authorization_url = (
        "https://github.com/example/benchmark/blob/"
        f"{commit}/authorization.json"
    )
    assert _require_score_authorization(prediction, authorization, provenance) == record
    release = tmp_path / "release.json"
    release_record = {
        "schema": "hao-gse164378-score-release/1.0",
        "status": "SEALED",
        "held_pairing_access_authorized": True,
        "authorization_path": "authorization.json",
        "authorization_sha256": _sha256(authorization),
        "authorization_bytes": authorization.stat().st_size,
        "authorization_public_url": authorization_url,
        "authorization_public_commit": commit,
        "runner_sha256": _sha256(Path(hao.__file__)),
        "protocol_sha256": _sha256(hao.PROTOCOL),
        "frozen_provenance": provenance,
    }
    release.write_text(json.dumps(release_record))
    assert _require_score_release(authorization, release, provenance) == release_record
    release_record["authorization_public_url"] = authorization_url.replace(
        "authorization.json", "other.json"
    )
    release.write_text(json.dumps(release_record))
    with pytest.raises(PermissionError, match="exact artifact path"):
        _require_score_release(authorization, release, provenance)
    record["prediction_public_commit"] = "0" * 39
    authorization.write_text(json.dumps(record))
    with pytest.raises(PermissionError, match="40 lowercase"):
        _require_score_authorization(prediction, authorization, provenance)


def test_gate_requires_primary_to_beat_best_matched_method():
    markers = 12
    outputs = len(hao.BLOCKS) * 4
    primary = np.linspace(-1.0, 1.0, markers * outputs).reshape(markers, outputs)
    arrays = {
        "field": np.zeros(
            (len(hao.DEVELOPMENT) + len(hao.HELD), markers, len(hao.BLOCKS), 2, 2)
        ),
        "baseline_field": np.zeros(
            (len(hao.DEVELOPMENT) + len(hao.HELD), markers, 2, 2)
        ),
        "pearson": np.zeros(
            (len(hao.DEVELOPMENT) + len(hao.HELD), markers, len(hao.BLOCKS), 3, 3)
        ),
        "deviance": np.zeros(
            (len(hao.DEVELOPMENT) + len(hao.HELD), markers, len(hao.BLOCKS), 3, 3)
        ),
    }
    arrays["field"][len(hao.DEVELOPMENT) :] = primary.reshape(
        markers, len(hao.BLOCKS), 2, 2
    )
    predictions = {
        "field_primary": primary,
        "field_direct": primary,
        "pearson_direct": np.zeros((markers, len(hao.BLOCKS) * 9)),
        "pearson_structured": np.zeros((markers, len(hao.BLOCKS) * 9)),
        "deviance_direct": np.zeros((markers, len(hao.BLOCKS) * 9)),
        "deviance_structured": np.zeros((markers, len(hao.BLOCKS) * 9)),
    }
    secondary = {
        "field_change_primary": primary,
        "field_change_direct": primary,
    }
    matched = (
        "field_direct",
        "field_zero",
        "field_scalar",
        "field_nuclear",
        "field_hypergraph",
        "field_endpoint_ridge",
        "covariance_direct",
        "field_membership_permuted",
    )
    losses = {
        "field_primary": np.full((3, markers, len(hao.BLOCKS)), 0.2),
        "field_destroyed": np.full((3, markers, len(hao.BLOCKS)), 0.5),
        "pearson_direct": np.full((3, markers, len(hao.BLOCKS)), 0.5),
        "pearson_structured": np.full((3, markers, len(hao.BLOCKS)), 0.5),
        "deviance_direct": np.full((3, markers, len(hao.BLOCKS)), 0.5),
        "deviance_structured": np.full((3, markers, len(hao.BLOCKS)), 0.5),
        **{
            name: np.full((3, markers, len(hao.BLOCKS)), 0.4)
            for name in matched
        },
    }
    clusters = [f"marker-{index}" for index in range(markers)]
    summary, _ = summarize(
        arrays, predictions, secondary, losses, marker_clusters=clusters
    )
    assert summary["gate_passed"] is True
    losses["field_zero"][:] = 0.1
    summary, _ = summarize(
        arrays, predictions, secondary, losses, marker_clusters=clusters
    )
    assert summary["gate_passed"] is False


def _gzip_bytes(value: bytes) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as stream:
        stream.write(value)
    return output.getvalue()


def _matrix_bytes(matrix: sparse.spmatrix) -> bytes:
    plain = io.BytesIO()
    mmwrite(plain, matrix, field="integer", symmetry="general")
    return _gzip_bytes(plain.getvalue())


def test_matrix_reader_rejects_axis_mismatch_and_invalid_entries(tmp_path):
    matrix = tmp_path / "matrix.mtx.gz"
    matrix.write_bytes(
        _gzip_bytes(
            b"%%MatrixMarket matrix coordinate integer general\n2 2 1\n1 1 4\n"
        )
    )
    with pytest.raises(ValueError, match="dimensions differ"):
        _read_matrix_subset(
            matrix,
            {0: 0},
            {0: 0},
            expected_rows=3,
            expected_columns=2,
            collect_column_totals=False,
        )
    matrix.write_bytes(
        _gzip_bytes(
            b"%%MatrixMarket matrix coordinate integer general\n2 2 1\n3 1 4\n"
        )
    )
    with pytest.raises(ValueError, match="invalid Matrix Market entry"):
        _read_matrix_subset(
            matrix,
            {0: 0},
            {0: 0},
            expected_rows=2,
            expected_columns=2,
            collect_column_totals=False,
        )


def test_streaming_reducer_hashes_direct_files_and_uses_exact_aliases(tmp_path):
    cells = [f"L1_CELL{i:02d}" for i in range(7)]
    genes = [f"G{i:02d}" for i in range(33)]
    adts = [f"A{i:02d}" for i in range(33)]
    rna_features = "".join(f"ENSG{i:05d}\t{gene}\n" for i, gene in enumerate(genes))
    members = {
        "GSM5008740_RNA_5P-barcodes.tsv.gz": _gzip_bytes(
            "".join(f"{cell}\n" for cell in cells).encode()
        ),
        "GSM5008740_RNA_5P-features.tsv.gz": _gzip_bytes(rna_features.encode()),
        "GSM5008740_RNA_5P-matrix.mtx.gz": _matrix_bytes(
            sparse.csc_matrix(np.arange(1, 33 * 7 + 1).reshape(33, 7))
        ),
        "GSM5008741_ADT_5P-barcodes.tsv.gz": _gzip_bytes(
            "".join(f"{cell}\n" for cell in cells).encode()
        ),
        "GSM5008741_ADT_5P-features.tsv.gz": _gzip_bytes(
            "".join(f"{adt}\n" for adt in adts).encode()
        ),
        "GSM5008741_ADT_5P-matrix.mtx.gz": _matrix_bytes(
            sparse.csc_matrix(np.arange(1, 33 * 7 + 1).reshape(33, 7))
        ),
    }
    source_files = {}
    for name, value in members.items():
        path = tmp_path / name
        path.write_bytes(value)
        source_files[name] = path
    metadata = tmp_path / "metadata.csv.gz"
    pd.DataFrame(
        {
            "Unnamed: 0": cells,
            "donor": list(hao.DEVELOPMENT) + list(hao.HELD),
            "time": [0, 2, 7, 0, 2, 7, 0],
            "celltype.l1": np.resize(np.asarray(hao.LINEAGES), len(cells)).tolist(),
        }
    ).to_csv(metadata, index=False, compression="gzip")
    aliases = tmp_path / "aliases.tsv"
    pd.DataFrame({"adt_feature": adts, "gene_symbol": genes}).to_csv(
        aliases, sep="\t", index=False
    )
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "schema": "hao-gse164378-source/1.0",
                "files": [
                    {
                        "name": "GSE164378_sc.meta.data_5P.csv.gz",
                        "bytes": metadata.stat().st_size,
                        "sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
                    },
                    *[
                        {
                            "name": name,
                            "bytes": len(value),
                            "url": f"https://example.org/{name}",
                        }
                        for name, value in members.items()
                    ],
                ],
                "required_5p_source_bytes": sum(
                    len(value) for value in members.values()
                ),
            }
        )
    )
    output = tmp_path / "reduced"
    reduce(
        rna_barcodes_path=source_files["GSM5008740_RNA_5P-barcodes.tsv.gz"],
        rna_features_path=source_files["GSM5008740_RNA_5P-features.tsv.gz"],
        rna_matrix_path=source_files["GSM5008740_RNA_5P-matrix.mtx.gz"],
        adt_barcodes_path=source_files["GSM5008741_ADT_5P-barcodes.tsv.gz"],
        adt_features_path=source_files["GSM5008741_ADT_5P-features.tsv.gz"],
        adt_matrix_path=source_files["GSM5008741_ADT_5P-matrix.mtx.gz"],
        metadata_path=metadata,
        aliases_path=aliases,
        source_path=source,
        output=output,
    )
    acquisition = json.loads((output / "source_acquisition.json").read_text())
    assert len(acquisition["files"]) == 6
    assert {
        record["name"]: record["sha256"] for record in acquisition["files"]
    } == {
        name: hashlib.sha256(value).hexdigest() for name, value in members.items()
    }
    marker = pd.read_csv(output / "markers.tsv", sep="\t")
    assert marker["adt_feature"].tolist() == adts
    assert marker["gene_symbol"].tolist() == genes
    with pytest.raises(FileExistsError, match="already exists"):
        reduce(
            rna_barcodes_path=source_files["GSM5008740_RNA_5P-barcodes.tsv.gz"],
            rna_features_path=source_files["GSM5008740_RNA_5P-features.tsv.gz"],
            rna_matrix_path=source_files["GSM5008740_RNA_5P-matrix.mtx.gz"],
            adt_barcodes_path=source_files["GSM5008741_ADT_5P-barcodes.tsv.gz"],
            adt_features_path=source_files["GSM5008741_ADT_5P-features.tsv.gz"],
            adt_matrix_path=source_files["GSM5008741_ADT_5P-matrix.mtx.gz"],
            metadata_path=metadata,
            aliases_path=aliases,
            source_path=source,
            output=output,
        )
