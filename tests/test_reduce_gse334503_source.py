from __future__ import annotations

import gzip
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from scipy import sparse
from scipy.io import mmwrite

from experiments import reduce_gse334503_source as subject


def _reference_row(name: str, index: int) -> dict[str, str]:
    return {
        "id": f"C{index:04d}",
        "name": name,
        "read": "R2",
        "pattern": "5PNNNNNNNNNN(BC)",
        "sequence": f"ACGT{index:011d}",
        "feature_type": "Antibody Capture",
    }


def _adt_reference() -> list[dict[str, str]]:
    names = [protein.removesuffix(".1") for _, protein in subject.PANEL]
    names.extend(
        [
            "Mouse IgG1; k isotype Ctrl",
            "Mouse IgG2a; k isotype Ctrl",
            "Mouse IgG2b; k isotype Ctrl",
            "Rat IgG2b; k Isotype Ctrl",
            "Rat IgG1; k isotype Ctrl",
            "Rat IgG2a; k Isotype Ctrl",
            "Armenian Hamster IgG Isotype Ctrl",
        ]
    )
    names.extend(f"PROTEIN{index:03d}" for index in range(137 - len(names)))
    return [_reference_row(name, index) for index, name in enumerate(names)]


def _hto_rows(batch: int = 1) -> list[dict[str, str]]:
    donors = (1, 2, 5, 8, 15, 25)
    rows = []
    tag = 1
    for donor in donors:
        for day in (0, 7):
            rows.append(
                {
                    "id": f"Human_HTO_{tag}",
                    "multiplex_sample": f"Donor{donor:03d}_Day{day}postVax",
                    "read": "R2",
                    "pattern": "5PNNNNNNNNNN(BC)",
                    "sequence": f"TGCA{tag:011d}",
                    "feature_type": "Antibody Capture",
                    "Batch": f"Batch{batch}",
                }
            )
            tag += 1
    rows.append(
        {
            "id": "Human_HTO_15",
            "multiplex_sample": "Ctrl",
            "read": "R2",
            "pattern": "5PNNNNNNNNNN(BC)",
            "sequence": "TGCA00000000015",
            "feature_type": "Antibody Capture",
            "Batch": f"Batch{batch}",
        }
    )
    return rows


def _feature_axes() -> tuple[list[str], list[str], list[dict[str, str]]]:
    gex = [gene for gene, _ in subject.PANEL]
    gex.extend(f"GENE{index:03d}" for index in range(200))
    gex.append("NONMT-CO1")
    reference = _adt_reference()
    gene_set = set(gex)
    biological = [
        row["name"] + (".1" if row["name"] in gene_set else "")
        for row in reference
    ]
    hto = [subject._hto_identity(row)[2] for row in _hto_rows()]
    return gex, biological + hto, reference


def _write_axis(path: Path, values: list[str]) -> None:
    with gzip.open(path, "wt") as stream:
        stream.writelines(f"{value}\n" for value in values)


def _write_matrix(path: Path, values: np.ndarray, field: str = "integer") -> None:
    with gzip.open(path, "wb") as stream:
        mmwrite(stream, sparse.coo_matrix(values), field=field)


def test_panel_and_stage_contract_are_exact_and_source_scoped(tmp_path: Path) -> None:
    assert len(subject.PANEL) == 22
    assert len(set(subject.PANEL)) == 22
    assert set(subject.SOURCE_BATCHES) == {1, 2, 3}
    assert subject._parse_batches("1,2") == (1, 2)
    assert subject._parse_batches("3") == (3,)
    assert subject._parse_batches("1,2,3") == (1, 2, 3)
    with pytest.raises(subject.argparse.ArgumentTypeError):
        subject._parse_batches("1,3")

    output, manifest = subject._output_paths((1, 2), None, None)
    assert output.name == "reduced_batches_1_2_v1.npz"
    assert manifest.name == "reduction_batches_1_2_manifest_v1.json"
    assert not subject.DEFAULT_CACHE.is_relative_to(subject.ROOT)
    assert subject._argument_parser().parse_args([]).batches == (1, 2)

    requested = []

    def record(url: str, path: Path, expected_bytes: int) -> None:
        requested.append((url, path.name, expected_bytes))

    original = subject._download
    try:
        subject._download = record
        subject._batch_paths(1, tmp_path)
    finally:
        subject._download = original
    assert len(requested) == 6
    assert all("GSM9789808" in url or "GSM9789809" in url for url, _, _ in requested)


def test_exact_feature_reference_split_and_hto_mapping() -> None:
    gex, adt, reference = _feature_axes()
    audit = subject._validate_feature_axes(1, gex, adt, reference, _hto_rows())
    assert audit["biological_adt_rows"] == 137
    assert audit["hto_rows"] == 13
    assert audit["mitochondrial_qc_evidence"] == {
        "status": "UNAVAILABLE_NO_CANONICAL_MITOCHONDRIAL_FEATURES",
        "applied": False,
        "recognition_rule": "feature name starts with 'MT-'",
        "recognized_feature_count": 0,
        "recognized_feature_axis": [],
        "evidence_axis_sha256": subject._axis_sha256(gex),
    }
    source_rows = subject._source_hto_rows(_hto_rows(), 1)
    identities = [subject._hto_identity(row) for row in source_rows]
    assert [identity[0] for identity in identities if identity[1] == "Day0"] == [
        "Donor001",
        "Donor002",
        "Donor005",
        "Donor008",
        "Donor015",
        "Donor025",
    ]
    assert identities[-1] == ("Ctrl", "Control", "Ctrl")

    drifted = adt.copy()
    drifted[136], drifted[137] = drifted[137], drifted[136]
    with pytest.raises(ValueError, match="first 137 ADT rows"):
        subject._validate_feature_axes(1, gex, drifted, reference, _hto_rows())

    with pytest.raises(ValueError, match="mitochondrial features"):
        subject._validate_feature_axes(
            1, [*gex, "MT-CO1"], adt, reference, _hto_rows()
        )


def test_integer_matrix_reader_checks_field_axes_and_duplicates(tmp_path: Path) -> None:
    integer = tmp_path / "integer.mtx.gz"
    _write_matrix(integer, np.asarray([[1, 0], [2, 3]], dtype=np.int32))
    observed, audit = subject._read_integer_matrix(integer)
    np.testing.assert_array_equal(observed.toarray(), [[1, 0], [2, 3]])
    assert audit["counts_nonnegative_integer"] is True

    real = tmp_path / "real.mtx.gz"
    _write_matrix(real, np.asarray([[1.5, 0], [0, 2.0]]), field="real")
    with pytest.raises(ValueError, match="coordinate integer general"):
        subject._read_integer_matrix(real)

    duplicate = tmp_path / "duplicate.mtx.gz"
    with gzip.open(duplicate, "wt") as stream:
        stream.write("%%MatrixMarket matrix coordinate integer general\n")
        stream.write("2 2 2\n1 1 1\n1 1 2\n")
    with pytest.raises(ValueError, match="duplicate coordinates"):
        subject._read_integer_matrix(duplicate)


def test_cached_reference_hash_drift_is_rejected_before_network(tmp_path: Path) -> None:
    name = "GSE334503_feature_reference_ADT.csv.gz"
    path = tmp_path / name
    path.write_bytes(b"x" * subject.EXPECTED_BYTES[name])
    with pytest.raises(ValueError, match="wrong SHA-256"):
        subject._download("https://example.invalid/not-accessed", path, path.stat().st_size)


def test_salted_selection_is_deterministic_and_enforces_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "CELL_BUDGET", 3)
    barcodes = [f"cell-{index}" for index in range(7)]
    assignment = np.asarray([0, 0, 0, 0, 0, 1, -1])
    rna_qc = np.asarray([True, True, False, True, True, True, True])
    selected = subject._select_donor_cells(
        1, "Donor001", barcodes, assignment, 0, rna_qc
    )
    expected = sorted(
        [0, 1, 3, 4],
        key=lambda index: (
            subject._selection_hash(1, "Donor001", barcodes[index]),
            barcodes[index],
        ),
    )[:3]
    assert selected.tolist() == expected
    with pytest.raises(ValueError, match="need 3"):
        subject._select_donor_cells(
            1, "Donor999", barcodes, assignment, 1, rna_qc
        )


def test_rna_qc_enforces_detected_gene_and_upper_umi_limits() -> None:
    total = np.asarray([1_000, 1_000, 70_000, 70_001, 0])
    detected = np.asarray([200, 199, 200, 200, 500])
    accepted, audit = subject._rna_qc_mask(total, detected)
    assert accepted.tolist() == [True, False, True, False, False]
    assert audit["above_maximum_total_rna"] == 1
    assert audit["below_minimum_detected_genes"] == 1
    assert "mitochondrial" not in " ".join(audit)


def test_adt_graph_profile_uses_cellwise_130_feature_clr() -> None:
    _, adt_features, reference = _feature_axes()
    denominator_rows, cognate_positions, specification = (
        subject._adt_graph_specification(adt_features, reference)
    )
    assert len(denominator_rows) == 130
    assert len(specification["excluded_isotype_reference_names"]) == 7
    assert specification["cell_clr_formula"] == (
        "clr_cell = log1p(count_130) - mean_j(log1p(count_130))"
    )

    cells = np.arange(4)[None, :]
    rows = np.arange(150)[:, None]
    values = 1 + ((rows * 2 + cells * 3 + rows * cells) % 11)
    adt = sparse.csr_matrix(values.astype(np.int32))
    selected = np.asarray([0, 2, 3])
    observed = subject._adt_graph_profile(
        adt, selected, denominator_rows, cognate_positions
    )
    counts_130 = values[denominator_rows][:, selected].T.astype(float)
    log_counts = np.log1p(counts_130)
    expected = (
        log_counts - log_counts.mean(axis=1, keepdims=True)
    ).mean(axis=0)[cognate_positions]
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-14)
    raw_mean_log = log_counts.mean(axis=0)[cognate_positions]
    assert not np.allclose(observed, raw_mean_log)


def _r_demux_available() -> bool:
    executable = shutil.which("Rscript")
    if executable is None:
        return False
    completed = subprocess.run(
        [
            executable,
            "-e",
            "quit(status=ifelse(requireNamespace('cluster',quietly=TRUE)&&requireNamespace('MASS',quietly=TRUE),0,1))",
        ],
        check=False,
        capture_output=True,
    )
    return completed.returncode == 0


@pytest.mark.skipif(not _r_demux_available(), reason="R cluster and MASS are required")
def test_seurat_htodemux_fixture_is_deterministic(tmp_path: Path) -> None:
    cells_per_group = 30
    truth = np.repeat(np.arange(13), cells_per_group)
    cells = np.arange(len(truth))[:, None]
    tags = np.arange(12)[None, :]
    counts = 1 + ((cells * 3 + tags * 5 + cells * tags) % 3)
    for index, group in enumerate(truth):
        if group < 12:
            counts[index, group] += 80 + index % 11
    counts = counts.astype(np.int32)

    first, first_audit = subject._seurat_htodemux(counts, tmp_path, "Rscript")
    second, second_audit = subject._seurat_htodemux(counts, tmp_path, "Rscript")
    np.testing.assert_array_equal(first, second)
    assert first_audit["cutoffs"] == [4, 6, 6, 4, 6, 6, 4, 6, 6, 4, 6, 6]
    assert first_audit["cutoffs"] == second_audit["cutoffs"]
    assert first_audit["classification_counts"] == {
        "negative": 30,
        "singlet": 360,
        "doublet": 0,
    }
    assert first_audit["classification_counts"] == second_audit["classification_counts"]
    signal = truth < 12
    assert np.mean(first[signal] == truth[signal]) > 0.98
    assert len(first_audit["fits"]) == 12


def test_loaded_batch_reduction_preserves_linked_axes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subject, "CELL_BUDGET", 2)
    gex_features, adt_features, reference = _feature_axes()
    barcodes = [f"cell-{index:03d}" for index in range(24)]
    gex_values = np.ones((len(gex_features), len(barcodes)), dtype=np.int32)
    gex_values[-1] = 0
    rows = np.arange(len(adt_features))[:, None]
    cells = np.arange(len(barcodes))[None, :]
    adt_values = (1 + ((rows * 2 + cells * 3 + rows * cells) % 11)).astype(
        np.int32
    )
    for cell, tag in enumerate(np.tile(np.arange(12), 2)):
        adt_values[137 + tag, cell] = 100

    paths = {
        "gex_barcodes": tmp_path / "gex_barcodes.tsv.gz",
        "gex_features": tmp_path / "gex_features.tsv.gz",
        "gex_matrix": tmp_path / "gex_matrix.mtx.gz",
        "adt_barcodes": tmp_path / "adt_barcodes.tsv.gz",
        "adt_features": tmp_path / "adt_features.tsv.gz",
        "adt_matrix": tmp_path / "adt_matrix.mtx.gz",
    }
    _write_axis(paths["gex_barcodes"], barcodes)
    _write_axis(paths["adt_barcodes"], barcodes)
    _write_axis(paths["gex_features"], gex_features)
    _write_axis(paths["adt_features"], adt_features)
    _write_matrix(paths["gex_matrix"], gex_values)
    _write_matrix(paths["adt_matrix"], adt_values)

    assignment = np.tile(np.arange(12), 2).astype(np.int16)

    def demux(counts: np.ndarray, cache: Path, rscript: str):
        del cache, rscript
        assert counts.shape == (24, 12)
        return assignment, {"method": "fixture"}

    monkeypatch.setattr(subject, "_seurat_htodemux", demux)
    donors, audit, graph_specification = subject._reduce_loaded_batch(
        1, paths, reference, _hto_rows(), tmp_path, "Rscript"
    )
    assert len(donors) == 6
    assert audit["cell_count"] == 24
    assert audit["rna_qc_pass_count"] == 24
    assert all(donor["rna_counts"].shape == (2, 22) for donor in donors)
    assert all(donor["adt_counts"].shape == (2, 22) for donor in donors)
    assert all(donor["adt_graph_profile"].shape == (22,) for donor in donors)
    assert graph_specification["denominator_feature_count"] == 130
    assert [donor["donor"] for donor in donors] == [
        "Donor001",
        "Donor002",
        "Donor005",
        "Donor008",
        "Donor015",
        "Donor025",
    ]
