import gzip
import io
from pathlib import Path
import tarfile

import numpy as np
import pytest

from mapreg.streamed_matrix_market import (
    MatrixMarketValidationError,
    read_tar_axes,
    read_tar_matrix_subset,
)


GENES = "bundle/genes.tsv"
BARCODES = "bundle/barcodes.tsv.gz"
MATRIX = "bundle/matrix.mtx"


def _archive(tmp_path: Path, matrix: str, *, genes=None, barcodes=None) -> Path:
    path = tmp_path / "matrix.tar.gz"
    members = {
        GENES: genes or "id1\tG1\nid2\tG2\nid3\tG3\n",
        BARCODES: barcodes or "bc1\nbc2\nbc3\n",
        MATRIX: matrix,
    }
    with tarfile.open(path, "w:gz") as archive:
        for name, text in members.items():
            payload = text.encode()
            if name.endswith(".gz"):
                payload = gzip.compress(payload)
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


def _matrix(entries: list[str], *, shape="3 3", declared=None, field="integer") -> str:
    count = len(entries) if declared is None else declared
    return "\n".join(
        [
            f"%%MatrixMarket matrix coordinate {field} general",
            "% test fixture",
            f"{shape} {count}",
            *entries,
            "",
        ]
    )


def _stream(path: Path, **overrides):
    axes = read_tar_axes(path, MATRIX, GENES, BARCODES)
    arguments = {
        "requested_rows": {"G3": 2, "G1": 0},
        "authorized_barcodes": ("bc3", "bc1"),
    }
    arguments.update(overrides)
    return read_tar_matrix_subset(path, axes, MATRIX, **arguments)


def test_axis_read_stops_after_matrix_preamble(tmp_path, monkeypatch):
    path = _archive(tmp_path, _matrix(["this body is deliberately invalid"]))
    opened = []
    original = tarfile.TarFile.extractfile

    def recording_extractfile(self, member):
        opened.append(member.name if isinstance(member, tarfile.TarInfo) else member)
        return original(self, member)

    monkeypatch.setattr(tarfile.TarFile, "extractfile", recording_extractfile)
    axes = read_tar_axes(path, MATRIX, GENES, BARCODES)

    assert axes.features == (("id1", "G1"), ("id2", "G2"), ("id3", "G3"))
    assert axes.barcodes == ("bc1", "bc2", "bc3")
    assert axes.matrix_shape == (3, 3)
    assert opened == [GENES, BARCODES, MATRIX]


def test_streams_only_requested_rectangle_and_audits_value_access(tmp_path):
    # This token exceeds Python 3.11's integer-string conversion limit. The read
    # succeeds because bc2 is unauthorized and its value is never converted.
    inaccessible_value = "9" * 5_000
    path = _archive(
        tmp_path,
        _matrix(
            [
                "1 1 5",
                "2 1 7",
                "3 1 2",
                f"1 2 {inaccessible_value}",
                "2 2 4",
                "1 3 6",
                "3 3 8",
            ]
        ),
    )

    counts, audit = _stream(path)

    np.testing.assert_array_equal(counts, [[8, 6], [2, 5]])
    assert audit.declared_entries == 7
    assert audit.entries_seen == 7
    assert audit.authorized_column_entries == 5
    assert audit.unauthorized_column_entries == 2
    assert audit.authorized_column_unrequested_row_entries == 1
    assert audit.selected_entries_materialized == 4
    assert audit.value_tokens_lexically_validated == 7
    assert audit.value_tokens_converted == 4
    assert audit.unauthorized_value_tokens_converted == 0
    assert audit.column_major_monotone


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (_matrix(["1 1 1"], field="real"), "header"),
        (_matrix(["1 1 1"], shape="2 3"), "dimensions"),
        (_matrix(["1 1 -1"]), "negative count"),
        (_matrix(["1 1 1.5"]), "non-integer"),
        (_matrix(["4 1 1"]), "out-of-range"),
        (_matrix(["1 1 1"], declared=2), "declares 2 entries"),
        (_matrix(["1 1 1", "1 1 2"]), "duplicates a coordinate"),
        (
            _matrix(["1 1 1", "3 2 1", "2 3 1", "2 1 1"]),
            "not monotone",
        ),
    ],
)
def test_rejects_malformed_or_unverifiable_matrices(tmp_path, matrix, message):
    path = _archive(tmp_path, matrix)
    with pytest.raises(MatrixMarketValidationError, match=message):
        _stream(path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"requested_rows": {"missing": 0}}, "does not resolve uniquely"),
        ({"authorized_barcodes": ["missing"]}, "absent from archive"),
        ({"requested_rows": {"G1": 2}}, "does not resolve uniquely"),
        ({"authorized_barcodes": []}, "must not be empty"),
    ],
)
def test_requires_exact_unique_requested_axes(tmp_path, overrides, message):
    path = _archive(tmp_path, _matrix(["1 1 1"]))
    with pytest.raises((ValueError, MatrixMarketValidationError), match=message):
        _stream(path, **overrides)


@pytest.mark.parametrize(
    ("genes", "barcodes", "message"),
    [
        ("id1\tG1\nid1\tG2\n", None, "duplicate features"),
        (None, "bc1\nbc1\n", "duplicate barcodes"),
        ("\tG1\n", None, "no feature identifier"),
    ],
)
def test_rejects_nonunique_or_incomplete_axis_members(
    tmp_path, genes, barcodes, message
):
    path = _archive(
        tmp_path,
        _matrix(["1 1 1"], shape="1 1"),
        genes=genes,
        barcodes=barcodes,
    )
    with pytest.raises(MatrixMarketValidationError, match=message):
        read_tar_axes(path, MATRIX, GENES, BARCODES)


def test_selected_count_must_fit_materialized_integer_dtype(tmp_path):
    path = _archive(
        tmp_path,
        _matrix([f"1 1 {np.iinfo(np.int64).max + 1}"]),
    )
    with pytest.raises(MatrixMarketValidationError, match="exceeds int64"):
        _stream(path, requested_rows={"G1": 0}, authorized_barcodes=["bc1"])


def test_requested_mapping_accepts_zero_and_one_based_indices_in_mapping_order(tmp_path):
    path = _archive(tmp_path, _matrix(["1 1 3", "3 1 9"]))
    counts, _ = _stream(
        path,
        requested_rows={"G3": 3, "G1": 0},
        authorized_barcodes=["bc1"],
    )
    np.testing.assert_array_equal(counts, [[9, 3]])
