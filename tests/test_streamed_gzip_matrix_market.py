from __future__ import annotations

from dataclasses import asdict, FrozenInstanceError
import gzip
import hashlib
import io
from pathlib import Path
import zlib

import numpy as np
import pytest

from mapreg.streamed_gzip_matrix_market import (
    GzipMatrixMarketPartialAudit,
    GzipMatrixMarketValidationError,
    reduce_gzip_matrix_market,
)


def _payload(
    entries: list[str],
    *,
    shape: tuple[int, int] = (4, 5),
    declared_nnz: int | None = None,
    banner: str = "%%MatrixMarket matrix coordinate integer general",
) -> bytes:
    nnz = len(entries) if declared_nnz is None else declared_nnz
    text = "\n".join(
        [
            banner,
            "% deterministic fixture",
            f"{shape[0]} {shape[1]} {nnz}",
            *entries,
            "",
        ]
    )
    return gzip.compress(text.encode("ascii"), mtime=0)


def _raw_payload(*lines: str) -> bytes:
    return gzip.compress(("\n".join(lines) + "\n").encode("ascii"), mtime=0)


def _reduce(source, **overrides):
    arguments = {
        "expected_shape": (4, 5),
        "selected_rows": (4, 1),
        "selected_columns": (5, 2),
    }
    arguments.update(overrides)
    return reduce_gzip_matrix_market(source, **arguments)


def test_unsorted_duplicate_coordinates_are_accumulated_and_fully_audited() -> None:
    compressed = _payload(
        [
            "4 5 3",
            "2 3 7",
            "1 2 5",
            "4 5 4",
            "2 3 11",
            "4 2 0",
            "1 2 6",
        ]
    )

    block, audit = _reduce(io.BytesIO(compressed))

    np.testing.assert_array_equal(block, [[7, 0], [0, 11]])
    assert audit.declared_nnz == 7
    assert audit.parsed_nnz == 7
    assert audit.selected_entries == 5
    assert audit.selected_distinct_coordinates == 3
    assert audit.selected_duplicate_entries == 2
    assert audit.zero_value_entries == 1
    assert audit.global_value_sum == 36
    assert audit.selected_value_sum == 18
    assert audit.gzip_stream_exhausted
    assert audit.output_dtype == "int64"


def test_integral_real_field_is_accepted_exactly_and_audited() -> None:
    compressed = _payload(
        [
            "4 5 3.0000000e+00",
            "1 2 5e0",
            "4 2 .0",
            "1 2 6.00E+0",
            "3 4 1.2300e2",
            "2 3 9.007199254740993e15",
        ],
        banner="%%MatrixMarket matrix coordinate real general",
    )

    block, audit = _reduce(io.BytesIO(compressed), allow_integral_real=True)

    np.testing.assert_array_equal(block, [[3, 0], [0, 11]])
    assert audit.banner == "%%MatrixMarket matrix coordinate real general"
    assert audit.global_value_sum == 9_007_199_254_741_130
    assert audit.zero_value_entries == 1
    assert audit.output_dtype == "int64"


@pytest.mark.parametrize(
    "value",
    [
        "-0.0",
        "-1.0",
        "1.5",
        "nan",
        "inf",
        "1_0.0",
        "0x1",
        "1e",
        "1e-1",
        "1e999999999999999999999",
        "1.0000000000000000001",
    ],
)
def test_real_field_rejects_non_count_values(value: str) -> None:
    compressed = _payload(
        [f"1 2 {value}"], banner="%%MatrixMarket matrix coordinate real general"
    )

    with pytest.raises(
        GzipMatrixMarketValidationError,
        match="expected a finite nonnegative integral real",
    ):
        _reduce(io.BytesIO(compressed), allow_integral_real=True)


def test_real_field_rejects_fractional_unselected_entry() -> None:
    compressed = _payload(
        ["2 3 1.5"], banner="%%MatrixMarket matrix coordinate real general"
    )

    with pytest.raises(
        GzipMatrixMarketValidationError,
        match="expected a finite nonnegative integral real",
    ):
        _reduce(io.BytesIO(compressed), allow_integral_real=True)


def test_real_field_is_opt_in() -> None:
    compressed = _payload(
        ["1 2 1.0"], banner="%%MatrixMarket matrix coordinate real general"
    )

    with pytest.raises(
        GzipMatrixMarketValidationError, match="require allow_integral_real=True"
    ):
        _reduce(io.BytesIO(compressed))


def test_real_field_enforces_exact_int64_bound() -> None:
    maximum = np.iinfo(np.int64).max
    accepted = _payload(
        [f"2 3 {maximum}.0"], banner="%%MatrixMarket matrix coordinate real general"
    )
    _, audit = _reduce(io.BytesIO(accepted), allow_integral_real=True)
    assert audit.global_value_sum == maximum

    rejected = _payload(
        ["2 3 9.223372036854775808e18"],
        banner="%%MatrixMarket matrix coordinate real general",
    )
    with pytest.raises(GzipMatrixMarketValidationError, match="exceeds int64"):
        _reduce(io.BytesIO(rejected), allow_integral_real=True)


def test_real_field_duplicate_accumulation_still_enforces_int64_bound() -> None:
    maximum = np.iinfo(np.int64).max
    compressed = _payload(
        [f"1 2 {maximum}.0", "1 2 1.0"],
        banner="%%MatrixMarket matrix coordinate real general",
    )

    with pytest.raises(GzipMatrixMarketValidationError, match="selected int64 cell"):
        _reduce(io.BytesIO(compressed), allow_integral_real=True)


def test_path_and_open_binary_stream_produce_identical_deterministic_audit(
    tmp_path: Path,
) -> None:
    compressed = _payload(["3 4 9", "1 2 2"])
    path = tmp_path / "matrix.mtx.gz"
    path.write_bytes(compressed)

    from_path = _reduce(path)
    stream = io.BytesIO(compressed)
    from_stream = _reduce(stream)

    np.testing.assert_array_equal(from_path[0], from_stream[0])
    assert asdict(from_path[1]) == asdict(from_stream[1])
    assert not stream.closed
    assert from_path[1].compressed_bytes == len(compressed)
    assert from_path[1].compressed_sha256 == hashlib.sha256(compressed).hexdigest()
    assert from_path[1].compressed_source_exhausted
    assert from_path[1].decompressed_sha256 == (
        "d5092ad03efe3b35b8609ad552229dab0ca7b80bb517f8cb977ce43035a8e664"
    )


def test_audit_binds_both_compressed_identity_and_decompressed_content() -> None:
    compressed = _payload(["1 2 3"])
    decompressed = gzip.decompress(compressed)

    _, audit = _reduce(io.BytesIO(compressed))

    assert audit.compressed_bytes == len(compressed)
    assert audit.compressed_sha256 == hashlib.sha256(compressed).hexdigest()
    assert audit.compressed_source_exhausted
    assert audit.decompressed_bytes == len(decompressed)
    assert audit.decompressed_sha256 == hashlib.sha256(decompressed).hexdigest()
    assert audit.decompressed_sha256 != hashlib.sha256(compressed).hexdigest()


def test_compressed_audit_distinguishes_encodings_of_identical_content() -> None:
    decompressed = gzip.decompress(_payload(["1 2 3"]))
    first = gzip.compress(decompressed, mtime=0)
    second = gzip.compress(decompressed, mtime=1)

    _, first_audit = _reduce(io.BytesIO(first))
    _, second_audit = _reduce(io.BytesIO(second))

    assert first_audit.decompressed_sha256 == second_audit.decompressed_sha256
    assert first_audit.compressed_sha256 != second_audit.compressed_sha256
    assert first_audit.compressed_sha256 == hashlib.sha256(first).hexdigest()
    assert second_audit.compressed_sha256 == hashlib.sha256(second).hexdigest()


def test_stream_audit_starts_at_the_callers_current_position() -> None:
    compressed = _payload(["1 2 3"])
    prefix = b"bytes outside the supplied stream position"
    stream = io.BytesIO(prefix + compressed)
    stream.seek(len(prefix))

    _, audit = _reduce(stream)

    assert audit.compressed_bytes == len(compressed)
    assert audit.compressed_sha256 == hashlib.sha256(compressed).hexdigest()
    assert stream.tell() == len(prefix) + len(compressed)


def test_selected_axis_order_controls_output_and_is_preserved_in_audit() -> None:
    compressed = _payload(["1 2 3", "1 5 5", "4 2 7", "4 5 11"])

    block, audit = _reduce(
        io.BytesIO(compressed),
        selected_rows=(1, 4),
        selected_columns=(2, 5),
    )

    np.testing.assert_array_equal(block, [[3, 5], [7, 11]])
    assert audit.selected_rows == (1, 4)
    assert audit.selected_columns == (2, 5)


def test_valid_matrix_can_have_no_entries_in_selected_block() -> None:
    block, audit = _reduce(io.BytesIO(_payload(["2 3 8", "3 4 1"])))

    np.testing.assert_array_equal(block, np.zeros((2, 2), dtype=np.int64))
    assert audit.selected_entries == 0
    assert audit.selected_distinct_coordinates == 0
    assert audit.selected_value_sum == 0
    assert audit.global_value_sum == 9


@pytest.mark.parametrize(
    ("compressed", "message"),
    [
        (
            _payload(
                ["1 1 1"],
                banner="%%matrixmarket matrix coordinate integer general",
            ),
            "banner must be exactly",
        ),
        (
            _payload(
                ["1 1 1"],
                banner="%%MatrixMarket matrix coordinate real symmetric",
            ),
            "banner must be exactly",
        ),
        (
            _payload(
                ["1 1 1"],
                banner="%%MatrixMarket matrix coordinate complex general",
            ),
            "banner must be exactly",
        ),
        (
            _payload(
                ["1 1 1"],
                banner="%%MatrixMarket matrix array real general",
            ),
            "banner must be exactly",
        ),
        (
            _payload(
                ["1 1 1"],
                banner="%%MatrixMarket matrix coordinate pattern general",
            ),
            "banner must be exactly",
        ),
        (_payload(["1 1 1"], shape=(5, 4)), "appear transposed"),
        (_payload(["1 1 1"], shape=(4, 6)), "do not match expected_shape"),
        (
            _raw_payload("%%MatrixMarket matrix coordinate integer general", "4 5"),
            "row count, column count, and nnz",
        ),
        (
            _raw_payload("%%MatrixMarket matrix coordinate integer general", "0 5 0"),
            "row count must be positive",
        ),
        (
            _raw_payload("%%MatrixMarket matrix coordinate integer general", "4 5 -1"),
            "invalid declared nnz",
        ),
        (_payload(["0 1 1"]), "row index must be positive"),
        (_payload(["5 1 1"]), "out-of-range coordinate"),
        (_payload(["1 6 1"]), "out-of-range coordinate"),
        (_payload(["1 1 -1"]), "invalid value"),
        (_payload(["1 1 1.0"]), "invalid value"),
        (_payload(["1 1 1e0"]), "invalid value"),
        (_payload(["1 1 1.5"]), "invalid value"),
        (_payload(["1 1 nan"]), "invalid value"),
        (_payload(["1 1 inf"]), "invalid value"),
        (_payload(["1 1 1"], declared_nnz=2), "declares 2 entries"),
        (
            _payload(["1 1 1", "2 2 2"], declared_nnz=1),
            "more entries than its declared nnz",
        ),
    ],
)
def test_rejects_malformed_banner_dimensions_indices_values_and_nnz(
    compressed: bytes, message: str
) -> None:
    with pytest.raises(GzipMatrixMarketValidationError, match=message):
        _reduce(io.BytesIO(compressed))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"expected_shape": (4,)}, "expected_shape"),
        ({"expected_shape": (4, 0)}, "expected column count"),
        ({"selected_rows": ()}, "must not be empty"),
        ({"selected_rows": (1, 1)}, "duplicate indices"),
        ({"selected_columns": (2, 2)}, "duplicate indices"),
        ({"selected_rows": (5,)}, "outside the expected matrix"),
        ({"selected_columns": (True,)}, "positive integer"),
    ],
)
def test_rejects_invalid_expected_shape_and_selection(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        _reduce(io.BytesIO(_payload([])), **overrides)


def test_rejects_selected_duplicate_accumulation_overflow() -> None:
    maximum = np.iinfo(np.int64).max
    compressed = _payload([f"1 2 {maximum}", "1 2 1"])

    with pytest.raises(GzipMatrixMarketValidationError, match="selected int64 cell"):
        _reduce(io.BytesIO(compressed))


def test_rejects_selected_total_overflow_across_distinct_cells() -> None:
    maximum = np.iinfo(np.int64).max
    compressed = _payload([f"4 5 {maximum}", "1 2 1"])

    with pytest.raises(GzipMatrixMarketValidationError, match="selected int64 total"):
        _reduce(io.BytesIO(compressed))


def test_rejects_unselected_global_accumulation_overflow() -> None:
    maximum = np.iinfo(np.int64).max
    compressed = _payload([f"2 3 {maximum}", "2 3 1"])

    with pytest.raises(GzipMatrixMarketValidationError, match="global int64 total"):
        _reduce(io.BytesIO(compressed))


def test_rejects_single_value_beyond_int64() -> None:
    compressed = _payload([f"2 3 {np.iinfo(np.int64).max + 1}"])

    with pytest.raises(GzipMatrixMarketValidationError, match="exceeds int64"):
        _reduce(io.BytesIO(compressed))


def test_rejects_truncated_gzip_even_after_complete_declared_matrix() -> None:
    compressed = _payload(["1 2 3"])
    truncated = compressed[:-3]

    with pytest.raises(
        GzipMatrixMarketValidationError, match="gzip stream failed"
    ) as caught:
        _reduce(io.BytesIO(truncated))

    partial = caught.value.partial_audit
    assert partial.compressed_bytes == len(truncated)
    assert partial.compressed_sha256 == hashlib.sha256(truncated).hexdigest()
    assert partial.compressed_source_exhausted
    assert not partial.gzip_stream_exhausted


@pytest.mark.parametrize(
    "suffix",
    [b"trailing bytes", gzip.compress(b"", mtime=0)],
    ids=["raw-trailing-data", "concatenated-gzip-member"],
)
def test_rejects_and_fully_identifies_content_after_the_single_member(
    suffix: bytes,
) -> None:
    combined = _payload(["1 2 3"]) + suffix

    with pytest.raises(
        GzipMatrixMarketValidationError,
        match="trailing data or concatenated members",
    ) as caught:
        _reduce(io.BytesIO(combined))

    partial = caught.value.partial_audit
    assert partial.compressed_bytes == len(combined)
    assert partial.compressed_sha256 == hashlib.sha256(combined).hexdigest()
    assert partial.compressed_source_exhausted
    assert not partial.gzip_stream_exhausted


def test_rejects_crc_corruption_with_complete_compressed_identity() -> None:
    original = _payload(["1 2 3"])
    compressed = bytearray(original)
    compressed[-8] ^= 1
    corrupted = bytes(compressed)

    with pytest.raises(
        GzipMatrixMarketValidationError, match="CRC check failed"
    ) as caught:
        _reduce(io.BytesIO(corrupted))

    partial = caught.value.partial_audit
    assert partial == GzipMatrixMarketPartialAudit(
        declared_nnz=None,
        parsed_nnz=0,
        compressed_bytes=len(corrupted),
        compressed_sha256=hashlib.sha256(corrupted).hexdigest(),
        compressed_source_exhausted=True,
        decompressed_bytes=0,
        decompressed_sha256=hashlib.sha256(b"").hexdigest(),
        gzip_stream_exhausted=False,
    )


def test_rejects_deterministic_corrupt_deflate_as_validation_failure() -> None:
    gzip_header = bytes.fromhex("1f8b08000000000000ff")
    invalid_reserved_block = b"\x07"
    invalid_stream = gzip_header + invalid_reserved_block + (b"\x00" * 8)

    with pytest.raises(
        GzipMatrixMarketValidationError, match="invalid block type"
    ) as caught:
        _reduce(io.BytesIO(invalid_stream))

    assert isinstance(caught.value.__cause__, zlib.error)
    assert caught.value.partial_audit == GzipMatrixMarketPartialAudit(
        declared_nnz=None,
        parsed_nnz=0,
        compressed_bytes=len(invalid_stream),
        compressed_sha256=hashlib.sha256(invalid_stream).hexdigest(),
        compressed_source_exhausted=True,
        decompressed_bytes=0,
        decompressed_sha256=hashlib.sha256(b"").hexdigest(),
        gzip_stream_exhausted=False,
    )


def test_parse_failure_retains_immutable_prefix_audit() -> None:
    prefix = (
        "%%MatrixMarket matrix coordinate integer general\n4 5 3\n1 2 7\nnot an entry\n"
    ).encode("ascii")
    full_content = prefix + b"3 4 9\n"
    compressed = gzip.compress(full_content, mtime=0)

    with pytest.raises(
        GzipMatrixMarketValidationError, match="invalid row index"
    ) as caught:
        _reduce(io.BytesIO(compressed))

    partial = caught.value.partial_audit
    assert partial == GzipMatrixMarketPartialAudit(
        declared_nnz=3,
        parsed_nnz=1,
        compressed_bytes=len(compressed),
        compressed_sha256=hashlib.sha256(compressed).hexdigest(),
        compressed_source_exhausted=True,
        decompressed_bytes=len(prefix),
        decompressed_sha256=hashlib.sha256(prefix).hexdigest(),
        gzip_stream_exhausted=False,
    )
    with pytest.raises(FrozenInstanceError):
        setattr(partial, "parsed_nnz", 2)


def test_early_parse_failure_hashes_exactly_the_compressed_prefix_read() -> None:
    suffix = b"".join(
        hashlib.sha256(index.to_bytes(4, "big")).digest() for index in range(4096)
    )
    compressed = gzip.compress(b"invalid banner\n" + suffix, mtime=0)

    class RecordingCappedStream(io.BytesIO):
        def __init__(self, payload: bytes) -> None:
            super().__init__(payload)
            self.consumed = bytearray()

        def read(self, size=-1):
            chunk = super().read(min(size, 4096))
            self.consumed.extend(chunk)
            return chunk

    stream = RecordingCappedStream(compressed)
    with pytest.raises(
        GzipMatrixMarketValidationError, match="banner must be exactly"
    ) as caught:
        _reduce(stream)

    partial = caught.value.partial_audit
    assert 0 < partial.compressed_bytes < len(compressed)
    assert partial.compressed_bytes == len(stream.consumed)
    assert partial.compressed_sha256 == hashlib.sha256(stream.consumed).hexdigest()
    assert not partial.compressed_source_exhausted
    assert not stream.closed


def test_declared_nnz_failure_retains_complete_content_as_partial_audit() -> None:
    compressed = _payload(["1 2 7"], declared_nnz=2)
    decompressed = gzip.decompress(compressed)

    with pytest.raises(
        GzipMatrixMarketValidationError, match="declares 2 entries"
    ) as caught:
        _reduce(io.BytesIO(compressed))

    assert caught.value.partial_audit == GzipMatrixMarketPartialAudit(
        declared_nnz=2,
        parsed_nnz=1,
        compressed_bytes=len(compressed),
        compressed_sha256=hashlib.sha256(compressed).hexdigest(),
        compressed_source_exhausted=True,
        decompressed_bytes=len(decompressed),
        decompressed_sha256=hashlib.sha256(decompressed).hexdigest(),
        gzip_stream_exhausted=True,
    )


def test_missing_path_and_generic_stream_io_errors_remain_distinct(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.mtx.gz"
    with pytest.raises(FileNotFoundError):
        _reduce(missing)

    expected = OSError("fixture source I/O failure")

    class FailingReadStream(io.BytesIO):
        def read(self, size=-1):
            raise expected

    stream = FailingReadStream(b"not read")
    with pytest.raises(OSError) as caught:
        _reduce(stream)

    assert caught.value is expected
    assert not isinstance(caught.value, GzipMatrixMarketValidationError)
    assert not stream.closed


def test_external_stream_remains_open_after_parser_validation_error() -> None:
    stream = io.BytesIO(_payload(["not an entry"]))

    with pytest.raises(GzipMatrixMarketValidationError):
        _reduce(stream)

    assert not stream.closed


def test_rejects_malformed_content_after_declared_entries() -> None:
    compressed = _payload(["1 2 3", "not an entry"], declared_nnz=1)

    with pytest.raises(GzipMatrixMarketValidationError, match="invalid row index"):
        _reduce(io.BytesIO(compressed))
