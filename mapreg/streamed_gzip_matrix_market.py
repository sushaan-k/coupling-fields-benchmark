"""Bounded-memory reduction of gzip-compressed Matrix Market count matrices."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import io
from numbers import Integral
from pathlib import Path
import re
from typing import BinaryIO, Iterator, Sequence
import zlib

import numpy as np


_INTEGER_BANNER = b"%%MatrixMarket matrix coordinate integer general"
_REAL_BANNER = b"%%MatrixMarket matrix coordinate real general"
_ALLOWED_BANNERS = frozenset((_INTEGER_BANNER, _REAL_BANNER))
_UNSIGNED_INTEGER = re.compile(rb"\+?[0-9]+\Z")
_REAL_NUMBER = re.compile(
    rb"\+?(?:(?:[0-9]+(?:\.[0-9]*)?)|(?:\.[0-9]+))(?:[eE][+-]?[0-9]+)?\Z"
)
_INT64_MAX = int(np.iinfo(np.int64).max)
_MAX_LINE_BYTES = 1_048_576
_COMPRESSED_CHUNK_BYTES = 65_536


@dataclass(frozen=True)
class GzipMatrixMarketPartialAudit:
    """Immutable parser state retained when complete validation fails."""

    declared_nnz: int | None
    parsed_nnz: int
    compressed_bytes: int
    compressed_sha256: str
    compressed_source_exhausted: bool
    decompressed_bytes: int
    decompressed_sha256: str
    gzip_stream_exhausted: bool


class GzipMatrixMarketValidationError(ValueError):
    """Raised when a gzip Matrix Market stream violates the reader contract."""

    def __init__(
        self,
        message: str,
        *,
        partial_audit: GzipMatrixMarketPartialAudit | None = None,
    ) -> None:
        super().__init__(message)
        self.partial_audit = partial_audit


@dataclass(frozen=True)
class GzipMatrixMarketAudit:
    """Evidence binding validated matrix content to one complete gzip stream."""

    banner: str
    matrix_shape: tuple[int, int]
    selected_rows: tuple[int, ...]
    selected_columns: tuple[int, ...]
    declared_nnz: int
    parsed_nnz: int
    comment_lines: int
    blank_lines: int
    zero_value_entries: int
    selected_entries: int
    selected_distinct_coordinates: int
    selected_duplicate_entries: int
    global_value_sum: int
    selected_value_sum: int
    compressed_bytes: int
    compressed_sha256: str
    compressed_source_exhausted: bool
    decompressed_bytes: int
    decompressed_sha256: str
    gzip_stream_exhausted: bool
    output_dtype: str


class _DecompressedState:
    def __init__(self) -> None:
        self.bytes_read = 0
        self._digest = hashlib.sha256()

    def readline(self, handle: BinaryIO, line_number: int) -> bytes:
        line = handle.readline(_MAX_LINE_BYTES + 1)
        self.bytes_read += len(line)
        self._digest.update(line)
        if len(line) > _MAX_LINE_BYTES:
            raise GzipMatrixMarketValidationError(
                f"matrix line {line_number} exceeds {_MAX_LINE_BYTES} bytes"
            )
        return line

    def hexdigest(self) -> str:
        return self._digest.copy().hexdigest()


class _CompressedState:
    def __init__(self) -> None:
        self.bytes_read = 0
        self.source_exhausted = False
        self._digest = hashlib.sha256()

    def read(self, handle: BinaryIO) -> bytes:
        chunk = handle.read(_COMPRESSED_CHUNK_BYTES)
        if chunk:
            self.bytes_read += len(chunk)
            self._digest.update(chunk)
        else:
            self.source_exhausted = True
        return chunk

    def hexdigest(self) -> str:
        return self._digest.copy().hexdigest()


class _StrictGzipReader(io.RawIOBase):
    """Stream one gzip member and reject every byte outside that member."""

    def __init__(self, source: BinaryIO, state: _CompressedState) -> None:
        super().__init__()
        self._source = source
        self._state = state
        self._decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
        self._compressed_pending = b""
        self._output = bytearray()
        self._validated = False

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray | memoryview) -> int:
        view = memoryview(buffer).cast("B")
        written = 0
        while written < len(view):
            if self._output:
                take = min(len(self._output), len(view) - written)
                view[written : written + take] = self._output[:take]
                del self._output[:take]
                written += take
                continue
            if self._validated:
                break
            self._decompress_more(len(view) - written)
        return written

    def _drain_source(self) -> bool:
        trailing = False
        while not self._state.source_exhausted:
            trailing |= bool(self._state.read(self._source))
        return trailing

    def _decompress_more(self, maximum_output: int) -> None:
        while not self._output and not self._validated:
            if self._compressed_pending:
                compressed = self._compressed_pending
                self._compressed_pending = b""
            else:
                compressed = self._state.read(self._source)
                if not compressed:
                    raise GzipMatrixMarketValidationError(
                        "gzip stream failed validation: compressed member is truncated"
                    )

            try:
                output = self._decompressor.decompress(compressed, maximum_output)
            except zlib.error:
                self._drain_source()
                raise
            self._output.extend(output)
            self._compressed_pending = self._decompressor.unconsumed_tail

            if self._decompressor.eof:
                trailing = bool(
                    self._decompressor.unused_data or self._compressed_pending
                )
                trailing |= self._drain_source()
                if trailing:
                    raise GzipMatrixMarketValidationError(
                        "gzip stream failed validation: trailing data or "
                        "concatenated members are not allowed"
                    )
                self._validated = True


def _partial_audit(
    decompressed: _DecompressedState,
    compressed: _CompressedState,
    *,
    declared_nnz: int | None,
    parsed_nnz: int,
    gzip_stream_exhausted: bool = False,
) -> GzipMatrixMarketPartialAudit:
    return GzipMatrixMarketPartialAudit(
        declared_nnz=declared_nnz,
        parsed_nnz=parsed_nnz,
        compressed_bytes=compressed.bytes_read,
        compressed_sha256=compressed.hexdigest(),
        compressed_source_exhausted=compressed.source_exhausted,
        decompressed_bytes=decompressed.bytes_read,
        decompressed_sha256=decompressed.hexdigest(),
        gzip_stream_exhausted=gzip_stream_exhausted,
    )


def _content(line: bytes) -> bytes:
    if line.endswith(b"\n"):
        line = line[:-1]
        if line.endswith(b"\r"):
            line = line[:-1]
    return line


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be a positive integer")
    parsed = int(value)
    if parsed <= 0 or parsed > _INT64_MAX:
        raise ValueError(f"{label} must be between 1 and int64 maximum")
    return parsed


def _expected_shape(expected_shape: Sequence[int]) -> tuple[int, int]:
    try:
        values = tuple(expected_shape)
    except TypeError as error:
        raise ValueError("expected_shape must contain two positive integers") from error
    if len(values) != 2:
        raise ValueError("expected_shape must contain two positive integers")
    return (
        _positive_int(values[0], label="expected row count"),
        _positive_int(values[1], label="expected column count"),
    )


def _selected_indices(
    values: Sequence[int], *, label: str, upper_bound: int
) -> tuple[int, ...]:
    try:
        supplied = tuple(values)
    except TypeError as error:
        raise ValueError(f"{label} must be a sequence of 1-based indices") from error
    if not supplied:
        raise ValueError(f"{label} must not be empty")
    selected = tuple(_positive_int(value, label=label) for value in supplied)
    if any(value > upper_bound for value in selected):
        raise ValueError(f"{label} contains an index outside the expected matrix")
    if len(selected) != len(set(selected)):
        raise ValueError(f"{label} contains duplicate indices")
    return selected


def _ascii_tokens(content: bytes, *, line_number: int) -> list[bytes]:
    try:
        content.decode("ascii")
    except UnicodeDecodeError as error:
        raise GzipMatrixMarketValidationError(
            f"matrix line {line_number} is not ASCII"
        ) from error
    return content.split()


def _uint64_token(
    token: bytes,
    *,
    label: str,
    line_number: int,
    positive: bool,
) -> int:
    if _UNSIGNED_INTEGER.fullmatch(token) is None:
        qualifier = "positive" if positive else "nonnegative"
        raise GzipMatrixMarketValidationError(
            f"matrix line {line_number} has invalid {label}; expected a {qualifier} integer"
        )
    digits = token[1:] if token.startswith(b"+") else token
    normalized = digits.lstrip(b"0") or b"0"
    maximum = str(_INT64_MAX).encode("ascii")
    if len(normalized) > len(maximum) or (
        len(normalized) == len(maximum) and normalized > maximum
    ):
        raise GzipMatrixMarketValidationError(
            f"matrix line {line_number} {label} exceeds int64 maximum"
        )
    value = int(normalized)
    if positive and value == 0:
        raise GzipMatrixMarketValidationError(
            f"matrix line {line_number} {label} must be positive"
        )
    return value


def _integral_real_token(token: bytes, *, line_number: int) -> int:
    if _REAL_NUMBER.fullmatch(token) is None:
        raise GzipMatrixMarketValidationError(
            f"matrix line {line_number} has invalid value; "
            "expected a finite nonnegative integral real"
        )
    try:
        parsed = Decimal(token.decode("ascii"))
    except InvalidOperation as error:
        raise GzipMatrixMarketValidationError(
            f"matrix line {line_number} has invalid value; "
            "expected a finite nonnegative integral real"
        ) from error
    if not parsed.is_finite() or parsed < 0 or parsed != parsed.to_integral_value():
        raise GzipMatrixMarketValidationError(
            f"matrix line {line_number} has invalid value; "
            "expected a finite nonnegative integral real"
        )
    if parsed > _INT64_MAX:
        raise GzipMatrixMarketValidationError(
            f"matrix line {line_number} value exceeds int64 maximum"
        )
    return int(parsed)


def _parse_dimensions(content: bytes, *, line_number: int) -> tuple[int, int, int]:
    tokens = _ascii_tokens(content, line_number=line_number)
    if len(tokens) != 3:
        raise GzipMatrixMarketValidationError(
            f"matrix line {line_number} must contain row count, column count, and nnz"
        )
    return (
        _uint64_token(
            tokens[0], label="row count", line_number=line_number, positive=True
        ),
        _uint64_token(
            tokens[1], label="column count", line_number=line_number, positive=True
        ),
        _uint64_token(
            tokens[2], label="declared nnz", line_number=line_number, positive=False
        ),
    )


def _parse_entry(
    content: bytes,
    *,
    line_number: int,
    shape: tuple[int, int],
    real_field: bool,
) -> tuple[int, int, int]:
    tokens = _ascii_tokens(content, line_number=line_number)
    if len(tokens) != 3:
        raise GzipMatrixMarketValidationError(
            f"matrix line {line_number} must contain row, column, and value"
        )
    row = _uint64_token(
        tokens[0], label="row index", line_number=line_number, positive=True
    )
    column = _uint64_token(
        tokens[1], label="column index", line_number=line_number, positive=True
    )
    if real_field:
        value = _integral_real_token(tokens[2], line_number=line_number)
    else:
        value = _uint64_token(
            tokens[2], label="value", line_number=line_number, positive=False
        )
    if row > shape[0] or column > shape[1]:
        raise GzipMatrixMarketValidationError(
            f"matrix line {line_number} has an out-of-range coordinate"
        )
    return row, column, value


@contextmanager
def _binary_source(source: str | Path | BinaryIO) -> Iterator[BinaryIO]:
    if isinstance(source, (str, Path)):
        with Path(source).open("rb") as handle:
            yield handle
        return
    if not hasattr(source, "read"):
        raise TypeError("source must be a local path or binary stream")
    yield source


def reduce_gzip_matrix_market(
    source: str | Path | BinaryIO,
    *,
    expected_shape: Sequence[int],
    selected_rows: Sequence[int],
    selected_columns: Sequence[int],
    allow_integral_real: bool = False,
) -> tuple[np.ndarray, GzipMatrixMarketAudit]:
    """Validate a complete ``.mtx.gz`` stream and materialize one selected block.

    Source coordinates and selected indices are 1-based. Output rows and columns
    follow the supplied selection order. Duplicate coordinates are accumulated.
    Real-field input is refused unless explicitly enabled, and even then every
    stored value must encode an exact nonnegative integer.
    """

    shape = _expected_shape(expected_shape)
    rows = _selected_indices(selected_rows, label="selected_rows", upper_bound=shape[0])
    columns = _selected_indices(
        selected_columns, label="selected_columns", upper_bound=shape[1]
    )
    row_output = {source_index: index for index, source_index in enumerate(rows)}
    column_output = {source_index: index for index, source_index in enumerate(columns)}
    block = np.zeros((len(rows), len(columns)), dtype=np.int64)
    selected_seen = np.zeros(block.shape, dtype=bool)

    compressed_state = _CompressedState()
    decompressed = _DecompressedState()
    comment_lines = 0
    blank_lines = 0
    line_number = 1
    declared_nnz: int | None = None
    parsed_nnz = 0
    zero_entries = 0
    selected_entries = 0
    selected_total = 0
    global_total = 0
    banner: bytes | None = None

    try:
        with _binary_source(source) as compressed:
            raw_matrix = _StrictGzipReader(compressed, compressed_state)
            with io.BufferedReader(raw_matrix) as matrix:
                banner_line = decompressed.readline(matrix, line_number)
                if not banner_line:
                    raise GzipMatrixMarketValidationError("matrix is empty")
                banner = _content(banner_line)
                if banner not in _ALLOWED_BANNERS:
                    raise GzipMatrixMarketValidationError(
                        "matrix banner must be exactly either "
                        "'%%MatrixMarket matrix coordinate integer general' or "
                        "'%%MatrixMarket matrix coordinate real general'"
                    )
                if banner == _REAL_BANNER and not allow_integral_real:
                    raise GzipMatrixMarketValidationError(
                        "real-field matrices require allow_integral_real=True"
                    )
                real_field = banner == _REAL_BANNER

                matrix_shape: tuple[int, int] | None = None
                while True:
                    line_number += 1
                    line = decompressed.readline(matrix, line_number)
                    if not line:
                        break
                    content = _content(line)
                    if not content.strip():
                        blank_lines += 1
                        continue
                    if content.startswith(b"%"):
                        _ascii_tokens(content, line_number=line_number)
                        comment_lines += 1
                        continue
                    matrix_rows, matrix_columns, declared_nnz = _parse_dimensions(
                        content, line_number=line_number
                    )
                    matrix_shape = (matrix_rows, matrix_columns)
                    break

                if matrix_shape is None or declared_nnz is None:
                    raise GzipMatrixMarketValidationError(
                        "matrix dimension line is missing"
                    )
                if matrix_shape != shape:
                    if shape[0] != shape[1] and matrix_shape == (shape[1], shape[0]):
                        raise GzipMatrixMarketValidationError(
                            "matrix dimensions appear transposed relative to expected_shape"
                        )
                    raise GzipMatrixMarketValidationError(
                        f"matrix dimensions {matrix_shape} do not match expected_shape {shape}"
                    )

                while True:
                    line_number += 1
                    line = decompressed.readline(matrix, line_number)
                    if not line:
                        break
                    content = _content(line)
                    if not content.strip():
                        blank_lines += 1
                        continue
                    if content.startswith(b"%"):
                        _ascii_tokens(content, line_number=line_number)
                        comment_lines += 1
                        continue

                    row, column, value = _parse_entry(
                        content,
                        line_number=line_number,
                        shape=shape,
                        real_field=real_field,
                    )
                    parsed_nnz += 1
                    if parsed_nnz > declared_nnz:
                        raise GzipMatrixMarketValidationError(
                            "matrix contains more entries than its declared nnz"
                        )
                    zero_entries += value == 0

                    output_row = row_output.get(row)
                    output_column = column_output.get(column)
                    if output_row is not None and output_column is not None:
                        current = int(block[output_row, output_column])
                        if value > _INT64_MAX - current:
                            raise GzipMatrixMarketValidationError(
                                f"matrix line {line_number} overflows a selected int64 cell"
                            )
                        if value > _INT64_MAX - selected_total:
                            raise GzipMatrixMarketValidationError(
                                f"matrix line {line_number} overflows selected int64 total"
                            )
                        block[output_row, output_column] = current + value
                        selected_seen[output_row, output_column] = True
                        selected_total += value
                        selected_entries += 1

                    if value > _INT64_MAX - global_total:
                        raise GzipMatrixMarketValidationError(
                            f"matrix line {line_number} overflows global int64 total"
                        )
                    global_total += value
    except GzipMatrixMarketValidationError as error:
        if error.partial_audit is not None:
            raise
        raise GzipMatrixMarketValidationError(
            str(error),
            partial_audit=_partial_audit(
                decompressed,
                compressed_state,
                declared_nnz=declared_nnz,
                parsed_nnz=parsed_nnz,
            ),
        ) from error
    except zlib.error as error:
        detail = (
            "CRC check failed" if "incorrect data check" in str(error) else str(error)
        )
        raise GzipMatrixMarketValidationError(
            f"gzip stream failed validation: {detail}",
            partial_audit=_partial_audit(
                decompressed,
                compressed_state,
                declared_nnz=declared_nnz,
                parsed_nnz=parsed_nnz,
            ),
        ) from error

    if declared_nnz is None:
        raise AssertionError("declared_nnz was not assigned")
    if banner is None:
        raise AssertionError("banner was not assigned")
    if parsed_nnz != declared_nnz:
        raise GzipMatrixMarketValidationError(
            f"matrix declares {declared_nnz} entries but contains {parsed_nnz}",
            partial_audit=_partial_audit(
                decompressed,
                compressed_state,
                declared_nnz=declared_nnz,
                parsed_nnz=parsed_nnz,
                gzip_stream_exhausted=True,
            ),
        )

    distinct_selected = int(selected_seen.sum())
    audit = GzipMatrixMarketAudit(
        banner=banner.decode("ascii"),
        matrix_shape=shape,
        selected_rows=rows,
        selected_columns=columns,
        declared_nnz=declared_nnz,
        parsed_nnz=parsed_nnz,
        comment_lines=comment_lines,
        blank_lines=blank_lines,
        zero_value_entries=zero_entries,
        selected_entries=selected_entries,
        selected_distinct_coordinates=distinct_selected,
        selected_duplicate_entries=selected_entries - distinct_selected,
        global_value_sum=global_total,
        selected_value_sum=selected_total,
        compressed_bytes=compressed_state.bytes_read,
        compressed_sha256=compressed_state.hexdigest(),
        compressed_source_exhausted=compressed_state.source_exhausted,
        decompressed_bytes=decompressed.bytes_read,
        decompressed_sha256=decompressed.hexdigest(),
        gzip_stream_exhausted=True,
        output_dtype=str(block.dtype),
    )
    return block, audit
