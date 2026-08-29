"""Strict, audited subset reads from Matrix Market files inside tar archives."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import gzip
import io
from pathlib import Path
import re
import tarfile
from typing import Iterator, Mapping, Sequence, TextIO

import numpy as np


_INDEX_TOKEN = re.compile(r"[1-9][0-9]*\Z")
_COUNT_TOKEN = re.compile(r"\+?[0-9]+\Z")


class MatrixMarketValidationError(ValueError):
    """Raised when an archive does not satisfy the strict reader contract."""


@dataclass(frozen=True)
class TarAxes:
    """Exact feature records, barcodes, and matrix dimensions from an archive."""

    features: tuple[tuple[str, ...], ...]
    barcodes: tuple[str, ...]
    matrix_shape: tuple[int, int]


@dataclass(frozen=True)
class MatrixMarketAccessAudit:
    """Counters proving which sparse values were inspected or converted."""

    declared_entries: int
    entries_seen: int
    authorized_column_entries: int
    unauthorized_column_entries: int
    authorized_column_unrequested_row_entries: int
    selected_entries_materialized: int
    value_tokens_lexically_validated: int
    value_tokens_converted: int
    unauthorized_value_tokens_converted: int
    row_major_monotone: bool
    column_major_monotone: bool


def _regular_member(archive: tarfile.TarFile, name: str) -> tarfile.TarInfo:
    matches = [member for member in archive.getmembers() if member.name == name]
    if len(matches) != 1:
        raise MatrixMarketValidationError(
            f"archive must contain exactly one member named {name!r}; found {len(matches)}"
        )
    if not matches[0].isfile():
        raise MatrixMarketValidationError(f"archive member {name!r} is not a regular file")
    return matches[0]


@contextmanager
def _text_member(
    archive: tarfile.TarFile, member: tarfile.TarInfo
) -> Iterator[TextIO]:
    raw = archive.extractfile(member)
    if raw is None:
        raise MatrixMarketValidationError(f"could not open archive member {member.name!r}")
    binary: io.BufferedIOBase | tarfile.ExFileObject
    if member.name.endswith(".gz"):
        binary = gzip.GzipFile(fileobj=raw)
    else:
        binary = raw
    text = io.TextIOWrapper(binary, encoding="utf-8", errors="strict", newline="")
    try:
        yield text
    finally:
        text.close()


def _read_label_member(
    archive: tarfile.TarFile,
    member_name: str,
    *,
    column: int,
    label: str,
) -> tuple[str, ...]:
    if column < 0:
        raise ValueError(f"{label} column must be nonnegative")
    member = _regular_member(archive, member_name)
    values: list[str] = []
    with _text_member(archive, member) as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\r\n").split("\t")
            if column >= len(fields) or not fields[column]:
                raise MatrixMarketValidationError(
                    f"{member_name}:{line_number} has no nonempty {label} column {column}"
                )
            values.append(fields[column])
    if not values:
        raise MatrixMarketValidationError(f"{member_name!r} contains no {label}s")
    if len(values) != len(set(values)):
        raise MatrixMarketValidationError(f"{member_name!r} contains duplicate {label}s")
    return tuple(values)


def _read_feature_records(
    archive: tarfile.TarFile, member_name: str
) -> tuple[tuple[str, ...], ...]:
    member = _regular_member(archive, member_name)
    records: list[tuple[str, ...]] = []
    with _text_member(archive, member) as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = tuple(line.rstrip("\r\n").split("\t"))
            if not fields[0]:
                raise MatrixMarketValidationError(
                    f"{member_name}:{line_number} has no feature identifier"
                )
            records.append(fields)
    if not records:
        raise MatrixMarketValidationError(f"{member_name!r} contains no features")
    if len(records) != len(set(records)) or len({row[0] for row in records}) != len(
        records
    ):
        raise MatrixMarketValidationError(f"{member_name!r} contains duplicate features")
    return tuple(records)


def _exact_names(names: Sequence[str], *, label: str) -> tuple[str, ...]:
    values = tuple(names)
    if not values:
        raise ValueError(f"{label} must not be empty")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{label} must contain nonempty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicates")
    return values


def _parse_index(token: str, *, label: str, line_number: int) -> int:
    if _INDEX_TOKEN.fullmatch(token) is None:
        raise MatrixMarketValidationError(
            f"matrix line {line_number} has invalid {label} index {token!r}"
        )
    return int(token)


def _parse_dimensions(tokens: list[str], *, line_number: int) -> tuple[int, int, int]:
    if len(tokens) != 3 or any(_COUNT_TOKEN.fullmatch(token) is None for token in tokens):
        raise MatrixMarketValidationError(
            f"matrix line {line_number} must contain three nonnegative dimensions"
        )
    rows, columns, entries = (int(token) for token in tokens)
    if rows == 0 or columns == 0:
        raise MatrixMarketValidationError("matrix dimensions must be positive")
    return rows, columns, entries


def _read_matrix_preamble(handle: TextIO) -> tuple[int, int, int, int]:
    header = handle.readline()
    if not header:
        raise MatrixMarketValidationError("matrix member is empty")
    if header.strip().lower() != "%%matrixmarket matrix coordinate integer general":
        raise MatrixMarketValidationError(
            "matrix header must be '%%MatrixMarket matrix coordinate integer general'"
        )
    line_number = 1
    for line in handle:
        line_number += 1
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        rows, columns, entries = _parse_dimensions(
            stripped.split(), line_number=line_number
        )
        return rows, columns, entries, line_number
    raise MatrixMarketValidationError("matrix dimensions are missing")


def read_tar_axes(
    archive_path: str | Path,
    matrix_member: str,
    genes_member: str,
    barcodes_member: str,
) -> TarAxes:
    """Read exact axes and the Matrix Market preamble without reading matrix entries."""

    with tarfile.open(archive_path, mode="r:gz") as archive:
        features = _read_feature_records(archive, genes_member)
        barcodes = _read_label_member(
            archive, barcodes_member, column=0, label="barcode"
        )
        member = _regular_member(archive, matrix_member)
        with _text_member(archive, member) as handle:
            rows, columns, _, _ = _read_matrix_preamble(handle)
    if rows != len(features) or columns != len(barcodes):
        raise MatrixMarketValidationError(
            "matrix dimensions do not exactly match feature and barcode members"
        )
    return TarAxes(features=features, barcodes=barcodes, matrix_shape=(rows, columns))


def _resolve_requested_rows(
    axes: TarAxes, requested_rows: Mapping[str, int]
) -> tuple[tuple[str, ...], dict[int, int]]:
    if not requested_rows:
        raise ValueError("requested_rows must not be empty")
    names: list[str] = []
    selected: dict[int, int] = {}
    feature_count = len(axes.features)
    for output_index, (name, supplied_index) in enumerate(requested_rows.items()):
        if not isinstance(name, str) or not name:
            raise ValueError("requested_rows keys must be nonempty strings")
        if isinstance(supplied_index, bool) or not isinstance(supplied_index, int):
            raise ValueError("requested_rows indices must be integers")
        candidates: list[int] = []
        if 0 <= supplied_index < feature_count and name in axes.features[supplied_index]:
            candidates.append(supplied_index + 1)
        if 1 <= supplied_index <= feature_count and name in axes.features[supplied_index - 1]:
            candidates.append(supplied_index)
        candidates = list(dict.fromkeys(candidates))
        if len(candidates) != 1:
            raise MatrixMarketValidationError(
                f"requested feature {name!r} does not resolve uniquely at index "
                f"{supplied_index} as either zero- or one-based"
            )
        source_index = candidates[0]
        if source_index in selected:
            raise ValueError("requested_rows resolves duplicate feature rows")
        names.append(name)
        selected[source_index] = output_index
    return tuple(names), selected


def _validate_axes(axes: TarAxes) -> None:
    if len(axes.matrix_shape) != 2 or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in axes.matrix_shape
    ):
        raise ValueError("axes.matrix_shape must contain two positive integers")
    if len(axes.features) != axes.matrix_shape[0]:
        raise MatrixMarketValidationError("feature coverage does not match matrix shape")
    if len(axes.barcodes) != axes.matrix_shape[1]:
        raise MatrixMarketValidationError("barcode coverage does not match matrix shape")
    if any(not row or not row[0] for row in axes.features):
        raise MatrixMarketValidationError("axes contain an empty feature identifier")
    if len(axes.features) != len(set(axes.features)) or len(
        {row[0] for row in axes.features}
    ) != len(axes.features):
        raise MatrixMarketValidationError("axes contain duplicate features")
    if any(not isinstance(barcode, str) or not barcode for barcode in axes.barcodes):
        raise MatrixMarketValidationError("axes contain an empty barcode")
    if len(axes.barcodes) != len(set(axes.barcodes)):
        raise MatrixMarketValidationError("axes contain duplicate barcodes")


def read_tar_matrix_subset(
    archive_path: str | Path,
    axes: TarAxes,
    matrix_member: str,
    requested_rows: Mapping[str, int],
    authorized_barcodes: Sequence[str],
) -> tuple[np.ndarray, MatrixMarketAccessAudit]:
    """Return authorized cells x requested rows in mapping insertion order."""

    _validate_axes(axes)
    _, selected_row_output = _resolve_requested_rows(axes, requested_rows)
    column_names = _exact_names(authorized_barcodes, label="authorized_barcodes")
    barcode_lookup = {name: index + 1 for index, name in enumerate(axes.barcodes)}
    missing_columns = sorted(set(column_names) - barcode_lookup.keys())
    if missing_columns:
        raise MatrixMarketValidationError(
            f"authorized barcodes absent from archive: {missing_columns!r}"
        )
    selected_column_output = {
        barcode_lookup[name]: output_index for output_index, name in enumerate(column_names)
    }
    counts = np.zeros((len(column_names), len(requested_rows)), dtype=np.int64)

    with tarfile.open(archive_path, mode="r:gz") as archive:
        member = _regular_member(archive, matrix_member)
        with _text_member(archive, member) as handle:
            source_rows, source_columns, declared_entries, line_number = (
                _read_matrix_preamble(handle)
            )
            if (source_rows, source_columns) != axes.matrix_shape:
                raise MatrixMarketValidationError(
                    "matrix dimensions do not exactly match the supplied axes"
                )

            entries_seen = 0
            authorized_entries = 0
            unauthorized_entries = 0
            authorized_unrequested = 0
            selected_entries = 0
            converted_values = 0
            previous: tuple[int, int] | None = None
            row_major_monotone = True
            column_major_monotone = True

            for line in handle:
                line_number += 1
                stripped = line.strip()
                if not stripped or stripped.startswith("%"):
                    continue
                tokens = stripped.split()
                if len(tokens) != 3:
                    raise MatrixMarketValidationError(
                        f"matrix line {line_number} must contain row, column, and value"
                    )
                row = _parse_index(tokens[0], label="row", line_number=line_number)
                column = _parse_index(tokens[1], label="column", line_number=line_number)
                value_token = tokens[2]
                if row > source_rows or column > source_columns:
                    raise MatrixMarketValidationError(
                        f"matrix line {line_number} has an out-of-range coordinate"
                    )
                if _COUNT_TOKEN.fullmatch(value_token) is None:
                    raise MatrixMarketValidationError(
                        f"matrix line {line_number} has a non-integer or negative count"
                    )

                coordinate = (row, column)
                if previous is not None:
                    if coordinate == previous:
                        raise MatrixMarketValidationError(
                            f"matrix line {line_number} duplicates a coordinate"
                        )
                    row_major_monotone &= coordinate > previous
                    column_major_monotone &= (column, row) > (previous[1], previous[0])
                previous = coordinate
                entries_seen += 1

                output_cell = selected_column_output.get(column)
                if output_cell is None:
                    unauthorized_entries += 1
                    continue
                authorized_entries += 1
                output_row = selected_row_output.get(row)
                if output_row is None:
                    authorized_unrequested += 1
                    continue
                value = int(value_token)
                if value > np.iinfo(np.int64).max:
                    raise MatrixMarketValidationError(
                        f"matrix line {line_number} count exceeds int64"
                    )
                counts[output_cell, output_row] = value
                converted_values += 1
                selected_entries += 1

    if entries_seen != declared_entries:
        raise MatrixMarketValidationError(
            f"matrix declares {declared_entries} entries but contains {entries_seen}"
        )
    if not (row_major_monotone or column_major_monotone):
        raise MatrixMarketValidationError(
            "matrix coordinates are not monotone; exact duplicate validation refused"
        )
    audit = MatrixMarketAccessAudit(
        declared_entries=declared_entries,
        entries_seen=entries_seen,
        authorized_column_entries=authorized_entries,
        unauthorized_column_entries=unauthorized_entries,
        authorized_column_unrequested_row_entries=authorized_unrequested,
        selected_entries_materialized=selected_entries,
        value_tokens_lexically_validated=entries_seen,
        value_tokens_converted=converted_values,
        unauthorized_value_tokens_converted=0,
        row_major_monotone=row_major_monotone,
        column_major_monotone=column_major_monotone,
    )
    return counts, audit
