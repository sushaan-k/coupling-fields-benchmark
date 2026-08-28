"""Reduce the six sealed GSE164378 5-prime RNA-ADT source files."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite


REQUIRED_SOURCE_FILES = (
    "GSM5008740_RNA_5P-barcodes.tsv.gz",
    "GSM5008740_RNA_5P-features.tsv.gz",
    "GSM5008740_RNA_5P-matrix.mtx.gz",
    "GSM5008741_ADT_5P-barcodes.tsv.gz",
    "GSM5008741_ADT_5P-features.tsv.gz",
    "GSM5008741_ADT_5P-matrix.mtx.gz",
)
OUTPUTS = (
    "adt_all.mtx.gz",
    "adt_features.tsv",
    "cells.tsv.gz",
    "markers.tsv",
    "rna_matched.mtx.gz",
    "source_acquisition.json",
)
DEVELOPMENT = ("P4", "P7", "P8", "P1")
HELD = ("P5", "P3", "P2")
LINEAGES = ("B", "CD4 T", "CD8 T", "Mono", "NK")
TIME_MAP = {0: 0, 2: 3, 7: 7}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


class _DigestReader:
    def __init__(self, stream: io.BufferedReader) -> None:
        self.stream = stream
        self.digest = hashlib.sha256()
        self.bytes = 0

    def read(self, size: int = -1) -> bytes:
        value = self.stream.read(size)
        self.digest.update(value)
        self.bytes += len(value)
        return value

    def readable(self) -> bool:
        return True


def _gzip_stream(path: Path) -> tuple[gzip.GzipFile, _DigestReader]:
    digest = _DigestReader(path.open("rb"))
    return gzip.GzipFile(fileobj=digest, mode="rb"), digest


def _read_tsv_file(path: Path) -> tuple[list[list[str]], dict[str, object]]:
    compressed, digest = _gzip_stream(path)
    with compressed:
        rows = [
            line.decode("utf-8").rstrip("\r\n").split("\t")
            for line in compressed
        ]
    if digest.bytes != path.stat().st_size:
        raise ValueError(f"compressed file byte count differs: {path.name}")
    return rows, {
        "name": path.name,
        "bytes": digest.bytes,
        "sha256": digest.digest.hexdigest(),
    }


def _read_matrix_subset(
    path: Path,
    row_lookup: dict[int, int],
    column_lookup: dict[int, int],
    *,
    expected_rows: int,
    expected_columns: int,
    collect_column_totals: bool,
) -> tuple[sparse.csc_matrix, np.ndarray | None, dict[str, object]]:
    compressed, digest = _gzip_stream(path)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    dimensions: tuple[int, int, int] | None = None
    observed = 0
    totals = np.zeros(len(column_lookup), dtype=float) if collect_column_totals else None
    with compressed:
        header = compressed.readline().decode("ascii").strip()
        if header != "%%MatrixMarket matrix coordinate integer general":
            raise ValueError(f"unsupported Matrix Market header in {path.name}")
        for raw in compressed:
            line = raw.decode("ascii").strip()
            if not line or line.startswith("%"):
                continue
            dimensions = tuple(int(token) for token in line.split())
            break
        if dimensions is None or len(dimensions) != 3:
            raise ValueError(f"missing Matrix Market dimensions in {path.name}")
        source_rows, source_columns, declared = dimensions
        if (
            source_rows != expected_rows
            or source_columns != expected_columns
            or declared < 0
        ):
            raise ValueError(
                f"Matrix Market dimensions differ from frozen axes in {path.name}"
            )
        if any(index < 0 or index >= source_rows for index in row_lookup):
            raise ValueError(f"selected row is outside {path.name}")
        if any(index < 0 or index >= source_columns for index in column_lookup):
            raise ValueError(f"selected column is outside {path.name}")
        for raw in compressed:
            line = raw.decode("ascii").strip()
            if not line:
                continue
            source_row, source_column, value = line.split()
            source_row_index = int(source_row) - 1
            source_column_index = int(source_column) - 1
            numeric = int(value)
            if (
                source_row_index < 0
                or source_row_index >= source_rows
                or source_column_index < 0
                or source_column_index >= source_columns
                or numeric < 0
            ):
                raise ValueError(f"invalid Matrix Market entry in {path.name}")
            observed += 1
            target_column = column_lookup.get(source_column_index)
            if target_column is None:
                continue
            if totals is not None:
                totals[target_column] += numeric
            target_row = row_lookup.get(source_row_index)
            if target_row is not None:
                rows.append(target_row)
                columns.append(target_column)
                values.append(numeric)
    if observed != declared:
        raise ValueError(f"Matrix Market nonzero count differs in {path.name}")
    if digest.bytes != path.stat().st_size:
        raise ValueError(f"compressed file byte count differs: {path.name}")
    matrix = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(len(row_lookup), len(column_lookup)),
        dtype=float,
    ).tocsc()
    return matrix, totals, {
        "name": path.name,
        "bytes": digest.bytes,
        "sha256": digest.digest.hexdigest(),
    }


def _write_gzip_text(path: Path, value: str) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(value.encode("utf-8"))


def _write_mtx_gzip(path: Path, matrix: sparse.spmatrix) -> None:
    with tempfile.NamedTemporaryFile(suffix=".mtx", delete=False) as handle:
        plain = Path(handle.name)
    try:
        mmwrite(plain, matrix, field="integer", symmetry="general")
        with plain.open("rb") as source, path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as target:
                shutil.copyfileobj(source, target, length=8 << 20)
    finally:
        plain.unlink(missing_ok=True)


def _source_record(manifest: dict, name: str) -> dict:
    return next(record for record in manifest["files"] if record["name"] == name)


def _feature_name(row: list[str]) -> str:
    nonempty = [value for value in row if value]
    if not nonempty:
        raise ValueError("empty feature row")
    return nonempty[1] if len(nonempty) >= 2 else nonempty[0]


def reduce(
    *,
    rna_barcodes_path: Path,
    rna_features_path: Path,
    rna_matrix_path: Path,
    adt_barcodes_path: Path,
    adt_features_path: Path,
    adt_matrix_path: Path,
    metadata_path: Path,
    aliases_path: Path,
    source_path: Path,
    output: Path,
) -> None:
    if output.exists():
        raise FileExistsError(f"reducer output path already exists: {output}")
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"reducer temporary path already exists: {temporary}")

    source = _read_json(source_path)
    if source.get("schema") != "hao-gse164378-source/1.0":
        raise ValueError("source manifest schema differs")
    metadata_record = _source_record(source, "GSE164378_sc.meta.data_5P.csv.gz")
    if (
        metadata_path.stat().st_size != metadata_record["bytes"]
        or _sha256(metadata_path) != metadata_record["sha256"]
    ):
        raise ValueError("metadata integrity differs from the frozen source manifest")

    aliases = pd.read_csv(aliases_path, sep="\t", dtype=str)
    if list(aliases.columns) != ["adt_feature", "gene_symbol"] or len(aliases) != 33:
        raise ValueError("the frozen alias table must contain exactly 33 mappings")
    if aliases.isna().any().any():
        raise ValueError("the frozen alias table must be one-to-one")
    if aliases["adt_feature"].duplicated().any() or aliases[
        "gene_symbol"
    ].duplicated().any():
        raise ValueError("the frozen alias table must be one-to-one")

    metadata = pd.read_csv(metadata_path)
    required_metadata = {"Unnamed: 0", "donor", "time", "celltype.l1"}
    if not required_metadata.issubset(metadata.columns):
        raise ValueError("GEO metadata lacks a frozen column")
    metadata = metadata.rename(columns={"Unnamed: 0": "cell_id"})
    if metadata["cell_id"].duplicated().any():
        raise ValueError("GEO metadata cell identifiers are not unique")
    eligible = (
        metadata["donor"].astype(str).isin(DEVELOPMENT + HELD)
        & metadata["time"].isin(TIME_MAP)
        & metadata["celltype.l1"].astype(str).isin(LINEAGES)
    )
    metadata = metadata.loc[
        eligible, ["cell_id", "donor", "time", "celltype.l1"]
    ].copy()
    if metadata.empty:
        raise ValueError("no cells satisfy the frozen metadata grid")

    source_paths = {
        "GSM5008740_RNA_5P-barcodes.tsv.gz": rna_barcodes_path,
        "GSM5008740_RNA_5P-features.tsv.gz": rna_features_path,
        "GSM5008740_RNA_5P-matrix.mtx.gz": rna_matrix_path,
        "GSM5008741_ADT_5P-barcodes.tsv.gz": adt_barcodes_path,
        "GSM5008741_ADT_5P-features.tsv.gz": adt_features_path,
        "GSM5008741_ADT_5P-matrix.mtx.gz": adt_matrix_path,
    }
    if set(source_paths) != set(REQUIRED_SOURCE_FILES):
        raise ValueError("source file set differs from the reducer")
    expected_files = {name: _source_record(source, name) for name in source_paths}
    for name, path in source_paths.items():
        if path.name != name:
            raise ValueError(f"source filename differs from the frozen manifest: {name}")
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"source must be a regular, non-symlink file: {name}")
        if path.stat().st_size != expected_files[name]["bytes"]:
            raise ValueError(f"source byte count differs from the frozen manifest: {name}")

    acquisition: dict[str, dict[str, object]] = {}
    rna_barcodes_rows, record = _read_tsv_file(rna_barcodes_path)
    acquisition[record["name"]] = record
    adt_barcodes_rows, record = _read_tsv_file(adt_barcodes_path)
    acquisition[record["name"]] = record
    rna_features, record = _read_tsv_file(rna_features_path)
    acquisition[record["name"]] = record
    adt_features, record = _read_tsv_file(adt_features_path)
    acquisition[record["name"]] = record

    rna_barcodes = [row[0] for row in rna_barcodes_rows]
    adt_barcodes = [row[0] for row in adt_barcodes_rows]
    if rna_barcodes != adt_barcodes or len(set(rna_barcodes)) != len(rna_barcodes):
        raise ValueError("RNA and ADT barcode files are not identical and unique")
    barcode_lookup = {value: index for index, value in enumerate(rna_barcodes)}
    missing = sorted(set(metadata["cell_id"].astype(str)) - set(barcode_lookup))
    if missing:
        raise ValueError("eligible GEO metadata cells are absent from 5-prime matrices")
    metadata["source_column"] = metadata["cell_id"].map(barcode_lookup)
    metadata = metadata.sort_values("source_column").reset_index(drop=True)
    column_lookup = {
        int(source_column): index
        for index, source_column in enumerate(metadata["source_column"])
    }

    rna_names = [_feature_name(row) for row in rna_features]
    rna_index: dict[str, list[int]] = {}
    for index, value in enumerate(rna_names):
        rna_index.setdefault(value, []).append(index)
    invalid_genes = {
        gene: len(rna_index.get(gene, []))
        for gene in aliases["gene_symbol"]
        if len(rna_index.get(gene, [])) != 1
    }
    if invalid_genes:
        raise ValueError(f"frozen RNA genes are missing or duplicated: {invalid_genes}")
    rna_rows = [rna_index[value][0] for value in aliases["gene_symbol"]]
    rna_row_lookup = {source_row: index for index, source_row in enumerate(rna_rows)}

    adt_names = [_feature_name(row) for row in adt_features]
    if len(set(adt_names)) != len(adt_names):
        raise ValueError("ADT feature names are not unique")
    adt_lookup = {value: index for index, value in enumerate(adt_names)}
    missing_adt = sorted(set(aliases["adt_feature"]) - set(adt_lookup))
    if missing_adt:
        raise ValueError(f"frozen ADT features are missing: {missing_adt}")
    adt_row_lookup = {index: index for index in range(len(adt_names))}

    rna, rna_total, record = _read_matrix_subset(
        rna_matrix_path,
        rna_row_lookup,
        column_lookup,
        expected_rows=len(rna_features),
        expected_columns=len(rna_barcodes),
        collect_column_totals=True,
    )
    acquisition[record["name"]] = record
    adt, _, record = _read_matrix_subset(
        adt_matrix_path,
        adt_row_lookup,
        column_lookup,
        expected_rows=len(adt_features),
        expected_columns=len(adt_barcodes),
        collect_column_totals=False,
    )
    acquisition[record["name"]] = record

    assert rna_total is not None
    if np.any(rna_total <= 0.0):
        raise ValueError("retained RNA library totals must be positive")
    if set(acquisition) != set(REQUIRED_SOURCE_FILES):
        raise ValueError("not every required source file was hashed")
    for name, observed in acquisition.items():
        if observed["bytes"] != expected_files[name]["bytes"]:
            raise ValueError(f"source acquisition byte count differs: {name}")
        expected_sha = expected_files[name].get("sha256")
        if expected_sha is not None and observed["sha256"] != expected_sha:
            raise ValueError(f"source acquisition SHA-256 differs: {name}")
        observed["url"] = expected_files[name]["url"]

    marker = pd.DataFrame(
        {
            "marker_id": [
                f"{adt}::{gene}"
                for adt, gene in zip(aliases["adt_feature"], aliases["gene_symbol"])
            ],
            "adt_feature": aliases["adt_feature"],
            "gene_symbol": aliases["gene_symbol"],
            "adt_row": [adt_lookup[value] for value in aliases["adt_feature"]],
        }
    )
    cell = pd.DataFrame(
        {
            "cell_id": metadata["cell_id"].astype(str),
            "donor": metadata["donor"].astype(str),
            "day": metadata["time"].map(TIME_MAP).astype(int),
            "source_time_code": metadata["time"].astype(int),
            "cell_type": metadata["celltype.l1"].astype(str),
            "rna_total": rna_total,
        }
    )
    source_acquisition = {
        "schema": "hao-gse164378-source-acquisition/1.0",
        "source_manifest_sha256": _sha256(source_path),
        "files": [acquisition[name] for name in REQUIRED_SOURCE_FILES],
    }

    temporary.mkdir(parents=True)
    try:
        _write_mtx_gzip(temporary / "rna_matched.mtx.gz", rna)
        _write_mtx_gzip(temporary / "adt_all.mtx.gz", adt)
        _write_gzip_text(
            temporary / "cells.tsv.gz", cell.to_csv(sep="\t", index=False)
        )
        (temporary / "markers.tsv").write_text(marker.to_csv(sep="\t", index=False))
        (temporary / "adt_features.tsv").write_text(
            pd.DataFrame({"adt_feature": adt_names}).to_csv(sep="\t", index=False)
        )
        (temporary / "source_acquisition.json").write_text(
            json.dumps(source_acquisition, indent=2, allow_nan=False) + "\n"
        )
        manifest_rows = []
        for relative in OUTPUTS:
            path = temporary / relative
            manifest_rows.append(
                {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)}
            )
        (temporary / "reducer_manifest.tsv").write_text(
            pd.DataFrame(manifest_rows).to_csv(sep="\t", index=False)
        )
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna-barcodes", required=True)
    parser.add_argument("--rna-features", required=True)
    parser.add_argument("--rna-matrix", required=True)
    parser.add_argument("--adt-barcodes", required=True)
    parser.add_argument("--adt-features", required=True)
    parser.add_argument("--adt-matrix", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--aliases", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reduce(
        rna_barcodes_path=Path(args.rna_barcodes),
        rna_features_path=Path(args.rna_features),
        rna_matrix_path=Path(args.rna_matrix),
        adt_barcodes_path=Path(args.adt_barcodes),
        adt_features_path=Path(args.adt_features),
        adt_matrix_path=Path(args.adt_matrix),
        metadata_path=Path(args.metadata),
        aliases_path=Path(args.aliases),
        source_path=Path(args.source_manifest),
        output=Path(args.output),
    )


if __name__ == "__main__":
    main()
