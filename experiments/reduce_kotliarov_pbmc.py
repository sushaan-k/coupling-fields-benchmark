"""Create leak-separated Kotliarov PBMC prediction and scoring bundles."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


DEVELOPMENT = (
    "200",
    "207",
    "212",
    "233",
    "237",
    "245",
    "256",
    "261",
    "273",
    "277",
)
HELD = ("201", "205", "215", "229", "234", "236", "250", "268", "279")
EXCLUDED_DONOR = "209"
LINEAGES = ("B", "CD4 T", "CD8 T", "NK", "Monocyte")
MINIMUM_CELLS_PER_DONOR_LINEAGE = 50
MINIMUM_RETAINED_LINEAGES = 4
MAD_SCALE = 1.4826

RNA_MATRIX_KEY = "assays/0/matrix.h5"
ADT_MATRIX_KEY = "alternative_experiments/0/assays/0/array.h5"
METADATA_KEY = "column_data/basic_columns.h5"
RNA_FEATURE_KEY = "row_data/basic_columns.h5"
ADT_FEATURE_KEY = "alternative_experiments/0/row_data/basic_columns.h5"
ASSET_MANIFEST_KEY = "manifest_2024-04-18.json"
METADATA_SOURCE_PREFIX = "data/development/kotliarov_pbmc_metadata_v1/"

PREDICTION_OUTPUTS = (
    "cells.tsv.gz",
    "markers.tsv",
    "entities.tsv",
    "rna_values.npy.gz",
    "rna_states.npy.gz",
    "development_cell_index.tsv.gz",
    "development_adt_values.npy.gz",
    "development_adt_states.npy.gz",
    "cuts.tsv",
    "held_adt_marginals.tsv",
    "qc_thresholds.tsv",
    "lineage_parameters.tsv",
    "source_acquisition.json",
)
SCORE_OUTPUTS = (
    "held_cells.tsv.gz",
    "held_adt_states.npy.gz",
    "score_binding.json",
)


def _digest(path: Path) -> dict[str, object]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            size += len(chunk)
            sha256.update(chunk)
            md5.update(chunk)
    return {"bytes": size, "md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def _sha256(path: Path) -> str:
    return str(_digest(path)["sha256"])


def _read_json(path: Path) -> dict:
    def reject(token: str) -> None:
        raise ValueError(f"non-finite JSON value: {token}")

    return json.loads(path.read_text(), parse_constant=reject)


def _decode(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values)
    if flat.dtype.kind in {"S", "O"}:
        return np.asarray(
            [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in flat
            ],
            dtype=str,
        )
    return flat


def _read_dataframe_columns(
    path: Path, required: tuple[str, ...]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with h5py.File(path, "r") as handle:
        if "data_frame" not in handle:
            raise ValueError(f"missing alabaster data_frame group: {path.name}")
        group = handle["data_frame"]
        names = [str(value) for value in _decode(group["column_names"][...])]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate data-frame columns: {path.name}")
        missing = sorted(set(required) - set(names))
        if missing:
            raise ValueError(f"missing frozen data-frame columns in {path.name}: {missing}")
        row_names = _decode(group["row_names"][...]).astype(str)
        if len(row_names) != int(group.attrs["row-count"]):
            raise ValueError(f"data-frame row count differs: {path.name}")
        output: dict[str, np.ndarray] = {}
        for name in required:
            item = group["data"][str(names.index(name))]
            if isinstance(item, h5py.Dataset):
                values = item[...]
            elif isinstance(item, h5py.Group) and {"codes", "levels"}.issubset(item):
                codes = np.asarray(item["codes"][...], dtype=np.int64)
                levels = _decode(item["levels"][...])
                if np.any(codes >= len(levels)):
                    raise ValueError(f"invalid factor codes in {path.name}: {name}")
                values = levels[codes]
            else:
                raise ValueError(f"unsupported data-frame column in {path.name}: {name}")
            if len(values) != len(row_names):
                raise ValueError(f"data-frame column length differs: {path.name}: {name}")
            output[name] = _decode(values)
    return row_names, output


def _read_row_names(path: Path) -> np.ndarray:
    row_names, _ = _read_dataframe_columns(path, ())
    return row_names


def _source_file(source: dict, key: str) -> dict:
    matches = [record for record in source.get("files", []) if record.get("key") == key]
    if len(matches) != 1:
        raise ValueError(f"source manifest must contain one {key} record")
    return matches[0]


def _metadata_sha(source: dict, suffix: str) -> str:
    expected_path = METADATA_SOURCE_PREFIX + suffix
    matches = [
        record
        for record in source.get("metadata_files", [])
        if record.get("path") == expected_path
    ]
    if len(matches) != 1:
        raise ValueError(f"source manifest must bind metadata path {suffix}")
    value = matches[0].get("sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"source manifest metadata SHA-256 is malformed: {suffix}")
    return value


def _require_regular(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"input must be a regular non-symlink file: {path}")


def _verify_metadata_file(path: Path, source: dict, suffix: str) -> dict[str, object]:
    _require_regular(path)
    observed = _digest(path)
    if observed["sha256"] != _metadata_sha(source, suffix):
        raise ValueError(f"metadata SHA-256 differs from frozen source manifest: {suffix}")
    return observed


def _verify_source_matrix(
    path: Path,
    record: dict,
    asset_record: dict,
) -> dict[str, object]:
    _require_regular(path)
    if path.name != record.get("name"):
        raise ValueError(f"source matrix filename differs: {record.get('key')}")
    if record.get("bytes") != asset_record.get("size") or record.get(
        "md5"
    ) != asset_record.get("md5sum"):
        raise ValueError(f"source and gypsum manifests disagree: {record.get('key')}")
    observed = _digest(path)
    if observed["bytes"] != record.get("bytes") or observed["md5"] != record.get(
        "md5"
    ):
        raise ValueError(f"source matrix byte count or MD5 differs: {record.get('key')}")
    expected_sha = record.get("sha256")
    if expected_sha is not None and observed["sha256"] != expected_sha:
        raise ValueError(f"source matrix SHA-256 differs: {record.get('key')}")
    return observed


def _read_csc_subset(
    path: Path,
    rows: np.ndarray,
    columns: np.ndarray,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    columns = np.asarray(columns, dtype=np.int64)
    if len(np.unique(rows)) != len(rows) or len(np.unique(columns)) != len(columns):
        raise ValueError("sparse subset axes must be unique")
    if np.any(np.diff(columns) <= 0):
        raise ValueError("sparse subset columns must be strictly increasing")
    with h5py.File(path, "r") as handle:
        if "compressed_sparse_matrix" not in handle:
            raise ValueError("RNA HDF5 lacks compressed_sparse_matrix")
        group = handle["compressed_sparse_matrix"]
        required = {"data", "indices", "indptr", "shape"}
        if not required.issubset(group):
            raise ValueError("RNA compressed_sparse_matrix is incomplete")
        layout = group.attrs.get("layout")
        if isinstance(layout, bytes):
            layout = layout.decode("utf-8")
        if layout != "CSC":
            raise ValueError("RNA compressed_sparse_matrix must use CSC layout")
        shape = tuple(int(value) for value in group["shape"][...])
        if shape != expected_shape:
            raise ValueError("RNA matrix dimensions differ from metadata axes")
        data = group["data"]
        indices = group["indices"]
        indptr = np.asarray(group["indptr"][...], dtype=np.int64)
        if (
            data.ndim != 1
            or indices.ndim != 1
            or len(data) != len(indices)
            or len(indptr) != shape[1] + 1
            or indptr[0] != 0
            or indptr[-1] != len(data)
            or np.any(np.diff(indptr) < 0)
        ):
            raise ValueError("RNA CSC structural arrays are invalid")
        if "missing-value-placeholder" in data.attrs:
            raise ValueError("RNA count matrix contains encoded missing values")
        if np.any(rows < 0) or np.any(rows >= shape[0]):
            raise ValueError("requested RNA row is outside the matrix")
        if np.any(columns < 0) or np.any(columns >= shape[1]):
            raise ValueError("requested RNA column is outside the matrix")

        row_map = np.full(shape[0], -1, dtype=np.int64)
        row_map[rows] = np.arange(len(rows), dtype=np.int64)
        column_map = np.full(shape[1], -1, dtype=np.int64)
        column_map[columns] = np.arange(len(columns), dtype=np.int64)
        output = np.zeros((len(rows), len(columns)), dtype=np.float64)

        for start_column in range(0, shape[1], 512):
            end_column = min(start_column + 512, shape[1])
            if not np.any(column_map[start_column:end_column] >= 0):
                continue
            start = int(indptr[start_column])
            end = int(indptr[end_column])
            block_rows = np.asarray(indices[start:end], dtype=np.int64)
            block_values = np.asarray(data[start:end], dtype=np.float64)
            if (
                np.any(block_rows < 0)
                or np.any(block_rows >= shape[0])
                or np.any(~np.isfinite(block_values))
                or np.any(block_values < 0)
            ):
                raise ValueError("RNA CSC contains an invalid count entry")
            block_columns = np.repeat(
                np.arange(start_column, end_column, dtype=np.int64),
                np.diff(indptr[start_column : end_column + 1]),
            )
            target_rows = row_map[block_rows]
            target_columns = column_map[block_columns]
            keep = (target_rows >= 0) & (target_columns >= 0)
            np.add.at(
                output,
                (target_rows[keep], target_columns[keep]),
                block_values[keep],
            )
    return output


def _read_dense_subset(
    path: Path,
    rows: np.ndarray,
    columns: np.ndarray,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    columns = np.asarray(columns, dtype=np.int64)
    if len(np.unique(rows)) != len(rows) or len(np.unique(columns)) != len(columns):
        raise ValueError("dense subset axes must be unique")
    if np.any(np.diff(columns) <= 0):
        raise ValueError("dense subset columns must be strictly increasing")
    with h5py.File(path, "r") as handle:
        if "dense_array" not in handle or "data" not in handle["dense_array"]:
            raise ValueError("ADT HDF5 lacks dense_array/data")
        group = handle["dense_array"]
        data = group["data"]
        if data.ndim != 2:
            raise ValueError("ADT dense_array/data must be two-dimensional")
        transposed = bool(int(group.attrs.get("transposed", 0)))
        logical_shape = data.shape[::-1] if transposed else data.shape
        if logical_shape != expected_shape:
            raise ValueError("ADT matrix dimensions differ from metadata axes")
        if "missing-value-placeholder" in data.attrs:
            raise ValueError("ADT count matrix contains encoded missing values")
        if np.any(rows < 0) or np.any(rows >= logical_shape[0]):
            raise ValueError("requested ADT row is outside the matrix")
        if np.any(columns < 0) or np.any(columns >= logical_shape[1]):
            raise ValueError("requested ADT column is outside the matrix")
        output = np.empty((len(rows), len(columns)), dtype=np.float64)
        for target, source in enumerate(rows):
            if transposed:
                output[target] = np.asarray(data[columns, source], dtype=np.float64)
            else:
                output[target] = np.asarray(data[source, columns], dtype=np.float64)
    if np.any(~np.isfinite(output)) or np.any(output < 0):
        raise ValueError("ADT dense array contains an invalid count entry")
    return output


def _read_aliases(path: Path) -> pd.DataFrame:
    aliases = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if list(aliases.columns) != ["adt_target", "gene_symbol", "module"]:
        raise ValueError("alias table columns differ from the frozen contract")
    if aliases.empty or (aliases == "").any().any():
        raise ValueError("alias table contains an empty value")
    if aliases["adt_target"].duplicated().any() or aliases[
        "gene_symbol"
    ].duplicated().any():
        raise ValueError("alias table must be exactly one-to-one")
    return aliases


def _read_lineage_markers(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    expected = ["lineage", "positive_markers", "negative_markers"]
    if list(table.columns) != expected or tuple(table["lineage"]) != LINEAGES:
        raise ValueError("lineage marker table differs from the frozen contract")
    for column in ("positive_markers", "negative_markers"):
        table[column] = table[column].map(
            lambda value: tuple(token.strip() for token in value.split(","))
        )
        if any(not values or any(not value for value in values) for values in table[column]):
            raise ValueError(f"lineage marker list is empty: {column}")
    for row in table.itertuples(index=False):
        if set(row.positive_markers) & set(row.negative_markers):
            raise ValueError(f"positive and negative markers overlap: {row.lineage}")
    return table


def _unique_indices(names: np.ndarray, requested: list[str], label: str) -> np.ndarray:
    lookup: dict[str, list[int]] = {}
    for index, name in enumerate(names.astype(str)):
        lookup.setdefault(name, []).append(index)
    invalid = {name: len(lookup.get(name, [])) for name in requested if len(lookup.get(name, [])) != 1}
    if invalid:
        raise ValueError(f"frozen {label} values are missing or duplicated: {invalid}")
    return np.asarray([lookup[name][0] for name in requested], dtype=np.int64)


def _median_mad(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, mad


def _apply_rna_qc(metadata: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    keep = np.ones(len(metadata), dtype=bool)
    records: list[dict[str, object]] = []
    for donor in DEVELOPMENT + HELD:
        donor_mask = metadata["donor"].to_numpy() == donor
        if not np.any(donor_mask):
            raise ValueError(f"frozen donor has no eligible cells: {donor}")
        donor_keep = np.ones(int(donor_mask.sum()), dtype=bool)
        row: dict[str, object] = {"donor": donor, "cells_before_qc": int(donor_mask.sum())}
        for column in ("nGene", "nUMI"):
            values = metadata.loc[donor_mask, column].to_numpy(dtype=float)
            median, mad = _median_mad(values)
            if mad > 0:
                lower, upper = median - 3 * mad, median + 3 * mad
                donor_keep &= (values >= lower) & (values <= upper)
            else:
                lower, upper = -math.inf, math.inf
            row.update(
                {
                    f"{column}_median": median,
                    f"{column}_mad": mad,
                    f"{column}_lower": lower,
                    f"{column}_upper": upper,
                }
            )
        values = metadata.loc[donor_mask, "pctMT"].to_numpy(dtype=float)
        median, mad = _median_mad(values)
        upper = min(0.20, median + 3 * mad) if mad > 0 else 0.20
        donor_keep &= values <= upper
        row.update(
            {"pctMT_median": median, "pctMT_mad": mad, "pctMT_upper": upper}
        )
        donor_indices = np.flatnonzero(donor_mask)
        keep[donor_indices] = donor_keep
        row["cells_after_qc"] = int(donor_keep.sum())
        records.append(row)
    return keep, pd.DataFrame(records)


def _assign_lineages(
    normalized: np.ndarray,
    gene_names: list[str],
    donors: np.ndarray,
    marker_table: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    gene_lookup = {name: index for index, name in enumerate(gene_names)}
    standardized = np.zeros_like(normalized, dtype=np.float64)
    parameters: list[dict[str, object]] = []
    marker_genes = list(
        dict.fromkeys(
            gene
            for row in marker_table.itertuples(index=False)
            for gene in (*row.positive_markers, *row.negative_markers)
        )
    )
    for donor in DEVELOPMENT + HELD:
        donor_mask = donors == donor
        for gene in marker_genes:
            index = gene_lookup[gene]
            values = normalized[index, donor_mask]
            median, mad = _median_mad(values)
            scale = MAD_SCALE * mad
            if scale > 0:
                standardized[index, donor_mask] = (values - median) / scale
            parameters.append(
                {
                    "donor": donor,
                    "gene_symbol": gene,
                    "median": median,
                    "mad": mad,
                    "scale": scale,
                    "informative": int(scale > 0),
                }
            )

    scores = np.empty((len(LINEAGES), normalized.shape[1]), dtype=np.float64)
    for lineage_index, row in enumerate(marker_table.itertuples(index=False)):
        positive = np.asarray([gene_lookup[gene] for gene in row.positive_markers])
        negative = np.asarray([gene_lookup[gene] for gene in row.negative_markers])
        scores[lineage_index] = standardized[positive].mean(axis=0) - standardized[
            negative
        ].mean(axis=0)
    best_index = np.argmax(scores, axis=0)
    best = scores[best_index, np.arange(scores.shape[1])]
    second = np.partition(scores, -2, axis=0)[-2]
    unique = np.sum(scores == best[None, :], axis=0) == 1
    eligible = np.isfinite(best) & unique & (best > 0)
    lineage = np.full(scores.shape[1], "", dtype=object)
    lineage[eligible] = np.asarray(LINEAGES, dtype=object)[best_index[eligible]]
    margin = best - second
    margin[~eligible] = np.nan
    return lineage.astype(str), best, margin, pd.DataFrame(parameters)


def _weighted_tertiles(values: np.ndarray, donors: np.ndarray) -> tuple[float, float]:
    donor_order = [donor for donor in DEVELOPMENT if np.any(donors == donor)]
    if donor_order != list(DEVELOPMENT):
        raise ValueError("a development donor is absent from a cut-estimation block")
    weights = np.empty(len(values), dtype=np.float64)
    for donor in donor_order:
        mask = donors == donor
        weights[mask] = 1.0 / (len(donor_order) * int(mask.sum()))
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    ordered_weights = weights[order]
    centers = np.cumsum(ordered_weights) - ordered_weights / 2
    return tuple(
        float(np.interp(probability, centers, ordered_values))
        for probability in (1 / 3, 2 / 3)
    )


def _states(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.searchsorted(np.asarray([lower, upper]), values, side="left").astype(
        np.int8
    )


def _write_gzip_text(path: Path, value: str) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(value.encode("utf-8"))


def _write_npy_gzip(path: Path, value: np.ndarray) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            np.save(stream, np.asarray(value), allow_pickle=False)


def read_npy_gzip(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        return np.load(stream, allow_pickle=False)


def _write_manifest(directory: Path, outputs: tuple[str, ...], name: str) -> list[dict[str, object]]:
    records = []
    for relative in outputs:
        path = directory / relative
        observed = _digest(path)
        records.append({"path": relative, "bytes": observed["bytes"], "sha256": observed["sha256"]})
    (directory / name).write_text(pd.DataFrame(records).to_csv(sep="\t", index=False))
    return records


def reduce(
    *,
    rna_matrix_path: Path,
    adt_matrix_path: Path,
    metadata_path: Path,
    rna_features_path: Path,
    adt_features_path: Path,
    asset_manifest_path: Path,
    source_manifest_path: Path,
    aliases_path: Path,
    lineage_markers_path: Path,
    prediction_output: Path,
    score_output: Path,
) -> None:
    for output in (prediction_output, score_output):
        if output.exists():
            raise FileExistsError(f"reducer output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    prediction_temporary = prediction_output.with_name(prediction_output.name + ".tmp")
    score_temporary = score_output.with_name(score_output.name + ".tmp")
    if prediction_temporary.exists() or score_temporary.exists():
        raise FileExistsError("a reducer temporary directory already exists")

    source = _read_json(source_manifest_path)
    if source.get("schema") != "kotliarov-pbmc-gypsum-source/1.0":
        raise ValueError("source manifest schema differs from version 1.0")
    asset_manifest_record = _verify_metadata_file(
        asset_manifest_path, source, ASSET_MANIFEST_KEY
    )
    asset_manifest = _read_json(asset_manifest_path)
    metadata_records = {
        METADATA_KEY: _verify_metadata_file(metadata_path, source, METADATA_KEY),
        RNA_FEATURE_KEY: _verify_metadata_file(
            rna_features_path, source, RNA_FEATURE_KEY
        ),
        ADT_FEATURE_KEY: _verify_metadata_file(
            adt_features_path, source, ADT_FEATURE_KEY
        ),
    }
    for key, path in (
        (METADATA_KEY, metadata_path),
        (RNA_FEATURE_KEY, rna_features_path),
        (ADT_FEATURE_KEY, adt_features_path),
    ):
        record = asset_manifest.get(key)
        observed = metadata_records[key]
        if not isinstance(record, dict) or observed["bytes"] != record.get(
            "size"
        ) or observed["md5"] != record.get("md5sum"):
            raise ValueError(f"metadata differs from gypsum asset manifest: {key}")

    rna_source = _source_file(source, "rna_matrix")
    adt_source = _source_file(source, "adt_matrix")
    rna_acquisition = _verify_source_matrix(
        rna_matrix_path, rna_source, asset_manifest[RNA_MATRIX_KEY]
    )
    adt_acquisition = _verify_source_matrix(
        adt_matrix_path, adt_source, asset_manifest[ADT_MATRIX_KEY]
    )
    aliases = _read_aliases(aliases_path)
    lineage_markers = _read_lineage_markers(lineage_markers_path)

    metadata_names, metadata_columns = _read_dataframe_columns(
        metadata_path,
        (
            "nGene",
            "nUMI",
            "pctMT",
            "batch",
            "sampleid",
            "joint_classification_global",
            "dmx_hto_match",
            "timepoint",
        ),
    )
    if len(np.unique(metadata_names)) != len(metadata_names):
        raise ValueError("cell identifiers are not unique")
    metadata = pd.DataFrame(
        {
            "cell_id": metadata_names,
            "source_column": np.arange(len(metadata_names), dtype=np.int64),
            "nGene": pd.to_numeric(metadata_columns["nGene"], errors="raise"),
            "nUMI": pd.to_numeric(metadata_columns["nUMI"], errors="raise"),
            "pctMT": pd.to_numeric(metadata_columns["pctMT"], errors="raise"),
            "batch": metadata_columns["batch"].astype(str),
            "donor": metadata_columns["sampleid"].astype(str),
            "joint": metadata_columns["joint_classification_global"].astype(str),
            "match": metadata_columns["dmx_hto_match"].astype(str),
            "timepoint": metadata_columns["timepoint"].astype(str),
        }
    )
    if np.any(~np.isfinite(metadata[["nGene", "nUMI", "pctMT"]].to_numpy(float))):
        raise ValueError("RNA QC metadata contains a nonfinite value")
    if (
        (metadata["nGene"] < 0).any()
        or (metadata["nUMI"] <= 0).any()
        or (metadata["pctMT"] < 0).any()
        or (metadata["pctMT"] > 1).any()
    ):
        raise ValueError("RNA QC metadata contains an invalid value")
    allowed = set(DEVELOPMENT + HELD + (EXCLUDED_DONOR,))
    if set(metadata["donor"]) != allowed:
        raise ValueError("deposited donor roster differs from the frozen split")
    split = np.full(len(metadata), "", dtype=object)
    split[metadata["donor"].isin(DEVELOPMENT)] = "development"
    split[metadata["donor"].isin(HELD)] = "held"
    metadata["split"] = split.astype(str)
    expected_batch = metadata["donor"].map(
        {**{donor: "1" for donor in DEVELOPMENT}, **{donor: "2" for donor in HELD}}
    )
    base_keep = (
        metadata["donor"].isin(DEVELOPMENT + HELD)
        & (metadata["batch"] == expected_batch)
        & (metadata["joint"] == "SNG_Singlet")
        & (metadata["match"] == "1")
        & (metadata["timepoint"] == "d0")
    )
    metadata = metadata.loc[base_keep].copy().reset_index(drop=True)
    if set(metadata["donor"]) != set(DEVELOPMENT + HELD):
        raise ValueError("a frozen donor is absent after metadata eligibility")
    qc_keep, qc_thresholds = _apply_rna_qc(metadata)
    metadata = metadata.loc[qc_keep].copy().reset_index(drop=True)

    rna_names = _read_row_names(rna_features_path)
    adt_row_names, adt_columns = _read_dataframe_columns(
        adt_features_path, ("target", "isotype")
    )
    adt_targets = adt_columns["target"].astype(str)
    isotype = np.asarray(adt_columns["isotype"]).astype(bool)
    non_isotype_rows = np.flatnonzero(~isotype)
    if len(non_isotype_rows) != 83:
        raise ValueError("the deposited ADT panel must contain exactly 83 non-isotypes")

    lineage_genes = list(
        dict.fromkeys(
            gene
            for row in lineage_markers.itertuples(index=False)
            for gene in (*row.positive_markers, *row.negative_markers)
        )
    )
    alias_genes = aliases["gene_symbol"].tolist()
    requested_genes = list(dict.fromkeys(alias_genes + lineage_genes))
    rna_rows = _unique_indices(rna_names, requested_genes, "RNA gene symbols")
    adt_alias_rows = _unique_indices(
        adt_targets, aliases["adt_target"].tolist(), "ADT targets"
    )
    if np.any(isotype[adt_alias_rows]):
        raise ValueError("the exact alias table contains an isotype ADT")

    source_columns = metadata["source_column"].to_numpy(dtype=np.int64)
    rna_counts = _read_csc_subset(
        rna_matrix_path,
        rna_rows,
        source_columns,
        (len(rna_names), len(metadata_names)),
    )
    normalized = np.log1p(
        rna_counts / metadata["nUMI"].to_numpy(dtype=float)[None, :] * 10_000.0
    )
    lineage, lineage_score, lineage_margin, lineage_parameters = _assign_lineages(
        normalized,
        requested_genes,
        metadata["donor"].to_numpy(dtype=str),
        lineage_markers,
    )
    metadata["lineage"] = lineage
    metadata["lineage_score"] = lineage_score
    metadata["lineage_margin"] = lineage_margin
    support = pd.crosstab(metadata["donor"], metadata["lineage"])
    retained_lineages = tuple(
        lineage_name
        for lineage_name in LINEAGES
        if lineage_name in support.columns
        and all(
            int(support.at[donor, lineage_name]) >= MINIMUM_CELLS_PER_DONOR_LINEAGE
            for donor in DEVELOPMENT + HELD
        )
    )
    if len(retained_lineages) < MINIMUM_RETAINED_LINEAGES:
        raise ValueError("fewer than four RNA-only lineages pass frozen donor support")
    metadata = metadata.loc[metadata["lineage"].isin(retained_lineages)].copy()
    metadata = metadata.sort_values("source_column").reset_index(drop=True)
    metadata.insert(0, "prediction_cell_index", np.arange(len(metadata), dtype=np.int64))

    selected_columns = metadata["source_column"].to_numpy(dtype=np.int64)
    selected_from_qc = np.searchsorted(source_columns, selected_columns)
    normalized = normalized[:, selected_from_qc]
    gene_lookup = {gene: index for index, gene in enumerate(requested_genes)}
    rna_values = normalized[[gene_lookup[gene] for gene in alias_genes]]

    adt_counts = _read_dense_subset(
        adt_matrix_path,
        non_isotype_rows,
        selected_columns,
        (len(adt_row_names), len(metadata_names)),
    )
    adt_clr = np.log1p(adt_counts)
    adt_clr -= adt_clr.mean(axis=0, keepdims=True)
    non_isotype_lookup = {
        source_row: index for index, source_row in enumerate(non_isotype_rows)
    }
    adt_values = adt_clr[[non_isotype_lookup[row] for row in adt_alias_rows]]

    marker_ids = [
        f"{target}::{gene}"
        for target, gene in zip(aliases["adt_target"], aliases["gene_symbol"])
    ]
    markers = pd.DataFrame(
        {
            "marker_index": np.arange(len(aliases), dtype=np.int64),
            "marker_id": marker_ids,
            "adt_target": aliases["adt_target"],
            "adt_feature": adt_row_names[adt_alias_rows],
            "adt_row": adt_alias_rows,
            "gene_symbol": aliases["gene_symbol"],
            "rna_row": [
                _unique_indices(rna_names, [gene], "RNA gene symbol")[0]
                for gene in aliases["gene_symbol"]
            ],
            "module": aliases["module"],
        }
    )
    entity_records = []
    for marker in markers.itertuples(index=False):
        for lineage_name in retained_lineages:
            entity_records.append(
                {
                    "entity_id": f"{marker.marker_id}::{lineage_name}",
                    "marker_index": marker.marker_index,
                    "marker_id": marker.marker_id,
                    "adt_target": marker.adt_target,
                    "gene_symbol": marker.gene_symbol,
                    "module": marker.module,
                    "lineage": lineage_name,
                    "eligible": 0,
                }
            )
    entities = pd.DataFrame(entity_records)

    donors = metadata["donor"].to_numpy(dtype=str)
    lineages = metadata["lineage"].to_numpy(dtype=str)
    split_values = metadata["split"].to_numpy(dtype=str)
    development_mask = split_values == "development"
    held_mask = split_values == "held"
    rna_states = np.empty(rna_values.shape, dtype=np.int8)
    adt_states = np.empty(adt_values.shape, dtype=np.int8)
    cut_records: list[dict[str, object]] = []
    for marker_index, marker_id in enumerate(marker_ids):
        for lineage_name in retained_lineages:
            block = lineages == lineage_name
            development_block = block & development_mask
            for modality, values, output_states in (
                ("RNA", rna_values, rna_states),
                ("ADT", adt_values, adt_states),
            ):
                lower, upper = _weighted_tertiles(
                    values[marker_index, development_block],
                    donors[development_block],
                )
                output_states[marker_index, block] = _states(
                    values[marker_index, block], lower, upper
                )
                cut_records.append(
                    {
                        "marker_index": marker_index,
                        "marker_id": marker_id,
                        "lineage": lineage_name,
                        "modality": modality,
                        "lower_cut": lower,
                        "upper_cut": upper,
                        "distinct": int(lower < upper),
                        "estimation": "development_donor_equal_weighted_tertiles",
                    }
                )
    cuts = pd.DataFrame(cut_records)

    held_marginal_records = []
    for donor in HELD:
        for lineage_name in retained_lineages:
            block = held_mask & (donors == donor) & (lineages == lineage_name)
            for marker_index, marker_id in enumerate(marker_ids):
                counts = np.bincount(adt_states[marker_index, block], minlength=3)
                for state, count in enumerate(counts):
                    held_marginal_records.append(
                        {
                            "donor": donor,
                            "lineage": lineage_name,
                            "marker_index": marker_index,
                            "marker_id": marker_id,
                            "state": state,
                            "count": int(count),
                        }
                    )
    held_marginals = pd.DataFrame(held_marginal_records)

    eligible_entities: list[str] = []
    for entity in entities.itertuples(index=False):
        marker_index = int(entity.marker_index)
        valid = True
        cut_block = cuts.loc[
            (cuts["marker_index"] == marker_index)
            & (cuts["lineage"] == entity.lineage)
        ]
        if len(cut_block) != 2 or not cut_block["distinct"].astype(bool).all():
            valid = False
        for donor in DEVELOPMENT + HELD:
            block = (donors == donor) & (lineages == entity.lineage)
            size = int(block.sum())
            for state_values in (rna_states, adt_states):
                counts = np.bincount(state_values[marker_index, block], minlength=3)
                if np.any(counts < 5) or np.any(counts / size < 0.02):
                    valid = False
        if valid:
            eligible_entities.append(entity.entity_id)
    entities = entities.loc[entities["entity_id"].isin(eligible_entities)].copy()
    entities["eligible"] = 1
    entities = entities[
        [
            "entity_id",
            "marker_index",
            "marker_id",
            "gene_symbol",
            "adt_target",
            "module",
            "lineage",
            "eligible",
        ]
    ].reset_index(drop=True)
    if len(entities) < 32 or entities["marker_id"].nunique() < 16:
        raise ValueError(
            "frozen separate-margin entity support is below 16 markers or 32 "
            f"entities: {entities['marker_id'].nunique()} markers, {len(entities)} entities"
        )
    eligible_keys = set(zip(entities["marker_id"], entities["lineage"]))
    cuts = cuts.loc[
        [
            (marker_id, lineage_name) in eligible_keys
            for marker_id, lineage_name in zip(cuts["marker_id"], cuts["lineage"])
        ]
    ].reset_index(drop=True)
    held_marginals = held_marginals.loc[
        [
            (marker_id, lineage_name) in eligible_keys
            for marker_id, lineage_name in zip(
                held_marginals["marker_id"], held_marginals["lineage"]
            )
        ]
    ].reset_index(drop=True)

    prediction_temporary.mkdir()
    score_temporary.mkdir()
    try:
        cell_columns = [
            "prediction_cell_index",
            "cell_id",
            "source_column",
            "donor",
            "batch",
            "split",
            "nGene",
            "nUMI",
            "pctMT",
            "lineage",
            "lineage_score",
            "lineage_margin",
        ]
        _write_gzip_text(
            prediction_temporary / "cells.tsv.gz",
            metadata[cell_columns].to_csv(sep="\t", index=False),
        )
        (prediction_temporary / "markers.tsv").write_text(
            markers.to_csv(sep="\t", index=False)
        )
        (prediction_temporary / "entities.tsv").write_text(
            entities.to_csv(sep="\t", index=False)
        )
        _write_npy_gzip(prediction_temporary / "rna_values.npy.gz", rna_values)
        _write_npy_gzip(prediction_temporary / "rna_states.npy.gz", rna_states)
        development_indices = np.flatnonzero(development_mask)
        development_index = pd.DataFrame(
            {
                "development_position": np.arange(len(development_indices)),
                "prediction_cell_index": development_indices,
                "cell_id": metadata.loc[development_mask, "cell_id"].to_numpy(),
            }
        )
        _write_gzip_text(
            prediction_temporary / "development_cell_index.tsv.gz",
            development_index.to_csv(sep="\t", index=False),
        )
        _write_npy_gzip(
            prediction_temporary / "development_adt_values.npy.gz",
            adt_values[:, development_mask],
        )
        _write_npy_gzip(
            prediction_temporary / "development_adt_states.npy.gz",
            adt_states[:, development_mask],
        )
        (prediction_temporary / "cuts.tsv").write_text(
            cuts.to_csv(sep="\t", index=False)
        )
        (prediction_temporary / "held_adt_marginals.tsv").write_text(
            held_marginals.to_csv(sep="\t", index=False)
        )
        (prediction_temporary / "qc_thresholds.tsv").write_text(
            qc_thresholds.to_csv(sep="\t", index=False)
        )
        (prediction_temporary / "lineage_parameters.tsv").write_text(
            lineage_parameters.to_csv(sep="\t", index=False)
        )
        acquisition = {
            "schema": "kotliarov-pbmc-reducer-source-acquisition/1.0",
            "source_manifest_sha256": _sha256(source_manifest_path),
            "asset_manifest": asset_manifest_record,
            "metadata_files": metadata_records,
            "rna_matrix": {**rna_acquisition, "url": rna_source["url"]},
            "adt_matrix": {**adt_acquisition, "url": adt_source["url"]},
            "alias_sha256": _sha256(aliases_path),
            "lineage_markers_sha256": _sha256(lineage_markers_path),
            "excluded_donor": EXCLUDED_DONOR,
            "development_donors": list(DEVELOPMENT),
            "held_donors": list(HELD),
            "retained_lineages": list(retained_lineages),
            "cells_after_qc_and_lineage_support": len(metadata),
        }
        (prediction_temporary / "source_acquisition.json").write_text(
            json.dumps(acquisition, indent=2, allow_nan=False, sort_keys=True) + "\n"
        )
        prediction_records = _write_manifest(
            prediction_temporary, PREDICTION_OUTPUTS, "prediction_manifest.tsv"
        )

        held_indices = np.flatnonzero(held_mask)
        held_cells = pd.DataFrame(
            {
                "held_position": np.arange(len(held_indices)),
                "prediction_cell_index": held_indices,
                "cell_id": metadata.loc[held_mask, "cell_id"].to_numpy(),
                "donor": donors[held_mask],
                "lineage": lineages[held_mask],
            }
        )
        _write_gzip_text(
            score_temporary / "held_cells.tsv.gz",
            held_cells.to_csv(sep="\t", index=False),
        )
        _write_npy_gzip(
            score_temporary / "held_adt_states.npy.gz", adt_states[:, held_mask]
        )
        payload = json.dumps(
            prediction_records, sort_keys=True, separators=(",", ":")
        ).encode()
        score_binding = {
            "schema": "kotliarov-pbmc-reducer-score-binding/1.0",
            "prediction_payload_sha256": hashlib.sha256(payload).hexdigest(),
            "prediction_payload": prediction_records,
            "held_cell_axis_sha256": _sha256(score_temporary / "held_cells.tsv.gz"),
            "held_adt_states_sha256": _sha256(
                score_temporary / "held_adt_states.npy.gz"
            ),
            "adt_source_sha256": adt_acquisition["sha256"],
        }
        (score_temporary / "score_binding.json").write_text(
            json.dumps(score_binding, indent=2, allow_nan=False, sort_keys=True) + "\n"
        )
        _write_manifest(score_temporary, SCORE_OUTPUTS, "score_manifest.tsv")

        prediction_temporary.rename(prediction_output)
        score_temporary.rename(score_output)
    except Exception:
        shutil.rmtree(prediction_temporary, ignore_errors=True)
        shutil.rmtree(score_temporary, ignore_errors=True)
        if prediction_output.exists() and not score_output.exists():
            shutil.rmtree(prediction_output, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna-matrix", required=True)
    parser.add_argument("--adt-matrix", required=True)
    parser.add_argument("--metadata-root", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--aliases", required=True)
    parser.add_argument("--lineage-markers", required=True)
    parser.add_argument("--prediction-output", required=True)
    parser.add_argument("--score-output", required=True)
    args = parser.parse_args()
    metadata_root = Path(args.metadata_root)
    reduce(
        rna_matrix_path=Path(args.rna_matrix),
        adt_matrix_path=Path(args.adt_matrix),
        metadata_path=metadata_root / METADATA_KEY,
        rna_features_path=metadata_root / RNA_FEATURE_KEY,
        adt_features_path=metadata_root / ADT_FEATURE_KEY,
        asset_manifest_path=metadata_root / ASSET_MANIFEST_KEY,
        source_manifest_path=Path(args.source_manifest),
        aliases_path=Path(args.aliases),
        lineage_markers_path=Path(args.lineage_markers),
        prediction_output=Path(args.prediction_output),
        score_output=Path(args.score_output),
    )


if __name__ == "__main__":
    main()
