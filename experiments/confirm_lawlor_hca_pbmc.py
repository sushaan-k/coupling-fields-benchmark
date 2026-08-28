"""Prospective held-donor RNA-protein confirmation on the Lawlor PBMC study."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread
from sklearn.linear_model import Ridge

from mapreg.classical_residuals import (
    conditional_poisson_residuals,
    poisson_independence_residuals,
)
from mapreg.coupling_fields import (
    association_coordinates,
    association_field,
    conditional_association_coordinates,
    field_from_coordinates,
    fit_structured_coupling_fields,
    helmert_contrast,
    inverse_permutation_variance_weights,
    normalized_hypergraph_laplacian,
)
from mapreg.table_prediction import (
    field_coordinates_to_table,
    ipf_to_margins,
    multinomial_deviance_per_observation,
    residual_coordinates_to_table,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/SECOND_HELD_UNIT_CONFIRMATION_PROTOCOL_2026-08-27.md"
DESIGNATION = (
    ROOT
    / "benchmark_release/coupling_fields_v1/LAWLOR_CANDIDATE_DESIGNATION.json"
)
SOURCE_MANIFEST = ROOT / "data/development/lawlor_hca_pbmc/source_manifest_v1.json"
SUPPORT = ROOT / "data/development/lawlor_hca_pbmc/metadata_support_v1.json"
ALIASES = ROOT / "data/development/lawlor_hca_pbmc/adt_gene_aliases_v1.tsv"
REDUCER = ROOT / "experiments/reduce_lawlor_hca_pbmc.R"
SCGPT = ROOT / "data/scgpt_gene_embeddings.npz"
OUTPUT = ROOT / "results/lawlor_hca_pbmc_confirmation.json"
IMPLEMENTATION_FILES = (
    ROOT / "mapreg/coupling_fields.py",
    ROOT / "mapreg/classical_residuals.py",
    ROOT / "mapreg/table_prediction.py",
)

SEED = 20260827
PERMUTATIONS = 64
BOOTSTRAPS = 2_000
PSEUDOCOUNT = 0.5
NUCLEAR_FRACTION = 0.1
GRAPH_PENALTY = 5.0
DEVELOPMENT = (
    "202937150118_R01C01",
    "202937150118_R03C01",
    "202937150091_R01C01",
    "202937150118_R06C01",
    "202937150118_R04C01",
    "202937150118_R07C01",
)
HELD = (
    "202937150118_R05C01",
    "202937150091_R02C01",
    "202937150118_R08C01",
    "202937150118_R02C01",
)
CONTRASTS = (
    ("CD3_CD28", "B"),
    ("CD3_CD28", "CD4T_Naive"),
    ("CD3_CD28", "CD4T_Mem"),
    ("CD3_CD28", "CD8T_Naive"),
    ("CD3_CD28", "CD8T_Mem"),
    ("LPS", "CD14_Mono"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed(*values: str) -> int:
    joined = "\0".join(values).encode()
    return int.from_bytes(hashlib.sha256(joined).digest()[:4], "big")


def _read_json(path: Path) -> dict:
    def reject(token: str) -> None:
        raise ValueError(f"non-finite JSON number: {token}")

    return json.loads(path.read_text(), parse_constant=reject)


def preflight(*, require_sealed: bool) -> dict[str, object]:
    designation = _read_json(DESIGNATION)
    expected = {
        "protocol_sha256": _sha256(PROTOCOL),
        "source_manifest_sha256": _sha256(SOURCE_MANIFEST),
        "metadata_support_sha256": _sha256(SUPPORT),
        "alias_sha256": _sha256(ALIASES),
        "embedding_sha256": _sha256(SCGPT),
    }
    for key, observed in expected.items():
        if designation[key] != observed:
            raise ValueError(f"designation {key} is stale")
    runner_sha = _sha256(Path(__file__))
    reducer_sha = _sha256(REDUCER)
    implementation_sha = {
        str(path.relative_to(ROOT)): _sha256(path) for path in IMPLEMENTATION_FILES
    }
    if designation.get("implementation_sha256") != implementation_sha:
        raise ValueError("designation implementation_sha256 is stale")
    bound_code = {
        "runner": str(Path(__file__).relative_to(ROOT)),
        "runner_sha256": runner_sha,
        "reducer": str(REDUCER.relative_to(ROOT)),
        "reducer_sha256": reducer_sha,
    }
    for key, observed in bound_code.items():
        if designation.get(key) != observed:
            raise ValueError(f"designation {key} is stale")
    if require_sealed:
        if designation["status"] != "SEALED":
            raise PermissionError("candidate designation is not SEALED")
        if not designation["outcome_access_authorized"]:
            raise PermissionError("outcome access is not authorized")
        if not designation.get("public_freeze_commit"):
            raise PermissionError("a public freeze commit is required")
        if not designation.get("public_freeze_url"):
            raise PermissionError("a public freeze URL is required")
    return {
        "designation_status": designation["status"],
        "outcome_access_authorized": designation["outcome_access_authorized"],
        **bound_code,
        "implementation_sha256": implementation_sha,
        **expected,
        "designation_sha256": _sha256(DESIGNATION),
    }


def _source_path(manifest: dict, name: str) -> dict:
    return next(record for record in manifest["files"] if record["name"] == name)


def reduce_inputs(
    *, rna: Path, adt: Path, annotations: Path, output: Path
) -> None:
    source = _read_json(SOURCE_MANIFEST)
    for path, name in (
        (rna, "CZI.PBMC.RNA.matrix.Rds"),
        (adt, "CZI.PBMC.ADT.matrix.Rds"),
        (annotations, "CZI.PBMC.cell.annotations.csv"),
    ):
        record = _source_path(source, name)
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise ValueError(f"source integrity failed for {name}")
    subprocess.run(
        [
            "Rscript",
            str(REDUCER),
            "--rna",
            str(rna),
            "--adt",
            str(adt),
            "--annotations",
            str(annotations),
            "--aliases",
            str(ALIASES),
            "--output",
            str(output),
        ],
        check=True,
    )


def _load_reduced(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    with gzip.open(path / "cells.tsv.gz", "rt") as stream:
        cells = pd.read_csv(stream, sep="\t")
    markers = pd.read_csv(path / "markers.tsv", sep="\t")
    with gzip.open(path / "rna_matched.mtx.gz", "rb") as stream:
        rna = mmread(stream).tocsr().astype(float)
    with gzip.open(path / "adt_all.mtx.gz", "rb") as stream:
        adt_all = mmread(stream).tocsr().astype(float)
    if rna.shape != (len(markers), len(cells)) or adt_all.shape[1] != len(cells):
        raise ValueError("reduced matrix dimensions do not match metadata")
    adt_rows = markers["adt_row"].to_numpy(dtype=int)
    if np.any(adt_rows < 0) or np.any(adt_rows >= adt_all.shape[0]):
        raise ValueError("ADT row index is out of range")
    rna_total = cells["rna_total"].to_numpy(dtype=float)
    if np.any(rna_total <= 0.0):
        raise ValueError("RNA library totals must be positive")
    rna_values = np.log1p(
        rna.toarray() / rna_total[None, :] * 10_000.0
    )
    adt_log = np.log1p(adt_all.toarray())
    adt_values = adt_log[adt_rows] - adt_log.mean(axis=0, keepdims=True)
    return cells, markers, rna_values, adt_values


def _state_thresholds(
    cells: pd.DataFrame, rna: np.ndarray, adt: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    donor = cells["donor"].astype(str).to_numpy()
    condition = cells["condition"].astype(str).to_numpy()
    cell_type = cells["cell_type"].astype(str).to_numpy()
    calibration = (
        np.isin(donor, DEVELOPMENT)
        & (condition == "Baseline")
        & np.isin(cell_type, [item[1] for item in CONTRASTS])
    )
    if np.count_nonzero(calibration) < 100:
        raise ValueError("development baseline calibration set is too small")
    rna_cut = np.quantile(rna[:, calibration], [1 / 3, 2 / 3], axis=1).T
    adt_cut = np.quantile(adt[:, calibration], [1 / 3, 2 / 3], axis=1).T
    rna_state = np.sum(rna[:, :, None] >= rna_cut[:, None, :], axis=2)
    adt_state = np.sum(adt[:, :, None] >= adt_cut[:, None, :], axis=2)
    eligible = (rna_cut[:, 0] < rna_cut[:, 1]) & (adt_cut[:, 0] < adt_cut[:, 1])
    for state in range(3):
        eligible &= np.mean(rna_state[:, calibration] == state, axis=1) >= 0.05
        eligible &= np.mean(adt_state[:, calibration] == state, axis=1) >= 0.05
    if np.count_nonzero(eligible) < 12:
        raise ValueError("fewer than 12 markers pass the frozen development support rule")
    return eligible, rna_state, adt_state


def _table(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.bincount(first * 3 + second, minlength=9).reshape(3, 3).astype(float)


def _canonical_states(
    row_margin: np.ndarray, column_margin: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(row_margin, dtype=int)
    columns = np.asarray(column_margin, dtype=int)
    if (
        rows.shape != (3,)
        or columns.shape != (3,)
        or np.any(rows <= 0)
        or np.any(columns <= 0)
        or rows.sum() != columns.sum()
    ):
        raise ValueError("every frozen table must have three positive matched margins")
    return np.repeat(np.arange(3), rows), np.repeat(np.arange(3), columns)


def _margin_stats(
    first: np.ndarray, second: np.ndarray, seed: int
) -> dict[str, object]:
    rows = np.bincount(first, minlength=3).astype(int)
    columns = np.bincount(second, minlength=3).astype(int)
    canonical_first, canonical_second = _canonical_states(rows, columns)
    basis = helmert_contrast(3)
    endpoint = np.concatenate(
        (
            basis.T @ (rows / rows.sum()),
            basis.T @ (columns / columns.sum()),
        )
    )
    field_reference = conditional_association_coordinates(
        canonical_first,
        canonical_second,
        first_levels=3,
        second_levels=3,
        pseudocount=PSEUDOCOUNT,
        permutations=PERMUTATIONS,
        seed=seed,
    )
    references: dict[str, object] = {
        "rows": rows.astype(float),
        "columns": columns.astype(float),
        "total": float(rows.sum()),
        "endpoint": endpoint,
        "field_null": field_reference.null_mean_coordinates,
        "field_destroyed": field_reference.destroyed_coordinates,
        "field_variance": field_reference.null_variance_coordinates,
    }
    for residual in ("pearson", "deviance"):
        estimate = conditional_poisson_residuals(
            canonical_first,
            canonical_second,
            first_levels=3,
            second_levels=3,
            residual=residual,
            permutations=PERMUTATIONS,
            seed=seed,
        )
        references[f"{residual}_null"] = estimate.null_mean_coordinates
        references[f"{residual}_destroyed"] = (
            estimate.destroyed_coordinates / np.sqrt(rows.sum())
        )
        references[f"{residual}_variance"] = (
            estimate.null_variance_coordinates / rows.sum()
        )
    return references


def _table_stats(
    first: np.ndarray, second: np.ndarray, seed: int
) -> dict[str, object]:
    table = _table(first, second)
    margins = _margin_stats(first, second, seed)
    n = float(margins["total"])
    probability = table / n
    basis = helmert_contrast(3)
    covariance = basis.T @ (
        probability - np.outer(probability.sum(axis=1), probability.sum(axis=0))
    ) @ basis
    field_raw = association_coordinates(
        association_field(table, pseudocount=PSEUDOCOUNT)
    )
    result: dict[str, object] = {
        **margins,
        "table": table,
        "field": field_raw - np.asarray(margins["field_null"]),
        "field_raw": field_raw,
        "covariance": covariance,
    }
    for residual in ("pearson", "deviance"):
        raw = poisson_independence_residuals(table, residual=residual)
        result[residual] = (
            raw - np.asarray(margins[f"{residual}_null"])
        ) / np.sqrt(n)
        result[f"{residual}_raw"] = raw
    return result


def _build_fields(
    cells: pd.DataFrame,
    marker_names: list[str],
    rna_state: np.ndarray,
    adt_state: np.ndarray,
    *,
    open_held_pairing: bool,
) -> dict[str, np.ndarray]:
    donors = DEVELOPMENT + HELD
    field_shape = (len(donors), len(marker_names), len(CONTRASTS), 2, 2)
    residual_shape = (len(donors), len(marker_names), len(CONTRASTS), 3, 3)
    scalar_shape = (len(donors), len(marker_names), len(CONTRASTS))
    arrays = {
        "field": np.full(field_shape, np.nan),
        "field_destroyed": np.full(field_shape, np.nan),
        "field_variance": np.full(field_shape, np.nan),
        "covariance": np.full(field_shape, np.nan),
        "pearson": np.full(residual_shape, np.nan),
        "pearson_destroyed": np.full(residual_shape, np.nan),
        "pearson_variance": np.full(residual_shape, np.nan),
        "deviance": np.full(residual_shape, np.nan),
        "deviance_destroyed": np.full(residual_shape, np.nan),
        "deviance_variance": np.full(residual_shape, np.nan),
        "baseline_field_raw": np.full(field_shape, np.nan),
        "baseline_field_null": np.full(field_shape, np.nan),
        "stimulus_field_null": np.full(field_shape, np.nan),
        "baseline_covariance": np.full(field_shape, np.nan),
        "baseline_pearson_raw": np.full(residual_shape, np.nan),
        "baseline_deviance_raw": np.full(residual_shape, np.nan),
        "baseline_pearson_null": np.full(residual_shape, np.nan),
        "baseline_deviance_null": np.full(residual_shape, np.nan),
        "stimulus_pearson_null": np.full(residual_shape, np.nan),
        "stimulus_deviance_null": np.full(residual_shape, np.nan),
        "baseline_total": np.full(scalar_shape, np.nan),
        "stimulus_total": np.full(scalar_shape, np.nan),
        "stimulus_rows": np.full((*scalar_shape, 3), np.nan),
        "stimulus_columns": np.full((*scalar_shape, 3), np.nan),
        "endpoint": np.full((*scalar_shape, 4), np.nan),
        "stimulus_table": np.full((*scalar_shape, 3, 3), np.nan),
    }
    donor_values = cells["donor"].astype(str).to_numpy()
    condition = cells["condition"].astype(str).to_numpy()
    cell_type = cells["cell_type"].astype(str).to_numpy()
    for donor_index, donor in enumerate(donors):
        for marker_index, marker in enumerate(marker_names):
            for contrast_index, (stimulus, lineage) in enumerate(CONTRASTS):
                common = (donor_values == donor) & (cell_type == lineage)
                baseline_mask = common & (condition == "Baseline")
                stimulus_mask = common & (condition == stimulus)
                if min(np.count_nonzero(baseline_mask), np.count_nonzero(stimulus_mask)) < 15:
                    raise ValueError("frozen donor-arm support unexpectedly failed")
                key = f"{donor}\0{marker}\0{stimulus}\0{lineage}"
                baseline = _table_stats(
                    rna_state[marker_index, baseline_mask],
                    adt_state[marker_index, baseline_mask],
                    _seed("baseline", key),
                )
                stimulus_first = rna_state[marker_index, stimulus_mask]
                stimulus_second = adt_state[marker_index, stimulus_mask]
                stimulus_seed = _seed("stimulus", key)
                stimulus_margins = _margin_stats(
                    stimulus_first, stimulus_second, stimulus_seed
                )
                held_sealed = donor in HELD and not open_held_pairing
                challenged = (
                    None
                    if held_sealed
                    else _table_stats(stimulus_first, stimulus_second, stimulus_seed)
                )
                location = donor_index, marker_index, contrast_index
                arrays["endpoint"][location] = np.asarray(
                    stimulus_margins["endpoint"]
                ) - np.asarray(baseline["endpoint"])
                arrays["baseline_field_raw"][location] = baseline["field_raw"]
                arrays["baseline_field_null"][location] = baseline["field_null"]
                arrays["stimulus_field_null"][location] = stimulus_margins[
                    "field_null"
                ]
                arrays["baseline_covariance"][location] = baseline["covariance"]
                arrays["baseline_total"][location] = baseline["total"]
                arrays["stimulus_total"][location] = stimulus_margins["total"]
                arrays["stimulus_rows"][location] = stimulus_margins["rows"]
                arrays["stimulus_columns"][location] = stimulus_margins["columns"]
                for residual in ("pearson", "deviance"):
                    arrays[f"baseline_{residual}_raw"][location] = baseline[
                        f"{residual}_raw"
                    ]
                    arrays[f"baseline_{residual}_null"][location] = baseline[
                        f"{residual}_null"
                    ]
                    arrays[f"stimulus_{residual}_null"][location] = stimulus_margins[
                        f"{residual}_null"
                    ]
                if held_sealed:
                    continue
                assert challenged is not None
                arrays["field"][location] = challenged["field"] - baseline["field"]
                arrays["field_destroyed"][location] = (
                    challenged["field_destroyed"] - baseline["field_destroyed"]
                )
                arrays["field_variance"][location] = (
                    challenged["field_variance"] + baseline["field_variance"]
                )
                arrays["covariance"][location] = (
                    challenged["covariance"] - baseline["covariance"]
                )
                for residual in ("pearson", "deviance"):
                    arrays[residual][location] = challenged[residual] - baseline[residual]
                    arrays[f"{residual}_destroyed"][location] = (
                        challenged[f"{residual}_destroyed"]
                        - baseline[f"{residual}_destroyed"]
                    )
                    arrays[f"{residual}_variance"][location] = (
                        challenged[f"{residual}_variance"]
                        + baseline[f"{residual}_variance"]
                    )
                arrays["stimulus_table"][location] = challenged["table"]
    return arrays


def _embedding_laplacian(genes: list[str]) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(SCGPT, allow_pickle=False) as archive:
        names = archive["gene_names"].astype(str)
        embedding = np.asarray(archive["embedding"], dtype=float)
    lookup = {name.upper(): index for index, name in enumerate(names)}
    incidence = np.zeros((len(genes), len(genes)), dtype=float)
    covered = [index for index, gene in enumerate(genes) if gene.upper() in lookup]
    if covered:
        values = np.asarray([embedding[lookup[genes[index].upper()]] for index in covered])
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        similarity = values @ values.T
        for local, entity in enumerate(covered):
            neighbors = np.argsort(similarity[local])[::-1][: min(6, len(covered))]
            incidence[[covered[index] for index in neighbors], entity] = 1.0
    for index in set(range(len(genes))) - set(covered):
        incidence[index, index] = 1.0
    return normalized_hypergraph_laplacian(incidence), {
        "path": str(SCGPT.relative_to(ROOT)),
        "sha256": _sha256(SCGPT),
        "covered_markers": len(covered),
    }


def _structured(
    values: np.ndarray,
    variance: np.ndarray,
    laplacian: np.ndarray,
    *, nuclear: float,
    graph: float,
) -> np.ndarray:
    largest = float(np.linalg.svd(values, compute_uv=False)[0])
    fit = fit_structured_coupling_fields(
        values,
        observation_weight=inverse_permutation_variance_weights(variance),
        graph_laplacian=laplacian if graph > 0.0 else None,
        nuclear_penalty=nuclear * largest,
        graph_penalty=graph,
        tolerance=1e-9,
    )
    if not fit.converged:
        raise RuntimeError("the fixed structured fit did not converge")
    return fit.coefficient


def _fit_predictions(
    arrays: dict[str, np.ndarray], genes: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    develop = slice(0, len(DEVELOPMENT))
    held = slice(len(DEVELOPMENT), len(DEVELOPMENT) + len(HELD))
    laplacian, embedding = _embedding_laplacian(genes)

    def mean_flat(name: str) -> np.ndarray:
        return arrays[name][develop].mean(axis=0).reshape(len(genes), -1)

    def variance_flat(name: str) -> np.ndarray:
        return arrays[name][develop].sum(axis=0).reshape(len(genes), -1) / len(DEVELOPMENT) ** 2

    field = mean_flat("field")
    variance = variance_flat("field_variance")
    signal_energy = float(np.sum(field**2))
    scalar = max(
        0.0,
        1.0 - float(np.sum(variance)) / max(signal_energy, np.finfo(float).eps),
    )
    generator = np.random.default_rng(_seed("membership-permuted"))
    permutation = generator.permutation(len(genes))
    permuted_laplacian = laplacian[np.ix_(permutation, permutation)]
    predictions = {
        "field_direct": field,
        "field_zero": np.zeros_like(field),
        "field_scalar": scalar * field,
        "field_nuclear": _structured(field, variance, laplacian, nuclear=0.1, graph=0.0),
        "field_hypergraph": _structured(field, variance, laplacian, nuclear=0.0, graph=5.0),
        "field_membership_permuted": _structured(
            field,
            variance,
            permuted_laplacian,
            nuclear=NUCLEAR_FRACTION,
            graph=GRAPH_PENALTY,
        ),
        "field_primary": _structured(
            field, variance, laplacian, nuclear=NUCLEAR_FRACTION, graph=GRAPH_PENALTY
        ),
        "field_destroyed": mean_flat("field_destroyed"),
        "covariance_direct": mean_flat("covariance"),
    }
    for residual in ("pearson", "deviance"):
        values = mean_flat(residual)
        residual_variance = variance_flat(f"{residual}_variance")
        predictions[f"{residual}_direct"] = values
        predictions[f"{residual}_structured"] = _structured(
            values,
            residual_variance,
            laplacian,
            nuclear=NUCLEAR_FRACTION,
            graph=GRAPH_PENALTY,
        )

    x_train = arrays["endpoint"][develop].reshape(-1, 24)
    y_train = arrays["field"][develop].reshape(-1, 24)
    endpoint = Ridge(alpha=0.1, fit_intercept=True).fit(x_train, y_train)
    predictions["field_endpoint_ridge"] = endpoint.predict(
        arrays["endpoint"][held].reshape(-1, 24)
    ).reshape(len(HELD), len(genes), 24)
    embedding["variance_scalar"] = scalar
    embedding["membership_permutation"] = permutation.tolist()
    return predictions, embedding


def _field_table(
    baseline_raw: np.ndarray,
    baseline_null: np.ndarray,
    stimulus_null: np.ndarray,
    contrast: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    raw_stimulus = baseline_raw + contrast + stimulus_null - baseline_null
    return field_coordinates_to_table(raw_stimulus, rows, columns)


def _covariance_table(
    anchor: np.ndarray,
    contrast: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    total = float(rows.sum())
    seed_probability = (
        np.outer(rows / total, columns / total)
        + field_from_coordinates(anchor + contrast)
    )
    floor = np.finfo(float).eps * max(1.0, total)
    return ipf_to_margins(np.maximum(seed_probability * total, floor), rows, columns)


def _residual_table(
    baseline_raw: np.ndarray,
    baseline_null: np.ndarray,
    stimulus_null: np.ndarray,
    baseline_total: float,
    stimulus_total: float,
    contrast: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    residual: str,
) -> np.ndarray:
    centered_baseline = (baseline_raw - baseline_null) / np.sqrt(baseline_total)
    raw_stimulus = (
        (centered_baseline + contrast) * np.sqrt(stimulus_total) + stimulus_null
    )
    return residual_coordinates_to_table(
        raw_stimulus, rows, columns, residual=residual
    )


def _predict_tables(
    arrays: dict[str, np.ndarray], predictions: dict[str, np.ndarray], markers: int
) -> dict[str, np.ndarray]:
    held_offset = len(DEVELOPMENT)
    predicted_tables = {
        name: np.empty((len(HELD), markers, len(CONTRASTS), 3, 3))
        for name in predictions
    }
    for donor in range(len(HELD)):
        source = held_offset + donor
        for marker in range(markers):
            for contrast in range(len(CONTRASTS)):
                location = source, marker, contrast
                rows = arrays["stimulus_rows"][location]
                columns = arrays["stimulus_columns"][location]
                for method, estimate in predictions.items():
                    if estimate.ndim == 3:
                        flat = estimate[donor, marker]
                    else:
                        flat = estimate[marker]
                    if method.startswith("field_"):
                        delta = flat.reshape(6, 2, 2)[contrast]
                        table = _field_table(
                            arrays["baseline_field_raw"][location],
                            arrays["baseline_field_null"][location],
                            arrays["stimulus_field_null"][location],
                            delta,
                            rows,
                            columns,
                        )
                    elif method.startswith("covariance_"):
                        delta = flat.reshape(6, 2, 2)[contrast]
                        table = _covariance_table(
                            arrays["baseline_covariance"][location],
                            delta,
                            rows,
                            columns,
                        )
                    elif method.startswith("pearson_"):
                        delta = flat.reshape(6, 3, 3)[contrast]
                        table = _residual_table(
                            arrays["baseline_pearson_raw"][location],
                            arrays["baseline_pearson_null"][location],
                            arrays["stimulus_pearson_null"][location],
                            arrays["baseline_total"][location],
                            arrays["stimulus_total"][location],
                            delta,
                            rows,
                            columns,
                            "pearson",
                        )
                    elif method.startswith("deviance_"):
                        delta = flat.reshape(6, 3, 3)[contrast]
                        table = _residual_table(
                            arrays["baseline_deviance_raw"][location],
                            arrays["baseline_deviance_null"][location],
                            arrays["stimulus_deviance_null"][location],
                            arrays["baseline_total"][location],
                            arrays["stimulus_total"][location],
                            delta,
                            rows,
                            columns,
                            "deviance",
                        )
                    else:
                        raise ValueError(f"unknown prediction family: {method}")
                    predicted_tables[method][donor, marker, contrast] = table
    return predicted_tables


def _score_tables(
    arrays: dict[str, np.ndarray], predicted_tables: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    held_truth = arrays["stimulus_table"][len(DEVELOPMENT) :]
    if not np.isfinite(held_truth).all():
        raise ValueError("held stimulus pairing was not opened for scoring")
    losses = {
        name: np.empty(values.shape[:3]) for name, values in predicted_tables.items()
    }
    for name, values in predicted_tables.items():
        for index in np.ndindex(values.shape[:3]):
            losses[name][index] = multinomial_deviance_per_observation(
                held_truth[index], values[index]
            )
    return losses


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    x = np.asarray(first, dtype=float).ravel()
    y = np.asarray(second, dtype=float).ravel()
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _interval(values: np.ndarray) -> list[float]:
    return np.quantile(values, [0.025, 0.975]).tolist()


def _representation_metrics(
    prediction: np.ndarray, truth: np.ndarray
) -> dict[str, float]:
    estimate = np.broadcast_to(prediction, truth.shape)
    denominator = max(float(np.mean(truth**2)), np.finfo(float).eps)
    return {
        "pooled_pearson": _correlation(estimate, truth),
        "standardized_rmse": float(
            np.sqrt(np.mean((estimate - truth) ** 2) / denominator)
        ),
    }


def summarize(
    arrays: dict[str, np.ndarray], predictions: dict[str, np.ndarray], losses: dict[str, np.ndarray]
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    truth = arrays["field"][len(DEVELOPMENT) :].reshape(len(HELD), len(predictions["field_direct"]), 24)
    primary = predictions["field_primary"]
    pooled = _correlation(np.broadcast_to(primary, truth.shape), truth)
    donor_r = [_correlation(primary, truth[index]) for index in range(len(HELD))]
    per_marker_loss = {name: values.mean(axis=(0, 2)) for name, values in losses.items()}
    classical = (
        "pearson_direct",
        "pearson_structured",
        "deviance_direct",
        "deviance_structured",
    )
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
    best_classical = min(classical, key=lambda name: float(per_marker_loss[name].mean()))
    best_matched = min(matched, key=lambda name: float(per_marker_loss[name].mean()))
    rng = np.random.default_rng(SEED)
    correlations = np.empty(BOOTSTRAPS)
    primary_minus_destroyed = np.empty(BOOTSTRAPS)
    primary_minus_classical = np.empty(BOOTSTRAPS)
    primary_minus_matched = np.empty(BOOTSTRAPS)
    marker_count = truth.shape[1]
    for draw in range(BOOTSTRAPS):
        index = rng.integers(0, marker_count, marker_count)
        correlations[draw] = _correlation(
            np.broadcast_to(primary[index], truth[:, index].shape), truth[:, index]
        )
        primary_loss = per_marker_loss["field_primary"][index].mean()
        primary_minus_destroyed[draw] = (
            primary_loss - per_marker_loss["field_destroyed"][index].mean()
        )
        selected = min(classical, key=lambda name: float(per_marker_loss[name][index].mean()))
        primary_minus_classical[draw] = (
            primary_loss - per_marker_loss[selected][index].mean()
        )
        selected_matched = min(
            matched, key=lambda name: float(per_marker_loss[name][index].mean())
        )
        primary_minus_matched[draw] = (
            primary_loss - per_marker_loss[selected_matched][index].mean()
        )
    correlation_ci = _interval(correlations)
    destroyed_ci = _interval(primary_minus_destroyed)
    classical_ci = _interval(primary_minus_classical)
    matched_ci = _interval(primary_minus_matched)
    passed = (
        correlation_ci[0] > 0.0
        and all(value > 0.0 for value in donor_r)
        and destroyed_ci[1] < 0.0
        and classical_ci[1] < 0.0
        and matched_ci[1] < 0.0
    )
    held_slice = slice(len(DEVELOPMENT), None)
    representation = {
        "field_primary": _representation_metrics(primary, truth),
        "field_direct": _representation_metrics(predictions["field_direct"], truth),
    }
    for family in ("pearson", "deviance"):
        residual_truth = arrays[family][held_slice].reshape(
            len(HELD), truth.shape[1], 54
        )
        for method in (f"{family}_direct", f"{family}_structured"):
            representation[method] = _representation_metrics(
                predictions[method], residual_truth
            )
    summary = {
        "primary_method": "field_primary",
        "pooled_held_field_correlation": pooled,
        "pooled_held_field_correlation_bootstrap_95_ci": correlation_ci,
        "held_donor_field_correlations": dict(zip(HELD, donor_r)),
        "mean_per_cell_deviance": {
            name: float(values.mean()) for name, values in losses.items()
        },
        "secondary_representation_metrics": representation,
        "best_classical": best_classical,
        "best_matched_field": best_matched,
        "primary_minus_destroyed_deviance_bootstrap_95_ci": destroyed_ci,
        "primary_minus_best_classical_deviance_bootstrap_95_ci": classical_ci,
        "primary_minus_best_matched_deviance_bootstrap_95_ci": matched_ci,
        "bootstrap_unit": "complete matched RNA-ADT marker",
        "bootstrap_draws": BOOTSTRAPS,
        "gate_passed": passed,
    }
    bootstrap = {
        "pooled_field_correlation": correlations,
        "primary_minus_destroyed_deviance": primary_minus_destroyed,
        "primary_minus_best_classical_deviance": primary_minus_classical,
        "primary_minus_best_matched_deviance": primary_minus_matched,
    }
    return summary, bootstrap


def _analysis_data(
    reduced: Path,
) -> tuple[
    pd.DataFrame,
    list[str],
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    cells, marker_frame, rna, adt = _load_reduced(reduced)
    eligible, rna_state, adt_state = _state_thresholds(cells, rna, adt)
    all_marker_ids = marker_frame["marker_id"].astype(str).to_numpy()
    excluded_marker_ids = all_marker_ids[~eligible]
    marker_frame = marker_frame.loc[eligible].reset_index(drop=True)
    rna_state = rna_state[eligible]
    adt_state = adt_state[eligible]
    marker_names = marker_frame["marker_id"].astype(str).tolist()
    genes = marker_frame["gene_symbol"].astype(str).tolist()
    return (
        cells,
        marker_names,
        genes,
        excluded_marker_ids,
        rna_state,
        adt_state,
    )


def predict(args: argparse.Namespace) -> None:
    provenance = preflight(require_sealed=True)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"prospective prediction already exists: {output}")
    reduced = Path(args.reduced)
    if not args.skip_reducer:
        reduce_inputs(
            rna=Path(args.rna),
            adt=Path(args.adt),
            annotations=Path(args.annotations),
            output=reduced,
        )
    (
        cells,
        marker_names,
        genes,
        excluded_marker_ids,
        rna_state,
        adt_state,
    ) = _analysis_data(reduced)
    sealed_arrays = _build_fields(
        cells,
        marker_names,
        rna_state,
        adt_state,
        open_held_pairing=False,
    )
    predictions, embedding = _fit_predictions(sealed_arrays, genes)
    predicted_tables = _predict_tables(
        sealed_arrays, predictions, len(marker_names)
    )
    record = {
        "schema": "lawlor-hca-pbmc-predictions/1.0",
        "status": "PREDICTIONS_FROZEN_PAIRING_UNOPENED",
        "scope": "held-donor stimulus-by-cell-type RNA-protein coupling; not CRISPR target transfer",
        "marker_ids": marker_names,
        "genes": genes,
        "development_support_exclusions": excluded_marker_ids.tolist(),
        "development_donors": list(DEVELOPMENT),
        "held_donors": list(HELD),
        "contrasts": [f"{stimulus}:{lineage}" for stimulus, lineage in CONTRASTS],
        "held_stimulus_rows": sealed_arrays["stimulus_rows"][
            len(DEVELOPMENT) :
        ].tolist(),
        "held_stimulus_columns": sealed_arrays["stimulus_columns"][
            len(DEVELOPMENT) :
        ].tolist(),
        "predictions": {name: value.tolist() for name, value in predictions.items()},
        "predicted_tables": {
            name: value.tolist() for name, value in predicted_tables.items()
        },
        "embedding": embedding,
        "provenance": provenance,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def _require_score_authorization(
    prediction_path: Path, authorization_path: Path
) -> dict[str, object]:
    authorization = _read_json(authorization_path)
    if authorization.get("status") != "SEALED":
        raise PermissionError("score authorization is not SEALED")
    if authorization.get("outcome_access_authorized") is not True:
        raise PermissionError("score authorization forbids held pairing access")
    if authorization.get("prediction_sha256") != _sha256(prediction_path):
        raise PermissionError("the prediction JSON hash differs from authorization")
    if authorization.get("prediction_bytes") != prediction_path.stat().st_size:
        raise PermissionError("the prediction JSON byte count differs from authorization")
    if not authorization.get("prediction_public_url"):
        raise PermissionError("the prediction hash has no public URL")
    if not authorization.get("prediction_public_commit"):
        raise PermissionError("the prediction hash has no immutable public commit")
    if authorization.get("runner_sha256") != _sha256(Path(__file__)):
        raise PermissionError("the scoring runner differs from authorization")
    return authorization


def _locked_predictions(
    path: Path,
) -> tuple[dict[str, object], dict[str, np.ndarray], dict[str, np.ndarray]]:
    record = _read_json(path)
    if record.get("schema") != "lawlor-hca-pbmc-predictions/1.0":
        raise ValueError("prediction JSON schema differs")
    if record.get("status") != "PREDICTIONS_FROZEN_PAIRING_UNOPENED":
        raise ValueError("prediction JSON is not frozen")
    predictions = {
        name: np.asarray(value, dtype=float)
        for name, value in record["predictions"].items()
    }
    tables = {
        name: np.asarray(value, dtype=float)
        for name, value in record["predicted_tables"].items()
    }
    if set(predictions) != set(tables):
        raise ValueError("locked prediction and table method sets differ")
    field_methods = {
        "field_direct",
        "field_zero",
        "field_scalar",
        "field_nuclear",
        "field_hypergraph",
        "field_membership_permuted",
        "field_primary",
        "field_destroyed",
    }
    residual_methods = {
        "pearson_direct",
        "pearson_structured",
        "deviance_direct",
        "deviance_structured",
    }
    expected_methods = field_methods | residual_methods | {
        "field_endpoint_ridge",
        "covariance_direct",
    }
    if set(predictions) != expected_methods:
        raise ValueError("locked prediction method set differs from the protocol")
    marker_count = len(record["marker_ids"])
    for name, values in predictions.items():
        if name == "field_endpoint_ridge":
            expected_shape = (len(HELD), marker_count, 24)
        elif name in residual_methods:
            expected_shape = (marker_count, 54)
        else:
            expected_shape = (marker_count, 24)
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise ValueError(f"locked prediction shape differs for {name}")
    if any(
        values.shape[:3] != (len(HELD), len(record["marker_ids"]), len(CONTRASTS))
        or values.shape[-2:] != (3, 3)
        or not np.isfinite(values).all()
        or np.any(values <= 0.0)
        for values in tables.values()
    ):
        raise ValueError("locked predicted tables are malformed")
    if record.get("provenance", {}).get("runner_sha256") != _sha256(Path(__file__)):
        raise ValueError("prediction JSON was produced by a different runner")
    return record, predictions, tables


def score(args: argparse.Namespace) -> None:
    provenance = preflight(require_sealed=True)
    prediction_path = Path(args.predictions)
    authorization = _require_score_authorization(
        prediction_path, Path(args.authorization)
    )
    output = Path(args.output)
    scored_path = output.with_name(output.stem + "_arrays.npz")
    if output.exists() or scored_path.exists():
        raise FileExistsError("prospective score output already exists")
    record, predictions, predicted_tables = _locked_predictions(prediction_path)
    (
        cells,
        marker_names,
        genes,
        excluded_marker_ids,
        rna_state,
        adt_state,
    ) = _analysis_data(Path(args.reduced))
    expected_metadata = {
        "marker_ids": marker_names,
        "genes": genes,
        "development_support_exclusions": excluded_marker_ids.tolist(),
        "development_donors": list(DEVELOPMENT),
        "held_donors": list(HELD),
        "contrasts": [f"{stimulus}:{lineage}" for stimulus, lineage in CONTRASTS],
    }
    for name, expected in expected_metadata.items():
        if record.get(name) != expected:
            raise ValueError(f"locked prediction {name} differs from scoring input")
    arrays = _build_fields(
        cells,
        marker_names,
        rna_state,
        adt_state,
        open_held_pairing=True,
    )
    np.testing.assert_allclose(
        np.asarray(record["held_stimulus_rows"]),
        arrays["stimulus_rows"][len(DEVELOPMENT) :],
    )
    np.testing.assert_allclose(
        np.asarray(record["held_stimulus_columns"]),
        arrays["stimulus_columns"][len(DEVELOPMENT) :],
    )
    for values in predicted_tables.values():
        np.testing.assert_allclose(
            values.sum(axis=-1), arrays["stimulus_rows"][len(DEVELOPMENT) :]
        )
        np.testing.assert_allclose(
            values.sum(axis=-2), arrays["stimulus_columns"][len(DEVELOPMENT) :]
        )
    losses = _score_tables(arrays, predicted_tables)
    summary, bootstrap = summarize(arrays, predictions, losses)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        scored_path,
        marker_ids=np.asarray(marker_names),
        genes=np.asarray(genes),
        held_donors=np.asarray(HELD),
        contrasts=np.asarray([f"{stimulus}:{lineage}" for stimulus, lineage in CONTRASTS]),
        held_field_truth=arrays["field"][len(DEVELOPMENT) :],
        **{f"prediction__{name}": value for name, value in predictions.items()},
        **{f"loss__{name}": value for name, value in losses.items()},
        **{f"table__{name}": value for name, value in predicted_tables.items()},
        **{f"bootstrap__{name}": value for name, value in bootstrap.items()},
    )
    result = {
        "schema": "lawlor-hca-pbmc-confirmation/1.0",
        "status": "PROMOTE" if summary["gate_passed"] else "REFUSE",
        "scope": "held-donor stimulus-by-cell-type RNA-protein coupling; not CRISPR target transfer",
        "development_donors": list(DEVELOPMENT),
        "held_donors": list(HELD),
        "contrasts": [f"{stimulus}:{lineage}" for stimulus, lineage in CONTRASTS],
        "matched_markers": marker_names,
        "marker_count": len(marker_names),
        "development_support_exclusions": excluded_marker_ids.tolist(),
        "states": 3,
        "fixed_margin_reference_permutations": PERMUTATIONS,
        "summary": summary,
        "prediction_json": str(prediction_path.relative_to(ROOT)),
        "prediction_json_sha256": _sha256(prediction_path),
        "prediction_public_url": authorization["prediction_public_url"],
        "prediction_public_commit": authorization["prediction_public_commit"],
        "scored_arrays": str(scored_path.relative_to(ROOT)),
        "scored_arrays_sha256": _sha256(scored_path),
        "provenance": provenance,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    prediction = subparsers.add_parser("predict")
    prediction.add_argument("--rna")
    prediction.add_argument("--adt")
    prediction.add_argument("--annotations")
    prediction.add_argument(
        "--reduced", default=str(ROOT / "data/development/lawlor_hca_pbmc/reduced_v1")
    )
    prediction.add_argument("--skip-reducer", action="store_true")
    prediction.add_argument(
        "--output",
        default=str(ROOT / "results/lawlor_hca_pbmc_predictions.json"),
    )
    scoring = subparsers.add_parser("score")
    scoring.add_argument(
        "--reduced", default=str(ROOT / "data/development/lawlor_hca_pbmc/reduced_v1")
    )
    scoring.add_argument(
        "--predictions",
        default=str(ROOT / "results/lawlor_hca_pbmc_predictions.json"),
    )
    scoring.add_argument(
        "--authorization",
        default=str(
            ROOT
            / "benchmark_release/coupling_fields_v1/LAWLOR_SCORE_AUTHORIZATION.json"
        ),
    )
    scoring.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(preflight(require_sealed=False), indent=2))
        return
    if args.command == "predict" and not args.skip_reducer and not all(
        (args.rna, args.adt, args.annotations)
    ):
        parser.error("--rna, --adt, and --annotations are required unless --skip-reducer")
    if args.command == "predict":
        predict(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
