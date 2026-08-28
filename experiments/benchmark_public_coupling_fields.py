"""Benchmark marginal-invariant coupling fields on public paired assays."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zlib
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import fisher_exact
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from mapreg.classical_residuals import conditional_poisson_residual_coordinates
from mapreg.coupling_fields import (
    conditional_association_coordinates,
    fit_structured_coupling_fields,
    helmert_contrast,
    inverse_permutation_variance_weights,
    normalized_hypergraph_laplacian,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PUBLIC = DATA / "external/public_scaleup"
CACHE = DATA / "development/public_coupling_atlas"
OUTPUT = ROOT / "results/public_coupling_atlas_benchmark.json"
SEED = 20260826
PSEUDOCOUNT = 0.5
BOOTSTRAPS = 2_000
DESTROYED_DRAWS = 64
FINAL_NUCLEAR_FRACTION = 0.1
FINAL_GRAPH_PENALTY = 5.0

FRANGIEH_FEATURES = {
    "CD117": "KIT",
    "CD119": "IFNGR1",
    "CD140a": "PDGFRA",
    "CD140b": "PDGFRB",
    "CD172a": "SIRPA",
    "CD184": "CXCR4",
    "CD202b": "TEK",
    "CD274": "CD274",
    "CD29": "ITGB1",
    "CD309": "KDR",
    "CD44": "CD44",
    "CD47": "CD47",
    "CD49f": "ITGA6",
    "CD58": "CD58",
    "CD59": "CD59",
    "CD61": "ITGB3",
    "HLA_A": "HLA-A",
    "HLA_E": "HLA-E",
    "CD9": "CD9",
}
PAPALEXI_FEATURES = {
    "CD86": "CD86",
    "PDL1": "CD274",
    "PDL2": "PDCD1LG2",
    "CD366": "HAVCR2",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode() if isinstance(value, bytes) else str(value) for value in values]
    )


def _column(handle: h5py.File, path: str) -> np.ndarray:
    object_ = handle[path]
    if isinstance(object_, h5py.Group):
        categories = _decode(object_["categories"][:])
        codes = np.asarray(object_["codes"][:], dtype=int)
        result = np.full(len(codes), "__missing__", dtype=object)
        present = codes >= 0
        result[present] = categories[codes[present]]
        return result.astype(str)
    if object_.dtype.kind in "OSU":
        return _decode(object_[:])
    return np.asarray(object_[:])


def _read_log_features(
    path: Path,
    feature_column: str,
    requested: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        if handle["X"].attrs.get("encoding-type") != "csc_matrix":
            raise ValueError(f"{path.name} is not stored as cell-by-feature CSC")
        names = _column(handle, feature_column).astype(str)
        lookup = {name: index for index, name in enumerate(names)}
        missing = sorted(set(requested) - set(lookup))
        if missing:
            raise ValueError(f"{path.name} lacks requested features: {missing}")
        cells = (
            _column(handle, "obs/cell_name").astype(str)
            if "cell_name" in handle["obs"]
            else _column(handle, "obs/_index").astype(str)
        )
        totals = np.asarray(handle["obs/ncounts"][:], dtype=float)
        valid = totals > 0.0
        denominator = totals.copy()
        denominator[~valid] = 1.0
        matrix = np.zeros((len(cells), len(requested)), dtype=np.float32)
        group = handle["X"]
        indptr = np.asarray(group["indptr"][:], dtype=np.int64)
        for column_index, feature in enumerate(requested):
            source = lookup[feature]
            start, stop = indptr[source : source + 2]
            rows = np.asarray(group["indices"][start:stop], dtype=np.int64)
            counts = np.asarray(group["data"][start:stop], dtype=float)
            matrix[rows, column_index] = np.log1p(
                counts / denominator[rows] * 10_000.0
            )
    return cells, matrix, np.asarray(requested, dtype=str), valid


def _stable_mod(values: np.ndarray, modulus: int, salt: str) -> np.ndarray:
    return np.asarray(
        [zlib.crc32((salt + "|" + value).encode()) % modulus for value in values],
        dtype=int,
    )


def _finite(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    return array


def _fit_state_encoder(
    first: np.ndarray,
    second: np.ndarray,
    calibration: np.ndarray,
    state_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if calibration.sum() < 10 * state_count:
        raise ValueError("state calibration subset is too small")

    def encode(values: np.ndarray, offset: int):
        center = values[calibration].mean(axis=0)
        scale = values[calibration].std(axis=0)
        scale[scale < 1e-6] = 1.0
        standardized = (values - center) / scale
        components = min(6, standardized.shape[1], int(calibration.sum()) - 1)
        pca = PCA(n_components=components, svd_solver="full")
        pca.fit(standardized[calibration])
        coordinates = pca.transform(standardized)
        cluster = KMeans(
            n_clusters=state_count,
            n_init=50,
            random_state=seed + offset,
        )
        cluster.fit(coordinates[calibration])
        return cluster.predict(coordinates), pca.explained_variance_ratio_

    first_state, first_variance = encode(first, 1)
    second_state, second_variance = encode(second, 2)
    return first_state, second_state, {
        "calibration_cells": int(calibration.sum()),
        "first_explained_variance": first_variance.tolist(),
        "second_explained_variance": second_variance.tolist(),
    }


def prepare_frangieh_cache(path: Path) -> dict[str, object]:
    rna_path = PUBLIC / "FrangiehIzar2021_RNA.h5ad"
    protein_path = PUBLIC / "FrangiehIzar2021_protein.h5ad"
    with h5py.File(rna_path, "r") as handle:
        cells = _column(handle, "obs/cell_name").astype(str)
        target = _column(handle, "obs/perturbation").astype(str)
        guide = _column(handle, "obs/sgRNA").astype(str)
        context = _column(handle, "obs/perturbation_2").astype(str)
        nperts = np.asarray(handle["obs/nperts"][:], dtype=int)
    protein_names = list(FRANGIEH_FEATURES)
    rna_names = [FRANGIEH_FEATURES[name] for name in protein_names]
    rna_cells, rna, _, rna_valid = _read_log_features(
        rna_path, "var/gene_symbol", rna_names
    )
    protein_cells, protein, _, protein_valid = _read_log_features(
        protein_path, "var/protein", protein_names
    )
    if not (np.array_equal(cells, rna_cells) and np.array_equal(cells, protein_cells)):
        raise ValueError("Frangieh RNA, protein, and metadata cell orders differ")

    calibration = (
        (target == "control")
        & (context == "Control")
        & rna_valid
        & protein_valid
        & (_stable_mod(cells, 5, "frangieh-calibration") == 0)
    )
    first_state, second_state, encoder = _fit_state_encoder(
        rna, protein, calibration, 3, SEED + 100
    )
    eligible_cell = ((target == "control") & (nperts == 0)) | (
        (target != "control") & (nperts == 1)
    )
    eligible_cell &= rna_valid & protein_valid
    eligible_cell &= ~calibration
    contexts = ("Control", "IFN\u03b3", "Co-culture")
    target_names = []
    target_guides = []
    attrition = []
    for name in sorted(set(target) - {"control"}):
        candidates = sorted(
            set(guide[eligible_cell & (target == name)]) - {"nan", "__missing__"}
        )
        counts = {
            candidate: [
                int(
                    np.count_nonzero(
                        eligible_cell
                        & (target == name)
                        & (guide == candidate)
                        & (context == condition)
                    )
                )
                for condition in contexts
            ]
            for candidate in candidates
        }
        retained = [candidate for candidate in candidates if min(counts[candidate]) >= 20]
        retained = sorted(retained, key=lambda value: (-min(counts[value]), value))[:3]
        passed = len(retained) == 3
        attrition.append(
            {
                "target": name,
                "eligible": passed,
                "guides_with_20_cells_per_context": len(
                    [candidate for candidate in candidates if min(counts[candidate]) >= 20]
                ),
            }
        )
        if passed:
            target_names.append(name)
            target_guides.append(sorted(retained))
    if len(target_names) < 100:
        raise ValueError("Frangieh target eligibility unexpectedly collapsed")
    control_unit = _stable_mod(cells, 3, "frangieh-control-unit")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cells=cells,
        target=target,
        guide=guide,
        context=context,
        eligible_cell=eligible_cell,
        control_unit=control_unit,
        first_state=first_state,
        second_state=second_state,
        target_names=np.asarray(target_names),
        target_guides=np.asarray(target_guides),
        protein_features=np.asarray(protein_names),
        rna_features=np.asarray(rna_names),
    )
    return {
        "cache": str(path.relative_to(ROOT)),
        "cache_sha256": _sha256(path),
        "eligible_targets": len(target_names),
        "encoder": encoder,
        "attrition": attrition,
    }


def _papalexi_target(guide: str) -> str:
    if guide.lower().startswith(("ntg", "control")):
        return "control"
    return re.sub(r"g\d+$", "", guide)


def prepare_papalexi_cache(path: Path) -> dict[str, object]:
    rna_path = PUBLIC / "PapalexiSatija2021_eccite_RNA.h5ad"
    protein_path = PUBLIC / "PapalexiSatija2021_eccite_protein.h5ad"
    with h5py.File(rna_path, "r") as handle:
        cells = _column(handle, "obs/_index").astype(str)
        guides = _column(handle, "obs/perturbation").astype(str)
        replicate = _column(handle, "obs/hto").astype(str)
    target = np.asarray([_papalexi_target(value) for value in guides])
    protein_names = list(PAPALEXI_FEATURES)
    rna_names = [PAPALEXI_FEATURES[name] for name in protein_names]
    rna_cells, rna, _, rna_valid = _read_log_features(
        rna_path, "var/gene_symbol", rna_names
    )
    protein_cells, protein, _, protein_valid = _read_log_features(
        protein_path, "var/protein", protein_names
    )
    if not (np.array_equal(cells, rna_cells) and np.array_equal(cells, protein_cells)):
        raise ValueError("Papalexi RNA, protein, and metadata cell orders differ")
    retained_replicates = np.asarray(["rep1-tx", "rep3-tx", "rep4-tx"])
    calibration = (
        (target == "control")
        & np.isin(replicate, retained_replicates)
        & rna_valid
        & protein_valid
        & (_stable_mod(cells, 5, "papalexi-calibration") == 0)
    )
    first_state, second_state, encoder = _fit_state_encoder(
        rna, protein, calibration, 3, SEED + 200
    )
    eligible_cell = (
        np.isin(replicate, retained_replicates)
        & rna_valid
        & protein_valid
        & ~calibration
    )
    target_names = []
    attrition = []
    for name in sorted(set(target) - {"control"}):
        counts = [
            int(np.count_nonzero(eligible_cell & (target == name) & (replicate == unit)))
            for unit in retained_replicates
        ]
        passed = min(counts) >= 20
        attrition.append({"target": name, "eligible": passed, "replicate_cells": counts})
        if passed:
            target_names.append(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cells=cells,
        target=target,
        guide=guides,
        replicate=replicate,
        eligible_cell=eligible_cell,
        first_state=first_state,
        second_state=second_state,
        target_names=np.asarray(target_names),
        replicate_names=retained_replicates,
        protein_features=np.asarray(protein_names),
        rna_features=np.asarray(rna_names),
    )
    return {
        "cache": str(path.relative_to(ROOT)),
        "cache_sha256": _sha256(path),
        "eligible_targets": len(target_names),
        "encoder": encoder,
        "attrition": attrition,
    }


def _table_statistics(
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray,
    state_count: int,
    seed: int,
) -> tuple[np.ndarray, ...]:
    indices = np.flatnonzero(mask)
    if len(indices) < 2:
        raise ValueError("each paired table needs at least two cells")
    first_values = first[indices]
    second_values = second[indices]

    table = np.bincount(
        first_values * state_count + second_values,
        minlength=state_count**2,
    ).reshape(state_count, state_count)
    probability = table / table.sum()
    basis = helmert_contrast(state_count)
    covariance_table = probability - np.outer(
        probability.sum(axis=1), probability.sum(axis=0)
    )
    covariance = (basis.T @ covariance_table @ basis).ravel()
    endpoint = np.concatenate(
        (basis.T @ probability.sum(axis=1), basis.T @ probability.sum(axis=0))
    )
    estimate = conditional_association_coordinates(
        first_values,
        second_values,
        first_levels=state_count,
        second_levels=state_count,
        pseudocount=PSEUDOCOUNT,
        permutations=DESTROYED_DRAWS,
        seed=seed,
    )
    poisson = conditional_poisson_residual_coordinates(
        first_values,
        second_values,
        first_levels=state_count,
        second_levels=state_count,
        residual="deviance",
        permutations=DESTROYED_DRAWS,
        seed=seed,
    )
    field = estimate.coordinates.ravel()
    destroyed = estimate.destroyed_coordinates.ravel()
    variance = estimate.null_variance_coordinates.ravel()
    return (
        field,
        covariance,
        endpoint,
        destroyed,
        variance,
        len(indices),
        poisson.coordinates.ravel(),
        poisson.destroyed_coordinates.ravel(),
        poisson.null_variance_coordinates.ravel(),
    )


def _build_perturbsci_fields() -> tuple[dict[str, np.ndarray], list[str], dict[str, object]]:
    path = DATA / "development/perturbsci_kinetics_gse218566/frozen_states_v1.npz"
    with np.load(path, allow_pickle=False) as cache:
        target = cache["target_genes"].astype(str)
        guide = cache["guides"].astype(str)
        first = cache["old_state"].astype(int)
        second = cache["new_state"].astype(int)
        targets = cache["eligible_targets"].astype(str).tolist()
    control = "NO-TARGET"
    control_guides = sorted(np.unique(guide[target == control]).tolist())
    target_guides = {
        name: sorted(np.unique(guide[target == name]).tolist()) for name in targets
    }
    arrays = {
        name: np.empty((3, len(targets), 4), dtype=float)
        for name in (
            "field",
            "covariance",
            "destroyed",
            "variance",
            "poisson_deviance",
            "poisson_deviance_destroyed",
            "poisson_deviance_variance",
        )
    }
    endpoint = np.empty((3, len(targets), 4), dtype=float)
    support = np.empty((3, len(targets)), dtype=int)
    for unit in range(3):
        selected_control = [
            value for position, value in enumerate(control_guides) if position % 3 == unit
        ]
        control_stats = _table_statistics(
            first,
            second,
            np.isin(guide, selected_control),
            3,
            SEED + 1_000 + unit,
        )
        for target_index, name in enumerate(targets):
            stats = _table_statistics(
                first,
                second,
                guide == target_guides[name][unit],
                3,
                SEED + 2_000 + 100 * unit + target_index,
            )
            arrays["field"][unit, target_index] = stats[0] - control_stats[0]
            arrays["covariance"][unit, target_index] = stats[1] - control_stats[1]
            endpoint[unit, target_index] = stats[2] - control_stats[2]
            arrays["destroyed"][unit, target_index] = stats[3] - control_stats[3]
            arrays["variance"][unit, target_index] = stats[4] + control_stats[4]
            arrays["poisson_deviance"][unit, target_index] = (
                stats[6] - control_stats[6]
            )
            arrays["poisson_deviance_destroyed"][unit, target_index] = (
                stats[7] - control_stats[7]
            )
            arrays["poisson_deviance_variance"][unit, target_index] = (
                stats[8] + control_stats[8]
            )
            support[unit, target_index] = stats[5]
    arrays["endpoint"] = endpoint
    return arrays, targets, {
        "source": str(path.relative_to(ROOT)),
        "source_sha256": _sha256(path),
        "units": "three sequence-distinct guide rotations",
        "minimum_target_unit_cells": int(support.min()),
    }


def _build_frangieh_fields(path: Path) -> tuple[dict[str, np.ndarray], list[str], dict[str, object]]:
    with np.load(path, allow_pickle=False) as cache:
        target = cache["target"].astype(str)
        guide = cache["guide"].astype(str)
        context = cache["context"].astype(str)
        eligible = cache["eligible_cell"].astype(bool)
        control_unit = cache["control_unit"].astype(int)
        first = cache["first_state"].astype(int)
        second = cache["second_state"].astype(int)
        targets = cache["target_names"].astype(str).tolist()
        target_guides = cache["target_guides"].astype(str)
    contexts = ("Control", "IFN\u03b3", "Co-culture")
    outputs = {
        name: np.empty((3, len(targets), 8), dtype=float)
        for name in (
            "field",
            "covariance",
            "destroyed",
            "variance",
            "poisson_deviance",
            "poisson_deviance_destroyed",
            "poisson_deviance_variance",
        )
    }
    endpoint = np.empty((3, len(targets), 8), dtype=float)
    support = np.empty((3, len(targets), 3), dtype=int)
    for unit in range(3):
        control_stats = []
        for context_index, condition in enumerate(contexts):
            control_stats.append(
                _table_statistics(
                    first,
                    second,
                    eligible
                    & (target == "control")
                    & (context == condition)
                    & (control_unit == unit),
                    3,
                    SEED + 10_000 + 100 * unit + context_index,
                )
            )
        for target_index, name in enumerate(targets):
            target_stats = []
            for context_index, condition in enumerate(contexts):
                stats = _table_statistics(
                    first,
                    second,
                    eligible
                    & (target == name)
                    & (guide == target_guides[target_index, unit])
                    & (context == condition),
                    3,
                    SEED + 20_000 + 10_000 * unit + 10 * target_index + context_index,
                )
                target_stats.append(stats)
                support[unit, target_index, context_index] = stats[5]
            for key, statistic_index in (
                ("field", 0),
                ("covariance", 1),
                ("endpoint", 2),
                ("destroyed", 3),
                ("poisson_deviance", 6),
                ("poisson_deviance_destroyed", 7),
            ):
                contrasts = []
                for challenged in (1, 2):
                    contrasts.append(
                        (target_stats[challenged][statistic_index] - target_stats[0][statistic_index])
                        - (control_stats[challenged][statistic_index] - control_stats[0][statistic_index])
                    )
                value = np.concatenate(contrasts)
                if key == "endpoint":
                    endpoint[unit, target_index] = value
                else:
                    outputs[key][unit, target_index] = value
            outputs["variance"][unit, target_index] = np.concatenate(
                [
                    target_stats[challenged][4]
                    + target_stats[0][4]
                    + control_stats[challenged][4]
                    + control_stats[0][4]
                    for challenged in (1, 2)
                ]
            )
            outputs["poisson_deviance_variance"][unit, target_index] = np.concatenate(
                [
                    target_stats[challenged][8]
                    + target_stats[0][8]
                    + control_stats[challenged][8]
                    + control_stats[0][8]
                    for challenged in (1, 2)
                ]
            )
    outputs["endpoint"] = endpoint
    return outputs, targets, {
        "source": str(path.relative_to(ROOT)),
        "source_sha256": _sha256(path),
        "units": "three sequence-distinct guides per target",
        "minimum_target_context_unit_cells": int(support.min()),
        "contexts": list(contexts),
    }


def _build_papalexi_fields(path: Path) -> tuple[dict[str, np.ndarray], list[str], dict[str, object]]:
    with np.load(path, allow_pickle=False) as cache:
        target = cache["target"].astype(str)
        replicate = cache["replicate"].astype(str)
        eligible = cache["eligible_cell"].astype(bool)
        first = cache["first_state"].astype(int)
        second = cache["second_state"].astype(int)
        targets = cache["target_names"].astype(str).tolist()
        units = cache["replicate_names"].astype(str).tolist()
    outputs = {
        name: np.empty((3, len(targets), 4), dtype=float)
        for name in (
            "field",
            "covariance",
            "destroyed",
            "variance",
            "poisson_deviance",
            "poisson_deviance_destroyed",
            "poisson_deviance_variance",
        )
    }
    endpoint = np.empty((3, len(targets), 4), dtype=float)
    support = np.empty((3, len(targets)), dtype=int)
    for unit_index, unit in enumerate(units):
        control_stats = _table_statistics(
            first,
            second,
            eligible & (target == "control") & (replicate == unit),
            3,
            SEED + 30_000 + unit_index,
        )
        for target_index, name in enumerate(targets):
            stats = _table_statistics(
                first,
                second,
                eligible & (target == name) & (replicate == unit),
                3,
                SEED + 40_000 + 1_000 * unit_index + target_index,
            )
            outputs["field"][unit_index, target_index] = stats[0] - control_stats[0]
            outputs["covariance"][unit_index, target_index] = stats[1] - control_stats[1]
            endpoint[unit_index, target_index] = stats[2] - control_stats[2]
            outputs["destroyed"][unit_index, target_index] = stats[3] - control_stats[3]
            outputs["variance"][unit_index, target_index] = stats[4] + control_stats[4]
            outputs["poisson_deviance"][unit_index, target_index] = (
                stats[6] - control_stats[6]
            )
            outputs["poisson_deviance_destroyed"][unit_index, target_index] = (
                stats[7] - control_stats[7]
            )
            outputs["poisson_deviance_variance"][unit_index, target_index] = (
                stats[8] + control_stats[8]
            )
            support[unit_index, target_index] = stats[5]
    outputs["endpoint"] = endpoint
    return outputs, targets, {
        "source": str(path.relative_to(ROOT)),
        "source_sha256": _sha256(path),
        "units": units,
        "minimum_target_replicate_cells": int(support.min()),
    }


def _embedding_laplacian(
    targets: list[str], path: Path, *, neighbors: int = 5
) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as archive:
        names = archive["gene_names"].astype(str)
        embedding = np.asarray(archive["embedding"], dtype=float)
    lookup = {name.upper(): index for index, name in enumerate(names)}
    present_target = [index for index, name in enumerate(targets) if name.upper() in lookup]
    incidence = np.zeros((len(targets), len(present_target)), dtype=float)
    if present_target:
        values = np.asarray([embedding[lookup[targets[index].upper()]] for index in present_target])
        norms = np.linalg.norm(values, axis=1)
        values = np.divide(values, norms[:, None], out=np.zeros_like(values), where=norms[:, None] > 0)
        similarity = np.einsum("ik,jk->ij", values, values, optimize=False)
        _finite("embedding similarity", similarity)
        for column, local_index in enumerate(range(len(present_target))):
            order = np.argsort(similarity[local_index])[::-1]
            members = [present_target[position] for position in order[: min(neighbors + 1, len(order))]]
            incidence[members, column] = 1.0
        laplacian = normalized_hypergraph_laplacian(incidence)
    else:
        laplacian = np.zeros((len(targets), len(targets)))
    return laplacian, {
        "embedding": str(path.relative_to(ROOT)),
        "embedding_sha256": _sha256(path),
        "covered_targets": len(present_target),
        "hyperedges": int(incidence.shape[1]),
        "neighbors_per_hyperedge": neighbors,
    }


def _structured_prediction(
    train: np.ndarray,
    development_first: np.ndarray,
    development_second: np.ndarray,
    laplacian: np.ndarray | None,
    mode: str,
) -> tuple[np.ndarray, dict[str, float | int]]:
    nuclear_grid = (0.0, 0.01, 0.03, 0.1, 0.3, 0.6, 1.0)
    graph_grid = (0.0, 0.05, 0.2, 1.0, 5.0, 20.0)
    if mode == "nuclear":
        candidates = [(value, 0.0) for value in nuclear_grid]
    elif mode == "graph":
        candidates = [(0.0, value) for value in graph_grid]
    elif mode == "full":
        candidates = [(first, second) for first in nuclear_grid for second in graph_grid]
    else:
        raise ValueError(f"unknown structured mode: {mode}")
    if mode in {"graph", "full"} and laplacian is None:
        raise ValueError("graph modes require a Laplacian")

    def fit(values: np.ndarray, nuclear_fraction: float, graph_penalty: float):
        largest = float(np.linalg.svd(values, compute_uv=False)[0])
        return fit_structured_coupling_fields(
            values,
            graph_laplacian=laplacian,
            nuclear_penalty=nuclear_fraction * largest,
            graph_penalty=graph_penalty,
            tolerance=1e-9,
        )

    scored = []
    for nuclear_fraction, graph_penalty in candidates:
        first_fit = fit(development_first, nuclear_fraction, graph_penalty)
        second_fit = fit(development_second, nuclear_fraction, graph_penalty)
        loss = 0.5 * (
            np.mean((first_fit.coefficient - development_second) ** 2)
            + np.mean((second_fit.coefficient - development_first) ** 2)
        )
        scored.append((float(loss), nuclear_fraction, graph_penalty))
    loss, nuclear_fraction, graph_penalty = min(scored)
    final = fit(train, nuclear_fraction, graph_penalty)
    return final.coefficient, {
        "development_mse": loss,
        "nuclear_fraction": nuclear_fraction,
        "graph_penalty": graph_penalty,
        "effective_rank": final.effective_rank,
        "iterations": final.iterations,
    }


def _fixed_weighted_prediction(
    train: np.ndarray,
    variance: np.ndarray,
    laplacian: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    weight = inverse_permutation_variance_weights(variance)
    largest = float(np.linalg.svd(train, compute_uv=False)[0])
    fit = fit_structured_coupling_fields(
        train,
        observation_weight=weight,
        graph_laplacian=laplacian,
        nuclear_penalty=FINAL_NUCLEAR_FRACTION * largest,
        graph_penalty=FINAL_GRAPH_PENALTY,
        tolerance=1e-9,
    )
    return fit.coefficient, {
        "nuclear_fraction": FINAL_NUCLEAR_FRACTION,
        "graph_penalty": FINAL_GRAPH_PENALTY,
        "minimum_weight": float(weight.min()),
        "maximum_weight": float(weight.max()),
        "effective_rank": fit.effective_rank,
        "iterations": fit.iterations,
    }


def _scalar_prediction(
    train: np.ndarray, development_first: np.ndarray, development_second: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    candidates = np.linspace(0.0, 1.25, 26)
    losses = [
        0.5
        * (
            np.mean((scale * development_first - development_second) ** 2)
            + np.mean((scale * development_second - development_first) ** 2)
        )
        for scale in candidates
    ]
    selected = float(candidates[int(np.argmin(losses))])
    return selected * train, {
        "development_mse": float(min(losses)),
        "scale": selected,
    }


def _endpoint_prediction(
    train_field: np.ndarray,
    held_endpoint: np.ndarray,
    train_endpoint: np.ndarray,
    development_field_first: np.ndarray,
    development_field_second: np.ndarray,
    development_endpoint_first: np.ndarray,
    development_endpoint_second: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    candidates = (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0)
    losses = []
    for penalty in candidates:
        first = Ridge(alpha=penalty).fit(
            development_endpoint_first, development_field_first
        )
        second = Ridge(alpha=penalty).fit(
            development_endpoint_second, development_field_second
        )
        losses.append(
            0.5
            * (
                np.mean(
                    (first.predict(development_endpoint_second) - development_field_second) ** 2
                )
                + np.mean(
                    (second.predict(development_endpoint_first) - development_field_first) ** 2
                )
            )
        )
    selected = float(candidates[int(np.argmin(losses))])
    model = Ridge(alpha=selected).fit(train_endpoint, train_field)
    return model.predict(held_endpoint), {
        "development_mse": float(min(losses)),
        "ridge_penalty": selected,
    }


def _marginal_residual_prediction(
    train_field: np.ndarray,
    held_endpoint: np.ndarray,
    train_endpoint: np.ndarray,
    development_field_first: np.ndarray,
    development_field_second: np.ndarray,
    development_endpoint_first: np.ndarray,
    development_endpoint_second: np.ndarray,
    laplacian: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    _, endpoint_selection = _endpoint_prediction(
        train_field,
        held_endpoint,
        train_endpoint,
        development_field_first,
        development_field_second,
        development_endpoint_first,
        development_endpoint_second,
    )
    penalty = float(endpoint_selection["ridge_penalty"])
    first_model = Ridge(alpha=penalty).fit(
        development_endpoint_first, development_field_first
    )
    second_model = Ridge(alpha=penalty).fit(
        development_endpoint_second, development_field_second
    )
    first_residual = development_field_first - first_model.predict(
        development_endpoint_first
    )
    second_residual = development_field_second - second_model.predict(
        development_endpoint_second
    )
    final_model = Ridge(alpha=penalty).fit(train_endpoint, train_field)
    train_residual = train_field - final_model.predict(train_endpoint)
    residual_prediction, residual_selection = _structured_prediction(
        train_residual,
        first_residual,
        second_residual,
        laplacian,
        "full",
    )
    prediction = final_model.predict(held_endpoint) + residual_prediction
    return prediction, {
        "endpoint": endpoint_selection,
        "residual": residual_selection,
        "residual_to_field_norm": float(
            np.linalg.norm(residual_prediction) / max(np.linalg.norm(prediction), 1e-12)
        ),
    }


def _field_metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float | None]:
    estimate_flat = estimate.ravel()
    truth_flat = truth.ravel()
    correlation = (
        float(np.corrcoef(estimate_flat, truth_flat)[0, 1])
        if np.std(estimate_flat) > 0.0 and np.std(truth_flat) > 0.0
        else None
    )
    estimate_norm = np.linalg.norm(estimate, axis=-1)
    truth_norm = np.linalg.norm(truth, axis=-1)
    cosine = np.divide(
        np.sum(estimate * truth, axis=-1),
        estimate_norm * truth_norm,
        out=np.zeros_like(estimate_norm),
        where=(estimate_norm * truth_norm) > 0.0,
    )
    normalized_rmse = float(
        np.sqrt(
            np.mean((estimate - truth) ** 2)
            / max(float(np.mean(truth**2)), 1e-12)
        )
    )
    return {
        "pooled_pearson": correlation,
        "macro_target_cosine": float(cosine.mean()),
        "standardized_rmse": normalized_rmse,
    }


def _bootstrap_metrics(
    estimate: np.ndarray, truth: np.ndarray, seed: int
) -> dict[str, list[float] | None]:
    rng = np.random.default_rng(seed)
    values = {name: [] for name in _field_metrics(estimate, truth)}
    for _ in range(BOOTSTRAPS):
        indices = rng.integers(0, estimate.shape[1], size=estimate.shape[1])
        metrics = _field_metrics(estimate[:, indices], truth[:, indices])
        for name, value in metrics.items():
            if value is not None:
                values[name].append(value)
    return {
        name: (
            [float(value) for value in np.quantile(samples, [0.025, 0.975])]
            if samples
            else None
        )
        for name, samples in values.items()
    }


def _bootstrap_metric_difference(
    first: np.ndarray,
    second: np.ndarray,
    truth: np.ndarray,
    metric: str,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(BOOTSTRAPS):
        indices = rng.integers(0, truth.shape[1], size=truth.shape[1])
        first_value = _field_metrics(first[:, indices], truth[:, indices])[metric]
        second_value = _field_metrics(second[:, indices], truth[:, indices])[metric]
        if first_value is not None and second_value is not None:
            differences.append(float(first_value - second_value))
    return [float(value) for value in np.quantile(differences, [0.025, 0.975])]


def _bootstrap_coordinate_family_difference(
    first: np.ndarray,
    first_truth: np.ndarray,
    second: np.ndarray,
    second_truth: np.ndarray,
    metric: str,
    seed: int,
) -> list[float]:
    """Compare coordinate families on the same held units and target draws."""

    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(BOOTSTRAPS):
        indices = rng.integers(0, first_truth.shape[1], size=first_truth.shape[1])
        first_value = _field_metrics(
            first[:, indices], first_truth[:, indices]
        )[metric]
        second_value = _field_metrics(
            second[:, indices], second_truth[:, indices]
        )[metric]
        if first_value is not None and second_value is not None:
            differences.append(float(first_value - second_value))
    return [float(value) for value in np.quantile(differences, [0.025, 0.975])]


def _run_panel(
    name: str,
    arrays: dict[str, np.ndarray],
    targets: list[str],
) -> dict[str, object]:
    if arrays["field"].shape[0] != 3:
        raise ValueError("the benchmark requires exactly three held units")
    for key, values in arrays.items():
        _finite(f"{name} {key}", values)
    scgpt_laplacian, scgpt = _embedding_laplacian(
        targets, DATA / "scgpt_gene_embeddings.npz"
    )
    gene2vec_laplacian, gene2vec = _embedding_laplacian(
        targets, DATA / "gene2vec_embeddings.npz"
    )
    permutation = np.random.default_rng(SEED + 500).permutation(len(targets))
    shuffled_laplacian = scgpt_laplacian[np.ix_(permutation, permutation)]
    methods = (
        "zero",
        "direct",
        "scalar_shrinkage",
        "nuclear",
        "scgpt_hypergraph",
        "nuclear_scgpt_hypergraph",
        "variance_weighted_nuclear_scgpt_hypergraph",
        "nuclear_gene2vec_hypergraph",
        "shuffled_hypergraph",
        "endpoint_ridge",
        "marginal_residual_atlas",
        "linear_cross_covariance",
        "destroyed_links",
    )
    predictions = {method: [] for method in methods}
    poisson_predictions: dict[str, list[np.ndarray]] = {
        "direct": [],
        "fixed_structured": [],
        "destroyed_links": [],
    }
    truths = []
    poisson_truths = []
    selections = []
    for held in range(3):
        training = [unit for unit in range(3) if unit != held]
        first, second = training
        development_first = arrays["field"][first]
        development_second = arrays["field"][second]
        train = 0.5 * (development_first + development_second)
        truth = arrays["field"][held]
        truths.append(truth)
        fold_selection: dict[str, object] = {"held_unit": held}

        poisson_first = arrays["poisson_deviance"][first]
        poisson_second = arrays["poisson_deviance"][second]
        poisson_train = 0.5 * (poisson_first + poisson_second)
        poisson_truths.append(arrays["poisson_deviance"][held])
        poisson_predictions["direct"].append(poisson_train)
        poisson_structured, fold_selection["poisson_deviance_fixed_structured"] = (
            _fixed_weighted_prediction(
                poisson_train,
                0.25
                * (
                    arrays["poisson_deviance_variance"][first]
                    + arrays["poisson_deviance_variance"][second]
                ),
                scgpt_laplacian,
            )
        )
        poisson_predictions["fixed_structured"].append(poisson_structured)
        poisson_predictions["destroyed_links"].append(
            0.5
            * (
                arrays["poisson_deviance_destroyed"][first]
                + arrays["poisson_deviance_destroyed"][second]
            )
        )

        predictions["zero"].append(np.zeros_like(train))
        predictions["direct"].append(train)
        predictions["scalar_shrinkage"].append(
            _scalar_prediction(train, development_first, development_second)[0]
        )
        nuclear, fold_selection["nuclear"] = _structured_prediction(
            train, development_first, development_second, None, "nuclear"
        )
        predictions["nuclear"].append(nuclear)
        graph, fold_selection["scgpt_hypergraph"] = _structured_prediction(
            train,
            development_first,
            development_second,
            scgpt_laplacian,
            "graph",
        )
        predictions["scgpt_hypergraph"].append(graph)
        full, fold_selection["nuclear_scgpt_hypergraph"] = _structured_prediction(
            train,
            development_first,
            development_second,
            scgpt_laplacian,
            "full",
        )
        predictions["nuclear_scgpt_hypergraph"].append(full)
        weighted, fold_selection["variance_weighted_nuclear_scgpt_hypergraph"] = (
            _fixed_weighted_prediction(
                train,
                0.25 * (arrays["variance"][first] + arrays["variance"][second]),
                scgpt_laplacian,
            )
        )
        predictions["variance_weighted_nuclear_scgpt_hypergraph"].append(weighted)
        gene2vec_full, fold_selection["nuclear_gene2vec_hypergraph"] = (
            _structured_prediction(
                train,
                development_first,
                development_second,
                gene2vec_laplacian,
                "full",
            )
        )
        predictions["nuclear_gene2vec_hypergraph"].append(gene2vec_full)
        shuffled, fold_selection["shuffled_hypergraph"] = _structured_prediction(
            train,
            development_first,
            development_second,
            shuffled_laplacian,
            "full",
        )
        predictions["shuffled_hypergraph"].append(shuffled)
        endpoint, fold_selection["endpoint_ridge"] = _endpoint_prediction(
            train,
            arrays["endpoint"][held],
            0.5 * (arrays["endpoint"][first] + arrays["endpoint"][second]),
            development_first,
            development_second,
            arrays["endpoint"][first],
            arrays["endpoint"][second],
        )
        predictions["endpoint_ridge"].append(endpoint)
        residual, fold_selection["marginal_residual_atlas"] = (
            _marginal_residual_prediction(
                train,
                arrays["endpoint"][held],
                0.5 * (arrays["endpoint"][first] + arrays["endpoint"][second]),
                development_first,
                development_second,
                arrays["endpoint"][first],
                arrays["endpoint"][second],
                scgpt_laplacian,
            )
        )
        predictions["marginal_residual_atlas"].append(residual)
        predictions["linear_cross_covariance"].append(
            0.5 * (arrays["covariance"][first] + arrays["covariance"][second])
        )
        predictions["destroyed_links"].append(
            0.5 * (arrays["destroyed"][first] + arrays["destroyed"][second])
        )
        selections.append(fold_selection)
    truth_array = np.asarray(truths)
    poisson_truth_array = np.asarray(poisson_truths)
    method_results = {}
    for method_index, method in enumerate(methods):
        estimate = np.asarray(predictions[method])
        _finite(f"{name} {method} prediction", estimate)
        method_results[method] = {
            "metrics": _field_metrics(estimate, truth_array),
            "target_bootstrap_95_ci": _bootstrap_metrics(
                estimate, truth_array, SEED + 10_000 * (method_index + 1)
            ),
        }
    poisson_method_results = {}
    for method_index, (method, values) in enumerate(poisson_predictions.items()):
        estimate = np.asarray(values)
        _finite(f"{name} Poisson deviance {method} prediction", estimate)
        poisson_method_results[method] = {
            "metrics": _field_metrics(estimate, poisson_truth_array),
            "target_bootstrap_95_ci": _bootstrap_metrics(
                estimate,
                poisson_truth_array,
                SEED + 700_000 + 10_000 * (method_index + 1),
            ),
        }
    primary = "variance_weighted_nuclear_scgpt_hypergraph"
    competitors = [
        "zero",
        "direct",
        "scalar_shrinkage",
        "nuclear",
        "scgpt_hypergraph",
        "nuclear_scgpt_hypergraph",
        "endpoint_ridge",
        "linear_cross_covariance",
    ]
    primary_rmse = method_results[primary]["metrics"]["standardized_rmse"]
    best_competitor = min(
        competitors,
        key=lambda method: method_results[method]["metrics"]["standardized_rmse"],
    )
    best_rmse = method_results[best_competitor]["metrics"]["standardized_rmse"]
    primary_prediction = np.asarray(predictions[primary])
    poisson_primary_prediction = np.asarray(
        poisson_predictions["fixed_structured"]
    )
    best_prediction = np.asarray(predictions[best_competitor])
    destroyed_prediction = np.asarray(predictions["destroyed_links"])
    loss_difference_interval = _bootstrap_metric_difference(
        primary_prediction,
        best_prediction,
        truth_array,
        "standardized_rmse",
        SEED + 910_000,
    )
    correlation_difference_interval = _bootstrap_metric_difference(
        primary_prediction,
        destroyed_prediction,
        truth_array,
        "pooled_pearson",
        SEED + 920_000,
    )
    primary_correlation_interval = method_results[primary]["target_bootstrap_95_ci"][
        "pooled_pearson"
    ]
    return {
        "panel": name,
        "targets": len(targets),
        "field_coordinates": int(arrays["field"].shape[-1]),
        "methods": method_results,
        "selection": selections,
        "embedding_hypergraphs": {"scgpt": scgpt, "gene2vec": gene2vec},
        "coordinate_family_comparison": {
            "protocol": (
                "same state tables, fixed-margin permutations, held-unit splits, "
                "target bootstraps, and fixed weighted nuclear-plus-scGPT-hypergraph "
                "estimator; each family is scored against its own held coordinates"
            ),
            "coupling_field": {
                "direct": method_results["direct"],
                "fixed_structured": method_results[primary],
                "destroyed_links": method_results["destroyed_links"],
            },
            "poisson_deviance_residual": poisson_method_results,
            "fixed_structured_coupling_minus_poisson": {
                "pooled_pearson": float(
                    method_results[primary]["metrics"]["pooled_pearson"]
                    - poisson_method_results["fixed_structured"]["metrics"][
                        "pooled_pearson"
                    ]
                ),
                "pooled_pearson_target_bootstrap_95_ci": (
                    _bootstrap_coordinate_family_difference(
                        primary_prediction,
                        truth_array,
                        poisson_primary_prediction,
                        poisson_truth_array,
                        "pooled_pearson",
                        SEED + 930_000,
                    )
                ),
                "standardized_rmse": float(
                    method_results[primary]["metrics"]["standardized_rmse"]
                    - poisson_method_results["fixed_structured"]["metrics"][
                        "standardized_rmse"
                    ]
                ),
                "standardized_rmse_target_bootstrap_95_ci": (
                    _bootstrap_coordinate_family_difference(
                        primary_prediction,
                        truth_array,
                        poisson_primary_prediction,
                        poisson_truth_array,
                        "standardized_rmse",
                        SEED + 940_000,
                    )
                ),
            },
        },
        "primary_vs_best_matched_competitor": {
            "best_competitor": best_competitor,
            "primary_standardized_rmse": primary_rmse,
            "best_competitor_standardized_rmse": best_rmse,
            "relative_rmse_reduction": float((best_rmse - primary_rmse) / best_rmse),
            "primary_wins": bool(primary_rmse < best_rmse),
            "primary_minus_best_standardized_rmse_bootstrap_95_ci": loss_difference_interval,
        },
        "pairing_signal": {
            "primary_minus_destroyed_pearson_bootstrap_95_ci": correlation_difference_interval,
            "detected": bool(
                primary_correlation_interval is not None
                and primary_correlation_interval[0] > 0.0
                and correlation_difference_interval[0] > 0.0
            ),
        },
        "truth": truth_array,
        "primary_prediction": primary_prediction,
    }


def _bh(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values, dtype=float)
    running = 1.0
    for rank_index in range(len(order) - 1, -1, -1):
        index = order[rank_index]
        running = min(running, float(p_values[index]) * len(order) / (rank_index + 1))
        adjusted[index] = running
    return adjusted


def _annotation_sets(targets: list[str], family: str) -> list[tuple[str, set[str]]]:
    universe = set(targets)
    annotations = []
    if family == "reactome":
        with (DATA / "ReactomePathways.gmt").open() as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                members = set(fields[2:]) & universe
                if 3 <= len(members) <= 30:
                    annotations.append((fields[0], members))
    elif family == "corum":
        with (DATA / "corum_humanComplexes_current.txt").open() as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                if row.get("organism") != "Human":
                    continue
                members = set(row["subunits_gene_name"].split(";")) & universe
                if 2 <= len(members) <= 20:
                    annotations.append((row["complex_name"], members))
    else:
        raise ValueError(f"unknown annotation family: {family}")
    return annotations


def _enrichment(panel: dict[str, object], targets: list[str]) -> dict[str, object]:
    truth = panel.pop("truth")
    prediction = panel.pop("primary_prediction")
    target_score = np.mean(np.linalg.norm(truth, axis=-1), axis=0)
    predicted_score = np.mean(np.linalg.norm(prediction, axis=-1), axis=0)
    selected_count = max(5, int(np.ceil(0.2 * len(targets))))
    observed_hits = set(np.asarray(targets)[np.argsort(target_score)[-selected_count:]])
    predicted_hits = set(np.asarray(targets)[np.argsort(predicted_score)[-selected_count:]])
    result = {}
    for family in ("reactome", "corum"):
        annotations = _annotation_sets(targets, family)
        rows = []
        for name, members in annotations:
            overlap = len(observed_hits & predicted_hits & members)
            joint_hits = observed_hits & predicted_hits
            table = [
                [overlap, len(joint_hits - members)],
                [len(members - joint_hits), len(set(targets) - members - joint_hits)],
            ]
            p_value = float(fisher_exact(table, alternative="greater").pvalue)
            rows.append(
                {
                    "annotation": name,
                    "members": len(members),
                    "replicated_top_field_members": overlap,
                    "p_value": p_value,
                }
            )
        if rows:
            adjusted = _bh(np.asarray([row["p_value"] for row in rows]))
            for row, q_value in zip(rows, adjusted):
                row["q_value"] = float(q_value)
            rows.sort(key=lambda row: (row["q_value"], row["p_value"], row["annotation"]))
        result[family] = {
            "tested_annotations": len(rows),
            "bh_significant": int(sum(row["q_value"] < 0.05 for row in rows)),
            "top_results": rows[:10],
        }
    result["definition"] = (
        "one-sided Fisher enrichment among targets in the top field-norm quintile "
        "in both held truth and primary prediction; BH correction within family"
    )
    return result


def run(output: Path, rebuild_cache: bool) -> dict[str, object]:
    frangieh_cache = CACHE / "frangieh_states_v1.npz"
    papalexi_cache = CACHE / "papalexi_states_v1.npz"
    preparation = {}
    if rebuild_cache or not frangieh_cache.exists():
        preparation["frangieh"] = prepare_frangieh_cache(frangieh_cache)
    else:
        preparation["frangieh"] = {
            "cache": str(frangieh_cache.relative_to(ROOT)),
            "cache_sha256": _sha256(frangieh_cache),
        }
    if rebuild_cache or not papalexi_cache.exists():
        preparation["papalexi"] = prepare_papalexi_cache(papalexi_cache)
    else:
        preparation["papalexi"] = {
            "cache": str(papalexi_cache.relative_to(ROOT)),
            "cache_sha256": _sha256(papalexi_cache),
        }

    panels = []
    sources = {}
    builders = (
        ("PerturbSci-Kinetics", _build_perturbsci_fields),
        ("Frangieh Perturb-CITE-seq", lambda: _build_frangieh_fields(frangieh_cache)),
        ("Papalexi ECCITE-seq", lambda: _build_papalexi_fields(papalexi_cache)),
    )
    # NumPy 2.0.2 linked against Accelerate can emit false floating-point
    # warnings for finite BLAS products. Every benchmark input and prediction is
    # checked explicitly above, so silence only those status-flag warnings here.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for name, builder in builders:
            arrays, targets, source = builder()
            panel = _run_panel(name, arrays, targets)
            panel["biology"] = _enrichment(panel, targets)
            panels.append(panel)
            sources[name] = source

    primary_wins = sum(
        panel["primary_vs_best_matched_competitor"]["primary_wins"] for panel in panels
    )
    result = {
        "schema": "v2p2r.public-coupling-atlas-benchmark/1",
        "seed": SEED,
        "pseudocount": PSEUDOCOUNT,
        "target_bootstrap_draws": BOOTSTRAPS,
        "destroyed_link_draws": DESTROYED_DRAWS,
        "preanalysis_contract": "docs/PUBLIC_COUPLING_ATLAS_BENCHMARK_FREEZE_2026-08-26.md",
        "preanalysis_contract_sha256": _sha256(
            ROOT / "docs/PUBLIC_COUPLING_ATLAS_BENCHMARK_FREEZE_2026-08-26.md"
        ),
        "preparation": preparation,
        "sources": sources,
        "panels": panels,
        "promotion_gate": {
            "required_primary_wins": 2,
            "observed_primary_wins": int(primary_wins),
            "passed": bool(primary_wins >= 2),
        },
        "implementation": {
            "benchmark_sha256": _sha256(Path(__file__)),
            "estimator_sha256": _sha256(ROOT / "mapreg/coupling_fields.py"),
            "classical_residuals_sha256": _sha256(
                ROOT / "mapreg/classical_residuals.py"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--rebuild-cache", action="store_true")
    arguments = parser.parse_args()
    result = run(arguments.output, arguments.rebuild_cache)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "panels": [panel["panel"] for panel in result["panels"]],
                "promotion_gate": result["promotion_gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
