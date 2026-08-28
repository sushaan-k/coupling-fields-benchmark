"""Prospective held-donor RNA-protein confirmation on the Lawlor PBMC study."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse

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
DESIGNATION = ROOT / "data/confirmation/pbmc_citeseq_hca/candidate_designation_v2.json"
SCORE_AUTHORIZATION = (
    ROOT / "data/confirmation/pbmc_citeseq_hca/score_authorization_v2.json"
)
AUTHORIZATION_TEMPLATE = (
    ROOT
    / "data/confirmation/pbmc_citeseq_hca/score_authorization_template_v2.json"
)
SCORE_RELEASE = ROOT / "data/confirmation/pbmc_citeseq_hca/score_release_v2.json"
SOURCE_MANIFEST = ROOT / "data/development/lawlor_hca_pbmc/source_manifest_v1.json"
SUPPORT = ROOT / "data/development/lawlor_hca_pbmc/metadata_support_v1.json"
ALIASES = ROOT / "data/development/lawlor_hca_pbmc/adt_gene_aliases_v1.tsv"
REDUCER = ROOT / "experiments/reduce_lawlor_hca_pbmc.R"
SCGPT = ROOT / "data/scgpt_gene_embeddings.npz"
OUTPUT = ROOT / "results/lawlor_hca_pbmc_confirmation.json"
ARRAYS_PATH = ROOT / "results/lawlor_hca_pbmc_confirmation_arrays.npz"
PREDICTION_PATH = ROOT / "results/lawlor_hca_pbmc_predictions.json"
REDUCED_PATH = ROOT / "data/development/lawlor_hca_pbmc/reduced_v2"
TEST_PATH = ROOT / "tests/test_lawlor_hca_pbmc_confirmation.py"
IMPLEMENTATION_FILES = (
    ROOT / "mapreg/coupling_fields.py",
    ROOT / "mapreg/classical_residuals.py",
    ROOT / "mapreg/table_prediction.py",
)
REDUCER_OUTPUTS = (
    "adt_all.mtx.gz",
    "adt_features.tsv",
    "cells.tsv.gz",
    "markers.tsv",
    "rna_matched.mtx.gz",
)
REDUCER_MANIFEST = "reducer_manifest.tsv"

SEED = 20260827
PERMUTATIONS = 64
BOOTSTRAPS = 2_000
PSEUDOCOUNT = 0.5
NUCLEAR_FRACTION = 0.1
GRAPH_PENALTY = 5.0
EXTERNAL_NEIGHBORS = 6
MINIMUM_ENTITY_PAIRS = 12
MINIMUM_MARKER_CLUSTERS = 12
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


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"path must be inside the repository: {path}") from error


def _reducer_artifact_bundle(reduced: Path) -> dict[str, object]:
    reduced = reduced.resolve()
    reduced_root = _repo_relative(reduced)
    if not reduced.is_dir() or reduced.is_symlink():
        raise ValueError("reduced output must be a real directory")

    manifest_path = reduced / REDUCER_MANIFEST
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("reducer manifest is missing or is a symlink")
    manifest = pd.read_csv(
        manifest_path,
        sep="\t",
        dtype={"path": str, "sha256": str},
    )
    if list(manifest.columns) != ["path", "bytes", "sha256"]:
        raise ValueError("reducer manifest columns differ from the protocol")
    if manifest["path"].duplicated().any():
        raise ValueError("reducer manifest paths are not unique")
    if set(manifest["path"]) != set(REDUCER_OUTPUTS):
        raise ValueError("reducer manifest output set differs from the protocol")

    entries = list(reduced.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ValueError("reduced directory contains a symbolic link")
    actual_files = {
        path.relative_to(reduced).as_posix() for path in entries if path.is_file()
    }
    expected_files = set(REDUCER_OUTPUTS) | {REDUCER_MANIFEST}
    if actual_files != expected_files:
        raise ValueError("reduced directory contains an unexpected file set")

    records: list[dict[str, object]] = []
    by_path = manifest.set_index("path")
    for relative in REDUCER_OUTPUTS:
        artifact = reduced / relative
        if not artifact.is_file() or artifact.is_symlink():
            raise ValueError(f"reducer output is missing or is a symlink: {relative}")
        declared_bytes = by_path.at[relative, "bytes"]
        if not isinstance(declared_bytes, (int, np.integer)):
            raise ValueError(
                f"reducer manifest byte count is not an integer: {relative}"
            )
        declared_sha = by_path.at[relative, "sha256"]
        if re.fullmatch(r"[0-9a-f]{64}", declared_sha) is None:
            raise ValueError(f"reducer manifest SHA-256 is malformed: {relative}")
        observed = {
            "path": relative,
            "bytes": artifact.stat().st_size,
            "sha256": _sha256(artifact),
        }
        if (
            observed["bytes"] != int(declared_bytes)
            or observed["sha256"] != declared_sha
        ):
            raise ValueError(f"reducer manifest does not match output: {relative}")
        records.append(observed)

    records.append(
        {
            "path": REDUCER_MANIFEST,
            "bytes": manifest_path.stat().st_size,
            "sha256": _sha256(manifest_path),
        }
    )
    return {
        "schema": "lawlor-reducer-artifact-bundle/1.0",
        "reduced_root": reduced_root,
        "artifacts": records,
    }


def _require_reducer_artifact_bundle(
    reduced: Path, expected: object
) -> dict[str, object]:
    try:
        observed = _reducer_artifact_bundle(reduced)
    except (OSError, ValueError) as error:
        raise PermissionError(
            "reducer artifacts do not satisfy the frozen prediction provenance"
        ) from error
    if observed != expected:
        raise PermissionError(
            "reducer artifact paths, byte counts, or SHA-256 values differ from the frozen prediction provenance"
        )
    return observed


def _require_github_commit_url(
    url: object, commit: object, *, blob_path: str | None
) -> None:
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise PermissionError(
            "public commit must be exactly 40 lowercase hexadecimal characters"
        )
    if not isinstance(url, str):
        raise PermissionError("public URL is missing")
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise PermissionError("public URL must be an immutable GitHub URL")
    parts = unquote(parsed.path).strip("/").split("/")
    route = "blob" if blob_path is not None else "commit"
    minimum = 5 if blob_path is not None else 4
    if len(parts) < minimum or parts[2] != route or parts[3] != commit:
        raise PermissionError("public URL does not contain the authorized commit")
    if not parts[0] or not parts[1]:
        raise PermissionError("public URL lacks a GitHub owner or repository")
    if blob_path is None:
        if len(parts) != 4:
            raise PermissionError("public freeze URL is not an exact GitHub commit URL")
    elif "/".join(parts[4:]) != blob_path:
        raise PermissionError(
            "public prediction URL does not contain the exact prediction path"
        )


def preflight(*, require_sealed: bool) -> dict[str, object]:
    designation = _read_json(DESIGNATION)
    if designation.get("schema") != "lawlor-pbmc-coupling-candidate-designation/2.0":
        raise ValueError("candidate designation schema differs from version 2.0")
    if require_sealed:
        if designation["status"] != "SEALED":
            raise PermissionError("candidate designation is not SEALED")
        if not designation["outcome_access_authorized"]:
            raise PermissionError("outcome access is not authorized")
        if not designation.get("public_freeze_commit"):
            raise PermissionError("a public freeze commit is required")
        if not designation.get("public_freeze_url"):
            raise PermissionError("a public freeze URL is required")
        _require_github_commit_url(
            designation["public_freeze_url"],
            designation["public_freeze_commit"],
            blob_path=None,
        )
    expected = {
        "protocol_sha256": _sha256(PROTOCOL),
        "source_manifest_sha256": _sha256(SOURCE_MANIFEST),
        "metadata_support_sha256": _sha256(SUPPORT),
        "alias_sha256": _sha256(ALIASES),
        "embedding_sha256": _sha256(SCGPT),
        "authorization_template_sha256": _sha256(AUTHORIZATION_TEMPLATE),
        "test_sha256": _sha256(TEST_PATH),
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
    frozen_paths = {
        "prediction_path": _repo_relative(PREDICTION_PATH),
        "reduced_path": _repo_relative(REDUCED_PATH),
        "score_authorization": _repo_relative(SCORE_AUTHORIZATION),
        "score_release": _repo_relative(SCORE_RELEASE),
        "score_output": _repo_relative(OUTPUT),
        "score_arrays": _repo_relative(ARRAYS_PATH),
    }
    for key, observed in frozen_paths.items():
        if designation.get(key) != observed:
            raise ValueError(f"designation {key} is stale")
    return {
        "designation_status": designation["status"],
        "outcome_access_authorized": designation["outcome_access_authorized"],
        **bound_code,
        **frozen_paths,
        "implementation_sha256": implementation_sha,
        **expected,
        "designation_sha256": _sha256(DESIGNATION),
    }


def _write_refusal(path: Path, *, stage: str, error: Exception) -> dict[str, object]:
    if path.exists():
        raise FileExistsError(f"refusal output already exists: {path}")
    message = str(error).replace(str(Path.home()), "~").replace(str(ROOT), ".")
    record = {
        "schema": "lawlor-hca-pbmc-refusal/2.0",
        "status": "REFUSE_EXECUTION",
        "stage": stage,
        "exception_type": type(error).__name__,
        "exception_message": message,
        "protocol_sha256": _sha256(PROTOCOL),
        "runner_sha256": _sha256(Path(__file__)),
        "designation_sha256": _sha256(DESIGNATION),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


def _source_path(manifest: dict, name: str) -> dict:
    return next(record for record in manifest["files"] if record["name"] == name)


def reduce_inputs(*, rna: Path, adt: Path, annotations: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"reducer output path already exists: {output}")
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


def _load_reduced(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
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
    rna_values = np.log1p(rna.toarray() / rna_total[None, :] * 10_000.0)
    adt_log = np.log1p(adt_all.toarray())
    adt_values = adt_log[adt_rows] - adt_log.mean(axis=0, keepdims=True)
    return cells, markers, rna_values, adt_values


def _minimum_marginal_fraction(
    states: np.ndarray,
    masks: list[np.ndarray],
) -> float:
    fractions = [
        np.bincount(states[mask], minlength=3).min() / np.count_nonzero(mask)
        for mask in masks
    ]
    return float(min(fractions))


def _state_thresholds(
    cells: pd.DataFrame,
    markers: pd.DataFrame,
    rna: np.ndarray,
    adt: np.ndarray,
) -> tuple[pd.DataFrame, list[dict[str, object]], np.ndarray, np.ndarray]:
    """Freeze lineage-specific states and select pairs from marginal support only."""

    donor = cells["donor"].astype(str).to_numpy()
    condition = cells["condition"].astype(str).to_numpy()
    cell_type = cells["cell_type"].astype(str).to_numpy()
    entities: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    rna_states: list[np.ndarray] = []
    adt_states: list[np.ndarray] = []

    for marker_index, marker in markers.reset_index(drop=True).iterrows():
        marker_id = str(marker["marker_id"])
        gene = str(marker["gene_symbol"])
        for contrast_index, (stimulus, lineage) in enumerate(CONTRASTS):
            contrast_id = f"{stimulus}:{lineage}"
            entity_id = f"{marker_id}::{contrast_id}"
            calibration = (
                np.isin(donor, DEVELOPMENT)
                & (condition == "Baseline")
                & (cell_type == lineage)
            )
            if np.count_nonzero(calibration) < 15 * len(DEVELOPMENT):
                raise ValueError(
                    f"development baseline calibration support failed for {contrast_id}"
                )
            rna_cut = np.quantile(rna[marker_index, calibration], [1 / 3, 2 / 3])
            adt_cut = np.quantile(adt[marker_index, calibration], [1 / 3, 2 / 3])
            rna_state = np.sum(
                rna[marker_index, :, None] >= rna_cut[None, :], axis=1
            )
            adt_state = np.sum(
                adt[marker_index, :, None] >= adt_cut[None, :], axis=1
            )

            development_masks: list[np.ndarray] = []
            held_masks: list[np.ndarray] = []
            for unit, destination in (
                *((value, development_masks) for value in DEVELOPMENT),
                *((value, held_masks) for value in HELD),
            ):
                common = (donor == unit) & (cell_type == lineage)
                destination.extend(
                    [
                        common & (condition == "Baseline"),
                        common & (condition == stimulus),
                    ]
                )
            if any(np.count_nonzero(mask) < 15 for mask in development_masks + held_masks):
                raise ValueError(f"frozen donor-arm support failed for {contrast_id}")

            development_rna = _minimum_marginal_fraction(
                rna_state, development_masks
            )
            development_adt = _minimum_marginal_fraction(
                adt_state, development_masks
            )
            held_rna = _minimum_marginal_fraction(rna_state, held_masks)
            held_adt = _minimum_marginal_fraction(adt_state, held_masks)
            reasons = []
            if not (rna_cut[0] < rna_cut[1] and adt_cut[0] < adt_cut[1]):
                reasons.append("NON_DISTINCT_DEVELOPMENT_BASELINE_CUTS")
            if min(development_rna, development_adt) < 0.05:
                reasons.append("DEVELOPMENT_MARGINAL_SUPPORT")
            if min(held_rna, held_adt) < 0.05:
                reasons.append("HELD_PAIRING_INDEPENDENT_MARGINAL_SUPPORT")

            record: dict[str, object] = {
                "entity_id": entity_id,
                "marker_id": marker_id,
                "gene_symbol": gene,
                "marker_index": int(marker_index),
                "contrast_index": int(contrast_index),
                "stimulus": stimulus,
                "lineage": lineage,
                "contrast_id": contrast_id,
                "rna_cut_low": float(rna_cut[0]),
                "rna_cut_high": float(rna_cut[1]),
                "adt_cut_low": float(adt_cut[0]),
                "adt_cut_high": float(adt_cut[1]),
                "development_minimum_rna_state_fraction": development_rna,
                "development_minimum_adt_state_fraction": development_adt,
                "held_minimum_rna_state_fraction": held_rna,
                "held_minimum_adt_state_fraction": held_adt,
            }
            if reasons:
                exclusions.append({**record, "reasons": reasons})
                continue
            entities.append(record)
            rna_states.append(rna_state)
            adt_states.append(adt_state)

    if len(entities) < MINIMUM_ENTITY_PAIRS:
        raise ValueError(
            f"fewer than {MINIMUM_ENTITY_PAIRS} marker-contrast pairs pass the frozen marginal-support rule"
        )
    marker_clusters = len({str(entity["marker_id"]) for entity in entities})
    if marker_clusters < MINIMUM_MARKER_CLUSTERS:
        raise ValueError(
            f"fewer than {MINIMUM_MARKER_CLUSTERS} unique marker clusters pass the frozen marginal-support rule"
        )
    return (
        pd.DataFrame(entities),
        exclusions,
        np.asarray(rna_states, dtype=np.int8),
        np.asarray(adt_states, dtype=np.int8),
    )


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
        references[f"{residual}_destroyed"] = estimate.destroyed_coordinates / np.sqrt(
            rows.sum()
        )
        references[f"{residual}_variance"] = (
            estimate.null_variance_coordinates / rows.sum()
        )
    return references


def _table_stats(first: np.ndarray, second: np.ndarray, seed: int) -> dict[str, object]:
    table = _table(first, second)
    margins = _margin_stats(first, second, seed)
    n = float(margins["total"])
    probability = table / n
    basis = helmert_contrast(3)
    covariance = (
        basis.T
        @ (probability - np.outer(probability.sum(axis=1), probability.sum(axis=0)))
        @ basis
    )
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
        result[residual] = (raw - np.asarray(margins[f"{residual}_null"])) / np.sqrt(n)
        result[f"{residual}_raw"] = raw
    return result


def _build_fields(
    cells: pd.DataFrame,
    entities: pd.DataFrame,
    rna_state: np.ndarray,
    adt_state: np.ndarray,
    *,
    open_held_baseline_pairing: bool,
    open_held_stimulus_pairing: bool,
) -> dict[str, np.ndarray]:
    if open_held_stimulus_pairing and not open_held_baseline_pairing:
        raise ValueError("held stimulus pairing cannot precede held baseline pairing")
    donors = DEVELOPMENT + HELD
    field_shape = (len(donors), len(entities), 2, 2)
    residual_shape = (len(donors), len(entities), 3, 3)
    scalar_shape = (len(donors), len(entities))
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
        for entity_index, entity in entities.iterrows():
            marker = str(entity["marker_id"])
            stimulus = str(entity["stimulus"])
            lineage = str(entity["lineage"])
            common = (donor_values == donor) & (cell_type == lineage)
            baseline_mask = common & (condition == "Baseline")
            stimulus_mask = common & (condition == stimulus)
            if (
                min(
                    np.count_nonzero(baseline_mask), np.count_nonzero(stimulus_mask)
                )
                < 15
            ):
                raise ValueError("frozen donor-arm support unexpectedly failed")
            key = f"{donor}\0{marker}\0{stimulus}\0{lineage}"
            baseline_first = rna_state[entity_index, baseline_mask]
            baseline_second = adt_state[entity_index, baseline_mask]
            baseline_seed = _seed("baseline", key)
            held = donor in HELD
            baseline_open = not held or open_held_baseline_pairing
            baseline = (
                _table_stats(baseline_first, baseline_second, baseline_seed)
                if baseline_open
                else _margin_stats(baseline_first, baseline_second, baseline_seed)
            )
            stimulus_first = rna_state[entity_index, stimulus_mask]
            stimulus_second = adt_state[entity_index, stimulus_mask]
            stimulus_seed = _seed("stimulus", key)
            stimulus_margins = _margin_stats(
                stimulus_first, stimulus_second, stimulus_seed
            )
            held_sealed = held and not open_held_stimulus_pairing
            challenged = (
                None
                if held_sealed
                else _table_stats(stimulus_first, stimulus_second, stimulus_seed)
            )
            location = donor_index, entity_index
            arrays["endpoint"][location] = np.asarray(
                stimulus_margins["endpoint"]
            ) - np.asarray(baseline["endpoint"])
            arrays["baseline_field_null"][location] = baseline["field_null"]
            arrays["stimulus_field_null"][location] = stimulus_margins["field_null"]
            arrays["baseline_total"][location] = baseline["total"]
            arrays["stimulus_total"][location] = stimulus_margins["total"]
            arrays["stimulus_rows"][location] = stimulus_margins["rows"]
            arrays["stimulus_columns"][location] = stimulus_margins["columns"]
            for residual in ("pearson", "deviance"):
                arrays[f"baseline_{residual}_null"][location] = baseline[
                    f"{residual}_null"
                ]
                arrays[f"stimulus_{residual}_null"][location] = stimulus_margins[
                    f"{residual}_null"
                ]
            if baseline_open:
                arrays["baseline_field_raw"][location] = baseline["field_raw"]
                arrays["baseline_covariance"][location] = baseline["covariance"]
                for residual in ("pearson", "deviance"):
                    arrays[f"baseline_{residual}_raw"][location] = baseline[
                        f"{residual}_raw"
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


def _membership_permuted_incidence(
    gene_incidence: np.ndarray,
    typed_incidence: np.ndarray,
    marker_values: np.ndarray,
    marker_order: list[str],
) -> tuple[np.ndarray, list[int]]:
    generator = np.random.default_rng(_seed("membership-permuted"))
    permutation = generator.permutation(len(marker_order)).tolist()
    permuted_gene = np.empty_like(gene_incidence)
    for destination, source in enumerate(permutation):
        destination_rows = marker_values == marker_order[destination]
        source_row = np.flatnonzero(marker_values == marker_order[source])[0]
        permuted_gene[destination_rows] = gene_incidence[source_row]
    return np.column_stack((permuted_gene, typed_incidence)), permutation


def _embedding_laplacian(
    entities: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    with np.load(SCGPT, allow_pickle=False) as archive:
        names = archive["gene_names"].astype(str)
        embedding = np.asarray(archive["embedding"], dtype=float)
    lookup = {name.upper(): index for index, name in enumerate(names)}
    marker_order = list(dict.fromkeys(entities["marker_id"].astype(str)))
    marker_gene = {
        marker: str(
            entities.loc[entities["marker_id"].astype(str) == marker, "gene_symbol"].iloc[
                0
            ]
        )
        for marker in marker_order
    }
    covered_markers = [
        marker for marker in marker_order if marker_gene[marker].upper() in lookup
    ]
    if len(covered_markers) < EXTERNAL_NEIGHBORS + 1:
        raise ValueError(
            "fewer than seven distinct eligible markers have frozen gene embeddings"
        )
    values = np.asarray(
        [embedding[lookup[marker_gene[marker].upper()]] for marker in covered_markers]
    )
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    similarity = values @ values.T

    gene_hyperedges: list[np.ndarray] = []
    neighbor_record: dict[str, list[str]] = {}
    marker_values = entities["marker_id"].astype(str).to_numpy()
    for local, marker in enumerate(covered_markers):
        order = np.argsort(similarity[local])[::-1]
        neighbors = [
            covered_markers[index]
            for index in order
            if covered_markers[index] != marker
        ][:EXTERNAL_NEIGHBORS]
        if len(neighbors) != EXTERNAL_NEIGHBORS:
            raise ValueError("the frozen embedding graph lacks six external neighbors")
        neighbor_record[marker] = neighbors
        gene_hyperedges.append(
            np.isin(marker_values, [marker, *neighbors]).astype(float)
        )
    for marker in marker_order:
        if marker in covered_markers:
            continue
        neighbor_record[marker] = []
        gene_hyperedges.append((marker_values == marker).astype(float))

    typed_hyperedges: list[np.ndarray] = []
    contrast_values = entities["contrast_id"].astype(str).to_numpy()
    for contrast in dict.fromkeys(contrast_values):
        typed_hyperedges.append((contrast_values == contrast).astype(float))
    lineage_values = entities["lineage"].astype(str).to_numpy()
    for lineage in dict.fromkeys(lineage_values):
        typed_hyperedges.append((lineage_values == lineage).astype(float))
    gene_incidence = np.column_stack(gene_hyperedges)
    typed_incidence = np.column_stack(typed_hyperedges)
    incidence = np.column_stack((gene_incidence, typed_incidence))
    permuted_incidence, membership_permutation = _membership_permuted_incidence(
        gene_incidence, typed_incidence, marker_values, marker_order
    )
    return (
        normalized_hypergraph_laplacian(incidence),
        normalized_hypergraph_laplacian(permuted_incidence),
        {
            "path": str(SCGPT.relative_to(ROOT)),
            "sha256": _sha256(SCGPT),
            "covered_markers": len(covered_markers),
            "eligible_markers": len(marker_order),
            "external_neighbors_per_covered_marker": EXTERNAL_NEIGHBORS,
            "embedding_neighbors": neighbor_record,
            "contrast_hyperedges": list(dict.fromkeys(contrast_values)),
            "lineage_hyperedges": list(dict.fromkeys(lineage_values)),
            "membership_permutation_marker_order": marker_order,
            "membership_permutation_source_index_by_destination": membership_permutation,
            "membership_control_preserves_typed_incidence": bool(
                np.array_equal(
                    incidence[:, gene_incidence.shape[1] :],
                    permuted_incidence[:, gene_incidence.shape[1] :],
                )
            ),
        },
    )


def _structured(
    values: np.ndarray,
    variance: np.ndarray,
    laplacian: np.ndarray,
    *,
    nuclear: float,
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
    arrays: dict[str, np.ndarray], entities: pd.DataFrame
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    develop = slice(0, len(DEVELOPMENT))
    held = slice(len(DEVELOPMENT), len(DEVELOPMENT) + len(HELD))
    laplacian, permuted_laplacian, embedding = _embedding_laplacian(entities)
    entity_count = len(entities)

    def mean_flat(name: str) -> np.ndarray:
        return arrays[name][develop].mean(axis=0).reshape(entity_count, -1)

    def variance_flat(name: str) -> np.ndarray:
        return (
            arrays[name][develop].sum(axis=0).reshape(entity_count, -1)
            / len(DEVELOPMENT) ** 2
        )

    field = mean_flat("field")
    variance = variance_flat("field_variance")
    signal_energy = float(np.sum(field**2))
    scalar = max(
        0.0,
        1.0 - float(np.sum(variance)) / max(signal_energy, np.finfo(float).eps),
    )
    predictions = {
        "field_direct": field,
        "field_zero": np.zeros_like(field),
        "field_scalar": scalar * field,
        "field_nuclear": _structured(
            field, variance, laplacian, nuclear=0.1, graph=0.0
        ),
        "field_hypergraph": _structured(
            field, variance, laplacian, nuclear=0.0, graph=5.0
        ),
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
        "field_destroyed": _structured(
            mean_flat("field_destroyed"),
            variance,
            laplacian,
            nuclear=NUCLEAR_FRACTION,
            graph=GRAPH_PENALTY,
        ),
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

    x_train = arrays["endpoint"][develop].reshape(-1, 4)
    y_train = arrays["field"][develop].reshape(-1, 4)
    endpoint = Ridge(alpha=0.1, fit_intercept=True).fit(x_train, y_train)
    predictions["field_endpoint_ridge"] = endpoint.predict(
        arrays["endpoint"][held].reshape(-1, 4)
    ).reshape(len(HELD), entity_count, 4)
    embedding["variance_scalar"] = scalar
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
    seed_probability = np.outer(rows / total, columns / total) + field_from_coordinates(
        anchor + contrast
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
    raw_stimulus = (centered_baseline + contrast) * np.sqrt(
        stimulus_total
    ) + stimulus_null
    return residual_coordinates_to_table(raw_stimulus, rows, columns, residual=residual)


def _predict_tables(
    arrays: dict[str, np.ndarray], predictions: dict[str, np.ndarray], entities: int
) -> dict[str, np.ndarray]:
    held_offset = len(DEVELOPMENT)
    predicted_tables = {
        name: np.empty((len(HELD), entities, 3, 3))
        for name in predictions
    }
    for donor in range(len(HELD)):
        source = held_offset + donor
        for entity in range(entities):
            location = source, entity
            rows = arrays["stimulus_rows"][location]
            columns = arrays["stimulus_columns"][location]
            for method, estimate in predictions.items():
                if estimate.ndim == 3:
                    flat = estimate[donor, entity]
                else:
                    flat = estimate[entity]
                if method.startswith("field_"):
                    table = _field_table(
                        arrays["baseline_field_raw"][location],
                        arrays["baseline_field_null"][location],
                        arrays["stimulus_field_null"][location],
                        flat.reshape(2, 2),
                        rows,
                        columns,
                    )
                elif method.startswith("covariance_"):
                    table = _covariance_table(
                        arrays["baseline_covariance"][location],
                        flat.reshape(2, 2),
                        rows,
                        columns,
                    )
                elif method.startswith("pearson_"):
                    table = _residual_table(
                        arrays["baseline_pearson_raw"][location],
                        arrays["baseline_pearson_null"][location],
                        arrays["stimulus_pearson_null"][location],
                        arrays["baseline_total"][location],
                        arrays["stimulus_total"][location],
                        flat.reshape(3, 3),
                        rows,
                        columns,
                        "pearson",
                    )
                elif method.startswith("deviance_"):
                    table = _residual_table(
                        arrays["baseline_deviance_raw"][location],
                        arrays["baseline_deviance_null"][location],
                        arrays["stimulus_deviance_null"][location],
                        arrays["baseline_total"][location],
                        arrays["stimulus_total"][location],
                        flat.reshape(3, 3),
                        rows,
                        columns,
                        "deviance",
                    )
                else:
                    raise ValueError(f"unknown prediction family: {method}")
                predicted_tables[method][donor, entity] = table
    return predicted_tables


def _require_locked_table_consistency(
    arrays: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    locked_tables: dict[str, np.ndarray],
    entities: int,
) -> None:
    recomputed = _predict_tables(arrays, predictions, entities)
    if set(recomputed) != set(locked_tables):
        raise PermissionError("locked and recomputed table method sets differ")
    for name, expected in recomputed.items():
        observed = locked_tables[name]
        if observed.shape != expected.shape or not np.allclose(
            observed, expected, rtol=1e-12, atol=1e-12
        ):
            raise PermissionError(
                f"locked table values are inconsistent with coordinates for {name}"
            )


def _require_locked_prediction_consistency(
    arrays: dict[str, np.ndarray],
    entities: pd.DataFrame,
    locked_predictions: dict[str, np.ndarray],
) -> None:
    recomputed, _ = _fit_predictions(arrays, entities)
    if set(recomputed) != set(locked_predictions):
        raise PermissionError("locked and recomputed prediction method sets differ")
    for name, expected in recomputed.items():
        observed = locked_predictions[name]
        if observed.shape != expected.shape or not np.array_equal(observed, expected):
            raise PermissionError(
                f"locked prediction coordinates differ from sealed development fit for {name}"
            )


def _prepare_locked_tables(
    cells: pd.DataFrame,
    entities: pd.DataFrame,
    rna_state: np.ndarray,
    adt_state: np.ndarray,
    record: dict[str, object],
    predictions: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    sealed_arrays = _build_fields(
        cells,
        entities,
        rna_state,
        adt_state,
        open_held_baseline_pairing=False,
        open_held_stimulus_pairing=False,
    )
    _require_locked_prediction_consistency(sealed_arrays, entities, predictions)
    held_rows = sealed_arrays["stimulus_rows"][len(DEVELOPMENT) :]
    held_columns = sealed_arrays["stimulus_columns"][len(DEVELOPMENT) :]
    if not np.array_equal(
        np.asarray(record["held_stimulus_rows"]), held_rows
    ) or not np.array_equal(
        np.asarray(record["held_stimulus_columns"]), held_columns
    ):
        raise PermissionError("locked held margins differ from the sealed scoring input")

    baseline_arrays = _build_fields(
        cells,
        entities,
        rna_state,
        adt_state,
        open_held_baseline_pairing=True,
        open_held_stimulus_pairing=False,
    )
    predicted_tables = _predict_tables(baseline_arrays, predictions, len(entities))
    for values in predicted_tables.values():
        if not np.allclose(values.sum(axis=-1), held_rows) or not np.allclose(
            values.sum(axis=-2), held_columns
        ):
            raise PermissionError("locked predicted tables do not match held margins")
    return baseline_arrays, predicted_tables


def _score_tables(
    arrays: dict[str, np.ndarray], predicted_tables: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    held_truth = arrays["stimulus_table"][len(DEVELOPMENT) :]
    if not np.isfinite(held_truth).all():
        raise ValueError("held stimulus pairing was not opened for scoring")
    losses = {
        name: np.empty(values.shape[:2]) for name, values in predicted_tables.items()
    }
    for name, values in predicted_tables.items():
        if values.shape != held_truth.shape:
            raise ValueError(f"predicted table axes differ from held truth for {name}")
        for index in np.ndindex(values.shape[:2]):
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
    arrays: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    losses: dict[str, np.ndarray],
    *,
    marker_clusters: list[str],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    truth = arrays["field"][len(DEVELOPMENT) :].reshape(
        len(HELD), len(predictions["field_direct"]), 4
    )
    if len(marker_clusters) != truth.shape[1]:
        raise ValueError("marker-cluster axis differs from the entity axis")
    primary = predictions["field_primary"]
    pooled = _correlation(np.broadcast_to(primary, truth.shape), truth)
    donor_r = [_correlation(primary, truth[index]) for index in range(len(HELD))]
    per_entity_loss = {name: values.mean(axis=0) for name, values in losses.items()}
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
    best_classical = min(
        classical, key=lambda name: float(per_entity_loss[name].mean())
    )
    best_matched = min(matched, key=lambda name: float(per_entity_loss[name].mean()))
    rng = np.random.default_rng(SEED)
    correlations = np.empty(BOOTSTRAPS)
    primary_minus_destroyed = np.empty(BOOTSTRAPS)
    primary_minus_classical = np.empty(BOOTSTRAPS)
    primary_minus_matched = np.empty(BOOTSTRAPS)
    marker_order = list(dict.fromkeys(marker_clusters))
    marker_blocks = [
        np.flatnonzero(np.asarray(marker_clusters) == marker) for marker in marker_order
    ]
    marker_count = len(marker_blocks)
    for draw in range(BOOTSTRAPS):
        sampled_markers = rng.integers(0, marker_count, marker_count)
        index = np.concatenate([marker_blocks[item] for item in sampled_markers])
        correlations[draw] = _correlation(
            np.broadcast_to(primary[index], truth[:, index].shape), truth[:, index]
        )
        primary_loss = per_entity_loss["field_primary"][index].mean()
        primary_minus_destroyed[draw] = (
            primary_loss - per_entity_loss["field_destroyed"][index].mean()
        )
        selected = min(
            classical, key=lambda name: float(per_entity_loss[name][index].mean())
        )
        primary_minus_classical[draw] = (
            primary_loss - per_entity_loss[selected][index].mean()
        )
        selected_matched = min(
            matched, key=lambda name: float(per_entity_loss[name][index].mean())
        )
        primary_minus_matched[draw] = (
            primary_loss - per_entity_loss[selected_matched][index].mean()
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
            len(HELD), truth.shape[1], 9
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
        "bootstrap_unit": "complete matched RNA-ADT marker cluster (all eligible contrast pairs)",
        "bootstrap_marker_clusters": marker_count,
        "eligible_marker_contrast_pairs": truth.shape[1],
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
    pd.DataFrame,
    list[dict[str, object]],
    np.ndarray,
    np.ndarray,
]:
    cells, marker_frame, rna, adt = _load_reduced(reduced)
    entities, exclusions, rna_state, adt_state = _state_thresholds(
        cells, marker_frame, rna, adt
    )
    return cells, entities, exclusions, rna_state, adt_state


def predict(args: argparse.Namespace) -> None:
    provenance = preflight(require_sealed=True)
    output = Path(args.output)
    if _repo_relative(output) != provenance["prediction_path"]:
        raise ValueError("prediction output path differs from the frozen designation")
    if output.exists():
        raise FileExistsError(f"prospective prediction already exists: {output}")
    reduced = Path(args.reduced)
    if _repo_relative(reduced) != provenance["reduced_path"]:
        raise ValueError("reducer output path differs from the frozen designation")
    try:
        reduce_inputs(
            rna=Path(args.rna),
            adt=Path(args.adt),
            annotations=Path(args.annotations),
            output=reduced,
        )
        reducer_artifacts = _reducer_artifact_bundle(reduced)
        (
            cells,
            entities,
            marginal_support_exclusions,
            rna_state,
            adt_state,
        ) = _analysis_data(reduced)
        sealed_arrays = _build_fields(
            cells,
            entities,
            rna_state,
            adt_state,
            open_held_baseline_pairing=False,
            open_held_stimulus_pairing=False,
        )
        predictions, embedding = _fit_predictions(sealed_arrays, entities)
        _require_reducer_artifact_bundle(reduced, reducer_artifacts)
        entity_records = json.loads(entities.to_json(orient="records"))
        record = {
            "schema": "lawlor-hca-pbmc-predictions/2.0",
            "status": "PREDICTIONS_FROZEN_PAIRING_UNOPENED",
            "scope": "held-donor stimulus-by-cell-type RNA-protein coupling; not CRISPR target transfer",
            "entity_ids": entities["entity_id"].astype(str).tolist(),
            "marker_ids": entities["marker_id"].astype(str).tolist(),
            "genes": entities["gene_symbol"].astype(str).tolist(),
            "entity_contrasts": entities["contrast_id"].astype(str).tolist(),
            "entity_lineages": entities["lineage"].astype(str).tolist(),
            "entity_definitions": entity_records,
            "marginal_support_exclusions": marginal_support_exclusions,
            "development_donors": list(DEVELOPMENT),
            "held_donors": list(HELD),
            "contrasts": [f"{stimulus}:{lineage}" for stimulus, lineage in CONTRASTS],
            "held_stimulus_rows": sealed_arrays["stimulus_rows"][
                len(DEVELOPMENT) :
            ].tolist(),
            "held_stimulus_columns": sealed_arrays["stimulus_columns"][
                len(DEVELOPMENT) :
            ].tolist(),
            "predictions": {
                name: value.tolist() for name, value in predictions.items()
            },
            "predicted_tables_stored": False,
            "table_reconstruction_rule": "score reconstructs every method from locked coordinates, held baseline anchors, pairing-independent held stimulus margins, and the bound implementation before held stimulus pairing is opened",
            "embedding": embedding,
            "provenance": {
                **provenance,
                "reducer_artifacts": reducer_artifacts,
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    except Exception as error:
        _write_refusal(output, stage="PREDICT", error=error)
        raise
def public_bind(args: argparse.Namespace) -> None:
    preflight(require_sealed=True)
    prediction = Path(args.predictions)
    if _repo_relative(prediction) != _repo_relative(PREDICTION_PATH):
        raise ValueError("prediction path differs from the frozen designation")
    if SCORE_AUTHORIZATION.exists():
        raise FileExistsError(f"score authorization already exists: {SCORE_AUTHORIZATION}")
    template = _read_json(AUTHORIZATION_TEMPLATE)
    if (
        template.get("schema")
        != "lawlor-hca-pbmc-score-authorization-template/2.0"
    ):
        raise ValueError("score authorization template schema differs")
    commit = args.public_commit
    url = args.public_url
    relative = _repo_relative(prediction)
    _require_github_commit_url(url, commit, blob_path=relative)
    record = {
        "schema": "lawlor-hca-pbmc-score-authorization/2.0",
        "status": "SEALED",
        "outcome_access_authorized": True,
        "candidate": "HCA:efea6426-510a-4b60-9a19-277e52bfa815",
        "prediction_path": relative,
        "prediction_sha256": _sha256(prediction),
        "prediction_bytes": prediction.stat().st_size,
        "prediction_public_url": url,
        "prediction_public_commit": commit,
        "runner": str(Path(__file__).relative_to(ROOT)),
        "runner_sha256": _sha256(Path(__file__)),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": _sha256(PROTOCOL),
        "publication_required_before_score": True,
    }
    SCORE_AUTHORIZATION.parent.mkdir(parents=True, exist_ok=True)
    SCORE_AUTHORIZATION.write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )


def _require_score_authorization(
    prediction_path: Path, authorization_path: Path
) -> dict[str, object]:
    authorization = _read_json(authorization_path)
    if authorization.get("schema") != "lawlor-hca-pbmc-score-authorization/2.0":
        raise PermissionError("score authorization schema differs from version 2.0")
    if authorization.get("status") != "SEALED":
        raise PermissionError("score authorization is not SEALED")
    if authorization.get("outcome_access_authorized") is not True:
        raise PermissionError("score authorization forbids held pairing access")
    if authorization.get("prediction_sha256") != _sha256(prediction_path):
        raise PermissionError("the prediction JSON hash differs from authorization")
    if authorization.get("prediction_bytes") != prediction_path.stat().st_size:
        raise PermissionError(
            "the prediction JSON byte count differs from authorization"
        )
    prediction_relative = _repo_relative(prediction_path)
    if authorization.get("prediction_path") != prediction_relative:
        raise PermissionError("the prediction path differs from authorization")
    _require_github_commit_url(
        authorization.get("prediction_public_url"),
        authorization.get("prediction_public_commit"),
        blob_path=prediction_relative,
    )
    if authorization.get("runner_sha256") != _sha256(Path(__file__)):
        raise PermissionError("the scoring runner differs from authorization")
    if authorization.get("protocol_sha256") != _sha256(PROTOCOL):
        raise PermissionError("the scoring protocol differs from authorization")
    return authorization


def _require_score_release(
    authorization_path: Path, release_path: Path
) -> dict[str, object]:
    release = _read_json(release_path)
    if release.get("schema") != "lawlor-hca-pbmc-score-release/2.0":
        raise PermissionError("score release schema differs")
    if release.get("status") != "SEALED":
        raise PermissionError("score release is not SEALED")
    if release.get("held_pairing_access_authorized") is not True:
        raise PermissionError("score release forbids held pairing access")
    relative = _repo_relative(authorization_path)
    if release.get("authorization_path") != relative:
        raise PermissionError("score release authorization path differs")
    if release.get("authorization_sha256") != _sha256(authorization_path):
        raise PermissionError("score release authorization hash differs")
    if release.get("authorization_bytes") != authorization_path.stat().st_size:
        raise PermissionError("score release authorization byte count differs")
    _require_github_commit_url(
        release.get("authorization_public_url"),
        release.get("authorization_public_commit"),
        blob_path=relative,
    )
    if release.get("runner_sha256") != _sha256(Path(__file__)):
        raise PermissionError("score release runner differs")
    if release.get("protocol_sha256") != _sha256(PROTOCOL):
        raise PermissionError("score release protocol differs")
    return release


def authorize_score(args: argparse.Namespace) -> None:
    provenance = preflight(require_sealed=True)
    prediction_path = Path(args.predictions)
    authorization_path = Path(args.authorization)
    if _repo_relative(prediction_path) != provenance["prediction_path"]:
        raise ValueError("prediction path differs from the frozen designation")
    if _repo_relative(authorization_path) != provenance["score_authorization"]:
        raise ValueError("score authorization path differs from the frozen designation")
    if SCORE_RELEASE.exists():
        raise FileExistsError(f"score release already exists: {SCORE_RELEASE}")
    _require_score_authorization(prediction_path, authorization_path)
    commit = args.authorization_public_commit
    url = args.authorization_public_url
    relative = _repo_relative(authorization_path)
    _require_github_commit_url(url, commit, blob_path=relative)
    release = {
        "schema": "lawlor-hca-pbmc-score-release/2.0",
        "status": "SEALED",
        "held_pairing_access_authorized": True,
        "authorization_path": relative,
        "authorization_sha256": _sha256(authorization_path),
        "authorization_bytes": authorization_path.stat().st_size,
        "authorization_public_url": url,
        "authorization_public_commit": commit,
        "runner": str(Path(__file__).relative_to(ROOT)),
        "runner_sha256": _sha256(Path(__file__)),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": _sha256(PROTOCOL),
    }
    SCORE_RELEASE.parent.mkdir(parents=True, exist_ok=True)
    SCORE_RELEASE.write_text(json.dumps(release, indent=2, allow_nan=False) + "\n")


def _locked_predictions(
    path: Path,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    record = _read_json(path)
    if record.get("schema") != "lawlor-hca-pbmc-predictions/2.0":
        raise ValueError("prediction JSON schema differs")
    if record.get("status") != "PREDICTIONS_FROZEN_PAIRING_UNOPENED":
        raise ValueError("prediction JSON is not frozen")
    if record.get("predicted_tables_stored") is not False:
        raise ValueError("prediction JSON must not store predicted tables")
    predictions = {
        name: np.asarray(value, dtype=float)
        for name, value in record["predictions"].items()
    }
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
    expected_methods = (
        field_methods
        | residual_methods
        | {
            "field_endpoint_ridge",
            "covariance_direct",
        }
    )
    if set(predictions) != expected_methods:
        raise ValueError("locked prediction method set differs from the protocol")
    entity_count = len(record["entity_ids"])
    if any(
        len(record.get(axis, [])) != entity_count
        for axis in (
            "marker_ids",
            "genes",
            "entity_contrasts",
            "entity_lineages",
            "entity_definitions",
        )
    ):
        raise ValueError("locked prediction entity metadata axes differ")
    for name, values in predictions.items():
        if name == "field_endpoint_ridge":
            expected_shape = (len(HELD), entity_count, 4)
        elif name in residual_methods:
            expected_shape = (entity_count, 9)
        else:
            expected_shape = (entity_count, 4)
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise ValueError(f"locked prediction shape differs for {name}")
    if record.get("provenance", {}).get("runner_sha256") != _sha256(Path(__file__)):
        raise ValueError("prediction JSON was produced by a different runner")
    return record, predictions


def score(args: argparse.Namespace) -> None:
    provenance = preflight(require_sealed=True)
    prediction_path = Path(args.predictions)
    if _repo_relative(prediction_path) != provenance["prediction_path"]:
        raise ValueError("prediction path differs from the frozen designation")
    authorization_path = Path(args.authorization)
    if _repo_relative(authorization_path) != provenance["score_authorization"]:
        raise ValueError("score authorization path differs from the frozen designation")
    authorization = _require_score_authorization(prediction_path, authorization_path)
    release_path = Path(args.release)
    if _repo_relative(release_path) != provenance["score_release"]:
        raise ValueError("score release path differs from the frozen designation")
    release = _require_score_release(authorization_path, release_path)
    output = Path(args.output)
    if _repo_relative(output) != provenance["score_output"]:
        raise ValueError("score output path differs from the frozen designation")
    scored_path = ARRAYS_PATH
    if _repo_relative(scored_path) != provenance["score_arrays"]:
        raise ValueError("score arrays path differs from the frozen designation")
    if output.exists() or scored_path.exists():
        raise FileExistsError("prospective score output already exists")
    reduced = Path(args.reduced)
    if _repo_relative(reduced) != provenance["reduced_path"]:
        raise ValueError("reducer output path differs from the frozen designation")
    try:
        record, predictions = _locked_predictions(prediction_path)
        reducer_artifacts = record.get("provenance", {}).get("reducer_artifacts")
        _require_reducer_artifact_bundle(reduced, reducer_artifacts)
        (
            cells,
            entities,
            marginal_support_exclusions,
            rna_state,
            adt_state,
        ) = _analysis_data(reduced)
        entity_records = json.loads(entities.to_json(orient="records"))
        expected_metadata = {
            "entity_ids": entities["entity_id"].astype(str).tolist(),
            "marker_ids": entities["marker_id"].astype(str).tolist(),
            "genes": entities["gene_symbol"].astype(str).tolist(),
            "entity_contrasts": entities["contrast_id"].astype(str).tolist(),
            "entity_lineages": entities["lineage"].astype(str).tolist(),
            "entity_definitions": entity_records,
            "marginal_support_exclusions": marginal_support_exclusions,
            "development_donors": list(DEVELOPMENT),
            "held_donors": list(HELD),
            "contrasts": [
                f"{stimulus}:{lineage}" for stimulus, lineage in CONTRASTS
            ],
        }
        for name, expected in expected_metadata.items():
            if record.get(name) != expected:
                raise ValueError(
                    f"locked prediction {name} differs from scoring input"
                )
        _, predicted_tables = _prepare_locked_tables(
            cells,
            entities,
            rna_state,
            adt_state,
            record,
            predictions,
        )
        _require_reducer_artifact_bundle(reduced, reducer_artifacts)
        arrays = _build_fields(
            cells,
            entities,
            rna_state,
            adt_state,
            open_held_baseline_pairing=True,
            open_held_stimulus_pairing=True,
        )
        losses = _score_tables(arrays, predicted_tables)
        summary, bootstrap = summarize(
            arrays,
            predictions,
            losses,
            marker_clusters=entities["marker_id"].astype(str).tolist(),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            scored_path,
            entity_ids=entities["entity_id"].astype(str).to_numpy(),
            marker_ids=entities["marker_id"].astype(str).to_numpy(),
            genes=entities["gene_symbol"].astype(str).to_numpy(),
            entity_contrasts=entities["contrast_id"].astype(str).to_numpy(),
            held_donors=np.asarray(HELD),
            held_field_truth=arrays["field"][len(DEVELOPMENT) :],
            **{f"prediction__{name}": value for name, value in predictions.items()},
            **{f"loss__{name}": value for name, value in losses.items()},
            **{f"table__{name}": value for name, value in predicted_tables.items()},
            **{f"bootstrap__{name}": value for name, value in bootstrap.items()},
        )
        result = {
            "schema": "lawlor-hca-pbmc-confirmation/2.0",
            "status": "PROMOTE" if summary["gate_passed"] else "REFUSE",
            "scope": "held-donor stimulus-by-cell-type RNA-protein coupling; not CRISPR target transfer",
            "development_donors": list(DEVELOPMENT),
            "held_donors": list(HELD),
            "contrasts": [f"{stimulus}:{lineage}" for stimulus, lineage in CONTRASTS],
            "entity_ids": entities["entity_id"].astype(str).tolist(),
            "marker_ids": entities["marker_id"].astype(str).tolist(),
            "entity_count": len(entities),
            "marker_cluster_count": entities["marker_id"].nunique(),
            "marginal_support_exclusions": marginal_support_exclusions,
            "states": 3,
            "fixed_margin_reference_permutations": PERMUTATIONS,
            "summary": summary,
            "prediction_json": str(prediction_path.relative_to(ROOT)),
            "prediction_json_sha256": _sha256(prediction_path),
            "prediction_public_url": authorization["prediction_public_url"],
            "prediction_public_commit": authorization["prediction_public_commit"],
            "score_authorization": {
                "path": _repo_relative(authorization_path),
                "sha256": _sha256(authorization_path),
                "bytes": authorization_path.stat().st_size,
                "public_url": release["authorization_public_url"],
                "public_commit": release["authorization_public_commit"],
            },
            "scored_arrays": str(scored_path.relative_to(ROOT)),
            "scored_arrays_sha256": _sha256(scored_path),
            "provenance": provenance,
        }
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except Exception as error:
        _write_refusal(output, stage="SCORE", error=error)
        raise
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    prediction = subparsers.add_parser("predict")
    prediction.add_argument("--rna")
    prediction.add_argument("--adt")
    prediction.add_argument("--annotations")
    prediction.add_argument("--reduced", default=str(REDUCED_PATH))
    prediction.add_argument(
        "--output",
        default=str(PREDICTION_PATH),
    )
    bind = subparsers.add_parser("public-bind")
    bind.add_argument("--predictions", default=str(PREDICTION_PATH))
    bind.add_argument("--public-commit", required=True)
    bind.add_argument("--public-url", required=True)
    authorize = subparsers.add_parser("authorize-score")
    authorize.add_argument("--predictions", default=str(PREDICTION_PATH))
    authorize.add_argument("--authorization", default=str(SCORE_AUTHORIZATION))
    authorize.add_argument("--authorization-public-commit", required=True)
    authorize.add_argument("--authorization-public-url", required=True)
    scoring = subparsers.add_parser("score")
    scoring.add_argument("--reduced", default=str(REDUCED_PATH))
    scoring.add_argument(
        "--predictions",
        default=str(PREDICTION_PATH),
    )
    scoring.add_argument(
        "--authorization",
        default=str(SCORE_AUTHORIZATION),
    )
    scoring.add_argument("--release", default=str(SCORE_RELEASE))
    scoring.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(preflight(require_sealed=False), indent=2))
        return
    if args.command == "predict" and not all((args.rna, args.adt, args.annotations)):
        parser.error("--rna, --adt, and --annotations are required")
    if args.command == "predict":
        predict(args)
    elif args.command == "public-bind":
        public_bind(args)
    elif args.command == "authorize-score":
        authorize_score(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
