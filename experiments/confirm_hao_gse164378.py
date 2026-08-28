"""Prospective held-donor RNA-protein confirmation on GSE164378."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import subprocess
import sys
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
PROTOCOL = ROOT / "docs/GSE164378_HAO_HELD_DONOR_CONFIRMATION_PROTOCOL_2026-08-28.md"
DESIGNATION = ROOT / "data/confirmation/hao_gse164378/candidate_designation_v1.json"
AUTH_TEMPLATE = ROOT / "data/confirmation/hao_gse164378/score_authorization_template_v1.json"
SCORE_AUTHORIZATION = ROOT / "data/confirmation/hao_gse164378/score_authorization_v1.json"
SCORE_RELEASE = ROOT / "data/confirmation/hao_gse164378/score_release_v1.json"
SOURCE_MANIFEST = ROOT / "data/development/hao_gse164378/source_manifest_v1.json"
SUPPORT = ROOT / "data/development/hao_gse164378/metadata_support_v1.json"
ALIASES = ROOT / "data/development/hao_gse164378/adt_gene_aliases_v1.tsv"
REDUCER = ROOT / "experiments/reduce_hao_gse164378.py"
TEST = ROOT / "tests/test_hao_gse164378_confirmation.py"
SCGPT = ROOT / "data/scgpt_gene_embeddings.npz"
OUTPUT = ROOT / "results/hao_gse164378_confirmation.json"
ARRAYS_PATH = ROOT / "results/hao_gse164378_confirmation_arrays.npz"
PREDICTION_PATH = ROOT / "results/hao_gse164378_predictions.json"
REDUCED_PATH = ROOT / "data/development/hao_gse164378/reduced_v1"
PREPARE_RECORD = ROOT / "data/development/hao_gse164378/prepared_v1.json"
PREPARE_REFUSAL = ROOT / "results/hao_gse164378_prepare_refusal.json"
PREDICTION_REFUSAL = ROOT / "results/hao_gse164378_prediction_refusal.json"
SCORE_REFUSAL = ROOT / "results/hao_gse164378_score_refusal.json"
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
    "source_acquisition.json",
)
REDUCER_MANIFEST = "reducer_manifest.tsv"

SEED = 20260828
PERMUTATIONS = 64
BOOTSTRAPS = 2_000
PSEUDOCOUNT = 0.5
NUCLEAR_FRACTION = 0.1
GRAPH_PENALTY = 5.0
DESCRIPTIVE_COVERAGE = 0.95
FAMILY_GATE_COVERAGE = 0.95
DEVELOPMENT = ("P4", "P7", "P8", "P1")
HELD = ("P5", "P3", "P2")
LINEAGES = ("B", "CD4 T", "CD8 T", "Mono", "NK")
DAYS = (3, 7)
BLOCKS = DAYS
MINIMUM_CELLS = 40


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
        raise PermissionError("public blob URL does not contain the exact artifact path")


def _reducer_artifact_bundle(reduced: Path) -> dict[str, object]:
    reduced = reduced.resolve()
    reduced_root = _repo_relative(reduced)
    if not reduced.is_dir() or reduced.is_symlink():
        raise ValueError("reduced output must be a real directory")
    manifest_path = reduced / REDUCER_MANIFEST
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("reducer manifest is missing or is a symlink")
    manifest = pd.read_csv(
        manifest_path, sep="\t", dtype={"path": str, "sha256": str}
    )
    if list(manifest.columns) != ["path", "bytes", "sha256"]:
        raise ValueError("reducer manifest columns differ from the protocol")
    if manifest["path"].duplicated().any() or set(manifest["path"]) != set(
        REDUCER_OUTPUTS
    ):
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
        declared_bytes = by_path.at[relative, "bytes"]
        declared_sha = by_path.at[relative, "sha256"]
        if not isinstance(declared_bytes, (int, np.integer)):
            raise ValueError(f"reducer byte count is not an integer: {relative}")
        if re.fullmatch(r"[0-9a-f]{64}", declared_sha) is None:
            raise ValueError(f"reducer SHA-256 is malformed: {relative}")
        observed = {
            "path": relative,
            "bytes": artifact.stat().st_size,
            "sha256": _sha256(artifact),
        }
        if observed["bytes"] != int(declared_bytes) or observed["sha256"] != declared_sha:
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
        "schema": "hao-reducer-artifact-bundle/1.0",
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


def preflight(*, require_sealed: bool) -> dict[str, object]:
    designation = _read_json(DESIGNATION)
    if designation.get("schema") != "hao-gse164378-coupling-candidate-designation/1.0":
        raise ValueError("candidate designation schema differs from version 1.0")
    if require_sealed:
        if designation.get("status") != "SEALED":
            raise PermissionError("candidate designation is not SEALED")
        if designation.get("outcome_access_authorized") is not True:
            raise PermissionError("outcome access is not authorized")
        _require_github_commit_url(
            designation.get("public_freeze_url"),
            designation.get("public_freeze_commit"),
            blob_path=None,
        )
    expected = {
        "protocol_sha256": _sha256(PROTOCOL),
        "source_manifest_sha256": _sha256(SOURCE_MANIFEST),
        "metadata_support_sha256": _sha256(SUPPORT),
        "alias_sha256": _sha256(ALIASES),
        "embedding_sha256": _sha256(SCGPT),
        "authorization_template_sha256": _sha256(AUTH_TEMPLATE),
        "test_sha256": _sha256(TEST),
    }
    for key, observed in expected.items():
        if designation.get(key) != observed:
            raise ValueError(f"designation {key} is stale")
    runner_sha = _sha256(Path(__file__))
    reducer_sha = _sha256(REDUCER)
    implementation_sha = {
        str(path.relative_to(ROOT)): _sha256(path) for path in IMPLEMENTATION_FILES
    }
    if designation.get("implementation_sha256") != implementation_sha:
        raise ValueError("designation implementation_sha256 is stale")
    bound = {
        "designation": _repo_relative(DESIGNATION),
        "runner": str(Path(__file__).relative_to(ROOT)),
        "runner_sha256": runner_sha,
        "reducer": str(REDUCER.relative_to(ROOT)),
        "reducer_sha256": reducer_sha,
        "reduced_path": _repo_relative(REDUCED_PATH),
        "prepare_record": _repo_relative(PREPARE_RECORD),
        "prediction_path": _repo_relative(PREDICTION_PATH),
        "score_authorization": _repo_relative(SCORE_AUTHORIZATION),
        "score_release": _repo_relative(SCORE_RELEASE),
        "score_output": _repo_relative(OUTPUT),
        "score_arrays": _repo_relative(ARRAYS_PATH),
        "prepare_refusal_path": _repo_relative(PREPARE_REFUSAL),
        "prediction_refusal_path": _repo_relative(PREDICTION_REFUSAL),
        "score_refusal_path": _repo_relative(SCORE_REFUSAL),
    }
    for key, observed in bound.items():
        if designation.get(key) != observed:
            raise ValueError(f"designation {key} is stale")
    declared_paths = {
        "protocol": _repo_relative(PROTOCOL),
        "source_manifest": _repo_relative(SOURCE_MANIFEST),
        "metadata_support_artifact": _repo_relative(SUPPORT),
        "alias_table": _repo_relative(ALIASES),
        "authorization_template": _repo_relative(AUTH_TEMPLATE),
        "embedding": _repo_relative(SCGPT),
        "test": _repo_relative(TEST),
    }
    for key, observed in declared_paths.items():
        if designation.get(key) != observed:
            raise ValueError(f"designation {key} path differs")
    frozen_constants = {
        "development_units": list(DEVELOPMENT),
        "held_units": list(HELD),
        "lineages": list(LINEAGES),
        "primary_blocks": [f"day{day}" for day in BLOCKS],
        "minimum_cells_per_donor_day_lineage": MINIMUM_CELLS,
        "bootstrap_draws": BOOTSTRAPS,
        "seed": SEED,
    }
    for key, observed in frozen_constants.items():
        if designation.get(key) != observed:
            raise ValueError(f"designation {key} differs from the runner")
    return {
        "designation_status": designation["status"],
        "outcome_access_authorized": designation["outcome_access_authorized"],
        **bound,
        "declared_paths": declared_paths,
        "frozen_constants": frozen_constants,
        "implementation_sha256": implementation_sha,
        **expected,
        "designation_sha256": _sha256(DESIGNATION),
    }


def _write_refusal(path: Path, *, stage: str, code: str, message: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusal artifact already exists: {path}")
    record = {
        "schema": "hao-gse164378-refusal/1.0",
        "status": "REFUSED",
        "stage": stage,
        "code": code,
        "message": message,
        "candidate": "GSE164378",
        "runner_sha256": _sha256(Path(__file__)),
        "protocol_sha256": _sha256(PROTOCOL),
        "designation_sha256": _sha256(DESIGNATION),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def prepare(args: argparse.Namespace) -> None:
    provenance = preflight(require_sealed=True)
    reduced = Path(args.reduced)
    if _repo_relative(reduced) != provenance["reduced_path"]:
        raise ValueError("reducer output path differs from the frozen designation")
    if PREPARE_RECORD.exists() or PREPARE_REFUSAL.exists():
        raise FileExistsError("prospective prepare record or refusal already exists")
    command = [
        sys.executable,
        str(REDUCER),
        "--rna-barcodes",
        str(Path(args.rna_barcodes)),
        "--rna-features",
        str(Path(args.rna_features)),
        "--rna-matrix",
        str(Path(args.rna_matrix)),
        "--adt-barcodes",
        str(Path(args.adt_barcodes)),
        "--adt-features",
        str(Path(args.adt_features)),
        "--adt-matrix",
        str(Path(args.adt_matrix)),
        "--metadata",
        str(Path(args.metadata)),
        "--aliases",
        str(ALIASES),
        "--source-manifest",
        str(SOURCE_MANIFEST),
        "--output",
        str(reduced),
    ]
    try:
        subprocess.run(command, check=True)
        bundle = _reducer_artifact_bundle(reduced)
        record = {
            "schema": "hao-gse164378-prepare-record/1.0",
            "status": "PREPARED_HELD_PAIRING_NOT_USED_BY_PREDICTION_PATH",
            "reducer_artifacts": bundle,
            "provenance": provenance,
        }
        PREPARE_RECORD.parent.mkdir(parents=True, exist_ok=True)
        PREPARE_RECORD.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    except (
        subprocess.CalledProcessError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        _write_refusal(
            PREPARE_REFUSAL,
            stage="prepare",
            code="SOURCE_OR_REDUCER_FAILURE",
            message=str(error),
        )
        raise


def _prepared_bundle(
    reduced: Path, expected_provenance: dict[str, object]
) -> dict[str, object]:
    record = _read_json(PREPARE_RECORD)
    if record.get("schema") != "hao-gse164378-prepare-record/1.0":
        raise PermissionError("prepare record schema differs")
    if record.get("status") != "PREPARED_HELD_PAIRING_NOT_USED_BY_PREDICTION_PATH":
        raise PermissionError("prepare record is not frozen")
    provenance = record.get("provenance", {})
    if provenance != expected_provenance:
        raise PermissionError("prepare record frozen provenance differs")
    return _require_reducer_artifact_bundle(reduced, record.get("reducer_artifacts"))


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


def _state_thresholds(
    cells: pd.DataFrame,
    markers: pd.DataFrame,
    rna: np.ndarray,
    adt: np.ndarray,
) -> tuple[list[str], list[str], list[str], list[str], np.ndarray, np.ndarray]:
    donor = cells["donor"].astype(str).to_numpy()
    day = cells["day"].to_numpy(dtype=int)
    cell_type = cells["cell_type"].astype(str).to_numpy()
    entity_ids: list[str] = []
    genes: list[str] = []
    entity_lineages: list[str] = []
    excluded: list[str] = []
    rna_states: list[np.ndarray] = []
    adt_states: list[np.ndarray] = []
    all_days = (0, *DAYS)
    for marker_index, marker in markers.iterrows():
        marker_id = str(marker["marker_id"])
        gene = str(marker["gene_symbol"])
        for lineage in LINEAGES:
            entity_id = f"{marker_id}@@{lineage}"
            calibration = (
                np.isin(donor, DEVELOPMENT)
                & (day == 0)
                & (cell_type == lineage)
            )
            if np.count_nonzero(calibration) < MINIMUM_CELLS:
                excluded.append(entity_id)
                continue
            rna_cut = np.quantile(
                rna[marker_index, calibration], [1 / 3, 2 / 3]
            )
            adt_cut = np.quantile(
                adt[marker_index, calibration], [1 / 3, 2 / 3]
            )
            if rna_cut[0] >= rna_cut[1] or adt_cut[0] >= adt_cut[1]:
                excluded.append(entity_id)
                continue
            rna_state = np.sum(rna[marker_index, :, None] >= rna_cut[None, :], axis=1)
            adt_state = np.sum(adt[marker_index, :, None] >= adt_cut[None, :], axis=1)
            supported = True
            for unit in DEVELOPMENT + HELD:
                for time in all_days:
                    mask = (
                        (donor == unit)
                        & (day == time)
                        & (cell_type == lineage)
                    )
                    if np.count_nonzero(mask) < MINIMUM_CELLS:
                        supported = False
                        break
                    for state in range(3):
                        if (
                            np.mean(rna_state[mask] == state) < 0.05
                            or np.mean(adt_state[mask] == state) < 0.05
                        ):
                            supported = False
                            break
                    if not supported:
                        break
                if not supported:
                    break
            if not supported:
                excluded.append(entity_id)
                continue
            entity_ids.append(entity_id)
            genes.append(gene)
            entity_lineages.append(lineage)
            rna_states.append(rna_state)
            adt_states.append(adt_state)
    if len(set(genes)) < 12:
        raise ValueError(
            "fewer than 12 unique cognate markers pass the frozen marginal support rule"
        )
    return (
        entity_ids,
        genes,
        entity_lineages,
        excluded,
        np.asarray(rna_states, dtype=int),
        np.asarray(adt_states, dtype=int),
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
        (basis.T @ (rows / rows.sum()), basis.T @ (columns / columns.sum()))
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
    total = float(margins["total"])
    probability = table / total
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
        ) / np.sqrt(total)
        result[f"{residual}_raw"] = raw
    return result


def _build_fields(
    cells: pd.DataFrame,
    entity_ids: list[str],
    entity_lineages: list[str],
    rna_state: np.ndarray,
    adt_state: np.ndarray,
    *,
    open_held_pairing: bool,
) -> dict[str, np.ndarray]:
    donors = DEVELOPMENT + HELD
    entities = len(entity_ids)
    blocks = len(BLOCKS)
    field_shape = (len(donors), entities, blocks, 2, 2)
    residual_shape = (len(donors), entities, blocks, 3, 3)
    scalar_shape = (len(donors), entities, blocks)
    baseline_shape = (len(donors), entities, 2, 2)
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
        "field_null": np.full(field_shape, np.nan),
        "pearson_null": np.full(residual_shape, np.nan),
        "deviance_null": np.full(residual_shape, np.nan),
        "rows": np.full((*scalar_shape, 3), np.nan),
        "columns": np.full((*scalar_shape, 3), np.nan),
        "total": np.full(scalar_shape, np.nan),
        "endpoint": np.full((*scalar_shape, 4), np.nan),
        "table": np.full((*scalar_shape, 3, 3), np.nan),
        "baseline_field": np.full(baseline_shape, np.nan),
        "baseline_field_variance": np.full(baseline_shape, np.nan),
    }
    donor_values = cells["donor"].astype(str).to_numpy()
    day_values = cells["day"].to_numpy(dtype=int)
    cell_type = cells["cell_type"].astype(str).to_numpy()
    for donor_index, donor in enumerate(donors):
        held_sealed = donor in HELD and not open_held_pairing
        for entity_index, (entity_id, lineage) in enumerate(
            zip(entity_ids, entity_lineages)
        ):
            common = (donor_values == donor) & (cell_type == lineage)
            baseline_mask = common & (day_values == 0)
            if np.count_nonzero(baseline_mask) < MINIMUM_CELLS:
                raise ValueError("frozen donor-day-lineage support unexpectedly failed")
            if not held_sealed:
                key = f"{donor}\0{entity_id}\0day0"
                baseline = _table_stats(
                    rna_state[entity_index, baseline_mask],
                    adt_state[entity_index, baseline_mask],
                    _seed("baseline", key),
                )
                location = donor_index, entity_index
                arrays["baseline_field"][location] = baseline["field"]
                arrays["baseline_field_variance"][location] = baseline[
                    "field_variance"
                ]
            for block_index, day in enumerate(BLOCKS):
                mask = (
                    (donor_values == donor)
                    & (day_values == day)
                    & (cell_type == lineage)
                )
                if np.count_nonzero(mask) < MINIMUM_CELLS:
                    raise ValueError("frozen donor-day-lineage support unexpectedly failed")
                first = rna_state[entity_index, mask]
                second = adt_state[entity_index, mask]
                key = f"{donor}\0{entity_id}\0day{day}"
                seed = _seed("post-vaccine", key)
                margins = _margin_stats(first, second, seed)
                location = donor_index, entity_index, block_index
                arrays["field_null"][location] = margins["field_null"]
                arrays["pearson_null"][location] = margins["pearson_null"]
                arrays["deviance_null"][location] = margins["deviance_null"]
                arrays["rows"][location] = margins["rows"]
                arrays["columns"][location] = margins["columns"]
                arrays["total"][location] = margins["total"]
                arrays["endpoint"][location] = margins["endpoint"]
                if held_sealed:
                    continue
                challenged = _table_stats(first, second, seed)
                arrays["field"][location] = challenged["field"]
                arrays["field_destroyed"][location] = challenged["field_destroyed"]
                arrays["field_variance"][location] = challenged["field_variance"]
                arrays["covariance"][location] = challenged["covariance"]
                arrays["table"][location] = challenged["table"]
                for residual in ("pearson", "deviance"):
                    arrays[residual][location] = challenged[residual]
                    arrays[f"{residual}_destroyed"][location] = challenged[
                        f"{residual}_destroyed"
                    ]
                    arrays[f"{residual}_variance"][location] = challenged[
                        f"{residual}_variance"
                    ]
    return arrays


def _embedding_laplacian(
    genes: list[str],
    entity_lineages: list[str],
    *,
    gene_membership_permutation: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    if len(genes) != len(entity_lineages):
        raise ValueError("gene and lineage entity axes differ")
    cluster_genes = list(dict.fromkeys(genes))
    if gene_membership_permutation is None:
        gene_membership_permutation = np.arange(len(cluster_genes))
    gene_membership_permutation = np.asarray(gene_membership_permutation, dtype=int)
    if (
        gene_membership_permutation.shape != (len(cluster_genes),)
        or set(gene_membership_permutation.tolist()) != set(range(len(cluster_genes)))
    ):
        raise ValueError("gene membership permutation is not a cluster permutation")
    reassigned = dict(
        zip(
            cluster_genes,
            [cluster_genes[index] for index in gene_membership_permutation],
        )
    )
    embedding_genes = [reassigned[gene] for gene in genes]
    with np.load(SCGPT, allow_pickle=False) as archive:
        names = archive["gene_names"].astype(str)
        embedding = np.asarray(archive["embedding"], dtype=float)
    lookup = {name.upper(): index for index, name in enumerate(names)}
    incidence = np.zeros((len(genes), len(genes)), dtype=float)
    covered = [
        index for index, gene in enumerate(embedding_genes) if gene.upper() in lookup
    ]
    if covered:
        values = np.asarray(
            [embedding[lookup[embedding_genes[index].upper()]] for index in covered]
        )
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        similarity = values @ values.T
        for local, entity in enumerate(covered):
            order = np.lexsort((np.asarray(covered), -similarity[local]))
            neighbors = [index for index in order if index != local][
                : min(6, len(covered) - 1)
            ]
            incidence[entity, entity] = 1.0
            incidence[[covered[index] for index in neighbors], entity] = 1.0
    for index in set(range(len(genes))) - set(covered):
        incidence[index, index] = 1.0
    represented_lineages = [
        lineage for lineage in LINEAGES if lineage in set(entity_lineages)
    ]
    lineage_incidence = np.zeros((len(genes), len(represented_lineages)), dtype=float)
    for entity, lineage in enumerate(entity_lineages):
        lineage_incidence[entity, represented_lineages.index(lineage)] = 1.0
    incidence = np.concatenate((incidence, lineage_incidence), axis=1)
    return normalized_hypergraph_laplacian(incidence), {
        "path": str(SCGPT.relative_to(ROOT)),
        "sha256": _sha256(SCGPT),
        "covered_entities": len(covered),
        "external_gene_neighbors": 6,
        "gene_clusters": cluster_genes,
        "gene_membership_permutation": gene_membership_permutation.tolist(),
        "lineage_hyperedges": represented_lineages,
        "lineage_memberships": {
            lineage: [
                index
                for index, entity_lineage in enumerate(entity_lineages)
                if entity_lineage == lineage
            ]
            for lineage in represented_lineages
        },
    }


def _structured(
    values: np.ndarray,
    variance: np.ndarray,
    laplacian: np.ndarray,
    *,
    nuclear: float,
    graph: float,
) -> tuple[np.ndarray, dict[str, object]]:
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
    return fit.coefficient, {
        "converged": fit.converged,
        "iterations": fit.iterations,
        "relative_step": fit.relative_step,
        "objective": fit.objective,
        "effective_rank": fit.effective_rank,
        "singular_values": fit.singular_values.tolist(),
        "nuclear_penalty": fit.nuclear_penalty,
        "graph_penalty": fit.graph_penalty,
    }


def _expanded_baseline(values: np.ndarray) -> np.ndarray:
    return np.repeat(values[:, :, None], len(BLOCKS), axis=2)


def _fit_predictions(
    arrays: dict[str, np.ndarray], genes: list[str], entity_lineages: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    develop = slice(0, len(DEVELOPMENT))
    held = slice(len(DEVELOPMENT), len(DEVELOPMENT) + len(HELD))
    laplacian, embedding = _embedding_laplacian(genes, entity_lineages)
    diagnostics: dict[str, dict[str, object]] = {}

    def structured(
        name: str,
        values: np.ndarray,
        variance: np.ndarray,
        graph_laplacian: np.ndarray,
        *,
        nuclear: float,
        graph: float,
    ) -> np.ndarray:
        coefficient, diagnostic = _structured(
            values,
            variance,
            graph_laplacian,
            nuclear=nuclear,
            graph=graph,
        )
        diagnostics[name] = diagnostic
        return coefficient

    def mean_flat(name: str) -> np.ndarray:
        return arrays[name][develop].mean(axis=0).reshape(len(genes), -1)

    def variance_flat(name: str) -> np.ndarray:
        return (
            arrays[name][develop].sum(axis=0).reshape(len(genes), -1)
            / len(DEVELOPMENT) ** 2
        )

    field = mean_flat("field")
    variance = variance_flat("field_variance")
    signal_energy = float(np.sum(field**2))
    scalar = max(
        0.0,
        1.0 - float(np.sum(variance)) / max(signal_energy, np.finfo(float).eps),
    )
    cluster_count = len(dict.fromkeys(genes))
    generator = np.random.default_rng(_seed("hao-membership-permuted"))
    permutation = generator.permutation(cluster_count)
    permuted_laplacian, permuted_embedding = _embedding_laplacian(
        genes,
        entity_lineages,
        gene_membership_permutation=permutation,
    )
    predictions = {
        "field_direct": field,
        "field_zero": np.zeros_like(field),
        "field_scalar": scalar * field,
        "field_nuclear": structured(
            "field_nuclear",
            field,
            variance,
            laplacian,
            nuclear=NUCLEAR_FRACTION,
            graph=0.0,
        ),
        "field_hypergraph": structured(
            "field_hypergraph",
            field,
            variance,
            laplacian,
            nuclear=0.0,
            graph=GRAPH_PENALTY,
        ),
        "field_membership_permuted": structured(
            "field_membership_permuted",
            field,
            variance,
            permuted_laplacian,
            nuclear=NUCLEAR_FRACTION,
            graph=GRAPH_PENALTY,
        ),
        "field_primary": structured(
            "field_primary",
            field,
            variance,
            laplacian,
            nuclear=NUCLEAR_FRACTION,
            graph=GRAPH_PENALTY,
        ),
        "field_destroyed": structured(
            "field_destroyed",
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
        predictions[f"{residual}_structured"] = structured(
            f"{residual}_structured",
            values,
            residual_variance,
            laplacian,
            nuclear=NUCLEAR_FRACTION,
            graph=GRAPH_PENALTY,
        )
    x_train = arrays["endpoint"][develop].reshape(-1, len(BLOCKS) * 4)
    y_train = arrays["field"][develop].reshape(-1, len(BLOCKS) * 4)
    endpoint = Ridge(alpha=0.1, fit_intercept=True).fit(x_train, y_train)
    predictions["field_endpoint_ridge"] = endpoint.predict(
        arrays["endpoint"][held].reshape(-1, len(BLOCKS) * 4)
    ).reshape(len(HELD), len(genes), len(BLOCKS) * 4)

    baseline = _expanded_baseline(arrays["baseline_field"][develop])
    baseline_variance = _expanded_baseline(
        arrays["baseline_field_variance"][develop]
    )
    changes = arrays["field"][develop] - baseline
    change_variance = arrays["field_variance"][develop] + baseline_variance
    change_mean = changes.mean(axis=0).reshape(len(genes), -1)
    change_var = change_variance.sum(axis=0).reshape(len(genes), -1) / len(
        DEVELOPMENT
    ) ** 2
    secondary = {
        "field_change_direct": change_mean,
        "field_change_primary": structured(
            "field_change_primary",
            change_mean,
            change_var,
            laplacian,
            nuclear=NUCLEAR_FRACTION,
            graph=GRAPH_PENALTY,
        ),
    }
    embedding["variance_scalar"] = scalar
    embedding["membership_permutation"] = permutation.tolist()
    embedding["membership_permuted_control"] = permuted_embedding
    embedding["destroyed_link"] = (
        "identical inverse-variance, nuclear, and hypergraph fit as field_primary"
    )
    embedding["structured_fit_diagnostics"] = diagnostics
    return predictions, secondary, embedding


def _field_table(
    centered: np.ndarray,
    null_mean: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    return field_coordinates_to_table(centered + null_mean, rows, columns)


def _covariance_table(
    covariance: np.ndarray, rows: np.ndarray, columns: np.ndarray
) -> np.ndarray:
    total = float(rows.sum())
    probability = (
        np.outer(rows / total, columns / total)
        + field_from_coordinates(covariance)
    )
    floor = np.finfo(float).eps * max(1.0, total)
    return ipf_to_margins(np.maximum(probability * total, floor), rows, columns)


def _residual_table(
    centered: np.ndarray,
    null_mean: np.ndarray,
    total: float,
    rows: np.ndarray,
    columns: np.ndarray,
    residual: str,
) -> np.ndarray:
    raw = centered * np.sqrt(total) + null_mean
    return residual_coordinates_to_table(raw, rows, columns, residual=residual)


def _predict_tables(
    arrays: dict[str, np.ndarray], predictions: dict[str, np.ndarray], markers: int
) -> dict[str, np.ndarray]:
    held_offset = len(DEVELOPMENT)
    result = {
        name: np.empty((len(HELD), markers, len(BLOCKS), 3, 3))
        for name in predictions
    }
    for donor in range(len(HELD)):
        source = held_offset + donor
        for marker in range(markers):
            for block in range(len(BLOCKS)):
                location = source, marker, block
                rows = arrays["rows"][location]
                columns = arrays["columns"][location]
                for method, estimate in predictions.items():
                    flat = estimate[donor, marker] if estimate.ndim == 3 else estimate[marker]
                    if method.startswith("field_"):
                        table = _field_table(
                            flat.reshape(len(BLOCKS), 2, 2)[block],
                            arrays["field_null"][location],
                            rows,
                            columns,
                        )
                    elif method.startswith("covariance_"):
                        table = _covariance_table(
                            flat.reshape(len(BLOCKS), 2, 2)[block], rows, columns
                        )
                    elif method.startswith("pearson_"):
                        table = _residual_table(
                            flat.reshape(len(BLOCKS), 3, 3)[block],
                            arrays["pearson_null"][location],
                            arrays["total"][location],
                            rows,
                            columns,
                            "pearson",
                        )
                    elif method.startswith("deviance_"):
                        table = _residual_table(
                            flat.reshape(len(BLOCKS), 3, 3)[block],
                            arrays["deviance_null"][location],
                            arrays["total"][location],
                            rows,
                            columns,
                            "deviance",
                        )
                    else:
                        raise ValueError(f"unknown prediction family: {method}")
                    result[method][donor, marker, block] = table
    return result


def _score_tables(
    arrays: dict[str, np.ndarray], predicted_tables: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    held_truth = arrays["table"][len(DEVELOPMENT) :]
    if not np.isfinite(held_truth).all():
        raise ValueError("held post-vaccine pairing was not opened for scoring")
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


def _interval(values: np.ndarray, coverage: float = 0.95) -> list[float]:
    alpha = 1.0 - coverage
    return np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0]).tolist()


def _representation_metrics(
    prediction: np.ndarray, truth: np.ndarray
) -> dict[str, float]:
    estimate = np.broadcast_to(prediction, truth.shape)
    denominator = max(float(np.mean(truth**2)), np.finfo(float).eps)
    return {
        "pooled_pearson": _correlation(estimate, truth),
        "standardized_rmse": float(
            np.sqrt(np.mean((estimate - truth) ** 2) / denominator
        )),
    }


def summarize(
    arrays: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    secondary: dict[str, np.ndarray],
    losses: dict[str, np.ndarray],
    *,
    marker_clusters: list[str],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    truth = arrays["field"][len(DEVELOPMENT) :].reshape(
        len(HELD), len(predictions["field_direct"]), len(BLOCKS) * 4
    )
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
    baseline = _expanded_baseline(arrays["baseline_field"][len(DEVELOPMENT) :])
    change_truth = (arrays["field"][len(DEVELOPMENT) :] - baseline).reshape(
        len(HELD), truth.shape[1], len(BLOCKS) * 4
    )
    change_primary = secondary["field_change_primary"]
    change_direct = secondary["field_change_direct"]

    if len(marker_clusters) != truth.shape[1]:
        raise ValueError("marker cluster axis differs from entity axis")
    cluster_names = list(dict.fromkeys(marker_clusters))
    cluster_indices = {
        name: np.flatnonzero(np.asarray(marker_clusters) == name)
        for name in cluster_names
    }
    rng = np.random.default_rng(SEED)
    correlations = np.empty(BOOTSTRAPS)
    change_correlations = np.empty(BOOTSTRAPS)
    primary_minus_destroyed = np.empty(BOOTSTRAPS)
    primary_minus_classical = np.empty(BOOTSTRAPS)
    primary_minus_matched = np.empty(BOOTSTRAPS)
    for draw in range(BOOTSTRAPS):
        selected = rng.integers(0, len(cluster_names), len(cluster_names))
        index = np.concatenate(
            [cluster_indices[cluster_names[position]] for position in selected]
        )
        correlations[draw] = _correlation(
            np.broadcast_to(primary[index], truth[:, index].shape), truth[:, index]
        )
        change_correlations[draw] = _correlation(
            np.broadcast_to(change_primary[index], change_truth[:, index].shape),
            change_truth[:, index],
        )
        primary_loss = per_marker_loss["field_primary"][index].mean()
        primary_minus_destroyed[draw] = (
            primary_loss - per_marker_loss["field_destroyed"][index].mean()
        )
        selected_classical = min(
            classical, key=lambda name: float(per_marker_loss[name][index].mean())
        )
        primary_minus_classical[draw] = (
            primary_loss - per_marker_loss[selected_classical][index].mean()
        )
        selected_matched = min(
            matched, key=lambda name: float(per_marker_loss[name][index].mean())
        )
        primary_minus_matched[draw] = (
            primary_loss - per_marker_loss[selected_matched][index].mean()
        )

    correlation_ci = _interval(correlations, FAMILY_GATE_COVERAGE)
    destroyed_ci = _interval(primary_minus_destroyed, FAMILY_GATE_COVERAGE)
    classical_ci = _interval(primary_minus_classical, FAMILY_GATE_COVERAGE)
    matched_ci = _interval(primary_minus_matched, FAMILY_GATE_COVERAGE)
    passed = (
        correlation_ci[0] > 0.0
        and all(value > 0.0 for value in donor_r)
        and destroyed_ci[1] < 0.0
        and classical_ci[1] < 0.0
        and matched_ci[1] < 0.0
    )
    representation = {
        "field_primary": _representation_metrics(primary, truth),
        "field_direct": _representation_metrics(predictions["field_direct"], truth),
    }
    for family in ("pearson", "deviance"):
        residual_truth = arrays[family][len(DEVELOPMENT) :].reshape(
            len(HELD), truth.shape[1], len(BLOCKS) * 9
        )
        for method in (f"{family}_direct", f"{family}_structured"):
            representation[method] = _representation_metrics(
                predictions[method], residual_truth
            )
    summary = {
        "primary_method": "field_primary",
        "primary_estimand": "absolute post-vaccine RNA-protein joint tables",
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
        "secondary_field_change": {
            "pooled_held_correlation_primary": _correlation(
                np.broadcast_to(change_primary, change_truth.shape), change_truth
            ),
            "pooled_held_correlation_direct": _correlation(
                np.broadcast_to(change_direct, change_truth.shape), change_truth
            ),
            "primary_bootstrap_95_ci": _interval(
                change_correlations, DESCRIPTIVE_COVERAGE
            ),
            "held_donor_correlations_primary": dict(
                zip(
                    HELD,
                    [
                        _correlation(change_primary, change_truth[index])
                        for index in range(len(HELD))
                    ],
                )
            ),
            "gate_role": "descriptive secondary; never used for promotion",
        },
        "bootstrap_unit": "cognate marker cluster retaining every supported lineage pair",
        "bootstrap_draws": BOOTSTRAPS,
        "inference_target": "conditional on the deposited held donors P5, P3, and P2; no donor-population inference",
        "multiplicity": {
            "scoreable_confirmatory_candidates": [
                "Lawlor HCA PBMC",
                "Hao GSE164378",
            ],
            "reporting_rule": "execute and report both; no stopping after a pass",
            "family_status": "closed; later candidates require a newly declared alpha-spending family",
            "directional_endpoint_rule": "each 95% two-sided endpoint is a one-sided alpha 0.025 test; Bonferroni over two candidates controls familywise alpha at 0.05",
            "poki_status": "not in the scoreable family because its frozen run produced no inferential test and cannot be promoted",
        },
        "gate_passed": passed,
    }
    bootstrap = {
        "pooled_field_correlation": correlations,
        "pooled_field_change_correlation": change_correlations,
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
    list[str],
    list[str],
    np.ndarray,
    np.ndarray,
]:
    cells, marker_frame, rna, adt = _load_reduced(reduced)
    entity_ids, genes, entity_lineages, excluded, rna_state, adt_state = (
        _state_thresholds(cells, marker_frame, rna, adt)
    )
    return (
        cells,
        entity_ids,
        genes,
        entity_lineages,
        excluded,
        rna_state,
        adt_state,
    )


def predict(args: argparse.Namespace) -> None:
    provenance = preflight(require_sealed=True)
    output = Path(args.output)
    if _repo_relative(output) != provenance["prediction_path"]:
        raise ValueError("prediction output path differs from the frozen designation")
    if output.exists() or PREDICTION_REFUSAL.exists():
        raise FileExistsError("prospective prediction or refusal already exists")
    reduced = Path(args.reduced)
    if _repo_relative(reduced) != provenance["reduced_path"]:
        raise ValueError("reducer output path differs from the frozen designation")
    try:
        bundle = _prepared_bundle(reduced, provenance)
        (
            cells,
            entity_ids,
            genes,
            entity_lineages,
            excluded,
            rna_state,
            adt_state,
        ) = _analysis_data(reduced)
        sealed = _build_fields(
            cells,
            entity_ids,
            entity_lineages,
            rna_state,
            adt_state,
            open_held_pairing=False,
        )
        predictions, secondary, embedding = _fit_predictions(
            sealed, genes, entity_lineages
        )
        _require_reducer_artifact_bundle(reduced, bundle)
    except (OSError, PermissionError, ValueError) as error:
        code = (
            "DEVELOPMENT_SUPPORT_FAILURE"
            if "support" in str(error).lower() or "fewer than 12" in str(error)
            else "PRETRUTH_INTEGRITY_OR_ANALYSIS_FAILURE"
        )
        _write_refusal(
            PREDICTION_REFUSAL,
            stage="predict",
            code=code,
            message=str(error),
        )
        raise
    except RuntimeError as error:
        _write_refusal(
            PREDICTION_REFUSAL,
            stage="predict",
            code="OPTIMIZATION_FAILURE",
            message=str(error),
        )
        raise
    try:
        record = {
            "schema": "hao-gse164378-predictions/1.0",
            "status": "PREDICTIONS_FROZEN_HELD_PAIRING_NOT_USED",
            "scope": "held-donor absolute post-vaccine RNA-protein coupling",
            "entity_ids": entity_ids,
            "genes": genes,
            "entity_lineages": entity_lineages,
            "marginal_support_exclusions": excluded,
            "development_donors": list(DEVELOPMENT),
            "held_donors": list(HELD),
            "blocks": [f"day{day}" for day in BLOCKS],
            "held_rows": sealed["rows"][len(DEVELOPMENT) :].tolist(),
            "held_columns": sealed["columns"][len(DEVELOPMENT) :].tolist(),
            "predictions": {
                name: value.tolist() for name, value in predictions.items()
            },
            "secondary_predictions": {
                name: value.tolist() for name, value in secondary.items()
            },
            "predicted_tables_stored": False,
            "table_reconstruction_rule": "score recomputes every table from locked coordinates, frozen held margins, and the bound implementation before the held-pairing scoring path is invoked",
            "embedding": embedding,
            "provenance": {**provenance, "reducer_artifacts": bundle},
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    except (OSError, TypeError, ValueError) as error:
        _write_refusal(
            PREDICTION_REFUSAL,
            stage="predict",
            code="PREDICTION_SERIALIZATION_FAILURE",
            message=str(error),
        )
        raise


def public_bind(args: argparse.Namespace) -> None:
    provenance = preflight(require_sealed=True)
    prediction = Path(args.predictions)
    if _repo_relative(prediction) != _repo_relative(PREDICTION_PATH):
        raise ValueError("prediction path differs from the frozen designation")
    if SCORE_AUTHORIZATION.exists():
        raise FileExistsError(f"score authorization already exists: {SCORE_AUTHORIZATION}")
    template = _read_json(AUTH_TEMPLATE)
    if template.get("schema") != "hao-gse164378-score-authorization-template/1.0":
        raise ValueError("score authorization template schema differs")
    commit = args.public_commit
    url = args.public_url
    relative = _repo_relative(prediction)
    _require_github_commit_url(url, commit, blob_path=relative)
    _locked_predictions(prediction, provenance)
    record = {
        "schema": "hao-gse164378-score-authorization/1.0",
        "status": "SEALED",
        "outcome_access_authorized": True,
        "candidate": "GSE164378",
        "prediction_path": relative,
        "prediction_sha256": _sha256(prediction),
        "prediction_bytes": prediction.stat().st_size,
        "prediction_public_url": url,
        "prediction_public_commit": commit,
        "runner": str(Path(__file__).relative_to(ROOT)),
        "runner_sha256": _sha256(Path(__file__)),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": _sha256(PROTOCOL),
        "frozen_provenance": provenance,
        "publication_required_before_score": True,
    }
    SCORE_AUTHORIZATION.parent.mkdir(parents=True, exist_ok=True)
    SCORE_AUTHORIZATION.write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )


def _require_score_authorization(
    prediction_path: Path,
    authorization_path: Path,
    expected_provenance: dict[str, object],
) -> dict[str, object]:
    authorization = _read_json(authorization_path)
    if authorization.get("schema") != "hao-gse164378-score-authorization/1.0":
        raise PermissionError("score authorization schema differs")
    if authorization.get("status") != "SEALED":
        raise PermissionError("score authorization is not SEALED")
    if authorization.get("outcome_access_authorized") is not True:
        raise PermissionError("score authorization forbids held pairing access")
    if authorization.get("prediction_sha256") != _sha256(prediction_path):
        raise PermissionError("the prediction JSON hash differs from authorization")
    if authorization.get("prediction_bytes") != prediction_path.stat().st_size:
        raise PermissionError("the prediction JSON byte count differs from authorization")
    relative = _repo_relative(prediction_path)
    if authorization.get("prediction_path") != relative:
        raise PermissionError("the prediction path differs from authorization")
    _require_github_commit_url(
        authorization.get("prediction_public_url"),
        authorization.get("prediction_public_commit"),
        blob_path=relative,
    )
    if authorization.get("runner_sha256") != _sha256(Path(__file__)):
        raise PermissionError("the scoring runner differs from authorization")
    if authorization.get("protocol_sha256") != _sha256(PROTOCOL):
        raise PermissionError("the scoring protocol differs from authorization")
    if authorization.get("frozen_provenance") != expected_provenance:
        raise PermissionError("score authorization frozen provenance differs")
    return authorization


def _require_score_release(
    authorization_path: Path,
    release_path: Path,
    expected_provenance: dict[str, object],
) -> dict[str, object]:
    release = _read_json(release_path)
    if release.get("schema") != "hao-gse164378-score-release/1.0":
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
    if release.get("frozen_provenance") != expected_provenance:
        raise PermissionError("score release frozen provenance differs")
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
    _require_score_authorization(prediction_path, authorization_path, provenance)
    commit = args.authorization_public_commit
    url = args.authorization_public_url
    relative = _repo_relative(authorization_path)
    _require_github_commit_url(url, commit, blob_path=relative)
    release = {
        "schema": "hao-gse164378-score-release/1.0",
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
        "frozen_provenance": provenance,
    }
    SCORE_RELEASE.parent.mkdir(parents=True, exist_ok=True)
    SCORE_RELEASE.write_text(json.dumps(release, indent=2, allow_nan=False) + "\n")


def _locked_predictions(
    path: Path, expected_provenance: dict[str, object]
) -> tuple[dict[str, object], dict[str, np.ndarray], dict[str, np.ndarray]]:
    record = _read_json(path)
    if record.get("schema") != "hao-gse164378-predictions/1.0":
        raise ValueError("prediction JSON schema differs")
    if record.get("status") != "PREDICTIONS_FROZEN_HELD_PAIRING_NOT_USED":
        raise ValueError("prediction JSON is not frozen")
    if record.get("predicted_tables_stored") is not False:
        raise ValueError("prediction JSON must not store predicted tables")
    predictions = {
        name: np.asarray(value, dtype=float)
        for name, value in record["predictions"].items()
    }
    secondary = {
        name: np.asarray(value, dtype=float)
        for name, value in record["secondary_predictions"].items()
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
    expected = field_methods | residual_methods | {
        "field_endpoint_ridge",
        "covariance_direct",
    }
    if set(predictions) != expected:
        raise ValueError("locked prediction method set differs from the protocol")
    if set(secondary) != {"field_change_direct", "field_change_primary"}:
        raise ValueError("locked secondary method set differs from the protocol")
    entity_count = len(record["entity_ids"])
    if (
        len(record.get("genes", [])) != entity_count
        or len(record.get("entity_lineages", [])) != entity_count
    ):
        raise ValueError("locked prediction entity metadata axes differ")
    for name, values in predictions.items():
        if name == "field_endpoint_ridge":
            expected_shape = (len(HELD), entity_count, len(BLOCKS) * 4)
        elif name in residual_methods:
            expected_shape = (entity_count, len(BLOCKS) * 9)
        else:
            expected_shape = (entity_count, len(BLOCKS) * 4)
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise ValueError(f"locked prediction shape differs for {name}")
    for name, values in secondary.items():
        if values.shape != (entity_count, len(BLOCKS) * 4) or not np.isfinite(
            values
        ).all():
            raise ValueError(f"locked secondary shape differs for {name}")
    prediction_provenance = record.get("provenance")
    if not isinstance(prediction_provenance, dict):
        raise ValueError("prediction JSON lacks frozen provenance")
    base_provenance = {
        key: value
        for key, value in prediction_provenance.items()
        if key != "reducer_artifacts"
    }
    if base_provenance != expected_provenance:
        raise ValueError("prediction JSON frozen provenance differs")
    return record, predictions, secondary


def score(args: argparse.Namespace) -> None:
    provenance = preflight(require_sealed=True)
    prediction_path = Path(args.predictions)
    if _repo_relative(prediction_path) != provenance["prediction_path"]:
        raise ValueError("prediction path differs from the frozen designation")
    authorization_path = Path(args.authorization)
    if _repo_relative(authorization_path) != provenance["score_authorization"]:
        raise ValueError("score authorization path differs from the frozen designation")
    authorization = _require_score_authorization(
        prediction_path, authorization_path, provenance
    )
    release_path = Path(args.release)
    if _repo_relative(release_path) != provenance["score_release"]:
        raise ValueError("score release path differs from the frozen designation")
    release = _require_score_release(authorization_path, release_path, provenance)
    output = Path(args.output)
    if _repo_relative(output) != provenance["score_output"]:
        raise ValueError("score output path differs from the frozen designation")
    arrays_path = ARRAYS_PATH
    if _repo_relative(arrays_path) != provenance["score_arrays"]:
        raise ValueError("score arrays path differs from the frozen designation")
    if output.exists() or arrays_path.exists() or SCORE_REFUSAL.exists():
        raise FileExistsError("prospective score output or refusal already exists")
    reduced = Path(args.reduced)
    if _repo_relative(reduced) != provenance["reduced_path"]:
        raise ValueError("reducer output path differs from the frozen designation")
    try:
        record, predictions, secondary = _locked_predictions(
            prediction_path, provenance
        )
        reducer_artifacts = record.get("provenance", {}).get("reducer_artifacts")
        _require_reducer_artifact_bundle(reduced, reducer_artifacts)
        (
            cells,
            entity_ids,
            genes,
            entity_lineages,
            excluded,
            rna_state,
            adt_state,
        ) = _analysis_data(reduced)
        expected_metadata = {
            "entity_ids": entity_ids,
            "genes": genes,
            "entity_lineages": entity_lineages,
            "marginal_support_exclusions": excluded,
            "development_donors": list(DEVELOPMENT),
            "held_donors": list(HELD),
            "blocks": [f"day{day}" for day in BLOCKS],
        }
        for name, expected in expected_metadata.items():
            if record.get(name) != expected:
                raise ValueError(f"locked prediction {name} differs from scoring input")
        _require_reducer_artifact_bundle(reduced, reducer_artifacts)
        sealed = _build_fields(
            cells,
            entity_ids,
            entity_lineages,
            rna_state,
            adt_state,
            open_held_pairing=False,
        )
        np.testing.assert_allclose(
            np.asarray(record["held_rows"]), sealed["rows"][len(DEVELOPMENT) :]
        )
        np.testing.assert_allclose(
            np.asarray(record["held_columns"]),
            sealed["columns"][len(DEVELOPMENT) :],
        )
        predicted_tables = _predict_tables(sealed, predictions, len(entity_ids))
        for values in predicted_tables.values():
            np.testing.assert_allclose(
                values.sum(axis=-1), sealed["rows"][len(DEVELOPMENT) :]
            )
            np.testing.assert_allclose(
                values.sum(axis=-2), sealed["columns"][len(DEVELOPMENT) :]
            )
        _require_reducer_artifact_bundle(reduced, reducer_artifacts)
        arrays = _build_fields(
            cells,
            entity_ids,
            entity_lineages,
            rna_state,
            adt_state,
            open_held_pairing=True,
        )
        losses = _score_tables(arrays, predicted_tables)
        summary, bootstrap = summarize(
            arrays, predictions, secondary, losses, marker_clusters=genes
        )
    except (
        KeyError,
        OSError,
        PermissionError,
        ValueError,
        RuntimeError,
        AssertionError,
    ) as error:
        _write_refusal(
            SCORE_REFUSAL,
            stage="score",
            code="RECONSTRUCTION_OR_SCORING_FAILURE",
            message=str(error),
        )
        raise

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            arrays_path,
            entity_ids=np.asarray(entity_ids),
            genes=np.asarray(genes),
            entity_lineages=np.asarray(entity_lineages),
            held_donors=np.asarray(HELD),
            blocks=np.asarray([f"day{day}" for day in BLOCKS]),
            held_field_truth=arrays["field"][len(DEVELOPMENT) :],
            held_baseline_field_truth=arrays["baseline_field"][len(DEVELOPMENT) :],
            **{f"loss_{name}": value for name, value in losses.items()},
            **{f"bootstrap_{name}": value for name, value in bootstrap.items()},
        )
        result = {
            "schema": "hao-gse164378-confirmation/1.0",
            "status": "PASS" if summary["gate_passed"] else "FAIL",
            "scope": "held-donor absolute post-vaccine RNA-protein coupling",
            "summary": summary,
            "prediction": {
                "path": _repo_relative(prediction_path),
                "sha256": _sha256(prediction_path),
                "bytes": prediction_path.stat().st_size,
                "public_url": authorization["prediction_public_url"],
                "public_commit": authorization["prediction_public_commit"],
            },
            "authorization": {
                "path": _repo_relative(authorization_path),
                "sha256": _sha256(authorization_path),
                "bytes": authorization_path.stat().st_size,
                "public_url": release["authorization_public_url"],
                "public_commit": release["authorization_public_commit"],
            },
            "arrays": {
                "path": _repo_relative(arrays_path),
                "sha256": _sha256(arrays_path),
                "bytes": arrays_path.stat().st_size,
            },
            "source_acquisition": _read_json(reduced / "source_acquisition.json"),
            "provenance": {**provenance, "reducer_artifacts": reducer_artifacts},
        }
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (KeyError, OSError, TypeError, ValueError) as error:
        _write_refusal(
            SCORE_REFUSAL,
            stage="score",
            code="RESULT_SERIALIZATION_FAILURE",
            message=str(error),
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--require-sealed", action="store_true")

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--rna-barcodes", required=True)
    prepare_parser.add_argument("--rna-features", required=True)
    prepare_parser.add_argument("--rna-matrix", required=True)
    prepare_parser.add_argument("--adt-barcodes", required=True)
    prepare_parser.add_argument("--adt-features", required=True)
    prepare_parser.add_argument("--adt-matrix", required=True)
    prepare_parser.add_argument("--metadata", required=True)
    prepare_parser.add_argument("--reduced", default=str(REDUCED_PATH))

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--reduced", default=str(REDUCED_PATH))
    predict_parser.add_argument("--output", default=str(PREDICTION_PATH))

    bind_parser = subparsers.add_parser("public-bind")
    bind_parser.add_argument("--predictions", default=str(PREDICTION_PATH))
    bind_parser.add_argument("--public-commit", required=True)
    bind_parser.add_argument("--public-url", required=True)

    authorize_parser = subparsers.add_parser("authorize-score")
    authorize_parser.add_argument("--predictions", default=str(PREDICTION_PATH))
    authorize_parser.add_argument("--authorization", default=str(SCORE_AUTHORIZATION))
    authorize_parser.add_argument("--authorization-public-commit", required=True)
    authorize_parser.add_argument("--authorization-public-url", required=True)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--reduced", default=str(REDUCED_PATH))
    score_parser.add_argument("--predictions", default=str(PREDICTION_PATH))
    score_parser.add_argument("--authorization", default=str(SCORE_AUTHORIZATION))
    score_parser.add_argument("--release", default=str(SCORE_RELEASE))
    score_parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(preflight(require_sealed=args.require_sealed), indent=2))
    elif args.command == "prepare":
        prepare(args)
    elif args.command == "predict":
        predict(args)
    elif args.command == "public-bind":
        public_bind(args)
    elif args.command == "authorize-score":
        authorize_score(args)
    elif args.command == "score":
        score(args)


if __name__ == "__main__":
    main()
