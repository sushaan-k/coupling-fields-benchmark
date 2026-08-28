"""Prospective held-batch RNA-protein confirmation on KotliarovPBMCData."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd

from mapreg.classical_residuals import (
    conditional_poisson_residuals,
    poisson_independence_residuals,
)
from mapreg.coupling_fields import (
    association_coordinates,
    association_field,
    conditional_association_coordinates,
    fit_structured_coupling_fields,
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
PROTOCOL = ROOT / "docs/KOTLIAROV_PBMC_HELD_BATCH_CONFIRMATION_PROTOCOL_2026-08-28.md"
DESIGNATION = ROOT / "data/confirmation/kotliarov_pbmc/candidate_designation_v1.json"
AUTH_TEMPLATE = ROOT / "data/confirmation/kotliarov_pbmc/score_authorization_template_v1.json"
SCORE_AUTHORIZATION = ROOT / "data/confirmation/kotliarov_pbmc/score_authorization_v1.json"
SCORE_RELEASE = ROOT / "data/confirmation/kotliarov_pbmc/score_release_v1.json"
SOURCE_MANIFEST = ROOT / "data/development/kotliarov_pbmc/source_manifest_v1.json"
SUPPORT = ROOT / "data/development/kotliarov_pbmc/metadata_support_v1.json"
ALIASES = ROOT / "data/development/kotliarov_pbmc/adt_gene_aliases_v1.tsv"
LINEAGE_MARKERS = ROOT / "data/development/kotliarov_pbmc/lineage_markers_v1.tsv"
REDUCER = ROOT / "experiments/reduce_kotliarov_pbmc.py"
TEST = ROOT / "tests/test_kotliarov_pbmc_confirmation.py"
SCGPT = ROOT / "data/scgpt_gene_embeddings.npz"
SCGPT_MANIFEST = ROOT / "data/scgpt_gene_embeddings_manifest.json"

PREDICTION_BUNDLE = ROOT / "data/development/kotliarov_pbmc/prediction_bundle_v1"
SCORE_BUNDLE = ROOT / "data/confirmation/kotliarov_pbmc/score_bundle_v1"
PREPARE_RECORD = ROOT / "data/development/kotliarov_pbmc/prepared_v1.json"
PREDICTION_PATH = ROOT / "results/kotliarov_pbmc_predictions.json"
OUTPUT = ROOT / "results/kotliarov_pbmc_confirmation.json"
ARRAYS_PATH = ROOT / "results/kotliarov_pbmc_confirmation_arrays.npz"
PREPARE_REFUSAL = ROOT / "results/kotliarov_pbmc_prepare_refusal.json"
PREDICTION_REFUSAL = ROOT / "results/kotliarov_pbmc_prediction_refusal.json"
SCORE_REFUSAL = ROOT / "results/kotliarov_pbmc_score_refusal.json"

IMPLEMENTATION_FILES = (
    ROOT / "mapreg/coupling_fields.py",
    ROOT / "mapreg/classical_residuals.py",
    ROOT / "mapreg/table_prediction.py",
)
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
PREDICTION_MANIFEST = "prediction_manifest.tsv"
SCORE_MANIFEST = "score_manifest.tsv"

SEED = 20260828
PERMUTATIONS = 64
BOOTSTRAPS = 10_000
PSEUDOCOUNT = 0.5
IPF_TOLERANCE = 1e-8
NUCLEAR_GRID = (0.0, 0.03, 0.1, 0.3, 1.0)
GRAPH_GRID = (0.0, 0.5, 2.0, 5.0, 10.0)
DEVELOPMENT = ("200", "207", "212", "233", "237", "245", "256", "261", "273", "277")
HELD = ("201", "205", "215", "229", "234", "236", "250", "268", "279")
EXCLUDED = ("209",)
HELD_HIGH = ("205", "215", "234", "250")
HELD_LOW = ("201", "229", "236", "268", "279")
LINEAGES = ("B", "CD4 T", "CD8 T", "NK", "Monocyte")
MINIMUM_LINEAGES = 4
MINIMUM_LINEAGE_CELLS = 50
MINIMUM_STATE_CELLS = 5
MINIMUM_STATE_FRACTION = 0.02
MINIMUM_MARKERS = 16
MINIMUM_ENTITIES = 32
MINIMUM_EMBEDDING_MARKERS = 12
EXTERNAL_NEIGHBORS = 6

FIELD_METHODS = (
    "field_primary",
    "field_direct",
    "field_zero",
    "field_scalar",
    "field_nuclear",
    "field_hypergraph",
    "field_membership_permuted",
    "field_destroyed",
)
CLASSICAL_METHODS = (
    "pearson_direct",
    "pearson_structured",
    "deviance_direct",
    "deviance_structured",
)
ALL_METHODS = (*FIELD_METHODS, "independence", *CLASSICAL_METHODS)
MATCHED_METHODS = (
    "field_direct",
    "field_zero",
    "field_scalar",
    "field_nuclear",
    "field_hypergraph",
    "field_membership_permuted",
    "independence",
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


def _read_json(path: Path) -> dict[str, object]:
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


def _manifest_rows(
    directory: Path, outputs: tuple[str, ...], manifest_name: str
) -> pd.DataFrame:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("reducer bundle must be a real directory")
    manifest_path = directory / manifest_name
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("reducer bundle manifest is missing or is a symlink")
    manifest = pd.read_csv(
        manifest_path, sep="\t", dtype={"path": str, "sha256": str}
    )
    if list(manifest.columns) != ["path", "bytes", "sha256"]:
        raise ValueError("reducer manifest columns differ from the protocol")
    if manifest["path"].duplicated().any() or set(manifest["path"]) != set(outputs):
        raise ValueError("reducer manifest output set differs from the protocol")
    entries = list(directory.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ValueError("reducer bundle contains a symbolic link")
    actual = {path.relative_to(directory).as_posix() for path in entries if path.is_file()}
    if actual != set(outputs) | {manifest_name}:
        raise ValueError("reducer bundle contains an unexpected file set")
    for row in manifest.itertuples(index=False):
        if not isinstance(row.bytes, (int, np.integer)) or int(row.bytes) < 0:
            raise ValueError(f"reducer byte count is invalid: {row.path}")
        if re.fullmatch(r"[0-9a-f]{64}", str(row.sha256)) is None:
            raise ValueError(f"reducer SHA-256 is malformed: {row.path}")
    return manifest


def _manifest_identity(
    directory: Path, outputs: tuple[str, ...], manifest_name: str
) -> dict[str, object]:
    _manifest_rows(directory, outputs, manifest_name)
    path = directory / manifest_name
    return {
        "path": _repo_relative(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _artifact_bundle(
    directory: Path, outputs: tuple[str, ...], manifest_name: str, schema: str
) -> dict[str, object]:
    manifest = _manifest_rows(directory, outputs, manifest_name).set_index("path")
    records: list[dict[str, object]] = []
    for relative in outputs:
        artifact = directory / relative
        observed = {
            "path": relative,
            "bytes": artifact.stat().st_size,
            "sha256": _sha256(artifact),
        }
        if (
            observed["bytes"] != int(manifest.at[relative, "bytes"])
            or observed["sha256"] != manifest.at[relative, "sha256"]
        ):
            raise ValueError(f"reducer manifest does not match output: {relative}")
        records.append(observed)
    manifest_path = directory / manifest_name
    records.append(
        {
            "path": manifest_name,
            "bytes": manifest_path.stat().st_size,
            "sha256": _sha256(manifest_path),
        }
    )
    return {
        "schema": schema,
        "root": _repo_relative(directory),
        "artifacts": records,
    }


def _prediction_artifact_bundle(path: Path = PREDICTION_BUNDLE) -> dict[str, object]:
    return _artifact_bundle(
        path,
        PREDICTION_OUTPUTS,
        PREDICTION_MANIFEST,
        "kotliarov-pbmc-prediction-artifact-bundle/1.0",
    )


def _score_artifact_bundle(path: Path = SCORE_BUNDLE) -> dict[str, object]:
    return _artifact_bundle(
        path,
        SCORE_OUTPUTS,
        SCORE_MANIFEST,
        "kotliarov-pbmc-score-artifact-bundle/1.0",
    )


def _require_bundle(observed: dict[str, object], expected: object, label: str) -> None:
    if observed != expected:
        raise PermissionError(f"{label} artifacts differ from the frozen provenance")


def preflight(*, require_sealed: bool) -> dict[str, object]:
    designation = _read_json(DESIGNATION)
    if designation.get("schema") != "kotliarov-pbmc-coupling-candidate-designation/1.0":
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
    if SCGPT.is_file():
        embedding_sha256 = _sha256(SCGPT)
    else:
        if require_sealed:
            raise FileNotFoundError(
                "sealed execution requires the checksum-matched scGPT embedding"
            )
        embedding_manifest = _read_json(SCGPT_MANIFEST)
        if embedding_manifest.get("schema") != "scgpt-gene-embedding-derivation/1.0":
            raise ValueError("embedding derivation manifest schema differs")
        output = embedding_manifest.get("output", {})
        if output.get("path") != _repo_relative(SCGPT):
            raise ValueError("embedding derivation manifest names a different output")
        embedding_sha256 = output.get("sha256")
        if not isinstance(embedding_sha256, str) or re.fullmatch(
            r"[0-9a-f]{64}", embedding_sha256
        ) is None:
            raise ValueError("embedding derivation manifest has an invalid SHA-256")
    hashes = {
        "protocol_sha256": _sha256(PROTOCOL),
        "source_manifest_sha256": _sha256(SOURCE_MANIFEST),
        "metadata_support_sha256": _sha256(SUPPORT),
        "alias_sha256": _sha256(ALIASES),
        "lineage_markers_sha256": _sha256(LINEAGE_MARKERS),
        "embedding_sha256": embedding_sha256,
        "embedding_manifest_sha256": _sha256(SCGPT_MANIFEST),
        "authorization_template_sha256": _sha256(AUTH_TEMPLATE),
        "runner_sha256": _sha256(Path(__file__)),
        "reducer_sha256": _sha256(REDUCER),
        "test_sha256": _sha256(TEST),
    }
    for key, value in hashes.items():
        if designation.get(key) != value:
            raise ValueError(f"designation {key} is stale")
    implementation = {
        _repo_relative(path): _sha256(path) for path in IMPLEMENTATION_FILES
    }
    if designation.get("implementation_sha256") != implementation:
        raise ValueError("designation implementation_sha256 is stale")
    paths = {
        "protocol": _repo_relative(PROTOCOL),
        "source_manifest": _repo_relative(SOURCE_MANIFEST),
        "metadata_support_artifact": _repo_relative(SUPPORT),
        "alias_table": _repo_relative(ALIASES),
        "lineage_markers": _repo_relative(LINEAGE_MARKERS),
        "embedding": _repo_relative(SCGPT),
        "embedding_manifest": _repo_relative(SCGPT_MANIFEST),
        "authorization_template": _repo_relative(AUTH_TEMPLATE),
        "runner": _repo_relative(Path(__file__)),
        "reducer": _repo_relative(REDUCER),
        "test": _repo_relative(TEST),
        "prediction_bundle": _repo_relative(PREDICTION_BUNDLE),
        "score_bundle": _repo_relative(SCORE_BUNDLE),
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
    for key, value in paths.items():
        if designation.get(key) != value:
            raise ValueError(f"designation {key} differs from the runner")
    constants: dict[str, object] = {
        "development_donors": list(DEVELOPMENT),
        "held_donors": list(HELD),
        "excluded_donors": list(EXCLUDED),
        "held_high_responders": list(HELD_HIGH),
        "held_low_responders": list(HELD_LOW),
        "lineages": list(LINEAGES),
        "minimum_lineages": MINIMUM_LINEAGES,
        "minimum_lineage_cells": MINIMUM_LINEAGE_CELLS,
        "minimum_state_cells": MINIMUM_STATE_CELLS,
        "minimum_state_fraction": MINIMUM_STATE_FRACTION,
        "minimum_markers": MINIMUM_MARKERS,
        "minimum_entities": MINIMUM_ENTITIES,
        "minimum_embedding_markers": MINIMUM_EMBEDDING_MARKERS,
        "external_neighbors": EXTERNAL_NEIGHBORS,
        "permutations": PERMUTATIONS,
        "bootstrap_draws": BOOTSTRAPS,
        "primary_nuclear_grid": list(NUCLEAR_GRID[1:]),
        "primary_graph_grid": list(GRAPH_GRID[1:]),
        "nuclear_only_grid": list(NUCLEAR_GRID[1:]),
        "hypergraph_only_grid": list(GRAPH_GRID[1:]),
        "seed": SEED,
    }
    for key, value in constants.items():
        if designation.get(key) != value:
            raise ValueError(f"designation {key} differs from the runner")
    return {
        "designation_status": designation["status"],
        "outcome_access_authorized": designation["outcome_access_authorized"],
        "designation_sha256": _sha256(DESIGNATION),
        "implementation_sha256": implementation,
        "paths": paths,
        "constants": constants,
        **hashes,
    }


def _write_refusal(path: Path, *, stage: str, code: str, message: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusal artifact already exists: {path}")
    record = {
        "schema": "kotliarov-pbmc-refusal/1.0",
        "status": "REFUSED",
        "stage": stage,
        "code": code,
        "message": message,
        "candidate": "KotliarovPBMCData",
        "runner_sha256": _sha256(Path(__file__)),
        "protocol_sha256": _sha256(PROTOCOL),
        "designation_sha256": _sha256(DESIGNATION),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def prepare(args: argparse.Namespace) -> None:
    provenance = preflight(require_sealed=True)
    prediction_bundle = Path(args.prediction_bundle)
    score_bundle = Path(args.score_bundle)
    if _repo_relative(prediction_bundle) != provenance["paths"]["prediction_bundle"]:
        raise ValueError("prediction bundle path differs from the frozen designation")
    if _repo_relative(score_bundle) != provenance["paths"]["score_bundle"]:
        raise ValueError("score bundle path differs from the frozen designation")
    if PREPARE_RECORD.exists() or PREPARE_REFUSAL.exists():
        raise FileExistsError("prospective prepare record or refusal already exists")
    command = [
        sys.executable,
        str(REDUCER),
        "--rna-matrix",
        str(Path(args.rna_matrix)),
        "--adt-matrix",
        str(Path(args.adt_matrix)),
        "--metadata-root",
        str(Path(args.metadata_root)),
        "--source-manifest",
        str(SOURCE_MANIFEST),
        "--aliases",
        str(ALIASES),
        "--lineage-markers",
        str(LINEAGE_MARKERS),
        "--prediction-output",
        str(prediction_bundle),
        "--score-output",
        str(score_bundle),
    ]
    try:
        subprocess.run(command, check=True)
        prediction_artifacts = _prediction_artifact_bundle(prediction_bundle)
        sealed_score_manifest = _manifest_identity(
            score_bundle, SCORE_OUTPUTS, SCORE_MANIFEST
        )
        record = {
            "schema": "kotliarov-pbmc-prepare-record/1.0",
            "status": "PREPARED_HELD_PAIRING_NOT_USED_BY_PREDICTION_PATH",
            "prediction_artifacts": prediction_artifacts,
            "sealed_score_manifest": sealed_score_manifest,
            "provenance": provenance,
        }
        PREPARE_RECORD.parent.mkdir(parents=True, exist_ok=True)
        PREPARE_RECORD.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    except (subprocess.CalledProcessError, OSError, TypeError, ValueError) as error:
        _write_refusal(
            PREPARE_REFUSAL,
            stage="prepare",
            code="SOURCE_OR_REDUCER_FAILURE",
            message=str(error),
        )
        raise


def _prepared_prediction_bundle(
    expected_provenance: dict[str, object], path: Path = PREDICTION_BUNDLE
) -> tuple[dict[str, object], dict[str, object]]:
    record = _read_json(PREPARE_RECORD)
    if record.get("schema") != "kotliarov-pbmc-prepare-record/1.0":
        raise PermissionError("prepare record schema differs")
    if record.get("status") != "PREPARED_HELD_PAIRING_NOT_USED_BY_PREDICTION_PATH":
        raise PermissionError("prepare record is not frozen")
    if record.get("provenance") != expected_provenance:
        raise PermissionError("prepare record frozen provenance differs")
    observed = _prediction_artifact_bundle(path)
    _require_bundle(observed, record.get("prediction_artifacts"), "prediction")
    return observed, record["sealed_score_manifest"]


def _load_npy_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        values = np.load(stream, allow_pickle=False)
    return np.asarray(values)


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def _load_prediction_data(path: Path = PREDICTION_BUNDLE) -> dict[str, object]:
    cells = _read_tsv(path / "cells.tsv.gz")
    markers = _read_tsv(path / "markers.tsv")
    entities = _read_tsv(path / "entities.tsv")
    development_index = _read_tsv(path / "development_cell_index.tsv.gz")
    held_marginals = _read_tsv(path / "held_adt_marginals.tsv")
    rna_values = _load_npy_gz(path / "rna_values.npy.gz").astype(float)
    rna_states = _load_npy_gz(path / "rna_states.npy.gz").astype(int)
    development_adt_values = _load_npy_gz(
        path / "development_adt_values.npy.gz"
    ).astype(float)
    development_adt_states = _load_npy_gz(
        path / "development_adt_states.npy.gz"
    ).astype(int)
    required_cells = {
        "prediction_cell_index",
        "cell_id",
        "donor",
        "lineage",
        "split",
    }
    required_markers = {"marker_id", "gene_symbol", "adt_target", "module"}
    required_entities = {
        "entity_id",
        "marker_index",
        "marker_id",
        "gene_symbol",
        "adt_target",
        "module",
        "lineage",
    }
    required_index = {"development_position", "prediction_cell_index", "cell_id"}
    required_margins = {"donor", "lineage", "marker_id", "state", "count"}
    for frame, required, label in (
        (cells, required_cells, "cells"),
        (markers, required_markers, "markers"),
        (entities, required_entities, "entities"),
        (development_index, required_index, "development cell index"),
        (held_marginals, required_margins, "held ADT marginals"),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} lacks frozen columns: {sorted(missing)}")
    if "eligible" in entities:
        eligible = entities["eligible"].astype(bool).to_numpy()
        exclusions = entities.loc[~eligible].to_dict(orient="records")
        entities = entities.loc[eligible].reset_index(drop=True)
    else:
        exclusions = []
    cells = cells.copy()
    markers = markers.copy()
    entities = entities.copy()
    cells["donor"] = cells["donor"].astype(str)
    markers["marker_id"] = markers["marker_id"].astype(str)
    entities["marker_id"] = entities["marker_id"].astype(str)
    held_marginals["donor"] = held_marginals["donor"].astype(str)
    held_marginals["marker_id"] = held_marginals["marker_id"].astype(str)
    if cells["prediction_cell_index"].tolist() != list(range(len(cells))):
        raise ValueError("prediction cells are not on a complete zero-based axis")
    development_index = development_index.sort_values("development_position")
    if development_index["development_position"].tolist() != list(range(len(development_index))):
        raise ValueError("development positions are not a complete zero-based axis")
    prediction_index = development_index["prediction_cell_index"].to_numpy(dtype=int)
    if (
        np.any(prediction_index < 0)
        or np.any(prediction_index >= len(cells))
        or len(set(prediction_index.tolist())) != len(prediction_index)
    ):
        raise ValueError("development prediction-cell indices are invalid")
    if development_index["cell_id"].astype(str).tolist() != cells.iloc[
        prediction_index
    ]["cell_id"].astype(str).tolist():
        raise ValueError("development cell axis does not match prediction cells")
    if set(cells.iloc[prediction_index]["split"]) != {"development"}:
        raise ValueError("development cell index contains held cells")
    if set(cells["donor"]) & set(EXCLUDED):
        raise ValueError("globally excluded donor 209 remains in the prediction bundle")
    if set(cells.loc[cells["split"] == "development", "donor"]) != set(DEVELOPMENT):
        raise ValueError("development donor axis differs from the protocol")
    if set(cells.loc[cells["split"] == "held", "donor"]) != set(HELD):
        raise ValueError("held donor axis differs from the protocol")
    marker_count = len(markers)
    if rna_values.shape != rna_states.shape or rna_values.shape != (
        marker_count,
        len(cells),
    ):
        raise ValueError("RNA arrays do not match marker and cell axes")
    if development_adt_values.shape != development_adt_states.shape or development_adt_values.shape != (
        marker_count,
        len(development_index),
    ):
        raise ValueError("development ADT arrays do not match frozen axes")
    if (
        not np.isfinite(rna_values).all()
        or not np.isfinite(development_adt_values).all()
        or np.any(rna_states < 0)
        or np.any(rna_states > 2)
        or np.any(development_adt_states < 0)
        or np.any(development_adt_states > 2)
    ):
        raise ValueError("prediction arrays contain invalid values")
    marker_indices = entities["marker_index"].to_numpy(dtype=int)
    if np.any(marker_indices < 0) or np.any(marker_indices >= marker_count):
        raise ValueError("entity marker index is outside the marker axis")
    for row in entities.itertuples(index=False):
        marker = markers.iloc[int(row.marker_index)]
        for column in ("marker_id", "gene_symbol", "adt_target", "module"):
            if str(getattr(row, column)) != str(marker[column]):
                raise ValueError(f"entity {column} does not match marker axis")
    if entities["entity_id"].duplicated().any():
        raise ValueError("entity identifiers are not unique")
    if len(set(entities["lineage"])) < MINIMUM_LINEAGES:
        raise ValueError("fewer than four lineages survived RNA-only processing")
    alias_frame = pd.read_csv(ALIASES, sep="\t")
    retained_targets = set(markers["adt_target"].astype(str))
    for alias in alias_frame.itertuples(index=False):
        if str(alias.adt_target) not in retained_targets:
            exclusions.append(
                {
                    "adt_target": str(alias.adt_target),
                    "gene_symbol": str(alias.gene_symbol),
                    "module": str(alias.module),
                    "lineage": None,
                    "reason": "exact RNA or ADT feature unavailable or duplicated",
                }
            )
    eligible_keys = set(
        zip(entities["marker_id"].astype(str), entities["lineage"].astype(str))
    )
    for marker in markers.itertuples(index=False):
        for lineage in sorted(set(cells["lineage"].astype(str))):
            if (str(marker.marker_id), lineage) not in eligible_keys:
                exclusions.append(
                    {
                        "adt_target": str(marker.adt_target),
                        "gene_symbol": str(marker.gene_symbol),
                        "module": str(marker.module),
                        "lineage": lineage,
                        "reason": "distinct-cut or separate-marginal support failure",
                    }
                )
    return {
        "cells": cells,
        "markers": markers,
        "entities": entities,
        "entity_exclusions": exclusions,
        "development_index": development_index,
        "prediction_index": prediction_index,
        "held_marginals": held_marginals,
        "rna_values": rna_values,
        "rna_states": rna_states,
        "development_adt_values": development_adt_values,
        "development_adt_states": development_adt_states,
    }


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


def _margin_stats_from_counts(
    rows: np.ndarray, columns: np.ndarray, seed: int
) -> dict[str, np.ndarray | float]:
    first, second = _canonical_states(rows, columns)
    total = float(np.sum(rows))
    field = conditional_association_coordinates(
        first,
        second,
        first_levels=3,
        second_levels=3,
        pseudocount=PSEUDOCOUNT,
        permutations=PERMUTATIONS,
        seed=seed,
    )
    result: dict[str, np.ndarray | float] = {
        "rows": np.asarray(rows, dtype=float),
        "columns": np.asarray(columns, dtype=float),
        "total": total,
        "field_null": field.null_mean_coordinates.ravel(),
        "field_destroyed": field.destroyed_coordinates.ravel(),
        "field_variance": field.null_variance_coordinates.ravel(),
    }
    for family in ("pearson", "deviance"):
        estimate = conditional_poisson_residuals(
            first,
            second,
            first_levels=3,
            second_levels=3,
            residual=family,
            permutations=PERMUTATIONS,
            seed=seed,
        )
        result[f"{family}_null"] = estimate.null_mean_coordinates.ravel()
        result[f"{family}_destroyed"] = (
            estimate.destroyed_coordinates / np.sqrt(total)
        ).ravel()
        result[f"{family}_variance"] = (
            estimate.null_variance_coordinates / total
        ).ravel()
    return result


def _table_stats(
    first: np.ndarray, second: np.ndarray, seed: int
) -> dict[str, np.ndarray | float]:
    table = _table(np.asarray(first, dtype=int), np.asarray(second, dtype=int))
    margins = _margin_stats_from_counts(table.sum(axis=1), table.sum(axis=0), seed)
    total = float(table.sum())
    raw_field = association_coordinates(
        association_field(table, pseudocount=PSEUDOCOUNT)
    ).ravel()
    result: dict[str, np.ndarray | float] = {
        **margins,
        "table": table,
        "field": raw_field - np.asarray(margins["field_null"]),
        "field_raw": raw_field,
    }
    for family in ("pearson", "deviance"):
        raw = poisson_independence_residuals(table, residual=family).ravel()
        result[family] = (
            raw - np.asarray(margins[f"{family}_null"])
        ) / np.sqrt(total)
        result[f"{family}_raw"] = raw
    return result


def _support_ok(counts: np.ndarray) -> bool:
    values = np.asarray(counts, dtype=int)
    return bool(
        values.shape == (3,)
        and np.all(values >= MINIMUM_STATE_CELLS)
        and np.all(values / values.sum() >= MINIMUM_STATE_FRACTION)
    )


def _validate_entities_and_margins(data: dict[str, object]) -> None:
    cells = data["cells"]
    entities = data["entities"]
    rna_states = data["rna_states"]
    dev_states = data["development_adt_states"]
    dev_cells = cells.iloc[data["prediction_index"]].reset_index(drop=True)
    held_marginals = data["held_marginals"]
    genes = entities["gene_symbol"].astype(str).tolist()
    if len(set(genes)) < MINIMUM_MARKERS:
        raise ValueError("fewer than 16 unique cognate markers pass support")
    if len(entities) < MINIMUM_ENTITIES:
        raise ValueError("fewer than 32 marker-lineage entities pass support")
    for donor in DEVELOPMENT + HELD:
        for lineage in sorted(set(entities["lineage"])):
            block = cells[
                (cells["donor"] == donor)
                & (cells["lineage"] == lineage)
                & (cells["split"] == ("development" if donor in DEVELOPMENT else "held"))
            ]
            if len(block) < MINIMUM_LINEAGE_CELLS:
                raise ValueError("frozen donor-lineage support is below 50 cells")
    for entity in entities.itertuples(index=False):
        marker = int(entity.marker_index)
        for donor in DEVELOPMENT:
            mask = (
                (dev_cells["donor"] == donor)
                & (dev_cells["lineage"] == entity.lineage)
            ).to_numpy()
            if not _support_ok(np.bincount(rna_states[marker, data["prediction_index"]][mask], minlength=3)):
                raise ValueError(f"development RNA marginal support failed: {entity.entity_id} {donor}")
            if not _support_ok(np.bincount(dev_states[marker, mask], minlength=3)):
                raise ValueError(f"development ADT marginal support failed: {entity.entity_id} {donor}")
        for donor in HELD:
            mask = (
                (cells["donor"] == donor)
                & (cells["lineage"] == entity.lineage)
                & (cells["split"] == "held")
            ).to_numpy()
            if not _support_ok(np.bincount(rna_states[marker, mask], minlength=3)):
                raise ValueError(f"held RNA marginal support failed: {entity.entity_id} {donor}")
            selected = held_marginals[
                (held_marginals["donor"] == donor)
                & (held_marginals["lineage"] == entity.lineage)
                & (held_marginals["marker_id"] == entity.marker_id)
            ].sort_values("state")
            if selected["state"].tolist() != [0, 1, 2] or not _support_ok(
                selected["count"].to_numpy(dtype=int)
            ):
                raise ValueError(f"held ADT marginal support failed: {entity.entity_id} {donor}")
            if int(selected["count"].sum()) != int(np.count_nonzero(mask)):
                raise ValueError("held RNA and ADT marginal totals differ")


def _development_stats_from_states(data: dict[str, object]) -> dict[str, np.ndarray]:
    cells = data["cells"].iloc[data["prediction_index"]].reset_index(drop=True)
    entities = data["entities"]
    rna = data["rna_states"][:, data["prediction_index"]]
    adt = data["development_adt_states"]
    return _stats_for_donors(cells, entities, rna, adt, DEVELOPMENT, "full-development")


def _stats_for_donors(
    cells: pd.DataFrame,
    entities: pd.DataFrame,
    rna_states: np.ndarray,
    adt_states: np.ndarray,
    donors: tuple[str, ...],
    seed_label: str,
) -> dict[str, np.ndarray]:
    count = len(entities)
    dimensions = {
        "field": 4,
        "field_destroyed": 4,
        "field_variance": 4,
        "field_null": 4,
        "pearson": 9,
        "pearson_destroyed": 9,
        "pearson_variance": 9,
        "pearson_null": 9,
        "deviance": 9,
        "deviance_destroyed": 9,
        "deviance_variance": 9,
        "deviance_null": 9,
    }
    arrays = {name: np.empty((len(donors), count, width)) for name, width in dimensions.items()}
    arrays.update(
        {
            "rows": np.empty((len(donors), count, 3)),
            "columns": np.empty((len(donors), count, 3)),
            "total": np.empty((len(donors), count)),
            "table": np.empty((len(donors), count, 3, 3)),
        }
    )
    donor_axis = cells["donor"].astype(str).to_numpy()
    lineage_axis = cells["lineage"].astype(str).to_numpy()
    for donor_index, donor in enumerate(donors):
        for entity_index, entity in enumerate(entities.itertuples(index=False)):
            mask = (donor_axis == donor) & (lineage_axis == str(entity.lineage))
            if np.count_nonzero(mask) < MINIMUM_LINEAGE_CELLS:
                raise ValueError("frozen donor-lineage support unexpectedly failed")
            marker = int(entity.marker_index)
            stats = _table_stats(
                rna_states[marker, mask],
                adt_states[marker, mask],
                _seed(seed_label, donor, str(entity.entity_id)),
            )
            for name in dimensions:
                arrays[name][donor_index, entity_index] = stats[name]
            for name in ("rows", "columns", "total", "table"):
                arrays[name][donor_index, entity_index] = stats[name]
    return arrays


def _weighted_quantile(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 1 or weights.shape != values.shape or len(values) < 1:
        raise ValueError("weighted quantile inputs must be equal nonempty vectors")
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("weighted quantile inputs must be finite with positive weights")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    positions = (np.cumsum(sorted_weights) - 0.5 * sorted_weights) / sorted_weights.sum()
    return np.interp([1 / 3, 2 / 3], positions, sorted_values)


def _donor_equal_cuts(values: np.ndarray, donors: np.ndarray) -> np.ndarray:
    donor_values = np.asarray(donors, dtype=str)
    unique = sorted(set(donor_values.tolist()))
    weights = np.empty(len(values), dtype=float)
    for donor in unique:
        mask = donor_values == donor
        weights[mask] = 1.0 / (len(unique) * np.count_nonzero(mask))
    cuts = _weighted_quantile(values, weights)
    if cuts[0] >= cuts[1]:
        raise ValueError("fold-specific donor-equal tertile cuts are not distinct")
    return cuts


def _lopo_folds(data: dict[str, object]) -> list[dict[str, np.ndarray]]:
    cells = data["cells"].iloc[data["prediction_index"]].reset_index(drop=True)
    entities = data["entities"]
    rna_values = data["rna_values"][:, data["prediction_index"]]
    adt_values = data["development_adt_values"]
    donor_axis = cells["donor"].astype(str).to_numpy()
    lineage_axis = cells["lineage"].astype(str).to_numpy()
    folds: list[dict[str, np.ndarray]] = []
    for held_out in DEVELOPMENT:
        training = tuple(donor for donor in DEVELOPMENT if donor != held_out)
        fold_rna = np.full_like(rna_values, -1, dtype=int)
        fold_adt = np.full_like(adt_values, -1, dtype=int)
        for entity in entities.itertuples(index=False):
            marker = int(entity.marker_index)
            lineage = lineage_axis == str(entity.lineage)
            calibration = lineage & np.isin(donor_axis, training)
            rna_cut = _donor_equal_cuts(rna_values[marker, calibration], donor_axis[calibration])
            adt_cut = _donor_equal_cuts(adt_values[marker, calibration], donor_axis[calibration])
            fold_rna[marker, lineage] = np.searchsorted(
                rna_cut, rna_values[marker, lineage], side="left"
            )
            fold_adt[marker, lineage] = np.searchsorted(
                adt_cut, adt_values[marker, lineage], side="left"
            )
        if np.any(fold_rna[entities["marker_index"].unique()] < -1) or np.any(
            fold_adt[entities["marker_index"].unique()] < -1
        ):
            raise ValueError("invalid fold-specific state labels")
        stats = _stats_for_donors(
            cells,
            entities,
            fold_rna,
            fold_adt,
            DEVELOPMENT,
            f"lopo-{held_out}",
        )
        stats["held_out_index"] = np.asarray(DEVELOPMENT.index(held_out))
        folds.append(stats)
    return folds


def _embedding_hypergraph(
    entities: pd.DataFrame, *, permute_external_membership: bool = False
) -> tuple[np.ndarray, dict[str, object]]:
    genes = entities["gene_symbol"].astype(str).tolist()
    unique_genes = list(dict.fromkeys(genes))
    with np.load(SCGPT, allow_pickle=False) as archive:
        names = archive["gene_names"].astype(str)
        embedding = np.asarray(archive["embedding"], dtype=float)
    lookup = {name.upper(): index for index, name in enumerate(names)}
    covered = [gene for gene in unique_genes if gene.upper() in lookup]
    if len(covered) < MINIMUM_EMBEDDING_MARKERS:
        raise ValueError("fewer than 12 frozen markers are covered by the embedding")
    represented = covered.copy()
    permutation = np.arange(len(covered))
    if permute_external_membership:
        permutation = np.random.default_rng(
            _seed("kotliarov-external-membership-permutation")
        ).permutation(len(covered))
        represented = [covered[index] for index in permutation]
    values = np.asarray([embedding[lookup[gene.upper()]] for gene in represented])
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    similarity = values @ values.T
    edges: list[np.ndarray] = []
    labels: list[str] = []

    def add_edge(label: str, membership: np.ndarray) -> None:
        if np.any(membership):
            edges.append(membership.astype(float))
            labels.append(label)

    genes_array = np.asarray(genes)
    for gene in unique_genes:
        add_edge(f"marker:{gene}", genes_array == gene)
    for lineage in sorted(set(entities["lineage"].astype(str))):
        add_edge(f"lineage:{lineage}", entities["lineage"].astype(str).to_numpy() == lineage)
    for module in sorted(set(entities["module"].astype(str))):
        add_edge(f"module:{module}", entities["module"].astype(str).to_numpy() == module)
    for index, gene in enumerate(covered):
        order = np.lexsort((np.asarray(covered), -similarity[index]))
        neighbors = [position for position in order if position != index][:EXTERNAL_NEIGHBORS]
        member_genes = {gene, *[covered[position] for position in neighbors]}
        add_edge(f"external:{gene}", np.isin(genes_array, sorted(member_genes)))
    incidence = np.column_stack(edges)
    return normalized_hypergraph_laplacian(incidence), {
        "embedding_path": _repo_relative(SCGPT),
        "embedding_sha256": _sha256(SCGPT),
        "covered_markers": covered,
        "external_neighbors": EXTERNAL_NEIGHBORS,
        "external_membership_permuted": permute_external_membership,
        "external_membership_permutation": permutation.tolist(),
        "hyperedge_labels": labels,
        "incidence_sha256": hashlib.sha256(
            np.ascontiguousarray(incidence).view(np.uint8)
        ).hexdigest(),
    }


def _mean_and_variance(
    values: np.ndarray, variance: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    count = values.shape[0]
    return values.mean(axis=0), variance.sum(axis=0) / count**2


def _structured_fit(
    values: np.ndarray,
    variance: np.ndarray,
    laplacian: np.ndarray,
    nuclear_multiplier: float,
    graph_penalty: float,
) -> tuple[np.ndarray, dict[str, object]]:
    values = np.asarray(values, dtype=float)
    variance = np.maximum(np.asarray(variance, dtype=float), np.finfo(float).eps)
    leading = float(np.linalg.svd(values, compute_uv=False)[0])
    fit = fit_structured_coupling_fields(
        values,
        observation_weight=inverse_permutation_variance_weights(variance),
        graph_laplacian=laplacian if graph_penalty > 0.0 else None,
        nuclear_penalty=nuclear_multiplier * leading,
        graph_penalty=graph_penalty,
        tolerance=1e-8,
    )
    if not fit.converged:
        raise RuntimeError("structured interaction atlas did not converge")
    return fit.coefficient, {
        "converged": fit.converged,
        "nuclear_multiplier": nuclear_multiplier,
        "nuclear_penalty": fit.nuclear_penalty,
        "graph_penalty": graph_penalty,
        "iterations": fit.iterations,
        "relative_step": fit.relative_step,
        "objective": fit.objective,
        "effective_rank": fit.effective_rank,
        "singular_values": fit.singular_values.tolist(),
    }


def _enforce_margins(
    table: np.ndarray, rows: np.ndarray, columns: np.ndarray
) -> np.ndarray:
    fitted = ipf_to_margins(
        np.maximum(np.asarray(table, dtype=float), np.finfo(float).tiny),
        rows,
        columns,
        tolerance=1e-12,
    )
    error = max(
        float(np.max(np.abs(fitted.sum(axis=1) - rows))),
        float(np.max(np.abs(fitted.sum(axis=0) - columns))),
    )
    if error > IPF_TOLERANCE:
        raise ValueError("table reconstruction margin error exceeds 1e-8")
    return fitted


def _field_table(
    centered: np.ndarray,
    null_mean: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    table = field_coordinates_to_table(
        np.asarray(centered).reshape(2, 2) + np.asarray(null_mean).reshape(2, 2),
        rows,
        columns,
    )
    return _enforce_margins(table, rows, columns)


def _residual_table(
    centered: np.ndarray,
    null_mean: np.ndarray,
    total: float,
    rows: np.ndarray,
    columns: np.ndarray,
    family: str,
) -> np.ndarray:
    raw = np.asarray(centered).reshape(3, 3) * np.sqrt(total) + np.asarray(
        null_mean
    ).reshape(3, 3)
    table = residual_coordinates_to_table(raw, rows, columns, residual=family)
    return _enforce_margins(table, rows, columns)


def _hierarchical_loss(
    entity_loss: np.ndarray, entities: pd.DataFrame
) -> float:
    values = np.asarray(entity_loss, dtype=float)
    marker_means = []
    for marker in list(dict.fromkeys(entities["marker_id"].astype(str))):
        marker_means.append(float(values[entities["marker_id"].astype(str).to_numpy() == marker].mean()))
    return float(np.mean(marker_means))


def _candidate_fold_loss(
    prediction: np.ndarray,
    fold: dict[str, np.ndarray],
    family: str,
    entities: pd.DataFrame,
) -> float:
    held = int(fold["held_out_index"])
    losses = np.empty(len(entities))
    for entity in range(len(entities)):
        rows = fold["rows"][held, entity]
        columns = fold["columns"][held, entity]
        if family == "field":
            table = _field_table(
                prediction[entity], fold["field_null"][held, entity], rows, columns
            )
        else:
            table = _residual_table(
                prediction[entity],
                fold[f"{family}_null"][held, entity],
                fold["total"][held, entity],
                rows,
                columns,
                family,
            )
        losses[entity] = multinomial_deviance_per_observation(
            fold["table"][held, entity], table
        )
    return _hierarchical_loss(losses, entities)


def _cv_grid(
    folds: list[dict[str, np.ndarray]],
    entities: pd.DataFrame,
    family: str,
    laplacian: np.ndarray,
    candidates: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    for nuclear, graph in sorted(candidates):
        losses = []
        for fold in folds:
            held = int(fold["held_out_index"])
            training = np.asarray([index for index in range(len(DEVELOPMENT)) if index != held])
            mean, variance = _mean_and_variance(
                fold[family][training], fold[f"{family}_variance"][training]
            )
            prediction, _ = _structured_fit(
                mean, variance, laplacian, nuclear, graph
            )
            losses.append(_candidate_fold_loss(prediction, fold, family, entities))
        records.append(
            {
                "nuclear_multiplier": nuclear,
                "graph_penalty": graph,
                "mean_lopo_deviance": float(np.mean(losses)),
                "held_donor_deviance": dict(zip(DEVELOPMENT, losses)),
            }
        )
    best = min(
        records,
        key=lambda row: (
            row["mean_lopo_deviance"],
            row["nuclear_multiplier"],
            row["graph_penalty"],
        ),
    )
    return (
        float(best["nuclear_multiplier"]),
        float(best["graph_penalty"]),
    ), records


def _fit_predictions(
    data: dict[str, object],
    development: dict[str, np.ndarray],
    folds: list[dict[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    entities = data["entities"]
    laplacian, graph = _embedding_hypergraph(entities)
    permuted_laplacian, permuted_graph = _embedding_hypergraph(
        entities, permute_external_membership=True
    )
    structured_grid = tuple(
        itertools.product(NUCLEAR_GRID[1:], GRAPH_GRID[1:])
    )
    primary_choice, primary_cv = _cv_grid(
        folds, entities, "field", laplacian, structured_grid
    )
    nuclear_choice, nuclear_cv = _cv_grid(
        folds,
        entities,
        "field",
        laplacian,
        tuple((value, 0.0) for value in NUCLEAR_GRID[1:]),
    )
    hypergraph_choice, hypergraph_cv = _cv_grid(
        folds,
        entities,
        "field",
        laplacian,
        tuple((0.0, value) for value in GRAPH_GRID[1:]),
    )
    permuted_choice, permuted_cv = _cv_grid(
        folds, entities, "field", permuted_laplacian, structured_grid
    )
    pearson_choice, pearson_cv = _cv_grid(
        folds, entities, "pearson", laplacian, structured_grid
    )
    deviance_choice, deviance_cv = _cv_grid(
        folds, entities, "deviance", laplacian, structured_grid
    )
    predictions: dict[str, np.ndarray] = {}
    diagnostics: dict[str, object] = {}
    field, field_variance = _mean_and_variance(
        development["field"], development["field_variance"]
    )
    predictions["field_direct"] = field
    predictions["field_zero"] = np.zeros_like(field)
    signal = float(np.sum(field**2))
    scalar = max(
        0.0,
        1.0 - float(np.sum(field_variance)) / max(signal, np.finfo(float).eps),
    )
    predictions["field_scalar"] = scalar * field
    for name, choice, graph_matrix in (
        ("field_primary", primary_choice, laplacian),
        ("field_nuclear", nuclear_choice, laplacian),
        ("field_hypergraph", hypergraph_choice, laplacian),
        ("field_membership_permuted", permuted_choice, permuted_laplacian),
    ):
        predictions[name], diagnostics[name] = _structured_fit(
            field, field_variance, graph_matrix, *choice
        )
    destroyed, _ = _mean_and_variance(
        development["field_destroyed"], development["field_variance"]
    )
    predictions["field_destroyed"], diagnostics["field_destroyed"] = _structured_fit(
        destroyed, field_variance, laplacian, *primary_choice
    )
    for family, choice in (
        ("pearson", pearson_choice),
        ("deviance", deviance_choice),
    ):
        mean, variance = _mean_and_variance(
            development[family], development[f"{family}_variance"]
        )
        predictions[f"{family}_direct"] = mean
        predictions[f"{family}_structured"], diagnostics[
            f"{family}_structured"
        ] = _structured_fit(mean, variance, laplacian, *choice)
    tuning = {
        "selection_unit": "leave-one-development-donor-out table deviance",
        "hierarchy": "lineages within marker, markers within donor, donors equally",
        "tie_rule": "smaller nuclear multiplier, then smaller graph penalty",
        "selected": {
            "field_primary": list(primary_choice),
            "field_nuclear": list(nuclear_choice),
            "field_hypergraph": list(hypergraph_choice),
            "field_membership_permuted": list(permuted_choice),
            "pearson_structured": list(pearson_choice),
            "deviance_structured": list(deviance_choice),
        },
        "cv": {
            "field_primary": primary_cv,
            "field_nuclear": nuclear_cv,
            "field_hypergraph": hypergraph_cv,
            "field_membership_permuted": permuted_cv,
            "pearson_structured": pearson_cv,
            "deviance_structured": deviance_cv,
        },
        "variance_scalar": scalar,
        "fit_diagnostics": diagnostics,
        "hypergraph": graph,
        "membership_permuted_hypergraph": permuted_graph,
    }
    return predictions, tuning


def _held_margin_stats(data: dict[str, object]) -> dict[str, np.ndarray]:
    cells = data["cells"]
    entities = data["entities"]
    rna_states = data["rna_states"]
    margins = data["held_marginals"]
    arrays = {
        "rows": np.empty((len(HELD), len(entities), 3)),
        "columns": np.empty((len(HELD), len(entities), 3)),
        "total": np.empty((len(HELD), len(entities))),
        "field_null": np.empty((len(HELD), len(entities), 4)),
        "pearson_null": np.empty((len(HELD), len(entities), 9)),
        "deviance_null": np.empty((len(HELD), len(entities), 9)),
    }
    for donor_index, donor in enumerate(HELD):
        for entity_index, entity in enumerate(entities.itertuples(index=False)):
            mask = (
                (cells["donor"] == donor)
                & (cells["lineage"] == entity.lineage)
                & (cells["split"] == "held")
            ).to_numpy()
            rows = np.bincount(rna_states[int(entity.marker_index), mask], minlength=3)
            selected = margins[
                (margins["donor"] == donor)
                & (margins["lineage"] == entity.lineage)
                & (margins["marker_id"] == entity.marker_id)
            ].sort_values("state")
            columns = selected["count"].to_numpy(dtype=int)
            stats = _margin_stats_from_counts(
                rows, columns, _seed("held-margin", donor, str(entity.entity_id))
            )
            for name in arrays:
                arrays[name][donor_index, entity_index] = stats[name]
    return arrays


def _predict_tables(
    predictions: dict[str, np.ndarray], margins: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    entities = next(iter(predictions.values())).shape[0]
    tables = {
        name: np.empty((len(HELD), entities, 3, 3)) for name in ALL_METHODS
    }
    for donor in range(len(HELD)):
        for entity in range(entities):
            rows = margins["rows"][donor, entity]
            columns = margins["columns"][donor, entity]
            for name in FIELD_METHODS:
                tables[name][donor, entity] = _field_table(
                    predictions[name][entity],
                    margins["field_null"][donor, entity],
                    rows,
                    columns,
                )
            tables["independence"][donor, entity] = _enforce_margins(
                np.outer(rows, columns) / rows.sum(), rows, columns
            )
            for family in ("pearson", "deviance"):
                for suffix in ("direct", "structured"):
                    name = f"{family}_{suffix}"
                    tables[name][donor, entity] = _residual_table(
                        predictions[name][entity],
                        margins[f"{family}_null"][donor, entity],
                        margins["total"][donor, entity],
                        rows,
                        columns,
                        family,
                    )
    return tables


def predict(args: argparse.Namespace) -> None:
    provenance = preflight(require_sealed=True)
    output = Path(args.output)
    if _repo_relative(output) != provenance["paths"]["prediction_path"]:
        raise ValueError("prediction output path differs from the frozen designation")
    if output.exists() or PREDICTION_REFUSAL.exists():
        raise FileExistsError("prospective prediction or refusal already exists")
    try:
        bundle, score_manifest = _prepared_prediction_bundle(provenance)
        data = _load_prediction_data(PREDICTION_BUNDLE)
        _validate_entities_and_margins(data)
        development = _development_stats_from_states(data)
        folds = _lopo_folds(data)
        predictions, tuning = _fit_predictions(data, development, folds)
        margins = _held_margin_stats(data)
        tables = _predict_tables(predictions, margins)
        _require_bundle(_prediction_artifact_bundle(), bundle, "prediction")
        record = {
            "schema": "kotliarov-pbmc-predictions/1.0",
            "status": "PREDICTIONS_FROZEN_HELD_PAIRING_NOT_USED",
            "scope": "held-batch same-cell RNA-protein joint distributions",
            "development_donors": list(DEVELOPMENT),
            "held_donors": list(HELD),
            "excluded_donors": list(EXCLUDED),
            "entity_ids": data["entities"]["entity_id"].astype(str).tolist(),
            "marker_ids": data["entities"]["marker_id"].astype(str).tolist(),
            "genes": data["entities"]["gene_symbol"].astype(str).tolist(),
            "entity_lineages": data["entities"]["lineage"].astype(str).tolist(),
            "entity_modules": data["entities"]["module"].astype(str).tolist(),
            "entity_marker_indices": data["entities"]["marker_index"].astype(int).tolist(),
            "marginal_support_exclusions": data["entity_exclusions"],
            "held_margins": {
                "rows": margins["rows"].tolist(),
                "columns": margins["columns"].tolist(),
            },
            "predictions": {name: value.tolist() for name, value in predictions.items()},
            "predicted_tables": {name: value.tolist() for name, value in tables.items()},
            "predicted_tables_stored": True,
            "held_pairing_used": False,
            "tuning": tuning,
            "sealed_score_manifest": score_manifest,
            "provenance": {**provenance, "prediction_artifacts": bundle},
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    except (KeyError, OSError, PermissionError, ValueError, RuntimeError) as error:
        code = (
            "DEVELOPMENT_SUPPORT_FAILURE"
            if "support" in str(error).lower() or "fewer than" in str(error).lower()
            else "PRETRUTH_INTEGRITY_OR_ANALYSIS_FAILURE"
        )
        if "converge" in str(error).lower():
            code = "OPTIMIZATION_FAILURE"
        _write_refusal(PREDICTION_REFUSAL, stage="predict", code=code, message=str(error))
        raise


def _locked_predictions(
    path: Path, expected_provenance: dict[str, object]
) -> tuple[dict[str, object], dict[str, np.ndarray], dict[str, np.ndarray]]:
    record = _read_json(path)
    if record.get("schema") != "kotliarov-pbmc-predictions/1.0":
        raise ValueError("prediction JSON schema differs")
    if record.get("status") != "PREDICTIONS_FROZEN_HELD_PAIRING_NOT_USED":
        raise ValueError("prediction JSON is not frozen")
    if record.get("held_pairing_used") is not False or record.get("predicted_tables_stored") is not True:
        raise ValueError("prediction JSON pairing boundary differs")
    entity_count = len(record.get("entity_ids", []))
    for key in (
        "marker_ids",
        "genes",
        "entity_lineages",
        "entity_modules",
        "entity_marker_indices",
    ):
        if len(record.get(key, [])) != entity_count:
            raise ValueError(f"prediction entity metadata axis differs: {key}")
    predictions = {
        name: np.asarray(value, dtype=float)
        for name, value in record.get("predictions", {}).items()
    }
    if set(predictions) != set(FIELD_METHODS) | set(CLASSICAL_METHODS):
        raise ValueError("locked prediction method set differs from the protocol")
    for name, values in predictions.items():
        width = 4 if name.startswith("field_") else 9
        if values.shape != (entity_count, width) or not np.isfinite(values).all():
            raise ValueError(f"locked prediction shape differs for {name}")
    tables = {
        name: np.asarray(value, dtype=float)
        for name, value in record.get("predicted_tables", {}).items()
    }
    if set(tables) != set(ALL_METHODS):
        raise ValueError("locked predicted-table method set differs")
    for name, values in tables.items():
        if values.shape != (len(HELD), entity_count, 3, 3) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"locked predicted-table shape differs for {name}")
    margins = record.get("held_margins", {})
    rows = np.asarray(margins.get("rows"), dtype=float)
    columns = np.asarray(margins.get("columns"), dtype=float)
    expected_margin_shape = (len(HELD), entity_count, 3)
    if (
        rows.shape != expected_margin_shape
        or columns.shape != expected_margin_shape
        or not np.isfinite(rows).all()
        or not np.isfinite(columns).all()
        or np.any(rows <= 0)
        or np.any(columns <= 0)
    ):
        raise ValueError("locked held margins are invalid")
    for name, values in tables.items():
        np.testing.assert_allclose(
            values.sum(axis=-1), rows, atol=IPF_TOLERANCE, rtol=0
        )
        np.testing.assert_allclose(
            values.sum(axis=-2), columns, atol=IPF_TOLERANCE, rtol=0
        )
    tuning = record.get("tuning")
    if not isinstance(tuning, dict):
        raise ValueError("locked prediction lacks tuning diagnostics")
    expected_fits = {
        "field_primary",
        "field_nuclear",
        "field_hypergraph",
        "field_membership_permuted",
        "field_destroyed",
        "pearson_structured",
        "deviance_structured",
    }
    diagnostics = tuning.get("fit_diagnostics", {})
    if set(diagnostics) != expected_fits or any(
        diagnostic.get("converged") is not True
        for diagnostic in diagnostics.values()
        if isinstance(diagnostic, dict)
    ) or any(not isinstance(diagnostic, dict) for diagnostic in diagnostics.values()):
        raise ValueError("locked structured-fit convergence diagnostics differ")
    selected = tuning.get("selected", {})
    primary = selected.get("field_primary")
    if (
        not isinstance(primary, list)
        or len(primary) != 2
        or primary[0] not in NUCLEAR_GRID[1:]
        or primary[1] not in GRAPH_GRID[1:]
    ):
        raise ValueError("locked primary hyperparameters are outside the positive grid")
    prediction_provenance = record.get("provenance")
    if not isinstance(prediction_provenance, dict):
        raise ValueError("prediction JSON lacks frozen provenance")
    base = {key: value for key, value in prediction_provenance.items() if key != "prediction_artifacts"}
    if base != expected_provenance:
        raise ValueError("prediction JSON frozen provenance differs")
    return record, predictions, tables


def public_bind(args: argparse.Namespace) -> None:
    provenance = preflight(require_sealed=True)
    prediction = Path(args.predictions)
    if _repo_relative(prediction) != provenance["paths"]["prediction_path"]:
        raise ValueError("prediction path differs from the frozen designation")
    if SCORE_AUTHORIZATION.exists():
        raise FileExistsError(f"score authorization already exists: {SCORE_AUTHORIZATION}")
    template = _read_json(AUTH_TEMPLATE)
    if template.get("schema") != "kotliarov-pbmc-score-authorization-template/1.0":
        raise ValueError("score authorization template schema differs")
    relative = _repo_relative(prediction)
    _require_github_commit_url(args.public_url, args.public_commit, blob_path=relative)
    prediction_record, _, _ = _locked_predictions(prediction, provenance)
    bundle, score_manifest = _prepared_prediction_bundle(provenance)
    if prediction_record["provenance"].get("prediction_artifacts") != bundle:
        raise PermissionError("published prediction bundle provenance differs")
    if prediction_record.get("sealed_score_manifest") != score_manifest:
        raise PermissionError("published prediction does not bind the score manifest")
    record = {
        "schema": "kotliarov-pbmc-score-authorization/1.0",
        "status": "SEALED",
        "outcome_access_authorized": True,
        "candidate": "KotliarovPBMCData",
        "prediction_path": relative,
        "prediction_sha256": _sha256(prediction),
        "prediction_bytes": prediction.stat().st_size,
        "prediction_public_url": args.public_url,
        "prediction_public_commit": args.public_commit,
        "runner_sha256": _sha256(Path(__file__)),
        "protocol_sha256": _sha256(PROTOCOL),
        "frozen_provenance": provenance,
        "publication_required_before_score": True,
    }
    SCORE_AUTHORIZATION.parent.mkdir(parents=True, exist_ok=True)
    SCORE_AUTHORIZATION.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def _require_score_authorization(
    prediction: Path,
    authorization: Path,
    expected_provenance: dict[str, object],
) -> dict[str, object]:
    record = _read_json(authorization)
    if record.get("schema") != "kotliarov-pbmc-score-authorization/1.0":
        raise PermissionError("score authorization schema differs")
    if record.get("status") != "SEALED" or record.get("outcome_access_authorized") is not True:
        raise PermissionError("score authorization forbids held pairing access")
    if record.get("prediction_sha256") != _sha256(prediction) or record.get(
        "prediction_bytes"
    ) != prediction.stat().st_size:
        raise PermissionError("prediction bytes differ from score authorization")
    relative = _repo_relative(prediction)
    if record.get("prediction_path") != relative:
        raise PermissionError("prediction path differs from score authorization")
    _require_github_commit_url(
        record.get("prediction_public_url"),
        record.get("prediction_public_commit"),
        blob_path=relative,
    )
    if record.get("runner_sha256") != _sha256(Path(__file__)) or record.get(
        "protocol_sha256"
    ) != _sha256(PROTOCOL):
        raise PermissionError("score authorization code or protocol hash differs")
    if record.get("frozen_provenance") != expected_provenance:
        raise PermissionError("score authorization frozen provenance differs")
    return record


def authorize_score(args: argparse.Namespace) -> None:
    provenance = preflight(require_sealed=True)
    prediction = Path(args.predictions)
    authorization = Path(args.authorization)
    if _repo_relative(prediction) != provenance["paths"]["prediction_path"]:
        raise ValueError("prediction path differs from frozen designation")
    if _repo_relative(authorization) != provenance["paths"]["score_authorization"]:
        raise ValueError("authorization path differs from frozen designation")
    if SCORE_RELEASE.exists():
        raise FileExistsError(f"score release already exists: {SCORE_RELEASE}")
    _require_score_authorization(prediction, authorization, provenance)
    relative = _repo_relative(authorization)
    _require_github_commit_url(
        args.authorization_public_url,
        args.authorization_public_commit,
        blob_path=relative,
    )
    record = {
        "schema": "kotliarov-pbmc-score-release/1.0",
        "status": "SEALED",
        "held_pairing_access_authorized": True,
        "authorization_path": relative,
        "authorization_sha256": _sha256(authorization),
        "authorization_bytes": authorization.stat().st_size,
        "authorization_public_url": args.authorization_public_url,
        "authorization_public_commit": args.authorization_public_commit,
        "runner_sha256": _sha256(Path(__file__)),
        "protocol_sha256": _sha256(PROTOCOL),
        "frozen_provenance": provenance,
    }
    SCORE_RELEASE.parent.mkdir(parents=True, exist_ok=True)
    SCORE_RELEASE.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def _require_score_release(
    authorization: Path,
    release: Path,
    expected_provenance: dict[str, object],
) -> dict[str, object]:
    record = _read_json(release)
    if record.get("schema") != "kotliarov-pbmc-score-release/1.0":
        raise PermissionError("score release schema differs")
    if record.get("status") != "SEALED" or record.get("held_pairing_access_authorized") is not True:
        raise PermissionError("score release forbids held pairing access")
    relative = _repo_relative(authorization)
    if record.get("authorization_path") != relative:
        raise PermissionError("score release authorization path differs")
    if record.get("authorization_sha256") != _sha256(authorization) or record.get(
        "authorization_bytes"
    ) != authorization.stat().st_size:
        raise PermissionError("authorization bytes differ from score release")
    _require_github_commit_url(
        record.get("authorization_public_url"),
        record.get("authorization_public_commit"),
        blob_path=relative,
    )
    if record.get("runner_sha256") != _sha256(Path(__file__)) or record.get(
        "protocol_sha256"
    ) != _sha256(PROTOCOL):
        raise PermissionError("score release code or protocol hash differs")
    if record.get("frozen_provenance") != expected_provenance:
        raise PermissionError("score release frozen provenance differs")
    return record


def _load_score_data(
    prediction_data: dict[str, object], path: Path = SCORE_BUNDLE
) -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    cells = _read_tsv(path / "held_cells.tsv.gz")
    states = _load_npy_gz(path / "held_adt_states.npy.gz").astype(int)
    binding = _read_json(path / "score_binding.json")
    required = {
        "held_position",
        "prediction_cell_index",
        "cell_id",
        "donor",
        "lineage",
    }
    if required - set(cells.columns):
        raise ValueError("score cell axis lacks frozen columns")
    cells = cells.sort_values("held_position").reset_index(drop=True)
    if cells["held_position"].tolist() != list(range(len(cells))):
        raise ValueError("score held positions are not a complete zero-based axis")
    prediction_index = cells["prediction_cell_index"].to_numpy(dtype=int)
    source_cells = prediction_data["cells"]
    if np.any(prediction_index < 0) or np.any(prediction_index >= len(source_cells)):
        raise ValueError("score prediction-cell indices are invalid")
    expected = source_cells.iloc[prediction_index]
    for column in ("cell_id", "donor", "lineage"):
        if cells[column].astype(str).tolist() != expected[column].astype(str).tolist():
            raise ValueError(f"score cell {column} axis differs from prediction")
    if set(expected["split"]) != {"held"} or set(cells["donor"].astype(str)) != set(HELD):
        raise ValueError("score bundle cell axis is not exactly the held donor split")
    if states.shape != (len(prediction_data["markers"]), len(cells)):
        raise ValueError("held ADT state array shape differs from score cell axis")
    if np.any(states < 0) or np.any(states > 2):
        raise ValueError("held ADT states contain invalid labels")
    return cells, states, binding


def _held_truth(
    prediction_data: dict[str, object], score_cells: pd.DataFrame, adt_states: np.ndarray
) -> dict[str, np.ndarray]:
    entities = prediction_data["entities"]
    prediction_cells = prediction_data["cells"]
    prediction_index = score_cells["prediction_cell_index"].to_numpy(dtype=int)
    rna_states = prediction_data["rna_states"][:, prediction_index]
    donor_axis = score_cells["donor"].astype(str).to_numpy()
    lineage_axis = score_cells["lineage"].astype(str).to_numpy()
    return _stats_for_donors(
        pd.DataFrame(
            {
                "donor": donor_axis,
                "lineage": lineage_axis,
                "cell_id": prediction_cells.iloc[prediction_index]["cell_id"].to_numpy(),
            }
        ),
        entities,
        rna_states,
        adt_states,
        HELD,
        "held-margin",
    )


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    x = np.asarray(first, dtype=float).ravel()
    y = np.asarray(second, dtype=float).ravel()
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _interval(values: np.ndarray) -> list[float]:
    return np.quantile(values, [0.025, 0.975]).tolist()


def _sign_flip_p(values: np.ndarray, alternative: str) -> float:
    observed = float(np.mean(values))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))))
    null = np.mean(signs * np.asarray(values)[None, :], axis=1)
    if alternative == "greater":
        return float(np.mean(null >= observed - 1e-15))
    if alternative == "less":
        return float(np.mean(null <= observed + 1e-15))
    raise ValueError("sign-flip alternative must be greater or less")


def _adaptive_best_sign_flip_p(contrasts: np.ndarray) -> float:
    """Exact left-tail test after reselecting the best comparator per assignment."""

    values = np.asarray(contrasts, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(HELD):
        raise ValueError("adaptive sign-flip contrasts must be method by held donor")
    observed = float(np.max(values.mean(axis=1)))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(HELD))))
    null = np.max(
        np.mean(values[None, :, :] * signs[:, None, :], axis=2), axis=1
    )
    return float(np.mean(null <= observed + 1e-15))


def _bh(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result.tolist()


def _donor_losses(
    losses: dict[str, np.ndarray], entities: pd.DataFrame
) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(
            [_hierarchical_loss(values[donor], entities) for donor in range(len(HELD))]
        )
        for name, values in losses.items()
    }


def _descriptive_subgroups(
    donor_loss: dict[str, np.ndarray], losses: dict[str, np.ndarray], entities: pd.DataFrame
) -> dict[str, object]:
    best_classical = min(CLASSICAL_METHODS, key=lambda name: donor_loss[name].mean())
    contrasts = {
        "unstructured": donor_loss["field_primary"] - donor_loss["field_direct"],
        "classical": donor_loss["field_primary"] - donor_loss[best_classical],
    }
    responder: dict[str, list[dict[str, object]]] = {}
    groups = {"high": HELD_HIGH, "low": HELD_LOW}
    for contrast, values in contrasts.items():
        rows = []
        p_values = []
        for label, donors in groups.items():
            index = np.asarray([HELD.index(donor) for donor in donors])
            selected = values[index]
            p = _sign_flip_p(selected, "less")
            p_values.append(p)
            rows.append({"group": label, "donors": list(donors), "mean_difference": float(selected.mean()), "sign_flip_p": p})
        for row, q in zip(rows, _bh(p_values)):
            row["bh_q"] = q
        responder[contrast] = rows
    lineage: dict[str, list[dict[str, object]]] = {}
    lineage_axis = entities["lineage"].astype(str).to_numpy()
    for label, comparator in (
        ("unstructured", "field_direct"),
        ("classical", best_classical),
    ):
        rows = []
        p_values = []
        for lineage_name in sorted(set(lineage_axis)):
            mask = lineage_axis == lineage_name
            differences = losses["field_primary"][:, mask].mean(axis=1) - losses[comparator][:, mask].mean(axis=1)
            p = _sign_flip_p(differences, "less")
            p_values.append(p)
            rows.append({"lineage": lineage_name, "mean_difference": float(differences.mean()), "sign_flip_p": p})
        for row, q in zip(rows, _bh(p_values)):
            row["bh_q"] = q
        lineage[label] = rows
    return {
        "best_classical": best_classical,
        "responder": responder,
        "lineage": lineage,
        "role": "descriptive; BH adjusted within each displayed contrast family and never used for promotion",
    }


def summarize(
    truth: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    losses: dict[str, np.ndarray],
    entities: pd.DataFrame,
    *,
    integrity_checks: dict[str, bool],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    if not integrity_checks or any(
        not isinstance(value, (bool, np.bool_)) for value in integrity_checks.values()
    ):
        raise ValueError("integrity checks must be an explicit nonempty boolean map")
    donor_loss = _donor_losses(losses, entities)
    donor_r = np.asarray(
        [
            _correlation(predictions["field_primary"], truth["field"][index])
            for index in range(len(HELD))
        ]
    )
    fisher = np.arctanh(np.clip(donor_r, -1 + 1e-12, 1 - 1e-12))
    direct = donor_loss["field_primary"] - donor_loss["field_direct"]
    destroyed = donor_loss["field_primary"] - donor_loss["field_destroyed"]
    best_classical = min(CLASSICAL_METHODS, key=lambda name: donor_loss[name].mean())
    best_matched = min(MATCHED_METHODS, key=lambda name: donor_loss[name].mean())
    classical_contrasts = np.asarray(
        [donor_loss["field_primary"] - donor_loss[name] for name in CLASSICAL_METHODS]
    )
    matched_contrasts = np.asarray(
        [donor_loss["field_primary"] - donor_loss[name] for name in MATCHED_METHODS]
    )
    classical = donor_loss["field_primary"] - donor_loss[best_classical]
    matched = donor_loss["field_primary"] - donor_loss[best_matched]
    rng = np.random.default_rng(SEED)
    high_index = np.asarray([HELD.index(donor) for donor in HELD_HIGH])
    low_index = np.asarray([HELD.index(donor) for donor in HELD_LOW])
    bootstrap = {
        "mean_fisher_z": np.empty(BOOTSTRAPS),
        "primary_minus_unstructured": np.empty(BOOTSTRAPS),
        "primary_minus_destroyed": np.empty(BOOTSTRAPS),
        "primary_minus_best_classical": np.empty(BOOTSTRAPS),
        "primary_minus_best_matched": np.empty(BOOTSTRAPS),
    }
    for draw in range(BOOTSTRAPS):
        selected = np.concatenate(
            (
                rng.choice(high_index, len(high_index), replace=True),
                rng.choice(low_index, len(low_index), replace=True),
            )
        )
        bootstrap["mean_fisher_z"][draw] = fisher[selected].mean()
        bootstrap["primary_minus_unstructured"][draw] = direct[selected].mean()
        bootstrap["primary_minus_destroyed"][draw] = destroyed[selected].mean()
        selected_classical = min(
            CLASSICAL_METHODS, key=lambda name: donor_loss[name][selected].mean()
        )
        selected_matched = min(
            MATCHED_METHODS, key=lambda name: donor_loss[name][selected].mean()
        )
        bootstrap["primary_minus_best_classical"][draw] = (
            donor_loss["field_primary"][selected].mean()
            - donor_loss[selected_classical][selected].mean()
        )
        bootstrap["primary_minus_best_matched"][draw] = (
            donor_loss["field_primary"][selected].mean()
            - donor_loss[selected_matched][selected].mean()
        )
    intervals = {name: _interval(values) for name, values in bootstrap.items()}
    p_values = {
        "mean_fisher_z": _sign_flip_p(fisher, "greater"),
        "primary_minus_unstructured": _sign_flip_p(direct, "less"),
        "primary_minus_destroyed": _sign_flip_p(destroyed, "less"),
        "primary_minus_best_classical": _adaptive_best_sign_flip_p(
            classical_contrasts
        ),
        "primary_minus_best_matched": _adaptive_best_sign_flip_p(
            matched_contrasts
        ),
    }
    relative_direct = -float(direct.mean()) / float(donor_loss["field_direct"].mean())
    relative_classical = -float(classical.mean()) / float(donor_loss[best_classical].mean())
    checks = {
        "field_correlation": bool(
            intervals["mean_fisher_z"][0] > 0
            and p_values["mean_fisher_z"] <= 0.025
            and np.count_nonzero(donor_r > 0) >= 8
        ),
        "unstructured_field": bool(
            intervals["primary_minus_unstructured"][1] < 0
            and p_values["primary_minus_unstructured"] <= 0.025
            and relative_direct >= 0.05
            and np.count_nonzero(direct < 0) >= 8
        ),
        "best_classical": bool(
            intervals["primary_minus_best_classical"][1] < 0
            and p_values["primary_minus_best_classical"] <= 0.025
            and relative_classical >= 0.05
            and np.count_nonzero(classical < 0) >= 8
        ),
        "destroyed_link": bool(
            intervals["primary_minus_destroyed"][1] < 0
            and p_values["primary_minus_destroyed"] <= 0.025
            and np.count_nonzero(destroyed < 0) >= 8
        ),
        "best_matched_field": bool(
            intervals["primary_minus_best_matched"][1] < 0
            and p_values["primary_minus_best_matched"] <= 0.025
            and np.count_nonzero(matched < 0) >= 8
        ),
        "integrity": bool(all(integrity_checks.values())),
    }
    per_entity_records = []
    for method, values in losses.items():
        for donor_index, donor in enumerate(HELD):
            for entity_index, entity in enumerate(entities.itertuples(index=False)):
                per_entity_records.append(
                    {
                        "method": method,
                        "donor": donor,
                        "marker_id": str(entity.marker_id),
                        "gene": str(entity.gene_symbol),
                        "lineage": str(entity.lineage),
                        "per_cell_deviance": float(values[donor_index, entity_index]),
                    }
                )
    summary = {
        "primary_method": "field_primary",
        "primary_endpoint": "held-table multinomial deviance per cell",
        "held_donor_field_correlations": dict(zip(HELD, donor_r.tolist())),
        "held_donor_fisher_z": dict(zip(HELD, fisher.tolist())),
        "mean_held_fisher_z": float(fisher.mean()),
        "mean_held_fisher_z_bootstrap_95_ci": intervals["mean_fisher_z"],
        "mean_per_cell_deviance_by_method": {
            name: float(values.mean()) for name, values in donor_loss.items()
        },
        "held_donor_deviance_by_method": {
            name: dict(zip(HELD, values.tolist())) for name, values in donor_loss.items()
        },
        "best_classical": best_classical,
        "best_matched_field": best_matched,
        "relative_deviance_reduction_vs_unstructured": relative_direct,
        "relative_deviance_reduction_vs_best_classical": relative_classical,
        "bootstrap_95_ci": intervals,
        "exact_one_sided_sign_flip_p": p_values,
        "favorable_donor_counts": {
            "positive_field_correlation": int(np.count_nonzero(donor_r > 0)),
            "primary_vs_unstructured": int(np.count_nonzero(direct < 0)),
            "primary_vs_best_classical": int(np.count_nonzero(classical < 0)),
            "primary_vs_destroyed": int(np.count_nonzero(destroyed < 0)),
            "primary_vs_best_matched": int(np.count_nonzero(matched < 0)),
        },
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
        "integrity_checks": dict(integrity_checks),
        "bootstrap_unit": "held donor, stratified to retain four high and five low responders",
        "bootstrap_draws": BOOTSTRAPS,
        "sign_flip_assignments": 2 ** len(HELD),
        "descriptive_subgroups": _descriptive_subgroups(donor_loss, losses, entities),
        "per_donor_marker_lineage_loss": per_entity_records,
    }
    return summary, bootstrap


def score(args: argparse.Namespace) -> None:
    provenance = preflight(require_sealed=True)
    prediction_path = Path(args.predictions)
    authorization_path = Path(args.authorization)
    release_path = Path(args.release)
    output = Path(args.output)
    expected_paths = provenance["paths"]
    for path, key in (
        (prediction_path, "prediction_path"),
        (authorization_path, "score_authorization"),
        (release_path, "score_release"),
        (output, "score_output"),
    ):
        if _repo_relative(path) != expected_paths[key]:
            raise ValueError(f"{key} differs from the frozen designation")
    if output.exists() or ARRAYS_PATH.exists() or SCORE_REFUSAL.exists():
        raise FileExistsError("prospective score output or refusal already exists")
    authorization = _require_score_authorization(
        prediction_path, authorization_path, provenance
    )
    release = _require_score_release(authorization_path, release_path, provenance)
    try:
        record, predictions, predicted_tables = _locked_predictions(
            prediction_path, provenance
        )
        prepared_bundle, expected_score_manifest = _prepared_prediction_bundle(provenance)
        _require_bundle(
            record["provenance"]["prediction_artifacts"], prepared_bundle, "prediction"
        )
        if record.get("sealed_score_manifest") != expected_score_manifest:
            raise PermissionError("prediction does not bind the prepared score manifest")
        score_artifacts = _score_artifact_bundle(SCORE_BUNDLE)
        observed_manifest = next(
            item for item in score_artifacts["artifacts"] if item["path"] == SCORE_MANIFEST
        )
        if observed_manifest != {
            "path": SCORE_MANIFEST,
            "bytes": expected_score_manifest["bytes"],
            "sha256": expected_score_manifest["sha256"],
        }:
            raise PermissionError("score bundle manifest differs from the prepared seal")
        prediction_data = _load_prediction_data(PREDICTION_BUNDLE)
        _validate_entities_and_margins(prediction_data)
        score_cells, adt_states, score_binding = _load_score_data(prediction_data)
        if score_binding.get("schema") != "kotliarov-pbmc-reducer-score-binding/1.0":
            raise PermissionError("score binding schema differs")
        if score_binding.get("held_cell_axis_sha256") != _sha256(
            SCORE_BUNDLE / "held_cells.tsv.gz"
        ):
            raise PermissionError("score binding held-cell axis hash differs")
        if score_binding.get("held_adt_states_sha256") != _sha256(
            SCORE_BUNDLE / "held_adt_states.npy.gz"
        ):
            raise PermissionError("score binding held-state hash differs")
        acquisition = _read_json(PREDICTION_BUNDLE / "source_acquisition.json")
        if score_binding.get("adt_source_sha256") != acquisition.get(
            "adt_matrix", {}
        ).get("sha256"):
            raise PermissionError("score binding ADT source hash differs")
        payload = score_binding.get("prediction_payload")
        expected_payload = prepared_bundle["artifacts"][:-1]
        if payload != expected_payload:
            raise PermissionError("score binding does not match the prediction payload")
        encoded_payload = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode()
        if score_binding.get("prediction_payload_sha256") != hashlib.sha256(
            encoded_payload
        ).hexdigest():
            raise PermissionError("score binding prediction-payload hash differs")
        truth = _held_truth(prediction_data, score_cells, adt_states)
        np.testing.assert_allclose(truth["rows"], np.asarray(record["held_margins"]["rows"]), atol=0, rtol=0)
        np.testing.assert_allclose(truth["columns"], np.asarray(record["held_margins"]["columns"]), atol=0, rtol=0)
        for name, tables in predicted_tables.items():
            np.testing.assert_allclose(tables.sum(axis=-1), truth["rows"], atol=IPF_TOLERANCE, rtol=0)
            np.testing.assert_allclose(tables.sum(axis=-2), truth["columns"], atol=IPF_TOLERANCE, rtol=0)
        losses = {
            name: np.asarray(
                [
                    [
                        multinomial_deviance_per_observation(
                            truth["table"][donor, entity], tables[donor, entity]
                        )
                        for entity in range(len(prediction_data["entities"]))
                    ]
                    for donor in range(len(HELD))
                ]
            )
            for name, tables in predicted_tables.items()
        }
        fit_diagnostics = record.get("tuning", {}).get("fit_diagnostics", {})
        expected_fits = {
            "field_primary",
            "field_nuclear",
            "field_hypergraph",
            "field_membership_permuted",
            "field_destroyed",
            "pearson_structured",
            "deviance_structured",
        }
        integrity_checks = {
            "source_and_artifact_hashes": True,
            "donor_209_excluded": "209"
            not in set(prediction_data["cells"]["donor"].astype(str)),
            "separate_margin_support": True,
            "prediction_bundle_excludes_held_pairing": not any(
                name in set(PREDICTION_OUTPUTS) for name in SCORE_OUTPUTS
            )
            and record.get("held_pairing_used") is False,
            "optimizer_convergence": set(fit_diagnostics) == expected_fits,
            "table_reconstruction": True,
            "score_binding": True,
        }
        summary, bootstrap = summarize(
            truth,
            predictions,
            losses,
            prediction_data["entities"],
            integrity_checks=integrity_checks,
        )
    except (KeyError, OSError, PermissionError, ValueError, RuntimeError, AssertionError) as error:
        _write_refusal(
            SCORE_REFUSAL,
            stage="score",
            code="RECONSTRUCTION_OR_SCORING_FAILURE",
            message=str(error),
        )
        raise
    try:
        ARRAYS_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            ARRAYS_PATH,
            entity_ids=np.asarray(record["entity_ids"]),
            held_donors=np.asarray(HELD),
            held_field_truth=truth["field"],
            **{f"loss_{name}": value for name, value in losses.items()},
            **{f"bootstrap_{name}": value for name, value in bootstrap.items()},
        )
        result = {
            "schema": "kotliarov-pbmc-confirmation/1.0",
            "status": "PASS" if summary["gate_passed"] else "FAIL",
            "scope": "held experimental batch with nine disjoint biological donors",
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
                "path": _repo_relative(ARRAYS_PATH),
                "sha256": _sha256(ARRAYS_PATH),
                "bytes": ARRAYS_PATH.stat().st_size,
            },
            "source_acquisition": _read_json(PREDICTION_BUNDLE / "source_acquisition.json"),
            "score_artifacts": score_artifacts,
            "provenance": provenance,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
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
    prepare_parser.add_argument("--rna-matrix", required=True)
    prepare_parser.add_argument("--adt-matrix", required=True)
    prepare_parser.add_argument("--metadata-root", required=True)
    prepare_parser.add_argument("--prediction-bundle", default=str(PREDICTION_BUNDLE))
    prepare_parser.add_argument("--score-bundle", default=str(SCORE_BUNDLE))
    predict_parser = subparsers.add_parser("predict")
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
