"""Run the frozen source-only Kotliarov PBMC binary coupling experiment."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import re
import subprocess
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any
import urllib.request

import numpy as np
import pandas as pd
import scipy

from experiments import confirm_combat_citeseq as classical
from experiments import confirm_gse309593_held_batches as engine
from experiments import reduce_kotliarov_pbmc as reader
from mapreg.heterogeneity_adaptive_coupling import (
    binary_table_from_helmert_coordinate,
    CouplingEstimationRefusal,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/KOTLIAROV_PBMC_BINARY_SOURCE_DEVELOPMENT_PROTOCOL_2026-08-29.md"
DESIGNATION = ROOT / "data/confirmation/kotliarov_pbmc_binary_v2/candidate_designation_v2.json"
SOURCE_MANIFEST = ROOT / "data/development/kotliarov_pbmc/source_manifest_v1.json"
TEST = ROOT / "tests/test_kotliarov_pbmc_binary_v2.py"
AUTHORIZATION = ROOT / "data/confirmation/kotliarov_pbmc_binary_v2/source_authorization_v2.json"
ATTEMPT = ROOT / "data/confirmation/kotliarov_pbmc_binary_v2/source_attempt_v2.json"
OUTPUT = ROOT / "results/development/kotliarov_pbmc_binary_v2_source_v2.json"

FROZEN_SOURCE_COMMIT = "043280178740f6da299bff6e70aeed86d3f5fdbe"
DEVELOPMENT = ("200", "207", "212", "233", "237", "245", "256", "261", "273", "277")
HELD = ("201", "205", "215", "229", "234", "236", "250", "268", "279")
EXCLUDED = ("209",)
PANEL = (
    ("CD3D", "CD3"),
    ("CD4", "CD4"),
    ("CD8A", "CD8"),
    ("MS4A1", "CD20"),
    ("CD14", "CD14"),
    ("FCGR3A", "CD16"),
    ("NCAM1", "CD56"),
    ("HLA-DRA", "HLA-DR"),
    ("IL7R", "CD127"),
)
CELL_BUDGET = 512
CELL_SALT = "KOTLIAROV-PBMC-BINARY-CELL-v2"
ADT_SALT = "KOTLIAROV-PBMC-BINARY-ADT-v2"
DESTROYED_SALT = "KOTLIAROV-PBMC-BINARY-DESTROYED-v2"
BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 20260829
TRANSPORT_GRID = (0.5, 0.75, 1.0, 1.25)
PRIMARY_BASE_GRID = tuple(product((0.1, 1.0, 10.0), (0.01, 0.1)))
RESIDUAL_GRID = tuple(product(("pearson", "deviance"), (False, True), TRANSPORT_GRID))

EXPECTED_INPUTS = {
    "matrix.h5": {
        "bytes": 82_850_550,
        "md5": "8778f578eced043bd993a474c8919139",
        "sha256": "7360396001a67a1e187f10d3b1307f2c390a7788788f10efe9999337b5f810d4",
    },
    "array.h5": {
        "bytes": 4_708_920,
        "md5": "d74235cc89e0c8edbd9637731e53c8d6",
        "sha256": "129e1da6986a7ac9b3aa3c1d7a147972c0b9d3e4691a0d83fd239a91c93f5362",
    },
}
EXPECTED_METADATA = {
    "column_data/basic_columns.h5": "9de81ec194efda0a81937d7524e07de327a65a3c510909f21195e633f1af8470",
    "row_data/basic_columns.h5": "a59b42b01d128476b0bca8dd90c62dcd3b0533a4e9c114b457a907b5df716998",
    "alternative_experiments/0/row_data/basic_columns.h5": "5122b7c387908552517fa88d39e8c4edb71726aeac3bb082cf4cb1aebd5f1866",
}

LOCAL_BINDING_PATHS = {
    "runner": "experiments/develop_kotliarov_pbmc_binary_v2.py",
    "runner_test": "tests/test_kotliarov_pbmc_binary_v2.py",
    "protocol": "docs/KOTLIAROV_PBMC_BINARY_SOURCE_DEVELOPMENT_PROTOCOL_2026-08-29.md",
    "designation": "data/confirmation/kotliarov_pbmc_binary_v2/candidate_designation_v2.json",
    "source_manifest": "data/development/kotliarov_pbmc/source_manifest_v1.json",
    "gse309593_binary_engine": "experiments/confirm_gse309593_held_batches.py",
    "combat_classical_engine": "experiments/confirm_combat_citeseq.py",
    "kotliarov_hdf5_reader": "experiments/reduce_kotliarov_pbmc.py",
    "gse279451_numerics": "experiments/evaluate_gse279451_sepsis_development.py",
    "gse279451_reducer": "experiments/reduce_gse279451_sepsis.py",
    "hierarchical_conditional_coupling": "mapreg/hierarchical_conditional_coupling.py",
    "common_effect_conditional": "mapreg/common_effect_conditional.py",
    "heterogeneity_adaptive_coupling": "mapreg/heterogeneity_adaptive_coupling.py",
    "classical_residuals": "mapreg/classical_residuals.py",
    "coupling_fields": "mapreg/coupling_fields.py",
    "table_prediction": "mapreg/table_prediction.py",
    "mapreg_package": "mapreg/__init__.py",
}
REQUIRED_THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


ENGINE_DEFAULTS = {
    "MARKER_COUNT": engine.MARKER_COUNT,
    "CELL_BUDGET": engine.CELL_BUDGET,
    "MINIMUM_INFORMATIVE_ENTITIES": engine.MINIMUM_INFORMATIVE_ENTITIES,
}


@contextmanager
def _engine_contract():
    previous = {
        "MARKER_COUNT": engine.MARKER_COUNT,
        "CELL_BUDGET": engine.CELL_BUDGET,
        "MINIMUM_INFORMATIVE_ENTITIES": engine.MINIMUM_INFORMATIVE_ENTITIES,
    }
    engine.MARKER_COUNT = len(PANEL)
    engine.CELL_BUDGET = CELL_BUDGET
    engine.MINIMUM_INFORMATIVE_ENTITIES = len(PANEL) ** 2
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(engine, name, value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            size += len(block)
            sha256.update(block)
            md5.update(block)
    return {"bytes": size, "md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _write_json_x(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _immutable_public_bytes(commit: str, relative: str) -> bytes:
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise PermissionError("public commit must be a full lowercase SHA-1")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise PermissionError("public artifact path must remain inside the repository")
    request = urllib.request.Request(
        "https://raw.githubusercontent.com/sushaan-k/coupling-fields-benchmark/"
        f"{commit}/{path.as_posix()}",
        headers={"User-Agent": "coupling-fields-benchmark/2.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
    except Exception as error:
        raise PermissionError(f"immutable public commit lacks {relative}") from error


def _runtime_contract() -> dict[str, Any]:
    return {
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "pandas_version": pd.__version__,
        "h5py_version": reader.h5py.__version__,
        "hdf5_runtime_version": reader.h5py.version.hdf5_version,
        "hdf5_runtime_version_tuple": list(reader.h5py.version.hdf5_version_tuple),
        "hdf5_built_version": ".".join(
            str(value) for value in reader.h5py.version.hdf5_built_version_tuple
        ),
        "hdf5_built_version_tuple": list(
            reader.h5py.version.hdf5_built_version_tuple
        ),
        "required_thread_environment": REQUIRED_THREAD_ENV,
    }


def _require_runtime_contract() -> dict[str, Any]:
    observed = {name: os.environ.get(name) for name in REQUIRED_THREAD_ENV}
    if observed != REQUIRED_THREAD_ENV:
        raise PermissionError(
            "source execution requires all frozen numerical thread variables to equal one"
        )
    return _runtime_contract()


def _require_ancestor(ancestor: str, descendant: str) -> None:
    if any(re.fullmatch(r"[0-9a-f]{40}", value) is None for value in (ancestor, descendant)):
        raise PermissionError("public history check requires full lowercase commit hashes")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise PermissionError("public freeze, implementation, and authorization history is not ordered")


def preflight() -> dict[str, Any]:
    designation = _read_json(DESIGNATION)
    expected_panel = [[gene, protein] for gene, protein in PANEL]
    if (
        designation.get("schema") != "kotliarov-pbmc-binary-coupling-candidate/2.0"
        or designation.get("status") != "SOURCE_ONLY_DEVELOPMENT_FROZEN_BEFORE_ADT_COUNT_ACCESS"
        or designation.get("development_donors") != list(DEVELOPMENT)
        or designation.get("held_donors") != list(HELD)
        or designation.get("excluded_donors") != list(EXCLUDED)
        or designation.get("ordered_cognates") != expected_panel
        or designation.get("cell_budget_per_donor") != CELL_BUDGET
        or designation.get("cell_selection_salt") != CELL_SALT
        or designation.get("adt_tie_salt") != ADT_SALT
        or designation.get("destroyed_link_salt") != DESTROYED_SALT
        or designation.get("entity_count") != len(PANEL) ** 2
        or designation.get("protocol_sha256") != _sha256(PROTOCOL)
        or designation.get("source_manifest_sha256") != _sha256(SOURCE_MANIFEST)
    ):
        raise PermissionError("frozen Kotliarov v2 designation differs from the runner")
    source_development = designation.get("source_development")
    access_rule = designation.get("access_rule")
    if (
        not isinstance(source_development, dict)
        or source_development.get("validation") != "leave-one-development-donor-out"
        or source_development.get("bootstrap_draws") != BOOTSTRAPS
        or source_development.get("bootstrap_seed") != BOOTSTRAP_SEED
        or source_development.get("promotion_relative_reduction") != 0.05
        or source_development.get("minimum_favorable_donors") != 8
        or not isinstance(access_rule, dict)
        or access_rule.get("source_phase_may_read_paired_development_values") is not True
        or access_rule.get("source_phase_may_read_held_adt_values") is not False
        or access_rule.get("held_phase_requires_public_source_promotion") is not True
        or access_rule.get("held_predictions_must_precede_held_adt_access") is not True
    ):
        raise PermissionError("frozen source-development or access rule differs")
    if (
        _immutable_public_bytes(FROZEN_SOURCE_COMMIT, _relative(PROTOCOL))
        != PROTOCOL.read_bytes()
    ):
        raise PermissionError("local protocol differs from the public source freeze")
    if (
        _immutable_public_bytes(FROZEN_SOURCE_COMMIT, _relative(DESIGNATION))
        != DESIGNATION.read_bytes()
    ):
        raise PermissionError("local designation differs from the public source freeze")
    return {
        "schema": "kotliarov-pbmc-binary-v2-preflight/2.0",
        "status": "SOURCE_IMPLEMENTATION_PREFLIGHT_PASS_NO_MATRIX_OPEN",
        "public_source_freeze_commit": FROZEN_SOURCE_COMMIT,
        "bindings": {
            label: {"path": relative, "sha256": _sha256(ROOT / relative)}
            for label, relative in sorted(LOCAL_BINDING_PATHS.items())
        },
        "source_input_contract": EXPECTED_INPUTS,
        "metadata_contract": EXPECTED_METADATA,
        "runtime_contract": _runtime_contract(),
        "held_adt_values_authorized": False,
        "matrix_files_opened": 0,
    }


def _validate_authorization(path: Path, authorization_commit: str) -> dict[str, Any]:
    if path.resolve() != AUTHORIZATION.resolve():
        raise PermissionError("source authorization must use the canonical repository path")
    authorization = _read_json(path)
    pre = preflight()
    expected_bindings = pre["bindings"]
    runtime = _require_runtime_contract()
    if (
        set(authorization)
        != {
            "schema",
            "status",
            "public_source_freeze_commit",
            "public_implementation_commit",
            "bindings",
            "source_input_contract",
            "runtime_contract",
            "held_adt_values_authorized",
            "source_attempt_path",
            "source_output_path",
        }
        or authorization.get("schema")
        != "kotliarov-pbmc-binary-v2-source-authorization/2.0"
        or authorization.get("status") != "SOURCE_PAIRED_DEVELOPMENT_ACCESS_AUTHORIZED"
        or authorization.get("public_source_freeze_commit") != FROZEN_SOURCE_COMMIT
        or authorization.get("bindings") != expected_bindings
        or authorization.get("source_input_contract") != EXPECTED_INPUTS
        or authorization.get("runtime_contract") != runtime
        or authorization.get("held_adt_values_authorized") is not False
        or authorization.get("source_attempt_path") != _relative(ATTEMPT)
        or authorization.get("source_output_path") != _relative(OUTPUT)
    ):
        raise PermissionError("source paired-value access is not authorized")
    implementation_commit = authorization.get("public_implementation_commit")
    if re.fullmatch(r"[0-9a-f]{40}", str(implementation_commit)) is None:
        raise PermissionError("authorization lacks a public implementation commit")
    for record in expected_bindings.values():
        if _immutable_public_bytes(str(implementation_commit), record["path"]) != (
            ROOT / record["path"]
        ).read_bytes():
            raise PermissionError(f"public implementation differs: {record['path']}")
    if _immutable_public_bytes(authorization_commit, _relative(path)) != path.read_bytes():
        raise PermissionError("local source authorization is not the public authorization")
    _require_ancestor(FROZEN_SOURCE_COMMIT, str(implementation_commit))
    _require_ancestor(str(implementation_commit), authorization_commit)
    return {
        "authorization_path": _relative(path),
        "authorization_sha256": _sha256(path),
        "public_authorization_commit": authorization_commit,
        "public_implementation_commit": str(implementation_commit),
        "runtime_contract": runtime,
    }


def _selected_source_cells(metadata_path: Path) -> pd.DataFrame:
    names, columns = reader._read_dataframe_columns(
        metadata_path,
        ("batch", "sampleid", "joint_classification_global", "dmx_hto_match", "timepoint"),
    )
    metadata = pd.DataFrame(
        {
            "cell_id": names.astype(str),
            "source_column": np.arange(len(names), dtype=np.int64),
            "batch": columns["batch"].astype(str),
            "donor": columns["sampleid"].astype(str),
            "joint": columns["joint_classification_global"].astype(str),
            "match": columns["dmx_hto_match"].astype(str),
            "timepoint": columns["timepoint"].astype(str),
        }
    )
    eligible = metadata.loc[
        metadata["donor"].isin(DEVELOPMENT)
        & (metadata["batch"] == "1")
        & (metadata["joint"] == "SNG_Singlet")
        & (metadata["match"] == "1")
        & (metadata["timepoint"] == "d0")
    ].copy()
    selected = []
    for donor in DEVELOPMENT:
        block = eligible.loc[eligible["donor"] == donor].copy()
        if len(block) < CELL_BUDGET:
            raise ValueError(f"development donor {donor} has fewer than 512 eligible cells")
        block["selection_hash"] = [
            hashlib.sha256(f"{CELL_SALT}\0{donor}\0{cell}".encode()).hexdigest()
            for cell in block["cell_id"]
        ]
        block = block.sort_values(["selection_hash", "cell_id"]).head(CELL_BUDGET)
        selected.append(block.drop(columns="selection_hash"))
    output = pd.concat(selected, ignore_index=True).sort_values("source_column").reset_index(drop=True)
    if set(output["donor"]) != set(DEVELOPMENT) or len(output) != CELL_BUDGET * len(DEVELOPMENT):
        raise AssertionError("source cell selection differs from the frozen donor budget")
    if output["donor"].isin(HELD + EXCLUDED).any():
        raise PermissionError("source selection contains a nondevelopment donor")
    return output


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _read_csc_exact_subset(
    path: Path,
    rows: np.ndarray,
    columns: np.ndarray,
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read numeric values from exactly the requested CSC columns."""

    selected_rows = np.asarray(rows, dtype=np.int64)
    selected_columns = np.asarray(columns, dtype=np.int64)
    if (
        len(np.unique(selected_rows)) != len(selected_rows)
        or len(np.unique(selected_columns)) != len(selected_columns)
        or np.any(np.diff(selected_columns) <= 0)
    ):
        raise ValueError("exact CSC subset axes must be unique and columns ordered")
    output = np.zeros((len(selected_rows), len(selected_columns)), dtype=float)
    column_reads: list[dict[str, int]] = []
    with reader.h5py.File(path, "r") as handle:
        group = handle.get("compressed_sparse_matrix")
        if not isinstance(group, reader.h5py.Group):
            raise ValueError("RNA HDF5 lacks compressed_sparse_matrix")
        required = {"data", "indices", "indptr", "shape"}
        if not required.issubset(group):
            raise ValueError("RNA compressed_sparse_matrix is incomplete")
        layout = group.attrs.get("layout")
        if isinstance(layout, bytes):
            layout = layout.decode()
        if layout != "CSC":
            raise ValueError("RNA compressed_sparse_matrix must use CSC layout")
        shape = tuple(int(value) for value in group["shape"][...])
        if shape != expected_shape:
            raise ValueError("RNA matrix dimensions differ from metadata axes")
        data = group["data"]
        indices = group["indices"]
        indptr = group["indptr"]
        if (
            not isinstance(data, reader.h5py.Dataset)
            or not isinstance(indices, reader.h5py.Dataset)
            or not isinstance(indptr, reader.h5py.Dataset)
            or data.ndim != 1
            or indices.ndim != 1
            or indptr.ndim != 1
            or len(data) != len(indices)
            or len(indptr) != shape[1] + 1
        ):
            raise ValueError("RNA CSC structural arrays are invalid")
        if "missing-value-placeholder" in data.attrs:
            raise ValueError("RNA count matrix contains encoded missing values")
        if (
            np.any(selected_rows < 0)
            or np.any(selected_rows >= shape[0])
            or np.any(selected_columns < 0)
            or np.any(selected_columns >= shape[1])
        ):
            raise ValueError("requested exact RNA subset is outside the matrix")
        row_map = np.full(shape[0], -1, dtype=np.int64)
        row_map[selected_rows] = np.arange(len(selected_rows), dtype=np.int64)
        for target_column, source_column in enumerate(selected_columns):
            start = int(indptr[int(source_column)])
            end = int(indptr[int(source_column) + 1])
            if not 0 <= start <= end <= len(data):
                raise ValueError("RNA CSC selected-column pointers are invalid")
            source_rows = np.asarray(indices[start:end], dtype=np.int64)
            source_values = np.asarray(data[start:end], dtype=float)
            if (
                len(source_rows) != len(source_values)
                or np.any(source_rows < 0)
                or np.any(source_rows >= shape[0])
                or np.any(~np.isfinite(source_values))
                or np.any(source_values < 0)
            ):
                raise ValueError("RNA CSC selected column contains an invalid entry")
            target_rows = row_map[source_rows]
            keep = target_rows >= 0
            np.add.at(output[:, target_column], target_rows[keep], source_values[keep])
            column_reads.append(
                {
                    "source_column": int(source_column),
                    "numeric_slice_start": start,
                    "numeric_slice_end": end,
                }
            )
    return output, {
        "reader": "exact_selected_column_csc",
        "selected_column_count": len(selected_columns),
        "selected_column_axis_sha256": _array_sha256(selected_columns),
        "indptr_scalar_reads": 2 * len(selected_columns),
        "indptr_full_vector_materialized": False,
        "indices_exact_slice_reads": len(selected_columns),
        "data_exact_slice_reads": len(selected_columns),
        "numeric_entries_read": int(
            sum(row["numeric_slice_end"] - row["numeric_slice_start"] for row in column_reads)
        ),
        "unselected_column_numeric_slices_read": 0,
        "numeric_slices_exactly_bound_to_selected_columns": True,
        "column_reads": column_reads,
    }


def _integer_counts(values: np.ndarray, label: str) -> np.ndarray:
    counts = np.asarray(values, dtype=float)
    if (
        not np.isfinite(counts).all()
        or np.any(counts < 0)
        or not np.array_equal(counts, np.rint(counts))
        or np.any(counts > np.iinfo(np.int64).max)
    ):
        raise ValueError(f"selected {label} values are not nonnegative integer counts")
    return counts.astype(np.int64)


def _midrank_states(counts: np.ndarray, cell_ids: list[str], donor: str) -> np.ndarray:
    values = _integer_counts(counts, "ADT")
    states = np.zeros(values.shape, dtype=np.uint8)
    for marker in range(values.shape[1]):
        order = sorted(
            range(len(cell_ids)),
            key=lambda index: (
                int(values[index, marker]),
                hashlib.sha256(f"{ADT_SALT}\0{donor}\0{marker}\0{cell_ids[index]}".encode()).hexdigest(),
                cell_ids[index],
            ),
        )
        states[np.asarray(order[CELL_BUDGET // 2 :], dtype=np.int64), marker] = 1
    if not np.all(states.sum(axis=0) == CELL_BUDGET // 2):
        raise AssertionError("ADT midrank split changed its frozen margin")
    return states


def _destroyed_states(states: np.ndarray, cell_ids: list[str], donor: str) -> np.ndarray:
    order = np.asarray(
        sorted(
            range(len(cell_ids)),
            key=lambda index: (
                hashlib.sha256(f"{DESTROYED_SALT}\0{donor}\0{cell_ids[index]}".encode()).hexdigest(),
                cell_ids[index],
            ),
        ),
        dtype=np.int64,
    )
    destroyed = np.empty_like(states)
    destroyed[order] = states[np.roll(order, 1)]
    if not np.array_equal(destroyed.sum(axis=0), states.sum(axis=0)):
        raise AssertionError("destroyed link changed protein margins")
    if sorted(map(tuple, destroyed.tolist())) != sorted(map(tuple, states.tolist())):
        raise AssertionError("destroyed link changed complete protein-state profiles")
    return destroyed


def _source_records(rna_path: Path, adt_path: Path, metadata_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not ATTEMPT.is_file():
        raise PermissionError("source count access requires the claimed one-shot attempt")
    metadata_files = {relative: metadata_root / relative for relative in EXPECTED_METADATA}
    for relative, path in metadata_files.items():
        if _sha256(path) != EXPECTED_METADATA[relative]:
            raise PermissionError(f"metadata SHA-256 differs: {relative}")
    cells = _selected_source_cells(metadata_files["column_data/basic_columns.h5"])
    rna_names = reader._read_row_names(metadata_files["row_data/basic_columns.h5"])
    adt_names, adt_columns = reader._read_dataframe_columns(
        metadata_files["alternative_experiments/0/row_data/basic_columns.h5"], ("target", "isotype")
    )
    rna_rows = reader._unique_indices(rna_names, [gene for gene, _ in PANEL], "RNA cognates")
    adt_rows = reader._unique_indices(adt_columns["target"].astype(str), [protein for _, protein in PANEL], "ADT cognates")
    if np.any(np.asarray(adt_columns["isotype"])[adt_rows].astype(bool)):
        raise ValueError("frozen ADT panel contains an isotype")
    source_columns = cells["source_column"].to_numpy(dtype=np.int64)
    rna, rna_access = _read_csc_exact_subset(
        rna_path, rna_rows, source_columns, (len(rna_names), 58_654)
    )
    adt = reader._read_dense_subset(adt_path, adt_rows, source_columns, (len(adt_names), 58_654))
    records: dict[str, dict[str, Any]] = {}
    for donor in DEVELOPMENT:
        indices = np.flatnonzero(cells["donor"].to_numpy(dtype=str) == donor)
        if len(indices) != CELL_BUDGET:
            raise AssertionError("source donor cell budget changed")
        identifiers = cells.iloc[indices]["cell_id"].astype(str).tolist()
        rna_counts = _integer_counts(rna[:, indices].T, "RNA")
        adt_counts = _integer_counts(adt[:, indices].T, "ADT")
        rna_states = (rna_counts > 0).astype(np.uint8)
        adt_states = _midrank_states(adt_counts, identifiers, donor)
        tables = engine._tables(rna_states, adt_states)
        support = engine._informative(tables)
        destroyed = engine._tables(rna_states, _destroyed_states(adt_states, identifiers, donor))
        records[donor] = {
            "tables": tables,
            "destroyed_tables": destroyed,
            "support": support,
            "rna_profile": rna_states.mean(axis=0),
            "adt_profile": np.log1p(adt_counts).mean(axis=0),
            "selected_cell_axis_sha256": hashlib.sha256(("\n".join(identifiers) + "\n").encode()).hexdigest(),
            "informative_coordinate_count": int(np.count_nonzero(support)),
        }
    audit = {
        "source_donors": list(DEVELOPMENT),
        "held_donors_read": [],
        "source_selected_cells": len(cells),
        "selected_cells_per_source_donor": CELL_BUDGET,
        "rna_rows": rna_rows.tolist(),
        "adt_rows": adt_rows.tolist(),
        "rna_axis": [gene for gene, _ in PANEL],
        "adt_axis": [protein for _, protein in PANEL],
        "full_deposited_cell_count": 58_654,
        "held_adt_columns_read": 0,
        "held_adt_dataset_values_read": 0,
        "source_attempt_preceded_rna_and_adt_matrix_access": True,
        "rna_matrix_access": rna_access,
    }
    return records, audit


def _training_arrays(records: dict[str, dict[str, Any]], donors: list[str], key: str = "tables") -> tuple[np.ndarray, ...]:
    tables = np.asarray([records[donor][key] for donor in donors])
    support = np.asarray([records[donor]["support"] for donor in donors])
    if not np.all(support.sum(axis=0) >= 2):
        raise ValueError("a frozen coordinate lacks two informative training donors")
    rna_profiles = np.asarray([records[donor]["rna_profile"] for donor in donors])
    adt_profiles = np.asarray([records[donor]["adt_profile"] for donor in donors])
    return tables, support, rna_profiles, adt_profiles


def _loss(truth: np.ndarray, prediction: np.ndarray, support: np.ndarray) -> float:
    observed = np.asarray(truth, dtype=float)
    fitted = np.asarray(prediction, dtype=float)
    mask = np.asarray(support, dtype=bool)
    if (
        observed.shape != (len(PANEL), len(PANEL), 2, 2)
        or fitted.shape != observed.shape
        or mask.shape != observed.shape[:2]
        or not np.any(mask)
    ):
        raise ValueError("donor loss inputs have incompatible shapes or empty support")
    if not np.isfinite(fitted).all() or np.any(fitted < -1e-9):
        raise FloatingPointError("prediction is not finite and nonnegative")
    fitted = np.maximum(fitted, 0.0)
    if not np.allclose(
        observed.sum(axis=-1), fitted.sum(axis=-1), rtol=0.0, atol=1e-9
    ) or not np.allclose(
        observed.sum(axis=-2), fitted.sum(axis=-2), rtol=0.0, atol=1e-9
    ):
        raise FloatingPointError("prediction changed a frozen target margin")
    positive = observed > 0
    if np.any(fitted[positive] <= 0):
        raise FloatingPointError("prediction assigns zero mass to an observed cell")
    terms = np.zeros_like(observed)
    terms[positive] = observed[positive] * np.log(
        observed[positive] / fitted[positive]
    )
    per_coordinate = 2.0 * terms.sum(axis=(-2, -1)) / observed.sum(axis=(-2, -1))
    if np.any(per_coordinate < -1e-12):
        raise FloatingPointError("prediction produced a materially negative deviance")
    per_coordinate = np.maximum(per_coordinate, 0.0)
    return float(per_coordinate[mask].mean())


def _odds_loss(
    truth: np.ndarray, log_odds: np.ndarray, alpha: float, support: np.ndarray
) -> float:
    rows, columns = engine._margins(truth)
    return _loss(
        truth,
        engine._predict_odds(log_odds, rows, columns, alpha),
        support,
    )


def _fit_pooled_poisson(tables: np.ndarray) -> dict[str, Any]:
    values = np.asarray(tables, dtype=np.int64)
    if values.ndim != 5 or values.shape[1:] != (len(PANEL), len(PANEL), 2, 2):
        raise ValueError("pooled Poisson table panel has the wrong shape")
    pooled = values.sum(axis=0)
    if np.any(pooled <= 0):
        raise CouplingEstimationRefusal(
            "pooled saturated Poisson interaction has a zero supported source cell"
        )
    log_odds = (
        np.log(pooled[..., 0, 0])
        + np.log(pooled[..., 1, 1])
        - np.log(pooled[..., 0, 1])
        - np.log(pooled[..., 1, 0])
    )
    maximum_error = 0.0
    for index in np.ndindex((len(PANEL), len(PANEL))):
        table = pooled[index]
        reconstructed = binary_table_from_helmert_coordinate(
            0.5 * float(log_odds[index]),
            table.sum(axis=1),
            table.sum(axis=0),
        )
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(reconstructed - table))) / float(table.sum()),
        )
    if maximum_error > 1e-8:
        raise CouplingEstimationRefusal(
            "pooled saturated Poisson source reconstruction missed its certificate"
        )
    return {
        "population_log_odds": log_odds,
        "fit_certificate": {
            "estimator": "unstratified saturated Poisson log-linear interaction",
            "includes_all_source_donor_tables_including_degenerate_margins": True,
            "supported_source_table_sha256": _array_sha256(pooled),
            "maximum_normalized_alpha_one_source_reconstruction_error": maximum_error,
            "threshold": 1e-8,
            "passes": True,
        },
    }


def _poisson_prediction(
    log_odds: np.ndarray, rows: np.ndarray, columns: np.ndarray, alpha: float
) -> np.ndarray:
    coordinate = np.asarray(log_odds, dtype=float)
    if coordinate.shape != (len(PANEL), len(PANEL)) or not np.isfinite(
        coordinate
    ).all():
        raise ValueError("pooled Poisson interaction coordinate is invalid")
    output = np.empty((len(PANEL), len(PANEL), 2, 2), dtype=float)
    for index in np.ndindex((len(PANEL), len(PANEL))):
        output[index] = binary_table_from_helmert_coordinate(
            0.5 * float(alpha) * float(coordinate[index]),
            rows[index],
            columns[index],
        )
    return output


def _poisson_loss(
    truth: np.ndarray, log_odds: np.ndarray, alpha: float, support: np.ndarray
) -> float:
    rows, columns = engine._margins(truth)
    return _loss(
        truth,
        _poisson_prediction(log_odds, rows, columns, alpha),
        support,
    )


def _classical_loss(
    truth: np.ndarray,
    model: dict[str, Any],
    alpha: float,
    support: np.ndarray,
) -> float:
    coordinate = np.asarray(model["source_coordinate"], dtype=float).reshape(len(PANEL), len(PANEL))
    rows, columns = engine._margins(truth)
    prediction = np.empty_like(truth, dtype=float)
    family = str(model["family"])
    engine_family = "root_deviance" if family == "deviance" else "pearson"
    for index in np.ndindex((len(PANEL), len(PANEL))):
        statistic = alpha * coordinate[index] * np.sqrt(float(CELL_BUDGET))
        pair_rows = rows[index]
        pair_columns = columns[index]
        if model["centered"]:
            total = int(pair_rows.sum())
            upper_left = max(
                0, int(pair_rows[0] + pair_columns[0] - total)
            )
            canonical = np.asarray(
                [
                    [upper_left, int(pair_rows[0] - upper_left)],
                    [
                        int(pair_columns[0] - upper_left),
                        int(pair_rows[1] - pair_columns[0] + upper_left),
                    ],
                ],
                dtype=np.int64,
            )
            statistic += classical.centered_classical_coordinate(canonical, statistic=family).null_mean_coordinate
        prediction[index] = engine._residual_table(
            float(statistic), pair_rows, pair_columns, engine_family
        )
    return _loss(truth, prediction, support)


def _select_complete(losses: dict[Any, np.ndarray]) -> tuple[Any, np.ndarray]:
    complete = {configuration: values for configuration, values in losses.items() if np.isfinite(values).all()}
    if not complete:
        raise RuntimeError("no frozen configuration completed every source-held fold")
    selected = min(complete, key=lambda configuration: (float(np.mean(complete[configuration])), configuration))
    return selected, complete[selected]


def _comparison(primary: np.ndarray, comparator: np.ndarray) -> dict[str, Any]:
    difference = np.asarray(primary, dtype=float) - np.asarray(comparator, dtype=float)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(0, len(DEVELOPMENT), size=(BOOTSTRAPS, len(DEVELOPMENT)))
    interval = np.quantile(difference[indices].mean(axis=1), [0.025, 0.975])
    relative = 1.0 - float(np.mean(primary) / np.mean(comparator))
    favorable = int(np.count_nonzero(difference < 0.0))
    passes = bool(relative >= 0.05 and interval[1] < 0.0 and favorable >= 8)
    return {
        "primary_mean_loss": float(np.mean(primary)),
        "comparator_mean_loss": float(np.mean(comparator)),
        "relative_loss_reduction": relative,
        "paired_donor_difference_95_ci": interval.tolist(),
        "bootstrap_draws": BOOTSTRAPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "favorable_donors": favorable,
        "donor_count": len(DEVELOPMENT),
        "donor_differences": {donor: float(value) for donor, value in zip(DEVELOPMENT, difference)},
        "passes": passes,
    }


def _develop(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    primary_losses = {
        (heterogeneity, ridge, alpha): np.full(len(DEVELOPMENT), np.nan)
        for heterogeneity, ridge in PRIMARY_BASE_GRID
        for alpha in TRANSPORT_GRID
    }
    common_losses = {alpha: np.full(len(DEVELOPMENT), np.nan) for alpha in TRANSPORT_GRID}
    poisson_losses = {alpha: np.full(len(DEVELOPMENT), np.nan) for alpha in TRANSPORT_GRID}
    residual_losses = {configuration: np.full(len(DEVELOPMENT), np.nan) for configuration in RESIDUAL_GRID}
    independence = np.full(len(DEVELOPMENT), np.nan)
    refusals: list[dict[str, Any]] = []
    full_mask = np.ones((len(PANEL), len(PANEL)), dtype=bool)
    support_diagnostics: dict[str, Any] = {}

    for index, held in enumerate(DEVELOPMENT):
        training = [donor for donor in DEVELOPMENT if donor != held]
        tables, support, rna_profiles, adt_profiles = _training_arrays(records, training)
        truth = records[held]["tables"]
        target_support = np.asarray(records[held]["support"], dtype=bool)
        if not np.any(target_support):
            raise ValueError(f"source-held donor {held} has no informative coordinate")
        training_support_count = support.sum(axis=0).astype(np.int64)
        support_diagnostics[held] = {
            "training_donors": training,
            "minimum_informative_training_donors": int(training_support_count.min()),
            "maximum_informative_training_donors": int(training_support_count.max()),
            "all_81_coordinates_have_at_least_two_training_donors": bool(
                np.all(training_support_count >= 2)
            ),
            "training_support_count_sha256": _array_sha256(training_support_count),
            "target_informative_coordinate_count": int(target_support.sum()),
            "shared_target_support_mask_sha256": _array_sha256(
                target_support.astype(np.uint8)
            ),
        }
        for heterogeneity, ridge in PRIMARY_BASE_GRID:
            configuration = engine.PrimaryConfig(1, heterogeneity, ridge, 0.0, 1.0)
            try:
                model = engine._fit_primary(tables, rna_profiles, adt_profiles, configuration, support)
            except (
                ValueError,
                FloatingPointError,
                CouplingEstimationRefusal,
                np.linalg.LinAlgError,
            ) as error:
                refusals.append({"fold": held, "method": "hierarchical", "configuration": asdict(configuration), "reason": str(error)})
                continue
            for alpha in TRANSPORT_GRID:
                try:
                    primary_losses[(heterogeneity, ridge, alpha)][index] = _odds_loss(
                        truth, model["population_log_odds"], alpha, target_support
                    )
                except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
                    refusals.append(
                        {
                            "fold": held,
                            "method": "hierarchical_prediction",
                            "configuration": {
                                **asdict(configuration),
                                "transport_multiplier": alpha,
                            },
                            "reason": str(error),
                        }
                    )
        try:
            common = engine._fit_common_effect(tables, full_mask, support)
            poisson = _fit_pooled_poisson(tables)
        except (
            ValueError,
            FloatingPointError,
            CouplingEstimationRefusal,
            np.linalg.LinAlgError,
        ) as error:
            raise RuntimeError(f"mandatory fitted classical comparator failed in fold {held}") from error
        for alpha in TRANSPORT_GRID:
            for method, fit, losses, loss_function in (
                ("common_effect_cmle", common, common_losses, _odds_loss),
                ("pooled_saturated_poisson", poisson, poisson_losses, _poisson_loss),
            ):
                try:
                    losses[alpha][index] = loss_function(
                        truth, fit["population_log_odds"], alpha, target_support
                    )
                except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
                    refusals.append(
                        {
                            "fold": held,
                            "method": f"{method}_prediction",
                            "transport_multiplier": alpha,
                            "reason": str(error),
                        }
                    )
        for family, centered in product(("pearson", "deviance"), (False, True)):
            try:
                model = classical._classical_model(tables, family, centered)
            except (
                ValueError,
                FloatingPointError,
                CouplingEstimationRefusal,
                np.linalg.LinAlgError,
            ) as error:
                refusals.append({"fold": held, "method": "classical_residual", "family": family, "centered": centered, "reason": str(error)})
                continue
            for alpha in TRANSPORT_GRID:
                try:
                    residual_losses[(family, centered, alpha)][index] = _classical_loss(
                        truth, model, alpha, target_support
                    )
                except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
                    refusals.append(
                        {
                            "fold": held,
                            "method": "classical_residual_prediction",
                            "family": family,
                            "centered": centered,
                            "transport_multiplier": alpha,
                            "reason": str(error),
                        }
                    )
        rows, columns = engine._margins(truth)
        independence[index] = _loss(
            truth, engine._independence(rows, columns), target_support
        )

    selected_primary, primary = _select_complete(primary_losses)
    selected_common, common = _select_complete(common_losses)
    selected_poisson, poisson = _select_complete(poisson_losses)
    selected_residual, residual = _select_complete(residual_losses)
    best_name, best = min(
        (("common_effect_cmle", common), ("pooled_saturated_poisson", poisson)),
        key=lambda pair: (float(np.mean(pair[1])), pair[0]),
    )

    destroyed = np.full(len(DEVELOPMENT), np.nan)
    heterogeneity, ridge, alpha = selected_primary
    for index, held in enumerate(DEVELOPMENT):
        training = [donor for donor in DEVELOPMENT if donor != held]
        tables, support, rna_profiles, adt_profiles = _training_arrays(records, training, key="destroyed_tables")
        model = engine._fit_primary(
            tables,
            rna_profiles,
            adt_profiles,
            engine.PrimaryConfig(1, heterogeneity, ridge, 0.0, 1.0),
            support,
        )
        destroyed[index] = _odds_loss(
            records[held]["tables"],
            model["population_log_odds"],
            alpha,
            records[held]["support"],
        )

    comparisons = {
        "selected_classical_residual": _comparison(primary, residual),
        "best_fitted_classical_interaction": {"selected_method": best_name, **_comparison(primary, best)},
        "destroyed_link": _comparison(primary, destroyed),
    }
    passes = all(record["passes"] for record in comparisons.values())
    frozen_models = None
    if passes:
        tables, support, rna_profiles, adt_profiles = _training_arrays(records, list(DEVELOPMENT))
        primary_fit = engine._fit_primary(
            tables,
            rna_profiles,
            adt_profiles,
            engine.PrimaryConfig(1, heterogeneity, ridge, 0.0, 1.0),
            support,
        )
        common_fit = engine._fit_common_effect(tables, full_mask, support)
        poisson_fit = _fit_pooled_poisson(tables)
        residual_fit = classical._classical_model(tables, selected_residual[0], selected_residual[1])
        destroyed_tables, destroyed_support, _, _ = _training_arrays(records, list(DEVELOPMENT), key="destroyed_tables")
        destroyed_fit = engine._fit_primary(
            destroyed_tables,
            rna_profiles,
            adt_profiles,
            engine.PrimaryConfig(1, heterogeneity, ridge, 0.0, 1.0),
            destroyed_support,
        )
        frozen_models = {
            "primary": {"kind": "conditional_log_odds", "alpha": alpha, "source_coordinate": primary_fit["population_log_odds"].ravel().tolist(), "fit_certificate": primary_fit["fit_certificate"]},
            "common_effect_cmle": {"kind": "conditional_log_odds", "alpha": selected_common, "source_coordinate": common_fit["population_log_odds"].ravel().tolist(), "fit_certificate": common_fit["fit_certificate"]},
            "pooled_saturated_poisson": {"kind": "pooled_saturated_poisson_loglinear_interaction", "alpha": selected_poisson, "source_coordinate": poisson_fit["population_log_odds"].ravel().tolist(), "fit_certificate": poisson_fit["fit_certificate"], "prediction": "fixed-margin log-linear reconstruction from half the transported log odds"},
            "selected_classical_residual": {**residual_fit, "alpha": selected_residual[2]},
            "destroyed_link": {"kind": "conditional_log_odds", "alpha": alpha, "source_coordinate": destroyed_fit["population_log_odds"].ravel().tolist(), "fit_certificate": destroyed_fit["fit_certificate"]},
        }

    def evaluations(
        losses: dict[Any, np.ndarray], labels: tuple[str, ...]
    ) -> list[dict[str, Any]]:
        rows = []
        for configuration, values in sorted(losses.items()):
            config = configuration if isinstance(configuration, tuple) else (configuration,)
            complete = bool(np.isfinite(values).all())
            row: dict[str, Any] = {
                "configuration": dict(zip(labels, config)),
                "status": "COMPLETE" if complete else "REFUSED",
                "completed_folds": int(np.isfinite(values).sum()),
            }
            if complete:
                row["mean_donor_equal_loss"] = float(np.mean(values))
                row["fold_losses"] = values.tolist()
            rows.append(row)
        return rows

    return {
        "status": "SOURCE_PROMOTION_PASS" if passes else "TERMINAL_SOURCE_PROMOTION_REFUSAL",
        "passes_source_promotion_gate": passes,
        "source_donor_axis": list(DEVELOPMENT),
        "support_diagnostics": support_diagnostics,
        "selected_primary": {"heterogeneity_penalty": selected_primary[0], "ridge_penalty": selected_primary[1], "graph_penalty": 0.0, "transport_multiplier": selected_primary[2]},
        "selected_classical_residual": {"family": selected_residual[0], "exact_null_centered": selected_residual[1], "transport_multiplier": selected_residual[2]},
        "selected_common_effect_transport": selected_common,
        "selected_pooled_poisson_transport": selected_poisson,
        "best_fitted_classical_interaction": best_name,
        "fold_losses": {
            "primary": primary.tolist(),
            "selected_classical_residual": residual.tolist(),
            "common_effect_cmle": common.tolist(),
            "pooled_saturated_poisson": poisson.tolist(),
            "best_fitted_classical_interaction": best.tolist(),
            "destroyed_link": destroyed.tolist(),
            "independence": independence.tolist(),
        },
        "candidate_evaluations": {
            "hierarchical": evaluations(
                primary_losses,
                ("heterogeneity_penalty", "ridge_penalty", "transport_multiplier"),
            ),
            "classical_residual": evaluations(
                residual_losses,
                ("family", "exact_null_centered", "transport_multiplier"),
            ),
            "common_effect_cmle": evaluations(
                common_losses, ("transport_multiplier",)
            ),
            "pooled_saturated_poisson": evaluations(
                poisson_losses, ("transport_multiplier",)
            ),
        },
        "comparisons": comparisons,
        "refusals": refusals,
        "frozen_source_models": frozen_models,
        "held_adt_access_authorized": False,
    }


def run_source(args: argparse.Namespace) -> None:
    if ATTEMPT.exists() or OUTPUT.exists():
        raise FileExistsError("Kotliarov v2 source attempt is already consumed")
    authorization = _validate_authorization(Path(args.authorization), args.authorization_commit)
    rna_path = Path(args.rna_matrix).resolve()
    adt_path = Path(args.adt_matrix).resolve()
    metadata_root = Path(args.metadata_root).resolve()
    _write_json_x(
        ATTEMPT,
        {
            "schema": "kotliarov-pbmc-binary-v2-source-attempt/2.0",
            "status": "CLAIMED_ONE_SHOT_BEFORE_COUNT_DATASET_OPEN",
            "created_at_utc": _timestamp(),
            "authorization": authorization,
            "expected_input_identities": EXPECTED_INPUTS,
            "matrix_byte_access_begins_after_this_record": True,
            "held_adt_values_authorized": False,
            "rerun_permitted": False,
        },
    )
    identities: dict[str, dict[str, Any]] = {}
    try:
        for path in (rna_path, adt_path):
            identities[path.name] = _file_identity(path)
        if identities != EXPECTED_INPUTS:
            raise PermissionError(
                "Kotliarov matrix bytes differ from the frozen public inputs"
            )
        with _engine_contract():
            records, access_audit = _source_records(
                rna_path, adt_path, metadata_root
            )
            result = _develop(records)
        result.update(
            {
                "schema": "kotliarov-pbmc-binary-v2-source-result/2.0",
                "source_attempt_sha256": _sha256(ATTEMPT),
                "authorization": authorization,
                "input_identities": identities,
                "access_audit": access_audit,
                "ordered_cognates": [[gene, protein] for gene, protein in PANEL],
            }
        )
    except Exception as error:
        result = {
            "schema": "kotliarov-pbmc-binary-v2-source-result/2.0",
            "status": "TERMINAL_SOURCE_EXECUTION_REFUSAL",
            "passes_source_promotion_gate": False,
            "reason_code": type(error).__name__,
            "reason": str(error),
            "source_attempt_sha256": _sha256(ATTEMPT),
            "authorization": authorization,
            "input_identities": identities,
            "held_adt_access_authorized": False,
        }
    _write_json_x(OUTPUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    source = subparsers.add_parser("source")
    source.add_argument("--rna-matrix", required=True)
    source.add_argument("--adt-matrix", required=True)
    source.add_argument("--metadata-root", required=True)
    source.add_argument("--authorization", default=str(AUTHORIZATION))
    source.add_argument("--authorization-commit", required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(preflight(), indent=2, sort_keys=True))
    else:
        run_source(args)


if __name__ == "__main__":
    main()
