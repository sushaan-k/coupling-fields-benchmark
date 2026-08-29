"""Prepare and score the sealed GSE181897 control-PBMC held stages.

``prepare`` reads separate RNA and ADT margins and freezes predictions without
constructing a same-cell table. ``score`` is the only command that may join the
two modalities. Internal pools 8--9 must pass before confirmation pools 10--11
can be authorized. Every numeric command has one exclusive attempt claim.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable

import h5py
import numpy as np

from experiments import develop_gse181897_source_models as source_model
from experiments import reduce_gse181897_source as source_reducer
from mapreg.heterogeneity_adaptive_coupling import (
    expected_binary_table_from_log_odds,
)


ROOT = Path(__file__).resolve().parents[1]
ACCESSION = "GSE181897"
PROTOCOL_PATH = ROOT / "data/confirmation/gse181897_control_citeseq/protocol_v1.json"
DEFAULT_PREFLIGHT = ROOT / "data/development/gse181897_source/axis_preflight_v2.json"
DEFAULT_SOURCE_CANDIDATE = (
    ROOT / "results/development/gse181897_source_candidate_v1.json"
)
CONFIRMATION_DIRECTORY = ROOT / "data/confirmation/gse181897_control_citeseq"
RESULT_DIRECTORY = ROOT / "results/confirmation/gse181897_control_citeseq"
DEFAULT_H5AD = source_reducer.DEFAULT_CACHE / source_reducer.SOURCE_H5AD_NAME

MARKER_COUNT = 17
COORDINATE_COUNT = MARKER_COUNT**2
CELL_BUDGET = 128
MINIMUM_COORDINATES = 232
MINIMUM_MARKER_POSITIVES = 4
MAXIMUM_MARKER_POSITIVES = 124
BOOTSTRAPS = 20_000
BOOTSTRAP_SALT = "GSE181897-CONTROL-HELD-BOOTSTRAP-v1"
TOPOLOGY_NULL_COUNT = 63
CLASSICAL_COMPARATORS = (
    "common_effect_cmle",
    "pooled_saturated_poisson",
    "primary_classical_residual",
)
BASE_METHODS = (
    "primary",
    "matched_graph_zero",
    "common_effect_cmle",
    "pooled_saturated_poisson",
    "pearson_residual",
    "root_deviance_residual",
    "primary_classical_residual",
    "destroyed_link",
    "independence",
)
TOPOLOGY_METHODS = tuple(
    f"topology_null_{index:02d}" for index in range(TOPOLOGY_NULL_COUNT)
)
METHOD_AXIS = BASE_METHODS + TOPOLOGY_METHODS
SOURCE_GATE_CHECKS = {
    "selected_graph_penalty_is_nonzero",
    "relative_equal_batch_reduction_vs_matched_zero_at_least_0_05",
    "at_least_7_of_8_outer_pool_means_improve",
    "at_least_27_of_39_source_donors_improve",
    "bootstrap_upper_95_vs_matched_zero_below_zero",
    "primary_mean_below_source_selected_classical",
    "bootstrap_upper_95_vs_source_selected_classical_below_zero",
    "all_mandatory_estimators_complete",
    "comparison_mask_floor_passes",
    "every_source_donor_support_floor_passes",
}

STAGE_DONORS = {
    "internal": source_reducer.INTERNAL_DONORS_BY_BATCH,
    "confirmation": source_reducer.CONFIRMATION_DONORS_BY_BATCH,
}
STAGE_EXCLUSIONS = {
    "internal": {"51": (9, 70), "52": (8, 91)},
    "confirmation": {},
}
STAGE_REQUIRED_DONORS = {"internal": 9, "confirmation": 12}
STAGE_FAVORABLE_DONORS = {"internal": 8, "confirmation": 10}
REQUIRED_THREAD_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
IMPLEMENTATION_FILES = source_reducer.CAMPAIGN_IMPLEMENTATION_FILES
PUBLIC_ORIGIN = "https://github.com/sushaan-k/coupling-fields-benchmark.git"

TAG_PREFIX = "gse181897-control-citeseq-v1-"
BASE_FREEZE_TAGS = (
    source_reducer.CANDIDATE_TAG,
    source_reducer.IMPLEMENTATION_TAG,
    source_reducer.AXIS_PREFLIGHT_TAG,
    source_reducer.SOURCE_AUTHORIZATION_TAG,
    f"{TAG_PREFIX}source-candidate",
)
ACTION_TAGS = {
    "internal_prepare": f"{TAG_PREFIX}internal-prepare-authorized",
    "internal_score": f"{TAG_PREFIX}internal-score-authorized",
    "confirmation_prepare": f"{TAG_PREFIX}confirmation-prepare-authorized",
    "confirmation_score": f"{TAG_PREFIX}confirmation-score-authorized",
}
UPSTREAM_TAGS = {
    "internal_prepare": (),
    "internal_score": (
        f"{TAG_PREFIX}internal-prepare-authorized",
        f"{TAG_PREFIX}internal-predictions",
    ),
    "confirmation_prepare": (
        f"{TAG_PREFIX}internal-prepare-authorized",
        f"{TAG_PREFIX}internal-predictions",
        f"{TAG_PREFIX}internal-score-authorized",
        f"{TAG_PREFIX}internal-result",
    ),
    "confirmation_score": (
        f"{TAG_PREFIX}internal-prepare-authorized",
        f"{TAG_PREFIX}internal-predictions",
        f"{TAG_PREFIX}internal-score-authorized",
        f"{TAG_PREFIX}internal-result",
        f"{TAG_PREFIX}confirmation-prepare-authorized",
        f"{TAG_PREFIX}confirmation-predictions",
    ),
}


def _stage_paths(stage: str) -> dict[str, Path]:
    if stage == "internal":
        stem = "internal"
        result = RESULT_DIRECTORY / "internal_validation_result_v1.json"
    elif stage == "confirmation":
        stem = "confirmation"
        result = RESULT_DIRECTORY / "primary_confirmation_result_v1.json"
    else:
        raise ValueError("unknown held stage")
    return {
        "h5ad": DEFAULT_H5AD,
        "axis_preflight": DEFAULT_PREFLIGHT,
        "source_candidate": DEFAULT_SOURCE_CANDIDATE,
        "margin_npz": CONFIRMATION_DIRECTORY / f"{stem}_margin_counts_v1.npz",
        "margin_manifest": (CONFIRMATION_DIRECTORY / f"{stem}_margin_manifest_v1.json"),
        "prediction_npz": CONFIRMATION_DIRECTORY / f"{stem}_predictions_v1.npz",
        "prediction_manifest": (
            CONFIRMATION_DIRECTORY / f"{stem}_prediction_manifest_v1.json"
        ),
        "prepare_terminal": (
            CONFIRMATION_DIRECTORY / f"{stem}_prepare_terminal_v1.json"
        ),
        "prepare_attempt": (CONFIRMATION_DIRECTORY / f"{stem}_prepare_attempt_v1.json"),
        "score_attempt": CONFIRMATION_DIRECTORY / f"{stem}_score_attempt_v1.json",
        "score_terminal": CONFIRMATION_DIRECTORY / f"{stem}_score_terminal_v1.json",
        "prepare_authorization": (
            CONFIRMATION_DIRECTORY / f"{stem}_prepare_authorization_v1.json"
        ),
        "score_authorization": (
            CONFIRMATION_DIRECTORY / f"{stem}_score_authorization_v1.json"
        ),
        "result": result,
    }


def _action_paths(action: str) -> dict[str, Path]:
    if action not in ACTION_TAGS:
        raise ValueError("unknown held action")
    stage = action.split("_", 1)[0]
    paths = _stage_paths(stage)
    common = {
        "h5ad": paths["h5ad"],
        "axis_preflight": paths["axis_preflight"],
        "source_candidate": paths["source_candidate"],
        "margin_npz": paths["margin_npz"],
        "margin_manifest": paths["margin_manifest"],
        "prediction_npz": paths["prediction_npz"],
        "prediction_manifest": paths["prediction_manifest"],
        "prepare_terminal": paths["prepare_terminal"],
    }
    if action.endswith("prepare"):
        common["attempt"] = paths["prepare_attempt"]
    else:
        common["attempt"] = paths["score_attempt"]
        common["score_terminal"] = paths["score_terminal"]
        common["result"] = paths["result"]
    if stage == "confirmation":
        common["internal_result"] = _stage_paths("internal")["result"]
    return common


def _input_artifact_keys(action: str) -> tuple[str, ...]:
    keys: tuple[str, ...] = ()
    if action.endswith("score"):
        keys += (
            "margin_npz",
            "margin_manifest",
            "prediction_npz",
            "prediction_manifest",
            "prepare_terminal",
        )
    if action.startswith("confirmation"):
        keys += ("internal_result",)
    return keys


@dataclass(frozen=True)
class BoundCandidate:
    artifact_sha256: str
    mask: np.ndarray
    coordinate_axis: tuple[str, ...]
    models: dict[str, dict[str, Any]]
    topology_nulls: tuple[dict[str, Any], ...]
    snapshot_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _axis_sha256(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON token {token}")
            ),
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path.name}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return payload


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())


def _write_npz_exclusive(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def _replace_json(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("r+") as stream:
        stream.seek(0)
        stream.truncate()
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())


def _contains_pending(value: object) -> bool:
    if isinstance(value, str):
        return "PENDING" in value
    if isinstance(value, dict):
        return any(_contains_pending(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_pending(item) for item in value)
    return False


def _require_runtime() -> None:
    wrong = {
        key: os.environ.get(key)
        for key, expected in REQUIRED_THREAD_ENVIRONMENT.items()
        if os.environ.get(key) != expected
    }
    if wrong:
        raise PermissionError(f"held runtime thread environment differs: {wrong}")


def _implementation_snapshot() -> dict[str, Any]:
    return {
        "runtime": source_reducer._campaign_runtime(),
        "files_sha256": source_reducer._campaign_implementation_hashes(),
    }


def _expected_donor_axis(stage: str) -> tuple[str, ...]:
    return tuple(
        donor
        for batch in sorted(STAGE_DONORS[stage])
        for donor in STAGE_DONORS[stage][batch]
    )


def _expected_batch_axis(stage: str) -> tuple[int, ...]:
    return tuple(
        batch
        for batch in sorted(STAGE_DONORS[stage])
        for _ in STAGE_DONORS[stage][batch]
    )


def _build_held_plan_from_axes(
    stage: str,
    barcodes: np.ndarray,
    batches: np.ndarray,
    conditions: np.ndarray,
    exp_ids: np.ndarray,
    free_ids: np.ndarray,
) -> source_reducer.SourcePlan:
    if stage not in STAGE_DONORS:
        raise ValueError("unknown held stage")
    length = len(barcodes)
    if not all(
        len(axis) == length for axis in (batches, conditions, exp_ids, free_ids)
    ):
        raise ValueError("held metadata axes have inconsistent lengths")
    if length == 0 or len(set(map(str, barcodes.tolist()))) != length:
        raise ValueError("held cell axis is empty or duplicated")
    stage_batches = tuple(sorted(STAGE_DONORS[stage]))
    stage_control = np.isin(batches.astype(int), stage_batches) & (
        conditions.astype(str) == source_reducer.CONTROL_CONDITION
    )
    expected_units = {
        (batch, donor)
        for batch, donors in STAGE_DONORS[stage].items()
        for donor in donors
    } | {(batch, donor) for donor, (batch, _) in STAGE_EXCLUSIONS[stage].items()}
    observed_units = set(
        zip(batches[stage_control].astype(int), exp_ids[stage_control].astype(str))
    )
    if observed_units != expected_units:
        raise ValueError(f"{stage} control units differ from the frozen allocation")
    expected_donors = set(_expected_donor_axis(stage))
    authorized = stage_control & np.isin(exp_ids.astype(str), tuple(expected_donors))
    donor_axis: list[str] = []
    free_axis: list[str] = []
    batch_axis: list[int] = []
    rows_by_donor: list[np.ndarray] = []
    cells_by_donor: list[np.ndarray] = []
    audits: list[dict[str, Any]] = []
    for batch in stage_batches:
        for donor in STAGE_DONORS[stage][batch]:
            eligible = np.flatnonzero(
                authorized
                & (batches.astype(int) == batch)
                & (exp_ids.astype(str) == donor)
            )
            if len(eligible) < CELL_BUDGET:
                raise ValueError(
                    f"{stage} batch {batch} donor {donor} has fewer than 128 controls"
                )
            donor_free_ids = sorted(set(free_ids[eligible].astype(str).tolist()))
            if len(donor_free_ids) != 1:
                raise ValueError(f"held donor {donor} has a nonunique free_id")
            ordered = sorted(
                eligible.tolist(),
                key=lambda row: (
                    source_reducer._selection_hash(batch, donor, str(barcodes[row])),
                    str(barcodes[row]),
                    row,
                ),
            )[:CELL_BUDGET]
            selected = np.asarray(sorted(ordered), dtype=np.int64)
            selected_cells = barcodes[selected].astype(str)
            donor_axis.append(donor)
            free_axis.append(donor_free_ids[0])
            batch_axis.append(batch)
            rows_by_donor.append(selected)
            cells_by_donor.append(selected_cells)
            audits.append(
                {
                    "batch": batch,
                    "exp_id": int(donor),
                    "free_id": int(donor_free_ids[0]),
                    "eligible_control_cells": len(eligible),
                    "selected_cells": CELL_BUDGET,
                    "selected_row_indices_sha256": _array_sha256(selected),
                    "selected_cell_axis_sha256": _axis_sha256(selected_cells),
                }
            )
    if tuple(donor_axis) != _expected_donor_axis(stage):
        raise AssertionError("held donor axis differs from the frozen order")
    selected_rows = np.stack(rows_by_donor)
    selected_barcodes = np.stack(cells_by_donor)
    if len(set(selected_rows.ravel().tolist())) != selected_rows.size:
        raise ValueError("held cell selection contains a duplicate row")
    if np.any(conditions[selected_rows.ravel()].astype(str) != "C"):
        raise PermissionError("a non-control row entered held selection")
    return source_reducer.SourcePlan(
        donor_axis=tuple(donor_axis),
        free_id_axis=tuple(free_axis),
        batch_axis=tuple(batch_axis),
        selected_rows=selected_rows,
        selected_barcodes=selected_barcodes,
        authorized_rows=authorized,
        donor_audit=tuple(audits),
    )


def _validate_axis_preflight(
    frozen: dict[str, Any], observed: source_reducer.AxisInspection
) -> None:
    if (
        frozen.get("schema") != "gse181897-axis-preflight/1.1"
        or frozen.get("status") != "AXES_FROZEN_UNIQUE_X_NUMERIC_UNREAD"
    ):
        raise PermissionError("axis preflight is not the frozen unread artifact")
    if (
        frozen.get("numeric_access", {}).get("decoded_X_entries") != 0
        or frozen.get("numeric_access", {}).get("matrix_datasets_indexed") != []
    ):
        raise PermissionError("axis preflight records numeric matrix access")
    obs = frozen.get("hdf5", {}).get("obs", {})
    var = frozen.get("hdf5", {}).get("var", {})
    if (
        obs.get("unique_rows") != 136_142
        or obs.get("index_is_unique") is not True
        or var.get("unique_rows") != 20_399
        or var.get("index_is_unique") is not True
    ):
        raise PermissionError("axis preflight lacks exact uniqueness certificates")
    for key in ("source", "hdf5", "panel", "source_plan"):
        first = json.loads(json.dumps(frozen.get(key), allow_nan=False))
        second = json.loads(json.dumps(observed.payload.get(key), allow_nan=False))
        if key == "source":
            first.pop("acquisition", None)
            second.pop("acquisition", None)
        if key == "hdf5":
            for frame in ("obs", "var"):
                first.get(frame, {}).pop("unique_rows", None)
                first.get(frame, {}).pop("index_is_unique", None)
                second.get(frame, {}).pop("unique_rows", None)
                second.get(frame, {}).pop("index_is_unique", None)
        if first != second:
            raise PermissionError(f"axis preflight {key} differs from the live H5AD")


def _inspect_held_axes(
    h5ad_path: Path, preflight_path: Path, stage: str
) -> tuple[source_reducer.AxisInspection, source_reducer.SourcePlan]:
    observed = source_reducer.inspect_axes(h5ad_path)
    frozen = _read_json(preflight_path)
    _validate_axis_preflight(frozen, observed)
    with h5py.File(h5ad_path, "r") as handle:
        obs = handle["obs"]
        barcodes = source_reducer._frame_index(obs)
        batches = source_reducer._frame_column(obs, "batch").astype(str)
        conditions = source_reducer._frame_column(obs, "cond").astype(str)
        exp_ids = source_reducer._frame_column(obs, "exp_id").astype(str)
        free_ids = source_reducer._frame_column(obs, "free_id").astype(str)
    plan = _build_held_plan_from_axes(
        stage, barcodes, batches, conditions, exp_ids, free_ids
    )
    return observed, plan


def _read_authorized_csr_states(
    matrix: h5py.Group,
    plan: source_reducer.SourcePlan,
    columns: tuple[int, ...],
    stage: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read binary states for frozen columns, scanning no unauthorized row.

    CSR row decoding necessarily scans every ``/X/indices`` entry in a selected
    row. ``/X/data`` is indexed only at positions belonging to frozen columns.
    """

    source_reducer._matrix_metadata(matrix)
    if stage not in STAGE_DONORS:
        raise ValueError("unknown held stage")
    if len(columns) != len(set(columns)) or any(
        column < 0 or column >= source_reducer.EXPECTED_X_SHAPE[1] for column in columns
    ):
        raise ValueError("held numeric feature columns are invalid")
    rows = np.asarray(plan.selected_rows, dtype=np.int64).ravel()
    if rows.size != len(plan.donor_axis) * CELL_BUDGET:
        raise ValueError("held numeric row plan has the wrong size")
    if np.any(rows < 0) or np.any(rows >= source_reducer.EXPECTED_X_SHAPE[0]):
        raise ValueError("held numeric row plan is out of range")
    if np.any(~plan.authorized_rows[rows]):
        raise PermissionError("held numeric row plan contains an unauthorized cell")
    expected_batches = set(STAGE_DONORS[stage])
    if set(plan.batch_axis) != expected_batches:
        raise PermissionError("held plan crossed its stage batch boundary")

    endpoints = np.unique(np.concatenate((rows, rows + 1))).astype(np.int64)
    indptr_values = np.asarray(matrix["indptr"][endpoints], dtype=np.int64)
    pointers = dict(zip(endpoints.tolist(), indptr_values.tolist()))
    lookup = np.full(source_reducer.EXPECTED_X_SHAPE[1], -1, dtype=np.int32)
    lookup[np.asarray(columns, dtype=np.int64)] = np.arange(
        len(columns), dtype=np.int32
    )
    states = np.zeros((len(rows), len(columns)), dtype=bool)
    index_entries_scanned = 0
    requested_data_entries = 0
    for output_row, source_row in enumerate(rows):
        start = pointers[int(source_row)]
        end = pointers[int(source_row + 1)]
        if start < 0 or end < start or end > source_reducer.EXPECTED_X_DATA_LENGTH:
            raise ValueError("held CSR row has malformed pointers")
        indices = np.asarray(matrix["indices"][start:end], dtype=np.int64)
        index_entries_scanned += len(indices)
        if (
            np.any(indices < 0)
            or np.any(indices >= source_reducer.EXPECTED_X_SHAPE[1])
            or len(np.unique(indices)) != len(indices)
        ):
            raise ValueError("held CSR row has invalid or duplicate indices")
        targets = lookup[indices]
        keep = np.flatnonzero(targets >= 0)
        if not len(keep):
            continue
        values = np.asarray(matrix["data"][start + keep], dtype=np.float64)
        if (
            not np.isfinite(values).all()
            or np.any(values < 0)
            or not np.array_equal(values, np.rint(values))
        ):
            raise ValueError("held frozen entries are not nonnegative integers")
        states[output_row, targets[keep]] = values > 0
        requested_data_entries += len(values)
    return states, {
        "matrix_datasets_indexed": ["/X/indptr", "/X/indices", "/X/data"],
        "selected_rows": len(rows),
        "selected_row_axis_sha256": _array_sha256(plan.selected_rows),
        "frozen_feature_columns": len(columns),
        "frozen_feature_column_axis_sha256": _array_sha256(
            np.asarray(columns, dtype=np.int64)
        ),
        "csr_indptr_entries_decoded": len(endpoints),
        "csr_index_entries_scanned": index_entries_scanned,
        "requested_X_data_entries_decoded": requested_data_entries,
        "unrequested_X_data_entries_decoded": 0,
        "out_of_panel_index_positions_scanned": (
            index_entries_scanned - requested_data_entries
        ),
        "out_of_panel_index_positions_used_for_membership_filter": (
            index_entries_scanned - requested_data_entries
        ),
        "out_of_panel_index_positions_retained_after_filter": 0,
        "out_of_panel_index_positions_used_for_scientific_aggregation": 0,
        "out_of_panel_index_positions_used_for_model_input": 0,
        "out_of_panel_X_data_entries_decoded": 0,
        "numeric_batches_decoded": sorted(expected_batches),
        "unauthorized_held_batch_rows_scanned": 0,
        "non_control_rows_scanned": 0,
        "unselected_authorized_rows_scanned": 0,
    }


def _read_margin_counts(
    matrix: h5py.Group,
    plan: source_reducer.SourcePlan,
    rna_columns: tuple[int, ...],
    adt_columns: tuple[int, ...],
    stage: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if set(rna_columns) & set(adt_columns):
        raise ValueError("RNA and ADT frozen feature columns overlap")
    rna_states, rna_audit = _read_authorized_csr_states(
        matrix, plan, rna_columns, stage
    )
    rna_positive = rna_states.reshape(
        len(plan.donor_axis), CELL_BUDGET, MARKER_COUNT
    ).sum(axis=1, dtype=np.int16)
    rna_states.fill(False)
    del rna_states
    adt_states, adt_audit = _read_authorized_csr_states(
        matrix, plan, adt_columns, stage
    )
    adt_positive = adt_states.reshape(
        len(plan.donor_axis), CELL_BUDGET, MARKER_COUNT
    ).sum(axis=1, dtype=np.int16)
    adt_states.fill(False)
    del adt_states
    return (
        rna_positive,
        adt_positive,
        {
            "mode": "MARGINS_ONLY_NO_SAME_CELL_MODALITY_JOIN",
            "modality_reads_sequential": True,
            "same_cell_joint_tables_constructed": 0,
            "same_cell_binary_rows_retained": 0,
            "rna": rna_audit,
            "adt": adt_audit,
            "combined_csr_index_entries_scanned": (
                rna_audit["csr_index_entries_scanned"]
                + adt_audit["csr_index_entries_scanned"]
            ),
            "combined_requested_X_data_entries_decoded": (
                rna_audit["requested_X_data_entries_decoded"]
                + adt_audit["requested_X_data_entries_decoded"]
            ),
            "combined_unrequested_X_data_entries_decoded": 0,
        },
    )


def _read_joint_states(
    matrix: h5py.Group,
    plan: source_reducer.SourcePlan,
    rna_columns: tuple[int, ...],
    adt_columns: tuple[int, ...],
    stage: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    columns = rna_columns + adt_columns
    states, audit = _read_authorized_csr_states(matrix, plan, columns, stage)
    states = states.reshape(len(plan.donor_axis), CELL_BUDGET, 2 * MARKER_COUNT)
    return (
        states[:, :, :MARKER_COUNT],
        states[:, :, MARKER_COUNT:],
        {
            **audit,
            "mode": "AUTHORIZED_SAME_CELL_SCORE_JOIN",
        },
    )


def _git(arguments: list[str]) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _local_tag_ids(tag: str) -> tuple[str, str]:
    if _git(["cat-file", "-t", tag]) != "tag":
        raise PermissionError(f"{tag} is not an annotated local tag")
    return _git(["rev-parse", tag]), _git(["rev-parse", f"{tag}^{{}}"])


def _remote_tag_ids(tag: str) -> tuple[str, str]:
    output = _git(
        [
            "ls-remote",
            PUBLIC_ORIGIN,
            f"refs/tags/{tag}",
            f"refs/tags/{tag}^{{}}",
        ]
    )
    records = {
        reference: object_id
        for object_id, reference in (
            line.split("\t", 1) for line in output.splitlines() if line
        )
    }
    direct = records.get(f"refs/tags/{tag}")
    peeled = records.get(f"refs/tags/{tag}^{{}}")
    if direct is None or peeled is None:
        raise PermissionError(f"public annotated tag {tag} is incomplete")
    return direct, peeled


def _git_blob_sha256(commit: str, path: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def _tag_snapshot(tag: str) -> dict[str, str]:
    local = _local_tag_ids(tag)
    remote = _remote_tag_ids(tag)
    if local != remote:
        raise PermissionError(f"local and public tags differ for {tag}")
    return {
        "tag": tag,
        "annotated_tag_object": local[0],
        "peeled_commit": local[1],
    }


def _require_ancestor(first: str, second: str) -> None:
    if (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", first, second],
            cwd=ROOT,
            check=False,
            capture_output=True,
        ).returncode
        != 0
    ):
        raise PermissionError("held public-freeze ancestry is invalid")


def _published_path_matches(tag: str, path: Path) -> None:
    relative = _display_path(path)
    if Path(relative).is_absolute():
        raise PermissionError("a public held artifact is outside the repository")
    completed = subprocess.run(
        ["git", "show", f"{tag}:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    if completed.stdout != path.read_bytes():
        raise PermissionError(f"public {tag} bytes differ for {relative}")


def _tagged_paths(action: str) -> dict[str, tuple[Path, ...]]:
    internal = _stage_paths("internal")
    confirmation = _stage_paths("confirmation")
    paths: dict[str, tuple[Path, ...]] = {
        source_reducer.CANDIDATE_TAG: (
            ROOT / "data/confirmation/gse181897_control_citeseq/"
            "candidate_designation_v1.json",
        ),
        source_reducer.IMPLEMENTATION_TAG: tuple(
            ROOT / relative for relative in IMPLEMENTATION_FILES
        ),
        source_reducer.AXIS_PREFLIGHT_TAG: (DEFAULT_PREFLIGHT,),
        source_reducer.SOURCE_AUTHORIZATION_TAG: (
            source_reducer.DEFAULT_SOURCE_AUTHORIZATION,
        ),
        f"{TAG_PREFIX}source-candidate": (
            DEFAULT_SOURCE_CANDIDATE,
            source_reducer.DEFAULT_MODEL_TERMINAL,
            source_reducer.DEFAULT_SOURCE_ATTEMPT,
        ),
        f"{TAG_PREFIX}internal-prepare-authorized": (
            internal["prepare_authorization"],
        ),
        f"{TAG_PREFIX}internal-predictions": (
            internal["margin_npz"],
            internal["margin_manifest"],
            internal["prediction_npz"],
            internal["prediction_manifest"],
            internal["prepare_terminal"],
            internal["prepare_attempt"],
        ),
        f"{TAG_PREFIX}internal-score-authorized": (internal["score_authorization"],),
        f"{TAG_PREFIX}internal-result": (
            internal["result"],
            internal["score_terminal"],
            internal["score_attempt"],
        ),
        f"{TAG_PREFIX}confirmation-prepare-authorized": (
            confirmation["prepare_authorization"],
        ),
        f"{TAG_PREFIX}confirmation-predictions": (
            confirmation["margin_npz"],
            confirmation["margin_manifest"],
            confirmation["prediction_npz"],
            confirmation["prediction_manifest"],
            confirmation["prepare_terminal"],
            confirmation["prepare_attempt"],
        ),
    }
    paths[ACTION_TAGS[action]] = (
        _stage_paths(action.split("_", 1)[0])[
            "prepare_authorization"
            if action.endswith("prepare")
            else "score_authorization"
        ],
    )
    return paths


def _verify_public_authorization(
    action: str, authorization_path: Path, authorization: dict[str, Any]
) -> dict[str, str]:
    predecessor_tags = BASE_FREEZE_TAGS + UPSTREAM_TAGS[action]
    records = authorization.get("verified_freeze_chain")
    if not isinstance(records, list) or len(records) != len(predecessor_tags):
        raise PermissionError("held authorization freeze-chain axis differs")
    snapshots = []
    tagged_paths = _tagged_paths(action)
    for expected_tag, recorded in zip(predecessor_tags, records):
        observed = _tag_snapshot(expected_tag)
        if recorded != observed:
            raise PermissionError(
                f"held authorization has wrong {expected_tag} binding"
            )
        snapshots.append(observed)
        for path in tagged_paths.get(expected_tag, ()):
            if not path.is_file():
                raise PermissionError(
                    f"required public artifact is absent: {path.name}"
                )
            _published_path_matches(expected_tag, path)
    authorization_snapshot = _tag_snapshot(ACTION_TAGS[action])
    _published_path_matches(ACTION_TAGS[action], authorization_path)
    lineage = [record["peeled_commit"] for record in snapshots]
    lineage.append(authorization_snapshot["peeled_commit"])
    for first, second in zip(lineage, lineage[1:]):
        _require_ancestor(first, second)
    return {
        "tag": authorization_snapshot["tag"],
        "tag_object": authorization_snapshot["annotated_tag_object"],
        "commit": authorization_snapshot["peeled_commit"],
    }


def _validate_bound_file(node: dict[str, Any], path: Path, label: str) -> None:
    if node.get("path") != _display_path(path) or node.get("sha256") != _sha256(path):
        raise PermissionError(f"authorization {label} binding differs")


def _verify_h5ad_after_numeric_read(path: Path) -> str:
    if (
        path.stat().st_size != source_reducer.SOURCE_H5AD_BYTES
        or _sha256(path) != source_reducer.SOURCE_H5AD_SHA256
    ):
        raise PermissionError("held H5AD changed during numeric access")
    return source_reducer.SOURCE_H5AD_SHA256


def _validate_authorization(
    authorization_path: Path,
    action: str,
    bound_paths: dict[str, Path],
) -> tuple[dict[str, Any], dict[str, str]]:
    expected_authorization_path = _stage_paths(action.split("_", 1)[0])[
        "prepare_authorization" if action.endswith("prepare") else "score_authorization"
    ]
    if authorization_path.resolve() != expected_authorization_path.resolve():
        raise PermissionError("held authorization path is not canonical")
    authorization = _read_json(authorization_path)
    if authorization.get("schema") != "gse181897-held-authorization/1.0":
        raise PermissionError("held authorization has the wrong schema")
    if (
        authorization.get("status") != "AUTHORIZED"
        or authorization.get("stage") != action
    ):
        raise PermissionError("held stage is not authorized")
    if _contains_pending(authorization):
        raise PermissionError("held authorization contains a PENDING binding")
    paths = authorization.get("paths")
    if not isinstance(paths, dict) or set(paths) != set(bound_paths):
        raise PermissionError("held authorization path axis differs")
    for key, path in bound_paths.items():
        if paths[key] != _display_path(path):
            raise PermissionError(f"held authorization has wrong {key} path")
    bindings = authorization.get("bindings")
    if not isinstance(bindings, dict):
        raise PermissionError("held authorization lacks bindings")
    _validate_bound_file(bindings.get("protocol", {}), PROTOCOL_PATH, "protocol")
    _validate_bound_file(
        bindings.get("axis_preflight", {}),
        bound_paths["axis_preflight"],
        "axis preflight",
    )
    _validate_bound_file(
        bindings.get("source_candidate", {}),
        bound_paths["source_candidate"],
        "source candidate",
    )
    _validate_bound_file(bindings.get("h5ad", {}), bound_paths["h5ad"], "H5AD")
    if (
        bound_paths["h5ad"].stat().st_size != source_reducer.SOURCE_H5AD_BYTES
        or bindings["h5ad"].get("sha256") != source_reducer.SOURCE_H5AD_SHA256
    ):
        raise PermissionError("held H5AD source binding differs")
    expected_inputs = {
        key: {
            "path": _display_path(bound_paths[key]),
            "sha256": _sha256(bound_paths[key]),
        }
        for key in _input_artifact_keys(action)
    }
    if bindings.get("input_artifacts") != expected_inputs:
        raise PermissionError("held authorization input-artifact bindings differ")
    implementation = _implementation_snapshot()
    if bindings.get("implementation") != implementation:
        raise PermissionError("held implementation snapshot differs")
    public = _verify_public_authorization(action, authorization_path, authorization)
    return authorization, public


def _claim_attempt(
    path: Path,
    stage_name: str,
    authorization_path: Path,
    authorization: dict[str, Any],
    public_freeze: dict[str, str],
) -> dict[str, Any]:
    attempt = {
        "schema": "gse181897-held-attempt/1.0",
        "status": "CLAIMED_BEFORE_NUMERIC_X_ACCESS",
        "stage": stage_name,
        "authorization_path": _display_path(authorization_path),
        "authorization_sha256": _sha256(authorization_path),
        "authorization_payload_sha256": hashlib.sha256(
            json.dumps(
                authorization, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest(),
        "public_freeze": public_freeze,
        "numeric_X_access_before_claim": 0,
        "consumption_rule": (
            "any later matrix access, interruption, refusal, exception, hash "
            "mismatch, or partial output consumes this attempt"
        ),
    }
    _write_json_exclusive(path, attempt)
    return attempt


def _execution_snapshot(
    action: str, authorization_path: Path, paths: dict[str, Path]
) -> dict[str, Any]:
    return {
        "authorization_sha256": _sha256(authorization_path),
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "axis_preflight_sha256": _sha256(paths["axis_preflight"]),
        "source_candidate_sha256": _sha256(paths["source_candidate"]),
        "attempt_sha256": _sha256(paths["attempt"]),
        "input_artifacts_sha256": {
            key: _sha256(paths[key]) for key in _input_artifact_keys(action)
        },
        "implementation": _implementation_snapshot(),
    }


def _require_array_hash(
    node: dict[str, Any], value_key: str, hash_key: str, shape: tuple[int, ...]
) -> np.ndarray:
    values = np.asarray(node.get(value_key), dtype=float)
    if values.shape != shape or not np.isfinite(values).all():
        raise PermissionError(f"frozen {value_key} array is malformed")
    if node.get(hash_key) != _array_sha256(values):
        raise PermissionError(f"frozen {value_key} hash differs")
    return values


def _validate_source_selection_contract(artifact: dict[str, Any]) -> None:
    development = artifact.get("development")
    if not isinstance(development, dict):
        raise PermissionError("source development record is absent")
    gate = development.get("source_go_gate")
    checks = gate.get("checks", {}) if isinstance(gate, dict) else {}
    if (
        not isinstance(gate, dict)
        or gate.get("passes") is not True
        or not isinstance(checks, dict)
        or set(checks) != SOURCE_GATE_CHECKS
        or not all(value is True for value in checks.values())
        or checks.get("comparison_mask_floor_passes") is not True
        or checks.get("every_source_donor_support_floor_passes") is not True
    ):
        raise PermissionError("source statistical or support gate is incomplete")
    selection = development.get("final_all_source_selection")
    if not isinstance(selection, dict):
        raise PermissionError("source final selection record is absent")
    base = selection.get("stage_a_graph_zero", {}).get("selected_configuration")
    primary = selection.get("stage_b_nonzero_hypergraph", {}).get(
        "selected_configuration"
    )
    fixed = selection.get("stage_b_nonzero_hypergraph", {}).get("fixed_from_stage_a")
    if not isinstance(base, dict) or not isinstance(primary, dict) or fixed != base:
        raise PermissionError("source staged selection binding differs")
    if (
        base.get("heterogeneity_penalty") not in source_model.HETEROGENEITY_GRID
        or base.get("ridge_penalty") not in source_model.RIDGE_GRID
        or base.get("transport_multiplier") not in source_model.TRANSPORT_GRID
        or primary.get("graph_neighbors") not in source_model.NEIGHBOR_GRID
        or primary.get("graph_penalty") not in source_model.GRAPH_GRID
        or primary.get("graph_penalty") == 0.0
        or any(primary.get(key) != base.get(key) for key in base)
    ):
        raise PermissionError("source selected configuration is off the frozen grid")
    candidate = artifact["candidate"]
    canonical = candidate.get("canonical_configuration", {})
    if any(canonical.get(key) != value for key, value in primary.items()):
        raise PermissionError("source final candidate differs from selected primary")
    models = candidate.get("models", {})
    if models.get("primary", {}).get("configuration") != primary:
        raise PermissionError("source primary model configuration differs")
    expected_zero = {**primary, "graph_penalty": 0.0}
    if models.get("matched_graph_zero", {}).get("configuration") != expected_zero:
        raise PermissionError("source graph-zero model is not matched")
    comparators = selection.get("comparators", {})
    for model_name, selection_key in (
        ("common_effect_cmle", "selected_common_transport"),
        ("pooled_saturated_poisson", "selected_poisson_transport"),
    ):
        alpha = comparators.get(selection_key)
        if alpha not in source_model.TRANSPORT_GRID or models.get(model_name, {}).get(
            "configuration"
        ) != {"transport_multiplier": alpha}:
            raise PermissionError(f"source {model_name} selection differs")
    residual_transports = comparators.get("selected_residual_transports")
    selected_residual = comparators.get("selected_residual")
    if (
        not isinstance(residual_transports, dict)
        or set(residual_transports) != set(source_model.RESIDUAL_FAMILIES)
        or any(
            alpha not in source_model.TRANSPORT_GRID
            for alpha in residual_transports.values()
        )
        or not isinstance(selected_residual, dict)
        or selected_residual.get("family") not in source_model.RESIDUAL_FAMILIES
        or selected_residual.get("transport_multiplier")
        != residual_transports[selected_residual["family"]]
    ):
        raise PermissionError("source residual selection differs")
    for family, alpha in residual_transports.items():
        if models.get(f"{family}_residual", {}).get("configuration") != {
            "family": family,
            "transport_multiplier": alpha,
        }:
            raise PermissionError("source residual model configuration differs")
    destroyed_alpha = comparators.get("selected_destroyed_transport")
    expected_destroyed = {**primary, "transport_multiplier": destroyed_alpha}
    if (
        destroyed_alpha not in source_model.TRANSPORT_GRID
        or models.get("destroyed_link", {}).get("configuration") != expected_destroyed
    ):
        raise PermissionError("source destroyed-link selection differs")


def _load_bound_candidate(path: Path) -> BoundCandidate:
    artifact = _read_json(path)
    statistical_go = artifact.get("source_statistical_go_gate_passed")
    if statistical_go is None:
        statistical_go = (
            artifact.get("development", {}).get("source_go_gate", {}).get("passes")
        )
    if (
        artifact.get("schema") != "gse181897-b0-b7-source-candidate/1.0"
        or artifact.get("status") != "SOURCE_GO_GATE_PASSED_CANDIDATE_FROZEN"
        or statistical_go is not True
        or artifact.get("internal_numeric_access_authorized") is not False
        or artifact.get("candidate") is None
    ):
        raise PermissionError("source artifact is not a passed frozen candidate")
    attempt = artifact.get("source_campaign_attempt")
    if (
        not isinstance(attempt, dict)
        or attempt.get("path") != _display_path(source_reducer.DEFAULT_SOURCE_ATTEMPT)
        or attempt.get("sha256") != _sha256(source_reducer.DEFAULT_SOURCE_ATTEMPT)
    ):
        raise PermissionError("source candidate lacks its exclusive attempt binding")
    terminal = _read_json(source_reducer.DEFAULT_MODEL_TERMINAL)
    if (
        terminal.get("schema") != "gse181897-source-model-terminal/1.0"
        or terminal.get("status") != "SOURCE_MODEL_COMPLETE_STATISTICAL_GO_PASSED"
        or terminal.get("attempt_path") != attempt["path"]
        or terminal.get("attempt_sha256") != attempt["sha256"]
        or terminal.get("model_output_path") != _display_path(path)
        or terminal.get("model_output_sha256") != _sha256(path)
        or terminal.get("internal_numeric_access_authorized") is not False
    ):
        raise PermissionError("source model terminal is not complete and bound")
    _validate_source_selection_contract(artifact)
    candidate = artifact["candidate"]
    expected_axis = tuple(
        f"{rna.rna_gene}|{adt.adt_feature}"
        for rna in source_reducer.PANEL
        for adt in source_reducer.PANEL
    )
    coordinate_axis = tuple(candidate.get("coordinate_axis", ()))
    if coordinate_axis != expected_axis or candidate.get(
        "coordinate_axis_sha256"
    ) != _axis_sha256(expected_axis):
        raise PermissionError("source candidate coordinate axis differs")
    comparison = candidate.get("comparison_mask", {})
    mask = np.asarray(comparison.get("mask"), dtype=np.uint8)
    if (
        mask.shape != (MARKER_COUNT, MARKER_COUNT)
        or not np.isin(mask, (0, 1)).all()
        or int(mask.sum()) < MINIMUM_COORDINATES
        or comparison.get("mask_sha256") != _array_sha256(mask)
    ):
        raise PermissionError("source candidate final comparison mask differs")
    if (
        comparison.get("coordinate_count") != int(mask.sum())
        or comparison.get("checks")
        != {
            "at_least_232_coordinates": True,
            "every_training_donor_has_at_least_232_coordinates": True,
        }
        or len(comparison.get("training_donor_supported_coordinate_counts", {})) != 39
        or any(
            int(count) < MINIMUM_COORDINATES
            for count in comparison.get(
                "training_donor_supported_coordinate_counts", {}
            ).values()
        )
        or comparison.get("strict_pooled_fixed_margin_interior") is not True
        or comparison.get("pooled_four_cell_positivity") is not True
    ):
        raise PermissionError("source candidate mask certificate differs")

    models = candidate.get("models")
    if not isinstance(models, dict) or not set(BASE_METHODS).issubset(models):
        raise PermissionError("source candidate model axis is incomplete")
    population_methods = (
        "primary",
        "matched_graph_zero",
        "common_effect_cmle",
        "pooled_saturated_poisson",
        "destroyed_link",
    )
    for name in population_methods:
        _require_array_hash(
            models[name],
            "population_log_odds",
            "population_log_odds_sha256",
            (MARKER_COUNT, MARKER_COUNT),
        )
        alpha = models[name].get("configuration", {}).get("transport_multiplier")
        if not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)):
            raise PermissionError(f"source candidate {name} multiplier is malformed")
    for name in ("pearson_residual", "root_deviance_residual"):
        _require_array_hash(
            models[name],
            "pooled_coordinate",
            "pooled_coordinate_sha256",
            (MARKER_COUNT, MARKER_COUNT),
        )
    selected = models["primary_classical_residual"]
    reference = selected.get("model_reference")
    if reference not in {"pearson_residual", "root_deviance_residual"}:
        raise PermissionError("source-selected residual reference differs")
    if selected != {
        **models[reference],
        "selection_role": "single source-CV-selected classical comparator",
        "model_reference": reference,
    }:
        raise PermissionError("source-selected residual bytes differ from reference")

    topology = candidate.get("topology_nulls", {})
    controls = topology.get("controls")
    if (
        topology.get("count") != TOPOLOGY_NULL_COUNT
        or topology.get("selection_aware") is not True
        or not isinstance(controls, list)
        or len(controls) != TOPOLOGY_NULL_COUNT
    ):
        raise PermissionError("source topology-null axis is incomplete")
    for index, control in enumerate(controls):
        if control.get("control_index") != index:
            raise PermissionError("source topology-null order differs")
        _require_array_hash(
            control,
            "population_log_odds",
            "population_log_odds_sha256",
            (MARKER_COUNT, MARKER_COUNT),
        )
        alpha = control.get("selected_configuration", {}).get("transport_multiplier")
        config = control.get("selected_configuration", {})
        base = artifact["development"]["final_all_source_selection"][
            "stage_a_graph_zero"
        ]["selected_configuration"]
        if (
            config.get("graph_neighbors") not in source_model.NEIGHBOR_GRID
            or config.get("graph_penalty") not in source_model.GRAPH_GRID
            or any(config.get(key) != base.get(key) for key in base)
            or alpha not in source_model.TRANSPORT_GRID
        ):
            raise PermissionError("source topology-null multiplier is malformed")

    snapshot = {
        "artifact_sha256": _sha256(path),
        "coordinate_axis_sha256": candidate["coordinate_axis_sha256"],
        "mask_sha256": comparison["mask_sha256"],
        "models": {
            name: models[name].get(
                "population_log_odds_sha256",
                models[name].get("pooled_coordinate_sha256", "recipient-margins"),
            )
            for name in BASE_METHODS
        },
        "topology": [control["population_log_odds_sha256"] for control in controls],
    }
    snapshot_sha256 = hashlib.sha256(
        json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return BoundCandidate(
        artifact_sha256=_sha256(path),
        mask=mask.astype(bool),
        coordinate_axis=coordinate_axis,
        models=models,
        topology_nulls=tuple(controls),
        snapshot_sha256=snapshot_sha256,
    )


def _margin_geometry(
    rna_positive: np.ndarray, adt_positive: np.ndarray, frozen_mask: np.ndarray
) -> dict[str, np.ndarray]:
    rna = np.asarray(rna_positive, dtype=np.int16)
    adt = np.asarray(adt_positive, dtype=np.int16)
    if (
        rna.ndim != 2
        or rna.shape != adt.shape
        or rna.shape[1] != MARKER_COUNT
        or np.any(rna < 0)
        or np.any(rna > CELL_BUDGET)
        or np.any(adt < 0)
        or np.any(adt > CELL_BUDGET)
    ):
        raise ValueError("held positive-count margins are malformed")
    rna_supported = (rna >= MINIMUM_MARKER_POSITIVES) & (
        rna <= MAXIMUM_MARKER_POSITIVES
    )
    adt_supported = (adt >= MINIMUM_MARKER_POSITIVES) & (
        adt <= MAXIMUM_MARKER_POSITIVES
    )
    row_one = np.broadcast_to(rna[:, :, None], (len(rna), MARKER_COUNT, MARKER_COUNT))
    column_one = np.broadcast_to(
        adt[:, None, :], (len(adt), MARKER_COUNT, MARKER_COUNT)
    )
    lower = np.maximum(0, row_one + column_one - CELL_BUDGET)
    upper = np.minimum(row_one, column_one)
    subject_support = (
        rna_supported[:, :, None] & adt_supported[:, None, :] & (lower < upper)
    )
    evaluation = subject_support & np.asarray(frozen_mask, dtype=bool)[None, :, :]
    counts = evaluation.sum(axis=(1, 2))
    if np.any(counts < MINIMUM_COORDINATES):
        raise PermissionError(
            f"a held donor has fewer than {MINIMUM_COORDINATES} frozen coordinates"
        )
    rows = np.stack(
        (
            CELL_BUDGET - row_one,
            row_one,
        ),
        axis=-1,
    ).astype(np.int16)
    columns = np.stack(
        (
            CELL_BUDGET - column_one,
            column_one,
        ),
        axis=-1,
    ).astype(np.int16)
    return {
        "rna_positive_counts": rna,
        "adt_positive_counts": adt,
        "subject_support": subject_support,
        "evaluation_mask": evaluation,
        "supported_coordinate_counts": counts.astype(np.int16),
        "row_margins": rows,
        "column_margins": columns,
    }


def _predict_population(
    log_odds: np.ndarray,
    alpha: float,
    rows: np.ndarray,
    columns: np.ndarray,
    evaluation: np.ndarray,
) -> np.ndarray:
    predicted = np.zeros((*evaluation.shape, 2, 2), dtype=np.float64)
    for index in np.argwhere(evaluation):
        donor, rna, adt = map(int, index)
        predicted[donor, rna, adt] = expected_binary_table_from_log_odds(
            float(alpha) * float(log_odds[rna, adt]),
            rows[donor, rna, adt],
            columns[donor, rna, adt],
        )
    return predicted


def _predict_residual(
    coordinate: np.ndarray,
    family: str,
    alpha: float,
    rows: np.ndarray,
    columns: np.ndarray,
    evaluation: np.ndarray,
) -> np.ndarray:
    predicted = np.zeros((*evaluation.shape, 2, 2), dtype=np.float64)
    for index in np.argwhere(evaluation):
        donor, rna, adt = map(int, index)
        target = float(alpha) * float(coordinate[rna, adt]) * math.sqrt(CELL_BUDGET)
        predicted[donor, rna, adt] = source_model._residual_table(
            target,
            rows[donor, rna, adt],
            columns[donor, rna, adt],
            family,
        )
    return predicted


def _build_frozen_predictions(
    margin: dict[str, np.ndarray], candidate: BoundCandidate
) -> np.ndarray:
    rows = margin["row_margins"]
    columns = margin["column_margins"]
    evaluation = margin["evaluation_mask"]
    predictions: list[np.ndarray] = []
    for name in BASE_METHODS:
        model = candidate.models[name]
        if name == "independence":
            predictions.append(
                _predict_population(
                    np.zeros((MARKER_COUNT, MARKER_COUNT)),
                    1.0,
                    rows,
                    columns,
                    evaluation,
                )
            )
        elif name in {
            "pearson_residual",
            "root_deviance_residual",
            "primary_classical_residual",
        }:
            configuration = model["configuration"]
            predictions.append(
                _predict_residual(
                    np.asarray(model["pooled_coordinate"], dtype=float),
                    str(configuration["family"]),
                    float(configuration["transport_multiplier"]),
                    rows,
                    columns,
                    evaluation,
                )
            )
        else:
            predictions.append(
                _predict_population(
                    np.asarray(model["population_log_odds"], dtype=float),
                    float(model["configuration"]["transport_multiplier"]),
                    rows,
                    columns,
                    evaluation,
                )
            )
    for control in candidate.topology_nulls:
        predictions.append(
            _predict_population(
                np.asarray(control["population_log_odds"], dtype=float),
                float(control["selected_configuration"]["transport_multiplier"]),
                rows,
                columns,
                evaluation,
            )
        )
    output = np.stack(predictions, axis=1)
    expected_shape = (
        len(rows),
        len(METHOD_AXIS),
        MARKER_COUNT,
        MARKER_COUNT,
        2,
        2,
    )
    if output.shape != expected_shape or not np.isfinite(output).all():
        raise FloatingPointError("held prediction array is malformed")
    for donor, method, rna, adt in np.argwhere(
        np.broadcast_to(
            evaluation[:, None, :, :],
            (len(rows), len(METHOD_AXIS), MARKER_COUNT, MARKER_COUNT),
        )
    ):
        table = output[donor, method, rna, adt]
        if not np.allclose(
            table.sum(axis=-1), rows[donor, rna, adt]
        ) or not np.allclose(table.sum(axis=-2), columns[donor, rna, adt]):
            raise FloatingPointError("held prediction changed a frozen margin")
    return output


def _tables_from_states_once(rna: np.ndarray, adt: np.ndarray) -> np.ndarray:
    first = np.asarray(rna, dtype=bool)
    second = np.asarray(adt, dtype=bool)
    expected = (len(first), CELL_BUDGET, MARKER_COUNT)
    if first.shape != expected or second.shape != expected:
        raise ValueError("held score states have the wrong axes")
    codes = 2 * first[:, :, :, None].astype(np.uint8) + second[:, :, None, :].astype(
        np.uint8
    )
    tables = np.stack([(codes == value).sum(axis=1) for value in range(4)], axis=-1)
    return tables.reshape(len(first), MARKER_COUNT, MARKER_COUNT, 2, 2).astype(np.int16)


def _score_predictions(
    observed: np.ndarray,
    predicted: np.ndarray,
    evaluation: np.ndarray,
) -> np.ndarray:
    if predicted.shape[:2] != (len(observed), len(METHOD_AXIS)):
        raise ValueError("held prediction method axis differs")
    losses = np.empty((len(observed), len(METHOD_AXIS)), dtype=np.float64)
    for donor in range(len(observed)):
        for method in range(len(METHOD_AXIS)):
            losses[donor, method] = source_model._deviance_loss(
                observed[donor], predicted[donor, method], evaluation[donor]
            )
    if not np.isfinite(losses).all():
        raise FloatingPointError("held loss array is nonfinite")
    return losses


def _equal_pool_mean(values: np.ndarray, batches: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    batch_axis = np.asarray(batches, dtype=int)
    return float(
        np.mean(
            [array[batch_axis == batch].mean() for batch in sorted(set(batch_axis))]
        )
    )


def _bootstrap_seed(stage: str, comparator: str) -> int:
    encoded = hashlib.sha256(f"{BOOTSTRAP_SALT}|{stage}|{comparator}".encode()).digest()
    return int.from_bytes(encoded[:8], "little")


def _paired_bootstrap(
    stage: str,
    primary: np.ndarray,
    comparator: np.ndarray,
    batches: np.ndarray,
    comparator_name: str,
) -> dict[str, Any]:
    difference = np.asarray(primary, dtype=float) - np.asarray(comparator, dtype=float)
    batch_axis = np.asarray(batches, dtype=int)
    if difference.shape != batch_axis.shape or not np.isfinite(difference).all():
        raise ValueError("held bootstrap axes differ")
    generator = np.random.default_rng(_bootstrap_seed(stage, comparator_name))
    draws = []
    batch_means: dict[str, float] = {}
    for batch in sorted(set(batch_axis)):
        indices = np.flatnonzero(batch_axis == batch)
        sampled = generator.integers(0, len(indices), size=(BOOTSTRAPS, len(indices)))
        draws.append(difference[indices][sampled].mean(axis=1))
        batch_means[str(batch)] = float(difference[indices].mean())
    distribution = np.mean(np.stack(draws), axis=0)
    interval = np.quantile(distribution, (0.025, 0.975), method="linear")
    return {
        "comparator": comparator_name,
        "draws": BOOTSTRAPS,
        "unit": "donor resampled within physical pool",
        "equal_pool_weight": True,
        "seed": _bootstrap_seed(stage, comparator_name),
        "salt": BOOTSTRAP_SALT,
        "observed_equal_pool_mean_difference": _equal_pool_mean(difference, batch_axis),
        "pool_mean_differences": batch_means,
        "interval_95_percent": interval.tolist(),
        "quantile_method": "numpy linear",
    }


def _exact_sign_test(differences: np.ndarray) -> dict[str, Any]:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("held sign test requires finite donor differences")
    nonzero = values[values != 0.0]
    favorable = int(np.count_nonzero(nonzero < 0.0))
    n = len(nonzero)
    p_value = (
        sum(math.comb(n, count) for count in range(favorable, n + 1)) / 2**n
        if n
        else 1.0
    )
    return {
        "donors": len(values),
        "nonzero_donors": n,
        "exact_ties": len(values) - n,
        "favorable_donors": favorable,
        "one_sided_p": float(p_value),
        "tie_rule": "exact zero differences omitted; n is the nonzero-pair count",
    }


def _held_gate(stage: str, losses: np.ndarray, batches: np.ndarray) -> dict[str, Any]:
    if stage not in STAGE_DONORS or losses.shape != (
        STAGE_REQUIRED_DONORS[stage],
        len(METHOD_AXIS),
    ):
        raise ValueError("held gate loss axis differs")
    index = {name: position for position, name in enumerate(METHOD_AXIS)}
    primary = losses[:, index["primary"]]
    zero = losses[:, index["matched_graph_zero"]]
    primary_mean = _equal_pool_mean(primary, batches)
    zero_mean = _equal_pool_mean(zero, batches)
    relative = 1.0 - primary_mean / zero_mean
    zero_difference = primary - zero
    zero_bootstrap = _paired_bootstrap(
        stage, primary, zero, batches, "matched_graph_zero"
    )
    sign = _exact_sign_test(zero_difference)
    comparisons: dict[str, Any] = {"matched_graph_zero": zero_bootstrap}
    classical_checks: dict[str, bool] = {}
    for name in CLASSICAL_COMPARATORS:
        result = _paired_bootstrap(
            stage, primary, losses[:, index[name]], batches, name
        )
        comparisons[name] = result
        classical_checks[f"mean_vs_{name}_below_zero"] = (
            result["observed_equal_pool_mean_difference"] < 0.0
        )
        classical_checks[f"bootstrap_upper_vs_{name}_below_zero"] = (
            result["interval_95_percent"][1] < 0.0
        )
    selected_pool_means = comparisons["primary_classical_residual"][
        "pool_mean_differences"
    ]
    destroyed = _paired_bootstrap(
        stage,
        primary,
        losses[:, index["destroyed_link"]],
        batches,
        "destroyed_link",
    )
    comparisons["destroyed_link"] = destroyed
    topology_means = np.asarray(
        [
            _equal_pool_mean(losses[:, index[name]], batches)
            for name in TOPOLOGY_METHODS
        ],
        dtype=float,
    )
    empirical_p = float((1 + np.count_nonzero(topology_means <= primary_mean)) / 64)
    topology_relative = 1.0 - primary_mean / float(np.median(topology_means))
    checks = {
        "relative_reduction_vs_matched_graph_zero_at_least_0_05": relative >= 0.05,
        "favorable_donors_vs_matched_graph_zero_at_least_threshold": (
            sign["favorable_donors"] >= STAGE_FAVORABLE_DONORS[stage]
        ),
        "exact_sign_p_vs_matched_graph_zero_at_most_0_025": (
            sign["one_sided_p"] <= 0.025
        ),
        "both_pool_means_vs_matched_graph_zero_below_zero": all(
            value < 0.0 for value in zero_bootstrap["pool_mean_differences"].values()
        ),
        "bootstrap_upper_vs_matched_graph_zero_below_zero": (
            zero_bootstrap["interval_95_percent"][1] < 0.0
        ),
        **classical_checks,
        "both_pool_means_vs_selected_classical_below_zero": all(
            value < 0.0 for value in selected_pool_means.values()
        ),
        "topology_empirical_p_at_most_0_05": empirical_p <= 0.05,
        "relative_reduction_vs_median_topology_null_at_least_0_03": (
            topology_relative >= 0.03
        ),
        "bootstrap_upper_vs_destroyed_link_below_zero": (
            destroyed["interval_95_percent"][1] < 0.0
        ),
        "independence_reported": np.isfinite(losses[:, index["independence"]]).all(),
    }
    return {
        "passes": all(checks.values()),
        "checks": checks,
        "required_donors": STAGE_REQUIRED_DONORS[stage],
        "required_favorable_donors_vs_matched_graph_zero": (
            STAGE_FAVORABLE_DONORS[stage]
        ),
        "primary_equal_pool_mean_loss": primary_mean,
        "matched_graph_zero_equal_pool_mean_loss": zero_mean,
        "relative_reduction_vs_matched_graph_zero": relative,
        "exact_sign_test_vs_matched_graph_zero": sign,
        "comparisons": comparisons,
        "topology": {
            "null_count": TOPOLOGY_NULL_COUNT,
            "null_equal_pool_mean_losses": topology_means.tolist(),
            "empirical_p": empirical_p,
            "median_null_equal_pool_mean_loss": float(np.median(topology_means)),
            "relative_reduction_vs_median_null": topology_relative,
        },
    }


def _array_records(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "dtype": np.asarray(values).dtype.str,
            "shape": list(np.asarray(values).shape),
            "sha256": _array_sha256(np.asarray(values)),
        }
        for name, values in arrays.items()
    }


def _load_npz_bound(
    path: Path, expected: dict[str, dict[str, Any]]
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise PermissionError(f"{path.name} member axis differs")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if _array_records(arrays) != expected:
        raise PermissionError(f"{path.name} array bytes differ")
    return arrays


def _selection_certificate(plan: source_reducer.SourcePlan) -> dict[str, Any]:
    return {
        "donor_axis": list(plan.donor_axis),
        "donor_axis_sha256": _axis_sha256(plan.donor_axis),
        "batch_axis": list(plan.batch_axis),
        "batch_axis_sha256": _array_sha256(np.asarray(plan.batch_axis, dtype=np.int16)),
        "cell_budget_per_donor": CELL_BUDGET,
        "selected_row_axis_sha256": _array_sha256(plan.selected_rows),
        "selected_cell_axes_sha256": [
            record["selected_cell_axis_sha256"] for record in plan.donor_audit
        ],
        "cell_identifiers_published": False,
        "row_indices_published": False,
    }


def _margin_arrays(
    margin: dict[str, np.ndarray], plan: source_reducer.SourcePlan
) -> dict[str, np.ndarray]:
    return {
        "donor_axis": np.asarray(plan.donor_axis, dtype=str),
        "batch_axis": np.asarray(plan.batch_axis, dtype=np.int16),
        "rna_positive_counts": margin["rna_positive_counts"].astype(np.int16),
        "adt_positive_counts": margin["adt_positive_counts"].astype(np.int16),
        "subject_support": margin["subject_support"].astype(np.uint8),
        "evaluation_mask": margin["evaluation_mask"].astype(np.uint8),
        "supported_coordinate_counts": margin["supported_coordinate_counts"].astype(
            np.int16
        ),
    }


def _prediction_arrays(
    predictions: np.ndarray,
    margin: dict[str, np.ndarray],
    plan: source_reducer.SourcePlan,
) -> dict[str, np.ndarray]:
    return {
        "donor_axis": np.asarray(plan.donor_axis, dtype=str),
        "batch_axis": np.asarray(plan.batch_axis, dtype=np.int16),
        "method_axis": np.asarray(METHOD_AXIS, dtype=str),
        "evaluation_mask": margin["evaluation_mask"].astype(np.uint8),
        "predicted_tables": np.asarray(predictions, dtype=np.float64),
    }


def _write_prepare_artifacts(
    stage: str,
    paths: dict[str, Path],
    attempt: dict[str, Any],
    plan: source_reducer.SourcePlan,
    candidate: BoundCandidate,
    margin: dict[str, np.ndarray],
    predictions: np.ndarray,
    numeric_audit: dict[str, Any],
) -> None:
    margin_arrays = _margin_arrays(margin, plan)
    _write_npz_exclusive(paths["margin_npz"], margin_arrays)
    margin_manifest = {
        "schema": "gse181897-held-margin-manifest/1.0",
        "status": "MARGINS_FROZEN_WITHOUT_SAME_CELL_LINKS",
        "created_at_utc": _timestamp(),
        "stage": stage,
        "source_candidate": {
            "path": _display_path(paths["source_candidate"]),
            "sha256": candidate.artifact_sha256,
            "snapshot_sha256": candidate.snapshot_sha256,
        },
        "attempt": {
            "path": _display_path(paths["attempt"]),
            "sha256": _sha256(paths["attempt"]),
        },
        "selection": _selection_certificate(plan),
        "same_cell_joint_tables_constructed": 0,
        "same_cell_binary_rows_retained": 0,
        "numeric_access": numeric_audit,
        "arrays": _array_records(margin_arrays),
        "output": {
            "path": _display_path(paths["margin_npz"]),
            "sha256": _sha256(paths["margin_npz"]),
        },
    }
    _write_json_exclusive(paths["margin_manifest"], margin_manifest)

    prediction_arrays = _prediction_arrays(predictions, margin, plan)
    _write_npz_exclusive(paths["prediction_npz"], prediction_arrays)
    prediction_manifest = {
        "schema": "gse181897-held-prediction-manifest/1.0",
        "status": "PREDICTIONS_FROZEN_BEFORE_SAME_CELL_SCORE_ACCESS",
        "created_at_utc": _timestamp(),
        "stage": stage,
        "source_candidate": margin_manifest["source_candidate"],
        "margin_inputs": {
            "npz_path": _display_path(paths["margin_npz"]),
            "npz_sha256": _sha256(paths["margin_npz"]),
            "manifest_path": _display_path(paths["margin_manifest"]),
            "manifest_sha256": _sha256(paths["margin_manifest"]),
        },
        "method_axis": list(METHOD_AXIS),
        "method_axis_sha256": _axis_sha256(METHOD_AXIS),
        "same_cell_joint_tables_available_to_prediction": False,
        "arrays": _array_records(prediction_arrays),
        "output": {
            "path": _display_path(paths["prediction_npz"]),
            "sha256": _sha256(paths["prediction_npz"]),
        },
    }
    _write_json_exclusive(paths["prediction_manifest"], prediction_manifest)


def _load_frozen_prepare_artifacts(
    stage: str, paths: dict[str, Path], candidate: BoundCandidate
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    terminal = _read_json(paths["prepare_terminal"])
    if (
        terminal.get("schema") != "gse181897-held-prepare-terminal/1.0"
        or terminal.get("status") != "HELD_PREPARE_COMPLETE"
        or terminal.get("stage") != stage
        or terminal.get("attempt_path") != _display_path(paths["attempt"])
        or terminal.get("attempt_sha256") != _sha256(paths["attempt"])
        or terminal.get("margin_npz_sha256") != _sha256(paths["margin_npz"])
        or terminal.get("margin_manifest_sha256") != _sha256(paths["margin_manifest"])
        or terminal.get("prediction_npz_sha256") != _sha256(paths["prediction_npz"])
        or terminal.get("prediction_manifest_sha256")
        != _sha256(paths["prediction_manifest"])
    ):
        raise PermissionError("held prepare terminal is not complete and bound")
    margin_manifest = _read_json(paths["margin_manifest"])
    if (
        margin_manifest.get("schema") != "gse181897-held-margin-manifest/1.0"
        or margin_manifest.get("status") != "MARGINS_FROZEN_WITHOUT_SAME_CELL_LINKS"
        or margin_manifest.get("stage") != stage
        or margin_manifest.get("same_cell_joint_tables_constructed") != 0
        or margin_manifest.get("same_cell_binary_rows_retained") != 0
        or margin_manifest.get("source_candidate")
        != {
            "path": _display_path(paths["source_candidate"]),
            "sha256": candidate.artifact_sha256,
            "snapshot_sha256": candidate.snapshot_sha256,
        }
        or margin_manifest.get("output")
        != {
            "path": _display_path(paths["margin_npz"]),
            "sha256": _sha256(paths["margin_npz"]),
        }
    ):
        raise PermissionError("held margin manifest differs")
    margin_arrays = _load_npz_bound(
        paths["margin_npz"], margin_manifest.get("arrays", {})
    )

    prediction_manifest = _read_json(paths["prediction_manifest"])
    if (
        prediction_manifest.get("schema") != "gse181897-held-prediction-manifest/1.0"
        or prediction_manifest.get("status")
        != "PREDICTIONS_FROZEN_BEFORE_SAME_CELL_SCORE_ACCESS"
        or prediction_manifest.get("stage") != stage
        or prediction_manifest.get("source_candidate")
        != margin_manifest["source_candidate"]
        or prediction_manifest.get("margin_inputs")
        != {
            "npz_path": _display_path(paths["margin_npz"]),
            "npz_sha256": _sha256(paths["margin_npz"]),
            "manifest_path": _display_path(paths["margin_manifest"]),
            "manifest_sha256": _sha256(paths["margin_manifest"]),
        }
        or prediction_manifest.get("method_axis") != list(METHOD_AXIS)
        or prediction_manifest.get("method_axis_sha256") != _axis_sha256(METHOD_AXIS)
        or prediction_manifest.get("same_cell_joint_tables_available_to_prediction")
        is not False
        or prediction_manifest.get("output")
        != {
            "path": _display_path(paths["prediction_npz"]),
            "sha256": _sha256(paths["prediction_npz"]),
        }
    ):
        raise PermissionError("held prediction manifest differs")
    prediction_arrays = _load_npz_bound(
        paths["prediction_npz"], prediction_manifest.get("arrays", {})
    )
    return margin_arrays, prediction_arrays, margin_manifest


def _validate_internal_pass(path: Path) -> dict[str, Any]:
    result = _read_json(path)
    terminal_path = _stage_paths("internal")["score_terminal"]
    terminal = _read_json(terminal_path)
    if (
        result.get("schema") != "gse181897-held-score/1.0"
        or result.get("stage") != "internal"
        or result.get("status") != "INTERNAL_VALIDATION_PASSED"
        or result.get("gate", {}).get("passes") is not True
        or terminal.get("schema") != "gse181897-held-score-terminal/1.0"
        or terminal.get("status") != "HELD_SCORE_COMPLETE"
        or terminal.get("stage") != "internal"
        or terminal.get("result_path") != _display_path(path)
        or terminal.get("result_sha256") != _sha256(path)
    ):
        raise PermissionError("public internal result did not pass every frozen gate")
    return result


def prepare(stage: str, authorization_path: Path) -> None:
    _require_runtime()
    action = f"{stage}_prepare"
    paths = _action_paths(action)
    for key in (
        "margin_npz",
        "margin_manifest",
        "prediction_npz",
        "prediction_manifest",
        "prepare_terminal",
    ):
        if paths[key].exists():
            raise FileExistsError(
                f"held prepare output already exists: {paths[key].name}"
            )
    authorization, public = _validate_authorization(authorization_path, action, paths)
    candidate = _load_bound_candidate(paths["source_candidate"])
    if stage == "confirmation":
        _validate_internal_pass(paths["internal_result"])
    inspection, plan = _inspect_held_axes(paths["h5ad"], paths["axis_preflight"], stage)
    attempt = _claim_attempt(
        paths["attempt"], action, authorization_path, authorization, public
    )
    started = {
        "schema": "gse181897-held-prepare-terminal/1.0",
        "status": "HELD_PREPARE_STARTED_ATTEMPT_CONSUMED",
        "created_at_utc": _timestamp(),
        "stage": stage,
        "attempt_path": _display_path(paths["attempt"]),
        "attempt_sha256": _sha256(paths["attempt"]),
    }
    _write_json_exclusive(paths["prepare_terminal"], started)
    execution_snapshot = _execution_snapshot(action, authorization_path, paths)
    try:
        with h5py.File(paths["h5ad"], "r") as handle:
            rna_positive, adt_positive, numeric_audit = _read_margin_counts(
                handle["X"],
                plan,
                inspection.rna_columns,
                inspection.adt_columns,
                stage,
            )
        numeric_audit["h5ad_sha256_after_numeric_read"] = (
            _verify_h5ad_after_numeric_read(paths["h5ad"])
        )
        margin = _margin_geometry(rna_positive, adt_positive, candidate.mask)
        predictions = _build_frozen_predictions(margin, candidate)
        if execution_snapshot != _execution_snapshot(action, authorization_path, paths):
            raise PermissionError("held prepare inputs changed during numeric access")
        _write_prepare_artifacts(
            stage,
            paths,
            attempt,
            plan,
            candidate,
            margin,
            predictions,
            numeric_audit,
        )
        _replace_json(
            paths["prepare_terminal"],
            {
                **started,
                "status": "HELD_PREPARE_COMPLETE",
                "completed_at_utc": _timestamp(),
                "margin_npz_sha256": _sha256(paths["margin_npz"]),
                "margin_manifest_sha256": _sha256(paths["margin_manifest"]),
                "prediction_npz_sha256": _sha256(paths["prediction_npz"]),
                "prediction_manifest_sha256": _sha256(paths["prediction_manifest"]),
            },
        )
    except BaseException as error:
        _replace_json(
            paths["prepare_terminal"],
            {
                **started,
                "status": "TERMINAL_HELD_PREPARE_REFUSAL",
                "failed_at_utc": _timestamp(),
                "reason_code": type(error).__name__,
                "reason": str(error)
                .replace(str(ROOT.resolve()), "<repository>")
                .replace(str(Path.home().resolve()), "<home>"),
                "rerun_forbidden": True,
            },
        )
        raise


def _validate_prepare_axes(
    stage: str,
    plan: source_reducer.SourcePlan,
    margin: dict[str, np.ndarray],
    prediction: dict[str, np.ndarray],
) -> None:
    donors = np.asarray(_expected_donor_axis(stage), dtype=str)
    batches = np.asarray(_expected_batch_axis(stage), dtype=np.int16)
    if (
        not np.array_equal(margin["donor_axis"], donors)
        or not np.array_equal(prediction["donor_axis"], donors)
        or not np.array_equal(margin["batch_axis"], batches)
        or not np.array_equal(prediction["batch_axis"], batches)
        or tuple(plan.donor_axis) != tuple(donors.tolist())
        or tuple(plan.batch_axis) != tuple(batches.tolist())
        or not np.array_equal(prediction["method_axis"], np.asarray(METHOD_AXIS))
        or not np.array_equal(margin["evaluation_mask"], prediction["evaluation_mask"])
    ):
        raise PermissionError("held prepared artifact axes differ")


def _score_claimed(
    stage: str,
    paths: dict[str, Path],
    candidate: BoundCandidate,
    plan: source_reducer.SourcePlan,
    inspection: source_reducer.AxisInspection,
    margin_arrays: dict[str, np.ndarray],
    prediction_arrays: dict[str, np.ndarray],
    attempt: dict[str, Any],
) -> dict[str, Any]:
    with h5py.File(paths["h5ad"], "r") as handle:
        rna_states, adt_states, numeric_audit = _read_joint_states(
            handle["X"],
            plan,
            inspection.rna_columns,
            inspection.adt_columns,
            stage,
        )
    numeric_audit["h5ad_sha256_after_numeric_read"] = _verify_h5ad_after_numeric_read(
        paths["h5ad"]
    )
    observed = _tables_from_states_once(rna_states, adt_states)
    observed_rna = observed.sum(axis=-1)[..., 1]
    observed_adt = observed.sum(axis=-2)[..., 1]
    if not np.array_equal(
        observed_rna, margin_arrays["rna_positive_counts"]
    ) or not np.array_equal(observed_adt, margin_arrays["adt_positive_counts"]):
        raise PermissionError("score-time joint tables differ from frozen margins")
    evaluation = margin_arrays["evaluation_mask"].astype(bool)
    predictions = prediction_arrays["predicted_tables"]
    losses = _score_predictions(observed, predictions, evaluation)
    gate = _held_gate(stage, losses, np.asarray(plan.batch_axis, dtype=np.int16))
    status = (
        "INTERNAL_VALIDATION_PASSED"
        if stage == "internal" and gate["passes"]
        else "INTERNAL_VALIDATION_FAILED"
        if stage == "internal"
        else "PRIMARY_CONFIRMATION_PASSED"
        if gate["passes"]
        else "PRIMARY_CONFIRMATION_FAILED"
    )
    result = {
        "schema": "gse181897-held-score/1.0",
        "status": status,
        "created_at_utc": _timestamp(),
        "stage": stage,
        "attempt": {
            "path": _display_path(paths["attempt"]),
            "sha256": _sha256(paths["attempt"]),
            "claim": attempt,
        },
        "source_candidate": {
            "path": _display_path(paths["source_candidate"]),
            "sha256": candidate.artifact_sha256,
            "snapshot_sha256": candidate.snapshot_sha256,
        },
        "prepared_inputs": {
            key: {"path": _display_path(paths[key]), "sha256": _sha256(paths[key])}
            for key in (
                "margin_npz",
                "margin_manifest",
                "prediction_npz",
                "prediction_manifest",
                "prepare_terminal",
            )
        },
        "donor_axis": list(plan.donor_axis),
        "batch_axis": list(plan.batch_axis),
        "method_axis": list(METHOD_AXIS),
        "evaluation_coordinate_counts": evaluation.sum(axis=(1, 2)).tolist(),
        "losses_by_donor_and_method": losses.tolist(),
        "loss_array_sha256": _array_sha256(losses),
        "numeric_access": {
            **numeric_audit,
            "same_cell_joint_table_construction_calls": 1,
            "same_cell_joint_tables_constructed": int(
                len(plan.donor_axis) * COORDINATE_COUNT
            ),
        },
        "gate": gate,
        "no_retuning_or_stage_pooling": True,
    }
    return result


def score(stage: str, authorization_path: Path) -> dict[str, Any]:
    _require_runtime()
    action = f"{stage}_score"
    paths = _action_paths(action)
    if paths["result"].exists() or paths["score_terminal"].exists():
        raise FileExistsError("held score output or terminal already exists")
    authorization, public = _validate_authorization(authorization_path, action, paths)
    candidate = _load_bound_candidate(paths["source_candidate"])
    if stage == "confirmation":
        _validate_internal_pass(paths["internal_result"])
    margin_arrays, prediction_arrays, margin_manifest = _load_frozen_prepare_artifacts(
        stage, paths, candidate
    )
    inspection, plan = _inspect_held_axes(paths["h5ad"], paths["axis_preflight"], stage)
    _validate_prepare_axes(stage, plan, margin_arrays, prediction_arrays)
    if margin_manifest.get("selection") != _selection_certificate(plan):
        raise PermissionError("held selected cells changed after prediction freeze")
    attempt = _claim_attempt(
        paths["attempt"], action, authorization_path, authorization, public
    )
    started = {
        "schema": "gse181897-held-score-terminal/1.0",
        "status": "HELD_SCORE_STARTED_ATTEMPT_CONSUMED",
        "created_at_utc": _timestamp(),
        "stage": stage,
        "attempt_path": _display_path(paths["attempt"]),
        "attempt_sha256": _sha256(paths["attempt"]),
    }
    _write_json_exclusive(paths["score_terminal"], started)
    execution_snapshot = _execution_snapshot(action, authorization_path, paths)
    try:
        result = _score_claimed(
            stage,
            paths,
            candidate,
            plan,
            inspection,
            margin_arrays,
            prediction_arrays,
            attempt,
        )
        if execution_snapshot != _execution_snapshot(action, authorization_path, paths):
            raise PermissionError("held score inputs changed during numeric access")
        _write_json_exclusive(paths["result"], result)
        _replace_json(
            paths["score_terminal"],
            {
                **started,
                "status": "HELD_SCORE_COMPLETE",
                "completed_at_utc": _timestamp(),
                "result_path": _display_path(paths["result"]),
                "result_sha256": _sha256(paths["result"]),
                "gate_passed": result["gate"]["passes"],
            },
        )
        return result
    except BaseException as error:
        _replace_json(
            paths["score_terminal"],
            {
                **started,
                "status": "TERMINAL_HELD_SCORE_REFUSAL",
                "failed_at_utc": _timestamp(),
                "reason_code": type(error).__name__,
                "reason": str(error)
                .replace(str(ROOT.resolve()), "<repository>")
                .replace(str(Path.home().resolve()), "<home>"),
                "rerun_forbidden": True,
            },
        )
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "score"))
    parser.add_argument("--stage", choices=("internal", "confirmation"), required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "prepare":
        prepare(args.stage, args.authorization)
    else:
        result = score(args.stage, args.authorization)
        print(json.dumps({"status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
