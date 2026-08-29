"""One-shot held-cohort confirmation on GSE144744 RNA and ADT counts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable
import urllib.request

import numpy as np

from mapreg.classical_residuals import poisson_independence_residuals
from mapreg.heterogeneity_adaptive_coupling import (
    CouplingEstimationRefusal,
    centered_haldane_log_odds,
    expected_binary_table_from_log_odds,
    fit_structured_conditional_log_odds,
    paule_mandel_pool,
)
from mapreg.hierarchical_conditional_coupling import (
    fit_hierarchical_conditional_log_odds,
)
from mapreg.streamed_matrix_market import read_tar_axes, read_tar_matrix_subset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/confirmation/gse144744_ms"
PROTOCOL = ROOT / "docs/GSE144744_MS_HELD_COHORT_CONFIRMATION_PROTOCOL_2026-08-29.md"
CANDIDATE = DATA_DIR / "candidate_designation_v1.json"
MANIFEST = DATA_DIR / "source_manifest_v1.json"
RUNTIME = DATA_DIR / "runtime_environment_v1.json"
SOURCE_RESULT = ROOT / "results/development/gse144744_ms_source_v1.json"
RNA_RESULT = ROOT / "results/gse144744_ms_rna_v1.json"
PREDICTION_RESULT = ROOT / "results/gse144744_ms_predictions_v1.json"
ADT_RESULT = ROOT / "results/gse144744_ms_adt_v1.json"
FINAL_RESULT = ROOT / "results/gse144744_ms_confirmation_v1.json"
SCORE_AUTHORIZATION = DATA_DIR / "score_authorization_v1.json"

PUBLIC_ORIGIN = "https://github.com/sushaan-k/coupling-fields-benchmark.git"
PROTOCOL_TAG = "gse144744-ms-v1-protocol"
SCORE_AUTHORIZATION_TAG = "gse144744-ms-v1-score-authorization"
ATTEMPT_TAGS = {
    "source": "gse144744-ms-v1-source-attempt",
    "rna": "gse144744-ms-v1-rna-attempt",
    "prediction": "gse144744-ms-v1-prediction-attempt",
    "adt": "gse144744-ms-v1-adt-attempt",
    "score": "gse144744-ms-v1-score-attempt",
}
COMPLETION_TAGS = {
    "source": "gse144744-ms-v1-source",
    "rna": "gse144744-ms-v1-rna",
    "prediction": "gse144744-ms-v1-prediction",
    "adt": "gse144744-ms-v1-adt",
    "score": "gse144744-ms-v1-result",
}
ATTEMPTS = {stage: DATA_DIR / f"{stage}_attempt_v1.json" for stage in ATTEMPT_TAGS}
JOURNALS = {
    stage: DATA_DIR / f"{stage}_access_v1.jsonl"
    for stage in ("source", "rna", "adt")
}
OUTPUTS = {
    "source": SOURCE_RESULT,
    "rna": RNA_RESULT,
    "prediction": PREDICTION_RESULT,
    "adt": ADT_RESULT,
    "score": FINAL_RESULT,
}
PROTOCOL_PATHS = (
    PROTOCOL,
    CANDIDATE,
    MANIFEST,
    RUNTIME,
    Path(__file__),
    ROOT / "mapreg/classical_residuals.py",
    ROOT / "mapreg/coupling_fields.py",
    ROOT / "mapreg/heterogeneity_adaptive_coupling.py",
    ROOT / "mapreg/hierarchical_conditional_coupling.py",
    ROOT / "mapreg/streamed_matrix_market.py",
    ROOT / "tests/test_gse144744_ms_confirmation.py",
    ROOT / "tests/test_streamed_matrix_market.py",
)

CELL_BUDGET = 512
MINIMUM_BINARY_MARGIN = 5
MINIMUM_ADT_NONZERO_FRACTION = 0.05
MAXIMUM_ADT_EQUAL_VALUE_FRACTION = 0.90
CANDIDATE_MARKER_COUNT = 29
MINIMUM_LOCKED_MARKERS = 12
MINIMUM_VALID_SOURCE_DONORS = 18
MINIMUM_VALID_HELD_DONORS = 20
MINIMUM_PAIR_FRACTION = 1.0
GRAPH_NEIGHBORS = 2
MAXIMUM_CONDITION_NUMBER = 1e12
BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 20260829
ALPHA = 0.0125
MINIMUM_FAVORABLE_SOURCE_DONORS = 14

SOURCE_PAIRS = (
    ("HH-OX-01", "HH-OX-02"), ("HH-OX-03", "HH-OX-04"),
    ("HH-OX-05", "HH-OX-06"), ("HH-OX-07", "HH-OX-08"),
    ("HH-OX-09", "HH-OX-10"), ("HH-OX-11", "HH-OX-12"),
    ("HH-OX-13", "HH-OX-14"), ("HH-OX-17", "HH-OX-18"),
    ("HH-OX-21", "HH-OX-22"),
)
SOURCE_DONORS = tuple(itertools.chain.from_iterable(SOURCE_PAIRS))
SOURCE_EXPERIMENTS = (
    "HE-MK-002", "HE-MK-003", "HE-MK-004", "HE-MK-005", "HE-MK-006"
)
SOURCE_EXPERIMENT_BY_PAIR = {
    ("HH-OX-03", "HH-OX-04"): "HE-MK-002",
    ("HH-OX-09", "HH-OX-10"): "HE-MK-003",
    ("HH-OX-13", "HH-OX-14"): "HE-MK-003",
    ("HH-OX-11", "HH-OX-12"): "HE-MK-004",
    ("HH-OX-21", "HH-OX-22"): "HE-MK-004",
    ("HH-OX-05", "HH-OX-06"): "HE-MK-005",
    ("HH-OX-07", "HH-OX-08"): "HE-MK-005",
    ("HH-OX-01", "HH-OX-02"): "HE-MK-006",
    ("HH-OX-17", "HH-OX-18"): "HE-MK-006",
}
HELD_PAIRS = (
    ("HH-OX-43", "HH-OX-44"), ("HH-OX-45", "HH-OX-46"),
    ("HH-OX-47", "HH-OX-48"), ("HH-OX-49", "HH-OX-50"),
    ("HH-OX-51", "HH-OX-52"), ("HH-OX-53", "HH-OX-54"),
    ("HH-OX-55", "HH-OX-56"), ("HH-OX-57", "HH-OX-58"),
    ("HH-OX-59", "HH-OX-60"), ("HH-OX-61", "HH-OX-62"),
)
HELD_DONORS = tuple(itertools.chain.from_iterable(HELD_PAIRS))

CELL_SALT = "GSE144744-MS-CELL-v1"
LIBRARY_SALT = "GSE144744-MS-LIBRARY-v1"
ADT_SALT = "GSE144744-MS-ADT-v1"
DESTROY_SALT = "GSE144744-MS-DESTROY-v1"
HETEROGENEITY_GRID = (0.1, 1.0, 10.0)
RIDGE_GRID = (0.01, 0.1, 1.0)
GRAPH_GRID = (0.01, 0.1, 1.0)
TRANSPORT_GRID = (0.5, 0.75, 1.0, 1.25)

RAW_RESIDUAL = "untuned_poisson_signed_deviance"
CALIBRATED_RESIDUAL = "calibrated_poisson_signed_deviance"
POOLED_POISSON = "pooled_saturated_poisson"
CLASSICAL_ORDER = (
    CALIBRATED_RESIDUAL,
    "common_effect_stratified_cmle",
    POOLED_POISSON,
    "paule_mandel_random_effects_log_odds",
)
ALL_CLASSICAL = (RAW_RESIDUAL,) + CLASSICAL_ORDER


class ConfirmationRefusal(RuntimeError):
    def __init__(self, code: str, details: dict[str, Any] | None = None):
        super().__init__(code)
        self.code = code
        self.details = details or {}


@dataclass(frozen=True, order=True)
class PrimaryConfig:
    heterogeneity_penalty: float
    ridge_penalty: float
    graph_penalty: float
    transport_multiplier: float


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json_x(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _append_jsonl(path: Path, payload: dict[str, Any], *, create: bool = False) -> None:
    flags = os.O_WRONLY | os.O_APPEND
    if create:
        flags |= os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "a") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _private(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        return resolved
    raise PermissionError("private cell-level artifacts must remain outside the repository")


def _runtime() -> dict[str, str]:
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": ".".join(str(value) for value in sys.version_info[:3]),
        "numpy_version": np.__version__,
        "scipy_version": __import__("scipy").__version__,
    }


def _require_runtime() -> None:
    expected = _read_json(RUNTIME)
    observed = _runtime()
    for key, value in observed.items():
        if expected.get(key) != value:
            raise PermissionError(f"runtime differs at {key}: {value}")
    for key, value in expected["thread_environment"].items():
        if os.environ.get(key) != value:
            raise PermissionError(f"thread environment differs at {key}")


def _remote_tag_commit(tag: str) -> str:
    output = subprocess.run(
        ["git", "ls-remote", "--tags", PUBLIC_ORIGIN, f"refs/tags/{tag}",
         f"refs/tags/{tag}^{{}}"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    commits = [line.split()[0] for line in output if line.strip()]
    if not commits:
        raise PermissionError(f"required public tag is absent: {tag}")
    return commits[-1]


def _require_public_tag(tag: str, paths: Iterable[Path]) -> str:
    commit = _remote_tag_commit(tag)
    for path in paths:
        relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
        published = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=ROOT, check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(published).hexdigest() != _sha256(path):
            raise PermissionError(f"public tag does not bind local bytes: {relative}")
    return commit


def _require_ancestor(ancestor: str, descendant: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise PermissionError("public stage ancestry differs")


def _stage_prerequisites(stage: str) -> tuple[tuple[str, tuple[Path, ...]], ...]:
    if stage == "source":
        return ()
    if stage == "rna":
        return ((COMPLETION_TAGS["source"], (SOURCE_RESULT, JOURNALS["source"])),)
    if stage == "prediction":
        return (
            (COMPLETION_TAGS["source"], (SOURCE_RESULT, JOURNALS["source"])),
            (COMPLETION_TAGS["rna"], (RNA_RESULT, JOURNALS["rna"])),
        )
    if stage == "adt":
        return (
            (COMPLETION_TAGS["source"], (SOURCE_RESULT, JOURNALS["source"])),
            (COMPLETION_TAGS["rna"], (RNA_RESULT, JOURNALS["rna"])),
            (COMPLETION_TAGS["prediction"], (PREDICTION_RESULT,)),
        )
    if stage == "score":
        return (
            (COMPLETION_TAGS["source"], (SOURCE_RESULT, JOURNALS["source"])),
            (COMPLETION_TAGS["rna"], (RNA_RESULT, JOURNALS["rna"])),
            (COMPLETION_TAGS["prediction"], (PREDICTION_RESULT,)),
            (COMPLETION_TAGS["adt"], (ADT_RESULT, JOURNALS["adt"])),
            (SCORE_AUTHORIZATION_TAG, (SCORE_AUTHORIZATION,)),
        )
    raise ValueError(stage)


def _require_stage_prerequisites(stage: str) -> tuple[str, ...]:
    protocol_commit = _remote_tag_commit(PROTOCOL_TAG)
    commits = []
    previous = protocol_commit
    for tag, paths in _stage_prerequisites(stage):
        commit = _require_public_tag(tag, paths)
        _require_ancestor(previous, commit)
        previous = commit
        commits.append(commit)
    status_requirements = {
        "source": (),
        "rna": ((SOURCE_RESULT, "SOURCE_PASS"),),
        "prediction": (
            (SOURCE_RESULT, "SOURCE_PASS"),
            (RNA_RESULT, "HELD_RNA_PASS"),
        ),
        "adt": (
            (SOURCE_RESULT, "SOURCE_PASS"),
            (RNA_RESULT, "HELD_RNA_PASS"),
            (PREDICTION_RESULT, "PREDICTIONS_FROZEN"),
        ),
        "score": (
            (SOURCE_RESULT, "SOURCE_PASS"),
            (RNA_RESULT, "HELD_RNA_PASS"),
            (PREDICTION_RESULT, "PREDICTIONS_FROZEN"),
            (ADT_RESULT, "HELD_ADT_PASS"),
            (SCORE_AUTHORIZATION, "SCORE_AUTHORIZED"),
        ),
    }
    for path, expected in status_requirements[stage]:
        if _read_json(path).get("status") != expected:
            raise PermissionError("public predecessor status did not authorize stage")
    return tuple(commits)


def claim_stage(stage: str) -> dict[str, Any]:
    if stage not in ATTEMPTS:
        raise ValueError(f"unknown stage: {stage}")
    _require_public_tag(PROTOCOL_TAG, PROTOCOL_PATHS)
    prerequisites = _require_stage_prerequisites(stage)
    payload = {
        "schema": "gse144744-ms-stage-attempt/1.0",
        "stage": stage,
        "status": "STARTED",
        "created_at_utc": _timestamp(),
        "protocol_tag": PROTOCOL_TAG,
        "protocol_commit": _remote_tag_commit(PROTOCOL_TAG),
        "prerequisite_commits": list(prerequisites),
    }
    _write_json_x(ATTEMPTS[stage], payload)
    if stage in JOURNALS:
        _append_jsonl(
            JOURNALS[stage],
            {"schema": "gse144744-ms-access/1.0", "stage": stage,
             "status": "OPENED_BEFORE_ASSAY_ACCESS", "created_at_utc": _timestamp()},
            create=True,
        )
    return payload


def _require_attempt(stage: str) -> None:
    attempt = _read_json(ATTEMPTS[stage])
    if attempt.get("stage") != stage or attempt.get("status") != "STARTED":
        raise PermissionError("stage attempt is invalid")
    paths = PROTOCOL_PATHS + (ATTEMPTS[stage],)
    if stage in JOURNALS:
        paths += (JOURNALS[stage],)
    attempt_commit = _require_public_tag(ATTEMPT_TAGS[stage], paths)
    protocol_commit = _remote_tag_commit(PROTOCOL_TAG)
    _require_ancestor(protocol_commit, attempt_commit)
    for commit in _require_stage_prerequisites(stage):
        _require_ancestor(commit, attempt_commit)


def _download(url: str, path: Path, expected_bytes: int, stage: str) -> Path:
    if path.exists():
        raise FileExistsError(f"fresh download path already exists: {path}")
    request = urllib.request.Request(url, headers={"User-Agent": "coupling-fields/1"})
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with urllib.request.urlopen(request, timeout=120) as response, path.open("xb") as out:
        while True:
            block = response.read(8 << 20)
            if not block:
                break
            out.write(block)
    observed = path.stat().st_size
    digest = _sha256(path)
    _append_jsonl(
        JOURNALS[stage],
        {"created_at_utc": _timestamp(), "status": "DOWNLOADED_AND_HASHED",
         "url": url, "bytes": observed, "sha256": digest},
    )
    if observed != expected_bytes:
        raise ConfirmationRefusal("OFFICIAL_FILE_BYTE_COUNT_MISMATCH")
    return path


def _manifest_file(name: str) -> dict[str, Any]:
    for record in _read_json(MANIFEST)["files"]:
        if record["name"] == name:
            return record
    raise KeyError(name)


def _markers() -> list[dict[str, str]]:
    rows = _read_json(CANDIDATE)["candidate_marker_pairs"]
    if (
        len(rows) != CANDIDATE_MARKER_COUNT
        or len({row["rna"] for row in rows}) != CANDIDATE_MARKER_COUNT
        or len({row["adt"] for row in rows}) != CANDIDATE_MARKER_COUNT
    ):
        raise PermissionError("candidate marker axis differs")
    return rows


def _metadata(path: Path) -> dict[str, dict[str, str]]:
    required = {
        "cell_names", "donor", "sample_10X", "match", "batch_pair", "group",
        "cohort", "V_10X", "exp_name", "nCount_RNA", "nCount_ADT", "basictype",
        "HASH",
    }
    output: dict[str, dict[str, str]] = {}
    with gzip.open(path, "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise PermissionError("per-cell metadata schema differs")
        for row in reader:
            cell = row["cell_names"]
            if not cell or cell in output:
                raise PermissionError("per-cell metadata has a duplicate cell identifier")
            output[cell] = row
    if len(output) != 497_705:
        raise PermissionError("per-cell metadata row count differs")
    return output


def _cohort_contract(row: dict[str, str], role: str) -> bool:
    if role == "source":
        return (
            row["donor"] in SOURCE_DONORS and row["cohort"] == "RRMS_HI"
            and row["V_10X"] == "V2" and row["group"] in {"MS2", "HI2"}
        )
    if role == "held":
        return (
            row["donor"] in HELD_DONORS and row["cohort"] == "PPMS_HI"
            and row["V_10X"] == "V3" and row["group"] in {"PPMS", "HI3"}
        )
    raise ValueError(role)


def _selected_cells(
    metadata: dict[str, dict[str, str]], role: str
) -> dict[str, list[str]]:
    donors = SOURCE_DONORS if role == "source" else HELD_DONORS
    libraries = _selected_libraries(metadata, role)
    pools = {donor: [] for donor in donors}
    for cell, row in metadata.items():
        if (
            row["nCount_ADT"] != "NA"
            and _cohort_contract(row, role)
            and row["sample_10X"] == libraries[row["donor"]]
        ):
            pools[row["donor"]].append(cell)
    output = {}
    for donor in donors:
        if len(pools[donor]) < CELL_BUDGET:
            raise ConfirmationRefusal(f"{role.upper()}_DONOR_BELOW_CELL_BUDGET")
        output[donor] = sorted(
            pools[donor],
            key=lambda cell: (
                hashlib.sha256(f"{CELL_SALT}|{donor}|{cell}".encode()).hexdigest(),
                cell,
            ),
        )[:CELL_BUDGET]
    return output


def _sample_suffix(sample: str, donor: str) -> int:
    prefix = f"{donor}_"
    if not sample.startswith(prefix) or not sample[len(prefix):].isdigit():
        raise PermissionError("sample_10X identifier differs")
    return int(sample[len(prefix):])


def _selected_libraries(
    metadata: dict[str, dict[str, str]], role: str
) -> dict[str, str]:
    donors = SOURCE_DONORS if role == "source" else HELD_DONORS
    counts: dict[str, dict[int, int]] = {donor: {} for donor in donors}
    hashes: dict[str, dict[int, set[str]]] = {donor: {} for donor in donors}
    for row in metadata.values():
        if row["nCount_ADT"] == "NA" or not _cohort_contract(row, role):
            continue
        suffix = _sample_suffix(row["sample_10X"], row["donor"])
        donor_counts = counts[row["donor"]]
        donor_counts[suffix] = donor_counts.get(suffix, 0) + 1
        hashes[row["donor"]].setdefault(suffix, set()).add(row["HASH"])
    if role == "source":
        selected = {}
        for donor in donors:
            eligible = [suffix for suffix, size in counts[donor].items()
                        if size >= CELL_BUDGET]
            if eligible != [1]:
                raise ConfirmationRefusal("SOURCE_LIBRARY_CONTRACT_DIFFERS", {
                    "donor": donor, "eligible_suffixes": sorted(eligible)
                })
            selected[donor] = f"{donor}_1"
        frozen = _read_json(CANDIDATE)["source"]["selected_libraries"]
        if list(selected.values()) != frozen:
            raise ConfirmationRefusal("SOURCE_LIBRARY_SELECTION_DIFFERS")
        return selected

    expected = _read_json(CANDIDATE)["held"]["selected_libraries"]
    expected_by_pair = {
        tuple(row["pair"]): (
            int(row["suffix"]), row["eligible_counts"], row["hash_pool"]
        )
        for row in expected
    }
    if set(expected_by_pair) != set(HELD_PAIRS) or len(expected_by_pair) != len(expected):
        raise PermissionError("frozen held library map differs")
    selected = {}
    for left, right in HELD_PAIRS:
        common = sorted(
            suffix for suffix in set(counts[left]) & set(counts[right])
            if counts[left][suffix] >= CELL_BUDGET
            and counts[right][suffix] >= CELL_BUDGET
        )
        if not common:
            raise ConfirmationRefusal("HELD_PAIR_HAS_NO_COMMON_ELIGIBLE_LIBRARY", {
                "pair": [left, right]
            })
        suffix = min(common, key=lambda value: (
            hashlib.sha256(
                f"{LIBRARY_SALT}|{left}-{right}|{value}".encode()
            ).hexdigest(),
            value,
        ))
        frozen_suffix, frozen_counts, frozen_hash = expected_by_pair[(left, right)]
        observed_counts = [counts[left][suffix], counts[right][suffix]]
        observed_hashes = [hashes[left][suffix], hashes[right][suffix]]
        if (
            suffix != frozen_suffix
            or observed_counts != frozen_counts
            or observed_hashes != [{frozen_hash}, {frozen_hash}]
        ):
            raise ConfirmationRefusal("HELD_LIBRARY_SELECTION_DIFFERS", {
                "pair": [left, right], "observed_suffix": suffix,
                "observed_counts": observed_counts,
                "observed_hashes": [sorted(value) for value in observed_hashes],
            })
        selected[left], selected[right] = f"{left}_{suffix}", f"{right}_{suffix}"
    return selected


def _validate_pairs(metadata: dict[str, dict[str, str]]) -> None:
    representative: dict[str, dict[str, str]] = {}
    fields = ("match", "batch_pair", "group", "cohort", "V_10X", "exp_name")
    for donor in SOURCE_DONORS + HELD_DONORS:
        contracts = {
            tuple(row[field] for field in fields)
            for row in metadata.values() if row["donor"] == donor
        }
        if len(contracts) != 1:
            raise PermissionError("donor metadata contract differs across cells")
        values = next(iter(contracts))
        representative[donor] = dict(zip(fields, values))
    pairs = SOURCE_PAIRS + HELD_PAIRS
    for odd, even in pairs:
        left, right = representative[odd], representative[even]
        if left["match"] != even or right["match"] != odd:
            raise PermissionError("deposited match relation differs")
        if left["batch_pair"] != right["batch_pair"]:
            raise PermissionError("deposited pair batch differs")
        if left["exp_name"] != right["exp_name"]:
            raise PermissionError("matched donors span experiments")
        if (odd, even) in SOURCE_EXPERIMENT_BY_PAIR:
            if left["exp_name"] != SOURCE_EXPERIMENT_BY_PAIR[(odd, even)]:
                raise PermissionError("source experiment allocation differs")
            if left["group"] != "MS2" or right["group"] != "HI2":
                raise PermissionError("source matched group allocation differs")
        else:
            if left["group"] != "PPMS" or right["group"] != "HI3":
                raise PermissionError("held matched group allocation differs")
            if left["exp_name"] not in {"HE-MK-015", "HE-MK-016", "HE-MK-017", "HE-MK-018"}:
                raise PermissionError("held experiment allocation differs")


def _resolve_rows(features: tuple[tuple[str, ...], ...], names: list[str]) -> list[int]:
    output = []
    for name in names:
        matches = [index for index, row in enumerate(features) if name in row]
        if len(matches) != 1:
            raise ConfirmationRefusal("FEATURE_DOES_NOT_RESOLVE_EXACTLY_ONCE", {"feature": name})
        output.append(matches[0])
    return output


def _adt_states(counts: np.ndarray, cells: list[str], donor: str) -> np.ndarray:
    values = np.asarray(counts, dtype=np.int64)
    if values.shape[0] != CELL_BUDGET or len(cells) != CELL_BUDGET:
        raise ValueError("ADT counts differ from cell budget")
    states = np.zeros(values.shape, dtype=np.uint8)
    for marker in range(values.shape[1]):
        order = sorted(
            range(CELL_BUDGET),
            key=lambda index: (
                values[index, marker],
                hashlib.sha256(
                    f"{ADT_SALT}|{donor}|{marker}|{cells[index]}".encode()
                ).hexdigest(),
                cells[index],
            ),
        )
        states[np.asarray(order[CELL_BUDGET // 2 :]), marker] = 1
    return states


def _rna_quality(counts: np.ndarray) -> dict[str, np.ndarray]:
    positive = np.sum(np.asarray(counts) > 0, axis=0)
    prevalence = positive / CELL_BUDGET
    return {
        "prevalence": prevalence,
        "positive_cells": positive,
        "valid": (
            (positive >= MINIMUM_BINARY_MARGIN)
            & (CELL_BUDGET - positive >= MINIMUM_BINARY_MARGIN)
        ),
    }


def _adt_quality(counts: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(counts)
    distinct = np.empty(values.shape[1], dtype=int)
    largest = np.empty(values.shape[1], dtype=float)
    nonzero = np.mean(values > 0, axis=0)
    interquartile_range = np.empty(values.shape[1], dtype=float)
    for marker in range(values.shape[1]):
        _, frequencies = np.unique(values[:, marker], return_counts=True)
        distinct[marker] = len(frequencies)
        largest[marker] = frequencies.max() / CELL_BUDGET
        quartiles = np.quantile(values[:, marker], [0.25, 0.75])
        interquartile_range[marker] = quartiles[1] - quartiles[0]
    return {
        "distinct": distinct,
        "largest_equal_fraction": largest,
        "nonzero_fraction": nonzero,
        "interquartile_range": interquartile_range,
        "valid": (
            (nonzero >= MINIMUM_ADT_NONZERO_FRACTION)
            & (interquartile_range > 0)
            & (largest <= MAXIMUM_ADT_EQUAL_VALUE_FRACTION)
        ),
    }


def _destroyed(states: np.ndarray, cells: list[str], donor: str) -> np.ndarray:
    order = np.asarray(sorted(
        range(CELL_BUDGET),
        key=lambda index: (
            hashlib.sha256(f"{DESTROY_SALT}|{donor}|{cells[index]}".encode()).hexdigest(),
            cells[index],
        ),
    ))
    output = np.empty_like(states)
    output[order] = states[np.roll(order, 1)]
    if not np.array_equal(output.sum(axis=0), states.sum(axis=0)):
        raise AssertionError("destroyed link changed ADT margins")
    return output


def _binary_tables(rna: np.ndarray, adt: np.ndarray) -> np.ndarray:
    first, second = np.asarray(rna, dtype=np.uint8), np.asarray(adt, dtype=np.uint8)
    if first.shape != second.shape or first.shape[0] != CELL_BUDGET:
        raise ValueError("binary state panels differ")
    size = first.shape[1]
    output = np.empty((size, size, 2, 2), dtype=np.int64)
    for row in range(size):
        for column in range(size):
            output[row, column] = np.bincount(
                2 * first[:, row] + second[:, column], minlength=4
            ).reshape(2, 2)
    return output


def _informative(tables: np.ndarray) -> np.ndarray:
    values = np.asarray(tables)
    rows, columns = values.sum(axis=-1), values.sum(axis=-2)
    total = values.sum(axis=(-2, -1))
    return np.minimum(rows[..., 0], columns[..., 0]) > np.maximum(
        0, rows[..., 0] + columns[..., 0] - total
    )


def _margins(tables: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(tables)
    return values.sum(axis=-1), values.sum(axis=-2)


def _knn_incidence(profiles: np.ndarray, minimum_markers: int = 2) -> np.ndarray:
    values = np.asarray(profiles, dtype=float).T
    if values.ndim != 2 or values.shape[0] < minimum_markers or values.shape[1] < 3:
        raise ValueError("marker profile dimensions differ")
    scale = values.std(axis=1, ddof=1)
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise CouplingEstimationRefusal("marker profile has zero variance")
    standardized = (values - values.mean(axis=1, keepdims=True)) / scale[:, None]
    edges: set[tuple[int, int]] = set()
    for marker in range(values.shape[0]):
        candidates = np.asarray([i for i in range(values.shape[0]) if i != marker])
        distances = np.linalg.norm(
            standardized[candidates] - standardized[marker][None, :], axis=1
        )
        order = candidates[np.lexsort((candidates, distances))]
        edges.update(tuple(sorted((marker, int(neighbor))))
                     for neighbor in order[:GRAPH_NEIGHBORS])
    incidence = np.zeros((values.shape[0], len(edges)))
    for column, (left, right) in enumerate(sorted(edges)):
        incidence[left, column] = incidence[right, column] = 1
    return incidence


def _fit_primary(
    tables: np.ndarray, rna_profiles: np.ndarray, adt_profiles: np.ndarray,
    config: PrimaryConfig,
) -> dict[str, Any]:
    size = tables.shape[1]
    if config.graph_penalty == 0:
        first = second = np.eye(size)
    else:
        first, second = _knn_incidence(rna_profiles), _knn_incidence(adt_profiles)
    fit = fit_hierarchical_conditional_log_odds(
        tables, first, second,
        heterogeneity_penalty=config.heterogeneity_penalty,
        ridge_penalty=config.ridge_penalty,
        graph_penalty=config.graph_penalty,
        minimum_informative_donors=2,
        maximum_condition_number=MAXIMUM_CONDITION_NUMBER,
    )
    return {
        "family": "graph_regularized_exact_fixed_margin_hierarchical_coupling",
        "configuration": asdict(config),
        "transport_multiplier": config.transport_multiplier,
        "population_log_odds": fit.population_log_odds,
        "support_count": fit.support_count,
        "certificate": {
            "scaled_gradient_norm": fit.scaled_gradient_norm,
            "schur_condition_number": fit.schur_condition_number,
            "theta_curvature_condition_number": fit.theta_curvature_condition_number,
            "iterations": fit.iterations,
            "rna_incidence_sha256": _array_sha256(first),
            "adt_incidence_sha256": _array_sha256(second),
        },
    }


def _signed_deviance(table: np.ndarray) -> float:
    values = np.asarray(table, dtype=float)
    residual = poisson_independence_residuals(values, residual="deviance")
    determinant = values[0, 0] * values[1, 1] - values[0, 1] * values[1, 0]
    return float(np.sign(determinant) * np.linalg.norm(residual))


def _fractional_deviance(table: np.ndarray) -> float:
    values = np.asarray(table, dtype=float)
    expected = np.outer(values.sum(axis=1), values.sum(axis=0)) / values.sum()
    positive = values > 0
    deviance = 2 * np.sum(values[positive] * np.log(values[positive] / expected[positive]))
    determinant = values[0, 0] * values[1, 1] - values[0, 1] * values[1, 0]
    return float(np.sign(determinant) * math.sqrt(max(float(deviance), 0)))


def _classical_table(coordinate: float, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    row, column = np.asarray(rows, dtype=float), np.asarray(columns, dtype=float)
    total = row.sum()
    lower, upper = max(0.0, row[0] + column[0] - total), min(row[0], column[0])

    def at(value: float) -> np.ndarray:
        return np.asarray([[value, row[0] - value],
                           [column[0] - value, row[1] - column[0] + value]])

    if upper <= lower:
        return at(lower)
    epsilon = min(1e-10, 0.25 * (upper - lower))
    left, right = lower + epsilon, upper - epsilon
    target = min(max(coordinate, _fractional_deviance(at(left))),
                 _fractional_deviance(at(right)))
    for _ in range(96):
        middle = (left + right) / 2
        if _fractional_deviance(at(middle)) < target:
            left = middle
        else:
            right = middle
    return at((left + right) / 2)


def _residual_pool(tables: np.ndarray) -> np.ndarray:
    support = _informative(tables)
    coordinates = np.full(support.shape, np.nan)
    for donor, row, column in np.argwhere(support):
        coordinates[donor, row, column] = _signed_deviance(
            tables[donor, row, column]
        ) / math.sqrt(CELL_BUDGET)
    if np.any(support.sum(axis=0) < 2):
        raise CouplingEstimationRefusal("Poisson residual lacks two donors")
    return np.nanmean(coordinates, axis=0)


def _fit_classical(method: str, tables: np.ndarray) -> dict[str, Any]:
    values = np.asarray(tables, dtype=np.int64)
    size = values.shape[1]
    if method in {RAW_RESIDUAL, CALIBRATED_RESIDUAL}:
        return {"family": "poisson_independence_signed_deviance_residual",
                "pooled_coordinate": _residual_pool(values),
                "transport_multiplier": 1.0}
    if method == "common_effect_stratified_cmle":
        fit = fit_structured_conditional_log_odds(
            values, np.eye(size), np.eye(size), initial_log_odds=np.zeros((size, size)),
            ridge_penalty=0, graph_penalty=0, minimum_informative_donors=2,
            maximum_condition_number=MAXIMUM_CONDITION_NUMBER, tolerance=1e-9,
        )
        return {"family": method, "population_log_odds": fit.log_odds,
                "transport_multiplier": 1.0}
    if method == "pooled_saturated_poisson":
        pooled = values.sum(axis=0)
        if np.any(pooled <= 0):
            raise CouplingEstimationRefusal("pooled Poisson table has zero cell")
        log_odds = (np.log(pooled[..., 0, 0]) + np.log(pooled[..., 1, 1])
                    - np.log(pooled[..., 0, 1]) - np.log(pooled[..., 1, 0]))
        return {"family": method, "population_log_odds": log_odds,
                "transport_multiplier": 1.0}
    if method == "paule_mandel_random_effects_log_odds":
        coordinates = np.zeros(values.shape[:3])
        variances = np.zeros_like(coordinates)
        support = np.zeros(values.shape[:3], dtype=bool)
        for index in np.ndindex(values.shape[:3]):
            estimate = centered_haldane_log_odds(values[index])
            coordinates[index] = estimate.observed_log_odds
            variances[index] = estimate.sampling_variance
            support[index] = estimate.supported
        pooled = paule_mandel_pool(coordinates, variances, support=support,
                                   minimum_donors=2)
        if not pooled.supported.all():
            raise CouplingEstimationRefusal("Paule-Mandel lacks support")
        return {"family": method, "population_log_odds": pooled.mean,
                "transport_multiplier": 1.0, "tau_squared": pooled.tau_squared}
    raise ValueError(method)


def _predict_model(model: dict[str, Any], rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    if model.get("family") == "target_margin_independence":
        total = rows.sum(axis=-1)
        return rows[..., :, None] * columns[..., None, :] / total[..., None, None]
    multiplier = float(model.get("transport_multiplier", 1.0))
    if "pooled_coordinate" in model:
        coordinate = np.asarray(model["pooled_coordinate"])
        output = np.empty((*coordinate.shape, 2, 2))
        for index in np.ndindex(coordinate.shape):
            output[index] = _classical_table(
                multiplier * coordinate[index] * math.sqrt(CELL_BUDGET),
                rows[index], columns[index],
            )
        return output
    field = np.asarray(model["population_log_odds"])
    output = np.empty((*field.shape, 2, 2))
    for index in np.ndindex(field.shape):
        output[index] = expected_binary_table_from_log_odds(
            multiplier * field[index], rows[index], columns[index]
        )
    return output


def _donor_loss(truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    observed, estimate, support = np.asarray(truth, float), np.asarray(prediction, float), np.asarray(mask, bool)
    if not np.any(support) or np.any(support & ~_informative(observed)):
        raise PermissionError("invalid scoring mask")
    if not np.allclose(observed.sum(axis=-1), estimate.sum(axis=-1)) or not np.allclose(
        observed.sum(axis=-2), estimate.sum(axis=-2)
    ):
        raise PermissionError("prediction changed target margins")
    observed, estimate = observed[support], estimate[support]
    positive = observed > 0
    if np.any(estimate[positive] <= 0):
        raise FloatingPointError("prediction assigns zero mass to truth")
    terms = np.zeros_like(observed)
    terms[positive] = observed[positive] * np.log(observed[positive] / estimate[positive])
    return float((2 * terms.sum(axis=(-2, -1)) / CELL_BUDGET).mean())


def _json_model(model: dict[str, Any]) -> dict[str, Any]:
    return {key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in model.items()}


def _model_losses(records: dict[str, dict[str, Any]], donors: Iterable[str],
                  model: dict[str, Any]) -> dict[str, float]:
    output = {}
    for donor in donors:
        truth, mask = records[donor]["tables"], records[donor]["informative"]
        rows, columns = _margins(truth)
        output[donor] = _donor_loss(truth, _predict_model(model, rows, columns), mask)
    return output


def _source_folds(records: dict[str, dict[str, Any]]) -> dict[str, tuple[str, ...]]:
    folds = {
        experiment: tuple(
            donor for donor in SOURCE_DONORS
            if records[donor]["exp_name"] == experiment
        )
        for experiment in SOURCE_EXPERIMENTS
    }
    if any(not donors for donors in folds.values()):
        raise ConfirmationRefusal("SOURCE_EXPERIMENT_FOLD_IS_EMPTY")
    if set(itertools.chain.from_iterable(folds.values())) != set(SOURCE_DONORS):
        raise PermissionError("source experiment allocation differs")
    return folds


def _experiment_equal_mean(
    losses: dict[str, float], folds: dict[str, tuple[str, ...]]
) -> float:
    return float(np.mean([
        np.mean([losses[donor] for donor in folds[experiment]])
        for experiment in SOURCE_EXPERIMENTS
    ]))


def _masked_tables(
    records: dict[str, dict[str, Any]], donors: Iterable[str],
    *, key: str = "tables",
) -> np.ndarray:
    output = []
    for donor in tuple(donors):
        tables = np.asarray(records[donor][key]).copy()
        tables[~np.asarray(records[donor]["informative"], dtype=bool)] = 0
        output.append(tables)
    return np.asarray(output)


def _training_profiles(
    records: dict[str, dict[str, Any]], donors: Iterable[str], modality: str
) -> np.ndarray:
    axis = tuple(donors)
    profiles = np.asarray([records[d][f"{modality}_profile"] for d in axis], float)
    valid = np.asarray([records[d][f"{modality}_valid"] for d in axis], bool)
    for marker in range(profiles.shape[1]):
        supported = valid[:, marker]
        if int(supported.sum()) < 2:
            raise CouplingEstimationRefusal(
                f"{modality} marker has fewer than two valid training donors"
            )
        profiles[~supported, marker] = profiles[supported, marker].mean()
    return profiles


def _arrays(
    records: dict[str, dict[str, Any]], donors: Iterable[str]
) -> tuple[np.ndarray, ...]:
    axis = tuple(donors)
    return (
        _masked_tables(records, axis),
        _training_profiles(records, axis, "rna"),
        _training_profiles(records, axis, "adt"),
    )


def _select_primary(
    records: dict[str, dict[str, Any]], graph_grid: tuple[float, ...]
) -> dict[str, Any]:
    folds = _source_folds(records)
    choices = []
    for heterogeneity, ridge, graph in itertools.product(
        HETEROGENEITY_GRID, RIDGE_GRID, graph_grid
    ):
        fold_models = {}
        refused = False
        for experiment, validation in folds.items():
            training = tuple(donor for donor in SOURCE_DONORS if donor not in validation)
            try:
                tables, rna, adt = _arrays(records, training)
                fold_models[experiment] = _fit_primary(
                    tables, rna, adt,
                    PrimaryConfig(heterogeneity, ridge, graph, 1.0),
                )
            except (ValueError, FloatingPointError, CouplingEstimationRefusal):
                refused = True
                break
        if refused:
            continue
        for transport in TRANSPORT_GRID:
            config = PrimaryConfig(heterogeneity, ridge, graph, transport)
            losses = {}
            for experiment, validation in folds.items():
                model = dict(fold_models[experiment])
                model["configuration"] = asdict(config)
                model["transport_multiplier"] = transport
                losses.update(_model_losses(records, validation, model))
            choices.append((
                _experiment_equal_mean(losses, folds), config, losses
            ))
    if not choices:
        raise ConfirmationRefusal("PRIMARY_SOURCE_CV_HAS_NO_CONFIGURATION")
    loss, config, losses = min(choices, key=lambda row: (row[0], row[1]))
    return {
        "experiment_equal_mean": loss,
        "configuration": config,
        "losses": losses,
        "complete_candidates": len(choices),
    }


def _select_classical(
    records: dict[str, dict[str, Any]], method: str
) -> dict[str, Any]:
    folds = _source_folds(records)
    fold_models = {}
    for experiment, validation in folds.items():
        training = tuple(donor for donor in SOURCE_DONORS if donor not in validation)
        fold_models[experiment] = _fit_classical(
            method, _masked_tables(records, training)
        )
    transports = (1.0,) if method == RAW_RESIDUAL else TRANSPORT_GRID
    choices = []
    for transport in transports:
        losses = {}
        for experiment, validation in folds.items():
            model = dict(fold_models[experiment])
            model["transport_multiplier"] = transport
            losses.update(_model_losses(records, validation, model))
        choices.append((
            _experiment_equal_mean(losses, folds), transport, losses
        ))
    loss, transport, losses = min(choices, key=lambda row: (row[0], row[1]))
    return {
        "experiment_equal_mean": loss,
        "transport_multiplier": transport,
        "losses": losses,
    }


def _source_gate(
    records: dict[str, dict[str, Any]], primary: dict[str, float],
    comparator: dict[str, float],
) -> dict[str, Any]:
    folds = _source_folds(records)
    first = _experiment_equal_mean(primary, folds)
    second = _experiment_equal_mean(comparator, folds)
    differences = {donor: primary[donor] - comparator[donor]
                   for donor in SOURCE_DONORS}
    experiment_differences = {
        experiment: float(np.mean([differences[d] for d in folds[experiment]]))
        for experiment in SOURCE_EXPERIMENTS
    }
    relative = 1 - first / second
    favorable = sum(value < 0 for value in differences.values())
    passes = (
        relative >= 0.05
        and favorable >= MINIMUM_FAVORABLE_SOURCE_DONORS
        and all(value < 0 for value in experiment_differences.values())
    )
    return {
        "primary_loss": first,
        "comparator_loss": second,
        "relative_reduction": relative,
        "favorable_donors": favorable,
        "experiment_mean_differences": experiment_differences,
        "donor_differences": differences,
        "passes": passes,
    }


def _source_tuning(
    records: dict[str, dict[str, Any]], markers: list[dict[str, str]]
) -> dict[str, Any]:
    primary = _select_primary(records, GRAPH_GRID)
    graph_zero = _select_primary(records, (0.0,))
    classical: dict[str, Any] = {}
    refusals = {}
    for method in ALL_CLASSICAL:
        try:
            classical[method] = _select_classical(records, method)
        except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
            refusals[method] = {"exception": type(error).__name__, "reason": str(error)}
    if RAW_RESIDUAL not in classical:
        raise ConfirmationRefusal("UNTUNED_POISSON_SOURCE_REFUSED", refusals)
    if POOLED_POISSON not in classical:
        raise ConfirmationRefusal("POOLED_LOG_LINEAR_SOURCE_REFUSED", refusals)
    estimable = [method for method in CLASSICAL_ORDER if method in classical]
    if not estimable:
        raise ConfirmationRefusal("NO_CALIBRATED_CLASSICAL_IS_ESTIMABLE", refusals)
    locked_method = min(estimable, key=lambda method: (
        classical[method]["experiment_equal_mean"], CLASSICAL_ORDER.index(method)
    ))
    locked_gate = _source_gate(
        records, primary["losses"], classical[locked_method]["losses"]
    )
    raw_gate = _source_gate(
        records, primary["losses"], classical[RAW_RESIDUAL]["losses"]
    )
    pooled_gate = _source_gate(
        records, primary["losses"], classical[POOLED_POISSON]["losses"]
    )
    if not locked_gate["passes"] or not raw_gate["passes"] or not pooled_gate["passes"]:
        raise ConfirmationRefusal("SOURCE_PRIMARY_DID_NOT_PASS_CLASSICAL_GATES", {
            "selected_primary": asdict(primary["configuration"]),
            "locked_classical": locked_method, "locked_gate": locked_gate,
            "raw_poisson_gate": raw_gate, "pooled_log_linear_gate": pooled_gate,
        })
    tables, rna, adt = _arrays(records, SOURCE_DONORS)
    selected: PrimaryConfig = primary["configuration"]
    zero_selected: PrimaryConfig = graph_zero["configuration"]
    final_primary = _fit_primary(tables, rna, adt, selected)
    final_zero = _fit_primary(tables, rna, adt, zero_selected)
    destroyed = _masked_tables(records, SOURCE_DONORS, key="destroyed_tables")
    final_destroyed = _fit_primary(destroyed, rna, adt, selected)
    models = {
        "primary": _json_model(final_primary),
        "graph_zero": _json_model(final_zero),
        "destroyed_link": _json_model(final_destroyed),
    }
    for method, selection in classical.items():
        final = _fit_classical(method, _masked_tables(records, SOURCE_DONORS))
        final["transport_multiplier"] = selection["transport_multiplier"]
        models[method] = _json_model(final)
    models["independence"] = {"family": "target_margin_independence"}
    return {
        "primary_marker_pairs": markers,
        "cross_validation": "leave_one_source_experiment_out",
        "selected_primary": asdict(selected),
        "selected_graph_zero": asdict(zero_selected),
        "locked_classical_method": locked_method,
        "source_locked_gate": locked_gate,
        "source_raw_poisson_gate": raw_gate,
        "source_pooled_log_linear_gate": pooled_gate,
        "primary_validation_losses": primary["losses"],
        "graph_zero_validation_losses": graph_zero["losses"],
        "classical_validation": {
            method: {key: value for key, value in selection.items() if key != "model"}
            for method, selection in classical.items()
        },
        "classical_refusals": refusals,
        "models": models,
    }


def _records_from_counts(
    selected: dict[str, list[str]], metadata: dict[str, dict[str, str]],
    rna_counts: np.ndarray, adt_counts: np.ndarray, markers: list[dict[str, str]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    donors = tuple(selected)
    expected = len(donors) * CELL_BUDGET
    if rna_counts.shape != (expected, len(markers)) or adt_counts.shape != rna_counts.shape:
        raise ValueError("streamed count panels differ from selected axis")
    raw: dict[str, dict[str, Any]] = {}
    rna_valid, adt_valid = [], []
    for index, donor in enumerate(donors):
        cells = selected[donor]
        slc = slice(index * CELL_BUDGET, (index + 1) * CELL_BUDGET)
        rna, adt = rna_counts[slc], adt_counts[slc]
        rna_q, adt_q = _rna_quality(rna), _adt_quality(adt)
        rna_valid.append(rna_q["valid"])
        adt_valid.append(adt_q["valid"])
        library = np.asarray([float(metadata[cell]["nCount_RNA"]) for cell in cells])
        if np.any(library <= 0) or not np.isfinite(library).all():
            raise ConfirmationRefusal("SOURCE_RNA_LIBRARY_SIZE_INVALID")
        experiments = {metadata[cell]["exp_name"] for cell in cells}
        if len(experiments) != 1 or next(iter(experiments)) not in SOURCE_EXPERIMENTS:
            raise PermissionError("source donor experiment differs")
        raw[donor] = {
            "cells": cells, "rna": rna, "adt": adt,
            "rna_valid": rna_q["valid"], "adt_valid": adt_q["valid"],
            "rna_profile": np.mean(np.log1p(10000 * rna / library[:, None]), axis=0),
            "adt_profile": np.mean(np.log1p(adt), axis=0),
            "exp_name": next(iter(experiments)),
        }
    rna_valid_array, adt_valid_array = np.asarray(rna_valid), np.asarray(adt_valid)
    source_supported = (
        (rna_valid_array.sum(axis=0) >= MINIMUM_VALID_SOURCE_DONORS)
        & (adt_valid_array.sum(axis=0) >= MINIMUM_VALID_SOURCE_DONORS)
    )
    locked_indices = np.flatnonzero(source_supported)
    if len(locked_indices) < MINIMUM_LOCKED_MARKERS:
        raise ConfirmationRefusal("FEWER_THAN_12_SOURCE_LOCKED_MARKERS", {
            "locked_markers": len(locked_indices),
            "rna_valid_counts": rna_valid_array.sum(axis=0).tolist(),
            "adt_valid_counts": adt_valid_array.sum(axis=0).tolist(),
        })
    locked_markers = [markers[index] for index in locked_indices]
    records = {}
    minimum_pairs = math.ceil(MINIMUM_PAIR_FRACTION * len(locked_markers) ** 2)
    for donor in donors:
        record = raw[donor]
        rna = record["rna"][:, locked_indices]
        adt = record["adt"][:, locked_indices]
        rna_state = (rna > 0).astype(np.uint8)
        adt_state = _adt_states(adt, record["cells"], donor)
        tables = _binary_tables(rna_state, adt_state)
        informative = (
            record["rna_valid"][locked_indices, None]
            & record["adt_valid"][None, locked_indices]
        )
        if int(informative.sum()) < minimum_pairs:
            raise ConfirmationRefusal("SOURCE_LOCKED_MAP_IS_NOT_FULLY_SUPPORTED", {
                "donor": donor, "informative_pairs": int(informative.sum()),
                "minimum_pairs": minimum_pairs,
            })
        records[donor] = {
            "tables": tables,
            "destroyed_tables": _binary_tables(
                rna_state, _destroyed(adt_state, record["cells"], donor)
            ),
            "informative": informative,
            "rna_profile": record["rna_profile"][locked_indices],
            "adt_profile": record["adt_profile"][locked_indices],
            "rna_valid": record["rna_valid"][locked_indices],
            "adt_valid": record["adt_valid"][locked_indices],
            "exp_name": record["exp_name"],
        }
    support = {
        "candidate_marker_pairs": len(markers),
        "locked_marker_pairs": len(locked_markers),
        "locked_candidate_indices": locked_indices.tolist(),
        "rna_valid_source_donors": rna_valid_array.sum(axis=0).tolist(),
        "adt_valid_source_donors": adt_valid_array.sum(axis=0).tolist(),
        "minimum_valid_source_donors": MINIMUM_VALID_SOURCE_DONORS,
        "minimum_informative_pairs_per_donor": minimum_pairs,
    }
    return records, locked_markers, support


def _axis_and_counts(
    archive: Path, matrix_member: str, genes_member: str, barcodes_member: str,
    requested_names: list[str], ordered_cells: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    axes = read_tar_axes(archive, matrix_member, genes_member, barcodes_member)
    rows = _resolve_rows(axes.features, requested_names)
    counts, audit = read_tar_matrix_subset(
        archive, axes, matrix_member, dict(zip(requested_names, rows)),
        tuple(ordered_cells),
    )
    access = asdict(audit)
    if (
        access["unauthorized_value_tokens_converted"] != 0
        or access["value_tokens_converted"]
        != access["selected_entries_materialized"]
    ):
        raise PermissionError("matrix conversion firewall audit failed")
    return counts, access


def _stage_axis_and_counts(
    stage: str, assay: str, archive: Path, matrix_member: str,
    genes_member: str, barcodes_member: str, requested_names: list[str],
    ordered_cells: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    if stage not in JOURNALS:
        raise ValueError("matrix access stage differs")
    _append_jsonl(JOURNALS[stage], {
        "created_at_utc": _timestamp(),
        "status": "MATRIX_SUBSET_READ_AUTHORIZED",
        "assay": assay,
        "matrix_member": matrix_member,
        "requested_rows": len(requested_names),
        "authorized_columns": len(ordered_cells),
        "requested_names_sha256": hashlib.sha256(
            "\n".join(requested_names).encode()
        ).hexdigest(),
        "authorized_cells_sha256": hashlib.sha256(
            "\n".join(ordered_cells).encode()
        ).hexdigest(),
    })
    counts, audit = _axis_and_counts(
        archive, matrix_member, genes_member, barcodes_member,
        requested_names, ordered_cells,
    )
    _append_jsonl(JOURNALS[stage], {
        "created_at_utc": _timestamp(),
        "status": "MATRIX_SUBSET_READ_COMPLETED",
        "assay": assay,
        "access_audit": audit,
    })
    return counts, audit


def source_stage(scratch: Path) -> dict[str, Any]:
    metadata_record = _manifest_file("GSE144744_metadata_per_cell.csv.gz")
    rna_record = _manifest_file("GSE144744_RNA_counts.tar.gz")
    adt_record = _manifest_file("GSE144744_ADT_counts.tar.gz")
    metadata_path = _download(metadata_record["url"], scratch / metadata_record["name"],
                              metadata_record["bytes"], "source")
    rna_path = _download(rna_record["url"], scratch / rna_record["name"],
                         rna_record["bytes"], "source")
    adt_path = _download(adt_record["url"], scratch / adt_record["name"],
                         adt_record["bytes"], "source")
    if _sha256(metadata_path) != metadata_record["sha256"] or _sha256(adt_path) != adt_record["sha256"]:
        raise ConfirmationRefusal("FROZEN_INPUT_SHA256_MISMATCH")
    metadata = _metadata(metadata_path)
    _validate_pairs(metadata)
    selected = _selected_cells(metadata, "source")
    ordered_cells = list(itertools.chain.from_iterable(selected.values()))
    markers = _markers()
    rna_counts, rna_audit = _stage_axis_and_counts(
        "source", "RNA", rna_path, "RNA_counts/matrix.mtx", "RNA_counts/genes.tsv",
        "RNA_counts/barcodes.tsv", [row["rna"] for row in markers], ordered_cells,
    )
    adt_counts, adt_audit = _stage_axis_and_counts(
        "source", "ADT", adt_path, "ADT_counts/matrix.mtx", "ADT_counts/genes.tsv",
        "ADT_counts/barcodes.tsv", [row["adt"] for row in markers], ordered_cells,
    )
    records, fixed_markers, support = _records_from_counts(
        selected, metadata, rna_counts, adt_counts, markers
    )
    tuning = _source_tuning(records, fixed_markers)
    return {
        "schema": "gse144744-ms-source-result/1.0",
        "status": "SOURCE_PASS",
        "created_at_utc": _timestamp(),
        "source_donors": list(SOURCE_DONORS),
        "source_libraries": _selected_libraries(metadata, "source"),
        "support": support,
        "tuning": tuning,
        "input_identity": {
            "rna_bytes": rna_path.stat().st_size, "rna_sha256": _sha256(rna_path),
            "adt_bytes": adt_path.stat().st_size, "adt_sha256": _sha256(adt_path),
            "metadata_sha256": _sha256(metadata_path),
        },
        "access_audit": {"rna": rna_audit, "adt": adt_audit,
                         "held_numeric_values_converted": 0},
    }


def _load_source_models() -> tuple[list[dict[str, str]], dict[str, dict[str, Any]], str]:
    source = _read_json(SOURCE_RESULT)
    if source.get("status") != "SOURCE_PASS":
        raise PermissionError("source stage did not pass")
    tuning = source["tuning"]
    return (
        tuning["primary_marker_pairs"], tuning["models"],
        tuning["locked_classical_method"],
    )


def rna_stage(scratch: Path, private_state: Path) -> dict[str, Any]:
    markers, _, _ = _load_source_models()
    metadata_record = _manifest_file("GSE144744_metadata_per_cell.csv.gz")
    rna_record = _manifest_file("GSE144744_RNA_counts.tar.gz")
    metadata_path = _download(metadata_record["url"], scratch / metadata_record["name"],
                              metadata_record["bytes"], "rna")
    rna_path = _download(rna_record["url"], scratch / rna_record["name"],
                         rna_record["bytes"], "rna")
    if _sha256(metadata_path) != metadata_record["sha256"]:
        raise ConfirmationRefusal("FROZEN_METADATA_SHA256_MISMATCH")
    source = _read_json(SOURCE_RESULT)
    if _sha256(rna_path) != source["input_identity"]["rna_sha256"]:
        raise ConfirmationRefusal("RNA_ARCHIVE_DIFFERS_FROM_SOURCE_STAGE")
    metadata = _metadata(metadata_path)
    _validate_pairs(metadata)
    selected = _selected_cells(metadata, "held")
    ordered_cells = list(itertools.chain.from_iterable(selected.values()))
    counts, audit = _stage_axis_and_counts(
        "rna", "RNA", rna_path, "RNA_counts/matrix.mtx", "RNA_counts/genes.tsv",
        "RNA_counts/barcodes.tsv", [row["rna"] for row in markers], ordered_cells,
    )
    states, valid, prevalence, margins = [], [], [], []
    for index, donor in enumerate(HELD_DONORS):
        panel = counts[index * CELL_BUDGET:(index + 1) * CELL_BUDGET]
        quality = _rna_quality(panel)
        states.append((panel > 0).astype(np.uint8))
        valid.append(quality["valid"])
        prevalence.append(quality["prevalence"])
        ones = np.sum(panel > 0, axis=0).astype(int)
        margins.append(np.stack([CELL_BUDGET - ones, ones], axis=-1))
    valid_array = np.asarray(valid)
    if np.any(valid_array.sum(axis=0) < MINIMUM_VALID_HELD_DONORS):
        raise ConfirmationRefusal("SOURCE_LOCKED_MARKER_HELD_RNA_SUPPORT_FAILED", {
            "valid_donor_counts": valid_array.sum(axis=0).tolist()
        })
    private = _private(private_state)
    private.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with private.open("xb") as stream:
        np.savez_compressed(
            stream, states=np.asarray(states), valid=valid_array,
            cells=np.asarray([selected[d] for d in HELD_DONORS]),
        )
    os.chmod(private, 0o600)
    return {
        "schema": "gse144744-ms-held-rna/1.0", "status": "HELD_RNA_PASS",
        "created_at_utc": _timestamp(), "donors": list(HELD_DONORS),
        "selected_libraries": _selected_libraries(metadata, "held"),
        "primary_marker_pairs": markers,
        "prevalence": np.asarray(prevalence).tolist(),
        "rna_margins": np.asarray(margins).tolist(),
        "valid": valid_array.tolist(), "valid_donor_counts": valid_array.sum(axis=0).tolist(),
        "private_state_sha256": _sha256(private), "access_audit": audit,
        "adt_matrix_member_opened": False,
    }


def prediction_stage() -> dict[str, Any]:
    rna = _read_json(RNA_RESULT)
    if rna.get("status") != "HELD_RNA_PASS":
        raise PermissionError("held RNA stage did not pass")
    markers, models, locked_classical = _load_source_models()
    predictions = {}
    for donor_index, donor in enumerate(HELD_DONORS):
        row = np.asarray(rna["rna_margins"][donor_index], dtype=int)
        column = np.tile(np.asarray([CELL_BUDGET // 2, CELL_BUDGET // 2]),
                         (len(markers), 1))
        rows = np.repeat(row[:, None, :], len(markers), axis=1)
        columns = np.repeat(column[None, :, :], len(markers), axis=0)
        predictions[donor] = {
            "rna_margins": row.tolist(), "adt_margins": column.tolist(),
            "methods": {name: _predict_model(model, rows, columns).tolist()
                        for name, model in models.items()},
        }
    return {
        "schema": "gse144744-ms-held-predictions/1.0",
        "status": "PREDICTIONS_FROZEN", "created_at_utc": _timestamp(),
        "donors": list(HELD_DONORS), "primary_marker_pairs": markers,
        "locked_classical_method": locked_classical,
        "predictions": predictions,
        "held_adt_numeric_values_converted": 0,
    }


def adt_stage(scratch: Path, private_state: Path) -> dict[str, Any]:
    if _read_json(PREDICTION_RESULT).get("status") != "PREDICTIONS_FROZEN":
        raise PermissionError("public prediction did not authorize held ADT access")
    markers, _, _ = _load_source_models()
    metadata_record = _manifest_file("GSE144744_metadata_per_cell.csv.gz")
    adt_record = _manifest_file("GSE144744_ADT_counts.tar.gz")
    metadata_path = _download(metadata_record["url"], scratch / metadata_record["name"],
                              metadata_record["bytes"], "adt")
    adt_path = _download(adt_record["url"], scratch / adt_record["name"],
                         adt_record["bytes"], "adt")
    if _sha256(metadata_path) != metadata_record["sha256"] or _sha256(adt_path) != adt_record["sha256"]:
        raise ConfirmationRefusal("FROZEN_INPUT_SHA256_MISMATCH")
    metadata = _metadata(metadata_path)
    selected = _selected_cells(metadata, "held")
    ordered_cells = list(itertools.chain.from_iterable(selected.values()))
    counts, audit = _stage_axis_and_counts(
        "adt", "ADT", adt_path, "ADT_counts/matrix.mtx", "ADT_counts/genes.tsv",
        "ADT_counts/barcodes.tsv", [row["adt"] for row in markers], ordered_cells,
    )
    states, valid, quality_rows = [], [], []
    for index, donor in enumerate(HELD_DONORS):
        panel = counts[index * CELL_BUDGET:(index + 1) * CELL_BUDGET]
        quality = _adt_quality(panel)
        states.append(_adt_states(panel, selected[donor], donor))
        valid.append(quality["valid"])
        quality_rows.append({"distinct": quality["distinct"].tolist(),
                             "largest_equal_fraction": quality["largest_equal_fraction"].tolist(),
                             "nonzero_fraction": quality["nonzero_fraction"].tolist(),
                             "interquartile_range": quality["interquartile_range"].tolist()})
    valid_array = np.asarray(valid)
    if np.any(valid_array.sum(axis=0) < MINIMUM_VALID_HELD_DONORS):
        raise ConfirmationRefusal("SOURCE_LOCKED_MARKER_HELD_ADT_SUPPORT_FAILED", {
            "valid_donor_counts": valid_array.sum(axis=0).tolist()
        })
    rna_valid = np.asarray(_read_json(RNA_RESULT)["valid"], dtype=bool)
    pair_counts = np.asarray([
        np.sum(rna_valid[index, :, None] & valid_array[index, None, :])
        for index in range(len(HELD_DONORS))
    ])
    minimum = math.ceil(MINIMUM_PAIR_FRACTION * len(markers) ** 2)
    if np.any(pair_counts < minimum):
        raise ConfirmationRefusal("HELD_LOCKED_MAP_IS_NOT_FULLY_SUPPORTED", {
            "informative_pair_counts": pair_counts.tolist(),
            "minimum_pairs": minimum,
        })
    private = _private(private_state)
    private.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with private.open("xb") as stream:
        np.savez_compressed(
            stream, states=np.asarray(states), valid=valid_array,
            cells=np.asarray([selected[d] for d in HELD_DONORS]),
        )
    os.chmod(private, 0o600)
    return {
        "schema": "gse144744-ms-held-adt/1.0", "status": "HELD_ADT_PASS",
        "created_at_utc": _timestamp(), "donors": list(HELD_DONORS),
        "selected_libraries": _selected_libraries(metadata, "held"),
        "primary_marker_pairs": markers,
        "valid": valid_array.tolist(), "quality": quality_rows,
        "valid_donor_counts": valid_array.sum(axis=0).tolist(),
        "informative_pair_counts": pair_counts.tolist(), "minimum_pairs": minimum,
        "private_state_sha256": _sha256(private), "access_audit": audit,
    }


def _binomial_tail(n: int, favorable: int) -> float:
    return sum(math.comb(n, value) for value in range(favorable, n + 1)) / 2 ** n


def _comparison(
    primary: dict[str, float], comparator: dict[str, float], metadata: dict[str, dict[str, str]],
    *, require_full_gate: bool,
) -> dict[str, Any]:
    differences = np.asarray([primary[d] - comparator[d] for d in HELD_DONORS])
    pair_differences = np.asarray([
        np.mean([primary[left] - comparator[left], primary[right] - comparator[right]])
        for left, right in HELD_PAIRS
    ])
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws = generator.integers(0, len(HELD_PAIRS), size=(BOOTSTRAPS, len(HELD_PAIRS)))
    interval = np.quantile(pair_differences[draws].mean(axis=1), [ALPHA, 1 - ALPHA])
    favorable_donors = int(np.sum(differences < 0))
    favorable_pairs = int(np.sum(pair_differences < 0))
    observed = float(pair_differences.mean())
    assignments = np.asarray(list(itertools.product((-1, 1), repeat=len(HELD_PAIRS))))
    magnitude_sign_flip_p = float(
        np.mean((assignments * pair_differences).mean(axis=1) <= observed)
    )
    exact_pair_sign_p = _binomial_tail(len(HELD_PAIRS), favorable_pairs)
    group = {}
    experiment = {}
    donor_annotations = {}
    for donor in HELD_DONORS:
        annotations = {
            (value["group"], value["exp_name"])
            for value in metadata.values() if value["donor"] == donor
        }
        if len(annotations) != 1:
            raise PermissionError("held donor annotations differ")
        donor_annotations[donor] = next(iter(annotations))
        group_name, experiment_name = donor_annotations[donor]
        difference = primary[donor] - comparator[donor]
        group.setdefault(group_name, []).append(difference)
        experiment.setdefault(experiment_name, []).append(difference)
    group_means = {name: float(np.mean(values)) for name, values in group.items()}
    experiment_means = {name: float(np.mean(values)) for name, values in experiment.items()}
    leave_one_pair_out = [float(np.delete(pair_differences, index).mean())
                          for index in range(len(pair_differences))]
    leave_one_experiment_out = {
        name: float(np.mean([
            primary[donor] - comparator[donor] for donor in HELD_DONORS
            if donor_annotations[donor][1] != name
        ]))
        for name in sorted(experiment)
    }
    primary_mean = float(np.mean(list(primary.values())))
    comparator_mean = float(np.mean(list(comparator.values())))
    relative = 1 - primary_mean / comparator_mean
    common = relative >= 0.05 and interval[1] < 0 and favorable_pairs >= 9
    full = (
        common and exact_pair_sign_p <= ALPHA
        and all(value < 0 for value in group_means.values())
        and all(value < 0 for value in experiment_means.values())
        and all(value < 0 for value in leave_one_pair_out)
        and all(value < 0 for value in leave_one_experiment_out.values())
    )
    return {
        "primary_mean_loss": primary_mean, "comparator_mean_loss": comparator_mean,
        "relative_reduction": relative, "paired_difference_97_5_ci": interval.tolist(),
        "bootstrap_draws": BOOTSTRAPS, "bootstrap_unit": "author-matched donor pair",
        "favorable_donors": favorable_donors, "favorable_pairs": favorable_pairs,
        "donor_sign_p": _binomial_tail(20, favorable_donors),
        "exact_matched_pair_sign_p": exact_pair_sign_p,
        "magnitude_weighted_sign_flip_p_sensitivity": magnitude_sign_flip_p,
        "group_mean_differences": group_means,
        "experiment_mean_differences": experiment_means,
        "leave_one_pair_out_means": leave_one_pair_out,
        "leave_one_experiment_out_means": leave_one_experiment_out,
        "donor_differences": {donor: float(value) for donor, value in zip(HELD_DONORS, differences)},
        "passes": bool(full if require_full_gate else common),
    }


def _score_public_paths() -> tuple[Path, ...]:
    return (
        SOURCE_RESULT, JOURNALS["source"], RNA_RESULT, JOURNALS["rna"],
        PREDICTION_RESULT, ADT_RESULT, JOURNALS["adt"],
    )


def _validate_score_authorization(rna_path: Path, adt_path: Path) -> None:
    authorization = _read_json(SCORE_AUTHORIZATION)
    if authorization.get("status") != "SCORE_AUTHORIZED":
        raise PermissionError("score authorization status differs")
    expected_public = {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in _score_public_paths()
    }
    if authorization.get("public_artifacts") != expected_public:
        raise PermissionError("score authorization public hashes differ")
    expected_private = {
        "rna": _sha256(rna_path),
        "adt": _sha256(adt_path),
    }
    if authorization.get("private_state_sha256") != expected_private:
        raise PermissionError("score authorization private hashes differ")


def score_stage(rna_private: Path, adt_private: Path, metadata_path: Path) -> dict[str, Any]:
    rna_path, adt_path = _private(rna_private), _private(adt_private)
    _validate_score_authorization(rna_path, adt_path)
    rna_public, adt_public = _read_json(RNA_RESULT), _read_json(ADT_RESULT)
    if (
        rna_public.get("status") != "HELD_RNA_PASS"
        or adt_public.get("status") != "HELD_ADT_PASS"
    ):
        raise PermissionError("held public stage status differs")
    if _sha256(rna_path) != rna_public["private_state_sha256"] or _sha256(adt_path) != adt_public["private_state_sha256"]:
        raise PermissionError("private state identity differs")
    prediction = _read_json(PREDICTION_RESULT)
    with np.load(rna_path, allow_pickle=False) as handle:
        rna_states, rna_valid, rna_cells = handle["states"], handle["valid"], handle["cells"]
    with np.load(adt_path, allow_pickle=False) as handle:
        adt_states, adt_valid, adt_cells = handle["states"], handle["valid"], handle["cells"]
    markers, source_models, locked_classical = _load_source_models()
    expected_donors = list(HELD_DONORS)
    expected_libraries = rna_public.get("selected_libraries")
    if any(public.get("donors") != expected_donors for public in (
        rna_public, prediction, adt_public
    )):
        raise PermissionError("held donor axis differs across staged artifacts")
    if any(public.get("primary_marker_pairs") != markers for public in (
        rna_public, prediction, adt_public
    )):
        raise PermissionError("held marker axis differs across staged artifacts")
    if (
        adt_public.get("selected_libraries") != expected_libraries
        or set(expected_libraries or {}) != set(HELD_DONORS)
    ):
        raise PermissionError("held library axis differs across staged artifacts")
    marker_count = len(markers)
    state_shape = (len(HELD_DONORS), CELL_BUDGET, marker_count)
    valid_shape = (len(HELD_DONORS), marker_count)
    cell_shape = (len(HELD_DONORS), CELL_BUDGET)
    if (
        rna_states.shape != state_shape or adt_states.shape != state_shape
        or rna_valid.shape != valid_shape or adt_valid.shape != valid_shape
        or rna_cells.shape != cell_shape or adt_cells.shape != cell_shape
    ):
        raise PermissionError("private held array shape differs")
    if (
        not np.array_equal(rna_valid, np.asarray(rna_public["valid"], dtype=bool))
        or not np.array_equal(adt_valid, np.asarray(adt_public["valid"], dtype=bool))
        or not rna_valid.all() or not adt_valid.all()
    ):
        raise PermissionError("held full-map support differs across staged artifacts")
    if not np.array_equal(rna_cells, adt_cells):
        raise PermissionError("held RNA and ADT cell axes differ")
    expected_methods = set(source_models)
    if set(prediction.get("predictions", {})) != set(HELD_DONORS):
        raise PermissionError("held prediction donor axis differs")
    for donor_index, donor in enumerate(HELD_DONORS):
        donor_prediction = prediction["predictions"][donor]
        if set(donor_prediction.get("methods", {})) != expected_methods:
            raise PermissionError("held prediction method axis differs")
        if donor_prediction.get("rna_margins") != rna_public["rna_margins"][donor_index]:
            raise PermissionError("held RNA margins differ across staged artifacts")
        if donor_prediction.get("adt_margins") != [
            [CELL_BUDGET // 2, CELL_BUDGET // 2] for _ in markers
        ]:
            raise PermissionError("held ADT design margins differ")
    method_losses: dict[str, dict[str, float]] = {}
    for donor_index, donor in enumerate(HELD_DONORS):
        truth = _binary_tables(rna_states[donor_index], adt_states[donor_index])
        mask = rna_valid[donor_index, :, None] & adt_valid[donor_index, None, :]
        for method, values in prediction["predictions"][donor]["methods"].items():
            estimate = np.asarray(values)
            if estimate.shape != (marker_count, marker_count, 2, 2):
                raise PermissionError("held prediction table axis differs")
            method_losses.setdefault(method, {})[donor] = _donor_loss(
                truth, estimate, mask
            )
    metadata_record = _manifest_file("GSE144744_metadata_per_cell.csv.gz")
    if _sha256(metadata_path) != metadata_record["sha256"]:
        raise PermissionError("score metadata identity differs")
    metadata = _metadata(metadata_path)
    _validate_pairs(metadata)
    if _selected_libraries(metadata, "held") != rna_public["selected_libraries"]:
        raise PermissionError("held library selection differs at scoring")
    locked = prediction["locked_classical_method"]
    if locked != locked_classical:
        raise PermissionError("locked classical method differs across artifacts")
    comparisons = {
        "locked_classical": _comparison(method_losses["primary"], method_losses[locked],
                                          metadata, require_full_gate=True),
        "untuned_raw_poisson": _comparison(method_losses["primary"], method_losses[RAW_RESIDUAL],
                                             metadata, require_full_gate=True),
        "pooled_log_linear": _comparison(
            method_losses["primary"], method_losses[POOLED_POISSON],
            metadata, require_full_gate=True,
        ),
        "destroyed_link": _comparison(method_losses["primary"], method_losses["destroyed_link"],
                                       metadata, require_full_gate=False),
        "graph_zero": _comparison(method_losses["primary"], method_losses["graph_zero"],
                                   metadata, require_full_gate=False),
    }
    passes = all(comparisons[name]["passes"] for name in (
        "locked_classical", "untuned_raw_poisson", "pooled_log_linear",
        "destroyed_link",
    ))
    return {
        "schema": "gse144744-ms-held-confirmation/1.0",
        "status": "CONFIRMATION_PASS" if passes else "COMPLETED_NEGATIVE_RESULT",
        "created_at_utc": _timestamp(), "donors": list(HELD_DONORS),
        "locked_classical_method": locked, "method_losses": method_losses,
        "comparisons": comparisons, "passes_primary_confirmation": passes,
        "graph_specific_superiority": comparisons["graph_zero"]["passes"],
    }


def authorize_score() -> dict[str, Any]:
    source_commit = _require_public_tag(
        COMPLETION_TAGS["source"], (SOURCE_RESULT, JOURNALS["source"])
    )
    rna_commit = _require_public_tag(
        COMPLETION_TAGS["rna"], (RNA_RESULT, JOURNALS["rna"])
    )
    prediction_commit = _require_public_tag(
        COMPLETION_TAGS["prediction"], (PREDICTION_RESULT,)
    )
    adt_commit = _require_public_tag(
        COMPLETION_TAGS["adt"], (ADT_RESULT, JOURNALS["adt"])
    )
    _require_ancestor(source_commit, rna_commit)
    _require_ancestor(rna_commit, prediction_commit)
    _require_ancestor(prediction_commit, adt_commit)
    rna = _read_json(RNA_RESULT)
    prediction = _read_json(PREDICTION_RESULT)
    adt = _read_json(ADT_RESULT)
    if (
        rna.get("status") != "HELD_RNA_PASS"
        or prediction.get("status") != "PREDICTIONS_FROZEN"
        or adt.get("status") != "HELD_ADT_PASS"
    ):
        raise PermissionError("score prerequisites did not pass")
    payload = {
        "schema": "gse144744-ms-score-authorization/1.0",
        "status": "SCORE_AUTHORIZED",
        "created_at_utc": _timestamp(),
        "protocol_tag": PROTOCOL_TAG,
        "adt_completion_tag": COMPLETION_TAGS["adt"],
        "adt_completion_commit": adt_commit,
        "public_artifacts": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in _score_public_paths()
        },
        "private_state_sha256": {
            "rna": rna["private_state_sha256"],
            "adt": adt["private_state_sha256"],
        },
    }
    _write_json_x(SCORE_AUTHORIZATION, payload)
    return payload


def verify_stage(stage: str) -> dict[str, str]:
    paths = (OUTPUTS[stage],)
    if stage in JOURNALS:
        paths += (JOURNALS[stage],)
    commit = _require_public_tag(COMPLETION_TAGS[stage], paths)
    attempt_commit = _remote_tag_commit(ATTEMPT_TAGS[stage])
    _require_ancestor(attempt_commit, commit)
    return {"stage": stage, "commit": commit}


def _run_stage(stage: str, function: Any) -> dict[str, Any]:
    _require_runtime()
    _require_attempt(stage)
    output = OUTPUTS[stage]
    if output.exists():
        raise FileExistsError(f"stage output already exists: {output}")
    try:
        payload = function()
    except ConfirmationRefusal as error:
        payload = {
            "schema": "gse144744-ms-terminal-stage/1.0", "stage": stage,
            "status": "TERMINAL_REFUSAL", "reason": error.code,
            "details": error.details, "created_at_utc": _timestamp(),
        }
    except BaseException as error:
        payload = {
            "schema": "gse144744-ms-terminal-stage/1.0", "stage": stage,
            "status": "TERMINAL_UNEXPECTED_EXCEPTION",
            "exception": type(error).__name__,
            "reason": f"UNEXPECTED_{type(error).__name__.upper()}",
            "created_at_utc": _timestamp(),
        }
    _write_json_x(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    claim = subparsers.add_parser("claim")
    claim.add_argument("stage", choices=tuple(ATTEMPTS))
    run = subparsers.add_parser("run")
    run.add_argument("stage", choices=tuple(ATTEMPTS))
    run.add_argument("--scratch", type=Path)
    run.add_argument("--private-rna", type=Path)
    run.add_argument("--private-adt", type=Path)
    run.add_argument("--metadata", type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("stage", choices=tuple(COMPLETION_TAGS))
    subparsers.add_parser("authorize-score")
    arguments = parser.parse_args()
    if arguments.command == "claim":
        result = claim_stage(arguments.stage)
    elif arguments.command == "verify":
        result = verify_stage(arguments.stage)
    elif arguments.command == "authorize-score":
        result = authorize_score()
    else:
        stage = arguments.stage
        if stage == "source":
            if arguments.scratch is None:
                parser.error("source requires --scratch")
            result = _run_stage(stage, lambda: source_stage(_private(arguments.scratch)))
        elif stage == "rna":
            if arguments.scratch is None or arguments.private_rna is None:
                parser.error("rna requires --scratch and --private-rna")
            result = _run_stage(stage, lambda: rna_stage(
                _private(arguments.scratch), _private(arguments.private_rna)
            ))
        elif stage == "prediction":
            result = _run_stage(stage, prediction_stage)
        elif stage == "adt":
            if arguments.scratch is None or arguments.private_adt is None:
                parser.error("adt requires --scratch and --private-adt")
            result = _run_stage(stage, lambda: adt_stage(
                _private(arguments.scratch), _private(arguments.private_adt)
            ))
        else:
            if any(value is None for value in (
                arguments.private_rna, arguments.private_adt, arguments.metadata
            )):
                parser.error("score requires --private-rna --private-adt --metadata")
            result = _run_stage(stage, lambda: score_stage(
                arguments.private_rna, arguments.private_adt, arguments.metadata
            ))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
