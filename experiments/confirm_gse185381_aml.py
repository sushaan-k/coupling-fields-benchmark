"""Prospectively frozen GSE185381 control-to-AML held-donor validation.

The public protocol and each attempt tag are hard access boundaries. ``source``
uses only the ten control donors. ``rna`` and ``adt`` reduce held AML assays in
separate processes, ``prediction`` freezes fixed-margin predictions without
opening paired states, and ``score`` is the first stage allowed to join them.
"""

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
import shutil
import subprocess
import sys
from typing import Any, Callable, Iterable
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np

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


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/confirmation/gse185381_aml"
DEFAULT_CANDIDATE = DATA_DIR / "candidate_designation_v1.json"
DEFAULT_PROTOCOL = DATA_DIR / "protocol_v1.json"
DEFAULT_RUNTIME = DATA_DIR / "runtime_environment_v1.json"
DEFAULT_MANIFEST = DATA_DIR / "source_manifest_v1.json"
DEFAULT_PREFLIGHT = ROOT / "results/development/gse185381_metadata_preflight_v1.json"
DEFAULT_PROTOCOL_DOCUMENT = (
    ROOT / "docs/GSE185381_CONTROL_TO_AML_FOLLOWON_PROTOCOL_2026-08-29.md"
)

DEFAULT_SOURCE = ROOT / "results/development/gse185381_aml_source_v1.json"
DEFAULT_RNA = ROOT / "results/development/gse185381_aml_rna_v1.json"
DEFAULT_ADT = ROOT / "results/development/gse185381_aml_adt_v1.json"
DEFAULT_PREDICTION = ROOT / "results/gse185381_aml_predictions_v1.json"
DEFAULT_SCORE_AUTHORIZATION = DATA_DIR / "score_authorization_v1.json"
DEFAULT_SCORE = ROOT / "results/gse185381_aml_confirmation_v1.json"

PROTOCOL_TAG = "gse185381-aml-v1-protocol"
SOURCE_ATTEMPT_TAG = "gse185381-aml-v1-source-attempt"
SOURCE_TAG = "gse185381-aml-v1-source"
RNA_ATTEMPT_TAG = "gse185381-aml-v1-rna-attempt"
RNA_TAG = "gse185381-aml-v1-rna"
ADT_ATTEMPT_TAG = "gse185381-aml-v1-adt-attempt"
ADT_TAG = "gse185381-aml-v1-adt"
PREDICTION_ATTEMPT_TAG = "gse185381-aml-v1-prediction-attempt"
PREDICTION_TAG = "gse185381-aml-v1-predictions"
SCORE_AUTHORIZATION_TAG = "gse185381-aml-v1-score-authorized"
SCORE_ATTEMPT_TAG = "gse185381-aml-v1-score-attempt"
RESULT_TAG = "gse185381-aml-v1-result"
PUBLIC_ORIGIN = "https://github.com/sushaan-k/coupling-fields-benchmark.git"

CELL_BUDGET = 384
MINIMUM_MARKERS = 9
MINIMUM_VALID_RNA_AXES = 9
MINIMUM_VALID_ADT_AXES = 9
MINIMUM_VALID_ORDERED_PAIRS = 128
MINIMUM_RNA_PREVALENCE = 0.05
MAXIMUM_RNA_PREVALENCE = 0.95
MAXIMUM_ADT_EQUAL_VALUE_FRACTION = 0.90
ASSAY_SHA256_POLICIES = {
    "rna": "compute only inside the first public authorized download: source stage for a source-selected pool, held RNA stage otherwise",
    "adt": "compute only inside the first public authorized download: source stage for a source-selected pool, held ADT stage otherwise",
}
MANIFEST_ASSAY_SHA256_STATUS = {
    "rna": "deferred to first authorized download: source stage for a source-selected pool, held RNA stage otherwise",
    "adt": "deferred to first authorized download: source stage for a source-selected pool, held ADT stage otherwise",
}
CELL_SELECTION_SALT = "GSE185381-CELL-BUDGET-v1"
ADT_TIE_SALT = "GSE185381-ADT-MIDRANK-v1"
DESTROYED_LINK_SALT = "GSE185381-DESTROYED-LINK-v1"
BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 20260829
MAXIMUM_CONDITION_NUMBER = 1e12
HETEROGENEITY_GRID = (0.1, 1.0, 10.0)
RIDGE_GRID = (0.01, 0.1, 1.0)
TRANSPORT_GRID = (0.5, 0.75, 1.0)
GRAPH_GRID = (0.1, 1.0)
GRAPH_NEIGHBORS = 2
RAW_RESIDUAL_METHOD = "poisson_independence_signed_deviance_residual_raw"
CLASSICAL_ORDER = (
    "poisson_independence_signed_deviance_residual",
    "common_effect_stratified_cmle",
    "pooled_saturated_poisson",
    "paule_mandel_random_effects_log_odds",
)
ALL_CLASSICAL_METHODS = (RAW_RESIDUAL_METHOD,) + CLASSICAL_ORDER
INDEPENDENCE_METHOD = "target_margin_independence"

PROTOCOL_BINDINGS = (
    ".gitattributes",
    "experiments/confirm_gse185381_aml.py",
    "tests/test_gse185381_aml_confirmation.py",
    "data/confirmation/gse185381_aml/candidate_designation_v1.json",
    "data/confirmation/gse185381_aml/protocol_v1.json",
    "data/confirmation/gse185381_aml/runtime_environment_v1.json",
    "data/confirmation/gse185381_aml/source_manifest_v1.json",
    "results/development/gse185381_metadata_preflight_v1.json",
    "docs/GSE185381_CONTROL_TO_AML_FOLLOWON_PROTOCOL_2026-08-29.md",
    "mapreg/heterogeneity_adaptive_coupling.py",
    "mapreg/hierarchical_conditional_coupling.py",
    "mapreg/__init__.py",
    "mapreg/classical_residuals.py",
    "mapreg/coupling_fields.py",
    "mapreg/factorial_coupling.py",
    "mapreg/table_prediction.py",
    "requirements.txt",
    "pyproject.toml",
)

STAGE_PATHS = {
    "source": DEFAULT_SOURCE,
    "rna": DEFAULT_RNA,
    "adt": DEFAULT_ADT,
    "prediction": DEFAULT_PREDICTION,
    "score": DEFAULT_SCORE,
}
ATTEMPT_PATHS = {
    stage: ROOT / f"results/development/gse185381_aml_{stage}_attempt_v1.jsonl"
    for stage in STAGE_PATHS
}
EXECUTION_CLAIM_PATHS = {
    stage: ROOT
    / f"results/development/gse185381_aml_{stage}_execution_consumed_v1.json"
    for stage in STAGE_PATHS
}
ACCESS_JOURNAL_PATHS = {
    stage: ROOT / f"results/development/gse185381_aml_{stage}_access_v1.jsonl"
    for stage in ("source", "rna", "adt")
}
ATTEMPT_TAGS = {
    "source": SOURCE_ATTEMPT_TAG,
    "rna": RNA_ATTEMPT_TAG,
    "adt": ADT_ATTEMPT_TAG,
    "prediction": PREDICTION_ATTEMPT_TAG,
    "score": SCORE_ATTEMPT_TAG,
}
COMPLETION_TAGS = {
    "source": SOURCE_TAG,
    "rna": RNA_TAG,
    "adt": ADT_TAG,
    "prediction": PREDICTION_TAG,
    "score": RESULT_TAG,
}


class ConfirmationRefusal(RuntimeError):
    """A prespecified terminal refusal before or during an assay stage."""

    def __init__(self, code: str):
        if not code or any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
            for character in code
        ):
            raise ValueError("refusal codes must be uppercase identifiers")
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, order=True)
class PrimaryConfig:
    heterogeneity_penalty: float
    ridge_penalty: float
    graph_penalty: float
    transport_multiplier: float


def _timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = json.load(stream, object_pairs_hook=_strict_object)
    if not isinstance(value, dict):
        raise PermissionError(f"{path.name} must contain one JSON object")
    return value


def _write_json_x(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"one-shot output already exists: {path}")
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_jsonl(path: Path, payload: dict[str, Any], *, create: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_APPEND
    flags |= os.O_CREAT | os.O_EXCL if create else 0
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "ab") as stream:
        stream.write(
            (
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode()
        )
        stream.flush()
        os.fsync(stream.fileno())


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _private_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        return resolved
    raise PermissionError(
        "private assay-derived artifacts must remain outside the repository"
    )


def _validate_public_payload(value: Any, key: str | None = None) -> None:
    forbidden_keys = {
        "states",
        "barcodes",
        "cell_ids",
        "selected_cell_ids",
        "selected",
        "pool_by_cell",
    }
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key in forbidden_keys:
                raise PermissionError(f"public payload contains private key {child_key}")
            _validate_public_payload(child, child_key)
    elif isinstance(value, list):
        for child in value:
            _validate_public_payload(child, key)
    elif isinstance(value, str) and (value.startswith("/") or value.startswith("~")):
        raise PermissionError(f"public payload contains a local path in {key}")


def _remote_tag_commit(tag: str) -> str:
    output = subprocess.run(
        [
            "git",
            "ls-remote",
            "--tags",
            PUBLIC_ORIGIN,
            f"refs/tags/{tag}",
            f"refs/tags/{tag}^{{}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    commits = [line.split()[0] for line in output if line.strip()]
    if not commits:
        raise PermissionError(f"required public tag is absent: {tag}")
    return commits[-1]


def _require_public_tag(tag: str, paths: Iterable[str]) -> str:
    commit = _remote_tag_commit(tag)
    for relative in paths:
        local = ROOT / relative
        if not local.is_file():
            raise PermissionError(f"bound path is absent: {relative}")
        published = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(published).hexdigest() != _sha256(local):
            raise PermissionError(f"public tag does not bind local bytes: {relative}")
    return commit


def _runtime_environment() -> dict[str, Any]:
    return {
        "python": {
            "implementation": sys.implementation.name,
            "version": ".".join(str(value) for value in sys.version_info[:3]),
        },
        "packages": {
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
        },
    }


def _require_runtime_environment() -> dict[str, Any]:
    specification = _read_json(DEFAULT_RUNTIME)
    if specification.get("schema") != "gse185381-runtime-environment/1.0":
        raise PermissionError("runtime specification schema differs")
    observed = _runtime_environment()
    if specification.get("required_runtime") != observed:
        raise PermissionError("runtime environment differs from frozen specification")
    return observed


def _manifest(candidate: dict[str, Any]) -> dict[str, Any]:
    manifest = _read_json(DEFAULT_MANIFEST)
    if manifest.get("schema") != "gse185381-aml-source-manifest/1.0":
        raise PermissionError("source manifest schema differs")
    files = manifest.get("pool_files")
    if not isinstance(files, list) or len(files) != len(candidate["pools"]):
        raise PermissionError("source manifest pool axis differs")
    by_pool = {row.get("pool_id"): row for row in files if isinstance(row, dict)}
    if set(by_pool) != {row["pool_id"] for row in candidate["pools"]}:
        raise PermissionError("source manifest pool identifiers differ")
    for pool in candidate["pools"]:
        record = by_pool[pool["pool_id"]]
        metadata = record.get("metadata", {})
        if any(
            metadata.get(key) != pool["metadata_file"].get(key)
            for key in ("bytes", "sha256")
        ) or metadata.get("url") != pool["metadata_url"]:
            raise PermissionError("source manifest metadata binding differs")
        for key in ("antibody_feature_reference", "gene_feature_list"):
            feature = record.get(key, {})
            if (
                not isinstance(feature.get("bytes"), int)
                or feature["bytes"] <= 0
                or not isinstance(feature.get("sha256"), str)
                or len(feature["sha256"]) != 64
            ):
                raise PermissionError("feature-reference manifest binding differs")
        for modality in ("rna", "adt"):
            assay = record.get(f"{modality}_assay", {})
            frozen = pool[f"{modality}_file"]
            if (
                assay.get("bytes") != frozen["bytes"]
                or assay.get("sha256") is not None
                or assay.get("url") != pool[f"{modality}_url"]
                or assay.get("sha256_status")
                != MANIFEST_ASSAY_SHA256_STATUS[modality]
            ):
                raise PermissionError("source manifest assay binding differs")
    return manifest


def _candidate() -> dict[str, Any]:
    value = _read_json(DEFAULT_CANDIDATE)
    if value.get("schema") != "gse185381-aml-candidate-designation/1.0":
        raise PermissionError("candidate designation schema differs")
    donors = value.get("donors")
    pools = value.get("pools")
    markers = value.get("markers")
    if (
        not isinstance(donors, list)
        or not isinstance(pools, list)
        or not isinstance(markers, list)
    ):
        raise PermissionError("candidate must declare donors, pools, and markers")
    source = [row for row in donors if row.get("role") == "source"]
    held = [row for row in donors if row.get("role") == "held"]
    if (
        len(donors) != 49
        or len(source) != 10
        or len(held) != 39
        or any(row.get("role") not in {"source", "held"} for row in donors)
    ):
        raise PermissionError(
            "candidate must freeze exactly 10 source and 39 held donors"
        )
    identifiers = [row.get("donor_id") for row in donors]
    if (
        any(not isinstance(value, str) or not value for value in identifiers)
        or len(set(identifiers)) != 49
    ):
        raise PermissionError("candidate donor identifiers must be unique strings")
    if any(
        not isinstance(row.get("acquisition_cluster"), str)
        or not row["acquisition_cluster"]
        for row in held
    ):
        raise PermissionError("every held donor requires an acquisition cluster")
    split = value.get("source_split", {})
    calibration = split.get("calibration", [])
    validation = split.get("validation", [])
    if (
        len(calibration) != 5
        or len(validation) != 5
        or set(calibration).intersection(validation)
        or set(calibration).union(validation) != {row["donor_id"] for row in source}
    ):
        raise PermissionError("candidate must freeze a disjoint 5/5 source split")
    components = split.get("components")
    if (
        split.get("pool_disjoint") is not True
        or not isinstance(components, list)
        or not components
    ):
        raise PermissionError("source split must declare pool-connected components")
    pools_by_id = {
        row.get("pool_id"): row
        for row in pools
        if isinstance(row, dict) and isinstance(row.get("pool_id"), str)
    }
    component_donors: list[str] = []
    component_ids: list[str] = []
    component_pool_ids: list[str] = []
    role_donors = {"calibration": set(), "validation": set()}
    source_ids = {row["donor_id"] for row in source}
    donor_by_id = {row["donor_id"]: row for row in donors}
    for component in components:
        if not isinstance(component, dict):
            raise PermissionError("source component must be an object")
        component_id = component.get("component_id")
        pool_id = component.get("selected_pool_id")
        members = component.get("donors")
        role = component.get("role")
        if (
            not isinstance(component_id, str)
            or not component_id
            or pool_id not in pools_by_id
            or not isinstance(members, list)
            or not members
            or role not in role_donors
            or any(member not in source_ids for member in members)
            or any(
                donor_by_id[member].get("selected_pool_id") != pool_id
                for member in members
            )
        ):
            raise PermissionError("source component contract differs")
        component_ids.append(component_id)
        component_pool_ids.append(pool_id)
        component_donors.extend(members)
        role_donors[role].update(members)
    if len(component_ids) != len(set(component_ids)):
        raise PermissionError("source component identifiers must be unique")
    if len(component_pool_ids) != len(set(component_pool_ids)):
        raise PermissionError("source components must use disjoint selected pools")
    if len(component_donors) != len(set(component_donors)) or set(
        component_donors
    ) != source_ids:
        raise PermissionError("source components must partition source donors")
    if role_donors["calibration"] != set(calibration) or role_donors[
        "validation"
    ] != set(validation):
        raise PermissionError("source component roles differ from the frozen split")
    if len(markers) != 16:
        raise PermissionError("candidate marker panel must contain 16 cognates")
    support = value.get("support_contract", {})
    if (
        support.get("minimum_valid_rna_axes_per_donor")
        != MINIMUM_VALID_RNA_AXES
        or support.get("minimum_valid_adt_axes_per_donor")
        != MINIMUM_VALID_ADT_AXES
        or support.get("minimum_valid_ordered_pairs_per_donor")
        != MINIMUM_VALID_ORDERED_PAIRS
        or support.get("total_ordered_pairs") != len(markers) ** 2
    ):
        raise PermissionError("candidate support contract differs")
    if len({row.get("rna_symbol") for row in markers}) != len(markers):
        raise PermissionError("RNA marker symbols must be unique")
    for marker in markers:
        aliases = marker.get("adt_aliases")
        if (
            not isinstance(aliases, list)
            or not aliases
            or any(not isinstance(alias, str) or not alias for alias in aliases)
        ):
            raise PermissionError("each marker requires exact ADT aliases")
    for pool in pools:
        if not all(
            isinstance(pool.get(key), str) and pool[key]
            for key in ("pool_id", "metadata_url", "rna_url", "adt_url")
        ):
            raise PermissionError(
                "each pool requires an id and three immutable file URLs"
            )
        for key in ("metadata_file",):
            record = pool.get(key)
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("bytes"), int)
                or record["bytes"] <= 0
                or not isinstance(record.get("sha256"), str)
                or len(record["sha256"]) != 64
            ):
                raise PermissionError("each pool file requires exact bytes and SHA-256")
        for modality in ("rna", "adt"):
            key = f"{modality}_file"
            record = pool.get(key)
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("bytes"), int)
                or record["bytes"] <= 0
                or record.get("sha256") is not None
                or record.get("sha256_policy") != ASSAY_SHA256_POLICIES[modality]
            ):
                raise PermissionError(
                    "assay files require official bytes and first-stage SHA-256 policy"
                )
    if len({pool["pool_id"] for pool in pools}) != len(pools):
        raise PermissionError("candidate pool identifiers must be unique")
    if any(
        row.get("selected_pool_id") not in pools_by_id
        for row in donors
    ):
        raise PermissionError("every donor requires one frozen selected pool")
    return value


def _source_components(
    candidate: dict[str, Any], role: str
) -> list[list[str]]:
    components = [
        list(component["donors"])
        for component in candidate["source_split"]["components"]
        if component["role"] == role
    ]
    if not components:
        raise PermissionError(f"source split has no {role} components")
    return components


def _protocol_paths(*extra: Path) -> tuple[str, ...]:
    return (*PROTOCOL_BINDINGS, *(_relative(path) for path in extra))


def _completion_paths(stage: str) -> tuple[str, ...]:
    paths = (
        _relative(STAGE_PATHS[stage]),
        _relative(ATTEMPT_PATHS[stage]),
        _relative(EXECUTION_CLAIM_PATHS[stage]),
    )
    if stage in ACCESS_JOURNAL_PATHS:
        paths += (_relative(ACCESS_JOURNAL_PATHS[stage]),)
    return paths


def _stage_prerequisites(stage: str) -> dict[str, Any]:
    if stage == "source":
        commit = _require_public_tag(PROTOCOL_TAG, PROTOCOL_BINDINGS)
        return {"protocol_tag": PROTOCOL_TAG, "protocol_commit": commit}
    if stage == "rna":
        _, commit = _require_completed_stage_artifact(
            SOURCE_TAG, DEFAULT_SOURCE, "source", require_success=True
        )
        return {
            "source_tag": SOURCE_TAG,
            "source_commit": commit,
            "source_sha256": _sha256(DEFAULT_SOURCE),
        }
    if stage == "adt":
        _, commit = _require_completed_stage_artifact(
            RNA_TAG, DEFAULT_RNA, "rna", require_success=True
        )
        return {
            "rna_tag": RNA_TAG,
            "rna_commit": commit,
            "rna_sha256": _sha256(DEFAULT_RNA),
        }
    if stage == "prediction":
        _, commit = _require_completed_stage_artifact(
            ADT_TAG, DEFAULT_ADT, "adt", require_success=True
        )
        return {
            "adt_tag": ADT_TAG,
            "adt_commit": commit,
            "adt_sha256": _sha256(DEFAULT_ADT),
        }
    if stage == "score":
        _, prediction_commit = _require_completed_stage_artifact(
            PREDICTION_TAG,
            DEFAULT_PREDICTION,
            "prediction",
            require_success=True,
        )
        authorization_commit = _require_public_tag(
            SCORE_AUTHORIZATION_TAG,
            (_relative(DEFAULT_SCORE_AUTHORIZATION),),
        )
        _require_score_authorization()
        return {
            "prediction_tag": PREDICTION_TAG,
            "prediction_commit": prediction_commit,
            "prediction_sha256": _sha256(DEFAULT_PREDICTION),
            "score_authorization_tag": SCORE_AUTHORIZATION_TAG,
            "score_authorization_commit": authorization_commit,
            "score_authorization_sha256": _sha256(DEFAULT_SCORE_AUTHORIZATION),
        }
    raise ValueError(f"unknown stage: {stage}")


def _access_records(stage: str) -> list[dict[str, Any]]:
    if stage not in ACCESS_JOURNAL_PATHS:
        return []
    path = ACCESS_JOURNAL_PATHS[stage]
    with path.open() as stream:
        rows = [
            json.loads(line, object_pairs_hook=_strict_object)
            for line in stream
            if line.strip()
        ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise PermissionError("assay access journal is malformed")
    header = rows[0]
    if (
        set(header)
        != {
            "schema",
            "stage",
            "status",
            "created_at_utc",
            "protocol_commit",
            "runtime_environment",
        }
        or header.get("schema") != "gse185381-aml-assay-access/1.0"
        or header.get("stage") != stage
        or header.get("status") != "OPENED_BEFORE_ASSAY_ACCESS"
        or not isinstance(header.get("created_at_utc"), str)
    ):
        raise PermissionError("assay access journal header differs")
    seen: set[tuple[str, str]] = set()
    candidate = _candidate() if len(rows) > 1 else None
    allowed_role = "source" if stage == "source" else "held"
    allowed_pool_ids = (
        {
            row["selected_pool_id"]
            for row in candidate["donors"]
            if row["role"] == allowed_role
        }
        if candidate is not None
        else set()
    )
    pool_by_id = (
        {row["pool_id"]: row for row in candidate["pools"]}
        if candidate is not None
        else {}
    )
    allowed_modalities = {"rna", "adt"} if stage == "source" else {stage}
    for row in rows[1:]:
        if (
            set(row)
            != {
                "schema",
                "stage",
                "status",
                "created_at_utc",
                "pool_id",
                "modality",
                "url",
                "filename",
                "expected_bytes",
                "expected_sha256",
                "observed_bytes",
                "observed_sha256",
            }
            or row.get("schema") != "gse185381-aml-assay-access/1.0"
            or row.get("stage") != stage
            or row.get("status") != "DOWNLOADED_AND_HASHED"
            or row.get("modality") not in {"rna", "adt"}
            or not isinstance(row.get("pool_id"), str)
            or not isinstance(row.get("created_at_utc"), str)
            or row["created_at_utc"] < header["created_at_utc"]
            or not isinstance(row.get("observed_bytes"), int)
            or not isinstance(row.get("observed_sha256"), str)
            or len(row["observed_sha256"]) != 64
        ):
            raise PermissionError("assay access journal record differs")
        identity = (row["modality"], row["pool_id"])
        if identity in seen:
            raise PermissionError("assay access journal repeats a pool modality")
        seen.add(identity)
        pool = pool_by_id.get(row["pool_id"])
        modality = row["modality"]
        if (
            row["pool_id"] not in allowed_pool_ids
            or modality not in allowed_modalities
            or pool is None
            or row.get("url") != pool[f"{modality}_url"]
            or row.get("filename") != _url_filename(pool[f"{modality}_url"])
            or row.get("expected_bytes") != pool[f"{modality}_file"]["bytes"]
            or row.get("expected_sha256")
            != pool[f"{modality}_file"].get("sha256")
        ):
            raise PermissionError("assay access record differs from the frozen pool")
    return rows


def _validate_access_header(
    stage: str,
    records: list[dict[str, Any]],
    protocol_commit: str,
    runtime: dict[str, Any],
    started_at_utc: str,
) -> None:
    if stage not in ACCESS_JOURNAL_PATHS:
        if records:
            raise PermissionError("nonassay stage has an assay access journal")
        return
    header = records[0]
    if (
        header.get("protocol_commit") != protocol_commit
        or header.get("runtime_environment") != runtime
        or header.get("created_at_utc") != started_at_utc
    ):
        raise PermissionError("assay access header differs from the STARTED event")


def _access_summary(stage: str) -> dict[str, Any] | None:
    if stage not in ACCESS_JOURNAL_PATHS:
        return None
    records = _access_records(stage)
    path = ACCESS_JOURNAL_PATHS[stage]
    return {
        "journal_path": _relative(path),
        "journal_sha256": _sha256(path),
        "downloaded_and_hashed_files": records[1:],
    }


def _published_jsonl(commit: str, path: Path) -> list[dict[str, Any]]:
    published = subprocess.run(
        ["git", "show", f"{commit}:{_relative(path)}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return [
        json.loads(line, object_pairs_hook=_strict_object)
        for line in published
        if line.strip()
    ]


def _require_public_attempt_prefix(stage: str, started: dict[str, Any]) -> str:
    commit = _remote_tag_commit(ATTEMPT_TAGS[stage])
    if _published_jsonl(commit, ATTEMPT_PATHS[stage]) != [started]:
        raise PermissionError("public attempt tag does not bind the STARTED event")
    if stage in ACCESS_JOURNAL_PATHS:
        local_records = _access_records(stage)
        if _published_jsonl(commit, ACCESS_JOURNAL_PATHS[stage]) != local_records[:1]:
            raise PermissionError("public attempt tag does not bind the access header")
    return commit


def _validate_started_attempt(
    stage: str,
    started: dict[str, Any],
    runtime: dict[str, Any],
    protocol_commit: str,
) -> None:
    if (
        set(started)
        != {
            "schema",
            "stage",
            "status",
            "created_at_utc",
            "attempt_tag_required_before_assay_access",
            "prerequisites",
            "protocol_commit",
            "runtime_environment",
            "one_shot",
        }
        or started.get("schema") != "gse185381-aml-stage-attempt/1.0"
        or started.get("stage") != stage
        or started.get("status") != "STARTED"
        or started.get("attempt_tag_required_before_assay_access")
        != ATTEMPT_TAGS[stage]
        or started.get("prerequisites") != _stage_prerequisites(stage)
        or started.get("protocol_commit") != protocol_commit
        or started.get("runtime_environment") != runtime
        or started.get("one_shot") is not True
        or not isinstance(started.get("created_at_utc"), str)
    ):
        raise PermissionError("STARTED event differs from the public stage contract")


def claim_stage(stage: str) -> dict[str, Any]:
    if stage not in STAGE_PATHS:
        raise ValueError(f"unknown stage: {stage}")
    if (
        STAGE_PATHS[stage].exists()
        or ATTEMPT_PATHS[stage].exists()
        or EXECUTION_CLAIM_PATHS[stage].exists()
        or (
            stage in ACCESS_JOURNAL_PATHS
            and ACCESS_JOURNAL_PATHS[stage].exists()
        )
    ):
        raise PermissionError(f"stage has already been claimed: {stage}")
    runtime = _require_runtime_environment()
    protocol_commit = _require_public_tag(PROTOCOL_TAG, PROTOCOL_BINDINGS)
    prerequisites = _stage_prerequisites(stage)
    payload = {
        "schema": "gse185381-aml-stage-attempt/1.0",
        "stage": stage,
        "status": "STARTED",
        "created_at_utc": _timestamp(),
        "attempt_tag_required_before_assay_access": ATTEMPT_TAGS[stage],
        "prerequisites": prerequisites,
        "protocol_commit": protocol_commit,
        "runtime_environment": runtime,
        "one_shot": True,
    }
    if stage in ACCESS_JOURNAL_PATHS:
        access_header = {
            "schema": "gse185381-aml-assay-access/1.0",
            "stage": stage,
            "status": "OPENED_BEFORE_ASSAY_ACCESS",
            "created_at_utc": payload["created_at_utc"],
            "protocol_commit": protocol_commit,
            "runtime_environment": runtime,
        }
        _validate_public_payload(access_header)
        _append_jsonl(ACCESS_JOURNAL_PATHS[stage], access_header, create=True)
    _append_jsonl(ATTEMPT_PATHS[stage], payload, create=True)
    return payload


def _attempt_records(stage: str) -> list[dict[str, Any]]:
    path = ATTEMPT_PATHS[stage]
    with path.open() as stream:
        rows = [
            json.loads(line, object_pairs_hook=_strict_object)
            for line in stream
            if line.strip()
        ]
    if not all(isinstance(row, dict) for row in rows):
        raise PermissionError("attempt ledger is malformed")
    return rows


def _require_public_attempt(stage: str) -> dict[str, Any]:
    runtime = _require_runtime_environment()
    records = _attempt_records(stage)
    if len(records) != 1:
        raise PermissionError("attempt ledger must contain exactly one STARTED event")
    protocol_commit = _require_public_tag(PROTOCOL_TAG, PROTOCOL_BINDINGS)
    _validate_started_attempt(stage, records[0], runtime, protocol_commit)
    attempt_paths = [_relative(ATTEMPT_PATHS[stage])]
    if stage in ACCESS_JOURNAL_PATHS:
        access_records = _access_records(stage)
        _validate_access_header(
            stage,
            access_records,
            protocol_commit,
            runtime,
            records[0]["created_at_utc"],
        )
        if len(access_records) != 1:
            raise PermissionError("assay access began before the public attempt")
        attempt_paths.append(_relative(ACCESS_JOURNAL_PATHS[stage]))
    _require_public_tag(
        ATTEMPT_TAGS[stage],
        attempt_paths,
    )
    return records[0]


def _run_claimed_stage(
    stage: str, body: Callable[[], dict[str, Any]]
) -> dict[str, Any]:
    if STAGE_PATHS[stage].exists():
        raise PermissionError(f"one-shot stage output already exists: {stage}")
    started = _require_public_attempt(stage)
    executing = {
        "schema": "gse185381-aml-stage-attempt/1.0",
        "stage": stage,
        "status": "EXECUTING_CONSUMED",
        "created_at_utc": _timestamp(),
        "protocol_commit": started["protocol_commit"],
        "runtime_environment": started["runtime_environment"],
        "interruption_consumes_stage": True,
    }
    _write_json_x(EXECUTION_CLAIM_PATHS[stage], executing)
    _append_jsonl(ATTEMPT_PATHS[stage], executing, create=False)
    try:
        payload = dict(body())
        payload["protocol_commit"] = started["protocol_commit"]
        payload["runtime_environment"] = started["runtime_environment"]
        access_summary = _access_summary(stage)
        if access_summary is not None:
            payload["assay_access"] = access_summary
        terminal_status = payload["status"]
        _validate_public_payload(payload)
        _write_json_x(STAGE_PATHS[stage], payload)
    except BaseException as error:
        if STAGE_PATHS[stage].exists():
            raise
        if isinstance(error, ConfirmationRefusal):
            code = error.code
        elif isinstance(error, CouplingEstimationRefusal):
            code = "COUPLING_ESTIMATION_REFUSAL"
        else:
            code = f"UNEXPECTED_{type(error).__name__.upper()}"
        payload = {
            "schema": f"gse185381-aml-{stage}-terminal/1.0",
            "status": f"TERMINAL_{stage.upper()}_REFUSAL",
            "created_at_utc": _timestamp(),
            "refusal_code": code,
            "attempt_created_at_utc": started["created_at_utc"],
            "protocol_commit": started["protocol_commit"],
            "runtime_environment": started["runtime_environment"],
            "rerun_or_rescue_permitted": False,
        }
        access_summary = _access_summary(stage)
        if access_summary is not None:
            payload["assay_access"] = access_summary
        terminal_status = payload["status"]
        _validate_public_payload(payload)
        _write_json_x(STAGE_PATHS[stage], payload)
    finished = {
        "schema": "gse185381-aml-stage-attempt/1.0",
        "stage": stage,
        "status": "FINISHED",
        "created_at_utc": _timestamp(),
        "terminal_status": terminal_status,
        "output_sha256": _sha256(STAGE_PATHS[stage]),
        "protocol_commit": started["protocol_commit"],
        "runtime_environment": started["runtime_environment"],
    }
    _append_jsonl(ATTEMPT_PATHS[stage], finished, create=False)
    return payload


def _require_completed(
    stage: str, *, require_success: bool = True
) -> dict[str, Any]:
    if (
        not STAGE_PATHS[stage].exists()
        or not ATTEMPT_PATHS[stage].exists()
        or not EXECUTION_CLAIM_PATHS[stage].exists()
    ):
        raise PermissionError(f"stage is not complete: {stage}")
    rows = _attempt_records(stage)
    if len(rows) != 3:
        raise PermissionError(f"stage ledger is incomplete: {stage}")
    runtime = _require_runtime_environment()
    protocol_commit = _require_public_tag(PROTOCOL_TAG, PROTOCOL_BINDINGS)
    _validate_started_attempt(stage, rows[0], runtime, protocol_commit)
    if stage in ACCESS_JOURNAL_PATHS:
        _validate_access_header(
            stage,
            _access_records(stage),
            protocol_commit,
            runtime,
            rows[0]["created_at_utc"],
        )
    if (
        set(rows[1])
        != {
            "schema",
            "stage",
            "status",
            "created_at_utc",
            "protocol_commit",
            "runtime_environment",
            "interruption_consumes_stage",
        }
        or rows[1].get("schema") != "gse185381-aml-stage-attempt/1.0"
        or rows[1].get("stage") != stage
        or rows[1].get("status") != "EXECUTING_CONSUMED"
        or rows[1].get("protocol_commit") != protocol_commit
        or rows[1].get("runtime_environment") != runtime
        or rows[1].get("interruption_consumes_stage") is not True
        or not isinstance(rows[1].get("created_at_utc"), str)
    ):
        raise PermissionError(f"execution ledger event differs: {stage}")
    if (
        set(rows[2])
        != {
            "schema",
            "stage",
            "status",
            "created_at_utc",
            "terminal_status",
            "output_sha256",
            "protocol_commit",
            "runtime_environment",
        }
        or rows[2].get("schema") != "gse185381-aml-stage-attempt/1.0"
        or rows[2].get("stage") != stage
        or rows[2].get("status") != "FINISHED"
        or rows[2].get("protocol_commit") != protocol_commit
        or rows[2].get("runtime_environment") != runtime
        or not isinstance(rows[2].get("created_at_utc"), str)
        or not isinstance(rows[2].get("terminal_status"), str)
        or not isinstance(rows[2].get("output_sha256"), str)
    ):
        raise PermissionError(f"finished ledger event differs: {stage}")
    consumed = _read_json(EXECUTION_CLAIM_PATHS[stage])
    if consumed != rows[1]:
        raise PermissionError(f"execution claim differs from ledger: {stage}")
    _require_public_attempt_prefix(stage, rows[0])
    if rows[2].get("output_sha256") != _sha256(STAGE_PATHS[stage]):
        raise PermissionError(f"stage output hash differs: {stage}")
    value = _read_json(STAGE_PATHS[stage])
    if (
        value.get("protocol_commit") != protocol_commit
        or value.get("runtime_environment") != runtime
        or rows[2].get("terminal_status") != value.get("status")
    ):
        raise PermissionError(f"stage output differs from ledger: {stage}")
    access_summary = _access_summary(stage)
    if access_summary is not None and value.get("assay_access") != access_summary:
        raise PermissionError(f"stage output differs from assay access journal: {stage}")
    if require_success and str(value.get("status", "")).startswith("TERMINAL_"):
        raise PermissionError(f"stage ended in terminal refusal: {stage}")
    return value


def _require_completed_stage_artifact(
    tag: str,
    path: Path,
    stage: str,
    *,
    require_success: bool,
) -> tuple[dict[str, Any], str]:
    if path != STAGE_PATHS[stage]:
        raise ValueError("stage path differs")
    commit = _require_public_tag(tag, _completion_paths(stage))
    return _require_completed(stage, require_success=require_success), commit


def verify_stage(stage: str) -> dict[str, Any]:
    result, commit = _require_completed_stage_artifact(
        COMPLETION_TAGS[stage],
        STAGE_PATHS[stage],
        stage,
        require_success=False,
    )
    return {
        "stage": stage,
        "status": "PUBLIC_STAGE_LINEAGE_VERIFIED",
        "commit": commit,
        "result": result,
    }


def verify_public_result() -> dict[str, Any]:
    result, commit = _require_completed_stage_artifact(
        RESULT_TAG, DEFAULT_SCORE, "score", require_success=False
    )
    if result.get("schema") != "gse185381-aml-held-validation/1.0" or result.get(
        "status"
    ) not in {
        "COMPLETED_ESTIMATOR_GATE_PASS",
        "COMPLETED_ESTIMATOR_GATE_FAIL_PRIMARY_VS_LOCKED_CLASSICAL",
    }:
        raise PermissionError("public held-validation result schema or status differs")
    return {
        "schema": "gse185381-aml-public-result-verification/1.0",
        "status": "PUBLIC_RESULT_LINEAGE_VERIFIED",
        "result_commit": commit,
        "result_sha256": _sha256(DEFAULT_SCORE),
        "held_validation_status": result["status"],
    }


def _url_filename(url: str) -> str:
    name = Path(urlparse(url).path).name
    if not name:
        raise ConfirmationRefusal("URL_HAS_NO_FILENAME")
    return name


def _append_assay_access(
    stage: str,
    pool_id: str,
    modality: str,
    url: str,
    destination: Path,
    expected: dict[str, Any],
) -> None:
    if stage not in ACCESS_JOURNAL_PATHS:
        raise PermissionError("assay access stage has no public journal")
    existing = _access_records(stage)
    if any(
        row.get("modality") == modality and row.get("pool_id") == pool_id
        for row in existing[1:]
    ):
        raise PermissionError("assay access journal repeats a pool modality")
    record = {
        "schema": "gse185381-aml-assay-access/1.0",
        "stage": stage,
        "status": "DOWNLOADED_AND_HASHED",
        "created_at_utc": _timestamp(),
        "pool_id": pool_id,
        "modality": modality,
        "url": url,
        "filename": destination.name,
        "expected_bytes": expected["bytes"],
        "expected_sha256": expected.get("sha256"),
        "observed_bytes": destination.stat().st_size,
        "observed_sha256": _sha256(destination),
    }
    _validate_public_payload(record)
    _append_jsonl(ACCESS_JOURNAL_PATHS[stage], record, create=False)


def _fetch(
    url: str,
    destination: Path,
    expected: dict[str, Any] | None = None,
    *,
    authorized_modality: str | None = None,
    authorized_stage: str | None = None,
    access_stage: str | None = None,
    pool_id: str | None = None,
) -> Path:
    if expected is None:
        raise PermissionError("every downloaded input requires frozen bytes and SHA-256")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"scratch file already exists: {destination}")
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urlopen(url) as response, temporary.open("xb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    temporary.replace(destination)
    if access_stage is not None:
        if authorized_modality is None or pool_id is None:
            raise PermissionError("journaled assay access lacks modality or pool")
        _append_assay_access(
            access_stage,
            pool_id,
            authorized_modality,
            url,
            destination,
            expected,
        )
    if destination.stat().st_size != int(expected["bytes"]):
        raise ConfirmationRefusal("DOWNLOADED_FILE_SIZE_DIFFERS")
    expected_sha = expected.get("sha256")
    if expected_sha is None:
        allowed_stage = {
            "rna": {"source", "held_rna"},
            "adt": {"source", "held_adt"},
        }
        if (
            authorized_modality not in allowed_stage
            or authorized_stage not in allowed_stage[authorized_modality]
            or expected.get("sha256_policy")
            != ASSAY_SHA256_POLICIES[authorized_modality]
        ):
            raise PermissionError("unhashed assay lacks the frozen hash policy")
    elif _sha256(destination) != expected_sha:
        raise ConfirmationRefusal("DOWNLOADED_FILE_HASH_DIFFERS")
    return destination


def _open_text(path: Path):
    return (
        gzip.open(path, "rt", newline="")
        if path.suffix == ".gz"
        else path.open("r", newline="")
    )


def _metadata_path(
    pool: dict[str, Any], metadata_dir: Path, scratch_dir: Path
) -> tuple[Path, bool]:
    name = pool.get("metadata_filename", _url_filename(pool["metadata_url"]))
    local = metadata_dir / name
    if local.exists():
        expected = pool["metadata_file"]
        if (
            local.stat().st_size != expected["bytes"]
            or _sha256(local) != expected["sha256"]
        ):
            raise ConfirmationRefusal("LOCAL_METADATA_BYTES_DIFFER")
        return local, False
    return _fetch(
        pool["metadata_url"], scratch_dir / name, pool.get("metadata_file")
    ), True


def _metadata_assignments(
    candidate: dict[str, Any], metadata_dir: Path, scratch_dir: Path
) -> dict[str, dict[str, str]]:
    columns = candidate.get("metadata_columns", {})
    cell_column = columns.get("cell", "cell")
    donor_column = columns.get("donor", "samples")
    pool_column = columns.get("pool", "orig.ident")
    designated = {row["donor_id"] for row in candidate["donors"]}
    output: dict[str, dict[str, str]] = {}
    for pool in candidate["pools"]:
        path, remove = _metadata_path(pool, metadata_dir, scratch_dir)
        try:
            with _open_text(path) as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None or not {
                    cell_column,
                    donor_column,
                    pool_column,
                } <= set(reader.fieldnames):
                    raise ConfirmationRefusal("METADATA_SCHEMA_DIFFERS")
                for row in reader:
                    donor = row[donor_column]
                    if donor not in designated:
                        continue
                    cell = row[cell_column]
                    if not cell or cell in output:
                        raise ConfirmationRefusal("METADATA_CELL_IDENTIFIER_DUPLICATED")
                    if row[pool_column] != pool["pool_id"]:
                        raise ConfirmationRefusal("METADATA_POOL_IDENTIFIER_DIFFERS")
                    output[cell] = {"donor": donor, "pool_id": pool["pool_id"]}
        finally:
            if remove:
                path.unlink(missing_ok=True)
    return output


def _selected_cells(
    assignments: dict[str, dict[str, str]],
    donors: list[str],
    selected_pool_by_donor: dict[str, str],
) -> dict[str, list[str]]:
    by_donor = {donor: [] for donor in donors}
    for cell, row in assignments.items():
        if (
            row["donor"] in by_donor
            and row["pool_id"] == selected_pool_by_donor[row["donor"]]
        ):
            by_donor[row["donor"]].append(cell)
    selected: dict[str, list[str]] = {}
    for donor in donors:
        cells = by_donor[donor]
        if len(cells) < CELL_BUDGET:
            raise ConfirmationRefusal(
                "DONOR_SELECTED_POOL_HAS_FEWER_THAN_384_PUBLISHER_FILTERED_CELLS"
            )
        selected[donor] = sorted(
            cells,
            key=lambda cell: (
                hashlib.sha256(
                    f"{CELL_SELECTION_SALT}|{donor}|{cell}".encode()
                ).hexdigest(),
                cell,
            ),
        )[:CELL_BUDGET]
    return selected


def _cell_header_index(header: list[str], selected: list[str]) -> list[int]:
    lookup: dict[str, int] = {}
    ambiguous: set[str] = set()
    for index, value in enumerate(header[1:], start=1):
        aliases = {value}
        if ":" in value:
            aliases.add(value.split(":", 1)[1])
        for alias in aliases:
            if alias in lookup and lookup[alias] != index:
                ambiguous.add(alias)
            else:
                lookup[alias] = index
    indices = []
    for cell in selected:
        candidates = [cell]
        if ":" in cell:
            candidates.append(cell.split(":", 1)[1])
        matches = {
            lookup[value]
            for value in candidates
            if value in lookup and value not in ambiguous
        }
        if len(matches) != 1:
            raise ConfirmationRefusal("MATRIX_HEADER_DOES_NOT_MATCH_SELECTED_CELL_AXIS")
        indices.append(matches.pop())
    if len(set(indices)) != len(indices):
        raise ConfirmationRefusal("MATRIX_HEADER_COLLAPSES_SELECTED_CELLS")
    return indices


def _canonical_feature(value: str) -> str:
    return "".join(character for character in value.upper() if character.isalnum())


def _marker_aliases(markers: list[dict[str, Any]], modality: str) -> list[set[str]]:
    output = []
    for marker in markers:
        aliases = (
            marker.get("rna_aliases", [marker["rna_symbol"]])
            if modality == "rna"
            else marker["adt_aliases"]
        )
        normalized = {_canonical_feature(alias) for alias in aliases}
        if len(normalized) != len(aliases):
            raise PermissionError("marker aliases collide after exact canonicalization")
        output.append(normalized)
    return output


def _read_matrix_rows(
    path: Path, selected_cells: list[str], markers: list[dict[str, Any]], modality: str
) -> np.ndarray:
    aliases = _marker_aliases(markers, modality)
    values = np.empty((len(selected_cells), len(markers)), dtype=float)
    hits = np.zeros(len(markers), dtype=np.int64)
    with _open_text(path) as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ConfirmationRefusal("ASSAY_MATRIX_IS_EMPTY") from error
        indices = _cell_header_index(header, selected_cells)
        if max(indices, default=0) >= len(header):
            raise ConfirmationRefusal("ASSAY_MATRIX_HEADER_IS_TRUNCATED")
        for row in reader:
            if not row:
                continue
            key = _canonical_feature(row[0])
            matched = [
                index for index, accepted in enumerate(aliases) if key in accepted
            ]
            if not matched:
                continue
            if len(matched) != 1 or hits[matched[0]]:
                raise ConfirmationRefusal("ASSAY_FEATURE_ALIAS_IS_AMBIGUOUS")
            marker = matched[0]
            if len(row) != len(header):
                raise ConfirmationRefusal("ASSAY_MATRIX_ROW_HEADER_ALIGNMENT_DIFFERS")
            try:
                column = np.asarray(
                    [float(row[index]) for index in indices], dtype=float
                )
            except ValueError as error:
                raise ConfirmationRefusal(
                    "ASSAY_MATRIX_CONTAINS_NONNUMERIC_VALUE"
                ) from error
            if not np.isfinite(column).all() or np.any(column < 0.0):
                raise ConfirmationRefusal("ASSAY_MATRIX_CONTAINS_INVALID_COUNT")
            values[:, marker] = column
            hits[marker] += 1
    if not np.all(hits == 1):
        raise ConfirmationRefusal("ASSAY_MATRIX_DOES_NOT_CONTAIN_EXACT_MARKER_PANEL")
    return values


def _pool_selected(
    selected: dict[str, list[str]], assignments: dict[str, dict[str, str]], pool_id: str
) -> tuple[list[str], list[tuple[str, int]]]:
    cells: list[str] = []
    destinations: list[tuple[str, int]] = []
    for donor, axis in selected.items():
        for index, cell in enumerate(axis):
            if assignments[cell]["pool_id"] == pool_id:
                cells.append(cell)
                destinations.append((donor, index))
    return cells, destinations


def _reduce_modality(
    candidate: dict[str, Any],
    selected: dict[str, list[str]],
    assignments: dict[str, dict[str, str]],
    modality: str,
    scratch_dir: Path,
    *,
    authorized_stage: str,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    _manifest(candidate)
    scratch = _private_path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    markers = candidate["markers"]
    output = {donor: np.full((CELL_BUDGET, len(markers)), np.nan) for donor in selected}
    observed_files: dict[str, dict[str, Any]] = {}
    access_stage = {
        "source": "source",
        "held_rna": "rna",
        "held_adt": "adt",
    }.get(authorized_stage)
    if access_stage is None:
        raise PermissionError("authorized assay stage has no public access journal")
    pools = {row["pool_id"]: row for row in candidate["pools"]}
    for pool_id in sorted(pools):
        cells, destinations = _pool_selected(selected, assignments, pool_id)
        if not cells:
            continue
        pool = pools[pool_id]
        url = pool[f"{modality}_url"]
        path = _fetch(
            url,
            scratch / _url_filename(url),
            pool.get(f"{modality}_file"),
            authorized_modality=modality,
            authorized_stage=authorized_stage,
            access_stage=access_stage,
            pool_id=pool_id,
        )
        try:
            observed_files[pool_id] = {
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            matrix = _read_matrix_rows(path, cells, markers, modality)
        finally:
            path.unlink(missing_ok=True)
        for row, (donor, index) in enumerate(destinations):
            output[donor][index] = matrix[row]
    if any(not np.isfinite(values).all() for values in output.values()):
        raise ConfirmationRefusal("SELECTED_CELL_COUNTS_ARE_INCOMPLETE")
    return output, observed_files


def _adt_states(counts: np.ndarray, cells: list[str], donor: str) -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    if values.ndim != 2 or values.shape[0] != CELL_BUDGET:
        raise ValueError("ADT counts must use the frozen cell budget")
    states = np.zeros(values.shape, dtype=np.uint8)
    for marker in range(values.shape[1]):
        order = sorted(
            range(CELL_BUDGET),
            key=lambda index: (
                values[index, marker],
                hashlib.sha256(
                    f"{ADT_TIE_SALT}|{donor}|{marker}|{cells[index]}".encode()
                ).hexdigest(),
                cells[index],
            ),
        )
        states[np.asarray(order[CELL_BUDGET // 2 :], dtype=int), marker] = 1
    return states


def _rna_axis_quality(counts: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(counts, dtype=float)
    if values.ndim != 2 or values.shape[0] != CELL_BUDGET:
        raise ValueError("RNA counts must use the frozen cell budget")
    prevalence = np.mean(values > 0.0, axis=0)
    valid = (prevalence >= MINIMUM_RNA_PREVALENCE) & (
        prevalence <= MAXIMUM_RNA_PREVALENCE
    )
    return {"prevalence": prevalence, "valid": valid}


def _adt_axis_quality(counts: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(counts, dtype=float)
    if values.ndim != 2 or values.shape[0] != CELL_BUDGET:
        raise ValueError("ADT counts must use the frozen cell budget")
    distinct = np.empty(values.shape[1], dtype=np.int64)
    largest_fraction = np.empty(values.shape[1], dtype=float)
    for marker in range(values.shape[1]):
        _, frequencies = np.unique(values[:, marker], return_counts=True)
        distinct[marker] = len(frequencies)
        largest_fraction[marker] = float(frequencies.max() / CELL_BUDGET)
    valid = (distinct >= 2) & (
        largest_fraction <= MAXIMUM_ADT_EQUAL_VALUE_FRACTION
    )
    return {
        "distinct_values": distinct,
        "largest_equal_value_fraction": largest_fraction,
        "valid": valid,
    }


def _destroyed_adt(states: np.ndarray, cells: list[str], donor: str) -> np.ndarray:
    order = np.asarray(
        sorted(
            range(CELL_BUDGET),
            key=lambda index: (
                hashlib.sha256(
                    f"{DESTROYED_LINK_SALT}|{donor}|{cells[index]}".encode()
                ).hexdigest(),
                cells[index],
            ),
        ),
        dtype=int,
    )
    output = np.empty_like(states)
    output[order] = np.asarray(states)[np.roll(order, 1)]
    if not np.array_equal(output.sum(axis=0), np.asarray(states).sum(axis=0)):
        raise AssertionError("destroyed-link rotation changed ADT margins")
    return output


def _binary_tables(rna: np.ndarray, adt: np.ndarray) -> np.ndarray:
    first = np.asarray(rna, dtype=np.uint8)
    second = np.asarray(adt, dtype=np.uint8)
    if first.shape != second.shape or first.shape[0] != CELL_BUDGET:
        raise ValueError("RNA and ADT state panels differ")
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
    rows = values.sum(axis=-1)
    columns = values.sum(axis=-2)
    total = values.sum(axis=(-2, -1))
    return np.minimum(rows[..., 0], columns[..., 0]) > np.maximum(
        0, rows[..., 0] + columns[..., 0] - total
    )


def _poisson_signed_deviance(table: np.ndarray) -> float:
    counts = np.asarray(table, dtype=float)
    if counts.shape != (2, 2) or not np.isfinite(counts).all() or np.any(counts < 0.0):
        raise ValueError("table must be one finite nonnegative 2x2 array")
    total = float(counts.sum())
    if total <= 0.0:
        raise ValueError("table total must be positive")
    expected = np.outer(counts.sum(axis=1), counts.sum(axis=0)) / total
    positive = counts > 0.0
    if np.any(expected[positive] <= 0.0):
        raise ValueError("positive observations require positive independence means")
    deviance = 2.0 * float(
        np.sum(counts[positive] * np.log(counts[positive] / expected[positive]))
    )
    determinant = float(counts[0, 0] * counts[1, 1] - counts[0, 1] * counts[1, 0])
    return (
        math.copysign(math.sqrt(max(deviance, 0.0)), determinant)
        if determinant
        else 0.0
    )


def _fractional_signed_deviance(table: np.ndarray) -> float:
    values = np.asarray(table, dtype=float)
    expected = np.outer(values.sum(axis=1), values.sum(axis=0)) / values.sum()
    positive = values > 0.0
    deviance = 2.0 * float(
        np.sum(values[positive] * np.log(values[positive] / expected[positive]))
    )
    determinant = float(values[0, 0] * values[1, 1] - values[0, 1] * values[1, 0])
    return (
        math.copysign(math.sqrt(max(deviance, 0.0)), determinant)
        if determinant
        else 0.0
    )


def _classical_table(
    coordinate: float, rows: np.ndarray, columns: np.ndarray
) -> np.ndarray:
    row = np.asarray(rows, dtype=float)
    column = np.asarray(columns, dtype=float)
    total = float(row.sum())
    lower = max(0.0, row[0] + column[0] - total)
    upper = min(row[0], column[0])
    if upper <= lower:
        return np.asarray(
            [[lower, row[0] - lower], [column[0] - lower, row[1] - column[0] + lower]]
        )
    epsilon = min(1e-10, 0.25 * (upper - lower))
    left, right = lower + epsilon, upper - epsilon

    def at(value: float) -> np.ndarray:
        return np.asarray(
            [[value, row[0] - value], [column[0] - value, row[1] - column[0] + value]]
        )

    target = min(
        max(float(coordinate), _fractional_signed_deviance(at(left))),
        _fractional_signed_deviance(at(right)),
    )
    for _ in range(96):
        middle = (left + right) / 2.0
        if _fractional_signed_deviance(at(middle)) < target:
            left = middle
        else:
            right = middle
    return at((left + right) / 2.0)


def _margins(tables: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(tables)
    return values.sum(axis=-1), values.sum(axis=-2)


def _knn_incidence(profiles: np.ndarray) -> np.ndarray:
    values = np.asarray(profiles, dtype=float).T
    if values.ndim != 2 or values.shape[0] < MINIMUM_MARKERS or values.shape[1] < 3:
        raise ValueError("source marker profile dimensions differ")
    scale = values.std(axis=1, ddof=1)
    if np.any(scale <= 0.0) or not np.isfinite(scale).all():
        raise CouplingEstimationRefusal("source marker profile has zero variance")
    standardized = (values - values.mean(axis=1, keepdims=True)) / scale[:, None]
    edges: set[tuple[int, int]] = set()
    for marker in range(values.shape[0]):
        candidates = np.asarray(
            [index for index in range(values.shape[0]) if index != marker]
        )
        distances = np.linalg.norm(
            standardized[candidates] - standardized[marker][None, :], axis=1
        )
        order = candidates[np.lexsort((candidates, distances))]
        edges.update(
            tuple(sorted((marker, int(neighbor))))
            for neighbor in order[:GRAPH_NEIGHBORS]
        )
    incidence = np.zeros((values.shape[0], len(edges)), dtype=float)
    for column, (left, right) in enumerate(sorted(edges)):
        incidence[left, column] = 1.0
        incidence[right, column] = 1.0
    return incidence


def _fit_primary(
    tables: np.ndarray,
    rna_profiles: np.ndarray,
    adt_profiles: np.ndarray,
    config: PrimaryConfig,
) -> dict[str, Any]:
    values = np.asarray(tables, dtype=np.int64)
    size = values.shape[1]
    if config.graph_penalty == 0.0:
        first = second = np.eye(size, dtype=float)
    else:
        first = _knn_incidence(rna_profiles)
        second = _knn_incidence(adt_profiles)
    fit = fit_hierarchical_conditional_log_odds(
        values,
        first,
        second,
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
        "fit_certificate": {
            "gradient_norm": fit.gradient_norm,
            "scaled_gradient_norm": fit.scaled_gradient_norm,
            "schur_condition_number": fit.schur_condition_number,
            "theta_curvature_condition_number": fit.theta_curvature_condition_number,
            "iterations": fit.iterations,
            "rna_incidence_sha256": _array_sha256(first),
            "adt_incidence_sha256": _array_sha256(second),
        },
    }


def _residual_pool(tables: np.ndarray) -> np.ndarray:
    values = np.asarray(tables)
    support = _informative(values)
    coordinates = np.full(support.shape, np.nan)
    for donor, row, column in np.argwhere(support):
        coordinates[donor, row, column] = _poisson_signed_deviance(
            values[donor, row, column]
        ) / math.sqrt(CELL_BUDGET)
    if np.any(np.sum(support, axis=0) < 2):
        raise ConfirmationRefusal("POISSON_RESIDUAL_HAS_FEWER_THAN_TWO_SOURCE_DONORS")
    pooled = np.nanmean(coordinates, axis=0)
    if not np.isfinite(pooled).all():
        raise ConfirmationRefusal("POISSON_RESIDUAL_IS_NONFINITE")
    return pooled


def _fit_residual(tables: np.ndarray) -> dict[str, Any]:
    values = np.asarray(tables, dtype=np.int64)
    return {
        "family": "poisson_independence_signed_deviance_residual",
        "transport_multiplier": 1.0,
        "pooled_coordinate": _residual_pool(values),
        "support_count": _informative(values).sum(axis=0),
        "source_donor_weighting": "equal among informative donors per ordered pair",
    }


def _fit_common_effect(tables: np.ndarray) -> dict[str, Any]:
    values = np.asarray(tables, dtype=np.int64)
    size = values.shape[1]
    identity = np.eye(size, dtype=float)
    fit = fit_structured_conditional_log_odds(
        values,
        identity,
        identity,
        initial_log_odds=np.zeros((size, size), dtype=float),
        ridge_penalty=0.0,
        graph_penalty=0.0,
        minimum_informative_donors=2,
        maximum_condition_number=MAXIMUM_CONDITION_NUMBER,
        tolerance=1e-9,
    )
    return {
        "family": "common_effect_stratified_conditional_maximum_likelihood",
        "transport_multiplier": 1.0,
        "population_log_odds": fit.log_odds,
        "support_count": fit.support_count,
        "fit_certificate": {
            "gradient_norm": fit.gradient_norm,
            "condition_number": fit.condition_number,
            "iterations": fit.iterations,
        },
    }


def _fit_pooled_poisson(tables: np.ndarray) -> dict[str, Any]:
    pooled = np.asarray(tables, dtype=np.int64).sum(axis=0)
    if np.any(pooled <= 0):
        raise CouplingEstimationRefusal(
            "pooled saturated Poisson interaction has a zero cell"
        )
    log_odds = np.log(pooled[..., 0, 0]) + np.log(pooled[..., 1, 1])
    log_odds -= np.log(pooled[..., 0, 1]) + np.log(pooled[..., 1, 0])
    if not np.isfinite(log_odds).all():
        raise CouplingEstimationRefusal(
            "pooled saturated Poisson interaction is nonfinite"
        )
    return {
        "family": "pooled_saturated_poisson_log_linear_interaction",
        "transport_multiplier": 1.0,
        "population_log_odds": log_odds,
        "pooled_tables_sha256": _array_sha256(pooled),
        "source_donor_weighting": "pooled cell counts",
    }


def _fit_paule_mandel_log_odds(tables: np.ndarray) -> dict[str, Any]:
    values = np.asarray(tables, dtype=np.int64)
    coordinates = np.zeros(values.shape[:3], dtype=float)
    variances = np.zeros_like(coordinates)
    support = np.zeros(values.shape[:3], dtype=bool)
    for index in np.ndindex(values.shape[:3]):
        estimate = centered_haldane_log_odds(values[index])
        coordinates[index] = estimate.observed_log_odds
        variances[index] = estimate.sampling_variance
        support[index] = estimate.supported
    pooled = paule_mandel_pool(
        coordinates,
        variances,
        support=support,
        minimum_donors=2,
    )
    if not pooled.supported.all() or not np.isfinite(pooled.mean).all():
        raise CouplingEstimationRefusal(
            "Paule-Mandel log odds lacks source support"
        )
    return {
        "family": "haldane_log_odds_paule_mandel_random_effects",
        "transport_multiplier": 1.0,
        "population_log_odds": pooled.mean,
        "support_count": pooled.support_count,
        "tau_squared": pooled.tau_squared,
        "variance_convention": "Haldane delta-method within-donor sampling variance",
        "pooling_solver": "Paule-Mandel",
    }


def _fit_classical(method: str, tables: np.ndarray) -> dict[str, Any]:
    if method == "poisson_independence_signed_deviance_residual":
        return _fit_residual(tables)
    if method == "common_effect_stratified_cmle":
        return _fit_common_effect(tables)
    if method == "pooled_saturated_poisson":
        return _fit_pooled_poisson(tables)
    if method == "paule_mandel_random_effects_log_odds":
        return _fit_paule_mandel_log_odds(tables)
    raise ValueError(f"unknown classical method: {method}")


def _predict_log_odds(
    log_odds: np.ndarray, rows: np.ndarray, columns: np.ndarray, multiplier: float
) -> np.ndarray:
    field = np.asarray(log_odds, dtype=float)
    output = np.empty((*field.shape, 2, 2), dtype=float)
    for index in np.ndindex(field.shape):
        output[index] = expected_binary_table_from_log_odds(
            float(multiplier) * field[index], rows[index], columns[index]
        )
    return output


def _predict_residual(
    pool: np.ndarray, rows: np.ndarray, columns: np.ndarray, multiplier: float
) -> np.ndarray:
    coordinates = np.asarray(pool, dtype=float)
    output = np.empty((*coordinates.shape, 2, 2), dtype=float)
    for index in np.ndindex(coordinates.shape):
        output[index] = _classical_table(
            float(multiplier) * coordinates[index] * math.sqrt(CELL_BUDGET),
            rows[index],
            columns[index],
        )
    return output


def _predict_model(
    model: dict[str, Any], rows: np.ndarray, columns: np.ndarray
) -> np.ndarray:
    row_values = np.asarray(rows, dtype=np.int64)
    column_values = np.asarray(columns, dtype=np.int64)
    shape = row_values.shape[:-1]
    if column_values.shape != row_values.shape:
        raise ValueError("row and column margins differ")
    if model.get("family") == "target_margin_independence":
        total = row_values.sum(axis=-1)
        return (
            row_values[..., :, None]
            * column_values[..., None, :]
            / total[..., None, None]
        )
    multiplier = float(model.get("transport_multiplier", 1.0))
    if "pooled_coordinate" in model:
        return _predict_residual(
            np.asarray(model["pooled_coordinate"], dtype=float),
            row_values,
            column_values,
            multiplier,
        )
    field = np.asarray(model["population_log_odds"], dtype=float)
    if field.shape != shape:
        raise ValueError("model field and margin axes differ")
    return _predict_log_odds(field, row_values, column_values, multiplier)


def _donor_loss(
    truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None
) -> float:
    observed = np.asarray(truth, dtype=float)
    estimate = np.asarray(prediction, dtype=float)
    support = _informative(observed) if mask is None else np.asarray(mask, dtype=bool)
    if support.shape != observed.shape[:-2] or np.any(
        support & ~_informative(observed)
    ):
        raise PermissionError("scoring mask includes a noninformative table")
    if np.count_nonzero(support) < MINIMUM_VALID_ORDERED_PAIRS:
        raise ConfirmationRefusal("FEWER_THAN_MINIMUM_VALID_ORDERED_PAIRS")
    if not np.allclose(observed.sum(axis=-1), estimate.sum(axis=-1)) or not np.allclose(
        observed.sum(axis=-2), estimate.sum(axis=-2)
    ):
        raise PermissionError("prediction changed a target margin")
    observed = observed[support]
    estimate = estimate[support]
    positive = observed > 0.0
    if np.any(estimate[positive] <= 0.0) or not np.isfinite(estimate).all():
        raise FloatingPointError("prediction assigns invalid mass")
    terms = np.zeros_like(observed)
    terms[positive] = observed[positive] * np.log(
        observed[positive] / estimate[positive]
    )
    return float((2.0 * terms.sum(axis=(-2, -1)) / CELL_BUDGET).mean())


def _require_axis_support(
    rna_valid: np.ndarray,
    adt_valid: np.ndarray,
    refusal_code: str,
) -> tuple[int, int, int]:
    rna_axis = np.asarray(rna_valid, dtype=bool)
    adt_axis = np.asarray(adt_valid, dtype=bool)
    if rna_axis.ndim != 1 or adt_axis.ndim != 1:
        raise PermissionError("axis support must be one-dimensional")
    valid_rna = int(np.count_nonzero(rna_axis))
    valid_adt = int(np.count_nonzero(adt_axis))
    valid_pairs = valid_rna * valid_adt
    if (
        valid_rna < MINIMUM_VALID_RNA_AXES
        or valid_adt < MINIMUM_VALID_ADT_AXES
        or valid_pairs < MINIMUM_VALID_ORDERED_PAIRS
    ):
        raise ConfirmationRefusal(refusal_code)
    return valid_rna, valid_adt, valid_pairs


def _source_records(
    candidate: dict[str, Any],
    selected: dict[str, list[str]],
    rna: dict[str, np.ndarray],
    adt: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for donor in selected:
        rna_states = (rna[donor] > 0.0).astype(np.uint8)
        rna_quality = _rna_axis_quality(rna[donor])
        adt_states = _adt_states(adt[donor], selected[donor], donor)
        adt_quality = _adt_axis_quality(adt[donor])
        table_rna = rna_states.copy()
        table_adt = adt_states.copy()
        table_rna[:, ~rna_quality["valid"]] = 0
        table_adt[:, ~adt_quality["valid"]] = 0
        valid_mask = np.outer(rna_quality["valid"], adt_quality["valid"])
        valid_rna, valid_adt, valid_pairs = _require_axis_support(
            rna_quality["valid"],
            adt_quality["valid"],
            "SOURCE_DONOR_FAILS_FROZEN_SUPPORT_CONTRACT",
        )
        records[donor] = {
            "tables": _binary_tables(table_rna, table_adt),
            "destroyed_tables": _binary_tables(
                table_rna,
                _destroyed_adt(table_adt, selected[donor], donor),
            ),
            "rna_profile": rna_states.mean(axis=0),
            "adt_profile": np.log1p(adt[donor]).mean(axis=0),
            "rna_valid": rna_quality["valid"],
            "adt_valid": adt_quality["valid"],
            "rna_prevalence": rna_quality["prevalence"],
            "adt_distinct_values": adt_quality["distinct_values"],
            "adt_largest_equal_value_fraction": adt_quality[
                "largest_equal_value_fraction"
            ],
            "valid_rna_axes": valid_rna,
            "valid_adt_axes": valid_adt,
            "valid_ordered_pairs": valid_pairs,
            "valid_mask": valid_mask,
        }
    return records


def _json_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in model.items()
    }


def _record_arrays(
    records: dict[str, dict[str, Any]], donors: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([records[donor]["tables"] for donor in donors]),
        np.asarray([records[donor]["rna_profile"] for donor in donors]),
        np.asarray([records[donor]["adt_profile"] for donor in donors]),
    )


def _primary_configs(graph_values: tuple[float, ...]) -> list[PrimaryConfig]:
    return [
        PrimaryConfig(heterogeneity, ridge, graph, transport)
        for heterogeneity, ridge, graph, transport in itertools.product(
            HETEROGENEITY_GRID, RIDGE_GRID, graph_values, TRANSPORT_GRID
        )
    ]


def _component_equal_mean(
    donor_losses: dict[str, float], components: list[list[str]]
) -> float:
    if {donor for component in components for donor in component} != set(donor_losses):
        raise ValueError("component and donor-loss axes differ")
    return float(
        np.mean(
            [np.mean([donor_losses[donor] for donor in component]) for component in components]
        )
    )


def _select_primary_validation(
    records: dict[str, dict[str, Any]],
    training: list[str],
    validation_components: list[list[str]],
    *,
    graph_values: tuple[float, ...],
) -> dict[str, Any]:
    validation = [donor for component in validation_components for donor in component]
    candidates = _primary_configs(graph_values)
    losses: dict[PrimaryConfig, dict[str, float]] = {config: {} for config in candidates}
    refused_structures = 0
    tables, rna_profiles, adt_profiles = _record_arrays(records, training)
    structural: dict[tuple[float, float, float], dict[str, Any] | BaseException] = {}
    for config in candidates:
        key = (
            config.heterogeneity_penalty,
            config.ridge_penalty,
            config.graph_penalty,
        )
        if key not in structural:
            try:
                structural[key] = _fit_primary(
                    tables,
                    rna_profiles,
                    adt_profiles,
                    PrimaryConfig(*key, 1.0),
                )
            except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
                structural[key] = error
                refused_structures += 1
        base = structural[key]
        if isinstance(base, BaseException):
            continue
        model = dict(base)
        model["transport_multiplier"] = config.transport_multiplier
        for donor in validation:
            truth = records[donor]["tables"]
            rows, columns = _margins(truth)
            losses[config][donor] = _donor_loss(
                truth,
                _predict_model(model, rows, columns),
                records[donor]["valid_mask"],
            )
    complete = [
        config
        for config, values in losses.items()
        if set(values) == set(validation) and np.isfinite(list(values.values())).all()
    ]
    if not complete:
        raise ConfirmationRefusal("PRIMARY_COMPONENT_CV_HAS_NO_COMPLETE_CONFIGURATION")
    selected = min(
        complete,
        key=lambda config: (
            _component_equal_mean(losses[config], validation_components),
            config,
        ),
    )
    return {
        "selected": selected,
        "selected_losses": losses[selected],
        "selected_component_equal_loss": _component_equal_mean(
            losses[selected], validation_components
        ),
        "complete_candidates": len(complete),
        "refused_structures": refused_structures,
    }


def _select_classical_transport_validation(
    records: dict[str, dict[str, Any]],
    training: list[str],
    validation_components: list[list[str]],
    method: str,
) -> dict[str, Any]:
    validation = [donor for component in validation_components for donor in component]
    losses: dict[float, dict[str, float]] = {
        multiplier: {} for multiplier in TRANSPORT_GRID
    }
    tables = np.asarray([records[donor]["tables"] for donor in training])
    model = _fit_classical(method, tables)
    for donor in validation:
        truth = records[donor]["tables"]
        rows, columns = _margins(truth)
        for multiplier in TRANSPORT_GRID:
            candidate = dict(model)
            candidate["transport_multiplier"] = multiplier
            losses[multiplier][donor] = _donor_loss(
                truth,
                _predict_model(candidate, rows, columns),
                records[donor]["valid_mask"],
            )
    selected = min(
        TRANSPORT_GRID,
        key=lambda multiplier: (
            _component_equal_mean(losses[multiplier], validation_components),
            multiplier,
        ),
    )
    if not np.isfinite(list(losses[selected].values())).all():
        raise CouplingEstimationRefusal("classical component CV is incomplete")
    return {
        "selected_transport_multiplier": selected,
        "selected_losses": losses[selected],
        "selected_component_equal_loss": _component_equal_mean(
            losses[selected], validation_components
        ),
    }


def _model_losses(
    records: dict[str, dict[str, Any]], donors: list[str], model: dict[str, Any]
) -> np.ndarray:
    output = np.empty(len(donors), dtype=float)
    for index, donor in enumerate(donors):
        truth = records[donor]["tables"]
        rows, columns = _margins(truth)
        output[index] = _donor_loss(
            truth,
            _predict_model(model, rows, columns),
            records[donor]["valid_mask"],
        )
    return output


def _lock_classical(
    validation_losses: dict[str, np.ndarray],
    validation: list[str],
    components: list[list[str]],
) -> str:
    estimable = [
        method for method in ALL_CLASSICAL_METHODS if method in validation_losses
    ]
    if not estimable:
        raise ConfirmationRefusal("NO_CLASSICAL_COMPARATOR_ESTIMABLE")
    return min(
        estimable,
        key=lambda method: (
            _component_equal_mean(
                dict(zip(validation, map(float, validation_losses[method]))),
                components,
            ),
            ALL_CLASSICAL_METHODS.index(method),
        ),
    )


def _classical_refusal_record(method: str, error: BaseException) -> dict[str, str]:
    detail = "_".join(
        part
        for part in "".join(
            character if character.isalnum() else " "
            for character in str(error).upper()
        ).split()
        if part
    )
    if not detail:
        detail = type(error).__name__.upper()
    return {
        "code": f"{method.upper()}__{detail}",
        "exception_family": type(error).__name__,
    }


def _source_tuning(
    candidate: dict[str, Any],
    records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    calibration = list(candidate["source_split"]["calibration"])
    validation = list(candidate["source_split"]["validation"])
    validation_components = _source_components(candidate, "validation")
    primary_selection = _select_primary_validation(
        records,
        calibration,
        validation_components,
        graph_values=GRAPH_GRID,
    )
    graph_zero_selection = _select_primary_validation(
        records,
        calibration,
        validation_components,
        graph_values=(0.0,),
    )
    selected_primary: PrimaryConfig = primary_selection["selected"]
    selected_graph_zero: PrimaryConfig = graph_zero_selection["selected"]
    calibration_tables, calibration_rna, calibration_adt = _record_arrays(
        records, calibration
    )
    all_source = calibration + validation
    all_tables, _, _ = _record_arrays(records, all_source)
    primary_calibration = _fit_primary(
        calibration_tables, calibration_rna, calibration_adt, selected_primary
    )
    graph_zero_calibration = _fit_primary(
        calibration_tables,
        calibration_rna,
        calibration_adt,
        selected_graph_zero,
    )
    final_primary = _fit_primary(
        all_tables, calibration_rna, calibration_adt, selected_primary
    )
    final_graph_zero = _fit_primary(
        all_tables, calibration_rna, calibration_adt, selected_graph_zero
    )
    destroyed_tables = np.asarray(
        [records[donor]["destroyed_tables"] for donor in all_source]
    )
    final_destroyed = _fit_primary(
        destroyed_tables, calibration_rna, calibration_adt, selected_primary
    )

    classical_calibration: dict[str, dict[str, Any]] = {}
    classical_full: dict[str, dict[str, Any]] = {}
    classical_selection: dict[str, dict[str, Any]] = {}
    validation_losses: dict[str, np.ndarray] = {}
    refusals: dict[str, dict[str, str]] = {}
    raw_calibration = _fit_residual(calibration_tables)
    raw_full = _fit_residual(all_tables)
    classical_calibration[RAW_RESIDUAL_METHOD] = raw_calibration
    classical_full[RAW_RESIDUAL_METHOD] = raw_full
    validation_losses[RAW_RESIDUAL_METHOD] = _model_losses(
        records, validation, raw_calibration
    )
    for method in CLASSICAL_ORDER:
        try:
            selection = _select_classical_transport_validation(
                records, calibration, validation_components, method
            )
            calibration_model = _fit_classical(method, calibration_tables)
            full_model = _fit_classical(method, all_tables)
            multiplier = selection["selected_transport_multiplier"]
            calibration_model["transport_multiplier"] = multiplier
            full_model["transport_multiplier"] = multiplier
            classical_calibration[method] = calibration_model
            classical_full[method] = full_model
            classical_selection[method] = {
                "selected_transport_multiplier": multiplier,
                "validation_donor_losses": selection["selected_losses"],
                "component_equal_validation_loss": selection[
                    "selected_component_equal_loss"
                ],
            }
            validation_losses[method] = _model_losses(
                records, validation, calibration_model
            )
        except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
            refusals[method] = _classical_refusal_record(method, error)
    if "poisson_independence_signed_deviance_residual" not in classical_full:
        raise ConfirmationRefusal("CALIBRATED_POISSON_RESIDUAL_REFUSED")
    locked = _lock_classical(
        validation_losses, validation, validation_components
    )
    primary_validation = _model_losses(records, validation, primary_calibration)
    graph_zero_validation = _model_losses(
        records, validation, graph_zero_calibration
    )
    models = {
        "primary": _json_model(final_primary),
        "graph_zero_ablation": _json_model(final_graph_zero),
        "destroyed_link": _json_model(final_destroyed),
        **{
            method: _json_model(classical_full[method])
            for method in ALL_CLASSICAL_METHODS
            if method in classical_full
        },
        INDEPENDENCE_METHOD: {"family": "target_margin_independence"},
    }
    return {
        "calibration_donors": calibration,
        "validation_donors": validation,
        "source_components": candidate["source_split"]["components"],
        "primary_selection": {
            "rule": "fit on five calibration donors; minimize equal-weight mean deviance across three validation pool components; lexicographic frozen-grid tie break",
            "selected_configuration": asdict(selected_primary),
            "validation_donor_losses": primary_selection["selected_losses"],
            "component_equal_validation_loss": primary_selection[
                "selected_component_equal_loss"
            ],
            "complete_candidates": primary_selection["complete_candidates"],
            "refused_structures": primary_selection["refused_structures"],
        },
        "graph_zero_selection": {
            "selected_configuration": asdict(selected_graph_zero),
            "validation_donor_losses": graph_zero_selection["selected_losses"],
            "component_equal_validation_loss": graph_zero_selection[
                "selected_component_equal_loss"
            ],
        },
        "source_validation": {
            "primary_losses": dict(zip(validation, map(float, primary_validation))),
            "graph_zero_losses": dict(
                zip(validation, map(float, graph_zero_validation))
            ),
            "classical_losses": {
                method: dict(zip(validation, map(float, losses)))
                for method, losses in validation_losses.items()
            },
            "component_equal_classical_losses": {
                method: _component_equal_mean(
                    dict(zip(validation, map(float, losses))),
                    validation_components,
                )
                for method, losses in validation_losses.items()
            },
        },
        "classical_lock": {
            "rule": "minimum equal-weight mean deviance across the three frozen source-validation pool components among estimable classical methods",
            "locked_method": locked,
            "method_order": list(ALL_CLASSICAL_METHODS),
            "source_validation_selection": classical_selection,
            "refusals": refusals,
        },
        "models": models,
        "available_methods": list(models),
        "locked_classical_method": locked,
    }


def _source_stage_body(scratch_dir: Path, metadata_dir: Path) -> dict[str, Any]:
    candidate = _candidate()
    source = [row["donor_id"] for row in candidate["donors"] if row["role"] == "source"]
    selected_pool_by_donor = {
        row["donor_id"]: row["selected_pool_id"]
        for row in candidate["donors"]
        if row["role"] == "source"
    }
    scratch = _private_path(scratch_dir)
    assignments = _metadata_assignments(candidate, metadata_dir, scratch / "metadata")
    selected = _selected_cells(assignments, source, selected_pool_by_donor)
    rna, rna_files = _reduce_modality(
        candidate,
        selected,
        assignments,
        "rna",
        scratch / "rna",
        authorized_stage="source",
    )
    adt, adt_files = _reduce_modality(
        candidate,
        selected,
        assignments,
        "adt",
        scratch / "adt",
        authorized_stage="source",
    )
    records = _source_records(candidate, selected, rna, adt)
    tuning = _source_tuning(candidate, records)
    return {
        "schema": "gse185381-aml-source/1.0",
        "status": "SOURCE_MODELS_FROZEN_FOR_HELD_DONOR_VALIDATION",
        "created_at_utc": _timestamp(),
        "candidate_sha256": _sha256(DEFAULT_CANDIDATE),
        "protocol_sha256": _sha256(DEFAULT_PROTOCOL),
        "cell_budget": CELL_BUDGET,
        "marker_panel": candidate["markers"],
        "source_donors": source,
        "selected_cell_axis_sha256": {
            donor: _canonical_json_sha256(selected[donor]) for donor in source
        },
        "source_table_sha256": {
            donor: _array_sha256(records[donor]["tables"]) for donor in source
        },
        "destroyed_table_sha256": {
            donor: _array_sha256(records[donor]["destroyed_tables"])
            for donor in source
        },
        "source_axis_quality": {
            donor: {
                "rna_detection_prevalence": records[donor][
                    "rna_prevalence"
                ].tolist(),
                "rna_axis_valid": records[donor]["rna_valid"].tolist(),
                "adt_distinct_values": records[donor][
                    "adt_distinct_values"
                ].tolist(),
                "adt_largest_equal_value_fraction": records[donor][
                    "adt_largest_equal_value_fraction"
                ].tolist(),
                "adt_axis_valid": records[donor]["adt_valid"].tolist(),
                "valid_rna_axes": records[donor]["valid_rna_axes"],
                "required_valid_rna_axes": MINIMUM_VALID_RNA_AXES,
                "valid_adt_axes": records[donor]["valid_adt_axes"],
                "required_valid_adt_axes": MINIMUM_VALID_ADT_AXES,
                "valid_ordered_pairs": records[donor]["valid_ordered_pairs"],
                "required_valid_ordered_pairs": MINIMUM_VALID_ORDERED_PAIRS,
            }
            for donor in source
        },
        "observed_source_assay_files": {"rna": rna_files, "adt": adt_files},
        "model": tuning,
        "one_selected_pool_per_source_donor": True,
        "held_cell_columns_selected_or_converted": 0,
        "held_numeric_values_used_for_source_tuning": 0,
    }


def _write_private(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    destination = _private_path(path)
    _write_json_x(destination, payload)
    return {"sha256": _sha256(destination), "bytes": destination.stat().st_size}


def _require_reused_assay_hashes_match_source(
    source: dict[str, Any],
    modality: str,
    observed: dict[str, dict[str, Any]],
) -> None:
    source_files = source["observed_source_assay_files"][modality]
    for pool_id in set(source_files).intersection(observed):
        if source_files[pool_id] != observed[pool_id]:
            raise PermissionError("reused source-selected assay file bytes differ")


def _target_selection(
    candidate: dict[str, Any], metadata_dir: Path, scratch_dir: Path
) -> tuple[dict[str, dict[str, str]], dict[str, list[str]]]:
    held = [row["donor_id"] for row in candidate["donors"] if row["role"] == "held"]
    selected_pool_by_donor = {
        row["donor_id"]: row["selected_pool_id"]
        for row in candidate["donors"]
        if row["role"] == "held"
    }
    assignments = _metadata_assignments(
        candidate, metadata_dir, scratch_dir / "metadata"
    )
    return assignments, _selected_cells(assignments, held, selected_pool_by_donor)


def _rna_stage_body(
    scratch_dir: Path, metadata_dir: Path, selection_bridge: Path, rna_states: Path
) -> dict[str, Any]:
    source = _require_completed("source")
    candidate = _candidate()
    scratch = _private_path(scratch_dir)
    assignments, selected = _target_selection(candidate, metadata_dir, scratch)
    bridge = {
        "schema": "gse185381-private-selection-bridge/1.0",
        "selected": selected,
        "pool_by_cell": {
            cell: assignments[cell]["pool_id"]
            for cells in selected.values()
            for cell in cells
        },
    }
    bridge_certificate = _write_private(selection_bridge, bridge)
    counts, observed_files = _reduce_modality(
        candidate,
        selected,
        assignments,
        "rna",
        scratch / "rna",
        authorized_stage="held_rna",
    )
    _require_reused_assay_hashes_match_source(source, "rna", observed_files)
    states = {
        donor: (counts[donor] > 0.0).astype(np.uint8) for donor in selected
    }
    quality = {donor: _rna_axis_quality(counts[donor]) for donor in selected}
    private = _write_private(
        rna_states,
        {
            "schema": "gse185381-private-rna-states/1.0",
            "states": {donor: values.tolist() for donor, values in states.items()},
            "selected_axis_sha256": {
                donor: _canonical_json_sha256(selected[donor]) for donor in selected
            },
        },
    )
    return {
        "schema": "gse185381-aml-rna/1.0",
        "status": "HELD_RNA_REDUCED_WITHOUT_ADT_ACCESS",
        "created_at_utc": _timestamp(),
        "source_sha256": _sha256(DEFAULT_SOURCE),
        "held_donors": list(selected),
        "marker_panel": source["marker_panel"],
        "selected_axis_sha256": {
            donor: _canonical_json_sha256(selected[donor]) for donor in selected
        },
        "row_margins": {
            donor: {
                candidate["markers"][index]["rna_symbol"]: [
                    int(CELL_BUDGET - values[:, index].sum()),
                    int(values[:, index].sum()),
                ]
                for index in range(len(candidate["markers"]))
            }
            for donor, values in states.items()
        },
        "rna_axis_quality": {
            donor: {
                "detection_prevalence": quality[donor]["prevalence"].tolist(),
                "axis_valid": quality[donor]["valid"].tolist(),
            }
            for donor in selected
        },
        "observed_rna_assay_files": observed_files,
        "selection_bridge": bridge_certificate,
        "rna_states": private,
        "held_adt_files_opened": 0,
        "held_adt_numeric_values_read": 0,
        "held_joint_tables_formed": 0,
    }


def _private_artifact(path: Path, certificate: dict[str, Any]) -> dict[str, Any]:
    value = _private_path(path)
    if (
        value.stat().st_size != certificate["bytes"]
        or _sha256(value) != certificate["sha256"]
    ):
        raise PermissionError("private artifact differs from its public certificate")
    return _read_json(value)


def _adt_stage_body(
    scratch_dir: Path, selection_bridge: Path, adt_states: Path
) -> dict[str, Any]:
    rna = _require_completed("rna")
    source = _require_completed("source")
    candidate = _candidate()
    bridge = _private_artifact(selection_bridge, rna["selection_bridge"])
    selected = bridge["selected"]
    assignments = {
        cell: {"donor": donor, "pool_id": bridge["pool_by_cell"][cell]}
        for donor, cells in selected.items()
        for cell in cells
    }
    counts, observed_files = _reduce_modality(
        candidate,
        selected,
        assignments,
        "adt",
        scratch_dir,
        authorized_stage="held_adt",
    )
    _require_reused_assay_hashes_match_source(source, "adt", observed_files)
    states = {
        donor: _adt_states(counts[donor], selected[donor], donor) for donor in selected
    }
    quality = {donor: _adt_axis_quality(counts[donor]) for donor in selected}
    private = _write_private(
        adt_states,
        {
            "schema": "gse185381-private-adt-states/1.0",
            "states": {donor: values.tolist() for donor, values in states.items()},
            "selected_axis_sha256": rna["selected_axis_sha256"],
        },
    )
    return {
        "schema": "gse185381-aml-adt/1.0",
        "status": "HELD_ADT_REDUCED_SEPARATELY_WITHOUT_RNA_STATES",
        "created_at_utc": _timestamp(),
        "rna_sha256": _sha256(DEFAULT_RNA),
        "held_donors": list(selected),
        "marker_panel": candidate["markers"],
        "selected_axis_sha256": rna["selected_axis_sha256"],
        "column_margins": {
            donor: {
                candidate["markers"][index]["rna_symbol"]: [
                    CELL_BUDGET // 2,
                    CELL_BUDGET // 2,
                ]
                for index in range(len(candidate["markers"]))
            }
            for donor in selected
        },
        "adt_axis_quality": {
            donor: {
                "distinct_values": quality[donor]["distinct_values"].tolist(),
                "largest_equal_value_fraction": quality[donor][
                    "largest_equal_value_fraction"
                ].tolist(),
                "axis_valid": quality[donor]["valid"].tolist(),
            }
            for donor in selected
        },
        "observed_adt_assay_files": observed_files,
        "adt_states": private,
        "held_rna_state_artifact_opened": False,
        "held_joint_tables_formed": 0,
    }


def _prediction_stage_body() -> dict[str, Any]:
    source = _require_completed("source")
    rna = _require_completed("rna")
    adt = _require_completed("adt")
    candidate = _candidate()
    donors = [row["donor_id"] for row in candidate["donors"] if row["role"] == "held"]
    if (
        rna["held_donors"] != donors
        or adt["held_donors"] != donors
        or rna["selected_axis_sha256"] != adt["selected_axis_sha256"]
    ):
        raise PermissionError("held RNA and ADT public axes differ")
    markers = candidate["markers"]
    models = source["model"]["models"]
    methods = list(source["model"]["available_methods"])
    locked = source["model"]["locked_classical_method"]
    selected_graph_penalty = source["model"]["primary_selection"][
        "selected_configuration"
    ]["graph_penalty"]
    if (
        methods != list(models)
        or methods[:3] != ["primary", "graph_zero_ablation", "destroyed_link"]
        or locked not in ALL_CLASSICAL_METHODS
        or locked not in methods
        or RAW_RESIDUAL_METHOD not in methods
        or INDEPENDENCE_METHOD not in methods
        or selected_graph_penalty not in GRAPH_GRID
    ):
        raise PermissionError("source method contract differs")
    samples = []
    for donor in donors:
        rows_one = np.asarray(
            [rna["row_margins"][donor][marker["rna_symbol"]] for marker in markers],
            dtype=np.int64,
        )
        columns_one = np.asarray(
            [adt["column_margins"][donor][marker["rna_symbol"]] for marker in markers],
            dtype=np.int64,
        )
        rows = np.repeat(rows_one[:, None, :], len(markers), axis=1)
        columns = np.repeat(columns_one[None, :, :], len(markers), axis=0)
        rna_valid = np.asarray(
            rna["rna_axis_quality"][donor]["axis_valid"], dtype=bool
        )
        adt_valid = np.asarray(
            adt["adt_axis_quality"][donor]["axis_valid"], dtype=bool
        )
        valid_mask = np.outer(rna_valid, adt_valid)
        valid_rna, valid_adt, valid_pairs = _require_axis_support(
            rna_valid,
            adt_valid,
            "HELD_DONOR_FAILS_FROZEN_SUPPORT_CONTRACT",
        )
        estimates = {
            method: _predict_model(models[method], rows, columns)
            for method in methods
        }
        donor_row = next(
            row for row in candidate["donors"] if row["donor_id"] == donor
        )
        samples.append(
            {
                "donor_id": donor,
                "acquisition_cluster": donor_row["acquisition_cluster"],
                "selected_pool_id": donor_row["selected_pool_id"],
                "selected_axis_sha256": rna["selected_axis_sha256"][donor],
                "row_margins": rows.tolist(),
                "column_margins": columns.tolist(),
                "valid_ordered_pair_mask": valid_mask.tolist(),
                "valid_ordered_pair_mask_sha256": _array_sha256(valid_mask),
                "valid_rna_axes": valid_rna,
                "required_valid_rna_axes": MINIMUM_VALID_RNA_AXES,
                "valid_adt_axes": valid_adt,
                "required_valid_adt_axes": MINIMUM_VALID_ADT_AXES,
                "valid_ordered_pairs": valid_pairs,
                "required_valid_ordered_pairs": MINIMUM_VALID_ORDERED_PAIRS,
                "predicted_tables": {
                    method: values.tolist() for method, values in estimates.items()
                },
                "prediction_sha256": {
                    method: _array_sha256(values)
                    for method, values in estimates.items()
                },
            }
        )
    return {
        "schema": "gse185381-aml-predictions/1.0",
        "status": "HELD_PREDICTIONS_FROZEN_BEFORE_RNA_ADT_PAIRING",
        "created_at_utc": _timestamp(),
        "source_sha256": _sha256(DEFAULT_SOURCE),
        "rna_sha256": _sha256(DEFAULT_RNA),
        "adt_sha256": _sha256(DEFAULT_ADT),
        "marker_panel": markers,
        "methods": methods,
        "locked_classical_method": locked,
        "samples": samples,
        "held_private_state_artifacts_opened": 0,
        "held_joint_tables_formed": 0,
    }


def make_score_authorization() -> dict[str, Any]:
    prediction, prediction_commit = _require_completed_stage_artifact(
        PREDICTION_TAG,
        DEFAULT_PREDICTION,
        "prediction",
        require_success=True,
    )
    if DEFAULT_SCORE_AUTHORIZATION.exists():
        raise FileExistsError("score authorization already exists")
    runtime = _require_runtime_environment()
    protocol_commit = _require_public_tag(PROTOCOL_TAG, PROTOCOL_BINDINGS)
    payload = {
        "schema": "gse185381-aml-score-authorization/1.0",
        "status": "AUTHORIZED_AFTER_PUBLIC_PREDICTION_FREEZE",
        "created_at_utc": _timestamp(),
        "prediction_tag": PREDICTION_TAG,
        "prediction_commit": prediction_commit,
        "prediction_sha256": _sha256(DEFAULT_PREDICTION),
        "prediction_payload_sha256": _canonical_json_sha256(prediction),
        "held_joint_tables_formed_before_authorization": 0,
        "held_rna_and_adt_states_opened_together_before_authorization": False,
        "required_public_tag": SCORE_AUTHORIZATION_TAG,
        "protocol_commit": protocol_commit,
        "runtime_environment": runtime,
    }
    _validate_public_payload(payload)
    _write_json_x(DEFAULT_SCORE_AUTHORIZATION, payload)
    return payload


def _require_score_authorization() -> dict[str, Any]:
    value = _read_json(DEFAULT_SCORE_AUTHORIZATION)
    prediction = _read_json(DEFAULT_PREDICTION)
    expected = {
        "schema": "gse185381-aml-score-authorization/1.0",
        "status": "AUTHORIZED_AFTER_PUBLIC_PREDICTION_FREEZE",
        "prediction_tag": PREDICTION_TAG,
        "prediction_sha256": _sha256(DEFAULT_PREDICTION),
        "prediction_payload_sha256": _canonical_json_sha256(prediction),
        "held_joint_tables_formed_before_authorization": 0,
        "held_rna_and_adt_states_opened_together_before_authorization": False,
        "required_public_tag": SCORE_AUTHORIZATION_TAG,
        "protocol_commit": _require_public_tag(PROTOCOL_TAG, PROTOCOL_BINDINGS),
        "runtime_environment": _require_runtime_environment(),
    }
    _, commit = _require_completed_stage_artifact(
        PREDICTION_TAG,
        DEFAULT_PREDICTION,
        "prediction",
        require_success=True,
    )
    expected["prediction_commit"] = commit
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise PermissionError(f"score authorization differs at {key}")
    _require_public_tag(
        SCORE_AUTHORIZATION_TAG, (_relative(DEFAULT_SCORE_AUTHORIZATION),)
    )
    _validate_public_payload(value)
    return value


def _cluster_bootstrap(
    primary: np.ndarray, comparator: np.ndarray, clusters: list[str]
) -> dict[str, Any]:
    first = np.asarray(primary, dtype=float)
    second = np.asarray(comparator, dtype=float)
    if first.shape != second.shape or first.ndim != 1 or len(clusters) != len(first):
        raise ValueError("paired losses and cluster axis differ")
    axis = np.asarray(clusters, dtype=object)
    labels = sorted(set(clusters))
    groups = [np.flatnonzero(axis == label) for label in labels]
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    differences = np.empty(BOOTSTRAPS)
    reductions = np.empty(BOOTSTRAPS)
    for draw in range(BOOTSTRAPS):
        sampled_groups = generator.integers(0, len(groups), size=len(groups))
        sampled = np.concatenate(
            [
                groups[index][
                    generator.integers(
                        0, len(groups[index]), size=len(groups[index])
                    )
                ]
                for index in sampled_groups
            ]
        )
        primary_mean = float(first[sampled].mean())
        comparator_mean = float(second[sampled].mean())
        if comparator_mean <= 0.0:
            raise FloatingPointError("bootstrap comparator mean is not positive")
        differences[draw] = primary_mean - comparator_mean
        reductions[draw] = 1.0 - primary_mean / comparator_mean
    quantiles = [0.0125, 0.9875]
    return {
        "draws": BOOTSTRAPS,
        "seed": BOOTSTRAP_SEED,
        "resampling_unit": "acquisition clusters, then donors within each sampled cluster at its original cluster size",
        "mean_difference_97_5_percent_interval": np.quantile(
            differences, quantiles, method="linear"
        ).tolist(),
        "mean_difference_98_75th_percentile": float(
            np.quantile(differences, 0.9875, method="linear")
        ),
        "relative_loss_reduction_97_5_percent_interval": np.quantile(
            reductions, quantiles, method="linear"
        ).tolist(),
        "relative_loss_reduction_98_75th_percentile": float(
            np.quantile(reductions, 0.9875, method="linear")
        ),
    }


def _exact_donor_sign_p(values: np.ndarray) -> dict[str, Any]:
    differences = np.asarray(values, dtype=float)
    nonzero = differences[differences != 0.0]
    favorable = int(np.count_nonzero(nonzero < 0.0))
    n = len(nonzero)
    p = (
        sum(math.comb(n, value) for value in range(favorable, n + 1)) / (2**n)
        if n
        else 1.0
    )
    return {"nonzero_donors": n, "favorable_donors": favorable, "one_sided_p": p}


def _exact_cluster_sign_flip(values: np.ndarray, clusters: list[str]) -> dict[str, Any]:
    differences = np.asarray(values, dtype=float)
    axis = np.asarray(clusters, dtype=object)
    labels = sorted(set(clusters))
    if len(labels) > 22:
        raise ConfirmationRefusal(
            "MORE_THAN_22_ACQUISITION_CLUSTERS_PRECLUDES_EXACT_SIGN_FLIP"
        )
    sums = np.asarray([differences[axis == label].sum() for label in labels])
    observed = float(differences.mean())
    favorable = 0
    draws = 1 << len(labels)
    for start in range(0, draws, 65536):
        integers = np.arange(start, min(start + 65536, draws), dtype=np.uint64)[:, None]
        bits = ((integers >> np.arange(len(labels), dtype=np.uint64)) & 1).astype(float)
        statistics = ((1.0 - 2.0 * bits) * sums).sum(axis=1) / len(differences)
        favorable += int(np.count_nonzero(statistics <= observed + 1e-15))
    return {
        "method": "exact",
        "acquisition_clusters": len(labels),
        "draws": draws,
        "observed_mean_difference": observed,
        "one_sided_p": favorable / draws,
    }


def _comparison(
    donors: list[str],
    clusters: list[str],
    primary: np.ndarray,
    comparator: np.ndarray,
) -> dict[str, Any]:
    first = np.asarray(primary, dtype=float)
    second = np.asarray(comparator, dtype=float)
    difference = first - second
    comparator_mean = float(second.mean())
    reduction = (
        1.0 - float(first.mean()) / comparator_mean if comparator_mean > 0.0 else None
    )
    favorable = int(np.count_nonzero(difference < 0.0))
    required = math.ceil(0.8 * len(donors))
    bootstrap = _cluster_bootstrap(first, second, clusters)
    sign = _exact_donor_sign_p(difference)
    cluster_sign = _exact_cluster_sign_flip(difference, clusters)
    axis = np.asarray(clusters, dtype=object)
    leave_one_out = {
        cluster: float(difference[axis != cluster].mean())
        for cluster in sorted(set(clusters))
    }
    checks = {
        "relative_loss_reduction_at_least_five_percent": reduction is not None
        and reduction >= 0.05,
        "cluster_bootstrap_mean_difference_upper_98_75_below_zero": bootstrap[
            "mean_difference_98_75th_percentile"
        ]
        < 0.0,
        "favorable_donor_fraction_at_least_0_8": favorable >= required,
        "exact_donor_sign_one_sided_p_at_most_0_0125": sign["one_sided_p"]
        <= 0.0125,
        "exact_acquisition_cluster_sign_flip_one_sided_p_at_most_0_0125": cluster_sign[
            "one_sided_p"
        ]
        <= 0.0125,
        "every_leave_one_cluster_out_mean_negative": all(
            value < 0.0 for value in leave_one_out.values()
        ),
    }
    return {
        "donors": len(donors),
        "primary_mean_loss": float(first.mean()),
        "comparator_mean_loss": comparator_mean,
        "relative_loss_reduction": reduction,
        "mean_paired_difference": float(difference.mean()),
        "acquisition_cluster_bootstrap": bootstrap,
        "favorable_donors": favorable,
        "required_favorable_donors": required,
        "exact_donor_sign_test": sign,
        "exact_acquisition_cluster_sign_flip": cluster_sign,
        "leave_one_cluster_out_mean_differences": leave_one_out,
        "checks": checks,
        "passes": all(checks.values()),
        "donor_differences": dict(zip(donors, map(float, difference))),
    }


def _score_stage_body(rna_states_path: Path, adt_states_path: Path) -> dict[str, Any]:
    prediction = _require_completed("prediction")
    source = _require_completed("source")
    rna_public = _require_completed("rna")
    adt_public = _require_completed("adt")
    _require_score_authorization()
    rna_private = _private_artifact(rna_states_path, rna_public["rna_states"])
    adt_private = _private_artifact(adt_states_path, adt_public["adt_states"])
    if (
        rna_private.get("schema") != "gse185381-private-rna-states/1.0"
        or adt_private.get("schema") != "gse185381-private-adt-states/1.0"
    ):
        raise PermissionError("private state schema differs")
    candidate = _candidate()
    donors = [row["donor_id"] for row in candidate["donors"] if row["role"] == "held"]
    cluster_by_donor = {
        row["donor_id"]: row["acquisition_cluster"]
        for row in candidate["donors"]
        if row["role"] == "held"
    }
    frozen = {row["donor_id"]: row for row in prediction["samples"]}
    methods = prediction["methods"]
    locked = source["model"]["locked_classical_method"]
    if (
        set(frozen) != set(donors)
        or prediction.get("locked_classical_method") != locked
        or locked not in methods
        or "primary" not in methods
        or RAW_RESIDUAL_METHOD not in methods
        or INDEPENDENCE_METHOD not in methods
    ):
        raise PermissionError("held prediction method or donor contract differs")
    losses = {
        "primary": np.empty(len(donors)),
        locked: np.empty(len(donors)),
    }
    truths: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    samples = []
    for donor_index, donor in enumerate(donors):
        rna_states = np.asarray(rna_private["states"][donor], dtype=np.uint8)
        adt_states = np.asarray(adt_private["states"][donor], dtype=np.uint8)
        truth = _binary_tables(rna_states, adt_states)
        truths[donor] = truth
        rows, columns = _margins(truth)
        record = frozen[donor]
        if (
            rows.tolist() != record["row_margins"]
            or columns.tolist() != record["column_margins"]
        ):
            raise PermissionError("held truth margins differ from frozen prediction")
        mask = np.asarray(record["valid_ordered_pair_mask"], dtype=bool)
        rna_valid = np.asarray(
            rna_public["rna_axis_quality"][donor]["axis_valid"], dtype=bool
        )
        adt_valid = np.asarray(
            adt_public["adt_axis_quality"][donor]["axis_valid"], dtype=bool
        )
        expected_rna = int(np.count_nonzero(rna_valid))
        expected_adt = int(np.count_nonzero(adt_valid))
        expected_pairs = expected_rna * expected_adt
        if (
            _array_sha256(mask) != record["valid_ordered_pair_mask_sha256"]
            or record.get("valid_rna_axes") != expected_rna
            or record.get("required_valid_rna_axes") != MINIMUM_VALID_RNA_AXES
            or record.get("valid_adt_axes") != expected_adt
            or record.get("required_valid_adt_axes") != MINIMUM_VALID_ADT_AXES
            or record.get("valid_ordered_pairs") != expected_pairs
            or record.get("required_valid_ordered_pairs")
            != MINIMUM_VALID_ORDERED_PAIRS
            or expected_rna < MINIMUM_VALID_RNA_AXES
            or expected_adt < MINIMUM_VALID_ADT_AXES
            or expected_pairs < MINIMUM_VALID_ORDERED_PAIRS
        ):
            raise PermissionError("frozen support contract differs")
        expected_mask = np.outer(rna_valid, adt_valid)
        if not np.array_equal(mask, expected_mask):
            raise PermissionError("frozen valid mask differs from separate axis QC")
        masks[donor] = mask
        donor_losses = {}
        for method in ("primary", locked):
            estimate = np.asarray(record["predicted_tables"][method], dtype=float)
            if _array_sha256(estimate) != record["prediction_sha256"][method]:
                raise PermissionError("frozen prediction hash differs")
            loss = _donor_loss(truth, estimate, mask)
            losses[method][donor_index] = loss
            donor_losses[method] = float(loss)
        samples.append(
            {
                "donor_id": donor,
                "acquisition_cluster": cluster_by_donor[donor],
                "truth_table_sha256": _array_sha256(truth),
                "valid_ordered_pairs": int(mask.sum()),
                "losses": donor_losses,
            }
        )
    def score_method(method: str) -> None:
        if method in losses:
            return
        if method not in methods:
            raise PermissionError("required held prediction method is absent")
        values = np.empty(len(donors), dtype=float)
        for donor_index, donor in enumerate(donors):
            record = frozen[donor]
            estimate = np.asarray(record["predicted_tables"][method], dtype=float)
            if _array_sha256(estimate) != record["prediction_sha256"][method]:
                raise PermissionError("frozen secondary prediction hash differs")
            loss = _donor_loss(truths[donor], estimate, masks[donor])
            values[donor_index] = loss
            samples[donor_index]["losses"][method] = float(loss)
        losses[method] = values

    score_method(RAW_RESIDUAL_METHOD)
    score_method(INDEPENDENCE_METHOD)
    cluster_axis = [cluster_by_donor[donor] for donor in donors]
    primary = losses["primary"]
    primary_comparison = _comparison(
        donors,
        cluster_axis,
        primary,
        losses[locked],
    )
    raw_poisson_comparison = (
        primary_comparison
        if locked == RAW_RESIDUAL_METHOD
        else _comparison(
            donors,
            cluster_axis,
            primary,
            losses[RAW_RESIDUAL_METHOD],
        )
    )
    independence_comparison = _comparison(
        donors,
        cluster_axis,
        primary,
        losses[INDEPENDENCE_METHOD],
    )
    passed = primary_comparison["passes"]
    classical_comparisons: dict[str, Any] = {
        RAW_RESIDUAL_METHOD: raw_poisson_comparison
    }
    graph_zero: dict[str, Any] = {
        "status": "NOT_EVALUATED_ESTIMATOR_GATE_FAILED",
        "serial_gate_position": 2,
    }
    destroyed: dict[str, Any] = {
        "status": "NOT_EVALUATED_ESTIMATOR_GATE_FAILED",
        "serial_gate_position": 3,
    }
    if passed:
        for method in methods:
            if method != "destroyed_link":
                score_method(method)
        classical_comparisons = {
            method: (
                primary_comparison
                if method == locked
                else _comparison(donors, cluster_axis, primary, losses[method])
            )
            for method in ALL_CLASSICAL_METHODS
            if method in losses
        }
        graph_zero = {
            **_comparison(
                donors, cluster_axis, primary, losses["graph_zero_ablation"]
            ),
            "status": "EVALUATED_AFTER_ESTIMATOR_GATE_PASS",
            "serial_gate_position": 2,
        }
        if graph_zero["passes"]:
            score_method("destroyed_link")
            destroyed = {
                **_comparison(
                    donors, cluster_axis, primary, losses["destroyed_link"]
                ),
                "status": "EVALUATED_AFTER_GRAPH_STRUCTURE_GATE_PASS",
                "serial_gate_position": 3,
            }
        else:
            destroyed = {
                "status": "NOT_EVALUATED_GRAPH_STRUCTURE_GATE_FAILED",
                "serial_gate_position": 3,
            }
    selected_graph_penalty = float(
        source["model"]["primary_selection"]["selected_configuration"][
            "graph_penalty"
        ]
    )
    broad_classical_support = bool(
        passed
        and classical_comparisons
        and all(row["passes"] for row in classical_comparisons.values())
    )
    graph_structure_support = bool(
        passed
        and selected_graph_penalty > 0.0
        and graph_zero.get("passes", False)
    )
    coupling_link_support = bool(
        graph_structure_support and destroyed.get("passes", False)
    )
    return {
        "schema": "gse185381-aml-held-validation/1.0",
        "status": (
            "COMPLETED_ESTIMATOR_GATE_PASS"
            if passed
            else "COMPLETED_ESTIMATOR_GATE_FAIL_PRIMARY_VS_LOCKED_CLASSICAL"
        ),
        "created_at_utc": _timestamp(),
        "prediction_sha256": _sha256(DEFAULT_PREDICTION),
        "score_authorization_sha256": _sha256(DEFAULT_SCORE_AUTHORIZATION),
        "locked_classical_method": locked,
        "selected_graph_penalty": selected_graph_penalty,
        "primary_vs_locked_classical": primary_comparison,
        "primary_vs_untuned_poisson_residual": raw_poisson_comparison,
        "confirmatory_gate_order": [
            "primary_vs_locked_classical",
            "primary_vs_graph_zero_ablation",
            "primary_vs_destroyed_link",
        ],
        "classical_head_to_head": classical_comparisons,
        "classical_head_to_head_inferential_role": "Only the source-locked comparator is the primary confirmatory gate; other classical contrasts are descriptive.",
        "graph_zero_serial_secondary": graph_zero,
        "destroyed_link_serial_secondary": destroyed,
        "target_margin_independence_head_to_head": independence_comparison,
        "target_margin_independence_inferential_role": "descriptive outside the serial confirmatory chain",
        "estimator_validation_supported": bool(passed),
        "untuned_poisson_residual_descriptive_gain": bool(
            raw_poisson_comparison["passes"]
        ),
        "target_margin_independence_descriptive_gain": bool(
            independence_comparison["passes"]
        ),
        "broad_classical_descriptive_gain": broad_classical_support,
        "graph_structure_supported": graph_structure_support,
        "coupling_link_supported": coupling_link_support,
        "structured_coupling_field_supported": bool(
            graph_structure_support and coupling_link_support
        ),
        "samples": samples,
        "outcome_failure_rule": "A completed score that misses a criterion is a negative result, not a QC refusal.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    claim = commands.add_parser("claim")
    claim.add_argument("stage", choices=tuple(STAGE_PATHS))
    source = commands.add_parser("run-source")
    source.add_argument("--scratch-dir", type=Path, required=True)
    source.add_argument("--metadata-dir", type=Path, required=True)
    rna = commands.add_parser("run-rna")
    rna.add_argument("--scratch-dir", type=Path, required=True)
    rna.add_argument("--metadata-dir", type=Path, required=True)
    rna.add_argument("--selection-bridge", type=Path, required=True)
    rna.add_argument("--rna-states", type=Path, required=True)
    adt = commands.add_parser("run-adt")
    adt.add_argument("--scratch-dir", type=Path, required=True)
    adt.add_argument("--selection-bridge", type=Path, required=True)
    adt.add_argument("--adt-states", type=Path, required=True)
    commands.add_parser("run-prediction")
    commands.add_parser("make-score-authorization")
    score = commands.add_parser("run-score")
    score.add_argument("--rna-states", type=Path, required=True)
    score.add_argument("--adt-states", type=Path, required=True)
    verify = commands.add_parser("verify-stage")
    verify.add_argument("stage", choices=tuple(STAGE_PATHS))
    commands.add_parser("verify-result")
    args = parser.parse_args()
    if args.command == "claim":
        payload = claim_stage(args.stage)
    elif args.command == "run-source":
        payload = _run_claimed_stage(
            "source", lambda: _source_stage_body(args.scratch_dir, args.metadata_dir)
        )
    elif args.command == "run-rna":
        payload = _run_claimed_stage(
            "rna",
            lambda: _rna_stage_body(
                args.scratch_dir,
                args.metadata_dir,
                args.selection_bridge,
                args.rna_states,
            ),
        )
    elif args.command == "run-adt":
        payload = _run_claimed_stage(
            "adt",
            lambda: _adt_stage_body(
                args.scratch_dir, args.selection_bridge, args.adt_states
            ),
        )
    elif args.command == "run-prediction":
        payload = _run_claimed_stage("prediction", _prediction_stage_body)
    elif args.command == "make-score-authorization":
        payload = make_score_authorization()
    elif args.command == "run-score":
        payload = _run_claimed_stage(
            "score", lambda: _score_stage_body(args.rna_states, args.adt_states)
        )
    elif args.command == "verify-stage":
        payload = verify_stage(args.stage)
    else:
        payload = verify_public_result()
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
