"""Sealed COMBAT CITE-seq held-donor and held-site confirmation.

The four CLI phases keep model selection, held-margin prediction, and held
pairing access separate.  The source manifest enumerates the 97 eligible
samples and binds the local H5AD; this module never infers the confirmatory
cohort from assay values.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import unquote, urlparse
import urllib.request

import h5py
import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln, logsumexp

from experiments import evaluate_gse279451_sepsis_development as numerics
from mapreg.heterogeneity_adaptive_coupling import (
    centered_classical_coordinate,
    centered_haldane_log_odds,
    fit_heterogeneity_adaptive_binary_coupling,
    paule_mandel_pool,
    signed_deviance_coordinate,
    signed_pearson_coordinate,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_MANIFEST = (
    ROOT / "data/confirmation/combat_citeseq/source_manifest_v1.json"
)
DEFAULT_REDUCED = ROOT / "data/development/combat_citeseq/reduced_v1.json"
DEFAULT_PILOT = ROOT / "results/development/combat_citeseq_development.json"
DEFAULT_PREDICTION = ROOT / "results/combat_citeseq_predictions.json"
DEFAULT_DEVELOPMENT_AUTHORIZATION = (
    ROOT / "data/confirmation/combat_citeseq/development_authorization_v1.json"
)
DEFAULT_MARGIN_AUTHORIZATION = (
    ROOT / "data/confirmation/combat_citeseq/held_rna_margin_authorization_v1.json"
)
DEFAULT_PREDICTION_ATTEMPT = (
    ROOT / "data/confirmation/combat_citeseq/prediction_attempt_v1.json"
)
DEFAULT_AUTHORIZATION = (
    ROOT / "data/confirmation/combat_citeseq/score_authorization_v1.json"
)
DEFAULT_SCORE = ROOT / "results/combat_citeseq_confirmation.json"
DEFAULT_SCORE_ATTEMPT = ROOT / "data/confirmation/combat_citeseq/score_attempt_v1.json"
DEFAULT_TERMINAL_REFUSAL = ROOT / "results/combat_citeseq_terminal_refusal.json"

MARKERS = ("CD4", "CD7", "CD14", "CD19", "CD33", "CD38", "CD44", "CD47", "CD52")
RNA_ENSEMBL = {
    "CD4": "ENSG00000010610",
    "CD7": "ENSG00000173762",
    "CD14": "ENSG00000170458",
    "CD19": "ENSG00000177455",
    "CD33": "ENSG00000105383",
    "CD38": "ENSG00000004468",
    "CD44": "ENSG00000026508",
    "CD47": "ENSG00000196776",
    "CD52": "ENSG00000169442",
}
ADT_FEATURE = {marker: f"AB_{marker}" for marker in MARKERS}
ADT_FEATURE["CD44"] = "AB_humanCD44"

CELL_TYPES = ("B", "ERYTH", "HSC", "MNP", "NK", "PB", "PLT", "T")
CELL_BUDGET = 512
CELL_SELECTION_SALT = "COMBAT-PBMC-CELL-BUDGET-v1"
ADT_TIE_SALT = "COMBAT-PBMC-ADT-v1"
DESTROYED_LINK_SALT = "COMBAT-DESTROYED-LINK-v1"
LABEL_PERMUTATION_SALT = "COMBAT-LABEL-PERMUTATION-v1"
CALIBRATION_SALT = "COMBAT-OXFORD-CALIBRATION-v1"
PILOT_SALT = "COMBAT-OXFORD-PILOT-v1"

CALIBRATION_IDS = (
    "S00024",
    "S00027",
    "G05077",
    "G05171",
    "S00002",
    "S00126",
    "S00045",
    "S00148",
    "H00052",
    "H00054",
    "N00032",
    "N00050",
)
PILOT_IDS = (
    "S00008",
    "S00020",
    "S00040",
    "S00052",
    "G05061",
    "G05097",
    "G05145",
    "G05164",
    "S00063",
    "S00076",
    "S00104",
    "S00114",
    "S00037",
    "S00042",
    "S00053",
    "S00134",
    "H00058",
    "H00064",
    "H00070",
    "H00072",
    "N00006",
    "N00024",
    "N00025",
    "N00047",
)

EXPECTED_SAMPLES = 97
EXPECTED_OXFORD = 87
EXPECTED_ST_GEORGES = 10
EXPECTED_SOURCE_STRATA = 6
EXPECTED_HELD_DONOR = 51
EXPECTED_HELD_SITE = 10
FROZEN_SAMPLE_UNIVERSE_SHA256 = (
    "ccbdc16f9c78178744075ea7859d581195abcb03963c5c96d3b1e97e324e3bcf"
)
CONFIG_GRID = tuple(itertools.product((1, 2), (0.01, 0.1), (0.1, 1.0)))
CLASSICAL_GRID = tuple(itertools.product(("pearson", "deviance"), (False, True)))
METHODS = (
    "primary",
    "best_residual",
    "raw_pm_haldane",
    "unstructured_centered_pm",
    "destroyed_link",
    "label_permuted_graph",
    "independence",
)
PROMOTION_COMPARATORS = ("best_residual", "destroyed_link")

BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 20260828
MAXIMUM_CONDITION_NUMBER = 1e12
MINIMUM_INFORMATIVE_ENTITIES = 64
OFFICIAL_H5AD_BYTES = 6_409_089_483
OFFICIAL_H5AD_MD5 = "87c6b1a733ea1adc37c808d9a357de74"
OFFICIAL_H5AD_SHA256 = (
    "f628a15f25b9ca2f7cdefeab271fd9d007c8d5e47eb80c4b807d5f65e86ff53d"
)
OFFICIAL_COMPOSITION_SHA256 = (
    "2ad7e92ab122ee52986d5748dbb23c335c02ec1f1f244943ce46fff94c585157"
)
PUBLIC_GITHUB_OWNER = "sushaan-k"
PUBLIC_GITHUB_REPOSITORY = "coupling-fields-benchmark"
PUBLIC_GITHUB_ORIGIN = (
    f"https://github.com/{PUBLIC_GITHUB_OWNER}/{PUBLIC_GITHUB_REPOSITORY}.git"
)

DEVELOPMENT_BINDING_PATHS = {
    "runner": "experiments/confirm_combat_citeseq.py",
    "runner_test": "tests/test_combat_citeseq_confirmation.py",
    "protocol": "docs/COMBAT_CITESEQ_HELD_CONFIRMATION_PROTOCOL_2026-08-28.md",
    "designation": "data/confirmation/combat_citeseq/candidate_designation_v1.json",
    "source_manifest": "data/confirmation/combat_citeseq/source_manifest_v1.json",
    "metadata_preflight_script": "experiments/preflight_combat_citeseq.py",
    "metadata_preflight_result": (
        "results/development/combat_citeseq_metadata_preflight.json"
    ),
    "fresh_clone_verification": (
        "docs/COMBAT_CITESEQ_PUBLIC_FREEZE_VERIFICATION_2026-08-28.json"
    ),
    "gse279451_evaluator": "experiments/evaluate_gse279451_sepsis_development.py",
    "gse279451_reducer": "experiments/reduce_gse279451_sepsis.py",
    "heterogeneity_adaptive_coupling": "mapreg/heterogeneity_adaptive_coupling.py",
    "hierarchical_conditional_coupling": (
        "mapreg/hierarchical_conditional_coupling.py"
    ),
    "coupling_fields": "mapreg/coupling_fields.py",
    "classical_residuals": "mapreg/classical_residuals.py",
    "table_prediction": "mapreg/table_prediction.py",
}


def _timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _is_utc_timestamp(value: Any) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", value
        )
        is not None
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _opaque_hashes(path: Path) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            sha256.update(block)
            md5.update(block)
    return sha256.hexdigest(), md5.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    def reject(token: str) -> None:
        raise ValueError(f"non-finite JSON number: {token}")

    payload = json.loads(path.read_text(), parse_constant=reject)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _write_json(
    path: Path, payload: dict[str, Any], *, exclusive: bool = False
) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode) as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_designated_paths(
    phase: str, paths: tuple[tuple[str, Path, Path], ...]
) -> None:
    for label, observed, expected in paths:
        if observed.resolve() != expected.resolve():
            raise PermissionError(f"{phase} {label} path differs from designation")


def _sanitized_error(phase: str, error: Exception) -> str:
    reason = {
        PermissionError: "authorization or frozen-artifact validation failed",
        FileExistsError: "an exclusive phase artifact already exists",
        TimeoutError: "an authorized subprocess timed out",
        ValueError: "a frozen data or numerical contract failed",
        FloatingPointError: "a numerical contract failed",
        RuntimeError: "an authorized runtime step failed",
    }.get(type(error), "an unexpected runtime step failed")
    return f"{phase}: {reason}"


def _run_terminal_phase(
    phase: str,
    attempt_path: Path,
    authorization_path: Path,
    refusal_path: Path,
    operation: Any,
) -> dict[str, Any]:
    if phase not in {"held_prediction", "held_score"}:
        raise ValueError("unknown terminal phase")
    expected_attempt = (
        DEFAULT_PREDICTION_ATTEMPT
        if phase == "held_prediction"
        else DEFAULT_SCORE_ATTEMPT
    )
    expected_authorization = (
        DEFAULT_MARGIN_AUTHORIZATION
        if phase == "held_prediction"
        else DEFAULT_AUTHORIZATION
    )
    _require_designated_paths(
        phase,
        (
            ("attempt", attempt_path, expected_attempt),
            ("authorization", authorization_path, expected_authorization),
            ("terminal refusal", refusal_path, DEFAULT_TERMINAL_REFUSAL),
        ),
    )
    if refusal_path.exists():
        raise FileExistsError("terminal refusal already exists; reruns are forbidden")
    try:
        return operation()
    except Exception as error:
        if attempt_path.is_file():
            payload = {
                "schema": "combat-citeseq-terminal-refusal/1.0",
                "status": "TERMINAL_REFUSAL",
                "created_at_utc": _timestamp(),
                "phase": phase,
                "error_type": type(error).__name__,
                "reason": _sanitized_error(phase, error),
                "attempt_path": _relative(attempt_path),
                "attempt_sha256": _sha256(attempt_path),
                "authorization_path": _relative(authorization_path),
                "authorization_sha256": (
                    _sha256(authorization_path)
                    if authorization_path.is_file()
                    else None
                ),
                "runner_sha256": _sha256(Path(__file__)),
                "rerun_permitted": False,
            }
            _write_json(refusal_path, payload, exclusive=True)
        raise


def _sample_value(record: dict[str, Any], canonical: str, official: str) -> str:
    value = record.get(canonical, record.get(official))
    if not isinstance(value, str) or not value:
        raise ValueError(f"sample record lacks {canonical}/{official}")
    return value


def _sample_records(source: dict[str, Any]) -> list[dict[str, str]]:
    raw = source.get("samples")
    if not isinstance(raw, list) or len(raw) != EXPECTED_SAMPLES:
        raise ValueError("source manifest must enumerate exactly 97 samples")
    records = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("every sample manifest row must be an object")
        records.append(
            {
                "sample": _sample_value(item, "sample", "scRNASeq_sample_ID"),
                "combat_id": _sample_value(item, "combat_id", "COMBAT_ID"),
                "source": _sample_value(item, "source", "Source"),
                "institute": _sample_value(item, "institute", "Institute"),
            }
        )
    samples = [record["sample"] for record in records]
    combat_ids = [record["combat_id"] for record in records]
    if len(set(samples)) != len(samples) or len(set(combat_ids)) != len(combat_ids):
        raise ValueError("source-manifest sample and COMBAT IDs must each be unique")
    universe = {
        record["combat_id"]: {
            "sample": record["sample"],
            "source": record["source"],
            "institute": record["institute"],
        }
        for record in records
    }
    if _canonical_json_sha256(universe) != FROZEN_SAMPLE_UNIVERSE_SHA256:
        raise PermissionError("source-manifest sample universe differs from freeze")
    return records


def _role_hash(salt: str, record: dict[str, str]) -> str:
    payload = "\0".join((salt, record["source"], record["combat_id"], record["sample"]))
    return hashlib.sha256(payload.encode()).hexdigest()


def assign_roles(records: list[dict[str, str]]) -> dict[str, str]:
    """Derive the frozen split from official metadata only."""

    oxford = [record for record in records if record["institute"] == "Oxford"]
    held_site = [record for record in records if record["institute"] == "St_Georges"]
    if len(oxford) != EXPECTED_OXFORD or len(held_site) != EXPECTED_ST_GEORGES:
        raise ValueError("manifest must contain 87 Oxford and 10 St_Georges samples")
    if len(oxford) + len(held_site) != len(records):
        raise ValueError("an unexpected Institute entered the frozen cohort")
    strata = sorted({record["source"] for record in oxford})
    if len(strata) != EXPECTED_SOURCE_STRATA:
        raise ValueError("Oxford samples must span exactly six Source strata")

    calibration: list[dict[str, str]] = []
    pilot: list[dict[str, str]] = []
    for source_name in strata:
        group = [record for record in oxford if record["source"] == source_name]
        if len(group) < 6:
            raise ValueError("every Oxford Source stratum needs at least six samples")
        first = sorted(
            group,
            key=lambda record: (_role_hash(CALIBRATION_SALT, record), record["sample"]),
        )
        calibration.extend(first[:2])
        remaining = first[2:]
        second = sorted(
            remaining,
            key=lambda record: (_role_hash(PILOT_SALT, record), record["sample"]),
        )
        pilot.extend(second[:4])

    if {record["combat_id"] for record in calibration} != set(CALIBRATION_IDS):
        raise PermissionError(
            "metadata-derived calibration COMBAT IDs differ from freeze"
        )
    if {record["combat_id"] for record in pilot} != set(PILOT_IDS):
        raise PermissionError("metadata-derived pilot COMBAT IDs differ from freeze")
    calibration_samples = {record["sample"] for record in calibration}
    pilot_samples = {record["sample"] for record in pilot}
    if calibration_samples & pilot_samples:
        raise AssertionError("calibration and pilot roles overlap")

    roles = {record["sample"]: "held_donor" for record in oxford}
    roles.update({record["sample"]: "held_site" for record in held_site})
    roles.update({sample: "calibration" for sample in calibration_samples})
    roles.update({sample: "pilot" for sample in pilot_samples})
    counts = {
        role: sum(value == role for value in roles.values())
        for role in set(roles.values())
    }
    expected = {
        "calibration": 12,
        "pilot": 24,
        "held_donor": EXPECTED_HELD_DONOR,
        "held_site": EXPECTED_HELD_SITE,
    }
    if counts != expected:
        raise AssertionError(f"role counts differ from freeze: {counts}")
    return roles


def _samples_for_ids(
    records: list[dict[str, str]], combat_ids: tuple[str, ...]
) -> tuple[str, ...]:
    by_id = {record["combat_id"]: record["sample"] for record in records}
    if any(combat_id not in by_id for combat_id in combat_ids):
        raise PermissionError("a frozen COMBAT ID is absent from source manifest")
    return tuple(by_id[combat_id] for combat_id in combat_ids)


def _resolved_h5ad(source_path: Path, source: dict[str, Any]) -> Path:
    record = source.get("h5ad")
    if not isinstance(record, dict):
        raise ValueError("source manifest lacks h5ad binding")
    runtime_path = os.environ.get("COMBAT_CITESEQ_H5AD")
    raw_path = runtime_path if runtime_path else record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(
            "set COMBAT_CITESEQ_H5AD; the public source manifest has no local path"
        )
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (source_path.parent / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_bytes = record.get("bytes")
    expected_hash = record.get("sha256")
    expected_md5 = record.get("md5")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes != OFFICIAL_H5AD_BYTES
        or path.stat().st_size != expected_bytes
    ):
        raise ValueError("H5AD byte count differs from source manifest")
    if expected_hash != OFFICIAL_H5AD_SHA256 or expected_md5 != OFFICIAL_H5AD_MD5:
        raise ValueError("source manifest differs from the frozen H5AD digests")
    filename = record.get("filename")
    if isinstance(filename, str) and path.name != filename:
        raise ValueError("runtime H5AD filename differs from source manifest")
    return path


def _validated_source(source_path: Path, *, verify_hash: bool) -> dict[str, Any]:
    source = _read_json(source_path)
    if (
        source.get("schema") != "combat-citeseq-source-manifest/1.0"
        or source.get("status") != "SOURCE_SEALED"
    ):
        raise ValueError("source manifest schema differs from version 1.0")
    records = _sample_records(source)
    roles = assign_roles(records)
    expected_markers = [
        {
            "marker": marker,
            "rna_ensembl": RNA_ENSEMBL[marker],
            "adt_feature": ADT_FEATURE[marker],
        }
        for marker in MARKERS
    ]
    h5ad_record = source.get("h5ad")
    if (
        not isinstance(h5ad_record, dict)
        or set(h5ad_record) != {"filename", "bytes", "md5", "sha256", "raw_matrix_path"}
        or h5ad_record.get("raw_matrix_path") != "layers/raw"
        or source.get("markers") != expected_markers
        or source.get("eligible_cell_types") != list(CELL_TYPES)
    ):
        raise PermissionError("source assay contract differs from freeze")
    raw_samples = source.get("samples")
    for item, record in zip(raw_samples, records):
        if (
            set(item)
            != {
                "combat_id",
                "sample",
                "source",
                "institute",
                "role",
                "eligible_pool_cells",
                "official_total_pbmc_count",
            }
            or item.get("role") != roles[record["sample"]]
            or isinstance(item.get("eligible_pool_cells"), bool)
            or not isinstance(item.get("eligible_pool_cells"), int)
            or item["eligible_pool_cells"] < CELL_BUDGET
            or isinstance(item.get("official_total_pbmc_count"), bool)
            or not isinstance(item.get("official_total_pbmc_count"), int)
            or item["official_total_pbmc_count"] <= 0
        ):
            raise PermissionError("source sample pool contract differs from freeze")
    h5ad = _resolved_h5ad(source_path, source)
    if verify_hash:
        sha256, md5 = _opaque_hashes(h5ad)
        if sha256 != source["h5ad"]["sha256"] or md5 != source["h5ad"]["md5"]:
            raise ValueError("H5AD digests differ from source manifest")
    return {
        "payload": source,
        "records": records,
        "roles": roles,
        "h5ad": h5ad,
        "source_manifest_sha256": _sha256(source_path),
        "h5ad_sha256": source["h5ad"]["sha256"],
    }


def seal_source(preflight_path: Path, output_path: Path) -> dict[str, Any]:
    """Convert a metadata-only PASS into the path-free 97-pair source seal."""

    _require_designated_paths(
        "seal-source",
        (
            (
                "preflight",
                preflight_path,
                ROOT / DEVELOPMENT_BINDING_PATHS["metadata_preflight_result"],
            ),
            ("source manifest", output_path, DEFAULT_SOURCE_MANIFEST),
        ),
    )
    preflight = _read_json(preflight_path)
    audit = preflight.get("access_audit", {})
    if (
        preflight.get("schema_version") != "combat_citeseq_metadata_preflight_v3"
        or preflight.get("status") != "PREFLIGHT_METADATA_CONTRACT_PASS"
        or preflight.get("warnings") != []
        or not isinstance(audit, dict)
        or audit.get("matrix_payload_reads") != 0
    ):
        raise PermissionError("source sealing requires a clean metadata-only v3 PASS")
    contract = preflight.get("frozen_sample_contract", {})
    expected = contract.get("expected_metadata_by_combat_id")
    counts = contract.get("designated_sample_counts")
    if (
        not isinstance(expected, dict)
        or not isinstance(counts, dict)
        or len(expected) != 97
    ):
        raise ValueError("preflight lacks the exact 97-pair sample universe")
    samples = []
    role_names = {
        "calibration": "calibration",
        "pilot_adaptive_development": "pilot",
        "oxford_held_confirmatory": "held_donor",
        "st_georges_held_confirmatory": "held_site",
    }
    for combat_id in sorted(expected):
        row = expected[combat_id]
        count = counts.get(combat_id)
        if not isinstance(row, dict) or not isinstance(count, dict):
            raise ValueError("preflight sample record is malformed")
        role = role_names.get(row.get("role"))
        if role is None:
            raise ValueError("preflight sample role is unknown")
        eligible = count.get("eligible_cells")
        official_total = count.get("composition_total_pbmc_count")
        if (
            isinstance(eligible, bool)
            or not isinstance(eligible, int)
            or eligible < CELL_BUDGET
            or isinstance(official_total, bool)
            or not isinstance(official_total, int)
            or official_total <= 0
        ):
            raise ValueError("preflight sample pool metadata is invalid")
        samples.append(
            {
                "combat_id": combat_id,
                "sample": row.get("scRNASeq_sample_ID"),
                "source": row.get("source"),
                "institute": row.get("institute"),
                "role": role,
                "eligible_pool_cells": eligible,
                "official_total_pbmc_count": official_total,
            }
        )
    records = _sample_records({"samples": samples})
    roles = assign_roles(records)
    if any(roles[row["sample"]] != row["role"] for row in samples):
        raise PermissionError("preflight roles differ from metadata-derived hash split")
    h5ad = preflight.get("input", {})
    composition = preflight.get("composition_contract", {})
    composition_input = (
        composition.get("input", {}) if isinstance(composition, dict) else {}
    )
    if not composition.get("valid"):
        raise PermissionError("official composition contract did not pass")
    if (
        h5ad.get("filename") != "COMBAT-CITESeq-DATA.h5ad"
        or h5ad.get("bytes") != OFFICIAL_H5AD_BYTES
        or h5ad.get("sha256") != OFFICIAL_H5AD_SHA256
    ):
        raise PermissionError("preflight input differs from the frozen H5AD")
    payload = {
        "schema": "combat-citeseq-source-manifest/1.0",
        "status": "SOURCE_SEALED",
        "created_at_utc": _timestamp(),
        "preflight_sha256": _sha256(preflight_path),
        "h5ad": {
            "filename": h5ad.get("filename"),
            "bytes": h5ad.get("bytes"),
            "md5": OFFICIAL_H5AD_MD5,
            "sha256": h5ad.get("sha256"),
            "raw_matrix_path": "layers/raw",
        },
        "composition": {
            "filename": composition_input.get("filename"),
            "bytes": composition_input.get("bytes"),
            "md5": composition_input.get("md5"),
            "sha256": composition_input.get("sha256"),
            "parent_archive": composition.get("parent_archive"),
        },
        "markers": [
            {
                "marker": marker,
                "rna_ensembl": RNA_ENSEMBL[marker],
                "adt_feature": ADT_FEATURE[marker],
            }
            for marker in MARKERS
        ],
        "eligible_cell_types": list(CELL_TYPES),
        "samples": samples,
    }
    text = json.dumps(payload, sort_keys=True, allow_nan=False)
    if "/Users/" in text or "file://" in text or '"path"' in text:
        raise ValueError("public source seal contains a local path")
    _write_json(output_path, payload, exclusive=True)
    return payload


def _repo_bound_path(
    record: Any, label: str, expected_relative: str
) -> tuple[Path, str]:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise PermissionError(f"development authorization lacks {label} binding")
    relative = record.get("path")
    digest = record.get("sha256")
    if (
        not isinstance(relative, str)
        or relative != expected_relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
    ):
        raise PermissionError(f"development authorization {label} binding is invalid")
    path = (ROOT / relative).resolve()
    if (
        ROOT.resolve() not in path.parents
        or not path.is_file()
        or _sha256(path) != digest
    ):
        raise PermissionError(f"development authorization {label} binding differs")
    return path, digest


def _immutable_public_bytes(relative: str, commit: str, label: str) -> bytes:
    path = Path(relative)
    if (
        not re.fullmatch(r"[0-9a-f]{40}", commit)
        or path.is_absolute()
        or ".." in path.parts
    ):
        raise PermissionError(f"{label} commit or path is not immutable")
    raw_url = (
        "https://raw.githubusercontent.com/"
        f"{PUBLIC_GITHUB_OWNER}/{PUBLIC_GITHUB_REPOSITORY}/{commit}/"
        f"{path.as_posix()}"
    )
    request = urllib.request.Request(
        raw_url, headers={"User-Agent": "coupling-fields/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
    except Exception as error:
        raise PermissionError(f"immutable public {label} fetch failed") from error


def _validated_development_authorization(
    authorization_path: Path,
    source_path: Path,
    authorization_commit: str,
) -> dict[str, Any]:
    """Authorize nonheld assay decoding without consulting local git state."""

    authorization = _read_json(authorization_path)
    if (
        authorization.get("schema") != "combat-citeseq-development-authorization/1.0"
        or authorization.get("status") != "OUTCOME_ACCESS_AUTHORIZED"
        or set(authorization)
        != {
            "schema",
            "status",
            "public_freeze_commit",
            "public_verification_commit",
            "bindings",
        }
    ):
        raise PermissionError("development outcome access is disabled")
    commit = authorization.get("public_freeze_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise PermissionError("development authorization lacks a public freeze commit")
    verification_commit = authorization.get("public_verification_commit")
    if not isinstance(verification_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", verification_commit
    ):
        raise PermissionError(
            "development authorization lacks a public verification commit"
        )
    bindings = authorization.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != set(
        DEVELOPMENT_BINDING_PATHS
    ):
        raise PermissionError("development authorization binding set differs")
    resolved = {
        label: _repo_bound_path(
            bindings[label], label, DEVELOPMENT_BINDING_PATHS[label]
        )
        for label in sorted(bindings)
    }
    if resolved["runner"][0] != Path(__file__).resolve():
        raise PermissionError("development authorization binds another runner")
    if resolved["source_manifest"][0] != source_path.resolve():
        raise PermissionError("development authorization binds another source manifest")
    verification_path = resolved["fresh_clone_verification"][0]
    if (
        _immutable_public_bytes(
            DEVELOPMENT_BINDING_PATHS["fresh_clone_verification"],
            verification_commit,
            "fresh-clone verification",
        )
        != verification_path.read_bytes()
    ):
        raise PermissionError("immutable public fresh-clone verification differs")
    for label, (path, _) in resolved.items():
        if label == "fresh_clone_verification":
            continue
        if (
            _immutable_public_bytes(
                DEVELOPMENT_BINDING_PATHS[label], commit, f"frozen {label}"
            )
            != path.read_bytes()
        ):
            raise PermissionError(f"immutable public frozen {label} differs")
    verification = _read_json(verification_path)
    expected_verification_bindings = {
        label: {
            "path": DEVELOPMENT_BINDING_PATHS[label],
            "sha256": resolved[label][1],
        }
        for label in DEVELOPMENT_BINDING_PATHS
        if label != "fresh_clone_verification"
    }
    if (
        set(verification)
        != {
            "schema",
            "status",
            "fresh_clone",
            "origin",
            "immutable_tag",
            "verified_commit",
            "designation_sha256",
            "protocol_sha256",
            "artifact_bindings",
            "source_h5ad_sha256",
            "composition_csv_sha256",
            "matrix_payload_reads",
            "all_bound_artifacts_match",
        }
        or verification.get("schema") != "combat-citeseq-public-freeze-verification/1.0"
        or verification.get("status") != "PASS"
        or verification.get("fresh_clone") is not True
        or verification.get("origin") != PUBLIC_GITHUB_ORIGIN
        or verification.get("verified_commit") != commit
        or verification.get("designation_sha256") != resolved["designation"][1]
        or verification.get("protocol_sha256") != resolved["protocol"][1]
        or verification.get("artifact_bindings") != expected_verification_bindings
        or verification.get("source_h5ad_sha256") != OFFICIAL_H5AD_SHA256
        or verification.get("composition_csv_sha256") != OFFICIAL_COMPOSITION_SHA256
        or verification.get("matrix_payload_reads") != 0
        or verification.get("all_bound_artifacts_match") is not True
        or not isinstance(verification.get("immutable_tag"), str)
        or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}",
            verification["immutable_tag"],
        )
    ):
        raise PermissionError("fresh-clone verification record differs from freeze")
    relative_authorization = _relative(authorization_path)
    public_bytes = _immutable_public_bytes(
        relative_authorization,
        authorization_commit,
        "development authorization",
    )
    local_bytes = authorization_path.read_bytes()
    if public_bytes != local_bytes:
        raise PermissionError("immutable public development authorization differs")
    return {
        "public_freeze_commit": commit,
        "public_verification_commit": verification_commit,
        "public_authorization_commit": authorization_commit,
        "authorization_sha256": _sha256(authorization_path),
        "binding_sha256": {label: value[1] for label, value in resolved.items()},
    }


def _decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
            for value in np.asarray(values)
        ],
        dtype=str,
    )


def _encoded_column(group: h5py.Group, name: str) -> np.ndarray:
    if name not in group:
        raise ValueError(f"H5AD dataframe lacks {name}")
    item = group[name]
    if isinstance(item, h5py.Dataset):
        values = item[...]
        if h5py.check_string_dtype(item.dtype) is not None:
            return _decode_strings(values)
        legacy = group.get("__categories")
        if (
            isinstance(legacy, h5py.Group)
            and name in legacy
            and np.issubdtype(item.dtype, np.integer)
        ):
            categories = _decode_strings(legacy[name][...])
            codes = np.asarray(values, dtype=np.int64)
            decoded = np.full(len(codes), "", dtype=object)
            valid = (codes >= 0) & (codes < len(categories))
            decoded[valid] = categories[codes[valid]]
            return np.asarray(decoded, dtype=str)
        return np.asarray(values)
    if (
        not isinstance(item, h5py.Group)
        or "categories" not in item
        or "codes" not in item
    ):
        raise ValueError(f"H5AD column {name} has an unsupported encoding")
    categories = _decode_strings(item["categories"][...])
    codes = np.asarray(item["codes"][...], dtype=np.int64)
    decoded = np.full(len(codes), "", dtype=object)
    valid = (codes >= 0) & (codes < len(categories))
    decoded[valid] = categories[codes[valid]]
    return np.asarray(decoded, dtype=str)


def _dataframe_index(group: h5py.Group) -> np.ndarray:
    raw = group.attrs.get("_index", "_index")
    name = raw.decode() if isinstance(raw, bytes) else str(raw)
    return _decode_strings(group[name][...])


def _feature_columns(handle: h5py.File) -> dict[str, list[int]]:
    var = handle.get("var")
    if not isinstance(var, h5py.Group):
        raise ValueError("H5AD var dataframe is absent")
    names = _dataframe_index(var)
    gene_ids = _encoded_column(var, "gene_ids")
    feature_types = _encoded_column(var, "feature_types")
    if not (len(names) == len(gene_ids) == len(feature_types)):
        raise ValueError("H5AD feature metadata axes differ")

    stripped = np.asarray(
        [re.sub(r"\.[0-9]+$", "", value) for value in gene_ids], dtype=str
    )
    result: dict[str, list[int]] = {"rna": [], "adt": []}
    for marker in MARKERS:
        rna = np.flatnonzero(
            (names == marker)
            & (stripped == RNA_ENSEMBL[marker])
            & (feature_types == "Gene Expression")
        )
        adt_name = ADT_FEATURE[marker]
        adt = np.flatnonzero(
            (names == adt_name) & (feature_types == "Antibody Capture")
        )
        if len(rna) != 1 or len(adt) != 1:
            raise ValueError(f"locked feature pair for {marker} is absent or nonunique")
        result["rna"].append(int(rna[0]))
        result["adt"].append(int(adt[0]))
    if set(result["rna"]) & set(result["adt"]):
        raise AssertionError("RNA and ADT feature columns overlap")
    return result


def _cell_hash(combat_id: str, sample: str, barcode: str) -> str:
    return hashlib.sha256(
        "\0".join((CELL_SELECTION_SALT, combat_id, sample, barcode)).encode()
    ).hexdigest()


def _selected_sample_rows(
    h5ad_path: Path,
    records: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    """Validate sample metadata and freeze 512 outcome-independent rows/sample."""

    with h5py.File(h5ad_path, "r") as handle:
        obs = handle.get("obs")
        if not isinstance(obs, h5py.Group):
            raise ValueError("H5AD obs dataframe is absent")
        barcodes = _dataframe_index(obs)
        sample = _encoded_column(obs, "scRNASeq_sample_ID")
        combat = _encoded_column(obs, "COMBAT_ID")
        source = _encoded_column(obs, "Source")
        institute = _encoded_column(obs, "Institute")
        cell_type = _encoded_column(obs, "Annotation_cell_type")
    lengths = {
        len(value) for value in (barcodes, sample, combat, source, institute, cell_type)
    }
    if lengths != {len(barcodes)} or len(set(barcodes.tolist())) != len(barcodes):
        raise ValueError("H5AD observation metadata axes differ or barcodes repeat")

    selected: dict[str, dict[str, Any]] = {}
    allowed = np.isin(cell_type, np.asarray(CELL_TYPES))
    for record in records:
        sample_rows = np.flatnonzero(sample == record["sample"])
        if not len(sample_rows):
            raise ValueError(f"sample {record['sample']} is absent from H5AD")
        for values, expected, label in (
            (combat, record["combat_id"], "COMBAT_ID"),
            (source, record["source"], "Source"),
            (institute, record["institute"], "Institute"),
        ):
            observed = set(values[sample_rows].tolist())
            if observed != {expected}:
                raise ValueError(f"sample {record['sample']} has inconsistent {label}")
        eligible = sample_rows[allowed[sample_rows]]
        if len(eligible) < CELL_BUDGET:
            raise ValueError(
                f"sample {record['sample']} has fewer than 512 eligible cells"
            )
        ordered = sorted(
            eligible.tolist(),
            key=lambda row: (
                _cell_hash(record["combat_id"], record["sample"], barcodes[row]),
                barcodes[row],
            ),
        )[:CELL_BUDGET]
        rows = np.asarray(sorted(ordered), dtype=np.int64)
        selected_barcodes = barcodes[rows]
        selected[record["sample"]] = {
            "rows": rows,
            "barcodes": selected_barcodes,
            "cell_types": cell_type[rows],
            "eligible_pool_cells": int(len(eligible)),
            "selected_barcode_sha256": hashlib.sha256(
                ("\n".join(sorted(selected_barcodes.tolist())) + "\n").encode()
            ).hexdigest(),
        }
    return selected


def _matrix_path(source: dict[str, Any]) -> str:
    value = source.get("h5ad", {}).get("raw_matrix_path", "layers/raw")
    if not isinstance(value, str) or not value or value.startswith("/"):
        raise ValueError("h5ad.raw_matrix_path must be a relative HDF5 path")
    return value


def _read_csr_feature_subset(
    matrix: h5py.Group | h5py.Dataset,
    rows: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    """Read values only at selected CSR row/feature intersections.

    CSR indices for selected rows are structural metadata.  Matrix values are
    fetched only at positions whose column is one of the frozen features.
    """

    selected_rows = np.asarray(rows, dtype=np.int64)
    selected_columns = np.asarray(columns, dtype=np.int64)
    if selected_rows.ndim != 1 or selected_columns.ndim != 1:
        raise ValueError("selected matrix axes must be vectors")
    if len(selected_rows) == 0 or len(set(selected_rows.tolist())) != len(
        selected_rows
    ):
        raise ValueError("selected matrix rows must be nonempty and unique")

    if isinstance(matrix, h5py.Dataset):
        if matrix.ndim != 2:
            raise ValueError("dense raw matrix must be two-dimensional")
        output = np.empty((len(selected_rows), len(selected_columns)), dtype=float)
        for index, row in enumerate(selected_rows):
            output[index] = np.asarray(matrix[int(row), selected_columns])
        return output

    encoding = matrix.attrs.get("encoding-type")
    if isinstance(encoding, bytes):
        encoding = encoding.decode()
    if encoding != "csr_matrix":
        raise ValueError("raw H5AD matrix must be CSR or dense")
    shape = tuple(int(value) for value in matrix.attrs.get("shape", ()))
    if len(shape) != 2:
        raise ValueError("raw CSR shape is absent")
    if (
        np.any(selected_rows < 0)
        or np.any(selected_rows >= shape[0])
        or np.any(selected_columns < 0)
        or np.any(selected_columns >= shape[1])
    ):
        raise IndexError("selected raw-matrix axis is out of range")

    indptr_dataset = matrix["indptr"]
    if tuple(indptr_dataset.shape) != (shape[0] + 1,):
        raise ValueError("raw CSR indptr length differs from shape")
    endpoints = np.unique(np.concatenate((selected_rows, selected_rows + 1)))
    pointer_values = np.asarray(indptr_dataset[endpoints], dtype=np.int64)
    if pointer_values.shape != endpoints.shape:
        raise ValueError("raw CSR row-pointer subset is malformed")
    indptr = dict(zip(endpoints.tolist(), pointer_values.tolist()))
    column_to_output = {
        int(column): index for index, column in enumerate(selected_columns)
    }
    positions: list[int] = []
    destinations: list[tuple[int, int]] = []
    indices_dataset = matrix["indices"]
    for output_row, row in enumerate(selected_rows):
        left, right = int(indptr[int(row)]), int(indptr[int(row) + 1])
        indices = np.asarray(indices_dataset[left:right], dtype=np.int64)
        if len(np.unique(indices)) != len(indices):
            raise ValueError("selected CSR row contains a duplicate column index")
        for offset, column in enumerate(indices):
            output_column = column_to_output.get(int(column))
            if output_column is not None:
                positions.append(left + offset)
                destinations.append((output_row, output_column))

    output = np.zeros((len(selected_rows), len(selected_columns)), dtype=float)
    if positions:
        position_array = np.asarray(positions, dtype=np.int64)
        if np.any(np.diff(position_array) <= 0):
            raise ValueError("selected CSR value positions are not strictly ordered")
        data = matrix["data"]
        values = []
        for start in range(0, len(position_array), 100_000):
            values.append(np.asarray(data[position_array[start : start + 100_000]]))
        selected_values = np.concatenate(values)
        for destination, value in zip(destinations, selected_values):
            output[destination] = float(value)
    if not np.isfinite(output).all() or np.any(output < 0.0):
        raise ValueError("raw selected counts must be finite and nonnegative")
    return output


def _read_modality(
    h5ad_path: Path,
    source: dict[str, Any],
    selections: dict[str, dict[str, Any]],
    samples: list[str],
    modality: str,
) -> dict[str, np.ndarray]:
    if modality not in {"rna", "adt"}:
        raise ValueError("modality must be rna or adt")
    if len(set(samples)) != len(samples) or any(
        sample not in selections for sample in samples
    ):
        raise ValueError("modality request contains an unknown or duplicate sample")
    all_rows = np.concatenate([selections[sample]["rows"] for sample in samples])
    row_order = np.argsort(all_rows, kind="mergesort")
    sorted_rows = all_rows[row_order]
    if np.any(np.diff(sorted_rows) == 0):
        raise ValueError("selected cells overlap between samples")
    with h5py.File(h5ad_path, "r") as handle:
        feature_columns = _feature_columns(handle)[modality]
        path = _matrix_path(source)
        if path not in handle:
            raise ValueError(f"raw matrix {path} is absent")
        sorted_values = _read_csr_feature_subset(
            handle[path], sorted_rows, np.asarray(feature_columns, dtype=np.int64)
        )
    unsorted = np.empty_like(sorted_values)
    unsorted[row_order] = sorted_values
    result = {}
    offset = 0
    for sample in samples:
        result[sample] = unsorted[offset : offset + CELL_BUDGET].T
        offset += CELL_BUDGET
    return result


def _integer_counts(values: np.ndarray, label: str) -> np.ndarray:
    counts = np.asarray(values, dtype=float)
    rounded = np.rint(counts)
    if (
        not np.isfinite(counts).all()
        or np.any(counts < 0.0)
        or not np.array_equal(counts, rounded)
    ):
        raise ValueError(f"{label} raw layer is not a nonnegative integer count matrix")
    return rounded.astype(np.int64)


def _adt_states(
    counts: np.ndarray,
    barcodes: np.ndarray,
    combat_id: str,
    sample: str,
) -> np.ndarray:
    values = _integer_counts(counts, "ADT")
    barcode_values = _decode_strings(barcodes)
    if (
        values.shape != (len(MARKERS), CELL_BUDGET)
        or len(barcode_values) != CELL_BUDGET
    ):
        raise ValueError("ADT state input has the wrong frozen shape")
    states = np.ones_like(values, dtype=np.uint8)
    for marker_index, marker in enumerate(MARKERS):
        tie = np.asarray(
            [
                hashlib.sha256(
                    "\0".join(
                        (ADT_TIE_SALT, combat_id, sample, barcode, marker)
                    ).encode()
                ).hexdigest()
                for barcode in barcode_values
            ]
        )
        order = np.lexsort((tie, values[marker_index]))
        states[marker_index, order[: CELL_BUDGET // 2]] = 0
    if np.any(states.sum(axis=1) != CELL_BUDGET // 2):
        raise AssertionError("ADT midrank did not create exact 256/256 margins")
    return states


def _destroyed_adt(states: np.ndarray, barcodes: np.ndarray, sample: str) -> np.ndarray:
    barcode_values = _decode_strings(barcodes)
    order = sorted(
        range(CELL_BUDGET),
        key=lambda index: (
            hashlib.sha256(
                "\0".join((DESTROYED_LINK_SALT, sample, barcode_values[index])).encode()
            ).hexdigest(),
            barcode_values[index],
        ),
    )
    destroyed = np.empty_like(states)
    for index, destination in enumerate(order):
        destroyed[:, destination] = states[:, order[(index + 1) % CELL_BUDGET]]
    return destroyed


def _form_tables(
    rna_states: dict[str, np.ndarray],
    adt_states: dict[str, np.ndarray],
    samples: list[str],
) -> np.ndarray:
    """Form every requested sample's 81 linked 2x2 tables in one call."""

    tables = np.empty((len(samples), len(MARKERS), len(MARKERS), 2, 2), dtype=np.int64)
    for sample_index, sample in enumerate(samples):
        rna = np.asarray(rna_states[sample], dtype=np.uint8)
        adt = np.asarray(adt_states[sample], dtype=np.uint8)
        if rna.shape != (len(MARKERS), CELL_BUDGET) or adt.shape != rna.shape:
            raise ValueError("binary state matrix has the wrong frozen shape")
        for first in range(len(MARKERS)):
            for second in range(len(MARKERS)):
                code = 2 * rna[first].astype(np.int64) + adt[second].astype(np.int64)
                tables[sample_index, first, second] = np.bincount(
                    code, minlength=4
                ).reshape(2, 2)
    return tables


def _stratum_profiles(
    rna: np.ndarray,
    adt_counts: np.ndarray,
    cell_types: np.ndarray,
) -> list[dict[str, Any]]:
    rna_values = np.asarray(rna, dtype=np.uint8)
    adt_values = _integer_counts(adt_counts, "ADT")
    labels = _decode_strings(cell_types)
    panel_total = adt_values.sum(axis=0)
    composition = np.divide(
        100.0 * adt_values,
        panel_total[None, :],
        out=np.zeros_like(adt_values, dtype=float),
        where=panel_total[None, :] > 0,
    )
    transformed = np.log1p(composition)
    result = []
    for label in CELL_TYPES:
        selected = labels == label
        if np.any(selected):
            result.append(
                {
                    "cell_type": label,
                    "cells": int(np.count_nonzero(selected)),
                    "rna_detection_prevalence": rna_values[:, selected]
                    .mean(axis=1)
                    .tolist(),
                    "adt_log_panel_fraction_mean": transformed[:, selected]
                    .mean(axis=1)
                    .tolist(),
                }
            )
    if not result:
        raise ValueError("selected sample has no frozen cell-type stratum")
    return result


def _informative(tables: np.ndarray) -> np.ndarray:
    values = np.asarray(tables)
    rows = values.sum(axis=-1)
    columns = values.sum(axis=-2)
    total = values.sum(axis=(-2, -1))
    lower = np.maximum(0, rows[..., 0] + columns[..., 0] - total)
    upper = np.minimum(rows[..., 0], columns[..., 0])
    return upper > lower


def _development_records(source: dict[str, Any]) -> list[dict[str, Any]]:
    selections = _selected_sample_rows(source["h5ad"], source["records"])
    development_samples = sorted(
        record["sample"]
        for record in source["records"]
        if source["roles"][record["sample"]] in {"calibration", "pilot"}
    )
    if len(development_samples) != 36:
        raise AssertionError("development split must contain exactly 36 samples")
    combat_by_sample = {
        record["sample"]: record["combat_id"] for record in source["records"]
    }

    rna_counts = _read_modality(
        source["h5ad"], source["payload"], selections, development_samples, "rna"
    )
    rna_states = {
        sample: (_integer_counts(rna_counts[sample], "RNA") > 0).astype(np.uint8)
        for sample in development_samples
    }
    adt_counts = _read_modality(
        source["h5ad"], source["payload"], selections, development_samples, "adt"
    )
    adt_states = {
        sample: _adt_states(
            adt_counts[sample],
            selections[sample]["barcodes"],
            combat_by_sample[sample],
            sample,
        )
        for sample in development_samples
    }
    tables = _form_tables(rna_states, adt_states, development_samples)
    destroyed_states = {
        sample: _destroyed_adt(
            adt_states[sample], selections[sample]["barcodes"], sample
        )
        for sample in development_samples
    }
    destroyed = _form_tables(rna_states, destroyed_states, development_samples)
    if not np.array_equal(
        tables.sum(axis=-1), destroyed.sum(axis=-1)
    ) or not np.array_equal(tables.sum(axis=-2), destroyed.sum(axis=-2)):
        raise AssertionError("destroyed-link control changed a fixed margin")

    records = []
    for index, sample in enumerate(development_samples):
        support = _informative(tables[index])
        if int(support.sum()) < MINIMUM_INFORMATIVE_ENTITIES:
            raise ValueError(f"sample {sample} has fewer than 64 informative entities")
        records.append(
            {
                "sample": sample,
                "combat_id": combat_by_sample[sample],
                "role": source["roles"][sample],
                "cells": CELL_BUDGET,
                "eligible_pool_cells": selections[sample]["eligible_pool_cells"],
                "selected_barcode_sha256": selections[sample][
                    "selected_barcode_sha256"
                ],
                "strata": _stratum_profiles(
                    rna_states[sample],
                    adt_counts[sample],
                    selections[sample]["cell_types"],
                ),
                "tables": tables[index].reshape(len(MARKERS) ** 2, 4).tolist(),
                "destroyed_tables": destroyed[index]
                .reshape(len(MARKERS) ** 2, 4)
                .tolist(),
                "informative": support.reshape(-1).tolist(),
            }
        )
    return records


def reduce_development(
    source_path: Path,
    authorization_path: Path,
    authorization_commit: str,
    output_path: Path,
) -> dict[str, Any]:
    """Decode only the 12 calibration and 24 pilot sample matrices."""

    _require_designated_paths(
        "reduce-development",
        (
            ("source manifest", source_path, DEFAULT_SOURCE_MANIFEST),
            (
                "development authorization",
                authorization_path,
                DEFAULT_DEVELOPMENT_AUTHORIZATION,
            ),
            ("output", output_path, DEFAULT_REDUCED),
        ),
    )
    if output_path.exists():
        raise FileExistsError("development reduction artifact already exists")
    authorization = _validated_development_authorization(
        authorization_path, source_path, authorization_commit
    )
    source = _validated_source(source_path, verify_hash=True)
    if (
        source["payload"].get("preflight_sha256")
        != authorization["binding_sha256"]["metadata_preflight_result"]
    ):
        raise PermissionError(
            "source seal and development authorization bind different preflights"
        )
    records = _development_records(source)

    payload = {
        "schema": "combat-citeseq-reduced-development/1.0",
        "status": "DEVELOPMENT_REDUCTION_COMPLETE",
        "created_at_utc": _timestamp(),
        "source_manifest_sha256": source["source_manifest_sha256"],
        "h5ad_sha256": source["h5ad_sha256"],
        "development_authorization": authorization,
        "markers": list(MARKERS),
        "cell_types": list(CELL_TYPES),
        "cells_per_sample": CELL_BUDGET,
        "cell_selection_salt": CELL_SELECTION_SALT,
        "adt_tie_salt": ADT_TIE_SALT,
        "samples": records,
        "access_audit": {
            "calibration_samples_read": 12,
            "pilot_samples_read": 24,
            "held_donor_matrix_rows_read": 0,
            "held_site_matrix_rows_read": 0,
            "modalities_read_sequentially": ["rna", "adt"],
            "matrix_values_decoded_only_for_frozen_features": 18,
        },
    }
    _write_json(output_path, payload, exclusive=True)
    return payload


def _validated_reduced(path: Path, source_path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    source = _validated_source(source_path, verify_hash=True)
    expected_top_level = {
        "schema",
        "status",
        "created_at_utc",
        "source_manifest_sha256",
        "h5ad_sha256",
        "development_authorization",
        "markers",
        "cell_types",
        "cells_per_sample",
        "cell_selection_salt",
        "adt_tie_salt",
        "samples",
        "access_audit",
    }
    expected_audit = {
        "calibration_samples_read": 12,
        "pilot_samples_read": 24,
        "held_donor_matrix_rows_read": 0,
        "held_site_matrix_rows_read": 0,
        "modalities_read_sequentially": ["rna", "adt"],
        "matrix_values_decoded_only_for_frozen_features": 18,
    }
    if (
        set(payload) != expected_top_level
        or payload.get("schema") != "combat-citeseq-reduced-development/1.0"
        or payload.get("status") != "DEVELOPMENT_REDUCTION_COMPLETE"
        or payload.get("source_manifest_sha256") != source["source_manifest_sha256"]
        or payload.get("h5ad_sha256") != source["h5ad_sha256"]
        or payload.get("markers") != list(MARKERS)
        or payload.get("cell_types") != list(CELL_TYPES)
        or payload.get("cells_per_sample") != CELL_BUDGET
        or payload.get("cell_selection_salt") != CELL_SELECTION_SALT
        or payload.get("adt_tie_salt") != ADT_TIE_SALT
        or payload.get("access_audit") != expected_audit
    ):
        raise PermissionError("reduced development artifact differs from freeze")
    embedded_authorization = payload.get("development_authorization")
    if not isinstance(embedded_authorization, dict) or set(embedded_authorization) != {
        "public_freeze_commit",
        "public_verification_commit",
        "public_authorization_commit",
        "authorization_sha256",
        "binding_sha256",
    }:
        raise PermissionError("reduced development authorization record is malformed")
    authorization_commit = embedded_authorization.get("public_authorization_commit")
    if not isinstance(authorization_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", authorization_commit
    ):
        raise PermissionError("reduced development authorization commit is invalid")
    revalidated_authorization = _validated_development_authorization(
        DEFAULT_DEVELOPMENT_AUTHORIZATION,
        source_path,
        authorization_commit,
    )
    if _canonical_json_sha256(embedded_authorization) != _canonical_json_sha256(
        revalidated_authorization
    ):
        raise PermissionError("reduced development authorization does not revalidate")
    records = payload.get("samples")
    if not isinstance(records, list) or len(records) != 36:
        raise ValueError("reduced development artifact must contain 36 samples")
    by_sample: dict[str, dict[str, Any]] = {}
    by_combat: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "sample",
            "combat_id",
            "role",
            "cells",
            "eligible_pool_cells",
            "selected_barcode_sha256",
            "strata",
            "tables",
            "destroyed_tables",
            "informative",
        }:
            raise ValueError("reduced sample record is malformed")
        sample = record.get("sample")
        combat_id = record.get("combat_id")
        if not isinstance(sample, str):
            raise ValueError("reduced sample name is malformed")
        source_record = next(
            (row for row in source["records"] if row["sample"] == sample), None
        )
        source_payload_record = next(
            (row for row in source["payload"]["samples"] if row["sample"] == sample),
            None,
        )
        if (
            sample in by_sample
            or sample not in source["roles"]
            or not isinstance(combat_id, str)
            or combat_id in by_combat
            or source_record is None
            or source_record["combat_id"] != combat_id
            or source_payload_record is None
        ):
            raise ValueError("reduced sample is duplicated or unknown")
        if record.get("role") != source["roles"][sample] or record["role"] not in {
            "calibration",
            "pilot",
        }:
            raise PermissionError("held sample entered reduced development")
        if (
            record.get("cells") != CELL_BUDGET
            or record.get("eligible_pool_cells")
            != source_payload_record["eligible_pool_cells"]
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(record.get("selected_barcode_sha256"))
            )
        ):
            raise PermissionError("reduced sample cell-selection contract differs")
        raw_tables = np.asarray(record.get("tables"))
        raw_destroyed = np.asarray(record.get("destroyed_tables"))
        if (
            raw_tables.shape != (len(MARKERS) ** 2, 4)
            or raw_destroyed.shape != raw_tables.shape
            or not np.issubdtype(raw_tables.dtype, np.integer)
            or not np.issubdtype(raw_destroyed.dtype, np.integer)
            or np.any(raw_tables < 0)
            or np.any(raw_destroyed < 0)
            or np.any(raw_tables.sum(axis=1) != CELL_BUDGET)
            or np.any(raw_destroyed.sum(axis=1) != CELL_BUDGET)
        ):
            raise ValueError("reduced 2x2 tables are invalid")
        tables = raw_tables.reshape(len(MARKERS), len(MARKERS), 2, 2)
        destroyed = raw_destroyed.reshape(tables.shape)
        if not np.array_equal(
            tables.sum(axis=-1), destroyed.sum(axis=-1)
        ) or not np.array_equal(tables.sum(axis=-2), destroyed.sum(axis=-2)):
            raise PermissionError("reduced destroyed-link margins differ")
        informative = np.asarray(record.get("informative"))
        expected_informative = _informative(tables).reshape(-1)
        if (
            informative.shape != (len(MARKERS) ** 2,)
            or informative.dtype != np.dtype(bool)
            or not np.array_equal(informative, expected_informative)
            or int(informative.sum()) < MINIMUM_INFORMATIVE_ENTITIES
        ):
            raise PermissionError("reduced informative mask does not recompute")
        strata = record.get("strata")
        if not isinstance(strata, list) or not strata:
            raise ValueError("reduced sample lacks marginal-profile strata")
        labels = []
        stratum_cells = 0
        for stratum in strata:
            if not isinstance(stratum, dict) or set(stratum) != {
                "cell_type",
                "cells",
                "rna_detection_prevalence",
                "adt_log_panel_fraction_mean",
            }:
                raise ValueError("marginal-profile stratum schema differs")
            label = stratum.get("cell_type")
            cells = stratum.get("cells")
            rna = np.asarray(stratum.get("rna_detection_prevalence"), dtype=float)
            adt = np.asarray(stratum.get("adt_log_panel_fraction_mean"), dtype=float)
            if (
                label not in CELL_TYPES
                or isinstance(cells, bool)
                or not isinstance(cells, int)
                or cells <= 0
                or rna.shape != (len(MARKERS),)
                or adt.shape != rna.shape
                or not np.isfinite(rna).all()
                or not np.isfinite(adt).all()
                or np.any((rna < 0.0) | (rna > 1.0))
                or np.any((adt < 0.0) | (adt > math.log1p(100.0)))
            ):
                raise ValueError("marginal-profile stratum values are invalid")
            labels.append(label)
            stratum_cells += cells
        if labels != [label for label in CELL_TYPES if label in labels]:
            raise ValueError("marginal-profile strata are not in locked label order")
        if len(set(labels)) != len(labels) or stratum_cells != CELL_BUDGET:
            raise ValueError("marginal-profile strata do not partition selected cells")
        by_sample[sample] = record
        by_combat[combat_id] = record
    if set(by_combat) != set(CALIBRATION_IDS) | set(PILOT_IDS):
        raise PermissionError("reduced development sample set differs from freeze")
    calibration_samples = _samples_for_ids(source["records"], CALIBRATION_IDS)
    pilot_samples = _samples_for_ids(source["records"], PILOT_IDS)
    expected_order = sorted(calibration_samples + pilot_samples)
    if [record["sample"] for record in records] != expected_order:
        raise PermissionError("reduced development sample order differs from producer")
    replayed_records = _development_records(source)
    if _canonical_json_sha256(records) != _canonical_json_sha256(replayed_records):
        raise PermissionError("reduced development records do not replay from source")
    return {
        "payload": payload,
        "source": source,
        "by_sample": by_sample,
        "calibration_samples": calibration_samples,
        "pilot_samples": pilot_samples,
    }


def _tables(
    records: dict[str, dict[str, Any]], samples: tuple[str, ...], key: str
) -> np.ndarray:
    values = np.asarray([records[sample][key] for sample in samples], dtype=np.int64)
    return values.reshape(len(samples), len(MARKERS), len(MARKERS), 2, 2)


def _marginal_profiles(
    records: dict[str, dict[str, Any]], samples: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rna = []
    adt = []
    columns = []
    for sample in samples:
        strata = records[sample]["strata"]
        for expected_label in CELL_TYPES:
            matches = [row for row in strata if row["cell_type"] == expected_label]
            if not matches:
                continue
            row = matches[0]
            first = np.asarray(row.get("rna_detection_prevalence"), dtype=float)
            second = np.asarray(row.get("adt_log_panel_fraction_mean"), dtype=float)
            if (
                first.shape != (len(MARKERS),)
                or second.shape != first.shape
                or not np.isfinite(first).all()
                or not np.isfinite(second).all()
            ):
                raise ValueError("marginal-profile stratum has invalid coordinates")
            rna.append(first)
            adt.append(second)
            columns.append(f"{sample}\0{expected_label}")
    if len(columns) < len(MARKERS) + 1:
        raise ValueError("too few nonempty sample-cell-type profile strata")
    return np.asarray(rna), np.asarray(adt), columns


def _knn_incidence(profiles: np.ndarray, neighbors: int) -> np.ndarray:
    values = np.asarray(profiles, dtype=float)
    k = int(neighbors)
    if values.ndim != 2 or values.shape[1] != len(MARKERS) or not 1 <= k < len(MARKERS):
        raise ValueError("marginal profiles or graph neighborhood are invalid")
    marker_profiles = values.T
    scale = marker_profiles.std(axis=1, ddof=1)
    if np.any(~np.isfinite(scale)) or np.any(scale == 0.0):
        names = [MARKERS[index] for index in np.flatnonzero(scale == 0.0)]
        raise ValueError(f"zero-variance graph marker profiles: {names}")
    standardized = (
        marker_profiles - marker_profiles.mean(axis=1, keepdims=True)
    ) / scale[:, None]
    edges: set[tuple[int, int]] = set()
    indices = np.arange(len(MARKERS))
    for first in indices:
        candidates = indices[indices != first]
        distance = np.linalg.norm(
            standardized[candidates] - standardized[first], axis=1
        )
        order = candidates[np.lexsort((candidates, distance))]
        edges.update(
            tuple(sorted((int(first), int(second)), key=lambda index: MARKERS[index]))
            for second in order[:k]
        )
    ordered = sorted(edges, key=lambda edge: (MARKERS[edge[0]], MARKERS[edge[1]]))
    incidence = np.zeros((len(MARKERS), len(ordered)), dtype=float)
    for column, (first, second) in enumerate(ordered):
        incidence[first, column] = 1.0
        incidence[second, column] = 1.0
    if not ordered or np.any(incidence.sum(axis=1) == 0.0):
        raise ValueError("marginal-profile kNN union has an isolated marker")
    return incidence


def _graphs(
    records: dict[str, dict[str, Any]], samples: tuple[str, ...], neighbors: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rna, adt, columns = _marginal_profiles(records, samples)
    first = _knn_incidence(rna, neighbors)
    second = _knn_incidence(adt, neighbors)
    return (
        first,
        second,
        {
            "neighbors": int(neighbors),
            "profile_columns": columns,
            "rna_incidence_sha256": _array_sha256(first),
            "adt_incidence_sha256": _array_sha256(second),
            "rna_incidence": first.tolist(),
            "adt_incidence": second.tolist(),
        },
    )


def _permuted_incidence(incidence: np.ndarray, modality: str) -> np.ndarray:
    marker_order = sorted(
        range(len(MARKERS)),
        key=lambda index: hashlib.sha256(
            "\0".join((LABEL_PERMUTATION_SALT, modality, MARKERS[index])).encode()
        ).hexdigest(),
    )
    graph_positions = sorted(range(len(MARKERS)), key=lambda index: MARKERS[index])
    source = np.asarray(incidence, dtype=float)
    permuted = np.empty_like(source)
    for marker, position in zip(marker_order, graph_positions):
        permuted[marker] = source[position]
    return permuted


def _field_model(
    tables: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    ridge: float,
    graph: float,
    label: str,
) -> dict[str, Any]:
    fit = fit_heterogeneity_adaptive_binary_coupling(
        tables,
        first,
        second,
        variance_floor=1e-8,
        precision_floor=1e-8,
        ridge_penalty=ridge,
        graph_penalty=graph,
        minimum_donors=2,
        maximum_condition_number=MAXIMUM_CONDITION_NUMBER,
    )
    if (
        not np.isfinite(fit.structured.estimate).all()
        or not np.isfinite(fit.structured.condition_number)
        or fit.structured.condition_number > MAXIMUM_CONDITION_NUMBER
        or np.any(fit.pooled.support_count < 2)
    ):
        raise ValueError(f"{label} misses its numerical certificate")
    return {
        "kind": "centered_haldane",
        "estimator": label,
        "source_coordinate": fit.structured.estimate.reshape(-1).tolist(),
        "certificate": {
            "solver": "precision-weighted product-graph SPD solve",
            "pooling_solver": "Paule-Mandel bracketed root finding",
            "condition_number": float(fit.structured.condition_number),
            "objective": float(fit.structured.objective),
            "precision_scale": float(fit.structured.precision_scale),
            "support_count_range": [
                int(fit.pooled.support_count.min()),
                int(fit.pooled.support_count.max()),
            ],
            "tau_squared_range": [
                float(fit.pooled.tau_squared.min()),
                float(fit.pooled.tau_squared.max()),
            ],
            "variance_convention": fit.variance_convention,
        },
    }


def _unstructured_model(tables: np.ndarray, ridge: float) -> dict[str, Any]:
    identity = np.eye(len(MARKERS), dtype=float)
    fit = fit_heterogeneity_adaptive_binary_coupling(
        tables,
        identity,
        identity,
        variance_floor=1e-8,
        precision_floor=1e-8,
        ridge_penalty=ridge,
        graph_penalty=0.0,
        minimum_donors=2,
        maximum_condition_number=MAXIMUM_CONDITION_NUMBER,
    )
    return {
        "kind": "centered_haldane",
        "estimator": "unstructured exact-null-centered Paule-Mandel Haldane field",
        "source_coordinate": fit.structured.estimate.reshape(-1).tolist(),
        "selected_ridge_penalty": float(ridge),
        "certificate": {
            "support_count_range": [
                int(fit.pooled.support_count.min()),
                int(fit.pooled.support_count.max()),
            ],
            "variance_convention": fit.variance_convention,
        },
    }


def _raw_haldane_model(tables: np.ndarray) -> dict[str, Any]:
    values = np.asarray(tables)
    coordinates = np.empty(values.shape[:3], dtype=float)
    variance = np.empty_like(coordinates)
    support = np.empty(values.shape[:3], dtype=bool)
    for index in np.ndindex(values.shape[:3]):
        estimate = centered_haldane_log_odds(values[index])
        coordinates[index] = estimate.observed_log_odds
        variance[index] = estimate.sampling_variance
        support[index] = estimate.supported
    pooled = paule_mandel_pool(
        coordinates,
        variance,
        support=support,
        variance_floor=1e-8,
        minimum_donors=2,
    )
    if np.any(~pooled.supported):
        raise ValueError("raw PM Haldane comparator lacks entity support")
    return {
        "kind": "raw_haldane",
        "estimator": "raw Haldane log-linear Paule-Mandel transfer",
        "source_coordinate": pooled.mean.reshape(-1).tolist(),
        "certificate": {
            "variance_convention": "Haldane delta-method sampling variance",
            "support_count_range": [
                int(pooled.support_count.min()),
                int(pooled.support_count.max()),
            ],
        },
    }


def _classical_model(tables: np.ndarray, family: str, centered: bool) -> dict[str, Any]:
    values = np.asarray(tables)
    coordinates = np.empty(values.shape[:3], dtype=float)
    variance = np.empty_like(coordinates)
    support = np.empty(values.shape[:3], dtype=bool)
    function = (
        signed_pearson_coordinate if family == "pearson" else signed_deviance_coordinate
    )
    for index in np.ndindex(values.shape[:3]):
        table = values[index]
        estimate = centered_classical_coordinate(table, statistic=family)
        coordinate = estimate.centered_coordinate if centered else function(table)
        total = float(table.sum())
        coordinates[index] = coordinate / math.sqrt(total)
        variance[index] = estimate.null_variance / total
        support[index] = estimate.supported
    pooled = paule_mandel_pool(
        coordinates,
        variance,
        support=support,
        variance_floor=1e-8,
        minimum_donors=2,
    )
    if np.any(~pooled.supported):
        raise ValueError("classical residual comparator lacks entity support")
    return {
        "kind": "classical_residual",
        "family": family,
        "centered": bool(centered),
        "source_coordinate": pooled.mean.reshape(-1).tolist(),
        "estimator": (
            f"{'exact-null-centered' if centered else 'raw'} signed-root "
            f"{'Poisson deviance' if family == 'deviance' else 'Pearson'} "
            "Paule-Mandel residual transfer"
        ),
        "certificate": {
            "sample_size_normalized": True,
            "source_normalization": "coordinate/sqrt(n_source)",
            "target_normalization": "pooled_coordinate*sqrt(n_target)",
            "variance_convention": "exact fixed-margin null variance divided by n",
            "support_count_range": [
                int(pooled.support_count.min()),
                int(pooled.support_count.max()),
            ],
        },
    }


def _integer_margins(
    rows: np.ndarray, columns: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(rows, dtype=float)
    second = np.asarray(columns, dtype=float)
    if (
        first.shape != (2,)
        or second.shape != (2,)
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
        or np.any(first < 0.0)
        or np.any(second < 0.0)
        or not np.array_equal(first, np.rint(first))
        or not np.array_equal(second, np.rint(second))
        or first.sum() != second.sum()
        or first.sum() <= 0.0
    ):
        raise ValueError("target margins must be matched nonnegative integer pairs")
    return first.astype(np.int64), second.astype(np.int64)


def _canonical_margin_table(rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    first, second = _integer_margins(rows, columns)
    lower = max(0, int(first[0] + second[0] - first.sum()))
    return np.asarray(
        [
            [lower, int(first[0] - lower)],
            [int(second[0] - lower), int(first[1] - second[0] + lower)],
        ],
        dtype=np.int64,
    )


def _haldane_statistic(x: float, rows: np.ndarray, columns: np.ndarray) -> float:
    r, c = _integer_margins(rows, columns)
    value = float(x)
    cells = np.asarray(
        [value, r[0] - value, c[0] - value, r[1] - c[0] + value],
        dtype=float,
    )
    if np.any(cells < -1e-12):
        raise ValueError("upper-left cell is outside fixed-margin support")
    corrected = np.maximum(cells, 0.0) + 0.5
    return float(
        math.log(corrected[0])
        + math.log(corrected[3])
        - math.log(corrected[1])
        - math.log(corrected[2])
    )


def _direct_haldane_table(
    coordinate: float,
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    centered: bool,
) -> np.ndarray:
    """Diagnostic direct h-inverse; not used for confirmatory prediction."""

    r, c = _integer_margins(rows, columns)
    total = int(r.sum())
    lower = float(max(0, int(r[0] + c[0] - total)))
    upper = float(min(r[0], c[0]))
    if upper <= lower:
        return _canonical_margin_table(r, c).astype(float)
    _, h, _, null_mean = _moment_support(
        "haldane", int(r[0]), int(r[1]), int(c[0]), int(c[1])
    )
    target = float(coordinate) + (null_mean if centered else 0.0)
    left_value = float(h[0])
    right_value = float(h[-1])
    tolerance = 1e-12 * max(1.0, abs(left_value), abs(right_value))
    if target < left_value - tolerance or target > right_value + tolerance:
        raise ValueError("Haldane coordinate is outside attainable fixed-margin range")
    if target <= left_value:
        x = lower
    elif target >= right_value:
        x = upper
    else:
        x = float(
            brentq(
                lambda value: _haldane_statistic(value, r, c) - target,
                lower,
                upper,
                xtol=1e-12,
                rtol=8.0 * np.finfo(float).eps,
                maxiter=128,
            )
        )
    return np.asarray([[x, r[0] - x], [c[0] - x, r[1] - c[0] + x]], dtype=float)


@lru_cache(maxsize=16384)
def _moment_support(
    family: str, r0: int, r1: int, c0: int, c1: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rows, columns = _integer_margins(np.asarray([r0, r1]), np.asarray([c0, c1]))
    total = int(rows.sum())
    lower = max(0, int(rows[0] + columns[0] - total))
    upper = min(int(rows[0]), int(columns[0]))
    support = np.arange(lower, upper + 1, dtype=float)
    logbase = (
        gammaln(columns[0] + 1.0)
        - gammaln(support + 1.0)
        - gammaln(columns[0] - support + 1.0)
        + gammaln(columns[1] + 1.0)
        - gammaln(rows[0] - support + 1.0)
        - gammaln(columns[1] - rows[0] + support + 1.0)
    )
    statistics = np.empty(len(support), dtype=float)
    for index, value in enumerate(support.astype(int)):
        table = np.asarray(
            [
                [value, int(rows[0] - value)],
                [int(columns[0] - value), int(rows[1] - columns[0] + value)],
            ]
        )
        if family == "haldane":
            statistics[index] = _haldane_statistic(float(value), rows, columns)
        elif family == "pearson":
            statistics[index] = signed_pearson_coordinate(table)
        elif family == "deviance":
            statistics[index] = signed_deviance_coordinate(table)
        else:
            raise ValueError("unknown moment-calibration statistic")
    if len(statistics) > 1 and np.any(np.diff(statistics) <= 0.0):
        raise FloatingPointError("fixed-margin statistic is not strictly monotone")
    probability = np.exp(logbase - logsumexp(logbase))
    null_mean = float(probability @ statistics)
    for array in (support, statistics, logbase):
        array.setflags(write=False)
    return support, statistics, logbase, null_mean


def _moment_calibrated_table(
    coordinate: float,
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    family: str,
    centered: bool,
) -> tuple[np.ndarray, float]:
    """Reconstruct by an exact fixed-margin exponential-tilt moment equation."""

    r, c = _integer_margins(rows, columns)
    support, statistic, logbase, null_mean = _moment_support(
        family, int(r[0]), int(r[1]), int(c[0]), int(c[1])
    )
    if len(support) == 1:
        theta = 0.0
        expected = float(support[0])
    else:
        offset = null_mean if centered else 0.0
        value = float(coordinate)
        lower_coordinate = float(statistic[0] - offset)
        upper_coordinate = float(statistic[-1] - offset)
        if value < lower_coordinate or value > upper_coordinate:
            raise ValueError(
                f"{family} coordinate is outside attainable fixed-margin range"
            )
        if value == lower_coordinate:
            theta = -math.inf
            expected = float(support[0])
        elif value == upper_coordinate:
            theta = math.inf
            expected = float(support[-1])
        else:
            target = math.fsum((value, offset))
            target = max(
                float(np.nextafter(statistic[0], statistic[-1])),
                min(
                    target,
                    float(np.nextafter(statistic[-1], statistic[0])),
                ),
            )

            def moment(parameter: float) -> tuple[float, float]:
                logmass = logbase + parameter * support
                probability = np.exp(logmass - logsumexp(logmass))
                return float(probability @ statistic), float(probability @ support)

            left, right = -1.0, 1.0
            while moment(left)[0] > target:
                left *= 2.0
                if left < -1e6:
                    raise FloatingPointError("could not bracket negative moment tilt")
            while moment(right)[0] < target:
                right *= 2.0
                if right > 1e6:
                    raise FloatingPointError("could not bracket positive moment tilt")
            theta = float(
                brentq(
                    lambda parameter: moment(parameter)[0] - target,
                    left,
                    right,
                    xtol=1e-12,
                    rtol=8.0 * np.finfo(float).eps,
                    maxiter=256,
                )
            )
            expected = moment(theta)[1]
    table = np.asarray(
        [
            [expected, r[0] - expected],
            [c[0] - expected, r[1] - c[0] + expected],
        ],
        dtype=float,
    )
    return table, theta


def _haldane_table(
    coordinate: float,
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    centered: bool,
) -> tuple[np.ndarray, float]:
    return _moment_calibrated_table(
        coordinate, rows, columns, family="haldane", centered=centered
    )


def _predict_method(
    model: dict[str, Any],
    row_margins: np.ndarray,
    column_margins: np.ndarray,
    *,
    boundary_flags: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    rows = np.asarray(row_margins, dtype=np.int64)
    columns = np.asarray(column_margins, dtype=np.int64)
    if rows.shape != (len(MARKERS), 2) or columns.shape != rows.shape:
        raise ValueError("held margins must have marker by two-state shape")
    kind = model.get("kind")
    output = np.empty((len(MARKERS), len(MARKERS), 2, 2), dtype=float)
    if kind == "independence":
        coordinate = None
    else:
        coordinate = np.asarray(model.get("source_coordinate"), dtype=float)
        if (
            coordinate.shape != (len(MARKERS) ** 2,)
            or not np.isfinite(coordinate).all()
        ):
            raise ValueError("frozen source coordinate has invalid shape or values")
        coordinate = coordinate.reshape(len(MARKERS), len(MARKERS))

    for first in range(len(MARKERS)):
        for second in range(len(MARKERS)):
            r, c = _integer_margins(rows[first], columns[second])
            total = int(r.sum())
            lower = max(0, int(r[0] + c[0] - total))
            upper = min(int(r[0]), int(c[0]))
            degenerate = lower == upper
            if kind == "centered_haldane":
                table, theta = _haldane_table(
                    float(coordinate[first, second]), r, c, centered=True
                )
                output[first, second] = table
            elif kind == "raw_haldane":
                table, theta = _haldane_table(
                    float(coordinate[first, second]), r, c, centered=False
                )
                output[first, second] = table
            elif kind == "classical_residual":
                family = model.get("family")
                if family not in {"pearson", "deviance"}:
                    raise ValueError("frozen residual family is invalid")
                statistic = float(coordinate[first, second]) * math.sqrt(float(r.sum()))
                table, theta = _moment_calibrated_table(
                    statistic,
                    r,
                    c,
                    family=family,
                    centered=model.get("centered") is True,
                )
                output[first, second] = table
            elif kind == "independence":
                output[first, second] = np.outer(r, c) / float(r.sum())
                theta = 0.0
            else:
                raise ValueError(f"unknown frozen model kind {kind}")
            predicted = output[first, second]
            if not np.isfinite(predicted).all() or np.any(predicted < 0.0):
                raise FloatingPointError("prediction is not finite and nonnegative")
            if not np.allclose(predicted.sum(axis=1), r, rtol=0.0, atol=1e-10):
                raise FloatingPointError("prediction changed a target row margin")
            if not np.allclose(predicted.sum(axis=0), c, rtol=0.0, atol=1e-10):
                raise FloatingPointError("prediction changed a target column margin")
            if boundary_flags is not None:
                if degenerate:
                    boundary_flags.append(
                        {
                            "rna_marker": MARKERS[first],
                            "adt_marker": MARKERS[second],
                            "status": "degenerate_unique_table",
                        }
                    )
                elif math.isinf(theta):
                    boundary_flags.append(
                        {
                            "rna_marker": MARKERS[first],
                            "adt_marker": MARKERS[second],
                            "status": "attainable_endpoint",
                            "theta": "+inf" if theta > 0 else "-inf",
                        }
                    )
    return output


def _sample_margins(tables: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(tables)
    if values.shape != (len(MARKERS), len(MARKERS), 2, 2):
        raise ValueError("sample table map has invalid shape")
    rows = values[:, 0].sum(axis=-1)
    columns = values[0].sum(axis=-2)
    expected_rows = np.broadcast_to(rows[:, None, :], values.sum(axis=-1).shape)
    expected_columns = np.broadcast_to(columns[None, :, :], values.sum(axis=-2).shape)
    if not np.array_equal(values.sum(axis=-1), expected_rows) or not np.array_equal(
        values.sum(axis=-2), expected_columns
    ):
        raise ValueError("entity tables do not share endpoint margins")
    return rows.astype(np.int64), columns.astype(np.int64)


def _donor_loss(
    truth: np.ndarray,
    predicted: np.ndarray,
    informative: np.ndarray | None = None,
) -> float:
    """Mean per-entity multinomial deviance with physical samples equal-weighted."""

    observed = np.asarray(truth, dtype=float).reshape(-1, 2, 2)
    fitted = np.asarray(predicted, dtype=float).reshape(observed.shape)
    support = (
        _informative(observed).reshape(-1)
        if informative is None
        else np.asarray(informative, dtype=bool)
    )
    if (
        support.shape != observed.shape[:1]
        or np.count_nonzero(support) < MINIMUM_INFORMATIVE_ENTITIES
    ):
        raise ValueError("target sample misses the fixed support floor")
    if not np.isfinite(fitted).all() or np.any(fitted < 0.0):
        raise FloatingPointError("prediction is not finite and nonnegative")
    if not np.allclose(
        observed.sum(axis=-1), fitted.sum(axis=-1), rtol=0.0, atol=1e-10
    ) or not np.allclose(
        observed.sum(axis=-2), fitted.sum(axis=-2), rtol=0.0, atol=1e-10
    ):
        raise FloatingPointError("prediction changed a target margin")
    positive = observed > 0.0
    if np.any(fitted[positive] == 0.0):
        return math.inf
    return numerics._donor_loss(observed, fitted, support)


def _model_losses(
    model: dict[str, Any],
    target_tables: np.ndarray,
    prediction_flags: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    values = np.asarray(target_tables)
    losses = np.empty(len(values), dtype=float)
    for index, truth in enumerate(values):
        rows, columns = _sample_margins(truth)
        flags: list[dict[str, Any]] = []
        prediction = _predict_method(
            model,
            rows,
            columns,
            boundary_flags=flags if prediction_flags is not None else None,
        )
        if prediction_flags is not None and flags:
            prediction_flags.append({"target_index": index, "flags": flags})
        losses[index] = _donor_loss(truth, prediction, _informative(truth).reshape(-1))
    if not np.isfinite(losses).all():
        raise FloatingPointError(
            "candidate assigns zero predicted mass to an observed positive cell"
        )
    return losses


def _named_prediction_flags(
    records: list[dict[str, Any]], samples: tuple[str, ...]
) -> list[dict[str, Any]]:
    return [
        {"sample": samples[record["target_index"]], "flags": record["flags"]}
        for record in records
    ]


def _comparison(
    samples: tuple[str, ...],
    primary: np.ndarray,
    comparator: np.ndarray,
    *,
    favorable_required: int | None,
) -> dict[str, Any]:
    first = np.asarray(primary, dtype=float)
    second = np.asarray(comparator, dtype=float)
    if (
        first.shape != (len(samples),)
        or second.shape != first.shape
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
        or second.mean() <= 0.0
    ):
        raise ValueError("paired comparison requires one finite loss per sample")
    difference = first - second
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(
        0, len(samples), size=(BOOTSTRAPS, len(samples)), endpoint=False
    )
    interval = np.quantile(difference[indices].mean(axis=1), [0.025, 0.975])
    relative = 1.0 - float(first.mean() / second.mean())
    favorable = int(np.count_nonzero(difference < 0.0))
    passes = None
    if favorable_required is not None:
        passes = bool(
            relative >= 0.05 and interval[1] < 0.0 and favorable >= favorable_required
        )
    return {
        "primary_mean_deviance_per_cell": float(first.mean()),
        "comparator_mean_deviance_per_cell": float(second.mean()),
        "relative_reduction": relative,
        "paired_difference_95_ci": interval.tolist(),
        "bootstrap_draws": BOOTSTRAPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_unit": "physical sample",
        "favorable_samples": favorable,
        "required_favorable_samples": favorable_required,
        "sample_differences_primary_minus_comparator": {
            sample: float(value) for sample, value in zip(samples, difference)
        },
        "passes": passes,
    }


def _configuration(config: tuple[int, float, float]) -> dict[str, Any]:
    neighbors, ridge, graph = config
    return {
        "graph_neighbors": int(neighbors),
        "ridge_penalty": float(ridge),
        "graph_penalty": float(graph),
        "variance_floor": 1e-8,
        "precision_floor": 1e-8,
        "minimum_donors": 2,
        "maximum_condition_number": MAXIMUM_CONDITION_NUMBER,
    }


def _attach_graph(
    model: dict[str, Any], config: tuple[int, float, float], graph: dict[str, Any]
) -> dict[str, Any]:
    return {**model, "selected_configuration": _configuration(config), "graph": graph}


def _fit_method_panel(
    records: dict[str, dict[str, Any]],
    samples: tuple[str, ...],
    config: tuple[int, float, float],
    residual_choice: tuple[str, bool],
    unstructured_ridge: float,
) -> dict[str, dict[str, Any]]:
    tables = _tables(records, samples, "tables")
    destroyed_tables = _tables(records, samples, "destroyed_tables")
    neighbors, ridge, graph_penalty = config
    first, second, graph = _graphs(records, samples, neighbors)
    primary = _attach_graph(
        _field_model(
            tables,
            first,
            second,
            ridge=ridge,
            graph=graph_penalty,
            label="marginal-profile product-graph centered PM Haldane field",
        ),
        config,
        graph,
    )
    destroyed = _attach_graph(
        _field_model(
            destroyed_tables,
            first,
            second,
            ridge=ridge,
            graph=graph_penalty,
            label="destroyed-link centered PM Haldane field",
        ),
        config,
        graph,
    )
    permuted_first = _permuted_incidence(first, "rna")
    permuted_second = _permuted_incidence(second, "adt")
    permuted_graph = {
        **graph,
        "rna_incidence": permuted_first.tolist(),
        "adt_incidence": permuted_second.tolist(),
        "rna_incidence_sha256": _array_sha256(permuted_first),
        "adt_incidence_sha256": _array_sha256(permuted_second),
    }
    label_permuted = _attach_graph(
        _field_model(
            tables,
            permuted_first,
            permuted_second,
            ridge=ridge,
            graph=graph_penalty,
            label="label-permuted product-graph centered PM Haldane field",
        ),
        config,
        permuted_graph,
    )
    return {
        "primary": primary,
        "best_residual": _classical_model(tables, *residual_choice),
        "raw_pm_haldane": _raw_haldane_model(tables),
        "unstructured_centered_pm": _unstructured_model(tables, unstructured_ridge),
        "destroyed_link": destroyed,
        "label_permuted_graph": label_permuted,
        "independence": {
            "kind": "independence",
            "estimator": "fixed-margin conditional independence",
        },
    }


def _pilot_analysis(
    records: dict[str, dict[str, Any]],
    calibration_samples: tuple[str, ...],
    pilot_samples: tuple[str, ...],
) -> dict[str, Any]:
    calibration_tables = _tables(records, calibration_samples, "tables")
    pilot_tables = _tables(records, pilot_samples, "tables")

    candidate_rows = []
    for config in CONFIG_GRID:
        neighbors, ridge, graph_penalty = config
        first, second, graph = _graphs(records, calibration_samples, neighbors)
        try:
            model = _attach_graph(
                _field_model(
                    calibration_tables,
                    first,
                    second,
                    ridge=ridge,
                    graph=graph_penalty,
                    label="marginal-profile product-graph centered PM Haldane field",
                ),
                config,
                graph,
            )
            flags: list[dict[str, Any]] = []
            losses = _model_losses(model, pilot_tables, flags)
            candidate_rows.append(
                {
                    "configuration": _configuration(config),
                    "status": "EVALUATED",
                    "mean_pilot_deviance_per_cell": float(losses.mean()),
                    "pilot_losses": {
                        sample: float(value)
                        for sample, value in zip(pilot_samples, losses)
                    },
                    "prediction_flags": _named_prediction_flags(flags, pilot_samples),
                }
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            candidate_rows.append(
                {
                    "configuration": _configuration(config),
                    "status": "REFUSED",
                    "reason": str(error),
                }
            )
    successful = [row for row in candidate_rows if row["status"] == "EVALUATED"]
    if not successful:
        raise RuntimeError("all eight primary configurations refused on pilot")
    selected_row = min(
        successful,
        key=lambda row: (
            row["mean_pilot_deviance_per_cell"],
            row["configuration"]["graph_neighbors"],
            row["configuration"]["ridge_penalty"],
            row["configuration"]["graph_penalty"],
        ),
    )
    selected_config = (
        int(selected_row["configuration"]["graph_neighbors"]),
        float(selected_row["configuration"]["ridge_penalty"]),
        float(selected_row["configuration"]["graph_penalty"]),
    )

    classical_rows = []
    for family, centered in CLASSICAL_GRID:
        try:
            model = _classical_model(calibration_tables, family, centered)
            flags = []
            losses = _model_losses(model, pilot_tables, flags)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            classical_rows.append(
                {
                    "family": family,
                    "centered": centered,
                    "status": "REFUSED",
                    "reason": str(error),
                }
            )
            continue
        classical_rows.append(
            {
                "family": family,
                "centered": centered,
                "status": "EVALUATED",
                "mean_pilot_deviance_per_cell": float(losses.mean()),
                "pilot_losses": {
                    sample: float(value) for sample, value in zip(pilot_samples, losses)
                },
                "prediction_flags": _named_prediction_flags(flags, pilot_samples),
            }
        )
    successful_classical = [
        row for row in classical_rows if row["status"] == "EVALUATED"
    ]
    if not successful_classical:
        raise RuntimeError("all matched Pearson/deviance comparators refused on pilot")
    selected_classical = min(
        successful_classical,
        key=lambda row: (
            row["mean_pilot_deviance_per_cell"],
            row["family"],
            row["centered"],
        ),
    )
    residual_choice = (
        str(selected_classical["family"]),
        bool(selected_classical["centered"]),
    )

    unstructured_rows = []
    for ridge in (0.0, 0.01, 0.1):
        try:
            model = _unstructured_model(calibration_tables, ridge)
            flags = []
            losses = _model_losses(model, pilot_tables, flags)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            unstructured_rows.append(
                {"ridge_penalty": ridge, "status": "REFUSED", "reason": str(error)}
            )
            continue
        unstructured_rows.append(
            {
                "ridge_penalty": ridge,
                "status": "EVALUATED",
                "mean_pilot_deviance_per_cell": float(losses.mean()),
                "pilot_losses": {
                    sample: float(value) for sample, value in zip(pilot_samples, losses)
                },
                "prediction_flags": _named_prediction_flags(flags, pilot_samples),
            }
        )
    successful_unstructured = [
        row for row in unstructured_rows if row["status"] == "EVALUATED"
    ]
    if not successful_unstructured:
        raise RuntimeError("all ridge-only PM Haldane ablations refused on pilot")
    selected_unstructured_ridge = float(
        min(
            successful_unstructured,
            key=lambda row: (row["mean_pilot_deviance_per_cell"], row["ridge_penalty"]),
        )["ridge_penalty"]
    )

    pilot_models = _fit_method_panel(
        records,
        calibration_samples,
        selected_config,
        residual_choice,
        selected_unstructured_ridge,
    )
    pilot_losses = {}
    pilot_prediction_flags = {}
    for name, model in pilot_models.items():
        flags = []
        pilot_losses[name] = _model_losses(model, pilot_tables, flags)
        pilot_prediction_flags[name] = _named_prediction_flags(flags, pilot_samples)
    comparisons = {
        name: _comparison(
            pilot_samples,
            pilot_losses["primary"],
            pilot_losses[name],
            favorable_required=19 if name in PROMOTION_COMPARATORS else None,
        )
        for name in METHODS[1:]
    }
    passes = all(comparisons[name]["passes"] for name in PROMOTION_COMPARATORS)

    frozen_models = None
    all_graph = None
    if passes:
        frozen_models = _fit_method_panel(
            records,
            calibration_samples + pilot_samples,
            selected_config,
            residual_choice,
            selected_unstructured_ridge,
        )
        all_graph = frozen_models["primary"]["graph"]
    return {
        "status": "PILOT_PASS" if passes else "PILOT_FAIL",
        "configuration_grid": [_configuration(config) for config in CONFIG_GRID],
        "primary_candidate_evaluations": candidate_rows,
        "classical_candidate_evaluations": classical_rows,
        "unstructured_candidate_evaluations": unstructured_rows,
        "selection": {
            "selected_primary_configuration": _configuration(selected_config),
            "selected_classical_residual": {
                "family": residual_choice[0],
                "centered": residual_choice[1],
            },
            "selected_unstructured_ridge_penalty": selected_unstructured_ridge,
            "fit_samples": list(calibration_samples),
            "selection_samples": list(pilot_samples),
            "refit_samples_after_gate": list(calibration_samples + pilot_samples),
            "retuned_after_gate": False,
        },
        "pilot_losses": {
            name: {sample: float(value) for sample, value in zip(pilot_samples, losses)}
            for name, losses in pilot_losses.items()
        },
        "pilot_prediction_flags": pilot_prediction_flags,
        "pilot_comparisons": comparisons,
        "promotion_comparators": list(PROMOTION_COMPARATORS),
        "passes_pilot_gate": passes,
        "frozen_source_models": frozen_models,
        "all_development_graph": all_graph,
    }


def fit_pilot(
    source_path: Path,
    reduced_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Select on 24 pilot samples, gate, then refit once on all 36."""

    _require_designated_paths(
        "fit-pilot",
        (
            ("source manifest", source_path, DEFAULT_SOURCE_MANIFEST),
            ("reduced development", reduced_path, DEFAULT_REDUCED),
            ("output", output_path, DEFAULT_PILOT),
        ),
    )
    reduced = _validated_reduced(reduced_path, source_path)
    records = reduced["by_sample"]
    calibration_samples = reduced["calibration_samples"]
    pilot_samples = reduced["pilot_samples"]
    analysis = _pilot_analysis(records, calibration_samples, pilot_samples)

    payload = {
        "schema": "combat-citeseq-pilot-fit/1.0",
        "created_at_utc": _timestamp(),
        "source_manifest_sha256": reduced["source"]["source_manifest_sha256"],
        "reduced_development_sha256": _sha256(reduced_path),
        "development_authorization_sha256": reduced["payload"][
            "development_authorization"
        ]["authorization_sha256"],
        "protocol_sha256": reduced["payload"]["development_authorization"][
            "binding_sha256"
        ]["protocol"],
        "designation_sha256": reduced["payload"]["development_authorization"][
            "binding_sha256"
        ]["designation"],
        "runner_sha256": _sha256(Path(__file__)),
        **analysis,
        "reconstruction": {
            "primary": "exact fixed-margin moment-calibrated exponential tilt of centered Haldane statistic",
            "classical": "exact fixed-margin moment-calibrated exponential tilt after target sqrt(n) restoration",
            "out_of_range": "refuse; never clip",
            "direct_h_inverse": "diagnostic only and not used for prediction",
        },
    }
    _write_json(output_path, payload, exclusive=True)
    return payload


def _validated_pilot(
    pilot_path: Path,
    source_path: Path,
    reduced_path: Path,
    *,
    require_pass: bool,
) -> dict[str, Any]:
    payload = _read_json(pilot_path)
    source_hash = _sha256(source_path)
    source_payload = _read_json(source_path)
    records = _sample_records(source_payload)
    assign_roles(records)
    calibration_samples = _samples_for_ids(records, CALIBRATION_IDS)
    pilot_samples = _samples_for_ids(records, PILOT_IDS)
    expected_top_level = {
        "schema",
        "created_at_utc",
        "source_manifest_sha256",
        "reduced_development_sha256",
        "development_authorization_sha256",
        "protocol_sha256",
        "designation_sha256",
        "runner_sha256",
        "status",
        "configuration_grid",
        "primary_candidate_evaluations",
        "classical_candidate_evaluations",
        "unstructured_candidate_evaluations",
        "selection",
        "pilot_losses",
        "pilot_prediction_flags",
        "pilot_comparisons",
        "promotion_comparators",
        "passes_pilot_gate",
        "frozen_source_models",
        "all_development_graph",
        "reconstruction",
    }
    expected_reconstruction = {
        "primary": "exact fixed-margin moment-calibrated exponential tilt of centered Haldane statistic",
        "classical": "exact fixed-margin moment-calibrated exponential tilt after target sqrt(n) restoration",
        "out_of_range": "refuse; never clip",
        "direct_h_inverse": "diagnostic only and not used for prediction",
    }
    if (
        set(payload) != expected_top_level
        or payload.get("schema") != "combat-citeseq-pilot-fit/1.0"
        or not _is_utc_timestamp(payload.get("created_at_utc"))
        or payload.get("source_manifest_sha256") != source_hash
        or payload.get("reduced_development_sha256") != _sha256(reduced_path)
        or payload.get("runner_sha256") != _sha256(Path(__file__))
        or payload.get("protocol_sha256")
        != _sha256(ROOT / DEVELOPMENT_BINDING_PATHS["protocol"])
        or payload.get("designation_sha256")
        != _sha256(ROOT / DEVELOPMENT_BINDING_PATHS["designation"])
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(payload.get("development_authorization_sha256"))
        )
        or payload.get("passes_pilot_gate")
        is not (payload.get("status") == "PILOT_PASS")
        or payload.get("reconstruction") != expected_reconstruction
    ):
        raise PermissionError("pilot result differs from the frozen runner/source")
    reduced = _validated_reduced(reduced_path, source_path)
    if (
        reduced["calibration_samples"] != calibration_samples
        or reduced["pilot_samples"] != pilot_samples
        or payload.get("development_authorization_sha256")
        != reduced["payload"]["development_authorization"]["authorization_sha256"]
    ):
        raise PermissionError("pilot and reduced-development sample orders differ")
    replay = _pilot_analysis(reduced["by_sample"], calibration_samples, pilot_samples)
    for field, expected in replay.items():
        if _canonical_json_sha256(payload.get(field)) != _canonical_json_sha256(
            expected
        ):
            raise PermissionError(f"pilot {field} does not replay exactly")
    if require_pass and replay["status"] != "PILOT_PASS":
        raise PermissionError("pilot gate did not authorize held-margin prediction")
    return payload


def _validated_margin_authorization(
    path: Path,
    source_path: Path,
    pilot_path: Path,
    authorization_commit: str,
) -> dict[str, Any]:
    payload = _read_json(path)
    if (
        payload.get("schema") != "combat-citeseq-held-rna-margin-authorization/1.0"
        or payload.get("status") != "RNA_MARGIN_ACCESS_AUTHORIZED"
    ):
        raise PermissionError("held RNA margin access is disabled")
    pilot = _read_json(pilot_path)
    expected = {
        "runner_sha256": _sha256(Path(__file__)),
        "source_manifest_sha256": _sha256(source_path),
        "pilot_result_sha256": _sha256(pilot_path),
        "protocol_sha256": _sha256(ROOT / DEVELOPMENT_BINDING_PATHS["protocol"]),
        "designation_sha256": _sha256(ROOT / DEVELOPMENT_BINDING_PATHS["designation"]),
        "reduced_development_sha256": pilot.get("reduced_development_sha256"),
        "development_authorization_sha256": pilot.get(
            "development_authorization_sha256"
        ),
    }
    if (
        pilot.get("protocol_sha256") != expected["protocol_sha256"]
        or pilot.get("designation_sha256") != expected["designation_sha256"]
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(expected["reduced_development_sha256"])
        )
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(expected["development_authorization_sha256"])
        )
    ):
        raise PermissionError("pilot transitive bindings differ from freeze")
    if set(payload) != {
        "schema",
        "status",
        "public_pilot_commit",
        *expected,
    }:
        raise PermissionError("held RNA margin authorization fields differ")
    for key, value in expected.items():
        if payload.get(key) != value:
            raise PermissionError(f"held RNA margin authorization {key} differs")
    commit = payload.get("public_pilot_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise PermissionError("held RNA margin authorization lacks public pilot commit")
    if (
        _immutable_public_bytes(_relative(pilot_path), commit, "pilot result")
        != pilot_path.read_bytes()
    ):
        raise PermissionError("immutable public pilot result differs")
    public_bytes = _immutable_public_bytes(
        _relative(path), authorization_commit, "held RNA margin authorization"
    )
    if public_bytes != path.read_bytes():
        raise PermissionError("immutable public held RNA margin authorization differs")
    return {
        "authorization_sha256": _sha256(path),
        "public_authorization_commit": authorization_commit,
        "public_pilot_commit": commit,
    }


def _held_rna_worker(
    source_path_string: str,
    pilot_path_string: str,
    authorization_path_string: str,
    authorization_commit: str,
    held_samples: tuple[str, ...],
    connection: Any,
) -> None:
    """Child process: emit only aggregate RNA margins and content digests."""

    try:
        source_path = Path(source_path_string)
        _validated_margin_authorization(
            Path(authorization_path_string),
            source_path,
            Path(pilot_path_string),
            authorization_commit,
        )
        source = _validated_source(source_path, verify_hash=False)
        if any(
            source["roles"].get(sample) not in {"held_donor", "held_site"}
            for sample in held_samples
        ):
            raise PermissionError("RNA margin worker received a nonheld sample")
        selections = _selected_sample_rows(source["h5ad"], source["records"])
        counts = _read_modality(
            source["h5ad"], source["payload"], selections, list(held_samples), "rna"
        )
        samples: dict[str, Any] = {}
        for sample in held_samples:
            states = (_integer_counts(counts[sample], "RNA") > 0).astype(np.uint8)
            ones = states.sum(axis=1).astype(np.int64)
            margins = np.column_stack((CELL_BUDGET - ones, ones))
            samples[sample] = {
                "rna_margins": margins.tolist(),
                "rna_margin_sha256": _array_sha256(margins),
                "selected_barcode_sha256": selections[sample][
                    "selected_barcode_sha256"
                ],
                "eligible_pool_cells": selections[sample]["eligible_pool_cells"],
            }
        del counts, states
        connection.send({"status": "PASS", "samples": samples})
    except Exception as error:
        connection.send(
            {
                "status": "REFUSED",
                "error_type": type(error).__name__,
                "reason": str(error),
            }
        )
    finally:
        connection.close()


def _extract_held_rna_margins(
    source_path: Path,
    pilot_path: Path,
    authorization_path: Path,
    authorization_commit: str,
    held_samples: tuple[str, ...],
) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_held_rna_worker,
        args=(
            str(source_path.resolve()),
            str(pilot_path.resolve()),
            str(authorization_path.resolve()),
            authorization_commit,
            held_samples,
            sender,
        ),
        name="combat-held-rna-margin-extractor",
    )
    process.start()
    sender.close()
    if not receiver.poll(6 * 60 * 60):
        receiver.close()
        process.terminate()
        process.join(timeout=30)
        raise TimeoutError("held RNA margin subprocess exceeded six hours")
    try:
        payload = receiver.recv()
    except EOFError as error:
        process.join(timeout=30)
        raise RuntimeError(
            "held RNA margin subprocess exited without a record"
        ) from error
    finally:
        receiver.close()
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join(timeout=30)
        raise RuntimeError("held RNA margin subprocess did not exit after reporting")
    if process.exitcode != 0 or payload.get("status") != "PASS":
        raise RuntimeError(
            f"held RNA margin subprocess refused: {payload.get('reason', process.exitcode)}"
        )
    return payload


def predict_held_margins(
    source_path: Path,
    pilot_path: Path,
    margin_authorization_path: Path,
    margin_authorization_commit: str,
    attempt_path: Path,
    output_path: Path,
    reduced_path: Path | None = None,
) -> dict[str, Any]:
    """Seal held predictions after aggregate RNA-margin extraction."""

    reduced_path = DEFAULT_REDUCED if reduced_path is None else reduced_path
    _require_designated_paths(
        "predict-held-margins",
        (
            ("source manifest", source_path, DEFAULT_SOURCE_MANIFEST),
            ("pilot result", pilot_path, DEFAULT_PILOT),
            ("reduced development", reduced_path, DEFAULT_REDUCED),
            (
                "margin authorization",
                margin_authorization_path,
                DEFAULT_MARGIN_AUTHORIZATION,
            ),
            ("attempt", attempt_path, DEFAULT_PREDICTION_ATTEMPT),
            ("output", output_path, DEFAULT_PREDICTION),
        ),
    )
    pilot = _validated_pilot(pilot_path, source_path, reduced_path, require_pass=True)
    permit = _validated_margin_authorization(
        margin_authorization_path,
        source_path,
        pilot_path,
        margin_authorization_commit,
    )
    source = _validated_source(source_path, verify_hash=True)
    if attempt_path.exists() or output_path.exists():
        raise FileExistsError("held prediction attempt or result already exists")
    held_samples = tuple(
        record["sample"]
        for record in source["records"]
        if source["roles"][record["sample"]] in {"held_donor", "held_site"}
    )
    held_samples = tuple(sorted(held_samples))
    if len(held_samples) != EXPECTED_HELD_DONOR + EXPECTED_HELD_SITE:
        raise AssertionError("held split must contain exactly 61 samples")
    _write_json(
        attempt_path,
        {
            "schema": "combat-citeseq-held-prediction-attempt/1.0",
            "status": "TERMINAL_ATTEMPT_STARTED",
            "created_at_utc": _timestamp(),
            "source_manifest_sha256": source["source_manifest_sha256"],
            "pilot_result_sha256": _sha256(pilot_path),
            "margin_authorization_sha256": permit["authorization_sha256"],
            "public_margin_authorization_commit": permit["public_authorization_commit"],
            "runner_sha256": _sha256(Path(__file__)),
            "held_margin_request_begins_after_this_record": True,
            "selected_row_csr_structural_scan_authorized": True,
            "held_adt_numeric_data_access_authorized": False,
        },
        exclusive=True,
    )
    aggregate = _extract_held_rna_margins(
        source_path,
        pilot_path,
        margin_authorization_path,
        margin_authorization_commit,
        held_samples,
    )
    models = pilot["frozen_source_models"]
    fixed_adt_margins = np.tile(
        np.asarray([[CELL_BUDGET // 2, CELL_BUDGET // 2]], dtype=np.int64),
        (len(MARKERS), 1),
    )
    sample_rows = []
    for sample in held_samples:
        record = aggregate["samples"].get(sample)
        if not isinstance(record, dict):
            raise ValueError("RNA margin subprocess omitted a held sample")
        rna_margins = np.asarray(record.get("rna_margins"), dtype=np.int64)
        if (
            rna_margins.shape != (len(MARKERS), 2)
            or np.any(rna_margins.sum(axis=1) != CELL_BUDGET)
            or _array_sha256(rna_margins) != record.get("rna_margin_sha256")
        ):
            raise ValueError("held RNA margin aggregate is invalid")
        predictions = {}
        boundaries = {}
        for name in METHODS:
            flags: list[dict[str, Any]] = []
            prediction = _predict_method(
                models[name],
                rna_margins,
                fixed_adt_margins,
                boundary_flags=flags,
            )
            predictions[name] = prediction.reshape(len(MARKERS) ** 2, 4).tolist()
            boundaries[name] = flags
        sample_rows.append(
            {
                "sample": sample,
                "role": source["roles"][sample],
                "rna_margins": rna_margins.tolist(),
                "adt_margins": fixed_adt_margins.tolist(),
                "rna_margin_sha256": record["rna_margin_sha256"],
                "selected_barcode_sha256": record["selected_barcode_sha256"],
                "eligible_pool_cells": record["eligible_pool_cells"],
                "predictions": predictions,
                "boundary_tilts": boundaries,
            }
        )
    payload = {
        "schema": "combat-citeseq-held-predictions/1.0",
        "status": "FROZEN_HELD_PREDICTIONS",
        "created_at_utc": _timestamp(),
        "source_manifest_sha256": source["source_manifest_sha256"],
        "h5ad_sha256": source["h5ad_sha256"],
        "pilot_result_sha256": _sha256(pilot_path),
        "runner_sha256": _sha256(Path(__file__)),
        "prediction_attempt": {
            "path": _relative(attempt_path),
            "sha256": _sha256(attempt_path),
        },
        "held_rna_margin_authorization": permit,
        "markers": list(MARKERS),
        "cells_per_sample": CELL_BUDGET,
        "samples": sample_rows,
        "access_audit": {
            "process_boundary": "spawned aggregate RNA-margin subprocess",
            "child_output": "aggregate 9x2 RNA margins and digests only",
            "selected_row_csr_structural_access": "indptr and full indices slices",
            "numeric_data_values_decoded": "nine frozen RNA columns only",
            "held_rna_samples_read": len(held_samples),
            "held_adt_numeric_data_values_read": 0,
            "held_adt_states_or_margins_formed": 0,
            "held_rna_adt_pairings_formed": 0,
            "held_truth_tables_formed": 0,
            "cell_vectors_serialized": False,
            "adt_margins": "fixed by frozen within-sample 256/256 midrank rule",
        },
    }
    _write_json(output_path, payload, exclusive=True)
    return payload


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("public artifacts must be inside the repository") from error


def _validated_prediction(
    prediction_path: Path,
    source_path: Path,
    pilot_path: Path,
    reduced_path: Path | None = None,
) -> dict[str, Any]:
    reduced_path = DEFAULT_REDUCED if reduced_path is None else reduced_path
    prediction = _read_json(prediction_path)
    source = _validated_source(source_path, verify_hash=False)
    pilot = _validated_pilot(pilot_path, source_path, reduced_path, require_pass=True)
    attempt_binding = prediction.get("prediction_attempt")
    if not isinstance(attempt_binding, dict):
        raise PermissionError("held prediction lacks its terminal attempt binding")
    expected_attempt_path = (
        "data/confirmation/combat_citeseq/prediction_attempt_v1.json"
    )
    if attempt_binding.get("path") != expected_attempt_path:
        raise PermissionError("held prediction attempt path differs from freeze")
    attempt_path = ROOT / expected_attempt_path
    if not attempt_path.is_file() or attempt_binding.get("sha256") != _sha256(
        attempt_path
    ):
        raise PermissionError("held prediction attempt bytes differ from freeze")
    attempt = _read_json(attempt_path)
    expected_attempt_keys = {
        "schema",
        "status",
        "created_at_utc",
        "source_manifest_sha256",
        "pilot_result_sha256",
        "margin_authorization_sha256",
        "public_margin_authorization_commit",
        "runner_sha256",
        "held_margin_request_begins_after_this_record",
        "selected_row_csr_structural_scan_authorized",
        "held_adt_numeric_data_access_authorized",
    }
    if (
        set(attempt) != expected_attempt_keys
        or attempt.get("schema") != "combat-citeseq-held-prediction-attempt/1.0"
        or attempt.get("status") != "TERMINAL_ATTEMPT_STARTED"
        or not _is_utc_timestamp(attempt.get("created_at_utc"))
        or attempt.get("source_manifest_sha256") != source["source_manifest_sha256"]
        or attempt.get("pilot_result_sha256") != _sha256(pilot_path)
        or attempt.get("runner_sha256") != _sha256(Path(__file__))
        or attempt.get("held_margin_request_begins_after_this_record") is not True
        or attempt.get("selected_row_csr_structural_scan_authorized") is not True
        or attempt.get("held_adt_numeric_data_access_authorized") is not False
        or not isinstance(attempt.get("public_margin_authorization_commit"), str)
        or not re.fullmatch(
            r"[0-9a-f]{40}", attempt["public_margin_authorization_commit"]
        )
    ):
        raise PermissionError("held prediction attempt record differs from freeze")
    if (
        set(prediction)
        != {
            "schema",
            "status",
            "created_at_utc",
            "source_manifest_sha256",
            "h5ad_sha256",
            "pilot_result_sha256",
            "runner_sha256",
            "prediction_attempt",
            "held_rna_margin_authorization",
            "markers",
            "cells_per_sample",
            "samples",
            "access_audit",
        }
        or prediction.get("schema") != "combat-citeseq-held-predictions/1.0"
        or prediction.get("status") != "FROZEN_HELD_PREDICTIONS"
        or not _is_utc_timestamp(prediction.get("created_at_utc"))
        or prediction.get("source_manifest_sha256") != source["source_manifest_sha256"]
        or prediction.get("h5ad_sha256") != source["h5ad_sha256"]
        or prediction.get("pilot_result_sha256") != _sha256(pilot_path)
        or prediction.get("runner_sha256") != _sha256(Path(__file__))
        or prediction.get("markers") != list(MARKERS)
        or prediction.get("cells_per_sample") != CELL_BUDGET
    ):
        raise PermissionError("held prediction artifact differs from freeze")
    expected_access_audit = {
        "process_boundary": "spawned aggregate RNA-margin subprocess",
        "child_output": "aggregate 9x2 RNA margins and digests only",
        "selected_row_csr_structural_access": "indptr and full indices slices",
        "numeric_data_values_decoded": "nine frozen RNA columns only",
        "held_rna_samples_read": EXPECTED_HELD_DONOR + EXPECTED_HELD_SITE,
        "held_adt_numeric_data_values_read": 0,
        "held_adt_states_or_margins_formed": 0,
        "held_rna_adt_pairings_formed": 0,
        "held_truth_tables_formed": 0,
        "cell_vectors_serialized": False,
        "adt_margins": "fixed by frozen within-sample 256/256 midrank rule",
    }
    if prediction.get("access_audit") != expected_access_audit:
        raise PermissionError("held prediction access audit differs from freeze")
    recorded_permit = prediction.get("held_rna_margin_authorization")
    if not isinstance(recorded_permit, dict) or set(recorded_permit) != {
        "authorization_sha256",
        "public_authorization_commit",
        "public_pilot_commit",
    }:
        raise PermissionError("held prediction permit record differs from freeze")
    commit = recorded_permit.get("public_authorization_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise PermissionError("held prediction permit lacks an immutable commit")
    revalidated_permit = _validated_margin_authorization(
        DEFAULT_MARGIN_AUTHORIZATION,
        source_path,
        pilot_path,
        commit,
    )
    if (
        recorded_permit != revalidated_permit
        or attempt.get("margin_authorization_sha256")
        != recorded_permit["authorization_sha256"]
        or attempt.get("public_margin_authorization_commit") != commit
    ):
        raise PermissionError("held prediction permit chain does not recompute")
    rows = prediction.get("samples")
    if not isinstance(rows, list) or len(rows) != 61:
        raise ValueError("held prediction must contain exactly 61 samples")
    expected = sorted(
        sample for sample, role in source["roles"].items() if role.startswith("held_")
    )
    if [row.get("sample") for row in rows if isinstance(row, dict)] != expected:
        raise PermissionError("held prediction sample order differs from freeze")
    models = pilot["frozen_source_models"]
    selections = _selected_sample_rows(source["h5ad"], source["records"])
    fixed_adt = np.tile(
        np.asarray([[CELL_BUDGET // 2, CELL_BUDGET // 2]], dtype=np.int64),
        (len(MARKERS), 1),
    )
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "sample",
            "role",
            "rna_margins",
            "adt_margins",
            "rna_margin_sha256",
            "selected_barcode_sha256",
            "eligible_pool_cells",
            "predictions",
            "boundary_tilts",
        }:
            raise PermissionError("held prediction row fields differ from freeze")
        if row.get("role") != source["roles"][row["sample"]]:
            raise PermissionError("held prediction role differs from source manifest")
        rna = np.asarray(row.get("rna_margins"), dtype=float)
        adt = np.asarray(row.get("adt_margins"), dtype=float)
        if (
            rna.shape != (len(MARKERS), 2)
            or adt.shape != rna.shape
            or not np.array_equal(adt, fixed_adt)
            or set(row.get("predictions", {})) != set(METHODS)
            or set(row.get("boundary_tilts", {})) != set(METHODS)
        ):
            raise ValueError("held prediction row has invalid margins or methods")
        for marker_index in range(len(MARKERS)):
            _integer_margins(rna[marker_index], adt[marker_index])
        rna = rna.astype(np.int64)
        adt = adt.astype(np.int64)
        if (
            row.get("rna_margin_sha256") != _array_sha256(rna)
            or row.get("selected_barcode_sha256")
            != selections[row["sample"]]["selected_barcode_sha256"]
            or row.get("eligible_pool_cells")
            != selections[row["sample"]]["eligible_pool_cells"]
        ):
            raise PermissionError(
                "held prediction aggregate digests differ from freeze"
            )
        for name in METHODS:
            flat = np.asarray(row["predictions"][name], dtype=float)
            if flat.shape != (len(MARKERS) ** 2, 4):
                raise ValueError("held predicted table map is invalid")
            tables = flat.reshape(len(MARKERS), len(MARKERS), 2, 2)
            if not np.isfinite(tables).all() or np.any(tables < 0.0):
                raise ValueError("held predicted table map is invalid")
            expected_rows = np.broadcast_to(rna[:, None, :], tables.sum(axis=-1).shape)
            expected_columns = np.broadcast_to(
                adt[None, :, :], tables.sum(axis=-2).shape
            )
            if not np.allclose(
                tables.sum(axis=-1), expected_rows, rtol=0.0, atol=1e-10
            ) or not np.allclose(
                tables.sum(axis=-2), expected_columns, rtol=0.0, atol=1e-10
            ):
                raise ValueError("held predicted table changed a frozen margin")
            flags: list[dict[str, Any]] = []
            recomputed = _predict_method(
                models[name], rna, adt, boundary_flags=flags
            ).reshape(len(MARKERS) ** 2, 4)
            if _canonical_json_sha256(flat.tolist()) != _canonical_json_sha256(
                recomputed.tolist()
            ) or _canonical_json_sha256(
                row["boundary_tilts"][name]
            ) != _canonical_json_sha256(flags):
                raise PermissionError(
                    "held prediction or boundary flags do not recompute exactly"
                )
    return prediction


def _validated_score_authorization(
    authorization_path: Path,
    prediction_path: Path,
    source_path: Path,
    pilot_path: Path,
    authorization_commit: str,
    reduced_path: Path | None = None,
) -> dict[str, Any]:
    reduced_path = DEFAULT_REDUCED if reduced_path is None else reduced_path
    authorization = _read_json(authorization_path)
    if (
        authorization.get("schema") != "combat-citeseq-score-authorization/1.0"
        or authorization.get("status") != "OUTCOME_ACCESS_AUTHORIZED"
    ):
        raise PermissionError("held pairing access is disabled")
    prediction = _validated_prediction(
        prediction_path, source_path, pilot_path, reduced_path
    )
    expected = {
        "prediction_path": _relative(prediction_path),
        "prediction_sha256": _sha256(prediction_path),
        "prediction_bytes": prediction_path.stat().st_size,
        "runner_sha256": _sha256(Path(__file__)),
        "source_manifest_sha256": _sha256(source_path),
        "pilot_result_sha256": _sha256(pilot_path),
    }
    if set(authorization) != {
        "schema",
        "status",
        *expected,
        "public_prediction_commit",
        "public_prediction_url",
    }:
        raise PermissionError("score authorization fields differ from freeze")
    for key, value in expected.items():
        if authorization.get(key) != value:
            raise PermissionError(f"score authorization {key} differs")
    commit = authorization.get("public_prediction_commit")
    url = authorization.get("public_prediction_url")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise PermissionError("score authorization lacks immutable prediction commit")
    if not isinstance(url, str):
        raise PermissionError("score authorization lacks public prediction URL")
    parsed = urlparse(url)
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    expected_tail = ["blob", commit, *expected["prediction_path"].split("/")]
    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or parsed.query
        or parsed.fragment
        or len(parts) != len(expected_tail) + 2
        or parts[:2] != [PUBLIC_GITHUB_OWNER, PUBLIC_GITHUB_REPOSITORY]
        or parts[-len(expected_tail) :] != expected_tail
    ):
        raise PermissionError("public prediction URL is not a bound GitHub blob")
    public_bytes = _immutable_public_bytes(
        expected["prediction_path"], commit, "held prediction"
    )
    local_bytes = prediction_path.read_bytes()
    if public_bytes != local_bytes:
        raise PermissionError("immutable public prediction bytes differ")
    public_authorization = _immutable_public_bytes(
        _relative(authorization_path), authorization_commit, "score authorization"
    )
    if public_authorization != authorization_path.read_bytes():
        raise PermissionError("immutable public score authorization differs")
    return {
        "prediction": prediction,
        "authorization_sha256": _sha256(authorization_path),
        "public_authorization_commit": authorization_commit,
        "public_prediction_commit": commit,
        "public_prediction_url": url,
    }


def _exact_sign_test(values: np.ndarray) -> dict[str, Any]:
    difference = np.asarray(values, dtype=float)
    if (
        difference.ndim != 1
        or len(difference) not in {EXPECTED_HELD_DONOR, EXPECTED_HELD_SITE}
        or not np.isfinite(difference).all()
    ):
        raise ValueError("sign-test input must contain 51 or 10 finite differences")
    favorable = int(np.count_nonzero(difference < 0.0))
    sample_count = len(difference)
    numerator = sum(
        math.comb(sample_count, count) for count in range(favorable, sample_count + 1)
    )
    return {
        "one_sided_p": float(numerator / (1 << sample_count)),
        "favorable_samples": favorable,
        "sample_count": sample_count,
        "exhaustive": True,
        "zeros_are_nonfavorable": True,
        "null": "independent favorable probability equals one half",
        "tail": "at least the observed favorable-sample count",
    }


def _held_panel_report(
    samples: tuple[str, ...], losses: dict[str, np.ndarray], panel: str
) -> dict[str, Any]:
    if panel == "Oxford":
        favorable = 41
    elif panel == "St_Georges":
        favorable = 9
    else:
        raise ValueError("unknown held panel")
    comparisons = {}
    for name in METHODS[1:]:
        primary_loss = losses["primary"]
        comparator_loss = losses[name]
        finite = np.isfinite(primary_loss) & np.isfinite(comparator_loss)
        if not np.all(finite):
            comparisons[name] = {
                "status": "NUMERICAL_FAILURE",
                "nonfinite_samples": [
                    sample for sample, keep in zip(samples, finite) if not keep
                ],
                "reason": (
                    "a limiting prediction assigned zero mass to an observed "
                    "positive cell"
                ),
                "passes": False if name in PROMOTION_COMPARATORS else None,
            }
            continue
        row = _comparison(
            samples,
            primary_loss,
            comparator_loss,
            favorable_required=favorable if name in PROMOTION_COMPARATORS else None,
        )
        if name in PROMOTION_COMPARATORS:
            test = _exact_sign_test(losses["primary"] - losses[name])
            row["sign_test"] = test
            row["passes"] = bool(row["passes"] and test["one_sided_p"] <= 0.025)
        comparisons[name] = row
    ridge_only = comparisons["unstructured_centered_pm"]
    graph_specific = (
        "paired_difference_95_ci" in ridge_only
        and ridge_only["paired_difference_95_ci"][1] < 0.0
    )
    field_transfer = all(comparisons[name]["passes"] for name in PROMOTION_COMPARATORS)
    return {
        "panel": panel,
        "samples": list(samples),
        "comparisons": comparisons,
        "field_transfer_status": (
            "FIELD_TRANSFER_PASS" if field_transfer else "FIELD_TRANSFER_FAIL"
        ),
        "passes_field_transfer": field_transfer,
        "passes_primary_confirmation": field_transfer,
        "supports_graph_specific_superiority": bool(graph_specific),
        "primary_method_status": (
            "PRIMARY_METHOD_PASS"
            if field_transfer and graph_specific
            else "PRIMARY_METHOD_FAIL"
        ),
        "passes_primary_method": bool(field_transfer and graph_specific),
        "graph_specific_rule": (
            "paired upper 95% CI for primary minus pilot-selected strongest "
            "ridge-only PM Haldane is below zero"
        ),
    }


def _confirmation_decision(
    oxford_report: dict[str, Any], st_georges_report: dict[str, Any]
) -> dict[str, Any]:
    field_transfer_pass = bool(
        oxford_report["passes_field_transfer"]
        and st_georges_report["passes_field_transfer"]
    )
    graph_specific_pass = bool(
        oxford_report["supports_graph_specific_superiority"]
        and st_georges_report["supports_graph_specific_superiority"]
    )
    primary_method_pass = bool(field_transfer_pass and graph_specific_pass)
    return {
        "status": ("CONFIRMATION_PASS" if primary_method_pass else "CONFIRMATION_FAIL"),
        "field_transfer_status": (
            "FIELD_TRANSFER_PASS" if field_transfer_pass else "FIELD_TRANSFER_FAIL"
        ),
        "passes_field_transfer": field_transfer_pass,
        "primary_method_status": (
            "PRIMARY_METHOD_PASS" if primary_method_pass else "PRIMARY_METHOD_FAIL"
        ),
        "passes_primary_method": primary_method_pass,
        "passes_full_promotion": primary_method_pass,
        "supports_graph_specific_claim_in_both_panels": graph_specific_pass,
    }


def score_held(
    source_path: Path,
    pilot_path: Path,
    prediction_path: Path,
    authorization_path: Path,
    authorization_commit: str,
    attempt_path: Path,
    output_path: Path,
    reduced_path: Path | None = None,
) -> dict[str, Any]:
    """Access held linkage once, after public prediction authorization."""

    reduced_path = DEFAULT_REDUCED if reduced_path is None else reduced_path
    _require_designated_paths(
        "score-held",
        (
            ("source manifest", source_path, DEFAULT_SOURCE_MANIFEST),
            ("pilot result", pilot_path, DEFAULT_PILOT),
            ("reduced development", reduced_path, DEFAULT_REDUCED),
            ("prediction", prediction_path, DEFAULT_PREDICTION),
            ("authorization", authorization_path, DEFAULT_AUTHORIZATION),
            ("attempt", attempt_path, DEFAULT_SCORE_ATTEMPT),
            ("output", output_path, DEFAULT_SCORE),
        ),
    )
    permit = _validated_score_authorization(
        authorization_path,
        prediction_path,
        source_path,
        pilot_path,
        authorization_commit,
        reduced_path,
    )
    if output_path.exists() or attempt_path.exists():
        raise FileExistsError("held score or terminal attempt already exists")
    _write_json(
        attempt_path,
        {
            "schema": "combat-citeseq-score-attempt/1.0",
            "status": "TERMINAL_ATTEMPT_STARTED",
            "created_at_utc": _timestamp(),
            "prediction_sha256": _sha256(prediction_path),
            "authorization_sha256": permit["authorization_sha256"],
            "public_authorization_commit": permit["public_authorization_commit"],
            "held_linkage_access_begins_after_this_record": True,
        },
        exclusive=True,
    )
    source = _validated_source(source_path, verify_hash=True)
    selections = _selected_sample_rows(source["h5ad"], source["records"])
    held_samples = tuple(
        sorted(
            sample
            for sample, role in source["roles"].items()
            if role.startswith("held_")
        )
    )
    rna_counts = _read_modality(
        source["h5ad"], source["payload"], selections, list(held_samples), "rna"
    )
    adt_counts = _read_modality(
        source["h5ad"], source["payload"], selections, list(held_samples), "adt"
    )
    combat_by_sample = {
        record["sample"]: record["combat_id"] for record in source["records"]
    }
    rna_states = {
        sample: (_integer_counts(rna_counts[sample], "RNA") > 0).astype(np.uint8)
        for sample in held_samples
    }
    adt_states = {
        sample: _adt_states(
            adt_counts[sample],
            selections[sample]["barcodes"],
            combat_by_sample[sample],
            sample,
        )
        for sample in held_samples
    }
    truth = _form_tables(rna_states, adt_states, list(held_samples))
    del rna_counts, adt_counts, rna_states, adt_states

    prediction_rows = {row["sample"]: row for row in permit["prediction"]["samples"]}
    losses = {name: np.empty(len(held_samples), dtype=float) for name in METHODS}
    truth_digests = {}
    informative_counts = {}
    excluded_pairs = {}
    for sample_index, sample in enumerate(held_samples):
        sample_truth = truth[sample_index]
        support = _informative(sample_truth).reshape(-1)
        if int(support.sum()) < MINIMUM_INFORMATIVE_ENTITIES:
            raise ValueError(
                f"held sample {sample} has fewer than 64 informative entities"
            )
        informative_counts[sample] = int(support.sum())
        excluded_pairs[sample] = [
            f"{MARKERS[index // len(MARKERS)]}:{MARKERS[index % len(MARKERS)]}"
            for index in np.flatnonzero(~support)
        ]
        rows, columns = _sample_margins(sample_truth)
        frozen = prediction_rows[sample]
        if not np.array_equal(
            rows, np.asarray(frozen["rna_margins"])
        ) or not np.array_equal(columns, np.asarray(frozen["adt_margins"])):
            raise PermissionError("held truth margins differ from frozen prediction")
        if frozen.get("selected_barcode_sha256") != selections[sample][
            "selected_barcode_sha256"
        ] or frozen.get("rna_margin_sha256") != _array_sha256(rows):
            raise PermissionError("held truth cell or RNA-margin digest differs")
        truth_digests[sample] = _array_sha256(sample_truth)
        for name in METHODS:
            predicted = np.asarray(frozen["predictions"][name], dtype=float).reshape(
                len(MARKERS), len(MARKERS), 2, 2
            )
            losses[name][sample_index] = _donor_loss(sample_truth, predicted, support)

    oxford = tuple(
        sample for sample in held_samples if source["roles"][sample] == "held_donor"
    )
    st_georges = tuple(
        sample for sample in held_samples if source["roles"][sample] == "held_site"
    )
    index = {sample: position for position, sample in enumerate(held_samples)}

    def panel_losses(samples: tuple[str, ...]) -> dict[str, np.ndarray]:
        positions = [index[sample] for sample in samples]
        return {name: values[positions] for name, values in losses.items()}

    oxford_report = _held_panel_report(oxford, panel_losses(oxford), "Oxford")
    st_georges_report = _held_panel_report(
        st_georges, panel_losses(st_georges), "St_Georges"
    )
    decision = _confirmation_decision(oxford_report, st_georges_report)
    payload = {
        "schema": "combat-citeseq-confirmation/1.0",
        **decision,
        "created_at_utc": _timestamp(),
        "prediction_sha256": _sha256(prediction_path),
        "authorization_sha256": permit["authorization_sha256"],
        "public_prediction_commit": permit["public_prediction_commit"],
        "panels": {
            "Oxford": oxford_report,
            "St_Georges": st_georges_report,
        },
        "full_promotion_requires_both_panels": True,
        "sample_losses": {
            sample: {
                name: (
                    float(losses[name][index[sample]])
                    if np.isfinite(losses[name][index[sample]])
                    else "+inf"
                )
                for name in METHODS
            }
            for sample in held_samples
        },
        "truth_table_sha256": truth_digests,
        "informative_pair_count": informative_counts,
        "excluded_noninformative_pairs": excluded_pairs,
        "access_audit": {
            "score_authorization_validated_before_attempt": True,
            "terminal_attempt_written_before_held_linkage_access": True,
            "held_rna_reads": 1,
            "held_adt_reads": 1,
            "held_truth_table_formation_calls": 1,
            "held_truth_tables_serialized": False,
        },
    }
    _write_json(output_path, payload, exclusive=True)
    return payload


def _add_runtime_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument(
        "--h5ad",
        type=Path,
        help="local COMBAT H5AD; never written to a public artifact",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="phase", required=True)

    seal = subparsers.add_parser("seal-source")
    seal.add_argument("--preflight", type=Path, required=True)
    seal.add_argument("--output", type=Path, default=DEFAULT_SOURCE_MANIFEST)

    reduce_parser = subparsers.add_parser("reduce-development")
    _add_runtime_source_arguments(reduce_parser)
    reduce_parser.add_argument(
        "--development-authorization",
        type=Path,
        default=DEFAULT_DEVELOPMENT_AUTHORIZATION,
    )
    reduce_parser.add_argument(
        "--authorization-commit",
        required=True,
        help="immutable public commit containing the development authorization",
    )
    reduce_parser.add_argument("--output", type=Path, default=DEFAULT_REDUCED)

    fit_parser = subparsers.add_parser("fit-pilot")
    _add_runtime_source_arguments(fit_parser)
    fit_parser.add_argument("--reduced", type=Path, default=DEFAULT_REDUCED)
    fit_parser.add_argument("--output", type=Path, default=DEFAULT_PILOT)

    predict_parser = subparsers.add_parser("predict-held-margins")
    _add_runtime_source_arguments(predict_parser)
    predict_parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    predict_parser.add_argument("--reduced", type=Path, default=DEFAULT_REDUCED)
    predict_parser.add_argument(
        "--margin-authorization",
        type=Path,
        default=DEFAULT_MARGIN_AUTHORIZATION,
    )
    predict_parser.add_argument(
        "--authorization-commit",
        required=True,
        help="immutable public commit containing the margin authorization",
    )
    predict_parser.add_argument(
        "--attempt", type=Path, default=DEFAULT_PREDICTION_ATTEMPT
    )
    predict_parser.add_argument("--output", type=Path, default=DEFAULT_PREDICTION)
    predict_parser.add_argument(
        "--terminal-refusal", type=Path, default=DEFAULT_TERMINAL_REFUSAL
    )

    score_parser = subparsers.add_parser("score-held")
    _add_runtime_source_arguments(score_parser)
    score_parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    score_parser.add_argument("--reduced", type=Path, default=DEFAULT_REDUCED)
    score_parser.add_argument("--prediction", type=Path, default=DEFAULT_PREDICTION)
    score_parser.add_argument(
        "--authorization", type=Path, default=DEFAULT_AUTHORIZATION
    )
    score_parser.add_argument(
        "--authorization-commit",
        required=True,
        help="immutable public commit containing the score authorization",
    )
    score_parser.add_argument("--attempt", type=Path, default=DEFAULT_SCORE_ATTEMPT)
    score_parser.add_argument("--output", type=Path, default=DEFAULT_SCORE)
    score_parser.add_argument(
        "--terminal-refusal", type=Path, default=DEFAULT_TERMINAL_REFUSAL
    )

    args = parser.parse_args()
    if getattr(args, "h5ad", None) is not None:
        os.environ["COMBAT_CITESEQ_H5AD"] = str(args.h5ad.resolve())
    if args.phase == "seal-source":
        payload = seal_source(args.preflight, args.output)
    elif args.phase == "reduce-development":
        payload = reduce_development(
            args.source_manifest,
            args.development_authorization,
            args.authorization_commit,
            args.output,
        )
    elif args.phase == "fit-pilot":
        payload = fit_pilot(args.source_manifest, args.reduced, args.output)
    elif args.phase == "predict-held-margins":
        payload = _run_terminal_phase(
            "held_prediction",
            args.attempt,
            args.margin_authorization,
            args.terminal_refusal,
            lambda: predict_held_margins(
                args.source_manifest,
                args.pilot,
                args.margin_authorization,
                args.authorization_commit,
                args.attempt,
                args.output,
                args.reduced,
            ),
        )
    else:
        payload = _run_terminal_phase(
            "held_score",
            args.attempt,
            args.authorization,
            args.terminal_refusal,
            lambda: score_held(
                args.source_manifest,
                args.pilot,
                args.prediction,
                args.authorization,
                args.authorization_commit,
                args.attempt,
                args.output,
                args.reduced,
            ),
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
