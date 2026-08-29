"""Staged GSE288020 MGUS-to-myeloma linked RNA/ADT confirmation.

``preflight`` reads source metadata, archive bytes for hashing, and HDF5 feature
schema only. ``develop`` is the first stage allowed to read calibration and
pilot count matrices. ``predict`` invokes the bound RNA-summary firewall on
held combined HDF5 files and freezes predictions before ``score`` evaluates
held ADT counts.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Iterable, Iterator

import h5py
import numpy as np
import scipy
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import confirm_gse158769_citeseq as base  # noqa: E402
from experiments import confirm_gse314416_citeseq as model_core  # noqa: E402
from mapreg.heterogeneity_adaptive_coupling import (  # noqa: E402
    CouplingEstimationRefusal,
)


ROOT = REPO_ROOT
DATA_DIR = ROOT / "data/confirmation/gse288020_citeseq"
SOURCE_CACHE = DATA_DIR / "source_cache"
DEFAULT_MANIFEST = DATA_DIR / "source_manifest_v1.json"
DEFAULT_DESIGNATION = DATA_DIR / "candidate_designation_v1.json"
DEFAULT_FILELIST = DATA_DIR / "metadata/filelist.txt"
DEFAULT_GEO_METADATA = SOURCE_CACHE / "GSE288020_samples_brief.soft"
DEFAULT_ACCESS = ROOT / "results/development/gse288020_schema_access_v1.json"
DEFAULT_PREFLIGHT = ROOT / "results/development/gse288020_schema_preflight_v1.json"
DEFAULT_RUNTIME_SPEC = (
    ROOT / "results/development/gse288020_runtime_environment_v1.json"
)
DEFAULT_DEVELOPMENT = ROOT / "results/development/gse288020_development_v1.json"
DEFAULT_DEVELOPMENT_ATTEMPT = (
    ROOT / "results/development/gse288020_development_attempt_v1.jsonl"
)
DEFAULT_PREDICTION = ROOT / "results/gse288020_held_predictions_v1.json"
DEFAULT_PREDICTION_ATTEMPT = (
    ROOT / "results/development/gse288020_prediction_attempt_v1.jsonl"
)
DEFAULT_SCORE = ROOT / "results/gse288020_confirmation_v1.json"
DEFAULT_SCORE_ATTEMPT = ROOT / "results/development/gse288020_score_attempt_v1.jsonl"
DEFAULT_PROTOCOL = (
    ROOT / "docs/GSE288020_MGUS_TO_MYELOMA_CITESEQ_CONFIRMATION_PROTOCOL_2026-08-28.md"
)
DEFAULT_REDUCER = ROOT / "experiments/reduce_gse288020_held_rna.py"

PUBLIC_ORIGIN = "https://github.com/sushaan-k/coupling-fields-benchmark.git"
PROTOCOL_TAG = "gse288020-citeseq-v1-protocol"
DEVELOPMENT_TAG = "gse288020-citeseq-v1-development"
PREDICTION_TAG = "gse288020-citeseq-v1-predictions"

SPLIT_SALT = "GSE288020-MGUS-SPLIT-v1"
CALIBRATION_DONORS = ("R001", "R005", "R008", "R009", "R010", "R013", "R014")
PILOT_DONORS = ("R003", "R006", "R015", "R016", "R020", "R023", "R024")
HELD_DONORS = (
    "E2228",
    "E2238",
    "E2242",
    "E2243",
    "E2263",
    "E2324",
    "E2326",
    "E2328",
    "E2329",
)

MARKERS = (
    "CD45",
    "CD3",
    "CD4",
    "CD8a",
    "CD20",
    "CD19",
    "CD27",
    "CD38",
    "CD138",
    "CD14",
    "CD11b",
    "CD11c",
    "CD33",
    "CD56",
    "CD161",
    "CXCR4",
)
RNA_IDS = (
    "ENSG00000081237",
    "ENSG00000198851",
    "ENSG00000010610",
    "ENSG00000153563",
    "ENSG00000156738",
    "ENSG00000177455",
    "ENSG00000139193",
    "ENSG00000004468",
    "ENSG00000115884",
    "ENSG00000170458",
    "ENSG00000169896",
    "ENSG00000140678",
    "ENSG00000105383",
    "ENSG00000149294",
    "ENSG00000111796",
    "ENSG00000121966",
)
RNA_SYMBOLS = (
    "PTPRC",
    "CD3E",
    "CD4",
    "CD8A",
    "MS4A1",
    "CD19",
    "CD27",
    "CD38",
    "SDC1",
    "CD14",
    "ITGAM",
    "ITGAX",
    "CD33",
    "NCAM1",
    "KLRB1",
    "CXCR4",
)
ADT_IDS = MARKERS

CELL_BUDGET = 512
MINIMUM_INFORMATIVE_ENTITIES = 192
CELL_SELECTION_SALT = "GSE288020-CELL-BUDGET-v1"
ADT_TIE_SALT = "GSE288020-ADT-MIDRANK-v1"
DESTROYED_LINK_SALT = "GSE288020-DESTROYED-LINK-v1"

MINIMUM_DETECTED_GENES = 200
MAXIMUM_MITOCHONDRIAL_FRACTION = 0.10
MAXIMUM_RNA_UMIS = 70_000
ROLE_MINIMUM_DONORS = {"calibration": 6, "pilot": 6, "held": 7}
ROLE_MINIMUM_PER_AGE = {"calibration": 2, "pilot": 2, "held": 3}

NEIGHBOR_GRID = (1, 2)
HETEROGENEITY_GRID = (0.1, 1.0, 10.0)
RIDGE_GRID = (0.01, 0.1)
GRAPH_GRID = (0.0, 0.1, 1.0)
TRANSPORT_GRID = (0.5, 0.75, 1.0, 1.25)
RESIDUAL_FAMILIES = ("pearson", "deviance")
CLASSICAL_METHODS = (
    "common_effect_stratified_cmle",
    "pooled_saturated_poisson_interaction",
)
BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 20260828

CANDIDATE_SEARCH_DISCLOSURE = {
    "sequential_public_candidate_search": True,
    "immediate_public_terminal_predecessors": [
        {
            "accession": "GSE314416",
            "terminal_tag": "gse314416-citeseq-v1.2-development",
            "outcome": "terminal pilot failure",
        },
        {
            "accession": "GSE158769",
            "terminal_tag": "gse158769-citeseq-v1.1-development",
            "outcome": "terminal source-schema refusal",
        },
        {
            "accession": "GSE189050",
            "terminal_tag": "gse189050-citeseq-v1.1-development",
            "outcome": "terminal development QC/support refusal",
        },
    ],
    "earlier_public_campaign_ledger": {
        "path": "docs/FINAL_PUBLIC_EVIDENCE_LEDGER.md",
        "sha256": "1a8965a29a0b3f7ce60cd304a3bad4fb2a581b514298dfc7a5d70a80976ffa79",
        "coverage": (
            "all earlier public scored, negative, and refused panels preceding "
            "the three immediate terminal predecessors"
        ),
    },
    "inference_scope": "candidate-specific",
    "familywise_adjustment_across_public_candidates": False,
    "campaign_wide_confirmatory_error_control_claimed": False,
    "disclosed_before_gse288020_barcode_or_count_access": True,
}

PROTOCOL_BINDINGS = (
    ".gitattributes",
    ".gitignore",
    "experiments/confirm_gse288020_citeseq.py",
    "experiments/reduce_gse288020_held_rna.py",
    "tests/test_gse288020_citeseq_confirmation.py",
    "docs/GSE288020_MGUS_TO_MYELOMA_CITESEQ_CONFIRMATION_PROTOCOL_2026-08-28.md",
    "docs/FINAL_PUBLIC_EVIDENCE_LEDGER.md",
    "data/confirmation/gse288020_citeseq/source_manifest_v1.json",
    "data/confirmation/gse288020_citeseq/candidate_designation_v1.json",
    "data/confirmation/gse288020_citeseq/metadata/filelist.txt",
    "results/development/gse288020_schema_access_v1.json",
    "results/development/gse288020_schema_preflight_v1.json",
    "results/development/gse288020_runtime_environment_v1.json",
    "experiments/confirm_gse158769_citeseq.py",
    "experiments/confirm_gse314416_citeseq.py",
    "mapreg/__init__.py",
    "mapreg/classical_residuals.py",
    "mapreg/coupling_fields.py",
    "mapreg/factorial_coupling.py",
    "mapreg/heterogeneity_adaptive_coupling.py",
    "mapreg/hierarchical_conditional_coupling.py",
    "mapreg/table_prediction.py",
    "requirements.txt",
    "pyproject.toml",
)


def _timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
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


def _axis_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _salted_hash(salt: str, *values: str) -> str:
    digest = hashlib.sha256(salt.encode())
    for value in values:
        digest.update(b"\0")
        digest.update(value.encode())
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def _state_estimand() -> dict[str, Any]:
    return {
        "analysis_unit": "physical donor",
        "cell_population": (
            "the 512 filtered cells selected by the frozen donor-specific hash rank"
        ),
        "rna_state": "one if the raw RNA count is positive, zero otherwise",
        "adt_state": (
            "one for the upper 256 raw ADT counts after the frozen donor-marker "
            "tie rank, zero for the lower 256"
        ),
        "ordered_pair_table": (
            "a within-donor 2x2 table for one RNA marker and one ADT marker"
        ),
        "coupling_parameter": (
            "conditional RNA-ADT log odds given the donor-specific RNA margin "
            "and fixed 256/256 ADT rank margin"
        ),
        "recipient_prediction": (
            "the expected 2x2 table at the recipient RNA margin and fixed ADT "
            "rank margin"
        ),
        "loss": (
            "donor-equal mean Poisson deviance over ordered pairs with a "
            "nondegenerate recipient RNA margin"
        ),
    }


def _runtime_environment() -> dict[str, Any]:
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "h5py": h5py.__version__,
        },
        "hdf5": {
            "runtime_version": h5py.version.hdf5_version,
            "runtime_version_tuple": list(h5py.version.hdf5_version_tuple),
            "built_against_version_tuple": list(
                h5py.version.hdf5_built_version_tuple
            ),
            "h5py_api_version": h5py.version.api_version,
        },
    }


def _require_runtime_environment() -> dict[str, Any]:
    specification = _read_json(DEFAULT_RUNTIME_SPEC)
    expected = specification.get("required_runtime")
    observed = _runtime_environment()
    if observed != expected:
        raise PermissionError(
            "runtime environment differs from the frozen exact specification"
        )
    return observed


def _write_json_atomic_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            os.fchmod(stream.fileno(), 0o644)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_stage_available(output: Path, expected: Path, attempt: Path) -> None:
    if output.resolve() != expected.resolve():
        raise PermissionError(f"stage output must be {_relative(expected)}")
    if output.exists() or attempt.exists():
        raise PermissionError("stage already has an output or attempt record")


def _append_attempt(path: Path, payload: dict[str, Any], *, create: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if create else "a"
    with path.open(mode) as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _run_claimed_stage(
    stage: str,
    output: Path,
    expected: Path,
    attempt: Path,
    authorization_commit: str,
    runtime_environment: dict[str, Any],
    function: Any,
) -> dict[str, Any]:
    _ensure_stage_available(output, expected, attempt)
    _append_attempt(
        attempt,
        {
            "authorization_commit": authorization_commit,
            "created_at_utc": _timestamp(),
            "event": "STARTED",
            "output": _relative(expected),
            "runtime_environment": runtime_environment,
            "stage": stage,
        },
        create=True,
    )
    try:
        payload = function()
        if output.exists():
            raise RuntimeError("stage body wrote the frozen output directly")
        if not isinstance(payload, dict) or not isinstance(payload.get("status"), str):
            raise TypeError("stage body must return one payload with a string status")
    except Exception as error:
        reason = str(error).replace(str(ROOT), "<repo>")
        payload = {
            "schema": f"gse288020-{stage}-exception/1.0",
            "status": f"TERMINAL_{stage.upper()}_EXCEPTION",
            "created_at_utc": _timestamp(),
            "authorization_commit": authorization_commit,
            "error_type": type(error).__name__,
            "reason": reason,
            "terminal_rule": "This stage may not be rerun or retuned.",
        }
    if output.exists():
        output.unlink()
    payload = dict(payload)
    payload["authorization_commit"] = authorization_commit
    payload["candidate_search_disclosure"] = CANDIDATE_SEARCH_DISCLOSURE
    payload["runtime_environment"] = runtime_environment
    payload["stage"] = stage
    payload["state_estimand"] = _state_estimand()
    _write_json_atomic_new(output, payload)
    _append_attempt(
        attempt,
        {
            "authorization_commit": authorization_commit,
            "created_at_utc": _timestamp(),
            "event": "FINISHED",
            "output": _relative(expected),
            "output_sha256": _sha256(output),
            "runtime_environment": runtime_environment,
            "stage": stage,
            "status": payload["status"],
        },
        create=False,
    )
    return payload


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _validate_prior_stage(
    stage: str,
    attempt_path: Path,
    result_path: Path,
    authorization_commit: str,
    runtime_environment: dict[str, Any],
    expected_status: str,
) -> dict[str, Any]:
    try:
        records = [json.loads(line) for line in attempt_path.read_text().splitlines()]
    except (OSError, json.JSONDecodeError) as error:
        raise PermissionError(f"{stage} attempt ledger is unreadable") from error
    if len(records) != 2:
        raise PermissionError(f"{stage} attempt ledger must contain exactly two records")
    started, finished = records
    started_keys = {
        "authorization_commit",
        "created_at_utc",
        "event",
        "output",
        "runtime_environment",
        "stage",
    }
    finished_keys = {
        "authorization_commit",
        "created_at_utc",
        "event",
        "output",
        "output_sha256",
        "runtime_environment",
        "stage",
        "status",
    }
    if set(started) != started_keys or set(finished) != finished_keys:
        raise PermissionError(f"{stage} attempt ledger fields differ from protocol")
    if started["event"] != "STARTED" or finished["event"] != "FINISHED":
        raise PermissionError(f"{stage} attempt ledger events are not STARTED/FINISHED")
    expected_output = _relative(result_path)
    for record in records:
        if record["authorization_commit"] != authorization_commit:
            raise PermissionError(f"{stage} authorization commit differs")
        if record["stage"] != stage:
            raise PermissionError(f"{stage} attempt ledger stage differs")
        if record["output"] != expected_output:
            raise PermissionError(f"{stage} attempt output path differs")
        if record["runtime_environment"] != runtime_environment:
            raise PermissionError(f"{stage} attempt runtime differs")
    result = _read_json(result_path)
    if result.get("authorization_commit") != authorization_commit:
        raise PermissionError(f"{stage} result authorization commit differs")
    if result.get("stage") != stage:
        raise PermissionError(f"{stage} result stage differs")
    if result.get("runtime_environment") != runtime_environment:
        raise PermissionError(f"{stage} result runtime differs")
    if finished["output_sha256"] != _sha256(result_path):
        raise PermissionError(f"{stage} result hash differs from attempt ledger")
    if finished["status"] != result.get("status"):
        raise PermissionError(f"{stage} result status differs from attempt ledger")
    if result.get("status") != expected_status:
        raise PermissionError(
            f"{stage} status does not authorize the requested downstream stage"
        )
    return result


def _require_public_tag(tag: str, paths: Iterable[str]) -> str:
    commit = subprocess.run(
        ["git", "rev-list", "-n", "1", tag],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    remote = subprocess.run(
        ["git", "ls-remote", PUBLIC_ORIGIN, f"refs/tags/{tag}^{{}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not remote or remote.split()[0] != commit:
        raise PermissionError(f"public tag {tag} does not resolve to the local commit")
    for relative in paths:
        tagged = subprocess.run(
            ["git", "show", f"{tag}:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if tagged != (ROOT / relative).read_bytes():
            raise PermissionError(f"{relative} differs from public tag {tag}")
    return commit


def _source_path(filename: str) -> Path:
    return SOURCE_CACHE / filename


def _parse_geo_metadata(path: Path) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    current: dict[str, str] = {}

    def finish() -> None:
        if not current:
            return
        required = {"gsm", "title", "disease", "raw_age", "library", "url"}
        if set(current) != required:
            raise PermissionError(
                f"official GEO sample metadata is incomplete for {current.get('gsm')}"
            )
        disease = " ".join(current["disease"].split())
        condition = {
            "monoclonal gammopathy of uncertain significance": "MGUS",
            "Multiple Myeloma": "MM",
        }.get(disease)
        if condition is None:
            raise PermissionError(f"unknown GEO disease label {disease}")
        raw_age = " ".join(current["raw_age"].split())
        if raw_age == "mmune-Age Young":
            raw_age = "Immune-Age Young"
        if raw_age not in {"Immune-Age Young", "Immune-Age Old"}:
            raise PermissionError(f"unknown GEO immune-age label {raw_age}")
        age = raw_age.rsplit(" ", 1)[-1]
        if not " ".join(current["title"].split()).endswith(age):
            raise PermissionError("GEO title and normalized immune-age label disagree")
        library = current["library"].strip()
        if not library.endswith("BM"):
            raise PermissionError(f"unexpected GEO library name {library}")
        filename = current["url"].rsplit("/", 1)[-1]
        samples.append(
            {
                "condition": condition,
                "donor": library[:-2],
                "filename": filename,
                "gsm": current["gsm"],
                "immune_age": age,
            }
        )
        current.clear()

    for raw in path.read_text().splitlines():
        line = raw.rstrip("\r")
        if line.startswith("^SAMPLE = "):
            finish()
            current["gsm"] = line.split("=", 1)[1].strip()
        elif line.startswith("!Sample_title = "):
            current["title"] = line.split("=", 1)[1].strip()
        elif line.startswith("!Sample_characteristics_ch1 = disease state:"):
            current["disease"] = line.split("disease state:", 1)[1].strip()
        elif line.startswith("!Sample_characteristics_ch1 = age group:"):
            current["raw_age"] = line.split("age group:", 1)[1].strip()
        elif line.startswith("!Sample_description = Library name:"):
            current["library"] = line.split("Library name:", 1)[1].strip()
        elif line.startswith("!Sample_supplementary_file_1 = "):
            current["url"] = line.split("=", 1)[1].strip()
    finish()
    if len(samples) != 23:
        raise PermissionError("official GEO metadata must contain 23 samples")
    return samples


def _roles_from_official_metadata(samples: list[dict[str, str]]) -> dict[str, str]:
    roles = {
        sample["donor"]: "held"
        for sample in samples
        if sample["condition"] == "MM"
    }
    for age, calibration_count in (("Young", 4), ("Old", 3)):
        donors = [
            sample["donor"]
            for sample in samples
            if sample["condition"] == "MGUS" and sample["immune_age"] == age
        ]
        ordered = sorted(
            donors,
            key=lambda donor: (
                hashlib.sha256(f"{SPLIT_SALT}|{age}|{donor}".encode()).hexdigest(),
                donor,
            ),
        )
        for index, donor in enumerate(ordered):
            roles[donor] = "calibration" if index < calibration_count else "pilot"
    if len(roles) != 23:
        raise PermissionError("official GEO metadata did not allocate 23 donors")
    return roles


def _designation() -> dict[str, Any]:
    value = _read_json(DEFAULT_DESIGNATION)
    samples = value.get("samples", [])
    if len(samples) != 23:
        raise PermissionError("designation must contain 23 physical donors")
    donors = [sample["donor"] for sample in samples]
    if len(set(donors)) != 23:
        raise PermissionError("designation repeats a physical donor")
    roles = {
        role: {sample["donor"] for sample in samples if sample["role"] == role}
        for role in ("calibration", "pilot", "held")
    }
    expected = {
        "calibration": set(CALIBRATION_DONORS),
        "pilot": set(PILOT_DONORS),
        "held": set(HELD_DONORS),
    }
    if roles != expected:
        raise PermissionError("designation differs from the frozen donor split")
    if any(
        sample["condition"] != ("MM" if sample["role"] == "held" else "MGUS")
        for sample in samples
    ):
        raise PermissionError("condition does not match the frozen study split")
    if DEFAULT_GEO_METADATA.exists():
        official = _parse_geo_metadata(DEFAULT_GEO_METADATA)
        official_roles = _roles_from_official_metadata(official)
        expected_samples = {
            sample["donor"]: {**sample, "role": official_roles[sample["donor"]]}
            for sample in official
        }
        if {sample["donor"]: sample for sample in samples} != expected_samples:
            raise PermissionError(
                "designation differs from parsed official GEO metadata"
            )
    return value


def _validate_archive_members(manifest: dict[str, Any]) -> None:
    expected = {
        Path(record["path"]).name: record for record in manifest["h5_files"]
    }
    archive_path = ROOT / manifest["archive"]["path"]
    with tarfile.open(archive_path, "r:") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        if {member.name for member in members} != set(expected):
            raise PermissionError("TAR member names differ from the source manifest")
        for member in members:
            record = expected[member.name]
            if member.size != record["bytes"]:
                raise PermissionError(f"TAR member size differs for {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise PermissionError(f"TAR member cannot be opened for {member.name}")
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
            if digest.hexdigest() != record["sha256"]:
                raise PermissionError(f"TAR member digest differs for {member.name}")


def _validate_source_bytes() -> dict[str, Any]:
    manifest = _read_json(DEFAULT_MANIFEST)
    records = [
        manifest["archive"],
        manifest["filelist"],
        manifest["official_geo_metadata"],
        *manifest["h5_files"],
    ]
    for record in records:
        path = ROOT / record["path"]
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise PermissionError(f"source bytes differ for {path.name}")
    _validate_archive_members(manifest)
    return manifest


def _decode(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def _feature_schema(path: Path) -> dict[str, list[str]]:
    with h5py.File(path, "r") as handle:
        return {
            key: _decode(handle[f"/matrix/features/{key}"][:])
            for key in ("id", "name", "feature_type")
        }


def _feature_schema_sha256(schema: dict[str, list[str]]) -> str:
    digest = hashlib.sha256()
    for row in zip(schema["id"], schema["name"], schema["feature_type"]):
        digest.update(("\t".join(row) + "\n").encode())
    return digest.hexdigest()


def _resolve_panel(schema: dict[str, list[str]]) -> dict[str, list[int]]:
    rna_rows = []
    adt_rows = []
    for identifier, symbol, antibody in zip(RNA_IDS, RNA_SYMBOLS, ADT_IDS):
        rna = [
            index
            for index, row in enumerate(
                zip(schema["id"], schema["name"], schema["feature_type"])
            )
            if row == (identifier, symbol, "Gene Expression")
        ]
        adt = [
            index
            for index, row in enumerate(zip(schema["id"], schema["feature_type"]))
            if row == (antibody, "Antibody Capture")
        ]
        if len(rna) != 1 or len(adt) != 1:
            raise PermissionError(
                f"panel mapping is not unique for {symbol}-{antibody}"
            )
        rna_rows.append(rna[0])
        adt_rows.append(adt[0])
    return {"rna": rna_rows, "adt": adt_rows}


def run_preflight(output_path: Path = DEFAULT_PREFLIGHT) -> dict[str, Any]:
    runtime_environment = _require_runtime_environment()
    manifest = _validate_source_bytes()
    designation = _designation()
    expected_schema = manifest["feature_schema_sha256"]
    resolved = None
    for sample in designation["samples"]:
        schema = _feature_schema(_source_path(sample["filename"]))
        if _feature_schema_sha256(schema) != expected_schema:
            raise PermissionError(f"feature schema differs for {sample['donor']}")
        current = _resolve_panel(schema)
        if resolved is None:
            resolved = current
        elif current != resolved:
            raise PermissionError("panel row indices differ across donors")
    access = _read_json(DEFAULT_ACCESS)
    if any(access["numeric_assay_values_accessed"].values()):
        raise PermissionError("pre-freeze access record reports numeric assay access")
    payload = {
        "schema": "gse288020-schema-preflight/1.0",
        "status": "PASS_BEFORE_BARCODE_OR_COUNT_ACCESS",
        "created_at_utc": _timestamp(),
        "accession": "GSE288020",
        "physical_donors": 23,
        "candidate_search_disclosure": CANDIDATE_SEARCH_DISCLOSURE,
        "runtime_environment": runtime_environment,
        "state_estimand": _state_estimand(),
        "split": {"calibration": 7, "pilot": 7, "held": 9},
        "feature_schema_sha256": expected_schema,
        "feature_counts": {"Gene Expression": 36601, "Antibody Capture": 55},
        "panel": {
            "cognate_mappings": len(MARKERS),
            "ordered_rna_adt_pairs": len(MARKERS) ** 2,
            "markers": list(MARKERS),
            "rna_rows_zero_based": resolved["rna"],
            "adt_rows_zero_based": resolved["adt"],
        },
        "access_audit": {
            "h5_feature_datasets_opened": [
                "/matrix/features/id",
                "/matrix/features/name",
                "/matrix/features/feature_type",
            ],
            "barcodes_dataset_opened": False,
            "matrix_data_dataset_opened": False,
            "matrix_indices_dataset_opened": False,
            "matrix_indptr_dataset_opened": False,
            "numeric_assay_values_accessed": 0,
        },
        "source_bindings": {
            "manifest_sha256": _sha256(DEFAULT_MANIFEST),
            "designation_sha256": _sha256(DEFAULT_DESIGNATION),
            "filelist_sha256": _sha256(DEFAULT_FILELIST),
            "official_geo_metadata_sha256": _sha256(DEFAULT_GEO_METADATA),
            "schema_access_sha256": _sha256(DEFAULT_ACCESS),
        },
    }
    _write_json_atomic_new(output_path, payload)
    return payload


def _sample_by_donor() -> dict[str, dict[str, Any]]:
    return {sample["donor"]: sample for sample in _designation()["samples"]}


def _read_donor_h5(
    donor: str, path: Path, *, return_adt: bool
) -> dict[str, Any]:
    schema = _feature_schema(path)
    rows = _resolve_panel(schema)
    feature_types = np.asarray(schema["feature_type"], dtype=object)
    gene_rows = np.flatnonzero(feature_types == "Gene Expression")
    mitochondrial_rows = np.asarray(
        [
            index
            for index, (name, kind) in enumerate(
                zip(schema["name"], schema["feature_type"])
            )
            if kind == "Gene Expression" and name.startswith("MT-")
        ],
        dtype=int,
    )
    with h5py.File(path, "r") as handle:
        barcodes = _decode(handle["/matrix/barcodes"][:])
        data = np.asarray(handle["/matrix/data"][:])
        indices = np.asarray(handle["/matrix/indices"][:], dtype=np.int64)
        indptr = np.asarray(handle["/matrix/indptr"][:], dtype=np.int64)
        shape = tuple(int(value) for value in handle["/matrix/shape"][:])
    if shape != (len(schema["name"]), len(barcodes)):
        raise PermissionError(f"matrix shape differs for {donor}")
    if len(indptr) != len(barcodes) + 1 or len(indices) != len(data):
        raise PermissionError(f"CSC structure differs for {donor}")
    if (
        indptr[0] != 0
        or indptr[-1] != len(data)
        or np.any(np.diff(indptr) < 0)
        or np.any(indices < 0)
        or np.any(indices >= shape[0])
    ):
        raise PermissionError(f"CSC pointers or row indices differ for {donor}")
    if len(set(barcodes)) != len(barcodes):
        raise PermissionError(f"barcodes are not unique for {donor}")

    matrix = sparse.csc_matrix((data, indices, indptr), shape=shape)
    gex = matrix[gene_rows]
    if np.any(gex.data < 0) or not np.issubdtype(gex.data.dtype, np.integer):
        raise PermissionError(f"RNA counts are not nonnegative integers for {donor}")
    detected = np.asarray(gex.getnnz(axis=0)).ravel()
    totals = np.asarray(gex.sum(axis=0)).ravel()
    mitochondrial = np.asarray(matrix[mitochondrial_rows].sum(axis=0)).ravel()
    fraction = np.divide(
        mitochondrial,
        totals,
        out=np.ones_like(mitochondrial, dtype=float),
        where=totals > 0,
    )
    eligible = (
        (detected >= MINIMUM_DETECTED_GENES)
        & (fraction <= MAXIMUM_MITOCHONDRIAL_FRACTION)
        & (totals <= MAXIMUM_RNA_UMIS)
    )
    eligible_indices = np.flatnonzero(eligible)
    ordered = sorted(
        eligible_indices,
        key=lambda index: (
            _salted_hash(CELL_SELECTION_SALT, donor, barcodes[index]),
            barcodes[index],
        ),
    )
    chosen = np.asarray(ordered[:CELL_BUDGET], dtype=int)
    selected_barcodes = [barcodes[index] for index in chosen]
    rna = np.asarray(matrix[rows["rna"]][:, chosen].toarray().T)
    positive = (rna > 0).sum(axis=0)
    informative_mask = (positive > 0) & (positive < CELL_BUDGET)
    informative_marker_count = int(np.count_nonzero(informative_mask))
    informative_pairs = informative_marker_count * len(MARKERS)
    output = {
        "donor": donor,
        "eligible": (
            len(chosen) == CELL_BUDGET
            and informative_pairs >= MINIMUM_INFORMATIVE_ENTITIES
        ),
        "filtered_barcodes": len(barcodes),
        "informative_marker_count": informative_marker_count,
        "informative_marker_identities": [
            marker for marker, retained in zip(MARKERS, informative_mask) if retained
        ],
        "informative_marker_support_mask": informative_mask.astype(bool).tolist(),
        "informative_ordered_pairs": informative_pairs,
        "qc_eligible_barcodes": len(eligible_indices),
        "selected_axis_sha256": _axis_sha256(selected_barcodes),
        "selected_cells": selected_barcodes,
        "rna": rna,
        "audit": {
            "combined_h5_opened": True,
            "barcodes_dataset_opened": True,
            "matrix_data_dataset_opened": True,
            "adt_count_elements_co_resident_in_decoded_csc": True,
            "adt_values_returned_to_caller": bool(return_adt),
            "adt_values_serialized": 0,
            "full_dense_feature_by_cell_matrix_materialized": False,
        },
    }
    if return_adt:
        adt = np.asarray(matrix[rows["adt"]][:, chosen].toarray().T)
        if np.any(adt < 0) or not np.issubdtype(adt.dtype, np.integer):
            raise PermissionError(f"ADT counts are not nonnegative integers for {donor}")
        output["adt"] = adt
    return output


def held_rna_reducer_payload(donor: str, path: Path) -> dict[str, Any]:
    value = _read_donor_h5(donor, path, return_adt=False)
    rna_state = (np.asarray(value["rna"]) > 0).astype(np.uint8)
    return {
        "schema": "gse288020-held-rna-reducer/1.0",
        "donor": donor,
        "eligible": value["eligible"],
        "filtered_barcodes": value["filtered_barcodes"],
        "informative_marker_count": value["informative_marker_count"],
        "informative_marker_identities": value["informative_marker_identities"],
        "informative_marker_support_mask": value[
            "informative_marker_support_mask"
        ],
        "informative_ordered_pairs": value["informative_ordered_pairs"],
        "qc_eligible_barcodes": value["qc_eligible_barcodes"],
        "selected_axis_sha256": value["selected_axis_sha256"],
        "rna_positive_counts": rna_state.sum(axis=0).astype(int).tolist(),
        "rna_state_sha256": _array_sha256(rna_state),
        "audit": {
            **value["audit"],
            "separate_bound_reducer_process_required": True,
            "barcodes_returned_to_parent": 0,
            "rna_cell_matrix_returned_to_parent": False,
            "adt_values_returned_to_parent": 0,
        },
    }


def _run_held_rna_reducer(donor: str, path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(DEFAULT_REDUCER),
            "--donor",
            donor,
            "--input",
            str(path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    forbidden = {"adt", "barcodes", "selected_cells", "rna"} & set(payload)
    if forbidden or payload.get("audit", {}).get("adt_values_returned_to_parent") != 0:
        raise PermissionError("held reducer returned a forbidden assay payload")
    return payload


def _support_gate(
    role: str, donors: tuple[str, ...], values: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    samples = _sample_by_donor()
    eligible = [donor for donor in donors if values[donor]["eligible"]]
    ages = {
        age: sum(samples[donor]["immune_age"] == age for donor in eligible)
        for age in ("Young", "Old")
    }
    checks = {
        "minimum_donor_support": len(eligible) >= ROLE_MINIMUM_DONORS[role],
        "minimum_support_in_each_immune_age_stratum": all(
            count >= ROLE_MINIMUM_PER_AGE[role] for count in ages.values()
        ),
        "every_retained_donor_has_192_informative_pairs": all(
            values[donor]["informative_ordered_pairs"]
            >= MINIMUM_INFORMATIVE_ENTITIES
            for donor in eligible
        ),
    }
    return {
        "role": role,
        "eligible_donors": eligible,
        "eligible_donor_count": len(eligible),
        "eligible_by_immune_age": ages,
        "donor_support": {
            donor: {
                "eligible": values[donor]["eligible"],
                "informative_marker_count": values[donor][
                    "informative_marker_count"
                ],
                "informative_marker_identities": values[donor][
                    "informative_marker_identities"
                ],
                "informative_marker_support_mask": values[donor][
                    "informative_marker_support_mask"
                ],
                "informative_ordered_pairs": values[donor]["informative_ordered_pairs"],
                "qc_eligible_barcodes": values[donor]["qc_eligible_barcodes"],
            }
            for donor in donors
        },
        "checks": checks,
        "passes": all(checks.values()),
    }


@contextmanager
def _configured_model() -> Iterator[None]:
    settings = {
        "MARKERS": MARKERS,
        "CELL_BUDGET": CELL_BUDGET,
        "MINIMUM_INFORMATIVE_ENTITIES": MINIMUM_INFORMATIVE_ENTITIES,
        "CELL_SELECTION_SALT": CELL_SELECTION_SALT,
        "ADT_TIE_SALT": ADT_TIE_SALT,
        "DESTROYED_LINK_SALT": DESTROYED_LINK_SALT,
        "NEIGHBOR_GRID": NEIGHBOR_GRID,
        "HETEROGENEITY_GRID": HETEROGENEITY_GRID,
        "RIDGE_GRID": RIDGE_GRID,
        "GRAPH_GRID": GRAPH_GRID,
        "TRANSPORT_GRID": TRANSPORT_GRID,
        "RESIDUAL_FAMILIES": RESIDUAL_FAMILIES,
        "BOOTSTRAPS": BOOTSTRAPS,
        "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
    }
    previous = {name: getattr(base, name) for name in settings}
    try:
        for name, value in settings.items():
            setattr(base, name, value)
        with base._configured_core():
            yield
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


def _records(
    donors: list[str], values: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    selected = {donor: values[donor]["selected_cells"] for donor in donors}
    counts = {
        donor: {"rna": values[donor]["rna"], "adt": values[donor]["adt"]}
        for donor in donors
    }
    with _configured_model():
        return base._records_from_counts(donors, selected, counts)


def _pooled_loglinear_interaction(tables: np.ndarray) -> np.ndarray:
    pooled = np.asarray(tables, dtype=float).sum(axis=0)
    if np.any(pooled <= 0.0):
        raise CouplingEstimationRefusal(
            "pooled saturated Poisson interaction has a structural zero"
        )
    return np.log(pooled[..., 0, 0] * pooled[..., 1, 1]) - np.log(
        pooled[..., 0, 1] * pooled[..., 1, 0]
    )


def _classical_log_odds(method: str, tables: np.ndarray) -> np.ndarray:
    if method == "common_effect_stratified_cmle":
        return np.asarray(model_core._fit_common_effect(tables)["population_log_odds"])
    if method == "pooled_saturated_poisson_interaction":
        return _pooled_loglinear_interaction(tables)
    raise ValueError(f"unknown classical method {method}")


def _select_classical(
    records: dict[str, dict[str, Any]], calibration: list[str], pilot: list[str]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    tables = np.asarray([records[donor]["tables"] for donor in calibration])
    selected: dict[str, Any] = {"methods": {}, "refusals": {}}
    losses: dict[str, np.ndarray] = {}
    with _configured_model():
        for method in CLASSICAL_METHODS:
            try:
                log_odds = _classical_log_odds(method, tables)
                candidates = {}
                for alpha in TRANSPORT_GRID:
                    values = np.empty(len(pilot))
                    for index, donor in enumerate(pilot):
                        truth = np.asarray(records[donor]["tables"])
                        rows, columns = model_core._margins(truth)
                        prediction = model_core._predict_log_odds(
                            log_odds, rows, columns, alpha
                        )
                        values[index] = model_core._donor_loss(truth, prediction)
                    candidates[alpha] = values
                alpha = min(
                    candidates,
                    key=lambda value: (float(candidates[value].mean()), value),
                )
                selected["methods"][method] = {
                    "transport_multiplier": alpha,
                    "candidate_mean_losses": {
                        str(value): float(candidate.mean())
                        for value, candidate in candidates.items()
                    },
                }
                losses[method] = candidates[alpha]
            except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
                selected["refusals"][method] = str(error)
    return selected, losses


def _refit_classical(
    records: dict[str, dict[str, Any]], donors: list[str], selected: dict[str, Any]
) -> dict[str, Any]:
    tables = np.asarray([records[donor]["tables"] for donor in donors])
    models = {}
    refusals = {}
    with _configured_model():
        for method, selection in selected["methods"].items():
            try:
                fit_certificate = None
                if method == "common_effect_stratified_cmle":
                    fitted = model_core._fit_common_effect(tables)
                    values = np.asarray(fitted.pop("population_log_odds"))
                    fit_certificate = fitted
                else:
                    values = _classical_log_odds(method, tables)
                models[method] = {
                    "population_log_odds": values.tolist(),
                    "transport_multiplier": selection["transport_multiplier"],
                }
                if fit_certificate is not None:
                    models[method]["fit_certificate"] = fit_certificate
                if method == "pooled_saturated_poisson_interaction":
                    models[method]["no_structural_zero"] = True
            except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
                refusals[method] = str(error)
    return {"models": models, "refusals": refusals}


def _exact_sign_test(differences: np.ndarray) -> dict[str, Any]:
    all_values = np.asarray(differences, dtype=float)
    nonzero = all_values
    nonzero = nonzero[nonzero != 0.0]
    favorable = int(np.count_nonzero(nonzero < 0.0))
    n = len(nonzero)
    tail = sum(math.comb(n, value) for value in range(favorable, n + 1)) / (2**n)
    return {
        "retained_donors": len(all_values),
        "nonzero_donors": n,
        "exact_ties": len(all_values) - n,
        "favorable_donors": favorable,
        "one_sided_p": tail,
    }


def _bootstrap_interval(differences: np.ndarray, seed_offset: int) -> list[float]:
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    draws = np.empty(BOOTSTRAPS)
    for start in range(0, BOOTSTRAPS, 1000):
        stop = min(start + 1000, BOOTSTRAPS)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        draws[start:stop] = values[indices].mean(axis=1)
    return np.quantile(draws, [0.025, 0.975], method="linear").tolist()


def _comparison(
    donors: list[str],
    primary: np.ndarray,
    comparator: np.ndarray,
    *,
    gating: bool,
    seed_offset: int,
    analysis_role: str,
) -> dict[str, Any]:
    if analysis_role not in {
        "post_selection_pilot_promotion_diagnostic",
        "held_confirmatory_inference",
    }:
        raise ValueError("unknown comparison analysis role")
    difference = np.asarray(primary) - np.asarray(comparator)
    samples = _sample_by_donor()
    ages = np.asarray([samples[donor]["immune_age"] for donor in donors])
    age_means = {
        age: float(difference[ages == age].mean()) for age in ("Young", "Old")
    }
    interval = _bootstrap_interval(difference, seed_offset)
    relative = 1.0 - float(np.mean(primary) / np.mean(comparator))
    favorable = int(np.count_nonzero(difference < 0.0))
    sign = _exact_sign_test(difference)
    if gating:
        checks = {
            "relative_deviance_reduction_at_least_five_percent": relative >= 0.05,
            "donor_bootstrap_upper_95_below_zero": interval[1] < 0.0,
            "favorable_donor_count_reached": favorable >= math.ceil(0.8 * len(donors)),
            "donor_sign_test_p_at_most_0_025": sign["one_sided_p"] <= 0.025,
            "each_immune_age_mean_difference_negative": all(
                value < 0.0 for value in age_means.values()
            ),
        }
    else:
        checks = {
            "lower_donor_equal_mean_deviance": float(difference.mean()) < 0.0,
            "donor_bootstrap_upper_95_below_zero": interval[1] < 0.0,
        }
    return {
        "analysis_role": analysis_role,
        "inference_claimed": analysis_role == "held_confirmatory_inference",
        "statistical_interpretation": (
            "held candidate-specific donor inference"
            if analysis_role == "held_confirmatory_inference"
            else "deterministic post-selection promotion diagnostic; not inference"
        ),
        "retained_donor_count": len(donors),
        "gating": gating,
        "primary_mean_loss": float(np.mean(primary)),
        "comparator_mean_loss": float(np.mean(comparator)),
        "relative_deviance_reduction": relative,
        "mean_paired_difference": float(difference.mean()),
        "donor_bootstrap_95_interval": interval,
        "bootstrap_draws": BOOTSTRAPS,
        "favorable_donors": favorable,
        "required_favorable_donors": math.ceil(0.8 * len(donors)),
        "donor_exact_sign_test": sign,
        "immune_age_mean_differences": age_means,
        "checks": checks,
        "passes": all(checks.values()),
        "donor_differences": {
            donor: float(value) for donor, value in zip(donors, difference)
        },
    }


def _gate(
    donors: list[str], losses: dict[str, np.ndarray], *, analysis_role: str
) -> dict[str, Any]:
    residual = _comparison(
        donors,
        losses["primary"],
        losses["best_residual"],
        gating=True,
        seed_offset=0,
        analysis_role=analysis_role,
    )
    destroyed = _comparison(
        donors,
        losses["primary"],
        losses["destroyed_link"],
        gating=True,
        seed_offset=1,
        analysis_role=analysis_role,
    )
    return {
        "primary_vs_selected_signed_residual": residual,
        "primary_vs_destroyed_link": destroyed,
        "passes": residual["passes"] and destroyed["passes"],
    }


def _classical_comparisons(
    donors: list[str],
    losses: dict[str, np.ndarray],
    refusals: dict[str, str] | None = None,
    *,
    analysis_role: str,
) -> dict[str, Any]:
    output = {}
    for index, method in enumerate(CLASSICAL_METHODS):
        if method in losses:
            output[method] = _comparison(
                donors,
                losses["primary"],
                losses[method],
                gating=False,
                seed_offset=10 + index,
                analysis_role=analysis_role,
            )
        else:
            output[method] = {
                "status": "REFUSED",
                "reason": (refusals or {}).get(method, "classical estimate unavailable"),
                "passes": False,
            }
    return output


def _load_development_values() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    samples = _sample_by_donor()
    values = {}
    for donor in CALIBRATION_DONORS + PILOT_DONORS:
        values[donor] = _read_donor_h5(
            donor, _source_path(samples[donor]["filename"]), return_adt=True
        )
    audit = {
        donor: {
            "filtered_barcodes": value["filtered_barcodes"],
            "informative_marker_count": value["informative_marker_count"],
            "informative_marker_identities": value["informative_marker_identities"],
            "informative_marker_support_mask": value[
                "informative_marker_support_mask"
            ],
            "informative_ordered_pairs": value["informative_ordered_pairs"],
            "qc_eligible_barcodes": value["qc_eligible_barcodes"],
            "selected_axis_sha256": value["selected_axis_sha256"],
            "matrix_access": value["audit"],
        }
        for donor, value in values.items()
    }
    return values, audit


def _loss_json(donors: list[str], losses: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        method: {donor: float(value) for donor, value in zip(donors, values)}
        for method, values in losses.items()
    }


def _finite_numeric_values(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_numeric_values(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_numeric_values(item) for item in value)
    if isinstance(value, (int, float, np.number)):
        return bool(np.isfinite(value))
    return True


def _source_only_external_study_bundle(
    records: dict[str, dict[str, Any]],
    values: dict[str, dict[str, Any]],
    calibration: list[str],
    pilot: list[str],
    selection: dict[str, Any],
    source_models: dict[str, Any],
    source_classical: dict[str, Any],
) -> dict[str, Any]:
    development = calibration + pilot
    declared_methods = (
        "primary",
        "matched_deviance_residual",
        "destroyed_link",
        "common_effect_cmle",
        "pooled_saturated_poisson",
    )
    deviance_candidates = [
        candidate
        for candidate in selection["residual_candidates"]
        if candidate["configuration"]["family"] == "deviance"
    ]
    if not deviance_candidates:
        raise CouplingEstimationRefusal("no deviance-residual pilot candidate exists")
    selected_deviance = min(
        deviance_candidates,
        key=lambda candidate: (
            candidate["mean_pilot_loss"],
            tuple(sorted(candidate["configuration"].items())),
        ),
    )
    tables = np.asarray([records[donor]["tables"] for donor in development])
    with _configured_model():
        deviance_coordinate = model_core._residual_pool(tables, "deviance")

    methods = {
        "primary": {**source_models["primary"], "status": "VALID"},
        "matched_deviance_residual": {
            "status": "VALID",
            "configuration": selected_deviance["configuration"],
            "pooled_coordinate": deviance_coordinate.tolist(),
        },
        "destroyed_link": {**source_models["destroyed_link"], "status": "VALID"},
    }
    refusals = {}
    classical_mapping = {
        "common_effect_cmle": "common_effect_stratified_cmle",
        "pooled_saturated_poisson": "pooled_saturated_poisson_interaction",
    }
    for exported, internal in classical_mapping.items():
        if internal in source_classical["models"]:
            methods[exported] = {
                **source_classical["models"][internal],
                "status": "VALID",
            }
        else:
            refusal = {
                "code": f"{exported.upper()}_REFUSED",
                "reason": source_classical["refusals"].get(
                    internal, "classical source model unavailable"
                ),
            }
            methods[exported] = {"status": "REFUSED", "refusal": refusal}
            refusals[exported] = refusal

    marker_support = np.asarray(
        [
            sum(
                bool(values[donor]["informative_marker_support_mask"][index])
                for donor in development
            )
            for index in range(len(MARKERS))
        ],
        dtype=np.int64,
    )
    pair_support = np.repeat(marker_support, len(MARKERS))
    retained_pair_mask = pair_support >= 2
    coordinate_keys = {
        "primary": "population_log_odds",
        "matched_deviance_residual": "pooled_coordinate",
        "destroyed_link": "population_log_odds",
        "common_effect_cmle": "population_log_odds",
        "pooled_saturated_poisson": "population_log_odds",
    }
    finite_coordinates = {
        method: (
            method in methods
            and methods[method].get("status") == "VALID"
            and coordinate_key in methods[method]
            and _finite_numeric_values(methods[method].get(coordinate_key))
        )
        for method, coordinate_key in coordinate_keys.items()
    }
    core_checks = {
        "exactly_14_retained_mgus_donors": (
            len(development) == 14
            and set(development) == set(CALIBRATION_DONORS + PILOT_DONORS)
        ),
        "exactly_7_retained_calibration_and_7_retained_pilot": (
            len(calibration) == 7 and len(pilot) == 7
        ),
        "all_256_pairs_have_at_least_2_informative_source_donors": bool(
            np.all(retained_pair_mask)
        ),
        "primary_residual_destroyed_coordinates_are_finite": all(
            finite_coordinates[method]
            for method in (
                "primary",
                "matched_deviance_residual",
                "destroyed_link",
            )
        ),
        "primary_optimizer_certificate_pass": (
            "fit_certificate" in methods["primary"]
            and _finite_numeric_values(methods["primary"]["fit_certificate"])
        ),
        "destroyed_optimizer_certificate_pass": (
            "fit_certificate" in methods["destroyed_link"]
            and _finite_numeric_values(methods["destroyed_link"]["fit_certificate"])
        ),
        "no_mm_or_gse309593_values_used": True,
    }
    classical_checks = {
        "both_classical_coordinates_are_finite": all(
            finite_coordinates[method]
            for method in ("common_effect_cmle", "pooled_saturated_poisson")
        ),
        "common_cmle_gradient_and_condition_certificate_pass": (
            "common_effect_cmle" in methods
            and "fit_certificate" in methods["common_effect_cmle"]
            and _finite_numeric_values(
                methods["common_effect_cmle"]["fit_certificate"]
            )
        ),
        "pooled_poisson_no_structural_zero_pass": (
            methods.get("pooled_saturated_poisson", {}).get(
                "no_structural_zero"
            )
            is True
        ),
    }
    core_passes = all(core_checks.values())
    classical_head_to_head_ready = all(classical_checks.values())
    external_study_ready = core_passes and classical_head_to_head_ready
    return {
        "schema": "gse288020-independent-study-source-model/1.0",
        "scope": (
            "fixed GSE288020 MGUS source model for a separately frozen external "
            "study; not a GSE288020 held-MM result"
        ),
        "public_export_location": (
            "results/development/gse288020_development_v1.json#/"
            "source_only_external_study_model"
        ),
        "source_accession": "GSE288020",
        "source_condition": "MGUS",
        "selection_provenance": {
            "split_salt": SPLIT_SALT,
            "designated_calibration_donors": list(CALIBRATION_DONORS),
            "designated_pilot_donors": list(PILOT_DONORS),
            "retained_calibration_donors": calibration,
            "retained_pilot_donors": pilot,
            "primary_configuration": selection["primary"],
            "deviance_residual_configuration": selected_deviance["configuration"],
            "gse288020_mm_values_used": False,
            "gse309593_values_used": False,
        },
        "refit_axis": {
            "designated_mgus_donors": list(CALIBRATION_DONORS + PILOT_DONORS),
            "retained_mgus_donors": development,
            "retained_mgus_donor_count": len(development),
        },
        "external_study_eligibility": {
            "internal_gse288020_refit_allows_12_to_14_retained_mgus_donors": True,
            "external_study_requires_all_14_designated_mgus_donors": True,
            "external_study_requires_7_calibration_and_7_pilot_donors": True,
            "external_study_requires_all_five_methods_valid": True,
            "target_assay_access_requires_external_study_ready": True,
        },
        "declared_method_order": list(declared_methods),
        "methods": methods,
        "method_refusals": refusals,
        "source_support": {
            "marker_order": list(MARKERS),
            "ordered_pair_axis": "RNA-major, then ADT marker order",
            "informative_source_donors_per_marker": marker_support.tolist(),
            "informative_source_donors_per_ordered_pair": pair_support.tolist(),
            "informative_source_donors_per_ordered_pair_sha256": _array_sha256(
                pair_support
            ),
            "retained_ordered_pair_support_mask": retained_pair_mask.tolist(),
            "retained_ordered_pair_count": int(np.count_nonzero(retained_pair_mask)),
        },
        "numerical_certificate": {
            "finite_coordinate_checks": finite_coordinates,
            "core_checks": core_checks,
            "classical_checks": classical_checks,
            "core_passes": core_passes,
            "classical_head_to_head_ready": classical_head_to_head_ready,
            "external_study_ready": external_study_ready,
            "passes": external_study_ready,
        },
    }


def _development_body(protocol_commit: str) -> dict[str, Any]:
    preflight = _read_json(DEFAULT_PREFLIGHT)
    if preflight.get("status") != "PASS_BEFORE_BARCODE_OR_COUNT_ACCESS":
        raise PermissionError("frozen schema preflight did not pass")
    _validate_source_bytes()
    values, access = _load_development_values()
    support = {
        "calibration": _support_gate("calibration", CALIBRATION_DONORS, values),
        "pilot": _support_gate("pilot", PILOT_DONORS, values),
    }
    if not all(value["passes"] for value in support.values()):
        return {
            "schema": "gse288020-development/1.0",
            "status": "TERMINAL_DEVELOPMENT_QC_OR_SUPPORT_FAILURE",
            "created_at_utc": _timestamp(),
            "protocol_commit": protocol_commit,
            "support": support,
            "access": access,
            "terminal_rule": "No held HDF5 barcode or matrix dataset may be opened.",
        }

    calibration = support["calibration"]["eligible_donors"]
    pilot = support["pilot"]["eligible_donors"]
    records = _records(calibration + pilot, values)
    with _configured_model():
        selection, _, losses = base._select_on_pilot(records, calibration, pilot)
    classical_selection, classical_losses = _select_classical(
        records, calibration, pilot
    )
    losses.update(classical_losses)
    gate = _gate(
        pilot,
        losses,
        analysis_role="post_selection_pilot_promotion_diagnostic",
    )
    development = calibration + pilot
    with _configured_model():
        source_models = model_core._fit_models(records, development, selection)
    source_classical = _refit_classical(
        records, development, classical_selection
    )
    source_bundle = _source_only_external_study_bundle(
        records,
        values,
        calibration,
        pilot,
        selection,
        source_models,
        source_classical,
    )
    payload: dict[str, Any] = {
        "schema": "gse288020-development/1.0",
        "status": (
            "PILOT_PROMOTION_PASS"
            if gate["passes"]
            else "TERMINAL_PILOT_PROMOTION_FAILURE"
        ),
        "created_at_utc": _timestamp(),
        "protocol_commit": protocol_commit,
        "calibration_donors": calibration,
        "pilot_donors": pilot,
        "retained_development_donors": development,
        "retained_development_donor_count": len(development),
        "support": support,
        "access": access,
        "panel": {
            "cognate_mappings": len(MARKERS),
            "ordered_rna_adt_pairs": len(MARKERS) ** 2,
            "markers": list(MARKERS),
        },
        "selection": selection,
        "pilot_losses": _loss_json(pilot, losses),
        "pilot_promotion_diagnostic": gate,
        "pilot_statistical_scope": (
            "Deterministic post-selection promotion diagnostics only; no "
            "confidence-interval or hypothesis-test interpretation is claimed."
        ),
        "classical_selection": classical_selection,
        "classical_head_to_head": _classical_comparisons(
            pilot,
            losses,
            classical_selection["refusals"],
            analysis_role="post_selection_pilot_promotion_diagnostic",
        ),
        "source_only_external_study_model": source_bundle,
        "gse288020_held_prediction_models": {
            "primary_residual_destroyed_models": source_models,
            "classical_models": source_classical,
        },
    }
    if not gate["passes"]:
        payload["terminal_rule"] = (
            "The source-only external-study model remains frozen, but no GSE288020 "
            "held HDF5 barcode or matrix dataset may be opened after this promotion "
            "diagnostic failure."
        )
    return payload


def run_development(output_path: Path = DEFAULT_DEVELOPMENT) -> dict[str, Any]:
    _ensure_stage_available(
        output_path, DEFAULT_DEVELOPMENT, DEFAULT_DEVELOPMENT_ATTEMPT
    )
    runtime_environment = _require_runtime_environment()
    protocol_commit = _require_public_tag(PROTOCOL_TAG, PROTOCOL_BINDINGS)
    return _run_claimed_stage(
        "development",
        output_path,
        DEFAULT_DEVELOPMENT,
        DEFAULT_DEVELOPMENT_ATTEMPT,
        protocol_commit,
        runtime_environment,
        lambda: _development_body(protocol_commit),
    )


def _margins_from_positive_counts(values: list[int]) -> tuple[np.ndarray, np.ndarray]:
    positives = np.asarray(values, dtype=int)
    rows = np.repeat(
        np.stack([CELL_BUDGET - positives, positives], axis=1)[:, None, :],
        len(MARKERS),
        axis=1,
    )
    columns = np.broadcast_to(
        np.asarray([CELL_BUDGET // 2, CELL_BUDGET // 2]),
        (len(MARKERS), len(MARKERS), 2),
    ).copy()
    return rows, columns


def _predict_methods(
    models: dict[str, Any],
    classical: dict[str, Any],
    rows: np.ndarray,
    columns: np.ndarray,
) -> dict[str, np.ndarray]:
    with _configured_model():
        output = model_core._predict_models(models, rows, columns)
        for method, value in classical.get("models", {}).items():
            output[method] = model_core._predict_log_odds(
                np.asarray(value["population_log_odds"]),
                rows,
                columns,
                value["transport_multiplier"],
            )
    return output


def _prediction_body(
    development_commit: str, development: dict[str, Any]
) -> dict[str, Any]:
    if development.get("status") != "PILOT_PROMOTION_PASS" or not development.get(
        "pilot_promotion_diagnostic", {}
    ).get("passes"):
        raise PermissionError("the frozen pilot promotion diagnostic did not pass")
    _validate_source_bytes()
    samples = _sample_by_donor()
    reduced = {
        donor: _run_held_rna_reducer(
            donor, _source_path(samples[donor]["filename"])
        )
        for donor in HELD_DONORS
    }
    support = _support_gate("held", HELD_DONORS, reduced)
    if not support["passes"]:
        return {
            "schema": "gse288020-held-predictions/1.0",
            "status": "TERMINAL_HELD_RNA_QC_OR_SUPPORT_FAILURE",
            "created_at_utc": _timestamp(),
            "development_commit": development_commit,
            "support": support,
            "reducer_audit": {donor: value["audit"] for donor, value in reduced.items()},
            "terminal_rule": "Held ADT scoring may not run.",
        }

    predictions = []
    for donor in support["eligible_donors"]:
        value = reduced[donor]
        rows, columns = _margins_from_positive_counts(value["rna_positive_counts"])
        source_model = development["gse288020_held_prediction_models"]
        estimates = _predict_methods(
            source_model["primary_residual_destroyed_models"],
            source_model["classical_models"],
            rows,
            columns,
        )
        predictions.append(
            {
                "donor": donor,
                "immune_age": samples[donor]["immune_age"],
                "selected_axis_sha256": value["selected_axis_sha256"],
                "held_rna_state_sha256": value["rna_state_sha256"],
                "row_margins": rows.tolist(),
                "column_margins": columns.tolist(),
                "predicted_tables": {
                    method: estimate.tolist() for method, estimate in estimates.items()
                },
                "prediction_sha256": {
                    method: _array_sha256(estimate)
                    for method, estimate in estimates.items()
                },
            }
        )
    classical_refusals = {
        **development.get("classical_selection", {}).get("refusals", {}),
        **development.get("gse288020_held_prediction_models", {})
        .get("classical_models", {})
        .get("refusals", {}),
    }
    payload = {
        "schema": "gse288020-held-predictions/1.0",
        "status": "HELD_PREDICTIONS_FROZEN_AFTER_BOUND_RNA_SUMMARY_FIREWALL",
        "created_at_utc": _timestamp(),
        "development_commit": development_commit,
        "development_sha256": _sha256(DEFAULT_DEVELOPMENT),
        "held_donors": support["eligible_donors"],
        "held_support": support,
        "combined_h5_boundary": (
            "The bound child reducer decoded combined CSC chunks containing RNA and "
            "ADT values; it returned only RNA margins and hashes."
        ),
        "held_adt_values_returned_to_parent": 0,
        "held_adt_values_serialized": 0,
        "classical_refusals": classical_refusals,
        "reducer_audit": {donor: value["audit"] for donor, value in reduced.items()},
        "predictions": predictions,
    }
    return payload


def run_prediction(output_path: Path = DEFAULT_PREDICTION) -> dict[str, Any]:
    _ensure_stage_available(
        output_path, DEFAULT_PREDICTION, DEFAULT_PREDICTION_ATTEMPT
    )
    runtime_environment = _require_runtime_environment()
    protocol_commit = _require_public_tag(PROTOCOL_TAG, PROTOCOL_BINDINGS)
    development_commit = _require_public_tag(
        DEVELOPMENT_TAG,
        (
            *PROTOCOL_BINDINGS,
            _relative(DEFAULT_DEVELOPMENT_ATTEMPT),
            _relative(DEFAULT_DEVELOPMENT),
        ),
    )
    development = _validate_prior_stage(
        "development",
        DEFAULT_DEVELOPMENT_ATTEMPT,
        DEFAULT_DEVELOPMENT,
        protocol_commit,
        runtime_environment,
        "PILOT_PROMOTION_PASS",
    )
    if not development.get("pilot_promotion_diagnostic", {}).get("passes"):
        raise PermissionError("development diagnostic does not authorize held access")
    return _run_claimed_stage(
        "prediction",
        output_path,
        DEFAULT_PREDICTION,
        DEFAULT_PREDICTION_ATTEMPT,
        development_commit,
        runtime_environment,
        lambda: _prediction_body(development_commit, development),
    )


def _score_body(
    prediction_commit: str, prediction: dict[str, Any]
) -> dict[str, Any]:
    if (
        prediction.get("status")
        != "HELD_PREDICTIONS_FROZEN_AFTER_BOUND_RNA_SUMMARY_FIREWALL"
        or prediction.get("held_adt_values_returned_to_parent") != 0
        or prediction.get("held_adt_values_serialized") != 0
    ):
        raise PermissionError("held predictions are not a valid frozen boundary")
    _validate_source_bytes()
    samples = _sample_by_donor()
    held = prediction["held_donors"]
    frozen = {value["donor"]: value for value in prediction["predictions"]}
    values = {
        donor: _read_donor_h5(
            donor, _source_path(samples[donor]["filename"]), return_adt=True
        )
        for donor in held
    }
    for donor in held:
        value = values[donor]
        rna_state = (np.asarray(value["rna"]) > 0).astype(np.uint8)
        rows, columns = _margins_from_positive_counts(
            rna_state.sum(axis=0).astype(int).tolist()
        )
        if (
            value["selected_axis_sha256"] != frozen[donor]["selected_axis_sha256"]
            or _array_sha256(rna_state) != frozen[donor]["held_rna_state_sha256"]
            or rows.tolist() != frozen[donor]["row_margins"]
            or columns.tolist() != frozen[donor]["column_margins"]
        ):
            raise PermissionError(f"held RNA boundary changed for {donor}")
    records = _records(held, values)
    methods = sorted(frozen[held[0]]["predicted_tables"])
    losses = {method: np.empty(len(held)) for method in methods}
    truth_hashes = {}
    with _configured_model():
        for donor_index, donor in enumerate(held):
            truth = np.asarray(records[donor]["tables"])
            truth_hashes[donor] = records[donor]["table_sha256"]
            for method in methods:
                estimate = np.asarray(frozen[donor]["predicted_tables"][method])
                if _array_sha256(estimate) != frozen[donor]["prediction_sha256"][method]:
                    raise PermissionError(f"held prediction changed for {donor}")
                losses[method][donor_index] = model_core._donor_loss(truth, estimate)
    gate = _gate(held, losses, analysis_role="held_confirmatory_inference")
    classical = _classical_comparisons(
        held,
        losses,
        prediction.get("classical_refusals", {}),
        analysis_role="held_confirmatory_inference",
    )
    transfer_pass = bool(gate["passes"])
    classical_gain = all(value["passes"] for value in classical.values())
    if transfer_pass and classical_gain:
        status = "TRANSFER_PASS_WITH_GAIN_OVER_BOTH_CLASSICAL_INTERACTIONS"
    elif transfer_pass:
        status = "TRANSFER_PASS_WITHOUT_GAIN_OVER_BOTH_CLASSICAL_INTERACTIONS"
    else:
        status = "TRANSFER_CONFIRMATION_FAIL"
    payload = {
        "schema": "gse288020-confirmation/1.0",
        "status": status,
        "created_at_utc": _timestamp(),
        "prediction_tag": PREDICTION_TAG,
        "prediction_commit": prediction_commit,
        "prediction_sha256": _sha256(DEFAULT_PREDICTION),
        "held_donors": held,
        "held_inference_scope": (
            "candidate-specific donor inference without familywise adjustment "
            "across the preceding public candidate search"
        ),
        "losses": _loss_json(held, losses),
        "transfer_confirmation_pass": transfer_pass,
        "primary_gate": gate,
        "classical_head_to_head": classical,
        "classical_refusals": prediction.get("classical_refusals", {}),
        "gain_over_both_classical_interactions": classical_gain,
        "truth_table_sha256": truth_hashes,
        "score_access": {donor: values[donor]["audit"] for donor in held},
    }
    return payload


def run_score(output_path: Path = DEFAULT_SCORE) -> dict[str, Any]:
    _ensure_stage_available(output_path, DEFAULT_SCORE, DEFAULT_SCORE_ATTEMPT)
    runtime_environment = _require_runtime_environment()
    protocol_commit = _require_public_tag(PROTOCOL_TAG, PROTOCOL_BINDINGS)
    development_commit = _require_public_tag(
        DEVELOPMENT_TAG,
        (
            *PROTOCOL_BINDINGS,
            _relative(DEFAULT_DEVELOPMENT_ATTEMPT),
            _relative(DEFAULT_DEVELOPMENT),
        ),
    )
    _validate_prior_stage(
        "development",
        DEFAULT_DEVELOPMENT_ATTEMPT,
        DEFAULT_DEVELOPMENT,
        protocol_commit,
        runtime_environment,
        "PILOT_PROMOTION_PASS",
    )
    prediction_commit = _require_public_tag(
        PREDICTION_TAG,
        (
            *PROTOCOL_BINDINGS,
            _relative(DEFAULT_DEVELOPMENT_ATTEMPT),
            _relative(DEFAULT_DEVELOPMENT),
            _relative(DEFAULT_PREDICTION_ATTEMPT),
            _relative(DEFAULT_PREDICTION),
        ),
    )
    prediction = _validate_prior_stage(
        "prediction",
        DEFAULT_PREDICTION_ATTEMPT,
        DEFAULT_PREDICTION,
        development_commit,
        runtime_environment,
        "HELD_PREDICTIONS_FROZEN_AFTER_BOUND_RNA_SUMMARY_FIREWALL",
    )
    if (
        prediction.get("held_adt_values_returned_to_parent") != 0
        or prediction.get("held_adt_values_serialized") != 0
    ):
        raise PermissionError("prediction boundary does not authorize held scoring")
    return _run_claimed_stage(
        "score",
        output_path,
        DEFAULT_SCORE,
        DEFAULT_SCORE_ATTEMPT,
        prediction_commit,
        runtime_environment,
        lambda: _score_body(prediction_commit, prediction),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, default in (
        ("preflight", DEFAULT_PREFLIGHT),
        ("develop", DEFAULT_DEVELOPMENT),
        ("predict", DEFAULT_PREDICTION),
        ("score", DEFAULT_SCORE),
    ):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--output", type=Path, default=default)
    args = parser.parse_args()
    if args.command == "preflight":
        payload = run_preflight(args.output)
    elif args.command == "develop":
        payload = run_development(args.output)
    elif args.command == "predict":
        payload = run_prediction(args.output)
    else:
        payload = run_score(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
