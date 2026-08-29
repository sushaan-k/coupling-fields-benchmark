"""Staged GSE189050 held-pool linked RNA/ADT confirmation.

``preflight`` reads committed metadata, archive headers, and feature tables only.
``develop`` is the first command allowed to read calibration and pilot matrices.
``predict`` reads held HTO and RNA values but never converts held biological ADT
values. ``score`` is the one-shot held biological-ADT evaluation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile
from typing import Any, BinaryIO, Iterable, Iterator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import confirm_gse158769_citeseq as base  # noqa: E402
from experiments import confirm_gse314416_citeseq as model_core  # noqa: E402
from mapreg.heterogeneity_adaptive_coupling import (  # noqa: E402
    CouplingEstimationRefusal,
)


ROOT = REPO_ROOT
DATA_DIR = ROOT / "data/confirmation/gse189050_citeseq"
SOURCE_CACHE = DATA_DIR / "source_cache"
DEFAULT_GEO_METADATA = DATA_DIR / "metadata/GSE189050_sample.csv.gz"
DEFAULT_AUTHOR_SHEET = DATA_DIR / "metadata/da_samplesheet_final.csv"
DEFAULT_MANIFEST = DATA_DIR / "source_manifest_v1.json"
DEFAULT_DESIGNATION = DATA_DIR / "candidate_designation_v1.json"
DEFAULT_ACCESS = ROOT / "results/development/gse189050_source_schema_access_v1.json"
DEFAULT_PREFLIGHT = ROOT / "results/development/gse189050_schema_preflight_v1.json"
DEFAULT_DEVELOPMENT = ROOT / "results/development/gse189050_development_v1.json"
DEFAULT_PREDICTION = ROOT / "results/gse189050_held_predictions_v1.json"
DEFAULT_SCORE = ROOT / "results/gse189050_confirmation_v1.json"
DEFAULT_PROTOCOL = (
    ROOT / "docs/GSE189050_SLE_CITESEQ_HELD_POOL_CONFIRMATION_PROTOCOL_2026-08-28.md"
)

PUBLIC_ORIGIN = "https://github.com/sushaan-k/coupling-fields-benchmark.git"
PROTOCOL_TAG = "gse189050-citeseq-v1-protocol"
DEVELOPMENT_TAG = "gse189050-citeseq-v1-development"
PREDICTION_TAG = "gse189050-citeseq-v1-predictions"

RUNS = ("s1a", "s1b", "s2a", "s2b", "s3a", "s3b", "s4a", "s4b", "s5a", "s5b")
CALIBRATION_POOLS = ("s1a", "s3a")
PILOT_POOLS = ("s2a", "s4a")
HELD_POOLS = ("s1b", "s2b", "s3b", "s4b", "s5a", "s5b")
ROLE_POOLS = {
    "calibration": CALIBRATION_POOLS,
    "pilot": PILOT_POOLS,
    "held": HELD_POOLS,
}

PRIMARY_MARKERS = (
    "CD1c",
    "CD11b",
    "CD11c",
    "CD14",
    "CD19",
    "CD27",
    "CD38",
    "CD58",
    "CD64",
    "CD85j",
    "CD86",
    "CD305",
)
PRIMARY_RNA_IDS = (
    "ENSG00000158481",
    "ENSG00000169896",
    "ENSG00000140678",
    "ENSG00000170458",
    "ENSG00000177455",
    "ENSG00000139193",
    "ENSG00000004468",
    "ENSG00000116815",
    "ENSG00000150337",
    "ENSG00000104972",
    "ENSG00000114013",
    "ENSG00000167613",
)
PRIMARY_RNA_SYMBOLS = (
    "CD1C",
    "ITGAM",
    "ITGAX",
    "CD14",
    "CD19",
    "CD27",
    "CD38",
    "CD58",
    "FCGR1A",
    "LILRB1",
    "CD86",
    "LAIR1",
)
PRIMARY_ADT = PRIMARY_MARKERS

LEGACY_MARKERS = (
    "CD14",
    "CD16",
    "CD11c",
    "CD19",
    "CD27",
    "CD38",
    "HLA-DR",
    "CD95",
    "CD305",
)
LEGACY_RNA_IDS = (
    "ENSG00000170458",
    "ENSG00000203747",
    "ENSG00000140678",
    "ENSG00000177455",
    "ENSG00000139193",
    "ENSG00000004468",
    "ENSG00000204287",
    "ENSG00000026103",
    "ENSG00000167613",
)
LEGACY_RNA_SYMBOLS = (
    "CD14",
    "FCGR3A",
    "ITGAX",
    "CD19",
    "CD27",
    "CD38",
    "HLA-DRA",
    "FAS",
    "LAIR1",
)
LEGACY_ADT = LEGACY_MARKERS

PANELS = {
    "primary": {
        "markers": PRIMARY_MARKERS,
        "rna_ids": PRIMARY_RNA_IDS,
        "rna_symbols": PRIMARY_RNA_SYMBOLS,
        "adt": PRIMARY_ADT,
        "minimum_informative": 108,
    },
    "legacy_secondary": {
        "markers": LEGACY_MARKERS,
        "rna_ids": LEGACY_RNA_IDS,
        "rna_symbols": LEGACY_RNA_SYMBOLS,
        "adt": LEGACY_ADT,
        "minimum_informative": 64,
    },
}

CELL_BUDGET = 512
CELL_SELECTION_SALT = "GSE189050-CELL-BUDGET-v1"
ADT_TIE_SALT = "GSE189050-ADT-MIDRANK-v1"
DESTROYED_LINK_SALT = "GSE189050-DESTROYED-LINK-v1"

HTO_MINIMUM_TOTAL = 20
HTO_MINIMUM_RATIO = 5.0
HTO_MINIMUM_TOP_FRACTION = 0.70
HUMAN_GEX_FRACTION_MINIMUM = 0.90

NEIGHBOR_GRID = (1, 2)
HETEROGENEITY_GRID = (0.1, 1.0, 10.0)
RIDGE_GRID = (0.01, 0.1)
GRAPH_GRID = (0.0, 0.1, 1.0)
TRANSPORT_GRID = (0.5, 0.75, 1.0, 1.25)
RESIDUAL_FAMILIES = ("pearson", "deviance")
BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 20260828

PROTOCOL_BINDINGS = (
    ".gitattributes",
    "experiments/confirm_gse189050_citeseq.py",
    "tests/test_gse189050_citeseq_confirmation.py",
    "docs/GSE189050_SLE_CITESEQ_HELD_POOL_CONFIRMATION_PROTOCOL_2026-08-28.md",
    "data/confirmation/gse189050_citeseq/source_manifest_v1.json",
    "data/confirmation/gse189050_citeseq/candidate_designation_v1.json",
    "data/confirmation/gse189050_citeseq/metadata/GSE189050_sample.csv.gz",
    "data/confirmation/gse189050_citeseq/metadata/da_samplesheet_final.csv",
    "results/development/gse189050_source_schema_access_v1.json",
    "results/development/gse189050_schema_preflight_v1.json",
    "experiments/confirm_gse158769_citeseq.py",
    "experiments/confirm_gse314416_citeseq.py",
    "mapreg/heterogeneity_adaptive_coupling.py",
    "mapreg/hierarchical_conditional_coupling.py",
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


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


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
    if not remote:
        remote = subprocess.run(
            ["git", "ls-remote", PUBLIC_ORIGIN, f"refs/tags/{tag}"],
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


def _archive_path(run: str) -> Path:
    record = _read_json(DEFAULT_MANIFEST)["archives"][run]
    return SOURCE_CACHE / f"{record['gsm']}_{run}_filtered_cellranger.tar.gz"


def _validate_source_bytes(runs: Iterable[str] = RUNS) -> None:
    archives = _read_json(DEFAULT_MANIFEST)["archives"]
    for run in runs:
        path = _archive_path(run)
        record = archives[run]
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise PermissionError(f"{run} archive differs from the frozen source")


def _canonical_adt(value: str) -> str:
    normalized = "".join(
        character for character in value.upper() if character.isalnum()
    )
    for suffix in (
        "ANTIBODYCAPTURE",
        "TOTALSEQA",
        "TOTALSEQB",
        "TOTALSEQC",
        "PROTEIN",
        "ADT",
    ):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def _parse_feature_bytes(value: bytes) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for line in value.decode().splitlines():
        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError("10x feature table does not have three columns")
        rows.append((fields[0], fields[1], fields[2]))
    if len(rows) != len(
        set((identifier, name, kind) for identifier, name, kind in rows)
    ):
        raise ValueError("10x feature table contains duplicate complete rows")
    return rows


def _schema_from_archive(run: str, path: Path) -> dict[str, Any]:
    record = _read_json(DEFAULT_MANIFEST)["archives"][run]
    expected_members = record["members"]
    observed: list[dict[str, Any]] = []
    feature_bytes: bytes | None = None
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            observed.append({"name": member.name, "bytes": member.size})
            if member.name == expected_members["features"]["name"]:
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"{run} feature member is not readable")
                feature_bytes = stream.read()
            if member.name == expected_members["matrix"]["name"]:
                break
    expected = [expected_members[key] for key in ("barcodes", "features", "matrix")]
    if observed != expected:
        raise PermissionError(f"{run} tar member schema differs")
    if (
        feature_bytes is None
        or _bytes_sha256(feature_bytes) != record["feature_table_sha256"]
    ):
        raise PermissionError(f"{run} feature table differs")
    rows = _parse_feature_bytes(feature_bytes)
    resolution = _resolve_features(rows, run)
    return {
        "archive_bytes": path.stat().st_size,
        "archive_sha256": _sha256(path),
        "feature_table_rows": len(rows),
        "feature_table_sha256": _bytes_sha256(feature_bytes),
        "gene_expression_features": sum(row[2] == "Gene Expression" for row in rows),
        "antibody_capture_features": sum(row[2] == "Antibody Capture" for row in rows),
        "members": observed,
        "primary_cognate_feature_mappings": resolution["primary_resolution"],
        "legacy_cognate_feature_mappings": resolution["legacy_resolution"],
        "hto_resolution": resolution["hto_resolution"],
    }


def _resolve_features(
    rows: list[tuple[str, str, str]], run: str, hto_tags: Iterable[str] | None = None
) -> dict[str, Any]:
    gex: dict[tuple[str, str], list[int]] = {}
    adt: dict[str, list[int]] = {}
    human = np.zeros(len(rows), dtype=bool)
    mouse = np.zeros(len(rows), dtype=bool)
    for index, (identifier, name, kind) in enumerate(rows):
        if kind == "Gene Expression":
            if identifier.startswith("GRCh38_") and name.startswith("GRCh38_"):
                human[index] = True
                key = (identifier.removeprefix("GRCh38_"), name.removeprefix("GRCh38_"))
                gex.setdefault(key, []).append(index)
            elif identifier.startswith("mm10_") and name.startswith("mm10_"):
                mouse[index] = True
        elif kind == "Antibody Capture":
            for label in {identifier, name}:
                adt.setdefault(_canonical_adt(label), []).append(index)

    panel_rows: dict[str, dict[str, tuple[int, ...]]] = {}
    panel_records: dict[str, list[dict[str, Any]]] = {}
    for panel_name, panel in PANELS.items():
        rna_rows: list[int] = []
        adt_rows: list[int] = []
        records = []
        for identifier, symbol, antibody in zip(
            panel["rna_ids"], panel["rna_symbols"], panel["adt"]
        ):
            rna_matches = gex.get((identifier, symbol), [])
            adt_matches = sorted(set(adt.get(_canonical_adt(antibody), [])))
            if len(rna_matches) != 1 or len(adt_matches) != 1:
                raise PermissionError(
                    f"{run} does not uniquely resolve {identifier}/{symbol}/{antibody}"
                )
            rna_rows.append(rna_matches[0])
            adt_rows.append(adt_matches[0])
            records.append(
                {
                    "rna_ensembl": identifier,
                    "rna_symbol": symbol,
                    "adt": antibody,
                    "rna_row_1based": rna_matches[0] + 1,
                    "adt_row_1based": adt_matches[0] + 1,
                }
            )
        panel_rows[panel_name] = {
            "rna": tuple(rna_rows),
            "adt": tuple(adt_rows),
        }
        panel_records[panel_name] = records

    if hto_tags is None:
        hto_tags = (
            ("HT-1", "HT-2", "HT-3", "HT-4")
            if run in RUNS[:4]
            else (
                "HT-1",
                "HT-2",
                "HT-3",
                "HT-4",
                "HT-5",
            )
        )
    hto_rows = []
    hto_records = []
    for tag in hto_tags:
        matches = sorted(set(adt.get(_canonical_adt(tag), [])))
        if len(matches) != 1:
            raise PermissionError(f"{run} does not uniquely resolve HTO {tag}")
        hto_rows.append(matches[0])
        hto_records.append({"tag": tag, "row_1based": matches[0] + 1})
    return {
        "human_rows": human,
        "mouse_rows": mouse,
        "panel_rows": panel_rows,
        "hto_rows": tuple(hto_rows),
        "primary_resolution": panel_records["primary"],
        "legacy_resolution": panel_records["legacy_secondary"],
        "hto_resolution": hto_records,
    }


def _metadata_inventory(
    geo_path: Path = DEFAULT_GEO_METADATA,
    author_path: Path = DEFAULT_AUTHOR_SHEET,
) -> dict[str, Any]:
    manifest = _read_json(DEFAULT_MANIFEST)["metadata"]
    for path, key in (
        (geo_path, "geo_sample_table"),
        (author_path, "author_sample_sheet"),
    ):
        record = manifest[key]
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise PermissionError(f"{key} differs from the frozen source")

    with gzip.open(geo_path, "rt", newline="") as stream:
        geo_rows = list(csv.DictReader(stream))
    with author_path.open(newline="") as stream:
        author_rows = list(csv.DictReader(stream))
    if len(geo_rows) != 46 or len(author_rows) != 46:
        raise PermissionError("GSE189050 metadata does not contain 46 subjects")

    geo_by_subject = {row["Sample name"]: row for row in geo_rows}
    author_by_subject = {row["Subject_id"]: row for row in author_rows}
    if len(geo_by_subject) != 46 or set(geo_by_subject) != set(author_by_subject):
        raise PermissionError("GEO and author subject axes differ")
    for subject, geo in geo_by_subject.items():
        if author_by_subject[subject]["run"] != geo["run"]:
            raise PermissionError(f"run assignment differs for {subject}")

    roles = {
        subject: next(role for role, pools in ROLE_POOLS.items() if geo["run"] in pools)
        for subject, geo in geo_by_subject.items()
    }
    expected = {"calibration": 9, "pilot": 9, "held": 28}
    if {role: list(roles.values()).count(role) for role in expected} != expected:
        raise PermissionError("metadata role counts differ from the designation")

    hto_by_pool: dict[str, dict[str, str]] = {}
    for row in author_rows:
        mapping = hto_by_pool.setdefault(row["run"], {})
        if row["Hashtag"] in mapping:
            raise PermissionError(f"duplicate HTO assignment in {row['run']}")
        mapping[row["Hashtag"]] = row["Subject_id"]
    if set(hto_by_pool) != set(RUNS) or any(
        len(hto_by_pool[run]) != (4 if run in RUNS[:4] else 5) for run in RUNS
    ):
        raise PermissionError("HTO mapping differs from the ten-pool design")

    author_status = {
        "Unaffected Control": ("control", "none"),
        "SLE INACT": ("SLE", "low"),
        "SLE ACT": ("SLE", "high"),
    }
    conflicts = []
    for subject in sorted(geo_by_subject):
        geo = geo_by_subject[subject]
        author = author_by_subject[subject]
        expected_status = author_status[author["classification"]]
        observed_status = (geo["subject_status"], geo["disease_activity"])
        if expected_status != observed_status or author["age"] != geo["age"]:
            conflicts.append(subject)
    if conflicts != ["SUB235957", "SUB236000"]:
        raise PermissionError("clinical metadata conflict set differs from disclosure")

    return {
        "geo_by_subject": geo_by_subject,
        "author_by_subject": author_by_subject,
        "roles": roles,
        "hto_by_pool": hto_by_pool,
        "clinical_conflicts": conflicts,
    }


def run_preflight(output_path: Path = DEFAULT_PREFLIGHT) -> dict[str, Any]:
    inventory = _metadata_inventory()
    _validate_source_bytes()
    schemas = {run: _schema_from_archive(run, _archive_path(run)) for run in RUNS}
    donors = []
    for subject in sorted(inventory["geo_by_subject"]):
        geo = inventory["geo_by_subject"][subject]
        author = inventory["author_by_subject"][subject]
        donors.append(
            {
                "subject": subject,
                "pool": geo["run"],
                "role": inventory["roles"][subject],
                "hto": author["Hashtag"],
                "ancestry": geo["ancestry"],
                "subject_status": geo["subject_status"],
                "disease_activity": geo["disease_activity"],
            }
        )
    payload = {
        "schema": "gse189050-schema-preflight/1.0",
        "status": "PASS_BEFORE_BARCODE_OR_MATRIX_VALUE_ACCESS",
        "created_at_utc": _timestamp(),
        "accession": "GSE189050",
        "bindings": {
            "source_manifest_sha256": _sha256(DEFAULT_MANIFEST),
            "designation_sha256": _sha256(DEFAULT_DESIGNATION),
            "access_record_sha256": _sha256(DEFAULT_ACCESS),
            "geo_metadata_sha256": _sha256(DEFAULT_GEO_METADATA),
            "author_sheet_sha256": _sha256(DEFAULT_AUTHOR_SHEET),
        },
        "access_audit": {
            "archives_downloaded": 10,
            "tar_member_headers_inspected": True,
            "feature_tables_read": 10,
            "barcode_values_read": 0,
            "matrix_members_opened": 0,
            "matrix_coordinate_lines_read": 0,
            "rna_numeric_values_read": 0,
            "hto_numeric_values_read": 0,
            "adt_numeric_values_read": 0,
        },
        "role_pools": {key: list(value) for key, value in ROLE_POOLS.items()},
        "role_counts": {"calibration": 9, "pilot": 9, "held": 28},
        "clinical_metadata_authority": "GEO sample table",
        "clinical_conflicts_ignored_from_author_sheet": inventory["clinical_conflicts"],
        "donors": donors,
        "pool_schemas": schemas,
    }
    _write_json(output_path, payload)
    return payload


def _validated_preflight() -> dict[str, Any]:
    frozen = _read_json(DEFAULT_PREFLIGHT)
    if frozen.get("status") != "PASS_BEFORE_BARCODE_OR_MATRIX_VALUE_ACCESS":
        raise PermissionError("schema preflight is not a clean pass")
    expected = {
        "source_manifest_sha256": _sha256(DEFAULT_MANIFEST),
        "designation_sha256": _sha256(DEFAULT_DESIGNATION),
        "access_record_sha256": _sha256(DEFAULT_ACCESS),
        "geo_metadata_sha256": _sha256(DEFAULT_GEO_METADATA),
        "author_sheet_sha256": _sha256(DEFAULT_AUTHOR_SHEET),
    }
    if frozen.get("bindings") != expected:
        raise PermissionError("schema preflight bindings differ")
    audit = frozen.get("access_audit", {})
    if any(
        audit.get(key) != 0
        for key in (
            "barcode_values_read",
            "matrix_members_opened",
            "matrix_coordinate_lines_read",
            "rna_numeric_values_read",
            "hto_numeric_values_read",
            "adt_numeric_values_read",
        )
    ):
        raise PermissionError("preflight records prohibited numeric access")
    return frozen


def _read_member_lines(stream: BinaryIO) -> list[str]:
    return [line.decode().rstrip("\r\n") for line in stream]


def _next_token(line: bytes, start: int) -> tuple[bytes, int]:
    length = len(line)
    while start < length and line[start] in b" \t\r\n":
        start += 1
    end = start
    while end < length and line[end] not in b" \t\r\n":
        end += 1
    if end == start:
        raise ValueError("MatrixMarket coordinate is missing a token")
    return line[start:end], end


def _coordinate_prefix(line: bytes) -> tuple[int, int, int]:
    row_token, cursor = _next_token(line, 0)
    column_token, cursor = _next_token(line, cursor)
    while cursor < len(line) and line[cursor] in b" \t":
        cursor += 1
    if cursor >= len(line):
        raise ValueError("MatrixMarket coordinate is missing its value")
    return int(row_token) - 1, int(column_token) - 1, cursor


def _matrix_dimensions(stream: BinaryIO) -> tuple[int, int, int]:
    header = stream.readline().rstrip(b"\r\n")
    if header != b"%%MatrixMarket matrix coordinate integer general":
        raise ValueError("matrix is not general integer coordinate MatrixMarket")
    while True:
        line = stream.readline()
        if not line:
            raise ValueError("MatrixMarket dimensions are absent")
        if not line.startswith(b"%"):
            fields = line.split()
            if len(fields) != 3:
                raise ValueError("MatrixMarket dimensions are malformed")
            return tuple(int(value) for value in fields)  # type: ignore[return-value]


def _read_pool_archive(
    run: str,
    path: Path,
    hto_by_tag: dict[str, str],
    *,
    read_biological_adt: bool,
) -> dict[str, Any]:
    record = _read_json(DEFAULT_MANIFEST)["archives"][run]
    members = record["members"]
    barcodes: list[str] | None = None
    features: list[tuple[str, str, str]] | None = None
    result: dict[str, Any] | None = None
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if member.name == members["barcodes"]["name"]:
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"{run} barcode member is unreadable")
                barcodes = _read_member_lines(stream)
                if not barcodes or len(barcodes) != len(set(barcodes)):
                    raise ValueError(f"{run} barcode axis is empty or nonunique")
            elif member.name == members["features"]["name"]:
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"{run} feature member is unreadable")
                feature_bytes = stream.read()
                if _bytes_sha256(feature_bytes) != record["feature_table_sha256"]:
                    raise PermissionError(f"{run} feature table differs")
                features = _parse_feature_bytes(feature_bytes)
            elif member.name == members["matrix"]["name"]:
                if barcodes is None or features is None:
                    raise ValueError(f"{run} archive member ordering differs")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"{run} matrix member is unreadable")
                result = _stream_pool_matrix(
                    run,
                    stream,
                    barcodes,
                    features,
                    hto_by_tag,
                    read_biological_adt=read_biological_adt,
                )
                break
    if result is None:
        raise ValueError(f"{run} archive lacks its matrix member")
    return result


def _stream_pool_matrix(
    run: str,
    stream: BinaryIO,
    barcodes: list[str],
    features: list[tuple[str, str, str]],
    hto_by_tag: dict[str, str],
    *,
    read_biological_adt: bool,
) -> dict[str, Any]:
    tags = tuple(sorted(hto_by_tag, key=lambda value: int(_canonical_adt(value)[2:])))
    resolution = _resolve_features(features, run, tags)
    rows, columns, nonzero = _matrix_dimensions(stream)
    if rows != len(features) or columns != len(barcodes):
        raise ValueError(f"{run} matrix dimensions differ from its axes")

    human_total = np.zeros(columns, dtype=np.int64)
    mouse_total = np.zeros(columns, dtype=np.int64)
    hto = np.zeros((columns, len(tags)), dtype=np.int64)
    panel_counts = {
        name: {
            "rna": np.zeros((columns, len(panel["markers"])), dtype=np.int64),
            **(
                {"adt": np.zeros((columns, len(panel["markers"])), dtype=np.int64)}
                if read_biological_adt
                else {}
            ),
        }
        for name, panel in PANELS.items()
    }
    hto_lookup = {row: index for index, row in enumerate(resolution["hto_rows"])}
    rna_lookup: dict[int, list[tuple[str, int]]] = {}
    adt_lookup: dict[int, list[tuple[str, int]]] = {}
    for panel_name, panel_rows in resolution["panel_rows"].items():
        for marker, row in enumerate(panel_rows["rna"]):
            rna_lookup.setdefault(row, []).append((panel_name, marker))
        for marker, row in enumerate(panel_rows["adt"]):
            adt_lookup.setdefault(row, []).append((panel_name, marker))

    converted = {"human_gex": 0, "mouse_gex": 0, "hto": 0, "biological_adt": 0}
    skipped_adt = 0
    lines = 0
    for line in stream:
        if not line.strip():
            continue
        row, column, value_start = _coordinate_prefix(line)
        if row < 0 or row >= rows or column < 0 or column >= columns:
            raise ValueError(f"{run} has an out-of-range MatrixMarket coordinate")
        lines += 1
        if resolution["human_rows"][row]:
            value = int(line[value_start:])
            human_total[column] += value
            converted["human_gex"] += 1
            for panel_name, marker in rna_lookup.get(row, ()):
                panel_counts[panel_name]["rna"][column, marker] += value
        elif resolution["mouse_rows"][row]:
            value = int(line[value_start:])
            mouse_total[column] += value
            converted["mouse_gex"] += 1
        elif row in hto_lookup:
            value = int(line[value_start:])
            hto[column, hto_lookup[row]] += value
            converted["hto"] += 1
        elif row in adt_lookup:
            if read_biological_adt:
                value = int(line[value_start:])
                for panel_name, marker in adt_lookup[row]:
                    panel_counts[panel_name]["adt"][column, marker] += value
                converted["biological_adt"] += 1
            else:
                skipped_adt += 1
    if lines != nonzero:
        raise ValueError(f"{run} MatrixMarket nonzero count differs from its header")
    if any(np.any(values < 0) for values in (human_total, mouse_total, hto)):
        raise ValueError(f"{run} contains negative counts")
    return {
        "barcodes": barcodes,
        "tags": tags,
        "human_total": human_total,
        "mouse_total": mouse_total,
        "hto": hto,
        "panels": panel_counts,
        "audit": {
            "matrix_coordinate_lines_seen": lines,
            "numeric_values_converted": converted,
            "biological_adt_coordinate_lines_skipped_without_value_conversion": skipped_adt,
            "biological_adt_values_serialized": 0,
            "full_matrix_materialized": False,
        },
    }


def _hto_classification(pool: dict[str, Any]) -> dict[str, Any]:
    hto = np.asarray(pool["hto"], dtype=np.int64)
    human = np.asarray(pool["human_total"], dtype=np.int64)
    mouse = np.asarray(pool["mouse_total"], dtype=np.int64)
    total = hto.sum(axis=1)
    order = np.argsort(hto, axis=1, kind="stable")
    top_index = order[:, -1]
    second_index = order[:, -2]
    row = np.arange(len(hto))
    top = hto[row, top_index]
    second = hto[row, second_index]
    unique = top > second
    ratio = (top + 1.0) / (second + 1.0)
    fraction = np.divide(
        top, total, out=np.zeros(len(top), dtype=float), where=total > 0
    )
    gex_total = human + mouse
    human_fraction = np.divide(
        human,
        gex_total,
        out=np.zeros(len(human), dtype=float),
        where=gex_total > 0,
    )
    hto_singlet = (
        unique
        & (total >= HTO_MINIMUM_TOTAL)
        & (ratio >= HTO_MINIMUM_RATIO)
        & (fraction >= HTO_MINIMUM_TOP_FRACTION)
    )
    accepted = hto_singlet & (human_fraction >= HUMAN_GEX_FRACTION_MINIMUM)
    negative = total < HTO_MINIMUM_TOTAL
    ambiguous = ~negative & ~hto_singlet
    return {
        "top_index": top_index,
        "accepted": accepted,
        "hto_singlet": hto_singlet,
        "negative": negative,
        "ambiguous": ambiguous,
        "human": human_fraction >= HUMAN_GEX_FRACTION_MINIMUM,
    }


def _pool_qc(
    run: str,
    pool: dict[str, Any],
    classification: dict[str, Any],
    hto_by_tag: dict[str, str],
) -> dict[str, Any]:
    human = np.asarray(classification["human"], dtype=bool)
    denominator = int(np.count_nonzero(human))
    if denominator == 0:
        fractions = {"singlet": 0.0, "negative": 1.0, "ambiguous": 0.0}
    else:
        fractions = {
            "singlet": float(
                np.count_nonzero(classification["accepted"] & human) / denominator
            ),
            "negative": float(
                np.count_nonzero(classification["negative"] & human) / denominator
            ),
            "ambiguous": float(
                np.count_nonzero(classification["ambiguous"] & human) / denominator
            ),
        }
    tags = tuple(pool["tags"])
    yields = {
        hto_by_tag[tag]: int(
            np.count_nonzero(
                classification["accepted"] & (classification["top_index"] == tag_index)
            )
        )
        for tag_index, tag in enumerate(tags)
    }
    represented = sorted(subject for subject, count in yields.items() if count > 0)
    retained = sorted(
        subject for subject, count in yields.items() if count >= CELL_BUDGET
    )
    positive_yields = [count for count in yields.values() if count > 0]
    imbalance = max(positive_yields, default=0) / max(
        min(positive_yields, default=1), 1
    )
    checks = {
        "exact_declared_hto_tag_set_represented": len(represented) == len(hto_by_tag),
        "human_pool_singlet_fraction_at_least_0_50": fractions["singlet"] >= 0.50,
        "human_pool_negative_fraction_at_most_0_30": fractions["negative"] <= 0.30,
        "human_pool_ambiguous_fraction_at_most_0_30": fractions["ambiguous"] <= 0.30,
        "positive_donor_yield_ratio_at_most_4": imbalance <= 4.0,
    }
    return {
        "pool": run,
        "filtered_barcodes": len(pool["barcodes"]),
        "human_barcodes": denominator,
        "classification_fractions_among_human": fractions,
        "accepted_singlets_by_subject": yields,
        "represented_subjects": represented,
        "retained_subjects_at_512": retained,
        "positive_donor_yield_ratio": imbalance,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _reduce_pool(
    run: str,
    pool: dict[str, Any],
    hto_by_tag: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    classification = _hto_classification(pool)
    qc = _pool_qc(run, pool, classification, hto_by_tag)
    records: dict[str, dict[str, Any]] = {}
    tags = tuple(pool["tags"])
    barcodes = list(pool["barcodes"])
    for tag_index, tag in enumerate(tags):
        subject = hto_by_tag[tag]
        candidates = np.flatnonzero(
            classification["accepted"] & (classification["top_index"] == tag_index)
        ).tolist()
        ordered = sorted(
            candidates,
            key=lambda index: (
                _salted_hash(
                    CELL_SELECTION_SALT,
                    subject,
                    f"{run}:{barcodes[index]}",
                ),
                barcodes[index],
            ),
        )
        selected = ordered[:CELL_BUDGET]
        cell_ids = [f"{run}:{barcodes[index]}" for index in selected]
        records[subject] = {
            "pool": run,
            "available_singlets": len(candidates),
            "eligible": len(candidates) >= CELL_BUDGET,
            "cells": cell_ids,
            "selected_axis_sha256": _axis_sha256(cell_ids),
            "panels": {
                panel_name: {
                    modality: np.asarray(values)[selected].copy()
                    for modality, values in panel.items()
                }
                for panel_name, panel in pool["panels"].items()
            },
        }
    return records, qc


def _support_gate(
    role: str,
    records: dict[str, dict[str, Any]],
    pools: Iterable[str],
) -> dict[str, Any]:
    pool_list = tuple(pools)
    eligible = sorted(
        subject
        for subject, record in records.items()
        if record["eligible"] and record["pool"] in pool_list
    )
    counts = {
        pool: sum(records[subject]["pool"] == pool for subject in eligible)
        for pool in pool_list
    }
    if role in {"calibration", "pilot"}:
        checks = {
            "at_least_7_donors": len(eligible) >= 7,
            "at_least_3_donors_per_pool": all(value >= 3 for value in counts.values()),
        }
    elif role == "held":
        checks = {
            "at_least_22_donors": len(eligible) >= 22,
            "all_six_pools_represented": all(value >= 1 for value in counts.values()),
        }
    else:
        raise ValueError(f"unknown role {role}")
    return {
        "role": role,
        "eligible_donors": eligible,
        "eligible_donor_count": len(eligible),
        "eligible_donors_by_pool": counts,
        "checks": checks,
        "passes": all(checks.values()),
    }


@contextmanager
def _configured_base(panel_name: str) -> Iterator[None]:
    panel = PANELS[panel_name]
    settings = {
        "MARKERS": tuple(panel["markers"]),
        "CELL_BUDGET": CELL_BUDGET,
        "MINIMUM_INFORMATIVE_ENTITIES": panel["minimum_informative"],
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
        yield
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


def _counts_for_panel(
    records: dict[str, dict[str, Any]], subjects: Iterable[str], panel_name: str
) -> tuple[dict[str, list[str]], dict[str, dict[str, np.ndarray]]]:
    selected = {subject: records[subject]["cells"] for subject in subjects}
    counts = {
        subject: {
            modality: np.asarray(values)
            for modality, values in records[subject]["panels"][panel_name].items()
        }
        for subject in subjects
    }
    return selected, counts


def _records_for_panel(
    records: dict[str, dict[str, Any]], subjects: list[str], panel_name: str
) -> dict[str, dict[str, Any]]:
    selected, counts = _counts_for_panel(records, subjects, panel_name)
    with _configured_base(panel_name):
        return base._records_from_counts(subjects, selected, counts)


def _primary_configs() -> list[model_core.PrimaryConfig]:
    return [
        model_core.PrimaryConfig(neighbor, heterogeneity, ridge, graph, transport)
        for neighbor, heterogeneity, ridge, graph, transport in product(
            NEIGHBOR_GRID,
            HETEROGENEITY_GRID,
            RIDGE_GRID,
            GRAPH_GRID,
            TRANSPORT_GRID,
        )
    ]


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


def _select_classical_head_to_head(
    panel_records: dict[str, dict[str, Any]],
    calibration: list[str],
    pilot: list[str],
    panel_name: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with _configured_base(panel_name):
        with base._configured_core():
            tables = np.asarray(
                [panel_records[subject]["tables"] for subject in calibration]
            )
            selected: dict[str, Any] = {
                "status": "PRESPECIFIED_NON_GATING_CLASSICAL_HEAD_TO_HEAD",
                "methods": {},
                "refusals": {},
            }
            pilot_losses: dict[str, np.ndarray] = {}
            for method in (
                "common_effect_stratified_cmle",
                "pooled_saturated_poisson_interaction",
            ):
                try:
                    log_odds = _classical_log_odds(method, tables)
                except (
                    ValueError,
                    FloatingPointError,
                    CouplingEstimationRefusal,
                ) as error:
                    selected["refusals"][method] = str(error)
                    continue
                by_alpha: dict[float, np.ndarray] = {}
                for alpha in TRANSPORT_GRID:
                    losses = np.empty(len(pilot))
                    for index, subject in enumerate(pilot):
                        truth = np.asarray(panel_records[subject]["tables"])
                        rows, columns = model_core._margins(truth)
                        prediction = model_core._predict_log_odds(
                            log_odds, rows, columns, alpha
                        )
                        losses[index] = model_core._donor_loss(truth, prediction)
                    by_alpha[alpha] = losses
                alpha = min(
                    by_alpha,
                    key=lambda value: (float(by_alpha[value].mean()), value),
                )
                selected["methods"][method] = {
                    "transport_multiplier": alpha,
                    "pilot_mean_loss": float(by_alpha[alpha].mean()),
                    "candidate_losses": {
                        str(value): float(losses.mean())
                        for value, losses in by_alpha.items()
                    },
                }
                pilot_losses[method] = by_alpha[alpha]
    return selected, pilot_losses


def _refit_classical_head_to_head(
    panel_records: dict[str, dict[str, Any]],
    development: list[str],
    panel_name: str,
    selected: dict[str, Any],
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    refusals: dict[str, str] = {}
    with _configured_base(panel_name):
        with base._configured_core():
            tables = np.asarray(
                [panel_records[subject]["tables"] for subject in development]
            )
            for method, selection in selected["methods"].items():
                try:
                    values = _classical_log_odds(method, tables)
                except (
                    ValueError,
                    FloatingPointError,
                    CouplingEstimationRefusal,
                ) as error:
                    refusals[method] = str(error)
                    continue
                models[method] = {
                    "transport_multiplier": selection["transport_multiplier"],
                    "population_log_odds": values.tolist(),
                }
    return {"models": models, "refusals": refusals}


def _exact_sign_test(differences: np.ndarray) -> dict[str, Any]:
    values = np.asarray(differences, dtype=float)
    nonzero = values[values != 0.0]
    favorable = int(np.count_nonzero(nonzero < 0.0))
    n = len(nonzero)
    tail = sum(math.comb(n, value) for value in range(favorable, n + 1)) / (2**n)
    return {
        "nonzero_donors": n,
        "favorable_donors": favorable,
        "one_sided_p": float(tail),
    }


def _bootstrap_mean(
    differences: np.ndarray,
    blocks: list[str] | None,
    *,
    seed_offset: int = 0,
) -> np.ndarray:
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    output = np.empty(BOOTSTRAPS)
    chunk = 1000
    if blocks is None:
        for start in range(0, BOOTSTRAPS, chunk):
            stop = min(start + chunk, BOOTSTRAPS)
            indices = rng.integers(0, len(values), size=(stop - start, len(values)))
            output[start:stop] = values[indices].mean(axis=1)
        return output
    axis = np.asarray(blocks, dtype=object)
    unique = np.asarray(sorted(set(blocks)), dtype=object)
    sums = np.asarray([values[axis == block].sum() for block in unique])
    counts = np.asarray([np.count_nonzero(axis == block) for block in unique])
    for start in range(0, BOOTSTRAPS, chunk):
        stop = min(start + chunk, BOOTSTRAPS)
        indices = rng.integers(0, len(unique), size=(stop - start, len(unique)))
        output[start:stop] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return output


def _comparison(
    subjects: list[str],
    subject_pools: dict[str, str],
    primary: np.ndarray,
    comparator: np.ndarray,
    *,
    held: bool,
    gating: bool,
) -> dict[str, Any]:
    difference = np.asarray(primary) - np.asarray(comparator)
    pools = [subject_pools[subject] for subject in subjects]
    independent_blocks = ["s5" if pool in {"s5a", "s5b"} else pool for pool in pools]
    primary_bootstrap = _bootstrap_mean(difference, independent_blocks)
    donor_bootstrap = _bootstrap_mean(difference, None, seed_offset=1)
    physical_pool_bootstrap = _bootstrap_mean(difference, pools, seed_offset=2)
    interval = np.quantile(primary_bootstrap, [0.025, 0.975], method="linear")
    donor_interval = np.quantile(donor_bootstrap, [0.025, 0.975], method="linear")
    physical_interval = np.quantile(
        physical_pool_bootstrap, [0.025, 0.975], method="linear"
    )
    pool_means = {
        pool: float(difference[np.asarray(pools) == pool].mean())
        for pool in sorted(set(pools))
    }
    block_means = {
        block: float(difference[np.asarray(independent_blocks) == block].mean())
        for block in sorted(set(independent_blocks))
    }
    relative = 1.0 - float(np.mean(primary) / np.mean(comparator))
    favorable = int(np.count_nonzero(difference < 0.0))
    required = math.ceil(0.8 * len(subjects))
    sign_test = _exact_sign_test(difference)
    checks = {
        "relative_deviance_reduction_at_least_five_percent": relative >= 0.05,
        "independent_run_block_bootstrap_upper_95_below_zero": interval[1] < 0.0,
        "favorable_donor_count_reached": favorable >= required,
        "donor_sign_test_p_at_most_0_025": sign_test["one_sided_p"] <= 0.025,
        "every_physical_pool_mean_negative": all(
            value < 0.0 for value in pool_means.values()
        ),
    }
    return {
        "gating": gating,
        "primary_mean_loss": float(np.mean(primary)),
        "comparator_mean_loss": float(np.mean(comparator)),
        "relative_deviance_reduction": relative,
        "mean_paired_difference": float(difference.mean()),
        "donor_bootstrap_95_interval": donor_interval.tolist(),
        "independent_run_block_bootstrap_95_interval": interval.tolist(),
        "physical_pool_block_bootstrap_sensitivity_95_interval": physical_interval.tolist(),
        "bootstrap_draws": BOOTSTRAPS,
        "favorable_donors": favorable,
        "required_favorable_donors": required,
        "donor_exact_sign_test": sign_test,
        "physical_pool_mean_differences": pool_means,
        "independent_run_block_mean_differences": block_means,
        "s5a_s5b_counted_as_one_independent_run_block": held,
        "checks": checks,
        "passes": all(checks.values()),
        "donor_differences": {
            subject: float(value) for subject, value in zip(subjects, difference)
        },
    }


def _gate(
    subjects: list[str],
    subject_pools: dict[str, str],
    losses: dict[str, np.ndarray],
    *,
    held: bool,
) -> dict[str, Any]:
    residual = _comparison(
        subjects,
        subject_pools,
        losses["primary"],
        losses["best_residual"],
        held=held,
        gating=True,
    )
    destroyed = _comparison(
        subjects,
        subject_pools,
        losses["primary"],
        losses["destroyed_link"],
        held=held,
        gating=True,
    )
    return {
        "primary_vs_selected_classical_residual": residual,
        "primary_vs_destroyed_link": destroyed,
        "passes": residual["passes"] and destroyed["passes"],
    }


def _load_pools(
    pools: Iterable[str],
    inventory: dict[str, Any],
    *,
    read_biological_adt: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    pool_qc: dict[str, Any] = {}
    stream_audit: dict[str, Any] = {}
    for run in pools:
        pool = _read_pool_archive(
            run,
            _archive_path(run),
            inventory["hto_by_pool"][run],
            read_biological_adt=read_biological_adt,
        )
        reduced, qc = _reduce_pool(run, pool, inventory["hto_by_pool"][run])
        overlap = set(records) & set(reduced)
        if overlap:
            raise PermissionError(
                f"subjects occur in multiple pools: {sorted(overlap)}"
            )
        records.update(reduced)
        pool_qc[run] = qc
        stream_audit[run] = pool["audit"]
    return records, {"pool_qc": pool_qc, "stream_access": stream_audit}


def _loss_json(subjects: list[str], losses: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        method: {subject: float(value) for subject, value in zip(subjects, values)}
        for method, values in losses.items()
    }


def _secondary_comparisons(
    subjects: list[str],
    subject_pools: dict[str, str],
    losses: dict[str, np.ndarray],
    *,
    held: bool,
) -> dict[str, Any]:
    return {
        method: _comparison(
            subjects,
            subject_pools,
            losses["primary"],
            losses[method],
            held=held,
            gating=False,
        )
        for method in (
            "common_effect_stratified_cmle",
            "pooled_saturated_poisson_interaction",
        )
        if method in losses
    }


def _develop_panel(
    panel_name: str,
    records: dict[str, dict[str, Any]],
    calibration: list[str],
    pilot: list[str],
) -> dict[str, Any]:
    panel_records = _records_for_panel(records, calibration + pilot, panel_name)
    with _configured_base(panel_name):
        selection, _, pilot_losses = base._select_on_pilot(
            panel_records, calibration, pilot
        )
    return {
        "panel_records": panel_records,
        "selection": selection,
        "pilot_losses": pilot_losses,
    }


def _refit_panel(
    panel_name: str,
    panel_records: dict[str, dict[str, Any]],
    development: list[str],
    selection: dict[str, Any],
) -> dict[str, Any]:
    with _configured_base(panel_name):
        with base._configured_core():
            return model_core._fit_models(panel_records, development, selection)


def run_development(output_path: Path = DEFAULT_DEVELOPMENT) -> dict[str, Any]:
    protocol_commit = _require_public_tag(PROTOCOL_TAG, PROTOCOL_BINDINGS)
    _validated_preflight()
    _validate_source_bytes()
    inventory = _metadata_inventory()
    records, access = _load_pools(
        CALIBRATION_POOLS + PILOT_POOLS,
        inventory,
        read_biological_adt=True,
    )
    support = {
        "calibration": _support_gate("calibration", records, CALIBRATION_POOLS),
        "pilot": _support_gate("pilot", records, PILOT_POOLS),
    }
    qc_pass = all(value["passes"] for value in access["pool_qc"].values())
    if not qc_pass or not all(value["passes"] for value in support.values()):
        payload = {
            "schema": "gse189050-development/1.0",
            "status": "TERMINAL_DEVELOPMENT_QC_OR_SUPPORT_FAILURE",
            "created_at_utc": _timestamp(),
            "protocol_commit": protocol_commit,
            "support": support,
            "access": access,
            "terminal_rule": "No held matrix or barcode value may be accessed.",
        }
        _write_json(output_path, payload)
        return payload

    calibration = support["calibration"]["eligible_donors"]
    pilot = support["pilot"]["eligible_donors"]
    subject_pools = {subject: records[subject]["pool"] for subject in records}
    primary = _develop_panel("primary", records, calibration, pilot)
    primary_losses = dict(primary["pilot_losses"])
    try:
        classical_selection, classical_losses = _select_classical_head_to_head(
            primary["panel_records"], calibration, pilot, "primary"
        )
    except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
        classical_selection = {
            "status": "NON_GATING_CLASSICAL_ESTIMATOR_REFUSAL",
            "reason": str(error),
        }
        classical_losses = {}
    primary_losses.update(classical_losses)
    gate = _gate(pilot, subject_pools, primary_losses, held=False)
    secondary_head_to_head = _secondary_comparisons(
        pilot, subject_pools, primary_losses, held=False
    )

    legacy: dict[str, Any]
    try:
        legacy_run = _develop_panel("legacy_secondary", records, calibration, pilot)
        legacy_gate = _gate(
            pilot, subject_pools, legacy_run["pilot_losses"], held=False
        )
        legacy = {
            "status": "COMPLETED_NON_GATING",
            "selection": legacy_run["selection"],
            "pilot_losses": _loss_json(pilot, legacy_run["pilot_losses"]),
            "pilot_gate_reported_not_enforced": legacy_gate,
        }
    except (ValueError, FloatingPointError, CouplingEstimationRefusal) as error:
        legacy = {
            "status": "NON_GATING_ESTIMATOR_REFUSAL",
            "reason": str(error),
        }

    payload = {
        "schema": "gse189050-development/1.0",
        "status": "PILOT_PASS" if gate["passes"] else "TERMINAL_PILOT_FAILURE",
        "created_at_utc": _timestamp(),
        "protocol_commit": protocol_commit,
        "calibration_donors": calibration,
        "pilot_donors": pilot,
        "subject_pools": subject_pools,
        "support": support,
        "access": access,
        "primary_panel": {
            "markers": list(PRIMARY_MARKERS),
            "cognate_feature_mappings": len(PRIMARY_MARKERS),
            "scored_ordered_rna_adt_pairs": len(PRIMARY_MARKERS) ** 2,
            "selection": primary["selection"],
            "pilot_losses": _loss_json(pilot, primary_losses),
            "pilot_gate": gate,
            "classical_head_to_head_selection": classical_selection,
            "classical_head_to_head_non_gating": secondary_head_to_head,
        },
        "legacy_secondary_panel": legacy,
    }
    if gate["passes"]:
        development = calibration + pilot
        payload["primary_panel"]["all_development_models"] = _refit_panel(
            "primary",
            primary["panel_records"],
            development,
            primary["selection"],
        )
        if classical_losses:
            classical_refit = _refit_classical_head_to_head(
                primary["panel_records"],
                development,
                "primary",
                classical_selection,
            )
            payload["primary_panel"]["classical_head_to_head_models"] = classical_refit[
                "models"
            ]
            payload["primary_panel"]["classical_head_to_head_refit_refusals"] = (
                classical_refit["refusals"]
            )
        if legacy.get("status") == "COMPLETED_NON_GATING":
            legacy["all_development_models"] = _refit_panel(
                "legacy_secondary",
                legacy_run["panel_records"],
                development,
                legacy_run["selection"],
            )
    else:
        payload["terminal_rule"] = (
            "No held matrix or barcode value may be accessed after pilot failure."
        )
    _write_json(output_path, payload)
    return payload


def _margins_from_rna(
    rna: np.ndarray, marker_count: int
) -> tuple[np.ndarray, np.ndarray]:
    positives = (np.asarray(rna) > 0).sum(axis=0)
    rows = np.repeat(
        np.stack([CELL_BUDGET - positives, positives], axis=1)[:, None, :],
        marker_count,
        axis=1,
    )
    columns = np.broadcast_to(
        np.asarray([CELL_BUDGET // 2, CELL_BUDGET // 2]),
        (marker_count, marker_count, 2),
    ).copy()
    return rows, columns


def _predict_panel(
    panel_name: str,
    records: dict[str, dict[str, Any]],
    subjects: list[str],
    models: dict[str, Any],
    classical_models: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    samples = []
    marker_count = len(PANELS[panel_name]["markers"])
    with _configured_base(panel_name):
        with base._configured_core():
            for subject in subjects:
                rna = np.asarray(records[subject]["panels"][panel_name]["rna"])
                rows, columns = _margins_from_rna(rna, marker_count)
                predictions = model_core._predict_models(models, rows, columns)
                if classical_models:
                    for method, classical in classical_models.items():
                        predictions[method] = model_core._predict_log_odds(
                            np.asarray(classical["population_log_odds"]),
                            rows,
                            columns,
                            classical["transport_multiplier"],
                        )
                samples.append(
                    {
                        "subject": subject,
                        "pool": records[subject]["pool"],
                        "selected_cell_axis_sha256": records[subject][
                            "selected_axis_sha256"
                        ],
                        "held_rna_state_sha256": _array_sha256(
                            (rna > 0).astype(np.uint8)
                        ),
                        "row_margins": rows.tolist(),
                        "column_margins": columns.tolist(),
                        "predicted_tables": {
                            method: values.tolist()
                            for method, values in predictions.items()
                        },
                        "prediction_sha256": {
                            method: _array_sha256(values)
                            for method, values in predictions.items()
                        },
                    }
                )
    return samples


def run_prediction(output_path: Path = DEFAULT_PREDICTION) -> dict[str, Any]:
    development_commit = _require_public_tag(
        DEVELOPMENT_TAG,
        (*PROTOCOL_BINDINGS, _relative(DEFAULT_DEVELOPMENT)),
    )
    development = _read_json(DEFAULT_DEVELOPMENT)
    if development.get("status") != "PILOT_PASS" or not development.get(
        "primary_panel", {}
    ).get("pilot_gate", {}).get("passes"):
        raise PermissionError("the frozen primary pilot gate did not pass")
    _validate_source_bytes()
    inventory = _metadata_inventory()
    records, access = _load_pools(HELD_POOLS, inventory, read_biological_adt=False)
    if any(
        audit["numeric_values_converted"]["biological_adt"] != 0
        or audit["biological_adt_values_serialized"] != 0
        for audit in access["stream_access"].values()
    ):
        raise PermissionError("held prediction converted or serialized biological ADT")
    support = _support_gate("held", records, HELD_POOLS)
    qc_pass = all(value["passes"] for value in access["pool_qc"].values())
    if not qc_pass or not support["passes"]:
        raise PermissionError(
            "held RNA/HTO QC or support gate failed before prediction"
        )
    subjects = support["eligible_donors"]
    primary = development["primary_panel"]
    panels = {
        "primary": _predict_panel(
            "primary",
            records,
            subjects,
            primary["all_development_models"],
            primary.get("classical_head_to_head_models"),
        )
    }
    legacy = development.get("legacy_secondary_panel", {})
    if legacy.get("status") == "COMPLETED_NON_GATING":
        panels["legacy_secondary"] = _predict_panel(
            "legacy_secondary",
            records,
            subjects,
            legacy["all_development_models"],
        )
    payload = {
        "schema": "gse189050-held-predictions/1.0",
        "status": "HELD_PREDICTIONS_FROZEN_WITHOUT_BIOLOGICAL_ADT_CONVERSION",
        "created_at_utc": _timestamp(),
        "development_commit": development_commit,
        "development_sha256": _sha256(DEFAULT_DEVELOPMENT),
        "held_donors": subjects,
        "held_support": support,
        "access": access,
        "held_biological_adt_values_converted": 0,
        "held_biological_adt_values_serialized": 0,
        "panels": panels,
    }
    _write_json(output_path, payload)
    return payload


def _score_panel(
    panel_name: str,
    records: dict[str, dict[str, Any]],
    subjects: list[str],
    frozen_samples: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    panel_records = _records_for_panel(records, subjects, panel_name)
    frozen = {sample["subject"]: sample for sample in frozen_samples}
    methods = sorted(frozen[subjects[0]]["predicted_tables"])
    losses = {method: np.empty(len(subjects)) for method in methods}
    samples = []
    with _configured_base(panel_name):
        with base._configured_core():
            for subject_index, subject in enumerate(subjects):
                truth = np.asarray(panel_records[subject]["tables"])
                rows, columns = model_core._margins(truth)
                sample = frozen[subject]
                if (
                    records[subject]["selected_axis_sha256"]
                    != sample["selected_cell_axis_sha256"]
                ):
                    raise PermissionError(
                        "held selected-cell axis changed after freeze"
                    )
                if (
                    rows.tolist() != sample["row_margins"]
                    or columns.tolist() != sample["column_margins"]
                ):
                    raise PermissionError(
                        "held margins changed after prediction freeze"
                    )
                donor_losses = {}
                for method in methods:
                    prediction = np.asarray(sample["predicted_tables"][method])
                    if _array_sha256(prediction) != sample["prediction_sha256"][method]:
                        raise PermissionError("held prediction hash differs")
                    loss = model_core._donor_loss(truth, prediction)
                    losses[method][subject_index] = loss
                    donor_losses[method] = float(loss)
                samples.append(
                    {
                        "subject": subject,
                        "pool": records[subject]["pool"],
                        "truth_table_sha256": panel_records[subject]["table_sha256"],
                        "losses": donor_losses,
                    }
                )
    return losses, samples


def run_score(output_path: Path = DEFAULT_SCORE) -> dict[str, Any]:
    prediction_commit = _require_public_tag(
        PREDICTION_TAG,
        (
            *PROTOCOL_BINDINGS,
            _relative(DEFAULT_DEVELOPMENT),
            _relative(DEFAULT_PREDICTION),
        ),
    )
    prediction = _read_json(DEFAULT_PREDICTION)
    if (
        prediction.get("status")
        != ("HELD_PREDICTIONS_FROZEN_WITHOUT_BIOLOGICAL_ADT_CONVERSION")
        or prediction.get("held_biological_adt_values_converted") != 0
    ):
        raise PermissionError("held prediction is not a clean RNA-only freeze")
    _validate_source_bytes()
    inventory = _metadata_inventory()
    records, access = _load_pools(HELD_POOLS, inventory, read_biological_adt=True)
    support = _support_gate("held", records, HELD_POOLS)
    subjects = support["eligible_donors"]
    qc_pass = all(value["passes"] for value in access["pool_qc"].values())
    if not qc_pass or not support["passes"] or subjects != prediction["held_donors"]:
        raise PermissionError("held donor support differs from prediction freeze")
    subject_pools = {subject: records[subject]["pool"] for subject in subjects}
    primary_losses, primary_samples = _score_panel(
        "primary", records, subjects, prediction["panels"]["primary"]
    )
    gate = _gate(subjects, subject_pools, primary_losses, held=True)
    secondary = _secondary_comparisons(
        subjects, subject_pools, primary_losses, held=True
    )
    legacy_result: dict[str, Any] | None = None
    if "legacy_secondary" in prediction["panels"]:
        legacy_losses, legacy_samples = _score_panel(
            "legacy_secondary",
            records,
            subjects,
            prediction["panels"]["legacy_secondary"],
        )
        legacy_result = {
            "losses": _loss_json(subjects, legacy_losses),
            "gate_reported_not_enforced": _gate(
                subjects, subject_pools, legacy_losses, held=True
            ),
            "samples": legacy_samples,
        }
    payload = {
        "schema": "gse189050-confirmation/1.0",
        "status": "CONFIRMATION_PASS" if gate["passes"] else "CONFIRMATION_FAIL",
        "created_at_utc": _timestamp(),
        "prediction_tag": PREDICTION_TAG,
        "prediction_commit": prediction_commit,
        "prediction_sha256": _sha256(DEFAULT_PREDICTION),
        "held_donors": subjects,
        "held_support": support,
        "access": access,
        "primary_panel": {
            "losses": _loss_json(subjects, primary_losses),
            "gate": gate,
            "classical_head_to_head_non_gating": secondary,
            "samples": primary_samples,
        },
        "legacy_secondary_panel": legacy_result,
    }
    _write_json(output_path, payload)
    return payload


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
