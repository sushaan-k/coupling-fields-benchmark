"""Verify the aggregate public benchmark release from local bytes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any


EXPECTED_ORIGIN = "https://github.com/sushaan-k/coupling-fields-benchmark"
EXPECTED_TAG = "coupling-fields-v2.0.1-public-benchmark"
NUMERIC_COMPARISON_FIELDS = {
    "primary_value",
    "comparator_value",
    "primary_minus_comparator",
    "paired_difference_ci_95_low",
    "paired_difference_ci_95_high",
    "relative_improvement",
    "relative_improvement_ci_95_low",
    "relative_improvement_ci_95_high",
    "p_value",
}


def _reject_nonfinite(token: str) -> None:
    raise ValueError(f"non-finite JSON number: {token}")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        rows = list(reader)
    fields = reader.fieldnames or []
    if not fields or any(None in row for row in rows):
        raise ValueError(f"malformed TSV: {path}")
    return fields, rows


def _check_numeric(value: str, label: str) -> None:
    if value == "":
        return
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite numeric cell: {label}")


def _verify_checksums(root: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    lines = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    if lines != sorted(lines, key=lambda line: line.split("  ", 1)[1]):
        raise ValueError("SHA256SUMS paths are not sorted")
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\n]+)", line)
        if match is None:
            raise ValueError("SHA256SUMS contains an invalid line")
        expected, relative = match.groups()
        if relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError(f"unsafe checksum path: {relative}")
        if relative in records:
            raise ValueError(f"duplicate checksum path: {relative}")
        path = root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"SHA256SUMS mismatch: {relative}")
        records[relative] = expected
    return records


def verify(root: Path, *, require_clean: bool = False, require_tag: bool = False) -> None:
    root = root.resolve()
    checksums = _verify_checksums(root)
    manifest = _load_json(root / "benchmark_manifest.json")
    if manifest["schema"] != "coupling-fields-public-benchmark/2.0":
        raise ValueError("unexpected benchmark manifest schema")
    if manifest["public_repository_url"] != EXPECTED_ORIGIN:
        raise ValueError("unexpected public repository URL")
    if manifest["intended_release_tag"] != EXPECTED_TAG:
        raise ValueError("unexpected release tag")
    if manifest["archive_doi"] is not None:
        raise ValueError("manifest invents an archive DOI")
    if manifest["code_license"] != "MIT":
        raise ValueError("manifest code license differs from the release license")

    ledgers = manifest["ledgers"]
    _, panels = _load_tsv(root / ledgers["panels"])
    _, comparisons = _load_tsv(root / ledgers["comparisons"])
    _, sequence = _load_tsv(root / ledgers["sequence"])
    panel_ids = [row["panel_id"] for row in panels]
    comparison_ids = [row["comparison_id"] for row in comparisons]
    sequence_ids = [row["sequence_id"] for row in sequence]
    if len(panel_ids) != len(set(panel_ids)):
        raise ValueError("duplicate panel_id")
    if len(comparison_ids) != len(set(comparison_ids)):
        raise ValueError("duplicate comparison_id")
    if len(sequence_ids) != len(set(sequence_ids)):
        raise ValueError("duplicate sequence_id")
    panel_id_set = set(panel_ids)
    if any(row["panel_id"] not in panel_id_set for row in comparisons + sequence):
        raise ValueError("comparison or sequence row references an unknown panel")

    for row in panels:
        for field in ("primary_value", "ci_95_low", "ci_95_high"):
            _check_numeric(row[field], f"{row['panel_id']}:{field}")
        artifact = row["result_artifact"]
        if not artifact or _sha256(root / artifact) != row["result_sha256"]:
            raise ValueError(f"panel artifact mismatch: {row['panel_id']}")
        if row["outcome_scored"] == "NO" and row["primary_value"]:
            raise ValueError(f"unscored panel has a primary value: {row['panel_id']}")

    for row in comparisons:
        for field in NUMERIC_COMPARISON_FIELDS:
            _check_numeric(row[field], f"{row['comparison_id']}:{field}")
        artifact = row["result_artifact"]
        if not artifact or _sha256(root / artifact) != row["result_sha256"]:
            raise ValueError(f"comparison artifact mismatch: {row['comparison_id']}")

    by_panel: dict[str, list[int]] = {}
    for row in sequence:
        ordinal = int(row["stage_ordinal"])
        by_panel.setdefault(row["panel_id"], []).append(ordinal)
        if row["artifact"]:
            if _sha256(root / row["artifact"]) != row["artifact_sha256"]:
                raise ValueError(f"sequence artifact mismatch: {row['sequence_id']}")
        elif row["artifact_sha256"]:
            raise ValueError(f"hash without sequence artifact: {row['sequence_id']}")
    for panel_id, ordinals in by_panel.items():
        if ordinals != sorted(ordinals) or len(ordinals) != len(set(ordinals)):
            raise ValueError(f"invalid stage order: {panel_id}")

    infrastructure = [
        row for row in panels if row["inference_role"] == "infrastructure_unevaluable"
    ]
    if len(infrastructure) != 1:
        raise ValueError("release must contain one infrastructure-unevaluable row")
    infrastructure_row = infrastructure[0]
    infrastructure_manifest = manifest["infrastructure_unevaluable"]
    if infrastructure_row["panel_id"] != infrastructure_manifest["panel_id"]:
        raise ValueError("infrastructure-unevaluable panel mismatch")
    if (
        infrastructure_row["primary_value"]
        or infrastructure_row["outcome_scored"] != "NO"
    ):
        raise ValueError("infrastructure-unevaluable panel encodes an outcome")
    if infrastructure_manifest["performance_values_recorded"] is not False:
        raise ValueError("infrastructure manifest claims a performance value")
    if infrastructure_manifest["scientific_decision"] is not None:
        raise ValueError("infrastructure manifest assigns a scientific decision")

    counts = manifest["counts"]
    observed_counts = {
        "panel_records": len(panels),
        "scored_panel_records": sum(row["outcome_scored"] == "YES" for row in panels),
        "procedural_refusal_records": sum(
            row["inference_role"] == "procedural_refusal" for row in panels
        ),
        "pending_records": 0,
        "infrastructure_unevaluable_records": len(infrastructure),
        "comparison_records": len(comparisons),
        "sequence_records": len(sequence),
    }
    if counts != observed_counts:
        raise ValueError("manifest counts do not match the ledgers")

    artifact_paths = [record["path"] for record in manifest["artifacts"]]
    if artifact_paths != sorted(set(artifact_paths)):
        raise ValueError("manifest artifact paths are not unique and sorted")
    for record in manifest["artifacts"]:
        path = root / record["path"]
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise ValueError(f"manifest artifact mismatch: {record['path']}")
        if checksums.get(record["path"]) != record["sha256"]:
            raise ValueError(f"manifest artifact absent from SHA256SUMS: {record['path']}")

    if require_clean:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip()
        if status:
            raise ValueError("release checkout is not clean")
    if require_tag:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip()
        if status:
            raise ValueError("tag verification requires a clean checkout")
        origin = subprocess.check_output(
            ["git", "remote", "get-url", "origin"], cwd=root, text=True
        ).strip()
        if origin.removesuffix(".git") != EXPECTED_ORIGIN:
            raise ValueError("fresh clone origin is not the public repository")
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        tag_commit = subprocess.check_output(
            ["git", "rev-list", "-n", "1", EXPECTED_TAG], cwd=root, text=True
        ).strip()
        if not tag_commit or tag_commit != head:
            raise ValueError("release tag does not resolve to HEAD")
        remote_tag = subprocess.check_output(
            ["git", "ls-remote", "--tags", "origin", f"refs/tags/{EXPECTED_TAG}"],
            cwd=root,
            text=True,
        ).strip()
        if not remote_tag or remote_tag.split()[0] != head:
            raise ValueError("public release tag does not resolve to HEAD")
        tracked = set(
            filter(
                None,
                subprocess.check_output(
                    ["git", "ls-files", "-z"], cwd=root
                ).decode().split("\0"),
            )
        )
        tracked.discard("SHA256SUMS")
        if tracked != set(checksums):
            raise ValueError("SHA256SUMS does not cover every tracked release byte")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--require-tag", action="store_true")
    args = parser.parse_args()
    verify(args.root, require_clean=args.require_clean, require_tag=args.require_tag)
    print("PUBLIC_BENCHMARK_RELEASE_VERIFICATION_PASS")


if __name__ == "__main__":
    main()
