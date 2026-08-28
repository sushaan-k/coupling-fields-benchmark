"""Verify a disabled Kotliarov freeze from a fresh public clone."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


EXPECTED_ORIGIN = "https://github.com/sushaan-k/coupling-fields-benchmark"
EXPECTED_TAG = "confirmatory-family-v3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    root = Path(args.repo).resolve()
    origin = subprocess.check_output(
        ["git", "remote", "get-url", "origin"], cwd=root, text=True
    ).strip()
    if origin.removesuffix(".git") != EXPECTED_ORIGIN:
        raise PermissionError("fresh clone origin is not the public benchmark repository")
    observed_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if observed_commit != args.commit:
        raise PermissionError("fresh clone is not at the declared commit")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).strip():
        raise PermissionError("fresh clone worktree is not clean")
    reflog = subprocess.check_output(
        ["git", "reflog", "--format=%gs", "HEAD"], cwd=root, text=True
    ).splitlines()
    if len(reflog) != 1 or not reflog[0].startswith("clone: from "):
        raise PermissionError("verification checkout is not an untouched fresh clone")
    remote_tag = subprocess.check_output(
        ["git", "ls-remote", "--tags", "origin", f"refs/tags/{EXPECTED_TAG}"],
        cwd=root,
        text=True,
    ).strip()
    if not remote_tag or remote_tag.split()[0] != observed_commit:
        raise PermissionError("public confirmatory-family-v3 tag differs from HEAD")

    checksum_records = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError("SHA256SUMS contains an invalid line")
        expected, relative = match.groups()
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"SHA256SUMS mismatch: {relative}")
        checksum_records[relative] = expected

    manifest_path = root / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "PUBLIC_CONFIRMATORY_FAMILY_V3_OUTCOME_ACCESS_DISABLED":
        raise PermissionError("benchmark manifest is not the disabled v3 freeze")
    if manifest.get("immutable_release_tag") != EXPECTED_TAG:
        raise PermissionError("benchmark manifest names a different immutable tag")
    if manifest.get("public_freeze_commit") is not None:
        raise PermissionError("disabled benchmark manifest unexpectedly names a commit")
    for artifact in manifest.get("artifacts", []):
        relative = artifact["path"]
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"manifest artifact is absent: {relative}")
        if path.stat().st_size != artifact["bytes"] or sha256(path) != artifact["sha256"]:
            raise ValueError(f"benchmark manifest mismatch: {relative}")

    designation_path = root / "KOTLIAROV_CANDIDATE_DESIGNATION.json"
    internal_designation = (
        root / "data/confirmation/kotliarov_pbmc/candidate_designation_v1.json"
    )
    if designation_path.read_bytes() != internal_designation.read_bytes():
        raise ValueError("root and internal Kotliarov designations differ")
    record = json.loads(designation_path.read_text())
    if record.get("status") != "OUTCOME_ACCESS_DISABLED":
        raise PermissionError("public candidate freeze is not disabled")
    if record.get("outcome_access_authorized") is not False:
        raise PermissionError("public candidate freeze authorizes outcomes")
    artifact_keys = {
        "protocol",
        "source_manifest",
        "metadata_support_artifact",
        "alias_table",
        "lineage_markers",
        "runner",
        "reducer",
        "test",
        "reducer_test",
        "authorization_template",
        "authorization_script",
        "verification_script",
        "embedding_manifest",
    }
    checked = []
    hash_keys = {
        "metadata_support_artifact": "metadata_support_sha256",
        "alias_table": "alias_sha256",
    }
    for key in sorted(artifact_keys):
        path = root / record[key]
        if sha256(path) != record[hash_keys.get(key, f"{key}_sha256")]:
            raise ValueError(f"artifact hash mismatch: {record[key]}")
        checked.append(record[key])
    embedding_manifest_path = root / record["embedding_manifest"]
    embedding_manifest = json.loads(embedding_manifest_path.read_text())
    if sha256(embedding_manifest_path) != record["embedding_manifest_sha256"]:
        raise ValueError("embedding derivation manifest hash differs")
    if embedding_manifest["output"]["path"] != record["embedding"]:
        raise ValueError("embedding path differs from the public derivation manifest")
    if embedding_manifest["output"]["sha256"] != record["embedding_sha256"]:
        raise ValueError("embedding hash differs from the public derivation manifest")
    for relative, expected in record["implementation_sha256"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"shared implementation mismatch: {relative}")
        checked.append(relative)
    print(
        json.dumps(
            {
                "schema": "kotliarov-public-freeze-verification/1.0",
                "status": "PASS",
                "fresh_clone": True,
                "origin": origin,
                "immutable_tag": EXPECTED_TAG,
                "public_freeze_commit": observed_commit,
                "designation_sha256": sha256(designation_path),
                "all_bound_artifacts_match": True,
                "sha256sum_records_verified": len(checksum_records),
                "manifest_artifacts_verified": len(manifest.get("artifacts", [])),
                "checked_artifacts": checked,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
