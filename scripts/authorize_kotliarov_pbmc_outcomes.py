"""Authorize Kotliarov outcome acquisition after an independent public verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
DESIGNATION = ROOT / "data/confirmation/kotliarov_pbmc/candidate_designation_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-commit", required=True)
    parser.add_argument("--public-url", required=True)
    parser.add_argument("--verification", required=True)
    parser.add_argument("--sealed-at-utc", required=True)
    args = parser.parse_args()

    if re.fullmatch(r"[0-9a-f]{40}", args.public_commit) is None:
        raise ValueError("public commit must be 40 lowercase hexadecimal characters")
    parsed = urlparse(args.public_url)
    expected_path = f"/sushaan-k/coupling-fields-benchmark/commit/{args.public_commit}"
    if parsed.scheme != "https" or parsed.netloc != "github.com" or parsed.path != expected_path:
        raise ValueError("public URL must identify the exact GitHub commit")

    verification_path = Path(args.verification)
    verification = json.loads(verification_path.read_text())
    if verification.get("status") != "PASS":
        raise PermissionError("independent verification did not pass")
    if verification.get("public_freeze_commit") != args.public_commit:
        raise PermissionError("verification names a different public commit")
    if verification.get("fresh_clone") is not True:
        raise PermissionError("verification was not performed from a fresh clone")
    if verification.get("all_bound_artifacts_match") is not True:
        raise PermissionError("verification did not match every bound artifact")

    record = json.loads(DESIGNATION.read_text())
    if record.get("schema") != "kotliarov-pbmc-coupling-candidate-designation/1.0":
        raise ValueError("designation schema differs")
    if record.get("status") != "OUTCOME_ACCESS_DISABLED":
        raise PermissionError("designation is not in the disabled phase")
    if record.get("outcome_access_authorized") is not False:
        raise PermissionError("disabled designation is internally inconsistent")
    hash_keys = {
        "metadata_support_artifact": "metadata_support_sha256",
        "alias_table": "alias_sha256",
    }
    for key, relative in record.items():
        hash_key = hash_keys.get(key, f"{key}_sha256")
        if key in {
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
            "embedding",
            "embedding_manifest",
            "authorization_script",
        }:
            if sha256(ROOT / relative) != record.get(hash_key):
                raise ValueError(f"bound artifact changed: {relative}")
    for relative, expected in record["implementation_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"shared implementation changed: {relative}")

    record.update(
        {
            "status": "SEALED",
            "public_freeze_commit": args.public_commit,
            "public_freeze_url": args.public_url,
            "sealed_at_utc": args.sealed_at_utc,
            "outcome_access_authorized": True,
            "public_verification_record": str(verification_path.relative_to(ROOT)),
            "public_verification_sha256": sha256(verification_path),
            "seal_blocker": None,
        }
    )
    DESIGNATION.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
