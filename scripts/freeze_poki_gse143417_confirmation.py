"""Bind the GSE143417 confirmation bytes before authorizing outcome access."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/confirmation/gse143417_pokiseq"
LOCK = DATA_DIR / "preanalysis_lock_v1.json"
RAW = DATA_DIR / "GSE143417_RAW.tar"
RAW_BYTES = 492_216_320
RAW_SHA256 = "6bc8bf810fbca8f0585c337ed143d39d8bfbc3f85d623894ebadf4c6f357b632"
BOUND_FILES = (
    "scripts/freeze_poki_gse143417_confirmation.py",
    "scripts/authorize_poki_gse143417_scoring.py",
    "experiments/confirm_poki_gse143417_conditional_fields.py",
    "mapreg/coupling_fields.py",
    "mapreg/classical_residuals.py",
    "mapreg/table_prediction.py",
    "docs/GSE143417_POKISEQ_HELD_DONOR_CONFIRMATION_PROTOCOL_2026-08-27.md",
    "data/confirmation/gse143417_pokiseq/candidate_designation_v1.json",
    "tests/test_confirm_poki_gse143417.py",
    "tests/test_table_prediction.py",
    "tests/test_classical_residuals_full.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authorize-outcome",
        action="store_true",
        help="authorize acquisition only after the protocol has a public commit",
    )
    parser.add_argument("--public-freeze-commit")
    parser.add_argument("--public-freeze-url")
    parser.add_argument("--output", type=Path, default=LOCK)
    arguments = parser.parse_args()
    if arguments.authorize_outcome:
        if (
            not isinstance(arguments.public_freeze_commit, str)
            or re.fullmatch(r"[0-9a-f]{40}", arguments.public_freeze_commit) is None
        ):
            raise ValueError("outcome authorization requires a full public freeze commit")
        if (
            not isinstance(arguments.public_freeze_url, str)
            or not arguments.public_freeze_url.startswith("https://github.com/")
            or arguments.public_freeze_commit not in arguments.public_freeze_url
        ):
            raise ValueError("outcome authorization requires a commit-bound public URL")
    elif arguments.public_freeze_commit or arguments.public_freeze_url:
        raise ValueError("public freeze fields require --authorize-outcome")
    missing = [name for name in BOUND_FILES if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError(f"cannot freeze missing files: {missing}")
    implementation = {name: _sha256(ROOT / name) for name in BOUND_FILES}
    lock = {
        "protocol": "gse143417-pokiseq-held-donor/1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SEALED" if arguments.authorize_outcome else "SEALED_PREOUTCOME",
        "outcome_access_authorized": bool(arguments.authorize_outcome),
        "public_freeze_commit": arguments.public_freeze_commit,
        "public_freeze_url": arguments.public_freeze_url,
        "candidate": "GSE143417 PoKI-seq Donor1 to held Donor2",
        "source": {
            "path": str(RAW.relative_to(ROOT)),
            "bytes": RAW_BYTES,
            "sha256": RAW_SHA256,
        },
        "analysis_stages": [
            "prepare cache mechanically",
            "write pre-truth predictions from held margins and allowed anchors",
            "form and score held target-TGFB pairing only after prediction hash exists",
        ],
        "primary_gate": [
            "held field correlation lower 95% endpoint > 0",
            "primary-minus-destroyed deviance upper 95% endpoint < 0",
            "primary-minus-best-classical deviance upper 95% endpoint < 0",
            "primary-minus-best-matched-field deviance upper 95% endpoint < 0",
        ],
        "implementation_sha256": implementation,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
