"""Seal the exact pre-truth prediction bytes before held pairing is opened."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.confirm_poki_gse143417_conditional_fields import (  # noqa: E402
    CACHE,
    LOCK,
    PREDICTIONS,
    PRETRUTH_DESIGNATION,
    _implementation_sha256,
    _sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--lock", type=Path, default=LOCK)
    parser.add_argument("--output", type=Path, default=PRETRUTH_DESIGNATION)
    parser.add_argument("--prediction-public-url", required=True)
    parser.add_argument("--prediction-public-commit", required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError("pre-truth designation already exists and is immutable")
    lock = json.loads(arguments.lock.read_text())
    if lock.get("status") != "SEALED" or lock.get("outcome_access_authorized") is not True:
        raise PermissionError("outcome lock is not authorized")
    predictions = json.loads(arguments.predictions.read_text())
    if predictions.get("stage") != "PRETRUTH_PREDICTIONS_WRITTEN":
        raise ValueError("prediction record is not at the frozen pre-truth stage")
    implementation = _implementation_sha256()
    provenance = predictions.get("provenance", {})
    if provenance.get("implementation_sha256") != implementation:
        raise ValueError("prediction implementation provenance does not match current bytes")
    cache_sha256 = _sha256(arguments.cache)
    if provenance.get("cache_sha256") != cache_sha256:
        raise ValueError("prediction cache provenance does not match the designated cache")
    if provenance.get("lock_sha256") != _sha256(arguments.lock):
        raise ValueError("prediction record was not made under the current outcome lock")
    if not re.fullmatch(r"[0-9a-f]{40}", arguments.prediction_public_commit):
        raise ValueError("prediction public commit must be a full 40-character Git SHA")
    if (
        not arguments.prediction_public_url.startswith("https://github.com/")
        or arguments.prediction_public_commit not in arguments.prediction_public_url
    ):
        raise ValueError("prediction public URL must bind the stated Git commit")
    designation = {
        "protocol": "gse143417-pokiseq-held-donor/1.0",
        "status": "SEALED_FOR_SCORING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pretruth_predictions": str(arguments.predictions.relative_to(ROOT)),
        "pretruth_predictions_sha256": _sha256(arguments.predictions),
        "pretruth_predictions_public_url": arguments.prediction_public_url,
        "pretruth_predictions_public_commit": arguments.prediction_public_commit,
        "cache": str(arguments.cache.relative_to(ROOT)),
        "cache_sha256": cache_sha256,
        "outcome_lock_sha256": _sha256(arguments.lock),
        "implementation_sha256": implementation,
        "sealed_outcome": "Donor2 target-TGFB pairing and joint tables",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(designation, indent=2, sort_keys=True) + "\n")
    print(json.dumps(designation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
