"""Bound held-RNA summary firewall for the combined GSE288020 10x HDF5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import confirm_gse288020_citeseq as confirmation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor", required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    payload = confirmation.held_rna_reducer_payload(args.donor, args.input)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
