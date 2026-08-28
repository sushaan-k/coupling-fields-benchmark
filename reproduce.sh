#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

shasum -a 256 -c SHA256SUMS

"$PYTHON" -m pytest -q -p no:cacheprovider \
  tests/test_public_artifact_integrity.py \
  tests/test_mapreg_public_api.py \
  tests/test_coupling_fields.py \
  tests/test_factorial_coupling.py \
  tests/test_classical_residuals_full.py \
  tests/test_table_prediction.py \
  tests/test_lawlor_hca_pbmc_confirmation.py \
  tests/test_confirm_poki_gse143417.py \
  tests/test_coupling_margin_invariance_simulation.py
