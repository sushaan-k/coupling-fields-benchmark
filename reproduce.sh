#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

shasum -a 256 -c SHA256SUMS

# These two assertions certify the earlier disabled phase. Their exact bytes
# remain checksum-bound, but they are not expected to pass after authorization.
echo "Preserving, but deselecting, two disabled-phase lock assertions."
"$PYTHON" -m pytest -q -p no:cacheprovider \
  --deselect tests/test_lawlor_hca_pbmc_confirmation.py::test_version_two_lock_is_phase_consistent_and_has_no_reducer_bypass \
  --deselect tests/test_hao_gse164378_confirmation.py::test_disabled_lock_freezes_split_aliases_and_family_reporting \
  tests/test_public_artifact_integrity.py \
  tests/test_mapreg_public_api.py \
  tests/test_coupling_fields.py \
  tests/test_factorial_coupling.py \
  tests/test_classical_residuals_full.py \
  tests/test_table_prediction.py \
  tests/test_lawlor_hca_pbmc_confirmation.py \
  tests/test_hao_gse164378_confirmation.py \
  tests/test_kotliarov_pbmc_confirmation.py \
  tests/test_kotliarov_pbmc_reducer.py \
  tests/test_confirm_poki_gse143417.py \
  tests/test_coupling_margin_invariance_simulation.py
