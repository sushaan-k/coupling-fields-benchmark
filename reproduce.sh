#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

shasum -a 256 -c SHA256SUMS
"$PYTHON" -m experiments.build_public_benchmark_release --check
"$PYTHON" -m scripts.verify_public_benchmark_release

# These two assertions certify the earlier disabled phase. Their exact bytes
# remain checksum-bound, but they are not expected to pass after authorization.
echo "Preserving, but deselecting, two disabled-phase lock assertions."
echo "Deselecting the BMMC deposited-axis assertion because its source is not redistributed."
"$PYTHON" -m pytest -q -p no:cacheprovider \
  --deselect tests/test_lawlor_hca_pbmc_confirmation.py::test_version_two_lock_is_phase_consistent_and_has_no_reducer_bypass \
  --deselect tests/test_hao_gse164378_confirmation.py::test_disabled_lock_freezes_split_aliases_and_family_reporting \
  --deselect tests/test_scmmib_bmmc_exact_development.py::test_actual_combined_axis_matches_locked_schema_without_count_access \
  --deselect tests/test_kotliarov_pbmc_confirmation.py::test_disabled_preflight_uses_bound_embedding_manifest_when_binary_is_omitted \
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
  tests/test_scmmib_bmmc_preflight.py \
  tests/test_scmmib_bmmc_confirmation.py \
  tests/test_scmmib_bmmc_exact_development.py \
  tests/test_heterogeneity_adaptive_coupling.py \
  tests/test_context_conditional_coupling.py \
  tests/test_hierarchical_conditional_coupling.py \
  tests/test_longitudinal_conditional_coupling.py \
  tests/test_gse279451_sepsis_confirmation.py \
  tests/test_evaluate_gse279451_sepsis_development.py \
  tests/test_reduce_gse299043_mln.py \
  tests/test_acquire_gse299043_nonheld.py \
  tests/test_evaluate_gse299043_mln_development.py \
  tests/test_gse299043_mln_confirmation.py \
  tests/test_combat_citeseq_preflight.py \
  tests/test_combat_citeseq_confirmation.py \
  tests/test_stephenson_citeseq_confirmation.py \
  tests/test_gse239452_citeseq_confirmation.py \
  tests/test_gse239452_post_access_correction.py \
  tests/test_gse239452_standard_poisson_posthoc.py \
  tests/test_gse314416_citeseq_confirmation.py \
  tests/test_gse179221_candidate.py \
  tests/test_gse179221_bmmc_confirmation.py \
  tests/test_exact_logodds_head_to_head.py \
  tests/test_common_effect_conditional.py \
  tests/test_gse214546_candidate.py \
  tests/test_gse214546_confirmation.py \
  tests/test_gse342939_confirmation.py \
  tests/test_gse342939_candidate.py \
  tests/test_public_benchmark_release_v2.py \
  tests/test_confirm_poki_gse143417.py \
  tests/test_coupling_margin_invariance_simulation.py

# The Kotliarov designation binds the pre-einsum estimator. Exercise that exact
# preflight in an isolated copy without replacing the current GSE dependency.
HISTORICAL_ROOT="$(mktemp -d)"
trap 'rm -rf "$HISTORICAL_ROOT"' EXIT
cp -R "$ROOT/." "$HISTORICAL_ROOT/repo"
cp \
  "$ROOT/mapreg/historical/coupling_fields_29a3875.py" \
  "$HISTORICAL_ROOT/repo/mapreg/coupling_fields.py"
(
  cd "$HISTORICAL_ROOT/repo"
  "$PYTHON" -m pytest -q -p no:cacheprovider \
    tests/test_kotliarov_pbmc_confirmation.py::test_disabled_preflight_uses_bound_embedding_manifest_when_binary_is_omitted
)
rm -rf "$HISTORICAL_ROOT"
trap - EXIT
