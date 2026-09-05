# Reanalysis source versions

These files preserve the exact source and specification bytes bound to completed
analyses. They are provenance snapshots, not additional implementations to use.

- `biological_analysis_specification.md` matches the specification hash in the
  first biological result, before the later random-effects numerical addenda.
- `fixed_margin_biology_counts_v1.py` matches that run's source hash. The current
  module adds aggregate-table replay without changing its scientific results.
- `fit_conditional_random_effects_before_initializer_retry.R` produced the
  first 47 accepted calibration-pair fits. It includes the fixed boundary
  profile and dual integration-tolerance checks.
- `fit_conditional_random_effects_initial_initializer_retry.R` produced nine
  further accepted calibration fits after adding the official lme4 initializer
  retry. The next version checks the variance score before that retry to avoid
  unnecessary interior optimization on boundary candidates.
- `fit_conditional_random_effects_before_zero_information_filter.R` produced
  the remaining 25 calibration fits and 42 accepted source fits. The current
  runner also removes singleton-support factors before calling metafor, avoiding
  infinite offsets in its approximate initializer. These factors equal one
  under every parameter value; no informative donor observations are removed.

The predictive result's `fits.calibration.random_effects.imported_fit_provenance`
and the corresponding source-phase record list each archived script, its SHA-256,
its source-table binding, and the exact
pair IDs it produced. All retained fits target the same conditional marginal
likelihood and passed the same independent likelihood check. The remaining fits
use `experiments/development/fit_conditional_random_effects.R`; its hash is recorded
under each phase's `binding`. Failed and interrupted attempts were not substituted
for accepted estimates. No source-table split or prediction grid changed.

The `stephenson_predictive_reanalysis_before_common_ridge_recovery` JSON and CSV
preserve the first completed score export, which included an explicit failure
of the calibration common-ridge optimizer. The matching Python runner and
specification are archived here as
`reanalyze_stephenson_prediction_before_common_ridge_recovery.py` and
`predictive_specification_before_common_ridge_recovery.md`. The final runner uses
the same strictly convex objective with a scalar score-root solver for that
failure. All other fits, transport choices, and recipient predictions are
unchanged. The final summary reuses existing full-cohort comparison intervals
and gives separate fixed-seed intervals for the 54-donor sensitivity and newly
oriented comparisons.

Run the current modules in `experiments/development/`. The composition control,
threshold sensitivity, and assay-signal sensitivity can be reproduced from the
small distributed sufficient tables and diagnostics without downloading cells.
All of these analyses are post hoc; the original confirmation remains unchanged.

## Exact-conditional replay

The executed R implementation used R 4.6.1, metafor 5.0.1, lme4 2.0.6, and
BiasedUrn 2.0.12. Each fitted pair records the metafor version; the companion
`results/development/stephenson_predictive_environment.json` also records Python,
NumPy, and SciPy versions. The optional external dependency can be installed
in a project-local library without changing the system R library:

```sh
mkdir -p .r-library
export COUPLING_R_LIBRARY="$PWD/.r-library"
Rscript -e '.libPaths(c(Sys.getenv("COUPLING_R_LIBRARY"), .libPaths())); install.packages(c("metafor", "lme4", "BiasedUrn"), repos="https://cloud.r-project.org")'
python3 -m experiments.development.reanalyze_stephenson_prediction \
  --tables data/development/stephenson_sufficient_tables.npz \
  --cache .cache/stephenson-exact-conditional \
  --output results/development/stephenson_predictive_rerun.json
```

Run from the repository root with its Python dependencies installed. A fresh R
installation may provide a newer package version; compare that recorded version
before interpreting numerical differences. The runner refuses to overwrite a
completed output or reuse a cache with different source tables or R code. It
fits four independent pair batches and retains numerical failures explicitly.

The published result can also be checked without R or any data download:

```sh
python3 -m pytest -q tests/test_stephenson_predictive_result.py
```

This checks complete fit and donor coverage, recomputes losses from the retained
pair scores, verifies the unchanged original analysis, and checks archived source
hashes. The `assay_qc_54` summary excludes C-8914 and C-8939 from the same predictions;
it does not refit sources or reselect transport parameters.
