# GSE217494 cardiac CITE-seq implementation clarifications v1

Frozen at `2026-08-30T08:49:39Z`, before any GSE217494 `matrix.mtx.gz`
body, Matrix Market header, or numeric entry was requested or read. These
clarifications resolve implementation details without changing the split,
estimand, model grid, comparators, source promotion rule, held confirmation
rule, or one-shot access boundary in the v1 protocol.

## Destroyed-link refits

Every destroyed-link cross-validation fold repeats marker eligibility,
ranking, RNA and ADT graph construction, and model fitting after the complete
training-heart ADT vectors have been shifted. The final destroyed-link fit
does the same on all source hearts. Complete-vector permutation preserves the
within-heart raw ADT distribution and its mean profile, so the selected marker
axis and both graphs must equal their unshifted counterparts. The runner
checks and reports this invariance; it does not assume it by reusing an
unshifted selection. Validation and held tables remain unshifted.

## Boundary-ineligible classical fit

The exact common-effect conditional field is eligible for the strongest
remaining classical comparator only when every selected entity has a finite
unpenalized source estimate in every fold and in the final fit. A boundary
estimate makes that method ineligible and is reported. The standardized
fixed-margin Pearson residual and independence remain required candidates.
The pooled and etiology-specific Poisson comparators remain mandatory, so a
nonfinite interaction in either Poisson fit is a terminal source refusal.

## Degenerate recipient margins

For a validation or held table with a degenerate row or column margin, every
method returns the unique margin-compatible table. Such a table has zero
deviance when observed and predicted margins agree. A Poisson odds ratio is
undefined at those margins; the implementation records an informative-margin
mask, applies the odds-ratio reconstruction certificate only where the mask is
true, and applies the row and column margin certificates everywhere. Undefined
reconstructed odds ratios are serialized as JSON `null`, never as a finite
certificate or a nonstandard `NaN` token.

## Secondary modules

An evaluable three-marker module uses `min(3, m - 1)` neighbors; the primary
axis continues to use exactly three neighbors. The structured primary is
refit on each complete within-module ordered pair set with its source-selected
primary hyperparameters and no retuning. The destroyed-link field is refit in
the same way with its own source-selected hyperparameters. Poisson, Pearson,
exact-common, and independence fields are fit entity by entity, so their
already-frozen estimates are subset to the same module pairs; this is
algebraically identical to refitting them on that subset.

## Acquisition and serialization

The acquisition layer independently counts and hashes the exact compressed
bytes received by each single HTTP GET. The Matrix Market reducer separately
hashes the fully exhausted decompressed stream and reports its parsed entry
count, duplicate accumulation, selected block, and gzip validation state.
Compressed and decompressed hashes are distinct certificates. Public JSON is
written with finite-number enforcement and contains no local scratch path,
private claim token, executable path, or raw exception text.

The bound assay runtime is CPython 3.9.6 on Darwin arm64 with NumPy 2.0.2 and
SciPy 1.13.1. `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS`, and `NUMEXPR_NUM_THREADS` must each equal `1` before
an attempt is claimed or consumed.
