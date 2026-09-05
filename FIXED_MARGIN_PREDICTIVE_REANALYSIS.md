# Predictive reanalysis of the fixed-margin benchmark

Analysis specification, 4 September 2026. This is a post hoc extension: the
Newcastle outcomes and original comparator results have already been observed.
It does not replace or extend the confirmatory status of the original analysis.

Use exactly the original 12 Cambridge calibration donors, 24 Cambridge pilot
donors, 56 Newcastle evaluation donors, nine matched markers, and 512 selected
cells per donor. Do not access unused Cambridge donors or other sealed panels.
Recover integer tables and verify the published donor losses before fitting any
new model. Retain the selected count matrix locally for sensitivity analyses.

## Estimator comparison

Compare the original hierarchical conditional estimator with an unpenalized
common-effect conditional estimator, a common-effect estimator with matched
population ridge penalty, and the classical hypergeometric-normal random-effects
model fitted by exact conditional marginal likelihood. Fit the latter with
metafor's CM.EL maximum-likelihood implementation and retain all numerical
failures. Check integration tolerance rather than accepting an optimizer's
success flag alone.

For each heterogeneous fit compare population plug-in reconstruction against
the conditional predictive mixture. When multiplying log odds by alpha, multiply
their standard deviation by alpha as well. Select alpha from {0.5,0.75,1,1.25}
using calibration fits and Cambridge pilot loss only; refit on all 36 Cambridge
donors. The current hierarchical estimator retains its original selected
penalties. The common-ridge comparator uses population ridge 0.01. An unchanged
population plug-in is always retained as a candidate and reported separately.

Primary descriptive endpoint: donor-equal multinomial deviance per cell on the
original informative pair mask. Also report favorable donors, paired donor
bootstrap intervals (20,000 draws), and predictive count-interval coverage and
width for heterogeneous predictive models. These intervals are conditional on
the source fit; they are not repeated-site uncertainty. Missing or nonconvergent
fits are reported, never removed to manufacture a favorable comparison.

## Biological sensitivities

At original median-rank states, permute complete ADT profiles within each source
donor and annotated cell type, using eight fixed independent seeds. Refit the
original hierarchical configuration. Compare with intact source linkage at the
same held tables. This preserves broad lineage composition but destroys pairing
within the annotation strata. It is an augmented biological diagnostic, not a
matched margins-only comparator, because it uses cell-type annotations.

Apply one additional ADT state rule: raw-count thresholds defined by the pooled
Cambridge median for each marker, then held fixed in Newcastle. These are
source-defined thresholds, not validated protein-positivity thresholds. Freeze
pair eligibility from source support, report every held donor and its informative
pair count, and compare fixed-configuration hierarchy with common conditional
estimation without retuning. Preserve negative and unscorable outcomes.

## Simulation

Extend the original fixed-margin generator by varying recipient heterogeneity
as well as source heterogeneity. Include exact common-effect, current hierarchy,
and oracle plug-in/mixture predictions. The oracle only diagnoses decision-rule
error; it is not a fitted-method performance claim. Retain original margin
conditions, report Monte Carlo uncertainty, and do not select scenarios or seeds
after examining their relative performance.

## Interpretation

Predictive integration and exact conditional random-effects fitting are classical
procedures. Any stronger method claim must follow their matched comparison, not
their renaming. A positive composition control would support transferable
within-stratum association, not molecular binding, causality, or a new module.
New results remain descriptive even if they meet the original numerical success
thresholds.

## Numerical boundary handling

Before the new cohort fits, a synthetic check found that metafor's exact
random-effects optimizer can fail on identical informative donor tables. For
this case alone, use the exact common-effect estimate with variance zero and
record `fit_route=analytic_identical_table_boundary`. This is the global optimum
of the same random-effects likelihood: every donor has the same likelihood
function, and averaging that function over any distribution of log odds cannot
exceed its maximum. Require an interior observed count and an absolute exact
conditional score below 1e-7. Verify the estimate and likelihood independently.
No quadrature comparison is required for this point-mass distribution.

All other pairs use the official exact-conditional random-effects ML fit at two
integration tolerances. A failure on non-identical tables alone is not evidence
for zero variance; numerical recovery requires the additional checks below.
Preserve unresolved failures and withhold the full-panel random-effects
comparison when any pair is unavailable.

If every informative donor observation is at the same lower or upper feasible
endpoint, the likelihood instead has an extended maximum as the population log
odds tend to minus or plus infinity. Record
`fit_route=extended_common_support_endpoint`, the endpoint direction, and an
unidentified heterogeneity variance. Prediction is the corresponding recipient
endpoint table, with a point-mass count interval. An incompatible recipient
table has infinite deviance; retain it explicitly, without pseudocounts, finite
coefficient clipping, or a bootstrap over only finite donor losses. Mixed source
endpoints do not qualify for this route.

The exact likelihood optimizer and integration tolerances are unchanged, but
metafor's ancillary numerical Hessian uses `hessianCtrl=list(r=4)` instead of
the default Richardson depth 16. Its implementation applies this setting to
standard-error calculation after fitting; those standard errors do not enter
prediction, model selection, or inference here. Check fitted parameters and
likelihood against the original setting on the first two calibration pairs.
Retain the original depth-16 partial run separately and refit the full panel
at depth 4 rather than mixing the two numerical records.

Before calculating any new held-query scores, the first calibration fits also
exposed routine near-zero-variance optimization failures. For those failures
or variance estimates below 1e-4, retry the same official likelihood with
`nlminb` at 1,000 iterations and then BFGS at 1,000 iterations. Both retries retain
the two integration tolerances. Select the largest validated likelihood; retain
every attempted optimizer and its result.

A zero-variance candidate on non-identical tables additionally requires a
negative one-sided variance score at the exact common-effect estimate and no
improvement on a fixed variance profile at
{0.0001, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16}.
For each profile point, maximize the same integrated likelihood over the mean
within 12 log-odds units of the common-effect estimate; reject a profile that
reaches that search boundary. The variance score is one-half the sum of squared
conditional scores minus conditional Fisher information. Profile calculations
independently enumerate the hypergeometric support and integrate over a standard
normal variable. Require likelihood agreement within 1e-6 for accepted estimates.
Record this numerical route separately from the analytically established
identical-table boundary; a finite profile is a numerical check, not a proof of
global optimality over every variance. Any unresolved fit remains unavailable.

Before new query scoring, independent debugging identified failures in metafor's
approximate lme4 initializer, before its exact likelihood optimization. If the
three specified optimizer attempts fail and the one-sided variance score does
not qualify for the boundary route, retry the same exact CM.EL fit with
`glmerCtrl=list(nAGQ0initStep=FALSE)` and the larger nlminb budget. This changes
only the initializer, not the final likelihood. Apply the same two integration
tolerances and independent likelihood check. Preserve previously validated fits
with their original script hashes; rerun unresolved pairs and label initializer
retries explicitly. No donor split, penalty, or transport grid changes.

Full-source fitting also exposed infinite offsets in the approximate initializer
for donor tables with a zero row. Pass only tables with non-singleton conditional
support to the official EE and ML calls, recording omitted donor identifiers.
Singleton-support tables contribute a constant likelihood factor of one for all
parameters, so this does not change the exact marginal likelihood or the donor
prediction panel. Retain earlier validated fits and retry affected source pairs.

The first completed scoring run reported failure of the existing common-ridge
Newton optimizer on the calibration panel. Preserve that scoring artifact. For
this failure only, solve the same graph-zero conditional objective by scalar
score roots, retaining the original population penalty 0.01 times the median
positive null Fisher information. Positive ridge makes each score strictly
increasing and its root unique. Require maximum absolute full-objective gradient
below 1e-8 and the original Hessian condition limit. Check against an existing
converged fit. Retain all random-effects fits, recipient predictions, and alpha
choices; perform only the previously unavailable common-ridge pilot selection.
