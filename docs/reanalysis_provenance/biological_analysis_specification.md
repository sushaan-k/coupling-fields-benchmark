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

All other pairs still use the official exact-conditional random-effects ML fit
at two integration tolerances. A failure on non-identical tables is not evidence
for a zero variance and is not replaced with an equal-effect estimate. Preserve
those failures and withhold the full-panel random-effects comparison when any
pair is unavailable.

If every informative donor observation is at the same lower or upper feasible
endpoint, the likelihood instead has an extended maximum as the population log
odds tend to minus or plus infinity. Record
`fit_route=extended_common_support_endpoint`, the endpoint direction, and an
unidentified heterogeneity variance. Prediction is the corresponding recipient
endpoint table, with a point-mass count interval. An incompatible recipient
table has infinite deviance; retain it explicitly, without pseudocounts, finite
coefficient clipping, or a bootstrap over only finite donor losses. Mixed source
endpoints do not qualify for this route.
