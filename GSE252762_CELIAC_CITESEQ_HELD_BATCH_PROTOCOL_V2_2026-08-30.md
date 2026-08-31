# GSE252762 celiac intestinal CITE-seq held-batch donor protocol v2

## Supersession and status

This protocol supersedes v1 before any RNA or CITE count-matrix request. A
pre-access methodological audit found that v1 did not require improvement over
recipient-margin independence and did not make its donor-stratified Poisson
fit an always-defined gate comparator. V2 corrects those defects without
changing any accession, donor, batch, cell, marker, state, context, split, or
access rule. V1 remains public as an immutable audit record.

The metadata-only preflight used official GEO metadata, barcode, and feature
files. It made zero numeric matrix requests. The deposited metadata contain RNA
and CITE quality-control columns; the preflight uses only barcode, batch,
location, sample, and disease-state labels. Frozen cell columns make subsequent
selection independent of matrix values.

## Question and cohort

The experiment tests whether a conditional coupling field learned from 16
intestinal-biopsy donors predicts RNA--surface-protein dependence in 13
donor-disjoint biopsies from an untouched sequencing batch. Recipient RNA
margins are opened and predictions are published before recipient CITE access.
Each deposited biopsy sample, described by the source study as a patient
sample, is treated as one donor-level inferential unit.

Eligible units are duodenal-biopsy samples from active celiac disease (ACD),
gluten-free diet (GFD), or healthy controls with at least 256 paired cells.
Potential celiac disease, gluten-challenge blood, and samples below the cell
floor are excluded. Nine source donors form calibration, seven form a disjoint
pilot, and all 13 eligible batch-6 donors form the held panel. Calibration and
pilot identities are fixed in the v1 metadata preflight. Batch 6 is excluded
from every selection, fit, and gate before held prediction. Processing batch is
therefore intentionally aligned to the held cohort.

## States and entities

For each donor, 256 cells are selected by SHA-256 rank of the deposited barcode,
donor identifier, and `GSE252762-CELIAC-CELL-v1`. RNA state is detection of at
least one raw UMI. Within each donor and protein, exactly 128 cells are assigned
the high state by descending raw CITE count with the frozen hash tie-break.
Cell-selection hashes encode
`GSE252762-CELIAC-CELL-v1\0sample_id\0barcode`; ties use barcode order. ADT
tie hashes encode
`GSE252762-CELIAC-ADT-TIE-v1\0sample_id\0protein\0barcode`; cells rank by
descending count, ascending digest, then barcode.

The marker correspondences are CD3D--CD3, CD4--CD4, CD8A--CD8, CD27--CD27,
CD38--CD38, CD44--CD44, CD69--CD69, ITGAE--CD103, and KLRB1--CD161. Each donor
supplies all 81 ordered RNA-marker by protein-marker binary tables. A frozen
complete-profile half-cycle permutation of CITE states within donor supplies
the destroyed-link control without changing any margin. It orders cells by
ascending SHA-256 of
`GSE252762-CELIAC-DESTROY-v1\0sample_id\0barcode`, then barcode, and assigns
the complete CITE-state vector at ordered position `(j + 128) mod 256` to
ordered position `j`.

## Primary estimator

The primary estimator is a context-conditioned exact conditional coupling
field. For each marker pair, log odds comprise a CELIAC or CONTROL population
coefficient and a donor deviation. The likelihood conditions on both margins
of every donor table. Positive coefficient and donor penalties give a unique
finite fit. A product-graph penalty links pairs that share their RNA or protein
marker.

For entity `e`, let `D[e]` be donors whose upper-left-count Frechet interval has
positive width,
`theta[d,e] = beta[context[d],e] + u[d,e]`, and `L` be the product-graph
Laplacian. The minimized objective is the sum over entities of the exact
conditional negative log likelihood divided by `|D[e]|`, the donor penalty
`lambda sum_{d in D[e]} u[d,e]^2 / (2 |D[e]|)`, plus
`0.01 ||beta||^2 / 2` and
`gamma sum_c beta[c]' L beta[c] / 2`. The likelihood is offset by its value at
zero without changing the minimizer. `L` is the Kronecker-sum normalized
Laplacian of one complete RNA-marker hyperedge and one complete protein-marker
hyperedge, rescaled to mean diagonal one: its diagonal is 1, its value is
`-1/16` between distinct pairs sharing exactly one marker, and it is zero
otherwise. Thus duplicating the complete donor panel leaves the estimator
unchanged.

Calibration leave-one-donor-out deviance selects donor penalty from
`{0.1, 1, 10}`, product-graph penalty from `{0, 0.05, 0.2}`, and transport
multiplier from `{0, 0.75, 1}`. The coefficient ridge is 0.01. Ties use the
listed order, with the null transport first. Every fold and final fit must pass
at least two informative donors per entity, scaled gradient at most `1e-8`,
coefficient-Schur and donor-curvature condition numbers at most `1e14`, and a
200-Newton-iteration limit. Failure of any certificate makes that grid point
incomplete in leave-one-donor-out selection; no complete grid point makes the
stage terminal. After selection, failure of the calibration, destroyed-link,
or all-16-donor refit is terminal, with no fallback or reselection. Recipient
prediction uses `alpha beta[context,e]` and no donor deviation. The
destroyed-link control reuses the real-data-selected donor penalty, graph
penalty, coefficient ridge, and transport multiplier; it is never tuned
separately.

Primary and destroyed-link predictions use the exact conditional mean, not a
continuous Poisson table. At integer recipient margins with feasible
upper-left support `x = lower,...,upper`, mass is proportional to
`choose(column0,x) choose(column1,row0-x) exp(alpha beta[context,e] x)`.
The predicted upper-left count is the expectation of `x`; the other three
expected counts are fixed by the margins.

## Matched Poisson comparator

The principal classical comparator is a donor-stratified Poisson log-linear
model. For donor `d`, binary RNA state `r`, binary protein state `p`, context
`c`, and marker pair `e`,

`log(mu[d,e,r,p]) = a[d,e] + b[d,e] r + g[d,e] p + theta[c,e] r p`.

The donor-specific nuisance effects are profiled so each fitted donor table has
its observed row and column margins. A table is informative when the Frechet
interval for its upper-left count has positive width. The context--entity
interaction minimizes the mean profiled negative log likelihood over
informative donor tables plus `0.01 theta^2 / 2`. Equivalently, it is the
unique root of `mean_d(n[d,e,0,0] - mu[d,e,0,0](theta)) - 0.01 theta`.
This fixed weak ridge makes the standard donor-fixed-effects interaction finite
under separation without pooling donors. A
context--entity with no informative donor has `theta = 0` and status
`NO_INFORMATION`. Zero-information tables are excluded from that mean. All
other fits must have a bracket within `[-16, 16]`, scaled penalized-score
residual `abs(F) / max(1, mean Frechet width)` at most `1e-10`, row/column-
margin and reconstructed-log-odds errors at most `1e-8`, and penalized
information, defined as mean data information plus 0.01, at least 0.01. Root
finding uses Brent's
method with absolute tolerance `1e-12`, relative tolerance four times machine
epsilon, and at most 256 iterations. Failure of any certificate is terminal.

The fitted interaction is transported as `alpha theta` to recipient margins.
The four expected counts are the unique table with those margins and
transported log odds. A degenerate recipient margin returns its unique table.
Calibration selects `alpha` from `{0, 0.5, 0.75, 1, 1.25}`; exact ties prefer
that listed order.

## Additional comparisons

The same source donors, contexts, recipient margins, entities, and calibration
folds are given to two additional classical predictors:

1. For each context--entity, donor tables are summed cellwise to `N`; the
   pooled Haldane field is
   `log((N00 + 0.5)(N11 + 0.5) / ((N01 + 0.5)(N10 + 0.5)))`.
   `alpha` times this field is reconstructed at recipient margins by the same
   unique-log-odds table used for the donor-stratified Poisson predictor.
2. For donor table `n`, independence expectation is
   `E = outer(row(n), column(n)) / sum(n)`. Its signed-root coordinate is
   `sign(n00 n11 - n01 n10) sqrt(2 sum_{n>0} n log(n/E))`. Coordinates are
   averaged equally over donors within context, multiplied by `alpha`, and
   inverted at recipient margins to the unique table with that signed-root
   coordinate. For a positive-width feasible interval, inversion clips to
   `[lower + epsilon, upper - epsilon]`, where
   `epsilon = min(1e-8, (upper - lower)/4)`, and uses 96 bisections. A
   zero-width interval returns its unique table.

Each selects transport from `{0, 0.5, 0.75, 1, 1.25}`. Independence is the
fixed zero-interaction predictor. Exact alpha ties prefer the listed order.
The unpenalized donor-profiled Poisson
interaction is reported per context--entity as `FINITE`, `BOUNDARY`, or
`NO_INFORMATION` and never substituted for a failed fit.

The strongest benchmark is selected on calibration only from independence,
donor-stratified ridge Poisson, pooled Haldane Poisson, and signed deviance.
It is the family with lowest leave-one-donor-out mean loss after its transport
selection. Exact ties prefer independence, donor-stratified ridge Poisson,
pooled Haldane Poisson, then signed deviance.

## Pilot promotion

Models selected on the nine calibration donors are evaluated once on the seven
pilot donors. Promotion requires all of the following:

1. lower primary mean deviance than independence, all three classical
   predictors, and destroyed links;
2. lower primary loss in at least five of seven donors versus independence,
   donor-stratified ridge Poisson, the calibration-frozen strongest benchmark,
   and destroyed links; and
3. at least 5% lower primary mean deviance than independence,
   donor-stratified ridge Poisson, and destroyed links.

Failure is terminal. Passing permits one refit using all 16 source donors and
authorizes held RNA access.

## Held prediction and decision

The prediction stage reads batch-6 RNA but not CITE. Because the protein-high
margin is fixed at 128 of 256 cells, every joint table can be predicted from
RNA margins alone. Predictions for every method, donor, and entity are
published with raw selected-count checkpoints before CITE access. A separate
public score authorization binds those bytes.

The score stage reads batch-6 CITE once and forms the 13 held truth panels.
Loss is multinomial deviance per cell, averaged over 81 entities within donor
and then equally over donors. All paired intervals are percentile intervals
from 20,000 deposited-condition-stratified bootstrap draws at the original
stratum sizes, using seed 25276201 and linear quantiles. Exact one-sided sign
tests discard tolerance-defined ties. Donor differences below `-1e-12`
deviance per cell
are favorable, differences with absolute value at most `1e-12` are ties, and
bootstrap intervals use the unrounded differences. If a required comparator
has exactly zero mean loss, its relative-reduction condition is false and the
stage fails its gate rather than raising an execution error. Relative reduction
is `1 - mean(primary loss) / mean(comparator loss)` in both pilot and held
gates. The `1e-12` favorable/tie rule also applies to pilot checks. NumPy
`default_rng(25276201)` (PCG64) draws integer indices separately in stratum order
ACD, GFD, CONTROL with shape `(20000, original stratum size)`; the same three
index arrays are reused for every comparator. Within each stratum, donors
retain their order in `metadata_preflight_v1.json`.

For independence, donor-stratified ridge Poisson, every additional classical
predictor, destroyed links, and the frozen strongest benchmark, the report
includes mean difference, relative reduction, favorable count, paired 95%
interval, and exact sign result. Confirmation requires every condition below:

1. lower primary mean loss than independence, all three classical predictors,
   and destroyed links;
2. for independence and donor-stratified ridge Poisson separately, a paired
   bootstrap upper endpoint below zero, lower primary loss in at least 10 of 13
   donors, one-sided sign probability at most 0.05, and at least 5% lower mean
   deviance;
3. for the calibration-frozen strongest benchmark, a paired-bootstrap upper
   endpoint below zero, at least 10 of 13 favorable donors, and one-sided sign
   probability at most 0.05; and
4. at least 5% lower mean deviance than destroyed links, with its paired
   bootstrap upper endpoint below zero.

If the strongest benchmark is independence or donor-stratified ridge Poisson,
its numerical comparison is computed once but every semantic criterion remains
binding. ACD, GFD, and control means are descriptive. The first complete held
result is terminal. No marker, sample, state, model grid, comparator, seed, or
criterion may change after source access.

## Public sequence

The immutable order is v2 candidate and metadata preflight, implementation,
source authorization, attempt, consumption, reduction checkpoint and terminal
pilot result, held-RNA authorization, attempt, consumption, reduction
checkpoint and frozen predictions, score authorization, attempt, consumption,
reduction checkpoint, and terminal held result. Every capability boundary is
committed, annotated-tagged, pushed, and verified against the public remote
before the protected request it authorizes. This is a publicly frozen
public-data holdout, not an external registry entry.
