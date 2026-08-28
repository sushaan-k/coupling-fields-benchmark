# BMMC held-donor confirmation protocol

Version 1.0, dated 28 August 2026. This document fixes the experiment before
any feature-level value from a held donor is indexed or decoded. The candidate
is the NeurIPS 2021 BMMC RNA--ADT dataset (GSE194122) distributed with SCMMIB.

## Confirmatory question

The experiment asks whether a dependence model learned from paired RNA and ADT
measurements in three non-held physical donors predicts raw binary RNA--ADT
tables in six donor-disjoint recipients better than matched classical
interaction-residual transfer. Prediction is conditional on each recipient's
RNA and ADT margins, so the endpoint is cross-modal dependence rather than
single-modality abundance.

The primary estimator is the audited heterogeneity-aware exact conditional
model in `mapreg/hierarchical_conditional_coupling.py`. It estimates donor-level
full log odds around a graph-regularized population effect by exact
fixed-margin noncentral-hypergeometric likelihood. The recipient table is the
exact noncentral-hypergeometric expectation at the observed recipient margins.
No pseudocount enters the observed or predicted table.

## Immutable allocation

The split is the donor-powered allocation in
`results/development/scmmib_bmmc_metadata_preflight.json`:

| Role | Physical donors | Sites | p10 cells |
|---|---|---|---:|
| Fit | 11466, 19593 | 3, 4 | 1,540 |
| Development | 15078 | 1, 2, 3, 4 | 3,067 |
| Held | 10886, 12710, 13272, 16710, 18303, 28045 | 1, 2, 3, 4 | 4,419 |

The three roles are physical-donor-disjoint. They are not site-disjoint: every
role overlaps another role in site. Site is therefore a measured nuisance, not
an independent replication axis. The six physical donors are the only units of
held inference. The deposited `is_train` field is ignored in every phase.

This is a backup confirmation, not a second chance after inspecting another
held result. `predict` and `score` are disabled if a Sanger score-attempt,
result, or refusal artifact exists. If Sanger is ever scored, BMMC cannot be
reported as an unadjusted replacement confirmation; using it would require a
new public protocol with a declared family-wise multiplicity adjustment.

Donor 15078 is the sole four-site bridge donor. Candidate settings are fitted
on donors 11466 and 19593 and evaluated on its four site batches, with batches
weighted equally. These four batches support model selection but do not create
four biological replicates. After selection, the model is refitted once on the
three non-held physical donors; the bridge donor contributes one aggregated
donor table. The six held donors are then scored once.

## Locked panel and binary tables

The biology-only marker axis is, in this order, `CD4`, `CD7`, `CD14`, `CD19`,
`CD33`, `CD38`, `CD44`, `CD47`, `CD52`, and `CD93`. Every ordered RNA-marker by
ADT-marker combination is retained, giving 100 entities. The panel was fixed
from feature names before matrix access.

RNA is one for a positive raw UMI count. ADT is dichotomized separately within
each physical-donor-by-batch stratum at its deterministic mid-rank: values are
ordered by count and ties by SHA-256 of `BMMC-ADT-v1`, donor, batch, barcode,
and marker. Exactly `floor(n/2)` cells receive state zero. This operation uses
one modality at a time and cannot inspect an RNA--ADT pairing. Tables are formed
only after the two modality states have been fixed.

A donor/entity is informative when its fixed margins admit at least two 2x2
tables. All 100 entities remain in the frozen panel. Losses use the common
margin-only informative mask for every method. A held donor is scorable only if
at least 80 of 100 entities are informative; otherwise the terminal result is a
refusal.

## Development and frozen methods

RNA and ADT marker hypergraphs are built only from the two fit donors. For each
marker and modality, the profile is its binary prevalence in every observed
fit-donor-by-broad-lineage stratum. Each candidate hyperedge contains a marker
and its `k` nearest markers by Euclidean profile distance, with lexical tie
breaking. Development selects `k` from 1, 2, and 3; heterogeneity penalty from
0.1, 1, and 10; ridge penalty from 0.01 and 0.1; graph penalty from 0.1, 0.3,
and 1; and transport multiplier from 0.75, 1, and 1.25. Selection minimizes
batch-equal bridge-donor deviance, then uses the lexicographically smallest
setting. Every fit must pass the estimator's gradient and conditioning
certificates. Software tests for the hierarchical estimator must pass before a
prediction artifact can be written.

The frozen report contains these methods:

- the selected heterogeneity-aware graph model;
- the same hierarchical model with graph penalty zero (ridge-only control);
- a common-effect exact conditional fit with the selected graph and penalties;
- a common-effect exact ridge-only fit;
- the selected model refitted after deterministic within-fit-donor-by-lineage
  ADT-row permutations that preserve all binary margins (destroyed-link
  control);
- the selected model with independently label-permuted RNA and ADT hypergraphs;
- independence;
- the strongest matched classical residual transfer.

The classical family is selected on development from signed Pearson and signed
Poisson-deviance interaction statistics, each raw or exactly centered under its
fixed-margin hypergeometric null, and transport multipliers 0.75, 1, and 1.25.
For source donor size `n`, its coordinate is divided by `sqrt(n)` before
donor-equal pooling. For recipient size `m`, the frozen coordinate is multiplied
by `sqrt(m)` and inverted at the recipient margins; the exact recipient null
mean is added only for a centered candidate. This is the head-to-head standard
log-linear/Poisson interaction-residual comparison.

## Outcome-access boundary

`experiments/confirm_scmmib_bmmc.py` separates `predict` from `score`.

`predict` remains disabled until
`data/confirmation/scmmib_bmmc/source_manifest_v1.json` binds the official
624,797,386-byte complete CITE H5AD, its internal RNA and ADT count/axis paths,
the p10 metadata, and all SHA-256 values. The source object is S3 bucket
`openproblems-bio`, key
`public/phase2-private-data/common/openproblems_bmmc_complete/openproblems_bmmc_cite_complete.h5ad`,
version `kyN5dZPIsYJ0NC8Y5ECK55TuiebegCII`, with multipart ETag
`15d0db3fb12efb77160e293cdeb98e11-75`. An unversioned object is inadmissible.
It may read metadata, feature and barcode axes, opaque bytes for hashes, and
top-level assay rows belonging to the fit and development donors. Selective
matrix reads receive an explicit forbidden-row vector containing every held
barcode. They cannot read a span crossing a held row, open `raw/X`, or form a
held margin or table. `predict` writes the complete development record, frozen
source model, zero-held-access audit, and a disabled score-authorization
template. Its output is not a public freeze until committed at an immutable
public Git hash.

`score` requires an authorization with status `OUTCOME_ACCESS_AUTHORIZED` that
binds the exact prediction, runner, protocol, source manifest, and development
result SHA-256 values plus a 40-character commit and immutable GitHub blob URL
for the prediction. It writes a terminal attempt marker before the first held
matrix read. An existing attempt, refusal, or result prohibits any rerun. Any
post-marker exception creates a terminal refusal.

After authorization, held RNA and ADT states and margins are computed
separately. Predictions from those margins are materialized before the paired
truth tables are formed once. No choice is reselected.

## Inference and decision rule

The primary loss is multinomial deviance per cell, averaged over informative
entities within donor and then equally over the six held donors. Each contrast
is the paired donor loss difference, primary minus comparator. The report gives
mean loss, relative reduction, favorable-donor count, a 20,000-sample paired
donor bootstrap interval, the exact one-sided sign-flip p-value, and the exact
one-sided binomial sign-test p-value. Nothing is refitted in the bootstrap.

Confirmation passes only if all conditions hold:

1. Mean primary deviance is at least 5% below the frozen strongest classical
   residual, and the paired 95% six-donor bootstrap interval is below zero.
2. All six donors favor the primary over that residual. The corresponding
   one-sided exact sign test is `1/64 = 0.015625`.
3. Against the destroyed-link, hierarchical ridge-only, and common-effect
   graph controls, mean deviance is at least 5% lower, the paired 95% interval
   is below zero, and at least five of six donors favor the primary.
4. Source integrity, public authorization, held-access order, fixed margins,
   support, raw-count, finiteness, gradient, and condition-number gates pass.
5. No Sanger held-score attempt, result, or refusal exists.

The label-permuted graph, common-effect ridge-only fit, independence, exact
sign-flip tests, and correlations are reported diagnostics, not additional
gates. A failed or refused result is the permanent held-donor outcome.

## Present status

Only the metadata and one sample's feature names have been decoded. The complete
CITE H5AD has not been downloaded; its SHA-256 and internal paired-assay paths
have not been audited. The 1.77 GB SCMMIB archive is not required by this
protocol and also remains undownloaded. This protocol can be committed now, but
a prediction artifact cannot yet be produced or publicly frozen, and outcome
access cannot be authorized.
