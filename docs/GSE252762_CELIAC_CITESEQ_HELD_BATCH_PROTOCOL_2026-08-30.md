# GSE252762 celiac intestinal CITE-seq held-batch donor protocol v1

## Question and status

This pre-outcome public-data holdout tests whether an exact conditional
coupling field learned from 16 intestinal-biopsy donors predicts RNA--surface
protein dependence in 13 donor-disjoint biopsies from an untouched sequencing
batch. The recipient RNA margins are opened and predictions are published
before the recipient CITE matrix is accessed. Physical donors are the
inferential units.

The metadata-only preflight used official GEO metadata, barcode, and feature
files. It did not request an RNA or CITE count matrix. The deposited metadata
contain RNA and CITE quality-control columns; the preflight uses only barcode,
batch, location, sample, and disease-state labels. The frozen selected-cell
columns make later selection independent of matrix values.

## Cohort

Eligible units are duodenal-biopsy samples from active celiac disease (ACD),
gluten-free diet (GFD), or healthy control donors with at least 256 paired
cells. Potential celiac disease samples, gluten-challenge blood samples, and
samples below the cell floor are excluded. Each eligible deposited sample is
one donor unit.

Nine source donors form calibration, seven form a disjoint pilot, and all 13
eligible batch-6 donors form the held panel. Calibration and pilot identities
are fixed in `metadata_preflight_v1.json`; the split contains ACD, GFD, and
control donors in both stages. Batch 6 is excluded from every selection, fit,
and gate before held prediction. The design is therefore donor-disjoint and
file-separated, with processing batch intentionally aligned to the held
cohort.

## States and entities

For each donor, 256 cells are selected by SHA-256 rank of the deposited barcode,
donor identifier, and `GSE252762-CELIAC-CELL-v1`. RNA state is detection of at
least one raw UMI. Within each donor and protein, exactly 128 cells are assigned
the high state by descending raw CITE count with a fixed hash tie-break.

The nine frozen RNA--protein marker correspondences are CD3D--CD3, CD4--CD4,
CD8A--CD8, CD27--CD27, CD38--CD38, CD44--CD44, CD69--CD69, ITGAE--CD103, and
KLRB1--CD161. Every donor supplies all 81 ordered RNA-marker by protein-marker
binary tables. A complete-profile half-cycle permutation of the CITE states
within donor supplies the destroyed-link control without changing a protein
margin.

## Primary estimator

The primary model is the context-conditioned exact conditional coupling field.
For each marker pair, the log odds contain a CELIAC or CONTROL population
coefficient and a donor deviation. The likelihood conditions on the two margins
of every donor table. Positive coefficient and donor penalties give a unique
finite fit. A product-graph penalty links marker pairs that share their RNA or
protein marker.

Calibration leave-one-donor-out deviance selects donor penalty in
`{0.1, 1, 10}`, product-graph penalty in `{0, 0.05, 0.2}`, and transport
multiplier in `{0.75, 1}`. The coefficient ridge is 0.01. Ties use the listed
numeric order. Each fold and final fit must pass the fixed gradient and
condition-number certificates.

## Classical comparisons

Two classical predictors receive the same source donors, CELIAC/CONTROL label,
recipient margins, 81 entities, and calibration folds.

1. Signed-root Poisson-deviance interaction residuals are averaged within
   context and inverted at recipient margins. Calibration selects a transport
   multiplier from `{0.5, 0.75, 1, 1.25}`.
2. A saturated fixed-interaction Poisson/log-linear predictor uses the
   context-pooled log odds after adding one-half to each pooled cell. This
   Jeffreys/Haldane stabilization is fixed in advance so a boundary maximum does
   not abort the experiment. Calibration selects the same multiplier grid.

The unpenalized profiled Poisson interaction is also fitted and reported for
each context--entity as `FINITE`, `BOUNDARY`, or `NO_INFORMATION`; its boundary
status cannot terminate the campaign. Independence is a diagnostic. The
strongest classical comparator is the classical family with the lower
calibration leave-one-donor-out loss and is fixed before pilot evaluation.

## Pilot promotion

The selected models are fitted on the nine calibration donors and evaluated
once on the seven pilot donors. Promotion requires:

1. lower sample-equal mean deviance for the primary than each classical
   predictor and the destroyed-link control;
2. lower primary loss in at least five of seven donors relative to the frozen
   strongest classical predictor and in at least five relative to destroyed
   links; and
3. at least 5% lower mean deviance than destroyed links.

A failure is terminal. A pass permits one refit using all 16 source donors and
authorizes held RNA access.

## Held prediction and decision

The prediction stage reads the batch-6 RNA matrix but not its CITE matrix. The
fixed protein-high margin is 128 of 256 cells, so recipient joint tables can be
predicted from RNA margins alone. Expected tables for every method, sample, and
entity are published with checksums before CITE access. A separate public score
authorization binds those bytes.

The score stage reads the batch-6 CITE matrix once and forms the 13 held truth
panels. Loss is multinomial deviance per cell, averaged over 81 entities within
donor and then equally across donors. Paired 95% intervals use 20,000 bootstrap
draws stratified by ACD, GFD, and control with seed 25276201. Exact one-sided
sign tests use donor-level loss differences.

Confirmation requires:

1. lower primary mean loss than both classical predictors;
2. an upper paired-bootstrap endpoint below zero against the frozen strongest
   classical predictor;
3. lower loss than that predictor in at least 10 of 13 donors and one-sided
   sign-test probability at most 0.05; and
4. at least 5% lower mean loss than destroyed links, with an upper paired
   bootstrap endpoint below zero.

ACD, GFD, and control means are reported without separate pass thresholds. The
first complete held result is terminal. No marker, sample, state, model grid,
comparator, seed, or criterion may change after CITE access.

## Public sequence

The immutable order is candidate and metadata preflight, implementation,
source attempt and terminal pilot result, held-RNA authorization and frozen
predictions, score authorization, and terminal held result. Every stage is
committed, annotated-tagged, pushed, and verified against the public remote
before the next protected matrix is requested. This is a publicly frozen
public-data holdout, not a registration in an external registry.
