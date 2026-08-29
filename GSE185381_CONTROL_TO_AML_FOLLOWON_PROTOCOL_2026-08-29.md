# GSE185381 control-to-AML follow-on protocol

Frozen 29 August 2026 from GEO metadata, gene feature lists, and antibody
feature references, before any processed RNA or ADT matrix was opened.

## Status and scope

GSE185381 is a post-GSE202150 follow-on dataset. It was selected after the
GSE202150 outcome was known and is not an independent replication of the
candidate-selection process. This branch receives directional alpha `0.0125`.
No error-rate claim is made for the broader historical dataset search. A miss,
refusal, interruption, or exception closes the branch; there is no adaptive
replacement.

The estimand is composition-inclusive transfer of RNA-ADT coupling from healthy
bone-marrow controls to AML bone marrow. It is not a cell-type-conditional or
causal effect.

## Frozen cohort

GEO contains 46 samples with paired processed CITE-seq RNA, ADT, and cell
metadata. Their metadata contain 10 controls and 41 AML or coded patient donors.
Each donor is restricted to its single pool with the largest publisher-filtered
cell count; an exact tie is broken by GSM accession. Cells are never aggregated
across pools.

All controls exceed the 384-cell budget. `PAWWEE` (184 cells) and `AML022`
(360 cells) fail it in their largest single pools and are excluded. `AML3266`
is RNA-only and is ineligible. The frozen analysis therefore has 10 source and
39 held donors. The 39 held donors occupy 13 run-date acquisition clusters;
GSM lanes from the same run date share one cluster.

The source split is pool-disjoint:

| Role | Selected pool component | Donors |
|---|---|---|
| calibration | GSM5613750 | Control2 |
| calibration | GSM5613756 | Control0004 |
| calibration | GSM5613757 | Control0058 |
| calibration | GSM5613769 | Control0082 |
| calibration | GSM5613775 | Control4003 |
| validation | GSM5613748 | Control1, Control3, Control4 |
| validation | GSM5613751 | Control5 |
| validation | GSM5613787 | Control0005 |

The candidate designation binds every donor, selected pool, metadata cell
count, acquisition cluster, pool file, and source component.

## Panel and states

The frozen cognates are `CD3D/CD3`, `CD8A/CD8`, `MS4A1/CD20`, `CD19/CD19`,
`CD14/CD14`, `ITGAM/CD11b`, `ITGAX/CD11c`, `CD33/CD33`, `NCAM1/CD56`,
`KLRB1/CD161`, `FCGR3A/CD16`, `CD38/CD38`, `IL3RA/CD123`, `CSF1R/CD115`,
`CD69/CD69`, and `TFRC/CD71`. Every RNA symbol occurs in all 46 public gene
feature lists. Each ADT maps to one invariant oligo sequence present in all 46
antibody feature references. The candidate records every observed exact alias.
An authorized assay stage refuses if a modality does not resolve each marker
exactly once.

For each donor, the 384 smallest salted hashes select cells from the frozen
pool. An RNA axis is valid when within-donor detection prevalence lies in
`[0.05,0.95]`. An ADT axis is valid when it has at least two distinct processed
ADT values and its largest equal-value fraction is at most `0.90`. RNA state is
processed count greater than zero. A valid ADT axis uses the deterministic
within-donor midrank: 192 low and 192 high cells, with salted hash and cell
identifier breaking value ties. An ordered pair is valid exactly when its RNA
row and ADT column are valid. Every source and held donor must retain at least
9 valid RNA axes, 9 valid ADT axes, and 128 of 256 ordered pairs before
prediction or scoring; otherwise the stage terminates without attrition or
threshold revision. The estimand is the donor-supported RNA-by-ADT submap, and
loss is averaged over each donor's informative ordered pairs.

## Source fitting

The primary estimator is a graph-regularized exact-fixed-margin hierarchical
coupling field. Separate two-nearest-neighbor marker graphs are derived from
calibration RNA and ADT profiles. Their product incidence regularizes the
ordered coupling field. The frozen grid crosses heterogeneity penalty
`{0.1,1,10}`, ridge penalty `{0.01,0.1,1}`, graph penalty `{0.1,1}`, and
transport multiplier `{0.5,0.75,1}`. Selection minimizes component-equal
deviance across the three validation pools, with lexicographic tie breaking.
The selected configuration is refit on all ten source donors. A separate
ablation fixes graph penalty to zero and cannot become the primary estimator.

The locked classical comparison includes:

1. the untuned row-plus-column Poisson-independence signed-root-deviance
   residual at multiplier one;
2. its source-calibrated transport version;
3. a common-effect stratified conditional maximum-likelihood interaction;
4. a donor-pooled saturated 2-by-2 Poisson log-linear interaction; and
5. a Paule-Mandel random-effects log-odds interaction.

The estimable family with minimum component-equal validation deviance is locked
before held access. The untuned Poisson residual is always reported as the
literal classical head-to-head, even if another family locks. Target-margin
independence is a descriptive nontransport control. Estimation failures remain
visible through deterministic method-specific support or numerical codes.

## Held separation

Only official metadata and feature schemas were inspected before this freeze.
RNA and ADT file SHA-256 values are intentionally unknown. Source-selected RNA
and ADT files are downloaded and hashed only after the public source-stage
claim. Files required only by held-selected pools are first downloaded and
hashed in the held RNA or ADT stage. For three source/held-overlap pools, the
physical RNA and ADT files are downloaded and hashed in source, but only frozen
source columns are converted; held columns are first converted in their held
modality stage. Official byte size and URL are frozen now.

Three source pools also contain unselected held-donor columns:
`2019-08-29-count-1`, `2019-10-25-count-5`, and `2020-03-18-count-1`. The source
stage may convert only frozen source-column indices. This is disclosed
co-residence, not a file-level blind.

Held RNA and ADT are reduced in separate stages. Public prediction may use held
RNA margins and the fixed 192/192 ADT margins, but not held ADT states or joint
tables. The prediction artifact must be public before score authorization.
Scoring is the first stage allowed to join held RNA and ADT states.

## Primary gate

Loss is donor-equal multinomial deviance per cell over informative ordered
pairs. The paired contrast is primary minus the source-locked strongest
classical comparator. A pass requires all of the following:

1. at least 5% lower mean loss;
2. a 20,000-draw acquisition-cluster-stratified paired bootstrap whose 98.75th
   percentile is below zero;
3. at least 32 of 39 favorable donors;
4. one-sided exact donor sign-test `p <= 0.0125`;
5. one-sided exact acquisition-cluster sign-flip `p <= 0.0125`; and
6. a negative mean contrast after removing each acquisition cluster in turn.

The confirmatory chain is fixed before access: primary versus the locked
classical comparator, primary versus graph-zero after a primary pass, then
primary versus the destroyed-link control after a graph-structure pass. This
serial order preserves the directional `0.0125` level. Untuned Poisson,
target-margin independence, and nonlocked classical contrasts are descriptive.
Missing a gate after scoring is a completed negative result, not a data-quality
refusal.

## Public execution

The public sequence is protocol tag, source attempt and result, RNA attempt and
result, ADT attempt and result, prediction attempt and public predictions,
score authorization, score attempt, and result. Every numeric stage has one
exclusive execution claim. Support failure, numerical failure, interruption,
and unexpected exception consume the attempt.

Each assay stage opens an append-only public access journal before download.
The attempt tag binds its empty header; each assay file is recorded immediately
with observed bytes and SHA-256; the completion tag binds the full journal in
both successful and terminal outcomes.

Before the protocol tag, tests must prove that the runner accepts exactly 39
held donors, enforces each `selected_pool_id`, keeps assay hashes deferred until
their authorized stages, uses component-equal source validation, and applies
the `0.0125` campaign thresholds. A protocol tag is invalid until all five
conditions hold.
