# GSE144744 held-cohort RNA-ADT coupling protocol

Frozen on 2026-08-29 from the official GEO record, per-cell and per-sample
metadata, the deposited ADT feature and barcode axes, and the published reagent
table. The ADT archive was downloaded for nonnumeric axis inspection, but its
`matrix.mtx` member was not opened. The RNA archive was not downloaded. No
numeric matrix value was decoded before this protocol was frozen.

## Question and scope

This one-shot test asks whether an RNA-to-surface-protein coupling field learned
from untreated relapsing-remitting multiple-sclerosis donors and matched
healthy controls transfers to an untreated primary-progressive cohort and its
separately matched controls. Source donors were assayed with 10x V2 in
`HE-MK-002` through `HE-MK-006`; held donors were assayed with 10x V3 in
`HE-MK-015` through `HE-MK-018`. The official source is
[GSE144744](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE144744),
which contains 497,705 transcriptomes and 355,433 surface-protein profiles
from 62 donors. The estimand is composition-inclusive same-cell RNA-ADT
association, not a causal or cell-type-conditional effect.

This candidate followed unsuccessful public-data branches. The directional
`alpha=0.0125` controls the four frozen comparisons within this candidate; it
does not retrospectively control the historical dataset search. A refusal,
failed gate, interruption, or negative result closes the branch. The complete
candidate ledger remains part of the public benchmark.

## Frozen cohorts and libraries

The source is nine author-matched `MS2`-`HI2` pairs:
`01/02`, `03/04`, `05/06`, `07/08`, `09/10`, `11/12`,
`13/14`, `17/18`, and `21/22`. Each source donor has one ADT-bearing
`sample_10X` library.

Source tuning uses leave-one-`exp_name`-out cross-validation:

| Held source experiment | Matched pairs |
|---|---|
| `HE-MK-002` | `03/04` |
| `HE-MK-003` | `09/10`, `13/14` |
| `HE-MK-004` | `11/12`, `21/22` |
| `HE-MK-005` | `05/06`, `07/08` |
| `HE-MK-006` | `01/02`, `17/18` |

The held cohort is ten author-matched `PPMS`-`HI3` pairs:
`43/44`, `45/46`, `47/48`, `49/50`, `51/52`, `53/54`,
`55/56`, `57/58`, `59/60`, and `61/62`. No held donor contributes
to source locking, graph construction, tuning, or comparator selection.

Held donors have multiple technical libraries. One common suffix per matched
pair is selected from metadata alone. Among suffixes with at least 512
ADT-bearing cells in both donors, the chosen suffix minimizes SHA-256 of
`GSE144744-MS-LIBRARY-v1|left-right|suffix`, then the integer suffix. The
frozen suffixes are `6,4,6,2,1,1,4,3,3,1` in pair order. Counts and exact
sample identifiers are recorded in the machine-readable designation.

Within each selected library, the 512 cells with the smallest SHA-256 values of
`GSE144744-MS-CELL-v1|donor|cell_id` form the analysis sample, with cell ID
breaking an exact digest tie. Eligibility requires an exact ADT-barcode join,
nonmissing deposited `nCount_ADT`, and the frozen cohort, chemistry, group,
experiment, donor, and library contracts. Pooling libraries is forbidden.

## Marker map and support

The candidate universe is every one-gene RNA-protein correspondence in the
deposited 38-antibody panel: 29 pairs listed in the machine-readable
designation. `CD45RA -> PTPRC`, `CD57 -> B3GAT1`, and
`HLA-DR -> HLA-DRA` are excluded because an isoform-specific epitope,
carbohydrate epitope, and conformational heterodimer are not one-gene protein
states. Multi-gene, rearranged-receptor, and ambiguous-antibody features are
also excluded. Symbols and labels must resolve exactly once; fuzzy matching,
duplicate aggregation, substitution, and relabeling are forbidden.

RNA state is raw UMI count greater than zero. An RNA axis is supported when at
least five of 512 cells occupy each binary state. For ADT, a marker must have
at least 5% nonzero counts, positive raw-count interquartile range, and no raw
count shared by more than 90% of cells. Cells are ordered by raw ADT count,
SHA-256 of `GSE144744-MS-ADT-v1|donor|locked_index|cell_id`, then cell
ID; the upper 256 are high and the lower 256 are low.

A candidate marker locks only if both modality checks pass in all 18 source
donors. At least 12 markers must lock. This marginal, source-only rule does not
inspect association outcomes. The entire locked map is then fixed. Every
locked marker must pass both checks in all 20 held donors, and every locked
ordered RNA-by-ADT pair must be scored for every donor. Held marker or donor
attrition is a terminal refusal.

## Estimator and controls

The primary estimator is a graph-regularized, exact-fixed-margin hierarchical
coupling field. For each ordered marker pair it estimates conditional RNA-ADT
log odds, shrinks donor effects toward a population field, and smooths the
field over the product of two marker graphs. Fixed-margin likelihood removes
abundance margins from the fitted association parameter.

The RNA graph uses donor means of
`log1p(10000 * marker_count / nCount_RNA)`; the ADT graph uses donor means of
`log1p(marker_count)`. Standardized Euclidean profiles define separate
two-nearest-neighbor graphs. Both graphs are rebuilt exclusively from the
training donors in every source fold.

The frozen grid crosses heterogeneity `{0.1,1,10}`, ridge
`{0.01,0.1,1}`, positive graph `{0.01,0.1,1}`, and transport
`{0.5,0.75,1,1.25}`. The selected configuration minimizes the equal mean of
the five held-experiment losses, with lexicographic tie breaking. Graph-zero
uses the same procedure with no graph penalty. After the source gate, all
models are refit once on all 18 source donors.

The classical inventory is:

1. untuned row-plus-column Poisson-independence signed-root deviance;
2. source-calibrated Poisson signed-root residual transfer;
3. common-effect stratified conditional maximum likelihood;
4. donor-pooled saturated 2-by-2 Poisson log-linear interaction; and
5. Paule-Mandel random-effects log odds.

Each method receives the same source folds, fixed marker map, tables, margins,
and transport grid where calibration is allowed. The minimum cross-validated
loss among estimable methods 2-5 is the source-locked classical comparator;
the order above breaks exact ties. Untuned Poisson is independently mandatory.
Refused classical methods remain reported and cannot be silently replaced.

The destroyed-link control orders cells by SHA-256 of
`GSE144744-MS-DESTROY-v1|donor|cell_id` and cyclically shifts the complete
ADT state vector by one position relative to RNA. It preserves donor-marker
margins and destroys same-cell pairing. Target-margin independence is
descriptive.

## Source gate

The source cross-validation gate is an adequacy screen, not held-cohort
inference. Against the locked classical method, untuned Poisson, and the
donor-pooled saturated Poisson log-linear interaction, the primary must achieve
at least 5% lower experiment-equal mean deviance, lower
loss in at least 14 of 18 out-of-fold donors, and a negative mean paired
difference in every held source experiment. Failure terminates the branch
before held numeric access.

## Staged held access

Both count matrices are streamed from the official `tar.gz` archives.
Feature and barcode axes are verified first. Matrix Market coordinates may be
parsed for routing, but a numeric value token is converted only inside the
authorized row-column rectangle. The access audit must report zero converted
unauthorized values.

The held-RNA stage may convert only locked RNA rows and the exact 10,240
selected held columns. It writes mode-0600 cell states outside the repository
and publishes exact RNA margins. It cannot open the ADT matrix member. Public
predictions use the source models, held RNA margins, and design-fixed
`[256,256]` ADT margins. The held-ADT stage is unauthorized until those
predictions are publicly tagged. It may then convert only locked ADT rows and
the identical held columns. A separate public score authorization binds both
private-state hashes and all staged public artifacts. Scoring is the first
operation permitted to join held RNA and ADT states.

## Held endpoint

For observed table `T`, margin-preserving prediction `P`, and `N=512`,
entity loss is `2/N * sum(T_ij log(T_ij/P_ij))`, with zero observed cells
contributing zero. Every donor is averaged over the same complete locked map.
The formal inferential units are the ten author-matched donor pairs.

Against the source-locked classical comparator, untuned Poisson, and the
donor-pooled saturated Poisson log-linear interaction, a pass requires all of:

1. at least 5% lower pair-equal mean loss;
2. a pair-cluster bootstrap 98.75% upper endpoint below zero from 20,000 draws;
3. at least nine of ten favorable pair means;
4. exact one-sided pair-sign `p <= 0.0125`;
5. negative mean differences in PPMS and HI3 separately;
6. a negative mean in each held experiment;
7. a negative mean after leaving out each matched pair; and
8. a negative mean after leaving out each held experiment.

The destroyed-link comparison must meet criteria 1-4. Graph-zero is
prespecified and reported as the test of incremental relational smoothing; it
does not rescue a failed classical comparison. Donor signs and the
magnitude-weighted pair sign-flip test are sensitivity summaries, not formal
randomization tests.

## Public execution

The immutable tag order is:

1. `gse144744-ms-v1-protocol`
2. `gse144744-ms-v1-source-attempt`
3. `gse144744-ms-v1-source`
4. `gse144744-ms-v1-rna-attempt`
5. `gse144744-ms-v1-rna`
6. `gse144744-ms-v1-prediction-attempt`
7. `gse144744-ms-v1-prediction`
8. `gse144744-ms-v1-adt-attempt`
9. `gse144744-ms-v1-adt`
10. `gse144744-ms-v1-score-authorization`
11. `gse144744-ms-v1-score-attempt`
12. `gse144744-ms-v1-result`

Every numeric stage has one exclusive execution claim. Each tag binds the
protocol, runner, numerical modules, reader, tests, input identities, access
journal, and prerequisite artifacts; stage commits must descend from their
public predecessors. An interruption or unexpected exception consumes the
stage and is published. No stage is rerun after numeric access.
