# Kotliarov PBMC held-batch confirmation protocol

**Protocol date:** 2026-08-28

**Dataset:** `KotliarovPBMCData`, `scRNAseq/kotliarov-pbmc-2020`

**Study:** Kotliarov et al., *Nature Medicine* 2020, DOI 10.1038/s41591-020-0769-8
**Status at designation:** outcome access disabled

## Confirmatory question

Can a structured interaction atlas learned from one CITE-seq experimental batch
predict same-cell RNA-protein joint distributions for biological donors in a
second batch, using only the held RNA distribution and unpaired ADT margins?
The primary endpoint is held-table multinomial deviance under identical observed
row and column margins. The primary method must improve on the unstructured
interaction mean, a pairing-destroyed control, and full Pearson and signed
Poisson-deviance residual transfer.

This protocol opens a new one-candidate confirmatory family. The earlier Lawlor
and Hao family is closed after two pre-outcome procedural refusals; neither run
formed a held joint table or spent inferential alpha. This candidate is executed
once and reported regardless of outcome. The public Git commit is a prospective,
commit-addressed analysis-plan freeze, not registration in a trial registry.

## Source and untouched outcomes

The source is the Bioconductor `scRNAseq` gypsum asset. Its deposited object has
32,738 RNA features, 87 ADTs, and 58,654 cells. The public documentation states
that 20 samples were processed in two experimental batches, with five high and
five low vaccine responders per batch, and that each pooled batch was spread
over six 10x lanes. Lanes and cells are never treated as biological replicates.

`data/development/kotliarov_pbmc/source_manifest_v1.json` fixes the exact gypsum
versions, URLs, byte counts, and published MD5 values. Before this freeze, only
the artifact manifest, metadata, RNA/ADT feature names, and HTTP headers were
opened. The 82,850,550-byte RNA matrix and 4,708,920-byte ADT matrix were not
downloaded or opened. Acquisition after authorization must satisfy both byte
count and MD5; SHA-256 values are then computed and bound into the reducer
manifest.

## Leak-free donor split

Donor 209 occurs in both experimental batches. Every cell assigned to donor 209
is therefore excluded before QC, lineage assignment, threshold estimation,
tuning, margin extraction, prediction, or scoring.

| Role | Batch | Donors |
|---|---:|---|
| Development | 1 | 200, 207, 212, 233, 237, 245, 256, 261, 273, 277 |
| Held confirmation | 2 | 201, 205, 215, 229, 234, 236, 250, 268, 279 |
| Excluded globally | both | 209 |

The retained metadata contains 29,374 development and 25,820 held cells before
RNA QC. Development contains five high and five low responders; held contains
four high and five low responders. Response class is never used by the
estimator or tuning. It only stratifies the donor bootstrap so resamples retain
the deposited 4:5 held composition.

## RNA-only cell processing

Cells must have concordant singlet assignments
(`joint_classification_global == SNG_Singlet` and `dmx_hto_match == 1`) and
timepoint `d0`. Within each donor, cells outside median plus or minus three MADs
for detected genes or RNA UMI total are removed. Cells above the smaller of 20%
mitochondrial reads and donor median plus three MADs are removed. Zero MAD leaves
the corresponding metric untrimmed except for the fixed mitochondrial cap. No
ADT total, ADT feature, author protein cluster, responder class, or held pairing
enters QC.

Lineages are assigned from RNA only using
`data/development/kotliarov_pbmc/lineage_markers_v1.tsv`. Counts are normalized
to 10,000 RNA UMIs and transformed with `log1p`. Within donor, each marker is
robustly standardized by its median and MAD; each lineage score is mean positive
marker score minus mean negative marker score. The highest score assigns B,
CD4 T, CD8 T, NK, or Monocyte. Exact score ties, nonfinite scores, and maxima not
greater than zero are excluded. A lineage is retained only if all 19 donors have
at least 50 retained cells; fewer than four retained lineages is a support
refusal.

## Exact cognate markers and states

The 71-row alias file
`data/development/kotliarov_pbmc/adt_gene_aliases_v1.tsv` is exact and
one-to-one. There is no fuzzy or normalized-token fallback. Four isotypes,
Annexin V, multigene antigens (CD3, HLA-ABC, HLA-DR, TCRgd), isoform-ambiguous
CD45RA/CD45RO, and IgA are excluded before outcome access. A missing or duplicate
RNA symbol or ADT target is a refusal for that alias; unsupported aliases are
reported and never replaced.

RNA values are `log1p(count / RNA_UMI_total * 10000)`. ADT values are cell-wise
CLR coordinates: `log1p(count)` minus the mean `log1p(count)` over all 83
non-isotype ADTs. For each marker-lineage entity and assay, two donor-equal
weighted tertile cuts are estimated from development donors only. In
leave-one-development-donor-out tuning, cuts are refit within the nine-donor
training fold. Frozen full-development cuts are applied unchanged to held cells.

An entity is eligible only if both cuts are distinct and, separately for every
retained donor, each RNA and ADT marginal state contains at least five cells and
2% of the donor-lineage block. Held eligibility uses only the two separate
margins before any pairing is opened. At least 16 unique markers, 32
marker-lineage entities, and 12 markers covered by the frozen external gene
embedding are required.

## Pairing seal

The noninteractive reducer emits two byte-bound bundles. The prediction bundle
contains cell metadata, RNA values, development paired ADT values, full-
development cuts, and held RNA and ADT state margins. It contains no held
per-cell ADT values, held joint table, held barcode-paired ADT state, or
pairing-derived statistic. The score-only bundle contains the held per-cell ADT
states on the identical hashed cell axis. Prediction must be invariant to an
arbitrary within-donor-lineage permutation of held ADT cells. The score bundle
is not opened until the prediction JSON and score authorization have been
published at immutable Git commits.

## Interaction field and primary estimator

Each donor-entity yields a 3 by 3 RNA-state by ADT-state table. The interaction
field is the four Helmert coordinates of the double-centered `log(N + 0.5)`
table minus the mean from 64 deterministic fixed-margin permutations. A 65th,
disjoint permutation is the pairing-destroyed control. Development donors have
equal weight.

The primary estimator jointly shrinks the entity-by-four-coordinate atlas with
a weighted nuclear norm and a normalized hypergraph Laplacian. Hyperedges join
the same marker across lineages, entities within a lineage, entities in the
same frozen protein module, and each marker with its six nearest covered genes
in the checksum-bound scGPT embedding. Clipped inverse permutation variance
supplies coordinate weights. This is a structured estimator of conditionally
centered log-linear interactions; it does not claim a new interaction estimand.

Hyperparameters are selected only by ten-fold leave-one-development-donor-out
table deviance, averaging first across lineages within marker, then across
markers, then donors. The primary grid is the Cartesian product of positive
nuclear multipliers `{0.03, 0.1, 0.3, 1}` times the leading singular value and
positive graph penalties `{0.5, 2, 5, 10}`. Boundary values at zero define the
prespecified direct, nuclear-only, and hypergraph-only controls; they are not
eligible to become the primary estimator. Ties select the smaller nuclear
multiplier, then the smaller graph penalty. Preprocessing and state cuts are
refit inside each fold. The selected estimator is refit once on all ten
development donors.

## Fixed comparisons

All methods receive the same held row and column margins and are scored on the
same reconstructed held tables.

1. Unstructured development-donor mean field.
2. Independence, scalar shrinkage, nuclear-only, hypergraph-only, external-
   membership-permuted hypergraph, and pairing-destroyed structured controls.
3. Full nine-cell Pearson independence residuals, direct and structured.
4. Full nine-cell signed Poisson-deviance independence residuals, direct and
   structured.

Each classical structured comparator uses the same hypergraph, inverse-null-
variance weighting, LOPO folds, penalty grid, tie rule, and held margins. Full
residual matrices are not projected to four coordinates. For every field or
residual method, the held finite-margin permutation mean and count scale are
restored, a strictly positive seed is formed, and iterative proportional
fitting enforces the observed margins to tolerance `1e-8`. Failure to converge
or any larger margin error is a refusal.

The uncentered zero-pseudocount field is the saturated Poisson row-by-column
interaction under sum-to-zero constraints. The implemented uncentered field
uses `N + 0.5`; the reported coordinates additionally subtract the fixed-margin
permutation mean. Pearson and signed-deviance residuals therefore test whether
that conditional representation and its structural estimator add predictive
value over standard independence-model residual transfer.

## Endpoint and inference

Primary loss is multinomial deviance per held cell. Within donor, loss is
averaged across eligible lineages within marker and then across markers; donors
are averaged equally. Representation correlation is computed per held donor on
the fixed eligible field-coordinate panel, Fisher-z transformed, and averaged
across donors.

The nine held donors are the biological replicates. Ten thousand paired donor-
cluster bootstrap draws use seed 20260828 and resample four high and five low
donors separately with replacement, retaining every cell, marker, lineage, and
method result attached to a donor. All `2^9 = 512` donor sign assignments are
enumerated for each paired loss contrast and for Fisher-z correlation. The
observed assignment is included. Cells and lanes are never resampled as
independent units.

## Exact promotion gate

The result is `PASS` only if every condition holds:

1. mean held Fisher-z field correlation has a two-sided 95% stratified donor-
   bootstrap lower endpoint above zero, exact one-sided sign-flip `p <= 0.025`,
   and at least 8 of 9 donor correlations are positive;
2. primary minus unstructured-field deviance has an upper 95% endpoint below
   zero, exact `p <= 0.025`, at least 5% point relative reduction, and at least
   8 of 9 donor differences below zero;
3. the same conditions hold against the best of Pearson-direct, Pearson-
   structured, deviance-direct, and deviance-structured, reselecting the best
   classical comparator within every bootstrap draw;
4. primary minus pairing-destroyed deviance has an upper endpoint below zero,
   exact `p <= 0.025`, and at least 8 of 9 donors favor primary;
5. primary minus the best matched non-primary field method has an upper endpoint
   below zero, exact `p <= 0.025`, and at least 8 of 9 donors favor primary; and
6. source integrity, global donor-209 exclusion, support, pairing invariance,
   optimizer convergence, table reconstruction, and artifact hashes all pass.

The conjunction is an intersection-union test, so no Bonferroni adjustment is
applied within this one candidate. Responder- and lineage-specific results are
descriptive and BH-adjusted within their declared families. No responder-
specific promotion claim is permitted.

## Execution and reporting

The disabled designation binds this protocol, source manifest, aliases, lineage
markers, runner, reducer, tests, embedding, and shared field/reconstruction
implementation. A fresh clone must verify the public candidate-freeze commit
before matrix acquisition is authorized. Prediction runs once. Its exact JSON
is then published and verified in a fresh clone. A later public authorization
commit binds that prediction before the held score bundle can be opened. Score
runs once and writes one result or one terminal refusal.

The public benchmark reports the Lawlor reducer refusal, the Hao support
refusal, and this candidate's complete result or refusal. It includes every
method/donor loss, exclusions, intervals, exact tests, acquisition hashes, and
immutable Git URLs. No split, alias, support threshold, penalty grid, method,
gate, or endpoint may be repaired and reported as this confirmation after
outcome access.
