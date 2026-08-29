# GSE164378 3-prime to GSE155673 external-study protocol

Frozen at `2026-08-29T11:12:00Z` from official GEO metadata, supplementary file listings,
barcode axes, feature schemas, and antibody reagent metadata. No Matrix Market
file was downloaded or opened, no assay row was inspected, and no numeric assay
value was read before this freeze.

## Status and inferential scope

This is a post-failure named endpoint selected through adaptive public-metadata
search after outcomes from earlier GSE202150 and GSE185381 branches were known.
Directional alpha `0.0125` applies only to this frozen endpoint. It does not
control the historical search-wide familywise error rate and does not make the
dataset selection prospective. A miss, refusal, interruption, or exception
closes the branch; the allocation is never recycled.

The source is the day-0 10x 3-prime CITE-seq assay from GSE164378. The held study
is the twelve-donor 10x 3-prime CITE-seq deposit GSE155673. The estimand is
composition-inclusive RNA-ADT coupling across PBMCs, not a cell-type-conditional
or causal effect. Official records are [GSE164378](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164378)
and [GSE155673](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE155673);
the associated papers are [Hao et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC8238499/)
and [Stephenson et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7665312/).

## Frozen donors

Every donor exceeds the fixed budget of 384 cells. The source split balances
both deposited source batches:

| Source role | Batch 1 | Batch 2 |
|---|---|---|
| Calibration | P1 (6,443), P3 (4,698) | P5 (7,020), P7 (8,909) |
| Validation | P2 (5,978), P4 (5,307) | P6 (6,093), P8 (8,916) |

The validation components are fixed as `[P2,P4]` and `[P6,P8]`. Configuration
loss is the equal mean of their component losses. The held units are the twelve
deposited donor files:

| File | Sample | Phenotype | Severity/day | Sex | Age | Cells |
|---|---|---|---|---:|---:|---:|
| cov01 | nCOV3EUHM | COVID-19 | Severe, day 15 | F | 75 | 4,491 |
| cov02 | nCOV7EUHM | COVID-19 | Moderate, day 9 | F | 53 | 1,891 |
| cov03 | nCOV1EUHM | COVID-19 | Moderate, day 2 | F | 75 | 6,233 |
| cov04 | nCOV6EUHM | COVID-19 | Severe, day 16 | M | 59 | 2,453 |
| cov07 | 280 | Healthy | not applicable | F | 84 | 5,485 |
| cov08 | 259 | Healthy | not applicable | F | 68 | 5,419 |
| cov09 | 279 | Healthy | not applicable | M | 38 | 4,827 |
| cov10 | nCOV021EUHM | COVID-19 | Severe, day 15 | F | 60 | 4,512 |
| cov11 | nCOV024EUHM | COVID-19 | Severe, day 8 | M | 48 | 7,874 |
| cov12 | nCOV0029EUHM | COVID-19 | Moderate, day 9 | F | 47 | 5,145 |
| cov17 | 265 | Healthy | not applicable | M | 90 | 9,143 |
| cov18 | 258 | Healthy | not applicable | F | 70 | 5,996 |

The held paper reports two experimental batches, but official GEO metadata do
not expose donor-to-batch membership. Donor/file is therefore the analysis unit;
no batch assignments or acquisition clusters are inferred.

## Frozen cognates

The one-to-one panel has 24 RNA genes and 24 exact ADT feature axes. Workbook
labels use underscores for duplicate reagents; deposited source ADT labels use
hyphens. Both strings are literal below. Source RNA features are resolved as
exact `(rna_symbol,rna_symbol)` rows because that deposit uses symbols in both
columns; held RNA features are resolved as exact `(Ensembl ID,rna_symbol)` rows.
The candidate designation freezes each held Ensembl ID.

| RNA | ADT | Catalog | Sequence | Clone | Source workbook / feature | Held feature |
|---|---|---:|---|---|---|---|
| CD3D | CD3 | 300475 | CTCATTGTAACTCCT | UCHT1 | CD3_1 / CD3-1 | CD3--UCHT1-TSA |
| CD8A | CD8 | 301067 | GCTGCGCTTTCCATT | RPA-T8 | CD8a / CD8a | CD8--RPA-T8-TSA |
| CD19 | CD19 | 302259 | CTGGGCAATTACTCG | HIB19 | CD19 / CD19 | CD19--HIB19-TSA |
| MS4A1 | CD20 | 302359 | TTCTGGGTCCCTAGA | 2H7 | CD20 / CD20 | CD20--2H7-TSA |
| NCAM1 | CD56 | 362557 | TCCTTTCCTGATAGG | 5.1H11 | CD56_1 / CD56-1 | CD56--5-1H11-TSA |
| CD69 | CD69 | 310947 | GTCTCTTGGCTTAAA | FN50 | CD69 / CD69 | CD69--FN50-TSA |
| CD28 | CD28 | 302955 | TGAGAACGACCCTAA | CD28.2 | CD28 / CD28 | CD28--CD28-2-TSA |
| FAS | CD95 | 305649 | CCAGCTCATTAGAGC | DX2 | CD95 / CD95 | CD95--DX2-TSA |
| PTPRC | CD45RA | 304157 | TCAATCCTTCCGCTT | HI100 | CD45RA / CD45RA | CD45RA--HI100-TSA |
| HLA-DRA | HLA-DR | 307659 | AATAGCGAGCAAGTA | L243 | HLA-DR / HLA-DR | HLA-DR--L243-TSA |
| CD14 | CD14 | 301855 | TCTCAGACCTCCGTA | M5E2 | CD14 / CD14 | CD14--M5E2-TSA |
| FCGR3A | CD16 | 302061 | AAGTTCACTCTTTGC | 3G8 | CD16 / CD16 | CD16--3G8-TSA |
| ITGAX | CD11c | 371519 | TACGCCTATAACTTG | S-HCL-3 | CD11c / CD11c | CD11c--S-HCL-3-TSA |
| CD1C | CD1c | 331539 | GAGCTACTTCACTCG | L161 | CD1c / CD1c | CD1c_BDCA1--L161-TSA |
| CLEC9A | CD370 | 353807 | CTGCATTTCAGTAAG | 8F9 | CD370 / CD370 | CD370_CLEC9A--8F9-TSA |
| IL3RA | CD123 | 306037 | CTTCACTCTGTCAGG | 6H6 | CD123 / CD123 | CD123--6H6-TSA |
| CD86 | CD86 | 305443 | GTCTTTGTCAGTGCA | IT2.2 | CD86 / CD86 | CD86--IT2-2-TSA |
| CD274 | CD274 | 329743 | GTTGTCCGACAATAC | 29E.2A3 | CD274 / CD274 | CD274_PD-L1--29E-2A3-TSA |
| CD163 | CD163 | 333635 | GCTTCTCCTTCCTTA | GHI/61 | CD163 / CD163 | CD163--GHI-61-TSA |
| CD27 | CD27 | 302847 | GCACTCCTGCATGTA | O323 | CD27 / CD27 | CD27--O323-TSA |
| CD38 | CD38 | 303541 | TGTACCCGCTTGTGA | HIT2 | CD38_1 / CD38-1 | CD38--HIT2-TSA |
| IL2RA | CD25 | 302643 | TTTGTCCTGTACGCC | BC96 | CD25 / CD25 | CD25--BC96-TSA |
| CD34 | CD34 | 343537 | GCAGAAATCTCCCTT | 581 | CD34 / CD34 | CD34--581-TSA |
| IL7R | CD127 | 351352 | GTGTGTTGTCCTATG | A019D5 | CD127 / CD127 | CD127--A019D5-TSA |

CD8 is an exact match: the source CD8a reagent is catalog 301067, clone RPA-T8,
and maps to held `TotalSeq-301067`; it is not the separate source CD8/SK1
reagent. CD45RA and CD45RO both map to the gene-level PTPRC row, so the frozen
lower-catalog rule retains CD45RA and excludes CD45RO. Catalog 350231 is
excluded because source and held reagent metadata call it CD103/Ber-ACT8 but the
held feature label calls it CD57. CD20 is present as exact held catalog and clone
in the held feature schema, although its catalog is omitted from the held HTO
CSV. The machine-readable designation records every Ensembl ID and exclusion.

## Cells, states, and support

For each donor, the 384 smallest SHA-256 digests of
`GSE164378-3P-GSE155673-CELL-BUDGET-v1|donor_id|cell_id` define the frozen
sample, with cell ID breaking digest ties.
RNA state is deposited UMI count greater than zero. An RNA axis is valid when
detection prevalence lies in `[0.05,0.95]`. An ADT axis is valid when it has at
least two distinct counts and the largest equal-count fraction is at most
`0.90`. For each locked marker, cells are ordered by raw ADT count, SHA-256 of
`GSE164378-3P-GSE155673-ADT-v1|donor_id|locked_marker_index|cell_id`, then cell
ID; the top 192 are high and the bottom 192 are low.

A marker enters the source-locked panel only when both axes are valid in every
one of P1-P8. At least 16 of the 24 markers must lock. Failure is terminal. In
the held study, every locked marker must have valid RNA support in at least 10
of 12 donors and valid ADT support in at least 10 of 12 donors. If `m` markers
lock, every held donor must retain at least `ceil(0.8*m^2)` informative ordered
RNA-by-ADT pairs. Markers and donors are never dropped, replaced, or relabeled
after these checks.

## Source fitting and eligibility

The primary estimator is the graph-regularized exact-fixed-margin hierarchical
coupling field. Its two-nearest-neighbor marker graphs use continuous donor
profiles, not binary states. For each donor and marker, the RNA coordinate is
the mean over frozen cells of `log1p(10,000 * raw RNA count / cell RNA library
size)`; the ADT coordinate is the mean of `log1p(raw ADT count)`. Each marker's
profile vector spans the calibration donors, and the two graphs induce the
product-graph penalty. The frozen grid crosses
heterogeneity `{0.1,1,10}`, ridge `{0.01,0.1,1}`, graph `{0.1,1}`, and transport
`{0.5,0.75,1}`. Selection minimizes equal-component deviance across `[P2,P4]`
and `[P6,P8]`, with lexicographic tie breaking. A graph-zero model is selected
separately and cannot replace the primary. For the destroyed-link control,
source cells are sorted within donor by SHA-256 of
`GSE164378-3P-GSE155673-DESTROY-v1|donor_id|cell_id`; complete ADT state vectors
are cyclically rotated by one position relative to RNA. This preserves every
donor-marker ADT margin and destroys same-cell pairing. The primary
source-selected hyperparameters remain fixed for this refit; marker identities
are not permuted and the control is not retuned.

The classical inventory is untuned Poisson-independence signed-root-deviance,
source-calibrated Poisson residual, common-effect stratified conditional MLE,
donor-pooled saturated 2-by-2 Poisson log-linear interaction, and Paule-Mandel
random-effects log odds. The minimum-loss estimable method among the four
source-calibrated families locks, with that stated order breaking exact ties.
Untuned raw Poisson is never eligible to lock and is retained separately.

Before held access, the selected primary must have strictly lower mean
validation deviance than both the locked comparator and untuned raw Poisson in
`[P2,P4]`, and must independently do so in `[P6,P8]`. An equality or loss in any
of these four checks is a terminal source refusal. After a pass, the same
continuous mean-log coordinates are recomputed across all eight source donors;
both marker graphs are rebuilt from those eight-donor profile vectors, and the
selected model is refit on all eight donors. Calibration-only graphs are not
reused.

## Row-aware held firewall

Each held Matrix Market file combines 33,538 RNA rows and 39 ADT rows. The RNA
stage may decompress and stream the physical file and parse coordinate row and
column tokens needed for routing. It converts, retains, aggregates, and
serializes numeric values only for the 24 frozen RNA rows. ADT-row numeric value
tokens are discarded without numeric conversion. The ADT stage applies the
converse rule. This is a row-aware information firewall, not a claim that
compressed bytes remain unread.

The held RNA attempt precedes first download and whole-file hashing. The later
ADT stage verifies the same physical bytes and hash before its row pass. RNA
and ADT stages may form marginal states but cannot join them. Public prediction
may use the source model, held RNA margins, design-fixed 192/192 ADT margins,
and the separately computed RNA- and ADT-axis validity masks required by the
frozen support gate. It cannot use raw held ADT counts. Scoring is the first
stage permitted to join held RNA and ADT states or form joint tables.

Each authorized file download has exactly three byte-transport attempts. A
failed partial file is deleted before the next attempt and is not journaled. A
completed attempt is hashed and receives a `DOWNLOADED_AND_HASHED` row with its
observed byte count and SHA-256 before comparison with the manifest. A byte or
hash mismatch is therefore public and terminal. Exhausting all three attempts
is terminal for the claimed stage. These transport attempts occur within one
execution claim; they do not permit an analytical rerun, changed code or inputs,
a new stage claim, or a threshold revision.

The global `GSE155673_barcodes.tsv.gz` is excluded because it duplicates cov01.
Held barcode axes remain donor-keyed throughout processing, and every selection
digest includes the donor file ID. This namespaces the 600 barcode strings that
recur across donor files, whose maximum multiplicity is three.

## Held gate

The endpoint is donor-equal multinomial deviance per cell over informative
ordered pairs. The primary paired difference is primary loss minus locked
comparator loss. A pass requires all of the following:

1. mean primary loss is at least 5% lower;
2. the upper 98.75th percentile of a 20,000-draw disease-stratified paired donor
   bootstrap is below zero;
3. at least 11 of 12 donors favor the primary;
4. the exact one-sided donor sign-test `p <= 0.0125`;
5. the exact one-sided 4,096-assignment paired sign-flip `p <= 0.0125`;
6. separate healthy and COVID-19 mean differences are negative;
7. separate moderate- and severe-COVID mean differences are negative; and
8. every leave-one-donor-out mean difference is negative.

The identical full gate is repeated against untuned raw Poisson and labeled
broad-classical support; the locked-comparator gate remains primary. The serial
chain is locked comparator, raw Poisson, graph zero, then destroyed link. A
later contrast is evaluated only after its predecessor passes. Target-margin
independence and nonlocked classical methods are descriptive and cannot rescue
a failure.

## Public execution

The immutable tag sequence is:

1. `gse164378-3p-gse155673-v1-protocol`
2. `gse164378-3p-gse155673-v1-source-attempt`
3. `gse164378-3p-gse155673-v1-source`
4. `gse164378-3p-gse155673-v1-rna-attempt`
5. `gse164378-3p-gse155673-v1-rna`
6. `gse164378-3p-gse155673-v1-adt-attempt`
7. `gse164378-3p-gse155673-v1-adt`
8. `gse164378-3p-gse155673-v1-prediction-attempt`
9. `gse164378-3p-gse155673-v1-prediction`
10. `gse164378-3p-gse155673-v1-score-authorization`
11. `gse164378-3p-gse155673-v1-score-attempt`
12. `gse164378-3p-gse155673-v1-result`

Every numeric stage has one exclusive execution claim. Attempt tags bind empty
append-only access journals; completion tags bind the resulting journals even
after terminal failure. Every attempt, completion, authorization, and result tag
must commit-descend all prerequisite commits named in its ledger; matching file
bytes without Git ancestry are insufficient. The score-authorization artifact
must bind an immutable public prediction and every frozen dependency in a later
descendant commit.

Before the protocol tag, tests must verify the exact donor set and source split,
the 24-marker table and all-eight source lock, the `>=16` and held-support gates,
the two validation-component source-eligibility checks, final all-eight graph
rebuilding, deferred assay hashes, the combined-file row firewall, the complete
held gate, and the exact tag sequence.
