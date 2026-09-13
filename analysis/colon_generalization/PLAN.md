# Blood-to-colon dependence transfer: one pre-access test

## Question and fixed method

Does retaining recipient assay patterns improve cross-assay prediction in a
new tissue, using the already published blood-reference interaction without
refitting? This is a prespecified analysis of public existing data, not a
prospective biological experiment or a registry-hosted preregistration.

Use the source interaction and product-Bernoulli prior probabilities from the
published adaptation-only analysis, `../assay_resolution/results/adaptation_only_results/source_fit.npz`.
Use the same nine RNA and protein markers: CD4, CD7, CD14, CD19, CD33, CD38,
CD44, CD47, CD52. No new source training, recipient hyperparameter selection,
outcome-based marker selection, or replacement cohort is permitted.

## Independent recipients

Use the public counts layer of `biopsy_RNA_ADT_joint_counts_processed.h5ad`,
Figshare file 54027674, article 21919356 version 3, accompanying GSE250490 /
GSE250498. The upstream MD5 is `d0dbb0fc4c95fbd5ce8e72cf418aad21`.
This colon-biopsy study includes healthy controls and ulcerative-colitis
patients receiving aminosalicylates or vedolizumab.

The metadata preflight read feature labels and donor, biopsy, library-type,
and condition identifiers, but no molecular matrix values. Eleven physical
donors have at least 512 cells labeled `GEX_CITE`: four healthy controls, four
UCNB, and three UCV. The twelfth donor has only 433 eligible cells and is
excluded by the fixed budget, not by a prediction score. Pool the eligible
biopsies within each donor, so biopsies and sequencing libraries are not
treated as independent replicates. Use all cell types, without expression-
or annotation-based cell selection. The deposited object is already processed
and its original study's cell filtering is inherited.

For each physical donor, sort eligible barcodes by SHA-256 of
`colon-generalization-v1|{CoLabs_patient}|{barcode}` and retain the first 512.
The first 256 are adaptation cells and the remainder are scoring cells.
`selection.json` records these matrix rows under anonymous donor labels;
the full metadata audit remains local. There are no count-based exclusion
rules. Missing values, noninteger or negative counts, axis disagreement, or
solver failure terminate the run and are reported; they do not justify a
different preprocessing choice.

## Permitted information and endpoint

RNA high means detection. Each protein's adaptation-cell median defines a
threshold; strictly greater counts are high and ties are low. Apply the
threshold unchanged to scoring cells. Smooth adaptation pattern counts with
the fixed source product-Bernoulli prior of total mass four and normalize by
260. Do not supply scoring-cell frequencies or pairings to prediction.

Evaluate individual frequencies, RNA patterns plus protein frequencies,
protein patterns plus RNA frequencies, both pattern distributions, and
independence at the same estimated frequencies. Save and hash predictions
before reading the separate scoring-count file for evaluation. Average twice
forward KL from the empirical scoring table to its prediction over all 81
pairs, including constant-marker pairs, then equally over the eleven donors.

Primary contrasts are both patterns versus (1) individual frequencies,
(2) protein patterns, and (3) independence. Paired donor bootstrap uses
20,000 draws and seed 9132026. Report nominal 95% intervals and Bonferroni
percentile intervals across exactly these three comparisons. A successful
generalization requires all three adjusted lower bounds above zero and a
mean relative deviance reduction of at least 10% versus individual frequencies.
Report absolute losses, donor wins, all five arms, and all contrasts regardless
of whether this criterion passes. Do not combine these new intervals with
earlier retrospective cohorts to rescue a failed result.

Disease-group summaries are descriptive because the groups have only 3--4
donors each. They do not establish a treatment effect. List all 81 query-level
gains as descriptive results without selecting significant biological stories.
No additional inferential endpoints or downstream estimator adaptation are
authorized by this protocol.

## Verification and access order

Before count access, bind this plan, selection, upstream manifest, reducer,
runner, tests, reconstruction implementation, and fixed source model in
`freeze.json`, and publish their Git commit. Record the public commit before
starting the count download. Verify upstream size and MD5, CSR structure,
feature axes, and donor row membership. Separate adaptation and scoring
counts on extraction. Unit tests cover state thresholds, marginal smoothing,
support aggregation, and paired bootstrap. Independently recompute scoring
tables and aggregate losses and compare one full-pattern reconstruction with
the existing log-domain solver. Preserve failures and all outputs.

The preceding search considered lupus PBMCs (GSE189050), vaccine-specific
memory B cells (GSE290006), and two mouse studies only at metadata level.
Colon was chosen before outcomes for its tissue shift and compatible public
assays, not for transfer performance. Previously exposed HCC, leukemia, and
blood cohorts and all terminal campaigns remain unchanged.

## Sources

- Study: https://doi.org/10.1038/s41467-024-45665-6
- Dataset: https://doi.org/10.6084/m9.figshare.21919356.v3
- GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE250490

The deposited dataset is licensed CC BY 4.0. The analysis inherits public
processed data, not access to controlled raw sequencing reads.
