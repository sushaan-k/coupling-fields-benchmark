# GSE217494 cardiac CITE-seq held-donor confirmation protocol v1

## Question and estimand

This one-way experiment tests whether a structured, etiology-conditioned RNA--protein coupling field learned from 14 human hearts predicts joint RNA--protein tables in eight donor-disjoint hearts beyond matched classical interaction models. The physical explanted heart is the inferential unit. The target is donor-equal mean multinomial deviance per cell across a source-selected ordered RNA--protein axis.

## Frozen split and access boundary

`data/confirmation/gse217494_heart/candidate_designation_v1.json` fixes an outcome-blind, etiology-stratified split. Two samples in each of Donor, acute myocardial infarction (AMI), ischemic cardiomyopathy (ICM), and non-ischemic cardiomyopathy (NICM) are held by the first and only semantic SHA-256 salt. The source contains 14 hearts and the held set contains eight.

Before publication of this protocol, no `matrix.mtx.gz` body, Matrix Market header, or entry was requested. Public GEO metadata, HTTP content-length headers, and all 22 feature and barcode axes were inspected after the split was fixed. The metadata contain cell labels and per-cell QC totals; neither the split nor any estimator choice used those values. All eight held matrix bodies remain physically unopened until source-only interaction fields are published.

Each sample is deposited as one 10x MEX triplet containing Gene Expression and Antibody Capture rows on a common barcode axis. Consequently, held RNA margins cannot be opened separately from held ADT values. The prediction artifact will instead contain source-only interaction fields for every method. After public score authorization, each held triplet is acquired once, its margins and truth are formed, every frozen field is reconstructed at those margins, and the triplet is deleted before the next request.

## Cells, states, and marker axis

For each heart, the 512 Cell Ranger-filtered MEX barcodes with the smallest SHA-256 values of `GSE217494-CELL-v1|sample|barcode` are retained. Selection uses no assay value or deposited cell-type label. Every barcode axis contains more than 5,000 cells. The donor-level estimand intentionally includes each heart's sampled cellular composition.

RNA state is raw UMI count greater than zero. For each protein, exactly 256 retained cells are assigned the high state by raw ADT rank; SHA-256 of `GSE217494-ADT-TIE-v1|sample|protein|barcode` breaks ties. Complete ADT profiles are cyclically shifted along the ordering induced by `GSE217494-DESTROY-v1|sample|barcode` to form the destroyed-link control.

Candidate markers are unique exact RNA-symbol/ADT-symbol cognates on the common feature axis; isotype, control, and nonhuman rows are excluded. The 249-symbol candidate axis and its canonical hash are bound in the designation. In every training heart, a candidate must have 16--496 RNA-positive cells and at least 16 ADT counts that differ from that heart's modal count for the marker. Its RNA detection profile and mean library-normalized ADT profile must each have finite, nonzero variance across training hearts. Eligible markers are ranked by descending minimum training balance `min(positive, negative)`, then descending median training balance, then symbol. The first 12 are retained; fewer than nine is a terminal support refusal. The estimand uses the complete ordered RNA-by-protein cross-product.

## Structured conditional field

For heart `d` and ordered marker pair `e`, the log odds ratio is

`theta[d,e] = x[d] beta[,e] + u[d,e]`,

where `x[d]` contains an intercept and AMI, ICM, and NICM indicators. The estimator minimizes donor-normalized exact fixed-margin conditional negative log likelihood plus positive quadratic penalties on donor deviations and coefficients and a product-graph penalty on the context fields. The RNA graph is constructed from training RNA detection profiles; the protein graph is constructed from training mean library-normalized ADT profiles. A zero-variance or nonfinite marker profile is ineligible before graph construction. Each graph is the deterministic symmetrized three-nearest-neighbor graph after featurewise centering and scaling. Its Laplacian is normalized to mean diagonal one, and the ordered-pair Laplacian is their Kronecker sum.

The objective is convex. Positive coefficient and deviation penalties make it coercive and strictly convex, so every accepted coordinate fit has a unique finite solution. The implementation returns a fit only when its scaled-gradient and factor-conditioning certificates pass. Recipient fields use the fitted etiology context with zero donor deviation and are reconstructed as exact noncentral-hypergeometric expected tables at observed recipient margins.

Leave-one-source-heart-out cross-validation selects deviation penalty `{0.3, 3}`, non-intercept coefficient ridge `{0.1, 1}`, graph penalty `{0, 0.01, 0.1, 1}`, and field transport `{0.75, 1, 1.25}`. The intercept ridge is `0.01`. Each fold repeats marker support filtering, marker ranking, and graph construction on its 13 training hearts. The validation heart contributes only its margins and truth to the prespecified fold loss; it contributes no profile, graph edge, support or ranking decision, mask, fit, or hyperparameter candidate construction. Donor-equal deviance determines the configuration; the displayed numeric tuple breaks exact ties. The final axis and graph are rebuilt from all 14 source hearts. A graph-specific claim is permitted only if cross-validation selects a positive graph penalty and the selected fit improves on the separately tuned graph-zero ablation in both source cross-validation and held scoring.

## Classical comparators

Every method receives the same source tables, selected entities, context labels, and held margins.

1. **Pooled fixed-interaction Poisson:** donor-specific row and column nuisance terms with one source-common interaction per entity.
2. **Etiology-specific fixed-interaction Poisson:** the same saturated log-linear model with a separate interaction in Donor, AMI, ICM, and NICM hearts.
3. **Standardized Poisson interaction residual:** source signed Pearson interaction residuals under fixed-margin independence, averaged within etiology and inverted at recipient margins.
4. **Exact common-effect conditional field:** a single fixed-margin conditional interaction per entity.
5. **Fixed-margin independence.**
6. **Destroyed links:** the complete primary pipeline refitted after the deterministic within-heart ADT shift.

The two Poisson comparators are fitted as standard saturated fixed-interaction log-linear models with donor-specific source nuisance terms. At recipient margins, row and column nuisance parameters are refitted under the transported interaction; their reconstruction does not use a noncentral-hypergeometric expectation. Each transportable classical field independently selects scale `{0.75, 1, 1.25}` by the same source folds. The strongest source-CV classical comparator is frozen before held access and is the classical gate comparator. Pooled and etiology-specific Poisson results are both reported regardless of which is stronger.

## Source promotion

Held access remains disabled unless all source reductions and final fits complete, at least nine markers survive, and leave-one-heart-out mean deviance for the structured primary is below both the strongest classical comparator and destroyed links. At least ten of 14 source hearts must favor the primary against each, and mean improvement must be positive in all four etiologies. Failure is terminal and produces no held prediction or score.

If promoted, the source artifact publishes the selected markers, graphs, hyperparameters, fitted etiology interaction fields for the primary and every comparator, source-fold losses, and hashes. It contains no held margin or assay value.

## Held confirmation

The primary passes only if, against both the frozen strongest classical comparator and destroyed links:

- donor-equal mean deviance is at least 5% lower;
- the upper endpoint of a 20,000-draw etiology-stratified paired-donor bootstrap 95% interval for primary-minus-comparator deviance is below zero;
- at least seven of eight held hearts favor the primary; and
- mean improvement is positive in Donor, AMI, ICM, and NICM hearts.

The exact one-sided sign probability for the favorable-heart count is reported. Bootstrap seed `21749401` and all salts are fixed above. The graph-zero ablation, every named classical method, per-etiology losses, and per-heart losses are reported without changing the gate.

## Biological secondary analyses

Three source-frozen summaries are evaluated after the primary decision: recovery of held nearest-neighbor marker relations in the predicted coupling field, within-module edge recovery for endothelial, fibroblast, and immune marker sets represented on the selected axis, and enrichment of the FAP/LRRC15 cardiac-fibrosis program when its members survive the source support rule. Target-label permutations and within-family Benjamini--Hochberg correction are applied. These analyses are secondary to the donor-level transfer decision.

## Terminality and reporting

Schema, support, numerical, source-promotion, prediction-publication, acquisition, and score failures are distinct terminal outcomes. Every matrix must have the Matrix Market coordinate/integer/general banner, dimensions equal to its frozen feature and barcode axes, finite nonnegative integer entries, and no out-of-range index. Duplicate coordinates are accumulated independent of row order by a bounded-memory selected-row reducer. A mismatch is terminal. The result records every requested file, observed byte count and SHA-256, parsed entries, deletion, and whether held access occurred. This protocol cannot be rerun after any terminal source or held result. The aggregate benchmark retains the campaign whether it passes, refuses, or becomes infrastructure-unevaluable.
