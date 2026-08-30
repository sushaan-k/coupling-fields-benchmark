# GSE217494 cardiac CITE-seq held-donor confirmation protocol v1

## Question and estimand

This one-way experiment tests whether a structured, etiology-conditioned RNA--protein coupling field learned from 14 human hearts predicts joint RNA--protein tables in eight donor-disjoint hearts beyond matched classical interaction models. The physical explanted heart is the inferential unit. The target is donor-equal mean multinomial deviance per cell across a source-selected ordered RNA--protein axis.

## Frozen split and access boundary

`data/confirmation/gse217494_heart/candidate_designation_v1.json` fixes an outcome-blind, etiology-stratified split. Two samples in each of Donor, acute myocardial infarction (AMI), ischemic cardiomyopathy (ICM), and non-ischemic cardiomyopathy (NICM) are held by the first and only semantic SHA-256 salt. The source contains 14 hearts and the held set contains eight.

Before publication of this protocol, no `matrix.mtx.gz` body, Matrix Market header, or entry was requested. Public GEO metadata, HTTP content-length headers, and all 22 feature and barcode axes were inspected after the split was fixed. The metadata contain cell labels and per-cell QC totals; neither the split nor any estimator choice used those values. All eight held matrix bodies remain physically unopened until source-only interaction fields are published.

Each sample is deposited as one 10x MEX triplet containing Gene Expression and Antibody Capture rows on a common barcode axis. Consequently, held RNA margins cannot be opened separately from held ADT values. The prediction artifact will instead contain source-only interaction fields for every method. After public score authorization, each held triplet is acquired once, its margins and truth are formed, every frozen field is reconstructed at those margins, and the triplet is deleted before the next request.

## Cells, states, and marker axis

For each heart, the 512 Cell Ranger-filtered MEX barcodes with the smallest `(SHA256(UTF8(GSE217494-CELL-v1|sample|barcode)), barcode)` tuples are retained. Selection uses no assay value or deposited cell-type label. Every barcode axis contains more than 5,000 cells. The donor-level estimand intentionally includes each heart's sampled cellular composition.

RNA state is raw UMI count greater than zero. For each protein, retained cells are ordered by `(-raw ADT count, SHA256(UTF8(GSE217494-ADT-TIE-v1|sample|protein|barcode)), barcode)`; the first 256 are assigned the high state. Tables use rows RNA-negative/RNA-positive and columns ADT-low/ADT-high, so positive coupling is `log(n00 n11 / (n01 n10))`. To form the destroyed-link control, retained cells are ordered by `(SHA256(UTF8(GSE217494-DESTROY-v1|sample|barcode)), barcode)`, and the complete 279-feature ADT vector at position `i` is replaced by the original vector at `(i + 256) mod 512`. In a source fold only training-heart ADTs are shifted; validation and held truth remain unshifted.

Candidate markers are unique exact RNA-symbol/ADT-symbol cognates on the common feature axis; isotype, control, and nonhuman rows are excluded. The 249 symbols, RNA and ADT row indices, and feature IDs are bound in `data/confirmation/gse217494_heart/cognate_axis_v1.tsv`. In every training heart, a candidate must have 16--496 RNA-positive cells and satisfy `512 - max_count_frequency >= 16` on its raw ADT counts. Its RNA detection profile is the fraction of retained cells with positive RNA. Its ADT profile is the retained-cell mean of `log1p(10^4 c[p]/max(1, sum_q c[q]))`; the denominator spans all 279 Antibody Capture rows, including controls. Both profiles must have finite, nonzero variance across training hearts. Eligible markers are ranked by descending minimum training balance `min(positive, negative)`, then descending median training balance, then symbol. The first 12 are retained; fewer than nine is a terminal support refusal. Every selected pair must have nondegenerate fixed-margin support in every training heart; support is never silently dropped. The estimand uses the complete ordered RNA-by-protein cross-product.

## Structured conditional field

For heart `d` and ordered marker pair `e`, the log odds ratio is

`theta[d,e] = x[d] beta[,e] + u[d,e]`,

where `x[d]` is the four-column one-hot encoding of Donor, AMI, ICM, and NICM. Every etiology coefficient receives the same ridge penalty, so relabeling etiologies only relabels the fitted fields. For the informative-heart set `I[e]`, the fitted objective is

`sum_e { |I[e]|^-1 sum_d ell[d,e](x[d]^T beta[e] + u[d,e]) + eta/(2|I[e]|) sum_d u[d,e]^2 + lambda/2 ||beta[e]||^2 } + gamma/2 sum_h beta[h]^T L beta[h]`,

where `ell` is the Fisher noncentral-hypergeometric conditional negative log likelihood up to a data-only constant. The same positive `lambda` applies to all four etiologies, and `L` acts on every etiology field. The exact solver uses scaled-gradient tolerance `1e-8`, maximum factor condition number `1e12`, and at most 100 Newton iterations. A fit is returned only when all three certificates pass.

The RNA graph uses training-heart RNA detection profiles; the protein graph uses training-heart ADT profiles. Each marker profile is centered across training hearts and divided by its Euclidean norm. Other markers are ranked by `(Euclidean distance, marker symbol)`, and the first three are retained. The unweighted undirected adjacency contains an edge when either endpoint selects the other. Its combinatorial Laplacian is divided by its mean diagonal. With protein varying fastest in the ordered-pair axis, `L = (L_RNA kron I + I kron L_ADT)/2`, which also has mean diagonal one. A nonfinite or zero-norm profile is ineligible before graph construction.

The objective is strictly convex and has one finite minimizer. Recipient fields set donor deviation to zero and use `theta = tau x^T beta`. If `r` and `c` are the RNA-positive and ADT-high margins among `N=512` cells, `z=n11` ranges from `max(0,r+c-N)` to `min(r,c)` with mass proportional to `choose(r,z) choose(N-r,c-z) exp(theta z)`. Prediction is its log-sum-exp-computed expectation. A validation or held table with a degenerate margin has its unique margin-compatible prediction and zero deviance.

Leave-one-source-heart-out cross-validation selects deviation penalty `{0.3, 3}`, common etiology-coefficient ridge `{0.1, 1}`, graph penalty `{0, 0.01, 0.1, 1}`, and field transport `{0.75, 1, 1.25}`. Each fold repeats marker support filtering, ranking, and graph construction on its 13 training hearts. The validation heart contributes only its margins and unshifted truth to loss. Donor-equal deviance determines the configuration; ascending `(deviation penalty, coefficient ridge, graph penalty, field transport)` breaks an exact tie. The final axis and graph are rebuilt from all 14 source hearts. The graph-zero ablation is independently retuned over the remaining grid. The destroyed-link control independently tunes the complete grid after shifting only each fold's training ADTs. A graph-specific gain is claimed only if `gamma>0` is selected, source-CV mean loss is lower than graph zero, and the held paired-bootstrap 97.5th percentile for graph-minus-zero loss is below zero.

## Classical comparators

Every method receives the same source tables, selected entities, context labels, and held margins.

1. **Pooled fixed-interaction Poisson:** donor-specific row and column nuisance terms with one source-common interaction per entity.
2. **Etiology-specific fixed-interaction Poisson:** the same saturated log-linear model with a separate interaction in Donor, AMI, ICM, and NICM hearts.
3. **Standardized fixed-margin Pearson residual:** source interaction residuals under the Poisson independence model, averaged within etiology and inverted at recipient margins.
4. **Exact common-effect conditional field:** a single fixed-margin conditional interaction per entity.
5. **Fixed-margin independence.**
6. **Destroyed links:** the complete primary pipeline refitted after the deterministic within-heart ADT shift.

For row `a` and column `b`, pooled Poisson fits `log(mu[d,e,a,b]) = alpha[d,e] + rho[d,e] a + kappa[d,e] b + theta[e] ab`; the etiology-specific model replaces `theta[e]` by `theta[h(d),e]`. Fits maximize unpenalized Poisson likelihood on every training heart, profiling donor-specific nuisance parameters. Degenerate-margin tables remain in the likelihood and contribute no interaction information. At recipient margins, `tau theta` is fixed and nuisance parameters are refitted by the unique log-linear same-margin solution; maximum margin error must not exceed `1e-8`. This is a Poisson expected table, not a conditional-hypergeometric expectation. A nonfinite mandatory interaction or failed reconstruction is a terminal source refusal; no pseudocount is used.

For the standardized residual comparator, independence has `E = rc/N` and fixed-margin variance `V = rc(N-r)(N-c)/(N^2(N-1))` for `z=n11`. The source coordinate is `(z-E)/sqrt(V)`, with zero assigned to a degenerate table. Coordinates are averaged donor-equally within etiology. At recipient margins, `z` is `clip(E + tau s sqrt(V), max(0,r+c-N), min(r,c))`, and the other cells follow from the margins. The exact common-effect comparator is the unpenalized conditional maximum-likelihood log odds shared across source hearts. Each transportable comparator selects `{0.75,1,1.25}` independently by the same folds. The strongest remaining classical comparator is the minimum-loss method among standardized residual, exact common effect, and independence, with ties resolved in that order. Pooled Poisson, etiology-specific Poisson, that frozen winner, and destroyed links are mandatory gate comparators; every candidate loss is published before held access.

## Loss and uncertainty

For observed table `n`, prediction `mu`, and `N=512`, deviance is `D(n,mu) = 2/N sum_ab n[ab] log(n[ab]/mu[ab])`, with `0 log 0 = 0`; a positive observed cell with zero prediction is a terminal numerical failure. Heart loss is the arithmetic mean over ordered pairs, and study loss is the arithmetic mean over hearts. A 5% reduction means `1 - L_primary/L_comparator >= 0.05`. A heart favors the primary only under strictly lower loss.

For each comparison, each of 20,000 bootstrap draws resamples two hearts with replacement within each etiology and averages the resulting eight paired loss differences. Sampling uses `numpy.random.default_rng(21749401)`; linear 0.025 and 0.975 quantiles define the percentile interval. The etiology criterion requires a strictly negative mean difference within each two-heart stratum. The reported one-sided sign probability is `sum_{j=k}^n choose(n,j) 2^-n`, excluding exact-zero differences from `n`.

## Source promotion

Held access remains disabled unless all source reductions and final fits complete, at least nine markers survive in every fold and the final refit, and leave-one-heart-out mean deviance for the structured primary is below pooled fixed-interaction Poisson, etiology-specific fixed-interaction Poisson, the strongest remaining classical comparator, and destroyed links. At least ten of 14 source hearts must favor the primary against each, and mean improvement must be positive in all four etiologies for every comparison. Failure is terminal and produces no held prediction or score.

If promoted, the source artifact publishes the selected markers, graphs, hyperparameters, fitted etiology interaction fields for the primary and every comparator, source-fold losses, and hashes. It contains no held margin or assay value.

## Held confirmation

The primary passes only if it meets every criterion below against pooled fixed-interaction Poisson, etiology-specific fixed-interaction Poisson, the frozen strongest remaining classical comparator, and destroyed links:

- donor-equal mean deviance is at least 5% lower;
- the upper endpoint of a 20,000-draw etiology-stratified paired-donor bootstrap 95% interval for primary-minus-comparator deviance is below zero;
- at least seven of eight held hearts favor the primary; and
- mean improvement is positive in Donor, AMI, ICM, and NICM hearts.

The exact one-sided sign probability for the favorable-heart count is reported. Bootstrap seed `21749401` and all salts are fixed above. The graph-zero ablation, every named classical method, per-etiology losses, and per-heart losses are reported without changing the gate.

## Biological secondary analyses

Three fixed modules are evaluated when at least three listed markers pass the source support rule: endothelial `{PECAM1, CDH5, KDR, ENG, TEK}`, fibroblast/fibrosis `{FAP, LRRC15, PDGFRA, PDGFRB, THY1, CDH11}`, and myeloid `{CD14, FCGR1A, FCGR2A, FCGR3A, CSF1R, MRC1, FOLR2}`. Each module is fit on its complete ordered within-module pairs using the primary hyperparameters without retuning. Held donor-level deviance differences are tested by all `2^8` paired sign permutations; Benjamini--Hochberg correction spans every evaluable module-by-mandatory-comparator test. Effect estimates, confidence intervals, exact adjusted p-values, and non-evaluable modules are reported. These secondary tests cannot change the primary decision.

An exploratory relational summary compares the three nearest RNA-marker neighbors induced by Euclidean distance between rows of predicted and observed fixed-margin standardized-residual fields. It reports mean top-three Jaccard overlap across held hearts against 10,000 joint marker-label permutations from `numpy.random.default_rng(21749402)`. This summary is explicitly exploratory.

## Terminality and reporting

Schema, support, numerical, source-promotion, publication, acquisition, crash, and score failures are distinct terminal outcomes. Source matrix bodies are requested in sample order `2,4,7,8,13,15,17,27,28,29,30,32,33,41`; held bodies are requested in order `1,5,6,9,12,34,39,42`. Each `matrix.mtx.gz` body receives one streaming GET, with no Range request or automatic retry. A partial transfer, interruption, parse failure, unexpected exception, or process death consumes the stage and is terminal; recovery may publish its audit but may not reopen a matrix.

Every matrix must have the Matrix Market coordinate/integer/general banner, dimensions equal to its frozen feature and barcode axes, finite nonnegative integer entries, and no out-of-range index. Duplicate coordinates are accumulated independent of row order by a bounded-memory selected-row reducer with checked 64-bit sums and full gzip CRC exhaustion. A mismatch is terminal. The result records every request, observed byte count and SHA-256, declared and parsed entries, deletion, and whether held access occurred.

Before the first source body byte, an exclusive attempt claim is committed, tagged, pushed, and verified on the public remote. A durable consumption marker is created before its private capability is consumed. Held authorization additionally binds the public source-pass artifact, all frozen fields and hashes, protocol, runner, transitive estimator modules, and runtime before the first held body byte. This protocol cannot be rerun after any terminal source or held result. The aggregate benchmark retains the campaign whether it passes, refuses, or becomes infrastructure-unevaluable.
