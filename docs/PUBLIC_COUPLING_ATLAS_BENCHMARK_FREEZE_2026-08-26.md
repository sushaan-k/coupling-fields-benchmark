# Public paired-assay coupling benchmark

**Frozen:** 26 August 2026, before reading cell-level Frangieh or Papalexi
RNA/protein values for this analysis.

## Question

The benchmark tests whether a marginal-invariant association field recovers
pairing-dependent, perturbation-specific structure in public paired single-cell
assays. It does not test response completion, zero-shot perturbation prediction,
or a prospective dual-pulse assay.

For a paired finite-state table `P_g(U,V)` in group `g`, the canonical field is

\[
A_g=H_U\log P_g H_V,
\]

where `H_U` and `H_V` remove row and column means. Contrasts of these fields
remove all separable changes in the two marginal distributions. The primary
estimator jointly shrinks the resulting target-by-field matrix with a nuclear
penalty and a source-frozen biological hypergraph penalty. Penalties are chosen
on training-guide or development-replicate comparisons only.

## Locked public panels

### PerturbSci-Kinetics, GSE218566

- Paired views: pre-existing and nascent RNA in the same cell.
- Groups: 85 targets with three guides and at least 60 cells per guide.
- Evaluation: three held-guide rotations already frozen in
  `data/development/perturbsci_kinetics_gse218566/protocol_v1.json`.
- Primary endpoint: pooled Pearson correlation with held-guide association
  fields; standardized RMSE is co-primary.

### Frangieh-Izar Perturb-CITE-seq

- Paired views: transcript and surface-protein states in the same melanoma cell.
- Contexts: Control, IFN-gamma, and co-culture.
- Cell eligibility: non-targeting controls or cells with exactly one assigned
  perturbation; the common target universe must have at least 30 eligible cells
  in every context and evaluation split.
- State encoder: matched transcript/protein markers only; log-normalization,
  standardization, PCA, and four-state k-means are fitted on a deterministic
  calibration subset of non-targeting Control cells and those cells are removed
  from every score.
- Evaluation: sequence-distinct guide pools, with deterministic cell halves as
  a technical sensitivity analysis.
- Primary endpoint: reproducibility of target-by-context factorial coupling
  fields. IFN-gamma and co-culture are reported separately.

### Papalexi-Satija ECCITE-seq

- Paired views: transcript and surface-protein states in the same THP-1 cell.
- Groups: 25 genetic targets plus non-targeting controls.
- Replicates: rep1-tx, rep3-tx, and rep4-tx; the nearly empty rep2 samples are
  excluded before outcome access.
- State encoder: the four measured protein targets and their cognate RNA genes;
  three-state encoders are fitted on a deterministic subset of non-targeting
  cells and those cells are removed from scoring.
- Primary endpoint: leave-one-replicate-out recovery of target-versus-control
  coupling fields.

### PerturbFate

- Paired views: labeled and unlabeled RNA in the same cell.
- Factorial contrast: perturbation by vemurafenib.
- The previously frozen descriptive gate is retained as a transparent negative
  panel. No threshold or state representation may be changed to manufacture a
  positive result.

## Estimators and controls

Every comparison uses the same cells, state encoder, target universe, and
held-unit split.

1. Direct smoothed log-odds field for each target.
2. Independent scalar shrinkage selected on the same development units.
3. Nuclear shrinkage across target-by-field coordinates.
4. Biological-hypergraph smoothing alone.
5. Nuclear plus biological-hypergraph shrinkage (primary).
6. Endpoint marginal features with a matched linear readout.
7. Linear cross-covariance interaction.
8. Within-arm link destruction, preserving all group labels and both margins.
9. Degree-preserving or membership-permuted hypergraph control.

Reactome and CORUM memberships are filtered without using coupling outcomes.
The full method is promoted only if it improves held-unit error over direct,
independent, nuclear-only, graph-only, and endpoint baselines on at least two
positive public panels. Otherwise the simpler surviving estimator becomes the
reported method.

## Uncertainty and biology

- Resampling units are targets for held-guide summaries and deposited
  biological replicates when present; cells are never called biological
  replicates.
- Ninety-five percent intervals use 2,000 deterministic bootstrap draws.
- Pathway and complex enrichment uses the eligible target universe and
  Benjamini-Hochberg correction within each declared annotation family.
- A biological example is reported only if it reproduces in every held unit,
  survives multiplicity correction, and is absent after link destruction.
- Failed, unsupported, or non-identifiable panels remain in the benchmark
  table and are not moved out of the main results.

## Promotion rule

The public-data paper proceeds only if the estimator has a pairing-dependent
positive result in at least two assay families, a held biological-replicate
result in at least one family, calibrated synthetic behavior, and an explicit
refusal or negative panel. Otherwise the manuscript remains a technical report.

## Post-gate estimator repair

The first locked run is preserved as
`results/public_coupling_atlas_benchmark_v1_predeclared.json` (SHA-256
`bb8ac4e4f8f4a708956dc00bf2f271cc4a70b8a93cee3055164f98a8d5ac6dde`). The
predeclared nuclear-plus-scGPT-hypergraph estimator improved direct coupling
fields in PerturbSci and Frangieh but lost to the matched endpoint readout in
all three panels, so it failed the promotion rule.

After that failure was observed, one estimator repair was declared: model the
coupling field predicted from the two endpoint margins, then estimate the
pairing-specific residual in cycle space with the same nuclear-plus-hypergraph
penalty. This decomposition is evaluated as `marginal_residual_atlas`. It is a
post-development method and cannot use the three original panels as untouched
confirmation. Its promotion requires a win on an independent public panel
acquired after this declaration; GSE278572 is the designated confirmation
candidate if donor, restimulation, guide, RNA, and protein identities pass a
metadata-only eligibility audit.

## Finite-sample repair and independent-confirmation freeze

The first post-gate run exposed a failure of the destroyed-link control in the
Frangieh panel: fields computed after link destruction remained reproducible.
This is finite-table leakage, not recoverable pairing structure. Adding a
pseudocount before the log transform makes the empirical field depend on state
margins, and nonlinear transformation of sparse null tables adds a second
margin-dependent bias. The `marginal_residual_atlas` results on the three
development panels therefore remain diagnostic and are not confirmatory.

The production field is now conditionally centered. For each paired state
table, it subtracts the mean cycle-space field from 64 deterministic
within-arm permutations that preserve both empirical margins. A held-out
permutation, centered against the other null draws, is the matched
link-destruction control. This repair was specified after observing the
Frangieh control failure and before opening any GSE278572 expression or protein
outcome. The original panels are used only to choose one fixed structured
shrinkage setting.

The metadata-only GSE278572 audit passed. After the predeclared Souporcell
exclusion, the released same-cell object contains 100,086 final single-guide
HTO-singlet cells from two Souporcell-resolved human
donors, 28 targets plus non-targeting controls, Treg and Teff cultures, resting
and restimulated arms, RNA, and 130 biological proteins. Every target occurs in
all eight donor-by-cell-type-by-stimulation arms; the minimum arm has 36 cells.

The confirmation protocol is frozen as follows.

- Donor A is development; donor B is the held biological confirmation.
- Three-state RNA and protein encoders use matched RNA-protein markers and only
  deterministic donor-A non-targeting calibration cells. Those cells are
  removed from every donor-A score. Donor-B expression and protein values do
  not participate in encoder fitting or any other development decision.
- Within each donor and cell type, the estimand is the target-versus-control
  change in the restimulated-versus-resting conditionally centered field. Treg
  and Teff coordinates are concatenated.
- Targets require at least 30 non-calibration cells in every donor-by-cell-type-
  by-stimulation arm. This threshold was chosen from metadata support alone.
- The primary prediction is a precision-weighted structured estimate of the
  donor-A conditional field. For target (g) and field coordinate (j), the
  permutation-null variance (v_{gj}) is the sum of the four independent arm
  variances entering the factorial contrast. Observation weights are
  (w_{gj}=\operatorname{clip}[(1/v_{gj})/\operatorname{median}(1/v),0.05,20]).
  The estimator minimizes the resulting weighted squared loss plus a nuclear
  penalty and a source-frozen scGPT-hypergraph Laplacian penalty. The nuclear
  threshold is `0.1` times the leading singular value of the unshrunk donor-A
  matrix and the graph penalty is `5`. This formula and every parameter were
  fixed after the GSE277747 development result and without using a donor-B
  summary, state, field, or score.
- Comparators are direct donor-A fields, scalar and nuclear shrinkage,
  hypergraph-only shrinkage, gene2vec and membership-permuted hypergraphs, an
  endpoint-margin ridge readout, linear cross-covariance, and destroyed links.
- Success requires lower donor-B standardized RMSE than every matched
  comparator, a positive pooled field correlation, and a paired 2,000-draw
  target-bootstrap interval below zero for the primary-minus-best-comparator
  squared-error difference. Donors are reported as `n=2`; target resampling is
  not described as population-level donor inference.

Two deviations in the original development implementation are retained in the
record: Frangieh used three, not four, states and required 20, not 30, cells per
guide and context. Those panels are consequently development evidence only.

### Independent RNA-ATAC confirmation

GSE277747 was acquired after the conditional-centering repair. Metadata and
object structure, but no RNA-ATAC association outcome, were inspected before
this protocol was fixed. The release contains 121,651 same-nucleus RNA-ATAC
profiles with guide identity; mouse non-targeting spike-in nuclei are excluded.

- A deterministic 20% subset of human non-targeting nuclei fits the state
  encoders and is removed from all scores.
- The RNA encoder uses the 256 most variable detected human genes in control
  cells. The ATAC encoder uses the 1,024 human peaks with highest Bernoulli
  variance in control cells. Each view is normalized within cell, standardized
  from controls, reduced to six principal components, and partitioned into
  three states.
- A target enters the held-guide benchmark only when two distinct guide
  sequences each retain at least 30 non-calibration nuclei. The two guide
  directions are scored symmetrically; when more than two guides qualify, the
  two with greatest metadata-only cell support are retained.
- The conditionally centered RNA-ATAC field is the outcome. Structured
  penalties remain fixed from the pre-existing development panels. Comparators
  and target-bootstrap uncertainty match the GSE278572 protocol.
- This is guide replication in one BT16 culture, not a biological-replicate or
  cross-context result.

Before any target-wise RNA-ATAC field was computed, the control-only encoder
adequacy check found that unconstrained k-means placed fewer than 1% of control
nuclei in one state in each modality. That representation fails the support
contract for a 3-by-3 field. The frozen encoder therefore uses control-fitted
tertiles of the first principal component in each modality; all six components
remain available for sensitivity analyses. This support repair used only the
non-targeting calibration cells and was fixed before target outcomes were
opened.

The corrected development run supported only PerturbSci; Frangieh and Papalexi
were assigned to refusal rather than allowed to tune a zero predictor. The
fixed structured setting for both independent confirmations is therefore the
modal PerturbSci choice: nuclear threshold `0.1` times the leading singular
value and scGPT-hypergraph penalty `5`. Endpoint ridge uses `alpha=0.1`.

GSE277747 was then scored under its locked held-guide protocol. Neither the
unweighted structured estimate nor the conditional field itself reproduced
across guides; the panel is retained as a refusal and is not used as positive
evidence. Before any donor-B outcome was computed, this result motivated one
final finite-sample change: the fixed inverse-permutation-variance weights
defined above. No state representation, penalty, threshold, comparator, or
success criterion was changed. GSE278572 donor B is the first held
biological-replicate evaluation of this final estimator.

During the initial all-cell matrix stream, donor-B raw counts entered an
in-memory buffer before a code correction made the implementation match the
already written 64-reference-permutation rule. The stream was interrupted;
no donor-B cache, encoder, field, statistic, score, or human-visible value was
produced. The corrected estimator was therefore fixed without a donor-B
summary or outcome, although donor B was not literally unopened at the byte
level. The aborted lock is preserved with the final run provenance.
