# GSE309593 held-batch confirmation protocol

Frozen on 29 August 2026 after axis-only inspection of the deposited RNA H5
and ADT CSV schemas. The preflight read H5 group names, RNA feature names and
types, ADT feature names, file sizes, and available file digests. It read no
barcode, cell identifier, RNA count, ADT count, joint table, association, or
loss. The executable contract binds the v1 candidate designation, its v2
non-temporal amendment, the corrected axis preflight, and the metadata record
for every subject and public file. The protocol binds the final runner and test
digests before it is committed or tagged. The protocol tag's peeled commit is
then derived from the verified local and public annotated tag because an
artifact cannot embed the commit that first contains its own final bytes.

## Question and allocation

This one-shot experiment asks whether same-cell RNA--surface-protein
dependence learned from four source batches predicts joint states in three
held, subject-disjoint batches after recipient margins are supplied. The data
are pretreatment multiple-myeloma bone-marrow samples. The estimand is
composition-inclusive association, not a causal or cell-type-conditional
effect.

The split is determined by deposited numeric batch labels, not acquisition
time. `B092`, `B099`, `B110`, and `B129` contain 14 source subjects:
`FH1001`--`FH1012` except `FH1013`, together with `FH1014` and `FH1017`.
Source selection uses four leave-one-batch-out folds. Every graph and model is
rebuilt from the three training batches in each fold.

`B162`, `B208`, and `B210` contain nine held subjects: `FH1016`, `FH1020`,
`FH1021`, `FH1022`, `FH1023`, `FH1024`, `FH1026`, `FH1027`, and `FH1028`.
The held batch sizes are three, two, and four. Subjects and batches are
disjoint. The v2 amendment corrects the earlier temporal description without
changing an allocation, file, endpoint, threshold, or decision rule.

## Public freeze chain

The candidate, amendment, and axis-preflight tags are annotated and frozen to
peeled commits `4356e571de915576f8dbc3a1d3943049abc560c6`,
`3aba63b1e1575790750cbb7aeda2f083492f7a8f`, and
`671882853f932c407288afa3728d1b1389f8168d`. Before a stage can be claimed,
the runner verifies each local and public-remote tag object and peeled commit,
compares the required artifact bytes with that tag, derives the protocol
commit from its annotated tag, and proves Git ancestry from candidate through
amendment, axis preflight, and protocol. Later stage tags must form the same
verified ancestry chain.

## Input boundary

Each RNA and ADT artifact is downloaded separately to scratch storage outside
the repository. Its bound file name and byte count are required, a SHA-256 is
computed and published, any preflight digest is verified, and the assay bytes
are deleted after reduction.

RNA H5 access is restricted to:

```text
matrix/barcodes
matrix/features/name
matrix/features/feature_type
matrix/data
matrix/indices
matrix/indptr
matrix/shape
```

The complete H5 container is transferred opaquely to scratch and hashed. Only
the listed datasets may be dereferenced, decompressed, or decoded. Each
reduced H5 publishes the sorted unique set of decoded dataset paths, which
must equal the seven-path allowlist exactly. This is a set certificate, not a
per-slice journal. The H5 groups `ADT`, `hash`, `pca`, and `well`, including
all descendants, cannot be dereferenced. Embedded H5 protein values are never
inputs; protein counts come only from the separately deposited ADT CSV files.
Source may parse source ADT identifiers and the frozen marker rows or columns.
Held ADT identifiers and values remain inaccessible until public score
authorization.

Every claim validates the exact CPython executable and version; NumPy, SciPy,
and h5py versions; HDF5 runtime, built-against, and h5py API tuples; operating
system and architecture; and the three single-thread environment variables.
The HDF5 runtime and built-against tuples are checked separately.

## Fixed panel and states

The panel comprises all 24 exact RNA--ADT mappings supported across the 23
files, in frozen candidate order:

```text
CD1C/CD1c       CD2/CD2         CD4/CD4         CD7/CD7
CD8A/CD8        ITGAM/CD11b     ITGAX/CD11c     CD14/CD14
CD19/CD19       MS4A1/CD20      CD22/CD22       CD27/CD27
CD33/CD33       CD34/CD34       CD36/CD36       CD38/CD38
CD40/CD40       CD47/CD47       FCGR1A/CD64     CD69/CD69
CD80/CD80       CD86/CD86       CD163/CD163     CX3CR1/CX3CR1
```

Each feature must resolve by exact, case-sensitive equality exactly once.
Aliasing, fuzzy matching, case folding, substitution, aggregation, marker
removal, and reordering are forbidden. The analysis covers the complete
24-by-24 ordered RNA-to-ADT panel, yielding 576 binary joint tables per
subject.

Each subject contributes 512 cells. RNA-QC-eligible cells have at least 200
detected genes, mitochondrial UMI fraction at most 0.10, and at most 70,000
RNA UMIs. Eligible barcodes are ranked by SHA-256 of
`GSE309593-HELD-BATCH-CELL-BUDGET-v1|subject|barcode`, then by barcode. The
first 512 are retained in deposited H5 order. RNA state is raw UMI count
greater than zero.

For each ADT marker, the same cells are ranked by raw count, SHA-256 of
`GSE309593-HELD-BATCH-ADT-MIDRANK-v1|subject|ordered-marker-index|barcode`,
and barcode. The lower and upper 256 cells receive rank-copula states 0 and 1.
This fixes every ADT rank margin at 256/256. Salted ordering resolves count
ties; the resulting halves are not measured biological low/high states when
raw counts tie. Every selected RNA barcode must resolve exactly once in the
matching ADT file.

A table is informative when its margins permit more than one feasible 2-by-2
table. Every scored method uses the same frozen source comparison mask
intersected with the subject's informative tables. Held scoring also removes
all 24 RNA pairs for any ADT marker with fewer than two distinct raw values in
that subject. At least 64 coordinates must remain. Support failure is terminal
and cannot be repaired by changing the panel or cell budget.

## Source-only comparison mask

The source stage freezes a single comparison axis before model selection. A
coordinate is retained only if every leave-one-source-batch-out training set
and the final 14-subject source set satisfy three conditions: at least two
subjects have informative margins and at least two distinct raw values for the
coordinate's ADT marker; the aggregate observed free-cell count is strictly
interior to the summed fixed-margin support; and all four cells are positive
after pooling every training-subject table whose raw ADT marker varies,
including fixed-margin-degenerate strata. These conditions guarantee a finite
unpenalized stratified common-effect estimate and finite saturated pooled
Poisson interaction. A boundary coordinate is excluded, never credited as an
estimator win.

At least 288 of 576 coordinates must remain. Each source subject must retain at
least 64 coordinates after the mask is intersected with its informative-margin
and raw-ADT-variation masks. Each held subject before authorization must have
at least 64 potentially scored coordinates from its informative margins. The
source artifact
publishes the ordered mask, retained indices and labels, fold and final support
counts, subject support counts, and a canonical hash. The hash is SHA-256 over
the dtype string `|u1`, int64 shape `[24,24]`, and the 576 C-order bytes of the
uint8 mask. Held data cannot change this axis.

## Coupling field

The primary estimator fits finite full log odds using the exact conditional
likelihood of each 2-by-2 table given its margins. Subject effects shrink
toward a 24-by-24 population field. Separate RNA and ADT feature graphs
regularize that field over their Cartesian product. RNA graph profiles are
subject-level positive fractions; ADT graph profiles are subject means of
`log1p` raw counts. Profiles are standardized on training subjects, and both
two-nearest-neighbor graphs are rebuilt inside each source fold. A zero graph
penalty uses identity incidences and constructs no profile graph.

The primary is fitted on the full 576-coordinate panel and requires at least
one informative source subject per coordinate. The residual and log-linear
comparators are fitted only on the frozen source comparison mask. Every model,
including the primary, is evaluated on the same mask intersected with the
subject's eligible coordinates.

The grid crosses heterogeneity penalties `0.1, 1, 10`, ridge penalties
`0.01, 0.1`, graph penalties `0, 0.03, 0.3`, and transport multipliers
`0, 0.25, 0.5, 0.75, 1, 1.25, 1.5`. A configuration with an incomplete fold
is ineligible, and every refusal is published. At least one primary
configuration must complete all four folds. Selection minimizes the equal
mean of the four held-out batch mean deviances; exact ties are resolved
lexicographically in the parameter order just stated. The selected final fit
must be valid. Recipient tables are exact noncentral-hypergeometric
expectations at the transported finite log odds and recipient margins.

## Mandatory comparisons

The comparison set is fixed before source numeric access:

1. signed Pearson and signed-root Poisson-deviance residual transfer, each
   calibrated over the seven transport multipliers in the same folds;
2. the primary classical comparator, a stratified common-effect exact
   conditional log odds fitted independently for each retained table by a
   deterministic bracketed score root, with the same multiplier grid;
3. a secondary saturated 2-by-2 Poisson log-linear interaction for each
   retained table, obtained after pooling every training-subject table whose
   raw ADT marker varies, including fixed-margin-degenerate strata, with the
   same multiplier grid;
4. an alpha-calibrated fixed-structure pairing ablation fitted after a one-step
   cyclic shift of each source subject's complete 24-marker ADT state vector
   along the salted cell order; and
5. row-plus-column Poisson independence fixed by recipient margins.

For residual transfer, the signed Pearson statistic or signed square root of
Poisson-independence deviance is divided by `sqrt(512)` before averaging over
informative training subjects. After multiplication by the calibrated alpha
and `sqrt(512)`, the statistic is inverted monotonically on the recipient
fixed-margin interval and clipped only to its attainable range. Each residual
family must retain at least one complete four-fold candidate.

Every multiplier for the common-effect and pooled-Poisson comparators must be
valid in every fold. The common-effect solver accepts only an interior finite
maximum with positive observed information and a certified score residual; it
uses no continuity correction, penalty, or multivariate optimizer. All
selected and final fits must be valid.

Every pooled table must have four positive cells. In each source fold and the
final fit, its interaction and pooled margins must reconstruct every retained
table with maximum absolute cell error divided by pooled total count at most
`1e-8`. The source artifact publishes per-ADT-marker pooled-subject counts, the
pooled-table array hash, the maximum normalized reconstruction error, and the
pass flag for each fold and final fit.

The destroyed-link transformation preserves each ADT margin and the full
within-ADT multivariate state distribution. Its graph structure,
graph-neighbor count, heterogeneity penalty, ridge penalty, and graph penalty
are fixed at the real-primary selection. Each fold refits destroyed training
tables and scores real validation truth on the common mask; only its transport
alpha is selected over the seven-point grid. Refusals and losses are
published. This is a fixed-structure ablation, not an independently tuned
competitor. Every method receives the same subjects, folds, recipient margins,
score masks, and deviance.

## Source promotion

For observed table (T), prediction (P), and (N=512), per-table loss is

\[
  \frac{2}{N}\sum_{i,j}T_{ij}\log\frac{T_{ij}}{P_{ij}},
\]

with zero observed entries contributing zero. Subject loss averages the
source comparison mask intersected with that subject's informative-margin and
raw-ADT-variation support.
Source selection averages subjects within each held-out batch and then weights
the four batch means equally.

Held numeric assay access requires all of the following:

1. the source-only comparison mask retains at least 288 coordinates and its
   hash and fold/final support certificates reproduce;
2. all 14 source subjects pass the 512-cell and masked 64-table support rules;
3. complete-case tuning and every mandatory selected or final fit satisfy the
   numerical-validity rules above, including a complete separately
   alpha-calibrated destroyed-link ablation and pooled-Poisson fold/final
   reconstruction error at most `1e-8`;
4. primary equal-batch mean loss is at least 5% below the selected residual;
5. at least 12 of 14 source subjects favor the primary over that residual;
6. the primary-minus-residual mean is negative in each source batch;
7. primary loss is at least 5% below independence;
8. at least 12 of 14 source subjects favor the primary over independence;
9. the primary-minus-independence mean is negative in each source batch; and
10. primary point loss is lower than both the separately source-tuned
   coordinatewise common-effect estimate and pooled saturated-Poisson
   interaction.

Failure publishes a terminal source-gate refusal and cannot authorize held RNA
or ADT numeric access.

## Four-stage firewall

The immutable execution order is `source`, `prediction`,
`score-authorization`, and `score`.

The source stage reads numeric RNA and ADT values only for the 14 source
subjects. It publishes the comparison mask and hash, every support
certificate, configuration refusal, fitted model, source loss, decision
component, and access certificate. Only a public source pass can authorize
prediction.

Prediction downloads only the nine held RNA H5 files under the RNA-only
allowlist. It selects cells, computes 24 RNA margins, combines them with the
design-fixed 256/256 ADT rank margins, and materializes every method's
predictions. Each held subject must have at least 64 potentially scored
coordinates after the source mask is intersected with its informative RNA
margins.
It does not download, open, range, parse, or hash a held ADT CSV and never
dereferences an H5 ADT dataset. Selected barcode axes and binary RNA states are
stored in a mode-0600 private artifact ignored by Git; only its hash and byte
count are published. Before authorization, the runner validates every frozen
margin, comparison-mask binding, method family, prediction array, array hash,
axis hash, and support count for all nine subjects.

Score authorization is a distinct public nonnumeric stage. It binds the
verified prediction-tag commit; source and prediction artifacts; candidate,
amendment, metadata, and preflight; protocol and runtime; disabled template;
Git ignore rule; runner and tests; every transitive numerical module; H5
dataset-set certificate; and private RNA-state hash. The disabled template cannot
authorize access. The completed authorization is committed, tagged, pushed,
and verified before score is claimed.

Score is the first stage permitted to open held ADT CSV identifiers or values,
construct held ADT states, or join held RNA and ADT states. It verifies every
bound artifact and rechecks each selected-cell axis before forming a joint
table. A marker with fewer than two distinct raw ADT values removes its full
24-coordinate column block from that subject's score mask. The final mask and
hash are published, applied identically to every method, and must retain at
least 64 coordinates.

Before opening a new access class, the runner validates the predecessor
artifact's subject, GSM, and batch axes; every named count and zero-access
assertion; every per-subject H5 dataset-set certificate; and the private-state
binding, margin, method family, prediction array, and array hash field-by-field.
Matching the artifact hash alone is insufficient. Every terminal payload,
including an unexpected exception or interrupted download, retains the
incremental nonnumeric audit accumulated before failure: file
requests, completed hashes, reductions, deletions, decode-start counters,
unique allowlisted H5 paths for which a read or length operation started,
modality counters, and whether any identifier, numeric value, state, or joint
table was formed. A started H5 operation is not certified as a completed
decode. The audit never contains raw identifiers, values, states, or tables.

## Held inference

The inferential units are the nine held subjects. Point estimates average
their losses equally. For each comparison, 20,000 batch-stratified paired
bootstrap draws use seed `20260829`: each draw resamples three subjects within
`B162`, two within `B208`, and four within `B210`. The reported interval is
the percentile 95% interval for primary-minus-comparator mean loss, computed
with NumPy `default_rng` and linear empirical quantiles. An
unstratified nine-subject bootstrap with seed `20260830` is reported as a
sensitivity analysis and does not enter a gate.

The formal transfer comparisons are the selected residual, independence, and
destroyed link. Each must satisfy all five conditions:

1. at least 5% lower subject-equal mean loss;
2. batch-stratified bootstrap upper endpoint below zero;
3. at least eight of nine subjects favorable;
4. one-sided exact subject sign-test `p <= 0.025`; and
5. a negative mean difference in each held batch.

Estimator-specific support additionally requires a lower primary point loss
and bootstrap upper endpoint below zero against both source-tuned classical
interactions: coordinatewise common-effect conditional estimation and pooled
saturated Poisson. These two comparisons have no additional 5% or sign-count
threshold. Passing transfer without both classical comparisons yields
`TRANSFER_PASS_WITHOUT_CLASSICAL_INCREMENT`; passing all five comparisons
yields `CONFIRMATION_PASS_WITH_CLASSICAL_INCREMENT`; every other completed
score yields `CONFIRMATION_FAIL`.

Inference is conditional on the frozen subject split, source-only comparison
mask, source-selected configurations, and nine observed held subjects.
Intervals and sign tests are comparisonwise and do not provide familywise or
campaign-wide error control across the public benchmark. Exact zero subject
differences are excluded from the one-sided binomial sign-test denominator;
the test assumes independent subject signs under its null.

## Irreversible release

Each numeric stage writes an exclusive attempt artifact and publishes its tag
before its first newly permitted numeric read. Authorization is claimed and
published before it can enable score. A success, refusal, interruption, or
unexpected exception consumes the stage. No subject, batch, marker, threshold,
comparator, endpoint, model family, or hyperparameter can change. Every source
refusal, prediction refusal, authorization failure, score refusal, completed
negative result, transfer-only pass, or full pass enters the public benchmark
and evidence ledger.

The earlier `gse309593-independent-study-v1-protocol` branch is separate. It
closed when its GSE288020 source authorization failed, before any GSE309593
assay file or identifier was accessed. This within-study held-batch experiment
does not rescue, rerun, amend, or replace that cross-study branch.

The executable sequence is:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
/usr/bin/python3 -m experiments.confirm_gse309593_held_batches claim-source
/usr/bin/python3 -m experiments.confirm_gse309593_held_batches source
/usr/bin/python3 -m experiments.confirm_gse309593_held_batches claim-prediction
/usr/bin/python3 -m experiments.confirm_gse309593_held_batches predict
/usr/bin/python3 -m experiments.confirm_gse309593_held_batches claim-score-authorization
/usr/bin/python3 -m experiments.confirm_gse309593_held_batches authorize-score
/usr/bin/python3 -m experiments.confirm_gse309593_held_batches claim-score
/usr/bin/python3 -m experiments.confirm_gse309593_held_batches score
```

Each claim or authorization artifact is committed, tagged, pushed, and
verified on the public remote before the following command runs.
