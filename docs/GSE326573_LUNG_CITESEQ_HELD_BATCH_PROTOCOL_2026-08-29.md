# GSE326573 lung CITE-seq held-batch protocol

Frozen on 29 August 2026 after axis-only inspection of the official GEO
archive. The preflight opened H5 shapes, feature names, and feature types. It
did not open a barcode, sparse pointer, sparse index, or count value and did
not construct a table, association, prediction, or loss. The executable
contract is bound to candidate tag `gse326573-lung-v1-candidate`, peeled commit
`489ab69915428bc8f4f614a5ceaadbb9e1fb0fdf`, and the 326,922,240-byte archive
with SHA-256
`9bb9a1a879fba1f01406f4463e4b707f853c854a0cc2112a16f34dc86b0011fb`.

## Question and allocation

This one-shot experiment asks whether same-cell RNA--surface-protein
dependence learned from three source batches predicts joint states in three
held, donor-disjoint batches after the recipient margins are supplied. The
data are lung CD3-positive T-cell CITE-seq from controls and fibrotic lung
disease. The estimand is composition-inclusive association, not a causal or
cell-type-conditional effect.

The deposited batch order fixes the allocation. `Batch1`, `Batch2`, and
`Batch3` contain 20 source donors: controls 1--8, IPF 1--7, CTD-ILD 2, and
IPAF 1--4. Source selection uses three leave-one-batch-out folds, rebuilding
both feature graphs and refitting every model in each fold.

`Batch4`, `Batch5`, and `Batch6` contain nine held donors: CTD-ILD 1, 3, 4,
and 5; controls 9 and 10; NSIP; and IPF 10 and 11. CTD-ILD 3 has two deposited
matrices. Their losses are computed separately and averaged before donor-equal
aggregation or inference. Later matrices for CTD-ILD 2 and IPF 1 are excluded
because those donors occur in source. The candidate designation binds every
GSM, file name, byte count, SHA-256 digest, batch, and role.

## Fixed panel and states

The panel comprises 11 unambiguous RNA--ADT cognates, in fixed order:
`CD8A/CD8`, `NCAM1/CD56`, `CD4/CD4`, `IL2RA/CD25`, `PDCD1/PD1`,
`ITGAE/CD103`, `CCR7/CCR7`, `CTLA4/CTLA4`, `LAG3/LAG3`, `CD28/CD28`, and
`KLRG1/KLRG1`. Every RNA feature and one frozen TotalSeq alias must resolve
exactly once in every file. The analysis covers the complete 11-by-11 ordered
RNA-to-ADT panel, yielding 121 binary joint tables per sample. Marker removal,
replacement, fuzzy matching, and reordering are forbidden.

Each sample contributes 512 cells. Unique deposited barcodes are ranked by
SHA-256 of `GSE326573-CELL-BUDGET-v1|GSM|barcode`, then by barcode; the first
512 are selected and retained in deposited order. RNA state is raw UMI count
greater than zero. Within each ADT marker, cells are ordered by raw count,
SHA-256 of
`GSE326573-ADT-MIDRANK-v1|GSM|ordered-marker-index|barcode`, and barcode. The
lower 256 cells are low and the upper 256 are high. The ADT margin is therefore
fixed at 256/256 before held ADT access.

A table is informative when its margins admit more than one feasible 2-by-2
table. Each sample must supply at least 64 informative tables. The same mask
is used for every method; failure is terminal rather than repaired by changing
the panel or cell budget.

## Coupling field

The primary estimator fits finite full log odds by exact conditional
likelihood, conditioning each 2-by-2 table on its observed margins. Sample
effects shrink toward an 11-by-11 population field. Separate RNA and ADT
two-nearest-neighbor graphs regularize that field over their Cartesian
product. RNA profiles are sample-level positive fractions. ADT profiles are
sample means of `log1p` raw counts. Profiles are standardized across the
training samples, and both graphs are rebuilt within each source fold.
The graph-zero candidate uses identity incidences and does not construct a
profile graph.

The frozen grid crosses heterogeneity penalties `0.1, 1, 10`, ridge penalties
`0.01, 0.1`, graph penalties `0, 0.03, 0.3`, and transport multipliers
`0, 0.25, 0.5, 0.75, 1, 1.25, 1.5`, with two graph neighbors. Zero-valued graph and transport
penalties nest the unsmoothed and independence limits. The selected configuration
minimizes the equal mean of the three held-batch mean deviances. Exact ties are
resolved lexicographically in the parameter order just stated. Recipient
tables are the exact noncentral-hypergeometric expectations at the transported
finite log odds and recipient margins.

## Mandatory comparisons

The complete comparison set is:

1. signed Pearson and signed-root Poisson-deviance residual transfer, each
   calibrated over multipliers `0, 0.25, 0.5, 0.75, 1, 1.25, 1.5` in the same
   folds;
2. a common-effect stratified exact conditional maximum-likelihood log odds,
   with its multiplier selected on the same source folds;
3. a donor-pooled saturated 2-by-2 Poisson log-linear interaction, with its
   multiplier selected on the same source folds;
4. a pairing-destroyed coupling field fitted after a deterministic one-step
   cyclic shift of the complete ADT state vector along a salted cell order;
5. row-plus-column Poisson independence fixed by recipient margins.

The destroyed-link transformation preserves every ADT margin and the complete
within-ADT multivariate state distribution. Every method receives the same
source donors, folds, marker axes, held margins, informative-table masks, and
deviance. Failure of a mandatory fit closes the branch before held access.

## Source promotion

For observed table (T), prediction (P), and (N=512), per-table loss is

\[
  \frac{2}{N}\sum_{i,j}T_{ij}\log\frac{T_{ij}}{P_{ij}},
\]

with zero observed entries contributing zero. Sample loss averages the common
informative tables. Source selection averages samples within held-out batch,
then weights the three batch means equally.

Held numeric access requires all of the following:

1. all 20 source donors pass the 512-cell and 64-table support rules;
2. every primary and mandatory classical source comparison is finite in all
   three folds;
3. primary equal-batch mean loss is at least 5% below the calibrated residual;
4. at least 16 of 20 source donors favor the primary over that residual;
5. the primary-minus-residual mean is negative in every source batch;
6. primary loss is at least 5% below independence, at least 16 of 20 source
   units favor it, and its mean difference is negative in every source batch;
7. primary point loss is lower than the separately source-tuned common-effect
   conditional estimate; and
8. primary point loss is lower than the separately source-tuned pooled
   saturated-Poisson interaction.

Failure publishes a terminal source refusal. It cannot authorize held access.

## Prediction before pairing

After a public source pass, the prediction stage selects the frozen held cell
sets and decodes their 11 RNA rows. It combines the observed RNA margins with
the design-fixed 256/256 ADT margins and materializes every method's complete
held prediction. Because both modalities share one sparse H5 container,
routing RNA entries may inspect sparse pointers and feature indices and thus
reveal co-resident feature support. It cannot request or convert an ADT data
value, construct an ADT state, or form an RNA--ADT table.

Predictions, margins, selected-cell hashes, source-model hash, and the hash of
the private mode-0600 RNA state artifact are published before a separate score
authorization is created. The authorization binds those exact bytes. Scoring
rederives the held axes and RNA-state hashes; it is the first stage permitted
to read ADT values, construct high/low ADT states, or join the modalities.

## Held inference

The inferential units are the nine biological units. The two CTD-ILD 3 matrix
losses are averaged first. For each comparison, 20,000 batch-stratified paired
bootstrap draws use seed `20260829`, resampling three units within each held
batch; the reported interval is the percentile 95% interval for
primary-minus-comparator mean loss. An unstratified nine-unit bootstrap with
seed `20260830` is reported as a sensitivity analysis and does not enter a gate.
The formal transfer comparisons are
the source-selected calibrated residual, independence, and destroyed-link
control. Each must satisfy all five conditions:

1. at least 5% lower donor-equal mean loss;
2. batch-stratified bootstrap upper endpoint below zero;
3. at least eight of nine donors favorable;
4. one-sided exact donor sign-test (p\leq0.025); and
5. a negative mean difference in each of `Batch4`, `Batch5`, and `Batch6`.

Estimator-specific support is a separate, stronger conclusion. It requires a
lower primary point loss and bootstrap upper endpoint below zero against both
the source-tuned common-effect conditional estimate and the source-tuned
pooled saturated-Poisson interaction. These comparisons have no additional
5% or sign-count threshold. A result that passes transfer but not both
classical comparisons is labeled `TRANSFER_PASS_WITHOUT_CLASSICAL_INCREMENT`,
not a full estimator confirmation.

## Irreversible release

The immutable stage order is candidate, protocol, source attempt, source,
prediction attempt, prediction, score authorization, score attempt, and
result, under the `gse326573-lung-v1-*` tag family. Every stage commit descends
from its verified predecessor and binds the candidate, axis preflight,
protocol, runtime, runner, numerical modules, tests, inputs, access journal,
and prerequisite artifacts. A stage writes an exclusive claim before its
first permitted numeric read. Success, refusal, interruption, or unexpected
exception consumes the stage. No threshold, donor, marker, comparator,
endpoint, or hyperparameter can be changed, and every outcome enters the
public benchmark.

The executable sequence is:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
/usr/bin/python3 -m experiments.confirm_gse326573_lung claim-source
/usr/bin/python3 -m experiments.confirm_gse326573_lung source
/usr/bin/python3 -m experiments.confirm_gse326573_lung claim-prediction
/usr/bin/python3 -m experiments.confirm_gse326573_lung predict
/usr/bin/python3 -m experiments.confirm_gse326573_lung authorize-score
/usr/bin/python3 -m experiments.confirm_gse326573_lung claim-score
/usr/bin/python3 -m experiments.confirm_gse326573_lung score
```

Each claim or authorization artifact is committed, tagged, pushed, and verified
on the public remote before the following numeric command is run.
