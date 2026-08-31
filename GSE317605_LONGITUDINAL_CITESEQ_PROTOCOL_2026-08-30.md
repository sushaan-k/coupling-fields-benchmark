# GSE317605 longitudinal CITE-seq held-patient protocol

## Status and question

This protocol is frozen before any GSE317605 count-matrix header or entry is
read. It tests whether a longitudinal hypergraph coupling field learned from
seven complete patients predicts patient-disjoint RNA--surface-protein
dependence during pembrolizumab plus GM-CSF treatment better than a tuned
classical time-conditioned Poisson interaction, fixed-margin independence,
destroyed links, and a retuned graph-zero exact-coupling ablation. The physical
patient is the inferential unit.

The candidate, machine-readable protocol, 84-pair sample manifest, and axis
access history are bound in:

- `data/confirmation/gse317605_longitudinal/candidate_designation_v1.json`;
- `data/confirmation/gse317605_longitudinal/protocol_v1.json`;
- `data/confirmation/gse317605_longitudinal/sample_manifest_v1.json`; and
- `data/confirmation/gse317605_longitudinal/axis_access_history_v1.json`.

## Pre-outcome access boundary

The pre-design audit read the GEO family SOFT, supplementary-file listing,
feature reference, GEX and ADT feature axes, and paired barcode axes. The audit
confirmed 33,538 ordered GEX features, 99 ordered ADT features, and exact
GEX--ADT barcode equality within all 84 patient--timepoint--replicate pairs.
The 84 replicate axes form 68 patient--timepoint visits; pooling technical
replicates yields 255--3,875 cells per visit, so all 68 exceed the frozen
192-cell budget. Replicate axes contain 29--3,846 rows. Fourteen contain
deposited duplicate barcode strings, totaling 17 rows beyond first occurrences
per modality, with at most two in one replicate axis. The manifest freezes the
row count, duplicate-row count, compressed-file SHA-256, and ordered-axis
SHA-256 for every one of the 168 barcode files.
No Matrix Market body, header, dimension line, or entry was requested. The
659,916,800-byte `GSE317605_RAW.tar` archive was not downloaded and is
permanently forbidden because it would collapse the sequential access
boundaries.

The audit supports the file universe, patient visits, marker coordinates, and
axis identities. It contains no assay value or outcome.

## Patient split

Fifteen patients contain T01--T04 in both modalities. They are sorted by

```text
SHA256("GSE317605-PATIENT-SPLIT-v1|COMPLETE|<integer patient id>")
```

and assigned without replacement:

| Role | Patients |
|---|---|
| Calibration | 23, 16, 14, 11, 10, 12, 17 |
| Pilot | 13, 19, 26 |
| Held complete | 24, 27, 22, 25, 15 |

The three remaining eligible patients are appended to held in integer order:
18 has T01--T03, 20 has T01, T02, and T04, and 21 has T01 and T03. No patient
crosses a role. Partial patients never enter selection or fitting.

## Cells and states

Technical replicates are pooled within patient and timepoint. A cell is the
tuple `(replicate, one-based deposited Matrix Market column, deposited barcode
string)`. The column retains distinct cells when deposited barcode strings are
duplicated. The 192-cell panel is the first 192 cells under ascending

```text
SHA256("GSE317605-CELL-v2\0<patient>\0<timepoint>\0<replicate>\0<one-based column>\0<barcode>")
```

RNA state is detected versus not detected. For each ADT marker within a
patient--timepoint panel, cells are ordered by descending count and then by

```text
SHA256("GSE317605-ADT-TIE-v2\0<patient>\0<timepoint>\0<marker>\0<replicate>\0<one-based column>\0<barcode>")
```

The first 96 cells are high and the remaining 96 are low. Thus every visit
contributes 256 ordered 2-by-2 RNA--ADT tables with identical cell totals and
nondegenerate ADT margins.

## Marker panel

Rows are one-based deposited Matrix Market coordinates.

| RNA | Ensembl | GEX row | ADT | ADT row |
|---|---|---:|---|---:|
| CD3E | ENSG00000198851 | 18652 | CD3 | 32 |
| CD4 | ENSG00000010610 | 20240 | CD4 | 31 |
| CD8A | ENSG00000153563 | 4006 | CD8 | 26 |
| MS4A1 | ENSG00000156738 | 17796 | CD20 | 7 |
| CD14 | ENSG00000170458 | 9636 | CD14 | 35 |
| FCGR3A | ENSG00000203747 | 2206 | CD16 | 50 |
| NCAM1 | ENSG00000149294 | 18577 | CD56 | 18 |
| IL7R | ENSG00000168685 | 8783 | CD127 | 27 |
| KLRD1 | ENSG00000134539 | 20343 | CD94 | 79 |
| SELL | ENSG00000188404 | 2300 | CD62L | 46 |
| CD27 | ENSG00000139193 | 20214 | CD27 | 24 |
| CD38 | ENSG00000004468 | 7358 | CD38 | 21 |
| ITGAM | ENSG00000169896 | 25807 | CD11b | 4 |
| CD2 | ENSG00000116824 | 1641 | CD2 | 28 |
| CD44 | ENSG00000026508 | 17511 | CD44 | 51 |
| CD7 | ENSG00000173762 | 28465 | CD7 | 40 |

The ordered GEX-axis SHA-256 is
`6bb91dd583b8ed7e4d6ea2efb6cb9b103b229573fec7ba2c1f7ba583994a21b1`;
the ordered ADT-axis SHA-256 is
`95797e25f128965db196858b0abf9a56487894431c5b792e54bbb53ccddfa1da`.

## Coupling field

Let `beta[t,e]` be the population log odds ratio for timepoint `t` and ordered
RNA--ADT pair `e`, and let `u[v,e]` be the deviation for donor--timepoint visit
`v`. Each informative table contributes its exact fixed-margin conditional
negative log likelihood at `beta[t(v),e] + u[v,e]`. The frozen objective is

```text
sum_t mean_{v: t(v)=t} [sum_e conditional_nll(v,e)
                        + eta/2 sum_e u[v,e]^2]
+ 0.01/2 sum_t,e beta[t,e]^2
+ gamma/2 sum_t beta[t]' L_pair beta[t]
+ tau/2 sum_t ||beta[t+1] - beta[t]||^2.
```

Conditional likelihood and deviation ridge therefore receive the same inverse
visit count within each timepoint. The beta ridge, hypergraph penalty, and
temporal penalty are each added once without visit-count scaling.

The fixed marker hypergraph has unit-weight overlapping edges:

- T: CD3E, CD4, CD8A, IL7R, SELL, CD27, CD2, CD7;
- B: MS4A1, CD27, CD38;
- NK: FCGR3A, NCAM1, KLRD1, CD2, CD7;
- myeloid: CD14, FCGR3A, ITGAM, CD44; and
- activation-memory: IL7R, SELL, CD27, CD38, CD44.

For incidence matrix `H`, unit edge weights `W`, vertex degrees `D_v`, and
edge sizes `D_e`, the normalized marker operator is

```text
L_H = I - D_v^(-1/2) H W D_e^(-1) H' D_v^(-1/2).
```

The ordered-pair operator is `L_pair = L_H tensor I + I tensor L_H`. Temporal
smoothing uses the unit-edge T01--T02--T03--T04 path. The direct optimizer
evaluates the exact conditional objective and analytic gradient; it returns
only with a finite convergence certificate.

The frozen grid is:

- deviation `eta` in `{0.1, 1, 10}`;
- marker hypergraph `gamma` in `{0, 0.05, 0.2}`;
- temporal `tau` in `{0, 0.05, 0.2}`; and
- transport multiplier in `{0, 0.5, 0.75, 1, 1.25}`.

The classical Poisson comparator crosses ridge
`{0.001, 0.01, 0.1, 1}` with the same five transport multipliers on the
identical leave-one-patient-out folds. Model selection compares exact
patient-equal cross-validation means without tolerance. Equal means are broken
by the ascending lexicographic tuple `(eta, gamma, tau, transport)` for every
exact-coupling selection and `(Poisson ridge, transport)` for Poisson.

Selection uses leave-one-physical-patient-out calibration. Hypergraph and
temporal penalties are rebuilt from no assay-derived graph: their operators
are fixed above. The graph-zero comparator sets `gamma=0` and reselects
`eta`, `tau`, and transport on the identical folds. A structure-zero
sensitivity sets `gamma=tau=0`; a temporal-zero sensitivity sets `tau=0` while
retaining the marker grid. Both reselect their remaining hyperparameters on the
same folds. A third sensitivity reconstructs target tables by the exact
fixed-margin conditional expectation using the interaction coefficients chosen
for the classical Poisson baseline. All three sensitivities are reported and
non-gating.

## Comparators and loss

The mandatory methods are:

1. the longitudinal hypergraph coupling field;
2. a donor-stratified, time-conditioned classical ridge Poisson log-linear
   interaction tuned over its frozen ridge-by-transport grid;
3. fixed-margin independence;
4. a destroyed-link field fitted after a one-cell cyclic ADT shift within
   each frozen visit axis; and
5. the retuned graph-zero exact-coupling ablation.

Every method receives the same training visits, time labels, target margins,
and 256 ordered pairs. Coupling-field methods reconstruct the exact
fixed-margin conditional expectation. The classical Poisson comparator instead
profiles its row and column nuisance terms through classical Poisson
log-linear reconstruction at each target margin; scoring it through the
conditional estimator is forbidden for the gating comparison. Loss is
multinomial deviance per selected cell, averaged across marker pairs within
visit, available visits within patient, and then patients equally.

## Gates

Calibration promotion requires the primary LOPO mean below every comparator,
at least 5% lower mean than both Poisson and destroyed link, at least five of
seven favorable patients against each comparator, and lower timepoint mean
than Poisson at three or more timepoints.

Pilot promotion requires all three patients favorable against Poisson and
destroyed link, at least 5% lower patient-equal mean than both, lower mean than
every comparator, and lower mean than Poisson at three or more timepoints.

The held decision uses eight patients. Against Poisson, graph-zero, and
destroyed link, it requires lower patient-equal mean, a completeness-stratified
exhaustive-resampling upper 95% endpoint below zero, at least seven favorable
patients, fixed-panel one-sided sign probability at most 0.05, and negative
mean difference in both complete and partial strata. It additionally requires
at least 5% reduction against Poisson and destroyed link, improvement over
Poisson at three or more timepoints, and lower overall mean than independence.
The resampling distribution exhausts all `5^5` ordered complete-patient and
`3^3` ordered partial-patient resamples with replacement and their Cartesian
product: 84,375 stratified resamples with no random seed. For each gating
comparator, the report gives two-sided 95% type-7 linear empirical percentile
intervals for paired additive loss difference and paired relative loss
reduction. The gate remains the upper endpoint below zero for additive loss
difference, defined as primary minus comparator. Relative reduction is
`(comparator - primary) / comparator`. Equality is nonfavorable. The sign
probability is the Binomial(8, 0.5) upper tail on the fixed panel.

## Irreversible execution

The public sequence is:

1. candidate and implementation freeze;
2. calibration GEX, then calibration ADT, then calibration result;
3. pilot GEX predictions, then pilot ADT and terminal pilot result;
4. fixed-hyperparameter refit on all ten complete source patients;
5. held GEX predictions, then held ADT and terminal held result.

Each stage first creates an exclusive attempt and header-only fsynced JSONL
journal. If claim construction fails before both are durable, it removes the
new capability and every partial claim artifact, fsyncs their parent
directories, and permits a fresh claim. This is the only rollback and precedes
publication or data access. A complete claim is published under its frozen
annotated attempt tag at the exact public origin. A private 32-byte capability
is then consumed into an exclusive, fsynced record and deleted before the first
GET.

After consumption, every exception produces one sanitized terminal refusal and
the stage cannot be retried. Public failures contain only an allowlisted code
and frozen hashes or counts, never exception text, traceback, response content,
URL, raw identifier, local path, or clinical text. Each designated file
receives one complete HTTPS GET without Range requests, retries, alternate
mirrors, or archive fallback. Every access and deletion event is fsynced. Each
matrix is preceded by exact ordered feature- and barcode-axis checks. Matrix
Market parsing validates and hashes the full compressed and decompressed
stream, including unselected entries. Integral `real general` counts are
accepted only by exact decimal parsing within int64.

A success result requires exactly one ordered `FILE_GET_STARTED`,
`FILE_GET_FINISHED`, and `FILE_DELETED` sequence for every designated feature,
barcode, and matrix file in the stage and no other GET. It reports expected and
observed file counts and binds the final journal hash. Before any dependent
stage, the result is committed under its frozen annotated result tag at the
exact public origin. Local and remote tag objects and peeled commits must agree;
tagged result bytes must match locally; and the result commit must descend from
the candidate freeze, implementation freeze, its own attempt, and every
prerequisite result. The machine-readable protocol fixes the attempt and result
tag names for every stage.

Pilot and held GEX stages write the selected cell identifiers and cell-level
RNA states to a mode-0600 bridge outside the repository. The public prediction
artifact binds that bridge by hash and byte count and exposes only state hashes
and marginal counts. No public artifact contains a raw cell identifier,
cell-level state vector, local filesystem path, or clinical free text.

Any durable consumption, result, or post-header journal refuses a rerun; a
published attempt is single-use. A failed calibration closes both pilot and
held access. A failed pilot closes held access. Held ADT is inaccessible until
held GEX predictions are public. The first complete held decision is terminal.
