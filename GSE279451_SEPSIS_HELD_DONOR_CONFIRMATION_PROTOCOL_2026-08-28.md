# GSE279451 held-donor exact conditional confirmation protocol

**Version:** 1.0, 28 August 2026  
**Candidate:** adult PBMC CITE-seq, `GSE279451`  
**Study:** Ye et al., *Nature Immunology* 27, 150-165 (2026), DOI
`10.1038/s41590-025-02345-x`  
**Outcome status:** disabled; no Matrix Market entry has been read

## Confirmatory question and family

The experiment asks whether donor-heterogeneity-aware exact conditional log
odds learned from 19 paired development donors predict same-cell RNA-protein
dependence in 21 donor-disjoint recipients better than the strongest matched
classical interaction-residual transfer. Every prediction is conditional on
the recipient's observed RNA and ADT margins. The target is cross-modal
dependence, not RNA abundance, protein abundance, cell composition, or disease
classification.

This is the next outcome-disabled candidate after the terminal SCMMIB BMMC
development refusal. BMMC is closed and cannot be repaired, rerun, promoted,
or revived as a backup. `GSE279451` may proceed only if no Sanger score-attempt,
held result, or held refusal exists. Any such Sanger artifact permanently
disables this candidate. The exact policy and the terminal BMMC artifact hash
are bound in `data/confirmation/gse279451_sepsis/family_policy_v1.json`.

## Source and zero-outcome preflight

GEO provides 40 sample-specific 10x bundles, each containing barcode TSV,
feature TSV, and feature-by-cell sparse integer MTX. Metadata identifies 3
adult healthy controls, 5 adult ICU controls, and 32 adults with sepsis: 14
abdominal, 14 respiratory, and 4 urinary infection-source samples. Nature
Supplementary Table 1 identifies the 40 adult CITE-seq sample names as 40
unique `Donor ID` values, and that set equals the 40 GEO sample names. One GSM
is therefore one biological donor. The workbook URL, byte count, SHA-256,
sheet, and two metadata columns used are bound in the source template; no
clinical outcome column entered the audit.

Before designation, only the GEO/Nature metadata, all barcode and feature axes,
and transfer headers were opened. The 40 axes contain 330,112 deposited cells,
37,487 RNA features, and 208 ADTs. All decompressed feature axes are identical,
SHA-256
`ff6a914dd33b3a3c2dd913ed439ed4b150fd8ab210595dec2447a283eb9b417b`.
No `matrix.mtx.gz` byte or entry was opened.

The complete tar is
`https://ftp.ncbi.nlm.nih.gov/geo/series/GSE279nnn/GSE279451/suppl/GSE279451_RAW.tar`,
2,334,412,800 bytes. Individual matrices total 2,319,077,199 compressed
bytes; axes total 15,230,404 bytes; the largest matrix is 85,929,901 bytes.
Acquisition and reduction must stream one donor at a time. Every acquired tar
member is byte-counted and SHA-256 hashed before parsing. The tar and a full
extraction must never coexist on this workspace.

Axes may be acquired and verified while outcomes remain disabled. Before the
first development MTX acquisition, the acquisition runner must exclusively
write `development_attempt_v1.json`, binding the axes and every executable
artifact. Any acquisition, integrity, support, numerical, reduction, or output
write failure after that marker writes the terminal development refusal,
deletes the current source MTX, and permanently forbids a rerun. A successful
run retains the attempt marker and binds its hash in the source and reduced
development artifacts.

The evaluator first hashes the raw reduced-artifact bytes without parsing them
and hashes the acquisition-attempt marker. It then exclusively writes
`evaluation_attempt_v1.json`, binding both hashes, before JSON parsing,
validation, cross-validation, or any numerical fit. Any parse, validation,
numerical availability, refit, serialization, or output-write failure after
that marker writes the terminal evaluation refusal and permanently forbids a
rerun. A completed finite cross-validation evaluation that simply misses a
promotion gate instead writes the terminal `DEVELOPMENT_FAIL` result; it is a
valid negative result, not an execution refusal, and likewise cannot be rerun.
A successful development result binds the evaluation-attempt hash.

## Immutable donor split

Within each metadata-only stratum `healthy`, `ICU`, `abdominal`, `respiratory`,
and `urinary`, accessions are sorted by ascending
`SHA256("gse279451-hierarchical-v1" + GSM accession)`. The first `floor(n/2)`
are development and the remainder are held.

| Role | Donors | Deposited cells | Healthy | ICU | Abd. | Resp. | Urinary |
|---|---:|---:|---:|---:|---:|---:|---:|
| Development | 19 | 159,540 | 1 | 2 | 7 | 7 | 2 |
| Held | 21 | 170,572 | 2 | 3 | 7 | 7 | 2 |

Development accessions are `GSM8571043`, `GSM8571044`, `GSM8571047`,
`GSM8571048`, `GSM8571049`, `GSM8571052`, `GSM8571055`, `GSM8571056`,
`GSM8571060`, `GSM8571061`, `GSM8571065`, `GSM8571068`, `GSM8571072`,
`GSM8571073`, `GSM8571074`, `GSM8571075`, `GSM8571077`, `GSM8571079`, and
`GSM8571081`.

Held accessions are `GSM8571042`, `GSM8571045`, `GSM8571046`, `GSM8571050`,
`GSM8571051`, `GSM8571053`, `GSM8571054`, `GSM8571057`, `GSM8571058`,
`GSM8571059`, `GSM8571062`, `GSM8571063`, `GSM8571064`, `GSM8571066`,
`GSM8571067`, `GSM8571069`, `GSM8571070`, `GSM8571071`, `GSM8571076`,
`GSM8571078`, and `GSM8571080`.

Clinical stratum, age, and sex may describe the sample and define the fixed
split. They do not enter fitting, weighting, model selection, or the primary
gate. Donors, not cells, markers, or clinical strata, are the units of
inference.

## Fixed panel, states, and tables

The panel is `CD4`, `CD7`, `CD14`, `CD19`, `CD33`, `CD38`, `CD44`, `CD47`,
and `CD52`. Each occurs exactly once on both modalities in every deposited
feature axis. `CD93`, present on RNA but absent from ADT, is excluded without
replacement. All 9 by 9 ordered RNA-marker by ADT-marker combinations are
retained, yielding 81 entities. No fuzzy alias, marker substitution, diagonal-
only analysis, or outcome-dependent panel repair is permitted.

The primary experiment uses exactly 1,024 cells per donor. Before any count is
opened, deposited barcodes are sorted by ascending SHA-256 of
`"GSE279451-CELL-BUDGET-v1" + accession + sample + barcode`; the first 1,024
are retained. Every donor has more than 5,500 deposited barcodes, so the rule
has no missing-donor branch. This outcome-independent budget prevents donor
assay depth from changing precision, support, or weighting and defines the
primary estimand. An all-deposited-cell analysis is permitted only after the
held pass/fail decision as a secondary sensitivity. It cannot trigger
reselection, repair a refusal, change a gate, or replace the primary result.

RNA state is one exactly when the raw UMI count is greater than zero. For each
donor and ADT marker, cells are ordered by raw ADT count, then by SHA-256 of
`"GSE279451-ADT-v1" + accession + barcode + marker`. Exactly 512 of the 1,024
cells are assigned the lower state and 512 the upper state. The rule
is per donor, never pooled, and uses no RNA value.

Each donor/entity yields a 2 by 2 table totaling exactly 1,024. An entity is margin-informative when
its fixed margins admit at least two tables. All 81 remain in the declared
panel; a donor is scoreable only when at least 64 are informative on the fixed
1,024-cell axis. Failure by
any development or held donor is terminal rather than handled by deleting the
donor or changing the panel.

No additional cell QC is applied in the primary experiment. The fixed 1,024
cells are selected directly from each deposited barcode axis by the hash rule
above. An empty, duplicate, malformed, or source-inconsistent axis is a terminal
integrity refusal, not a branch that filters cells or changes the budget.

## Primary estimator

The primary method is
`mapreg.hierarchical_conditional_coupling.fit_hierarchical_conditional_log_odds`.
For every ordered entity it fits donor-level log odds around a graph-regularized
population log odds by the exact conditional likelihood over fixed-margin 2 by
2 tables. No pseudocount enters an observed table, fitted likelihood, or held
prediction. Every fit must pass convergence, scaled-gradient, boundary-
recession, Schur-condition, and curvature-condition certificates.

RNA and ADT marker graphs are built anew inside each development fold using
only its 18 training donors. An RNA marker profile is its donor vector of
binary detection prevalences. An ADT marker profile is its donor vector of
mean per-cell `log1p(100 * marker count / panel ADT total)`, with a zero vector
for a cell whose nine-marker ADT total is zero. Each marker profile is centered
by its training-donor mean and divided by its sample standard deviation with
`ddof=1`. A zero-variance marker is a numerical refusal. Neighbors minimize
Euclidean distance between those standardized donor vectors. Directed
nearest-neighbor ties follow the locked marker order; the endpoint graph is the
undirected union of the directed neighborhoods, represented by one unweighted
two-endpoint incidence column per undirected edge in lexical marker order. These
marginal summaries contain no RNA--ADT cell pairing. Neighborhood size,
heterogeneity penalty, ridge penalty, graph penalty, and transport multiplier
are all selected jointly. The finite grid is fixed before matrix access:

- graph neighborhood size: `1, 2, 3`;
- heterogeneity penalty: `0.1, 1, 10`;
- ridge penalty: `0.01, 0.1`;
- graph penalty: `0.1, 0.3, 1`;
- transport multiplier: `0.75, 1, 1.25`.

Every hyperparameter is selected by 19-fold leave-one-development-donor-out
CV on the fixed 1,024 cells per donor. Each fold refits states, graphs, the primary estimator, every control, and
every residual candidate on 18 donors, then evaluates the omitted donor.
Loss is multinomial deviance per cell, averaged over informative entities
within donor and then over donors equally. Ties use the lexicographically
smallest tuple in the order listed above. After the development gate, the
selected configuration is refit once on all 19 donors.

## Classical head-to-head and controls

The strongest classical residual is selected within the same 19-fold CV from
signed Pearson and signed Poisson-deviance interactions, each raw or exactly
centered under its fixed-margin hypergeometric null, crossed with transport
multipliers `0.75, 1, 1.25`. For a source donor with `n` cells, the coordinate
is divided by `sqrt(n)` before donor-equal pooling over only donors whose
fixed-margin support has width greater than one; every entity requires at
least two such source donors. Width-one donors carry no coupling information
and are not entered as forced zero coordinates. At recipient size `m`, the
frozen coordinate is multiplied by `sqrt(m)` and inverted at the recipient's
fixed margins. The exact recipient null mean is restored only for a centered
candidate. This is the prespecified strongest residual head-to-head.

The residual coordinate is the one-degree-of-freedom signed square-root
Pearson chi-square or signed square-root Poisson deviance, with sign given by
the 2 by 2 determinant. Exact-null centering enumerates the integer
hypergeometric fixed-margin support. Prediction uses a continuous fixed-margin
inverse, not rounded counts: for row margins `(r0,r1)` and column margins
`(c0,c1)`, set `x=T00`, `L=max(0,r0-c1,c0-r1)`, `U=min(r0,c0)`, and
`T(x)=((x,r0-x),(c0-x,r1-c0+x))`. Extend the selected signed statistic to this
continuous table, evaluate its monotone attainable range at
`nextafter(L,U)` and `nextafter(U,L)`, clip the transported target coordinate to
that closed machine-interior range, and solve for `x` by exactly 128 bisection
steps. A centered candidate restores the recipient's enumerated exact-null
mean before clipping and inversion. No integer rounding, IPF floor, count
pseudocount, or truth-dependent clipping is permitted; every reconstructed
margin must agree to absolute tolerance `1e-10`.

The fixed controls are:

1. hierarchical ridge-only, with graph penalty zero;
2. common-effect exact conditional graph fit;
3. common-effect exact conditional ridge-only fit;
4. destroyed-link hierarchical graph fit after deterministic within-
   development-donor ADT barcode permutations preserving every margin;
5. label-permuted hierarchical graph with independent deterministic RNA and
   ADT marker-label permutations;
6. independence.

Control selection uses exactly the primary folds, losses, grids, and tie rule.
The destroyed-link control applies one common ADT-cell permutation per donor,
preserving every ADT marker margin and all within-ADT relationships while
breaking RNA--ADT pairing. The destroyed-link and label-permuted seeds are
fixed from accession, donor, marker, and the literal labels `destroyed-v1` and
`label-permuted-v1`.

For every method/comparator pair and donor, let `d` be primary deviance minus
comparator deviance, so negative values favor the primary. Relative reduction
is `(mean comparator deviance - mean primary deviance) / mean comparator
deviance`, with each mean donor-equal and a required strictly positive finite
denominator. Bootstrap intervals use one shared donor-index resample matrix per
phase from NumPy `default_rng(20260828)`: 20,000 draws of `D` donors with
replacement for `D=19` or `D=21`. The reported 95% percentile interval uses
`numpy.quantile(..., [0.025, 0.975], method="linear")`; the gate uses its 0.975
endpoint. A zero donor difference is not favorable.

## Development gate

The 19 held-in-turn development donors remain biological replicates. For each
of the strongest classical residual, destroyed-link hierarchical graph,
hierarchical ridge-only, and common-effect graph, the primary must have:

1. at least 5% lower donor-equal mean deviance;
2. a paired 20,000-draw donor-bootstrap 95% upper endpoint below zero; and
3. at least 15 of 19 donor loss differences below zero.

All four conjunctions must pass before the all-19-donor refit and prediction
artifact can be written. Common-effect ridge-only, label-permuted graph, and
independence are reported but are not additional development gates. A failed
development gate closes this candidate without held access.

## Held-access seal and terminal score

`predict` packages only the source identities, passing development result, and
all-19-donor frozen source models. It reads no held MTX byte, margin, state, or
table. The prediction JSON must be committed publicly. Held access requires an
authorization binding the prediction, runner, reducer, protocol, designation,
family policy, source manifest, and development result hashes to a 40-character
public Git commit and immutable GitHub blob URL.

The source, evaluation, prediction, authorization, and score artifacts also
bind the exact bytes of the hierarchical and heterogeneity estimators,
classical residuals, coupling-fields utilities, table reconstruction, and their
five focused test modules. Before the held attempt marker, the runner fully
reconstructs the expected prediction from the revalidated development result,
requires exact equality of design, selection, gate, model, and bindings, then
fetches the prediction bytes from the authorized 40-character GitHub commit.
The remote bytes and SHA-256 must be byte-identical to the local prediction;
network failure or mismatch refuses before any held attempt or MTX access.

`score` verifies family availability and every binding before writing an
exclusive terminal attempt marker. Only after that write may it acquire, hash,
and decode one held donor MTX at a time. It first forms RNA and ADT margins in
separate passes, materializes every method's prediction at those margins, and
records a prediction hash. Only then may paired RNA and ADT states coexist long
enough to form the 81 truth tables. Cell-level vectors are destroyed before
the next donor. Any exception after the attempt marker writes a terminal
refusal; there is no rerun. The refusal records the exception type, a
local-path/email-sanitized message, completed donor IDs, every acquired held
member byte count and hash, every completed prediction-materialization path
and hash, the current donor, and whether its source matrix was deleted. It
never serializes a cell vector or local absolute path.

The selected-barcode-axis digest is SHA-256 of the selected barcodes in the
locked hash order, each encoded as UTF-8 and terminated by `\n`, including the
final barcode. Development and held phases use this identical convention.

Poison tests must prove that a missing or altered authorization, Sanger
terminal artifact, missing BMMC terminal closure, held accession passed to the
development reducer, path traversal, symlink escape, wrong member hash, or an
attempt to form a held table before predictions all refuse before the relevant
MTX open.

## Held gate and reporting

Held deviance is averaged over informative entities within donor, then over
the 21 donors equally. A paired 20,000-draw donor bootstrap and exact one-sided
sign-flip test are computed on donor loss differences without refitting.

The sign-flip statistic is the donor-equal mean of the 21 differences `d`.
All `2^21` sign vectors are enumerated in binary lexical order, multiplying the
observed difference vector elementwise. The one-sided p-value is the fraction
of permuted means less than or equal to the observed mean. The tail is
inclusive; exact-zero differences remain zero under both signs and their
duplicate sign assignments remain in the denominator. No random or asymptotic
sign-flip approximation is permitted.

Confirmation passes only if, against both the strongest classical residual and
the destroyed-link control, the primary has:

1. at least 5% lower donor-equal mean deviance;
2. a paired-bootstrap 95% upper endpoint below zero;
3. at least 16 of 21 donors with lower deviance; and
4. exact one-sided sign-flip `p <= 0.025`.

The primary must also pass source/member integrity, split, state, support,
optimizer, margin reconstruction, pairing order, artifact hash, family, and
one-shot gates. Results against ridge-only, common-effect, label-permuted, and
independence controls are mandatory diagnostics. Every method/donor loss,
exclusion, numerical certificate, member hash, interval, and exact test is
reported regardless of direction. No clinical-stratum-specific promotion
claim is permitted.

For every held donor, the result records the full 81-position informative mask
in the frozen RNA-major, ADT-minor entity order and lists every excluded entity
by zero-based index, RNA marker, and ADT marker. The reported donor-equal loss
is therefore mechanically tied to the exact entity subset used for that donor.

## Present status

The protocol, narrative and JSON preflights, designation, family policy,
disabled source and authorization templates, acquisition, streaming reducer,
one-shot evaluator and scorer, and poison tests are complete and checksum-bound
in the public release working tree. No active source manifest, development
attempt, reduced development artifact, evaluation attempt, development result,
prediction commit, or score authorization exists. Outcome access remains
disabled.
