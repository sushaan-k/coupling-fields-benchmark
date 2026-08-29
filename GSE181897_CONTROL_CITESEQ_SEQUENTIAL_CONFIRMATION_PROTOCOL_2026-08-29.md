# GSE181897 control CITE-seq sequential confirmation protocol, version 1.1

Frozen from metadata on 29 August 2026, before any AnnData `X` value was read.
The machine-readable designation and protocol are normative; this document
summarizes their design and decision rules.

Version 1.1 is a pre-outcome descendant amendment. The original version 1.0
remains immutable at annotated tag
`gse181897-control-citeseq-v1-candidate`, commit
`b69d19da98aeba46880dcb62082e57daa73a82c9`, with protocol SHA-256
`ec2501565dc55df02a8c48dd0c2955a90abd8b91083a600a305334a09d1074ff`.
This amendment fixes deterministic Stage-B, neighbor, and sign-test tie rules,
discloses CSR index scanning, specifies supported-table pooling for Poisson,
adds the source-attempt claim, and requires
the version 2 axis artifact with explicit obs and var uniqueness certificates.
It does not alter any cohort, panel, state, mask floor, grid, loss, comparator,
gate, or decision threshold. At amendment, the first axis preflight had read
metadata and HDF5 dataset shapes only: zero `/X/data`, `/X/indices`, and
`/X/indptr` entries had been read. Numeric access remains closed until
`axis_preflight_v2.json` has schema `gse181897-axis-preflight/1.1`, status
`AXES_FROZEN_UNIQUE_X_NUMERIC_UNREAD`, and exact unique-index certificates for
136,142 obs rows and 20,399 var rows.

## Scope

[GSE181897](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE181897)
contains population-scale CITE-seq of 64 PBMC donors distributed across 12
physical pools and six conditions. This experiment uses only unstimulated
aliquots with exact metadata value `cond == "C"`. It tests transfer of
composition-inclusive RNA-surface-protein dependence across donors and pools.
It does not estimate cell-type-specific coupling, stimulus response, or a
causal effect.

The development, internal-validation, and confirmation control aliquots are
donor-disjoint and physical-pool-disjoint. They are not whole-file-disjoint.
All conditions share one H5AD, and non-control aliquots from sealed donors can
occur in other pools. Those rows are never analyzed. Access control therefore
applies to decoded H5AD rows and features, not possession of the opaque public
file.

## Metadata freeze

The raw H5AD is `GSE181897_concat.4.raw.h5ad.gz`, with expected compressed and
uncompressed sizes of 1,011,162,509 and 3,063,713,137 bytes. The immutable
version 1 preflight recorded both SHA-256 digests. They are known but do not
authorize numeric access: version 2 must recertify and publicly bind those
digests, the exact HDF5 paths, CSR encoding, unique obs and var axes, and axis
hashes after the implementation freeze.

The metadata expectations are:

- H5AD shape: 136,142 by 20,399; both axes have unique identifiers.
- Feature composition: 20,303 `GRCh38` genes and 96 `BD99AbSeq` proteins.
- Required obs columns: `cond`, `exp_id`, `free_id`, and `batch`.
- `cond == "C"`: 22,732 cells and 64 donors.
- `cond == "0"`: a documented 455-cell mispool anomaly, always excluded.
- `exp_id` and `free_id`: a donor-level bijection among retained controls.
- `batch`: the integer axis 0 through 11.

`exp_id == 0` is a valid internal-validation donor and is unrelated to the
excluded string condition `cond == "0"`.

## Cohorts

Each retained donor must have at least 128 control cells. Four donors are
excluded by this rule: `exp_id` 23 (2 cells), 62 (64), 51 (70), and 52 (91).
No other exclusion or replacement is permitted. Donors are serialized by
batch and then integer `exp_id`.

| Stage | Pools | Donors |
|---|---:|---|
| Development | 0 | 34, 35, 45, 57, 58 |
| Development | 1 | 18, 22, 24, 55, 59, 61 |
| Development | 2 | 13, 29, 31, 42, 43 |
| Development | 3 | 5, 12, 30, 36, 60, 63 |
| Development | 4 | 10, 14, 38, 47 |
| Development | 5 | 11, 15, 25, 39, 49 |
| Development | 6 | 1, 4, 7, 27 |
| Development | 7 | 9, 32, 37, 44 |
| Internal validation | 8 | 3, 16, 19, 48, 50 |
| Internal validation | 9 | 0, 2, 17, 33 |
| Primary confirmation | 10 | 8, 21, 28, 41, 53, 56 |
| Primary confirmation | 11 | 6, 20, 26, 40, 46, 54 |

The inferential unit is the donor. Cells and RNA-by-protein coordinates are
repeated measurements within donor. Physical pools define validation blocks
and bootstrap strata.

## Frozen panel and states

The 17 cognates are:

| RNA (`var/genome == GRCh38`) | Protein target | Exact protein feature (`var/genome == BD99AbSeq`) |
|---|---|---|
| CD1C | CD1c | `CD1c\|CD1C` |
| CD2 | CD2 | `CD2\|CD2` |
| CD4 | CD4 | `CD4\|CD4` |
| CD7 | CD7 | `CD7\|CD7` |
| CD8A | CD8 | `CD8\|CD8A` |
| ITGAM | CD11b | `CD11b\|ITGAM` |
| ITGAX | CD11c | `CD11c\|ITGAX` |
| CD14 | CD14 | `CD14\|CD14` |
| MS4A1 | CD20 | `CD20\|MS4A1` |
| CD27 | CD27 | `CD27\|CD27` |
| CD33 | CD33 | `CD33\|CD33` |
| CD34 | CD34 | `CD34\|CD34` |
| CD38 | CD38 | `CD38\|CD38` |
| CD69 | CD69 | `CD69\|CD69` |
| CD80 | CD80 | `CD80\|CD80` |
| CD86 | CD86 | `CD86\|CD86` |
| CD163 | CD163 | `CD163\|CD163` |

The short protein target is display metadata. Resolution uses the exact
composite feature. `var/feature_types` cannot separate modalities because this
file labels both as Gene Expression. The committed axis preflight must verify
every exact label and modality once; until then the panel remains a frozen
expectation rather than a verified axis certificate.

For each donor, select 128 unique obs identifiers by ascending
`SHA256('GSE181897-CONTROL-CELL-BUDGET-v1|' + str(batch) + '|' + str(exp_id) + '|' + obs_index)`,
then obs identifier, and restore deposited order. RNA and protein state are
both raw count greater than zero. A marker is supported in a donor when 4 to
124 cells are positive. The resulting 17 by 17 field has 289 ordered
coordinates.

## Training masks and geometry

Every inner or outer source training problem constructs its own mask. With
`D` training donors, a coordinate requires valid informative tables in at
least `ceil(D/2)` donors, pooled observed `n11` strictly inside the summed
fixed-margin support, and four positive cells in the unstratified pooled 2 by 2
table. Validation availability never enters a training mask, and masks are not
intersected across folds. The all-39-source mask alone is frozen for internal
and confirmation scoring. Every training mask, the final mask, and every
evaluated donor must retain at least 232 of 289 coordinates.

RNA geometry uses donor Jeffreys detection logits. Protein geometry uses the
donor mean of cellwise CLR values computed over all 96 BD99AbSeq proteins.
Within each training set, marker profiles are centered within physical pool
and divided by pooled within-pool standard deviation. A zero or nonfinite
standard deviation is a refusal. Validation profiles never enter this fit.

## Estimator and source selection

The primary estimator is a penalty-complete, exact-conditional product-
hypergraph coupling field. Separate RNA and protein hypergraphs contain one
marker-centered edge per marker: the marker and its `k` nearest neighbors.
Their Cartesian-product Laplacian regularizes population RNA-protein log odds;
positive heterogeneity and ridge penalties yield a finite penalized prediction
for zero-support coordinates. Such values are reported as penalty-propagated,
not data-identified.

Source evaluation is nested leave-one-pool-out. Each of eight outer folds seals
one source pool. Inner leave-one-pool-out validation over the other seven pools
selects:

- Stage A: heterogeneity penalty `{0.1, 1, 10}`, ridge `{0.01, 0.1}`, and
  transport multiplier `{0.5, 0.75, 1, 1.25, 1.5}` under graph zero.
- Stage B: `k` in `{2, 3}` and graph penalty `{0.01, 0.03, 0.1, 0.3}`, with
  Stage A fixed.

Every mask, normalization, hypergraph, and estimator is rebuilt inside the
training fold. Loss is equal across pools and donors within pool. After a
source pass, the same selection is run over all eight source pools and refit on
all 39 donors. Stage A exact ties resolve by heterogeneity penalty, ridge
penalty, and transport multiplier; Stage B exact ties resolve by graph penalty
and then `k`.

Within each marker-centered hypergraph, equal Euclidean neighbor distances
resolve by lower zero-based marker index before duplicate hyperedges are
removed.

The source gate requires a nonzero graph penalty; at least 5% lower equal-pool
loss than matched graph zero; improvement in at least 7 of 8 outer pools and
27 of 39 donors; and a 20,000-draw within-pool paired-bootstrap upper 95%
endpoint below zero. The primary must also beat the source-selected classical
coordinate in mean loss with a bootstrap upper endpoint below zero. A failure
is published and closes internal access.

After a source pass, 63 deterministic, selection-aware topology nulls rerun
the complete Stage B source selection under independently permuted RNA and
protein incidence rows. All null fits are frozen before internal access.

## Comparators

The mandatory head-to-head set comprises matched graph zero, exact
donor-stratified common-effect conditional MLE, unstratified saturated pooled
Poisson interaction, and the source-selected signed Pearson or signed-root
Poisson-deviance coordinate. Target-margin independence is always reported.
A deterministic within-donor cyclic shift of complete binary protein rows is
the destroyed-link control.

Pooled Poisson uses exactly the supported donor-coordinate tables available to
the primary and omits unsupported donor-coordinate tables; only donor strata
are removed. Held CSR access reports the full number of `/X/indices` entries
scanned for selected rows separately from `/X/data` values decoded for the
frozen columns. No out-of-panel data value, non-control row, or unauthorized
held row may be decoded or retained.

Every method receives the same source folds, donor states, target margins,
comparison mask, and per-coordinate multinomial deviance. Classical methods
select only their transport multiplier, and the residual family is locked by
source CV. Internal and confirmation outcomes cannot select any component.

## Sealed gates

Internal validation scores the nine donors in pools 8 and 9 once. Against
matched graph zero, a pass requires at least 5% lower equal-pool loss, 8 of 9
favorable donors, one-sided exact sign `p <= 0.025`, both pool means lower, and
a paired-bootstrap upper 95% endpoint below zero. Against common-effect MLE,
pooled Poisson, and the selected residual coordinate, the equal-pool mean and
bootstrap upper endpoint must be negative; both pool means must also improve
against the selected residual. The topology empirical `p` must be at most
0.05, loss must be at least 3% below the median null, and the destroyed-link
bootstrap upper endpoint must be negative.

Only a complete internal pass opens the 12 confirmation donors in pools 10 and
11. The same rules apply, with at least 10 of 12 favorable donors versus graph
zero. Internal and confirmation donors are never pooled for selection,
inference, or rescue. A confirmation pass is same-study, donor-disjoint and
pool-disjoint evidence, not independent-study replication.

Bootstrap inference uses 20,000 paired donor resamples within physical pool
and an equal-pool mean. Cells and coordinates are never resampled as independent
units. Donor-level and pool-level loss vectors, confidence intervals, exact
sign tests, controls, nulls, exclusions, and refusals are all published. The
exact sign test omits zero paired differences and uses the number of nonzero
donor pairs as its binomial `n`.

## Access order

The public chain is candidate tag, implementation and runtime tag, axis-only
preflight tag, public source authorization, source result, internal margins and public predictions,
internal score, confirmation margins and public predictions, and confirmation
score. Each numeric stage has one exclusive attempt claim. A hash mismatch,
support failure, exception, or interruption after that claim consumes the
stage. There is no retuning, rerun, donor replacement, stage pooling, or rescue.

Source access additionally requires a public authorization at
`data/development/gse181897_source/source_campaign_authorization_v1.json`.
Before the first source `/X/indptr`, `/X/indices`, or `/X/data` read, the runner
must create `data/development/gse181897_source/source_attempt_v1.json`
locally with `O_CREAT|O_EXCL` and fsync it. The attempt is included in the later
published result; it does not receive a separate pre-access tag. Any subsequent
access, partial output, interruption, refusal, exception, or hash mismatch
consumes that attempt. The version 1.1 public tag and implementation-bound
authorization are both closed gates until published.

No numeric `X` dataset may be decoded until the prescribed implementation,
axis-v2, and authorization bindings have been committed, tagged, pushed, and
verified through the frozen local, remote, ancestry, and byte-identity checks.
