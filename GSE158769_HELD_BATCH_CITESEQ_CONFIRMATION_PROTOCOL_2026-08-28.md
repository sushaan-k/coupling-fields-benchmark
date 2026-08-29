# GSE158769 held-batch linked RNA/ADT confirmation protocol

Frozen 2026-08-28 before download or decompression of
`GSE158769_exprs_raw.tsv.gz` and before access to any RNA or ADT count value.

## Scientific question

Can a conditional RNA-protein coupling field learned in 16 processing batches
predict cell-matched dependence in 16 independently held processing batches
after restoring each held donor's observed RNA and fixed ADT margins? The
required comparisons are the strongest pilot-selected signed classical
residual and a destroyed-link control.

GSE158769 contains 500,089 QC-passing memory T cells from 259 donors in 46
processing batches, with linked RNA and a 31-antibody TotalSeq-A panel. This is
an independent-study, held-donor and held-batch confirmation. It is not a held
laboratory or held-site evaluation: all batches come from the Nathan et al.
cohort (`10.1038/s41590-021-00933-1`). Upstream cell inclusion used CD3 and
CD45RO protein measurements. The estimand is therefore conditional coupling
among the deposited QC-passing memory T cells; no author UMAP or cluster label
enters selection, fitting, or evaluation.

## Source orientation and bounded access

GEO deposits one 783,271,940-byte gzip whose TSV rows are features and columns
are cells; the first column contains the feature label. The orientation was
established from the GEO description and author analysis code without opening
the matrix. RNA and ADT rows share this physical gzip, so protection is
enforced at tokenization and serialization rather than by claiming that held
ADT compressed bytes remain physically unread. The reducer holds one
decompressed row at a time, tokenizes only frozen feature rows and selected
cell columns, and never writes a decompressed TSV or selected cell-level count
matrix.

The source file has no upstream checksum. The protocol tag therefore binds its
GEO URL and byte count with a null digest. After the tag is public, `acquire`
downloads the gzip as opaque bytes, computes SHA-256 without decompression, and
writes a source-access record. Development is forbidden until that record is
committed and publicly verified under `gse158769-citeseq-v1-source`.

## Metadata-only eligibility and split

The metadata file has SHA-256
`5f5f44dbcf7dc28054f7f124560e3322e4ed9557ff5dbec0e48d2337af1c7f45`.
All split decisions use only donor, processing batch, TB status, cell ID, and
deposited cell count.

Twelve donors appearing in more than one processing batch are excluded. A
further 13 single-batch donors with fewer than 512 cells are excluded. The
explicit donor exclusions are in the candidate designation. The remaining 234
donors are allocated by the following exact batch arrays, whose order is also
frozen:

- calibration: `15,42,27,36,7,2,28,10,30,40,45,12,6,37,13,32` (85 donors,
  173,764 available cells, 42 cases and 43 controls);
- pilot: `33,41,34,38,11,29,24,18,22,9,14,17,35,4` (69 donors, 130,420
  available cells, 31 cases and 38 controls);
- held: `21,43,25,20,1,46,16,44,39,26,19,3,8,5,23,31` (80 donors, 148,228
  available cells, 39 cases and 41 controls).

For each eligible donor, the 512 cells with the smallest lexicographic pair
`(SHA256(UTF8("GSE158769-CELL-BUDGET-v1") || NUL || UTF8(donor) || NUL ||
UTF8(cell_id)), cell_id)` are selected. The metadata preflight commits each
selected-axis digest without accessing the raw matrix.

## Frozen panel and states

The nine cognate pairs are CD3E-CD3, CD4-CD4, CD5-CD5, CD8A-CD8a,
KLRB1-CD161, IL7R-CD127, CD27-CD27, CD38-CD38, and DPP4-CD26. Exact accepted
ADT source aliases are in the candidate designation. The raw format has no
modality column. RNA therefore resolves to the first exact gene-symbol
occurrence and ADT to the last accepted antibody-label occurrence; one source
row may not satisfy both modalities. Missing, coincident, or ambiguous frozen
features cause a terminal schema refusal. No value-dependent marker replacement
is allowed.

RNA state is `count > 0`. Within each donor and marker, ADT counts are ordered
by `(count, SHA256("GSE158769-ADT-v1" || NUL || donor || NUL || cell_id || NUL
|| canonical_marker), cell_id)`. The first 256 cells are low and the remaining
256 high. Each ordered RNA-ADT pair gives a 2-by-2 table; 64 of 81 pairs must be
informative in every scored donor.

The destroyed-link control orders cells by
`(SHA256("GSE158769-DESTROYED-LINK-v1" || NUL || donor || NUL || cell_id),
cell_id)` and cyclically shifts complete ADT state profiles by one place in
that order. It preserves every donor-marker margin and destroys cell pairing.

## Estimators and fixed selection

The primary is the donor-heterogeneity-aware exact conditional
noncentral-hypergeometric field. It models one 9-by-9 population log-odds field
with donor effects and optional k-nearest-neighbor graph regularization on
donor-level RNA detection and ADT abundance profiles. The 144 configurations
are the Cartesian product of:

- graph neighbors: `1, 2`;
- heterogeneity penalty: `0.1, 1, 10`;
- ridge penalty: `0.01, 0.1`;
- graph penalty: `0, 0.1, 1`;
- transport multiplier: `0.5, 0.75, 1, 1.25`.

Neighbor count is inert at graph penalty zero but both nominal configurations
remain in the 144-element frozen grid. Fits use calibration donors only. Mean
donor-equal pilot deviance selects the primary configuration, with dataclass
lexicographic order breaking ties. The graph-zero member with lowest pilot
loss is retained as a diagnostic.

The required classical comparator pools donor-equal signed Pearson or
signed-root Poisson-deviance coordinates from calibration donors and crosses
each family with the same four transport multipliers. Mean pilot deviance
selects its family and multiplier. It is a standard fixed-margin residual
transfer, not a weakened null. The destroyed-link field uses the selected
primary configuration. A common-effect exact conditional estimator is reported
when its fit certificate passes but is non-gating.

All predictions restore recipient margins through the exact
noncentral-hypergeometric expectation. Loss is mean per-pair multinomial
deviance divided by 512. If the pilot passes, the selected configurations are
refit once on all 154 development donors without retuning.

## Batch-block gates

The pilot must beat both required comparators by all four criteria:

1. at least 5% donor-equal mean deviance reduction;
2. upper 95% endpoint below zero from 20,000 paired processing-batch bootstrap
   draws with seed `20260828`;
3. lower loss in at least 56 of 69 donors;
4. lower batch-mean loss in at least 12 of 14 batches.

Each bootstrap resamples whole batches with replacement and evaluates the
donor-equal mean across every donor in the sampled blocks. Pilot failure is
terminal: no held RNA or held ADT value may then be tokenized, and no threshold,
panel, split, or configuration may be revised.

The held gate repeats the four criteria with at least 64 of 80 donors and 13 of
16 batches, and additionally requires a one-sided exact sign-flip
`p <= 0.025` over the 16 batch-mean paired differences. Both required
comparators must pass. The held outcome is scored once.

## Public barriers

1. Commit the runner, tests, this protocol, source/designation files,
   metadata-only access record, and metadata preflight. Push and independently
   verify the annotated tag `gse158769-citeseq-v1-protocol`. No raw count gzip
   may be downloaded or decompressed before this barrier.
2. Run `acquire`, commit its opaque-byte SHA-256 record, and push and verify
   `gse158769-citeseq-v1-source`. Run calibration plus pilot only. If the pilot
   fails, publish the terminal result and stop without touching held values.
3. If the pilot passes, publish and verify
   `gse158769-citeseq-v1-development`. Run `predict`; it recognizes and
   tokenizes only held RNA rows. ADT candidate rows pass through the gzip
   inflater but their value fields are never tokenized, converted, retained,
   logged, or serialized. Publish and verify
   `gse158769-citeseq-v1-predictions`.
4. Only after the prediction tag is public may `score` tokenize held ADT rows.
   It verifies frozen donor, margin, and prediction hashes, performs the single
   held evaluation, and publishes pass or failure.

Synthetic poison-row tests require the RNA-only prediction reader to complete
when every ADT value token is nonnumeric. This makes accidental held-ADT
tokenization fail before any real held execution.
