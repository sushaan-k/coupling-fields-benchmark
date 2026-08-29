# GSE189050 held-pool linked RNA/ADT confirmation protocol

Frozen after downloading and hashing the ten filtered CellRanger archives and
inspecting only tar member headers and feature tables. No barcode value,
MatrixMarket header, coordinate line, RNA count, HTO count, or biological ADT
count was opened before this protocol freeze.

## v1.1 runtime-closure amendment

An independent audit of the public v1 protocol tag found that its binding list
included the two directly imported estimator modules but omitted five
transitive `mapreg` runtime modules. No barcode or matrix member had been opened,
so the scientific protocol and source bytes remained outcome-blind. Version
1.1 adds the complete seven-file `mapreg/*.py` runtime closure plus
`requirements.txt` and `pyproject.toml` to the immutable binding list. It does
not change the source, split, panel, demultiplexing, QC, estimator grid,
comparators, thresholds, seeds, or gates. The v1 tag remains immutable and is
not an execution authorization.

## Scientific question

Can a conditional RNA-protein coupling field learned in two calibration pools
predict cell-matched dependence in six held pools after recipient RNA and fixed
protein margins are restored? The primary field must beat the pilot-selected
signed classical residual and a destroyed-link field. Two additional classical
estimators are prespecified secondary head-to-head comparisons.

GSE189050 contains linked RNA, antibody capture, and hashtag counts from 46
physical subjects in ten multiplex pools of CD2-depleted PBMC. Calibration,
pilot selection, and held evaluation are pool-disjoint. The physical subject is
the inferential unit. No disease label, ancestry, age, author cluster, or
embedding enters feature selection, fitting, tuning, or evaluation.

## Sources and schema-only access

The source manifest binds all archive URLs, byte counts, SHA-256 digests, tar
member names and sizes, and per-pool feature-table digests. Each archive has
exactly three members in this order: barcodes, features, and a feature-by-cell
integer coordinate MatrixMarket matrix. The first four pools have 68,886 gene
expression and 56 antibody-capture features; the last six have 68,886 and 57,
respectively. The difference is the fifth HTO tag.

The schema preflight reads the committed GEO and author sample sheets, tar
headers, and feature-table bytes. It does not extract barcode content or open a
matrix member. The author sheet supplies only `(run, Subject_id, Hashtag)`.
GEO is authoritative for clinical covariates. The author sheet swaps the
clinical labels and ages of `SUB235957` and `SUB236000` relative to GEO; those
author fields are ignored and no clinical field is gating.

## Split

- calibration: `s1a,s3a`, 9 subjects (4 EA, 5 AA);
- pilot: `s2a,s4a`, 9 subjects (4 EA, 5 AA);
- held: `s1b,s2b,s3b,s4b,s5a,s5b`, 28 subjects (15 EA, 13 AA).

The split is fixed before barcode or matrix access. No subject occurs in more
than one pool.

## Frozen primary and secondary panels

The primary B/myeloid cross-assay panel contains 12 markers and scores their
12-by-12 Cartesian product (144 ordered RNA-ADT pairs). Its 12 cognate feature
mappings are
CD1C-CD1c, ITGAM-CD11b, ITGAX-CD11c, CD14-CD14, CD19-CD19, CD27-CD27,
CD38-CD38, CD58-CD58, FCGR1A-CD64, LILRB1-CD85j, CD86-CD86, and
LAIR1-CD305. Each RNA row must match one exact `GRCh38_`-prefixed Ensembl and
symbol pair. Each ADT row must be unique after uppercase ASCII-separator
removal and removal of at most one terminal suffix from the fixed set
`PROTEIN, ADT, ANTIBODYCAPTURE, TOTALSEQA, TOTALSEQB, TOTALSEQC`. Mouse rows
cannot satisfy an RNA feature. A missing or nonunique match is terminal.

The earlier nine-marker cross-assay panel has nine cognate mappings and scores
81 ordered pairs (CD14, CD16, CD11c, CD19, CD27, CD38, HLA-DR, CD95, and
CD305 with their frozen cognate RNA genes). It is evaluated only as a
non-gating secondary panel. The ambiguous HLA-DR and CD16 mappings do not enter
the primary claim.

## Fixed HTO demultiplexing and QC

Seurat was unavailable in both the default R 4.6.1 library and the project R
library before count access. No package will be installed and no substitute
will be selected after access. The transparent frozen rule is therefore used.
A barcode is an HTO singlet only when the maximum tag is unique, total HTO
count is at least 20, `(top+1)/(second+1)` is at least 5, and `top/total` is at
least 0.70. It is retained only when its human fraction among human-plus-mouse
RNA UMIs is at least 0.90. There is no rescue, quantile fitting, yield-driven
threshold change, or manual reassignment.

Before estimator fitting, each opened pool must satisfy all of the following:

1. every HTO tag declared for that pool has at least one accepted singlet;
2. at least 50% of human barcodes are accepted singlets;
3. no more than 30% of human barcodes are HTO-negative (`total < 20`);
4. no more than 30% are HTO-ambiguous or doublet-like;
5. the largest-to-smallest positive donor singlet yield is at most 4.

A retained donor must have at least 512 accepted singlets. The 512 cells with
the smallest salted SHA-256 ranks are selected. Calibration and pilot each
require at least seven retained donors and at least three per pool. Held
prediction requires at least 22 retained donors and representation of all six
held pools. Gates are never relaxed.

RNA state is `count > 0`. Within each donor-marker pair, ADT counts are ordered
by count, a frozen salted hash, and cell ID; the upper 256 cells define the high
state. Primary donor loss requires at least 108 informative pairs among 144.
The legacy panel retains the earlier 64-of-81 requirement. Destroyed-link ADT
profiles are shifted once along an independent frozen hashed cell order,
preserving every donor-marker margin.

## Estimators and pilot selection

The primary is the donor-heterogeneity conditional coupling field. Calibration
fits the Cartesian grid of graph neighbors `1,2`, heterogeneity penalty
`0.1,1,10`, ridge penalty `0.01,0.1`, graph penalty `0,0.1,1`, and transport
multiplier `0.5,0.75,1,1.25` (144 nominal configurations). Donor-equal pilot
deviance selects the configuration with lexicographic tie breaking. The signed
Pearson and signed-root Poisson-deviance comparators use the same transport
grid; pilot deviance selects one family and multiplier. A graph-zero field is
diagnostic only.

Two classical comparisons are bound before matrix access:

- a common-effect stratified conditional maximum-likelihood log-odds field;
- a donor-pooled saturated Poisson log-linear interaction field, computed as
  the pooled table log odds ratio and refused if a pooled cell is zero.

Each classical field is fit on calibration donors, selects only its transport
multiplier on the pilot from `0.5,0.75,1,1.25`, and is then refit on all
development donors without retuning. Both are reconstructed at each recipient's
observed margins. They are prespecified secondary comparisons, not additions to
the primary promotion gate. Paired donor-bootstrap and independent run-block
intervals are reported for both.

## Pilot and held gates

The primary must beat both the selected signed residual and destroyed-link
field by every criterion:

1. at least 5% donor-equal mean deviance reduction;
2. upper 95% endpoint below zero from 20,000 paired independent-run-block
   bootstrap draws with seed `20260828`;
3. lower donor loss in at least `ceil(0.8 n)` donors;
4. one-sided exact donor sign-test `p <= 0.025` after discarding exact ties;
5. negative mean paired difference in every physical pool.

Pilot blocks are `s2a` and `s4a`. Held primary blocks are `s1b,s2b,s3b,s4b,s5`,
where `s5a` and `s5b` are combined because they are not independent run-level
units. All six physical held pool means must nevertheless be negative. A
six-physical-pool bootstrap is reported only as sensitivity and is not called
an exact or independent-pool test. Pilot failure is terminal.

## Public barriers

1. Publish and independently verify the annotated tag
   `gse189050-citeseq-v1.1-protocol`, which binds the runner, tests, protocol,
   manifests, source metadata, actual feature-schema preflight, and inherited
   estimator implementation. No barcode or matrix member may be opened before
   this verification.
2. Run calibration plus pilot only. Publish a terminal failure immediately, or
   if the primary gate passes, publish and independently verify
   `gse189050-citeseq-v1-development`.
3. Open held barcodes, HTO, and RNA only. The sparse reader parses coordinate
   row and column indices but does not convert, retain, log, or serialize the
   value token of any frozen biological ADT row. Publish and independently
   verify `gse189050-citeseq-v1-predictions`.
4. Only after the prediction tag is verified may the one-shot scorer convert
   held biological ADT values. It rechecks source, cell-axis, margin, and
   prediction hashes and publishes pass or failure without retuning.

Synthetic poison-value tests require held prediction to succeed when every
target ADT value token is nonnumeric, while the score reader must reject the
same token. No full MatrixMarket file is extracted or materialized at any
stage.
