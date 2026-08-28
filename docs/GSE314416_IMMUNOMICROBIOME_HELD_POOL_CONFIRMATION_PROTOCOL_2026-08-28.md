# GSE314416 held-pool CITE-seq confirmation protocol

Frozen 2026-08-28 before downloading or opening any GEX or ADT H5 matrix.

## Scientific question

Can an RNA-protein conditional coupling field learned from healthy baseline
participants in two acquisition pools predict cell-matched dependence in five
unseen pools, after restoring each recipient donor's margins? The primary
comparison is against the best calibration-selected signed classical residual.

GSE314416 contains 110 healthy baseline participants. Pools DB1--DB7 used the
same 10X 5-prime Next GEM v1.1 chemistry and the same deposited TotalSeq-C
feature reference. Follow-up pools DB8--DB14 used v2 chemistry and are excluded.
Pool is an acquisition batch, not a biological outcome. The prediction target
is within-donor RNA-ADT pairing conditional on donor margins; no participant
class is inferred from pool membership.

## Metadata-only eligibility and split

The SHA-256-bound baseline metadata are exported by
`experiments/export_gse314416_metadata.R`. A physical donor is eligible with at
least 512 deposited baseline cells and membership in exactly one baseline pool.
The one donor with 34 cells is excluded.

- Calibration: 12 DB1 donors selected by
  `SHA256(GSE314416-CALIBRATION-DONOR-v1|donor)`; the explicit list is in the
  candidate designation.
- Pilot: the four remaining eligible DB1 donors and all 16 eligible DB2 donors
  (`n=20`).
- Held: every eligible donor in DB3--DB7 (`n=77`). These donors and acquisition
  pools are untouched during development.

For each donor, 512 cells are selected by sorting
`SHA256(GSE314416-CELL-BUDGET-v1|donor|cell_id)`, then cell ID. The preflight
records one selected-axis digest per donor without opening an H5 file.

## Frozen primary panel and states

The primary is the complete nine-marker cognate panel used in the preceding
COMBAT and GSE239452 evaluations: CD4, CD7, CD14, CD19, CD33, CD38, CD44, CD47,
and CD52. Ensembl and TotalSeq-C feature IDs are fixed in the candidate
designation and checked against the public feature reference.

RNA is binary (`count > 0`). Each donor-marker ADT vector is split into exactly
256 low and 256 high cells. Counts are ordered first; ties are ordered by
`SHA256(GSE314416-ADT-MEDIAN-v1|donor|marker|cell_id)`. Every ordered RNA-ADT
pair produces a 2-by-2 table. The destroyed-link control deterministically
permutes complete ADT state rows using
`SHA256(GSE314416-DESTROYED-LINK-v1|donor|cell_id)`.

## Estimators fixed before count access

The primary estimator is the exact conditional noncentral-hypergeometric field
with hierarchical donor effects. Calibration leave-one-donor-out CV searches:

- graph neighbors: `1, 2`;
- heterogeneity penalty: `0.1, 1, 10`;
- ridge penalty: `0.01, 0.1`;
- graph penalty: `0, 0.1, 0.3, 1`;
- transport multiplier: `0.5, 0.75, 1, 1.25`.

Neighbor count is ignored when graph penalty is zero. Ties are resolved by
mean loss followed by dataclass lexicographic order. A graph-zero configuration
is retained as a diagnostic.

The classical candidates are signed Pearson and signed-root Poisson-deviance
coordinates, donor-equal pooled, crossed with the same four transport
multipliers. A coordinate is divided by `sqrt(512)` before pooling and restored
after transport. Inversion searches the feasible 2-by-2 interval at endpoints
offset by `min(1e-10, one quarter of interval width)`, then uses 96 bisection
steps. This rule preserves negative signed-root-deviance coordinates and is
covered by a frozen regression test.

A nongating common-effect exact conditional baseline fits one unpenalized
stratified log odds per RNA-ADT pair across development donors and transports
it with multiplier one. It separates fixed-margin conditioning from the
hierarchical donor-shrinkage contribution. A boundary optimum or failed
gradient/condition certificate is reported as a baseline refusal and does not
alter the required residual comparison.

All models are evaluated by mean per-entity multinomial deviance divided by
512. A donor must have at least 64 informative primary pairs. Fits require two
informative development donors per entity and condition number at most `1e12`.

## Secondary broad panel

A non-gating 24-marker cognate panel is frozen in the candidate designation.
It contains the full primary panel plus 15 prespecified PBMC lineage and
activation markers with unambiguous feature-to-Ensembl mappings. It uses the
same cells and states, a fixed graph-zero exact estimator (`eta=1`, ridge
`0.1`, transport `1`) and a fixed signed-root Poisson-deviance comparator
(transport `1`). At least 432 of 576 pairs must be informative per donor. Its
pilot and held paired donor intervals are reported separately; it cannot rescue
or overturn the primary gate.

## One-shot gates and uncertainty

Both pilot and held primary gates require, against both the selected residual
and destroyed link:

1. at least 5% mean deviance reduction;
2. an upper paired-donor bootstrap 95% endpoint below zero;
3. exact loss below comparator loss in at least 15/20 pilot donors or 58/77
   held donors.

There are 20,000 paired donor bootstraps with seed `20260828`. Failure of the
pilot gate permanently refuses held scoring. The held outcome is scored once.

## Access sequence

1. Commit and publicly tag this protocol, runner, tests, source/designation
   manifests, and metadata-only preflight as `gse314416-citeseq-v1-protocol`.
2. Download the opaque GEO archive and extract DB1--DB7 H5 files. Run
   `develop`; only DB1--DB2 are opened. Commit the pilot result and selected
   exact/residual configurations, then publicly tag
   `gse314416-citeseq-v1-development`.
3. Run `predict`; only held GEX marker rows are opened. Recipient RNA margins
   and all predicted tables are committed and publicly tagged
   `gse314416-citeseq-v1-predictions`. No held ADT H5 is opened.
4. Run `score`; this is the first held ADT access. Verify frozen margins and
   table hashes, evaluate the 77 donors once, and publish pass or failure.

Each numeric command verifies the applicable public tag and byte identity of
its bound inputs before opening an H5 file.
