# GSE179221 BMMC CITE-seq held-donor protocol

## Question and freeze boundary

The candidate designation is interpreted together with
`pre_access_implementation_amendment_v1.json`. That amendment freezes feature
resolution, graph construction, comparator reconstruction, fold-specific
masks, deterministic ties, and the staged access firewall before any H5 body
is requested.

This one-shot experiment tests whether RNA--surface-protein dependence learned
from eight bone-marrow donors predicts paired dependence in ten donor-disjoint
marrow samples. The source and held sets are stratified across healthy, IgM
MGUS, and Waldenstrom marrow before any H5 body is requested. The physical
donor is the sole inferential unit.

The public study reports 84,128 CITE-seq cells from four healthy donors, seven
IgM MGUS donors, and seven Waldenstrom donors. GEO deposits one raw 10x feature
barcode H5 per donor. The candidate freeze used only the GEO record, the
official file inventory, the primary paper, and HTTP HEAD responses. No H5
body, barcode, feature axis, assay value, association, prediction, or loss had
been accessed.

## Allocation and files

Within each disease stratum, GSM accessions are ordered by SHA-256 of

```text
GSE179221-BMMC-CITESEQ-HELD-v1 NUL stratum NUL GSM
```

and the first half, rounded upward, is held. This yields two healthy, three
MGUS, and three WM source donors; two healthy, four MGUS, and four WM held
donors. The exact accessions, filenames, byte counts, and order are bound in
`candidate_designation_v1.json`. The all-donor GEO tar is forbidden. Before
source promotion, only the eight source URLs may receive a GET request.

## Fixed panel and states

The nine cognates are `CD3D--CD3`, `NCAM1--CD56`, `CD19--CD19`,
`CD14--CD14`, `FCGR3A--CD16`, `MS4A1--CD20`, `CD27--CD27`,
`CD38--CD38`, and `CD79B--CD79b`, in that order. The paper identifies these
proteins in the profiled panel. The implementation must resolve each RNA symbol
and a publicly frozen ADT alias exactly once in every source and held feature
schema. Failure of any cognate is terminal; feature metadata cannot add,
replace, or reorder a marker. The analysis covers all 81 ordered RNA-to-ADT
pairs.

RNA-only QC retains cells with at least 200 detected genes, mitochondrial UMI
fraction at most 0.10, and at most 70,000 RNA UMIs. Eligible barcodes are
ordered by SHA-256 of the frozen cell salt, donor, and barcode; the first 512
are retained. A donor with fewer than 512 eligible cells is a terminal support
refusal. RNA state is raw detection. Each ADT is split within donor at its
midrank into 256 low and 256 high cells, with the frozen tie salt resolving
ties. The destroyed-link control cyclically shifts complete ADT state vectors
by one position along the salted cell order, preserving every marginal and the
multivariate protein profile.

## Source models and classical head-to-head

Source selection uses eight leave-one-donor-out folds. Every graph, comparison
mask, and fit is rebuilt from the seven training donors. A validation donor
contributes only its recipient margins and truth for scoring; its paired counts
cannot alter the fold mask. The final prediction mask is rebuilt from all eight
source donors. The primary hierarchical exact
conditional coupling field crosses heterogeneity penalty `0.1, 1, 10`,
population ridge `0.01, 0.1`, graph penalty `0, 0.03, 0.3`, and transport
multiplier `0, 0.25, 0.5, 0.75, 1, 1.25, 1.5`. Only configurations completing
every fold are eligible.

Five controls are mandatory and use the same donors, coordinates, recipient
margins, and deviance:

1. source-selected signed Pearson or signed-root Poisson-deviance residual
   transfer;
2. the frozen destroyed-link refit;
3. one unpenalized exact conditional common log odds per coordinate;
4. a standard pooled saturated Poisson row-plus-column-plus-interaction fit,
   reconstructed at new margins by refitting row and column nuisance parameters
   with its interaction fixed; and
5. recipient-margin independence.

The pooled Poisson comparator is the classical plug-in log-linear prediction.
It does not pass its source coefficient through a conditional
noncentral-hypergeometric expectation.

## Source promotion

Each fold-specific comparison mask is determined from its seven training
donors before candidate scoring. Every retained coordinate must have a finite
interior common-effect estimate and a finite pooled Poisson interaction in that
training set. Its intersection with the validation donor's margin support must
contain at least 64 coordinates. The final held mask is determined only from
the eight source donors.

Source promotion requires all of the following:

- every selected and final estimator satisfies its numerical certificate;
- the primary has at least 5% lower donor-equal mean deviance than the selected
  residual, standard pooled Poisson, destroyed-link fit, and independence;
- the paired-donor bootstrap upper 95% endpoint is below zero against the
  residual, pooled Poisson, and destroyed-link fit;
- at least seven of eight source donors favor the primary in each of those
  three comparisons; and
- the primary point loss is below the common-effect conditional fit.

The source bootstrap uses 20,000 donor resamples and seed 20260830. A missed
criterion, incomplete fold, or failed certificate is terminal. No held H5 is
then requested.

## Prediction firewall and held score

After source promotion, the source and comparator coordinates are published
and independently verified before any held H5 GET. The ten held files are then
downloaded one at a time, checked against their frozen byte counts, hashed,
reduced, and deleted.

The margin stage performs the frozen RNA QC and cell selection, derives RNA and
ADT states separately, and publishes margins and recipient predictions without
forming a joint RNA--ADT table. A separate authorization binds those
predictions. The score stage joins the two state artifacts once, forms the 81
tables, and writes only aggregate tables and losses.

The endpoint is donor-equal mean multinomial deviance per cell. Paired 95%
intervals use 20,000 donor-bootstrap draws with seed 20260830. Exact one-sided
sign tests treat the ten donors as the units. Disease-stratum means and
leave-one-donor-out means are reported regardless of outcome.

Confirmation requires at least 5% lower loss, a paired-bootstrap upper endpoint
below zero, improvement in at least nine of ten donors, and exact one-sided
sign-test `P <= 0.025` against each of the selected residual, standard pooled
Poisson, and destroyed-link controls. Primary point loss must also be below the
common-effect conditional ablation. A supported score that misses any
criterion is a completed negative result. No donor can be substituted and no
stage can be rerun.

## Reproducibility record

Candidate freeze, implementation, source attempt, source result, source
coordinates, held margins, predictions, score authorization, and held score
are separate checksum-bound public artifacts. Each stage logs requested URLs,
file identities, HDF5 datasets dereferenced, donor axes, and output hashes.
