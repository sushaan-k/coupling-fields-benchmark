# GSE202150 acute-infection prospective held-batch CITE-seq validation

Frozen on 29 August 2026 after review of the official GEO file manifest,
feature reference, sample-to-HTO map, and additional sample annotations. One
IOF1 H5 file was initially inspected at `/matrix/features/*` and
`/matrix/shape`. The final preflight downloaded all 19 H5 files, bound their
SHA-256 digests, and opened only `id`, `name`, and `feature_type` on each
feature axis. Whole-file hashing read opaque bytes but did not decode an assay
dataset. No barcode, sparse index, RNA count, HTO count, or ADT count was
opened before this freeze. The response matrices remain numerically
unexamined.

## Question and units

The experiment asks whether an RNA-protein coupling field learned from acute
infection subjects in two acquisition batches predicts cell-matched dependence
in two untouched batches after each held subject's RNA margin and fixed ADT
midrank margin are restored. The physical subject is the inferential unit.

The official tables identify 33 infected physical subjects, each with exactly
one sample labeled `acute`. Their acquisition-batch counts are 8, 8, 8, and 9.
IOF1 is source calibration, IOF2 is the source-only pilot, and IOF3 plus IOF4
are held, giving 16 source and 17 held subjects. A subject never crosses a
role. Pre-infection, convalescent, and late-convalescent samples are excluded.
Healthy aliquots `HD105_Control` and `HD108_Control`, repeated across all four
batches, never enter estimation, method selection, or inference; they are used
only for descriptive batch QC after confirmatory scoring.

The public candidate designation records every retained acute sample, physical
subject, pathogen, loading batch, role, and official unordered HTO pair. The
HTO mapping is not reconstructed from row order. Within each cell, the official
unique names `HTO-N_h-3-IH-A` map to H5 feature names `HTO-N` only when the
suffix and numeric prefix match exactly. The HTO features are ranked by count.
Assignment requires two strictly positive,
uniquely ranked top tags whose unordered pair matches exactly one official
sample in that loading batch. Ties and unmapped pairs are excluded.

## Panel, states, and sampling

The exact 13-cognate panel is `CD3E/CD3`, `CD4/CD4`, `CD8A/CD8`,
`MS4A1/CD20`, `CD19/CD19`, `CD27/CD27`, `CD38/CD38`, `CD14/CD14`,
`ITGAM/CD11b`, `ITGAX/CD11c`, `CD33/CD33`, `NCAM1/CD56`, and
`KLRB1/CD161`. The submitted panel has `CD45RA`, not total `CD45`, so
`PTPRC/CD45` is excluded rather than silently replaced. Every RNA symbol and
ADT target must resolve exactly once in every H5 feature schema. All 169 ordered
RNA-by-ADT pairs are retained.

RNA state is `count > 0`. ADT state is the within-subject deterministic
midrank: the 192 highest of 384 selected cells, with salted SHA-256 and
composite cell identifier breaking count ties. Every ADT margin is therefore
192/192. The submitted filtered matrix supplies candidate cells. RNA QC
requires at least 200 detected genes, mitochondrial fraction at most 0.10, and
at most 70,000 RNA UMIs. Eligible cells are pooled across a batch's physical
libraries by `library|barcode`; the 384 smallest salted SHA-256 identifiers are
selected per acute subject. Every designated subject must contribute 384 cells
and at least 100 informative ordered pairs. There is no attrition, threshold
retuning, replacement, or rescue.

## Source fitting and comparator lock

The primary method is exact fixed-margin hierarchical conditional coupling.
IOF1 leave-one-subject-out validation selects heterogeneity penalty
`0.1,1,10`, ridge penalty `0.01,0.1`, source-derived two-neighbor product-graph
penalty `0,0.1,1`, and transport multiplier `0.5,0.75,1,1.25`. Mean subject-
equal deviance is minimized with lexicographic configuration tie breaking.

The same IOF1 folds select a transport multiplier from `0.5,0.75,1,1.25` for
each classical family. This removes scale calibration as an advantage of the
primary estimator. Four classical predictions are retained:

1. the literal row-plus-column Poisson-independence signed-root deviance
   residual, pooled with equal subject weights and transported at multiplier
   one;
2. the same residual with its multiplier selected by IOF1 leave-one-subject-out
   validation;
3. a source-calibrated common-effect stratified conditional maximum-likelihood
   interaction;
4. a source-calibrated donor-pooled saturated 2-by-2 Poisson log-linear
   interaction.

IOF2 locks the estimable classical prediction with the lowest subject-equal
mean deviance, with the order above breaking exact ties. The primary and
classical configurations are not retuned on IOF2. Every retained model is then
refit on all 16 source subjects. A classical boundary, zero-cell, or numerical
refusal is reported explicitly. Both raw and calibrated direct residuals are
mandatory; a refusal is terminal. Primary performance on IOF2 is reported but
is not a held-access gate.

The broad gain-over-classical claim additionally requires at least one valid
interaction fit, lower primary IOF2 mean loss than the locked classical method,
and a supported held advantage over every estimable classical method. Failure
of that claim contract does not invalidate the narrower held comparison
against the prospectively locked strongest classical comparator.

## Held prediction separation

After a public source result, the held prediction stage may decode HTO and RNA
counts from IOF3 and IOF4. The combined 10x H5 stores all modalities in one CSC
vector. The reducer reads the complete `indices` and `indptr` arrays, so the
nonzero support of co-resident ADT rows is visible; whole-file SHA-256 also
reads opaque file bytes. It requests `data` only at HTO or Gene Expression
positions and never slices a whole mixed-modality column. No held ADT count is
decoded, returned, serialized, or used before predictions are public.

For each held subject the public prediction stores RNA margins, fixed 192/192
ADT margins, same-margin predicted tables for every valid method, and a hash of
the selected composite-cell axis. Cell identifiers and RNA state vectors go to
a private artifact outside the repository; only its byte count and SHA-256 are
public. The prediction result must be published at
`gse202150-citeseq-v1-predictions` before score authorization is created.

## One-shot held inference

After the public score-authorization and score-attempt tags, scoring is the
first stage allowed to decode held ADT counts or form held RNA-ADT tables.
The primary endpoint compares subject-equal multinomial deviance against the
locked best classical method on the same 17 subjects and the same per-subject
informative-pair masks. It passes only if all conditions hold:

1. mean loss is at least 5% lower;
2. the 97.5th percentile of a 20,000-draw paired subject bootstrap is below
   zero;
3. at least 14 of 17 subjects favor the primary;
4. the one-sided exact donor sign-test has `p <= 0.025`;
5. mean paired differences are negative in both IOF3 and IOF4;
6. every leave-one-pathogen-out mean difference is negative.

With only two held batches, an exact batch sign-flip test cannot attain
`p <= 0.025`; it is not misrepresented as inferential evidence. Both batch
means are required negative instead. An exhaustive sign flip over the seven
pathogen means is reported as a prespecified heterogeneity sensitivity, not as
a distribution-free primary gate. The pairing-destroyed field is evaluated
only after the primary endpoint passes. All classical losses and refusals are
reported regardless of direction. Bridge-donor QC is computed last and cannot
change any prediction, comparison, or pass decision.

## Irreversible public stages

Each numeric stage requires a public `STARTED` ledger before its first assay
value is read. Immediately before the numeric body, the runner revalidates the
public protocol bytes, prerequisite result, and exact CPython, NumPy, SciPy,
h5py, and HDF5 runtime. It creates an exclusive atomic execution claim, then
appends `EXECUTING_CONSUMED`; a concurrent or later invocation cannot enter the
numeric body. The runner writes one exclusive output and appends one `FINISHED`
record. Success, support refusal,
numerical refusal, interruption, or unexpected exception consumes the stage;
rerun and adaptive rescue are forbidden. The sequence is:

1. metadata/schema preflight, protocol commit, and
   `gse202150-citeseq-v1-protocol`;
2. source claim/tag, one source run, source result/tag;
3. prediction claim/tag, one RNA-only held prediction, prediction result/tag;
4. score authorization/tag, score claim/tag, one held ADT score, result/tag;
5. read-only verification that the public result tag binds the score output,
   completed ledger, and atomic execution claim.

Candidate-specific intervals and tail probabilities are reported. This study
was selected from metadata after earlier candidates ended in public terminal
failures. Its untouched batches therefore provide a prospectively locked,
dataset-specific held-subject validation; no campaign-wide error rate is
claimed.
