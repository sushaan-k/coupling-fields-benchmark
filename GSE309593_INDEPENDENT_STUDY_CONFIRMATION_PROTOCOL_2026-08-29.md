# GSE309593 independent-study confirmation protocol

## Frozen question

Can RNA--protein coupling fields selected and fitted in GSE288020 predict
cell-matched RNA--ADT dependence in an independent pretreatment bone-marrow
CITE-seq study? GSE309593 contributes recipient margins and one terminal score.
It contributes no model choice, penalty, transport multiplier, marker mapping,
threshold, comparator variant, or rescue rule.

This protocol was drafted from public metadata and feature-panel documentation.
No GSE309593 H5, ADT CSV, cell identifier, barcode, or assay value was
downloaded or opened.

## Source-only correction

The public candidate tag is preserved unchanged. Its preliminary GSE288020
split and all-23 final-refit sentence are superseded transparently by
`source_split_amendment_v1.json`.

GSE288020 configuration selection uses the final salted, immune-age-stratified
MGUS split:

- calibration: `R001,R005,R008,R009,R010,R013,R014`;
- pilot: `R003,R006,R015,R016,R020,R023,R024`.

After the source development artifact is public, the selected primary,
matched residual, destroyed-link control, common-effect CMLE, and pooled
saturated-Poisson interaction are refitted on these 14 MGUS donors. The nine
GSE288020 myeloma outcomes never enter the independent-study source artifact.
GSE309593 proceeds when the 14-donor source fits meet their frozen support and
numerical certificates, irrespective of the separate MGUS-to-myeloma
diagnostic. The diagnostic remains a reportable GSE288020 result, not a branch
gate for the independent study.

The source bundle is written atomically inside the publicly tagged GSE288020
development artifact at
`results/development/gse288020_development_v1.json#/source_only_external_study_model`.
It contains the exact 14-subject axis, 16 source markers, source-only selection
provenance, fit certificates, transport multipliers, and 16-by-16 coordinate
fields. The bundle records zero use of the nine MM outcomes and zero use of
GSE309593 values. Altered axes, unsupported pairs, a missing or refused method,
nonfinite coordinates, or a failed certificate prevent recipient access. All
five methods and `external_study_ready` are required, making both classical
head-to-head comparisons mandatory. Authorization also verifies the canonical
GSE288020 protocol and development annotated tags, the exact two-record
development attempt ledger, runtime specification, result status, protocol
commit, artifact hash, and nested-model hash. The nested-model hash is SHA-256
of its UTF-8 JSON serialization with sorted keys, comma and colon separators,
and nonfinite values forbidden.

## Recipient cohort and panel

The recipient cohort is all 23 distinct, non-bridge `MM Pre-Treatment`
subjects in the candidate designation, spanning batches B092, B099, B110,
B129, B162, B208, and B210. No target-study split or exclusion is chosen after
access.

The marker panel is the ordered intersection of four fixed schemas:

1. the candidate designation's ordered cognate list;
2. the source model's exact RNA-symbol axis;
3. exact RNA symbols in every recipient H5;
4. exact ADT target headers in every recipient CSV.

The intersection is by RNA symbol across studies; source and recipient antibody
labels remain separately recorded. Numeric values cannot add, remove, or
reorder a marker. At least nine cognates must remain. Every ordered RNA-to-ADT
pair among retained markers is predicted and, where its margins are
nondegenerate, scored.

## Staged access

Each stage has two commands. `claim` creates the sole STARTED ledger record
without target-file access. That record must be committed, pushed, and verified
under the designated annotated attempt tag. The corresponding `run-*` command
then verifies the public tag and appends an irreversible execution-begins
record before opening its first target file. Success or terminal refusal
appends one final record. A hard interruption therefore leaves a closed
attempt rather than a rerunnable one. A claimed stage is never rerun.

### RNA stage

The runner downloads one designated H5 at a time, checks the metadata-frozen
byte count, hashes it, reduces it, and deletes the temporary file. Its finite
reader supports 10x feature-by-cell CSC, AnnData cell-by-feature CSR or CSC,
and dense AnnData count matrices. For AnnData, the structural priority is
`layers/counts`, `raw/X`, then `X`. Selected values must be nonnegative integer
counts; otherwise the stage refuses.

For each subject, RNA-only QC retains cells with at least 200 detected genes, a
mitochondrial UMI fraction at most 0.10, and at most 70,000 RNA UMIs, matching
the source rule. The stage orders eligible H5 identifiers by SHA-256 of the
frozen cell-selection salt, subject, and identifier, then selects the first
512. It reads no ADT file. RNA state is raw UMI count greater than zero. The
public artifact contains file hashes, schema and sparse-structure audits, QC
counts, salted axis hashes, state hashes, and row margins. A private bridge
contains only the 512 raw technical identifiers; a separate private artifact
contains RNA states. Both private files must reside outside the repository,
and only their hashes and byte counts are public.

### ADT stage

The ADT process receives the identifier-only bridge, never the RNA-state file.
It downloads one designated CSV at a time, verifies its byte count, hashes it,
and deletes it after reduction. The predeclared parser accepts either cells by
markers or markers by cells, determined only from exact header membership.
Every RNA-selected identifier must occur exactly once. Missing or duplicated
identifiers cause terminal refusal; no fallback or resampling is allowed.

For each retained ADT, the stage computes a marginal-only support certificate.
A subject-marker is supported when no more than 128 cells belong to the
raw-count tie block crossing its 256/256 boundary. Supported counts are ordered
within subject; hashes of the frozen tie salt, subject, target, and identifier
break remaining ties, with 256 cells assigned to each state. An unsupported
subject-marker receives a 512/0 sentinel margin and no high/low biological
state. Every ordered pair using it is excluded for that subject. Numeric values
cannot remove or reorder a marker on the global panel. The public artifact
contains the support mask, tie diagnostics, margins, and state hashes. The
private ADT-state artifact contains no raw identifier axis and remains outside
the repository.

### Prediction stage

Prediction reads the two public margin artifacts and the public source bundle;
it opens neither private state artifact. The five fixed methods reconstruct a
2-by-2 table at every recipient pair of margins:

- the hierarchical conditional coupling field;
- the matched signed-root Poisson-deviance residual;
- the frozen destroyed-source-link field;
- the stratified common-effect exact conditional MLE;
- the donor-pooled saturated-Poisson interaction.

The primary, destroyed-link, CMLE, and pooled-Poisson coordinates are transported
as source log odds. The residual uses its source-selected multiplier and the
frozen signed-root-deviance normalization. Predictions, axes, margins, and
array hashes are published before joint recipient tables exist.

Margin support is also fixed here. Unsupported subject-marker ADTs contribute
no informative pairs. A subject is eligible with at least 64 nondegenerate
ordered pairs. At least 18 subjects and at least one subject from every batch
are required. This gate uses margins only and precedes recipient RNA--ADT
pairing.

### Score stage

After the prediction tag is independently verified, a populated score
authorization is published. The score attempt is then claimed and tagged. The
score process is the first process allowed to open both private state
artifacts. It verifies every public hash and margin, forms each eligible
subject's joint tables once, and writes no raw identifier or state vector to
the public result.

## Estimand, comparisons, and decision rule

The physical subject is the inferential unit. For each subject, loss is mean
multinomial deviance per cell over its informative ordered pairs. The study
reports donor-equal means. The primary interval resamples subjects with
replacement separately within each of the seven fixed batches, preserves each
batch's observed subject count, and recomputes the donor-equal mean. It uses
20,000 draws and seed 20260829. Batch-block and unstratified donor bootstrap
intervals and a leave-one-batch jackknife-t interval with six degrees of freedom
are reported as sensitivities. Exact sign-flip inference uses the seven batch
mean differences (128 assignments), and every leave-one-batch-out donor-equal
mean is reported. The sign-flip calculation assumes independent batches and
sign-symmetric batch effects under the null; it is a prespecified sensitivity
and gate, not the primary interval.

Confirmation requires both primary comparisons, against the matched residual
and destroyed-link control, to satisfy all six criteria:

- at least 5% relative deviance reduction;
- upper endpoint of the within-batch stratified paired-donor bootstrap 95%
  interval below zero;
- improvement in at least 80% of eligible subjects;
- negative mean paired difference in every batch;
- one-sided exact batch sign-flip p-value at most 0.025;
- negative donor-equal mean difference after leaving out each batch in turn.

The common-effect CMLE and pooled-Poisson comparisons are mandatory
head-to-head results. An estimator-level classical gain is claimed only when
the primary point loss is lower and the same primary 95% interval excludes zero
in the favorable direction against both. A supported analysis that misses a
criterion is published as a completed negative result. QC or support failure
before joint scoring is a terminal refusal. Neither outcome permits an adaptive
retry.

## Public and private provenance

Public artifacts contain repository-relative paths, official accessions,
public subject codes, file names, byte counts, cryptographic hashes, margins,
predictions, and aggregate losses. They contain no local filesystem path, raw
cell identifier, barcode, private state vector, submitter contact field, or
clinical free text. The protocol tag binds the runner, adversarial tests,
candidate history, complete supersession matrix, sequential-candidate
disclosure, amendment, templates, and exact runtime environment. Every later
authorization binds those files transitively by SHA-256. Any positive result is
limited to composition-inclusive cross-condition transfer among
source-QC-matched selected cell mixtures; it does not establish
cell-type-conditional invariance or same-population causal transport.
