# Stephenson unused-Cambridge held-donor confirmation protocol

**Pre-outcome status (2026-08-29):** designated with numeric matrix access
disabled. This protocol, the executable runner, its focused tests, the compact
classical fields, and the candidate designation must be committed, tagged, and
verified from a fresh public clone before a separate authorization can permit
RNA-margin access. No `unused_source` matrix value may be decoded before then.

## Question and fixed panel

Can source fields frozen from 12 calibration and 24 pilot Cambridge donors
predict linked RNA--protein states in the 11 Cambridge donors assigned
`unused_source` by the original public allocation?

The evaluation panel is exactly `CV0073`, `CV0094`, `CV0100`, `CV0134`,
`CV0176`, `CV0180`, `CV0200`, `CV0201`, `CV0257`, `CV0911`, and `CV0940`.
Donors cannot be removed or replaced. Each contributes the original
deterministically selected 512 cells, nine RNA markers, nine ADT markers, and 81
ordered RNA--ADT pairs. RNA state is raw count greater than zero. Within each
donor and marker, the original count-order and SHA-256 tie rule assigns exactly
256 low and 256 high ADT cells.

The original public allocation named all 11 donors before the Cambridge pilot
or Newcastle score. The versioned development path selected only `calibration`
and `pilot`; the original prediction and score selected only `held_site`. The
post-hoc classical-field audit likewise read calibration, pilot, and Newcastle
rows only. The pre-access artifact replays these exclusions without opening an
H5AD.

## Frozen methods

No estimator is fitted or tuned in this campaign. It reuses:

1. the hierarchical exact conditional field frozen in
   `results/development/stephenson_citeseq_development.json`;
2. its selected signed-root Poisson-deviance residual comparator;
3. its margin-preserving destroyed-link field;
4. the common-effect exact conditional CMLE; and
5. the donor-pooled saturated Poisson interaction.

The last two fields and their source-only selected transport multipliers are
copied exactly into
`data/development/stephenson_unused_cambridge/classical_fields_v1.json`. The
saturated Poisson interaction is the pooled sample log odds of each 2-by-2
table. The compact artifact records its nonconfirmatory post-hoc provenance;
the complete source audit is distributed as
`results/development/classical_interaction_baselines_posthoc.json`, and the
runner verifies every compact field against that artifact. The present donor
panel supplies these fields' first outcome-blind evaluation.

From the repository root, the committed pre-access record is replayed without
opening the H5AD by running
`python3 -m experiments.confirm_stephenson_unused_cambridge verify-preaccess --check-existing`.

## Remote source contract

The only numerical input is the official E-MTAB-10026 H5AD, 7,187,322,881
bytes, SHA-256
`ec48f328f2e884c23376c8aa1f26041e11625762be5c30b0bd0869aa8bb1a334`.
The runner accepts either that exact local file or the fixed BioStudies HTTPS
URL recorded in the checksum-bound source manifest. A local file is streamed
once to recompute its complete SHA-256. For remote use, an HTTP HEAD request
must resolve to the frozen EBI range endpoint with the exact content length and
`Accept-Ranges: bytes`, ETag, and modification time; the manifest supplies the
checksum from the previously downloaded complete object. A seekable `fsspec`
HTTP file then reads that resolved endpoint without downloading a second full
copy.

Metadata, RNA, and ADT are opened in separate handles. Prediction opens metadata
and RNA only. Score opens metadata, RNA, and ADT sequentially. Sparse row
indices may be inspected to locate the nine frozen feature columns; `/data`
values are requested only at those feature positions. No unrequested
featurewise statistic enters any model.

## One-shot access boundary

Prediction authorization must bind repository-relative paths and exact public
bytes for the runner, test, protocol, designation, pre-access result, compact
fields, original source manifest, original development result, and numerical
modules. It also binds the implementation commit. The runner verifies each
local byte against that immutable public commit and verifies the authorization
itself at a later public commit.

After source identity verification, prediction writes an exclusive terminal
attempt record before the first HDF5 open. It reads only the nine RNA features,
forms 9-by-2 margins, combines them with the predetermined `[256,256]` ADT
margins, and publishes all five methods' expected tables. ADT numeric values,
RNA--ADT pairings, and truth tables remain unavailable.

Score requires a separately committed authorization that binds the exact public
prediction bytes. After source identity verification it writes an exclusive
terminal attempt before the first HDF5 open, then reads RNA and ADT once and
scores the frozen tables. An exception after an attempt is terminal. Existing
attempt, result, or refusal artifacts prohibit a rerun.

## Loss, inference, and gates

Loss is donor-equal mean multinomial deviance per cell over at least 64
informative marker pairs, identical to the original Stephenson analysis. Each
comparison uses 20,000 fixed-seed paired donor-bootstrap draws and an exact
one-sided donor-level paired sign-flip test over all 2,048 assignments.

The common-effect exact conditional CMLE and selected residual are the two
primary inferential comparators. Each comparison must show at least 5% lower
mean loss for the hierarchical estimator, a paired-bootstrap 95% upper endpoint
below zero, and exact one-sided paired sign-flip `p <= 0.05`. The destroyed-link
validity control must meet the same effect-size and confidence-interval
criteria. Favorable-donor counts and all inferential statistics are reported,
but no separate favorable-count threshold is imposed.

The donor-pooled saturated Poisson interaction is a mandatory reported
secondary comparator and does not veto confirmation. The field-transfer gate
requires the selected-residual inferential pass and the destroyed-link validity
pass. The hierarchical-increment gate requires the exact-CMLE inferential pass.
Full confirmation requires both gates. Every completed result is published
unchanged, whether it passes or fails.

An earlier local draft treated all four comparators as co-primary and added a
10-of-11 favorable-donor veto. It was retired before any unused-donor matrix
access. The final rule makes the exact CMLE the test of hierarchical shrinkage,
the selected residual the test of fixed-margin transfer, and the pooled
Poisson fit a mandatory secondary estimate. Requiring both primary comparisons
to pass is an intersection-union rule, so the two tests do not require a
multiplicity correction for the composite claim. The paired sign-flip test uses
the loss differences rather than discarding their magnitudes; the bootstrap
interval and 5% effect threshold remain additional requirements.

## Interpretation

A pass is a prospective donor-held confirmation within the Stephenson study. It
is not an independent-study or cross-site replication. A failure is terminal
and cannot be repaired by changing donors, markers, thresholds, fields,
transport multipliers, or comparator roles.
