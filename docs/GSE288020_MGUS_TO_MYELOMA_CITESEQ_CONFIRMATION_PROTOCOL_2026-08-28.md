# GSE288020 MGUS-to-myeloma linked RNA/ADT confirmation protocol

Frozen after public metadata review, opaque source download and hashing, TAR
member listing, and inspection of only three HDF5 feature-schema datasets in
each donor file. No barcode dataset, sparse-matrix structural dataset, RNA
count, or ADT count was opened before this freeze.

## Question and scope

Can a conditional RNA-protein coupling field learned in MGUS bone-marrow
mononuclear cells predict cell-matched dependence in donors with multiple
myeloma after each recipient's RNA margin and the fixed ADT rank margin are
restored? GSE288020 contains one linked 10x RNA/ADT HDF5 file for each of 23
physical donors: 14 with MGUS and nine with multiple myeloma. The test is a
donor-disjoint disease-shift confirmation within one study, not an independent
laboratory replication.

The physical donor is the inferential unit and each donor has a separate 10x
library. Disease and immune-age labels do not enter the estimator. Disease
defines the held shift; immune age is used only to balance development and to
require support in both strata.

## Sequential-candidate disclosure

GSE288020 was selected in a public sequential search. Its three immediate
terminal predecessors were GSE314416, which failed its pilot promotion
diagnostic; GSE158769, which refused on source schema; and GSE189050, which
refused on development QC/support. Their terminal tags are serialized in every
GSE288020 artifact. The earlier scored, negative, and refused candidates are
enumerated in the byte-bound `docs/FINAL_PUBLIC_EVIDENCE_LEDGER.md`; the ledger
and three immediate predecessors jointly disclose the prior public campaign.
All intervals and tail probabilities in this protocol are candidate-specific.
No familywise adjustment or campaign-wide confirmatory error control is claimed
across that search. This disclosure was made before any GSE288020 barcode or
count value was opened.

## Pre-freeze source access

The public `GSE288020_RAW.tar` archive is 347,228,160 bytes and contains 23 H5
members totaling 347,203,083 bytes. The source manifest binds the archive,
every member's byte count and SHA-256 digest, GEO's 2,216-byte file list, and
the 65,720-byte official GEO sample-metadata response. Before any HDF5 dataset
was opened, the archive was downloaded, byte-hashed, listed, and extracted.
The protocol preflight streams every TAR member and verifies that its byte count
and SHA-256 digest match the corresponding extracted H5.

The official GEO response is an upstream input retrieved from the URL and hash
in the source manifest. It is not copied into the public tag because the full
response contains submitter contact information. The bound runner parses only
GSM accession, title, disease state, immune-age label, library name, and
supplementary filename. The public candidate designation contains those derived
sample fields and no submitter contact data. Its test reconstructs all 23
records, including the sole normalization below, from a synthetic response with
the same declared schema.

Schema inspection then opened exactly these datasets in all 23 files:

- `/matrix/features/id`;
- `/matrix/features/name`;
- `/matrix/features/feature_type`.

The feature schemas are identical: 36,601 gene-expression features and 55
antibody-capture features, with schema SHA-256
`c1c610528da078a489d473e33c11c8c9d6fef4be4b80f42954d521140cb520fa`.
The preflight did not open `/matrix/barcodes`, `/matrix/data`,
`/matrix/indices`, `/matrix/indptr`, or `/matrix/shape`. The access record and
schema preflight are immutable protocol bindings.

GSM8757540 has `mmune-Age Young` in one GEO characteristic because the initial
`I` is missing; the sample title states `Immune-Age Young`. That title supplies
the normalized age stratum. No other metadata correction is made.

## Frozen split

All 14 MGUS donors form development. Within each GEO immune-age stratum, donors
are ordered by SHA-256 of
`GSE288020-MGUS-SPLIT-v1|<age>|<donor>`, with donor identifier as an exact-hash
tie break. The first four young and first three old donors form calibration;
the remainder form pilot. This produces:

- calibration: `R001,R005,R008,R009,R010,R013,R014` (four young, three old);
- pilot: `R003,R006,R015,R016,R020,R023,R024` (three young, four old);
- held: all nine MM donors, `E2228,E2238,E2242,E2243,E2263,E2324,E2326,E2328,E2329`
  (four young, five old).

No donor appears in more than one role. Calibration and pilot outcomes may be
read only after the protocol tag is public and independently verified. Held
barcodes and matrix datasets remain closed until a public development pass.

## Panel and state construction

The primary cross-lineage BMMC panel contains 16 conventional cognate RNA-ADT
mappings and scores their 16-by-16 Cartesian product (256 ordered pairs):

`PTPRC-CD45`, `CD3E-CD3`, `CD4-CD4`, `CD8A-CD8a`, `MS4A1-CD20`,
`CD19-CD19`, `CD27-CD27`, `CD38-CD38`, `SDC1-CD138`, `CD14-CD14`,
`ITGAM-CD11b`, `ITGAX-CD11c`, `CD33-CD33`, `NCAM1-CD56`,
`KLRB1-CD161`, and `CXCR4-CXCR4`.

Each RNA feature must match its exact Ensembl ID, symbol, and `Gene Expression`
type. Each antibody must match its exact feature ID and `Antibody Capture`
type. A missing, repeated, or shifted mapping is terminal. At least 192 of 256
ordered pairs must have a nondegenerate RNA margin for a donor loss to be
defined; the ADT rank margin is exactly 256/256 by construction. Every stage
serializes each donor's 16-marker support mask, informative marker identities,
and resulting ordered-pair count.

RNA state is `count > 0`. For each donor and ADT marker, the 512 selected cells
are sorted by raw count, a frozen salted SHA-256 tie rank, and barcode; the
upper 256 define the high state. Recipient predictions therefore restore the
observed RNA margin and the exact 256/256 ADT rank margin. Destroyed-link
controls rotate the complete ADT-state vector once along an independently
salted cell order, preserving every donor-marker margin.

The estimand is the conditional RNA-ADT log odds in each within-donor 2x2 table,
given that donor's RNA-detection margin and the fixed ADT rank margin. A
recipient prediction is the expected table at those recipient margins. The
loss is donor-equal mean Poisson deviance over ordered pairs with nondegenerate
recipient RNA margins. This state estimand is serialized in every artifact.

## Cell QC, sampling, and attrition

The input is the submitted Cell Ranger filtered matrix. A cell is eligible when
it has at least 200 detected RNA genes, mitochondrial RNA fraction at most
0.10, and total RNA UMIs at most 70,000, matching the study's reported
single-cell filters where they are expressible from the submitted matrix.
There is no threshold fitting, clustering, label transfer, manual rescue, or
yield-based change. The 512 eligible barcodes with smallest salted SHA-256
ranks are selected per donor.

Calibration and pilot each require at least six retained donors and at least
two in each immune-age stratum. Held prediction requires at least seven donors
and at least three in each age stratum. Every retained donor must supply 512
selected cells and at least 192 informative ordered RNA-ADT pairs. Feature
integrity, unique barcodes, valid CSC structure, and nonnegative integer counts
are mandatory. A failed development support gate is terminal before any held
HDF5 matrix or barcode dataset is opened. The RNA-summary firewall computes
the held informative-pair count from RNA margins alone; a failed held support
gate is terminal before scoring.

## Estimator and prespecified controls

The primary estimator is the donor-heterogeneity conditional coupling field.
Calibration fits the Cartesian grid of graph neighbors `1,2`, heterogeneity
penalty `0.1,1,10`, ridge penalty `0.01,0.1`, graph penalty `0,0.1,1`, and
transport multiplier `0.5,0.75,1,1.25` (144 nominal configurations). Donor-
equal pilot deviance selects one complete configuration with lexicographic tie
breaking. No held value changes the grid, panel, state rule, or selection.

The two primary promotion controls are:

- the pilot-selected signed Pearson or signed-root Poisson-deviance residual,
  using the same transport grid;
- a coupling field fitted after the frozen within-donor destroyed-link
  rotation.

Two matched classical interaction estimators are prespecified:

- common-effect stratified conditional maximum likelihood;
- donor-pooled saturated Poisson log-linear interaction, equal to the pooled
  log odds ratio and refused when a pooled table cell is zero.

Each classical estimator is fit on calibration, selects only its transport
multiplier from `0.5,0.75,1,1.25` on pilot, and is refit on all retained
development donors after valid support and deterministic selection. The
primary, destroyed-link, and residual models are refit on the same retained
axis. This internal GSE288020 refit may contain 12--14 retained donors and
occurs even when the pilot promotion diagnostic fails.

The canonical source-only export declares exactly five external-study slots:
primary, signed-root Poisson-deviance residual with source-selected transport,
destroyed link, common-effect CMLE, and pooled saturated Poisson. Each slot is
marked valid or carries an explicit refusal. The artifact separately reports
`core_passes`, `classical_head_to_head_ready`, and `external_study_ready`.
External-study eligibility requires all 14 designated MGUS donors, the complete
7/7 calibration-pilot split, and valid models in all five slots;
`external_study_ready` is the conjunction of those requirements, and the legacy
`passes` field has the same value. A 12- or 13-donor source refit remains valid
for the internal GSE288020 held protocol but is ineligible for external-study
target access. The export records per-pair support, numerical certificates, and
explicit non-use of MM or external-study values. A promotion failure still
forbids all GSE288020 held access. All valid methods are reconstructed at
identical recipient margins and scored by donor-equal Poisson deviance. Results
and refusals are reported regardless of direction.

## Promotion diagnostic and held inference

The primary must beat both promotion controls separately by every criterion:

1. at least 5% reduction in donor-equal mean deviance;
2. upper endpoint below zero in a 20,000-draw paired donor bootstrap interval,
   seed `20260828` (with fixed comparator-specific offsets);
3. lower loss in at least `ceil(0.8 n)` donors;
4. one-sided exact donor sign-test `p <= 0.025` after exact ties are discarded;
5. negative mean paired difference in both immune-age strata.

Configuration and transport selection use the same retained pilot donors on
which promotion is evaluated. Pilot bootstrap intervals and exact-binomial
tails are therefore deterministic post-selection promotion diagnostics, not
confidence intervals or hypothesis tests. Only the one-shot held comparison is
inferential, with the physical donor as its replication unit.

Calibration and pilot retain 6--7 donors each; held prediction retains 7--9.
All 12--14 retained development donors enter the fixed source refit. Exact ties
are removed from the sign-test denominator but remain in retained `n` for the
`ceil(0.8 n)` favorable-count requirement. Consequently, a six-donor pilot
requires six favorable non-ties. A seven-donor pilot requires seven favorable
non-ties, or six favorable non-ties plus one exact tie. Held `n=7` has the same
rule; held `n=8` requires eight favorable non-ties, or seven plus one tie; held
`n=9` requires at least eight favorable non-ties. These consequences are
predeclared and serialized with retained `n`, non-tied `n`, and tie count.

Added value over each classical interaction estimator is a prespecified
head-to-head result, distinct from the transfer-confirmation gate. It requires
both a lower donor-equal mean deviance and a paired donor-bootstrap upper 95%
endpoint below zero. No arbitrary percentage threshold is imposed. Exact sign
statistics, age-stratum means, effect sizes, and intervals are still reported.
For held donors, the bootstrap interval is a candidate-specific paired-donor
95% confidence interval and the exact-binomial tail is a candidate-specific
test. The final artifact reports `transfer_confirmation_pass` separately from
`gain_over_both_classical_interactions`; only their conjunction supports a
claim of transfer plus added value over both classical interactions.

## Combined-HDF5 boundary

Each donor's RNA and ADT counts occupy one CSC `matrix/data` dataset. A held
RNA stage therefore cannot truthfully claim that the raw HDF5 or all physical
chunks containing ADT elements were untouched. After a public development
pass, a separately invoked, protocol-bound child RNA-summary firewall may open
the held combined HDF5, decode its CSC arrays, apply RNA-only QC, and return
only:

- the selected-cell-axis hash;
- the binary RNA-state hash;
- 16 RNA-positive margins;
- aggregate eligibility counts and an explicit access audit.

The child returns no barcode, cell-level RNA matrix, ADT value, ADT statistic,
combined-matrix sparsity statistic, or ADT-derived margin. No numeric validation
is applied to ADT entries. Global CSC structure remains
an unavoidable shared-file integrity check, but no structural statistic is
returned. Its parent rejects forbidden payload keys and freezes all predicted
tables before held scoring. A poison test changes valid held ADT magnitudes and
sparsity while holding RNA fixed and requires a byte-identical prediction
payload. The prediction artifact explicitly states that ADT values were
co-resident in decoded CSC data; it must never be described as blinded or as
zero raw ADT access. The enforceable claim is zero ADT value or statistic
returned to the prediction process and zero ADT value or statistic serialized
or logged.

## Public barriers

The exact assay runtime is CPython 3.9.6, NumPy 2.0.2, SciPy 1.13.1, h5py
3.14.0, and HDF5 runtime/build 1.14.6 with h5py API 1.8. The bound runtime
artifact and implementation encode this fingerprint. A mismatch refuses before
tag verification and before a `STARTED` attempt is claimed. The observed exact
fingerprint is serialized in every attempt and result.

1. Publish and independently verify annotated tag
   `gse288020-citeseq-v1-protocol`, binding the complete runtime closure,
   environment, tests, source hashes, designation, access record, feature
   preflight, and prior public campaign ledger. No barcode or matrix dataset may
   be opened before verification.
2. Run calibration and pilot only. A support failure publishes a terminal
   artifact without a model. After valid support and deterministic selection,
   refit the source-only external-study model on every retained MGUS donor and
   serialize it regardless of the promotion diagnostic. A promotion failure is
   terminal for GSE288020 held access. A pass permits publication of
   `gse288020-citeseq-v1-development` for independent verification.
3. Invoke the bound RNA-summary firewall on held donors and freeze predictions.
   On held-RNA QC failure, publish terminal status and stop. Otherwise publish
   and independently verify `gse288020-citeseq-v1-predictions` before scoring.
4. Run the one-shot held score, report pass or failure and all four matched
   comparisons, and stop without retuning.

Every assay stage accepts only its exact frozen output path. After runtime and
tag authorization, but before the first source open, it creates an append-only
`STARTED` record. Stage bodies return data and cannot write outputs. The stage
controller writes exactly one new JSON artifact through an atomic temporary-file
link, converts every body exception into one terminal payload, and appends one
`FINISHED` record with its status and SHA-256. An unexpected body-written output
is discarded and terminalized. The existence of an output or attempt ledger
prevents a second run.

Before any downstream attempt, the runner validates the preceding public tag
and the complete two-record ledger semantically: exactly `STARTED` then
`FINISHED`, exact authorization commit, stage, output path, runtime fingerprint,
result hash, and result status. It also verifies the authorization commit,
stage, runtime, and permitted status inside the result. Any mismatch refuses
before a downstream attempt is claimed.

Every public-stage runner verifies that every file in the complete runtime,
test, environment, protocol, and provenance binding is byte-identical to the
corresponding remote annotated tag. An adversarial test mutates each bound file
in turn and requires rejection.
