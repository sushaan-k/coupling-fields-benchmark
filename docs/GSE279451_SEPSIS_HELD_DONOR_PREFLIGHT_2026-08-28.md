# GSE279451 sepsis CITE-seq held-donor preflight

**Date:** 2026-08-28  
**Status:** viable candidate; all matrix outcomes disabled  
**Machine-readable record:**
`data/development/gse279451_sepsis/metadata_preflight_v1.json`

## One candidate

The additional candidate is `GSE279451`, the adult PBMC CITE-seq arm of Ye et
al., *Nature Immunology* 2026, DOI `10.1038/s41590-025-02345-x`. GEO deposits
40 separately named adult samples: 3 healthy controls, 5 ICU controls, and 32
sepsis samples (14 abdominal, 14 respiratory, and 4 urinary). Each sample has a
10x barcode axis, a joint gene/antibody feature axis, and one sparse MTX count
file. This candidate is not Arce, PoKI, Lawlor, Hao, Kotliarov, either SCMMIB
candidate, or the NeurIPS BMMC study.

This preflight opened only primary-source GEO/Nature metadata, all barcode
axes, all feature axes, and transfer headers. No `matrix.mtx.gz` file or byte
range was downloaded, indexed, or decoded. The candidate is viable for a
prospective confirmation, but it is not yet an executable public freeze.

## Concrete acquisition

- GEO record: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE279451>
- Complete processed-count tar:
  <https://ftp.ncbi.nlm.nih.gov/geo/series/GSE279nnn/GSE279451/suppl/GSE279451_RAW.tar>
- Primary paper: <https://doi.org/10.1038/s41590-025-02345-x>
- GEO MINiML metadata:
  <https://ftp.ncbi.nlm.nih.gov/geo/series/GSE279nnn/GSE279451/miniml/GSE279451_family.xml.tgz>
- Per-donor files:
  `https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8571nnn/<GSM>/suppl/<filename>`

FTP reports exactly 2,334,412,800 bytes for the raw tar. The 40 individual
matrix gzip files total 2,319,077,199 bytes; barcode and feature axes total
15,230,404 bytes. The largest single matrix gzip is 85,929,901 bytes. The
workspace had 3,750,584,320 free bytes at preflight, so the tar fits, but
retaining both the tar and a full extraction is unsafe. A reducer must stream
one donor at a time, verify it, emit only the locked reduced representation,
and remove the source MTX before acquiring the next donor.

## Feature and cell axes

Every sample has 37,695 features: 37,487 `Gene Expression` and 208 `Antibody
Capture`. All 40 decompressed feature files are byte-identical with SHA-256
`ff6a914dd33b3a3c2dd913ed439ed4b150fd8ab210595dec2447a283eb9b417b`.
The barcode axes contain 330,112 cells, with 5,570 to 11,922 cells per donor.

The exact overlap with the already frozen ten-marker BMMC biology panel is:

`CD4, CD7, CD14, CD19, CD33, CD38, CD44, CD47, CD52`

All nine occur exactly once on both the RNA and ADT axes in every sample.
`CD93` occurs on the RNA axis but not the ADT axis and is excluded without
replacement. The fixed panel is therefore all 81 ordered RNA-marker by
ADT-marker pairs. No fuzzy aliases or outcome-dependent substitutions are
allowed.

## Donor split

The split is fixed from metadata only. Within each of five deposited strata
(`healthy`, `ICU`, `abdominal`, `respiratory`, `urinary`), sort ascending by
`SHA256("gse279451-hierarchical-v1" + GSM accession)`. The first `floor(n/2)`
are development and the remainder are held.

| Role | Donors | Barcode-axis cells | Healthy | ICU | Abd. | Resp. | Urinary |
|---|---:|---:|---:|---:|---:|---:|---:|
| Development | 19 | 159,540 | 1 | 2 | 7 | 7 | 2 |
| Held | 21 | 170,572 | 2 | 3 | 7 | 7 | 2 |

Development accessions:
`GSM8571043, GSM8571044, GSM8571047, GSM8571048, GSM8571049, GSM8571052,
GSM8571055, GSM8571056, GSM8571060, GSM8571061, GSM8571065, GSM8571068,
GSM8571072, GSM8571073, GSM8571074, GSM8571075, GSM8571077, GSM8571079,
GSM8571081`.

Held accessions:
`GSM8571042, GSM8571045, GSM8571046, GSM8571050, GSM8571051, GSM8571053,
GSM8571054, GSM8571057, GSM8571058, GSM8571059, GSM8571062, GSM8571063,
GSM8571064, GSM8571066, GSM8571067, GSM8571069, GSM8571070, GSM8571071,
GSM8571076, GSM8571078, GSM8571080`.

Nature Supplementary Table 1 contains exactly 40 unique adult rows marked for
CITE-seq, and its `Donor ID` set exactly equals the 40 GEO sample names. This
primary metadata verifies one biological donor per GSM. The workbook URL,
69,665-byte size, SHA-256, sheet, and the only two columns read are bound in the
machine-readable preflight and source template; no clinical outcome column was
used.

## Prospective estimator protocol

Use the audited estimator in
`mapreg/hierarchical_conditional_coupling.py`. RNA states are raw-UMI positive
versus zero. The primary assay-depth budget is exactly 1,024 cells per donor,
chosen before matrix access by ascending
`SHA256("GSE279451-CELL-BUDGET-v1" + accession + sample + barcode)`. ADT states are deterministic donor-wise mid-ranks, with ties
broken by a frozen SHA-256 rule. This creates an exact 2 by 2 table for each of
the 81 ordered entities in each donor. No additional primary cell QC or lineage
assignment is allowed. A malformed or source-inconsistent axis is a terminal
refusal, not a branch that filters or replaces cells. No author protein cluster
or paired-modality label may enter tuning or support selection.

Tune the existing hierarchical penalties by leave-one-development-donor-out
deviance across the 19 development donors, donor-weighted equally, then refit
once on all development donors. The held estimator receives only separately
formed RNA and ADT margins. Pairing-derived held tables are formed once, after
the prediction artifact is public and a commit-addressed authorization is
issued. Predictions must be invariant to arbitrary held within-donor ADT
barcode permutations.

An all-deposited-cell analysis is secondary and may run only after the held
decision. It cannot reselect a model, alter support, repair a refusal, or
replace the 1,024-cell primary result.

The two held gate comparisons are held multinomial deviance per cell against
the strongest development-selected classical conditional-interaction residual
and the destroyed-link hierarchical graph control. Hierarchical ridge-only,
common-effect, permuted-graph, and independence controls remain mandatory
diagnostics but are not additional held promotion gates. A held donor is
scoreable only if at least 64 of 81 entities are margin-informative; any donor
failure is a terminal confirmation refusal rather than a donor deletion.

## Power and promotion boundary

The 21 held donors are the only biological replicates. For each gate
comparator, the exact one-sided test exhausts all `2^21` sign flips of the
paired donor loss differences, uses the mean of primary minus comparator loss
as its statistic, and counts the inclusive lower tail; zero differences retain
both tied sign assignments. The threshold is `p <= 0.025`. This is not a
binomial sign test. Sign-flip power, effect-size power, and paired-loss-variance
power cannot be estimated without prohibited donor loss differences.

A future frozen confirmation should require all of the following: at least 5%
mean held deviance reduction against the strongest classical residual; a 95%
paired donor-bootstrap upper endpoint below zero; exact one-sided sign-flip
`p <= 0.025`; at least 16 of 21 donors favorable; and the same directional
gate against the destroyed-link control. Ridge-only and common-effect remain
reported diagnostics and are not held promotion gates.
The exact sign-flip, 5% reduction, bootstrap endpoint, and 16-of-21 conditions
apply only against the strongest residual and destroyed-link comparator. No
claim is earned by cell-level sample size alone.

## Current stop condition

This preflight authorizes no outcome access. The protocol, source template,
reducer, evaluator, one-shot runner, model/comparator configuration, poison
tests, candidate designation, family policy, and disabled authorization
template are now complete and checksum-bound in the public release working
tree. The active source manifest and both terminal development attempt markers
do not exist. Nonheld count acquisition remains disabled until this exact plan
is published; held acquisition remains disabled until the later immutable
prediction authorization.
