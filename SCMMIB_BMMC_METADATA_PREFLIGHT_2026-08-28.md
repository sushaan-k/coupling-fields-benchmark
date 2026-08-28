# SCMMIB BMMC RNA-ADT confirmation: metadata preflight

**Status:** candidate eligible; outcome access not frozen or authorized  
**Dataset:** NeurIPS 2021 BMMC CITE-seq / GSE194122  
**Decision date:** 2026-08-28

## Decision

The dataset supports an honest donor-disjoint confirmation with six untouched
physical donors. It does not support six held donors and at least two physical
donors in each of fit and development: the deposited design contains nine
physical donors, including one donor processed at all four sites. The proposed
split therefore has two fit donors, the four-site bridge donor in development,
and six held donors. This is a viable backup if one-donor development is
acceptable; it is a terminal no-go if two independent development donors are a
requirement.

The site-disjoint alternative remains structurally clean but has only three
held physical donors. Its smallest possible exact sign-test P-value is 0.25
two-sided, so it cannot establish donor-level significance.

## Sources and integrity

The 1,771,132,942-byte SCMMIB v2 archive was not downloaded. Its Figshare file
record is <https://ndownloader.figshare.com/files/49589406> (file 49589406, MD5
`e25a6ac213ddea23dbab1f5ee8caefa9`) in
<https://figshare.com/articles/dataset/27161451/2>.

Preflight used two smaller official files:

| File | Source | Bytes | SHA-256 |
|---|---|---:|---|
| `BMMC_RNA+ADT_p10_metadata.csv.gz` | [SCMMIB commit `5341740b`](https://github.com/bm2-lab/SCMMI_Benchmark/commit/5341740b541c9d8050fb74009c1605aa1bd1b27a) | 348,154 | `b267d4a820b062d0a05227c9cab61d389dcf924c3a6e062fb2389ce1be2f6e4f` |
| `s1d1_filtered_feature_bc_matrix.h5` | [Figshare file 40347877](https://ndownloader.figshare.com/files/40347877) | 22,848,647 | `322c30a7a4905f7f113472442d4aa2c81a1ad736c86651f6c0b81e5b2ff94ac8` |

The feature file's deposited MD5 is
`a99285913ea3f3d22600d3d2f8a88e34`. The metadata URL is pinned to the exact
SCMMIB commit. The dataset is released under CC BY 4.0.

## Donor-disjoint candidate

The allocation is deterministic and outcome-independent. Donor 15078 is the
only physical donor present at all four sites and is reserved for development.
The eight site-specific donors are sorted by SHA-256 of
`GSE194122:SCMMIB-v2:` followed by `DonorID`; the first six are held and the
remaining two are fit.

| Role | Physical donors | Sites | p10 cells |
|---|---|---|---:|
| Fit | 11466, 19593 | 3, 4 | 1,540 |
| Development | 15078 | 1, 2, 3, 4 | 3,067 |
| Held | 10886, 12710, 13272, 16710, 18303, 28045 | 1, 2, 3, 4 | 4,419 |

The roles are physical-donor-disjoint and intentionally not site-disjoint.
Every role contains all ten deposited broad lineages. With six held donors, an
all-positive exact sign test reaches P = 0.03125 two-sided and P = 0.015625
one-sided. The unit of that inference is the physical donor, never a site,
batch, cell, marker pair, or technical aliquot.

The original `is_train` labels are ignored. All model choices, state rules,
penalties, gates, and comparisons must be fixed using fit and development only.
After that lock, fit and development may be combined for one refit and the six
held donors scored once.

## Site-disjoint alternative

For a pure held-site stress test, fit sites 1 and 2 after removing donor 15078,
develop on site 4 after removing donor 15078, and hold all of site 3. This gives
four fit donors, two development donors, and three held donors. It is suitable
for descriptive held-site performance, not donor-level exact significance.

## Matched marker panel

The s1d1 feature schema contains 36,601 RNA genes and 140 ADT features. After
removing six isotype controls, 134 biological ADTs remain; 37 names match an
RNA gene exactly. The locked ten-marker panel is `CD4`, `CD7`, `CD14`, `CD19`,
`CD33`, `CD38`, `CD44`, `CD47`, `CD52`, and `CD93`, yielding 100 ordered
RNA-by-ADT pairs. This panel was chosen from feature names and biological
coverage only.

## Access audit

Decoded material was limited to the SCMMIB p10 metadata and the s1d1 feature
names and types. The metadata include deposited per-cell QC summaries such as
total ADT counts; no feature-level expression or protein value was read. The
s1d1 HDF5 file was hashed opaquely, but `matrix/data`, `matrix/indices`, and
`matrix/indptr` were not opened. No feature-level outcome array, held 2-by-2
table, or held coupling statistic was decoded for either proposed holdout. The
large SCMMIB archive was not downloaded. No COVID/Sanger path or artifact was
accessed.

The machine-readable record is
`results/development/scmmib_bmmc_metadata_preflight.json`. It is a preflight,
not a public freeze, preregistration, prediction artifact, or scored result.
