# GSE299043 held-site RNA-protein confirmation protocol

**Version:** 1.0, 28 August 2026
**Candidate:** mesenteric lymph node CITE-seq, `GSE299043`
**Study:** Trzupek et al., *Nature Immunology* (2025), DOI
`10.1038/s41590-025-02241-4`
**Outcome status:** disabled; no H5AD member has been opened

## Question

This experiment tests whether RNA-protein coupling learned from ten Cambridge
organ donors predicts same-cell coupling in ten donor-disjoint recipients
collected through LiveOnNY and processed at Columbia. Prediction is conditional
on each recipient's observed RNA and protein margins. The primary comparison
is against the strongest matched signed Pearson or Poisson-deviance interaction
residual, not against an independence-only baseline.

The candidate becomes active only after the terminal GSE279451 development
refusal. Its donor split, tissue, marker IDs, cell selection, demultiplexing,
estimators, controls, gates, code, and source-member manifest must be published
under an immutable release before the first H5AD is opened.

## Cohort and source seal

The biological unit is one physical donor. Development donors are `591C`,
`621B`, `637C`, `640C`, `647C`, `689C`, `694B`, `759B`, `768B`, and `778C`.
Held donors are `D512`, `D520`, `D523`, `D528`, `D529`, `D533`, `D534`,
`D543`, `D564`, and `D570`. All included libraries use 10x 5-prime v2 with
TotalSeq-C and have deposited sample metadata containing mesenteric lymph node.
The development and held sets therefore differ by donor, acquisition program,
site, and processing laboratory.

The source manifest fixes 207 GEO H5AD URLs and HEAD-derived lengths:
56 development members totaling 2,991,542,178 bytes and 151 held members
totaling 4,766,004,153 bytes. Its metadata table has SHA-256
`dfc929364a40895620c39897a671542670f1cf1e89058cf4ff02f51d16c86933`.
Members are downloaded, byte-counted, hashed, reduced, and deleted one at a
time. A transient transport failure permits at most three total download
attempts, with the partial file deleted before each retry. Byte-count, decode,
schema, and analysis failures are never retried. Development acquisition may
request only the 56 Cambridge members. Held members require a passing
development result, a publicly committed prediction, commit-bound
authorization, and a terminal score-attempt marker.

## H5AD and tissue contract

The deposited objects are the author's aligned Cell Ranger outputs. `X` is a
cell-by-feature raw UMI matrix. Cell barcodes are the observation index;
donor, site, and tissue labels are not taken from observations. Feature
identity uses `var/gene_ids` together with `var/feature_types`, never the
possibly suffixed display name. Selected values must be finite, nonnegative,
and integer-valued.

HTOs are `Antibody Capture` features whose IDs begin with the file's donor ID
and a hyphen. At least two donor HTOs are required before HashSolo
classification. The sole exception is the metadata-preflight row for
`GSE299043_694B_001.CZI-IA11512689.v2.h5ad`, whose tissue field is exactly
`pooled:mesenteric lymph node`. All cells in that member are assigned to MLN
only when its donor is `694B` and its sole normalized donor HTO is exactly
`694B-MLN-206`; every other panel with fewer than two donor HTOs is a terminal
refusal. This exception is bound to the metadata-table SHA-256 above and frozen
in the source manifest. Otherwise, HashSolo is run independently within each
H5AD with priors `(0.05, 0.70, 0.25)`, no pre-existing clusters, and the number
of noise tags equal to the tag count minus one. Only singlets assigned to a
frozen MLN tag are eligible. The exact tags and the four-library
`759B-MLN-1` to `759B-MLN-263` normalization are fixed in the source manifest.
Per-library classifications are pooled by donor using `filename + barcode` as
the cell key. No RNA or nine-marker ADT value enters tissue assignment or cell
selection.

## States and entities

The marker order is `CD4`, `CD7`, `CD14`, `CD19`, `CD33`, `CD38`, `CD44`,
`CD47`, and `CD52`. The exact RNA feature IDs are
`ENSG00000010610`, `ENSG00000173762`, `ENSG00000170458`,
`ENSG00000177455`, `ENSG00000105383`, `ENSG00000004468`,
`ENSG00000026508`, `ENSG00000196776`, and `ENSG00000169442`. The matched
ADT feature IDs are `C0072`, `C0066`, `C0081`, `C0050`, `C0052`, `C0389`,
`C0073`, `C0026`, and `C0033`. Each ID must have one exact match of the
declared feature type. All 81 ordered RNA-marker by ADT-marker entities are
retained.

Each donor contributes exactly 512 MLN singlets: eligible cells are sorted by
ascending SHA-256 of `GSE299043-MLN-CELL-BUDGET-v1 + donor + filename +
barcode`, and the first 512 are retained. Fewer than 512 cells is a terminal
support refusal. RNA state is one for a raw UMI count greater than zero. Within
each donor and ADT marker, cells are ordered by count and then by SHA-256 of
`GSE299043-MLN-ADT-v1 + donor + filename + barcode + marker`; exactly 256
receive the lower state and 256 the upper state. Each entity is a 2-by-2 table
totaling 512 cells. A donor is scoreable only if at least 64 entities have
nondegenerate fixed margins.

## Estimators and model selection

The primary estimator is donor-heterogeneity-aware exact conditional log odds
regularized over RNA and ADT marker graphs constructed only from marginal
profiles. Every development fold rebuilds the graphs and refits every method
on nine donors. The fixed grid is:

- graph neighborhood size: `1, 2, 3`;
- heterogeneity penalty: `0.1, 1, 10`;
- ridge penalty: `0.01, 0.1`;
- graph penalty: `0.1, 0.3, 1`;
- transport multiplier: `0.75, 1, 1.25`.

Selection uses ten-fold leave-one-development-donor-out prediction. Loss is
multinomial deviance per cell, averaged over informative entities within donor
and then equally over donors. Ties use the first tuple in the declared grid.
Every fitted conditional model must pass convergence, gradient, boundary,
curvature, and condition-number certificates.

The classical family contains signed square-root Pearson and signed-root
Poisson-deviance interactions, each raw or centered by its exact fixed-margin
hypergeometric null, crossed with the three transport multipliers. Source
coordinates are divided by `sqrt(n)`, pooled equally over informative donors,
multiplied by `sqrt(m)` in the recipient, and inverted at the recipient's
observed margins. The strongest candidate is selected by the same donor folds
and loss as the primary.

Controls are destroyed-link hierarchical coupling, hierarchical ridge-only
coupling, label-permuted marker graphs, and independence. Destroyed links use a
fixed within-donor ADT-cell permutation that preserves every marginal table.
The required development families are primary, strongest classical residual,
destroyed link, and hierarchical ridge-only. An unavailable required family is
a terminal evaluation refusal. Label-permuted graphs and independence are
reported but do not block an otherwise complete decision.

## Promotion gates

For each gate comparison, `d` is primary donor loss minus comparator donor
loss. Relative reduction is one minus the ratio of donor-equal mean losses.
Intervals use 20,000 shared donor bootstrap resamples from NumPy
`default_rng(20260828)` and linear 2.5th and 97.5th percentiles. Zero differences
are not favorable.

Development passes only if the primary beats each of the strongest classical
residual, destroyed-link, and hierarchical ridge-only controls by at least 5%,
the paired bootstrap upper endpoint is below zero, and at least 8 of 10 donor
differences are negative. Failure closes the candidate without held access.
Because model selection and this gate use the same ten development donors, the
development interval is a promotion heuristic, not confirmatory inference.

After a development pass, `predict` writes the all-development frozen models
without opening a held H5AD. The prediction must be committed publicly, fetched
byte-for-byte from its immutable commit URL, and bound by a public score
authorization before held access.

Held scoring uses the same fixed 512-cell rule. It first performs the HTO-only
census required to identify the selected cells. It then obtains RNA and ADT
margins in separated passes, materializes and hashes every prediction, and only
afterward forms the paired truth tables. Cell vectors are never serialized.
The held decision requires the primary to beat the strongest classical
residual and destroyed-link control by at least 5%, with bootstrap upper
endpoint below zero, at least 8 of 10 favorable donors, and an exact exhaustive
one-sided sign-flip `p <= 0.025` for each comparison. The first terminal held
result or refusal is final. The held donors do not enter model selection; only
this held-site decision is confirmatory.

## Reporting

The public table retains every development failure, held failure, and refusal.
A development pass is not confirmatory evidence. A held pass supports only
cross-site transfer of conditional RNA-protein dependence for this fixed MLN
panel and state definition. Other tissues, modalities, panels, and state rules
are outside this experiment.
