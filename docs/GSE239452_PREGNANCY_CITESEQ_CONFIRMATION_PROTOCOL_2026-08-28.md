# GSE239452 pregnancy CITE-seq confirmation protocol

**Frozen question.** Can a donor-heterogeneity-aware exact conditional coupling field learned from nonpregnant PBMCs predict linked RNA-protein states in an untouched pregnant cohort, after conditioning on each recipient's RNA and protein margins?

**Study.** Oh et al., Cell Reports 2024, DOI `10.1016/j.celrep.2024.114933`, PMID `39504241`; GEO SuperSeries `GSE239452`, ADT subseries `GSE239449`, and GEX subseries `GSE239450`.

**Protocol date.** 2026-08-28.

## Prospective boundary

This protocol, its runner, tests, source manifest, designation, and metadata-only preflight must be committed to the public benchmark before any deposited `X`, `raw/X`, or layer value is read. The preflight opened HDF5 structure, text axes, matrix shapes, and sparse-dataset shapes only. It read zero normalized or raw matrix payload values. Its immutable result is `results/development/gse239452_citeseq_metadata_preflight.json`, currently SHA-256 `6283bbe2a9a36f7c8548d78253a1c6101c88986ad77f7cf0eeb0cd6a40db7f60`.

Held ADT barcode axes are metadata and were used to establish the exact GEX-ADT common-cell universe. Opaque SHA-256 hashing reads file bytes but never decodes a matrix value. Held ADT `raw/X/data`, `indices`, and `indptr` values remain undecoded through prediction. The held prediction phase may read held GEX raw counts only after the pilot result passes and is published. Held ADT numerical values and RNA-ADT pairings may be formed only after the held predictions and score authorization are published.

Any exception after a terminal attempt record writes `results/development/gse239452_citeseq_terminal_refusal.json` and permanently closes the candidate. Existing attempt, prediction, score, or refusal artifacts prohibit a rerun.

## Sources and metadata

The official umbrella archives are:

| Modality | Accession | Bytes | SHA-256 |
|---|---:|---:|---|
| ADT | `GSE239449_RAW.tar` | 139,806,720 | `ad222e3988c9d9668159b027c61db1d12a2ce34f10a3e619f5849a5205e1252f` |
| GEX | `GSE239450_RAW.tar` | 1,289,472,000 | `295cc705cb505c3108e6d53c4baa899fb02ea804d5211f9597c9df9ecf7c9b60` |
| TotalSeq-C panel | `GSE239449_TotalSeqC_Annotated.xlsx` | 22,895 | `a55d147acc02842e496923cb4fb504fdbd151b9a9ba7a90741aeb88d718e1733` |

The source manifest enumerates every per-donor accession, filename, byte count, and exact role. The preflight records SHA-256 digests of every downloaded per-donor archive and extracted H5AD without reading matrix payloads.

The nine frozen markers are CD4, CD7, CD14, CD19, CD33, CD38, CD44, CD47, and CD52. RNA features match exact gene symbols on `raw/var/featurekey`. ADT features match exact `Featurekey` values `ADT_C0072`, `ADT_C0066`, `ADT_C0081`, `ADT_C0050`, `ADT_C0052`, `ADT_C0389`, `ADT_C0073`, `ADT_C0026`, and `ADT_C0033`, with exact `NameInData` and official panel-ID checks. No fuzzy alias is permitted.

The deposit uses two documented barcode encodings. Nonpregnant GEX barcodes have a terminal concatenation suffix of the form `-<integer>-0` or `-<integer>-1` after the 10x `-1`; pregnant GEX barcodes end in `-Pregnant`. Removing exactly that terminal deposit suffix maps them to the ADT index. Any other encoding refuses.

## Donor split

The original salted nonpregnant split is retained without replacement. Metadata preflight found that donor 100 has 378 ADT rows and 378 canonical paired barcodes, below the fixed 512-cell budget. Donor 78 has no ADT accession. Both exclusions are recorded before numerical access. Donor 100 is dropped from its original calibration allocation; no donor is rehashed or substituted.

| Role | Donors |
|---|---|
| Calibration, 7 | 47, 31, 223, 77, 191, 321, 213 |
| Adaptive pilot, 8 | 94, 103, 350, 182, 1, 325, 382, 50 |
| Untouched pregnant confirmation, 9 | OB7-CTRL, 705385-SEV, 803763-ASX, 324058-ASX, 644394-CTRL, 915348-SEV, 729106-CTRL, 105199-ASX, 101607-SEV |
| Prospective metadata exclusions | 100, 78 |

The held cohort contains three control, three asymptomatic, and three severe pregnant donors. Every eligible donor has at least 512 exact common barcodes. The smallest development pool is donor 191 with 1,196; the smallest held pool is donor 101607-SEV with 987.

## Cell and state construction

For each donor, rank the exact common barcode set by SHA-256 of `GSE239452-COMMON-CELL-v1|donor|barcode`, breaking a hash tie by the barcode string, and take the first 512. Counts never enter selection.

For RNA marker (g), state 1 means raw UMI count greater than zero. For ADT marker (p), rank the 512 raw counts, breaking ties by SHA-256 of `GSE239452-ADT-MEDIAN-v1|donor|marker|barcode`; the lower 256 cells receive state 0 and the upper 256 state 1. Every recipient ADT margin is therefore fixed at 256/256 before ADT numerical access.

Each donor yields 81 ordered RNA-marker by ADT-marker 2-by-2 tables. A donor-entity table is informative when its fixed-margin upper-left support contains at least two values. Donor loss is the equal-entity mean multinomial deviance per cell over informative entities. Fewer than 64 informative entities refuses the donor.

## Primary estimator

For donor $d$ and ordered marker pair $e$, let $\theta_{de}$ be the full conditional log odds ratio. The primary estimator minimizes the exact fixed-margin noncentral-hypergeometric negative log likelihood plus a quadratic donor-to-population penalty and population ridge/product-graph penalties. This is the implemented `fit_hierarchical_conditional_log_odds` estimator; it is not a Haldane-coordinate transfer.

The product graph is built inside each calibration fold from source marginal profiles only. RNA profiles are marker-positive fractions. ADT profiles are donor means of `log1p(100 * marker_count / nine_marker_total)`. Within each modality, the union of directed k-nearest-neighbor choices forms the incidence matrix; the two incidences induce the product hypergraph. Graph penalty zero is a valid primary candidate and uses no graph information.

The calibration grid is:

- graph neighbors: 1, 2;
- donor heterogeneity penalty: 0.1, 1, 10;
- population ridge penalty: 0.01, 0.1;
- graph penalty: 0, 0.1, 0.3, 1;
- transport multiplier alpha: 0.5, 0.75, 1, 1.25.

Graph-neighbor values are equivalent when the graph penalty is zero, so only neighbor value 1 is evaluated for those candidates. Seven-fold leave-one-calibration-donor-out prediction selects the finite candidate with the lowest donor-equal mean loss; exact ties use the serialized configuration order.

For recipient RNA margin $r$, fixed ADT margin $c=(256,256)$, and frozen finite population log odds $\mu$, prediction is the exact noncentral-hypergeometric expected table at log odds $\alpha\mu$ and margins $r,c$. Finite $\mu$ always defines a feasible expected table. The primary does not invert a bounded Haldane statistic and never clips a transferred coordinate.

## Comparators and diagnostic

The matched classical comparator pools the donor-equal, sample-size-normalized signed coordinate from either Pearson residuals or Poisson-deviance residuals under the row-plus-column independence model. Calibration cross-validation jointly selects residual family and alpha from 0.5, 0.75, 1, and 1.25. The selected coordinate is restored at recipient sample size and inverted at the exact recipient margins. This is the strongest frozen residual comparator, not a version fixed at alpha=1.

The destroyed-link control applies one donor-specific SHA-256 row permutation to the complete ADT state matrix, preserving every RNA and ADT margin and within-ADT dependence while breaking RNA-ADT cell pairing. It fits the same exact estimator and uses the primary configuration selected on the intact calibration data.

A graph-zero candidate is selected within the same calibration CV and reported on pilot and held donors. It is diagnostic only. Neither a positive graph penalty nor superiority to graph zero is required for promotion.

## Adaptive pilot

After calibration-only selection, fit the intact primary, destroyed-link control, strongest residual, and graph-zero diagnostic on all seven calibration donors. Score their frozen predictions on the eight pilot donors. Against each promotion comparator, `best_residual` and `destroyed_link`, promotion requires all of:

1. at least 5% reduction in donor-equal mean deviance;
2. the upper endpoint of a 20,000-draw paired donor-bootstrap 95% interval for primary-minus-comparator mean loss is below zero;
3. lower primary loss in at least 7 of 8 donors.

Bootstrap draws use `default_rng(20260828)`, resample physical donors, share the same index matrix across comparisons, and use linear 2.5% and 97.5% quantiles. Failure closes held numerical access. On a pass, refit the frozen configurations once on all 15 development donors; no held information enters selection or refitting.

## Held prediction and score

After the pilot result and held-GEX authorization are public, read the held GEX raw counts for the previously selected 512 common cells. Form nine RNA margins and combine them with the predetermined 256/256 ADT margins. Freeze all primary, residual, destroyed-link, and graph-zero expected tables in `results/gse239452_citeseq_predictions.json`. The prediction artifact records zero held ADT numeric values read, zero held pairings, and zero held truth tables.

After that prediction and its score authorization are public, read held ADT raw counts once, construct the deterministic states and 81 truth tables, and score the nine donors. Against each promotion comparator, the held result requires the same 5% and paired-bootstrap criteria plus lower primary loss in at least 8 of 9 donors. The graph-zero result remains diagnostic. The score report serializes donor losses and table digests, not held truth tables.

## Public authorization chain

Development authorization binds the exact public bytes of the runner, preflight, both focused tests, this protocol, the designation, source manifest, metadata preflight, and three numerical library modules at an immutable protocol commit. The authorization file is itself verified at a later immutable commit before the development attempt record is written.

Held-GEX authorization binds the exact public pilot-result bytes. Score authorization binds the exact public held-prediction bytes. Each authorization file is independently verified at its own immutable public commit. Local and public bytes must match before the corresponding attempt record is written.
